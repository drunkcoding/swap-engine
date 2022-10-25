from typing import Tuple
import numpy as np
import torch
from torch import embedding, nn
from deepspeed.pipe import PipelineModule, LayerSpec


from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers import GPT2LMHeadModel

from hfutils.pipe.base import (
    PipeMethods,
    format_inputs,
    format_outputs,
    get_embed_dim,
    get_embed_dropout,
    get_num_layers,
)


class GPTEmbeddingPipe(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = get_embed_dim(config)

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if config.model_type != "gptj":
            # print("max_position_embeddings")
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(get_embed_dropout(config))

        # self.deepspeed_enabled = ds

    def forward(self, args):
        # (
        #     input_ids,
        #     attention_mask,
        # ) = format_inputs(args, self.deepspeed_enabled)
        input_ids, attention_mask = args
        position_ids = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # batch_size = input_ids.shape[0]

        device = input_ids.device  # if input_ids is not None else inputs_embeds.device
        past_length = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if hasattr(self, "wpe"):
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        # batch_size, sequence_length = input_ids.shape[:2]
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     attention_mask = attention_mask[:, None, None, :]
        #     attention_mask = (1.0 - attention_mask) * -10000.0

        return attention_mask, hidden_states
        # return format_outputs(
        #     (
        #         attention_mask,
        #         hidden_states
        #     ), self.deepspeed_enabled
        # )


class GPTBlockPipe(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        # print(config)

        if config.model_type == "gptj":
            self.block = GPTJBlock(config)
        elif config.model_type == "gpt2":
            self.block = GPT2Block(config, layer_idx)
        elif config.model_type == "gpt_neo":
            self.block = GPTNeoBlock(config, layer_idx)
        else:
            raise NotImplementedError()

        self.layer_idx = layer_idx
        # self.deepspeed_enabled = ds

        if layer_idx == get_num_layers(config) - 1:
            self.embed_dim = get_embed_dim(config)
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, args):
        # (
        #     attention_mask,
        #     hidden_states,
        # ) = format_inputs(args, self.deepspeed_enabled)
        attention_mask, hidden_states = args

        # outputs = self.block(hidden_states, attention_mask=attention_mask,)
        outputs = self.block(hidden_states, attention_mask=None)
        hidden_states = outputs[0]
        if hasattr(self, "ln_f"):
            hidden_states = self.ln_f(hidden_states)

        return attention_mask, hidden_states
        # return format_outputs(
        #     (
        #         attention_mask,
        #         hidden_states
        #     ), self.deepspeed_enabled
        # )


class GPTOutputPipe(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_labels = config.num_labels
        # self.score = nn.Linear(self.embed_dim, self.num_labels, bias=False)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.model_type == "gptj"
        )
        # self.deepspeed_enabled = ds
    
    def forward(self, args):
        # (
        #     attention_mask,
        #     hidden_states,
        # ) = format_inputs(args, self.deepspeed_enabled)
        attention_mask, hidden_states = args
        # hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

        # input_shape = input_ids.size()
        # output_shape = input_shape + (hidden_states.size(-1),)
        # hidden_states = hidden_states.view(*output_shape)
        # output = self.score(hidden_states)

        # batch_size, _ = input_ids.shape[:2]

        # sequence_lengths = torch.ne(input_ids, 50256).sum(-1) - 1

        # pooled_logits = output[range(batch_size), sequence_lengths]
        # return pooled_logits


class GPTLMHeadModelPipe(nn.Module, PipeMethods):
    def __init__(self, model: GPT2LMHeadModel, exec_map: Tuple = None) -> None:
        super().__init__()

        config = model.config

        self.n_layers = get_num_layers(config)

        self.layers = []
        embedding = GPTEmbeddingPipe(config)
        embedding.wte.load_state_dict(model.transformer.wte.state_dict())
        if hasattr(embedding, "wpe"):
            embedding.wpe.load_state_dict(model.transformer.wpe.state_dict())
        embedding.drop.load_state_dict(model.transformer.drop.state_dict())
        self.layers.append(embedding)

        for i in range(self.n_layers):
            block = GPTBlockPipe(config, i)
            block.block.load_state_dict(model.transformer.h[i].state_dict())
            if hasattr(block, "ln_f"):
                block.ln_f.load_state_dict(model.transformer.ln_f.state_dict())
            self.layers.append(block)

        head = GPTOutputPipe(config)
        head.lm_head.load_state_dict(model.lm_head.state_dict())
        self.layers.append(head)

        self.layer_param = [
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ]
        self.total_params = sum(self.layer_param)

        self.layers = nn.ModuleList(self.layers)

        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))

    @torch.no_grad()
    def forward(self, args, output_hidden_states=False):
        outputs = args
        all_hidden_states = ()
        for idx in range(*self.exec_map):
            outputs = self.layers[idx](outputs)
            if output_hidden_states:
                if idx != len(self.layers) - 1:
                    all_hidden_states = all_hidden_states + (outputs[2],)
        if output_hidden_states:
            return (outputs, all_hidden_states)
        return outputs  # if isinstance(outputs, Tuple) else (outputs, )


class GPTPytorchPipeRandom(nn.Module, PipeMethods):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.n_layers = get_num_layers(config)

        self.layer_specs = [
            LayerSpec(GPTEmbeddingPipe, config),
            *[LayerSpec(GPTBlockPipe, config, i) for i in range(self.n_layers)],
            # LayerSpec(BertPoolerPipe, encoder_config, True),
            LayerSpec(GPTOutputPipe, config),
        ]

        self.layers = [torch.nn.Module() for _ in self.layer_specs]
        self.layer_param = [1] * len(self.layer_specs)
        self.total_params = self.total_params = sum(self.layer_param)

    @torch.no_grad()
    def forward(self, args, output_hidden_states=False):
        outputs = args
        all_hidden_states = ()
        for idx in range(*self.exec_map):
            outputs = self.layers[idx](outputs)
            if output_hidden_states:
                if idx != len(self.layers) - 1:
                    all_hidden_states = all_hidden_states + (outputs[2],)
        if output_hidden_states:
            return (outputs, all_hidden_states)
        return outputs  # if isinstance(outputs, Tuple) else (outputs, )

class GPTDeepSpeedPipe(PipelineModule):
    def __init__(self, config, **kwargs) -> None:
        self.n_layers = get_num_layers(config)

        specs = [
            LayerSpec(GPTEmbeddingPipe, config),
            *[LayerSpec(GPTBlockPipe, config, i) for i in range(self.n_layers)],
            # LayerSpec(BertPoolerPipe, encoder_config, True),
            LayerSpec(GPTOutputPipe, config),
        ]

        super().__init__(layers=specs, **kwargs)


GPT_INPUTS = {
    GPTEmbeddingPipe.__name__: ["input_ids", "attention_mask"],
    GPTBlockPipe.__name__: ["attention_mask", "hidden_states"],
    GPTOutputPipe.__name__: ["attention_mask", "hidden_states"],
}

GPT_OUTPUTS = {
    GPTEmbeddingPipe.__name__: ["attention_mask", "hidden_states"],
    GPTBlockPipe.__name__: ["attention_mask", "hidden_states"],
    GPTOutputPipe.__name__: ["logits"],
}

