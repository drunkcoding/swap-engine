import copy
from typing import Tuple
import torch
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import Embeddings, TransformerBlock
from transformers import DistilBertForQuestionAnswering

from torch import nn
import numpy as np
from deepspeed.pipe import PipelineModule, LayerSpec

from hfutils.pipe.base import PipeMethods, format_inputs, format_outputs, get_num_layers

class EmbeddingsPipe(Embeddings):
    def __init__(self, config: DistilBertConfig) -> None:
        super().__init__(config)
        # self.deepspeed_enabled = ds

    def forward(self, args):
        if len(args) == 2:
            args = args + (None, )
        input_ids, attention_mask, hidden_states = args # format_inputs(args, self.deepspeed_enabled)

        hidden_states = super().forward(input_ids)

        return attention_mask, hidden_states
        # return format_outputs(
        #     (attention_mask, hidden_states), self.deepspeed_enabled
        # )

class TransformerPipe(TransformerBlock):
    def __init__(self, config: DistilBertConfig):
        super().__init__(config)
        # self.deepspeed_enabled = ds

    def forward(self, args):
        attention_mask, hidden_states = args # format_inputs(args, self.deepspeed_enabled)

        hidden_states = super().forward(hidden_states, attention_mask)[-1]
        # hidden_states = layer_outputs

        return attention_mask, hidden_states
        # return format_outputs(
        #     (attention_mask, hidden_states), self.deepspeed_enabled
        # )

class DistilBertQuestionAnsweringHeadPipe(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        # self.deepspeed_enabled = ds

        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

    def forward(self, args):
        attention_mask, hidden_states = args # format_inputs(args, self.deepspeed_enabled)

        hidden_states = self.dropout(hidden_states)
        logits = self.qa_outputs(hidden_states)

        return logits
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        # end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        
        # return format_outputs(
        #     (start_logits, end_logits), self.deepspeed_enabled
        # )



class DistilBertPyTorchPipeForQuestionAnswering(nn.Module, PipeMethods):
    def __init__(self, model: DistilBertForQuestionAnswering, exec_map: Tuple = None) -> None:
        super().__init__()

        config = model.config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(encoder_config)

        self.layers = []
        encoder_embed = EmbeddingsPipe(encoder_config)
        encoder_embed.load_state_dict(model.distilbert.embeddings.state_dict())
        self.layers.append(encoder_embed)

        for i in range(self.n_layers):
            encoder_block = TransformerPipe(encoder_config)
            encoder_block.load_state_dict(model.distilbert.transformer.layer[i].state_dict())
            self.layers.append(encoder_block)

        qa_outputs = DistilBertQuestionAnsweringHeadPipe(encoder_config)
        qa_outputs.qa_outputs.load_state_dict(model.qa_outputs.state_dict())
        self.layers.append(qa_outputs)

        self.total_params = sum([
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ])

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
                    all_hidden_states = all_hidden_states + (outputs[1],)
        if output_hidden_states:
            return (
                outputs,
                all_hidden_states
            )
        return outputs # if isinstance(outputs, Tuple) else (outputs, )


class DistilBertDeepSpeedPipeForQuestionAnswering(PipelineModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(EmbeddingsPipe, encoder_config),
            *[
                LayerSpec(TransformerPipe, encoder_config)
                for _ in range(self.n_layers)
            ],
            # LayerSpec(BertPoolerPipe, encoder_config, True),
            LayerSpec(DistilBertQuestionAnsweringHeadPipe, encoder_config),
        ]

        super().__init__(layers=encoder_specs, **kwargs)

DISTILBERT_INPUTS = {
    EmbeddingsPipe.__name__: ["input_ids", "attention_mask"],
    TransformerPipe.__name__: ["attention_mask", "hidden_states"],
    DistilBertQuestionAnsweringHeadPipe.__name__: ["attention_mask", "hidden_states"],
}

DISTILBERT_OUTPUTS = {
    EmbeddingsPipe.__name__: ["attention_mask", "hidden_states"],
    TransformerPipe.__name__: ["attention_mask", "hidden_states"],
    DistilBertQuestionAnsweringHeadPipe.__name__: ["logits"],
}