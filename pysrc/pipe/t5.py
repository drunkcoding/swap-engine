import copy
from functools import partial
import gc
from typing import Tuple
import numpy as np
from torch import nn
import torch

from deepspeed.pipe import PipelineModule, LayerSpec
from transformers import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

from hfutils.pipe.base import PipeMethods, format_inputs, format_outputs, get_embed_dim, get_extended_attention_mask, get_num_layers, init_all, invert_attention_mask, prepare_decoder_input_ids_for_generation

class T5EmbeddingPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = get_embed_dim(config)
        self.embed = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.is_decoder = config.is_decoder

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def _decoder_forward(self, args):
        (
            encoder_attention_mask,
            extended_attention_mask,
            encoder_hidden_states,
        ) = format_inputs(args, self.deepspeed_enabled)

        decoder_input_ids = prepare_decoder_input_ids_for_generation(encoder_attention_mask,
                                                                        self.config.decoder_start_token_id,
                                                                        self.config.eos_token_id)
        decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape, dtype=torch.long)
        input_shape = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        encoder_extended_attention_mask = invert_attention_mask(
            encoder_attention_mask
        )

        extended_attention_mask = get_extended_attention_mask(
            decoder_attention_mask, input_shape, decoder_input_ids.device, True
        )

        decoder_hidden_states = self.embed(decoder_input_ids)
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        return (
            encoder_extended_attention_mask,
            encoder_hidden_states,
            extended_attention_mask,
            decoder_hidden_states
        )

    def _encoder_forward(self, args):
        encoder_input_ids, encoder_attention_mask = format_inputs(args, self.deepspeed_enabled)
        encoder_hidden_states = self.embed(encoder_input_ids)
        encoder_hidden_states = self.dropout(encoder_hidden_states)

        input_shape = encoder_input_ids.size()
        extended_attention_mask = get_extended_attention_mask(
            encoder_attention_mask, input_shape, encoder_input_ids.device, False
        )

        return (
            encoder_attention_mask,
            extended_attention_mask,
            encoder_hidden_states
        )

    def forward(self, args):
        # print(os.getpid(), "T5EmbeddingPipe", self.is_decoder)
        # if len(args) == 2:
        # args += tuple([torch.Tensor([127873]).to(args[0].device)] * 6) if self.deepspeed_enabled else tuple([None] * 6)
        
        if self.is_decoder:
            return self._decoder_forward(args)
        return self._encoder_forward(args)


        # (
        #     encoder_input_ids,
        #     encoder_attention_mask,
        #     encoder_hidden_states,
        #     decoder_input_ids,
        #     decoder_attention_mask,
        #     decoder_hidden_states,
        #     position_bias,
        #     encoder_decoder_position_bias
        # ) = format_inputs(args, self.deepspeed_enabled)

        # if self.is_decoder:
        #     decoder_input_ids = prepare_decoder_input_ids_for_generation(encoder_input_ids,
        #                                                                  self.config.decoder_start_token_id,
        #                                                                  self.config.eos_token_id)
        #     decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape, dtype=torch.long)
        #     input_shape = decoder_input_ids.size()
        #     decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        # if self.is_decoder:
        #     decoder_hidden_states = self.embed(decoder_input_ids)
        #     decoder_hidden_states = self.dropout(decoder_hidden_states)
        # else:
        #     encoder_hidden_states = self.embed(encoder_input_ids)
        #     encoder_hidden_states = self.dropout(encoder_hidden_states)

        # return format_outputs(
        #     (
        #         encoder_input_ids,
        #         encoder_attention_mask,
        #         encoder_hidden_states,
        #         decoder_input_ids,
        #         decoder_attention_mask,
        #         decoder_hidden_states,
        #         None,   # position_bias
        #         encoder_decoder_position_bias
        #     ), self.deepspeed_enabled
        # )


class T5BlockPipe(nn.Module):
    def __init__(self, config: T5Config, i: int, ds=False) -> None:
        super().__init__()

        self.block = T5Block(config, has_relative_attention_bias=bool(i == 0))
        self.is_decoder = config.is_decoder
        self.block_idx = i

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def _decoder_forward(self, args):
        if self.block_idx == 0:
            (
                encoder_extended_attention_mask,
                encoder_hidden_states,
                extended_attention_mask,
                decoder_hidden_states
            ) = args
            position_bias = None
            encoder_decoder_position_bias = None
        else:
            (
                encoder_extended_attention_mask,
                encoder_hidden_states,
                extended_attention_mask,
                position_bias,
                encoder_decoder_position_bias,
                decoder_hidden_states,
            ) = args
        layer_outputs = self.block(
            decoder_hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
        decoder_hidden_states = layer_outputs[0]
        position_bias = layer_outputs[2]
        encoder_decoder_position_bias = layer_outputs[3]

        return (
            encoder_extended_attention_mask,
            encoder_hidden_states,
            extended_attention_mask,
            position_bias,
            encoder_decoder_position_bias,
            decoder_hidden_states,
        )

    def _encoder_forward(self, args):
        if self.block_idx == 0:
            (
                encoder_attention_mask,
                extended_attention_mask,
                encoder_hidden_states
            ) = args
            position_bias = None
        else:
            (
                encoder_attention_mask,
                extended_attention_mask,
                position_bias,
                encoder_hidden_states,
            ) = args
        layer_outputs = self.block(
            encoder_hidden_states,
            position_bias=position_bias,
            attention_mask=extended_attention_mask,
        )
        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
        encoder_hidden_states = layer_outputs[0]
        position_bias = layer_outputs[2]

        return (
            encoder_attention_mask,
            extended_attention_mask,
            position_bias,
            encoder_hidden_states,
        )

    def forward(self, args):
        # print(os.getpid(), "T5BlockPipe", self.block_idx, self.is_decoder)
        
        if self.is_decoder:
            return self._decoder_forward(args)
        return self._encoder_forward(args)
        
        # (
        #     encoder_input_ids,
        #     encoder_attention_mask,
        #     encoder_hidden_states,
        #     decoder_input_ids,
        #     decoder_attention_mask,
        #     decoder_hidden_states,
        #     position_bias,
        #     encoder_decoder_position_bias
        # ) = format_inputs(args, self.deepspeed_enabled)

        # if self.is_decoder:
        #     # print("block decoder")
        #     input_shape = decoder_input_ids.size()
        #     extended_attention_mask = get_extended_attention_mask(
        #         decoder_attention_mask, input_shape, decoder_input_ids.device, True
        #     )
        #     encoder_extended_attention_mask = invert_attention_mask(
        #         encoder_attention_mask
        #     )
        #     # print(decoder_hidden_states, encoder_hidden_states)
        #     layer_outputs = self.block(
        #         decoder_hidden_states,
        #         attention_mask=extended_attention_mask,
        #         position_bias=position_bias,
        #         encoder_decoder_position_bias=encoder_decoder_position_bias,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_extended_attention_mask,
        #     )
        #     layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
        #     decoder_hidden_states = layer_outputs[0]
        #     position_bias = layer_outputs[2]
        #     encoder_decoder_position_bias = layer_outputs[3]
        # if not self.is_decoder:
        #     # print("block encoder")
        #     input_shape = encoder_input_ids.size()
        #     extended_attention_mask = get_extended_attention_mask(
        #         encoder_attention_mask, input_shape, encoder_input_ids.device, False
        #     )
        #     layer_outputs = self.block(
        #         encoder_hidden_states,
        #         position_bias=position_bias,
        #         attention_mask=extended_attention_mask,
        #     )
        #     layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
        #     encoder_hidden_states = layer_outputs[0]
        #     position_bias = layer_outputs[2]

        # return format_outputs(
        #     (
        #         encoder_input_ids,
        #         encoder_attention_mask,
        #         encoder_hidden_states,
        #         decoder_input_ids,
        #         decoder_attention_mask,
        #         decoder_hidden_states,
        #         position_bias,
        #         encoder_decoder_position_bias
        #     ), self.deepspeed_enabled
        # )


class T5LMHeadPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.embed_dim = get_embed_dim(config)
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        self.is_decoder = config.is_decoder

        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def forward(self, args):
        # print(os.getpid(), "T5LMHeadPipe", self.is_decoder)
        (
            encoder_extended_attention_mask,
            encoder_hidden_states,
            extended_attention_mask,
            decoder_hidden_states,
        ) = args

        return self.lm_head(decoder_hidden_states)


class T5StackFFPipe(nn.Module):
    def __init__(self, config: T5Config, ds=False) -> None:
        super().__init__()

        self.embed_dim = get_embed_dim(config)

        self.final_layer_norm = T5LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.is_decoder = config.is_decoder
        
        self.deepspeed_enabled = ds

        init_all(self, torch.nn.init.normal_, mean=0., std=1) 

    def _decoder_forward(self, args):
        (
            encoder_extended_attention_mask,
            encoder_hidden_states,
            extended_attention_mask,
            position_bias,
            encoder_decoder_position_bias,
            decoder_hidden_states,
        ) = args

        decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        return (
            encoder_extended_attention_mask,
            encoder_hidden_states,
            extended_attention_mask,
            decoder_hidden_states,
        )

    def _encoder_forward(self, args):
        (
            encoder_attention_mask,
            extended_attention_mask,
            position_bias,
            encoder_hidden_states
        ) = args
        
        encoder_hidden_states = self.final_layer_norm(encoder_hidden_states)
        encoder_hidden_states = self.dropout(encoder_hidden_states)

        return (
            encoder_attention_mask,
            extended_attention_mask,
            encoder_hidden_states,
        )

    def forward(self, args):
        if self.is_decoder:
            return self._decoder_forward(args)
        return self._encoder_forward(args)
        # print(os.getpid(), "T5StackFFPipe", self.is_decoder)
        # (
        #     encoder_input_ids,
        #     encoder_attention_mask,
        #     encoder_hidden_states,
        #     decoder_input_ids,
        #     decoder_attention_mask,
        #     decoder_hidden_states,
        #     position_bias,
        #     encoder_decoder_position_bias
        # ) = format_inputs(args)  if self.deepspeed_enabled else args

        # if self.is_decoder:
        #     decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
        #     decoder_hidden_states = self.dropout(decoder_hidden_states)
        # if not self.is_decoder:
        #     encoder_hidden_states = self.final_layer_norm(encoder_hidden_states)
        #     encoder_hidden_states = self.dropout(encoder_hidden_states)

        # return format_outputs(
        #     (
        #         encoder_input_ids,
        #         encoder_attention_mask,
        #         encoder_hidden_states,
        #         decoder_input_ids,
        #         decoder_attention_mask,
        #         decoder_hidden_states,
        #         position_bias,
        #         encoder_decoder_position_bias
        #     ), self.deepspeed_enabled
        # )


class T5PyTorchPipe(nn.Module, PipeMethods):
    def __init__(self, model: T5ForConditionalGeneration, exec_map: Tuple = None) -> None:
        super().__init__()
        
        config = model.config
        # self.total_params = sum([np.prod(p.size()) for p in model.parameters()])

        # self.embed_dim = get_embed_dim(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.n_layers = get_num_layers(config)

        self.layers = []

        encoder_embed = T5EmbeddingPipe(encoder_config)
        encoder_embed.embed.load_state_dict(model.shared.state_dict())
        self.layers.append(encoder_embed)
        for i in range(self.n_layers):
            encoder_block = T5BlockPipe(encoder_config, i)
            encoder_block.block.load_state_dict(model.encoder.block[i].state_dict())
            self.layers.append(encoder_block)
        encoder_stack_ff = T5StackFFPipe(encoder_config)
        encoder_stack_ff.final_layer_norm.load_state_dict(model.encoder.final_layer_norm.state_dict())
        encoder_stack_ff.dropout.load_state_dict(model.encoder.dropout.state_dict())
        self.layers.append(encoder_stack_ff)


        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        decoder_embed = T5EmbeddingPipe(decoder_config)
        decoder_embed.embed.load_state_dict(model.shared.state_dict())
        self.layers.append(decoder_embed)
        for i in range(self.n_layers):
            decoder_block = T5BlockPipe(decoder_config, i)
            decoder_block.block.load_state_dict(model.decoder.block[i].state_dict())
            self.layers.append(decoder_block)
        decoder_stack_ff = T5StackFFPipe(decoder_config)
        decoder_stack_ff.final_layer_norm.load_state_dict(model.decoder.final_layer_norm.state_dict())
        decoder_stack_ff.dropout.load_state_dict(model.decoder.dropout.state_dict())
        self.layers.append(decoder_stack_ff)

        lm_head = T5LMHeadPipe(decoder_config)
        lm_head.lm_head.load_state_dict(model.lm_head.state_dict())
        self.layers.append(lm_head)
        
        self.layer_param = [
            sum([np.prod(p.size()) for p in layer.parameters()])
            for layer in self.layers
        ]
        self.total_params = sum(self.layer_param)

        # super().__init__(layers=encoder_specs + decoder_specs, **kwargs)
        self.layers = nn.ModuleList(self.layers)
        self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))     

    @torch.no_grad()
    def forward(self, args, output_hidden_states=False):
        outputs = args
        all_hidden_states = ()
        for idx in range(*self.exec_map):
            outputs = self.layers[idx](outputs)
            if output_hidden_states:
                if not (isinstance(self.layers[idx], T5BlockPipe) and self.layers[idx].block_idx == self.n_layers-1):
                    if idx != len(self.layers) - 1:
                        all_hidden_states = all_hidden_states + (
                            outputs[-1],
                        )
        if output_hidden_states:
            return (
                outputs,
                all_hidden_states
            )
        return outputs # if isinstance(outputs, Tuple) else (outputs, )

class T5PytorchPipeRandom(nn.Module, PipeMethods):
    def __init__(self, config: T5Config, **kwargs) -> None:
        super().__init__()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(T5EmbeddingPipe, encoder_config),
            *[LayerSpec(T5BlockPipe, encoder_config, i) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, encoder_config),
        ]

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        decoder_specs = [
            LayerSpec(T5EmbeddingPipe, decoder_config),
            *[LayerSpec(T5BlockPipe, decoder_config, i) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, decoder_config),
            LayerSpec(T5LMHeadPipe, decoder_config),
        ]

        self.layer_specs = encoder_specs + decoder_specs
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
                if not (isinstance(self.layers[idx], T5BlockPipe) and self.layers[idx].block_idx == self.n_layers-1):
                    if idx != len(self.layers) - 1:
                        all_hidden_states = all_hidden_states + (
                            outputs[-1],
                        )
        if output_hidden_states:
            return (
                outputs,
                all_hidden_states
            )
        return outputs # if isinstance(outputs, Tuple) else (outputs, )

class T5DeepSpeedPipe(PipelineModule):
    def __init__(self, config: T5Config, **kwargs) -> None:
        # self.embed_dim = get_embed_dim(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.n_layers = get_num_layers(config)

        encoder_specs = [
            LayerSpec(T5EmbeddingPipe, encoder_config, True),
            *[LayerSpec(T5BlockPipe, encoder_config, i, True) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, encoder_config, True),
        ]

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        decoder_specs = [
            LayerSpec(T5EmbeddingPipe, decoder_config, True),
            *[LayerSpec(T5BlockPipe, decoder_config, i, True) for i in range(self.n_layers)],
            LayerSpec(T5StackFFPipe, decoder_config, True),
            LayerSpec(T5LMHeadPipe, decoder_config, True),
        ]

        super().__init__(layers=encoder_specs + decoder_specs, **kwargs)


T5_ENCODER_INPUTS = {
    T5EmbeddingPipe.__name__: [
        'encoder_input_ids', 
        'encoder_attention_mask'
    ],
    T5BlockPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'encoder_hidden_states'
    ],
    T5StackFFPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'position_bias', 
        'encoder_hidden_states'
    ],
}

T5_ENCODER_OUTPUTS = {
    T5EmbeddingPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'encoder_hidden_states'
    ],
    T5BlockPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'encoder_hidden_states'
    ],
    T5StackFFPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'encoder_hidden_states'
    ],
}

T5_DECODER_INPUTS = {
    T5EmbeddingPipe.__name__: [
        'encoder_attention_mask', 
        'extended_attention_mask', 
        'encoder_hidden_states',
    ],
    T5BlockPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'decoder_hidden_states'
    ],
    T5StackFFPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'position_bias',
        'encoder_decoder_position_bias',
        'decoder_hidden_states',
    ],
    T5LMHeadPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'decoder_hidden_states',
    ]
}

T5_DECODER_OUTPUTS = {
    T5EmbeddingPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'decoder_hidden_states'
    ],
    T5BlockPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'position_bias',
        'encoder_decoder_position_bias',
        'decoder_hidden_states',
    ],
    T5StackFFPipe.__name__: [
        'encoder_extended_attention_mask',
        'encoder_hidden_states',
        'extended_attention_mask',
        'decoder_hidden_states',
    ],
    T5LMHeadPipe.__name__: [
        'logits',
    ]
}