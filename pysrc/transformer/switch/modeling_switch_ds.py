from typing import Callable, List
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
import torch.nn as nn
import torch

import copy
import collections


from .modeling_switch import (
    T5DenseActDense,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    LayerNorm,
    invert_attention_mask,
    get_extended_attention_mask,
)
from .configuration_switch import SwitchConfig

import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.runtime.pipe import LayerSpec, PipelineModule


class ModuleLayerSpec(LayerSpec):
    def __init__(self, module, *args, **kwargs):
        super().__init__(type(module), *args, **kwargs)
        self.module = module

    def build(self, log=False):
        cls_ins = super().build(log)
        if self.module is not None:
            # print(type(self.module), type(cls_ins))
            # print(self.module.state_dict().keys(), cls_ins.state_dict().keys())
            # print(self.module_args, self.module_kwargs)
            cls_ins.load_state_dict(self.module.state_dict())
        return cls_ins


class EncoderEmbedPipe(nn.Module):
    def __init__(self, config: SwitchConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):

        batch_size, seq_length = encoder_input_ids.shape

        encoder_hidden_states = self.embedding(encoder_input_ids)
        encoder_hidden_states = self.dropout(encoder_hidden_states)

        input_shape = encoder_input_ids.size()
        extended_encoder_attention_mask = get_extended_attention_mask(
            encoder_attention_mask, input_shape, encoder_input_ids.device, False
        )

        encoder_position_bias = torch.empty((batch_size, self.num_heads, seq_length, 1))

        return (
            encoder_hidden_states,
            extended_encoder_attention_mask,
            encoder_position_bias,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )


class SwitchEncoderBlockPipe(nn.Module):
    def __init__(self, config: SwitchConfig, has_relative_attention_bias):
        super().__init__()
        self.has_relative_attention_bias = (
            has_relative_attention_bias  # the first block has relative attention bias
        )
        self.attention = T5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        extended_encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask,
    ):
        encoder_hidden_states, encoder_position_bias = self.attention(
            encoder_hidden_states,
            extended_encoder_attention_mask,
            encoder_position_bias,
        )

        return (
            encoder_hidden_states,
            encoder_position_bias,
            extended_encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )


class SwitchEncoderMoEPipe(MoE):
    def __init__(self, config: SwitchConfig):
        self.embed_dim = config.d_model
        self.num_experts = config.num_experts
        self.eval_capacity_factor = config.eval_capacity_factor
        expert = T5DenseActDense(config)
        super().__init__(
            hidden_size=self.embed_dim,
            expert=expert,
            capacity_factor=self.eval_capacity_factor,
        )

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        extended_encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask,
    ):
        encoder_hidden_states = super().forward(encoder_hidden_states)
        return (
            encoder_hidden_states,
            encoder_position_bias,
            extended_encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )


class SwitchEncoderLayerFFPipe(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        extended_encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask,
    ):
        forwarded_states = self.layer_norm(encoder_hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        encoder_hidden_states = encoder_hidden_states + self.dropout(forwarded_states)
        return (
            encoder_hidden_states,
            encoder_position_bias,
            extended_encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )


class SwitchEncoderFinalLayerNormPipe(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        extended_encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask,
    ):
        encoder_hidden_states = self.layer_norm(encoder_hidden_states)
        encoder_hidden_states = self.dropout(encoder_hidden_states)
        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )


class DecoderTokenEmbeddings(nn.Module):
    def __init__(self, config: SwitchConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask,
    ):
        input_shape = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

        decoder_extended_attention_mask = get_extended_attention_mask(
            decoder_attention_mask, input_shape, decoder_input_ids.device, True
        )

        decoder_hidden_states = self.embedding(decoder_input_ids)
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        batch_size, seq_length = decoder_input_ids.shape
        decoder_position_bias = torch.empty((batch_size, self.num_heads, seq_length, 1))

        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )


class SwitchDecoderBlockPipe(nn.Module):
    def __init__(self, config: SwitchConfig, has_relative_attention_bias):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.attention = T5LayerSelfAttention(config, has_relative_attention_bias)
        self.cross_attention = T5LayerCrossAttention(config)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias,
    ):
        decoder_hidden_states, decoder_position_bias = self.attention(
            encoder_hidden_states,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )

        decoder_hidden_states, encoder_position_bias = self.cross_attention(
            decoder_hidden_states,
            encoder_hidden_states,
            encoder_extended_attention_mask,
            encoder_position_bias,
        )

        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )


class SwitchDecoderMoEPipe(MoE):
    def __init__(self, config: SwitchConfig):
        self.embed_dim = config.d_model
        self.num_experts = config.num_experts
        self.eval_capacity_factor = config.eval_capacity_factor
        expert = T5DenseActDense(config)
        super().__init__(
            hidden_size=self.embed_dim,
            expert=expert,
            capacity_factor=self.eval_capacity_factor,
        )

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias,
    ):
        decoder_hidden_states = super().forward(decoder_hidden_states)
        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )


class SwitchDecoderLayerFFPipe(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias,
    ):
        forwarded_states = self.layer_norm(decoder_hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        decoder_hidden_states = decoder_hidden_states + self.dropout(forwarded_states)
        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )


class SwitchDecoderFinalLayerNormPipe(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias,
    ):
        decoder_hidden_states = self.layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.dropout(decoder_hidden_states)
        return decoder_hidden_states

class SwitchModelDeepSpeed(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()

        self.n_heads = config.num_heads
        self.n_experts = config.num_experts
        self.d_model = config.d_model
        self.n_layers = config.num_layers

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # encoder layers
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder_layers.append(EncoderEmbedPipe(encoder_config))
        for i in range(encoder_config.num_layers):
            self.encoder_layers.append(
                SwitchEncoderBlockPipe(encoder_config, bool(i == 0))
            )
            # moe is attached every two layers
            if i % 2 == 1:
                self.encoder_layers.append(SwitchEncoderMoEPipe(encoder_config))
            else:
                self.encoder_layers.append(SwitchEncoderLayerFFPipe(encoder_config))
        self.encoder_layers.append(SwitchEncoderFinalLayerNormPipe(encoder_config))

        # decoder layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder_layers.append(DecoderTokenEmbeddings(decoder_config))
        for i in range(decoder_config.num_layers):
            self.decoder_layers.append(SwitchDecoderBlockPipe(decoder_config, bool(i == 0)))
            # moe is attached every two layers
            if i % 2 == 1:
                self.decoder_layers.append(SwitchDecoderMoEPipe(decoder_config))
            else:
                self.decoder_layers.append(
                    SwitchDecoderLayerFFPipe(decoder_config)
                )
        self.decoder_layers.append(
            SwitchDecoderFinalLayerNormPipe(decoder_config)
        )
        # self.decoder = T5Stack(decoder_config, self.token_embedder)

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):

        # encoder embedding
        (
            encoder_hidden_states,
            encoder_extended_attention_mask,
            encoder_position_bias,
        ) = self.encoder_layers[0](encoder_input_ids, encoder_attention_mask)

        # encoder layers
        k = 1
        # batch_size, seq_len = encoder_input_ids.shape
        # encoder_position_bias = torch.empty((batch_size, self.n_heads, seq_len, 1))
        while k < len(self.encoder_layers):
            if isinstance(self.encoder_layers[k], SwitchEncoderBlockPipe):
                encoder_hidden_states, encoder_position_bias = self.encoder_layers[k](
                    encoder_hidden_states,
                    encoder_position_bias,
                    encoder_extended_attention_mask,
                )
            elif isinstance(self.encoder_layers[k], SwitchEncoderFinalLayerNormPipe):
                encoder_hidden_states = self.encoder_layers[k](encoder_hidden_states)
            elif isinstance(self.encoder_layers[k], SwitchEncoderLayerFFPipe):
                encoder_hidden_states = self.encoder_layers[k](encoder_hidden_states)
            elif isinstance(self.encoder_layers[k], SwitchEncoderMoEPipe):
                encoder_hidden_states = self.encoder_layers[k](encoder_hidden_states)
            else:
                raise NotImplementedError

            k += 1

        # decoder embedding
        (
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        ) = self.decoder_layers[0](
            decoder_input_ids, decoder_attention_mask, encoder_attention_mask
        )
        encoder_decoder_position_bias = encoder_position_bias

        # decoder layers
        k = 1
        # batch_size, seq_len = decoder_input_ids.shape
        # decoder_position_bias = torch.empty((batch_size, self.n_heads, seq_len, 1))
        while k < len(self.decoder_layers):
            if isinstance(self.decoder_layers[k], SwitchDecoderBlockPipe):
                (
                    decoder_hidden_states,
                    decoder_position_bias,
                    encoder_decoder_position_bias,
                ) = self.decoder_layers[k](
                    decoder_hidden_states,
                    encoder_hidden_states,
                    decoder_extended_attention_mask,
                    encoder_extended_attention_mask,
                    decoder_position_bias,
                    encoder_decoder_position_bias,
                )
            elif isinstance(self.decoder_layers[k], SwitchDecoderFinalLayerNormPipe):
                decoder_hidden_states = self.decoder_layers[k](decoder_hidden_states)
            elif isinstance(self.decoder_layers[k], SwitchDecoderLayerFFPipe):
                decoder_hidden_states = self.decoder_layers[k](decoder_hidden_states)
            elif isinstance(self.encoder_layers[k], SwitchDecoderMoEPipe):
                decoder_hidden_states = self.encoder_layers[k](decoder_hidden_states)
            else:
                raise NotImplementedError

            k += 1

        return decoder_hidden_states


class SwitchModelDeepSpeedPipe(PipelineModule):
    def __init__(self, config: SwitchConfig, module: SwitchModelDeepSpeed, **kwargs):
        layer_specs = []

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        k = 0
        for i, layer in enumerate(module.encoder_layers):
            if isinstance(layer, SwitchEncoderBlockPipe):
                layer_specs.append(
                    ModuleLayerSpec(layer, encoder_config, bool(k==0))
                )
            else:
                layer_specs.append(
                    ModuleLayerSpec(layer, encoder_config)
                )
            if isinstance(layer, SwitchEncoderMoEPipe) or isinstance(layer, SwitchEncoderLayerFFPipe):
                k += 1

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
    
        k = 0
        for i, layer in enumerate(module.decoder_layers):
            if isinstance(layer, SwitchDecoderBlockPipe):
                layer_specs.append(
                    ModuleLayerSpec(layer, encoder_config, bool(k==0))
                )
            else:
                layer_specs.append(
                ModuleLayerSpec(layer, decoder_config)
            )
            if isinstance(layer, SwitchDecoderMoEPipe) or isinstance(layer, SwitchDecoderLayerFFPipe):
                k += 1

        super().__init__(layers=layer_specs, **kwargs)

    