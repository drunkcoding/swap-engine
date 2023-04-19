from typing import Callable, List
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
import torch.nn as nn
import torch

import copy
from .modeling_switch_transformers import T5LayerCrossAttention, T5LayerSelfAttention, SwitchDenseGatedActDense, SwitchDenseActDense, get_extended_attention_mask, invert_attention_mask
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersLayerNorm,
    SwitchTransformersSparseMLP
)

import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.runtime.pipe import LayerSpec, PipelineModule

class ModuleLayerSpec(LayerSpec):
    def __init__(self, module_type, module, *args, **kwargs):
        super().__init__(module_type, *args, **kwargs)
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
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):

        input_shape = encoder_input_ids.size()
        input_ids = encoder_input_ids.view(-1, input_shape[-1])

        batch_size, seq_length = input_shape

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_encoder_attention_mask = get_extended_attention_mask(
            encoder_attention_mask, input_shape, is_decoder=False
        )
        encoder_position_bias = (
            self.compute_bias(seq_length, seq_length).to(input_ids.device)
            + extended_encoder_attention_mask
        )
        encoder_hidden_states = self.embed_tokens(input_ids)

        return (
            encoder_hidden_states,
            extended_encoder_attention_mask,
            encoder_position_bias,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )

class DecoderEmbedPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.dropout = nn.Dropout(config.dropout_rate)

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
            decoder_attention_mask, input_shape, True
        )

        decoder_hidden_states = self.embed_tokens(decoder_input_ids)

        batch_size, seq_length = decoder_input_ids.shape
        decoder_position_bias = (
            self.compute_bias(seq_length, seq_length).to(decoder_input_ids.device)
            + decoder_extended_attention_mask
        )

        encoder_decoder_position_bias = (
            torch.zeros(
                (1, self.n_heads, seq_length, encoder_attention_mask.shape[1]),
                device=decoder_hidden_states.device,
                dtype=decoder_hidden_states.dtype,
            )
            + encoder_extended_attention_mask
        )

        return (
            encoder_hidden_states,
            encoder_position_bias,
            decoder_hidden_states,
            encoder_extended_attention_mask,
            decoder_extended_attention_mask,
            decoder_position_bias,
        )


class EncoderFFPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_gated: bool = False):
        super().__init__()
        if is_gated:
            self.mlp = SwitchDenseGatedActDense(config)
        else:
            self.mlp = SwitchDenseActDense(config)
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def forward(self, encoder_hidden_states,
        encoder_position_bias,
        extended_encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_attention_mask):
        encoder_hidden_states = encoder_hidden_states + self.mlp(self.layer_norm(encoder_hidden_states))
        return (
            encoder_hidden_states,
            encoder_position_bias,
            extended_encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_attention_mask,
        )

class DecoderFFPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_gated: bool = False):
        super().__init__()
        if is_gated:
            self.mlp = SwitchDenseGatedActDense(config)
        else:
            self.mlp = SwitchDenseActDense(config)
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def forward(self, encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias):
        decoder_hidden_states = decoder_hidden_states + self.mlp(self.layer_norm(decoder_hidden_states))
        return (
            encoder_hidden_states,
        encoder_position_bias,
        decoder_hidden_states,
        encoder_extended_attention_mask,
        decoder_extended_attention_mask,
        decoder_position_bias
        )

class SwitchEncoderBlockPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, has_relative_attention_bias):
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
        encoder_hidden_states = self.attention(
            encoder_hidden_states,
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


class SwitchEncoderMoEPipe(SwitchTransformersSparseMLP):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)

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

class SwitchEncoderFinalLayerNormPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
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




class SwitchDecoderBlockPipe(nn.Module):
    def __init__(
        self, config: SwitchTransformersConfig, has_relative_attention_bias=False
    ):
        super().__init__()
        self.has_relative_attention_bias = (
            has_relative_attention_bias  # the first block has relative attention bias
        )
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
        decoder_hidden_states = self.attention(
            decoder_hidden_states,
            decoder_position_bias,
        )

        decoder_hidden_states = self.cross_attention(
            decoder_hidden_states,
            encoder_hidden_states,
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


class SwitchDecoderMoEPipe(SwitchTransformersSparseMLP):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)

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

class SwitchDecoderFinalLayerNormPipe(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
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

class SwitchModelDeepSpeedPipe(PipelineModule):
    def __init__(self, config: SwitchTransformersConfig, module: SwitchTransformersForConditionalGeneration, **kwargs):
        layer_specs = []

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        layer_specs.append(
            ModuleLayerSpec(EncoderEmbedPipe, module.encoder.embed_tokens, encoder_config)
        )

        for i in range(config.num_layers):
            layer_specs.append(
                ModuleLayerSpec(SwitchDecoderBlockPipe, module, encoder_config)
            )
            if isinstance(layer, SwitchEncoderBlockPipe):
                layer_specs.append(
                    ModuleLayerSpec(layer, encoder_config, bool(k==0))
                )
            else:
                layer_specs.append(
                    ModuleLayerSpec(layer, encoder_config)
                )
            if isinstance(layer, SwitchEncoderMoEPipe) or isinstance(layer, EncoderFFPipe):
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
            if isinstance(layer, SwitchDecoderMoEPipe) or isinstance(layer, DecoderFFPipe):
                k += 1

        super().__init__(layers=layer_specs, **kwargs)

    