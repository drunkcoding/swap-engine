import copy
import gc
import math
from turtle import forward
from typing import Tuple, Union
import numpy as np
import torch
from deepspeed.pipe import PipelineModule, LayerSpec

from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings,
    ViTPooler,
    ViTOutput,
    ViTIntermediate,
)
from transformers import ViTForImageClassification

from torch import nn

from pysrc.pipe.base import PipeMethods, get_num_layers


class ViTEmbeddingsPipe(ViTEmbeddings):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        # self.deepspeed_enabled = ds

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# No pooler needed
class ViTPoolerPipe(ViTPooler):
    def __init__(self, config: ViTConfig):
        super().__init__(config)

    def forward(self, hidden_states):
        # hidden_states = args[0]
        pooled_output = super().forward(hidden_states)
        return pooled_output


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor):
        self_outputs = self.attention(hidden_states)

        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class ViTAttentionPipe(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTAttention(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        normed_hidden_states = self.layernorm_before(hidden_states)
        attention_output = self.attention(normed_hidden_states)
        hidden_states = attention_output + hidden_states
        # hidden_states = self.layernorm_after(hidden_states)
        return hidden_states


class ViTAfterAttentionPipe(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

    def forward(self, hidden_states):
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


# class ViTIntermediatePipe(nn.Module):
#     def __init__(self, config: ViTConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.intermediate_act_fn = nn.GELU()

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class ViTOutputPipe(nn.Module):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         hidden_states = hidden_states + input_tensor

#         return hidden_states

# class ViTLayerPipe(ViTLayer):
#     def __init__(self, config: ViTConfig):
#         super().__init__(config)
#         # self.deepspeed_enabled = ds

#     def forward(self, hidden_states):
#         hidden_states = self.layernorm_before(hidden_states)
#         self_attention_outputs = self.attention(hidden_states)
#         attention_output = self_attention_outputs[0]

#         # first residual connection
#         hidden_states = attention_output + hidden_states

#         # in ViT, layernorm is also applied after self-attention
#         layer_output = self.layernorm_after(hidden_states)
#         layer_output = self.intermediate(layer_output)

#         # second residual connection is done here
#         layer_output = self.output(layer_output, hidden_states)

#         return layer_output


class ViTClassifierPipe(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

    def forward(self, hidden_states):
        # hidden_states = args[0]
        hidden_states = self.layernorm(hidden_states)
        logits = self.classifier(hidden_states[:, 0, :])
        return logits


class ViTModelPipe(nn.Module):
    def __init__(self, model: ViTForImageClassification):
        super().__init__()
        config = model.config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False

        self.n_layers = get_num_layers(encoder_config)
        self.layers = []

        # Add embedding layer to layers
        encoder_embed = ViTEmbeddingsPipe(encoder_config)
        encoder_embed.load_state_dict(model.vit.embeddings.state_dict())
        self.layers.append(encoder_embed)

        # for all layers add attention, intermediate, output layers
        for i in range(self.n_layers):
            attention_layer = ViTAttentionPipe(encoder_config)
            attention_layer.attention.load_state_dict(
                model.vit.encoder.layer[i].attention.state_dict()
            )
            attention_layer.layernorm_before.load_state_dict(
                model.vit.encoder.layer[i].layernorm_before.state_dict()
            )
            # attention_layer.layernorm_after.load_state_dict(model.vit.encoder.layer[i].layernorm_after.state_dict())
            self.layers.append(attention_layer)

            after_attention_layer = ViTAfterAttentionPipe(encoder_config)
            after_attention_layer.layernorm_after.load_state_dict(
                model.vit.encoder.layer[i].layernorm_after.state_dict()
            )
            after_attention_layer.intermediate.load_state_dict(
                model.vit.encoder.layer[i].intermediate.state_dict()
            )
            after_attention_layer.output.load_state_dict(
                model.vit.encoder.layer[i].output.state_dict()
            )
            self.layers.append(after_attention_layer)

            # intermediate_layer = ViTIntermediatePipe(encoder_config)
            # intermediate_layer.load_state_dict(model.vit.encoder.layer[i].intermediate.state_dict())
            # self.layers.append(intermediate_layer)

            # output_layer = ViTOutputPipe(encoder_config)
            # output_layer.load_state_dict(model.vit.encoder.layer[i].output.state_dict())
            # self.layers.append(output_layer)

        # add classifier layer to layers
        encoder_classifier = ViTClassifierPipe(encoder_config)
        encoder_classifier.classifier.load_state_dict(model.classifier.state_dict())
        encoder_classifier.layernorm.load_state_dict(model.vit.layernorm.state_dict())
        self.layers.append(encoder_classifier)

        self.layers = nn.ModuleList(self.layers)
        self.total_layers = len(self.layers)

    def forward(self, pixel_values):
        hidden_states = pixel_values
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        return hidden_states

    def forward_with_hidden_states(self, pixel_values):
        hidden_states = pixel_values
        all_hidden_states = ()
        for idx, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states)
            if (idx == 0 or idx % 2 == 0) and idx != self.total_layers - 1:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states, all_hidden_states)
        


# class ViTPyTorchPipeForImageClassification(nn.Module, PipeMethods):
#     def __init__(
#         self, model: ViTForImageClassification, exec_map: Tuple = None
#     ) -> None:
#         super().__init__()

#         config = model.config
#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False

#         self.n_layers = get_num_layers(encoder_config)

#         self.layers = []
#         encoder_embed = ViTEmbeddingsPipe(encoder_config)
#         encoder_embed.load_state_dict(model.vit.embeddings.state_dict())
#         self.layers.append(encoder_embed)

#         for i in range(self.n_layers):
#             encoder_block = ViTLayerPipe(encoder_config)
#             encoder_block.load_state_dict(model.vit.encoder.layer[i].state_dict())
#             self.layers.append(encoder_block)

#         classifier = ViTClassifierPipe(encoder_config)
#         classifier.classifier.load_state_dict(model.classifier.state_dict())
#         classifier.layernorm.load_state_dict(model.vit.layernorm.state_dict())
#         self.layers.append(classifier)

#         self.layer_param = [
#             sum([np.prod(p.size()) for p in layer.parameters()])
#             for layer in self.layers
#         ]
#         self.total_params = sum(self.layer_param)

#         self.layers = nn.ModuleList(self.layers)

#         self.exec_map = exec_map if exec_map is not None else (0, len(self.layers))

#     @torch.no_grad()
#     def forward(self, args, output_hidden_states=False):
#         outputs = args
#         all_hidden_states = ()
#         for idx in range(*self.exec_map):
#             outputs = self.layers[idx](outputs)
#             if output_hidden_states:
#                 if idx != len(self.layers) - 1:
#                     all_hidden_states = all_hidden_states + (outputs,)
#         if output_hidden_states:
#             return (outputs, all_hidden_states)
#         return outputs  # if isinstance(outputs, Tuple) else (outputs, )


# VIT_INPUTS = {
#     ViTEmbeddingsPipe.__name__: ["pixel_values"],
#     ViTLayerPipe.__name__: ["hidden_states"],
#     ViTClassifierPipe.__name__: ["hidden_states"],
# }

# VIT_OUTPUTS = {
#     ViTEmbeddingsPipe.__name__: ["hidden_states"],
#     ViTLayerPipe.__name__: ["hidden_states"],
#     ViTClassifierPipe.__name__: ["logits"],
# }
