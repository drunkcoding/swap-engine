import copy
import math
from typing import Callable, List, Tuple
import torch.nn as nn
from transformers import SwitchTransformersConfig
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersLayerNorm,
    SwitchTransformersDenseActDense,
    SwitchTransformersDenseGatedActDense,
    SwitchTransformersLayerCrossAttention,
    SwitchTransformersLayerSelfAttention,
)
import torch


def invert_attention_mask(encoder_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

    Returns:
        :obj:`torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})"
        )
    # encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
    return encoder_extended_attention_mask


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: List[int],
    is_decoder: bool = False,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    device = attention_mask.device
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = (
                seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                <= seq_ids[None, :, None]
            )
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    dim=-1,
                )

            extended_attention_mask = (
                causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
    return extended_attention_mask


class AttentionBias(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.n_heads = config.num_heads
        self.is_decoder = config.is_decoder

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, self.n_heads
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = torch.zeros_like(
            relative_position, device=relative_position.device, dtype=torch.long
        )
        if bidirectional:
            num_buckets = math.floor(num_buckets / 2)
            relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = math.floor(num_buckets / 2)
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int):
        """Compute binned relative position bias"""
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values


class EncoderTokenEmbeddings(AttentionBias):
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, encoder_input_ids, encoder_attention_mask):

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
            encoder_position_bias,
        )


class DecoderTokenEmbeddings(AttentionBias):
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, decoder_input_ids, decoder_attention_mask, encoder_attention_mask
    ):
        input_shape = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
        extended_attention_mask = get_extended_attention_mask(
            decoder_attention_mask, input_shape, True
        )

        decoder_hidden_states = self.embed_tokens(decoder_input_ids)

        batch_size, seq_length = decoder_input_ids.shape
        decoder_position_bias = (
            self.compute_bias(seq_length, seq_length).to(decoder_input_ids.device)
            + extended_attention_mask
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
            decoder_hidden_states,
            decoder_position_bias,
            encoder_decoder_position_bias,
        )


class SwitchRouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(
            config.hidden_size, self.num_experts, bias=config.router_bias
        )
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def _compute_router_probabilities(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: [num_groups, tokens_per_group, num_experts]
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(
            router_logits, dim=-1, dtype=router_logits.dtype
        ).to(hidden_states.dtype)
        return router_probabilities, router_logits

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forwarded_states = self.layer_norm(hidden_states)
        router_probs, _ = self._compute_router_probabilities(forwarded_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(
            expert_index, num_classes=self.num_experts
        )

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)

        # print("expert_index", torch.argmax(expert_index, dim=-1))
        # print("router_probs", router_probs)
        # print("forwarded_states", forwarded_states)
        return forwarded_states, expert_index, router_probs


class SwitchLMPredictionHead(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model_dim = config.d_model

    def forward(self, hidden_states):
        hidden_states = hidden_states * (self.model_dim**-0.5)
        return self.lm_head(hidden_states)


class SwitchDenseActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wo(self.act(self.wi(hidden_states)))
        return hidden_states

from transformers.activations import ACT2FN

# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->SwitchTransformers
class SwitchDenseGatedActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        return self.wo(self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states))

class SwitchLayerFF(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_gated: bool = False):
        super().__init__()
        if is_gated:
            self.mlp = SwitchDenseGatedActDense(config)
        else:
            self.mlp = SwitchDenseActDense(config)
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(self.layer_norm(hidden_states))
        return hidden_states


class SwitchFinalLayerNorm(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        return self.layer_norm(hidden_states)


class SwitchAttention(nn.Module):
    def __init__(
        self, config: SwitchTransformersConfig, is_cross_attention: bool = False
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.is_cross_attention = is_cross_attention

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def shape(self, states: torch.Tensor, batch_size: int):
        """projection"""
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(self, states: torch.Tensor, batch_size: int):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def forward(
        self,
        hidden_states,
        position_bias,
        key_value_states,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

        batch_size, seq_length = hidden_states.shape[:2]
        # real_seq_length = seq_length

        # key_length = (
        #     real_seq_length
        #     if not self.is_cross_attention
        #     else key_value_states.shape[1]
        # )

        # query_states = self.shape(
        #     self.q(hidden_states), batch_size
        # )  # (batch_size, n_heads, seq_length, dim_per_head)
        # key_states = self.shape(
        #     self.k(hidden_states if not self.is_cross_attention else key_value_states),
        #     batch_size,
        # )
        # value_states = self.shape(
        #     self.v(hidden_states if not self.is_cross_attention else key_value_states),
        #     batch_size,
        # )

        # print("query_states", self.is_decoder, query_states.shape)
        # print("key_states", self.is_decoder, key_states.shape)
        # print("value_states", self.is_decoder, value_states.shape)

        # compute scores
        # position_bias = position_bias + attention_mask
        scores = (
            torch.matmul(
                self.shape(self.q(hidden_states), batch_size),
                self.shape(
                    self.k(
                        hidden_states
                        if not self.is_cross_attention
                        else key_value_states
                    ),
                    batch_size,
                ).transpose(3, 2),
            )
            + position_bias
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # print("scores", self.is_decoder, scores.shape)

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)

        # print("attn_weights", self.is_decoder, attn_weights.shape)

        attn_output = self.unshape(
            torch.matmul(
                attn_weights,
                self.shape(
                    self.v(
                        hidden_states
                        if not self.is_cross_attention
                        else key_value_states
                    ),
                    batch_size,
                ),
            ),
            batch_size,
        )  # (batch_size, seq_length, dim)

        # print("attn_output", self.is_decoder, attn_output.shape)

        attn_output = self.o(attn_output)

        # print("attn_output", self.is_decoder, attn_output.shape)

        return attn_output


# class SwitchAttentionRelative(SwitchAttention):
#     def __init__(self, config: SwitchTransformersConfig):
#         super().__init__(config)

#         self.relative_attention_num_buckets = config.relative_attention_num_buckets
#         self.relative_attention_max_distance = config.relative_attention_max_distance

#         self.relative_attention_bias = nn.Embedding(
#             self.relative_attention_num_buckets, self.n_heads
#         )

#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         position_bias,
#         key_value_states,
#     ):
#         """
#         Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
#         """
#         # Input is (batch_size, seq_length, dim)
#         # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
#         # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

#         batch_size, seq_length = hidden_states.shape[:2]
#         real_seq_length = seq_length
#         key_length = real_seq_length

#         # compute scores
#         scores = torch.matmul(
#             self.shape(self.q(hidden_states), batch_size),
#             self.shape(
#                 self.k(hidden_states),
#                 batch_size,
#             ).transpose(3, 2),
#         )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

#         position_bias = self.compute_bias(real_seq_length, key_length)
#         position_bias = position_bias + attention_mask
#         scores += position_bias
#         attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
#             scores
#         )  # (batch_size, n_heads, seq_length, key_length)
#         attn_output = self.o(
#             self.unshape(
#                 torch.matmul(
#                     attn_weights,
#                     self.shape(
#                         self.v(hidden_states),
#                         batch_size,
#                     ),
#                 ),
#                 batch_size,
#             )  # (batch_size, seq_length, dim)
#         )

#         outputs = (attn_output,) + (position_bias,)
#         return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.SelfAttention = SwitchAttention(config, False)

        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        # attention_mask,
        position_bias,
    ):
        attention_output = self.SelfAttention(
            self.layer_norm(hidden_states),
            # attention_mask,
            position_bias,
            torch.empty(0),
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = SwitchAttention(config, True)
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        decoder_hidden_states,
        encoder_hidden_states,
        # encoder_extended_attention_mask,
        encoder_decoder_position_bias,
    ):
        attention_output = self.EncDecAttention(
            hidden_states=self.layer_norm(decoder_hidden_states),
            # attention_mask=encoder_extended_attention_mask,
            position_bias=encoder_decoder_position_bias,
            key_value_states=encoder_hidden_states,
        )
        decoder_hidden_states = decoder_hidden_states + attention_output
        return decoder_hidden_states  # decoder_hidden_states


class SwitchEncoderBlock(nn.Module):
    def __init__(
        self, config: SwitchTransformersConfig, has_relative_attention_bias=False
    ):
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
        # extended_attention_mask,
    ):
        encoder_hidden_states = self.attention(
            encoder_hidden_states,
            # extended_attention_mask,
            encoder_position_bias,
        )  # encoder_hidden_states
        return encoder_hidden_states


class SwitchDecoderBlock(nn.Module):
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
        decoder_hidden_states,
        encoder_hidden_states,
        # extended_attention_mask,
        # encoder_extended_attention_mask,
        decoder_position_bias,
        encoder_decoder_position_bias,
    ):
        decoder_hidden_states = self.attention(
            decoder_hidden_states,
            # extended_attention_mask,
            decoder_position_bias,
        )  # decoder_hidden_states

        decoder_hidden_states = self.cross_attention(
            decoder_hidden_states,
            encoder_hidden_states,
            # encoder_extended_attention_mask,
            encoder_decoder_position_bias,
        )

        return decoder_hidden_states


class SwitchSparseMLP(nn.Module):
    def __init__(
        self,
        config: SwitchTransformersConfig,
        expert_class: nn.Module = SwitchDenseActDense,
    ):
        super().__init__()
        self.router = SwitchRouter(config)
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # expert_index, router_probs
        forwarded_states, router_mask, router_probs = self.router(hidden_states)
        next_states = forwarded_states.clone()

        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            if torch.any(token_indices):
                next_states[token_indices] = expert(forwarded_states[token_indices])
                # print("output", idx, next_states[token_indices])
        return hidden_states + router_probs * next_states


class SwitchMLPLayer(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_sparse=False):
        super().__init__()
        self.is_sparse = is_sparse

        if not self.is_sparse:
            self.mlp = SwitchLayerFF(config)
        else:
            self.mlp = SwitchSparseMLP(config)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class SwitchBlock(nn.Module):
    def __init__(
        self,
        config: SwitchTransformersConfig,
        has_relative_attention_bias=False,
        is_sparse=False,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.is_sparse = is_sparse
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(SwitchMLPLayer(config, is_sparse=self.is_sparse))

    def forward(
        self,
        decoder_hidden_states=None,
        encoder_hidden_states=None,
        # extended_attention_mask=None,
        # encoder_extended_attention_mask=None,
        decoder_position_bias=None,
        encoder_decoder_position_bias=None,
    ):

        if self.is_decoder:
            hidden_state = self.layer[0](
                decoder_hidden_states,
                # extended_attention_mask,
                decoder_position_bias,
            )
            hidden_state = self.layer[1](
                hidden_state,
                encoder_hidden_states,
                # encoder_extended_attention_mask,
                encoder_decoder_position_bias,
            )
        else:
            hidden_state = self.layer[0](
                encoder_hidden_states,
                # extended_attention_mask,
                encoder_decoder_position_bias,
            )

        hidden_state = self.layer[-1](hidden_state)
        return hidden_state


class SwitchStack(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, embed_tokens=None):
        super().__init__()
        if config.is_decoder:
            self.embed_tokens = DecoderTokenEmbeddings(config)
        else:
            self.embed_tokens = EncoderTokenEmbeddings(config)

        if embed_tokens is not None:
            self.embed_tokens.embed_tokens.weight = embed_tokens.weight

        self.is_decoder = config.is_decoder
        self.num_layers = config.num_layers

        sparse_step = (
            config.decoder_sparse_step
            if self.is_decoder
            else config.encoder_sparse_step
        )
        config.num_layers = (
            config.num_decoder_layers if self.is_decoder else config.num_layers
        )
        self.block = nn.ModuleList()

        for i in range(config.num_layers):

            is_sparse = (i % sparse_step == 1) if sparse_step > 0 else False

            self.block.append(
                SwitchBlock(
                    config,
                    has_relative_attention_bias=bool(i == 0),
                    is_sparse=is_sparse,
                )
            )

        self.final_layer_norm = SwitchFinalLayerNorm(config)

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_position_bias=None,
        output_hidden_states=False,
    ):

        all_hidden_states = () if output_hidden_states else None

        if self.is_decoder:
            (
                hidden_states,
                decoder_position_bias,
                encoder_decoder_position_bias,
            ) = self.embed_tokens(input_ids, attention_mask, encoder_attention_mask)
            # print("encoder_decoder_position_bias", encoder_decoder_position_bias.shape)
        else:
            (
                hidden_states,
                encoder_position_bias,
            ) = self.embed_tokens(input_ids, attention_mask)

        for i in range(self.num_layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.is_decoder:
                hidden_states = self.block[i](
                    decoder_hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    # extended_attention_mask=extended_attention_mask,
                    # encoder_extended_attention_mask=encoder_extended_attention_mask,
                    decoder_position_bias=decoder_position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                )
            else:
                hidden_states = self.block[i](
                    encoder_hidden_states=hidden_states,
                    # extended_attention_mask=extended_encoder_attention_mask,
                    encoder_decoder_position_bias=encoder_position_bias,
                )

        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


class SwitchModel(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.model_dim = config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SwitchStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = SwitchStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        output_hidden_states=False,
    ):

        all_hidden_states = () if output_hidden_states else None
        encoder_hidden_states, hidden_states = self.encoder(
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
        )
        all_hidden_states = all_hidden_states + hidden_states

        decoder_hidden_states, hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        all_hidden_states = all_hidden_states + hidden_states
        # print("decoder_hidden_states", decoder_hidden_states)
        decoder_hidden_states = decoder_hidden_states * (self.model_dim**-0.5)
        lm_logits = self.lm_head(decoder_hidden_states)

        return lm_logits, all_hidden_states
