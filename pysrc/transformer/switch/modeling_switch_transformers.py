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

    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class EncoderTokenEmbeddings(nn.Module):
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, encoder_input_ids, encoder_attention_mask):

        input_shape = encoder_input_ids.size()
        input_ids = encoder_input_ids.view(-1, input_shape[-1])
        encoder_hidden_states = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_encoder_attention_mask = get_extended_attention_mask(
            encoder_attention_mask, input_shape, is_decoder=False
        )
        encoder_position_bias = torch.empty((batch_size, self.num_heads, seq_length, 1))
        encoder_hidden_states = self.dropout(encoder_hidden_states)

        return (
            encoder_hidden_states,
            extended_encoder_attention_mask,
            encoder_position_bias,
        )


class DecoderTokenEmbeddings(nn.Module):
    def __init__(self, config: SwitchTransformersConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
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
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        batch_size, seq_length = decoder_input_ids.shape
        decoder_position_bias = torch.empty((batch_size, self.num_heads, seq_length, 1))

        return (
            decoder_hidden_states,
            encoder_extended_attention_mask,
            extended_attention_mask,
            decoder_position_bias,
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
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens

    def _compute_router_probabilities(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.jitter_noise > 0:
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(
                hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            uniform_distrib = uniform_distrib * (
                distrib_lower_bound - distrib_upper_bound
            )

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= uniform_distrib

        # Shape: [num_groups, tokens_per_group, num_experts]
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(
            router_logits, dim=-1, dtype=router_logits.dtype
        ).to(hidden_states.dtype)
        return router_probabilities, router_logits

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

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
        return expert_index, router_probs

class SwitchLMPredictionHead(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

class SwitchLayerFF(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_gated: bool = False):
        super().__init__()
        if is_gated:
            self.mlp = SwitchTransformersDenseGatedActDense(config)
        else:
            self.mlp = SwitchTransformersDenseActDense(config)
        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class SwitchFinalLayerNorm(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwitchAttention(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

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

    def project(
        self,
        hidden_states: torch.Tensor,
        proj_layer: Callable[[torch.Tensor], torch.Tensor],
        key_value_states,
        past_key_value,
        batch_size: int,
    ):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = self.shape(proj_layer(hidden_states), batch_size)
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = self.shape(proj_layer(key_value_states), batch_size)

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_bias,
        key_value_states,
        is_decoder: bool,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        key_length = real_seq_length if not is_decoder else key_value_states.shape[1]

        # get query states
        query_states = self.shape(
            self.q(hidden_states), batch_size
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        key_states = self.shape(
            self.k(key_value_states if is_decoder else hidden_states),
            batch_size,
        )
        value_states = self.shape(
            self.v(key_value_states if is_decoder else hidden_states),
            batch_size,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = self.unshape(
            torch.matmul(attn_weights, value_states), batch_size
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        outputs = (attn_output,) + (position_bias,)
        return outputs


class SwitchAttentionRelative(SwitchAttention):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)

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
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
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

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_bias,
        key_value_states,
        is_decoder: bool,
        # layer_head_mask,
        # query_length: int = None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        # if past_key_value is not None:
        #     assert (
        #         len(past_key_value) == 2
        #     ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        #     real_seq_length += (
        #         past_key_value[0].shape[2] if query_length is None else query_length
        #     )

        key_length = real_seq_length if not is_decoder else key_value_states.shape[1]

        # get query states
        query_states = self.shape(
            self.q(hidden_states), batch_size
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        # if key_value_states is None:
        #     # self-attn
        #     # (batch_size, n_heads, seq_length, dim_per_head)
        #     key_states = self.shape(self.k(hidden_states), batch_size)
        # elif past_key_value is None:
        #     # cross-attn
        #     # (batch_size, n_heads, seq_length, dim_per_head)
        #     key_states = self.shape(self.k(key_value_states), batch_size)
        key_states = self.shape(
            self.k(key_value_states if is_decoder else hidden_states),
            batch_size,
        )
        value_states = self.shape(
            self.v(key_value_states if is_decoder else hidden_states),
            batch_size,
        )
        # key_states = self.project(
        #     hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        #     , batch_size
        # )
        # value_states = self.project(
        #     hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        #     , batch_size
        # )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if not is_decoder:
            position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            # if past_key_value is not None:
            #     position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if attention_mask is not None:
                position_bias = (
                    position_bias + attention_mask
                )  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # # Mask heads if we want to
        # if layer_head_mask is not None:
        #     attn_weights = attn_weights * layer_head_mask

        attn_output = self.unshape(
            torch.matmul(attn_weights, value_states), batch_size
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        # present_key_value_state = (
        #     (key_states, value_states) if (self.is_decoder and use_cache) else None
        # )
        # outputs = (attn_output,) + (None,) + (position_bias,)
        outputs = (attn_output,) + (position_bias,)

        # if output_attentions:
        #     outputs = outputs + (attn_weights,)
        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.SelfAttention = (
            SwitchAttentionRelative(config)
            if has_relative_attention_bias
            else SwitchAttention(config)
        )
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_bias,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states, attention_mask, position_bias, torch.empty(0), False
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return hidden_states, attention_output[1]  # hidden_states, position_bias

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = SwitchAttention(config)
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        decoder_hidden_states,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
    ):
        normed_hidden_states = self.layer_norm(decoder_hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            encoder_extended_attention_mask,
            encoder_decoder_position_bias,
            encoder_hidden_states,
            True,
        )
        decoder_hidden_states = decoder_hidden_states + self.dropout(
            attention_output[0]
        )
        return (
            decoder_hidden_states,
            attention_output[1],
        )  # decoder_hidden_states, encoder_decoder_position_bias



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
        extended_attention_mask,
    ):
        encoder_hidden_states, encoder_position_bias = self.attention(
            encoder_hidden_states,
            extended_attention_mask,
            encoder_position_bias,
        )  # encoder_hidden_states, position_bias

        return encoder_hidden_states, encoder_position_bias


class SwitchDecoderBlock(nn.Module):
    def __init__(
        self, config: SwitchTransformersConfig, has_relative_attention_bias=False
    ):
        super().__init__()
        self.has_relative_attention_bias = (
            has_relative_attention_bias  # the first block has relative attention bias
        )
        self.attention = T5LayerSelfAttention(
            config, has_relative_attention_bias
        )
        self.cross_attention = T5LayerCrossAttention(config)

    def forward(
        self,
        decoder_hidden_states,
        encoder_hidden_states,
        extended_attention_mask,
        encoder_extended_attention_mask,
        decoder_position_bias,
        encoder_decoder_position_bias,
    ):
        decoder_hidden_states, decoder_position_bias = self.attention(
            encoder_hidden_states,
            extended_attention_mask,
            decoder_position_bias,
        )  # decoder_hidden_states, decoder_position_bias

        decoder_hidden_states, encoder_decoder_position_bias = self.cross_attention(
            decoder_hidden_states,
            encoder_hidden_states,
            encoder_extended_attention_mask,
            encoder_decoder_position_bias,
        )

        return (
            decoder_hidden_states,
            decoder_position_bias,
            encoder_decoder_position_bias,
        )
