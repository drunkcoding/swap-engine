from typing import Callable, List
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy

from transformers.activations import ACT2FN
from .configuration_switch import SwitchConfig


def prepare_decoder_input_ids_for_generation(
    batch_size: int,
    decoder_start_token_id: int,
    device: torch.device,
) -> torch.LongTensor:
    decoder_input_ids = (
        torch.ones((batch_size, 1), dtype=torch.long, device=device)
        * decoder_start_token_id
    )
    return decoder_input_ids


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
    device: torch.device,
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


class T5DenseActDense(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class SwitchExpert(nn.Module):
    def __init__(self, config: SwitchConfig, expert_num: int):
        super().__init__()
        self.expert_num = expert_num
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        token_features = self.wi(hidden_states)
        token_features = self.act(token_features)
        token_features = self.dropout(token_features)
        token_features = self.wo(token_features)
        return token_features

    # def forward(self, hidden_states, routes):
    #     indexes_list = torch.nonzero(routes == self.expert_num).flatten()
    #     if len(indexes_list) == 0:
    #         return torch.ones_like(hidden_states[..., 0])

    #     token_features = hidden_states.reshape(-1, hidden_states.shape[-1])[
    #         indexes_list
    #     ]
    #     # token_features = super().forward(token_features)

    #     token_features = self.wi(token_features)
    #     token_features = self.act(token_features)
    #     token_features = self.dropout(token_features)
    #     token_features = self.wo(token_features)

    #     # hidden_states.view(-1, hidden_states.shape[-1])[indexes_list] = token_features
    #     # return hidden_states
    #     # print(token_features.shape)
    #     return token_features

        # token_features = hidden_states.view(-1, hidden_states.shape[-1])[
        #     indexes_list[self.expert_num], ...
        # ]


class EncoderTokenEmbeddings(nn.Module):
    def __init__(self, config: SwitchConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, encoder_input_ids, encoder_attention_mask):

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
        )


class DecoderTokenEmbeddings(nn.Module):
    def __init__(self, config: SwitchConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, decoder_input_ids, decoder_attention_mask, encoder_attention_mask
    ):
        # decoder_input_ids = self._prepare_decoder_input_ids_for_generation(
        #     encoder_attention_mask.shape[0],
        #     self.config.decoder_start_token_id,
        #     self.config.eos_token_id,
        #     device=encoder_attention_mask.device,
        # )
        # decoder_input_ids = prepare_decoder_input_ids_for_generation(
        #     encoder_attention_mask.shape[0],
        #     0,
        #     device=encoder_attention_mask.device,
        # )
        # decoder_attention_mask = decoder_input_ids.new_ones(
        #     decoder_input_ids.shape, dtype=torch.long
        # )
        input_shape = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])

        encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

        extended_attention_mask = get_extended_attention_mask(
            decoder_attention_mask, input_shape, decoder_input_ids.device, True
        )

        decoder_hidden_states = self.embedding(decoder_input_ids)
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        batch_size, seq_length = decoder_input_ids.shape
        decoder_position_bias = torch.empty((batch_size, self.num_heads, seq_length, 1))

        return (
            decoder_hidden_states,
            encoder_extended_attention_mask,
            extended_attention_mask,
            decoder_position_bias,
        )


# class RelativePositionBiases(nn.Module):
#     def __init__(self, config: SwitchConfig):
#         super().__init__()

#         self.relative_attention_num_buckets = config.relative_attention_num_buckets
#         self.relative_attention_max_distance = config.relative_attention_max_distance
#         self.rel_embedding = nn.Embedding(
#             self.relative_attention_num_buckets, self.n_heads
#         )

#     @staticmethod
#     def _relative_position_bucket(
#         relative_position, bidirectional=True, num_buckets=32, max_distance=128
#     ):
#         """
#         Adapted from Mesh Tensorflow:
#         https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

#         Translate relative position to a bucket number for relative attention. The relative position is defined as
#         memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
#         position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
#         small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
#         positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
#         This should allow for more graceful generalization to longer sequences than the model has been trained on

#         Args:
#             relative_position: an int32 Tensor
#             bidirectional: a boolean - whether the attention is bidirectional
#             num_buckets: an integer
#             max_distance: an integer

#         Returns:
#             a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
#         """
#         relative_buckets = 0
#         if bidirectional:
#             num_buckets //= 2
#             relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
#             relative_position = torch.abs(relative_position)
#         else:
#             relative_position = -torch.min(
#                 relative_position, torch.zeros_like(relative_position)
#             )
#         # now relative_position is in the range [0, inf)

#         # half of the buckets are for exact increments in positions
#         max_exact = num_buckets // 2
#         is_small = relative_position < max_exact

#         # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
#         relative_position_if_large = max_exact + (
#             torch.log(relative_position.float() / max_exact)
#             / math.log(max_distance / max_exact)
#             * (num_buckets - max_exact)
#         ).to(torch.long)
#         relative_position_if_large = torch.min(
#             relative_position_if_large,
#             torch.full_like(relative_position_if_large, num_buckets - 1),
#         )

#         relative_buckets += torch.where(
#             is_small, relative_position, relative_position_if_large
#         )
#         return relative_buckets

#     def compute_bias(self, query_length, key_length, device=None):
#         """Compute binned relative position bias"""
#         if device is None:
#             device = self.rel_embedding.weight.device
#         context_position = torch.arange(query_length, dtype=torch.long, device=device)[
#             :, None
#         ]
#         memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
#             None, :
#         ]
#         relative_position = (
#             memory_position - context_position
#         )  # shape (query_length, key_length)
#         relative_position_bucket = self._relative_position_bucket(
#             relative_position,  # shape (query_length, key_length)
#             bidirectional=(not self.is_decoder),
#             num_buckets=self.relative_attention_num_buckets,
#             max_distance=self.relative_attention_max_distance,
#         )
#         values = self.rel_embedding(
#             relative_position_bucket
#         )  # shape (query_length, key_length, num_heads)
#         values = values.permute([2, 0, 1]).unsqueeze(
#             0
#         )  # shape (1, num_heads, query_length, key_length)
#         return values

#     def forward(self, query_length, key_length, device=None):
#         """Input shape: Time(SeqLen)"""
#         device = self.rel_embedding.weight.device
#         if query_length == key_length and query_length == getattr(
#             self, "cached_key_length", None
#         ):
#             return self.cache
#         else:
#             values = self.compute_bias(query_length, key_length, device=device)
#             self.cache = values
#             self.cached_key_length = key_length
#             return values


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        # convert into half-precision if necessary
        if self.scale.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.scale.dtype)

        return self.scale * hidden_states


class SwitchLayerFF(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class SwitchFinalLayerNorm(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwitchAggregator(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.n_experts = config.num_experts
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, expert_output, routes, route_prob_max):
        # Capture the shape to change shapes later
        batch_size, seq_len, d_model = hidden_states.shape

        # Initialize an empty tensor to store outputs
        final_output = hidden_states.new_zeros((batch_size * seq_len, d_model))

        # Assign to final output
        for i in range(self.n_experts):
            indexes_list = torch.nonzero(routes == i)
            if len(indexes_list) > 0:
                final_output[indexes_list] = expert_output[
                    i
                ]  # .view(-1, hidden_states.shape[-1])[indexes_list]

        final_output = final_output * route_prob_max.view(-1, 1)
        # if self.is_scale_prob:
        #     # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
        #     final_output = final_output * route_prob_max.view(-1, 1)
        # else:
        #     # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
        #     # (this is something we experimented with).
        #     final_output = final_output * (
        #         route_prob_max / route_prob_max.detach()
        #     ).view(-1, 1)

        # Change the shape of the final output back to `[batch_size, seq_len, d_model]`
        final_output = final_output.view(batch_size, seq_len, d_model)
        hidden_states = hidden_states + self.dropout(final_output)
        return hidden_states


class SwitchRouter(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = config.d_model

        self.n_experts = config.num_experts
        self.capacity_factor = config.eval_capacity_factor

        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # print(hidden_states.shape)
        batch_size, seq_len, _ = hidden_states.shape

        num_tokens = batch_size * seq_len
        capacity = torch.ceil(torch.tensor(num_tokens / self.n_experts) * self.capacity_factor).to(
            torch.long
        )

        # Flatten the sequence and batch dimensions
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.router(hidden_states))
        # print(route_prob.shape)

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        # print(route_prob_max.shape, routes.shape)
        print(routes)
        routes = routes.long()

        # expert_mask = F.one_hot(routes, self.n_experts).long()
        # print(expert_mask)

        # token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # token_priority = torch.max(token_priority, dim=-1)

        # print(token_priority)
        # exit()
        return routes, route_prob_max

        # # Get indexes of tokens going to each expert
        # indexes_list = []
        # for i in range(self.n_experts):
        #     token_list = torch.nonzero(routes == i).flatten()
        #     indexes_list.append((routes == i).nonzero().view(-1))
        # indexes_list = torch.cat(
        #     [
        #         torch.eq(routes, i).nonzero(as_tuple=True)[0]
        #         for i in range(self.n_experts)
        #     ]
        # )
        # # indexes_list = indexes_list.reshape(batch_size, -1)
        # # route_prob_max = route_prob_max.reshape(batch_size, -1)

        # return indexes_list, route_prob_max


# from transformers.pytorch_utils import (
#     find_pruneable_heads_and_indices,
#     prune_linear_layer,
# )


class SwitchAttention(nn.Module):
    def __init__(self, config: SwitchConfig):
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
    def __init__(self, config: SwitchConfig):
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


# class T5Attention(nn.Module):
#     def __init__(self, config: SwitchConfig, has_relative_attention_bias):
#         super().__init__()
#         self.is_decoder = config.is_decoder
#         self.has_relative_attention_bias = has_relative_attention_bias
#         self.relative_attention_num_buckets = config.relative_attention_num_buckets
#         self.relative_attention_max_distance = config.relative_attention_max_distance
#         self.d_model = config.d_model
#         self.key_value_proj_dim = config.d_kv
#         self.n_heads = config.num_heads
#         self.dropout = config.dropout_rate
#         self.inner_dim = self.n_heads * self.key_value_proj_dim

#         # Mesh TensorFlow initialization to avoid scaling before softmax
#         self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
#         self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
#         self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
#         self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

#         if self.has_relative_attention_bias:
#             self.relative_attention_bias = nn.Embedding(
#                 self.relative_attention_num_buckets, self.n_heads
#             )
#         self.pruned_heads = set()
#         self.gradient_checkpointing = False

#     @staticmethod
#     def _relative_position_bucket(
#         relative_position, bidirectional: bool=True, num_buckets: int=32, max_distance: int=128
#     ):
#         """
#         Adapted from Mesh Tensorflow:
#         https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

#         Translate relative position to a bucket number for relative attention. The relative position is defined as
#         memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
#         position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
#         small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
#         positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
#         This should allow for more graceful generalization to longer sequences than the model has been trained on

#         Args:
#             relative_position: an int32 Tensor
#             bidirectional: a boolean - whether the attention is bidirectional
#             num_buckets: an integer
#             max_distance: an integer

#         Returns:
#             a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
#         """
#         relative_buckets = 0
#         if bidirectional:
#             num_buckets = torch.floor(num_buckets / 2)
#             relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
#             relative_position = torch.abs(relative_position)
#         else:
#             relative_position = -torch.min(
#                 relative_position, torch.zeros_like(relative_position)
#             )
#         # now relative_position is in the range [0, inf)

#         # half of the buckets are for exact increments in positions
#         max_exact = torch.floor(num_buckets / 2)
#         is_small = relative_position < max_exact

#         # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
#         relative_position_if_large = max_exact + (
#             torch.log(relative_position.float() / max_exact)
#             / math.log(max_distance / max_exact)
#             * (num_buckets - max_exact)
#         ).to(torch.long)
#         relative_position_if_large = torch.min(
#             relative_position_if_large,
#             torch.full_like(relative_position_if_large, num_buckets - 1),
#         )

#         relative_buckets += torch.where(
#             is_small, relative_position, relative_position_if_large
#         )
#         return relative_buckets

#     def compute_bias(self, query_length: int, key_length: int, device: torch.device):
#         """Compute binned relative position bias"""
#         if device is None:
#             device = self.relative_attention_bias.weight.device
#         context_position = torch.arange(query_length, dtype=torch.long, device=device)[
#             :, None
#         ]
#         memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
#             None, :
#         ]
#         relative_position = (
#             memory_position - context_position
#         )  # shape (query_length, key_length)
#         relative_position_bucket = self._relative_position_bucket(
#             relative_position,  # shape (query_length, key_length)
#             bidirectional=(not self.is_decoder),
#             num_buckets=self.relative_attention_num_buckets,
#             max_distance=self.relative_attention_max_distance,
#         )
#         values = self.relative_attention_bias(
#             relative_position_bucket
#         )  # shape (query_length, key_length, num_heads)
#         values = values.permute([2, 0, 1]).unsqueeze(
#             0
#         )  # shape (1, num_heads, query_length, key_length)
#         return values
#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         position_bias,
#         key_value_states,
#         is_decoder: bool,
#         # layer_head_mask,
#         # query_length: int = None,
#     ):
#         """
#         Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
#         """
#         # Input is (batch_size, seq_length, dim)
#         # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
#         # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

#         batch_size, seq_length = hidden_states.shape[:2]

#         real_seq_length = seq_length

#         # if past_key_value is not None:
#         #     assert (
#         #         len(past_key_value) == 2
#         #     ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
#         #     real_seq_length += (
#         #         past_key_value[0].shape[2] if query_length is None else query_length
#         #     )

#         key_length = real_seq_length if not is_decoder else key_value_states.shape[1]

#         # get query states
#         query_states = self.shape(
#             self.q(hidden_states), batch_size
#         )  # (batch_size, n_heads, seq_length, dim_per_head)

#         # get key/value states
#         # if key_value_states is None:
#         #     # self-attn
#         #     # (batch_size, n_heads, seq_length, dim_per_head)
#         #     key_states = self.shape(self.k(hidden_states), batch_size)
#         # elif past_key_value is None:
#         #     # cross-attn
#         #     # (batch_size, n_heads, seq_length, dim_per_head)
#         #     key_states = self.shape(self.k(key_value_states), batch_size)
#         key_states = self.shape(
#             self.k(key_value_states if is_decoder else hidden_states),
#             batch_size,
#         )
#         value_states = self.shape(
#             self.v(key_value_states if is_decoder else hidden_states),
#             batch_size,
#         )
#         # key_states = self.project(
#         #     hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
#         #     , batch_size
#         # )
#         # value_states = self.project(
#         #     hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
#         #     , batch_size
#         # )

#         # compute scores
#         scores = torch.matmul(
#             query_states, key_states.transpose(3, 2)
#         )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

#         if self.has_relative_attention_bias and not is_decoder:
#             position_bias = self.compute_bias(
#                 real_seq_length, key_length, device=scores.device
#             )

#             # if key and values are already calculated
#             # we want only the last query position bias
#             # if past_key_value is not None:
#             #     position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

#             if attention_mask is not None:
#                 position_bias = (
#                     position_bias + attention_mask
#                 )  # (batch_size, n_heads, seq_length, key_length)

#         scores += position_bias
#         attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
#             scores
#         )  # (batch_size, n_heads, seq_length, key_length)
#         attn_weights = nn.functional.dropout(
#             attn_weights, p=self.dropout, training=self.training
#         )  # (batch_size, n_heads, seq_length, key_length)

#         # # Mask heads if we want to
#         # if layer_head_mask is not None:
#         #     attn_weights = attn_weights * layer_head_mask

#         attn_output = self.unshape(
#             torch.matmul(attn_weights, value_states), batch_size
#         )  # (batch_size, seq_length, dim)
#         attn_output = self.o(attn_output)

#         # present_key_value_state = (
#         #     (key_states, value_states) if (self.is_decoder and use_cache) else None
#         # )
#         # outputs = (attn_output,) + (None,) + (position_bias,)
#         outputs = (attn_output,) + (position_bias,)

#         # if output_attentions:
#         #     outputs = outputs + (attn_weights,)
#         return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.SelfAttention = (
            SwitchAttentionRelative(config)
            if has_relative_attention_bias
            else SwitchAttention(config)
        )
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
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
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
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
    def __init__(self, config: SwitchConfig, has_relative_attention_bias=False):
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
        # encoder_hidden_states: (batch_size, seq_length, dim)
        # encoder_attention_mask: (batch_size, seq_length)
        # position_bias: (batch_size, n_heads, seq_length, key_length)
        # extended_attention_mask: (batch_size, n_heads, seq_length, key_length)

        # if self.has_relative_attention_bias:
        #     input_shape = attention_mask.size()
        #     extended_attention_mask = get_extended_attention_mask(
        #         attention_mask, input_shape, attention_mask.device, False
        #     )
        encoder_hidden_states, encoder_position_bias = self.attention(
            encoder_hidden_states,
            extended_attention_mask,
            encoder_position_bias,
        )  # encoder_hidden_states, position_bias

        return encoder_hidden_states, encoder_position_bias


class SwitchDecoderBlock(nn.Module):
    def __init__(self, config: SwitchConfig, has_relative_attention_bias=False):
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
        extended_attention_mask,
        encoder_extended_attention_mask,
        decoder_position_bias,
        encoder_decoder_position_bias,
    ):
        # decoder_hidden_states: (batch_size, seq_length, dim)
        # encoder_hidden_states: (batch_size, seq_length, dim)
        # decoder_attention_mask: (batch_size, seq_length)
        # position_bias: (batch_size, n_heads, seq_length, key_length)
        # encoder_decoder_position_bias: (batch_size, n_heads, seq_length, key_length)

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


# class SwitchAttention(nn.Module):
#     def __init__(self, config: SwitchConfig, has_relative_attention_bias=False):
#         super().__init__()
#         self.is_decoder = config.is_decoder
#         self.layer = nn.ModuleList()
#         self.layer.append(
#             T5LayerSelfAttention(
#                 config, has_relative_attention_bias=has_relative_attention_bias
#             )
#         )
#         if self.is_decoder:
#             self.layer.append(T5LayerCrossAttention(config))

#         # self.layer.append(T5LayerFF(config))

#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         position_bias,
#         encoder_decoder_position_bias,
#         encoder_hidden_states,
#         encoder_attention_mask,
#     ):

#         self_attention_outputs = self.layer[0](
#             hidden_states,
#             attention_mask=attention_mask,
#             position_bias=position_bias,
#         )

#         # return hidden_states

#         hidden_states, self_position_bias = self_attention_outputs
#         # attention_outputs = self_attention_outputs[
#         #     2:
#         # ]  # Keep self-attention outputs and relative position weights

#         # clamp inf values to enable fp16 training
#         if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
#             clamp_value = torch.finfo(hidden_states.dtype).max - 1000
#             hidden_states = torch.clamp(
#                 hidden_states, min=-clamp_value, max=clamp_value
#             )

#         # return hidden_states

#         do_cross_attention = self.is_decoder and encoder_hidden_states is not None
#         if do_cross_attention:
#             # the actual query length is unknown for cross attention
#             # if using past key value states. Need to inject it here
#             # if present_key_value_state is not None:
#             #     query_length = present_key_value_state[0].shape[2]
#             # else:
#             #     query_length = None

#             cross_attention_outputs = self.layer[1](
#                 hidden_states,
#                 key_value_states=encoder_hidden_states,
#                 attention_mask=encoder_attention_mask,
#                 # query_length=query_length,
#             )
#             hidden_states, cross_position_bias = cross_attention_outputs[0]

#             # clamp inf values to enable fp16 training
#             if (
#                 hidden_states.dtype == torch.float16
#                 and torch.isinf(hidden_states).any()
#             ):
#                 clamp_value = torch.finfo(hidden_states.dtype).max - 1000
#                 hidden_states = torch.clamp(
#                     hidden_states, min=-clamp_value, max=clamp_value
#                 )

#             # # Combine self attn and cross attn key value states
#             # if present_key_value_state is not None:
#             #     present_key_value_state = (
#             #         present_key_value_state + cross_attention_outputs[1]
#             #     )

#             # Keep cross-attention outputs and relative position weights
#             # attention_outputs = attention_outputs + cross_attention_outputs[2:]

#         outputs = (
#             (hidden_states,) + (self_position_bias,) + (cross_position_bias,)
#             if do_cross_attention
#             else (hidden_states,) + (self_position_bias,)
#         )
#         return hidden_states


class SwitchModel(nn.Module):
    def __init__(self, config: SwitchConfig):
        super().__init__()

        # self.config = config
        self.n_heads = config.num_heads
        self.n_experts = config.num_experts
        self.d_model = config.d_model
        self.n_layers = config.num_layers

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # shared embedding layer
        self.token_embedder = nn.Embedding(config.vocab_size, config.d_model)

        # encoder layers
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder_layers.append(EncoderTokenEmbeddings(encoder_config))
        for i in range(encoder_config.num_layers):
            self.encoder_layers.append(SwitchEncoderBlock(encoder_config, bool(i == 0)))
            # moe is attached every two layers
            if i % 2 == 1:
                self.encoder_layers.append(SwitchRouter(encoder_config))
                for expert_num in range(encoder_config.num_experts):
                    self.encoder_layers.append(SwitchExpert(encoder_config, expert_num))
                self.encoder_layers.append(SwitchAggregator(encoder_config))
            else:
                self.encoder_layers.append(SwitchLayerFF(encoder_config))
        self.encoder_layers.append(SwitchFinalLayerNorm(encoder_config))
        # self.encoder = T5Stack(encoder_config, self.token_embedder)

        # decoder layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder_layers.append(DecoderTokenEmbeddings(decoder_config))
        for i in range(decoder_config.num_layers):
            self.decoder_layers.append(SwitchDecoderBlock(decoder_config, bool(i == 0)))
            # moe is attached every two layers
            if i % 2 == 1:
                self.decoder_layers.append(SwitchRouter(decoder_config))
                for expert_num in range(decoder_config.num_experts):
                    self.decoder_layers.append(SwitchExpert(decoder_config, expert_num))
                self.decoder_layers.append(SwitchAggregator(decoder_config))
            else:
                self.decoder_layers.append(SwitchLayerFF(decoder_config))
        self.decoder_layers.append(SwitchFinalLayerNorm(decoder_config))
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
            if isinstance(self.encoder_layers[k], SwitchEncoderBlock):
                encoder_hidden_states, encoder_position_bias = self.encoder_layers[k](
                    encoder_hidden_states,
                    encoder_position_bias,
                    encoder_extended_attention_mask,
                )
            elif isinstance(self.encoder_layers[k], SwitchFinalLayerNorm):
                encoder_hidden_states = self.encoder_layers[k](encoder_hidden_states)
            elif isinstance(self.encoder_layers[k], SwitchLayerFF):
                encoder_hidden_states = self.encoder_layers[k](encoder_hidden_states)
            elif isinstance(self.encoder_layers[k], SwitchRouter):
                routes, route_prob_max = self.encoder_layers[k](encoder_hidden_states)
                k += 1
                expert_outputs = []
                for i in range(self.n_experts):
                    output = self.encoder_layers[k](encoder_hidden_states, routes)
                    expert_outputs.append(output)
                    k += 1
                # expert_outputs = torch.cat(expert_outputs, dim=0)
            elif isinstance(self.encoder_layers[k], SwitchAggregator):
                encoder_hidden_states = self.encoder_layers[k](
                    encoder_hidden_states, expert_outputs, routes, route_prob_max
                )
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
            if isinstance(self.decoder_layers[k], SwitchDecoderBlock):
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
            elif isinstance(self.decoder_layers[k], SwitchFinalLayerNorm):
                decoder_hidden_states = self.decoder_layers[k](decoder_hidden_states)
            elif isinstance(self.decoder_layers[k], SwitchLayerFF):
                decoder_hidden_states = self.decoder_layers[k](decoder_hidden_states)
            elif isinstance(self.decoder_layers[k], SwitchRouter):

                routes, route_prob_max = self.decoder_layers[k](decoder_hidden_states)
                k += 1
                expert_outputs = []
                for i in range(self.n_experts):
                    output = self.decoder_layers[k](decoder_hidden_states, routes)
                    expert_outputs.append(output)
                    k += 1
                # expert_outputs = torch.cat(expert_outputs, dim=0)
            elif isinstance(self.decoder_layers[k], SwitchAggregator):
                decoder_hidden_states = self.decoder_layers[k](
                    decoder_hidden_states, expert_outputs, routes, route_prob_max
                )
            else:
                raise NotImplementedError

            k += 1

        return decoder_hidden_states
