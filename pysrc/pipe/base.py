import gc
from typing import Tuple
import numpy as np
import torch

def get_embed_dropout(config):
    if hasattr(config, 'embd_pdrop'):
        return config.embd_pdrop
    if hasattr(config, 'embed_dropout'):
        return config.embed_dropout

def get_embed_dim(config):
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "n_embd"):
        return config.n_embd
    if hasattr(config, "d_model"):
        return config.d_model

def get_num_layers(config):
    if hasattr(config, "num_layers"):
        return config.num_layers
    if hasattr(config, "n_layer"):
        return config.n_layer
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def format_inputs(args, ds):
    if not ds: return args
    return tuple([None if torch.sum(t) == 127873 else t for t in args])


def format_outputs(args, ds):
    if not ds: return args
    shape = args[0].shape
    device = args[0].device
    return tuple([torch.Tensor([127873]).to(device) if t is None else t for t in args])


class PipeMethods:
    def convert(self, device):
        for idx in range(*self.exec_map):
            self.layers[idx] = self.layers[idx].to(device)

        # use placeholder to save more memory
        for i in range(len(self.layers)):
            if i < self.exec_map[0] or i >= self.exec_map[1]:
                self.layers[i] = torch.nn.Module()

        torch.cuda.empty_cache()
        gc.collect()
        self.device = device

    def convert_layer_specs(self, device):
        l, h = self.exec_map
        for idx, layer_cls in enumerate(self.layer_specs):
            if idx >= l and idx < h:
                self.layers[idx] = layer_cls.build().to(device)

        torch.cuda.empty_cache()
        gc.collect()
        self.device = device

    def partition_by_parameter(self, stage, parts, synthetic=False):
        l_params = self.total_params / parts * stage
        h_params = self.total_params / parts * (stage + 1) if stage != parts - 1 else self.total_params

        print("partition_by_parameter", self.total_params, l_params, h_params, flush=True)

        # if synthetic:
        #     layer_params = np.cumsum([1] * self.total_params)
        # else:
        #     layer_params = [sum([np.prod(p.size()) for p in self.layers[idx].parameters()]) for idx in range(len(self.layers))]
        layer_params = np.cumsum(self.layer_param)
        responsible_layers = np.argwhere((layer_params > l_params) & (layer_params <= h_params)).flatten()

        print("responsible_layers", layer_params, responsible_layers, flush=True)

        self.exec_map = (responsible_layers[0], responsible_layers[-1]+1)

# def get_decoder_start_token_id(
#         decoder_start_token_id: int = None, bos_token_id: int = None
# ) -> int:
#     decoder_start_token_id = (
#         decoder_start_token_id
#         if decoder_start_token_id is not None
#         else self.config.decoder_start_token_id
#     )
#     bos_token_id = (
#         bos_token_id  # if bos_token_id is not None else self.config.bos_token_id
#     )

#     if decoder_start_token_id is not None:
#         return decoder_start_token_id
#     elif (
#             hasattr(self.config, "decoder")
#             and hasattr(self.config.decoder, "decoder_start_token_id")
#             and self.config.decoder.decoder_start_token_id is not None
#     ):
#         return self.config.decoder.decoder_start_token_id
#     elif bos_token_id is not None:
#         return bos_token_id
#     elif (
#             hasattr(self.config, "decoder")
#             and hasattr(self.config.decoder, "bos_token_id")
#             and self.config.decoder.bos_token_id is not None
#     ):
#         return self.config.decoder.bos_token_id
#     raise ValueError(
#         "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
#     )

def prepare_decoder_input_ids_for_generation(
        input_ids: torch.LongTensor,
        decoder_start_token_id: int,
        bos_token_id: int = None,
) -> torch.LongTensor:
    # decoder_start_token_id = get_decoder_start_token_id(
    #     decoder_start_token_id, bos_token_id
    # )
    decoder_input_ids = (
            torch.ones(
                (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device
            )
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
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    # encoder_extended_attention_mask = encoder_extended_attention_mask.to(
    #     dtype=self.dtype
    # )  # fp16 compatibility
    #
    # if self.dtype == torch.float16:
    #     encoder_extended_attention_mask = (
    #                                               1.0 - encoder_extended_attention_mask
    #                                       ) * -1e4
    # elif self.dtype == torch.float32:
    encoder_extended_attention_mask = (
                                              1.0 - encoder_extended_attention_mask
                                      ) * -1e9
    # else:
    #     raise ValueError(
    #         f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
    #     )

    return encoder_extended_attention_mask


def get_extended_attention_mask(
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        is_decoder=False,
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
                    axis=-1,
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

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(
    #     dtype=self.dtype
    # )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
