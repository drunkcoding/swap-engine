import gc
import torch
import jax
import os

from pyutils.ckpt_config import *

from pysrc.transformer.switch import SwitchModel, SwitchConfig
from pysrc.transformer.switch.modeling_switch import *


def jax2tensor(arr):
    """Converts a JAX array to a PyTorch tensor."""
    return torch.from_numpy(jax.device_get(arr))


def extract_t5x_attention_state_dict(source_state_dict, layer_type, layer_idx):
    jax_attention_layer = source_state_dict[layer_type][f"layers_{layer_idx}"].pop(
        "attention" if layer_type == "encoder" else "self_attention"
    )
    torch_attention_layer = {
        "SelfAttention.q.weight": jax2tensor(jax_attention_layer["query"]["kernel"]),
        "SelfAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
        "SelfAttention.v.weight": jax2tensor(jax_attention_layer["value"]["kernel"]),
        "SelfAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
        "layer_norm.scale": jax2tensor(
            source_state_dict[layer_type][f"layers_{layer_idx}"][
                "pre_attention_layer_norm"
                if layer_type == "encoder"
                else "pre_self_attention_layer_norm"
            ]["scale"]
        ),
    }
    if layer_idx == 0:
        torch_attention_layer[
            "SelfAttention.relative_attention_bias.weight"
        ] = jax2tensor(
            source_state_dict[layer_type]["relpos_bias"]["rel_embedding"]
        ).transpose(
            0, 1
        )

    return torch_attention_layer


def extract_t5x_cross_attention_state_dict(source_state_dict, layer_idx):
    jax_attention_layer = source_state_dict["decoder"][f"layers_{layer_idx}"].pop(
        "encoder_decoder_attention"
    )
    torch_attention_layer = {
        "EncDecAttention.q.weight": jax2tensor(jax_attention_layer["query"]["kernel"]),
        "EncDecAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
        "EncDecAttention.v.weight": jax2tensor(jax_attention_layer["value"]["kernel"]),
        "EncDecAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
        "layer_norm.scale": jax2tensor(
            source_state_dict["decoder"][f"layers_{layer_idx}"][
                "pre_cross_attention_layer_norm"
            ]["scale"]
        ),
    }

    return torch_attention_layer


def extract_t5x_router_state_dict(source_state_dict, layer_type, layer_idx):
    jax_router_layer = source_state_dict[layer_type][f"layers_{layer_idx}"]["mlp"][
        "router"
    ]
    torch_router_layer = {
        "router.weight": jax2tensor(
            jax_router_layer["router_weights"]["w"]["kernel"]
        ).transpose(0, 1),
        "layer_norm.scale": jax2tensor(
            source_state_dict[layer_type][f"layers_{layer_idx}"]["pre_mlp_layer_norm"][
                "scale"
            ]
        ),
    }

    return torch_router_layer


def extract_t5x_expert_state_dict(
    source_state_dict, layer_type, layer_idx, num_experts
):
    jax_mlp_layer = source_state_dict[layer_type][f"layers_{layer_idx}"]["mlp"].pop(
        "expert"
    )
    wi = jax2tensor(jax_mlp_layer["wi"]["kernel"])
    wo = jax2tensor(jax_mlp_layer["wo"]["kernel"])

    torch_expert_layers = []
    for j in range(num_experts):
        torch_mlp_layer = {
            "wi.weight": wi[j].transpose(0, 1),
            "wo.weight": wo[j].transpose(0, 1),
        }
    torch_expert_layers.append(torch_mlp_layer)

    return torch_expert_layers


def extract_t5x_ff_state_dict(source_state_dict, layer_type, layer_idx):
    jax_mlp_layer = source_state_dict[layer_type][f"layers_{layer_idx}"].pop("mlp")
    torch_mlp_layer = {
        "DenseReluDense.wi.weight": jax2tensor(jax_mlp_layer["wi"]["kernel"]).transpose(
            0, 1
        ),
        "DenseReluDense.wo.weight": jax2tensor(jax_mlp_layer["wo"]["kernel"]).transpose(
            0, 1
        ),
        "layer_norm.scale": jax2tensor(
            source_state_dict[layer_type][f"layers_{layer_idx}"]["pre_mlp_layer_norm"][
                "scale"
            ]
        ),
    }
    return torch_mlp_layer


def extract_t5x_final_state_dict(source_state_dict, layer_type):
    torch_final_layer = {
        "layer_norm.scale": jax2tensor(
            source_state_dict[layer_type][f"{layer_type}_norm"]["scale"]
        ),
    }
    return torch_final_layer


def copy_t5x_weights(config: SwitchConfig, model_repo, model_name, source_state_dict):
    token_embedder = {
        "embedding.weight": jax2tensor(
            source_state_dict["token_embedder"].pop("embedding")
        )
    }

    embed_module = EncoderTokenEmbeddings(config)
    embed_module.load_state_dict(token_embedder)
    export_torchscript_model(
        embed_module,
        model_repo,
        "%s_encoder_embed" % model_name,
        get_t5x_encoder_embed_triton_config(),
    )

    embed_module = DecoderTokenEmbeddings(config)
    embed_module.load_state_dict(token_embedder)
    export_torchscript_model(
        embed_module,
        model_repo,
        "%s_decoder_embed" % model_name,
        get_t5x_decoder_embed_triton_config(),
    )

    del source_state_dict["token_embedder"]
    del embed_module

    gc.collect()

    for i in range(config.num_layers):
        torch_attention_layer = extract_t5x_attention_state_dict(
            source_state_dict, "encoder", i
        )
        encoder_block_module = SwitchEncoderBlock(config, bool(i == 0))
        encoder_block_module.attention.load_state_dict(torch_attention_layer)
        export_torchscript_model(
            encoder_block_module,
            model_repo,
            "%s_encoder_block_%d" % (model_name, i),
            get_t5x_encoder_block_triton_config(),
        )

        torch_attention_layer = extract_t5x_attention_state_dict(
            source_state_dict, "decoder", i
        )
        decoder_block_module = SwitchDecoderBlock(config, bool(i == 0))
        decoder_block_module.attention.load_state_dict(torch_attention_layer)
        torch_attention_layer = extract_t5x_cross_attention_state_dict(
            source_state_dict, i
        )
        decoder_block_module.cross_attention.load_state_dict(torch_attention_layer)
        export_torchscript_model(
            decoder_block_module,
            model_repo,
            "%s_decoder_block_%d" % (model_name, i),
            get_t5x_decoder_block_triton_config(),
        )

        # feed forward layers
        if i % 2 == 1:
            # Routers
            router_module = SwitchRouter(config)
            torch_router_layer = extract_t5x_router_state_dict(
                source_state_dict, "encoder", i
            )
            router_module.load_state_dict(torch_router_layer)
            export_torchscript_model(
                router_module,
                model_repo,
                "%s_encoder_router_%d" % (model_name, i),
                get_router_triton_config(),
            )

            torch_router_layer = extract_t5x_router_state_dict(
                source_state_dict, "decoder", i
            )
            router_module.load_state_dict(torch_router_layer)
            export_torchscript_model(
                router_module,
                model_repo,
                "%s_decoder_router_%d" % (model_name, i),
                get_router_triton_config(),
            )

            # Experts
            expert_module = SwitchExpert(config)
            torch_expert_layers = extract_t5x_expert_state_dict(
                source_state_dict, "encoder", i, config.num_experts
            )
            for j, torch_expert_layer in enumerate(torch_expert_layers):
                expert_module.load_state_dict(torch_expert_layer)
                export_torchscript_model(
                    expert_module,
                    model_repo,
                    "%s_encoder_expert_%d_%d" % (model_name, i, j),
                    get_expert_triton_config(),
                )
 
            torch_expert_layers = extract_t5x_expert_state_dict(
                source_state_dict, "decoder", i, config.num_experts
            )
            for j, torch_expert_layer in enumerate(torch_expert_layers):
                expert_module.load_state_dict(torch_expert_layer)
                export_torchscript_model(
                    expert_module,
                    model_repo,
                    "%s_decoder_expert_%d_%d" % (model_name, i, j),
                    get_expert_triton_config(),
                )
        else:
            # normal MLP
            ff_module = SwitchLayerFF(config)
            torch_mlp_layer = extract_t5x_ff_state_dict(source_state_dict, "encoder", i)
            ff_module.load_state_dict(torch_mlp_layer)
            export_torchscript_model(
                ff_module,
                model_repo,
                "%s_encoder_ff_%d" % (model_name, i),
                get_ff_triton_config(),
            )

            torch_mlp_layer = extract_t5x_ff_state_dict(source_state_dict, "decoder", i)
            ff_module.load_state_dict(torch_mlp_layer)
            export_torchscript_model(
                ff_module,
                model_repo,
                "%s_decoder_ff_%d" % (model_name, i),
                get_ff_triton_config(),
            )

    final_module = SwitchFinalLayerNorm(config)
    torch_final_layer = extract_t5x_final_state_dict(source_state_dict, "encoder")
    final_module.load_state_dict(torch_final_layer)
    export_torchscript_model(
        final_module,
        model_repo,
        "%s_encoder_final" % model_name,
        get_ff_triton_config(),
    )

    torch_final_layer = extract_t5x_final_state_dict(source_state_dict, "decoder")
    final_module.load_state_dict(torch_final_layer)
    export_torchscript_model(
        final_module,
        model_repo,
        "%s_decoder_final" % model_name,
        get_ff_triton_config(),
    )
