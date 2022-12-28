from dataclasses import dataclass, field
import gc
import json
import traceback
from transformers import SwitchTransformersForConditionalGeneration, HfArgumentParser
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseGatedActDense,
)
import torch
import multiprocess as mp
from huggingface_hub import hf_hub_download

from pyutils.ckpt_config import *
import pyutils.ckpt_config as ckpt_config
from pyutils.ckpt_load import export_torchscript_model
from pysrc.transformer.switch.modeling_switch_transformers import *


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    model_path: str = field(metadata={"help": "Path to model cache directory."})

    def __post_init__(self):
        self.model_tag = self.model_name.split("/")[1]
        self.model_repo = "_".join(["model_repo", self.model_tag])
        "%s_encoder_embed" % self.model_tag,


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]


def get_cache_partition(key):
    return hf_hub_download(args.model_name, filename=key, cache_dir=args.model_path)


try:
    # test if is a large checkpoint split into multiple files
    index_path = get_cache_partition("pytorch_model.bin.index.json")
    # load json file index_path
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # create a reverse mapping from file name to weight name list
    file_to_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_to_weights:
            file_to_weights[file_name] = []
        file_to_weights[file_name].append(weight_name)

    pool = mp.Pool(processes=mp.cpu_count())
    partitions = pool.map(get_cache_partition, file_to_weights.keys())
    partitions = dict(zip(file_to_weights.keys(), partitions))
    pool.close()
    pool.join()
except:
    # test if is a single file
    index_path = get_cache_partition("pytorch_model.bin")
    partitions = [index_path]
    partitions = dict(zip(["pytorch_model.bin"], partitions))

    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        args.model_name, cache_dir=args.model_path
    )
    model = model.state_dict()

    weight_map = {}
    for key in list(model.keys()):
        weight_map[key] = "pytorch_model.bin"
    
    file_to_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_to_weights:
            file_to_weights[file_name] = []
        file_to_weights[file_name].append(weight_name)

    del model
    gc.collect()

print("Model partitions: ", partitions.keys(), flush=True)

config = SwitchTransformersConfig.from_pretrained(
    args.model_name, cache_dir=args.model_path
)
# model = SwitchTransformersForConditionalGeneration.from_pretrained(
#     args.model_name, cache_dir=args.model_path, low_memory_mode=True
# )
# model.eval()
# model_state_dict = model.state_dict()
# keys = list(model_state_dict.keys())


def get_key_init(layer_type, layer_idx):
    return "%s.block.%d." % (layer_type, layer_idx)


def ignore_except():
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                ret = f(*args, **kwargs)
                if ret is not None:
                    print("Success in %s" % f.__name__)
                    return ret
                return None
            except:
                pass
                # print("Error in %s" % f.__name__, traceback.print_stack())

        return applicator

    return decorate


def load_missing_keys(keys):
    missing_dict = {}
    for key in keys:
        ckpt_name = weight_map[key]
        ckpt_path = get_cache_partition(ckpt_name)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing_dict[key] = state_dict[key]

    return missing_dict


@ignore_except()
def load_embed(model_state_dict, layer_type):
    state_dict = {"embed_tokens.weight": model_state_dict["shared.weight"]}
    embed_cls = (
        EncoderTokenEmbeddings if layer_type == "encoder" else DecoderTokenEmbeddings
    )
    embed = embed_cls(config)
    embed.load_state_dict(state_dict)
    export_torchscript_model(
        embed,
        args.model_repo,
        "%s_%s_embed" % (args.model_tag, layer_type),
        getattr(ckpt_config, f"get_t5x_{layer_type}_embed_triton_config")(),
    )


loaded_partitions = {}

def load_mlp(layer_type, layer_idx):
    key_init = get_key_init(layer_type, layer_idx)
    padding_str = "layer.1." if layer_type == "encoder" else "layer.2."
    key_init = key_init + padding_str

    if "xxl" in args.model_name:
        wi_0 = weight_map[key_init + "mlp.wi_0.weight"]
        wi_1 = weight_map[key_init + "mlp.wi_1.weight"]
        wo = weight_map[key_init + "mlp.wo.weight"]
        layer_norm = weight_map[key_init + "layer_norm.weight"]
        files = set([wi_0, wi_1, wo, layer_norm])
    else:
        wi = weight_map[key_init + "mlp.wi.weight"]
        wo = weight_map[key_init + "mlp.wo.weight"]
        layer_norm = weight_map[key_init + "layer_norm.weight"]
        files = set([wi, wo, layer_norm])

    for f in files:
        if f not in loaded_partitions:
            print("loading", f, partitions[f], flush=True)
            loaded_partitions[f] = torch.load(partitions[f], map_location="cpu")
    print("loaded_partitions", loaded_partitions.keys(), flush=True)
    print("key_init", key_init, "files", files, flush=True)
    # for f in files:
    #     print("loaded_partitions", f, loaded_partitions[f].keys(), flush=True)

    if "xxl" in args.model_name:
        torch_mlp_layer = {
            "mlp.wi_0.weight": loaded_partitions[wi_0].pop(key_init + "mlp.wi_0.weight"),
            "mlp.wi_1.weight": loaded_partitions[wi_1].pop(key_init + "mlp.wi_1.weight"),
            "mlp.wo.weight": loaded_partitions[wo].pop(key_init + "mlp.wo.weight"),
            "layer_norm.weight": loaded_partitions[layer_norm].pop(key_init + "layer_norm.weight"),
        }
    else:      
        torch_mlp_layer = {
            "mlp.wi.weight": loaded_partitions[wi].pop(key_init + "mlp.wi.weight"),
            "mlp.wo.weight": loaded_partitions[wo].pop(key_init + "mlp.wo.weight"),
            "layer_norm.weight": loaded_partitions[layer_norm].pop(key_init + "layer_norm.weight"),
        }

    mlp_module = SwitchLayerFF(config, is_gated="xxl" in args.model_name)
    print("torch_mlp_layer", torch_mlp_layer.keys(), flush=True)
    print("mlp_module", mlp_module.state_dict().keys(), flush=True)
    mlp_module.load_state_dict(torch_mlp_layer)

    export_torchscript_model(
        mlp_module,
        args.model_repo,
        "%s_%s_ff_%d" % (args.model_tag, layer_type, layer_idx),
        get_ff_triton_config(),
    )


@ignore_except()
def load_final_layer(model_state_dict, layer_type):
    final_layer = SwitchFinalLayerNorm(config)
    torch_final_layer = {
        "layer_norm.weight": model_state_dict.pop(
            "%s.final_layer_norm.weight" % layer_type
        ),
    }
    final_layer.load_state_dict(torch_final_layer)
    export_torchscript_model(
        final_layer,
        args.model_repo,
        "%s_%s_final" % (args.model_tag, layer_type),
        get_ff_triton_config(),
    )


@ignore_except()
def load_router(model_state_dict, layer_type, layer_idx):
    key_init = get_key_init(layer_type, layer_idx)
    padding_str = "layer.1." if layer_type == "encoder" else "layer.2."
    key_init = key_init + padding_str

    router_module = SwitchRouter(config)
    torch_router_layer = {
        "classifier.weight": model_state_dict.pop(
            key_init + "mlp.router.classifier.weight"
        ),
    }
    router_module.load_state_dict(torch_router_layer)
    export_torchscript_model(
        router_module,
        args.model_repo,
        "%s_%s_router_%d" % (args.model_tag, layer_type, layer_idx),
        get_router_triton_config(),
    )


@ignore_except()
def load_lm_head(model_state_dict):
    if "xxl" in args.model_name:
        torch_lm_layer = {
            "lm_head.weight": model_state_dict.pop("decoder.lm_head.weight"),
        }
    else:
        torch_lm_layer = {
            "lm_head.weight": model_state_dict.pop("lm_head.weight"),
        }
    lm_module = SwitchLMPredictionHead(config)
    lm_module.load_state_dict(torch_lm_layer)
    export_torchscript_model(
        lm_module,
        args.model_repo,
        "%s_lm_head" % args.model_tag,
        get_lm_head_triton_config(),
    )


def load_partition(partition):
    p_file, p_path = partition
    model_state_dict = torch.load(p_path, map_location="cpu")
    keys = list(model_state_dict.keys())

    print("Loading model from %s" % p_path, flush=True)
    print("Model keys: %s" % keys, flush=True)

    for i in range(config.num_layers):
        load_embed(model_state_dict, "encoder")
        load_embed(model_state_dict, "decoder")

        # load_block(model_state_dict, "encoder", i)
        # load_block(model_state_dict, "decoder", i)

        if i % 2 == 1:
            load_router(model_state_dict, "encoder", i)
            load_router(model_state_dict, "decoder", i)
            # load_experts(model_state_dict, "encoder", i)
            # load_experts(model_state_dict, "decoder", i)
        # else:
        #     load_mlp(model_state_dict, "encoder", i)
        #     load_mlp(model_state_dict, "decoder", i)

    load_final_layer(model_state_dict, "encoder")
    load_final_layer(model_state_dict, "decoder")
    load_lm_head(model_state_dict)

    # return model_state_dict
    return {}


pool = mp.Pool(processes=5)
model_states = pool.map(load_partition, partitions.items())


def load_experts(layer_type, layer_idx):
    key_init = get_key_init(layer_type, layer_idx)
    padding_str = "layer.1." if layer_type == "encoder" else "layer.2."
    key_init = key_init + padding_str

    expert_cls = (
        SwitchTransformersDenseGatedActDense
        if "xxl" in args.model_name
        else SwitchTransformersDenseActDense
    )
    expert_module = expert_cls(config)
    print("Loading number experts %d" % config.num_experts, flush=True)
    for j in range(config.num_experts):
        expert_key = key_init + "mlp.experts.expert_%d." % j

        print("Loading experts from %s" % expert_key, flush=True)

        if "xxl" in args.model_name:
            wi_0 = weight_map[expert_key + "wi_0.weight"]
            wi_1 = weight_map[expert_key + "wi_1.weight"]
            wo = weight_map[expert_key + "wo.weight"]
            files = set([wi_0, wi_1, wo])
        else:
            wi = weight_map[expert_key + "wi.weight"]
            wo = weight_map[expert_key + "wo.weight"]
            files = set([wi, wo])

        for f in files:
            if f not in loaded_partitions:
                loaded_partitions[f] = torch.load(partitions[f], map_location="cpu")

        print(files, loaded_partitions.keys(), flush=True)

        if "xxl" in args.model_name:
            torch_expert_layers = {
                "wi_0.weight": loaded_partitions[wi_0].pop(expert_key + "wi_0.weight"),
                "wi_1.weight": loaded_partitions[wi_1].pop(expert_key + "wi_1.weight"),
                "wo.weight": loaded_partitions[wo].pop(expert_key + "wo.weight"),
            }
        else:
            torch_expert_layers = {
                "wi.weight": loaded_partitions[wi].pop(expert_key + "wi.weight"),
                "wo.weight": loaded_partitions[wo].pop(expert_key + "wo.weight"),
            }

        expert_module.load_state_dict(torch_expert_layers)
        export_torchscript_model(
            expert_module,
            args.model_repo,
            "%s_%s_expert_%d_%d" % (args.model_tag, layer_type, layer_idx, j),
            get_expert_triton_config(),
        )


def load_block(layer_type, layer_idx):
    key_init = get_key_init(layer_type, layer_idx)
    self_attention_key = key_init + "layer.0."

    self_attention_key_k = self_attention_key + "SelfAttention.k.weight"
    self_attention_key_v = self_attention_key + "SelfAttention.v.weight"
    self_attention_key_q = self_attention_key + "SelfAttention.q.weight"
    self_attention_key_o = self_attention_key + "SelfAttention.o.weight"

    files = set(
        [
            weight_map[self_attention_key_k],
            weight_map[self_attention_key_v],
            weight_map[self_attention_key_q],
            weight_map[self_attention_key_o],
        ]
    )
    for f in files:
        if f not in loaded_partitions:
            loaded_partitions[f] = torch.load(partitions[f], map_location="cpu")

    torch_attention_layer = {
        "attention.SelfAttention.k.weight": loaded_partitions[
            weight_map[self_attention_key_k]
        ].pop(self_attention_key_k),
        "attention.SelfAttention.v.weight": loaded_partitions[
            weight_map[self_attention_key_v]
        ].pop(self_attention_key_v),
        "attention.SelfAttention.q.weight": loaded_partitions[
            weight_map[self_attention_key_q]
        ].pop(self_attention_key_q),
        "attention.SelfAttention.o.weight": loaded_partitions[
            weight_map[self_attention_key_o]
        ].pop(self_attention_key_o),
    }
    if layer_idx == 0:
        relative_attention_key = (
            key_init + "layer.0.SelfAttention.relative_attention_bias.weight"
        )
        file = weight_map[relative_attention_key]
        if file not in loaded_partitions:
            loaded_partitions[file] = torch.load(partitions[file], map_location="cpu")

        torch_attention_layer[
            "attention.SelfAttention.relative_attention_bias.weight"
        ] = loaded_partitions[file].pop(relative_attention_key)

    if layer_type == "decoder":
        cross_attention_key = key_init + "layer.1."

        cross_attention_key_k = cross_attention_key + "EncDecAttention.k.weight"
        cross_attention_key_v = cross_attention_key + "EncDecAttention.v.weight"
        cross_attention_key_q = cross_attention_key + "EncDecAttention.q.weight"
        cross_attention_key_o = cross_attention_key + "EncDecAttention.o.weight"

        files = set(
            [
                weight_map[cross_attention_key_k],
                weight_map[cross_attention_key_v],
                weight_map[cross_attention_key_q],
                weight_map[cross_attention_key_o],
            ]
        )

        for f in files:
            if f not in loaded_partitions:
                loaded_partitions[f] = torch.load(partitions[f], map_location="cpu")

        torch_attention_layer[
            "cross_attention.EncDecAttention.k.weight"
        ] = loaded_partitions[weight_map[cross_attention_key_k]].pop(
            cross_attention_key_k
        )
        torch_attention_layer[
            "cross_attention.EncDecAttention.v.weight"
        ] = loaded_partitions[weight_map[cross_attention_key_v]].pop(
            cross_attention_key_v
        )
        torch_attention_layer[
            "cross_attention.EncDecAttention.q.weight"
        ] = loaded_partitions[weight_map[cross_attention_key_q]].pop(
            cross_attention_key_q
        )
        torch_attention_layer[
            "cross_attention.EncDecAttention.o.weight"
        ] = loaded_partitions[weight_map[cross_attention_key_o]].pop(
            cross_attention_key_o
        )

        cross_attention_layer_norm_key = key_init + "layer.1.layer_norm.weight"
        file = weight_map[cross_attention_layer_norm_key]
        if file not in loaded_partitions:
            loaded_partitions[file] = torch.load(partitions[file], map_location="cpu")

        torch_attention_layer["cross_attention.layer_norm.weight"] = loaded_partitions[
            weight_map[cross_attention_layer_norm_key]
        ].pop(cross_attention_layer_norm_key)

    attention_layer_norm_key = key_init + "layer.0.layer_norm.weight"
    file = weight_map[attention_layer_norm_key]
    if file not in loaded_partitions:
        loaded_partitions[file] = torch.load(partitions[file], map_location="cpu")

    torch_attention_layer["attention.layer_norm.weight"] = loaded_partitions[
        weight_map[attention_layer_norm_key]
    ].pop(attention_layer_norm_key)

    block_module_cls = (
        SwitchEncoderBlock if layer_type == "encoder" else SwitchDecoderBlock
    )
    block_module = block_module_cls(config, bool(layer_idx == 0))
    block_module.load_state_dict(torch_attention_layer)

    export_torchscript_model(
        block_module,
        args.model_repo,
        "%s_%s_block_%d" % (args.model_tag, layer_type, layer_idx),
        getattr(ckpt_config, f"get_t5x_{layer_type}_block_triton_config")(),
    )


# for partition in partitions.items():
#     p_file, p_path = partition
#     model_state_dict = torch.load(p_path, map_location="cpu")
#     print(p_file, model_state_dict.keys())
#     # load_partition(partition)

for i in range(config.num_layers):
    load_block("encoder", i)
    load_block("decoder", i)
    if i % 2 == 1:
        load_experts("encoder", i)
        load_experts("decoder", i)
    else:
        load_mlp("encoder", i)
        load_mlp("decoder", i)

    gc.collect()

# for i in range(config.num_layers):
#     key_init = "encoder.block.%d." % i
#     print(key_init)
#     torch_attention_layer = {
#         key.replace(key_init + "layer.0", "attention"): model_state_dict.pop(key)
#         for key in keys
#         if key.startswith(key_init)
#         and not "mlp" in key
#         and not "layer.1.layer_norm" in key
#     }
#     encoder_block_module = SwitchEncoderBlock(config, bool(i == 0))
#     encoder_block_module.load_state_dict(torch_attention_layer)
#     export_torchscript_model(
#         encoder_block_module,
#         args.model_repo,
#         "%s_encoder_block_%d" % (args.model_tag, i),
#         get_t5x_encoder_block_triton_config(),
#     )

#     key_init = "decoder.block.%d." % i
#     torch_attention_layer = {
#         key.replace(key_init + "layer.0.", ""): model_state_dict.pop(key)
#         for key in keys
#         if key.startswith(key_init)
#         and "layer.0" in key
#         and not "mlp" in key
#         and not "layer.2.layer_norm" in key
#     }

#     decoder_block_module = SwitchDecoderBlock(config, bool(i == 0))
#     decoder_block_module.attention.load_state_dict(torch_attention_layer)

#     torch_attention_layer = {
#         key.replace(key_init + "layer.1.", ""): model_state_dict.pop(key)
#         for key in keys
#         if key.startswith(key_init)
#         and "layer.1" in key
#         and not "mlp" in key
#         and not "layer.2.layer_norm" in key
#     }
#     decoder_block_module.cross_attention.load_state_dict(torch_attention_layer)
#     export_torchscript_model(
#         decoder_block_module,
#         args.model_repo,
#         "%s_decoder_block_%d" % (args.model_tag, i),
#         get_t5x_decoder_block_triton_config(),
#     )

#     if i % 2 == 1:
#         key_init = "encoder.block.%d." % i
#         router_module = SwitchRouter(config)
#         torch_router_layer = {
#             key.replace(key_init + "layer.1.mlp.router.", ""): model_state_dict.pop(key)
#             for key in keys
#             if key.startswith(key_init)
#             and "layer.1" in key
#             and not "expert" in key
#             and not "layer_norm" in key
#         }
#         router_module.load_state_dict(torch_router_layer)
#         export_torchscript_model(
#             router_module,
#             args.model_repo,
#             "%s_encoder_router_%d" % (args.model_tag, i),
#             get_router_triton_config(),
#         )

#         key_init = "decoder.block.%d." % i
#         router_module = SwitchRouter(config)
#         torch_router_layer = {
#             key.replace(key_init + "layer.2.mlp.router.", ""): model_state_dict.pop(key)
#             for key in keys
#             if key.startswith(key_init)
#             and "layer.2" in key
#             and not "expert" in key
#             and not "layer_norm" in key
#         }
#         router_module.load_state_dict(torch_router_layer)
#         export_torchscript_model(
#             router_module,
#             args.model_repo,
#             "%s_decoder_router_%d" % (args.model_tag, i),
#             get_router_triton_config(),
#         )

#         expert_module = SwitchTransformersDenseActDense(config)
#         for j in range(config.num_experts):
#             key_init = "encoder.block.%d." % i
#             # decoder.block.11.layer.2.mlp.experts.expert_0.wo.weight
#             torch_expert_layers = {
#                 key.replace(
#                     key_init + f"layer.1.mlp.experts.expert_{j}.", ""
#                 ): model_state_dict.pop(key)
#                 for key in keys
#                 if key.startswith(key_init)
#                 and "layer.1" in key
#                 and f"expert_{j}" in key
#             }
#             expert_module.load_state_dict(torch_expert_layers)
#             export_torchscript_model(
#                 expert_module,
#                 args.model_repo,
#                 "%s_encoder_expert_%d_%d" % (args.model_tag, i, j),
#                 get_expert_triton_config(),
#             )

#             key_init = "decoder.block.%d." % i
#             torch_expert_layers = {
#                 key.replace(
#                     key_init + f"layer.2.mlp.experts.expert_{j}.", ""
#                 ): model_state_dict.pop(key)
#                 for key in keys
#                 if key.startswith(key_init)
#                 and "layer.2" in key
#                 and f"expert_{j}" in key
#             }
#             expert_module.load_state_dict(torch_expert_layers)
#             export_torchscript_model(
#                 expert_module,
#                 args.model_repo,
#                 "%s_decoder_expert_%d_%d" % (args.model_tag, i, j),
#                 get_expert_triton_config(),
#             )
#     else:
#         key_init = "encoder.block.%d." % i
#         ff_module = SwitchLayerFF(config)
#         torch_mlp_layer = {
#             key.replace(key_init + f"layer.1.", ""): model_state_dict.pop(key)
#             for key in keys
#             if key.startswith(key_init) and "layer.1" in key
#         }
#         ff_module.load_state_dict(torch_mlp_layer)
#         export_torchscript_model(
#             ff_module,
#             args.model_repo,
#             "%s_encoder_ff_%d" % (args.model_tag, i),
#             get_ff_triton_config(),
#         )

#         key_init = "decoder.block.%d." % i
#         ff_module = SwitchLayerFF(config)
#         torch_mlp_layer = {
#             key.replace(key_init + f"layer.2.", ""): model_state_dict.pop(key)
#             for key in keys
#             if key.startswith(key_init) and "layer.2" in key
#         }
#         ff_module.load_state_dict(torch_mlp_layer)
#         export_torchscript_model(
#             ff_module,
#             args.model_repo,
#             "%s_decoder_ff_%d" % (args.model_tag, i),
#             get_ff_triton_config(),
#         )

# final_module = SwitchFinalLayerNorm(config)
# torch_final_layer = {
#     "layer_norm.weight": model_state_dict.pop("encoder.final_layer_norm.weight"),
# }
# final_module.load_state_dict(torch_final_layer)
# export_torchscript_model(
#     final_module,
#     args.model_repo,
#     "%s_encoder_final" % args.model_tag,
#     get_ff_triton_config(),
# )

# torch_final_layer = {
#     "layer_norm.weight": model_state_dict.pop("decoder.final_layer_norm.weight"),
# }
# final_module.load_state_dict(torch_final_layer)
# export_torchscript_model(
#     final_module,
#     args.model_repo,
#     "%s_decoder_final" % args.model_tag,
#     get_ff_triton_config(),
# )

# torch_lm_layer = {
#     "lm_head.weight": model_state_dict.pop("lm_head.weight"),
# }
# lm_module = SwitchLMPredictionHead(config)
# lm_module.load_state_dict(torch_lm_layer)
# export_torchscript_model(
#     lm_module,
#     args.model_repo,
#     "%s_lm_head" % args.model_tag,
#     get_lm_head_triton_config(),
# )
