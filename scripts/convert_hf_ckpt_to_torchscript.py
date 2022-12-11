from dataclasses import dataclass, field
import gc
from transformers import SwitchTransformersForConditionalGeneration, HfArgumentParser
import torch

from pyutils.ckpt_config import *
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

model = SwitchTransformersForConditionalGeneration.from_pretrained(
    args.model_name, cache_dir=args.model_path
)

model_state_dict = model.state_dict()

keys = list(model_state_dict.keys())

print(keys)

encoder_embed = EncoderTokenEmbeddings(model.config)
encoder_embed.load_state_dict(
    {"embed_tokens.weight": model_state_dict.pop("encoder.embed_tokens.weight")}
)
export_torchscript_model(
    encoder_embed,
    args.model_repo,
    "%s_encoder_embed" % args.model_tag,
    get_t5x_encoder_embed_triton_config(),
)
del encoder_embed
gc.collect()

decoder_embed = DecoderTokenEmbeddings(model.config)
decoder_embed.load_state_dict(
    {"embed_tokens.weight": model_state_dict.pop("decoder.embed_tokens.weight")}
)
export_torchscript_model(
    decoder_embed,
    args.model_repo,
    "%s_decoder_embed" % args.model_tag,
    get_t5x_decoder_embed_triton_config(),
)
del decoder_embed
gc.collect()

def get_key_init(layer_type, layer_idx):
    return "%s.block.%d." % (layer_type, layer_idx)

for i in range(model.config.num_layers):
    key_init = "encoder.block.%d." % i
    print(key_init)
    torch_attention_layer = {
        key.replace(key_init + "layer.0", "attention"): model_state_dict.pop(key)
        for key in keys
        if key.startswith(key_init) and not "mlp" in key and not "layer.1.layer_norm" in key
    }
    encoder_block_module = SwitchEncoderBlock(model.config, bool(i == 0))
    encoder_block_module.load_state_dict(torch_attention_layer)
    export_torchscript_model(
        encoder_block_module,
        args.model_repo,
        "%s_encoder_block_%d" % (args.model_tag, i),
        get_t5x_encoder_block_triton_config(),
    )

    key_init = "decoder.block.%d." % i
    torch_attention_layer = {
        key.replace(key_init + "layer.0.", ""): model_state_dict.pop(key)
        for key in keys
        if key.startswith(key_init) and "layer.0" in key and not "mlp" in key and not "layer.2.layer_norm" in key
    }

    decoder_block_module = SwitchDecoderBlock(model.config, bool(i == 0))
    decoder_block_module.attention.load_state_dict(torch_attention_layer)

    torch_attention_layer = {
        key.replace(key_init + "layer.1.", ""): model_state_dict.pop(key)
        for key in keys
        if key.startswith(key_init) and "layer.1" in key and not "mlp" in key and not "layer.2.layer_norm" in key
    }
    decoder_block_module.cross_attention.load_state_dict(torch_attention_layer)
    export_torchscript_model(
        decoder_block_module,
        args.model_repo,
        "%s_decoder_block_%d" % (args.model_tag, i),
        get_t5x_decoder_block_triton_config(),
    )

    if i % 2 == 1:
        key_init = "encoder.block.%d." % i
        router_module = SwitchRouter(model.config)
        torch_router_layer = {
            key.replace(key_init + "layer.1.mlp.router.", ""): model_state_dict.pop(key)
            for key in keys
            if key.startswith(key_init) and "layer.1" in key and not "expert" in key and not "layer_norm" in key
        }
        router_module.load_state_dict(torch_router_layer)
        export_torchscript_model(
            router_module,
            args.model_repo,
            "%s_encoder_router_%d" % (args.model_tag, i),
            get_router_triton_config(),
        )

        key_init = "decoder.block.%d." % i
        router_module = SwitchRouter(model.config)
        torch_router_layer = {
            key.replace(key_init + "layer.2.mlp.router.", ""): model_state_dict.pop(key)
            for key in keys
            if key.startswith(key_init) and "layer.2" in key and not "expert" in key and not "layer_norm" in key
        }
        router_module.load_state_dict(torch_router_layer)
        export_torchscript_model(
            router_module,
            args.model_repo,
            "%s_decoder_router_%d" % (args.model_tag, i),
            get_router_triton_config(),
        )

        expert_module = SwitchTransformersDenseActDense(model.config)
        for j in range(model.config.num_experts):
            key_init = "encoder.block.%d." % i
            # decoder.block.11.layer.2.mlp.experts.expert_0.wo.weight
            torch_expert_layers = {
                key.replace(key_init + f"layer.1.mlp.experts.expert_{j}.", ""): model_state_dict.pop(key)
                for key in keys
                if key.startswith(key_init) and "layer.1" in key and f"expert_{j}" in key
            }
            expert_module.load_state_dict(torch_expert_layers)
            export_torchscript_model(
                expert_module,
                args.model_repo,
                "%s_encoder_expert_%d_%d" % (args.model_tag, i, j),
                get_expert_triton_config(),
            )

            key_init = "decoder.block.%d." % i
            torch_expert_layers = {
                key.replace(key_init + f"layer.2.mlp.experts.expert_{j}.", ""): model_state_dict.pop(key)
                for key in keys
                if key.startswith(key_init) and "layer.2" in key and f"expert_{j}" in key
            }
            expert_module.load_state_dict(torch_expert_layers)
            export_torchscript_model(
                expert_module,
                args.model_repo,
                "%s_decoder_expert_%d_%d" % (args.model_tag, i, j),
                get_expert_triton_config(),
            )
    else:
        key_init = "encoder.block.%d." % i
        ff_module = SwitchLayerFF(model.config)
        torch_mlp_layer = {
            key.replace(key_init + f"layer.1.", ""): model_state_dict.pop(key)
            for key in keys
            if key.startswith(key_init) and "layer.1" in key
        }
        ff_module.load_state_dict(torch_mlp_layer)
        export_torchscript_model(
            ff_module,
            args.model_repo,
            "%s_encoder_ff_%d" % (args.model_tag, i),
            get_ff_triton_config(),
        )

        key_init = "decoder.block.%d." % i
        ff_module = SwitchLayerFF(model.config)
        torch_mlp_layer = {
            key.replace(key_init + f"layer.2.", ""): model_state_dict.pop(key)
            for key in keys
            if key.startswith(key_init) and "layer.2" in key
        }
        ff_module.load_state_dict(torch_mlp_layer)
        export_torchscript_model(
            ff_module,
            args.model_repo,
            "%s_decoder_ff_%d" % (args.model_tag, i),
            get_ff_triton_config(),
        )

final_module = SwitchFinalLayerNorm(model.config)
torch_final_layer = {
    "layer_norm.weight": model_state_dict.pop("encoder.final_layer_norm.weight"),
}
final_module.load_state_dict(torch_final_layer)
export_torchscript_model(
    final_module,
    args.model_repo,
    "%s_encoder_final" % args.model_tag,
    get_ff_triton_config(),
)

torch_final_layer = {
    "layer_norm.weight": model_state_dict.pop("decoder.final_layer_norm.weight"),
}
final_module.load_state_dict(torch_final_layer)
export_torchscript_model(
    final_module,
    args.model_repo,
    "%s_decoder_final" % args.model_tag,
    get_ff_triton_config(),
)

torch_lm_layer = {
    "lm_head.weight": model_state_dict.pop("lm_head.weight"),
}
lm_module = SwitchLMPredictionHead(model.config)
lm_module.load_state_dict(torch_lm_layer)
export_torchscript_model(
    lm_module,
    args.model_repo,
    "%s_lm_head" % args.model_tag,
    get_lm_head_triton_config(),
)
