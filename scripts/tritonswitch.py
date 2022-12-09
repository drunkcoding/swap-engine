from pysrc.transformer.switch.modeling_switch import *
from pysrc.transformer.switch.configuration_switch import SwitchConfig
import torch
import os


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


CKPT_PATH = "/mnt/xly/checkpoints/t5x-torchscript/moe/base/e128"
MODEL_REPOSITORY = "model_repo_t5x"
MODEL_NAME = "t5x-base-e128"


def save_config(config, model_name):
    model_path = os.path.join(MODEL_REPOSITORY, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "config.pbtxt"), "w") as f:
        f.write(config)


def export_torchscript_model(model, model_name, config):
    model_path = os.path.join(MODEL_REPOSITORY, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    torch.jit.save(torch.jit.script(model), os.path.join(model_path, "0", "model.pt"))
    save_config(config, model_name)


config = SwitchConfig.from_pretrained("config/t5x/base")

model = SwitchModel(config)
state_dict = torch.load(os.path.join(CKPT_PATH, "model.pth"), map_location="cpu")
model.load_state_dict(state_dict)

CONFIG_PLATFORM = """platform: "pytorch_libtorch" """

# encoder_input_ids (batch_size, seq_len)
CONFIG_ENCODER_EMBEDDING = """
    %s
    input [
        {
        name: "encoder_input_ids"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
        },
        {
        name: "encoder_attention_mask"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
        }
    ]
    output [
        {
        name: "encoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "extended_encoder_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        },
        {
        name: "encoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
)

export_torchscript_model(
    model.encoder_layers[0], "%s_encoder_embed" % MODEL_NAME, CONFIG_ENCODER_EMBEDDING
)


CONFIG_DECODER_EMBEDDING = """
    %s
    input [
        {
        name: "decoder_input_ids"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
        },
        {
        name: "decoder_attention_mask"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
        },
        {
        name: "encoder_attention_mask"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
        }
    ]
    output [
        {
        name: "decoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "encoder_extended_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        },
        {
        name: "extended_decoder_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        },
        {
        name: "decoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        }
    ]   
""" % (
    CONFIG_PLATFORM,
    config.d_model,
)

export_torchscript_model(
    model.decoder_layers[0], "%s_decoder_embed" % MODEL_NAME, CONFIG_DECODER_EMBEDDING
)


# ,
#         {
#         name: "routes"
#         data_type: TYPE_INT64
#         dims: [ -1 ]
#         }

CONFIG_EXPERT = """
    %s
    input [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
        }
    ]
    output [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
        }
    ]
""" % CONFIG_PLATFORM

CONFIG_ROUTER = """
    %s
    input [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        }
    ]
    output [
        {
        name: "routes"
        data_type: TYPE_INT64
        dims: [ -1]
        },
        {
        name: "route_prob_max"
        data_type: TYPE_FP32
        dims: [ -1 ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
)

CONFIG_AGGREGATOR = """
    %s
    input [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "expert_output"
        data_type: TYPE_FP32
        dims: [ %d, -1, -1, %d ]
        },
        {
        name: "routes"
        data_type: TYPE_INT64
        dims: [ -1 ]
        },
        {
        name: "route_prob_max"
        data_type: TYPE_FP32
        dims: [ -1 ]
        }
    ]
    output [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
    config.num_experts,
    config.d_model,
    config.d_model,
)

CONFIG_FORWARD = """
    %s
    input [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        }
    ]
    output [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
    config.d_model,
)


CONFIG_ENCODER_BLOCK = """
    %s
    input [
        {
        name: "encoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "encoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, %d, -1, -1 ]
        },
        {
        name: "encoder_extended_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        }
    ]
    output [
        {
        name: "encoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "encoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, %d, -1, -1 ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
    config.num_heads,
    config.d_model,
    config.num_heads,
)

CONFIG_DECODER_BLOCK = """
    %s
    input [
        {
        name: "decoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "encoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "decoder_extended_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        },
        {
        name: "encoder_extended_attention_mask"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        },
        {
        name: "decoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, %d, -1, -1 ]
        },
        {
        name: "encoder_decoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1, -1 ]
        }
    ]
    output [
        {
        name: "decoder_hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, %d ]
        },
        {
        name: "decoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, %d, -1, -1 ]
        },
        {
        name: "encoder_decoder_position_bias"
        data_type: TYPE_FP32
        dims: [ -1, %d, -1, -1 ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
    config.d_model,
    config.d_model,
    config.num_heads,
    config.d_model,
    config.num_heads,
    config.num_heads,
)



CONFIG_FINAL = """ 
    %s
    input [
        {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
        }
    ]
    output [
        {   
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
        }
    ]
""" % (
    CONFIG_PLATFORM,
)

# CONFIG_DECODER_FINAL = """ 
#     %s
#     input [
#         name: "decoder_hidden_states"
#         data_type: TYPE_FP32
#         dims: [ -1, -1, -1 ]
#     ]
#     output [
#         {   
#         name: "decoder_hidden_states"
#         data_type: TYPE_FP32
#         dims: [ -1, -1, -1 ]
#         }
#     ]
# """ % (
#     CONFIG_PLATFORM,
# )


k = 1
for layer_idx in range(config.num_layers):
    export_torchscript_model(
        model.encoder_layers[layer_idx + k],
        "%s_encoder_block_%d" % (MODEL_NAME, layer_idx),
        CONFIG_ENCODER_BLOCK,
    )
    export_torchscript_model(
        model.decoder_layers[layer_idx + k],
        "%s_decoder_block_%d" % (MODEL_NAME, layer_idx),
        CONFIG_DECODER_BLOCK,
    )
    k += 1
    if layer_idx % 2 == 1:
        export_torchscript_model(
            model.encoder_layers[layer_idx + k],
            "%s_encoder_router_%d" % (MODEL_NAME, layer_idx),
            CONFIG_ROUTER,
        )
        export_torchscript_model(
            model.decoder_layers[layer_idx + k],
            "%s_decoder_router_%d" % (MODEL_NAME, layer_idx),
            CONFIG_ROUTER,
        )

        for expert_idx in range(config.num_experts):
            k += 1
            export_torchscript_model(
                model.encoder_layers[layer_idx + k],
                "%s_encoder_expert_%d_%d" % (MODEL_NAME, layer_idx, expert_idx),
                CONFIG_EXPERT,
            )
            export_torchscript_model(
                model.decoder_layers[layer_idx + k],
                "%s_decoder_expert_%d_%d" % (MODEL_NAME, layer_idx, expert_idx),
                CONFIG_EXPERT,
            )
        k += 1

        # k += 1
        # export_torchscript_model(
        #     model.encoder_layers[layer_idx + k],
        #     "%s_encoder_aggregator_%d" % (MODEL_NAME, layer_idx),
        #     CONFIG_AGGREGATOR,
        # )
        # export_torchscript_model(
        #     model.decoder_layers[layer_idx + k],
        #     "%s_decoder_aggregator_%d" % (MODEL_NAME, layer_idx),
        #     CONFIG_AGGREGATOR,
        # )
    else:
        export_torchscript_model(
            model.encoder_layers[layer_idx + k],
            "%s_encoder_ff_%d" % (MODEL_NAME, layer_idx),
            CONFIG_FORWARD,
        )
        export_torchscript_model(
            model.decoder_layers[layer_idx + k],
            "%s_decoder_ff_%d" % (MODEL_NAME, layer_idx),
            CONFIG_FORWARD,
        )

assert layer_idx + k < len(model.encoder_layers) - 1

export_torchscript_model(
    model.encoder_layers[-1],
    "%s_encoder_final" % (MODEL_NAME),
    CONFIG_FINAL,
)
export_torchscript_model(
    model.decoder_layers[-1],
    "%s_decoder_final" % (MODEL_NAME),
    CONFIG_FINAL,
)
