import os
import torch


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_triton_config(config, model_repo, model_name):
    model_path = os.path.join(model_repo, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "config.pbtxt"), "w") as f:
        f.write(config)


def export_torchscript_model(model, model_repo, model_name, config):
    model_path = os.path.join(model_repo, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    torch.jit.save(torch.jit.script(model), os.path.join(model_path, "0", "model.pt"))
    save_triton_config(config, model_repo, model_name)


CONFIG_PLATFORM = """platform: "pytorch_libtorch" """


def get_t5x_encoder_embed_triton_config():
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
    dims: [ -1, -1, -1 ]
    },
    {
    name: "encoder_position_bias"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    }
]
""" % (
        CONFIG_PLATFORM,
    )

    return CONFIG_ENCODER_EMBEDDING


def get_t5x_decoder_embed_triton_config():
    CONFIG_DECODER_EMBEDDING = """
platform: "pytorch_libtorch"
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
    dims: [ -1, -1, -1 ]
    },
    {
    name: "decoder_position_bias"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    },
    {
    name: "encoder_decoder_position_bias"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    }
]   
"""

    return CONFIG_DECODER_EMBEDDING


def get_expert_triton_config():
    CONFIG_EXPERT = """
platform: "pytorch_libtorch"
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
"""

    return CONFIG_EXPERT


def get_router_triton_config():
    CONFIG_ROUTER = """
platform: "pytorch_libtorch"
input [
    {
    name: "hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    }
]
output [
    {
    name: "forwarded_states"
    data_type: TYPE_FP32
    dims: [ -1 , -1, -1 ]
    },
    {
    name: "routes"
    data_type: TYPE_INT64
    dims: [ -1 , -1, -1 ]
    },
    {
    name: "route_prob_max"
    data_type: TYPE_FP32
    dims: [ -1 , -1, -1 ]
    }
]
    """

    return CONFIG_ROUTER

def get_lm_head_triton_config():
    CONFIG_LM_HEAD = """
platform: "pytorch_libtorch"
input [
    {
    name: "hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    }
]
output [
    {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    }
]
"""

    return CONFIG_LM_HEAD

# for both ff and  layer norm
def get_ff_triton_config():
    CONFIG_FORWARD = """
platform: "pytorch_libtorch"
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
"""
    return CONFIG_FORWARD


def get_t5x_encoder_block_triton_config():
    CONFIG_ENCODER_BLOCK = """
platform: "pytorch_libtorch"
input [
    {
    name: "encoder_hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    },
    {
    name: "encoder_position_bias"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    }
]
output [
    {
    name: "encoder_hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    }
]
"""

    return CONFIG_ENCODER_BLOCK


def get_t5x_decoder_block_triton_config():
    CONFIG_DECODER_BLOCK = """
platform: "pytorch_libtorch"
input [
    {
    name: "decoder_hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    },
    {
    name: "encoder_hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    },
    {
    name: "decoder_position_bias"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
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
    dims: [ -1, -1, -1 ]
    }
]
"""

    return CONFIG_DECODER_BLOCK
