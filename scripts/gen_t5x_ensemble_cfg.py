import os
from typing import Union
from transformers import SwitchTransformersConfig

MODEL_NAME = "switch-base-8"
# CKPT_PATH = "/mnt/xly/checkpoints/t5x-torchscript/moe/base/e128"
MODEL_REPOSITORY = "model_repo_switch-base-8"

config = SwitchTransformersConfig.from_pretrained(f"google/{MODEL_NAME}")

def generate_expert_layers(i, num_experts, layer: str):
    layer_cofig = []
    layer_cofig.append(
        """
        {
            model_name: "%s_%s_router_%d"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            output_map {
                key: "routes"
                value: "%s"
            }
            output_map {
                key: "route_prob_max"
                value: "%s"
            }
        } 
        """
        % (
            MODEL_NAME,
            layer,
            i,
            "%s_hidden_states_block_%d" % (layer, i),
            "%s_routes_%d" % (layer, i),
            "%s_route_prob_max_%d" % (layer, i),
        )
    )
    # for k in range(config.num_experts):
    #     layer_cofig.append(
    #         """
    #         {
    #             model_name: "%s_%s_expert_%d_%d"
    #             input_map {
    #                 key: "hidden_states"
    #                 value: "%s"
    #             }
    #             input_map {
    #                 key: "routes"
    #                 value: "%s"
    #             }
    #             output_map {
    #                 key: "hidden_states"
    #                 value: "%s"
    #             }
    #         }
    #         """
    #         % (
    #             MODEL_NAME,
    #             layer,
    #             i,
    #             k,
    #             "%s_hidden_states_block_%d" % (layer, i),
    #             "%s_routes_%d" % (layer, i),
    #             "%s_expert_%d_%d" % (layer, i, k),
    #         )
    #     )
    layer_cofig.append(
        """
        {
            model_name: "%s_%s_preagg_%d"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            input_map {
                key: "routes"
                value: "%s"
            }
            input_map {
                key: "route_prob_max"
                value: "%s"
            }
            output_map {
                key: "hidden_states"
                value: "%s"
            }
        }
        """
        % (
            MODEL_NAME,
            layer,
            i,
            # "".join(
            #     [
            #         """
            #         input_map {
            #             key: "expert_output_%d"
            #             value: "%s"
            #         }
            #         """
            #         % (k, "%s_expert_%d_%d" % (layer, i, k))
            #         for k in range(config.num_experts)
            #     ]
            # ),
            "%s_hidden_states_block_%d" % (layer, i),
            # "%s_hidden_states_agg_%d" % (layer, i),
            "%s_routes_%d" % (layer, i),
            "%s_route_prob_max_%d" % (layer, i),
            "%s_hidden_states_%d" % (layer, i + 1),
        )
    )


    return layer_cofig


def generate_ff_layer(i, layer: str):
    layer_cofig = []
    layer_cofig.append(
        """
        {
            model_name: "%s_%s_ff_%d"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            output_map {
                key: "hidden_states"
                value: "%s"
            }
        } 
        """
        % (
            MODEL_NAME,
            layer,
            i,
            "%s_hidden_states_block_%d" % (layer, i),
            "%s_hidden_states_%d" % (layer, i + 1),
        )
    )
    return layer_cofig


def generate_final_layer(layer: str):
    layer_cofig = []
    layer_cofig.append(
        """
        {
            model_name: "%s_%s_final"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            output_map {
                key: "hidden_states"
                value: "%s"
            }
        } 
        """
        % (
            MODEL_NAME,
            layer,
            "%s_hidden_states_%d" % (layer, config.num_layers),
            "%s_hidden_states" % layer,
        )
    )
    return layer_cofig

def generate_lm_head():
    layer_cofig = []
    layer_cofig.append(
        """
        {
            model_name: "%s_lm_head"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            output_map {
                key: "hidden_states"
                value: "%s"
            }
        } 
        """
        % (
            MODEL_NAME,
            "decoder_hidden_states",
            "logits",
        )
    )
    return layer_cofig

ensemble_steps = []
ensemble_steps.append(
    """
    {
        model_name: "%s_encoder_embed"
        input_map {
            key: "encoder_input_ids"
            value: "%s"
        }
        input_map {
            key: "encoder_attention_mask"
            value: "%s"
        }
        output_map {
            key: "encoder_hidden_states"
            value: "%s"
        }
        output_map {
            key: "extended_encoder_attention_mask"
            value: "%s"
        }
        output_map {
            key: "encoder_position_bias"
            value: "%s"
        }
    }
"""
    % (
        MODEL_NAME,
        "encoder_input_ids",
        "encoder_attention_mask",
        "encoder_hidden_states_0",
        "extended_encoder_attention_mask",
        "encoder_position_bias_0",
    )
)

for i in range(config.num_layers):
    ensemble_steps.append(
        """
    {
        model_name: "%s_encoder_block_%d"
        input_map {
            key: "encoder_hidden_states"
            value: "%s"
        }
        input_map {
            key: "encoder_position_bias"
            value: "%s"
        }
        input_map {
            key: "encoder_extended_attention_mask"
            value: "%s"
        }
        output_map {
            key: "encoder_hidden_states"
            value: "%s"
        }
        output_map {
            key: "encoder_position_bias"
            value: "%s"
        }
    }
"""
        % (
            MODEL_NAME,
            i,
            "encoder_hidden_states_%d" % i,
            "encoder_position_bias_%d" % i,
            "extended_encoder_attention_mask",
            "encoder_hidden_states_block_%d" % i,
            "encoder_position_bias_%d" % (i + 1),
        )
    )

    if i % 2 == 1:
        ensemble_steps += generate_expert_layers(i, config.num_experts, "encoder")
    else:
        ensemble_steps += generate_ff_layer(i, "encoder")


ensemble_steps += generate_final_layer("encoder")

# ==================================================================================================
# Decoder
# ==================================================================================================
#

ensemble_steps.append(
    """
        {
            model_name: "%s_decoder_embed"
            input_map {
                key: "decoder_input_ids"
                value: "%s"
            }
            input_map {
                key: "decoder_attention_mask"
                value: "%s"
            }
            input_map {
                key: "encoder_attention_mask"
                value: "%s"
            }
            output_map {
                key: "decoder_hidden_states"
                value: "%s"
            }
            output_map {
                key: "encoder_extended_attention_mask"
                value: "%s"
            }
            output_map {
                key: "extended_decoder_attention_mask"
                value: "%s"
            }
            output_map {
                key: "decoder_position_bias"
                value: "%s"
            }
        }
    """
    % (
        MODEL_NAME,
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_attention_mask",
        "decoder_hidden_states_0",
        "encoder_extended_attention_mask",
        "extended_decoder_attention_mask",
        "decoder_position_bias_0",
    )
)

for i in range(config.num_layers):
    ensemble_steps.append(
        """
            {
                model_name: "%s_decoder_block_%d"
                input_map {
                    key: "decoder_hidden_states"
                    value: "%s"
                }
                input_map {
                    key: "encoder_hidden_states"
                    value: "%s"
                }
                input_map {
                    key: "decoder_position_bias"
                    value: "%s"
                }
                input_map {
                    key: "encoder_decoder_position_bias"
                    value: "%s"
                }
                input_map {
                    key: "encoder_extended_attention_mask"
                    value: "%s"
                }
                input_map {
                    key: "decoder_extended_attention_mask"
                    value: "%s"
                }
                output_map {
                    key: "decoder_hidden_states"
                    value: "%s"
                }
                output_map {
                    key: "decoder_position_bias"
                    value: "%s"
                }
                output_map {
                    key: "encoder_decoder_position_bias"
                    value: "%s"
                }
            }
        """
        % (
            MODEL_NAME,
            i,
            "decoder_hidden_states_%d" % i,
            "encoder_hidden_states",
            "decoder_position_bias_%d" % i,
            "encoder_position_bias_%d" % (config.num_layers + i),
            "encoder_extended_attention_mask",
            "extended_decoder_attention_mask",
            "decoder_hidden_states_block_%d" % i,
            "decoder_position_bias_%d" % (i + 1),
            "encoder_position_bias_%d" % (config.num_layers + i + 1),
        )
    )

    if i % 2 == 1:
        ensemble_steps += generate_expert_layers(i, config.num_experts, "decoder")
    else:
        ensemble_steps += generate_ff_layer(i, "decoder")


ensemble_steps += generate_final_layer("decoder")

ensemble_steps += generate_lm_head()


CONFIG_ENSEMBLE = """
name: "%s-ensemble"
platform: "ensemble"
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
    },
    {
        name: "decoder_input_ids"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
    },
    {
        name: "decoder_attention_mask"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
    }
]
output [
    {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
    },
    {
    name: "encoder_position_bias_%d"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    },
    {
    name: "decoder_position_bias_%d"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
    }
]
ensemble_scheduling {
    step [
        %s
    ]
}
""" % (
    MODEL_NAME,
    config.num_layers * 2,
    config.num_layers,
    ",\n".join(ensemble_steps),
)

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_config(config, model_name):
    model_path = os.path.join(MODEL_REPOSITORY, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "config.pbtxt"), "w") as f:
        f.write(config)


save_config(CONFIG_ENSEMBLE, f"{MODEL_NAME}-ensemble")
