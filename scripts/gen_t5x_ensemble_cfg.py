from dataclasses import dataclass, field
import os
from typing import Union
from transformers import SwitchTransformersConfig, HfArgumentParser


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})

    def __post_init__(self):
        self.model_repo = "_".join(["model_repo", self.model_name])


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

config = SwitchTransformersConfig.from_pretrained(f"google/{args.model_name}")


def generate_expert_layers(i, num_experts, layer: str):
    layer_cofig = []
    # layer_cofig.append(
    #     """
    #     {
    #         model_name: "%s_%s_router_%d"
    #         input_map {
    #             key: "hidden_states"
    #             value: "%s"
    #         }
    #         output_map {
    #             key: "forwarded_states"
    #             value: "%s"
    #         }
    #         output_map {
    #             key: "routes"
    #             value: "%s"
    #         }
    #         output_map {
    #             key: "route_prob_max"
    #             value: "%s"
    #         }
    #     } 
    #     """
    #     % (
    #         args.model_name,
    #         layer,
    #         i,
    #         "%s_hidden_states_block_%d" % (layer, i),
    #         "%s_forwarded_states_block_%d" % (layer, i),
    #         "%s_routes_%d" % (layer, i),
    #         "%s_route_prob_max_%d" % (layer, i),
    #     )
    # )
    layer_cofig.append(
        """
        {
            model_name: "%s_%s_preagg_%d"
            input_map {
                key: "hidden_states"
                value: "%s"
            }
            input_map {
                key: "forwarded_states"
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
            args.model_name,
            layer,
            i,
            "%s_hidden_states_block_%d" % (layer, i),
            "%s_forwarded_states_block_%d" % (layer, i),
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
            args.model_name,
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
            args.model_name,
            layer,
            "%s_hidden_states_%d"
            % (
                layer,
                config.num_sparse_decoder_layers * 2
                if layer == "decoder"
                else config.num_sparse_encoder_layers * 2,
            ),
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
                key: "logits"
                value: "%s"
            }
        } 
        """
        % (
            args.model_name,
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
            key: "encoder_position_bias"
            value: "%s"
        }
    }
"""
    % (
        args.model_name,
        "encoder_input_ids",
        "encoder_attention_mask",
        "encoder_hidden_states_0",
        "encoder_position_bias",
    )
)

for i in range(config.num_sparse_encoder_layers * 2):
    if i % 2 == 1:
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
    output_map {
        key: "encoder_hidden_states"
        value: "%s"
    }
    output_map {
        key: "forwarded_states"
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
            args.model_name,
            i,
            "encoder_hidden_states_%d" % i,
            "encoder_position_bias",
            "encoder_hidden_states_block_%d" % i,
            "encoder_forwarded_states_block_%d" % i,
            "encoder_routes_%d" % i,
            "encoder_route_prob_max_%d" % i,
        )
    )
        ensemble_steps += generate_expert_layers(i, config.num_experts, "encoder")
    else:
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
    output_map {
        key: "encoder_hidden_states"
        value: "%s"
    }
}
"""
        % (
            args.model_name,
            i,
            "encoder_hidden_states_%d" % i,
            "encoder_position_bias",
            "encoder_hidden_states_%d" % (i+1),
        )
    )

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
        args.model_name,
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_attention_mask",
        "decoder_hidden_states_0",
        "decoder_position_bias",
        "encoder_decoder_position_bias",
    )
)

for i in range(config.num_sparse_decoder_layers * 2):
    

    if i % 2 == 1:
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
                    output_map {
                        key: "decoder_hidden_states"
                        value: "%s"
                    }
                    output_map {
                        key: "forwarded_states"
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
                args.model_name,
                i,
                "decoder_hidden_states_%d" % i,
                "encoder_hidden_states",
                "decoder_position_bias",
                "encoder_decoder_position_bias",
                "decoder_hidden_states_block_%d" % i,
                "decoder_forwarded_states_block_%d" % i,
                "decoder_routes_%d" % i,
                "decoder_route_prob_max_%d" % i,
            )
        )
        ensemble_steps += generate_expert_layers(i, config.num_experts, "decoder")
    else:
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
                output_map {
                    key: "decoder_hidden_states"
                    value: "%s"
                }
            }
        """
        % (
            args.model_name,
            i,
            "decoder_hidden_states_%d" % i,
            "encoder_hidden_states",
            "decoder_position_bias",
            "encoder_decoder_position_bias",
            "decoder_hidden_states_%d" % (i+1),
        )
    )


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
    }
]
ensemble_scheduling {
    step [
        %s
    ]
}
""" % (
    args.model_name,
    ",\n".join(ensemble_steps),
)


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_config(config, model_name):
    model_path = os.path.join(args.model_repo, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "config.pbtxt"), "w") as f:
        f.write(config)


save_config(CONFIG_ENSEMBLE, f"{args.model_name}-ensemble")
