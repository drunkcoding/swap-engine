# coding=utf-8
""" T5 Switch Transformer model configuration"""
from typing import Dict, Mapping

from transformers.models.t5.configuration_t5 import T5Config
from transformers.utils import logging


class SwitchConfig(T5Config):
    model_type = "switch"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "mlp_dim": "d_expert",
    }
    dense_act_fn = "gelu"

    def __init__(self, **kwargs):
        
        # print(kwargs.keys())
        moe_params = kwargs.get("moe_params", None)

        self.backend_name = kwargs.get("backend_name", None)

        if moe_params is not None:
            self.num_experts = moe_params['num_experts']
            self.d_expert = moe_params["d_expert"]
            self.top_k = moe_params["top_k"]
            self.eval_capacity_factor = moe_params["eval_capacity_factor"]
            self.train_capacity_factor = moe_params["train_capacity_factor"]

        super().__init__(**kwargs)
