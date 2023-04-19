from typing import Callable, List
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
import torch.nn as nn
import torch

import copy
from .modeling_switch_transformers import (
    T5LayerCrossAttention,
    T5LayerSelfAttention,
    SwitchDenseGatedActDense,
    SwitchDenseActDense,
    get_extended_attention_mask,
    invert_attention_mask,
)
from transformers import (
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersLayerNorm,
    SwitchTransformersSparseMLP,
    SwitchTransformersDenseActDense,
)

import deepspeed
from deepspeed.moe.layer import MoE


class SwitchTransformerDeepSpeed(SwitchTransformersForConditionalGeneration):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.config = config

    def replace_moe_layer(self):
        for i, block in enumerate(self.encoder.block):
            if block.layer[-1].is_sparse:
                state_dict = block.layer[-1].mlp.state_dict()
                block.layer[-1].mlp = MoE(
                    self.config.d_model,
                    SwitchTransformersDenseActDense,
                    self.config.num_experts,
                    capacity_factor=1.0,
                    eval_capacity_factor=2.0,
                    use_rts=False,
                )
                block.layer[-1].mlp.load_state_dict(state_dict)
