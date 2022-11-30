import os
import sys
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
from torch import nn
import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vmoe.nn.vit_moe import VisionTransformerMoe

# CKPT_DIR = "/mnt/xly/checkpoints/vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_medium"
CKPT_DIR = "/mnt/xly/checkpoints/t5x/moe/switch_classic/base/e8/checkpoint_500100/"

DEFAULT_TEST_CONFIG = {
    'num_classes': 1000,
    'patch_size': (4, 4),
    'hidden_size': 512,
    'encoder': {
        'num_layers': 8,
        'mlp_dim': 2048,
        'num_heads': 8,
        'moe': {
            'layers': (1,),
            'num_experts': 8,
            'group_size': 2,
            'router': {
                'num_selected_experts': 1,
                'noise_std': 1e-3,
                'importance_loss_weight': 0.02,
                'load_loss_weight': 0.02,
                'dispatcher': {
                    'name': 'einsum',
                    'capacity': 2,
                    'batch_priority': False,
                    'bfloat16': False,
                }
            },
        },
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
    },
    'classifier': 'gap',
    'representation_size': None,
}


# model = VisionTransformerMoe(**DEFAULT_TEST_CONFIG)

restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None, prefix="ckpt")
shard_count = restored_state["shard_count"]
index = restored_state["index"]
print(index.keys())

