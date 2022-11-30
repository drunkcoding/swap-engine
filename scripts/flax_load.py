import os
import sys
from transformers.modeling_flax_pytorch_utils import (
    load_flax_checkpoint_in_pytorch_model,
)
from torch import nn
import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CKPT_DIR = "/mnt/xly/checkpoints/vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_medium"
CKPT_DIR = "/mnt/xly/checkpoints/t5x/moe/switch_classic/base/e8/checkpoint_500100/"
restored_state = checkpoints.restore_checkpoint(
    ckpt_dir=CKPT_DIR, target=None, prefix=""
)
print(restored_state)
