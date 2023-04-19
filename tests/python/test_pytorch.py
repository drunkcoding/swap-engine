from dataclasses import dataclass
import torch
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   enable_wrap,
   wrap,
)
import torch.nn as nn
from typing import Dict
import os
from dataloader import *
from functools import partial
from transformers import (
    SwitchTransformersForConditionalGeneration,
    AutoTokenizer,
    default_data_collator,
    HfArgumentParser,
)
import datasets
import transformers
import numpy as np
from tqdm import tqdm
import argparse

# @dataclass
# class ModelArguments:
    

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=0)
args = parser.parse_args()

MODEL_NAME = "switch-base-128"
DATASET_NAME, TASK_NAME = "glue", "mnli_matched"
# DATASET_NAME, TASK_NAME = "super_glue", "boolq"
# DATASET_NAME, TASK_NAME = "squad", None


tokenizer = AutoTokenizer.from_pretrained(
    f"google/switch-base-8", cache_dir="/mnt/data/xly/.cache"
)

sentence1_key, sentence2_key = sentence_keys(DATASET_NAME, TASK_NAME)
process_func = partial(
    preprocess_function,
    tokenizer=tokenizer,
    sentence1_key=sentence1_key,
    sentence2_key=sentence2_key,
)
dataloader = load_dataset(DATASET_NAME, TASK_NAME, process_func)

print("dataloader loaded")



model = SwitchTransformersForConditionalGeneration.from_pretrained(
    f"google/{MODEL_NAME}", cache_dir="/mnt/data/xly/.cache"
)

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12355"

world_size = os.environ["WORLD_SIZE"]
local_rank = os.environ["LOCAL_RANK"]
# use torch FSDP to wrap the model
torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=int(world_size),
    rank=int(local_rank),
)
wrapper_kwargs = dict(
    cpu_offload=CPUOffload(offload_params=True),
    process_group=torch.distributed.new_group(),
)
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
   model = wrap(model)

for batch in tqdm(dataloader):
    # print(batch)
    input_ids = np.asarray(batch["input_ids"]).astype(np.int32)
    attention_mask = np.asarray(batch["attention_mask"]).astype(np.int32)

    decoder_length = 3
    idx = np.random.choice(np.arange(20), decoder_length, replace=False)
    idx = np.sort(idx)
    decoder_input_ids = input_ids.copy()
    decoder_input_ids = decoder_input_ids[:, :32]
    decoder_input_ids[:, idx] = np.array([x for x in range(decoder_length)]) + 32000
    # insert 0 at begining of decoder input ids
    decoder_input_ids = np.insert(decoder_input_ids, 0, 0, axis=1)
    decoder_attention_mask = np.ones_like(decoder_input_ids).astype(np.int32)

    # to cuda
    input_ids = torch.tensor(input_ids).cuda()
    attention_mask = torch.tensor(attention_mask).cuda()
    decoder_input_ids = torch.tensor(decoder_input_ids).cuda()
    decoder_attention_mask = torch.tensor(decoder_attention_mask).cuda()

    model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
