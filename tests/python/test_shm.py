import os
import torch
from dataloader import *
from functools import partial
from transformers import (
    SwitchTransformersForConditionalGeneration,
    AutoTokenizer,
    default_data_collator,
)
import torch
import datasets
import transformers
import numpy as np
from tqdm import tqdm

MODEL_NAME = "switch-base-128"
DATASET_NAME, TASK_NAME = "glue", "mnli_matched"
# DATASET_NAME, TASK_NAME = "super_glue", "boolq"
# DATASET_NAME, TASK_NAME = "squad", None

print(torch.__version__)

print(__file__)

# Load the allocator
new_alloc = torch.cuda.CUDAPluggableAllocator(
    "/mnt/raid0nvme1/xly/swap-engine/tests/python/alloc.so", "my_malloc", "my_free"
)
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
torch.cuda.memory.set_per_process_memory_fraction(0.6)

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
    f"google/{MODEL_NAME}", cache_dir="/mnt/data/xly/.cache", #torch_dtype=torch.float16
)
model.to("cuda:0")

print("model loaded")

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
