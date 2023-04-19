import bminf
from dataloader import load_dataset, preprocess_function
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

transformers.logging.set_verbosity_error()


sentence1_key, sentence2_key = "premise", "hypothesis"
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-32", cache_dir="/mnt/data/xly/.cache"
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/switch-base-32", cache_dir="/mnt/data/xly/.cache"
)
process_func = partial(
    preprocess_function,
    tokenizer=tokenizer,
    sentence1_key=sentence1_key,
    sentence2_key=sentence2_key,
)
dataloader = load_dataset("glue", "mnli_matched", process_func)

gpu_total_memory = torch.cuda.get_device_properties(0).total_memory

with torch.cuda.device(0):
    model = bminf.wrapper(
        model, quantization=False, memory_limit=gpu_total_memory * 0.2
    )

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
