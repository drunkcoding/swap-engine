import asyncio
from dataclasses import dataclass, field
from email import parser
import json
import os
import time
import uuid
import grpc
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from transformers import T5Tokenizer, default_data_collator
import datasets
import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm

@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    num_processes: int = field(metadata={"help": "Number of processes to use"})
    batch_size: int = field(default=8, metadata={"help": "Batch size to use"})

parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

torch.set_printoptions(profile="full")
sentence1_key, sentence2_key = "premise", "hypothesis"

def format_triton_input(input: np.ndarray, name: str):
    triton_input = grpcclient.InferInput(
        name,
        input.shape,
        utils.np_to_triton_dtype(input.dtype),
    )
    triton_input.set_data_from_numpy(input)
    return triton_input


def format_triton_output(name: str):
    triton_output = grpcclient.InferRequestedOutput(
        name,
    )
    return triton_output


def preprocess_function(examples):
    # Tokenize the texts
    example = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*example, padding="max_length", max_length=128, truncation=True)

    return result


raw_datasets = datasets.load_dataset("glue", "mnli")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
print(train_dataset)

URL = "localhost:8001"

def query_server():
    triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    count = 0
    times = []
    for batch in tqdm(dataloader):
        # print(batch)
        input_ids = np.asarray(batch["input_ids"]).astype(np.int32)
        attention_mask = np.asarray(batch["attention_mask"]).astype(np.int32)
        triton_inputs = [
            format_triton_input(input_ids, "encoder_input_ids"),
            format_triton_input(attention_mask, "encoder_attention_mask"),
            format_triton_input(np.zeros((1, 1)).astype(np.int32), "decoder_input_ids"),
            format_triton_input(np.ones((1, 1)).astype(np.int32), "decoder_attention_mask"),
        ]
        triton_outputs = [format_triton_output("logits")]

        start_time = time.perf_counter()
        results = triton_client.infer(
            f"{args.model_name}-ensemble",
            triton_inputs,
            outputs=triton_outputs,
            request_id=uuid.uuid4().hex,
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)


import multiprocessing as mp

if __name__ == "__main__":
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=query_server)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()