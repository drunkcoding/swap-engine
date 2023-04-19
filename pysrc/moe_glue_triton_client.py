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

# mute all warnings
import warnings
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    num_processes: int = field(metadata={"help": "Number of processes to use"})
    batch_size: int = field(default=8, metadata={"help": "Batch size to use"})
    dataset: str = field(default="glue", metadata={"help": "Dataset to use"})
    task: str = field(default=None, metadata={"help": "Task to use"})

    def __post_init__(self):
        if self.task == "mnli":
            self.task = "mnli_matched"
        if self.dataset == "squad":
            self.split = "validation"
        else:
            self.split = "test"


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

torch.set_printoptions(profile="full")

if args.dataset == "glue" and "mnli" in args.task:
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "glue" and args.task == "rte":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "sst2":
    sentence1_key, sentence2_key = "sentence", None
elif args.dataset == "glue" and args.task == "cola":
    sentence1_key, sentence2_key = "sentence", None
elif args.dataset == "glue" and args.task == "mrpc":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "qqp":
    sentence1_key, sentence2_key = "question1", "question2"
elif args.dataset == "glue" and args.task == "qnli":
    sentence1_key, sentence2_key = "question", "sentence"
elif args.dataset == "glue" and args.task == "stsb":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "wnli":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "boolq":
    sentence1_key, sentence2_key = "question", "passage"
elif args.dataset == "super_glue" and args.task == "cb":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "super_glue" and args.task == "copa":
    sentence1_key, sentence2_key = "premise", "choice1"
elif args.dataset == "super_glue" and args.task == "multirc":
    sentence1_key, sentence2_key = "paragraph", "question"
elif args.dataset == "super_glue" and args.task == "record":
    sentence1_key, sentence2_key = "passage", "question"
elif args.dataset == "super_glue" and args.task == "rte":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "wic":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "wsc":
    sentence1_key, sentence2_key = "text", "target"
elif args.dataset == "super_glue" and args.task == "wsc.fixed":
    sentence1_key, sentence2_key = "text", "target"
elif args.dataset == "super_glue" and args.task == "axg":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "super_glue" and args.task == "axb":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "squad":
    sentence1_key, sentence2_key = "context", "question"
else:
    raise ValueError(f"Unknown dataset/task combination: {args.dataset}/{args.task}")


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


print(args.dataset, args.task)
raw_datasets = datasets.load_dataset(args.dataset, args.task)

print(raw_datasets)

tokenizer = T5Tokenizer.from_pretrained("t5-small")


processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets[args.split].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets[args.split]
print(train_dataset)

URL = "localhost:60051"


def query_server():
    triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    count = 0
    times = []
    for batch in tqdm(dataloader):
        # print(batch)
        input_ids = np.asarray(batch["input_ids"]).astype(np.int32)
        attention_mask = np.asarray(batch["attention_mask"]).astype(np.int32)

        # if args.batch_size == 1:
        #     non_zeros = attention_mask > 0
        #     input_ids = input_ids[non_zeros]
        #     attention_mask = attention_mask[non_zeros]

        #     input_ids = np.expand_dims(input_ids, axis=0)
        #     attention_mask = np.expand_dims(attention_mask, axis=0)

        decoder_length = 3
        idx = np.random.choice(np.arange(20), decoder_length, replace=False)
        idx = np.sort(idx)
        decoder_input_ids = input_ids.copy()
        decoder_input_ids = decoder_input_ids[:, :32]
        decoder_input_ids[:, idx] = np.array([x for x in range(decoder_length)]) + 32000
        # insert 0 at begining of decoder input ids
        decoder_input_ids = np.insert(decoder_input_ids, 0, 0, axis=1)
        decoder_attention_mask = np.ones_like(decoder_input_ids).astype(np.int32)

        triton_inputs = [
            format_triton_input(input_ids, "encoder_input_ids"),
            format_triton_input(attention_mask, "encoder_attention_mask"),
            # format_triton_input(np.zeros((input_ids.shape[0], 1)).astype(np.int32), "decoder_input_ids"),
            # format_triton_input(np.ones((input_ids.shape[0], 1)).astype(np.int32), "decoder_attention_mask"),
            format_triton_input(decoder_input_ids, "decoder_input_ids"),
            format_triton_input(decoder_attention_mask, "decoder_attention_mask"),
        ]
        triton_outputs = [format_triton_output("logits")]

        start_time = time.perf_counter()
        results = triton_client.infer(
            f"{args.model_name}-ensemble",
            triton_inputs,
            outputs=triton_outputs,
            request_id=uuid.uuid4().hex,
            sequence_id=0,
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
