import asyncio
from dataclasses import dataclass, field
from email import parser
from itertools import count
import json
import os
import time
import uuid
import grpc
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pysrc.cascader import CascadeHandler
from pysrc.connector import TritonLocalConnector, DeepspeedLocalConnector
from pyutils.vit_images_preprocess import ViTFeatureExtractorTransforms, vit_collate_fn
from transformers import HfArgumentParser
from torchvision.datasets import ImageNet

# os.environ["GRPC_TRACE"]="all"
# os.environ["GRPC_VERBOSITY"]="debug"

@dataclass
class ClientArguments:
    config: str = field(metadata={"help": "configuration file for confidence"})
    model_path: str = field(metadata={"help": "model path with tokenizers"})
    dataset_path: str = field(metadata={"help": "Path to ImageNet dataset"})
    server_type: str = field(
        metadata={"help": "Type of server to use [deepspeed, triton]"}
    )
    model_name: str = field(metadata={"help": "model name"})
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    url: str = field(
        default="localhost:50051",
        metadata={"help": "Inference server URL. Default is localhost:50051."},
    )

def get_input_name(server_type, model_name):
    if server_type == "triton":
        return "input"
    elif server_type == "deepspeed":
        return "pixel_values"
    else:
        raise ValueError(f"Unknown server type {server_type}")

def get_output_name(server_type, model_name):
    if server_type == "triton":
        return "output"
    elif server_type == "deepspeed":
        return "logits"
    else:
        raise ValueError(f"Unknown server type {server_type}")

parser = HfArgumentParser((ClientArguments,))
args = parser.parse_args_into_dataclasses()[0]

config = json.load(open(args.config, "r"))

connector = (
    TritonLocalConnector(config, url=args.url, verbose=args.verbose, prefetch=False)
    if args.server_type == "triton"
    else DeepspeedLocalConnector(config, url=args.url, verbose=args.verbose)
)

dataset = ImageNet(
    args.dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.model_path, split="val"),
)

torch.random.manual_seed(0)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=vit_collate_fn,
)

count = 0
times = []
for batch in tqdm(dataloader):
    inputs = {
        get_input_name(args.server_type, args.model_name): batch["pixel_values"].numpy().astype(np.float32),
    }
    outputs = {
        get_output_name(args.server_type, args.model_name):  np.zeros((batch["pixel_values"].shape[0], 1000), dtype=np.float32),
    }
    # print(inputs)
    # print(outputs)
    start_time = time.perf_counter()
    outputs = connector.infer(args.model_name, inputs, outputs, uuid.uuid4().hex)
    end_time = time.perf_counter()
    times.append(end_time - start_time)
    # print(f"triton inference time: {end_time - start_time}")
    # if count > 10:
    #     break
    # if count % 1000 == 0:
    #     print(f"Average inference time: {np.mean(times)}")
