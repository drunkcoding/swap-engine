from dataclasses import dataclass, field
from email import parser
from itertools import count
import json
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


@dataclass
class ClientArguments:
    config: str = field(metadata={"help": "configuration file for confidence"})
    model_path: str = field(metadata={"help": "model path with tokenizers"})
    dataset_path: str = field(metadata={"help": "Path to ImageNet dataset"})
    server_type: str = field(
        metadata={"help": "Type of server to use [deepspeed, triton]"}
    )
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    url: str = field(
        default="localhost:50051",
        metadata={"help": "Inference server URL. Default is localhost:50051."},
    )


parser = HfArgumentParser((ClientArguments,))
args = parser.parse_args_into_dataclasses()[0]

config = json.load(open(args.config, "r"))

connector = (
    TritonLocalConnector(args.url, args.verbose)
    if args.server_type == "triton"
    else DeepspeedLocalConnector(args.url, args.verbose)
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
    shuffle=True,
    num_workers=0,
    collate_fn=vit_collate_fn,
)

handler = CascadeHandler(config, connector)


for batch in tqdm(dataloader):
    inputs = {
        "pixel_values": batch["pixel_values"].numpy(),
    }
    outputs = handler(inputs)

