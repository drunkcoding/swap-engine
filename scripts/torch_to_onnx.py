# Some standard imports
from dataclasses import dataclass, field
import os

import torch.onnx
from torch.utils.data import DataLoader

from transformers import HfArgumentParser
from transformers import ViTForImageClassification
import transformers

from torchvision.datasets import ImageNet

from baseline.pyutils.vit_images_preprocess import (
    ViTFeatureExtractorTransforms,
    vit_collate_fn,
)


@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(metadata={"help": "transformer model type"})
    dataset_path: str = field(metadata={"help": "Path to ImageNet dataset"})


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

# prepare dataset
if "vit" in args.model_type.lower():
    dataset = ImageNet(
        args.dataset_path,
        split="val",
        transform=ViTFeatureExtractorTransforms(args.model_path, split="val"),
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=20,
        collate_fn=vit_collate_fn,
        batch_size=32,
    )


# prepare model
if "vit" in args.model_type.lower():
    model = transformers.AutoModelForImageClassification.from_pretrained(
        args.model_path
    )
    model.eval()
    model.cuda()


# convert to onnx

for dummy_input in dataloader:
    dummy_input = dummy_input["pixel_values"].cuda()
    break
torch.onnx.export(
    model,
    dummy_input,
    os.path.join(args.model_path, "model.onnx"),
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
    verbose=False,
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
)
