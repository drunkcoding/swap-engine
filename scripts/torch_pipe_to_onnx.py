# Some standard imports
from dataclasses import dataclass, field
from multiprocessing import dummy
import os
import numpy as np
import torch.onnx
import torch
from torch.utils.data import DataLoader

from transformers import HfArgumentParser
from transformers import ViTForImageClassification, ViTConfig
import transformers

from pysrc.pipe.vit import ViTModelPipe

from torchvision.datasets import ImageNet

from pyutils.vit_images_preprocess import (
    ViTFeatureExtractorTransforms,
    vit_collate_fn,
)

import torch

torch.manual_seed(0)

def save_torchscript_model(model, path):
    model.eval()
    model.cpu()
    model = torch.jit.script(model)
    model.save(path)

@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    export_type: str = field(metadata={"help": "Type of export [onnx, torchscript]"})
    model_type: str = field(metadata={"help": "transformer model type"})
    dataset_path: str = field(metadata={"help": "Path to ImageNet dataset"})
    model_repo: str = field(metadata={"help": "Path to model repo"})
    cfg_only: bool = field(default=False, metadata={"help": "Only import config"})
    cuda: bool = field(default=False, metadata={"help": "trace on gpu"})

    def __post_init__(self):
        self.model_basename = os.path.basename(self.model_path)


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
    if args.cfg_only:
        config = ViTConfig.from_pretrained(args.model_path)
        model_tf = ViTForImageClassification(config)
    else:
        model_tf = transformers.AutoModelForImageClassification.from_pretrained(
            args.model_path
        )
    model_tf.eval()

    # prepare model for pipe
    model = ViTModelPipe(model_tf)
    model.eval()
if args.cuda:
    model.cuda()
    model_tf = model_tf.cuda()

# convert to onnx

for batch in dataloader:
    dummy_input = batch["pixel_values"]  # .cuda()

    break
if args.cuda:
    dummy_input = dummy_input.cuda()
    batch = {name: value.cuda() for name, value in batch.items()}

with torch.no_grad():
    for i, layer_module in enumerate(model.layers):
        save_path = os.path.join(args.model_repo, args.model_basename + f"_{i}", "0")
        try:
            os.mkdir(os.path.dirname(save_path))
            os.mkdir(save_path)
        except FileExistsError:
            pass

        if args.export_type == "onnx":
            torch.onnx.export(
                layer_module,
                dummy_input,
                os.path.join(save_path, "model.onnx"),
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        elif args.export_type == "torchscript":
            torch.jit.save(
                torch.jit.script(layer_module),
                os.path.join(save_path, "model.pt"),
            )
            config_text = """
            platform: "pytorch_libtorch"
            input [
                {
                name: "input"
                data_type: TYPE_FP32
                dims: [ %s ]
                }
            ]
            output [
                {
                name: "output"
                data_type: TYPE_FP32
                dims: [ %s ]
                }
            ]
            """ % (
                "-1, 3, 224, 224" if i == 0 else "-1, -1, -1",
                "-1, -1, -1" if i != len(model.layers) - 1 else "-1, 1000",
            )
            with open(
                os.path.join(os.path.dirname(save_path), "config.pbtxt"), "w"
            ) as f:
                f.write(config_text)
        dummy_input = layer_module(dummy_input)
        # print(dummy_input)

    # save the entire model
    save_path = os.path.join(args.model_repo, args.model_basename, "0")
    try:
        os.mkdir(os.path.dirname(save_path))
        os.mkdir(save_path)
    except FileExistsError:
        pass

    if args.export_type == "torchscript":
        torch.jit.save(
            torch.jit.script(model),
            os.path.join(save_path, "model.pt"),
        )
        config_text = """
            platform: "pytorch_libtorch"
            input [
                {
                name: "input"
                data_type: TYPE_FP32
                dims: [ -1, 3, 224, 224 ]
                }
            ]
            output [
                {
                name: "output"
                data_type: TYPE_FP32
                dims: [ -1, 1000 ]
                }
            ]
            """
        with open(os.path.join(os.path.dirname(save_path), "config.pbtxt"), "w") as f:
            f.write(config_text)

    # Create tritonserver ensemble pbtxt

    ensemble_steps = []
    for i, layer_module in enumerate(model.layers):
        ensemble_steps.append(
            """
            {
                model_name: "%s"
                model_version: 0
                input_map {
                    key: "input"
                    value: "%s"
                }
                output_map {
                    key: "output"
                    value: "%s"
                }
            }
            """
            % (
                args.model_basename + f"_{i}",
                f"input" if i == 0 else f"output_{i-1}",
                f"output_{i}" if i < len(model.layers) - 1 else "output",
            )
        )

    ensemble_name = args.model_basename + "_ensemble"
    ensemble_text = """
    name: "%s"
    platform: "ensemble"
    input [
        {
        name: "input"
        data_type: TYPE_FP32
        dims: [ -1, 3, 224, 224 ]
        }
    ]
    output [
        {
        name: "output"
        data_type: TYPE_FP32
        dims: [ -1, 1000 ]
        }
    ]
    ensemble_scheduling {
        step [
        %s
        ]
    }
    """ % (
        ensemble_name,
        ",".join(ensemble_steps),
    )

    save_path = os.path.join(args.model_repo, ensemble_name, "0")
    try:
        os.mkdir(os.path.dirname(save_path))
        os.mkdir(save_path)
    except FileExistsError:
        pass

    with open(os.path.join(os.path.dirname(save_path), "config.pbtxt"), "w") as f:
        f.write(ensemble_text)

    outputs_tf = model_tf(
        batch["pixel_values"], return_dict=True, output_hidden_states=True
    )
    outputs_pipe = model.forward_with_hidden_states(batch["pixel_values"])

    hidden_states_tf = outputs_tf.hidden_states
    hidden_states_pipe = outputs_pipe[1]

    print("hidden_states_tf", len(hidden_states_tf))
    print("hidden_states_pipe", len(hidden_states_pipe))

    for idx in range(len(hidden_states_tf)):
        print("hidden_states_tf", hidden_states_tf[idx].shape)
        print("hidden_states_pipe", hidden_states_pipe[idx].shape)
        print(
            "diff", torch.isclose(hidden_states_tf[idx], hidden_states_pipe[idx]).all()
        )
        print("")

    logits_tf = outputs_tf.logits
    logits_pipe = outputs_pipe[0]

    print("logits_tf", logits_tf.shape)
    print("logits_pipe", logits_pipe.shape)
    print("diff", torch.isclose(logits_tf, logits_pipe).all())
