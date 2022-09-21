from dataclasses import dataclass, field
import os
import re
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(metadata={"help": "transformer model type"})

    def __post_init__(self):
        self.info_path = os.path.join(os.path.dirname(self.model_path), "model.info")


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]


os.system(
    "polygraphy inspect model %s --mode=basic > %s " % (args.model_path, args.info_path)
)

pattern = r"input\..*\}\n.*-> \{(input.*)\}"

with open(args.info_path, "r") as f:
    info = f.read()
    inputs = re.findall(pattern, info)
    print(inputs)


if "vit" in args.model_type.lower():
    inputs = ["pixel_values"] + inputs + ["logits"]

print(inputs)

# polygraphy surgeon extract --inputs pixel_values --outputs logits
for i in range(len(inputs) - 1):
    input_name = inputs[i]
    output_name = inputs[i + 1]
    os.system(
        "polygraphy surgeon extract %s -o %s --inputs %s --outputs %s"
        % (
            args.model_path,
            os.path.join(os.path.dirname(args.model_path), f"surgeon_{i}.onnx"),
            input_name + ":auto:auto",
            output_name + ":auto",
        )
    )
