from dataclasses import dataclass, field
import os
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import HfArgumentParser

@dataclass
class Arguments:
    model_path: str = field(metadata={"help": "Path to the model to export"})

    def __post_init__(self):
        self.output_path = os.path.join(os.path.dirname(self.model_path), "model.onnx")

parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

# load model and tokenizer
model = AutoModel.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

print(dummy_model_input)

# export
torch.onnx.export(
    model, 
    ({
        "input_ids": dummy_model_input.input_ids,
        "attention_mask": dummy_model_input.attention_mask,
    }, ),
    f=args.output_path,  
    input_names=['input_ids', 'attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                  'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
    opset_version=12, 
)
