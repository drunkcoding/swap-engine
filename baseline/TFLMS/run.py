from transformers import SwitchTransformersForConditionalGeneration
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import torch
import onnx
from onnx_tf.backend import prepare

@dataclass
class Arguments:
    model_name: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: str = field(
        default=None,
    )


parser = HfArgumentParser((Arguments))
args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, cache_dir=args.cache_dir
)
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    args.model_name, cache_dir=args.cache_dir
)

# convert model to TensorFlow2

input_ids = torch.ones((1, 128), dtype=torch.long)
attention_mask = torch.ones((1, 128), dtype=torch.long)
decoder_input_ids = input_ids.clone()
decoder_input_ids = decoder_input_ids[:, :32]
decoder_attention_mask = torch.ones_like(decoder_input_ids)

torch.onnx.export(
    model,
    (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask),
    "model.onnx",
    opset_version=11,
)

model = onnx.load('model.onnx')
tf_rep = prepare(model)
tf_rep.export_graph('model.pb')
