from dataclasses import dataclass, field
import gc
import json
import time
import traceback
import uuid
import numpy as np
from transformers import SwitchTransformersForConditionalGeneration, HfArgumentParser
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseGatedActDense,
)
import torch
import multiprocess as mp
import re

from pyutils.ckpt_config import *
import pyutils.ckpt_config as ckpt_config
from pyutils.ckpt_load import export_torchscript_model
from pysrc.transformer.switch.modeling_switch_transformers import *


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    model_path: str = field(metadata={"help": "Path to model cache directory."})

    def __post_init__(self):
        self.model_tag = self.model_name.split("/")[1]
        self.model_repo = "_".join(["model_repo", self.model_tag])
        "%s_encoder_embed" % self.model_tag,


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

config = SwitchTransformersConfig.from_pretrained(
    args.model_name, cache_dir=args.model_path
)
gold_model = SwitchTransformersForConditionalGeneration.from_pretrained(
    args.model_name, cache_dir=args.model_path
)
gold_model.eval()
gold_model.to("cuda")

input_ids = torch.randint(0, 30000, (1, 128)).to(torch.int32).cuda()
attention_mask = torch.ones_like(input_ids).cuda().to(torch.int32)
attention_mask = torch.cat([attention_mask, attention_mask]).cuda()
input_ids = torch.cat([input_ids, input_ids]).cuda()
decoder_input_ids = torch.zeros((input_ids.shape[0], 1)).cuda().to(torch.int32)
decoder_attention_mask = torch.ones((input_ids.shape[0], 1)).cuda().to(torch.int32)

print("input_ids", input_ids.shape)
print("attention_mask", attention_mask.shape)
print("decoder_input_ids", decoder_input_ids.shape)
print("decoder_attention_mask", decoder_attention_mask.shape)

start_time = time.time()
with torch.no_grad():
    output = gold_model(
        input_ids,
        attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_router_logits=True,
        return_dict=True,
    )
end_time = time.time()
# print("gold_model inference time", end_time - start_time)

print(output.keys())

gold_logits = output.logits
gold_encoder_hidden_states = output.encoder_hidden_states
gold_decoder_hidden_states = output.decoder_hidden_states
gold_hidden_states = gold_encoder_hidden_states + gold_decoder_hidden_states

test_model = SwitchModel(config)

gold_state_dict = gold_model.state_dict()

# print("gold_states", gold_model.state_dict().keys())
# print("test_states", test_model.state_dict().keys())

# copy state dict
key_list = list(gold_state_dict.keys())
for key in key_list:
    # print("key", key, type(key))
    old_key = key
    if "mlp." in key and ("wi" in key or "wo" in key) and not "expert" in key:
        key = re.sub(r"mlp\.", "mlp.mlp.", key, 1)
    if "embed_tokens" in key:
        key = re.sub(r"embed_tokens\.", "embed_tokens.embed_tokens.", key, 1)
    if "final_layer_norm" in key:
        key = re.sub(r"final_layer_norm\.", "final_layer_norm.layer_norm.", key, 1)
    
    for i in range(config.num_layers):
        if f"encoder.block.{i}.layer.1.layer_norm" in key or f"decoder.block.{i}.layer.2.layer_norm" in key:
            if i % 2 == 0:
                key = re.sub(r"layer_norm\.", "mlp.layer_norm.", key, 1)
            else:
                key = re.sub(r"layer_norm\.", "mlp.router.layer_norm.", key, 1)
    
    if key != old_key:
        # print("key", key, "old_key", old_key)
        gold_state_dict[key] = gold_state_dict.pop(old_key)

gold_state_dict["encoder.embed_tokens.relative_attention_bias.weight"] = gold_state_dict.pop( "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
gold_state_dict["decoder.embed_tokens.relative_attention_bias.weight"] = gold_state_dict.pop( "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")

# print("gold_states", gold_state_dict.keys())
# print("test_states", test_model.state_dict().keys())
with torch.no_grad():
    test_model.load_state_dict(gold_state_dict) 
test_model.eval()
test_model.to("cuda")

with torch.no_grad():
    output = test_model(
        input_ids,
        attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_hidden_states=True,
    )

test_logits = output[0]
test_hidden_states = output[1]
print(len(gold_hidden_states), len(test_hidden_states))

for i in range(len(gold_hidden_states)):
    print("gold_hidden_states", gold_hidden_states[i].shape)
    print("test_hidden_states", test_hidden_states[i].shape)
    print(i, torch.allclose(gold_hidden_states[i], test_hidden_states[i]))
    print(gold_hidden_states[i])

print(torch.allclose(test_logits, gold_logits, atol=1e-3))
# print(test_logits, gold_logits)

import tritonclient.grpc as grpcclient
from tritonclient import utils

URL = "localhost:50051"
triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)

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

input_ids = input_ids.detach().cpu().numpy()
attention_mask = attention_mask.detach().cpu().numpy()
decoder_input_ids = decoder_input_ids.detach().cpu().numpy()
decoder_attention_mask = decoder_attention_mask.detach().cpu().numpy()
triton_inputs = [
    format_triton_input(input_ids, "encoder_input_ids"),
    format_triton_input(attention_mask, "encoder_attention_mask"),
    format_triton_input(decoder_input_ids, "decoder_input_ids"),
    format_triton_input(decoder_attention_mask, "decoder_attention_mask"),
]
triton_outputs = [format_triton_output("logits")]

ensemble_name = args.model_name.split("/")[-1]
results = triton_client.infer(
    f"{ensemble_name}-ensemble",
    triton_inputs,
    outputs=triton_outputs,
    request_id=uuid.uuid4().hex,
    sequence_id=0,
)

triton_logits = results.as_numpy("logits")
print(torch.allclose(torch.from_numpy(triton_logits).to("cuda"), gold_logits, atol=1e-3))

print(triton_logits)
print(gold_logits)
print("====================================================================")
