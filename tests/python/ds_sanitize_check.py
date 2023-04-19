from transformers import (
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense,
    SwitchTransformersLayerFF,
)

import deepspeed
from deepspeed.moe.layer import MoE

NUM_GPUS = 2

class SwitchFFDeepSpeed(SwitchTransformersLayerFF):
    def __init__(self, config: SwitchTransformersConfig, is_sparse=False):
        super().__init__(config, is_sparse)

    def forward(self, hidden_states, output_router_logits):
        if self.is_sparse:
            forwarded_states = self.layer_norm(hidden_states)
            forwarded_states = self.mlp(forwarded_states)
            forwarded_states = forwarded_states[0]
            output = hidden_states + self.dropout(forwarded_states)
            return output
        else:
            return super().forward(hidden_states, output_router_logits)


class SwitchTransformerDeepSpeed(SwitchTransformersForConditionalGeneration):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.num_experts = config.num_experts

    def replace_moe_layer(self):
        for i, block in enumerate(self.encoder.block):
            ff_ds = SwitchFFDeepSpeed(self.config, is_sparse=block.layer[-1].is_sparse)
            ff_ds.load_state_dict(block.layer[-1].state_dict())
            block.layer[-1] = ff_ds
            if block.layer[-1].is_sparse:
                state_dict = block.layer[-1].mlp.state_dict()
                deepspeed_moe = MoE(
                    self.config.d_model,
                    block.layer[-1].mlp.experts["expert_0"],
                    self.num_experts,
                    capacity_factor=1.0,
                    eval_capacity_factor=2.0,
                    min_capacity=0,
                    use_rts=False,
                    ep_size=NUM_GPUS,
                )
                deepspeed_moe.eval()
                target_dict = {
                    "deepspeed_moe.gate.wg.weight": state_dict[
                        "router.classifier.weight"
                    ],
                }
                for i in range(self.num_experts):
                    target_dict[
                        f"deepspeed_moe.experts.deepspeed_experts.{i}.wi.weight"
                    ] = state_dict[f"experts.expert_{i}.wi.weight"]
                    target_dict[
                        f"deepspeed_moe.experts.deepspeed_experts.{i}.wo.weight"
                    ] = state_dict[f"experts.expert_{i}.wo.weight"]
                deepspeed_moe.load_state_dict(target_dict, strict=False)
                block.layer[-1].mlp = deepspeed_moe


import asyncio
from dataclasses import dataclass, field
from email import parser
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    SwitchTransformersForConditionalGeneration,
    T5ForConditionalGeneration,
    Trainer,
)
from transformers import T5Tokenizer, default_data_collator
import datasets
import deepspeed
from transformers.deepspeed import deepspeed_init


# mute all warnings
import warnings

warnings.filterwarnings("ignore")


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        return None


@dataclass
class ModelArguments:
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process"})
    deepspeed_config: str = field(
        default=None, metadata={"help": "Path to the deepspeed config file"}
    )

    def __post_init__(self):
        self.dataset = "super_glue"
        self.task = "boolq"
        self.batch_size = 1
        self.split = "test"
        self.model_path = "/mnt/raid0nvme1/xly/.cache"
        self.model_name = "switch-base-8"


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

sentence1_key, sentence2_key = "question", "passage"
def preprocess_function(examples):
    # Tokenize the texts
    example = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*example, padding="max_length", max_length=128, truncation=True)

    return result

raw_datasets = datasets.load_dataset(args.dataset, args.task)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets[args.split].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets[args.split]
print(train_dataset)

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=default_data_collator,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
)

batch= next(iter(dataloader))
input_ids = np.asarray(batch["input_ids"]).astype(np.int32)
attention_mask = np.asarray(batch["attention_mask"]).astype(np.int32)

decoder_length = 3
idx = np.random.choice(np.arange(20), decoder_length, replace=False)
idx = np.sort(idx)
decoder_input_ids = input_ids.copy()
decoder_input_ids = decoder_input_ids[:, :32]
decoder_input_ids[:, idx] = np.array([x for x in range(decoder_length)]) + 32000
decoder_input_ids = np.insert(decoder_input_ids, 0, 0, axis=1)
decoder_attention_mask = np.ones_like(decoder_input_ids).astype(np.int32)
input_ids = torch.Tensor(input_ids).long().to("cuda")
attention_mask = torch.Tensor(attention_mask).long().to("cuda")
decoder_input_ids = torch.Tensor(decoder_input_ids).long().to("cuda")
decoder_attention_mask = torch.Tensor(decoder_attention_mask).long().to("cuda")

print("SwitchTransformersForConditionalGeneration")

config = AutoConfig.from_pretrained(
    f"google/{args.model_name}", cache_dir=args.model_path
)
model_gold = SwitchTransformersForConditionalGeneration.from_pretrained(
    f"google/{args.model_name}", config=config, cache_dir=args.model_path
)
model_gold = model_gold.to("cuda")
model_gold.eval()
with torch.no_grad():
    outputs_gold = model_gold(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    hidden_states_gold = outputs_gold.encoder_hidden_states
del model_gold

model = SwitchTransformerDeepSpeed.from_pretrained(
    f"google/{args.model_name}", config=config, cache_dir=args.model_path
)
model.replace_moe_layer()
model.eval()

with open(args.deepspeed_config, "r") as f:
    zero_config = json.load(f)
zero_config = zero_config["zero_optimization"]
model = deepspeed.init_inference(
    model=model,
    config={
        "tensor_parallel": {
            "enabled": True,
            "tp_size": NUM_GPUS,
        },
        "quant": {
            "enabled": False,
        },
        # "zero": zero_config,
    },
)
model.eval()

input_ids = input_ids.to("cuda")
attention_mask = attention_mask.to("cuda")
decoder_input_ids = decoder_input_ids.to("cuda")
decoder_attention_mask = decoder_attention_mask.to("cuda")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    hidden_states = outputs.encoder_hidden_states

if args.local_rank == 0:
    print(outputs_gold.logits)
    print(outputs.logits)

    print(len(hidden_states_gold))
    print(len(hidden_states))

    print("=====================================")
    for i in range(len(hidden_states_gold)):
        print(hidden_states_gold[i].shape)
        print(hidden_states[i].shape)
        print(hidden_states_gold[i])
        print(hidden_states[i])
        print("=====================================")