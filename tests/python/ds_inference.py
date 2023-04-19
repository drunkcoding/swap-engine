import asyncio
from dataclasses import dataclass, field
from email import parser
import gc
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
    AutoTokenizer,
    Trainer,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense,
)
from transformers import T5Tokenizer, default_data_collator
import datasets
import deepspeed
from transformers.deepspeed import deepspeed_init
from deepspeed.moe.layer import MoE
from dataloader import *
from huggingface_hub import hf_hub_download
import multiprocessing as mp

# mute all warnings
import warnings

warnings.filterwarnings("ignore")


def get_cache_partition(key):
    path = hf_hub_download(
        f"google/{args.model_name}", filename=key, cache_dir=args.model_path
    )
    gc.collect()
    return path


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        return None


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    model_path: str = field(metadata={"help": "Path to the model"})
    batch_size: int = field(default=8, metadata={"help": "Batch size to use"})
    dataset: str = field(default="glue", metadata={"help": "Dataset to use"})
    task: str = field(default=None, metadata={"help": "Task to use"})
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process"})
    deepspeed_config: str = field(
        default=None, metadata={"help": "Path to the deepspeed config file"}
    )

    def __post_init__(self):
        if self.task == "mnli":
            self.task = "mnli_matched"
        if self.dataset == "squad":
            self.split = "validation"
        else:
            self.split = "test"


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
print(args)
# torch.set_printoptions(profile="full")

sentence1_key, sentence2_key = sentence_keys(args.dataset, args.task)


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

print(raw_datasets)

tokenizer = AutoTokenizer.from_pretrained(
    f"google/{args.model_name}", cache_dir=args.model_path
)

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
    shuffle=True,
    drop_last=True,
)

print("SwitchTransformersForConditionalGeneration")

config = AutoConfig.from_pretrained(
    f"google/{args.model_name}", cache_dir=args.model_path
)

# # test if is a large checkpoint split into multiple files
# index_path = get_cache_partition("pytorch_model.bin.index.json")
# # load json file index_path
# with open(index_path) as f:
#     weight_map = json.load(f)["weight_map"]

# # create a reverse mapping from file name to weight name list
# file_to_weights = {}
# for weight_name, file_name in weight_map.items():
#     if file_name not in file_to_weights:
#         file_to_weights[file_name] = []
#     file_to_weights[file_name].append(weight_name)

# # get unique file names
# file_names = list(set(file_to_weights.keys()))
# print("Model files: ", file_names, flush=True)


# print("Model partitions: ", partitions.keys(), flush=True)
# exit()
model = AutoModelForSeq2SeqLM.from_pretrained(
    f"google/{args.model_name}",
    config=config,
    cache_dir=args.model_path,
)
state_dict = model.state_dict()


def load(module: torch.nn.Module, prefix=""):
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    print(f"Loading {prefix}...")
    with deepspeed.zero.GatheredParameters(
        list(module.parameters(recurse=False)), modifier_rank=0
    ):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, False, [], [], [])

    for name, child in module._modules.items():
        if child is not None:
            load(child, prefix + name + ".")


with deepspeed.zero.Init(
    config_dict_or_path=args.deepspeed_config,
):
    model = SwitchTransformersForConditionalGeneration(config)
    model.eval()

load(model)
# del state_dict

# for partition in tqdm(file_names):
#     print(f"Loading partition {partition}...")
#     path = get_cache_partition(partition)
#     state_dict = torch.load(path, map_location="cpu")
#     load(model, state_dict)
#     del state_dict
#     gc.collect()
#     torch.cuda.empty_cache()

# print("SwitchTransformersForConditionalGeneration Model Loaded")
# dummy = DummyModule()
# optimizer = torch.optim.Adam(dummy.parameters(), lr=0.0001 )
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lambda x: min(1.0, (x + 1) / 100)
# )
# params_group = [
#     {
#         "params": [p for n, p in dummy.named_parameters()],
#         "lr": 0.0001,
#     }
# ]
model_parameters = [next(model.parameters())]

# with open(args.deepspeed_config, "r") as f:
#     zero_config = json.load(f)
# zero_config = zero_config["zero_optimization"]

# deepspeed_engine = deepspeed.init_inference(
#     model=model,
#     config={
#         "dtype": torch.float32,
#         "tensor_parallel": {
#             "enabled": True,
#             "tp_size": 1,
#         },
#         "quant": {
#             "enabled": False,
#         },
#         "zero": {
#             "stage": 3,
#         }
#     },
# )
deepspeed_engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model_parameters,
)
model = deepspeed_engine.module
model.eval()

# print("SwitchTransformersForConditionalGeneration Model Initialized")
with torch.no_grad():
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

        outputs = model(
            input_ids=torch.Tensor(input_ids).long().to("cuda"),
            attention_mask=torch.Tensor(attention_mask).long().to("cuda"),
            decoder_input_ids=torch.Tensor(decoder_input_ids).long().to("cuda"),
            decoder_attention_mask=torch.Tensor(decoder_attention_mask)
            .long()
            .to("cuda"),
        )
        del outputs
