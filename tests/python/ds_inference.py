import asyncio
from dataclasses import dataclass, field
from email import parser
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
args = parser.parse_args_into_dataclasses()[0]

# torch.set_printoptions(profile="full")

if args.dataset == "glue" and "mnli" in args.task:
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "glue" and args.task == "rte":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "sst2":
    sentence1_key, sentence2_key = "sentence", None
elif args.dataset == "glue" and args.task == "cola":
    sentence1_key, sentence2_key = "sentence", None
elif args.dataset == "glue" and args.task == "mrpc":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "qqp":
    sentence1_key, sentence2_key = "question1", "question2"
elif args.dataset == "glue" and args.task == "qnli":
    sentence1_key, sentence2_key = "question", "sentence"
elif args.dataset == "glue" and args.task == "stsb":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "glue" and args.task == "wnli":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "boolq":
    sentence1_key, sentence2_key = "question", "passage"
elif args.dataset == "super_glue" and args.task == "cb":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "super_glue" and args.task == "copa":
    sentence1_key, sentence2_key = "premise", "choice1"
elif args.dataset == "super_glue" and args.task == "multirc":
    sentence1_key, sentence2_key = "paragraph", "question"
elif args.dataset == "super_glue" and args.task == "record":
    sentence1_key, sentence2_key = "passage", "question"
elif args.dataset == "super_glue" and args.task == "rte":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "wic":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif args.dataset == "super_glue" and args.task == "wsc":
    sentence1_key, sentence2_key = "text", "target"
elif args.dataset == "super_glue" and args.task == "wsc.fixed":
    sentence1_key, sentence2_key = "text", "target"
elif args.dataset == "super_glue" and args.task == "axg":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "super_glue" and args.task == "axb":
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif args.dataset == "squad":
    sentence1_key, sentence2_key = "context", "question"
else:
    raise ValueError(f"Unknown dataset/task combination: {args.dataset}/{args.task}")


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
    shuffle=True,
    drop_last=True,
)

print("SwitchTransformersForConditionalGeneration")

config = AutoConfig.from_pretrained(
    f"google/{args.model_name}", cache_dir=args.model_path
)

model_origin = AutoModelForSeq2SeqLM.from_pretrained(
    f"google/{args.model_name}", config=config, cache_dir=args.model_path
)
state_dict = model_origin.state_dict()


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


with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
    model = SwitchTransformersForConditionalGeneration(config)
    model.eval()

load(model, "")

del model_origin
del state_dict

print("SwitchTransformersForConditionalGeneration Model Loaded")

deepspeed_engine, _, _, _ = deepspeed.initialize(args=args, model=model)
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
