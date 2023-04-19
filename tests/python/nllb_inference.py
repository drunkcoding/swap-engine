import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersConfig,
)
from dataclasses import dataclass, field
from nllb_moe import NllbMoeForConditionalGeneration, NllbMoeConfig
from dataloader import *


@dataclass
class ModelArguments:
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process"})
    deepspeed_config: str = field(
        default=None, metadata={"help": "Path to the deepspeed config file"}
    )

    def __post_init__(self):
        # self.task = "mnli_matched"
        # self.dataset = "glue"
        self.task = "boolq"
        self.dataset = "super_glue"
        self.split = "test"
        self.batch_size = 1


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

config = NllbMoeConfig.from_pretrained(
    "facebook/nllb-moe-54b", cache_dir="/mnt/data/xly/.cache"
)

model = NllbMoeForConditionalGeneration.from_pretrained(
    "facebook/nllb-moe-54b", cache_dir="/mnt/data/xly/.cache", config=config
)
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-moe-54b", cache_dir="/mnt/data/xly/.cache"
)

# config = SwitchTransformersConfig.from_pretrained(
#     "google/switch-base-128", cache_dir="/mnt/data/xly/.cache"
# )

# model = SwitchTransformersForConditionalGeneration.from_pretrained(
#     "google/switch-base-128", cache_dir="/mnt/data/xly/.cache", config=config
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "google/switch-base-128", cache_dir="/mnt/data/xly/.cache"
# )

sentence1_key, sentence2_key = sentence_keys(args.dataset, args.task)


def preprocess_function(examples):
    # Tokenize the texts
    example = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*example, padding="do_not_pad", truncation=False)

    return result


raw_datasets = datasets.load_dataset(args.dataset, args.task)

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

# print("SwitchTransformersForConditionalGeneration Model Initialized")
with torch.no_grad():
    for batch in tqdm(dataloader):
        # print(batch)
        input_ids = np.asarray(batch["input_ids"]).astype(np.int32)
        attention_mask = np.asarray(batch["attention_mask"]).astype(np.int32)
        # # if args.batch_size == 1:
        # #     non_zeros = attention_mask > 0
        # #     input_ids = input_ids[non_zeros]
        # #     attention_mask = attention_mask[non_zeros]

        # #     input_ids = np.expand_dims(input_ids, axis=0)
        # #     attention_mask = np.expand_dims(attention_mask, axis=0)

        # decoder_length = 3
        # idx = np.random.choice(np.arange(20), decoder_length, replace=False)
        # idx = np.sort(idx)
        # decoder_input_ids = input_ids.copy()
        # decoder_input_ids = decoder_input_ids[:, :32]
        # decoder_input_ids[:, idx] = np.array([x for x in range(decoder_length)]) + 32000
        # # insert 0 at begining of decoder input ids
        # decoder_input_ids = np.insert(decoder_input_ids, 0, 0, axis=1)
        # decoder_attention_mask = np.ones_like(decoder_input_ids).astype(np.int32)

        # print(input_ids.shape, attention_mask.shape, decoder_input_ids.shape, decoder_attention_mask.shape)

        outputs = model(
            input_ids=torch.Tensor(input_ids).long().to("cpu"),
            attention_mask=torch.Tensor(attention_mask).long().to("cpu"),
            decoder_input_ids=torch.Tensor([[1]]).long().to("cpu"),
            decoder_attention_mask=torch.Tensor([[1]]).long().to("cpu"),
        )
        del outputs
