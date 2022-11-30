import argparse
import sys
import os

sys.path.append(os.getcwd())

from pysrc.transformer.switch.modeling_switch_ds import SwitchModelDeepSpeed, SwitchModelDeepSpeedPipe
from pysrc.transformer.switch.configuration_switch import SwitchConfig
import torch
import os
from transformers import T5Tokenizer, default_data_collator
import datasets
import deepspeed


parser = argparse.ArgumentParser(description="CIFAR")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher",
)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

torch.set_printoptions(profile="full")

sentence1_key, sentence2_key = "premise", "hypothesis"


def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)

    return result


raw_datasets = datasets.load_dataset("glue", "mnli")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
print(train_dataset)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=default_data_collator,
    batch_size=8,
    shuffle=True,
    drop_last=True,
)

# torch.distributed.init_process_group("nccl", )

deepspeed.init_distributed()

CKPT_PATH = "/mnt/xly/checkpoints/t5x-torchscript/moe/base/e128"
config = SwitchConfig.from_pretrained("config/t5x/base")
model = SwitchModelDeepSpeed(config)
model_pipe = SwitchModelDeepSpeedPipe(config, model, num_stages=torch.cuda.device_count())

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}

    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = create_moe_param_groups(model)


model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=model_pipe)

# state_dict = torch.load(os.path.join(CKPT_PATH, "model_ds.pth"), map_location="cpu")
# model_engine.load_checkpoint(
#     os.path.join(CKPT_PATH, "model_ds.pth"),
#     load_module_only=True,
#     load_optimizer_states=False,
#     load_lr_scheduler_states=False,
# )
model.eval()

# model = model.to("cuda")
with torch.no_grad():

    for batch in dataloader:
        # print(batch)
        print("batch start")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = torch.ones((1, 1), dtype=torch.long)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        outputs = model_engine(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        print("batch end")
