from pysrc.transformer.switch.modeling_switch import SwitchModel
from pysrc.transformer.switch.configuration_switch import SwitchConfig
import torch
import os
from transformers import T5Tokenizer, default_data_collator
import datasets

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
    train_dataset, collate_fn=default_data_collator, batch_size=8, shuffle=True, drop_last=True
)

CKPT_PATH = "/mnt/xly/checkpoints/t5x-torchscript/moe/base/e128"

# encoded_input = tokenizer(
#     "A reference client implementation for the playback of MPEG DASH via JavaScript and compliant browsers. Learn more about DASH IF Reference Client on our wiki.",
#     return_tensors="pt",
# )
# # encoded_input = encoded_input.to("cuda")
# print(encoded_input)

config = SwitchConfig.from_pretrained("config/t5x/base")
model = SwitchModel(config)
state_dict = torch.load(os.path.join(CKPT_PATH, "model.pth"), map_location="cpu")
model.load_state_dict(state_dict)

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
        outputs = model(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        print("batch end")
