import transformers
import datasets
from transformers import T5Tokenizer, default_data_collator
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")


def preprocess_function(examples):
    # Tokenize the texts
    example = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*example, padding="do_not_pad")

    return result


# load GLUE dataset

# sentence1_key, sentence2_key = "premise", "hypothesis"
# raw_datasets = datasets.load_dataset("glue", "mnli_mismatched")

# sentence1_key, sentence2_key = "question", "passage"
# raw_datasets = datasets.load_dataset("super_glue", "boolq")

sentence1_key, sentence2_key = "context", "question"
raw_datasets = datasets.load_dataset("squad")

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
    desc="Running tokenizer on dataset",
)
dataset = train_dataset = processed_datasets["validation"]
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=default_data_collator,
    batch_size=1,
    shuffle=True,
    drop_last=True,
)

# calculate average length of input

total_length = 0
for batch in dataloader:
    batch_size, seq_len = batch["input_ids"].shape
    total_length += batch_size * seq_len

avg_length = total_length / len(dataloader.dataset)

print("Average length of input: ", avg_length)
