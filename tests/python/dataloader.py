import torch
import datasets 
from transformers import default_data_collator

def load_dataset(dataset, task=None, preprocess_function=None):
    raw_datasets = datasets.load_dataset(dataset, task)
    if dataset == "squad":
        split = "validation"
    else:
        split = "test"
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets[split].column_names,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets[split]
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=8,
        shuffle=True,
        drop_last=True,
    )

    return dataloader

def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key=None):
    # Tokenize the texts
    example = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*example, padding="max_length", max_length=128, truncation=True)

    return result

def sentence_keys(dataset, task=None):
    if dataset == "glue" and "mnli" in task:
        sentence1_key, sentence2_key = "premise", "hypothesis"
    elif dataset == "glue" and task == "rte":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "glue" and task == "sst2":
        sentence1_key, sentence2_key = "sentence", None
    elif dataset == "glue" and task == "cola":
        sentence1_key, sentence2_key = "sentence", None
    elif dataset == "glue" and task == "mrpc":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "glue" and task == "qqp":
        sentence1_key, sentence2_key = "question1", "question2"
    elif dataset == "glue" and task == "qnli":
        sentence1_key, sentence2_key = "question", "sentence"
    elif dataset == "glue" and task == "stsb":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "glue" and task == "wnli":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "super_glue" and task == "boolq":
        sentence1_key, sentence2_key = "question", "passage"
    elif dataset == "super_glue" and task == "cb":
        sentence1_key, sentence2_key = "premise", "hypothesis"
    elif dataset == "super_glue" and task == "copa":
        sentence1_key, sentence2_key = "premise", "choice1"
    elif dataset == "super_glue" and task == "multirc":
        sentence1_key, sentence2_key = "paragraph", "question"
    elif dataset == "super_glue" and task == "record":
        sentence1_key, sentence2_key = "passage", "question"
    elif dataset == "super_glue" and task == "rte":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "super_glue" and task == "wic":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif dataset == "super_glue" and task == "wsc":
        sentence1_key, sentence2_key = "text", "target"
    elif dataset == "super_glue" and task == "wsc.fixed":
        sentence1_key, sentence2_key = "text", "target"
    elif dataset == "super_glue" and task == "axg":
        sentence1_key, sentence2_key = "premise", "hypothesis"
    elif dataset == "super_glue" and task == "axb":
        sentence1_key, sentence2_key = "premise", "hypothesis"
    elif dataset == "squad":
        sentence1_key, sentence2_key = "context", "question"
    else:
        raise ValueError(f"Unknown dataset/task combination: {dataset}/{task}")
    
    return sentence1_key, sentence2_key
