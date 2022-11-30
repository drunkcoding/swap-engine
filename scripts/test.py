from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import numpy as np
import random
import torch
import time
import datetime
import os

CHECKPOINT = "bert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 3

FORWARD_CSV = "forward_pass_results.csv"
BACKWARD_CSV = "backward_pass_results.csv"

"""
SET UP CSV HANDLERS FOR ENERGY MEASUREMENT
"""


def setup_csv_handler(filename):
    if os.path.isfile(filename):
        os.remove(filename)
    return CSVHandler(filename)


forward_csv_handler = setup_csv_handler(FORWARD_CSV)
backward_csv_handler = setup_csv_handler(BACKWARD_CSV)

"""
TRAINING LOOP HELPER METHODS
"""


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


@measure_energy(handler=forward_csv_handler)
def forward_pass(model, b_ids, b_masks, b_labels):
    return model(b_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)


@measure_energy(handler=backward_csv_handler)
def backward_pass(loss):
    loss.backward()


"""
PRE-PROCESSING HELPER METHODS
"""


def tokenize_function(instance):
    return tokenizer(
        instance["sentence"],
        truncation=True,
    )


def get_sorted_dataset(dataset):
    return sorted(dataset, key=lambda x: len(x["input_ids"]))


def get_batches(dataset):
    batches = []
    while len(dataset) > 0:
        to_take = min(BATCH_SIZE, len(dataset))
        select = random.randint(0, len(dataset) - to_take)
        batch = dataset[select : (select + to_take)]
        batches.append(batch)

        del dataset[select : (select + to_take)]

    return batches


def pad_batch(batch):
    max_size = max([len(instance["input_ids"]) for instance in batch])

    for instance in batch:
        tokens = instance["input_ids"]
        num_pads = max_size - len(tokens)

        new_input_ids = tokens + [tokenizer.pad_token_id] * num_pads
        new_attn_mask = [1] * len(tokens) + [0] * num_pads

        instance["input_ids"] = new_input_ids
        instance["attention_mask"] = new_attn_mask

    return batch


def dynamically_pad(batched_dataset):
    for batch in batched_dataset:
        batch = pad_batch(batch)

    return batched_dataset


def get_field_from_batches(field, batches):
    return [torch.tensor([instance[field] for instance in batch]) for batch in batches]


def pre_process(dataset):
    data_sorted = get_sorted_dataset(dataset)

    batches = get_batches(data_sorted)
    batches_padded = dynamically_pad(batches)

    ids = get_field_from_batches("input_ids", batches_padded)
    labels = get_field_from_batches("label", batches_padded)
    masks = get_field_from_batches("attention_mask", batches_padded)

    return (ids, masks, labels)


"""
TRAINING LOOP SET-UP
"""

datasets = load_dataset("glue", "cola")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, do_lower_case=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)
training_data = pre_process(tokenized_datasets["train"])[0]

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
optimiser = AdamW(model.parameters(), lr=1e-5)
num_train_steps = len(training_data) * EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimiser,
    num_warmup_steps=0,
    num_training_steps=num_train_steps,
)

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device = torch.device("cuda:0") if available_gpus else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(len(training_data)))

"""
TRAINING LOOP
"""

for epoch in range(EPOCHS):
    ids, masks, labels = pre_process(tokenized_datasets["train"])

    """
    TRAINING
    """

    print("")
    print(f"======== Epoch {epoch + 1} / {EPOCHS} ========")
    print("Training...")

    total_loss = 0
    t0 = time.time()
    model.train()
    progress_bar.reset()

    for i in range(len(ids)):
        b_ids = ids[i].to(device)
        b_masks = masks[i].to(device)
        b_labels = labels[i].to(device)

        model.zero_grad()
        outputs = forward_pass(model, b_ids, b_masks, b_labels)

        loss = outputs.loss
        total_loss += loss.item()
        backward_pass(loss)

        optimiser.step()
        lr_scheduler.step()
        optimiser.zero_grad()
        progress_bar.update(1)

    avg_train_loss = total_loss / len(ids)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    """
    VALIDATION
    """

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    ids, masks, labels = pre_process(tokenized_datasets["validation"])

    for i in range(len(ids)):
        b_ids = ids[i].to(device)
        b_masks = masks[i].to(device)
        b_labels = labels[i].to(device)

        with torch.no_grad():
            outputs = model(b_ids, token_type_ids=None, attention_mask=b_masks)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

backward_csv_handler.save_data()
forward_csv_handler.save_data()
print("")
print("Training complete!")
