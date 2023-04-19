import pandas as pd
import torch

from transformers import AutoConfig
from transformers import AutoModel

# from pyJoules.energy_meter import measure_energy
# from pyJoules.handler.csv_handler import CSVHandler

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device = torch.device("cuda:2") if available_gpus else torch.device("cpu")

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def evaluate_layer(row):
    model_name, module_name = row["model_name"], row["level_name"].split(":")[1]

    print(model_name, module_name)

    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True

    model = AutoModel.from_config(config)
    model = model.eval().to(device)

    hook = None

    # Find first instance of module within model
    for _, module in model.named_modules():
        if type(module).__name__ == module_name:
            print(module_name)
            hook = Hook(module)
            break

    input_ids = torch.randint(1000, size=(int(row['batch_size']), int(row['seq_len'])), dtype=torch.long, device=device)
    atttion_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
    _ = model(input_ids, attention_mask=atttion_mask, token_type_ids=type_ids)

    print(hook.input)
    print(hook.output)
    hook.close()
    print("\n")

def main():
    df = pd.read_csv("module_level_features.csv", sep=",")
    df = df.head(2)

    for index, row in df.iterrows():
        evaluate_layer(row)

if __name__ == "__main__":
    main()