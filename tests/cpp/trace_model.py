# import torch
# import torchvision

# # An instance of your model.
# model = torchvision.models.resnet18()

# # An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)

# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("traced_resnet_model.pt")

import time
import torch
from transformers import SwitchTransformersForConditionalGeneration, AutoTokenizer

model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-large-128", cache_dir="/mnt/data/xly/.cache"
)

model.encoder.block[1].to("cuda:1")
model.encoder.block[1].to("cpu")

for i in range(10):
    start_time = time.time()
    model.encoder.block[1].to("cuda:1")
    end_time = time.time()
    print("model loaded in {} seconds".format(end_time - start_time))
    model.encoder.block[1].to("cpu")

print("=====================================")

for i in range(10):
    start_time = time.time()
    model.encoder.block[0].to("cuda:1")
    end_time = time.time()
    print("model loaded in {} seconds".format(end_time - start_time))
    model.encoder.block[0].to("cpu")
