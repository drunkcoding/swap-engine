import torch
import time

MB = 1024 * 1024
model_path = "/mnt/xly/swap-engine/model_repo_switch-large-128/switch-large-128_encoder_expert_15_3/0/model.pt"

start_time = time.perf_counter()
model = torch.jit.load(model_path)
end_time = time.perf_counter()
print("load model time: ", end_time - start_time)

# print free gpu memory
print(torch.cuda.memory_allocated() / MB)

start_time = time.perf_counter()
model.to("cuda")
end_time = time.perf_counter()
print("load model time: ", end_time - start_time)

model.to("cpu")

start_time = time.perf_counter()
model.to("cuda")
end_time = time.perf_counter()
print("load model time: ", end_time - start_time)

# print free gpu memory
print(torch.cuda.memory_allocated() / MB)

hidden_states = torch.randn(1, 128, 768).to("cuda")

print(torch.cuda.memory_allocated() / MB)

with torch.no_grad():
    while True:
        model(hidden_states)

# print free gpu memory
print(torch.cuda.memory_allocated() / MB)

