import re
import os

log_file = "plots/prefetch/switch-base-128_mix_1.log"
with open(log_file, "r") as f:
    lines = f.readlines()

# remove all lines with ["fuser", "model_lifecycle"]
lines = [
    line
    for line in lines
    if "fuser" not in line and "model_lifecycle" not in line and "expert" in line
]

lock_wait_lines = [
    line
    for line in lines
    if "lock wait" in line
]

emplace_gpu_time_lines = [
    line
    for line in lines
    if "emplace time" in line and "->cuda" in line
]

emplace_cpu_time_lines = [
    line
    for line in lines
    if "emplace time" in line and "lazy->cpu" in line
]

emplace_priority_time_lines = [
    line
    for line in emplace_gpu_time_lines
    if "priority: 0" in line
]

print("lock_wait_lines", len(lock_wait_lines))
print("emplace_gpu_time_lines", len(emplace_gpu_time_lines))
print("emplace_cpu_time_lines", len(emplace_cpu_time_lines))
print("emplace_priority_time_lines", len(emplace_priority_time_lines))

print("emplace_gpu_time_lines", emplace_gpu_time_lines[0])
# for i, line in enumerate(lines):
#     print(line)
#     if i > 100:
#         break
