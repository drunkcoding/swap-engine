import re
import os
import scipy.stats as stats
import numpy as np

log_file = "beamwidth10.log"
with open(log_file, "r") as f:
    lines = f.readlines()

# remove all lines with ["fuser", "model_lifecycle"]
lines = [
    line
    for line in lines
    if "fuser" not in line and "model_lifecycle" not in line and "expert" in line
]

candidates_wait_lines = [
    line
    for line in lines
    if "candidates" in line
]

pattern = re.compile(r"candidates time: (\d+) us")
groups = [pattern.search(line) for line in candidates_wait_lines]
candidates_wait_times = [int(group.group(1)) for group in groups]
candidates_wait_times = np.array(candidates_wait_times)
# candidates_wait_times = candidates_wait_times / 1000.0
# remove 99% outliers
candidates_wait_times = candidates_wait_times[candidates_wait_times < np.percentile(candidates_wait_times, 95)]
print("candidates_wait_lines", len(candidates_wait_lines))
print("candidates_wait_times", len(candidates_wait_times))

print("candidates_wait_times", stats.describe(candidates_wait_times))
print("candidates_wait_times", np.mean(candidates_wait_times))
print("candidates_wait_times", np.median(candidates_wait_times))

candidates_wait_times = candidates_wait_times[-1000:]

# plot histogram
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams.update({"font.size": 36})

# read float from file
with open("test.txt", "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [float(line) for line in lines]
lines = np.array(lines)
lines = lines * 1e6 # convert to us

plt.figure(figsize=(15, 10))
sns.ecdfplot(candidates_wait_times, linewidth=3, color="red", label="Archer")
sns.ecdfplot(lines, linewidth=3, linestyle=(0, (5,10)), color="red", label="KV-Caching")
# sns.histplot(candidates_wait_times, kde=True, stat="probability", bins=200)
# plt.xlim(0, 1000)
# plt.ylim(0, 0.6)
plt.xlabel("Predictor Latency (us)")
# plt.ylabel("Probability")
plt.legend()
plt.savefig("plots/candidates_wait_times.pdf", bbox_inches="tight")

