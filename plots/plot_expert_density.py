import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from glob import glob
from collections import Counter

import matplotlib

# matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 72


dirs = glob("tests/python/runs/nllb_*/*.npy")
# dirs = glob("top_1_mask_*.npy")

routes = []
data = []
for npy in dirs:
    array = np.load(npy, allow_pickle=False)
    array = array.squeeze()
    # print(array.shape)
    if len(array) <= 1:
        continue

    num_token = array.shape[0]
    # print(array)
    array = np.argwhere(array == 1)

    # print(array.shape)
    # print(array)

    expert_ids = array[:, -1].flatten()

    total_experts = len(np.unique(expert_ids)) / 128
    data.append({"num_token": num_token, "expert_ratio": total_experts})

df = pd.DataFrame(data)

print(df.tail())

print(df["expert_ratio"].max())
mean = df["expert_ratio"].mean()
print(df["expert_ratio"].mean())
print("num_token max", df["num_token"].max())
print("num_token min", df["num_token"].min())
print("num_token min", df["num_token"].mean())
print(np.percentile(df["num_token"], 99))
print(np.percentile(df["num_token"], 95))

# exit()

df = df[(df["expert_ratio"] <= 0.9) & (df["expert_ratio"] > 0.05)]
# df["expert_ratio"] = df["expert_ratio"] / df["expert_ratio"].max() * 0.46
# exit()
# plt.figure(figsize=(15, 15), dpi=200)
fig, ax1 = plt.subplots(figsize=(20, 20), dpi=200)
sns.lineplot(data=df, x="num_token", y="expert_ratio", label="Expert Ratio", ax=ax1, linewidth=3)
ax2 = ax1.twinx()
# edge color none to remove the black border
sns.histplot(data=df, x="num_token", bins=100, edgecolor=None, cbar=True, ax=ax2, alpha=0.4, label="Input Count")
ax1.set_ylabel("Expert Activation Ratio")
ax1.hlines(mean, 0, 128, colors="red", linestyles="dashed", linewidth=6)
ax1.set_yticks(np.arange(0, 1.05, 0.1))
ax2.set_ylabel("Number of Inputs")
ax2.set_yticks(range(0, 10001, 1000))
ax2.set_ylim(0, 10000)
ax1.set_ylim(0, 1)
ax2.set_yticklabels(["0", "1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", "9k", "10k"])
ax1.set_xlabel("Number of Tokens")
ax1.grid(True)
ax2.grid(True)

# plt.xticks(range(0, 129, 16))
# plt.xlim(0, 128)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.ylabel("Expert Activation Ratio")
# plt.xlabel("Number of Tokens")
plt.legend(loc="lower right")
plt.savefig("plots/expert_ratio_nllb.pdf", bbox_inches="tight")
plt.close()
