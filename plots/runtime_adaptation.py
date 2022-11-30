import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {"size": 48}
matplotlib.rc("font", **font)
matplotlib.rcParams["hatch.linewidth"] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

colors = ["#afd5ff", "#006dc1", "#003671"]
palette = sns.color_palette(colors)

fake_x = np.linspace(1, 100, endpoint=True)

df = pd.DataFrame(
    {"time_elapsed": fake_x, "latency": np.sin(fake_x), "throughput": np.cos(fake_x)}
)

plt.figure(figsize=(30, 10), dpi=300)
ax1 = sns.lineplot(x="time_elapsed", y="latency", data=df, palette=palette)
ax2 = ax1.twinx()
sns.lineplot(x="time_elapsed", y="throughput", data=df, palette=palette, ax=ax2)


# plt.yscale("log")

# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)

ax1.set_ylabel("Latency (ms)", fontsize=48)
ax1.set_xlabel("Time Elapsed (s)", fontsize=48)
ax2.set_ylabel("Throughput (R/s)", fontsize=48)

for label in ax1.xaxis.get_majorticklabels():
    label.set_fontsize(48)
for label in ax2.xaxis.get_majorticklabels():
    label.set_fontsize(48)
for label in ax1.yaxis.get_majorticklabels():
    label.set_fontsize(48)
for label in ax2.yaxis.get_majorticklabels():
    label.set_fontsize(48)

# plt.xticks(fontsize=48)
# plt.yticks(fontsize=48)

plt.savefig(f"plots/{os.path.basename(__file__)}.png", bbox_inches="tight")
