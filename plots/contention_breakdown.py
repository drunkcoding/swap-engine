import itertools
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

font = {"size": 48}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

colors = ["#afd5ff", "#006dc1", "#003671"]
colors = [colors[0]]
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"type": "None", "latency": 64, "tail": 64},
        {"type": "Layer", "latency": 44, "tail": 44},
        {"type": "LL", "latency": 33, "tail": 33},
        {"type": "Layer+LL", "latency": 23, "tail": 23},
        {"type": "Priority", "latency": 1.5, "tail": 1.5},
    ]
)
plt.figure(figsize=(20, 10), dpi=300)
ax = sns.barplot(x="type", y="latency", data=df, palette=palette)

for i, bar in enumerate(ax.patches):
    bar.set_edgecolor('k')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=48, rotation=15)
plt.yticks(fontsize=48)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)
plt.ylabel("Latency (ms)", fontsize=48)
# plt.legend(bbox_to_anchor=[0, axbox.y0+0.2,1,1], loc='upper center', ncol=4, fontsize=48)

plt.savefig(f"plots/{os.path.basename(__file__)}.pdf", bbox_inches="tight")