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
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"model": "Cascade", "variant": "DeepSpeed", "throughput": 3},
        {"model": "Cascade", "variant": "SwapAdvisor", "throughput": 3},
        {"model": "Cascade", "variant": "Ours", "throughput": 3},
        {"model": "MoE", "variant": "DeepSpeed", "throughput": 5},
        {"model": "MoE", "variant": "SwapAdvisor", "throughput": 5},
        {"model": "MoE", "variant": "Ours", "throughput": 5},
    ]
)

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])

plt.figure(figsize=(30, 10), dpi=300)
ax = sns.barplot(x="model", y="throughput", hue="variant", data=df, palette=palette)

for i, bar in enumerate(ax.patches):
    bar.set_edgecolor('k')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)
plt.ylabel("Throughput (R/s)", fontsize=48)
plt.legend(bbox_to_anchor=[0, axbox.y0+0.2,1,1], loc='upper center', ncol=4, fontsize=48)

plt.savefig(f"plots/{os.path.basename(__file__)}.png", bbox_inches="tight")