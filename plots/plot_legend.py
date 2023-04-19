import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 36})
# colors = ["#e6f2ff", "#80bdff", "#007bff", "#003e80"]
colors = ["#80bdff", "#007bff", "#003e80"]
data = [
    {"Model": "DeepSpeed(SSD)", "Dataset": "GLUE", "Latency": 0.1},
    {"Model": "DeepSpeed(SSD)", "Dataset": "SQuAD", "Latency": 0.2},
    {"Model": "DeepSpeed(SSD)", "Dataset": "SuperGLUE", "Latency": 0.3},
    {"Model": "DeepSpeed(DRAM)", "Dataset": "GLUE", "Latency": 0.4},
    {"Model": "DeepSpeed(DRAM)", "Dataset": "SQuAD", "Latency": 0.5},
    {"Model": "DeepSpeed(DRAM)", "Dataset": "SuperGLUE", "Latency": 0.6},
    # {"Model": "CUDA-UM(DRAM)", "Dataset": "GLUE", "Latency": 0.7},
    # {"Model": "CUDA-UM(DRAM)", "Dataset": "SQuAD", "Latency": 0.8},
    # {"Model": "CUDA-UM(DRAM)", "Dataset": "SuperGLUE", "Latency": 0.9},
    {"Model": "Archer(DRAM+SSD)", "Dataset": "GLUE", "Latency": 1.0},
    {"Model": "Archer(DRAM+SSD)", "Dataset": "SQuAD", "Latency": 1.1},
    {"Model": "Archer(DRAM+SSD)", "Dataset": "SuperGLUE", "Latency": 1.2},
]

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")

#plot barplot

# ax = sns.barplot(x="Dataset", y="Latency", hue="Model", data=df, palette=colors)
plt.plot([1,1], [.5,.5], color="#80bdff", linestyle='solid', linewidth=5, label="DeepSpeed(SSD)", marker="o", markersize=24, markerfacecolor="#80bdff", markeredgecolor="black")
plt.plot([1,1], [.5,.5], color="#007bff", linestyle='solid', linewidth=5, label="DeepSpeed(DRAM)", marker="^", markersize=24, markerfacecolor="#007bff", markeredgecolor="black")
plt.plot([1,1], [.5,.5], color="#003e80", linestyle='solid', linewidth=5, label="Archer(DRAM+SSD)", marker="D", markersize=24, markerfacecolor="#003e80", markeredgecolor="black")
# add black box around bars
# for i,thisbar in enumerate(ax.patches):
#     thisbar.set_edgecolor('k')


def export_legend(legend, filename="plots/legend_scale.pdf", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
# axbox = ax.get_position()
legend = plt.legend(bbox_to_anchor=[100, 0,1,1], loc='upper right', ncol=3, fontsize=48)
export_legend(legend)

# ax.set_xlabel("Dataset")
# ax.set_ylabel("Latency (s)")
# ax.set_ylim(0, 1.5)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
# plt.savefig("plots/legend.pdf", bbox_inches='tight')