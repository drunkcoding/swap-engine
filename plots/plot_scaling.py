import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


font = {"size": 56}
matplotlib.rc("font", **font)

cm = 1 / 2.54  # centimeters in inches
# sns.set(rc={"figure.figsize": (15 * cm, 10 * cm), "axes.labelsize": 36})
# sns.set(font_scale=2)
sns.set_theme(style="white")

# colors = ['#e6f2ff', '#80bdff', '#007bff', '#003e80']
# colors = ["#e6f2ff", "#80bdff", "#003e80"]
colors = ["#80bdff", "#007bff", "#003e80"]
system = ["DS(SSD)", "DS(CPU)", "Archer(CPU+SSD)"]
labels = ["DeepSpeed(SSD)", "DeepSpeed(DRAM)", "Archer(DRAM+SSD)"]
markers = ["o", "^", "D"]
system_colors = dict(zip(system, colors))
system_labels = dict(zip(system, labels))
system_markers = dict(zip(system, markers))
palette = sns.color_palette(colors)

df = pd.read_csv("plots/e2e_scaling.csv")

print(df)

models = df["model"].unique()
df["latency"] = df["latency"] / 1000
df = df[df.latency != 0]

for i, model in enumerate(models):
    df_single = df[df["model"] == model]

    # df_ds = df_single[df_single["system"] == "DS(SSD)"]
    # latency_max = df_ds["latency"].max()
    # latency_min = df_ds["latency"].min()
    # ratio_max = df_ds["ratio"].max()
    # ratio_min = df_ds["ratio"].min()

    # print(latency_max, latency_min, ratio_max, ratio_min)
    system = df_single["system"].unique()

    plt.figure(figsize=(12, 10))
    for sys in system:
        df_sys = df_single[df_single["system"] == sys]
        plt.plot(
            df_sys["ratio"],
            df_sys["latency"],
            label=system_labels[sys],
            color=system_colors[sys],
            linewidth=5,
            marker=system_markers[sys],
            markersize=24,
            markerfacecolor=system_colors[sys],
            markeredgecolor="black",
        )
    # save the legend to a file
    # plt.legend(loc="upper left", ncol=1, fontsize=24, facecolor="gray", framealpha=0.2)
    plt.ylabel("Latency (s)", fontsize=48)
    plt.xlabel("Out-of-core Ratio (%)", fontsize=48)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.grid(True, which="both", axis="both")
    plt.xlim(0, 101 )
    if model == "switch-large-128":
        plt.yticks(range(0,26,5))
    # ax.legend(fontsize=36)
    # plt.legend(loc='upper left', ncol=1, fontsize=24, facecolor="gray", framealpha=0.2)
    # plt.yscale("log")
    plt.savefig(f"plots/e2e_scaling_{model}.pdf", bbox_inches="tight")
    plt.close()

exit()


def size_func(x):
    if "switch-base-128" in x:
        return 1
    elif "switch-base-256" in x:
        return 2
    elif "switch-large-128" in x:
        return 4
    elif "switch-xxl-128" in x:
        return 10


df["size"] = df["model"].apply(size_func)
df["latency"] = df["latency"] / 1000
df = df[df.latency != 0]


fig, _ = plt.subplots(figsize=(30 * cm, 20 * cm))
ax = sns.scatterplot(
    x="ratio",
    y="latency",
    hue="system",
    size="model",
    sizes={
        "switch-base-128": 40,
        "switch-base-256": 100,
        "switch-large-128": 300,
        "switch-xxl-128": 1000,
    },
    alpha=0.9,
    palette=palette,
    data=df,
    edgecolor="black",
    linewidth=1.5,
    legend="brief",
)

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# axbox = ax.get_position()

sns.move_legend(ax, "upper left", ncol=1)
# sns.move_legend(ax, "upper right", fontsize=36)
plt.ylabel("Latency (s)", fontsize=36)
plt.xlabel("Out-of-core Ratio (%)", fontsize=36)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.grid(True, which="both", axis="both")
plt.xlim(0, 110)
# ax.legend(fontsize=36)
plt.legend(loc="upper left", ncol=1, fontsize=24, facecolor="gray", framealpha=0.2)
# plt.yscale("log")
plt.savefig("plots/e2e_scaling.pdf", bbox_inches="tight")
