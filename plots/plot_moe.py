from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

data = [
    {"name": "V-MoE-H/14E2", "parameters": 7.2e9, "year": 2021.6, "type": "ViT", "dense": 0.632e9},
    {"name": "V-MoE-H/14L5", "parameters": 2.7e9, "year": 2021.6, "type": "ViT", "dense": 0.632e9},
    {"name": "V-MoE-L/16E2", "parameters": 3.4e9, "year": 2021.6, "type": "ViT", "dense": 0.307e9},
    # {"name": "GShard", "parameters": 600e9, "year": 2020},
    # {"name": "NLLB-200-MoE", "parameters": 54.4e9, "year": 2022.7, "type": "NLLB", "dense": 1.3e9},
    {"name": "switch-base-128", "parameters": 7e9, "year": 2021.1, "type": "Switch", "dense": 0.220e9},
    {"name": "switch-base-256", "parameters": 12e9, "year": 2021.1, "type": "Switch", "dense": 0.220e9},
    {"name": "switch-large-128", "parameters": 26e9, "year": 2021.1, "type": "Switch", "dense": 0.770e9},
    {"name": "switch-xxl-128", "parameters": 395e9, "year": 2021.1, "type": "Switch", "dense": 11.3e9},
    {"name": "switch-c-128", "parameters": 1571e9, "year": 2021.1, "type": "Switch"},
    {"name": "M6-32", "parameters": 1.4e9, "year": 2021.5, "type": "M6", "dense": 0.327e9},
    {"name": "M6-128", "parameters": 10.8e9, "year": 2021.5, "type": "M6", "dense": 0.654e9},
    {"name": "M6-512", "parameters": 103.2e9, "year": 2021.5, "type": "M6", "dense": 1.57e9},
    {"name": "M6-960", "parameters":  1002.7e9, "year": 2021.5, "type": "M6", "dense": 10e9},
]

df = pd.DataFrame(data)

# convert parameters to TB
df["parameters"] = df["parameters"] / 1e12 * 4

# sns scatter plot year vs parameters with name label beside each point
sns.scatterplot(x="type", y="parameters", data=df, s=200)
plt.xlabel("")
plt.ylabel("Parameters (TB)")
# plt.yscale("log")
# for line in range(0, df.shape[0]):
#     plt.text(
#         df.year[line] + 0.2,
#         df.parameters[line],
#         df.name[line],
#         horizontalalignment="left",
#         size="medium",
#         color="black",
#         weight="semibold",
#     )

plt.savefig("plots/moe.png", bbox_inches="tight")