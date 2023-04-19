import plotly.express as px
from glob import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

files = glob("tests/python/runs/switch_glue/*.npy")
files = sorted(files, key=os.path.getctime)
print(len(files))
routes = []
df_data = []

for _ in range(6):
    routes.append([])

layer_idx = 0
data_count = 0
for data in tqdm(files):
    array = np.load(data, allow_pickle=False)
    array = array.squeeze()

    # print(array)
    array = np.argwhere(array == 1)

    # print(array.shape)
    # print(array)

    expert_ids = array[:, -1].flatten()

    # if expert_ids length < 128, pad with 70
    if len(expert_ids) < 1000:
        expert_ids = np.pad(
            expert_ids, (0, 1000 - len(expert_ids)), "constant", constant_values=-1
        )
    # print(expert_ids.shape)
    routes[layer_idx].append(expert_ids.tolist())
    # df_data.append({"expert_ids": expert_ids.tolist()})
    layer_idx = (layer_idx + 1) % 6
    # data_count += 1

    # if data_count == 6600:
    #     break


routes = np.array(routes)
print(routes.shape)

for sample in range(routes.shape[1]):
    for e in range(routes.shape[2]):
        line = {
            "Layer" + str(layer_idx): routes[layer_idx, sample, e]
            for layer_idx in range(6)
            if routes[layer_idx, sample, e] != -1
        }
        df_data.append(line)
print(routes.shape)

df = pd.DataFrame(df_data)
print(df.shape)
df = df.dropna()

# concert all columns to int
for col in df.columns:
    df[col] = df[col].astype(int)
    df[col] = df[col].astype(str)
    # append E to the front of each value
    df[col] = "E" + df[col]

print(df.head())
print(df.shape)
# convert first column to int
# df["count"] = df["count"].astype(int)

from collections import Counter

layer0_counter = Counter(df["Layer0"].tolist())
layer1_counter = Counter(df["Layer1"].tolist())
layer2_counter = Counter(df["Layer2"].tolist())
# print(counter)
layer0_most_popular = layer0_counter.most_common(5)
layer1_most_popular = layer1_counter.most_common(5)
layer2_most_popular = layer2_counter.most_common(5)
print(layer0_most_popular)
print(layer1_most_popular)
print(layer2_most_popular)

df = df[(df["Layer0"] == "E81") & (df["Layer1"] == "E72")]
df = df.sort_values(by=["Layer2"])
print(df)
print(df.shape)

exit()


# filter = df["Layer1"].isin([x[0] for x in layer1_most_popular]) & (df["Layer1"] != 70)
# print(filter)
# expert_ids = list(most_popular.keys())
# df = pd.concat([df[df["Layer0"] == 89].sample(20), df[df["Layer0"] == 70].sample(20)])
print(df.shape)
df = df[["Layer0", "Layer1", "Layer2"]]
# df = df.iloc[:, :2]

# filter out the least popular experts
df = df[
    df["Layer0"].isin([x[0] for x in layer0_most_popular])
    & df["Layer1"].isin([x[0] for x in layer1_most_popular])
    & df["Layer2"].isin([x[0] for x in layer2_most_popular])
    & (df["Layer1"] != 70)
]


# change df column names

print(df.head())
print(df.shape)
# df = df.sample(50).reset_index(drop=True)

df = df.sort_values(by=["Layer1"])

highlighted_category = "E81"

# df.columns = ["Layer0</br></br>Most Frequent Expert", "Layer1\nMost Frequent Experts", "Layer2\nMost Frequent Experts"]


fig = px.parallel_categories(df)
fig.update_traces(
    line={
        "color": df["Layer0"].apply(
            lambda x: "#007bff" if x == highlighted_category else 'rgba(255, 255, 255, 0)'
        )
        # 'width': df['Layer1'].apply(lambda x: 2 if x == highlighted_category else 1)
    },
    hoveron="color",
    bundlecolors=True,
    labelfont={"size": 24, 'family': 'Times'},
    tickfont={"size": 24, 'family': 'Times'},
    dimensions=[{"categoryorder": "category ascending"} for k in range(3)],
)
fig.write_image("plots/parallel_categories.png")
# fig.show()
