from dataclasses import dataclass, field
import re
import numpy as np
from transformers import HfArgumentParser
import plotly.express as px
import pandas as pd

COLORS = px.colors.qualitative.D3


def hex_to_rgba(h, alpha):
    """
    converts color value in hex format to rgba format with alpha transparency
    """
    return tuple([int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] + [alpha])


@dataclass
class Arguments:
    log_file: str = field(default="token.log")


parser = HfArgumentParser((Arguments,))
args = parser.parse_args_into_dataclasses()[0]

with open(args.log_file) as f:
    data = f.readlines()

tokens = []
all_tokens = []
record_flag = False
count = 0
for _, line in enumerate(data):
    if "batch start" in line:
        record_flag = True

    if record_flag:
        tokens.append(line)

    if "batch end" in line:
        record_flag = False
        lines = "\n".join(tokens)
        count += 1
        if count > 300:
            break
        groups = re.findall(r"tensor\(\[([0-9, \n\t]*)\]\)", lines)
        
        tokens = [
            list(map(lambda x: int(x.strip()), t.strip().split(","))) for t in groups
        ]
        tokens = np.array(tokens)
        tokens = tokens.T
        # tokens = tokens[tokens[:, 0] != 70, :]
        all_tokens.append(tokens)

        print(tokens.shape)
        tokens = []

all_tokens = np.concatenate(all_tokens, axis=0)
print(all_tokens.shape)

df = pd.DataFrame(all_tokens, columns=[str(i) for i in range(all_tokens.shape[1])])
df = df.sample(200)
print(df)
print(df.info())
fig = px.parallel_categories(
    df,
    # dimensions=[str(i) for i in range(all_tokens.shape[1])],
    # color= "rgba" + str(hex_to_rgba(h=COLORS[0], alpha=0.25)),
)
fig.update_traces(
    dimensions=[
        {"categoryorder": "category descending"} for _ in range(all_tokens.shape[1])
    ]
)
fig.show()
# fig.write_image("plots/moe_cond_prob.png")
