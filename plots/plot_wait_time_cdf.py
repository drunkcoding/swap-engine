import re
import os

def get_wait_time(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    wait_time = []
    for line in lines:
        if "expert" in line:
            if "device wait time" in line:
                t = float(re.findall(r"device wait time: (\d+)", line)[0])
            # if "emplace time" in line and "cpu->cuda" in line:
            #     t = float(re.findall(r"emplace time (\d+)", line)[0])
            # elif "time cost" in line:
            #     t = float(re.findall(r"time cost (\d+)", line)[0])
            else:
                continue
            wait_time.append(t)

    # print(wait_time)
    wait_time.sort()

    wait_time = np.array(wait_time)
    print(wait_time)
    wait_time = wait_time[-2000:]
    wait_time = wait_time[wait_time < np.percentile(wait_time, 99)]
    wait_time = wait_time / 1000  # convert to ms

    # if "deepspeed" in log_path:
    #     wait_time = wait_time / 5

    # wait_time = wait_time[wait_time < 500]

    return wait_time


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 48

# plot CDF

# remove 95% outliers from wait_time

model_name = "switch-large-128"

file_folder = os.path.dirname(os.path.realpath(__file__))

orbital_wait_time = get_wait_time(
    os.path.join(file_folder, f"prefetch/{model_name}_glue_1.log")
)
counter_wait_time = get_wait_time(
    os.path.join(file_folder, f"prefetch/{model_name}_glue_4.log")
    # os.path.join(file_folder, f"prefetch/{model_name}_glue_4.log")
)
# noise = np.random.normal(4, 2, len(counter_wait_time))
# noise[noise < 5] = 0
# counter_wait_time += noise
# deepspeed_wait_time = get_wait_time(
#     os.path.join(file_folder, f"deepspeed/{model_name}_super_glue_4.log")
# )
# counter_wait_time /= 8
# orbital_wait_time /= 4
plt.figure(figsize=(15, 10), dpi=200)
sns.set_style("whitegrid")
sns.ecdfplot(
    orbital_wait_time,
    color="red",
    linestyle="-",
    label="Archer(w/ Ordering)",
    linewidth=4,
)
sns.ecdfplot(
    counter_wait_time, color="red", linestyle="--", label="Archer(w/o Ordering)", linewidth=4
)
# sns.ecdfplot(
#     deepspeed_wait_time, color="red", linestyle=":", label="DS", linewidth=4
# )

# legend = ax.legend(loc="center left", frameon=True)
plt.legend(loc="center right", frameon=True, prop={"size": 48})

plt.xlabel("Expert Ready Latency (ms)")
plt.ylabel("")
# plt.xlim(0, 5000)
plt.ylim(0, 1)
plt.savefig(
    f"plots/wait_time_cdf_{model_name}.pdf", bbox_inches="tight", pad_inches=0.1
)
