import re

log_path = "/mnt/xly/swap-engine/docker.log"
with open(log_path, "r") as f:
    lines = f.readlines()

wait_time = []
for line in lines:
    if "wait time:" in line:
        wait_time.append(float(re.findall(r"wait time: (\d+)", line)[0]))

wait_time.sort()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# plot CDF

# remove 95% outliers from wait_time

wait_time = np.array(wait_time)
wait_time = wait_time[wait_time < np.percentile(wait_time, 95)]

wait_time = wait_time / 1000 # convert to ms

plt.figure()
sns.set_style("whitegrid")
sns.ecdfplot(wait_time, color="red")

# draw vertial line at 54ms
plt.axvline(x=20, color="blue", linestyle="--")

plt.xlabel("wait time (ms)")
plt.ylabel("CDF")
plt.savefig("plots/wait_time_cdf.png")
