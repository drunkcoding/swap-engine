import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as st
from scipy import stats

log_path = "/mnt/raid0nvme1/xly/swap-engine/plots/prefetch/switch-base-128_mix_1.log"
# log_path = "/mnt/raid0nvme1/xly/swap-engine/docker.log"
with open(log_path, "r") as f:
    lines = f.readlines()

wait_time = []
for line in lines: 
    # if "device wait time:" in line and "expert" in line:
    #     wait_time.append(float(re.findall(r"device wait time: (\d+)", line)[0]))
    # if "move cost" in line:
    #     group = re.findall(r"\((\d+)MB\).*move cost (\d+)", line)[0]
    #     if int(group[1]) > 0 and int(group[0]) > 0:
    #         wait_time.append(int(group[0]) / int(group[1]))
    if "emplace time" in line and "expert" in line:
        wait_time.append(float(re.findall(r"emplace time (\d+)", line)[0]))
    # if "move cost" in line and "expert" in line:
    #     wait_time.append(float(re.findall(r"move cost (\d+)", line)[0]))
    #     if wait_time[-1] > 10000:
    #         print(line)
    # if "data time" in line and "expert" in line:
    #     wait_time.append(float(re.findall(r"data time: (\d+)", line)[0]))
    # if "priority: 0" in line and "time:" in line:
    #     wait_time.append(float(re.findall(r"time: (\d+)", line)[0]))

wait_time = np.array(wait_time)
print(wait_time.shape)

wait_time = wait_time[-2000:]
wait_time = wait_time / 1000 # convert to ms

lock_wait_time = []
for line in lines:
    if "lock wait time:" in line:
        lock_wait_time.append(float(re.findall(r"lock wait time: (\d+)", line)[0]))
lock_wait_time = np.array(lock_wait_time)

# wait_time.sort()

# wait_time = wait_time + lock_wait_time

# plot CDF

# remove 95% outliers from wait_time


# wait_time = wait_time[80000:]
# wait_time = wait_time[wait_time < np.percentile(wait_time, 99)]
# wait_time = wait_time[-2000:]
print("wait_time", wait_time, max(wait_time))
print(np.sum(wait_time < 1) / len(wait_time))
exit()

plt.figure()
sns.set_style("whitegrid")
sns.ecdfplot(wait_time, color="red")

# draw vertial line at 54ms
# plt.axvline(x=20, color="blue", linestyle="--")

plt.xlabel("wait time (ms)")
plt.ylabel("CDF")
plt.savefig("plots/wait_time_cdf.png")
