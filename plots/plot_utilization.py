import numpy as np
import matplotlib.pyplot as plt

# Path: plots/plot_utilization.py

data_path = "tests/python/utilization.npy"

data = np.load(data_path, allow_pickle=False)
print(len(data))
# get max number in data in a size 10 moving window 
max_data = np.zeros(len(data))
for i in range(len(data)):
    if i < 10:
        max_data[i] = max(data[:i+1])
    else:
        max_data[i] = max(data[i-10:i+1])



x = np.arange(len(max_data))
# plot utilization as time series
fig, ax = plt.subplots()
ax.plot(x, max_data, label="utilization")
plt.savefig("plots/utilization.png")

