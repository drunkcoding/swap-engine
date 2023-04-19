import numpy as np
import pynvml
import time

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Total memory: {} MB".format(info.total / 1024 / 1024))
print("Free memory: {} MB".format(info.free / 1024 / 1024))
print("Used memory: {} MB".format(info.used / 1024 / 1024))

# get utilization from pynvml

utils = []
while True:
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    # print("GPU Utilization: {} %".format(utilization.gpu))
    # print("Memory Utilization: {} %".format(utilization.memory))

    utils.append(utilization.gpu + utilization.memory)
    np.save("tests/python/utilization.npy", utils, allow_pickle=False)

    print("max utilization: {} %".format(np.max(utils)))



