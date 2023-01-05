# This script is used to test the simulator.
# The result is printed to stdout.
# The fetch process is printed to log files.
python simulator.py --route-path /mnt/xly/switch-base-128/ --stride 1 --fetch-engine StreamFetch  # stream fetch next 1
python simulator.py --route-path /mnt/xly/switch-base-128/ --stride 10 --fetch-engine StreamFetch  # stream fetch next 10
python simulator.py --route-path /mnt/xly/switch-base-128/ --stride 10 --fetch-engine StreamFetchLRU  # stream fetch next 10 with LRU
python simulator.py --route-path /mnt/xly/switch-base-128/ --stride 10 --fetch-engine LFUFetch  # LFU fetch next 10
python simulator.py --route-path /mnt/xly/switch-base-128/ --stride 10 --fetch-engine LFUKFetch --k 10  # LFU-K fetch next 10 with k=10