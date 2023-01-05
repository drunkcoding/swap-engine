import argparse
import os
import time
import tqdm

import numpy as np

from fetch_engines import StreamFetch, StreamFetchLRU, LFUFetch, LFUKFetch


def generate_routes(n_layer, n_expert):
  routes = np.random.choice([0, 1], size=(n_layer, n_expert))
  return routes


def generate_expert_states(n_layer, n_expert, mem_capacity):
  expert_states = np.zeros(shape=n_layer * n_expert, dtype=np.int32)
  expert_states[:mem_capacity] = 1
  return expert_states.reshape(n_layer, n_expert)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--route-path", type=str, default="")
  parser.add_argument("--n-layer", type=int, default=12)
  parser.add_argument("--n-expert", type=int, default=128)
  parser.add_argument("--mem-capacity", type=int, default=300)
  parser.add_argument("--n-request", type=int, default=2)
  parser.add_argument("--fetch-engine", type=str, default="StreamFetch")
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--k", type=int, default=1)
  args = parser.parse_args()
  return args


args = get_args()
fetch_engine = args.fetch_engine
stride = args.stride
n_layer = args.n_layer
n_expert = args.n_expert
mem_capacity = args.mem_capacity
filename_template = "routes_{}_{}_{}.npy"  # encoder/decoder, layer_index, request_uuid

log_file_format = f"{fetch_engine}_layer_{n_layer}_expert_{n_expert}_mem_{mem_capacity}_stride_{stride}_k_{args.k}_{time.time()}.log"
log_file = open(log_file_format, "w")

assert n_expert * n_layer > mem_capacity
assert n_layer > 0 and n_expert > 0

requests = []
if args.route_path == "":
  # random generate routes according to n_layer, n_expert, n_request
  n_request = args.n_request
  assert n_request > 0

  for i in range(n_request):
    routes = generate_routes(n_layer, n_expert)
    requests.append(routes)
  # print("requests:")
  # print(requests)
  # print("init expert states:")
  # print(expert_states)
else:
  # get all request uuids
  files = [filename for filename in os.listdir(args.route_path) if filename.endswith('.npy')]
  uuids = set()
  for file in files:
    uuids.add(file.split('_')[-1].split('.')[0])
  for uuid in uuids:
    # load encoder routes, layer id in range(1, n_layer * 2, 2)
    routes = []
    for layer_id in range(1, n_layer, 2):
      for layer_type in ["encoder", "decoder"]:
        filename = filename_template.format(layer_type, layer_id, uuid)
        route = np.load(os.path.join(args.route_path, filename))
        route = route.sum(axis=1).sum(axis=0)
        routes.append(route)
    routes = np.array(routes)
    assert routes.shape == (n_layer, n_expert)
    requests.append(routes)


expert_states = generate_expert_states(n_layer, n_expert, mem_capacity)
# different fetch engine

if fetch_engine == "StreamFetch":
  fetch_engine = StreamFetch(n_layer, n_expert, mem_capacity, expert_states, stride)
elif fetch_engine == "StreamFetchLRU":
  fetch_engine = StreamFetchLRU(n_layer, n_expert, mem_capacity, expert_states, stride)
elif fetch_engine == "LFUFetch":
  fetch_engine = LFUFetch(n_layer, n_expert, mem_capacity, expert_states, stride)
elif fetch_engine == "LFUKFetch":
  fetch_engine = LFUKFetch(n_layer, n_expert, mem_capacity, expert_states, stride, args.k)
else:
  raise NotImplementedError

n_hit = 0
n_req = 0
total_fetch = 0

for req_id, routes in enumerate(tqdm.tqdm(requests)):
  log_file.write(f"Request {req_id}:\n")
  for layer_id in range(n_layer):
    for expert_id in range(n_expert):
      if routes[layer_id][expert_id] > 0:
        n_req += 1
        # try to access expert i
        if fetch_engine.access(layer_id, expert_id):
          # print to log file
          log_file.write(f"\tL {layer_id} E {expert_id} is hit\n")
          n_hit += 1
        else:  # expert i not in GPU
          log_file.write(f"\tL {layer_id} E {expert_id} is fetched\n")
          n_fetched_expert = fetch_engine.fetch(layer_id, expert_id)
          assert fetch_engine.access(layer_id, expert_id)
          total_fetch += n_fetched_expert
      else:
        log_file.write(f"\tL {layer_id} E {expert_id} is not accessed\n")
  fetch_engine.global_update(routes)
print("hit rate:", n_hit/n_req)
print("total fetch:", total_fetch)