from dataclasses import dataclass, field
from transformers import SwitchTransformersConfig, HfArgumentParser
import os


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})

    def __post_init__(self):
        self.model_repo = "_".join(["model_repo", self.model_name])


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]


def generate_preaggregate_config(num_experts, model_name, layer_name, layer_idx):
    code = """
import triton_python_backend_utils as pb_utils
import numpy as np
import os
import hashlib
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import tritonclient
from functools import partial
from tritonclient import utils
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from multiprocessing import shared_memory
import tritonclient.utils.cuda_shared_memory as cudashm


class TritonPythonModel:
    def initialize(self, args):
        print("Initialized...")

        self.backend_name = "%s"
        self.num_experts = %d
        self.layer_name = "%s"
        self.layer_idx = %d

        self.client = grpcclient.InferenceServerClient(
            url="localhost:60051", verbose=False
        )

        self.data_path = f"/opt/data/{self.backend_name}"
        try:
            os.mkdir(os.path.dirname(self.data_path))
            os.mkdir(self.data_path)
        except:
            pass

        self.device = "cuda:" + args["model_instance_device_id"]

        try:
            self.shm = shared_memory.SharedMemory(
                name="expert_count", create=True, size=4
            )
            # set the value of the shared memory to 0
            self.shm.buf[:4] = bytearray([0, 0, 0, 0])
        except:
            self.shm = shared_memory.SharedMemory(name="expert_count")


    def execute(self, requests):
        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            routes = self.parse_input(request, "routes")
            forwarded_states = self.parse_input(request, "forwarded_states")

            request_id = request.request_id()
            # req_id_md5 = hashlib.md5(str(request_id).encode()).hexdigest()
            # np.save(f"{self.data_path}/routes_{self.layer_name}_{self.layer_idx}_{req_id_md5}", routes, allow_pickle=False)
            # np.save(f"{self.data_path}/hidden_states_{self.layer_name}_{self.layer_idx}_{req_id_md5}", hidden_states, allow_pickle=False)
            # np.save(f"{self.data_path}/route_prob_max_{self.layer_name}_{self.layer_idx}_{req_id_md5}", route_prob_max, allow_pickle=False)

            expert_outputs = [None] * self.num_experts
            sequence_id = request.correlation_id()

            def callback(forwarded_states, outputs, index, result, error):
                if error:
                    print(f"error: {error}", flush=True)
                else:
                    forwarded_states[index] = result.as_numpy("hidden_states")
                    outputs.append(0)

            activation_cnt = 0
            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                if np.any(indexes_list):
                    activation_cnt += 1

            # write the number of active experts to the shared memory 4 bytes uint32
            self.shm.buf[:4] = activation_cnt.to_bytes(4, byteorder="little")

            outputs = []
            activation_cnt = 0
            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                self.expert_name = (
                    f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}"
                )
                if np.any(indexes_list):
                    token_features = forwarded_states[indexes_list]
                    output = self.prepare_output("hidden_states")
                    token_features = self.prepare_input("hidden_states", token_features)
                    expert_outputs[i] = self.client.async_infer(
                        f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}",
                        [token_features],
                        partial(callback, forwarded_states, outputs, indexes_list),
                        outputs=[output],
                        request_id=request_id,
                        sequence_id=(sequence_id & 0xFFFFFFFF) | ((i + 1) << 32),
                    )
                    activation_cnt += 1

            hidden_states = self.parse_input(request, "hidden_states")
            route_prob_max = self.parse_input(request, "route_prob_max")

            while len(outputs) < activation_cnt:
                pass

            activation_cnt = 0
            self.shm.buf[:4] = activation_cnt.to_bytes(4, byteorder="little")

            # hidden_states = torch.Tensor(hidden_states).to("cuda:0")
            # route_prob_max = torch.Tensor(route_prob_max).to("cuda:0")
            # forwarded_states = torch.Tensor(forwarded_states).to("cuda:0")
            hidden_states = hidden_states + route_prob_max * forwarded_states
            # hidden_states = hidden_states.detach().cpu().numpy()
            # torch.cuda.empty_cache()

            out_tensor = self.parse_output(hidden_states, "hidden_states")
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

            # print(responses, flush=True)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def prepare_input(self, name: str, input: np.ndarray):
        triton_input = grpcclient.InferInput(
            name,
            input.shape,
            tritonclient.utils.np_to_triton_dtype(input.dtype),
        )
        triton_input.set_data_from_numpy(input)
        return triton_input

    def prepare_output(self, name: str):
        triton_output = grpcclient.InferRequestedOutput(
            name,
        )
        return triton_output
    
    def parse_input(self, request, field):
        input = pb_utils.get_input_tensor_by_name(request, field).as_numpy()
        return input

    def parse_output(self, output, field):
        return pb_utils.Tensor(field, output)

    def finalize(self):
        print("Cleaning up...")

    """ % (
        args.model_name,
        num_experts,
        layer_name,
        layer_idx,
    )

    model_path = os.path.join(args.model_repo, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "0", "model.py"), "w") as f:
        f.write(code)

    config = """
name: "%s"
backend: "python"
input [
    {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [-1, -1, -1]
    },
    {
        name: "forwarded_states"
        data_type: TYPE_FP32
        dims: [-1, -1, -1]
    },
    {
        name: "routes"
        data_type: TYPE_INT64
        dims: [-1, -1, -1]
    },
    {
        name: "route_prob_max"
        data_type: TYPE_FP32
        dims: [-1, -1, -1]
    }
]
output [
    {
        name: "hidden_states"
        data_type: TYPE_FP32
        dims: [-1, -1, -1]
    }
]
instance_group [
    {
    kind: KIND_CPU
    count: 1
    }
]
    """ % (model_name)

    with open(os.path.join(model_path, "config.pbtxt"), "w") as f:
        f.write(config)


config = SwitchTransformersConfig.from_pretrained(f"google/{args.model_name}")
print(config)
for layer_idx in range(config.num_layers):
    # code generation
    if layer_idx % 2 == 1:
        generate_preaggregate_config(
            config.num_experts,
            "%s_encoder_preagg_%d" % (args.model_name, layer_idx),
            "encoder",
            layer_idx,
        )
        generate_preaggregate_config(
            config.num_experts,
            "%s_decoder_preagg_%d" % (args.model_name, layer_idx),
            "decoder",
            layer_idx,
        )
