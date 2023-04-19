from dataclasses import dataclass, field
from transformers import SwitchTransformersConfig, HfArgumentParser
import os


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name of the model from HuggingFace"})
    num_gpu: int = field(default=1, metadata={"help": "Number of GPUs to use"})

    def __post_init__(self):
        self.model_repo = "_".join(["model_repo", self.model_name])

        self.gpu_ids = [i for i in range(self.num_gpu)]


parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]


g_index = 0


def get_gid():
    global g_index
    g_index += 1
    return args.gpu_ids[g_index % len(args.gpu_ids)]


def generate_preaggregate_config(num_experts, model_name, layer_name, layer_idx):
    code = """
import triton_python_backend_utils as pb_utils
import numpy as np
import os
import hashlib
import tritonclient
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from multiprocessing import shared_memory

class TritonPythonModel:
    def initialize(self, args):
        print('Initialized...')

        self.backend_name = "%s"
        self.num_experts = %d
        self.layer_name =  "%s"
        self.layer_idx = %d

        self.client = grpcclient.InferenceServerClient(
            url="localhost:50051", verbose=False
        )

        self.data_path = f"/opt/data/{self.backend_name}"
        try:
            os.mkdir(os.path.dirname(self.data_path))
            os.mkdir(self.data_path)
        except:
            pass

        try:
            self.shm = shared_memory.SharedMemory(name="expert_count", create=True, size=4)
            # set the value of the shared memory to 0
            buffer[:4] = bytearray([0, 0, 0, 0])
        except:
            self.shm = shared_memory.SharedMemory(name="expert_count")
    
    def dummy_callback(self, result, error):
        pass

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
            routes = pb_utils.get_input_tensor_by_name(request, "routes").as_numpy()
            # print("routes", routes, flush=True)
            hidden_states = pb_utils.get_input_tensor_by_name(request, "hidden_states").as_numpy()
            forwarded_states = pb_utils.get_input_tensor_by_name(request, "forwarded_states").as_numpy()
            # print("hidden_states", hidden_states, flush=True)
            route_prob_max = pb_utils.get_input_tensor_by_name(request, "route_prob_max").as_numpy()
            # print("route_prob_max", route_prob_max, flush=True)
            batch_size, seq_len, d_model = hidden_states.shape

            activation_cnt = 0
            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                if np.any(indexes_list):
                    activation_cnt += 1

            # write the number of active experts to the shared memory 4 bytes uint32
            self.shm.buf[:4] = activation_cnt.to_bytes(4, byteorder='little')
        
            expert_outputs = [None] * self.num_experts
            model_name = f"expert-{self.layer_name}-{self.layer_idx}"
            sequence_id = request.correlation_id()
            request_id = request.request_id()

            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                if np.any(indexes_list):
                    token_features = forwarded_states[indexes_list]
                    token_features = self.prepare_input("hidden_states", token_features)
                    output = self.prepare_output("hidden_states")
                    expert_outputs[i] = self.client.infer(
                        f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}",
                        [token_features],
                        outputs=[output],
                        request_id=request_id,
                        sequence_id=(sequence_id & 0xFFFFFFFF) | ((i+1) << 32),
                    )

                    output = expert_outputs[i].as_numpy("hidden_states")
                    forwarded_states[indexes_list] = output

            # for i in range(self.num_experts):
            #     indexes_list = routes[:, :, i].astype(bool)
            #     if np.any(indexes_list):
            #         output = expert_outputs[i].as_numpy("hidden_states")
            #         indexes_list = routes[:, :, i].astype(bool)
            #         forwarded_states[indexes_list] = output

            hidden_states = hidden_states + route_prob_max * forwarded_states

            activation_cnt = 0
            self.shm.buf[:4] = activation_cnt.to_bytes(4, byteorder='little')
            
            out_tensor = pb_utils.Tensor("hidden_states", hidden_states)
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

    def finalize(self):
        print('Cleaning up...')
    
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
    """ % (
            model_name
        )

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
