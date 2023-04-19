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

config = SwitchTransformersConfig.from_pretrained(args.model_name)

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
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # (batch_size, sequence_length, num_expert)
        inputs = [
            {
                "name": "hidden_states",
                "data_type": "TYPE_FP32",
                "dims": [-1, -1, -1],
            },
            {
                "name": "forwarded_states",
                "data_type": "TYPE_FP32",
                "dims": [-1, -1, -1],
            },
            {
                "name": "routes",
                "data_type": "TYPE_INT64",
                "dims": [-1, -1, -1],
            },
            {
                "name": "route_prob_max",
                "data_type": "TYPE_FP32",
                "dims": [-1, -1, -1],
            },
        ]
        outputs = [
            {"name": "hidden_states", "data_type": "TYPE_FP32", "dims": [-1, -1, -1]}
        ]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        print("Initialized...")

        self.backend_name = "%s"
        self.num_experts = %d
        self.layer_name = "%s"
        self.layer_idx = %d

        self.layer_str = (
            f"{self.backend_name}_{self.layer_name}_{self.layer_idx}"
        )

        self.client = grpcclient.InferenceServerClient(
            url="localhost:50051", verbose=False
        )
        self.client.unregister_system_shared_memory()
        self.client.unregister_cuda_shared_memory()

        # size of fp32 * 128 * 1 * hidden_size
        self.input_byte_size = 4 * 128 * 1 * %d
        self.output_byte_size = 4 * 128 * 1 * %d

        self.shm_name_input = f"{self.layer_str}_input"
        self.shm_handle_input = cudashm.create_shared_memory_region(shm_name_input, input_byte_size, 0)
        self.client.register_cuda_shared_memory(
            shm_name_input, cudashm.get_raw_handle(self.shm_handle_input), 0, input_byte_size
        )

        self.shm_name_output = f"{self.layer_str}_output"
        self.shm_handle_output = cudashm.create_shared_memory_region(shm_name_output, output_byte_size, 0)
        self.client.register_cuda_shared_memory(
            shm_name_output, cudashm.get_raw_handle(self.shm_handle_output), 0, output_byte_size
        )

        self.data_path = f"/opt/data/{self.backend_name}"
        try:
            os.mkdir(os.path.dirname(self.data_path))
            os.mkdir(self.data_path)
        except:
            pass

        self.device = "cuda:" + args["model_instance_device_id"]
        self.offsets = [0]

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

            expert_outputs = [None] * self.num_experts
            sequence_id = request.correlation_id()

            def shm_callback(forwarded_states, outputs, index, name, result, error):
                if error:
                    print(f"error: {error}", flush=True)
                else:
                    hidden_states = result.get_output("hidden_states")
                    hidden_states = cudashm.get_contents_as_numpy(
                        self.shm_handles[name],
                        utils.triton_to_np_dtype(hidden_states.datatype),
                        hidden_states.shape,
                    )
                    forwarded_states[index] = hidden_states
                    outputs.append(0)

            outputs = []
            activation_cnt = 0
            offset = 0
            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                self.expert_name = (
                    f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}"
                )
                if np.any(indexes_list):
                    token_features = forwarded_states[indexes_list]
                    offset += token_features.nbytes
                    self.offsets.append(offset)
                    output = self.prepare_shm_output("hidden_states", token_features)
                    token_features = self.prepare_shm_input("hidden_states", token_features)
                    expert_outputs[i] = self.client.async_infer(
                        f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}",
                        [token_features],
                        partial(shm_callback, forwarded_states, outputs, indexes_list, f"{self.expert_name}_output"),
                        outputs=[output],
                        request_id=request_id,
                        sequence_id=(sequence_id & 0xFFFFFFFF) | ((i + 1) << 32),
                    )
                    activation_cnt += 1

            hidden_states = self.parse_input(request, "hidden_states")
            route_prob_max = self.parse_input(request, "route_prob_max")

            while len(outputs) < activation_cnt:
                pass

            # move to GPU
            hidden_states = torch.Tensor(hidden_states).to("cuda:0")
            route_prob_max = torch.Tensor(route_prob_max).to("cuda:0")
            forwarded_states = torch.Tensor(forwarded_states).to("cuda:0")
            hidden_states = hidden_states + route_prob_max * forwarded_states
            hidden_states = hidden_states.detach().cpu().numpy()

            out_tensor = self.parse_output(hidden_states, "hidden_states")
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        self.offsets = [0]

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def prepare_shm_input(self, name: str, input: np.ndarray):
        input_byte_size = input.nbytes
        cudashm.set_shared_memory_region(self.shm_handle_input, [input], offset=self.offsets[-2])
        triton_input = grpcclient.InferInput(
            name,
            input.shape,
            tritonclient.utils.np_to_triton_dtype(input.dtype),
        )
        triton_input.set_shared_memory(shm_name_input, input_byte_size, offset=self.offsets[-2])

        return triton_input

    def prepare_shm_output(self, name: str, output: np.ndarray):
        output_byte_size = output.nbytes
        cudashm.set_shared_memory_region(self.shm_handle_output, [output], offset=self.offsets[-2])
        triton_output = grpcclient.InferRequestedOutput(
            name,
        )
        triton_output.set_shared_memory(self.shm_name_output, output_byte_size, offset=self.offsets[-2])
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
        config.d_model,
    )

    model_path = os.path.join(args.model_repo, model_name)
    make_dir_if_not_exists(model_path)
    make_dir_if_not_exists(os.path.join(model_path, "0"))
    with open(os.path.join(model_path, "0", "model.py"), "w") as f:
        f.write(code)


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
