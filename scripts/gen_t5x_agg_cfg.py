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
import tritonclient
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # (batch_size, sequence_length, num_expert)
        inputs = [
            {
                'name': "hidden_states",
                'data_type': 'TYPE_FP32',
                'dims': [ -1, -1, -1 ],
            },
            {
                'name': "forwarded_states",
                'data_type': 'TYPE_FP32',
                'dims': [ -1, -1, -1 ],
            },
            {
                'name': "routes",
                'data_type': 'TYPE_INT64',
                'dims': [ -1 , -1, -1 ],
            },
            {
                'name': "route_prob_max",
                'data_type': 'TYPE_FP32',
                'dims': [ -1 , -1, -1 ],
            }
        ]
        outputs = [{
            'name': 'hidden_states',
            'data_type': 'TYPE_FP32',
            'dims': [ -1, -1, -1 ]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        print('Initialized...')

        self.backend_name = "%s"
        self.num_experts = %d
        self.layer_name =  "%s"
        self.layer_idx = %d

        self.client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=False, concurrency=self.num_experts
        )

        self.data_path = f"/opt/data/{self.backend_name}"
        try:
            os.mkdir(os.path.dirname(self.data_path))
            os.mkdir(self.data_path)
        except:
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
            forwarded_states = pb_utils.get_input_tensor_by_name(request, "forwarded_states").as_numpy()
            # print("forwarded_states", forwarded_states, flush=True)
            # print("routes", routes, flush=True)
            hidden_states = pb_utils.get_input_tensor_by_name(request, "hidden_states").as_numpy()
            # print("hidden_states", hidden_states, flush=True)
            route_prob_max = pb_utils.get_input_tensor_by_name(request, "route_prob_max").as_numpy()
            # print("route_prob_max", route_prob_max, flush=True)
            batch_size, seq_len, d_model = hidden_states.shape
        
        
            request_id = request.request_id()
            req_id_md5 = hashlib.md5(str(request_id).encode()).hexdigest()
            np.save(f"{self.data_path}/routes_{self.layer_name}_{self.layer_idx}_{req_id_md5}", routes, allow_pickle=False)
            # np.save(f"{self.data_path}/hidden_states_{self.layer_name}_{self.layer_idx}_{req_id_md5}", hidden_states, allow_pickle=False)
            # np.save(f"{self.data_path}/route_prob_max_{self.layer_name}_{self.layer_idx}_{req_id_md5}", route_prob_max, allow_pickle=False)
            
            expert_outputs = [None] * self.num_experts
            model_name = f"expert-{self.layer_name}-{self.layer_idx}"
            sequence_id = request.correlation_id()


            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                if np.any(indexes_list):
                    token_features = forwarded_states[indexes_list]
                    # print("token_features", token_features, flush=True)
                    token_features = self.prepare_input("hidden_states", token_features)
                    # expert_routes = self.prepare_input("routes", routes)
                    output = self.prepare_output("hidden_states")
                    # print("output", output, flush=True)
                    expert_outputs[i] = self.client.async_infer(
                        f"{self.backend_name}_{self.layer_name}_expert_{self.layer_idx}_{i}",
                        [token_features],
                        outputs=[output],
                        request_id=request_id,
                        sequence_id=(sequence_id & 0xFFFFFFFF) | ((i+1) << 32),
                    )

            for i in range(self.num_experts):
                indexes_list = routes[:, :, i].astype(bool)
                if np.any(indexes_list):
                    output = expert_outputs[i].get_result()
                    output = output.as_numpy("hidden_states")
                    forwarded_states[indexes_list] = output
                    # print("output", i, forwarded_states[indexes_list], flush=True)

            hidden_states = hidden_states + route_prob_max * forwarded_states

            # print(hidden_states, flush=True)
            
            out_tensor = pb_utils.Tensor("hidden_states", hidden_states)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

            # print(responses, flush=True)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def prepare_input(self, name: str, input: np.ndarray):
        triton_input = httpclient.InferInput(
            name,
            input.shape,
            tritonclient.utils.np_to_triton_dtype(input.dtype),
        )
        triton_input.set_data_from_numpy(input)
        return triton_input

    def prepare_output(self, name: str):
        triton_output = httpclient.InferRequestedOutput(
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
