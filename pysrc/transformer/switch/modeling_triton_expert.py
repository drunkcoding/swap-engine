import tritonclient
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
import uuid
import numpy as np
import torch

from .configuration_switch import SwitchConfig


class SwitchModelTriton:
    def __init__(self, config: SwitchConfig, layer_idx: int):
        self.backend_name = config.backend_name
        self.num_experts = config.num_experts
        self.layer_name = "decoder" if config.is_decoder else "encoder"
        self.layer_idx = layer_idx

        self.client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False
        )
        # self.client.start_stream()

        # clean up the shm
        self.client.unregister_system_shared_memory()
        self.client.unregister_cuda_shared_memory()

        self.shm_handles = {}
        # register the shared memory for all I/O tensors, assuming maximum 64 parallel requests for now

    def __del__(self):
        self.client.unregister_system_shared_memory()
        self.client.unregister_cuda_shared_memory()
        # self.client.stop_stream()
        self.client.close()

    def __call__(
        self,
        hidden_states: np.ndarray,
        routes: np.ndarray,
        route_prob_max: np.ndarray,
        request_id: str,
    ):
        expert_outputs = [None] * self.num_experts
        model_name = f"expert-{self.layer_name}-{self.layer_idx}"
        sequence_id = hash(model_name)
        for i in range(self.num_experts):
            indexes_list = np.flatnonzero(routes == self.expert_num)
            if len(indexes_list) > 0:
                token_features = hidden_states.view(-1, hidden_states.shape[-1])[
                    indexes_list
                ]
                token_features = self.prepare_input("hidden_states", token_features)
                output = self.prepare_output("hidden_states")
                expert_outputs[i] = self.client.async_infer(
                    "%s_%s_expert_%d_%d"
                    % (self.backend_name, self.layer_name, self.layer_idx, i),
                    [token_features],
                    [output],
                    request_id=request_id,
                    sequence_id=sequence_id,
                )

        batch_size, seq_len, d_model = hidden_states.shape
        final_output = np.zeros_like(hidden_states).reshape(
            (batch_size * seq_len, d_model)
        )
        for i in range(self.num_experts):
            if expert_outputs[i] is not None:
                output = expert_outputs[i].get_result()
                output = output.as_numpy("hidden_states")
                indexes_list = np.flatnonzero(routes == i)
                final_output[indexes_list] = output

        final_output = final_output * route_prob_max.reshape(-1, 1)
        final_output = final_output.reshape(batch_size, seq_len, d_model)
        hidden_states = hidden_states + final_output

        return hidden_states

    def prepare_input(self, name: str, input: np.ndarray):
        triton_input = grpcclient.InferInput(
            name,
            input.shape,
            tritonclient.utils.np_to_triton_dtype(input.dtype),
        )
        triton_input.set_data_from_numpy(input)

        return triton_input

    def prepare_output(self, name: str):
        triton_output = grpcclient.InferRequestedOutput(name)
        return triton_output

    def prepare_shm_input(self, name: str, input: np.ndarray):
        triton_input = grpcclient.InferInput(
            name,
            input.shape,
            tritonclient.utils.np_to_triton_dtype(input.dtype),
        )

        self._register_shm(name, input.nbytes, "cuda")
        triton_input.set_shared_memory(name, input.nbytes)

        return triton_input

    def prepare_shm_output(self, name: str, output: np.ndarray):
        triton_output = grpcclient.InferRequestedOutput(name)

        self._register_shm(name, output.nbytes, "cuda")
        triton_output.set_shared_memory(name, output.nbytes)

        return triton_output

    def _register_shm(self, name, byte_size, shm_type):
        if name in self.shm_handles:
            return self.shm_handles[name]

        if shm_type == "cuda":
            shm_handle = cudashm.create_shared_memory_region(name, byte_size, 0)
            shm_handle = cudashm.register_shared_memory(
                name, cudashm.get_raw_handle(shm_handle), 0, byte_size
            )
        else:
            shm_handle = shm.create_shared_memory_region(name, byte_size, 0)
            shm_handle = shm.register_shared_memory(name, shm_handle, 0, byte_size)
        self.shm_handles[name] = shm_handle
        return shm_handle

    def _unregister_shm(self, name, shm_type):
        if name in self.shm_handles:
            if shm_type == "cuda":
                cudashm.unregister_shared_memory(self.shm_handles[name])
                cudashm.destroy_shared_memory_region(self.shm_handles[name])
            else:
                shm.unregister_shared_memory(self.shm_handles[name])
                shm.destroy_shared_memory_region(self.shm_handles[name])
            del self.shm_handles[name]
