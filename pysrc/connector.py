from abc import ABC, abstractmethod
import sys
from typing import Dict, List, Union
import uuid
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
import grpc

from protos.message_pb2_grpc import ModelInferenceStub
from protos.message_pb2 import (
    InferenceRequest,
    DataType,
    InferenceResponse,
    TensorProto,
)
from protos.pb_dtype import numpy_to_pb_dtype, pb_to_numpy_dtype

class BaseConnector(ABC):
    @abstractmethod
    def __init__(self, url: str="localhost:8001", verbose: bool=False, shm=False, prefetch=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def __del__(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, model_name: str, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray], session_id: str):
        raise NotImplementedError

    @staticmethod
    def prepare_inputs(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return inputs

class DeepspeedLocalConnector(BaseConnector):
    def __init__(self, url: str = "localhost:50051", verbose: bool = False, shm=False, prefetch=False) -> None:
        self.url = url
        self.channel = grpc.insecure_channel(url)
        self.stub = ModelInferenceStub(self.channel)

    def __del__(self):
        pass

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray], session_id: str, model_version=""):
        
        request = InferenceRequest()
        request.model_name = model_name
        request.model_version = model_version
        request.session_id = uuid.uuid4().hex

        for key, value in inputs.items():
            input = TensorProto()
            # value = value.numpy()
            input.name = key
            input.dtype = numpy_to_pb_dtype(value.dtype)
            input.shape.extend(value.shape)
            input.data = value.tobytes()

            request.input_data.append(input)

        for key, value in outputs.items():
            output = TensorProto()
            output.name = key
            output.dtype = numpy_to_pb_dtype(value.dtype)
            output.shape.extend(value.shape)

            request.output_data.append(output)

        response = self.stub.InferenceHandle(request)
        outputs = {}
        for tensor_pb in response.output_data:
            tensor = np.frombuffer(tensor_pb.data, dtype=pb_to_numpy_dtype(tensor_pb.dtype))
            tensor = tensor.reshape(tensor_pb.shape)
            outputs[tensor_pb.name] = tensor
        return outputs

class TritonLocalConnector(BaseConnector):
    def __init__(self, url: str="localhost:8001", verbose: bool=False, shm=False, prefetch=False) -> None:
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)


        # To make sure no shared memory regions are registered with the server.
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

        self.shm_enabled = shm
        self.shm_registered = False
        self.prefetch = prefetch

        self.shm_handles = dict()
    

    def __del__(self):
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray], session_id: str):
        #TODO prefetch while inferencinga
        
        triton_inputs = [
            self._format_triton_input(arr, name)
            for name, arr in inputs.items()
        ]
        triton_outputs = [
            self._format_triton_output(arr, name)
            for name, arr in outputs.items()
        ]
        results = self.triton_client.infer(
            model_name, triton_inputs, outputs=triton_outputs, request_id=session_id
        )

        # TODO double copy is not good: 1) get_contents_as_numpy 2) assignment
        for name, arr in outputs.items():
            output = results.get_output(name) if self.shm_enabled else results.as_numpy(name)
            if output is not None:
                if self.shm_enabled:
                    outputs[name] = cudashm.get_contents_as_numpy(
                        self.shm_op_handles[name],
                        utils.triton_to_np_dtype(output.datatype),
                        arr.shape,
                    )
                else:
                    outputs[name] = output
            else:
                print("%s is missing in the response." % name)
                sys.exit(1)

        return outputs

    def _format_triton_input(self, input:np.ndarray, name:str):
        triton_input = grpcclient.InferInput(
            name,
            input.shape,
            utils.np_to_triton_dtype(input.dtype),
        )

        if self.shm_enabled:
            self._register_shm(name, input.nbytes, "cuda")
            triton_input.set_shared_memory(name, input.nbytes)
        else:
            triton_input.set_data_from_numpy(input)

        return triton_input

    def _format_triton_output(self, output :np.ndarray, name:str):
        triton_output = grpcclient.InferRequestedOutput(
            name,
        )

        if self.shm_enabled:
            self._register_shm(name, output.nbytes, "cuda")
            triton_output.set_shared_memory(name, output.nbytes)

        return triton_output

    def _register_shm(self, name, byte_size, shm_type):
        if name in self.shm_handles:
            return self.shm_handles[name]
        
        if shm_type == "cuda":
            shm_handle = cudashm.create_shared_memory_region(
                name, byte_size, 0
            )
            shm_handle = cudashm.register_shared_memory(
                name, cudashm.get_raw_handle(shm_handle), 0, byte_size
            )
        else:
            shm_handle = shm.create_shared_memory_region(
                name, byte_size, 0
            )
            shm_handle = shm.register_shared_memory(
                name, shm_handle, 0, byte_size
            )
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
