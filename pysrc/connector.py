from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
from configparser import ConfigParser
import json
import os
from pyexpat import model
import sys
import time
from typing import Dict, List, Union
import uuid
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
import grpc
from tqdm import tqdm

from protos.message_pb2_grpc import ModelInferenceStub
from protos.message_pb2 import (
    InferenceRequest,
    DataType,
    InferenceResponse,
    TensorProto,
)
from protos.pb_dtype import numpy_to_pb_dtype, pb_to_numpy_dtype
from pyutils.timer import timeit


class BaseConnector(ABC):
    @abstractmethod
    def __init__(
        self,
        config: Dict,
        url: str = "localhost:8001",
        verbose: bool = False,
        shm=False,
        prefetch=False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __del__(self):
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        session_id: str,
    ):
        raise NotImplementedError

    @staticmethod
    def prepare_inputs(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return inputs


class DeepspeedLocalConnector(BaseConnector):
    def __init__(
        self,
        config: Dict,
        url: str = "localhost:50051",
        verbose: bool = False,
        shm=False,
        prefetch=False,
    ) -> None:
        self.url = url
        self.channel = grpc.insecure_channel(url)
        self.stub = ModelInferenceStub(self.channel)

    def __del__(self):
        pass

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        session_id: str,
        model_version="",
    ):

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
        # print(request)
        response = self.stub.InferenceHandle(request)
        outputs = {}
        for tensor_pb in response.output_data:
            tensor = np.frombuffer(
                tensor_pb.data, dtype=pb_to_numpy_dtype(tensor_pb.dtype)
            )
            tensor = tensor.reshape(tensor_pb.shape)
            outputs[tensor_pb.name] = tensor
        return outputs


class TritonLocalConnector(BaseConnector):
    def __init__(
        self,
        config: Dict,
        url: str = "localhost:8001",
        verbose: bool = False,
        shm=False,
        prefetch=False,
    ) -> None:
        # try:
        #     self.triton_client = grpcclient.InferenceServerClient(
        #         url=url, verbose=verbose
        #     )
        # except Exception as e:
        #     print("channel creation failed: " + str(e))
        #     sys.exit(1)

        self.triton_client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

        # To make sure no shared memory regions are registered with the server.
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

        self.shm_enabled = shm
        self.shm_registered = False
        self.prefetch = prefetch

        self.shm_handles = dict()

        self.ensembles = config["ensembles"]
        self.num_ensembles = len(self.ensembles)
        self.config = config

        # before any prefetch, we need to unload all models
        self.model_cache = set()
        if self.prefetch:
            for model_name in self.ensembles:
                for i in range(config[model_name]["npart"]):
                    self.unload_model(model_name, i)

        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.model_cache_futures = []
        self.model_cache_tail = None

    def __del__(self):
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

    def prefetch_next_model(self, current_model_name: str, skip: int = 1):
        assert skip > 0

        pos = current_model_name.rfind("_")
        model_name = current_model_name[:pos]
        submodule_id = int(current_model_name[pos + 1 :])

        futures = []
        for idx in range(self.ensembles.index(model_name), self.num_ensembles):
            next_model_name = self.ensembles[idx]

            for k in range(submodule_id + 1, self.config[model_name]["npart"]):
                full_model_name = next_model_name + "_" + str(k)
                futures.append(
                    self.executor.submit(self.load_model, next_model_name, k)
                )
                skip -= 1

                if skip == 0:
                    self.model_cache_tail = full_model_name
                    return futures

            submodule_id = -1

            # skip = skip - min(skip, self.config[model_name]["npart"] - submodule_id)

        return []

        # if submodule_id == self.config[model_name]["npart"] - 1:
        #     next_ensemble_id = self.ensembles.index(model_name) + 1
        #     if next_ensemble_id == self.num_ensembles:
        #         return
        #     next_model_name = self.ensembles[next_ensemble_id] + "_0"
        # else:
        #     next_model_name = model_name + "_" + str(submodule_id + 1)

        # return self.executor.submit(self.load_model, next_model_name)

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        session_id: str,
    ):
        # TODO prefetch while inferencing
        futures = []

        start_time = time.perf_counter()
        if model_name not in self.model_cache and self.prefetch:
            print("WARNING: model not loaded, loading now")
            self.load_model(model_name)
            futures.extend(self.prefetch_next_model(model_name, skip=20))
            end_time = time.perf_counter()
            print("cold load time: ", end_time - start_time)

        # do a naive prefetch
        if self.prefetch and len(self.model_cache) < 10:
            futures.extend(self.prefetch_next_model(self.model_cache_tail, 10))
            # task = asyncio.create_task(self.prefetch_next_model(model_name))
            # self.prefetch_next_model(model_name)

        triton_inputs = [
            self._format_triton_input(arr, name) for name, arr in inputs.items()
        ]
        triton_outputs = [
            self._format_triton_output(arr, name) for name, arr in outputs.items()
        ]

        # print("========================")
        # print(triton_inputs, triton_outputs)

        
        results = self.triton_client.infer(
            model_name, triton_inputs, outputs=triton_outputs, request_id=session_id
        )
        

        # TODO double copy is not good: 1) get_contents_as_numpy 2) assignment
        for name, arr in outputs.items():
            output = (
                results.get_output(name) if self.shm_enabled else results.as_numpy(name)
            )
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

        # wait for prefetch to finish and unload the model to save memory
        start_time = time.perf_counter()
        if self.prefetch:
            # await task
            # asyncio.create_task(self.unload_model(model_name))
            if len(futures) > 0:
                futures[0].result()
                futures = futures[1:]

            self.executor.submit(self.unload_model, model_name)

        end_time = time.perf_counter()
        # print(f"wait prefetch time: {end_time - start_time}")

        return outputs

    def _format_triton_input(self, input: np.ndarray, name: str):
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

    def _format_triton_output(self, output: np.ndarray, name: str):
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

    def load_model(self, model_name: str, submodule_id: int = None):
        full_model_name = (
            model_name + "_" + str(submodule_id)
            if submodule_id is not None
            else model_name
        )
        pos = full_model_name.rfind("_")
        model_name = full_model_name[:pos]
        submodule_id = int(full_model_name[pos + 1 :])
        # print("Loading model: %s" % full_model_name)
        config = {
            "name": full_model_name,
            "platform": "pytorch_libtorch",
            "input": [
                {
                    "name": "input",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, 3, 224, 224] if submodule_id == 0 else [-1, -1, -1],
                }
            ],
            "output": [
                {
                    "name": "output",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, 1000]
                    if submodule_id == self.config[model_name]["npart"] - 1
                    else [-1, -1, -1],
                }
            ],
            "instance_group": [{"kind": "KIND_GPU", "count": 1, "gpus": [0]}],
        }
        self.triton_client.load_model(full_model_name, config=json.dumps(config))
        self.model_cache.add(full_model_name)

        # print("load model: %s" % full_model_name, self.model_cache)

    def unload_model(self, model_name: str, submodule_id: int = None):
        full_model_name = (
            model_name + "_" + str(submodule_id)
            if submodule_id is not None
            else model_name
        )
        submodule_id = int(full_model_name.split("_")[-1])
        # print("Unloading model: %s" % full_model_name)
        config = {
            "name": full_model_name,
            "instance_group": [{"kind": "KIND_CPU", "count": 1}],
        }
        # self.triton_client.load_model(full_model_name, config=json.dumps(config))
        self.triton_client.unload_model(full_model_name)
        self.model_cache.discard(full_model_name)

        # print("unload_model model: %s" % full_model_name, self.model_cache)
        # try:
        #     self.triton_client.unload_model(full_model_name)
        # except Exception as e:
        #     print("WARNING" + str(e))

    def get_model_outputs_as_numpy(self, model_name: str):
        model_config = self.triton_client.get_model_config(model_name)

        if self.shm_enabled:
            return {
                output.name: np.zeros(
                    tuple(output.dims), dtype=utils.triton_to_np_dtype(output.datatype)
                )
                for output in model_config.config.output
            }
        else:
            return {output.name: None for output in model_config.config.output}
