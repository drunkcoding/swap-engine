from ctypes import util
from dataclasses import dataclass, field
import io
from math import prod
import sys
from typing import overload
import torch
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm

from transformers import HfArgumentParser
import grpc
from concurrent import futures

from protos.pb_dtype import (
    pb_to_numpy_dtype,
    pb_to_str_dtype,
    pb_to_dtype_size,
    pb_to_triton_dtype,
)
from protos import message_pb2_grpc

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RouterArguments:
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    url: str = field(
        default="localhost:8001",
        metadata={"help": "Inference server URL. Default is localhost:8001."},
    )


parser = HfArgumentParser((RouterArguments,))
args = parser.parse_args_into_dataclasses()[0]

from protos.message_pb2_grpc import ModelInferenceServicer
import protos.message_pb2 as message_pb2


def to_bytes(tensor):
    buff = io.BytesIO()
    np.save(tensor, buff)
    return buff.read()


tensor_bytes = lambda tensor_pb: np.prod(tensor_pb.shape) * pb_to_dtype_size(
    tensor_pb.dtype
)


class TritonServicer(ModelInferenceServicer):
    def __init__(self, args):
        # self.args = args
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=args.url, verbose=args.verbose
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        # To make sure no shared memory regions are registered with the server.
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

        self.shm_registered = False

    def InferenceHandle(self, request, context):

        # TODO - add support for model name and version check
        # name must be unique -- will used in shm region name
        inputs = {
            tensor_pb.name: np.frombuffer(
                tensor_pb.data, dtype=pb_to_numpy_dtype(tensor_pb.dtype)
            ).reshape(tensor_pb.shape)
            for tensor_pb in request.input_data
        }

        # logger.debug(f"inputs: {inputs}")

        if not self.shm_registered:
            self.shm_registered = True

            self.shm_ip_handles = {}
            for tensor_pb in request.input_data:
                # Create inputs in Shared Memory and store shared memory handles
                # TODO - add support for multiple deivces device_id in create_shared_memory_region
                logger.debug(
                    f"tensor_pb: {tensor_pb.name}, {tensor_pb.shape}, {tensor_pb.dtype}"
                )
                # logger.debug(f"tensor_pb: {type(tensor_pb.shape)}")
                input_byte_size = tensor_bytes(tensor_pb)
                shm_ip_handle = cudashm.create_shared_memory_region(
                    tensor_pb.name, input_byte_size, 0
                )
                self.shm_ip_handles[tensor_pb.name] = shm_ip_handle
                self.triton_client.register_cuda_shared_memory(
                    tensor_pb.name,
                    cudashm.get_raw_handle(shm_ip_handle),
                    0,
                    input_byte_size,
                )

            self.shm_op_handles = {}
            for tensor_pb in request.output_data:
                output_byte_size = tensor_bytes(tensor_pb)
                shm_op_handle = cudashm.create_shared_memory_region(
                    tensor_pb.name, output_byte_size, 0
                )
                self.shm_op_handles[tensor_pb.name] = shm_op_handle
                self.triton_client.register_cuda_shared_memory(
                    tensor_pb.name,
                    cudashm.get_raw_handle(shm_op_handle),
                    0,
                    output_byte_size,
                )

        # Put input data values into shared memory
        for name, data in inputs.items():
            cudashm.set_shared_memory_region(self.shm_ip_handles[name], [data])

        # Set the parameters to use data from shared memory
        inputs = []
        for tensor_pb in request.input_data:
            input_byte_size = tensor_bytes(tensor_pb)
            inputs.append(
                grpcclient.InferInput(
                    tensor_pb.name,
                    tensor_pb.shape,
                    utils.np_to_triton_dtype(pb_to_numpy_dtype(tensor_pb.dtype)),
                )
            )
            inputs[-1].set_shared_memory(tensor_pb.name, input_byte_size)

        logger.debug(inputs)

        outputs = []
        for tensor_pb in request.output_data:
            output_byte_size = tensor_bytes(tensor_pb)
            outputs.append(grpcclient.InferRequestedOutput(tensor_pb.name))
            outputs[-1].set_shared_memory(tensor_pb.name, output_byte_size)

        logger.debug(outputs)

        results = self.triton_client.infer(
            model_name=request.model_name, inputs=inputs, outputs=outputs,request_id=request.session_id
        )

        # logger.debug(self.triton_client.get_cuda_shared_memory_status())

        response = message_pb2.InferenceResponse()
        response.session_id = request.session_id
        for tensor_pb in request.output_data:
            out_tensor = message_pb2.TensorProto()
            out_tensor.CopyFrom(tensor_pb)

            logger.debug(f"tensor_pb: {tensor_pb}")

            output = results.get_output(tensor_pb.name)
            # print(output)
            if output is not None:
                output_np = cudashm.get_contents_as_numpy(
                    self.shm_op_handles[tensor_pb.name],
                    utils.triton_to_np_dtype(output.datatype),
                    tensor_pb.shape,
                )
            else:
                print("%s is missing in the response." % tensor_pb.name)
                sys.exit(1)

            out_tensor.data = output_np.tobytes()
            response.output_data.append(out_tensor)

        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_ModelInferenceServicer_to_server(TritonServicer(args), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


serve()
