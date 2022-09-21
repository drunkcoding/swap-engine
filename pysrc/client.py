from dataclasses import dataclass, field
from email import parser
from itertools import count
import time
import uuid
import grpc
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from protos.message_pb2_grpc import ModelInferenceStub
from protos.message_pb2 import (
    InferenceRequest,
    DataType,
    InferenceResponse,
    TensorProto,
)
from pyutils.vit_images_preprocess import ViTFeatureExtractorTransforms, vit_collate_fn
from protos.pb_dtype import numpy_to_pb_dtype, numpy_to_bytes, pb_to_numpy_dtype
from transformers import HfArgumentParser


@dataclass
class ClientArguments:
    model_name: str = field(metadata={"help": "model name on triton server"})
    model_path: str = field(metadata={"help": "model path with tokenizers"})
    dataset_path: str = field(metadata={"help": "Path to ImageNet dataset"})
    verbose: bool = field(default=False, metadata={"help": "Enable verbose output"})
    url: str = field(
        default="localhost:50051",
        metadata={"help": "Inference server URL. Default is localhost:50051."},
    )


class GRPCClient:
    def __init__(self, url):
        self.url = url
        self.channel = grpc.insecure_channel(url)
        self.stub = ModelInferenceStub(self.channel)

    def predict(self, inputs, model_name, model_version):
        request = InferenceRequest()
        request.model_name = model_name
        request.model_version = model_version
        request.session_id = uuid.uuid4().hex

        batch_size = inputs[list(inputs.keys())[0]].shape[0]

        for key, value in inputs.items():
            input = TensorProto()
            value = value.numpy()
            input.name = key
            input.dtype = numpy_to_pb_dtype(value.dtype)
            input.shape.extend(value.shape)
            input.data = value.tobytes()

            request.input_data.append(input)

        logits = TensorProto()
        logits.name = "logits"
        logits.dtype = DataType.DT_FLOAT
        logits.shape.extend([batch_size, 1000])

        request.output_data.append(logits)

        response = self.stub.InferenceHandle(request)
        outputs = {}
        for tensor_pb in response.output_data:
            tensor = np.frombuffer(tensor_pb.data, dtype=pb_to_numpy_dtype(tensor_pb.dtype))
            tensor = tensor.reshape(tensor_pb.shape)
            outputs[tensor_pb.name] = tensor
        return outputs


parser = HfArgumentParser((ClientArguments,))
args = parser.parse_args_into_dataclasses()[0]

client = GRPCClient(args.url)

from torchvision.datasets import ImageNet

dataset = ImageNet(
    args.dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.model_path, split="val"),
)

torch.random.manual_seed(0)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=vit_collate_fn,
)

count = 0
for batch in tqdm(dataloader):
    inputs = {
        "pixel_values": batch["pixel_values"],
    }
    start_time = time.perf_counter()
    outputs = client.predict(inputs, args.model_name, "1")
    end_time = time.perf_counter()
    # print(end_time - start_time)

    count += 1
    # if count == 10:
    #     break
