from dataclasses import dataclass, field
from inspect import ArgSpec
from transformers import GPT2Tokenizer
from transformers import HfArgumentParser
import torch
from datasets import load_dataset
import io
from tqdm import tqdm

from transformers import AutoFeatureExtractor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from torchvision.datasets import ImageNet

import grpc
import message_pb2_grpc
import message_pb2

class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, split="train"):
        transform = []

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path
        )

        if feature_extractor.do_resize:
            transform.append(
                RandomResizedCrop(feature_extractor.size) if split == "train" else Resize(feature_extractor.size)
            )

        transform.append(RandomHorizontalFlip() if split == "train" else CenterCrop(feature_extractor.size))
        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x.convert("RGB"))

def vit_collate_fn(batch):
    # print("==batchsize====", len(batch))
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = torch.tensor(transposed_data[1])
    return {"pixel_values": inp, 'labels': tgt}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    dataset_path: str = field(
        metadata={
            "help": "Path to ImageNet dataset"
        }
    )

parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

dataset = ImageNet(
    args.dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.model_name_or_path, split="val"),
)

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

val_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)


def to_bytes(tensor):
    buff = io.BytesIO()
    torch.save(tensor, buff)
    return buff.read()


def load_encodings(encodings):
    max_length = 512
    stride = 128

    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        # input_ids = input_ids.to(torch.int8)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) != max_length:
            continue

        yield input_ids, target_ids, trg_len, end_loc


def make_inference_request(*args, model_name, model_version):
    request = message_pb2.InferenceRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.input_names.append("input_ids")
    request.output_names.append("logits")

    input_ids = message_pb2.TensorProto()
    input_ids.dtype = message_pb2.DataType.DT_INT64
    input_ids.shape = list(args[0].shape)
    input_ids.data = to_bytes(args[0].detach().cpu())

    request.input_data.append(input_ids)
    return request


with grpc.insecure_channel("localhost:50051") as channel:
    stub = message_pb2_grpc.ModelInferenceStub(channel)
    for input_ids, target_ids, trg_len, end_loc in load_encodings(encodings):
        request = make_inference_request(
            input_ids, model_name="opt-350m", model_version="1"
        )
        response = stub.InferenceHandle(request)
