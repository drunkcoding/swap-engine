from calendar import c
from dataclasses import dataclass, field
from email import parser
import numpy as np
from typing import Optional
from urllib import response
import torch_ort
import deepspeed
from transformers import HfArgumentParser, ViTConfig, ViTForImageClassification
import torch
from tqdm import tqdm
import os
import cProfile, pstats, io
from pstats import SortKey
import grpc
from concurrent import futures
from protos.pb_dtype import pb_to_numpy_dtype

from protos.message_pb2_grpc import ModelInferenceServicer
import protos.message_pb2 as message_pb2
from protos import message_pb2_grpc

torch.manual_seed(0)

class DeepSpeedServicer(ModelInferenceServicer):
    def __init__(self, args):
        self.args = args

        self.model_dict = {}

        if "vit" in args.model_type.lower():
            for model_path in args.model_paths:
                if args.cfg_only:
                    config = ViTConfig.from_pretrained(model_path)
                    model = ViTForImageClassification(config)
                else:
                    model = ViTForImageClassification.from_pretrained(
                        model_path
                    )
                model.eval()
                # model = model.cuda()

                # model = torch_ort.ORTModule(model)

                model, _, _, _ = deepspeed.initialize(args, model)
                model.eval()

                self.model_dict[os.path.basename(model_path)] = model

        torch.cuda.empty_cache()

    @torch.no_grad()
    def InferenceHandle(self, request, context):

        # TODO - add support for model name and version check
        model = self.model_dict[request.model_name]

        inputs = {
            tensor_pb.name: torch.as_tensor(np.frombuffer(
                tensor_pb.data, dtype=pb_to_numpy_dtype(tensor_pb.dtype)
            ).reshape(tensor_pb.shape), device="cuda").detach()
            for tensor_pb in request.input_data
        }

        outputs = model(**inputs, return_dict=True)

        # print(outputs)

        response = message_pb2.InferenceResponse()
        response.session_id = request.session_id
        for tensor_pb in request.output_data:
            out_tensor = message_pb2.TensorProto()
            out_tensor.CopyFrom(tensor_pb)
            out_tensor.data = outputs[tensor_pb.name].detach().cpu().numpy().tobytes()
            # out_tensor.dtype = message_pb2.DT_FLOAT
            # out_tensor.shape.extend(outputs[tensor_pb.name].shape)

            response.output_data.append(out_tensor)

        return response

@dataclass
class ServerArguments:
    model_paths: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(metadata={"help": "transformer model type"})
    deepspeed_config: str = field(
        default=None,
        metadata={"help": "Path to deepspeed json config file"},
    )
    cfg_only: bool = field(default=False, metadata={"help": "Only import config"})
    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )

    def __post_init__(self):
        self.model_paths = self.model_paths.split(",")

parser = HfArgumentParser((ServerArguments, ))
args = parser.parse_args_into_dataclasses()[0]

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_ModelInferenceServicer_to_server(DeepSpeedServicer(args), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

serve()

# tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

# val_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
# encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
# encodings.input_ids = encodings.input_ids.to(torch.long).to("cuda")

# def load_encodings(encodings):
#     max_length = 512
#     stride = 128

#     for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
#         begin_loc = max(i + stride - max_length, 0)
#         end_loc = min(i + stride, encodings.input_ids.size(1))
#         trg_len = end_loc - i  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc]
#         # input_ids = input_ids.to(torch.int8)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         if input_ids.size(1) != max_length:
#             continue
        
#         yield input_ids, target_ids, trg_len, end_loc

# deepspeed.init_distributed()

# pr = cProfile.Profile()
# pr.enable()
# with torch.no_grad():
#     for input_ids, target_ids, trg_len, end_loc in load_encodings(encodings):
#         # input_ids = input_ids.to("cuda")
#         engine(input_ids=input_ids)
#         count += 1
#         # torch.cuda.empty_cache()

#         if count == 30:
#             break
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.TIME
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# ps.dump_stats("server-stage0.prof")
# print(s.getvalue())

# while True:
#     pass
