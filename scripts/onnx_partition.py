from dataclasses import dataclass, field
import os
import re
from transformers import HfArgumentParser
import onnx_graphsurgeon as gs
import onnx

def return_submodels(graph: gs.Graph, inputs):
    tensors = graph.tensors()
    print(tensors)
    # print(tensors["1058"], type(tensors["1058"]))

    submodules = []
    for i in range(len(inputs) - 1):
        input_name = inputs[i]
        output_name = inputs[i + 1]
        submodules.append({
            "inputs": [tensors[input_name]],
            "outputs": [tensors[output_name]],
        })

    return submodules



@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(metadata={"help": "transformer model type"})

    def __post_init__(self):
        self.info_path = os.path.join(os.path.dirname(self.model_path), "model.info")




parser = HfArgumentParser((ModelArguments,))
args = parser.parse_args_into_dataclasses()[0]

model = onnx.load(args.model_path, load_external_data=True)

os.system(
    "polygraphy inspect model %s --mode=basic > %s " % (args.model_path, args.info_path)
)

pattern = r"input\..*\}\n.*-> \{(input.*)\}"

with open(args.info_path, "r") as f:
    info = f.read()
    inputs = re.findall(pattern, info)
    print(inputs)


if "vit" in args.model_type.lower():
    inputs = ["pixel_values"] + inputs + ["logits"]

print(inputs)

for i in range(len(inputs) - 1):
    graph = gs.import_onnx(model)
    submodels = return_submodels(graph, inputs)

    print("preparing submodels %s" % i)

    graph.inputs = submodels[i]["inputs"]
    graph.outputs = submodels[i]["outputs"]

    graph.cleanup(True, True, True)

    path_to_save_model = os.path.join(os.path.dirname(args.model_path), f"surgeon_{i}.onnx")

    new_model = gs.export_onnx(graph, do_type_check=False)
    onnx.save(new_model, path_to_save_model)

    print("Model saved to:", path_to_save_model)
    print("Inputs:", graph.inputs)
    print("Outputs:", graph.outputs)

    print(graph.tensors().keys())

# # polygraphy surgeon extract --inputs pixel_values --outputs logits
# for i in range(len(inputs) - 1):
#     input_name = inputs[i]
#     output_name = inputs[i + 1]
#     os.system(
#         "polygraphy surgeon extract %s -o %s --inputs %s --outputs %s --save-external-data %s --no-save-all-tensors-to-one-file"
#         % (
#             args.model_path,
#             os.path.join(os.path.dirname(args.model_path), f"surgeon_{i}.onnx"),
#             input_name + ":auto:auto",
#             output_name + ":auto",
#             os.path.dirname(args.model_path),
#         )
#     )
