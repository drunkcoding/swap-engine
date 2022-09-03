import torch.nn as nn
import torch

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=2000),
            nn.Linear(in_features=2000, out_features=2000),
            nn.Linear(in_features=2000, out_features=2),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

# Create the super-resolution model by using the above model definition.
torch_model = DummyNet()

dummy_model_input = torch.rand(3,2)

torch.onnx.export(
    torch_model, 
    (dummy_model_input, ),
    f="torch_model.onnx",  
    input_names=['x_in'], 
    output_names=['x_out'], 
    do_constant_folding=True, 
    opset_version=12, 
)