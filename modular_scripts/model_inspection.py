#import ./model_builder
from model_builder import FashionVGG
import torch
import torchvision
model_path = "../models/v1fashionvgg.pth"

model_1 = FashionVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=3
)

model_1.load_state_dict(torch.load(model_path))

"""
This is a multi-line comment. Take a hint
"""

model_1(x).shape