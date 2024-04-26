import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from pathlib import Path

from model_builder import FashionVGG
from train import class_names

data_path = Path("../data")

custom_image_path = data_path / "glasses.jpg"

custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
custom_image = custom_image / 255.

#print(f"Custom image tensor: \n {custom_image}\n ")
#print(f"Custom image shape: {custom_image.shape}\n ")
#print(f"Custom image dtype:{custom_image.dtype} ")

#plt.imshow(custom_image.permute(1,2,0))
#plt.title(f"Image Shape:{custom_image.shape} ")
#plt.axis(False)
#plt.show()

custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])
custom_image_transformed = custom_image_transform(custom_image)
print(f"New shape: {custom_image_transformed.shape} ")

model_path = "../models/v1fashionvgg.pth"
model_1 = FashionVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(class_names)
)
model_1.load_state_dict(torch.load(model_path))
#print(model_1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_1.eval()

with torch.inference_mode():
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")

    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities:{custom_image_pred_probs} ")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction Label: {custom_image_pred_label} ")

custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
print(custom_image_pred_class)