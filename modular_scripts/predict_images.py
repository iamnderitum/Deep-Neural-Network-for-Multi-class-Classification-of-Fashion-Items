import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from typing import List
from pathlib import Path
import glob
import os
import random

from model_builder import FashionVGG
from train import class_names

device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_and_predict_images(model: torch.nn.Module,
                            directory_path: str,
                            class_names: List[str] = None,
                            transform=None,
                            device: torch.device = device,
                            num_images: int = 24
                            ):
    
    model.to(device)
    model.eval()

    # Get all image files in the directory with supported extensions
    image_paths = glob.glob(os.path.join(directory_path,"*.jpg")) + \
                  glob.glob(os.path.join(directory_path,"*.jpeg")) + \
                  glob.glob(os.path.join(directory_path,"*.png"))
    
    # Take 25 images in the directory
    image_paths = random.sample(image_paths, min(len(image_paths), num_images))

    matching_predictions = 0
    non_matching_predictions = 0

    #andom_samples_idx = random.sample(range(len()))
    for i, image_path in enumerate(image_paths):
        # Load and preprocess the image
        IMAGE_NAME = os.path.basename(image_path)
        target_image = torchvision.io.read_image(image_path).type(torch.float32)
        target_image = target_image / 255.

        if transform:
            target_image = transform(target_image)

        target_image = target_image.unsqueeze(dim=0).to(device)

        # Make predictions
        with torch.inference_mode():
            target_image_pred = model(target_image)

        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        plt.subplot(4, 6, i+1)
        plt.imshow(target_image.squeeze().permute(1, 2, 0))

        image_name_first_four = IMAGE_NAME[:4]
        predicted_image_class = class_names[target_image_pred_label.cpu().item()]
        predicted_image_class_first_four = predicted_image_class[:4]
        print(f"class name: {predicted_image_class_first_four} | image name: {image_name_first_four}")
        if class_names:
            title = f"Pred: {class_names[target_image_pred_label.cpu().item()]} \nProbability: {target_image_pred_probs.max().cpu().item():.2f} "
            #if image_name_first_four == predicted_image_class_first_four:
                #matching_predictions += 1

            #else:
                #non_matching_predictions += 1
            if any(image_name_first_four == predicted_image_class_first_four for cls_name in class_names):
                print(f"class name: {predicted_image_class_first_four} | image name: {image_name_first_four}")
                matching_predictions += 1

            else:
                non_matching_predictions += 1
        else:
            title = f"Pred: {target_image_pred_label.item()} \nProbability: {target_image_pred_probs.max().cpu().item():.2f} "
        
        plt.title(title)
        plt.axis(False)

    
    
    print(f"Matching Predictions: {matching_predictions}")
    print(f"Non Matching Predictions: {non_matching_predictions}")
    accuracy_percentage = (matching_predictions / num_images) * 100
    #lt.subplots_adjust(left=1, right=2, top=2, bottom=1, wspace=0.1, hspace=0.1)
    plt.suptitle(f"Model Predictions Statistics:\n Correct Predictions: {matching_predictions} / 25 || Accuracy: {accuracy_percentage:.3f} %")
    plt.tight_layout()
    plt.show()

directory_path = "../data/to_pred"

custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64))
])
model_path = "../models/v1fashionvgg.pth"
model_1 = FashionVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(class_names)
)
model_1.load_state_dict(torch.load(model_path))
plot_and_predict_images(
    model=model_1,
    directory_path=directory_path,
    class_names=class_names,
    transform=custom_image_transform,
    device=device
)