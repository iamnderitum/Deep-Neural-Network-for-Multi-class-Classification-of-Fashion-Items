
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import transforms, datasets

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from tqdm.auto import tqdm
from typing import List

from model_builder import FashionVGG
from train import class_names, test_dataloader

test_dir = "../data/glasses_shoes_trousers/test"
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
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

y_preds = []

model_1.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.inference_mode():
    for batch, (X, y) in tqdm(enumerate(test_dataloader)):
        X, y = X.to(device), y.to(device)

        y_logit = model_1(X)

        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

        y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)

test_data = datasets.ImageFolder(test_dir, transform=custom_image_transform)
confmat = ConfusionMatrix(num_classes = len(class_names),task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                             target=test_data.targets)
    
fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7)
)