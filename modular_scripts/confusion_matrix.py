import torch
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



device = "cuda" if torch.cuda.is_available() else "cpu"
def confusion_matrix(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          test_dir: str,
                          #test_transfrom: transforms.Compose,
                          class_names: List[str]=None,
                          device: torch.device = device):
    
    y_preds = []
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)

            y_logit = model(X)

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


confusion_matrix(
    model=model_1,
    dataloader=test_dataloader,
    test_dir=test_dir,
    class_names=class_names
)
