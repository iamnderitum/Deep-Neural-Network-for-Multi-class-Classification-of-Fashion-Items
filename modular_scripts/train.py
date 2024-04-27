
import os
import torch
from torchvision import transforms

import argparse

import data_setup,\
        engine,\
        model_builder,\
        utils


NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

def parse_args():
    parser = argparse.ArgumentParser(description="Train FashionVGG model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch Size for the training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of Epochs to train")
    parser.add_argument("--hidden_units", type=int, default=10,
                        help="Number of hidden units")
    
    return parser.parse_args()

args = parse_args()
# Setup directories
train_dir = "../data/glasses_shoes_trousers/train"
test_dir = "../data/glasses_shoes_trousers/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

#CREATE transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=args.batch_size
)

# Create model with help from model_builder.py
model = model_builder.FashionVGG(
    input_shape=3,
    hidden_units=args.hidden_units,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr)

# Start training with help from engine.py

engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=args.epochs,
    device=device
)

utils.save_model(model=model,
                 target_dir="../models",
                 model_name="v1fashionvgg.pth"
                )
