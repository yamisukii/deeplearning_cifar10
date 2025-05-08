# Feel free to change the imports according to your implementation and needs
import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import \
    DeepClassifier  # etc. change to your model
from assignment_1_code.trainer import ImgClassificationTrainer
from torchvision.models import resnet18


def train(args):

    # Implement this function so that it trains a specific model as described in the instruction.md file
    # feel free to change the code snippets given here, they are just to give you an initial structure
    # but do not have to be used if you want to do it differently
    # For device handling you can take a look at pytorch documentation

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    data_dir = Path("assignments/assignment_1/cifar-10-batches-py")
    train_data = CIFAR10Dataset(
        fdir=data_dir, subset=Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(
        fdir=data_dir, subset=Subset.VALIDATION, transform=val_transform)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output matches 10 CIFAR classes
    net = resnet18(num_classes=len(train_data.classes))
    model = DeepClassifier(net)
    model.to(device)

    # Optimizer, loss, scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.num_epochs,
        training_save_dir=model_save_dir,
        batch_size=128,
        val_frequency=val_frequency,
    )

    trainer.train()


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)
