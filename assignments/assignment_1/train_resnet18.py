import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.trainer import ImgClassificationTrainer
from torchvision.models import resnet18


def train(args):
    # Data Augmentation
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(size=(32, 32), padding=4),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    data_dir = Path("dlvc_ss25/assignments/assignment_1/cifar-10-batches-py")
    train_data = CIFAR10Dataset(
        fdir=data_dir, subset=Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(
        fdir=data_dir, subset=Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load resnet18 and replace final layer with dropout + classifier
    net = resnet18(num_classes=10)
    net.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(net.fc.in_features, 10)
    )

    model = DeepClassifier(net)
    model.to(device)

    # Optimizer, loss, and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    # Accuracy tracking
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    model_save_dir = Path("saved_models/resnet18")
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
        val_frequency=5,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train ResNet18")
    args.add_argument("-d", "--gpu_id", default="0",
                      type=str, help="GPU to use")
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.num_epochs = 30

    train(args)
