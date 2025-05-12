import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.cnn import YourCNN
from assignment_1_code.trainer import ImgClassificationTrainer


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

    # Instantiate CNN and wrap in DeepClassifier
    net = YourCNN()
    model = DeepClassifier(net)
    model.to(device)

    # Optimizer, loss, and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    # Accuracy metrics
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    # Save model dir
    model_save_dir = Path("saved_models/yourcnn")
    model_save_dir.mkdir(exist_ok=True)

    # Create trainer
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
        run_name="cnn_aug_wd_drop"
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train YourCNN")
    args.add_argument("-d", "--gpu_id", default="0",
                      type=str, help="GPU to use")
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.num_epochs = 30

    train(args)
