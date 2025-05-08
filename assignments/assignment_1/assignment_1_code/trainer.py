from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger
from tqdm import tqdm


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """
        super().__init__()

        self.model = model.to(device)  # Ensure model is on the correct device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        # Initialize W&B logger
        self.logger = WandBLogger(
            enabled=True,
            model=model,
            run_name="resnet18_run"  # optional: customize name
        )

        self.best_val_pc_acc = 0.0

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.train_metric.update(outputs.detach(), targets)
            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        acc = self.train_metric.accuracy()
        pc_acc = self.train_metric.per_class_accuracy()

        print(f"\n______epoch {epoch_idx}")
        print(f"accuracy: {acc:.4f}")
        print(f"per class accuracy: {pc_acc:.4f}")
        for class_name in self.train_metric.classes:
            total = self.train_metric.total_pred[class_name]
            correct = self.train_metric.correct_pred[class_name]
            class_acc = correct / total if total > 0 else 0.0
            print(f"Accuracy for class: {class_name:<6} is {class_acc:.2f}")

        return avg_loss, acc, pc_acc

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self.val_metric.reset()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.val_metric.update(outputs, targets)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        acc = self.val_metric.accuracy()
        pc_acc = self.val_metric.per_class_accuracy()

        print(f"\n______epoch {epoch_idx}")
        print(f"accuracy: {acc:.4f}")
        print(f"per class accuracy: {pc_acc:.4f}")
        for class_name in self.val_metric.classes:
            total = self.val_metric.total_pred[class_name]
            correct = self.val_metric.correct_pred[class_name]
            class_acc = correct / total if total > 0 else 0.0
            print(f"Accuracy for class: {class_name:<6} is {class_acc:.2f}")

        return avg_loss, acc, pc_acc

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        self.best_val_pc_acc = 0.0  # track best per-class accuracy

        for epoch in range(1, self.num_epochs + 1):
            # Train on this epoch
            train_loss, train_acc, train_pc_acc = self._train_epoch(epoch)

            # Step learning rate scheduler (if used)
            self.lr_scheduler.step()

            # Log training metrics
            self.logger.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/per_class_accuracy": train_pc_acc,
                "epoch": epoch
            })

            # Run validation periodically or on last epoch
            if epoch % self.val_frequency == 0 or epoch == self.num_epochs:
                val_loss, val_acc, val_pc_acc = self._val_epoch(epoch)

                # Log validation metrics
                self.logger.log({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/per_class_accuracy": val_pc_acc,
                    "epoch": epoch
                })

                # Save best model
                if val_pc_acc > self.best_val_pc_acc:
                    print(
                        f"Saving best model: Per-Class Acc improved from {self.best_val_pc_acc:.4f} → {val_pc_acc:.4f}")
                    self.best_val_pc_acc = val_pc_acc
                    self.model.save(self.training_save_dir, suffix="best")

        self.logger.finish()
