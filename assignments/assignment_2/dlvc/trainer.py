import collections
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

from .wandb_logger import WandBLogger

# from dlvc.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass


class ImgSemSegTrainer(BaseTrainer):
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
        device: torch.device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        # core objects
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric

        # data
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # misc state
        self.device = device
        self.num_epochs = num_epochs
        self.val_frequency = val_frequency
        self.training_save_dir = training_save_dir
        self.training_save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_miou = 0.0
        run_name = f"{model.__class__.__name__}_{self.num_epochs}ep_{train_data.__class__.__name__}"
        self.logger = WandBLogger(
            enabled=True,
            model=model,
            run_name=run_name
        )

        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
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

        '''

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """

        self.model.train()
        self.train_metric.reset()
        running_loss = 0.0

        pbar = tqdm(self.train_loader,
                    desc=f"Train  Epoch {epoch_idx}", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            if masks.ndim == 4 and masks.shape[1] == 1:   # (B,1,H,W) → (B,H,W)
                masks = masks[:, 0]

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            if isinstance(logits, dict):          # torchvision returns dict
                logits = logits["out"]
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            self.train_metric.update(logits.detach().cpu(), masks.cpu())

            pbar.set_postfix(loss=loss.item(), mIoU=self.train_metric.mIoU())

        avg_loss = running_loss / len(self.train_loader)
        miou = self.train_metric.mIoU()
        print(
            f"[Epoch {epoch_idx:03d}]  train loss: {avg_loss:.4f}  mIoU: {miou:.4f}")
        return avg_loss, miou

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self.val_metric.reset()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader,
                        desc=f"Val    Epoch {epoch_idx}", leave=False)
            for imgs, masks in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # (B,1,H,W) → (B,H,W)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]

                logits = self.model(imgs)
                if isinstance(logits, dict):
                    logits = logits["out"]
                loss = self.loss_fn(logits, masks)

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                self.val_metric.update(logits.detach().cpu(), masks.cpu())

                pbar.set_postfix(loss=loss.item(), mIoU=self.val_metric.mIoU())

        avg_loss = running_loss / len(self.val_loader)
        miou = self.val_metric.mIoU()
        print(
            f"[Epoch {epoch_idx:03d}]  val   loss: {avg_loss:.4f}  mIoU: {miou:.4f}")
        return avg_loss, miou

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Logs training and validation metrics every epoch.
        Only runs validation periodically based on val_frequency.
        """
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_miou = self._train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Default values for val loss/miou
            val_loss = float('nan')
            val_miou = float('nan')

            # Run validation
            if epoch % self.val_frequency == 0 or epoch == self.num_epochs:
                val_loss, val_miou = self._val_epoch(epoch)

                if val_miou > self.best_val_miou:
                    self.best_val_miou = val_miou
                    best_path = self.training_save_dir / "best_model.pt"
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  → Saved new best model (mIoU {val_miou:.4f})")

            self.logger.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/mIoU": train_miou,
                "val/loss": val_loss,
                "val/mIoU": val_miou
            })

            # Always save last epoch
            if epoch == self.num_epochs:
                torch.save(
                    self.model.state_dict(),
                    self.training_save_dir / "last_model.pt",
                )

        self.logger.finish()

    def dispose(self) -> None:
        del self.model
        torch.cuda.empty_cache()
