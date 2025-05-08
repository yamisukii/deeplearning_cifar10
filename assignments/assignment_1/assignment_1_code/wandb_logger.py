from typing import Dict

import torch
import wandb

wandb.login(key="b19e3bab4d5cb7f5ac324d071632f424d4950ae0")


class WandBLogger:

    def __init__(self, enabled=True,
                 model: torch.nn.modules = None,
                 run_name: str = None) -> None:

        self.enabled = enabled

        if self.enabled:
            wandb.init(entity="yamisuki-tu-wien",
                       project="dl_cifar10",
                       group="resnet18")
            if run_name is None:
                wandb.run.name = wandb.run.id
            else:
                wandb.run.name = run_name

            if model is not None:
                self.watch(model)

    def watch(self, model, log_freq: int = 1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self):
        if self.enabled:
            wandb.finish()
