from pathlib import Path

import torch
import torch.nn as nn


class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # Set filename
        filename = f"model.pth" if suffix is None else f"model_{suffix}.pth"
        save_path = save_dir / filename

        # Save the state_dict of self.net (the actual model)
        torch.save(self.net.state_dict(), save_path)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        # Ensure the path exists
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load the state_dict into the model
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(state_dict)
