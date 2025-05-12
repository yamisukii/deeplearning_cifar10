from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class YourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → [B, 64, 16, 16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → [B, 128, 8, 8]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → [B, 256, 4, 4]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)                        # → [B, 256*4*4]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
