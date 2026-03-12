"""
PneumoniaCNN – lightweight 2-layer CNN for PneumoniaMNIST (28×28, 1-channel).

Architecture: Conv→Pool→Conv→Pool→FC→FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaCNN(nn.Module):
    """Simple 2-layer CNN for binary pneumonia classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.pool = nn.MaxPool2d(2, 2)                             # → 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 14x14 → 14x14
        # after second pool: 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 32, 7, 7)
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
