"""3-layer classical CNN baseline."""
import torch
import torch.nn as nn


class ClassicalCNN(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.classifier(h)
