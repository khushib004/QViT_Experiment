"""3-layer classical CNN baseline with Grad-CAM / embedding hooks."""
import torch
import torch.nn as nn


class ClassicalCNN(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Last conv stage = Grad-CAM target layer.
        self.block3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base * 4, num_classes)
        self._feature_maps = None  # cached (B, C, H, W) from block3

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.block3

    def forward_features(self, x):
        h = self.block3(self.block2(self.block1(x)))
        self._feature_maps = h
        return h

    def forward(self, x):
        h = self.forward_features(x)
        return self.classifier(self.pool(h).flatten(1))

    @torch.no_grad()
    def embed(self, x):
        return self.pool(self.forward_features(x)).flatten(1)
