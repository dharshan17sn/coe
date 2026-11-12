"""
Custom CNN Architecture for Cyclone Intensity Estimation & Classification.
Architecture (Matching Review Presentation Slides 9 & 11):
- 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D)
- Global Average Pooling / Adaptive Pooling
- Dropout Regularization
- Multi-Task Heads: Wind Speed Regression (knots) + IMD Intensity Classification
"""

import torch
import torch.nn as nn


class CustomCycloneCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8):
        super(CustomCycloneCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # 256 x 4 x 4 = 4096

        self.dense = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )

        # Head 1: Wind Speed (knots)
        self.wind_head = nn.Linear(64, 1)

        # Head 2: IMD Class (0 to 7)
        self.class_head = nn.Linear(64, num_classes)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat)
        flattened = torch.flatten(pooled, 1)
        latent = self.dense(flattened)
        wind_pred = self.wind_head(latent)
        class_logits = self.class_head(latent)
        return wind_pred, class_logits

