"""
Modified AlexNet Architecture for Satellite Cyclone Feature Extraction.
Architecture (Matching Review Presentation Slides 9 & 11):
- Uses AlexNet convolutional backbone adapted for satellite imagery
- Replaces standard 1000-class classifier with custom cyclone heads:
  1. Wind speed regression (knots)
  2. IMD intensity classification
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class ModifiedAlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, pretrained=False):
        super(ModifiedAlexNet, self).__init__()

        # Base AlexNet features
        if pretrained:
            base_model = alexnet(weights=AlexNet_Weights.DEFAULT)
        else:
            base_model = alexnet(weights=None)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Modified custom head for cyclone imagery
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )

        self.wind_head = nn.Linear(128, 1)
        self.class_head = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        latent = self.classifier(flattened)
        wind_pred = self.wind_head(latent)
        class_logits = self.class_head(latent)
        return wind_pred, class_logits
