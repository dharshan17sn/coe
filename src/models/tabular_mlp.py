"""
Tabular Multi-Layer Perceptron (MLP) architecture for Cyclone Intensity Prediction.
Supports both [Physical] (with pressure) and [No_Pressure] ablation configurations.
Multi-task capability:
1. Wind Speed Regression (knots)
2. Central Pressure Regression (hPa)
3. IMD Cyclone Intensity Classification (7 classes)
"""

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, in_features, hidden_dims=[128, 64, 32], dropout_rate=0.15, num_classes=8):
        super(TabularMLP, self).__init__()

        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Head 1: Continuous Wind Speed Regression (knots)
        self.wind_head = nn.Linear(prev_dim, 1)

        # Head 2: IMD Intensity Classification (0: Low, 1: D, 2: DD, 3: CS, 4: SCS, 5: VSCS, 6: ESCS, 7: SuCS)
        self.class_head = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        wind_pred = self.wind_head(features)
        class_logits = self.class_head(features)
        return wind_pred, class_logits
