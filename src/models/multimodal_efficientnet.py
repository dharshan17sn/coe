"""
Multimodal EfficientNet-B0 Architecture for Satellite Cyclone Prediction.
Architecture (Matching Review Presentation Slides 13 & 14):
- Visual Branch: EfficientNet-B0 extracts spatial cloud features & eye structure.
- Physical Branch: Tabular MLP processes central pressure, location, and distance.
- Fusion Layer: Concatenates visual embeddings and physical features.
- Multi-Task Prediction:
  1. Future Wind Speed Regression (knots)
  2. IMD Intensity Classification
Supports:
- Multimodal EfficientNet-B0 [Physical]
- Multimodal EfficientNet-B0 [No Pressure] (Ablation study)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultimodalEfficientNet(nn.Module):
    def __init__(self, tabular_in_dim=5, num_classes=8, pretrained=False):
        super(MultimodalEfficientNet, self).__init__()

        # Visual Branch (EfficientNet-B0 backbone)
        if pretrained:
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            base = efficientnet_b0(weights=None)

        self.visual_features = base.features
        self.visual_pool = nn.AdaptiveAvgPool2d(1)
        # EfficientNet-B0 outputs 1280 features
        self.visual_proj = nn.Sequential(
            nn.Linear(1280, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Tabular / Physical Branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Multimodal Fusion (64 visual + 64 tabular = 128)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Multi-task heads
        self.wind_head = nn.Linear(64, 1)
        self.class_head = nn.Linear(64, num_classes)

    def forward(self, img, tab):
        # 1. Visual embedding
        v_feat = self.visual_features(img)
        v_pooled = self.visual_pool(v_feat)
        v_flat = torch.flatten(v_pooled, 1)
        v_emb = self.visual_proj(v_flat)

        # 2. Tabular embedding
        t_emb = self.tabular_branch(tab)

        # 3. Multimodal fusion
        fused = torch.cat([v_emb, t_emb], dim=1)
        latent = self.fusion_mlp(fused)

        # 4. Predictions
        wind_pred = self.wind_head(latent)
        class_logits = self.class_head(latent)
        return wind_pred, class_logits
