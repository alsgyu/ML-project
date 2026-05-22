from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TabularOnlyMLP(nn.Module):
    """20차원 정형 데이터만 사용해 5개 손가락 압력을 동시에 예측하는 MLP 베이스라인."""

    def __init__(self, input_dim: int = 20, output_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, output_dim),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.net(tabular)


class LateFusionResNetRegressor(nn.Module):
    """ResNet-18 이미지 특징과 정형 데이터를 concat하는 5출력 Late Fusion 회귀 모델."""

    def __init__(self, tabular_dim: int = 20, output_dim: int = 5, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 마지막 FC layer를 제거하면 avgpool 이후 512차원 이미지 특징이 남는다.
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feature_dim = resnet.fc.in_features
        fusion_dim = self.image_feature_dim + tabular_dim

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, output_dim),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_backbone(image)
        image_features = torch.flatten(image_features, start_dim=1)
        fused_features = torch.cat([image_features, tabular], dim=1)
        return self.regression_head(fused_features)


def build_multimodal_optimizer(
    model: LateFusionResNetRegressor,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """ResNet backbone과 regression head에 서로 다른 학습률을 적용한다."""

    return torch.optim.AdamW(
        [
            {"params": model.vision_backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": model.regression_head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]
    )
