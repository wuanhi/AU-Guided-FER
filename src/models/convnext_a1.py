"""
Ablation A1: ConvNeXt-Tiny + AU Spatial Cosine Supervision
Extract Stage 2 feature F2 (Bx384x14x14), calculate T = ChannelAvg(F2)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from src.models.convnext_backbone import build_convnext_tiny

class ConvNeXt_A1(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes)
            T: (B, 1, 14, 14) - Feature map from Stage 2
        """
        # Forward Stem, Stage 0, 1, 2
        for i in range(3):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)

        F2 = x # (B, 384, 14, 14)
        # Extract spatial representation T by Channel Average Pooling 
        T = F2.mean(dim=1, keepdim=True)  # (B, 1, 14, 14)

        x = self.backbone.downsample_layers[3](F2)
        x = self.backbone.stages[3](x)

        # Global Average Pooling + LayerNorm + Classification Head
        x = x.mean([-2, -1])
        x = self.backbone.norm(x)
        logits = self.backbone.head(x)

        return logits, T


def build_convnext_a1(
    num_classes: int = 7,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> ConvNeXt_A1:
    return ConvNeXt_A1(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )