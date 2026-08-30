"""
Ablation A2: ConvNeXt-Tiny + Anatomical Region Decomposition (K=6) + LRP Attention + Fixed Fusion.
- Extract Stage 2 feature F (384x14x14).
- Module LRP contain 6 LANet => 6 attention maps A_1..A_6 (Sigmoid).
- Extract 6 feature region F'_k = F ⊙ A_k -> GAP -> V (6, 384) -> Regional Logits.
- Global Branch through Stage 3 -> Global Logits.
- Fixed Fusion: z_final = 0.5 * z_global + 0.5 * z_region 
"""
from __future__ import annotations
import torch
import torch.nn as nn
from src.models.convnext_backbone import build_convnext_tiny


class LANet(nn.Module):
    """Local Attention NetworkC: 2 layer conv 1x1 => 1 attention map."""
    def __init__(self, in_channels: int = 384, reduction: int = 8) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, 1, H, W)


class ConvNeXt_A2(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        num_regions: int = 6,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        fusion_type: str = "fixed_avg", 
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_regions = num_regions
        self.fusion_type = fusion_type

        # Backbone ConvNeXt-Tiny
        self.backbone = build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        # Module LRP: 6 LANets ~ 6 regions giải phẫu
        self.lrp_lanets = nn.ModuleList([
            LANet(in_channels=384, reduction=8) for _ in range(num_regions)
        ])

        # Regional Classification Head
        # 6 regions * 384 channels = 2304
        self.regional_head = nn.Linear(num_regions * 384, num_classes)
        nn.init.trunc_normal_(self.regional_head.weight, std=0.02)
        nn.init.zeros_(self.regional_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:  (B, num_classes) - Fused logits
            A:       (B, 6, 14, 14)   - 6 learned attention maps
        """
        B = x.shape[0]

        # Extract Stage 2 feature F ---
        for i in range(3):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)

        F2 = x  # (B, 384, 14, 14)

        # Regional Branch
        # 6 attention maps cross LRP LANets
        att_maps = [lanet(F2) for lanet in self.lrp_lanets]  # List of 6 x (B, 1, 14, 14)
        A = torch.cat(att_maps, dim=1)                       # (B, 6, 14, 14)
        A = torch.sigmoid(A)                                 

        # F'_k = F ⊙ A_k
        F_exp = F2.unsqueeze(1)                              # (B, 1, 384, 14, 14)
        A_exp = A.unsqueeze(2)                               # (B, 6, 1, 14, 14)
        F_prime = F_exp * A_exp                              # (B, 6, 384, 14, 14)

        #Global Average Pooling 
        V = F_prime.mean(dim=[-2, -1])                       # (B, 6, 384)
        r_region = V.view(B, -1)                             # (B, 2304)
        z_region = self.regional_head(r_region)              # (B, 7)

        # Global Branch
        x_global = self.backbone.downsample_layers[3](F2)
        x_global = self.backbone.stages[3](x_global)
        x_global = x_global.mean([-2, -1])
        r_global = self.backbone.norm(x_global)
        z_global = self.backbone.head(r_global)              # (B, 7)

        # Fixed Fusion
        if self.fusion_type == "fixed_avg":
            logits = 0.5 * z_global + 0.5 * z_region
        elif self.fusion_type == "residual":
            logits = z_global + z_region
        else:
            logits = 0.5 * z_global + 0.5 * z_region

        return logits, A


def build_convnext_a2(
    num_classes: int = 7,
    num_regions: int = 6,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    fusion_type: str = "fixed_avg",
) -> ConvNeXt_A2:
    return ConvNeXt_A2(
        num_classes=num_classes,
        num_regions=num_regions,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        fusion_type=fusion_type,
    )