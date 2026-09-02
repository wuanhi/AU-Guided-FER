"""
A2: ConvNeXt-Tiny + Anatomical Region Decomposition (K=6) + LRP Attention
- V1 (ConvNeXt_A2_Ver1): Fixed Fusion 50/50 
- V2 (ConvNeXt_A2_Ver2): Symmetric Adaptive Fusion Gate (gate_mlp)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from src.models.convnext_backbone import build_convnext_tiny

class LANet(nn.Module):
    """Local Attention Network: 2 layer Conv 1x1 => 1 attention map"""
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


class ConvNeXt_A2_Ver1(nn.Module):
    """ A2 Version 1: Mixed / Fixed Fusion """
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
        # Module LRP: 6 LANets ~ 6 regional attention maps (B, 6, 14, 14)
        self.lrp_lanets = nn.ModuleList([
            LANet(in_channels=384, reduction=8) for _ in range(num_regions)
        ])
        # Regional Classification Head (6*384 channels = 2304)
        reg_dim = num_regions * 384
        self.regional_head = nn.Linear(reg_dim, num_classes)
        nn.init.trunc_normal_(self.regional_head.weight, std=0.02)
        nn.init.zeros_(self.regional_head.bias)

    def extract_features(self, x: torch.Tensor):
        """Extract z_global, z_region, A and features representation"""
        B = x.shape[0]
        # Stage 0 -> 2
        for i in range(3):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
        F2 = x  # (B, 384, 14, 14)
        # Branch Regional
        att_maps = [lanet(F2) for lanet in self.lrp_lanets]
        A = torch.sigmoid(torch.cat(att_maps, dim=1))       # (B, 6, 14, 14)
        F_prime = F2.unsqueeze(1) * A.unsqueeze(2)          # (B, 6, 384, 14, 14)
        V = F_prime.mean(dim=[-2, -1])                      # (B, 6, 384)
        r_region = V.view(B, -1)                            # (B, 2304)
        z_region = self.regional_head(r_region)             # (B, 7)
        # Branch Global
        x_global = self.backbone.downsample_layers[3](F2)
        x_global = self.backbone.stages[3](x_global).mean([-2, -1])
        r_global = self.backbone.norm(x_global)             # (B, 768)
        z_global = self.backbone.head(r_global)             # (B, 7)

        return z_global, z_region, A, r_global, r_region

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        z_global, z_region, A, _, _ = self.extract_features(x)
        if self.fusion_type == "fixed_avg":
            logits = 0.5 * z_global + 0.5 * z_region
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        if return_gate:
            g_dummy = torch.full_like(logits, 0.5)
            return logits, A, g_dummy
        return logits, A


class ConvNeXt_A2_Ver2(ConvNeXt_A2_Ver1):
    """ A2 V2: Symmetric Adaptive Fusion Gate Module (gate_mlp) """
    def __init__(
        self,
        num_classes: int = 7,
        num_regions: int = 6,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        fusion_type: str = "symmetric_adaptive",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_regions=num_regions,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            fusion_type=fusion_type,
        )
        # add module gate_mlp for V2
        in_dim = 768 + (num_regions * 384)  # 768 + 2304 = 3072
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )
        nn.init.trunc_normal_(self.gate_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.zeros_(self.gate_mlp[2].bias)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        z_global, z_region, A, r_global, r_region = self.extract_features(x)
        r_fusion = torch.cat([r_global, r_region], dim=1)  # (B, 3072)
        g = torch.sigmoid(self.gate_mlp(r_fusion))         # (B, 7)
        if self.fusion_type == "symmetric_adaptive":
            logits = (1.0 - g) * z_global + g * z_region   
        else:  # 'fixed_avg'
            logits = 0.5 * z_global + 0.5 * z_region

        if return_gate:
            return logits, A, g
        return logits, A


def build_convnext_a2(
    version: str = "v2",  
    num_classes: int = 7,
    num_regions: int = 6,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    fusion_type: str | None = None,
) -> nn.Module:
    """
    version='v1': ConvNeXt_A2_Ver1 (fixed_avg)
    version='v2': ConvNeXt_A2_Ver2 (symmetric_adaptive)
    """
    ver = str(version).lower().strip()
    if ver in ("v1", "ver1", "1"):
        default_fusion = "fixed_avg" if fusion_type is None else fusion_type
        return ConvNeXt_A2_Ver1(
            num_classes=num_classes,
            num_regions=num_regions,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            fusion_type=default_fusion,
        )
    elif ver in ("v2", "ver2", "2"):
        default_fusion = "symmetric_adaptive" if fusion_type is None else fusion_type
        return ConvNeXt_A2_Ver2(
            num_classes=num_classes,
            num_regions=num_regions,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            fusion_type=default_fusion,
        )
    else:
        raise ValueError(f"Không hỗ trợ version '{version}'. Vui lòng chọn 'v1' hoặc 'v2'.")