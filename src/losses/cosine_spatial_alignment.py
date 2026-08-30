"""
Cosine Spatial Alignment Loss
L_prior = (1 / sum(M_k)) * sum_k [ M_k * (1 - CosineSimilarity(vec(T_k), vec(P_k))) ]
Only use indices where valid_masks = 1.0
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSpatialAlignmentLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        T: torch.Tensor,           # (B, K, H, W) - Learned Attention Maps 
        P: torch.Tensor,           # (B, K, H, W) - Anatomical Prior Heatmaps 
        valid_masks: torch.Tensor  # (B, K) or (B,) 
    ) -> torch.Tensor:
        """ CosineSimilarity between T và P per channel/region """
        if T.shape != P.shape:
            raise ValueError(
                f"[CosineSpatialAlignmentLoss] Shape mismatch: T shape {tuple(T.shape)} "
                f"P shape {tuple(P.shape)}. "
                f"Hãy đảm bảo SpatialPriorGenerator được khởi tạo đúng số kênh (num_regions={T.shape[1]})."
            )

        B, K, H, W = T.shape

        # Flatten -> (B*K, H*W)
        T_flat = T.contiguous().view(B * K, H * W)
        P_flat = P.contiguous().view(B * K, H * W)

        # Cosine Similarity per region channel
        cosine_sim = F.cosine_similarity(T_flat, P_flat, dim=1, eps=self.eps)  # (B*K,)
        loss_per_region = 1.0 - cosine_sim                                      # (B*K,)

        # Mask valid
        mask = valid_masks.contiguous().view(B * K)
        valid_count = mask.sum()

        if valid_count == 0:
            return torch.tensor(0.0, device=T.device, dtype=T.dtype, requires_grad=True)

        loss_cos = (loss_per_region * mask).sum() / (valid_count + self.eps)
        return loss_cos
