from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spatial_prior_generator import SpatialPriorGenerator, _check_geometric_plausibility
from src.data.transforms import get_val_transform
from src.models.convnext_a1 import build_convnext_a1
from src.models.convnext_a2 import build_convnext_a2
from src.models.convnext_backbone import build_convnext_tiny
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
EMOTION_NAMES = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def decode_image_rgb(pixels_str: str, target_size: int = 224) -> tuple[np.ndarray, Image.Image]:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L").resize((target_size, target_size), Image.BILINEAR)
    pil_img_rgb = pil_img.convert("RGB")
    return np.array(pil_img_rgb, dtype=np.uint8), pil_img_rgb
def overlay_heatmap(img_rgb: np.ndarray, heatmap_14x14: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    h_min, h_max = heatmap_14x14.min(), heatmap_14x14.max()
    if h_max - h_min > 1e-8:
        h_norm = (heatmap_14x14 - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(heatmap_14x14)
    h_resized = cv2.resize(h_norm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    h_resized = np.clip(h_resized, 0.0, 1.0)
    h_color = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, h_color, alpha, 0)

def get_a0_stage2_attention(model_a0, x_tensor):
    x = x_tensor
    for i in range(3):
        x = model_a0.downsample_layers[i](x)
        x = model_a0.stages[i](x)
    F2 = x  # (1, 384, 14, 14)
    T = F2.mean(dim=1, keepdim=False).squeeze(0)  # (14, 14)
    return T.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 2 Attention Heatmap: A0 vs A1 vs A2 V1 vs A2 V2")
    parser.add_argument("--ckpt-a0", default="checkpoints/convnext_tiny_base/best.pt", type=str)
    parser.add_argument("--ckpt-a1", default="checkpoints/convnext_tiny_a1/best.pt", type=str)
    parser.add_argument("--ckpt-a2-v1", default="checkpoints/convnext_tiny_a2_v1/best.pt", type=str)
    parser.add_argument("--ckpt-a2-v2", default="checkpoints/convnext_tiny_a2_v2/best.pt", type=str)
    parser.add_argument("--cfg-a0", default="configs/A0/convnext_tiny_base.yaml", type=str)
    parser.add_argument("--cfg-a1", default="configs/A1/convnext_tiny_a1.yaml", type=str)
    parser.add_argument("--cfg-a2-v1", default="configs/A2/convnext_tiny_a2_v1.yaml", type=str)
    parser.add_argument("--cfg-a2-v2", default="configs/A2/convnext_tiny_a2_v2.yaml", type=str)
    parser.add_argument("--data-config", default="configs/data/fer2013.yaml", type=str)
    parser.add_argument("--landmarks-path", default="data/fer2013_landmarks.pkl", type=str)
    parser.add_argument("--alpha", default=0.35, type=float, help="(DEFAULT 0.35)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "stage2_attention_comparison_all_emotions.png"

    print(f"[INFO] Load landmarks.. ")
    with open(args.landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)

    data_cfg = load_config(args.data_config)
    cfg_a0 = load_config(args.cfg_a0)
    cfg_a1 = load_config(args.cfg_a1)
    df = pd.read_csv(data_cfg["dataset"]["csv_path"])
    val_df = df[df["Usage"] == data_cfg["splits"]["val"]].copy()

    print(f"[INFO] Load models on {device}...")
    # A0
    model_a0 = build_convnext_tiny(num_classes=7, pretrained=False).to(device)
    load_checkpoint(path=args.ckpt_a0, model=model_a0, device=device)
    model_a0.eval()
    # A1
    model_a1 = build_convnext_a1(num_classes=7, pretrained=False).to(device)
    load_checkpoint(path=args.ckpt_a1, model=model_a1, device=device)
    model_a1.eval()
    # A2 V1 (Fixed 50/50 Fusion)
    model_a2_v1 = build_convnext_a2(version="v1", num_classes=7, num_regions=6, pretrained=False).to(device)
    load_checkpoint(path=args.ckpt_a2_v1, model=model_a2_v1, device=device)
    model_a2_v1.eval()
    # A2 V2 (Symmetric Adaptive Gated Fusion)
    model_a2_v2 = build_convnext_a2(version="v2", num_classes=7, num_regions=6, pretrained=False).to(device)
    load_checkpoint(path=args.ckpt_a2_v2, model=model_a2_v2, device=device)
    model_a2_v2.eval()
    transform = get_val_transform(data_cfg["image"]["size"], cfg_a0["augmentation"])
    prior_gen = SpatialPriorGenerator(target_size=14, orig_size=224, num_regions=6)

    selected_samples = {}
    for emo_idx in range(7):
        subset = val_df[val_df["emotion"] == emo_idx]
        for orig_idx in subset.index:
            key = f"{val_df.loc[orig_idx, 'Usage']}/{orig_idx}"
            if key in landmarks_dict:
                lndmks = landmarks_dict[key]
                if _check_geometric_plausibility(lndmks):
                    selected_samples[emo_idx] = (orig_idx, lndmks)
                    break

    fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(22, 22))
    fig.suptitle("STAGE 2 FEATURE ATTENTION PROGRESSION (A0 -> A1 -> A2 V1 -> A2 V2) ACROSS ALL EMOTIONS", 
                 fontsize=16, fontweight="bold", y=0.995)

    headers = [
        "Input + 68 Landmarks",
        "A0 Base (ConvNeXt)",
        "A1 (Global AU T)",
        "A2 V1 (Fixed 6-Region)",
        "A2 V2 (Adaptive Gate)",
        "Ground-Truth Prior P"
    ]
    for col_idx, h_text in enumerate(headers):
        axes[0, col_idx].set_title(h_text, fontsize=11, fontweight="bold", pad=8)

    for row_idx, emo_idx in enumerate(range(7)):
        orig_idx, lndmks = selected_samples[emo_idx]
        row_data = df.loc[orig_idx]
        emo_name = EMOTION_NAMES[emo_idx]
        img_np, pil_img = decode_image_rgb(row_data["pixels"], target_size=224)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            # A0
            att_a0 = get_a0_stage2_attention(model_a0, input_tensor)
            # A1
            _, T_a1 = model_a1(input_tensor)
            att_a1 = T_a1.squeeze().cpu().numpy()
            # A2 V1
            _, A_a2_v1 = model_a2_v1(input_tensor)
            att_a2_v1_max = A_a2_v1.squeeze(0).max(dim=0)[0].cpu().numpy()
            # A2 V2
            _, A_a2_v2 = model_a2_v2(input_tensor)
            att_a2_v2_max = A_a2_v2.squeeze(0).max(dim=0)[0].cpu().numpy()

        heatmap_P, valid_masks = prior_gen.generate(lndmks, emo_idx)
        P_np = heatmap_P.numpy()
        P_merged = P_np.max(axis=0) if P_np.max() > 0 else np.zeros((14, 14), dtype=np.float32)
        # col1: original img + Landmarks
        img_draw = img_np.copy()
        if lndmks is not None:
            for pt in lndmks:
                cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        axes[row_idx, 0].imshow(img_draw)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].text(6, 24, f"GT: {emo_name}", color="white", fontsize=9, fontweight="bold",
                              bbox=dict(facecolor="blue", alpha=0.7, pad=2))

        # col2: A0 Base
        overlay_a0 = overlay_heatmap(img_np, att_a0, alpha=args.alpha)
        axes[row_idx, 1].imshow(overlay_a0)
        axes[row_idx, 1].axis("off")
        # col3: A1 Global AU
        overlay_a1 = overlay_heatmap(img_np, att_a1, alpha=args.alpha)
        axes[row_idx, 2].imshow(overlay_a1)
        axes[row_idx, 2].axis("off")
        # col4: A2 V1 (Fixed 50/50)
        overlay_a2_v1 = overlay_heatmap(img_np, att_a2_v1_max, alpha=args.alpha)
        axes[row_idx, 3].imshow(overlay_a2_v1)
        axes[row_idx, 3].axis("off")
        # col5: A2 V2 (Adaptive Gate)
        overlay_a2_v2 = overlay_heatmap(img_np, att_a2_v2_max, alpha=args.alpha)
        axes[row_idx, 4].imshow(overlay_a2_v2)
        axes[row_idx, 4].axis("off")
        # col6: Ground Truth Prior P
        if emo_idx == 6 or valid_masks.sum() == 0:
            axes[row_idx, 5].imshow(img_np)
            axes[row_idx, 5].axis("off")
            axes[row_idx, 5].text(6, 24, "Neutral: No AU Prior", color="white", fontsize=9, fontweight="bold",
                                  bbox=dict(facecolor="black", alpha=0.7, pad=2))
        else:
            overlay_p = overlay_heatmap(img_np, P_merged, alpha=args.alpha)
            axes[row_idx, 5].imshow(overlay_p)
            axes[row_idx, 5].axis("off")

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS]: {out_img_path.resolve()}")

if __name__ == "__main__":
    main()