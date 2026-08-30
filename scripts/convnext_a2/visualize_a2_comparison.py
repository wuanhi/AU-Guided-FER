"""
- Col 1: Original input face img + Ground Truth (GT) + Predicted label + Confidence
- Col 2..7: Comparison of Learned Attention Maps (A_k) vs Spatial Priors (P_k) for each anatomical region (Brow, Eye, Cheek, Nose, Mouth, Jaw).
"""
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
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spatial_prior_generator import SpatialPriorGenerator
from src.data.transforms import get_val_transform
from src.models.convnext_a2 import build_convnext_a2
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config

EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
REGION_NAMES = ["Brow", "Eye", "Cheek", "Nose", "Mouth", "Jaw"]


def decode_image_rgb(pixels_str: str, target_size: int = 224) -> tuple[np.ndarray, Image.Image]:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L")
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    pil_img_rgb = pil_img.convert("RGB")
    return np.array(pil_img_rgb, dtype=np.uint8), pil_img_rgb


def main():
    parser = argparse.ArgumentParser(description="Visualize A2: 6 Regional Attention Maps vs Priors")
    parser.add_argument("--checkpoint", default="checkpoints/convnext_tiny_a2/best.pt", type=str)
    parser.add_argument("--config", default="configs/A2/convnext_tiny_a2.yaml", type=str)
    parser.add_argument("--data-config", default="configs/data/fer2013.yaml", type=str)
    parser.add_argument("--landmarks-path", default="data/fer2013_landmarks.pkl", type=str)
    parser.add_argument("--num-samples", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Không tìm thấy checkpoint tại: {ckpt_path}")
        return

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    with open(args.landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)

    df = pd.read_csv(data_cfg["dataset"]["csv_path"])
    val_df = df[df["Usage"] == data_cfg["splits"]["val"]].copy()

    model = build_convnext_a2(
        num_classes=cfg["model"]["num_classes"],
        num_regions=6,
        pretrained=False,
        fusion_type=cfg["model"].get("fusion_type", "symmetric_adaptive"),
    ).to(device)

    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    transform = get_val_transform(data_cfg["image"]["size"], cfg["augmentation"])
    prior_gen = SpatialPriorGenerator(target_size=14, orig_size=224, num_regions=6)

    # Select one sample per emotion for visualization
    selected_indices = []
    for emo_idx in range(7):
        subset = val_df[val_df["emotion"] == emo_idx]
        for orig_idx in subset.index:
            key = f"{val_df.loc[orig_idx, 'Usage']}/{orig_idx}"
            if key in landmarks_dict:
                selected_indices.append(orig_idx)
                break

    n_rows = len(selected_indices)
    fig, axes = plt.subplots(nrows=n_rows, ncols=7, figsize=(22, 3.2 * n_rows))
    fig.suptitle("A2 Analysis: 6 Anatomical Region Attention Maps (A_k) vs Input", fontsize=16, fontweight="bold", y=0.995)

    axes[0, 0].set_title("Input Face", fontsize=11, fontweight="bold")
    for k in range(6):
        axes[0, k + 1].set_title(f"Region {k}: {REGION_NAMES[k]}", fontsize=11, fontweight="bold")

    for row_idx, orig_idx in enumerate(selected_indices):
        row = df.loc[orig_idx]
        gt_label = int(row["emotion"])
        gt_name = EMOTIONS[gt_label]
        key = f"{row['Usage']}/{orig_idx}"
        landmarks = landmarks_dict.get(key, None)

        img_np, pil_img = decode_image_rgb(row["pixels"], target_size=224)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        heatmap_P, valid_masks = prior_gen.generate(landmarks, gt_label)

        with torch.no_grad():
            logits, A = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_conf = probs[0, pred_label].item() * 100.0
            A_maps = A.squeeze(0).cpu().numpy()  # (6, 14, 14)

        pred_name = EMOTIONS[pred_label]
        is_correct = (pred_label == gt_label)
        color = "green" if is_correct else "red"

        # Col 1: Original input face image
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].axis("off")
        status_text = f"GT: {gt_name}\nPred: {pred_name} ({pred_conf:.1f}%)"
        axes[row_idx, 0].text(6, 28, status_text, color="white", fontsize=9, fontweight="bold",
                              bbox=dict(facecolor=color, alpha=0.7, pad=2))

        # Col 2..7: 6 Region Maps
        for k in range(6):
            A_k = A_maps[k]
            A_k_norm = (A_k - A_k.min()) / (A_k.max() - A_k.min() + 1e-8)
            axes[row_idx, k + 1].imshow(A_k_norm, cmap="jet", vmin=0.0, vmax=1.0)
            axes[row_idx, k + 1].axis("off")

            has_prior = valid_masks[k].item() == 1.0
            prior_tag = "Has Prior" if has_prior else "No Prior"
            tag_color = "black" if has_prior else "gray"
            axes[row_idx, k + 1].text(0.5, 1.2, f"{REGION_NAMES[k]} ({prior_tag})", color="white", fontsize=8,
                                      bbox=dict(facecolor=tag_color, alpha=0.7))

    plt.tight_layout()
    out_dir = Path("outputs/convnext_tiny_a2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a2_regional_attention_comparison.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SUCCESS] Đã lưu ảnh đối chiếu tại: {out_file.resolve()}")


if __name__ == "__main__":
    main()