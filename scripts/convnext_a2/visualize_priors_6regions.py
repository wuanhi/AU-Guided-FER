""" Sanity Check Visualizer """
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spatial_prior_generator import SpatialPriorGenerator, REGION_TO_AUS
from src.utils.config import load_config

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
REGION_NAMES = ["Brow", "Eye", "Cheek", "Nose", "Mouth", "Jaw"]


def decode_image_rgb(pixels_str: str, target_size: int = 224) -> tuple[np.ndarray, Image.Image]:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L")
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    pil_img_rgb = pil_img.convert("RGB")
    return np.array(pil_img_rgb, dtype=np.uint8), pil_img_rgb


def overlay_heatmap_on_image(img_rgb: np.ndarray, heatmap_14x14: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if heatmap_14x14.max() == 0:
        return (img_rgb * 0.4).astype(np.uint8)
    h_resized = cv2.resize(heatmap_14x14, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    h_resized = np.clip(h_resized, 0.0, 1.0)
    h_color = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, h_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Sanity Check: Visualize 6 Anatomical Prior Heatmaps")
    parser.add_argument("--csv-path", default="data/fer2013.csv", type=str)
    parser.add_argument("--landmarks-path", default="data/fer2013_landmarks.pkl", type=str)
    parser.add_argument("--num-samples-per-emotion", default=1, type=int, help="Số mẫu cho mỗi lớp cảm xúc")
    parser.add_argument("--output-path", default="outputs/convnext_tiny_a2/visualize_6regions_priors.png", type=str)

    args = parser.parse_args()

    landmarks_path = Path(args.landmarks_path)
    if not landmarks_path.exists():
        print(f"[ERROR] Không tìm thấy file Landmark PKL tại: {landmarks_path}")
        return
    print(f"[INFO] Đang nạp Landmarks từ: {landmarks_path}...")
    with open(landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)
    print(f"[INFO] Đang nạp CSV từ: {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    val_df = df[df["Usage"] == "PublicTest"].copy()

    prior_gen = SpatialPriorGenerator(target_size=14, orig_size=224, sigma_14=1.1, num_regions=6)
    selected_indices = []
    for emo_idx in range(7):
        subset = val_df[val_df["emotion"] == emo_idx]
        found = 0
        for orig_idx in subset.index:
            key = f"{val_df.loc[orig_idx, 'Usage']}/{orig_idx}"
            if key in landmarks_dict and landmarks_dict[key] is not None:
                selected_indices.append(orig_idx)
                found += 1
                if found >= args.num_samples_per_emotion:
                    break

    n_rows = len(selected_indices)
    n_cols = 8  # 1 Original + 6 Regions + 1 Merged

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(24, 3.2 * n_rows))
    fig.suptitle(
        "ANATOMICAL SPATIAL PRIOR SANITY CHECK (6 REGIONS: Brow, Eye, Cheek, Nose, Mouth, Jaw)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    axes[0, 0].set_title("Input + 68 Landmarks", fontsize=11, fontweight="bold")
    for k in range(6):
        axes[0, k + 1].set_title(f"P{k}: {REGION_NAMES[k]}", fontsize=11, fontweight="bold")
    axes[0, 7].set_title("Composite Prior (All)", fontsize=11, fontweight="bold")

    for row_idx, orig_idx in enumerate(selected_indices):
        row = df.loc[orig_idx]
        gt_label = int(row["emotion"])
        gt_name = EMOTIONS[gt_label]
        key = f"{row['Usage']}/{orig_idx}"
        landmarks = landmarks_dict.get(key, None)

        img_np, _ = decode_image_rgb(row["pixels"], target_size=224)

        # 6 Region Priors P & Valid Mask M
        heatmap_P, valid_masks = prior_gen.generate(landmarks, gt_label)
        P_np = heatmap_P.numpy()       # (6, 14, 14)
        M_np = valid_masks.numpy()     # (6,)

        # col 0: Original Image + Landmarks
        img_with_pts = img_np.copy()
        if landmarks is not None:
            for pt in landmarks:
                cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        axes[row_idx, 0].imshow(img_with_pts)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].text(
            6, 26, f"GT: {gt_name}", color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="blue", alpha=0.7, pad=2)
        )

        # col 1-6: 6 Region Priors
        for k in range(6):
            P_k = P_np[k]
            is_valid = M_np[k] == 1.0

            overlay = overlay_heatmap_on_image(img_np, P_k, alpha=0.6)
            axes[row_idx, k + 1].imshow(overlay)
            axes[row_idx, k + 1].axis("off")

            status_txt = f"{REGION_NAMES[k]}: ACTIVE" if is_valid else f"{REGION_NAMES[k]}: MASK=0"
            tag_color = "darkgreen" if is_valid else "black"
            axes[row_idx, k + 1].text(
                6, 24, status_txt, color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=tag_color, alpha=0.7, pad=2)
            )

        # --- col 7: Composite Prior (All Regions) ---
        merged_P = np.max(P_np, axis=0)
        overlay_all = overlay_heatmap_on_image(img_np, merged_P, alpha=0.6)
        axes[row_idx, 7].imshow(overlay_all)
        axes[row_idx, 7].axis("off")
        axes[row_idx, 7].text(
            6, 24, f"Merged ({int(M_np.sum())}/6 active)", color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor="purple", alpha=0.7, pad=2)
        )

    plt.tight_layout()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Successfully]: {out_path.resolve()}")

if __name__ == "__main__":
    main()