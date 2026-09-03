"""
Visualize Codebook of Basic Facial Expressions and Associated Action Units (AUs)
- Col1: Original img FER2013 + 68 Landmarks
- Col2-Col7: Action Unit (AU) activated by the Codebook
- Col8: Merged Prior Heatmap A / P
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
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spatial_prior_generator import (
    EXPRESSION_TO_AUS,
    _check_geometric_plausibility,
    _draw_au_ellipsoid,
)

EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
}
AU_DESCRIPTIONS = {
    0: "AU1 (Inner Brow)",
    1: "AU2 (Outer Brow)",
    2: "AU4 (Brow Lowerer)",
    3: "AU5 (Upper Lid Raiser)",
    4: "AU6 (Cheek Raiser)",
    5: "AU7 (Lid Tightener)",
    6: "AU9 (Nose Wrinkler)",
    7: "AU10 (Upper Lip Raiser)",
    8: "AU12 (Lip Corner Puller)",
    9: "AU14 (Dimpler)",
    10: "AU15 (Lip Corner Depressor)",
    11: "AU17 (Chin Raiser)",
    12: "AU20 (Lip Stretcher)",
    13: "AU23 (Lip Tightener)",
    14: "AU24 (Lip Pressor)",
    15: "AU25 (Lips Part)",
    16: "AU26 (Jaw Drop)",
    18: "AU27 (Mouth Stretcher)",
    19: "AU16 (Lower Lip Depressor)",
}

def decode_image_rgb(pixels_str: str, target_size: int = 224) -> np.ndarray:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L").resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)
def draw_landmarks_on_image(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    img_draw = img_rgb.copy()
    for pt in landmarks:
        cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    return img_draw
def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heatmap_norm = np.clip(heatmap, 0.0, 1.0)
    h_resized = cv2.resize(heatmap_norm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    h_color = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, h_color, alpha, 0)
def main():
    parser = argparse.ArgumentParser(description="Visualize Figure 2: Codebook of FER2013 AUs")
    parser.add_argument("--csv-path", default="data/fer2013.csv", type=str)
    parser.add_argument("--landmarks-path", default="data/fer2013_landmarks.pkl", type=str)
    parser.add_argument("--output-path", default="outputs/visualize_codebook_fer2013.png", type=str)
    args = parser.parse_args()

    landmarks_path = Path(args.landmarks_path)
    if not landmarks_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file landmark: {landmarks_path}")

    with open(landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)

    df = pd.read_csv(args.csv_path)
    val_df = df[df["Usage"] == "PublicTest"].copy()

    selected_samples = {}
    for emo_idx in range(6):  # 6 basic emotions (not including neutral)
        subset = val_df[val_df["emotion"] == emo_idx]
        for orig_idx in subset.index:
            key = f"PublicTest/{orig_idx}"
            if key in landmarks_dict:
                lndmks = landmarks_dict[key]
                if _check_geometric_plausibility(lndmks):
                    selected_samples[emo_idx] = (orig_idx, lndmks)
                    break

    fig = plt.figure(figsize=(20, 15))
    plt.suptitle("Codebook of Basic Facial Expressions and Associated Action Units (FER2013)", fontsize=16, fontweight="bold", y=0.98)
    rows = 6
    for row_idx, emo_idx in enumerate(range(6)):
        orig_idx, lndmks = selected_samples[emo_idx]
        row_data = df.loc[orig_idx]
        img_rgb = decode_image_rgb(row_data["pixels"], target_size=224)
        img_with_lndmks = draw_landmarks_on_image(img_rgb, lndmks)
        active_aus = EXPRESSION_TO_AUS[emo_idx]
        lndmks_int = [(int(pt[0]), int(pt[1])) for pt in lndmks]

        # Original img + Landmark
        ax = plt.subplot2grid((rows, 8), (row_idx, 0))
        ax.imshow(img_with_lndmks)
        ax.axis("off")
        ax.set_title(f"{EMOTION_NAMES[emo_idx]}\n(Input + 68 LM)", fontsize=10, fontweight="bold", pad=4)

        # Each AU Map (Col2-Col7)
        au_maps = []
        for col_idx, au_idx in enumerate(active_aus[:6]):  
            single_map, ok = _draw_au_ellipsoid(au_idx, 224, 224, lndmks_int)
            if ok:
                au_maps.append(single_map)
                overlay = overlay_heatmap(img_rgb, single_map, alpha=0.6)
                ax_au = plt.subplot2grid((rows, 8), (row_idx, col_idx + 1))
                ax_au.imshow(overlay)
                ax_au.axis("off")
                au_label = AU_DESCRIPTIONS.get(au_idx, f"AU {au_idx}")
                ax_au.set_title(au_label, fontsize=9, pad=4)

        # Merged AU Map A / P
        if au_maps:
            merged_map = np.max(np.stack(au_maps, axis=0), axis=0)
            resized_14 = cv2.resize(merged_map, (14, 14), interpolation=cv2.INTER_AREA)
            smoothed_14 = cv2.GaussianBlur(resized_14, (5, 5), sigmaX=1.1, sigmaY=1.1)
            if smoothed_14.max() > 0:
                smoothed_14 = smoothed_14 / smoothed_14.max()
            final_overlay = overlay_heatmap(img_rgb, cv2.resize(smoothed_14, (224, 224)), alpha=0.6)
        else:
            final_overlay = img_rgb

        ax_final = plt.subplot2grid((rows, 8), (row_idx, 7))
        ax_final.imshow(final_overlay)
        ax_final.axis("off")
        ax_final.set_title("Merged Prior (Map A)", fontsize=10, fontweight="bold", color="darkblue", pad=4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS]: {out_file.resolve()}")

if __name__ == "__main__":
    main()