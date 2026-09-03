from __future__ import annotations
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

EMOTION_NAMES = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}
ERROR_CATEGORIES = [
    "Missing Landmark",
    "IOD Too Small (< 25px)",
    "Mouth Asymmetry (< 0.35)",
    "Eye Asymmetry (< 0.40)",
    "Aspect Ratio Skew (< 0.55 or > 1.80)",
]

def classify_landmark_error(lndmks: np.ndarray | None) -> tuple[bool, str, str]:
    """Returns: (is_valid, category_name, detail_metric_str)"""
    if lndmks is None:
        return False, "Missing Landmark", "No Landmark Points"

    try:
        left_eye = lndmks[36:42].mean(axis=0)
        right_eye = lndmks[42:48].mean(axis=0)
        nose = lndmks[30]

        # Inter-ocular distance
        iod = np.linalg.norm(left_eye - right_eye)
        if iod < 25.0:
            return False, "IOD Too Small (< 25px)", f"IOD: {iod:.1f}px (< 25px)"
        # Mouth symmetry
        d_ml = np.linalg.norm(nose - lndmks[48])
        d_mr = np.linalg.norm(nose - lndmks[54])
        mouth_sym = min(d_ml, d_mr) / (max(d_ml, d_mr) + 1e-6)
        if mouth_sym < 0.35:
            return False, "Mouth Asymmetry (< 0.35)", f"Mouth Sym: {mouth_sym:.2f} (< 0.35)"
        # Eye symmetry
        d_el = np.linalg.norm(nose - left_eye)
        d_er = np.linalg.norm(nose - right_eye)
        eye_sym = min(d_el, d_er) / (max(d_el, d_er) + 1e-6)
        if eye_sym < 0.40:
            return False, "Eye Asymmetry (< 0.40)", f"Eye Sym: {eye_sym:.2f} (< 0.40)"
        # Aspect ratio skew
        w = lndmks[:, 0].max() - lndmks[:, 0].min()
        h = lndmks[:, 1].max() - lndmks[:, 1].min()
        aspect = w / (h + 1e-6)
        if aspect < 0.55 or aspect > 1.80:
            return False, "Aspect Ratio Skew (< 0.55 or > 1.80)", f"Aspect: {aspect:.2f} (∉ [0.55, 1.8])"
        return True, "Valid", "Valid"
    except Exception as e:
        return False, "Missing Landmark", f"Exception: {str(e)}"
def decode_image_rgb(pixels_str: str, target_size: int = 224) -> np.ndarray:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    img_gray = pixels.reshape(48, 48)
    pil_img = Image.fromarray(img_gray, mode="L").resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)
def main():
    csv_path = Path("data/fer2013.csv")
    pkl_path = Path("data/fer2013_landmarks.pkl")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "noise_sample.png"

    if not csv_path.exists() or not pkl_path.exists():
        print(f"[ERROR] Không tìm thấy file: {csv_path} hoặc {pkl_path}")
        return

    print(f"[INFO] Đang nạp dữ liệu từ {csv_path} và {pkl_path}...")
    df = pd.read_csv(csv_path)
    with open(pkl_path, "rb") as f:
        landmarks_dict = pickle.load(f)

    total_samples = len(df)
    valid_count = 0
    invalid_count = 0
    split_stats = {"Training": [0, 0], "PublicTest": [0, 0], "PrivateTest": [0, 0]}
    
    # category -> list of (orig_idx, row, lndmks, metric_str)
    category_samples: dict[str, list] = {cat: [] for cat in ERROR_CATEGORIES}

    for orig_idx, row in df.iterrows():
        split = row["Usage"]
        key = f"{split}/{orig_idx}"
        lndmks = landmarks_dict.get(key, None)
        
        is_valid, category, metric_str = classify_landmark_error(lndmks)

        if is_valid:
            valid_count += 1
            if split in split_stats:
                split_stats[split][0] += 1
        else:
            invalid_count += 1
            if split in split_stats:
                split_stats[split][1] += 1
            if category in category_samples:
                category_samples[category].append((orig_idx, row, lndmks, metric_str))

    print(" THỐNG KÊ CHẤT LƯỢNG LANDMARKS TOÀN BỘ TẬP DỮ LIỆU FER2013 ")
    print("=" * 75)
    print(f"  Tổng số ảnh khảo sát      : {total_samples:,}")
    print(f"  Hợp lệ (Valid Landmarks)   : {valid_count:,} ({valid_count / total_samples * 100:.2f}%)")
    print(f"  Nhiễu / Không hợp lệ       : {invalid_count:,} ({invalid_count / total_samples * 100:.2f}%)")
    print("-" * 75)
    print(f"  {'Tập phân chia (Split)':<20} | {'Hợp lệ (Valid)':<15} | {'Nhiễu (Invalid)':<15} | {'Tỷ lệ hợp lệ':<12}")
    print("-" * 75)
    for split_name, (v, inv) in split_stats.items():
        total_s = v + inv
        pct = (v / total_s * 100) if total_s > 0 else 0
        print(f"  {split_name:<20} | {v:<15,} | {inv:<15,} | {pct:.2f}%")
    print("-" * 75)
    print("  Phân bố chi tiết theo từng nhóm lỗi:")
    for cat in ERROR_CATEGORIES:
        cnt = len(category_samples[cat])
        pct = (cnt / invalid_count * 100) if invalid_count > 0 else 0
        print(f"    - {cat:<40}: {cnt:<6,} ảnh ({pct:.1f}%)")
    print("=" * 75)

    # 3 COLS x N_Error ROWS
    active_categories = [cat for cat in ERROR_CATEGORIES if len(category_samples[cat]) > 0]
    n_rows = len(active_categories)
    n_cols = 3

    if n_rows == 0:
        print("[INFO] Không phát hiện lỗi nào để trực quan hóa.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.6 * n_rows))
    fig.suptitle("NOISE & INVALID LANDMARK SAMPLES (3 REPRESENTATIVE SAMPLES PER ERROR TYPE)", 
                 fontsize=14, fontweight="bold", y=0.995)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, cat in enumerate(active_categories):
        samples = category_samples[cat][:n_cols]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(samples):
                orig_idx, row, lndmks, metric_str = samples[col_idx]
                img_rgb = decode_image_rgb(row["pixels"], target_size=224)
                emo_name = EMOTION_NAMES.get(int(row["emotion"]), "Unknown")

                if lndmks is not None:
                    for pt in lndmks:
                        cv2.circle(img_rgb, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

                ax.imshow(img_rgb)
                ax.axis("off")
                title_text = f"Sample {col_idx + 1} | Idx: {orig_idx} | GT: {emo_name}\n[{metric_str}]"
                ax.set_title(title_text, fontsize=9, color="darkred", fontweight="bold", pad=4)
            else:
                ax.axis("off")

        axes[row_idx, 0].text(-20, 112, f"Error: {cat}", 
                              fontsize=11, fontweight="bold", color="darkred", 
                              rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS]: {out_img_path.resolve()}")

if __name__ == "__main__":
    main()