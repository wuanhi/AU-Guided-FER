"""
AU Spatial Prior Generator (Global AU Heatmap (num_regions = 1, mode='global') & A2 Anatomical 6-Region Decomposition (num_regions = 6, mode='regional')
Gaussian Smoothing (sigma=1.1) 
Geometric Plausibility Check (Valid Mask = 0).
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Action Units (FACS) Tables
REGION_TO_AUS: dict[int, list[int]] = {
    0: [0, 1, 2],                             # Region 0: Brow (Inner/Outer/Lowerer)
    1: [3, 5],                                # Region 1: Eye (Upper Lid Raiser / Tightener)
    2: [4],                                   # Region 2: Cheek (Cheek Raiser)
    3: [6],                                   # Region 3: Nose (Nose Wrinkler)
    4: [7, 8, 9, 10, 12, 13, 14, 15, 19],     # Region 4: Mouth / Lips
    5: [11, 16, 18],                          # Region 5: Jaw / Chin
}
NUM_REGIONS: int = len(REGION_TO_AUS)  # 6

# Emotion (0..6) to Action Units
EXPRESSION_TO_AUS: dict[int, list[int]] = {
    0: [2, 3, 5, 7, 11, 13, 14, 15, 16],  # Angry
    1: [6, 3, 19, 11, 15, 16],            # Disgust
    2: [0, 1, 2, 3, 12, 15, 16, 18],      # Fear
    3: [4, 8, 15],                        # Happy
    4: [0, 2, 4, 10, 11],                 # Sad
    5: [0, 1, 3, 16, 18],                 # Surprise
    # 6: Neutral -> None AU
}


def _check_geometric_plausibility(lndmks: np.ndarray) -> bool:
    """ Check valid geometric plausibility of facial landmarks """
    try:
        # 1. Tọa độ tâm mắt trái (36..41), mắt phải (42..47), chóp mũi (30)
        left_eye = lndmks[36:42].mean(axis=0)
        right_eye = lndmks[42:48].mean(axis=0)
        nose = lndmks[30]

        # 2. Kiểm tra khoảng cách liên mắt (Inter-ocular distance)
        iod = np.linalg.norm(left_eye - right_eye)
        if iod < 25.0:  # Quá nhỏ hoặc bị co rúm
            return False

        # 3. Kiểm tra tính đối xứng của khóe miệng (48: khóe trái, 54: khóe phải) so với mũi
        d_ml = np.linalg.norm(nose - lndmks[48])
        d_mr = np.linalg.norm(nose - lndmks[54])
        mouth_sym = min(d_ml, d_mr) / (max(d_ml, d_mr) + 1e-6)

        # 4. Kiểm tra tính đối xứng của mắt so với mũi
        d_el = np.linalg.norm(nose - left_eye)
        d_er = np.linalg.norm(nose - right_eye)
        eye_sym = min(d_el, d_er) / (max(d_el, d_er) + 1e-6)

        # Nếu độ lệch đối xứng quá lớn
        if mouth_sym < 0.35 or eye_sym < 0.40:
            return False

        # 5. Kiểm tra bounding box khuôn mặt
        w = lndmks[:, 0].max() - lndmks[:, 0].min()
        h = lndmks[:, 1].max() - lndmks[:, 1].min()
        aspect = w / (h + 1e-6)
        if aspect < 0.55 or aspect > 1.80:
            return False

        return True
    except Exception:
        return False


def _draw_au_ellipsoid(au: int, h: int, w: int, lndmks: list[tuple[int, int]]) -> tuple[np.ndarray, bool]:
    att_map = np.zeros((h, w), dtype=np.float32)
    col = 1.0
    a, s, e = 0, 0, 360
    f = cv2.FILLED

    try:
        if au == 0:  # Inner Brow Raiser
            cv2.ellipse(att_map, lndmks[20], (round(w / 8), round(h / 10)), a, s, e, col, f)
            cv2.ellipse(att_map, lndmks[23], (round(w / 8), round(h / 10)), a, s, e, col, f)
        elif au == 1:  # Outer Brow Raiser
            cv2.ellipse(att_map, lndmks[18], (round(w / 8), round(h / 10)), a, s, e, col, f)
            cv2.ellipse(att_map, lndmks[25], (round(w / 8), round(h / 10)), a, s, e, col, f)
        elif au == 2:  # Brow Lowerer
            x = int((lndmks[19][0] + lndmks[24][0]) / 2)
            y = int((lndmks[19][1] + lndmks[24][1]) / 2)
            major = max(int((lndmks[24][0] - lndmks[19][0]) / 2), 10)
            minor = max(int((lndmks[24][1] - lndmks[19][1]) / 2), 10)
            cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)
        elif au in [3, 5]:  # Upper Lid Raiser / Lid Tightener
            x1, y1 = int((lndmks[36][0] + lndmks[39][0]) / 2), int((lndmks[38][1] + lndmks[41][1]) / 2)
            cv2.ellipse(att_map, (x1, y1), (max(int((lndmks[39][0] - lndmks[36][0]) / 2), 5), 6), a, s, e, col, f)
            x2, y2 = int((lndmks[42][0] + lndmks[45][0]) / 2), int((lndmks[44][1] + lndmks[47][1]) / 2)
            cv2.ellipse(att_map, (x2, y2), (max(int((lndmks[45][0] - lndmks[42][0]) / 2), 5), 6), a, s, e, col, f)
        elif au == 4:  # Cheek Raiser
            cv2.ellipse(att_map, (lndmks[41][0], lndmks[41][1] + round(h / 8)), (round(w / 10), round(h / 10)), a, s, e, col, f)
            cv2.ellipse(att_map, (lndmks[46][0], lndmks[46][1] + round(h / 8)), (round(w / 10), round(h / 10)), a, s, e, col, f)
        elif au == 6:  # Nose Wrinkler
            cv2.ellipse(att_map, (lndmks[31][0], lndmks[29][1]), (15, 15), a, s, e, col, f)
            cv2.ellipse(att_map, (lndmks[35][0], lndmks[29][1]), (15, 15), a, s, e, col, f)
        elif au == 7:  # Upper Lip Raiser
            cv2.ellipse(att_map, (int((lndmks[48][0] + lndmks[54][0]) / 2), lndmks[51][1]), (20, 15), a, s, e, col, f)
        elif au in [8, 9, 10]:  # Lip Corners
            cv2.ellipse(att_map, lndmks[48], (16, 16), a, s, e, col, f)
            cv2.ellipse(att_map, lndmks[54], (16, 16), a, s, e, col, f)
        elif au == 11:  # Chin Raiser
            cv2.ellipse(att_map, (int((lndmks[57][0] + lndmks[8][0]) / 2), int((lndmks[57][1] + lndmks[8][1]) / 2)), (16, 16), a, s, e, col, f)
        elif au == 12:  # Lip Stretcher
            cv2.ellipse(att_map, lndmks[48], (20, 15), a, s, e, col, f)
            cv2.ellipse(att_map, lndmks[54], (20, 15), a, s, e, col, f)
        elif au in [13, 14, 15]:  # Lips Part / Pressor / Tightener
            x, y = int((lndmks[48][0] + lndmks[54][0]) / 2), int((lndmks[51][1] + lndmks[57][1]) / 2)
            cv2.ellipse(att_map, (x, y), (max(int((lndmks[54][0] - lndmks[48][0]) / 2), 10), 10), a, s, e, col, f)
        elif au in [16, 18]:  # Jaw Drop / Mouth Stretcher
            x, y = int((lndmks[48][0] + lndmks[54][0]) / 2), int((lndmks[51][1] + lndmks[8][1]) / 2)
            cv2.ellipse(att_map, (x, y), (max(int((lndmks[54][0] - lndmks[48][0]) / 2), 10), 20), a, s, e, col, f)
        elif au == 19:  # Lower Lip Depressor
            cv2.ellipse(att_map, lndmks[57], (20, 15), a, s, e, col, f)
        else:
            return att_map, False
        return att_map, True
    except Exception:
        return att_map, False


class SpatialPriorGenerator:
    def __init__(
        self,
        target_size: int = 14,
        orig_size: int = 224,
        sigma_14: float = 1.1,
        num_regions: int = 1,
        mode: Optional[str] = None,
    ) -> None:
        self.target_size = target_size
        self.orig_size = orig_size
        self.sigma_14 = sigma_14

        if mode is not None:
            if mode in ("global", "single", "a1"):
                self.num_regions = 1
            elif mode in ("regional", "multi", "a2"):
                self.num_regions = NUM_REGIONS
            else:
                raise ValueError(f"Unknown mode '{mode}'. Choose 'global' or 'regional'.")
        else:
            self.num_regions = num_regions

    def _generate_global(self, landmarks: np.ndarray, emotion_label: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A1 mode: Gộp toàn bộ AU của cảm xúc thành 1 heatmap duy nhất (1, 14, 14)."""
        aus = EXPRESSION_TO_AUS[emotion_label]
        lndmks_int = [(int(pt[0]), int(pt[1])) for pt in landmarks]

        au_maps = []
        for au_idx in aus:
            single_map, ok = _draw_au_ellipsoid(au_idx, self.orig_size, self.orig_size, lndmks_int)
            if ok:
                au_maps.append(single_map)

        if not au_maps:
            return torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

        # Gộp tất cả AU theo toán tử MAX pixel-wise
        merged_map = np.max(np.stack(au_maps, axis=0), axis=0)

        # Resize về 14x14 và làm mịn Gaussian
        resized_14 = cv2.resize(merged_map, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        smoothed_14 = cv2.GaussianBlur(resized_14, (5, 5), sigmaX=self.sigma_14, sigmaY=self.sigma_14)

        # Chuẩn hóa về [0, 1]
        max_val = smoothed_14.max()
        if max_val > 0:
            smoothed_14 = smoothed_14 / max_val

        heatmap_P = torch.from_numpy(smoothed_14).float().unsqueeze(0)  # (1, 14, 14)
        valid_mask = torch.ones(1, dtype=torch.float32)                 # (1,)

        return heatmap_P, valid_mask

    def _generate_regional(self, landmarks: np.ndarray, emotion_label: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A2 mode: Tách 6 vùng giải phẫu thành 6 heatmaps (6, 14, 14)."""
        emotion_aus = set(EXPRESSION_TO_AUS[emotion_label])
        lndmks_int = [(int(pt[0]), int(pt[1])) for pt in landmarks]

        region_maps = []
        valid_masks_list = []

        for k in range(NUM_REGIONS):
            reg_target_aus = [au for au in REGION_TO_AUS[k] if au in emotion_aus]

            if not reg_target_aus:
                region_maps.append(np.zeros((self.target_size, self.target_size), dtype=np.float32))
                valid_masks_list.append(0.0)
                continue

            au_drawn = []
            for au_idx in reg_target_aus:
                single_map, ok = _draw_au_ellipsoid(au_idx, self.orig_size, self.orig_size, lndmks_int)
                if ok:
                    au_drawn.append(single_map)

            if not au_drawn:
                region_maps.append(np.zeros((self.target_size, self.target_size), dtype=np.float32))
                valid_masks_list.append(0.0)
                continue

            merged_reg = np.max(np.stack(au_drawn, axis=0), axis=0)
            resized_14 = cv2.resize(merged_reg, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            smoothed_14 = cv2.GaussianBlur(resized_14, (5, 5), sigmaX=self.sigma_14, sigmaY=self.sigma_14)

            max_val = smoothed_14.max()
            if max_val > 0:
                smoothed_14 = smoothed_14 / max_val

            region_maps.append(smoothed_14)
            valid_masks_list.append(1.0)

        heatmap_P = torch.from_numpy(np.stack(region_maps, axis=0)).float()  # (6, 14, 14)
        valid_masks = torch.tensor(valid_masks_list, dtype=torch.float32)     # (6,)

        return heatmap_P, valid_masks

    def generate(self, landmarks: Optional[np.ndarray], emotion_label: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        heatmap_P:   Tensor (num_regions, target_size, target_size)
        valid_masks: Tensor (num_regions,)
        """
        empty_P = torch.zeros(self.num_regions, self.target_size, self.target_size, dtype=torch.float32)
        empty_M = torch.zeros(self.num_regions, dtype=torch.float32)

        # 1. Kiểm tra Neutral, landmark None, hoặc nhãn không hợp lệ
        if landmarks is None or emotion_label == 6 or emotion_label not in EXPRESSION_TO_AUS:
            return empty_P, empty_M

        # 2. Lọc các mẫu bị biến dạng hình học / che khuất nặng
        if not _check_geometric_plausibility(landmarks):
            return empty_P, empty_M

        if self.num_regions == 1:
            return self._generate_global(landmarks, emotion_label)
        else:
            return self._generate_regional(landmarks, emotion_label)