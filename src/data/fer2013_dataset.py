from __future__ import annotations
import logging
from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

class FER2013Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "Training",
        transform: Optional[Callable] = None,
        spatial_prior_generator=None,
        landmarks_dict: Optional[dict] = None,
    ) -> None:
        full_df = pd.read_csv(csv_path)
        split_df = full_df[full_df["Usage"] == split].copy()
        split_df["_original_idx"] = split_df.index
        split_df = split_df.reset_index(drop=True)

        self.df = split_df
        self.split = split
        self.transform = transform
        self.spatial_prior_generator = spatial_prior_generator
        self.landmarks_dict = landmarks_dict

        self.has_prior = (
            self.spatial_prior_generator is not None
            and self.landmarks_dict is not None
        )

        logger.info(
            "[FER2013Dataset] split='%s' | samples=%d | prior_mode=%s",
            self.split,
            len(self.df),
            "ENABLED (A1/A2)" if self.has_prior else "DISABLED (A0 Baseline)",
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image, mode="L")
        label = int(row["emotion"])
        if self.transform:
            image = self.transform(image)

        if self.has_prior:
            original_idx = int(row["_original_idx"])
            pkl_key = f"{self.split}/{original_idx}"
            landmarks = self.landmarks_dict.get(pkl_key, None)
            heatmap_P, valid_mask = self.spatial_prior_generator.generate(
                landmarks=landmarks,
                emotion_label=label,
            )
            return image, label, heatmap_P, valid_mask

        return image, label