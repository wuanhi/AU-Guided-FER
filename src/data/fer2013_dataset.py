import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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
    def __init__(self, csv_path, split="Training", transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Usage"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image, mode="L")
        label = int(row["emotion"])
        if self.transform:
            image = self.transform(image)
        return image, label