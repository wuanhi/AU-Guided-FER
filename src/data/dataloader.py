from __future__ import annotations
from torch.utils.data import DataLoader
from src.data.fer2013_dataset import FER2013Dataset
from src.data.transforms import (
    get_test_transform,
    get_train_transform,
    get_val_transform,
)

def build_dataloaders(data_cfg: dict, experiment_cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    csv_path = data_cfg["dataset"]["csv_path"]
    image_size = data_cfg["image"]["size"]
    augmentation_cfg = experiment_cfg["augmentation"]
    loader_cfg = data_cfg["dataloader"]

    train_transform = get_train_transform(
        image_size=image_size,
        augmentation_cfg=augmentation_cfg,
    )
    val_transform = get_val_transform(
        image_size=image_size,
        augmentation_cfg=augmentation_cfg,
    )
    test_transform = get_test_transform(
        image_size=image_size,
        augmentation_cfg=augmentation_cfg,
    )

    train_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["train"],
        transform=train_transform,
    )

    val_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["val"],
        transform=val_transform,
    )

    test_dataset = FER2013Dataset(
        csv_path=csv_path,
        split=data_cfg["splits"]["test"],
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["train_batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg["val_batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["test_batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader