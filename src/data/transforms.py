import torch
from torchvision import transforms


def get_train_transform(
    image_size,
    augmentation_cfg,
):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(
                augmentation_cfg["resize_before_crop"]
            ),
            transforms.RandomRotation(
                degrees=augmentation_cfg[
                    "rotation_degrees"
                ]
            ),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),
        ]
    )


def get_val_transform(
    image_size,
    augmentation_cfg,
):
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(
                augmentation_cfg["resize_before_crop"]
            ),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),
        ]
    )


def get_test_transform(
    image_size,
    augmentation_cfg,
):
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(
                augmentation_cfg["resize_before_crop"]
            ),
            transforms.TenCrop(image_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.ToTensor()(crop)
                        for crop in crops
                    ]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        crop.repeat(3, 1, 1)
                        for crop in crops
                    ]
                )
            ),
        ]
    )