import torch.nn as nn
def build_classification_loss(cfg):
    loss_cfg = cfg["loss"]
    name = loss_cfg["name"].lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=loss_cfg[
                "label_smoothing"
            ]
        )

    raise ValueError(
        f"Unsupported loss: {name}"
    )