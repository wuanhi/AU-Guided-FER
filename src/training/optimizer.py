import torch

def build_optimizer(model, cfg):
    opt_cfg = cfg["optimizer"]

    name = opt_cfg["name"].lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
        )

    raise ValueError(
        f"Unsupported optimizer: {name}"
    )