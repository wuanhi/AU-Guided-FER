from pathlib import Path
import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    ema,
    scaler,
    scheduler,
    best_metric,
    epoch=None,
    config=None,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_metric,
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if config is not None:
        checkpoint["config"] = config

    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint(
    path,
    model,
    optimizer=None,
    ema=None,
    scaler=None,
    scheduler=None,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if (
        optimizer is not None
        and "opt" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["opt"]
        )

    if (
        ema is not None
        and "ema" in checkpoint
    ):
        ema.load_state_dict(
            checkpoint["ema"]
        )

    if (
        scaler is not None
        and "scaler" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    if (
        scheduler is not None
        and "scheduler" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    return checkpoint