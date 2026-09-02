from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import torch
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[torch.nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[object] = None,
    best_metric: Optional[float] = None,
    epoch: Optional[int] = None,
    config: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "best_acc": best_metric,
    }

    if optimizer is not None:
        checkpoint["opt"] = optimizer.state_dict()
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)

def _resolve_checkpoint_path(
    path: Optional[Union[str, Path]],
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
) -> Path:
    if path is None:
        raise ValueError("[ERROR] Need 'path' or ('hf_repo_id', 'hf_filename').")
    path_str = str(path).strip()
    if path_str.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        parts = path_str.replace("hf://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid hf:// format: '{path_str}'. Expected: 'hf://username/repo_name/file.pt'")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        print(f"[HuggingFace] Downloading {filename} from repo {repo_id}...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return Path(downloaded_path)

    local_path = Path(path_str)
    if not local_path.exists():
        raise FileNotFoundError(f"[ERROR] File checkpoint not found at: {local_path.resolve()}")

    return local_path

def load_checkpoint(
    path: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[torch.nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[object] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
) -> dict:
    actual_path = _resolve_checkpoint_path(path, hf_repo_id, hf_filename)
    device_obj = torch.device(device) if isinstance(device, str) else device
    try:
        checkpoint = torch.load(actual_path, map_location=device_obj, weights_only=False)
    except TypeError:
        checkpoint = torch.load(actual_path, map_location=device_obj)

    # Nạp Model
    if model is not None:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and isinstance(checkpoint, dict) and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])

    if ema is not None and isinstance(checkpoint, dict) and "ema" in checkpoint:
        try:
            ema.load_state_dict(checkpoint["ema"])
        except Exception as e:
            if hasattr(ema, "ema_model") and isinstance(checkpoint["ema"], dict):
                ema.ema_model.load_state_dict(checkpoint["ema"], strict=False)
            else:
                print(f"[WARNING] Cannot load EMA state_dict: {e}")

    if scaler is not None and isinstance(checkpoint, dict) and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}