"""
1. Load offline landmarks: data/fer2013_landmarks.pkl
2. SpatialPriorGenerator for AU Heatmap P (1 channel, 14x14)
3. Model: ConvNeXt_A1 trích xuất Feature Stage 2 -> T (Channel Average Pooling)
4. Loss: L_total = L_CE(logits, labels) + lambda_cos * L_cos(T, P)
"""
from __future__ import annotations
import argparse
import csv
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from ema_pytorch import EMA
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloaders
from src.data.spatial_prior_generator import SpatialPriorGenerator
from src.evaluation.metrics import compute_metrics
from src.losses.classification import build_classification_loss
from src.losses.cosine_spatial_alignment import CosineSpatialAlignmentLoss
from src.models.convnext_a1 import build_convnext_a1
from src.training.checkpoint import save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.config import load_config
from src.utils.seed import set_seed


def _train_one_epoch_a1(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    scheduler,
    loss_cls_fn,
    loss_cos_fn,
    device: torch.device,
    scaler,
    ema,
    use_amp: bool,
    gradient_clip: float,
    gradient_accumulation_steps: int,
    lambda_cos: float,
) -> dict:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"

    batch_losses_total = []
    batch_losses_cls = []
    batch_losses_cos = []
    batch_accuracies = []
    all_targets = []
    all_predictions = []

    pbar = tqdm(dataloader, desc="Training A1")

    for batch_idx, (images, labels, heatmaps_P, valid_masks) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        heatmaps_P = heatmaps_P.to(device)
        valid_masks = valid_masks.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, T = model(images)

            loss_cls = loss_cls_fn(logits, labels)
            loss_cos = loss_cos_fn(T, heatmaps_P, valid_masks)
            loss = loss_cls + lambda_cos * loss_cos

        predictions = torch.argmax(logits, dim=1)
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            ema.update()
            scheduler.step()

        acc = (predictions == labels).sum().item() / labels.size(0)
        batch_losses_total.append(loss.item())
        batch_losses_cls.append(loss_cls.item())
        batch_losses_cos.append(loss_cos.item())
        batch_accuracies.append(acc)

        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        pbar.set_postfix({
            "loss": f"{np.mean(batch_losses_total):.4f}",
            "cls": f"{np.mean(batch_losses_cls):.4f}",
            "cos": f"{np.mean(batch_losses_cos):.4f}",
            "acc": f"{np.mean(batch_accuracies) * 100:.1f}%",
        })

    metrics = compute_metrics(all_targets, all_predictions)
    metrics["loss"] = float(np.mean(batch_losses_total))
    metrics["loss_cls"] = float(np.mean(batch_losses_cls))
    metrics["loss_cos"] = float(np.mean(batch_losses_cos))
    metrics["accuracy"] = float(np.mean(batch_accuracies))
    return metrics


@torch.no_grad()
def _evaluate_one_epoch_a1(
    model: torch.nn.Module,
    dataloader,
    loss_cls_fn,
    loss_cos_fn,
    device: torch.device,
    use_amp: bool,
    lambda_cos: float,
) -> dict:
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    batch_losses_total = []
    batch_losses_cls = []
    batch_losses_cos = []
    all_targets = []
    all_predictions = []

    pbar = tqdm(dataloader, desc="Validation A1")

    for images, labels, heatmaps_P, valid_masks in pbar:
        images = images.to(device)
        labels = labels.to(device)
        heatmaps_P = heatmaps_P.to(device)
        valid_masks = valid_masks.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, T = model(images)
            loss_cls = loss_cls_fn(logits, labels)
            loss_cos = loss_cos_fn(T, heatmaps_P, valid_masks)
            loss = loss_cls + lambda_cos * loss_cos

        predictions = torch.argmax(logits, dim=1)
        batch_losses_total.append(loss.item())
        batch_losses_cls.append(loss_cls.item())
        batch_losses_cos.append(loss_cos.item())

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_metrics(all_targets, all_predictions)
    metrics["loss"] = float(np.mean(batch_losses_total))
    metrics["loss_cls"] = float(np.mean(batch_losses_cls))
    metrics["loss_cos"] = float(np.mean(batch_losses_cos))
    return metrics


def _append_epoch_log_a1(path: Path, epoch: int, lr: float, train_metrics: dict, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_metrics["loss"],
        "train_loss_cls": train_metrics["loss_cls"],
        "train_loss_cos": train_metrics["loss_cos"],
        "train_acc": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_metrics["loss"],
        "val_loss_cls": val_metrics["loss_cls"],
        "val_loss_cos": val_metrics["loss_cos"],
        "val_acc": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="FER2013 Stage 1 – Ablation A1 Training")
    parser.add_argument("--config", required=True, type=str, help="Path to A1 config (e.g. configs/A1/convnext_tiny_a1.yaml)")
    parser.add_argument("--data-config", required=True, type=str, help="Path to data config (e.g. configs/data/fer2013.yaml)")
    parser.add_argument("--landmarks-path", type=str, default="data/fer2013_landmarks.pkl", help="Path to offline extracted landmarks .pkl")
    parser.add_argument("--lambda-cos", type=float, default=None, help="Override lambda_cos weight in YAML")

    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    
    lambda_cos: float = args.lambda_cos if args.lambda_cos is not None else cfg["training"].get("lambda_cos", 0.1)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Lambda Cosine: {lambda_cos}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load offline landmarks
    landmarks_path = Path(args.landmarks_path)
    if not landmarks_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Landmark PKL file not found at: {landmarks_path}\n"
        )
    with open(landmarks_path, "rb") as f:
        landmarks_dict = pickle.load(f)
    print(f"[A1] Loaded landmarks for {len(landmarks_dict):,} images.")

    # Prior Generator (A1: Global AU Heatmap)
    prior_gen = SpatialPriorGenerator(target_size=14, orig_size=224, num_regions=1)

    # Dataloaders
    train_loader, val_loader, _ = build_dataloaders(
        data_cfg,
        cfg,
        spatial_prior_generator=prior_gen,
        landmarks_dict=landmarks_dict,
    )
    print(f"Train samples: {len(train_loader.dataset):,} | Val samples: {len(val_loader.dataset):,}")

    # ConvNeXt_A1
    model = build_convnext_a1(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    # Loss functions
    loss_cls_fn = build_classification_loss(cfg)
    loss_cos_fn = CosineSpatialAlignmentLoss().to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = cfg["training"]["amp"]
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = EMA(model, beta=cfg["ema"]["beta"], update_every=cfg["ema"]["update_every"]).to(device)

    run_name = Path(args.config).stem
    output_dir = Path("outputs") / run_name
    checkpoint_dir = Path("checkpoints") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    early_stop_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    max_epochs = cfg["training"]["epochs"]

    for epoch in range(1, max_epochs + 1):
        print(f"\n# Epoch {epoch}/{max_epochs}")
        train_metrics = _train_one_epoch_a1(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_cls_fn=loss_cls_fn,
            loss_cos_fn=loss_cos_fn,
            device=device,
            scaler=scaler,
            ema=ema,
            use_amp=use_amp,
            gradient_clip=cfg["training"]["gradient_clip"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            lambda_cos=lambda_cos,
        )

        val_metrics = _evaluate_one_epoch_a1(
            model=model,
            dataloader=val_loader,
            loss_cls_fn=loss_cls_fn,
            loss_cos_fn=loss_cos_fn,
            device=device,
            use_amp=use_amp,
            lambda_cos=lambda_cos,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train | Loss: {train_metrics['loss']:.4f} (CLS: {train_metrics['loss_cls']:.4f}, COS: {train_metrics['loss_cos']:.4f}) | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f} (CLS: {val_metrics['loss_cls']:.4f}, COS: {val_metrics['loss_cos']:.4f}) | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['macro_f1']:.4f}")

        _append_epoch_log_a1(output_dir / "history_a1.csv", epoch, current_lr, train_metrics, val_metrics)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            early_stop_counter = 0
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_val_acc,
                epoch=epoch,
                config=cfg,
            )
            print(f"--> Saved new best checkpoint (Val Acc={best_val_acc:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Validation accuracy did not improve for {patience} epochs. Early stopping.")
                break

    print("\n[THÀNH CÔNG] Huấn luyện A1 hoàn tất!")
    print(f"Best Checkpoint: {(checkpoint_dir / 'best.pt').resolve()}")

if __name__ == "__main__":
    main()