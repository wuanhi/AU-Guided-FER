from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from ema_pytorch import EMA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloaders
from src.losses.classification import build_classification_loss
from src.models.convnext_backbone import build_convnext_tiny
from src.training.checkpoint import save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import evaluate_one_epoch, train_one_epoch
from src.utils.config import load_config
from src.utils.logger import append_epoch_log
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt Baseline Training (A0)")
    parser.add_argument("--config", required=True, type=str, help="Path to A0 config (e.g. configs/A0/convnext_tiny_base.yaml)")
    parser.add_argument("--data-config", required=True, type=str, help="Path to data config (e.g. configs/data/fer2013.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataloaders
    train_loader, val_loader, _ = build_dataloaders(data_cfg, cfg)
    print(f"Train samples: {len(train_loader.dataset):,} | Val samples: {len(val_loader.dataset):,}")

    # Model, Loss, Optimizer, Scheduler
    model = build_convnext_tiny(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    loss_fn = build_classification_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = cfg["training"]["amp"]
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"]["update_every"],
    ).to(device)

    run_name = Path(args.config).stem
    output_dir = Path("outputs") / run_name
    checkpoint_dir = Path("checkpoints") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    early_stop_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    max_epochs = cfg["training"]["epochs"]

    for epoch in range(1, max_epochs + 1):
        print(f"\n# Epoch {epoch}/{max_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            ema=ema,
            use_amp=use_amp,
            gradient_clip=cfg["training"]["gradient_clip"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['macro_f1']:.4f}")
        print(f"LR: {current_lr:.8f}")

        append_epoch_log(
            path=output_dir / "history.csv",
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        val_accuracy = val_metrics["accuracy"]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_val_accuracy,
                epoch=epoch,
                config=cfg,
            )
            print(f"Saved new best checkpoint (Val Acc={best_val_accuracy:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Validation accuracy did not improve for {patience} epochs. Early stopping.")
                break

    print("\n[THÀNH CÔNG] Quá trình huấn luyện đã hoàn tất!")
    print(f"Best Checkpoint: {(checkpoint_dir / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()