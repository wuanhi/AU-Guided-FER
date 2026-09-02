from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from ema_pytorch import EMA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloaders
from src.evaluation.metrics import EMOTION_NAMES, compute_metrics
from src.models.convnext_a1 import build_convnext_a1
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate_tencrop(ema, dataloader, device: torch.device, use_amp: bool) -> tuple[dict, list, list]:
    ema.eval()
    amp_enabled = use_amp and device.type == "cuda"
    all_targets, all_predictions = [], []

    pbar = tqdm(dataloader, desc="Evaluating EMA Model (TenCrop)")

    for batch_data in pbar:
        images = batch_data[0]
        labels = batch_data[1]

        # TenCrop: images shape (B, 10, C, H, W)
        bs, ncrops, c, h, w = images.shape
        images = images.view(-1, c, h, w).to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = ema(images)
            logits = out[0] if isinstance(out, (tuple, list)) else out

        logits = logits.view(bs, ncrops, -1)
        outputs_avg = logits.mean(dim=1)
        predictions = torch.argmax(outputs_avg, dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_metrics(all_targets, all_predictions)
    return metrics, all_targets, all_predictions


def plot_and_save_confusion_matrix(y_true: list, y_pred: list, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_NAMES))))
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[name.capitalize() for name in EMOTION_NAMES],
        yticklabels=[name.capitalize() for name in EMOTION_NAMES],
        cbar_kws={"label": "Tỷ lệ chính xác (%)"},
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Emotion", fontsize=11, fontweight="bold")
    plt.ylabel("True Emotion", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Đã lưu ma trận nhầm lẫn tại: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Ablation A1 Test Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/convnext_tiny_a1/best.pt", help="Path to checkpoint (Local path or hf://twuan/repo/file.pt)")
    parser.add_argument("--config", type=str, default="configs/A1/convnext_tiny_a1.yaml", help="Path to A1 config")
    parser.add_argument("--data-config", type=str, default="configs/data/fer2013.yaml", help="Path to data config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(cfg["training"]["seed"])

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    _, _, test_loader = build_dataloaders(data_cfg, cfg)
    print(f"Số lượng mẫu kiểm thử (Test Samples): {len(test_loader.dataset):,}")

    model = build_convnext_a1(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"]["update_every"],
    ).to(device)

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        ema=ema,
        device=device,
    )
    print(f"\n[INFO] Đã nạp thành công Checkpoint: {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"[INFO] Checkpoint Epoch đạt Best: {checkpoint['epoch']}")
    if "best_acc" in checkpoint:
        print(f"[INFO] Best Validation Accuracy: {checkpoint['best_acc'] * 100:.2f}%")

    use_amp = cfg["training"]["amp"]
    run_name = Path(args.config).stem
    out_dir = Path("outputs") / run_name

    metrics, y_true, y_pred = evaluate_tencrop(
        ema=ema,
        dataloader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    print("\n" + "=" * 65)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN EMA MODEL (A1) ")
    print("=" * 65)
    print(f"  • Overall Accuracy   : {metrics['accuracy'] * 100:.2f}%  ({metrics['accuracy']:.4f})")
    print(f"  • Macro-F1 Score     : {metrics['macro_f1'] * 100:.2f}%  ({metrics['macro_f1']:.4f})")
    print(f"  • Balanced Accuracy  : {metrics['balanced_accuracy'] * 100:.2f}%  ({metrics['balanced_accuracy']:.4f})")
    print("-" * 65)
    print(f"  {'Emotion':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for emo_name, val in metrics["per_class"].items():
        print(f"  {emo_name.capitalize():<10} | {val['precision'] * 100:8.2f}% | {val['recall'] * 100:8.2f}% | {val['f1'] * 100:8.2f}% | {val['support']:<8}")
    print("=" * 65)

    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        out_path=out_dir / "confusion_matrix_a1_ema.png",
        title="A1 Confusion Matrix (ConvNeXt-Tiny + AU Cosine - TenCrop)",
    )

if __name__ == "__main__":
    main()