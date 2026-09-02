"""fusion-type [symmetric_adaptive | fixed_avg"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from ema_pytorch import EMA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.data.dataloader import build_dataloaders
from src.evaluation.metrics import EMOTION_NAMES, compute_metrics
from src.models.convnext_a2 import build_convnext_a2
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed

@torch.no_grad()
def evaluate_tencrop_a2(
    ema,
    dataloader,
    device: torch.device,
    use_amp: bool,
    fusion_mode: str = "symmetric_adaptive",
    desc: str = "Evaluating",
) -> tuple[dict, list, list, np.ndarray]:
    ema.eval()
    target_model = ema.ema_model if hasattr(ema, "ema_model") else ema
    target_model.fusion_type = fusion_mode

    amp_enabled = use_amp and device.type == "cuda"
    all_targets, all_predictions = [], []
    all_gates = []

    pbar = tqdm(dataloader, desc=f"{desc} [{fusion_mode.upper()}]")

    for batch_data in pbar:
        images = batch_data[0]
        labels = batch_data[1]

        bs, ncrops, c, h, w = images.shape
        images = images.view(-1, c, h, w).to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = target_model(images, return_gate=True)
            if isinstance(out, tuple) and len(out) >= 3:
                logits, _, g = out
            elif isinstance(out, (tuple, list)):
                logits = out[0]
                g = torch.full_like(logits, 0.5)
            else:
                logits = out
                g = torch.full_like(logits, 0.5)

        logits = logits.view(bs, ncrops, -1)
        outputs_avg = logits.mean(dim=1)
        predictions = torch.argmax(outputs_avg, dim=1)
        g_avg = g.view(bs, ncrops, -1).mean(dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_gates.append(g_avg.cpu().numpy())

    metrics = compute_metrics(all_targets, all_predictions)
    gates_np = np.concatenate(all_gates, axis=0) if all_gates else np.array([])
    return metrics, all_targets, all_predictions, gates_np

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
    print(f"[INFO] Saved confusion matrix at: {out_path.resolve()}")

def print_results(metrics: dict, title: str, gates_np: np.ndarray = None, y_true: list = None):
    print("\n" + "=" * 75)
    print(f" {title} ")
    print("=" * 75)
    print(f"  • Overall Accuracy   : {metrics['accuracy'] * 100:.2f}%  ({metrics['accuracy']:.4f})")
    print(f"  • Macro-F1 Score     : {metrics['macro_f1'] * 100:.2f}%  ({metrics['macro_f1']:.4f})")
    print(f"  • Balanced Accuracy  : {metrics['balanced_accuracy'] * 100:.2f}%  ({metrics['balanced_accuracy']:.4f})")
    print("-" * 75)

    has_gates = gates_np is not None and len(gates_np) > 0 and y_true is not None
    if has_gates:
        print(f"  {'Emotion':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8} | {'Mean Gate g':<10}")
    else:
        print(f"  {'Emotion':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)

    y_true_arr = np.array(y_true) if y_true is not None else None
    for i, (emo_name, val) in enumerate(metrics["per_class"].items()):
        gate_str = ""
        if has_gates and y_true_arr is not None:
            mask = y_true_arr == i
            if mask.sum() > 0:
                mean_g = gates_np[mask, i].mean()
                gate_str = f" | {mean_g:.4f} ({mean_g*100:.1f}%)"
            else:
                gate_str = f" | N/A"
        print(f"  {emo_name.capitalize():<10} | {val['precision'] * 100:8.2f}% | {val['recall'] * 100:8.2f}% | {val['f1'] * 100:8.2f}% | {val['support']:<8}{gate_str}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Ablation A2 Flexible Test Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default="configs/A2/convnext_tiny_a2_v2.yaml", help="Path to A2 config")
    parser.add_argument("--data-config", type=str, default="configs/data/fer2013.yaml", help="Path to data config")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        choices=["v1", "v2"],
        help="'v1' or 'v2' (default from YAML)",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="compare_both",
        choices=["symmetric_adaptive", "fixed_avg", "compare_both"],
        help="Fusion mode: 'symmetric_adaptive', 'fixed_avg' or 'compare_both'",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(cfg["training"]["seed"])

    version: str = args.version if args.version is not None else cfg["model"].get("version", "v2")

    device = torch.device(args.device)
    print(f"Device: {device} | Test Version: {version.upper()}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.checkpoint is None:
        run_name = Path(args.config).stem
        ckpt_path = Path("checkpoints") / run_name / "best.pt"
        if not ckpt_path.exists():
            fallback_dir = "convnext_tiny_a2_v1" if version == "v1" else "convnext_tiny_a2"
            ckpt_path = str(Path("checkpoints") / fallback_dir / "best.pt")
        else:
            ckpt_path = str(ckpt_path)
    else:
        ckpt_path = args.checkpoint

    _, _, test_loader = build_dataloaders(data_cfg, cfg)
    print(f"Số lượng mẫu kiểm thử (Test Samples): {len(test_loader.dataset):,}")

    model = build_convnext_a2(
        version=version,
        num_classes=cfg["model"]["num_classes"],
        num_regions=cfg["model"].get("num_regions", 6),
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
    ).to(device)

    ema = EMA(
        model,
        beta=cfg["ema"]["beta"],
        update_every=cfg["ema"]["update_every"],
    ).to(device)

    checkpoint = load_checkpoint(
        path=ckpt_path,
        model=model,
        ema=ema,
        device=device,
    )
    print(f"\n[INFO] Đã nạp thành công Checkpoint ({version.upper()}): {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"[INFO] Checkpoint Epoch đạt Best: {checkpoint['epoch']}")
    if "best_acc" in checkpoint:
        print(f"[INFO] Best Validation Accuracy: {checkpoint['best_acc'] * 100:.2f}%")

    use_amp = cfg["training"]["amp"]
    run_name = Path(args.config).stem
    out_dir = Path("outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

   # fusion modes 
    if version == "v1":
        modes_to_run = ["fixed_avg"] if args.fusion_type == "compare_both" else [args.fusion_type]
    else:
        modes_to_run = (
            ["symmetric_adaptive", "fixed_avg"]
            if args.fusion_type == "compare_both"
            else [args.fusion_type]
        )

    summary = {}

    for mode in modes_to_run:
        mode_title = "SYMMETRIC ADAPTIVE FUSION" if mode == "symmetric_adaptive" else "FIXED 50/50 FUSION"
        metrics, y_true, y_pred, gates = evaluate_tencrop_a2(
            ema=ema,
            dataloader=test_loader,
            device=device,
            use_amp=use_amp,
            fusion_mode=mode,
        )

        print_results(
            metrics=metrics,
            title=f"KẾT QUẢ TEST: {mode_title} (A2 {version.upper()} - TenCrop)",
            gates_np=gates if mode == "symmetric_adaptive" else None,
            y_true=y_true,
        )

        cm_file = out_dir / f"confusion_matrix_a2_{mode}.png"
        plot_and_save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            out_path=cm_file,
            title=f"A2 {version.upper()} Confusion Matrix ({mode_title})",
        )

        summary[mode_title] = metrics

    if len(modes_to_run) > 1:
        print("\n" + "#" * 75)
        print(" BẢNG TỔNG KẾT SO SÁNH: ADAPTIVE FUSION VS FIXED FUSION ")
        print("#" * 75)
        print(f" {'Chế độ Fusion':<30} | {'Accuracy':<12} | {'Macro-F1':<12} | {'Balanced Acc':<12}")
        print("-" * 75)
        for m_name, m_val in summary.items():
            print(f" {m_name:<30} | {m_val['accuracy']*100:6.2f}% ({m_val['accuracy']:.4f}) | {m_val['macro_f1']*100:6.2f}% ({m_val['macro_f1']:.4f}) | {m_val['balanced_accuracy']*100:6.2f}% ({m_val['balanced_accuracy']:.4f})")
        print("#" * 75)

if __name__ == "__main__":
    main()