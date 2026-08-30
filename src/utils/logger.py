import csv
from pathlib import Path

def append_epoch_log(
    path,
    epoch,
    lr,
    train_metrics,
    val_metrics
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    row = {
        "epoch": epoch,
        "lr": lr,

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],

        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)