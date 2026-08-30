from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

EMOTION_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=list(range(7)),
    average="macro",
    zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(7)),
        zero_division=0
    )

    per_class = {}

    for i, name in enumerate(EMOTION_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "per_class": per_class
    }