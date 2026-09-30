"""Classification metrics (pure NumPy, no training / plotting code).

Functions:
  confusion_matrix(y_true, y_pred, num_classes)
  compute_metrics(y_true, y_pred, classes, y_conf=None) -> dict
  print_report(metrics)
"""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows = ground truth, columns = prediction."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def compute_metrics(y_true, y_pred, classes, y_conf=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, len(classes))

    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true samples per class
    predicted = cm.sum(axis=0)        # predicted samples per class
    precision = _div(tp, predicted)
    recall = _div(tp, support)
    f1 = _div(2 * precision * recall, precision + recall)

    weights = _div(support, support.sum())
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
        "weighted": {"precision": float((precision * weights).sum()),
                     "recall": float((recall * weights).sum()),
                     "f1": float((f1 * weights).sum())},
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "num_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }
    if y_conf is not None:
        y_conf = np.asarray(y_conf)
        correct = y_true == y_pred
        metrics["mean_confidence"] = float(y_conf.mean())
        metrics["mean_confidence_correct"] = float(y_conf[correct].mean()) if correct.any() else 0.0
        metrics["mean_confidence_wrong"] = float(y_conf[~correct].mean()) if (~correct).any() else 0.0
    return metrics


def print_report(m):
    print(f"\nSamples : {m['num_samples']}")
    print(f"Accuracy: {m['accuracy']:.4f}\n")
    print(f"{'class':22s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>10s}")
    for c, v in m["per_class"].items():
        print(f"{c:22s}{v['precision']:10.4f}{v['recall']:10.4f}{v['f1']:10.4f}{v['support']:10d}")
    for avg in ("macro", "weighted"):
        v = m[avg]
        print(f"{avg + ' avg':22s}{v['precision']:10.4f}{v['recall']:10.4f}{v['f1']:10.4f}{m['num_samples']:10d}")
    if "mean_confidence" in m:
        print(f"\nMean confidence: all={m['mean_confidence']:.3f} | "
              f"correct={m['mean_confidence_correct']:.3f} | wrong={m['mean_confidence_wrong']:.3f}")
