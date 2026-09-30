"""Visualisation of evaluation results.

Every image is drawn with:  GT (ground truth) / Prediction / Confidence
  green title = correct, red title = wrong.

Used by evaluation.py, or run standalone on a saved predictions file:
  python vis.py --predictions outputs/eval/predictions.json --out_dir outputs/eval/vis --mode wrong
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_confusion_matrix(cm, classes, path):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.6 * len(classes) * 2), max(5, 0.9 * len(classes) + 2)))
    for ax, data, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "d"),
                                 (axes[1], norm, "Confusion matrix (normalised)", ".2f")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        thr = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thr else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_per_class_f1(per_class, path):
    names = list(per_class)
    f1 = [per_class[c]["f1"] for c in names]
    plt.figure(figsize=(max(6, 0.9 * len(names)), 4))
    bars = plt.bar(names, f1, color="steelblue")
    plt.ylim(0, 1.05); plt.ylabel("F1-score"); plt.title("Per-class F1")
    plt.xticks(rotation=45, ha="right")
    for b, v in zip(bars, f1):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_prediction_grids(records, out_dir, prefix, per_page=12, cols=4, max_pages=5):
    """records: list of dicts {path, gt, pred, conf} (gt/pred are class names)."""
    if not records:
        print(f"[vis] no images for '{prefix}'"); return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    pages = [records[i:i + per_page] for i in range(0, len(records), per_page)][:max_pages]
    for pi, chunk in enumerate(pages, 1):
        rows = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, r in zip(axes, chunk):
            ax.imshow(Image.open(r["path"]).convert("RGB"))
            ok = r["gt"] == r["pred"]
            ax.set_title(f"GT: {r['gt']}\nPred: {r['pred']}\nConf: {r['conf']:.1%}",
                         color="green" if ok else "red", fontsize=10)
        plt.tight_layout()
        p = out_dir / f"{prefix}_{pi:02d}.png"
        plt.savefig(p, dpi=120); plt.close()
        saved.append(p)
    print(f"[vis] saved {len(saved)} page(s) -> {out_dir}/{prefix}_XX.png")
    return saved


def visualize_predictions(records, out_dir, mode="all", per_page=12, max_pages=5, seed=0):
    """mode: 'correct' | 'wrong' | 'all' (saves both)."""
    correct = [r for r in records if r["gt"] == r["pred"]]
    wrong = [r for r in records if r["gt"] != r["pred"]]
    random.Random(seed).shuffle(correct)                 # random sample of successes
    wrong.sort(key=lambda r: -r["conf"])                 # most confident mistakes first
    if mode in ("correct", "all"):
        save_prediction_grids(correct, out_dir, "correct", per_page, max_pages=max_pages)
    if mode in ("wrong", "all"):
        save_prediction_grids(wrong, out_dir, "wrong", per_page, max_pages=max_pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="predictions.json written by evaluation.py")
    ap.add_argument("--out_dir", default="outputs/eval/vis")
    ap.add_argument("--mode", default="all", choices=["all", "correct", "wrong"])
    ap.add_argument("--per_page", type=int, default=12)
    ap.add_argument("--max_pages", type=int, default=5)
    args = ap.parse_args()
    data = json.loads(Path(args.predictions).read_text())
    records = data["records"] if isinstance(data, dict) else data
    records = [r for r in records if r.get("gt")]
    visualize_predictions(records, args.out_dir, args.mode, args.per_page, args.max_pages)


if __name__ == "__main__":
    main()
