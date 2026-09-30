"""Evaluate the fine-tuned model.

Two modes:
  1) Run the model on a split:      python evaluation.py --data_dir data --weights outputs/best_model.pth
  2) Use SAVED inference results:   python evaluation.py --predictions outputs/inference/predictions.json

Both use metric.py for scores and vis.py for the GT / Pred / Conf images.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metric import compute_metrics, print_report
from vis import plot_confusion_matrix, plot_per_class_f1, visualize_predictions


def predict_split(args):
    """Mode 1: run the fine-tuned model on data_dir/split."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from tqdm import tqdm
    from train import get_device, get_transforms, load_finetuned

    device = get_device()
    model, classes, img_size = load_finetuned(args.weights, device)
    ds = datasets.ImageFolder(Path(args.data_dir) / args.split, get_transforms(img_size, train=False))
    if ds.classes != classes:
        raise SystemExit(f"Class mismatch.\nModel: {classes}\nData : {ds.classes}")
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"Evaluating on '{args.split}': {len(ds)} images, {len(classes)} classes")

    preds, confs = [], []
    with torch.no_grad():
        for x, _ in tqdm(loader):
            conf, pred = torch.softmax(model(x.to(device)), dim=1).cpu().max(1)
            preds += pred.tolist(); confs += conf.tolist()
    records = [{"path": ds.samples[i][0], "gt": classes[ds.samples[i][1]],
                "pred": classes[preds[i]], "conf": float(confs[i])} for i in range(len(ds))]
    return classes, records


def load_predictions(path):
    """Mode 2: load results saved by inference.py."""
    data = json.loads(Path(path).read_text())
    classes, records = data["classes"], data["records"]
    labeled = [r for r in records if r.get("gt")]
    if not labeled:
        raise SystemExit("The predictions file has no ground truth (gt). Run inference.py on a folder "
                         "with class sub-folders, e.g. data/test")
    if len(labeled) < len(records):
        print(f"Skipping {len(records) - len(labeled)} image(s) without ground truth")
    return classes, labeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=None, help="predictions.json saved by inference.py")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test", help="test or val")
    ap.add_argument("--weights", default="outputs/best_model.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs/eval")
    ap.add_argument("--per_page", type=int, default=12, help="images per visualisation page")
    ap.add_argument("--max_pages", type=int, default=5, help="max pages per correct/wrong group")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes, records = load_predictions(args.predictions) if args.predictions else predict_split(args)

    idx = {c: i for i, c in enumerate(classes)}
    y_true = [idx[r["gt"]] for r in records]
    y_pred = [idx[r["pred"]] for r in records]
    y_conf = [r["conf"] for r in records]

    metrics = compute_metrics(y_true, y_pred, classes, y_conf)
    print_report(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps({"classes": classes, "records": records}, indent=2))

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), classes, out_dir / "confusion_matrix.png")
    plot_per_class_f1(metrics["per_class"], out_dir / "per_class_f1.png")
    visualize_predictions(records, out_dir / "vis", mode="all",
                          per_page=args.per_page, max_pages=args.max_pages)
    print(f"\nAll results saved in {out_dir}/")


if __name__ == "__main__":
    main()
