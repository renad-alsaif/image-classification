"""Split a dataset (one sub-folder per class) into train / val / test by percentage.

Input:                    Output:
  raw/                      data/
    car/*.jpg                 train/car/*.jpg
    bus/*.jpg                 val/car/*.jpg
    ...                       test/car/*.jpg

The split is stratified: every class is divided using the same percentages.

Usage:
  python split.py --src raw --dst data --train 70 --val 15 --test 15
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--dst", default="data", help="output folder")
    ap.add_argument("--train", type=float, default=70, help="train percentage")
    ap.add_argument("--val", type=float, default=15, help="validation percentage")
    ap.add_argument("--test", type=float, default=15, help="test percentage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 100) > 1e-6:
        raise SystemExit(f"Percentages must sum to 100 (got {args.train + args.val + args.test})")

    random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class sub-folders found in '{src}'")

    transfer = shutil.move if args.move else shutil.copy2
    totals = {"train": 0, "val": 0, "test": 0}
    print(f"{'class':22s}{'total':>7s}{'train':>7s}{'val':>7s}{'test':>7s}")

    for cls in class_dirs:
        files = sorted(f for f in cls.rglob("*") if f.suffix.lower() in IMG_EXT)
        random.shuffle(files)
        n = len(files)
        n_train = round(n * args.train / 100)
        n_val = round(n * args.val / 100)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],   # remainder, so no image is lost
        }
        for split, items in splits.items():
            out = dst / split / cls.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                transfer(str(f), str(out / f.name))
            totals[split] += len(items)
        print(f"{cls.name:22s}{n:7d}{len(splits['train']):7d}{len(splits['val']):7d}{len(splits['test']):7d}")

    print(f"{'TOTAL':22s}{sum(totals.values()):7d}{totals['train']:7d}{totals['val']:7d}{totals['test']:7d}")
    print(f"\nDone. Dataset written to '{dst}/'")


if __name__ == "__main__":
    main()
