"""Run the fine-tuned model on an image or folder.

Single image:
    Prints the predicted class and confidence only.

Folder:
    Runs inference on all images and saves predictions as JSON and CSV.

Examples:

Single image:
    python inference.py \
        --image "data/test/Bikes/Bike (8).jpg" \
        --weights outputs/best_model.pth

Folder:
    python inference.py \
        --image data/test \
        --weights outputs/best_model.pth \
        --out outputs/inference/predictions.json

Evaluation:
    python evaluation.py \
        --predictions outputs/inference/predictions.json
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from train import get_device, get_transforms, load_finetuned


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_single_image(model, tfm, image_path, classes, device, topk):
    """Run inference on a single image and print the result."""

    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    k = min(topk, len(classes))
    conf, idx = probs.topk(k)

    print("\nPrediction")
    print("-" * 30)
    print(f"Image:      {image_path}")
    print(f"Prediction: {classes[idx[0]]}")
    print(f"Confidence: {conf[0].item() * 100:.2f}%")

    if k > 1:
        print("\nTop predictions:")

        for c, j in zip(conf, idx):
            print(
                f"  {classes[j]:<20} "
                f"{c.item() * 100:.2f}%"
            )


def predict_folder(
    model,
    tfm,
    image_dir,
    classes,
    device,
    topk,
    batch_size,
    out,
):
    """Run inference on a folder and save JSON + CSV results."""

    paths = sorted(
        f
        for f in image_dir.rglob("*")
        if f.suffix.lower() in IMG_EXT
    )

    if not paths:
        raise SystemExit("No images found.")

    print(f"Running inference on {len(paths)} image(s)")

    records = []

    k = min(topk, len(classes))

    for i in tqdm(
        range(0, len(paths), batch_size),
        desc="Inference",
    ):
        batch_paths = paths[i:i + batch_size]

        images = []

        for f in batch_paths:
            image = Image.open(f).convert("RGB")
            images.append(tfm(image))

        x = torch.stack(images).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu()

        conf, idx = probs.topk(k, dim=1)

        for f, c, ix in zip(
            batch_paths,
            conf,
            idx,
        ):
            # If the image is inside a class folder,
            # use the folder name as the ground truth.
            gt = (
                f.parent.name
                if f.parent.name in classes
                else None
            )

            records.append(
                {
                    "path": str(f),
                    "gt": gt,
                    "pred": classes[ix[0]],
                    "conf": float(c[0]),
                    "topk": [
                        {
                            "class": classes[j],
                            "conf": float(cc),
                        }
                        for cc, j in zip(c, ix)
                    ],
                }
            )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    out = Path(out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            {
                "classes": classes,
                "records": records,
            },
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = out.with_suffix(".csv")

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        writer = csv.writer(fh)

        writer.writerow(
            [
                "path",
                "gt",
                "pred",
                "conf",
            ]
        )

        for record in records:
            writer.writerow(
                [
                    record["path"],
                    record["gt"] or "",
                    record["pred"],
                    f"{record['conf']:.4f}",
                ]
            )

    print("\nInference completed.")
    print(f"Saved JSON: {out}")
    print(f"Saved CSV:  {csv_path}")

    if not any(
        record["gt"]
        for record in records
    ):
        print(
            "\nNote: no class folders detected, "
            "so gt is empty. Evaluation requires "
            "ground-truth labels."
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the fine-tuned image classification "
            "model on a single image or folder."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a single image or image folder.",
    )

    parser.add_argument(
        "--weights",
        default="outputs/best_model.pth",
        help="Path to the trained model weights.",
    )

    parser.add_argument(
        "--out",
        default="outputs/inference/predictions.json",
        help=(
            "Output JSON path for folder inference. "
            "CSV is saved next to it."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for folder inference.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    device = get_device()

    model, classes, img_size = load_finetuned(
        args.weights,
        device,
    )

    model.eval()

    tfm = get_transforms(
        img_size,
        train=False,
    )

    image_path = Path(args.image)

    # -----------------------------------------------------
    # Single image
    # -----------------------------------------------------

    if image_path.is_file():

        if image_path.suffix.lower() not in IMG_EXT:
            raise SystemExit(
                f"Unsupported image format: "
                f"{image_path.suffix}"
            )

        predict_single_image(
            model=model,
            tfm=tfm,
            image_path=image_path,
            classes=classes,
            device=device,
            topk=args.topk,
        )

        return

    # -----------------------------------------------------
    # Folder
    # -----------------------------------------------------

    if image_path.is_dir():

        predict_folder(
            model=model,
            tfm=tfm,
            image_dir=image_path,
            classes=classes,
            device=device,
            topk=args.topk,
            batch_size=args.batch_size,
            out=args.out,
        )

        return

    raise SystemExit(
        f"Image or folder not found: {image_path}"
    )


if __name__ == "__main__":
    main()
