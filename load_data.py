"""Download any dataset from Kaggle."""
import argparse
from pathlib import Path

import kagglehub


def main():
    parser = argparse.ArgumentParser(description="Download a dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset identifier, e.g. owner/dataset-name",
    )
    parser.add_argument(
        "--out",
        default="examples/example_classification/dataset",
        help="Directory where the dataset will be downloaded",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        args.dataset,
        output_dir=str(out),
    )

    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
