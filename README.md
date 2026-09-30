# Image Classification Pipeline

A simple, reusable **image classification pipeline** .


The project supports **two main use cases**:

1. **Generic Image Classification** — use the full pipeline for any single-label image classification dataset.
2. **Vehicle Classification** — a complete example showing how the same pipeline can be applied to a vehicle classification task.

The core code is shared between both use cases. Only the dataset, classes, and output directory change.

---

## Workflow

The pipeline follows a simple end-to-end workflow:

```text
Load Data → Split → Train / Fine-Tune → Inference → Evaluation
    │          │             │                │           │
load_data.py split.py      train.py      inference.py evaluation.py
```

![alt text](use-case/vehicle_classification/workflow.png)

### 1. Load Data

Download a classification dataset from Kaggle using its dataset identifier.

```bash
python load_data.py \
    --dataset owner/dataset-name \
    --out use-case/example_classification/dataset
```

> A local dataset can also be used directly; this step  is optional.

### 2. Split

Split the dataset into train, validation, and test sets.

```bash
python split.py \
    --src use-case/example_classification/dataset \
    --dst use-case/example_classification/data \
    --train 70 \
    --val 15 \
    --test 15
```

### 3. Train / Fine-Tune

Train an ImageNet-pretrained model on the target dataset.

```bash
python train.py \
    --data_dir use-case/example_classification/data \
    --arch resnet50 \
    --epochs 10 \
    --freeze_epochs 2 \
    --out_dir use-case/example_classification/outputs
```

Training uses two stages: classifier-head training followed by full fine-tuning.

### 4. Inference

Run the trained model on a single image or a folder of images.

```bash
python inference.py \
    --image use-case/example_classification/data/test \
    --weights use-case/example_classification/outputs/best_model.pth \
    --out use-case/example_classification/outputs/inference/predictions.json
```

### 5. Evaluation

Evaluate predictions and generate classification metrics and visualisations.

```bash
python evaluation.py \
    --predictions use-case/example_classification/outputs/inference/predictions.json \
    --out_dir use-case/example_classification/outputs/eval
```

Evaluation includes accuracy, precision, recall, F1-score, confusion matrix, per-class F1, and prediction visualisations.

---

## Project Structure

```text
image-classification-pipeline/
├── README.md
├── requirements.txt
│
├── load_data.py          # Download a dataset from Kaggle
├── split.py              # Train / validation / test split
├── train.py              # Pretrained model + fine-tuning
├── inference.py          # Run predictions
├── evaluation.py         # Evaluate predictions
├── metric.py             # Classification metrics
├── vis.py                # Evaluation visualisations
│
├── README/
│   ├── generic_classification.md
│   └── vehicle_classification.md
│
└── use-case/
    └── vehicle_classification/
        ├── dataset/      # Downloaded/original dataset
        ├── data/         # Train / val / test split
        └── outputs/      # Model and evaluation results
```

---

## Use Case 1 — Generic Image Classification

The pipeline can be used for any **single-label image classification** dataset.

Expected dataset format:

```text
my_dataset/
├── class_a/
│   ├── image_01.jpg
│   └── image_02.jpg
├── class_b/
│   ├── image_01.jpg
│   └── image_02.jpg
└── class_c/
    └── image_01.jpg
```

Class names are detected automatically from the folder names.

See [Generic Image Classification](README/generic_classification.md) for the complete workflow.

---

## Use Case 2 — Vehicle Classification

Vehicle Classification is an example application of the generic pipeline.

The repository includes the trained model and evaluation outputs from the vehicle classification experiment.

See [Vehicle Classification](README/vehicle_classification.md) for dataset details and reproduction steps.

---

## Supported Models

The pipeline uses ImageNet-pretrained models:

| Argument | Architecture |
|---|---|
| `resnet50` | ResNet-50 |
| `efficientnet_b0` | EfficientNet-B0 |
| `mobilenet_v3_large` | MobileNetV3-Large |

The number of classes is detected automatically from the dataset.

---

## Training Strategy

The training process uses **transfer learning followed by fine-tuning**:

1. **Classifier-head training** — the pretrained backbone is frozen.
2. **Full fine-tuning** — the complete model is unfrozen and trained with a smaller learning rate.

This allows the same pipeline to be reused for different classification datasets, including small datasets.

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. A GPU is optional but recommended for training.

---

## Documentation

- [Generic Image Classification](README/generic_classification.md)
- [Vehicle Classification](README/vehicle_classification.md)



## Credits

**Author:** Renad Alsaif

This project was developed by **Renad Alsaif** as part of the **SDAIA Academy** program.

**SDAIA Academy:** https://github.com/SDAIAAcademy
