# Generic Image Classification

This use case shows how to apply the pipeline to any **single-label image classification** dataset.

## Workflow

```text
Kaggle / Local Dataset
        ↓
    Load Data
        ↓
      Split
        ↓
 Train / Fine-Tune
        ↓
    Inference
        ↓
   Evaluation
```

## 1. Load a Dataset from Kaggle

Use the Kaggle dataset identifier in the format `owner/dataset-name`:

```bash
python load_data.py \
    --dataset owner/dataset-name \
    --out examples/example_classification/dataset
```

The loader is generic and is not tied to a specific dataset.

If the dataset is already available locally, skip this step.

## 2. Dataset Format

The source dataset should contain one folder per class:

```text
my_dataset/
├── class_a/
├── class_b/
└── class_c/
```

Each image should belong to one class.

## 3. Split the Dataset

```bash
python split.py \
    --src examples/example_classification/dataset \
    --dst examples/example_classification/data \
    --train 70 \
    --val 15 \
    --test 15
```

Output:

```text
examples/example_classification/data/
├── train/
├── val/
└── test/
```

The split is performed separately for each class and is reproducible with `--seed`.

## 4. Train / Fine-Tune

```bash
python train.py \
    --data_dir examples/example_classification/data \
    --arch resnet50 \
    --epochs 10 \
    --freeze_epochs 2 \
    --out_dir examples/example_classification/outputs
```

Available architectures:

- `resnet50`
- `efficientnet_b0`
- `mobilenet_v3_large`

The model starts from ImageNet-pretrained weights and automatically replaces the classification head for the target number of classes.

Training outputs:

```text
outputs/
├── best_model.pth
├── history.json
└── tuning_curves.png
```

## 5. Inference

For a folder:

```bash
python inference.py \
    --image examples/example_classification/data/test \
    --weights examples/example_classification/outputs/best_model.pth \
    --out examples/example_classification/outputs/inference/predictions.json
```

For a single image:

```bash
python inference.py \
    --image path/to/image.jpg \
    --weights examples/example_classification/outputs/best_model.pth
```

The output contains the predicted class, confidence, and top-k predictions.

## 6. Evaluation

```bash
python evaluation.py \
    --predictions examples/example_classification/outputs/inference/predictions.json \
    --out_dir examples/example_classification/outputs/eval
```

Evaluation produces:

```text
outputs/eval/
├── metrics.json
├── confusion_matrix.png
├── per_class_f1.png
├── predictions.json
└── vis/
```

Metrics include accuracy, precision, recall, F1-score, and per-class results.

## Notes

- The model predicts only the classes it was trained on.
- For a new dataset, train a new model rather than reusing a model trained for another task.
- The pipeline supports single-label image classification, not object detection or segmentation.





## Credits

**Author:** Renad Alsaif

This project was developed by **Renad Alsaif** as part of the **SDAIA Academy** program.

**SDAIA Academy:** https://github.com/SDAIAAcademy
