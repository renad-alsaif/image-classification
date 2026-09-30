"""Train a classification model using ImageNet-pretrained weights and fine-tuning.

No training from scratch: the backbone always starts from pretrained weights.
  Stage 1 (--freeze_epochs): backbone frozen, only the new classifier head is trained.
  Stage 2: whole network unfrozen and fine-tuned with a smaller learning rate.

Usage:
  python train.py --data_dir data --arch resnet50 --epochs 10 --freeze_epochs 2

Shared model and transform helpers are used by inference.py and evaluation.py.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_ARCHS = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]


# --------------------------------------------------------------------------- helpers
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(img_size=224, train=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:  # augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])


def build_pretrained_model(arch, num_classes):
    """Load public ImageNet-pretrained weights and swap in a new classification head."""
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return m


def set_backbone_frozen(model, arch, frozen):
    head_prefix = "fc." if arch == "resnet50" else "classifier."
    for name, p in model.named_parameters():
        p.requires_grad = (not frozen) or name.startswith(head_prefix)


def load_finetuned(weights_path, device):
    """Load a checkpoint saved by this script. Returns (model, classes, img_size)."""
    ckpt = torch.load(weights_path, map_location=device)
    model = build_pretrained_model(ckpt["arch"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt["classes"], ckpt["img_size"]


# --------------------------------------------------------------------------- tuning
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="tune" if train else "val"):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def make_optimizer(model, lr, wd):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in zip(ax, ("loss", "acc"), ("Loss", "Accuracy")):
        a.plot(hist[f"train_{key}"], label="train"); a.plot(hist[f"val_{key}"], label="val")
        a.set_title(title); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="folder containing train/ and val/")
    ap.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--freeze_epochs", type=int, default=2, help="head-only epochs at the start")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for head-only stage")
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="LR for full fine-tuning stage")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Pretrained model: {args.arch}")

    data = Path(args.data_dir)
    train_ds = datasets.ImageFolder(data / "train", get_transforms(args.img_size, train=True))
    val_ds = datasets.ImageFolder(data / "val", get_transforms(args.img_size, train=False))
    classes = train_ds.classes
    assert classes == val_ds.classes, "train/ and val/ must contain the same class folders"
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_pretrained_model(args.arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, optimizer, scheduler = 0.0, None, None
    freeze_epochs = min(args.freeze_epochs, args.epochs)

    for epoch in range(args.epochs):
        if epoch == 0 and freeze_epochs > 0:
            set_backbone_frozen(model, args.arch, True)
            optimizer = make_optimizer(model, args.lr, args.weight_decay)
            print(f"\n=== Stage 1: classifier head only ({freeze_epochs} epochs) ===")
        if epoch == freeze_epochs:
            set_backbone_frozen(model, args.arch, False)
            optimizer = make_optimizer(model, args.ft_lr, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - freeze_epochs))
            print(f"\n=== Stage 2: full fine-tuning ({args.epochs - freeze_epochs} epochs) ===")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[k].append(v)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time()-t0:.0f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state": model.state_dict(), "classes": classes, "arch": args.arch,
                        "img_size": args.img_size, "val_acc": va_acc}, out_dir / "best_model.pth")
            print(f"  -> saved best fine-tuned model (val acc {va_acc:.4f})")

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, out_dir / "tuning_curves.png")
    print(f"\nDone. Best val acc: {best_acc:.4f}. Saved to {out_dir}/best_model.pth")


if __name__ == "__main__":
    main()
