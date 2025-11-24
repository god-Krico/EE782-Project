# eval_models.py
"""
Evaluate ONLY the six baseline best models on test set.

Models evaluated:
    best_resnet18.pth
    best_resnet50.pth
    best_densenet121.pth
    best_densenet201.pth
    best_efficientnet_b0.pth
    best_efficientnet_b4.pth

Outputs (in checkpoints/eval_results/):
 - accuracies_bar.png
 - confmat_<model>.png
 - summary.txt
"""

import torch
import os
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

from models import create_model
from datasets import get_dataloader

# ---------------------------------------------------------------
# ** Only these six models will be evaluated **
# ---------------------------------------------------------------
TARGET_MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b4"
]

CHECKPOINT_ROOT = Path("checkpoints")
OUT_DIR = CHECKPOINT_ROOT / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def load_checkpoint(model, ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state_dict = ck["model_state_dict"]
    else:
        state_dict = ck
    # fix potential module. prefixes
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=False)
    return model


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            p = out.argmax(dim=1)
            preds.extend(p.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    cm = confusion_matrix(labels_all, preds)
    return acc, cm


def plot_confusion(cm, classes, outpath, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        text = f"{val:.2f}" if normalize else f"{int(val)}"
        plt.text(j, i, text, ha="center", color="white" if val > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():

    # ------------------------------------------------------------
    # Find ONLY the best_*.pth for our six models
    # ------------------------------------------------------------
    available_ckpts = {}

    for model_name in TARGET_MODELS:
        pattern = f"best_{model_name}.pth"
        found = list(CHECKPOINT_ROOT.rglob(pattern))
        if found:
            available_ckpts[model_name] = found[0]
        else:
            print(f"WARNING: {pattern} NOT FOUND, skipping.")

    if not available_ckpts:
        print("No matching best model checkpoints found.")
        return

    test_loader, test_ds = get_dataloader(
        root_dir="dataset",
        split="test",
        batch_size=64,
        img_size=64,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    class_names = test_ds.classes

    results = []

    for model_name, ckpt_path in available_ckpts.items():
        print(f"\nEvaluating {model_name} -> {ckpt_path}")

        # Build model
        model = create_model(
            name=model_name,
            num_classes=len(class_names),
            pretrained=False,
            in_chans=3
        ).to(DEVICE)

        # Load checkpoint
        model = load_checkpoint(model, ckpt_path, DEVICE)

        # Evaluate
        acc, cm = evaluate(model, test_loader, DEVICE)
        print(f"  Test Accuracy: {acc:.4f}")

        results.append((model_name, acc, cm))

        # Save confusion matrix images
        plot_confusion(cm, class_names,
                       OUT_DIR / f"confmat_{model_name}.png",
                       normalize=False,
                       title=f"Confusion Matrix: {model_name}")

        plot_confusion(cm, class_names,
                       OUT_DIR / f"confmat_{model_name}_norm.png",
                       normalize=True,
                       title=f"Normalized Confusion Matrix: {model_name}")

    # ------------------------------------------------------------
    # Plot test accuracy comparison bar chart
    # ------------------------------------------------------------
    results.sort(key=lambda x: x[1], reverse=True)

    names = [r[0] for r in results]
    accs = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    xs = np.arange(len(names))

    plt.bar(xs, accs, color="steelblue")
    plt.xticks(xs, names, rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.96, 1.0)
    plt.title("Test Accuracy Comparison (Six Baseline Models)")

    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "accuracies_bar.png")
    plt.close()

    # Save summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Model\tTestAcc\n")
        for name, acc, cm in results:
            f.write(f"{name}\t{acc:.4f}\n")

    print("\nEvaluation complete.")
    print("Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
