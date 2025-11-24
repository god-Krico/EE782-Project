# robustness_test.py
"""
Robustness test suite for the six baseline models.

Usage:
    python robustness_test.py --batch_size 64 --img_size 64

Outputs (checkpoints/robustness_results/):
 - accuracies_<corruption>.csv    (rows: severity, cols: models)
 - accuracies_<corruption>.png    (overlay plot of accuracy vs severity for all models)
 - summary.txt

Notes:
 - Expects the best checkpoints to be at:
     checkpoints/best_resnet18.pth
     checkpoints/best_resnet50.pth
     checkpoints/best_densenet121.pth
     checkpoints/best_densenet201.pth
     checkpoints/best_efficientnet_b0.pth
     checkpoints/best_efficientnet_b4.pth
 - Uses torchvision transforms + custom corruption transforms applied on-the-fly.
 - Uses ImageNet normalization (modify if your training used different stats).
"""

import argparse
from pathlib import Path
import time
import csv
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import create_model

# ---------- Settings ----------
TARGET_MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b4",
]

CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR = CHECKPOINT_DIR / "robustness_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet normalization (change if your pipeline used different means/stds)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------- Corruption helpers ----------
class AddGaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, img):
        # img: PIL -> convert to numpy
        a = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.std, a.shape).astype(np.float32)
        a = a + noise
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).astype(np.uint8)
        return Image.fromarray(a)

class GaussianBlurPIL(object):
    def __init__(self, radius=1.0):
        self.radius = radius
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class BrightnessAdjust(object):
    def __init__(self, factor=1.0):
        self.factor = factor
    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self.factor)

class RandomOcclusion(object):
    def __init__(self, occ_frac=0.2, occ_square=False, seed=None):
        """
        occ_frac: fraction of image area to occlude (0..1)
        """
        self.occ_frac = occ_frac
        self.occ_square = occ_square
        self.rng = np.random.RandomState(seed)
    def __call__(self, img):
        w, h = img.size
        area = w * h
        occ_area = int(self.occ_frac * area)
        # choose a rectangle size (attempt square-ish)
        if self.occ_square:
            side = int(math.sqrt(occ_area))
            x1 = self.rng.randint(0, max(1, w - side))
            y1 = self.rng.randint(0, max(1, h - side))
            x2 = x1 + side
            y2 = y1 + side
        else:
            # choose random rect such that width*height ~ occ_area
            rw = max(1, int(w * min(0.9, 0.5 + self.rng.rand()*0.5)))
            rh = max(1, int(occ_area / rw))
            x1 = self.rng.randint(0, max(1, w - rw))
            y1 = self.rng.randint(0, max(1, h - rh))
            x2 = min(w, x1 + rw)
            y2 = min(h, y1 + rh)
        img_np = np.array(img)
        img_np[y1:y2, x1:x2, :] = 0  # black occlusion
        return Image.fromarray(img_np.astype(np.uint8))

# ---------- Utility functions ----------
def build_transform(img_size, corruption=None):
    """
    corruption is a tuple: (name, severity_value) or None.
    Supported corruption names: 'noise', 'blur', 'brightness', 'occlusion'
    severity_value: numeric (std for noise, radius for blur, factor for brightness, fraction for occlusion)
    """
    transform_list = [transforms.Resize((img_size, img_size))]
    if corruption is None:
        pass
    else:
        name, val = corruption
        if name == "noise":
            transform_list.append(AddGaussianNoise(std=val))
        elif name == "blur":
            transform_list.append(GaussianBlurPIL(radius=val))
        elif name == "brightness":
            transform_list.append(BrightnessAdjust(factor=val))
        elif name == "occlusion":
            transform_list.append(RandomOcclusion(occ_frac=val))
        else:
            raise ValueError("Unknown corruption: " + str(name))

    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(transform_list)

def load_checkpoint_into_model(model, ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    elif isinstance(ck, dict) and all(k.startswith("module.") for k in ck.keys()):
        sd = ck
    else:
        sd = ck
    # strip module.
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "") if isinstance(k, str) else k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    return model

@torch.no_grad()
def evaluate_model_on_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / max(1, total)

# ---------- Main routine ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset", help="root dataset folder (must have train/val/test)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default auto)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)
    batch_size = args.batch_size
    img_size = args.img_size

    # confirm available checkpoints for all target models
    ckpt_map = {}
    for m in TARGET_MODELS:
        ckpt = CHECKPOINT_DIR / f"{m}_adamw" / f"best_{m}.pth"
        if ckpt.exists():
            ckpt_map[m] = ckpt
        else:
            print(f"WARNING: checkpoint missing for {m} at {ckpt}; it will be skipped.")

    if not ckpt_map:
        print("No checkpoints found for any target models. Exiting.")
        return

    # Define corruptions and severity levels
    corruptions = {
        "noise": [0.00, 0.02, 0.04, 0.06, 0.08],            # gaussian std
        "blur":  [0.0, 0.5, 1.0, 1.5, 2.0],                # gaussian blur radius
        "brightness": [1.0, 0.8, 0.6, 0.4, 0.2],           # brightness factor (<1 darker)
        "occlusion": [0.0, 0.05, 0.10, 0.20, 0.30]         # fraction of image occluded
    }

    results_all = {c: {} for c in corruptions.keys()}  # store accuracies: results_all[corruption][model] = [vals per severity]

    # We'll iterate corruption types
    for corr_name, severity_list in corruptions.items():
        print("\n=== Corruption:", corr_name, " severities:", severity_list)
        # create a dataset/loader for each severity and evaluate all models on it
        # We'll keep one loader per severity to avoid re-reading files too often.
        loaders = []
        for sev in severity_list:
            if corr_name == "noise":
                corr = ("noise", sev)
            elif corr_name == "blur":
                corr = ("blur", sev)
            elif corr_name == "brightness":
                corr = ("brightness", sev)
            elif corr_name == "occlusion":
                corr = ("occlusion", sev)
            else:
                corr = None
            transform = build_transform(img_size, corruption=corr if sev > 0 else None)
            ds = datasets.ImageFolder(root=str(Path(args.data_root) / "test"), transform=transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
            loaders.append((sev, loader))
            if sev == severity_list[0]:
                class_names = ds.classes
                print("  Found classes:", class_names)

        # Evaluate each model on all severities
        for model_name, ckpt_path in ckpt_map.items():
            accs = []
            print(" Evaluating model:", model_name)
            # create model instance
            try:
                model = create_model(name=model_name, num_classes=len(class_names), pretrained=False, in_chans=3, freeze_backbone=False)
            except Exception as e:
                print("  Failed to create model via dispatcher:", e)
                continue
            try:
                model = load_checkpoint_into_model(model, ckpt_path, device)
            except Exception as e:
                print("  Failed to load checkpoint for", model_name, ":", e)
                continue
            model = model.to(device)
            model.eval()
            # evaluate each severity loader
            for sev, loader in loaders:
                print(f"   severity {sev} ... ", end="", flush=True)
                acc = evaluate_model_on_loader(model, loader, device)
                print(f"{acc:.4f}")
                accs.append(acc)
            results_all[corr_name][model_name] = accs

    # Save CSVs and plots per corruption
    for corr_name, sev_map in results_all.items():
        severities = corruptions[corr_name]
        csv_path = OUT_DIR / f"accuracies_{corr_name}.csv"
        # header: severity, model1, model2, ...
        models_sorted = sorted(sev_map.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["severity"] + models_sorted)
            for i, sev in enumerate(corruptions[corr_name]):
                row = [sev]
                for m in models_sorted:
                    row.append(sev_map[m][i] if m in sev_map else "")
                writer.writerow(row)
        print("Wrote CSV:", csv_path)

        # plot overlay
        plt.figure(figsize=(8,5))
        for m in models_sorted:
            y = sev_map.get(m, [math.nan]*len(severities))
            plt.plot(severities, y, marker='o', label=m)
        plt.xlabel("Severity")
        plt.ylabel("Test accuracy")
        plt.title(f"Robustness: {corr_name}")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        # For tight ranges, zoom the y-axis automatically
        all_vals = []
        for m in models_sorted:
            all_vals.extend([v for v in sev_map.get(m, []) if v is not None])
        if all_vals:
            ymin = max(0.0, min(all_vals) - 0.01)
            ymax = min(1.0, max(all_vals) + 0.01)
            # if values are close, zoom tighter
            if ymax - ymin < 0.1:
                ymin = max(0.0, min(all_vals) - 0.02)
            plt.ylim(ymin, ymax)
        out_png = OUT_DIR / f"accuracies_{corr_name}.png"
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print("Saved plot:", out_png)

    # Save a short summary
    with open(OUT_DIR / "summary_robust.txt", "w") as f:
        f.write(f"Robustness test run at {time.ctime()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Models evaluated: {sorted(list(ckpt_map.keys()))}\n")
        f.write("Corruptions:\n")
        for c, sev in corruptions.items():
            f.write(f"  {c}: {sev}\n")
    print("Done. Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
