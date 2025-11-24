# datasets.py
"""
Robust Data pipeline for EuroSAT (RGB) dataset.

Usage:
    from datasets import get_dataloader, dataset_stats, find_corrupted

    dataset_stats("dataset")                     # quick overview
    dl, ds = get_dataloader("dataset", "train")  # dataloader + dataset

Notes:
 - Designed to work in Jupyter and in scripts (num_workers default = 0 in notebooks).
 - Skips unreadable images and reports them.
"""

import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from collections import Counter
from typing import Tuple, List


class EuroSATDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 64, transform=None, skip_invalid: bool = True):
        """
        root_dir: path containing 'train','val','test' folders
        split: "train" / "val" / "test"
        img_size: image size for default transforms
        transform: torchvision transform (overrides default)
        skip_invalid: if True, skip images that cannot be opened (and report them)
        """
        self.split = split
        self.root = os.path.join(root_dir, split)
        if not os.path.isdir(self.root):
            raise ValueError(f"Split directory does not exist: {self.root}")

        # classes are directories inside split folder
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        self.invalid_files = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    full = os.path.join(cls_dir, fname)
                    if skip_invalid:
                        try:
                            # quick open check
                            with Image.open(full) as im:
                                im.verify()  # verify file isn't truncated
                            # If no exception, append
                            self.samples.append((full, self.class_to_idx[cls]))
                        except (UnidentifiedImageError, OSError, ValueError) as e:
                            self.invalid_files.append((full, str(e)))
                    else:
                        self.samples.append((full, self.class_to_idx[cls]))

        # default transform if none provided
        self.transform = transform or self.default_transform(img_size)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image samples found for split '{split}' in {self.root}")

    def default_transform(self, img_size: int):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            # In production you might fallback or raise; we raise to make errors explicit
            raise RuntimeError(f"Failed to open image {path} : {e}")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_invalid_files(self) -> List[Tuple[str, str]]:
        """Return list of (path, error) for files that failed verification at init."""
        return self.invalid_files


def get_transforms(img_size: int = 64, split: str = "train", randaugment: bool = False):
    """Return torchvision transforms for a given split."""
    if split == "train":
        t = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)
        ]
        if randaugment:
            # optional; torchvision >=0.9 includes RandAugment
            try:
                from torchvision.transforms import RandAugment
                t.append(RandAugment())
            except Exception:
                pass
        t += [transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])]
        return transforms.Compose(t)
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(root_dir: str = "dataset",
                   split: str = "train",
                   batch_size: int = 32,
                   img_size: int = 64,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   pin_memory: bool = False,
                   randaugment: bool = False) -> Tuple[DataLoader, EuroSATDataset]:
    """
    Create DataLoader and Dataset for a split.
    Defaults use num_workers=0 (safe for Jupyter). In VSCode scripts you can increase num_workers.
    """
    tf = get_transforms(img_size=img_size, split=split, randaugment=randaugment)
    ds = EuroSATDataset(root_dir=root_dir, split=split, img_size=img_size, transform=tf, skip_invalid=True)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=(split == "train") and shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory)
    return dl, ds


# Utility functions

def dataset_stats(root_dir: str = "dataset"):
    """
    Print counts for train/val/test and per-class counts for each split.
    """
    stats = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            stats[split] = None
            continue
        class_counts = {}
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            cnt = len([f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
            class_counts[cls] = cnt
            total += cnt
        stats[split] = {"total": total, "per_class": class_counts}
    # print nicely
    for split, info in stats.items():
        if info is None:
            print(f"{split}: not found")
        else:
            print(f"{split}: {info['total']} images")
            for cls, cnt in info["per_class"].items():
                print(f"   {cls}: {cnt}")
    return stats


def find_corrupted(root_dir: str = "dataset") -> List[Tuple[str, str]]:
    """
    Scan dataset and try to open all images. Return list of (path, error) for corrupted/unreadable images.
    Use when you suspect some files are broken.
    WARNING: can be slow on large datasets.
    """
    corrupted = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as e:
                    corrupted.append((path, str(e)))
    if len(corrupted) > 0:
        print(f"Found {len(corrupted)} corrupted images.")
    else:
        print("No corrupted images found.")
    return corrupted


# Quick test (only run when executed as script)
if __name__ == "__main__":
    print("Running quick dataset diagnostics for './dataset' ...")
    dataset_stats("dataset")
    dl, ds = get_dataloader("dataset", "train", batch_size=8, img_size=64, num_workers=0, pin_memory=torch.cuda.is_available())
    print("Sample count:", len(ds))
    if len(ds) > 0:
        x, y = next(iter(dl))
        print("Batch shapes:", x.shape, y.shape)
    invalid = ds.get_invalid_files()
    if invalid:
        print("Invalid files detected during init (first 5):")
        for p, e in invalid[:5]:
            print(p, "->", e)
