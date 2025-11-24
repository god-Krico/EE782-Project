# train.py
"""
Train script for EuroSAT experiments with loss & accuracy plotting.

Usage example:
    conda activate pytorch_gpu
    python train.py --data dataset --model resnet50 --pretrained --epochs 25 --batch_size 64 --lr 0.01 --optimizer sgd --scheduler step --save_dir checkpoints

Generates:
 - checkpoints/best_<model>.pth
 - checkpoints/loss_curve_<model>.png  (train & val loss per epoch)
 - checkpoints/acc_curve_<model>.png   (train & val accuracy per epoch)
 - checkpoints/summary.txt             (training summary: params, time, best val acc, etc.)
"""

import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from datasets import get_dataloader
from models import create_model, count_parameters

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device):
    """
    Evaluate model on dataloader; returns (avg_loss, accuracy).
    Guards against empty dataloaders by returning (0.0, 0.0).
    """
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    if total == 0:
        return 0.0, 0.0
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def save_plots(save_dir: Path, train_losses, val_losses, train_accs, val_accs, model_name: str):
    """Save loss and accuracy plots to save_dir with model_name in titles & file names."""
    epochs = list(range(1, len(train_losses) + 1))
    title_suffix = f" ({model_name})"

    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss' + title_suffix)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    loss_path = save_dir / f"loss_curve_{model_name}.png"
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Val Accuracy' + title_suffix)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    acc_path = save_dir / f"acc_curve_{model_name}.png"
    plt.savefig(acc_path)
    plt.close()

def _format_seconds(sec: float) -> str:
    """Format seconds into H:M:S string."""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------
# Training loop
# -----------------------
def train_loop(args):
    seed_everything(args.seed)

    # cuDNN performance/repro tradeoff
    # Set deterministic=True if you need reproducible runs (slower). Default: benchmark=True for speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = None
    if args.tensorboard:
        tb_dir = save_dir / "runs"
        writer = SummaryWriter(log_dir=str(tb_dir))

    # DataLoaders
    pin_memory = True if device.type == "cuda" else False
    train_loader, train_ds = get_dataloader(root_dir=args.data, split="train",
                                            batch_size=args.batch_size, img_size=args.img_size,
                                            num_workers=args.num_workers, pin_memory=pin_memory,
                                            randaugment=args.randaugment)
    val_loader, val_ds = get_dataloader(root_dir=args.data, split="val",
                                        batch_size=args.batch_size, img_size=args.img_size,
                                        num_workers=args.num_workers, pin_memory=pin_memory,
                                        randaugment=False)

    num_classes = len(train_ds.classes)
    print(f"Found {len(train_ds)} train samples, {len(val_ds)} val samples, {num_classes} classes")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train or Val dataset is empty. Check your dataset splits and paths.")

    # Model
    model = create_model(name=args.model, num_classes=num_classes, pretrained=args.pretrained, in_chans=3, freeze_backbone=False)
    total_params, trainable_params = count_parameters(model, verbose=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer: choose 'sgd' or 'adamw'")

    # Scheduler
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_epoch = -1
    global_step = 0

    # arrays for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            global_step += 1
            if writer:
                writer.add_scalar("train/loss_step", loss.item(), global_step)

        epoch_time = time.time() - epoch_start_time

        # scheduler step (per epoch)
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)

        # record for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # save plots each epoch so you can monitor progress mid-run
        try:
            save_plots(save_dir, train_losses, val_losses, train_accs, val_accs, model_name=args.model)
        except Exception as e:
            print("Warning: failed to save plots:", e)

        epoch_time_msg = f"{epoch_time:.1f}s"
        epoch_time_total = time.time() - epoch_start_time
        epoch_time_avg = epoch_time  # per-epoch measured above

        epoch_time_str = _format_seconds(epoch_time_avg)

        epoch_time_elapsed = time.time() - total_start_time

        print(f"Epoch {epoch} done | epoch_time={epoch_time_str} | train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if writer:
            writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            # log LR
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("train/lr", current_lr, epoch)

        # Save best checkpoint by val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": vars(args)
            }
            torch.save(ckpt, save_dir / f"best_{args.model}.pth")
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {save_dir}")

        # periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, save_dir / f"epoch_{epoch}_{args.model}.pth")

    total_time = time.time() - total_start_time
    if writer:
        writer.close()

    # Save a summary file with timing and parameter info
    summary_path = save_dir / "summary.txt"
    try:
        with open(summary_path, "w") as f:
            f.write("Training summary\n")
            f.write("================\n")
            f.write(f"Start time: {datetime.fromtimestamp(total_start_time).isoformat()}\n")
            f.write(f"End time:   {datetime.fromtimestamp(time.time()).isoformat()}\n")
            f.write(f"Total training time (H:M:S): {_format_seconds(total_time)}\n")
            per_epoch = total_time / max(1, args.epochs)
            f.write(f"Avg time per epoch (s): {per_epoch:.2f}\n")
            f.write("\nModel & dataset\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Num classes: {num_classes}\n")
            f.write("\nParameters\n")
            f.write(f"Total parameters: {total_params}\n")
            f.write(f"Trainable parameters: {trainable_params}\n")
            f.write("\nTraining summary\n")
            f.write(f"Epochs run: {args.epochs}\n")
            f.write(f"Best val acc: {best_val_acc:.6f} at epoch {best_epoch}\n")
            f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
            f.write(f"Final train acc:  {train_accs[-1]:.6f}\n")
            f.write(f"Final val loss:   {val_losses[-1]:.6f}\n")
            f.write(f"Final val acc:    {val_accs[-1]:.6f}\n")
            f.write("\nCLI args\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")
        print("Saved training summary to", summary_path)
    except Exception as e:
        print("Warning: failed to write summary:", e)

    print("Training finished. Best val_acc:", best_val_acc)
    print(f"Loss and accuracy curves saved to {save_dir / f'loss_curve_{args.model}.png'} and {save_dir / f'acc_curve_{args.model}.png'}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset", help="path to dataset root (train/val/test)")
    parser.add_argument("--model", type=str, default="resnet50", help="model name (timm)")
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "none"])
    parser.add_argument("--step_size", type=int, default=10, help="stepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="stepLR gamma")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true", help="write tensorboard logs")
    parser.add_argument("--randaugment", action="store_true", help="use RandAugment in training transforms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true", help="force CPU even if CUDA is available")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
