🌍 EE782 – EuroSAT Classification Using CNN Architectures

This project implements and compares six state-of-the-art deep CNN architectures on the EuroSAT Remote Sensing Dataset using PyTorch.
The work includes training, evaluation, robustness testing, confusion matrices, and performance comparison to determine the most reliable model for satellite image classification.

🚀 Models Implemented

We evaluate two models from each family:

ResNet Family

resnet18

resnet50

DenseNet Family

densenet121

densenet201

EfficientNet Family

efficientnet_b0

efficientnet_b4

These 6 models form the baseline set for all experiments.

📂 Project Structure
EE782-Project/
│
├── train.py                     # Training script (saves best weights & summary)
├── models.py                    # Model factory for all 6 architectures
├── datasets.py                  # Dataloader + augmentations
├── eval_models.py               # Test accuracy + confusion matrices
├── robustness_test.py           # Robustness testing (noise, blur, fog, contrast, occlusion)
│
├── checkpoints/
│   ├── <model>_adamw/
│   │     ├── best_<model>.pth
│   │     ├── loss_curve_<model>.png
│   │     ├── acc_curve_<model>.png
│   │     ├── summary.txt
│   │     └── confusionmatrix_robust/
│   │           ├── confmat_<corruption>_sevX.png
│   │           ├── confmat_<corruption>_sevX_norm.png
│
│   └── robustness_results/
│         ├── accuracies_<corruption>.csv
│         ├── accuracies_<corruption>.png
│         ├── accuracies_all_corruptions.png
│         └── summary_robust.txt
│
├── dataset/                     # Train / Val / Test folders (after split)
└── README.md

📦 Dataset

The project uses the EuroSAT RGB dataset (10 classes):

AnnualCrop

Forest

HerbaceousVegetation

Highway

Industrial

Pasture

PermanentCrop

Residential

River

SeaLake

Dataset splitting:

70% Train
15% Validation
15% Test

🔧 Environment Setup
conda create -n pytorch_gpu python=3.10
conda activate pytorch_gpu

pip install torch torchvision timm matplotlib scikit-learn tqdm tensorboard


Ensure CUDA is working:

import torch
print(torch.cuda.is_available())

🏋️ Training

Each model is trained individually using:

python train.py \
  --model resnet50 \
  --pretrained \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-3 \
  --optimizer adamw \
  --scheduler cosine \
  --img_size 64 \
  --save_dir checkpoints/resnet50_adamw


Training outputs:

best_<model>.pth

loss & accuracy curves

training summary (time, parameters, best acc)

📊 Testing + Confusion Matrix

To evaluate best checkpoints on the test set:

python eval_models.py


Outputs:

Test accuracy for each model

Confusion matrices

accuracies_bar.png

summary.txt

🧪 Robustness Testing

We test all 6 models against six corruption types:

✔ Gaussian Noise
✔ Gaussian Blur
✔ Brightness Changes
✔ Contrast Changes
✔ Occlusion
✔ Fog / Haze

Run:

python robustness_test.py


Outputs:

CSV files with accuracies per severity

Plots for each corruption

Combined plot of all corruptions

Confusion matrices saved per model

summary_robust.txt with rankings and averages

📈 What This Project Provides

Standard accuracy comparison (train/val/test)

Deep robustness evaluation

Per-severity breakdown across all models

Confusion matrices (clean + corrupted inputs)

Training curves and summaries

Parameter count and training speed logs

Perfect for an IEEE-style paper, including:

tables

plots

robustness analysis

model comparison

recommendations

🧠 Which Model is Best?

Based on accuracy, robustness, and efficiency, conclusions can be drawn by analyzing:

summary.txt

summary_robust.txt

All generated plots

Typically:

EfficientNet-B4 is strongest overall

DenseNet121 offers best trade-off

ResNet18 is fastest and smallest
—but your results will provide concrete evidence.
