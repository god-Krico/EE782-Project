# 🛰️ EE782 – Assignment 2: Satellite Image Classification using Deep CNNs

## 👨‍💻 Group Members
This project was completed by:

* **Karan Satarkar** – 23B0708
* **Ashwin Mankar** – 23B0726

---

## ✨ Project Overview
This assignment focuses on building an AI system that can accurately classify remote-sensing satellite images from the **EuroSAT dataset** into 10 different land-use categories.

We thoroughly evaluated the performance and robustness of six state-of-the-art deep learning architectures across three model families:

* **ResNet Family:** **ResNet18**, **ResNet50**
* **DenseNet Family:** **DenseNet121**, **DenseNet201**
* **EfficientNet Family:** **EfficientNet-B0**, **EfficientNet-B4**

Each model was trained, evaluated, and stress-tested for robustness using the **PyTorch** framework.

---

## 📦 Core Libraries & Dependencies
The following core libraries were used for this project:

| Library | Purpose |
| :--- | :--- |
| **PyTorch** (`torch`, `torchvision`) | Core framework for model training, evaluation, and GPU acceleration. |
| **timm** (`PyTorch Image Models`) | Provides highly optimized implementations of **ResNet**, **DenseNet**, and **EfficientNet** architectures. |
| **scikit-learn** (`sklearn`) | Used for generating confusion matrices and calculating performance metrics. |
| **matplotlib** | For generating visual plots (accuracy/loss curves, robustness graphs). |
| **numpy** | Core library for numerical and array processing. |
| **tqdm** | Provides elegant progress bars during training and evaluation loops. |
| **Pillow** (`PIL`) | Used for image loading and applying various corruptions (blur, noise, fog) for robustness testing. |

---

## 🌍 Dataset Used: EuroSAT
We use the **EuroSAT RGB dataset** derived from Sentinel-2 satellite images. It consists of 10 distinct land-use categories:

* **AnnualCrop**
* **Forest**
* **HerbaceousVegetation**
* **Highway**
* **Industrial**
* **Pasture**
* **PermanentCrop**
* **Residential**
* **River**
* **SeaLake**

The dataset is automatically split for training and testing purposes:
* **70%** — Train
* **15%** — Validation
* **15%** — Test

---

## 🚀 How to Run the Project

### 1. Install Dependencies
It is highly recommended to use a Conda environment for GPU compatibility.

```bash
# Create and activate a new environment
conda create -n pytorch_gpu python=3.10
conda activate pytorch_gpu

# Install all necessary packages
pip install torch torchvision timm matplotlib scikit-learn tqdm pillow tensorboard
```
All the codes would also run if you do not have CUDA integration. It would instead use the CPU but would take longer to train.

### 2. Prepare the Dataset

Simply run the provided **create_dataset.py** to automatically split, and organize the EuroSAT dataset into the required `train`, `val`, and `test` folders.
You may need to specify the location of the downloaded dataset in the code. Also, we have excluded the analysis of multispectral images (folder named **all bands**), so you exclude that too.

### 3. Training a Model
Use the `train.py` script with arguments to specify the model and training parameters.

**Example Command (Training ResNet50):**

```bash
python train.py 
    --model resnet50 
    --pretrained 
    --epochs 20 
    --batch_size 64 
    --lr 1e-3 
    --optimizer adamw 
    --scheduler step
    --save_dir checkpoints/resnet50_adamw
```
Run this command on your VScode Terminal to start with the training of individual models.

### 4. Evaluating All 6 Models
You simply need to run the **eval_models.py** to test all the models.
```bash
python eval_models.py
```
This generates:
*Test accuracy for every model
*Confusion matrices (confmat_<model>.png)
*accuracies_bar.png
*summary.txt
