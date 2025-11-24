import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "EuroSAT"        # folder you downloaded
OUTPUT_DIR = "dataset"         # new folder with splits

os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = ["train", "val", "test"]
for split in splits:
    for cls in os.listdir(DATASET_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, cls)
    images = os.listdir(class_dir)

    train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "train", cls))

    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "val", cls))

    for img in test_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "test", cls))

print("Dataset split complete!")
