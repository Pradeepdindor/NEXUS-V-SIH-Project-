import os
from pathlib import Path

base = Path("dataset")
n1_img_dir = base / "n1" / "images"
n1_lbl_dir = base / "n1" / "labels-YOLO"

n1_imgs = {f.stem: f for f in n1_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
n1_lbls = {f.stem: f for f in n1_lbl_dir.glob("*.txt")}

common_n1 = set(n1_imgs.keys()) & set(n1_lbls.keys())
print(f"N1: Total images = {len(n1_imgs)}, Total labels = {len(n1_lbls)}, Matched pairs = {len(common_n1)}")

n2_train_img_dir = base / "n2" / "train" / "images"
n2_train_lbl_dir = base / "n2" / "train" / "labels"
n2_train_imgs = {f.stem: f for f in n2_train_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
n2_train_lbls = {f.stem: f for f in n2_train_lbl_dir.glob("*.txt")}
common_n2_train = set(n2_train_imgs.keys()) & set(n2_train_lbls.keys())
print(f"N2 Train: Total images = {len(n2_train_imgs)}, Total labels = {len(n2_train_lbls)}, Matched pairs = {len(common_n2_train)}")

n2_val_img_dir = base / "n2" / "valid" / "images"
n2_val_lbl_dir = base / "n2" / "valid" / "labels"
n2_val_imgs = {f.stem: f for f in n2_val_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
n2_val_lbls = {f.stem: f for f in n2_val_lbl_dir.glob("*.txt")}
common_n2_val = set(n2_val_imgs.keys()) & set(n2_val_lbls.keys())
print(f"N2 Valid: Total images = {len(n2_val_imgs)}, Total labels = {len(n2_val_lbls)}, Matched pairs = {len(common_n2_val)}")
