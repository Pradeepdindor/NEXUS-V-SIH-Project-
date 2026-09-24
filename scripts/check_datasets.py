import os
from pathlib import Path

base = Path("dataset")
n1_img = list((base / "n1" / "images").glob("*.*"))
n1_lbl = list((base / "n1" / "labels-YOLO").glob("*.txt"))
print(f"N1: {len(n1_img)} images, {len(n1_lbl)} YOLO label files")

n2_train_img = list((base / "n2" / "train" / "images").glob("*.*"))
n2_train_lbl = list((base / "n2" / "train" / "labels").glob("*.txt"))
n2_val_img = list((base / "n2" / "valid" / "images").glob("*.*"))
n2_val_lbl = list((base / "n2" / "valid" / "labels").glob("*.txt"))
print(f"N2 train: {len(n2_train_img)} images, {len(n2_train_lbl)} labels")
print(f"N2 valid: {len(n2_val_img)} images, {len(n2_val_lbl)} labels")

# Let's inspect class distributions in N1
n1_classes = {}
for lf in n1_lbl:
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = parts[0]
                n1_classes[cid] = n1_classes.get(cid, 0) + 1

print(f"N1 class occurrences: {n1_classes}")

# Let's inspect class distributions in N2
n2_classes = {}
for lf in n2_train_lbl + n2_val_lbl:
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = parts[0]
                n2_classes[cid] = n2_classes.get(cid, 0) + 1

print(f"N2 class occurrences: {n2_classes}")
