"""
Smart India Hackathon - Problem Statement 26124 / 24124
AI-Powered Mobile Urban Intelligence Platform
Script: Prepare Fresh Unified Dataset from N1 and N2 (scripts/prepare_fresh_dataset.py)
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
N1_DIR = DATASET_DIR / "n1"
N2_DIR = DATASET_DIR / "n2"
OUT_DIR = DATASET_DIR / "fresh_dataset"

def build_fresh_dataset():
    print("=" * 80)
    print("  🚀 BUILDING FRESH UNIFIED DATASET (N1 + N2) FOR YOLOV8 TRAINING")
    print("=" * 80)
    print(f"  * Project Root:  {BASE_DIR}")
    print(f"  * N1 Source:     {N1_DIR}")
    print(f"  * N2 Source:     {N2_DIR}")
    print(f"  * Target Output: {OUT_DIR}")
    print("=" * 80)

    # 1. Clear previous fresh_dataset if exists
    if OUT_DIR.exists():
        print(f"[PREP] Clearing existing {OUT_DIR}...")
        shutil.rmtree(OUT_DIR)

    # 2. Create directory structure
    train_img_dir = OUT_DIR / "train" / "images"
    train_lbl_dir = OUT_DIR / "train" / "labels"
    val_img_dir = OUT_DIR / "val" / "images"
    val_lbl_dir = OUT_DIR / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Process N2 (Potholes)
    # N2 train
    n2_train_imgs = list((N2_DIR / "train" / "images").glob("*.*"))
    print(f"\n[PREP] Copying N2 train set ({len(n2_train_imgs)} images)...")
    n2_train_count = 0
    for img_path in n2_train_imgs:
        stem = img_path.stem
        lbl_path = N2_DIR / "train" / "labels" / f"{stem}.txt"
        if lbl_path.exists():
            dest_img = train_img_dir / f"n2_{img_path.name}"
            dest_lbl = train_lbl_dir / f"n2_{stem}.txt"
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            n2_train_count += 1

    # N2 val
    n2_val_imgs = list((N2_DIR / "valid" / "images").glob("*.*"))
    print(f"[PREP] Copying N2 val set ({len(n2_val_imgs)} images)...")
    n2_val_count = 0
    for img_path in n2_val_imgs:
        stem = img_path.stem
        lbl_path = N2_DIR / "valid" / "labels" / f"{stem}.txt"
        if lbl_path.exists():
            dest_img = val_img_dir / f"n2_{img_path.name}"
            dest_lbl = val_lbl_dir / f"n2_{stem}.txt"
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            n2_val_count += 1

    # 4. Process N1 (Potholes, Cracks, Manholes)
    n1_img_dir = N1_DIR / "images"
    n1_lbl_dir = N1_DIR / "labels-YOLO"
    n1_imgs = [p for p in n1_img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(n1_imgs)

    split_idx = int(len(n1_imgs) * 0.80)
    n1_train = n1_imgs[:split_idx]
    n1_val = n1_imgs[split_idx:]

    print(f"\n[PREP] Copying N1 train set ({len(n1_train)} images, 80% split)...")
    n1_train_count = 0
    for img_path in n1_train:
        stem = img_path.stem
        lbl_path = n1_lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            dest_img = train_img_dir / f"n1_{img_path.name}"
            dest_lbl = train_lbl_dir / f"n1_{stem}.txt"
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            n1_train_count += 1

    print(f"[PREP] Copying N1 val set ({len(n1_val)} images, 20% split)...")
    n1_val_count = 0
    for img_path in n1_val:
        stem = img_path.stem
        lbl_path = n1_lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            dest_img = val_img_dir / f"n1_{img_path.name}"
            dest_lbl = val_lbl_dir / f"n1_{stem}.txt"
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            n1_val_count += 1

    # 5. Create fresh data.yaml
    # Normalize path with forward slashes for cross-platform compatibility
    yaml_path_str = str(OUT_DIR).replace("\\", "/")
    yaml_content = f"""# ==============================================================================
# Fresh Custom YOLOv8 Road Defect Dataset (N1 + N2)
# Smart India Hackathon - Problem Statement 26124 / 24124
# Classes: 0: pothole, 1: crack, 2: manhole
# ==============================================================================

path: {yaml_path_str}
train: train/images
val: val/images

nc: 3
names:
  0: pothole
  1: crack
  2: manhole
"""

    yaml_file = OUT_DIR / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Also update dataset/dataset_template.yaml to mirror this fresh dataset
    template_file = DATASET_DIR / "dataset_template.yaml"
    with open(template_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 6. Verify and compute class distribution
    print("\n" + "=" * 80)
    print("  📊 FRESH DATASET VERIFICATION & METRICS")
    print("=" * 80)
    total_train = len(list(train_img_dir.glob("*.*")))
    total_val = len(list(val_img_dir.glob("*.*")))
    print(f"  * Total Training Images:   {total_train} (N2: {n2_train_count} + N1: {n1_train_count})")
    print(f"  * Total Validation Images: {total_val} (N2: {n2_val_count} + N1: {n1_val_count})")
    print(f"  * Total Combined Images:   {total_train + total_val}")
    print(f"  * Config YAML saved to:    {yaml_file}")

    train_classes = {}
    for lf in train_lbl_dir.glob("*.txt"):
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if p:
                    train_classes[p[0]] = train_classes.get(p[0], 0) + 1

    val_classes = {}
    for lf in val_lbl_dir.glob("*.txt"):
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if p:
                    val_classes[p[0]] = val_classes.get(p[0], 0) + 1

    name_map = {"0": "pothole", "1": "crack", "2": "manhole"}
    print("\n  Class Instances in Training Set:")
    for cid in sorted(train_classes.keys()):
        print(f"    - Class {cid} ({name_map.get(cid, 'unknown')}): {train_classes[cid]} instances")

    print("\n  Class Instances in Validation Set:")
    for cid in sorted(val_classes.keys()):
        print(f"    - Class {cid} ({name_map.get(cid, 'unknown')}): {val_classes[cid]} instances")

    print("=" * 80)
    print("  ✅ FRESH DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    build_fresh_dataset()
