"""
Smart India Hackathon - Problem Statement 24124 / 26124
AI-Powered Mobile Urban Intelligence Platform
Module: Custom YOLOv8 Road Defect Model Trainer (dataset/train_model.py)
"""

import os
# Fix OpenMP runtime duplicate issue on Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# Resolve Project Root & Paths
FILE_DIR = Path(__file__).resolve().parent
BASE_DIR = FILE_DIR.parent if FILE_DIR.name == "dataset" else FILE_DIR
MODELS_DIR = BASE_DIR / "model" / "weights"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CUSTOM_WEIGHTS = MODELS_DIR / "urban_defects_yolov8.pt"
DEFAULT_DATA_YAML = (
    BASE_DIR / "dataset" / "fresh_dataset" / "data.yaml"
    if (BASE_DIR / "dataset" / "fresh_dataset" / "data.yaml").exists()
    else BASE_DIR / "dataset" / "dataset_template.yaml"
)
DEFAULT_BASE_MODEL = (
    MODELS_DIR / "yolov8n.pt"
    if (MODELS_DIR / "yolov8n.pt").exists()
    else (BASE_DIR / "yolov8n.pt" if (BASE_DIR / "yolov8n.pt").exists() else "yolov8n.pt")
)


def train_custom_road_detector(
    data_yaml: str = str(DEFAULT_DATA_YAML),
    base_model: str = str(DEFAULT_BASE_MODEL),
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    workers: int = 2,
    save_as_default: bool = True,
):
    """
    Trains a custom YOLOv8 model on Road Damage Dataset (RDD_SPLIT)
    and exports the best checkpoint directly to model/weights/urban_defects_yolov8.pt.
    """
    # Determine execution device
    if device == "auto" or device == "0":
        if torch.cuda.is_available():
            chosen_device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            dev_desc = f"NVIDIA GPU 0 ({gpu_name})"
        else:
            chosen_device = "cpu"
            dev_desc = "CPU (PyTorch CUDA not found)"
    else:
        chosen_device = device
        dev_desc = f"Specified device ({device})"

    print("=" * 80)
    print("  🚀 SIH URBAN INTELLIGENCE | CUSTOM YOLOv8 ROAD DEFECT MODEL TRAINING")
    print("=" * 80)
    print(f"  * Base Weights:     {base_model}")
    print(f"  * Dataset Config:   {data_yaml}")
    print(f"  * Epochs:           {epochs}")
    print(f"  * Image Resolution: {img_size}x{img_size}")
    print(f"  * Batch Size:       {batch_size}")
    print(f"  * Compute Hardware: {dev_desc}")
    print(f"  * Dataloader Workers: {workers}")
    print(f"  * Target Output:    {DEFAULT_CUSTOM_WEIGHTS}")
    print("=" * 80)

    # 1. Initialize YOLOv8 Model
    print(f"\n[TRAIN] Initializing YOLO model from '{base_model}'...")
    model = YOLO(base_model)

    # 2. Start Training Loop
    print("\n[TRAIN] Starting training on road defect dataset...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=chosen_device,
        patience=15,          # Early stopping patience
        save=True,
        workers=workers,
        project=str(BASE_DIR / "runs" / "detect"),
        name="road_defects_train",
        exist_ok=True,
        verbose=True,
    )

    # 3. Save / Export Trained Model Checkpoint
    train_save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else (BASE_DIR / "runs" / "detect" / "road_defects_train")
    best_pt_path = train_save_dir / "weights" / "best.pt"

    if best_pt_path.exists() and save_as_default:
        shutil.copy(best_pt_path, DEFAULT_CUSTOM_WEIGHTS)
        print("\n" + "=" * 80)
        print("  🎉 TRAINING COMPLETE & MODEL CHECKPOINT DEPLOYED!")
        print(f"  * Source Checkpoint: {best_pt_path}")
        print(f"  * Deployed Weights:  {DEFAULT_CUSTOM_WEIGHTS}")
        print("=" * 80)
    elif best_pt_path.exists():
        print(f"\n[INFO] Best weights saved at: {best_pt_path}")
    else:
        print("\n[WARNING] Could not find best.pt checkpoint.")

    # 4. Run Validation
    print("\n[VAL] Evaluating model on validation set...")
    val_results = model.val()
    print("\n[VAL] Validation Results:")
    print(f"  * mAP50:    {val_results.box.map50:.4f}")
    print(f"  * mAP50-95: {val_results.box.map:.4f}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Custom Road Defect Datasets for SIH 26124")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_BASE_MODEL), help="Pretrained base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce to 8 or 4 if memory constrained)")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: 'auto', '0', or 'cpu'")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader worker processes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_road_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        workers=args.workers,
    )
