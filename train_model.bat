@echo off
title SIH 26124 - Custom YOLOv8 Road Defect Model Trainer
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

SET "KMP_DUPLICATE_LIB_OK=TRUE"
SET "PYTHONIOENCODING=utf-8"

echo Starting Custom YOLOv8 Road Defect Training on Fresh Unified N1+N2 Dataset...
"%PY_EXEC%" dataset/train_model.py --data dataset/fresh_dataset/data.yaml --epochs 30 --imgsz 640 --batch 16 --device 0
pause

