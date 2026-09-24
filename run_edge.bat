@echo off
title SIH 26124 - Mobile Edge Sensing AI Pipeline
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo Starting Edge Sensing Pipeline from Synthetic Video Feed...
"%PY_EXEC%" model/edge_sensing.py --source demo --save-video
pause
