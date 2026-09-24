@echo off
title SIH 26124 - FastAPI Central Command Backend
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
"%PY_EXEC%" backend/server.py
pause
