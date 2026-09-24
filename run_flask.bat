@echo off
title SIH 26124 - Flask REST API & Leaflet GIS Server
cls
SET "KMP_DUPLICATE_LIB_OK=TRUE"
SET "PYTHONIOENCODING=utf-8"
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo Starting Flask REST API and Leaflet GIS Server on http://127.0.0.1:5000 ...
"%PY_EXEC%" backend/flask_app.py
pause
