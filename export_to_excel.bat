@echo off
title SIH 26124 - Export Database to Excel
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo Exporting SQLite road hazard database to Microsoft Excel CSV...
"%PY_EXEC%" scripts\export_to_excel.py
pause
