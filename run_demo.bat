@echo off
title SIH 26124 - Mobile Urban Intelligence Platform
cls
echo ===============================================================================
echo   SMART INDIA HACKATHON 26124: AI MOBILE URBAN INTELLIGENCE PLATFORM
echo ===============================================================================
echo   Searching for active Anaconda Python runtime...

IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo   Using Python: %PY_EXEC%
echo ===============================================================================
"%PY_EXEC%" run_demo.py
pause
