@echo off
title SIH 26124 - Streamlit Judge Dashboard
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

echo Starting Streamlit Demonstration Dashboard on http://localhost:8501 ...
"%PY_EXEC%" -m streamlit run frontend/app.py
pause
