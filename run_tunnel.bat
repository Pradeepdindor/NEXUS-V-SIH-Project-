@echo off
title SIH 26124 - Cloudflare Public Tunnel
cls
IF EXIST "D:\anaconda\python.exe" (
    SET "PY_EXEC=D:\anaconda\python.exe"
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    SET "PY_EXEC=C:\Users\%USERNAME%\anaconda3\python.exe"
) ELSE (
    SET "PY_EXEC=python"
)

"%PY_EXEC%" "%~dp0scripts\start_tunnel.py"
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Running direct cloudflared tunnel fallback...
    "%~dp0cloudflared.exe" tunnel --protocol http2 --url http://127.0.0.1:5000
)
pause
