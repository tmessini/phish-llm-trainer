@echo off
echo Setting environment variables...
set OTEL_SDK_DISABLED=true
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Starting training...
echo.

REM Run from the project root directory
cd /d %~dp0

echo Running: python train_wrapper.py
python train_wrapper.py

pause