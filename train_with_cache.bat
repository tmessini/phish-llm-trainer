@echo off
echo Setting Hugging Face cache to D: drive...
set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache\transformers
set HF_DATASETS_CACHE=D:\huggingface_cache\datasets
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Disabling OpenTelemetry to avoid connection warnings...
set OTEL_SDK_DISABLED=true

echo Creating cache directories...
mkdir D:\huggingface_cache 2>nul
mkdir D:\huggingface_cache\transformers 2>nul
mkdir D:\huggingface_cache\datasets 2>nul

echo Starting training...
python train_wrapper.py