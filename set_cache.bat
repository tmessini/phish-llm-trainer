@echo off
echo Setting Hugging Face cache to D: drive...
set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache\transformers
set HF_DATASETS_CACHE=D:\huggingface_cache\datasets

echo Creating cache directories...
mkdir D:\huggingface_cache 2>nul
mkdir D:\huggingface_cache\transformers 2>nul
mkdir D:\huggingface_cache\datasets 2>nul

echo Environment variables set. Now run your Python script.
echo python src/train.py