#!/usr/bin/env python
"""Wrapper script to run clean training from project root"""
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

import warnings
warnings.filterwarnings('ignore', message='.*OpenTelemetry.*')
warnings.filterwarnings('ignore', message='.*localhost:4318.*')

from src.train import main

if __name__ == "__main__":
    main()