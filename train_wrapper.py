#!/usr/bin/env python
"""Wrapper script to run full training from project root"""
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

from src.train import main

if __name__ == "__main__":
    main()