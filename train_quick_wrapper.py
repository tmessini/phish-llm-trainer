#!/usr/bin/env python
"""Wrapper script to run quick training from project root"""
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

from src.train_quick import main

if __name__ == "__main__":
    main()