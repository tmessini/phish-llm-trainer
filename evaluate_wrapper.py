#!/usr/bin/env python
"""Wrapper script to run model evaluation from project root"""
import os
import sys
os.environ['OTEL_SDK_DISABLED'] = 'true'

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluate import main

if __name__ == "__main__":
    main()