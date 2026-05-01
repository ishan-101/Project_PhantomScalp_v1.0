"""
Futures Feature Engineering Module

This module exposes the main entry point for computing all futures features.
"""

from .pipeline import run_futures_pipeline

__all__ = [
    "run_futures_pipeline",
]