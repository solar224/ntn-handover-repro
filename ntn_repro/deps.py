from __future__ import annotations

import importlib.util


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_numpy():
    if not has_module("numpy"):
        raise RuntimeError("NumPy is required. Install dependencies with: python -m pip install -r requirements.txt")
    import numpy as np

    return np


def require_torch():
    if not has_module("torch"):
        raise RuntimeError("PyTorch is required. Install dependencies with: python -m pip install -r requirements.txt")
    import torch

    return torch


def require_matplotlib():
    if not has_module("matplotlib"):
        raise RuntimeError("Matplotlib is required. Install dependencies with: python -m pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    return plt

