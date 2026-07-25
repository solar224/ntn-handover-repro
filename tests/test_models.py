import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is required")


def test_transformer_forward_shape():
    import torch

    from ntn_repro.models import TransformerTrajectoryModel

    model = TransformerTrajectoryModel(history_length=5, horizon=2, d_model=16, nhead=4, num_layers=1, dim_feedforward=32).module
    y = model(torch.zeros(3, 5, 3))
    assert tuple(y.shape) == (3, 2, 3)

