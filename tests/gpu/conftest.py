import pytest


def pytest_collection_modifyitems(config, items):
    """Skip gpu-marked tests when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return
    except ImportError:
        pass
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
