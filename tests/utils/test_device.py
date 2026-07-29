"""Test device detection."""
import torch
from src.utils.device import get_device

def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)

def test_get_device_is_cpu_or_cuda():
    device = get_device()
    assert str(device) in ["cpu", "cuda"]
