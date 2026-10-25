"""Backends package for krnl."""

from .device import Device
from .GPU_processer import OpenCLBackend, get_gpu_info
from .tensor import (
    Tensor,
    full_like,
    ones_like,
    rand,
    rand_like,
    zeros_like,
)

__all__ = [
    "Tensor",
    "Device",
    "OpenCLBackend",
    "get_gpu_info",
    "ones_like",
    "zeros_like",
    "full_like",
    "rand",
    "rand_like",
]
