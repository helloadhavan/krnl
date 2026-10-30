"""krnl public package API."""

from . import nn
from .backends.save import save
from .backends.load import load
from .backends import (
    Device,
    OpenCLBackend,
    Tensor,
    full_like,
    get_gpu_info,
    ones_like,
    rand,
    rand_like,
    zeros_like,
)

__version__ = "0.3.0"
__author__ = "Adhavan"
__email__ = "helloadhavan@gmail.com"

__all__ = [
    "Tensor",
    "Device",
    "OpenCLBackend",
    "get_gpu_info",
    "ones_like",
    "zeros_like",
    "full_like",
    "rand_like",
    "rand",
    "save",
    "load",
    "nn",
]
