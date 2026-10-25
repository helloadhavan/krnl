"""
Krnl - A simple deep learning library
"""

from backends.tensor import *
from backends.device import Device
from backends.GPU_processer import OpenCLBackend

import nn

__all__ = [
    'Tensor',
    'zeros_like',
    'full_like',
    'Device',
    'OpenCLBackend',
    'ones_like',
    'nn'
]

__version__ = '0.1.0'
__author__ = 'Adhavan'
__email__ = 'helloadhavan@gmail.com'