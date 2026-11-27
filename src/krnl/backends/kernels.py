"""
OpenCL C code definitions for kernels.

Each constant is loaded at import time from the corresponding .c file
in ``src/krnl/kernels/``.
"""

from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"


def _read_kernel(filename: str) -> str:
    """Return the contents of *filename* inside the kernels directory."""
    return (_KERNEL_DIR / filename).read_text()


ELEMENTWISE_KERNEL = _read_kernel("elementwise.c")
ELEMENTWISE_SCALER_KERNELS = _read_kernel("scalar.c")
MATMUL_KERNEL = _read_kernel("matmul.c")
TRANSPOSE_KERNEL = _read_kernel("transpose.c")
SOFTMAX_KERNEL = _read_kernel("softmax.c")
SIGMOID_KERNEL = _read_kernel("sigmoid.c")
INDEX_KERNEL = _read_kernel("index.c")
