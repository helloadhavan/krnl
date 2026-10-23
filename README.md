# krnl

`krnl` is a Python tensor library with CPU and GPU backends.

The CPU backend uses NumPy. The GPU backend uses PyOpenCL to execute operations on an OpenCL-compatible device.

## Features

- CPU tensor operations using NumPy
- GPU operations using PyOpenCL
- Upload NumPy arrays to the GPU
- Element-wise tensor addition
- Element-wise tensor multiplication
- Matrix multiplication using a custom OpenCL kernel
- Initial support for `float32` tensors

## Requirements

- Python 3.12
- NumPy
- PyOpenCL
- An OpenCL-compatible device and runtime for GPU operations

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install numpy pyopencl
```
OR

```bash
pip install -r requirements.txt
```

PyOpenCL also requires a working OpenCL implementation for your hardware. Installing the Python package alone may not be sufficient.

## Project Structure

The project may be organized as follows:

```text
krnl/
├── backends/
│   ├── kernels.py
│   └── GPU_processer.py
├── tests/
│   └── test_opencl_backend.py
└── README.md
```

The file names above are illustrative; adjust them to match the actual project structure.

## GPU Backend

The OpenCL backend manages the OpenCL context and command queue. Arrays are uploaded to the device before GPU operations are performed.

### Example

```python
import numpy as np
from backends.GPU_processer import OpenCLBackend

backend = OpenCLBackend()

a = backend.upload(
    np.array([[1, 2], [3, 4]], dtype=np.float32)
)

b = backend.upload(
    np.array([[5, 6], [7, 8]], dtype=np.float32)
)

addition = backend.add(a, b)
multiplication = backend.multiply(a, b)
matrix_product = backend.matmul(a, b)

print(addition.get())
print(multiplication.get())
print(matrix_product.get())
```

Output:

```text
[[ 6.  8.]
 [10. 12.]]

[[ 5. 12.]
 [21. 32.]]

[[19. 22.]
 [43. 50.]]
```

The `.get()` method copies a device array to host memory as a NumPy array.

## Supported Operations

### Addition

Adds corresponding elements of two tensors with matching shapes.

```python
result = backend.add(a, b)
```

### Element-wise multiplication

Multiplies corresponding elements of two tensors with matching shapes.

```python
result = backend.multiply(a, b)
```

### Matrix multiplication

Multiplies two two-dimensional tensors. The number of columns in the first matrix must equal the number of rows in the second matrix.

```python
result = backend.matmul(a, b)
```

For example, a matrix with shape `(2, 3)` can be multiplied by a matrix with shape `(3, 4)`. The resulting matrix has shape `(2, 4)`.

## Data Types

The initial GPU implementation supports `float32`.

Example:

```python
data = np.array(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=np.float32
)
```

Unsupported data types should raise a `TypeError`.

## Testing

Run the GPU backend test from the project root:

```bash
python tests/test_opencl_backend.py
```

GPU tests require a working OpenCL runtime and a compatible device.

## Current Limitations

- GPU operations initially support `float32`.
- The initial matrix multiplication kernel prioritizes correctness over performance.
- Broadcasting is not implemented in the described GPU operations.
- GPU-to-CPU transfers occur when retrieving results with `.get()`.
- Automatic device selection and CPU/GPU dispatch depend on the surrounding library implementation.

## Development

The library is under development. Future work may include:

- A unified `Tensor` interface
- Backend selection and operation dispatch
- Additional tensor operations
- More data types
- Shape and broadcasting support
- Optimized GPU kernels
- Automated tests for CPU and GPU backends
