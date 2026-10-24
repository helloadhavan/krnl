# krnl

krnl is a lightweight Python library for experimenting with tensor operations, automatic differentiation, and device-aware execution. It is designed as a small educational framework for working with NumPy-backed CPU tensors and an OpenCL-backed GPU backend.

The project currently focuses on:

- Tensor math with basic autograd support
- CPU execution using NumPy
- GPU execution using PyOpenCL
- Simple training-style examples using gradient descent
- A minimal device abstraction for CPU and GPU selection

## Features

- Custom `Tensor` class with basic forward and backward pass support
- Element-wise addition and multiplication
- Scalar power operations
- Gradient accumulation for simple learning tasks
- OpenCL backend for float32 GPU arrays
- Matrix multiplication via a custom OpenCL kernel
- CPU/GPU device metadata access

## Project status

This repository is a small experimental library rather than a production-ready deep learning framework. It is useful for learning how a tensor library is structured, how autograd works, and how OpenCL can be used for accelerated numeric workloads.

## Requirements

- Python 3.12+
- NumPy
- PyOpenCL
- psutil
- An OpenCL-compatible GPU runtime (for GPU execution)

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to install the main libraries manually:

```bash
pip install numpy pyopencl psutil
```

> OpenCL support depends on the host system having a valid OpenCL runtime and compatible GPU drivers installed.

## Repository structure

```text
krnl/
├── README.md
├── requirements.txt
├── version.txt
└── src/
    ├── backends/
    │   ├── GPU_processer.py
    │   ├── device.py
    │   ├── kernels.py
    │   └── tensor.py
    └── tests/
        ├── test_opencl_backend.py
        ├── test_simple_model_implementation.py
        └── test_tensor.py
```

## Core components

### Tensor class

The `src/backends/tensor.py` module defines a minimal tensor implementation with automatic differentiation. It supports:

- `+`, `-`, `*`, and `**` operations
- gradient accumulation via `backward()`
- scalar loss-based optimization routines

Example:

```python
from src.backends.tensor import Tensor

x = Tensor(2.0)
w = Tensor(3.0, requires_grad=True)
b = Tensor(4.0, requires_grad=True)

y = x * w + b
loss = (y - 10.0) ** 2

loss.backward()

print(y.data)
print(w.grad)
print(b.grad)
```

### Device abstraction

The `Device` class in `src/backends/device.py` provides a simple interface for representing CPU or GPU devices. It can report hardware information and is used by the tensor layer to choose an execution target.

### OpenCL backend

The GPU backend in `src/backends/GPU_processer.py` uses PyOpenCL to:

- create an OpenCL context and command queue
- upload NumPy arrays to the GPU
- perform element-wise add/multiply
- run matrix multiplication with a custom OpenCL kernel

Example:

```python
import numpy as np
from src.backends.GPU_processer import OpenCLBackend

backend = OpenCLBackend()

a = backend.upload(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = backend.upload(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

print(backend.add(a, b))
print(backend.multiply(a, b))
print(backend.matmul(a, b))
```

## Training example

The project includes a simple optimization example in `src/tests/test_simple_model_implementation.py`.

```python
from src.backends import tensor
from random import randint

x = tensor.Tensor(randint(1, 10))
desired = x * tensor.Tensor(randint(1, 4)) + tensor.Tensor(randint(1, 2))

weight = tensor.Tensor(randint(1, 10), requires_grad=True)
bias = tensor.Tensor(randint(1, 10), requires_grad=True)

for _ in range(100):
    weight.grad = None
    bias.grad = None

    y = x * weight + bias
    error = y - desired
    loss = error ** 2

    loss.backward()

    learning_rate = 0.01
    weight.data -= learning_rate * weight.grad
    bias.data -= learning_rate * bias.grad
```

This demonstrates a minimal gradient-descent workflow using the custom `Tensor` type.

## Running tests

From the project root:

```bash
python src/tests/test_tensor.py
python src/tests/test_opencl_backend.py
```

The OpenCL tests require a working OpenCL installation and compatible device.

## Current scope

This library currently provides a compact foundation for experimenting with:

- tensor arithmetic
- backpropagation for simple scalar losses
- NumPy-backed CPU computations
- PyOpenCL-backed GPU computations

The implementation is intentionally minimal and is well suited for learning how tensor frameworks are structured.

## License

This project does not currently declare a license file. If you plan to distribute or use it publicly, you may want to add one explicitly.

## Contributing

Contributions are welcome for:

- broader tensor operations
- better autodiff coverage
- more robust device handling
- performance improvements
- additional tests
- documentation and examples

If you are extending the project, keep the code simple and educational so it remains easy to understand for experimentation and learning.
