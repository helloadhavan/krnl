# krnl

krnl is a small, experimental machine learning library built in Python with NumPy and OpenCL. It is designed for learning how tensor libraries work, how automatic differentiation is implemented, and how CPU/GPU execution can be exposed through a very small API.

This project is intentionally lightweight and educational rather than production-ready. It focuses on the core ideas behind modern ML frameworks: tensors, computational graphs, gradients, and simple device-aware execution.

## Features

- Custom tensor implementation with forward and backward passes
- Element-wise arithmetic for scalar and array operations
- Gradient accumulation for training-style workflows
- Scalar loss computation and optimization using backpropagation
- Minimal device abstraction for CPU and GPU selection
- OpenCL backend for float32 GPU computations
- Simple neural network building blocks such as `Linear` and `ReLU`

## Installation

Clone the repository:

```bash
git clone https://github.com/helloadhavan/krnl.git
cd krnl
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional GPU setup

To use the OpenCL backend, make sure your system has a valid OpenCL runtime and compatible GPU drivers installed.

## Quick start

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

This example creates a small computational graph, computes a scalar loss, and backpropagates gradients through the graph.

## Core components

### Tensor

The `Tensor` class is the foundation of the library. It stores data, tracks device placement, and supports differentiation through custom `__add__`, `__sub__`, `__mul__`, and `__pow__` implementations. The library also includes a `backward()` method for reverse-mode autodiff.

### Device abstraction

The `Device` class provides a simple CPU/GPU abstraction. It lets the rest of the library reason about where a tensor should be computed and what hardware features are available.

### OpenCL backend

The GPU path uses PyOpenCL to upload arrays to the device, perform operations, and run a basic matrix multiplication kernel. This makes the project useful for learning how accelerator-backed tensor operations fit into a small ML stack.

### Neural network utilities

The project includes a minimal `src/nn` layer for building simple modules, including:

- `Module`
- `Parameter`
- `Linear`
- `ReLU`

These utilities are intentionally simple and easy to trace through when learning how training loops and model building blocks work.

## Repository structure

```text
krnl/
├── README.md
├── __init__.py
├── requirements.txt
├── version.txt
├── src/
│   ├── backends/
│   │   ├── GPU_processer.py
│   │   ├── __init__.py
│   │   ├── device.py
│   │   ├── kernels.py
│   │   └── tensor.py
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── model_utils.py
│   │   └── optim.py
│   └── tests/
│       ├── test_opencl_backend.py
│       ├── test_simple_model_implementation.py
│       └── test_tensor.py
└── .gitignore
```

## Running tests

From the project root:

```bash
python src/tests/test_tensor.py
python src/tests/test_opencl_backend.py
```

The OpenCL tests depend on an available OpenCL installation and a compatible device.

## Project status

This library is a small educational framework for building intuition around deep learning internals. It is not intended to replace mature libraries such as PyTorch or TensorFlow.

Current focus areas include:

- tensor arithmetic
- basic gradient-based optimization
- CPU execution with NumPy
- GPU execution with OpenCL
- minimal neural network primitives

## Contributing

Contributions are welcome for:

- additional tensor operations
- more robust autograd support
- improved device handling
- better model utilities
- performance improvements
- tests and documentation

If you are extending the project, keep the implementation simple and readable so it remains useful as a learning tool.

## License

This project does not currently include a license file. If you plan to distribute or publish it publicly, consider adding an explicit license such as MIT or Apache 2.0.

## Acknowledgements

This project is built for experimentation and education in the areas of tensor math, computational graphs, and device-aware execution.
