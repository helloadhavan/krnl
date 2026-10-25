# krnl

`krnl` is a lightweight, educational machine learning library built in Python on top of NumPy, with experimental OpenCL support for device-aware tensor computation. The project is designed to help understand the core ideas behind modern deep learning frameworks: tensors, autograd, computational graphs, layers, losses, and optimization.

This library aims to stay small and readable rather than production-focused. It is intentionally useful for learning how neural networks and training loops work under the hood.

## Highlights

- Tensor-based autograd engine
- NumPy-backed CPU execution
- OpenCL backend experiments for GPU-style computation
- Basic neural network building blocks
- Optimizer implementations for training
- Common loss functions for regression and learning tasks
- Simple, traceable code for educational use

## Installation

Clone the repository:

```bash
git clone https://github.com/helloadhavan/krnl.git
cd krnl
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional OpenCL setup

To use the OpenCL backend, make sure your system has a compatible OpenCL runtime and drivers installed.

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

## Core concepts

### Tensor

The `Tensor` class is the foundation of the project. It stores numeric data, tracks computation history, and supports differentiation through custom arithmetic and autograd operations.

Key features include:

- scalar and array arithmetic
- forward pass tracking
- backward gradient propagation
- device awareness for CPU/GPU execution
- support for differentiable operations used in model training

### Device abstraction

The `Device` abstraction allows the library to reason about whether computation should happen on CPU or a GPU-like backend. This keeps the design simple while making it easier to experiment with accelerator-backed operations.

### OpenCL backend

The OpenCL path uses PyOpenCL to move arrays to an accelerator, execute simple kernels, and support basic tensor operations. It is useful for experimentation and understanding how low-level device execution fits into a training loop.

## Neural network utilities

The library includes a minimal `src/nn` module with building blocks for simple models.

### Module and Parameter

`Module` is the base class for neural network components. It provides a simple interface for registering trainable parameters.

`Parameter` wraps a tensor and marks it as trainable so that it participates in gradient-based optimization.

### Layers

The library includes a few basic layer types:

- `Linear`: a dense layer implementing `y = xW^T + b`
- `ReLU`: elementwise rectified linear activation
- `Sigmoid`: elementwise sigmoid activation

Example:

```python
from src.nn.model_utils import Linear, ReLU
from src.backends.tensor import Tensor

layer1 = Linear(in_features=4, out_features=8)
activation = ReLU()

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = activation(layer1(x))
```

## Loss functions

The project includes a small set of regression-style loss implementations in `src/nn/loss.py`.

### MSELoss

Mean Squared Error:

```python
(predicted - target) ** 2
```

This computes the squared error across the prediction and is commonly used for regression tasks.

### MAELoss

Mean Absolute Error:

```python
abs(predicted - target)
```

This is useful when you want the average magnitude of the prediction error without squaring it.

### HuberLoss

A smooth piecewise loss that behaves like squared error for small residuals and absolute error for larger residuals. It is often used to reduce sensitivity to outliers.

Example:

```python
from src.nn.loss import MSELoss, MAELoss, HuberLoss

criterion = MSELoss()
loss = criterion(predictions, targets)
```

## Optimizers

The library includes several optimization strategies in `src/nn/optim.py`.

### SGD

Stochastic Gradient Descent updates parameters by moving them in the direction opposite the gradient:

```python
parameter = parameter - learning_rate * gradient
```

This is the simplest gradient-based optimizer and is useful for understanding the basics of training.

### Adam

Adam combines momentum and adaptive learning rates using first and second moment estimates. It is one of the most common modern optimizers and is useful when training neural networks more stably than plain SGD.

### Muon

Muon is a momentum-based optimizer with an orthogonalized update direction. It introduces a more structure-aware update strategy and is included as an experimental optimizer in the library.

### MuonAdam

This optimizer combines Muon-style orthogonal updates with Adam-like moment estimates, making it a hybrid approach for experimentation and learning.

Example:

```python
from src.nn.optim import SGD, Adam, Muon, MuonAdam

sgd = SGD(parameters=model_parameters, learning_rate=0.01)
adam = Adam(parameters=model_parameters, learning_rate=0.001)
muon = Muon(parameters=model_parameters, learning_rate=0.02)
muon_adam = MuonAdam(parameters=model_parameters, learning_rate=0.02)
```

## Training workflow

A typical training loop looks like this:

```python
from src.nn.loss import MSELoss
from src.nn.optim import SGD

model = ...
criterion = MSELoss()
optimizer = SGD(model._parameters.values(), learning_rate=0.01)

for batch_x, batch_y in data:
    predictions = model(batch_x)
    loss = criterion(predictions, batch_y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

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

The OpenCL tests require a valid OpenCL installation and compatible device support.

## Project status

`krnl` is an educational framework for understanding the internals of deep learning systems. It is not intended to replace established libraries such as PyTorch or TensorFlow.

Current focus areas include:

- tensor arithmetic and autograd
- gradient-based optimization
- CPU execution with NumPy
- GPU/OpenCL experimentation
- small neural network primitives

## Contributing

Contributions are welcome in areas such as:

- more layer types
- additional loss functions
- more optimization algorithms
- clearer documentation
- stronger tests
- device and performance improvements

The project remains intentionally simple and readable so it can serve as a learning tool.

## License

This repository does not currently include a license file. If you plan to distribute or publish it publicly, consider adding a license such as MIT or Apache 2.0.

## Acknowledgements

This project is built for experimentation and education in tensor math, computational graphs, optimization, and device-aware execution.
