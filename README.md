# krnl

`krnl` is a lightweight, educational machine learning library written in Python on top of NumPy, with experimental OpenCL support for device-aware tensor computation. It is designed to help developers and students understand how neural networks, autograd, and training loops work under the hood.

The project intentionally stays small, readable, and easy to follow rather than aiming to match the scale or production readiness of larger libraries such as PyTorch or TensorFlow.

## Highlights

- Tensor-based autograd engine
- NumPy-backed CPU execution
- Experimental OpenCL backend for GPU-style computation
- Basic neural network building blocks
- Optimizer implementations for training
- Common loss functions for regression and learning tasks
- Clear, traceable code for educational use

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install krnl
```

### Optional OpenCL setup

To use the OpenCL backend, make sure your system has a compatible OpenCL runtime and drivers installed.

## Quick start

```python
import numpy as np

from krnl import nn
from krnl import Tensor


def main():
    dx = []
    dy = []

    for i in range(10):
        dx.append(float(i))
        dy.append(float(i * 2 + 1))

    model = nn.Linear(
        in_features=1,
        out_features=1,
        bias=True,
    )

    criterion = nn.loss.MSELoss()

    optimizer = nn.optim.MuonAdamAux(
        model._parameters.values(),
        learning_rate=0.001,
    )

    weight = model._parameters["weight"]
    bias = model._parameters["bias"]

    print("Initial weight:", weight.data, "| Initial bias:", bias.data)

    for epoch in range(100):
        total_loss = 0.0

        for x_value, y_value in zip(dx, dy):
            optimizer.zero_grad()

            x = Tensor([[x_value]])
            target = Tensor([[y_value]])

            prediction = model(x)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += float(np.asarray(loss.data).reshape(-1)[0])

        if epoch % 10 == 0:
            average_loss = total_loss / len(dx)
            print(
                f"epoch={epoch:03d}, "
                f"loss={average_loss:.6f}, "
                f"weight={weight.data}, "
                f"bias={bias.data}"
            )

    print("\nFinal predictions:")

    for x_value, expected_value in zip(dx, dy):
        x = Tensor([[x_value]])
        prediction = model(x)
        print(
            f"x={x_value:.1f}, "
            f"predicted={prediction.data}, "
            f"expected={expected_value:.1f}"
        )


if __name__ == "__main__":
    main()
```

This example builds a small computational graph, computes a scalar loss, and backpropagates gradients through the graph.

## Core concepts

### Tensor

The `Tensor` class is the foundation of the project. It stores numeric data, tracks computation history, and supports differentiation through custom arithmetic and autograd operations.

Key features include:

- scalar and array arithmetic
- forward-pass tracking
- backward gradient propagation
- device awareness for CPU/GPU execution
- support for differentiable operations used in model training

### Device abstraction

The `Device` abstraction allows the library to reason about whether computation should happen on CPU or a GPU-like backend. This keeps the design simple while making it easier to experiment with accelerators and custom execution paths.

### OpenCL backend

The OpenCL path uses PyOpenCL to move arrays to an accelerator, execute simple kernels, and support basic tensor operations. It is useful for experimentation and for understanding how low-level device execution fits into a training pipeline.

## Neural network utilities

The library includes a minimal `nn` module with building blocks for simple models.

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
from nn.model_utils import Linear, ReLU
from backends.backends.tensor import Tensor

layer1 = Linear(in_features=4, out_features=8)
activation = ReLU()

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = activation(layer1(x))
```

## Loss functions

The project includes a small set of regression-style loss implementations in `nn`.

### MSELoss

Mean squared error:

```python
(predicted - target) ** 2
```

This computes the squared error across the prediction and is commonly used for regression tasks.

### MAELoss

Mean absolute error:

```python
abs(predicted - target)
```

This is useful when you want the average magnitude of the prediction error without squaring it.

### HuberLoss

A smooth piecewise loss that behaves like squared error for small residuals and absolute error for larger residuals. It is often used to reduce sensitivity to outliers.

Example:

```python
from nn import MSELoss, MAELoss, HuberLoss

criterion = MSELoss()
loss = criterion(predictions, targets)
```

## Optimizers

The library includes several optimization strategies in `nn`.

### SGD

Stochastic Gradient Descent updates parameters by moving them in the direction opposite the gradient:

```python
parameter = parameter - learning_rate * gradient
```

This is the simplest gradient-based optimizer and is useful for understanding the basics of training.

### Adam

Adam combines momentum and adaptive learning rates using first and second moment estimates. It is one of the most common modern optimizers and is useful for training neural networks more stably than plain SGD.

### Muon

Muon is a momentum-based optimizer with an orthogonalized update direction. It introduces a more structure-aware update strategy and is included as an experimental optimizer in the library.

### MuonAdam

This optimizer combines Muon-style orthogonal updates with Adam-like moment estimates, making it a hybrid approach for experimentation and learning.

Example:

```python
from nn.optim import SGD, Adam, Muon, MuonAdam

sgd = SGD(parameters=model_parameters, learning_rate=0.01)
adam = Adam(parameters=model_parameters, learning_rate=0.001)
muon = Muon(parameters=model_parameters, learning_rate=0.02)
muon_adam = MuonAdam(parameters=model_parameters, learning_rate=0.02)
```

## Training workflow

A typical training loop looks like this:

```python
from nn import MSELoss
from nn.optim import SGD

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
