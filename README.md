# krnl

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/krnl?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/krnl) 
![python](https://img.shields.io/badge/language-python-blue?style=flat-square)

`krnl` is a lightweight, educational machine learning library written in Python on top of NumPy, with experimental OpenCL support for device-aware tensor computation.

The project intentionally stays small, readable, and easy to follow rather than being production ready.

📖 [Documentation](https://krnl.readthedocs.io/en/latest/)

## Highlights

- Tensor-based autograd engine
- NumPy-backed CPU execution
- Experimental OpenCL backend for GPU-style computation
- Neural network building blocks (Linear, ReLU, Sigmoid, Softmax, Tanh, GELU, SiLU, Sequential, Residual)
- Loss functions (MSE, MAE, Huber, BCE, CrossEntropy)
- Optimizers (SGD, Adam, Muon, MuonAdam, MuonAdamAux)
- Dataset and DataLoader for mini-batch training
- High-level Trainer API with configurable training arguments
- Model saving and loading

## Installation

```bash
pip install krnl
```

### Optional: OpenCL setup

To use the OpenCL backend, make sure your system has a compatible OpenCL runtime and drivers installed.

## Quick start

```python
import numpy as np
from krnl import nn, Tensor

def main():
    dx = [float(i) for i in range(10)]
    dy = [float(i * 2 + 1) for i in range(10)]

    model = nn.Linear(in_features=1, out_features=1, bias=True)
    criterion = nn.MSELoss()
    optimizer = nn.optim.Adam(model._parameters.values(), learning_rate=0.01)

    for epoch in range(100):
        total_loss = 0.0
        for x_val, y_val in zip(dx, dy):
            optimizer.zero_grad()
            pred = model(Tensor([[x_val]]))
            loss = criterion(pred, Tensor([[y_val]]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d}, loss={total_loss / len(dx):.6f}")

if __name__ == "__main__":
    main()
```

## Core concepts

### Tensor

The `Tensor` class is the foundation of the library. It stores numeric data, tracks computation history, and supports reverse-mode automatic differentiation.

```python
from krnl import Tensor

a = Tensor([[1.0, 2.0]], requires_grad=True)
b = Tensor([[3.0, 4.0]], requires_grad=True)
c = (a * b).sum()
c.backward()
print(a.grad)  # [3.0, 4.0]
```

Key operations: `+`, `-`, `*`, `@`, `**`, `-x`, `abs()`, `relu()`, `sigmoid()`, `softmax()`, `log()`, `clip()`, `sum()`, `mean()`, `.T`, `reshape()`.

### Device abstraction

```python
from krnl import Device, Tensor

cpu_tensor = Tensor([1.0, 2.0])
gpu_tensor = cpu_tensor.to(Device("GPU"))
back_on_cpu = gpu_tensor.to(Device("CPU"))
```

### OpenCL backend

The OpenCL path uses PyOpenCL to move arrays to an accelerator and execute kernels for tensor operations. Supported GPU operations include arithmetic, matrix multiplication, activations (relu, sigmoid, softmax), and reductions.

## Neural network modules

The `nn` module provides building blocks for simple models.

### Layers and activations

| Module | Description |
|---|---|
| `Linear(in, out, bias)` | Dense layer: `y = xW^T + b` |
| `ReLU` | Rectified linear activation |
| `Sigmoid` | Sigmoid activation |
| `Softmax` | Row-wise softmax (numerically stable) |
| `Tanh` | Hyperbolic tangent activation |
| `GELU` | Gaussian error linear unit (sigmoid approx.) |
| `SiLU` | Sigmoid linear unit: `x * sigmoid(x)` |
| `Sequential(*layers)` | Chain layers into a pipeline |
| `Residual(model)` | Skip connection: `x + model(x)` |

```python
from krnl import nn, Tensor

model = nn.Sequential(
    nn.Linear(4, 8, bias=True),
    nn.ReLU(),
    nn.Linear(8, 1, bias=True),
)

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = model(x)
```

### Train / eval mode

```python
model.train()          # training mode (default)
model.eval()           # evaluation mode
print(model.training)  # False
```

## Loss functions

| Loss | Description |
|---|---|
| `MSELoss` | Mean squared error |
| `MAELoss` | Mean absolute error |
| `HuberLoss(delta)` | Smooth L1 loss |
| `BCELoss` | Binary cross-entropy |
| `CrossEntropy` | Multi-class cross-entropy with built-in softmax |

All loss functions support `reduction="mean"` (default), `"sum"`, or `"none"`.

```python
from krnl import nn

criterion = nn.MSELoss()
loss = criterion(predictions, targets)
loss.backward()
```

## Optimizers

| Optimizer | Description |
|---|---|
| `SGD` | Stochastic gradient descent |
| `Adam` | Adaptive moment estimation |
| `Muon` | Momentum with orthogonalized updates |
| `MuonAdam` | Hybrid Muon + Adam |
| `MuonAdamAux` | MuonAdam with auxiliary parameters |

```python
from krnl import nn

optimizer = nn.optim.Adam(model._parameters.values(), learning_rate=0.001)
```

## Dataset and DataLoader

`Dataset` wraps input and label tensors, while `DataLoader` iterates over a dataset in shuffled mini-batches.

```python
import numpy as np
from krnl import Tensor
from krnl.data import Dataset, DataLoader

inputs = Tensor(np.random.randn(100, 4).astype(np.float32))
labels = Tensor(np.random.randn(100, 1).astype(np.float32))

dataset = Dataset(inputs, labels)
loader  = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

for batch_inputs, batch_labels, batch_idx in loader:
    print(batch_inputs.shape, batch_labels.shape)
```

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `32` | Samples per batch |
| `shuffle` | `False` | Shuffle indices each pass |
| `drop_last` | `False` | Drop the last incomplete batch |

## Trainer

The `Trainer` class provides a high-level training loop that handles forward passes, loss computation, backpropagation, and optimizer steps automatically.

```python
from krnl import nn
from krnl.data import Dataset, DataLoader
from krnl.train import Trainer, TrainerArguments

model = nn.Sequential(
    nn.Linear(4, 8, bias=True),
    nn.ReLU(),
    nn.Linear(8, 1, bias=True),
)

dataset = Dataset(inputs, labels)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

args = TrainerArguments(
    learning_rate=0.01,
    num_epochs=50,
    optimizer="adam",
    loss_function="mse",
)

trainer = Trainer(model, loader, args)
trainer.train(log_epochs=10)
```

`TrainerArguments` accepts:

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `3e-4` | Optimizer step size |
| `batch_size` | `32` | Batch size (informational) |
| `num_epochs` | `10` | Number of training epochs |
| `optimizer` | `"adam"` | `"adam"`, `"sgd"`, `"muon"`, or `"muon_adam"` |
| `loss_function` | `"mse"` | `"mse"`, `"mae"`, `"huber"`, or a `BaseCriterion` instance |

## Saving and loading

```python
from krnl import save, load

# Save model state
save(model.get_model_state_dict(), "model.krnl")

# Load into a fresh model
state = load("model.krnl")
new_model.register_state_dict(state)
```

## Repository structure

```text
krnl/
├── src/krnl/
│   ├── __init__.py
│   ├── backends/
│   │   ├── tensor.py          # Tensor class and autograd
│   │   ├── device.py          # CPU/GPU device abstraction
│   │   ├── GPU_processer.py   # OpenCL backend and GPUArray
│   │   ├── kernels.py         # OpenCL kernel source code
│   │   ├── save.py            # Model serialization
│   │   └── load.py            # Model deserialization
│   ├── data/
│   │   ├── dataset.py         # Dataset class
│   │   └── dataloader.py      # DataLoader for mini-batch iteration
│   ├── nn/
│   │   ├── model_utils.py     # Module, Linear, activations
│   │   ├── loss.py            # Loss functions
│   │   └── optim.py           # Optimizers
│   └── train/
│       ├── trainer.py         # High-level training loop
│       └── args.py            # Training arguments / hyperparameters
├── tests/
├── docs/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Running tests

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python tests/test_tensor.py
python tests/test_loss_functions.py
```

The OpenCL tests require a valid OpenCL installation and compatible device.

## Project status

`krnl` is an educational framework for understanding the internals of deep learning systems. It is not intended to replace established libraries such as PyTorch or TensorFlow.

## Contributing

Contributions are welcome. The project remains intentionally simple and readable so it can serve as a learning tool.

## License

This project is licensed under the [MIT License](LICENSE).
