# krnl 0.7.1 [Latest] Documentation 

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/krnl?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/krnl)

`krnl` is a lightweight, educational machine learning library written in Python on top of NumPy, with experimental OpenCL support for device-aware tensor computation.

## Highlights

- Tensor-based autograd engine
- NumPy-backed CPU execution
- Experimental OpenCL backend for GPU-style computation
- Basic neural network building blocks
- Optimizer implementations for training
- Common loss functions for regression and learning tasks
- Dataset and DataLoader for mini-batch training
- High-level Trainer API with configurable training arguments

## Table of Contents

- [Getting Started](getting_started.md) — installation, quick start, and first training loop
- [Tensor](tensor.md) — the core data structure and autograd engine
- [Device](device.md) — CPU and GPU device abstraction
- [Neural Network Modules](nn.md) — layers, activations, and the `Module` base class
- [Loss Functions](loss.md) — MSE, MAE, Huber, BCE and cross-entropy loss
- [Optimizers](optimizers.md) — SGD, Adam, Muon, and MuonAdam
- [Dataset and DataLoader](data.md) — mini-batch data loading and iteration
- [Training](training.md) — Trainer, TrainerArguments, and the high-level training loop
- [Saving and Loading](saving_loading.md) — persisting and restoring model and optimizer state
- [OpenCL Backend](opencl_backend.md) — GPU computation via PyOpenCL
- [API Reference](api_reference.md) — complete public API listing

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
