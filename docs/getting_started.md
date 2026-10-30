# Getting Started

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

### Optional: OpenCL setup

To use the GPU backend, make sure your system has a compatible OpenCL runtime and drivers installed. The library uses [PyOpenCL](https://documen.tician.de/pyopencl/) under the hood.

## Quick Start

```python
import numpy as np
from krnl import nn, Tensor

# Prepare data: y = 2x + 1
dx = [float(i) for i in range(10)]
dy = [float(i * 2 + 1) for i in range(10)]

# Build a simple linear model
model = nn.Linear(in_features=1, out_features=1, bias=True)

# Choose a loss function and optimizer
criterion = nn.loss.MSELoss()
optimizer = nn.optim.SGD(model._parameters.values(), learning_rate=0.01)

# Training loop
for epoch in range(100):
    total_loss = 0.0

    for x_val, y_val in zip(dx, dy):
        optimizer.zero_grad()

        x = Tensor([[x_val]])
        target = Tensor([[y_val]])

        prediction = model(x)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += float(np.asarray(loss.data).reshape(-1)[0])

    if epoch % 20 == 0:
        print(f"epoch={epoch:03d}, loss={total_loss / len(dx):.6f}")
```

## Training on GPU

To run the same model on an OpenCL GPU, pass a `Device` when creating the model and tensors:

```python
from krnl import nn, Tensor, Device

device = Device("GPU", device_id=0)

model = nn.Linear(in_features=1, out_features=1, bias=True, device=device)
optimizer = nn.optim.Adam(model._parameters.values(), learning_rate=0.001)
criterion = nn.loss.MSELoss()

x = Tensor([[1.0]], device=device)
target = Tensor([[3.0]], device=device)

prediction = model(x)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

## Saving and Loading

```python
from krnl import save, load

# Save
save(model.get_model_state_dict(), "my_model")
save(optimizer.get_state_dict(), "my_optimizer")

# Load
model_state = load("my_model")
optim_state = load("my_optimizer")

# Restore into a fresh model
new_model = nn.Linear(in_features=1, out_features=1, bias=True)
new_model.register_state_dict(model_state)
```
