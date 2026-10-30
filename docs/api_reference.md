# API Reference

Complete listing of the public API exported by the `krnl` package.

## Top-Level Imports

```python
import krnl
```

| Name | Type | Description |
|---|---|---|
| `krnl.Tensor` | class | Core tensor with autograd support |
| `krnl.Device` | class | CPU/GPU device abstraction |
| `krnl.save` | function | Serialize state dicts and tensors to disk |
| `krnl.load` | function | Deserialize state dicts and tensors from disk |
| `krnl.rand` | function | Create a random tensor with a given shape |
| `krnl.ones_like` | function | Create a ones tensor matching another tensor's shape |
| `krnl.zeros_like` | function | Create a zeros tensor matching another tensor's shape |
| `krnl.full_like` | function | Create a filled tensor matching another tensor's shape |
| `krnl.rand_like` | function | Create a random tensor matching another tensor's shape |
| `krnl.OpenCLBackend` | class | Low-level OpenCL backend (advanced) |
| `krnl.get_gpu_info` | function | List available OpenCL GPUs |
| `krnl.nn` | module | Neural network building blocks |

## krnl.nn

```python
from krnl import nn
```

### Modules and Layers

| Name | Source | Description |
|---|---|---|
| `nn.Module` | [`model_utils.py`](../src/krnl/nn/model_utils.py) | Base class for neural network components |
| `nn.Parameter` | [`model_utils.py`](../src/krnl/nn/model_utils.py) | Trainable tensor (`requires_grad=True`) |
| `nn.Linear` | [`model_utils.py`](../src/krnl/nn/model_utils.py) | Fully connected layer: `y = x @ W + b` |
| `nn.ReLU` | [`model_utils.py`](../src/krnl/nn/model_utils.py) | Rectified Linear Unit activation |
| `nn.Sigmoid` | [`model_utils.py`](../src/krnl/nn/model_utils.py) | Sigmoid activation |

### Loss Functions

| Name | Source | Description |
|---|---|---|
| `nn.MSELoss` | [`loss.py`](../src/krnl/nn/loss.py) | Mean Squared Error |
| `nn.MAELoss` | [`loss.py`](../src/krnl/nn/loss.py) | Mean Absolute Error |
| `nn.HuberLoss` | [`loss.py`](../src/krnl/nn/loss.py) | Huber loss (smooth L1) |
| `nn.loss.BaseCriterion` | [`loss.py`](../src/krnl/nn/loss.py) | Abstract base for custom losses |

### Optimizers

| Name | Source | Description |
|---|---|---|
| `nn.optim.Optimizer` | [`optim.py`](../src/krnl/nn/optim.py) | Abstract base optimizer |
| `nn.optim.SGD` | [`optim.py`](../src/krnl/nn/optim.py) | Stochastic Gradient Descent |
| `nn.optim.Adam` | [`optim.py`](../src/krnl/nn/optim.py) | Adam optimizer |
| `nn.optim.Muon` | [`optim.py`](../src/krnl/nn/optim.py) | Orthogonalized momentum optimizer |
| `nn.optim.MuonAdam` | [`optim.py`](../src/krnl/nn/optim.py) | Muon + Adam hybrid |
| `nn.optim.MuonWithAdamAux` | [`optim.py`](../src/krnl/nn/optim.py) | Alias for `MuonAdam` |
| `nn.optim.MuonWithAdamAuxiliary` | [`optim.py`](../src/krnl/nn/optim.py) | Alias for `MuonAdam` |
| `nn.optim.MuonAdamAux` | [`optim.py`](../src/krnl/nn/optim.py) | Alias for `MuonAdam` |
| `nn.optim.AdamAux` | [`optim.py`](../src/krnl/nn/optim.py) | Alias for `MuonAdam` |

## Tensor

### Constructor

```python
Tensor(data, device=None, requires_grad=False)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `.data` | `ndarray` / `GPUArray` | Raw underlying data |
| `.grad` | `ndarray` / `GPUArray` / `None` | Accumulated gradient |
| `.shape` | `tuple` | Data shape |
| `.ndim` | `int` | Number of dimensions |
| `.T` | `ndarray` / `GPUArray` | Transposed data |
| `.device` | `Device` | Current device |
| `.requires_grad` | `bool` | Whether gradients are tracked |

### Methods

| Method | Returns | Description |
|---|---|---|
| `.backward()` | `None` | Backpropagate gradients (scalar tensors only) |
| `.relu()` | `Tensor` | Element-wise ReLU |
| `.sigmoid()` | `Tensor` | Element-wise sigmoid |
| `.reshape(shape)` | `Tensor` | Reshape (differentiable) |
| `.to(device)` | `None` | Move tensor to device (in-place) |
| `.tolist()` | `list` | Convert to Python list |
| `.zero_grad()` | `None` | Clear gradient |

### Operators

`+`, `-`, `*`, `@`, `**`, `abs()`, `>`, `<` — all with reverse variants where applicable.

## Device

### Constructor

```python
Device(device, device_id=0)
```

- `device`: `"CPU"` or `"GPU"`
- `device_id`: GPU index (default `0`)

### Methods

| Method | Returns | Description |
|---|---|---|
| `.get_device_info()` | `dict` | Hardware information for this device |

## save

```python
save(data: dict | Tensor, path: str) -> None
```

Serialize a state dict or tensor to a pickle file. All tensors are converted to numpy arrays.

## load

```python
load(path: str, device: Device | None = None) -> dict | Parameter
```

Deserialize a pickle file and reconstruct `Parameter` objects from numpy arrays. Defaults to CPU if no device is specified.

## Factory Functions

| Function | Signature | Description |
|---|---|---|
| `rand(shape, device=None)` | `-> Tensor` | Random uniform `[0, 1)` |
| `ones_like(tensor)` | `-> Tensor` | Ones with same shape/device |
| `zeros_like(tensor)` | `-> Tensor` | Zeros with same shape/device |
| `full_like(tensor, fill_value)` | `-> Tensor` | Filled with same shape/device |
| `rand_like(tensor)` | `-> Tensor` | Random with same shape/device |
