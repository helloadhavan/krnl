# Tensor

The `Tensor` class is the foundation of krnl. It stores numeric data, tracks computation history, and supports automatic differentiation through a dynamic computational graph.

**Source:** [`src/krnl/backends/tensor.py`](../src/krnl/backends/tensor.py)

## Creating Tensors

```python
from krnl import Tensor, Device

# From a Python list (defaults to CPU, float32)
t = Tensor([1.0, 2.0, 3.0])

# 2-D tensor
t = Tensor([[1.0, 2.0], [3.0, 4.0]])

# With gradient tracking
t = Tensor([1.0, 2.0], requires_grad=True)

# On a specific device
t = Tensor([1.0, 2.0], device=Device("GPU"))
```

## Constructor

```python
Tensor(data, device=None, requires_grad=False, _children=(), _op="")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like, `Tensor`, or `GPUArray` | — | The numeric data |
| `device` | `Device` or `None` | `Device("CPU")` | Target device |
| `requires_grad` | `bool` | `False` | Whether to track gradients |
| `_children` | tuple | `()` | Parent tensors in the computation graph (internal) |
| `_op` | `str` | `""` | Operation label (internal) |

## Properties

| Property | Returns | Description |
|---|---|---|
| `shape` | `tuple` | Shape of the underlying data |
| `ndim` | `int` | Number of dimensions |
| `T` | ndarray / GPUArray | Transposed data |
| `data` | ndarray / GPUArray | Raw underlying data |
| `grad` | ndarray / GPUArray / `None` | Accumulated gradient |

## Arithmetic Operations

All arithmetic operations return a new `Tensor` and build the computation graph for autograd.

| Operation | Syntax | Notes |
|---|---|---|
| Addition | `a + b` | Supports broadcasting |
| Subtraction | `a - b` | Supports broadcasting |
| Multiplication | `a * b` | Element-wise, supports broadcasting |
| Matrix multiply | `a @ b` | Requires 2-D tensors |
| Power | `a ** n` | `n` must be int or float |
| Absolute value | `abs(a)` | Element-wise |
| ReLU | `a.relu()` | `max(0, x)` element-wise |
| Sigmoid | `a.sigmoid()` | `1 / (1 + exp(-x))` element-wise |

All operations support reverse variants (`__radd__`, `__rsub__`, `__rmul__`) so scalars can appear on the left side.

## Autograd

krnl uses reverse-mode automatic differentiation. Call `.backward()` on a scalar tensor to compute gradients for all upstream tensors that have `requires_grad=True`.

```python
x = Tensor([[2.0]], requires_grad=True)
y = x ** 2 + x * 3
y.backward()

print(x.grad)  # dy/dx = 2*x + 3 = 7.0
```

### How it works

1. Each operation records its parent tensors (`_children`) and a `_backward` closure.
2. `.backward()` performs a topological sort of the graph and calls each `_backward` in reverse order.
3. Gradients are accumulated via `_accumulate_grad`, which handles the case where a tensor is used in multiple operations.
4. Broadcasted gradients are reduced to match the original tensor shape via `_reduce_broadcasted_gradient`.

### Constraints

- `.backward()` can only be called on a scalar (single-element) tensor.
- Gradient computation is not supported across different devices.

## Device Transfer

```python
t = Tensor([1.0, 2.0])
t.to(Device("GPU"))   # move to GPU in-place
t.to(Device("CPU"))   # move back to CPU in-place
```

## Other Methods

| Method | Description |
|---|---|
| `reshape(shape)` | Returns a reshaped tensor (differentiable) |
| `tolist()` | Converts data to a Python list |
| `zero_grad()` | Sets `self.grad = None` |

## Factory Functions

These are available at the top level of the `krnl` package:

```python
from krnl import rand, ones_like, zeros_like, full_like, rand_like

t = rand((3, 4))                          # random uniform [0, 1)
t = rand((3, 4), device=Device("GPU"))    # random on GPU
t = ones_like(existing_tensor)
t = zeros_like(existing_tensor)
t = full_like(existing_tensor, 42.0)
t = rand_like(existing_tensor)
```
