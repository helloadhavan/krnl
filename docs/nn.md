# Neural Network Modules

The `nn` module provides building blocks for constructing and training simple neural networks.

**Source:** [`src/krnl/nn/model_utils.py`](../src/krnl/nn/model_utils.py)

## Module

`Module` is the base class for all neural network components. It manages named parameters and provides a standard interface for forward passes and gradient clearing.

```python
from krnl import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1, bias=True)

        # Register parameters from sub-layers
        for name, param in self.layer1._parameters.items():
            self.register_parameter(f"layer1_{name}", param)
        for name, param in self.layer2._parameters.items():
            self.register_parameter(f"layer2_{name}", param)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

### Methods

| Method | Description |
|---|---|
| `register_parameter(name, parameter)` | Register a named `Parameter` on the module |
| `forward(x)` | Override this to define the forward pass |
| `zero_grad()` | Set all parameter gradients to `None` |
| `get_model_state_dict()` | Return a dict with `"states"` and `"parameters"` |
| `register_state_dict(dict)` | Restore parameters and states from a dict |
| `__call__(x)` | Calls `forward(x)` |

### State Dict Format

```python
{
    "states": {},                    # user-defined state
    "parameters": {
        "weight": Parameter(...),
        "bias": Parameter(...),
    }
}
```

## Parameter

`Parameter` is a `Tensor` subclass with `requires_grad=True` by default. It marks a tensor as trainable so optimizers can update it.

```python
from krnl.nn import Parameter
from krnl import Tensor

p = Parameter([1.0, 2.0, 3.0])
print(p.requires_grad)  # True
```

### Constructor

```python
Parameter(data, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like or `Tensor` | — | Initial parameter data |
| `device` | `Device` or `None` | `Device("CPU")` | Target device |

## Linear

A fully connected (dense) layer that computes `y = x @ W + b`.

```python
from krnl import nn, Tensor, Device

layer = nn.Linear(in_features=4, out_features=8, bias=True, device=Device("CPU"))
x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = layer(x)
print(y.shape)  # (1, 8)
```

### Constructor

```python
Linear(in_features, out_features, bias=False, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | — | Size of the input dimension |
| `out_features` | `int` | — | Size of the output dimension |
| `bias` | `bool` | `False` | Whether to include a bias term |
| `device` | `Device` or `None` | `Device("CPU")` | Target device for parameters |

### Registered Parameters

- `"weight"` — shape `(out_features, in_features)`, randomly initialized
- `"bias"` — shape `(out_features,)`, randomly initialized (only if `bias=True`)

## ReLU

Elementwise Rectified Linear Unit activation: `max(0, x)`.

```python
relu = nn.ReLU()
y = relu(x)
```

Supports backpropagation — the gradient is `1` where `x > 0` and `0` elsewhere.

## Sigmoid

Elementwise sigmoid activation: `1 / (1 + exp(-x))`.

```python
sigmoid = nn.Sigmoid()
y = sigmoid(x)
```

Supports backpropagation with the derivative `sigmoid(x) * (1 - sigmoid(x))`.

## Tanh

Elementwise hyperbolic tangent activation: `tanh(x) = 2 * sigmoid(2x) - 1`.

Outputs are in the range `(-1, 1)`. Tanh is an odd function, meaning `tanh(-x) = -tanh(x)`.

```python
tanh = nn.Tanh()
y = tanh(x)
```

Supports backpropagation with the derivative `1 - tanh(x)²`.

## GELU

Gaussian Error Linear Unit activation using the sigmoid approximation: `GELU(x) ≈ x * sigmoid(1.702 * x)`.

GELU is a smooth, non-monotonic activation used in Transformers (BERT, GPT). Unlike ReLU, it allows small negative values to pass through, which can improve training dynamics.

```python
gelu = nn.GELU()
y = gelu(x)
```

Supports backpropagation through the computation graph automatically.

## Sequential

A container that chains multiple layers and executes them in order. It automatically collects parameters from all child `Module` instances, making it easy to pass them to an optimizer.

```python
from krnl import nn

model = nn.Sequential(
    nn.Linear(4, 8, bias=True),
    nn.ReLU(),
    nn.Linear(8, 1, bias=True),
    nn.Sigmoid(),
)

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = model(x)

# All parameters are collected automatically
optimizer = nn.optim.Adam(model._parameters.values())
```

### Constructor

```python
Sequential(*layers)
```

| Parameter | Type | Description |
|---|---|---|
| `*layers` | `Module` | Any number of `Module` instances to chain in order |

### Parameter Collection

`Sequential` iterates over its child layers at construction time. For each child that is a `Module`, it registers all of that child's parameters under the name `layer_{i}_{name}` (where `i` is the layer index and `name` is the parameter name within that layer).

```python
seq = nn.Sequential(
    nn.Linear(2, 4, bias=True),   # layer_0_weight, layer_0_bias
    nn.Tanh(),                     # no parameters
    nn.Linear(4, 1, bias=True),   # layer_2_weight, layer_2_bias
)

print(list(seq._parameters.keys()))
# ['layer_0_weight', 'layer_0_bias', 'layer_2_weight', 'layer_2_bias']
```
