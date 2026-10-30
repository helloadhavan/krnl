# Optimizers

krnl includes several gradient-based optimizers for training neural networks. All optimizers share a common base class and interface.

**Source:** [`src/krnl/nn/optim.py`](../src/krnl/nn/optim.py)

## Common Interface

Every optimizer inherits from `Optimizer` and provides:

| Method | Description |
|---|---|
| `zero_grad()` | Set all parameter gradients to `None` |
| `step()` | Update parameters using their current gradients |
| `get_state_dict()` | Return a serializable dict of parameters and optimizer state |
| `register_state_dict(dict)` | Restore parameters and state from a dict |

### Base Constructor

```python
Optimizer(parameters, defaults=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parameters` | iterable of `Parameter` | — | Trainable parameters to optimize |
| `defaults` | `dict` or `None` | `None` | Default hyperparameters stored in `self.state` |

### State Dict Format

```python
{
    "parameters": [Parameter(...), Parameter(...)],
    "state": {"lr": 0.01, ...}
}
```

## SGD

Stochastic Gradient Descent — the simplest optimizer. Updates each parameter by subtracting the gradient scaled by the learning rate.

```
parameter = parameter - lr * gradient
```

```python
optimizer = nn.optim.SGD(model._parameters.values(), learning_rate=0.01)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parameters` | iterable | — | Trainable parameters |
| `learning_rate` | `float` | `0.01` | Step size |

## Adam

Adaptive Moment Estimation — combines momentum (first moment) and adaptive learning rates (second moment) for more stable training.

```python
optimizer = nn.optim.Adam(
    model._parameters.values(),
    learning_rate=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-8,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parameters` | iterable | — | Trainable parameters |
| `learning_rate` | `float` | `0.001` | Step size |
| `betas` | `tuple(float, float)` | `(0.9, 0.999)` | Exponential decay rates for moment estimates |
| `epsilon` | `float` | `1e-8` | Small constant for numerical stability |

### How it works

1. Maintains running estimates of the first moment (mean) `m` and second moment (uncentered variance) `v` of the gradient.
2. Applies bias correction to both moments.
3. Updates: `parameter -= lr * m_hat / (sqrt(v_hat) + epsilon)`

## Muon

A momentum-based optimizer with orthogonalized update directions. The update is computed by applying SVD-based orthogonalization to the momentum buffer, producing a more structure-aware step.

```python
optimizer = nn.optim.Muon(
    model._parameters.values(),
    learning_rate=0.02,
    momentum=0.9,
    weight_decay=0.0,
    eps=1e-8,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parameters` | iterable | — | Trainable parameters |
| `learning_rate` | `float` | `0.02` | Step size |
| `momentum` | `float` | `0.9` | Momentum coefficient |
| `weight_decay` | `float` | `0.0` | L2 regularization strength |
| `eps` | `float` | `1e-8` | Small constant for numerical stability |

### How it works

1. Accumulates gradient into a momentum buffer.
2. Orthogonalizes the momentum via SVD (`U @ Vh`).
3. Normalizes the update by its norm.
4. Applies the scaled update to parameters.

## MuonAdam

A hybrid optimizer combining Muon-style orthogonal updates with Adam-like first and second moment tracking. This is an experimental optimizer for research and learning.

```python
optimizer = nn.optim.MuonAdam(
    model._parameters.values(),
    learning_rate=0.02,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=0.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parameters` | iterable | — | Trainable parameters |
| `learning_rate` | `float` | `0.02` | Step size |
| `betas` | `tuple(float, float)` | `(0.9, 0.999)` | Exponential decay rates for moment estimates |
| `epsilon` | `float` | `1e-8` | Small constant for numerical stability |
| `weight_decay` | `float` | `0.0` | L2 regularization strength |

### Aliases

`MuonAdam` is also available under these names for convenience:

- `nn.optim.MuonWithAdamAux`
- `nn.optim.MuonWithAdamAuxiliary`
- `nn.optim.MuonAdamAux`
- `nn.optim.AdamAux`

## Typical Training Loop

```python
from krnl import nn, Tensor

model = nn.Linear(in_features=4, out_features=1, bias=True)
criterion = nn.loss.MSELoss()
optimizer = nn.optim.Adam(model._parameters.values(), learning_rate=0.001)

for epoch in range(100):
    optimizer.zero_grad()

    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    target = Tensor([[10.0]])

    prediction = model(x)
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()
```
