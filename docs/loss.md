# Loss Functions

krnl provides a set of loss functions for regression and classification tasks. All loss functions are callable objects that take `(predicted, target)` tensors and return a differentiable `Tensor`.

**Source:** [`src/krnl/nn/loss.py`](../src/krnl/nn/loss.py)

## Usage

```python
from krnl import nn, Tensor

criterion = nn.loss.MSELoss()

predicted = Tensor([[2.5]])
target = Tensor([[3.0]])

loss = criterion(predicted, target)
loss.backward()
```

## Reduction Modes

All loss functions accept a `reduction` parameter that controls how the element-wise loss is aggregated:

| Mode | Description |
|---|---|
| `"mean"` | Return the mean of all element-wise losses (default) |
| `"sum"` | Return the sum of all element-wise losses |
| `"none"` | Return the unreduced element-wise loss tensor |

```python
# Mean reduction (default)
criterion = nn.MSELoss(reduction="mean")

# Sum reduction
criterion = nn.MSELoss(reduction="sum")

# No reduction — returns per-element loss
criterion = nn.MSELoss(reduction="none")
```

## MSELoss

Mean Squared Error — computes the squared difference between prediction and target.

```
loss = (predicted - target) ** 2
```

```python
criterion = nn.MSELoss()
loss = criterion(predicted, target)
```

### Constructor

```python
MSELoss(reduction="mean")
```

Commonly used for regression tasks. The gradient pushes predictions toward the target, with larger errors producing stronger gradients.

## MAELoss

Mean Absolute Error — computes the absolute difference between prediction and target.

```
loss = |predicted - target|
```

```python
criterion = nn.MAELoss()
loss = criterion(predicted, target)
```

### Constructor

```python
MAELoss(reduction="mean")
```

Useful when you want the average magnitude of prediction error without squaring. Less sensitive to outliers than MSE.

## HuberLoss

A smooth loss that behaves like squared error for small residuals. Combines the benefits of MSE (smooth gradients near zero) with robustness to outliers.

```python
criterion = nn.HuberLoss(delta=1.0)
loss = criterion(predicted, target)
```

### Constructor

```python
HuberLoss(delta=1.0, reduction="mean")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `delta` | `float` | `1.0` | Threshold parameter (currently uses quadratic form) |
| `reduction` | `str` | `"mean"` | Reduction mode: `"mean"`, `"sum"`, or `"none"` |

The current implementation uses the quadratic (MSE-like) form `0.5 * error²`, which is equivalent to Huber loss when all errors are within the delta threshold.

## BCELoss

Binary Cross-Entropy Loss — measures the cross-entropy between predicted probabilities and binary targets.

```
loss = -(target * log(predicted) + (1 - target) * log(1 - predicted))
```

```python
criterion = nn.BCELoss()

predicted = Tensor([0.7, 0.3, 0.9])  # probabilities in [0, 1]
target = Tensor([1.0, 0.0, 1.0])     # binary labels

loss = criterion(predicted, target)
loss.backward()
```

### Constructor

```python
BCELoss(reduction="mean", epsilon=1e-15)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reduction` | `str` | `"mean"` | Reduction mode: `"mean"`, `"sum"`, or `"none"` |
| `epsilon` | `float` | `1e-15` | Small value for numerical stability (clamps predictions) |

### Requirements

- `predicted` values must be in the range `[0, 1]` (e.g., output of a sigmoid layer)
- `predicted` and `target` must have the same shape
- Predictions are internally clamped to `[epsilon, 1 - epsilon]` to avoid `log(0)`

### Typical Usage

BCELoss is commonly paired with a `Sigmoid` activation for binary classification:

```python
from krnl import nn, Tensor

model = nn.Sequential(
    nn.Linear(4, 8, bias=True),
    nn.ReLU(),
    nn.Linear(8, 1, bias=True),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = nn.optim.Adam(model._parameters.values())

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
target = Tensor([[1.0]])

prediction = model(x)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

## BaseCriterion

All loss functions inherit from `BaseCriterion`. To create a custom loss:

```python
from krnl.nn.loss import BaseCriterion
from krnl import Tensor

class MyLoss(BaseCriterion):
    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        return self._reduce((predicted - target) ** 4)  # quartic loss
```

The `_reduce` method applies the configured reduction mode automatically.

## Shape Requirements

All built-in loss functions require `predicted` and `target` to have the same shape. A `ValueError` is raised if shapes do not match.
