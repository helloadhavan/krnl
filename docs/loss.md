# Loss Functions

krnl provides a small set of loss functions for regression and learning tasks. All loss functions are callable objects that take `(predicted, target)` tensors and return a differentiable `Tensor`.

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

## MSELoss

Mean Squared Error — computes the squared difference between prediction and target.

```
loss = (predicted - target) ** 2
```

```python
criterion = nn.MSELoss()
loss = criterion(predicted, target)
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

Useful when you want the average magnitude of prediction error without squaring. Less sensitive to outliers than MSE.

## HuberLoss

A smooth piecewise loss that behaves like squared error for small residuals and absolute error for larger ones.

```python
criterion = nn.loss.HuberLoss()
loss = criterion(predicted, target, mae_toggle_error=1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `predicted` | `Tensor` | — | Model predictions |
| `target` | `Tensor` | — | Ground truth values |
| `mae_toggle_error` | `float` | `1` | Threshold for switching between MSE and MAE behavior |

When `|error| > mae_toggle_error`, the loss uses absolute error. Otherwise it uses squared error. This reduces sensitivity to outliers while maintaining smooth gradients near zero.

## BaseCriterion

All loss functions inherit from `BaseCriterion`. To create a custom loss:

```python
from krnl.nn.loss import BaseCriterion
from krnl import Tensor

class MyLoss(BaseCriterion):
    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        return (predicted - target) ** 4  # quartic loss
```

## Shape Requirements

All built-in loss functions require `predicted` and `target` to have the same shape. A `ValueError` is raised if shapes do not match.
