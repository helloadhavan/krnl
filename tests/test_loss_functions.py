import sys
import numpy as np

sys.path.insert(0, "src")

from krnl.backends.tensor import Tensor
from krnl.nn.loss import MSELoss, MAELoss, HuberLoss, BCELoss


def test_mse_loss():
    print("=== MSELoss ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = MSELoss(reduction="mean")
    loss = loss_fn(predicted, target)
    print(f"  predicted: {predicted.data}")
    print(f"  target:    {target.data}")
    print(f"  loss:      {loss.item():.4f}")
    # expected: mean((0.5^2 + 0.5^2 + 0.5^2)) = 0.25
    print(f"  expected:  0.2500")

    loss.backward()
    print(f"  grad:      {predicted.grad}")
    print()


def test_mse_loss_sum():
    print("=== MSELoss (sum) ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = MSELoss(reduction="sum")
    loss = loss_fn(predicted, target)
    print(f"  loss:      {loss.item():.4f}")
    # expected: sum((0.5^2 + 0.5^2 + 0.5^2)) = 0.75
    print(f"  expected:  0.7500")
    print()


def test_mae_loss():
    print("=== MAELoss ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = MAELoss(reduction="mean")
    loss = loss_fn(predicted, target)
    print(f"  predicted: {predicted.data}")
    print(f"  target:    {target.data}")
    print(f"  loss:      {loss.item():.4f}")
    # expected: mean(|0.5| + |0.5| + |0.5|) = 0.5
    print(f"  expected:  0.5000")

    loss.backward()
    print(f"  grad:      {predicted.grad}")
    print()


def test_mae_loss_sum():
    print("=== MAELoss (sum) ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = MAELoss(reduction="sum")
    loss = loss_fn(predicted, target)
    print(f"  loss:      {loss.item():.4f}")
    # expected: sum(|0.5| + |0.5| + |0.5|) = 1.5
    print(f"  expected:  1.5000")
    print()


def test_huber_loss():
    print("=== HuberLoss ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = HuberLoss(delta=1.0, reduction="mean")
    loss = loss_fn(predicted, target)
    print(f"  predicted: {predicted.data}")
    print(f"  target:    {target.data}")
    print(f"  loss:      {loss.item():.4f}")
    # expected: mean(0.5 * 0.25 + 0.5 * 0.25 + 0.5 * 0.25) = 0.125
    print(f"  expected:  0.1250")

    loss.backward()
    print(f"  grad:      {predicted.grad}")
    print()


def test_huber_loss_sum():
    print("=== HuberLoss (sum) ===")
    predicted = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])

    loss_fn = HuberLoss(delta=1.0, reduction="sum")
    loss = loss_fn(predicted, target)
    print(f"  loss:      {loss.item():.4f}")
    # expected: sum(0.5 * 0.25 + 0.5 * 0.25 + 0.5 * 0.25) = 0.375
    print(f"  expected:  0.3750")
    print()


def test_bce_loss():
    print("=== BCELoss ===")
    predicted = Tensor([0.7, 0.3, 0.9], requires_grad=True)
    target = Tensor([1.0, 0.0, 1.0])

    loss_fn = BCELoss(reduction="mean")
    loss = loss_fn(predicted, target)
    print(f"  predicted: {predicted.data}")
    print(f"  target:    {target.data}")
    print(f"  loss:      {loss.item():.4f}")

    # manual: -( 1*log(0.7) + 0*log(0.3) + 1*log(0.9) ) / 3
    #       + -( 0*log(0.3) + 1*log(0.7) + 0*log(0.1) ) / 3
    manual = -(np.log(0.7) + np.log(0.7) + np.log(0.9)) / 3
    print(f"  expected:  {manual:.4f}")

    loss.backward()
    print(f"  grad:      {predicted.grad}")
    print()


def test_bce_loss_sum():
    print("=== BCELoss (sum) ===")
    predicted = Tensor([0.8, 0.2], requires_grad=True)
    target = Tensor([1.0, 0.0])

    loss_fn = BCELoss(reduction="sum")
    loss = loss_fn(predicted, target)
    print(f"  loss:      {loss.item():.4f}")

    manual = -(np.log(0.8) + np.log(0.8))
    print(f"  expected:  {manual:.4f}")
    print()


if __name__ == "__main__":
    test_mse_loss()
    test_mse_loss_sum()
    test_mae_loss()
    test_mae_loss_sum()
    test_huber_loss()
    test_huber_loss_sum()
    test_bce_loss()
    test_bce_loss_sum()
    print("All loss function tests completed.")
