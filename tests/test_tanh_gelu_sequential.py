"""
Tests for Tanh, GELU, and Sequential layers.
"""

import numpy as np

from src.krnl.backends.tensor import Tensor
from src.krnl.nn.model_utils import Tanh, GELU, Sequential, Linear, ReLU, Sigmoid


# ── Tanh ─────────────────────────────────────────────────────────────

def test_tanh():
    print("Tanh Tests")

    tanh = Tanh()

    # Forward: compare against np.tanh
    x = Tensor(np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    out = tanh(x)
    expected = np.tanh(x.data)
    print(f"  input:    {x.data}")
    print(f"  output:   {out.data}")
    print(f"  expected: {expected}")
    print(f"  max diff: {np.max(np.abs(out.data - expected)):.6f}")

    # Backward: gradient should flow
    x = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    out = tanh(x)
    loss = out ** 2
    loss.backward()
    print(f"  grad at x=1.0: {x.grad}")
    print()


# ── GELU ─────────────────────────────────────────────────────────────

def test_gelu():
    print("GELU Tests")

    gelu = GELU()

    # Forward: compare against sigmoid approximation reference
    vals = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    x = Tensor(vals)
    out = gelu(x)
    expected = vals * (1.0 / (1.0 + np.exp(-1.702 * vals)))
    print(f"  input:    {x.data}")
    print(f"  output:   {out.data}")
    print(f"  expected: {expected}")
    print(f"  max diff: {np.max(np.abs(out.data - expected)):.6f}")

    # Backward: gradient should flow
    x = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    out = gelu(x)
    loss = out ** 2
    loss.backward()
    print(f"  grad at x=1.0: {x.grad}")
    print()


# ── Sequential ───────────────────────────────────────────────────────

def test_sequential():
    print("Sequential Tests")

    # Chain activations: ReLU -> Sigmoid
    seq = Sequential(ReLU(), Sigmoid())
    x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
    out = seq(x)
    relu_out = np.maximum(x.data, 0.0)
    expected = 1.0 / (1.0 + np.exp(-relu_out))
    print(f"  ReLU -> Sigmoid")
    print(f"    input:    {x.data}")
    print(f"    output:   {out.data}")
    print(f"    expected: {expected}")

    # Sequential with Linear: check parameter collection
    l1 = Linear(in_features=1, out_features=1, bias=True)
    l2 = Linear(in_features=1, out_features=1, bias=True)
    seq = Sequential(l1, Tanh(), l2, Sigmoid())
    print(f"  Parameters collected: {len(seq._parameters)} (expected 4)")

    # Forward through Linear + activations
    x = Tensor(np.array([[3.0]], dtype=np.float32))
    out = seq(x)
    print(f"    input shape:  {x.data.shape}")
    print(f"    output shape: {out.data.shape}")
    print(f"    output value: {out.data}")

    # Backward through Sequential
    x = Tensor(np.array([[2.0]], dtype=np.float32), requires_grad=True)
    linear = Linear(in_features=1, out_features=1, bias=False)
    seq = Sequential(linear, Tanh())
    out = seq(x)
    loss = out ** 2
    loss.backward()
    print(f"  Backward through Sequential:")
    print(f"    weight grad: {linear._parameters['weight'].grad}")
    print(f"    input grad:  {x.grad}")
    print()


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tanh()
    test_gelu()
    test_sequential()
    print("Done.")
