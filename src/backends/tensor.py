from __future__ import annotations

import numpy as np
import pyopencl.array as cl_array

from src.backends.GPU_processer import OpenCLBackend
from src.backends.device import Device


class Tensor:
    def __init__(
        self,
        data,
        device: Device | None = None,
        requires_grad: bool = False,
        _children=(),
        _op: str = "",
    ):
        if device is None:
            device = Device("CPU")

        self.device = device
        self.requires_grad = requires_grad
        self.grad = None

        if self.device.device == "GPU":
            self.backend = OpenCLBackend.instance()

        if isinstance(data, Tensor):
            self.data = data.data

        elif self.device.device == "CPU":
            self.data = np.asarray(data, dtype=np.float32)

        elif self.device.device == "GPU":
            if isinstance(data, cl_array.Array):
                self.data = data
            else:
                data = np.asarray(data, dtype=np.float32)
                self.data = self.backend.upload(data)

        else:
            raise ValueError(
                f"Invalid device type: {self.device.device}"
            )

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __eq__(self, other):
        """
        Tensors are graph nodes, so compare them by identity.
        """
        return self is other

    def __hash__(self):
        """
        Required because Tensor objects are stored in sets.
        """
        return id(self)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _ensure_tensor(self, value):
        """
        Convert numbers and arrays to tensors on this tensor's device.
        """
        if isinstance(value, Tensor):
            if value.device.device != self.device.device:
                raise ValueError(
                    "Both tensors must use the same device"
                )
            return value

        return Tensor(
            value,
            device=self.device,
            requires_grad=False,
        )

    @staticmethod
    def _accumulate_grad(tensor, grad):
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad = tensor.grad + grad

    @staticmethod
    def _reduce_broadcasted_gradient(grad, target_shape):
        """
        Reduce a broadcasted gradient to the original tensor shape.
        """
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(
                    axis=axis,
                    keepdims=True,
                )

        return grad

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------
    # Addition
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data + other.data,
            device=self.device,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if self.requires_grad:
                grad = self._reduce_broadcasted_gradient(
                    out.grad,
                    self.data.shape,
                )
                self._accumulate_grad(self, grad)

            if other.requires_grad:
                grad = self._reduce_broadcasted_gradient(
                    out.grad,
                    other.data.shape,
                )
                self._accumulate_grad(other, grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # ------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------

    def __sub__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data - other.data,
            device=self.device,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="-",
        )

        def _backward():
            if self.requires_grad:
                grad = self._reduce_broadcasted_gradient(
                    out.grad,
                    self.data.shape,
                )
                self._accumulate_grad(self, grad)

            if other.requires_grad:
                grad = -out.grad

                grad = self._reduce_broadcasted_gradient(
                    grad,
                    other.data.shape,
                )
                self._accumulate_grad(other, grad)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other - self

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------

    def __mul__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data * other.data,
            device=self.device,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data

                grad = self._reduce_broadcasted_gradient(
                    grad,
                    self.data.shape,
                )
                self._accumulate_grad(self, grad)

            if other.requires_grad:
                grad = out.grad * self.data

                grad = self._reduce_broadcasted_gradient(
                    grad,
                    other.data.shape,
                )
                self._accumulate_grad(other, grad)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError(
                "power must be an int or float"
            )

        out = Tensor(
            self.data ** power,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}",
        )

        def _backward():
            if self.requires_grad:
                grad = (
                    out.grad
                    * power
                    * self.data ** (power - 1)
                )

                self._accumulate_grad(self, grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """
        Run reverse-mode automatic differentiation.

        The output must be scalar.
        """
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output or loss"
            )

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor in visited:
                return

            visited.add(tensor)

            for parent in tensor._prev:
                build_topo(parent)

            topo.append(tensor)

        build_topo(self)

        # d(loss) / d(loss) = 1
        if self.device.device == "CPU":
            self.grad = np.ones_like(
                self.data,
                dtype=np.float32,
            )
        else:
            self.grad = cl_array.ones_like(
                self.data,
                dtype=np.float32,
            )

        # Reverse topological order applies the chain rule.
        for tensor in reversed(topo):
            tensor._backward()

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward():
            if not self.requires_grad:
                return

            mask = (self.data > 0).astype(np.float32)
            grad = out.grad * mask

            self.grad = (
                grad
                if self.grad is None
                else self.grad + grad
            )

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Comparisons and representation
    # ------------------------------------------------------------------

    def __gt__(self, value):
        if not isinstance(value, Tensor):
            value = self._ensure_tensor(value)

        return np.all(self.data > value.data)

    def __lt__(self, value):
        if not isinstance(value, Tensor):
            value = self._ensure_tensor(value)

        return np.all(self.data < value.data)

    def __repr__(self):
        return (
            f"Tensor("
            f"data={self.data}, "
            f"grad={self.grad}, "
            f"requires_grad={self.requires_grad}"
            f")"
        )

def ones_like(tensor: Tensor) -> Tensor:
    """Create a new tensor filled with ones, having the same shape and device as the input tensor."""
    return Tensor(np.ones_like(tensor.data), device=tensor.device)

def zeros_like(tensor: Tensor) -> Tensor:
    """Create a new tensor filled with zeros, having the same shape and device as the input tensor."""
    return Tensor(np.zeros_like(tensor.data), device=tensor.device)

def full_like(tensor: Tensor, fill_value: float) -> Tensor:
    """Create a new tensor filled with a specified value, having the same shape and device as the input tensor."""
    return Tensor(np.full_like(tensor.data, fill_value), device=tensor.device)

def rand_like(tensor: Tensor) -> Tensor:
    """Create a new tensor filled with random values, having the same shape and device as the input tensor."""
    return Tensor(np.random.rand(*tensor.data.shape).astype(np.float32), device=tensor.device)

def rand(shape, device=None) -> Tensor:
    """Create a new tensor filled with random values, having the specified shape and device."""
    device = device if device is not None else Device("CPU")
    return Tensor(np.random.rand(*shape).astype(np.float32), device=device)