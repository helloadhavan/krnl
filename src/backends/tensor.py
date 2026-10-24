from __future__ import annotations

from src.backends.GPU_processer import OpenCLBackend, cl_array
from src.backends.device import Device
import numpy as np



class Tensor:
    def __init__(self, data: Tensor | list | np.ndarray | cl_array, device: Device = Device("CPU"), requires_grad=False, _children=(), _op=""):
        if device == "GPU":
            self.backend = OpenCLBackend.instance()

        if isinstance(data, Tensor):
            self.data = data.data
        else:
            if device.device == "CPU":
                self.data = np.array(data, dtype=np.float32)
            elif device.device == "GPU":
                if isinstance(data, cl_array.Array):
                    self.data = data
                else:
                    self.data = self.backend.upload(np.array(data, dtype=np.float32))
            else:
                raise ValueError(f"Invalid device type: {device.device}")


        self.device = device
        self.requires_grad = requires_grad
        self.grad = None

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other.data,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad

                # Sum gradients across broadcast dimensions.
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = out.grad

                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                other.grad = (
                    grad if other.grad is None else other.grad + grad
                )

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data * other.data,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data

                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = out.grad * self.data

                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                other.grad = (
                    grad if other.grad is None else other.grad + grad
                )

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        out = Tensor(
            self.data ** power,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad * power * self.data ** (power - 1)
                self.grad = (
                    grad if self.grad is None else self.grad + grad
                )

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data - other.data,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="-",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad

                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                self.grad = (
                    grad if self.grad is None else self.grad + grad
                )

            if other.requires_grad:
                grad = -out.grad

                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)

                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad = grad.sum(axis=axis, keepdims=True)

                other.grad = (
                    grad if other.grad is None else other.grad + grad
                )

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output or loss"
            )

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)

                for child in tensor._prev:
                    build_topo(child)

                topo.append(tensor)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float32)
        for tensor in reversed(topo):
            tensor._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

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