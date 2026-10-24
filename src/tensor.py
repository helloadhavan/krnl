from src.backends.GPU_processer import OpenCLBackend
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other.data,
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
