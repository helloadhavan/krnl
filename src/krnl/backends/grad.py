"""
Autograd engine for krnl Tensors.

``GradEngine`` provides the topological-sort based backward pass.
``AutoGrad`` contains every per-operation backward function used by
:class:`~krnl.backends.tensor.Tensor`.

Each static method in ``AutoGrad`` accepts the forward-pass context
(operands, output tensor, etc.) and returns a closure that, when called,
propagates ``out.grad`` back to the operands.
"""

from __future__ import annotations

import numpy as np


class GradEngine:
    """Topological-sort backward pass over a Tensor computation graph."""

    @staticmethod
    def backward(root_tensor):
        """
        Run reverse-mode autodiff starting from *root_tensor*.

        Parameters
        ----------
        root_tensor : Tensor
            Must be a **scalar** (single-element) tensor.
        """
        from .GPU_processer import GPUArray

        if root_tensor.data.size != 1:
            raise ValueError("backward() requires a scalar output or loss")

        # --- build topological order ---
        topo: list = []
        visited: set = set()

        def _build(tensor):
            if tensor in visited:
                return
            visited.add(tensor)
            for parent in tensor._prev:
                _build(parent)
            topo.append(tensor)

        _build(root_tensor)

        # --- seed gradient ---
        if root_tensor.device.device == "CPU":
            root_tensor.grad = np.ones_like(root_tensor.data, dtype=np.float32)
        else:
            root_tensor.grad = GPUArray.ones(
                root_tensor.data.shape,
                backend=root_tensor.backend,
            )

        # --- propagate ---
        for tensor in reversed(topo):
            tensor._backward()


class AutoGrad:
    """
    Collection of backward-closure factories, one per Tensor operation.

    Every method is a ``@staticmethod`` that returns a *closure*.  The
    closure is stored as ``out._backward`` by the corresponding forward
    method in :class:`Tensor`.
    """

    # ------------------------------------------------------------------
    # helpers (shared by several backward functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_grad(tensor, grad):
        """Add *grad* into ``tensor.grad``, creating it if necessary."""
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad = tensor.grad + grad

    @staticmethod
    def _reduce_broadcasted_gradient(grad, target_shape, backend=None):
        """
        Reduce *grad* so that its shape matches *target_shape*.

        Handles the case where the forward pass broadcast a smaller
        operand to a larger shape.
        """
        from .GPU_processer import GPUArray

        if isinstance(grad, GPUArray):
            if grad.shape == target_shape:
                return grad
            host = np.asarray(grad.get(), dtype=np.float32)
            while host.ndim > len(target_shape):
                host = host.sum(axis=0)
            for axis, size in enumerate(target_shape):
                if size == 1:
                    host = host.sum(axis=axis, keepdims=True)
            return GPUArray(host, backend=grad.backend if backend is None else backend)

        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    # ------------------------------------------------------------------
    # element-wise arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def add_backward(self_tensor, other, out):
        """Backward for ``self_tensor + other``."""
        acc = AutoGrad._accumulate_grad
        red = AutoGrad._reduce_broadcasted_gradient

        def _backward():
            if self_tensor.requires_grad:
                grad = red(out.grad, self_tensor.data.shape)
                acc(self_tensor, grad)
            if other.requires_grad:
                grad = red(out.grad, other.data.shape)
                acc(other, grad)

        return _backward

    @staticmethod
    def sub_backward(self_tensor, other, out):
        """Backward for ``self_tensor - other``."""
        acc = AutoGrad._accumulate_grad
        red = AutoGrad._reduce_broadcasted_gradient

        def _backward():
            if self_tensor.requires_grad:
                grad = red(out.grad, self_tensor.data.shape)
                acc(self_tensor, grad)
            if other.requires_grad:
                grad = -out.grad
                grad = red(grad, other.data.shape)
                acc(other, grad)

        return _backward

    @staticmethod
    def mul_backward(self_tensor, other, out):
        """Backward for ``self_tensor * other``."""
        acc = AutoGrad._accumulate_grad
        red = AutoGrad._reduce_broadcasted_gradient

        def _backward():
            if self_tensor.requires_grad:
                grad = out.grad * other.data
                grad = red(grad, self_tensor.data.shape)
                acc(self_tensor, grad)
            if other.requires_grad:
                grad = out.grad * self_tensor.data
                grad = red(grad, other.data.shape)
                acc(other, grad)

        return _backward

    @staticmethod
    def div_backward(self_tensor, other, out):
        """Backward for ``self_tensor / other``."""
        acc = AutoGrad._accumulate_grad
        red = AutoGrad._reduce_broadcasted_gradient

        def _backward():
            if self_tensor.requires_grad:
                # d(a/b)/da = 1/b
                if self_tensor.device.device == "GPU":
                    grad = self_tensor.backend.div(out.grad, other.data)
                else:
                    grad = out.grad / other.data
                grad = red(grad, self_tensor.data.shape)
                acc(self_tensor, grad)

            if other.requires_grad:
                # d(a/b)/db = -a / b^2
                if self_tensor.device.device == "GPU":
                    neg_self = self_tensor.backend.neg(self_tensor.data)
                    b_sq = self_tensor.backend.mul(other.data, other.data)
                    grad = self_tensor.backend.mul(
                        out.grad,
                        self_tensor.backend.div(neg_self, b_sq),
                    )
                else:
                    grad = out.grad * (-self_tensor.data / (other.data ** 2))
                grad = red(grad, other.data.shape)
                acc(other, grad)

        return _backward

    @staticmethod
    def neg_backward(self_tensor, out):
        """Backward for ``-self_tensor``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            if self_tensor.device.device == "GPU":
                grad = self_tensor.backend.neg(out.grad)
            else:
                grad = -out.grad
            acc(self_tensor, grad)

        return _backward

    @staticmethod
    def abs_backward(self_tensor, out):
        """Backward for ``abs(self_tensor)``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            sign = np.sign(self_tensor.data).astype(np.float32)
            grad = out.grad * sign
            acc(self_tensor, grad)

        return _backward

    @staticmethod
    def pow_backward(self_tensor, power, out):
        """Backward for ``self_tensor ** power``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if self_tensor.requires_grad:
                grad = out.grad * power * self_tensor.data ** (power - 1)
                acc(self_tensor, grad)

        return _backward

    # ------------------------------------------------------------------
    # matrix operations
    # ------------------------------------------------------------------

    @staticmethod
    def matmul_backward(self_tensor, other, out):
        """Backward for ``self_tensor @ other``."""
        from .GPU_processer import GPUArray

        acc = AutoGrad._accumulate_grad
        red = AutoGrad._reduce_broadcasted_gradient

        def _backward():
            if self_tensor.device.device == "GPU":
                other_T = GPUArray(
                    self_tensor.backend.transpose(other.data),
                    backend=self_tensor.backend,
                )
                self_T = GPUArray(
                    self_tensor.backend.transpose(self_tensor.data),
                    backend=self_tensor.backend,
                )
            else:
                other_T = other.data.T
                self_T = self_tensor.data.T

            if self_tensor.requires_grad:
                if self_tensor.device.device == "GPU":
                    grad = GPUArray(
                        self_tensor.backend.matmul(out.grad, other_T),
                        backend=self_tensor.backend,
                    )
                else:
                    grad = out.grad @ other_T
                grad = red(grad, self_tensor.data.shape)
                acc(self_tensor, grad)

            if other.requires_grad:
                if self_tensor.device.device == "GPU":
                    grad = GPUArray(
                        self_tensor.backend.matmul(self_T, out.grad),
                        backend=self_tensor.backend,
                    )
                else:
                    grad = self_T @ out.grad
                grad = red(grad, other.data.shape)
                acc(other, grad)

        return _backward

    @staticmethod
    def transpose_backward(self_tensor, out):
        """Backward for ``self_tensor.T``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if self_tensor.requires_grad and out.grad is not None:
                if self_tensor.device.device == "GPU":
                    grad = self_tensor.backend.transpose(out.grad)
                else:
                    grad = np.ascontiguousarray(out.grad.T)
                acc(self_tensor, grad)

        return _backward

    # ------------------------------------------------------------------
    # activation functions
    # ------------------------------------------------------------------

    @staticmethod
    def relu_backward(self_tensor, out):
        """Backward for ``self_tensor.relu()``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            mask = (self_tensor.data > 0).astype(np.float32)
            grad = out.grad * mask
            acc(self_tensor, grad)

        return _backward

    @staticmethod
    def sigmoid_backward(self_tensor, out):
        """Backward for ``self_tensor.sigmoid()``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            grad = out.grad * out.data * (1.0 - out.data)
            acc(self_tensor, grad)

        return _backward

    @staticmethod
    def softmax_backward(self_tensor, out):
        """
        Backward for ``self_tensor.softmax()``.

        Uses the Jacobian-vector product:
        ``ds/dx = s * (dL/ds - dot(dL/ds, s))``
        """
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            s = out.data
            dot = (out.grad * s).sum(axis=1, keepdims=True)
            grad = s * (out.grad - dot)
            acc(self_tensor, grad.astype(np.float32))

        return _backward

    # ------------------------------------------------------------------
    # reduction / shape operations
    # ------------------------------------------------------------------

    @staticmethod
    def sum_backward(self_tensor, out):
        """Backward for ``self_tensor.sum()``."""
        from .GPU_processer import GPUArray

        acc = AutoGrad._accumulate_grad

        def _backward():
            if self_tensor.requires_grad and out.grad is not None:
                if self_tensor.device.device == "GPU":
                    grad = GPUArray(
                        np.full(
                            self_tensor.data.shape,
                            out.grad.get().item(),
                            dtype=np.float32,
                        ),
                        backend=self_tensor.backend,
                    )
                else:
                    grad = np.full(
                        self_tensor.data.shape,
                        out.grad.item(),
                        dtype=np.float32,
                    )
                acc(self_tensor, grad)

        return _backward

    @staticmethod
    def reshape_backward(self_tensor, out):
        """Backward for ``self_tensor.reshape(shape)``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if self_tensor.requires_grad and out.grad is not None:
                grad = out.grad.reshape(self_tensor.data.shape)
                acc(self_tensor, grad)

        return _backward

    @staticmethod
    def log_backward(self_tensor, out):
        """Backward for ``self_tensor.log()``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            if self_tensor.device.device == "GPU":
                grad = self_tensor.backend.div(out.grad, self_tensor.data)
            else:
                grad = out.grad / self_tensor.data
            acc(self_tensor, grad)

        return _backward

    @staticmethod
    def clip_backward(self_tensor, minv, maxv, out):
        """Backward for ``self_tensor.clip(minv, maxv)``."""
        acc = AutoGrad._accumulate_grad

        def _backward():
            if not self_tensor.requires_grad:
                return
            mask = (
                (self_tensor.data >= minv.data) & (self_tensor.data <= maxv.data)
            ).astype(np.float32)
            grad = out.grad * mask
            acc(self_tensor, grad)

        return _backward

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    @staticmethod
    def index_backward(self_tensor, key, out):
        """Backward for ``self_tensor[key]``."""
        from .tensor import Tensor

        acc = AutoGrad._accumulate_grad

        def _backward():
            if self_tensor.device.device == "GPU":
                grad_np = np.asarray(
                    out.grad.data.get()
                    if hasattr(out.grad.data, "get")
                    else out.grad.data,
                    dtype=np.float32,
                )
                self_grad = np.zeros(self_tensor.data.shape, dtype=np.float32)
                if isinstance(key, (int, np.integer)):
                    self_grad[key] += grad_np.reshape(self_grad[key].shape)
                elif isinstance(key, slice):
                    indices = np.arange(*key.indices(self_tensor.data.shape[0]))
                    for i, idx in enumerate(indices):
                        self_grad[idx] += grad_np[i]
                elif isinstance(key, (list, np.ndarray)):
                    np.add.at(self_grad, key, grad_np)
                else:
                    self_grad[key] += grad_np
                grad_tensor = Tensor(self_grad, device=self_tensor.device)
                acc(self_tensor, grad_tensor)
            else:
                grad = np.zeros_like(self_tensor.data)
                if isinstance(key, (int, np.integer)):
                    grad[key] += np.asarray(out.grad.data).reshape(grad[key].shape)
                elif isinstance(key, slice):
                    grad[key] += np.asarray(out.grad.data)
                elif isinstance(key, (list, np.ndarray)):
                    np.add.at(grad, key, np.asarray(out.grad.data))
                else:
                    grad[key] += np.asarray(out.grad.data)
                acc(self_tensor, Tensor(grad, device=self_tensor.device))

        return _backward


__all__ = ["GradEngine", "AutoGrad"]
