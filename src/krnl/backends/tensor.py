from __future__ import annotations

import numpy as np
import pyopencl.array as cl_array

from .GPU_processer import (
    GPUArray,
    OpenCLBackend,
)
from .device import Device
from .grad import AutoGrad, GradEngine


class Tensor:
    def __init__(
        self,
        data,
        device: Device | None = None,
        requires_grad: bool = False,
        _children=(),
        _op: str = "",
        grad_engine: GradEngine | None = None,
    ):
        if device is None:
            device = Device("CPU")

        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_engine = grad_engine if grad_engine is not None else GradEngine()

        if self.device.device == "GPU":
            self.backend = OpenCLBackend.instance(
                device_id=self.device.device_id
            )

            if isinstance(data, Tensor):
                self.data = data.data

            elif isinstance(data, GPUArray):
                self.data = data

            else:
                self.data = GPUArray(
                    data,
                    backend=self.backend,
                )
        elif self.device.device == "CPU":
            self.backend = None
            if isinstance(data, Tensor):
                data = data.data
            self.data = np.asarray(
                data,
                dtype=np.float32,
            )
        else:
            raise TypeError("Non supported device")

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def _ensure_tensor(self, value):
        if isinstance(value, Tensor):
            if value.device.device != self.device.device:
                raise ValueError("Both tensors must use the same device")
            return value

        return Tensor(value, device=self.device, requires_grad=False, grad_engine=self.grad_engine)

    def zero_grad(self):
        self.grad = None

    def detach(self, requires_grad=False):
        """Return a new Tensor with the same data but disconnected from the computation graph."""
        return Tensor(self.data, device=self.device, requires_grad=requires_grad, grad_engine=self.grad_engine)

    def __add__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.backend.add(self.data, other.data) if self.device.device == "GPU" else self.data + other.data,
            device=self.device,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op="+",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.add_backward(self, other, out)
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.backend.sub(self.data, other.data) if self.device.device == "GPU" else self.data - other.data,
            device=self.device,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op="-",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.sub_backward(self, other, out)
        return out

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other - self

    def __mul__(self, other):
        other = self._ensure_tensor(other)

        out = Tensor(
            self.backend.mul(self.data, other.data) if self.device.device == "GPU" else self.data * other.data,
            device=self.device,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op="*",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.mul_backward(self, other, out)
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._ensure_tensor(other)

        if self.device.device == "GPU":
            result_data = self.backend.div(self.data, other.data)
        else:
            result_data = self.data / other.data

        out = Tensor(
            result_data,
            device=self.device,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op="/",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.div_backward(self, other, out)
        return out

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other / self

    def __matmul__(self, other):
        other = self._ensure_tensor(other)

        if self.device.device == "GPU":
            result_data = self.backend.matmul(self.data, other.data)
        else:
            result_data = self.data @ other.data

        out = Tensor(
            result_data,
            device=self.device,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op="@",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.matmul_backward(self, other, out)
        return out

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("power must be an int or float")

        out = Tensor(
            self.data ** power,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.pow_backward(self, power, out)
        return out

    def backward(self):
        self.grad_engine.backward(self)

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.relu_backward(self, out)
        return out

    def sigmoid(self):
        if self.device.device == "CPU":
            sigmoid_data = (1.0 / (1.0 + np.exp(-self.data))).astype(np.float32)
        elif self.device.device == "GPU":
            sigmoid_data = self.backend.sigmoid(self.data)
        else:
            raise ValueError(f"Invalid device type: {self.device.device}")

        out = Tensor(
            sigmoid_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.sigmoid_backward(self, out)
        return out

    def softmax(self):
        if self.data.ndim != 2:
            raise ValueError("softmax currently supports 2-D tensors only")

        if self.device.device == "GPU":
            softmax_data = self.backend.softmax(self.data)
        else:
            # Numerically stable softmax: subtract max per row
            shifted = self.data - self.data.max(axis=1, keepdims=True)
            exp_data = np.exp(shifted)
            softmax_data = (exp_data / exp_data.sum(axis=1, keepdims=True)).astype(np.float32)

        out = Tensor(
            softmax_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.softmax_backward(self, out)
        return out

    def log(self):
        if self.device.device == "GPU":
            log_data = self.backend.log(self.data)
        else:
            log_data = np.log(self.data).astype(np.float32)

        out = Tensor(
            log_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.log_backward(self, out)
        return out

    def __abs__(self):
        out = Tensor(
            np.abs(self.data),
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="abs",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.abs_backward(self, out)
        return out

    def __neg__(self):
        if self.device.device == "GPU":
            neg_data = self.backend.neg(self.data)
        else:
            neg_data = -self.data

        out = Tensor(
            neg_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.neg_backward(self, out)
        return out

    def __gt__(self, value):
        if not isinstance(value, Tensor):
            value = self._ensure_tensor(value)
        return np.all(self.data > value.data)

    def __lt__(self, value):
        if not isinstance(value, Tensor):
            value = self._ensure_tensor(value)
        return np.all(self.data < value.data)

    def __getitem__(self, key):
        if self.device.device == "GPU":
            indexed_data = self.data[key]
        else:
            indexed_data = self.data[key]

        out = Tensor(
            indexed_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="index",
            grad_engine=self.grad_engine,
        )

        if self.requires_grad:
            out._backward = self.grad_engine.index_backward(self, key, out)

        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        if self.device.device == "GPU":
            transposed = self.backend.transpose(self.data)
        else:
            transposed = np.ascontiguousarray(self.data.T)

        out = Tensor(
            transposed,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="T",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.transpose_backward(self, out)
        return out

    def to(self, device: Device):
        self.device = device
        if device.device == "GPU":
            self.backend = OpenCLBackend.instance(
                device_id=self.device.device_id
            )
            if isinstance(self.data, GPUArray):
                pass
            else:
                self.data = GPUArray(
                    self.data,
                    backend=self.backend,
                )
        elif device.device == "CPU":
            self.backend = None
            if isinstance(self.data, GPUArray):
                self.data = self.data.array

            self.data = np.asarray(
                self.data,
                dtype=np.float32,
            )
        else:
            raise TypeError("Non supported device")

    def reshape(self, shape):
        if self.device.device == "GPU":
            data = self.data.reshape(shape)
        else:
            data = self.data.reshape(shape)

        out = Tensor(
            data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.reshape_backward(self, out)
        return out

    def sum(self):
        if self.device.device == "GPU":
            sum_data = self.data.sum()
        else:
            sum_data = np.array(self.data.sum(), dtype=np.float32)

        out = Tensor(
            sum_data,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.sum_backward(self, out)
        return out

    def clip(self, minv, maxv):
        minv = self._ensure_tensor(minv)
        maxv = self._ensure_tensor(maxv)

        clipped = np.clip(
            self.data if self.device.device == "CPU" else self.data,
            minv.data,
            maxv.data,
        )

        out = Tensor(
            clipped,
            device=self.device,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="clip",
            grad_engine=self.grad_engine,
        )

        out._backward = self.grad_engine.clip_backward(self, minv, maxv, out)
        return out

    def mean(self):
        n = self.data.size
        return self.sum() * (1.0 / n)

    def item(self):
        """Return the value of a single-element tensor as a Python scalar."""
        if self.data.size != 1:
            raise ValueError("item() requires a single-element tensor")
        if self.device.device == "GPU":
            return float(self.data.get().item())
        return float(self.data.item())

    def tolist(self):
        if self.device.device == "CPU":
            return self.data.tolist()
        elif self.device.device == "GPU":
            return self.data.tolist()
        else:
            raise TypeError("Non supported device")

    def _is_gpu(self):
        return self.device.device == "GPU"

    def _ones_like_data(self):
        if self._is_gpu():
            return cl_array.ones(self.data.queue, self.data.shape, dtype=np.float32)
        return np.ones_like(self.data, dtype=np.float32)

    def _zeros_like_data(self):
        if self._is_gpu():
            return cl_array.zeros(self.data.queue, self.data.shape, dtype=np.float32)
        return np.zeros_like(self.data, dtype=np.float32)


def _normalize_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _create_tensor_from_host(data, device):
    data = np.asarray(data, dtype=np.float32)

    if device.device == "CPU":
        return Tensor(data, device=device)

    if device.device == "GPU":
        backend = OpenCLBackend.instance()
        gpu_data = backend.upload(data)
        return Tensor(gpu_data, device=device)

    raise ValueError(f"Invalid device type: {device.device}")


def ones_like(tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("ones_like() expects a Tensor")

    data = np.ones(tensor.data.shape, dtype=np.float32)
    return _create_tensor_from_host(data, tensor.device)


def zeros_like(tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("zeros_like() expects a Tensor")

    data = np.zeros(tensor.data.shape, dtype=np.float32)
    return _create_tensor_from_host(data, tensor.device)


def full_like(tensor: Tensor, fill_value: float) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("full_like() expects a Tensor")

    data = np.full(tensor.data.shape, fill_value, dtype=np.float32)
    return _create_tensor_from_host(data, tensor.device)


def rand_like(tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("rand_like() expects a Tensor")

    data = np.random.rand(*tensor.data.shape).astype(np.float32)
    return _create_tensor_from_host(data, tensor.device)


def rand(shape, device=None) -> Tensor:
    if device is None:
        device = Device("CPU")

    shape = _normalize_shape(shape)
    data = np.random.rand(*shape).astype(np.float32)
    return _create_tensor_from_host(data, device)

def zeros(shape, device=None) -> Tensor:
    if device is None:
        device = Device("CPU")

    shape = _normalize_shape(shape)
    data = np.zeros(shape, dtype=np.float32)
    return _create_tensor_from_host(data, device)

def ones(shape, device=None) -> Tensor:
    if device is None:
        device = Device("CPU")

    shape = _normalize_shape(shape)
    data = np.ones(shape, dtype=np.float32)
    return _create_tensor_from_host(data, device)
