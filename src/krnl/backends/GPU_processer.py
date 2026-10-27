from __future__ import annotations

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

try:
    import pyclblast
except ImportError:
    pyclblast = None

from .kernels import (
    ELEMENTWISE_KERNEL,
    MATMUL_KERNEL,
    TRANSPOSE_KERNEL,
    ELEMENTWISE_SCALER_KERNELS
)


def get_gpu_info():
    """
    Return information about all available OpenCL GPUs.
    """

    result = []

    for platform in cl.get_platforms():
        devices = platform.get_devices(
            device_type=cl.device_type.GPU
        )

        for device in devices:
            result.append(
                {
                    "platform": platform.name,
                    "device": device.name,
                    "max_compute_units": (
                        device.max_compute_units
                    ),
                    "max_work_group_size": (
                        device.max_work_group_size
                    ),
                    "global_mem_size": device.global_mem_size,
                    "local_mem_size": device.local_mem_size,
                    "max_clock_frequency": (
                        device.max_clock_frequency
                    ),
                }
            )

    return result


class OpenCLBackend:
    """
    OpenCL backend.

    GPUArray uses this class for memory allocation and
    kernel execution.
    """

    _instances = {}

    @classmethod
    def instance(cls, device_id=0):
        if device_id not in cls._instances:
            cls._instances[device_id] = cls(device_id)

        return cls._instances[device_id]

    def __init__(self, device_id=0):
        self.context, self.device = self._create_context(
            device_id
        )

        self.queue = cl.CommandQueue(self.context)

        self.elementwise_program = cl.Program(
            self.context,
            ELEMENTWISE_KERNEL,
        ).build()

        self.matmul_program = cl.Program(
            self.context,
            MATMUL_KERNEL,
        ).build()

        self.transpose_program = cl.Program(
            self.context,
            TRANSPOSE_KERNEL,
        ).build()

        self.elementwise_scaler_program = cl.Program(
            self.context,
            ELEMENTWISE_SCALER_KERNELS
        ).build()

    @staticmethod
    def _create_context(device_id):
        platforms = cl.get_platforms()
        devices = []

        for platform in platforms:
            devices.extend(
                platform.get_devices(
                    device_type=cl.device_type.GPU
                )
            )

        if not devices:
            raise RuntimeError(
                "No OpenCL GPU was found."
            )

        if device_id < 0 or device_id >= len(devices):
            raise ValueError(
                f"Invalid GPU device_id={device_id}. "
                f"Available GPUs: {len(devices)}."
            )

        device = devices[device_id]
        context = cl.Context(devices=[device])

        return context, device

    def upload(self, data):
        data = np.asarray(data, dtype=np.float32)
        data = np.ascontiguousarray(data)

        return cl_array.to_device(
            self.queue,
            data,
        )

    def empty(self, shape):
        return cl_array.empty(
            self.queue,
            shape,
            dtype=np.float32,
        )

    def zeros(self, shape):
        out = cl_array.empty(
            self.queue,
            shape,
            dtype=np.float32,
        )

        self.elementwise_program.fill_f32(
            self.queue,
            self._global_size(out.size),
            None,
            out.data,
            np.float32(0),
            np.uint32(out.size),
        )

    def ones(self, shape):
        out = cl_array.empty(
            self.queue,
            shape,
            dtype=np.float32,
        )

        self.elementwise_program.fill_f32(
            self.queue,
            self._global_size(out.size),
            None,
            out.data,
            np.float32(1),
            np.uint32(out.size),
        )

    def _validate_array(self, value, name="value"):
        if isinstance(value, GPUArray):
            value = value.array

        if not isinstance(value, cl_array.Array):
            raise TypeError(
                f"{name} must be a GPUArray or "
                "pyopencl.array.Array"
            )

        if value.dtype != np.dtype(np.float32):
            raise TypeError(
                f"{name} must have dtype float32"
            )

        return value

    def _validate_binary_inputs(self, a, b):
        a = self._validate_array(a, "a")
        b = self._validate_array(b, "b")

        if a.queue.context != b.queue.context:
            raise ValueError(
                "Arrays must use the same OpenCL context."
            )

        if a.shape != b.shape:
            raise ValueError(
                f"Shapes must match: {a.shape} != {b.shape}"
            )

        return a, b

    @staticmethod
    def _global_size(size):
        return (int(size),)

    def add(self, a, b):
        a, b = self._validate_binary_inputs(a, b)
        out = cl_array.empty_like(a)

        self.elementwise_program.add_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def sub(self, a, b):
        a, b = self._validate_binary_inputs(a, b)
        out = cl_array.empty_like(a)

        self.elementwise_program.sub_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def mul(self, a, b):
        a, b = self._validate_binary_inputs(a, b)
        out = cl_array.empty_like(a)

        self.elementwise_program.mul_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def div(self, a, b):
        a, b = self._validate_binary_inputs(a, b)
        out = cl_array.empty_like(a)

        self.elementwise_program.div_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def neg(self, a):
        a = self._validate_array(a)
        out = cl_array.empty_like(a)

        self.elementwise_program.neg_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def abs(self, a):
        a = self._validate_array(a)
        out = cl_array.empty_like(a)

        self.elementwise_program.abs_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def power(self, a, exponent):
        a = self._validate_array(a)
        out = cl_array.empty_like(a)

        self.elementwise_program.pow_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            out.data,
            np.float32(exponent),
            np.uint32(a.size),
        )

        return out

    def mul_scalar(self, a, scalar):
        a = self._validate_array(a, "a")

        out = cl_array.empty_like(a)

        self.elementwise_program.mul_scalar_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            np.float32(scalar),
            out.data,
            np.uint32(a.size),
        )

        return out

    def add_scalar(self, a, scalar):
        a = self._validate_array(a, "a")

        out = cl_array.empty_like(a)

        self.elementwise_program.mul_scalar_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            np.float32(scalar),
            out.data,
            np.uint32(a.size),
        )

        return out

    def relu(self, a):
        a = self._validate_array(a)
        out = cl_array.empty_like(a)

        self.elementwise_program.relu_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def sigmoid(self, a):
        a = self._validate_array(a)
        out = cl_array.empty_like(a)

        self.elementwise_program.sigmoid_f32(
            self.queue,
            self._global_size(a.size),
            None,
            a.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def fill(self, shape, value):
        out = cl_array.empty(
            self.queue,
            shape,
            dtype=np.float32,
        )

        self.elementwise_program.fill_f32(
            self.queue,
            self._global_size(out.size),
            None,
            out.data,
            np.float32(value),
            np.uint32(out.size),
        )

        return out

    def add_row_bias(self, matrix, bias):
        """
        Add a vector bias to every row.

        matrix shape: (rows, columns)
        bias shape:   (columns,)
        """

        matrix = self._validate_array(matrix, "matrix")
        bias = self._validate_array(bias, "bias")

        if matrix.ndim != 2:
            raise ValueError(
                "matrix must be 2-dimensional"
            )

        if bias.ndim != 1:
            raise ValueError(
                "bias must be 1-dimensional"
            )

        rows, columns = matrix.shape

        if bias.shape != (columns,):
            raise ValueError(
                f"Expected bias shape {(columns,)}, "
                f"got {bias.shape}"
            )

        out = cl_array.empty_like(matrix)

        self.elementwise_program.add_row_bias_f32(
            self.queue,
            self._global_size(matrix.size),
            None,
            matrix.data,
            bias.data,
            out.data,
            np.uint32(rows),
            np.uint32(columns),
        )

        return out

    def sum_to_row(self, matrix):
        """
        Sum a matrix across rows.

        Input:  (rows, columns)
        Output: (columns,)
        """

        matrix = self._validate_array(matrix, "matrix")

        if matrix.ndim != 2:
            raise ValueError(
                "matrix must be 2-dimensional"
            )

        rows, columns = matrix.shape

        out = cl_array.empty(
            self.queue,
            (columns,),
            dtype=np.float32,
        )

        self.elementwise_program.sum_to_row_f32(
            self.queue,
            (int(columns),),
            None,
            matrix.data,
            out.data,
            np.uint32(rows),
            np.uint32(columns),
        )

        return out

    def transpose(self, a):
        """
        Transpose a 2-D array into a contiguous GPU array.
        """

        a = self._validate_array(a)

        if a.ndim != 2:
            raise ValueError(
                "transpose currently supports 2-D arrays only"
            )

        rows, columns = a.shape

        out = cl_array.empty(
            self.queue,
            (columns, rows),
            dtype=np.float32,
        )

        self.transpose_program.transpose_f32(
            self.queue,
            (int(rows), int(columns)),
            None,
            a.data,
            out.data,
            np.uint32(rows),
            np.uint32(columns),
        )

        return out

    def matmul(self, a, b):
        """
        Matrix multiplication.

        Uses PyCLBlast when available. Otherwise it uses
        the fallback OpenCL kernel.
        """

        a = self._validate_array(a, "a")
        b = self._validate_array(b, "b")

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                "matmul requires 2-D arrays"
            )

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Incompatible shapes: "
                f"{a.shape} @ {b.shape}"
            )

        if a.queue.context != b.queue.context:
            raise ValueError(
                "Arrays must use the same OpenCL context."
            )

        m, k = a.shape
        _, n = b.shape

        # Ensure row-major contiguous arrays.
        a = a.copy()
        b = b.copy()

        out = cl_array.empty(
            self.queue,
            (m, n),
            dtype=np.float32,
        )

        if pyclblast is not None:
            pyclblast.gemm(
                self.queue,
                m,
                n,
                k,
                a,
                b,
                out,
                a_ld=k,
                b_ld=n,
                c_ld=n,
            )
        else:
            self.matmul_program.matmul_f32(
                self.queue,
                (int(m), int(n)),
                None,
                a.data,
                b.data,
                out.data,
                np.uint32(m),
                np.uint32(k),
                np.uint32(n),
            )

        return out


class GPUArray:
    """
    GPU-resident float32 array.

    The underlying pyopencl.array.Array is available through
    the .array property.
    """

    def __init__(self, data, backend=None):
        self.backend = (
            backend
            if backend is not None
            else OpenCLBackend.instance()
        )

        if isinstance(data, GPUArray):
            self.array = data.array
            return

        if isinstance(data, cl_array.Array):
            if data.dtype != np.dtype(np.float32):
                raise TypeError(
                    "GPUArray only supports float32"
                )

            self.array = data
            return

        self.array = self.backend.upload(data)

    @classmethod
    def empty(cls, shape, backend=None):
        backend = backend or OpenCLBackend.instance()

        return cls(
            backend.empty(shape),
            backend=backend,
        )

    @classmethod
    def zeros(cls, shape, backend=None):
        backend = backend or OpenCLBackend.instance()

        return cls(
            backend.zeros(shape),
            backend=backend,
        )

    @classmethod
    def ones(cls, shape, backend=None):
        backend = backend or OpenCLBackend.instance()

        return cls(
            backend.ones(shape),
            backend=backend,
        )

    @property
    def data(self):
        return self.array

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def queue(self):
        return self.array.queue

    @property
    def context(self):
        return self.array.queue.context

    def get(self):
        return self.array.get()

    def wait(self):
        self.queue.finish()
        return self

    def copy(self):
        return GPUArray(
            self.array.copy(),
            backend=self.backend,
        )

    def _coerce(self, other):
        if isinstance(other, GPUArray):
            return other

        if isinstance(other, cl_array.Array):
            return GPUArray(
                other,
                backend=self.backend,
            )

        if np.isscalar(other):
            other = np.full(
                self.shape,
                other,
                dtype=np.float32,
            )

        return GPUArray(
            other,
            backend=self.backend,
        )

    def _binary(self, other, operation):
        other = self._coerce(other)

        if self.shape != other.shape:
            raise ValueError(
                f"GPUArray shapes must match: "
                f"{self.shape} != {other.shape}"
            )

        if self.context != other.context:
            raise ValueError(
                "GPUArray objects use different contexts"
            )

        result = getattr(self.backend, operation)(
            self.array,
            other.array,
        )

        return GPUArray(
            result,
            backend=self.backend,
        )

    def __add__(self, other):
        return self._binary(other, "add") if not np.isscalar(other) else self._binary(other, "add_scaler")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        return self._binary(other, "mul") if not np.isscalar(other) else self._binary(other, "mul_scaler")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._binary(other, "div")

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __neg__(self):
        return GPUArray(
            self.backend.neg(self.array),
            backend=self.backend,
        )

    def __abs__(self):
        return GPUArray(
            self.backend.abs(self.array),
            backend=self.backend,
        )

    def __pow__(self, exponent):
        return GPUArray(
            self.backend.power(
                self.array,
                exponent,
            ),
            backend=self.backend,
        )

    def __matmul__(self, other):
        other = self._coerce(other)

        return GPUArray(
            self.backend.matmul(
                self.array,
                other.array,
            ),
            backend=self.backend,
        )

    def transpose(self):
        return GPUArray(
            self.backend.transpose(self.array),
            backend=self.backend,
        )

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape):
        return GPUArray(
            self.array.reshape(shape),
            backend=self.backend,
        )

    def sum(self):
        return GPUArray(
            cl_array.sum(self.array),
            backend=self.backend,
        )

    def mean(self):
        return self.sum() / float(self.size)

    def sigmoid(self):
        return GPUArray(
            self.backend.sigmoid(self.array),
            backend=self.backend,
        )

    def relu(self):
        return GPUArray(
            self.backend.relu(self.array),
            backend=self.backend,
        )

    def __repr__(self):
        return (
            f"GPUArray("
            f"shape={self.shape}, "
            f"dtype={self.dtype}"
            f")"
        )