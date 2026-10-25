import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import Program

from .kernels import MATMUL_KERNEL, SIGMOID_KERNEL, ELEMENTWISE_KERNEL


def get_gpu_info():
    platforms = cl.get_platforms()
    gpu_info = []

    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for device in devices:
            info = {
                "platform": platform.name,
                "device": device.name,
                "max_compute_units": device.max_compute_units,
                "max_work_group_size": device.max_work_group_size,
                "global_mem_size": device.global_mem_size,
                "local_mem_size": device.local_mem_size,
                "max_clock_frequency": device.max_clock_frequency,
            }
            gpu_info.append(info)

    return gpu_info


class OpenCLBackend(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.device = self.queue.device

        self.elementwise_program = Program(
            self.context,
            ELEMENTWISE_KERNEL
        ).build()

        self.matmul_program = Program(
            self.context,
            MATMUL_KERNEL
        ).build()

        self.sigmoid_program = Program(
            self.context,
            SIGMOID_KERNEL
        ).build()

    def upload(self, data):
        """
        Upload a NumPy array to the GPU.

        The backend currently supports float32 tensors.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)

        return cl_array.to_device(
            self.queue,
            data
        )

    def _validate_binary_inputs(self, a, b):
        """
        Validate inputs for elementwise operations.
        """
        if not isinstance(a, cl_array.Array):
            raise TypeError("a must be a pyopencl.array.Array")

        if not isinstance(b, cl_array.Array):
            raise TypeError("b must be a pyopencl.array.Array")

        if a.shape != b.shape:
            raise ValueError(
                f"Tensor shapes must match: {a.shape} != {b.shape}"
            )

        if a.dtype != b.dtype:
            raise TypeError(
                f"Tensor dtypes must match: {a.dtype} != {b.dtype}"
            )

        if a.dtype != np.dtype(np.float32):
            raise TypeError(
                "This OpenCL backend currently supports float32 only"
            )

    def _elementwise_global_size(self, n):
        """
        Return a one-dimensional launch size.

        Exact sizing is valid because each kernel also checks i < n.
        """
        return (n,)

    def add(self, a, b):
        """
        Elementwise addition: a + b.
        """
        self._validate_binary_inputs(a, b)

        out = cl_array.empty_like(a)

        self.elementwise_program.add_f32(
            self.queue,
            self._elementwise_global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def sub(self, a, b):
        """
        Elementwise subtraction: a - b.
        """
        self._validate_binary_inputs(a, b)

        out = cl_array.empty_like(a)

        self.elementwise_program.sub_f32(
            self.queue,
            self._elementwise_global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def mul(self, a, b):
        """
        Elementwise multiplication: a * b.
        """
        self._validate_binary_inputs(a, b)

        out = cl_array.empty_like(a)

        self.elementwise_program.mul_f32(
            self.queue,
            self._elementwise_global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def div(self, a, b):
        """
        Elementwise division: a / b.

        Division by zero follows the device's floating-point behavior.
        """
        self._validate_binary_inputs(a, b)

        out = cl_array.empty_like(a)

        self.elementwise_program.div_f32(
            self.queue,
            self._elementwise_global_size(a.size),
            None,
            a.data,
            b.data,
            out.data,
            np.uint32(a.size),
        )

        return out

    def matmul(self, a, b, tile_size=16):
        """
        Tiled matrix multiplication.

        A has shape (M, K)
        B has shape (K, N)
        C has shape (M, N)
        """
        if not isinstance(a, cl_array.Array):
            raise TypeError("a must be a pyopencl.array.Array")

        if not isinstance(b, cl_array.Array):
            raise TypeError("b must be a pyopencl.array.Array")

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                "matmul requires two 2D tensors"
            )

        if a.dtype != np.dtype(np.float32):
            raise TypeError("a must use float32")

        if b.dtype != np.dtype(np.float32):
            raise TypeError("b must use float32")

        M, K = a.shape
        K2, N = b.shape

        if K != K2:
            raise ValueError(
                f"Incompatible shapes for matmul: "
                f"{a.shape} and {b.shape}"
            )

        if not isinstance(tile_size, int):
            raise TypeError("tile_size must be an integer")

        if tile_size <= 0:
            raise ValueError(
                "tile_size must be greater than zero"
            )

        local_size = (
            tile_size,
            tile_size,
        )

        work_items_per_group = tile_size * tile_size

        if work_items_per_group > self.device.max_work_group_size:
            raise ValueError(
                f"tile_size={tile_size} requires "
                f"{work_items_per_group} work-items, but this device "
                f"supports at most "
                f"{self.device.max_work_group_size}"
            )

        float_size = np.dtype(np.float32).itemsize
        tile_bytes = tile_size * tile_size * float_size

        required_local_memory = 2 * tile_bytes

        if required_local_memory > self.device.local_mem_size:
            raise ValueError(
                f"tile_size={tile_size} requires "
                f"{required_local_memory} bytes of local memory, but "
                f"the device provides only "
                f"{self.device.local_mem_size} bytes"
            )

        out = cl_array.empty(
            self.queue,
            (M, N),
            dtype=np.float32
        )


        global_rows = (
            (M + tile_size - 1) // tile_size
        ) * tile_size

        global_cols = (
            (N + tile_size - 1) // tile_size
        ) * tile_size

        global_size = (
            global_rows,
            global_cols,
        )

        self.matmul_program.matmul_tiled_f32(
            self.queue,
            global_size,
            local_size,

            a.data,
            b.data,
            out.data,

            np.uint32(M),
            np.uint32(K),
            np.uint32(N),

            np.uint32(tile_size),

            cl.LocalMemory(tile_bytes),
            cl.LocalMemory(tile_bytes),
        )

        return out

    def __add__(self, a, b):
        return self.add(a, b)

    def __sub__(self, a, b):
        return self.sub(a, b)

    def __mul__(self, a, b):
        return self.mul(a, b)

    def __truediv__(self, a, b):
        return self.div(a, b)

    def __matmul__(self, a, b):
        return self.matmul(a, b)

    def sigmoid(self, input_array: cl_array.Array):
        if not isinstance(input_array, cl_array.Array):
            raise TypeError("sigmoid expects a pyopencl.array.Array")
        if input_array.dtype != np.dtype(np.float32):
            raise TypeError("sigmoid supports float32 arrays only")
        output = cl_array.empty_like(input_array)
        count = np.uint32(input_array.size)
        self.sigmoid_program.sigmoid_f32(
                                    self.queue,
                                    (input_array.size,),
                                    None,
                                    input_array.data,
                                    output.data,
                                    count
        )
        return output