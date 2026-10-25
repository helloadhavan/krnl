import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import Program
from backends.backends.kernels import MATMUL_KERNEL, SIGMOID_KERNEL

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

class OpenCLBackend:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        self.matmul_program = Program(self.context, MATMUL_KERNEL).build()
        self.sigmoid_program = Program(self.context, SIGMOID_KERNEL).build()

    def upload(self, data):
        data = np.ascontiguousarray(data)
        return cl_array.to_device(self.queue, data)

    def __add__(self, a: cl_array.Array, b: cl_array.Array):
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match")
        if a.dtype != b.dtype:
            raise TypeError("Tensor dtypes must match")
        if a.dtype != np.dtype(np.float32):
            raise TypeError("Initial OpenCL backend supports float32 only")

        result = a + b

        return result

    def __mul__(self, a: cl_array.Array, b: cl_array.Array):
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match")
        if a.dtype != b.dtype:
            raise TypeError("Tensor dtypes must match")
        if a.dtype != np.dtype(np.float32):
            raise TypeError("Initial OpenCL backend supports float32 only")

        result = a * b

        return result

    def __sub__(self, a: cl_array.Array, b: cl_array.Array):
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match")
        if a.dtype != b.dtype:
            raise TypeError("Tensor dtypes must match")
        if a.dtype != np.dtype(np.float32):
            raise TypeError("Initial OpenCL backend supports float32 only")

        result = a - b

        return result

    def __matmul__(self, a: cl_array.Array, b: cl_array.Array):
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("matmul requires two 2D tensors")

        M, K = a.shape
        K2, N = b.shape

        if K != K2:
            raise ValueError(
                f"Incompatible shapes for matmul: {a.shape} and {b.shape}"
            )

        if a.dtype != np.dtype(np.float32) or b.dtype != np.dtype(np.float32):
            raise TypeError("matmul supports float32 only")

        out = cl_array.empty(
            self.queue,
            (M, N),
            dtype=np.float32
        )

        self.matmul_program.matmul_f32(
            self.queue,
            (M, N),
            None,
            a.data,
            b.data,
            out.data,
            np.int32(M),
            np.int32(K),
            np.int32(N)
        )

    def sigmoid(self, input_array: cl_array.Array):
        """
        Apply sigmoid element-wise to a GPU float32 array.

        sigmoid(x) = 1 / (1 + exp(-x))
        """

        if not isinstance(input_array, cl_array.Array):
            raise TypeError(
                "sigmoid expects a pyopencl.array.Array"
            )

        if input_array.dtype != np.dtype(np.float32):
            raise TypeError(
                "sigmoid supports float32 arrays only"
            )

        output = cl_array.empty_like(input_array)

        count = np.uint32(input_array.size)

        self.sigmoid_program.sigmoid_f32(
            self.queue,
            (input_array.size,),
            None,
            input_array.data,
            output.data,
            count,
        )

        return output


