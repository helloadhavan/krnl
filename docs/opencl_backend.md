# OpenCL Backend

krnl includes an experimental GPU backend built on [PyOpenCL](https://documen.tician.de/pyopencl/). It provides device-resident float32 arrays and custom OpenCL kernels for tensor operations.

**Source:** [`src/krnl/backends/GPU_processer.py`](../src/krnl/backends/GPU_processer.py)

## Requirements

- A system with an OpenCL-compatible GPU
- OpenCL runtime and drivers installed
- Python packages: `pyopencl`, optionally `pyclblast` for accelerated matrix multiplication

## Architecture

The GPU backend has two main classes:

### OpenCLBackend

A singleton-per-device manager that handles context creation, kernel compilation, and low-level operations.

```python
from krnl.backends.GPU_processer import OpenCLBackend

backend = OpenCLBackend.instance(device_id=0)
```

The backend compiles four OpenCL kernel programs at initialization:

| Program | Kernels |
|---|---|
| `elementwise_program` | `add_f32`, `sub_f32`, `mul_f32`, `div_f32`, `neg_f32`, `abs_f32`, `pow_f32`, `relu_f32`, `sigmoid_f32`, `fill_f32`, `mul_scalar_f32`, `add_row_bias_f32`, `sum_to_row_f32` |
| `matmul_program` | `matmul_f32` |
| `transpose_program` | `transpose_f32` |
| `elementwise_scaler_program` | `add_scalar_f32`, `sub_scalar_f32`, `div_scalar_f32` |

#### Key Methods

| Method | Description |
|---|---|
| `upload(data)` | Transfer a numpy array to GPU memory |
| `empty(shape)` | Allocate uninitialized GPU memory |
| `zeros(shape)` | Allocate zero-filled GPU memory |
| `ones(shape)` | Allocate ones-filled GPU memory |
| `add(a, b)` | Element-wise addition |
| `sub(a, b)` | Element-wise subtraction |
| `mul(a, b)` | Element-wise multiplication |
| `div(a, b)` | Element-wise division |
| `neg(a)` | Element-wise negation |
| `abs(a)` | Element-wise absolute value |
| `power(a, exponent)` | Element-wise power |
| `mul_scalar(a, scalar)` | Multiply by scalar |
| `add_scalar(a, scalar)` | Add scalar |
| `sub_scalar(a, scalar)` | Subtract scalar |
| `div_scalar(a, scalar)` | Divide by scalar |
| `relu(a)` | Element-wise ReLU |
| `sigmoid(a)` | Element-wise sigmoid |
| `fill(shape, value)` | Create array filled with a value |
| `matmul(a, b)` | Matrix multiplication (uses PyCLBlast if available) |
| `transpose(a)` | 2-D matrix transpose |
| `add_row_bias(matrix, bias)` | Add bias vector to each row |
| `sum_to_row(matrix)` | Sum matrix across rows |

### GPUArray

A high-level wrapper around `pyopencl.array.Array` that provides Python operator overloading and automatic scalar/array dispatch.

```python
from krnl.backends.GPU_processer import GPUArray, OpenCLBackend

backend = OpenCLBackend.instance()
a = GPUArray([1.0, 2.0, 3.0], backend=backend)
b = GPUArray([4.0, 5.0, 6.0], backend=backend)

c = a + b        # element-wise add
d = a * 2.0      # scalar multiply
e = a @ b_2d     # matrix multiply
```

#### Properties

| Property | Description |
|---|---|
| `shape` | Array shape |
| `ndim` | Number of dimensions |
| `size` | Total number of elements |
| `dtype` | Data type (always `float32`) |
| `queue` | OpenCL command queue |
| `data` | Underlying `pyopencl.array.Array` |
| `T` | Transposed array |

#### Methods

| Method | Description |
|---|---|
| `get()` | Copy data back to host as numpy array |
| `copy()` | Create a GPU-side copy |
| `wait()` | Block until all queued operations complete |
| `reshape(shape)` | Reshape the array |
| `sum()` | Sum all elements |
| `mean()` | Mean of all elements |
| `sigmoid()` | Element-wise sigmoid |
| `relu()` | Element-wise ReLU |
| `tolist()` | Convert to Python list (transfers to host) |

#### Supported Operators

`+`, `-`, `*`, `/`, `**`, `@`, `-` (unary), `abs()` — all with automatic scalar detection and reverse variants.

## Matrix Multiplication Acceleration

When [PyCLBlast](https://github.com/CNugteren/CLBlast) is installed, `matmul` uses the optimized GEMM routine. Otherwise it falls back to the custom OpenCL kernel. PyCLBlast typically provides significant speedups for larger matrices.

## Discovering GPUs

```python
from krnl import get_gpu_info

for gpu in get_gpu_info():
    print(f"{gpu['device']}: {gpu['global_mem_size'] / 1e9:.1f} GB")
```
