"""
OpenCL kernel source definitions.

This file contains only OpenCL C source code.
Kernel compilation and execution are handled by GPU_processer.py.
"""


ELEMENTWISE_KERNEL = r"""
__kernel void add_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__kernel void sub_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] - b[i];
    }
}


__kernel void mul_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] * b[i];
    }
}


__kernel void div_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] / b[i];
    }
}


__kernel void neg_f32(
    __global const float *a,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = -a[i];
    }
}


__kernel void abs_f32(
    __global const float *a,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = fabs(a[i]);
    }
}


__kernel void pow_f32(
    __global const float *a,
    __global float *out,
    const float exponent,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = pow(a[i], exponent);
    }
}


__kernel void relu_f32(
    __global const float *a,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}


__kernel void relu_backward_f32(
    __global const float *input,
    __global const float *gradient,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = input[i] > 0.0f ? gradient[i] : 0.0f;
    }
}


__kernel void sigmoid_f32(
    __global const float *input,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}


__kernel void sigmoid_backward_f32(
    __global const float *output,
    __global const float *gradient,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        const float y = output[i];
        out[i] = gradient[i] * y * (1.0f - y);
    }
}


__kernel void fill_f32(
    __global float *out,
    const float value,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = value;
    }
}


__kernel void add_row_bias_f32(
    __global const float *input,
    __global const float *bias,
    __global float *out,
    const uint rows,
    const uint columns
)
{
    const uint i = get_global_id(0);
    const uint total = rows * columns;

    if (i < total) {
        const uint column = i % columns;
        out[i] = input[i] + bias[column];
    }
}


__kernel void sum_to_row_f32(
    __global const float *input,
    __global float *out,
    const uint rows,
    const uint columns
)
{
    const uint column = get_global_id(0);

    if (column < columns) {
        float value = 0.0f;

        for (uint row = 0; row < rows; ++row) {
            value += input[row * columns + column];
        }

        out[column] = value;
    }
}
"""


MATMUL_KERNEL = r"""
__kernel void matmul_f32(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const uint M,
    const uint K,
    const uint N
)
{
    const uint row = get_global_id(0);
    const uint column = get_global_id(1);

    if (row < M && column < N) {
        float value = 0.0f;

        for (uint k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + column];
        }

        C[row * N + column] = value;
    }
}
"""


TRANSPOSE_KERNEL = r"""
__kernel void transpose_f32(
    __global const float *input,
    __global float *output,
    const uint rows,
    const uint columns
)
{
    const uint row = get_global_id(0);
    const uint column = get_global_id(1);

    if (row < rows && column < columns) {
        output[column * rows + row] =
            input[row * columns + column];
    }
}
"""

SIGMOID_KERNEL = """
__kernel void sigmoid_activation(__global const float* input, 
                                 __global float* output, 
                                 const unsigned int count) 
{
    int id = get_global_id(0);
    
    if (id < count) {
        output[id] = 1.0f / (1.0f + exp(-input[id]));
    }
}
"""