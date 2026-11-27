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

__kernel void log_f32(
    __global const float *a,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = log(a[i]);
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

__kernel void mul_scalar_f32(
    __global const float *a,
    const float scalar,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] * scalar;
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

ELEMENTWISE_SCALER_KERNELS = """

__kernel void mul_scalar_f32(
    __global const float *a,
    const float scalar,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] * scalar;
    }
}

__kernel void add_scalar_f32(
    __global const float *a,
    const float scalar,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] + scalar;
    }
}

__kernel void sub_scalar_f32(
    __global const float *a,
    const float scalar,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] - scalar;
    }
}

__kernel void div_scalar_f32(
    __global const float *a,
    const float scalar,
    __global float *out,
    const uint n
)
{
    const uint i = get_global_id(0);

    if (i < n) {
        out[i] = a[i] / scalar;
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

SOFTMAX_KERNEL = r"""
__kernel void softmax_f32(
    __global const float *input,
    __global float *output,
    const uint rows,
    const uint cols
)
{
    const uint row = get_global_id(0);

    if (row >= rows) return;

    const uint offset = row * cols;

    // Step 1: find max for numerical stability
    float max_val = input[offset];
    for (uint j = 1; j < cols; ++j) {
        float v = input[offset + j];
        if (v > max_val) max_val = v;
    }

    // Step 2: compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (uint j = 0; j < cols; ++j) {
        float e = exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum_exp += e;
    }

    // Step 3: normalize
    for (uint j = 0; j < cols; ++j) {
        output[offset + j] /= sum_exp;
    }
}
"""

INDEX_KERNEL = r"""
__kernel void index_select_f32(
    __global const float *input,
    __global const int   *indices,
    __global float       *output,
    const uint num_indices,
    const uint cols
)
{
    const uint gid = get_global_id(0);
    const uint total = num_indices * cols;

    if (gid < total) {
        const uint out_row = gid / cols;
        const uint col     = gid % cols;
        const uint src_row = indices[out_row];
        output[gid] = input[src_row * cols + col];
    }
}
"""

IM2COL_KERNEL = r"""
__kernel void im2col_f32(
    __global const float *input,
    __global float       *cols_out,
    const uint N,
    const uint C_in,
    const uint H,
    const uint W,
    const uint kH,
    const uint kW,
    const uint stride,
    const uint padding,
    const uint H_out,
    const uint W_out
)
{
    /*
     * Each work-item handles one element in the output column matrix.
     * cols_out shape: (N * H_out * W_out,  C_in * kH * kW)
     * Global ID maps to a flat index into that matrix.
     */
    const uint gid = get_global_id(0);
    const uint col_w = C_in * kH * kW;          /* columns per row */
    const uint total = N * H_out * W_out * col_w;

    if (gid >= total) return;

    /* Decode position in the output matrix */
    const uint row   = gid / col_w;             /* which patch */
    const uint col   = gid % col_w;             /* which element in patch */

    const uint n     = row / (H_out * W_out);
    const uint hw    = row % (H_out * W_out);
    const uint h_out = hw / W_out;
    const uint w_out = hw % W_out;

    /* Decode channel / kernel position from col */
    const uint c     = col / (kH * kW);
    const uint kk    = col % (kH * kW);
    const uint kh    = kk / kW;
    const uint kw    = kk % kW;

    /* Map back to input coordinates */
    const int h_in = (int)(h_out * stride + kh) - (int)padding;
    const int w_in = (int)(w_out * stride + kw) - (int)padding;

    float val = 0.0f;
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const uint input_idx = ((n * C_in + c) * H + (uint)h_in) * W + (uint)w_in;
        val = input[input_idx];
    }

    cols_out[gid] = val;
}
"""

COL2IM_KERNEL = r"""
__kernel void col2im_f32(
    __global const float *cols_in,
    __global float       *output,
    const uint N,
    const uint C_in,
    const uint H,
    const uint W,
    const uint kH,
    const uint kW,
    const uint stride,
    const uint padding,
    const uint H_out,
    const uint W_out
)
{
    /*
     * Each work-item handles one element in the input gradient (N, C_in, H, W).
     * We scatter-add from the column matrix back to the input.
     */
    const uint gid = get_global_id(0);
    const uint total = N * C_in * H * W;

    if (gid >= total) return;

    /* Decode position in the input tensor */
    const uint n = gid / (C_in * H * W);
    const uint r = gid % (C_in * H * W);
    const uint c = r / (H * W);
    const uint hw = r % (H * W);
    const uint h = hw / W;
    const uint w = hw % W;

    /* Padded coordinates */
    const int h_pad = (int)h + (int)padding;
    const int w_pad = (int)w + (int)padding;

    const uint col_w = C_in * kH * kW;   /* columns per row in col matrix */

    float val = 0.0f;

    /* Iterate over all kernel positions that could have read this input pixel */
    for (uint kh = 0; kh < kH; ++kh) {
        for (uint kw = 0; kw < kW; ++kw) {
            int h_out_candidate = h_pad - (int)kh;
            int w_out_candidate = w_pad - (int)kw;

            /* Check if this maps to a valid output position with the given stride */
            if (h_out_candidate < 0 || w_out_candidate < 0) continue;
            if (h_out_candidate % (int)stride != 0) continue;
            if (w_out_candidate % (int)stride != 0) continue;

            uint ho = (uint)h_out_candidate / stride;
            uint wo = (uint)w_out_candidate / stride;

            if (ho >= H_out || wo >= W_out) continue;

            /* Row in the column matrix */
            uint row = (n * H_out + ho) * W_out + wo;
            /* Column in the column matrix */
            uint col_idx = c * kH * kW + kh * kW + kw;

            val += cols_in[row * col_w + col_idx];
        }
    }

    output[gid] = val;
}
"""