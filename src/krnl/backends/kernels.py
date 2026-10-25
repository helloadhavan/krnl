ELEMENTWISE_KERNEL = r"""
__kernel void add_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint index = get_global_id(0);

    if (index < n) {
        out[index] = a[index] + b[index];
    }
}


__kernel void sub_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint index = get_global_id(0);

    if (index < n) {
        out[index] = a[index] - b[index];
    }
}


__kernel void mul_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint index = get_global_id(0);

    if (index < n) {
        out[index] = a[index] * b[index];
    }
}


__kernel void div_f32(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const uint n
)
{
    const uint index = get_global_id(0);

    if (index < n) {
        out[index] = a[index] / b[index];
    }
}
"""


MATMUL_KERNEL = r"""
__kernel void matmul_tiled_f32(
    __global const float *A,
    __global const float *B,
    __global float *C,

    const uint M,
    const uint K,
    const uint N,

    const uint tile_size,

    __local float *tile_A,
    __local float *tile_B
)
{
    const uint local_row = get_local_id(0);
    const uint local_col = get_local_id(1);

    const uint group_row = get_group_id(0);
    const uint group_col = get_group_id(1);

    const uint row = group_row * tile_size + local_row;
    const uint col = group_col * tile_size + local_col;

    const uint local_index =
        local_row * tile_size + local_col;

    float accumulator = 0.0f;

    const uint number_of_tiles =
        (K + tile_size - 1) / tile_size;

    for (uint tile = 0; tile < number_of_tiles; tile++) {
        const uint a_col = tile * tile_size + local_col;
        const uint b_row = tile * tile_size + local_row;

        /*
         * Load one element of A into local memory.
         * Out-of-range elements are padded with zero.
         */
        if (row < M && a_col < K) {
            tile_A[local_index] =
                A[row * K + a_col];
        } else {
            tile_A[local_index] = 0.0f;
        }

        /*
         * Load one element of B into local memory.
         * Out-of-range elements are padded with zero.
         */
        if (b_row < K && col < N) {
            tile_B[local_index] =
                B[b_row * N + col];
        } else {
            tile_B[local_index] = 0.0f;
        }

        /*
         * Make sure the entire tile has been loaded before
         * any work-item reads from it.
         */
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < tile_size; i++) {
            accumulator +=
                tile_A[local_row * tile_size + i] *
                tile_B[i * tile_size + local_col];
        }

        /*
         * Make sure no work-item starts overwriting the tile
         * while another work-item is still reading it.
         */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /*
     * Do not write padded rows or columns.
     */
    if (row < M && col < N) {
        C[row * N + col] = accumulator;
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