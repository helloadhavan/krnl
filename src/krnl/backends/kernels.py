MATMUL_KERNEL = """
__kernel void matmul_f32(
    __global const float *a,
    __global const float *b,
    __global float *c,
    const int M,
    const int K,
    const int N
)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }

        c[row * N + col] = sum;
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