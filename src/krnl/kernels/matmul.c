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