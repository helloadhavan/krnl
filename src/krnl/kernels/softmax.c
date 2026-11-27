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