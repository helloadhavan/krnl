__kernel void sigmoid_activation(__global const float* input,
                                 __global float* output,
                                 const unsigned int count)
{
    int id = get_global_id(0);

    if (id < count) {
        output[id] = 1.0f / (1.0f + exp(-input[id]));
    }
}