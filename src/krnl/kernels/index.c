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