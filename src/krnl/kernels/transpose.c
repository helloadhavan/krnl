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