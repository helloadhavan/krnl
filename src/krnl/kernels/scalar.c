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