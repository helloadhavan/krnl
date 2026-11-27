from krnl import zeros


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    x: (N, C_in, H, W)
    weight: (C_out, C_in, kH, kW)
    bias: (C_out,) or None
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape


    if padding > 0:
        x_padded = ((N, C_in, H + 2 * padding, W + 2 * padding))
        x_padded[:, :, padding:padding + H, padding:padding + W] = x
    else:
        x_padded = x

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    out = torch.zeros((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            patch = x_padded[:, :, h_start:h_start + kH, w_start:w_start + kW]
            out[:, :, i, j] = (patch[:, None, :, :, :] * weight[None, :, :, :, :]).sum(dim=(2, 3, 4))

    # 5. Add bias if provided
    if bias is not None:
        out += bias[None, :, None, None]

    return out
