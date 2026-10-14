"""Inference RMSNorm without backward statistics or autograd setup.

Matches FLA's fused residual arithmetic: normalize the fp32 sum and store a
separate residual rounded to its dtype. Inputs are never overwritten.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, R, W, Y, SUM, stride_x, stride_r, eps,
                D: tl.constexpr, BLOCK: tl.constexpr, HAS_RESIDUAL: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + col, mask=col < D, other=0.).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * stride_r + col, mask=col < D, other=0.).to(tl.float32)
        tl.store(SUM + row * D + col, x, mask=col < D)
    scale = 1 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
    w = tl.load(W + col, mask=col < D, other=0.).to(tl.float32)
    tl.store(Y + row * D + col, x * scale * w, mask=col < D)


def _run(x, w, eps, residual=None):
    d = x.shape[-1]
    rows = x.numel() // d
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    summed = None if residual is None else torch.empty(x.shape, device=x.device, dtype=residual.dtype)
    x = x.reshape(rows, d)
    if residual is not None:
        residual = residual.reshape(rows, d)
    if rows:
        _rms_kernel[(rows,)](
            x, residual, w, y, summed, x.stride(-2),
            0 if residual is None else residual.stride(-2), eps,
            D=d, BLOCK=triton.next_power_of_2(d), HAS_RESIDUAL=residual is not None,
            num_warps=4 if d <= 4096 else 8)
    return y, summed


def rms_norm(x, w, eps):
    return _run(x, w, eps)[0]


def add_rms_norm(x, residual, w, eps):
    return _run(x, w, eps, residual)
