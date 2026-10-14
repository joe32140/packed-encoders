"""Triton kernels for post-norm ("sandwich") residual blocks.

A post-norm block normalizes its sublayer's output before adding it to the residual stream,
then the next sublayer normalizes that sum again. Each kernel here rounds where the eager
module chain rounds (every module returns the model dtype), so it reproduces the chain's
result rather than an fp32 idealisation.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rms_kernel(X, W, Y, stride_t, stride_h, R, eps, in_scale, H: tl.constexpr, D: tl.constexpr,
                BR: tl.constexpr, HAS_W: tl.constexpr, SCALE_IN: tl.constexpr):
    """BR (token, head) rows: optionally x * in_scale rounded to the model dtype, then
    RMSNorm(x) [* w] in fp32, rounded once."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    m = r < R
    offs = tl.arange(0, D)
    x = tl.load(X + (r // H)[:, None] * stride_t + (r % H)[:, None] * stride_h + offs[None, :], mask=m[:, None],
                other=0.).to(tl.float32)
    if SCALE_IN:
        x = (x * in_scale).to(Y.dtype.element_ty).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)[:, None]
    if HAS_W:
        y = y * tl.load(W + offs)[None, :]
    tl.store(Y + r[:, None] * D + offs[None, :], y.to(Y.dtype.element_ty), mask=m[:, None])


def rms_norm(x: Tensor, w: Tensor | None, eps: float, in_scale: float | None = None) -> Tensor:
    """x: (T, D) or (T, H, D) with unit last stride (other strides free, so a head slice of a
    projection is read in place); w: (D,) fp32, or None for an unscaled norm; in_scale: a
    multiplier applied first. Returns a contiguous tensor of x's shape."""
    T, H, D = (x.shape[0], 1, x.shape[1]) if x.ndim == 2 else x.shape
    stride_h = 0 if x.ndim == 2 else x.stride(1)
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    R = T * H
    if R:
        BR = max(1, 2048 // D)
        _rms_kernel[(triton.cdiv(R, BR),)](x, w if w is not None else x, y, x.stride(0), stride_h, R, eps,
                                           1.0 if in_scale is None else in_scale, H=H, D=D, BR=BR,
                                           HAS_W=w is not None, SCALE_IN=in_scale is not None, num_warps=4)
    return y


@triton.jit
def _post_norm_add_kernel(DELTA, RES, W, S, WN, OUT, NORM, T, eps, D: tl.constexpr, BR: tl.constexpr,
                          HAS_S: tl.constexpr, HAS_NEXT: tl.constexpr):
    """BR tokens: residual + RMSNorm(delta) * w, then * s, then RMSNorm(sum) * wn, each
    module's output rounded to the model dtype."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    m = r < T
    offs = tl.arange(0, D)
    idx = r[:, None] * D + offs[None, :]
    d = tl.load(DELTA + idx, mask=m[:, None], other=0.).to(tl.float32)
    n = d * tl.rsqrt(tl.sum(d * d, axis=1) / D + eps)[:, None] * tl.load(W + offs)[None, :]
    n = n.to(OUT.dtype.element_ty).to(tl.float32)
    s = (tl.load(RES + idx, mask=m[:, None], other=0.).to(tl.float32) + n).to(OUT.dtype.element_ty)
    if HAS_S:
        s = (s.to(tl.float32) * tl.load(S).to(tl.float32)).to(OUT.dtype.element_ty)
    tl.store(OUT + idx, s, mask=m[:, None])
    if HAS_NEXT:
        sf = s.to(tl.float32)
        h = sf * tl.rsqrt(tl.sum(sf * sf, axis=1) / D + eps)[:, None] * tl.load(WN + offs)[None, :]
        tl.store(NORM + idx, h.to(NORM.dtype.element_ty), mask=m[:, None])


def post_norm_add(delta: Tensor, residual: Tensor, w: Tensor, eps: float, scalar: Tensor | None = None,
                  w_next: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
    """delta, residual: contiguous (T, D); w, w_next: (D,) fp32; scalar: (1,) fp32. Returns the
    new residual stream, and with `w_next` also its norm for the next sublayer."""
    T, D = delta.shape
    out = torch.empty_like(residual)
    norm = torch.empty_like(residual) if w_next is not None else out
    if T:
        BR = max(1, 2048 // D)
        _post_norm_add_kernel[(triton.cdiv(T, BR),)](
            delta, residual, w, scalar if scalar is not None else w, w_next if w_next is not None else w, out, norm,
            T, eps, D=D, BR=BR, HAS_S=scalar is not None, HAS_NEXT=w_next is not None, num_warps=4)
    return (out, norm) if w_next is not None else out


_GELU_B, _GELU_WARPS = 2048, 16
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@triton.jit
def _gelu_mul_kernel(G, U, Y, stride_g, stride_u, T, c, D: tl.constexpr, B: tl.constexpr):
    """tanh GELU(g) rounded to the model dtype, times u, rounded."""
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    m = offs < T
    row, col = offs // D, offs % D
    g = tl.load(G + row * stride_g + col, mask=m, other=0.).to(tl.float32)
    u = tl.load(U + row * stride_u + col, mask=m, other=0.).to(tl.float32)
    z = c * (g + 0.044715 * g * g * g)
    a = (0.5 * g * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + offs, (a * u).to(Y.dtype.element_ty), mask=m)


def gelu_mul(g: Tensor, u: Tensor) -> Tensor:
    """g, u: (T, D) with unit last stride (two halves of one GEMM, or a slice of a wider tensor).
    Returns a contiguous (T, D)."""
    T, D = g.shape
    y = torch.empty((T, D), device=g.device, dtype=g.dtype)
    n = T * D
    if n:
        _gelu_mul_kernel[(triton.cdiv(n, _GELU_B),)](g, u, y, g.stride(0), u.stride(0), n, _SQRT_2_OVER_PI, D=D,
                                                     B=_GELU_B, num_warps=_GELU_WARPS)
    return y
