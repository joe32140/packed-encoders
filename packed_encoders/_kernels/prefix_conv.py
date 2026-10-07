"""Repair causal-convolution branch boundaries in one CuTe launch.

Each output reads its parent's last taps and its own segment directly. FP32
accumulation, optional bias and SiLU precede one rounding to the output dtype.
Only fix_rows are written; inputs and metadata may be reused in CUDA graphs.
"""
from functools import lru_cache

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from packed_encoders._kernels._compile_cache import current_cute_stream, get_compiled


@lru_cache(None)
def _build(kd, vd, width, has_bias, has_history):
    channels = 2 * kd + vd

    @cute.kernel
    def kernel(x: cute.Tensor, w: cute.Tensor, bias: cute.Tensor, rows: cute.Tensor,
               taps: cute.Tensor, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, history: cute.Tensor):
        tile, f, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        c = tile * 128 + tid
        if c < channels:
            y = Float32(0)
            for tap in cutlass.range_constexpr(width):
                source = taps[f, tap]
                value = Float32(0)
                if cutlass.const_expr(has_history):
                    if source < 0:
                        value = history[source + width - 1, c].to(Float32)
                    else:
                        value = x[source, c].to(Float32)
                else:
                    value = x[source, c].to(Float32)
                y += value * w[c, tap].to(Float32)
            if cutlass.const_expr(has_bias):
                y += bias[c].to(Float32)
            y = y / (Float32(1) + cute.math.exp(-y))
            row = rows[f]
            if c < kd:
                q[row, c] = q.element_type(y)
            elif c < 2 * kd:
                k[row, c - kd] = k.element_type(y)
            else:
                v[row, c - 2 * kd] = v.element_type(y)

    @cute.jit
    def launch(x: cute.Tensor, w: cute.Tensor, bias: cute.Tensor, rows: cute.Tensor,
               taps: cute.Tensor, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, history: cute.Tensor,
               stream: cuda_driver.CUstream):
        kernel(x, w, bias, rows, taps, q, k, v, history).launch(
            grid=((channels + 127) // 128, cute.size(rows), 1), block=(128, 1, 1), stream=stream)

    return launch


def continue_conv(x, weight, bias, rows, taps, outs, history=None):
    if rows.numel() == 0:
        return
    kd, vd, width = outs[0].shape[1], outs[2].shape[1], weight.shape[1]
    launcher = _build(kd, vd, width, bias is not None, history is not None)
    def matrix(t):
        return from_dlpack(t.detach()).mark_layout_dynamic(leading_dim=1)
    args = (matrix(x), matrix(weight),
            from_dlpack((weight.reshape(-1) if bias is None else bias).detach()).mark_layout_dynamic(),
            from_dlpack(rows).mark_layout_dynamic(), matrix(taps),
            *(matrix(t) for t in outs), matrix(x if history is None else history), current_cute_stream())
    compiled = get_compiled(launcher, args, key=(x.dtype, weight.dtype, outs[0].dtype, rows.dtype, taps.dtype))
    compiled(*args)
