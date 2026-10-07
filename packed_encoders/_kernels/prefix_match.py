"""Exact pairwise longest-common-prefix lengths for small packed GPU batches.

One warp compares a pair directly in packed storage, stopping at the first
mismatching warp-sized tile. No hashes, padded token matrix, or GPU-sized sort.
The kernel itself is capturable; the current forest builder still reads its
small B x B result on the host for FLA's host sequence metadata.
"""
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from packed_encoders._kernels._compile_cache import current_cute_stream, get_compiled


@cute.kernel
def _kernel(ids: cute.Tensor, cu: cute.Tensor, out: cute.Tensor):
    r, p, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()
    first = Int32(0)
    if p < r:
        start, other = cu[r], cu[p]
        n = cu[r + 1] - start
        m = cu[p + 1] - other
        limit = n if n < m else m
        first = Int32(limit)
        base = Int32(0)
        while base < limit and first == limit:
            col = base + lane
            found = Int32(limit)
            if col < limit:
                if ids[start + col] != ids[other + col]:
                    found = col
            for delta in (16, 8, 4, 2, 1):
                neighbor = cute.arch.shuffle_sync_bfly(found, offset=delta, mask=-1, mask_and_clamp=31)
                found = found if found < neighbor else neighbor
            first = found
            base += 32
    if lane == 0:
        out[r, p] = first


@cute.jit
def _launch(ids: cute.Tensor, cu: cute.Tensor, out: cute.Tensor, stream: cuda_driver.CUstream):
    _kernel(ids, cu, out).launch(grid=(cute.size(out, mode=[0]), cute.size(out, mode=[1]), 1),
                               block=(32, 1, 1), stream=stream)


def pairwise_prefix_lengths(ids, cu):
    n = cu.numel() - 1
    out = torch.empty((n, n), device=ids.device, dtype=torch.int32)
    args = (from_dlpack(ids).mark_layout_dynamic(), from_dlpack(cu).mark_layout_dynamic(),
            from_dlpack(out).mark_layout_dynamic(leading_dim=1), current_cute_stream())
    get_compiled(_launch, args, key=(ids.dtype, cu.dtype))(*args)
    return out
