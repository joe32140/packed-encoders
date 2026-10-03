"""Noncausal attention pieces; sequence/window policy remains engine-owned."""

import torch
import torch.nn.functional as F

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece

BHSD = Contract("attention", "BHSD q,k,v -> BS(HD)",
                "noncausal; no dropout; SDPA additive mask or Flash local window; packed int32 boundaries")
BSHD = Contract("attention", "BSHD q,k,v -> BS(HD)",
                "noncausal; no dropout; Flash/Triton local window; packed int32 boundaries", autograd=False)


def _dense(q, k, v, mask, window, scaling):
    if window != (-1, -1):
        pos = torch.arange(q.shape[-2], device=q.device)
        delta = pos[:, None] - pos[None, :]
        band = torch.zeros_like(delta, dtype=torch.float32).masked_fill(
            (delta > window[0]) | (delta < -window[1]), float("-inf"))
        mask = band if mask is None else mask.float() + band
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         attn_mask=None if mask is None else mask.float(),
                                         scale=scaling).to(q.dtype)


def reference(q, k, v, *, mask, window, scaling, backend, cu_seqlens=None, max_seqlen=None):
    if backend not in ("flash", "triton"):
        out = _dense(q, k, v, mask, (-1, -1), scaling)
    elif cu_seqlens is None:
        out = _dense(q, k, v, None, window, scaling)
    else:
        # An intentionally simple, independent oracle; CPU reads are validation-only.
        bounds = cu_seqlens.tolist()
        out = torch.zeros_like(q)
        for a, b in zip(bounds, bounds[1:]):
            if b > a:
                out[:, :, a:b] = _dense(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], None, window, scaling)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def reference_bshd(q, k, v, *, window, scaling, backend="flash", cu_seqlens=None, max_seqlen=None):
    return reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask=None,
                     window=window, scaling=scaling, backend=backend,
                     cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def check(q, k, v, *, mask, window, scaling, backend, cu_seqlens=None, max_seqlen=None):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise UnsupportedTargetError("attention piece requires equal Q/K/V shapes")
    if backend not in ("sdpa", "auto", "flash", "triton") or scaling <= 0:
        raise UnsupportedTargetError("unsupported attention backend or scale")
    if cu_seqlens is not None and (q.shape[0] != 1 or cu_seqlens.dtype != torch.int32 or max_seqlen is None):
        raise UnsupportedTargetError("packed attention requires B=1, int32 boundaries and max_seqlen")


def check_bshd(q, k, v, *, window, scaling, backend="flash", cu_seqlens=None, max_seqlen=None):
    check(q, k, v, mask=None, window=window, scaling=scaling, backend=backend,
          cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    if backend not in ("flash", "triton"):
        raise UnsupportedTargetError("BSHD attention requires Flash or Triton")


def attention() -> Piece:
    from packed_encoders.ops import attention as execute

    return Piece("attention-bhsd", BHSD, execute, reference, check)


def attention_bshd() -> Piece:
    from packed_encoders.ops import attention_bshd as execute

    return Piece("attention-bshd", BSHD, execute, reference_bshd, check_bshd)
