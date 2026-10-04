"""Reusable full-dimension, split-half RoPE; no architecture/config dependencies.

Positions and theta belong to the caller. Tables may be gathered for arbitrary
positions before execute(). These pieces do not support interleaved/partial RoPE,
unequal Q/K head counts, or gradients with respect to the supplied tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Contract, Piece


BHSD = Contract("rope", "BHSD q,k -> BHSD q,k",
                "full split-half; equal Q/K shapes; shared [S,D] tables; CUDA bf16; Q/K gradients only")
QKV_BSHD = Contract("rope", "BS(3HD) qkv -> BSHD q,k",
                    "full split-half; equal Q/K heads; shared [S,D] tables; CUDA bf16; contiguous qkv",
                    autograd=False)


def prepare_tables(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    """Shared [1,S,D] tables; callers own storage, positions, lifetime and caching."""
    if head_dim <= 0 or head_dim % 2 or seq_len <= 0 or theta <= 0:
        raise UnsupportedTargetError("split-half RoPE needs a positive even dimension, length and theta")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:].float(), x[..., :half].float()), dim=-1)
    return (x.float() * cos.float() + rotated * sin.float()).to(x.dtype)


def reference(q, k, cos, sin):
    cos, sin = cos.reshape(-1, q.shape[-1])[None, None], sin.reshape(-1, q.shape[-1])[None, None]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def reference_qkv(qkv, h, d, cos, sin):
    q, k, _ = qkv.view(*qkv.shape[:-1], 3, h, d).unbind(-3)
    cos, sin = cos.reshape(-1, d)[None, :, None], sin.reshape(-1, d)[None, :, None]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _tables(cos, sin, s, d, device, dtype):
    if s <= 0 or d < 16 or d % 2 or cos.shape != sin.shape or cos.shape not in ((s, d), (1, s, d)):
        raise UnsupportedTargetError("RoPE requires shared [S,D] or [1,S,D] full split-half tables")
    if cos.device != device or sin.device != device or cos.dtype != dtype or sin.dtype != dtype:
        raise UnsupportedTargetError("RoPE tables must match Q/K device and dtype")
    if cos.requires_grad or sin.requires_grad:
        raise UnsupportedTargetError("RoPE table gradients are unsupported")


def check(q, k, cos, sin):
    if q.ndim != 4 or q.shape != k.shape or q.device != k.device or q.dtype != k.dtype:
        raise UnsupportedTargetError("RoPE requires equal-shape BHSD Q/K tensors")
    if q.device.type != "cuda" or q.dtype != torch.bfloat16:
        raise UnsupportedTargetError("this RoPE implementation requires CUDA bf16 inputs")
    _tables(cos, sin, q.shape[2], q.shape[3], q.device, q.dtype)


def check_qkv(qkv, h, d, cos, sin):
    from packed_encoders._kernels.rope import _bshd_applicable

    if qkv.ndim != 3 or qkv.shape[-1] != 3 * h * d or not qkv.is_contiguous():
        raise UnsupportedTargetError("QKV RoPE requires contiguous [B,S,3HD]")
    if qkv.device.type != "cuda" or qkv.dtype != torch.bfloat16 or not _bshd_applicable(h, d):
        raise UnsupportedTargetError("unsupported CUDA bf16 QKV RoPE head geometry")
    _tables(cos, sin, qkv.shape[1], d, qkv.device, qkv.dtype)


@dataclass(frozen=True)
class RoPEPiece(Piece):
    # Parameter/table preparation is shared across engines too, but stays outside
    # execution. No model fields or implicit position rebuilding are involved.
    prepare = staticmethod(prepare_tables)


def split_half_rope() -> RoPEPiece:
    from packed_encoders.ops import fused_apply_rope

    return RoPEPiece("split-half-rope-bhsd", BHSD, fused_apply_rope, reference, check)


def split_half_qkv_rope() -> RoPEPiece:
    from packed_encoders._kernels.rope import apply_rope_bshd

    return RoPEPiece("split-half-rope-qkv-bshd", QKV_BSHD, apply_rope_bshd, reference_qkv, check_qkv)
