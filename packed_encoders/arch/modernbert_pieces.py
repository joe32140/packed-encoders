"""ModernBERT's explicitly selected operation pieces and bound execution calls.

Only the schedule and this selection are architecture-specific. Individual pieces
in packed_encoders.pieces take tensors/parameters, never a ModernBERT model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Callable

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces import Piece, PieceValidation, RoPEPiece


@dataclass(frozen=True)
class ModernBertOps:
    layer_norm: Callable
    add_layer_norm: Callable
    linear: Callable
    dense_linear: Callable
    rope: Callable
    rope_qkv: Callable
    attention: Callable
    attention_bshd: Callable
    geglu: Callable
    rope_tables: Callable


@dataclass(frozen=True)
class ModernBertPieces:
    layer_norm: Piece
    add_layer_norm: Piece
    linear: Piece
    dense_linear: Piece
    rope: RoPEPiece
    rope_qkv: RoPEPiece
    attention: Piece
    attention_bshd: Piece
    geglu: Piece

    def all(self) -> tuple[Piece, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def bind(self) -> ModernBertOps:
        """Check compatibility once and bind direct calls, with no evaluator layer."""
        from packed_encoders.pieces import attention, numerical, rope

        required = (numerical.LN, numerical.ADD_LN, numerical.LINEAR, numerical.LINEAR,
                    rope.BHSD, rope.QKV_BSHD, attention.BHSD, attention.BSHD, numerical.GEGLU)
        for slot, piece, contract in zip(fields(self), self.all(), required):
            if piece.contract != contract:
                raise UnsupportedTargetError(f"incompatible {slot.name} piece {piece.name}: expected {contract}")
        return ModernBertOps(*(p.execute for p in self.all()), rope_tables=self.rope.prepare)

    def validate(self, model):
        """Real-weight fixtures for each selected op; full-engine parity is separate."""
        import torch
        from packed_encoders import ops
        from packed_encoders.config import ModernBertParams

        params = ModernBertParams.from_hf_config(model.config)
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(17)

        def rand(*shape):
            return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

        layer = model.layers[-1]
        x = rand(1, 17, params.hidden_size)
        w = layer.mlp_norm.weight.detach().to(torch.bfloat16)
        reports = {}

        def probe(slot, *args, **kwargs):
            piece = getattr(self, slot)
            reports[slot] = piece.validate(*args, **kwargs)

        def skip(slot, reason):
            piece = getattr(self, slot)
            reports[slot] = PieceValidation(piece.name, None, piece.rtol, piece.atol, reason)

        probe("layer_norm", x, w, layer.mlp_norm.eps)
        probe("add_layer_norm", x, rand(*x.shape), w, layer.mlp_norm.eps)
        wi = layer.attn.Wqkv.weight.detach().to(torch.bfloat16)
        probe("linear", x, wi)
        probe("dense_linear", x, wi)
        probe("geglu", rand(1, 17, layer.mlp.Wi.weight.shape[0]))
        h, d = params.num_attention_heads, params.head_dim
        q, k, v = rand(1, h, 17, d), rand(1, h, 17, d), rand(1, h, 17, d)
        cos, sin = self.rope.prepare(17, d, params.global_rope_theta, device, q.dtype)
        probe("rope", q, k, cos, sin)
        probe("attention", q, k, v, mask=None, window=(-1, -1), scaling=params.scaling, backend="sdpa")
        if ops._bshd_applicable(h, d):
            qkv = rand(1, 17, 3 * h * d)
            probe("rope_qkv", qkv, h, d, cos, sin)
            try:
                ops._load_flash_attn()
                backend = "flash"
            except ImportError:
                # Packed short attention is available only under its existing
                # invariants; don't import another engine's toolchain to probe it.
                from packed_encoders.forward import _packed_short_invariants
                backend = "triton" if _packed_short_invariants(
                    params, device=device, dtype=q.dtype, max_seqlen=17,
                    cu_seqlens=torch.tensor([0, 17], dtype=torch.int32, device=device),
                ) else None
            if backend is not None:
                cu = torch.tensor([0, 7, 17], dtype=torch.int32, device=device)
                probe("attention_bshd", q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                      v.transpose(1, 2), window=(params.sliding_half_window,) * 2,
                      scaling=params.scaling, backend=backend, cu_seqlens=cu, max_seqlen=10)
            else:
                skip("attention_bshd", "no supported BSHD attention backend in this environment")
        else:
            skip("rope_qkv", "head geometry uses the BHSD path")
            skip("attention_bshd", "head geometry uses the BHSD path")
        return reports


@lru_cache(maxsize=1)
def default_pieces() -> ModernBertPieces:
    from packed_encoders.pieces.numerical import layer_norm, add_layer_norm, linear, geglu
    from packed_encoders.pieces.rope import split_half_rope, split_half_qkv_rope
    from packed_encoders.pieces.attention import attention, attention_bshd

    return ModernBertPieces(layer_norm(), add_layer_norm(), linear(), linear(dense=True),
                            split_half_rope(), split_half_qkv_rope(), attention(), attention_bshd(), geglu())


@lru_cache(maxsize=1)
def default_execution() -> ModernBertOps:
    return default_pieces().bind()
