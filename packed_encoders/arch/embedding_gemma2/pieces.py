"""EmbeddingGemma 2's numerical composition: post-norm residual pieces, the fused Q/K norm + RoPE,
and the varlen attention pieces (one prepared choice per attention kind)."""
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Piece


@dataclass(frozen=True)
class EmbeddingGemma2Pieces:
    linear: Piece
    rms_norm: Piece
    post_norm_add: Piece
    gelu_mul: Piece
    qk_norm_rope_pair: Piece
    attention_packed: Piece
    attention_padded: Piece

    def all(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def bind(self):
        from packed_encoders.pieces import hybrid as h
        from packed_encoders.pieces import post_norm as p
        from packed_encoders.pieces.numerical import LINEAR
        from packed_encoders.pieces.varlen import PACKED, PADDED

        contracts = (LINEAR, p.RMS_ROWS, p.POST_ADD, p.GELU_MUL, h.QK_PAIR, PACKED, PADDED)
        for slot, piece, required in zip(fields(self), self.all(), contracts):
            if piece.contract != required:
                raise UnsupportedTargetError(f"incompatible {slot.name} piece {piece.name}: expected {required}")
        return SimpleNamespace(**{f.name: getattr(self, f.name).execute for f in fields(self)})

    def validate(self, engine):
        import torch

        from packed_encoders.runtime.graphs import PaddedStatic
        from packed_encoders.runtime.staging import packed_layout_host

        gen = torch.Generator(device=engine.device).manual_seed(17)

        def rand(*shape):
            return torch.randn(shape, device=engine.device, dtype=engine.dtype, generator=gen)

        reports = {}

        def probe(slot, *args, **kwargs):
            result = getattr(self, slot).validate(*args, **kwargs)
            previous = reports.get(slot)
            if previous is not None and previous.max_abs_error is not None:
                result = replace(result, max_abs_error=max(previous.max_abs_error, result.max_abs_error))
            reports[slot] = result

        n, d = 17, engine.model_dim
        first = engine.layers[0]
        x = rand(n, d)
        probe("linear", x, first.gate_up)
        probe("rms_norm", x, first.w_in, engine.eps)
        probe("rms_norm", rand(n, len(engine.layers) * engine.ple_dim).view(n, -1, engine.ple_dim), engine.w_ple_norm,
              engine.eps, engine.ple_in_scale)
        probe("post_norm_add", rand(n, d), x, first.w_post_attn, engine.eps)
        probe("post_norm_add", rand(n, d), x, first.w_post_ple, engine.eps, first.scalar, first.w_in)
        g, u = rand(n, 2 * first.inter).split(first.inter, dim=-1)
        probe("gelu_mul", g, u)
        lengths = [5, 1, 11]
        cu = packed_layout_host(lengths)[0].to(engine.device, torch.int32)
        static = PaddedStatic(2, 9, torch.zeros(2, 9, dtype=torch.long, device=engine.device),
                              torch.tensor([0, 5, 9, 18, 18], dtype=torch.int32, device=engine.device),
                              torch.tensor([5, 9], dtype=torch.int32, device=engine.device))
        for kind, choice in engine.attention.items():
            L = next(layer for layer in engine.layers if layer.kind == kind)
            cos, sin = engine._rope(torch.arange(n, device=engine.device))[kind]
            qw = L.nq * L.hd
            proj = rand(n, qw + 2 * L.nkv * L.hd)
            probe("qk_norm_rope_pair", proj, 0, L.hd, L.nq, qw, L.hd, L.nkv, L.wqk, cos, sin, L.qk_eps)
            probe("rms_norm", proj[:, qw + L.nkv * L.hd:].view(n, L.nkv, L.hd), None, L.qk_eps)
            probe("attention_packed", choice, rand(n, L.nq, L.hd), rand(n, L.nkv, L.hd), rand(n, L.nkv, L.hd), cu, 11,
                  lengths, False)
            probe("attention_padded", choice, rand(18, L.nq, L.hd), rand(18, L.nkv, L.hd), rand(18, L.nkv, L.hd),
                  static, False)
        return reports


def default_pieces():
    from packed_encoders._kernels import hybrid as k
    from packed_encoders.pieces import hybrid as h
    from packed_encoders.pieces.numerical import linear
    from packed_encoders.pieces.post_norm import post_norm_pieces
    from packed_encoders.pieces.varlen import attention_pieces

    post = post_norm_pieces()
    packed, padded, _ = attention_pieces()
    return EmbeddingGemma2Pieces(
        linear=linear(dense=True), rms_norm=post["rms_norm"], post_norm_add=post["post_norm_add"],
        gelu_mul=post["gelu_mul"],
        qk_norm_rope_pair=Piece("triton-rms-partial-rope-pair", h.QK_PAIR, k.qk_norm_rope_pair, h.ref_rope_pair,
                                h.check_rope_pair),
        attention_packed=packed, attention_padded=padded)
