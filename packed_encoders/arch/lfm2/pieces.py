"""LFM2 and LFM3's numerical composition: the shared hybrid pieces, the gated short convolution, and
the post-norm residual add of LFM3's sandwich norms."""
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

from packed_encoders.errors import UnsupportedTargetError
from packed_encoders.pieces.base import Piece, PieceValidation


@dataclass(frozen=True)
class Lfm2Pieces:
    linear: Piece
    rms_norm: Piece
    add_rms_norm: Piece
    swiglu: Piece
    short_conv: Piece
    qk_norm_rope_pair: Piece
    attention_packed: Piece
    attention_padded: Piece
    attention_prefixed: Piece
    post_norm_add: Piece

    def all(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def bind(self):
        from packed_encoders.pieces import hybrid as h
        from packed_encoders.pieces.numerical import LINEAR
        from packed_encoders.pieces.post_norm import POST_ADD
        from packed_encoders.pieces.varlen import PACKED, PADDED, PREFIXED

        contracts = (LINEAR, h.RMS, h.ADD_RMS, h.SWIGLU, h.SHORT_CONV, h.QK_PAIR, PACKED, PADDED, PREFIXED, POST_ADD)
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

        def skip(slot, reason):
            p = getattr(self, slot)
            reports[slot] = PieceValidation(p.name, None, p.rtol, p.atol, reason)

        n, d = 17, engine.hidden_size
        first = engine.layers[0]
        x = rand(n, d)
        probe("linear", x, first.w13)
        probe("rms_norm", x, first.w_op, engine.eps)
        probe("add_rms_norm", x, rand(n, d), first.w_op, engine.eps)
        probe("swiglu", rand(n, first.inter), rand(n, first.inter))
        if engine.sandwich:
            probe("post_norm_add", rand(n, d), rand(n, d), first.w_post_op, engine.eps, None, first.w_ffn)
        else:
            skip("post_norm_add", "no sandwich norms")
        lengths = [5, 1, 11]
        cu_cpu, pos = packed_layout_host(lengths)
        conv = next((l for l in engine.layers if not l.attn), None)
        if conv is not None:
            probe("short_conv", rand(n, 3 * d), conv.conv_w, pos.to(engine.device))
            # the rectangular layout of CUDA graphs: positions restart per row
            probe("short_conv", rand(18, 3 * d), conv.conv_w, torch.arange(9, device=engine.device).repeat(2))
        else:
            skip("short_conv", "no convolution layer")
        attn = next((l for l in engine.layers if l.attn), None)
        if attn is not None:
            static = PaddedStatic(2, 9, torch.zeros(2, 9, dtype=torch.long, device=engine.device),
                                  torch.tensor([0, 5, 9, 18, 18], dtype=torch.int32, device=engine.device),
                                  torch.tensor([5, 9], dtype=torch.int32, device=engine.device))
            lq, lk = [5, 1, 11], [9, 1, 30]
            cq, ck = (packed_layout_host(x)[0].to(engine.device, torch.int32) for x in (lq, lk))
            rope = engine._rope(torch.arange(n, device=engine.device))
            for kind, attention in engine.attentions.items():        # each attention kind: its window and table
                attn = next(l for l in engine.layers if l.kind == kind)
                cos, sin = rope[kind]
                qw = attn.nq * attn.hd
                probe("qk_norm_rope_pair", rand(n, qw + 2 * attn.nkv * attn.hd), 0, attn.hd, attn.nq, qw, attn.hd,
                      attn.nkv, attn.wqk, cos, sin, attn.qk_eps)
                q, k, v = rand(n, attn.nq, attn.hd), rand(n, attn.nkv, attn.hd), rand(n, attn.nkv, attn.hd)
                probe("attention_packed", attention, q, k, v, cu_cpu.to(engine.device, torch.int32), 11, lengths,
                      engine.causal)
                probe("attention_padded", attention, rand(18, attn.nq, attn.hd),
                      rand(18, attn.nkv, attn.hd), rand(18, attn.nkv, attn.hd), static, engine.causal)
                modes = []                        # sharing reads it causally, grouped rows both ways (full only)
                if engine.share_rejected is None:
                    modes.append(engine.causal)
                if kind == "full_attention" and engine.grouped_rejected is None:
                    modes.append(False)
                for causal in modes:
                    try:
                        probe("attention_prefixed", attention, rand(17, attn.nq, attn.hd),
                              rand(40, attn.nkv, attn.hd), rand(40, attn.nkv, attn.hd), cq, ck, 11, 30, lq, lk,
                              causal)
                    except Exception as exc:      # both are optional: turn the mode off rather than fail the pack
                        reason = f"attention_prefixed validation failed: {type(exc).__name__}: {str(exc)[:200]}"
                        if causal:
                            engine.share_rejected = reason
                            engine.min_shared_prefix = 0
                        else:
                            engine.grouped_rejected = reason
            if "attention_prefixed" not in reports:
                skip("attention_prefixed", f"no shared prefixes: {engine.share_rejected}; "
                                           f"no grouped rows: {engine.grouped_rejected}")
        else:
            for slot in ("qk_norm_rope_pair", "attention_packed", "attention_padded", "attention_prefixed"):
                skip(slot, "no softmax attention layer")
        return reports


def default_pieces():
    from packed_encoders._kernels.inference_norm import rms_norm, add_rms_norm
    from packed_encoders.pieces.hybrid import RMS, ADD_RMS, ref_rms, ref_add_rms, check_rms, check_add_rms
    from packed_encoders.pieces.hybrid import default_numerical, short_conv_piece
    from packed_encoders.pieces.post_norm import post_norm_pieces
    from packed_encoders.pieces.varlen import attention_pieces

    numerical = default_numerical()
    packed, padded, prefixed = attention_pieces()
    return Lfm2Pieces(linear=numerical["linear"],
                      rms_norm=Piece("triton-inference-rms-norm", RMS, rms_norm, ref_rms, check_rms),
                      add_rms_norm=Piece("triton-inference-add-rms-norm", ADD_RMS, add_rms_norm, ref_add_rms, check_add_rms),
                      swiglu=numerical["swiglu"],
                      short_conv=short_conv_piece(), qk_norm_rope_pair=numerical["qk_norm_rope_pair"],
                      attention_packed=packed, attention_padded=padded, attention_prefixed=prefixed,
                      post_norm_add=post_norm_pieces()["post_norm_add"])
