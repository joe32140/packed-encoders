"""Execution strategies over the same Qwen3.5 weights and layer operations.

The layout selects a stateless strategy before the layer loop. Ordinary execution
has no prefix state or branching; only PrefixExecution handles state continuation.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


class OrdinaryExecution:
    """Independent sequences, causal or noncausal full attention, eager or graphed."""

    @staticmethod
    def continue_conv(engine, layer, projection, outputs, layout):
        pass

    @staticmethod
    def delta_rule(engine, L, q, k, v, a, b, lay):
        kernel = engine.ops.gdn_recurrent if engine._use_recurrent(lay) else engine.ops.gdn_chunk
        return kernel(q, k, v, a, b, L.A_log, L.dt_bias, lay.cu, lay.cu_cpu)

    @staticmethod
    def attention(engine, q, k, v, lay):
        if lay.static is None:
            return engine.ops.attention_packed(engine.attention, q, k, v, lay.cu32, lay.max_len, lay.lengths, engine.causal)
        return engine.ops.attention_padded(engine.attention, q, k, v, lay.static, engine.causal)


class PrefixExecution:
    """Causal forests and persistent prefixes; reuses the backbone's layer primitives."""

    @staticmethod
    def continue_conv(engine, L, x, outs, lay):
        if lay.prefix is not None:
            PrefixExecution._prefix_history(engine, L, x, outs, lay)
        else:
            PrefixExecution._continue_conv(engine, L, x, outs, lay.share)

    @staticmethod
    def delta_rule(engine, L, q, k, v, a, b, lay) -> Tensor:
        if lay.prefix is not None:
            cache = lay.prefix.layers[lay.layer_index]
            initial = None if lay.prefix_write else cache.state.expand(len(lay.lengths), -1, -1, -1).contiguous()
            if not lay.prefix_write:
                return engine.ops.gdn_resume(q, k, v, a, b, L.A_log, L.dt_bias, initial, lay.cu, lay.cu_cpu,
                                           output_final_state=False)
            out, final = engine.ops.gdn_resume(q, k, v, a, b, L.A_log, L.dt_bias, initial, lay.cu, lay.cu_cpu)
            cache.state = final.clone()
            return out
        sh = lay.share
        # Roots from a zero state, then every child from its root's final state.
        R, run = sh.root_tokens, engine.ops.gdn_resume
        o_roots, state = run(q[:, :R], k[:, :R], v[:, :R], a[:, :R], b[:, :R], L.A_log, L.dt_bias, None,
                             sh.cu_roots, sh.cu_roots_cpu)
        o_kids = run(q[:, R:], k[:, R:], v[:, R:], a[:, R:], b[:, R:], L.A_log, L.dt_bias,
                     state.index_select(0, sh.parent), sh.cu_kids, sh.cu_kids_cpu, output_final_state=False)
        return torch.cat([o_roots, o_kids], 1)

    @staticmethod
    def _prefix_history(engine, L, x, outs, lay):
        cache = lay.prefix.layers[lay.layer_index]
        width, channels = L.conv_w.shape[1], L.bounds[-1]
        if lay.prefix_write:
            history = x[-min(width - 1, x.shape[0]):, :channels] if width > 1 else x[:0, :channels]
            cache.history = torch.cat([x.new_zeros((max(0, width - 1 - x.shape[0]), channels)), history]).clone()
            return
        meta = lay.prefix_meta
        if engine._prefix_conv is not None:
            engine._prefix_conv(x, L.conv_w, L.conv_b, meta.fix_rows, meta.fix_taps, outs, history=cache.history)
            return
        joined = torch.cat([cache.history, x[:, :channels]])
        taps = joined[meta.fix_taps + width - 1].float()
        y = torch.einsum("fwc,cw->fc", taps, L.conv_w.float())
        if L.conv_b is not None:
            y = y + L.conv_b.float()
        y = F.silu(y).to(x.dtype)
        for out, i, j in zip(outs, L.bounds, L.bounds[1:]):
            out.index_copy_(0, meta.fix_rows, y[:, i:j])

    @staticmethod
    def _continue_conv(engine, L, x: Tensor, outs: tuple[Tensor, ...], sh) -> None:
        """The packed conv restarts at each segment; a child's first width-1 tokens redo theirs over
        their root's last tokens (fp32, rounded once, as the conv kernels do), in place."""
        if engine._prefix_conv is not None:
            engine._prefix_conv(x, L.conv_w, L.conv_b, sh.fix_rows, sh.fix_taps, outs)
            return
        taps = x[:, : L.bounds[-1]].index_select(0, sh.fix_taps.view(-1)).view(*sh.fix_taps.shape, -1)
        y = torch.einsum("fwc,cw->fc", taps.float(), L.conv_w.float())
        if L.conv_b is not None:
            y = y + L.conv_b.float()
        y = F.silu(y).to(x.dtype)
        for out, i, j in zip(outs, L.bounds, L.bounds[1:]):
            out.index_copy_(0, sh.fix_rows, y[:, i:j])

    @staticmethod
    def attention(engine, q, k, v, lay):
        if lay.prefix is not None and not lay.prefix_write:
            cache, meta = lay.prefix.layers[lay.layer_index], lay.prefix_meta
            keys = torch.cat([cache.k, k]).index_select(0, meta.kv_idx)
            values = torch.cat([cache.v, v]).index_select(0, meta.kv_idx)
            o = engine.ops.attention_prefixed(engine.attention, q, keys, values, lay.cu32, meta.cu_k32,
                                              lay.max_len, max(meta.kv_lengths), lay.lengths, meta.kv_lengths, True)
        elif lay.share is not None:          # each child's keys: its root's prefix, then its own
            sh = lay.share
            o = engine.ops.attention_prefixed(engine.attention, q, k.index_select(0, sh.kv_idx), v.index_select(0, sh.kv_idx),
                                              lay.cu32, sh.cu_k32, lay.max_len, sh.max_k, lay.lengths, sh.kv_lengths,
                                              engine.causal)
        else:
            o = OrdinaryExecution.attention(engine, q, k, v, lay)
        if lay.prefix is not None and lay.prefix_write:
            cache = lay.prefix.layers[lay.layer_index]
            cache.k, cache.v = k.clone(), v.clone()
        return o


ORDINARY = OrdinaryExecution()
PREFIX = PrefixExecution()
