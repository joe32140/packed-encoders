"""Exact fast forward for an LFM2 or LFM3 text backbone (gated short convolutions + GQA softmax attention).

Same math as the model, fewer and fatter kernels:

- merged projections: attention q|k|v (and LFM3's output gate) and MLP w1|w3, one GEMM each. As in
  the Qwen3.5 engine the merged weight *is* the HF weight (each Linear is re-pointed at its rows;
  `release()` separates them).
- the gated short conv `C * conv(B * x)` in one Triton launch reading the in-projection in place,
  positions restarting per sequence (`pieces.hybrid.SHORT_CONV`), so packed rows never mix.
- q/k RMSNorm + RoPE in one launch reading the projection in place; varlen attention chosen
  by probe (runtime.attention), one choice per attention kind; inference residual-add + RMSNorm
  and a Triton SwiGLU.

LFM3 is LFM2 with three more things, each read from the module rather than assumed:
`sliding_attention` layers (a local causal window and a RoPE table of their own), a sigmoid
output gate per attention head (its rows ride the q|k|v GEMM), and sandwich norms (each
sublayer's output is normalized before the residual add: one fused post-norm add, which also
returns the next sublayer's norm, pieces.post_norm). Its full-attention RoPE leaves a zero tail of
frequencies, which a full-width table carries unchanged. The conv width is read from the weights.

LFM2's RMSNorm scales by `weight` (no `1 +`); the fused norms take fp32 copies of the weights,
tracked by `sync_norms` like the Qwen3.5 engine's.

Four entry points share the layer code: `forward_packed` (eager, padding-free, token ids or
input embeddings), `padded_core` (the CUDA-graph region, right-padded `(rows, S)`; exact
because the conv and the attention are causal — see runtime.graphs), `forward_shared`
(opt in: rows that start with the same tokens run that prefix once — see runtime.sharing;
`prepare_shared_layout` + `shared_core` are its graph-capturable halves, runtime.shared_graphs)
and `forward_grouped` (rows laid out `[context][group 1]..[group N]`, read with grouped attention
on the full-attention layers; see `grouped_layout_host`).
With no recurrent state to resume, the whole forest is one pass: a child's first conv taps are
recomputed over its root's last tokens, and its attention reads the root's keys and values
through the prefixed varlen kernel.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import require_independent_parameters, require_plain, share_rows, unshare_rows
from packed_encoders.errors import UnsupportedTargetError, ValidationError
from packed_encoders.runtime.attention import AttentionChoice, select_attention
from packed_encoders.runtime.graphs import PaddedStatic
from packed_encoders.runtime.sharing import SharedPlan, plan_shared_prefixes
from packed_encoders.runtime.staging import PinnedStager, packed_layout_host

KERNEL_TOLERANCE = 2e-2        # max |fused - model's module|, relative to the module's max |output| (bf16: a few ulp)
# Shared prefixes are opt in (`min_shared_prefix` starts at 0). validate() checks the shared pass with
# prefixes of at least this many tokens when the caller hasn't set a threshold.
SHARED_PREFIX_MIN = 64
ATTENTION_KINDS = ("full_attention", "sliding_attention")


class _Layer:
    pass


@dataclass
class _Layout:
    pos: Tensor                        # (T,) position of each token in its segment (the conv restarts there)
    rope: dict[str, tuple[Tensor, Tensor]] | None = None   # per attention kind: cos, sin at each token's row position
    cu32: Tensor | None = None         # packed: int32 cu_seqlens (attention)
    max_len: int = 0
    lengths: list[int] | None = None
    static: PaddedStatic | None = None
    share: _Shared | None = None       # forward_shared: segments are roots, then children
    grouped: _Grouped | None = None    # forward_grouped: the full-attention segments


@dataclass
class _Shared:
    """Device side of a SharedPlan (runtime.sharing)."""

    fix_rows: Tensor                   # child tokens whose conv taps reach back into the root
    fix_taps: Tensor                   # [F, K] rows under those taps, oldest first
    kv_idx: Tensor                     # rows forming each segment's keys and values
    cu_k32: Tensor
    max_k: int
    kv_lengths: list[int]


@dataclass
class _Grouped:
    """Device side of `grouped_layout_host`: query segments in token order, keys gathered per segment."""

    cu_q32: Tensor
    max_q: int
    lengths_q: list[int]
    kv_idx: Tensor
    cu_k32: Tensor
    max_k: int
    lengths_k: list[int]


class Lfm2Engine:
    def __init__(self, text_model: nn.Module, *, causal: bool = True, attention_order: Sequence[str] | None = None,
                 pieces=None, family: str = "LFM2"):
        from packed_encoders.arch.lfm2.pieces import default_pieces

        self.family = family
        if not causal or not getattr(text_model.config, "is_causal", True):
            # A bidirectional LFM2 also changes the convolution (centred taps), not just the mask.
            raise UnsupportedTargetError(f"the {family} engine runs causal backbones only")
        self.composition = pieces or default_pieces()
        self.ops = self.composition.bind()
        require_independent_parameters(text_model, family)
        cfg = text_model.config
        self.tm, self.cfg, self.causal = text_model, cfg, causal
        self.embed = text_model.embed_tokens.weight
        self.device, self.dtype = self.embed.device, self.embed.dtype
        if self.device.type != "cuda":
            raise UnsupportedTargetError(f"the {family} engine needs CUDA weights; model is on {self.device}")
        self.hidden_size = cfg.hidden_size
        self.eps = cfg.norm_eps
        self.sandwich = bool(getattr(cfg, "use_sandwich_norm", False))
        self._norms: list[list] = []            # [parameter (detached), fp32 copy, version seen]; see sync_norms
        self.shared: list[nn.Linear] = []
        self.layers: list[_Layer] = []
        self.attentions: dict[str, AttentionChoice] = {}
        n = cfg.num_hidden_layers
        kinds = list(getattr(cfg, "layer_types", None) or [])
        if len(kinds) < n:
            raise UnsupportedTargetError("config.layer_types is missing or shorter than num_hidden_layers")
        for layer, kind in zip(text_model.layers[:n], kinds[:n]):
            _require_supported(layer, kind, self.sandwich)          # every layer before any re-pointing
        # LFM3's rotary embedding keeps one table per attention kind; LFM2's has one for all
        self._rope_kinds = getattr(text_model.rotary_emb, "layer_types", None)
        missing = {k for k in kinds[:n] if k != "conv"} - set(self._rope_kinds or ATTENTION_KINDS)
        if missing:
            raise UnsupportedTargetError(f"no RoPE table for {sorted(missing)} layers")
        try:
            with torch.cuda.device(self.device):
                self.w_final = self._scale(text_model.embedding_norm.weight)
                for layer, kind in zip(text_model.layers[:n], kinds[:n]):
                    self.layers.append(self._prepare(layer, kind))
                self.kernel_errors: dict[str, float] = {}
                firsts = [next((L for L in self.layers if L.kind == kind), None) for kind in ATTENTION_KINDS]
                for kind, attn in zip(ATTENTION_KINDS, firsts):
                    if attn is not None:
                        self.attentions[kind] = select_attention(
                            n_heads=attn.nq, n_kv_heads=attn.nkv, head_dim=attn.hd, causal=causal,
                            device=self.device, dtype=self.dtype, order=attention_order, window=attn.window,
                            bidirectional_prefixed=kind == "full_attention")
                for kind, attn in zip(ATTENTION_KINDS, firsts):
                    if attn is not None:
                        key = "qk_norm_rope" if kind == "full_attention" else f"qk_norm_rope:{kind}"
                        self.kernel_errors[key] = self._check_qk(attn)
                self.attention: AttentionChoice | None = next(iter(self.attentions.values()), None)
                conv = next((L for L in self.layers if not L.attn), None)
                if conv is not None:
                    self.kernel_errors["short_conv"] = self._check_conv(conv)
                self.conv_width = conv.conv_w.shape[1] if conv is not None else 1
                self.min_shared_prefix = 0                          # off until the caller opts in
                self.share_rejected: str | None = None              # None: forward_shared can run here
                if any(a.prefixed is None for a in self.attentions.values()):
                    self.share_rejected = "no prefixed attention kernel"
                full = self.attentions.get("full_attention")
                self.grouped_rejected: str | None = None            # None: forward_grouped can run here
                if full is not None and full.prefixed is None:
                    self.grouped_rejected = "no prefixed attention kernel"
                self._stager = PinnedStager(self.device)
        except BaseException:
            self.release(rollback=True)                          # leave the model exactly as we found it
            raise

    def release(self, *, rollback: bool = False) -> None:
        """Drop the merged weights, then give every projection its own storage back."""
        self.layers.clear()
        self._norms.clear()
        self.w_final = None
        unshare_rows(self.shared, rollback=rollback)

    # ------------------------------------------------------------------ weights
    def _prepare(self, layer: nn.Module, kind: str) -> _Layer:
        L = _Layer()
        L.kind, L.attn = kind, kind != "conv"
        L.w_op = self._scale(layer.operator_norm.weight)
        L.w_ffn = self._scale(layer.ffn_norm.weight)
        if self.sandwich:
            L.w_post_op = self._scale(layer.post_operator_norm.weight)
            L.w_post_ffn = self._scale(layer.post_ffn_norm.weight)
        mlp = layer.feed_forward
        L.w13 = share_rows([mlp.w1, mlp.w3], self.shared)
        L.inter = mlp.w1.weight.shape[0]
        L.w2 = mlp.w2.weight
        if L.attn:
            sa = layer.self_attn
            L.module = sa
            L.hd = sa.head_dim
            L.nq, L.nkv = sa.q_proj.weight.shape[0] // L.hd, sa.k_proj.weight.shape[0] // L.hd
            gate = getattr(sa, "gate_proj", None)
            L.gated = gate is not None
            # 8-row aligned, so the v and gate views keep 16-byte row strides (the flash kernels need them)
            L.qkv = share_rows([sa.q_proj, sa.k_proj, sa.v_proj] + ([gate] if L.gated else []), self.shared, align=8)
            q_norm, k_norm = _qk_norms(sa)
            L.wqk = torch.empty((2, L.hd), device=self.device, dtype=torch.float32)   # q row, k row
            self._scale(q_norm.weight, out=L.wqk[0])
            self._scale(k_norm.weight, out=L.wqk[1])
            L.qk_eps = q_norm.variance_epsilon
            L.out = _out_proj(sa).weight
            # HF's sliding mask keeps the keys fewer than `sliding_window` positions back
            L.window = (int(sa.sliding_window) - 1, 0) if kind == "sliding_attention" else (-1, -1)
        else:
            sc = layer.conv
            L.module = sc
            L.in_proj, L.out = sc.in_proj.weight, sc.out_proj.weight
            L.conv_w = sc.conv.weight.detach().squeeze(1)          # (D, K) taps, oldest first; the parameter's storage
            if not L.conv_w.is_contiguous():
                raise UnsupportedTargetError(f"{self.family} convolution weights must be contiguous")
        return L

    def _scale(self, weight: Tensor, out: Tensor | None = None) -> Tensor:
        """fp32 copy of a norm weight for the fused norm kernels; tracked (sync_norms)."""
        src = weight.detach()                    # shares the parameter's storage and version counter
        if out is None:
            out = torch.empty(src.shape, device=src.device, dtype=torch.float32)
        out.copy_(src)
        self._norms.append([src, out, src._version])
        return out

    def sync_norms(self) -> None:
        """Re-copy every norm weight whose parameter changed since it was copied (an optimizer step,
        `load_state_dict`). In place, so captured graphs stay valid. Writes through `.data` bypass the
        version counter: re-pack after those."""
        for entry in self._norms:
            src, out, seen = entry
            if src._version != seen:
                out.copy_(src)
                entry[2] = src._version

    def _check_qk(self, L: _Layer) -> float:
        """The fused q/k norm + RoPE against the model's own RMSNorms + apply_rotary_pos_emb."""
        apply_rotary_pos_emb = sys.modules[type(L.module).__module__].apply_rotary_pos_emb
        q_norm, k_norm = _qk_norms(L.module)
        g = torch.Generator(device=self.device).manual_seed(0)
        T = 37
        qw = L.nq * L.hd
        proj = torch.randn(T, qw + 2 * L.nkv * L.hd, device=self.device, dtype=self.dtype, generator=g)
        cos, sin = self._rope(torch.arange(T, device=self.device) * 7)[L.kind]
        with torch.no_grad():
            qx, kx = proj[:, :qw].view(T, L.nq, L.hd), proj[:, qw: qw + L.nkv * L.hd].view(T, L.nkv, L.hd)
            rq, rk = apply_rotary_pos_emb(q_norm(qx), k_norm(kx), cos, sin, unsqueeze_dim=1)
            pq, pk = self.ops.qk_norm_rope_pair(proj, 0, L.hd, L.nq, qw, L.hd, L.nkv, L.wqk, cos, sin, L.qk_eps)
            err = max(_rel(pq, rq), _rel(pk, rk))
        if not err < KERNEL_TOLERANCE:
            raise ValidationError(f"fused q/k norm + RoPE mismatch vs the model's modules ({L.kind}): "
                                  f"relative error {err:.2e}")
        return err

    def _check_conv(self, L: _Layer) -> float:
        """The packed short conv against the model's own module, run per sequence."""
        g = torch.Generator(device=self.device).manual_seed(0)
        lengths = [5, 20, 1, 12]
        h = torch.randn(sum(lengths), self.hidden_size, device=self.device, dtype=self.dtype, generator=g)
        _, pos = packed_layout_host(lengths)
        with torch.no_grad():
            got = self._conv(L, h, _Layout(pos=pos.to(self.device)))
            ref = torch.cat([L.module(x[None])[0] for x in h.split(lengths)])
        err = _rel(got, ref)
        if not err < KERNEL_TOLERANCE:
            raise ValidationError(f"packed short convolution mismatch vs the model's module: relative error {err:.2e}")
        return err

    def _rope(self, pos: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
        """cos, sin at `pos` for each attention kind."""
        probe = torch.empty(1, device=self.device, dtype=self.dtype)
        if self._rope_kinds is None:
            cos, sin = self.tm.rotary_emb(probe, pos[None])
            return dict.fromkeys(self.attentions, (cos[0], sin[0]))
        tables = {}
        for kind in self.attentions:
            cos, sin = self.tm.rotary_emb(probe, pos[None], kind)
            tables[kind] = (cos[0], sin[0])
        return tables

    # ------------------------------------------------------------------ layers
    def _conv(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        proj = self.ops.linear(h, L.in_proj)
        y = self.ops.short_conv(proj, L.conv_w, lay.pos)
        if lay.share is not None:
            self._continue_conv(L, proj, y, lay.share)
        return self.ops.linear(y, L.out)

    def _continue_conv(self, L: _Layer, proj: Tensor, y: Tensor, sh: _Shared) -> None:
        """The packed conv restarts at each segment; a child's first K-1 tokens redo theirs over
        their root's last tokens, rounding where the kernel does (B * x, the conv, the gate), in place."""
        from packed_encoders._kernels.short_conv import continue_conv

        continue_conv(proj, L.conv_w, y, sh.fix_rows, sh.fix_taps)

    def _attn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        N, nq, nkv, hd = h.shape[0], L.nq, L.nkv, L.hd
        out = self.ops.linear(h, L.qkv)
        qw, kvw = nq * hd, nkv * hd
        cos, sin = lay.rope[L.kind]
        q, k = self.ops.qk_norm_rope_pair(out, 0, hd, nq, qw, hd, nkv, L.wqk, cos, sin, L.qk_eps)
        v = out[:, qw + kvw: qw + 2 * kvw].view(N, nkv, hd)
        attention = self.attentions[L.kind]
        if lay.grouped is not None and L.kind == "full_attention":    # each segment reads its keys both ways
            gr = lay.grouped
            o = self.ops.attention_prefixed(attention, q, k.index_select(0, gr.kv_idx), v.index_select(0, gr.kv_idx),
                                            gr.cu_q32, gr.cu_k32, gr.max_q, gr.max_k, gr.lengths_q, gr.lengths_k,
                                            False)
        elif lay.share is not None:          # each child's keys: its root's prefix, then its own
            sh = lay.share
            o = self.ops.attention_prefixed(attention, q, k.index_select(0, sh.kv_idx), v.index_select(0, sh.kv_idx),
                                            lay.cu32, sh.cu_k32, lay.max_len, sh.max_k, lay.lengths, sh.kv_lengths,
                                            self.causal)
        elif lay.static is None:
            o = self.ops.attention_packed(attention, q, k, v, lay.cu32, lay.max_len, lay.lengths, self.causal)
        else:
            o = self.ops.attention_padded(attention, q, k, v, lay.static, self.causal)
        if L.gated:                          # times sigmoid(gate) per head, each rounded where the model rounds
            o = o.view(N, nq, hd) * torch.sigmoid(out[:, qw + 2 * kvw: qw + 2 * kvw + nq])[:, :, None]
        return self.ops.linear(o.reshape(N, qw), L.out)

    def _mix(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        return self._attn(L, h, lay) if L.attn else self._conv(L, h, lay)

    def _trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        if self.sandwich:
            return self._sandwich_trunk(x, lay)
        residual, delta, eps = x, None, self.eps
        for L in self.layers:
            if delta is None:
                h = self.ops.rms_norm(residual, L.w_op, eps)
            else:
                h, residual = self.ops.add_rms_norm(delta, residual, L.w_op, eps)
            h, residual = self.ops.add_rms_norm(self._mix(L, h, lay), residual, L.w_ffn, eps)
            g, u = self.ops.linear(h, L.w13).split(L.inter, dim=-1)
            delta = self.ops.linear(self.ops.swiglu(g, u), L.w2)
        out, _ = self.ops.add_rms_norm(delta, residual, self.w_final, eps)
        return out

    def _sandwich_trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        """Each sublayer's output normalized and added to the residual stream, and the sum normalized
        for the next sublayer (after the last, the final norm): one launch per sublayer."""
        residual, eps = x.contiguous(), self.eps
        h = self.ops.rms_norm(residual, self.layers[0].w_op, eps)
        for i, L in enumerate(self.layers):
            residual, h = self.ops.post_norm_add(self._mix(L, h, lay), residual, L.w_post_op, eps, None, L.w_ffn)
            g, u = self.ops.linear(h, L.w13).split(L.inter, dim=-1)
            w_next = self.layers[i + 1].w_op if i + 1 < len(self.layers) else self.w_final
            residual, h = self.ops.post_norm_add(self.ops.linear(self.ops.swiglu(g, u), L.w2), residual,
                                                 L.w_post_ffn, eps, None, w_next)
        return h

    # ------------------------------------------------------------------ entry points
    @torch.no_grad()
    def forward_packed(self, ids: Tensor | None, lengths: Sequence[int], *, embeds: Tensor | None = None) -> Tensor:
        """Eager, padding-free. ids: (T,) device ids back to back, or embeds: (T, hidden) input
        embeddings in their place; lengths: host ints (all > 0). Returns the final-normed hidden
        states (T, hidden)."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            cu, pos = packed_layout_host(lengths)
            d_pos, d_cu = self._stager.put([pos, cu])
            lay = _Layout(rope=self._rope(d_pos), pos=d_pos, cu32=d_cu.to(torch.int32), max_len=int(max(lengths)),
                          lengths=list(lengths))
            return self._trunk(self._embed(ids, embeds), lay)

    @torch.no_grad()
    def forward_grouped(self, ids: Tensor | None, lengths: Sequence[int], groups: Sequence[Sequence[int]], *,
                        embeds: Tensor | None = None) -> Tensor:
        """`forward_packed` for rows laid out `[context][group 1]..[group N]` (`grouped_layout_host`),
        `groups[i]` row i's group lengths in order (host ints). Sliding-window and convolution layers
        stay causal over each row."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            cu, pos = packed_layout_host(lengths)
            lengths_q, lengths_k, kv_idx = grouped_layout_host(lengths, groups)
            cu_q, _ = packed_layout_host(lengths_q)
            cu_k, _ = packed_layout_host(lengths_k)
            d_pos, d_cu, d_cu_q, d_cu_k, d_kv = self._stager.put([pos, cu, cu_q, cu_k, kv_idx])
            grouped = _Grouped(cu_q32=d_cu_q.to(torch.int32), max_q=max(lengths_q), lengths_q=lengths_q, kv_idx=d_kv,
                               cu_k32=d_cu_k.to(torch.int32), max_k=max(lengths_k), lengths_k=lengths_k)
            lay = _Layout(rope=self._rope(d_pos), pos=d_pos, cu32=d_cu.to(torch.int32), max_len=int(max(lengths)),
                          lengths=list(lengths), grouped=grouped)
            return self._trunk(self._embed(ids, embeds), lay)

    def _embed(self, ids: Tensor | None, embeds: Tensor | None) -> Tensor:
        return F.embedding(ids.reshape(-1), self.embed) if embeds is None else embeds.to(self.dtype)

    def plan_sharing(self, ids: Tensor | None, lengths: Sequence[int], *, embeds: Tensor | None = None) -> SharedPlan | None:
        """A plan for `forward_shared` when rows share at least `min_shared_prefix` leading tokens.
        Input embeddings are matched row for row: equal rows are the same token (or the same image
        feature), so a wrapper that merges images into the embeddings shares those too."""
        if not self.min_shared_prefix or len(lengths) < 2:
            return None
        if ids is None:
            ids = _row_hashes(embeds)
        plan = plan_shared_prefixes(ids.reshape(-1), lengths, min_prefix=self.min_shared_prefix,
                                    conv_width=self.conv_width)
        if plan is not None and embeds is not None:
            # Hashes can collide: every token must take the forward row of an identical embedding.
            d_src, d_out = self._stager.put([plan.src, plan.out])
            if not torch.equal(embeds.index_select(0, d_src.index_select(0, d_out)), embeds):
                return None
        return plan

    def prepare_shared_layout(self, plan: SharedPlan) -> SimpleNamespace:
        """Stage one forest outside capture (runtime.shared_graphs); host lengths fix the launch geometry."""
        cu, _ = packed_layout_host(plan.lengths)
        cu_k, _ = packed_layout_host(plan.kv_lengths)
        src, out, rope, pos, d_cu, d_cu_k, fix_rows, fix_taps, kv_idx = self._stager.put(
            [plan.src, plan.out, plan.rope_pos, plan.conv_pos, cu, cu_k, plan.fix_rows, plan.fix_taps, plan.kv_idx])
        share = _Shared(fix_rows=fix_rows, fix_taps=fix_taps.view(plan.fix_taps.shape), kv_idx=kv_idx,
                        cu_k32=d_cu_k.to(torch.int32), max_k=max(plan.kv_lengths), kv_lengths=plan.kv_lengths)
        lay = _Layout(rope=self._rope(rope), pos=pos, cu32=d_cu.to(torch.int32), max_len=max(plan.lengths),
                      lengths=plan.lengths, share=share)
        return SimpleNamespace(src=src, out=out, layout=lay)

    def shared_core(self, ids: Tensor | None, static: SimpleNamespace, *, embeds: Tensor | None = None) -> Tensor:
        if embeds is None:
            x = F.embedding(ids.reshape(-1).index_select(0, static.src), self.embed)
        else:
            x = embeds.to(self.dtype).index_select(0, static.src)
        return self._trunk(x, static.layout).index_select(0, static.out)

    @torch.no_grad()
    def forward_shared(self, ids: Tensor | None, plan: SharedPlan, *, embeds: Tensor | None = None) -> Tensor:
        """Run each shared prefix once; hidden states (T, hidden) in the caller's layout."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            return self.shared_core(ids, self.prepare_shared_layout(plan), embeds=embeds)

    def prepare_static(self, static: PaddedStatic) -> None:
        pos = torch.arange(static.seq, device=self.device).repeat(static.rows)
        static.extras["rope"] = self._rope(pos)
        static.extras["pos"] = pos

    def padded_core(self, static: PaddedStatic) -> Tensor:
        x = F.embedding(static.ids.view(-1), self.embed)
        lay = _Layout(rope=static.extras["rope"], pos=static.extras["pos"], max_len=static.seq, static=static)
        return self._trunk(x, lay)


def grouped_layout_host(lengths: Sequence[int], groups: Sequence[Sequence[int]]) -> tuple[list[int], list[int], Tensor]:
    """Full-attention segments of rows laid out `[context][group 1]..[group N]`: `groups[i]` is
    row i's group lengths, its context the tokens before them. A context token reads every token
    of its row, a group token its row's context and its own group, both ways; a row with no
    groups is all context and reads all of itself. The query segments are the contexts and
    groups in token order, so queries need no gather. Returns their lengths, each one's key
    count, and the packed rows forming each segment's keys, back to back (host int64)."""
    lengths_q, lengths_k, starts, runs = [], [], [], []
    row = 0
    for n, own in zip(lengths, groups):
        context = n - sum(own)
        if context:
            lengths_q.append(context)
            lengths_k.append(n)
            starts.append(row)
            runs.append(n)
        at = row + context
        for g in own:
            lengths_q.append(g)
            lengths_k.append(context + g)
            starts += [row, at]
            runs += [context, g]
            at += g
        row += n
    starts, runs = torch.tensor(starts, dtype=torch.long), torch.tensor(runs, dtype=torch.long)
    offsets = torch.cumsum(runs, 0) - runs
    kv_idx = torch.arange(int(runs.sum())) + torch.repeat_interleave(starts - offsets, runs)
    return lengths_q, lengths_k, kv_idx


def _qk_norms(sa: nn.Module) -> tuple[nn.Module, nn.Module]:
    """LFM2 names them q_layernorm/k_layernorm, LFM3 q_norm/k_norm."""
    if hasattr(sa, "q_layernorm"):
        return sa.q_layernorm, sa.k_layernorm
    return sa.q_norm, sa.k_norm


def _out_proj(sa: nn.Module) -> nn.Module:
    """LFM2 names it out_proj, LFM3 o_proj."""
    return sa.out_proj if hasattr(sa, "out_proj") else sa.o_proj


def _require_supported(layer: nn.Module, kind: str, sandwich: bool) -> None:
    """Everything `_prepare` reads must be a plain dense weight; checked for all layers first, so a
    refusal leaves every parameter's storage untouched."""
    if kind not in ("conv", *ATTENTION_KINDS):
        raise UnsupportedTargetError(f"unsupported layer type {kind!r}")
    mlp = layer.feed_forward
    if not all(hasattr(mlp, a) for a in ("w1", "w2", "w3")):
        raise UnsupportedTargetError("only the dense SwiGLU MLP is supported (no MoE layers)")
    if sandwich and not all(hasattr(layer, a) for a in ("post_operator_norm", "post_ffn_norm")):
        raise UnsupportedTargetError("use_sandwich_norm is set but the layer has no post norms")
    linears = [mlp.w1, mlp.w2, mlp.w3]
    if kind != "conv":
        sa = layer.self_attn
        window = getattr(sa, "sliding_window", None)
        if (kind == "sliding_attention") != (window is not None):
            raise UnsupportedTargetError(f"a {kind} layer with sliding_window={window!r}")
        gate = getattr(sa, "gate_proj", None)
        linears += [sa.q_proj, sa.k_proj, sa.v_proj, _out_proj(sa)] + ([gate] if gate is not None else [])
    else:
        if layer.conv.conv.bias is not None:
            raise UnsupportedTargetError("convolutions with a bias are not supported")
        linears += [layer.conv.in_proj, layer.conv.out_proj]
    if not all(isinstance(getattr(lin, "weight", None), nn.Parameter) for lin in linears):
        raise UnsupportedTargetError("every projection must hold its weight as a parameter (fused or wrapped "
                                     "projections are not supported; pack before fusing them)")
    require_plain(linears)
    if kind != "conv" and gate is not None and gate.weight.shape[0] * sa.head_dim != sa.q_proj.weight.shape[0]:
        raise UnsupportedTargetError("only an attention output gate per head is supported")


def _row_hashes(embeds: Tensor) -> Tensor:
    """A 64-bit hash of each embedding row's bits (one GEMV-sized pass; int64 products wrap).
    `plan_sharing` checks the rows a plan pairs are bit-identical, so a collision costs only sharing."""
    bits = embeds.reshape(embeds.shape[0], -1)
    bits = (bits.view(torch.int16) if bits.element_size() == 2 else bits.view(torch.int32)).long()
    g = torch.Generator(device=bits.device).manual_seed(0x5EED)
    mult = torch.randint(-(2 ** 62), 2 ** 62, (bits.shape[1],), device=bits.device, generator=g) | 1
    return (bits * mult).sum(1)


def _rel(got: Tensor, ref: Tensor) -> float:
    ref = ref.float()
    return (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
