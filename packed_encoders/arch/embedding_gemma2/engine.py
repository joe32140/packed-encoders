"""Exact fast forward for an EmbeddingGemma 2 text backbone (bidirectional GQA, sliding and full layers).

Same math as the model, fewer and fatter kernels:

- merged projections: attention q|k|v and MLP gate|up, one GEMM each. As in the hybrid engines
  the merged weight *is* the HF weight (each Linear is re-pointed at its rows; `release()`
  separates them).
- every layer is three post-norm residual blocks (attention, MLP, then the per-layer embedding
  gate). One Triton launch per block adds the normed output to the residual stream and norms
  the sum for the next block (`pieces.post_norm`), rounding where the modules round.
- q/k RMSNorm + RoPE in one launch reading the q|k|v projection in place; v is RMS-normed without
  a scale. Attention is bidirectional, scaled by 1, and windowed on sliding layers; each kind
  (sliding, full: their head dims differ) has its own kernel, chosen by probe (runtime.attention).

Two entry points share the layer code: `forward_packed` (eager, padding-free, token ids or input
embeddings) and `padded_core` (the CUDA-graph region, right-padded `(rows, S)`; exact because
attention treats each row as `[real | pad]` segments — see runtime.graphs). Both return the
final norm projected by `embedding_projection`, per token.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import (
    require_independent_parameters,
    require_plain,
    share_rows,
    unshare_rows,
)
from packed_encoders.errors import UnsupportedTargetError, ValidationError
from packed_encoders.runtime.attention import AttentionChoice, select_attention
from packed_encoders.runtime.graphs import PaddedStatic
from packed_encoders.runtime.staging import PinnedStager, packed_layout_host

KERNEL_TOLERANCE = 2e-2        # max |fused - model's module|, relative to the module's max |output| (bf16: a few ulp)
_KINDS = ("sliding_attention", "full_attention")


class _Layer:
    pass


@dataclass
class _Layout:
    rope: dict[str, tuple[Tensor, Tensor]]   # attention kind -> RoPE tables at each token's position in its row
    cu32: Tensor | None = None               # packed: int32 cu_seqlens (attention)
    max_len: int = 0
    lengths: list[int] | None = None
    static: PaddedStatic | None = None


class EmbeddingGemma2Engine:
    def __init__(self, text_model: nn.Module, *, attention_order: Sequence[str] | None = None, pieces=None):
        from packed_encoders.arch.embedding_gemma2.pieces import default_pieces

        self.composition = pieces or default_pieces()
        self.ops = self.composition.bind()
        require_independent_parameters(text_model, "EmbeddingGemma 2")
        cfg = text_model.config
        self.tm, self.cfg = text_model, cfg
        self.embed = text_model.embed_tokens.weight
        self.device, self.dtype = self.embed.device, self.embed.dtype
        if self.device.type != "cuda":
            raise UnsupportedTargetError(f"the EmbeddingGemma 2 engine needs CUDA weights; model is on {self.device}")
        self.model_dim = cfg.hidden_size                                    # input embeddings
        self.hidden_size = text_model.embedding_projection.weight.shape[0]  # the per-token output
        self.eps = cfg.rms_norm_eps
        self._norms: list[list] = []            # [tensor (detached), fp32 copy, version seen]; see sync_norms
        self.shared: list[nn.Linear] = []
        self.layers: list[_Layer] = []
        n = cfg.num_hidden_layers
        kinds = list(getattr(cfg, "layer_types", None) or [])
        if len(kinds) < n:
            raise UnsupportedTargetError("config.layer_types is missing or shorter than num_hidden_layers")
        _require_supported(text_model, kinds[:n])          # every layer before any re-pointing
        try:
            with torch.cuda.device(self.device):
                ple = text_model.ple
                self.w_ple_in = ple.per_layer_model_projection.weight
                self.ple_in_scale = float(ple.per_layer_model_projection_scale)
                self.ple_dim = ple.hidden_size_per_layer_input
                self.w_ple_norm = self._scale(ple.per_layer_projection_norm.weight)
                self.w_final = self._scale(text_model.norm.weight)
                self.w_out = text_model.embedding_projection.weight
                for layer, kind in zip(text_model.layers[:n], kinds[:n]):
                    self.layers.append(self._prepare(layer, kind))
                self.kinds = [kind for kind in _KINDS if kind in kinds[:n]]
                self.attention: dict[str, AttentionChoice] = {}
                self.kernel_errors: dict[str, float] = {}
                for kind in self.kinds:
                    L = next(L for L in self.layers if L.kind == kind)
                    # HF's bidirectional window: |query - key| <= sliding_window
                    window = (cfg.sliding_window, cfg.sliding_window) if kind == "sliding_attention" else (-1, -1)
                    self.attention[kind] = select_attention(n_heads=L.nq, n_kv_heads=L.nkv, head_dim=L.hd, causal=False,
                                                            device=self.device, dtype=self.dtype,
                                                            order=attention_order, window=window, scale=L.scaling)
                    self.kernel_errors[f"qk_norm_rope:{kind}"] = self._check_qk(L)
                self._stager = PinnedStager(self.device)
        except BaseException:
            self.release(rollback=True)                          # leave the model exactly as we found it
            raise

    def release(self, *, rollback: bool = False) -> None:
        """Drop the merged weights, then give every projection its own storage back."""
        self.layers.clear()
        self._norms.clear()
        self.w_final = self.w_ple_norm = None
        unshare_rows(self.shared, rollback=rollback)

    # ------------------------------------------------------------------ weights
    def _prepare(self, layer: nn.Module, kind: str) -> _Layer:
        L = _Layer()
        L.kind = kind
        sa = layer.self_attn
        L.module = sa
        L.hd = sa.head_dim
        L.nq, L.nkv = sa.q_proj.weight.shape[0] // L.hd, sa.k_proj.weight.shape[0] // L.hd
        L.scaling = float(sa.scaling)
        L.qkv = share_rows([sa.q_proj, sa.k_proj, sa.v_proj], self.shared)
        L.wqk = torch.empty((2, L.hd), device=self.device, dtype=torch.float32)   # q row, k row
        self._scale(sa.q_norm.weight, out=L.wqk[0])
        self._scale(sa.k_norm.weight, out=L.wqk[1])
        L.qk_eps = sa.q_norm.eps
        L.out = sa.o_proj.weight
        L.w_in = self._scale(layer.input_layernorm.weight)
        L.w_post_attn = self._scale(layer.post_attention_layernorm.weight)
        L.w_pre_ffn = self._scale(layer.pre_feedforward_layernorm.weight)
        L.w_post_ffn = self._scale(layer.post_feedforward_layernorm.weight)
        mlp = layer.mlp
        L.gate_up = share_rows([mlp.gate_proj, mlp.up_proj], self.shared)
        L.inter = mlp.gate_proj.weight.shape[0]
        L.down = mlp.down_proj.weight
        pb = layer.ple_block
        L.ple_gate, L.ple_proj = pb.per_layer_input_gate.weight, pb.per_layer_projection.weight
        L.w_post_ple = self._scale(pb.post_per_layer_input_norm.weight)
        L.scalar = self._scale(layer.layer_scalar)
        return L

    def _scale(self, weight: Tensor, out: Tensor | None = None) -> Tensor:
        """fp32 copy of a norm weight (or the layer scalar) for the fused kernels; tracked (sync_norms)."""
        src = weight.detach()                    # shares the tensor's storage and version counter
        if out is None:
            out = torch.empty(src.shape, device=src.device, dtype=torch.float32)
        out.copy_(src)
        self._norms.append([src, out, src._version])
        return out

    def sync_norms(self) -> None:
        """Re-copy every norm weight whose tensor changed since it was copied (an optimizer step,
        `load_state_dict`). In place, so captured graphs stay valid. Writes through `.data` bypass the
        version counter: re-pack after those."""
        for entry in self._norms:
            src, out, seen = entry
            if src._version != seen:
                out.copy_(src)
                entry[2] = src._version

    def _check_qk(self, L: _Layer) -> float:
        """The fused q/k norm + RoPE against the model's own RMSNorms + apply_rotary_pos_emb."""
        from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
            apply_rotary_pos_emb,
        )

        g = torch.Generator(device=self.device).manual_seed(0)
        T = 37
        qw = L.nq * L.hd
        proj = torch.randn(T, qw + 2 * L.nkv * L.hd, device=self.device, dtype=self.dtype, generator=g)
        cos, sin = self._rope(torch.arange(T, device=self.device) * 7)[L.kind]
        with torch.no_grad():
            qx, kx = proj[:, :qw].view(T, L.nq, L.hd), proj[:, qw: qw + L.nkv * L.hd].view(T, L.nkv, L.hd)
            rq, rk = apply_rotary_pos_emb(L.module.q_norm(qx), L.module.k_norm(kx), cos, sin, unsqueeze_dim=1)
            pq, pk = self.ops.qk_norm_rope_pair(proj, 0, L.hd, L.nq, qw, L.hd, L.nkv, L.wqk, cos, sin, L.qk_eps)
            err = max(_rel(pq, rq), _rel(pk, rk))
        if not err < KERNEL_TOLERANCE:
            raise ValidationError(f"fused q/k norm + RoPE mismatch vs the model's modules: relative error {err:.2e}")
        return err

    def _rope(self, pos: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
        probe = torch.empty(1, device=self.device, dtype=self.dtype)
        tables = {}
        for kind in self.kinds:
            cos, sin = self.tm.rotary_emb(probe, pos[None], kind)
            tables[kind] = (cos[0], sin[0])
        return tables

    # ------------------------------------------------------------------ layers
    def _attn(self, L: _Layer, h: Tensor, lay: _Layout) -> Tensor:
        N, nq, nkv, hd = h.shape[0], L.nq, L.nkv, L.hd
        out = self.ops.linear(h, L.qkv)
        qw, kw = nq * hd, nkv * hd
        cos, sin = lay.rope[L.kind]
        q, k = self.ops.qk_norm_rope_pair(out, 0, hd, nq, qw, hd, nkv, L.wqk, cos, sin, L.qk_eps)
        v = self.ops.rms_norm(out[:, qw + kw:].view(N, nkv, hd), None, L.qk_eps)
        choice = self.attention[L.kind]
        if lay.static is None:
            o = self.ops.attention_packed(choice, q, k, v, lay.cu32, lay.max_len, lay.lengths, False)
        else:
            o = self.ops.attention_padded(choice, q, k, v, lay.static, False)
        return self.ops.linear(o.reshape(N, nq * hd), L.out)

    def _trunk(self, x: Tensor, lay: _Layout) -> Tensor:
        T, eps, ops = x.shape[0], self.eps, self.ops
        # Every layer's slice of the per-layer embeddings, from the input embeddings alone.
        ple = ops.rms_norm(ops.linear(x, self.w_ple_in).view(T, len(self.layers), self.ple_dim), self.w_ple_norm, eps,
                           self.ple_in_scale)
        h = ops.rms_norm(x, self.layers[0].w_in, eps)
        for i, L in enumerate(self.layers):
            x, h = ops.post_norm_add(self._attn(L, h, lay), x, L.w_post_attn, eps, None, L.w_pre_ffn)
            g, u = ops.linear(h, L.gate_up).split(L.inter, dim=-1)
            x = ops.post_norm_add(ops.linear(ops.gelu_mul(g, u), L.down), x, L.w_post_ffn, eps)
            p = ops.linear(ops.gelu_mul(ops.linear(x, L.ple_gate), ple[:, i]), L.ple_proj)
            following = self.layers[i + 1].w_in if i + 1 < len(self.layers) else self.w_final
            x, h = ops.post_norm_add(p, x, L.w_post_ple, eps, L.scalar, following)
        return ops.linear(h, self.w_out)

    def _embed(self, ids: Tensor) -> Tensor:
        return F.embedding(ids.reshape(-1), self.embed) * self.tm.embed_tokens.embed_scale.to(self.dtype)

    # ------------------------------------------------------------------ entry points
    @torch.no_grad()
    def forward_packed(self, ids: Tensor | None, lengths: Sequence[int], *, embeds: Tensor | None = None) -> Tensor:
        """Eager, padding-free. ids: (T,) device ids back to back, or embeds: (T, model_dim) input
        embeddings in their place (scaled, as the model's `embed_tokens` returns them); lengths: host
        ints (all > 0). Returns the projected hidden states (T, hidden_size)."""
        with torch.cuda.device(self.device):
            self.sync_norms()
            cu, pos = packed_layout_host(lengths)
            d_pos, d_cu = self._stager.put([pos, cu])
            x = self._embed(ids) if embeds is None else embeds.to(self.dtype).contiguous()
            lay = _Layout(rope=self._rope(d_pos), cu32=d_cu.to(torch.int32), max_len=int(max(lengths)),
                          lengths=list(lengths))
            return self._trunk(x, lay)

    def prepare_static(self, static: PaddedStatic) -> None:
        static.extras["rope"] = self._rope(torch.arange(static.seq, device=self.device).repeat(static.rows))

    def padded_core(self, static: PaddedStatic) -> Tensor:
        lay = _Layout(rope=static.extras["rope"], max_len=static.seq, static=static)
        return self._trunk(self._embed(static.ids), lay)


def _require_supported(text_model: nn.Module, kinds: Sequence[str]) -> None:
    """Everything the engine reads must be a plain dense weight in the layout it computes; checked
    for every layer first, so a refusal leaves every parameter's storage untouched."""
    cfg = text_model.config
    if getattr(cfg, "hidden_activation", None) != "gelu_pytorch_tanh":
        raise UnsupportedTargetError(f"unsupported activation {getattr(cfg, 'hidden_activation', None)!r} "
                                     "(the engine computes tanh GELU)")
    dims = {"hidden_size": cfg.hidden_size, "hidden_size_per_layer_input": text_model.ple.hidden_size_per_layer_input}
    linears = [text_model.ple.per_layer_model_projection, text_model.embedding_projection]
    for layer, kind in zip(text_model.layers, kinds):
        if kind not in _KINDS:
            raise UnsupportedTargetError(f"unsupported EmbeddingGemma 2 layer type {kind!r}")
        sa = layer.self_attn
        if getattr(sa, "is_causal", False):
            raise UnsupportedTargetError("the EmbeddingGemma 2 engine runs bidirectional attention only")
        if not (sa.q_norm.with_scale and sa.k_norm.with_scale) or sa.v_norm.with_scale:
            raise UnsupportedTargetError("expected scaled q/k norms and an unscaled v norm")
        dims[f"head_dim ({kind})"] = sa.head_dim
        mlp, pb = layer.mlp, layer.ple_block
        linears += [sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj,
                    pb.per_layer_input_gate, pb.per_layer_projection]
    odd = {k: v for k, v in dims.items() if v <= 0 or v & (v - 1)}
    if odd:
        raise UnsupportedTargetError(f"the fused norms need power-of-two widths; got {odd}")
    require_plain(linears)


def _rel(got: Tensor, ref: Tensor) -> float:
    ref = ref.float()
    return (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
