"""Choose a varlen attention kernel for one head geometry by measuring it.

Whether a kernel serves a shape depends on the GPU, the wheel, and the head dim (FA4 on
sm_90/sm_100, FA2 wheels built for some torch/CUDA pairs, torch's own varlen kernel from
2.10, head_dim 256 support varying across all three). A lookup table goes stale; a probe
does not. Each candidate must import, run, and match fp32 per-segment SDPA on a probe that
includes an empty segment and grouped KV heads, in both layouts the engines use:

- packed: tokens back to back, `cu_seqlens` (eager forward)
- padded: `(rows, S)` rectangles with `[real | pad]` segments per row (CUDA graphs)

The first candidate that passes wins. `sdpa` always passes, so selection never fails; it
only gets slower. Rejections are recorded with their reason for `validate()` reports.

Causal engines also get a *prefixed* form: each query segment is the tail of a longer key
segment (rows continuing a shared prefix, see arch.qwen3_5.sharing), so the causal mask is
aligned to the end. It is probed the same way; when the winner's fails, SDPA serves it.

A choice is made for one attention *kind*: optionally a local window `(left, right)`
(`-1`: unbounded) and a softmax scale (default `head_dim ** -0.5`). Both are bound into
its kernels, and the probe runs segments longer than the window.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from packed_encoders.errors import ValidationError
from packed_encoders.runtime.graphs import PaddedStatic

PROBE_TOLERANCE = 2e-2  # max abs error vs fp32 SDPA on unit-variance bf16 inputs


@dataclass
class AttentionChoice:
    """`packed(q, k, v, cu32, max_len, lengths, causal)` and `padded(q, k, v, static, causal)`
    both take (tokens, heads, head_dim) q/k/v (k/v may have fewer heads) and return q's shape.
    `prefixed(q, k, v, cu_q32, cu_k32, max_q, max_k, lengths_q, lengths_k, causal)` (causal
    engines only, else None): query segment i is the last `lengths_q[i]` of key segment i.
    Every form applies `window` and `scale`."""

    name: str
    packed: Callable[..., Tensor]
    padded: Callable[..., Tensor]
    max_abs_err: float = 0.0
    rejected: dict[str, str] = field(default_factory=dict)
    prefixed: Callable[..., Tensor] | None = None
    prefixed_name: str | None = None
    prefixed_max_abs_err: float | None = None
    window: tuple[int, int] = (-1, -1)
    scale: float | None = None


# ---------------------------------------------------------------------------- candidates


def _flash2(window, scale) -> tuple[Callable, ...]:
    from flash_attn import flash_attn_varlen_func as f

    kw = dict(dropout_p=0.0, softmax_scale=scale, window_size=window)

    def packed(q, k, v, cu32, max_len, lengths, causal):
        return f(q, k, v, cu32, cu32, max_len, max_len, causal=causal, **kw)

    def padded(q, k, v, static, causal):
        return f(q, k, v, static.seg_cu32, static.seg_cu32, static.seq, static.seq, causal=causal, **kw)

    def prefixed(q, k, v, cu_q, cu_k, max_q, max_k, lengths_q, lengths_k, causal):
        return f(q, k, v, cu_q, cu_k, max_q, max_k, causal=causal, **kw)

    return packed, padded, prefixed


def _flash4(window, scale) -> tuple[Callable, ...]:
    from flash_attn.cute import flash_attn_varlen_func as f

    unbounded = tuple(None if w < 0 else w for w in window)      # FA4 spells "no limit" None

    def call(q, k, v, cu_q, cu_k, max_q, max_k, causal):
        out = f(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
                softmax_scale=scale, causal=causal, window_size=unbounded)
        return out[0] if isinstance(out, (tuple, list)) else out

    return (lambda q, k, v, cu32, max_len, lengths, causal: call(q, k, v, cu32, cu32, max_len, max_len, causal),
            lambda q, k, v, static, causal: call(q, k, v, static.seg_cu32, static.seg_cu32, static.seq, static.seq, causal),
            lambda q, k, v, cu_q, cu_k, max_q, max_k, lengths_q, lengths_k, causal:
                call(q, k, v, cu_q, cu_k, max_q, max_k, causal))


def _repeat_kv(q: Tensor, k: Tensor, v: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    r = q.shape[dim] // k.shape[dim]
    return (k, v) if r == 1 else (k.repeat_interleave(r, dim), v.repeat_interleave(r, dim))


def _torch_varlen(window, scale) -> tuple[Callable, ...]:
    from torch.nn.attention.varlen import varlen_attn as f  # torch >= 2.10

    params = inspect.signature(f).parameters
    if window != (-1, -1) and "window_size" not in params:
        raise NotImplementedError("this torch varlen_attn has no local window")
    kwargs = {"scale": scale} if scale is not None else {}
    if window != (-1, -1):
        kwargs["window_size"] = list(window)
    # torch 2.11's public varlen_attn has no is_causal; a window with no right context is causal.
    causal_kw = ({"is_causal": True} if "is_causal" in params
                 else {"window_size": [window[0], 0]} if "window_size" in params else None)

    def call(q, k, v, cu_q, cu_k, max_q, max_k, causal):
        if causal and causal_kw is None:
            raise NotImplementedError("this torch varlen_attn has no causal mode")
        k, v = _repeat_kv(q, k, v, 1)
        return f(q, k, v, cu_q, cu_k, max_q, max_k, **{**kwargs, **(causal_kw if causal else {})})

    return (lambda q, k, v, cu32, max_len, lengths, causal: call(q, k, v, cu32, cu32, max_len, max_len, causal),
            lambda q, k, v, static, causal: call(q, k, v, static.seg_cu32, static.seg_cu32, static.seq, static.seq, causal),
            lambda q, k, v, cu_q, cu_k, max_q, max_k, lengths_q, lengths_k, causal:
                call(q, k, v, cu_q, cu_k, max_q, max_k, causal))


def _sdpa(window, scale) -> tuple[Callable, ...]:
    def packed(q, k, v, cu32, max_len, lengths, causal):
        # Eager only (reads host lengths): one SDPA per sequence.
        return _segments(q, k, v, list(lengths), list(lengths), causal, window, scale)

    def padded(q, k, v, static: PaddedStatic, causal):
        # Capture-safe: the [real | pad] segment mask is built on device from `lens`.
        rows, seq = static.rows, static.seq
        k, v = _repeat_kv(q, k, v, 1)
        qb, kb, vb = (t.view(rows, seq, t.shape[1], t.shape[2]).transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(seq, device=q.device)
        segment = pos[None, :] >= static.lens[:, None]                     # False real, True pad
        allowed = segment[:, :, None] == segment[:, None, :]
        # A window bounds position differences, so one band serves both segments of a row.
        band = _band(seq, seq, causal, window, q.device)
        if band is not None:
            allowed = allowed & band
        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=allowed[:, None], scale=scale)
        return out.transpose(1, 2).reshape(rows * seq, q.shape[1], q.shape[2])

    def prefixed(q, k, v, cu_q, cu_k, max_q, max_k, lengths_q, lengths_k, causal):
        # Eager only: one SDPA per segment. SDPA's is_causal aligns a short query block to the
        # start, so the end-aligned mask is explicit.
        return _segments(q, k, v, list(lengths_q), list(lengths_k), causal, window, scale)

    return packed, padded, prefixed


def _band(n_q, n_k, causal, window, device):
    """Which keys each query may read, the queries being the last `n_q` of `n_k` positions;
    None when every key is allowed."""
    left, right = window
    right = 0 if causal else right
    if left < 0 and right < 0:
        return None
    delta = torch.arange(n_k - n_q, n_k, device=device)[:, None] - torch.arange(n_k, device=device)[None, :]
    allowed = torch.ones(n_q, n_k, dtype=torch.bool, device=device)
    if left >= 0:
        allowed &= delta <= left
    if right >= 0:
        allowed &= delta >= -right
    return allowed


def _segments(q, k, v, lengths_q, lengths_k, causal, window=(-1, -1), scale=None):
    k, v = _repeat_kv(q, k, v, 1)
    outs = []
    for qs, ks, vs in zip(q.split(lengths_q), k.split(lengths_k), v.split(lengths_k)):
        if not qs.shape[0]:
            continue
        mask = _band(qs.shape[0], ks.shape[0], causal, window, q.device)
        outs.append(F.scaled_dot_product_attention(qs.transpose(0, 1), ks.transpose(0, 1), vs.transpose(0, 1),
                                                   attn_mask=mask, scale=scale).transpose(0, 1))
    return torch.cat(outs) if outs else torch.empty_like(q)


_CANDIDATES: dict[str, Callable[..., tuple[Callable, ...]]] = {  # (window, scale) -> kernels
    "flash2": _flash2, "flash4": _flash4, "torch_varlen": _torch_varlen, "sdpa": _sdpa,
}


def default_order(device: torch.device) -> tuple[str, ...]:
    major = torch.cuda.get_device_capability(device)[0]
    if major in (9, 10, 11):                        # FA4 (CuteDSL) targets Hopper / datacenter Blackwell
        return ("flash4", "flash2", "torch_varlen", "sdpa")
    return ("flash2", "torch_varlen", "flash4", "sdpa")


# ---------------------------------------------------------------------------- probe


def _reference(q, k, v, lengths, causal, window=(-1, -1), scale=None):
    """fp32 per-segment SDPA."""
    return _segments(q.float(), k.float(), v.float(), list(lengths), list(lengths), causal, window, scale)


def _probe(packed, padded, *, n_heads, n_kv_heads, head_dim, causal, device, dtype, window, scale) -> float:
    gen = torch.Generator(device=device).manual_seed(0)
    # A window is only exercised by segments longer than it, on both sides of some token.
    span = 2 * max(window) + 3

    def rnd(t, h):
        return torch.randn(t, h, head_dim, device=device, dtype=dtype, generator=gen)

    # packed layout, including an empty sequence
    lengths = [5, max(64, span), 0, 37]
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q, k, v = rnd(total, n_heads), rnd(total, n_kv_heads), rnd(total, n_kv_heads)
    ref = _reference(q, k, v, lengths, causal, window, scale)
    err = (packed(q, k, v, cu, max(lengths), lengths, causal).float() - ref).abs().max().item()

    # padded layout: rows of S with [real | pad]; only real rows are compared
    rows, seq = 3, max(48, span)
    lens = [5, seq, 0]
    starts = [i * seq for i in range(rows)]
    seg = [0]
    for s, n in zip(starts, lens):
        seg += [s + n, s + seq]
    static = PaddedStatic(rows=rows, seq=seq, ids=torch.zeros(rows, seq, dtype=torch.long, device=device),
                          seg_cu32=torch.tensor(seg, dtype=torch.int32, device=device),
                          lens=torch.tensor(lens, dtype=torch.int32, device=device))
    q, k, v = rnd(rows * seq, n_heads), rnd(rows * seq, n_kv_heads), rnd(rows * seq, n_kv_heads)
    out = padded(q, k, v, static, causal).float()
    keep = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lens)]).to(device)
    ref = _reference(q.index_select(0, keep), k.index_select(0, keep), v.index_select(0, keep),
                     [n for n in lens], causal, window, scale)
    err = max(err, (out.index_select(0, keep) - ref).abs().max().item())
    return err


def reference_prefixed(q, k, v, lengths_q, lengths_k, causal, window=(-1, -1), scale=None):
    """fp32 per-segment SDPA, each query segment aligned to the end of its key segment."""
    return _segments(q.float(), k.float(), v.float(), list(lengths_q), list(lengths_k), causal, window, scale)


def _probe_prefixed(prefixed, *, n_heads, n_kv_heads, head_dim, device, dtype, window, scale) -> float:
    gen = torch.Generator(device=device).manual_seed(1)
    lengths_q, lengths_k = [5, 1, 37, 16], [40, 64, 37, 80]     # a short tail, one token, a square, a long prefix

    def rnd(t, h):
        return torch.randn(t, h, head_dim, device=device, dtype=dtype, generator=gen)

    def cu(lengths):
        return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)

    q, k, v = rnd(sum(lengths_q), n_heads), rnd(sum(lengths_k), n_kv_heads), rnd(sum(lengths_k), n_kv_heads)
    got = prefixed(q, k, v, cu(lengths_q), cu(lengths_k), max(lengths_q), max(lengths_k), lengths_q, lengths_k, True)
    ref = reference_prefixed(q, k, v, lengths_q, lengths_k, True, window, scale)
    return (got.float() - ref).abs().max().item()


def select_attention(
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    order: Sequence[str] | None = None,
    window: tuple[int, int] = (-1, -1),
    scale: float | None = None,
) -> AttentionChoice:
    """Return the first candidate in `order` that imports, runs, and matches fp32 SDPA; for a
    causal model, also its prefixed form (or SDPA's, when the candidate's misses its probe).
    `window` (left, right; -1 unbounded) and `scale` (None: head_dim ** -0.5) are part of the
    attention kind: bound into the kernels and probed."""
    rejected: dict[str, str] = {}
    window = tuple(int(w) for w in window)
    geometry = dict(n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, device=device, dtype=dtype,
                    window=window, scale=scale)
    for name in order or default_order(device):
        try:
            packed, padded, prefixed = _CANDIDATES[name](window, scale)
            err = _probe(packed, padded, causal=causal, **geometry)
        except Exception as exc:  # noqa: BLE001 — any import/launch failure just rejects the candidate
            rejected[name] = _reason(exc)
            continue
        if not err < PROBE_TOLERANCE:
            rejected[name] = f"mismatch vs fp32 SDPA: max abs err {err:.3g}"
            continue
        choice = AttentionChoice(name=name, packed=packed, padded=padded, max_abs_err=err, rejected=rejected,
                                 window=window, scale=scale)
        if causal:
            for prefixed_name, fn in ((name, prefixed), ("sdpa", _sdpa(window, scale)[2])):
                try:
                    perr = _probe_prefixed(fn, **geometry)
                except Exception as exc:  # noqa: BLE001
                    rejected[f"{prefixed_name}:prefixed"] = _reason(exc)
                    continue
                if perr < PROBE_TOLERANCE:
                    choice.prefixed, choice.prefixed_name, choice.prefixed_max_abs_err = fn, prefixed_name, perr
                    break
                rejected[f"{prefixed_name}:prefixed"] = f"mismatch vs fp32 SDPA: max abs err {perr:.3g}"
        return choice
    raise ValidationError(f"no attention candidate passed: {rejected}")


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
