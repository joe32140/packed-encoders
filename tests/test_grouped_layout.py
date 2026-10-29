"""Grouped rows: the engine's key layout is exactly the pattern `grouped_attention_masks` documents.

    pytest tests/test_grouped_layout.py -q        # host only, no GPU

`forward_grouped` runs each full-attention layer as query segments (contexts and groups in token
order), each reading every key of its gathered key segment. Rebuilt as a dense mask, that layout
must equal the full-attention mask the model's own forward is checked under, for any geometry.
Pack-time validation can't tell some wrong patterns from bf16 noise (on a trained 2B decision
model, groups also reading the groups before them score 0.996 mean cosine), so the pattern is
pinned here, exactly.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from packed_encoders.arch.lfm2 import grouped_attention_masks
from packed_encoders.arch.lfm2.engine import grouped_layout_host


def _from_layout(lengths, groups):
    lengths_q, lengths_k, kv_idx = grouped_layout_host(lengths, groups)
    T = sum(lengths)
    allowed = torch.zeros(T, T, dtype=torch.bool)
    q0 = k0 = 0
    for nq, nk in zip(lengths_q, lengths_k):
        keys = kv_idx[k0:k0 + nk]
        assert keys.unique().numel() == nk          # a repeated key would count twice in the softmax
        allowed[q0:q0 + nq, keys] = True
        q0, k0 = q0 + nq, k0 + nk
    assert q0 == T and k0 == kv_idx.numel()
    return allowed


def _from_masks(lengths, groups):
    cfg = SimpleNamespace(layer_types=["full_attention"], sliding_window=None)
    full = grouped_attention_masks(cfg, lengths, groups, max(lengths), torch.float32, "cpu")["full_attention"]
    allowed = torch.zeros(sum(lengths), sum(lengths), dtype=torch.bool)
    at = 0
    for i, n in enumerate(lengths):
        allowed[at:at + n, at:at + n] = full[i, 0, :n, :n] == 0
        at += n
    return allowed


def _geometries():
    yield "context only", [9], [[]]
    yield "one group, the whole row", [6], [[6]]
    yield "groups without a context", [7], [[3, 4]]
    yield "one-token context and groups", [4], [[1, 1, 1]]
    yield "mixed rows", [60, 25, 29, 7], [[7, 1, 12], [], [20], [3, 3]]
    rng = random.Random(0)
    for case in range(200):
        groups = [[rng.randint(1, 12) for _ in range(rng.randint(0, 6))] for _ in range(rng.randint(1, 5))]
        yield f"random {case}", [sum(own) + rng.randint(0, 20) or 1 for own in groups], groups


@pytest.mark.parametrize(("lengths", "groups"), [g[1:] for g in _geometries()], ids=[g[0] for g in _geometries()])
def test_layout_is_exactly_the_documented_full_attention_mask(lengths, groups):
    assert torch.equal(_from_layout(lengths, groups), _from_masks(lengths, groups))


def test_the_documented_mask_is_the_grouped_pattern():
    """The mask itself, spelled out for one row: a 2-token context, then groups of 2 and 1."""
    rows = ["1 1 1 1 1",                         # context: the whole row
            "1 1 1 1 1",
            "1 1 1 1 0",                         # first group: context, then itself both ways
            "1 1 1 1 0",
            "1 1 0 0 1"]                         # second group: context, then itself
    expected = torch.tensor([[int(x) for x in r.split()] for r in rows], dtype=torch.bool)
    assert torch.equal(_from_masks([5], [[2, 1]]), expected)
