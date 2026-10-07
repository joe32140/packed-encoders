"""Shared-prefix planning for the Qwen3.5 engine (arch.qwen3_5.sharing): CPU only.

The GPU engine runs the plan in tests/test_qwen35.py; here a toy causal model with the same three
mixers (a short causal conv, a decaying recurrence, softmax attention over positions) runs each row
in full and runs the plan's forest, and the two must agree exactly.
"""

from __future__ import annotations

import pytest
import torch

from packed_encoders.arch.qwen3_5.sharing import plan_shared_prefixes

W = 4      # Qwen3.5's GatedDeltaNet conv width


def _batch(seed=0):
    """Two groups (prefixes of 80 and 66 tokens), a row sharing only 10 tokens with group a, a row
    shorter than the minimum, and a row that is its group's prefix plus one token."""
    g = torch.Generator().manual_seed(seed)

    def rand(n):
        return torch.randint(0, 500, (n,), generator=g)

    a, b = rand(80), rand(66)
    rows = [torch.cat([a, rand(7)]), rand(40), torch.cat([b, rand(12)]), torch.cat([a, rand(1)]),
            torch.cat([a[:10], rand(90)]), torch.cat([b, rand(3)]), torch.cat([a, rand(20)])]
    return rows, torch.cat(rows), [len(r) for r in rows]


def _segments(x, lengths):
    return list(x.split(lengths))


def test_plan_groups_rows_and_maps_every_token_back():
    rows, ids, lengths = _batch()
    plan = plan_shared_prefixes(ids, lengths, min_prefix=64, conv_width=W)
    assert plan is not None
    # roots: a's prefix (at row 0), row 1, b's prefix (at row 2), row 4; then one child per grouped row
    assert plan.n_roots == 4 and plan.lengths[:4] == [80, 40, 66, 100]
    assert plan.lengths[4:] == [7, 1, 20, 12, 3] and plan.parent.tolist() == [0, 0, 0, 2, 2]
    assert plan.saved_tokens == 2 * 80 + 66
    fwd = ids[plan.src]
    assert torch.equal(fwd[plan.out], ids)                          # every caller token is computed once
    pos = torch.cat([torch.arange(n) for n in lengths])
    assert torch.equal(plan.rope_pos[plan.out], pos)                # at its own position in its row
    assert torch.equal(plan.conv_pos, torch.cat([torch.arange(n) for n in plan.lengths]))
    # keys: a root's own tokens; a child's whole row (its root's prefix, then its own tokens)
    keys = _segments(fwd[plan.kv_idx], plan.kv_lengths)
    caller = [0, 1, 2, 4, 0, 3, 6, 2, 5]                            # the row each segment continues
    for seg, (r, n) in enumerate(zip(caller, plan.kv_lengths)):
        assert torch.equal(keys[seg], rows[r][:n]), seg
    # a child's first W-1 tokens: conv taps are the W tokens of its row ending at that token
    for row, taps in zip(plan.fix_rows.tolist(), plan.fix_taps):
        seg = int(torch.searchsorted(torch.tensor(plan.lengths).cumsum(0), row, right=True))
        p = int(plan.rope_pos[row])
        assert torch.equal(fwd[taps], rows[caller[seg]][p - W + 1: p + 1])


def test_prefix_keeps_one_token_per_row_and_needs_the_minimum():
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, 500, (70,), generator=g)
    plan = plan_shared_prefixes(torch.cat([a, a]), [70, 70], min_prefix=64, conv_width=W)
    assert plan.lengths == [69, 1, 1]                               # identical rows still each keep a token
    assert plan_shared_prefixes(torch.cat([a, a]), [70, 70], min_prefix=70, conv_width=W) is None
    assert plan_shared_prefixes(torch.cat([a[:60], a[:60]]), [60, 60], min_prefix=64, conv_width=W) is None
    assert plan_shared_prefixes(torch.cat([a, a.flip(0)]), [70, 70], min_prefix=64, conv_width=W) is None
    assert plan_shared_prefixes(a, [70], min_prefix=64, conv_width=W) is None
    with pytest.raises(ValueError, match="conv width"):
        plan_shared_prefixes(a, [70], min_prefix=2, conv_width=W)


def _toy(dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb, w = torch.randn(500, dim, generator=g, dtype=torch.float64), torch.randn(W, dim, generator=g, dtype=torch.float64)
    decay = torch.rand(dim, generator=g, dtype=torch.float64)

    def conv(x):                                   # taps oldest first, zero before the sequence
        pad = torch.cat([x.new_zeros(W - 1, x.shape[1]), x])
        return sum(pad[i: i + len(x)] * w[i] for i in range(W))

    def scan(y, state):
        out = []
        for t in y:
            state = decay * state + t
            out.append(state)
        return torch.stack(out), state

    def attend(q, k, qpos, kpos):                  # causal by position, with a position-dependent bias
        s = q @ k.T / 4 - 0.05 * (qpos[:, None] - kpos[None]).abs()
        return torch.softmax(s.masked_fill(kpos[None] > qpos[:, None], float("-inf")), -1) @ k

    return emb, w, conv, scan, attend


def test_forest_reproduces_each_row_run_in_full():
    rows, ids, lengths = _batch(seed=3)
    emb, w, conv, scan, attend = _toy()
    want = []
    for r in rows:
        h = scan(conv(emb[r]), torch.zeros(emb.shape[1], dtype=torch.float64))[0]
        want.append(attend(h, h, torch.arange(len(r)), torch.arange(len(r))))
    want = torch.cat(want)

    plan = plan_shared_prefixes(ids, lengths, min_prefix=64, conv_width=W)
    x = emb[ids[plan.src]]
    y = torch.cat([conv(s) for s in _segments(x, plan.lengths)])     # restarts at every segment
    y[plan.fix_rows] = (x[plan.fix_taps] * w).sum(1)                 # children read their root's last tokens
    h, finals = [], []
    for s in _segments(y, plan.lengths)[: plan.n_roots]:
        o, last = scan(s, torch.zeros(y.shape[1], dtype=torch.float64))
        h.append(o)
        finals.append(last)
    for s, p in zip(_segments(y, plan.lengths)[plan.n_roots:], plan.parent.tolist()):
        h.append(scan(s, finals[p])[0])
    h = torch.cat(h)
    k, kpos = _segments(h[plan.kv_idx], plan.kv_lengths), _segments(plan.rope_pos[plan.kv_idx], plan.kv_lengths)
    got = torch.cat([attend(q, kk, qp, kp) for q, kk, qp, kp in
                     zip(_segments(h, plan.lengths), k, _segments(plan.rope_pos, plan.lengths), kpos)])
    torch.testing.assert_close(got[plan.out], want, rtol=1e-12, atol=1e-12)
