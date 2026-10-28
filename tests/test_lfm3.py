"""LFM3 hybrid (LFM2 + sliding-window attention, a per-head output gate, sandwich norms) — engine,
grouped rows, HF entry.

    pytest tests/test_lfm3.py -q        # a tiny random LFM3 (its modeling code, no weights)

LFM3's modeling code is not in a transformers release yet; it ships with its checkpoints. These
tests take transformers' own classes when present, else the code of a pinned checkpoint revision
(three small files from the Hub, cached), and skip when neither is reachable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the LFM3 engine needs an sm_80+ CUDA GPU",
)
fla = pytest.importorskip("fla")

CODE_REPO, CODE_REVISION = "LiquidAI/LFM3-1.7B", "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
VOCAB, WINDOW = 1024, 16
KINDS = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "sliding_attention"]


def _classes():
    try:
        from transformers.models.lfm3 import modeling_lfm3
        return modeling_lfm3.Lfm3Config, modeling_lfm3.Lfm3Model
    except ImportError:
        pass
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        cls = [get_class_from_dynamic_module(ref, CODE_REPO, revision=CODE_REVISION)
               for ref in ("configuration_lfm3.Lfm3Config", "modeling_lfm3.Lfm3Model")]
    except Exception as exc:  # noqa: BLE001 — offline, or the Hub unreachable
        pytest.skip(f"LFM3 modeling code unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    return tuple(cls)


def _config(**overrides):
    Config, _ = _classes()
    cfg = dict(vocab_size=VOCAB, hidden_size=256, intermediate_size=768, num_hidden_layers=len(KINDS),
               num_attention_heads=4, num_key_value_heads=2, layer_types=list(KINDS), sliding_window=WINDOW,
               max_position_embeddings=4096, use_cache=False, pad_token_id=None, bos_token_id=None,
               eos_token_id=None)
    cfg.update(overrides)
    return Config(**cfg)


def _randomize(model):
    torch.manual_seed(0)
    for p in model.parameters():                     # random init is too small to exercise the kernels
        if p.dim() > 1:
            p.data.normal_(0, 0.05)
        else:
            p.data.uniform_(0.5, 1.5)                # norm weights away from 1: catches a missing scale
    return model


def _model(**overrides):
    _, Model = _classes()
    return _randomize(Model(_config(**overrides))).to("cuda", torch.bfloat16).eval()


@pytest.fixture(scope="module")
def tiny():
    return _model()


def _hf_hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None], use_cache=False).last_hidden_state[0].float()


def _cos(got, ref):
    return F.cosine_similarity(got.float(), ref.float(), dim=-1)


def _ids(n, seed):
    return torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).cuda()


# ---------------------------------------------------------------------------- engine


@pytest.mark.parametrize("lengths", [[3, 70, 1, 17, 40], [600, 90, 2]], ids=["short", "long"])
def test_engine_matches_hf_per_sequence_packed_and_graphed(tiny, lengths):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    engine = Lfm2Engine(tiny, family="LFM3")
    try:
        assert engine.sandwich and engine.conv_width == 7
        assert engine.attentions["sliding_attention"].window == (WINDOW - 1, 0)
        assert all(L.gated for L in engine.layers if L.attn)
        seqs = [_ids(n, i) for i, n in enumerate(lengths)]
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        got = engine.forward_packed(torch.cat(seqs), lengths).float()
        assert _cos(got, ref).min() > 0.99
        runner = PaddedGraphRunner(engine, PaddedGraphConfig(row_buckets=(len(lengths),), max_graphs=1))
        graphed = runner(torch.cat(seqs), lengths)
        assert graphed is not None and _cos(graphed, ref).min() > 0.99
    finally:
        engine.release(rollback=True)


def test_release_restores_independent_weights(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    before = {n: p.detach().clone() for n, p in tiny.named_parameters()}
    Lfm2Engine(tiny, family="LFM3").release(rollback=True)
    ptrs = [p.untyped_storage().data_ptr() for p in tiny.parameters()]
    assert len(set(ptrs)) == len(ptrs)
    assert all(torch.equal(before[n], p) for n, p in tiny.named_parameters())


def test_fused_projections_are_refused_before_mutation(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine
    from packed_encoders.errors import UnsupportedTargetError

    class Part(nn.Module):                           # a projection without its own weight
        def forward(self, x):
            return x

    attn = tiny.layers[KINDS.index("sliding_attention")].self_attn
    original = attn.gate_proj
    attn.gate_proj = Part()
    ptrs = [p.data_ptr() for p in tiny.parameters()]
    try:
        with pytest.raises(UnsupportedTargetError, match="weight as a parameter"):
            Lfm2Engine(tiny, family="LFM3")
        assert [p.data_ptr() for p in tiny.parameters()] == ptrs
    finally:
        attn.gate_proj = original


# ---------------------------------------------------------------------------- grouped rows


def test_grouped_rows_match_the_model_under_the_same_masks(tiny):
    from packed_encoders.arch.lfm2 import grouped_attention_masks
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    contexts, groups = [40, 25, 9, 3], [[7, 1, 12], [], [20], [5, 30, 2]]
    lengths = [c + sum(g) for c, g in zip(contexts, groups)]
    seqs = [_ids(n, 10 + i) for i, n in enumerate(lengths)]
    ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    real = (torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]).cuda()
    masks = grouped_attention_masks(tiny.config, lengths, groups, ids.shape[1], torch.bfloat16, "cuda")
    engine = Lfm2Engine(tiny, family="LFM3")
    try:
        with torch.no_grad():
            ref = tiny(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state[real]
            causal = tiny(input_ids=ids, attention_mask=real.long(), use_cache=False).last_hidden_state[real]
        got = engine.forward_grouped(torch.cat(seqs), lengths, groups)
        assert _cos(got, ref).min() > 0.99
        assert _cos(got, causal).min() < 0.9          # grouped rows are not the causal forward
    finally:
        engine.release(rollback=True)


def test_one_full_attention_layer_reads_the_context_and_its_own_group_only():
    """An independent check of the grouped pattern, without a mask: through one full-attention
    layer (the rest is per token), a group reads its context and itself, both ways, and nothing
    else, while the context reads every group."""
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    model = _model(num_hidden_layers=1, layer_types=["full_attention"])
    engine = Lfm2Engine(model, family="LFM3")
    try:
        context, groups = _ids(30, 1), [_ids(n, 2 + n) for n in (5, 9, 4)]
        sizes = [[len(g) for g in groups]]
        row = torch.cat([context, *groups])
        n, c = len(row), len(context)
        got = engine.forward_grouped(row, [n], sizes)
        # the first group, at the same positions, reads exactly what the row of it alone reads
        alone = engine.forward_grouped(row[:c + 5], [c + 5], [[5]])
        assert _cos(got[c:c + 5], alone[c:]).min() > 0.9999
        # changing the other groups leaves the second group untouched, and changes the context
        other = torch.cat([context, _ids(5, 98), groups[1], _ids(4, 99)])
        changed = engine.forward_grouped(other, [n], sizes)
        assert _cos(got[c + 5:c + 14], changed[c + 5:c + 14]).min() > 0.9999
        assert _cos(got[:c], changed[:c]).min() < 0.999
        # the group reads itself both ways: its first token hears its last
        last = torch.cat([context, groups[0], groups[1][:-1], _ids(1, 97), groups[2]])
        assert _cos(got[c + 5], engine.forward_grouped(last, [n], sizes)[c + 5]) < 0.9999
        # a lone group and a row of context alone both read all of themselves
        assert _cos(engine.forward_grouped(row, [n], [[n - 2]]), engine.forward_grouped(row, [n], [[]])).min() > 0.9999
    finally:
        engine.release(rollback=True)


def test_forward_grouped_checks_its_arguments(tiny):
    import packed_encoders as pe
    from packed_encoders.batch import PackedBatch
    from packed_encoders.errors import PackedEncodersError

    pe.pack(tiny, cuda_graph=False)
    try:
        packed = pe.get_engine(tiny)
        assert packed.state.report.grouped_rejected is None and packed.state.report.grouped_cos_min > 0.85
        ids = _ids(20, 0)
        batch = PackedBatch(input_ids=ids, host_lengths=(12, 8))
        with torch.no_grad():
            ours = packed.forward_grouped(batch, [[4, 3], [8]])
            emb = packed.forward_grouped(PackedBatch(input_ids=None, host_lengths=(12, 8)), [[4, 3], [8]],
                                         inputs_embeds=F.embedding(ids, tiny.embed_tokens.weight))
            assert _cos(ours, emb).min() > 0.999
            for bad in ([[4, 9], [8]], [[4, 3]], [[0], [8]], [[4.0], [8]]):
                with pytest.raises(PackedEncodersError):
                    packed.forward_grouped(batch, bad)
        with pytest.raises(PackedEncodersError, match="no_grad"):
            packed.forward_grouped(batch, [[4, 3], [8]])
    finally:
        pe.unpack(tiny)


# ---------------------------------------------------------------------------- HF entry


class _EmbedsWrapper(nn.Module):
    """A multimodal-style wrapper: feeds its language model input embeddings."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm

    def forward(self, input_ids, attention_mask):
        embeds = self.language_model.embed_tokens(input_ids)
        return self.language_model(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False)


@pytest.mark.parametrize("wrapped", [False, True], ids=["backbone", "wrapper"])
def test_hf_entry_packs_and_keeps_the_contract(wrapped):
    import packed_encoders as pe

    lm = _model()
    model = _EmbedsWrapper(lm) if wrapped else lm
    rows = [_ids(n, 20 + n) for n in (5, 40, 23)]
    S = max(map(len, rows))
    right = torch.zeros(len(rows), S, dtype=torch.long, device="cuda")
    left = torch.zeros_like(right)
    masks = {"right": torch.zeros_like(right, dtype=torch.bool), "left": torch.zeros_like(right, dtype=torch.bool)}
    for i, x in enumerate(rows):
        right[i, :len(x)], left[i, S - len(x):] = x, x
        masks["right"][i, :len(x)], masks["left"][i, S - len(x):] = True, True

    def run(ids, mask):
        with torch.no_grad():
            if wrapped:
                return model(ids, mask.long()).last_hidden_state
            return model(input_ids=ids, attention_mask=mask.long(), use_cache=False).last_hidden_state

    stock = {"right": run(right, masks["right"]), "left": run(left, masks["left"]),
             "single-left": run(left[:1], masks["left"][:1])}
    pe.pack(model)
    try:
        packed = pe.get_engine(model)
        assert type(packed.owner).__name__ == "Lfm3Hybrid"
        calls = []
        engine = packed.state.engine
        forward_packed = engine.forward_packed
        engine.forward_packed = lambda *a, **k: calls.append(1) or forward_packed(*a, **k)
        got = {"right": run(right, masks["right"]), "left": run(left, masks["left"]),
               "single-left": run(left[:1], masks["left"][:1])}
        engine.forward_packed = forward_packed
        for name, ids_mask in (("right", masks["right"]), ("left", masks["left"]), ("single-left", masks["left"][:1])):
            assert _cos(got[name][ids_mask], stock[name][ids_mask]).min() > 0.99, name
            assert (got[name][~ids_mask] == 0).all(), name
        if wrapped:                                   # embeddings run the eager engine, a lone left-padded row too
            assert len(calls) == 3
    finally:
        pe.unpack(model)


def test_a_prepared_mask_runs_the_original_forward(tiny):
    import packed_encoders as pe
    from packed_encoders.arch.lfm2 import grouped_attention_masks

    lengths, groups = [30, 12], [[6, 4], []]
    ids = torch.nn.utils.rnn.pad_sequence([_ids(n, n) for n in lengths], batch_first=True)
    masks = grouped_attention_masks(tiny.config, lengths, groups, ids.shape[1], torch.bfloat16, "cuda")
    with torch.no_grad():
        stock = tiny(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state
    pe.pack(tiny, cuda_graph=False)
    try:
        engine = pe.get_engine(tiny).state.engine
        engine.forward_packed = None                  # any engine call would fail
        with pytest.warns(UserWarning, match="original forward"), torch.no_grad():
            got = tiny(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state
        assert torch.equal(got, stock)
    finally:
        del engine.forward_packed
        pe.unpack(tiny)


def test_shared_prefixes_past_the_window_match_full_rows(tiny):
    from packed_encoders.arch.lfm2.engine import Lfm2Engine

    engine = Lfm2Engine(tiny, family="LFM3")
    try:
        engine.min_shared_prefix = 24
        state = _ids(3 * WINDOW + 5, 5)                 # the children's sliding layers reach back past the root's start
        seqs = [torch.cat([state, _ids(n, 40 + n)]) for n in (1, 6, 30)] + [_ids(50, 7)]
        lengths = [len(s) for s in seqs]
        ids = torch.cat(seqs)
        plan = engine.plan_sharing(ids, lengths)
        assert plan is not None and plan.saved_tokens > 0
        got = engine.forward_shared(ids, plan).float()
        ref = engine.forward_packed(ids, lengths).float()
        assert _cos(got, ref).min() > 0.99
        assert _cos(got, torch.cat([_hf_hidden(tiny, s) for s in seqs])).min() > 0.99
    finally:
        engine.release(rollback=True)
