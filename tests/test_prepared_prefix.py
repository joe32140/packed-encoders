import copy

import pytest
import torch
import torch.nn.functional as F

import packed_encoders as pe
from test_qwen35 import tiny

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("prefix_len", [1, 3, 64, 129])
@pytest.mark.parametrize("backend", ["cute", "torch"])
def test_prepared_prefix_matches_full_rows_and_is_reusable(tiny, prefix_len, backend):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    engine = packed.state.engine
    if backend == "torch":
        engine._prefix_conv = None
        engine.fused = False
    g = torch.Generator().manual_seed(42)
    prefix = torch.randint(0, 1024, (prefix_len,), generator=g).cuda()
    with torch.no_grad():
        cache = packed.prepare_prefix(prefix)
        snapshot = [(c.state.clone() if c.state is not None else c.k.clone()) for c in cache.layers]
        assert cache.nbytes > 0
        for lengths in ([1, 9, 130], [65, 257], [1]):
            seqs = [torch.randint(0, 1024, (n,), generator=g).cuda() for n in lengths]
            got = cache.forward_suffixes(torch.cat(seqs), lengths).float()
            ref = torch.cat([packed.state.original_forward(torch.cat([prefix, s])[None], use_cache=False)
                             .last_hidden_state[0, prefix_len:].float() for s in seqs])
            cos = F.cosine_similarity(got, ref, dim=-1)
            assert cos.mean() > 0.999 and cos.min() > 0.99
        for c, before in zip(cache.layers, snapshot):
            assert torch.equal(c.state if c.state is not None else c.k, before)
        original = packed.state.original_forward(prefix[None], use_cache=False).last_hidden_state[0]
        assert F.cosine_similarity(cache.hidden_states.float(), original.float(), dim=-1).min() > 0.99
    pe.unpack(model)
    assert cache.nbytes == 0 and cache.hidden_states is None and not cache.layers
    with pytest.raises(pe.PackedEncodersError, match="closed"):
        cache.forward_suffixes(prefix, [prefix_len])


def test_prepared_prefix_budget_and_weight_invalidation(tiny):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    ids = torch.arange(65, device="cuda")
    with torch.no_grad():
        with pytest.raises(pe.PackedEncodersError, match="budget"):
            packed.prepare_prefix(ids, max_bytes=1)
        cache = packed.prepare_prefix(ids)
        model.norm.weight.add_(0.01)
        with pytest.raises(pe.PackedEncodersError, match="weights changed"):
            cache.forward_suffixes(ids, [65])
        assert cache.nbytes == 0
        with packed.prepare_prefix(ids) as fresh:
            assert fresh.forward_suffixes(ids, [65]).shape == (65, 256)
        assert fresh.nbytes == 0
    pe.unpack(model)


def test_prepared_prefix_rejects_invalid_inputs_and_grad(tiny):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    ids = torch.arange(10, device="cuda")
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        packed.prepare_prefix(ids)
    with torch.no_grad():
        with packed.prepare_prefix(ids) as cache:
            for bad, lengths in [(ids[None], [10]), (ids.int(), [10]), (ids, [0, 10]),
                                 (ids, [9]), (ids[:0], []), (ids.cpu(), [10])]:
                with pytest.raises(pe.PackedEncodersError):
                    cache.forward_suffixes(bad, lengths)
            with torch.autocast("cuda", dtype=torch.float16):
                with pytest.raises(pe.PackedEncodersError, match="autocast"):
                    cache.forward_suffixes(ids, [10])
        with pytest.raises(pe.PackedEncodersError, match="closed"):
            cache.forward_suffixes(ids, [10])
    pe.unpack(model)


@pytest.mark.parametrize("lengths", [[1, 9, 130], [65, 257]])
@pytest.mark.parametrize("backend", ["cute", "torch"])
def test_suffix_graph_replays_new_tokens_and_closes(tiny, lengths, backend):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    if backend == "torch":
        packed.state.engine._prefix_conv = None
        packed.state.engine.fused = False
    with torch.no_grad():
        prefix = packed.prepare_prefix(torch.arange(65, device="cuda"))
        graph = prefix.capture_suffixes(lengths)
        previous = None
        for _ in range(3):
            # More distinct layouts than FLA's four-entry metadata cache. The
            # graph must own its chunk indices after those entries are evicted.
            for n in range(5, 11):
                prefix.forward_suffixes(torch.arange(n, device="cuda"), [n])
            ids = torch.randint(0, 1024, (sum(lengths),), device="cuda")
            expected = prefix.forward_suffixes(ids, lengths).float()
            actual = graph(ids)
            assert F.cosine_similarity(actual.float(), expected, dim=-1).min() > .999
            if previous is not None:
                assert torch.equal(previous, saved)
            previous, saved = actual, actual.clone()
        model.norm.weight.add_(.01)
        with pytest.raises(pe.PackedEncodersError, match="weights changed"):
            graph(ids)
        with pytest.raises(pe.PackedEncodersError, match="closed"):
            graph(ids)
    pe.unpack(model)


def test_suffix_graph_validation_and_prefix_lifetime(tiny):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    ids = torch.arange(10, device="cuda")
    with torch.no_grad():
        prefix = packed.prepare_prefix(ids)
        for lengths in ([], [0], [-1], [1.0], [True]):
            with pytest.raises(pe.PackedEncodersError, match="positive host"):
                prefix.capture_suffixes(lengths)
        graph = prefix.capture_suffixes([10])
        for bad in (ids[:9], ids.int(), ids.cpu(), ids[None]):
            with pytest.raises(pe.PackedEncodersError):
                graph(bad)
        with torch.autocast("cuda", dtype=torch.float16):
            with pytest.raises(pe.PackedEncodersError, match="autocast"):
                graph(ids)
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        graph(ids)
    with pytest.raises(pe.PackedEncodersError, match="no_grad"):
        prefix.capture_suffixes([10])
    pe.unpack(model)
    with torch.no_grad(), pytest.raises(pe.PackedEncodersError, match="closed"):
        graph(ids)
