import copy

import pytest
import torch
import torch.nn.functional as F

import packed_encoders as pe
from packed_encoders.arch.qwen3_5.prefix_graphs import SharedGraphRunner
from packed_encoders.runtime.graphs import PaddedGraphConfig
from test_qwen35 import tiny
from test_qwen35_sharing import _batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("backend", ["flash", "sdpa"])
def test_shared_graph_changed_inputs_topology_eviction_and_output_lifetime(tiny, backend):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False, attention_backend=backend)
    packed = pe.get_engine(model)
    engine = packed.state.engine
    packed.min_shared_prefix = 64
    runner = SharedGraphRunner(engine, PaddedGraphConfig(max_graphs=2))
    with torch.no_grad():
        previous = None
        for seed, order in [(0, None), (1, None), (2, [0, 2, 1, 3, 4, 5, 6]),
                            (3, [6, 5, 4, 3, 2, 1, 0]), (4, None)]:
            rows, _, _ = _batch(seed)
            if order:
                rows = [rows[i] for i in order]
            ids, lengths = torch.cat(rows).cuda(), [len(r) for r in rows]
            plan = engine.plan_sharing(ids, lengths)
            expected = engine.forward_shared(ids, plan).float()
            actual = runner(ids, plan)
            assert F.cosine_similarity(actual.float(), expected, dim=-1).min() > .999
            assert runner.num_graphs <= 2
            if seed == 1:
                assert runner.num_graphs == 1  # different token values, identical plan
            if previous is not None:
                assert torch.equal(previous, saved)
            previous, saved = actual, actual.clone()
        runner.config = PaddedGraphConfig(max_tokens=1)
        assert runner(ids, plan) is None
    del runner
    pe.unpack(model)


def test_sharing_graph_flags_and_oom_recovery(tiny, monkeypatch):
    from packed_encoders.arch.qwen3_5 import _hidden

    model = copy.deepcopy(tiny)
    pe.pack(model)
    packed = pe.get_engine(model)
    packed.min_shared_prefix = 64
    _, ids, lengths = _batch()
    ids = ids.cuda()
    state = packed.state
    with torch.no_grad():
        with pe.no_cuda_graph(model):
            expected = _hidden(state, ids, lengths, graphs=True)
        assert state.shared_runner is None
        actual = _hidden(state, ids, lengths, graphs=True)
        torch.testing.assert_close(actual, expected)
        assert state.shared_runner.num_graphs == 1
        monkeypatch.setenv("PACKED_ENCODERS_GRAPH", "0")
        with monkeypatch.context() as patch:
            patch.setattr(SharedGraphRunner, "__call__", lambda *a: pytest.fail("graphs disabled"))
            _hidden(state, ids, lengths, graphs=True)
        monkeypatch.delenv("PACKED_ENCODERS_GRAPH")
        def oom(*args):
            raise torch.OutOfMemoryError("injected shared capture OOM")
        monkeypatch.setattr(SharedGraphRunner, "__call__", oom)
        with pytest.warns(UserWarning, match="dropped"):
            actual = _hidden(state, ids, lengths, graphs=True)
        torch.testing.assert_close(actual, expected)
        assert state.shared_runner is None and state.runner is None and not state.graph_enabled
    pe.unpack(model)
