"""Requested graph modes must not obscure the benchmark's actual execution path."""
from types import SimpleNamespace

import pytest

from benchmarks.hf_forward_bench import record_execution
from packed_encoders.runtime.shared_graphs import SharedGraphRunner


@pytest.mark.parametrize("path", ["packed_eager", "shared_eager", "packed_graph", "shared_graph", "hf_fallback"])
def test_execution_probe_distinguishes_replay_and_eager_fallback(monkeypatch, path):
    output = object()
    engine = SimpleNamespace(forward_packed=lambda: output, forward_shared=lambda: output)
    runner = SimpleNamespace(_replay=lambda: output)
    state = SimpleNamespace(engine=engine, runner=runner, original_forward=lambda: output)
    packed = SimpleNamespace(state=state)
    monkeypatch.setattr(SharedGraphRunner, "__call__", lambda *args: output if path == "shared_graph" else None)
    shared = SharedGraphRunner(engine, None)
    originals = (engine.forward_packed, engine.forward_shared, runner._replay, state.original_forward)

    def forward():
        if path in ("shared_graph", "shared_eager"):
            return shared(None, None) or engine.forward_shared()
        if path == "packed_graph":
            return runner._replay()
        if path == "hf_fallback":
            return state.original_forward()
        return engine.forward_packed()

    actual, execution = record_execution(packed, forward)
    assert actual is output
    assert execution["path"] == path
    assert execution["calls"][path] == 1
    assert sum(execution["calls"].values()) == 1
    assert (engine.forward_packed, engine.forward_shared, runner._replay, state.original_forward) == originals
