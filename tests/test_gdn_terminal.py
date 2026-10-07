"""Terminal continuation outputs need no recurrent state allocation or writes."""
import copy
import pytest
import torch

import packed_encoders as pe
from packed_encoders.pieces.hybrid import gdn_resume_piece
from test_qwen35 import tiny

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('lengths', [[1, 7, 65], [63, 128, 257]])
@pytest.mark.parametrize('has_initial', [False, True])
@torch.no_grad()
def test_terminal_output_matches_stateful_and_reference(lengths, has_initial, monkeypatch):
    import fla.ops.gated_delta_rule as module
    original = module.chunk_gated_delta_rule
    calls = []
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs['output_final_state'], result[1] is None))
        return result
    monkeypatch.setattr(module, 'chunk_gated_delta_rule', record)
    piece = gdn_resume_piece()
    torch.manual_seed(81)
    n = sum(lengths)
    def rand(*shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    q, k = rand(1, n, 2, 32), rand(1, n, 2, 32)
    v, a, b = rand(1, n, 4, 32), rand(1, n, 4), rand(1, n, 4)
    log, bias = torch.randn(4, device='cuda'), torch.randn(4, device='cuda')
    cpu = torch.tensor([0]+lengths).cumsum(0)
    initial = torch.randn(3, 4, 32, 32, device='cuda') * .1 if has_initial else None
    saved = initial.clone() if has_initial else None
    args = (q, k, v, a, b, log, bias, initial, cpu.cuda(), cpu)
    piece.validate(*args, output_final_state=False)
    out, final = piece.execute(*args)
    terminal = piece.execute(*args, output_final_state=False)
    assert torch.equal(terminal, out)
    assert final.shape == (3, 4, 32, 32)
    assert calls == [(False, True), (True, False), (False, True)]
    if has_initial:
        assert torch.equal(initial, saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = piece.execute(*args, output_final_state=False)
    for _ in range(2):
        q.normal_()
        graph.replay()
        expected, _ = piece.execute(*args)
        assert torch.equal(captured, expected)


@torch.no_grad()
def test_engine_requests_state_only_for_prefixes_and_roots(tiny):
    model = copy.deepcopy(tiny)
    pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    engine = packed.state.engine
    original, calls = engine.ops.gdn_resume, []
    def record(*args, **kwargs):
        calls.append(kwargs.get('output_final_state', True))
        return original(*args, **kwargs)
    engine.ops.gdn_resume = record
    count = sum(layer.linear for layer in engine.layers)
    ids = torch.arange(65, device='cuda')
    with packed.prepare_prefix(ids) as prefix:
        assert calls == [True] * count
        calls.clear()
        prefix.forward_suffixes(ids, [1, 64])
        assert calls == [False] * count
    calls.clear()
    packed.min_shared_prefix = 4
    rows = [torch.cat([ids, torch.tensor([i], device='cuda')]) for i in (100, 101)]
    flat = torch.cat(rows)
    engine.forward_shared(flat, engine.plan_sharing(flat, [66, 66]))
    assert calls == [True, False] * count
    pe.unpack(model)


def test_terminal_probe_failure_disables_only_sharing(tiny, monkeypatch):
    from packed_encoders.pieces.base import Piece
    original = Piece.validate
    def validate(piece, *args, **kwargs):
        if piece.contract.operation == 'gated_delta_rule_resume' and kwargs.get('output_final_state') is False:
            raise RuntimeError('injected terminal probe failure')
        return original(piece, *args, **kwargs)
    monkeypatch.setattr(Piece, 'validate', validate)
    model = copy.deepcopy(tiny)
    with pytest.warns(UserWarning, match='every row runs in full'):
        pe.pack(model, cuda_graph=False)
    packed = pe.get_engine(model)
    assert 'injected terminal probe failure' in packed.state.engine.share_rejected
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(9, device='cuda')[None], use_cache=False).last_hidden_state).all()
    pe.unpack(model)
