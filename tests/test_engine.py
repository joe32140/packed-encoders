"""ModernBERT's prepared boundary, controls and lifecycle on the real CUDA path."""

import pytest
import torch

import packed_encoders as pe
from packed_encoders.errors import PackedEncodersError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def model():
    from transformers import AutoModel

    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", dtype=torch.bfloat16).cuda().eval()
    yield model
    pe.unpack(model)


def batch(model, lengths=(7, 29, 64)):
    gen = torch.Generator(device="cuda").manual_seed(17)
    ids = torch.randint(5, model.config.vocab_size, (sum(lengths),), device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    return pe.PackedBatch(ids, cu, max(lengths), pos)


def test_prepared_boundary_replays_and_controls_both_entry_points(model):
    from packed_encoders.forward import packed_forward

    original = model.forward
    pe.pack(model, attention_backend="flash", validate=False,
            cuda_graph=pe.GraphConfig(max_batch=8, max_seq=128, pad_to=32))
    engine = pe.get_engine(model)
    b = batch(model)
    state = engine.state
    with torch.inference_mode():
        actual = engine.forward_packed(b).clone()
        direct = packed_forward(model, state.params, b.input_ids, b.cu_seqlens, b.max_seqlen, b.position_ids)
        torch.testing.assert_close(actual, direct, rtol=0, atol=0)
        b2 = batch(model, (9, 27, 64))  # same token bucket, changed boundaries
        second = engine.forward_packed(b2).clone()
        with pe.no_cuda_graph(model):
            eager = engine.forward_packed(b2)
        torch.testing.assert_close(second, eager, rtol=0.03, atol=0.04)
        runner = state.packed_graph_runner

        class Forbidden:
            def __call__(self, *args):
                raise AssertionError("disabled graph was replayed")

        state.packed_graph_runner = Forbidden()
        with pe.no_cuda_graph(model):
            engine.forward_packed(b)
            padded = b.input_ids.new_zeros(3, 64)
            mask = torch.zeros_like(padded)
            off = 0
            for row, length in enumerate((7, 29, 64)):
                padded[row, :length] = b.input_ids[off:off + length]
                mask[row, :length] = 1
                off += length
            model(padded, mask)
        state.packed_graph_runner = runner
        pe.set_cuda_graph(model, False)
        assert not state.graph_enabled
        pe.set_cuda_graph(model, True)
        assert state.graph_enabled
    pe.unpack(model)
    assert model.forward == original
    assert state.packed_graph_runner is None
    with pytest.raises(PackedEncodersError, match="closed"):
        engine.forward_packed(b)


def test_validation_uses_saved_oracle_and_restores_updated_weights(model):
    from packed_encoders.arch.modernbert import ModernBert

    original = model.forward
    weight = next(model.parameters())
    ptr = weight.data_ptr()
    pe.pack(model, engine=ModernBert())
    engine = pe.get_engine(model)
    assert engine.capabilities.training and engine.capabilities.training_capture
    assert set(engine.state.validation_report.cosine) == {128, 512, 2048}
    assert len(engine.state.validation_report.pieces) == 9
    assert all(report.skipped_reason is None for report in engine.state.validation_report.pieces.values())
    assert not engine.graph_enabled
    result = engine.validate(seq_lens=(32,))
    assert result.engine == "modernbert" and result.details.cosine[32] > 0.997
    assert pe.validate(model, seq_lens=(32,)).cosine[32] > 0.997
    with torch.no_grad():
        weight.add_(0.01)
    updated = weight.detach().clone()
    pe.unpack(model)
    assert model.forward == original and next(model.parameters()) is weight
    assert weight.data_ptr() == ptr
    torch.testing.assert_close(weight, updated, rtol=0, atol=0)


def test_prepared_boundary_preserves_backward(model):
    from packed_encoders.forward import packed_forward

    pe.pack(model, attention_backend="flash", validate=False)
    engine = pe.get_engine(model)
    b = batch(model, (8, 16))
    model.train()
    output = engine.forward_packed(b)
    gradient = torch.ones_like(output)
    weights = tuple(model.parameters())
    actual = torch.autograd.grad(output, weights, gradient)
    reference = packed_forward(model, engine.state.params, b.input_ids, b.cu_seqlens, b.max_seqlen, b.position_ids)
    expected = torch.autograd.grad(reference, weights, gradient)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
