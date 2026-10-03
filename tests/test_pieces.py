"""Executable pieces: semantic rejection, independent references and real reuse."""

from collections import Counter
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from packed_encoders.errors import UnsupportedTargetError, ValidationError
from packed_encoders.pieces import Contract, Piece

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_piece_validator_rejects_wrong_execution():
    piece = Piece("bad", Contract("identity", "T -> T", "identity"),
                  lambda x: x + 1, lambda x: x, lambda x: None)
    with pytest.raises(ValidationError, match="bad.*failed reference"):
        piece.validate(torch.ones(4))


def test_incompatible_rope_convention_is_rejected_before_binding():
    from packed_encoders.arch.modernbert_pieces import default_pieces

    pieces = default_pieces()
    incompatible = replace(pieces.rope, contract=replace(pieces.rope.contract, semantics="interleaved pairs"))
    with pytest.raises(UnsupportedTargetError, match="incompatible rope"):
        replace(pieces, rope=incompatible).bind()


def test_rope_table_preparation_matches_analytic_angles():
    from packed_encoders.pieces.rope import prepare_tables

    cos, sin = prepare_tables(4, 4, 10000, "cpu", torch.float32)
    angles = torch.tensor([[0, 0, 0, 0], [1, .01, 1, .01], [2, .02, 2, .02], [3, .03, 3, .03]])
    torch.testing.assert_close(cos[0], angles.cos())
    torch.testing.assert_close(sin[0], angles.sin())
    with pytest.raises(UnsupportedTargetError):
        prepare_tables(4, 3, 10000, "cpu", torch.float32)


@needs_cuda
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_rope_piece_forward_backward_and_packed_positions(head_dim):
    from packed_encoders.pieces import split_half_rope

    torch.manual_seed(19)
    piece = split_half_rope()
    # Deliberately nonsequential positions, independent of any ModernBERT config.
    cos, sin = piece.prepare(64, head_dim, 10000, "cuda", torch.bfloat16)
    pos = torch.tensor([0, 4, 9, 0, 1, 31], device="cuda")
    cos, sin = cos[0, pos], sin[0, pos]
    q = torch.randn(1, 8, 6, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    assert piece.validate(q, k, cos, sin).max_abs_error < .04
    dq, dk = torch.randn_like(q), torch.randn_like(k)
    actual = torch.autograd.grad(piece.execute(q, k, cos, sin), (q, k), (dq, dk))
    expected = torch.autograd.grad(piece.reference(q, k, cos, sin), (q, k), (dq, dk))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=.03, atol=.03)
    with pytest.raises(UnsupportedTargetError, match="equal-shape"):
        piece.validate(q, k[:, :4], cos, sin)
    with pytest.raises(UnsupportedTargetError, match="full split-half"):
        piece.validate(q, k, cos[:, :head_dim // 2], sin[:, :head_dim // 2])


class CausalBlock(nn.Module):
    """A second consumer: decoder attention, different heads/theta/position schedule.

    Receives a RoPE piece; it has no ModernBERT dependencies or model adapter.
    """
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.qkv = nn.Linear(512, 1536, bias=False, device="cuda", dtype=torch.bfloat16)

    def forward(self, x, cos, sin, *, reference=False):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, 8, 64).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        rotate = self.rope.reference if reference else self.rope.execute
        q, k = rotate(q, k, cos, sin)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@needs_cuda
def test_same_rope_instance_is_used_by_modernbert_and_a_causal_block():
    import packed_encoders as pe
    from packed_encoders.arch.modernbert import ModernBert
    from packed_encoders.arch.modernbert_pieces import default_pieces

    pieces = default_pieces()
    block = CausalBlock(pieces.rope)
    model = tiny_model()
    try:
        pe.pack(model, engine=ModernBert(pieces=pieces), attention_backend="sdpa", validate=False)
        assert pe.get_engine(model).composition.rope is block.rope
        with torch.no_grad():
            model(torch.randint(5, 128, (1, 16), device="cuda"))
    finally:
        pe.unpack(model)
    cos, sin = block.rope.prepare(32, 64, 10000, "cuda", torch.bfloat16)
    x = torch.randn(2, 32, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual, expected = block(x, cos, sin), block(x, cos, sin, reference=True)
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.02)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, block.qkv.weight), grad)
    reference_grads = torch.autograd.grad(expected, (x, block.qkv.weight), grad)
    for a, e in zip(actual_grads, reference_grads):
        torch.testing.assert_close(a, e, rtol=.04, atol=.04)


def tiny_model():
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                          num_attention_heads=4, local_attention=128, vocab_size=128,
                          pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg.global_attn_every_n_layers = 3
    return ModernBertModel(cfg).cuda().to(torch.bfloat16).eval()


@needs_cuda
def test_selected_operations_reach_eager_capture_and_training():
    import packed_encoders as pe
    from packed_encoders.arch.modernbert import ModernBert
    from packed_encoders.arch.modernbert_pieces import default_pieces

    calls = Counter()
    pieces = default_pieces()

    def counted(name, execute):
        def run(*args, **kwargs):
            calls[name] += 1
            return execute(*args, **kwargs)
        return run

    selected = replace(pieces, **{
        name: replace(getattr(pieces, name), execute=counted(name, getattr(pieces, name).execute))
        for name in pieces.__dataclass_fields__
    })
    model = tiny_model()
    engine = ModernBert(pieces=selected)
    try:
        pe.pack(model, engine=engine, attention_backend="flash", validate=False,
                cuda_graph=pe.GraphConfig(max_batch=2, max_seq=64, pad_to=32))
        packed = pe.get_engine(model)
        assert packed.composition is selected
        ids = torch.randint(5, 128, (24,), device="cuda")
        batch = pe.PackedBatch(ids, torch.tensor([0, 8, 24], dtype=torch.int32, device="cuda"),
                               16, torch.cat((torch.arange(8), torch.arange(16))).cuda())
        with torch.inference_mode():
            packed.forward_packed(batch)  # warmup + capture uses selected pieces
            assert all(calls[name] for name in ("rope_qkv", "attention_bshd", "linear", "dense_linear",
                                                "layer_norm", "add_layer_norm", "geglu"))
            before = dict(calls)
            packed.forward_packed(batch)  # replay does not interpret pieces in Python
            assert dict(calls) == before
            with pe.no_cuda_graph(model):
                packed.forward_packed(batch)
            assert calls["rope_qkv"] > before["rope_qkv"]
        model.train()
        packed.forward_packed(batch).float().square().sum().backward()
        assert calls["rope"] and calls["attention"]
        assert model.layers[0].attn.Wqkv.weight.grad is not None
    finally:
        pe.unpack(model)

    model = tiny_model().train()
    try:
        pe.pack(model, engine=engine, attention_backend="sdpa",
                train_cuda_graph=True, validate=False)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        ids = torch.randint(5, 128, (1, 16), device="cuda")
        calls.clear()
        model(ids).last_hidden_state.float().square().sum().backward()
        assert calls["rope"] and calls["attention"]
        assert pe.get_engine(model).state.train_graph_runner._cache
        before = dict(calls)
        model.zero_grad(set_to_none=False)
        model(ids).last_hidden_state.float().square().sum().backward()
        assert dict(calls) == before
    finally:
        pe.unpack(model)


@needs_cuda
def test_bad_selected_piece_fails_pack_and_leaves_original_model():
    import packed_encoders as pe
    from packed_encoders.arch.modernbert import ModernBert
    from packed_encoders.arch.modernbert_pieces import default_pieces
    from packed_encoders.state import ATTR, INSTALL_ATTR

    pieces = default_pieces()
    bad = replace(pieces.rope, execute=lambda q, k, cos, sin: (torch.zeros_like(q), torch.zeros_like(k)))
    model = tiny_model()
    original = model.forward
    with pytest.raises(ValidationError, match="rope.*failed reference"):
        pe.pack(model, engine=ModernBert(pieces=replace(pieces, rope=bad)))
    assert model.forward == original
    assert not hasattr(model, ATTR) and not hasattr(model, INSTALL_ATTR)


@needs_cuda
@pytest.mark.parametrize("slot", ["layer_norm", "add_layer_norm", "linear", "geglu"])
def test_numerical_piece_gradients_match_independent_reference(slot):
    from packed_encoders.arch.modernbert_pieces import default_pieces

    torch.manual_seed(23)
    piece = getattr(default_pieces(), slot)
    x = torch.randn(2, 7, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if slot == "geglu":
        args = (x,)
    elif slot == "linear":
        args = (x, torch.randn(128, 256, device="cuda", dtype=x.dtype, requires_grad=True) * .03)
    else:
        weight = torch.randn(256, device="cuda", dtype=x.dtype, requires_grad=True)
        args = (x, weight, 1e-5) if slot == "layer_norm" else (
            x, torch.randn_like(x, requires_grad=True), weight, 1e-5)
    reference_args = tuple(a.detach().clone().requires_grad_(a.requires_grad) if isinstance(a, torch.Tensor) else a
                           for a in args)
    expected = piece.reference(*reference_args)
    actual = piece.execute(*args)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    gradients = tuple(torch.randn_like(x) for x in actual)
    ga = torch.autograd.grad(actual, tuple(a for a in args if isinstance(a, torch.Tensor)), gradients)
    ge = torch.autograd.grad(expected, tuple(a for a in reference_args if isinstance(a, torch.Tensor)), gradients)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=.04, atol=.06)
