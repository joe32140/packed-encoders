"""EmbeddingGemma 2 (bidirectional GQA, sliding and full layers, per-layer embeddings) — kernels, engine, HF entry.

    pytest tests/test_embedding_gemma2.py -q        # a tiny random EmbeddingGemma 2 (no download)

The engine tests build a random 6-layer text model with a 16-token window, so short rows cross
it, and compare the engine with HF's own forward per sequence, packed and graphed; the entry
tests pack the text backbone and the multimodal wrapper and keep their contract.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the EmbeddingGemma 2 engine needs an sm_80+ CUDA GPU",
)
eg2 = pytest.importorskip("transformers.models.embedding_gemma2.modeling_embedding_gemma2")

VOCAB, WINDOW = 1024, 16
PLACEHOLDERS = dict(image_token_id=1000, video_token_id=1001, audio_token_id=1002)


def _text_config(**overrides):
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2TextConfig,
    )

    cfg = dict(vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=6, num_attention_heads=4,
               num_key_value_heads=2, head_dim=32, global_head_dim=64, num_global_key_value_heads=1,
               sliding_window_pattern=3, sliding_window=WINDOW, hidden_size_per_layer_input=64, embedding_dim=96,
               max_position_embeddings=4096)
    cfg.update(overrides)
    return EmbeddingGemma2TextConfig(**cfg)


def _randomize(model):
    torch.manual_seed(0)
    for p in model.parameters():                     # random init is too small to exercise the kernels
        if p.dim() > 1:
            p.data.normal_(0, 0.05)
        else:
            p.data.uniform_(0.5, 1.5)                # norm weights away from 1: catches a missing scale
    for layer in model.layers:
        layer.layer_scalar.uniform_(0.3, 1.0)        # checkpoints scale every layer's output
        # Attention is scaled by 1, so unit-scale keys make it nearly one-hot and a random model
        # chaotic in bf16 (HF's own bf16 forward ends near 0.75 cosine of its fp32 one). The
        # checkpoint scales its keys down (k_norm ~0.12); so does this one.
        layer.self_attn.k_norm.weight.data.uniform_(0.1, 0.2)
    return model


@pytest.fixture(scope="module")
def tiny():
    return _randomize(eg2.EmbeddingGemma2TextModel(_text_config())).to("cuda", torch.bfloat16).eval()


def _hf_hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None]).last_hidden_state[0].float()


# ---------------------------------------------------------------------------- kernels


@torch.no_grad()
def test_post_norm_kernels_match_their_references():
    from packed_encoders.pieces.post_norm import post_norm_pieces

    pieces = post_norm_pieces()
    g = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)

    w = (torch.rand(128, device="cuda", generator=g) + 0.5).bfloat16().float()
    scalar = torch.tensor([0.37], device="cuda").bfloat16().float()
    proj = rand(37, 4 * 32 + 2 * 2 * 32)
    for args in [(rand(37, 128), w, 1e-6), (rand(37, 6 * 64).view(37, 6, 64), w[:64], 1e-6, 128 ** -0.5),
                 (proj[:, 192:].view(37, 2, 32), None, 1e-6)]:
        pieces["rms_norm"].validate(*args)
    assert pieces["rms_norm"].execute(rand(0, 128), w, 1e-6).shape == (0, 128)
    x = rand(37, 128)
    for args in [(rand(37, 128), x, w, 1e-6), (rand(37, 128), x, w, 1e-6, scalar, w * 0.5)]:
        assert pieces["post_norm_add"].validate(*args).max_abs_error <= 2 ** -6
    g_u = rand(37, 512)
    for args in [g_u.split(256, dim=-1), (rand(37, 64), rand(37, 6, 64)[:, 2])]:
        assert pieces["gelu_mul"].validate(*args).max_abs_error == 0.0


def test_post_norm_add_rejects_missing_primary_scale():
    from packed_encoders.errors import UnsupportedTargetError
    from packed_encoders.pieces.post_norm import post_norm_pieces

    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(UnsupportedTargetError, match="primary scale"):
        post_norm_pieces()["post_norm_add"].validate(x, x, None, 1e-6)


@pytest.mark.parametrize("order", [None, ("torch_varlen",), ("sdpa",)], ids=["default", "torch_varlen", "sdpa"])
def test_attention_probe_honours_window_and_scale(order):
    from packed_encoders.runtime import attention as att

    geometry = dict(n_heads=4, n_kv_heads=2, head_dim=64, device=torch.device("cuda"), dtype=torch.bfloat16)
    choice = att.select_attention(causal=False, order=order, window=(WINDOW, WINDOW), scale=1.0, **geometry)
    assert choice.max_abs_err < att.PROBE_TOLERANCE and choice.window == (WINDOW, WINDOW) and choice.scale == 1.0
    # The probe runs rows longer than the window: a kernel that ignores it is caught.
    unwindowed = att._sdpa((-1, -1), 1.0)
    err = att._probe(*unwindowed[:2], causal=False, window=(WINDOW, WINDOW), scale=1.0, **geometry)
    assert err > att.PROBE_TOLERANCE


# ---------------------------------------------------------------------------- engine


@torch.no_grad()
@pytest.mark.parametrize("lengths", [[9, 70, 1, 33], [200, 5, 64]])
def test_engine_matches_hf_per_sequence_packed_and_graphed(tiny, lengths):
    from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    eng = EmbeddingGemma2Engine(tiny)
    try:
        assert set(eng.attention) == {"sliding_attention", "full_attention"}
        assert eng.attention["sliding_attention"].window == (WINDOW, WINDOW)
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, VOCAB, (n,), generator=g).cuda() for n in lengths]
        ids = torch.cat(seqs)
        ref = torch.cat([_hf_hidden(tiny, s) for s in seqs])
        packed = eng.forward_packed(ids, lengths).float()
        assert packed.shape == (sum(lengths), 96)
        cos = F.cosine_similarity(packed, ref, dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (cos.mean().item(), cos.min().item())
        via_embeds = eng.forward_packed(None, lengths, embeds=tiny.embed_tokens(ids)).float()
        assert torch.equal(via_embeds, packed)

        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(4, 8), max_seq=256, max_tokens=2048))
        graphed = runner(ids, lengths).float()
        again = runner(ids, lengths).float()          # replay, not capture
        cos = F.cosine_similarity(graphed, packed, dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99
        assert torch.equal(graphed, again)

        # Both must be at least as close to the fp32 model as HF's own bf16 forward is.
        fp32 = copy.deepcopy(tiny).float()
        truth = torch.cat([_hf_hidden(fp32, s) for s in seqs])
        hf_bf16 = F.cosine_similarity(ref, truth, dim=-1).mean().item()
        for x in (packed, graphed):
            assert F.cosine_similarity(x, truth, dim=-1).mean().item() > hf_bf16 - 1e-4
    finally:
        eng.release()


def test_release_restores_independent_weights(tiny):
    from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine

    model = copy.deepcopy(tiny)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    eng = EmbeddingGemma2Engine(model)
    sa = model.layers[2].self_attn
    assert sa.q_proj.weight.untyped_storage().data_ptr() == sa.k_proj.weight.untyped_storage().data_ptr()
    eng.release()
    assert sa.q_proj.weight.untyped_storage().data_ptr() != sa.k_proj.weight.untyped_storage().data_ptr()
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


@pytest.mark.parametrize("where", ["bias", "lora", "activation"])
def test_unsupported_models_are_refused_before_mutation(tiny, where):
    from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine
    from packed_encoders.errors import UnsupportedTargetError

    model = copy.deepcopy(tiny)
    if where == "bias":
        o = model.layers[3].self_attn.o_proj
        o.bias = nn.Parameter(torch.zeros(o.out_features, device="cuda", dtype=torch.bfloat16))
    elif where == "lora":
        model.layers[4].ple_block.per_layer_projection.lora_A = nn.Identity()
    else:
        model.config.hidden_activation = "gelu"
    ptr = model.layers[0].mlp.gate_proj.weight.untyped_storage().data_ptr()
    with pytest.raises(UnsupportedTargetError):
        EmbeddingGemma2Engine(model)
    assert model.layers[0].mlp.gate_proj.weight.untyped_storage().data_ptr() == ptr


@torch.no_grad()
def test_norms_and_layer_scalars_changed_after_packing_reach_eager_and_captured_graphs(tiny):
    from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine
    from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner

    model = copy.deepcopy(tiny)
    eng = EmbeddingGemma2Engine(model)
    try:
        lengths = [12, 30]
        ids = torch.randint(0, VOCAB, (sum(lengths),), generator=torch.Generator().manual_seed(2)).cuda()
        runner = PaddedGraphRunner(eng, PaddedGraphConfig(row_buckets=(2,), max_seq=64, max_tokens=256))
        runner(ids, lengths)                                     # captured with the old values
        for layer in model.layers:
            layer.post_feedforward_layernorm.weight.mul_(1.7)
            layer.self_attn.k_norm.weight.mul_(0.5)
            layer.layer_scalar.mul_(0.5)
        model.norm.weight.add_(0.3)
        model.ple.per_layer_projection_norm.weight.mul_(1.3)
        ref = torch.cat([_hf_hidden(model, s) for s in ids.split(lengths)])
        eager = eng.forward_packed(ids, lengths).float()
        eng.sync_norms()                                         # the HF entry syncs before each replay
        graphed = runner(ids, lengths).float()
        for x in (eager, graphed):
            assert F.cosine_similarity(x, ref, dim=-1).min().item() > 0.99
    finally:
        eng.release()


# ---------------------------------------------------------------------------- HF entry


def _stock(tiny, kind):
    if kind == "text_model":
        return copy.deepcopy(tiny)
    cfg = eg2.EmbeddingGemma2Config(text_config=tiny.config.to_dict(), vision_config=None, audio_config=None,
                                    **PLACEHOLDERS)
    model = eg2.EmbeddingGemma2Model(cfg).to("cuda", torch.bfloat16).eval()
    model.language_model.load_state_dict(tiny.state_dict())
    for mine, theirs in zip(model.language_model.layers, tiny.layers):
        mine.layer_scalar.copy_(theirs.layer_scalar)
    return model


@pytest.mark.parametrize("kind", ["text_model", "wrapper"])
def test_hf_entry_packs_stock_models_and_keeps_their_contract(tiny, kind):
    import packed_encoders as pe
    from packed_encoders.state import ATTR

    model = _stock(tiny, kind)
    g = torch.Generator().manual_seed(5)
    lengths = [9, 70, 1, 33]
    S = max(lengths)
    ids = torch.randint(0, VOCAB - 50, (len(lengths), S), generator=g).cuda()
    if kind == "wrapper":
        ids[0, 2], ids[1, 40], ids[3, 0] = PLACEHOLDERS.values()   # served as padding, as the wrapper does
    right = (torch.arange(S)[None] < torch.tensor(lengths)[:, None]).long().cuda()
    left = right.flip(1)

    def run(m=right):
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=m).last_hidden_state

    stock = {"right": run(right), "left": run(left)}
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    pe.pack(model)
    state = getattr(model, ATTR)
    assert state.report.eager_cos_mean > 0.999 and state.report.embeds_cos_mean > 0.999

    for side, m in (("right", right), ("left", left)):
        out = run(m)
        real = m.bool()
        assert out.shape == stock[side].shape and out.dtype == stock[side].dtype
        cos = F.cosine_similarity(out[real].float(), stock[side][real].float(), dim=-1)
        assert cos.mean().item() > 0.999 and cos.min().item() > 0.99, (side, cos.mean().item(), cos.min().item())
        assert not out[~real].any()                            # pads come back as zeros
    assert state.runner.num_graphs > 0

    # Calls the engine can't serve run the model's own forward, unchanged.
    calls, original = [], state.original_forward
    state.original_forward = lambda *a, **k: calls.append(1) or original(*a, **k)
    holes = right.clone()
    holes[1, 3] = 0
    with pytest.warns(UserWarning, match="original forward"):
        run(holes)
    with torch.enable_grad():
        model(input_ids=ids, attention_mask=right)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=right, position_ids=torch.arange(S, device="cuda")[None])
        if kind == "wrapper":                                  # embeddings: the wrapper reads placeholders off them
            model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=right)
    assert len(calls) == (4 if kind == "wrapper" else 3)
    state.original_forward = original

    pe.unpack(model)
    assert getattr(model, ATTR, None) is None
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n]), n
    assert torch.allclose(run(right), stock["right"], atol=1e-3)


def test_backbone_inputs_embeds_run_packed(tiny):
    import packed_encoders as pe
    from packed_encoders.state import ATTR

    model = copy.deepcopy(tiny)
    ids = torch.randint(0, VOCAB, (3, 40), generator=torch.Generator().manual_seed(7)).cuda()
    mask = torch.ones_like(ids)
    mask[1, 25:] = 0
    with torch.no_grad():
        embeds = model.embed_tokens(ids)
        stock = model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        pe.pack(model)
        state = getattr(model, ATTR)
        calls, original = [], state.original_forward
        state.original_forward = lambda *a, **k: calls.append(1) or original(*a, **k)
        out = model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    assert not calls
    real = mask.bool()
    cos = F.cosine_similarity(out[real].float(), stock[real].float(), dim=-1)
    assert cos.mean().item() > 0.999 and cos.min().item() > 0.99
    pe.unpack(model)


@torch.no_grad()
@pytest.mark.parametrize("path", ["eager", "embeds", "graph"])
def test_validation_rejects_zeroed_short_sequences(tiny, monkeypatch, path):
    import packed_encoders as pe
    from packed_encoders.arch.embedding_gemma2.engine import EmbeddingGemma2Engine
    from packed_encoders.errors import ValidationError
    from packed_encoders.state import ATTR

    model = copy.deepcopy(tiny)
    if path == "graph":
        original = EmbeddingGemma2Engine.padded_core

        def broken(self, static):
            out = original(self, static)
            # Use the live device lengths: the faulty operation must also run during replay.
            short = (static.lens <= 7).repeat_interleave(static.seq)[:, None]
            return out.masked_fill(short, 0)

        monkeypatch.setattr(EmbeddingGemma2Engine, "padded_core", broken)
    else:
        original = EmbeddingGemma2Engine.forward_packed

        def broken(self, ids, lengths, *, embeds=None):
            out = original(self, ids, lengths, embeds=embeds)
            if (path == "embeds") == (embeds is not None):
                out = out.clone()
                offset = 0
                for length in lengths:
                    if length <= 7:
                        out[offset:offset + length] = 0
                    offset += length
            return out

        monkeypatch.setattr(EmbeddingGemma2Engine, "forward_packed", broken)

    # These rows are <1% of either default batch, so a batch-mean-only gate accepts the fault.
    with pytest.raises(ValidationError, match="worst sequence mean 0.000000"):
        pe.pack(model)
    assert getattr(model, ATTR, None) is None


@torch.no_grad()
@pytest.mark.parametrize("kind", ["text_model", "wrapper"])
def test_prepared_mask_dictionary_falls_back_to_hf(tiny, kind):
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

    import packed_encoders as pe
    from packed_encoders.state import ATTR

    model = _stock(tiny, kind)
    text = model if kind == "text_model" else model.language_model
    ids = torch.randint(5, VOCAB - 50, (2, 40), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, 20:] = 0
    kwargs = dict(config=text.config, inputs_embeds=text.embed_tokens(ids), attention_mask=mask)
    prepared = dict(full_attention=create_bidirectional_mask(**kwargs),
                    sliding_attention=create_bidirectional_sliding_window_mask(**kwargs))
    stock = model(input_ids=ids, attention_mask=prepared).last_hidden_state
    pe.pack(model)
    try:
        state = getattr(model, ATTR)
        original, calls = state.original_forward, []
        state.original_forward = lambda *a, **k: calls.append(1) or original(*a, **k)
        with pytest.warns(UserWarning, match="original forward"):
            out = model(input_ids=ids, attention_mask=prepared).last_hidden_state
        assert calls == [1]
        assert torch.equal(out, stock)
    finally:
        pe.unpack(model)


@torch.no_grad()
@pytest.mark.parametrize("kind", ["text_model", "wrapper"])
@pytest.mark.parametrize("graphs", [False, True])
def test_training_dropout_falls_back_after_packing_in_eval(tiny, kind, graphs):
    import packed_encoders as pe
    from packed_encoders.state import ATTR

    model = _stock(tiny, kind)
    text = model if kind == "text_model" else model.language_model
    text.config.attention_dropout = 0.5
    for layer in text.layers:
        layer.self_attn.attention_dropout = 0.5
    ids = torch.randint(5, VOCAB - 50, (2, 40), device="cuda")
    pe.pack(model, cuda_graph=graphs)
    try:
        state = getattr(model, ATTR)
        before = model(input_ids=ids).last_hidden_state
        if graphs:
            assert state.runner.num_graphs > 0
        original, calls = state.original_forward, []
        state.original_forward = lambda *a, **k: calls.append(1) or original(*a, **k)
        model.train()
        torch.manual_seed(17)
        stock = original(input_ids=ids).last_hidden_state
        torch.manual_seed(17)
        with pytest.warns(UserWarning, match="training dropout"):
            out = model(input_ids=ids).last_hidden_state
        assert calls == [1]
        assert torch.equal(out, stock)  # identical stochastic computation with the same RNG state

        model.eval()
        assert torch.equal(model(input_ids=ids).last_hidden_state, before)
        assert calls == [1]            # eval returns to the packed path
    finally:
        pe.unpack(model)
