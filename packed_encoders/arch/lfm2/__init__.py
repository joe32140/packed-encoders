"""LFM2 hybrid backbones (gated short convolutions + GQA softmax attention).

The engine (`engine.Lfm2Engine`) is per backbone; the adapter patches the text backbone itself
(`Lfm2Model`), found directly, under a task head (`.model`), or as the `.language_model` of a
multimodal wrapper. Its patched forward keeps HF's contract: `input_ids` or `inputs_embeds`
plus `attention_mask` (right or left padded) in, padded `last_hidden_state` out. A multimodal
wrapper merges its image features into `inputs_embeds` before calling the backbone, so image
batches run packed too.

What `pack()` installs, per batch:
  - no grad, rows sharing at least `min_shared_prefix` leading tokens or embedding rows (opt
    in; a decision model asking several questions about one context): each shared prefix runs
    once, token ids as one CUDA graph per plan (runtime.shared_graphs), embeddings eagerly
    (`engine.forward_shared`).
  - other token-id batches, no grad: CUDA graph replay when the batch fits a `(rows, S)`
    bucket, else the eager packed engine.
  - other input-embedding batches, no grad: the eager packed engine.
  - grad-enabled calls, a KV cache (pass `use_cache=False`, or set it in the config), extra
    outputs, explicit `position_ids`, a mask with holes or any mask but a 2D padding mask (a
    prepared per-layer-type mapping, say): the original forward, untouched.

Rows laid out `[context][group 1]..[group N]` (a decision model's state, then several questions
read in one pass) run with grouped attention through `PackedLfm2.forward_grouped`; see there.
The LFM3 entry (packed_encoders.arch.lfm3) shares this module and the engine.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders.arch.weights import validation_ids_below
from packed_encoders.runtime.cuda_epilogue import CUDAEpilogueSupport
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError, ValidationError
from packed_encoders.runtime.graphs import PaddedGraphConfig, PaddedGraphRunner, graphs_globally_disabled
from packed_encoders.runtime.shared_graphs import SharedGraphRunner
from packed_encoders.runtime.staging import PinnedStager
from packed_encoders.state import ATTR

MIN_CAPABILITY = (8, 0)   # bf16 tensor cores; the Triton kernels target sm_80+
# Validation measures the engine against the model's own forward on this device: per-token
# cosine over real tokens, bf16 vs bf16, for eager and graphed alike. Graphs are not held to
# eager: a bucket's padded GEMMs round differently, and over a deep stack two bf16 paths drift
# apart more than either drifts from fp32 (measurements: docs/architectures.md, LFM2).
# The gate is therefore loose, and set from what a broken engine scores rather than from how
# close a correct one happens to land on one device. On d1-3B the engine is closer to an fp32
# forward than the model's own bf16 forward is, yet the two bf16 paths sit 0.9989 apart on the
# short validation batch (0.9970 replayed from a graph, which pads every row to its bucket and
# so changes the GEMM shapes), and a 0.999 mean refused that checkpoint on an RTX 4090 and a
# 5090 while passing on an L40S. Breaking one piece at a time on the same batches scores 0.0184
# to 0.9218 by mean and 0.064 to 0.643 by minimum, so these thresholds keep a margin of about
# 0.07 and 0.21 to a real error (measurements: docs/architectures.md, LFM2).
MEAN_COS, MIN_COS = 0.99, 0.85
_BACKENDS = {None: None, "auto": None, "flash": ("flash4", "flash2", "torch_varlen"), "sdpa": ("sdpa",)}


def _lfm2_text_model(module: object) -> nn.Module | None:
    """`Lfm2Model`, or the one a wrapper holds as `.language_model`. By class name along the MRO,
    so matching never imports transformers' LFM2 code."""
    if not isinstance(module, nn.Module):
        return None
    if any(c.__name__ == "Lfm2Model" and c.__module__.startswith("transformers.models.lfm2.")
           for c in type(module).__mro__):
        return module
    lm = getattr(module, "language_model", None)
    return lm if lm is not module and _lfm2_text_model(lm) is lm else None


def grouped_attention_masks(cfg, lengths, groups, seq: int, dtype: torch.dtype, device) -> dict[str, Tensor | None]:
    """What `forward_grouped` computes, as masks the model's own forward takes (a mapping keyed by
    layer type, additive, `(rows, 1, seq, seq)`) for its rows right padded to `seq`: grouped on
    full-attention layers, causal within the window on sliding ones, and the convolutions
    unmasked (a right pad never reaches a real token). A pad query reads itself, so no row of
    the mask is empty. For checking the engine against the model; it costs `rows * seq**2`."""
    B = len(lengths)
    q, k = torch.arange(seq)[:, None], torch.arange(seq)[None, :]
    full = torch.zeros(B, seq, seq, dtype=torch.bool)
    for i, (n, own) in enumerate(zip(lengths, groups)):
        context = n - sum(own)
        full[i, :n, :context] = True              # every token reads the context
        full[i, :context, :n] = True              # the context reads the whole row
        at = context
        for size in own:
            full[i, at:at + size, at:at + size] = True
            at += size
    real = torch.arange(seq)[None, :] < torch.tensor(lengths)[:, None]
    pad_self = (~real)[:, :, None] & (q == k)[None]
    allowed = {"full_attention": full}
    window = getattr(cfg, "sliding_window", None)
    if window:
        allowed["sliding_attention"] = ((k <= q) & (q - k < window))[None] & real[:, None, :]
    low = torch.finfo(dtype).min
    masks: dict[str, Tensor | None] = {}
    for kind in set(cfg.layer_types):
        if kind in allowed:
            keep = allowed[kind] | pad_self
            masks[kind] = torch.where(keep, 0.0, low).to(device=device, dtype=dtype)[:, None]
        else:
            masks[kind] = None
    return masks


@dataclass
class Lfm2Report:
    capability: tuple[int, int]
    torch_version: str
    attention_backend: str
    attention_max_abs_err: float
    attention_rejected: dict[str, str]
    kernel_errors: dict[str, float]
    batches: tuple[tuple[int, ...], ...]
    eager_cos_mean: float | None = None       # the worst batch's per-token cosine vs the model's forward
    eager_cos_min: float | None = None
    embeds_cos_mean: float | None = None      # the inputs_embeds entry, eager
    embeds_cos_min: float | None = None
    graph_cos_mean: float | None = None       # graph replay, also against the model's forward
    graph_cos_min: float | None = None
    shared_prefix_rejected: str | None = None  # why rows can't share prefixes here (None: they can, opt in)
    shared_cos_mean: float | None = None       # rows continuing shared prefixes (ids and embeddings) vs the model
    shared_cos_min: float | None = None
    grouped_rejected: str | None = None        # why forward_grouped is unavailable here (None: it runs)
    grouped_cos_mean: float | None = None      # grouped rows vs the model's forward under the same masks
    grouped_cos_min: float | None = None
    pieces: dict = field(default_factory=dict)

    def record(self, name: str, got: Tensor, ref: Tensor) -> tuple[float, float]:
        """Per-token cosine of `got` vs `ref` into `{name}_cos_mean/min`, keeping the worst batch."""
        cos = F.cosine_similarity(got, ref, dim=-1)
        mean, low = cos.mean().item(), cos.min().item()
        for stat, value in (("mean", mean), ("min", low)):
            prev = getattr(self, f"{name}_cos_{stat}")
            setattr(self, f"{name}_cos_{stat}", value if prev is None else min(prev, value))
        return mean, low


@dataclass
class Lfm2State:
    arch: str
    engine: Any
    original_forward: Any
    runner: PaddedGraphRunner | None
    graph_enabled: bool
    stager: PinnedStager
    report: Lfm2Report | None = None
    fallback_warned: bool = False
    shared_runner: SharedGraphRunner | None = None
    single_row_conv_masked: bool = False      # the model masks a lone left-padded row's conv inputs (LFM3)

    def set_cuda_graph(self, enabled: bool, config: PaddedGraphConfig | None = None) -> None:
        if config is not None and not isinstance(config, PaddedGraphConfig):
            raise PackedEncodersError(f"LFM2 graphs take a runtime.PaddedGraphConfig, got {type(config).__name__}")
        if enabled and (self.runner is None or config is not None):
            self.runner = PaddedGraphRunner(self.engine, config)
            self.shared_runner = None
        self.graph_enabled = enabled


def _hidden_once(state: Lfm2State, ids: Tensor | None, lengths, embeds: Tensor | None, graphs: bool) -> Tensor:
    use_graph = (embeds is None and graphs and state.runner is not None and state.graph_enabled
                 and not graphs_globally_disabled() and not torch.is_autocast_enabled("cuda"))
    if use_graph:
        state.engine.sync_norms()             # norms trained since packing reach the graphs too
    plan = state.engine.plan_sharing(ids, lengths, embeds=embeds)
    if plan is not None:
        if use_graph:
            if state.shared_runner is None:
                state.shared_runner = SharedGraphRunner(state.engine, state.runner.config)
            hidden = state.shared_runner(ids, plan)
            if hidden is not None:
                return hidden
        return state.engine.forward_shared(ids, plan, embeds=embeds)
    hidden = None
    if use_graph:
        hidden = state.runner(ids, lengths)
    if hidden is None:
        hidden = state.engine.forward_packed(ids, lengths, embeds=embeds)
    return hidden


def _hidden(state: Lfm2State, ids: Tensor | None, lengths, *, embeds: Tensor | None = None,
            graphs: bool = True) -> Tensor:
    # Each attempt in its own frame, so a failed attempt's allocations are released before the retry.
    for attempt in (0, 1):
        try:
            return _hidden_once(state, ids, lengths, embeds, graphs)
        except torch.OutOfMemoryError as exc:
            if attempt or state.runner is None:
                raise
            reason = (str(exc).splitlines() or ["CUDA out of memory"])[0][:160]
        _drop_graphs(state, reason)          # outside the handler: its traceback would keep the graphs alive


def _drop_graphs(state: Lfm2State, reason: str) -> None:
    """Out of GPU memory with graphs held: free them all (their pool can't lend to eager) and go on."""
    with torch.cuda.device(state.engine.device):
        state.runner, state.graph_enabled = None, False
        state.shared_runner = None
        gc.collect()
        torch.cuda.empty_cache()
    warnings.warn(f"packed-encoders: out of GPU memory with CUDA graphs held ({reason}); dropped them and "
                  "continuing without. Disable graphs with cuda_graph=False or reduce the buckets with "
                  "PaddedGraphConfig(max_tokens=...).", stacklevel=3)


def _hf_rows(state: Lfm2State, input_ids, embeds, attention_mask, args, kwargs) -> tuple[list[int], list[int]] | None:
    """Per-row token counts and first real column, or None when the engine can't serve the call."""
    if args or torch.is_grad_enabled() or (input_ids is None) == (embeds is None):
        return None
    if input_ids is not None and input_ids.dim() != 2:
        return None
    if embeds is not None and (embeds.dim() != 3 or embeds.shape[-1] != state.engine.hidden_size):
        return None
    cfg = state.engine.cfg
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
        use_cache = getattr(cfg, "use_cache", False)
    if use_cache or any(kwargs.get(k, getattr(cfg, k, False)) for k in ("output_hidden_states", "output_attentions")):
        return None
    for k, v in kwargs.items():
        if k == "return_dict" or v is None or v is False:
            continue
        return None          # position_ids, a cache, extra outputs
    B, S = (input_ids if embeds is None else embeds).shape[:2]
    if attention_mask is None:
        return [S] * B, [0] * B
    if not isinstance(attention_mask, Tensor) or attention_mask.shape != (B, S):
        return None          # a prepared mask (a per-layer-type mapping, a 4D mask): only the model reads it
    real = attention_mask != 0
    lens = real.sum(1)
    col = torch.arange(S, device=real.device)
    right = (real == (col < lens[:, None])).all(1)
    left = (real == (col >= S - lens[:, None])).all(1)
    flat = torch.cat([lens, right.long(), left.long()]).tolist()      # the only device->host read
    lengths, right, left = flat[:B], flat[B:2 * B], flat[2 * B:]
    if min(lengths) == 0 or not all(r or l for r, l in zip(right, left)):
        return None          # an empty row, or a mask with holes
    if B == 1 and not right[0] and not state.single_row_conv_masked:
        # LFM2 skips masking short-conv inputs for a single row. Left pads can therefore
        # contribute to its first real tokens; removing them would change the result.
        return None
    return lengths, [0 if r else S - n for n, r in zip(lengths, right)]


def _make_hf_forward(module: nn.Module, state: Lfm2State):
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def forward(input_ids=None, attention_mask=None, *args, **kwargs):
        embeds = kwargs.pop("inputs_embeds", None)
        rows = _hf_rows(state, input_ids, embeds, attention_mask, args, kwargs)
        if rows is None:
            if not state.fallback_warned:
                state.fallback_warned = True
                warnings.warn("packed-encoders: this call runs the model's original forward (gradients, a KV cache "
                              "or use_cache, position_ids, extra outputs, single-row left padding, a mask with holes "
                              "or a prepared mask)", stacklevel=2)
            if embeds is not None:
                kwargs["inputs_embeds"] = embeds
            return state.original_forward(input_ids, attention_mask, *args, **kwargs)
        lengths, starts = rows
        B, S = (input_ids if embeds is None else embeds).shape[:2]
        ids = None if input_ids is None else input_ids.reshape(-1)
        flat = None if embeds is None else embeds.reshape(B * S, -1)
        if sum(lengths) == B * S:
            out = _hidden(state, ids, lengths, embeds=flat).view(B, S, -1)
        else:
            # Row i's tokens are the flat slots [i*S + start_i, i*S + start_i + len_i); built on host.
            lens = torch.as_tensor(lengths, dtype=torch.long)
            first = torch.arange(B) * S + torch.as_tensor(starts, dtype=torch.long)
            idx = torch.arange(int(lens.sum())) + torch.repeat_interleave(first - (torch.cumsum(lens, 0) - lens), lens)
            (d_idx,) = state.stager.put([idx])
            hidden = _hidden(state, None if ids is None else ids.index_select(0, d_idx), lengths,
                             embeds=None if flat is None else flat.index_select(0, d_idx))
            out = hidden.new_zeros((B * S, hidden.shape[-1])).index_copy_(0, d_idx, hidden).view(B, S, -1)
        return_dict = kwargs.get("return_dict")
        if return_dict is None:
            return_dict = getattr(module.config, "return_dict", True)
        return BaseModelOutputWithPast(last_hidden_state=out) if return_dict else (out,)

    return forward


class HFLfm2Adapter:
    name = "hf-lfm2"
    find = staticmethod(_lfm2_text_model)

    def bind(self, module: object) -> ModelBinding | None:
        lm = self.find(module)
        return ModelBinding(self, lm, lm) if lm is not None else None

    def install(self, binding: ModelBinding, packed) -> None:
        setattr(binding.patch_target, ATTR, packed.state)
        binding.patch_target.forward = _make_hf_forward(binding.patch_target, packed.state)


def _check_options(options, family="LFM2"):
    unknown = options.keys() - {"cuda_graph", "train_cuda_graph", "attention_backend", "validate"}
    if unknown:
        raise PackedEncodersError(f"unsupported {family} options: {sorted(unknown)}")
    graph = options.get("cuda_graph")
    if graph is not None and not isinstance(graph, (bool, PaddedGraphConfig)):
        raise PackedEncodersError(f"{family} graphs take a bool or PaddedGraphConfig")
    if options.get("train_cuda_graph") not in (None, False):
        raise PackedEncodersError(f"training graphs are not implemented for {family}")
    if options.get("attention_backend") not in _BACKENDS:
        raise PackedEncodersError(f"unsupported {family} attention_backend: {options['attention_backend']!r}")


class Lfm2Hybrid:
    name = "lfm2"
    family = "LFM2"
    adapters = (HFLfm2Adapter(),)
    single_row_conv_masked = False      # LFM2's conv skips the padding mask for one row (see _hf_rows)

    def __init__(self, *, pieces=None):
        self.pieces = pieces

    def _require_env(self, module: nn.Module) -> torch.device:
        try:
            param = next(module.parameters())
        except StopIteration as exc:
            raise ValidationError("the model has no parameters") from exc
        if param.device.type != "cuda":
            raise ValidationError(f"the {self.family} engine needs CUDA weights; the model is on {param.device}")
        cap = torch.cuda.get_device_capability(param.device)
        if cap < MIN_CAPABILITY:
            raise ValidationError(f"sm_{cap[0]}{cap[1]} is below sm_80 (bf16 + Triton kernels)")
        if param.dtype != torch.bfloat16:
            raise ValidationError(f"the {self.family} engine runs bf16 weights; load with torch_dtype=torch.bfloat16")
        try:
            import fla.modules.layernorm  # noqa: F401  (the fused residual + RMSNorm)
        except ImportError as exc:
            raise ValidationError(f"{self.family} packing needs flash-linear-attention for its fused norms") from exc
        return param.device

    def _validate(self, module: nn.Module, *, batches: tuple[tuple[int, ...], ...] = ((7, 129, 300, 64), (3, 21, 30)),
                  graphs: bool = True, _engine=None, _state: Lfm2State | None = None) -> Lfm2Report:
        """Build (or reuse) the engine and compare it with the model's own forward on random text
        rows, on this device: token ids eager and graphed, and input embeddings eager. The report
        keeps the worst batch. Raises ValidationError on any miss."""
        device = self._require_env(module)
        state = _state if _state is not None else getattr(module, ATTR, None)
        engine = _engine or (state.engine if state is not None else None)
        built_here = engine is None
        if built_here:
            engine = self._engine(module, None)
        try:
            report = Lfm2Report(
                capability=torch.cuda.get_device_capability(device), torch_version=torch.__version__,
                attention_backend=engine.attention.name if engine.attention else "none",
                attention_max_abs_err=engine.attention.max_abs_err if engine.attention else 0.0,
                attention_rejected=dict(engine.attention.rejected) if engine.attention else {},
                kernel_errors=dict(engine.kernel_errors), batches=tuple(tuple(b) for b in batches))
            with torch.cuda.device(device):
                report.pieces = engine.composition.validate(engine)
            oracle = state.original_forward if state is not None else module.forward
            probe_state = state or Lfm2State(arch=self.name, engine=engine, original_forward=oracle, runner=None,
                                             graph_enabled=False, stager=PinnedStager(engine.device),
                                             single_row_conv_masked=self.single_row_conv_masked)
            for lengths in batches:
                self._check_numerics(oracle, probe_state, lengths, graphs, report)
            if engine.share_rejected is None:
                self._check_sharing(oracle, probe_state, report)
            report.shared_prefix_rejected = engine.share_rejected
            if engine.grouped_rejected is None:
                self._check_grouped(oracle, engine, report)
            report.grouped_rejected = engine.grouped_rejected
        finally:
            if built_here:
                engine.release(rollback=True)
        return report

    @staticmethod
    def _reference(oracle, engine, seqs: list[Tensor]) -> Tensor:
        """The model's own forward on right-padded `seqs`: real tokens in packed order (fp32)."""
        lengths = [len(x) for x in seqs]
        ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(engine.device)
        mask = (torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]).to(engine.device)
        try:
            return oracle(input_ids=ids, attention_mask=mask.long(), use_cache=False)[0][mask].float()
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"the model's own forward failed on the validation batch ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}); cannot certify the engine here. Pass validate=False to "
                "skip at your own risk.") from exc

    def _check_sharing(self, oracle, state: Lfm2State, report: Lfm2Report) -> None:
        """Rows continuing two shared prefixes (children of 1 to 20 tokens) beside rows sharing nothing,
        as token ids and as input embeddings, against the model's own forward, whether or not sharing
        is on. A miss makes sharing unavailable (recorded) rather than failing the pack."""
        from packed_encoders.arch.lfm2.engine import SHARED_PREFIX_MIN

        engine, threshold = state.engine, state.engine.min_shared_prefix
        m = threshold or SHARED_PREFIX_MIN
        g = torch.Generator().manual_seed(1)

        def rand(n):
            return torch.randint(0, validation_ids_below(engine.cfg), (n,), generator=g)

        a, b = rand(m + 30), rand(m + 6)
        seqs = [torch.cat([a, rand(7)]), rand(40), torch.cat([b, rand(12)]), torch.cat([a, rand(1)]),
                rand(m + 50), torch.cat([b, rand(3)]), torch.cat([a, rand(20)])]
        lengths = [len(x) for x in seqs]
        reason = None
        try:
            engine.min_shared_prefix = m
            ids = torch.cat(seqs).to(engine.device)
            embeds = F.embedding(ids, engine.embed)
            with torch.no_grad():
                ref = self._reference(oracle, engine, seqs)
                for name, plan_ids, emb in (("ids", ids, None), ("embeds", None, embeds)):
                    plan = engine.plan_sharing(plan_ids, lengths, embeds=emb)
                    if plan is None or plan.saved_tokens == 0:
                        raise ValidationError(f"the shared-prefix validation batch formed no group ({name})")
                    report.record("shared", engine.forward_shared(plan_ids, plan, embeds=emb).float(), ref)
            if not (report.shared_cos_mean >= MEAN_COS and report.shared_cos_min >= MIN_COS):
                reason = (f"rows continuing shared prefixes diverged from the model's forward: per-token cosine "
                          f"mean {report.shared_cos_mean:.6f} min {report.shared_cos_min:.5f}")
        except Exception as exc:  # optional optimization; ordinary numerics have already passed
            reason = f"shared-prefix validation failed: {type(exc).__name__}: {str(exc)[:200]}"
        finally:
            engine.min_shared_prefix = threshold if reason is None else 0
        if reason is not None:
            engine.share_rejected = reason
            warnings.warn(f"packed-encoders: {reason}; every row runs in full", stacklevel=4)

    def _check_grouped(self, oracle, engine, report: Lfm2Report) -> None:
        """Grouped rows (contexts of 1 token to 44 past the sliding window, 256 without one, with none
        to four groups of 1 to 30) as token ids and as input embeddings, against the model's own
        forward under the same masks given per layer type (`grouped_attention_masks`). A miss, or a
        forward that takes no such masks, makes forward_grouped unavailable (recorded) rather than
        failing the pack. The gate catches what varies with the machine and the model's code
        (kernels, gathers, mask handling); the pattern itself is fixed code, pinned exactly by
        tests/test_grouped_layout.py, since some wrong patterns score inside bf16 noise here."""
        g = torch.Generator().manual_seed(2)
        long = (getattr(engine.cfg, "sliding_window", None) or 256) + 44
        contexts, groups = [40, 25, 9, long, 1], [[7, 1, 12], [], [20], [16, 30, 5, 22], [3, 3]]
        lengths = [c + sum(own) for c, own in zip(contexts, groups)]
        seqs = [torch.randint(0, validation_ids_below(engine.cfg), (n,), generator=g) for n in lengths]
        reason = None
        try:
            ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(engine.device)
            real = (torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]).to(engine.device)
            masks = grouped_attention_masks(engine.cfg, lengths, groups, ids.shape[1], engine.dtype, engine.device)
            packed = torch.cat(seqs).to(engine.device)
            with torch.no_grad():
                ref = oracle(input_ids=ids, attention_mask=masks, use_cache=False)[0][real].float()
                for got in (engine.forward_grouped(packed, lengths, groups),
                            engine.forward_grouped(None, lengths, groups, embeds=F.embedding(packed, engine.embed))):
                    report.record("grouped", got.float(), ref)
            if not (report.grouped_cos_mean >= MEAN_COS and report.grouped_cos_min >= MIN_COS):
                reason = (f"grouped rows diverged from the model's forward under the same masks: per-token cosine "
                          f"mean {report.grouped_cos_mean:.6f} min {report.grouped_cos_min:.5f}")
        except Exception as exc:  # optional; ordinary numerics have already passed
            reason = f"grouped-row validation failed: {type(exc).__name__}: {str(exc)[:200]}"
        if reason is not None:
            engine.grouped_rejected = reason
            warnings.warn(f"packed-encoders: {reason}; forward_grouped is unavailable", stacklevel=4)

    def _check_numerics(self, oracle, state: Lfm2State, lengths, graphs, report: Lfm2Report) -> None:
        engine = state.engine
        g = torch.Generator().manual_seed(0)
        seqs = [torch.randint(0, validation_ids_below(engine.cfg), (n,), generator=g) for n in lengths]
        B = len(seqs)
        packed = torch.cat(seqs).to(engine.device)
        with torch.no_grad():
            ref = self._reference(oracle, engine, seqs)
            eager = _hidden(state, packed, list(lengths), graphs=False).float()
            embeds = F.embedding(packed, engine.embed)
            via_embeds = _hidden(state, None, list(lengths), embeds=embeds, graphs=False).float()
            for name, got in (("eager", eager), ("embeds", via_embeds)):
                mean, low = report.record(name, got, ref)
                if not (mean >= MEAN_COS and low >= MIN_COS):
                    raise ValidationError(f"engine ({name}) diverged from the model's forward on rows {tuple(lengths)}: "
                                          f"per-token cosine mean {mean:.6f} min {low:.5f}")
            if graphs:
                runner = PaddedGraphRunner(engine, PaddedGraphConfig(row_buckets=(B,), max_graphs=1))
                graphed = runner(packed, list(lengths)).float()
                del runner
                mean, low = report.record("graph", graphed, ref)
                if not (mean >= MEAN_COS and low >= MIN_COS):
                    raise ValidationError(f"CUDA-graph replay diverged from the model's forward on rows {tuple(lengths)}: "
                                          f"per-token cosine mean {mean:.6f} min {low:.5f}")

    def _engine(self, module: nn.Module, attention_backend: str | None):
        from packed_encoders.arch.lfm2.engine import Lfm2Engine

        if attention_backend not in _BACKENDS:
            raise PackedEncodersError(f"attention_backend for {self.family} must be one of "
                                      f"{sorted(map(str, _BACKENDS))}")
        return Lfm2Engine(module, causal=True, attention_order=_BACKENDS[attention_backend], pieces=self.pieces,
                          family=self.family)

    def validate(self, binding: ModelBinding, **kwargs) -> ValidationResult:
        return ValidationResult(self.name, self._validate(binding.patch_target, **kwargs))

    def prepare(self, binding: ModelBinding, options):
        _check_options(options, self.family)
        module = binding.patch_target
        self._require_env(module)
        engine = self._engine(module, options.get("attention_backend"))
        try:
            graph = options.get("cuda_graph")
            use_graphs = graph is None or bool(graph)
            runner = PaddedGraphRunner(engine, graph if isinstance(graph, PaddedGraphConfig) else None) if use_graphs else None
            state = Lfm2State(arch=self.name, engine=engine, original_forward=module.forward, runner=runner,
                              graph_enabled=use_graphs, stager=PinnedStager(engine.device),
                              single_row_conv_masked=self.single_row_conv_masked)
            packed = PackedLfm2(self, binding, state, options.get("attention_backend"))
            if options.get("validate", True):
                state.report = packed.validate(graphs=use_graphs).details
            return packed
        except BaseException:
            engine.release(rollback=True)
            raise


class PackedLfm2(CUDAEpilogueSupport):
    capabilities = Capabilities(inference_capture=True, original_forward_fallback=True)

    def __init__(self, owner, binding, state, backend):
        self.owner, self.binding, self.state = owner, binding, state
        self.attention_backend = backend
        self.composition = state.engine.composition
        self.pieces = self.composition.all()
        self._closed = False
        self._init_epilogues()

    def _require_open(self):
        if self._closed:
            raise PackedEncodersError("this packed engine has been closed; pack the model again")

    @property
    def graph_enabled(self):
        return self.state.graph_enabled

    @graph_enabled.setter
    def graph_enabled(self, enabled):
        self._require_open()
        self.state.graph_enabled = enabled

    @property
    def min_shared_prefix(self) -> int:
        """Rows that share at least this many leading tokens run that prefix once; 0, the default,
        runs every row in full. The report's `shared_prefix_rejected` says why a model can't share."""
        return self.state.engine.min_shared_prefix

    @min_shared_prefix.setter
    def min_shared_prefix(self, tokens: int) -> None:
        self._require_open()
        engine = self.state.engine
        if tokens < 0:
            raise PackedEncodersError("min_shared_prefix must be >= 0")
        if tokens and engine.share_rejected:
            raise PackedEncodersError(f"shared prefixes are unavailable here: {engine.share_rejected}")
        engine.min_shared_prefix = max(int(tokens), engine.conv_width) if tokens else 0

    def forward_packed(self, batch):
        """Final-normed hidden states for flat device int64 ids and positive host_lengths;
        positions restart at zero for each sequence. Inference only."""
        self._require_open()
        ids, lengths = batch.input_ids, batch.host_lengths
        family = self.owner.family
        if torch.is_grad_enabled():
            raise PackedEncodersError(f"{family} forward_packed requires no_grad or inference_mode")
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device != self.state.engine.device:
            raise PackedEncodersError(f"{family} requires flat int64 token IDs on the weights' device")
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise PackedEncodersError(f"{family} requires positive host_lengths summing to the token count")
        if batch.cu_seqlens is not None or batch.position_ids is not None or batch.max_seqlen is not None:
            raise PackedEncodersError(f"{family} takes host_lengths only; positions restart per sequence")
        return _hidden(self.state, ids, lengths)

    def forward_grouped(self, batch, groups, *, inputs_embeds: Tensor | None = None) -> Tensor:
        """Rows laid out `[context][group 1]..[group N]`, each read in one pass with grouped
        attention. On full-attention layers a context token reads every token of its row and a
        group token its row's context and its own group, both ways; sliding-window and convolution
        layers stay causal over the row. Groups are independent within one full-attention layer
        only: each writes into the context there and reads it back at the next, and the causal
        layers carry the groups before it, so across the stack a group hears the others.
        A model trained to read its questions after one shared state this way is served as trained.
        A row with one group reads all of itself, as does a row of context alone.

        `batch`: flat device int64 `input_ids` with positive `host_lengths` (positions restart at
        zero per row), or `input_ids=None` with `inputs_embeds` (tokens, hidden) in their place.
        `groups[i]`: row i's group lengths in order, host ints > 0 summing to at most its length;
        the tokens before them are its context. Returns final-normed hidden states (tokens, hidden),
        eager. Inference only; `validate()` checks it against the model's own forward under the
        same masks (`grouped_attention_masks`)."""
        self._require_open()
        engine, family = self.state.engine, self.owner.family
        ids, lengths = batch.input_ids, batch.host_lengths
        if torch.is_grad_enabled():
            raise PackedEncodersError(f"{family} forward_grouped requires no_grad or inference_mode")
        if engine.grouped_rejected:
            raise PackedEncodersError(f"grouped rows are unavailable here: {engine.grouped_rejected}")
        if (ids is None) == (inputs_embeds is None):
            raise PackedEncodersError("forward_grouped takes input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device != engine.device:
                raise PackedEncodersError(f"{family} requires flat int64 token IDs on the weights' device")
            tokens = ids.numel()
        else:
            if (inputs_embeds.ndim != 2 or inputs_embeds.shape[1] != engine.hidden_size
                    or inputs_embeds.device != engine.device):
                raise PackedEncodersError("inputs_embeds must be (tokens, hidden) on the weights' device")
            tokens = inputs_embeds.shape[0]
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != tokens:
            raise PackedEncodersError(f"{family} requires positive host_lengths summing to the token count")
        if batch.cu_seqlens is not None or batch.position_ids is not None or batch.max_seqlen is not None:
            raise PackedEncodersError(f"{family} takes host_lengths only; positions restart per sequence")
        groups = [list(own) for own in groups]
        if len(groups) != len(lengths) or any(
                any(type(g) is not int or g <= 0 for g in own) or sum(own) > n for n, own in zip(lengths, groups)):
            raise PackedEncodersError("groups needs one list per row of positive group lengths within the row")
        return engine.forward_grouped(ids, lengths, groups, embeds=inputs_embeds)

    def validate(self, **kwargs):
        self._require_open()
        return ValidationResult(self.owner.name, self.owner._validate(
            self.binding.patch_target, _engine=self.state.engine, _state=self.state, **kwargs))

    def configure(self, options):
        self._require_open()
        _check_options(options, self.owner.family)
        if "attention_backend" in options and options["attention_backend"] != self.attention_backend:
            raise PackedEncodersError("unpack before changing the prepared attention backend")
        if "cuda_graph" in options and options["cuda_graph"] is not None:
            graph = options["cuda_graph"]
            self.set_cuda_graph(bool(graph), graph if isinstance(graph, PaddedGraphConfig) else None)

    def set_cuda_graph(self, enabled, config=None):
        self._require_open()
        self.state.set_cuda_graph(enabled, config)

    def set_train_cuda_graph(self, enabled, config=None):
        self._require_open()
        if enabled or config is not None:
            raise PackedEncodersError(f"training graphs are not implemented for {self.owner.family}")

    def close(self, *, rollback=False):
        if self._closed:
            return
        self._close_epilogues()
        self.state.runner = None
        self.state.shared_runner = None
        self.state.graph_enabled = False
        self.state.engine.release(rollback=rollback)
        self.state.engine._stager = None
        self.state.stager = None
        self._closed = True


__all__ = ["HFLfm2Adapter", "Lfm2Hybrid", "Lfm2Report", "PackedLfm2", "grouped_attention_masks"]
