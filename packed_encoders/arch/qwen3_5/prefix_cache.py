"""Explicit, inference-only prefix ownership for causal Qwen3.5 backbones."""
from dataclasses import dataclass
import weakref

import torch

from packed_encoders.errors import PackedEncodersError


@dataclass
class PrefixLayer:
    history: torch.Tensor | None = None
    state: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


class PreparedPrefix:
    """GPU state for one prefix, bound to one packed engine and weight version.

    `forward_suffixes` returns only the suffix tokens, in packed order, before
    any wrapper head. `hidden_states` contains the prefix's final hidden states.
    Treat all cached tensors as read-only. Call `close()` to release storage;
    unpacking the model also closes its handles. Mutation through `.data` or an
    external raw pointer bypasses PyTorch versioning and requires explicit close.
    """

    def __init__(self, packed, ids, max_bytes):
        packed._require_open()
        engine = packed.state.engine
        self._owner = weakref.ref(packed)
        self.layers = []
        self.hidden_states = None
        self._closed = False
        self.num_tokens = ids.numel()
        self.nbytes = 0
        if torch.is_grad_enabled():
            raise PackedEncodersError("prepare_prefix requires no_grad or inference_mode")
        if torch.is_autocast_enabled("cuda"):
            raise PackedEncodersError("prepared prefixes require autocast disabled")
        if not engine.causal or engine.share_rejected:
            raise PackedEncodersError(f"prefix reuse unavailable: {engine.share_rejected or 'noncausal model'}")
        self._check_ids(engine, ids, [self.num_tokens])
        rope = getattr(engine.cfg, "rope_parameters", {}) or {}
        if rope.get("rope_type", "default") != "default":
            raise PackedEncodersError("prepared prefixes currently require default, length-independent RoPE")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise PackedEncodersError("max_bytes must be a positive integer")
        element = engine.embed.element_size()
        estimate = self.num_tokens * engine.hidden_size * element
        for layer in engine.layers:
            if layer.linear:
                estimate += (layer.conv_w.shape[1] - 1) * layer.bounds[-1] * element
                estimate += layer.nv * layer.hk * layer.hv * 4
            else:
                estimate += 2 * self.num_tokens * layer.nkv * layer.hd * element
        if estimate > max_bytes:
            raise PackedEncodersError(f"prefix needs {estimate} bytes of retained state; budget is {max_bytes}")
        self.layers = [PrefixLayer() for _ in engine.layers]
        try:
            self.hidden_states = engine.forward_prefix(ids, [self.num_tokens], self, write=True).clone()
            tensors = [self.hidden_states] + [t for layer in self.layers for t in vars(layer).values() if t is not None]
            self.nbytes = sum(t.untyped_storage().nbytes() for t in tensors)
            if self.nbytes > max_bytes:
                raise PackedEncodersError(f"prefix retained {self.nbytes} bytes; budget is {max_bytes}")
            self._versions = self._weight_versions(engine)
            engine._prefix_caches.add(self)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _weight_versions(engine):
        return tuple((id(p), p.data_ptr(), p._version) for p in engine.tm.parameters())

    @staticmethod
    def _check_ids(engine, ids, lengths):
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device != engine.device:
            raise PackedEncodersError("prefix reuse requires flat int64 IDs on the weights' device")
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise PackedEncodersError("prefix reuse requires positive host lengths summing to the token count")

    def forward_suffixes(self, input_ids, host_lengths):
        """Run each packed suffix from the same immutable prefix; returns [sum(lengths), hidden]."""
        if self._closed:
            raise PackedEncodersError("prepared prefix is closed")
        owner = self._owner()
        if owner is None or owner._closed:
            self.close()
            raise PackedEncodersError("prepared prefix's packed model has been closed")
        if torch.is_grad_enabled():
            raise PackedEncodersError("forward_suffixes requires no_grad or inference_mode")
        if torch.is_autocast_enabled("cuda"):
            raise PackedEncodersError("prepared prefixes require autocast disabled")
        engine = owner.state.engine
        if self._versions != self._weight_versions(engine):
            self.close()
            raise PackedEncodersError("model weights changed; prepare the prefix again")
        self._check_ids(engine, input_ids, host_lengths)
        return engine.forward_prefix(input_ids, list(host_lengths), self)

    def close(self):
        self.layers.clear()
        self.hidden_states = None
        self.nbytes = 0
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise PackedEncodersError("prepared prefix is closed")
        return self

    def __exit__(self, *exc):
        self.close()
