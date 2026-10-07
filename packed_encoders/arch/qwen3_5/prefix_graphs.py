"""Full encoder CUDA graphs with host-planned, fixed prefix/suffix geometry."""
from collections import OrderedDict
import weakref

import torch

from packed_encoders.errors import PackedEncodersError


def retain_fla_metadata(engine, layout):
    """Pin the chunk indices consumed by FLA 0.5's conv and GDN kernels.

    Warmup populates FLA's bounded identity cache outside the graph pool. CUDA
    graphs only retain its addresses, so global cache eviction must not free it.
    These calls match FLA's argument identities and its 64-token chunk size.
    """
    if not any(layer.linear for layer in engine.layers):
        return []
    from fla.ops.utils.index import prepare_chunk_indices
    from packed_encoders.arch.qwen3_5.engine import fla_tensor_cache
    pairs = [(layout.cu, layout.cu_cpu)] if not engine.fused else []
    if layout.share is None:
        pairs.append((layout.cu, layout.cu_cpu))
    else:
        sh = layout.share
        pairs.extend([(sh.cu_roots, sh.cu_roots_cpu), (sh.cu_kids, sh.cu_kids_cpu)])
    with fla_tensor_cache():
        return [prepare_chunk_indices(cu, 64, cu_seqlens_cpu=cpu) for cu, cpu in pairs]


class CapturedForward:
    """Fixed-address inputs and output; callers clone the output before the next replay."""

    @torch.inference_mode()
    def __init__(self, engine, count, forward, *, layout, warmup=2, pool=None):
        with torch.cuda.device(engine.device):
            self.ids = torch.zeros(count, dtype=torch.long, device=engine.device)
            side = torch.cuda.Stream(device=engine.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(2, warmup)):
                    forward(self.ids)
                self._fla_metadata = retain_fla_metadata(engine, layout)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=pool, stream=side):
                self.output = forward(self.ids)

    def replay(self, ids):
        self.ids.copy_(ids)
        self.graph.replay()
        return self.output.clone()


class SuffixGraph:
    """An explicit graph for one prepared prefix and fixed positive suffix lengths.

    Token values can change on every call. Outputs own their storage. Close the
    handle to release the graph's activation pool (separate from prefix.nbytes).
    """

    @torch.inference_mode()
    def __init__(self, prefix, lengths):
        engine = prefix._require_valid()
        self._prefix = weakref.ref(prefix)
        self.lengths = tuple(lengths)
        self._captured = None
        with torch.cuda.device(engine.device):
            engine.sync_norms()
            self._layout = engine.prepare_prefix_layout(list(lengths), prefix)
            self._captured = CapturedForward(engine, sum(lengths),
                                             lambda ids: engine.prefix_core(ids, self._layout), layout=self._layout)

    def __call__(self, input_ids):
        if self._captured is None:
            raise PackedEncodersError("suffix graph is closed")
        prefix = self._prefix()
        if prefix is None:
            self.close()
            raise PackedEncodersError("prepared prefix is closed")
        engine = prefix._require_valid()
        prefix._check_ids(engine, input_ids, self.lengths)
        with torch.inference_mode(), torch.cuda.device(engine.device):
            return self._captured.replay(input_ids)

    def close(self):
        self._captured = None
        self._layout = None

    def __enter__(self):
        if self._captured is None:
            raise PackedEncodersError("suffix graph is closed")
        return self

    def __exit__(self, *exc):
        self.close()


class SharedGraphRunner:
    """Bounded graphs for exact sharing plans; token values are never part of the key.

    Prefix discovery stays outside capture. All embedding, encoder layers, and
    reconstruction of shared tokens execute in one replay. The pool is shared
    across plans; calls must be serialized, like the padded graph runner.
    """

    def __init__(self, engine, config):
        self.engine, self.config = engine, config
        self._cache = OrderedDict()
        self._pool = None

    @property
    def num_graphs(self):
        return len(self._cache)

    @staticmethod
    def _key(plan):
        # Include mappings, not just shapes: equal segment lengths can have
        # different parents, source row order, and output reconstruction.
        return (tuple(plan.lengths), tuple(plan.kv_lengths), plan.n_roots, plan.root_tokens,
                *(tuple(getattr(plan, name).reshape(-1).tolist()) for name in
                  ('src', 'out', 'rope_pos', 'conv_pos', 'parent', 'fix_rows', 'fix_taps', 'kv_idx')))

    @torch.inference_mode()
    def __call__(self, ids, plan):
        cfg = self.config
        if ids.numel() > cfg.max_tokens or max(plan.kv_lengths) > cfg.max_seq or cfg.max_graphs <= 0:
            return None
        with torch.cuda.device(self.engine.device):
            key = self._key(plan)
            entry = self._cache.get(key)
            if entry is None:
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                static = self.engine.prepare_shared_layout(plan)
                cap = CapturedForward(self.engine, ids.numel(), lambda x: self.engine.shared_core(x, static),
                                      layout=static.layout, warmup=cfg.warmup, pool=self._pool)
                entry = (cap, static)  # retain every metadata tensor used by the graph
                self._cache[key] = entry
                while len(self._cache) > cfg.max_graphs:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return entry[0].replay(ids.reshape(-1))
