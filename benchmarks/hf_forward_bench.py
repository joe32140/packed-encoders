"""Any model `pack()` supports, through its HF forward: latency and parity against the model's own forward.

    python benchmarks/hf_forward_bench.py --model <model id> --out forward.json [--engine qwen3_5|lfm2|...]

`AutoModel` loads the checkpoint; the registry picks the engine (`--engine` overrides it) and the
module it patches, which this script times with token ids. When the loaded model wraps that module
(a multimodal wrapper, say), the wrapper is timed too. Rows are random tokens from the regular
vocabulary, right padded. Two kinds of batch:
  rows    independent rows: one 128-token row, 8 and 32 rows of 32 to 512 tokens, 8 x 1,024 tokens
  shared  one `--state`-token state, then 16 to 40 question tokens per row (`--questions` rows)

Per batch, ms (median of `--trials` passes after a warmup that also captures graphs):
  stock          the model's own forward, before pack
  packed_graphs  sharing off, CUDA graphs on (pack's default)
  packed_eager   sharing off, graphs off
  shared_graphs  `min_shared_prefix = --min-prefix`, graphs on (shared batches, engines that share)
  shared_eager   the same, graphs off
  wrapper        the loaded model, pack's defaults (sharing on for shared batches), when it wraps the target
These names describe requested modes. `execution` records the path actually used in an
untimed parity call, including eager fallback for inputs outside the graph bounds.
Parity per real token: per-token cosine of each packed variant against stock.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import statistics
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F


def make_batch(rows):
    S = max(map(len, rows))
    ids, mask = torch.zeros(len(rows), S, dtype=torch.long), torch.zeros(len(rows), S, dtype=torch.long)
    for i, x in enumerate(rows):
        ids[i, : len(x)], mask[i, : len(x)] = x, 1
    return ids.cuda(), mask.cuda()


def workloads(vocab, states, questions):
    def rand(g, n):
        return torch.randint(0, vocab, (n,), generator=g)

    g = torch.Generator().manual_seed(0)
    out = {"1 x 128": ("rows", [rand(g, 128)])}
    for n in (8, 32):
        out[f"{n} rows, 32-512"] = ("rows", [rand(g, int(k)) for k in torch.randint(32, 513, (n,), generator=g)])
    out["8 x 1024"] = ("rows", [rand(g, 1024) for _ in range(8)])
    for P, N in zip(states, questions):
        state = rand(g, P)
        out[f"state {P}, {N} questions"] = (
            "shared", [torch.cat([state, rand(g, int(k))]) for k in torch.randint(16, 41, (N,), generator=g)])
    return {name: (kind, make_batch(rows)) for name, (kind, rows) in out.items()}


@torch.no_grad()
def ms(fn, trials):
    for _ in range(3):                                       # warmup, graph capture
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def cosine(x, ref):
    cos = F.cosine_similarity(x.float(), ref, dim=-1)
    return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item()}


def record_execution(packed, fn):
    """Trace one untimed call; leave timed passes and the runtime's hot path untouched."""
    from packed_encoders.runtime.shared_graphs import SharedGraphRunner

    counts = dict(packed_graph=0, shared_graph=0, packed_eager=0, shared_eager=0, hf_fallback=0)
    state = packed.state
    with ExitStack() as stack:
        def watch(owner, name, path):
            original = getattr(owner, name, None)
            if not callable(original):
                return

            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                if path != "shared_graph" or result is not None:
                    counts[path] += 1
                return result

            stack.enter_context(patch.object(owner, name, observed))

        engine = getattr(state, "engine", None)
        watch(engine, "forward_packed", "packed_eager")
        watch(engine, "forward_shared", "shared_eager")
        watch(getattr(state, "runner", None), "_replay", "packed_graph")
        watch(SharedGraphRunner, "__call__", "shared_graph")
        watch(state, "original_forward", "hf_fallback")
        output = fn()
    paths = [name for name, count in counts.items() if count]
    return output, {"path": paths[0] if len(paths) == 1 else "mixed" if paths else "unknown", "calls": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", help="a registered engine name; default: the registry's choice")
    ap.add_argument("--state", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--questions", type=int, nargs="+", default=[8, 8, 16])
    ap.add_argument("--min-prefix", type=int, default=64)
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    if len(a.state) != len(a.questions):
        raise SystemExit("--state and --questions pair up: give as many of each")

    from transformers import AutoModel

    import packed_encoders as pe
    from packed_encoders.arch import registered
    from packed_encoders.arch.weights import validation_ids_below
    from packed_encoders.locate import select_engine

    engine = None
    if a.engine:
        engine = next((e for e in registered() if e.name == a.engine), None)
        if engine is None:
            raise SystemExit(f"no engine {a.engine!r}; registered: {[e.name for e in registered()]}")
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    chosen, binding = select_engine(model, engine=engine)
    target = binding.patch_target
    wrapper = model if target is not model else None
    cfg = getattr(target.config, "text_config", target.config)
    kwargs = {"use_cache": False} if hasattr(cfg, "use_cache") else {}

    def hidden(m, ids, mask):
        return m(input_ids=ids, attention_mask=mask, **kwargs).last_hidden_state

    res = {"model": a.model, "engine": chosen.name, "target": type(target).__name__,
           "wrapper": type(wrapper).__name__ if wrapper is not None else None, "gpu": torch.cuda.get_device_name(0),
           "torch": str(torch.__version__), "stock_attention": getattr(cfg, "_attn_implementation", None),
           "batches": {}}
    data = workloads(validation_ids_below(cfg), a.state, a.questions)
    stock = {}
    with torch.no_grad():
        for name, (_, (ids, mask)) in data.items():          # the model's own forward, before pack
            stock[name] = (hidden(target, ids, mask)[mask.bool()].float(),
                           ms(lambda: hidden(target, ids, mask), a.trials))

    pe.pack(model, engine=engine)
    packed = pe.get_engine(model)
    rep = getattr(packed.state, "report", None)
    shares = hasattr(packed, "min_shared_prefix") and not getattr(rep, "shared_prefix_rejected", "unsupported")
    res["pack"] = {"attention": getattr(rep, "attention_backend", None), "shares_prefixes": shares,
                   "shared_prefix_rejected": getattr(rep, "shared_prefix_rejected", None)}
    print(f"{a.model} | {res['engine']} on {res['target']} | {res['gpu']} | attention {res['pack']['attention']} | "
          f"shared prefixes: {shares}", flush=True)
    for name, (kind, (ids, mask)) in data.items():
        ref, t_stock = stock[name]
        real = mask.bool()
        share = a.min_prefix if kind == "shared" and shares else 0
        variants = {"packed_graphs": (target, True, 0), "packed_eager": (target, False, 0)}
        if share:
            variants.update(shared_graphs=(target, True, share), shared_eager=(target, False, share))
        if wrapper is not None:
            variants["wrapper"] = (wrapper, True, share)
        row = {"kind": kind, "rows": ids.shape[0], "tokens": int(real.sum()), "padding": 1 - real.float().mean().item(),
               "ms": {"stock": t_stock}, "parity": {}, "execution": {}}
        for variant, (m, graphs, min_prefix) in variants.items():
            if shares:
                packed.min_shared_prefix = min_prefix
            with torch.no_grad() if graphs else pe.no_cuda_graph(model):
                row["ms"][variant] = ms(lambda: hidden(m, ids, mask), a.trials)
                with torch.no_grad():
                    output, execution = record_execution(packed, lambda: hidden(m, ids, mask))
                    row["execution"][variant] = execution
                    row["parity"][variant] = cosine(output[real], ref)
        if share:
            packed.min_shared_prefix = share
            plan = packed.state.engine.plan_sharing(ids[real], real.sum(1).tolist())
            row["saved"] = plan.saved_tokens / row["tokens"] if plan else 0.0
            packed.min_shared_prefix = 0
        row["speedup_vs_stock_by_variant"] = {k: t_stock / t for k, t in row["ms"].items() if k != "stock"}
        row["fastest_variant"] = max(row["speedup_vs_stock_by_variant"], key=row["speedup_vs_stock_by_variant"].get)
        row["speedup_vs_stock"] = row["speedup_vs_stock_by_variant"][row["fastest_variant"]]
        res["batches"][name] = row
        times = "  ".join(f"{k} {t:.1f}" for k, t in row["ms"].items())
        worst = min(p["cos_min"] for p in row["parity"].values())
        print(f"{name:>24} | pad {row['padding']:.1%} | {times} | {row['speedup_vs_stock']:.2f}x | "
              f"cos min {worst:.4f}", flush=True)
        print("  execution: " + "  ".join(f"{k}={v['path']}" for k, v in row["execution"].items()), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("HF FORWARD BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
