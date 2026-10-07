"""Shared prefixes on a causal Qwen3.5 model: throughput with and without sharing, and parity.

    python benchmarks/qwen35_shared_prefix_bench.py --model Qwen/Qwen3.5-0.8B --out r.json

Any checkpoint `AutoModel` loads as a causal `Qwen3_5Model` or `Qwen3_5TextModel` works; a decision
model packs its backbone the same way. Rows are random tokens. Per setting, `--rows` rows come in
groups of k (`--rows-per-prefix`) that share a `--prefix`-token prefix and continue with 1 to
`--suffix` tokens of their own. A group's rows are consecutive, batched `--batch` at a time and
padded on `--padding-side`.

Variants on one packed model, bf16, through the HF forward (inputs already on the GPU, one warmup
pass that also captures graphs, median of `--trials` passes):
  full_graphs  sharing off (the default), CUDA graphs on (pack's default)
  full_eager   sharing off, graphs off
  shared       `min_shared_prefix = --min-prefix` (always eager)
`saved` is the fraction of the setting's tokens the shared variant does not compute. Parity per
real token: shared and full_eager against the model's own forward, before pack.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time

import torch
import torch.nn.functional as F


def make_batches(g, vocab, *, rows, k, prefix, suffix, batch, side):
    seqs = []
    for _ in range(rows // k):
        p = torch.randint(0, vocab, (prefix,), generator=g)
        for _ in range(k):
            n = int(torch.randint(1, suffix + 1, (1,), generator=g))
            seqs.append(torch.cat([p, torch.randint(0, vocab, (n,), generator=g)]))
    out = []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s: s + batch]
        S = max(map(len, chunk))
        ids, mask = torch.zeros(len(chunk), S, dtype=torch.long), torch.zeros(len(chunk), S, dtype=torch.long)
        for i, x in enumerate(chunk):
            cols = slice(S - len(x), S) if side == "left" else slice(0, len(x))
            ids[i, cols], mask[i, cols] = x, 1
        out.append((ids.cuda(), mask.cuda()))
    return out


@torch.no_grad()
def run_pass(model, batches, collect=False):
    outs = []
    for ids, mask in batches:
        h = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        if collect:
            outs.append(h[mask.bool()].float())
    torch.cuda.synchronize()
    return torch.cat(outs) if collect else None


def timed(model, batches, trials):
    out = run_pass(model, batches, collect=True)          # warmup, graph capture, outputs
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        run_pass(model, batches)
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def cosine(x, ref):
    cos = F.cosine_similarity(x, ref, dim=-1)
    return {"cos_mean": cos.mean().item(), "cos_min": cos.min().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--suffix", type=int, nargs="+", default=[32, 1024])
    ap.add_argument("--rows-per-prefix", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--padding-side", choices=("left", "right"), default="left")
    ap.add_argument("--min-prefix", type=int, default=64)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()

    from transformers import AutoModel

    import packed_encoders as pe
    from packed_encoders.arch.qwen3_5 import _validation_ids_below

    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    cfg = getattr(model.config, "text_config", model.config)
    res = {"model": a.model, "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
           "batch": a.batch, "rows": a.rows, "padding_side": a.padding_side, "settings": []}
    g = torch.Generator().manual_seed(0)
    settings = [dict(prefix=p, suffix=s, k=k) for p, s, k in itertools.product(a.prefix, a.suffix, a.rows_per_prefix)]
    data = [make_batches(g, _validation_ids_below(cfg), rows=a.rows, batch=a.batch, side=a.padding_side, **s) for s in settings]
    stock = [run_pass(model, b, collect=True).cpu() for b in data]     # the model's own forward, before pack

    pe.pack(model)
    packed = pe.get_engine(model)
    engine, rep = packed.state.engine, packed.state.report
    res["pack"] = {"attention": rep.attention_backend, "shared_prefix_rejected": rep.shared_prefix_rejected,
                   "shared_cos_mean": rep.shared_cos_mean, "shared_cos_min": rep.shared_cos_min}
    print(f"{a.model} | {res['gpu']} | attention {rep.attention_backend} | shared pass cos "
          f"{rep.shared_cos_mean} / {rep.shared_cos_min}", flush=True)
    if rep.shared_prefix_rejected:
        raise SystemExit(f"this model can't share prefixes: {rep.shared_prefix_rejected}")
    print(f"{'prefix':>6} {'suffix':>6} {'k':>2} {'saved':>6} | {'full_graphs':>11} {'full_eager':>10} {'shared':>8} "
          f"items/s | shared/best  cos vs stock", flush=True)
    for s, batches, ref in zip(settings, data, stock):
        packed.min_shared_prefix = 0
        _, t_graphs = timed(model, batches, a.trials)
        with pe.no_cuda_graph(model):
            eager, t_eager = timed(model, batches, a.trials)
        packed.min_shared_prefix = a.min_prefix
        shared, t_shared = timed(model, batches, a.trials)
        total = saved = 0
        for ids, mask in batches:
            lengths = mask.sum(1).tolist()
            plan = engine.plan_sharing(ids[mask.bool()], lengths)
            total, saved = total + sum(lengths), saved + (plan.saved_tokens if plan else 0)
        row = {**s, "tokens": total, "saved": saved / total}
        n = a.rows
        row.update(items_per_s={"full_graphs": n / t_graphs, "full_eager": n / t_eager, "shared": n / t_shared},
                   shared_vs_best_full=min(t_graphs, t_eager) / t_shared,
                   parity={"shared_vs_stock": cosine(shared, ref.cuda()), "full_eager_vs_stock": cosine(eager, ref.cuda())})
        res["settings"].append(row)
        ips, par = row["items_per_s"], row["parity"]["shared_vs_stock"]
        print(f"{s['prefix']:>6} {s['suffix']:>6} {s['k']:>2} {row['saved']:>6.1%} | {ips['full_graphs']:>11.1f} "
              f"{ips['full_eager']:>10.1f} {ips['shared']:>8.1f}         | {row['shared_vs_best_full']:>10.2f}x  "
              f"{par['cos_mean']:.5f} / {par['cos_min']:.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("SHARED PREFIX BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
