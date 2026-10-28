"""Grouped rows (`[context][group 1]..[group N]`, `forward_grouped`) on any model whose engine serves them:
latency and parity against the model's own forward under the same masks.

    python benchmarks/grouped_rows_bench.py --model <model id> --out grouped.json [--trust-remote-code]

`AutoModel` loads the checkpoint and the registry picks the engine and the module it patches (the
text backbone). Rows are random tokens from the regular vocabulary: a context, then groups of 16 to
40 tokens. Per batch, ms (median of `--trials` passes after a warmup):
  stock    the model's own forward, right padded, under `grouped_attention_masks` (a mapping of
           additive masks per layer type, which its attention takes as given)
  grouped  `forward_grouped`, padding-free and eager
  causal   `forward_packed` on the same rows, read causally: the price of grouping
Parity per real token: per-token cosine of `grouped` against `stock`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def workloads(vocab):
    g = torch.Generator().manual_seed(0)

    def rows(n_rows, context, n_groups):
        out = []
        for _ in range(n_rows):
            sizes = torch.randint(16, 41, (n_groups,), generator=g).tolist()
            out.append((torch.randint(0, vocab, (context + sum(sizes),), generator=g), sizes))
        return out

    return {
        "1 x (64 + 4 groups)": rows(1, 64, 4),
        "1 x (64 + 16 groups)": rows(1, 64, 16),
        "64 x (64 + 4 groups)": rows(64, 64, 4),
        "1 x (1024 + 16 groups)": rows(1, 1024, 16),
        "1 x (5120 + 16 groups)": rows(1, 5120, 16),
        "8 x (2048 + 4 groups)": rows(8, 2048, 4),
    }


@torch.no_grad()
def ms(fn, trials):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true", help="load modeling code shipped with the checkpoint")
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()

    from transformers import AutoModel

    import packed_encoders as pe
    from packed_encoders.arch.lfm2 import grouped_attention_masks
    from packed_encoders.arch.weights import validation_ids_below
    from packed_encoders.batch import PackedBatch
    from packed_encoders.locate import select_engine

    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16,
                                      trust_remote_code=a.trust_remote_code).to("cuda").eval()
    chosen, binding = select_engine(model)
    target = binding.patch_target
    cfg = target.config
    data = workloads(validation_ids_below(cfg))
    inputs = {}
    for name, rows in data.items():
        lengths, groups = [len(x) for x, _ in rows], [own for _, own in rows]
        ids = torch.nn.utils.rnn.pad_sequence([x for x, _ in rows], batch_first=True).cuda()
        real = (torch.arange(ids.shape[1])[None] < torch.tensor(lengths)[:, None]).cuda()
        masks = grouped_attention_masks(cfg, lengths, groups, ids.shape[1], torch.bfloat16, "cuda")
        inputs[name] = (ids, real, masks, lengths, groups)

    def stock_call(name):
        ids, _, masks, _, _ = inputs[name]
        return target(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state

    stock = {}
    with torch.no_grad():
        for name in data:                                      # the model's own forward, before pack
            stock[name] = (stock_call(name)[inputs[name][1]].float(), ms(lambda: stock_call(name), a.trials))
    pe.pack(model)
    packed = pe.get_engine(model)
    rep = packed.state.report
    res = {"model": a.model, "engine": chosen.name, "target": type(target).__name__,
           "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
           "stock_attention": getattr(cfg, "_attn_implementation", None),
           "attention": {k: v.name for k, v in packed.state.engine.attentions.items()},
           "grouped_rejected": rep.grouped_rejected, "batches": {}}
    print(f"{a.model} | {chosen.name} on {res['target']} | {res['gpu']} | attention {res['attention']}", flush=True)
    for name in data:
        ids, real, _, lengths, groups = inputs[name]
        batch = PackedBatch(input_ids=ids[real], host_lengths=tuple(lengths))
        ref, t_stock = stock[name]
        with torch.no_grad():
            got = packed.forward_grouped(batch, groups).float()
        cos = F.cosine_similarity(got, ref, dim=-1)
        row = {"rows": len(lengths), "tokens": sum(lengths), "padding": 1 - real.float().mean().item(),
               "ms": {"stock": t_stock,
                      "grouped": ms(lambda: packed.forward_grouped(batch, groups), a.trials),
                      "causal": ms(lambda: packed.forward_packed(batch), a.trials)},
               "parity": {"cos_mean": cos.mean().item(), "cos_min": cos.min().item()}}
        row["speedup_vs_stock"] = t_stock / row["ms"]["grouped"]
        res["batches"][name] = row
        times = "  ".join(f"{k} {t:.1f}" for k, t in row["ms"].items())
        print(f"{name:>24} | pad {row['padding']:.1%} | {times} | {row['speedup_vs_stock']:.2f}x | "
              f"cos mean {row['parity']['cos_mean']:.5f} min {row['parity']['cos_min']:.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("GROUPED ROWS BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
