"""Same-environment ModernBERT refactor comparison; one source tree per process.

Run with --source pointing at an immutable baseline or the candidate checkout.
Synthetic, seeded token batches isolate the encoder from tokenizer/data changes.
These timings cover encoder execution, not end-to-end encode or retrieval quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--cases", help="Only measure case names containing this substring")
    ap.add_argument("--cases-regex", help="Only measure case names matching this regular expression")
    ap.add_argument("--prepared", action="store_true", help="Use the new prepared packed boundary when available")
    args = ap.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import torch
    from transformers import AutoModel
    import packed_encoders as pe
    from packed_encoders.config import ModernBertParams
    from packed_encoders.forward import packed_forward

    assert Path(pe.__file__).resolve().is_relative_to(source), pe.__file__
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = AutoModel.from_pretrained(
        "answerdotai/ModernBERT-base", dtype=torch.bfloat16,
    ).cuda().eval()
    params = ModernBertParams.from_hf_config(model.config)
    results = {
        "source": str(source), "import": pe.__file__,
        "source_sha256": hashlib.sha256(b"".join(
            str(p.relative_to(source)).encode() + p.read_bytes()
            for p in sorted((source / "packed_encoders").rglob("*.py"))
        )).hexdigest(),
        "model_revision": model.config._commit_hash,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "python": sys.version,
        "prepared_boundary": args.prepared and hasattr(pe, "get_engine"),
        "versions": {k: importlib.metadata.version(k) for k in (
            "torch", "triton", "transformers", "nvidia-cutlass-dsl", "flash-attn",
        )}, "cases": {},
    }

    def measure(name, fn, ids):
        if args.cases and args.cases not in name:
            return
        if args.cases_regex and not re.search(args.cases_regex, name):
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) / args.iterations)
        results["cases"][name] = {
            "seconds": samples, "median_seconds": statistics.median(samples),
            "cold_seconds": cold,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "input_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        }
        output = fn()
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        if output is not None:
            values = output.detach().float().cpu()
            results["cases"][name]["output_sha256"] = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        print(name, results["cases"][name]["median_seconds"], flush=True)

    def batch(lengths):
        gen = torch.Generator().manual_seed(17)
        flat = torch.randint(5, model.config.vocab_size, (sum(lengths),), generator=gen).cuda()
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        pos = torch.cat([torch.arange(n, device="cuda") for n in lengths])
        padded = torch.zeros(len(lengths), max(lengths), dtype=torch.long, device="cuda")
        mask = torch.zeros_like(padded)
        offset = 0
        for row, n in enumerate(lengths):
            padded[row, :n] = flat[offset:offset + n]
            mask[row, :n] = 1
            offset += n
        return flat, cu, pos, padded, mask

    with torch.inference_mode():
        for label, lengths in (("query", (17, 24, 31, 40, 48, 53, 60, 64)),
                               ("document", (256, 384, 512, 1024))):
            flat, cu, pos, ids, mask = batch(lengths)
            for backend in ("flash", "auto", "sdpa"):
                for graphs in (False, True):
                    cfg = pe.GraphConfig(max_batch=8, max_seq=128, pad_to=32) if graphs else False
                    start = time.perf_counter()
                    pe.pack(model, attention_backend=backend, cuda_graph=cfg, validate=False)
                    torch.cuda.synchronize()
                    prep = time.perf_counter() - start
                    for entry in ("padded", "packed"):
                        name = f"{label}/{backend}/graphs={graphs}/{entry}"
                        fn = (lambda: model(ids, mask)) if entry == "padded" else (
                            lambda: packed_forward(model, params, flat, cu, max(lengths), pos))
                        if entry == "packed" and args.prepared and hasattr(pe, "get_engine"):
                            engine = pe.get_engine(model)
                            packed_batch = pe.PackedBatch(flat, cu, max(lengths), pos)
                            fn = lambda: engine.forward_packed(packed_batch)
                        measure(name, fn, flat)
                        if name in results["cases"]:
                            results["cases"][name]["prepare_seconds"] = prep
                    pe.unpack(model)
                    gc.collect()
                    torch.cuda.empty_cache()

    model.train()
    _, _, _, ids, mask = batch((64, 64))
    for graphs in (False, True):
        pe.pack(model, attention_backend="sdpa", train_cuda_graph=graphs, validate=False)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        def step():
            model.zero_grad(set_to_none=False)
            loss = model(ids, mask).last_hidden_state.float().square().mean()
            loss.backward()
            return loss.detach()

        measure(f"training/graphs={graphs}", step, ids)
        pe.unpack(model)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
