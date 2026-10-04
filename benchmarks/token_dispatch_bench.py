"""Calibrate the Triton/CuteDSL token-count crossover for fused tail kernels.

This benchmark intentionally calls the public kernel wrappers: allocation, DLPack
conversion, stream wrapping, and launch dispatch are part of the decision made by
``packed_encoders.ops``.  The leading dimensions are also varied at a fixed number
of rows to verify that packed batch/sequence boundaries do not affect the result.

Example (RTX 5090):

    .venv/bin/python benchmarks/token_dispatch_bench.py \
        --output benchmarks/results/token_dispatch_5090.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from packed_encoders._kernels.geglu import geglu as cute_geglu
from packed_encoders._kernels.layer_norm import layer_norm as cute_layer_norm
from packed_encoders._kernels.triton_layer_norm import (
    geglu as triton_geglu,
    layer_norm as triton_layer_norm,
)


DEFAULT_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
)


def _parse_tokens(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(","))
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("tokens must be positive comma-separated integers")
    return values


def _synchronize() -> None:
    torch.cuda.synchronize()


def _warm_gpu(seconds: float = 1.0) -> None:
    """Bring clocks out of the idle state before collecting paired samples."""
    a = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(a, b)
    _synchronize()
    del a, b


def _estimate_iterations(fns: tuple[Callable[[], object], ...], target_ms: float) -> int:
    probe = 5
    _synchronize()
    start = time.perf_counter()
    for _ in range(probe):
        for fn in fns:
            fn()
    _synchronize()
    per_pair = (time.perf_counter() - start) / probe
    return max(2, min(20_000, math.ceil((target_ms / 1000.0) / per_pair)))


def _paired_time(
    first: Callable[[], object],
    second: Callable[[], object],
    *,
    samples: int,
    target_ms: float,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Time two wrappers in alternating order and return per-call microseconds."""
    for _ in range(5):
        first()
        second()
    _synchronize()
    iterations = _estimate_iterations((first, second), target_ms)

    values = ([], [])
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        fns = (first, second)
        for index in order:
            _synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                fns[index]()
            _synchronize()
            values[index].append(
                (time.perf_counter() - start) * 1e6 / iterations
            )

    def summary(xs: list[float]) -> dict[str, float]:
        ordered = sorted(xs)
        return {
            "median_us": statistics.median(ordered),
            "min_us": ordered[0],
            "max_us": ordered[-1],
        }

    return summary(values[0]), summary(values[1]), iterations


def _bench_shape(
    *,
    kernel: str,
    tokens: int,
    leading_shape: tuple[int, ...],
    hidden: int,
    intermediate: int,
    samples: int,
    target_ms: float,
) -> dict[str, object]:
    if math.prod(leading_shape) != tokens:
        raise ValueError(f"{leading_shape=} does not contain {tokens} tokens")

    if kernel == "layer_norm":
        x = torch.randn(*leading_shape, hidden, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        first = lambda: triton_layer_norm(x, weight, 1e-5)
        second = lambda: cute_layer_norm(x, weight, 1e-5)
    elif kernel == "geglu":
        x = torch.randn(
            *leading_shape, 2 * intermediate, device="cuda", dtype=torch.bfloat16
        )
        first = lambda: triton_geglu(x)
        second = lambda: cute_geglu(x)
    else:  # pragma: no cover - internal caller controls this
        raise ValueError(kernel)

    # First calls compile and populate wrapper caches; neither belongs in timing.
    first()
    second()
    _synchronize()
    triton, cute, iterations = _paired_time(
        first, second, samples=samples, target_ms=target_ms
    )
    speedup = triton["median_us"] / cute["median_us"]
    result = {
        "kernel": kernel,
        "tokens": tokens,
        "leading_shape": list(leading_shape),
        "iterations": iterations,
        "triton": triton,
        "cute": cute,
        "cute_speedup": speedup,
        "winner": "cute" if speedup > 1.0 else "triton",
    }
    del x
    if kernel == "layer_norm":
        del weight
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _print_row(row: dict[str, object]) -> None:
    triton = row["triton"]["median_us"]
    cute = row["cute"]["median_us"]
    mark = "C" if row["winner"] == "cute" else "T"
    print(
        f"  [{mark}] M={row['tokens']:>6} shape={str(tuple(row['leading_shape'])):<16} "
        f"triton={triton:>9.3f} us  cute={cute:>9.3f} us  "
        f"T/C={row['cute_speedup']:.3f}x",
        flush=True,
    )


def _factorizations(tokens: int) -> list[tuple[int, ...]]:
    shapes = [(tokens,)]
    for batch in (8, 128):
        if tokens % batch == 0:
            shapes.append((batch, tokens // batch))
    return shapes


def run_benchmark(
    *,
    tokens: tuple[int, ...] = DEFAULT_TOKENS,
    hidden: int = 768,
    intermediate: int = 1152,
    samples: int = 7,
    target_ms: float = 40.0,
    factorization_tokens: tuple[int, ...] = (8192, 262144),
    kernels: tuple[str, ...] = ("layer_norm", "geglu"),
    verbose: bool = True,
) -> dict[str, object]:
    """Run the sweep and return a JSON-serializable result dictionary."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    metadata = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(cc),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cutlass_dsl": __import__("cutlass").__version__,
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "python": platform.python_version(),
        "dtype": "bfloat16",
        "hidden": hidden,
        "intermediate": intermediate,
        "samples": samples,
        "target_ms": target_ms,
        "kernels": list(kernels),
        "timing": "median wall time over repeated public-wrapper calls, synchronized per sample",
    }
    if verbose:
        print(json.dumps(metadata, indent=2), flush=True)
    _warm_gpu()

    rows: list[dict[str, object]] = []
    tokens = tuple(dict.fromkeys(tokens))
    for kernel in kernels:
        if verbose:
            print(f"\n{kernel} forward sweep", flush=True)
        # A reverse pass spreads any temperature/clock drift across small and large M.
        for direction, values in (("ascending", tokens), ("descending", tokens[::-1])):
            if verbose:
                print(f" {direction}", flush=True)
            for count in values:
                row = _bench_shape(
                    kernel=kernel,
                    tokens=count,
                    leading_shape=(count,),
                    hidden=hidden,
                    intermediate=intermediate,
                    samples=samples,
                    target_ms=target_ms,
                )
                row["direction"] = direction
                rows.append(row)
                if verbose:
                    _print_row(row)

    factorization_rows: list[dict[str, object]] = []
    for factor_count in factorization_tokens:
        if verbose:
            print(f"\nfixed-M factorization check (M={factor_count})", flush=True)
        for kernel in kernels:
            for shape in _factorizations(factor_count):
                row = _bench_shape(
                    kernel=kernel,
                    tokens=factor_count,
                    leading_shape=shape,
                    hidden=hidden,
                    intermediate=intermediate,
                    samples=samples,
                    target_ms=target_ms,
                )
                factorization_rows.append(row)
                if verbose:
                    _print_row(row)

    return {
        "metadata": metadata,
        "rows": rows,
        "factorization_rows": factorization_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_parse_tokens, default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--intermediate", type=int, default=1152)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=40.0)
    parser.add_argument(
        "--factorization-tokens",
        type=_parse_tokens,
        default=(8192, 262144),
        help=(
            "Fixed token counts tested as flat, B=8, and B=128 tensors "
            "(comma-separated)"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_benchmark(
        tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        samples=args.samples,
        target_ms=args.target_ms,
        factorization_tokens=args.factorization_tokens,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
