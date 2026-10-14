# PR #11: simplified inference candidate

Candidate: fused shared-convolution boundaries and inference-only RMSNorm. The earlier autotuning, hashing, graph-key and SDPA changes have been set aside. Only the seven existing hf_forward_bench cases are measured; no new performance probes or ablations.

Baseline: a636c5a, containing the three review fixes. Both variants use the same benchmark harness, checkpoint and attention order (torch_varlen, sdpa). Each GPU has two alternating process runs per variant, with ten measured calls per case and three warmups. Values are medians of the two run medians; wall time includes completion synchronization.

The d1 benchmark retains the explicit skip of pack-time certification because the fixed baseline failed the default numerical check on the 5090. These timings are exploratory. The harness still reports per-case parity and actual execution paths. No validation thresholds were changed.

## 5090

| Joe case | Variant | Baseline ms | Candidate ms | Reduction | Actual path |
|---|---|---:|---:|---:|---|
| 1 x 128 | stock | 8.239 | 8.170 | +0.83% | stock |
| 1 x 128 | packed_graphs | 6.221 | 6.270 | -0.79% | packed_graph |
| 1 x 128 | packed_eager | 6.542 | 6.497 | +0.68% | packed_eager |
| 1 x 128 | wrapper | 6.565 | 6.411 | +2.35% | packed_eager |
| 8 rows, 32-512 | stock | 141.059 | 141.627 | -0.40% | stock |
| 8 rows, 32-512 | packed_graphs | 83.246 | 83.163 | +0.10% | packed_graph |
| 8 rows, 32-512 | packed_eager | 66.661 | 68.342 | -2.52% | packed_eager |
| 8 rows, 32-512 | wrapper | 67.985 | 68.419 | -0.64% | packed_eager |
| 32 rows, 32-512 | stock | 562.499 | 564.412 | -0.34% | stock |
| 32 rows, 32-512 | packed_graphs | 341.308 | 343.646 | -0.68% | packed_graph |
| 32 rows, 32-512 | packed_eager | 285.479 | 287.071 | -0.56% | packed_eager |
| 32 rows, 32-512 | wrapper | 283.183 | 285.132 | -0.69% | packed_eager |
| 8 x 1024 | stock | 291.440 | 293.377 | -0.66% | stock |
| 8 x 1024 | packed_graphs | 251.766 | 251.884 | -0.05% | packed_graph |
| 8 x 1024 | packed_eager | 252.155 | 252.609 | -0.18% | packed_eager |
| 8 x 1024 | wrapper | 252.826 | 250.718 | +0.83% | packed_eager |
| state 256, 8 questions | stock | 81.796 | 81.867 | -0.09% | stock |
| state 256, 8 questions | packed_graphs | 72.530 | 71.803 | +1.00% | packed_graph |
| state 256, 8 questions | packed_eager | 68.956 | 68.926 | +0.04% | packed_eager |
| state 256, 8 questions | shared_graphs | 21.684 | 20.606 | +4.97% | shared_graph |
| state 256, 8 questions | shared_eager | 22.178 | 21.001 | +5.31% | shared_eager |
| state 256, 8 questions | wrapper | 22.003 | 20.819 | +5.38% | shared_eager |
| state 1024, 8 questions | stock | 307.582 | 309.260 | -0.55% | stock |
| state 1024, 8 questions | packed_graphs | 259.217 | 261.344 | -0.82% | packed_graph |
| state 1024, 8 questions | packed_eager | 256.857 | 256.812 | +0.02% | packed_eager |
| state 1024, 8 questions | shared_graphs | 40.548 | 39.279 | +3.13% | shared_graph |
| state 1024, 8 questions | shared_eager | 40.233 | 39.511 | +1.79% | shared_eager |
| state 1024, 8 questions | wrapper | 41.303 | 40.335 | +2.34% | shared_eager |
| state 2048, 16 questions | stock | 1228.320 | 1233.244 | -0.40% | stock |
| state 2048, 16 questions | packed_graphs | 958.525 | 964.082 | -0.58% | packed_eager |
| state 2048, 16 questions | packed_eager | 961.987 | 962.886 | -0.09% | packed_eager |
| state 2048, 16 questions | shared_graphs | 79.653 | 79.831 | -0.22% | shared_eager |
| state 2048, 16 questions | shared_eager | 80.523 | 79.692 | +1.03% | shared_eager |
| state 2048, 16 questions | wrapper | 82.753 | 81.141 | +1.95% | shared_eager |

| Case | Eager baseline runs | Eager candidate runs |
|---|---|---|
| 1 x 128 | 6.686, 6.398 | 6.539, 6.456 |
| 8 rows, 32-512 | 66.238, 67.083 | 68.618, 68.066 |
| 32 rows, 32-512 | 283.865, 287.094 | 286.711, 287.431 |
| 8 x 1024 | 251.277, 253.033 | 252.131, 253.088 |
| state 256, 8 questions | 22.243, 22.113 | 20.910, 21.092 |
| state 1024, 8 questions | 39.489, 40.977 | 39.980, 39.042 |
| state 2048, 16 questions | 80.073, 80.973 | 79.497, 79.886 |

Distinct eager cases with at least 20% latency reduction: 0. None.

## l40s

| Joe case | Variant | Baseline ms | Candidate ms | Reduction | Actual path |
|---|---|---:|---:|---:|---|
| 1 x 128 | stock | 23.497 | 21.216 | +9.71% | stock |
| 1 x 128 | packed_graphs | 9.661 | 9.707 | -0.47% | packed_graph |
| 1 x 128 | packed_eager | 19.498 | 14.230 | +27.02% | packed_eager |
| 1 x 128 | wrapper | 19.525 | 13.644 | +30.12% | packed_eager |
| 8 rows, 32-512 | stock | 122.865 | 124.933 | -1.68% | stock |
| 8 rows, 32-512 | packed_graphs | 74.640 | 76.529 | -2.53% | packed_graph |
| 8 rows, 32-512 | packed_eager | 58.304 | 60.390 | -3.58% | packed_eager |
| 8 rows, 32-512 | wrapper | 58.467 | 60.192 | -2.95% | packed_eager |
| 32 rows, 32-512 | stock | 596.254 | 600.842 | -0.77% | stock |
| 32 rows, 32-512 | packed_graphs | 307.393 | 319.710 | -4.01% | packed_graph |
| 32 rows, 32-512 | packed_eager | 282.259 | 293.877 | -4.12% | packed_eager |
| 32 rows, 32-512 | wrapper | 282.761 | 294.588 | -4.18% | packed_eager |
| 8 x 1024 | stock | 268.440 | 272.694 | -1.58% | stock |
| 8 x 1024 | packed_graphs | 228.901 | 239.674 | -4.71% | packed_graph |
| 8 x 1024 | packed_eager | 232.003 | 239.848 | -3.38% | packed_eager |
| 8 x 1024 | wrapper | 231.756 | 240.054 | -3.58% | packed_eager |
| state 256, 8 questions | stock | 78.215 | 79.064 | -1.09% | stock |
| state 256, 8 questions | packed_graphs | 69.022 | 71.311 | -3.32% | packed_graph |
| state 256, 8 questions | packed_eager | 65.132 | 67.125 | -3.06% | packed_eager |
| state 256, 8 questions | shared_graphs | 17.381 | 16.807 | +3.30% | shared_graph |
| state 256, 8 questions | shared_eager | 28.828 | 17.635 | +38.83% | shared_eager |
| state 256, 8 questions | wrapper | 29.069 | 17.890 | +38.46% | shared_eager |
| state 1024, 8 questions | stock | 286.489 | 294.799 | -2.90% | stock |
| state 1024, 8 questions | packed_graphs | 245.689 | 255.707 | -4.08% | packed_graph |
| state 1024, 8 questions | packed_eager | 242.027 | 247.922 | -2.44% | packed_eager |
| state 1024, 8 questions | shared_graphs | 39.241 | 39.566 | -0.83% | shared_graph |
| state 1024, 8 questions | shared_eager | 40.020 | 39.855 | +0.41% | shared_eager |
| state 1024, 8 questions | wrapper | 40.440 | 40.212 | +0.56% | shared_eager |
| state 2048, 16 questions | stock | 1289.900 | 1314.668 | -1.92% | stock |
| state 2048, 16 questions | packed_graphs | 1017.064 | 1043.246 | -2.57% | packed_eager |
| state 2048, 16 questions | packed_eager | 1033.953 | 1037.871 | -0.38% | packed_eager |
| state 2048, 16 questions | shared_graphs | 81.538 | 81.480 | +0.07% | shared_eager |
| state 2048, 16 questions | shared_eager | 82.173 | 81.832 | +0.41% | shared_eager |
| state 2048, 16 questions | wrapper | 85.514 | 85.834 | -0.37% | shared_eager |

| Case | Eager baseline runs | Eager candidate runs |
|---|---|---|
| 1 x 128 | 19.849, 19.147 | 14.076, 14.384 |
| 8 rows, 32-512 | 58.021, 58.587 | 60.619, 60.160 |
| 32 rows, 32-512 | 280.682, 283.836 | 295.332, 292.421 |
| 8 x 1024 | 229.580, 234.426 | 240.342, 239.354 |
| state 256, 8 questions | 28.835, 28.821 | 17.645, 17.624 |
| state 1024, 8 questions | 39.303, 40.736 | 39.749, 39.961 |
| state 2048, 16 questions | 80.778, 83.567 | 81.377, 82.288 |

Distinct eager cases with at least 20% latency reduction: 2. 1 x 128, state 256, 8 questions

Cloud correctness checks:

```
...........................                                              [100%]
27 passed in 22.78s
.....................................                                    [100%]
37 passed in 14.04s
```

## Checks and artifacts

Focused local regression checks: 37 passed. The user approved committing and pushing this candidate to PR #11's `feat/lfm2` branch on 2026-10-14.

Local raw runs and the reproduction helpers live in the ignored benchmarks/results/pr11_overhead/ directory. The original broader candidate and its measurements remain archived separately. The tables above include every measured variant.

The inference norm uses a fixed launch instead of FLA's autotuner, so reduction order can differ. It preserves FLA's fp32 residual-sum normalization rather than changing that rounding convention. Independent kernel tests compare against FLA and check that inputs remain unchanged. Real-checkpoint parity for every Joe case remains recorded in the raw JSON; this is not a claim of bit-identical outputs or successful default pack-time certification.

## Decision for review

The two L40S eager gains clear the requested 20% latency-reduction threshold in distinct workloads: 1 x 128 and state 256 with 8 questions. The candidate also has measured L40S regressions of about 3–4% in the mixed-row and 8 x 1024 ordinary cases, including their graph variants. Their exact cause was not isolated; they should not be dismissed as noise. The 5090 gains remain modest. This is a targeted improvement with a measured tradeoff, not an across-the-board speedup. The user approved this candidate after reviewing these results.
