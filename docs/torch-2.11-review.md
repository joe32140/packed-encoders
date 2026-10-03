# Torch 2.11 review for PR #5

## Decision

Keep the existing Torch/Triton pins for this review. A straight 2.8 → 2.11 bump is
**not a drop-in upgrade of the current installation contract**: it loses the
configured prebuilt FlashAttention-2 wheel. Qwen's adaptation is tested in an
isolated 2.11 environment; that does not establish a replacement for every current
ModernBERT backend. No remote GPU benchmarks were run.

## What carries over

- Python 3.10–3.14 remains supported by Torch 2.11. The existing Python 3.14-specific
  Torch 2.9 branch could be removed if the package later standardizes on 2.11.
- CUDA 12.8 builds remain available from the existing PyTorch index. Keep that
  explicit index: PyPI's Torch 2.11 default is CUDA 13.0, which changes the runtime
  and minimum driver requirements independently of the engine refactor.
- Torch 2.11.0+cu128 resolves Triton 3.6.0. The package's current Triton 3.4/3.5
  constraints must change together with Torch; changing only Torch is unsatisfiable.
- On the RTX 5090, the tested CuteDSL 4.5.2 kernels, Triton kernels, SDPA paths,
  autograd and training-graph tests run under Torch 2.11. This is correctness
  evidence, not a performance comparison or a claim about untested GPUs.
- The real topk-embed-v1-xsmall checkpoint passes eager/graphed encode parity,
  validation and unpack restoration with Torch 2.11, Transformers 5.9 and FLA 0.5.1.

## What does not carry over automatically

The `fa2` source in `pyproject.toml` is a CPython 3.11, Torch 2.8, CUDA 12 wheel.
It cannot be retained as the supported binary for Torch 2.11. The official GitHub
release assets inspected for this review contain no Torch 2.11 wheel. Installing
`flash-attn==2.8.3.post1` in the isolated 2.11 environment attempts a source build;
it fails here because `nvcc`/`CUDA_HOME` are absent. This proves the prebuilt path
is missing, not that FA2 cannot be built against 2.11.

The existing suite without FA2 has 12 failures and 11 skips (115 passes before the
new registry tests). Eleven failures are explicit FlashAttention imports; the
remaining failure asserts that every ModernBERT piece was probed, including its
unavailable BSHD backend. Skipped Flash tests must not be counted as upgrade parity.
With the existing Torch 2.8/FA2 environment and the refactored source, the complete
suite passes (Qwen is skipped there because FLA is absent).

Before promoting 2.11 as the default:

1. Build/provide and test a compatible FA2 binary, or explicitly choose and validate
   a replacement backend. Do not silently replace Flash with SDPA and call it parity.
2. Update Torch and Triton together, remove/replace the hard-coded 2.8 FA2 source,
   regenerate `uv.lock`, and update the CPU CI matrix and installation documentation.
3. Recheck optional FA4 on its actual supported GPUs and repeat performance tests;
   those are deliberately outside this session. Existing benchmark claims describe
   their original environments, not a freshly validated 2.11 installation matrix.

The current `qwen3_5` extra only adds FLA; it does not override the base Torch pin.
It is therefore insufficient to reproduce the checkpoint environment by itself.
For this review we installed dependencies explicitly in a separate environment
and imported the working tree directly, without changing the project's lockfile
or the user's existing environment. A future packaging decision must address this
rather than advertise `pip install packed-encoders[qwen3_5]` as the complete topk setup.

## Reproduction environment

GPU: local RTX 5090 (sm_120), driver 615.71.09.

- Qwen: Python 3.12.12, torch 2.11.0+cu128, triton 3.6.0,
  transformers 5.9.0, flash-linear-attention/fla-core 0.5.1,
  nvidia-cutlass-dsl 4.5.2, sentence-transformers 6.1.0,
  torchvision 0.26.0+cu128.
- Existing backend check: Python 3.11.14, torch 2.8.0+cu128,
  transformers 5.3.0, the existing compatible compiled FA2 environment.
- CPU: actual Torch CPU wheels, not just GPU tests hidden with an environment flag.

The opt-in checkpoint test is run with `PE_TEST_TOPK=1`; a broken checkpoint oracle
now fails that explicitly requested test instead of being converted into a skip.
The multi-GPU test still requires two devices. Final results are recorded below.

## Sources

- [PyTorch 2.11 release announcement](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Official FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases)
- [Topk model and requirements](https://huggingface.co/topk-io/topk-embed-v1-xsmall/tree/main)

| Run | Result |
| --- | --- |
| Full suite, Torch 2.8.0+cu128 / Triton 3.4.0 / FA2 2.8.3.post1, RTX 5090 | 143 passed, 1 skipped (optional FLA/Qwen module absent) |
| Qwen suite, Torch 2.11.0+cu128, RTX 5090, `PE_TEST_TOPK=1` | 25 passed, 1 skipped (requires two GPUs) |
| Full suite, Python 3.10.19 / Torch 2.11.0+cpu | 64 passed, 80 skipped (CUDA/optional FLA) |
| Full suite, Python 3.14.7 / Torch 2.11.0+cpu | 64 passed, 80 skipped (CUDA/optional FLA) |

Commands (from the repository root, using the corresponding environment):

```bash
python -m pytest tests -q --disable-warnings
PE_TEST_TOPK=1 python -m pytest tests/test_qwen35.py -q --disable-warnings
```

Logs from this session: `/tmp/pe-pr5-modern-28.log`,
`/tmp/pe-pr5-qwen-final.log`, `/tmp/pe-pr5-cpu310.log`,
`/tmp/pe-pr5-cpu314.log`, and the incomplete-backend upgrade run
`/tmp/pe-pr5-modern-211.log`. The installation failure is preserved at
`/tmp/pe-pr5-fa2-install.log`.
