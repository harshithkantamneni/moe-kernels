# moe-kernels

MoE grouped-GEMM kernel work on an NVIDIA H200 NVL: a stage-span benchmark
harness, real captured routing traces, roofline analysis, and kernels.

## What this is for

The expert FFN in a Mixture-of-Experts layer is a **grouped GEMM**: every expert
gets a different number of tokens, which breaks standard batched GEMM. Current
implementations handle the ragged shape by padding each expert's group up to a
whole tile of a single global `BLOCK_M`. That is where the open problems are.

This repository exists to find a specific weakness by measurement, then attack
it. The measurement half is built. The kernels are being written.

**The gap this is aimed at.** Across vLLM, SGLang, PyTorch's persistent
cache-aware grouped GEMM, MegaBlocks, and OpenAI's `triton_kernels`, no
implementation publishes a **skewed-routing sweep**. Every published grouped-GEMM
number is measured on balanced or synthetic routing. Meanwhile the padding tax
is `E * (BLOCK_M - 1)` rows: at 256 experts with `BLOCK_M=64` that is 16,128
phantom rows, which at 2048 real rows is 8x the actual work. SGLang's
`tuning_fused_moe_triton_sep.py` is the only tuner in the field that measures
against captured routing, and even it emits one global `BLOCK_M` shared by every
expert. Nobody varies tile height per expert under skew.

So: measure on real routing, on the hardware, and see where it actually breaks.

## Architecture

A MoE layer is six ordered stages:

```
router -> permute -> up_gemm -> act -> down_gemm -> unpermute
```

An implementation declares the **contiguous span** of stages it covers. A plain
grouped GEMM is `covers=("up_gemm",)`. A fused up-projection plus SwiGLU is
`("up_gemm", "act")`. The fused down-projection plus scatter is
`("down_gemm", "unpermute")`.

A pipeline is a *tiling* of the six canonical stages by spans, validated for
coverage, dataflow and environment compatibility before anything runs. This
makes "fused versus unfused" two tilings of one pipeline rather than two
programs: same driver, same oracle, one line of config apart.

It also makes the cost model honest. FLOPs are tiling-invariant at
`6 * rows * F * H`; **bytes are not**. A span that fuses `up_gemm` with `act`
never materialises `h_up`, so neither the store nor the reload appears in the
model, and arithmetic intensity becomes a property of the tiling. Whether a
fusion should help is then a roofline prediction you can check against
measurement instead of a claim.

## Layout

```
moe/
  spec.py          MoEConfig, RoutingSpec, BenchSpec; verified model geometry
  state.py         MoEState, field contracts, shape validation
  stages.py        canonical stages, span contracts, implementation registry
  pipeline.py      tiling validation: coverage, dataflow, environment
  reference/       naive torch implementation of every stage; the fp32 oracle
  kernels/         YOUR kernels (see kernels/TEMPLATE.md)
  baselines/       vLLM, SGLang, torch
  routing/         parametric skew, imbalance metrics, trace capture and replay
  bench/           cost model, schema, timing, driver, roofline, profiles, CLI
  runner/          cross-virtualenv execution
scripts/           setup_runpod.sh, run_all.sh, capture_traces.py, plot.py
tests/             CPU tests; GPU tests behind a `gpu` marker
traces/            committed expert-count histograms (kilobytes, never weights)
```

The core (`spec`, `state`, `stages`, `pipeline`, `routing`, `bench.bytes_model`,
`bench.schema`) imports no torch and runs on a laptop. About 70% of the
repository is testable without a GPU, which is the point: the expensive box only
runs kernels.

## Methodology

Things this harness records rather than assumes, because most published MoE
numbers omit them and are therefore not comparable to each other:

- **L2 residency.** H200 has 50 MiB of L2. Whether expert weights are resident
  changes small-batch results by more than most kernel optimisations do, so the
  flush state is a swept axis and a recorded column. The flush reads rather than
  writes, so it leaves no dirty lines to be written back inside the next timed
  interval.
- **Launch mode.** Eager and CUDA-graph replay are measured separately, and a
  graph row re-earns its correctness verdict against the *replayed* output,
  because graph replay reuses fixed buffers and would otherwise let a kernel
  that skips a tail tile show the previous replay's correct values.
- **Non-capturability is a result.** An implementation that syncs with the host
  cannot be used in real MoE inference. It gets a CSV row saying so, not a
  silent omission that would condition every aggregate on capture-friendliness.
- **Correctness gates timing.** No timing number is written for an
  implementation that did not pass the golden-fp32 oracle on that exact cell in
  that same run. The metric is scale-free (`max|got-ref| / max|ref|`), because
  an absolute tolerance is either vacuous or impossible depending on geometry.
- **Compulsory traffic is labelled as such.** `arith_intensity_compulsory` is an
  upper bound on true intensity and `compulsory_gbps` is not achieved HBM
  bandwidth. Named so nobody reads them as measurements.
- **Clock and thermal drift.** Sampled around every cell and flagged, so a hot
  rented box cannot masquerade as a kernel regression.

## Routing traces

The differentiator. Stored as per-layer expert-count histograms, not token logs:
only the multiset of group sizes affects the grouped GEMM, so a histogram is
sufficient, it is kilobytes, and it can live in git. Replay reconstructs a
concrete top-k assignment whose histogram matches the capture **exactly**.

Capture runs at both prefill and **decode**. Decode is single-token steps, which
is the memory-bound many-expert regime this project targets, and decode-time
routing traces are what almost nobody publishes.

**DeepSeek-V3 routing is not captured and is not claimed.** At 1369 GB in bf16
it does not fit on one H200, or five. Its geometry is benchmarked with
parametric routing, labelled synthetic wherever it appears.

## Hardware

`moe/bench/hardware/h200_nvl.yaml` carries the roofline peaks with a citation.
Two traps it exists to avoid: every tensor-core figure NVIDIA publishes for H200
is the **sparsity** number and dense is exactly half; and H200 NVL and H200 SXM
share memory (141 GB, 4.8 TB/s) but not compute (835.5 versus 989.5 dense BF16).
`roofline.py` refuses to draw an uncited roof.

## Running it

Free, on a laptop:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
bash scripts/run_all.sh --dry-run --profile standard
```

On the GPU box, see [docs/RUNPOD.md](docs/RUNPOD.md).

## Writing a kernel

See [moe/kernels/TEMPLATE.md](moe/kernels/TEMPLATE.md). The short version: pick
your span, declare it, produce every field in `self.writes`, and do not use
`.item()`, a `.tolist()`, or a host-side loop over experts, because all three
break CUDA-graph capture and CUDA graphs are how MoE inference actually runs.

## Status

Harness complete and tested on CPU. Traces not yet captured. Kernels not yet
written. No performance numbers have been produced, and none are claimed.
