# moe-kernels

MoE grouped-GEMM kernel work on an NVIDIA H200 NVL: a stage-span benchmark
harness, real captured routing traces, roofline analysis, and kernels.

## What this is for

The expert FFN in a Mixture-of-Experts layer is a **grouped GEMM**: every expert
gets a different number of tokens, which breaks standard batched GEMM. Current
implementations handle the ragged shape by padding each expert's group up to a
whole tile of a single global `BLOCK_M`. That is where the open problems are.

This repository exists to find a specific weakness by measurement, then attack
it. The measurement half is built and has run: 100,144 rows on an H200 and an
A100, plus 26 ladder reports from three further arms. It killed the original premise, so this is a measurement study and
`moe/kernels/` stays empty unless a result argues for filling it. What it found
is in [docs/FINDINGS.md](docs/FINDINGS.md); the working state, including what
was retracted and why, is in [docs/STUDY.md](docs/STUDY.md).

**What this is aimed at.** Every MoE grouped-GEMM implementation I looked at
selects its tile height from the *mean* tokens per expert: Inductor gates on
`m_avg = m // g`, `triton_kernels` derives from `expected_slice_size`,
MegaBlocks fixes `BLOCK_M = 128`, and vLLM and SGLang ship configs keyed by
batch size with a single global `BLOCK_M`. Under balanced routing the mean is
the right statistic, since every expert is the mean.

Under skewed routing it stops being. Arithmetic intensity works out to
rows-per-expert, so on this H200 an expert crosses the roofline ridge at about
163 rows (162.8 FLOP/byte on the card's 2026-09-02 calibration), and a skewed
launch contains experts on both sides of it at once: at
`zipf:1.2` and 4096 tokens, 35 experts are compute-bound and hold 73% of the
rows while 221 are memory-bound. That is one draw, and a typical one: over 40
resamples of the same distribution the split runs 31-36 compute-bound experts
holding 70.5-73.5% of the rows. Under uniform routing the mix never occurs at
any batch size measured.

The ridge is quoted per calibration because its compute term does not
reproduce. Calibrations of this same H200 gave 4374.5 to 4377.2 GB/s of
bandwidth, 0.06% apart, but 701.6 to 770.9 TFLOP/s of dense bf16, 9.9% apart,
and not because of clock: the run at the highest clock reached the lowest
fraction of its own peak. An earlier draft of this paragraph said ~166 rows,
and a later one quoted a band of 160 to 176 rows as if it were the card's
uncertainty about its own ridge (retracted 2026-09-02: that band is two
sessions' compute ceilings failing to reproduce, no card's own ridge, and every
published ladder report scored against it has been rescored to the attached
card's own calibration, H200 162.8 and A100 145.8 FLOP/byte). The number to
quote is the ridge of the calibration a row was measured against, and
`results/published/CALIBRATION_PROVENANCE.md` says which that is for every arm.

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
  bench/           cost model, schema, driver, roofline, profiles, CLI, and the
                   four apparatus modules: timing (the one instrument),
                   exit_codes (the one table), provenance (the one block),
                   ai_model (the byte model and what a fit returns). Read
                   docs/APPARATUS.md before any of them.
  runner/          cross-virtualenv execution
scripts/           setup_runpod.sh, run_all.sh (the sweep), h200_gaps_session.sh
                   (the session driver for every open experiment),
                   capture_traces.py, plot.py, and one script per arm
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

- **L2 residency.** H200 has 60 MiB of L2 (the harness records `l2_bytes = 62914560`; an earlier draft of this line said 50). Whether expert weights are resident
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
- **Clock under load, flagged on LEVEL and DRIFT.** `moe/bench/timing.py`
  polls the SM clock from a background thread WHILE the trials run and sets
  two verdicts per cell: LEVEL, the loaded clock is within 5% of the clock the
  roof was measured at, and DRIFT, the first and last under-load samples agree
  within 5% in either direction. An earlier version of this line said the
  clock was "sampled around every cell and flagged" (retracted 2026-09-02:
  that flag compared two idle-instant samples and fired on a drop, so it
  detected whether the first sample had caught the idle boost, not
  throttling; on the alpha-0558 arm it flagged 91% of vLLM rows above T=4096
  while flagged and unflagged replicates timed at ratio 0.998). Every
  `merged.csv` under `results/published/` predates the LEVEL flag.

## Routing traces

Stored as per-layer expert-count histograms, not token logs:
only the multiset of group sizes affects the grouped GEMM, so a histogram is
sufficient, it is kilobytes, and it can live in git. Replay reconstructs a
concrete top-k assignment whose histogram matches the capture **exactly**.

Capture runs at both prefill and **decode**. Decode is single-token steps, which
is the memory-bound many-expert regime this project targets. A captured decode
histogram is what the parametric distributions in the sweep stand in for.

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

Free, on a laptop, in this order:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest tests/ -q
bash scripts/run_all.sh --dry-run --profile standard
```

The third line is not optional. The `dev` extra deliberately ships no torch
(`requirements/base.txt` says why: on the pod the base venv inherits the
CUDA-matched torch from the image, and guessing a CUDA wheel tag from a
laptop is how a build stops matching the driver), but `tests/conftest.py` and
`moe/bench/timing.py` both import torch at module level. Without the CPU
wheel, `pytest` fails while loading `conftest.py` with zero tests collected and
`run_all.sh --dry-run` exits 1 from `timing.py`. Any current CPU wheel will
do for the laptop, which runs no kernel; the pod pins 2.13.0 and
`requirements/resolved-base.txt` records it. `tests/test_docs.py` is part of
the suite and fails if any number on this page or under `docs/` has drifted
from the tree.

On a laptop with only the base venv, the last command prints
`[run_all] REFUSE: profile 'standard' spans sglang,vllm, and this box will run
only base` and exits 0. That is the plan refusing to pretend it can sweep
frameworks this box does not have; it used to sweep torch alone and print a
summary, which is how a 45-minute pod session once produced no vLLM rows.

On the GPU box, see [docs/RUNPOD.md](docs/RUNPOD.md) for the environment and
[docs/POD_RUNBOOK.md](docs/POD_RUNBOOK.md) for the session driver,
`scripts/h200_gaps_session.sh`. What the instrument is, and what every gate's
exit code means, is one page: [docs/APPARATUS.md](docs/APPARATUS.md).

## Writing a kernel

See [moe/kernels/TEMPLATE.md](moe/kernels/TEMPLATE.md). The short version: pick
your span, declare it, produce every field in `self.writes`, and do not use
`.item()`, a `.tolist()`, or a host-side loop over experts, because all three
break CUDA-graph capture and CUDA graphs are how MoE inference actually runs.

## Status

Harness complete; 3819 tests collected off-GPU (`pytest --collect-only -q`;
`tests/test_docs.py` fails when this line goes stale). 14 published arms in
`results/published/`: 11 carry a `merged.csv`, 100,144 rows in all, 72,760 of
them current (the rest superseded and kept for provenance), and 3 are ladder
arms carrying 26 `*.report.json` files and no CSV. Five claims: C1 and C2
established, C3 established and rescoped three times, C4 confirmed and closed,
C5 NOT established (three uncontrolled variables). The tile-corrected roofline
is proposed and not validated, and its BLOCK_M = 128 row is retracted in the
form it was published (see the RETRACTIONS section of
[docs/FINDINGS.md](docs/FINDINGS.md)). Kernels not written, and the padding-tax
premise that motivated them is dead. Read FINDINGS before quoting any number
from this repository: several published figures have been retracted, and that
file says which, when, and where the corrected number lives.
