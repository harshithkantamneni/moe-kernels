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
156 rows (155.9 FLOP/byte on the card's 2026-09-10 calibration), and a skewed
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
card's own calibration, H200 162.8 and A100 145.8 FLOP/byte). The H200's own
ridge has since moved again, to 155.9 on the 2026-09-10 calibration, because
that session sampled the dense GEMM's clock while it ran rather than after it.
Three calibrations of one card now read 162.8, 152.8 and 155.9, which is the
same non-reproducing compute term the paragraph above describes and is why the
band was withdrawn rather than widened. The number to quote is the ridge of the
calibration a row was measured against, and
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
  two verdicts per cell: LEVEL, the loaded clock is inside a 0.95 to 1.05 band
  around the clock the roof was measured at, with the side named on a failure,
  and DRIFT, the first and last under-load samples agree within 5% in either
  direction. **DRIFT is the only exclusion.** A clock that moved across a
  cell's own trials makes its median a blend of two operating points and the
  time belongs to neither, which no rescaling repairs. NEITHER LEVEL SIDE
  EXCLUDES ANYTHING: under a 700 W cap the clock is set per tile by the
  kernel's own draw, so a LOW cell is a hungry tile at its steady state and a
  HIGH one a memory-shaped cell boosting. Both are kept, the side is recorded
  on the row, every compute-bound gate reads the fixed roof, and
  `pct_of_roof_at_cell_clock`, the roof rescaled to the cell's own clock, is
  printed beside it as issue efficiency and is never a gate input. Until
  2026-09-09 a LOW cell was excluded, which made the study's two primary tiles
  unmeasurable on the H200: BLOCK_M=128 at BN=64 holds 1395 MHz against a
  calibration GEMM at 1485. An earlier version
  of this line said the clock was "sampled around every cell and flagged"
  (retracted 2026-09-02:
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

Harness complete; TESTCOUNT tests collected off-GPU (`pytest --collect-only -q`;
`tests/test_docs.py` fails when this line goes stale). 14 published arms in
`results/published/`: 11 carry a `merged.csv`, 100,144 rows in all, 72,760 of
them current (the rest superseded and kept for provenance), and 3 are ladder
arms carrying 26 `*.report.json` files and no CSV. Two further directories
there are whole H200 sessions, kept so every verdict they printed can be
re-derived and marked with a `KIND` file so the provenance census reads them as
sessions rather than arms.

**What the 2026-09-10 session measured.** Twenty arms, 150 minutes, 2,328
clocked cells. Three results need no fitted model: the cost of one extra M-tile
is 0.68 to 1.37 complete streams of the layer's expert weights across 23
ladders; at `BLOCK_M=16` and `GROUP_SIZE_M=1`, 95.4% of the kernel's wall clock
is accounted for by one full weight re-read per tile (89.158 ms measured
against 85.054 ms of streaming); and `BLOCK_M=16` peaks at 0.099 of the card's
dense compute roof at every batch size reachable, against 0.537 for a
`BLOCK_M=256` control on the same layer.

**Where the model failed, and it is not where the residual test said.** The
three-term traffic model has exactly one term that depends on `BLOCK_SIZE_N`,
and that term is proportional to `BLOCK_M`, so the measured width dependence
must double when the height doubles. It does not move: the ratios are 1.115,
1.203 and 0.923 where the model requires 2.000, at z of -303, -121 and -89,
measured on the slope alone with no fitted intercept anywhere in the path. The
mechanism the model was written to express survives; that particular
parameterisation of it does not. The replacement the data supports is a cost
going as `1/BLOCK_SIZE_N` and *not* with `BLOCK_M`, which fits 3.1x better at
equal parameter count, but whether it is traffic or time this grid cannot say.

**The re-read is a property of the schedule.** `GROUP_SIZE_M` from 1 to 16
moves the per-tile cost by 24% while compiling byte-identical shared memory, an
identical PTX instruction census and identical occupancy. The model has no slot
for a schedule.

**Two results that stand on their own.** Of 56 power-of-two tiles, the 17 whose
arithmetic-intensity ceiling clears this card's ridge all need an accumulator of
at least the entire 65,536-register per-block file, on every NVIDIA
architecture from sm_70 to sm_100: the register file runs out exactly where the
arithmetic would have become sufficient, so no positive control exists at
vLLM's shipped `BLOCK_SIZE_N`. And under a 700 W cap held to within 2% across
2,328 cells, the achieved SM clock spans 1,275 to 1,935 MHz as a function of
the tile alone, so any tile comparison scored against a roof measured at one
clock is comparing two machines.

**Claims.** C1 and C2 established, C3 established and rescoped three times, C4
confirmed and closed, C5 not established and now shown unresolvable: the
cross-card difference is +0.0117 against a detection limit of 0.0908, and
changing one compiler scheduling flag on a single card moves the same quantity
by +0.0101.

**Do not quote these.** `alpha_b = 0.9794 +/- 0.0113` is not a measurement: it
comes from a fit whose partner parameter is -0.8143, a value `moe/bench/ai_model.py`
refuses as impossible, its honest interval is nearer +/-0.048, and a quarter of
the bootstrap draws fall outside `[0, 1]` entirely. The 207% contradiction of
TEMPO compares a bound with a number. The `cap/ridge = 0.080` headline is
computed from an alpha the same module refuses to invert. For `BLOCK_M <= 64`
the ceiling binds under every value this study has ever held; at
`BLOCK_M = 128`, the tile vLLM actually ships, the verdict flips across the
candidate range (0.81 of the ridge at 0.9794 against 1.31 at 0.5977) and is
therefore not established either way.

Kernels not written, and the padding-tax premise that motivated them is dead.
Read FINDINGS before quoting any number from this repository: several published
figures have been retracted, and that file says which, when, and where the
corrected number lives.
