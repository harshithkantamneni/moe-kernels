# RunPod runbook

The GPU meter runs only during execution. Everything else happens on a laptop.

## One-time setup

**1. Create a NETWORK VOLUME** in the region you will rent H200s in.

Network Volume, not the pod's Volume Disk. A Volume Disk survives stop/start but
dies with the pod; a Network Volume survives termination and can be attached to
a different pod later. Since the whole workflow is spin up, run, terminate, the
distinction is the difference between paying for the environment once and paying
for it every session.

**Sizing.** Storage bills monthly whether or not a pod runs, and RunPod volumes
can be grown but not shrunk, so start at the smallest size that unblocks you:

| you need | size | what it holds |
|---|---|---|
| benchmarks only | **100 GB** | three venvs (~50 GB), uv/Triton caches (~15 GB), results |
| + capture DeepSeek-V2-Lite | 150 GB | adds a 31 GB model |
| + capture Mixtral | 250 GB | adds a 93 GB model |
| + capture Qwen2-57B | 350 GB | adds a 115 GB model |

**Start at 100 GB.** The sweeps generate random weights and download nothing at
all: only `capture_traces.py` pulls a model. So tests, calibration, smoke and the
standard sweep all fit in the first row, and you grow the volume on the session
you actually capture traces.

**2. Container disk: 50 GB.** This is ephemeral scratch, billed only while the
pod runs. The caches are redirected onto the volume, but wheel extraction for
vLLM and SGLang still needs several GB of temp space, and running out mid-install
is a slow failure on a metered box. `setup_runpod.sh` now checks free space
before building each environment and aborts early rather than dying halfway.

The volume is **region-locked**. That region's H200 availability becomes your
availability. If it is dry you either wait or lose the volume's benefit, so pick
a region with depth rather than the cheapest hourly rate.

**3. Launch a pod** from the official RunPod PyTorch template (newest offered;
2.8.0 + CUDA 12.8 at time of writing), attach the network volume at
`/workspace`, and pick your H200.

Note which H200: SXM and NVL share memory (141 GB, 4.8 TB/s) but not compute
(989.5 vs 835.5 dense BF16). The repo carries a profile for each and selects by
device name; `plot.py` refuses to plot rows from one against the other's roof.

**The template's torch is not the published torch, and for profiling that
matters.** `runpod-torch-v280` ships torch 2.8; the published rows were measured
on 2.13.0+cu130 with Triton 3.7.1. The baseline still resolves on 2.8, via the
private `torch._grouped_mm` name, so a sweep runs. But a different torch ships a
different CUTLASS, so the grouped GEMM you would profile is not the one the
published rows describe, and `scripts/profile_open_questions.sh` refuses to run
against a mismatch rather than answering a different question confidently. Pin
the base venv:

```bash
MOE_BASE_TORCH='torch==2.13.0' \
MOE_TORCH_INDEX='https://download.pytorch.org/whl/cu130' \
  bash scripts/setup_runpod.sh --base-python 3.12 base
```

`--base-python` implies an isolated base venv (no `--system-site-packages`),
and `bash scripts/setup_runpod.sh --dry-run --base-python 3.12 base` prints
exactly what it would build, for free. Until 2026-09-02 this section told the
operator to EDIT `scripts/setup_runpod.sh` (`setup_env base
--system-site-packages` into `setup_env base --python 3.12`). That line no
longer exists, and the edit was never a pin: it modified a tracked file, so
`pod_session.sh` P1 failed and `driver.py` stamped `git_dirty=True` on every
row of the session, which is why the alpha-0558, three-way and v2lite arms are
100% dirty. The pin is a flag now and the session records which was used.
Check `nvidia-smi` first, since a cu130 wheel needs driver r580+; if the image
is older, use the `cu128` index instead. The index must match the driver, not
the image name.

**Recalibrate on each new pod, and PUBLISH the result.** `measured_nvidia_h200.yaml`
matches by device NAME, so a second H200 pod silently reuses the first one's
ceilings. Same part, so they should be close, but "measure this machine" is
the whole methodology (the H200's dense bf16 moved 7.1% between two sessions
while its bandwidth held to 0.014%), and the run takes about three minutes:

```bash
python scripts/calibrate_hardware.py --publish   # -> results/calibration/<run id>/measured_nvidia_h200.yaml
                                                 #    AND moe/bench/hardware/measured_nvidia_h200.yaml
git diff moe/bench/hardware/                     # did the ceilings actually move?
```

`--publish` is the whole recipe. Without it the calibration lands only on an
untracked path under `results/calibration/`, `git diff moe/bench/hardware/` is
empty, the operator reads "the ceilings did not move", and
`roofline.load_measured()` keeps returning the PREVIOUS pod's ruler for every
row measured afterwards, labelled "measured on this machine". This section
recommended exactly that no-op until 2026-09-03. The session driver,
`scripts/h200_gaps_session.sh`, runs the published form as arm 0 and refuses
the rest of the session unless the tracked yaml carries a `provenance.utc`
stamp from this session AND the ledger says the arm was DONE
(`docs/POD_RUNBOOK.md`). Commit the yaml with the session's results; it is the
one tracked file the session is meant to change.

**3. Bootstrap.** Clone into the volume so the repo survives the pod:

```bash
cd /workspace && git clone https://github.com/<you>/moe-kernels repo && cd repo
bash scripts/setup_runpod.sh
```

First run installs and writes `requirements/resolved-*.txt`. **Commit those.**
A later session installs FROM the resolved set when one is present and still
true, and REFUSES when it has drifted from the top-level file, naming what
drifted; `bash scripts/setup_runpod.sh --check` asks the same question from a
laptop and spends nothing, and `--fresh` re-resolves on purpose. Until
2026-09-02 this paragraph promised the resolved set was installed and no code
path read it: a fresh volume re-resolved `base.txt`, whose top-level pins are
unversioned, so a stranger's environment was whatever the index held that day.
As this is written `resolved-base.txt` IS stale (it predates the `nvidia-ml-py`
line and the `transformers<4.54` cap, and its `torch==2.13.0` carries no
`+cu130` local tag, so the CUDA index is not encoded in it); the next pod fixes
that with `--fresh`, and only a pod can, because the CUDA wheels are the thing
being resolved.

Later sessions detect an unchanged requirements file by content hash and finish
in about a second.

## Every session after that

Two drivers exist and they answer different questions. The sweep:

```bash
cd /workspace/repo && bash scripts/run_all.sh --profile standard --max-minutes 45
```

and the open experiments, every arm of the next session in the order their
results are read, with its own ledger and resume:

```bash
cd /workspace/repo && bash scripts/h200_gaps_session.sh --dry-run   # first, on the laptop
cd /workspace/repo && bash scripts/h200_gaps_session.sh             # on the pod
```

`docs/POD_RUNBOOK.md` is the operator page for the second. The first does:
pull, idempotent setup, **test suite**, smoke, sweep, plots, summary.
The test suite runs before the sweep on purpose. A failure there costs seconds;
discovering the same failure after an hour of benchmarking costs an hour.

Stop the pod when the summary prints. The volume keeps the environments, the
caches, the traces, and the results.

## Before you spend anything

`--dry-run` runs on your laptop. It builds and validates every tiling in the
matrix and reports its size without touching a GPU:

```bash
bash scripts/run_all.sh --dry-run --profile standard
```

Invalid tilings, unsupported dtypes, mixed-environment pipelines and missing
traces all surface here, for free.

## Resuming

Every run writes a JSONL manifest beside its CSV and flushes each row to disk.
Kill the pod mid-sweep and you lose at most one cell.

```bash
bash scripts/run_all.sh --profile standard --run-id <id from the previous run>
```

Completed work is skipped before the expensive fp32 oracle runs. Deterministic
outcomes (a correctness failure, a non-capturable implementation) are terminal
and are not retried. Transient outcomes (a CUDA OOM, a crash) stay retryable, so
one bad moment does not permanently blank a cell from every future run.

## Environment layout on the volume

```
/workspace/
  repo/                  this repository
  venvs/base/            your kernels + the harness (inherits the image's torch)
  venvs/vllm/            vLLM 0.27.1 and its own torch
  venvs/sglang/          SGLang 0.5.18 and its own torch
  hf-cache/              HF_HOME
  triton-cache/          TRITON_CACHE_DIR
  torchinductor-cache/
  results/               run CSVs, manifests, merged.csv
  traces/raw/            tier-2 traces, not committed
```

`TRITON_CACHE_DIR` on the volume matters more than it looks. Without it, every
session recompiles every autotuned kernel variant from scratch, which is minutes
of metered time per spin-up and grows as your autotune space does.

## Why three virtualenvs

vLLM 0.27.1 and SGLang 0.5.18 agree on torch (2.13.0) and Triton (3.7.1). They
disagree on four exact pins: `flashinfer-python`, `nvidia-cutlass-dsl`,
`quack-kernels`, and `outlines_core` (0.2.14 vs 0.1.26, a major-version split of
a compiled Rust extension). `pip install vllm sglang` into one environment fails
resolution, and `--no-deps` only defers the problem to two compiled kernel
packages built against different flashinfer and cutlass-DSL.

The harness already knows about this: every span declares `env`, and
`pipeline._check_env` refuses a tiling that mixes two frameworks. The runner
shells into each venv and `schema.merge_csvs` combines the outputs.

## Capturing traces

One session, then never again:

```bash
python scripts/capture_traces.py --model mixtral-8x7b --phase decode --corpus chat
python scripts/capture_traces.py --model mixtral-8x7b --phase prefill --corpus code
python scripts/capture_traces.py --model deepseek-v2-lite --phase decode --corpus chat
git add traces/*.npz && git commit
```

Traces are kilobytes. Model weights are never written to the repo.

Mixtral is NOT gated any more. `mistralai/Mixtral-8x7B-Instruct-v0.1` is
apache-2.0, `model_info` reports `gated=False`, and `config.json` downloads with
no credentials at all -- verified on an H200 pod on 2026-09-01. This document
told you to accept a licence that no longer exists, which cost a session's worth
of confusion; check `gated` rather than trusting either this line or the model
card. A token is still worth setting, but for RATE and not for access: HF
rate-limits anonymous transfers and step 0 pulls 93.4 GB.

### What fits on one H200 (141 GB, bf16)

Both H200 parts have 141 GB and 4.8 TB/s, so this table holds for SXM and NVL
alike. They differ only in compute (989.5 vs 835.5 dense BF16), which is why
there are two hardware profiles and why `plot.py` refuses to plot rows from one
against the other's roof. An H100 NVL is 94 GB and would change every verdict
below.

| model | full model | capturable here |
|---|---|---|
| mixtral-8x7b | 93.4 GB | yes, comfortably |
| qwen2-57b-a14b | 114.8 GB | yes, ~26 GB left for KV and activations |
| deepseek-v2-lite | 31.4 GB | yes, the cheap 64-expert proxy |
| deepseek-v3 | 1369 GB | **no**, needs 5+ H200s |

DeepSeek-V3's routing cannot be captured on this hardware. Benchmark its
**geometry** with parametric routing and say so explicitly wherever the results
appear. Claiming a captured V3 trace would be false and is the kind of thing a
reviewer checks first.

## Pod lifecycle stays manual

These scripts never create or destroy pods. An automation bug that spins up an
H200 and fails to stop it is the most expensive failure available here, and it
is not worth the convenience. Start and stop from the RunPod console or
`runpodctl` yourself.

## Getting results off the pod

Code flows one way (your machine -> GitHub -> pod) and results flow the other,
but nothing does it automatically. After a run worth keeping:

```bash
bash scripts/publish_results.sh --label first-smoke
```

That copies the run's CSVs and manifests into
`results/published/<date>-<device>-<label>/`, includes the per-device calibration
the efficiency columns were quoted against, writes a readable `SUMMARY.md`,
commits, and pushes. The device comes from the rows' own `gpu_name` column, not
from the machine you happen to be publishing from, so each GPU keeps a separate
published arm off the one shared harness.

`results/` is gitignored on purpose: raw runs are large, machine-specific, and
regenerable. `results/published/` is tracked on purpose. Publishing is a
decision, not a side effect.

**Pushing needs credentials on the pod.** A public clone over HTTPS can pull but
not push. Once per pod:

```bash
gh auth login          # or set up a fine-grained PAT with contents:write
```

If push fails the commit is still made locally, so nothing is lost; you can push
later or copy the directory off with `runpodctl send`.

## Profiling is not available, and what replaces it

`ncu` fails on a rented pod with `ERR_NVGPUCTRPERM`. GPU performance counters
are gated behind a host kernel-module flag
(`NVreg_RestrictProfilingToAdminUsers=0`) that a container tenant cannot set,
and RunPod containers are not privileged. Assume no counters and design around
it rather than planning a session that discovers this at the console.

That means **actual DRAM traffic cannot be measured**, so the compulsory-bytes
model cannot be validated directly. Three substitutes, none of which need
counters:

**1. Measure the ceilings instead of quoting them.**

```bash
python scripts/calibrate_hardware.py --publish
```

STREAM-style copy, triad and write on buffers far larger than L2, plus a large
square BF16 GEMM through cuBLAS. Writes an untracked copy under
`results/calibration/<run id>/` and, with `--publish`, the tracked
`moe/bench/hardware/measured_<device>.yaml` that `roofline.load_measured()`
actually reads; `run_all.sh` creates the latter automatically on a pod that
lacks it. One file per device, so calibrating a
second GPU does not overwrite the first and an earlier sweep can still be
re-plotted against its own roof. A sweep refuses to start against a calibration
measured on a different part. Efficiency is then quoted against what this
machine actually delivers rather than a datasheet peak it will never reach,
which is both fairer to your kernel and far easier to defend in public. Expect roughly 75-90% of spec bandwidth and 70-85% of spec
dense BF16; if you measure above spec, your buffer fit in cache and the number
is not a DRAM measurement.

**2. Bound the re-read factor arithmetically.** For a cell that is genuinely
memory bound, `time x achievable_bandwidth` bounds the bytes that could have
moved, and dividing by the compulsory minimum gives `implied_traffic_ratio`. A
value near 1 is strong evidence the kernel moves close to the minimum traffic.
It is an **upper** bound, not a measurement: it also absorbs low occupancy and
latency stalls, so a large ratio says "something costs you", not specifically
"you re-read". The column is only emitted when compulsory intensity is below the
ridge, which is sound because compulsory intensity is itself an upper bound on
true intensity.

**3. Read cache behaviour off the flush axis.** Every cell is already timed with
L2 flushed and with L2 warm. The difference, times achievable bandwidth,
estimates the traffic the cache absorbed. `scripts/plot.py` draws this as
`l2_absorption_<dtype>.png`. It is the counter-free stand-in for a hit-rate
metric and it costs nothing extra, because the axis is swept anyway.

**Worth one test at the start of your first session**: `nsys` uses CUDA tracing
rather than performance counters and often works where `ncu` does not. It will
not give you DRAM bytes, but it does give per-kernel timing attribution and
launch overhead, which is exactly the evidence the eager-versus-graph question
needs. If it runs, use it there.

If direct traffic measurement ever becomes essential, the weaker ask comes
first: NVIDIA's own ERR_NVGPUCTRPERM page names `--cap-add=SYS_ADMIN` on the
container as the container-side remedy, one Linux capability per instance
rather than a host module flag and a reboot, and nothing in this project has
asked a provider for it yet (`docs/COUNTERS.md` section 2). Failing that it
needs bare metal or a VM whose guest kernel you own, not a different RunPod
template. And before the `nsys` test above: the RunPod image ships a
TARGET-ONLY Nsight Systems, without the `QdstrmImporter` that turns a capture
into a report, so every 2026-09-01 attempt failed at conversion including the
control that requested no metrics at all. Check for
`host-linux-x64/QdstrmImporter` first, or install a current build
(`docs/COUNTERS.md` section 1).

## Clock discipline

`moe/bench/timing.py` polls the SM clock from a background thread WHILE every
cell's trials run and records two verdicts per row: LEVEL, the loaded clock is
inside the band 0.95 to 1.05 of the clock this card's roof was measured at
(`clock_level_ok`, with `clock_level_side` naming `low` or `high` on a
failure), and DRIFT, the first and last under-load samples agree within 5% in
either direction (`clock_drift_ok`). SINCE 2026-09-09, DRIFT alone excludes a
row and the LEVEL side is recorded and excludes nothing. Under a 700 W cap the
clock under load is set per tile by the kernel's own power draw, so `high` is
the expected state of a memory-bound cell on the H200 (the calibration's memory
load holds 1980 MHz) and `low` is the expected state of a hungry tile
(BLOCK_M=128 holds a median 1395 MHz over 215 cells against the 1485 MHz GEMM
reference); `docs/APPARATUS.md` section 1 has the per-tile table and the
session that produced it. Neither side is an exclusion: what is wrong on both
is the fixed-roof fraction, and `pct_of_roof_at_cell_clock` is the column to
read beside it. Until 2026-09-09 this section said "Only LOW or DRIFT excludes
a row", which excluded the study's two primary tiles from measurability on this
card. LEVEL needs the reference clock from this card's
published calibration, which is the second reason `--publish` above is not
optional. Until 2026-09-02 this section said the harness sampled the clock
"before and after every cell" and flagged a drift over 5% (retracted: that
compared two idle-instant samples and fired on a drop, so it detected whether
the first sample had caught the idle boost rather than throttling; on the
alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged and
unflagged replicates timed at ratio 0.998). Either flag records a symptom, not
a control. If your provider permits it, lock clocks and enable persistence mode
before a publication run, and note in the results what you locked them to. On
rented hardware, thermal state is the largest source of run-to-run disagreement.
