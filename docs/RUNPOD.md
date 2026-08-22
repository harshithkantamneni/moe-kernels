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

**3. Bootstrap.** Clone into the volume so the repo survives the pod:

```bash
cd /workspace && git clone https://github.com/<you>/moe-kernels repo && cd repo
bash scripts/setup_runpod.sh
```

First run installs and writes `requirements/resolved-*.txt`. **Commit those.**
Every later session then installs the exact resolved set rather than
re-resolving, and anyone reproducing your numbers gets the same environment.

Later sessions detect an unchanged requirements file by content hash and finish
in about a second.

## Every session after that

```bash
cd /workspace/repo && bash scripts/run_all.sh --profile standard --max-minutes 45
```

That does: pull, idempotent setup, **test suite**, smoke, sweep, plots, summary.
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

Mixtral is gated on Hugging Face. Accept the license and `huggingface-cli login`
before the session, or the download fails after you have already started paying.

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
python scripts/calibrate_hardware.py
```

STREAM-style copy, triad and write on buffers far larger than L2, plus a large
square BF16 GEMM through cuBLAS. Writes `moe/bench/hardware/measured.yaml`,
which `run_all.sh` creates automatically on a pod that lacks it. Efficiency is
then quoted against what this machine actually delivers rather than a datasheet
peak it will never reach, which is both fairer to your kernel and far easier to
defend in public. Expect roughly 75-90% of spec bandwidth and 70-85% of spec
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

If direct traffic measurement ever becomes essential, it needs bare metal or a
provider that grants privileged containers, not a different RunPod template.

## Clock discipline

The harness samples SM clock and temperature before and after every cell and
flags rows where they drifted more than 5%. That records a symptom, not a
control. If your provider permits it, lock clocks and enable persistence mode
before a publication run, and note in the results what you locked them to. On
rented hardware, thermal state is the largest source of run-to-run disagreement.
