# RunPod runbook

The GPU meter runs only during execution. Everything else happens on a laptop.

## One-time setup

**1. Create a network volume** in the region you will rent H200s in. Size it for
roughly 250 GB: three virtualenvs, plus a 93 GB Mixtral download, plus a 115 GB
Qwen2-57B download if you capture from it. Storage bills monthly whether or not
a pod is running.

The volume is **region-locked**. That region's H200 availability becomes your
availability. If it is dry you either wait or lose the volume's benefit, so pick
a region with depth rather than the cheapest hourly rate.

**2. Launch a pod** from a RunPod PyTorch CUDA template, attach the volume at
`/workspace`, pick H200 NVL.

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

## Clock discipline

The harness samples SM clock and temperature before and after every cell and
flags rows where they drifted more than 5%. That records a symptom, not a
control. If your provider permits it, lock clocks and enable persistence mode
before a publication run, and note in the results what you locked them to. On
rented hardware, thermal state is the largest source of run-to-run disagreement.
