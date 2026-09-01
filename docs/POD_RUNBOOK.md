# Pod runbook

The human-readable companion to `scripts/pod_session.sh`. That script is the
session; this file says what each gate MEANS, so a tired person at 02:00 can act
on a FAIL without rereading `docs/FINDINGS.md`.

`docs/RUNPOD.md` is how to get a pod and an environment. This is what to do once
you have one.

**The whole session is one command.**

```bash
cd /workspace/repo
bash scripts/pod_session.sh --label alpha-0558
```

It prints PASS or FAIL against a NUMBER at every step, says on each FAIL whether
to continue or stop, and refuses to call the session finished until it has
verified that everything worth keeping exists somewhere that outlives the pod.

---

## Before you rent anything

All of this runs on a laptop, costs nothing, and catches most of what would
otherwise be discovered on the meter.

```bash
bash scripts/pod_session.sh --dry-run          # every step, printed not run
.venv/bin/python -m pytest tests/ -q           # 1052 passed, 34 skipped
bash scripts/run_all.sh --dry-run --profile crossing-uniform
```

The dry run degrades cleanly with no GPU and no vLLM: GPU checks report SKIP with
the reason, everything else runs for real. What it will NOT catch is anything
about the actual card, which is what the pre-flight is for.

Two things to do by hand before the pod exists, because both are slow to
discover late:

1. **Accept the Mixtral licence** and `huggingface-cli login`. Mixtral is gated,
   and the failure arrives after the download has started.
2. **Check volume size.** The sweeps download nothing at all -- they generate
   random weights -- but step 7 pulls 93.4 GB. 100 GB of Network Volume covers
   everything except trace capture; capturing Mixtral needs 250 GB.

---

## The pre-flight, and what each check kills

Runs automatically, takes about three minutes, spends almost nothing. Every check
below exists because something in this list actually went wrong on this project.

Run it alone with `bash scripts/pod_session.sh --preflight-only`.

| id | check | the class of failure it kills |
|---|---|---|
| P1 | working tree is clean | 76.7% of the canonical published pool carries `git_dirty = True`, and two whole arms are 100% dirty. Those rows are not reproducible from the commit they name. Waive with `--allow-dirty` and the waiver is recorded in the ledger. |
| P2a | GPU name matches `--expect-gpu` | `measured_<device>.yaml` resolves by NAME, so a second H200 pod silently inherits the first one's ceilings and an A100 would be scored against an H200 roof. **FATAL.** |
| P2b | at least 80 GB of device memory | deepseek-v3 at `E=256,N=2048` needs tens of GB for one layer. **FATAL.** |
| P2c | driver r580+ | a cu130 torch wheel needs it; on an older driver use the cu128 index. The index matches the DRIVER, not the image name. |
| P3 | torch 2.13.0 and Triton 3.7.1 | a different torch is a different CUTLASS, so the grouped GEMM you profile is not the one the published rows describe; a different Triton emits different PTX, so step 5 would answer about another compiler. |
| P4 | `override_config` binds and releases, on a shape vLLM has never seen | steps 2, 3 and 4 are all `override_config` experiments. If the hook does not bind they sweep nothing while printing a full table. deepseek-v3 (`E=256,N=2048`) is the sharpest probe available because vLLM v0.27.1 ships no tuned file for it on any card or dtype, so a bind failure cannot hide behind a file that happens to agree. **FATAL.** |
| P5 | an isolated `TRITON_CACHE_DIR` really produces PTX | with the shared `$WORKSPACE/triton-cache` inherited, every fused_moe specialisation is already built, nothing recompiles, no `.ptx` is written, and the dump script exits saying the kernel never compiled. This is very likely why the A100 was never successfully dumped. **FATAL.** |
| P6 | 110 GB on the volume, 10 GB on the container | the 93 GB download, and the several GB of temp space wheel extraction needs. |
| P7 | `entitled_ridge` still refuses 2 of the 10 published arms | the guard that stops an arm being quoted against another session's ruler. A change that silently stops refusing is invisible in any table. |
| P8 | Hugging Face auth | Mixtral is gated. |
| P9 | the exact exfil paths are committable | an unanchored `plots/` rule matched at any depth and silently swallowed `results/published/<arm>/plots/*.png` on every publish. Zero `.png` files are tracked under `results/published/` across all ten arms. **FATAL.** |
| P10 | which profiler exists | informational. `ncu` fails on a rented pod with `ERR_NVGPUCTRPERM`; `nsys` uses CUDA tracing and often works. |
| P11a | the six step scripts exist and parse | three of them are written concurrently by other people. |
| P11b | those scripts accept the flags this session passes | a renamed flag should cost a line here, not an argparse error forty minutes in. |
| P12 | the test suite | a failure here costs seconds; the same failure found after an hour of benchmarking costs an hour, and every row in between is suspect. **FATAL.** |
| P13 | no active throttle, card under 60 C | thermal state is the largest source of run-to-run disagreement on rented hardware, and the harness records the symptom rather than controlling it. |
| P14 | the session directory is on a different mount from `/` | a Network Volume at `/workspace` survives termination; the container filesystem does not, and `/workspace` on a pod without a volume attached looks identical. **FATAL.** |

Pre-flight runs to the END even after a fatal, so one three-minute pass shows you
every problem rather than one per re-rent. It then stops before spending
anything, and reports the FIRST fatal, which is usually the cause of the rest.

---

## The session

| offset | step | script | what it costs |
|---|---|---|---|
| 0:00 | 0. mixtral weights, backgrounded | `huggingface_hub.snapshot_download` | nothing on the critical path |
| 0:02 | 1. fp8 same-session calibration | `scripts/calibrate_hardware.py` | about 3 min |
| 0:05 | 2. BLOCK_M sweep, multi-tile | `scripts/block_m_crossing_sweep.py` | about 45 min |
| 0:50 | 3. GROUP_SIZE_M sweep | `scripts/group_m_alpha_sweep.py` | about 20 min |
| 1:10 | 4. tuned vs forced fallback | `scripts/tuned_vs_fallback.py` | about 30 min |
| 1:40 | 5. config and ISA provenance, PTX | `scripts/check_mma_path.sh` x 3 cells | about 30 min |
| 2:10 | 6. dense uniform grid | `scripts/run_all.sh --profile crossing-uniform` | about 90 min |
| 3:40 | 7. trace capture | `scripts/capture_traces.py` | about 40 min |
| 4:20 | 8. exfil | `scripts/publish_results.sh` plus checks | about 10 min |

Resume anywhere: `--from 5`, or `--only 6`. A step that already has a PASS and no
FAIL in `$SESSION/LEDGER.tsv` is skipped; `--force` re-runs it. The sweeps resume
through the harness's own `--run-id` manifest, which flushes per cell, so aborting
costs at most one cell.

---

## Step 0 (0:00) -- mixtral weights

**Why first.** 93.4 GB at roughly 90 minutes. Step 7 needs it at 3:40. Starting it
at 0:00 is the difference between a 4 hour 30 session and a 6 hour one.

**PASS** the background process started. **FAIL** nothing started, and step 7 will
say so again with the log tail.

**If it is still downloading at 3:40**, do not kill the pod. The volume keeps the
partial download and `snapshot_download` resumes. Capture `deepseek-v2-lite`
(31.4 GB, ungated) in the meantime.

---

## Step 1 (0:02) -- the fp8 same-session calibration

**One line, and it gates two results.**

Two published statements currently rest on a calibration that did not belong to
the session that measured them:

1. the dtype headline (bf16 1.162 against fp8 1.361 pooled, 1.475 matched) comes
   from an arm `entitled_ridge` REFUSES, because that arm's calibration measured
   no fp8 ceiling;
2. every fp8 row in it carries `achieved_peak_tflops = 0.0`, so its
   `implied_traffic_ratio` column is empty and `alpha` cannot be fit across
   dtypes at all.

**What this step does and does not fix.** Measuring an fp8 ceiling here makes
every fp8 row measured TODAY quotable. It does NOT rescue the published fp8 arm.
Restamping those old rows against today's ruler would be exactly the borrowed
calibration of `docs/INSTRUMENTATION.md` defect 7, and `recompute_ceilings.py` is
the wrong tool for it. The published arm stays refused until its rows are
re-measured.

**Prediction**, from the three existing H200 calibrations:

| quantity | expect | note |
|---|---|---|
| bandwidth, triad | 4374-4377 GB/s | reproduces to 0.06% across sessions |
| dense bf16 | 701-771 TFLOP/s | the term that does NOT reproduce: 9.9% spread |
| bf16 ridge | 160.3-176.2 FLOP/byte | the band every absolute figure carries |
| fp8_e4m3 | about 1409 TFLOP/s | 1.83x the bf16 figure |

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| S1a exit 0 | **STOP.** Without a calibration the sweep runs with EMPTY efficiency columns. On one occasion an H100 pod silently satisfied the repo's committed H200 yaml and the whole sweep produced blank efficiency. |
| S1b `checked_on` is today | **STOP.** The yaml was not rewritten, so this session would publish against another session's ruler. That is defect 7 exactly, and it cost claim C5 its target for three days. |
| S1c fp8 peak > 0 | any fp8 row measured today is again unquotable. **On an A100 this SHOULD fail**: Ampere has no fp8 tensor cores and a number there would be fiction. |
| S1d bandwidth 3900-4800 | above the band means your buffer fit in cache and it is not a DRAM measurement; below means the card is contended. Every `implied_traffic_ratio` is divided by this. |
| S1e ridge 150-185 | every AI-cap prediction in this session is stated against 160.3. Outside the band, re-derive them before reading step 2. |
| S1f fp8 rows carry a peak | the yaml has the ceiling but the driver is not stamping it onto rows, which repeats the defect this step exists to close. Look at `moe/bench/driver.py` around `achieved_peak_tflops`. |

The step also snapshots the yaml and its sha256 into `$SESSION/calibration/`.
Step 8 checks that sum again. That guard exists because on 2026-08-28 a
recalibration overwrote `measured_<device>.yaml` between a sweep finishing at
19:21 and its publish at 19:35; the two rulers disagree by 9.9% on the compute
ceiling and nothing downstream could tell.

---

## Step 2 (0:05) -- the BLOCK_M sweep, and the session's central result

**The science.** Arithmetic intensity of an MoE expert GEMM is

```
AI(r) = (2r/b) / Q(r),    Q(r) = 1 + alpha (ceil(r/BM) - 1)
```

The first M-tile reads the expert's weights in full and each additional M-tile
re-reads them, discounted by L2 by a factor `alpha`. Two consequences. AI is
BOUNDED at `2 BM / (alpha b)`, so a tile height can put the compute roof
permanently out of reach. And the crossing solves `R = ridge b Q(R) / 2`, which is
a step function on both sides and can therefore have several solutions or none.

**alpha was refit on 2026-09-01 from 0.10 to 0.558.** 90% band 0.529-0.588 over
10,813 admitted rows, placebo -0.002. The 0.10 was an estimator artefact: it came
from minimising the CV of a POOLED ratio, an objective that falls 0.7% across its
whole range and lets alpha absorb a between-cell level trend running the wrong
way. Changing only the estimator on the original 151 rows gives 0.484.

**Prediction, ridge 160.3, bf16.**

| BLOCK_M | AI cap | crossing | mixtral | qwen2 | deepseek-v3 |
|---:|---:|---|---:|---:|---:|
| 32 | 57 | **none, at any batch** | -- | -- | -- |
| 64 | 115 | **none, at any batch** | -- | -- | -- |
| 128 | 229 | R = 250 | 999 tok | 1998 | 7992 |
| 256 | 459 | R = 160 | 641 tok | 1282 | 5130 |

128 and 256 must separate by **1.56x**. At the retracted alpha = 0.10 all four
crossed and the spread was 1.10x. The two alphas are QUALITATIVELY different here,
which is what makes this worth a pod.

**Run it in the multi-tile regime.** C3 measured the tile at T=16, where every
expert is one tile at every BLOCK_M, so there were no re-reads to save and only
occupancy could move. That regime is not this one and its null result does not
transfer. Wave count must exceed about 10 on BOTH sides of a step, so occupancy is
saturated and any remaining movement is traffic.

**What PASS means for the paper.** The tile-corrected roofline survives a test
that could have killed it, and the ceiling `2 BM / (alpha b)` becomes a measured
result rather than a proposal. It also makes a strong statement about production:
vLLM's tuned configs run BLOCK_M = 16 through the whole decode range, and at
alpha = 0.558 that caps AI at 29 against a ridge of 160.3, so a decode-configured
MoE kernel is structurally incapable of reaching its compute roof at any batch
size.

**What FAIL means, and there are two different ones.**

- **32 or 64 DOES cross.** Then `alpha < 0.0998` after all, the ceiling is real
  but higher than the refit says, and the refit needs redoing. This is the more
  interesting failure and it is worth wanting.
- **128 and 256 separate by about 1.10x rather than 1.56x.** Then the uncorrected
  `2R/b` describes the data and the whole tile-corrected section retracts.

Neither is a broken run. Both are results. Do not retry either.

**Confound, named rather than hidden.** BLOCK_SIZE_M sizes the register
accumulator, so changing it changes occupancy as well as traffic, and a time
change is ambiguous between the two. That is why the sweep is designed around
where the crossing sits rather than around raw time.

**If the step dies.** Steps 3 and 4 still stand: they measure how alpha VARIES,
which is a separate claim from its level.

---

## Step 3 (0:50) -- GROUP_SIZE_M, is alpha a scalar

**What it tests.** The refit found alpha falling with GROUP_SIZE_M: 0.570 at 1,
0.488 at 16. That is exactly what a swizzle-for-L2-reuse mechanism predicts, which
turns alpha from a fudge factor into something with a named cause. But
**GROUP_SIZE_M 32 and 64 have ZERO discriminating rows in the published pool**, so
the direction is untested beyond 16 and cannot be tested from existing data at any
effort. Only `override_config` varying it at fixed batch settles it.

**Prediction.** alpha keeps falling monotonically at 32 and 64, and the fall
flattens as the swizzle stops buying reuse.

**PASS** means alpha may be reported with a mechanism instead of as an
unexplained constant. **FAIL**, meaning a rise at some point, refutes the swizzle
story and alpha goes back to being a fitted number. A non-monotonic alpha is a
real result: report it, do not retry it.

Note either way that alpha is not a scalar. It already drifts with BLOCK_M, 0.466
at 64 and 0.625 at 128, so any single number carries a range.

---

## Step 4 (1:10) -- tuned config against a forced fallback

**What it tests.** Only 2 of the 8 (model x card) cells in this study have a tuned
vLLM config at all, both on the H200: `E=8,N=14336` (mixtral) and `E=64,N=2560`
(qwen2). The other six take the hardcoded bf16 ladder, `M<=32 -> 16`,
`M<=96 -> 32`, `M<=512 -> 64`, else 128, and vLLM says so on the log with "Using
default MoE config. Performance might be sub-optimal!". Nothing in this study has
ever measured what that warning is worth.

**Prediction.** In the memory-bound regime the difference is small, because the
tile is not on the critical path there. In the multi-tile regime the tuned file
climbs to BLOCK_M = 128 at M = 256 while the fallback sits at 64, and the ceiling
says only 128 can ever cross. So the gap should OPEN with batch rather than being a
constant offset.

**Why it matters beyond the number.** This repo once published "BLOCK_M is not a
knob" and had to retract it, because vLLM and SGLang both ship tuned
`BLOCK_SIZE_M`. The honest question is what the tuning buys, not whether it
exists, and this step is the answer.

**PASS** gives a number for the cost of running six of eight cells on a fallback
ladder. **FAIL** as "no difference at any batch" is publishable too and says the
tuned files buy nothing in this regime.

---

## Step 5 (1:40) -- config and ISA provenance, and the file that must leave the pod

**Why this step is really about exfil.** C1 and C3 are the only claims in
`docs/FINDINGS.md` that rest on transient pod output. The PTX dumps, the CUTLASS
kernel names and the "Using default MoE config" warning were all quoted from run
logs that were never committed. Every other claim in that file can be recomputed
from `results/published/` on a laptop; those two cannot be checked without a GPU,
which is the hole a reviewer opens first.

**Three H200 cells, three different predictions**, derived by `tile_resolve`
against the vLLM v0.27.1 config snapshot:

| cell | resolved tile | ISA prediction | config line |
|---|---|---|---|
| deepseek-v3 T=16 | BM=16, warps=4 | wgmma **= 0**, mma.sync > 0 | "Using default MoE config" |
| deepseek-v3 T=256 | BM=64, warps=8 | wgmma **> 0** | "Using default MoE config" |
| mixtral T=256 | BM=128, BN=256, warps=8 | wgmma **> 0** | "Using configuration from" |

Triton selects Hopper's `wgmma.mma_async.m64nNk16` only when
`BLOCK_M % 64 == 0` AND `num_warps % 4 == 0`. So the first cell declining it and
the other two reaching it is ONE prediction, not three. The third is the tuned
specialisation FINDINGS names as never having been compiled to disk.

**On the A100 the prediction is different and simpler.** Below compute capability
9.0 `getMMAVersionSafe` returns `{2}` alone, so NO tile reaches the warpgroup
instruction at any size. Both cards means two pods: run `--only 5` again on an
A100 session and exfil that census too.

**PASS on all three** turns "the decode path is on the Ampere-era instruction, on
Hopper silicon" from an inference into a measurement, with a committed file behind
it.

**FAIL, and what each one means.**

- **wgmma present at T=16.** C3 is REFUTED. Triton is reaching M=64 some other way
  and the inference from `BLOCK_SIZE_M = 16` was wrong.
- **No wgmma at BM=128.** Triton declines the warpgroup instruction even when the
  tile allows it, which is a finding in its own right and a bigger one than C3.
- **config line "unlogged".** vLLM emits it once per `(E, N, dtype, device)` via
  `info_once`, so a second cell in the same process is silent. Check this was the
  first `fused_experts` call and that the log level lets info through. Without the
  line, every tile statement about the ten published v3 arms stays DERIVED from
  vLLM's source rather than OBSERVED.

**Each cell gets its own dump directory.** `check_mma_path.sh` begins with
`rm -rf` on its `--out`, so three cells sharing one directory would leave only the
last one's PTX.

**The dumps leave as a tarball.** `*.ptx`, `*.so`, `*.nsys-rep` and `*.qdrep` are
gitignored at any depth on purpose, so raw dumps cannot be committed. The step
writes `$SESSION/exfil/ptx-<card>.tar.gz` and a readable
`$SESSION/exfil/ISA_CENSUS.txt`, and pre-flight P9 has already proved both
filenames are committable.

---

## Step 6 (2:10) -- the dense uniform grid

**The profile.** `crossing-uniform`: uniform routing only, 7 seeds, a `2^(1/4)`
grid from 1 to 16384 straddling the ridge band, L2-WARM eager.

Three things in that sentence are decisions, not defaults.

- **Uniform only.** `2R/b` is a uniform-routing statement and pooling the seven
  routing regimes is INVALID for a crossing, not merely noisy: under skew the busy
  experts are compute-bound while the quiet ones are still memory-bound at the same
  batch, so the layer straddles the ridge and there is no single crossing. Dropping
  `--routing uniform` from any crossing command changes the numbers by up to 4.3x.
- **L2-warm.** The cold basis loses 5 of 8 one-stage crossings to throttle
  exclusion.
- **Read it with `octave_ladders`.** Fed whole to `crossing_from_points` this grid
  is biased 4-18% LOW and twice as wide as the powers-of-two grid it extends.

**Why the tile should be pinned.** Along an unpinned grid the tile CHANGES with
the token count -- mixtral climbs 16, 32, 64, 128 across the sweep -- so the
staircase the detector reads is partly the config ladder stepping and partly the
roofline transition, and `crossing_from_points` returns the FIRST crossing, which
is usually a tile step. That is instrument defect 3.

The step probes for a pinning hook by running one cell with
`MOE_FORCE_TILE={"BLOCK_SIZE_M":128,...}` and reading the OBSERVED `tile_block_m`
column back out of the CSV, because "the env var was set" and "the kernel ran that
tile" are different facts. **If gate S6a fails, nothing in the sweep path honours
the hook**, the grid runs on vLLM's own ladder, and its crossing may NOT be
described as tile-pinned. Use step 2 for the pinned answer instead. The unpinned
grid is still worth having: it is the crossing under the configuration that
actually ships.

**Prediction** at a pinned BLOCK_M = 128: mixtral crosses near 999 tokens, qwen2
near 1998, deepseek-v3 near 7992. At BLOCK_M = 64 the cap is 115 and no crossing
exists anywhere on the grid.

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| S6a tile pinning honoured | the grid is unpinned. See above. Not fatal. |
| S6b sweep exit 0 | resume with `--from 6` and the run id the step printed. |
| S6c zero correctness failures | **STOP.** The kernel computed the wrong layer, so every timing in the arm is a timing of the wrong thing. Do not publish it. |
| S6d under 5% throttled | throttled rows are excluded from crossing detection, so a high rate narrows the grid the detector can actually use. |
| S6e coverage against the planner's own row count | the sweep stopped short, almost certainly on `--max-minutes`. Resume rather than reading a crossing off a truncated grid. |

**Two defaults that are not measurements, and that any new analysis of these rows
must filter.** `ms_p50 = 0.0` means the cell never ran, not that it took no time;
`crossing.timed_rows` drops those. `implied_traffic_ratio = 0.0` means the column
does not apply, because `driver.py` writes it only for memory-bound cells; nothing
guards that one and the filter has to be written per analysis.

---

## Step 7 (3:40) -- trace capture

**What it fixes.** `traces/` holds a single `.gitkeep`. Every routing distribution
in this study is parametric, so every claim about realistic skew rests on zipf, hot
and dirichlet standing in for measurements never taken, and `capture_traces.py`
has never been run. Captures are kilobytes and are committed on purpose; model
weights never enter the repo.

**What fits on one H200 (141 GB, bf16).**

| model | weights | capturable here |
|---|---:|---|
| mixtral-8x7b | 93.4 GB | yes, comfortably |
| qwen2-57b-a14b | 114.8 GB | yes, about 26 GB left for KV and activations |
| deepseek-v2-lite | 31.4 GB | yes, the cheap 64-expert proxy |
| deepseek-v3 | 1369 GB | **no**, needs 5+ cards |

DeepSeek-V3's routing cannot be captured on this hardware. Benchmark its GEOMETRY
with parametric routing and say so explicitly wherever the result appears.
Claiming a captured V3 trace would be false and is the kind of thing a reviewer
checks first.

**FAIL** for a model means its skew claims stay parametric. That is a visible gap
rather than a silent one, because `traces/` is tracked.

---

## Step 8 (4:20) -- exfil, and nothing is torn down before it passes

**This repo has lost work twice at exactly this point.** Every published figure
was dropped by an unanchored `plots/` rule that `git add` applied silently, and
the A100 PTX was never produced because a shared Triton cache meant nothing
recompiled. Both losses were invisible at the terminal.

**What must leave the pod.**

1. the sweep CSVs and manifests, via `publish_results.sh`, with the calibration
   they were measured against beside them
2. that calibration's sha256, unchanged since step 1 wrote it
3. `ISA_CENSUS.txt` and `ptx-<card>.tar.gz` from step 5
4. `traces/*.npz` from step 7
5. every step log, so a number can be traced to the run that made it
6. `LEDGER.tsv`, the machine-readable record of every gate above

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| S8a every expected artefact exists and is non-empty | a step reported success and produced nothing. Check its log BEFORE tearing down; re-running one step is minutes and re-renting is an hour. An artefact is expected only when its producing step passed, so a skipped step never shows here. |
| S8b the calibration is byte-identical to step 1's snapshot | **STOP.** The ruler moved under the results and `publish_results.sh` would copy a calibration these rows were never measured against. Restore `$SESSION/calibration/` first. |
| S8c nothing is silently gitignored | **STOP.** `git add` drops an ignored file without a word. Rename the artefact or fix the rule. |
| S8d publish_results exit 0 | the commit may still exist locally: a public clone over HTTPS can pull but not push. `gh auth login`, or copy the session off with `runpodctl send $SESSION`. |
| S8e session artefacts tracked | the census and the logs exist on the pod and nowhere else, and they are what makes C1 and C3 checkable without a GPU. |
| S8f local HEAD equals the remote | the commit is local only. Push it or send the directory. |
| S8g MANIFEST.sha256 written | without it a truncated copy off the pod is indistinguishable from a complete one. |

**Then, and only then, stop the pod.** Everything under `$SESSION` is on the
Network Volume and survives termination. Everything under `/` does not.

---

## Reading the output

**Exit codes.** 0 every gate passed or was deliberately skipped. 1 a soft gate
failed and the session continued with a named consequence. 2 a fatal gate failed
and the session stopped. 3 the script was used wrongly.

**The ledger.** `$SESSION/LEDGER.tsv` is tab-separated: step, name, status,
observed, gate, consequence. It is what `--from` reads and what a post-mortem
should start from.

```bash
column -t -s"$(printf '\t')" "$SESSION/LEDGER.tsv"
awk -F'\t' '$3=="FAIL"' "$SESSION/LEDGER.tsv"
```

**The contract a step script honours.** It prints at least one line carrying the
word PASS and no line carrying FAIL, either as the first non-blank token or
immediately after a colon:

```
PASS  no crossing found on a grid reaching 16384
BLOCK_M=32: PASS  no crossing found on a grid reaching 16384
```

So prose in a step script must not start a line with PASS or FAIL, or put either
straight after a colon. `scripts/block_m_crossing_sweep.py` also offers
`--fail-on-gate`, which turns its scientific verdict into an exit code; this
session deliberately does not use it, because a scientific FAIL is a legitimate
outcome and should not be reported as a crashed run.

---

## Failure playbook

| symptom | almost certainly | do this |
|---|---|---|
| PTX dump is empty | a cache hit; Triton does not recompile what it has already built | check `TRITON_CACHE_DIR` is per-run, not `$WORKSPACE/triton-cache`. P5 tests this before you spend anything. |
| "Using ..." config line missing | vLLM logs it once per `(E,N,dtype,device)` via `info_once` | make sure the cell is the first `fused_experts` call in the process, and that info-level logging is on |
| every efficiency column is empty | no calibration resolved for this device | `python scripts/calibrate_hardware.py`; the file resolves by device NAME |
| sweep says the calibration is foreign | `measured_<device>.yaml` was overwritten between sweep and publish | restore `$SESSION/calibration/`, then publish |
| `ncu` says ERR_NVGPUCTRPERM | expected; performance counters need a host module flag a container tenant cannot set | use `nsys`, the measured ceilings, and the L2 flush axis. See `docs/RUNPOD.md`. |
| override_config appears to do nothing | the hook moved between vLLM versions | P4 catches this. `try_get_optimal_moe_config` reads it via `get_config()`, so it exists under some name |
| a step crashed on `--out` / `--out-dir` | a step script renamed a flag | P11b catches this. Fix the invocation in `scripts/pod_session.sh` |
| the whole script is a bash syntax error | an apostrophe inside a heredoc that sits inside `$( )` | bash 3.2 tracks quotes through it, and reports the error hundreds of lines away. No apostrophes in those blocks. |

---

## What this session does not settle

Worth keeping in view, because a good result here is easy to over-read.

- **DRAM traffic is still modelled, not counted.** Every byte figure remains
  compulsory-traffic arithmetic. `nsys` runs, its `--gpu-metrics-device` route is
  untested, and that is the open path rather than a closed door.
- **One GPU holding every expert.** DeepSeek runs decode on DP144+EP144 precisely
  to scale the aggregate batch past this ridge, and at that scale all-to-all
  communication dominates rather than the GEMM. Half the corrections in this
  project came from stating a single-node result as universal.
- **The regime is pure decode at modest concurrency.** At a few thousand tokens
  per forward, three of four models are already compute-bound. Crossing over in
  pure decode needs 316 concurrent sequences for mixtral and 3,010 for
  deepseek-v3.
- **The sweep is unsharded and unquantized.** Every cell is TP=1. Real serving
  shards, which changes `N` and therefore the config lookup, the tile and the
  block count.
- **The offload regime is untouched.** No host-to-device transfer has ever been
  measured here and the byte model has no offload path.
