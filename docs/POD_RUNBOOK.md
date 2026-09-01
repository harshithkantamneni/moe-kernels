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
.venv/bin/python -m pytest tests/ -q           # must be green; the count moves
bash scripts/run_all.sh --dry-run --profile crossing-uniform
.venv/bin/python scripts/alias_ablation.py --synthetic refit   # step 2b, no GPU
.venv/bin/python scripts/nsys_dram_probe.py --explain          # P-nsys, no GPU
```

The dry run degrades cleanly with no GPU and no vLLM: GPU checks report SKIP with
the reason, everything else runs for real. What it will NOT catch is anything
about the actual card, which is what the pre-flight is for.

**The test count is deliberately not written down here.** It was stale by a
hundred tests twice, because scripts land on this branch faster than the number
in a doc gets corrected, and a stale number teaches people to ignore the line.
P12 gates on `exit 0`, not on a count, and that is the thing to reproduce.

The last two lines are worth running before the pod because both exercise a whole
gate ladder off-GPU. `--synthetic refit` runs step 2b against a planted alpha and
must exit 0; `--synthetic retracted` plants 0.10 and must exit 1 with P1 FAIL,
which is what proves the gate can still refuse. `--explain` prints the sampling
arithmetic that decides what nsys can and cannot measure here, and it needs no
hardware at all.

Two things to do by hand before the pod exists, because both are slow to
discover late:

1. **Accept the Mixtral licence** and `huggingface-cli login`. Mixtral is gated,
   and the failure arrives after the download has started.
2. **Check volume size.** The sweeps download nothing at all -- they generate
   random weights -- but step 7 pulls 93.4 GB. 100 GB of Network Volume covers
   everything except trace capture; capturing Mixtral needs 250 GB.

---

## The pre-flight, and what each check kills

Runs automatically, takes about five minutes, spends almost nothing. Every check
below exists because something in this list actually went wrong on this project,
with one exception: P-nsys is not there to stop a failure, it is there because
what it finds changes what every later step is allowed to claim.

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
| P10 | which profiler exists | informational. `ncu` fails on a rented pod with `ERR_NVGPUCTRPERM`; `nsys` traces CUDA and usually works, but tracing kernels is not counting bytes and P-nsys below asks the harder question. |
| P11a | the step scripts exist and parse | several are written concurrently by other people. |
| P11b | those scripts accept the flags this session passes | a renamed flag should cost a line here, not an argparse error forty minutes in. |
| P12 | the test suite | a failure here costs seconds; the same failure found after an hour of benchmarking costs an hour, and every row in between is suspect. **FATAL.** |
| P13 | no active throttle, card under 60 C | thermal state is the largest source of run-to-run disagreement on rented hardware, and the harness records the symptom rather than controlling it. |
| P14 | the session directory is on a different mount from `/` | a Network Volume at `/workspace` survives termination; the container filesystem does not, and `/workspace` on a pod without a volume attached looks identical. **FATAL.** |
| P-nsys | can DRAM traffic be COUNTED here rather than modelled | not a failure gate. It decides whether steps 1 to 4 quote MEASURED or INFERRED bytes. **Never fatal, and absence is not even a soft FAIL.** Runs LAST in pre-flight because it is the only check that costs minutes. |

Pre-flight runs to the END even after a fatal, so one pass shows you every
problem rather than one per re-rent. It then stops before spending anything, and
reports the FIRST fatal, which is usually the cause of the rest. P-nsys is the
exception: once a fatal has tripped it is skipped, because that session is not
going to run and there is nothing for its answer to multiply.

### P-nsys, and why it moved to the front

**Every byte figure in this study is arithmetic.** Nothing here has ever counted a
DRAM transaction. `ncu` is walled off on a rented pod by `ERR_NVGPUCTRPERM`, so
compulsory-traffic bytes are computed from the shapes and divided into a measured
bandwidth. `nsys` reaches the DRAM counters by a different mechanism, sampling
rather than instrumenting, and whether it works on a given pod is an open
question this project has never answered.

**NOTHING IN THIS SECTION HAS TOUCHED A GPU, INCLUDING THE NUMBERS THAT LOOK LIKE
MEASUREMENTS.** The probe has never been run against a card. On a laptop it exits
3 with `REFUSED: no nsys on PATH`, which is what it should do and is not a
result. Three different kinds of statement live in this section and they are not
equally solid:

| statement | where it comes from | verified? |
|---|---|---|
| default 10 kHz, ceiling 200 kHz | NVIDIA's nsys documentation, hard-coded at `nsys_metrics.py:287` | **NO.** Never confirmed against any card, let alone an H200 SXM. |
| a 54 us launch buys 10.8 samples at that ceiling | arithmetic, `54e-6 * 200e3` | arithmetic is sound; it INHERITS the row above. |
| quantisation is charged per window, not per sample | arithmetic over the sampling model | same inheritance. |
| alpha to +/-0.156 at 10% traffic error | `alpha_uncertainty()`, propagated through the ladder | same, and it also assumes the sampler is unbiased, which is exactly what the calibration rung is for. |
| the sampler works on a rented pod at all | -- | **UNTESTED.** This is the probe's actual job. |

The direction of the risk is one-sided and worth stating: if the pod's real
ceiling is BELOW 200 kHz -- and the doc figure is a ceiling, so it can only be
lower -- then every conclusion above gets WORSE, not better. Fewer samples per
window, a larger edge term, a wider alpha band. A ceiling of 50 kHz would put a
54 us launch at 2.7 samples and widen the two-tile alpha band past the point
where it separates 0.558 from 0.10 at all. The probe MEASURES the rate actually
delivered rather than trusting the flag -- median inter-sample delta out of
`GPU_METRICS`, written to `$SESSION/nsys/probe.json` as `observed_sample_hz`
alongside a boolean `sample_rate_honoured` -- and that number, not the constant
in the source, is what any later claim must cite.

That check was added because the resolution arithmetic was self-referential
without it: `resolve()` divides the window by the period of the rate it
REQUESTED, so a build that clamped 200 kHz to 50 kHz would have left every
traffic total correct and every stated confidence wrong by a factor of four,
with nothing in the output looking odd. A clamp now sets `sample_rate_honoured`
false and forces `resolution_ok` false regardless of how the window scored;
`null` means the rate could not be measured at all, which voids the verdict
rather than widening it. The pre-check verdict is kept beside it as
`resolution_ok_before_rate_check` so the two causes stay distinguishable.

The one claim here that does NOT depend on the rate: **a single launch is
unmeasurable by orders of magnitude, not by a factor.** Reaching the minimum
samples per window for one 54 us kernel needs something above 370 kHz, so no
plausible correction to the doc figure rescues per-launch profiling. The merged
window design stands regardless of what the pod reports.

**It is a force multiplier, not a result**, and that is the whole reason it is in
pre-flight. If the sampler works, then steps 2, 3 and 4 and the dtype headline
stop being inferences from a byte model and become measurements checked against
counted bytes. Knowing that at 0:03 lets those steps say so. Knowing it at 3:00
is worth nothing.

**What it actually runs.** `scripts/nsys_dram_probe.py --calibrate`, capped at
twelve seconds per attempt. It asks the installed `nsys` which flags it offers
rather than guessing a spelling, walks a six-rung ladder that varies one thing per
rung, and scores each rung by whether `GPU_METRICS` came back non-empty with a
DRAM metric in it -- not by the exit code, because nsys exits 0 while writing a
report with no metrics in it. The last rung is a CONTROL with no metrics
requested, which separates "no sampler" from "no nsys". Then it profiles a
workload whose DRAM traffic is KNOWN without any model and checks the sampler
against it.

**Three outcomes, and only one of them is a FAIL.**

| outcome | what the session does |
|---|---|
| the sampler works and passes the known-traffic case | traffic is **MEASURED**. Step 2 gets a companion `--measure` run, and every later step says MEASURED. |
| no invocation sampled a DRAM metric | traffic is **INFERRED**. This is the EXPECTED outcome on a rented pod, it is an INFO row and not a FAIL, and the session runs exactly as it did before. |
| the sampler works and gets the known answer WRONG | a soft **FAIL** (`PnsysCal`). This is the one outcome worth a verdict, because an instrument that answers and answers wrongly would have corrupted every step after it silently. Traffic stays INFERRED. |

**What it does NOT buy, and this matters.** Even a working sampler cannot measure
a single MoE kernel launch. The rate ceiling nsys offers is 200 kHz, so one 54 us
launch buys ten samples and the quantisation is charged per WINDOW rather than per
sample: 10 ms of kernel time is usable as one contiguous window and useless as ten
separate ones. Profiling a thousand separate launches buys samples and zero
accuracy. So wrapping a 45-minute sweep in nsys would be pointless, and the
session does not do it. What IS measurable is a long contiguous run of
back-to-back launches merged into one window, which is why the traffic
measurement is a companion run of the cell rather than a trace over the sweep.

At 10% traffic error the sampler pins alpha to about +/-0.156 on a two-tile cell.
That **discriminates 0.558 from the retracted 0.10 and cannot pin either**;
+/-0.05 needs 3.2%. More M-tiles per expert sharpen it, which is why the cell
profiled is a deep one.

**Skip it** with `--no-nsys`. The session then runs as it did before P-nsys
existed and everything says INFERRED.

---

## The session

| offset | step | script | what it costs |
|---|---|---|---|
| 0:00 | 0. mixtral weights, backgrounded | `huggingface_hub.snapshot_download` | nothing on the critical path |
| 0:00 | pre-flight, P-nsys included | `scripts/nsys_dram_probe.py --calibrate` | about 5 min |
| 0:05 | 1. fp8 same-session calibration | `scripts/calibrate_hardware.py` | about 3 min |
| 0:08 | 2. BLOCK_M sweep, multi-tile | `scripts/block_m_crossing_sweep.py` | 4-6 min |
| 0:14 | 2. traffic measurement of the cell | `scripts/nsys_dram_probe.py --measure` | 2 min, only if P-nsys said yes |
| 0:16 | **2b. alias ablation, the independent alpha** | `scripts/alias_ablation.py --run` | 6-10 min |
| 0:26 | 3. GROUP_SIZE_M sweep | `scripts/group_m_alpha_sweep.py` | 12-15 min |
| 0:41 | 4. tuned vs forced fallback | `scripts/tuned_vs_fallback.py` | capped at 35 min |
| 1:16 | 5. config and ISA provenance, PTX | `scripts/check_mma_path.sh` x 3 cells | 15-30 min |
| 1:46 | 6. dense uniform grid | `scripts/run_all.sh --profile crossing-uniform` | about 54 min |
| 2:40 | 7. trace capture | `scripts/capture_traces.py` | 25-40 min |
| 3:20 | 8. exfil, and the two alphas reconciled | `scripts/publish_results.sh` plus checks | about 10 min |

**Does it still fit? Yes, with more room than the old plan had.** Worst case on
the pessimistic arm of every range reaches exfil at 3:20 and finishes at 3:30,
against 4:20 and 4:30 before; best case is 2:51. Two things bought that back: the
BLOCK_M sweep and the GROUP_SIZE_M sweep are both quicker than the original
estimates, and the two additions are small. The 93 GB download needs about 90
minutes and step 7 does not start until 2:40, so it is still nowhere near the
critical path -- which was the reason step 0 exists.

**Step 2b runs immediately after step 2 on purpose.** Both estimate alpha. One
session and one thermal state means a disagreement between them cannot be
explained away by the card having been a different card.

Resume anywhere: `--from 5`, or `--only 6`. `--only` takes any step id including
`2b`; `--from` takes a number, and `--from 2` includes 2b because 2b sits at
position 2 in the running order. A step that already has a PASS and no FAIL in
`$SESSION/LEDGER.tsv` is skipped; `--force` re-runs it. The sweeps resume through
the harness's own `--run-id` manifest, which flushes per cell, so aborting costs
at most one cell.

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

## Step 1 (0:05) -- the fp8 same-session calibration

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

## Step 2 (0:08) -- the BLOCK_M sweep, and the session's central result

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

**Read it with step 2b, not alone.** This step tests a consequence of alpha
against a byte model. Step 2b measures alpha without one. A disagreement between
them is not noise, and this step on its own cannot tell a wrong alpha from a wrong
byte model.

---

## Step 2b (0:16) -- the alias ablation, and the second alpha

**Why a second estimate at all.** `alpha = 0.558` carries the whole
tile-corrected roofline, and it comes from ONE regression against a byte model
with no tile term in it: `alpha_refit` fits it out of `implied_traffic_ratio`,
which is `time x bandwidth / COMPULSORY BYTES`. C4 in `docs/FINDINGS.md` is a
CONFIRMED finding that the compulsory-byte ruler was itself wrong by 1.85% until
it was fixed. A number a paper's headline rests on should not rest on one
estimator over one derived column.

**The method, which touches none of that.** Run the same grouped-GEMM access
pattern twice at `n = 1, 2, 4, 8` M-tiles per expert.

- **NORMAL** reads each expert weight block once per M-tile, so its HBM weight
  traffic is `W (1 + alpha (n-1))` in units of one full pass.
- **ALIASED** points every weight load at ONE resident tile. The loads still
  execute, the instruction stream is identical, but they HIT L2 instead of
  missing to HBM.

Everything else -- activation reads, output writes, arithmetic, launch, grid --
is identical and cancels:

```
D(n) = T_normal(n) - T_aliased(n) = W (1 + alpha (n-1)),   D(1) = W
alpha = (D(n)/D(1) - 1) / (n - 1)
```

`W` is never predicted from bytes and a bandwidth. It is MEASURED, in the same
units, in the same session, by the same clock. Rows per expert is an exact
multiple of `BLOCK_M` at every rung, so the padding term is exactly zero and the
tile count is the only thing that moves.

**Why the answer is an interval and not a point.** The aliased variant issues the
same loads; what it does not do is MISS. So both variants push `n W` bytes through
L2, and that common cost cancels only if L2 and HBM service ADD. A streaming
kernel runs closer to `max(L2, HBM)`, where the naive difference estimator fits to
`(alpha - r)/(1 - r)` and, at an `r` near 0.5 on this card, a true 0.558 would
come back as **0.018** -- which is to say the naive estimator is biased toward
almost exactly the value this repo retracted. So a second estimator, exact under
`max` and biased the other way under addition, is run beside it and the answer is
the BRACKET between them. The report prints the measured `r`, which is what sets
the bracket width.

**PREDICTION.** The interval contains 0.558 and excludes 0.10 and TEMPO's 0.33.

**The gates**, all of which appear in the log as `[PASS]` or `[FAIL]` lines that
`pod_session.sh` counts.

| gate | meaning of a FAIL |
|---|---|
| ISA | the two variants did not compile to the same instruction stream, so the difference is not measuring what it claims. **Nothing from the run may be quoted whatever P1 says.** The gate counts the UNION of `ld.global` and `cp.async`, because Triton pipelines the loads at `num_stages > 1` and counting `ld.global` alone would read zero. |
| correctness | a variant did not reproduce its closed form, so the aliasing scalars did not do what was intended. |
| placebo | two identical launches differ by a large fraction of D. The design is measuring noise. |
| signal | the weight read is not most of what the kernel does, so the difference is measuring something the label does not cover. |
| form | `D(n)` is not affine in `(n-1)`, so slope-over-intercept is not alpha. |
| control | an L2-resident geometry, whose per-expert block fits in L2 and therefore has no HBM re-read to save, showed an extra-tile cost anyway. Whatever it is, it is not weight traffic. |
| resolution | the interval is too wide to separate the candidates. The run says **NOT TESTABLE** rather than picking one. |
| P1 | the interval does not contain the refit. **This is the refutation and it is a result.** The gate line names which candidate the interval DOES contain. |

**Exit codes are meanings, not error levels.** `0` every gate passed. `1` a gate
failed, which is a refutation. `3` it could not run here; the deepseek-v3 rung
needs about 16 GiB free on the card. `4` the design did not identify alpha and the
honest answer is NOT TESTABLE.

**What this is NOT.** It is not vLLM's `fused_moe_kernel`. It has vLLM's B-pointer
arithmetic, vLLM's `GROUP_SIZE_M` swizzle, vLLM's `[E, N, K]` layout and vLLM's
tile constants, with the reduction replaced by `acc += tl.sum(a) + tl.sum(b)` on
`docs/STUDY.md`'s own instruction. That replacement is what keeps the estimator
unbiased: with a real `tl.dot` the aliased variant goes compute-bound while the
normal one stays memory-bound, `D(n)` loses a copy of the per-tile compute cost
and alpha is biased DOWN. `--compute dot` runs it that way anyway, prints the
bound, and refuses to answer P1 with it.

**One residual confound, bounded rather than removed.** Aliasing to a single tile
pins every load to a handful of L2 slices, so the aliased ladder is served more
slowly than aggregate L2 bandwidth suggests. That inflates it, and the effect is
contained by the bracket rather than eliminated. Spreading the alias would remove
the closed form the correctness gate checks against, which is a worse trade.

**Rehearse it off-GPU** with `--synthetic refit` (must exit 0) and
`--synthetic retracted` (must exit 1 with P1 FAIL). `--replay <dir>` re-reports a
finished run without a GPU.

---

## Step 3 (0:26) -- GROUP_SIZE_M, is alpha a scalar

**What it tests.** The refit found alpha falling with GROUP_SIZE_M: 0.570 at 1,
0.488 at 16. That is exactly what a swizzle-for-L2-reuse mechanism predicts, which
turns alpha from a fudge factor into something with a named cause. But
**GROUP_SIZE_M 32 and 64 have ZERO discriminating rows in the published pool**, so
the direction is untested beyond 16 and cannot be tested from existing data at any
effort. Only override_config varying it settles it, and NOT at fixed batch:
group_m_alpha_sweep.py deliberately refuses that instruction, because one batch
cannot identify alpha under this estimator -- the token count IS the intercept,
so a single x-level is absorbed exactly and only curvature is left. On the
design's own x values at 0.5% noise the top rung alone gives a 90% band of
0.373-0.756 against the seven-rung ladder's 0.552-0.580, 14x narrower against an
effect size of 0.082. The ladder is identical across every GROUP_SIZE_M, so the
cross-setting comparison is still at fixed design.

**Prediction.** alpha keeps falling monotonically at 32. **g=64 is NOT answerable on the
mixtral arm this step runs**: num_pid_m tops out at 59-61, so g=64 saturates at
every rung and the setting cannot be distinguished from g=32. Answering it needs a
second arm, `--model qwen2-57b-a14b --tokens 32,64,128,256,512,768,1024`, which
`pod_session.sh` does NOT invoke. Run it by hand if step 3 holds at 32 and the
trend matters, and the fall
flattens as the swizzle stops buying reuse.

**PASS** means alpha may be reported with a mechanism instead of as an
unexplained constant. **FAIL**, meaning a rise at some point, refutes the swizzle
story and alpha goes back to being a fitted number. A non-monotonic alpha is a
real result: report it, do not retry it.

Note either way that alpha is not a scalar. It already drifts with BLOCK_M, 0.466
at 64 and 0.625 at 128, so any single number carries a range.

**This step is also the session's REGRESSION estimate of alpha.** Its
`GROUP_SIZE_M = 1` row is what step 2b's ablation is reconciled against in the
summary, and unlike the published refit it was measured on this card in this
session. If step 3 does not run, the reconciliation falls back to the published
0.558 / 0.529-0.588 and says in the file that the number is not from this card.

---

## Step 4 (0:41) -- tuned config against a forced fallback

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

## Step 5 (1:16) -- config and ISA provenance, and the file that must leave the pod

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

## Step 6 (1:46) -- the dense uniform grid

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

## Step 7 (2:40) -- trace capture

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

## Step 8 (3:20) -- exfil, and nothing is torn down before it passes

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
7. **`$SESSION/alias_ablation/`**, step 2b's report and per-cell measurements.
   This is the only estimate of alpha in the session that does not go through the
   byte model, so it is the only thing that can corroborate or refute the number
   everything else is quoted against.
8. **`TRAFFIC_PROVENANCE.txt`** and `$SESSION/nsys/probe.json`, which say whether
   any byte figure here was counted or modelled.
9. **`ALPHA_RECONCILIATION.txt`**, the two estimates side by side with their
   intervals and the verdict on whether they agree.

Items 7, 8 and 9 were added on 2026-09-01. The manifest had already omitted the
outputs of steps 2, 3 and 4 once -- the entire scientific payload -- and a pod
torn down after a clean run would have taken the answer with it. Items 8 and 9 are
listed UNCONDITIONALLY, unlike everything above them, because they are written by
the session rather than by a step and their whole point is to exist even when a
step did not run: "traffic was INFERRED because nsys is not installed" and "only
one alpha exists" are both results that have to survive teardown.

**The two alphas are reconciled here, before the manifest is built.** Step 8
starts by reading step 3's regression alpha and step 2b's ablation interval out of
their logs, writing both to `ALPHA_RECONCILIATION.txt`, and saying plainly whether
they agree. The summary then prints that file ABOVE the gate counts, so it is the
last thing on the screen. A session that measured the study's central parameter
twice and buried the disagreement four hundred lines up a log would be a session
that answered the question and told nobody.

**Read the intervals with their asymmetry.** The regression band is a cluster
bootstrap over sampling noise. The ablation interval is a BRACKET between two
estimators biased in opposite directions by whether L2 and HBM service add or
compose as a max. They are not the same kind of object, so overlap is weak
evidence of agreement and disjoint is strong evidence of disagreement.

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| SRa the two alphas agree | the intervals are DISJOINT. **Do not publish either number.** Two routes that share no byte model, no bandwidth and no estimator disagree about the parameter the tile-corrected roofline rests on, so at least one of them is measuring something other than the extra-tile cost. Check first whether the ablation's ISA gate passed and whether the regression ran in the multi-tile regime at all. If only one estimate exists the gate SKIPs and says so. |
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
word PASS and no line carrying FAIL, in one of exactly TWO shapes, matched
exactly rather than loosely:

```
  [PASS] regime: every cell is memory bound          <- group_m, tuned_vs_fallback, alias_ablation
GATE 7  PASS  the crossing moved with BLOCK_M        <- block_m_crossing_sweep
```

**This was wrong until 2026-09-01 and the correction is worth knowing about.** The
first version of the matcher looked for PASS as the first non-blank token or
straight after a colon, and it was validated against synthetic log lines rather
than real ones. Against the actual output of the three step scripts it read **0
PASS and 0 FAIL from all three**, so those gates FAILED on a perfect run and a
genuine refutation was equally invisible. The entire scientific payload of the
session had no verdict channel. The rule is deliberately not a bare whitespace
boundary, because `block_m` prints the prose line "a FAIL here is the interesting
answer" and other scripts print "2 PASS, 1 FAIL" summaries; both would
false-positive under a looser rule, and a false PASS is the one direction this
must never fail in.

So prose in a step script must not begin a line with `[PASS]` or `[FAIL]`, or with
`GATE <n> PASS`. Verified against the real output of all four scripts the session
counts, including `scripts/alias_ablation.py`, which prints 13 `[PASS]` lines on
`--synthetic refit` and 12 PASS plus 1 `[FAIL]` on `--synthetic retracted`.

`scripts/nsys_dram_probe.py` deliberately does NOT use this convention and is not
read with it: it is gated on its exit code and on the contents of `probe.json`,
because "nsys exited 0" and "nsys sampled something" are different facts and only
the report distinguishes them.

`scripts/block_m_crossing_sweep.py` offers `--fail-on-gate`, which turns its
scientific verdict into an exit code, and **step 2 passes it**. Without it the
sweep exits 0 with gates failed, so the exit-code gate would pass on a refutation
and the session's central result would have no way to report one. The consequence
line on that gate says in as many words that a FAIL there is a result and not a
crashed run.

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
| P-nsys says INFERRED | almost always: this pod has no DRAM sampler, which is the expected case | nothing. The session is unaffected and every traffic figure says INFERRED. Read the CONTROL rung in `$SESSION/logs/nsys_probe.log`: it separates "no sampler" from "no nsys". |
| `PnsysCal` FAILS | the sampler answered and got a KNOWN answer wrong | do not quote anything measured through it. The session continues with traffic INFERRED. This is the failure the calibration exists to catch and it is worth more than the measurement would have been. |
| step 2b exits 3 | not enough free memory, or no triton | the deepseek-v3 rung needs about 16 GiB free. Run it after the card is idle, or with `--models mixtral-8x7b,qwen2-57b-a14b,deepseek-v2-lite`. |
| step 2b says NOT TESTABLE (exit 4) | the bracket is too wide to separate 0.10 from 0.558 | read the printed `r`, the aliased ladder's per-tile cost. It sets the bracket width, and a wide bracket means L2 service is a large share of the cost. Do NOT pick a candidate from a wide interval. |
| the two alphas DISAGREE | one route is measuring something other than the extra-tile cost | check the ablation ISA gate first, then whether step 3 ran in the multi-tile regime. Do not publish either number until it is resolved. This is a result, not a bug. |
| `--from 2b` is rejected | `--from` takes a number | `--from 2` includes 2b; `--only 2b` runs it alone. |

---

## What this session does not settle

Worth keeping in view, because a good result here is easy to over-read.

- **The nsys route is UNVERIFIED, and its supporting arithmetic is
  doc-sourced.** No part of the probe has run against a GPU, and the 200 kHz
  ceiling every sampling figure rests on is NVIDIA's documented number rather
  than one this project has observed. Cite `observed_sample_hz` from the probe's
  own output rather than the constant, and check `sample_rate_honoured` before
  quoting any resolution verdict. See the provenance table in the P-nsys
  section.
- **DRAM traffic is modelled unless P-nsys says otherwise, and even then only in
  aggregate.** Pre-flight now tests the `--gpu-metrics-device` route rather than
  leaving it untested, and the session labels every traffic figure MEASURED or
  INFERRED accordingly. But sampling is not a per-launch counter substitute and
  must not be described as one: a single MoE kernel launch is not measurable at
  any rate nsys offers, and what a working sampler buys is aggregate traffic over
  a long contiguous run of back-to-back launches merged into one window. That is
  enough to validate or refute the compulsory byte model at roughly the 1% level
  and enough to choose between 0.558 and 0.10. It is not enough to attribute
  bytes to a launch, and it is device-wide, so a neighbour process on the pod is
  bounded by the idle baseline rather than excluded.
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
