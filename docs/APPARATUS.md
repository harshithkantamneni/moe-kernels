# The apparatus: four modules a reviewer reads before the code

Written 2026-09-03, after the six-standard audit and the rebuild that followed
it. `docs/FINDINGS.md` says what was measured, `docs/STUDY.md` how each claim
got where it is, `docs/INSTRUMENTATION.md` what the old instrument did to the
numbers. This page says what the instrument IS now, in one place, so that a
number in a report can be traced to the code that produced it without opening
four modules first.

Every statement here is about the tree at the commit this file was written at.
Where a module's docstring says more, the docstring wins; this page is the map.

| module | what it is | the failure it exists to prevent |
|---|---|---|
| `moe/bench/timing.py` | one timing instrument, `time_kernel`, named by `TIMING_BASIS` | two timers whose numbers were compared as if one had made them |
| `moe/bench/exit_codes.py` | one exit-code table, one `RESULT:` line, for every experiment script and the session driver | two integers meaning two things in two files, and a summary that grepped prose |
| `moe/bench/provenance.py` | one provenance block in every report and CSV, and one run-id rule | 26 reports with no commit, card, ruler or timing basis; three run-id collisions that published one run's numbers under another's heading |
| `moe/bench/ai_model.py` | the byte model with both operand re-reads, and what a ladder fit actually returns over it | a cap read off a fitted alpha as `2BM/(alpha b)`, which is high by `(1+phi+delta)` |

---

## 1. `moe/bench/timing.py`: the instrument

**What it is.** `time_kernel(fn, *, warmup_ms, target_ms=200, trials=3,
l2_flush=True, reference_clock_mhz=None)` returns one `KernelTiming` and is the
only way this repository times anything it publishes. In order: warm up under
sustained load until `warmup_ms` of GPU time has been delivered; size `iters` so
that one trial lasts `target_ms`, from the warmup's own queue-deep per-call
time; prime one CUDA event pair per iteration; start a background clock poller;
run `trials` trials, flushing L2 before every call when `l2_flush`, with one
synchronise per trial; stop the poller; summarise.

**Its name.** `TIMING_BASIS = "queue-deep/l2-flush/clock-under-load/v2"` is
written into every row the instrument produces (`KernelTiming.instrument`, CSV
schema v5 column `instrument`). A row without that string was measured by
something else. The version suffix moves when a change would move a published
number and never otherwise.

**Why it exists.** Until 2026-09-02 the roof was timed queue-deep with L2
flushed per iteration while every ladder script carried a private `time_call`
that created its events inside the loop, recorded the start on an idle GPU,
synchronised after every iteration, never flushed and read no clock. The
exposed host prefix was bounded at about 0.18 ms per `fused_experts` call on the
H200 pod and 0.30 ms on the A100 pod: an 8-16% bias in alpha at the smallest
ladder cells, different per card, of the order of the cross-card effect the
study registered.

**Warmup is a duration, not a count.** `warm_until` runs `fn` in batches under
sustained load until `warmup_ms` of measured GPU time has passed, and refuses
(`TimingRefused`) at `warmup_ms <= 0`. A count was the wrong unit: a 1 ms kernel
needs hundreds of calls before the governor reacts and a 30 ms GEMM needs one,
so ladders that compared cells warmed at 5 calls against cells warmed at 20
were comparing clock states. Consequences for the scripts: `--warmup` /
`--warmup-ms` is milliseconds (default 300); `--iters` is RETIRED as a timing
knob and kept only in the run id, because the instrument sizes iterations from
the warmup's own per-call time. Any recipe that passes `--warmup 0 --iters 1`
cannot run on this instrument.

**Three verdicts on every cell**, all Optional, where `None` means "not
determined" and never "zero":

| flag | column | what it asks | the retired check it replaces |
|---|---|---|---|
| LEVEL | `clock_level_ok`, side in `clock_level_side` | was the SM clock under load inside the band [`LEVEL_FRACTION = 0.95`, `LEVEL_HIGH_FRACTION = 1.05`] of the reference clock the ROOF was measured at, in EITHER direction; a failure names its side, `low` or `high`. IT IS A RECORD, NOT AN EXCLUSION: since 2026-09-09 no consumer drops a row for it | nothing asked this. A card sitting at 1500 MHz for a whole cell, with a roof measured at 1980, passed the old check with drift 0.0. And until 2026-09-04 (03df2d4) LEVEL itself was one-sided, `load >= 0.95 * reference`, so a memory-shaped cell boosted to 1980 against a 1515 roof passed with its fixed-roof fraction inflated by 1.31x |
| DRIFT | `clock_drift_ok` | do the first and last under-load samples agree within `DRIFT_FRACTION = 0.05`, in EITHER direction. THE ONLY EXCLUSION: a cell is excluded if and only if this verdict failed | the old `clock_drift` fired only on a >5% DROP between two idle-instant samples |
| host-bound | `host_bound` | had the GPU fewer than `HOST_BOUND_BACKLOG_ITERS = 2` iterations of work queued when the host finished enqueueing any trial | nothing; the drained-queue case silently included host enqueue time in the interval |

**What the old throttle flag detected.** The retired `throttled` column compared
two idle-instant clock samples taken either side of a cell, both after a
synchronise, and fired on a drop over 5%. On the published alpha-0558 arm it
flagged 91% of vLLM rows above T=4096 while flagged and unflagged replicates of
the same cell timed at ratio 0.998 with identical end clocks. It was detecting
whether the START sample had caught the idle boost, not throttling. Every
"N rows throttled" figure in a pre-v5 `SUMMARY.md`, and every "unthrottled"
basis in `docs/FINDINGS.md`, is that flag. DRIFT under load is what replaced
it, and DRIFT is what a consumer filters on; a rise fails it as a drop does,
because a rise means the warmup did not reach the operating point. LEVEL is
scored beside it and says which clock state the cell ran in, which is a fact
about the tile rather than a reason to drop the row (next paragraph). Until
2026-09-09 this page said a consumer "must test LEVEL AND DRIFT"; that is the
rule that made the study's two primary tiles unmeasurable on the H200.

**The rule every consumer must apply, and what the side is for.** SINCE
2026-09-09 A CELL IS EXCLUDED IF AND ONLY IF `clock_drift_ok` IS FALSE. The
LEVEL side is written on every row and excludes nothing on either side.

Why the rule changed, in the numbers that changed it. The first H200 gaps
session (2026-09-09, `results/published/2026-09-09-nvidia_h200-gaps-session/`)
left 750 cells with an under-load clock on every one of them, and a census over
them says the SM clock under load is not a property of the card's health but of
the KERNEL, set per tile by that kernel's own power draw under the 700 W cap:

| what is held fixed | median SM clock under load | cells |
|---|---:|---|
| BLOCK_M=128 (any BLOCK_N; 1395 at BLOCK_N=64 alone, over 136) | 1395 MHz | 215 |
| BLOCK_M=256 | 1650 MHz | 311 |
| BLOCK_M=32 (any GROUP_SIZE_M) | 1736 MHz | 68 |
| BLOCK_M=64, GROUP_SIZE_M=1 | 1358 MHz | 16 treads, 15 of them below the band |
| memory-shaped cells (the calibration's own streaming load, cap_test's flush duty at T=56) | 1950-1980 MHz | |
| the calibration's dense bf16 8192^3 GEMM, 691 W | 1485 MHz (samples 1470-1515) | the reference |
| at fixed BLOCK_M=256: BLOCK_N=32 / 64 / 128 | 1725 / 1620 / 1560 MHz | 311 |

The reference is therefore not the middle of anything: 1485 MHz at 691 W sits
near the LOW end of what dense tensor work does on this card, and a +/-5% band
around it is 74 MHz where the session spans 660. Under the rule in force until
2026-09-09, LEVEL-low excluded 15 of the roofline arm's 39 cells (every
multi-tile BLOCK_M=128 subject cell; five of them missed the 1410.75 MHz floor
by 0.75 MHz, a twentieth of the 15 MHz the clock can even move in) and 110 of
the depth arm's 168 treads, in every rep and at every depth,
while excluding nothing at all in the arms whose tiles happen to sit near the
GEMM's clock. That is a rule against a TILE, and it made the study's two
primary tiles unmeasurable on this card on any rerun. There is no "level" state
to demand of a power-capped kernel.

What the side is for, then. `low` is a hungry tile at its own steady state;
`high` is a memory-shaped cell boosting above the reference, which on this card
is the normal state of every memory-bound tread. Both are sound timings. What
is wrong for both is the FIXED-roof fraction `pct_of_achieved_tflops`, scaled
by `load / reference`, and the driver writes the correction onto the row:
`roof_at_cell_clock_tflops` (`roofline.roof_at_clock`) and
`pct_of_roof_at_cell_clock`, scored only when the reference is graded
`under-load` and refused with the reason in `roof_note` otherwise
(`schema.has_cell_clock_roof` is the predicate to split a pool on). Which of
the two a gate reads is settled and is not a matter of taste: **compute-bound
CLAIM gates read the FIXED roof fraction**, because the GEMM and every cell ran
under the same 700 W cap, so the fixed roof is the fair delivered-throughput
comparison; **the own-clock fraction is printed beside it on every point line
and every gate line as issue efficiency**, and is a gate input only for
memory-shaped cells and cross-card work, labelled. Scoring a compute-bound
claim at the cell's own clock credits a tile for its own throttle: on the
2026-09-09 roofline cells it flips C3's control-subject gap from +0.053 to
-0.032, which is a sign change on a registered claim.

The rule in one line: DRIFT alone excludes; the LEVEL side is recorded and
excludes nothing; a compute-bound gate reads `pct_of_achieved_tflops` and
prints `pct_of_roof_at_cell_clock` beside it. The driver's `throttled` column
encodes exactly `clock_drift_ok is False` (`moe/bench/driver.py`), so a
consumer may branch on it instead of re-deriving anything. What failure this
prevents: a consumer that treats `clock_level_ok = failed` as "exclude" drops
either every boosted row (the one-sided reading found at thirteen consumers on
2026-09-08, the fifteenth instance of a fix landing at one of two call sites)
or every hungry tile (the two-sided reading that landed two arms INVALID on
2026-09-09, the sixteenth). A consumer is correct only when a planted
1980-against-1485 row with side `high` and a planted 1395-against-1485 row with
side `low` are both KEPT and a planted DRIFT row is EXCLUDED.

**What DRIFT is, and why it is the instrument's job and not the gate's.** All
135 drifted cells of the 2026-09-09 session are the first cell of a rep after a
workload change: the governor settling, on the shortest cell, in the direction
the new tile's power draw demands (1875 to 1725, 1560 to 1650, 1560 to 1650,
1620 to 1710 on the four with samples on disk). A gate cannot fix that, and
widening it would only hide it. The instrument does: after `warmup_ms`,
`warm_until` keeps warming until two consecutive NVML reads agree within one
15 MHz step (capped at 3x `warmup_ms`), records `settle_ms` and the number of
extra warm calls on the row, and a DRIFT note names the direction (settling
upward, dropping, oscillating). `TIMING_BASIS` moves to a v4 string for it,
because it moves published numbers. Every writer persists the first and last
under-load samples and the sample list, and `power_w` is read at the same NVML
call and carried on every row, so a future LOW can be told apart as a throttled
card rather than a hungry tile. Any band edge that survives anywhere snaps to
the 15 MHz grid: the retired LOW edge at 1410.75 sat inside a step, so 1410 was
LOW and 1425 was level for timings 0.1% apart, and 23 of 148 LOW verdicts sat
within one step of it.

**The reference clock, and the fp8 family.** LEVEL is a comparison and half a
comparison is not a verdict, so `clock_level_ok` is `None` unless
`reference_clock_mhz` is supplied. `roofline.reference_clock(gpu_name)` reads
it from THIS card's committed calibration,
`moe/bench/hardware/measured_<card>.yaml`, trying in order
`detail.gemm_clock.median_mhz`, `detail.gemm_clock_mhz`, then
`detail.settle.final_mhz`, and returns `mhz=None` with a reason when the card
has no measured file or the file carries no clock. That is why a session starts
with `scripts/calibrate_hardware.py --publish` and refuses to continue without
it (section 5). The fp8 family resolves its own reference, the fp8 GEMM's
under-load median (1395 MHz at 690 W on this card), and that too is A RECORD:
fp8 compute-bound cells are scored against the fixed 1469.9 TFLOP/s fp8 roof
with `1469.9 x load / 1395` carried beside it, and there is no fp8 band either
(76 of 84 fp8 rows sit above 1395).

**Two things the instrument does not cover yet, stated so nobody infers them.**
The roof itself is timed by `moe.bench.calibrate` through `timing.time_eager`
(queue-deep, pre-primed events, L2 flushed), and `calibrate_hardware.py`
records that function's name in its `instrument` field rather than
`TIMING_BASIS`. And every `merged.csv` under `results/published/` is schema v3
or v4: no published row was measured on `time_kernel`, and a pre-v5 row reads
back with the retired timer's name through `schema.instrument_of` rather than
with a blank.

---

## 2. `moe/bench/exit_codes.py`: the table and the one greppable line

**The table.**

| code | name | meaning | ledger treatment |
|---:|---|---|---|
| 0 | `DONE` | measured; every VALIDITY gate PASSED; every CLAIM gate PASSED | finished, latched |
| 1 | `CLAIM_FAIL` | measured; VALIDITY passed; at least one CLAIM gate did not. A RESULT, not a retry | finished, latched |
| 2 | `REFUSED` | nothing measured; a precondition was not met. Free | finished; re-attempted on the next run because a refusal costs nothing and the usual reason to re-run is that the precondition was fixed |
| 3 | `INVALID` | measured; a VALIDITY gate FAILED after measuring. Nothing on the page may be quoted | finished, latched, NOT auto-retried |
| 4 | `ERROR` | crashed; an exception the script did not plan for | `RETRY` |
| anything else | | a signal, a shell 127, a traceback that escaped | `RETRY`: a code nobody chose carries no information |

**The rule** (`classify(gates)`), applied in order: any VALIDITY gate not PASS
gives INVALID; else any CLAIM gate not PASS gives CLAIM_FAIL; else DONE.
"Not PASS" covers FAIL and UNKNOWN alike, because a gate that could not decide
has not passed, and `classify([])` RAISES rather than returning DONE, because
"a check that examined nothing reports zero failures" is this project's
documented failure shape. REFUSED and ERROR are never returned by `classify`:
a refusal is decided before any gate is scored and an error by the handler
around the whole script.

**Why INVALID is not REFUSED.** Both are unquotable. REFUSED cost nothing and
is the script protecting the pod; INVALID cost the whole arm and is the
instrument reporting that it broke while in use. Until 2026-09-02 the driver
read 2 as REFUSED while `scripts/memory_branch_anchor.py` spent 2 on a VALIDITY
failure after an eight-minute measurement, so a measured-and-invalid run was
logged as free and printed "REFUSED BEFORE MEASURING. Nothing below is a gate".

**The line.**

    RESULT: <KIND> <NAME> <VERDICT> <detail>

`KIND` is `VALIDITY` or `CLAIM`; `NAME` is one token with no whitespace;
`VERDICT` is `PASS`, `FAIL` or `UNKNOWN`; `detail` is the rest of the line and
may be empty. Anchored at column zero. `result_line` renders it and refuses a
name with whitespace or a detail with a newline; `parse_result_lines` reads it
back and matches nothing else. The driver's summary greps `^RESULT: ` and
nothing else: a line that merely contains "PASS" or "floor" is prose, and prose
is what the previous summary read a pre-registered expectation out of. Every
scored gate prints exactly one such line; nothing that is not a scored gate
prints one.

**The second opinion.** `classify_text(log)` recomputes the exit code a log's
RESULT lines imply. Compared with the code the process returned, a
disagreement is one of two things: the script printed one verdict and exited
another, which is the defect the module exists to prevent; or the script
printed every line and THEN crashed, in which case the process code wins and
the arm is RETRY with the traceback as the reason. A log with no RESULT lines
raises `NoGatesScored`, which is what a REFUSED log looks like from here, and
also what an arm that was never scored looks like: the summary reports that
silence as a finding rather than finding words.

**Self-test.** `python -m moe.bench.exit_codes --self-test` walks every row of
the table, every ledger word, one RESULT line buried in prose, and asserts that
a planted VALIDITY FAIL is not DONE. The proof that the self-test can itself
return 1 is `tests/test_exit_codes.py`, which plants a wrong expectation in the
table.

---

## 3. `moe/bench/provenance.py`: the block and the run id

**The block.** `provenance_block(instrument=..., ridge=..., ridge_source=...,
bandwidth=..., bandwidth_source=..., warmup_ms=..., iters=..., target_ms=...)`
probes the machine and returns a frozen `Provenance` with these fields:

    provenance_version   git_sha   git_dirty   git_dirty_files   utc   hostname
    gpu_name   driver_version   cuda_version   python   torch   triton   vllm   sglang
    instrument   ridge   ridge_source   bandwidth   bandwidth_source
    warmup_ms   iters   target_ms
    missing: {field: reason}

It never raises for anything the machine did. Every field it could not
determine is `None` and `missing[field]` says why: "no CUDA", "not a git
repository", "package not installed", "not supplied by caller", "torch import
failed: OSError". A block with a `None` in it is complete; a block that guessed
is not, which is why there is no "unknown" card, no "0" sha and no default
bandwidth. A ridge or bandwidth supplied WITHOUT its source is recorded, since
it is what the run used, and its `*_source` is listed in `missing` as
"supplied without a source", so the publish gate fails on that report and the
script has to say where the number came from.

`git_dirty_files` counts FILES from `git status --porcelain
--untracked-files=all`, untracked files included, because on 2026-09-01 the pod
ledger recorded "0 dirty file(s)" for a session in which `calibrate` had
already rewritten a tracked yaml.

**Where it goes.** `.as_dict()` nests under a `provenance` key in a JSON
report; `.as_columns()` flattens into a CSV row under the `prov_` prefix with
`prov_missing` as one `field=reason; field=reason` string; `.stamp(payload)`
also writes the five keys the audit's publish gate checks at the top level
(`git_sha`, `gpu_name`, `ridge_source`, `bandwidth_source`, `instrument`) and
raises `ProvenanceCollision` if any is already present with a different value.
A key whose value is `None` is still written as `None`: an off-GPU stamp gives
`gpu_name: null` beside `missing.gpu_name: "no CUDA"`, so a gate that checks
only for the PRESENCE of the keys is rubber-stamping.

**Run ids.** `run_id(card=..., **swept)` returns
`<card_slug>-<k1><v1>-<k2><v2>-...-<hash8>`: the card first, where `ls` shows
it; every swept knob rendered in sorted name order; a sha1 over the full key so
any change to any value changes the hash. It raises `NoCard` on a missing or
empty card and `UnresolvedKnob` on a `None` or empty-string value. The three
collisions it exists to prevent are the same defect three times, a knob left
out of the id so a second run derived the first run's directory, resumed into
it, measured nothing and printed the first run's numbers under the second run's
heading: GROUP_SIZE_M (a G=16 run published G=1's timings); the card (the same
seven report names exist under both the H200 and the A100 surface arms, for
sm_count 132 and 108); and `--iters` / `--warmup` (a 200-iteration re-run
landed in the 50-iteration directory). Knobs that only re-analyse existing
cells (`--ridge`, `--alpha`, a band) belong OUT of the key.

**Where it is not.** Zero of the 26 committed `*.report.json` under
`results/published/` carry the block; they predate it, and
`results/published/CALIBRATION_PROVENANCE.md` is what says which ruler each
arm may quote in the meantime.

---

## 4. `moe/bench/ai_model.py`: the byte model, and what a fit returns

**The three terms.** For a tiled GEMM `C[M,N] = A[M,K] @ B[K,N]` with tiles
`BM x BN`, each of the `ceil(M/BM)` M-tiles reads a `K x BN` slice of B and
each of the `ceil(N/BN)` N-tiles reads a `BM x K` slice of A:

    A bytes = M K b (1 + alpha_a (m - 1))      activations, re-read per N-tile
    B bytes = K N b (1 + alpha_b (n - 1))      weights, re-read per M-tile
    C bytes = M N b                            written once

`alpha_b` is the study's alpha, the fraction of a fresh weight read that each
extra M-tile costs after L2. `alpha_a` is its counterpart on the activations,
and it HAS NO MEASUREMENT ANYWHERE IN THIS REPOSITORY: every function takes it
as an argument and none assumes one.

**What a ladder fit returns.** A timing ladder `t(n)` over `n = 1, 2, 3, ...`
full M-tiles is, in units of one weight read `W = K N b`,

    t(n) = (1 - alpha_b + delta) + n (alpha_b + phi)

with `phi` the activation-plus-output cost of ONE M-tile in weight-read units
(`phi()`; at `BN | N` it is `alpha_a BM/BN + BM/K + (1 - alpha_a) BM/N`) and
`delta` the fused layer's fixed cost in the same units. The estimator this
repository uses, `LadderFit.alpha = B/(A+B)`, slope over fitted level at n=1,
therefore returns

    alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                    (EXA)

The first version of this module asserted instead `alpha_fitted = alpha_b +
alpha_a BM/BN + BM/K` (LIN), which is (EXA)'s numerator with the level taken
as 1. No estimator in the repository does that, and the two agree only as
`phi -> 0`; on mixtral at BN=64, phi is 0.16 / 0.32 / 0.64 at BM = 64 / 128 /
256, so the replacement is not an approximation this study may use. (EXA) is
verified by execution: `tests/test_ai_model.py` runs `LadderFit` on a
`traffic()` ladder and asserts (EXA) against what comes back, and an
independent review on 2026-09-03 exercised the identity over 155,520 cases
with 0 failures.

**The bracket.** Under (EXA),

    2 BM / (alpha_fitted b) = (1 + phi + delta) * 2 BM / (b (alpha_b + phi))

so every cap computed as `2BM/(alpha b)` from a ladder alpha is HIGH by the
factor `(1 + phi + delta)` (`lin_overstatement`). The corrected cap is
`cap_from_fitted(alpha_fitted, block_m, b, phi, delta)`, which inverts (EXA)
through `alpha_b_from_fitted` and REFUSES (`AIModelRefused`) a recovered
`alpha_b` outside `[0, 1]`, a NaN, or an infinite input. Because `phi` depends
on `alpha_a` and `alpha_a` is unmeasured, EVERY corrected cap in this
repository is a bracket over `alpha_a`, never a point, and a report that prints
one must print the `alpha_a` it was evaluated at beside it. The often-quoted
16% / 32% / 64% overstatement at BM = 64 / 128 / 256 holds at `alpha_a =
0.143` and nowhere else; that value is the (LIN)-era solve of two anchor
points, which the module withdraws (the same two points through (EXA) give
`alpha_b = 0.07`, `alpha_a = 0.72`, and two points cannot choose a reading).
`docs/FINDINGS.md`'s RETRACTIONS section carries the bracket for the BLOCK_M =
128 row.

**What this retired.** The cap identity "`2BM/(alpha_fitted b)` is exactly the
three-term cap, so the published caps stand" (commit f732035); the BLOCK_M =
128 straddle of the ridge on either card; `alpha_b = 0.307` and its 2% match
to TEMPO's `b2/b`, which was a unit artefact of reading through (LIN). All
three are listed with their corrected numbers in `docs/FINDINGS.md` and
`docs/STUDY.md`.

---

## 5. The session driver: `scripts/h200_gaps_session.sh`

The driver measures nothing itself. It owns the ORDER of the arms, the
exit-code contract (its `ledger_state` is a shell mirror of section 2, and
`tests/test_h200_gaps_session.py` asserts the two agree code for code), and the
closing summary. `docs/POD_RUNBOOK.md` is its operator page; this section is
the vocabulary.

**Ledger states, measuring mode** (`$SESSION/ARMS.tsv`, one row per arm, last
row wins):

| word | from | resume behaviour |
|---|---|---|
| `DONE` | exit 0 | skipped on every later run |
| `CLAIM_FAIL` | exit 1 from a file that imports `moe.bench.exit_codes` | LATCHED: skipped on every later run. Delete the row to force one |
| `UNKNOWN` | exit 1 from a file that does NOT import the module | NOT latched, not RETRY either; disclosed by name, read the log and decide |
| `REFUSED` | exit 2 | re-attempted on the next run, because it cost nothing |
| `INVALID` | exit 3 | LATCHED, not auto-retried: the instrument broke while in use |
| `RETRY` | 4 or anything outside the table | re-attempted; the session exits 4 over it |
| `NOT_PLANNED` | the arm was deliberately not run | named with the reason, never silently absent |

**Ledger states, `--dry-run`** (`$SESSION/ARMS-dryrun.tsv`): `PLANNED` (the
arm printed its plan; rc 0, or rc 2 after printing one), `PLAN_REFUSED` (rc 2
with nothing printed, which off a GPU box is the refusal working),
`BROKEN` (anything else, a traceback included; the session exits 3),
`NOT_PLANNED`.

**Run ids and results.** Every arm is given `MOE_RESULTS_DIR=<root>/gaps-<card>`
with the card in the path, and an operator-supplied `MOE_RESULTS_DIR` that
does not contain the card is REFUSED, because a network volume outlives the
pod and a second card would otherwise resume the first card's directories.

**The calibration gate.** Arm 0 runs `scripts/calibrate_hardware.py --publish`.
After it, the driver refuses the rest of the session unless BOTH halves answer:
the tracked `moe/bench/hardware/measured_<card>.yaml` carries a
`provenance.utc` at or after this session's own start (`PUBLISHED`, against
`MISSING`, `UNDATED`, `STALE`, `NO_BASELINE`), AND the calibrate row in this
session's ledger reads `DONE`. The second half is not redundant:
`calibrate_hardware.py` copies the yaml into the tree BEFORE it scores a gate,
so an INVALID calibration leaves a fresh-stamped ruler nothing verified.

---

## 6. Where to look next

- `docs/POD_RUNBOOK.md`: the driver as an operator runs it, `SESSION=` resume,
  and the arm ledger with what each arm buys.
- `docs/FINDINGS.md` and `docs/STUDY.md`: the RETRACTIONS sections at the top
  of each, which list what the modules above withdrew and where the corrected
  number lives.
- `tests/test_docs.py`: every count this page and its neighbours quote (tests,
  arms, rows, ridges, the (EXA) bracket) is asserted against the tree, so a
  stale number fails the suite rather than teaching a reader to ignore the
  line.
