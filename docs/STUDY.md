# The study: what this project is now, and what each piece is for

Written 2026-08-27. Read this first if you have been away from the project.

This file is the working state: how each claim got where it is, what was
retracted, what runs where, what to do next. The RESULTS are in
`docs/FINDINGS.md`, organised around C1-C5 and regenerated from the published
rows.

## RETRACTIONS (read before any table below)

Added 2026-09-03. This file is the working state and is NOT regenerated from
the rows, so it carried several figures for days after the tree withdrew them.
Each entry says what was withdrawn, when, why, and where the corrected number
lives; the retracted text below is left in place and marked inline
"(retracted ...)" so the history is visible without git. The mechanism behind
(a)-(c) is `moe/bench/ai_model.py`; behind (e) and (f) it is
`moe/bench/timing.py` and `moe/bench/roofline.py`; `docs/APPARATUS.md` is the
one-page summary of both.

- **(a) The cap identity, `cap = 2 BM / (alpha b)`.** Withdrawn 2026-09-02.
  The study's alpha is a ladder fit, `B/(A+B)`, and over the three-term byte
  model that fit returns `(alpha_b + phi) / (1 + phi + delta)`, so a cap read
  from it as `2 BM / (alpha b)` is high by `(1 + phi + delta)`: 1.11x and
  1.21x on the two BLOCK_M=128 ladders, up to 1.36x at BM=128 on the
  single-GEMM model with alpha_a = 0.143. `alpha_a` is unmeasured, so every
  corrected cap is a bracket over it, never a point. Corrected form:
  `ai_model.cap_from_fitted`. Where this file says "AI is BOUNDED at
  2 BM / (alpha b)" it is stating the three-term cap at alpha_b, which is
  right, and reading a fitted alpha into that slot, which is not.
- **(b) The BLOCK_M=128 straddle.** Withdrawn 2026-09-02. The published caps
  of 150.4 (A100, above its 145.8) and 158.6 (H200, below its 162.8) were the
  (a) identity. Through (EXA) with the fused layer's own phi and delta at its
  floor they are 135.4 (0.929 of the A100 ridge) and 130.7 (0.803 of the
  H200 ridge), both upper bounds, no straddle on either card
  (`scripts/bm128_depth.py --audit`). `docs/FINDINGS.md` RETRACTIONS carries
  the bracket for the pooled 0.558 row.
- **(c) alpha_b = 0.307 and the 2% match to TEMPO's b2/b.** Withdrawn
  2026-09-02: a unit artefact of reading two anchor points through the (LIN)
  form; through (EXA) the same G=1 ladders read alpha_b near 0.92, which is the
  number that has to be explained (`scripts/bn_decomposition.py`, LADDER
  column). This file never quoted 0.307; listed because scripts and STUDY's
  readers did.
- **(d) The direction of alpha with GROUP_SIZE_M, on either card.** Withdrawn
  2026-09-02. The pooled surfaces' fall of the swizzle column (H200 0.827 to
  0.675, A100 0.896 to 0.782) was composition: the G=1 median mixed two models
  and the G=64 median was one, and the one matched cell on the A100 moves the
  OTHER way, by 0.028, with no paired MDE possible from one cell. No direction
  is established; the refit's "0.570 at 1, 0.488 at 16" below stands as a
  fit over the published pool and predicts nothing beyond it
  (`SURFACE.txt` and `SURFACE.pooled.txt` in the two cross-card-s3 arms).
- **(e) The ridge "band" 160.3-176.2.** Withdrawn 2026-09-02 as a ridge of
  any card. It is this one H200's compute ceiling failing to reproduce across
  sessions (`docs/INSTRUMENTATION.md` entry 6), and 26 ladder reports on BOTH
  cards had been scored against it. The committed calibrations gave H200 162.8
  and A100 145.8 FLOP/byte when this entry was written; the H200's was
  re-measured to 152.8 on 2026-09-09, which is that ceiling failing to
  reproduce once more and rescores nothing. Every report has been rescored to the attached
  card's own figure with `rescored_from` keeping the withdrawn one, and each
  ladder arm's `NOTE.md` says so. Where this file quotes 160.3 or 176.2 it
  names one calibration of one card, and is marked where it called that a
  band.
- **(f) The throttle flag.** Withdrawn 2026-09-02. It compared two
  idle-instant clock samples either side of a cell and fired on a drop, so it
  detected whether the first sample had caught the idle boost: on the
  alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged and
  unflagged replicates timed at ratio 0.998. "Unthrottled" bases in this file
  are that flag. The replacement is LEVEL and DRIFT under load
  (`docs/APPARATUS.md` section 1), and no published row carries it yet.
- **(g) `--warmup` is milliseconds of sustained load, not a count**, and
  `--iters` is retired as a timing knob. Not quoted in this file.
- **(h) Gate 3 of the BLOCK_M sweep is two-sided against the ridge band**,
  not a one-sided test against a 0.33 midpoint. Not quoted in this file.
- **(i) The regime table.** Withdrawn 2026-09-02: "59 of 87, up to 32, and
  16/32/64 never" was counted on seed rows. On max rows, which is what
  `moe_align_block_size` pads to, and on UNIFORM routing: BLOCK_M=128 runs
  multi-tile in 65 of 87 cells, up to 33-34 tiles per expert; BLOCK_M=16 in
  1 of 24 and BLOCK_M=64 in 5 of 112. Skewed routings were never counted.
- **(j) The alias arm's plan and pod figure agree at 13.7 min**; the 11.6 the
  plan used to print was the probe left off the plan page. This entry read 13.0
  until 2026-09-09, which is the BOOKING and not the figure. Not quoted in this
  file.
- **The C2 headline is at pooled routing.** The fp8/bf16 crossing table below
  (454 / 810 / 922 / 3240 bf16 tokens, 1.15 +/- 0.07) pools seven routing
  regimes, which `docs/FINDINGS.md` C5 shows is invalid for a crossing.
  Uniform-only the bf16 crossings are 313 / 730 / 931 / 2925 and the headline
  is 1.131 +/- 0.095: the centre barely moves and the dispersion grows 38%
  (`scripts/dtype_tile_confound.py`, `PUBLISHED_SHIFT`). The table is left
  as published and marked.
- **The whole-layer crossing table** below was scored at ridge 176.2 on
  pooled routing; the arm's own rows carry 160.3 and pooling is invalid.
  Re-scored 2026-09-01 in `docs/FINDINGS.md` ("What a whole MoE layer costs");
  the table here is marked.
- **The C5 table** below is the pooled, wrong-target reading that FINDINGS
  rewrote on 2026-08-31 (wrong target, invalid routing pool, two cards on
  different kernels; the "monotonic in expert count" pattern is a pooling
  artefact). Marked; FINDINGS C5 is current.
- **(m) `alpha_b = 0.9794 +/- 0.0113`, (n) `cap/ridge = 0.080`, and (o) the
  three-term model's activation re-read term.** Withdrawn or refuted
  2026-09-10; `docs/FINDINGS.md` RETRACTIONS (m), (n) and (o) carry the
  arithmetic, and the section below carries the corrected readings. The short
  form: the first is a fit whose partner `ai_model` refuses and whose honest
  interval is +/- 0.048; the second is computed from an alpha `ai_model`
  refuses to invert, and at BLOCK_M = 128 the verdict flips across the
  candidate range; the third is refuted on the ladder slope alone at 89 to 303
  sigma. **Never quote the 207% TEMPO contradiction**: it compares a bound
  with a number.

## What the 2026-09-09 H200 session settled, and what it left

Added 2026-09-09, after the first run of `scripts/h200_gaps_session.sh` on a
rented H200: sixteen arms, 33 minutes of wall clock against a three-hour
booking, everything committed under
`results/published/2026-09-09-nvidia_h200-gaps-session/`. `docs/FINDINGS.md`
has the numbers and the per-arm table; this is what it means for the study's
working state.

**Settled.**

- **STUDY item 3's loose end is CLOSED.** `check_mma_path.sh` at fixed T=256
  and num_warps=4, two arms differing only in BLOCK_M: wgmma (m64n64k16)
  appears exactly where `BLOCK_M % 64 == 0` and nowhere else, BLOCK_M=16 emits
  `mma.sync` only, and the two arms compiled distinct PTX checksums. The
  instruction is selected by the tile height, not the batch or the warp count.
  4 of 4 gates. Item 3 below is struck through accordingly.
- **The anchor is measured, for every published ladder.** The BLOCK_M=32
  anchor rate is 76.2-77.9% of the 4814.3 GB/s pin rate, swizzle-invariant to
  2.28%, and the fitted slope is anchor-independent to 0.31%. The evaluation's
  "weakest link" (an extrapolated memory-branch level) now has a measured n=1
  tread behind it.
- **The DRAM counter route is OPEN on a rented pod**, for the first time: ncu
  2025.1.1 attaches with no permission error (nsys absent). `alpha_b` as a
  NUMBER rather than an interval is now bookable over the alpha-surface cell.
  This is the highest-value open experiment in the study and it should be
  booked next. **This bullet read "at 15 minutes" until 2026-09-10**: that
  figure priced a one-launch recipe the instrument does not run, and the plan
  page now budgets an hour of GPU and two pod-hours end to end for twelve
  profiled invocations. That is the price of ONE BLOCK_N; the traffic-versus-
  time contrast is two of them, so the driver books two arms and 240 minutes.

**Retracted or re-qualified, which is most of what the session bought.**

- **No alpha in this repository may be quoted as a point.** The corpus rescore
  found 4 of 40 published alphas implying more than the card's pin rate, 12 of
  40 outside their own anchor bracket, and a median re-anchoring shift of
  0.094 against the 0.05 those numbers are quoted to. Quote the anchor
  interval. `SURFACE.txt`'s "0 of 12 fits within 0.05 of the pooled 0.558" is
  WITHDRAWN: 4 of 40 brackets contain 0.558. The `BLOCK_M <= 64` cap SURVIVES
  at the bracket's most generous alpha, worst 0.678 of the ridge, which is the
  one load-bearing claim the rescore leaves standing.
- **A crossing inside the ridge band is a band.** The ruler arm reproduces the
  bandwidth patterns to 0.05% across sessions and shows the compute term moves
  5.0x more than the denominator choice, but the 2.2% denominator swing flips
  90 of 53,188 classified rows, all within 6% of the ridge. Crossings inside
  144.9-152.8 FLOP/byte inherit the ruler's bias.
- **The paper's headline configuration still has no confirming arm on sm_90.**
  Both BLOCK_N=256 rooflines REFUSED from register-file arithmetic before
  spending GPU time: the BLOCK_M=256 control needs 65536 of 65536 registers per
  block at every warp and stage count.

**The clock rule changed, and it is a finding rather than a setting.** Over 750
cells the under-load SM clock is set PER TILE by the kernel's own power draw
under the 700 W cap (BLOCK_M=128 median 1395 MHz over 196 cells, BLOCK_M=256 1650,
BLOCK_M=32 1474 at GROUP_SIZE_M=1 and 1740 from GROUP_SIZE_M=8 up,
memory-shaped 1950-1980), and the calibration GEMM's 1485 MHz
at 691 W sits near the LOW end of dense work rather than in the middle. So the
old +/-5% LEVEL band was a rule against a tile: it excluded the study's two
primary tiles from measurability on this card while excluding nothing in the
arms whose tiles happen to sit near the GEMM. From 2026-09-09 a cell is
excluded if and only if its DRIFT verdict failed; the LEVEL side is recorded on
every row and excludes nothing; compute-bound CLAIM gates read the FIXED roof
fraction and print the own-clock fraction beside it as issue efficiency
(`docs/APPARATUS.md` section 1). For the paper, the per-tile clock is itself a
result about power-capped MoE kernels.

**Left open, and what it costs.** Six arms landed INVALID and each named an
apparatus defect rather than a fact about the card: the clock rule
(roofline), a reference the arm could never qualify at its own pairing
(bm128_depth), a vacuity check scaling to the reference instead of the swept
set (bn_g16), a probe that cleared no pinning so P1 was never asked
(alias_ablation), an `r_max` default that made V1 unsatisfiable from the plan
page (cap_test), and a vLLM `override_config` with no try/finally that leaked
one arm's Triton config into 41 others (dtype). `docs/FINDINGS.md` has the
per-arm table. The next session re-runs them with `--new` (an INVALID row is
latched and no resume re-runs it), ~71 priced minutes:

```bash
bash scripts/h200_gaps_session.sh --new \
  --only calibrate,pin_probe-n64-g1,roofline-n64-g1,cap_test,bn_g16,dtype,bm128_depth,alias_ablation
```

`roofline-n64-g1` is expected to reach CLAIM_FAIL, and that is its result: on
the committed cells C3 reads +0.053 against a 0.10 gate and C4 reads 0.552
against a 0.95 gate.

**THAT COMMAND IS SPENT.** All eight of those arms ran on 2026-09-10 and the
booking below the next heading is the current one. The block is left standing
because it is what the session was bought against, and a reader who runs the
line above re-books five arms that now hold a result.

## What the 2026-09-10 H200 session settled, and what it left

Added 2026-09-10, after the second run of `scripts/h200_gaps_session.sh` on a
rented H200: twenty arms, everything committed under
`results/published/2026-09-10-nvidia_h200-gaps-session/` (6 DONE, 5 REFUSED,
5 CLAIM_FAIL, 4 INVALID). `docs/FINDINGS.md` has the numbers, the intervals
and the per-arm reading; this is what it means for the study's working state.
Every figure below was recomputed from the committed cells.

**The estimator is what was wrong, and it is replaced.** `LadderFit.alpha =
B/(A+B)` divides the ladder's slope by a level extrapolated to `n = 0`, a
place no tread was measured. Over the session's 23 ladders five have a
NEGATIVE intercept and five return a value above 1.0 on a quantity that cannot
exceed 1; `alpha_upper = B/(A+B-D)` exceeds 1 exactly when `D > A`, which held
in four of `bn_g16`'s six cells and in no others: arithmetic, not physics.
The replacement has no level in it: `w` = (ms per extra M-tile) / (ms to
stream the layer's whole expert weight set once at the card's measured rate),
which on mixtral bf16 is a 2.8186 GB weight set and **0.6443 ms** per stream.
Both are printed; only `w` may be quoted. `docs/APPARATUS.md` section 4 is the
one-page statement.

**Settled.**

- **The per-M-tile cost, measured in weight streams: 0.68 to 1.37** across 23
  ladders at BLOCK_M 16 to 64, per-repeat sd 0.002 to 0.005 over 17 repeats.
- **95.4% of BLOCK_M=16's wall clock at GROUP_SIZE_M=1 is one full re-read per
  tile**: 89.158 ms measured at n = 132 M-tiles against 85.054 ms of streaming.
  No fit, no anchor, no extrapolation.
- **The activation re-read term is REFUTED, model-free.** Its BLOCK_N
  dependence must double when BLOCK_M doubles; measured it is 1.115 +/- 0.003,
  1.203 +/- 0.007 and 0.923 +/- 0.012 against a required 2.000, at
  z = -303 / -121 / -89. What fits 3.1x better at equal parameter count goes as
  `1/BLOCK_N` with no BLOCK_M in it.
- **GROUP_SIZE_M moves the cost 24%** at a pinned tile, pinned num_stages and
  num_warps and an identical modelled residency, and it replicates across 12
  fresh processes. So `alpha_b` is a function of the SCHEDULE and the model has
  no slot for one.
- **BLOCK_M=16 peaks at 0.099 of the roof against 0.537 for a BLOCK_M=256
  control**, over 162 of 162 cells swept to 132 M-tiles.
- **The register file runs out exactly where arithmetic intensity would
  suffice**: of 56 power-of-two tiles, 17 clear this card's ridge and the
  smallest accumulator among them is 65536 registers against a per-block file
  of 65536. There is no positive control at vLLM's shipped BLOCK_SIZE_N on any
  card this study can reach.
- **The clock is per tile under the power cap**: 1275-1935 MHz, a 1.52x range,
  at a power held within 2% of 700 W in 95% of 2328 cells.

**Retracted, and this is the session's main product.**

- **`alpha_b = 0.9794 +/- 0.0113` is not a measurement.** Its partner
  `alpha_a = -0.8143` is one `ai_model` refuses; its honest interval is
  +/- 0.048 by leave-one-tread-out over the whole chain, and 30.5% of
  D-propagated draws put it outside [0, 1]. **Never quote the 207% TEMPO
  contradiction**: it compares a bound with a number.
- **`cap/ridge = 0.080` is withdrawn as a headline**, being computed from an
  alpha `ai_model` refuses to invert. What stands: for **BLOCK_M <= 64 the cap
  binds under every reading this study has held**; at **BLOCK_M = 128, the tile
  vLLM actually ships, the verdict FLIPS** across the candidate range (0.807 of
  ridge at 0.9794 against 1.293 at 0.5977, threshold 0.784) and is therefore
  NOT ESTABLISHED. And the reachability caveat: under balanced routing every
  shipped bucket at BLOCK_M <= 64 sits at exactly one M-tile per expert.

**Left open, and what it costs.** One experiment decides more than the rest
together, and its route read OPEN for the second rental running (ncu
2025.1.1.0 attached with no permission error, `cap_eff 0xa80425fb`, no
`sys_admin`), so it is bookable rather than aspirational. A DRAM read at ONE
BLOCK_N gives `alpha_b = (dR/dn - a_per_tile)/W`, a traffic slope with no
level, no delta and no assumed bandwidth (today the same six cells return
0.6087, 0.5930 or 0.5143 depending only on which rate is assumed). A read at
TWO BLOCK_N at a fixed BLOCK_M decides the other half, whether the term that
replaces the activation re-read is TRAFFIC or TIME (3.85 GB against 2.06 GB
per M-tile at BLOCK_M=64, or the same bytes at both), and that is a contrast
between two cells, so it is two arms. Second is a THIRD BLOCK_M in the
`bn_g16` grid: the current grid has
two heights that yield a memory branch, which is why every candidate extra
term correlates +0.72 to +0.98 with the activation column and nothing is
identifiable. The next session is `--new`, ~300 priced minutes:

```bash
bash scripts/h200_gaps_session.sh --new \
  --only calibrate,pin_probe-n64-g1,bn_g16,dtype,counter_plan,counter-n32-m64,counter-n128-m64
```

THE COUNTER IS TWO ARMS AND BOTH ARE BOOKED, which is what took that figure
from ~180 to ~300. A DRAM read at ONE BLOCK_N buys `alpha_b` as a traffic
slope and nothing else; the traffic-versus-time contrast is BETWEEN
BLOCK_N=32 and BLOCK_N=128 at the same BLOCK_M=64, so one cell cannot ask it.
`GROUP_SIZE_M` stays pinned at 16 on both, and a G=1 cell is a third arm this
session does not book, which is worth saying because the session measured the
per-M-tile cost moving 24% between G=1 and G=16.

`bn_g16` is expected to reach CLAIM_FAIL again and that is its result; each
counter arm is expected to reach DONE or CLAIM_FAIL and either is the
headline.
Three arms are deliberately NOT in that set: `roofline-n64-g1`,
`alias_ablation` and `noise_floor` each hold an INVALID whose cause is a gate
or an instrument rather than a flag, so re-running them buys the same word for
the same minutes.

## What changed

This started as a kernel project: build a grouped GEMM that beats the incumbent
by attacking the padding tax under skewed routing. **That premise is dead**, and
it was killed by this repo's own measurements. Padding is either zero or free:

- above batch 256, vLLM's autotuner sizes `BLOCK_SIZE_M` to exactly
  rows-per-expert, so padding waste is 0%
- below it, waste hits 50-100% and costs nothing, because 2 us of wasted
  arithmetic hides inside a 20 us weight read

So the project is now a **measurement study**. `moe/kernels/` stays empty unless
something in the results argues for filling it.

## The hypothesis

> The standard mental model of what limits an MoE grouped GEMM is wrong in four
> specific, measurable ways, and every error points the same direction: cost is
> attributed to arithmetic when it belongs to memory.

Each claim below is one probe at that.

## The claims

**C1. The CUTLASS tile is 64, not 128, and it was never a choice.**
`torch.nn.functional.grouped_mm` on Hopper reports `TileShape M,N = 64,128`,
MMA atom `MMA_64x128x16_F32BF16BF16_SS`, schedule `Pingpong`, identical at
T = 1, 16, 256, 1024, 4096. Hopper's `wgmma.mma_async.m64nNk16` has **M fixed at
64** by the instruction set, so no selection was ever happening.
STATUS: **established.** Refutes a figure in three published write-ups,
including two of ours.

**C2. Arithmetic intensity is `2R/b`, independent of expert architecture.**
Every weight element is used once per row (2 FLOPs) and read once (`b` bytes),
so for `N` weight elements across any number of layers, `AI = 2NR / Nb = 2R/b`.
`N` is a SUM over layers and cancels, so shapes and layer counts are irrelevant.
For bf16 that is `AI = rows per expert`.

STATUS: **ESTABLISHED, on a prediction that could have refuted it. Measured
2026-08-28, 19,908 fp8 rows across three kernels, published at
`results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/`.**

THE PREDICTION THAT WAS WRONG WAS MINE, NOT C2'S. The fp8 sweep was built to
test a "2x crossing shift": halve the weight bytes, halve the batch at which a
model crosses its ridge. But the ridge is `peak_FLOPS / bandwidth`, and bf16 ->
fp8 halves `b` AND doubles `peak_FLOPS`, because the same silicon runs fp8
tensor cores at twice the bf16 rate:

    fp8:   2R/1 = 2 * ridge_bf16   ->   R = ridge_bf16
    bf16:  2R/2 =     ridge_bf16   ->   R = ridge_bf16

The same rows per expert, so the same crossing. Both sides of the roofline scale
together and their intersection does not move. The 2x figure came from holding
the ridge at its bf16 value while changing the format.

MEASURED. Crossings are recovered from TIME (`moe/bench/crossing.py`: the slope
of `log ms` against `log T` passing 0.5, floored at the saturation batch), never
from the byte model, so the prediction is refutable.

fp8/bf16 crossing ratio -- corrected theory says 1.00, the retracted 2x says 0.50:

| model            | vLLM | SGLang |
|------------------|-----:|-------:|
| mixtral-8x7b     | 1.25 |   1.16 |
| qwen2-57b-a14b   | 1.11 |   1.15 |
| deepseek-v2-lite | 1.06 |   1.16 |
| deepseek-v3      | 1.07 |   1.23 |

**1.15 +/- 0.07** over eight measurements from two unrelated kernels, which also
agree with EACH OTHER on absolute bf16 crossings to within a few percent (454 vs
464, 810 vs 819, 3240 vs 3048). (Retracted 2026-09-02 as the headline: these
are POOLED over seven routing regimes; uniform-only the bf16 crossings are
313 / 730 / 931 / 2925 and the ratio is 1.131 +/- 0.095, RETRACTIONS above.) The traffic reduction is real and appears in the
TIME rather than the crossing: mixtral at T=512 goes 1.1567 -> 0.6383 ms, 0.55x.

A CONFOUND ON THE 1.15, found 2026-08-28 while rescoping C3. vLLM's tuned
configs pick a DIFFERENT TILE for the two dtypes on the same shape. The mixtral
sweep loaded `E=8,N=14336,dtype=fp8_w8a8`, which sets `BLOCK_SIZE_M = 64` from
M=1, while its bf16 twin sets 16 to 32 at low M. So the fp8 arm ran on taller
tiles than the bf16 arm throughout, and the dtype comparison silently varied the
tile as well.

The direction of the bias is known and matches. A taller tile is 3.1 to 3.6x
faster above the ridge (fewer weight re-reads, `alpha ~ 0.21` per extra tile),
which speeds the compute-bound side and pushes the crossing LATER. Measured
fp8/bf16 is 1.15 against a predicted 1.00, later, so some unknown part of that
0.15 is tile rather than dtype.

Dtype-invariance survives it: the retracted alternative needs 0.50, and no tile
effect of this size closes a gap that wide. But 1.15 is not a pure dtype
measurement, and separating them needs a run with `BLOCK_SIZE_M` pinned equal
across both dtypes, which `override_config` can do and this study has not done.

THE THIRD KERNEL: ITS TIMINGS ARE FIXED, ITS CROSSINGS ARE UNUSABLE.
`torch_scaled_grouped_mm_*` first reported 0.44 +/- 0.13, which looks like
support for the retracted 2x prediction and was an artefact: the span quantised
activations INSIDE the timed region, because `_scaled_grouped_mm` needs both
operands in fp8 while the harness hands out bf16 activations for vLLM's sake.
The giveaway was direct -- deepseek-v2-lite at T=8192 measured 1.9855 ms in fp8
against bf16's 1.0503, and fp8 moves half the weight bytes so it cannot be
slower.

FIXED AND RE-MEASURED (`6652c66`, arm `-fp8-refixed`, 9,408 rows). The same cell
is now 0.7659 ms, 0.73x of bf16, reproducing the smoke's 0.7666 to 0.3%. A 2.59x
change, and it scaled with tokens. The timings from this span are now sound.

BUT THE CROSSINGS FROM IT STILL ARE NOT:

| model            |   up | down |
|------------------|-----:|-----:|
| mixtral-8x7b     | 0.87 | 1.99 |
| qwen2-57b-a14b   | 0.88 | 0.61 |
| deepseek-v2-lite | 0.63 | 0.41 |
| deepseek-v3      |   -- | 0.52 |
| **mean of 7**    | **0.84 +/- 0.53** ||

CORRECTED 2026-08-31. This table previously read `0.52 | 1.23` on the last row
and `0.89 +/- 0.51` for the mean. deepseek-v3's fp8 `up` span has NO crossing:
its slope peaks at 0.497 at T=8192 and never reaches the threshold, under every
filter including `--include-throttled`. The 0.52 is the `down` value, and the
1.23 does not reproduce from either the refixed arm or the superseded one.

The mean moved from 0.44 to 0.84, toward the corrected theory's 1.00, which is
what removing a token-scaling bias should do. But 0.41 to 1.99 is a five-fold
range, against 1.15 +/- 0.07 from the two production kernels. Nothing can be
concluded from a spread that wide.

The likely cause is the method, not the span. A single GEMM's time-against-T
curve is flatter and less structured than a fused layer's, so the slope has less
to cross and the 0.5 threshold lands wherever local noise puts it. `up` and
`down` are the same arithmetic on the same cells and disagree by 2.3x on mixtral,
which is not a property either dtype has.

SO: dtype-invariance rests on vLLM and SGLang. The one-stage span contributes its
TIMES, which is what the five-stage/one-stage separation below uses, and does not
contribute a third crossing measurement. Getting one needs a token grid dense
enough to locate a shallow slope change, which this profile's powers of two are
not.

WHERE THE REMAINING OFFSET LIVES, and it is not the hardware. bf16
measured/predicted, split by how much of the layer the span covers:

| model            |  F/H | vLLM | SGLang | torch_up | torch_down |
|------------------|-----:|-----:|-------:|---------:|-----------:|
| mixtral-8x7b     | 3.50 | 0.71 |   0.72 |     1.46 |       0.64 |
| qwen2-57b-a14b   | 0.71 | 0.63 |   0.64 |     1.00 |       1.18 |
| deepseek-v2-lite | 0.69 | 0.54 |   0.60 |     1.05 |       1.19 |
| deepseek-v3      | 0.29 | 0.63 |   0.59 |     1.26 |       1.27 |
| **mean**         |      | **0.63 +/- 0.06** || **1.13 +/- 0.24** ||

The FIVE-STAGE kernels sit at 0.63 with a tight spread. The ONE-STAGE grouped
GEMM sits at 1.13: `2R/b` predicts it about right. Same hardware, same ridge, so
the offset is not the kernel falling short of datasheet peak -- it belongs to the
extra stages, whose permute, activation and unpermute traffic the weights-only
model never counted.

READ THOSE ABSOLUTES AGAINST THE CALIBRATION THEY WERE SCORED WITH, NOT AS
FIXED. Two calibrations of the same H200 give (the two figures were called a
"ridge band" here until 2026-09-02; retracted, RETRACTIONS (e): they are one
card's compute ceiling failing to reproduce, not a band any card owns, and the
card's own 2026-09-02 calibration gives 162.8):

    bandwidth   4377.2 -> 4374.5 GB/s       0.06% apart
    bf16 GEMM    701.6 ->  770.9 TFLOP/s    9.9% apart
    ridge        160.3 ->  176.2 FLOP/byte  9.9% apart

The bandwidth reproduces; the compute term does not. THE CLOCK IS NOT THE
EXPLANATION, corrected 2026-08-31: across the three H200 calibrations the GEMM
clock moves 20.6% and the achieved rate moves 9.9% the OTHER WAY. 1845 MHz
reached 71.4% of its own clock's peak, 1560 MHz reached 83.2%, 1530 MHz reached
93.2%. Clock normalisation does not collapse the band; the spread lives in
achieved efficiency, and what drives that is not established. So the ridge is a
RANGE, wider than the +/-1.5% this document previously quoted, and the whole
table moves with it: at ridge 176.2 the means are 0.58 and 1.03 rather than 0.63
and 1.13.

WHAT SURVIVES THE BAND is the comparison between span extents. Both sides divide
by the same predicted crossing, so the ridge cancels ALGEBRAICALLY: five-stage
over one-stage is 0.561 at ridge 160.3, at 176.2, and at any other value
(`tests/test_ridge_band.py`). A five-stage span crosses at 56% of the batch a
one-stage span does, and no calibration uncertainty touches that.

AND IT SURVIVES A BETTER BYTE MODEL TOO. The predictions above solve `2R/b`,
which is the weight-dominated LIMIT of the general GEMM intensity

    AI = 2MNK / ((MK + KN + MN) b)   ->   2M/b   when KN dominates

while every measured row is scored with the full model, activations included.
Two models on the two sides of one comparison. `2R/b` overstates AI by ~4% for
mixtral at its crossing and ~7% for deepseek-v3, and overstating AI understates
the batch needed, so every prediction here is systematically low by 5 to 18%.
`crossing_batch_full` solves the same byte model the rows use:

    |                     | five-stage | one-stage | separation |
    |---------------------|-----------:|----------:|-----------:|
    | 2R/b,  ridge 160.3  |       0.63 |      1.13 |      0.561 |
    | full,  ridge 160.3  |       0.58 |      1.03 |      0.563 |
    | full,  ridge 176.2  |       0.52 |      0.92 |      0.563 |

The one-stage span lands within about 10% of prediction under every combination,
0.92 to 1.13, which is the agreement `2R/b` was reaching for. The five-stage span
sits at 0.52 to 0.63 whatever is done to the model. And the separation is 0.563
throughout.

So the claim is the SEPARATION, and the absolutes are quoted with the
calibration they were scored against. (Downgraded 2026-09-01 in
`docs/FINDINGS.md`: the detector reads the first upcrossing of a tile
staircase and 8 of 16 cells cross twice; on the last crossing the separation
is 0.889, not 0.563, and the two spans run different tiles. The 0.563 is
probably an artefact and FINDINGS says why.)

THE fp8 RIDGE RATIO IS MEASURED NOW, AND IT IS NOT 2. The `ridge_fp8 =
2 x ridge_bf16` above is the datasheet relationship. Measured on the H200:

    bf16    770.9 TFLOP/s at 1530 MHz   93.2% of that clock's peak
    fp8    1409.2 TFLOP/s at 1740 MHz   74.9% of that clock's peak

The two GEMMs ran at different clocks, so the 1.828 the calibration records
conflates format with clock. Per clock it is 1.607: fp8 reaches materially less
of its own peak than bf16 does of its.

That makes C2's prediction WORSE, not better, and the honest statement is that
this calibration cannot discriminate:

    ridge ratio 2.000 (datasheet)          predicts crossing_fp8/bf16 = 1.000
    ridge ratio 1.828 (this calibration)   predicts 0.914
    ridge ratio 2.008 (vs the older bf16)  predicts 1.004
    MEASURED, two production kernels                1.150 +/- 0.07

The prediction spans 0.914 to 1.004 across two measurements of one machine,
because the bf16 denominator moves 9.9%. Measured is 1.150. Dtype-invariance
holds against the naive 0.50 by a wide margin either way; the ARITHMETIC is not
pinned down to better than about 10% until the GEMM clock is controlled.

NOT CLAIMED. An earlier reading had the one-stage deviation ordered by expert
shape (`F/H`), matching `ridge.py`'s prediction that mixtral would deviate most.
That ordering came from an input set that double-counted a superseded arm. On
the canonical set it is 1.05, 1.09, 1.12, 1.27 against F/H of 3.50, 0.71, 0.69,
0.29 -- monotonic in the means, but mixtral's internal disagreement makes its
mean unreliable and the effect is too weak to assert. See
`moe/bench/published.py` for why the input set now defends itself.

**C3. Below roughly 100 rows per expert, and in bf16, vLLM emits no warpgroup MMA.**
Hopper's `wgmma.mma_async.m64nNk16` has M fixed at 64, so any tile shorter than
that runs on `mma.sync`, the Ampere-era instruction. The question is when vLLM's
chosen tile is shorter than 64, and what it costs when it is not.

STATUS: **ESTABLISHED, measured 2026-08-27/28, and RESCOPED twice since.**

THE MEASUREMENT. `scripts/check_mma_path.sh` on a real deepseek-v3 T=16 cell:
both compiled `fused_moe_kernel` variants show `wgmma=0`, `mma.sync=16`, and
every one of the 32 tensor-core instructions is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Forcing `BLOCK_M >= 64`
does reach `wgmma.mma_async.sync.aligned.m64n32k16` and `m64n64k16`, and is
1.7 to 9% SLOWER: the capability is there and declining it is correct, because
the tensor core idles waiting on weights either way while a short tile buys
occupancy.

FIRST RESCOPE: IT IS NOT ABOUT "DECODE". The kernel sees M, the rows entering
the layer, and cannot tell whether they came from one prefill or from a thousand
concurrent decodes. A serving system with enough concurrency is in decode AND at
large M simultaneously, and there the tuned config picks a taller tile and does
emit wgmma. The claim is about a REGIME, in rows per expert, which batching can
leave. Reading the tuned H200 configs: `E=1,N=3072` steps 16 -> 32 -> 64 -> 128
by M=128, while `E=128,N=1024` stays on 16 until M=1536. That spread is the
`E/k` dilution appearing in someone else's grid search, since a constant
rows-per-expert threshold means `M_switch` scales with `E/k`.

SECOND RESCOPE: IT IS bf16-SPECIFIC. The same shapes tuned for fp8 pick a
warpgroup tile at M=1:

    E=8,N=14336   fp8   1:64, 2:64, 4:64, 8:64, 16:64 ...
    E=8,N=14336   bf16  1:16, 2:32, 4:16, 8:16, 16:16 ...
    E=64,N=2560   fp8   1:64, 2:64, 4:64 ...
    E=64,N=2560   bf16  1:16, 2:16, 4:16 ...

So at fp8 decode vLLM reaches wgmma immediately, and C3 describes the bf16 path.

AND THE MEASURED CELL WAS RUNNING A FALLBACK, NOT A TUNED CONFIG. deepseek-v3 is
`E=256,N=2048`, and no tuned H200 config exists for it: the run log prints
`Using default MoE config. Performance might be sub-optimal!`. So the 16 in that
PTX comes from `get_default_config`'s hardcoded small-M branch rather than from
a grid search. A GPU MODE reader raised exactly this, and was right. It does not
weaken the measurement, since a fallback is what deepseek-v3 actually runs, but
it changes what the 16 is EVIDENCE of: a default, not a tuned optimum.

**C4. STREAM-style calibration may understate achievable read bandwidth.**
A production kernel sustained 4483.4 GB/s where `calibrate_hardware.py` reaches
4389.4 GB/s on a pure read. If the calibration is measuring its own achieved rate
rather than a ceiling, every percent-of-ceiling figure computed that way is
pessimistic, ours and everyone else's.
STATUS: **CONFIRMED and closed, 2026-08-28.** The cause was the SHAPE of the
read, not the clock. `calibrate.py` measured it as `torch.sum(a, dim=0)` on a 1-D
buffer into a scalar: a full tree reduction, which bounds on ATen's reduction
rather than on DRAM. Reducing a 2-D view along the contiguous axis instead gives
thousands of independent reductions and no global combine.

Measured on the same card before and after the fix:

    read ceiling   4389.3  ->  4470.7 GB/s     +1.85%

That closes the anomaly. The 83 rows implied 4483.4 GB/s, which is 102.4% of the
named `triad` ceiling and looked impossible, but is **100.28% of the corrected
read ceiling**: at the ceiling within three parts in a thousand, not above it.
Those kernels were running at essentially 100% of achievable read bandwidth on
pure weight streaming, which is a strong result rather than a broken one.

A clock hypothesis was tested first and REFUTED. Settling under a memory load
instead of a matmul is correct in itself, and the memory settle converges at
1980 MHz as designed, but it moved triad by +0.05% and read by -0.00%: the
existing two-pass warmup had already handled the clock.

Confirmed independently on an A100, where the flaw is unmissable: the same
benchmark reports read at 1770 GB/s against triad's 1798, and triad moves three
times the bytes. `calibrate.py` detects that case and refuses to name read as a
ceiling. On the H200 it hid, because read landed just above triad.

**Caveat on the ridge.** Bandwidth is stable to 0.005% across three
calibrations (4377.0 / 4377.0 / 4377.2). The GEMM is not: it lands at 1560 or
1845 MHz depending on the run, giving 701.6 or 712.4 TFLOP/s and a ridge of
160.3 or 162.8. (Corrected 2026-08-31 in FINDINGS and retracted here 2026-09-02:
the clock is NOT the explanation, the achieved rate moves the other way from
the clock, the spread across all calibrations is 9.9-12%, not +/-1.5%, and
none of it is a band the card owns. The ridge to quote is the one of the
calibration a row was scored against; the card's own 2026-09-02 calibration
gives 162.8.) Mixtral's predicted crossing spans
641 to 651 across that range, all inside the same measured bracket, so C2 is
unaffected.

**C5. Does the crossing scale with the ridge across architectures?**
`AI = 2R/b` says the crossing is at a fixed rows-per-expert set by the ridge, so
two cards with different ridges should cross at rows-per-expert in the same
proportion. H200 ridge 176.2, A100 ridge 145.7. (The whole section below is
RETRACTED as scored, 2026-08-31, and superseded by `docs/FINDINGS.md` C5: the
table pools seven routing regimes, the H200 arm is not entitled to 176.2, and
the two cards ran different kernels. Kept as the history of the claim.)

STATUS: **PARTIAL, measured 2026-08-28, RESCORED 2026-08-31.** Same profile, same
kernel, one run per card, `vllm_fused_experts` bf16.

THE TARGET IS 0.83, NOT 1.00, AND THIS SECTION PREVIOUSLY USED 1.00. For bf16
`b = 2`, so `2R/b = ridge` puts the crossing at `R = ridge` rows per expert, a
DIFFERENT R on each card. The ratio the two cards should show is
145.7 / 176.2 = 0.83, or 0.91 if the H200 ridge is taken at the low end of its
band. A measured ratio of 1.00 means the two cards crossed at the same rows per
expert, which is what NO ridge scaling looks like.

| model            |   E | A100 | H200 | ratio | vs target 0.83 |
|------------------|----:|-----:|-----:|------:|---------------:|
| mixtral-8x7b     |   8 |   58 |  136 |  0.43 |           0.52 |
| qwen2-57b-a14b   |  64 |   81 |  114 |  0.71 |           0.86 |
| deepseek-v2-lite |  64 |   74 |   84 |  0.88 |       **1.06** |
| deepseek-v3      | 256 |  110 |  109 |  1.01 |           1.22 |

So deepseek-v2-lite is the model that scales with the ridge, within 6% and inside
the band. deepseek-v3 overshoots by 22%. This section previously reported
deepseek-v3 as agreeing "to 1% across two architectures, three years apart",
which was agreement with the null. No model confirms cleanly and none refutes by
an order of magnitude.

WHAT SURVIVES (retracted 2026-08-31: it does not; under uniform routing the
scores are 0.88 / 1.14 / 1.18 / 1.14, not monotonic, a pooling artefact,
FINDINGS C5): the deviation is monotonic in EXPERT COUNT either way, 0.52, 0.86,
1.06, 1.22 against E of 8, 64, 64, 256, and expert count is not a term in the
model. Correcting the target moves the deviation from "approaches agreement from
below" to "crosses agreement between 64 and 256 experts" without changing its
ordering, and the ordering is what needs explaining.

THE HYPOTHESIS, STATED AS UNTESTED. Expert count sets how many thread blocks a
launch has. With 8 experts there may not be enough to fill 108 SMs (A100) or 132
(H200) until T is large, so the time-against-T curve is shaped by an occupancy
ramp rather than by the roofline, and what the detector finds is not the ridge.
With 256 experts there are always enough blocks. The A100 has fewer SMs, so any
block-count effect should hit it harder, which is the direction observed. Four
points and a plausible story are not evidence, and this is recorded as a
hypothesis with an experiment attached: sweep expert count at FIXED
rows-per-expert and see whether the deviation follows blocks or follows E.

WHAT DOES TRANSFER. The five-stage offset is a property of the fused span on
both cards: measured/predicted is 0.55 +/- 0.15 on the A100 against 0.63 +/- 0.12
on the H200. The means differ by 13% and the spreads overlap heavily, so read it
as consistent across two machines rather than as the same number. The 13% is the
C5 discrepancy appearing in another coordinate.

## What a whole MoE layer costs, and how much of it is routing

MEASURED 2026-08-28, `results/published/2026-08-28-nvidia_h200-h200-whole-layer/`,
9,408 rows. The first complete-layer measurement in this project: every framework
span covers five of six stages and leaves the router out, so until now the study
could not say what a full layer costs. `__pipeline__:vllm_fused_experts` times
`ref_router` plus the fused kernel as one cell.

THE ROUTER IS A FIXED COST, and at decode it is a third of the layer:

| model            | T=1 layer | router | share | T=4096 share |
|------------------|----------:|-------:|------:|-------------:|
| mixtral-8x7b     |    0.2707 | 0.0814 | 30.1% |         1.7% |
| qwen2-57b-a14b   |    0.2019 | 0.0773 | 38.3% |         3.4% |
| deepseek-v2-lite |    0.1591 | 0.0762 | 47.9% |         7.1% |
| deepseek-v3      |    0.2867 | 0.0974 | 34.0% |         4.9% |

The absolute cost barely moves with batch -- roughly 0.05 to 0.10 ms everywhere,
which is launch and dispatch overhead for a matmul and a top-k, not work. So its
SHARE collapses as the batch grows while the number itself does not. At T=1,
between 30% and 48% of an MoE layer is deciding which experts to use.

That is the same story the rest of the study tells from the other end: at decode
nothing is FLOP-bound, and what dominates is whatever does not scale.

CAVEAT, and it bounds the claim. This is `ref_router`: a PyTorch matmul plus a
top-k, the harness's reference. A production router is fused and faster, so 30
to 48% is an UPPER bound on the share, not a measurement of what vLLM spends. It
does establish that a whole-layer number is not the fused span's number, and how
much is missing.

THE CROSSING IS UNMOVED, which is the confirmation. Same run, ridge 176.2
(retracted 2026-09-01: this arm's own rows carry 160.3 and `entitled_ridge`
refuses 176.2, and the crossings below are pooled over routing; re-scored at
160.3 on uniform routing in `docs/FINDINGS.md`, where the conclusion survives
with predicted 641 / 1282 / 1710 / 5130 against 316 / 787 / 931 / 3010):

| model            | predicted | span | whole layer |
|------------------|----------:|-----:|------------:|
| mixtral-8x7b     |       705 |  543 |         549 |
| qwen2-57b-a14b   |      1410 |  914 |         960 |
| deepseek-v2-lite |      1879 |  897 |        1006 |
| deepseek-v3      |      5638 | 3474 |        3375 |

One to twelve percent apart, mostly under five. A fixed cost added to a
bandwidth-driven turning point should not move it, and it does not. The
five-stage ratio here (0.63 mean) also reproduces the 0.58 to 0.63 seen in the
earlier sweeps, which is the run-to-run spread the ridge band predicts.

## Two analysis bugs that had to be fixed before any of this was readable

Both produced confident, wrong numbers rather than errors, which is the failure
mode this project is most exposed to.

**A row that was never timed is not a measurement of zero.** A skipped or
uncapturable graph mode still writes a row, with `ms_p50` left at its 0.0
default; `run_all.sh --dry-run` says so in as many words. Feeding those to a
median dragged it toward zero and the first fp8 report concluded deepseek-v3
crossed at 2 tokens. 2,356 of 11,264 rows were untimed. `crossing.timed_rows`
now drops them and the report states how many, rather than silently using fewer.

**Below `E/k`, a batch does not touch every expert.** mixtral at T=1 reaches 2
of 8, so active experts and weight traffic grow WITH the batch and time rises
nearly linearly. That slope crosses 0.5 for a reason unrelated to the ridge, and
the detector stopped there: mixtral reported 5 tokens, deepseek-v3 reported 25.
`2R/b` assumes all E experts are active, so those points are outside the claim's
domain rather than evidence against it. `crossing_from_points` now floors at the
saturation batch, which `ridge.saturation_batch` already computed.

## Supporting results, already measured

- **Span extent is a trap.** `grouped_mm` covers 1 of 6 canonical stages;
  `fused_experts` covers 5. Comparing their milliseconds compares a GEMM to a
  fused block. Recorded per row in `covers`, enforced by `scripts/compare.py`.
- **Distance from the compulsory byte floor**, L2-cold eager unthrottled
  ("unthrottled" by the retired idle-instant flag, RETRACTIONS (f)),
  n=3225: vLLM 1.16x, SGLang 1.17x, torch `grouped_mm` 1.62x, reference 12.43x.
- **Bimodality is real but cheap.** At deepseek T=4096 zipf:2.0, 24 of 248 active
  experts hold 89% of the rows. One global tile costs only 1.00x-1.18x of ideal
  weight traffic, so per-expert tiling is a 5-15% target, not a 2x one.
- **83 rows report the impossible.** All vLLM, all deepseek-v3, all T in
  {16,32,64}, 56 unthrottled. Peak 4483.4 GB/s = 102.1% of measured read, and
  **zero rows anywhere exceed the pin rate**. Explanation C4.
- **Routing is not reproducible off the GPU.** `cli.build_routing_source` passes
  `device=args.device`, so a GPU run samples with a CUDA generator. Same seed on
  CPU gives a different expert assignment. Row totals match, distributions do
  not. NOT fixed; fixing it changes future routing relative to published rows.

## The two additions that make it a study rather than a report

The sweep is 3 models x 14 token counts x 7 routings x 3 seeds, and **1 dtype,
1 device**. Both degenerate axes are where the value is.

**A. fp8 -- a prediction test.** C2 says `AI = 2R/b`. Halving `b` must double
arithmetic intensity and halve the ridge crossing: deepseek-v3 from ~5,500 tokens
to ~2,750. That is falsifiable, quantitative, and derived before being measured.
Needs the baselines to declare and support fp8; `spec.py` already knows
`fp8_e4m3` and `fp8_e5m2` at 1 byte and the byte model is already
dtype-parametric.

**B. A second device -- a generalisation test.** The crossing scales with the
ridge. An A100's ridge is 153.02 Op/B against the H200's 206.15 (Yun et al.
Table I). Running the same sweep on an A100 and checking the crossings move as
predicted is the cheapest possible falsification test, and the device-agnostic
work is already done and unused.

## What runs where

| piece | where | what it does |
|---|---|---|
| `scripts/run_all.sh` | pod | the whole session: tests, calibration, smoke, sweep, plots |
| `scripts/calibrate_hardware.py` | pod | measures this card's real bandwidth and bf16 rate |
| `scripts/calibrate_read_variants.py` | pod | **C4**: naive vs vectorised read, is our ceiling too low |
| `scripts/check_mma_path.sh` | pod | **C3**: dumps PTX, greps wgmma against mma.sync |
| `scripts/kernel_name.py` | pod | **C1**: reads the CUTLASS tile out of the profiler |
| `scripts/tile_sweep.py` | pod | **C3**: forces BLOCK_M past 64 and dumps the PTX it emits |
| `scripts/compare.py` | anywhere | span-aware comparison, refuses to hide the extent |
| `scripts/crossing_report.py` | anywhere | **C2/C5**: reads the crossing off measured TIME |
| `scripts/efficiency_report.py` | anywhere | is the crossing offset achieved-versus-peak? |
| `scripts/recompute_ceilings.py` | anywhere | re-derives a published arm's ceiling columns |
| `scripts/plot.py` | anywhere | figures, one set per dtype present in the rows |
| `scripts/publish_results.sh` | pod | commits a result set back to the repo |
| `scripts/setup_runpod.sh` | pod | builds the venvs and reports what the card is |
| `scripts/sweep_progress.py` | pod | how far a running sweep has got, and its real rate |
| `scripts/capture_traces.py` | pod | real routing distributions from a real MoE model |
| `scripts/probe_baseline_api.py` | pod | what MoE entry points this venv's framework exposes |
| `scripts/probe_baseline_types.py` | pod | and the argument types of the one it exposes |
| `scripts/profile_open_questions.sh` | pod | the two questions the first sweep could not answer |
| `scripts/preflight_cutile.py` | pod | whether cuTile is worth more of this pod's time |
| `moe/bench/ridge.py` | anywhere | predicts the crossing per model from a calibration |
| `moe/bench/published.py` | anywhere | which published arms an analysis should read |
| `moe/bench/timing.py` | pod | THE instrument, `time_kernel`; LEVEL, DRIFT and host-bound verdicts per cell (`docs/APPARATUS.md`) |
| `moe/bench/exit_codes.py` | anywhere | the one exit-code table and the `RESULT:` line every arm prints |
| `moe/bench/provenance.py` | anywhere | the provenance block and the run-id rule |
| `moe/bench/ai_model.py` | anywhere | the byte model with both re-reads, and what a ladder fit returns over it |
| `scripts/h200_gaps_session.sh` | pod | every open experiment as one resumable session with a ledger (`docs/POD_RUNBOOK.md`) |
| `tests/` | anywhere | all green off-GPU; the count moves, `README.md` quotes it and `tests/test_docs.py` checks it |

## Order of work

1. ~~`check_mma_path.sh`~~ DONE 2026-08-27. C3 established.
2. ~~`calibrate_read_variants.py`~~ DONE 2026-08-27. C4 resolved: the anomaly was
   the ruler. Follow-up, not yet done: settle clocks before the bandwidth
   patterns in `calibrate.py`, and stop naming a tree reduction as the read
   ceiling. Both change the ruler every published number used, so they are a
   deliberate re-baseline rather than a patch.
3. ~~Tile sweep~~ DONE 2026-08-27. Forcing BLOCK_SIZE_M never helps: 16->32 is
   flat (padded MACs are free, MEASURED not modelled), 64 is 1.7-9% slower even
   though it reaches WGMMA, 128 is 27-30% slower. Padded arithmetic hides while
   it is ~20% of the memory time and costs above 40%. Occupancy confound not
   separated, and it does not need to be: the hypothesis was that bigger would
   help. ~~Loose end: confirm the instruction actually switched by re-running
   check_mma_path.sh under the override.~~ CLOSED 2026-09-09 by the
   `mma_switch` arm of the H200 gaps session, which ran exactly that command at
   two BLOCK_M values with T, warps, stages and BLOCK_N held: wgmma
   (m64n64k16) appears exactly where `BLOCK_M % 64 == 0` and nowhere else,
   BLOCK_M=16 emits `mma.sync` only, distinct PTX checksums per arm, 4 of 4
   gates.
4. The ablation, using the GPU MODE method: alias B by taking the tile offset
   modulo so every iteration reloads the same tile (loads execute, L2 hits, no
   HBM traffic, nothing folds since values are runtime); `acc += tl.sum(b) +
   tl.sum(a)` to keep loads live on the compute side. Settles critical path
   without relying on the byte model, which C4 puts under suspicion.
4. fp8 baselines, then re-sweep. Tests C2's prediction.
5. A100 hour, same sweep. Tests C2's generality.
6. ~~Rewrite FINDINGS around C1-C4~~ DONE 2026-08-31, around C1-C5, at
   `docs/FINDINGS.md`. Study-level rather than arm-level: the arm-local
   FINDINGS in the superseded three-way directory is now banner-marked as
   historical. Every table in the new file names the command that regenerates
   it, and every number in it was recomputed from the rows rather than carried
   forward, which is how the two corrections above were found.

## Standing scope limit

Every claim is about **one GPU holding every expert**. It is not a claim about
how frontier MoE is served: DeepSeek runs decode on DP144+EP144 precisely to
scale the aggregate batch past this ridge, and at that scale all-to-all
communication dominates, not the GEMM. Half the corrections in this project came
from stating single-node results as universal. Do not do it again.
