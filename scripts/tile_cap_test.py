#!/usr/bin/env python
"""Does the AI cap formula hold at BLOCK_SIZE_M=16? A FORMULA test, not a claim.

    python scripts/tile_cap_test.py --dry-run           # plan, predictions, MDE, cost. No GPU
    python scripts/tile_cap_test.py --self-test 0.558   # the refit world, verdicts asserted
    python scripts/tile_cap_test.py --self-test 0.10    # the retracted world, verdicts asserted
    python scripts/tile_cap_test.py --self-test 0.14    # the only world where C2 can FAIL
    python scripts/tile_cap_test.py --plant-noise 0.02 --self-test 0.558   # at a spread you name
    python scripts/tile_cap_test.py                     # the pod run, ~80 s of H200
    python scripts/tile_cap_test.py --control 128 --num-stages 3   # A100-safe control

WHAT THIS TESTS. The ceiling an M-tile height puts on arithmetic intensity, on
the one ladder in this study where `alpha` is cleanly identifiable. The study
writes that ceiling as `cap = 2 BM / (alpha b)`, and READ FROM A LADDER FIT that
expression is NOT the ceiling: `LadderFit.alpha` is B/(A+B), which
`moe/bench/ai_model.py` shows returns `(alpha_b + phi) / (1 + phi + delta)`
(EXA) on the three-term byte ladder, so `2 BM / (alpha_fitted b)` is the exact
ceiling times `1 + phi + delta`. That factor is a BRACKET here and not a number,
because `alpha_a` (the activation-side miss fraction inside `phi`) has no
measurement anywhere in this repository; C2 prints the bracket beside the cap
and scores the LIN number, which is the HIGHER one and so the harder bar for a
claim that the ceiling is low. A tile that is memory bound at EVERY tread has no
compute branch for the fit to run into and imports no alpha from another block
size, and BLOCK_M=16 is the extreme case of that. Across all 26 published
reports the FORCED block sizes are 32, 64, 128 and 256 and nothing else, so the
formula has never been exercised at the bottom of its range, and that is what
this buys. The experiment is worth the 80 seconds.

WHAT IT DOES NOT CARRY, AND AN EARLIER VERSION OF THIS HEADER SAID IT DID. The
production claim. It is BLOCK_M=128 that carries that, and the demotion is not a
judgement call -- it is the observed-tile data.

THE OBSERVED-TILE DATA, ON BOTH BASES. `results/published/2026-09-01-nvidia_h200-
alpha-0558/merged.csv` is the one published arm that RECORDS the tile vLLM chose
(`tile_block_m`, with `tile_config_source` in {vllm_default, vllm_tuned}). A cell
is one (model, token count) under UNIFORM routing, 132 cells of 7 routing seeds
each, 924 seed-rows. Two readings of rows per expert exist and they do not agree:

    BLOCK_M     load_mean_rows              load_max_rows (what production pads to)
              multi-tile   max tiles     multi-tile cells   max tiles   seed-rows
       16       0 of 24        1             1 of 24            2        1 of 168
       32       0 of  5        1             0 of  5            1        0 of  35
       64       0 of 16        1             2 of 16            2        5 of 112
      128      59 of 87       32            66 of 87           34      455 of 609

`load_max_rows` IS THE ONE PRODUCTION SEES: `moe_align_block_size` pads every
expert to its own row count, so the launch grid holds the BUSIEST expert's tile
count and the mean is a number about the routing histogram, not about what ran.
A cell is counted multi-tile on that basis when ANY of its seven seeds needed a
second tile; at the median seed the 128 line reads 65 of 87, which is the figure
`scripts/h200_gaps_session.sh` carries. An earlier version of this table gave
only the mean-rows column and read it as "16, 32 and 64 never run multi-tile",
which the padded count refutes in isolated cells: BLOCK_M=16 reaches 2 tiles in
one cell (qwen2 at T=64, one seed, busiest expert 18 rows) and BLOCK_M=64 in two.
`OBSERVED_MULTI_TILE_MAX_ROWS` and `..._MEAN_ROWS` below carry both columns and
tests/test_tile_cap.py recounts both from the CSV.

WHY THAT STILL RETIRES THE PRODUCTION FRAMING AT THIS TILE. The re-read term is
`Q(n) = 1 + alpha (n - 1)`, which is exactly 1 at n = 1, so it only exists above
one tile per expert. vLLM does pick 16 at small batch -- `get_default_config`
returns it for M <= 32, and 24 of the 132 cells above took that ladder -- and in
23 of those 24 the busiest expert holds ONE tile; in the 24th it holds two, once
in seven seeds, so Q reaches 1 + alpha in 1 of 168 seed-rows and 1 in the rest.
The cap at 16 is therefore real and reached for in ISOLATED CELLS, never as a
regime, and a cap that binds in one seed-row of 168 is a fact about the formula,
not about a shipped kernel. The same holds at 32 (never) and 64 (5 of 112).

BLOCK_M=128 IS THE PRODUCTION REGIME, and `scripts/bm128_depth.py` is where its
claim lives. It is the only tile height vLLM runs multi-tile as a regime, up to
34 M-tiles per expert on the padded count (32 on mean rows), and the only one
whose cap and the ridge are close enough for the answer to be in doubt. The caps
that file used to print, 150.4 against a calibrated ridge of 145.8 on the A100
and 158.6 against 162.8 on the H200, were LIN caps, and that straddle DOES NOT
EXIST: read through (EXA) at the fused layer's own `phi` they are 135.4 and
130.7, both UPPER bounds (delta taken as zero) and both below their own card's
ridge. Whether the cap matters in production is settled there, not here.

THE SENTENCE THIS FILE WAS BUILT TO EARN, and the half of it that survives.
`docs/FINDINGS.md` ("Three readouts from one sweep", which then lists four -- the
miscount is why the fourth never ran) says:

    AND THE CAP: force BLOCK_M = 16 and sweep T as far as the grid allows. The
    formula says no crossing exists. If one appears, alpha < 0.0998 and the cap
    is real but higher than assumed. If none appears, a decode-tuned MoE kernel
    is structurally incapable of reaching its compute roof.

The first two sentences are the formula test and this script runs them. The
third is the production reading, and the table above retires it AT THIS TILE:
nothing about a shipped kernel follows from where the ceiling of a tile height
sits when that tile height never reaches for it.

WHY THIS IS NOT A ONE-TILE SWEEP, which is the whole design question.
`scripts/block_m_crossing_sweep.py --tiles 16 --r-max 4096` already runs the
words above and already prints `BLOCK_M= 16 cap 28.7 NO CROSSING EVER` (verified
off GPU, 2026-09-01). Its five gates then say almost nothing, because three of
them are comparisons ACROSS tiles:

  * GATE 2 compares `min(block_sizes)` with `max(block_sizes)`. With one tile
    those are the same setting, the ratio is exactly 1.000 against a 1.50 gate,
    and the FAIL is arithmetic, not evidence. Run and confirmed, 2026-09-01:
    `ms(BM=16) / ms(BM=16) = 1.000x`.
  * GATE 3 asks whether the fitted re-read fraction's interval overlaps the
    pre-registered `ALPHA_BAND` in BOTH directions (its retired one-sided 0.33
    threshold is printed beside the verdict, not scored), and on a one-tile
    run it goes UNDECIDED with "No ladder had two memory-bound treads,
    so no block size measured the re-read fraction". That is not a shortage of
    treads: BLOCK_M=16 is memory bound at EVERY tread and is the best alpha
    estimator in the study. It is that the compute branch membership is decided
    against a REFERENCE ladder, and with one block size the only candidate
    reference is the ladder being classified. A second, larger tile is what
    makes alpha at 16 identifiable at all, which is C2 here.
  * GATE 4 IS the cap claim, and it refuses to score without a POSITIVE
    CONTROL: some other block size that reached the roof inside the same grid.
    One tile has no other tile, so `Bracketing.positive_control` is None,
    `sufficient` is False, and the gate goes UNDECIDED. That refusal is right.
    An absence recorded by an instrument never shown to detect a presence is
    not evidence of absence.

So the fourth readout as written is not runnable, and this script runs
BLOCK_M=16 alongside ONE larger tile as the control. It is a thin runner: the
grid, the timing, the resume, the compile assay, the ladder fit and three of the
gates are `block_m_crossing_sweep`'s, imported and called rather than
reimplemented. What is new here is the pairing, the depth argument, the split
between validity and claim, and the two cap gates.

WHAT THE SIBLING'S GATE 4 STILL CANNOT DO, and why this is not a duplicate of
it. Its bracketing horizon is `2 x` the crossing the retracted alpha predicts
for the block size under test. At BLOCK_M=64 that is a real number. At
BLOCK_M=16 the retracted alpha predicts NO crossing either -- its ceiling is
160.0 against the H200's own calibrated ridge of 162.8 (the 160.3 the module
band starts at is no card's calibration; on the A100's 145.8 that same ceiling
sits ABOVE the ridge) -- so `crossing_rows` is None, the horizon
collapses to `2 x 0`, and every depth clears it. The gate that decides whether
an absence is bracketed would pass after a single tread. `retracted_horizon_
tiles` below asks the question that threshold is actually about instead.

WHICH CONTROL, AND WHY 256. At alpha=0.558 only 128 and 256 have an AI cap above
the ridge, so only they can cross and only they can be a control. 256's crossing
lands in tread 1, where `Q = 1`, at BOTH ends of the ridge band AND under the
retracted alpha -- r = 160.3 and 176.2 rows per expert, both inside one 256-row
tile, and so is the H200's own 162.8 and the A100's 145.8 (the band is the one
`--self-test` pins; see requirement 3 below for why it is no card's own). Its
crossing therefore does not depend on the parameter under test. 128's lands in
tread 2 at ridge 160.3 or 162.8 and tread 3 at 176.2, moving with alpha, which
makes a control whose own behaviour is part of the argument.

The price of 256 is shared memory, and it is NOT the number an earlier draft of
this header carried. 4 stages x (256x64 + 64x64) x 2 B is 163,840 bytes, which
is 160 KiB against sm_90's 232,448 and sm_80's 166,912 -- it fits on the A100
too, by 3 KiB. The draft said "past the A100's 164 KB", from mixing KB with KiB
on a constant nobody recomputed, which is this project's failure mode 6 in one
sentence. The plan now PRINTS the bill from `tile_resources` for both tiles and
REFUSES before a pod is rented if either cannot run as pinned, so no version of
that sentence has to be trusted. `--control 128 --num-stages 3` remains
available and is a WEAKER control, not an equal one.

HOW DEEP IS DEEP ENOUGH, and why the default `--r-max` is derived rather than
chosen. Three requirements; the third is binding and it is the only one that
makes a PASS mean anything.

  1. THE FIT NEEDS TREADS. `MIN_MEMORY_TREADS = 3` is the floor
     `block_m_crossing_sweep` sets before an alpha may decide a verdict (two
     points make a line with no residual). BLOCK_M=16 hands one tread per 16
     rows and never leaves the memory branch, so this is free.
  2. THE MODELLED AI MUST HAVE FINISHED RISING. `AI(n)/cap = alpha n /
     (1 + alpha (n-1))`, so 95% of the ceiling needs `n >= 19 (1-alpha)/alpha`:
     16 treads at alpha=0.558, 17 at the band's low end 0.529.
  3. THE RETRACTED WORLD MUST HAVE BEEN GIVEN ITS CHANCE TO TRIP GATE C1, and
     C1 asks two things with two different horizons. Both are fractions of
     `ridge x bandwidth`. At alpha=0.10 the cap tile's ceiling is 160.0 Op/B
     against a ridge of 160.3 to 176.2 -- the band `--self-test` PINS, which is
     the sibling module's constant and no card's own calibration (H200 162.8,
     A100 145.8; a measured run resolves the attached card's and the two
     horizons move with it) -- so that world does eventually trip both, and
     the depths at which it does are:

         condition                    threshold   ridge 160.3   ridge 176.2
         discriminating (midpoint)      0.589      13 tiles      13 tiles
         near the roof                  0.850      52 tiles     132 tiles

     The near-roof horizon at the band's worst end binds, so the default r_max
     is 132 x 16 = 2112 rows per expert (T = 8448 on mixtral, about 80 s of
     H200), derived at run time from `RETRACTED_ALPHA` and `RIDGE_BAND` and
     printed with its derivation.

     THE THRESHOLD AND THE HORIZON ARE COUPLED ON PURPOSE, and that is what
     makes the trap structurally impossible rather than merely watched for: a
     condition is LIVE only at a depth where the retracted world would have
     failed it. A shallower run scores the conditions that are live and reports
     the others as NOT TESTABLE, so "BLOCK_M=16 never got near the roof" cannot
     be earned by stopping early -- at 26 treads the retracted world would not
     have got near it either.

WHAT NO DEPTH CAN RULE OUT, stated here rather than left implied. The family
`alpha < 2 BM / (b ridge) = 0.0998` puts a REAL crossing at BLOCK_M=16, and
pushes it arbitrarily deep as alpha approaches that value from below. No finite
sweep excludes it. What excludes it is MEASURING alpha, which is gate C2, on the
one ladder in this study that is memory bound at every tread.

THE TWO WORLDS, computed from the model at import and reprinted before the run
so nothing here can be adjusted after seeing data:

                                          alpha=0.558      alpha=0.10
      AI cap at BLOCK_M=16                  28.7 Op/B       160.0 Op/B
      cap / ridge 160.3                     0.179           0.998
      cap / ridge 176.2                     0.163           0.908
      peak of ridge x bandwidth at r=2112   0.176           0.893   <- C1
      ms(BM=16)/ms(BM=256) at r=2048        5.672x          1.119x  <- C3, gate 1.50
      a crossing at BLOCK_M=16              none            none    <- NOT a readout

C1's thresholds are 0.85 (near the roof) and 0.589 (the midpoint of the two
predicted ceilings at ridge 160.3), and the measured 0.176 above is the same
number the sibling's own gate 4 reports for this tile, because both are taken
against `ridge x bandwidth` and not against the sweep's own best.

THE LAST ROW IS THE TRAP IN THE ORIGINAL WORDING. At alpha=0.10 the cap is 160.0
and the low ridge is 160.3, so the retracted world ALSO predicts no crossing, by
0.3 Op/B. "Did a crossing appear" therefore does not separate the two worlds at
all; how CLOSE the tile gets does. That is why C1 is a roof FRACTION against a
threshold and not a yes/no, and why the depth in requirement 3 is what the whole
run is buying.

WHY THE MEASURED CAP IS BIASED, IN WHICH DIRECTION, AND BY HOW MUCH. This used
to be two sentences of sign argument with no number in them, and the number is
available. `LadderFit.alpha` is `B/(A+B)` fitted on raw times, which is the LIN
blend, and `moe.bench.ai_model` names what that blend actually estimates:

    alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                     (EXA)

`alpha_b` is the weight-re-read fraction the cap formula wants. `phi` is the
ACTIVATION re-read an extra M-tile also pays, in units of one weight read, and
`delta` is the fixed per-call bytes in the same units (`ai_model.phi`,
`ai_model.decompose`). Both are positive, so at `alpha_b < 1` the blend pulls
alpha_fitted TOWARD one and the direction of the bias is not the same at every
alpha; what is invariant is that `cap = 2 BM / (alpha b)` is decreasing in
alpha, so an alpha read through the LIN identity as if it were `alpha_b`
overstates the cap by exactly `1 + phi + delta`. `ai_model.lin_overstatement`
returns that factor and C2 PRINTS IT BESIDE THE CAP: at BLOCK_M=128/BN=64 it is
1.31, which is larger than the cap-to-ridge gap the study is deciding, so a cap
quoted without it is not a bound on anything. At BLOCK_M=16/BN=64 the activation
share is a quarter of the weight share and the factor is smaller, which is the
other reason this tile is where the formula is testable.

The activation correction `analyse` applies removes the activation slope from
`B` before dividing, which is the `phi` term of EXA taken off by measurement
rather than by model; `delta` stays in, so the printed factor is what remains
between the corrected alpha and `alpha_b`.

VALIDITY GATES (V) VERSUS CLAIM GATES (C). A V that FAILs means no number on
this page may be quoted: the kernel was not the one asked for, the instrument
was never shown to work, or the sweep was too shallow for an absence to mean
anything. A C that FAILs is a result and is meant to be publishable as one:
C1 failing says a decode-tuned tile DOES approach its compute roof, which
retracts the ceiling this study put on it.

THE DENOMINATOR, said once because it is the difference between a control and a
tautology. Every SCORED roof fraction here is against `ridge x bandwidth`, never
against the run's own plateau. The plateau is the maximum over the same cells,
so a control read against it scores 1.00 by construction and the check examines
nothing.

A SECOND FRACTION IS PRINTED BESIDE THE FIRST AND IS NEVER SCORED. On this card
the under-load SM clock is set per tile by the kernel's own power draw under the
700 W cap, so a fraction of the fixed roof mixes how well a tile uses the
machine with what clock the governor gave it. `issue_ladder` divides instead by
the roof at the tread's own clock, which separates the two; the pair is written
`fixed/own-clock` on every throughput line and appended to V3's and C1's
measured lines as `own-clock`. No threshold anywhere reads it.

THE NUMERATOR HAS A RULER OF ITS OWN, AND `ridge x bandwidth` IS NOT IT. That
product is the DENSE cuBLAS peak: one GEMM, no gate, no `moe_align_block_size`,
no scatter of tokens to experts, no SiLU, no `moe_sum`, and no second GEMM
chained behind the first. `fused_experts` is all of those in one call and its
FLOPs are counted from the two GEMMs alone, so a fused layer at its own
structural best still reports a fraction well under one -- not because a tile is
capped, but because the ruler is measuring a different kernel. Across the 26
published reports the sibling checked, the fused-layer plateau ran 46.5-75.6% of
`ridge x bandwidth` and no arm has ever exceeded 0.54 on the H200 or 0.64 on the
A100.

V3 USED TO ASK FOR 0.95 OF THAT PRODUCT, and that is the defect this file
carried. A validity gate demanding the fused layer reach the dense peak fails on
every card that exists, and a VALIDITY FAIL makes the whole page unquotable, so
the arm was INVALID by construction before it was scheduled: nine pod minutes
whose outcome was computable from the calibration alone. V3 now scores the
control against the FUSED-LAYER ROOF -- `FUSED_PLATEAU_BAND[0]` of
`ridge x bandwidth`, the low end of that published band -- and is two-sided: a
control BELOW it means nothing in this sweep is near any roof and the absence at
the cap tile is unbracketed; a control ABOVE `ridge x bandwidth` is impossible
for a fused layer and means the ruler, not the kernel, is wrong, which is
UNDECIDED rather than a pass. The dense-peak fraction is still computed and
still printed, labelled as the diagnostic it is, because it is the number that
compares across cards. It is not the verdict.

WHAT IT WRITES, AND WHERE IT SURVIVES TEARDOWN. Under `$MOE_RESULTS_DIR`, else
`/workspace/results` when it exists (the RunPod network volume, which outlives
the pod), else `<repo>/results`:

    <results>/tile_cap/<run-id>/cells.csv      one row per (BLOCK_M, T)
    <results>/tile_cap/<run-id>/report.json    plan, predictions, gate verdicts
    <results>/tile_cap/<run-id>/report.txt     exactly what was printed
    <results>/tile_cap/<run-id>/triton-cache/  per-setting compile evidence

`git check-ignore` IS RUN on the resolved output directory and its verdict is
printed in the plan; it is not asserted from a reading of `.gitignore`, because
the answer differs for `results/`, `results/published/` and a path outside the
work tree, and a sentence is right for exactly one of them. `cells.csv` is
appended and flushed per cell and a re-run with the same arguments resumes it,
so an abort costs the cell in flight.

THE RUN ID CARRIES EVERY ARGUMENT THAT CHANGES A MEASURED CELL, and that
includes the CARD (the operator sweeps it by moving pods, and the volume
outlives the pod), `--iters`, `--warmup`, `--cell-budget-ms` (they change the
milliseconds of every cell), and `--self-test` with its noise, which also
prefixes the directory with `synthetic-` and stamps `"synthetic": true` into
`report.json`. `--alpha`, `--ridge` and `--ridge-band` are deliberately NOT in
the key: they change the analysis over a set of cells, not the cells.

THE RIDGE IS RESOLVED FROM THE ATTACHED DEVICE'S CALIBRATION, through the
sibling's `resolve_ridge`, and a measured run with no calibration for its own
device REFUSES. Both claim gates are fractions of a ridge, so scoring an A100
run against the module's 160.3 H200 band -- while reading the bandwidth off the
A100 -- would assemble the roof out of two machines and put every verdict 1.10x
out. `--self-test` pins the band instead of resolving it, for the same reason
its bandwidth is pinned: a replay that reads the hardware is not a replay.

EXIT CODES AND THE ONE GREPPABLE LINE. `moe.bench.exit_codes` owns both, so this
runner and the driver cannot come to different views of what an integer means.
Every scored gate prints exactly one `RESULT: KIND NAME VERDICT detail` line and
nothing else in this file's output starts with `RESULT: `; the process code is
`exit_codes.classify` over the same gate objects, so `classify_text` on the log
recomputes it. DONE 0, CLAIM_FAIL 1, REFUSED 2 (nothing measured), INVALID 3 (a
validity gate failed AFTER measuring, nothing quotable), ERROR 4. NOTHING IS
FOLDED: `--fail-on-gate` is retired, accepted and ignored. It used to report a
CLAIM_FAIL as DONE, and `--self-test 0.10` is the live instance -- that world's
log carries `RESULT: CLAIM ... FAIL`, `classify_text` reads 1 out of it, and the
process returned 0. A falsified pre-registered claim IS a successful experiment,
and CLAIM_FAIL is already the code that says so: 1 is in `FINISHED_CODES` and
`ledger_state(1)` is "CLAIM_FAIL", so the ledger never retries it. `--dry-run`
exits REFUSED (2), not DONE, for the reason spelled out at that branch: it
scores no gate, prints no RESULT line, and `classify_text` on a log with none
raises `NoGatesScored`, which is the REFUSED shape.

THE PLAN STATES A MINIMUM DETECTABLE EFFECT, derived from a stated noise
assumption rather than from a hope. `--plant-noise` is that assumption and its
default is the published per-cell spread, so `--self-test` and `--dry-run` both
argue at the noise a pod actually produces instead of at zero. Every gate's MDE
is printed beside its threshold in the plan: a gate whose threshold sits inside
its own MDE cannot decide anything, and V2 is the one that does, which is why V2
REFUSES above a spread rather than widening.

THE SELF-TEST PLANTS WORLDS WITH REGISTERED VERDICTS. `SELF_TEST_WORLDS` names
each planted alpha, what it is planted to demonstrate, and the verdict every
gate must return in it; `--self-test ALPHA` asserts them and exits ERROR when a
world comes out other than registered. Three worlds, and the third exists
because the first two could not fail C2: at alpha=0.558 C2 PASSES and at the
retracted 0.10 the memory branch runs parallel to the compute branch (ridge/cap
= 1.002, inside the tolerance) so the fit declines to name an alpha and C2 comes
back UNDECIDED. `C2_FAIL_ALPHA` sits between them, where the fit identifies an
alpha and that alpha puts the ceiling above the discriminator, which is the only
place C2's FAIL branch is reachable.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# `scripts/` is not a package and never has been, so the sibling sweep is
# imported by putting its directory on the path rather than by a relative
# import. Registering it under its own module name (which a plain `import`
# does) matters: `@dataclass` resolves annotations through
# `sys.modules[cls.__module__]`, and loading the file by path without
# registering it first fails inside the decorator with an error that names
# nothing useful.
sys.path.insert(0, str(HERE))

# The MDE arithmetic lives in `replicate_noise_floor` -- the t and z quantiles,
# the three designs and the reasons they differ -- and is imported rather than
# restated here for the same reason the sweep's gates are: a second
# implementation of a power calculation agrees with the first until it does
# not, and the way it disagrees is a gate that looks decisive and is not. The
# EXA identity and the lin_overstatement factor come from `moe.bench.ai_model`
# through the sibling's `cap_overstatement`, for the same reason.
import block_m_crossing_sweep as SWEEP  # noqa: E402
import replicate_noise_floor as NOISE  # noqa: E402

from moe.bench import exit_codes, roofline  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

# --------------------------------------------------------------------------
# Everything this script argues about, stated before any of it is used. The
# three physical constants are IMPORTED rather than restated: a copy of alpha
# here that drifted from the sweep's would make the two scripts disagree about
# which world they are in while both printed a confident table.
# --------------------------------------------------------------------------

ALPHA = SWEEP.ALPHA                       # 0.558, refit 2026-08-31
ALPHA_BAND = SWEEP.ALPHA_BAND             # (0.529, 0.588)
RETRACTED_ALPHA = SWEEP.RETRACTED_ALPHA   # 0.10, the world every gate discriminates against
#: (160.3, 176.2) Op/B: WITHDRAWN 2026-09-02 as any card's ridge. It is two compute
#: calibrations of one H200 9.9% apart (its own ridge is 162.8, the A100's 145.8),
#: pinned by --self-test only as the HYPOTHESIS the sibling module labels it, so a
#: planned run can mislabel nothing. A measured run resolves the attached card's.
RIDGE_BAND = SWEEP.RIDGE_BAND             # withdrawn pair, HYPOTHESIS_RIDGE_SOURCE

#: The tile under test. 16 is what `get_default_config` returns for M <= 32,
#: which is the decode regime on every shape in this study that has no tuned
#: file -- and in the observed-tile arm every one of the 24 cells that took that
#: ladder held ONE M-tile per expert. So this is the tile whose cap is easiest to
#: MEASURE, not the tile whose cap decides anything in production; BLOCK_M=128 is
#: that one, and `scripts/bm128_depth.py` is where it is asked.
CAP_TILE = 16

#: The positive control. See the header for why tread 1 makes 256 the control
#: whose crossing does not depend on the parameter under test.
DEFAULT_CONTROL = 256

#: WHICH TILE HEIGHTS vLLM ACTUALLY RUNS MULTI-TILE, counted from the one
#: published arm that records the tile it chose:
#: `results/published/2026-09-01-nvidia_h200-alpha-0558/merged.csv`, column
#: `tile_block_m` where `tile_config_source` is vllm_default or vllm_tuned. A
#: cell is one (model, num_tokens) under uniform routing; 132 cells, 7 routing
#: seeds each. Both readings of rows per expert are carried, because the first
#: version of this table carried only the mean and read "never" off it.
#:
#: `(cells run multi-tile, cells, max M-tiles per expert)`, rows per expert read
#: as `load_mean_rows`. The mean is a fact about the routing histogram, not
#: about the launch, and it is kept because it is the basis the earlier count
#: and `scripts/bm128_roofline.py` quote.
OBSERVED_MULTI_TILE_MEAN_ROWS: dict[int, tuple[int, int, int]] = {
    16: (0, 24, 1),
    32: (0, 5, 1),
    64: (0, 16, 1),
    128: (59, 87, 32),
}

#: The same triple read as `load_max_rows`, the BUSIEST expert, which is what
#: `moe_align_block_size` pads to and therefore the tile count the launch grid
#: holds: THIS is the basis production sees. A cell counts as multi-tile when
#: ANY of its seven seeds needed a second tile. This is the table that demotes
#: this experiment from a production claim to a formula test: the re-read term
#: `Q(n) = 1 + alpha (n - 1)` is exactly 1 at one tile, and at 16 a second tile
#: appears in one cell, one seed. A tile height ABSENT from this dict was not
#: observed at all, and the report says "not observed" rather than assuming
#: either answer.
OBSERVED_MULTI_TILE_MAX_ROWS: dict[int, tuple[int, int, int]] = {
    16: (1, 24, 2),
    32: (0, 5, 1),
    64: (2, 16, 2),
    128: (66, 87, 34),
}

#: `(seed-rows run multi-tile, seed-rows)` on `load_max_rows`, the per-seed
#: count behind the cell count above: the isolated cells at 16 and 64 are 1 of
#: 168 and 5 of 112 seed-rows, and 128's regime is 455 of 609.
OBSERVED_SEED_ROWS_MAX_ROWS: dict[int, tuple[int, int]] = {
    16: (1, 168),
    32: (0, 35),
    64: (5, 112),
    128: (455, 609),
}


def observed_note(block_m: int) -> str:
    """One sentence on what vLLM was seen to do at this tile height, or a refusal.

    REFUSES rather than defaults. "no cell ran multi-tile" and "no cell was
    observed" are different statements and the second must never print as the
    first: the whole demotion rests on this count, so a tile height nobody
    measured has to say so.

    BOTH BASES ARE PRINTED AND THE PADDED ONE DECIDES THE VERB. The first
    version of this note printed the mean-rows count alone and said "never
    reached for in production" of a tile that fires on the padded count in one
    cell; that kept the guarantee above and broke a second one, which is that
    "no cell ran multi-tile" must be true on the count the launch actually
    uses. So a tile height with any multi-tile cell on `load_max_rows` says
    "isolated cells", one with none says "never", and both print the mean-rows
    count beside it so the two readings cannot be confused again.
    """
    seen_max = OBSERVED_MULTI_TILE_MAX_ROWS.get(block_m)
    seen_mean = OBSERVED_MULTI_TILE_MEAN_ROWS.get(block_m)
    if seen_max is None or seen_mean is None:
        return (f"BLOCK_M={block_m} does not appear in the observed-tile arm, "
                "so whether vLLM ever runs it multi-tile is UNMEASURED here")
    multi, cells, top = seen_max
    mean_multi, _, mean_top = seen_mean
    seed_multi, seed_rows = OBSERVED_SEED_ROWS_MAX_ROWS[block_m]
    mean_clause = (f"on mean rows {mean_multi} of {cells} cells, up to "
                   f"{mean_top} tile{'s' if mean_top != 1 else ''}")
    if multi == 0:
        return (f"vLLM ran BLOCK_M={block_m} as ONE M-tile per expert in "
                f"{cells} of {cells} observed cells on the padded count "
                f"(load_max_rows, {seed_rows} seed-rows; {mean_clause}), where "
                "Q(n) = 1 exactly, so this ceiling is never reached for in "
                "production")
    if seed_multi * 10 < seed_rows:
        return (f"vLLM ran BLOCK_M={block_m} multi-tile in ISOLATED CELLS: "
                f"{multi} of {cells} observed cells on the padded count "
                f"(load_max_rows), {seed_multi} of {seed_rows} seed-rows, up to "
                f"{top} M-tiles per expert; {mean_clause}. Q(n) = 1 everywhere "
                "else, so this ceiling is reached for in production only there "
                "and never as a regime")
    return (f"vLLM ran BLOCK_M={block_m} multi-tile in {multi} of {cells} "
            f"observed cells on the padded count (load_max_rows; {seed_multi} "
            f"of {seed_rows} seed-rows), up to {top} M-tiles per expert; "
            f"{mean_clause}")

#: C1's threshold, and the same number gate 4 of the parent sweep uses, so "near
#: the roof" means one thing across the two scripts. The model puts the refit
#: world at 0.176 and the retracted one at 0.893 with the default depth, so 0.85
#: sits inside the gap and closer to the world it must be able to catch.
ROOF_FRACTION = SWEEP.GATE4_ROOF_FRACTION

#: V4: the modelled AI at the deepest tread, as a fraction of its own ceiling.
#: A curve still climbing has not finished rising and its top is not its top.
SATURATION_FLOOR = 0.95

#: V2: a control that has reached a compute roof stops gaining. 2% is twice the
#: per-cell timing spread this harness produces on a quiet H200, and it is a
#: FLOOR: the gate is `max(this, 3 x measured spread)`, the same rule the parent
#: sweep applies to its memory-branch margin. Fixed at 2% it FAILED every
#: self-test above 1% lognormal noise, which is inside what a real pod produces
#: -- the statistic is a ratio of two single cells, so it carries `sqrt(2)`
#: times the per-cell spread and a gate below that is a coin flip that voids the
#: whole page.
CONTROL_FLAT_GAIN = 0.02

#: ...and a ceiling on that widening. Past this the flatness test has been
#: relaxed until it accepts a ladder still visibly climbing, so V2 stops
#: answering instead of answering yes. A run this noisy has a bigger problem
#: than the control.
CONTROL_FLAT_GAIN_MAX = 0.10

#: V2: `compute_reference`'s own tolerance for calling a ladder proportional to
#: its tile count. Restated here because V2 is scored against it and a gate
#: whose threshold lives in another module's default argument cannot be read.
PROPORTIONALITY_MAX_ERR = 0.05

#: The fraction of `ridge x bandwidth` at which the sibling calls a ladder
#: compute bound. Imported rather than restated so the two scripts cannot come
#: to different answers about what "reached the DENSE roof" means. V3 no longer
#: scores against it -- see `FUSED_PLATEAU_BAND` -- and it is kept because the
#: dense-peak fraction is still computed and printed as the cross-card
#: diagnostic.
COMPUTE_BOUND_FRACTION = SWEEP.COMPUTE_BOUND_FRACTION

#: THE FUSED LAYER'S OWN CEILING, as a fraction of `ridge x bandwidth`, measured
#: across the 26 published reports the sibling's `bracketing` docstring counts:
#: their plateaus run 46.5% to 75.6% of the card's dense peak, and no arm has
#: exceeded 0.54 on the H200 or 0.64 on the A100.
#:
#: WHY A BAND AND NOT A POINT. `fused_experts` is a gate, an alignment kernel,
#: two GEMMs, a SiLU and a reduction, and only the two GEMMs' FLOPs are counted;
#: how much of the rest lands in the measured interval moves with the model, the
#: dtype and the token count. The low end is what a run must CLEAR to have shown
#: this apparatus can drive a fused layer to a roof at all; the high end is the
#: most any fused layer in this study has managed and is printed beside the
#: verdict so a control near it is not read as a control that fell short.
FUSED_PLATEAU_BAND = (0.465, 0.756)

#: V3's floor: the low end of that band. A control under it did not reach any
#: roof, fused or dense, and the absence at the cap tile is then an absence
#: recorded by an instrument never shown to detect a presence.
FUSED_ROOF_FLOOR = FUSED_PLATEAU_BAND[0]

#: ...and V3's ceiling. A fused layer counting only its two GEMMs' FLOPs cannot
#: exceed the DENSE peak; a control above it says the ridge, the bandwidth or
#: the FLOP count belongs to another machine, which is a broken ruler and not a
#: strong kernel. 1.0 exactly, with a tolerance for the one legitimate way to
#: land marginally over it: a synthetic world generated AT the roof plus timing
#: noise.
FUSED_ROOF_CEILING = 1.0

#: FLOOR on how far past the dense peak a control may land before V3 calls the
#: ruler broken. The gate widens it to `max(this, 3 x the measured per-cell
#: spread)`, the same rule V2's flatness uses and for the same reason: the
#: statistic is a MAXIMUM over the control's treads, so it carries the per-cell
#: spread with a positive bias. Measured off the planted world: at a spread of
#: 0.00% the synthetic control lands at exactly 1.000 of the dense peak, and its
#: peak rises to 1.02 / 1.04 / 1.07 at spreads of 1 / 2 / 3%, all of which are
#: noise and none of which is a broken ruler. A ruler assembled from two
#: machines is wrong by 10% (the two cards' ridges differ by that) and a wrong
#: FLOP count is wrong by a factor, so nothing this gate must catch is inside
#: the widened band.
FUSED_ROOF_CEILING_TOLERANCE = 0.05

#: The per-cell relative timing spread this study actually produces, and the
#: default `--plant-noise`. The published H200 ladders run 0.76-1.82% and the
#: A100 ones 0.48-0.61%; 1.5% is inside the H200 range and near its top, which
#: is the end a gate has to survive.
#:
#: A ZERO DEFAULT IS THE DEFECT THIS REPLACES. Both planted worlds used to be
#: noiseless, so V2's `max(2%, 3 x spread)` and C1's thresholds were only ever
#: exercised at a spread of 0.00% -- the one value no pod produces -- and the
#: self-test could not have discovered that a gate is a coin flip at the noise
#: the hardware delivers.
PUBLISHED_CELL_SPREAD = 0.015

#: The alpha whose world makes C2 FAIL, and the reason a third world exists.
#: See `SELF_TEST_WORLDS`: at 0.558 C2 passes and at 0.10 the memory branch runs
#: parallel to the compute branch so the fit refuses to name an alpha at all.
#: 0.14 puts `cap = 2*16/(0.14*2) = 114.3` Op/B, which is 0.713 of the 160.3
#: the self-test pins (a module constant, not a card's ridge; the verdict is
#: the same at 162.8) -- above the 0.589 discriminator, so C2 FAILS -- while
#: `ridge/cap = 1.40`
#: sits outside the 15% parallel-branch tolerance, so the fit still identifies
#: it. Derived once, here, rather than tuned until the world came out right.
C2_FAIL_ALPHA = 0.14

#: Treads before an alpha may decide anything. The parent's constant, reused for
#: the same reason it exists there.
MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS

#: Exactly-full stacks EACH tile has to put on the grid before a cell is timed.
#: Three is the floor a through-origin fit reads, and V1 has always demanded it
#: AFTER the sweep. It is a PLAN-TIME requirement now.
#:
#: 2026-09-09, H200: `r_max = depth.rows` was 688 on that card's own ridge band,
#: `build_grid` stopped at 672 because 688 is not a multiple of --row-step 32,
#: and only 256 and 512 are multiples of the 256 control. The plan printed
#: "BM=256:2" and ran anyway; 72 cells were measured to a foregone V1 FAIL and
#: nothing on the page was quotable.
V1_ALIGNED_NEEDED = 3

#: Control stacks the grid is FLOORED at, which is a stronger requirement than
#: V1's three and comes from V2 rather than from V1. V2 reads the control's LAST
#: tread gain: on the model itself a 3-tread control is still amortising its
#: fixed cost and gains +3.10% against a +/-3.00% gate, while a 4-tread control
#: reads -1.97% and 5 and 6 treads read +0.38% and +0.60% (planted alpha 1.0 and
#: 0.558 worlds at a 1% spread, H200 band). Three treads would satisfy V1 and
#: then fail V2 for a reason that is a property of the grid, not of the card.
CONTROL_STACKS_FLOOR = 4

#: The published H200 triad ceiling, and the bandwidth `--self-test` uses unless
#: one is given on the command line.
#:
#: A REPLAY THAT READS THE HARDWARE IS NOT A REPLAY. `resolve_bandwidth` prefers
#: THIS MACHINE's calibration, which is right for a run that predicts
#: milliseconds on the card it is running on and wrong for a synthetic world:
#: the same `--self-test 0.558` would generate different cells on the pod than
#: on a laptop, the two would print different numbers under the same command,
#: and the test suite could not pin either. Pinned here so a self-test is
#: hermetic and identical everywhere.
PUBLISHED_H200_GBPS = 4374.5

PASS, FAIL, UNDECIDED = SWEEP.PASS, SWEEP.FAIL, SWEEP.UNDECIDED
VALIDITY, CLAIM = "VALIDITY", "CLAIM"


#: The card slug a run id carries when no device is attached, i.e. every
#: --dry-run and every --self-test on a laptop. Visible rather than blank, so a
#: laptop directory cannot be mistaken for the one a pod would write to.
NO_CARD_SLUG = "nocard"


def detect_card_slug() -> str:
    """Slug for the ATTACHED device, or `NO_CARD_SLUG`.

    THE CARD IS A SWEPT PARAMETER: it is swept by the operator moving pods, and
    the results root defaults to `/workspace/results`, a RunPod network volume
    that outlives the pod. Without the card in the run id the same command on an
    H200 and then an A100 shares one directory, the second finds every cell
    present, skips all 162, spends no GPU time and prints the first card's
    timings under the second's heading -- with C1 and C2 scored against the
    second card's ridge. That has already been committed once in this repo: the
    A100 and H200 cross-card arms carry IDENTICAL report filenames.
    """
    try:
        import torch
    except ImportError:
        return NO_CARD_SLUG
    try:
        if not torch.cuda.is_available():
            return NO_CARD_SLUG
        name = torch.cuda.get_device_name(0)
    except Exception:                                   # noqa: BLE001
        # A driver that is present but unusable is not a card identity. Naming
        # it `nocard` keeps the run out of a real card's directory.
        return NO_CARD_SLUG
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or NO_CARD_SLUG


# --------------------------------------------------------------------------
# Refusals. Typed, because "this cannot be measured" and "this measured zero"
# have to be distinguishable by a caller, and because a bare exception in a
# 2000-cell pod run says nothing about whether the cells are still good.
# --------------------------------------------------------------------------

class CapTestRefusal(RuntimeError):
    """Base: the script declines to produce a number rather than produce one."""


class NonDiscriminating(CapTestRefusal):
    """The requested design cannot tell the two worlds apart, at any depth.

    Raised at PLAN time, before a pod is billed. The case it exists for: a cap
    tile so small that even the retracted alpha's ceiling sits below the gate
    threshold, so C1 could never FAIL and a PASS would mean nothing. That is
    failure mode 1 (a gate that cannot fail is as useless as one that cannot
    pass), and it is silent unless something checks for it.
    """


class Unmeasurable(CapTestRefusal):
    """A quantity the cells cannot support. Never substituted with a default."""


class SiblingChanged(CapTestRefusal):
    """`block_m_crossing_sweep` no longer exposes what this runner calls.

    THE COST OF BEING A THIN RUNNER. Everything metered here belongs to the
    sibling: its grid, its timer, its resume, its compile assay, its ladder fit
    and three of its gates. That is the right trade -- a second implementation
    of any of them would agree with the first until it did not -- and the bill
    is that a change over there lands here as a TypeError, which on a pod
    arrives AFTER the sweep and destroys the run rather than the plan. It has
    already happened once: `compute_reference` gained four required keyword
    arguments (the LEVEL checks) between this file being written and being
    tested. So the API is probed before anything is spent, and a mismatch is a
    refusal that names the function.
    """


#: Names and keyword arguments this runner needs from the sibling. Checked by
#: `require_sweep_api`, which runs before the plan is printed.
REQUIRED_SWEEP_API: dict[str, tuple[str, ...]] = {
    "build_grid": (),
    "synthetic_cells": ("alpha", "ridge", "bandwidth_gbps", "b", "sm_count"),
    "run_sweep": (),
    "compute_reference": ("cfg", "ridge", "bandwidth_gbps", "b", "pinned",
                          "capability"),
    "fit_ladder": (),
    "ladder_points": (),
    "activation_slope_ms": (),
    "gate_0_override": (),
    "gate_1_steps": ("alpha", "ridge", "bandwidth_gbps", "b", "noise"),
    "gate_2_direction": ("alpha", "retracted", "ridge", "bandwidth_gbps", "b",
                         "block_sizes"),
    "predictions": (),
    "predict_tile": (),
    "ai_cap": (),
    "q_of_tiles": (),
    "model_ms": ("alpha", "ridge", "bandwidth_gbps", "b"),
    "estimated_seconds": ("alpha", "ridge", "bandwidth_gbps", "b", "iters",
                          "warmup", "cell_budget_ms"),
    "tokens_for_rows": (),
    "rows_step": (),
    "tile_resource_plan": (),
    "resolve_capability": ("synthetic",),
    "resolve_bandwidth": (),
    # Added 2026-09-02 with the ridge fix. If the sibling drops or renames it,
    # this runner must REFUSE rather than fall back to `RIDGE_BAND[0]`, which
    # is the constant the fix removed.
    "resolve_ridge": ("synthetic",),
    "results_root": (),
    "missing_gpu_stack": (),
    "_throughput_ladder": (),
    # Added 2026-09-02 with the instrument, the EXA label and the provenance
    # block. Each is a number this report PRINTS, so a rename over there must
    # refuse here rather than arrive as an AttributeError after the sweep.
    "timing_basis": (),
    "observed_iters": (),
    "iters_line": (),
    "cap_overstatement": (),
    # Added 2026-09-09 with the shape-refusal repair below. `control_reference`
    # BUILDS one of these, and a dataclass built positionally is worse than a
    # missing name: a reorder of the first five same-typed fields over there
    # would land here as wrong values in the right slots, silently. Named
    # fields plus this entry make a rename a refusal before the plan is printed.
    "ComputeReference": ("block_m", "overhead_ms", "slope_per_tile",
                         "mean_rel_err", "note"),
}

#: Module-level values read from the sibling. `FIXED` is the one that matters
#: for the probe's ORDER: `build_parser` reads it for its defaults, so it is
#: touched before a single line of this file's own logic runs, and a probe that
#: fired after `parse_args` would fire after the AttributeError it exists to
#: replace. The rest are read at import, where a rename fails loudly anyway.
REQUIRED_SWEEP_CONSTANTS: tuple[str, ...] = (
    "ALPHA", "ALPHA_BAND", "RETRACTED_ALPHA", "RIDGE_BAND", "FIXED",
    "GATE4_ROOF_FRACTION", "GATE2_RATIO", "COMPUTE_BOUND_FRACTION",
    "MIN_MEMORY_TREADS", "MEMORY_BRANCH_MARGIN", "PARALLEL_BRANCH_TOLERANCE",
    "DEFAULT_SM_COUNT", "RidgeUnavailable", "SYNTHETIC_INSTRUMENT",
)


def require_sweep_api() -> None:
    """Refuse now if the sibling has moved, rather than mid-run.

    Checks that every function this file calls exists and still accepts the
    keyword arguments it is called with. It cannot check semantics, and does not
    pretend to: it turns the loudest and most likely class of drift into a
    message that names the function instead of a traceback out of `analyse`.
    """
    import inspect
    problems = [f"the constant {name} is gone"
                for name in REQUIRED_SWEEP_CONSTANTS
                if not hasattr(SWEEP, name)]
    for name, kwargs in REQUIRED_SWEEP_API.items():
        fn = getattr(SWEEP, name, None)
        if fn is None:
            problems.append(f"{name} is gone")
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):                   # pragma: no cover
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        missing = [k for k in kwargs if k not in params]
        if missing:
            problems.append(f"{name} no longer takes {missing}")
        required = [n for n, p in params.items()
                    if p.default is inspect.Parameter.empty
                    and p.kind is inspect.Parameter.KEYWORD_ONLY
                    and n not in kwargs]
        if required:
            problems.append(f"{name} now requires {required}, which this "
                            "runner does not pass")
    if problems:
        raise SiblingChanged(
            "scripts/block_m_crossing_sweep.py has changed under this runner:\n  "
            + "\n  ".join(problems)
            + "\nEverything metered here is the sibling's, so this is not a "
              "wrapper that can carry on with a stale copy. Read the sibling's "
              "new signature and update REQUIRED_SWEEP_API and the call in "
              "`analyse` together.")


# --------------------------------------------------------------------------
# The depth argument. Pure arithmetic over the model in `block_m_crossing_sweep`
# so it is checkable off GPU and by the test suite.
# --------------------------------------------------------------------------

def saturation(tiles: int, alpha: float) -> float:
    """`AI(n) / cap`: how much of its own ceiling the modelled AI has reached.

    `AI(n) = (2 n BM / b) / Q(n)` and `cap = 2 BM / (alpha b)`, so BLOCK_M and
    the dtype cancel exactly and this is a function of the tile COUNT and alpha
    alone. Worth knowing: the depth requirement below is the same in tiles for
    every block size and every dtype.
    """
    if alpha <= 0:
        raise Unmeasurable(
            f"saturation is undefined at alpha={alpha}: with no re-read cost "
            "the AI is unbounded and has no ceiling to be a fraction of")
    return alpha * tiles / SWEEP.q_of_tiles(tiles, alpha)


def tiles_for_saturation(alpha: float, target: float = SATURATION_FLOOR) -> int:
    """Smallest tread count with `saturation >= target`: `t(1-a) / (a(1-t))`."""
    if not 0.0 < target < 1.0:
        raise Unmeasurable(f"saturation target {target} is not a fraction")
    if alpha <= 0:
        raise Unmeasurable("no ceiling exists at alpha <= 0")
    return max(1, math.ceil(target * (1.0 - alpha) / (alpha * (1.0 - target))))


def retracted_horizon_tiles(block_m: int, *, retracted: float, ridge: float,
                            b: int, roof_fraction: float,
                            max_tiles: int = 1_000_000) -> int:
    """Depth at which the RETRACTED world would trip C1, which is the horizon.

    THE NUMBER THAT MAKES AN ABSENCE EVIDENCE. C1 reports the fraction of
    `ridge x bandwidth` the cap tile reached, and fails above `roof_fraction`.
    A sweep that stopped before the competing hypothesis would
    have crossed that line has not tested it, and its PASS is "we stopped early"
    dressed as a ceiling.

    The parent sweep's `Bracketing` derives its horizon from the retracted
    world's CROSSING, which is the right quantity at BLOCK_M=64 and a VACUOUS
    one here: at BLOCK_M=16 the retracted alpha predicts no crossing either on
    the H200 (cap 160.0 against its own 162.8, or the pinned 160.3),
    `crossing_rows` is None, its horizon
    collapses to `2 * 0.0` and every depth clears it. A check that examined
    nothing reports no failures, so this asks the question the threshold is
    actually about instead.
    """
    cap = SWEEP.ai_cap(block_m, retracted, b)
    target = roof_fraction * ridge
    if cap <= target:
        raise NonDiscriminating(
            f"at alpha={retracted} the BLOCK_M={block_m} AI ceiling is "
            f"{cap:.1f} Op/B, below the {roof_fraction:.0%} of ridge "
            f"{ridge:.1f} that gate C1 fails at ({target:.1f} Op/B). The "
            "retracted world could never trip C1 at any depth, so a PASS would "
            "not rule it out and this run would buy nothing. Pick a cap tile "
            "whose retracted ceiling clears the threshold, or score the claim "
            "on C2 (the measured alpha) alone.")
    for n in range(1, max_tiles + 1):
        if (2.0 * n * block_m / b) / SWEEP.q_of_tiles(n, retracted) >= target:
            return n
    raise NonDiscriminating(                              # pragma: no cover
        f"ceiling {cap:.1f} clears the target {target:.1f} but no depth under "
        f"{max_tiles} tiles reached it; the cap and the scan disagree")


@dataclass(frozen=True)
class Depth:
    """How deep the run must go, and which of C1's two conditions that buys.

    C1 ASKS TWO THINGS AND THEY HAVE DIFFERENT HORIZONS.

      NEAR THE ROOF, `peak <= 0.85 of ridge x bandwidth`, which is the plain
      reading of "structurally incapable of reaching its compute roof".
      DISCRIMINATING, `peak <= the midpoint of the two worlds' predicted
      ceilings`, which is the reading that rules the retracted alpha out.

    The retracted world crosses the second at 13 tiles and the first at 132, so
    a sweep that stops between them can rule out alpha=0.10 and CANNOT say the
    tile never got near the roof -- at that depth the retracted world would not
    have got near it either, so passing says nothing. Which conditions a given
    depth makes live is therefore computed and printed, and C1 scores only the
    live ones. Printed in the plan BEFORE the sweep runs, because the only cheap
    moment to find that a grid cannot support its own conclusion is before the
    pod is rented.
    """

    tiles: int
    rows: int
    #: `(label, threshold, {ridge: tiles})` per condition, worst ridge binding.
    horizon_roof: dict[float, int]
    horizon_disc: dict[float, int]
    saturation_tiles: int
    binding: str

    @property
    def roof_tiles(self) -> int:
        return max(self.horizon_roof.values())

    @property
    def disc_tiles(self) -> int:
        return max(self.horizon_disc.values())

    def roof_condition_live(self, reached: int) -> bool:
        return reached >= self.roof_tiles

    def disc_condition_live(self, reached: int) -> bool:
        return reached >= self.disc_tiles

    def summary(self) -> str:
        """One line, for a gate that has no `cfg` to turn rows into tokens."""
        return (f"{self.tiles} tiles = max(near-roof horizon {self.roof_tiles}, "
                f"discriminating horizon {self.disc_tiles}, saturation "
                f"{self.saturation_tiles}, fit floor {MIN_MEMORY_TREADS}); "
                f"binding: {self.binding}")

    def lines(self, cfg, cap_tile: int) -> list[str]:
        out = [f"depth required: {self.tiles} tiles of {cap_tile} rows = "
               f"{self.rows} rows per expert "
               f"(T = {SWEEP.tokens_for_rows(cfg, self.rows)})"]
        for label, horizons in (("near-roof", self.horizon_roof),
                                ("discriminating", self.horizon_disc)):
            for ridge, tiles in sorted(horizons.items()):
                out.append(
                    f"  {tiles:5d} tiles: depth at which alpha="
                    f"{RETRACTED_ALPHA} would trip C1's {label} condition at "
                    f"ridge {ridge}")
        out.append(f"  {self.saturation_tiles:5d} tiles: modelled AI within "
                   f"{1 - SATURATION_FLOOR:.0%} of its own ceiling at alpha="
                   f"{ALPHA_BAND[0]}, the band's low end")
        out.append(f"  binding requirement: {self.binding}")
        return out


def required_depth(cap_tile: int, *, b: int, ridge_band: tuple[float, float],
                   alpha_low: float = ALPHA_BAND[0],
                   roof_fraction: float = ROOF_FRACTION) -> Depth:
    """Every requirement from the header, and the largest of them.

    `alpha_low` rather than `alpha` for saturation: it is slowest at the LOW end
    of the band, so the band's low end is the conservative one to plan against.
    Both ends of the RIDGE band are computed for each condition and the deeper
    one binds, for the same reason: a depth that only works at one end of a band
    the study cannot narrow is a depth that only works if we are lucky.

    `ridge_band` IS REQUIRED AND HAS NO DEFAULT, for the same reason `--ridge`
    no longer defaults to `RIDGE_BAND[0]`. The depth is how far the retracted
    world would have to be swept before it tripped C1, and that horizon is set
    by the ridge -- 13 tiles at 160.3, 132 at 176.2. The module band is a
    2026-08-26 H200 figure, so planning an A100 run against it buys the wrong
    number of cells: at 145.7 the retracted ceiling 160.0 sits ABOVE the ridge
    and the horizons are different again. A default here would put that constant
    back one call site below the one the fix removed it from, so the caller
    resolves the band for the attached device and passes it in or does not get a
    depth at all.
    """
    roof, disc = {}, {}
    for ridge in ridge_band:
        roof[ridge] = retracted_horizon_tiles(
            cap_tile, retracted=RETRACTED_ALPHA, ridge=ridge, b=b,
            roof_fraction=roof_fraction)
        disc[ridge] = retracted_horizon_tiles(
            cap_tile, retracted=RETRACTED_ALPHA, ridge=ridge, b=b,
            roof_fraction=cap_discriminator(cap_tile, ridge, b))
    sat = tiles_for_saturation(alpha_low)
    tiles = max(max(roof.values()), max(disc.values()), sat, MIN_MEMORY_TREADS)
    binding = ("the near-roof horizon at the worst end of the ridge band"
               if tiles == max(roof.values()) else
               "the discriminating horizon at the worst end of the ridge band"
               if tiles == max(disc.values()) else
               f"AI saturation at alpha={alpha_low}"
               if tiles == sat else
               f"the {MIN_MEMORY_TREADS}-tread floor on an alpha fit")
    return Depth(tiles, tiles * cap_tile, roof, disc, sat, binding)


# --------------------------------------------------------------------------
# Gates. The parent's `Gate` carries no statement of what a FAIL costs, and for
# this experiment that statement is the difference between "stop, the run is
# void" and "publish it, the ceiling is not where we said". So gates are wrapped
# rather than reused verbatim, and the three imported ones are adopted.
# --------------------------------------------------------------------------

@dataclass
class CapGate:
    tag: str
    kind: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    #: Read as "if this FAILS, <consequence>". Printed on a PASS too, so a
    #: reader knows what was at stake without re-deriving it.
    consequence: str
    lines: list[str] = field(default_factory=list)

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`'s vocabulary.

        This file has said UNDECIDED since it was written and the shared table
        says UNKNOWN. They are the same state -- the gate did not decide -- and
        the table's spelling wins at the boundary, because `classify` refuses a
        verdict it does not recognise rather than letting it fall through a
        comparison and be scored as whatever the fallthrough happened to be.
        Both count AGAINST the gate.

        The NAME is the tag, which is one token by construction, so the RESULT
        line a driver greps carries `V3` and `C2` rather than a sentence.
        """
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.tag,
                exit_codes.UNKNOWN if self.verdict == UNDECIDED else self.verdict)

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        Rendered by `moe.bench.exit_codes.result_line` and read back by
        `parse_result_lines`, anchored at column zero. The human `V3 VALIDITY
        PASS ...` line below it is for a reader and for
        `scripts/h200_gaps_session.sh`; both are kept because they have
        different readers, and only this one is the machine contract. Nothing
        else this file prints starts with `RESULT: `.
        """
        detail = (f"[{self.kind}] {self.claim} | measured {self.measured} "
                  f"| gate {self.threshold}")
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"{self.tag:3s} {self.kind:8s} {self.verdict:9s} {self.claim}",
               f"             measured {self.measured}   gate {self.threshold}",
               f"             if this FAILS: {self.consequence}"]
        out += [f"             {line}" for line in self.lines]
        return out

    def as_dict(self) -> dict:
        return {"tag": self.tag, "kind": self.kind, "claim": self.claim,
                "verdict": self.verdict, "measured": self.measured,
                "gate": self.threshold, "consequence": self.consequence}


def adopt(gate: SWEEP.Gate, tag: str, kind: str, consequence: str) -> CapGate:
    """Wrap a `block_m_crossing_sweep` gate without restating its logic.

    The imported gates are run on the same cells by the same code the pod run of
    the parent sweep uses. Re-implementing "did the step land at n x BLOCK_M"
    here would produce a second answer that agrees with the first until it does
    not, which is the failure this project keeps hitting.
    """
    return CapGate(tag, kind, gate.claim, gate.verdict, gate.measured,
                   gate.threshold, consequence, list(gate.lines))


def gate_v1_non_vacuity(cells, *, tiles, planned_cells: int,
                        aligned_needed: int = V1_ALIGNED_NEEDED) -> CapGate:
    """Did the run actually measure the grid it planned.

    A CHECK THAT EXAMINED NOTHING REPORTS NO FAILURES. Every gate below reads
    ladders, and a ladder assembled from four surviving cells out of ninety-six
    still fits, still reports an alpha and still passes. So the row counts are a
    gate: cells measured against cells planned, exactly-full tile stacks per
    tile, and failures named rather than dropped.

    Distinct `(BLOCK_M, tokens)` pairs, not rows: a cell that failed in an
    earlier session stays in `cells.csv` forever and is retried, so after a
    successful retry the file holds two rows for it and a raw row count would
    read as more work than happened.

    THE TREADS COUNTED HERE ARE THE TREADS THE LADDERS BELOW WILL HOLD, which
    means `SWEEP.ladder_treads` counts them and not a second walk over the raw
    cells. Until 2026-09-09 this gate counted every aligned cell that ran and
    reported 24 and 2 on the committed corpus while the ladders held 21 and 1:
    it over-reported by exactly the cells a drifting clock excludes, which is
    the one thing a gate whose stated job is "a check that examined nothing
    reports no failures" must not do. The excluded count is named on its own
    detail line, because this page printed the word `drift` nowhere at all
    while silently dropping 8 of 72 cells from every fit on it.
    """
    ok = {(c.block_m, c.tokens) for c in cells if c.status == "ok" and c.ms_p50 > 0}
    failed = [c for c in cells
              if c.status != "ok" and (c.block_m, c.tokens) not in ok]
    treads = {bm: SWEEP.ladder_treads(cells, bm) for bm in tiles}
    aligned = {bm: len(points) for bm, (points, _drifted) in treads.items()}
    drifted = {bm: n for bm, (_points, n) in treads.items()}
    detail = [f"{len(ok)} of {planned_cells} planned cells measured",
              "exactly-full tile stacks per setting: "
              + ", ".join(f"BM={bm}:{aligned[bm]}" for bm in tiles),
              "cells excluded because the clock DRIFTED across their own "
              "trials, and so absent from every ladder above: "
              + ", ".join(f"BM={bm}:{drifted[bm]}" for bm in tiles)
              + (" (none)" if not sum(drifted.values()) else ""),
              f"{len(failed)} cell(s) failed and were not recovered"]
    for c in failed[:5]:
        detail.append(f"  BM={c.block_m} T={c.tokens}: {c.detail}")
    if failed and any("shared memory" in (c.detail or "").lower() for c in failed):
        detail.append("  a shared-memory failure at the control is the known "
                      "BLOCK_SIZE_M=256 x 4-stage case: re-run the WHOLE thing "
                      "with --num-stages 3, or --control 128.")
    short = [bm for bm in tiles if aligned[bm] < aligned_needed]
    verdict = PASS if (len(ok) == planned_cells and not failed and not short) else FAIL
    return CapGate(
        "V1", VALIDITY, "the run measured the grid it planned", verdict,
        f"{len(ok)}/{planned_cells} cells, "
        + "/".join(str(aligned[bm]) for bm in tiles) + " aligned treads",
        f"all {planned_cells} cells and >= {aligned_needed} aligned treads per tile",
        "the ladders below were fitted on a grid with holes in it, and no roof "
        "fraction, alpha or ratio on this page may be quoted",
        detail)


#: Treads the sibling's walk needs before it will fit a ladder at all
#: (`compute_reference`'s own `if len(pts) < 3: continue`). Named here because
#: `walk_reached` has to reproduce which ladder that walk stopped at.
SWEEP_MIN_TREADS = 3


def walk_reached(cells, tiles: tuple[int, ...],
                 ref: SWEEP.ComputeReference) -> int | None:
    """Which ladder the sibling's walk actually stopped at, or None.

    `ComputeReference` NAMES ITS LADDER IN TWO OF THE SIBLING'S FOUR EXITS AND
    IN NEITHER OF THE OTHER TWO. A qualified reference carries `block_m`; a
    LEVEL refusal carries `refused_block_m`; a ladder refused on SHAPE
    (`c <= 0 or err > max_err`, block_m_crossing_sweep.py:1971) leaves BOTH at
    None, which is exactly what the "no ladder had the 3 treads" exit leaves
    behind too. The two are told apart by the fit error: the shape refusal
    carries the error that failed, the empty walk carries `math.inf`. Which
    ladder the shape refusal belongs to is then recoverable by walking the
    tiles the way the sibling does, largest first, because that refusal happens
    at the FIRST ladder with enough treads and does not fall through.

    2026-09-09: `control_reference` tested `control_tile in (ref.block_m,
    ref.refused_block_m)` instead, so a control refused on SHAPE looked like a
    control the walk never reached, and the wrapper replaced "not proportional
    to its tile count (32.7% mean error)" with "has 4 exactly-full tread(s)
    against the 3 a through-origin fit needs", which states a condition the
    control SATISFIES as its failure and throws away the number that did fail.
    """
    if ref.block_m is not None:
        return ref.block_m
    if ref.refused_block_m is not None:
        return ref.refused_block_m
    if ref.mean_rel_err == math.inf:
        return None
    for bm in sorted(tiles, reverse=True):
        if len(SWEEP.ladder_points(cells, bm)) >= SWEEP_MIN_TREADS:
            return bm
    return None                                          # pragma: no cover


def control_reference(cells, *, tiles: tuple[int, ...], control_tile: int, cfg,
                      ridge: float, bandwidth_gbps: float, b: int,
                      pinned: dict | None = None,
                      capability=None) -> SWEEP.ComputeReference:
    """Qualify the CONTROL ladder as the compute branch, or decline in its name.

    THE CAP TILE IS THE SUBJECT AND MUST NEVER CLASSIFY ITSELF. The sibling's
    `compute_reference` walks the block sizes from largest down and skips any
    ladder with fewer than three treads, which is right for a sweep of many
    tiles and wrong for a two-tile experiment where the smaller of the two is
    the thing being measured. On 2026-09-09 the H200 grid gave the control two
    treads, the walk fell through to BLOCK_M=16, fitted the cap tile's own
    ladder at 0.72% error and then refused it on non-vacuity at 1.044, a
    physically correct refusal since a per-tile slope equal to one full weight
    read IS alpha ~ 1, and the report printed "BLOCK_M=16 ... its LEVEL is
    wrong" for the subject of the experiment while the control's own numbers
    (C = 1.9902 ms/tile, 0.96% through-origin error, every level check passed)
    were never computed. Every tread at the cap tile is classified against the
    reference, so a reference taken from the cap tile makes the answer.

    EVERY SWEPT LADDER STILL TAKES PART IN THE LEVEL CHECKS, which is why
    `tiles` and not `(control_tile,)` goes to `_level_checks`: non-vacuity
    scales the candidate's slope to the SMALLEST swept block size, and telling
    the sibling that 256 is the smallest tile on the grid moves the H200
    control's vacuity ratio from 0.106 to 1.690 and refuses it. Candidacy is
    what is restricted here, not the comparison set.
    """
    ref = SWEEP.compute_reference(cells, tiles, cfg=cfg, ridge=ridge,
                                  bandwidth_gbps=bandwidth_gbps, b=b,
                                  pinned=pinned, capability=capability)
    reached = walk_reached(cells, tiles, ref)
    if reached == control_tile:
        # THE CONTROL'S OWN VERDICT, whichever of the three it is: qualified,
        # refused on its LEVEL, or refused on its SHAPE. It is returned
        # untouched because it already declines in the control's name and
        # carries the number that failed. Restating a shape refusal as a tread
        # count would print a condition the control SATISFIES as its failure.
        return ref
    treads = len(SWEEP.ladder_points(cells, control_tile))
    fell_through = (f", and the sweep's own walk reached BLOCK_M={reached} "
                    "next, which is the cap tile and is not a candidate"
                    if reached is not None else "")
    return SWEEP.ComputeReference(
        block_m=None, overhead_ms=0.0, slope_per_tile=None,
        mean_rel_err=math.inf,
        note=f"BLOCK_M={control_tile}, the control and the ONLY candidate, has "
             f"{treads} exactly-full tread(s) against the "
             f"{SWEEP_MIN_TREADS} a through-origin fit needs{fell_through}. No "
             "compute branch was qualified: membership falls back to a split "
             "search and NO alpha may decide a verdict")


def gate_v2_control(ref: SWEEP.ComputeReference, tp_control, *,
                    control_tile: int, noise: float,
                    issue: dict[int, float] | None = None) -> CapGate:
    """Is the control's ladder SHAPED like a compute branch.

    Two readings, neither of which can be satisfied by the sweep's own maximum:

      * PROPORTIONALITY, plus the sibling's LEVEL checks. `compute_reference`
        qualifies a ladder as a compute branch when `t = C n` through the origin
        fits it AND the slope is the right size. A memory-bound ladder `A + B n`
        carries a real intercept and misses the through-origin line by about 11%
        on this grid; a SPILLED kernel is proportional to its tile count too,
        which is how a slope 44x too steep once became this study's compute
        branch, and that is what the level checks are for.
      * FLATNESS. A ladder that has reached its roof stops gaining throughput
        per tread. The last tread's gain over the one before it is that, read
        directly off the cells with no fit in between.

    WHETHER IT REACHED THE ROOF IS A DIFFERENT QUESTION, and V3 asks it. A
    ladder can be proportional and flat at half the roof; this gate is about
    the SHAPE the reference is taken from, because every tread at the cap tile
    is classified against it.
    """
    # THE FIRST CONDITION IS AN IDENTITY, not a number: the reference has to BE
    # the control. On 2026-09-09 it was not, and `ref.mean_rel_err` then
    # belonged to whichever ladder the sibling tried last, the cap tile's own
    # 0.7%, so the FAIL line printed a number comfortably inside its printed
    # bound and named nothing that had failed. Say which ladder the reference
    # is on every line, passing or failing.
    proportional = ref.block_m == control_tile
    # THREE WAYS TO BE NOT-THE-CONTROL AND A NAME FOR EACH. The third arrived
    # 2026-09-09 with `control_reference`'s repair: a ladder refused on SHAPE
    # names itself in neither `block_m` nor `refused_block_m`, and until this
    # branch existed it printed as "none", which read as a sweep that had
    # qualified nothing when in fact it had refused the control by name.
    # `control_reference` is the only producer of what arrives here, and it
    # lets a both-None reference through ONLY when the control is the ladder
    # that was refused on shape; its own decline carries `math.inf`.
    which = (f"BLOCK_M={ref.block_m}" if ref.block_m is not None
             else f"BLOCK_M={ref.refused_block_m}, REFUSED on its level"
             if ref.refused_block_m is not None
             else f"BLOCK_M={control_tile}, REFUSED on its SHAPE at "
                  f"{ref.mean_rel_err:.1%} against a line through the origin"
             if ref.mean_rel_err < math.inf else "none")
    shape_refused = (ref.block_m is None and ref.refused_block_m is None
                     and ref.mean_rel_err < math.inf)
    gain = (tp_control[-1][1] / tp_control[-2][1] - 1.0
            if len(tp_control) >= 2 and tp_control[-2][1] > 0 else None)
    # The gain is a ratio of two single cells and so carries `sqrt(2)` times the
    # per-cell spread. Widening the gate with the spread is the parent sweep's
    # own rule for its memory-branch margin, and it is the difference between a
    # validity gate and a coin flip on a card that is not perfectly quiet.
    flat_gate = max(CONTROL_FLAT_GAIN, 3.0 * noise)
    lines = [f"compute reference: {which} (the only candidate is the control "
             f"BLOCK_M={control_tile}; the cap tile is the subject and may not "
             "classify itself)",
             f"  {ref.note}",
             "throughput per tread against ridge x bandwidth, "
             "fixed roof / the roof at the tread's own clock: "
             + tread_points(tp_control, issue)]
    if gain is None:
        lines.append("fewer than two treads at the control, so there is no "
                     "gain to read and flatness cannot be tested")
    else:
        lines.append(f"last tread gained {gain:+.2%} over the one before it, "
                     f"against a gate of +/-{flat_gate:.2%} = max(2%, 3 x the "
                     f"{noise:.2%} median per-cell spread)")
    if flat_gate > CONTROL_FLAT_GAIN_MAX:
        return CapGate(
            "V2", VALIDITY,
            f"the control BLOCK_M={control_tile} reached a compute roof in this grid",
            UNDECIDED, f"median per-cell spread {noise:.2%}",
            f"3 x spread must stay under {CONTROL_FLAT_GAIN_MAX:.0%}",
            "the timing is too noisy for flatness to mean anything, so the "
            "control was never shown to reach a roof and C1 may not be quoted",
            lines + ["widening the flatness gate to cover this spread would "
                     "accept a ladder still visibly climbing, so the gate stops "
                     "answering rather than answering yes. Re-run with more "
                     "--iters, or on a card that is not throttling."])
    flat = gain is not None and abs(gain) <= flat_gate
    verdict = PASS if (proportional and flat) else FAIL
    return CapGate(
        "V2", VALIDITY,
        f"the control BLOCK_M={control_tile} reached a compute roof in this grid",
        verdict,
        (f"proportional to {ref.mean_rel_err:.1%}" if proportional
         else f"the control BLOCK_M={control_tile} is NOT proportional: "
              f"{ref.mean_rel_err:.1%} against a line through the origin"
         if shape_refused else f"reference is {which}, not the control")
        + (f", last tread {gain:+.2%}" if gain is not None else ", no gain readable"),
        f"the reference IS BLOCK_M={control_tile}, its through-origin fit "
        f"within {PROPORTIONALITY_MAX_ERR:.0%}, and its last tread within "
        f"+/-{flat_gate:.2%}",
        "the compute branch every tread at the cap tile is classified against "
        "is not one, so neither C1 nor C2 may be quoted. Three ways in, and "
        "the lines above say which: the control's own LEVEL checks refused it "
        "and name themselves; its ladder is not proportional to its tile "
        "count, and the fit error that failed is quoted; or it is too short to "
        "qualify at all, which the plan now refuses before the pod is rented. "
        "Membership then falls back to a split search, which invents an alpha "
        "rather than declining to",
        lines)


def fused_layer_roof(roof_tflops: float) -> float:
    """`FUSED_ROOF_FLOOR x ridge x bandwidth`, in TFLOP/s. The ruler for V3.

    ONE FUNCTION SO THE PLAN AND THE GATE CANNOT DIVERGE. The plan registers the
    number a control must clear and the gate scores against it; the two reading
    different constants is a gate scored against a threshold registered for
    something else, which is the shape of half the findings in this repo.
    """
    if roof_tflops <= 0:
        raise Unmeasurable(
            f"ridge x bandwidth is {roof_tflops} TFLOP/s, so there is no dense "
            "peak to take a fused-layer fraction of")
    return FUSED_ROOF_FLOOR * roof_tflops


def issue_ladder(cells, block_m: int, roof_tflops: float,
                 reference_mhz: float | None) -> dict[int, float]:
    """`{tiles: useful throughput / the roof AT THAT CELL'S OWN CLOCK}`.

    R2'S SECOND NUMBER, AND NEVER A GATE INPUT. Every gate on this page scores
    against the FIXED roof, `ridge x bandwidth`, which is the dense GEMM's
    achieved figure at the clock the calibration measured it at. A cell that
    ran at another clock issued at another rate, so its fraction of the fixed
    roof mixes "how well this tile uses the machine" with "what clock the
    governor gave this tile under the 700 W cap". Dividing instead by
    `roof_at_clock` separates the two: the result is ISSUE EFFICIENCY, printed
    beside the fixed fraction so a reader can see both, and the fixed one is
    what any threshold is stated against.

    EMPTY, never a substitute, when the run has no reference clock: `--ridge`
    given on the command line is the operator's assertion and the calibration's
    clock does not describe it, and a `--self-test` replay reads no hardware at
    all. A tread whose row carried no under-load clock is simply absent from the
    mapping, and the point line prints "-" for it, because a missing second
    number is not a zero.
    """
    out: dict[int, float] = {}
    if not reference_mhz or roof_tflops <= 0:
        return out
    for c in cells:
        if c.block_m != block_m or not c.aligned or c.status != "ok":
            continue
        roof = roofline.roof_at_clock(roof_tflops, reference_mhz,
                                      getattr(c, "sm_clock_load_mhz", None))
        if roof:
            out[c.tiles_per_expert] = c.useful_tflops / roof
    return out


def tread_points(tp, issue: dict[int, float] | None) -> str:
    """`n=N:fixed/own` per tread, the FIXED fraction first.

    One formatter for all three ladder lines, so the two fractions cannot end
    up in a different order on different gates.
    """
    return ", ".join(
        f"n={n}:{v:.3f}/"
        + (f"{own:.3f}" if (own := (issue or {}).get(n)) is not None else "-")
        for n, v in tp)


def peak_pair(tp, issue: dict[int, float] | None) -> tuple[float, float | None]:
    """The best FIXED fraction, and the own-clock fraction OF THAT SAME TREAD.

    The same tread and not the best of each: the pair exists to say what one
    cell did against two rulers, and taking two maxima over different cells
    would print a ratio no cell ever had.
    """
    n, top = max(tp, key=lambda point: point[1])
    return top, (issue or {}).get(n)


def issue_suffix(own: float | None) -> str:
    """`, own-clock 0.577` for a gate's measured line, or nothing. R2: printed
    beside the fixed fraction, never scored against."""
    return f", own-clock {own:.3f}" if own is not None else ""


def gate_v3_control_roof(tp_control, *, control_tile: int, roof_tflops: float,
                         plateau: float, noise: float = 0.0,
                         issue: dict[int, float] | None = None) -> CapGate:
    """Did anything in this sweep reach a roof A FUSED LAYER CAN REACH.

    WHY THE DENSE PEAK IS THE WRONG RULER FOR A FUSED LAYER, which is what this
    gate used to be scored against and why it could not pass. `ridge x
    bandwidth` is the cuBLAS peak: one dense GEMM, nothing else in the interval.
    `fused_experts` is a gating matmul, `moe_align_block_size`, a scatter of
    tokens into per-expert row blocks, a GEMM, a SiLU-and-multiply, a second
    GEMM and `moe_sum` -- and the FLOPs in the numerator are counted from the
    two GEMMs alone. Everything else is time in the denominator with no work in
    the numerator, so a fused layer running perfectly still reports a fraction
    well under one. Across the 26 published reports the fused-layer plateau ran
    46.5-75.6% of the dense peak; no arm has exceeded 0.54 on the H200 or 0.64
    on the A100. Asking for 0.95 of the dense peak was therefore asking the
    fused layer to stop being a fused layer, and a VALIDITY gate that fails on
    every card that exists makes the arm INVALID by construction: nine pod
    minutes whose verdict was computable from the calibration before the pod was
    rented.

    THE RULER NOW, and it is two-sided so both branches are reachable.

      FLOOR, `FUSED_ROOF_FLOOR` of the dense peak. Below it nothing in the sweep
      is near any roof at all -- not the fused one either -- so the control
      never demonstrated that this apparatus can drive a ladder to a ceiling,
      and an absence at the cap tile is an absence recorded by an instrument
      never shown to detect a presence. FAIL.
      CEILING, the dense peak itself. A fused layer counting only its GEMM
      FLOPs cannot exceed it. Above it the ridge, the bandwidth or the FLOP
      count belongs to a different machine, which is a broken ruler and not a
      strong kernel, so the honest verdict is UNDECIDED: nothing here can be
      scored against a roof that is wrong.

    THE DENSE-PEAK FRACTION IS STILL PRINTED, labelled, because it is the number
    that compares across cards and against the sibling's gate 4. It is not the
    verdict, and the plateau is not the verdict either: the plateau is the
    maximum over the same cells, so a control read against IT scores 1.00 by
    construction and the check would examine nothing.

    WHAT A NON-PASS COSTS, precisely, AND WHAT IT DOES NOT SPARE. This is a
    VALIDITY gate, `moe.bench.exit_codes` maps a VALIDITY non-PASS to INVALID,
    and INVALID means nothing on the page may be quoted. That covers C2. The
    consequence text used to end at "C2 SURVIVES this", which contradicted the
    exit code the gate itself produces and would send a reader off to quote a
    C2 from a voided page.

    What is true of C2 is narrower and is about RECOVERY, not about reading.
    C1 compares a throughput with a roof, so it needs this run to have produced
    a throughput at a roof under the same conditions, and if the control did not
    there is no such throughput anywhere. C2's number never divides by the roof:
    it fits a re-read fraction from the cap tile's own treads and compares the
    ceiling that implies with the ridge. So re-running the CONTROL alone
    recovers C2, and the cap tile's cells do not have to be measured again --
    which is the whole reason C2 is in the report, and is a statement about what
    the next run must cost, not a licence to quote this one.
    """
    fused_roof = fused_layer_roof(roof_tflops)
    # The upper wall is widened by the measured spread for the same reason V2's
    # flatness gate is: `top` is a MAXIMUM over the control's treads, so it
    # carries the per-cell spread and carries it upward.
    ceiling = FUSED_ROOF_CEILING * (
        1.0 + max(FUSED_ROOF_CEILING_TOLERANCE, 3.0 * noise))
    threshold = (f"in [{FUSED_ROOF_FLOOR:.3f}, {ceiling:.3f}] of "
                 f"ridge x bandwidth, i.e. >= the fused-layer roof "
                 f"{fused_roof:.0f} TFLOP/s and <= the dense peak "
                 f"{roof_tflops:.0f} widened by max("
                 f"{FUSED_ROOF_CEILING_TOLERANCE:.0%}, 3 x the {noise:.2%} "
                 "per-cell spread)")
    consequence = (
        "nothing in this sweep reached a roof a fused layer can reach, so C1 "
        "is a statement about the instrument rather than about the tile. THE "
        "WHOLE PAGE IS UNQUOTABLE, C2 INCLUDED: this is a VALIDITY gate and a "
        "VALIDITY gate that is not PASS exits INVALID under "
        "moe.bench.exit_codes, which means nothing in the report may be "
        "quoted. What is true of C2 is narrower and is not a licence: C2's "
        "NUMBER is arithmetically independent of the roof -- it fits a re-read "
        "fraction from the cap tile's own treads and never divides by ridge x "
        "bandwidth -- so re-running the CONTROL is enough to recover it and the "
        "cap tile's cells need not be re-measured. Until that control is "
        "re-run, C2 is not a result")
    if not tp_control:
        return CapGate(
            "V3", VALIDITY,
            f"the control BLOCK_M={control_tile} reached the fused-layer roof",
            UNDECIDED, "no exactly-full tile stack at the control", threshold,
            consequence,
            ["The control ran no aligned cell, so there is no throughput to "
             "compare with any roof."])
    top, top_issue = peak_pair(tp_control, issue)
    lines = [
        f"the fused-layer roof is {FUSED_ROOF_FLOOR:.3f} x ridge x bandwidth = "
        f"{fused_roof:.0f} TFLOP/s. `ridge x bandwidth` = {roof_tflops:.0f} "
        "TFLOP/s is the DENSE cuBLAS peak and this layer is a gate, an "
        "alignment kernel, two GEMMs, a SiLU and a reduction with only the two "
        "GEMMs' FLOPs counted, so it is the wrong ruler for a control and is "
        "printed below as a diagnostic only",
        f"DIAGNOSTIC, not the verdict: peak {top:.3f} of the dense peak"
        + issue_suffix(top_issue) + ", "
        + f"against the {FUSED_PLATEAU_BAND[0]:.3f}-{FUSED_PLATEAU_BAND[1]:.3f} "
        "band the 26 published fused-layer reports occupy and the "
        f"{COMPUTE_BOUND_FRACTION:.2f} the sibling calls dense-compute-bound",
        f"the sweep's best useful throughput is {plateau:.1f} TFLOP/s, "
        f"{plateau / roof_tflops:.1%} of the dense peak. Against that plateau "
        "the control would score 1.00 by construction, which is why neither "
        "side of this gate is read against it",
        "throughput per tread against the dense peak, fixed roof / the roof "
        "at the tread's own clock: " + tread_points(tp_control, issue),
        "the second fraction is ISSUE EFFICIENCY and is not what this gate "
        "reads: the fixed roof is the dense GEMM's achieved figure at the "
        "clock the calibration measured it at, and it is the only denominator "
        "any threshold on this page is stated against"]
    if top > ceiling:
        return CapGate(
            "V3", VALIDITY,
            f"the control BLOCK_M={control_tile} reached the fused-layer roof",
            UNDECIDED,
            f"peak {top:.3f} of the dense peak" + issue_suffix(top_issue),
            threshold,
            consequence,
            lines + [f"The control is ABOVE the dense peak by "
                     f"{top - FUSED_ROOF_CEILING:.1%}, which a fused layer "
                     "counting only its two GEMMs' FLOPs cannot be. The ridge, "
                     "the bandwidth or the FLOP count is wrong, so nothing here "
                     "may be scored against this roof and the gate REFUSES "
                     "rather than reporting a control that beat physics. Check "
                     "that the ridge and the bandwidth came from the SAME card "
                     "as the cells."])
    verdict = PASS if top >= FUSED_ROOF_FLOOR else FAIL
    return CapGate(
        "V3", VALIDITY,
        f"the control BLOCK_M={control_tile} reached the fused-layer roof",
        verdict, f"peak {top:.3f} of the dense peak "
                 f"({top * roof_tflops:.0f} TFLOP/s)" + issue_suffix(top_issue),
        threshold, consequence, lines)


def gate_v4_depth(reached_tiles: int, depth: Depth, *, cap_tile: int,
                  alpha: float) -> CapGate:
    """Was the sweep deep enough for a PASS on C1 to exclude anything.

    Scored against the DISCRIMINATING horizon, which is the shallower of C1's
    two conditions and the one that must be live for C1 to rule anything out.
    The near-roof horizon is printed beside it, and C1 says for itself which of
    its conditions the depth reached made testable.
    """
    sat = saturation(reached_tiles, alpha) if reached_tiles else 0.0
    sat_retracted = saturation(reached_tiles, RETRACTED_ALPHA) if reached_tiles else 0.0
    verdict = PASS if depth.disc_condition_live(reached_tiles) else FAIL
    lines = [f"swept to {reached_tiles} tiles of {cap_tile} rows = "
             f"{reached_tiles * cap_tile} rows per expert",
             f"modelled AI is {sat:.1%} of its ceiling at alpha={alpha:.3f}, "
             f"and {sat_retracted:.1%} of it at alpha={RETRACTED_ALPHA}",
             f"C1's near-roof condition needs {depth.roof_tiles} tiles and is "
             + ("LIVE" if depth.roof_condition_live(reached_tiles)
                else "NOT TESTABLE at this depth: the retracted world would not "
                     "have got near the roof either, so passing it says nothing"),
             "a sweep short of the horizon PASSES C1 for the wrong reason: the "
             "retracted world simply had not got there yet",
             "required depth: " + depth.summary()]
    return CapGate(
        "V4", VALIDITY,
        "the sweep reached the depth at which the retracted world would trip C1",
        verdict, f"{reached_tiles} tiles", f">= {depth.disc_tiles} tiles",
        "the sweep stopped before the competing hypothesis would have shown "
        "itself, so C1 rules nothing out and its PASS may not be quoted",
        lines)


def gate_c1_roof_fraction(tp_cap, depth: Depth, *, cap_tile: int, alpha: float,
                          ridge: float, b: int, roof_tflops: float,
                          discriminator: float,
                          issue: dict[int, float] | None = None) -> CapGate:
    """THE CAP, read off the throughput with no fit in between.

    The highest fraction of `ridge x bandwidth` the cap tile ever reached.
    Nothing is fitted, so this survives every argument about branch membership,
    and the denominator is the ABSOLUTE roof rather than the sweep's own
    maximum, so the number means what the claim says and is comparable with the
    sibling sweep's gate 4.

    TWO CONDITIONS, SCORED ONLY WHERE THE DEPTH MADE THEM LIVE.

      NEAR THE ROOF: `<= 0.85`. The plain reading of "structurally incapable of
      reaching its compute roof".
      DISCRIMINATING: `<= the midpoint of the two worlds' predicted ceilings`.
      The reading that rules the retracted alpha out.

    A condition the retracted world could not have failed at this depth is not
    scored and is reported as NOT TESTABLE, because passing it would be a
    statement about where the sweep stopped. If neither is live the gate is
    UNDECIDED rather than PASS.
    """
    if not tp_cap:
        raise Unmeasurable(
            f"no exactly-full tile stack at BLOCK_M={cap_tile}: there is no "
            "throughput to take a roof fraction of, and 0.0 would read as "
            "'never got near the roof', which is the verdict this gate exists "
            "to earn")
    top, top_issue = peak_pair(tp_cap, issue)
    reached = max(n for n, _ in tp_cap)
    cap = SWEEP.ai_cap(cap_tile, alpha, b)
    cap_retracted = SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b)
    live = [(name, threshold) for name, threshold, is_live in (
        ("near-roof", ROOF_FRACTION, depth.roof_condition_live(reached)),
        ("discriminating", discriminator, depth.disc_condition_live(reached)))
        if is_live]
    lines = [f"predicted ceiling at alpha={alpha:.3f} is cap/ridge = {cap:.1f}/"
             f"{ridge:.1f} = {cap / ridge:.3f}",
             f"predicted ceiling at the retracted alpha={RETRACTED_ALPHA} is "
             f"{cap_retracted:.1f}/{ridge:.1f} = {cap_retracted / ridge:.3f}",
             f"the roof is ridge x bandwidth = {roof_tflops:.0f} TFLOP/s, not "
             "the sweep's own best, which would make the fraction a comparison "
             "with itself",
             "conditions: " + ", ".join(
                 f"{name} <= {threshold:.3f} "
                 + ("LIVE" if (name, threshold) in live else "NOT TESTABLE")
                 for name, threshold in (("near-roof", ROOF_FRACTION),
                                         ("discriminating", discriminator))),
             "throughput per tread against the roof, fixed roof / the roof at "
             "the tread's own clock: " + tread_points(tp_cap[-8:], issue)
             + (f"  (last 8 of {len(tp_cap)} treads)" if len(tp_cap) > 8 else ""),
             "the second fraction is ISSUE EFFICIENCY, printed beside the "
             "first and never scored: both conditions above are stated "
             "against the FIXED roof"]
    claim = f"BLOCK_M={cap_tile} never gets near the compute roof, at any batch"
    consequence = (f"BLOCK_M={cap_tile} DOES approach its compute roof, which "
                   "retracts the structural ceiling this study put on a "
                   "decode-tuned MoE kernel and is the publishable answer in "
                   "the other direction")
    if not live:
        return CapGate(
            "C1", CLAIM, claim, UNDECIDED,
            f"peak {top:.3f} of the roof" + issue_suffix(top_issue),
            "no condition is testable at this depth", consequence,
            lines + ["Neither threshold could have been tripped by the "
                     "retracted world at this depth, so a PASS would report "
                     "where the sweep stopped. Raise --r-max; V4 says to what."])
    verdict = PASS if all(top <= threshold for _, threshold in live) else FAIL
    return CapGate(
        "C1", CLAIM, claim, verdict,
        f"peak {top:.3f} of the roof" + issue_suffix(top_issue),
        " and ".join(f"<= {threshold:.3f} ({name})" for name, threshold in live),
        consequence, lines)


def gate_c2_measured_cap(fit, corrected: float | None, *, cap_tile: int,
                         ridge: float, b: int, discriminator: float,
                         cfg=None, block_n: int = 0) -> CapGate:
    """THE CAP, from the re-read fraction this ladder measures itself.

    `cap = 2 BM / (alpha b)` with alpha fitted on the cap tile's OWN treads,
    which is the one place in this study where alpha is cleanly identifiable: a
    tile that never crosses is memory bound at every tread, so there is no
    compute branch for the fit to run into and no import from another block
    size.

    THE STRUCTURAL THRESHOLD DOES NOT DISCRIMINATE, and the gate says so rather
    than taking credit for it. "No crossing exists" is `cap < ridge`, i.e.
    `alpha > 2 BM / (b ridge)`, which is 0.0983 at the H200's own 162.8 and
    0.1097 at the A100's 145.8 (0.0998 at the 160.3 the self-test pins, a
    module constant that is no card's calibration). The retracted alpha=0.10
    clears the H200's by 1.7% and the A100's not at all, so on the H200 both
    worlds predict no crossing and on the A100 the retracted one predicts a
    crossing; the printed NOTE beside the plan says which applies. So the gate
    is the MIDPOINT of the two worlds' predicted `cap/ridge`, computed from the
    two registered alphas at the ridge in use, and the structural comparison is
    printed beside it as the weaker statement it is.

    THE CAP IT SCORES IS A LIN CAP AND THE GATE SAYS BY HOW MUCH. `LadderFit`
    returns `B/(A+B)`, and `moe.bench.ai_model` shows that estimator returns

        alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                (EXA)

    on the three-term byte ladder, so `2 BM / (alpha_fitted b)` is the exact cap
    times `1 + phi + delta`. `ai_model.lin_overstatement` is that factor and the
    sibling's `cap_overstatement` brackets it over the unmeasured `alpha_a`; it
    is PRINTED beside the cap, and the exact cap it implies is printed under it.
    The gate is still scored on the LIN cap, deliberately: the overstatement
    runs the SAME way as the two measurement biases, so the scored number is an
    upper bound on the ceiling and C2's claim is that the ceiling is LOW. A gate
    scored on the exact cap would be easier to pass, and the factor is printed
    so a reader can see how much easier.
    """
    if fit is None or corrected is None or fit.memory_points < MIN_MEMORY_TREADS:
        treads = fit.memory_points if fit is not None else 0
        return CapGate(
            "C2", CLAIM,
            f"the re-read fraction measured at BLOCK_M={cap_tile} puts its AI "
            "ceiling far below the ridge",
            UNDECIDED, f"alpha from {treads} memory-bound tread(s)",
            f"needs >= {MIN_MEMORY_TREADS}",
            "the mechanical half of the cap claim is unmeasured and only C1 "
            "carries it",
            [f"BLOCK_M={cap_tile} should be memory bound at EVERY tread, so "
             "too few of them is itself a finding and not a shrug.",
             f"ladder basis: {fit.basis if fit is not None else 'no ladder'}",
             "THE READING THAT MATTERS. `B/C = ridge/cap` exactly, so a memory "
             "branch running parallel to the compute branch -- which is what "
             "the discard message above says when it appears -- is a tile "
             "sitting ON its own crossing. That is precisely the retracted "
             f"world at BLOCK_M={cap_tile}: cap "
             f"{SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b):.1f} against ridge "
             f"{ridge:.1f} is a ratio of "
             f"{ridge / SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b):.3f}, inside "
             f"the {SWEEP.PARALLEL_BRANCH_TOLERANCE:.0%} tolerance, so the fit "
             "refuses to name an alpha rather than inventing one. Read C1 with "
             "that in mind: an UNDECIDED here alongside a C1 FAIL is the "
             "retracted world's signature, not a broken instrument.",
             "The other two causes are a mis-scaled compute reference and a "
             "ladder that is not the one this run thinks it is. Read V1 and V2 "
             "before reading any of this as a property of the hardware."])
    cap = SWEEP.ai_cap(cap_tile, corrected, b)
    structural = 2.0 * cap_tile / (b * ridge)
    verdict = PASS if cap / ridge <= discriminator else FAIL
    # R8's label, adopted from the sibling rather than recomputed: one function
    # owns "how much is a LIN cap high by" and both scripts print the same
    # number. `cfg` is optional only so a caller with no model config still gets
    # a gate; when it is absent the gate SAYS the factor is unstated rather than
    # printing a cap as if it were exact.
    exa_lines: list[str]
    if cfg is not None and block_n:
        lo, hi = SWEEP.cap_overstatement(cfg, cap_tile, block_n, b)
        exa_lines = [
            "EXA: a B/(A+B) ladder fit returns (alpha_b + phi)/(1 + phi + "
            "delta), so the cap above is the EXACT cap times "
            f"ai_model.lin_overstatement = 1 + phi + delta = {lo:.3f}-{hi:.3f} "
            "here. A BRACKET and not a number, because alpha_a -- the miss "
            "fraction on the ACTIVATION re-read -- has no measurement anywhere "
            "in this repository: the ends are alpha_a = 0 and alpha_a = 1 with "
            "delta taken as zero, so both are LOWER bounds on the "
            "overstatement, and the run's own overhead_ms is the measured "
            "stand-in for delta",
            f"exact cap implied: {cap / hi:.1f}-{cap / lo:.1f} Op/B = "
            f"{cap / (hi * ridge):.3f}-{cap / (lo * ridge):.3f} of ridge "
            f"{ridge:.1f}. The gate is scored on the LIN cap {cap:.1f}, which "
            "is the HIGHER number and therefore the harder bar for a claim that "
            "the ceiling is low",
        ]
    else:
        exa_lines = ["EXA: the lin_overstatement factor 1 + phi + delta is NOT "
                     "stated here because no model config reached this gate, so "
                     "the cap above is a LIN cap of unquantified excess. Do not "
                     "quote it as a bound."]
    return CapGate(
        "C2", CLAIM,
        f"the re-read fraction measured at BLOCK_M={cap_tile} puts its AI "
        "ceiling far below the ridge",
        verdict, f"cap/ridge = {cap:.1f}/{ridge:.1f} = {cap / ridge:.3f}",
        f"<= {discriminator:.3f}, the midpoint of the two worlds",
        f"the measured re-read fraction is small enough to put BLOCK_M="
        f"{cap_tile} within reach of the roof, which is the retracted world's "
        "prediction and not this study's",
        exa_lines
        + [f"alpha {corrected:.3f} activation-corrected ({fit.alpha:.3f} raw) "
         f"over {fit.memory_points} memory-bound treads, fit error "
         f"{fit.mean_rel_err:.2%}",
         f"both biases run the safe way: the fixed cost sits in alpha's "
         f"denominator and the activation correction removes traffic from its "
         f"numerator, so {corrected:.3f} is a LOWER bound on alpha and "
         f"{cap:.1f} is an UPPER bound on the ceiling",
         f"the structural claim -- no crossing at all -- is cap < ridge, i.e. "
         f"alpha > {structural:.4f}, and it is "
         f"{'MET' if corrected > structural else 'NOT met'}. It is printed and "
         f"not gated on: alpha={RETRACTED_ALPHA} also clears it, so it cannot "
         "tell the two worlds apart",
         f"per-tile slopes: memory B={fit.slope_memory:.4f} ms against compute "
         f"C={fit.compute_slope:.4f} ms; C > B is the condition for a crossing "
         f"to exist and it is "
         f"{'MET' if (fit.compute_slope or 0) > (fit.slope_memory or 0) else 'NOT met'}"
         if fit.slope_memory and fit.compute_slope else
         "one of the two branch slopes is missing, so the slope form of the "
         "crossing condition cannot be stated"])


def slope_relative_se(ys, spread: float) -> float:
    """Relative sd of an OLS slope through `(1..N, ys)` when each point carries
    a RELATIVE spread of `spread`.

    THE TEXTBOOK FORM IS THE WRONG ONE HERE and the difference is three orders
    of magnitude, so it is worth the six lines. `sd(B) = sigma / sqrt(Sxx)`
    assumes every point carries the same ABSOLUTE sd. A timing ladder does not:
    tread 8 takes eight times as long as tread 1 and carries eight times the
    absolute jitter at the same relative spread, and the treads that most
    constrain a slope are exactly the far ones. So the errors are propagated as
    `sd_i = spread * y_i` through the OLS weights,

        sd(B) = spread * sqrt(sum_i c_i^2 y_i^2),   c_i = (x_i - xbar) / Sxx

    which is exact for this estimator on this error model. Returned RELATIVE to
    the fitted slope, because everything downstream of it is a ratio.

    REFUSES below two treads and on a flat ladder: one point is not a slope, and
    a slope of zero has no relative anything.
    """
    n = len(ys)
    if n < 2:
        raise Unmeasurable(
            f"{n} tread(s): a slope needs two points, so there is no standard "
            "error to state an MDE from")
    if spread <= 0:
        raise Unmeasurable(
            f"--plant-noise {spread}: an MDE is a multiple of a standard "
            "deviation, and at a spread of zero every effect is detectable, "
            "which is a statement about the planted world and not about any "
            "pod. State the spread you believe the card has.")
    xs = list(range(1, n + 1))
    xbar = sum(xs) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    ybar = sum(ys) / n
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sxx
    if slope <= 0:
        raise Unmeasurable(
            "the modelled ladder has a non-positive slope, so there is no "
            "per-tile cost for an MDE to be a fraction of")
    var = sum((((x - xbar) / sxx) * spread * y) ** 2 for x, y in zip(xs, ys, strict=True))
    return math.sqrt(var) / slope


def mde_lines(cfg, *, cap_tile: int, spread: float, alpha: float, ridge: float,
              b: int, bandwidth_gbps: float, treads: int,
              discriminator: float) -> list[str]:
    """The minimum detectable effect of every gate that has one, before the run.

    HOEFLER & BELLI RULE 12, AND THE FINDING THAT PUT IT HERE. Not one arm in
    this study stated an MDE, and two of them carry thresholds tighter than the
    noise the pod delivers: a gate whose threshold sits inside its own MDE
    cannot decide anything, and it does not say so, because it prints a
    threshold either way. Every number below is derived from ONE stated
    assumption -- `--plant-noise`, defaulting to the published per-cell spread
    -- and the assumption is printed first so a reader can see it is an
    assumption and not a measurement.

    Three designs, because the three gates are three different statistics.

      C1 is one measured cell's roof fraction against a REGISTERED threshold,
      which carries no sampling error of its own. Known-sigma, one sample:
      `(z(1-a/2) + z(power)) sigma`. `mde_external_sigma` at two nominal
      conditions is that quantity, its `sqrt(2/n)` factor being exactly 1
      there; it is called rather than restated so the quantiles come from one
      place.
      C2 is a cap from an OLS slope over the cap tile's own treads, so its
      spread is `slope_relative_se` propagated through `cap ~ 1/alpha`. With
      the level `A + B` held fixed, `d(alpha)/alpha = (1 - alpha) dB/B`, and
      `cap` is `1/alpha` times a constant, so the cap inherits that fraction.
      V2 is a RATIO OF TWO SINGLE CELLS, so it carries the two-condition form
      at one replicate each: `mde_external_sigma(spread, 1)`, about 3.96 sigma.

    EVERY LINE IS PRINTED IN THE SAME UNITS AS THE GATE IT IS ABOUT. An MDE
    quoted relative beside a threshold stated absolute is the unit confusion
    that made a 0.307-versus-0.311 agreement in this study an artefact, so each
    line below carries the relative figure, the absolute figure at the refit
    world's own prediction, and the distance to the threshold in MDE units.
    """
    if spread <= 0:
        raise Unmeasurable(
            f"--plant-noise {spread}: an MDE is a multiple of a standard "
            "deviation, and at a spread of zero every effect is detectable, "
            "which is a statement about the planted world and not about any "
            "pod. State the spread you believe the card has.")
    z_one = NOISE.mde_external_sigma(spread, 2)   # (z_a + z_b) * sigma
    z_two = NOISE.mde_external_sigma(spread, 1)   # the same, times sqrt(2)
    cap_refit = SWEEP.ai_cap(cap_tile, alpha, b) / ridge
    cap_retracted = SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b) / ridge
    ys = [SWEEP.model_ms(cfg, n * cap_tile, cap_tile, alpha=alpha, ridge=ridge,
                         bandwidth_gbps=bandwidth_gbps, b=b)
          for n in range(1, treads + 1)]
    slope_rel = slope_relative_se(ys, spread)
    cap_rel = (1.0 - alpha) * slope_rel
    c1_abs = z_one * cap_refit
    c2_abs = z_one * cap_rel * cap_refit
    flat_gate = max(CONTROL_FLAT_GAIN, 3.0 * spread)
    return [
        "",
        "MINIMUM DETECTABLE EFFECT, from ONE stated assumption and no other",
        f"  assumed per-cell relative timing spread {spread:.2%} "
        "(--plant-noise; the published H200 ladders run 0.76-1.82% and the "
        f"A100 ones 0.48-0.61%, so the default {PUBLISHED_CELL_SPREAD:.2%} is "
        "near the top of the range a gate has to survive, not a floor)",
        f"  two-sided {NOISE.TEST_LEVEL:.0%} at {NOISE.TEST_POWER:.0%} power "
        "throughout, which is the convention every MDE in this repository uses",
        f"  C1  peak roof fraction, ONE cell against a registered threshold: "
        f"MDE {z_one:.2%} relative = {c1_abs:.4f} of the roof at the refit "
        f"world's predicted ceiling {cap_refit:.3f}. The discriminating "
        f"threshold {discriminator:.3f} sits "
        f"{abs(discriminator - cap_refit) / c1_abs:.0f}x that away, and the "
        f"retracted world's ceiling {cap_retracted:.3f} sits "
        f"{abs(cap_retracted - cap_refit) / c1_abs:.0f}x away. C1 is not the "
        "gate that runs out of power",
        f"  C2  cap/ridge from an alpha fitted over {treads} treads: the "
        f"slope's relative sd is {slope_rel:.3%} (heteroscedastic OLS, sd_i = "
        f"spread x t_i), so the cap's is {cap_rel:.3%} and the MDE is "
        f"{z_one * cap_rel:.3%} relative = {c2_abs:.2e} of the ridge. The "
        f"discriminator sits {abs(discriminator - cap_refit) / c2_abs:.0f}x "
        "that away. THIS IS THE OPTIMISTIC END and is stated as one: it "
        "assumes every tread is on the memory branch and the two-line model is "
        "exactly right, so it bounds the TIMING noise and not the model error, "
        "which at this tile height is what actually limits the fit",
        f"  V2  flatness, a RATIO OF TWO SINGLE CELLS and so sqrt(2) times the "
        f"per-cell spread: MDE {z_two:.2%}, against a gate of max("
        f"{CONTROL_FLAT_GAIN:.0%}, 3 x spread) = {flat_gate:.2%}. "
        + ("THE GATE IS INSIDE ITS OWN MDE at this spread: a true gain between "
           f"{flat_gate:.2%} and {z_two:.2%} would be missed more often than "
           "not. That is why V2 REFUSES (UNDECIDED) once 3 x spread passes "
           f"{CONTROL_FLAT_GAIN_MAX:.0%} instead of widening further"
           if flat_gate < z_two else
           "the gate is wider than the MDE, so a true gain past it is seen"),
    ]


def cap_discriminator(cap_tile: int, ridge: float, b: int) -> float:
    """The threshold C1 and C2 share: the midpoint of the two worlds' `cap/ridge`.

    Computed from the REGISTERED alphas rather than typed in, so it cannot drift
    away from the values the rest of the file argues about, and so a reader can
    see the gate sitting between two predictions instead of beside one.

    IT TAKES NO ALPHA ARGUMENT, and that is the fix for a real bug rather than
    an aesthetic. It used to accept the run's alpha, `analyse` passed the one it
    was called with, and under `--self-test 0.10` the threshold came out at
    0.998 -- the midpoint of the retracted world with itself. A gate that moves
    with the hypothesis under test is a prediction adjusted after seeing the
    data, and there is no way to notice from the printed line, which reports a
    threshold either way.
    """
    return (SWEEP.ai_cap(cap_tile, ALPHA, b)
            + SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b)) / (2.0 * ridge)


# --------------------------------------------------------------------------
# The plan and the predictions. Printed BEFORE the sweep runs, and again inside
# the report, so "registered before the run" is a property of the transcript and
# not of a comment.
# --------------------------------------------------------------------------

def prediction_lines(cfg, *, cap_tile: int, control_tile: int, alpha: float,
                     ridge: float, b: int, bandwidth_gbps: float, depth: Depth,
                     r_max: int,
                     ridge_band: tuple[float, float] | None = None,
                     ridge_source: str = "", band_source: str = "") -> list[str]:
    # THE BAND IS THE ATTACHED CARD'S, NOT THE MODULE'S. Every crossing column
    # below is `2 BM / (alpha b)` compared with a ridge, so a prediction table
    # printed at 160.3 on a card that calibrates at 145.7 registers a
    # prediction about neither machine. Unstated means DEGENERATE -- this run's
    # own ridge twice over, which is honest about being one calibration -- and
    # never the module band, which is how 160.3 reached seven A100 reports.
    lo, hi = (ridge, ridge) if ridge_band is None else (min(ridge_band),
                                                        max(ridge_band))
    tiles = (cap_tile, control_tile)
    preds_lo = SWEEP.predictions(tiles, alpha, lo, b)
    preds_hi = SWEEP.predictions(tiles, alpha, hi, b)
    retr = SWEEP.predictions(tiles, RETRACTED_ALPHA, lo, b)
    out = ["", "PREDICTIONS, registered before the run and printed before any "
                "measurement",
           f"  alpha {alpha:.3f} (band {ALPHA_BAND[0]}-{ALPHA_BAND[1]}) against "
           f"the retracted {RETRACTED_ALPHA}, ridge band "
           f"{lo}-{hi} Op/B, {b} bytes per element",
           f"  ridge       {ridge:.2f} Op/B, "
           + (ridge_source or "source not stated"),
           f"  ridge band  {lo:.2f}-{hi:.2f} Op/B, "
           + (band_source or "source not stated"),
           f"  BLOCK_M   AI cap   crossing @{lo:<6.1f}      crossing @{hi:<6.1f}"
           f"      retracted @{lo:<6.1f}"]

    def where(pred):
        if pred.crossing_rows is None:
            return "NO CROSSING EVER    "
        tok = pred.crossing_tokens(cfg.num_experts, cfg.top_k)
        return (f"r={pred.crossing_rows:6.1f} T={tok:6.0f} n="
                f"{pred.first_compute_tread}")

    for bm in tiles:
        out.append(f"  {bm:7d} {preds_lo[bm].ai_cap:8.1f}   {where(preds_lo[bm])}  "
                   f" {where(preds_hi[bm])}   {where(retr[bm])}")

    top = (r_max // math.lcm(cap_tile, control_tile)) * math.lcm(cap_tile, control_tile)
    out.append("")
    out.append("  the two worlds at this grid's depth, which is what every gate "
               "below discriminates on")
    for label, a in (("refit    ", alpha), ("retracted", RETRACTED_ALPHA)):
        cap = SWEEP.ai_cap(cap_tile, a, b)
        sat = saturation(r_max // cap_tile, a)
        ratio = (SWEEP.model_ms(cfg, top, cap_tile, alpha=a, ridge=ridge,
                                bandwidth_gbps=bandwidth_gbps, b=b)
                 / SWEEP.model_ms(cfg, top, control_tile, alpha=a, ridge=ridge,
                                  bandwidth_gbps=bandwidth_gbps, b=b))
        out.append(
            f"    alpha={a:<5.3f} {label}  cap {cap:6.1f} Op/B = "
            f"{cap / ridge:.3f} of ridge {ridge:.1f}   AI at the last tread is "
            f"{sat:.1%} of it   ms(BM={cap_tile})/ms(BM={control_tile}) at "
            f"r={top} is {ratio:.3f}x")
    disc = cap_discriminator(cap_tile, ridge, b)
    out.append(f"    C1 fails above {ROOF_FRACTION:.2f} of ridge x bandwidth "
               f"(near-roof) or above {disc:.3f} of it (discriminating), each "
               f"scored only where the depth makes it live; C2 fails above "
               f"{disc:.3f} of the ridge; C3 fails below "
               f"{SWEEP.GATE2_RATIO:.2f}x")
    out.append("")
    out += depth.lines(cfg, cap_tile)
    out.append("")
    retracted_cap = SWEEP.ai_cap(cap_tile, RETRACTED_ALPHA, b)
    out.append("  NOT A READOUT: whether a crossing appears at BLOCK_M="
               f"{cap_tile}. On the H200 band both worlds say none does "
               f"({retracted_cap:.1f} against ridge {lo} is "
               f"{abs(retracted_cap - lo) / lo:.1%} of headroom), so the "
               "presence or absence of one separates nothing there. How CLOSE "
               "the tile gets is the readout, and that is C1."
               + ("" if retracted_cap <= lo else
                  f"  NOTE: at THIS band's low end {lo} the retracted ceiling "
                  f"{retracted_cap:.1f} is ABOVE the ridge, so on this card the "
                  "retracted world does predict a crossing and its absence is "
                  "informative as well."))
    out.append(f"  NOT A PRODUCTION CLAIM: {observed_note(cap_tile)}. This run "
               "tests the CAP FORMULA at this tile height. BLOCK_M=128 is the "
               "regime that carries the production claim and "
               "scripts/bm128_depth.py is where it is asked.")
    out.append("  NOT RUN: the parent sweep's gate 3 (the fitted alpha's "
               f"interval against ALPHA_BAND [{SWEEP.ALPHA_BAND[0]}, "
               f"{SWEEP.ALPHA_BAND[1]}] in BOTH directions; its retired "
               f"one-sided {SWEEP.GATE3_ALPHA_DISCRIMINATOR:.2f} threshold is "
               "printed there, not scored) is superseded here by C2, which "
               "fits the same re-read fraction and then does the thing the "
               "cap claim needs: converts it to a ceiling, prints the EXA "
               "factor that ceiling is high by, and compares it with the "
               "ridge.")
    return out


def plan_lines(cfg, args, *, tiles, grid, depth: Depth, b: int,
               bandwidth_gbps: float, bw_source: str, out_dir: Path,
               pinned: dict, run_id: str, resources, card: str,
               git_note: str) -> list[str]:
    aligned = {bm: [r for r in grid if r % bm == 0] for bm in tiles}
    lines = [
        f"experiment  tile_cap / {run_id}",
        f"model       {args.model} E={cfg.num_experts} k={cfg.top_k}  "
        f"{args.dtype} ({b} bytes)",
        f"tiles       cap BLOCK_M={tiles[0]}, positive control BLOCK_M={tiles[1]}",
        f"pinned      {pinned}",
        f"grid        {len(grid)} rows-per-expert x {len(tiles)} tiles = "
        f"{len(grid) * len(tiles)} cells",
        f"            r in [{grid[0]}, {grid[-1]}], T in "
        f"[{SWEEP.tokens_for_rows(cfg, grid[0])}, "
        f"{SWEEP.tokens_for_rows(cfg, grid[-1])}], token step "
        f"{SWEEP.rows_step(cfg)}",
        "            exactly-full tile stacks (all a ladder fit can read): "
        + ", ".join(f"BM={bm}:{len(aligned[bm])}" for bm in tiles),
        f"bandwidth   {bandwidth_gbps:.1f} GB/s, {bw_source}",
        # The card is printed and is in the run id above. Every verdict here is
        # scored against a per-card ridge, and $MOE_RESULTS_DIR is a volume two
        # pods share, so a report that does not name its card is unrecoverable
        # afterwards.
        f"card        {card}"
        + ("   (no CUDA device: this is a plan or a replay, not a measurement)"
           if card == NO_CARD_SLUG else ""),
        f"WRITES TO   {out_dir}",
        "            cells.csv (appended per cell), report.txt, report.json, "
        "triton-cache/",
        # ASKED, not asserted. `git check-ignore` is run on the actual --out,
        # because the answer differs for results/, results/published/ and a
        # path outside the work tree, and a sentence copied out of .gitignore
        # is right for exactly one of them.
        f"git         {git_note}",
        "resources    one CTA's bill under the pinned constants, so a setting "
        "that cannot physically run is refused here and not diagnosed from its "
        "timing afterwards:",
    ]
    lines += [resources[bm].render() for bm in tiles]
    lines.append(
        f"            deepest exactly-full stack at BLOCK_M={tiles[0]}: "
        f"{grid_depth(grid, tiles[0])} tiles, against the {depth.tiles} V4 "
        "requires")
    # A SHORT GRID IS A REFUSAL AND NOT A WARNING; `grid_refusal` states it and
    # `_main` returns REFUSED on it, right after this plan is printed. Until
    # 2026-09-09 this line read "WARNING: ... Raise --r-max" and the run went on
    # to measure 72 cells against gates the grid had already made unsatisfiable.
    return lines


def grid_depth(grid, cap_tile: int) -> int:
    """Tiles in the deepest exactly-full stack of `cap_tile` on this grid."""
    aligned = [r for r in grid if r % cap_tile == 0]
    return max(aligned) // cap_tile if aligned else 0


def grid_refusal(grid, *, tiles: tuple[int, ...], depth: Depth,
                 row_step: int) -> str:
    """Empty when the grid can satisfy V1 and V4, else why it cannot.

    ASKED BEFORE THE POD IS RENTED, because every gate below reads ladders and
    a ladder needs treads: a grid that cannot put three exactly-full stacks on
    each tile has decided V1 FAIL before the first cell is timed, and one whose
    deepest cap stack is short of the horizon has decided V4 the same way. On
    2026-09-09 both were true of the H200 grid, the plan printed "BM=256:2" and
    a WARNING, and 72 measured cells produced nothing quotable.
    """
    stacks = {bm: sum(1 for r in grid if r % bm == 0) for bm in tiles}
    short = {bm: n for bm, n in stacks.items() if n < V1_ALIGNED_NEEDED}
    deepest = grid_depth(grid, tiles[0])
    if not short and deepest >= depth.tiles:
        return ""
    why = ["exactly-full tile stacks on this grid: "
           + ", ".join(f"BLOCK_M={bm}:{stacks[bm]}" for bm in tiles)]
    why += [f"  BLOCK_M={bm} has {n} exactly-full stack(s) against the "
            f"{V1_ALIGNED_NEEDED} V1 requires"
            for bm, n in sorted(short.items())]
    if deepest < depth.tiles:
        why.append(f"  the deepest BLOCK_M={tiles[0]} stack is {deepest} tiles "
                   f"against the {depth.tiles} V4 requires")
    needed = max(V1_ALIGNED_NEEDED * max(tiles), depth.rows,
                 CONTROL_STACKS_FLOOR * max(tiles))
    needed += (-needed) % row_step
    why.append(f"  raise --r-max to at least {needed}, or lower --row-step so "
               f"more multiples of {max(tiles)} land on the grid")
    return "\n".join(why)


# --------------------------------------------------------------------------
# The analysis. Pure: cells in, report out. No GPU, no I/O, so `--self-test`
# and the test suite exercise exactly what the pod run prints.
# --------------------------------------------------------------------------

@dataclass
class Report:
    lines: list[str]
    gates: list[CapGate]
    payload: dict

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def analyse(cells, cfg, *, cap_tile: int, control_tile: int, alpha: float,
            ridge: float, bandwidth_gbps: float, b: int, model_name: str,
            dtype: str, compiles: dict, executed: dict, sm_count: int,
            sm_source: str, depth: Depth, planned_cells: int,
            header: list[str], pinned: dict | None = None,
            capability: tuple[int, int] | None = None,
            ridge_band: tuple[float, float] | None = None,
            ridge_source: str = "", band_source: str = "",
            card: str = NO_CARD_SLUG, ridge_device: str = "",
            synthetic: bool = False, prov=None,
            reference_mhz: float | None = None,
            reference_clock_source: str = "") -> Report:
    # `ridge_band` IS NOT DEFAULTED TO `RIDGE_BAND`, which is one machine's
    # 2026-08-26 calibration and is exactly how all 7 published A100 reports
    # came to carry a band belonging to neither card. Unstated gives this run's
    # own ridge twice over -- a degenerate band is honest about being one
    # calibration, a borrowed one is not -- and `report.json` carries the source
    # string beside it either way.
    ridge_band = (ridge, ridge) if ridge_band is None else (min(ridge_band),
                                                            max(ridge_band))
    tiles = (cap_tile, control_tile)
    ok = [c for c in cells if c.status == "ok" and c.ms_p50 > 0]
    aligned = [c for c in ok if c.aligned]
    if not aligned:
        raise Unmeasurable(
            "not one exactly-full tile stack survived. Every quantity below is "
            "read off aligned cells, a partially-filled tread reports a "
            "throughput that depends on where in the tread it was sampled, and "
            "there is no honest substitute for the measurement.")
    plateau = max(c.useful_tflops for c in aligned)
    if plateau <= 0:
        raise Unmeasurable(
            "the best useful throughput in the sweep is 0 TFLOP/s, so every "
            "roof fraction would be a division by zero. Cells exist but carry "
            "no time; read cells.csv before believing the hardware.")
    noise = statistics.median([c.rel_spread for c in ok])
    roof_tflops = ridge * bandwidth_gbps * 1e9 / 1e12
    if roof_tflops <= 0:
        raise Unmeasurable(
            f"ridge {ridge} x bandwidth {bandwidth_gbps} GB/s is not a positive "
            "compute roof, so every roof fraction below would be a division by "
            "zero. Give --ridge and --bandwidth-gbps, or calibrate the box.")

    # The sibling's LEVEL checks, which is why this carries `pinned` and the
    # capability: a compute branch with the right SHAPE and the wrong LEVEL --
    # a spilled kernel is proportional to its tile count too -- once became this
    # study's reference at 44x too steep, and every ladder in the report is
    # classified against it. A reference that is refused there makes C2
    # UNDECIDED here, which is the honest outcome and not a hole.
    #
    # THE CONTROL IS THE ONLY CANDIDATE; see `control_reference`.
    ref = control_reference(ok, tiles=tiles, control_tile=control_tile, cfg=cfg,
                            ridge=ridge, bandwidth_gbps=bandwidth_gbps, b=b,
                            pinned=pinned, capability=capability)
    # Same margin rule the parent uses: the reference slope carries the timing
    # spread too, and a compute branch estimated 2% low makes every
    # compute-bound tread look memory bound.
    margin = max(SWEEP.MEMORY_BRANCH_MARGIN, 3.0 * noise)
    fits = {bm: SWEEP.fit_ladder(SWEEP.ladder_points(ok, bm), bm, ref, margin)
            for bm in tiles}
    # THE DENOMINATOR IS `ridge x bandwidth`, NOT THE PLATEAU. Against the
    # sweep's own maximum some block size always scores 1.00 -- the plateau IS
    # that maximum -- so a control read that way could never fail and the check
    # would examine nothing. The sibling's `bracketing` docstring records the
    # cost of getting this wrong: across 26 published reports the plateau ran
    # 46.5-75.6% of the card's own roof, so nothing in any of them reached one.
    #
    # `_throughput_ladder` is private to the sibling and used anyway, on
    # purpose: a second implementation of "roof fraction per tread" is exactly
    # the kind of duplicate that agrees with the original until it does not.
    tp_cap = SWEEP._throughput_ladder(ok, cap_tile, roof_tflops)
    tp_control = SWEEP._throughput_ladder(ok, control_tile, roof_tflops)
    # R2's second number, beside every fraction above and inside none of them.
    issue_cap = issue_ladder(ok, cap_tile, roof_tflops, reference_mhz)
    issue_control = issue_ladder(ok, control_tile, roof_tflops, reference_mhz)

    fit_cap = fits.get(cap_tile)
    corrected = None
    if fit_cap is not None and fit_cap.alpha is not None and fit_cap.load_ms:
        corrected = ((fit_cap.slope_memory
                      - SWEEP.activation_slope_ms(cfg, cap_tile, bandwidth_gbps))
                     / fit_cap.load_ms)

    lines = list(header)
    lines.append("")
    lines.append(f"MEASURED  {model_name} {dtype}  {len(ok)} cells, "
                 f"{len(aligned)} of them exactly-full tile stacks")
    lines.append(f"  compute plateau {plateau:.1f} TFLOP/s useful, which is "
                 f"{plateau / roof_tflops:.1%} of ridge x bandwidth "
                 f"({roof_tflops:.0f} TFLOP/s)")
    lines.append("  every SCORED roof fraction below is against ridge x "
                 "bandwidth and NOT against that plateau: the plateau is the "
                 "maximum over the same cells, so a control read against it "
                 "scores 1.00 by construction. Far below 100% means nothing in "
                 "the sweep reached a roof, which is what V3 tests. A second, "
                 "UNSCORED fraction is printed beside it; the next line says "
                 "what it is")
    lines.append(f"  per-cell timing spread, median {noise:.2%}; memory-branch "
                 f"margin raised to {margin:.2%}")
    # R2: two rulers on every throughput line, and only the first is scored.
    lines.append(
        "  every throughput below is printed as FIXED ROOF / OWN-CLOCK ROOF. "
        "The first is the fraction of `ridge x bandwidth`, the dense GEMM's "
        "achieved figure at the clock the calibration measured it at, and it "
        "is the ONLY one any gate reads. The second is the same throughput "
        "over that roof scaled to the clock the tread itself ran at, which is "
        "issue efficiency: on this card the under-load clock is set per tile "
        "by the kernel's own power draw under the cap, so the two differ by "
        "as much as the governor moved"
        + (f". Reference clock {reference_mhz:.0f} MHz, {reference_clock_source}"
           if reference_mhz else
           ". NOT AVAILABLE on this run, so every second fraction reads '-': "
           + (reference_clock_source or "no reference clock was resolved")))
    lines.append(f"  {sm_count} SMs ({sm_source})")
    # THE QUALIFICATION, AS NUMBERS AGAINST THRESHOLDS, whether it passed or
    # failed. The sibling sweep prints this block; before 2026-09-09 this file
    # printed only `ref.note`, so the H200 report said "its LEVEL is wrong" and
    # never printed the 1.044 non-vacuity ratio that was the failing check, the
    # one number that would have told a reader the refused ladder's slope IS one
    # full weight read per tile.
    lines += ["  " + line for line in ref.render()]
    lines.append("")
    lines.append("THE LADDERS: milliseconds per exactly-full tile stack")
    lines.append("  BLOCK_M  treads  memory-bound  alpha  alpha-corrected  "
                 "B ms/tile  C ms/tile  fit err  basis")
    for bm in tiles:
        f = fits[bm]
        corr = corrected if bm == cap_tile else None
        lines.append(
            f"  {bm:7d}  {len(f.points):6d}  {f.memory_points:12d}  "
            + (f"{f.alpha:5.3f}" if f.alpha is not None else "  n/a")
            + "  " + (f"{corr:14.3f}" if corr is not None else "           n/a")
            + "  " + (f"{f.slope_memory:9.4f}" if f.slope_memory is not None else "      n/a")
            + "  " + (f"{f.compute_slope:9.4f}" if f.compute_slope else "      n/a")
            + f"  {f.mean_rel_err:6.2%}  {f.basis}")

    reached = max((n for n, _ in tp_cap), default=0)
    discriminator = cap_discriminator(cap_tile, ridge, b)
    preds_lo = SWEEP.predictions(tiles, alpha, ridge_band[0], b)

    gates = [
        adopt(SWEEP.gate_0_override(compiles, executed, tiles), "V0", VALIDITY,
              "the two settings may have been one kernel, every difference "
              "below is a comparison of that kernel with itself, and nothing on "
              "this page is evidence"),
        gate_v1_non_vacuity(cells, tiles=tiles, planned_cells=planned_cells),
        gate_v2_control(ref, tp_control, control_tile=control_tile,
                        issue=issue_control,
                        noise=noise),
        gate_v3_control_roof(tp_control, control_tile=control_tile,
                             issue=issue_control,
                             roof_tflops=roof_tflops, plateau=plateau,
                             noise=noise),
        gate_v4_depth(reached, depth, cap_tile=cap_tile, alpha=alpha),
        gate_c1_roof_fraction(tp_cap, depth, cap_tile=cap_tile, alpha=alpha,
                              issue=issue_cap,
                              ridge=ridge, b=b, roof_tflops=roof_tflops,
                              discriminator=discriminator),
        gate_c2_measured_cap(fit_cap, corrected, cap_tile=cap_tile, ridge=ridge,
                             b=b, discriminator=discriminator, cfg=cfg,
                             block_n=(pinned or SWEEP.FIXED)["BLOCK_SIZE_N"]),
        adopt(SWEEP.gate_2_direction(ok, cfg, alpha=alpha,
                                     retracted=RETRACTED_ALPHA, ridge=ridge,
                                     bandwidth_gbps=bandwidth_gbps, b=b,
                                     block_sizes=tiles),
              "C3", CLAIM,
              "time did NOT fall with the tile height in the multi-tile regime, "
              "so the extra weight re-reads a short tile pays are outweighed by "
              "padded arithmetic or lost occupancy, and the traffic reading of "
              "the cap is wrong"),
        adopt(SWEEP.gate_1_steps(ok, cfg, preds_lo, alpha=alpha, ridge=ridge,
                                 bandwidth_gbps=bandwidth_gbps, b=b, noise=noise),
              "C4", CLAIM,
              "the time steps do not land where the tile count changes, so "
              "whatever the ladders measured is not the tile quantum and the "
              "mechanism behind C1 and C2 is not the one claimed"),
    ]

    lines.append("")
    lines.append("GATES.  V = validity: a FAIL voids the page.  "
                 "C = claim: a FAIL is a result.")
    for g in gates:
        lines += g.render()
        lines.append("")

    void = [g.tag for g in gates if g.kind == VALIDITY and g.verdict != PASS]
    claims = [g for g in gates if g.kind == CLAIM]
    failed = [g.tag for g in claims if g.verdict == FAIL]
    if void:
        lines.append(f"READING IT. Validity gates {void} did not pass. No roof "
                     "fraction, alpha or ratio on this page may be quoted, and "
                     "each of those gates says what to change.")
    elif not failed:
        lines.append(
            f"READING IT. BLOCK_M={cap_tile} was swept to {reached} tiles "
            f"({reached * cap_tile} rows per expert), reached "
            f"{max(v for _, v in tp_cap):.1%} of ridge x bandwidth, "
            f"and its own ladder puts its AI ceiling at "
            + (f"{SWEEP.ai_cap(cap_tile, corrected, b):.1f} Op/B"
               if corrected is not None else "an unmeasured value")
            + f" against a ridge of {ridge:.1f}. The cap formula holds at this "
            "tile height, and the depth in V4 is what makes that an exclusion "
            "rather than a shrug."
            + " WHAT IT DOES NOT SAY: that a shipped kernel is stuck below "
            "its roof. " + observed_note(cap_tile) + ". BLOCK_M=128 is the "
            "multi-tile regime and scripts/bm128_depth.py is where the "
            "production reading is settled.")
    else:
        lines.append(
            f"READING IT. Claim gates {failed} FAILED with every validity gate "
            f"passing, so this is a result and not a broken run: the cap at "
            f"BLOCK_M={cap_tile} is not where this study put it. Each failing "
            "gate names what it retracts.")

    payload = {
        "experiment": "tile_cap",
        "cap_tile": cap_tile, "control_tile": control_tile,
        "alpha": alpha, "alpha_band": list(ALPHA_BAND),
        "retracted_alpha": RETRACTED_ALPHA,
        "ridge": ridge, "ridge_band": list(ridge_band), "dtype_bytes": b,
        # PROVENANCE PER NUMBER. A ridge with no source is a constant from
        # documentation wearing a measurement's clothes, and both claim gates
        # are scored against this one.
        "ridge_source": ridge_source, "ridge_band_source": band_source,
        "ridge_device": ridge_device,
        # THE CARD, AND WHETHER A CARD WAS INVOLVED AT ALL. `synthetic` is the
        # field that separates a metered pod run from a laptop replay in the
        # only machine-readable artefact this script writes; without it a
        # `--self-test 0.10` report.json is indistinguishable from a
        # measurement of the retracted world.
        "card": card, "synthetic": synthetic,
        "model": model_name, "dtype": dtype, "fixed": pinned or SWEEP.FIXED,
        "plateau_tflops": plateau, "model_roof_tflops": roof_tflops,
        # THE TWO ROOFS, BOTH NAMED. `model_roof_tflops` is the DENSE peak and
        # is what every SCORED roof fraction is a fraction of (the unscored
        # issue efficiency beside it divides by that peak scaled to the tread's
        # own clock, and `reference_clock_mhz` below is what it scaled from);
        # `fused_layer_roof_tflops` is what V3 scores a control against, and the
        # band it comes from is beside it so a reader can see it is measured
        # rather than chosen. A report carrying one roof and calling it "the
        # roof" is what let a gate demand the dense peak of a fused layer.
        "fused_layer_roof_tflops": FUSED_ROOF_FLOOR * roof_tflops,
        "fused_plateau_band": list(FUSED_PLATEAU_BAND),
        # The factor by which the LIN cap below overstates the exact one; see
        # gate C2. A cap in a JSON file with no factor beside it is the shape
        # that put a 31% overstatement into the study's headline table.
        "lin_overstatement": list(
            SWEEP.cap_overstatement(cfg, cap_tile,
                                    (pinned or SWEEP.FIXED)["BLOCK_SIZE_N"], b)),
        "timing_spread_median": noise, "memory_branch_margin": margin,
        "sm_count": sm_count, "sm_source": sm_source,
        "planned_cells": planned_cells, "measured_cells": len(ok),
        "depth_required": asdict(depth), "depth_reached_tiles": reached,
        "roof_fraction_gate": ROOF_FRACTION,
        "cap_discriminator": discriminator,
        "alpha_measured": fit_cap.alpha if fit_cap else None,
        "alpha_corrected": corrected,
        "ai_cap_measured": (SWEEP.ai_cap(cap_tile, corrected, b)
                            if corrected is not None else None),
        "peak_roof_fraction": {str(bm): (max((v for _, v in tp), default=None))
                               for bm, tp in ((cap_tile, tp_cap),
                                              (control_tile, tp_control))},
        # R2: the same peak against the roof AT THAT TREAD'S OWN CLOCK, the
        # tread being the one that took the peak above and not a second
        # maximum over other cells. None where the run has no reference clock
        # (--ridge on the command line, a --self-test replay) or the peak
        # tread's row carried no under-load clock. Never a gate input; the
        # source string says where the number came from or why there is none.
        "peak_issue_efficiency": {
            str(bm): (peak_pair(tp, iss)[1] if tp else None)
            for bm, tp, iss in ((cap_tile, tp_cap, issue_cap),
                                (control_tile, tp_control, issue_control))},
        "reference_clock_mhz": reference_mhz,
        "reference_clock_source": reference_clock_source,
        # THE REFERENCE'S OWN QUALIFICATION, in the machine-readable artefact
        # and not only in the prose. `ref.note` alone said "its LEVEL is wrong"
        # on 2026-09-09 and report.json carried the same sentence, so which
        # check failed and by how much survived nowhere.
        "compute_reference": ref.note,
        "compute_reference_candidate": control_tile,
        "compute_reference_block_m": ref.block_m,
        "compute_reference_refused_block_m": ref.refused_block_m,
        # The number that failed when the refusal was on SHAPE, where neither
        # block_m field is set and `compute_reference_refusals` stays empty
        # because `refusals` is the LEVEL checks' list. Without it the only
        # machine-readable trace of a shape refusal was the prose note.
        "compute_reference_mean_rel_err": (None if ref.mean_rel_err == math.inf
                                           else ref.mean_rel_err),
        "compute_reference_refusals": list(ref.refusals),
        "compute_reference_roof_fraction": ref.roof_fraction,
        "compute_reference_vacuity_ratio": ref.vacuity_ratio,
        "compute_reference_level_ratio": ref.level_ratio,
        "compute_reference_level_comparisons": ref.level_comparisons,
        "ladder": {str(bm): {"points": list(f.points),
                             "memory_points": f.memory_points,
                             "alpha": f.alpha,
                             "slope_memory": f.slope_memory,
                             "slope_compute": f.compute_slope,
                             "crosses": f.crosses,
                             "basis": f.basis,
                             "mean_rel_err": f.mean_rel_err}
                   for bm, f in fits.items()},
        "gates": [g.as_dict() for g in gates],
    }
    # THE ENVIRONMENT THE NUMBERS CAME OUT OF, in the one machine-readable
    # artefact that outlives the log. `Provenance.stamp` puts git_sha, gpu_name,
    # ridge_source, bandwidth_source and instrument at the payload's top level
    # and RAISES if the payload already carries a different value for one of
    # them, which is how the block and the report are made to agree rather than
    # merely coexist. Absent (a direct `analyse` call from a test) the payload
    # says so, because a report with no provenance key and a report whose
    # provenance is unknown must not look the same.
    if prov is None:
        payload["provenance"] = None
        payload["provenance_note"] = (
            "no provenance block was supplied to analyse(); this payload came "
            "from a direct call and names no git sha, no device and no "
            "instrument")
    else:
        payload = prov.stamp(payload)
    return Report(lines, gates, payload)


# --------------------------------------------------------------------------
# Persistence and CLI.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """ASK GIT whether it would keep this path. Never assert it from memory.

    An earlier version of this function was a sentence -- "results/* is
    gitignored except results/published/, so this run commits nothing" -- read
    off `.gitignore` on 2026-09-01 and printed unchanged for every `--out`. It
    was wrong for `--out results/published/<arm>` (where the output IS
    committed, and the operator would tarball it instead) and meaningless for
    `--out /tmp/...` (where git has no opinion), and it would go stale the day
    the rule changed. Every sibling script in this change set shells out;
    re-implementing the pattern rules is what got them wrong in the first place.

    rc 0 ignored, rc 1 kept, anything else UNVERIFIED and said so: rc 128 is
    what `git check-ignore` returns for a path outside the work tree, which is
    the pod default `/workspace/results/...`, and reporting that as "tracked"
    is the failure mode this whole function exists to prevent.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=str(HERE.parent), capture_output=True,
                              timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path UNVERIFIED"
    if proc.returncode == 0:
        return ("IGNORED by git: nothing written here enters the repo. That is "
                "the intended deal for a pod run -- point the results root at "
                "the network volume so teardown does not take it -- and "
                "scripts/publish_results.sh is how a report is committed.")
    if proc.returncode == 1:
        return "git WILL KEEP this path: anything written here is committable."
    return (f"git check-ignore exited {proc.returncode}; path UNVERIFIED "
            f"({proc.stderr.decode(errors='replace').strip()}). Common cause: "
            "the path is outside this work tree, e.g. /workspace/results on a "
            "pod, which git has no opinion about at all.")


@dataclass(frozen=True)
class PlantedWorld:
    """A synthetic alpha, what it is planted to demonstrate, and the verdicts.

    A SELF-TEST THAT ASSERTS NOTHING IS A SMOKE TEST. Both worlds this file used
    to plant were run, printed and left for a reader to eyeball, at a spread of
    0.00% -- so nothing checked that the gates SEPARATED them, nothing exercised
    the noise-floored thresholds at a noise any pod produces, and C2's FAIL
    branch was never reached in either. `expect` is the registration: the
    verdict every named gate must return in this world, checked by `check`, and
    a mismatch is a defect in the apparatus rather than a result about anything.

    A gate absent from `expect` is deliberately unregistered and not asserted;
    a gate NAMED in `expect` that the report does not contain is itself a
    mismatch, because a registration that silently matches nothing is the
    check-that-examined-nothing shape one level up.
    """

    alpha: float
    why: str
    expect: dict[str, str]

    def applies(self, *, tiles: tuple[int, int], reached_rows: int,
                needed_rows: int) -> str:
        """"" when this registration governs the run, else why it does not.

        A REGISTRATION IS ABOUT A DESIGN, NOT ONLY ABOUT AN ALPHA. Every verdict
        in `expect` was derived at the DERIVED depth with the registered tile
        pair; at `--r-max 512` the same alpha gives V1 FAIL (the grid has holes
        by design), V2 FAIL (two treads at the control is under
        `compute_reference`'s three) and a different C2, and none of that is a
        defect. Asserting the registration there would turn a deliberately
        shallow probe into a self-test failure, which is the check firing on the
        one thing it is not about.
        """
        if tiles != (CAP_TILE, DEFAULT_CONTROL):
            return (f"the registered verdicts are for the tile pair "
                    f"{(CAP_TILE, DEFAULT_CONTROL)} and this run is {tiles}")
        if reached_rows < needed_rows:
            return (f"the registered verdicts are for a grid at least "
                    f"{needed_rows} rows per expert deep (what V4 requires) "
                    f"and this run reaches {reached_rows}")
        return ""

    def check(self, report) -> list[str]:
        """The registered verdicts that did not come back. Empty is a pass."""
        got = {g.tag: g.verdict for g in report.gates}
        bad = []
        for tag, want in sorted(self.expect.items()):
            if tag not in got:
                bad.append(f"{tag}: registered {want}, but the report has no "
                           f"gate {tag}")
            elif got[tag] != want:
                bad.append(f"{tag}: registered {want}, got {got[tag]}")
        return bad


#: The worlds `--self-test ALPHA` knows, keyed on the planted alpha. A world not
#: in this table still RUNS -- exploring one is the point of taking a float --
#: and simply asserts nothing, which the transcript says out loud.
SELF_TEST_WORLDS: dict[float, PlantedWorld] = {
    ALPHA: PlantedWorld(
        ALPHA,
        "the refit world this study says it is in: every gate passes, and the "
        "cap tile sits far under any roof",
        {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
         "C1": PASS, "C2": PASS, "C3": PASS, "C4": PASS}),
    RETRACTED_ALPHA: PlantedWorld(
        RETRACTED_ALPHA,
        "the retracted world every gate discriminates against: the claims fail "
        "and no validity gate does, because a rejection has to come from the "
        "data and not from the instrument",
        {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
         "C1": FAIL, "C3": FAIL,
         # C2 is UNDECIDED here BY MECHANISM and that is the registration. At
         # alpha=0.10 the cap tile's ceiling is 160.0 Op/B against a ridge of
         # 160.3, so its memory branch runs parallel to the compute branch --
         # ratio 1.002, inside PARALLEL_BRANCH_TOLERANCE -- and the fit refuses
         # to name an alpha rather than inventing one. Registering PASS or FAIL
         # here would be registering a bug.
         "C2": UNDECIDED}),
    C2_FAIL_ALPHA: PlantedWorld(
        C2_FAIL_ALPHA,
        "the only world where C2's FAIL branch is reachable: the fit CAN name "
        "an alpha (ridge/cap = 1.40, outside the parallel-branch tolerance) and "
        "that alpha puts the ceiling above the discriminator",
        {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
         "C2": FAIL}),
}


def default_run_id(args, r_max: int, card: str) -> str:
    """Derived from every argument that changes a measured cell.

    EVERY SWEPT KNOB IS IN THE KEY, including both tiles and the RESOLVED r_max
    rather than the `0` that asks for it to be derived. The failure this
    prevents has already happened once in this repo: two settings deriving the
    same id, the second resuming the first, skipping every completed cell and
    printing the first's numbers under the second's label. `--alpha` and
    `--ridge` are NOT in the key and must not be: they change the analysis of a
    set of cells, not the cells, so two analyses of one sweep belong in one
    directory.

    FOUR THINGS WERE MISSING FROM THAT LIST until 2026-09-02, and the docstring
    above claimed otherwise, which is worse than not claiming it:

      * THE CARD. See `detect_card_slug`. It is first in the visible name so
        two cards are distinguishable in `ls` and not only by a hash.
      * `--iters`, `--warmup` and `--cell-budget-ms`. These are not analysis
        knobs: they change the measured milliseconds of every cell, and
        `run_sweep` resumes from `cells.csv` keyed on `(BLOCK_M, tokens)`. A
        `--iters 200` re-run after a `--iters 50` run landed in the same
        directory, skipped all 162 cells and printed the 50-iteration timings
        under the 200-iteration label -- invisibly, because the report renders
        `pinned` and the argument echo from argv rather than from the cells it
        read. Proven off GPU: both invocations derived
        `mixtral-8x7b-bf16-bm16v256-r2112-g1-n64-4882bb`.
      * `--self-test`, and the noise applied to it. A self-test writes a
        `report.json` into whatever directory its arguments name, and with the
        planted alpha out of the key that was the SAME directory a measured pod
        run uses: one free laptop command overwrote the metered run's only
        machine-readable artefact with a synthetic one, and the replacement
        carried alpha=0.10, the retracted world this experiment exists to
        exclude. `synthetic` also prefixes the visible name; `report.json`
        carries `synthetic: true` besides.

    THE KEY IS NO LONGER BUILT HERE. `moe.bench.provenance.run_id` owns the
    rule -- card first, knobs sorted, canonicalised and hashed whole, an
    unresolved knob REFUSED rather than named as `None` -- and this function's
    job shrank to naming which knobs are swept. Two run-id schemes in one
    repository is how two settings came to derive one directory in the first
    place, and the argument for a local `hashlib` call was always that it was
    only six lines.
    """
    swept = {
        "model": args.model, "dtype": args.dtype,
        "bm": args.cap_tile, "ctl": args.control,
        "r": r_max, "step": args.row_step, "probes": args.step_probes,
        "seed": args.seed, "g": args.group_m, "n": args.block_n,
        "stages": args.num_stages, "iters": args.iters,
        "warmup": args.warmup, "budget": args.cell_budget_ms,
        "trials": args.trials, "l2flush": not args.no_l2_flush,
        # NEVER None: `provenance.run_id` refuses an unresolved knob, and it is
        # right to. "measured" is a resolved value that says a real card was
        # asked; a planted alpha is a different resolved value.
        "planted": "measured" if args.self_test is None else args.self_test,
        "plantnoise": args.plant_noise,
    }
    prefix = "synthetic-" if args.self_test is not None else ""
    return prefix + PV.run_id(card=card, **swept)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 makes the whole "
                         "rows-per-expert range reachable at four times the "
                         "token count, and it is one of the two shapes vLLM "
                         "ships a tuned bf16 H200 file for, so the fallback "
                         "tile this script is about is a comparison and not "
                         "the only option")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="bf16 or fp16. The cap claim is a bf16 statement: the "
                         "fp8 tuned files pick BLOCK_SIZE_M=64 from M=1 upward "
                         "and never visit 16 at all")
    ap.add_argument("--cap-tile", type=int, default=CAP_TILE,
                    help="the tile under test. 16 is what get_default_config "
                         "returns for M <= 32")
    ap.add_argument("--control", type=int, default=DEFAULT_CONTROL,
                    help="the positive control, which must CROSS inside the "
                         "grid or the absence at the cap tile is unbracketed. "
                         "256 crosses in tread 1 under both alphas and both "
                         "ends of the ridge band; 128 crosses in tread 2 or 3 "
                         "depending on the ridge and is the A100-safe fallback "
                         "at --num-stages 3")
    ap.add_argument("--r-max", type=int, default=0,
                    help="largest rows per expert. 0 DERIVES it from the depth "
                         "the retracted alpha would need to trip C1 at the "
                         "ridge band's high end, which is the only depth at "
                         "which a PASS excludes anything")
    ap.add_argument("--row-step", type=int, default=32)
    ap.add_argument("--step-probes", type=int, default=6,
                    help="tile boundaries per tile to bracket for C4")
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"],
                    help="pipeline stages, applied to BOTH settings. "
                         "BLOCK_SIZE_M=256 at 4 stages asks for 163,840 bytes "
                         "of shared memory, which is 160 KiB: inside sm_90's "
                         "232,448 and inside sm_80's 166,912 by 3 KiB, so it "
                         "fits the A100 too. The plan prints the bill from "
                         "tile_resources and refuses before a pod is rented "
                         "rather than asking anyone to trust that arithmetic. "
                         "Lowering it here moves both settings together, which "
                         "is what keeps the sweep pinned")
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"],
                    help="the swizzle width, applied to BOTH settings. 1 is "
                         "what the fallback ladder holds across the decode "
                         "range, so it is the setting the tile under test "
                         "actually ships with. alpha is measured AT this "
                         "swizzle and the ceiling 2 BM/(alpha b) moves with "
                         "it, so the cap claim is a claim at G=1. The per-G "
                         "medians once quoted here (0.84/0.73/0.68/0.67 at "
                         "G=1/8/16/64 'on both cards') were POOLED, unpaired "
                         "medians over different fits; the one matched A100 "
                         "cell moves the other way, so no direction in G is "
                         "established and none is assumed here")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"],
                    help="the N tile, applied to BOTH settings. An extra M-tile "
                         "re-reads activations as well as weights in the ratio "
                         "BLOCK_M/BLOCK_N, so at 64 the cap tile re-reads them "
                         "a quarter as often as it re-reads weights and the "
                         "activation correction on alpha is small")
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob and kept in the run id. "
                         "moe.bench.timing.time_kernel sizes the iteration "
                         "count per cell from --cell-budget-ms and the "
                         "warmup's own queue-deep per-call time; a cell's real "
                         "count is a column in cells.csv")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. The sibling's units, because the "
                         "sibling's run_sweep is what times every cell here")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per cell; the percentiles are over "
                         "iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the roof was measured flushed and a "
                         "warm-L2 cell is not comparable with it. Recorded per "
                         "row either way")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its iteration count from it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sm-count", type=int, default=0,
                    help="0 asks the driver; only needed off-GPU")
    ap.add_argument("--capability", default="",
                    help="compute capability as MAJOR.MINOR, e.g. 9.0 for the "
                         "H200. Empty asks the device, and a synthetic run has "
                         "no device, so naming it here is how --dry-run on a "
                         "laptop gets the shared-memory verdict the pod would "
                         "give. An unknown capability is reported as unknown "
                         "and never assumed to fit")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="the ridge C1 and C2 are scored against, as an "
                         "assertion in this run's own command line. 0 RESOLVES "
                         "it from the attached device's calibration, and a "
                         "measured run with no calibration for its own device "
                         "REFUSES rather than borrowing a constant: the module "
                         f"band {RIDGE_BAND[0]}-{RIDGE_BAND[1]} is a "
                         "2026-08-26 H200 figure, and the A100 calibrates at "
                         "145.7, so scoring an A100 run against it makes both "
                         "claim gates wrong by 1.10x. --dry-run and "
                         "--self-test may fall back to the band as a stated "
                         "HYPOTHESIS, because they measure nothing")
    ap.add_argument("--ridge-band", default="",
                    help="LO,HI to go with --ridge. Without it a given --ridge "
                         "makes the band degenerate, which is honest: one "
                         "asserted number is not a band")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0,
                    help="0 reads this machine's calibration, else 4374.5")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, then stop")
    ap.add_argument("--self-test", type=float, default=None, metavar="ALPHA",
                    help="generate the cells from the model at this alpha and "
                         "run the whole analysis on them, off GPU. An alpha in "
                         "SELF_TEST_WORLDS also ASSERTS that world's registered "
                         "verdicts and exits ERROR if one comes back different")
    ap.add_argument("--plant-noise", "--self-test-noise", type=float,
                    default=PUBLISHED_CELL_SPREAD, dest="plant_noise",
                    metavar="SIGMA",
                    help="lognormal sigma applied to every synthetic cell, and "
                         "the noise assumption every MDE in the plan is derived "
                         f"from. Defaults to {PUBLISHED_CELL_SPREAD}, the "
                         "published per-cell spread; 0 plants a world no pod "
                         "produces and exercises the noise-floored thresholds "
                         "at the one value they cannot fail at")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="RETIRED 2026-09-02 and accepted so old driver lines "
                         f"still parse. A failed CLAIM gate now always exits "
                         f"{exit_codes.CLAIM_FAIL} CLAIM_FAIL, which the ledger "
                         "reads as a finished result rather than a retry; "
                         "folding it into 0 made the log disagree with the "
                         "process, and --self-test 0.10 was the live instance")
    return ap


def _main(argv=None) -> int:
    """The body, and the one place an exit code is chosen. `main` wraps it.

    Every return is a member of `moe.bench.exit_codes`'s table: REFUSED (2)
    before anything is measured -- `--dry-run` included, because a plan scores
    no gate -- ERROR (4) for a planted world that came out other than
    registered, and otherwise `classify` over the scored gates with nothing
    folded. ERROR for an exception nobody planned for is the OTHER half, and it
    is in `main`: until 2026-09-02 this file had no top-level handler and such
    an exception exited the interpreter's 1, which the driver reads as
    CLAIM_FAIL. `--fail-on-gate` is retired: a CLAIM_FAIL
    is returned as 1 whether or not it is passed, because a falsified
    pre-registered claim is a successful experiment and 1 is already the code
    that says so to the ledger. A VALIDITY failure is INVALID either way, since
    nothing on the page may be quoted after one.
    """
    # BEFORE `build_parser`, which reads `SWEEP.FIXED` for its defaults. A probe
    # after `parse_args` fires after the AttributeError it exists to replace.
    try:
        require_sweep_api()
    except CapTestRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    tiles = (args.cap_tile, args.control)
    if args.cap_tile >= args.control:
        print(f"REFUSED: --cap-tile {args.cap_tile} is not smaller than "
              f"--control {args.control}. The control exists to be the tile "
              "that CAN cross while the cap tile cannot; ordering them the "
              "other way makes every comparison below read backwards.")
        return exit_codes.REFUSED
    synthetic = args.self_test is not None
    if synthetic and not args.bandwidth_gbps:
        bandwidth = PUBLISHED_H200_GBPS
        bw_source = ("published H200 triad ceiling, PINNED for --self-test so "
                     "the replay is identical on every machine")
    else:
        bandwidth, bw_source = SWEEP.resolve_bandwidth(args)

    # THE RIDGE IS RESOLVED, NEVER INHERITED. It used to default to
    # `RIDGE_BAND[0]` = 160.3, a 2026-08-26 H200 constant, while the bandwidth
    # beside it in the very same product was read off the attached card: on an
    # A100 that assembles `ridge x bandwidth` out of two machines and puts the
    # implied roof 9.9% above the card's own measured 262.4 TFLOP/s, with both
    # claim gates off by 160.3/145.8. `--self-test` is pinned to the module band
    # instead of resolved, for the same reason its bandwidth is: a replay that
    # reads the hardware is not a replay, and the suite could not pin either end
    # of it.
    if synthetic and not args.ridge:
        ridge, ridge_band = RIDGE_BAND[0], RIDGE_BAND
        ridge_source = band_source = (
            "PINNED for --self-test: the module's H200 band, so the replay is "
            "identical on every machine and belongs to no attached device")
        ridge_device = ""
        ridge_kind = "pinned"
    else:
        try:
            rr = SWEEP.resolve_ridge(args, synthetic=synthetic or args.dry_run)
        except SWEEP.RidgeUnavailable as exc:
            print(f"REFUSED: {exc}")
            return exit_codes.REFUSED
        ridge, ridge_band = rr.ridge, rr.band
        ridge_source, band_source, ridge_device = rr.source, rr.band_source, rr.device
        ridge_kind = getattr(rr, "source_kind", "")

    # R2's SECOND RULER, AND IT COMES OUT OF THE SAME FILE AS THE FIRST OR NOT
    # AT ALL. The own-clock roof is `peak x load / reference`, so `reference`
    # has to be the clock the peak in `ridge x bandwidth` was measured at.
    # That holds for exactly one of the three ways this run can get a ridge:
    # the attached device's own calibration. `--ridge` given on the command
    # line is the operator's assertion about some other machine's peak and the
    # calibration's clock does not describe it, and `--self-test` reads no
    # hardware on purpose, so both print the fixed fraction alone and say why.
    # `usable_for_roof` is the calibration's own grade: the idle scalar the
    # older files carry has a 30% spread and scaling a roof by it would move
    # every second fraction by up to that much under the name of a correction.
    reference_mhz: float | None = None
    if synthetic:
        reference_clock_source = (
            "PINNED for --self-test: no hardware is read, so there is no "
            "clock to scale a roof by")
    elif ridge_kind != "calibration":
        reference_clock_source = (
            f"the ridge came from elsewhere ({ridge_kind or 'unstated'}), so "
            "this card's calibration clock does not describe the roof it "
            "states")
    else:
        rc = roofline.reference_clock(
            ridge_device or None, family=roofline.reference_family(args.dtype))
        reference_clock_source = rc.source
        if rc.usable_for_roof:
            reference_mhz = rc.mhz
        elif rc.mhz:
            reference_clock_source += (
                f" [grade {rc.grade!r}, not the under-load median, so it is "
                "recorded and NOT used to scale a roof]")

    try:
        depth = required_depth(args.cap_tile, b=b, ridge_band=ridge_band)
    except CapTestRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    # THE DEFAULT GRID MUST CARRY V1, V2 AND V4. `depth.rows` is the cap tile's
    # own horizon and nothing else: it need not be a multiple of --row-step, and
    # it knows nothing about the control. On the H200's own ridge band it is
    # 688 = 43 x 16, `build_grid` stops at 672, and exactly two multiples of the
    # 256 control land on the grid. Floor it at the control's stacks as well and
    # round UP to the step, which puts the H200 default at 1024.
    floor = max(depth.rows, CONTROL_STACKS_FLOOR * args.control)
    floor += (-floor) % args.row_step
    r_max = args.r_max or floor
    card = detect_card_slug()

    grid = SWEEP.build_grid(cfg, tiles, r_max, args.row_step, args.step_probes)
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  GROUP_SIZE_M=args.group_m, BLOCK_SIZE_N=args.block_n)
    run_id = args.run_id or default_run_id(args, r_max, card)
    out_dir = (args.out or SWEEP.results_root()) / "tile_cap" / run_id
    csv_path = out_dir / "cells.csv"
    cache_root = out_dir / "triton-cache"

    capability = SWEEP.resolve_capability(
        args, synthetic=args.self_test is not None or args.dry_run)
    resources, refused = SWEEP.tile_resource_plan(pinned, tiles, b, capability)

    header = plan_lines(cfg, args, tiles=tiles, grid=grid, depth=depth, b=b,
                        bandwidth_gbps=bandwidth, bw_source=bw_source,
                        out_dir=out_dir, pinned=pinned, run_id=run_id,
                        resources=resources, card=card,
                        git_note=git_visibility(out_dir))
    header += prediction_lines(cfg, cap_tile=args.cap_tile,
                               control_tile=args.control, alpha=args.alpha,
                               ridge=ridge, b=b, bandwidth_gbps=bandwidth,
                               depth=depth, r_max=r_max, ridge_band=ridge_band,
                               ridge_source=ridge_source,
                               band_source=band_source)
    # THE MDE IS PART OF THE PLAN, not of the post mortem. It is derived from
    # `--plant-noise` and printed with that assumption named, before the pod is
    # rented, because the only cheap moment to find that a gate cannot resolve
    # the effect it is registered against is before it is paid for.
    aligned_cap = [r for r in grid if r % args.cap_tile == 0]
    try:
        header += mde_lines(
            cfg, cap_tile=args.cap_tile, spread=args.plant_noise,
            alpha=args.alpha, ridge=ridge, b=b, bandwidth_gbps=bandwidth,
            treads=len(aligned_cap),
            discriminator=cap_discriminator(args.cap_tile, ridge, b))
    except CapTestRefusal as exc:
        header += ["", f"MINIMUM DETECTABLE EFFECT: not stateable. {exc}"]
    print("\n".join(header))

    # Both tiles are load bearing and neither can be dropped: without the cap
    # tile there is no claim and without the control there is no instrument. So
    # a setting that cannot run is a refusal here rather than a sweep that
    # quietly becomes one tile wide.
    if refused:
        print("\nREFUSED: a pinned setting cannot physically run.")
        for bm, why in sorted(refused.items()):
            print(f"  BLOCK_M={bm}: {why}")
        print("  Neither tile is optional -- the cap tile IS the claim and the "
              "control IS the instrument -- so this is a refusal and not a "
              "dropped setting. Lower --num-stages, raise --block-n, or pick "
              "another --control, and note that moving any of them moves BOTH "
              "arms, which is what keeps the comparison pinned.")
        return exit_codes.REFUSED

    # A GRID THAT CANNOT SATISFY ITS OWN VALIDITY GATES IS REFUSED HERE, not
    # measured and then voided. This is the 2026-09-09 H200 arm: the plan
    # printed "BM=256:2" beside a WARNING, 72 cells were timed, V1 FAILed on
    # the stack count the plan had already printed, and nothing on the page was
    # quotable. --r-max is honoured as given; the default is floored above.
    short = grid_refusal(grid, tiles=tiles, depth=depth, row_step=args.row_step)
    if short:
        print("\nREFUSED: the grid cannot satisfy its own validity gates, so "
              "nothing measured on it would be quotable.")
        print("\n".join("  " + line for line in short.splitlines()))
        return exit_codes.REFUSED

    if args.dry_run:
        secs = SWEEP.estimated_seconds(
            cfg, grid, tiles, alpha=args.alpha, ridge=ridge,
            bandwidth_gbps=bandwidth, b=b, warmup_ms=args.warmup,
            trials=args.trials, cell_budget_ms=args.cell_budget_ms)
        print(f"\nestimated GPU time {secs:.0f} s at the model's own timings, "
              "excluding compiles and allocation")
        # REFUSED (2) AND NOT DONE (0). A dry run scores no gate, so it prints
        # no RESULT line, and `exit_codes.classify_text` over this log raises
        # `NoGatesScored` -- which that module documents as what a REFUSED log
        # looks like from there. DONE says "measured; every VALIDITY and CLAIM
        # gate PASSED", and this run measured nothing, so the log and the code
        # disagreed in the one direction the shared table exists to stop.
        #
        # THE REPOSITORY DISAGREED WITH ITSELF AND THIS IS THE SIDE THAT WON.
        # On 2026-09-02 six scripts returned DONE from `--dry-run` -- this file,
        # `bm128_depth`, `bm128_roofline`, `bn_decomposition`, and
        # `block_m_crossing_sweep` and `ruler_rebaseline` as the bare literal 0
        # -- and seven returned REFUSED (`calibrate_hardware`,
        # `dtype_tile_confound`, `memory_branch_anchor`, `occupancy_vs_swizzle`,
        # `rescore_published_reports`, `span_extent_separation`, `tile_sweep`).
        # REFUSED is the only one of the two a log can be checked against. The
        # driver reads both as finished and re-queues neither: `dry_state` maps
        # 0 to PLANNED and 2 to PLAN_REFUSED, and `arm` retries neither state.
        print("\n".join(["", "=" * 72,
                         "REFUSED. Nothing was measured and nothing was "
                         "written.",
                         "  reason: --dry-run was given",
                         "  Everything above is arithmetic over this repo's "
                         "calibration and vLLM's",
                         "  resource model. No gate was scored, so no RESULT "
                         "line was printed and",
                         "  none of it is a result. Run --self-test <alpha> for "
                         "the planted worlds,",
                         "  or the bare command on the pod.",
                         "=" * 72]))
        return exit_codes.REFUSED

    if args.self_test is None:
        missing = SWEEP.missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return exit_codes.REFUSED

    out_dir.mkdir(parents=True, exist_ok=True)
    planned = len(grid) * len(tiles)

    # ONE PROVENANCE BLOCK PER RUN, built after the rulers resolve so it carries
    # their sources and before any measurement so every artefact of one run
    # carries one block. `instrument` is the SYNTHETIC name under --self-test
    # and the real one otherwise: a planted report carrying the real
    # instrument's name satisfies a presence check while describing an
    # instrument the run never touched. `iters` is None because --iters is
    # retired as a timing knob and each cell carries the count the instrument
    # actually used; `observed_iters` fills the median in afterwards.
    ridge_src = f"{ridge_source}"
    prov = PV.provenance_block(
        instrument=(SWEEP.SYNTHETIC_INSTRUMENT if synthetic
                    else SWEEP.timing_basis()),
        ridge=ridge, ridge_source=ridge_src,
        bandwidth=bandwidth, bandwidth_source=bw_source,
        warmup_ms=args.warmup, iters=None, target_ms=args.cell_budget_ms)

    if args.self_test is not None:
        alpha = args.self_test
        sm_count = args.sm_count or SWEEP.DEFAULT_SM_COUNT
        cells = SWEEP.synthetic_cells(cfg, grid, tiles, alpha=alpha,
                                      ridge=ridge, bandwidth_gbps=bandwidth,
                                      b=b, sm_count=sm_count,
                                      noise=args.plant_noise, seed=args.seed,
                                      warmup_ms=args.warmup,
                                      trials=args.trials,
                                      l2_flush=not args.no_l2_flush)
        compiles = {bm: 1 for bm in tiles}
        executed = dict(compiles)
        world = SELF_TEST_WORLDS.get(alpha)
        print(f"\nSELF TEST: cells GENERATED from the model at alpha={alpha} "
              f"with a planted spread of {args.plant_noise:.2%}. Nothing here "
              "was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
        print("  registered world: " + (world.why if world else
                                        "NONE. This alpha is not in "
                                        "SELF_TEST_WORLDS, so no verdict is "
                                        "asserted and this run only proves the "
                                        "analysis executes."))
    else:
        import torch
        alpha = args.alpha
        world = None
        sm_count = (args.sm_count
                    or torch.cuda.get_device_properties(0).multi_processor_count)
        started = time.time()
        cells, compiles, executed = SWEEP.run_sweep(
            args, cfg, grid, tiles, csv_path, cache_root, b, pinned, prov=prov)
        print(f"\nswept in {time.time() - started:.0f} s")

    sm_source = ("given on the command line" if args.sm_count
                 else "reported by the driver" if args.self_test is None
                 else f"assumed H200 default {SWEEP.DEFAULT_SM_COUNT}")
    try:
        report = analyse(cells, cfg, cap_tile=args.cap_tile,
                         control_tile=args.control, alpha=alpha,
                         ridge=ridge, bandwidth_gbps=bandwidth, b=b,
                         model_name=args.model, dtype=args.dtype,
                         compiles=compiles, executed=executed,
                         sm_count=sm_count, sm_source=sm_source, depth=depth,
                         planned_cells=planned, header=header, pinned=pinned,
                         capability=capability, ridge_band=ridge_band,
                         ridge_source=ridge_src, band_source=band_source,
                         card=card, ridge_device=ridge_device,
                         synthetic=synthetic,
                         reference_mhz=reference_mhz,
                         reference_clock_source=reference_clock_source,
                         prov=SWEEP.observed_iters(prov, cells))
    except CapTestRefusal as exc:
        print(f"\nREFUSED: {exc}")
        print(f"the cells that did land are at {csv_path} and a re-run resumes "
              "them, so nothing measured is lost")
        return exit_codes.REFUSED

    # The plan and the predictions are already on the terminal, printed before
    # the sweep ran, which is the only order that makes "registered before the
    # run" a property of the transcript. report.txt carries them again so the
    # FILE is self-contained; stdout does not repeat them.
    print("\n".join(report.lines[len(header):]))
    print(SWEEP.iters_line(cells))
    (out_dir / "report.txt").write_text(report.text())
    (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
    print(f"cells    {csv_path}")
    print(f"report   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")

    # THE PLANTED WORLD IS CHECKED AGAINST ITS REGISTRATION, and a mismatch is
    # ERROR rather than any code in the gate table. A self-test that came out
    # differently from the world it planted has not produced a result about
    # anything -- the apparatus is broken -- and INVALID or CLAIM_FAIL would
    # both invite a reader to interpret it.
    if world is not None:
        why_not = world.applies(tiles=tiles, reached_rows=r_max,
                                needed_rows=depth.rows)
        if why_not:
            print(f"SELF-TEST NOT ASSERTED  the alpha={alpha} registration was "
                  f"not applied: {why_not}. The run above is still a real "
                  "analysis of a planted world; it simply is not the design the "
                  "verdicts were registered for.")
            world = None
    if world is not None:
        bad = world.check(report)
        for line in bad:
            print(f"SELF-TEST MISMATCH  {line}")
        if bad:
            print(f"the planted world alpha={alpha} did not return its "
                  f"registered verdicts ({len(bad)} of "
                  f"{len(world.expect)} gates). This is a defect in the gates "
                  "or in the model that generates the cells, not a finding "
                  "about hardware; nothing here may be read as a result.")
            return exit_codes.ERROR
        print(f"SELF-TEST OK  all {len(world.expect)} registered verdicts in "
              f"the alpha={alpha} world came back as registered")

    # THE EXIT CODE COMES FROM THE SHARED TABLE, over the SAME gate objects that
    # printed the RESULT lines, so `exit_codes.classify_text` on this log
    # recomputes the code the process returned.
    #
    # NOTHING IS FOLDED INTO DONE, and `--fail-on-gate` is why this is a
    # paragraph and not a branch. Until 2026-09-02 a CLAIM_FAIL was described in
    # words and RETURNED AS 0 unless the flag was passed, so `--self-test 0.10`
    # printed nine RESULT lines that `classify_text` reads as CLAIM_FAIL and the
    # process said DONE -- the exact split the comment above claims cannot
    # happen, in the file that prints it. The masking was obsolete once the
    # shared table landed: CLAIM_FAIL (1) is in `FINISHED_CODES` and
    # `ledger_state(1)` is "CLAIM_FAIL", so 1 already tells the driver "this is
    # a result, do not retry it" and 0 protects nothing. The flag is accepted
    # and ignored so an old driver line still parses.
    rc = exit_codes.classify(g.scored() for g in report.gates)
    print(f"exit     {exit_codes.describe(rc)}")
    if rc == exit_codes.CLAIM_FAIL:
        print("         a claim that did not pass is a RESULT and the arm is "
              "FINISHED, not broken.")
    return rc


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it
    is in FINISHED_CODES, so the driver files the arm as finished, skips it on
    every resume, leaves RETRY_ARMS at zero and exits the session 0 over an arm
    that never measured. A torch OOM, a truncated report or a drifted import
    would be published as one of this experiment's registered outcomes. ERROR
    (4) is outside FINISHED_CODES precisely so the driver can tell "the
    apparatus broke" from "the claim did not hold", and the traceback is
    printed first rather than swallowed, because a code without one tells an
    operator nothing about what to fix.

    Wrapped around `_main` rather than installed at the `__main__` guard so the
    contract holds for a caller of `main()` -- the tests, and anything that
    imports this file -- as well as for the CLI. `SystemExit` is a
    `BaseException` and passes through untouched: a refusal is not a crash.
    """
    try:
        return _main(argv)
    except Exception as exc:                            # noqa: BLE001
        # A REFUSAL FROM THE INSTRUMENT IS NOT A CRASH. This script times through
        # SWEEP.run_sweep, which since 2026-09-03 re-raises timing.TimingRefused
        # rather than filing it per cell. Left to the handler below it would be
        # printed as a traceback and exit ERROR (4), a retryable crash, when it
        # is REFUSED (2), a precondition with the remedy in its message. timing
        # is imported here rather than at the top because it imports torch, and
        # --dry-run is a laptop path; if it cannot be imported, no TimingRefused
        # could have been raised, so the answer is the crash branch.
        try:
            from moe.bench import timing
            refused = isinstance(exc, timing.TimingRefused)
        except ImportError:
            refused = False
        if refused:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return exit_codes.REFUSED
        traceback.print_exc()
        print("ERROR: tile_cap_test crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim "
              "failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
