#!/usr/bin/env python
"""Sweep BLOCK_SIZE_N at fixed BLOCK_M: the only clean separation of alpha_a from alpha_b.

    python scripts/bn_decomposition.py --self-test    # plant four worlds, off GPU
    python scripts/bn_decomposition.py --dry-run      # the plan, the predictions, the cost
    python scripts/bn_decomposition.py                # the pod run

WHY BLOCK_SIZE_N AND NOTHING ELSE. What a ladder fit returns is not the weight
miss fraction. The estimator this repository uses, `LadderFit.alpha` = B/(A+B),
returns on the three-term byte ladder EXACTLY

    alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                     (EXA)

with `phi = d0 + alpha_a (BM/BN - e)` the activation-plus-output cost of one
M-tile in weight-read units (derived below) and `delta` the fused layer's fixed
cost in the same units. The blend an earlier version of this header wrote,

    alpha_fitted = alpha_b + alpha_a (BM/BN) + BM/K                        (LIN)

is (EXA)'s numerator with the level taken as 1, the reading `moe/bench/ai_model.py`
withdrew on 2026-09-02, and it is kept here only to be compared against. Under
either reading BM appears in TWO places (the activation ratio BM/BN and the
per-tile cost BM/K) while BN appears in exactly one. So
sweeping BM moves alpha_a and alpha_b together and can never separate them --
which is why the study's current alpha_a is a TWO-POINT slope between one BN and
another. Sweeping BN at FIXED BM moves exactly one term. Three or more BN values
give a fit where there were slopes, and -- this is the half that matters more --
they leave a RESIDUAL, which is the only thing in this study that can say
whether the three terms are ALL of it.

HOW MANY TWO-POINT SLOPES THIS REPO ACTUALLY HAS: ONE PAIR, AT TWO BLOCK_M.
Until 2026-09-02 this file said the band [0.10, 0.15] came from "published
two-point A100 slopes 0.106, 0.102, 0.129, 0.119". Those four numbers are in no
file under `results/published`: the A100 BN=256 arm has no identifiable ladder
at all (its compute reference is the 43.6x one this file exists to refuse, and
ANCHOR_RESCORE withdrew it), and the qwen2 H200 BN=256 arm fits nothing either.
The whole repo contains exactly one pair of arms that differ in BLOCK_SIZE_N and
in nothing else -- H200 s4 mixtral at GROUP_SIZE_M=1, BN=64 against BN=256 --
and `published_two_point_alpha_a()` reads it out of those two committed files
rather than quoting it. It gives TWO values, from the two BLOCK_M that fit in
both arms, and they disagree:

    BM=32   alpha 1.0073 (BN=64) - 0.9001 (BN=256) over ds = 0.375  ->  0.286
    BM=64   alpha 0.9327 (BN=64) - 0.8235 (BN=256) over ds = 0.750  ->  0.146

each with a two-point sd of `PUBLISHED_ALPHA_SD/ds` = 0.086 and 0.043, so the
gap between them is 1.5 sigma and neither is two sigma from zero. `ALPHA_A_BAND`
is built from those two numbers and their sds, is 3-6x wider than the sd of its
own inputs, and C1 says so on its own gate line: at this width a PASS is weak
evidence and only a FAIL is informative. Fixing that width is what a THREE-point
BN fit is for, which is this experiment.

THE IDENTITY THAT MAKES BN THE LEVER, derived rather than asserted, because it
is what the whole experiment rests on. One expert holding `r` rows runs
`n = ceil(r/BM)` M-tiles. Per M-tile the up GEMM sweeps `ceil(2F/BN)` N-tiles
and the down GEMM `ceil(H/BN)`, and EACH N-tile re-reads that tile's slice of
the activations. Count the elements an extra M-tile costs:

    weights re-read      alpha_b W                  W = 3 F H, the expert
    activations, once    BM (2H + 3F)                   x_perm h_up h_act y_perm
    activations, again   alpha_a BM (H ceil(2F/BN) + F ceil(H/BN) - H - F)

and the middle of those three brackets is EXACTLY `W/BN`:

    H (2F/BN) + F (H/BN) = 3 F H / BN = W / BN

so the activation re-read per extra M-tile is `alpha_a BM W / BN` -- the study's
own BM/BN ratio, falling straight out, with the model's E, F and H cancelling
against the weight term. Write `phi` for the whole activation-plus-output cost
of one M-tile in units of one full weight read,

    phi = BM (2H + 3F)/W + alpha_a (BM/BN - BM (H+F)/W) = d0 + alpha_a (s - e)

with `s = BM/BN`, and the fitted alpha is then, EXACTLY,

    alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                     (EXA)

where `delta` is the fused layer's fixed cost in the same units. (LIN) is (EXA)
linearised at small phi. THEY ARE NOT INTERCHANGEABLE HERE. At BM=128, BN=64 on
mixtral phi is 0.32, so the denominator is a third again as big as one, and the
two readings of the SAME two published points disagree by a factor of thirty:

    H200 mixtral G=1: alpha_fitted 0.9327 at BN=64, 0.8235 at BN=256
      through (LIN)   alpha_a = 0.146            -- inside the predicted band
      through (EXA)   alpha_a = 4.7              -- impossible, a miss fraction

That disagreement is not a detail to be tidied up before the run. It IS the
experiment: two points cannot say which reading is right, three can, and the
answer decides whether `alpha_b` may be compared with TEMPO's b2/b at all.

WHAT IS BEING TESTED, IN ORDER OF WHAT IT WOULD COST TO BE WRONG.

  1. THE MODEL. If those terms are all of it, alpha_fitted is a straight line in
     the right coordinates and the residual is measurement noise. If a term is
     missing -- wave quantisation, a launch cost that scales with N-tiles, or
     alpha_b ITSELF moving with BN because BN changes the order the L2 is walked
     in -- the residual has STRUCTURE, and its shape names the missing term.
     C2 gates the residual against the run's own bootstrap noise and REFUSES to
     call the model complete when it exceeds it. A pass is a much stronger
     statement than a fitted number.
  2. alpha_a, as the SLOPE against `s = BM/BN` rather than a two-point
     difference. Predicted 0.10 to 0.15.
  3. alpha_b, as the intercept, and it must be THE SAME at every BM. Nothing in
     the model lets a weight miss fraction depend on the tile height, so fitting
     BM=16, 32, 64 and 128 separately turns a parameter into a testable
     invariant. On 2026-09-10 the 128 cells produced nothing at any BLOCK_N, so
     the invariant was tested over two heights; 16 is swept from that date on,
     and the reason is in the paragraph on BM=16 below.
  4. alpha_b against TEMPO (arXiv:2608.13057), which publishes b2/b = 0.311 and
     0.319 for the weight-side re-read and models NO activation-side re-read at
     all, so alpha_a has no counterpart in the closest prior work.

BLOCK_M=128 IS THE PRIMARY AND THE REASON IS PRODUCTION, NOT CONVENIENCE. It is
the only tile vLLM's fallback ladder runs multi-tile as a REGIME: in the one
published arm that records the tile actually chosen (uniform routing, 132 cells
of 7 seeds), counted on `load_max_rows`, which is what `moe_align_block_size`
pads to, 128 runs multi-tile in 66 of 87 cells (65 at the median seed), up to
34 tiles per expert; 16 does in 1 of 24 cells and 64 in 2 of 16 (1 of 168 and 5
of 112 seed-rows), isolated cells, and 32 never in 5. On `load_mean_rows`, the
basis an earlier version of this paragraph quoted as "16, 32 and 64 never", the
small tiles do read never and 128 reads 59 of 87 up to 32 tiles; the mean is a
fact about the routing histogram and not about the launch, which is why the
padded count is the one stated first. The re-read term only exists when there is
more than one tile, so 128 is the only block size where alpha_b is a production
quantity rather than a curiosity.
BM=64 and BM=32 are swept beside it because they are where alpha is ROBUSTLY
identifiable, and because the invariance in (3) needs more than one BM.

AND BM=16 IS SWEPT BECAUSE THE PRIMARY DID NOT SURVIVE. On 2026-09-10 this arm
ran on an H200 at GROUP_SIZE_M=16 and BLOCK_M=128 produced NO alpha at ANY of
the three BLOCK_N: its memory branch came within the parallel-branch tolerance
of the compute branch at all three and was discarded. So a fit registered on
three heights was made on TWO, over six cells, for two parameters, and at two
heights the model's only BLOCK_N-dependent column, which is strictly
proportional to BLOCK_M, is not separable from the `1/BN` column that fits the
same six cells 3.1x better at equal parameter count. `design_collinearity`
computes that: 0.818 over the two-height grid, 0.663 with BM=16 added at all
three BN. A SMALLER subject tile is further from its compute branch and so the
one least likely to lose it, and that is measured rather than assumed.
`published_small_tile_branch` reads the same session's tile_cap arm, which ran
BLOCK_M=16 at BLOCK_N=64 under the same num_warps and num_stages and found 56 of
56 treads memory bound at B/C = 5.46, all nine of its gates passing.
THE GAIN IS THE BN=32 CELL AND NOT THE HEIGHT AS SUCH: BM=16 at BN=64 and 128
alone leaves the collinearity at 0.819, i.e. unchanged. That cell is also the
one setting in the grid whose warp grid is wider than its output tile
(512 elements against 8 warps times one 128-element m16n8k16 tile), which the
plan page now prints as the third bill. It is a RECORD and not a refusal: the
card holds the tile at 24 KiB of shared memory and 2 accumulator registers per
thread, the kernel computes the right answer, and the two settings that sit
exactly ON that boundary have both run and been published.

THE PRECONDITION, AND IT KILLED THE LAST ATTEMPT AT THIS SWEEP. Every alpha here
is a membership decision against a COMPUTE REFERENCE, and the previous BN sweep
took its reference from BLOCK_M=256 at BLOCK_N=256, where one M-tile took
249.765 ms on the A100 against 5.724 ms for the identical setting at BN=64. It
qualified: the qualification tested PROPORTIONALITY, a line 43.6x too steep is
perfectly proportional, and it passed at 0.2% mean error. Every tread in the arm
was then classified against a compute branch 44x too steep, nothing could stand
above it, and all 8 cells printed as a tidy null. Three defences here, all of
them refusals rather than warnings:

  * THE SETTING IS REFUSED BEFORE IT IS TIMED. `BLOCK_M x BLOCK_N` fp32
    accumulators at num_warps=8 need `BM BN / 256` registers per thread against
    a hardware maximum of 255, and `num_stages (BM BK + BK BN) b` bytes of
    shared memory against 163 KiB on an A100 and 227 on an H200. At BM=BN=256
    that is 256 registers and 192 KiB: the accumulator alone does not fit on
    EITHER card and the pipeline does not fit on the A100. Both bills are pure
    arithmetic on the pinned constants, so `--dry-run` prints the same refusal
    on a laptop that the pod would.
  * THE REFERENCE'S LEVEL IS CHECKED IN ABSOLUTE UNITS, against the ATTACHED
    card's own calibrated ceiling: a reference slope implies an achieved
    TFLOP/s, and it has to land inside [25%, 100%] of that ceiling. The corrupt
    A100 reference implies 1.4% and its H200 twin 12.3%; the 22 sound published
    references run 38.2% to 63.7%.
  * AND ACROSS BN, which is the check only THIS sweep can make, because it holds
    every BN in one run on one card. The achieved rate moves with BN by tens of
    percent through occupancy; it does not move by 40x. The spread of the
    implied rates across BN is gated directly.

A BN arm whose reference fails any of those contributes NO cell, and the report
says the fit lost a point to a refusal rather than to a sweep that lacked
treads. With fewer than three BN arms surviving, alpha_a is UNIDENTIFIED and
every claim gate reads UNKNOWN. That is the honest failure mode and it is
pre-registered as one.

AND WHEN AN ARM CANNOT QUALIFY ONE, IT BORROWS. At small BLOCK_N every tile
height on this hardware is memory bound -- the activation re-read is large
enough that even BLOCK_M=256 never reaches its compute branch -- so the arm with
the MOST leverage on alpha_a is the one least able to produce a reference of its
own. `import_reference` lends it one from the arms that did, at the cost of an
assumption stated out loud (the achieved compute rate does not move with
BLOCK_N), allowed only when two arms agree within V3's bar, and stamped
IMPORTED on every cell it touches.

THE OTHER WAY A CELL GOES MISSING, also pre-registered. `fit_ladder` DISCARDS
the memory branch when `|B/C - 1| <= 0.15`, because two branches within 15% of
each other are one line and a fit that reads a stretch of the compute branch as
a memory branch reports that branch's slope as alpha. `B/C` is predictable per
cell before the run, and it says the PRIMARY is the cell most likely to go: the
measured median over the 22 published BLOCK_M=128 ladders is 0.991, which is
INSIDE the tolerance. That prediction was registered as 2 of 3 BN points and the
2026-09-10 H200 run returned ZERO of three, so the sentence that stood here
until that date, "the pooled fit carries the residual test", described an arm
that had not happened. What carries the residual test is the heights that do
keep a branch: BM=64, BM=32 and, from 2026-09-10, BM=16, all of which sit far
outside the band at every BN. Those predictions are ANCHORED on that measured
0.991 and not on a calibrated ridge: `B/C` carries the kernel's OWN achieved
FLOP-per-byte, which is not the card's ridge, and the calibrated form says the
BLOCK_M=256 reference at BN=64 is memory bound when it is measurably the
qualified compute reference in 22 of the 24 published arms.

AND THE SWIZZLE DECIDES WHETHER ANY OF IT IS RESOLVABLE. The response moves with
alpha_a as `g1 (1 - alpha_b)/(1 + phi)^2`, so the design's whole power is
proportional to `1 - alpha_b`. At GROUP_SIZE_M=1 -- the production fallback, and
this file's default because it is what the study pins elsewhere -- the corpus
puts alpha near 0.93 and the lever is worth a small fraction of its size at
GROUP_SIZE_M=16. THE SPREAD THAT FOLLOWS IS COMPUTED AND NEVER QUOTED: this file
used to print "sd 0.11-0.13 ... at any rep count" as a literal string while the
number the code produced at those settings was 0.176, so `design_power()` now
plants the TRUTH world at the run's own noise and the plan prints the sd it got,
beside the effect that sd can resolve (`mde_one_sample`). A G=1 run measures
alpha_b and the invariance and reads UNKNOWN on alpha_a; `--group-m 16` is the
setting that resolves it, and `--reps` buys the rest.

AND AT G=1 IT DECIDES C2 AS WELL, WHICH IS WHY THAT GATE NOW HAS A POWER GUARD.
The residual gate is the reason this experiment exists, and a gate is only a
gate if both its outcomes can occur. They cannot at every pinning: planted at
GROUP_SIZE_M=1 the MISSING world, the same world `--self-test` uses to prove
C2 discriminates, comes back WELL UNDER the 4.0 ceiling and PASSES, the
identical verdict the TRUTH world gets, so a C2 PASS there would be reported
as "the three terms are all of it" from a test that cannot say otherwise. THE
CHI2 IS NO LONGER WRITTEN DOWN HERE: three paragraphs in this file said "chi2
1.78" and that was the figure of the THREE-height design at one --draws and one
--plant-noise. The fourth height moved it to 1.09 at the settings the tests use
and 2.70 at the CLI's defaults, and the verdict, which is what the sentence is
about, did not move at all. The number is computed by `c2_power_probe` and
printed with its settings beside it. The
real run therefore plants that world ITSELF, at its own measured across-repeat
spread and its own swizzle, BEFORE it scores C2 (`c2_power_probe`), and C2 reads
UNKNOWN with the reason whenever the missing-term world would have passed. The
audit that found this also found S4 passing only for planted noise at or below
about 1%, against published H200 spreads of 0.76-1.82%, which is why
`--plant-noise` now RESOLVES to a measured spread instead of defaulting to the
middle of that range.

THE INSTRUMENT IS THE SHARED ONE. Every cell is timed by
`moe.bench.timing.time_kernel` under `TIMING_BASIS`: queue-deep, L2 flushed per
iteration, SM clock sampled under load, warmup in MILLISECONDS of delivered load.
The private `time_call` this file used until 2026-09-02 synchronised per
iteration with events created inside the loop and no flush, which is not the
instrument the compute roof was measured with; it let 0.18-0.30 ms of host
enqueue inside the measured interval, a per-card bias of 8-16% in the fitted
alpha at the smallest ladder cells. Every timing on disk from before that change
is therefore not comparable with the roof this file scores against, and the
`instrument` column on every row says which one produced it.

WHAT IT WRITES. Under `$MOE_RESULTS_DIR`, else `/workspace/results` (the RunPod
network volume, which outlives the pod), else `<repo>/results`:

    <results>/bn_decomposition/<run-id>/cells.csv     one row per tread per rep
    <results>/bn_decomposition/<run-id>/CARD          the card that wrote it
    <results>/bn_decomposition/<run-id>/report.txt    exactly what was printed
    <results>/bn_decomposition/<run-id>/report.json   fits, gates, provenance
    <results>/bn_decomposition/<run-id>/triton-cache/ per-(BN,BM) compile evidence

`cells.csv` is appended and flushed per timing and a re-run resumes it. Each row
carries the `KernelTiming` columns the instrument produced (instrument,
warmup_ms, iters, trials, sm_clock_load_mhz, sm_clock_start_mhz,
sm_clock_end_mhz, clock_samples_mhz, power_w, clock_level_ok,
clock_level_side, clock_drift_ok, l2_flush) and the run's `provenance`
columns, so a row can be attributed to a commit, a card and a ruler without the
report beside it. The four clock-evidence columns were added on 2026-09-09
with the settle-on-clock warmup and this list was one writer behind them until
the same day: a drifted row that cannot say WHICH WAY its clock went is the
defect they exist to close, so a description that omits them describes the
artefact before the fix. The run id is built by
`moe.bench.provenance.run_id` from EVERY swept knob AND the card, because the
results root is a network volume shared between pods and this repo has already
had one card silently report another's timings twice.

HOW IT EXITS. Through `moe.bench.exit_codes.classify`, over the same gate
objects that printed the report: 0 DONE, 1 CLAIM_FAIL, 2 REFUSED, 3 INVALID,
4 ERROR. Every scored gate prints exactly one `RESULT: <KIND> <NAME> <VERDICT>
<detail>` line at column zero and nothing else in the output has that shape, so
the session driver reads verdicts rather than grepping prose -- the failure that
once let a REFUSED log's pre-registered `C1 ... [PASS]` be summarised as a
measurement.

AND WHAT A GROUP_SIZE_M=1 ARM EXITS WITH, because that arm is scheduled
unconditionally and two of its gates are PREDICTED not to pass there. C4 is
predicted to FAIL and C6, the estimator's sharpness, is predicted to FAIL with
it: both are CLAIM gates, so `classify` returns 1 CLAIM_FAIL and `_exit_over`
reports that as 0 without `--fail-on-gate`, which is what the driver's ledger
reads as finished. It is NOT 3 INVALID. C6 was a VALIDITY gate until 2026-09-02
and INVALID is what a G=1 arm returned, so the driver logged every one of them
RETRY and re-measured 11 GPU minutes on every pass, while the report under it
read UNKNOWN on alpha_a and PASS on everything a G=1 run is there for. INVALID
means nothing on the page may be quoted, and a predicted shortfall in one
estimator's power is not that; see `gate_sharpness`. The failing gates are still
on the page, still one RESULT line each, and `--fail-on-gate` still returns the
1 for a caller that wants a claim shortfall to be an error.

AND WHAT EVERY CELL IS SCORED ON BESIDE ITS ALPHA, FROM 2026-09-10. Two columns
were added to the cell table and to `report.json`, and neither is a new
estimator of the same thing:

  * `w = B / T_w`, the memory branch's slope in units of ONE COMPLETE STREAM of
    this layer's expert weights at the card's own measured rate. There is no
    fitted level in it, no intercept, no delta and no D, which is exactly what
    is wrong with `alpha = B/(A+B)` and its D-corrected partner: those divide by
    an extrapolation back to zero tiles over a lever arm of up to 44 treads on a
    ladder that is not exactly affine. `w` is the statistic every clean result
    in the 2026-09-10 analysis was expressed in. It is IMPORTED and not defined
    here, from `moe.bench.weights` and by name, see
    `WEIGHT_STREAM_CANDIDATES`; this file keeps no copy of that arithmetic to
    fall back to, and the page names the module every number on it was
    divided by.
  * the A-VERSUS-D label. `alpha_upper = B/(A + B - D)` exceeds 1 exactly when
    `D > A`, which is arithmetic and not physics. In the committed 2026-09-10
    arm that held in four of six cells and in NO others, and those are exactly
    the four whose `alpha_upper` came back above 1. Every cell now carries `A`,
    `B`, `D` and the label, so a reader can tell an unphysical measurement from
    an unphysical subtraction without re-fitting anything.

NEITHER C1's BAND NOR C6's BAR MOVED FOR THE FOURTH HEIGHT, and the arithmetic
is why. `ALPHA_A_BAND` is derived by `alpha_a_band_from_published` from the ONE
pair of committed arms that differ in BLOCK_SIZE_N and nothing else, at the two
BLOCK_M that fit in both; `ALPHA_A_SD_CEILING` is the sharpest two-point slope
in that same corpus, read back by `gate_sharpness`. Both are properties of what
the corpus has ALREADY measured, and adding a height to a future sweep changes
neither. Re-registering them here would have been a pre-registration moving
because new information arrived, which is the one thing a pre-registration may
not do. What the fourth height moves is what the run can RESOLVE, and that is
the design-power line, computed and printed on every plan page.

AND THAT LINE IS NO LONGER ONE SEED. `design_power` planted one world at
`--seed` and printed the sd that came back. At the fallback spread, which is the
worst published across-repeat spread and what a plan with no measurement of its
own is scored at, that figure ran 0.0223, 0.0868, 0.3102, 0.0684, 0.3445 and
0.4171 over six consecutive seeds of the THREE-height design, and seed 0 was the
only one of the six that cleared the 0.025 bar. A number that decides whether a
pod is rented may not be a property of the seed. Five realisations are planted,
the median is printed with all five beside it, and the verdict is scored on the
WORST. It is a pre-existing defect and not one the fourth height introduced: the
same six seeds on the FOUR-height design give 0.1774, 0.0831, 0.1136, 0.0696,
0.1734 and 0.0219.

OFF GPU. `--dry-run` prints the plan, the identification line the fourth height
was added for, the resource bill, the warp-grid bill, the per-cell predictions,
the computed design power and its MDE, and the cost. `--self-test`
plants four worlds -- the exact model, the model with a term missing, a world
with no activation re-read at all, and a noise-only world -- and checks the gates
come out DIFFERENT in each, which is the claim that they discriminate rather
than the claim that they pass.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402


def _load_sweep():
    """Load `block_m_crossing_sweep` BY PATH, and name what is missing.

    `scripts/` is not a package, so a bare import works only when this file is
    the entry point and fails silently when a test loads it by path.

    THE FIT IS IMPORTED, NEVER COPIED. Every alpha this script reports has to be
    the same estimator the study publishes, or the decomposition would be of a
    quantity nobody else measures. That file is also under active edit by
    another workstream, so names AND signatures are probed here, on a laptop,
    with a sentence that names the drift -- rather than as a TypeError thirty
    seconds into a metered pod session.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    needed = ("FIXED", "MEMORY_BRANCH_MARGIN", "MIN_MEMORY_TREADS",
              "PARALLEL_BRANCH_TOLERANCE", "SMEM_PER_BLOCK_BYTES",
              "MAX_REGISTERS_PER_THREAD", "compute_reference", "fit_ladder",
              "ComputeReference",
              "ladder_points", "make_cell", "tile_resources",
              "parse_capability", "tokens_for_rows", "rows_quantum",
              "results_root", "useful_flops",
              "weight_bytes_per_expert", "activation_bytes_per_row",
              "activation_slope_ms",
              "resolve_ridge", "RidgeUnavailable", "missing_gpu_stack",
              "find_override", "count_new", "balanced_ids",
              # THE INSTRUMENT'S NAMES, not the retired loop's. `time_call` was
              # in this list until 2026-09-02 and its presence was the only
              # thing standing between this file and a pod run timed with an
              # instrument the roof was never measured with. It is deliberately
              # NOT probed for any more: the symbol still exists over there and
              # raises `RetiredInstrument` when called, so asking for it would
              # keep passing while meaning nothing.
              "timing_basis", "reference_clock_mhz", "SYNTHETIC_INSTRUMENT")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/block_m_crossing_sweep.py no longer exports "
            f"{', '.join(missing)}. This script is deliberately scored by that "
            "file's fit rather than a private copy, so the two move together. "
            "Re-point the import; do not fork the fit.")
    import inspect
    for name, required in (("compute_reference", ("cfg", "ridge",
                                                  "bandwidth_gbps", "b",
                                                  "pinned", "capability")),
                           ("fit_ladder", ("block_m", "ref", "margin")),
                           ("make_cell", ("sm_count", "block_n")),
                           ("tile_resources", ("pinned", "block_m",
                                               "dtype_bytes", "capability"))):
        params = inspect.signature(getattr(module, name)).parameters
        gone = [p for p in required if p not in params]
        if gone:
            raise SystemExit(
                f"block_m_crossing_sweep.{name} no longer takes "
                f"{', '.join(gone)}. That file is under active edit and this "
                "one calls into it on the pod path; re-check the call sites in "
                "`fit_arm` and `main` before spending GPU time.")
    return module


SWEEP = _load_sweep()


def _load_power():
    """Load the study's power arithmetic BY PATH, for the same reason as the fit.

    `scripts/replicate_noise_floor.py` holds the two-sample, external-sigma and
    paired MDE forms and the normal and Student quantiles they are built on. An
    MDE quoted here from a fourth private copy of the same algebra is how "3.5x
    underpowered" once turned into an argument about whether it meant 3.5 or 12,
    so this file borrows the quantiles and names the ONE form it adds
    (`mde_one_sample`) rather than re-deriving any of them.

    IT ALSO CARRIES `sizing_sigma`, the accessor that decides WHICH sigma every
    MDE below is against: the measured between-replicate floor when part (a) has
    run on a card, the declared s3/s4 proxy labelled ASSUMED when it has not.
    The check is a list rather than a comment because this file already shipped
    the failure the accessor exists to remove -- it read `prior_sd` out of the
    JSON itself, which is the proxy and stays the proxy after a measured floor
    is published into the same file.
    """
    spec = importlib.util.spec_from_file_location(
        "replicate_noise_floor", ROOT / "scripts" / "replicate_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    missing = [n for n in ("normal_ppf", "TEST_LEVEL", "TEST_POWER",
                           "sizing_sigma", "NoiseFloorUnmeasured")
               if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/replicate_noise_floor.py no longer exports "
            f"{', '.join(missing)}. Every MDE printed by this file is one of "
            "that module's quantiles times a standard error, and the sigma "
            "under it is that module's one accessor; re-point the import "
            "rather than inlining a z value or re-reading prior_sd.")
    return module


POWER = _load_power()

TOLERANCE = SWEEP.PARALLEL_BRANCH_TOLERANCE
MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS


# --------------------------------------------------------------------------
# The numbers this script is arguing about, every one of them stated before any
# code that could be mistaken for measuring them.
# --------------------------------------------------------------------------

#: The primary. Not a flag: every sentence in the docstring is about the tile
#: vLLM actually runs multi-tile, and a `--subject` switch would let a run
#: answer a different question under this script's name.
PRIMARY_BLOCK_M = 128

#: Swept beside it. 64 and 32 are where alpha is robustly identifiable -- their
#: `B/C` sits far outside the parallel-branch band on both cards -- and the
#: alpha_b invariance test needs more than one BM to be a test at all. 16 joined
#: them on 2026-09-10 for the same two reasons and one more, below.
#:
#: 16 WAS ADDED ON 2026-09-10 BECAUSE THE GRID THIS ARM RAN COULD NOT IDENTIFY
#: ITS OWN MODEL. The 2026-09-10 H200 arm swept 32/64/128 against the 256
#: reference and BLOCK_M=128 produced NO alpha at ANY BLOCK_N: its memory branch
#: came within the parallel-branch tolerance of the compute branch at all three
#: and was discarded, so the fit that was pre-registered on three heights was
#: made on TWO, over six cells, for two parameters. `committed_bn_cells()`
#: reads those nine cells back out of the committed run rather than quoting
#: them. With two heights the model's activation column is not separable from
#: its rivals: `design_collinearity()` computes the correlation between
#: `g1(BM, BN)` and the `1/BN` column that fits the same data 3.1x better at
#: equal parameter count, and over the two-height grid it is 0.818.
#:
#: WHY 16 AND NOT A SECOND LARGE TILE, argued from what ran rather than from
#: preference. A smaller subject tile is further from its compute branch, so its
#: memory branch is the one least likely to be discarded, and the corpus
#: measures exactly that: `published_small_tile_branch()` reads the 2026-09-10
#: tile_cap arm, which ran BLOCK_M=16 at BLOCK_N=64 with the same num_warps and
#: num_stages this file pins, and found 56 of 56 treads memory bound at
#: B/C = 5.46 against a discard band of |B/C - 1| <= 0.15. Every one of that
#: arm's nine gates passed. Going the other way, to a fourth LARGE tile, adds a
#: height on the side where the branch is already being discarded.
#:
#: WHAT IT BUYS, COMPUTED AND NOT ASSERTED. Adding 16 at all three BLOCK_N drops
#: that collinearity from 0.818 to 0.663. THE GAIN IS ENTIRELY THE BN=32 CELL:
#: at BN=64 and BN=128 alone it is 0.819, i.e. nothing. So BLOCK_M=16 x
#: BLOCK_N=32 is not one cell of four, it is the cell the height is being added
#: for, and `warp_tile_bill` prints the one thing about it that is not like the
#: others rather than leaving it to be discovered on the pod.
SUBJECT_BLOCK_M = (16, 32, 64, 128)

#: The MMAv2 output tile, `m16n8k16`, in elements. It is what one warp holds
#: when Triton cannot reach the Hopper warpgroup path, which is every setting
#: here with BLOCK_M < 64: `block_m_crossing_sweep.FIXED` pins num_warps=8
#: precisely because that satisfies `BLOCK_M % 64 == 0 AND num_warps % 4 == 0`
#: "at every setting that can reach it", and 16 and 32 cannot.
#:
#: It is a RECORD AND NOT A REFUSAL, and the distinction is the point. The
#: shared-memory and accumulator bills are refusals because a setting that
#: exceeds either does not run the tiling this sweep is about at all: a
#: spilled kernel still returns a time. An under-occupied warp grid runs the
#: tiling correctly and merely wastes warps, and both cells that sit exactly ON
#: this boundary have run: BLOCK_M=32 x BLOCK_N=32 in the committed bn arm
#: (alpha 0.7927) and BLOCK_M=16 x BLOCK_N=64 in tile_cap. What the record buys
#: is that the one cell BELOW it, BLOCK_M=16 x BLOCK_N=32 at 512 elements
#: against 8 x 128, is named on the plan page before the pod runs it.
#:
#: THE RISK LANDS ON BOTH BRANCHES AND UNTIL 2026-09-10 THIS NOTE SAID
#: OTHERWISE. It scoped the incomparability to the COMPUTE branch, and nothing
#: argued the MEMORY branch was untouched. Four of eight warps holding no
#: output tile cuts memory-level parallelism as well as issue: fewer
#: independent `cp.async` streams in flight per CTA is fewer outstanding loads,
#: which inflates `B` and therefore `w` and `alpha` at that cell. So the
#: honest statement is that BOTH of this cell's branches are suspect, and it
#: matters more than it would at any other cell: `design_collinearity` puts the
#: whole 0.818 -> 0.663 identification gain on BLOCK_N=32, since BLOCK_M=16 at
#: BN=64 and BN=128 alone leaves the correlation at 0.819, unmoved.
#:
#: THE CORPUS CANNOT SETTLE IT, so the run registers a check instead.
#: BLOCK_M=32 x BLOCK_N=32 sits exactly ON the boundary and is "full", and
#: nothing published is under-occupied, so there is no measured under-occupied
#: ladder anywhere to compare against. `under_occupied_watch` states the check
#: the run itself can make: (16, 32)'s `w` against the trend that (32, 32) and
#: (64, 32) set at the SAME BLOCK_N, where the model and its `1/BN` rival both
#: predict a `w` that is smooth in BLOCK_M. A (16, 32) sitting above that trend
#: is the signature of the warp grid and not of the tiling, and it must be read
#: before the cell is used to separate the two hypotheses.
MMA_TILE_ELEMENTS = 16 * 8

#: The compute reference. `C ~ BLOCK_M` with no free parameter, so one ladder
#: that is compute bound throughout gives `C` at every block size, and 256 is
#: the only block size this study has found compute bound at tread 1. It must
#: be STRICTLY LARGER than every subject: a subject promoted to reference has
#: no memory branch by assumption and reports no alpha, which is exactly how the
#: published H200 BN=256 arm lost its BLOCK_M=128 cell without saying so.
REFERENCE_BLOCK_M = 256

#: The lever. Powers of two only -- `tl.arange` needs one -- and 16 is the
#: floor `tl.dot` accepts. 256 is absent by arithmetic and not by preference:
#: `BM=256 x BN=256` needs 256 accumulator registers per thread against a
#: hardware maximum of 255, so no reference can be timed there on ANY card, and
#: an arm with no reference contributes nothing but GPU time. `--block-n-list`
#: can ask for it anyway; the resource bill will refuse it out loud.
DEFAULT_BLOCK_N = (32, 64, 128)

#: How many BN arms must survive their reference before alpha_a is a fit rather
#: than a slope. Two points and two unknowns leave nothing over to test the
#: model with, which is the state this whole experiment exists to leave.
MIN_BN_POINTS = 3

#: The ONE pair of committed arms that differ in BLOCK_SIZE_N and in nothing
#: else, and the only measurement of alpha_a anywhere in this repository. Both
#: are H200 s4, mixtral, GROUP_SIZE_M=1, the same session, the same ridge.
PUBLISHED_BN_PAIR = (
    ROOT / "results" / "published" / "2026-09-01-nvidia_h200-alpha-surface-s4"
    / "mixtral-8x7b-bf16-r1024-g1-n64-d66ad3.report.json",
    ROOT / "results" / "published" / "2026-09-01-nvidia_h200-alpha-surface-s4"
    / "mixtral-8x7b-bf16-r1024-g1-n256-16cc16.report.json",
)

#: Where the cross-arm alpha floor lives. Read, never quoted, and read through
#: TWO accessors that are not interchangeable: `published_prior_sd` for the
#: sigma this run's MDEs are sized against, which prefers a MEASURED
#: between-replicate floor, and `preregistration_sigma` for the DECLARED prior
#: the pre-registered band is widened by, which is pinned. `PUBLISHED_ALPHA_SD`
#: below is the literal and the second of those is what checks it still is.
NOISE_FLOOR_PATH = ROOT / "results" / "published" / "NOISE_FLOOR.json"

#: The 2026-09-10 H200 session, which is this file's own test corpus: the arm it
#: ran, and the tile_cap arm whose BLOCK_M=16 ladder is the evidence the fourth
#: height rests on. Read on every plan page, never quoted: the numbers in the
#: `SUBJECT_BLOCK_M` note above are re-derived from these two directories by
#: `committed_bn_cells` and `published_small_tile_branch`, so a re-publish
#: that changed them would refuse rather than leave the paragraph standing.
SESSION_2026_09_10 = (ROOT / "results" / "published"
                      / "2026-09-10-nvidia_h200-gaps-session" / "results")
COMMITTED_BN_RUN = (
    SESSION_2026_09_10 / "bn_decomposition"
    / ("nvidia_h200-bm32_64_128-budget400.0-dtypebf16-flushtrue-g16-iters50"
       "-k64-modelmixtral_8x7b-n32_64-b59b409f"))
COMMITTED_SMALL_TILE_RUN = (
    SESSION_2026_09_10 / "tile_cap"
    / ("nvidia_h200-bm16-budget400.0-ctl256-dtypebf16-g1-iters50-l2flushtrue"
       "-modelmixtral_8x7b-n64-plant-ce661a4a"))

#: The tile height whose branch survival is the argument for the fourth subject,
#: and the arm that measured it. Named rather than inlined so the reader of
#: `published_small_tile_branch` sees which ladder is being read out of a report
#: that also carries a BLOCK_M=256 control.
SMALL_TILE_BLOCK_M = 16

#: Predicted alpha_a, PRE-REGISTERED AS A LITERAL AND CHECKED AGAINST THE FILES
#: IT CAME FROM. `alpha_a_band_from_published` re-derives it from
#: `PUBLISHED_BN_PAIR` on every run and `check_alpha_a_band` refuses when the two
#: disagree, so the band cannot quietly stop describing the corpus.
#:
#: WHAT IT USED TO SAY AND WHY THAT MATTERED. Until 2026-09-02 this was
#: (0.10, 0.15) "from the published two-point A100 slopes 0.106, 0.102, 0.129,
#: 0.119". Those four numbers exist in no file under results/published. The two
#: that do exist come from the one BN pair above and read 0.286 (BM=32) and
#: 0.146 (BM=64) with two-point sds of 0.086 and 0.043, so the retired band
#: excluded the larger of its own two inputs: a measured 0.28 would have printed
#: "FAIL high" against a band the repo's own data already sat outside.
#:
#: WHAT IT SAYS NOW: both measured slopes, each widened by its own sd, rounded
#: outward to a hundredth. That is 0.28 wide against input sds of 0.043-0.086,
#: i.e. 3.2 to 6.5 times the spread of the numbers that built it, and C1's gate
#: line says so: a PASS inside a band this wide is weak evidence and only a FAIL
#: is informative. Narrowing it is what a THREE-point BN fit is for, and that is
#: the experiment. Widening rather than keeping the old width is the honest
#: direction: the old width was not a tighter prior, it was a wrong one.
ALPHA_A_BAND = (0.10, 0.38)

#: TEMPO (arXiv:2608.13057) publishes these for the weight-side re-read, in two
#: configurations. They are the closest prior work and the only external number
#: alpha_b can be checked against. TEMPO models no activation-side re-read, so
#: alpha_a has no counterpart there.
TEMPO_B2_OVER_B = (0.311, 0.319)

#: How far alpha_b may sit from TEMPO's pair before C4 calls it a disagreement,
#: as a fraction. There is NO prior agreement for this to be set relative to:
#: the "decomposed 0.307 corroborates TEMPO to 2-4%" line the study once
#: printed is WITHDRAWN, because 0.307 was ALPHA_BY_BLOCK_M solved through
#: (LIN), a unit artefact of the estimator rather than a measurement of
#: alpha_b (the same two points through (EXA) give alpha_b 0.07, and the G=1
#: ladders read near 0.92). 0.15 is the width a three-point BN fit at this
#: run's bootstrap spread can resolve, and C4 names the swizzle it was scored
#: at -- see the note on C4.
TEMPO_TOLERANCE = 0.15

#: alpha_b measured at two block sizes must agree: nothing in the model lets a
#: weight miss fraction depend on the tile height. The threshold is in units of
#: the run's own bootstrap spread rather than a fixed number of alpha, because
#: the whole point is to compare a difference with the noise it was measured
#: through.
INVARIANCE_SIGMA = 3.0

#: The residual gate, C2, and the headline. `chi2 = sum (r/sigma)^2 / (n - p)`
#: over the surviving cells, with sigma the bootstrap spread of that cell's own
#: alpha. At or below 1 the residual IS the noise. 4.0 is two sigma RMS and is
#: the bar for calling the three terms complete; above it the model is missing
#: something and the structure test below names its shape.
RESIDUAL_CHI2_CEILING = 4.0

#: A structure test is only worth reading when the residual is big enough to
#: have a shape. Below this chi2 the correlations are reported and not read.
STRUCTURE_MIN_CHI2 = 1.0

#: |correlation| between the residual and a candidate missing regressor, past
#: which the residual is called structured rather than scattered.
STRUCTURE_CORRELATION = 0.90

#: alpha_a's bootstrap spread has to be smaller than this or C1 cannot
#: distinguish the predicted band from its alternatives and reads UNKNOWN
#: instead of PASS.
#:
#: IT IS NO LONGER HALF THE BAND WIDTH, and the number did not move. It was
#: written as "half the width of [0.10, 0.15]", so widening the band to the one
#: the corpus supports would have carried the ceiling to 0.14 and made C6 pass
#: on an estimator six times looser than the one it was written to require. A
#: gate that loosens because its hypothesis got vaguer is not a gate. The bar is
#: instead the thing this experiment claims to improve on: the SHARPEST
#: two-point slope in the corpus has sd 0.043 (BM=64), and a three-point fit
#: that cannot beat the two-point reading it replaces has not replaced it. 0.025
#: is comfortably inside that, and `--self-test` is what shows whether a given
#: pinning reaches it.
#:
#: `gate_sharpness` READS that 0.043 back out of the corpus rather than printing
#: this paragraph's copy of it, because the gate line and the constant's
#: justification drifted apart once already: the rule string went on saying
#: "half the width of the [0.10, 0.38] band" for a band whose half-width is
#: 0.14, five times this ceiling, beside the very widening that refuted it.
ALPHA_A_SD_CEILING = 0.025

#: A compute reference must imply at least this fraction of the ATTACHED card's
#: calibrated peak, and no more than all of it. The corrupt A100 BLOCK_N=256
#: reference implies 1.4% and its H200 twin 12.3%; the 22 sound published
#: references run 38.2% to 63.7%. Nothing runs faster than the roof, so the top
#: end is physics; 25% is set below every sound reference and 18x above the
#: worst corrupt one.
REFERENCE_LEVEL_FLOOR = 0.25
REFERENCE_LEVEL_CEILING = 1.0

#: Spread of the reference's implied rate ACROSS BN, past which the references
#: are not all measuring the same machine. Occupancy really does move the
#: achieved rate with BN -- the N-tile count per M-tile changes by 4x across
#: this grid -- so the bar is loose. The failure it exists to catch is 43.6x.
REFERENCE_CROSS_BN_SPREAD = 2.0

#: Across-repeat spread above which a whole arm's timings are too noisy to fit,
#: whatever they say. The published H200 ladders sit at 0.76-1.82% on one pass
#: and the A100 ones at 0.48-0.61%.
MAX_REPLICATE_SPREAD = 0.02

#: An inversion -- time falling as tiles rise -- beyond this many across-repeat
#: standard deviations is a fault and not noise. Two rather than three because
#: the direction is known a priori: both branches have positive slope.
MONOTONE_SIGMA = 2.0

#: Resamples behind every spread reported here. Fixed and seeded, so two
#: readers of one cells.csv get the same intervals.
BOOTSTRAP_DRAWS = 1000

#: The published cross-arm floor on alpha, carried for context only and never
#: used as a gate: `NOISE_FLOOR_PATH` records a paired s3-vs-s4 sd of 0.0323 over
#: 11 cells and a `prior_sd` of 0.0228 (that sd over root two, an upper bound
#: because it confounds num_stages). This run's own bootstrap is what C2 is
#: scored against, because a floor measured on other arms cannot know how noisy
#: THIS pod was.
#:
#: IT IS THE DECLARED PRIOR AND NOT NECESSARILY WHAT THE REPORT PRINTS. Once
#: part (a) publishes a measured between-replicate floor, `published_prior_sd`
#: returns THAT, labelled MEASURED, and this literal goes on describing the
#: pre-registration alone. The two are checked against different accessors on
#: purpose; see `preregistration_sigma`.
#:
#: THE TWO ARE DIFFERENT KINDS OF NOISE AND THE REPORT NOW PRINTS BOTH. The
#: bootstrap resamples WITHIN-PROCESS WARM REPEATS: one process, one allocation,
#: one clock state, repeats interleaved round-robin. It cannot see anything that
#: changes between processes -- a re-import, a re-allocation, a different pod
#: hour -- and every published cross-arm difference in this study is exactly
#: that kind of comparison. So an interval from here is a LOWER bound on the
#: uncertainty of any number compared across arms, and this floor is the only
#: measured upper bound the repo has. Quoting the first alone is how an effect
#: smaller than the floor gets a sigma that makes it look resolved.
PUBLISHED_ALPHA_SD = 0.0228

#: `--plant-noise` when nothing measured is available to resolve it from: the
#: WORST across-repeat spread the corpus has shown (H200 ladders, 1.82%), not
#: the middle of the range. S4's design-power verdict is a claim about whether
#: the pod run can answer P1, and a claim like that scored at the median of the
#: observed spreads is scored at a pod half the sessions were noisier than. The
#: audit measured what that choice bought: at 0.8% S4 passes at G=16 and at
#: 1.5-2.0% it fails, so the old default made the design look resolvable at a
#: spread the H200 has repeatedly exceeded.
PLANT_NOISE_FALLBACK = 0.0182
PLANT_NOISE_FALLBACK_SOURCE = (
    "the worst published across-repeat spread (H200 s4 ladders 0.76-1.82%, "
    "A100 0.48-0.61%); nothing measured was available to resolve it from")

#: The token `--plant-noise` contributes to the run id when it was not given.
#: The RESOLVED value must never enter the id: on a real run it is derived from
#: the cells the id names, so an id containing it would depend on its own
#: directory's contents and change halfway through the sweep.
PLANT_NOISE_AUTO = "auto"

#: Bootstrap draws behind the C2 power probe and the plan's design-power line,
#: as distinct from the draws behind the report's own intervals. Fewer, because
#: what is read off them is one sd rather than a published interval: the
#: relative error of a bootstrap sd is about 1/sqrt(2 draws), which is 5% here,
#: and the probe runs twice on the pod path where each draw rebuilds every arm.
POWER_PROBE_DRAWS = 200

#: Planted realisations behind the design-power line. FIVE AND NOT ONE, from
#: 2026-09-10: the single-seed figure this line used to carry swings 19-fold
#: between seeds at the fallback spread, and it is the number an operator reads
#: to decide whether to rent a pod. Five is the smallest odd count whose median
#: is not one draw and whose maximum is a real worst case; it costs five times
#: `--power-draws` of arithmetic and no GPU at all. The C2 power probe is NOT
#: multiplied by it: that probe asks a different question, about the run that
#: has already been paid for, at the spread that run measured.
DESIGN_POWER_SEEDS = 5

#: The card slug a run id carries when no device is attached: every --dry-run
#: and every --self-test on a laptop. Visible rather than blank, so a laptop
#: directory cannot be mistaken for the one a pod would write to.
NO_CARD_SLUG = "nocard"

#: Two worlds, both stated as (alpha_b, alpha_a), used ONLY to predict and to
#: cost. Neither is a measurement of this run.
#:
#:  LIN-ERA  the study's own ALPHA_BY_BLOCK_M = {64: 0.466, 128: 0.625} solved
#:           through (LIN), the (0.307, 0.143) pair moe/bench/ai_model.py USED
#:           to quote and has WITHDRAWN: a unit artefact of reading a B/(A+B)
#:           fit as if it divided by weight bytes alone (the same two points
#:           through (EXA) give alpha_b 0.07 and alpha_a 0.72), and pooled
#:           over GROUP_SIZE_M besides, which swings alpha by 0.39. Kept as a
#:           prediction world because it is the reading the retraction
#:           replaces, and a run has to be able to land in it.
#:  LADDER   this study's own G=1 ladder fits, which measure alpha_fitted at
#:           0.916-0.951 at BM=64, BN=64 on BOTH cards with all 16 treads memory
#:           bound. Read through (EXA) with alpha_a at the (LIN) value, that is
#:           alpha_b near 0.92 -- a nearly full re-read per M-tile, which is
#:           what GROUP_SIZE_M=1 means: consecutive M-tiles of one expert are
#:           scheduled far apart and the L2 keeps nothing.
#:
#: They disagree about alpha_b by a factor of three and they are BOTH derived
#: from published numbers in this repo. Which one this run lands in is a result.
WORLD_LIN_ERA = (0.307, 0.143)
WORLD_LADDER = (0.920, 0.146)

#: alpha_b implied by the published mixtral BLOCK_M=64, BLOCK_N=64 ladder fits,
#: BY SWIZZLE, which is what `--self-test` plants and what the design-power gate
#: is scored on. Read off the committed reports through (EXA) at delta = 0:
#: `alpha_b = alpha + phi (alpha - 1)` with phi(64, 64) = 0.158.
#:
#:     G= 1  alpha 0.9475 -> 0.939     G= 8  alpha 0.6994 -> 0.652
#:     G=16  alpha 0.6595 -> 0.606     G=64  alpha 0.7447 -> 0.705
#:
#: TAKEN AT delta = 0, WHICH IS THE FAVOURABLE END. A positive fixed cost pushes
#: the implied alpha_b UP, and a higher alpha_b makes the BN lever weaker, so
#: the design power computed from this table is an UPPER bound on the design's
#: power and the real run is at least this hard. At G=1 the fixed-cost-corrected
#: value goes ABOVE 1, which no miss fraction can be -- see C5.
PLANTED_ALPHA_B = {1: 0.94, 8: 0.65, 16: 0.61, 64: 0.71}


def planted_alpha_b(group_m: int) -> float:
    """The corpus's alpha_b at this swizzle, or the G=1 value as the hard case."""
    return PLANTED_ALPHA_B.get(group_m, PLANTED_ALPHA_B[1])


# --------------------------------------------------------------------------
# What the corpus actually says, read out of the committed files rather than
# quoted from them. Every number in this section had a version that was typed
# into a comment and did not survive a check against its own source.
# --------------------------------------------------------------------------

class CorpusMissing(RuntimeError):
    """A committed file this file's pre-registration rests on is not readable.

    Raised, not defaulted around. A band whose provenance cannot be read is a
    band whose provenance cannot be checked, and this file already shipped one
    of those: four A100 slopes that no file ever contained.
    """


def _read_report(path: Path, what: str) -> dict:
    """One committed report.json, or a refusal that names the file and the use.

    Every corpus reader in this file goes through here so that a moved or
    re-published directory refuses on a laptop, in one sentence, rather than
    raising a decode error thirty seconds into a metered pod session.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CorpusMissing(
            f"{path} is not there, and it is what {what}. Nothing in this file "
            "may quote that number from memory; re-point the constant at the "
            "published directory or re-publish it.") from exc
    except json.JSONDecodeError as exc:
        raise CorpusMissing(
            f"{path} is not readable JSON ({exc}), and it is what {what}.") from exc


@dataclass(frozen=True)
class SmallTileBranch:
    """What the corpus measured at the tile height this file is adding.

    `B/C` is the statistic the parallel-branch tolerance is applied to, so it is
    the one that says whether a height yields a branch at all, and it is formed
    here from the two slopes the tile_cap report publishes rather than from any
    fit made here.
    """

    block_m: int
    block_n: int
    group_m: int
    num_warps: int
    num_stages: int
    memory_points: int
    treads: int
    slope_memory: float
    slope_compute: float
    gates_passed: int
    gates_total: int
    source: str

    @property
    def branch_ratio(self) -> float:
        """`B/C`. Discarded by `fit_ladder` when it lands within the tolerance."""
        return self.slope_memory / self.slope_compute

    def lines(self) -> list[str]:
        return [
            f"BLOCK_M={self.block_m} measured at BLOCK_N={self.block_n}, "
            f"GROUP_SIZE_M={self.group_m}, num_warps={self.num_warps}, "
            f"num_stages={self.num_stages}: {self.memory_points} of "
            f"{self.treads} treads memory bound at B/C = "
            f"{self.branch_ratio:.2f}, against the parallel-branch tolerance "
            f"|B/C - 1| <= {TOLERANCE:.2f} that discarded every BLOCK_M="
            f"{PRIMARY_BLOCK_M} cell in the committed bn arm. "
            f"{self.gates_passed} of {self.gates_total} gates PASS.",
            f"  source: {self.source}"]


def published_small_tile_branch(path: Path | None = None) -> SmallTileBranch:
    """The committed BLOCK_M=16 ladder, read out of tile_cap's own report.

    THE REASON THE FOURTH HEIGHT IS NOT A GUESS. `SUBJECT_BLOCK_M`'s note claims
    that a smaller subject tile is further from its compute branch and therefore
    the one least likely to have its memory branch discarded. That claim is
    checkable, and this is the check: the same card, the same num_warps and
    num_stages, one BLOCK_N in common with this sweep, and the two slopes whose
    ratio the discard rule is applied to.
    """
    path = path or (COMMITTED_SMALL_TILE_RUN / "report.json")
    what = ("says whether BLOCK_M=16 yields a memory branch at all, which is "
            "the whole argument for sweeping it")
    payload = _read_report(path, what)
    ladders = payload.get("ladder") or {}
    ladder = ladders.get(str(SMALL_TILE_BLOCK_M))
    if not ladder or ladder.get("slope_memory") is None:
        raise CorpusMissing(
            f"{path} carries no BLOCK_M={SMALL_TILE_BLOCK_M} ladder with a "
            f"memory branch, and it is what {what}. Without it the fourth "
            "subject height rests on nothing measured.")
    fixed = payload.get("fixed") or {}
    gates = payload.get("gates") or []
    return SmallTileBranch(
        block_m=SMALL_TILE_BLOCK_M,
        block_n=int(fixed.get("BLOCK_SIZE_N", 0)),
        group_m=int(fixed.get("GROUP_SIZE_M", 0)),
        num_warps=int(fixed.get("num_warps", 0)),
        num_stages=int(fixed.get("num_stages", 0)),
        memory_points=int(ladder["memory_points"]),
        treads=len(ladder.get("points") or ()),
        slope_memory=float(ladder["slope_memory"]),
        slope_compute=float(ladder["slope_compute"]),
        gates_passed=sum(1 for g in gates if g.get("verdict") == "PASS"),
        gates_total=len(gates),
        source=str(path))


def committed_bn_cells(path: Path | None = None
                       ) -> list[tuple[int, int, float | None]]:
    """`(BLOCK_N, BLOCK_M, alpha)` for every cell of the committed bn arm.

    The arm this file ran on 2026-09-10, read back so that "BLOCK_M=128 produced
    no alpha at any BLOCK_N" is a fact this run re-derives rather than a
    sentence left over from the session it describes.
    """
    what = ("records which cells the committed BLOCK_N sweep actually yielded, "
            "and so which heights the fit was really made over")
    payload = _read_report(path or (COMMITTED_BN_RUN / "report.json"), what)
    return [(int(c["block_n"]), int(c["block_m"]), c["alpha"])
            for c in payload.get("cells", [])]


def design_collinearity(cfg, cells) -> float | None:
    """|corr| between the model's activation column and the `1/BN` rival.

    THE NUMBER THAT SAYS WHETHER A HEIGHT WAS WORTH ADDING. The three-term
    model's only BLOCK_N-dependent term is `alpha_a g1(BM, BN)`, which is
    strictly proportional to BLOCK_M. The 2026-09-10 session refuted that on the
    slope alone and named the replacement that fits 3.1x better at equal
    parameter count: a cost going as `1/BN` and NOT with BLOCK_M. Over a grid
    where those two columns are collinear the two hypotheses are one hypothesis,
    and no amount of repeats separates them.

    Returned as a magnitude, and computed from the DESIGN alone, the block
    sizes and not any timing, so `--dry-run` prints it before the pod is rented.
    """
    g1 = [g1_term(cfg, bm, bn) for bm, bn in cells]
    inv = [1.0 / bn for _, bn in cells]
    r = pearson(g1, inv)
    return None if r is None else abs(r)


@dataclass(frozen=True)
class TwoPoint:
    """One two-point alpha_a slope, and everything needed to doubt it."""

    block_m: int
    block_n_lo: int
    block_n_hi: int
    alpha_lo: float
    alpha_hi: float
    #: `BM (1/BN_lo - 1/BN_hi)`, the change in `s = BM/BN` the pair spans. The
    #: whole two-point weakness lives here: it is 0.375 at BM=32 and 0.75 at
    #: BM=64, so the SAME alpha difference reads twice as large a slope at the
    #: smaller tile, and the sd divides by it too.
    delta_s: float
    slope: float
    sd: float
    source: str

    def line(self) -> str:
        return (f"BM={self.block_m:3d}  alpha {self.alpha_lo:.4f} "
                f"(BN={self.block_n_lo}) - {self.alpha_hi:.4f} "
                f"(BN={self.block_n_hi}) over ds {self.delta_s:.3f}  ->  "
                f"alpha_a {self.slope:.3f} +/- {self.sd:.3f}")


@dataclass(frozen=True)
class CrossArmFloor:
    """The sigma this run's intervals are read against, and the ONE WORD that
    says whether anybody measured it.

    The word travels WITH the number, in one object composed in one place,
    because the failure being fixed here is a floor printed without it: a sigma
    on a page with no basis beside it reads as a measurement, and for most of
    this study's life it is the declared prior instead.
    """

    sd: float
    #: MEASURED (a between-replicate floor from part (a)) or ASSUMED (the
    #: s3/s4 proxy, an upper bound that confounds num_stages with rerun noise).
    basis: str
    source: str

    def line(self) -> str:
        """The report's own sentence. One composer, so the word cannot be
        dropped at one of the places the floor is printed and kept at the
        other."""
        return (f"cross-arm floor {self.sd:.4f} {self.basis} ({self.source}), "
                "carried beside the intervals and never used as a gate")

    def clause(self) -> str:
        """The tail `mde_line` hangs off an MDE.

        THE CLOSING HALF TURNS ON THE BASIS, AND SO DOES THE WORD "BOUND".
        "the repo's only MEASURED upper bound" was printed unconditionally
        while the floor was the declared proxy, which is the sentence this
        slice began by stopping; the first fix kept "a MEASURED upper one",
        which is the same mistake one word smaller. UPPER BOUND was only ever
        true of the s3/s4 proxy, and only because that proxy confounds
        `num_stages` with rerun noise and so can only overstate the spread. A
        between-replicate floor measures exactly the quantity wanted and is a
        POINT ESTIMATE of it, not a bound on it.

        WHAT THE SURVIVING WORD PRINTED. Under a measured floor of 0.0091,
        `--self-test --plant-noise 0.008` produced "from sd 0.0098 ... it is a
        lower bound and the floor is a MEASURED upper one": the named upper
        number smaller than the named lower one inside one clause, which reads
        as an instrument that cannot be trusted rather than as two spreads that
        happen to be close. `alpha_surface.print_mde` is the sibling printer
        this slice wrote and it drops the words on its MEASURED branch, so this
        was one of the two places that had to agree, fixed at one of them.
        """
        if self.basis == "MEASURED":
            tail = ("a between-replicate POINT ESTIMATE of the same quantity "
                    "and not a bound on it, so the two are comparable in "
                    "either direction")
        else:
            tail = "the DECLARED upper one and not a measurement"
        return (f". Beside it, the cross-arm floor {self.sd:.4f} "
                f"{self.basis} ({self.source}): this run's own bootstrap "
                "resamples WITHIN-PROCESS warm repeats and cannot see anything "
                "that changes between processes, so it is a lower bound and "
                f"the floor is {tail}")


def published_prior_sd(path: Path | None = None) -> CrossArmFloor:
    """The floor every interval this run prints is read against, with its basis.

    THE BUG THIS FIXES, and it is the second call site of a fix that already
    landed. This read `payload["prior_sd"]` straight out of NOISE_FLOOR.json.
    That field is the s3-vs-s4 PROXY and it stays the proxy after part (a)
    spends 120 minutes of card measuring a real between-replicate spread: the
    measurement is published into `replicate_floor` of the SAME file. So every
    MDE this script printed, and every "below the detection limit" verdict it
    issued, would have gone on being scored against the assumption after the
    arm that bought the measurement had run.
    `replicate_noise_floor.sizing_sigma` is the ONE accessor that prefers the
    measurement, falls back to the proxy with the word ASSUMED, and never
    invents a number; this is now the only way this file gets a sigma to print.

    IT IS NOT THE SIGMA THE PRE-REGISTERED BAND IS WIDENED BY. That one is
    `preregistration_sigma`, and the difference is deliberate; see there.

    Raises `CorpusMissing` rather than defaulting, including on a file that is
    present but truncated: `sizing_sigma` parses before its own guards run, so
    a decode error arrives here as a ValueError and would otherwise reach the
    pod path as a traceback instead of a REFUSED.
    """
    target = Path(path or NOISE_FLOOR_PATH)
    try:
        sd, basis, source = POWER.sizing_sigma(target)
        # `sizing_sigma` names the file by ABSOLUTE path and this string is
        # written into report.txt. A floor line that reads
        # /workspace/whatever is one nobody can compare across pods, and
        # `scripts/alpha_surface.py` does the same to the same string: one
        # rule, and it is now at both of its call sites.
        source = source.replace(str(ROOT) + "/", "")
    except POWER.NoiseFloorUnmeasured as exc:
        raise CorpusMissing(
            f"{target} carries no sigma this run can be sized against: {exc}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise CorpusMissing(
            f"{target} is not readable ({type(exc).__name__}), and it is the "
            "only cross-arm alpha floor in this repository. Every interval "
            "this run reports comes from within-process repeats and needs that "
            "floor printed beside it to be read honestly.") from exc
    return CrossArmFloor(float(sd), basis, source)


def preregistration_sigma(path: Path | None = None) -> tuple[float, str]:
    """The DECLARED prior, `prior_sd`, pinned on purpose. `(sd, source)`.

    WHY THIS IS NOT `published_prior_sd`, which now prefers a measurement. The
    two-point slopes below are widened by a sigma, `ALPHA_A_BAND` is the band
    those widened slopes produce, and `check_alpha_a_band` REFUSES the run when
    the literal band and the re-derived one disagree. Feed a measured floor into
    that chain and the band moves the instant part (a) publishes: at a floor of
    0.0091 the two-point sds fall to 0.034 and 0.017, the band re-derives as
    (0.12, 0.33), and the first pod command after the noise-floor arm refuses
    with "the pre-registered alpha_a band is not what the committed BN pair now
    says". Nothing about the corpus would have changed. `ALPHA_A_SD_CEILING`
    fails the same way: it is set inside the sharpest two-point sd, 0.043, and a
    measured floor would put that sharpest sd at 0.017, below the ceiling that
    exists to be beaten.

    A PRE-REGISTRATION THAT MOVES WHEN NEW INFORMATION ARRIVES IS NOT ONE. The
    band was registered against the declared prior and stays registered against
    it; the measurement changes what this run can RESOLVE, which is the MDE and
    the floor printed beside every interval, and those are what
    `published_prior_sd` feeds.
    """
    target = Path(path or NOISE_FLOOR_PATH)
    try:
        doc = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        raise CorpusMissing(
            f"{target} is not readable ({type(exc).__name__}), and the "
            "pre-registered alpha_a band is widened by the prior it declares. "
            "A band derived from a file nobody can open is a band whose "
            "provenance cannot be checked.") from exc
    sd = doc.get("prior_sd")
    if not isinstance(sd, (int, float)) or isinstance(sd, bool) or not sd > 0:
        raise CorpusMissing(f"{target} carries no positive prior_sd; got {sd!r}")
    return float(sd), str(doc.get("prior_sd_source",
                                  "no source recorded in the file"))


def read_cross_arm_floor(path: Path | None = None
                         ) -> tuple[CrossArmFloor | None, str]:
    """`(floor, "")`, or `(None, why not)` for a caller that must print anyway.

    `_main` calls `published_prior_sd` directly and REFUSES on it, because a run
    whose floor cannot be read should not be paid for. The report and the S4
    gate run AFTER that check and must still render when a test hands them a
    tree with no published corpus in it, so they take the pair and print
    UNAVAILABLE with the reason. Both of them go through here rather than
    writing their own try/except, which is how the two of them came to print
    two different sentences about the same file.
    """
    try:
        return published_prior_sd(path), ""
    except CorpusMissing as exc:
        return None, str(exc)


def published_two_point_alpha_a(pair=PUBLISHED_BN_PAIR, sigma: float | None = None
                                ) -> list[TwoPoint]:
    """Every two-point alpha_a the corpus supports, read out of the two reports.

    THE SLOPE. (LIN) says `alpha = alpha_b + alpha_a s + BM/K` with `s = BM/BN`,
    and BM is held fixed across the pair, so everything but the middle term
    cancels and `alpha_a = (alpha(BN_lo) - alpha(BN_hi)) / (BM (1/BN_lo -
    1/BN_hi))`. Two points, two unknowns, nothing left over: that is exactly the
    state this experiment exists to leave, and reading it here is how the band
    it leaves gets a provenance.

    THE COLUMN IS `alpha_corrected`, the published alpha with the reference's
    fixed cost removed, because that is the quantity the decomposition fits.
    The raw `alpha` column gives 0.284 and 0.145 against this one's 0.286 and
    0.146, so the choice moves nothing; it is stated because an unstated one is
    how two readings of one number become an argument.

    `sigma` is the sd of ONE arm's alpha; the difference of two carries
    `sigma sqrt(2)`, and the slope that sd over `delta_s`. Default is the
    DECLARED prior (`preregistration_sigma`), which is an upper bound and
    therefore the conservative choice for a band this is going to be widened
    by, and which is pinned rather than measured so that the band and the
    sharpness ceiling built on these sds cannot move under a run that has
    already been pre-registered against them.
    """
    if sigma is None:
        sigma, _ = preregistration_sigma()
    docs = []
    for path in pair:
        try:
            docs.append((path, json.loads(Path(path).read_text())))
        except (OSError, ValueError) as exc:
            raise CorpusMissing(
                f"{path} is not readable ({type(exc).__name__}). It is one of "
                "the two arms the pre-registered alpha_a band is derived from, "
                "and a band derived from a file nobody can open is the defect "
                "this function was written to remove.") from exc
    (lo_path, lo_doc), (hi_path, hi_doc) = docs
    bn_lo = int(lo_doc["fixed"]["BLOCK_SIZE_N"])
    bn_hi = int(hi_doc["fixed"]["BLOCK_SIZE_N"])
    if bn_lo == bn_hi:
        raise CorpusMissing(
            f"{lo_path.name} and {hi_path.name} both pin BLOCK_SIZE_N={bn_lo}, "
            "so they are not a BN pair and no slope exists between them")
    for key in ("GROUP_SIZE_M", "num_stages", "BLOCK_SIZE_K"):
        if lo_doc["fixed"].get(key) != hi_doc["fixed"].get(key):
            raise CorpusMissing(
                f"{lo_path.name} and {hi_path.name} differ in {key} as well as "
                "in BLOCK_SIZE_N, so the difference between them is not a "
                "BLOCK_SIZE_N effect")
    out: list[TwoPoint] = []
    for bm_key, lo_fit in sorted(lo_doc.get("ladder", {}).items(),
                                 key=lambda kv: int(kv[0])):
        hi_fit = hi_doc.get("ladder", {}).get(bm_key)
        if hi_fit is None:
            continue
        a_lo, a_hi = lo_fit.get("alpha_corrected"), hi_fit.get("alpha_corrected")
        if a_lo is None or a_hi is None:
            continue
        bm = int(bm_key)
        ds = bm * (1.0 / bn_lo - 1.0 / bn_hi)
        if ds == 0:
            continue
        out.append(TwoPoint(bm, bn_lo, bn_hi, float(a_lo), float(a_hi), ds,
                            (float(a_lo) - float(a_hi)) / ds,
                            sigma * math.sqrt(2.0) / abs(ds),
                            f"{lo_path.name} vs {hi_path.name}"))
    if not out:
        raise CorpusMissing(
            f"{lo_path.name} and {hi_path.name} share no BLOCK_M with an "
            "identifiable ladder in BOTH arms, so the repo contains no "
            "two-point alpha_a at all and the band has no provenance")
    return out


def alpha_a_band_from_published(points: list[TwoPoint] | None = None
                                ) -> tuple[float, float]:
    """The band the corpus supports: every measured slope, widened by its own sd.

    Rounded OUTWARD to a hundredth, so the band can only ever be looser than its
    inputs and a rounding can never exclude a measurement that is inside.
    """
    points = points if points is not None else published_two_point_alpha_a()
    lo = min(p.slope - p.sd for p in points)
    hi = max(p.slope + p.sd for p in points)
    return (math.floor(lo * 100.0) / 100.0, math.ceil(hi * 100.0) / 100.0)


def check_alpha_a_band(band=ALPHA_A_BAND) -> tuple[list[TwoPoint], list[str]]:
    """Re-derive the band and REFUSE when the literal no longer matches.

    Called before any GPU time on every path that scores C1. The pre-registered
    band stays a literal -- a hypothesis nobody can see is not pre-registered --
    and this is what stops the literal outliving the files it came from, which
    is the exact failure it is replacing.

    THE LINES NAME THEIR SIGMA AND SAY IT IS PINNED, added 2026-09-03. Every
    `+/-` printed below, and the `input sds of ...` that follows them, is the
    DECLARED prior over that pair's `delta_s` and nothing else. The report
    prints a `cross-arm floor ... MEASURED 0.0091` fourteen lines under a
    `+/- 0.086` built from 0.0229, and until this clause existed no sentence on
    the page connected the two: the natural misreading is that the two-point
    reading has become sharper than the C6 ceiling and that the ceiling is
    stale. It has not, and it is not. `preregistration_sigma` argues why the
    pinning is right, but that argument lived only in the source and an
    operator reads the page.
    """
    points = published_two_point_alpha_a()
    derived = alpha_a_band_from_published(points)
    if tuple(round(v, 4) for v in derived) != tuple(round(v, 4) for v in band):
        raise CorpusMissing(
            f"the pre-registered alpha_a band {band} is not what the committed "
            f"BN pair now says ({derived}). Either the corpus was re-published "
            "or the literal was edited without its source; do not score C1 "
            "against a band whose provenance no longer reads back.")
    widest = max(p.sd for p in points)
    narrowest = min(p.sd for p in points)
    width = band[1] - band[0]
    # `preregistration_sigma` and NOT `published_prior_sd`: naming the wrong
    # one here would print the very substitution the sentence warns against.
    # It cannot raise, because the call above went through the same reader.
    prior, _ = preregistration_sigma()
    lines = [
        f"alpha_a band [{band[0]:.2f}, {band[1]:.2f}], derived from the ONLY "
        "pair of committed arms that differ in BLOCK_SIZE_N and nothing else:"]
    lines += ["  " + p.line() for p in points]
    lines += [
        f"  source: {points[0].source}",
        f"  every +/- above is the DECLARED prior {prior:.4f}, times sqrt(2) "
        "for the difference of two arms, divided by that pair's ds. It is "
        "PINNED to the declared number and deliberately NOT re-read from a "
        "measured floor: a pre-registration that moves when new information "
        "arrives is not one, so this band and the C6 bar built on it stay "
        "where they were registered even after part (a) publishes. What a "
        "measured floor moves is what the run can RESOLVE, which is the MDE "
        "and the cross-arm floor printed beside every interval below.",
        f"  the band is {width:.2f} wide against input sds of "
        f"{narrowest:.3f}-{widest:.3f}, i.e. {width / widest:.1f}-"
        f"{width / narrowest:.1f}x the spread of the numbers that built it. A "
        "PASS inside it is weak evidence; a FAIL outside it is not. Narrowing "
        "it is what a three-point BN fit is for, which is this run.",
    ]
    return points, lines


# --------------------------------------------------------------------------
# Power, in the one form this file adds to the study's shared arithmetic.
# --------------------------------------------------------------------------

def mde_one_sample(sd: float, *, level: float | None = None,
                   power: float | None = None) -> float:
    """Smallest alpha_a offset a single estimate with this sd can resolve.

    `(z_{1-level/2} + z_{power}) sd`, the one-sample known-sigma form, which is
    the design this file runs: ONE fit, one bootstrap sd, compared against a
    pre-registered band rather than against a second arm.
    `scripts/replicate_noise_floor.py` holds the two-sample, external-sigma and
    paired forms and this is the fourth; naming it rather than reaching for
    `mde_external_sigma` matters because that one carries a `sqrt(2)` for the
    second arm this design does not have and would overstate the limit by 41%.

    Raises on a non-positive sd: an MDE from no spread is not zero, it is
    unknown, and zero is the value that would make every effect look resolvable.
    """
    if sd is None or not sd > 0:
        raise ValueError(f"an MDE needs a positive sd, got {sd!r}")
    level = POWER.TEST_LEVEL if level is None else level
    power = POWER.TEST_POWER if power is None else power
    return (POWER.normal_ppf(1.0 - level / 2.0) + POWER.normal_ppf(power)) * sd


def mde_line(sd: float | None, *, what: str, assumption: str,
             floor: CrossArmFloor | None = None) -> str:
    """The one MDE sentence every plan and every report prints.

    THE NOISE ASSUMPTION IS IN THE SENTENCE, not in a constant three screens
    away, because an MDE is a statement about an assumed spread and quoting it
    without one is how a limit becomes a fact. When `sd` is None the line says
    the MDE is UNKNOWN and why; it never prints a number.

    `floor` IS THE OBJECT AND NOT A BARE NUMBER, changed 2026-09-03. It used to
    be `(float, str)` and the basis word did not exist, so the same sentence
    was printed whether the sigma had been measured or assumed. Passing the
    object makes the word impossible to drop at one of the two call sites that
    hand a floor in, which is the shape of defect this rebuild has now found
    eight times.
    """
    head = f"MDE ({what}, two-sided {POWER.TEST_LEVEL:.0%} at " \
           f"{POWER.TEST_POWER:.0%} power): "
    if sd is None or not sd > 0:
        body = f"UNKNOWN, no spread to derive one from. Assumption: {assumption}"
    else:
        body = (f"{mde_one_sample(sd):.4f} in alpha_a, from sd {sd:.4f}. "
                f"Assumption: {assumption}")
    if floor is not None:
        body += floor.clause()
    return head + body


# --------------------------------------------------------------------------
# The geometry. Pure arithmetic on a model config: no torch, no GPU, no files.
# Every number the predictions and the fit use comes from here, so a reader can
# check the experiment without running it.
# --------------------------------------------------------------------------

def weight_elements(cfg) -> int:
    """`W = 3 F H`: up `[H, 2F]` plus down `[F, H]`, one expert, in elements."""
    return 3 * cfg.intermediate_size * cfg.hidden_size


#: Where the weight-stream slope lives, as `(module, ms-per-stream,
#: streams-per-tile)`. PROBED BY NAME, NEVER FORKED: the whole point of the
#: statistic is that one definition of it is used everywhere, and this file has
#: already been the place where a second copy of an estimator went quietly out
#: of step with the first.
#:
#: IT LANDED ON 2026-09-10 AND THIS TUPLE DID NOT FOLLOW IT. The pair shipped in
#: `moe.bench.weights`, and this tuple named `moe.bench.weight_stream` and
#: `moe.bench.ai_model`, neither of which exports it. So nothing resolved, the
#: local provisional copy stayed in service, two estimators of one statistic ran
#: side by side, and the guard test written to catch exactly that passed: it
#: only fires on a module this tuple names. Both copies happened to agree to
#: every digit (0.6443482339382172 ms on mixtral bf16 at 4374.2997 GB/s), which
#: is what a silent fork looks like on the day it lands and not a defence.
#:
#: THE INTERFACE IS THE SHIPPED ONE, not the one this note used to register.
#: The registration said `weight_stream_ms(cfg, dtype_bytes, bandwidth_gbps)`
#: and `weight_streams_per_tile(slope_ms_per_tile, stream_ms) -> float`; what
#: `moe.bench.weights` ships, and what every call site here now passes, is:
#:
#:     weight_stream_ms(model, dtype: str, bandwidth_gbps) -> float
#:         milliseconds to stream the ROUTED expert weight set ONCE at that
#:         rate, i.e. `num_experts x 3 F H x bytes(dtype) / bandwidth`. The
#:         dtype is a NAME and not a byte count, because two dtypes share a
#:         width and a weight set is not a guess.
#:     weight_streams_per_tile(slope_ms, model, dtype, bandwidth_gbps, *,
#:                             bandwidth_source="") -> WeightStreamSlope
#:         that ladder's `B` divided by the stream, carried with the rate it
#:         was divided by. `.streams` is `w`; the object exists so the number
#:         and its denominator cannot drift apart in the prose.
#:
#: The pair is resolved once, at import, and the source string travels with
#: every number computed from it, onto the page and into report.json.
WEIGHT_STREAM_CANDIDATES = (
    ("moe.bench.weights", "weight_stream_ms", "weight_streams_per_tile"),
)


def _load_weight_stream():
    """The shared weight-stream estimator, and where it came from.

    THERE IS NO LOCAL FALLBACK, and its absence is the fix. A fallback is what
    let this file run its own copy for a day after the shared one landed, so a
    module that cannot be imported is now an ImportError at import time rather
    than a second estimator with a label on it.
    """
    tried = []
    for module_name, ms_name, per_tile_name in WEIGHT_STREAM_CANDIDATES:
        tried.append(module_name)
        try:
            module = importlib.import_module(module_name)
        except Exception:                                      # noqa: BLE001
            continue
        ms = getattr(module, ms_name, None)
        per_tile = getattr(module, per_tile_name, None)
        if callable(ms) and callable(per_tile):
            return ms, per_tile, f"{module_name}.{ms_name}"
    raise ImportError(
        "no weight-stream estimator: none of " + ", ".join(tried)
        + " exports weight_stream_ms and weight_streams_per_tile. This file "
          "deliberately keeps no copy of that arithmetic to fall back to, "
          "because keeping one is how it came to run its own for a day after "
          "the shared estimator landed")


WEIGHT_STREAM_MS, WEIGHT_STREAMS_PER_TILE, WEIGHT_STREAM_SOURCE = (
    _load_weight_stream())


def act_once_elements(cfg) -> int:
    """`2H + 3F` per row: x_perm, h_up, h_act, y_perm, each touched once.

    The same count `block_m_crossing_sweep.activation_bytes_per_row` uses, in
    elements rather than bytes, and checked against it in the tests: a
    divergence here would put this script's decomposition and that file's
    `alpha-corrected` column on two different definitions of the same traffic.
    """
    return 2 * cfg.hidden_size + 3 * cfg.intermediate_size


def act_reread_elements(cfg, block_n: int) -> float:
    """`W/BN - H - F` per row: the activation bytes an EXTRA N-tile re-reads.

    The up GEMM's A operand `[r, H]` is re-read once per N-tile of `2F`, the
    down GEMM's `[r, F]` once per N-tile of `H`, and

        H ceil(2F/BN) + F ceil(H/BN) = 3 F H / BN = W / BN

    when BN divides both, which it does for every power of two on this grid.
    Subtracting the first read of each leaves what the SECOND and later N-tiles
    cost, which is the term alpha_a multiplies.
    """
    if block_n <= 0:
        raise ValueError(f"BLOCK_SIZE_N={block_n} must be positive")
    up = cfg.hidden_size * math.ceil(2 * cfg.intermediate_size / block_n)
    down = cfg.intermediate_size * math.ceil(cfg.hidden_size / block_n)
    return float(up + down - cfg.hidden_size - cfg.intermediate_size)


def effective_k(cfg) -> float:
    """The `K` in `BM/K`, derived instead of assumed.

    (LIN)'s third term is the traffic an extra M-tile carries that is neither a
    weight re-read nor an activation re-read: the operands read once and the
    outputs written once, `BM (2H + 3F)` elements against `W` weight elements.
    Writing that as `BM/K` gives

        K = W / (2H + 3F) = 3 F H / (2H + 3F)

    which is 3440.6 on mixtral, not the 4096 that `moe/bench/ai_model.py` uses
    by taking the up GEMM's reduction dimension. The difference is 0.006 in
    alpha at BM=128 -- below this study's noise floor and stated anyway, because
    an unnamed 19% in a constant is how a term stops being checkable.
    """
    return weight_elements(cfg) / act_once_elements(cfg)


def d0_term(cfg, block_m: int) -> float:
    """`BM (2H + 3F) / W`, the read-once traffic of one M-tile in weight units."""
    return block_m * act_once_elements(cfg) / weight_elements(cfg)


def g1_term(cfg, block_m: int, block_n: int) -> float:
    """`BM (W/BN - H - F) / W`, the coefficient alpha_a multiplies.

    Equal to `BM/BN` up to `BM (H+F)/W`, which is 0.013 at BM=128 on mixtral --
    0.7% of the leading term. Carried exactly rather than approximated, since
    carrying it costs one multiplication and dropping it puts a known bias in
    the one slope this experiment exists to measure.
    """
    return block_m * act_reread_elements(cfg, block_n) / weight_elements(cfg)


def phi(cfg, block_m: int, block_n: int, alpha_a: float) -> float:
    """`d0 + alpha_a (s - e)`: one M-tile's non-weight traffic, in weight reads."""
    return d0_term(cfg, block_m) + alpha_a * g1_term(cfg, block_m, block_n)


def alpha_fitted_exact(cfg, block_m: int, block_n: int, *, alpha_b: float,
                       alpha_a: float, delta: float = 0.0) -> float:
    """(EXA): `(alpha_b + phi) / (1 + phi + delta)`, what a ladder fit returns.

    `delta` is the fused layer's fixed cost -- router, align, launch -- in units
    of one full weight read. It is inside the fit's denominator because the
    branch is fitted on RAW times, which is deliberate upstream: subtracting an
    extrapolated fixed cost hands its error straight to alpha. Its effect here
    is to push every alpha DOWN by a known sign, and it is fitted rather than
    assumed because it is common to every cell in an arm.
    """
    p = phi(cfg, block_m, block_n, alpha_a)
    return (alpha_b + p) / (1.0 + p + delta)


def alpha_fitted_linear(cfg, block_m: int, block_n: int, *, alpha_b: float,
                        alpha_a: float) -> float:
    """(LIN): `alpha_b + alpha_a (BM/BN) + BM/K`, the study's written form.

    Kept beside (EXA) and never quietly replaced by it. Every published alpha_b
    in this study was decomposed through (LIN), so a run that reported only
    (EXA) would be answering a different question from the one the study asked.
    """
    return (alpha_b + alpha_a * block_m / block_n
            + block_m / effective_k(cfg))


#: THE MEASURED ANCHOR for `B/C`, and the reason the predictions below are not
#: drawn from a calibrated ridge alone.
#:
#: `B/C = b alpha_fitted (1 + phi) rho / (2 BM)` with `rho` the kernel's OWN
#: achieved FLOP-per-byte, which is NOT the card's calibrated ridge: the
#: published references reach 38-64% of peak compute while the memory side
#: reaches a higher fraction of peak bandwidth, so a calibrated-ridge B/C runs
#: about 15% high and predicts, wrongly, that the BLOCK_M=256 reference at
#: BLOCK_N=64 is memory bound. It is not: it is the qualified compute reference
#: in 22 of the 24 published BN=64 arms on both cards.
#:
#: So `rho` is taken from the corpus instead, at ONE anchor cell, and everything
#: else is scaled by a RATIO in which rho cancels:
#:
#:     B/C(BM, BN) = ANCHOR x (128/BM) x (alpha_b + phi(BM,BN))
#:                                     / (alpha_b + phi(128, 64))
#:
#: 0.991 is the median `B/C` over the 22 published BLOCK_M=128 ladders that have
#: a compute reference surviving a level check (range 0.795-1.175, sd 0.078;
#: scripts/bm128_depth.py --audit is the evidence). Checked against three
#: independent facts it was not fitted to: it puts BM=64 at 1.73 and BM=32 at
#: 3.20 (both memory bound at every tread, which is what every published arm
#: measures: 16 and 33 memory treads) and BM=256 at 0.66 (compute bound
#: throughout, which is what qualified it as the reference).
#:
#: ASSUMPTION A1, named because the run tests it: rho does not move with BN.
#: It must move somewhat -- the N-tile count per M-tile changes fourfold across
#: this grid and occupancy with it -- and the report measures exactly that, as
#: the reference's implied TFLOP/s per arm, which V3 gates at 2x.
ANCHOR_RATIO = 0.991
ANCHOR_BLOCK_M = 128
ANCHOR_BLOCK_N = 64
ANCHOR_SPREAD = 0.078


def anchored_ratio(cfg, block_m: int, block_n: int, *, alpha_b: float,
                   alpha_a: float) -> float:
    """`B/C` scaled from the measured anchor, with the achieved rho cancelled."""
    num = alpha_b + phi(cfg, block_m, block_n, alpha_a)
    den = alpha_b + phi(cfg, ANCHOR_BLOCK_M, ANCHOR_BLOCK_N, alpha_a)
    return ANCHOR_RATIO * (ANCHOR_BLOCK_M / block_m) * num / den


def achieved_rho(cfg, b: int, *, alpha_b: float, alpha_a: float) -> float:
    """The kernel's OWN FLOP-per-byte, inverted out of the measured anchor.

    `B/C = b (alpha_b + phi) rho / (2 BM)`, so one measured B/C fixes rho. This
    is what `--self-test` plants with, and planting with the CALIBRATED ridge
    instead is not a detail: at 160.3 Op/B the BLOCK_M=256 reference comes out
    memory bound and every planted arm is refused, which is a self test of the
    refusal path and of nothing else. The gap between the two numbers -- about
    102 against 160 -- is the same 35% the corpus shows between the achieved
    compute fraction and the achieved bandwidth fraction, and it is the reason
    every prediction in this file is anchored rather than calibrated.
    """
    return (ANCHOR_RATIO * 2.0 * ANCHOR_BLOCK_M
            / (b * (alpha_b + phi(cfg, ANCHOR_BLOCK_M, ANCHOR_BLOCK_N,
                                  alpha_a))))


def branch_ratio(cfg, block_m: int, block_n: int, *, alpha_b: float,
                 alpha_a: float, ridge: float, b: int) -> float:
    """`B/C = ridge b (alpha_b + phi) / (2 BM)`, and E, F, H cancel exactly.

    The quantity that decides whether a cell yields an alpha at all. Inside
    `1 +/- PARALLEL_BRANCH_TOLERANCE` the fit DISCARDS the memory branch,
    because two branches within 15% of each other are one line. Predictable per
    cell before the run, which is why the report can register which cells it
    expects to lose and to what.
    """
    return (ridge * b * (alpha_b + phi(cfg, block_m, block_n, alpha_a))
            / (2.0 * block_m))


def memory_treads(cfg, block_m: int, block_n: int, *, alpha_b: float,
                  alpha_a: float, ratio: float, treads: int) -> int:
    """How many of the first `treads` stand above the compute branch.

    `t(n) = max(L(1 + a(n-1)), C n)` with `a` the fitted alpha and
    `C = L a / ratio`. Counted by evaluating the two lines rather than by a
    closed form, because the closed form has to handle `a > 1` and `ratio > 1`
    and the four sign cases between them, and a loop over eight treads does not.

    The one fact worth stating out loud, because it decides the whole grid:
    tread 1 is memory bound if and only if `ratio > a`, NOT whenever
    `ratio > 1`. A ladder can have `B/C` below 1 and still be memory bound
    everywhere it is swept.
    """
    a = alpha_fitted_exact(cfg, block_m, block_n, alpha_b=alpha_b,
                           alpha_a=alpha_a)
    if a <= 0 or ratio <= 0:
        return 0
    c = a / ratio                      # in units of L
    n = 0
    for i in range(1, treads + 1):
        if 1.0 + a * (i - 1) > c * i * (1.0 + SWEEP.MEMORY_BRANCH_MARGIN):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------
# The fit. Two forms, both linear in their parameters once written in the right
# coordinates, both solved by the same three lines of linear algebra.
# --------------------------------------------------------------------------

def ols(rows: list[list[float]], ys: list[float]) -> list[float] | None:
    """Least squares by normal equations, or None when the design is singular.

    Small and explicit rather than numpy, because the design is at most 3x3, the
    singular case has to be a REFUSAL rather than a pseudo-inverse that returns
    a confident number for an unidentified parameter, and a reader checking this
    experiment should not have to trust a library call to see what was fitted.
    """
    p = len(rows[0]) if rows else 0
    if p == 0 or len(rows) < p:
        return None
    a = [[sum(r[i] * r[j] for r in rows) for j in range(p)] + [
        sum(r[i] * y for r, y in zip(rows, ys, strict=True))] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(p):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, p + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][p] / a[i][i] for i in range(p)]


def pearson(xs, ys) -> float | None:
    """Correlation, or None when either side is constant. Used by the structure
    test, where a constant column means the candidate term is not being probed
    by this design at all -- which is not the same as it being absent."""
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class AlphaCell:
    """One (BLOCK_N, BLOCK_M) ladder's fitted alpha, and why it is or is not one."""

    block_n: int
    block_m: int
    #: `LadderFit.alpha`, the study's own estimator. None when the fit refused.
    alpha: float | None
    #: `LadderFit.alpha_upper` = B/(L - D), the SAME slope over the memory
    #: branch's level with the fused layer's fixed cost taken out, `D` measured
    #: on this arm's compute reference rather than assumed. THE PRIMARY
    #: OBSERVABLE, and the reason is arithmetic: `alpha = (alpha_b + phi)
    #: / (1 + phi + delta)` carries `delta = D/L_w` as a third unknown whose
    #: design column is `-alpha`, and alpha varies by only 10% across this
    #: whole grid, so that column is nearly the intercept's. Fitted anyway it
    #: turns 0.4% timing noise into alpha_b = -0.05 from data planted at 0.92.
    #: `alpha_upper` removes delta exactly -- B/(L-D) = (alpha_b + phi)/(1 +
    #: phi) -- and leaves two parameters and a well-conditioned design.
    alpha_upper: float | None
    alpha_corrected: float | None
    memory_points: int
    treads: int
    spread: float | None
    basis: str
    #: Empty when this cell carries an alpha; otherwise the reason it does not,
    #: in the words of whichever refusal produced it.
    blank: str = ""
    #: THE THREE MILLISECOND FIGURES EVERY ALPHA ON THIS PAGE IS BUILT FROM,
    #: carried per cell from 2026-09-10 so that the estimator's arithmetic is
    #: visible beside its output instead of being reconstructible only by
    #: re-fitting. `B` is the memory branch's slope, `A` its intercept at zero
    #: tiles, `D` the arm's compute reference's fitted fixed cost.
    slope_memory_ms: float | None = None
    intercept_ms: float | None = None
    fixed_cost_ms: float | None = None
    #: `w = B / T_w`: the slope in units of ONE COMPLETE STREAM of the layer's
    #: expert weights at the card's own measured rate. No fitted level, no
    #: intercept, no delta, no D, which is the whole reason it is here beside
    #: `alpha`. See `WEIGHT_STREAM_CANDIDATES` for where the estimator lives.
    weight_streams: float | None = None
    #: THE A-VERSUS-D LABEL. `alpha_upper = B/(A + B - D)` exceeds 1 exactly
    #: when `D > A`, which is arithmetic and not physics: the reference's
    #: extrapolated fixed cost is larger than the ladder's own extrapolated
    #: intercept, so the corrected level goes below `B`. In the committed
    #: 2026-09-10 arm that held in four of six cells and in NO others, and those
    #: four are exactly the four whose `alpha_upper` came back above 1. A cell
    #: carrying True is a cell whose `alpha_upper` may not be read as a miss
    #: fraction, whatever its value.
    fixed_cost_exceeds_intercept: bool | None = None

    @property
    def unphysical_by_arithmetic(self) -> bool:
        """`D > A`, i.e. this cell's `alpha_upper` is above 1 by construction."""
        return self.fixed_cost_exceeds_intercept is True

    @property
    def ad_label(self) -> str:
        """The A-versus-D label as it is printed: `D>A`, `D<A` or blank."""
        if self.fixed_cost_exceeds_intercept is None:
            return ""
        return "D>A" if self.fixed_cost_exceeds_intercept else "D<A"

    @property
    def usable(self) -> bool:
        return (self.alpha is not None and self.alpha_upper is not None
                and not self.blank)

    def observable(self, form: str) -> float:
        """The quantity a given form is fitted on."""
        return self.alpha if form in ("LIN", "EXA3") else self.alpha_upper


@dataclass(frozen=True)
class Decomposition:
    """alpha_b and alpha_a from a set of cells, in one of the two forms.

    THE COORDINATES, because they are the whole trick.

      (EXA)  alpha = (alpha_b + phi)/(1 + phi + delta), phi = d0 + alpha_a g1,
             which rearranges to a LINEAR model with no approximation:

                 alpha - d0 (1 - alpha)
                     = alpha_b (1) + alpha_a (g1 (1 - alpha)) + delta (-alpha)

             so the response and both regressors are built from the measured
             alpha and known geometry, and OLS returns all three parameters.
             The regressors carry the measured alpha, which makes this an
             errors-in-variables fit; the bootstrap resamples the underlying
             repeats and so propagates that correctly, where a textbook standard
             error would not.

      (LIN)  alpha - BM/K = alpha_b (1) + alpha_a (BM/BN), the form the study
             wrote and the form every published alpha_b was decomposed through.

    `delta` is only identifiable when alpha VARIES across the cells, since its
    column is `-alpha`; with three BN values at one BM it varies by design, and
    `sd_delta` says whether it actually came out.
    """

    form: str
    block_m: int | None
    alpha_b: float | None
    alpha_a: float | None
    delta: float | None
    n_cells: int
    n_params: int
    residuals: tuple[float, ...]
    #: One per cell, aligned with `residuals`: the label of the cell it belongs
    #: to, so a structured residual can be attributed rather than described.
    labels: tuple[str, ...]
    xs: tuple[float, ...]
    note: str

    @property
    def dof(self) -> int:
        return self.n_cells - self.n_params

    @property
    def rms(self) -> float | None:
        """Residual RMS in ALPHA units. The response is `alpha - d0(1-alpha)`,
        whose derivative in alpha is `1 + d0`, so a residual is divided by that
        to be read as a discrepancy in the quantity the study publishes."""
        if not self.residuals:
            return None
        return math.sqrt(statistics.fmean(r * r for r in self.residuals))


def _design(cells, cfg, form: str, alphas: dict[tuple[int, int], float]):
    """Response, design rows and labels for one set of cells in one form."""
    ys: list[float] = []
    rows: list[list[float]] = []
    labels: list[str] = []
    xs: list[float] = []
    for c in cells:
        u = alphas[(c.block_n, c.block_m)]
        d0 = d0_term(cfg, c.block_m)
        g1 = g1_term(cfg, c.block_m, c.block_n)
        if form == "EXA":
            # u is alpha_upper, so delta is already out and two parameters are
            # all there is: u = (alpha_b + phi)/(1 + phi) rearranges to
            #     u - d0(1-u) = alpha_b + alpha_a g1 (1-u)
            ys.append(u - d0 * (1.0 - u))
            rows.append([1.0, g1 * (1.0 - u)])
        elif form == "EXA3":
            # u is the RAW alpha and delta is a third parameter. Kept as a
            # cross-check and never gated: see AlphaCell.alpha_upper.
            ys.append(u - d0 * (1.0 - u))
            rows.append([1.0, g1 * (1.0 - u), -u])
        else:
            ys.append(u - c.block_m / effective_k(cfg))
            rows.append([1.0, c.block_m / c.block_n])
        labels.append(f"BN={c.block_n} BM={c.block_m}")
        xs.append(c.block_m / c.block_n)
    return ys, rows, labels, xs


def decompose(cells, cfg, form: str = "EXA",
              alphas: dict[tuple[int, int], float] | None = None,
              block_m: int | None = None) -> Decomposition:
    """Fit one set of cells. `block_m=None` pools every BM in the set.

    A POOLED FIT AND A PER-BM FIT ANSWER DIFFERENT QUESTIONS and both are run.
    Pooling buys degrees of freedom for the residual test, which is what C2
    needs; it also ASSUMES the thing C3 is trying to check, that alpha_b does
    not move with BM. So C3 is scored on the per-BM fits and C2 on the pooled
    one, and when C3 fails the pooled residual is the evidence for what failed.
    """
    used = [c for c in cells if c.usable
            and (block_m is None or c.block_m == block_m)]
    alphas = alphas or {(c.block_n, c.block_m): c.observable(form)
                        for c in used}
    n_params = 3 if form == "EXA3" else 2
    if len(used) < n_params + 1:
        return Decomposition(
            form, block_m, None, None, None, len(used), n_params, (), (), (),
            f"{len(used)} usable cell(s) against {n_params} parameters: a fit "
            "that cannot leave a residual cannot test the model it fits, so "
            "nothing is reported rather than a number with no degrees of "
            "freedom behind it")
    ys, rows, labels, xs = _design(used, cfg, form, alphas)
    beta = ols(rows, ys)
    if beta is None:
        return Decomposition(
            form, block_m, None, None, None, len(used), n_params, (),
            tuple(labels), tuple(xs),
            "the design is singular: the cells do not vary in the coordinate "
            "this form needs. A BN sweep at one BN, or an EXA fit whose alphas "
            "are all equal, has nothing to separate")
    resid = tuple(y - sum(b * r for b, r in zip(beta, row, strict=True))
                  for y, row in zip(ys, rows, strict=True))
    delta = beta[2] if form == "EXA3" else None
    return Decomposition(
        form, block_m, beta[0], beta[1], delta, len(used), n_params, resid,
        tuple(labels), tuple(xs),
        f"{form} over {len(used)} cells, {len(used) - n_params} degree(s) of "
        "freedom")


@dataclass(frozen=True)
class Structure:
    """What shape the residual has, when it has one.

    Three candidate missing terms, each a column the three-term model does NOT
    contain, and each testable as a correlation with the residual:

      quadratic in BM/BN   a second-order re-read, or (LIN) being used where
                           (EXA) was needed
      per-N-tile cost      a launch or wave cost proportional to ceil(2F/BN),
                           which is what a fixed overhead PER N-TILE would look
                           like and is not in the model at all
      alpha_b(BN)          the weight miss fraction itself moving with BN,
                           because BN changes the order the L2 is walked in.
                           Its signature is a residual monotone in BN at fixed
                           BM -- which is also what the first column would give,
                           so the two are reported together and neither is
                           claimed alone.
    """

    correlations: dict[str, float | None]
    worst_name: str
    worst_value: float | None
    read: bool

    def line(self) -> str:
        if not self.read:
            return ("residual too small to have a shape; correlations reported "
                    "and not read")
        if self.worst_value is None:
            return "no candidate column varies across these cells"
        return (f"worst correlation {self.worst_value:+.3f} against "
                f"{self.worst_name}")


def structure_of(fit: Decomposition, cells, cfg, chi2: float | None
                 ) -> Structure:
    """Correlate the residual with each candidate missing term."""
    used = [c for c in cells if c.usable
            and (fit.block_m is None or c.block_m == fit.block_m)]
    if len(used) != len(fit.residuals):
        return Structure({}, "", None, False)
    cols = {
        "(BM/BN)^2": [(c.block_m / c.block_n) ** 2 for c in used],
        "N-tiles per M-tile": [
            math.ceil(2 * cfg.intermediate_size / c.block_n)
            + math.ceil(cfg.hidden_size / c.block_n) for c in used],
        "1/BN": [1.0 / c.block_n for c in used],
        "BLOCK_M": [float(c.block_m) for c in used],
    }
    corrs = {k: pearson(v, list(fit.residuals)) for k, v in cols.items()}
    named = [(k, v) for k, v in corrs.items() if v is not None]
    worst = max(named, key=lambda kv: abs(kv[1])) if named else ("", None)
    return Structure(corrs, worst[0], worst[1],
                     chi2 is not None and chi2 >= STRUCTURE_MIN_CHI2)


# --------------------------------------------------------------------------
# The ladders: one timing per tread per repeat, and what is read off them.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """One timing of one tread of one (BLOCK_N, BLOCK_M) setting. The CSV row.

    THE INSTRUMENT'S OWN COLUMNS ARE PART OF THE ROW, not of the report beside
    it. `instrument`, `warmup_ms`, `iters`, `trials`, `sm_clock_load_mhz`,
    `clock_level_ok`, `clock_drift_ok` and `l2_flush` come straight off the
    `moe.bench.timing.KernelTiming` that produced `ms_p50`. Until 2026-09-02
    none of them existed here and the report recorded a warmup and an iteration
    count from argv, which is how ladders warmed at 5 calls were compared with
    ladders warmed at 20 and nothing in any file said so. A row that carries
    `instrument` can be excluded by a later reader; a row that does not cannot.
    A row from a planted world carries the sweep's `SYNTHETIC_INSTRUMENT`, so
    "not measured" is a value in the column rather than an absence.

    The clock flags are Optional and None means "not determined" -- no NVML, a
    trial too short for the poller, no calibration to compare a level against --
    never "fine". Both must be read: the drop-only flag they replace was the one
    the audit found detecting whether the first sample caught the idle boost.

    `clock_level_side` SAYS WHICH WAY A LEVEL FAILURE WENT: `timing.LEVEL_LOW`,
    `timing.LEVEL_HIGH`, or "" for level or undetermined. LEVEL has been
    two-sided since 03df2d4 (2026-09-03), and SINCE 2026-09-09 NEITHER SIDE
    EXCLUDES: the under-load clock is set per tile by the kernel's own power
    draw under the cap, so a memory-shaped tread boosting to 1980 MHz and a
    BLOCK_M=128 tread holding 1395 are both at their own operating point.
    `clock_excluded` is the one reader of the verdicts and it excludes on
    DRIFT alone; the side is a record. A row that carries `clock_level_ok`
    False and no side was written before this column existed, when LEVEL was
    one-sided and False meant below.

    `sm_clock_start_mhz`, `sm_clock_end_mhz`, `clock_samples_mhz` and
    `power_w` are the evidence a DRIFT verdict rests on. Until 2026-09-09 six
    of the seven writers in this repo kept only the median, so the 135 drifted
    rows of the H200 gaps session say a clock moved and cannot say which way
    or how far, and no rerun can tell a settling warmup from a throttle.
    """

    block_n: int
    block_m: int
    tiles: int
    rows_per_expert: int
    tokens: int
    rep: int
    ms_p50: float
    ms_min: float
    ms_stdev: float
    iters: int
    status: str = "ok"
    detail: str = ""
    instrument: str = ""
    warmup_ms: float = 0.0
    trials: int = 0
    l2_flush: bool = False
    sm_clock_load_mhz: float | None = None
    #: The first and last under-load sample, the whole sample list (space-
    #: joined MHz) and the draw at the same NVML call. Persisted since
    #: 2026-09-09 so a DRIFT row says which way the clock went and a LEVEL
    #: side can be told apart as a hungry tile from a throttled card.
    sm_clock_start_mhz: float | None = None
    sm_clock_end_mhz: float | None = None
    clock_samples_mhz: str = ""
    power_w: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    host_bound: bool | None = None
    clock_level_side: str = ""


SAMPLE_FIELDS = list(Sample.__dataclass_fields__)


def clock_side_of(t) -> str:
    """Which side of the LEVEL band the instrument saw a cell on.

    `timing.LEVEL_LOW`, `timing.LEVEL_HIGH`, or "" for level or undetermined.
    The record's own `clock_level_side` is preferred. A record that failed
    LEVEL and carries no side (a fake built before the field existed on
    2026-09-03) has it derived from its own load and reference, the rule
    `moe.bench.driver` applies to the same records; one with neither answers
    "", which is "no side recorded" and not "level".

    SINCE 2026-09-09 THE SIDE IS A RECORD AND NOT A FILTER. `clock_excluded`
    below reads DRIFT alone; the side is written on the row and printed in the
    clock-state block. It is NOT what this file's own-clock roof is computed
    from: `RefVerdict.own_clock_tflops` rescales the fixed roof by the median
    under-load clock of the whole reference ladder, one number per arm, and no
    per-row side enters it. This sentence claimed the side was scored from
    until 2026-09-09.
    """
    from moe.bench import timing

    side = getattr(t, "clock_level_side", "") or ""
    if not side and getattr(t, "clock_level_ok", None) is False:
        side = timing.level_side(getattr(t, "sm_clock_load_mhz", None),
                                 getattr(t, "reference_clock_mhz", None)) or ""
    return side


def clock_samples_of(t) -> dict:
    """The under-load clock evidence a DRIFT verdict rests on, as row columns.

    THE FIRST AND LAST SAMPLE WERE COMPUTED AND THROWN AWAY. `time_kernel` has
    put `sm_clock_start_mhz` and `sm_clock_end_mhz` on every `KernelTiming`
    since the clock-under-load instrument landed, and six of the seven writers
    in this repo kept only the median. The 2026-09-09 H200 session therefore
    ended with 135 rows that say DRIFT and cannot say which way the clock went:
    "the governor was still settling after a workload change" had to be argued
    from where the drifted cells sat in each rep rather than from the cells.
    Persisted from here on so the next session can be read off its own rows.

    Every field is fetched with `getattr` because the instrument gained
    `clock_samples_mhz` and `power_w` after these rows first existed: a record
    without them writes the column EMPTY, which is NOT DETERMINED and never
    zero. The sample list is space-joined integers, one representation that
    serves a CSV cell and a JSON value alike.
    """
    samples = getattr(t, "clock_samples_mhz", None) or ()
    return {
        "sm_clock_start_mhz": getattr(t, "sm_clock_start_mhz", None),
        "sm_clock_end_mhz": getattr(t, "sm_clock_end_mhz", None),
        "clock_samples_mhz": " ".join(f"{c:.0f}" for c in samples),
        "power_w": getattr(t, "power_w", None),
    }


def clock_excluded(level_ok: bool | None, side: str,
                   drift_ok: bool | None) -> bool:
    """Do a tread's clock verdicts exclude it. DRIFT does; no LEVEL side does.

    THE RULE CHANGED ON 2026-09-09. Until then this file excluded a tread
    whose LEVEL failed LOW, and before 2026-09-08 one whose LEVEL failed at
    all, which took a memory-shaped tread boosted to 1980 MHz for one that had
    run cold. The 750-cell census of the H200 gaps session settled what the LOW
    side is. Under the 700 W cap the under-load clock is an OUTCOME of the
    tread, set per tile by the kernel's own power draw: BLOCK_M=128 at
    BLOCK_N=64 sat at 1380-1410 MHz in every rep and every tread, BLOCK_M=256
    at 1620-1755, memory-shaped treads at 1950-1980, against a calibration
    GEMM that itself held 1485 MHz at 691 W, near the LOW end of what dense
    work does on this card. A band around that GEMM's operating point therefore
    excludes a TILE and not a defect: it dropped 148 treads session-wide,
    every one of them the steady state of one of the two tile families this
    study is about, and it would drop the same ones on every rerun.

    DRIFT survives, because it says something else: the clock MOVED while the
    tread was timed, so the median load is a blend of two clocks and the time
    is not a time at one operating point. All 135 drifts in that session were
    the governor settling on the first tread of a rep after a workload change,
    which is an instrument problem and is fixed at the instrument.

    `level_ok` and `side` are still taken and still written on the row. The
    side is a RECORD of where the tread ran and it excludes nothing. It is not
    an input to any roof: this file's own-clock roof is per ARM, from the
    reference ladder's median clock (`RefVerdict.own_clock_tflops`), and until
    2026-09-09 this sentence said the side was what it was scored from. None
    is not determined and an exclusion has to be positively established, so
    only a False DRIFT excludes.
    """
    return drift_ok is False


def clock_state(samples: list[Sample]) -> dict:
    """How many timed treads sat where against the roof's clock, in one block.

    Counts and never a verdict: this file drops no tread for its clock (the
    fit reads `status == "ok"`), so the block is what a reader of report.json
    has to decide whether the ladders were timed at the clock the roof was.

    THE FIVE COUNTS PARTITION THE TIMED TREADS AND DO NOT OVERLAP. Until
    2026-09-09 `level`, `low`, `high` and `unknown` filtered on the LEVEL
    verdict alone, so a tread that both drifted and sat level was counted in
    `level` AND in `drift`, and the printed line called the same tread kept
    and excluded-shaped in one sentence: on this session's own bn_decomposition
    ladders it read "43 level, 161 steady HIGH, 8 DRIFT failed" over 204 timed
    treads, and 43 + 161 already was 204. The four LEVEL counts now take the
    steady treads only, `drift` takes the rest, and the five sum to `timed`.
    The drifted treads keep their own LEVEL breakdown in `drift_level`,
    `drift_low`, `drift_high` and `drift_unknown`, because "which way was the
    clock when it moved" is still a fact about the run, just not a fact about
    a kept tread.

    `drift` is the excluded state `clock_excluded` names. `low` and `high` are
    both KEPT since 2026-09-09 and counted apart because the side is the
    record of the operating point the tread ran at. `unknown` is the treads
    whose LEVEL was not determined, counted separately because a run that
    could not read its clocks and a run whose clocks were fine are not the
    same state.
    """
    from moe.bench import timing

    timed = [s for s in samples if s.status == "ok"]
    drifted = [s for s in timed if s.clock_drift_ok is False]
    steady = [s for s in timed if s.clock_drift_ok is not False]

    def sides(rows: list[Sample]) -> dict:
        return {
            "level": sum(1 for s in rows if s.clock_level_ok is True),
            "low": sum(1 for s in rows if s.clock_level_ok is False
                       and s.clock_level_side != timing.LEVEL_HIGH),
            "high": sum(1 for s in rows if s.clock_level_ok is False
                        and s.clock_level_side == timing.LEVEL_HIGH),
            "unknown": sum(1 for s in rows if s.clock_level_ok is None),
        }

    kept = sides(steady)
    moved = sides(drifted)
    return {
        "timed": len(timed),
        "level": kept["level"],
        "low": kept["low"],
        "high": kept["high"],
        "drift": len(drifted),
        "unknown": kept["unknown"],
        "drift_level": moved["level"],
        "drift_low": moved["low"],
        "drift_high": moved["high"],
        "drift_unknown": moved["unknown"],
        "excluded_shaped": sum(
            1 for s in timed
            if clock_excluded(s.clock_level_ok, s.clock_level_side,
                              s.clock_drift_ok)),
        "rule": "DRIFT excludes; BOTH LEVEL sides are kept with the side "
                "recorded, because the under-load clock is set per tile by the "
                "kernel's own power draw under the cap; the fixed roof is what "
                "the compute-bound gates score against; the four LEVEL counts "
                "are over the STEADY treads only and the five counts partition "
                "the timed treads; this fit drops no tread for its clock",
    }


def clock_state_lines(state: dict) -> list[str]:
    """The printed form of `clock_state`, saying which side each count is.

    Two lines since 2026-09-09: the first is the steady treads by LEVEL side,
    the second the drifted ones by LEVEL side. Before that the LEVEL counts
    included the drifted treads and the line named one tread as kept and as
    excluded-shaped at once.
    """
    lines = [
        f"  {state['timed']} timed treads: {state['level']} steady level, "
        f"{state['low']} steady LOW (kept, side recorded), "
        f"{state['high']} steady HIGH (kept, side recorded), "
        f"{state['drift']} DRIFT failed (excluded, and not in the three "
        f"counts before it), "
        f"{state['unknown']} steady with LEVEL not determined",
    ]
    if state["drift"]:
        lines.append(
            f"  the {state['drift']} drifted treads by side: "
            f"{state['drift_level']} level, {state['drift_low']} LOW, "
            f"{state['drift_high']} HIGH, {state['drift_unknown']} with LEVEL "
            "not determined; a tread whose clock moved is not steady at any "
            "side, so none of them is counted as kept")
    lines.append(
        "  scored against the fixed roof; this fit drops no tread for its "
        "clock, and None means NOT DETERMINED, never fine")
    return lines


def ladder_rows(cfg, block_m: int, r_max: int, max_treads: int) -> list[int]:
    """Exactly-full tile stacks only: `r = n BM`, zero padding, one per tread.

    No background grid and no step probes. This experiment reads ladders and
    nothing else, so a row that is not a tread top is GPU time spent on a number
    no gate here looks at.

    REFUSES when the model's routing cannot form `n BM` rows as an integer token
    count, rather than nudging: a nudged row is not a full tile stack and a fit
    over partly-filled treads is a fit over padding. `rows_quantum` is 1 for
    mixtral and qwen2 and 3 for deepseek-v2-lite, which is the model that died
    three seconds into an unattended run once already.
    """
    q = SWEEP.rows_quantum(cfg)
    out = []
    for n in range(1, max(0, r_max // block_m) + 1):
        if len(out) >= max_treads:
            break
        r = n * block_m
        if r % q:
            raise SystemExit(
                f"{cfg.num_experts} experts at top-k {cfg.top_k} need rows per "
                f"expert to be a multiple of {q}, and {r} (tread {n} at "
                f"BLOCK_M={block_m}) is not. This model cannot form an exactly "
                "full tile stack at this block size; choose another --model.")
        out.append(r)
    if not out:
        raise SystemExit(
            f"--r-max {r_max} is below one tile at BLOCK_M={block_m}: that "
            "ladder would have no treads and the arm no reference.")
    return out


def collapse(samples, block_n: int, block_m: int, rng=None
             ) -> tuple[list[tuple[int, float]], float | None]:
    """Per-tread median across repeats, and the median across-repeat spread.

    The median across REPEATS rather than one pass's own median: a repeat is a
    fresh call at a fresh point in the pod's thermal history, and this study's
    one non-monotone published ladder is exactly what a single pass cannot tell
    from a mechanism.

    With `rng` the repeats are resampled WITH REPLACEMENT, which is the
    bootstrap: everything downstream -- the reference, the membership decision,
    the alpha, the decomposition -- is then recomputed on that draw, so the
    interval on alpha_a carries the instability of the membership decision and
    not only the scatter of the timings.
    """
    by: dict[int, list[float]] = {}
    for s in samples:
        if (s.block_n == block_n and s.block_m == block_m
                and s.status == "ok" and s.ms_p50 > 0):
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = []
    for n, vals in sorted(by.items()):
        draw = ([rng.choice(vals) for _ in vals] if rng is not None else vals)
        points.append((n, statistics.median(draw)))
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1 and statistics.median(v) > 0]
    return points, (statistics.median(spreads) if spreads else None)


def inversions(points) -> list[tuple[int, float]]:
    """Tread boundaries where time FALLS as tiles rise, with the relative drop.

    Not a mechanism under either branch: both have positive slope. Reported per
    arm and gated in units of the arm's own across-repeat spread.
    """
    out = []
    for (_, t0), (n1, t1) in zip(points, points[1:], strict=False):
        if t1 < t0 and t0 > 0:
            out.append((n1, (t0 - t1) / t0))
    return out


# --------------------------------------------------------------------------
# The compute reference, and the three refusals that stand between a corrupt
# one and a report.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RefVerdict:
    """One BN arm's compute reference: what qualified it, or what refused it."""

    block_n: int
    block_m: int | None
    slope_per_tile: float | None
    overhead_ms: float
    implied_tflops: float | None
    ceiling_tflops: float
    fraction: float | None
    refusals: tuple[str, ...]
    note: str
    #: The sweep's own `ComputeReference`, carried so `fit_ladder` is handed the
    #: same object the study's estimator expects rather than a reconstruction.
    ref: object = None
    #: OWN when this arm qualified its own compute branch, IMPORTED when it was
    #: taken from the arms that did. The distinction is the sweep's own
    #: OBSERVED/DERIVED/IMPORTED taxonomy and it travels into report.json,
    #: because a cell resting on another arm's ruler is not the same
    #: measurement as one resting on its own.
    basis: str = "OWN"
    #: What the import assumed, in words, when it is one.
    import_note: str = ""
    #: The median under-load SM clock of the ladder this reference was fitted
    #: from, and the clock the FIXED roof was measured at. Both records, both
    #: on the row since 2026-09-09: the achieved rate is scored against the
    #: fixed roof (the GEMM and every cell ran under the same 700 W cap, so
    #: that is the fair delivered-throughput comparison) and the own-clock
    #: fraction is printed beside it as ISSUE EFFICIENCY, never as a gate
    #: input. On an IMPORTED basis this stays the BORROWER's clock while the
    #: rate is the lenders', so `own_clock_fraction` is None there and only
    #: the roof is printed; see that property.
    load_clock_mhz: float | None = None
    reference_clock_mhz: float | None = None

    @property
    def ok(self) -> bool:
        return not self.refusals and self.block_m is not None

    @property
    def own_clock_tflops(self) -> float | None:
        """The compute roof at the clock THIS ladder ran at, not at the GEMM's.

        The fixed roof assumed the calibration GEMM's issue rate. A ladder that
        held 1725 MHz had 1725/1485 of that issue rate available to it, so its
        fraction of the fixed roof is inflated by the ratio and its fraction of
        THIS number is what the kernel got out of the machine it actually ran
        on. None when either clock is missing: an unknown is not a 1.0.
        """
        if not (self.load_clock_mhz and self.reference_clock_mhz):
            return None
        return self.ceiling_tflops * self.load_clock_mhz / self.reference_clock_mhz

    @property
    def own_clock_fraction(self) -> float | None:
        """Issue efficiency: the implied rate over the roof at its own clock.

        NONE FOR AN IMPORTED BASIS, since 2026-09-09. `import_reference` builds
        the borrower's `implied_tflops` from the LENDERS' median slope, fitted
        on the lenders' ladders at the lenders' clocks, while `load_clock_mhz`
        stays the borrower's own measured operating point. The ratio of the two
        is therefore a rate measured at one clock over a roof rescaled to a
        different one, and it is not this arm's issue efficiency however it is
        labelled. It did not bite the 2026-09-09 corpus (all three arms
        qualified OWN and nothing was imported) and would have bitten the first
        rerun in which an arm borrowed a branch, which is the case
        `import_reference` exists for. `own_clock_tflops` stays, because a roof
        at a measured clock is still a fact about this arm; only the fraction
        mixes two ladders.
        """
        roof = self.own_clock_tflops
        if self.imported or roof is None or not roof or self.implied_tflops is None:
            return None
        return self.implied_tflops / roof

    @property
    def imported(self) -> bool:
        return self.basis == "IMPORTED"

    def render(self) -> list[str]:
        out = [f"  BN={self.block_n:4d}  [{self.basis}] {self.note}"]
        if self.import_note:
            out.append(f"            {self.import_note}")
        if self.fraction is not None:
            out.append(
                f"            LEVEL {self.implied_tflops:8.1f} TFLOP/s = "
                f"{self.fraction:6.1%} of {self.ceiling_tflops:.1f}   gate "
                f"[{REFERENCE_LEVEL_FLOOR:.0%}, "
                f"{REFERENCE_LEVEL_CEILING:.0%}]")
            # BOTH ROOFS ON EVERY POINT LINE. The gate reads the fixed one;
            # this line says what the kernel got out of the clock it actually
            # held, which on the H200 moves 1560-1725 MHz across this grid.
            own = self.own_clock_fraction
            if own is not None:
                out.append(
                    f"            issue efficiency {own:6.1%} of "
                    f"{self.own_clock_tflops:.1f} TFLOP/s, the roof at this "
                    f"ladder's own {self.load_clock_mhz:.0f} MHz (record, "
                    "never a gate input)")
            elif self.imported and self.own_clock_tflops is not None:
                # NO ISSUE EFFICIENCY ON A BORROWED BRANCH. The rate above was
                # fitted on the lenders' ladders at the lenders' clocks; the
                # roof beside it is rescaled to THIS arm's clock. Dividing them
                # was what this line did until 2026-09-09 and the sentence it
                # printed ("this ladder's own N MHz") described neither half.
                out.append(
                    f"            no issue efficiency: the rate above is the "
                    f"lenders' (IMPORTED slope), this arm's own "
                    f"{self.load_clock_mhz:.0f} MHz puts its roof at "
                    f"{self.own_clock_tflops:.1f} TFLOP/s, and a rate measured "
                    "at one clock over a roof at another is not an efficiency")
        for why in self.refusals:
            out.append(f"            REFUSED: {why}")
        return out


def implied_tflops(cfg, block_m: int, slope_ms_per_tile: float) -> float:
    """A reference slope read as an achieved rate.

    One M-tile per expert at `block_m` rows is `E BM` padded rows and
    `6 E BM F H` flops, so the slope names a rate directly. This is a LEVEL and
    not a shape: the A100 BLOCK_N=256 reference was proportional to its tile
    count to 0.2% while implying 3.6 TFLOP/s against that card's 262.4.
    """
    if slope_ms_per_tile <= 0:
        return math.inf
    return (SWEEP.useful_flops(cfg, cfg.num_experts * block_m)
            / (slope_ms_per_tile * 1e-3) / 1e12)


def qualify_reference(cells, block_sizes, block_n: int, *, cfg, ridge: float,
                      bandwidth_gbps: float, b: int, pinned: dict,
                      capability, ceiling_tflops: float,
                      subjects=SUBJECT_BLOCK_M,
                      load_clock_mhz: float | None = None,
                      reference_clock_mhz: float | None = None) -> RefVerdict:
    """Qualify one BN arm's compute reference, on SHAPE then on LEVEL.

    Four refusals, in the order they became necessary:

      1. THE SWEEP'S OWN, imported rather than reimplemented: proportionality
         through the origin, the tile-resource bill, the roof ceiling, the
         non-vacuity bound and the cross-ladder comparison at matched
         exactly-full rows. A private copy of that would drift from the estimator
         the study publishes, and this arm has to be judged by the same one.
      2. THE REFERENCE MUST OUTRANK EVERY SUBJECT. `compute_reference` takes the
         largest ladder that qualifies, so when BLOCK_M=256 is refused for its
         registers the next candidate is 128 -- the primary subject. A subject
         promoted to reference has no memory branch BY ASSUMPTION and reports no
         alpha, which is how the published H200 BN=256 arm lost its BLOCK_M=128
         cell under a caption that blamed tread count. Refused here, out loud.
      3. THE LEVEL IN ABSOLUTE UNITS, against the ATTACHED card's calibrated
         peak. Shape is scale free; 249.765 ms per tile has the same shape as
         5.724 and passed at 0.2%.
      4. AND, IN `cross_bn_refusal` BELOW, ACROSS BN -- the check only a sweep
         holding every BN on one card in one session can make.

    `block_sizes` IS THE SWEPT DESIGN AND NOT THE LADDERS MEASURED SO FAR.
    Refusal 1's non-vacuity bound scales the compute branch to the smallest
    size in this tuple, so passing only the reference's own size makes the
    bound 1.675 of one full weight read at the roof on the H200 and refuses
    every kernel there is. Both call sites in this file pass the subjects.

    `load_clock_mhz` and `reference_clock_mhz` are RECORDS carried onto the
    verdict. The level gate reads the fixed-roof fraction; the own-clock
    fraction is printed beside it.
    """
    ref = SWEEP.compute_reference(
        cells, block_sizes, cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
        capability=capability)
    why: list[str] = list(ref.refusals)
    if ref.block_m is None:
        return RefVerdict(block_n, None, None, 0.0, None, ceiling_tflops, None,
                          tuple(why) or (ref.note,), ref.note, ref,
                          load_clock_mhz=load_clock_mhz,
                          reference_clock_mhz=reference_clock_mhz)
    biggest = max(subjects)
    if ref.block_m <= biggest:
        why.append(
            f"the qualified reference is BLOCK_M={ref.block_m}, which is not "
            f"above the largest subject ({biggest}). A subject used as its own "
            "compute branch has no memory branch by assumption and yields no "
            "alpha, so this arm would contribute a blank cell that looks like a "
            "measurement. The usual cause is BLOCK_M="
            f"{REFERENCE_BLOCK_M} being unrunnable at BLOCK_N={block_n}")
    rate = implied_tflops(cfg, ref.block_m, ref.slope_per_tile or 0.0)
    frac = rate / ceiling_tflops if ceiling_tflops > 0 else None
    if frac is None or not (REFERENCE_LEVEL_FLOOR <= frac
                            <= REFERENCE_LEVEL_CEILING):
        why.append(
            f"the reference implies {rate:.1f} TFLOP/s, "
            + (f"{frac:.1%}" if frac is not None else "an unknown fraction")
            + f" of this card's calibrated {ceiling_tflops:.1f}, outside "
            f"[{REFERENCE_LEVEL_FLOOR:.0%}, {REFERENCE_LEVEL_CEILING:.0%}]. "
            "Below the floor the reference is a slow line and every tread of "
            "every ladder in this arm is classified against it; above the "
            "ceiling it beats the card, so the ceiling belongs to another "
            "machine or the FLOP count is wrong")
    return RefVerdict(block_n, ref.block_m, ref.slope_per_tile,
                      ref.overhead_ms, rate, ceiling_tflops, frac, tuple(why),
                      ref.note, ref, load_clock_mhz=load_clock_mhz,
                      reference_clock_mhz=reference_clock_mhz)


def reference_load_clock(samples, block_n: int) -> float | None:
    """The median under-load SM clock of one arm's REFERENCE ladder.

    The reference ladder only, never the arm's subjects: the number is the
    operating point the qualified compute branch was measured at, and the
    subjects run other tiles at other draws. Drifted rows are left out because
    their median load is a blend of two clocks, which is the one thing the
    DRIFT verdict says. Rows with no clock contribute nothing; None when there
    are none, and None is NOT DETERMINED rather than "the reference clock".

    On 2026-09-09 this reads 1725 / 1620 / 1560 MHz at BLOCK_N=32 / 64 / 128
    against a 1485 MHz calibration GEMM: the reference tile got 4-16% more
    issue rate than the fixed roof assumed, which is why the fraction printed
    beside the fixed one is a different number and not a rounding.
    """
    clocks = [s.sm_clock_load_mhz for s in samples
              if s.block_n == block_n and s.block_m == REFERENCE_BLOCK_M
              and s.status == "ok" and s.clock_drift_ok is not False
              and s.sm_clock_load_mhz]
    return statistics.median(clocks) if clocks else None


def cross_bn_normalised_spread(verdicts: list[RefVerdict]) -> float | None:
    """The cross-BN spread with every rate rescaled to the reference clock.

    V3's gate reads the RAW spread, because the fixed roof is the fair
    delivered-throughput comparison: the calibration GEMM and every cell in
    this arm ran under the same 700 W cap, and what the study compares across
    BN is what the kernel delivered. This is the same spread with each arm's
    rate divided by its own clock's share of the reference clock, i.e. what
    the arms would have spread by had they all been issued at 1485 MHz. It is
    printed BESIDE the gate number and never gates: on 2026-09-09 the raw
    spread is 1.95x, under the 2.0x bar, and the normalised one is 2.15x, over
    it, so which number gates is the whole of the arm's verdict and it is
    pre-registered rather than chosen after the fact.

    None when fewer than two arms carry both a rate and a clock, AND None when
    the arms that carry a clock are not the same arms `cross_bn_refusal` scored.
    The gate reads every verdict with a finite rate; this one additionally needs
    both clocks, so one arm with an unreadable reference clock would silently
    compute the two spreads over two different sets of arms while the page
    prints them as "the same spread with each arm rescaled to its own clock".
    On 2026-09-09 all three arms carried a clock (1725 / 1620 / 1560 MHz), so
    raw 1.95x and normalised 2.15x are the same comparison and the guard costs
    nothing there; it costs the whole line on the first run where a clock is
    missing, which is the correct price for a comparison that is not the one
    the sentence claims.
    """
    scored = [v for v in verdicts
              if v.implied_tflops and math.isfinite(v.implied_tflops)]
    usable = [v for v in scored if v.load_clock_mhz and v.reference_clock_mhz]
    if len(usable) != len(scored):
        return None
    rates = [v.implied_tflops * v.reference_clock_mhz / v.load_clock_mhz
             for v in usable]
    if len(rates) < 2 or min(rates) <= 0:
        return None
    return max(rates) / min(rates)


def cross_bn_refusal(verdicts: list[RefVerdict]) -> tuple[str, float | None]:
    """Do the arms' references agree about what machine they are on.

    The achieved rate genuinely moves with BN -- the N-tile count per M-tile
    changes fourfold across this grid, and occupancy with it -- so this bar is
    loose by design. What it catches is the failure that actually happened: one
    arm's reference 43.6x the others', from a kernel that spilled its
    accumulator to local memory and returned a time that was still perfectly
    proportional to its tile count.

    Returns the refusal and the spread, and the spread is reported whether or
    not it refuses, because a number under a bar is evidence and a blank is not.
    """
    rates = [(v.block_n, v.implied_tflops) for v in verdicts
             if v.implied_tflops and math.isfinite(v.implied_tflops)]
    if len(rates) < 2:
        return "", None
    lo = min(r for _, r in rates)
    hi = max(r for _, r in rates)
    spread = hi / lo if lo > 0 else math.inf
    if spread > REFERENCE_CROSS_BN_SPREAD:
        worst = max(rates, key=lambda kv: kv[1])[0]
        best = min(rates, key=lambda kv: kv[1])[0]
        return (f"the compute references disagree by {spread:.1f}x across BN "
                f"(fastest at BN={worst}, slowest at BN={best}), past the "
                f"{REFERENCE_CROSS_BN_SPREAD:.1f}x bar. They are ladders of the "
                "same kernel on the same card in one session, so a factor that "
                "size is a setting that did not run as pinned, not occupancy"), spread
    return "", spread


#: How many arms must have qualified a compute branch of their own before one
#: may be lent to an arm that did not. Two, so the lender's own consistency is
#: checkable; one would be an assumption with nothing to test it against.
MIN_IMPORT_SOURCES = 2

#: Fraction of the card's calibrated peak a PLANTED kernel runs at. The 22 sound
#: published references reach 38.2% to 63.7%, so 50% is the middle of the
#: measured range and is what `--self-test` plants; it is never used on measured
#: data.
PLANT_COMPUTE_FRACTION = 0.50


def import_reference(target: RefVerdict, sources: list[RefVerdict], cfg,
                     spread: float | None) -> RefVerdict:
    """Lend a compute branch to an arm that has none, or leave it refused.

    WHY THIS EXISTS RATHER THAN A REFUSAL. At small BLOCK_N every tile height on
    this hardware is memory bound -- the activation re-read is large enough that
    even BLOCK_M=256 never reaches its compute branch -- so the arm with the
    MOST leverage on alpha_a is the one least able to qualify a reference of its
    own. Refusing it costs the experiment its widest point in `BM/BN` and, with
    three BN values, costs it the fit.

    WHAT IT ASSUMES, stated because it is an assumption and not a measurement:
    that the kernel's achieved compute rate does not move with BLOCK_N. It must
    move somewhat, since the N-tile count per M-tile changes fourfold across
    this grid. THE RUN MEASURES EXACTLY THAT: the lending arms' own implied
    rates are the variation, they are printed, and V3 gates their spread. An
    import is allowed only when at least `MIN_IMPORT_SOURCES` arms qualified and
    their rates agree within that bar, so the assumption is refused by the same
    number that would have refused a corrupt reference.

    The imported slope is the MEDIAN of the lenders', not the mean and not the
    nearest: two lenders that disagree are caught by V3 rather than averaged,
    and the median is what survives a third lender being wrong.
    """
    usable = [v for v in sources if v.ok and v.slope_per_tile]
    if len(usable) < MIN_IMPORT_SOURCES:
        return target
    rates = [v.implied_tflops for v in usable if v.implied_tflops]
    if not rates or min(rates) <= 0:
        return target
    if max(rates) / min(rates) > REFERENCE_CROSS_BN_SPREAD:
        return target
    slope = statistics.median([v.slope_per_tile for v in usable])
    overhead = statistics.median([v.overhead_ms for v in usable])
    block_m = usable[0].block_m
    if any(v.block_m != block_m for v in usable):
        return target
    ref = SWEEP.ComputeReference(
        block_m, overhead, slope, 0.0,
        f"IMPORTED from BLOCK_N {sorted(v.block_n for v in usable)}: compute "
        f"branch {slope:.4f} ms per tile at BLOCK_M={block_m}, fixed cost "
        f"{overhead:.4f} ms")
    rate = implied_tflops(cfg, block_m, slope)
    return RefVerdict(
        target.block_n, block_m, slope, overhead, rate, target.ceiling_tflops,
        rate / target.ceiling_tflops if target.ceiling_tflops > 0 else None,
        (), ref.note, ref,
        # THE BORROWER'S OWN CLOCK, not the lender's: the slope is borrowed,
        # the operating point this arm ran at is measured and stays its own.
        load_clock_mhz=target.load_clock_mhz,
        reference_clock_mhz=target.reference_clock_mhz, basis="IMPORTED",
        import_note=(
            "this arm qualified NO compute branch of its own ("
            + (target.refusals[0][:90] if target.refusals else "no candidate")
            + "). Every alpha below rests on the assumption that the achieved "
              "compute rate does not move with BLOCK_N; the lenders' own rates "
              "are printed above and V3 gates their spread"))


def arm_alphas(samples, cfg, *, block_ns, subjects, ridge: float,
               bandwidth_gbps: float, dtype: str, base_pinned: dict, capability,
               ceiling_tflops: float, sm_count: int, rng=None,
               reference_clock_mhz: float | None = None
               ) -> tuple[list[AlphaCell], list[RefVerdict], dict[int, float | None]]:
    """Every arm's cells, from the raw timings, through the study's own fit.

    ONE PASS, so that the bootstrap can call it again on a resample and get the
    whole chain re-decided: the reference, its refusals, the membership margin
    and the alphas. A bootstrap that resampled only the alphas would report the
    scatter of the timings and hide the instability of the membership decision,
    which at BLOCK_M=128 is the larger of the two.
    """
    cells: list[AlphaCell] = []
    spreads: dict[int, float | None] = {}
    # THE DTYPE TRAVELS AS A NAME, and the byte width is derived from it here.
    # `moe.bench.weights` takes a name because two dtypes share a width and a
    # weight set derived from a width alone is a guess; the tile-resource bill
    # below takes the width, so the one conversion is done once, here.
    b = dtype_bytes(dtype)
    # ONE COMPLETE WEIGHT STREAM, ONCE, for the whole arm: every cell's `w` is
    # against the same denominator, so two cells' `w` differ only in their
    # slopes. Zero bandwidth is a planted world with no rate, and it gives None
    # rather than a division.
    stream_ms = (WEIGHT_STREAM_MS(cfg, dtype, bandwidth_gbps)
                 if bandwidth_gbps > 0 else None)
    # The operating point each arm's reference ladder held, read off the rows
    # once. A record on every verdict, printed beside the fixed-roof fraction,
    # and never a gate input; see `reference_load_clock`.
    load_clock_by_bn = {bn: reference_load_clock(samples, bn) for bn in block_ns}
    ladders_by_bn: dict[int, dict[int, list[tuple[int, float]]]] = {}
    cells_by_bn: dict[int, list] = {}
    verdict_by_bn: dict[int, RefVerdict] = {}

    # PASS ONE: qualify every arm's own compute branch, and NOTHING else. The
    # import in pass two needs to know which arms qualified before it can lend,
    # so the two cannot be interleaved -- and the pod run measures in the same
    # order for the same reason.
    for bn in block_ns:
        pinned = dict(base_pinned, BLOCK_SIZE_N=bn)
        ladders: dict[int, list[tuple[int, float]]] = {}
        seen_spreads: list[float] = []
        for bm in (*subjects, REFERENCE_BLOCK_M):
            pts, sp = collapse(samples, bn, bm, rng)
            if pts:
                ladders[bm] = pts
            if sp is not None:
                seen_spreads.append(sp)
        if not ladders:
            continue
        spreads[bn] = statistics.median(seen_spreads) if seen_spreads else None
        ladders_by_bn[bn] = ladders
        cells_by_bn[bn] = [
            SWEEP.make_cell(cfg, n * bm, bm, ms, sm_count=sm_count, block_n=bn)
            for bm, pts in ladders.items() for n, ms in pts]
        # THE BLOCK SIZES ARE THE DESIGN, NOT THE LADDERS ON DISK. The
        # sweep's NON-VACUITY check scales the compute branch to the SMALLEST
        # SWEPT block size (`block_m_crossing_sweep._level_checks`). Until
        # 2026-09-09 this passed `tuple(sorted(ladders))`, which collapses to
        # (256,) whenever no subject ladder has been measured yet, and the
        # references are measured first by design, so it was ALWAYS (256,).
        # Scaled to BLOCK_M=256 itself the ratio is 1.675 AT THE ROOF on the
        # H200 with mixtral against a gate of < 1.0: unpassable by a kernel at
        # 100% of peak. On 2026-09-09 it refused all three references at
        # 4.559 / 3.068 / 2.339, the run skipped every subject, and 344 s of
        # measured references produced no page. With the swept sizes the same
        # three qualify at 0.570 / 0.383 / 0.292.
        verdict_by_bn[bn] = qualify_reference(
            cells_by_bn[bn],
            tuple(sorted(set(ladders) | set(subjects) | {REFERENCE_BLOCK_M})),
            bn, cfg=cfg, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
            capability=capability, ceiling_tflops=ceiling_tflops,
            subjects=subjects,
            load_clock_mhz=load_clock_by_bn.get(bn),
            reference_clock_mhz=reference_clock_mhz)

    # PASS TWO: lend a branch to the arms that have none, then fit.
    qualified = [v for v in verdict_by_bn.values() if v.ok]
    for bn in list(verdict_by_bn):
        if not verdict_by_bn[bn].ok:
            verdict_by_bn[bn] = import_reference(
                verdict_by_bn[bn], qualified, cfg, spreads.get(bn))
    verdicts = [verdict_by_bn[bn] for bn in block_ns if bn in verdict_by_bn]

    for bn in block_ns:
        if bn not in verdict_by_bn:
            continue
        ladders = ladders_by_bn[bn]
        sweep_cells = cells_by_bn[bn]
        verdict = verdict_by_bn[bn]
        spread = spreads.get(bn)
        # THE MARGIN IS RAISED TO THREE TIMES THIS ARM'S OWN TIMING SPREAD, the
        # same rule `block_m_crossing_sweep.analyse` applies: the reference
        # slope carries that spread too, and a compute branch estimated 2% low
        # makes every compute-bound tread look memory bound.
        margin = max(SWEEP.MEMORY_BRANCH_MARGIN, 3.0 * (spread or 0.0))
        for bm in subjects:
            if bm not in ladders:
                continue
            pts = SWEEP.ladder_points(sweep_cells, bm)
            if not verdict.ok:
                cells.append(AlphaCell(
                    bn, bm, None, None, None, 0, len(pts), spread,
                    "no qualified compute reference in this arm",
                    blank="reference_refused"))
                continue
            fit = SWEEP.fit_ladder(pts, bm, verdict.ref, margin)
            act = SWEEP.activation_slope_ms(cfg, bm, bandwidth_gbps)
            corrected = None
            if fit.slope_memory is not None and fit.load_ms:
                corrected = (fit.slope_memory - act) / fit.load_ms
            blank = ""
            if fit.alpha is None:
                blank = "no_memory_branch"
            elif fit.memory_points < MIN_MEMORY_TREADS:
                blank = "too_few_memory_treads"
            elif fit.alpha_upper is None:
                # `L - D <= 0`: the reference's fixed cost swallows the memory
                # branch's whole level, so this cell cannot be corrected for it
                # and is refused rather than fitted with the raw alpha in a
                # design that assumes the correction was made.
                blank = "fixed_cost_exceeds_branch_level"
            basis = fit.basis
            if verdict.imported:
                basis = "IMPORTED branch; " + basis
            # THE SLOPE IN WEIGHT-STREAM UNITS, AND THE A-VERSUS-D LABEL, both
            # recorded whether or not the cell survives its blanks. `w` needs
            # no fitted level and `D > A` is the arithmetic that put four of
            # this arm's six `alpha_upper` values above 1, so both are worth
            # having on a cell the alpha fit refused.
            streams = None
            if (fit.slope_memory is not None and stream_ms
                    and math.isfinite(fit.slope_memory)):
                # The shared estimator returns a `WeightStreamSlope` carrying
                # the rate it divided by; the cell keeps the scalar, and the
                # rate is on the page and in report.json beside it as
                # `weight_stream_ms`. A NON-FINITE slope is not offered to it:
                # a degenerate branch fit produces one and the estimator
                # refuses it by design, and a refusal here would take down an
                # arm over a cell that measured nothing.
                streams = WEIGHT_STREAMS_PER_TILE(
                    fit.slope_memory, cfg, dtype, bandwidth_gbps,
                    bandwidth_source="the run's own calibrated rate").streams
            over = (None if fit.intercept is None
                    else fit.overhead_ms > fit.intercept)
            cells.append(AlphaCell(
                bn, bm, fit.alpha if not blank else None,
                fit.alpha_upper if not blank else None,
                corrected if not blank else None, fit.memory_points, len(pts),
                spread, basis, blank=blank,
                slope_memory_ms=fit.slope_memory, intercept_ms=fit.intercept,
                fixed_cost_ms=fit.overhead_ms, weight_streams=streams,
                fixed_cost_exceeds_intercept=over))
    return cells, verdicts, spreads


@dataclass
class Bootstrap:
    """Spreads on everything the gates read, from resampled repeats.

    `survival` is not a diagnostic afterthought. A cell whose alpha exists in
    the point estimate but vanishes in half the draws did not have a stable
    membership decision, and its sigma computed over the draws where it
    survived is a spread conditioned on surviving -- which is narrower than the
    truth and in the direction that makes C2 pass. It is reported per cell and
    gated.
    """

    draws: int
    per_cell_sd: dict[tuple[int, int], float]
    survival: dict[tuple[int, int], float]
    alpha_a_sd: float | None
    alpha_b_sd: float | None
    delta_sd: float | None
    alpha_b_by_bm_sd: dict[int, float]
    note: str


def run_bootstrap(samples, cfg, keys, *, draws: int, seed: int, form: str,
                  **kw) -> Bootstrap:
    """Resample repeats, rebuild every arm, refit, and report the spreads."""
    reps = {s.rep for s in samples if s.status == "ok"}
    if len(reps) < 2 or draws <= 0:
        return Bootstrap(0, {}, {}, None, None, None, {},
                         f"{len(reps)} repeat(s) and {draws} draw(s): a spread "
                         "needs at least two repeats to resample and at least "
                         "one draw to report. Every interval below is absent "
                         "rather than zero, and every gate that needs one reads "
                         "UNKNOWN")
    per_cell: dict[tuple[int, int], list[float]] = {k: [] for k in keys}
    a_vals: list[float] = []
    b_vals: list[float] = []
    d_vals: list[float] = []
    by_bm: dict[int, list[float]] = {}
    rng = random.Random(seed)
    for _ in range(draws):
        cells, _, _ = arm_alphas(samples, cfg, rng=rng, **kw)
        for c in cells:
            if c.usable and (c.block_n, c.block_m) in per_cell:
                per_cell[(c.block_n, c.block_m)].append(c.observable(form))
        fit = decompose(cells, cfg, form)
        if fit.alpha_a is not None:
            a_vals.append(fit.alpha_a)
            b_vals.append(fit.alpha_b)
            if fit.delta is not None:
                d_vals.append(fit.delta)
        for bm in sorted({c.block_m for c in cells}):
            per_bm = decompose(cells, cfg, form, block_m=bm)
            if per_bm.alpha_b is not None:
                by_bm.setdefault(bm, []).append(per_bm.alpha_b)

    def sd(v):
        return statistics.pstdev(v) if len(v) > 1 else None

    return Bootstrap(
        draws,
        {k: statistics.pstdev(v) for k, v in per_cell.items() if len(v) > 1},
        {k: len(v) / draws for k, v in per_cell.items()},
        sd(a_vals), sd(b_vals), sd(d_vals),
        {bm: statistics.pstdev(v) for bm, v in by_bm.items() if len(v) > 1},
        f"{draws} draws, repeats resampled with replacement, whole chain "
        "re-decided per draw (reference, refusals, membership, alpha)")


def chi_square(fit: Decomposition, cells, boot: Bootstrap, cfg
               ) -> tuple[float | None, str]:
    """`sum (r/sigma)^2 / dof`, the residual against the noise it was fitted through.

    THE GATE THIS SCRIPT EXISTS FOR, so its refusals matter as much as its
    number. Returns None -- never a number -- when there is no sigma to divide
    by, when a cell has no bootstrap spread, or when the fit has no degrees of
    freedom. A residual divided by an assumed noise floor would answer this
    question with the floor.

    The residual is in the response's units; the response is
    `alpha - d0 (1 - alpha)`, so a discrepancy of `x` in alpha appears as
    `x (1 + d0)` here and sigma is scaled by the same factor rather than the
    residual being scaled down -- identical arithmetic, but it keeps the printed
    sigma comparable with a published alpha spread.
    """
    if fit.alpha_a is None or fit.dof <= 0:
        return None, "no fit with degrees of freedom to test"
    if not boot.per_cell_sd:
        return None, ("no bootstrap spread: " + boot.note)
    used = [c for c in cells if c.usable
            and (fit.block_m is None or c.block_m == fit.block_m)]
    total = 0.0
    for c, r in zip(used, fit.residuals, strict=True):
        sigma = boot.per_cell_sd.get((c.block_n, c.block_m))
        if not sigma or sigma <= 0:
            return None, (f"BN={c.block_n} BM={c.block_m} has no bootstrap "
                          "spread, so its residual cannot be weighed against "
                          "anything")
        total += (r / (sigma * (1.0 + d0_term(cfg, c.block_m)))) ** 2
    return total / fit.dof, (f"{fit.dof} degree(s) of freedom, sigma per cell "
                             "from this run's own bootstrap")


# --------------------------------------------------------------------------
# Gates. Every one is a number against a threshold, and every VALIDITY gate
# says what a FAIL invalidates.
# --------------------------------------------------------------------------

VALIDITY, CLAIM = "VALIDITY", "CLAIM"


@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` prints UNKNOWN and never PASS: a check that could not run is
    not a check that passed, and this study has already published one report
    whose blanks were a property of its reference rather than of its data.
    """

    kind: str
    name: str
    prediction: str
    rule: str
    passed: bool | None
    observed: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The one-token name this gate answers to on its RESULT line.

        The first word of `name` -- V0, C2, S4 -- because
        `moe.bench.exit_codes.result_line` requires a name with no whitespace
        and the driver greps for these words. Taking it from `name` rather than
        from a second table means the printed heading and the machine-readable
        line can never disagree about which gate they are.
        """
        return self.name.split()[0]

    def scored(self) -> tuple[str, str, str]:
        """`(kind, token, verdict)` in `moe.bench.exit_codes`'s vocabulary."""
        return (self.kind, self.token,
                {True: exit_codes.PASS, False: exit_codes.FAIL,
                 None: exit_codes.UNKNOWN}[self.passed])

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        `RESULT: <KIND> <NAME> <VERDICT> <detail>` at column zero, rendered and
        read back by `moe.bench.exit_codes`. Nothing else this file prints has
        that shape. The predecessor of this line was free text, and the session
        driver's summary grep for `floor|sigma` matched a REFUSED log 18 times
        and printed a pre-registered `C1 ... [PASS]` as measured output.
        """
        detail = f"{self.prediction} | saw {self.observed} | gate {self.rule}"
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        out = [self.result_line(),
               f"[{tag}] {self.kind:8s} {self.name}  {self.prediction}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is not True and self.invalidates:
            out.append(f"         a non-PASS here invalidates: {self.invalidates}")
        out += [f"         {line}" for line in self.lines]
        return out


def render_gates(gates: list[Gate]) -> list[str]:
    """Every gate, each with exactly one RESULT line, then the tally.

    The tally is prose and deliberately does not look like a result: a summary
    line that could be parsed as a verdict is how a count of gates becomes a
    gate.
    """
    out: list[str] = []
    for g in gates:
        out += g.render()
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    if not gates:
        # `classify([])` raises rather than returning DONE, for the reason the
        # exit-code module gives: a check that examined nothing reports zero
        # failures. Said in words here so a report with no gates does not end in
        # a traceback.
        return out + ["", "NO GATES WERE SCORED. Nothing on this page is a "
                          "verdict."]
    implied = exit_codes.classify(g.scored() for g in gates)
    return out + ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN",
                  f"the gates imply {exit_codes.describe(implied)}"]


def gate_non_vacuity(counts: dict[str, int]) -> Gate:
    """A check that examined nothing also reports zero failures.

    Every gate below can pass by having no data. This one asserts the data
    existed and names the counts, so a reader sees WHICH work happened rather
    than trusting that some did.
    """
    empty = sorted(k for k, v in counts.items() if v <= 0)
    return Gate(VALIDITY, "V0 non-vacuity", "this report examined real work",
                "every counted quantity is above zero", not empty,
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "every gate in this report: a check with no input reports no "
                "failures",
                [f"nothing was counted for: {', '.join(empty)}"] if empty else [])


def gate_override(compiles: dict, executed: dict) -> Gate:
    """Did `override_config` actually change the kernel at every setting.

    If it silently failed, every (BN, BM) ran ONE kernel, alpha is identical
    across BN by construction, the fitted alpha_a is exactly zero and the
    residual is exactly noise -- a tidy, false PASS on the two gates this
    script exists for. A setting that changed the tile constants MUST have
    compiled a new Triton specialisation, so counting the artefacts that appear
    while it runs is a direct assay.

    THE CACHE KEY HAS TO CARRY BN. `block_m_crossing_sweep.arm_triton_cache`
    keys its per-setting directory on BLOCK_M alone, which is right for a sweep
    that varies only BLOCK_M and wrong here: two BN arms would share one
    directory, the second would find it warm, compile nothing, and be scored as
    a broken override. `arm_cache` below keys on both.
    """
    ran = [k for k, v in executed.items() if v > 0]
    resumed = [k for k in executed if k not in ran]
    missing = [k for k in ran if compiles.get(k, 0) <= 0]
    counts = ", ".join(f"n{bn}/bm{bm}:{compiles.get((bn, bm), 0)}"
                       for bn, bm in sorted(executed))
    invalid = ("every number in this report: one kernel compared with itself "
               "gives alpha_a = 0 and a residual of pure noise, which is a "
               "PASS on C1's alternative and on C2")
    if missing:
        return Gate(VALIDITY, "V1 override took effect",
                    "every (BN, BM) setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran cells",
                    False, counts, invalid,
                    [f"settings that ran cells and compiled nothing: {missing}",
                     "Either override_config did not take effect or "
                     "TRITON_CACHE_DIR was warm. Both are fatal in the same way."])
    if resumed:
        return Gate(VALIDITY, "V1 override took effect",
                    "every (BN, BM) setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran cells",
                    None, counts, invalid,
                    [f"{len(resumed)} setting(s) ran no cells this session: "
                     "every timing was already in cells.csv. The assay belongs "
                     "to the session that measured them and cannot be "
                     "inherited."])
    return Gate(VALIDITY, "V1 override took effect",
                "every (BN, BM) setting compiled its own kernel",
                ">= 1 fresh Triton artefact per setting that ran cells",
                True, counts, invalid)


def gate_reference_level(verdicts: list[RefVerdict]) -> Gate:
    """Every arm's compute reference runs at a rate its card could produce."""
    if not verdicts:
        return Gate(VALIDITY, "V2 reference level",
                    "each arm's compute reference runs at a plausible rate",
                    f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, "
                    f"{REFERENCE_LEVEL_CEILING:.0%}] of the attached card's "
                    "calibrated peak",
                    None, "no arm produced a reference to score",
                    "every membership decision, hence every alpha in this report")
    bad = [v for v in verdicts if not v.ok]
    borrowed = [v for v in verdicts if v.ok and v.imported]
    obs = "; ".join(
        f"BN={v.block_n} " + (f"{v.fraction:.1%}" if v.fraction is not None
                              else "no reference")
        + (" (IMPORTED)" if v.imported else "")
        for v in verdicts)
    # THE GATE READS THE FIXED-ROOF FRACTION ABOVE; the same rates against the
    # roof at each arm's own clock go on the line beside it, because on this
    # card the three reference ladders held 1725 / 1620 / 1560 MHz and the two
    # readings differ by 3-5 points. Printed, never scored.
    own = "; ".join(
        f"BN={v.block_n} {v.own_clock_fraction:.1%} at "
        f"{v.load_clock_mhz:.0f} MHz"
        for v in verdicts if v.own_clock_fraction is not None)
    if own:
        obs += f" | issue efficiency (record, not the gate input): {own}"
    # An IMPORTED arm contributes no issue efficiency and says so rather than
    # going silently missing from the list: see `own_clock_fraction`.
    no_own = [v.block_n for v in verdicts
              if v.imported and v.own_clock_fraction is None]
    if no_own:
        obs += ("; no issue efficiency for BN="
                + ",".join(str(n) for n in no_own)
                + " (IMPORTED slope, so the rate is the lenders' and the "
                  "clock is this arm's)")
    return Gate(VALIDITY, "V2 reference level",
                "each arm's compute reference runs at a plausible rate",
                f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, "
                f"{REFERENCE_LEVEL_CEILING:.0%}] of the attached card's "
                "calibrated peak, and the reference outranks every subject",
                not bad, obs,
                "every alpha in the arms that failed, and with them the points "
                "the decomposition is fitted through",
                [f"BN={v.block_n}: {w}" for v in bad for w in v.refusals]
                + [f"BN={v.block_n} runs on an IMPORTED compute branch: "
                   f"{v.import_note}" for v in borrowed]
                + ["`compute_reference` tests PROPORTIONALITY and never LEVEL; "
                   "a line 43.6x too steep is perfectly proportional and "
                   "qualified at 0.2% mean error.",
                   "An arm on an IMPORTED branch is not a failure of this gate "
                   "-- it passed the same level bar, on another arm's ladder -- "
                   "but it is an ASSUMPTION, and V3's cross-BN spread is what "
                   "stands behind it."])


def gate_cross_bn(refusal: str, spread: float | None,
                  normalised: float | None = None) -> Gate:
    """V3, scored on the RAW rates with the clock-normalised spread beside it.

    THE GATE INPUT IS THE RAW, FIXED-ROOF SPREAD and the bar stays at
    REFERENCE_CROSS_BN_SPREAD. The GEMM that set the roof and every cell in
    this arm ran under the same 700 W cap, so delivered throughput is the
    comparable quantity across BN. The normalised spread (every arm's rate
    rescaled to the reference clock, so it answers "how far apart would these
    be at one issue rate") is PRINTED, because on 2026-09-09 the two land on
    opposite sides of the bar (raw 1.95x, normalised 2.15x) and a page that
    showed only the passing one would be choosing its verdict after the fact.
    """
    both = (f"{spread:.2f}x raw (fixed roof, the gate input)"
            if spread is not None else
            "fewer than two arms produced a reference")
    if normalised is not None:
        both += (f"; {normalised:.2f}x normalised to each arm's own clock "
                 "(record, not the gate input)")
    return Gate(VALIDITY, "V3 references agree across BN",
                "the arms' compute references describe one machine",
                f"max/min implied TFLOP/s across BN <= "
                f"{REFERENCE_CROSS_BN_SPREAD:.1f}x, on the RAW rates against "
                "the fixed roof",
                None if spread is None else not refusal,
                both,
                "the comparison BETWEEN arms, which is the entire experiment: "
                "alpha_a is a difference across BN and cannot survive the "
                "references moving 40x between them",
                [refusal] if refusal else [])


def gate_ladders(spreads, inversion_rows, survival) -> Gate:
    """The timings themselves: quiet enough, monotone, and stably classified."""
    noisy = {bn: s for bn, s in spreads.items()
             if s is not None and s > MAX_REPLICATE_SPREAD}
    unstable = {k: v for k, v in survival.items() if v < 0.5}
    ok = not noisy and not inversion_rows and not unstable
    return Gate(VALIDITY, "V4 ladders are readable",
                "quiet, monotone ladders and stable membership",
                f"across-repeat spread <= {MAX_REPLICATE_SPREAD:.0%}, zero "
                f"inversions beyond {MONOTONE_SIGMA:.0f} sigma, every cell "
                "surviving >= 50% of bootstrap draws",
                ok,
                "spread " + ", ".join(
                    f"BN={bn}:" + ("n/a" if s is None else f"{s:.3%}")
                    for bn, s in sorted(spreads.items()))
                + f"; {len(inversion_rows)} inversion(s); "
                + f"{len(unstable)} unstable cell(s)",
                "the fitted slopes, which are what alpha is: OLS gives the last "
                "tread the most leverage, and a cell whose membership flips "
                "between draws has a sigma conditioned on surviving",
                [f"inversion at {row}" for row in inversion_rows]
                + [f"cell BN={k[0]} BM={k[1]} survived {v:.0%} of draws"
                   for k, v in sorted(unstable.items())])


def gate_identifiable(cells, primary: int) -> Gate:
    """Enough surviving BN arms to make alpha_a a fit rather than a slope."""
    by_bm: dict[int, list[int]] = {}
    for c in cells:
        if c.usable:
            by_bm.setdefault(c.block_m, []).append(c.block_n)
    n_primary = len(by_bm.get(primary, []))
    total = sum(len(v) for v in by_bm.values())
    ok = total >= MIN_BN_POINTS + 1 and len({c.block_n for c in cells
                                             if c.usable}) >= MIN_BN_POINTS
    return Gate(VALIDITY, "V5 identifiability",
                f"at least {MIN_BN_POINTS} BN values survive with an alpha",
                f">= {MIN_BN_POINTS} distinct BN, and > {MIN_BN_POINTS} cells "
                "in total so the pooled fit keeps a degree of freedom",
                ok,
                f"{total} usable cell(s) over "
                f"{len({c.block_n for c in cells if c.usable})} BN value(s); "
                + ", ".join(f"BM={bm}: BN {sorted(v)}"
                            for bm, v in sorted(by_bm.items())),
                "C1, C2, C3 and C4 alike: with two BN points and two unknowns "
                "the fit is exact, the residual is identically zero, and the "
                "model cannot be tested at all",
                [f"the primary BLOCK_M={primary} carries {n_primary} BN "
                 "value(s). The parallel-branch tolerance is the predicted "
                 "reason for a shortfall there and it is not a defect in the "
                 "run: |B/C - 1| at 128 is predicted at 0.10 on the A100 "
                 f"against a tolerance of {TOLERANCE:.2f}.",
                 "IT IS NO LONGER ONLY A PREDICTION. On 2026-09-10 the primary "
                 "carried ZERO BN values on the H200: every BLOCK_M=128 cell "
                 "at every BLOCK_N had its memory branch discarded, so this "
                 "gate's shortfall clause is the arm's normal state and not "
                 f"its bad day. BLOCK_M={SMALL_TILE_BLOCK_M} is swept because "
                 "a fit over the two heights that survived cannot separate the "
                 "activation column from the 1/BN column that beats it."])


def band_provenance_lines() -> list[str]:
    """`check_alpha_a_band`'s lines, or ONE line naming why they cannot be read.

    Every place that prints the band's provenance reads it from the two
    committed reports through here. It exists because a gate must not RECITE a
    provenance: until 2026-09-02 C1's own lines named four A100 slopes that are
    in no committed file, the plan output was corrected and this path was not,
    and nothing in the code could tell the two apart.

    It refuses in words rather than raising. Scoring happens after the pod time
    is already spent and an exception there would take the whole report with it;
    a report saying its band cannot be re-derived is worth more than no report.
    `_main` still calls `check_alpha_a_band` BEFORE measuring, where the same
    failure is free and the run refuses outright.
    """
    try:
        return list(check_alpha_a_band()[1])
    except CorpusMissing as exc:
        return [f"BAND PROVENANCE UNREADABLE, do not quote C1: {exc}"]


def sharpest_two_point_sd() -> tuple[float | None, str]:
    """The tightest two-point sd in the committed corpus, and where it is from.

    This is what `ALPHA_A_SD_CEILING` is set against: the reading a three-point
    fit exists to replace. Read rather than quoted, for the reason the constant
    gives. The ceiling used to be justified as "half the band width", and half a
    band that later widened is not a bar, it is whatever the hypothesis happened
    to become.
    """
    try:
        points = published_two_point_alpha_a()
    except CorpusMissing as exc:
        return None, str(exc)
    best = min(points, key=lambda p: p.sd)
    return best.sd, f"BM={best.block_m} of {best.source}"


def gate_sharpness(boot: Bootstrap) -> Gate:
    """C6: is the estimator sharp enough for C1 to mean anything.

    A CLAIM GATE AND NOT A VALIDITY ONE, changed 2026-09-02, and the reason is
    the shared table's own vocabulary. VALIDITY means the instrument broke and
    NOTHING on the page may be quoted (`exit_codes`, INVALID); this gate says
    one thing only, that alpha_a's interval is too wide for C1 to be tested,
    which is exactly what its `invalidates` field has always said. alpha_b, the
    invariance and the TEMPO comparison do not pass through alpha_a's spread and
    stay quotable, and C1 already reads UNKNOWN on its own when `sharp` is
    False, so the claim is not established either way.

    WHAT IT COST AS A VALIDITY GATE. The spread is PREDICTED to miss this bar at
    GROUP_SIZE_M=1: the design's power scales with `1 - alpha_b` and the corpus
    puts alpha_b near 0.93 there. So every G=1 arm exited 3 INVALID, the session
    driver reads anything but its listed finished code as RETRY, and the arm was
    re-measured on every pass at 11 GPU minutes a time while its own report said
    a G=1 run measures alpha_b and the invariance. A pinning whose power was
    predicted, printed in the plan and then met is not a broken instrument. It
    is a pre-registered expectation the world declined, which is what CLAIM_FAIL
    is for and what C4 at the same pinning already was.
    """
    sd = boot.alpha_a_sd
    sharpest, sharpest_source = sharpest_two_point_sd()
    rule = f"bootstrap sd(alpha_a) <= {ALPHA_A_SD_CEILING:.3f}"
    if sharpest is None:
        rule += (", the sharpest two-point sd in the committed corpus, which "
                 f"cannot be read here: {sharpest_source}")
    else:
        rule += (f", inside the {sharpest:.3f} sd of the sharpest two-point "
                 "slope this three-point fit replaces. That sd is the DECLARED "
                 "prior over the pair's ds and is PINNED there, so a measured "
                 "cross-arm floor does not lower this bar and does not make "
                 "the two-point reading sharper than it was registered at. "
                 "NOT half the band width either: the band is "
                 f"{ALPHA_A_BAND[1] - ALPHA_A_BAND[0]:.2f} wide and half of it "
                 "would be a looser bar than the reading being replaced")
    return Gate(CLAIM, "C6 estimator sharpness",
                "alpha_a's interval is tighter than the two-point reading it "
                "replaces",
                rule,
                None if sd is None else sd <= ALPHA_A_SD_CEILING,
                "no bootstrap spread" if sd is None else f"sd = {sd:.4f}",
                "C1 ALONE, which reads UNKNOWN: an estimator whose interval is "
                "wider than the hypothesis it tests has not tested it, and a "
                "PASS would be an artefact of the band's width. alpha_b, C3 and "
                "C5 do not pass through this spread and stay quotable",
                [boot.note]
                + ([] if sharpest is None
                   else [f"the bar is read from {sharpest_source}"]))


def gate_alpha_a(fit: Decomposition, boot: Bootstrap, sharp: bool,
                 band_lines: list[str] | None = None) -> Gate:
    """C1, scored against a band whose provenance is READ and never recited.

    `band_lines` is `check_alpha_a_band`'s second return value, threaded from
    the caller that already re-derived the band before spending a pod minute.
    When nobody passes it this gate reads the corpus itself through
    `band_provenance_lines` rather than printing a remembered sentence.

    THIS GATE IS WHAT THE PROVENANCE FINDING WAS ABOUT. The retired text named
    four A100 two-point slopes, 0.106, 0.102, 0.129, 0.119, that exist in no
    committed file, and it survived the fix that corrected the plan output
    because the plan and the gate printed the band from two different places.
    The gate is the copy that matters: it is what report.txt shows, what
    report.json carries under "gates", and what surrounds the RESULT line.
    """
    lo, hi = ALPHA_A_BAND
    val = fit.alpha_a
    sd = boot.alpha_a_sd
    inside = None if (val is None or not sharp) else lo <= val <= hi
    observed = ("not fitted" if val is None else
                f"alpha_a = {val:.4f}"
                + (f" +/- {sd:.4f}" if sd else " (no interval)"))
    if val is not None and not sharp:
        observed += " -- UNKNOWN, not scored: C6 says the interval is too wide"
    lines = (list(band_lines) if band_lines is not None
             else band_provenance_lines())
    return Gate(CLAIM, "C1 alpha_a", f"alpha_a lands in [{lo:.2f}, {hi:.2f}]",
                f"{lo:.2f} <= alpha_a <= {hi:.2f} from the {fit.form} fit",
                inside, observed,
                lines=lines + [
                    "A FAIL ABOVE the band with a clean residual would say both "
                    "committed two-point slopes were biased low by the "
                    "denominator the two-point form drops; a FAIL at or near "
                    "zero says BN does not move alpha at all and there is no "
                    "activation re-read to decompose. A PASS is weak: the two "
                    "slopes above disagree by a factor of two and the band "
                    "covers both, which is the width a three-point fit is here "
                    "to narrow."])


def gate_residual(fit: Decomposition, chi2: float | None, why: str,
                  struct: Structure, power: C2Power | None = None) -> Gate:
    """C2, and the one that matters more than the parameters.

    A fit always returns numbers. This asks whether the numbers describe the
    data: if the three terms are all of it, the residual is the measurement
    noise and chi2 is about 1. Structure in the residual means a term is
    missing, and `struct` names which candidate column it lines up with.

    THE POWER GUARD, added 2026-09-02, and the reason it is here rather than in
    the caller. This gate can PASS at a pinning where it cannot FAIL: planted at
    GROUP_SIZE_M=1 the missing-term world comes back well under the 4.0
    ceiling, which is the same PASS the TRUTH world gets, and the session
    driver schedules that arm unconditionally. `power` is the verdict of a
    planted MISSING world at THIS run's own swizzle and its own measured spread
    (`c2_power_probe`), and when it did not discriminate this gate reads UNKNOWN
    and says which. `power=None` means nobody asked -- the two calls that pass
    None are the probe's own scoring and `--self-test`'s four worlds, both of
    which ARE the question and must not ask it of themselves.

    A guard is not a softening. UNKNOWN counts against a CLAIM gate exactly as
    FAIL does (`moe.bench.exit_codes`), so an arm whose C2 has no power
    CLASSIFIES as 1 CLAIM_FAIL and its C2 can never be quoted as model
    completeness: the RESULT line says `CLAIM C2 UNKNOWN` and the reason is
    beside it.

    WHAT THE PROCESS ACTUALLY RETURNS IS 0 UNLESS `--fail-on-gate`. `_exit_over`
    reports CLAIM_FAIL as DONE by default, deliberately and out loud, because a
    claim that did not pass is a result and not a retry, and the session driver
    runs both bn arms without that flag. So the exit code is NOT what stops a
    powerless C2 being read as completeness; the printed verdict is, which is
    why the verdict is a machine-readable RESULT line rather than prose. A
    caller that needs the shortfall to be an error passes `--fail-on-gate` and
    gets the 1.
    """
    passed = None if chi2 is None else chi2 <= RESIDUAL_CHI2_CEILING
    observed = (f"chi2 = {chi2:.2f} over {fit.dof} dof; residual RMS "
                f"{fit.rms:.4f}" if chi2 is not None and fit.rms is not None
                else f"not computable: {why}")
    power_lines: list[str] = []
    if power is not None:
        power_lines = power.lines()
        if power.discriminates is not True:
            passed = None
            observed = ("NO POWER: " + power.reason()
                        + ("" if chi2 is None
                           else f". This run's own chi2 was {chi2:.2f}, which "
                                "is not quotable as a verdict"))
    return Gate(CLAIM, "C2 model completeness",
                "the three terms are ALL of it: the residual is noise",
                f"chi2 = sum (r/sigma)^2 / dof <= {RESIDUAL_CHI2_CEILING:.1f}, "
                "sigma from this run's own bootstrap, AND the same gate must "
                "FAIL on a planted missing-term world at this pinning",
                passed,
                observed,
                lines=power_lines + [f"structure: {struct.line()}",
                       "A FAIL is a RESULT and the more interesting one: it "
                       "says alpha_fitted is not the blend the study writes "
                       "down, and the correlation above names the term to add. "
                       "The candidate this run is built to see is alpha_b "
                       "itself moving with BN, which the model forbids and "
                       "which the published G=1 pair already hints at: the two "
                       "BN values there imply alpha_b 0.92 and 0.81."]
                + [f"  residual {r:+.4f} at {label}"
                   for r, label in zip(fit.residuals, fit.labels, strict=True)])


def gate_invariance(per_bm: dict[int, Decomposition], boot: Bootstrap) -> Gate:
    """alpha_b must not depend on BLOCK_M. Nothing in the model lets it."""
    vals = {bm: f.alpha_b for bm, f in per_bm.items() if f.alpha_b is not None}
    if len(vals) < 2:
        return Gate(CLAIM, "C3 alpha_b invariance",
                    "alpha_b is the same at every BLOCK_M",
                    f"|alpha_b(BM1) - alpha_b(BM2)| <= "
                    f"{INVARIANCE_SIGMA:.0f} sigma",
                    None, f"{len(vals)} block size(s) produced an alpha_b",
                    lines=["Two BM values are needed for this to be a test at "
                           "all; the usual reason for one is the "
                           "parallel-branch tolerance taking BLOCK_M=128."])
    order = sorted(vals)
    worst = (None, 0.0, 0.0)
    for i, bm1 in enumerate(order):
        for bm2 in order[i + 1:]:
            gap = abs(vals[bm1] - vals[bm2])
            s1 = boot.alpha_b_by_bm_sd.get(bm1)
            s2 = boot.alpha_b_by_bm_sd.get(bm2)
            sigma = math.hypot(s1 or 0.0, s2 or 0.0)
            n_sigma = gap / sigma if sigma > 0 else math.inf
            if worst[0] is None or n_sigma > worst[2]:
                worst = (f"BM={bm1} vs BM={bm2}", gap, n_sigma)
    have_sigma = bool(boot.alpha_b_by_bm_sd)
    return Gate(CLAIM, "C3 alpha_b invariance",
                "alpha_b is the same at every BLOCK_M",
                f"worst pair within {INVARIANCE_SIGMA:.0f} bootstrap sigma",
                None if not have_sigma else worst[2] <= INVARIANCE_SIGMA,
                ", ".join(f"BM={bm}: {v:.4f}" for bm, v in sorted(vals.items()))
                + f"; worst {worst[0]} gap {worst[1]:.4f} = "
                + ("no sigma" if not have_sigma else f"{worst[2]:.1f} sigma"),
                lines=["A FAIL says the fitted intercept absorbs something that "
                       "scales with BM and is not in the model -- which is the "
                       "same statement C2 makes, arriving by a route that does "
                       "not need a residual."])


def gate_physicality(fit: Decomposition, boot: Bootstrap) -> Gate:
    """alpha_b is a MISS FRACTION, so it lives in [0, 1] or it is not one.

    The check `moe/bench/ai_model.py` already enforces on its inputs, applied
    to the number this run fits. Above 1 an extra M-tile costs MORE than
    reading the whole expert once, which nothing in the model can produce; the
    quantity being divided by the weight bytes then contains traffic that is
    not weight traffic, which is the same statement C2 makes and is the reason
    this file exists.

    IT IS A LIVE RISK AT GROUP_SIZE_M=1 AND THE ARITHMETIC SAYS SO IN ADVANCE.
    The published mixtral BM=64, BN=64 alpha is 0.9475; removing a fixed cost of
    the size the compute reference measures puts alpha_b at about 1.16. Scored
    against the bootstrap spread rather than as a bare inequality, because that
    correction runs through `D`, which a four-tread ladder extrapolates loosely.
    """
    val = fit.alpha_b
    sd = boot.alpha_b_sd
    if val is None:
        return Gate(CLAIM, "C5 alpha_b is a miss fraction",
                    "0 <= alpha_b <= 1", "within 3 bootstrap sigma of [0, 1]",
                    None, "not fitted")
    over = max(0.0, val - 1.0, -val)
    n_sigma = over / sd if sd else (math.inf if over > 0 else 0.0)
    return Gate(CLAIM, "C5 alpha_b is a miss fraction",
                "0 <= alpha_b <= 1", "within 3 bootstrap sigma of [0, 1]",
                n_sigma <= 3.0,
                f"alpha_b = {val:.4f}"
                + (f" +/- {sd:.4f}" if sd else " (no spread)")
                + (f", outside [0, 1] by {over:.4f} = "
                   + ("no sigma" if not sd else f"{n_sigma:.1f} sigma")
                   if over > 0 else ", inside [0, 1]"),
                lines=["A FAIL above 1 is not a fitting artefact to be clipped: "
                       "it says the per-M-tile cost divided by the weight bytes "
                       "contains traffic the three terms do not name, which is "
                       "C2's verdict arriving without a residual."])


def gate_tempo(fit: Decomposition, boot: Bootstrap, group_m: int) -> Gate:
    """alpha_b against the closest prior work, with the swizzle named."""
    val = fit.alpha_b
    lo, hi = min(TEMPO_B2_OVER_B), max(TEMPO_B2_OVER_B)
    if val is None:
        near = None
        obs = "not fitted"
    else:
        near = min(abs(val / t - 1.0) for t in TEMPO_B2_OVER_B)
        obs = (f"alpha_b = {val:.4f}"
               + (f" +/- {boot.alpha_b_sd:.4f}" if boot.alpha_b_sd else "")
               + f", nearest TEMPO value off by {near:.1%}")
    return Gate(CLAIM, "C4 alpha_b against TEMPO",
                f"alpha_b corroborates TEMPO's b2/b of {lo:.3f}/{hi:.3f}",
                f"within {TEMPO_TOLERANCE:.0%} of the nearer TEMPO value",
                None if near is None else near <= TEMPO_TOLERANCE, obs,
                lines=[f"THIS RUN IS PINNED AT GROUP_SIZE_M={group_m} and "
                       "alpha_b is a property of that swizzle, not of the "
                       "kernel. The study's earlier 0.307, and its '2-4% from "
                       "TEMPO', are WITHDRAWN: that number was ALPHA_BY_BLOCK_M "
                       "solved through (LIN), a unit artefact of reading a "
                       "B/(A+B) fit as if it divided by weight bytes alone, not "
                       "a measurement of alpha_b; through (EXA) the same G=1 "
                       "ladders read alpha_b near 0.92, which is what "
                       "'consecutive M-tiles of one expert are scheduled far "
                       "apart and the L2 keeps nothing' should look like. It "
                       "was also pooled over GROUP_SIZE_M 1, 8, 16 and 64, "
                       "which swing alpha by 0.39. So a FAIL at G=1 does NOT "
                       "refute TEMPO, and a PASS at any G is the first number "
                       "in this study that may be compared with TEMPO at all. "
                       "Re-run at --group-m 16 to compare like with like."])


# --------------------------------------------------------------------------
# The registered predictions, printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def cell_table(cfg, b: int, ridge: float, block_ns, subjects, treads: dict
               ) -> list[str]:
    """Predicted alpha_fitted and B/C for every cell, in both worlds.

    Printed BEFORE the run and carried into the report, so "the model predicted
    this" is checkable rather than remembered. The two worlds are the study's
    withdrawn (LIN)-era refit and the study's own G=1 ladder fits read through
    (EXA); they disagree by a factor of three on alpha_b and this run lands in
    one of them.
    """
    out = ["                 LIN-ERA (0.307, 0.143)         "
           "LADDER (0.920, 0.146)      calibrated-ridge B/C",
           "  BM   BN    a_fit    B/C  treads      a_fit    B/C  treads    "
           "LIN-ERA LADDER"]
    for bm in sorted(subjects) + [REFERENCE_BLOCK_M]:
        for bn in sorted(block_ns):
            row = f"  {bm:3d} {bn:4d} "
            for ab, aa in (WORLD_LIN_ERA, WORLD_LADDER):
                a = alpha_fitted_exact(cfg, bm, bn, alpha_b=ab, alpha_a=aa)
                r = anchored_ratio(cfg, bm, bn, alpha_b=ab, alpha_a=aa)
                n = memory_treads(cfg, bm, bn, alpha_b=ab, alpha_a=aa,
                                  ratio=r, treads=treads.get(bm, 8))
                flag = "*" if abs(r - 1.0) <= TOLERANCE else " "
                row += f"  {a:6.3f} {r:6.3f}{flag} {n:3d}/{treads.get(bm, 8):<3d}"
            row += "    " + "  ".join(
                f"{branch_ratio(cfg, bm, bn, alpha_b=ab, alpha_a=aa, ridge=ridge, b=b):6.3f}"
                for ab, aa in (WORLD_LIN_ERA, WORLD_LADDER))
            out.append(row)
    out += [
        "  * = |B/C - 1| <= the parallel-branch tolerance. On a SUBJECT row "
        "the fit DISCARDS the memory branch",
        "  there and the cell yields NO alpha whatever the tread column says, "
        "because two branches within 15% of",
        f"  each other are one line. On the BLOCK_M={REFERENCE_BLOCK_M} "
        "REFERENCE row it is harmless: that ladder is never fitted",
        "  for a memory branch, only for proportionality, and 0/m treads is "
        "what qualifies it.",
        "  treads n/m = memory-bound treads out of the ladder's length. 0/m is "
        "a compute reference; m/m is a cell",
        "  whose alpha needs no reference for membership. The B/C columns are "
        f"ANCHORED at {ANCHOR_RATIO:.3f}, the measured median over the 22",
        "  published BLOCK_M=128 ladders, so the kernel's achieved rho "
        "cancels. The last two columns are the same",
        "  quantity computed from the CALIBRATED ridge instead; they run about "
        "15% high, and taken literally they say",
        f"  the BLOCK_M={REFERENCE_BLOCK_M} reference at BN={ANCHOR_BLOCK_N} "
        "is memory bound -- which it measurably is not, in 22 of 24 "
        "published arms."]
    return out


def predictions_text(cfg, b: int, ridge: float, ridge_source: str, block_ns,
                     subjects, treads: dict, group_m: int,
                     band_lines: list[str]) -> str:
    """P1..P7, with P1's band carrying the provenance it is derived from.

    `band_lines` comes from `check_alpha_a_band`, which re-reads the two
    committed arms the band is built on and refuses when the literal no longer
    matches them. It is passed in rather than recomputed here so that the
    refusal happens once, before any GPU time, instead of inside a print.
    """
    lo, hi = ALPHA_A_BAND
    keff = effective_k(cfg)
    return "\n".join([
        "## Predictions, registered before anything is measured", "",
        "THE ARITHMETIC (check it, do not test it; it is algebra, not a claim)",
        f"  W = 3 F H = {weight_elements(cfg):,} elements per expert; "
        f"one M-tile carries 2H + 3F = {act_once_elements(cfg):,} elements per "
        "row read or written once,",
        f"  and W/BN re-read per extra N-tile. K = W/(2H+3F) = {keff:.1f} "
        f"(ai_model.py uses the up-GEMM's K = {cfg.hidden_size}, a "
        f"{abs(keff / cfg.hidden_size - 1):.0%} difference,",
        f"  worth "
        f"{abs(PRIMARY_BLOCK_M / keff - PRIMARY_BLOCK_M / cfg.hidden_size):.4f}"
        f" in alpha at BM={PRIMARY_BLOCK_M}).",
        "  alpha_fitted = (alpha_b + phi)/(1 + phi + delta)   EXACT",
        "  alpha_fitted = alpha_b + alpha_a (BM/BN) + BM/K    the study's form, "
        "the same thing linearised at small phi",
        f"  B/C = ridge b (alpha_b + phi) / (2 BM), scored at ridge = "
        f"{ridge:.2f} Op/B ({ridge_source}).", "",
        "P1  alpha_a, as a SLOPE over three or more BN and not a two-point "
        f"difference, lands in [{lo:.2f}, {hi:.2f}].",
        *("    BASIS  " + line if i == 0 else "           " + line
          for i, line in enumerate(band_lines)),
        "    A FAIL at zero says BN does not move alpha and there is no "
        "activation re-read to model.",
        "    A FAIL high, with a clean residual, says the two-point slopes were "
        "biased by the denominator.", "",
        "P2  THE RESIDUAL IS NOISE: chi2 <= "
        f"{RESIDUAL_CHI2_CEILING:.1f} against this run's own bootstrap sigma.",
        "    This is the registered expectation and the one most likely to "
        "FAIL, because the published",
        "    G=1 pair ALREADY disagrees with the model: read through (EXA), "
        "BN=64 implies alpha_b 0.92",
        "    and BN=256 implies 0.81, a 0.107 gap against a published cross-arm "
        f"alpha spread of {PUBLISHED_ALPHA_SD:.3f}.",
        "    A FAIL names its shape: the candidate is alpha_b itself moving "
        "with BN, because BN changes",
        "    the order the L2 is walked in -- a term the three-term model "
        "forbids.", "",
        "P3  alpha_b is the SAME at every BLOCK_M, within "
        f"{INVARIANCE_SIGMA:.0f} bootstrap sigma. Nothing in the model lets a",
        "    weight miss fraction depend on the tile height, so this is the "
        "cheapest test of the whole",
        "    decomposition and it needs no external number.", "",
        f"P4  alpha_b at GROUP_SIZE_M={group_m}, against TEMPO's b2/b of "
        "0.311/0.319. NO prior agreement is claimed:",
        "    the study's earlier 'decomposed 0.307 corroborates TEMPO to 2-4%' "
        "is WITHDRAWN, because 0.307",
        "    was ALPHA_BY_BLOCK_M solved through (LIN), a unit artefact of the "
        "estimator, and pooled over",
        "    GROUP_SIZE_M 1, 8, 16 and 64 besides. This run pins ONE swizzle. "
        "Read through (EXA) the study's",
        "    own G=1 ladders put alpha_b near 0.92, so C4 is predicted to FAIL "
        f"at G={group_m} unless it is 16 or",
        "    above -- and that failure is a statement about the swizzle, not "
        "about TEMPO. Either verdict is the",
        "    first alpha_b in this study that may be compared with TEMPO at "
        "all.", "",
        "P5  WHICH CELLS GO MISSING, registered because two of them are "
        "predicted to.",
        f"    THE PRIMARY AT BN={ANCHOR_BLOCK_N} IS PREDICTED TO BE DISCARDED: "
        f"the anchor is {ANCHOR_RATIO:.3f}, inside the parallel-branch",
        f"    tolerance of {TOLERANCE:.2f}, and that is the measured median of "
        "22 published ladders rather than a guess.",
        "    At BN=32 the primary was predicted to escape UP (every tread "
        "memory bound) and at BN=128 to keep a",
        "    prefix, so it was registered to contribute 2 of 3 points. IT "
        "CONTRIBUTED NONE. In the committed",
        "    2026-09-10 H200 arm all three BLOCK_M=128 cells came back blank, "
        "at all three BLOCK_N, and the",
        "    pooled fit that was registered on three heights was made over "
        "TWO. That is why a FOURTH height",
        f"    is swept: BLOCK_M={SMALL_TILE_BLOCK_M} is the height furthest "
        "from its compute branch and so the one",
        "    least likely to lose it, and the identification line on the plan "
        "page prints what it buys.",
        "    BM=64, BM=32 and BM=16 sit far outside the parallel-branch band "
        "at every BN, which is what makes",
        "    the sweep survive the primary going missing a second time.",
        f"    THE BN=32 ARM IS THE ONE AT RISK: its BLOCK_M="
        f"{REFERENCE_BLOCK_M} reference is predicted compute bound in the "
        "LADDER",
        "    world and MEMORY bound in the LIN-ERA one, so the two worlds "
        "disagree about whether the arm exists.",
        "    If it is refused only two BN values remain, V5 FAILS, alpha_a is "
        "UNIDENTIFIED, and the deliverable is",
        "    that refusal plus this prediction naming it in advance -- not a "
        "fitted number.", "",
        "P6  BLOCK_N=256 CANNOT BE MEASURED AT ALL on either card as pinned: "
        f"BM={REFERENCE_BLOCK_M} x BN=256 needs 256",
        "    accumulator registers per thread against a maximum of 255, so the "
        "arm has no reference. It is",
        "    excluded by arithmetic before any GPU time, and the resource bill "
        "below prints the number.", "",
        "P7  The references agree across BN to within "
        f"{REFERENCE_CROSS_BN_SPREAD:.1f}x. The failure this bar exists for is "
        "43.6x,",
        "    from a kernel that spilled its accumulator and still fitted a line "
        "through the origin to 0.2%.", "",
        "PREDICTED CELLS, both worlds, before the run:",
        *cell_table(cfg, b, ridge, block_ns, subjects, treads),
    ])


# --------------------------------------------------------------------------
# The plan: every cell, its resource bill, and what it costs.
# --------------------------------------------------------------------------

def planted_ms(cfg, block_m: int, block_n: int, tiles: int, *, alpha_b: float,
               alpha_a: float, ridge: float, bandwidth_gbps: float, b: int,
               overhead_ms: float) -> float:
    """`D + max(L(1 + a(n-1)), C n)` from the model, in milliseconds.

    The generator for `--self-test` and the pricer for `--dry-run`. It is the
    model UNDER TEST, so nothing that reads it may be read as evidence for it:
    its job is to say what the data would look like in a named world.
    """
    a = alpha_fitted_exact(cfg, block_m, block_n, alpha_b=alpha_b,
                           alpha_a=alpha_a)
    load_bytes = (cfg.num_experts * weight_elements(cfg) * b
                  * (1.0 + phi(cfg, block_m, block_n, alpha_a)))
    load_ms = 1e3 * load_bytes / (bandwidth_gbps * 1e9)
    compute_ms = 1e3 * SWEEP.useful_flops(cfg, cfg.num_experts * block_m) / (
        ridge * bandwidth_gbps * 1e9)
    return overhead_ms + max(load_ms * (1.0 + a * (tiles - 1)),
                             compute_ms * tiles)


@dataclass(frozen=True)
class WarpTile:
    """How many of a CTA's warps hold an output tile of their own.

    THE THIRD BILL, added 2026-09-10 with the fourth subject height, and it is a
    RECORD where the other two are refusals. See `MMA_TILE_ELEMENTS`: shared
    memory and accumulator registers are hard limits and a setting that exceeds
    either does not run the tiling this sweep is about; a warp grid wider than
    the output tile runs it correctly and wastes warps. What this exists to stop
    is the wasted warps being DISCOVERED, on a pod, in a cell's branches.

    BOTH BRANCHES, not the compute one alone. This class was introduced saying
    the under-occupied cell's COMPUTE branch is what is incomparable; idle warps
    also carry fewer outstanding loads, so `B` and `w` are exposed too. The
    scope was widened the same day the finding was made and the check the run
    can make about it is `under_occupied_watch`.
    """

    block_m: int
    block_n: int
    num_warps: int

    @property
    def elements(self) -> int:
        return self.block_m * self.block_n

    @property
    def needed(self) -> int:
        return self.num_warps * MMA_TILE_ELEMENTS

    @property
    def warps_with_a_tile(self) -> int:
        return max(1, min(self.num_warps, self.elements // MMA_TILE_ELEMENTS))

    @property
    def occupied(self) -> bool:
        return self.elements >= self.needed

    def render(self) -> str:
        return (f"  BLOCK_M={self.block_m:4d}  output {self.elements:6d} "
                f"elements of {self.needed:6d} "
                f"({self.num_warps} warps x one {MMA_TILE_ELEMENTS}-element "
                "m16n8k16 tile)  "
                + ("full" if self.occupied else
                   f"UNDER-OCCUPIED: {self.warps_with_a_tile} of "
                   f"{self.num_warps} warps hold a tile"))


def warp_tile_bill(pinned: dict, block_m: int) -> WarpTile:
    """One setting's warp-grid occupancy. Arithmetic on the pinned constants."""
    return WarpTile(block_m, int(pinned["BLOCK_SIZE_N"]),
                    int(pinned["num_warps"]))


def under_occupied_cells(base_pinned: dict, subjects, block_ns
                         ) -> tuple[tuple[int, int], ...]:
    """Every swept `(BLOCK_M, BLOCK_N)` whose warp grid is wider than its tile."""
    return tuple(
        (bm, bn) for bn in block_ns for bm in subjects
        if not warp_tile_bill(dict(base_pinned, BLOCK_SIZE_N=bn), bm).occupied)


def _occupied_peers(base_pinned: dict, subjects, bm: int, bn: int
                    ) -> list[int]:
    """The swept heights at the SAME BLOCK_N whose warp grid is full."""
    return sorted(
        o for o in subjects if o != bm
        and warp_tile_bill(dict(base_pinned, BLOCK_SIZE_N=bn), o).occupied)


def _trend_pair(peers):
    """The two full-warp-grid heights the trend line is drawn through.

    ONE EXPRESSION, READ BY THE REGISTRATION AND BY THE READING, and it was two.
    Until 2026-09-10 `under_occupied_watch` REGISTERED the check against
    `peers[0], peers[1]` and `under_occupied_reading` PERFORMED it against
    `peers[0], peers[-1]`. At the default subjects that is BM=32/64 registered
    against BM=32/128 performed, a threefold difference in the excess reported
    at the one cell the whole identification gain rests on. The corpus hid it:
    every BLOCK_M=128 cell of the committed arm carries no `w`, so the reading's
    own filter left exactly two peers and the two expressions agreed by
    accident. It fires the moment a third height carries a `w`, which is the
    outcome this round exists to produce.

    THE TWO LOWEST, and the reason is not symmetry. The line is EXTRAPOLATED
    DOWN to a height below every peer, so the shortest extrapolation, off the
    nearest peers, is the least model-dependent one available; and the highest
    swept height is BLOCK_M=128, whose branch membership is the least reliable
    in the sweep, since it produced no alpha at any BLOCK_N in the committed
    arm. It is also what was REGISTERED before the run, and moving the
    performed check onto the registration is the only direction of this repair
    that leaves a pre-registration pre-registered.

    Takes anything indexable: the watch passes heights, the reading passes
    `(height, w)` pairs, and both get the same two elements of the same list.
    """
    return peers[0], peers[1]


def under_occupied_watch(base_pinned: dict, subjects, block_ns) -> list[str]:
    """The check registered against each under-occupied cell, before the run.

    REGISTERED 2026-09-10, AND THE REASON IS A GAP IN THE THIRD BILL'S OWN
    ARGUMENT. `MMA_TILE_ELEMENTS` scoped the incomparability of an
    under-occupied cell to its COMPUTE branch, and nothing argued the memory
    branch was untouched: idle warps carry fewer outstanding loads as well as
    fewer issue slots, which inflates `B`, `w` and `alpha` at exactly the cell
    the fourth height is being added for. The corpus cannot settle it, since
    the two published cells that touch the boundary sit exactly ON it and are
    full and nothing published is under-occupied, so the check is registered
    here rather than argued, and `under_occupied_reading` performs it on the
    cells the run returns.

    The comparison is within one BLOCK_N on purpose. Both live hypotheses about
    the BLOCK_N dependence, the model's `alpha_a g1` and the `1/BN` rival, are
    smooth in BLOCK_M at fixed BLOCK_N, so a slope that jumps at the height
    where the warp grid empties is the warp grid and not the tiling.
    """
    out: list[str] = []
    for bm, bn in under_occupied_cells(base_pinned, subjects, block_ns):
        peers = _occupied_peers(base_pinned, subjects, bm, bn)
        if len(peers) < 2:
            out.append(
                f"REGISTERED CHECK, BM={bm} x BN={bn}: NOT AVAILABLE at this "
                f"pinning. It needs two full-warp-grid heights at the same "
                f"BLOCK_N to set a trend and this sweep has {len(peers)}, so "
                "this cell's slope arrives with nothing to be read against.")
            continue
        lo, hi = _trend_pair(peers)
        out.append(
            f"REGISTERED CHECK, BM={bm} x BN={bn}: read its w against the line "
            f"BM={lo} and BM={hi} set at the SAME BN={bn}. Both "
            "live hypotheses are smooth in")
        out.append(
            "  BLOCK_M there, so a w ABOVE that line is the warp grid and "
            "not the tiling, and the cell must not be used to separate them "
            "until the excess is read.")
    return out or ["every swept cell fills its warp grid, so there is no "
                   "under-occupied slope to watch"]


def under_occupied_reading(cells, base_pinned: dict, subjects, block_ns
                           ) -> list[str]:
    """`under_occupied_watch`'s check, performed on the cells a run returned.

    The trend is a straight line in `log2(BLOCK_M)` through two peers' `w`,
    which is the coarsest form that uses two of them and introduces no
    parameter of its own. WHICH TWO IS `_trend_pair`, and it is the same
    expression the registration reads, which until 2026-09-10 it was not.
    Reported as an excess, never as a gate: what to do about an inflated `B` is
    a decision for whoever reads the arm, and a bar invented here would be a
    bar with no measurement behind it.
    """
    by_key = {(c.block_m, c.block_n): c for c in cells}
    out: list[str] = []
    for bm, bn in under_occupied_cells(base_pinned, subjects, block_ns):
        cell = by_key.get((bm, bn))
        if cell is None or cell.weight_streams is None:
            out.append(f"  under-occupied BM={bm} x BN={bn}: no w on this "
                       "page, so the registered warp-grid check cannot be "
                       "read")
            continue
        peers = [(o, by_key[(o, bn)].weight_streams)
                 for o in _occupied_peers(base_pinned, subjects, bm, bn)
                 if (o, bn) in by_key
                 and by_key[(o, bn)].weight_streams is not None]
        if len(peers) < 2:
            out.append(f"  under-occupied BM={bm} x BN={bn}: w = "
                       f"{cell.weight_streams:.4f}, and fewer than two "
                       "full-grid heights at this BN carry a w, so the "
                       "registered check has nothing to read it against")
            continue
        (m0, w0), (m1, w1) = _trend_pair(peers)
        slope = (w1 - w0) / (math.log2(m1) - math.log2(m0))
        expect = w0 + slope * (math.log2(bm) - math.log2(m0))
        out.append(
            f"  under-occupied BM={bm} x BN={bn}: w = {cell.weight_streams:.4f} "
            f"against {expect:.4f} on the BM={m0}/{m1} line at this BN, "
            f"{(cell.weight_streams / expect - 1):+.1%}. Above it is the warp "
            "grid, not the tiling.")
    return out


@dataclass(frozen=True)
class Plan:
    """Everything the pod run will do, computable on a laptop."""

    model: str
    dtype: str
    base_pinned: dict
    block_ns: tuple[int, ...]
    subjects: tuple[int, ...]
    rows: dict[tuple[int, int], list[int]]
    refusals: dict[tuple[int, int], str]
    reps: int
    group_m: int
    warmup_ms: float
    trials: int
    l2_flush: bool
    cell_budget_ms: float
    seconds: float
    #: The design-power reading, computed rather than quoted. See `design_power`.
    power: DesignPower | None = None
    #: The cross-arm floor printed beside every interval this run reports,
    #: carrying the word that says whether it was measured or assumed.
    floor: CrossArmFloor | None = None

    @property
    def timings(self) -> int:
        return self.reps * sum(len(v) for v in self.rows.values())

    def lines(self, cfg) -> list[str]:
        out = [
            f"model        {self.model} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       {self.base_pinned}  (BLOCK_SIZE_N is the sweep)",
            f"BN grid      {list(self.block_ns)}",
            f"subjects     {list(self.subjects)}, primary "
            f"{PRIMARY_BLOCK_M}; reference {REFERENCE_BLOCK_M}. EVERY arm's "
            "reference is measured first, across all arms, and level-checked "
            "before any subject costs anything -- an arm that qualifies none of "
            "its own can still borrow one, and whether it can is not knowable "
            "until the others exist",
            f"repeats      {self.reps} round-robin passes per setting",
        ]
        out += self.identification_lines(cfg)
        out += self.power_lines()
        out += [
            f"timing       {self.warmup_ms:.0f} ms of warmup under load, then "
            f"{self.trials} trial(s) of {self.cell_budget_ms:.0f} ms of kernel "
            f"time each, L2 flush {'ON' if self.l2_flush else 'OFF'}; the "
            "instrument sizes its own iteration count from the warmup's "
            "queue-deep per-call time",
            f"timings      {self.timings} "
            f"({sum(len(v) for v in self.rows.values())} treads x {self.reps} "
            "reps)",
            f"estimate     {self.seconds:.0f} s of GPU at what the instrument "
            "charges (warmup + trials x budget per timing), excluding compiles "
            "and allocation",
        ]
        for (bn, bm), rows in sorted(self.rows.items()):
            out.append(f"  BN={bn:4d} BM={bm:4d}  treads {len(rows):2d}  "
                       f"rows {rows[0]}..{rows[-1]}  T "
                       f"{SWEEP.tokens_for_rows(cfg, rows[0])}.."
                       f"{SWEEP.tokens_for_rows(cfg, rows[-1])}")
        for (bn, bm), why in sorted(self.refusals.items()):
            out.append(f"  REFUSED BN={bn} BM={bm}: {why}")
        return out

    def identification_lines(self, cfg) -> list[str]:
        """Why THESE heights, and what the fourth one buys, both computed.

        ADDED 2026-09-10 WITH THE FOURTH HEIGHT. The arm that ran on that date
        swept three heights and fitted TWO, because every BLOCK_M=128 cell had
        its memory branch discarded, and at two heights the model's activation
        column and the `1/BN` column that beat it are collinear. A plan page
        that prints four heights and does not say what the fourth is for is a
        page a reader has to take on trust, and the number is cheap: it is a
        property of the block sizes alone.
        """
        design = [(bm, bn) for bm in self.subjects for bn in self.block_ns]
        out = []
        r = design_collinearity(cfg, design)
        if r is not None:
            out.append(
                f"identification  corr(g1, 1/BN) over this design's "
                f"{len(design)} cells = {r:.3f}. g1 is the three-term model's "
                "only BLOCK_N-dependent")
            out.append(
                "             column and is strictly proportional to BLOCK_M; "
                "1/BN is the rival that fits the 2026-09-10 arm 3.1x")
            out.append(
                "             better at equal parameter count. Where the two "
                "are collinear no repeat count separates them.")
        measured = [(bm, bn) for bm in self.subjects if bm != SMALL_TILE_BLOCK_M
                    for bn in self.block_ns]
        r_without = design_collinearity(cfg, measured) if measured else None
        if r is not None and r_without is not None and measured != design:
            out.append(
                f"             without BLOCK_M={SMALL_TILE_BLOCK_M} the same "
                f"number is {r_without:.3f} over {len(measured)} cells, which "
                "is what the fourth height is worth.")
        # THE PAGE RE-DERIVES THE MEASUREMENT IT RESTS ON, from 2026-09-10.
        # `SESSION_2026_09_10`'s note said the numbers in the `SUBJECT_BLOCK_M`
        # paragraph are re-derived by `committed_bn_cells` AND
        # `published_small_tile_branch` on every plan page, and only the second
        # of the two was ever called: `committed_bn_cells` had no runtime call
        # site at all, so a re-publish that changed which cells the bn arm
        # yielded was caught by the test suite and NOT by the page. Both are
        # read here now, and both refuse out loud rather than leaving the
        # paragraph standing.
        try:
            committed = committed_bn_cells()
        except CorpusMissing as exc:
            out.append(f"             COMMITTED-ARM EVIDENCE UNREADABLE: {exc}")
        else:
            primary = [a for _, bm, a in committed if bm == PRIMARY_BLOCK_M]
            yielded = sorted({bm for _, bm, a in committed if a is not None})
            out.append(
                f"             the committed 2026-09-10 arm yielded "
                f"{sum(1 for a in primary if a is not None)} of "
                f"{len(primary)} BLOCK_M={PRIMARY_BLOCK_M} cells with an "
                f"alpha, so the fit registered on "
                f"{len({bm for _, bm, _ in committed})} heights was made over "
                f"{len(yielded)}: {yielded}.")
        try:
            branch = published_small_tile_branch()
        except CorpusMissing as exc:
            out.append(f"             SMALL-TILE EVIDENCE UNREADABLE: {exc}")
        else:
            out += ["             " + line for line in branch.lines()]
        return out

    def warp_lines(self) -> list[str]:
        """The warp-grid bill, one line per swept setting. A record, not a gate.

        The other two bills are printed by `block_m_crossing_sweep`'s own
        `TileResources.render`, which knows about shared memory and accumulator
        registers and nothing about how many warps hold a piece of the output.
        At BLOCK_M >= 64 that never mattered. It matters at 16.
        """
        out = ["WARP-GRID BILL, one CTA. A RECORD, NOT A REFUSAL: an "
               "under-occupied warp grid computes the right",
               "answer with warps to spare, so the cell is measured, but "
               "NEITHER of its branches is comparable with",
               "the rest of the grid: idle warps cost issue slots on the "
               "compute side and outstanding loads on the",
               "memory side, so C is understated and B, and therefore w and "
               "alpha, are overstated. It must never be",
               "promoted to a reference, and its slope must be read against "
               "the check below before it is used."]
        for bn in self.block_ns:
            pinned = dict(self.base_pinned, BLOCK_SIZE_N=bn)
            for bm in (*self.subjects, REFERENCE_BLOCK_M):
                out.append(f"  BN={bn:4d}"
                           + warp_tile_bill(pinned, bm).render())
        out += ["  " + line for line in
                under_occupied_watch(self.base_pinned, self.subjects,
                                     self.block_ns)]
        return out

    def power_lines(self) -> list[str]:
        """The design-power paragraph, EVERY NUMBER IN IT COMPUTED.

        This paragraph used to end with the literal string "sd 0.11-0.13 against
        a 0.025 bar ... at any rep count tried", beside a claim that the run
        "still measures alpha_b, C2 and C3". Both were wrong in the same way:
        the sd the code produced at those settings was 0.176, and C2 at that
        swizzle cannot fail. So the sd is now whatever `design_power` measured
        on a planted world at THIS pinning, the MDE beside it says what that sd
        can resolve, and the C2 sentence is a verdict from the same probe rather
        than a promise.
        """
        lever = ((1 - planted_alpha_b(self.group_m))
                 / (1 - planted_alpha_b(16)))
        out = [
            f"design power at GROUP_SIZE_M={self.group_m}: the corpus puts "
            f"alpha_b near {planted_alpha_b(self.group_m):.2f} there, and the "
            "response moves with alpha_a as g1 (1 - alpha_b)/(1 + phi)^2, so "
            f"the lever is {lever:.0%} of its size at GROUP_SIZE_M=16."]
        if self.power is None:
            out.append(
                "             design power NOT COMPUTED for this plan, so "
                "neither the spread nor the MDE below is available. Nothing "
                "here says the design works; it says nobody asked.")
            return out
        out += ["             " + line for line in self.power.lines()]
        out.append("             " + mde_line(
            self.power.alpha_a_sd, what="alpha_a, planted TRUTH world",
            assumption=(f"lognormal spread {self.power.noise:.2%} on every "
                        f"timing, from {self.power.noise_source}; "
                        f"{self.power.draws} bootstrap draws"),
            floor=self.floor))
        return out


def build_plan(args, cfg, b: int, capability, ridge: float,
               bandwidth_gbps: float, *, power: DesignPower | None = None,
               floor: CrossArmFloor | None = None) -> Plan:
    """The grid, the resource refusals, and the cost -- all off GPU.

    A SETTING THAT CANNOT HOLD ITS ACCUMULATOR IS DROPPED HERE, where it is
    chosen, and not diagnosed afterwards: a spilled kernel still returns a time,
    that time is still proportional to its tile count, and this study has
    already published 8 cells classified against one.

    AND AN ARM WHOSE REFERENCE IS DROPPED IS DROPPED WHOLE. The subjects of an
    arm with no reference cannot be classified, so measuring them would buy
    nothing but GPU time and a row of blanks that reads like data.

    THE COST IS PRICED AS `time_kernel` CHARGES, which is not what the retired
    loop charged. That instrument warms for a DURATION and then runs `--trials`
    trials each sized to `--cell-budget-ms` of kernel time, so a timing costs
    `warmup_ms + trials x cell_budget_ms` whatever the kernel's own duration is.
    The old estimate multiplied a per-call time by a call count and read a
    warmup duration as a count of calls, which under-priced every fast cell.
    """
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    subjects = tuple(int(v) for v in args.tiles.split(","))
    block_ns = tuple(int(v) for v in args.block_n_list.split(","))
    rows: dict[tuple[int, int], list[int]] = {}
    refusals: dict[tuple[int, int], str] = {}
    kept_ns: list[int] = []
    for bn in block_ns:
        pinned = dict(base, BLOCK_SIZE_N=bn)
        arm: dict[int, list[int]] = {}
        arm_refusals: dict[int, str] = {}
        for bm in (*subjects, REFERENCE_BLOCK_M):
            res = SWEEP.tile_resources(pinned, bm, b, capability)
            if res.refusal:
                arm_refusals[bm] = res.refusal
                continue
            arm[bm] = ladder_rows(cfg, bm, args.r_max, args.max_treads)
        if REFERENCE_BLOCK_M not in arm:
            why = arm_refusals.get(
                REFERENCE_BLOCK_M,
                f"BLOCK_M={REFERENCE_BLOCK_M} has no tread at --r-max "
                f"{args.r_max}")
            for bm in (*subjects, REFERENCE_BLOCK_M):
                refusals[(bn, bm)] = (
                    f"the whole BN={bn} arm is dropped because its reference "
                    f"cannot run: {why}")
            continue
        kept_ns.append(bn)
        for bm, r in arm.items():
            rows[(bn, bm)] = r
        for bm, why in arm_refusals.items():
            refusals[(bn, bm)] = why
    per_timing_ms = args.warmup + args.trials * args.cell_budget_ms
    total = args.reps * sum(len(rs) for rs in rows.values()) * per_timing_ms
    return Plan(args.model, args.dtype, base, tuple(kept_ns), subjects, rows,
                refusals, args.reps, args.group_m, args.warmup, args.trials,
                not args.no_l2_flush, args.cell_budget_ms, total / 1e3,
                power=power, floor=floor)


# --------------------------------------------------------------------------
# Persistence, identity, and the two ways this repo has already lost an arm.
# --------------------------------------------------------------------------

def append_sample(path: Path, sample: Sample, prov=None) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else.

    The run's provenance columns ride on every row. A cells.csv is the artefact
    that outlives the pod, and the audit's finding was that not one of the ten
    session scripts wrote a git sha or a card into one: a file of milliseconds
    with no commit behind it is an anecdote. The block is built ONCE per run and
    passed in, so a mid-run `git commit` cannot give two rows two shas.

    APPENDING TO A FILE WITH A DIFFERENT HEADER REFUSES, since 2026-09-09. The
    header is written once, when the file is new, and every later row is
    written positionally under it. Four clock-evidence columns were added to
    the row on 2026-09-09, so a resume into a cells.csv written before that day
    would have appended rows with four extra fields in a different order under
    the old header and silently misaligned the file. The runbook says --new for
    the booked rerun and `read_samples` reads by header name, so no committed
    file is affected; this is the check that makes the next added column say so
    instead of corrupting a directory.
    """
    new = not path.exists()
    row = asdict(sample)
    if prov is not None:
        row.update(prov.as_columns())
    if not new:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), None)
        if header is not None and header != list(row):
            added = [c for c in row if c not in header]
            gone = [c for c in header if c not in row]
            raise SystemExit(
                f"cells.csv at {path} was written with a different set of "
                f"columns than this run writes (added: {added or 'none'}; "
                f"missing: {gone or 'none'}). Appending would put the new "
                "fields under the old header and misalign every row from here "
                "on. Re-run into a NEW directory rather than resuming this one")
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _opt_float(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "", "None") else None
    except ValueError:
        return None


def _opt_bool(text: str | None) -> bool | None:
    if text in (None, "", "None"):
        return None
    return str(text).strip().lower() in ("1", "true", "yes")


def read_samples(path: Path) -> tuple[set[tuple[int, int, int, int]], list[Sample]]:
    """Timings already on disk, so a re-run resumes rather than repeats.

    Only SUCCESSFUL timings count as done: the common failure here is a pod
    that lost its device or a setting that ran out of shared memory, both of
    which a re-run can leave behind, and a real failure fails again in
    milliseconds.

    THE INSTRUMENT COLUMNS ARE READ BACK AS ABSENT, NEVER AS DEFAULTS, when the
    file predates them: `instrument` becomes the empty string and every clock
    flag None. A resumed directory whose older half says nothing about how it
    was timed must be visibly missing that, because those rows were produced by
    the retired per-iteration loop and are not comparable with the roof.
    """
    if not path.exists():
        return set(), []
    out: list[Sample] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Sample(
                block_n=int(row["block_n"]), block_m=int(row["block_m"]),
                tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), rep=int(row["rep"]),
                ms_p50=float(row["ms_p50"]), ms_min=float(row["ms_min"]),
                ms_stdev=float(row["ms_stdev"]), iters=int(row["iters"]),
                status=row.get("status", "ok"), detail=row.get("detail", ""),
                instrument=row.get("instrument", ""),
                warmup_ms=_opt_float(row.get("warmup_ms")) or 0.0,
                trials=int(row.get("trials") or 0),
                l2_flush=bool(_opt_bool(row.get("l2_flush"))),
                sm_clock_load_mhz=_opt_float(row.get("sm_clock_load_mhz")),
                sm_clock_start_mhz=_opt_float(row.get("sm_clock_start_mhz")),
                sm_clock_end_mhz=_opt_float(row.get("sm_clock_end_mhz")),
                clock_samples_mhz=row.get("clock_samples_mhz", "") or "",
                power_w=_opt_float(row.get("power_w")),
                clock_level_ok=_opt_bool(row.get("clock_level_ok")),
                clock_drift_ok=_opt_bool(row.get("clock_drift_ok")),
                host_bound=_opt_bool(row.get("host_bound")),
                # Tri-state, and "" on a row from before 2026-09-08 is the
                # one-sided era's "no side recorded". Since 2026-09-09 the
                # side is a record either way and excludes nothing.
                clock_level_side=row.get("clock_level_side", "") or ""))
    return ({(s.block_n, s.block_m, s.tiles, s.rep)
             for s in out if s.status == "ok"}, out)


def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file.

    `.gitignore` excludes `results/*` and re-includes only `results/published/`,
    so a run that writes anywhere else under the repo produces files `git add
    -A` silently drops. This project has already lost every published plot that
    way. Checked with `git check-ignore` rather than by re-implementing the
    pattern rules, because the pattern rules are what got it wrong.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=ROOT, capture_output=True, timeout=15,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path unverified"
    if proc.returncode == 0:
        return ("IGNORED by git. Nothing written here enters the repo. Publish "
                "with scripts/publish_results.sh, or point --out at "
                "results/published/<date>-<gpu>-bn-decomposition")
    if proc.returncode == 1:
        return "git will keep this path"
    return (f"git check-ignore exited {proc.returncode}; path unverified "
            f"({proc.stderr.decode(errors='replace').strip()})")


def detect_card_slug() -> str | None:
    """Slug for the ATTACHED device, or None when there is no device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return re.sub(r"[^a-z0-9]+", "_",
                  torch.cuda.get_device_name(0).lower()).strip("_")


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter AND the card, so two settings cannot
    collide.

    This is not a precaution, it is a repair. The sweep beside this one lost a
    whole arm to an id that omitted GROUP_SIZE_M: the second run resumed into
    the first's directory, found every timing present, skipped all of them, and
    printed the first run's numbers under the second's heading. The SAME
    omission then survived in four more fields until 2026-09-02, including THE
    CARD -- and the proof is committed: two published directories, an A100 one
    and an H200 one, both contain a report named
    `mixtral-8x7b-bf16-r1024-g1-n64-4867a2`, for sm_count 108 and 132.

    BUILT BY `moe.bench.provenance.run_id` AS OF 2026-09-02, not by a private
    hash here. Three scripts had each re-implemented a subset of the same rule
    and each had left a different knob out. That function refuses a missing card
    (`NoCard`) and a None or empty knob (`UnresolvedKnob`), hashes the knobs in
    sorted order so the id does not depend on the order they were named in, and
    puts the card slug at the FRONT where `ls` shows it.

    BLOCK_SIZE_N IS THE SWEEP HERE, so the LIST is in the key rather than a
    value: two runs over different BN grids are different experiments even when
    the grids overlap, because `--block-n-list 32,64,128` and `64,128` fit
    different numbers of points and V5 reads a different verdict.

    THREE KNOBS ARE NEW TO THE KEY AND ALL THREE SET THE MEASURED
    MILLISECONDS: `warmup` (now a DURATION of sustained load, not a call count),
    `trials`, and `flush`, which decides whether every timed iteration starts
    with a cold L2. A flushed and an unflushed sweep must never share a
    directory; the roof was measured flushed.

    `plant_noise` IS IN THE KEY AS THE OPERATOR'S CHOICE, NEVER AS THE RESOLVED
    VALUE. When it is not given it contributes the token "auto", because the
    resolved number is derived from the cells this id names: an id containing it
    would depend on its own directory's contents and would change halfway
    through the sweep, which is the resume collision this function exists to
    prevent, arriving from the other direction.

    `--ridge`, `--bandwidth-gbps`, `--draws` and `--power-draws` stay OUT: they
    re-analyse a set of timings rather than change one, and two analyses of one
    sweep belong in one directory.
    """
    return PV.run_id(
        card=card,
        model=args.model,
        dtype=args.dtype,
        n=tuple(int(v) for v in args.block_n_list.split(",")),
        bm=tuple(int(v) for v in args.tiles.split(",")),
        ref=REFERENCE_BLOCK_M,
        r=args.r_max,
        treads=args.max_treads,
        reps=args.reps,
        g=args.group_m,
        k=args.block_k,
        stages=args.num_stages,
        warps=args.num_warps,
        iters=args.iters,
        warmup=args.warmup,
        trials=args.trials,
        flush=not args.no_l2_flush,
        budget=args.cell_budget_ms,
        seed=args.seed,
        plantnoise=(PLANT_NOISE_AUTO if args.plant_noise is None
                    else args.plant_noise),
    )


def arm_cache(root: Path, block_n: int, block_m: int) -> Path:
    """Point Triton at a fresh directory for THIS (BN, BM), before it compiles.

    Deliberately NOT `block_m_crossing_sweep.arm_triton_cache`, which keys on
    BLOCK_M alone. That is right for a sweep whose only variable is BLOCK_M and
    wrong here: two BN arms at one BM would share a directory, the second would
    find it warm, compile nothing, and be scored by V1 as a broken override --
    the same class of collision as a run id that omits a swept knob, one level
    down.
    """
    directory = root / f"n{block_n}-bm{block_m}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(directory)
    return directory


def measure_setting(args, cfg, block_n: int, block_m: int, rows: list[int],
                    csv_path: Path, cache_root: Path, pinned: dict, done,
                    samples: list[Sample], *, prov=None,
                    reference_clock_mhz: float | None = None) -> tuple[int, int]:
    """Time one (BN, BM) setting, `--reps` round-robin passes over its treads.

    ROUND ROBIN INSIDE THE SETTING. Measuring tread 1 fifty times and then
    tread 8 fifty times puts every tread at a different point in the pod's
    thermal history, and the resulting monotone drift IS a slope -- the very
    quantity being fitted. One pass over all treads per repeat spreads that
    drift across the ladder instead of aligning it with the x axis.

    THE INSTRUMENT IS `moe.bench.timing.time_kernel` AND NOTHING ELSE. It warms
    for `--warmup` MILLISECONDS of delivered GPU load, sizes its own iteration
    count so a trial lasts `--cell-budget-ms` of kernel time, primes one event
    pair per iteration outside the loop, flushes L2 before every timed call and
    samples the SM clock under load. The private `SWEEP.time_call` this used
    until 2026-09-02 did none of that, and the roof every alpha here is scored
    against was measured queue-deep, so the two were never comparable: the audit
    put the bias at 8-16% in the fitted alpha at the smallest cells, DIFFERENT
    PER CARD, which is the size of the cross-card effect this study registered.

    `reference_clock_mhz` is the clock the roof was measured at, resolved once
    for the whole arm. Without it every cell's LEVEL verdict is None, meaning
    "not determined": a level is relative to something and neither this function
    nor the instrument will invent the something.

    A REFUSAL FROM THE INSTRUMENT IS NOT A FAILED CELL. `TimingRefused` says the
    measurement could not be made at all (no CUDA and no injected fakes), and
    catching it per cell would write a grid of `status="failed"` rows and burn
    the whole arm to learn it once. It is re-raised, and the arm stops.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.bench import timing
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, _ = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    arm_cache(cache_root, block_n, block_m)
    seen: set[Path] = set()
    SWEEP.count_new(cache_root, seen)
    compiles = executed = 0
    built: dict[int, tuple] = {}

    for rep in range(1, args.reps + 1):
        for r in rows:
            tokens = SWEEP.tokens_for_rows(cfg, r)
            if (block_n, block_m, r // block_m, rep) in done:
                continue
            if tokens not in built:
                spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                                 routing=RoutingSpec("uniform", 0.0),
                                 seed=args.seed)
                x, weights = make_inputs(spec, device="cuda")
                ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
                w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                               device="cuda")
                kw = vllm_call_kwargs(spec)
                kw["activation"] = MoEActivation(kw["activation"])
                built = {tokens: (x, weights, ids, w, kw)}   # one cell live
            x, weights, ids, w, kw = built[tokens]
            executed += 1
            conf = dict(pinned, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n)

            def call(_f=fused_experts, _x=x, _wt=weights, _w=w, _i=ids, _k=kw):
                return _f(hidden_states=_x, w1=_wt.w1, w2=_wt.w2,
                          topk_weights=_w, topk_ids=_i, **_k)

            try:
                with override_config(conf):
                    call()
                    torch.cuda.synchronize()
                    compiles += SWEEP.count_new(cache_root, seen)
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=reference_clock_mhz)
                sample = Sample(
                    block_n, block_m, r // block_m, r, tokens, rep,
                    t.ms_p50, t.ms_min, t.ms_std, t.iters,
                    instrument=t.instrument, warmup_ms=t.warmup_ms,
                    trials=t.trials, l2_flush=t.l2_flush,
                    sm_clock_load_mhz=t.sm_clock_load_mhz,
                    clock_level_ok=t.clock_level_ok,
                    clock_drift_ok=t.clock_drift_ok, host_bound=t.host_bound,
                    clock_level_side=clock_side_of(t), **clock_samples_of(t))
                # THE SIDE IS RECORDED HERE AND EXCLUDES NOTHING. Since
                # 2026-09-09 `clock_excluded` reads DRIFT alone, so DRIFT and
                # host-bound get the exclusion-shaped marker and BOTH LEVEL
                # sides are named as kept: on this card a steady LOW is a
                # hungry tile and a steady HIGH is a memory-shaped one, and
                # each is an operating point the row records.
                if (clock_excluded(t.clock_level_ok, sample.clock_level_side,
                                   t.clock_drift_ok) or t.host_bound):
                    print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
                elif sample.clock_level_side:
                    print(f"  ^ kept (LEVEL {sample.clock_level_side} is "
                          f"recorded, not excluded): "
                          f"{t.clock_note or ''}".rstrip())
            except timing.TimingRefused:
                raise
            except Exception as exc:                    # noqa: BLE001
                sample = Sample(block_n, block_m, r // block_m, r, tokens, rep,
                                0.0, 0.0, 0.0, 0, "failed",
                                f"{type(exc).__name__}: {exc}")
                print(f"  BN={block_n} BM={block_m} n={r // block_m} rep={rep} "
                      f"FAILED {sample.detail}")
                if "shared memory" in str(exc).lower():
                    print("  ^ re-run the WHOLE sweep with --num-stages "
                          f"{max(1, pinned['num_stages'] - 1)}. Dropping stages "
                          "for one setting alone would unpin the thing this "
                          "sweep holds fixed.")
            samples.append(sample)
            append_sample(csv_path, sample, prov)
            print(f"  BN={block_n:4d} BM={block_m:4d} n={r // block_m:2d} "
                  f"rep={rep:2d} r={r:5d} T={tokens:6d}  "
                  f"{sample.ms_p50:9.4f} ms ({sample.iters} iters)")
    return compiles, executed


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

def analyse_run(samples, cfg, args, *, ridge: float, bandwidth_gbps: float,
                b: int, ceiling_tflops: float, ceiling_source: str, capability,
                base_pinned: dict, compiles: dict, executed: dict,
                sm_count: int, block_ns, subjects, draws: int | None = None,
                reference_clock_mhz: float | None = None,
                probe_c2_power: bool = False,
                plant_noise: float | None = None,
                band_lines: list[str] | None = None,
                bootstrap_seed: int | None = None,
                ) -> tuple[list[str], list[Gate], dict]:
    """Everything read off the timings, as text, gates and a payload.

    `probe_c2_power` decides whether C2 is scored with a power guard. It is True
    exactly once per real run and False everywhere else, and the everywhere-else
    is not a convenience: the guard's own probe scores planted worlds through
    this same function, so a probe that probed would not terminate, and
    `--self-test`'s four worlds ARE the discrimination question and cannot be
    asked to answer it about themselves.

    `plant_noise` overrides the spread the probe plants at; None means use the
    one this run measured (`measured_spread`), which is the point of running the
    probe inside the run rather than from the command line.

    `band_lines` is the alpha_a band's provenance, already re-derived by `_main`
    before any GPU time, handed down so C1 prints the band it was actually
    scored against. None means C1 reads the corpus itself; it never recites.

    `bootstrap_seed` is the resampling seed, and it exists because until
    2026-09-10 there was only `args.seed`. `design_power` varies the PLANTED
    world's seed per realisation, so a caller that could not also move the
    bootstrap got five worlds resampled with one frozen draw sequence: the fix
    had been applied at one of its two call sites. Measured before the fix,
    `--dry-run --group-m 16 --power-seeds 5 --seed 0` returned 0.1774, 0.0836,
    0.0801, 0.0252, 0.2684 while `--seed k --power-seeds 1` for k=0..4 returned
    0.1774, 0.0831, 0.1136, 0.0696, 0.1734, and only the base seed agreed. None
    means use `args.seed`, so a real run is unchanged.

    `reference_clock_mhz` is the clock the FIXED roof was measured at, passed
    in the way `ceiling_tflops` is rather than resolved here, so a replay on a
    machine with no card scores the session's own numbers. It gates nothing:
    it is the denominator of the issue-efficiency figure printed beside every
    fixed-roof fraction, and of V3's normalised spread.
    """
    # ONE DTYPE, TWO REPRESENTATIONS, RECONCILED HERE. `b` is the byte WIDTH
    # the tile-resource bill needs; `args.dtype` is the NAME the weight-set byte
    # count needs, because two dtypes can share a width and a weight set
    # derived from a width alone is a guess. In a real run both come from
    # `dtype_bytes(args.dtype)` at one line in main; a caller that hands in a
    # `b` from some other dtype is describing two experiments, and says so here
    # rather than silently scoring the alphas at one width and every `w` at
    # another.
    if b != dtype_bytes(args.dtype):
        raise ValueError(
            f"b={b} is not the width of --dtype {args.dtype} "
            f"({dtype_bytes(args.dtype)}): the tile-resource bill and the "
            "weight-stream denominator would be computed for two dtypes")
    kw = dict(block_ns=block_ns, subjects=subjects, ridge=ridge,
              bandwidth_gbps=bandwidth_gbps, dtype=args.dtype,
              base_pinned=base_pinned,
              capability=capability, ceiling_tflops=ceiling_tflops,
              sm_count=sm_count, reference_clock_mhz=reference_clock_mhz)
    draws = args.draws if draws is None else draws
    cells, verdicts, spreads = arm_alphas(samples, cfg, **kw)
    keys = [(c.block_n, c.block_m) for c in cells if c.usable]

    inversion_rows: list[str] = []
    for bn in block_ns:
        for bm in (*subjects, REFERENCE_BLOCK_M):
            pts, sp = collapse(samples, bn, bm)
            for tread, drop in inversions(pts):
                if sp is None or drop >= MONOTONE_SIGMA * sp:
                    inversion_rows.append(
                        f"BN={bn} BM={bm} tread {tread}: time falls {drop:.3%}"
                        + (f" at spread {sp:.3%}" if sp else " (spread unknown)"))

    boot = run_bootstrap(samples, cfg, keys, draws=draws,
                         seed=args.seed if bootstrap_seed is None
                         else bootstrap_seed,
                         form="EXA", **kw)
    fit = decompose(cells, cfg, "EXA")
    fit_lin = decompose(cells, cfg, "LIN")
    fit_raw3 = decompose(cells, cfg, "EXA3")
    per_bm = {bm: decompose(cells, cfg, "EXA", block_m=bm)
              for bm in sorted({c.block_m for c in cells if c.usable})}
    per_bm_lin = {bm: decompose(cells, cfg, "LIN", block_m=bm)
                  for bm in per_bm}
    chi2, why = chi_square(fit, cells, boot, cfg)
    struct = structure_of(fit, cells, cfg, chi2)
    cross_why, cross_spread = cross_bn_refusal(verdicts)
    cross_normalised = cross_bn_normalised_spread(verdicts)

    lines = ["", "## The arms", ""]
    for v in verdicts:
        lines += v.render()
    if cross_spread is not None:
        lines.append(f"  across BN, implied rates span {cross_spread:.2f}x "
                     f"(gate <= {REFERENCE_CROSS_BN_SPREAD:.1f}x, on the raw "
                     "rates against the fixed roof)")
        if cross_normalised is not None:
            lines.append(
                f"  the same spread with each arm rescaled to its own clock "
                f"is {cross_normalised:.2f}x, printed as a record: the arms "
                "held different clocks under the power cap and the gate is "
                "pre-registered on the raw rates")

    # THE SLOPE IN WEIGHT-STREAM UNITS SITS BESIDE THE ALPHA, and the A-versus-D
    # label sits beside both. Added 2026-09-10, when the session's own analysis
    # showed that every unphysical number this study has argued about came out
    # of `B/(A+B)` and its `D`-corrected partner rather than out of the data:
    # `w` has no fitted level in it at all, and `D>A` marks the cells where
    # `alpha_upper > 1` is arithmetic. Neither replaces `alpha`, which is what
    # every gate here is still scored on; they are printed so a reader can see
    # which of the three a given cell's oddity belongs to.
    stream_ms = (WEIGHT_STREAM_MS(cfg, args.dtype, bandwidth_gbps)
                 if bandwidth_gbps > 0 else None)
    lines += ["", "## The cells", "",
              "   BN   BM   treads  mem   alpha    corrected       w  A/D   "
              "sigma  surv  basis"]
    for c in sorted(cells, key=lambda c: (c.block_m, c.block_n)):
        sd = boot.per_cell_sd.get((c.block_n, c.block_m))
        surv = boot.survival.get((c.block_n, c.block_m))
        lines.append(
            f"  {c.block_n:4d} {c.block_m:4d}   {c.treads:4d} {c.memory_points:4d}   "
            + (f"{c.alpha:7.4f}" if c.alpha is not None else "  BLANK")
            + ("  " + (f"{c.alpha_corrected:9.4f}"
                       if c.alpha_corrected is not None else "     n/a"))
            + ("  " + (f"{c.weight_streams:6.4f}"
                       if c.weight_streams is not None else "   n/a"))
            + f"  {c.ad_label or '   ':3s}"
            + ("  " + (f"{sd:6.4f}" if sd else "   n/a"))
            + ("  " + (f"{surv:4.0%}" if surv is not None else " n/a"))
            + "  " + (c.blank or c.basis)[:64])
    lines += [
        "  w = B / T_w, the memory branch's slope in units of ONE COMPLETE "
        "stream of this layer's expert weights",
        "  at the card's own rate. "
        + (f"T_w = {stream_ms:.4f} ms for {cfg.num_experts} x 3 F H x {b} B "
           f"at {bandwidth_gbps:.1f} GB/s." if stream_ms
           else "T_w UNAVAILABLE: no bandwidth to stream at.")
        + " NO fitted level, no intercept,",
        "  no delta and no D enter it, which is what makes it comparable "
        "across arms that alpha is not.",
        f"  estimator: {WEIGHT_STREAM_SOURCE}",
        "  A/D = whether this arm's compute reference's fitted fixed cost D "
        "exceeds this ladder's own",
        "  intercept A. `alpha_upper = B/(A + B - D)` is above 1 EXACTLY when "
        "D > A, so a D>A cell's",
        "  alpha_upper is unphysical by arithmetic and not by measurement. In "
        "the committed 2026-09-10",
        "  arm that held in four of six cells and in no others."]
    # THE WARP-GRID CHECK, READ. `under_occupied_watch` registers it on the
    # plan page before the pod runs; this is the same check performed on the
    # cells that came back. It is here and not in the gates because an inflated
    # B at an under-occupied cell is a reading, not a verdict.
    lines += under_occupied_reading(cells, base_pinned, subjects, block_ns)
    fitted = [c for c in sorted(cells, key=lambda c: (c.block_m, c.block_n))
              if c.slope_memory_ms is not None and c.intercept_ms is not None]
    if not fitted:
        lines.append("  no cell produced a memory branch, so there is no A, B "
                     "or D on this page to read the labels against")
    for c in fitted:
        lines.append(
            f"    BN={c.block_n:4d} BM={c.block_m:4d}  A={c.intercept_ms:8.5f} "
            f"B={c.slope_memory_ms:8.5f} D={c.fixed_cost_ms:8.5f} ms  "
            f"{c.ad_label}")

    lines += ["", "## The decomposition", ""]
    for name, f in (
            ("EXACT  alpha_upper = (alpha_b + phi)/(1 + phi), 2 parameters",
             fit),
            ("LINEAR alpha = alpha_b + alpha_a BM/BN + BM/K, the study's form",
             fit_lin),
            ("RAW3   alpha = (alpha_b + phi)/(1 + phi + delta), delta FITTED "
             "-- cross-check only, never gated", fit_raw3)):
        if f.alpha_a is None:
            lines.append(f"  {name}: {f.note}")
            continue
        lines.append(
            f"  {name}"
            f"\n    alpha_b {f.alpha_b:8.4f}"
            + (f" +/- {boot.alpha_b_sd:.4f}" if boot.alpha_b_sd
               and f.form == "EXA" else "")
            + f"   alpha_a {f.alpha_a:8.4f}"
            + (f" +/- {boot.alpha_a_sd:.4f}" if boot.alpha_a_sd
               and f.form == "EXA" else "")
            + (f"   delta {f.delta:8.4f}"
               + (f" +/- {boot.delta_sd:.4f}" if boot.delta_sd else "")
               if f.delta is not None else "")
            + f"\n    {f.note}, residual RMS "
            + (f"{f.rms:.4f}" if f.rms is not None else "n/a"))
    lines.append("  THE TWO FORMS ARE THE SAME MODEL, linearised or not. Where "
                 "they disagree, phi is not small: it is "
                 f"{phi(cfg, PRIMARY_BLOCK_M, 64, ALPHA_A_BAND[1]):.3f} at "
                 f"BM={PRIMARY_BLOCK_M}, BN=64 on this model, so the linear "
                 "form's denominator is off by that much.")
    lines += ["", "  per BLOCK_M (this is what C3 reads):"]
    for bm in sorted(per_bm):
        e, ln = per_bm[bm], per_bm_lin[bm]
        lines.append(
            f"    BM={bm:4d}  EXA alpha_b "
            + (f"{e.alpha_b:7.4f} alpha_a {e.alpha_a:7.4f}"
               if e.alpha_a is not None else f"  --      ({e.note[:48]})")
            + (f"   LIN alpha_b {ln.alpha_b:7.4f} alpha_a {ln.alpha_a:7.4f}"
               if ln.alpha_a is not None else ""))

    counts = {"bn arms with a reference": sum(1 for v in verdicts if v.ok),
              "arms on their OWN branch": sum(1 for v in verdicts
                                              if v.ok and not v.imported),
              "usable alpha cells": len(keys),
              "bootstrap draws": boot.draws,
              "timings read": sum(1 for s in samples if s.status == "ok"),
              "treads fitted": sum(c.treads for c in cells)}
    sharp = boot.alpha_a_sd is not None and boot.alpha_a_sd <= ALPHA_A_SD_CEILING

    # THE POWER PROBE RUNS HERE, AFTER THE SPREAD IS KNOWN AND BEFORE C2 IS
    # SCORED. It plants the missing-term world at THIS run's own across-repeat
    # spread and this run's own swizzle and asks whether C2 could have failed.
    # Inside the probe's own scoring `probe_c2_power` is False, which is the
    # recursion guard; see `analyse_run`'s docstring.
    spread, spread_source = measured_spread(spreads)
    power: C2Power | None = None
    if probe_c2_power:
        noise, noise_source = resolve_plant_noise(plant_noise, spread,
                                                  spread_source)
        power = c2_power_probe(
            cfg, args, b=b, ceiling_tflops=ceiling_tflops,
            capability=capability, block_ns=block_ns, subjects=subjects,
            sm_count=sm_count, noise=noise, noise_source=noise_source,
            draws=getattr(args, "power_draws", POWER_PROBE_DRAWS))

    floor, floor_why = read_cross_arm_floor()
    lines += ["", "## Noise, and which kind of it", "",
              "  " + boot.note,
              "  SCOPE: these resamples are of WITHIN-PROCESS WARM REPEATS -- "
              "one process, one allocation, one clock state, repeats "
              "interleaved round robin. Nothing that changes between processes "
              "is in them, and every cross-arm comparison this study publishes "
              "is exactly that kind of comparison, so every interval below is "
              "a LOWER bound on the uncertainty of a number compared across "
              "arms.",
              "  " + (floor.line() if floor is not None
                      else f"cross-arm floor UNAVAILABLE: {floor_why}"),
              "  " + mde_line(boot.alpha_a_sd, what="alpha_a, this run",
                              assumption=(
                                  "this run's own bootstrap over "
                                  f"{boot.draws} draws; measured across-repeat "
                                  "spread "
                                  + ("unknown" if spread is None
                                     else f"{spread:.2%}")
                                  + f" ({spread_source})"),
                              floor=floor)]
    if power is not None:
        lines += ["  " + line for line in power.lines()]

    # NOT IN `counts`: `gate_non_vacuity` fails on any zero there, and a run
    # with zero LOW treads is the run one wants.
    clocks = clock_state(samples)
    lines += ["", "## Clock state of the timed treads", ""]
    lines += clock_state_lines(clocks)

    gates = [
        gate_non_vacuity(counts),
        gate_override(compiles, executed),
        gate_reference_level(verdicts),
        gate_cross_bn(cross_why, cross_spread, cross_normalised),
        gate_ladders(spreads, inversion_rows, boot.survival),
        gate_identifiable(cells, PRIMARY_BLOCK_M),
        gate_sharpness(boot),
        gate_alpha_a(fit, boot, sharp, band_lines),
        gate_residual(fit, chi2, why, struct, power),
        gate_invariance(per_bm, boot),
        gate_physicality(fit, boot),
        gate_tempo(fit, boot, args.group_m),
    ]
    payload = {
        "ridge": ridge, "bandwidth_gbps": bandwidth_gbps,
        # THE SCOPE TRAVELS WITH THE NUMBER, and until 2026-09-10 it did not
        # travel with this one: the cells carried `weight_streams` as a bare
        # ratio while its denominator and the identity of the estimator that
        # produced it were printed on the page and NOWHERE in report.json. A
        # reader working from the JSON alone could not tell this file's own
        # copy from the shared estimator, which is exactly the distinction
        # `WEIGHT_STREAM_CANDIDATES` exists to keep, and on 2026-09-10 the copy
        # was the one that ran.
        "weight_stream_ms": stream_ms,
        "weight_stream_source": WEIGHT_STREAM_SOURCE,
        "ceiling_tflops": ceiling_tflops, "ceiling_source": ceiling_source,
        "pinned": base_pinned, "block_ns": list(block_ns),
        "subjects": list(subjects), "reference_block_m": REFERENCE_BLOCK_M,
        # Rebuilt WITHOUT `ref`, which holds the sweep's own dataclass and is
        # not JSON. EVERY OTHER FIELD IS CARRIED, and this rebuild is a second
        # call site of the record: `basis` and `import_note` are here because
        # a cell resting on another arm's ruler has to be distinguishable in
        # the file and not only in the printout, and the two clocks are here
        # because the same is true of the operating point it ran at. The two
        # own-clock numbers are properties and are written out by hand, so
        # report.json carries the figure the page printed rather than the
        # ingredients for it.
        "arms": [dict({k: v for k, v in asdict(
            RefVerdict(v.block_n, v.block_m, v.slope_per_tile, v.overhead_ms,
                       v.implied_tflops, v.ceiling_tflops, v.fraction,
                       v.refusals, v.note, basis=v.basis,
                       import_note=v.import_note,
                       load_clock_mhz=v.load_clock_mhz,
                       reference_clock_mhz=v.reference_clock_mhz)).items()
            if k != "ref"},
            roof_at_cell_clock_tflops=v.own_clock_tflops,
            own_clock_fraction=v.own_clock_fraction)
            for v in verdicts],
        "cells": [asdict(c) for c in cells],
        "cell_sigma": {f"{k[0]}:{k[1]}": v for k, v in boot.per_cell_sd.items()},
        "cell_survival": {f"{k[0]}:{k[1]}": v for k, v in boot.survival.items()},
        "fits": {name: {"form": f.form, "block_m": f.block_m,
                        "alpha_b": f.alpha_b, "alpha_a": f.alpha_a,
                        "delta": f.delta, "n_cells": f.n_cells,
                        "dof": f.dof, "rms": f.rms,
                        "residuals": list(f.residuals),
                        "labels": list(f.labels), "note": f.note}
                 for name, f in [("pooled_exact", fit),
                                ("pooled_linear", fit_lin),
                                ("pooled_raw_delta_fitted", fit_raw3)]
                 + [(f"exact_bm{bm}", v) for bm, v in per_bm.items()]
                 + [(f"linear_bm{bm}", v) for bm, v in per_bm_lin.items()]},
        "chi2": chi2, "chi2_note": why,
        "structure": {"correlations": struct.correlations,
                      "worst": struct.worst_name, "value": struct.worst_value,
                      "read": struct.read},
        "bootstrap": {"draws": boot.draws, "alpha_a_sd": boot.alpha_a_sd,
                      "alpha_b_sd": boot.alpha_b_sd, "delta_sd": boot.delta_sd,
                      "alpha_b_by_bm_sd": boot.alpha_b_by_bm_sd,
                      "note": boot.note,
                      # THE SCOPE TRAVELS WITH THE NUMBER. A reader working from
                      # report.json alone must not have to know that these
                      # resamples cannot see between-process noise.
                      "scope": "within-process warm repeats, resampled with "
                               "replacement; blind to anything that changes "
                               "between processes",
                      "mde_alpha_a": (mde_one_sample(boot.alpha_a_sd)
                                      if boot.alpha_a_sd else None),
                      "mde_convention": f"one-sample, two-sided "
                                        f"{POWER.TEST_LEVEL}, power "
                                        f"{POWER.TEST_POWER}",
                      "cross_arm_prior_sd": None if floor is None else floor.sd,
                      "cross_arm_prior_sd_basis": ("UNAVAILABLE" if floor is None
                                                   else floor.basis),
                      "cross_arm_prior_sd_source": (floor_why if floor is None
                                                    else floor.source)},
        "measured_spread": spread, "measured_spread_source": spread_source,
        "clock_state": clocks,
        "c2_power": (None if power is None else
                     {**asdict(power), "discriminates": power.discriminates,
                      "reason": power.reason()}),
        "alpha_a_band": list(ALPHA_A_BAND),
        "spreads": spreads, "inversions": inversion_rows,
        "gates": [asdict(g) for g in gates],
        "result_lines": [g.result_line() for g in gates],
    }
    return lines, gates, payload


# --------------------------------------------------------------------------
# The two questions asked of a PLANTED world before a real one is scored.
#
# Both are the same machinery `--self-test` uses, called from inside the run
# rather than from the command line, because the audit's finding was that the
# off-GPU self-test was run at GROUP_SIZE_M=16 and the pod arm was scheduled at
# 1, where the same gates cannot discriminate. A check that is only performed at
# a setting other than the one being paid for is not a check on that setting.
# --------------------------------------------------------------------------

#: The missing term `--self-test`'s MISSING world plants and the C2 power probe
#: re-plants: alpha_b rising with `(BM/BN)^2`, a shape the three terms cannot
#: absorb. Sized several times the published cross-arm alpha spread and small
#: enough not to move a cell into another regime, so a C2 FAIL is the residual
#: gate seeing a term and not the grid collapsing. It is module level so that
#: the world the probe asks about and the world the self-test advertises are the
#: SAME world; two copies would let the pod be guarded against one shape while
#: the report claimed the other.
MISSING_TERM_SIZE = 0.004


def missing_term(block_m: int, block_n: int) -> float:
    """The MISSING world's extra alpha_b, in units of alpha."""
    return MISSING_TERM_SIZE * (block_m / block_n) ** 2


def bn_drift_term(block_m: int, block_n: int) -> float:      # noqa: ARG001
    """BN-DRIFT: alpha_b itself moving with 1/BN, which the model forbids.

    The specific alternative this experiment was built to see, and the one the
    published G=1 BN pair already hints at (its two BN values imply alpha_b 0.92
    and 0.81).
    """
    return 6.0 / block_n


def measured_spread(spreads: dict) -> tuple[float | None, str]:
    """This run's own across-repeat spread, as one number, and where it is from.

    The median over the arms of each arm's median across-repeat relative
    standard deviation. A median rather than the worst, because one arm made
    noisy by a single failed tread would otherwise set the noise every planted
    world is generated at; the worst is printed beside it in V4.
    """
    seen = [s for s in spreads.values() if s is not None and s > 0]
    if not seen:
        return None, ("no arm produced an across-repeat spread; a spread needs "
                      "at least two repeats of one tread")
    return (statistics.median(seen),
            f"this run's own repeats, median over {len(seen)} arm(s)")


def resolve_plant_noise(explicit: float | None, measured: float | None,
                        measured_source: str) -> tuple[float, str]:
    """The spread every planted world is generated at, and its provenance.

    IN ORDER: what the operator asked for, then what THIS POD MEASURED, then the
    worst spread the corpus has published. The middle one is the change the
    audit asked for. `--plant-noise` used to default to 0.008, the middle of the
    published range, and every design-power verdict this file printed was a
    verdict about a pod quieter than half the sessions in the corpus: at 0.008
    S4 passes at G=16 and at 0.015-0.020 it fails. A default that decides
    whether an experiment is worth paying for has to come from the machine that
    will run it.
    """
    if explicit is not None:
        return float(explicit), "given on the command line"
    if measured is not None:
        return float(measured), measured_source
    return PLANT_NOISE_FALLBACK, PLANT_NOISE_FALLBACK_SOURCE


def planted_world_gates(cfg, args, *, alpha_b: float, alpha_a: float, extra,
                        noise: float, b: int, ceiling_tflops: float,
                        capability, block_ns, subjects, sm_count: int,
                        draws: int, seed: int | None = None
                        ) -> tuple[dict, dict]:
    """Generate one named world and score it exactly as a real run is scored.

    Returns `({gate key: Gate}, payload)`. The gate key is the first token of
    the gate's name, so a caller asks for "C2" rather than matching a sentence.

    THE PLANTED BANDWIDTH IS DERIVED FROM THE PLANTED RATE, NOT ASSUMED.
    `planted_ms` charges compute at `rho x bandwidth`, so planting the anchor's
    rho beside the card's PEAK bandwidth plants a kernel running at 114% of its
    card -- which V2 then refuses, and the probe would be testing the level gate
    against a world the level gate is right to reject. The compute rate is
    planted at `PLANT_COMPUTE_FRACTION` of the ceiling, inside the 38-64% the
    published references reach, and the bandwidth follows from it.

    ONE SEED MOVES BOTH HALVES OF THE REALISATION. `seed` reaches the planted
    timings AND the bootstrap that resamples them, so realisation k here is
    exactly what `--seed k --power-seeds 1` prints. Before 2026-09-10 it reached
    only the timings and `design_power`'s five realisations shared one frozen
    draw sequence.

    `probe_c2_power` is OFF for this call and that is the recursion guard:
    scoring a planted world must not itself plant a world to ask whether its own
    C2 had power, or the probe would never terminate.
    """
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    rho = achieved_rho(cfg, b, alpha_b=alpha_b, alpha_a=alpha_a)
    bw = PLANT_COMPUTE_FRACTION * ceiling_tflops * 1e3 / rho
    samples = planted_samples(cfg, args, alpha_b=alpha_b, alpha_a=alpha_a,
                              ridge=rho, bandwidth_gbps=bw, b=b,
                              block_ns=block_ns, subjects=subjects, extra=extra,
                              noise=noise,
                              seed=args.seed if seed is None else seed)
    compiles = {(bn, bm): 1 for bn in block_ns
                for bm in (*subjects, REFERENCE_BLOCK_M)}
    _, gates, payload = analyse_run(
        samples, cfg, args, ridge=rho, bandwidth_gbps=bw, b=b,
        ceiling_tflops=ceiling_tflops, ceiling_source="planted",
        capability=capability, base_pinned=base, compiles=compiles,
        executed=dict(compiles), sm_count=sm_count, block_ns=block_ns,
        subjects=subjects, draws=draws, probe_c2_power=False,
        bootstrap_seed=args.seed if seed is None else seed)
    return {g.name.split()[0]: g for g in gates}, payload


@dataclass(frozen=True)
class DesignPower:
    """What a planted TRUTH world says this pinning can resolve. Computed.

    Replaces the sentence "alpha_a's own spread is 0.11 to 0.13 ... at every rep
    count tried", which was a string in a docstring and in a plan line while the
    number the code produced at those settings was 0.176. Every field here comes
    from a bootstrap that just ran.
    """

    group_m: int
    reps: int
    noise: float
    noise_source: str
    draws: int
    alpha_a_sd: float | None
    alpha_b_sd: float | None
    note: str = ""
    #: EVERY seed's sd, not one. See `design_power`: at the fallback spread the
    #: single-seed figure this record used to carry swings 19-fold between
    #: seeds, and it is the number that decides whether the arm is paid for.
    alpha_a_sds: tuple[float, ...] = ()
    #: The same realisations for alpha_b. `alpha_b_sd` is their median, which is
    #: what `alpha_a_sd` has always been; before 2026-09-10 it was the LAST
    #: iteration's value, so the two fields on one record were computed over
    #: different things.
    alpha_b_sds: tuple[float, ...] = ()

    @property
    def worst_alpha_a_sd(self) -> float | None:
        """The widest spread any planted realisation gave. What `resolves` uses."""
        return max(self.alpha_a_sds) if self.alpha_a_sds else self.alpha_a_sd

    @property
    def resolves(self) -> bool | None:
        """True, False, or None for "no spread came back to judge".

        SCORED ON THE WORST REALISATION AND NOT THE TYPICAL ONE, from
        2026-09-10. It is a strictly harder bar than the single draw it
        replaces: a design that resolves alpha_a on three seeds out of five has
        not been shown to resolve it on the pod, and the seed a plan happens to
        run with is the least interesting fact about the design.
        """
        worst = self.worst_alpha_a_sd
        if worst is None:
            return None
        return worst <= ALPHA_A_SD_CEILING

    def lines(self) -> list[str]:
        if self.alpha_a_sd is None:
            return [f"design power UNKNOWN: no bootstrap spread came back from "
                    f"the planted TRUTH world ({self.note or 'no note'})"]
        verdict = ("RESOLVES alpha_a" if self.resolves else
                   "CANNOT RESOLVE alpha_a")
        out = [f"planted TRUTH world at GROUP_SIZE_M={self.group_m}, "
               f"{self.reps} reps, spread {self.noise:.2%} "
               f"({self.noise_source}): sd(alpha_a) = {self.alpha_a_sd:.4f} "
               f"against the {ALPHA_A_SD_CEILING:.3f} C1 needs -> {verdict}"]
        if len(self.alpha_a_sds) > 1:
            out.append(
                f"that sd is the MEDIAN over {len(self.alpha_a_sds)} planted "
                f"realisations, which ran "
                + ", ".join(f"{v:.4f}" for v in sorted(self.alpha_a_sds))
                + f"; the verdict is scored on the WORST of them "
                  f"({max(self.alpha_a_sds):.4f}). One realisation is not a "
                  "property of the design: at the worst published spread this "
                  "figure swings by an order of magnitude between seeds, and "
                  "it is what decides whether the arm is paid for.")
        if self.resolves is False:
            out.append("AT THIS PINNING C1 WILL READ UNKNOWN however the run "
                       "goes. --group-m 16 is the setting that resolves it; "
                       "--reps buys the rest, and this line is what says how "
                       "much it bought.")
            out.append("C6 will FAIL with it and both are CLAIM gates, so the "
                       "arm is expected to end 1 CLAIM_FAIL, printed as exit 0 "
                       "without --fail-on-gate. It is NOT 3 INVALID and must "
                       "not be re-measured: alpha_b, C3 and C5 are what a run "
                       "at this pinning is for and they stay quotable.")
        return out


def design_power(cfg, args, *, b: int, ceiling_tflops: float, capability,
                 block_ns, subjects, sm_count: int, noise: float,
                 noise_source: str, draws: int) -> DesignPower:
    """Plant the TRUTH world at this pinning and report what it could resolve.

    Off GPU, before the pod, and printed by `--dry-run`: the whole point is that
    it is a property of the PINNING and of the pod's noise, not of the run, so
    it can be known before the run is paid for.

    OVER SEVERAL PLANTED REALISATIONS SINCE 2026-09-10, AND THE REASON IS A
    MEASUREMENT. This function planted ONE world at `--seed` and printed the sd
    that came back. At `--plant-noise 0.008` that is stable to about 30% across
    seeds; at the fallback 1.82%, which is the WORST published across-repeat
    spread and the spread a plan with no measurement of its own is scored at,
    the same figure ran 0.0223, 0.0868, 0.3102, 0.0684, 0.3445 and 0.4171 over
    six consecutive seeds of the THREE-height design, a 19-fold swing, with
    seed 0 the single value that clears the 0.025 bar. A design-power verdict
    that is a property of the seed decided whether an arm was worth renting a
    pod for. The median is now printed and the WORST is what `resolves` reads.

    IT IS NOT A DEFECT OF THE FOURTH HEIGHT AND MUST NOT BE READ AS ONE. The
    same six seeds on the FOUR-height design give 0.1774, 0.0831, 0.1136,
    0.0696, 0.1734 and 0.0219: the same instability, over a narrower range.
    Both designs resolve alpha_a at 0.8% and neither is shown to at 1.8%.

    A REALISATION IS THE WORLD AND THE RESAMPLE TOGETHER, SINCE 2026-09-10. The
    loop's `seed` used to reach `planted_samples` only; the bootstrap inside
    `analyse_run` went on resampling at `args.seed`, frozen across every
    iteration. So the five realisations this line reported were not the five
    the two sequences above name, and `--seed k --power-seeds 1` did not
    reproduce the k-th of them: at `--group-m 16` the loop returned 0.1774,
    0.0836, 0.0801, 0.0252, 0.2684 against a single-seed 0.1774, 0.0831,
    0.1136, 0.0696, 0.1734, agreeing only at the base seed. `bootstrap_seed`
    carries the loop's seed the rest of the way, and both sequences above are
    now what `--power-seeds 6` prints, seed for seed.
    """
    sds: list[float] = []
    b_sds: list[float] = []
    note = ""
    for seed in range(args.seed, args.seed + max(1, args.power_seeds)):
        _, payload = planted_world_gates(
            cfg, args, alpha_b=planted_alpha_b(args.group_m), alpha_a=0.14,
            extra=None, noise=noise, b=b, ceiling_tflops=ceiling_tflops,
            capability=capability, block_ns=block_ns, subjects=subjects,
            sm_count=sm_count, draws=draws, seed=seed)
        boot = payload["bootstrap"]
        note = note or boot["note"]
        if boot["alpha_a_sd"] is not None:
            sds.append(float(boot["alpha_a_sd"]))
        if boot.get("alpha_b_sd") is not None:
            b_sds.append(float(boot["alpha_b_sd"]))
    # BOTH FIELDS OVER THE SAME REALISATIONS. Until 2026-09-10 `alpha_a_sd` was
    # the median over the loop while `alpha_b_sd` was whatever the LAST
    # iteration happened to leave in `boot`, so one record carried two
    # statistics computed over different things. Nothing printed alpha_b_sd, so
    # nothing had read it wrong yet.
    return DesignPower(args.group_m, args.reps, noise, noise_source, draws,
                       statistics.median(sds) if sds else None,
                       statistics.median(b_sds) if b_sds else None, note,
                       alpha_a_sds=tuple(sds), alpha_b_sds=tuple(b_sds))


@dataclass(frozen=True)
class C2Power:
    """Whether C2 could have FAILED at this run's own swizzle and spread.

    THE GATE THIS RECORD GUARDS IS THE ONE THE EXPERIMENT EXISTS FOR. C2 says
    "the three terms are ALL of it", and at GROUP_SIZE_M=1 the audit found the
    planted MISSING-term world coming back well under the 4.0 ceiling: a
    PASS, the same verdict the TRUTH world gets. A C2 PASS there
    would have been published as model completeness by a test that cannot say
    otherwise, and the driver schedules that arm unconditionally.

    So the real run plants that world itself, at its OWN measured across-repeat
    spread and its OWN swizzle, before it scores anything, and C2 reads UNKNOWN
    with `reason()` whenever the missing-term world would have passed. UNKNOWN
    counts against the gate by `moe.bench.exit_codes`'s rule, so the gates
    CLASSIFY as CLAIM_FAIL rather than DONE: a gate that could not decide has
    not passed. The process still exits 0 unless `--fail-on-gate` is given
    (`_exit_over`, which says so in the log), so what carries this verdict off
    the page is the `RESULT: CLAIM C2 UNKNOWN` line and not the exit code.
    """

    ran: bool
    group_m: int
    reps: int
    noise: float
    noise_source: str
    draws: int
    truth_pass: bool | None = None
    missing_pass: bool | None = None
    truth_chi2: float | None = None
    missing_chi2: float | None = None
    note: str = ""

    @property
    def discriminates(self) -> bool | None:
        """True only when the probe ran AND answered differently in the two worlds.

        None means the probe could not run, which is not the same as "it has
        power" and must not be scored as if it were.
        """
        if not self.ran or self.truth_pass is None or self.missing_pass is None:
            return None
        return self.truth_pass is True and self.missing_pass is False

    def reason(self) -> str:
        """One line, safe for a RESULT detail: no newlines, names the numbers."""
        where = (f"GROUP_SIZE_M={self.group_m}, {self.reps} reps, spread "
                 f"{self.noise:.2%} from {self.noise_source}, "
                 f"{self.draws} draws")
        if not self.ran:
            return f"the C2 power probe could not run ({self.note}); {where}"
        if self.missing_pass is not False:
            return (
                "the planted missing-term world passes at this swizzle / noise "
                f"({where}: MISSING chi2 "
                + ("n/a" if self.missing_chi2 is None
                   else f"{self.missing_chi2:.2f}")
                + f" against a ceiling of {RESIDUAL_CHI2_CEILING:.1f}, the same "
                  "verdict TRUTH gets), so a PASS here could not have been a FAIL")
        if self.truth_pass is not True:
            return (f"the planted TRUTH world FAILS C2 at this pinning ({where}: "
                    "chi2 "
                    + ("n/a" if self.truth_chi2 is None
                       else f"{self.truth_chi2:.2f}")
                    + "), so C2 fails on a world the model describes and a real "
                      "FAIL would say nothing")
        return (f"the probe discriminates at this pinning ({where}: TRUTH "
                "PASSES, MISSING FAILS), so C2's verdict below is a verdict")

    def lines(self) -> list[str]:
        return [f"C2 power probe: {self.reason()}"]


def c2_power_probe(cfg, args, *, b: int, ceiling_tflops: float, capability,
                   block_ns, subjects, sm_count: int, noise: float,
                   noise_source: str, draws: int) -> C2Power:
    """Plant TRUTH and MISSING at this pinning and ask whether C2 can tell them apart.

    Two worlds and no more: a gate discriminates if the world it is looking for
    fails it and the world it is not passes. The other two self-test worlds
    (NO-A, BN-DRIFT) test different claims and would double the probe's cost for
    a question C2's power does not depend on.

    THE SEEDS ARE DIFFERENT ON PURPOSE. Both worlds run at `--seed`, which is the
    same noise realisation, so the only difference between them is the planted
    term. That is what makes "MISSING passes too" a statement about the gate
    rather than about a lucky draw.
    """
    common = dict(b=b, ceiling_tflops=ceiling_tflops, capability=capability,
                  block_ns=block_ns, subjects=subjects, sm_count=sm_count,
                  draws=draws)
    ab = planted_alpha_b(args.group_m)
    try:
        truth_gates, truth_payload = planted_world_gates(
            cfg, args, alpha_b=ab, alpha_a=0.14, extra=None, noise=noise,
            **common)
        missing_gates, missing_payload = planted_world_gates(
            cfg, args, alpha_b=ab, alpha_a=0.14, extra=missing_term,
            noise=noise, **common)
    except Exception as exc:                                # noqa: BLE001
        # NOT swallowed into a PASS. A probe that crashed has established
        # nothing, `discriminates` is None, and C2 reads UNKNOWN with this
        # sentence as its reason.
        return C2Power(False, args.group_m, args.reps, noise, noise_source,
                       draws, note=f"{type(exc).__name__}: {exc}")
    return C2Power(True, args.group_m, args.reps, noise, noise_source, draws,
                   truth_pass=truth_gates["C2"].passed,
                   missing_pass=missing_gates["C2"].passed,
                   truth_chi2=truth_payload["chi2"],
                   missing_chi2=missing_payload["chi2"])


# --------------------------------------------------------------------------
# Self test: plant four worlds and check the gates come out DIFFERENT.
# --------------------------------------------------------------------------

def planted_samples(cfg, args, *, alpha_b: float, alpha_a: float,
                    ridge: float, bandwidth_gbps: float, b: int,
                    block_ns, subjects, extra=None, noise: float = 0.004,
                    seed: int = 0) -> list[Sample]:
    """Timings generated FROM the model, so the whole analysis has a known answer.

    `extra(block_m, block_n) -> float` adds a term the model does NOT contain,
    in units of alpha. That is what makes the residual gate testable: a gate
    that cannot be made to FAIL by a missing term is not testing for one.

    EVERY ROW CARRIES `SYNTHETIC_INSTRUMENT`, the sweep's own name for "this was
    not measured". `instrument` is a column on the CSV and one of the five keys
    a publish gate reads at the top of a report, so a planted row that carried
    `TIMING_BASIS` would satisfy that gate while describing an instrument no
    process ever ran. "Not measured" is a VALUE here, never an absence.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    for bn in block_ns:
        for bm in (*subjects, REFERENCE_BLOCK_M):
            a_extra = extra(bm, bn) if extra else 0.0
            for r in ladder_rows(cfg, bm, args.r_max, args.max_treads):
                n = r // bm
                for rep in range(1, args.reps + 1):
                    ms = planted_ms(cfg, bm, bn, n, alpha_b=alpha_b + a_extra,
                                    alpha_a=alpha_a, ridge=ridge,
                                    bandwidth_gbps=bandwidth_gbps, b=b,
                                    overhead_ms=args.overhead_ms)
                    ms *= math.exp(rng.gauss(0.0, noise))
                    out.append(Sample(
                        bn, bm, n, r, SWEEP.tokens_for_rows(cfg, r), rep,
                        ms, ms, ms * noise, 0,
                        instrument=SWEEP.SYNTHETIC_INSTRUMENT,
                        warmup_ms=0.0, trials=0, l2_flush=False))
    return out


def self_test(args, cfg, b: int, ridge: float, bandwidth_gbps: float,
              ceiling_tflops: float, capability, block_ns, subjects, *,
              noise: float, noise_source: str
              ) -> tuple[list[str], list[Gate]]:
    """Four worlds. The claim is that the gates DISCRIMINATE, not that they pass.

      TRUTH     the exact model at the corpus's alpha_b for this swizzle and
                alpha_a = 0.14. C1 and C2 must PASS and alpha_a must come back
                near what it was planted at.
      MISSING   the same, plus a term the model does not contain: alpha_b
                rising with (BM/BN)^2. C2 must FAIL and the structure test must
                name the quadratic column.
      NO-A      alpha_a = 0. C1 must FAIL at the bottom of the band, and C2
                must still PASS -- a wrong parameter with a clean residual is a
                different verdict from a wrong model, and the two gates have to
                tell them apart.
      BN-DRIFT  alpha_b moving linearly with 1/BN, which is the specific
                alternative this experiment was built to see. C2 must FAIL.

    `noise` is RESOLVED BY THE CALLER and its provenance printed, because the
    verdicts here are only as good as it is: at 0.8% S4 passes at G=16 and at
    1.5-2.0% -- inside the published H200 range -- it fails. It used to be an
    argparse default of 0.008, the middle of that range, which made every
    design-power verdict a verdict about a pod quieter than half the corpus.

    THE FOUR WORLDS ARE SCORED WITH `probe_c2_power` OFF, through
    `planted_world_gates`. C2's power guard asks exactly the question S2 asks,
    so leaving it on would let a guarded UNKNOWN in the MISSING world satisfy
    "C2 FAILS in MISSING" -- a gate proving itself with its own guard.
    """
    ab0 = planted_alpha_b(args.group_m)
    worlds = {
        "TRUTH": (ab0, 0.14, None),
        "MISSING": (ab0, 0.14, missing_term),
        "NO-A": (ab0, 0.0, None),
        "BN-DRIFT": (ab0, 0.14, bn_drift_term),
    }
    rho0 = achieved_rho(cfg, b, alpha_b=ab0, alpha_a=0.14)
    lines = ["", "## Self test: four planted worlds", "",
             "  planted at the achieved rho that reproduces the measured "
             f"anchor: {rho0:.1f} Op/B against a calibrated {ridge:.1f}; "
             f"alpha_b planted at {ab0:.3f}, the corpus value at "
             f"GROUP_SIZE_M={args.group_m}",
             f"  every timing carries a lognormal spread of {noise:.2%}, from "
             f"{noise_source}", "",
             "  world      alpha_a  sd(a_a)    alpha_b   chi2 cells   C2     "
             "structure"]
    gates: list[Gate] = []
    verdicts: dict[str, tuple] = {}
    for name, (ab, aa, extra) in worlds.items():
        by, pay = planted_world_gates(
            cfg, args, alpha_b=ab, alpha_a=aa, extra=extra, noise=noise, b=b,
            ceiling_tflops=ceiling_tflops, capability=capability,
            block_ns=block_ns, subjects=subjects,
            sm_count=args.sm_count or 132, draws=args.draws)
        fitted = pay["fits"]["pooled_exact"]
        chi2 = pay["chi2"]
        sd = pay["bootstrap"]["alpha_a_sd"]
        verdicts[name] = (fitted["alpha_a"], fitted["alpha_b"], chi2,
                          by["C1"].passed, by["C2"].passed,
                          pay["structure"]["worst"], sd, fitted["n_cells"])
        lines.append(
            f"  {name:9s} "
            + ("    n/a " if fitted["alpha_a"] is None
               else f"{fitted['alpha_a']:8.4f}")
            + ("   n/a " if sd is None else f"{sd:7.4f}")
            + ("    n/a " if fitted["alpha_b"] is None
               else f"{fitted['alpha_b']:9.4f}")
            + ("   n/a" if chi2 is None else f"{chi2:7.2f}")
            + f"  {fitted['n_cells']:3d}"
            + f"  {str(by['C2'].passed):5s}  "
            + str(pay["structure"]["worst"]))

    truth_a, truth_b, _, _, truth_c2, _, truth_sd, _ = verdicts["TRUTH"]
    tol = max(0.03, 3.0 * (truth_sd or 0.0))
    ok_truth = (truth_a is not None and abs(truth_a - 0.14) <= tol
                and truth_b is not None and abs(truth_b - ab0) <= 0.05
                and truth_c2 is True)
    gates.append(Gate(
        VALIDITY, "S1 recovers a planted world",
        "the fit is UNBIASED: it returns what it was planted with, to within "
        "its own spread",
        f"|alpha_a - 0.140| <= max(0.030, 3 sd), |alpha_b - {ab0:.3f}| <= "
        "0.050, and C2 PASSES, in the TRUTH world",
        ok_truth,
        ("alpha_a not fitted" if truth_a is None
         else f"alpha_a = {truth_a:.4f} +/- "
              + ("n/a" if truth_sd is None else f"{truth_sd:.4f}")
              + f" (tolerance {tol:.4f}), alpha_b = {truth_b:.4f}, "
                f"C2 = {truth_c2}"),
        "everything: an estimator that cannot recover a world it was handed "
        "cannot be read on a world it was not",
        ["Scored against the estimator's OWN spread, not a fixed number, "
         "because bias and precision are different failures and S4 scores the "
         "second one."]))
    discriminates = (verdicts["MISSING"][4] is False
                     and verdicts["BN-DRIFT"][4] is False
                     and verdicts["TRUTH"][4] is True)
    gates.append(Gate(
        VALIDITY, "S2 the residual gate discriminates",
        "C2 FAILS in the two worlds with a missing term and PASSES in the one "
        "without",
        "C2 = PASS in TRUTH, FAIL in MISSING and in BN-DRIFT",
        discriminates,
        ", ".join(f"{k}: C2={v[4]}" for k, v in verdicts.items()),
        "C2 itself: a gate that answers the same in every world settles "
        "nothing, and C2 is the reason this experiment exists"))
    a_no, _, _, _, c2_no, _, sd_no, _ = verdicts["NO-A"]
    tol_no = max(0.03, 3.0 * (sd_no or 0.0))
    gates.append(Gate(
        VALIDITY, "S3 a wrong parameter is not a wrong model",
        "with alpha_a planted at zero the fit says zero, and C2 still calls "
        "the model complete",
        "|alpha_a - 0.000| <= max(0.030, 3 sd) and C2 = PASS in the NO-A world",
        a_no is not None and abs(a_no) <= tol_no and c2_no is True,
        "alpha_a = " + ("n/a" if a_no is None else f"{a_no:.4f}")
        + " +/- " + ("n/a" if sd_no is None else f"{sd_no:.4f}")
        + f" (tolerance {tol_no:.4f}), C2 = {c2_no}",
        "the separation between the parameter and the model: a world with no "
        "activation re-read is still a world the three terms DESCRIBE, and a "
        "residual gate that failed there would be failing on a parameter value"))
    ceiling_ok = truth_sd is not None and truth_sd <= ALPHA_A_SD_CEILING
    floor, _ = read_cross_arm_floor()
    gates.append(Gate(
        VALIDITY, "S4 the design resolves alpha_a",
        "at the PINNED settings, alpha_a's spread beats the two-point reading "
        "this fit replaces, which is C6's bar and not the band's width",
        f"sd(alpha_a) <= {ALPHA_A_SD_CEILING:.3f} in the TRUTH world",
        ceiling_ok,
        "no spread" if truth_sd is None else f"sd = {truth_sd:.4f} at "
        f"GROUP_SIZE_M={args.group_m}, {args.reps} reps, planted spread "
        f"{noise:.2%} ({noise_source})",
        "C1 on the real run: the same settings will produce the same spread, "
        "so a FAIL here says the POD RUN CANNOT ANSWER P1 and should be "
        "re-pinned before it is paid for",
        [mde_line(truth_sd, what="alpha_a, planted TRUTH world",
                  assumption=(f"lognormal spread {noise:.2%} on every timing, "
                              f"from {noise_source}; {args.draws} draws"),
                  floor=floor),
         "THE LEVER IS THE SWIZZLE, and the arithmetic says why. The response "
         "moves with alpha_a as g1 (1 - alpha_b)/(1 + phi)^2, so the whole "
         "design's power is proportional to (1 - alpha_b): at "
         f"GROUP_SIZE_M={args.group_m} the corpus puts alpha_b at "
         f"{ab0:.2f}, so the lever is "
         f"{(1 - ab0) / (1 - planted_alpha_b(16)):.0%} of its size at "
         "GROUP_SIZE_M=16. --reps buys some of the rest, and this gate is what "
         "says how much: no number here is quoted from a previous run.",
         "If this gate FAILS, re-run --dry-run --group-m 16 before the pod."]))
    gates.append(Gate(
        VALIDITY, "S5 C2 can still fail here",
        "the residual gate has POWER at this pinning: the missing-term world "
        "does not pass it",
        "C2 = FAIL in the MISSING world at this GROUP_SIZE_M and this spread",
        verdicts["MISSING"][4] is False,
        f"MISSING C2 = {verdicts['MISSING'][4]} at chi2 "
        + ("n/a" if verdicts["MISSING"][2] is None
           else f"{verdicts['MISSING'][2]:.2f}")
        + f" against a ceiling of {RESIDUAL_CHI2_CEILING:.1f}; TRUTH C2 = "
        + f"{verdicts['TRUTH'][4]} at chi2 "
        + ("n/a" if verdicts["TRUTH"][2] is None
           else f"{verdicts['TRUTH'][2]:.2f}"),
        "C2 on the real run: this is the same probe the run performs on itself "
        "before scoring C2, so a FAIL here says the real run's C2 will read "
        "UNKNOWN and the arm cannot establish model completeness",
        ["S2 asks whether C2 answers DIFFERENTLY across the four worlds and can "
         "be satisfied by a C2 that fails everywhere. This asks the one thing "
         "the pod arm's headline depends on: that a PASS could have been a "
         "FAIL. It is stated separately because the audit found the driver "
         "self-testing at GROUP_SIZE_M=16, where both hold, and scheduling the "
         "arm at 1, where this one does not."]))
    return lines, gates


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b",
                    choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 puts the whole "
                         "rows-per-expert range inside a reachable token count, "
                         "and its G=1 BLOCK_M=64 alpha is measured on BOTH "
                         "cards, which is the point the predictions are anchored "
                         "to")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="not fp8: halving the weight bytes doubles the AI cap "
                         "and moves every cell out of the regime the published "
                         "alphas were measured in")
    ap.add_argument("--block-n-list", default=",".join(
        str(v) for v in DEFAULT_BLOCK_N),
        help="THE SWEEP. Powers of two only (tl.arange), 16 is the floor "
             "tl.dot accepts, and 256 is excluded by arithmetic rather than "
             "preference: BLOCK_M=256 x BLOCK_N=256 needs 256 accumulator "
             "registers per thread against a hardware maximum of 255, so that "
             "arm can have no compute reference on any card. Ask for it and "
             "the resource bill refuses it with the number")
    ap.add_argument("--tiles", default=",".join(str(v) for v in SUBJECT_BLOCK_M),
                    help="subject block sizes. 128 is the primary and the only "
                         "one production runs multi-tile as a regime (66 of 87 "
                         "cells on the padded count; 16 and 64 fire in "
                         "isolated cells, 32 never); 64 and 32 are where alpha "
                         "is robustly identifiable and are what makes the "
                         "alpha_b invariance a test; 16 was added on "
                         "2026-09-10 because the 128 cells came back blank at "
                         "every BLOCK_N and a two-height fit cannot separate "
                         "the model's activation column from the 1/BN rival "
                         "that beats it. The plan's identification line "
                         "prints the correlation with and without it")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="largest rows per expert. 1024 is 4 treads at "
                         "BLOCK_M=256, which is the reference's whole ladder")
    ap.add_argument("--max-treads", type=int, default=8,
                    help="treads per ladder, counted from n=1. Caps the cost "
                         "at the small block sizes, where --r-max alone would "
                         "buy 32 treads of a ladder that is already straight")
    ap.add_argument("--reps", type=int, default=17,
                    help="round-robin passes per setting. Two is the minimum "
                         "for any spread at all and every interval here is a "
                         "resample of these, but 17 is a POWER choice, not a "
                         "caution: planted at the published 0.8%% across-repeat "
                         "spread, alpha_a's own spread runs 0.126 at 5 reps, "
                         "0.086 at 9 and 0.0097 at 17, against the 0.025 C1 "
                         "needs. The drop is not smooth because membership "
                         "decisions stop flipping between draws. Run "
                         "--self-test --plant-noise <your spread> to redo that "
                         "calculation for a noisier pod")
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"],
                    help="the swizzle width, pinned across the whole sweep. 1 "
                         "is what vLLM's fallback ladder runs, so it is the "
                         "production setting -- and it is NOT neutral: alpha_b "
                         "is a property of the swizzle, the study measures "
                         "0.92-1.02 at 1 against 0.58-0.62 at 8 and above, and "
                         "C4's comparison with TEMPO is a comparison at THIS "
                         "value")
    ap.add_argument("--block-k", type=int, default=SWEEP.FIXED["BLOCK_SIZE_K"])
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"],
                    help="pipeline depth, applied to every setting. It is in "
                         "the shared-memory bill: BLOCK_M=128 x BLOCK_N=256 at "
                         "4 stages asks 192 KiB, which an A100 refuses and an "
                         "H200 allows")
    ap.add_argument("--num-warps", type=int, default=SWEEP.FIXED["num_warps"],
                    help="warps per CTA, applied to every setting. It is the "
                         "denominator of the accumulator register bill, so "
                         "raising it would let BLOCK_N=256 run -- and would "
                         "unpin the achieved compute rate the references are "
                         "compared across")
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob on 2026-09-02 and kept only "
                         "in the run id. moe.bench.timing.time_kernel sizes the "
                         "iteration count per cell from --cell-budget-ms and "
                         "the warmup's own queue-deep per-call time. It stays "
                         "in the id because cells measured under the retired "
                         "loop exist on disk and a directory must not be "
                         "resumed into across that change")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED 2026-09-02: a 1 ms "
                         "kernel needs hundreds of calls before the governor "
                         "reacts and a 30 ms one needs a few, so ladders warmed "
                         "at a fixed COUNT were compared at different clock "
                         "states. The instrument warms for a duration measured "
                         "with the same events the trials use")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per timing; the percentiles are "
                         "over iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the compute roof every alpha here is "
                         "scored against was measured flushed, and a warm-L2 "
                         "cell is not comparable with it. Recorded per row and "
                         "in the run id, so a flushed and an unflushed sweep "
                         "can never share a directory")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its own iteration count from it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS,
                    help="bootstrap resamples behind every interval and behind "
                         "C2's sigma")
    ap.add_argument("--sm-count", type=int, default=0,
                    help="0 asks the driver; only needed off GPU")
    ap.add_argument("--capability", default="",
                    help="compute capability MAJOR.MINOR, e.g. 8.0 for the "
                         "A100 or 9.0 for the H200. Empty asks the driver; "
                         "give it off GPU to get the shared-memory verdicts. "
                         "The register check needs no device and runs either way")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="Op/B. 0 reads the ATTACHED device's own calibration "
                         "and REFUSES when there is none")
    ap.add_argument("--ridge-band", default="",
                    help="LO,HI in Op/B, only meaningful with --ridge")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0,
                    help="0 reads this machine's calibration")
    ap.add_argument("--plant-noise", type=float, default=None,
                    help="lognormal sigma on every planted timing, used by "
                         "--self-test, by --dry-run's design-power line and by "
                         "the C2 power probe the real run performs on itself. "
                         "DEFAULT IS RESOLVED, NOT FIXED: this run's own "
                         "measured across-repeat spread when it has one, else "
                         "the WORST published spread "
                         f"({PLANT_NOISE_FALLBACK * 100:.2f}%%). "
                         "It used to default to 0.008, the middle of the "
                         "published range, and S4 passes at 0.008 and fails at "
                         "0.015-0.020, so that default decided whether the "
                         "experiment looked worth paying for by describing a "
                         "pod quieter than half the corpus")
    ap.add_argument("--power-seeds", type=int, default=DESIGN_POWER_SEEDS,
                    help="planted realisations behind the design-power line, "
                         "counted up from --seed. ONE IS NOT ENOUGH AND THAT "
                         "IS MEASURED: at the fallback spread the sd this line "
                         "reports ran 0.0223 to 0.4171 over six consecutive "
                         "seeds of the three-height design, and seed 0 was the "
                         "only one that cleared the bar. The median is printed "
                         "and the WORST decides the verdict. It stays out of "
                         "the run id for the same reason --power-draws does: "
                         "it re-analyses a design rather than changing a timing")
    ap.add_argument("--power-draws", type=int, default=POWER_PROBE_DRAWS,
                    help="bootstrap draws behind the design-power line and the "
                         "C2 power probe, as distinct from --draws behind the "
                         "report's own intervals. Fewer, because one sd is "
                         "read off them and a bootstrap sd's own relative "
                         "error is about 1/sqrt(2 draws)")
    ap.add_argument("--overhead-ms", type=float, default=0.15,
                    help="the fused layer's fixed cost, used ONLY to plant "
                         "self-test worlds and to price the run. The measured "
                         "run fits it as `delta` instead of assuming it")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--card", default="",
                    help="card slug the run id is built from. Read from the "
                         "attached device by default and REFUSED if it "
                         "contradicts one; its only use is printing a pod's "
                         "real path from a laptop")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--require-git-visible", action="store_true",
                    help="refuse to run when the output path is git-ignored")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the resource bill, the predictions "
                         "and the cost, then stop")
    ap.add_argument("--self-test", action="store_true",
                    help="plant four worlds and check the gates tell them "
                         "apart, off GPU")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit 1 CLAIM_FAIL unless every gate passes. Off by "
                         "default: C4 and C6 are both predicted to FAIL at "
                         "GROUP_SIZE_M=1, and a falsified prediction is a "
                         "result, not an error. It never softens 3 INVALID")
    return ap


def _hypothesis_ceiling(dtype: str) -> tuple[float, str]:
    """A ceiling for planning only, labelled so it can never pass for measured."""
    return (712.259 if dtype == "bf16" else 712.259,
            "HYPOTHESIS: the 2026-09-01 H200 bf16 calibration in this repo, "
            "which belongs to no attached device")


def _exit_over(gates: list[Gate], args) -> int:
    """The exit code the gates imply, through `moe.bench.exit_codes.classify`.

    VALIDITY before CLAIM, UNKNOWN counting against both, and an empty list
    refused rather than reported as DONE -- all of that is the shared table's,
    not this file's. Two integers meaning two different things in two files is
    the defect that module is named against, and this file used to return 1 for
    "some gate did not pass" and 2 for "refused", which collided with the
    driver's own reading of 2.

    `--fail-on-gate` softens ONE case and says so out loud: without it a
    CLAIM_FAIL is reported as DONE, because a claim gate that failed is a
    RESULT and not a broken run, and C4 is PREDICTED to fail at
    GROUP_SIZE_M=1. It never softens INVALID: a VALIDITY gate that did not pass
    means nothing on the page may be quoted, whatever the operator asked for.
    """
    rc = exit_codes.classify(g.scored() for g in gates)
    if rc == exit_codes.CLAIM_FAIL and not args.fail_on_gate:
        print(f"exit     {exit_codes.describe(exit_codes.CLAIM_FAIL)}")
        print(f"         reported as exit {exit_codes.DONE} without "
              "--fail-on-gate: a claim that did not pass is a RESULT. Pass "
              f"--fail-on-gate to return {exit_codes.CLAIM_FAIL} CLAIM_FAIL "
              "instead.")
        return exit_codes.DONE
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def _main(argv=None) -> int:                                    # noqa: C901
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    synthetic = bool(args.dry_run or args.self_test)
    subjects = tuple(int(v) for v in args.tiles.split(","))
    if REFERENCE_BLOCK_M in subjects:
        print(f"REFUSED: BLOCK_M={REFERENCE_BLOCK_M} is the compute reference "
              "and cannot also be a subject. A ladder used as its own compute "
              "branch has no memory branch by assumption and yields no alpha.")
        return exit_codes.REFUSED

    # THE PRE-REGISTERED BAND IS CHECKED AGAINST THE FILES IT CAME FROM BEFORE
    # ANYTHING ELSE HAPPENS. The band this replaced cited four A100 slopes that
    # exist in no file, and nothing in the code could have noticed. A run whose
    # C1 would be scored against a band that no longer reads back refuses here,
    # at no cost, rather than on the pod.
    try:
        _, band_lines = check_alpha_a_band()
        floor = published_prior_sd()
    except CorpusMissing as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED

    capability = SWEEP.parse_capability(args.capability)
    if capability is None and not synthetic:
        try:
            import torch
            capability = tuple(torch.cuda.get_device_capability(0))
        except Exception:                                      # noqa: BLE001
            capability = None

    # RESOLVED BEFORE ANY GPU TIME, so a run with no ridge it may quote costs
    # nothing and says why. `resolve_ridge` reads the ATTACHED device's own
    # calibration and refuses when there is none; the module constant it used
    # to fall back to put a stale H200 band into seven published A100 reports.
    try:
        rr = SWEEP.resolve_ridge(args, synthetic=synthetic)
    except SWEEP.RidgeUnavailable as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    bandwidth, bw_source = args.bandwidth_gbps, "given on the command line"
    ceiling, ceiling_source = 0.0, ""
    try:
        from moe.bench.roofline import load_measured
        hw = load_measured()
    except Exception as exc:                                   # noqa: BLE001
        hw, hw_note = None, str(exc)
    else:
        hw_note = ""
    if hw is not None:
        if not bandwidth:
            bandwidth = hw.bandwidth_bytes_s / 1e9
            bw_source = f"this machine's calibration ({hw.name})"
        ceiling = hw.peak(args.dtype) / 1e12
        ceiling_source = f"measured on {hw.name}"
    else:
        ceiling, ceiling_source = _hypothesis_ceiling(args.dtype)
        if hw_note:
            ceiling_source += f" (calibration unreadable: {hw_note})"
        if not bandwidth:
            bandwidth, bw_source = 4374.5, (
                "HYPOTHESIS: the published H200 triad ceiling, no calibration "
                "on this box")

    detected = detect_card_slug()
    card = args.card or detected or NO_CARD_SLUG
    if args.card and detected and args.card != detected:
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return exit_codes.REFUSED
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bn_decomposition" / run_id
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    # THE SPREAD EVERY PLANTED WORLD IS GENERATED AT, resolved once, before the
    # plan is priced and before any gate is scored, and printed with its
    # provenance wherever it is used.
    #
    # THE ORDER IS THE FIX. `--plant-noise` used to be an argparse default of
    # 0.008 -- the middle of the published range -- and S4 passes at 0.008 and
    # fails at 0.015-0.020, so that one number decided whether the experiment
    # looked worth paying for, while describing a pod quieter than half the
    # corpus. It now comes from THIS run's own cells when there are any on disk
    # (a resume, or a replay of a pod directory pointed at with --out), and
    # otherwise from the WORST published spread, which is the conservative end.
    _, prior_samples = read_samples(csv_path)
    _, _, prior_spreads = arm_alphas(
        prior_samples, cfg, block_ns=tuple(int(v) for v in
                                           args.block_n_list.split(",")),
        subjects=subjects, ridge=rr.ridge, bandwidth_gbps=bandwidth,
        dtype=args.dtype,
        base_pinned=dict(SWEEP.FIXED, num_stages=args.num_stages,
                         num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                         BLOCK_SIZE_K=args.block_k),
        capability=capability, ceiling_tflops=ceiling,
        sm_count=args.sm_count or 132) if prior_samples else ([], [], {})
    on_disk, on_disk_source = measured_spread(prior_spreads)
    noise, noise_source = resolve_plant_noise(args.plant_noise, on_disk,
                                              on_disk_source)

    # BUILT TWICE, AND CHEAPLY: the design-power probe needs to know which BN
    # arms survive the resource bill before it can plant a world on them, and
    # the plan the report prints carries that probe's answer. Both passes are
    # pure arithmetic on the pinned constants and neither touches a device.
    grid_only = build_plan(args, cfg, b, capability, rr.ridge, bandwidth)
    block_ns = grid_only.block_ns
    treads = {bm: len(rs) for (bn, bm), rs in grid_only.rows.items()}

    # THE DESIGN-POWER LINE IS COMPUTED HERE, on a planted TRUTH world at this
    # pinning and this noise, and it is what replaces the literal string
    # "sd 0.11-0.13 ... at any rep count tried" that this file printed while the
    # code produced 0.176 at the same settings.
    power = (design_power(cfg, args, b=b, ceiling_tflops=ceiling,
                          capability=capability, block_ns=block_ns,
                          subjects=subjects, sm_count=args.sm_count or 132,
                          noise=noise, noise_source=noise_source,
                          draws=args.power_draws)
             if block_ns else None)
    plan = build_plan(args, cfg, b, capability, rr.ridge, bandwidth,
                      power=power, floor=floor)

    lines = [
        "experiment  bn_decomposition: separate alpha_a from alpha_b, and test "
        "whether the model is complete", "",
        predictions_text(cfg, b, rr.ridge, rr.source, block_ns, subjects,
                         treads, args.group_m, band_lines), "",
        "## The plan", ""]
    lines += plan.lines(cfg)
    lines += [
        f"ridge        {rr.ridge:.3f} Op/B, {rr.source}",
        f"ridge band   {rr.band[0]:.2f}-{rr.band[1]:.2f}, {rr.band_source}",
        f"bandwidth    {bandwidth:.1f} GB/s, {bw_source}",
        f"ceiling      {ceiling:.1f} TFLOP/s {args.dtype}, {ceiling_source}",
        "instrument   " + (SWEEP.timing_basis() or
                           "NOT NAMEABLE on this box (no importable torch), "
                           "which is also the case in which nothing here "
                           "measures anything"),
        f"card         {card}" + ("" if detected else
                                  "  (NO DEVICE: this id is the 'nocard' one "
                                  "and is not what a pod derives; --card "
                                  "<slug> prints that)"),
        "",
        "TILE RESOURCE BILL, one CTA, at "
        + (f"sm_{capability[0]}{capability[1]}" if capability
           else "an UNKNOWN device (--capability MAJOR.MINOR gives the "
                "shared-memory verdict; the register check runs regardless)")]

    base_pinned = plan.base_pinned
    for bn in tuple(int(v) for v in args.block_n_list.split(",")):
        for bm in (*subjects, REFERENCE_BLOCK_M):
            res = SWEEP.tile_resources(dict(base_pinned, BLOCK_SIZE_N=bn), bm,
                                       b, capability)
            lines.append(f"  BN={bn:4d}" + res.render())
    lines += [""] + plan.warp_lines()
    lines += [
        "",
        "WEIGHT-STREAM SLOPE  " + WEIGHT_STREAM_SOURCE,
        "             one complete stream of this layer's expert weights is "
        + (f"{WEIGHT_STREAM_MS(cfg, args.dtype, bandwidth):.4f} ms at "
           f"{bandwidth:.1f} GB/s" if bandwidth > 0 else "UNAVAILABLE: no "
           "bandwidth to stream at")
        + ". Every cell is scored on w = B / that, beside its alpha.",
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per tread per repeat, flushed), CARD, "
        "report.txt, report.json, triton-cache/"]

    # THE DESIGN CHECK, BEFORE ANY GPU TIME AND BEFORE THE SELF TEST. Fewer
    # than three surviving BN arms is not a degraded run, it is a run that
    # cannot answer its own question: two points and two unknowns fit exactly
    # and leave no residual, so C1 has no interval and C2 has nothing to test.
    # The usual cause is the shared-memory bill at --num-stages 4 on an A100,
    # where the BLOCK_M=256 reference at BLOCK_N=128 asks 192 KiB against 163;
    # at 3 stages it asks 144 and the arm comes back.
    underpowered = ""
    if len(block_ns) < MIN_BN_POINTS:
        underpowered = (
            f"only {len(block_ns)} BN arm(s) survive the resource bill, "
            f"against the {MIN_BN_POINTS} this experiment needs. alpha_a would "
            "be a two-point slope again, which is the thing this run exists to "
            "replace.\n    Try --num-stages "
            f"{max(1, args.num_stages - 1)}: the bill above is linear in "
            "stages, and one stage fewer is what fits the BLOCK_M="
            f"{REFERENCE_BLOCK_M} reference on an A100. Every setting moves "
            "together, so the sweep stays pinned.")
        lines += ["", f"REFUSED: {underpowered}"]

    if args.dry_run:
        print("\n".join(lines))
        return exit_codes.DONE if not underpowered else exit_codes.REFUSED

    if args.self_test:
        more, gates = self_test(args, cfg, b, rr.ridge, bandwidth, ceiling,
                                capability, block_ns, subjects, noise=noise,
                                noise_source=noise_source)
        print("\n".join(lines + more + ["", "## Gates", ""]
                        + render_gates(gates)))
        return _exit_over(gates, args)

    if underpowered:
        print("\n".join(lines))
        return exit_codes.REFUSED

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        print(f"\n{missing.split('.')[0]}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --self-test  four planted worlds, checking the gates "
              "discriminate\n"
              "  --dry-run    the plan, the resource bill and the cost")
        return exit_codes.REFUSED

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return exit_codes.REFUSED
    if not block_ns:
        print("\n".join(lines))
        print("\nREFUSED: every BN arm was dropped by the resource bill. "
              "Nothing to measure.")
        return exit_codes.REFUSED
    if hw is None:
        print("\n".join(lines))
        print("\nREFUSED: no calibration for the attached device. V2 scores "
              "every compute reference against THIS card's measured peak, and "
              "there is nothing here to accept or refuse one with. Run "
              "scripts/calibrate_hardware.py first.")
        return exit_codes.REFUSED

    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    print("\n".join(lines))

    # THE RESUME GUARD, belt as well as braces. The card is already in the run
    # id, so another card lands in another directory and cannot normally reach
    # this cells.csv. This catches the ways it could anyway: an explicit --out
    # or --run-id aiming two cards at one place, or a directory copied between
    # pods.
    if csv_path.exists():
        written_by = card_path.read_text().strip() if card_path.exists() else ""
        if written_by != card:
            print(f"REFUSED to resume {csv_path}: written by card "
                  f"{written_by or '<unrecorded>'!r} and this run is {card!r}. "
                  "Resuming would report one card's treads against the other's "
                  "ridge. Move or delete that directory deliberately.")
            return exit_codes.REFUSED
    card_path.write_text(card + "\n")

    sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
    done, samples = read_samples(csv_path)
    compiles: dict[tuple[int, int], int] = {}
    executed: dict[tuple[int, int], int] = {}

    # THE ROW-LEVEL PROVENANCE, BUILT ONCE. A mid-run commit must not give two
    # rows two shas, and `iters` is None here on purpose: each row carries the
    # count the instrument sized for that cell, and a run-wide number in the
    # block would contradict most of them. The report's own copy is stamped
    # after the sweep with the median the cells actually used.
    row_prov = PV.provenance_block(
        instrument=SWEEP.timing_basis(), ridge=rr.ridge, ridge_source=rr.source,
        bandwidth=bandwidth, bandwidth_source=bw_source,
        warmup_ms=args.warmup, target_ms=args.cell_budget_ms, iters=None)

    # THE CLOCK THE ROOF WAS MEASURED AT, resolved once before any cell so the
    # whole arm is scored against one number and a mid-sweep yaml rewrite cannot
    # move it. Without it every cell's LEVEL verdict is None, which means "not
    # determined" and excludes nothing; a guessed reference would exclude real
    # treads.
    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every cell's clock LEVEL "
                  "verdict will be None and no cell can be excluded for it"))
    instrument_kw = dict(prov=row_prov,
                         reference_clock_mhz=reference_clock)
    started = time.time()
    # EVERY REFERENCE FIRST, ACROSS ALL ARMS, AND THEN THE SUBJECTS. Two
    # reasons, and the second is why it is not merely tidy:
    #
    #  * a reference 43.6x too slow classifies every subject tread in its arm,
    #    so measuring subjects before the reference is qualified spends the
    #    expensive half of the run on cells that cannot be read;
    #  * an arm that qualifies NO branch of its own can still borrow one -- see
    #    `import_reference` -- and whether it can is not knowable until the
    #    other arms have been measured. Arm-at-a-time would refuse the BN=32
    #    subjects before the BN=64 arm existed to lend to them, and BN=32 is the
    #    widest point in BM/BN this grid has.
    #
    # The references are 4 treads each and the cheapest thing in the run, so
    # the ordering costs nothing and buys both.
    early: dict[int, RefVerdict] = {}
    for bn in block_ns:
        pinned = dict(base_pinned, BLOCK_SIZE_N=bn)
        print(f"\n-- BN={bn}: reference ladder, BLOCK_M={REFERENCE_BLOCK_M} --")
        c, e = measure_setting(args, cfg, bn, REFERENCE_BLOCK_M,
                               plan.rows[(bn, REFERENCE_BLOCK_M)], csv_path,
                               cache_root, pinned, done, samples,
                               **instrument_kw)
        compiles[(bn, REFERENCE_BLOCK_M)], executed[(bn, REFERENCE_BLOCK_M)] = c, e
        pts, _ = collapse(samples, bn, REFERENCE_BLOCK_M)
        early_cells = [SWEEP.make_cell(cfg, n * REFERENCE_BLOCK_M,
                                       REFERENCE_BLOCK_M, ms,
                                       sm_count=sm_count, block_n=bn)
                       for n, ms in pts]
        # THE SWEPT BLOCK SIZES, THE SECOND CALL SITE OF THE SAME RULE.
        # `(REFERENCE_BLOCK_M,)` here scaled the non-vacuity check to
        # BLOCK_M=256 itself, which no kernel can pass; see `arm_alphas`.
        early[bn] = qualify_reference(
            early_cells, (*subjects, REFERENCE_BLOCK_M), bn, cfg=cfg,
            ridge=rr.ridge,
            bandwidth_gbps=bandwidth, b=b, pinned=pinned,
            capability=capability, ceiling_tflops=ceiling, subjects=subjects,
            load_clock_mhz=reference_load_clock(samples, bn),
            reference_clock_mhz=reference_clock)
        for line in early[bn].render():
            print(line)

    lenders = [v for v in early.values() if v.ok]
    for bn in list(early):
        if not early[bn].ok:
            early[bn] = import_reference(early[bn], lenders, cfg, None)
            print(f"\nBN={bn}: " + ("branch IMPORTED, subjects will be "
                                    "measured -- " + early[bn].import_note
                                    if early[bn].imported else
                                    "no branch of its own and none to import"))
    usable_ns = [bn for bn in block_ns if early[bn].ok]
    if len(usable_ns) < MIN_BN_POINTS:
        # THE TEXT SAID THE OPPOSITE OF THE LOOP BELOW UNTIL 2026-09-09.
        # It read "the subjects are measured anyway" while the next statement
        # skipped every arm without a branch, and the 2026-09-09 log printed
        # both three lines apart.
        print(f"\nWARNING: only {len(usable_ns)} of {len(block_ns)} arms have "
              f"a compute branch, against the {MIN_BN_POINTS} V5 needs. The "
              "subjects of the arms WITHOUT one are SKIPPED below (an alpha "
              "against no reference is a blank that looks like a measurement), "
              "so alpha_a will be UNIDENTIFIED, V5 will FAIL and every claim "
              "gate will read UNKNOWN. This is the outcome P5 registered as "
              "possible.")

    for bn in block_ns:
        pinned = dict(base_pinned, BLOCK_SIZE_N=bn)
        if not early[bn].ok:
            print(f"\n-- BN={bn}: SKIPPING the subjects. No compute branch, "
                  "own or imported, so every alpha in this arm would be a "
                  "blank that looks like a measurement. The arm contributes a "
                  "refusal, which is a result, and no GPU time.")
            continue
        for bm in subjects:
            if (bn, bm) not in plan.rows:
                continue
            print(f"\n-- BN={bn}: subject ladder, BLOCK_M={bm} --")
            c, e = measure_setting(args, cfg, bn, bm, plan.rows[(bn, bm)],
                                   csv_path, cache_root, pinned, done,
                                   samples, **instrument_kw)
            compiles[(bn, bm)], executed[(bn, bm)] = c, e
    print(f"\nmeasured in {time.time() - started:.0f} s")

    # `probe_c2_power=True` EXACTLY HERE AND NOWHERE ELSE. The measured run is
    # the only thing that has a measured spread to plant at, and C2 is the gate
    # the arm's headline rests on. See `gate_residual`.
    more, gates, payload = analyse_run(
        samples, cfg, args, ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
        ceiling_tflops=ceiling, ceiling_source=ceiling_source,
        capability=capability, base_pinned=base_pinned, compiles=compiles,
        executed=executed, sm_count=sm_count, block_ns=block_ns,
        subjects=subjects, probe_c2_power=True, plant_noise=args.plant_noise,
        band_lines=band_lines, reference_clock_mhz=reference_clock)
    payload["gpu"] = torch.cuda.get_device_name(0)
    payload["run_id"] = run_id
    payload["ridge_source"] = rr.source
    payload["alpha_a_band_provenance"] = band_lines
    # THE ITERATION COUNT IN THE BLOCK IS THE INSTRUMENT'S, NOT THE KNOB'S:
    # `--iters` is retired and `time_kernel` sizes each cell from the budget,
    # so recording the argparse default would contradict every row of the CSV.
    iters_seen = [s.iters for s in samples
                  if s.status == "ok" and s.iters > 0]
    payload = PV.provenance_block(
        instrument=SWEEP.timing_basis(),
        ridge=rr.ridge, ridge_source=rr.source,
        bandwidth=bandwidth, bandwidth_source=bw_source,
        warmup_ms=args.warmup, target_ms=args.cell_budget_ms,
        iters=int(statistics.median(iters_seen)) if iters_seen else None,
    ).stamp(payload)
    text = "\n".join(lines + more + ["", "## Gates", ""] + render_gates(gates))
    print("\n".join(more + ["", "## Gates", ""] + render_gates(gates)))
    (out_dir / "report.txt").write_text(text)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path), ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return _exit_over(gates, args)


def main(argv=None) -> int:
    """Convert a string SystemExit into exit code 2, which is what REFUSED means.

    `raise SystemExit("some sentence")` exits ONE. Every refusal in this file
    was written that way, so a run that refused before measuring anything --
    no calibration for the attached device, an import that drifted, a tile that
    cannot run as pinned -- exited with the same code a run that MEASURED and
    then failed a claim gate would have. The session driver could not tell them
    apart, and this script's own contract says 2 means refused and 1 is reserved
    for --fail-on-gate. Found in review on 2026-09-02, live, on this laptop.

    Caught here rather than at twenty raise sites so the contract holds for a
    caller of main() as well as for the CLI, and so a new refusal added later
    cannot reintroduce the bug by forgetting the code.

    AN UNPLANNED CRASH IS ERROR (4), and it is the other half of the same
    contract. Left to propagate, an unexpected exception exits the interpreter
    ONE as well, so a torch OOM and a refuted claim arrived at the driver as
    one number: FINISHED_CODES holds 1, so the arm is filed as finished,
    skipped on every resume, RETRY_ARMS stays 0 and the session exits 0 over an
    arm that never measured. ERROR (4) is outside FINISHED_CODES precisely so
    "the apparatus broke" can be told from "the claim did not hold". The
    traceback is printed first and not swallowed, because a code without one
    tells an operator nothing about what to fix.
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = exc.code if exc.code.startswith("REFUSED") else f"REFUSED: {exc.code}"
            print(msg, file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: bn_decomposition crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim "
              "failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
