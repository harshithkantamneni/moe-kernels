#!/usr/bin/env python
"""Does the AI cap formula hold at BLOCK_SIZE_M=16? A FORMULA test, not a claim.

    python scripts/tile_cap_test.py --dry-run           # plan, predictions, cost. No GPU
    python scripts/tile_cap_test.py --self-test 0.558   # the whole analysis, off GPU
    python scripts/tile_cap_test.py --self-test 0.10    # the same analysis, retracted world
    python scripts/tile_cap_test.py                     # the pod run, ~80 s of H200
    python scripts/tile_cap_test.py --control 128 --num-stages 3   # A100-safe control

WHAT THIS TESTS. `cap = 2 BM / (alpha b)` -- the ceiling an M-tile height puts on
arithmetic intensity -- on the one ladder in this study where `alpha` is cleanly
identifiable. A tile that is memory bound at EVERY tread has no compute branch
for the fit to run into and imports no alpha from another block size, and
BLOCK_M=16 is the extreme case of that. Across all 26 published reports the
FORCED block sizes are 32, 64, 128 and 256 and nothing else, so the formula has
never been exercised at the bottom of its range, and that is what this buys. The
experiment is worth the 80 seconds.

WHAT IT DOES NOT CARRY, AND AN EARLIER VERSION OF THIS HEADER SAID IT DID. The
production claim. It is BLOCK_M=128 that carries that, and the demotion is not a
judgement call -- it is the observed-tile data.

THE OBSERVED-TILE DATA. `results/published/2026-09-01-nvidia_h200-alpha-0558/
merged.csv` is the one published arm that RECORDS the tile vLLM chose
(`tile_block_m`, with `tile_config_source` in {vllm_default, vllm_tuned}), 132
distinct cells, rows per expert read as `load_mean_rows`:

    BLOCK_M    cells run multi-tile     max M-tiles per expert
       16          0 of 24                      1
       32          0 of  5                      1
       64          0 of 16                      1
      128         59 of 87                     32

Read instead at the BUSIEST expert (`load_max_rows`) the only lines that move are
16, where 1 of the 24 cells reaches 2 tiles, and 128, which goes to 65 of 87 and
33 tiles. Both readings say the same thing about the tile under test.

WHY THAT RETIRES THE PRODUCTION FRAMING. The re-read term is `Q(n) = 1 + alpha
(n - 1)`, which is exactly 1 at n = 1, so it only exists above one tile per
expert. vLLM does pick 16 at small batch -- `get_default_config` returns it for
M <= 32, and 24 of the 132 cells above took that ladder -- but at every batch
where it picks 16 the expert holds ONE tile. The cap at 16 is therefore REAL and
NEVER APPROACHED, and a cap that binds nothing is a fact about the formula, not
about a shipped kernel. The same holds at 32 and 64.

BLOCK_M=128 IS THE PRODUCTION REGIME, and `scripts/bm128_depth.py` is where its
claim lives. It is the only tile height vLLM runs multi-tile at all, up to 32
M-tiles per expert, and the only one whose cap and the ridge are close enough for
the answer to be in doubt: that file computes cap 150.4 against a calibrated
ridge of 145.8 on the A100 and 158.6 against 162.8 on the H200. Whether the cap
matters in production is settled there, not here.

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
  * GATE 3 asks whether the fitted re-read fraction exceeds 0.33, and on a
    one-tile run it goes UNDECIDED with "No ladder had two memory-bound treads,
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
160.0 against a ridge of 160.3 -- so `crossing_rows` is None, the horizon
collapses to `2 x 0`, and every depth clears it. The gate that decides whether
an absence is bracketed would pass after a single tread. `retracted_horizon_
tiles` below asks the question that threshold is actually about instead.

WHICH CONTROL, AND WHY 256. At alpha=0.558 only 128 and 256 have an AI cap above
the ridge, so only they can cross and only they can be a control. 256's crossing
lands in tread 1, where `Q = 1`, at BOTH ends of the ridge band AND under the
retracted alpha -- r = 160.3 and 176.2 rows per expert, both inside one 256-row
tile. Its crossing therefore does not depend on the parameter under test. 128's
lands in tread 2 at ridge 160.3 and tread 3 at 176.2, moving with alpha, which
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
     against a ridge of 160.3 to 176.2, so that world does eventually trip
     both, and the depths at which it does are:

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

WHY THE MEASURED CAP IS BIASED IN THE SAFE DIRECTION. `LadderFit.alpha` is
`B/(A+B)` fitted on raw times, so the fused layer's fixed cost sits in the
denominator and pushes alpha DOWN; the activation correction removes traffic
from `B` and pushes it DOWN again. `cap = 2 BM / (alpha b)` is decreasing in
alpha, so both biases push the MEASURED CAP UP. C2 claims the cap is low, so it
is scored on an upper bound and a PASS survives both biases.

VALIDITY GATES (V) VERSUS CLAIM GATES (C). A V that FAILs means no number on
this page may be quoted: the kernel was not the one asked for, the instrument
was never shown to work, or the sweep was too shallow for an absence to mean
anything. A C that FAILs is a result and is meant to be publishable as one:
C1 failing says a decode-tuned tile DOES approach its compute roof, which
retracts the ceiling this study put on it.

THE DENOMINATOR, said once because it is the difference between a control and a
tautology. Every roof fraction here is against `ridge x bandwidth`, never
against the run's own plateau. The plateau is the maximum over the same cells,
so a control read against it scores 1.00 by construction and the check examines
nothing. The sibling's `bracketing` docstring carries the cost of getting that
wrong: across 26 published reports the plateau ran 46.5-75.6% of the card's own
roof, so nothing in any of them reached a compute roof, and V3 here is the gate
that says so out loud.

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

EXIT CODES. 0 the run happened and the page is readable, whatever the claim
gates said. 1 a validity gate did not pass, so nothing may be quoted (or
`--fail-on-gate` and some gate did not pass). 2 nothing was measured: a refusal,
a missing GPU stack, or arguments that could not answer the question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
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

import block_m_crossing_sweep as SWEEP  # noqa: E402

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
RIDGE_BAND = SWEEP.RIDGE_BAND             # (160.3, 176.2) Op/B, both ends of one card

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
#: `tile_block_m` where `tile_config_source` is vllm_default or vllm_tuned, 132
#: distinct cells, rows per expert read as `load_mean_rows`.
#:
#: `(cells run multi-tile, cells, max M-tiles per expert)`. This is the table
#: that demotes this experiment from a production claim to a formula test: the
#: re-read term `Q(n) = 1 + alpha (n - 1)` is exactly 1 at one tile, so a cap at
#: a tile height that never runs multi-tile is real and never approached. A tile
#: height ABSENT from this dict was not observed at all, and the report says
#: "not observed" rather than assuming either answer.
OBSERVED_MULTI_TILE: dict[int, tuple[int, int, int]] = {
    16: (0, 24, 1),
    32: (0, 5, 1),
    64: (0, 16, 1),
    128: (59, 87, 32),
}


def observed_note(block_m: int) -> str:
    """One sentence on what vLLM was seen to do at this tile height, or a refusal.

    REFUSES rather than defaults. "no cell ran multi-tile" and "no cell was
    observed" are different statements and the second must never print as the
    first: the whole demotion rests on this count, so a tile height nobody
    measured has to say so.
    """
    seen = OBSERVED_MULTI_TILE.get(block_m)
    if seen is None:
        return (f"BLOCK_M={block_m} does not appear in the observed-tile arm, "
                "so whether vLLM ever runs it multi-tile is UNMEASURED here")
    multi, cells, top = seen
    if multi == 0:
        return (f"vLLM ran BLOCK_M={block_m} as ONE M-tile per expert in "
                f"{cells} of {cells} observed cells, where Q(n) = 1 exactly, so "
                "this ceiling is never reached for in production")
    return (f"vLLM ran BLOCK_M={block_m} multi-tile in {multi} of {cells} "
            f"observed cells, up to {top} M-tiles per expert")

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

#: V3: the fraction of `ridge x bandwidth` a ladder must reach before this run
#: has DEMONSTRATED that anything can reach a compute roof. The sibling's
#: constant, imported rather than restated so the two scripts cannot come to
#: different answers about what "reached the roof" means.
COMPUTE_BOUND_FRACTION = SWEEP.COMPUTE_BOUND_FRACTION

#: Treads before an alpha may decide anything. The parent's constant, reused for
#: the same reason it exists there.
MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS

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
    "DEFAULT_SM_COUNT", "RidgeUnavailable",
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
    one here: at BLOCK_M=16 the retracted alpha predicts no crossing either
    (cap 160.0 against ridge 160.3), `crossing_rows` is None, its horizon
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

    def render(self) -> list[str]:
        out = [f"{self.tag:3s} {self.kind:8s} {self.verdict:9s} {self.claim}",
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
                        aligned_needed: int = 3) -> CapGate:
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
    """
    ok = {(c.block_m, c.tokens) for c in cells if c.status == "ok" and c.ms_p50 > 0}
    failed = [c for c in cells
              if c.status != "ok" and (c.block_m, c.tokens) not in ok]
    aligned = {bm: len({c.tokens for c in cells
                        if c.block_m == bm and c.aligned and c.status == "ok"
                        and c.ms_p50 > 0})
               for bm in tiles}
    detail = [f"{len(ok)} of {planned_cells} planned cells measured",
              "exactly-full tile stacks per setting: "
              + ", ".join(f"BM={bm}:{aligned[bm]}" for bm in tiles),
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


def gate_v2_control(ref: SWEEP.ComputeReference, tp_control, *,
                    control_tile: int, noise: float) -> CapGate:
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
    proportional = ref.block_m == control_tile
    gain = (tp_control[-1][1] / tp_control[-2][1] - 1.0
            if len(tp_control) >= 2 and tp_control[-2][1] > 0 else None)
    # The gain is a ratio of two single cells and so carries `sqrt(2)` times the
    # per-cell spread. Widening the gate with the spread is the parent sweep's
    # own rule for its memory-branch margin, and it is the difference between a
    # validity gate and a coin flip on a card that is not perfectly quiet.
    flat_gate = max(CONTROL_FLAT_GAIN, 3.0 * noise)
    lines = [f"compute reference: {ref.note}",
             "throughput per tread against ridge x bandwidth: "
             + ", ".join(f"n={n}:{v:.3f}" for n, v in tp_control)]
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
        f"proportional to {ref.mean_rel_err:.1%}"
        + (f", last tread {gain:+.2%}" if gain is not None else ", no gain readable"),
        f"through-origin fit within {PROPORTIONALITY_MAX_ERR:.0%} and last "
        f"tread within +/-{flat_gate:.2%}",
        "the compute branch every tread at the cap tile is classified against "
        "is not one, so neither C1 nor C2 may be quoted. A REFUSAL from the "
        "level checks is the likeliest cause and it names itself in the note "
        "above; membership then falls back to a split search, which invents "
        "an alpha rather than declining to",
        lines)


def gate_v3_control_roof(tp_control, *, control_tile: int, roof_tflops: float,
                         plateau: float) -> CapGate:
    """Did anything in this sweep actually reach the compute roof.

    THE POSITIVE CONTROL, and it is scored against `ridge x bandwidth` rather
    than against the run's own plateau. Against the plateau something always
    reaches 1.00, because the plateau IS the maximum over the same cells, so the
    check would examine nothing. The sibling's own note records what that costs:
    across 26 published reports the plateau ran 46.5-75.6% of the card's
    `ridge x bandwidth`, so NOTHING in any of those sweeps reached a compute
    roof and none of them was entitled to read an absence as evidence about one.

    WHAT A FAIL COSTS, precisely. C1 compares a throughput with the roof, so it
    needs this run to have produced a throughput near the roof under the same
    conditions -- otherwise "the cap tile is at 0.18 of the roof" is a statement
    about the instrument. C2 needs no roof at all: it fits a re-read fraction
    from the cap tile's own treads and compares the ceiling that implies with
    the ridge. So a FAIL here voids C1 and leaves C2 standing, and that is the
    whole reason C2 is in the report.
    """
    if not tp_control:
        return CapGate(
            "V3", VALIDITY,
            f"the control BLOCK_M={control_tile} reached the compute roof",
            UNDECIDED, "no exactly-full tile stack at the control", "n/a",
            "C1 has no positive control and may not be quoted",
            ["The control ran no aligned cell, so there is no throughput to "
             "compare with the roof."])
    top = max(v for _, v in tp_control)
    verdict = PASS if top >= COMPUTE_BOUND_FRACTION else FAIL
    return CapGate(
        "V3", VALIDITY,
        f"the control BLOCK_M={control_tile} reached the compute roof",
        verdict, f"peak {top:.3f} of ridge x bandwidth ({roof_tflops:.0f} TFLOP/s)",
        f">= {COMPUTE_BOUND_FRACTION:.2f}",
        "nothing in this sweep reached the roof, so C1 is a statement about "
        "the instrument rather than about the tile and may not be quoted. C2 "
        "SURVIVES this: it needs no roof, only the cap tile's own treads",
        [f"the sweep's best useful throughput is {plateau:.1f} TFLOP/s, "
         f"{plateau / roof_tflops:.1%} of ridge x bandwidth",
         "across the 26 published reports the sibling checked, that ratio ran "
         "46.5-75.6%, so this gate failing is the EXPECTED outcome on a card "
         "whose calibration is a triad ceiling rather than a GEMM roof, and it "
         "is a fact about the ruler as much as about the kernel",
         "throughput per tread against the roof: "
         + ", ".join(f"n={n}:{v:.3f}" for n, v in tp_control)])


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
                          discriminator: float) -> CapGate:
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
    top = max(v for _, v in tp_cap)
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
             "throughput per tread against the roof: "
             + ", ".join(f"n={n}:{v:.3f}" for n, v in tp_cap[-8:])
             + (f"  (last 8 of {len(tp_cap)} treads)" if len(tp_cap) > 8 else "")]
    claim = f"BLOCK_M={cap_tile} never gets near the compute roof, at any batch"
    consequence = (f"BLOCK_M={cap_tile} DOES approach its compute roof, which "
                   "retracts the structural ceiling this study put on a "
                   "decode-tuned MoE kernel and is the publishable answer in "
                   "the other direction")
    if not live:
        return CapGate(
            "C1", CLAIM, claim, UNDECIDED, f"peak {top:.3f} of the roof",
            "no condition is testable at this depth", consequence,
            lines + ["Neither threshold could have been tripped by the "
                     "retracted world at this depth, so a PASS would report "
                     "where the sweep stopped. Raise --r-max; V4 says to what."])
    verdict = PASS if all(top <= threshold for _, threshold in live) else FAIL
    return CapGate(
        "C1", CLAIM, claim, verdict, f"peak {top:.3f} of the roof",
        " and ".join(f"<= {threshold:.3f} ({name})" for name, threshold in live),
        consequence, lines)


def gate_c2_measured_cap(fit, corrected: float | None, *, cap_tile: int,
                         ridge: float, b: int, discriminator: float) -> CapGate:
    """THE CAP, from the re-read fraction this ladder measures itself.

    `cap = 2 BM / (alpha b)` with alpha fitted on the cap tile's OWN treads,
    which is the one place in this study where alpha is cleanly identifiable: a
    tile that never crosses is memory bound at every tread, so there is no
    compute branch for the fit to run into and no import from another block
    size.

    THE STRUCTURAL THRESHOLD DOES NOT DISCRIMINATE, and the gate says so rather
    than taking credit for it. "No crossing exists" is `cap < ridge`, i.e.
    `alpha > 2 BM / (b ridge) = 0.0998` at ridge 160.3, and the retracted
    alpha=0.10 clears that by 0.2%. Both worlds predict no crossing. So the gate
    is the MIDPOINT of the two worlds' predicted `cap/ridge`, computed from the
    two registered alphas at the ridge in use, and the structural comparison is
    printed beside it as the weaker statement it is.
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
    return CapGate(
        "C2", CLAIM,
        f"the re-read fraction measured at BLOCK_M={cap_tile} puts its AI "
        "ceiling far below the ridge",
        verdict, f"cap/ridge = {cap:.1f}/{ridge:.1f} = {cap / ridge:.3f}",
        f"<= {discriminator:.3f}, the midpoint of the two worlds",
        f"the measured re-read fraction is small enough to put BLOCK_M="
        f"{cap_tile} within reach of the roof, which is the retracted world's "
        "prediction and not this study's",
        [f"alpha {corrected:.3f} activation-corrected ({fit.alpha:.3f} raw) "
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
    out.append("  NOT RUN: the parent sweep's gate 3 (alpha against a 0.33 "
               "midpoint) is superseded here by C2, which fits the same "
               "re-read fraction and then does the thing the cap claim needs: "
               "converts it to a ceiling and compares that with the ridge.")
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
    deepest = max(aligned[tiles[0]]) // tiles[0] if aligned[tiles[0]] else 0
    if deepest < depth.tiles:
        lines.append(
            f"            WARNING: the grid's deepest exactly-full stack at "
            f"BLOCK_M={tiles[0]} is {deepest} tiles, short of the {depth.tiles} "
            "V4 requires. Raise --r-max, or lower --row-step so more multiples "
            f"of {tiles[0]} land on the grid.")
    return lines


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
            synthetic: bool = False) -> Report:
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
    ref = SWEEP.compute_reference(ok, tiles, cfg=cfg, ridge=ridge,
                                  bandwidth_gbps=bandwidth_gbps, b=b,
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
    lines.append("  every roof fraction below is against ridge x bandwidth and "
                 "NOT against that plateau: the plateau is the maximum over the "
                 "same cells, so a control read against it scores 1.00 by "
                 "construction. Far below 100% means nothing in the sweep "
                 "reached a roof, which is what V3 tests")
    lines.append(f"  per-cell timing spread, median {noise:.2%}; memory-branch "
                 f"margin raised to {margin:.2%}")
    lines.append(f"  {sm_count} SMs ({sm_source})")
    lines.append(f"  compute reference: {ref.note}")
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
                        noise=noise),
        gate_v3_control_roof(tp_control, control_tile=control_tile,
                             roof_tflops=roof_tflops, plateau=plateau),
        gate_v4_depth(reached, depth, cap_tile=cap_tile, alpha=alpha),
        gate_c1_roof_fraction(tp_cap, depth, cap_tile=cap_tile, alpha=alpha,
                              ridge=ridge, b=b, roof_tflops=roof_tflops,
                              discriminator=discriminator),
        gate_c2_measured_cap(fit_cap, corrected, cap_tile=cap_tile, ridge=ridge,
                             b=b, discriminator=discriminator),
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
        "compute_reference": ref.note,
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
    """
    key = json.dumps({"card": card, "model": args.model, "dtype": args.dtype,
                      "cap_tile": args.cap_tile, "control": args.control,
                      "r_max": r_max, "row_step": args.row_step,
                      "probes": args.step_probes, "seed": args.seed,
                      "group_m": args.group_m, "block_n": args.block_n,
                      "num_stages": args.num_stages,
                      "iters": args.iters, "warmup": args.warmup,
                      "budget": args.cell_budget_ms,
                      "self_test": args.self_test,
                      "self_test_noise": args.self_test_noise},
                     sort_keys=True)
    prefix = "synthetic-" if args.self_test is not None else ""
    return (f"{prefix}{card}-{args.model}-{args.dtype}-"
            f"bm{args.cap_tile}v{args.control}-"
            f"r{r_max}-g{args.group_m}-n{args.block_n}-"
            f"{hashlib.sha1(key.encode()).hexdigest()[:6]}")


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
                         "swizzle: it runs 0.84/0.73/0.68/0.67 at G=1/8/16/64 "
                         "on both cards, so the ceiling 2 BM/(alpha b) moves "
                         "with this number and the cap claim is a claim at G=1")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"],
                    help="the N tile, applied to BOTH settings. An extra M-tile "
                         "re-reads activations as well as weights in the ratio "
                         "BLOCK_M/BLOCK_N, so at 64 the cap tile re-reads them "
                         "a quarter as often as it re-reads weights and the "
                         "activation correction on alpha is small")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="iterations are cut so one cell stays inside this")
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
                         "run the whole analysis on them, off GPU")
    ap.add_argument("--self-test-noise", type=float, default=0.0,
                    help="lognormal sigma applied to every synthetic cell")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero unless every gate passes; off by "
                         "default because a falsified claim gate is a "
                         "successful run, not a failed one")
    return ap


def main(argv=None) -> int:
    # BEFORE `build_parser`, which reads `SWEEP.FIXED` for its defaults. A probe
    # after `parse_args` fires after the AttributeError it exists to replace.
    try:
        require_sweep_api()
    except CapTestRefusal as exc:
        print(f"REFUSED: {exc}")
        return 2
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    tiles = (args.cap_tile, args.control)
    if args.cap_tile >= args.control:
        print(f"--cap-tile {args.cap_tile} is not smaller than --control "
              f"{args.control}. The control exists to be the tile that CAN "
              "cross while the cap tile cannot; ordering them the other way "
              "makes every comparison below read backwards.")
        return 2
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
    else:
        try:
            rr = SWEEP.resolve_ridge(args, synthetic=synthetic or args.dry_run)
        except SWEEP.RidgeUnavailable as exc:
            print(f"REFUSED: {exc}")
            return 2
        ridge, ridge_band = rr.ridge, rr.band
        ridge_source, band_source, ridge_device = rr.source, rr.band_source, rr.device

    try:
        depth = required_depth(args.cap_tile, b=b, ridge_band=ridge_band)
    except CapTestRefusal as exc:
        print(f"REFUSED: {exc}")
        return 2
    r_max = args.r_max or depth.rows
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
        return 2

    if args.dry_run:
        secs = SWEEP.estimated_seconds(
            cfg, grid, tiles, alpha=args.alpha, ridge=ridge,
            bandwidth_gbps=bandwidth, b=b, iters=args.iters,
            warmup=args.warmup, cell_budget_ms=args.cell_budget_ms)
        print(f"\nestimated GPU time {secs:.0f} s at the model's own timings, "
              "excluding compiles and allocation")
        print("nothing was measured and nothing was written")
        return 0

    if args.self_test is None:
        missing = SWEEP.missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    planned = len(grid) * len(tiles)

    if args.self_test is not None:
        alpha = args.self_test
        sm_count = args.sm_count or SWEEP.DEFAULT_SM_COUNT
        cells = SWEEP.synthetic_cells(cfg, grid, tiles, alpha=alpha,
                                      ridge=ridge, bandwidth_gbps=bandwidth,
                                      b=b, sm_count=sm_count,
                                      noise=args.self_test_noise, seed=args.seed)
        compiles = {bm: 1 for bm in tiles}
        executed = dict(compiles)
        print(f"\nSELF TEST: cells GENERATED from the model at alpha={alpha}. "
              "Nothing here was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
    else:
        import torch
        alpha = args.alpha
        sm_count = (args.sm_count
                    or torch.cuda.get_device_properties(0).multi_processor_count)
        started = time.time()
        cells, compiles, executed = SWEEP.run_sweep(
            args, cfg, grid, tiles, csv_path, cache_root, b, pinned)
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
                         ridge_source=ridge_source, band_source=band_source,
                         card=card, ridge_device=ridge_device,
                         synthetic=synthetic)
    except CapTestRefusal as exc:
        print(f"\nREFUSED: {exc}")
        print(f"the cells that did land are at {csv_path} and a re-run resumes "
              "them, so nothing measured is lost")
        return 2

    # The plan and the predictions are already on the terminal, printed before
    # the sweep ran, which is the only order that makes "registered before the
    # run" a property of the transcript. report.txt carries them again so the
    # FILE is self-contained; stdout does not repeat them.
    print("\n".join(report.lines[len(header):]))
    (out_dir / "report.txt").write_text(report.text())
    (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
    print(f"cells    {csv_path}")
    print(f"report   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")

    if args.fail_on_gate and any(g.verdict != PASS for g in report.gates):
        return 1
    if any(g.kind == VALIDITY and g.verdict != PASS for g in report.gates):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
