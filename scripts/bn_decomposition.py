#!/usr/bin/env python
"""Sweep BLOCK_SIZE_N at fixed BLOCK_M: the only clean separation of alpha_a from alpha_b.

    python scripts/bn_decomposition.py --self-test    # plant four worlds, off GPU
    python scripts/bn_decomposition.py --dry-run      # the plan, the predictions, the cost
    python scripts/bn_decomposition.py                # the pod run

WHY BLOCK_SIZE_N AND NOTHING ELSE. What a ladder fit returns is not the weight
miss fraction. It is the blend

    alpha_fitted = alpha_b + alpha_a (BM/BN) + BM/K                        (LIN)

and BM appears in TWO of those terms while BN appears in exactly one. So
sweeping BM moves alpha_a and alpha_b together and can never separate them --
which is why the study's current alpha_a is a TWO-POINT slope between BN=64 and
BN=256, stable on the A100 (0.106, 0.102, 0.129, 0.119) and not on the H200,
where one of the two-point values comes out NEGATIVE and a miss fraction cannot
be. Sweeping BN at FIXED BM moves exactly one term. Three or more BN values give
a fit where there were slopes, and -- this is the half that matters more -- they
leave a RESIDUAL, which is the only thing in this study that can say whether the
three terms are ALL of it.

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
     BM=32, 64 and 128 separately turns a parameter into a testable invariant.
  4. alpha_b against TEMPO (arXiv:2608.13057), which publishes b2/b = 0.311 and
     0.319 for the weight-side re-read and models NO activation-side re-read at
     all, so alpha_a has no counterpart in the closest prior work.

BLOCK_M=128 IS THE PRIMARY AND THE REASON IS PRODUCTION, NOT CONVENIENCE. It is
the only tile vLLM's fallback ladder ever runs multi-tile: in the one published
arm that records the tile actually chosen, BLOCK_M 16, 32 and 64 run a single
M-tile per expert in every one of 45 cells, and 128 runs up to 32 tiles in 59 of
87. The re-read term only exists when there is more than one tile, so 128 is the
only block size where alpha_b is a production quantity rather than a curiosity.
BM=64 and BM=32 are swept beside it because they are where alpha is ROBUSTLY
identifiable, and because the invariance in (3) needs more than one BM.

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
INSIDE the tolerance. So BLOCK_M=128 is expected to yield 2 of 3 BN points and
the pooled fit to carry the residual test, while BM=64 and BM=32 sit far outside
the band at every BN. Those predictions are ANCHORED on that measured 0.991 and
not on a calibrated ridge: `B/C` carries the kernel's OWN achieved
FLOP-per-byte, which is not the card's ridge, and the calibrated form says the
BLOCK_M=256 reference at BN=64 is memory bound when it is measurably the
qualified compute reference in 22 of the 24 published arms.

AND THE SWIZZLE DECIDES WHETHER ANY OF IT IS RESOLVABLE. The response moves with
alpha_a as `g1 (1 - alpha_b)/(1 + phi)^2`, so the design's whole power is
proportional to `1 - alpha_b`. At GROUP_SIZE_M=1 -- the production fallback, and
this file's default because it is what the study pins elsewhere -- the corpus
puts alpha near 0.93 and the lever is worth 15% of its size at GROUP_SIZE_M=16;
planted at that swizzle, alpha_a's own spread is 0.11 to 0.13 against the 0.025
C1 needs, at every rep count tried. So a G=1 run measures alpha_b, the residual
and the invariance, and reads UNKNOWN on alpha_a; `--group-m 16` is the setting
that resolves it, and `--reps` buys the rest as 1/sqrt(reps). S4 computes this
before any GPU time and the plan prints it, because it is a property of the
pinning and not of the pod.

WHAT IT WRITES. Under `$MOE_RESULTS_DIR`, else `/workspace/results` (the RunPod
network volume, which outlives the pod), else `<repo>/results`:

    <results>/bn_decomposition/<run-id>/cells.csv     one row per tread per rep
    <results>/bn_decomposition/<run-id>/CARD          the card that wrote it
    <results>/bn_decomposition/<run-id>/report.txt    exactly what was printed
    <results>/bn_decomposition/<run-id>/report.json   fits, gates, provenance
    <results>/bn_decomposition/<run-id>/triton-cache/ per-(BN,BM) compile evidence

`cells.csv` is appended and flushed per timing and a re-run resumes it. The run
id carries EVERY swept knob AND the card, because the results root is a network
volume shared between pods and this repo has already had one card silently
report another's timings twice.

OFF GPU. `--dry-run` prints the plan, the resource bill, the per-cell
predictions and the cost. `--self-test` plants four worlds -- the exact model,
the model with a term missing, a world with no activation re-read at all, and a
noise-only world -- and checks the gates come out DIFFERENT in each, which is
the claim that they discriminate rather than the claim that they pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
              "results_root", "scaled_iters", "useful_flops",
              "weight_bytes_per_expert", "activation_bytes_per_row",
              "activation_slope_ms",
              "resolve_ridge", "RidgeUnavailable", "missing_gpu_stack",
              "find_override", "count_new", "time_call", "balanced_ids")
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
#: alpha_b invariance test needs more than one BM to be a test at all.
SUBJECT_BLOCK_M = (32, 64, 128)

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

#: Predicted alpha_a. The published two-point A100 slopes are 0.106, 0.102,
#: 0.129 and 0.119; `moe/bench/ai_model.py` derives 0.143 from the study's own
#: ALPHA_BY_BLOCK_M pair; the H200 G=1 mixtral pair gives 0.146 read through
#: (LIN). The band is those, rounded outward.
ALPHA_A_BAND = (0.10, 0.15)

#: TEMPO (arXiv:2608.13057) publishes these for the weight-side re-read, in two
#: configurations. They are the closest prior work and the only external number
#: alpha_b can be checked against. TEMPO models no activation-side re-read, so
#: alpha_a has no counterpart there.
TEMPO_B2_OVER_B = (0.311, 0.319)

#: How far alpha_b may sit from TEMPO's pair before C4 calls it a disagreement,
#: as a fraction. 0.15 is set wider than the 2-4% the study's decomposed 0.307
#: reaches, because that 0.307 comes from a swizzle-POOLED refit and this run is
#: pinned to ONE GROUP_SIZE_M -- see the note on C4.
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
#: instead of PASS. Half the band width: an estimator whose interval is wider
#: than the hypothesis it tests has not tested it.
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
#: used as a gate: results/published/NOISE_FLOOR.json records a paired s3-vs-s4
#: sd of 0.0323 over 11 cells and a prior sd of 0.0228. This run's own bootstrap
#: is what C2 is scored against, because a floor measured on other arms cannot
#: know how noisy THIS pod was.
PUBLISHED_ALPHA_SD = 0.0228

#: The card slug a run id carries when no device is attached: every --dry-run
#: and every --self-test on a laptop. Visible rather than blank, so a laptop
#: directory cannot be mistaken for the one a pod would write to.
NO_CARD_SLUG = "nocard"

#: Two worlds, both stated as (alpha_b, alpha_a), used ONLY to predict and to
#: cost. Neither is a measurement of this run.
#:
#:  POOLED   the study's own ALPHA_BY_BLOCK_M = {64: 0.466, 128: 0.625} read
#:           through (LIN), which is where moe/bench/ai_model.py's 0.307/0.143
#:           comes from. It is pooled over GROUP_SIZE_M, which swings alpha by
#:           0.39, so it describes no single pinned setting.
#:  LADDER   this study's own G=1 ladder fits, which measure alpha_fitted at
#:           0.916-0.951 at BM=64, BN=64 on BOTH cards with all 16 treads memory
#:           bound. Read through (EXA) with alpha_a at the (LIN) value, that is
#:           alpha_b near 0.92 -- a nearly full re-read per M-tile, which is
#:           what GROUP_SIZE_M=1 means: consecutive M-tiles of one expert are
#:           scheduled far apart and the L2 keeps nothing.
#:
#: They disagree about alpha_b by a factor of three and they are BOTH derived
#: from published numbers in this repo. Which one this run lands in is a result.
WORLD_POOLED = (0.307, 0.143)
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
# The geometry. Pure arithmetic on a model config: no torch, no GPU, no files.
# Every number the predictions and the fit use comes from here, so a reader can
# check the experiment without running it.
# --------------------------------------------------------------------------

def weight_elements(cfg) -> int:
    """`W = 3 F H`: up `[H, 2F]` plus down `[F, H]`, one expert, in elements."""
    return 3 * cfg.intermediate_size * cfg.hidden_size


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
    """One timing of one tread of one (BLOCK_N, BLOCK_M) setting. The CSV row."""

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


SAMPLE_FIELDS = list(Sample.__dataclass_fields__)


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

    @property
    def ok(self) -> bool:
        return not self.refusals and self.block_m is not None

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
                      subjects=SUBJECT_BLOCK_M) -> RefVerdict:
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
    """
    ref = SWEEP.compute_reference(
        cells, block_sizes, cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
        capability=capability)
    why: list[str] = list(ref.refusals)
    if ref.block_m is None:
        return RefVerdict(block_n, None, None, 0.0, None, ceiling_tflops, None,
                          tuple(why) or (ref.note,), ref.note, ref)
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
                      ref.note, ref)


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
        (), ref.note, ref, basis="IMPORTED",
        import_note=(
            "this arm qualified NO compute branch of its own ("
            + (target.refusals[0][:90] if target.refusals else "no candidate")
            + "). Every alpha below rests on the assumption that the achieved "
              "compute rate does not move with BLOCK_N; the lenders' own rates "
              "are printed above and V3 gates their spread"))


def arm_alphas(samples, cfg, *, block_ns, subjects, ridge: float,
               bandwidth_gbps: float, b: int, base_pinned: dict, capability,
               ceiling_tflops: float, sm_count: int, rng=None
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
        verdict_by_bn[bn] = qualify_reference(
            cells_by_bn[bn], tuple(sorted(ladders)), bn, cfg=cfg, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
            capability=capability, ceiling_tflops=ceiling_tflops,
            subjects=subjects)

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
            cells.append(AlphaCell(
                bn, bm, fit.alpha if not blank else None,
                fit.alpha_upper if not blank else None,
                corrected if not blank else None, fit.memory_points, len(pts),
                spread, basis, blank=blank))
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

    def render(self) -> list[str]:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        out = [f"[{tag}] {self.kind:8s} {self.name}  {self.prediction}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is not True and self.invalidates:
            out.append(f"         a non-PASS here invalidates: {self.invalidates}")
        out += [f"         {line}" for line in self.lines]
        return out


def render_gates(gates: list[Gate]) -> list[str]:
    out: list[str] = []
    for g in gates:
        out += g.render()
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    return out + ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN"]


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


def gate_cross_bn(refusal: str, spread: float | None) -> Gate:
    return Gate(VALIDITY, "V3 references agree across BN",
                "the arms' compute references describe one machine",
                f"max/min implied TFLOP/s across BN <= "
                f"{REFERENCE_CROSS_BN_SPREAD:.1f}x",
                None if spread is None else not refusal,
                (f"{spread:.2f}x" if spread is not None else
                 "fewer than two arms produced a reference"),
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
                 f"against a tolerance of {TOLERANCE:.2f}."])


def gate_sharpness(boot: Bootstrap) -> Gate:
    """Is the estimator sharp enough for C1 to mean anything."""
    sd = boot.alpha_a_sd
    return Gate(VALIDITY, "V6 estimator sharpness",
                "alpha_a's interval is narrower than the band it is tested "
                "against",
                f"bootstrap sd(alpha_a) <= {ALPHA_A_SD_CEILING:.3f}, half the "
                f"width of the [{ALPHA_A_BAND[0]:.2f}, {ALPHA_A_BAND[1]:.2f}] "
                "band",
                None if sd is None else sd <= ALPHA_A_SD_CEILING,
                "no bootstrap spread" if sd is None else f"sd = {sd:.4f}",
                "C1: an estimator whose interval is wider than the hypothesis "
                "it tests has not tested it, and a PASS would be an artefact "
                "of the band's width",
                [boot.note])


def gate_alpha_a(fit: Decomposition, boot: Bootstrap, sharp: bool) -> Gate:
    lo, hi = ALPHA_A_BAND
    val = fit.alpha_a
    sd = boot.alpha_a_sd
    inside = None if (val is None or not sharp) else lo <= val <= hi
    return Gate(CLAIM, "C1 alpha_a", f"alpha_a lands in [{lo:.2f}, {hi:.2f}]",
                f"{lo:.2f} <= alpha_a <= {hi:.2f} from the {fit.form} fit",
                inside,
                "not fitted" if val is None else
                f"alpha_a = {val:.4f}"
                + (f" +/- {sd:.4f}" if sd else " (no interval)"),
                lines=["The band is the published two-point slopes (A100: "
                       "0.106, 0.102, 0.129, 0.119), ai_model.py's 0.143 from "
                       "the study's own ALPHA_BY_BLOCK_M pair, and the H200 "
                       "G=1 mixtral pair's 0.146 -- all read through (LIN). A "
                       "FAIL well ABOVE the band with a clean residual would "
                       "say the two-point slopes were biased by the missing "
                       "denominator; a FAIL at zero says BN does not move "
                       "alpha at all and the activation re-read is not there."])


def gate_residual(fit: Decomposition, chi2: float | None, why: str,
                  struct: Structure) -> Gate:
    """C2, and the one that matters more than the parameters.

    A fit always returns numbers. This asks whether the numbers describe the
    data: if the three terms are all of it, the residual is the measurement
    noise and chi2 is about 1. Structure in the residual means a term is
    missing, and `struct` names which candidate column it lines up with.
    """
    passed = None if chi2 is None else chi2 <= RESIDUAL_CHI2_CEILING
    return Gate(CLAIM, "C2 model completeness",
                "the three terms are ALL of it: the residual is noise",
                f"chi2 = sum (r/sigma)^2 / dof <= {RESIDUAL_CHI2_CEILING:.1f}, "
                "sigma from this run's own bootstrap",
                passed,
                (f"chi2 = {chi2:.2f} over {fit.dof} dof; residual RMS "
                 f"{fit.rms:.4f}" if chi2 is not None and fit.rms is not None
                 else f"not computable: {why}"),
                lines=[f"structure: {struct.line()}",
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
                       "kernel. The study's own 0.307 is a POOLED refit across "
                       "GROUP_SIZE_M 1, 8, 16 and 64, which swing alpha by "
                       "0.39; its G=1 ladder fits sit near 0.92, which is what "
                       "'consecutive M-tiles of one expert are scheduled far "
                       "apart and the L2 keeps nothing' should look like. So a "
                       "FAIL at G=1 does NOT refute TEMPO: it says the number "
                       "the study compared with TEMPO was pooled over a knob "
                       "TEMPO holds fixed. Re-run at --group-m 16 to compare "
                       "like with like."])


# --------------------------------------------------------------------------
# The registered predictions, printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def cell_table(cfg, b: int, ridge: float, block_ns, subjects, treads: dict
               ) -> list[str]:
    """Predicted alpha_fitted and B/C for every cell, in both worlds.

    Printed BEFORE the run and carried into the report, so "the model predicted
    this" is checkable rather than remembered. The two worlds are the study's
    own pooled refit and the study's own G=1 ladder fits; they disagree by a
    factor of three on alpha_b and this run lands in one of them.
    """
    out = ["                 POOLED (0.307, 0.143)          "
           "LADDER (0.920, 0.146)      calibrated-ridge B/C",
           "  BM   BN    a_fit    B/C  treads      a_fit    B/C  treads    "
           "POOLED  LADDER"]
    for bm in sorted(subjects) + [REFERENCE_BLOCK_M]:
        for bn in sorted(block_ns):
            row = f"  {bm:3d} {bn:4d} "
            for ab, aa in (WORLD_POOLED, WORLD_LADDER):
                a = alpha_fitted_exact(cfg, bm, bn, alpha_b=ab, alpha_a=aa)
                r = anchored_ratio(cfg, bm, bn, alpha_b=ab, alpha_a=aa)
                n = memory_treads(cfg, bm, bn, alpha_b=ab, alpha_a=aa,
                                  ratio=r, treads=treads.get(bm, 8))
                flag = "*" if abs(r - 1.0) <= TOLERANCE else " "
                row += f"  {a:6.3f} {r:6.3f}{flag} {n:3d}/{treads.get(bm, 8):<3d}"
            row += "    " + "  ".join(
                f"{branch_ratio(cfg, bm, bn, alpha_b=ab, alpha_a=aa, ridge=ridge, b=b):6.3f}"
                for ab, aa in (WORLD_POOLED, WORLD_LADDER))
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
                     subjects, treads: dict, group_m: int) -> str:
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
        "    BASIS  published two-point A100 slopes 0.106, 0.102, 0.129, 0.119; "
        "ai_model.py's 0.143 from",
        "           ALPHA_BY_BLOCK_M; the H200 G=1 mixtral pair's 0.146. All of "
        "those are (LIN) readings.",
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
        f"P4  alpha_b at GROUP_SIZE_M={group_m}. The study's decomposed 0.307 "
        "corroborates TEMPO's b2/b of",
        "    0.311/0.319 to 2-4%, and that 0.307 is POOLED over GROUP_SIZE_M 1, "
        "8, 16 and 64, which swing",
        "    alpha by 0.39. This run pins ONE swizzle. At G=1 the study's own "
        "ladders say alpha_b is near",
        f"    0.92, so C4 is predicted to FAIL at G={group_m} unless it is 16 "
        "or above -- and that failure is",
        "    a statement about the pooling, not about TEMPO.", "",
        "P5  WHICH CELLS GO MISSING, registered because two of them are "
        "predicted to.",
        f"    THE PRIMARY AT BN={ANCHOR_BLOCK_N} IS PREDICTED TO BE DISCARDED: "
        f"the anchor is {ANCHOR_RATIO:.3f}, inside the parallel-branch",
        f"    tolerance of {TOLERANCE:.2f}, and that is the measured median of "
        "22 published ladders rather than a guess.",
        "    At BN=32 the primary is predicted to escape UP (every tread "
        "memory bound) and at BN=128 to keep a",
        "    prefix, so it should contribute 2 of 3 points and the POOLED fit "
        "carries the residual test. BM=64",
        "    and BM=32 sit far outside the band at every BN, which is what "
        "makes the sweep survive that.",
        f"    THE BN=32 ARM IS THE ONE AT RISK: its BLOCK_M="
        f"{REFERENCE_BLOCK_M} reference is predicted compute bound in the "
        "LADDER",
        "    world and MEMORY bound in the POOLED one, so the two worlds "
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
    iters: int
    warmup: int
    cell_budget_ms: float
    seconds: float

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
            f"design power at GROUP_SIZE_M={self.group_m}: the corpus puts "
            f"alpha_b near {planted_alpha_b(self.group_m):.2f} there, and the "
            "response moves with alpha_a as g1 (1 - alpha_b)/(1 + phi)^2, so "
            f"the lever is {(1 - planted_alpha_b(self.group_m)) / (1 - planted_alpha_b(16)):.0%} "
            "of its size at GROUP_SIZE_M=16."
            + ("  AT THIS SWIZZLE THE DESIGN CANNOT RESOLVE alpha_a AT ANY REP "
               "COUNT TRIED (sd 0.11-0.13 against a 0.025 bar): the run still "
               "measures alpha_b, C2 and C3, but C1 will read UNKNOWN. "
               "--group-m 16 is the setting that resolves it."
               if planted_alpha_b(self.group_m) > 0.85 else ""),
            f"timing       {self.warmup} warmup + up to {self.iters} iters, cut "
            f"to keep one timing inside {self.cell_budget_ms:.0f} ms",
            f"timings      {self.timings} "
            f"({sum(len(v) for v in self.rows.values())} treads x {self.reps} "
            "reps)",
            f"estimate     {self.seconds:.0f} s of GPU at the model's own "
            "timings, excluding compiles and allocation",
        ]
        for (bn, bm), rows in sorted(self.rows.items()):
            out.append(f"  BN={bn:4d} BM={bm:4d}  treads {len(rows):2d}  "
                       f"rows {rows[0]}..{rows[-1]}  T "
                       f"{SWEEP.tokens_for_rows(cfg, rows[0])}.."
                       f"{SWEEP.tokens_for_rows(cfg, rows[-1])}")
        for (bn, bm), why in sorted(self.refusals.items()):
            out.append(f"  REFUSED BN={bn} BM={bm}: {why}")
        return out


def build_plan(args, cfg, b: int, capability, ridge: float,
               bandwidth_gbps: float) -> Plan:
    """The grid, the resource refusals, and the cost -- all off GPU.

    A SETTING THAT CANNOT HOLD ITS ACCUMULATOR IS DROPPED HERE, where it is
    chosen, and not diagnosed afterwards: a spilled kernel still returns a time,
    that time is still proportional to its tile count, and this study has
    already published 8 cells classified against one.

    AND AN ARM WHOSE REFERENCE IS DROPPED IS DROPPED WHOLE. The subjects of an
    arm with no reference cannot be classified, so measuring them would buy
    nothing but GPU time and a row of blanks that reads like data.
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
    total = 0.0
    ab, aa = WORLD_LADDER
    for (bn, bm), rs in rows.items():
        for r in rs:
            ms = planted_ms(cfg, bm, bn, r // bm, alpha_b=ab, alpha_a=aa,
                            ridge=ridge, bandwidth_gbps=bandwidth_gbps, b=b,
                            overhead_ms=args.overhead_ms)
            iters = SWEEP.scaled_iters(ms, args.iters, args.cell_budget_ms)
            total += args.reps * ms * (args.warmup + iters)
    return Plan(args.model, args.dtype, base, tuple(kept_ns), subjects, rows,
                refusals, args.reps, args.group_m, args.iters, args.warmup,
                args.cell_budget_ms, total / 1e3)


# --------------------------------------------------------------------------
# Persistence, identity, and the two ways this repo has already lost an arm.
# --------------------------------------------------------------------------

def append_sample(path: Path, sample: Sample) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(sample))
        fh.flush()


def read_samples(path: Path) -> tuple[set[tuple[int, int, int, int]], list[Sample]]:
    """Timings already on disk, so a re-run resumes rather than repeats.

    Only SUCCESSFUL timings count as done: the common failure here is a pod
    that lost its device or a setting that ran out of shared memory, both of
    which a re-run can leave behind, and a real failure fails again in
    milliseconds.
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
                status=row.get("status", "ok"), detail=row.get("detail", "")))
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

    BLOCK_SIZE_N IS THE SWEEP HERE, so the LIST is in the key rather than a
    value: two runs over different BN grids are different experiments even when
    the grids overlap, because `--block-n-list 32,64,128` and `64,128` fit
    different numbers of points and V5 reads a different verdict.

    `--ridge`, `--bandwidth-gbps` and `--draws` stay OUT: they re-analyse a set
    of timings rather than change one, and two analyses of one sweep belong in
    one directory.
    """
    key = json.dumps({"card": card, "model": args.model, "dtype": args.dtype,
                      "block_ns": args.block_n_list, "tiles": args.tiles,
                      "reference": REFERENCE_BLOCK_M, "r_max": args.r_max,
                      "max_treads": args.max_treads, "reps": args.reps,
                      "group_m": args.group_m, "block_k": args.block_k,
                      "num_stages": args.num_stages,
                      "num_warps": args.num_warps, "iters": args.iters,
                      "warmup": args.warmup, "budget": args.cell_budget_ms,
                      "seed": args.seed}, sort_keys=True)
    ns = args.block_n_list.replace(",", "_")
    ms = args.tiles.replace(",", "_")
    return (f"{card}-{args.model}-{args.dtype}-n{ns}-bm{ms}-r{args.r_max}"
            f"-t{args.max_treads}-g{args.group_m}-k{args.block_k}"
            f"-s{args.num_stages}-w{args.num_warps}-x{args.reps}-"
            f"{hashlib.sha1(key.encode()).hexdigest()[:6]}")


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
                    samples: list[Sample]) -> tuple[int, int]:
    """Time one (BN, BM) setting, `--reps` round-robin passes over its treads.

    ROUND ROBIN INSIDE THE SETTING. Measuring tread 1 fifty times and then
    tread 8 fifty times puts every tread at a different point in the pod's
    thermal history, and the resulting monotone drift IS a slope -- the very
    quantity being fitted. One pass over all treads per repeat spreads that
    drift across the ladder instead of aligning it with the x axis.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
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
                    ms0, _, _ = SWEEP.time_call(call, 1, 3)
                    iters = SWEEP.scaled_iters(ms0, args.iters,
                                               args.cell_budget_ms)
                    ms, mn, sd = SWEEP.time_call(call, args.warmup, iters)
                sample = Sample(block_n, block_m, r // block_m, r, tokens, rep,
                                ms, mn, sd, iters)
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
            append_sample(csv_path, sample)
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
                sm_count: int, block_ns, subjects
                ) -> tuple[list[str], list[Gate], dict]:
    """Everything read off the timings, as text, gates and a payload."""
    kw = dict(block_ns=block_ns, subjects=subjects, ridge=ridge,
              bandwidth_gbps=bandwidth_gbps, b=b, base_pinned=base_pinned,
              capability=capability, ceiling_tflops=ceiling_tflops,
              sm_count=sm_count)
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

    boot = run_bootstrap(samples, cfg, keys, draws=args.draws, seed=args.seed,
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

    lines = ["", "## The arms", ""]
    for v in verdicts:
        lines += v.render()
    if cross_spread is not None:
        lines.append(f"  across BN, implied rates span {cross_spread:.2f}x "
                     f"(gate <= {REFERENCE_CROSS_BN_SPREAD:.1f}x)")

    lines += ["", "## The cells", "",
              "   BN   BM   treads  mem   alpha    corrected   sigma  surv  "
              "basis"]
    for c in sorted(cells, key=lambda c: (c.block_m, c.block_n)):
        sd = boot.per_cell_sd.get((c.block_n, c.block_m))
        surv = boot.survival.get((c.block_n, c.block_m))
        lines.append(
            f"  {c.block_n:4d} {c.block_m:4d}   {c.treads:4d} {c.memory_points:4d}   "
            + (f"{c.alpha:7.4f}" if c.alpha is not None else "  BLANK")
            + ("  " + (f"{c.alpha_corrected:9.4f}"
                       if c.alpha_corrected is not None else "     n/a"))
            + ("  " + (f"{sd:6.4f}" if sd else "   n/a"))
            + ("  " + (f"{surv:4.0%}" if surv is not None else " n/a"))
            + "  " + (c.blank or c.basis)[:64])

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
    gates = [
        gate_non_vacuity(counts),
        gate_override(compiles, executed),
        gate_reference_level(verdicts),
        gate_cross_bn(cross_why, cross_spread),
        gate_ladders(spreads, inversion_rows, boot.survival),
        gate_identifiable(cells, PRIMARY_BLOCK_M),
        gate_sharpness(boot),
        gate_alpha_a(fit, boot, sharp),
        gate_residual(fit, chi2, why, struct),
        gate_invariance(per_bm, boot),
        gate_physicality(fit, boot),
        gate_tempo(fit, boot, args.group_m),
    ]
    payload = {
        "ridge": ridge, "bandwidth_gbps": bandwidth_gbps,
        "ceiling_tflops": ceiling_tflops, "ceiling_source": ceiling_source,
        "pinned": base_pinned, "block_ns": list(block_ns),
        "subjects": list(subjects), "reference_block_m": REFERENCE_BLOCK_M,
        # Rebuilt WITHOUT `ref`, which holds the sweep's own dataclass and is
        # not JSON. Every other field is carried, `basis` and `import_note`
        # included: a cell resting on another arm's ruler has to be
        # distinguishable in the file and not only in the printout.
        "arms": [{k: v for k, v in asdict(
            RefVerdict(v.block_n, v.block_m, v.slope_per_tile, v.overhead_ms,
                       v.implied_tflops, v.ceiling_tflops, v.fraction,
                       v.refusals, v.note, basis=v.basis,
                       import_note=v.import_note)).items() if k != "ref"}
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
                      "note": boot.note},
        "spreads": spreads, "inversions": inversion_rows,
        "gates": [asdict(g) for g in gates],
    }
    return lines, gates, payload


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
                    out.append(Sample(bn, bm, n, r, SWEEP.tokens_for_rows(cfg, r),
                                      rep, ms, ms, ms * noise, args.iters))
    return out


def self_test(args, cfg, b: int, ridge: float, bandwidth_gbps: float,
              ceiling_tflops: float, capability, block_ns, subjects
              ) -> tuple[list[str], list[Gate]]:
    """Four worlds. The claim is that the gates DISCRIMINATE, not that they pass.

      TRUTH     the exact model at (alpha_b, alpha_a) = (0.31, 0.14).
                C1 and C2 must PASS and alpha_a must come back near 0.14.
      MISSING   the same, plus a term the model does not contain: alpha_b
                rising with (BM/BN)^2. C2 must FAIL and the structure test must
                name the quadratic column.
      NO-A      alpha_a = 0. C1 must FAIL at the bottom of the band, and C2
                must still PASS -- a wrong parameter with a clean residual is a
                different verdict from a wrong model, and the two gates have to
                tell them apart.
      BN-DRIFT  alpha_b moving linearly with 1/BN, which is the specific
                alternative this experiment was built to see. C2 must FAIL.
    """
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    # PLANTED AT THE WORLD THE CORPUS MEASURES, not at a round number: alpha_b
    # 0.92 is what this study's own G=1 ladders imply, and the worlds are
    # planted at the ACHIEVED rho that reproduces the measured anchor. The
    # perturbations are sized to be several times the published cross-arm alpha
    # spread and small enough not to move a cell into another regime, so a FAIL
    # is the residual gate seeing a missing term and not the grid collapsing.
    ab0 = planted_alpha_b(args.group_m)
    worlds = {
        "TRUTH": (ab0, 0.14, None),
        "MISSING": (ab0, 0.14, lambda bm, bn: 0.004 * (bm / bn) ** 2),
        "NO-A": (ab0, 0.0, None),
        "BN-DRIFT": (ab0, 0.14, lambda bm, bn: 6.0 / bn),
    }
    rho0 = achieved_rho(cfg, b, alpha_b=planted_alpha_b(args.group_m),
                        alpha_a=0.14)
    lines = ["", "## Self test: four planted worlds", "",
             "  planted at the achieved rho that reproduces the measured "
             f"anchor: {rho0:.1f} Op/B against a calibrated {ridge:.1f}; "
             f"alpha_b planted at {planted_alpha_b(args.group_m):.3f}, the "
             f"corpus value at GROUP_SIZE_M={args.group_m}", "",
             "  world      alpha_a  sd(a_a)    alpha_b   chi2 cells   C2     "
             "structure"]
    gates: list[Gate] = []
    verdicts: dict[str, tuple] = {}
    for name, (ab, aa, extra) in worlds.items():
        rho = achieved_rho(cfg, b, alpha_b=ab, alpha_a=aa)
        # THE PLANTED BANDWIDTH IS DERIVED FROM THE PLANTED RATE, NOT ASSUMED.
        # `planted_ms` charges compute at `rho x bandwidth`, so planting the
        # anchor's rho beside the card's PEAK bandwidth plants a kernel running
        # at 114% of its card -- which this file's own V2 then refuses, and the
        # self test would be testing the level gate against a world the level
        # gate is right to reject. The compute rate is planted at
        # PLANT_COMPUTE_FRACTION of the ceiling, inside the 38-64% the published
        # references reach, and the bandwidth follows from it.
        bw = PLANT_COMPUTE_FRACTION * ceiling_tflops * 1e3 / rho
        samples = planted_samples(cfg, args, alpha_b=ab, alpha_a=aa,
                                  ridge=rho, bandwidth_gbps=bw,
                                  b=b, block_ns=block_ns, subjects=subjects,
                                  extra=extra, noise=args.plant_noise,
                                  seed=args.seed)
        compiles = {(bn, bm): 1 for bn in block_ns
                    for bm in (*subjects, REFERENCE_BLOCK_M)}
        _, g, pay = analyse_run(
            samples, cfg, args, ridge=rho, bandwidth_gbps=bw,
            b=b, ceiling_tflops=ceiling_tflops,
            ceiling_source="planted", capability=capability,
            base_pinned=base, compiles=compiles, executed=dict(compiles),
            sm_count=args.sm_count or 132, block_ns=block_ns,
            subjects=subjects)
        by = {gate.name.split()[0]: gate for gate in g}
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
    gates.append(Gate(
        VALIDITY, "S4 the design resolves alpha_a",
        "at the PINNED settings, alpha_a's spread is smaller than the band C1 "
        "tests it against",
        f"sd(alpha_a) <= {ALPHA_A_SD_CEILING:.3f} in the TRUTH world",
        ceiling_ok,
        "no spread" if truth_sd is None else f"sd = {truth_sd:.4f} at "
        f"GROUP_SIZE_M={args.group_m}, {args.reps} reps",
        "C1 on the real run: the same settings will produce the same spread, "
        "so a FAIL here says the POD RUN CANNOT ANSWER P1 and should be "
        "re-pinned before it is paid for",
        ["THE LEVER IS THE SWIZZLE, and the arithmetic says why. The response "
         "moves with alpha_a as g1 (1 - alpha_b)/(1 + phi)^2, so the whole "
         "design's power is proportional to (1 - alpha_b): at "
         "GROUP_SIZE_M=1 this study measures alpha near 0.93 and the lever is "
         "worth 0.07 of its full size, while at 8 or 16 it measures 0.65-0.75 "
         "and the lever is 4x stronger. --reps buys the rest, as 1/sqrt(reps).",
         "If this gate FAILS, re-run --dry-run --group-m 16 before the pod."]))
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
                         "one production runs multi-tile; 64 and 32 are where "
                         "alpha is robustly identifiable and are what makes "
                         "the alpha_b invariance a test")
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
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0)
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
    ap.add_argument("--plant-noise", type=float, default=0.008,
                    help="lognormal sigma on every planted timing, used ONLY "
                         "by --self-test. 0.8%% is the middle of the published "
                         "across-repeat spreads (0.48%% on the A100 arms, up to "
                         "1.82%% on the H200 ones). S4's design-power verdict "
                         "is only as good as this number, so set it to what "
                         "the pod actually shows before trusting it")
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
                    help="exit non-zero unless every gate passes. Off by "
                         "default: C4 is predicted to FAIL at GROUP_SIZE_M=1 "
                         "and a falsified prediction is a result, not an error")
    return ap


def _hypothesis_ceiling(dtype: str) -> tuple[float, str]:
    """A ceiling for planning only, labelled so it can never pass for measured."""
    return (712.259 if dtype == "bf16" else 712.259,
            "HYPOTHESIS: the 2026-09-01 H200 bf16 calibration in this repo, "
            "which belongs to no attached device")


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
        return 2

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
        return 2
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

    plan = build_plan(args, cfg, b, capability, rr.ridge, bandwidth)
    block_ns = plan.block_ns
    treads = {bm: len(rs) for (bn, bm), rs in plan.rows.items()}

    detected = detect_card_slug()
    card = args.card or detected or NO_CARD_SLUG
    if args.card and detected and args.card != detected:
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return 2
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bn_decomposition" / run_id
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    lines = [
        "experiment  bn_decomposition: separate alpha_a from alpha_b, and test "
        "whether the model is complete", "",
        predictions_text(cfg, b, rr.ridge, rr.source, block_ns, subjects,
                         treads, args.group_m), "",
        "## The plan", ""]
    lines += plan.lines(cfg)
    lines += [
        f"ridge        {rr.ridge:.3f} Op/B, {rr.source}",
        f"ridge band   {rr.band[0]:.2f}-{rr.band[1]:.2f}, {rr.band_source}",
        f"bandwidth    {bandwidth:.1f} GB/s, {bw_source}",
        f"ceiling      {ceiling:.1f} TFLOP/s {args.dtype}, {ceiling_source}",
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
    lines += [
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
        return 0 if not underpowered else 2

    if args.self_test:
        more, gates = self_test(args, cfg, b, rr.ridge, bandwidth, ceiling,
                                capability, block_ns, subjects)
        print("\n".join(lines + more + ["", "## Gates", ""]
                        + render_gates(gates)))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    if underpowered:
        print("\n".join(lines))
        return 2

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        print(f"\n{missing.split('.')[0]}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --self-test  four planted worlds, checking the gates "
              "discriminate\n"
              "  --dry-run    the plan, the resource bill and the cost")
        return 2

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return 2
    if not block_ns:
        print("\n".join(lines))
        print("\nREFUSED: every BN arm was dropped by the resource bill. "
              "Nothing to measure.")
        return 2
    if hw is None:
        print("\n".join(lines))
        print("\nREFUSED: no calibration for the attached device. V2 scores "
              "every compute reference against THIS card's measured peak, and "
              "there is nothing here to accept or refuse one with. Run "
              "scripts/calibrate_hardware.py first.")
        return 2

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
            return 2
    card_path.write_text(card + "\n")

    sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
    done, samples = read_samples(csv_path)
    compiles: dict[tuple[int, int], int] = {}
    executed: dict[tuple[int, int], int] = {}
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
                               cache_root, pinned, done, samples)
        compiles[(bn, REFERENCE_BLOCK_M)], executed[(bn, REFERENCE_BLOCK_M)] = c, e
        pts, _ = collapse(samples, bn, REFERENCE_BLOCK_M)
        early_cells = [SWEEP.make_cell(cfg, n * REFERENCE_BLOCK_M,
                                       REFERENCE_BLOCK_M, ms,
                                       sm_count=sm_count, block_n=bn)
                       for n, ms in pts]
        early[bn] = qualify_reference(
            early_cells, (REFERENCE_BLOCK_M,), bn, cfg=cfg, ridge=rr.ridge,
            bandwidth_gbps=bandwidth, b=b, pinned=pinned,
            capability=capability, ceiling_tflops=ceiling, subjects=subjects)
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
        print(f"\nWARNING: only {len(usable_ns)} of {len(block_ns)} arms have "
              f"a compute branch, against the {MIN_BN_POINTS} V5 needs. The "
              "subjects are measured anyway -- the timings are the expensive "
              "part and they are worth having on disk -- but alpha_a will be "
              "UNIDENTIFIED and every claim gate will read UNKNOWN. This is "
              "the outcome P5 registered as possible.")

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
                                   csv_path, cache_root, pinned, done, samples)
            compiles[(bn, bm)], executed[(bn, bm)] = c, e
    print(f"\nmeasured in {time.time() - started:.0f} s")

    more, gates, payload = analyse_run(
        samples, cfg, args, ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
        ceiling_tflops=ceiling, ceiling_source=ceiling_source,
        capability=capability, base_pinned=base_pinned, compiles=compiles,
        executed=executed, sm_count=sm_count, block_ns=block_ns,
        subjects=subjects)
    payload["gpu"] = torch.cuda.get_device_name(0)
    payload["run_id"] = run_id
    payload["ridge_source"] = rr.source
    text = "\n".join(lines + more + ["", "## Gates", ""] + render_gates(gates))
    print("\n".join(more + ["", "## Gates", ""] + render_gates(gates)))
    (out_dir / "report.txt").write_text(text)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path), ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return 1 if (args.fail_on_gate
                 and any(g.passed is not True for g in gates)) else 0


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
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = exc.code if exc.code.startswith("REFUSED") else f"REFUSED: {exc.code}"
            print(msg, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
