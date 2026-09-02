#!/usr/bin/env python
"""Can BLOCK_SIZE_M=128 ever show FIVE clean memory-bound treads? Arithmetic first.

    python scripts/bm128_depth.py --audit        # the answer, off GPU, from published data
    python scripts/bm128_depth.py --self-test    # plant three worlds, check the gates flip
    python scripts/bm128_depth.py --dry-run      # the pod plan, the grid and the cost
    python scripts/bm128_depth.py                # the pod run

WHY THIS EXISTS. BLOCK_SIZE_M=128 is the one tile height where the study's
arithmetic-intensity cap straddles the hardware ridge -- cap 150.4 against a
calibrated ridge of 145.8 on the A100, 158.6 against 162.8 on the H200 -- and it
is the tile vLLM's fallback ladder actually runs in every multi-tile decode cell
(up to 32 M-tiles per expert among the arm's vLLM rows; the 34
sometimes quoted counts cutlass and sglang rows, which run no Triton tile).
Every other block size in
this study either sits far below the ridge (32, 64: the cap binds, and that is
the study's one surviving result) or far above it (256: compute bound at every
tread, which is why it serves as the compute reference). So 128 is the only tile
height where "does the cap matter in production" is a live question, and the
entire 128 row of the published alpha surface rests on TWO fits, one per card.

Both of those two fits are broken, and the same arithmetic explains why no third
one exists. This script is that arithmetic, the gates that would have caught the
two, and the pod run that would settle it if the arithmetic said it could be
settled. It does not.

THE MEASUREMENT THE LADDER FIT ACTUALLY MAKES. At `r = n BM` the layer's time is
`t(n) = D + max(A + B n, C n)`: a memory branch of slope `B` (one more M-tile
re-reads a fraction `alpha` of the expert's weights) and a compute branch of
slope `C` (one more M-tile does `BM` more rows of padded arithmetic). A tread is
called memory bound when it stands above `D + C n` by the margin, and
`scripts/block_m_crossing_sweep.py` then DISCARDS the whole memory branch when

    |B / C - 1| <= PARALLEL_BRANCH_TOLERANCE

because two branches within 15% of each other are one line, and a fit that reads
a stretch of the compute branch as a memory branch reports that branch's slope as
alpha. That rejection is right, and it is the thing that closes this experiment.

THE ONE IDENTITY EVERYTHING BELOW TURNS ON. Write `rho` for the ratio of the
achieved compute rate to the achieved memory rate of THIS kernel (its own ridge,
in FLOP per byte), `b` for the bytes in a weight element, `alpha` for the re-read
fraction. Then, per M-tile,

    B = (weight bytes) alpha / BW_achieved        C = (2 BM x flops/row) / peak_achieved

    B / C = alpha b rho / (2 BM) = ridge / ai_cap

and the expert count, the hidden size and the intermediate size CANCEL EXACTLY.
So `B / C` -- which decides both how many treads are memory bound and whether the
fit is allowed to speak at all -- does not depend on the model. Not on mixtral
against qwen2, not on a bigger expert, not on `--r-max`, not on how deep the
ladder is swept. That kills four of the five levers this experiment was handed.

THE TWO ESCAPE ROUTES, and both are closed at BM=128 on this hardware.

  ESCAPE UP  (B/C >= 1.15): the memory branch is the steeper line, nothing ever
      crosses, and EVERY tread is memory bound, so five of them is only a
      question of sweeping to 5 x 128 = 640 rows per expert. It needs

          alpha rho >= (1 + tol) 2 BM / b = 1.15 x 128 = 147.2 FLOP/byte

      Across the 22 published BM=128 ladders with a compute reference that
      survives a LEVEL check, measured `alpha rho` runs 101.8 to 150.4, median
      126.8 -- so the median has to move +16.1% to arrive. TWO ladders clear
      147.2 and NEITHER survives its own diagnostics: the A100 one runs
      backwards at its last tread, and the H200 one has a single memory tread
      and a per-tread slope of 1.01, 1.05, 1.01, 0.97, 1.37, 1.62, 1.31, which
      rises and then falls and so is not describable by any two-line model. On
      the A100 the route is additionally shut by physics rather than by margin:
      alpha would have to be >= 1.010 against that card's calibrated ridge of
      145.8, and alpha > 1 means an extra M-tile costs MORE than reading the
      whole expert once.

  ESCAPE DOWN (B/C <= 0.85): the memory branch is the shallower line, so it is a
      finite PREFIX, and the prefix length is

          n* = [ratio (1 - alpha) / alpha - D/C] / (1 + margin - ratio)

      Maximised under the tolerance constraint, `n` clean treads need
      `rho >= (1 + tol (n - 1)) 2 BM / b`, which at n = 5 is 1.6 x 128 = 204.8
      FLOP/byte against calibrated ridges of 145.8 (A100) and 162.8 (H200). But
      the same constraint has a form that needs NO CARD AT ALL:

          alpha <= (1 - tol) / [(1 - tol) + n (tol + margin)] = 0.85 / 1.70 = 0.500

      charging zero overhead, so 0.500 is a ceiling and not an estimate. The
      lowest activation-corrected alpha this study has measured at ANY block
      size, on either card, in any of 23 arms, is 0.596. Escape down is shut by
      numbers the study already published, with no ridge and no bandwidth in the
      argument.

WHAT IS LEFT IN BETWEEN is the discard band, and BM=128 sits in it: 19 of the 22
valid published ladders have `B/C` between 0.877 and 1.101, and the median is
0.991. At BLOCK_M=128 the two branches ARE the same line, to about 1%. That is
not a near miss to be tuned away -- it is what "the cap sits on the ridge" means,
measured. The levers move the median `B/C` by 3.2% (BLOCK_SIZE_N), 7.5%
(GROUP_SIZE_M) and 8.5% (card); the model moves it 3.4% in the median of eight
same-session mixtral/qwen2 pairs, against the 6.4x a "bigger weights to re-read"
mechanism would need. The requirement is +16.1%, in a direction no lever points.

SO THE ANSWER IS NO, and the deliverable is the arithmetic plus two gates the
study did not have:

  * MONOTONICITY. The A100 fit that the whole 128 row rests on runs BACKWARDS at
    its last tread: 25.8076 ms at 7 tiles, 25.4883 ms at 8. Time falling as rows
    rise is not a mechanism. That single tread is also the ONLY point on that
    ladder's compute branch, so it alone sets the `C` the memory branch is
    compared against.
  * TOLERANCE MARGIN. That comparison then clears the 15% tolerance by 1.0e-4:
    |B/C - 1| = 0.150101. Raise that one tread by 0.010% -- 25.4883 ms to
    25.4909 ms, 0.0026 ms -- and `memory_points` goes from 7 to 0, the memory
    branch is discarded, and the A100's only BLOCK_M=128 alpha ceases to exist.
    A verdict that survives on 0.0026 ms must not print the same word as one
    that survives on 0.3, so every margin here is reported as a number, as a
    fraction of the tolerance, and in units of its own bootstrap spread.

Both gates are run against every published BM=128 ladder by `--audit`, which
needs no GPU and is the evidence for everything above.

WHAT THE POD RUN ADDS IF IT IS RUN ANYWAY. `--dry-run` prints it: the BM=256
reference ladder FIRST (so a poisoned reference costs four cells and not the
whole run), a level check on it before the subject is measured at all, then the
BM=128 ladder measured `--reps` times in round-robin so that an inversion can be
told from noise -- which one pass cannot do, and which is exactly how the A100
ladder shipped. Nothing here can raise `B/C`; the run exists to measure the
margin honestly and to record the refusal.

A NOTE ON REUSE. The ladder fit, the compute reference and the tolerance are
IMPORTED from `scripts/block_m_crossing_sweep.py`, never copied and never
edited: this script has to be judged by the same fit the study publishes, and a
private copy would drift. `compute_reference` grew four required keyword-only
arguments while this file was being written, so `_load_sweep` probes SIGNATURES
and not only names: a rename must produce a sentence on a laptop rather than a
TypeError thirty seconds into a metered pod session.

`reference_level` here OVERLAPS that file's own level checks and is kept
deliberately, so do not delete it as a duplicate. The two do different jobs.
Theirs REFUSES a bad reference at qualification time and is what protects the pod
run; this one REPORTS the implied achieved TFLOP/s as a number and a fraction of
the card's ceiling, which is what `--audit` needs to score the 26 already-published
reports, whose references were qualified before any level test existed and cannot
be re-refused retrospectively. Theirs is three readings and strictly stronger on
live data; this one is one reading and works on a JSON file. If they are ever
merged, the audit path is the one that has to keep working.
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

from moe.bench.roofline import HARDWARE_DIR, load_hardware  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402


def _load_sweep():
    """Load `block_m_crossing_sweep` BY PATH, and name what is missing.

    `scripts/` is not a package, so a bare import works only when this file is
    the entry point and silently fails when a test loads it by path. Loading by
    path makes both work. The symbol check is not defensive noise: that file is
    under active edit by another workstream, and a renamed `fit_ladder` must
    produce a sentence naming it rather than an AttributeError three frames
    down inside a gate.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    needed = ("PARALLEL_BRANCH_TOLERANCE", "MEMORY_BRANCH_MARGIN",
              "MIN_MEMORY_TREADS", "FIXED", "ComputeReference", "fit_ladder",
              "ladder_points", "compute_reference", "make_cell", "model_ms",
              "tokens_for_rows", "rows_quantum", "results_root", "scaled_iters",
              "activation_slope_ms", "useful_flops", "_line")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/block_m_crossing_sweep.py no longer exports "
            f"{', '.join(missing)}. This script is deliberately scored by that "
            "file's fit rather than a private copy, so the two move together. "
            "Re-point the import; do not fork the fit.")
    # SIGNATURES, not only names. `compute_reference` grew four required
    # keyword-only arguments while this file was being written, and an existence
    # check passed it straight through to a TypeError raised thirty seconds into
    # a metered pod session. Everything callable from the GPU path is probed here
    # instead, on a laptop, with a sentence that names the drift.
    import inspect
    for name, required in (("compute_reference", ("cfg", "ridge",
                                                  "bandwidth_gbps", "b")),
                           ("fit_ladder", ("block_m",)),
                           ("make_cell", ("sm_count", "block_n"))):
        params = inspect.signature(getattr(module, name)).parameters
        gone = [p for p in required if p not in params]
        if gone:
            raise SystemExit(
                f"block_m_crossing_sweep.{name} no longer takes "
                f"{', '.join(gone)}. That file is under active edit and this "
                "one calls into it on the pod path; re-check the call sites in "
                "analyse_run and main before spending GPU time.")
    return module


SWEEP = _load_sweep()

TOLERANCE = SWEEP.PARALLEL_BRANCH_TOLERANCE
MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS


# --------------------------------------------------------------------------
# The thresholds this script is arguing about, all stated before any code.
# --------------------------------------------------------------------------

#: The subject. Not a parameter: every sentence in the docstring is about 128,
#: and a `--block-m` flag would let a run answer a different question under this
#: script's name.
SUBJECT_BLOCK_M = 128

#: The compute reference. `C ~ BLOCK_M` with no free parameter, so one ladder
#: that is compute bound throughout gives `C` at every block size. 256 is the
#: only block size this study has ever found compute bound at tread 1.
REFERENCE_BLOCK_M = 256

#: The ask: five clean memory-bound treads at the subject block size.
TARGET_TREADS = 5

#: A compute reference must imply at least this fraction of the ATTACHED card's
#: calibrated bf16 ceiling.
#:
#: THE GATE THE STUDY DID NOT HAVE. `compute_reference` qualifies a ladder by
#: PROPORTIONALITY to its tile count and never by LEVEL, and a line 43.6x too
#: steep is still perfectly proportional. The A100 BLOCK_N=256 arm's reference
#: took 249.765 ms for one BLOCK_M=256 tile against 5.724 ms for the identical
#: setting in its BLOCK_N=64 twin; it implies 3.6 TFLOP/s, 1.4% of that card's
#: 262.4, and it classified every tread of every ladder in the arm as compute
#: bound. Its H200 twin implies 87.7 TFLOP/s, 12.3%. The 22 references that are
#: not corrupt imply 38.2% to 63.7%, so 25% separates them with room on both
#: sides and is not fitted to the failures.
REFERENCE_LEVEL_FLOOR = 0.25

#: An inversion bigger than this many across-repeat standard deviations is a
#: fault and not noise. Two sigma rather than three because the direction is
#: known a priori: `t(n)` is `D + max(A + Bn, Cn)` with `B, C > 0`, so time
#: cannot fall as tiles rise under EITHER branch, and a one-sided departure
#: needs less evidence than a two-sided one.
MONOTONE_SIGMA = 2.0

#: A tolerance verdict has to clear the tolerance by this many bootstrap
#: standard deviations before it may be read as a verdict. The published A100
#: fit clears by 0.000101, which is 0.07% of the tolerance and 0.02 sigma.
MARGIN_SIGMA = 3.0

#: Resamples used to put a spread on `|B/C - 1|`. Fixed rather than tuned, and
#: seeded, so two readers of the same ladder get the same margin.
BOOTSTRAP_DRAWS = 2000

#: `max(a + bn, D + cn)` is CONVEX, so its per-tread slope is non-decreasing and
#: takes at most two values. A slope that rises and then falls is not describable
#: by any two-line model -- but a per-tread slope is a DIFFERENCE of two timings,
#: so its noise is amplified by roughly `sqrt(2) n` over the ladder's own spread
#: and reaches 15% by tread 7 on the noisier H200 arms.
#:
#: A FIXED RELATIVE THRESHOLD IS THEREFORE NOT A GATE. At 5% this fired on 20 of
#: the 22 published ladders, which is a detector that has learned to say yes. The
#: threshold is propagated from the run's own timing spread instead, and a drop
#: has to clear BOTH this many standard deviations and the relative floor below.
SLOPE_DROP_SIGMA = 3.0

#: A drop inside this fraction is not worth reporting however small the spread
#: is, because the two-line model's own two slopes differ by more than it.
SLOPE_DROP_FLOOR = 0.05

#: Across-repeat spread above this makes a margin unreadable, whatever it is.
#: The published H200 ladders sit at 0.76-1.82% on ONE pass, the A100 ones at
#: 0.48-0.61%; 2% is a ceiling on the noise a replicated run is allowed to have
#: before its verdict is withdrawn rather than reported.
MAX_REPLICATE_SPREAD = 0.02

#: Calibration files, by the substring that identifies the card in a published
#: arm's directory name. Read rather than hardcoded: the A100's ridge is
#: 262.371/1.79936 = 145.81 and the H200's is 712.259/4.37476 = 162.81, and the
#: seven published A100 reports carry 160.3 -- an H200 band belonging to neither
#: card -- precisely because a number like that was written down once.
CALIBRATION_SLUGS = {"a100": "measured_nvidia_a100_sxm4_80gb",
                     "h200": "measured_nvidia_h200"}


# --------------------------------------------------------------------------
# The depth law. Pure arithmetic: no torch, no GPU, no files.
# --------------------------------------------------------------------------

def branch_ratio(block_m: int, b: int, alpha: float, rho: float) -> float:
    """`B / C = alpha b rho / (2 BM)`, the fitted-slope ratio.

    Also `ridge / ai_cap`, which is why it decides everything: below 1 the
    compute branch is steeper and the memory branch is a finite prefix, above 1
    nothing ever crosses, and within `TOLERANCE` of 1 the fit refuses to call
    them two branches at all. The model's E, F and H are not arguments because
    they cancel; see the module docstring.
    """
    return alpha * b * rho / (2.0 * block_m)


def escape_up_alpha_rho(block_m: int, b: int, tol: float = TOLERANCE) -> float:
    """`alpha rho` needed for the memory branch to be the steeper line.

    At or above this the ladder never crosses, every tread is memory bound, and
    depth is bounded only by `--r-max`. 147.2 FLOP/byte at BLOCK_M=128, bf16.
    """
    return (1.0 + tol) * 2.0 * block_m / b


def escape_down_rho(block_m: int, b: int, treads: int,
                    tol: float = TOLERANCE) -> float:
    """Achieved ridge needed for `treads` clean memory treads BELOW the crossing.

    In the prefix regime a tread is memory bound while `n < n*` with
    `n* = rho (1 - alpha) / (2 BM / b - alpha rho)`. `n*` rises with `alpha rho`,
    which the tolerance caps at `(1 - tol) 2 BM / b`; substituting that cap and
    solving `n* >= treads` gives

        rho >= (1 + tol (treads - 1)) 2 BM / b

    which is 204.8 FLOP/byte for five treads at BLOCK_M=128, bf16. The two cards
    in this study calibrate at 145.8 and 162.8.
    """
    return (1.0 + tol * (treads - 1)) * 2.0 * block_m / b


def prefix_depth(ratio: float, alpha: float, overhead_over_c: float,
                 margin: float) -> float | None:
    """How many leading treads stand above the compute branch, from the fit alone.

    A tread is memory bound when `A + B n > D + C n (1 + margin)`. With
    `A = B (1 - alpha) / alpha` and `ratio = B / C` that is

        n < [ratio (1 - alpha) / alpha - D / C] / (1 + margin - ratio)

    None means EVERY tread qualifies, which is a different answer from "many"
    and is what `ratio >= 1 + margin` says. Returns a float on purpose: the
    integer count is `floor` of it, and rounding here would hide how close a
    ladder sits to gaining or losing a tread.
    """
    denom = 1.0 + margin - ratio
    if denom <= 0:
        return None
    if alpha <= 0:
        raise ValueError("alpha must be positive to have a memory branch at all")
    return (ratio * (1.0 - alpha) / alpha - overhead_over_c) / denom


def escape_down_alpha(treads: int, tol: float = TOLERANCE,
                      margin: float = SWEEP.MEMORY_BRANCH_MARGIN) -> float:
    """The largest `alpha` that can give `treads` clean treads below the crossing.

    THE SAME BOUND AS `escape_down_rho`, RESTATED IN THE ONE QUANTITY THIS STUDY
    ACTUALLY MEASURES. `n*` rises with `ratio`, and the tolerance caps `ratio` at
    `1 - tol`, so substituting that cap and dropping the (non-negative) overhead
    term gives the most generous possible reading:

        (1 - tol)(1 - alpha) / alpha >= treads (tol + margin)
        alpha <= (1 - tol) / [(1 - tol) + treads (tol + margin)]

    At five treads that is 0.85 / 1.70 = 0.500. It needs no ridge, no bandwidth
    and no card: the study's own alphas settle it, and the lowest alpha it has
    measured anywhere is 0.596. Charging any overhead at all only lowers the
    bound further, so 0.500 is a ceiling and not an estimate.

    Independent of BLOCK_M, which is the surprising part and is correct: BLOCK_M
    moves `ratio`, and this bound is evaluated at the one `ratio` the tolerance
    allows, so the tile height has already been used up.
    """
    return (1.0 - tol) / ((1.0 - tol) + treads * (tol + margin))


@dataclass(frozen=True)
class DepthVerdict:
    """What the law says about one (block_m, b, alpha, rho) point."""

    block_m: int
    dtype_bytes: int
    alpha: float
    rho: float
    ratio: float
    regime: str
    reachable_treads: float | None
    needed_alpha_rho: float
    needed_rho: float

    @property
    def feasible(self) -> bool:
        return (self.regime == "escape-up"
                or (self.regime == "escape-down"
                    and self.reachable_treads is not None
                    and self.reachable_treads >= TARGET_TREADS))

    def line(self) -> str:
        depth = ("every tread" if self.reachable_treads is None
                 else f"{self.reachable_treads:.2f} treads")
        return (f"alpha {self.alpha:.3f} x rho {self.rho:7.1f} -> "
                f"B/C {self.ratio:.3f}  {self.regime:12s}  {depth:>13s}   "
                f"{'FEASIBLE' if self.feasible else 'no'}")


def depth_verdict(block_m: int, b: int, alpha: float, rho: float, *,
                  overhead_over_c: float = 0.0,
                  margin: float = SWEEP.MEMORY_BRANCH_MARGIN,
                  treads: int = TARGET_TREADS) -> DepthVerdict:
    """The law, evaluated. Three regimes, and the middle one is where 128 lives."""
    ratio = branch_ratio(block_m, b, alpha, rho)
    if abs(ratio - 1.0) <= TOLERANCE:
        # Not "few treads": the fit throws the memory branch away entirely, so
        # the ladder reports NO alpha however many treads stood above the line.
        return DepthVerdict(block_m, b, alpha, rho, ratio, "discarded", 0.0,
                            escape_up_alpha_rho(block_m, b),
                            escape_down_rho(block_m, b, treads))
    if ratio > 1.0:
        return DepthVerdict(block_m, b, alpha, rho, ratio, "escape-up", None,
                            escape_up_alpha_rho(block_m, b),
                            escape_down_rho(block_m, b, treads))
    return DepthVerdict(block_m, b, alpha, rho, ratio, "escape-down",
                        prefix_depth(ratio, alpha, overhead_over_c, margin),
                        escape_up_alpha_rho(block_m, b),
                        escape_down_rho(block_m, b, treads))


# --------------------------------------------------------------------------
# Ladder diagnostics. Everything here takes points and returns a NUMBER; none
# of it decides anything on its own, so a reader can check each separately.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Inversion:
    """One place where time fell as tiles rose."""

    n_lo: int
    n_hi: int
    ms_lo: float
    ms_hi: float
    rel: float
    sigma: float | None

    def line(self) -> str:
        sig = "spread unknown" if self.sigma is None else f"{self.sigma:.1f} sigma"
        return (f"n={self.n_lo}->{self.n_hi}  {self.ms_lo:9.4f} -> "
                f"{self.ms_hi:9.4f} ms  {self.rel:+.3%}  ({sig})")


def inversions(points, spread: float | None) -> list[Inversion]:
    """Every tread whose time is below its predecessor's.

    Under `t(n) = D + max(A + Bn, Cn)` with both slopes positive, time is
    strictly increasing in `n` on BOTH branches. So an inversion is never a
    mechanism; it is a clock, a thermal ramp, a cache that was warm for one
    point and not the next, or a single unlucky median. It matters because OLS
    hands the last tread the most leverage on the slope, and the slope IS alpha:
    on the published A100 ladder the inverted tread is also the only point on
    the compute branch, so it sets both lines at once.

    `spread` is the relative timing spread the run measured. None is carried
    through as None rather than replaced by a guess, because "we do not know how
    noisy this was" and "this was 0% noisy" are different states and only one of
    them lets an inversion be dismissed.
    """
    out = []
    for (n_lo, lo), (n_hi, hi) in zip(points, points[1:], strict=False):
        if hi >= lo:
            continue
        rel = hi / lo - 1.0
        out.append(Inversion(n_lo, n_hi, lo, hi, rel,
                             abs(rel) / spread if spread else None))
    return out


def slope_sequence(points) -> list[float]:
    """Per-tread marginal cost, `(t(n+1) - t(n)) / (n+1 - n)`."""
    return [(hi - lo) / (n_hi - n_lo)
            for (n_lo, lo), (n_hi, hi) in zip(points, points[1:], strict=False)]


@dataclass(frozen=True)
class SlopeDrop:
    index: int
    before: float
    after: float
    rel: float
    sigma: float | None

    def line(self) -> str:
        sig = "spread unknown" if self.sigma is None else f"{self.sigma:.1f} sigma"
        return (f"slope {self.before:.4f} -> {self.after:.4f} ms/tile after "
                f"tread {self.index + 2}  {self.rel:+.1%}  ({sig})")


def _slope_drop_sd(points, i: int, spread: float) -> float:
    """Propagated sd of `slope[i+1] - slope[i]` under multiplicative timing noise.

    Three timings are involved and the middle one appears in BOTH slopes with
    opposite sign, so its variance ADDS rather than cancelling. Writing
    `d1 = n[i+1] - n[i]` and `d2 = n[i+2] - n[i+1]`,

        Var = s^2 [ (t0^2 + t1^2)/d1^2 + (t1^2 + t2^2)/d2^2 + 2 t1^2/(d1 d2) ]

    Getting the covariance term wrong understates this by about a third, which
    is the difference between a gate that fires on one ladder and one that fires
    on twenty.
    """
    (n0, t0), (n1, t1), (n2, t2) = points[i], points[i + 1], points[i + 2]
    d1, d2 = float(n1 - n0), float(n2 - n1)
    var = spread ** 2 * ((t0 ** 2 + t1 ** 2) / d1 ** 2
                         + (t1 ** 2 + t2 ** 2) / d2 ** 2
                         + 2.0 * t1 ** 2 / (d1 * d2))
    return math.sqrt(var)


def slope_drops(points, spread: float | None,
                sigma_gate: float = SLOPE_DROP_SIGMA,
                floor: float = SLOPE_DROP_FLOOR) -> list[SlopeDrop]:
    """Where the ladder stops being convex, which max-affine cannot be.

    `max(A + Bn, D + Cn)` is a maximum of two increasing lines, so its slope is
    non-decreasing and takes at most two values. A slope that RISES and then
    FALLS is not that shape at all, and a two-line fit through it reports the
    slope of a curve rather than of a branch.

    A drop must clear the relative floor AND the propagated noise. With `spread`
    None there is no noise model, so every drop over the floor is returned with
    `sigma=None` and the gate reads UNKNOWN rather than passing.

    WHAT THIS GATE CANNOT DO ON THE PUBLISHED H200 LADDERS, said here because the
    obvious reading of it is wrong. H200 mixtral at GROUP_SIZE_M=64 runs
    1.01, 1.05, 1.01, 0.97, 1.37, 1.62, 1.31 ms per tile -- a 19.3% fall -- and
    this gate does NOT fire on it, because at that arm's 1.71% timing spread the
    propagated sd of a slope difference at tread 7 is 0.34 ms/tile against a
    slope of 1.3, so 19.3% is 0.9 sigma. That is not the gate being lenient; it
    is the measurement being unable to resolve a per-tread slope at all. A
    single-pass BLOCK_M=128 ladder on the H200 cannot distinguish a bend from
    noise, which is the same reason it cannot distinguish two branches, and it is
    why `--reps` exists.

    The spread used is the run's WITHIN-call spread, which is smaller than the
    across-pass spread a replicated run would report, so every sigma here is an
    over-estimate and this gate is conservative in the direction of not firing.
    """
    slopes = slope_sequence(points)
    out = []
    for i, (a, c) in enumerate(zip(slopes, slopes[1:], strict=False)):
        if a <= 0 or c >= a * (1.0 - floor):
            continue
        sd = _slope_drop_sd(points, i, spread) if spread else None
        sig = abs(c - a) / sd if sd and sd > 0 else None
        if sig is not None and sig < sigma_gate:
            continue
        out.append(SlopeDrop(i, a, c, c / a - 1.0, sig))
    return out


@dataclass(frozen=True)
class Margin:
    """How far a memory branch stands from being the compute branch again."""

    ratio: float
    distance: float
    margin: float
    sd: float | None
    sigma: float | None
    #: 'replicates' or 'parametric spread s=...'. Never empty: a margin whose
    #: noise model is unstated is a margin nobody can weigh.
    basis: str
    draws: int

    @property
    def clears(self) -> bool:
        return self.margin > 0

    @property
    def confident(self) -> bool:
        return self.clears and self.sigma is not None and self.sigma >= MARGIN_SIGMA

    def line(self) -> str:
        sig = "sigma unknown" if self.sigma is None else f"{self.sigma:.2f} sigma"
        return (f"B/C {self.ratio:.6f}  |B/C - 1| {self.distance:.6f}  "
                f"tolerance {TOLERANCE:.2f}  margin {self.margin:+.6f} "
                f"({self.margin / TOLERANCE:+.2%} of the tolerance, {sig})")


def _refit_ratio(points, k: int, compute_points, c_ref: float | None,
                 overhead: float) -> float | None:
    """`B / C` for a FIXED branch assignment, which is what a bootstrap needs.

    Membership is held at `k` across resamples on purpose. Letting it move would
    make the resampled quantity a mixture of "this slope under noise" and "a
    different set of treads", and the margin is a statement about the first.
    """
    if k < 2:
        return None
    _, b = SWEEP._line([float(n) for n, _ in points[:k]],
                       [ms for _, ms in points[:k]])
    c = c_ref
    if compute_points:
        c = SWEEP._through_origin([float(n) for n, _ in compute_points],
                                  [ms - overhead for _, ms in compute_points])
    if not c:
        return None
    return b / c


def margin_of(points, k: int, *, c_ref: float | None, overhead: float,
              spread: float | None = None,
              replicates: dict[int, list[float]] | None = None,
              draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> Margin:
    """`|B/C - 1| - tolerance`, with a spread on it.

    REFUSES rather than returning a bare pass. With neither replicates nor a
    measured spread there is no noise model, `sd` and `sigma` come back None,
    and the gate that reads this reports UNKNOWN. Returning 0.0 for an
    unmeasured spread would turn every knife-edge verdict into a confident one,
    which is the exact failure this whole file is about.

    With replicates the resample is nonparametric: each tread's value is drawn
    from its own repeated measurements. Without them it is parametric --
    lognormal at the run's own median relative spread -- and `basis` says so,
    because a parametric interval on one pass cannot see a drift that repeated
    passes would have caught.
    """
    compute_points = points[k:]
    ratio = _refit_ratio(points, k, compute_points, c_ref, overhead)
    if ratio is None:
        return Margin(float("nan"), float("nan"), float("nan"), None, None,
                      "no memory branch: fewer than 2 treads above the compute "
                      "branch, so there is no slope to compare", 0)
    distance = abs(ratio - 1.0)
    base = Margin(ratio, distance, distance - TOLERANCE, None, None, "", 0)

    if replicates:
        basis = f"replicates ({min(len(v) for v in replicates.values())}+ per tread)"
    elif spread and spread > 0:
        basis = f"parametric spread sigma={spread:.4f} (ONE pass, no replicates)"
    else:
        return Margin(ratio, distance, distance - TOLERANCE, None, None,
                      "no noise model: the run recorded neither replicates nor a "
                      "timing spread, so this margin cannot be weighed", 0)

    rng = random.Random(seed)
    dists = []
    for _ in range(draws):
        drawn = []
        for n, ms in points:
            if replicates and replicates.get(n):
                drawn.append((n, rng.choice(replicates[n])))
            else:
                drawn.append((n, ms * math.exp(rng.gauss(0.0, spread or 0.0))))
        # The reference slope carries the same noise as the ladder, so it is
        # resampled too. Holding it fixed understates the spread and would make
        # the margin gate more lenient than the data supports.
        c = c_ref * math.exp(rng.gauss(0.0, spread or 0.0)) if c_ref else None
        r = _refit_ratio(drawn, k, drawn[k:], c, overhead)
        if r is not None:
            dists.append(abs(r - 1.0))
    if len(dists) < 2:
        return Margin(ratio, distance, base.margin, None, None,
                      basis + "; every resample lost the branch, so no spread "
                              "could be formed", len(dists))
    sd = statistics.pstdev(dists)
    sigma = (distance - TOLERANCE) / sd if sd > 0 else None
    return Margin(ratio, distance, distance - TOLERANCE, sd, sigma, basis,
                  len(dists))


@dataclass(frozen=True)
class RefLevel:
    """Is the compute reference a compute branch, or just a straight slow line."""

    block_m: int
    slope_ms_per_tile: float
    implied_tflops: float
    ceiling_tflops: float
    fraction: float
    source: str

    @property
    def passes(self) -> bool:
        return REFERENCE_LEVEL_FLOOR <= self.fraction <= 1.0

    def line(self) -> str:
        return (f"BLOCK_M={self.block_m} reference {self.slope_ms_per_tile:.4f} "
                f"ms/tile implies {self.implied_tflops:7.1f} TFLOP/s = "
                f"{self.fraction:6.1%} of {self.ceiling_tflops:.1f} "
                f"({self.source})")


def reference_level(cfg, block_m: int, slope_ms_per_tile: float,
                    ceiling_tflops: float, source: str) -> RefLevel:
    """Turn a reference slope into achieved TFLOP/s and compare it to the card.

    One M-tile per expert at `block_m` rows is `E BM` padded rows and
    `6 E BM F H` flops, so the slope names an achieved rate directly. The whole
    point is that this is a LEVEL and not a shape: `compute_reference` already
    checks that the reference ladder is proportional to its tile count, and the
    A100 BLOCK_N=256 reference was proportional to 0.2% while being 43.6x too
    slow.
    """
    flops = SWEEP.useful_flops(cfg, cfg.num_experts * block_m)
    implied = flops / (slope_ms_per_tile * 1e-3) / 1e12
    return RefLevel(block_m, slope_ms_per_tile, implied, ceiling_tflops,
                    implied / ceiling_tflops if ceiling_tflops > 0 else float("inf"),
                    source)


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

VALIDITY, CLAIM = "VALIDITY", "CLAIM"


@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` prints UNKNOWN and never PASS. `invalidates` is required on a
    VALIDITY gate and says what may not be quoted if it fails, because a failed
    gate whose consequence is unstated gets read as a warning.
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
            out.append(f"         a FAIL here invalidates: {self.invalidates}")
        out += [f"         {line}" for line in self.lines]
        return out


def render_gates(gates: list[Gate]) -> list[str]:
    out: list[str] = []
    for g in gates:
        out += g.render()
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    out += ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN"]
    return out


def gate_reference_level(level: RefLevel | None) -> Gate:
    if level is None:
        return Gate(VALIDITY, "V1 reference level",
                    "the compute reference runs at a plausible rate",
                    f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of "
                    "the attached card's calibrated bf16 ceiling",
                    None, "no compute reference was qualified at all",
                    "every membership decision, hence every alpha and every "
                    "tread count in this report")
    return Gate(VALIDITY, "V1 reference level",
                "the compute reference runs at a plausible rate",
                f"implied TFLOP/s in [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of the "
                "card's calibrated bf16 ceiling",
                level.passes, level.line(),
                "every membership decision, hence every alpha and every tread "
                "count in this report: a reference too slow puts its line above "
                "every tread and calls the whole ladder compute bound",
                ["`compute_reference` tests PROPORTIONALITY and never LEVEL; a "
                 "line 43.6x too steep is still perfectly proportional."])


def gate_monotone(found: list[Inversion], boundaries: int,
                  spread: float | None) -> Gate:
    bad = [i for i in found if i.sigma is None or i.sigma >= MONOTONE_SIGMA]
    if boundaries <= 0:
        return Gate(VALIDITY, "V2 monotone ladder",
                    "time rises with every extra M-tile",
                    f"zero inversions beyond {MONOTONE_SIGMA:.0f} sigma",
                    None, "the ladder had fewer than two treads, so no boundary "
                          "was examined",
                    "the fitted slopes, which are what alpha is")
    obs = (f"{boundaries} tread boundaries examined, {len(found)} inversions, "
           f"{len(bad)} beyond {MONOTONE_SIGMA:.0f} sigma"
           + (f" (spread {spread:.3%})" if spread else " (spread unknown)"))
    return Gate(VALIDITY, "V2 monotone ladder",
                "time rises with every extra M-tile",
                f"zero inversions beyond {MONOTONE_SIGMA:.0f} sigma",
                not bad, obs,
                "the fitted slopes, which are what alpha is: OLS gives the last "
                "tread the most leverage, and on the published A100 ladder the "
                "inverted tread is also the only point on the compute branch",
                [i.line() for i in found])


def gate_convex(drops: list[SlopeDrop], slopes: list[float],
                spread: float | None) -> Gate:
    rule = (f"zero slope drops over {SLOPE_DROP_FLOOR:.0%} and "
            f"{SLOPE_DROP_SIGMA:.0f} sigma")
    invalidates = ("the two-line fit: max(A+Bn, D+Cn) is convex, so a slope "
                   "that rises then falls is not describable by ANY two-line "
                   "model and its fitted B/C is the slope of a bend")
    if len(slopes) < 2:
        return Gate(VALIDITY, "V3 max-affine shape",
                    "the per-tread slope never rises and then falls", rule,
                    None, "fewer than three treads, so no second difference "
                          "exists", invalidates)
    if spread is None:
        return Gate(VALIDITY, "V3 max-affine shape",
                    "the per-tread slope never rises and then falls", rule,
                    None, f"{len(drops)} drops over {SLOPE_DROP_FLOOR:.0%}, but "
                          "no timing spread to weigh them against",
                    invalidates, [d.line() for d in drops])
    return Gate(VALIDITY, "V3 max-affine shape",
                "the per-tread slope never rises and then falls", rule,
                not drops,
                f"{len(slopes)} per-tread slopes at spread {spread:.3%}, "
                f"{len(drops)} drops clearing both bars: "
                + " ".join(f"{s:.3f}" for s in slopes),
                invalidates, [d.line() for d in drops])


def gate_depth(memory_treads: int, ladder_treads: int) -> Gate:
    return Gate(CLAIM, "C1 depth",
                f"at least {TARGET_TREADS} clean memory-bound treads at "
                f"BLOCK_M={SUBJECT_BLOCK_M}",
                f">= {TARGET_TREADS} treads standing above the compute branch",
                memory_treads >= TARGET_TREADS,
                f"{memory_treads} of {ladder_treads} treads above the compute "
                f"branch (a verdict also needs {MIN_MEMORY_TREADS})",
                lines=["A FAIL is the registered expectation and is a result: "
                       "see the escape thresholds above."])


def gate_margin(margin: Margin) -> Gate:
    if margin.sigma is None:
        return Gate(CLAIM, "C2 tolerance margin",
                    "the memory branch is not the compute branch again",
                    f"|B/C - 1| - {TOLERANCE:.2f} >= {MARGIN_SIGMA:.0f} sd",
                    None, margin.basis if math.isnan(margin.ratio) else margin.line(),
                    lines=[margin.basis])
    return Gate(CLAIM, "C2 tolerance margin",
                "the memory branch is not the compute branch again",
                f"|B/C - 1| - {TOLERANCE:.2f} >= {MARGIN_SIGMA:.0f} sd",
                margin.confident, margin.line(),
                lines=[f"noise model: {margin.basis}, {margin.draws} resamples, "
                       f"sd {margin.sd:.6f}",
                       "clearing the tolerance is not enough: the published A100 "
                       "fit clears by 0.000101, which is 0.07% of the tolerance "
                       "and 0.02 sd, and a 0.010% change in ONE tread reverses "
                       "it."])


def gate_non_vacuity(counts: dict[str, int]) -> Gate:
    """A check that examined nothing also reports zero failures.

    Every gate above can pass by having no data. This one asserts the data
    existed, and it names the count so a reader can see WHICH work happened
    rather than trusting that some did.
    """
    empty = sorted(k for k, v in counts.items() if v <= 0)
    return Gate(VALIDITY, "V0 non-vacuity", "this report examined real work",
                "every counted quantity is above zero",
                not empty,
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "every gate in this report: a check with no input reports no "
                "failures",
                [f"nothing was counted for: {', '.join(empty)}"] if empty else [])


# --------------------------------------------------------------------------
# The registered predictions, printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def predictions_text(b: int = 2) -> str:
    up = escape_up_alpha_rho(SUBJECT_BLOCK_M, b)
    down = escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS)
    return f"""\
## Predictions, registered before anything is measured

THE LAW (arithmetic, not a prediction; check it, do not test it)
  B/C = alpha b rho / (2 BM) = ridge / ai_cap, and E, F, H cancel exactly.
  escape UP   needs alpha x rho >= {up:.1f} FLOP/byte  -> every tread memory bound
  escape DOWN needs rho        >= {down:.1f} FLOP/byte  -> {TARGET_TREADS} treads below the crossing
    equivalently alpha <= {escape_down_alpha(TARGET_TREADS):.3f}, a bound needing no card
  in between, |B/C - 1| <= {TOLERANCE:.2f}, the fit discards the memory branch entirely.

P1  A NEW BM=128 ladder lands at B/C in [0.94, 1.10].
    22 published ladders with a valid reference: 0.795 to 1.175, median 0.991,
    sd 0.078. FAIL means B/C at 128 is not stationary across arms and the
    feasibility argument below rests on a quantity that moves.
P2  C1 FAILS: fewer than {TARGET_TREADS} clean memory-bound treads.
    Escape up needs {up:.1f} and the corpus tops out at 150.4, on a ladder whose
    slope rises then falls. Escape down needs rho >= {down:.1f} and the highest
    achieved rho on any published ladder is 166.5, against calibrated ridges of
    145.8 (A100) and 162.8 (H200). FAIL here is the good outcome and would mean
    the cap CAN be measured at the tile vLLM actually runs.
P3  C2 FAILS: no BM=128 fit clears the tolerance by {MARGIN_SIGMA:.0f} sd.
P4  EXPERT SIZE does not enter B/C. mixtral's per-expert weight is 6.4x qwen2's
    (3 F H = 176.2 M elements against 27.5 M), so a "bigger weights to re-read"
    mechanism predicts B/C scaling like 6.4. The identity predicts NO size term
    at all: the model can still move B/C, but only by moving alpha or rho, which
    it does by tens of percent and not by factors. Gate: at matched card,
    GROUP_SIZE_M and BLOCK_N, |B/C(mixtral) / B/C(qwen2) - 1| < 0.20 in the
    median. FAIL at 6.4 means the identity is wrong and the feasibility argument
    goes with it; FAIL at 0.3 means alpha differs by more between models than
    this bound allows and the bound was too tight.
P5  --r-max does not move the memory-tread COUNT in the prefix regime: n* has no
    r_max in it. Sweeping deeper adds treads to measure, not treads that qualify.
    UNKNOWN in --audit unless the corpus holds two depths at one setting.
P6  V2 FAILS on exactly one published ladder -- A100 qwen2-57b-a14b at
    GROUP_SIZE_M=64, whose tread 8 is 1.237% below tread 7 at a 0.482% spread
    (2.6 sigma) -- and PASSES on the other 21.
P7  Of the ladders that HAVE a compute reference, the level bar fails on exactly
    the two BLOCK_N=256 arms -- 1.4% (A100) and 12.3% (H200) of their cards'
    calibrated ceilings -- and passes on the other 22 at 38.2% to 63.7%. Two
    further ladders have no reference at all to level-check and are counted
    separately, because "the reference is corrupt" and "there is no reference"
    are different states and only the first is what this bar is for.
P8  Of the two published BM=128 fits the study quotes, ZERO survive V1+V2+V3+C2.
    FAIL means one of them is admissible and the 128 row of the alpha surface
    stands.
P9  No alpha this study has measured, at ANY block size, is at or below
    {escape_down_alpha(TARGET_TREADS):.3f}. The lowest is 0.596 (H200
    deepseek-v2-lite, BLOCK_M=32, activation-corrected). FAIL means the
    escape-down route is open on hardware the study already owns and the pod run
    should be aimed at whichever arm produced it."""


# --------------------------------------------------------------------------
# The published corpus: the evidence for every number above.
# --------------------------------------------------------------------------

ARM_NAME = re.compile(r"^(?P<model>.+?)-(?P<dtype>bf16|fp16)-r(?P<r_max>\d+)"
                      r"-g(?P<group_m>\d+)-n(?P<block_n>\d+)-")


@dataclass
class LadderRecord:
    """One published BLOCK_M=128 ladder, with everything a gate needs."""

    arm: str
    card: str
    model: str
    dtype: str
    group_m: int
    block_n: int
    points: list[tuple[int, float]]
    spread: float
    overhead_ms: float
    ref_block_m: int | None
    ref_slope: float | None
    memory_points: int
    published_alpha: float | None
    ceiling_tflops: float
    path: str
    #: The lowest activation-corrected alpha this ARM measured at ANY block
    #: size, and the block size it came from. This is the evidence against the
    #: escape-down route, and it lives at BLOCK_M=32 and 64 -- the block sizes
    #: where alpha is identifiable -- not at 128, where it is not.
    min_alpha: float | None = None
    min_alpha_block_m: int | None = None

    @property
    def c_ref(self) -> float | None:
        """`C` at 128, scaled from the reference by `C ~ BLOCK_M`."""
        if self.ref_slope is None or not self.ref_block_m:
            return None
        return self.ref_slope * SUBJECT_BLOCK_M / self.ref_block_m


def _card_of(arm: str) -> str | None:
    low = arm.lower()
    for key in CALIBRATION_SLUGS:
        if key in low:
            return key
    return None


def _ceiling(card: str, dtype: str, directory: Path | None = None) -> float:
    hw = load_hardware(CALIBRATION_SLUGS[card], directory=directory)
    return hw.peak(dtype) / 1e12


def load_corpus(published: Path, dtype: str = "bf16",
                hardware_dir: Path | None = None
                ) -> tuple[list[LadderRecord], list[str]]:
    """Every published BLOCK_M=128 ladder, and a NAMED reason for each skip.

    Skips are returned rather than swallowed. A corpus audit that quietly drops
    the arms it cannot parse reports a cleaner picture than the data supports,
    and the two arms most worth looking at here are the two that are broken.
    """
    records: list[LadderRecord] = []
    skipped: list[str] = []
    for path in sorted(published.glob("*/*.report.json")):
        arm = path.parent.name
        card = _card_of(arm)
        if card is None:
            skipped.append(f"{arm}/{path.name}: no card in the arm name")
            continue
        m = ARM_NAME.match(path.name)
        if not m:
            skipped.append(f"{arm}/{path.name}: filename does not carry the arm")
            continue
        if m["dtype"] != dtype:
            skipped.append(f"{arm}/{path.name}: dtype {m['dtype']}, not {dtype}")
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append(f"{arm}/{path.name}: unreadable ({exc})")
            continue
        fit = (payload.get("ladder") or {}).get(str(SUBJECT_BLOCK_M))
        if not fit or len(fit.get("points") or []) < 2:
            skipped.append(f"{arm}/{path.name}: no BLOCK_M={SUBJECT_BLOCK_M} ladder")
            continue
        if m["model"] not in MODEL_CONFIGS:
            skipped.append(f"{arm}/{path.name}: unknown model {m['model']!r}")
            continue
        ref = payload.get("compute_reference") or {}
        alphas = [(v.get("alpha_corrected") or v.get("alpha"), int(k))
                  for k, v in (payload.get("ladder") or {}).items()
                  if (v.get("alpha_corrected") or v.get("alpha"))]
        best = min(alphas) if alphas else (None, None)
        records.append(LadderRecord(
            arm=arm, card=card, model=m["model"], dtype=m["dtype"],
            group_m=int(m["group_m"]), block_n=int(m["block_n"]),
            points=[(int(n), float(ms)) for n, ms in fit["points"]],
            spread=float(payload.get("timing_spread_median") or 0.0),
            overhead_ms=float(payload.get("overhead_ms") or 0.0),
            ref_block_m=ref.get("block_m"), ref_slope=ref.get("slope_per_tile"),
            memory_points=int(fit.get("memory_points") or 0),
            published_alpha=fit.get("alpha"),
            ceiling_tflops=_ceiling(card, m["dtype"], hardware_dir),
            path=str(path), min_alpha=best[0], min_alpha_block_m=best[1]))
    return records, skipped


@dataclass
class AuditRow:
    """One corpus ladder, after every diagnostic has been run on it."""

    record: LadderRecord
    level: RefLevel | None
    inversions: list[Inversion]
    drops: list[SlopeDrop]
    ratio: float | None
    margin: Margin
    admissible: bool
    reasons: list[str]


def audit_record(rec: LadderRecord, seed: int = 0,
                 draws: int = BOOTSTRAP_DRAWS) -> AuditRow:
    """Run V1, V2, V3 and C2 over one published ladder.

    The branch assignment used for `B` is the LEADING RUN the fit itself
    reported (`memory_points`), except that a ladder the fit gave 0 or 1 treads
    is still measured over its whole length -- not to claim it has a memory
    branch, but so its `B/C` can be read and compared against the escape
    thresholds. Which of the two happened is in `reasons`.
    """
    cfg = MODEL_CONFIGS[rec.model]
    level = None
    if rec.ref_slope and rec.ref_block_m:
        level = reference_level(cfg, rec.ref_block_m, rec.ref_slope,
                                rec.ceiling_tflops,
                                f"{rec.card.upper()} calibration")
    inv = inversions(rec.points, rec.spread or None)
    drops = slope_drops(rec.points, rec.spread or None)
    k = rec.memory_points if rec.memory_points >= 2 else len(rec.points)
    ratio = _refit_ratio(rec.points, k, rec.points[k:], rec.c_ref,
                         rec.overhead_ms)
    margin = margin_of(rec.points, k, c_ref=rec.c_ref,
                       overhead=rec.overhead_ms, spread=rec.spread or None,
                       draws=draws, seed=seed)

    reasons = []
    if level is not None and not level.passes:
        reasons.append(f"V1 reference at {level.fraction:.1%} of the ceiling")
    if level is None:
        reasons.append("V1 no compute reference")
    bad_inv = [i for i in inv if i.sigma is None or i.sigma >= MONOTONE_SIGMA]
    if bad_inv:
        reasons.append(f"V2 {len(bad_inv)} inversion(s) beyond "
                       f"{MONOTONE_SIGMA:.0f} sigma")
    if drops:
        reasons.append(f"V3 {len(drops)} slope drop(s) over "
                       f"{SLOPE_DROP_FLOOR:.0%} and {SLOPE_DROP_SIGMA:.0f} sigma")
    if rec.memory_points < MIN_MEMORY_TREADS:
        reasons.append(f"below MIN_MEMORY_TREADS: {rec.memory_points} treads "
                       f"stand above the compute branch, a verdict needs "
                       f"{MIN_MEMORY_TREADS}")
    if not margin.confident:
        reasons.append("C2 margin not " + (f"{MARGIN_SIGMA:.0f} sd clear"
                                           if margin.clears else "cleared"))
    return AuditRow(rec, level, inv, drops, ratio, margin, not reasons, reasons)


#: What a "bigger weights cost more to re-read" mechanism would predict for the
#: mixtral/qwen2 B/C ratio, and what the identity predicts instead. Stated as
#: constants so the gate names the alternative it rules out.
EXPERT_SIZE_RATIO_BOUND = 0.20


def _weight_elements(model: str) -> int:
    cfg = MODEL_CONFIGS[model]
    return 3 * cfg.intermediate_size * cfg.hidden_size


def _gate_escape_down_alpha(rows: list[AuditRow]) -> Gate:
    """P9: is any measured alpha low enough for the escape-down route to exist.

    Scored over EVERY block size in every arm, not only BLOCK_M=128, because
    alpha is identifiable at 32 and 64 and is not at 128 -- and the bound is a
    property of the model, not of the tile height (see `escape_down_alpha`). A
    PASS here means the route is shut by numbers this study already published,
    with no ridge, no bandwidth and no card in the argument.
    """
    bound = escape_down_alpha(TARGET_TREADS)
    seen = [(r.record.min_alpha, r.record.min_alpha_block_m, r.record)
            for r in rows if r.record.min_alpha is not None]
    if not seen:
        return Gate(CLAIM, "P9 escape-down is shut by alpha alone",
                    f"no measured alpha is at or below {bound:.3f}",
                    f"min measured alpha > {bound:.3f}", None,
                    "no arm reported an identifiable alpha at any block size")
    lo = min(seen)
    return Gate(CLAIM, "P9 escape-down is shut by alpha alone",
                f"no measured alpha is at or below {bound:.3f}",
                f"min measured alpha > {bound:.3f}", lo[0] > bound,
                f"lowest of {len(seen)} arms: alpha {lo[0]:.3f} at BLOCK_M="
                f"{lo[1]} ({lo[2].card.upper()} {lo[2].model} g{lo[2].group_m} "
                f"n{lo[2].block_n}), against the {bound:.3f} bound",
                lines=[f"{TARGET_TREADS} clean treads below the crossing need "
                       f"alpha <= {bound:.3f} even charging zero overhead; the "
                       "bound falls further once overhead is charged."])


def _gate_expert_size(valid: list[AuditRow]) -> Gate:
    """P4: does per-expert weight size move B/C at all.

    Paired inside an ARM directory, so the two models were measured on the same
    card in the same session at the same swizzle and BLOCK_N. An unpaired
    comparison would carry the card and the session, and the card alone is worth
    8.5%.
    """
    pairs = []
    by_key: dict[tuple, dict[str, float]] = {}
    for r in valid:
        if r.ratio is None:
            continue
        key = (r.record.arm, r.record.group_m, r.record.block_n)
        by_key.setdefault(key, {})[r.record.model] = r.ratio
    for key, models in sorted(by_key.items()):
        if "mixtral-8x7b" in models and "qwen2-57b-a14b" in models:
            pairs.append((key, models["mixtral-8x7b"] / models["qwen2-57b-a14b"]))
    if not pairs:
        return Gate(CLAIM, "P4 expert size does not enter B/C",
                    "a 6.4x bigger expert does not make B/C 6.4x bigger",
                    f"median |ratio - 1| < {EXPERT_SIZE_RATIO_BOUND:.2f}", None,
                    "no arm measured both models at one setting, so nothing is "
                    "paired and the comparison would carry the card")
    size = _weight_elements("mixtral-8x7b") / _weight_elements("qwen2-57b-a14b")
    worst = max(abs(v - 1.0) for _, v in pairs)
    med = statistics.median(abs(v - 1.0) for _, v in pairs)
    return Gate(CLAIM, "P4 expert size does not enter B/C",
                "a 6.4x bigger expert does not make B/C 6.4x bigger",
                f"median |B/C(mixtral)/B/C(qwen2) - 1| < "
                f"{EXPERT_SIZE_RATIO_BOUND:.2f}",
                med < EXPERT_SIZE_RATIO_BOUND,
                f"{len(pairs)} matched pairs, median |ratio-1| {med:.3f}, worst "
                f"{worst:.3f}; a per-expert-size mechanism predicts {size:.1f}",
                lines=[f"{key[0]} g{key[1]} n{key[2]}: {v:.3f}x"
                       for key, v in pairs])


def _gate_depth_invariance(valid: list[AuditRow]) -> Gate:
    """P5: does sweeping deeper add treads that QUALIFY as memory bound.

    `n*` has no `r_max` in it, so depth buys treads to measure and not treads
    that pass. Checkable only where the corpus holds two ladder lengths at one
    setting; UNKNOWN otherwise, because "we never varied it" is not evidence.
    """
    by_key: dict[tuple, set[int]] = {}
    for r in valid:
        key = (r.record.card, r.record.model, r.record.group_m,
               r.record.block_n)
        by_key.setdefault(key, set()).add(len(r.record.points))
    varied = {k: v for k, v in by_key.items() if len(v) > 1}
    if not varied:
        return Gate(CLAIM, "P5 depth does not buy qualifying treads",
                    "the memory-tread count does not rise with ladder length",
                    "no rise in memory_points across ladder lengths", None,
                    f"every setting in the corpus was swept to one depth "
                    f"({sorted({n for v in by_key.values() for n in v})} treads); "
                    "P5 needs two depths at one setting and the corpus has none")
    return Gate(CLAIM, "P5 depth does not buy qualifying treads",
                "the memory-tread count does not rise with ladder length",
                "no rise in memory_points across ladder lengths", None,
                f"{len(varied)} setting(s) hold two depths; this gate is not yet "
                "implemented for them and must not print PASS on that basis")


def audit_report(rows: list[AuditRow], skipped: list[str], b: int = 2
                 ) -> tuple[list[str], list[Gate], dict]:
    """The corpus table, the scored predictions and the gates over all of it."""
    out: list[str] = []
    up = escape_up_alpha_rho(SUBJECT_BLOCK_M, b)
    down = escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS)

    valid = [r for r in rows if r.level is not None and r.level.passes]
    ratios = [r.ratio for r in valid if r.ratio is not None]

    out += ["", "## Every published BLOCK_M=128 ladder", "",
            f"{'card':5s} {'model':16s} {'G':>3s} {'BN':>4s} {'ref%':>6s} "
            f"{'n':>2s} {'mem':>3s} {'B/C':>7s} {'a*rho':>7s} {'margin':>9s} "
            f"{'inv':>3s} {'drop':>4s}  verdict"]
    for r in sorted(rows, key=lambda r: -(r.ratio or -1)):
        rec = r.record
        frac = f"{r.level.fraction:6.1%}" if r.level else "   n/a"
        ratio = f"{r.ratio:7.3f}" if r.ratio is not None else "    n/a"
        arho = f"{r.ratio * 2 * SUBJECT_BLOCK_M / b:7.1f}" if r.ratio else "    n/a"
        marg = (f"{r.margin.margin:+9.6f}" if not math.isnan(r.margin.margin)
                else "      n/a")
        verdict = "ADMISSIBLE" if r.admissible else "; ".join(r.reasons)
        out.append(f"{rec.card.upper():5s} {rec.model[:16]:16s} {rec.group_m:3d} "
                   f"{rec.block_n:4d} {frac} {len(rec.points):2d} "
                   f"{rec.memory_points:3d} {ratio} {arho} {marg} "
                   f"{len(r.inversions):3d} {len(r.drops):4d}  {verdict}")
    for line in skipped:
        out.append(f"      SKIPPED  {line}")
    out += ["",
            "  B/C on a row with fewer than 2 memory treads is the OLS slope of "
            "the WHOLE ladder over the",
            "  independent compute reference. That is not a claim that a memory "
            "branch exists -- it is the",
            "  marginal cost per M-tile asked whether it looks memory-like or "
            "compute-like, which is the only",
            "  form the question has when the ladder is one line. Rows with a "
            "memory branch use that branch,",
            "  and its own compute treads for C, exactly as `fit_ladder` does."]

    out += ["", "## The feasibility arithmetic", ""]
    if ratios:
        arhos = [x * 2 * SUBJECT_BLOCK_M / b for x in ratios]
        out += [f"valid-reference ladders: {len(ratios)}",
                f"  B/C        min {min(ratios):.4f}  median "
                f"{statistics.median(ratios):.4f}  max {max(ratios):.4f}  "
                f"sd {statistics.pstdev(ratios):.4f}",
                f"  alpha x rho  min {min(arhos):.1f}  median "
                f"{statistics.median(arhos):.1f}  max {max(arhos):.1f}",
                f"  escape UP needs alpha x rho >= {up:.1f}: reached by "
                f"{sum(1 for x in arhos if x >= up)} of {len(arhos)}",
                f"  escape DOWN needs B/C <= {1 - TOLERANCE:.2f}: reached by "
                f"{sum(1 for x in ratios if x <= 1 - TOLERANCE)} of {len(ratios)}",
                f"  the median must move {up / statistics.median(arhos) - 1:+.1%} "
                "to reach the escape-up threshold"]
        out += ["", "  what each lever is worth, as the spread of the median B/C:"]
        for key, get in (("card", lambda r: r.record.card.upper()),
                         ("model", lambda r: r.record.model),
                         ("GROUP_SIZE_M", lambda r: r.record.group_m),
                         ("BLOCK_SIZE_N", lambda r: r.record.block_n)):
            groups: dict = {}
            for r in valid:
                if r.ratio is not None:
                    groups.setdefault(get(r), []).append(r.ratio)
            meds = {k: statistics.median(v) for k, v in groups.items()}
            if len(meds) < 2:
                out.append(f"    {key:14s} only one level in the corpus")
                continue
            lo, hi = min(meds.values()), max(meds.values())
            body = "  ".join(f"{k}={v:.3f}" for k, v in sorted(meds.items(),
                                                              key=lambda kv: str(kv[0])))
            out.append(f"    {key:14s} {body}   spread {hi / lo - 1:+.1%}")
        out += ["    the `model` row is NOT evidence that expert size matters: "
                "it is dominated by",
                "    deepseek-v2-lite's single two-tread ladder. The matched "
                "mixtral/qwen2 pairs are P4."]

    counts = {"reports read": len(rows) + len(skipped),
              "BM=128 ladders": len(rows),
              "valid references": len(valid),
              "tread boundaries": sum(max(0, len(r.record.points) - 1) for r in rows),
              "bootstrap resamples": sum(r.margin.draws for r in rows)}
    gates = [gate_non_vacuity(counts)]

    # P6 / P7 / P8 are scored here because the corpus can settle them today.
    fail_v2 = [r for r in rows
               if any(i.sigma is None or i.sigma >= MONOTONE_SIGMA
                      for i in r.inversions)]
    gates.append(Gate(
        CLAIM, "P6 monotonicity discriminates",
        "exactly one published ladder runs backwards",
        "V2 fails on 1 ladder", len(fail_v2) == 1,
        f"{len(fail_v2)} of {len(rows)} fail V2: "
        + ("; ".join(f"{r.record.card.upper()} {r.record.model} "
                     f"g{r.record.group_m}" for r in fail_v2) or "none"),
        lines=[i.line() for r in fail_v2 for i in r.inversions]))

    # Split deliberately: a corrupt reference and a missing one fail the same
    # gate for different reasons, and lumping them turned this prediction into a
    # count nobody could check.
    no_ref = [r for r in rows if r.level is None]
    bad_level = [r for r in rows if r.level is not None and not r.level.passes]
    gates.append(Gate(
        CLAIM, "P7 the level bar finds the corrupt references",
        "exactly the two BLOCK_N=256 arms fail the level bar",
        "the level bar fails on 2 ladders that HAVE a reference, both "
        "BLOCK_SIZE_N=256",
        len(bad_level) == 2 and all(r.record.block_n == 256 for r in bad_level),
        f"{len(bad_level)} of {len(rows) - len(no_ref)} referenced ladders fail "
        "the level bar: "
        + ("; ".join(f"{r.record.card.upper()} BN={r.record.block_n} at "
                     f"{r.level.fraction:.1%}" for r in bad_level) or "none")
        + f"; {len(no_ref)} further ladder(s) have NO reference to check: "
        + ("; ".join(f"{r.record.card.upper()} {r.record.model} "
                     f"g{r.record.group_m} n{r.record.block_n}" for r in no_ref)
           or "none"),
        lines=["`compute_reference` tests proportionality only, so a reference "
               "43.6x too slow qualified and then classified every tread of "
               "every ladder in its arm as compute bound."]))

    gates.append(_gate_escape_down_alpha(rows))
    gates.append(_gate_expert_size(valid))
    gates.append(_gate_depth_invariance(valid))

    quoted = [r for r in rows if r.record.memory_points >= 2]
    survivors = [r for r in quoted if r.admissible]
    gates.append(Gate(
        CLAIM, "P8 no admissible BM=128 fit exists",
        "zero of the study's BLOCK_M=128 fits survive V1+V2+V3+C2",
        "0 survivors", len(survivors) == 0,
        f"{len(quoted)} ladder(s) reported a memory branch, {len(survivors)} "
        "survive",
        lines=[f"{r.record.card.upper()} {r.record.model} g{r.record.group_m} "
               f"n{r.record.block_n}: " + "; ".join(r.reasons) for r in quoted]))

    payload = {
        "subject_block_m": SUBJECT_BLOCK_M,
        "tolerance": TOLERANCE,
        "target_treads": TARGET_TREADS,
        "escape_up_alpha_rho": up,
        "escape_down_rho": down,
        "counts": counts,
        "ladders": [{"arm": r.record.arm, "card": r.record.card,
                     "model": r.record.model, "group_m": r.record.group_m,
                     "block_n": r.record.block_n,
                     "treads": len(r.record.points),
                     "memory_points": r.record.memory_points,
                     "reference_fraction": r.level.fraction if r.level else None,
                     "ratio": r.ratio, "margin": None if math.isnan(r.margin.margin)
                     else r.margin.margin,
                     "margin_sigma": r.margin.sigma,
                     "inversions": len(r.inversions), "slope_drops": len(r.drops),
                     "admissible": r.admissible, "reasons": r.reasons}
                    for r in rows],
        "skipped": skipped,
        "gates": [{"kind": g.kind, "name": g.name, "verdict":
                   {True: "PASS", False: "FAIL", None: "UNKNOWN"}[g.passed],
                   "observed": g.observed} for g in gates],
    }
    return out, gates, payload


# --------------------------------------------------------------------------
# The pod run: a replicated ladder at the subject block size.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """One timing of one tread in one repeat. The CSV row."""

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


def ladder_rows(cfg, block_m: int, r_max: int) -> list[int]:
    """Exactly-full tile stacks only: `r = n BM`, zero padding, one per tread.

    No probes and no background grid. This experiment reads the ladder and
    nothing else, so a grid point that is not a tread top is GPU time spent on
    a number no gate here will look at. REFUSES when the model's routing makes
    `n BM` an impossible token count rather than nudging it: a nudged row is not
    a full tile stack, and a fit over partly-filled treads is a fit over
    padding.
    """
    q = SWEEP.rows_quantum(cfg)
    rows = []
    for n in range(1, r_max // block_m + 1):
        r = n * block_m
        if r % q:
            raise SystemExit(
                f"{cfg.num_experts} experts at top-k {cfg.top_k} need rows per "
                f"expert to be a multiple of {q}, and {r} (tread {n} at "
                f"BLOCK_M={block_m}) is not. This model cannot form an exactly "
                f"full tile stack at this block size; choose another --model.")
        rows.append(r)
    if not rows:
        raise SystemExit(f"--r-max {r_max} is below one tile at BLOCK_M="
                         f"{block_m}; nothing to measure.")
    return rows


@dataclass(frozen=True)
class Plan:
    """Everything the pod run will do, computable on a laptop."""

    model: str
    dtype: str
    pinned: dict
    reference_rows: list[int]
    subject_rows: list[int]
    reps: int
    iters: int
    warmup: int
    cell_budget_ms: float
    estimated_seconds: float
    cells: int

    def lines(self, cfg) -> list[str]:
        return [
            f"model        {self.model} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       {self.pinned}",
            f"reference    BLOCK_M={REFERENCE_BLOCK_M} at rows "
            f"{self.reference_rows} -- MEASURED FIRST, and its level checked "
            "before the subject costs anything",
            f"subject      BLOCK_M={SUBJECT_BLOCK_M} at rows {self.subject_rows}",
            f"repeats      {self.reps} round-robin passes per setting, so an "
            "inversion can be told from noise",
            f"timing       {self.warmup} warmup + up to {self.iters} iters, cut "
            f"to keep a cell inside {self.cell_budget_ms:.0f} ms",
            f"cells        {self.cells} timings "
            f"({len(self.reference_rows) + len(self.subject_rows)} treads x "
            f"{self.reps} reps)",
            f"estimate     {self.estimated_seconds:.0f} s of GPU at the model's "
            "own timings, excluding compiles and allocation",
        ]


def build_plan(args, cfg, b: int, alpha: float, ridge: float,
               bandwidth_gbps: float) -> Plan:
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                  BLOCK_SIZE_N=args.block_n, BLOCK_SIZE_K=args.block_k)
    ref_rows = ladder_rows(cfg, REFERENCE_BLOCK_M, args.r_max)
    sub_rows = ladder_rows(cfg, SUBJECT_BLOCK_M, args.r_max)
    total = 0.0
    for bm, rows in ((REFERENCE_BLOCK_M, ref_rows), (SUBJECT_BLOCK_M, sub_rows)):
        for r in rows:
            ms = SWEEP.model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                                bandwidth_gbps=bandwidth_gbps, b=b)
            iters = SWEEP.scaled_iters(ms, args.iters, args.cell_budget_ms)
            total += args.reps * ms * (args.warmup + iters)
    return Plan(args.model, args.dtype, pinned, ref_rows, sub_rows, args.reps,
                args.iters, args.warmup, args.cell_budget_ms, total / 1e3,
                args.reps * (len(ref_rows) + len(sub_rows)))


def collapse(samples: list[Sample], block_m: int
             ) -> tuple[list[tuple[int, float]], dict[int, list[float]], float | None]:
    """Per-tread median across repeats, the repeats themselves, and the spread.

    The median across REPEATS rather than the single-pass median: a repeat is a
    fresh compile-free call at a fresh point in the pod's thermal history, and
    the study's one non-monotone ladder is exactly what a single pass cannot
    distinguish from a mechanism.
    """
    by: dict[int, list[float]] = {}
    for s in samples:
        if s.block_m == block_m and s.status == "ok" and s.ms_p50 > 0:
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = [(n, statistics.median(v)) for n, v in sorted(by.items())]
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1 and statistics.median(v) > 0]
    return points, by, statistics.median(spreads) if spreads else None


def drift(samples: list[Sample], block_m: int) -> float | None:
    """Median relative change from the first repeat to the last, per tread.

    A pod whose clocks ramp during a setting produces a ladder that bends, and a
    ladder that bends is read by a two-line fit as a second mechanism. Reported
    rather than gated because the sign is informative either way: a negative
    drift is a warm-up, a positive one is a thermal fade.
    """
    firsts: dict[int, float] = {}
    lasts: dict[int, float] = {}
    for s in sorted(samples, key=lambda s: s.rep):
        if s.block_m != block_m or s.status != "ok" or s.ms_p50 <= 0:
            continue
        firsts.setdefault(s.tiles, s.ms_p50)
        lasts[s.tiles] = s.ms_p50
    rel = [lasts[n] / firsts[n] - 1.0 for n in firsts if firsts[n] > 0]
    return statistics.median(rel) if rel else None


def append_sample(path: Path, sample: Sample) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(sample))
        fh.flush()


def read_samples(path: Path) -> tuple[set[tuple[int, int, int]], list[Sample]]:
    if not path.exists():
        return set(), []
    out: list[Sample] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Sample(
                block_m=int(row["block_m"]), tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), rep=int(row["rep"]),
                ms_p50=float(row["ms_p50"]), ms_min=float(row["ms_min"]),
                ms_stdev=float(row["ms_stdev"]), iters=int(row["iters"]),
                status=row.get("status", "ok"), detail=row.get("detail", "")))
    # Only successful timings count as done, for the same reason the sweep does
    # it: a failed cell is usually a pod that lost its device, and a real
    # failure fails again in milliseconds.
    return {(s.block_m, s.tiles, s.rep) for s in out if s.status == "ok"}, out


def measure_setting(args, cfg, block_m: int, rows: list[int], csv_path: Path,
                    cache_root: Path, pinned: dict, done, samples: list[Sample]
                    ) -> tuple[int, int]:
    """Time one block size, `--reps` round-robin passes over its treads.

    ROUND ROBIN INSIDE THE SETTING, not tread-by-tread to completion. Measuring
    tread 1 fifty times and then tread 8 fifty times puts every tread at a
    different point in the pod's thermal history, and the resulting monotone
    drift IS a slope -- the very quantity being fitted. One pass over all treads
    per repeat spreads that drift across the whole ladder instead of aligning it
    with the x axis.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, _ = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    SWEEP.arm_triton_cache(cache_root, block_m)
    seen: set[Path] = set()
    SWEEP.count_new(cache_root, seen)
    compiles = executed = 0
    built: dict[int, tuple] = {}

    for rep in range(1, args.reps + 1):
        for r in rows:
            tokens = SWEEP.tokens_for_rows(cfg, r)
            if (block_m, r // block_m, rep) in done:
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
            conf = dict(pinned, BLOCK_SIZE_M=block_m)

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
                sample = Sample(block_m, r // block_m, r, tokens, rep, ms, mn,
                                sd, iters)
            except Exception as exc:                    # noqa: BLE001
                sample = Sample(block_m, r // block_m, r, tokens, rep, 0.0, 0.0,
                                0.0, 0, "failed", f"{type(exc).__name__}: {exc}")
                print(f"  BM={block_m} n={r // block_m} rep={rep} FAILED "
                      f"{sample.detail}")
            samples.append(sample)
            append_sample(csv_path, sample)
            print(f"  BM={block_m:3d} n={r // block_m:2d} rep={rep:2d} "
                  f"r={r:5d} T={tokens:6d}  {sample.ms_p50:9.4f} ms "
                  f"({sample.iters} iters)")
    return compiles, executed


def analyse_run(samples, cfg, b: int, ceiling_tflops: float, ceiling_source: str,
                compiles: dict[int, int], executed: dict[int, int], *,
                ridge: float, bandwidth_gbps: float, pinned: dict | None = None,
                seed: int = 0, draws: int = BOOTSTRAP_DRAWS
                ) -> tuple[list[str], list[Gate], dict]:
    """The measured run's gates. Same diagnostics the audit runs on the corpus."""
    out: list[str] = []
    ref_points, _, ref_spread = collapse(samples, REFERENCE_BLOCK_M)
    sub_points, sub_reps, sub_spread = collapse(samples, SUBJECT_BLOCK_M)

    # The compute reference comes from the study's own qualification, run over
    # Cell objects built from these medians, so this run is judged by the same
    # code the published surface was.
    cells = []
    for bm, points in ((REFERENCE_BLOCK_M, ref_points),
                       (SUBJECT_BLOCK_M, sub_points)):
        for n, ms in points:
            cells.append(SWEEP.make_cell(cfg, n * bm, bm, ms,
                                         sm_count=1, block_n=1))
    ref = SWEEP.compute_reference(
        cells, (SUBJECT_BLOCK_M, REFERENCE_BLOCK_M), cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned)
    level = (reference_level(cfg, ref.block_m, ref.slope_per_tile,
                             ceiling_tflops, ceiling_source)
             if ref.block_m and ref.slope_per_tile else None)

    moved = drift(samples, SUBJECT_BLOCK_M)
    margin_band = max(SWEEP.MEMORY_BRANCH_MARGIN, 3.0 * (sub_spread or 0.0))
    fit = SWEEP.fit_ladder(sub_points, SUBJECT_BLOCK_M, ref, margin=margin_band)
    c_ref = ref.slope_for(SUBJECT_BLOCK_M)
    k = fit.memory_points if fit.memory_points >= 2 else len(sub_points)
    margin = margin_of(sub_points, k, c_ref=c_ref, overhead=ref.overhead_ms,
                       spread=sub_spread, replicates=sub_reps, draws=draws,
                       seed=seed)
    inv = inversions(sub_points, sub_spread)
    drops = slope_drops(sub_points, sub_spread)

    out += ["", "## The measured ladder", "",
            f"reference    {ref.note}",
            f"             {level.line() if level else 'no level: no reference'}",
            f"subject      {len(sub_points)} treads, across-repeat spread "
            + (f"{sub_spread:.3%}" if sub_spread else "UNKNOWN (one repeat)"),
            f"             membership margin {margin_band:.3%} "
            f"(floor {SWEEP.MEMORY_BRANCH_MARGIN:.0%}, 3x the measured spread)",
            "             drift first->last repeat "
            + (f"{moved:+.3%}" if moved is not None else "unknown"),
            f"             {fit.basis}",
            f"             memory treads {fit.memory_points}, alpha "
            + (f"{fit.alpha:.4f}" if fit.alpha is not None
               else "NOT IDENTIFIABLE"),
            f"             {margin.line()}", "",
            f"{'n':>3s} {'rows':>6s} {'ms':>10s} {'slope':>9s} {'reps':>5s} "
            f"{'spread':>8s}"]
    slopes = slope_sequence(sub_points)
    for i, (n, ms) in enumerate(sub_points):
        reps = sub_reps.get(n, [])
        sp = (statistics.pstdev(reps) / ms) if len(reps) > 1 and ms > 0 else None
        out.append(f"{n:3d} {n * SUBJECT_BLOCK_M:6d} {ms:10.4f} "
                   + (f"{slopes[i - 1]:9.4f}" if i else "        -")
                   + f" {len(reps):5d} "
                   + (f"{sp:8.3%}" if sp is not None else "       -"))

    gates = [
        gate_non_vacuity({
            "timings": len(samples),
            "subject treads": len(sub_points),
            "reference treads": len(ref_points),
            "repeats per tread": min((len(v) for v in sub_reps.values()),
                                     default=0),
            "bootstrap resamples": margin.draws,
            "settings that executed": sum(1 for v in executed.values() if v > 0),
        }),
        _gate_override(compiles, executed),
        gate_reference_level(level),
        gate_monotone(inv, max(0, len(sub_points) - 1), sub_spread),
        gate_convex(drops, slopes, sub_spread),
        _gate_replication(sub_reps, sub_spread),
        gate_depth(fit.memory_points, len(sub_points)),
        gate_margin(margin),
        _gate_law(fit, margin, c_ref, ref.overhead_ms, margin_band),
    ]
    payload = {
        "subject_block_m": SUBJECT_BLOCK_M,
        "reference": {"block_m": ref.block_m, "slope_per_tile": ref.slope_per_tile,
                      "overhead_ms": ref.overhead_ms, "note": ref.note,
                      "level_fraction": level.fraction if level else None},
        "subject_points": sub_points,
        "subject_spread": sub_spread,
        "memory_points": fit.memory_points,
        "alpha": fit.alpha, "alpha_upper": fit.alpha_upper,
        "slope_memory": fit.slope_memory, "slope_compute_ref": c_ref,
        "ratio": margin.ratio if not math.isnan(margin.ratio) else None,
        "margin": None if math.isnan(margin.margin) else margin.margin,
        "margin_sd": margin.sd, "margin_sigma": margin.sigma,
        "margin_basis": margin.basis,
        "inversions": [asdict(i) for i in inv],
        "slope_drops": [asdict(d) for d in drops],
        "gates": [{"kind": g.kind, "name": g.name, "verdict":
                   {True: "PASS", False: "FAIL", None: "UNKNOWN"}[g.passed],
                   "observed": g.observed} for g in gates],
    }
    return out, gates, payload


def _gate_override(compiles: dict[int, int], executed: dict[int, int]) -> Gate:
    """Did `override_config` change the kernel at each setting.

    Same assay as the sweep's gate 0 and for the same reason: if the override
    silently failed, both settings ran one kernel, `C` and `B` are the same
    line by construction, and the tolerance verdict is a comparison of a kernel
    with itself. A setting that executed nothing this session is UNDECIDED, not
    a failure: the assay belongs to whichever session ran the cells.
    """
    ran = [bm for bm, n in executed.items() if n > 0]
    resumed = [bm for bm, n in executed.items() if n <= 0]
    missing = [bm for bm in ran if compiles.get(bm, 0) <= 0]
    counts = ", ".join(f"BM={bm}:{compiles.get(bm, 0)}"
                       for bm in sorted(executed))
    if missing:
        return Gate(VALIDITY, "V4 override took effect",
                    "each setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran", False,
                    f"{counts}; {missing} ran cells and compiled nothing",
                    "both slopes: one kernel timed twice makes B/C exactly 1 "
                    "and the tolerance discard a foregone conclusion")
    if resumed and not ran:
        return Gate(VALIDITY, "V4 override took effect",
                    "each setting compiled its own kernel",
                    ">= 1 fresh Triton artefact per setting that ran", None,
                    f"{counts}; every setting was resumed from cells.csv, so "
                    "this session ran no assay",
                    "both slopes")
    return Gate(VALIDITY, "V4 override took effect",
                "each setting compiled its own kernel",
                ">= 1 fresh Triton artefact per setting that ran", True, counts,
                "both slopes")


def _gate_replication(reps: dict[int, list[float]], spread: float | None) -> Gate:
    worst = min((len(v) for v in reps.values()), default=0)
    if spread is None:
        return Gate(VALIDITY, "V5 replication",
                    "the ladder was measured often enough to weigh an inversion",
                    f">= 2 repeats per tread and spread <= "
                    f"{MAX_REPLICATE_SPREAD:.0%}", None,
                    f"{worst} repeat(s) on the thinnest tread, so no "
                    "across-repeat spread exists",
                    "V2 and C2, which both need a noise band; without one an "
                    "inversion cannot be told from a mechanism, which is how "
                    "the published A100 ladder shipped")
    return Gate(VALIDITY, "V5 replication",
                "the ladder was measured often enough to weigh an inversion",
                f">= 2 repeats per tread and spread <= "
                f"{MAX_REPLICATE_SPREAD:.0%}",
                worst >= 2 and spread <= MAX_REPLICATE_SPREAD,
                f"{worst} repeats on the thinnest tread, median across-repeat "
                f"spread {spread:.3%}",
                "V2 and C2, which both need a noise band")


def _gate_law(fit, margin: Margin, c_ref: float | None, overhead: float,
              band: float) -> Gate:
    """Does the depth law predict the tread count the fit actually found.

    The law is arithmetic, so a mismatch is never the law being 'wrong about
    this card': it means the ladder is not two lines. That is the same finding
    V3 reports from the other direction, and having both makes it checkable
    rather than asserted.
    """
    if fit.alpha is None or margin.ratio is None or math.isnan(margin.ratio) \
            or not c_ref:
        return Gate(CLAIM, "C3 the law predicts the depth",
                    "the observed tread count matches n* from the fitted "
                    "(alpha, B/C)",
                    "within 1 tread", None,
                    "no memory branch was identified, so there is no alpha to "
                    "put into the law")
    n_star = prefix_depth(margin.ratio, fit.alpha, overhead / c_ref, band)
    predicted = len(fit.points) if n_star is None else max(0, math.floor(n_star))
    return Gate(CLAIM, "C3 the law predicts the depth",
                "the observed tread count matches n* from the fitted (alpha, B/C)",
                "within 1 tread", abs(predicted - fit.memory_points) <= 1,
                f"law predicts {predicted} "
                + ("(every tread: B/C is above the membership margin)"
                   if n_star is None else f"(n* = {n_star:.2f})")
                + f", fit found {fit.memory_points}",
                lines=["A mismatch says the ladder is not two lines, which is "
                       "the same thing V3 reports from the other direction."])


# --------------------------------------------------------------------------
# Output paths, and whether git will keep them.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file.

    `.gitignore` ignores `results/*` and re-includes only `results/published/`,
    so a run that writes anywhere else under the repo produces files that `git
    add -A` silently drops. This project has already lost every published plot
    that way. Checked with `git check-ignore` rather than by re-implementing
    the pattern rules, because the pattern rules are what got it wrong.
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
                "results/published/<date>-<gpu>-bm128-depth")
    if proc.returncode == 1:
        return "git will keep this path"
    return (f"git check-ignore exited {proc.returncode}; path unverified "
            f"({proc.stderr.decode(errors='replace').strip()})")


#: The card slug a run id carries when NO device is attached, i.e. every
#: --dry-run on a laptop. Visible rather than blank, so a dry run cannot be
#: mistaken for printing the path a pod will really write to.
UNKNOWN_CARD_SLUG = "nocard"


def detect_card_slug() -> str | None:
    """Slug for the ATTACHED device, or None when there is no device.

    Resolved before the run id is built, because the card belongs IN the id and
    the id is the directory a resumed run reads its treads back out of.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter, so two settings cannot collide.

    The sweep this script is built beside lost a whole arm to a run id that
    omitted GROUP_SIZE_M: the second run resumed into the first's directory,
    found every cell present, skipped all of them, and printed the first run's
    timings under the second's heading. BLOCK_SIZE_K, num_warps and --reps are
    in here for the same reason, even though only one value of each has ever
    been run.

    THE CARD IS THE NEXT FIELD IN THAT SAME LIST, and the one that would have
    bitten hardest here. It is not swept by this script, it is swept by the
    operator moving to another pod, and `$MOE_RESULTS_DIR` is a network volume
    that outlives a pod. Every number this file computes -- B/C, alpha x rho,
    both escape thresholds -- is scored against a per-card calibrated ridge,
    145.8 on the A100 against 162.8 on the H200. Two cards sharing one directory
    would have the second silently report the first's treads against its own
    ridge, which is a hybrid of two machines and is exactly the defect that put
    a stale H200 band into seven published A100 reports.
    """
    key = json.dumps({"card": card, "model": args.model, "dtype": args.dtype,
                      "r_max": args.r_max, "reps": args.reps,
                      "iters": args.iters, "warmup": args.warmup,
                      "budget": args.cell_budget_ms, "seed": args.seed,
                      "group_m": args.group_m, "block_n": args.block_n,
                      "block_k": args.block_k, "num_stages": args.num_stages,
                      "num_warps": args.num_warps,
                      "subject": SUBJECT_BLOCK_M,
                      "reference": REFERENCE_BLOCK_M}, sort_keys=True)
    return (f"{card}-{args.model}-{args.dtype}-r{args.r_max}-g{args.group_m}"
            f"-n{args.block_n}-k{args.block_k}-s{args.num_stages}"
            f"-w{args.num_warps}-x{args.reps}-"
            f"{hashlib.sha1(key.encode()).hexdigest()[:6]}")


# --------------------------------------------------------------------------
# Self test: plant three worlds, check the gates tell them apart.
# --------------------------------------------------------------------------

def planted_ladder(treads: int, *, alpha: float, rho: float, block_m: int,
                   b: int, load_ms: float, overhead_ms: float,
                   noise: float = 0.0, seed: int = 0) -> list[tuple[int, float]]:
    """A ladder generated FROM the law, so the gates have a known answer.

    `t(n) = D + max(L (1 + alpha(n-1)), C n)` with `C = L alpha / ratio`, which
    is the two-line model with `B / C` set to exactly what `branch_ratio` says.
    """
    rng = random.Random(seed)
    ratio = branch_ratio(block_m, b, alpha, rho)
    c = load_ms * alpha / ratio
    out = []
    for n in range(1, treads + 1):
        ms = overhead_ms + max(load_ms * (1.0 + alpha * (n - 1)), c * n)
        out.append((n, ms * (math.exp(rng.gauss(0.0, noise)) if noise else 1.0)))
    return out


def self_test(b: int = 2, treads: int = 8) -> tuple[list[str], list[Gate]]:
    """Three worlds and the verdicts they must produce.

    The point is not that the code runs; it is that the gates DISCRIMINATE. A
    depth gate that passes in every world would have passed on the published
    A100 ladder too.
    """
    load_ms, overhead_ms = 4.0, 0.5
    worlds = [
        # alpha x rho = 166.2, over the 147.2 the escape needs.
        ("escape-up   alpha 0.95 rho 175", 0.95, 175.0, True),
        # Where BOTH cards actually sit at BLOCK_M=128: alpha x rho = 127.6
        # against the 128 that makes the two branches one line.
        ("straddle    alpha 0.88 rho 145", 0.88, 145.0, False),
        # alpha x rho = 96, under the 108.8 cap, and rho = 320 is 2.2x the
        # A100's calibrated 145.8 and 2.0x the H200's 162.8. This world is
        # reachable by the gates and by NO CARD IN THIS STUDY, which is the
        # whole finding: the gates can see a depth that the hardware cannot
        # produce.
        ("escape-down alpha 0.30 rho 320", 0.30, 320.0, True),
    ]
    out = ["", "## Self test: planted worlds, real gates", "",
           f"{'world':34s} {'B/C':>7s} {'regime':>12s} {'law n*':>8s} "
           f"{'mem':>4s} {'margin':>10s}  verdict"]
    gates: list[Gate] = []
    for name, alpha, rho, expect in worlds:
        pts = planted_ladder(treads, alpha=alpha, rho=rho,
                             block_m=SUBJECT_BLOCK_M, b=b, load_ms=load_ms,
                             overhead_ms=overhead_ms)
        # The PLANTED compute slope is used, not one recovered from the ladder:
        # this test is about whether the gates discriminate, and recovering C
        # from the same points would make the comparison circular.
        ratio = branch_ratio(SUBJECT_BLOCK_M, b, alpha, rho)
        c_true = load_ms * alpha / ratio
        law = depth_verdict(SUBJECT_BLOCK_M, b, alpha, rho,
                            overhead_over_c=overhead_ms / c_true)
        k = sum(1 for n, ms in pts if ms > overhead_ms + c_true * n * 1.02)
        margin = margin_of(pts, k if k >= 2 else len(pts), c_ref=c_true,
                           overhead=overhead_ms, spread=0.005, draws=400, seed=1)
        got = k >= TARGET_TREADS and margin.confident
        out.append(f"{name:34s} {law.ratio:7.3f} {law.regime:>12s} "
                   + (f"{'inf':>8s}" if law.reachable_treads is None
                      else f"{law.reachable_treads:8.2f}")
                   + f" {k:4d} {margin.margin:+10.6f}  "
                   + (f"{TARGET_TREADS}+ clean treads" if got else "no"))
        gates.append(Gate(
            VALIDITY, f"S {name.split()[0]}",
            f"the planted world is called {'feasible' if expect else 'infeasible'}",
            f"{TARGET_TREADS} treads and a {MARGIN_SIGMA:.0f} sd margin "
            f"{'reached' if expect else 'not reached'}",
            got == expect, f"{k} treads, margin {margin.margin:+.6f}, "
            + ("confident" if margin.confident else "not confident"),
            "the gates themselves: a gate that answers the same in every world "
            "cannot settle this experiment"))
    return out, gates


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true",
                    help="run the gates over every published BLOCK_M=128 "
                         "ladder and print the feasibility arithmetic. No GPU, "
                         "and the evidence for the docstring")
    ap.add_argument("--published", type=Path,
                    default=ROOT / "results" / "published",
                    help="where the published arms live")
    ap.add_argument("--hardware-dir", type=Path, default=HARDWARE_DIR,
                    help="calibration yaml directory; the ceiling every "
                         "reference level is scored against")
    ap.add_argument("--self-test", action="store_true",
                    help="plant three worlds from the law and check the gates "
                         "tell them apart, off GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, then stop")
    ap.add_argument("--model", default="qwen2-57b-a14b",
                    choices=sorted(MODEL_CONFIGS),
                    help="qwen2 by default: its BLOCK_M=128 ladder is the one "
                         "the A100 surface quotes, so a re-run is comparable")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="not fp8: halving the weight bytes DOUBLES the AI cap, "
                         "which moves B/C the wrong way and off the corpus")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="largest rows per expert; 1024 is 8 treads at "
                         "BLOCK_M=128 and 4 at 256. Raising it cannot change "
                         "WHICH treads are memory bound (see P5), only how many "
                         "exist to be measured")
    ap.add_argument("--reps", type=int, default=7,
                    help="round-robin passes over the ladder. The published "
                         "ladders had ONE, which is why a 1.24%% inversion could "
                         "not be told from noise")
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"],
                    help="the swizzle width. Moves alpha, and so moves B/C -- "
                         "by 6.1%% across the published corpus, against the "
                         "16.1%% the escape needs")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"],
                    help="N tile. 256 is the arm whose BLOCK_M=256 reference is "
                         "43.6x too slow on the A100; V1 refuses it")
    ap.add_argument("--block-k", type=int, default=SWEEP.FIXED["BLOCK_SIZE_K"],
                    help="K tile. Degrading it lowers the achieved compute rate, "
                         "which lowers B/C -- the wrong direction for escape up "
                         "and the wrong direction for escape down as well")
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"])
    ap.add_argument("--num-warps", type=int, default=SWEEP.FIXED["num_warps"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS,
                    help="bootstrap resamples behind every reported margin")
    ap.add_argument("--alpha", type=float, default=SWEEP.ALPHA,
                    help="only used to COST the run in --dry-run")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="only used to cost the run; 0 reads the attached "
                         "device's calibration")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--card", default="",
                    help="card slug the run id is built from. Read from the "
                         "attached device by default and REFUSED if it "
                         "contradicts one; its only real use is printing a "
                         "pod's exact path from a laptop dry run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-git-visible", action="store_true",
                    help="refuse to run when the output path is git-ignored")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero unless every gate passes. Off by "
                         "default: the registered expectation here is that the "
                         "CLAIM gates FAIL, and a failed prediction is a result")
    return ap


def _cost_inputs(args):
    """Ridge and bandwidth used ONLY to price the run, named with their source."""
    ridge, ridge_src = args.ridge, "given on the command line"
    bandwidth, bw_src = args.bandwidth_gbps, "given on the command line"
    if not ridge or not bandwidth:
        try:
            from moe.bench.roofline import load_measured
            hw = load_measured()
        except Exception as exc:                        # noqa: BLE001
            hw, exc_note = None, str(exc)
        else:
            exc_note = ""
        if hw is not None:
            if not ridge:
                ridge = hw.ridge_point(args.dtype)
                ridge_src = f"this machine's calibration ({hw.name})"
            if not bandwidth:
                bandwidth = hw.bandwidth_bytes_s / 1e9
                bw_src = f"this machine's calibration ({hw.name})"
        else:
            # A cost estimate is not a measurement, so a hypothesis is allowed
            # here where it would be refused in a verdict -- and it is labelled
            # so it can never be mistaken for one.
            if not ridge:
                ridge, ridge_src = 145.813, (
                    "HYPOTHESIS: the A100 calibration in this repo, no device "
                    "attached" + (f" ({exc_note})" if exc_note else ""))
            if not bandwidth:
                bandwidth, bw_src = 1799.4, (
                    "HYPOTHESIS: the A100 calibration in this repo, no device "
                    "attached")
    return ridge, ridge_src, bandwidth, bw_src


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    lines: list[str] = [f"experiment  bm128_depth: can BLOCK_M={SUBJECT_BLOCK_M} "
                        f"show {TARGET_TREADS} clean memory-bound treads?", "",
                        predictions_text(b)]
    gates: list[Gate] = []
    payload: dict = {"predictions": predictions_text(b)}

    if args.audit or args.self_test:
        if args.self_test:
            more, g = self_test(b)
            lines += more
            gates += g
        if args.audit:
            records, skipped = load_corpus(args.published, args.dtype,
                                           args.hardware_dir)
            rows = [audit_record(r, seed=args.seed, draws=args.draws)
                    for r in records]
            more, g, pay = audit_report(rows, skipped, b)
            lines += more
            gates += g
            payload["audit"] = pay
        lines += ["", "## Gates", ""] + render_gates(gates)
        print("\n".join(lines))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    ridge, ridge_src, bandwidth, bw_src = _cost_inputs(args)
    plan = build_plan(args, cfg, b, args.alpha, ridge, bandwidth)
    # THE CARD, RESOLVED BEFORE THE RUN ID. See default_run_id: the results root
    # is a network volume shared between pods and every verdict here is scored
    # against a per-card ridge, so two cards must not derive one directory.
    detected = detect_card_slug()
    card = args.card or detected or UNKNOWN_CARD_SLUG
    if args.card and detected and args.card != detected:
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return 2
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bm128_depth" / run_id
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    lines += ["", "## The plan", ""] + plan.lines(cfg) + [
        f"costing      ridge {ridge:.3f} ({ridge_src}); bandwidth "
        f"{bandwidth:.1f} GB/s ({bw_src})",
        f"card         {card}" + ("" if detected else
                                  f"  (NO DEVICE ATTACHED: the id above is the "
                                  f"{UNKNOWN_CARD_SLUG!r} one and is not what a "
                                  "pod derives; pass --card <slug> for that)"),
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per tread per repeat, flushed), "
        "CARD, report.txt, report.json, triton-cache/"]

    if args.dry_run:
        law = depth_verdict(SUBJECT_BLOCK_M, b, args.alpha, ridge)
        lines += ["", "## What the law says about this plan before it runs", "",
                  "  " + law.line(),
                  f"  escape up needs alpha x rho >= "
                  f"{escape_up_alpha_rho(SUBJECT_BLOCK_M, b):.1f}; this plan "
                  f"sits at {args.alpha * ridge:.1f}",
                  f"  escape down needs rho >= "
                  f"{escape_down_rho(SUBJECT_BLOCK_M, b, TARGET_TREADS):.1f}; "
                  f"this card calibrates at {ridge:.1f}, or equivalently "
                  f"alpha <= {escape_down_alpha(TARGET_TREADS):.3f} with no "
                  "card in the argument",
                  "",
                  "  READ THE COSTING ALPHA AND RIDGE AS A COSTING, NOT AS A "
                  "PREDICTION. The pooled",
                  f"  alpha {args.alpha:.3f} is fitted over BLOCK_M <= 64 and "
                  f"{ridge:.1f} is a CALIBRATION ridge, not the",
                  f"  achieved one; together they say alpha x rho = "
                  f"{args.alpha * ridge:.1f}, while the 22 measured",
                  "  BLOCK_M=128 ladders say 126.8 in the median. The measured "
                  "number is the one that",
                  "  decides feasibility; this pair only decides how many "
                  "seconds the run takes.",
                  "  Nothing in --r-max, --model or --block-n appears in either "
                  "threshold. Run --audit for the measured corpus."]
        print("\n".join(lines))
        return 0

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return 2

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        # `missing_gpu_stack` points at the SWEEP's own off-GPU modes, and its
        # `--self-test` takes a float where this script's takes no argument.
        # Naming this file's modes here stops a reader following an invocation
        # that does not exist.
        first = missing.split(".")[0]
        print(f"\n{first}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --audit      the gates over every published BLOCK_M=128 ladder\n"
              "  --self-test  three planted worlds, checking the gates "
              "discriminate\n"
              "  --dry-run    the pod plan, the grid and the cost")
        return 2

    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    print("\n".join(lines))

    try:
        from moe.bench.roofline import load_measured
        hw = load_measured()
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(
            f"no usable calibration for the attached device ({exc}). V1 scores "
            "the compute reference against THIS card's measured bf16 ceiling "
            "and there is nothing to refuse or accept it with. Run "
            "scripts/calibrate_hardware.py first.") from exc
    if hw is None:
        raise SystemExit(
            "no calibration for the attached device. V1 needs this card's own "
            "bf16 ceiling; the seven published A100 reports were scored against "
            "an H200 number precisely because a missing calibration was allowed "
            "to fall back. Run scripts/calibrate_hardware.py first.")
    ceiling = hw.peak(args.dtype) / 1e12
    # RE-DERIVED FROM THE ATTACHED DEVICE, overriding whatever `_cost_inputs`
    # was willing to assume. Those values are allowed to be a labelled
    # HYPOTHESIS because they only price the run; from here they also feed the
    # compute reference's qualification, which is a measurement decision, and a
    # hypothesis ridge reaching a qualification is exactly how seven published
    # A100 reports came to be scored against an H200 number.
    ridge = hw.ridge_point(args.dtype)
    bandwidth = hw.bandwidth_bytes_s / 1e9
    print(f"\nqualification ridge {ridge:.3f} Op/B and bandwidth "
          f"{bandwidth:.1f} GB/s, both from {hw.name}")

    # THE RESUME GUARD, belt as well as braces. The card is already in the run
    # id, so another card lands in another directory and cannot normally reach
    # this cells.csv at all. This catches the ways it could anyway: an explicit
    # --out or --run-id aiming two cards at one place, a directory copied
    # between pods, or a cells.csv written before the card entered the id. It
    # REFUSES rather than starting over, because silently discarding measured
    # treads is its own way to lose an arm.
    if csv_path.exists():
        written_by = card_path.read_text().strip() if card_path.exists() else ""
        if written_by != card:
            raise SystemExit(
                f"REFUSED to resume {csv_path}: written by card "
                f"{written_by or '<unrecorded, pre-card-in-id>'!r} and this run "
                f"is {card!r}. Resuming would report one card's treads against "
                "the other's ridge -- 145.8 against 162.8 Op/B -- which is a "
                "hybrid of two machines. Move or delete that directory "
                "deliberately. Nothing measured.")
    card_path.write_text(card + "\n")

    done, samples = read_samples(csv_path)
    compiles: dict[int, int] = {}
    executed: dict[int, int] = {}
    started = time.time()

    # THE REFERENCE FIRST, and its level checked before the subject costs
    # anything. A reference 43.6x too slow classifies every subject tread, so
    # measuring the subject before knowing the reference is sound is spending
    # the expensive half of the run on cells that cannot be read.
    print(f"\n-- reference ladder, BLOCK_M={REFERENCE_BLOCK_M} --")
    c, e = measure_setting(args, cfg, REFERENCE_BLOCK_M, plan.reference_rows,
                           csv_path, cache_root, plan.pinned, done, samples)
    compiles[REFERENCE_BLOCK_M], executed[REFERENCE_BLOCK_M] = c, e

    ref_points, _, _ = collapse(samples, REFERENCE_BLOCK_M)
    ref_cells = [SWEEP.make_cell(cfg, n * REFERENCE_BLOCK_M, REFERENCE_BLOCK_M,
                                 ms, sm_count=1, block_n=1)
                 for n, ms in ref_points]
    early = SWEEP.compute_reference(
        ref_cells, (REFERENCE_BLOCK_M,), cfg=cfg, ridge=ridge,
        bandwidth_gbps=bandwidth, b=b, pinned=plan.pinned)
    if early.block_m and early.slope_per_tile:
        lvl = reference_level(cfg, early.block_m, early.slope_per_tile, ceiling,
                              hw.name)
        print(f"\n{lvl.line()}")
        if not lvl.passes:
            print("REFUSING to measure the subject: the compute reference is "
                  f"outside [{REFERENCE_LEVEL_FLOOR:.0%}, 100%] of this card's "
                  "ceiling, so every membership decision it would make is void. "
                  "This is the A100 BLOCK_N=256 failure, caught before the "
                  "expensive half of the run.")
            print(f"cells   {csv_path}")
            return 1
    else:
        print(f"\nreference did not qualify: {early.note}")

    print(f"\n-- subject ladder, BLOCK_M={SUBJECT_BLOCK_M} --")
    c, e = measure_setting(args, cfg, SUBJECT_BLOCK_M, plan.subject_rows,
                           csv_path, cache_root, plan.pinned, done, samples)
    compiles[SUBJECT_BLOCK_M], executed[SUBJECT_BLOCK_M] = c, e
    print(f"\nmeasured in {time.time() - started:.0f} s")

    more, g, pay = analyse_run(samples, cfg, b, ceiling, hw.name, compiles,
                               executed, ridge=ridge, bandwidth_gbps=bandwidth,
                               pinned=plan.pinned, seed=args.seed,
                               draws=args.draws)
    gates += g
    payload["run"] = pay
    payload["gpu"] = torch.cuda.get_device_name(0)
    text = "\n".join(lines + more + ["", "## Gates", ""] + render_gates(gates))
    print("\n".join(more + ["", "## Gates", ""] + render_gates(gates)))
    (out_dir / "report.txt").write_text(text)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path),
                        ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        # Every path, not just the directory: `.gitignore` re-includes
        # `results/published/` under a blanket `results/*` exclusion, and this
        # repo has already lost every published figure to a pattern that
        # matched at a depth nobody checked.
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return 1 if (args.fail_on_gate
                 and any(g.passed is not True for g in gates)) else 0


if __name__ == "__main__":
    sys.exit(main())
