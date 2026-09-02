#!/usr/bin/env python3
"""alpha's denominator, measured instead of extrapolated.

    python scripts/memory_branch_anchor.py --rescore            # free, no GPU
    python scripts/memory_branch_anchor.py --rescore --dry-run  # plan only
    python scripts/memory_branch_anchor.py --measure --dry-run  # plan + cost
    python scripts/memory_branch_anchor.py --measure --model qwen2-57b-a14b \
        --tiles 32,64 --group-m 1,8,16,64

WHY THIS EXISTS. Every alpha this study has published is `B / L`, where `B` is
the memory branch's per-tile slope and `L = A + B` is that branch's fitted level
at one M-tile. `B` is a slope over 16 to 33 treads. `L` is an EXTRAPOLATION to
n=1, and the data contradicts it: refitting all 40 anchorable ladders by OLS
reproduces the published `slope_memory` to four decimals, so the published line
IS the line under test -- and the MEASURED n=1 tread sits ABOVE that line in 12
of 12 A100 fits, by 1.7% to 37.7% of its own value. Three defensible anchors
give three answers for one cell (mixtral G=16 BLOCK_M=32): the published `A + B`
gives 0.647, the measured `t(1)` gives 0.452, and a refit that drops n=1 and n=2
gives 0.705. Four A100 fits go further and are not merely uncertain but
IMPOSSIBLE: their fitted `A + B` moves the weight set at 102% to 112% of the
card's 2039 GB/s pin rate.

So the number nobody can defend is not alpha. It is L.

THE FIX, in two lines of arithmetic. Write the traffic model out with the
activation term in it, at `n` M-tiles per expert:

    bytes(n) = W (1 + alpha (n - 1)) + Act1 n
    t(n)     = D + bytes(n) / BW                       D >= 0, the fixed cost

`W` is the weight set, exact from the config. `Act1` is the activation traffic
one extra tile carries, exact from the same bytes model the sweep already uses.
Differentiating in `n` and rearranging:

    B     = (W alpha + Act1) / BW
    alpha = (B BW - Act1) / W                                            (*)

ALPHA IS A ONE-PARAMETER FAMILY IN BW, THE ACHIEVED DRAM BANDWIDTH ON THE
MEMORY BRANCH, AND NOTHING ELSE IN (*) IS UNKNOWN. That is the whole
identification problem, stated in one sentence, and it is the sentence the
published point estimate hides: `B / (A + B)` silently picks one BW, and on four
A100 fits it picks one the card does not have.

BW IS BOUNDED AT BOTH ENDS, BY MEASUREMENT, WITHOUT COUNTERS.

  LOWER.  At n = 1 there is exactly ONE M-tile per expert, so every weight byte
          is read exactly once: `bytes(1) = W + Act1`, with no alpha in it. The
          kernel demonstrably completed that traffic in `t(1)` ms, and `t(1)`
          also contains the fixed cost `D >= 0`. So
                BW >= (W + Act1) / t(1) =: BW_1.
          This end needs no model at all. It is a measured time divided by a
          byte count that is fixed by the geometry.

          ASSUMPTION A, named because it is the ONE thing here that is assumed
          rather than measured: the branch does not achieve LESS bandwidth than
          the n=1 tread did. More tiles means more requests in flight, so the
          direction is the physical one, but it is an assumption and gate V6
          scores it -- where the measured anchor sits BELOW the fitted branch,
          the shortfall has to be inside the arm's own timing spread. It is, on
          every committed fit: the two negative elevations are -1.7 and -0.7
          spreads. Without ASSUMPTION A the LOWER end of the bracket does not
          exist at all, only the upper one does, and this file says that plainly
          instead of hiding it inside a bound.

  UPPER.  No read of `W` bytes can beat the rate this machine has been shown to
          sustain. `scripts/calibrate_hardware.py` measured four STREAM patterns
          on both cards on 2026-09-02 at the same commit as the sweeps; the
          largest is the ceiling, and the memory bus pin rate is the hard
          fallback behind it. So
                BW <= BW_ceiling.

Feeding both ends through (*) gives a BRACKET on alpha rather than a point:

    alpha in [ (B BW_1 - Act1) / W , (B BW_ceiling - Act1) / W ]

and every published alpha can be asked one question it has never been asked:
does it lie inside its own bracket, and does the bandwidth it implies exist.

WHY THE BRACKET IS NOT CIRCULAR. `B` is refitted here WITHOUT the anchor tread.
The published branch is fitted on a prefix that INCLUDES n=1, so `B / t(1)`
computed off the published slope would use the anchor twice. Dropping it moves
`B` by -0.03% to +0.79% across the 40 committed fits -- `B` is identified and
`L` is not, which is the cleanest available statement of what is wrong -- but
the bracket is built on the anchor-free slope regardless, because "small" is not
"independent".

WHAT THIS DOES NOT DO. It does not identify BW inside the bracket. A DRAM
counter would, and it is blocked: `ncu` fails with ERR_NVGPUCTRPERM on rented
pods and the RunPod image's `nsys` cannot convert its own capture. NO
COUNTER-FREE METHOD IDENTIFIES BW ON THE BRANCH, and this file says so rather
than inventing one. What it delivers instead is an interval whose two ends are
both measured quantities, which is a defensible object where the point estimate
was not.

WHICH ALPHA THIS BRACKETS, since `moe/bench/ai_model.py` (2026-09-02) showed
there are two. A ladder divides the WHOLE per-extra-M-tile cost by the weight
bytes, so what it returns is

    alpha_fitted = alpha_b + alpha_a (BM/BN) + BM/K

where alpha_b is the weight re-read the study means by "alpha" and alpha_a is
its unmeasured counterpart on the activations. This file brackets `B`, so it
brackets alpha_fitted, MINUS the first-order activation traffic it subtracts as
`Act1` -- the same quantity the reports print as `alpha-corrected`. Splitting
alpha_fitted into alpha_b and alpha_a needs three BLOCK_N values and belongs to
that module's lane; nothing here assumes a value for alpha_a. The bracket is
still the right correction to every published number, because every published
number is a value of this same composite.

THE MECHANISM, and the part of the evaluation's story that the committed data
REFUTES. The elevation separates on the swizzle with no overlap on 11 of the 12
A100 fits (G=1: 5.6, 9.8, 11.7, 14.3%; G>1: 16.9, 17.2, 17.7, 30.1, 33.2, 36.8,
37.7%), and the proposed mechanism was that a swizzle group of width G spans G
DIFFERENT experts at n=1 and can reuse nothing. That mechanism predicts the
elevation persists while n < G. It does not: at G=64 the per-tread residual
against the fitted branch runs +36.8%, +9.8%, +0.8%, -1.1%, ... and is inside
the noise from n=3 on. The anomaly is an n=1 and n=2 phenomenon whose SIZE grows
with G, not a deficit that lasts G treads. The twelfth fit is a direct
counterexample to the separation as stated: G=64 at BLOCK_M=128 elevates 1.7%,
below every G=1 value. This file therefore does not correct for the swizzle. It
brackets, which needs no mechanism, and it prints the residual profile so the
mechanism claim can be read off the data instead of asserted.

TWO MODES.

  --rescore  Free. Scores every committed report under `results/published/`,
             emits the bracket for every anchorable fit, and states the size of
             the correction to every published alpha. No GPU, no pod, seconds.
  --measure  The GPU arm. Re-measures the anchor tread at every GROUP_SIZE_M so
             the objection "the anchor is measured at a different reuse
             condition" is answered by measurement rather than by argument; and
             re-measures the branch from n=2 up so the slope and the anchor come
             from one process at one clock state. It also re-times the read
             ceiling on the ACTUAL weight buffers and refuses to score if the
             committed calibration is below what those buffers achieve, because
             a ceiling under the data is not a ceiling.

EXIT CODES. 0 every gate passed. 1 only CLAIM gates failed, which is a result
and not a broken run: the claims those gates carry are refuted and the report
says which. 2 a VALIDITY gate failed, so the instrument is broken and no number
on the page may be quoted. 3 nothing was scored or measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO / "moe" / "bench" / "hardware"
PUBLISHED = REPO / "results" / "published"

PASS, FAIL, REFUSED = "PASS", "FAIL", "REFUSED"

#: alpha is a fraction of a weight re-read. Outside [0, 1] it is not a physical
#: quantity, so a bracket end past either edge is CLIPPED and the clip is
#: printed. Silently keeping alpha_hi = 1.26 would put a "bracket" around values
#: the model forbids and make the containment gate pass on nonsense.
ALPHA_FLOOR, ALPHA_ROOF = 0.0, 1.0

#: ASSUMPTION A's tolerance, in units of the arm's OWN median timing spread. A
#: measured anchor a little BELOW the fitted branch is the anchor sitting ON the
#: branch, which is what a healthy fit looks like; a long way below would be
#: the branch running slower than n=1 and would delete the bracket's lower end.
#: 2.0 spreads is where the committed data separates: the two negative
#: elevations are -1.73 and -0.70 spreads and the next positive one is +0.21.
ANCHOR_BELOW_BRANCH_SPREADS = 2.0

#: A compute reference whose implied dense throughput is below this fraction of
#: the card's own measured dense peak is POISONED, and every membership decision
#: it made is void. The BN=256 arm is the known case: its BLOCK_M=256 reference
#: takes 249.765 ms for one tile on the A100 against 5.724 ms for the identical
#: setting in its BN=64 twin, which is 1.4% of the card's 262 TFLOP/s. The
#: qualification test in the sweep checks PROPORTIONALITY and never checks
#: LEVEL, and a line 44x too steep is still perfectly proportional. 0.25 is far
#: below every healthy arm (the lowest is 63.7%) and far above the poisoned one.
POISONED_REFERENCE_FRACTION = 0.25

#: How far the anchor-free slope may sit from the published `slope_memory`
#: before the two are not the same branch. This is a REPRODUCTION check on the
#: published line, not a tolerance on the physics: refitting the published
#: prefix by OLS must return the published slope to floating point, or this
#: script is scoring a line the report never drew.
SLOPE_REPRODUCTION_REL = 1e-6

#: Fewer scored fits than this and the run examined too little to carry a
#: verdict. A check that examined nothing also reports zero failures, and this
#: study has shipped that shape before.
MIN_SCORED_FITS = 20

#: The pooled refit the 2026-09-01 arm published and SURFACE.txt scores against.
POOLED_ALPHA = 0.558


# --------------------------------------------------------------------------
# Bytes. Everything here is arithmetic over the model config -- no timing, no
# device -- so it is exact and testable on a laptop.
# --------------------------------------------------------------------------

def weight_bytes(cfg, dtype: str) -> int:
    """The whole expert set, w1 and w2, at the working dtype.

    The ladder holds routing balanced and every expert receives rows at every
    tread, so this is the traffic ONE full weight read moves, at every n. It is
    the numerator of the anchor and it has no fitted parameter in it.
    """
    return cfg.weight_bytes(dtype)


def activation_bytes_per_row(cfg) -> int:
    """x_perm, h_up, h_act, y_perm: the traffic that grows WITH the batch.

    `2 H + 3 F` elements at the ACTIVATION dtype, which is 2 bytes in every arm
    this study has published including the fp8-weight ones -- charging
    activations at the weight dtype would report traffic that was never moved.

    Transcribed from `scripts/block_m_crossing_sweep.py:activation_bytes_per_row`
    rather than imported: that file is under concurrent edit by another workflow
    and a bracket that changes because someone else refactored a helper is not a
    bracket. `tests/test_memory_branch_anchor.py` cross-checks the two whenever
    the sweep is importable, so a divergence is caught rather than assumed away.
    """
    return (2 * cfg.hidden_size + 3 * cfg.intermediate_size) * 2


def anchor_bytes(cfg, dtype: str, block_m: int) -> tuple[int, int]:
    """`(W, Act1)`: the weight set, and what ONE extra M-tile per expert carries.

    At n = 1 the kernel moves `W + Act1` and not one byte more: there is exactly
    one M-tile per expert, so no weight byte can be read twice whatever L2 does.
    That is the fact the whole bracket rests on and it is a statement about the
    geometry, not about the cache.
    """
    return weight_bytes(cfg, dtype), cfg.num_experts * block_m * activation_bytes_per_row(cfg)


def alpha_at_bandwidth(slope_ms: float, bandwidth_gbps: float, w_bytes: int,
                       act1_bytes: int) -> float:
    """`(B BW - Act1) / W`, equation (*) of the module docstring.

    Monotone increasing in `BW`, which is why bounding the bandwidth bounds
    alpha and why the two ends of the bracket come from the two ends of the
    bandwidth interval with no search.
    """
    moved = slope_ms * 1e-3 * bandwidth_gbps * 1e9
    return (moved - act1_bytes) / w_bytes


def bandwidth_for_alpha(alpha: float, slope_ms: float, w_bytes: int,
                        act1_bytes: int) -> float:
    """(*) inverted: the DRAM bandwidth a quoted alpha implies, in GB/s.

    This is the question the published numbers were never asked. Four A100 fits
    answer it above the card's pin rate, which refutes them outright rather than
    merely widening them.
    """
    if slope_ms <= 0:
        raise ValueError("a non-positive branch slope implies no bandwidth")
    return (alpha * w_bytes + act1_bytes) / (slope_ms * 1e-3) / 1e9


# --------------------------------------------------------------------------
# The bracket.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Bracket:
    """alpha between two measured bandwidths, plus everything used to build it.

    `raw_lo`/`raw_hi` are before the [0, 1] clip and are kept because a raw_hi
    above 1 is itself information: it says the branch cannot be running at the
    card's ceiling, since alpha > 1 would mean an extra tile misses more than a
    whole weight read.
    """

    block_m: int
    slope_ms: float
    slope_ms_published: float
    anchor_ms: float
    w_bytes: int
    act1_bytes: int
    bw_anchor_gbps: float
    bw_ceiling_gbps: float
    bw_pin_gbps: float
    raw_lo: float
    raw_hi: float
    raw_hi_pin: float

    @property
    def lo(self) -> float:
        return min(max(self.raw_lo, ALPHA_FLOOR), ALPHA_ROOF)

    @property
    def hi(self) -> float:
        return min(max(self.raw_hi, ALPHA_FLOOR), ALPHA_ROOF)

    @property
    def hi_pin(self) -> float:
        return min(max(self.raw_hi_pin, ALPHA_FLOOR), ALPHA_ROOF)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def clipped(self) -> bool:
        return self.raw_hi > ALPHA_ROOF or self.raw_lo < ALPHA_FLOOR

    def contains(self, value: float, tol: float = 1e-9) -> bool:
        return self.lo - tol <= value <= self.hi + tol

    def ai_cap(self, dtype_b: int, alpha: float) -> float:
        """`2 BM / (b alpha)`: the arithmetic intensity this tile height caps at.

        The same expression the sweep publishes as `ai_cap`, and it is the
        CORRECT one as long as the alpha put into it is the fitted composite.
        `moe/bench/ai_model.py` gives the full cap as

            2 / (b (alpha_b/BM + alpha_a/BN + 1/K))

        and dividing `alpha_fitted = alpha_b + alpha_a (BM/BN) + BM/K` by BM
        turns that denominator into `alpha_fitted / BM` exactly. So the two
        agree; what the study got wrong was the SPLIT, not the ceiling.

        TWO WAYS THIS IS CONSERVATIVE, both in the direction that makes the
        BLOCK_M <= 64 claim harder rather than easier to keep. Taken at the
        bracket's LOW alpha it is the LARGEST cap the anchor ambiguity allows.
        And the alpha handed to it has the first-order activation traffic
        subtracted, which removes a positive term from the denominator and so
        OVERSTATES the cap again. A tile that still cannot reach the ridge under
        both of those has not been helped over the line by either.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive to cap arithmetic intensity")
        return 2.0 * self.block_m / (dtype_b * alpha)


def bracket_alpha(slope_ms: float, slope_ms_published: float, anchor_ms: float,
                  block_m: int, w_bytes: int, act1_bytes: int,
                  bw_ceiling_gbps: float, bw_pin_gbps: float) -> Bracket:
    """Both ends, from two measured times and one measured ceiling.

    REFUSES rather than defaulting. A non-positive anchor or slope is an absent
    measurement, and returning 0.0 for an absent measurement is how a study
    reports a confident number for something it never ran.
    """
    if anchor_ms <= 0:
        raise ValueError("no measured n=1 tread: the anchor is absent, not zero")
    if slope_ms <= 0:
        raise ValueError("no memory-branch slope: alpha is undefined, not zero")
    if bw_ceiling_gbps <= 0 or bw_pin_gbps <= 0:
        raise ValueError("no measured bandwidth ceiling for this card")
    bw_anchor = (w_bytes + act1_bytes) / (anchor_ms * 1e-3) / 1e9
    return Bracket(
        block_m=block_m, slope_ms=slope_ms, slope_ms_published=slope_ms_published,
        anchor_ms=anchor_ms, w_bytes=w_bytes, act1_bytes=act1_bytes,
        bw_anchor_gbps=bw_anchor, bw_ceiling_gbps=bw_ceiling_gbps,
        bw_pin_gbps=bw_pin_gbps,
        raw_lo=alpha_at_bandwidth(slope_ms, bw_anchor, w_bytes, act1_bytes),
        raw_hi=alpha_at_bandwidth(slope_ms, bw_ceiling_gbps, w_bytes, act1_bytes),
        raw_hi_pin=alpha_at_bandwidth(slope_ms, bw_pin_gbps, w_bytes, act1_bytes),
    )


# --------------------------------------------------------------------------
# Fitting. One OLS, used twice: once on the published prefix to prove we are
# scoring the published line, once without the anchor to build the bracket.
# --------------------------------------------------------------------------

def ols(xs, ys) -> tuple[float, float]:
    """Ordinary least squares `y = a + b x`. Two points give an exact line."""
    if len(xs) < 2:
        raise ValueError("a line needs two points; fewer is not a fit")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("all treads at one tile count: the slope is undefined")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    return my - b * mx, b


def residual_profile(points, intercept: float, slope: float) -> list[tuple[int, float]]:
    """`(n, (t - fitted) / t)` per tread: where the branch misses, and by how much.

    Printed rather than summarised because the elevation's SHAPE is the evidence
    that decides between "the swizzle spans G experts for n < G" and "n = 1 and
    n = 2 are special". A single number for the misfit cannot tell those apart.
    """
    return [(int(n), (ms - (intercept + slope * n)) / ms) for n, ms in points if ms > 0]


# --------------------------------------------------------------------------
# Card calibration. The upper end of the bracket, and the only number here that
# comes from a different process than the ladder.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    slug: str
    name: str
    checked_on: str
    measured_commit: str
    patterns: dict[str, float]
    ceiling_pattern: str
    ceiling_gbps: float
    pin_gbps: float
    dense_tflops: dict[str, float]
    ridge: float

    def describe(self) -> str:
        pats = ", ".join(f"{k} {v:.0f}" for k, v in sorted(self.patterns.items()))
        return (f"{self.name}: ceiling {self.ceiling_gbps:.1f} GB/s "
                f"({self.ceiling_pattern}), pin {self.pin_gbps:.1f}, "
                f"ridge {self.ridge:.2f} FLOP/byte, measured {self.checked_on} "
                f"@ {self.measured_commit[:8]}  [{pats}]")


def available_calibrations(directory: Path | None = None) -> list[str]:
    d = directory or HARDWARE_DIR
    return sorted(p.name[len("measured_"):-len(".yaml")]
                  for p in d.glob("measured_*.yaml"))


def load_calibration(slug: str, directory: Path | None = None) -> Calibration:
    """The card's own contemporaneous measurement, or a refusal.

    REFUSES on a missing file rather than falling back to a datasheet. A spec
    sheet is a pin rate, not an achieved rate, and quoting one as the ceiling
    would widen every bracket by whatever this card cannot actually reach --
    which is 12% on the H200 and 8% on the A100. It is also exactly the defect
    that put a stale H200 ridge band of 160.3 on all seven A100 reports.
    """
    import yaml

    path = (directory or HARDWARE_DIR) / f"measured_{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no measured calibration at {path}. The bracket's upper end is a "
            f"MEASURED ceiling for THIS card; run scripts/calibrate_hardware.py "
            f"on it. Known: {', '.join(available_calibrations(directory)) or 'none'}")
    data = yaml.safe_load(path.read_text())
    if not data.get("verified"):
        raise ValueError(f"{path} is marked verified: false; it may not set a ceiling")
    detail = data.get("detail") or {}
    patterns = {p["pattern"]: float(p["gbps"]) for p in (detail.get("bandwidth_patterns") or [])}
    if not patterns:
        raise ValueError(f"{path} records no bandwidth patterns; there is no ceiling in it")
    # The MAXIMUM over patterns, not the canonical triad. The upper end of the
    # bracket must be a rate this machine has been SHOWN to sustain, and taking
    # the largest demonstrated rate is the conservative direction: it widens the
    # bracket and makes fewer published fits impossible. On the A100 that is
    # `write` at 1879.1 against triad's 1799.4, and the four impossible fits are
    # impossible against either.
    ceiling_pattern = max(patterns, key=lambda k: patterns[k])
    observed = data.get("observed") or {}
    pin = float(observed.get("pin_rate_gbps") or 0.0)
    if pin <= 0:
        raise ValueError(f"{path} records no memory bus pin rate; the hard bound is missing")
    bw_tb_s = float(data["memory"]["bandwidth_tb_s"])
    dense = {k: float(v) for k, v in (data.get("compute_dense_tflops") or {}).items() if v}
    if not dense:
        raise ValueError(f"{path} records no dense compute peak; the ridge is undefined")
    return Calibration(
        slug=slug, name=data.get("name", slug), checked_on=str(data.get("checked_on", "")),
        measured_commit=str(data.get("measured_commit", "")), patterns=patterns,
        ceiling_pattern=ceiling_pattern, ceiling_gbps=patterns[ceiling_pattern], pin_gbps=pin,
        dense_tflops=dense, ridge=dense.get("bf16", max(dense.values())) / bw_tb_s)


def calibration_slug_for(arm_name: str, slugs) -> str | None:
    """Which card an arm directory belongs to.

    LONGEST match wins. `nvidia_h200` is a prefix of nothing here, but
    `nvidia_a100_sxm4_80gb` contains no other slug and a shortest-match rule
    would happily bind an A100 arm to an H200 calibration if a shorter slug ever
    appeared -- which is the same class of error as the stale 160.3 ridge: a
    number from the wrong machine, printed without a complaint.
    """
    hits = [s for s in slugs if s in arm_name]
    return max(hits, key=len) if hits else None


# --------------------------------------------------------------------------
# Reading the committed reports.
# --------------------------------------------------------------------------

@dataclass
class ScoredFit:
    """One ladder fit, re-anchored. Every field is either measured or derived
    from a measured field by arithmetic with no free parameter."""

    arm: str
    card: str
    model: str
    dtype: str
    group_m: int
    block_n: int
    num_stages: int
    block_m: int
    treads: int
    alpha_published: float
    alpha_published_corrected: float
    slope_published: float
    slope_refit_full: float
    slope_refit_no_anchor: float
    anchor_ms: float
    fitted_level_ms: float
    anchor_elevation: float
    #: The arm's own median timing spread, so an elevation can be read against
    #: the noise it was measured through instead of against zero.
    timing_spread: float
    elevation_in_spreads: float
    bw_anchor_gbps: float
    bw_published_gbps: float
    bw_ceiling_gbps: float
    bw_pin_gbps: float
    alpha_lo: float
    alpha_hi: float
    alpha_hi_pin: float
    clipped: bool
    contains_published: bool
    contains_published_corrected: bool
    contains_pooled: bool
    physical_vs_ceiling: bool
    physical_vs_pin: bool
    ridge: float
    cap_over_ridge_at_lo: float
    residuals: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Refusal:
    arm: str
    model: str
    group_m: int
    block_n: int
    block_m: int
    reason: str


def implied_reference_tflops(cfg, block_m: int, slope_per_tile_ms: float) -> float:
    """Dense throughput the compute reference implies, in TFLOP/s.

    One tread of the reference ladder is `E` M-tiles of `BLOCK_M` rows, and each
    row costs `6 F H` flops across up and down. A reference that implies 1.4% of
    the card's dense peak did not measure a compute branch, whatever its
    proportionality residual said.
    """
    rows = cfg.num_experts * block_m
    flops = 6.0 * rows * cfg.intermediate_size * cfg.hidden_size
    return flops / (slope_per_tile_ms * 1e-3) / 1e12


def score_report(path: Path, cal: Calibration) -> tuple[list[ScoredFit], list[Refusal]]:
    """Every identifiable ladder in one report, bracketed or refused."""
    report = json.loads(path.read_text())
    model = report["model"]
    if model not in MODEL_CONFIGS:
        return [], [Refusal(path.parent.name, model, -1, -1, -1,
                            f"model {model!r} is not in MODEL_CONFIGS")]
    cfg = MODEL_CONFIGS[model]
    dtype = report["dtype"]
    # The arm's OWN noise, not a constant. Gate V6 reads every elevation against
    # it, and a fixed percentage would call the same shortfall significant on a
    # 0.39%-spread arm and on a 1.82%-spread one.
    spread = float(report.get("timing_spread_median") or 0.0)
    fixed = report["fixed"]
    group_m, block_n = int(fixed["GROUP_SIZE_M"]), int(fixed["BLOCK_SIZE_N"])
    stages = int(fixed["num_stages"])
    arm = path.parent.name
    fits: list[ScoredFit] = []
    refusals: list[Refusal] = []

    ref = report.get("compute_reference") or {}
    ref_bm, ref_slope = ref.get("block_m"), ref.get("slope_per_tile")
    poisoned = ""
    if ref_bm and ref_slope:
        implied = implied_reference_tflops(cfg, int(ref_bm), float(ref_slope))
        peak = cal.dense_tflops.get("bf16", max(cal.dense_tflops.values()))
        if implied < POISONED_REFERENCE_FRACTION * peak:
            poisoned = (f"compute reference at BLOCK_M={ref_bm} implies "
                        f"{implied:.1f} TFLOP/s, {implied / peak:.1%} of this card's "
                        f"measured {peak:.0f}; it classified every tread in this "
                        f"report and cannot be trusted to have found a memory branch")

    if poisoned:
        # ONE refusal for the whole report, raised BEFORE the identifiability
        # filter. The known-corrupt BN=256 arms print zero identifiable ladders
        # under a caption blaming tread count, so a check that only looked at
        # identifiable ladders would fire on nothing and read as a clean PASS --
        # which is the same silent-zero shape this file exists to refuse.
        ident = sum(1 for lad in report["ladder"].values() if lad.get("slope_memory"))
        return [], [Refusal(
            arm, model, group_m, block_n, -1,
            f"{poisoned}. {ident} identifiable ladder(s) in this report are excluded, "
            "and its 'not identifiable' verdicts are not evidence about the tile")]

    for key, ladder in sorted(report["ladder"].items(), key=lambda kv: int(kv[0])):
        block_m = int(key)
        if ladder.get("slope_memory") is None:
            continue                      # not identifiable upstream; nothing to re-anchor
        points = [(int(n), float(ms)) for n, ms in ladder["points"] if float(ms) > 0]
        by_n = dict(points)
        if 1 not in by_n:
            refusals.append(Refusal(
                arm, model, group_m, block_n, block_m,
                f"no n=1 tread in this ladder (lowest is n={min(by_n) if by_n else 'none'}); "
                "the anchor is a MEASURED single-tile time and there is nothing to "
                "substitute for it"))
            continue
        # `memory_points` is a COUNT of LEADING treads, not a tile-count cutoff.
        # `points` is already in ascending tile order, so the memory branch is
        # the first k entries. Selecting by `n <= k` instead would agree on this
        # grid -- where the tile counts happen to run 1, 2, 3, ... -- and would
        # quietly select the wrong treads on any grid that skipped a value.
        prefix = points[:ladder["memory_points"]]
        if len(prefix) < 4:
            refusals.append(Refusal(
                arm, model, group_m, block_n, block_m,
                f"memory branch has {len(prefix)} treads; dropping the anchor leaves "
                "too few to refit a slope independently of it"))
            continue
        xs = [float(n) for n, _ in prefix]
        ys = [ms for _, ms in prefix]
        a_full, b_full = ols(xs, ys)
        _, b_free = ols(xs[1:], ys[1:])
        anchor = by_n[1]
        w_bytes, act1 = anchor_bytes(cfg, dtype, block_m)
        br = bracket_alpha(b_free, float(ladder["slope_memory"]), anchor, block_m,
                           w_bytes, act1, cal.ceiling_gbps, cal.pin_gbps)
        a_pub = float(ladder["alpha"])
        a_pub_c = float(ladder["alpha_corrected"])
        bw_pub = bandwidth_for_alpha(a_pub, b_full, w_bytes, act1)
        fits.append(ScoredFit(
            arm=arm, card=cal.slug, model=model, dtype=dtype, group_m=group_m,
            block_n=block_n, num_stages=stages, block_m=block_m, treads=len(prefix),
            alpha_published=a_pub, alpha_published_corrected=a_pub_c,
            slope_published=float(ladder["slope_memory"]), slope_refit_full=b_full,
            slope_refit_no_anchor=b_free, anchor_ms=anchor,
            fitted_level_ms=a_full + b_full,
            anchor_elevation=(anchor - (a_full + b_full)) / anchor,
            timing_spread=spread,
            elevation_in_spreads=((anchor - (a_full + b_full)) / anchor / spread
                                  if spread > 0 else math.inf),
            bw_anchor_gbps=br.bw_anchor_gbps, bw_published_gbps=bw_pub,
            bw_ceiling_gbps=cal.ceiling_gbps, bw_pin_gbps=cal.pin_gbps,
            alpha_lo=br.lo, alpha_hi=br.hi, alpha_hi_pin=br.hi_pin, clipped=br.clipped,
            contains_published=br.contains(a_pub),
            contains_published_corrected=br.contains(a_pub_c),
            contains_pooled=br.contains(POOLED_ALPHA),
            physical_vs_ceiling=bw_pub <= cal.ceiling_gbps,
            physical_vs_pin=bw_pub <= cal.pin_gbps,
            ridge=cal.ridge,
            cap_over_ridge_at_lo=br.ai_cap(dtype_bytes(dtype), br.lo) / cal.ridge,
            residuals=residual_profile(prefix, a_full, b_full),
        ))
    return fits, refusals


def scan_published(root: Path) -> tuple[list[ScoredFit], list[Refusal], dict[str, Calibration]]:
    slugs = available_calibrations()
    fits: list[ScoredFit] = []
    refusals: list[Refusal] = []
    cals: dict[str, Calibration] = {}
    for report in sorted(root.glob("*/*.report.json")):
        arm = report.parent.name
        if (report.parent / "SUPERSEDED").exists():
            refusals.append(Refusal(arm, "-", -1, -1, -1,
                                    "arm is marked SUPERSEDED; counting it would "
                                    "weight its rows twice"))
            continue
        slug = calibration_slug_for(arm, slugs)
        if slug is None:
            refusals.append(Refusal(arm, "-", -1, -1, -1,
                                    f"no calibration slug in the arm name; known: "
                                    f"{', '.join(slugs) or 'none'}"))
            continue
        if slug not in cals:
            cals[slug] = load_calibration(slug)
        f, r = score_report(report, cals[slug])
        fits += f
        refusals += r
    return fits, refusals, cals


# --------------------------------------------------------------------------
# Gates. A number against a threshold, PASS or FAIL, and every FAIL says what
# it invalidates. VALIDITY gates are about the instrument and a FAIL there voids
# the page; CLAIM gates carry the findings and a FAIL there is a result.
# --------------------------------------------------------------------------

@dataclass
class Gate:
    number: str
    kind: str            # "VALIDITY" or "CLAIM"
    claim: str
    verdict: str
    measured: str
    threshold: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    def render(self) -> list[str]:
        out = [f"GATE {self.number:3s} {self.kind:8s} {self.verdict:7s} {self.claim}",
               f"                        measured {self.measured}   gate {self.threshold}"]
        out += [f"                        {line}" for line in self.lines]
        if self.verdict == FAIL and self.invalidates:
            out.append(f"                        INVALIDATES: {self.invalidates}")
        return out


def gate_v1_non_vacuity(fits, refusals) -> Gate:
    cards = {f.card for f in fits}
    ok = len(fits) >= MIN_SCORED_FITS and len(cards) >= 1
    return Gate(
        "V1", "VALIDITY", "the run actually scored something",
        PASS if ok else FAIL,
        f"{len(fits)} fits over {len(cards)} card(s), {len(refusals)} refusal(s)",
        f">= {MIN_SCORED_FITS} fits on >= 1 card",
        "every count and every interval below, which would be a report on an "
        "empty set reporting zero failures")


def gate_v2_slope_reproduction(fits) -> Gate:
    """Are we scoring the line the reports drew, or a different one."""
    worst, where = -1.0, ""
    for f in fits:
        rel = abs(f.slope_refit_full / f.slope_published - 1.0)
        if rel > worst:
            worst, where = rel, f"{f.arm}/{f.model} G={f.group_m} BM={f.block_m}"
    if worst < 0.0:
        return Gate("V2", "VALIDITY", "OLS on the published prefix returns the "
                    "published slope", FAIL, "no fits to reproduce",
                    f"<= {SLOPE_REPRODUCTION_REL:.0e}", "the correction table")
    ok = worst <= SLOPE_REPRODUCTION_REL
    return Gate(
        "V2", "VALIDITY", "OLS on the published prefix returns the published slope",
        PASS if ok else FAIL,
        f"worst relative difference {worst:.2e} at {where or 'n/a'}",
        f"<= {SLOPE_REPRODUCTION_REL:.0e}",
        "the correction table, which would be comparing this script's line "
        "against a published alpha computed from a different one")


def gate_v3_anchor_present(fits, refusals) -> Gate:
    """Anchorless fits are REFUSED and listed, never defaulted."""
    missing = [r for r in refusals if "no n=1 tread" in r.reason]
    ok = all(f.anchor_ms > 0 for f in fits)
    return Gate(
        "V3", "VALIDITY", "every scored fit carries a MEASURED n=1 tread",
        PASS if ok else FAIL,
        f"{len(fits)} scored with an anchor, {len(missing)} refused for want of one",
        "no scored fit may have anchor_ms <= 0",
        "the lower end of every bracket, which would be a division by a time "
        "that was never measured")


def gate_v4_poisoned_reference(refusals, expected_arms: int) -> Gate:
    """NON-VACUITY on the detector itself: it has to fire on the known case.

    The BN=256 arm is corrupt on both cards and the study's own qualification
    test cleared it, because that test checks whether the reference ladder is
    PROPORTIONAL to its tile count and never checks its LEVEL. If this level
    check fires on nothing, it is not evidence that the arms are clean; it is
    evidence that the check does not work.
    """
    fired = {r.arm + "/" + r.model for r in refusals if "compute reference" in r.reason}
    ok = len(fired) >= expected_arms
    return Gate(
        "V4", "VALIDITY", "the poisoned-compute-reference check fires on the known case",
        PASS if ok else FAIL,
        f"fired on {len(fired)} arm/model pair(s): {', '.join(sorted(fired)) or 'none'}",
        f">= {expected_arms} (the BN=256 arms, whose reference is 43.6x too slow "
        f"on the A100 and 4.1x on the H200)",
        "every OTHER arm's clean bill of health, since a check that fires on "
        "nothing reports zero failures whether or not there are any")


def gate_v5_bracket_order(fits) -> Gate:
    """Is the interval an interval.

    Reported at the TIGHTEST case -- the anchor closest to the ceiling -- because
    that is the fit where an inversion would appear first, and a mean over 40
    comfortable fits would hide it.
    """
    tight = max(fits, key=lambda f: f.bw_anchor_gbps / f.bw_ceiling_gbps, default=None)
    if tight is None:
        return Gate("V5", "VALIDITY", "the bracket is ordered", FAIL, "no fits", "lo <= hi",
                    "everything below")
    ratio = tight.bw_anchor_gbps / tight.bw_ceiling_gbps
    ok = all(f.alpha_lo <= f.alpha_hi + 1e-12 for f in fits) and ratio <= 1.0
    return Gate(
        "V5", "VALIDITY", "the anchor rate never exceeds the card's measured ceiling",
        PASS if ok else FAIL,
        f"tightest {ratio:.1%} at {tight.arm}/{tight.model} G={tight.group_m} "
        f"BM={tight.block_m} ({tight.bw_anchor_gbps:.0f} against "
        f"{tight.bw_ceiling_gbps:.0f} GB/s)",
        "<= 100%",
        "the bracket outright: an anchor faster than the ceiling means the "
        "ceiling is not one, and the interval would be inverted")


def gate_v6_assumption_a(fits) -> Gate:
    """ASSUMPTION A, scored: does the branch ever run slower than the anchor.

    The bracket's LOWER end is the only part of this file that rests on
    something other than a measurement, and it rests on exactly one thing: the
    memory branch does not achieve less bandwidth than the n=1 tread did. Where
    the measured anchor sits BELOW the fitted branch that assumption is being
    contradicted, and the question is whether it is being contradicted by more
    than the arm's own noise.

    A FAIL does not merely widen the interval, it DELETES its lower end: with no
    floor on the branch's bandwidth there is no floor on alpha, and the honest
    report would then be an upper bound alone.
    """
    below = [f for f in fits if f.anchor_elevation < 0]
    worst = min(below, key=lambda f: f.elevation_in_spreads, default=None)
    ok = all(abs(f.elevation_in_spreads) <= ANCHOR_BELOW_BRANCH_SPREADS
             for f in below)
    return Gate(
        "V6", "VALIDITY", "ASSUMPTION A: no branch runs slower than its own anchor",
        PASS if ok else FAIL,
        (f"{len(below)} of {len(fits)} anchors sit below the fitted branch; "
         f"deepest {worst.elevation_in_spreads:+.2f} spreads "
         f"({worst.anchor_elevation:+.2%} against a {worst.timing_spread:.2%} "
         f"spread) at {worst.arm[:26]}/{worst.model} G={worst.group_m} "
         f"BM={worst.block_m}") if worst else
        f"0 of {len(fits)} anchors sit below the fitted branch",
        f"within {ANCHOR_BELOW_BRANCH_SPREADS:.0f} of the arm's own timing spread",
        "the LOWER end of every bracket, which is the only part of this file "
        "that is assumed rather than measured. Without it the honest report is "
        "an upper bound on alpha and nothing else")


def gate_c1_physicality(fits) -> Gate:
    bad = [f for f in fits if not f.physical_vs_pin]
    worst = max(fits, key=lambda f: f.bw_published_gbps / f.bw_pin_gbps, default=None)
    return Gate(
        "C1", "CLAIM", "every published alpha implies a bandwidth the card has",
        PASS if not bad else FAIL,
        f"{len(bad)} of {len(fits)} fits imply more than the pin rate; worst "
        f"{worst.bw_published_gbps:.0f} GB/s against {worst.bw_pin_gbps:.0f} "
        f"({worst.bw_published_gbps / worst.bw_pin_gbps:.1%}) at {worst.arm}/"
        f"{worst.model} G={worst.group_m} BM={worst.block_m}" if worst else "no fits",
        "0 fits above the pin rate",
        "the listed alphas as point estimates. They are not uncertain, they are "
        "impossible: no fitted level may move the weight set faster than the bus.",
        [f"  {f.arm[:26]:26s} {f.model[:14]:14s} G={f.group_m:2d} BM={f.block_m:3d}  "
         f"alpha {f.alpha_published:.3f} implies {f.bw_published_gbps:6.0f} GB/s "
         f"= {f.bw_published_gbps / f.bw_pin_gbps:.1%} of pin" for f in bad])


def why_outside(f: ScoredFit) -> str:
    """WHICH way a published alpha misses its bracket, said out loud.

    The directions are different defects and lumping them into "outside" would
    hide most of them. ABOVE ROOF is a published alpha greater than 1, which the
    model forbids outright: an extra M-tile cannot miss more than a whole weight
    read. ABOVE means the fitted level is too small for the card's bandwidth --
    the extrapolation the anchor exists to replace. Everything else is BELOW
    `alpha_lo`, and there the SIGN OF THE ANCHOR'S ELEVATION decides whether the
    published point is wrong or the lower end of the bracket is.

    THE SIGN IS LOAD BEARING AND AN EARLIER VERSION TOOK ITS ABSOLUTE VALUE.
    `elevation_in_spreads` is `(anchor - fitted_level) / anchor / spread`, so it
    is POSITIVE when the measured n=1 tread is SLOWER than the fitted branch --
    the normal case, ASSUMPTION A comfortably satisfied, `alpha_lo` sound. It is
    NEGATIVE when the anchor sits below its own branch, which is exactly the
    condition that strains ASSUMPTION A and inflates `alpha_lo`. Excusing a miss
    as "the anchor lies ON the branch" is only available in the negative case;
    `abs()` handed that exculpation to three positive-elevation fits whose
    anchors sat comfortably ABOVE the branch, and the old fall-through then told
    a +3.51-spread fit that its tread was FASTER than the fitted level, two lines
    under a table printing +1.69%. Both bugs were legibility, not arithmetic --
    which is the half of a FAIL a reader actually acts on.

    The number quoted is the MISS -- how far below `alpha_lo` the published alpha
    sits, in alpha -- because that is the quantity C2 scored. The elevation is
    printed beside it as the evidence for the reading, never as the miss itself.
    """
    if f.alpha_published_corrected > ALPHA_ROOF:
        return "ABOVE ROOF: alpha > 1 is not a fraction of a weight read"
    if f.alpha_published_corrected > f.alpha_hi:
        return "ABOVE: the fitted level is too small for the card's bandwidth"
    miss = f.alpha_lo - f.alpha_published_corrected
    elev = f.elevation_in_spreads
    if elev >= 0.0:
        return (f"BELOW alpha_lo by {miss:.3f} in alpha; the anchor sits "
                f"{elev:+.2f} spreads ABOVE its own branch, so ASSUMPTION A "
                "holds and the lower end stands: the published point really is "
                "under its own bound")
    if -elev <= ANCHOR_BELOW_BRANCH_SPREADS:
        return (f"BELOW alpha_lo by {miss:.3f} in alpha; the anchor sits "
                f"{elev:+.2f} spreads BELOW its own branch, within noise of "
                "lying ON it, so ASSUMPTION A is strained and alpha_lo is "
                "inflated -- the interval being tight rather than the published "
                "point being wrong")
    return (f"BELOW alpha_lo by {miss:.3f} in alpha; the anchor sits {elev:+.2f} "
            "spreads BELOW its own branch, past noise. ASSUMPTION A FAILS at "
            "this cell, so alpha_lo may not be quoted here at all and the miss "
            "is evidence about the bracket, not about the published point")


def gate_c2_containment(fits) -> Gate:
    out = [f for f in fits if not f.contains_published_corrected]
    return Gate(
        "C2", "CLAIM", "every published alpha lies inside its own anchor bracket",
        PASS if not out else FAIL,
        f"{len(out)} of {len(fits)} published alpha-corrected values fall outside",
        "0 outside",
        "the listed point estimates. The bracket's two ends are a measured time "
        "and a measured ceiling; a value outside it is not supported by either.",
        [f"  {f.arm[:26]:26s} {f.model[:14]:14s} G={f.group_m:2d} BM={f.block_m:3d}  "
         f"published {f.alpha_published_corrected:.3f} vs bracket "
         f"[{f.alpha_lo:.3f}, {f.alpha_hi:.3f}]  {why_outside(f)}" for f in out])


def gate_c3_tile_cap(fits) -> Gate:
    """The study's one surviving result, re-run at the anchor's most generous end.

    The cap claim is safest at the LOW alpha, because a smaller alpha means a
    HIGHER arithmetic-intensity cap and so the best chance a BLOCK_M <= 64
    kernel has of reaching its compute roof. If it still cannot at alpha_lo, the
    anchor ambiguity does not touch the claim.

    Scored against each card's OWN ridge, not the 160.3 stamped on every report,
    and with the cap formula `moe/bench/ai_model.py` corrected -- which agrees
    with `2 BM / (b alpha)` once the alpha is the fitted composite, and which
    this gate over-states anyway because the activation term is subtracted from
    that alpha. Both errors run toward a PASS being harder to earn.
    """
    small = [f for f in fits if f.block_m <= 64]
    if not small:
        return Gate("C3", "CLAIM", "BLOCK_M <= 64 caps below the ridge", FAIL,
                    "no BLOCK_M <= 64 fits scored", "< 1.0 for every fit",
                    "the study's one surviving result, which cannot be checked here")
    worst = max(small, key=lambda f: f.cap_over_ridge_at_lo)
    ok = worst.cap_over_ridge_at_lo < 1.0
    by_bm = {}
    for f in small:
        by_bm.setdefault(f.block_m, []).append(f.cap_over_ridge_at_lo)
    return Gate(
        "C3", "CLAIM", "at the bracket's LOWEST alpha, BLOCK_M <= 64 still caps below the ridge",
        PASS if ok else FAIL,
        f"worst {worst.cap_over_ridge_at_lo:.3f} of the ridge at {worst.arm[:26]}/"
        f"{worst.model} G={worst.group_m} BM={worst.block_m}",
        "< 1.000 for every BLOCK_M <= 64 fit",
        "the one finding that survived the adversarial evaluation. A FAIL here "
        "means the cap was an artefact of the anchor and not a property of the tile.",
        [f"  BM={bm:3d}  cap/ridge in [{min(v):.3f}, {max(v):.3f}] over {len(v)} fits"
         for bm, v in sorted(by_bm.items())])


def gate_c4_pooled_alpha(fits) -> Gate:
    """SURFACE.txt's `0 of 12 fits within 0.05 of 0.558`, re-scored."""
    hits = [f for f in fits if f.contains_pooled]
    return Gate(
        "C4", "CLAIM", f"no anchor bracket admits the pooled alpha {POOLED_ALPHA}",
        PASS if not hits else FAIL,
        f"{len(hits)} of {len(fits)} brackets contain {POOLED_ALPHA}",
        "0 brackets",
        f"SURFACE.txt's line \"alpha = {POOLED_ALPHA} ...: 0 of 12 fits within "
        "0.05\". That count is a statement about POINT estimates whose anchor is "
        "unidentified; the listed fits are consistent with the pooled value once "
        "the anchor is bracketed, so the sentence must be withdrawn or requalified.",
        [f"  {f.arm[:26]:26s} {f.model[:14]:14s} G={f.group_m:2d} BM={f.block_m:3d}  "
         f"bracket [{f.alpha_lo:.3f}, {f.alpha_hi:.3f}] contains {POOLED_ALPHA}"
         for f in hits])


def gate_c5_correction_size(fits) -> Gate:
    """How big the correction is, as a gate so it cannot be read as a footnote."""
    if not fits:
        return Gate("C5", "CLAIM", "the anchor correction is small", FAIL, "no fits",
                    "median |published - bracket midpoint| <= 0.05", "the whole table")
    shifts = [abs(f.alpha_published_corrected - 0.5 * (f.alpha_lo + f.alpha_hi))
              for f in fits]
    widths = [f.alpha_hi - f.alpha_lo for f in fits]
    med = statistics.median(shifts)
    ok = med <= 0.05
    return Gate(
        "C5", "CLAIM", "re-anchoring moves the published alpha by less than 0.05",
        PASS if ok else FAIL,
        f"median shift {med:.3f} (max {max(shifts):.3f}); median bracket width "
        f"{statistics.median(widths):.3f} (max {max(widths):.3f})",
        "median shift <= 0.05",
        "any published alpha quoted to three decimals. The anchor moves the "
        "number by more than the precision it is quoted at, so it may only be "
        "quoted as an interval.")


# --------------------------------------------------------------------------
# Predictions, registered with numbers before anything is measured.
# --------------------------------------------------------------------------

@dataclass
class Prediction:
    number: str
    statement: str
    basis: str
    invalidates: str

    def render(self) -> list[str]:
        return [f"  {self.number}  {self.statement}",
                f"      BASIS       {self.basis}",
                f"      A FAIL      {self.invalidates}"]


PREDICTIONS = [
    Prediction(
        "P1", "the anchor t(1) at a fixed BLOCK_M varies by <= 4.0% across "
              "GROUP_SIZE_M in {1, 8, 16}, and by <= 8.0% including 64",
        "12 committed (arm, model, BLOCK_M) groups give 0.36% to 3.10% across "
        "G <= 16 and up to 6.74% including G=64",
        "L_hi = t(1) as a condition-free upper bound. The bracket's top end "
        "would then have to be taken over swizzle conditions, widening it"),
    Prediction(
        "P2", "the anchor rate (W + Act1) / t(1) at BLOCK_M=32 lands between "
              "64% and 78% of the card's pin rate",
        "21 committed BLOCK_M=32 fits give 1370-1477 GB/s on the A100 "
        "(67.2-72.4% of 2039) and 3344-3522 on the H200 (68.0-71.6% of 4916.7)",
        "the carry-across. A new anchor at a different rate is not the same "
        "physical event as the one the committed arms measured, and the "
        "re-scoring of those arms could not be quoted from this run"),
    Prediction(
        "P3", "dropping the anchor tread from the branch fit moves the slope B "
              "by <= 1.5%",
        "40 committed fits move by -0.03% to +0.79% when n=1 is dropped",
        "the independence of the two ends. B and t(1) would then share the "
        "anchor and B / t(1) would be partly a restatement of it, not a bound"),
    Prediction(
        "P4", "the anchor rate never exceeds the card's measured ceiling: "
              "BW_1 / BW_ceiling <= 1.0 on every cell",
        "the tightest committed case is 78.6% (1477 GB/s against the A100's "
        "1879.1 write pattern)",
        "the bracket outright on that card. An anchor above the ceiling means "
        "the ceiling is not one and the interval inverts"),
]

PREDICTION_PROVENANCE = (
    "P1-P4 are OUT OF SAMPLE for --measure: they are stated from the committed "
    "reports and predict a run that has not happened. They are NOT out of sample "
    "for --rescore, which re-reads the same reports; there the honest object is "
    "the gate thresholds, which are constants at the top of this file and are "
    "printed below BEFORE the table they score. Said plainly so nobody reads a "
    "reproduction as a prediction.")


# --------------------------------------------------------------------------
# Output plumbing.
# --------------------------------------------------------------------------

def git_ignored(path: Path) -> bool | None:
    """Would git drop this file. None when git cannot answer.

    Checked and PRINTED rather than assumed. `results/*` is ignored with only
    `!results/published/` excepted, and this repo has already lost every figure
    of ten published arms to an unanchored `plots/` rule that matched at any
    depth: publish logged "included N figures" and git added none.
    """
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)],
                           cwd=REPO, capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return True                       # git will drop it
    if r.returncode == 1:
        return False                      # git will keep it
    # ANY OTHER CODE IS "GIT CANNOT ANSWER", AND IT IS NOT A NO. `check-ignore`
    # exits 128 for a path outside the work tree, which is every path on a pod:
    # --measure writes to $MOE_RESULTS_DIR, a network volume at /workspace. The
    # old `returncode == 0` collapsed 128 into False and printed a pod path as a
    # "tracked path", which is the opposite of true -- git has no opinion about
    # it at all and nothing there enters the repo without publish_results.sh.
    return None


#: The re-scoring lands as two FILES beside `CALIBRATION_PROVENANCE.md` and
#: `NOISE_FLOOR.json`, not as a dated directory. `results/published/*/` is the
#: namespace for measurement ARMS: `moe/bench/published.py` and
#: `tests/test_calibration_provenance.py` both enumerate arms with `is_dir()`,
#: so a `2026-09-01-anchor-rescore/` directory would be counted as an eleventh
#: arm and asked for a calibration provenance it does not have. It is a
#: cross-arm artefact, and the stable names also make a re-run idempotent
#: instead of leaving one directory per day.
RESCORE_STEM = "ANCHOR_RESCORE"


def report_output_paths(out_dir: Path, lines: list[str], payload: dict) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{RESCORE_STEM}.txt"
    js = out_dir / f"{RESCORE_STEM}.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    notes = []
    for p in (txt, js):
        ig = git_ignored(p)
        state = "IGNORED BY GIT -- this file will not be committed" if ig else (
            "tracked path" if ig is False else "git could not answer")
        notes.append(f"  {p}  [{state}]")
    return notes


# --------------------------------------------------------------------------
# The rescore mode.
# --------------------------------------------------------------------------

def render_fit_table(fits: list[ScoredFit]) -> list[str]:
    out = [
        "",
        "PER-FIT BRACKET. alpha_pub is the report's own alpha-corrected column. "
        "The bracket is",
        "[(B*BW_1 - Act1)/W, (B*BW_ceil - Act1)/W] with B refitted WITHOUT the "
        "anchor tread.",
        "elev is how far the measured n=1 tread stands above the fitted branch, "
        "as a fraction of itself.",
        "",
        f"  {'card':6s} {'model':10s} {'G':>3s} {'BN':>4s} {'st':>3s} {'BM':>4s} "
        f"{'a_pub':>6s} {'lo':>6s} {'hi':>6s} {'hi@pin':>7s} {'elev':>7s} "
        f"{'BW_pub':>7s} {'BW_1':>6s} {'in?':>4s} {'cap/ridge':>9s}",
        "  " + "-" * 108,
    ]
    for f in sorted(fits, key=lambda f: (f.card, f.arm, f.model, f.group_m, f.block_m)):
        flag = "in" if f.contains_published_corrected else "OUT"
        marks = []
        if not f.physical_vs_pin:
            marks.append("!bus")
        if f.contains_pooled:
            marks.append(f"~{POOLED_ALPHA}")
        if f.clipped:
            marks.append("clip")
        out.append(
            f"  {f.card[7:13]:6s} {f.model[:10]:10s} {f.group_m:3d} {f.block_n:4d} "
            f"{f.num_stages:3d} {f.block_m:4d} {f.alpha_published_corrected:6.3f} "
            f"{f.alpha_lo:6.3f} {f.alpha_hi:6.3f} {f.alpha_hi_pin:7.3f} "
            f"{f.anchor_elevation:+7.1%} {f.bw_published_gbps:7.0f} "
            f"{f.bw_anchor_gbps:6.0f} {flag:>4s} {f.cap_over_ridge_at_lo:9.3f}"
            + ("  " + " ".join(marks) if marks else ""))
    return out


def render_residuals(fits: list[ScoredFit], max_treads: int = 8) -> list[str]:
    """Where the branch misses, per tread.

    Printed because the shape decides the mechanism question, and because a
    reader who wants to disagree with the bracket should be able to see the same
    thing this file saw rather than a summary of it.
    """
    out = ["",
           "RESIDUAL PROFILE against the published branch, per tread, as a "
           "fraction of the measured time.",
           "If the anomaly were 'a swizzle group of width G spans G experts', "
           "it would persist while n < G.",
           ""]
    for f in sorted(fits, key=lambda f: (f.card, f.model, f.block_m, f.group_m)):
        prof = " ".join(f"{n}:{r:+.1%}" for n, r in f.residuals[:max_treads])
        out.append(f"  {f.card[7:13]:6s} {f.model[:10]:10s} G={f.group_m:2d} "
                   f"BM={f.block_m:3d}  {prof}")
    return out


def render_summary(fits: list[ScoredFit]) -> list[str]:
    """The four numbers a reader needs before the gates.

    Split by CARD, because the defect is not the same size on both: the A100
    fits sit at 67-72% of pin at the anchor and their fitted levels run past the
    bus, while the H200's fitted levels are physically possible and merely
    unidentified. One pooled number would hide that.
    """
    out = ["", "SUMMARY"]
    by_card: dict[str, list[ScoredFit]] = {}
    for f in fits:
        by_card.setdefault(f.card, []).append(f)
    for card, group in sorted(by_card.items()):
        elev = [f.anchor_elevation for f in group]
        widths = [f.alpha_hi - f.alpha_lo for f in group]
        shifts = [abs(f.alpha_published_corrected - 0.5 * (f.alpha_lo + f.alpha_hi))
                  for f in group]
        up = sum(1 for e in elev if e > 0)
        mean_anchor = statistics.fmean(f.bw_anchor_gbps for f in group)
        ceiling = group[0].bw_ceiling_gbps
        out += [
            f"  {card}  ({len(group)} fits)",
            f"    anchor above the fitted branch   {up} of {len(group)}, "
            f"{min(elev):+.1%} to {max(elev):+.1%} of the measured tread",
            f"    bracket width                    {min(widths):.3f} to "
            f"{max(widths):.3f} in alpha (median {statistics.median(widths):.3f})",
            f"    shift from the published point   median {statistics.median(shifts):.3f}, "
            f"max {max(shifts):.3f}",
            f"    published anchor above the bus   "
            f"{sum(1 for f in group if not f.physical_vs_pin)} of {len(group)}",
            f"    what sets the width              the anchor reaches "
            f"{mean_anchor:.0f} GB/s and the ceiling is {ceiling:.0f}, a "
            f"{ceiling / mean_anchor - 1:.0%} window in BW and so in alpha",
        ]
    out += ["",
            "  READ THE WIDTH AS THE ANSWER, not as a failure of the method. It is the",
            "  gap between what the kernel achieved at n=1 and what the card can do,",
            "  and only a DRAM counter closes it. ncu returns ERR_NVGPUCTRPERM on a",
            "  rented pod and the image's nsys cannot convert its own capture, so no",
            "  counter-free method identifies the branch's bandwidth and this file does",
            "  not pretend one does.",
            "",
            "  B IS IDENTIFIED AND L IS NOT, which is the whole finding in one line:",
            f"    dropping the anchor tread moves the slope by at most "
            f"{max(abs(f.slope_refit_no_anchor / f.slope_refit_full - 1.0) for f in fits):.2%},",
            "    while the level it is divided by is unidentified over the width above."]
    return out


def render_withdrawals(fits: list[ScoredFit], gates: list[Gate]) -> list[str]:
    """What this run says can no longer be published as written."""
    out = ["", "=" * 78, "WITHDRAWALS -- published cells and sentences this run says cannot stand",
           "=" * 78]
    failed = {g.number for g in gates if g.verdict == FAIL}
    n = 0
    if "C1" in failed:
        bad = [f for f in fits if not f.physical_vs_pin]
        n += 1
        out += ["",
                f"W{n}. {len(bad)} published alpha values are IMPOSSIBLE, not uncertain.",
                "    Their fitted level moves the weight set faster than the memory bus:"]
        out += [f"      {f.arm}  {f.model} G={f.group_m} BM={f.block_m}: "
                f"alpha {f.alpha_published:.3f} needs {f.bw_published_gbps:.0f} GB/s "
                f"on a {f.bw_pin_gbps:.0f} GB/s bus" for f in bad]
    if "C2" in failed:
        out_of = [f for f in fits if not f.contains_published_corrected]
        n += 1
        out += ["",
                f"W{n}. {len(out_of)} of {len(fits)} published alpha-corrected values "
                "fall outside their own anchor bracket and may not be quoted as "
                "point estimates."]
    if "C4" in failed:
        hits = [f for f in fits if f.contains_pooled]
        n += 1
        out += ["",
                f"W{n}. SURFACE.txt's \"alpha = {POOLED_ALPHA} ...: 0 of 12 fits within "
                f"0.05\" must be withdrawn or requalified. {len(hits)} bracket(s) "
                f"contain {POOLED_ALPHA}; the 0-of-12 count is a property of an "
                "unidentified anchor, not of the data."]
    if "C5" in failed:
        n += 1
        out += ["",
                f"W{n}. No alpha in this study may be quoted as a number. Every "
                "reported alpha carries an anchor interval wider than the "
                "precision it is printed at, and this file's `anchor.json` "
                "carries the interval per fit."]
    if "C3" not in failed:
        out += ["",
                "NOT WITHDRAWN. The BLOCK_M <= 64 arithmetic-intensity cap survives. "
                "It is scored at the bracket's most generous alpha, which is the "
                "best case for a small tile reaching its roof, and it still caps "
                "below the ridge on every fit."]
    if n == 0:
        out += ["", "None. Every claim gate passed."]
    return out


def run_rescore(args) -> int:
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("MEMORY BRANCH ANCHOR -- alpha's denominator, measured instead of extrapolated")
    say("=" * 78)
    say(f"published root : {args.published}")
    say("mode           : --rescore (no GPU, no measurement, seconds)")
    say("")
    say("THE MODEL, so every column below can be checked by hand:")
    say("    bytes(n) = W (1 + alpha (n-1)) + Act1 n      t(n) = D + bytes(n)/BW, D >= 0")
    say("    B = (W alpha + Act1) / BW      =>      alpha = (B BW - Act1) / W")
    say("    BW >= (W + Act1) / t(1)   because at n=1 every weight byte is read once")
    say("    BW <= the card's measured streaming ceiling")
    say("")
    say("REGISTERED PREDICTIONS")
    for p in PREDICTIONS:
        for line in p.render():
            say(line)
    say("")
    say(f"  PROVENANCE  {PREDICTION_PROVENANCE}")
    say("")
    say("REGISTERED GATE THRESHOLDS, constants in this file, printed before the table")
    say(f"  V1 non-vacuity            >= {MIN_SCORED_FITS} scored fits")
    say(f"  V2 slope reproduction     <= {SLOPE_REPRODUCTION_REL:.0e} relative")
    say("  V3 anchor present         every scored fit has a measured n=1 tread")
    say(f"  V4 poisoned reference     the level check fires on >= {args.expect_poisoned} arm(s)")
    say("  V5 bracket order          anchor rate <= measured ceiling on every fit")
    say(f"  V6 ASSUMPTION A           an anchor below its branch is within "
        f"{ANCHOR_BELOW_BRANCH_SPREADS:.0f} timing spreads")
    say("  C1 physicality            0 published alphas above the pin rate")
    say("  C2 containment            0 published alphas outside their bracket")
    say("  C3 tile cap               cap/ridge < 1.000 for every BLOCK_M <= 64 fit")
    say(f"  C4 pooled alpha           0 brackets contain {POOLED_ALPHA}")
    say("  C5 correction size        median |published - bracket midpoint| <= 0.05")
    say("")

    if args.dry_run:
        reports = sorted(args.published.glob("*/*.report.json"))
        say(f"DRY RUN. {len(reports)} report(s) would be scored, at no GPU cost:")
        for r in reports:
            say(f"  {r.relative_to(REPO)}")
        say("")
        say("Nothing was measured and nothing was written.")
        return 3

    fits, refusals, cals = scan_published(args.published)
    say("CARD CALIBRATIONS USED (the bracket's upper end, and the ridge for C3)")
    for slug in sorted(cals):
        say(f"  {cals[slug].describe()}")
    say("")
    say("  RIDGE PROVENANCE, because C3 is scored against it. Every one of these")
    say("  reports carries ridge=160.3 and ridge_band=[160.3, 176.2], because the")
    say("  sweep's --ridge defaults to RIDGE_BAND[0] and the cross-card driver")
    say("  never passes it. 160.3 is a stale H200 band and belongs to NEITHER")
    say("  card. C3 uses each card's OWN contemporaneous calibration instead:")
    for slug in sorted(cals):
        c = cals[slug]
        peak = c.dense_tflops.get("bf16", max(c.dense_tflops.values()))
        say(f"    {slug:24s} ridge {c.ridge:6.2f} FLOP/byte  "
            f"(dense {peak:.2f} TFLOP/s over its own triad bandwidth, "
            f"measured {c.checked_on})")
    say("  The default itself belongs to another workflow's lane and is reported,")
    say("  not edited, here.")
    say("")
    if refusals:
        say(f"REFUSED, {len(refusals)} fit(s) -- listed, never defaulted to a number")
        for r in refusals:
            say(f"  {r.arm[:30]:30s} {r.model[:14]:14s} G={r.group_m:2d} BN={r.block_n:3d} "
                f"BM={r.block_m:3d}")
            say(f"      {r.reason}")
        say("")
    if not fits:
        say("NOTHING SCORED. No report carried an identifiable ladder with an anchor.")
        return 3

    for line in render_fit_table(fits):
        say(line)
    for line in render_summary(fits):
        say(line)
    if args.residuals:
        for line in render_residuals(fits):
            say(line)

    gates = [
        gate_v1_non_vacuity(fits, refusals),
        gate_v2_slope_reproduction(fits),
        gate_v3_anchor_present(fits, refusals),
        gate_v4_poisoned_reference(refusals, args.expect_poisoned),
        gate_v5_bracket_order(fits),
        gate_v6_assumption_a(fits),
        gate_c1_physicality(fits),
        gate_c2_containment(fits),
        gate_c3_tile_cap(fits),
        gate_c4_pooled_alpha(fits),
        gate_c5_correction_size(fits),
    ]
    say("")
    say("=" * 78)
    say("GATES")
    say("=" * 78)
    for g in gates:
        for line in g.render():
            say(line)
    for line in render_withdrawals(fits, gates):
        say(line)

    payload = {
        "generated_by": "scripts/memory_branch_anchor.py --rescore",
        "pooled_alpha": POOLED_ALPHA,
        "calibrations": {s: asdict(c) for s, c in cals.items()},
        "gates": [asdict(g) for g in gates],
        "refusals": [asdict(r) for r in refusals],
        "fits": [asdict(f) for f in fits],
    }
    say("")
    say("WROTE")
    for note in report_output_paths(args.out_dir, lines, payload):
        print(note)

    validity_failed = any(g.verdict == FAIL and g.kind == "VALIDITY" for g in gates)
    claim_failed = any(g.verdict == FAIL and g.kind == "CLAIM" for g in gates)
    if validity_failed:
        return 2
    return 1 if claim_failed else 0


# --------------------------------------------------------------------------
# The measure mode. Its job is not to produce a new alpha; it is to make both
# ends of the bracket come from one process at one clock state, and to answer
# the "different reuse condition" objection by measurement.
# --------------------------------------------------------------------------

#: The card slug a plan carries when NO device is attached, which is every
#: --dry-run on a laptop. It is a visible placeholder rather than a blank on
#: purpose: a dry run must not print the same directory a real run would use, or
#: the operator checks a path that the pod will never write to.
UNKNOWN_CARD_SLUG = "nocard"


def detect_card() -> tuple[str, str] | None:
    """`(device name, slug)` for the ATTACHED device, or None if there is none.

    Kept separate from `run_measure` so the run id can be built BEFORE the plan
    is rendered. The card has to be in the id, and the id is printed as part of
    the plan, so the device has to be resolved before anything else happens.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    return name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class MeasurePlan:
    #: WHICH CARD. Not decorative and not derivable from the rest of the plan.
    #: The whole point of this file is a bracket whose upper end is a per-card
    #: ceiling, so two cards running one plan are two different measurements --
    #: and both write to `$MOE_RESULTS_DIR`, which on RunPod is a network volume
    #: shared between pods. Without this field the A100 and the H200 derive the
    #: same id, the second run finds every cell present, resumes, skips all of
    #: them, and publishes the first card's timings under the second card's
    #: calibration. That is not hypothetical: the study's two cross-card arms are
    #: committed under IDENTICAL filenames for exactly this reason.
    card: str
    model: str
    dtype: str
    block_sizes: tuple[int, ...]
    group_sizes: tuple[int, ...]
    slope_tiles: tuple[int, ...]
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    seed: int
    iters: int
    warmup: int
    stream_reps: int

    @property
    def cells(self) -> list[tuple[int, int, int]]:
        """`(BLOCK_M, GROUP_SIZE_M, tiles)` -- the anchor first, then the branch."""
        out = []
        for bm in self.block_sizes:
            for g in self.group_sizes:
                for n in (1, *self.slope_tiles):
                    out.append((bm, g, n))
        return out

    def run_id(self) -> str:
        """EVERY swept knob is in the key AND in the visible name.

        The sweep this study already ships lost a whole G=16 arm to an id that
        omitted GROUP_SIZE_M: the run derived the G=1 id, resumed into its
        directory, skipped every cell as already done, and printed G=1's timings
        under a G=16 heading. Nothing looked wrong. The knobs are in the name so
        two runs are also distinguishable in `ls`.

        THE CARD IS ONE OF THE KNOBS. It is not something this script sweeps, it
        is something the OPERATOR sweeps by moving to another pod, and the
        results root is a network volume that outlives the pod. The failure is
        the same shape as the GROUP_SIZE_M one and is worse, because a bracket's
        upper end is a per-card ceiling: the second card would publish the first
        card's timings against its own ridge and its own pin rate.
        """
        key = json.dumps(asdict(self), sort_keys=True)
        groups = "_".join(str(g) for g in self.group_sizes)
        tiles = "_".join(str(t) for t in self.block_sizes)
        return (f"{self.card}-{self.model}-{self.dtype}-bm{tiles}-g{groups}"
                f"-n{self.block_n}-k{self.block_k}-w{self.num_warps}"
                f"-s{self.num_stages}"
                f"-{hashlib.sha1(key.encode()).hexdigest()[:6]}")

    def estimated_seconds(self, per_cell_s: float = 6.0, compile_s: float = 12.0) -> float:
        """Timed work plus one Triton compile per distinct (BLOCK_M, GROUP_SIZE_M).

        Deliberately crude and stated as such. Its job is to stop a run being
        started without a wall-clock number attached, not to be accurate.
        """
        settings = len(self.block_sizes) * len(self.group_sizes)
        return len(self.cells) * per_cell_s + settings * compile_s + self.stream_reps * 2.0


def render_plan(plan: MeasurePlan, out_dir: Path) -> list[str]:
    cells = plan.cells
    lines = [
        "MEASURE PLAN",
        f"  model / dtype        {plan.model} / {plan.dtype}",
        f"  BLOCK_SIZE_M         {', '.join(str(b) for b in plan.block_sizes)}",
        f"  GROUP_SIZE_M         {', '.join(str(g) for g in plan.group_sizes)}",
        "  anchor tread         n = 1 (one M-tile per expert: every weight byte "
        "read exactly once)",
        f"  branch treads        n = {', '.join(str(t) for t in plan.slope_tiles)}",
        f"  pinned               BLOCK_SIZE_N={plan.block_n} BLOCK_SIZE_K={plan.block_k} "
        f"num_warps={plan.num_warps} num_stages={plan.num_stages}",
        f"  timing               {plan.warmup} warmup, {plan.iters} timed, seed {plan.seed}",
        f"  stream check         {plan.stream_reps} repeats over the real w1/w2 buffers",
        "",
        f"  cells                {len(cells)} "
        f"({len(plan.block_sizes)} BLOCK_M x {len(plan.group_sizes)} G x "
        f"{1 + len(plan.slope_tiles)} treads)",
        f"  estimated wall time  {plan.estimated_seconds() / 60.0:.1f} min "
        f"(crude: {len(cells)} cells at 6 s plus one compile per setting)",
        f"  run id               {plan.run_id()}",
        f"  output               {out_dir}",
        "",
        "  WHAT EACH ARM BUYS, and why the committed data cannot supply it:",
        "    stream  the read ceiling on the ACTUAL weight buffers, in this "
        "process at this",
        "            clock state. The committed calibration used a synthetic "
        "8 GiB buffer in a",
        "            different process; it sets the ceiling and this only checks "
        "it is one.",
        "    anchor  t(1) at every GROUP_SIZE_M with everything else pinned. This "
        "answers the",
        "            objection that the anchor is measured at a different L2-reuse "
        "condition",
        "            from the branch it anchors -- by measurement rather than by "
        "argument.",
        "    branch  treads from n=2 up, so the slope never sees the anchor and "
        "the two ends",
        "            of the bracket are independent.",
        "",
        "  WHAT IT STILL CANNOT DO. It does not identify BW on the branch. That "
        "needs a DRAM",
        "  counter; ncu returns ERR_NVGPUCTRPERM on rented pods and the image's "
        "nsys cannot",
        "  convert its own capture. The deliverable is the interval, not a point.",
    ]
    return lines


def find_override_config():
    """vLLM's own tuning hook, probed rather than assumed.

    Duplicated from `scripts/block_m_crossing_sweep.py` on purpose: that file is
    under concurrent edit by another workflow, and an anchor measurement that
    changes because someone refactored a helper is not an anchor. Fifteen lines
    is a cheap price for an arm that cannot be moved out from under it.
    """
    import importlib
    for name in ("vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.fused_moe",
                 "vllm.model_executor.layers.fused_moe.config"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, name
    raise SystemExit("could not find vLLM's override_config; check the installed version")


#: P1's threshold. The anchor may not depend on the swizzle by more than this,
#: or `t(1)` is a condition-specific number and not a condition-free bound.
#: Split because G=64 on the A100 already sits at 6.7% in the committed data and
#: pretending one threshold covers both would either excuse a real drift at
#: G<=16 or fail a known-good G=64.
ANCHOR_INVARIANCE_SMALL_G = 0.040
ANCHOR_INVARIANCE_ALL_G = 0.080

#: P2's band, as a fraction of the card's PIN rate. Wide on purpose: it is a
#: check that the new anchor is the same physical event the committed ladders
#: measured, not a precision claim about the kernel.
ANCHOR_RATE_BAND = (0.64, 0.78)

#: P3's threshold. Above this the slope is not independent of the anchor and
#: `B / t(1)` stops being a bound on alpha and starts being a restatement of it.
SLOPE_INDEPENDENCE_REL = 0.015

#: How many branch treads a --measure grid needs before P3 can be scored against
#: that threshold. It was taken from ladders of 16 and 33 treads. On a planted
#: ladder the slope moves 3.2% when the anchor is dropped at 8 treads, 1.4% at 12
#: and 0.8% at 16, so a short grid would FAIL P3 for a reason that is about the
#: grid and not about the kernel -- and a threshold that fails for the wrong
#: reason teaches a reader to ignore it.
MIN_BRANCH_TREADS = 12


@dataclass
class MeasuredFit:
    """One (BLOCK_M, GROUP_SIZE_M) cell of the GPU arm, bracketed.

    Deliberately separate from `ScoredFit`: this one has no published alpha to
    compare against, and reusing the same record would have forced a placeholder
    into the comparison columns. A placeholder in a comparison column is how a
    study reports a correction it never computed.
    """

    block_m: int
    group_m: int
    treads: int
    anchor_ms: float
    slope_with_anchor: float
    slope_no_anchor: float
    slope_shift: float
    bw_anchor_gbps: float
    anchor_rate_of_pin: float
    alpha_lo: float
    alpha_hi: float
    alpha_hi_pin: float
    cap_over_ridge_at_lo: float
    residuals: list[tuple[int, float]] = field(default_factory=list)


def fits_from_cells(cells, cfg, model_dtype: str, block_n: int,
                    cal: Calibration) -> tuple[list[MeasuredFit], list[Refusal]]:
    """Bracket every (BLOCK_M, GROUP_SIZE_M) cell of a measured run.

    Pure: takes rows, returns records. No device, no globals, so the GPU arm's
    entire verdict path is exercised by the laptop test suite instead of only
    on a rented pod, which is where this study has previously discovered that a
    scoring path did not work.
    """
    fits: list[MeasuredFit] = []
    refusals: list[Refusal] = []
    groups: dict[tuple[int, int], dict[int, float]] = {}
    for row in cells:
        if row.get("status") != "ok" or float(row.get("ms_p50", 0.0)) <= 0:
            continue
        groups.setdefault((int(row["block_m"]), int(row["group_m"])), {})[
            int(row["tiles"])] = float(row["ms_p50"])
    for (bm, g), by_n in sorted(groups.items()):
        if 1 not in by_n:
            refusals.append(Refusal("measured", cfg.name, g, block_n, bm,
                                    "the n=1 anchor cell did not measure; there is "
                                    "nothing to anchor this bracket on"))
            continue
        branch = sorted((n, ms) for n, ms in by_n.items() if n >= 2)
        if len(branch) < 3:
            refusals.append(Refusal("measured", cfg.name, g, block_n, bm,
                                    f"{len(branch)} branch tread(s) at n>=2; a slope "
                                    "independent of the anchor needs at least 3"))
            continue
        xs = [float(n) for n, _ in branch]
        ys = [ms for _, ms in branch]
        a_free, b_free = ols(xs, ys)
        allpts = sorted(by_n.items())
        _, b_all = ols([float(n) for n, _ in allpts], [ms for _, ms in allpts])
        w_bytes, act1 = anchor_bytes(cfg, model_dtype, bm)
        br = bracket_alpha(b_free, b_free, by_n[1], bm, w_bytes, act1,
                           cal.ceiling_gbps, cal.pin_gbps)
        fits.append(MeasuredFit(
            block_m=bm, group_m=g, treads=len(by_n), anchor_ms=by_n[1],
            slope_with_anchor=b_all, slope_no_anchor=b_free,
            slope_shift=abs(b_free / b_all - 1.0),
            bw_anchor_gbps=br.bw_anchor_gbps,
            anchor_rate_of_pin=br.bw_anchor_gbps / cal.pin_gbps,
            alpha_lo=br.lo, alpha_hi=br.hi, alpha_hi_pin=br.hi_pin,
            cap_over_ridge_at_lo=br.ai_cap(dtype_bytes(model_dtype), br.lo) / cal.ridge,
            residuals=residual_profile(allpts, a_free, b_free)))
    return fits, refusals


def gate_m0_stream(stream: dict | None, cal: Calibration) -> Gate:
    """Is the committed ceiling actually above what these buffers achieve."""
    if not stream:
        return Gate("M0", "VALIDITY", "the committed ceiling is above the measured "
                    "read rate on the real weight buffers", FAIL,
                    "the stream check did not run", "measured <= ceiling",
                    "the bracket's upper end, which would then rest on a ceiling "
                    "this run never checked")
    ratio = stream["gbps"] / cal.ceiling_gbps
    return Gate(
        "M0", "VALIDITY", "the committed ceiling is above the measured read rate "
        "on the real weight buffers",
        PASS if ratio <= 1.0 else FAIL,
        f"{stream['gbps']:.1f} GB/s against a ceiling of {cal.ceiling_gbps:.1f} "
        f"({ratio:.1%})", "<= 100%",
        "the bracket's upper end. A ceiling below the data is not a ceiling, and "
        "every alpha_hi computed from it is too low")


def gate_m1_anchor_invariance(fits) -> Gate:
    """P1, scored. Does the anchor depend on the swizzle it is supposed to bound."""
    by_bm: dict[int, dict[int, float]] = {}
    for f in fits:
        by_bm.setdefault(f.block_m, {})[f.group_m] = f.anchor_ms
    lines, worst_small, worst_all = [], 0.0, 0.0
    for bm, by_g in sorted(by_bm.items()):
        small = [ms for g, ms in by_g.items() if g <= 16]
        allv = list(by_g.values())
        s = max(small) / min(small) - 1.0 if len(small) > 1 else 0.0
        a = max(allv) / min(allv) - 1.0 if len(allv) > 1 else 0.0
        worst_small, worst_all = max(worst_small, s), max(worst_all, a)
        lines.append(f"  BM={bm:3d}  G={sorted(by_g)}  spread G<=16 {s:.2%}, "
                     f"all G {a:.2%}")
    ok = (worst_small <= ANCHOR_INVARIANCE_SMALL_G
          and worst_all <= ANCHOR_INVARIANCE_ALL_G)
    return Gate(
        "M1", "CLAIM", "P1: the anchor t(1) does not depend on the swizzle",
        PASS if ok else FAIL,
        f"worst spread {worst_small:.2%} across G<=16, {worst_all:.2%} across all G",
        f"<= {ANCHOR_INVARIANCE_SMALL_G:.1%} and <= {ANCHOR_INVARIANCE_ALL_G:.1%}",
        "t(1) as a condition-free upper bound on L. The bracket's top end would "
        "have to be taken over swizzle conditions instead, widening every interval",
        lines)


def gate_m2_anchor_rate(fits, cal: Calibration, block_m: int = 32) -> Gate:
    """P2, scored. Is this the same physical event the committed arms measured."""
    sel = [f for f in fits if f.block_m == block_m]
    if not sel:
        return Gate("M2", "CLAIM", f"P2: the BLOCK_M={block_m} anchor rate is in band",
                    FAIL, f"no BLOCK_M={block_m} cells measured",
                    f"{ANCHOR_RATE_BAND[0]:.0%}-{ANCHOR_RATE_BAND[1]:.0%} of pin",
                    "the carry-across to the committed arms, which is scored at "
                    f"BLOCK_M={block_m}")
    lo = min(f.anchor_rate_of_pin for f in sel)
    hi = max(f.anchor_rate_of_pin for f in sel)
    ok = ANCHOR_RATE_BAND[0] <= lo and hi <= ANCHOR_RATE_BAND[1]
    return Gate(
        "M2", "CLAIM", f"P2: the BLOCK_M={block_m} anchor rate is in band",
        PASS if ok else FAIL,
        f"{lo:.1%}-{hi:.1%} of the {cal.pin_gbps:.0f} GB/s pin rate over "
        f"{len(sel)} cells",
        f"{ANCHOR_RATE_BAND[0]:.0%}-{ANCHOR_RATE_BAND[1]:.0%} of pin",
        "the carry-across. An anchor at a different rate is not the event the "
        "committed ladders measured, and this run's brackets could not be quoted "
        "for them")


def gate_m3_slope_independence(fits) -> Gate:
    """P3, scored. Are the two ends of the bracket independent."""
    if not fits:
        return Gate("M3", "CLAIM", "P3: the slope does not depend on the anchor",
                    FAIL, "no cells", f"<= {SLOPE_INDEPENDENCE_REL:.1%}",
                    "the independence of the bracket's two ends")
    worst = max(fits, key=lambda f: f.slope_shift)
    ok = worst.slope_shift <= SLOPE_INDEPENDENCE_REL
    return Gate(
        "M3", "CLAIM", "P3: the slope does not depend on the anchor",
        PASS if ok else FAIL,
        f"worst {worst.slope_shift:.2%} at BM={worst.block_m} G={worst.group_m} "
        f"over {worst.treads} treads",
        f"<= {SLOPE_INDEPENDENCE_REL:.1%} (calibrated on 16- and 33-tread ladders)",
        "B / t(1) as a BOUND. If the slope moves when the anchor leaves the fit, "
        "the low end of the bracket is partly a restatement of the anchor")


def gate_m4_bracket_order(fits, cal: Calibration) -> Gate:
    """P4, scored."""
    if not fits:
        return Gate("M4", "VALIDITY", "P4: the bracket is ordered", FAIL, "no cells",
                    "anchor rate <= ceiling", "every interval below")
    tight = max(fits, key=lambda f: f.bw_anchor_gbps)
    ratio = tight.bw_anchor_gbps / cal.ceiling_gbps
    ok = ratio <= 1.0 and all(f.alpha_lo <= f.alpha_hi + 1e-12 for f in fits)
    return Gate(
        "M4", "VALIDITY", "P4: the anchor rate never exceeds the measured ceiling",
        PASS if ok else FAIL,
        f"tightest {ratio:.1%} at BM={tight.block_m} G={tight.group_m}", "<= 100%",
        "the bracket outright on this card: an anchor above the ceiling inverts "
        "the interval")


def gate_m5_completeness(cells, planned: int) -> Gate:
    """NON-VACUITY: the run has to have done the work it planned."""
    ok_cells = [c for c in cells if c.get("status") == "ok"]
    ok = len(ok_cells) == planned and planned > 0
    failed = [c for c in cells if c.get("status") != "ok"]
    return Gate(
        "M5", "VALIDITY", "every planned cell measured",
        PASS if ok else FAIL,
        f"{len(ok_cells)} of {planned} planned cells ok, {len(failed)} failed",
        "all planned cells ok",
        "every count below. A cell that failed silently is a bracket built on a "
        "different grid from the one the plan printed",
        [f"  BM={c['block_m']} G={c['group_m']} n={c['tiles']}: {c['detail']}"
         for c in failed[:10]])


def render_measured_table(fits: list[MeasuredFit]) -> list[str]:
    out = ["",
           "MEASURED BRACKETS. B is refitted on n>=2 only; the anchor is the "
           "measured n=1 cell.",
           "",
           f"  {'BM':>4s} {'G':>4s} {'t(1) ms':>9s} {'B ms':>8s} {'dB':>6s} "
           f"{'BW_1':>7s} {'%pin':>6s} {'alpha_lo':>8s} {'alpha_hi':>8s} "
           f"{'cap/ridge':>9s}",
           "  " + "-" * 82]
    for f in sorted(fits, key=lambda f: (f.block_m, f.group_m)):
        out.append(f"  {f.block_m:4d} {f.group_m:4d} {f.anchor_ms:9.4f} "
                   f"{f.slope_no_anchor:8.4f} {f.slope_shift:6.2%} "
                   f"{f.bw_anchor_gbps:7.0f} {f.anchor_rate_of_pin:6.1%} "
                   f"{f.alpha_lo:8.3f} {f.alpha_hi:8.3f} "
                   f"{f.cap_over_ridge_at_lo:9.3f}")
    return out


def score_measured(cells, cfg, dtype: str, block_n: int, cal: Calibration,
                   stream: dict | None, planned: int
                   ) -> tuple[list[MeasuredFit], list[Refusal], list[Gate], list[str]]:
    """The GPU arm's whole verdict path, with no device in it."""
    fits, refusals = fits_from_cells(cells, cfg, dtype, block_n, cal)
    gates = [
        gate_m5_completeness(cells, planned),
        gate_m0_stream(stream, cal),
        gate_m4_bracket_order(fits, cal),
        gate_m1_anchor_invariance(fits),
        gate_m2_anchor_rate(fits, cal),
        gate_m3_slope_independence(fits),
    ]
    lines = render_measured_table(fits)
    return fits, refusals, gates, lines


def time_call(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    """Median, min and stdev milliseconds over CUDA events.

    No L2 flush, matching the sweep this re-anchors: the anchor has to be the
    same physical event those ladders measured, and a flush the ladders did not
    do would make it a different one.
    """
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), min(out), statistics.pstdev(out)


def stream_check(weights, reps: int) -> dict:
    """Read the real weight buffers and report GB/s.

    A REDUCTION along the contiguous axis, which is what
    `moe/bench/calibrate.py` documents as the read pattern: one pass, every byte
    touched once, and the per-row outputs keep the global combine out of the
    number. This is a LOWER bound on the machine's read rate by construction,
    which is the direction that makes it a valid CHECK on the ceiling: if this
    lower bound exceeds the committed ceiling, the ceiling is wrong.

    No custom kernel. The probe kernel in `moe/bench/read_probe.py` would be a
    tighter instrument and it belongs to another workflow; this arm only needs
    to know whether the ceiling is above the floor.
    """
    total = 0
    views = []
    for t in (weights.w1, weights.w2):
        flat = t.reshape(-1)
        cols = 4096
        rows = flat.numel() // cols
        if rows == 0:
            raise ValueError("weight tensor smaller than one probe row")
        views.append(flat[:rows * cols].view(rows, cols))
        total += rows * cols * t.element_size()

    def once():
        for v in views:
            v.sum(dim=1)

    ms, mn, sd = time_call(once, warmup=3, iters=reps)
    return {"bytes": total, "ms_p50": ms, "ms_min": mn, "ms_stdev": sd,
            "gbps": total / (ms * 1e-3) / 1e9,
            "gbps_from_min": total / (mn * 1e-3) / 1e9}


def run_measure(args) -> int:
    """The GPU arm. Prints the plan and the cost first, whatever happens next."""
    # THE CARD IS RESOLVED BEFORE THE PLAN, because it is IN the plan and the
    # plan's id is the directory the run resumes into. Resolving it later would
    # print one path and write another.
    detected = detect_card()
    card = args.card or (detected[1] if detected else UNKNOWN_CARD_SLUG)
    if args.card and detected and args.card != detected[1]:
        # REFUSE rather than trust the flag. --card exists so a laptop dry run
        # can print the path the pod will really use; letting it override an
        # ATTACHED device would let one card write into another's directory,
        # which is the collision this field was added to close.
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected[0]!r} (slug {detected[1]!r}). --card may name a card "
              "that is absent, never contradict one that is present. "
              "Nothing measured.")
        return 3
    plan = MeasurePlan(
        card=card, model=args.model, dtype=args.dtype,
        block_sizes=tuple(int(v) for v in args.tiles.split(",")),
        group_sizes=tuple(int(v) for v in args.group_m.split(",")),
        slope_tiles=tuple(int(v) for v in args.slope_tiles.split(",")),
        block_n=args.block_n, block_k=args.block_k, num_warps=args.num_warps,
        num_stages=args.num_stages, seed=args.seed, iters=args.iters,
        warmup=args.warmup, stream_reps=args.stream_reps)
    out_dir = args.out_dir / plan.run_id()

    for line in render_plan(plan, out_dir):
        print(line)
    if card == UNKNOWN_CARD_SLUG:
        print()
        print("  NO DEVICE ATTACHED, so the run id above carries the placeholder "
              f"card {UNKNOWN_CARD_SLUG!r} and is NOT the id a pod will derive. "
              "Pass --card <slug> to print the pod's real path from here.")
    print()
    print("REGISTERED PREDICTIONS, before the device is touched")
    for p in PREDICTIONS:
        for line in p.render():
            print(line)
    print()
    print(f"  PROVENANCE  {PREDICTION_PROVENANCE}")
    print()
    ig = git_ignored(out_dir)
    print(f"  OUTPUT PATH  {out_dir}  "
          f"[{'ignored by git' if ig else 'tracked path' if ig is False else 'git silent'}]")
    print()

    if args.dry_run:
        print("DRY RUN. Nothing was measured, nothing was written, no GPU was used.")
        return 3

    try:
        import torch
    except ImportError:
        print("REFUSED: torch is not installed. Nothing measured.")
        return 3
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device. The anchor is a measured time and there is "
              "nothing here to measure it on. Nothing measured.")
        return 3

    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    gpu = torch.cuda.get_device_name(0)
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    if slug != plan.card:
        # The device moved between the plan being built and here. It cannot
        # happen in one process today, and if it ever does the run id is stale
        # and the resume below would be reading another card's cells.
        print(f"REFUSED: the plan was built for card {plan.card!r} and the "
              f"attached device is now {slug!r}. The run id, and so the resume "
              "directory, belongs to the first. Nothing measured.")
        return 3
    slugs = available_calibrations()
    match = calibration_slug_for(slug, slugs) or (slug if slug in slugs else None)
    if match is None:
        print(f"REFUSED: no measured calibration for {gpu!r} (slug {slug!r}). "
              f"Known: {', '.join(slugs) or 'none'}. The bracket's upper end must "
              "be a measured ceiling for THIS card. Nothing measured.")
        return 3
    cal = load_calibration(match)
    print(f"device: {gpu}")
    print(f"ceiling: {cal.describe()}")
    print()

    cfg = MODEL_CONFIGS[plan.model]
    override_config, where = find_override_config()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs
    print(f"override hook: {where}.override_config")

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.json"
    # RESUME. The run id is derived from every swept knob precisely so that
    # re-running the same command lands in the same directory and finishes the
    # work instead of repeating it. Without this the id's whole purpose is dead
    # code, and a pod killed at cell 100 of 128 costs the whole 15 minutes
    # again. Only cells that SUCCEEDED count as done: a failure is retried,
    # because the common ones here are a lost device and a shared-memory
    # rejection, and a real failure fails again in milliseconds.
    #
    # AND THE RESUME IS GUARDED ON THE CARD, belt as well as braces. The card is
    # in the run id, so a second card lands in a different directory and cannot
    # normally reach a foreign cells.json at all. This check is what catches the
    # ways it could anyway: an explicit --out-dir pointing both runs at one
    # place, a directory copied between pods, or a cells.json written before the
    # card entered the id. It REFUSES rather than starting over, because
    # silently discarding a measured file is its own way to lose an arm.
    rows: list[dict] = []
    done: set[tuple[int, int, int]] = set()
    if cells_path.exists():
        stored = json.loads(cells_path.read_text())
        if isinstance(stored, dict):
            written_by = str(stored.get("card") or "")
            rows = list(stored.get("cells") or [])
        else:
            # The legacy shape: a bare list, with no record of which card wrote
            # it. That is precisely the unknown this guard exists for, so it is
            # refused instead of assumed to be ours.
            written_by, rows = "", list(stored)
        if written_by != plan.card:
            print(f"REFUSED to resume {cells_path}: it was written by card "
                  f"{written_by or '<unrecorded, pre-card-in-id>'!r} and this "
                  f"run is {plan.card!r}. Resuming would publish one card's "
                  "timings under the other's calibration, which is the exact "
                  "defect the card in the run id closes. Move or delete that "
                  "file deliberately. Nothing measured.")
            return 3
        done = {(int(r["block_m"]), int(r["group_m"]), int(r["tiles"]))
                for r in rows if r.get("status") == "ok"}
        print(f"resuming: {len(done)} of {len(plan.cells)} cells already measured")
    stream: dict | None = None
    inputs: dict[int, tuple] = {}
    started = time.time()

    from moe.routing.distributions import realize_counts

    for bm in plan.block_sizes:
        for g in plan.group_sizes:
            for n in (1, *plan.slope_tiles):
                if (bm, g, n) in done:
                    continue
                rows_per_expert = bm * n
                tokens = rows_per_expert * cfg.num_experts // cfg.top_k
                if tokens not in inputs:
                    spec = BenchSpec(cfg, num_tokens=tokens, dtype=plan.dtype,
                                     routing=RoutingSpec("uniform", 0.0),
                                     seed=plan.seed)
                    x_, weights_ = make_inputs(spec, device="cuda")
                    per = tokens * cfg.top_k // cfg.num_experts
                    ids_ = realize_counts([per] * cfg.num_experts, tokens,
                                          cfg.top_k, device="cuda")
                    w_ = torch.full(ids_.shape, 1.0 / cfg.top_k,
                                    dtype=torch.float32, device="cuda")
                    kw_ = vllm_call_kwargs(spec)
                    kw_["activation"] = MoEActivation(kw_["activation"])
                    # ONE cell live at a time: the weight set is 2.8-3.5 GB and
                    # holding 16 tread's worth would run the card out of memory
                    # in a way that would be reported as a kernel failure.
                    inputs = {tokens: (x_, weights_, ids_, w_, kw_)}
                    torch.cuda.empty_cache()
                x, weights, ids, w, kw = inputs[tokens]
                conf = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": plan.block_n,
                        "BLOCK_SIZE_K": plan.block_k, "GROUP_SIZE_M": g,
                        "num_warps": plan.num_warps, "num_stages": plan.num_stages}

                def call(_x=x, _w=weights, _ids=ids, _tw=w, _kw=kw):
                    return fused_experts(hidden_states=_x, w1=_w.w1, w2=_w.w2,
                                         topk_weights=_tw, topk_ids=_ids, **_kw)

                try:
                    with override_config(conf):
                        call()
                        torch.cuda.synchronize()
                        ms, mn, sd = time_call(call, plan.warmup, plan.iters)
                    status, detail = "ok", ""
                except Exception as exc:                       # noqa: BLE001
                    ms = mn = sd = 0.0
                    status, detail = "failed", f"{type(exc).__name__}: {exc}"
                if stream is None and status == "ok":
                    stream = stream_check(weights, plan.stream_reps)
                rows.append({"block_m": bm, "group_m": g, "tiles": n,
                             "rows_per_expert": rows_per_expert, "tokens": tokens,
                             "ms_p50": ms, "ms_min": mn, "ms_stdev": sd,
                             "status": status, "detail": detail})
                cells_path.write_text(json.dumps(
                    {"card": plan.card, "run_id": plan.run_id(), "cells": rows},
                    indent=1) + "\n")
                print(f"  BM={bm:3d} G={g:3d} n={n:3d} r={rows_per_expert:5d} "
                      f"{ms:9.4f} ms  {status}{('  ' + detail) if detail else ''}")

    elapsed = time.time() - started
    print()
    print(f"measured {len(rows)} cells in {elapsed / 60.0:.1f} min")
    if stream:
        print(f"stream check on the real weight buffers: {stream['gbps']:.1f} GB/s "
              f"(ceiling {cal.ceiling_gbps:.1f})")
    fits, refusals, gates, table = score_measured(
        rows, cfg, plan.dtype, plan.block_n, cal, stream, len(plan.cells))
    payload = {"plan": asdict(plan), "run_id": plan.run_id(), "gpu": gpu,
               "calibration": asdict(cal), "stream_check": stream, "cells": rows,
               "elapsed_s": elapsed, "fits": [asdict(f) for f in fits],
               "refusals": [asdict(r) for r in refusals],
               "gates": [asdict(g) for g in gates]}
    (out_dir / "measure.json").write_text(json.dumps(payload, indent=1) + "\n")
    for line in table:
        print(line)
    print()
    for r in refusals:
        print(f"REFUSED  BM={r.block_m} G={r.group_m}: {r.reason}")
    print()
    for g in gates:
        for line in g.render():
            print(line)
    print()
    print(f"wrote {out_dir / 'measure.json'}")
    print("The brackets above stand on their own. To carry them onto the "
          "committed arms, run --rescore, whose P2 band is what licenses that.")
    if any(g.verdict == FAIL and g.kind == "VALIDITY" for g in gates):
        return 2
    return 1 if any(g.verdict == FAIL for g in gates) else 0


def run_score_measured(args) -> int:
    """Score a `measure.json` a pod already wrote, on any machine.

    The pod is rented by the hour and the scoring is free. Splitting them means
    a run whose verdict path has a bug does not have to be paid for twice.
    """
    path = args.score_measured
    if not path.exists():
        print(f"REFUSED: no such file {path}. Nothing scored.")
        return 3
    payload = json.loads(path.read_text())
    plan = payload["plan"]
    cfg = MODEL_CONFIGS[plan["model"]]
    cal = Calibration(**payload["calibration"])
    planned = (len(plan["block_sizes"]) * len(plan["group_sizes"])
               * (1 + len(plan["slope_tiles"])))
    print(f"scoring {path}")
    print(f"device: {payload.get('gpu', 'unknown')}")
    print(f"ceiling: {cal.describe()}")
    fits, refusals, gates, table = score_measured(
        payload["cells"], cfg, plan["dtype"], plan["block_n"], cal,
        payload.get("stream_check"), planned)
    for line in table:
        print(line)
    print()
    for r in refusals:
        print(f"REFUSED  BM={r.block_m} G={r.group_m}: {r.reason}")
    for g in gates:
        for line in g.render():
            print(line)
    if not fits:
        return 3
    if any(g.verdict == FAIL and g.kind == "VALIDITY" for g in gates):
        return 2
    return 1 if any(g.verdict == FAIL for g in gates) else 0


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    The same order `scripts/run_all.sh` resolves it in, so a measured arm lands
    beside every other one on the volume that outlives the pod.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return REPO / "results"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rescore", action="store_true",
                      help="score every committed report under results/published (default)")
    mode.add_argument("--measure", action="store_true",
                      help="the GPU arm: re-measure the anchor at every GROUP_SIZE_M")
    mode.add_argument("--score-measured", type=Path, default=None,
                      help="score a measure.json a pod already wrote; no GPU needed")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan and its cost, measure nothing, exit 3")
    ap.add_argument("--published", type=Path, default=PUBLISHED,
                    help="root of the committed reports")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory the report lands in; defaults to "
                         "results/published (as ANCHOR_RESCORE.txt/.json, files "
                         "and not an arm directory) for --rescore, and to the "
                         "results volume for --measure")
    ap.add_argument("--residuals", action="store_true",
                    help="print the per-tread residual profile of every fit")
    ap.add_argument("--expect-poisoned", type=int, default=2,
                    help="how many arm/model pairs the poisoned-reference check "
                         "must fire on; NON-VACUITY, not a tolerance")
    ap.add_argument("--card", default="",
                    help="card slug to build the --measure run id from. Read "
                         "from the attached device by default, and REFUSED if "
                         "it contradicts one. Its only real use is printing a "
                         "pod's exact path from a laptop dry run")
    ap.add_argument("--model", default="qwen2-57b-a14b", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--tiles", default="32,64",
                    help="BLOCK_SIZE_M values to anchor")
    ap.add_argument("--group-m", default="1,8,16,64",
                    help="GROUP_SIZE_M values; the anchor is measured at each")
    ap.add_argument("--slope-tiles",
                    default=",".join(str(n) for n in range(2, 17)),
                    help="branch treads, all >= 2 so the slope never sees the anchor. "
                         "The default is a DENSE 2..16 ladder, matching the committed "
                         "BLOCK_M=64 ladders, because P3's threshold was taken from "
                         "16- and 33-tread fits and does not transfer to a short one")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--stream-reps", type=int, default=20)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.score_measured is not None:
        return run_score_measured(args)
    if not args.measure:
        args.rescore = True
    if args.out_dir is None:
        if args.measure:
            args.out_dir = results_root() / "memory_branch_anchor"
        else:
            # results/published itself, NOT a dated subdirectory: see RESCORE_STEM.
            args.out_dir = PUBLISHED
    branch = [int(t) for t in args.slope_tiles.split(",")]
    if len(branch) < MIN_BRANCH_TREADS:
        raise SystemExit(
            f"--slope-tiles gives {len(branch)} branch treads and P3 needs at least "
            f"{MIN_BRANCH_TREADS}. On a planted ladder the slope moves 3.2% when the "
            "anchor is dropped at 8 treads, 1.4% at 12 and 0.8% at 16, against a "
            f"{SLOPE_INDEPENDENCE_REL:.1%} threshold taken from the committed 16- and "
            "33-tread fits. A short branch would fail P3 for a reason that is about "
            "the grid and not about the kernel.")
    if any(t < 2 for t in branch):
        raise SystemExit("--slope-tiles must all be >= 2: the anchor may not be "
                         "inside the slope it is compared against")
    return run_measure(args) if args.measure else run_rescore(args)


if __name__ == "__main__":
    sys.exit(main())
