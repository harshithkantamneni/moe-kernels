#!/usr/bin/env python3
"""alpha's denominator, measured instead of extrapolated.

    python scripts/memory_branch_anchor.py --self-test           # planted worlds
    python scripts/memory_branch_anchor.py --rescore            # free, no GPU
    python scripts/memory_branch_anchor.py --rescore --publish  # ... into the tree
    python scripts/memory_branch_anchor.py --measure --dry-run  # plan + cost + MDE
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
there are two and the first version of this paragraph named the wrong one. Write
`phi = Act1 / W` for one M-tile's activation and output traffic in units of one
full weight read, and `delta = D BW / W` for the fixed cost in the same units.
The estimator every published alpha comes from is `B / (A + B)`, the slope over
the fitted LEVEL, and on the ladder above that is

    alpha_published = (alpha_b + phi) / (1 + phi + delta)                 (EXA)

NOT the blend `alpha_b + alpha_a (BM/BN) + BM/K` this file used to assert as
what a ladder returns. That blend is (EXA)'s numerator with its level taken as
1, and no estimator in this repository divides a slope by the weight bytes
alone. This file does not fit `B / (A + B)` at all. It solves (*) for the
slope's own weight term, so `alpha_lo` and `alpha_hi` bracket `alpha_b`
DIRECTLY, with `Act1` already subtracted -- the same quantity the reports print
as `alpha-corrected`. Splitting `phi` into an activation re-read and an output
write needs three BLOCK_N values and belongs to that module's lane; nothing here
assumes a value for alpha_a. The bracket is still the right correction to every
published number, because a published alpha is (EXA) over the same `B` and the
same `Act1`.

WHAT THAT COSTS THE CAP, which is the number this file prints beside the ridge.
The ceiling one tile height reaches is `2 BM / (b (alpha_b + phi))`, the form
`ai_model.exact_cap` carries, so the sweep's published `ai_cap = 2 BM / (b
alpha)` omits `phi` from the denominator and is HIGH: by exactly
`ai_model.lin_overstatement = 1 + phi + delta` when the alpha put into it came
from a `B / (A + B)` fit, and by `(alpha_b + phi) / alpha_b`, which is larger
still, when it came from a bracket end like this file's. `Bracket.ai_cap` is
therefore computed through `ai_model.cap_from_fitted`, `Bracket.lin_cap` keeps
the retracted form so the size of the correction can be printed rather than
described, and the fit table carries both as `cap/ridge` and `lin/ridge`.
`delta` is passed as zero and that is exact rather than optimistic here: the
bracket is built from the SLOPE, which no fixed cost enters, and a fixed cost
does not survive the limit that defines a cap either.

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

THREE MODES.

  --rescore  Free. Scores every committed report under `results/published/`,
             emits the bracket for every anchorable fit, and states the size of
             the correction to every published alpha. No GPU, no pod, seconds.
             Writes to an UNTRACKED session path; `--publish` is the only thing
             that rewrites the committed ANCHOR_RESCORE pair.
  --measure  The GPU arm. Re-measures the anchor tread at every GROUP_SIZE_M so
             the objection "the anchor is measured at a different reuse
             condition" is answered by measurement rather than by argument; and
             re-measures the branch from n=2 up so the slope and the anchor come
             from one process at one clock state. It also re-times the read
             ceiling on the ACTUAL weight buffers and refuses to score if the
             committed calibration is below what those buffers achieve, because
             a ceiling under the data is not a ceiling. Every cell carries its
             own pin assay; see below.
  --self-test  Free, no device, seconds. Scores the PLANTED worlds registered in
             `SELF_TEST_WORLDS`, one per FAIL branch of every M gate plus the
             world in which the tile pin silently failed, and checks each one
             against the exit code this file has registered for it.

THE PIN ASSAY, WITHOUT WHICH THE OTHER GATES MEAN NOTHING. `--measure` forces
BLOCK_SIZE_M and GROUP_SIZE_M through vLLM's own `override_config`. Until
2026-09-02 it entered that context and assumed it took, and this arm's headline
result -- "t(1) does not depend on the swizzle" -- is EXACTLY the signature a
silently failed override produces: one kernel ran at every setting, so every
difference is zero and the report reads as a tidy null. That was not argued, it
was executed. The audit fed `score_measured` 128 synthetic cells carrying the
failed-pin signature (one anchor and one slope, repeated at G=1, 8, 16 and 64)
and gates M0 through M5 all returned PASS, exit 0. Gate M6 now stands in front
of them, on three legs, two of them recorded per cell in `cells.json`:

  WHETHER THE HOOK TOOK. The tile is forced through
  `moe.baselines._framework_config.forcing_tile_config`, which probes the hook
  the way this file used to and then reads `get_config()` back, refusing a
  context that was entered and did not take. A cell it refuses is recorded
  `failed` with `PIN NOT HONOURED` and never timed.

  WHAT vLLM HANDED THE KERNEL. The first call of every cell runs inside
  `moe.baselines._framework_config.recording_tile_config`, which watches
  `try_get_optimal_moe_config` return the config the kernel is actually built
  from. The six tile constants that came back are written into `cells.json`
  beside the six that were asked for, and a cell whose observed tile is not its
  requested tile, or whose source is not the override, voids the run: INVALID,
  not a retry.

  WHETHER A NEW KERNEL WAS BUILT. `TRITON_CACHE_DIR` is pointed at a fresh
  directory per (BLOCK_SIZE_M, GROUP_SIZE_M) and the artefacts that appear while
  that setting first runs are counted -- the assay
  `scripts/block_m_crossing_sweep.py:gate_0_override` calls "the gate that
  decides whether the other four mean anything". A setting that ran cells and
  compiled nothing new ran a kernel that already existed, which at a new tile
  constant is the same failure seen from the other side.

Neither assay is sufficient alone. A hook that stores an override can echo it
back to a reader while the kernel ignores it, so the read-back needs the compile
count; a warm cache compiles nothing while the override works perfectly, so the
compile count needs the read-back. Both are recorded PER CELL and both are
persisted, so a resumed run inherits the evidence of the session that measured
the cells instead of failing for want of an assay it cannot repeat.

THE INSTRUMENT IS THE SWEEP'S. Every cell, and the stream check, are timed by
`moe.bench.timing.time_kernel` at the ladders' own warmup DURATION, because the
anchor has to be the same physical event those ladders measured and since
2026-09-02 they are queue-deep, flush L2 between iterations and sample the SM
clock under load. The verbatim `time_call` this file used to carry did none of
that: it synchronised every iteration with events created inside the loop, which
puts 0.18-0.30 ms of host enqueue time inside the measured interval, and it
warmed for 5 CALLS against the ladders' 20 while the two were compared tread for
tread. Every `KernelTiming` column travels into `cells.json`, so the state each
cell was timed in is on the page rather than in the operator's memory.

EXIT CODES ARE `moe/bench/exit_codes.py`'s, AND THIS FILE USED TO INVERT THEM.
0 DONE, 1 CLAIM_FAIL (measured; a pre-registered claim was refuted, which is a
result and not a retry), 2 REFUSED (nothing was measured; free), 3 INVALID
(measured, and a VALIDITY gate failed; nothing on the page may be quoted),
4 ERROR. The table this file documented until 2026-09-02 read 2 for "a VALIDITY
gate failed AFTER the eight-minute measurement" and 3 for "nothing measured",
the exact opposite of the session driver's, so an invalid run was recorded
REFUSED under the driver's heading "REFUSED BEFORE MEASURING. Nothing below is a
gate", while every genuine refusal was queued as a retry. The stream check is
stored in `cells.json` beside the cells it was measured with for the same
family of reason: it runs only on a freshly timed cell, so a fully resumed run
used to fail M0 for ever and could never reach DONE.

Every scored gate prints one `RESULT: ` line, rendered by `exit_codes`, and
nothing else this file prints starts with that prefix. `--self-test` prints
none at all: a planted world's verdict is not a result about this machine, and a
driver that grepped one would read a plant as a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

#: `moe.bench.timing` is imported LAZILY, everywhere, and this comment is the
#: reason: it imports torch at module scope, while `--rescore`, `--dry-run` and
#: `--self-test` are documented to run on a laptop with no torch at all, and an
#: import here would turn those three into an ImportError before argparse ran.
#: `ai_model`, `exit_codes` and `provenance` import nothing heavier than the
#: standard library, so they are imported normally. The same rule covers
#: `moe.baselines._framework_config`, which reaches torch through `moe.quant`.

REPO = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO / "moe" / "bench" / "hardware"
PUBLISHED = REPO / "results" / "published"

#: The verdict and kind vocabulary is `moe.bench.exit_codes`', not this file's.
#: Two spellings of PASS in one repository is how a verdict falls through a
#: comparison and is scored as whatever the fallthrough happened to be.
PASS, FAIL, UNKNOWN = exit_codes.PASS, exit_codes.FAIL, exit_codes.UNKNOWN
VALIDITY, CLAIM = exit_codes.VALIDITY, exit_codes.CLAIM


def timing_basis() -> str | None:
    """The name of the instrument this file times with, or None off-torch.

    None is not a default: it says the instrument could not be NAMED on this
    machine because torch is absent, which is the laptop `--dry-run`,
    `--rescore` and `--self-test` case, where nothing was timed either.
    """
    try:
        from moe.bench.timing import TIMING_BASIS
    except Exception:                                   # noqa: BLE001
        # Broad on purpose: a torch that is INSTALLED and broken raises OSError
        # on a missing libcudart rather than ImportError (the pod failure
        # `moe/bench/provenance.py` records), and naming the instrument is
        # never worth taking a whole report down for.
        return None
    return TIMING_BASIS

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

#: THE NOISE ASSUMPTION, named once so every MDE in this file's plan output is
#: derived from a number a reader can disagree with rather than from a habit.
#: It is the relative spread of a repeated cell timing: the published H200
#: replicates run 0.76% to 1.82% with a median of 0.77%, so the median is the
#: default and the range is why `--noise` exists. It is an ASSUMPTION about the
#: pod this arm has not yet run on, not a measurement of it, and the plan says
#: so where it prints it.
CELL_SPREAD_REL = 0.0077

#: Two-sided, 5% size, 80% power: 1.96 + 0.84. The multiplier that turns a
#: standard deviation into a smallest detectable difference, written out so the
#: convention behind every MDE below is visible instead of folded into a
#: constant.
MDE_Z = 2.80

#: C5's threshold, AND IT IS A PRIOR. It is the precision published alphas are
#: quoted at (three decimals, so a shift of 0.05 is 50 times the last digit),
#: not a quantity derived from `CELL_SPREAD_REL`, and the plan prints it beside
#: the MDE so the two are never confused. The audit found this literal carrying
#: no justification at all; it now carries this one, which is a statement about
#: how the number is REPORTED rather than about how well it was measured.
C5_SHIFT_PRIOR = 0.05


def mde_ratio(spread_rel: float = CELL_SPREAD_REL) -> float:
    """Smallest RATIO of two cell timings this instrument can call different.

    `z sigma sqrt(2)`: two independent cells, each carrying `spread_rel` of
    relative noise, compared to each other. This is what M1's thresholds have to
    clear to mean anything, and printing it beside them is the only way a reader
    can tell a threshold that can decide from one that cannot.
    """
    return MDE_Z * spread_rel * math.sqrt(2.0)


def mde_alpha(alpha: float = POOLED_ALPHA, act1_over_w: float = 0.0,
              spread_rel: float = CELL_SPREAD_REL) -> float:
    """The same difference, carried through (*) into alpha.

    `alpha_lo = B (W + Act1) / (t(1) W) - Act1/W`, so a relative perturbation
    `eps` of `t(1)` moves it by `eps (alpha + Act1/W)`. A LOWER BOUND on the
    real MDE, and labelled as one wherever it is printed: it propagates the
    anchor's noise only, and the branch slope carries its own.
    """
    return mde_ratio(spread_rel) * (alpha + act1_over_w)


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

    @property
    def phi(self) -> float:
        """`Act1 / W`: one M-tile's activation and output traffic, in units of
        one full weight read.

        The same quantity `moe/bench/ai_model.py` calls `phi`, evaluated on the
        FUSED layer this file measures rather than on a single GEMM, and the
        term the sweep's `alpha-corrected` column subtracts. It is exact
        arithmetic over the model config, so it carries no fitted parameter and
        no assumption about alpha_a.
        """
        return self.act1_bytes / self.w_bytes

    def lin_cap(self, dtype_b: int, alpha: float) -> float:
        """`2 BM / (b alpha)`: the RETRACTED cap, kept only to be divided by.

        This is the expression the sweep publishes as `ai_cap` and the one every
        cap number in this study was computed with. It omits `phi` from the
        denominator, so it is HIGH -- by `1 + phi + delta` when the alpha came
        from a `B / (A + B)` fit, and by `(alpha_b + phi) / alpha_b` when it
        came from a bracket end like `lo`. It stays on the page beside
        `ai_cap` because a correction whose size is not printed is a correction
        the next reader has to take on trust, which is how the identity this
        method used to assert as exact survived unexamined until 2026-09-02.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive to cap arithmetic intensity")
        return 2.0 * self.block_m / (dtype_b * alpha)

    def ai_cap(self, dtype_b: int, alpha: float) -> float:
        """`2 BM / (b (alpha_b + phi))`: the ceiling this tile height really has.

        `alpha` is an alpha_b -- a bracket end, with `Act1` already subtracted
        -- and the cap is `moe/bench/ai_model.exact_cap`'s form evaluated on the
        fused layer. It is reached THROUGH `ai_model.cap_from_fitted` rather
        than written out here, so that one module owns the arithmetic and the
        recovered miss fraction passes that module's [0, 1] wall; the round trip
        maps alpha_b to the (EXA) reading `(alpha_b + phi) / (1 + phi)` and back,
        and `cap_from_fitted` refuses anything the three-term model cannot hold.

        `delta` IS ZERO AND THAT IS EXACT, not a convenient default. `delta` is
        the fused layer's fixed cost divided by the weight read, and it enters
        (EXA) only through the LEVEL. This file never divides by a level: (*) is
        solved on the SLOPE, so alpha_b comes out fixed-cost-free, and a fixed
        cost does not survive the M -> infinity limit that defines a cap either.

        WHAT THIS IS STILL CONSERVATIVE ABOUT, in the direction that makes the
        BLOCK_M <= 64 claim harder rather than easier to keep: taken at the
        bracket's LOW alpha it is the LARGEST cap the anchor ambiguity allows. A
        tile that still cannot reach the ridge there has not been helped over
        the line by the anchor. The SECOND conservatism this docstring used to
        claim -- that subtracting the activation term overstates the cap again
        -- was the retracted reading itself, and is gone: the term is now back
        in the denominator where it belongs.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive to cap arithmetic intensity")
        level = ai_model.lin_overstatement(phi=self.phi, delta=0.0)
        return ai_model.cap_from_fitted((alpha + self.phi) / level,
                                        block_m=self.block_m, b=dtype_b,
                                        phi=self.phi, delta=0.0)


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
    #: The SM clock the ROOF was measured at, for `timing.clock_flags`' LEVEL
    #: verdict. None means this calibration records no clock, and then LEVEL is
    #: None on every cell -- "not determined", never "fine". A guessed reference
    #: would exclude real cells or admit throttled ones, both silently.
    reference_clock_mhz: float | None = None
    #: Which field the clock came from, because the three that could supply it
    #: have disagreed by 450 MHz on one H200.
    reference_clock_source: str = ""

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
    clock, clock_source = reference_clock_from(detail, slug)
    return Calibration(
        slug=slug, name=data.get("name", slug), checked_on=str(data.get("checked_on", "")),
        measured_commit=str(data.get("measured_commit", "")), patterns=patterns,
        ceiling_pattern=ceiling_pattern, ceiling_gbps=patterns[ceiling_pattern], pin_gbps=pin,
        dense_tflops=dense, ridge=dense.get("bf16", max(dense.values())) / bw_tb_s,
        reference_clock_mhz=clock, reference_clock_source=clock_source)


def reference_clock_from(detail: dict, slug: str) -> tuple[float | None, str]:
    """The clock the roof was measured at, and which field said so.

    THE SAME THREE FIELDS IN THE SAME ORDER as `block_m_crossing_sweep`'s
    `reference_clock_mhz`, deliberately: the ladders this arm re-anchors take
    their LEVEL verdict from that order, and an anchor scored against a
    different reference would be excluded (or admitted) on a rule the ladders
    never applied. Most direct first: samples taken WHILE the calibration's
    dense GEMM ran, then the scalar that GEMM published, then the compute
    settle's final plateau. They have disagreed -- eleven calibrations of one
    H200 recorded 1485-1935 MHz for the scalar while their own settle histories
    sat at 1455-1515 -- so the field that answered is returned with the number.

    `(None, reason)` when the calibration carries no clock at all. That is not a
    failure: `clock_level_ok` is then None on every cell, which is "not
    determined", and nothing is excluded on a number nobody measured.
    """
    median = ((detail.get("gemm_clock") or {}).get("median_mhz"))
    if median:
        return float(median), (f"{slug}: median of the samples taken while the "
                               "calibration's dense GEMM ran")
    scalar = detail.get("gemm_clock_mhz")
    if scalar:
        return float(scalar), (f"{slug}: gemm_clock_mhz, the scalar the "
                               "calibration published for its dense GEMM")
    plateau = (detail.get("settle") or {}).get("final_mhz")
    if plateau:
        return float(plateau), (f"{slug}: the compute settle's final plateau; "
                                "the calibration recorded no GEMM clock")
    return None, (f"{slug}: the calibration carries no clock, so the LEVEL of "
                  "every cell's under-load clock is not determinable here")


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
# What ridge the committed reports carry. Counted, never asserted.
# --------------------------------------------------------------------------

#: The ridge every published sweep was RUN with: an H200 figure, 701.6 TFLOP/s
#: over 4377.2 GB/s, that `--ridge` defaulted to and that the cross-card driver
#: passed to arms on two different cards. It is here so the census can say how
#: many reports still carry it; it is not a fallback and nothing is scored
#: against it.
SWEPT_RIDGE = 160.3

#: How far a report's stamped ridge may sit from its card's calibrated one and
#: still count as that card's. The reports round to one decimal (162.8) and the
#: calibrations do not (162.81), so anything tighter than half of the last
#: printed place would classify a correctly rescored report as a stranger.
RIDGE_MATCH_TOL = 0.05


def ridge_census(root: Path, cals: dict[str, Calibration]) -> dict:
    """Which ridge each committed report stamps, read off the files.

    THIS EXISTS BECAUSE THE PARAGRAPH IT REPLACES WAS A LITERAL. Until
    2026-09-02 the rescore transcript said, as hardcoded text, "Every one of
    these reports carries ridge=160.3 and ridge_band=[160.3, 176.2] ... 160.3 is
    a stale H200 band and belongs to NEITHER card". On this same branch
    `scripts/rescore_published_reports.py` rewrote all 26 reports to their own
    card's calibration, so that sentence regenerated FALSE on every run and a
    reader who opened one of the reports it describes found a different number
    in it. A census cannot go stale that way: it counts what is on disk now, and
    it NAMES every report that still carries the swept ridge or a ridge its own
    card's calibration does not give, instead of asserting that none do.

    `rescored_from` is counted too, because the substitution is real history and
    the fix must not erase it: a report that was rescored says what it was
    rescored FROM, and that is where 160.3 now lives.
    """
    own, swept, stranger, unattributed, rescored = 0, [], [], [], 0
    per_card: dict[str, list[float]] = {}
    total = 0
    for report in sorted(root.glob("*/*.report.json")):
        if (report.parent / "SUPERSEDED").exists():
            continue
        total += 1
        try:
            doc = json.loads(report.read_text())
        except (OSError, ValueError):
            stranger.append(f"{report.parent.name}/{report.name} is unreadable")
            continue
        name = f"{report.parent.name}/{report.name}"
        if doc.get("rescored_from"):
            rescored += 1
        ridge = doc.get("ridge")
        if ridge is None:
            stranger.append(f"{name} stamps no ridge at all")
            continue
        slug = calibration_slug_for(report.parent.name, list(cals))
        if slug is None:
            unattributed.append(f"{name} names no card this run calibrated")
            continue
        per_card.setdefault(slug, []).append(float(ridge))
        if abs(float(ridge) - cals[slug].ridge) <= RIDGE_MATCH_TOL:
            own += 1
        elif abs(float(ridge) - SWEPT_RIDGE) <= RIDGE_MATCH_TOL:
            swept.append(f"{name} still carries the swept {SWEPT_RIDGE}")
        else:
            stranger.append(f"{name} carries ridge {ridge}, which is neither "
                            f"{slug}'s {cals[slug].ridge:.2f} nor the swept "
                            f"{SWEPT_RIDGE}")
    return {"total": total, "own_card": own, "rescored_from": rescored,
            "still_swept": swept, "strangers": stranger,
            "unattributed": unattributed, "per_card": per_card}


def render_ridge_census(census: dict, cals: dict[str, Calibration]) -> list[str]:
    """The census as the transcript prints it, with both verdicts spelled out.

    The clean case is a sentence AND a count, never a count alone: "26 of 26"
    with no statement of what was checked is the shape the retracted paragraph
    had. The dirty case lists the files, because a reader who is told some
    report disagrees with its own card has to be able to open that report.
    """
    out = ["  RIDGE PROVENANCE, because C3 is scored against it. Counted from "
           "the report files",
           "  themselves on this run, not asserted: a sentence about them went "
           "stale once already.",
           f"    {census['total']} report(s) examined; {census['own_card']} "
           f"stamp their own card's calibrated ridge,",
           f"    and {census['rescored_from']} carry a `rescored_from` block "
           f"naming what they were swept with",
           f"    (the {SWEPT_RIDGE} default, an H200 band that belongs to "
           "NEITHER card, which",
           "    `scripts/rescore_published_reports.py` replaced on this branch)."]
    for slug in sorted(census["per_card"]):
        vals = census["per_card"][slug]
        seen = ", ".join(f"{v:.2f}" for v in sorted(set(vals)))
        peak = cals[slug].dense_tflops.get("bf16", max(cals[slug].dense_tflops.values()))
        out.append(f"    {slug:24s} calibration {cals[slug].ridge:6.2f} "
                   f"FLOP/byte (dense {peak:.2f} TFLOP/s over its own triad "
                   f"bandwidth,")
        out.append(f"    {'':24s} measured {cals[slug].checked_on}); "
                   f"{len(vals)} report(s) stamp {seen}")
    for label, rows in (("STILL CARRYING THE SWEPT RIDGE", census["still_swept"]),
                        ("STAMPING A RIDGE THEIR OWN CARD DOES NOT GIVE",
                         census["strangers"]),
                        ("NAMING NO CALIBRATED CARD", census["unattributed"])):
        if rows:
            out.append(f"    {label}, {len(rows)}:")
            out += [f"      {r}" for r in rows]
    if not (census["still_swept"] or census["strangers"] or census["unattributed"]):
        out.append("    No report carries the swept ridge, a stranger ridge, or "
                   "an unattributable card.")
    out.append("  C3 uses each card's OWN contemporaneous calibration and never "
               "a report's stamp.")
    return out


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
    #: `Act1 / W` for this cell: what one M-tile costs in activations and output,
    #: in units of one full weight read. Carried per fit because it is the whole
    #: difference between the two cap columns and it depends on the model and the
    #: tile, so a single number could not stand for it.
    phi: float
    #: The RETRACTED `2 BM / (b alpha_lo)` over the same ridge. Kept beside the
    #: corrected column so the size of the correction is on the page. Where this
    #: is above 1.0 and `cap_over_ridge_at_lo` is below it, a published cap
    #: cleared a ridge that the corrected one does not.
    lin_over_ridge_at_lo: float
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
            phi=br.phi,
            lin_over_ridge_at_lo=br.lin_cap(dtype_bytes(dtype), br.lo) / cal.ridge,
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

#: The one-token name each gate answers to on its `RESULT:` line. A name is one
#: run of non-whitespace by `moe.bench.exit_codes.result_line`'s own rule, and
#: these are the words a driver greps, so they are fixed here rather than
#: derived from the claim text -- which is prose and gets edited.
GATE_NAMES = {
    "V1": "scored_something",
    "V2": "slope_reproduction",
    "V3": "anchor_present",
    "V4": "poisoned_reference_fires",
    "V5": "bracket_ordered",
    "V6": "assumption_a",
    "C1": "physicality",
    "C2": "containment",
    "C3": "tile_cap",
    "C4": "pooled_alpha_excluded",
    "C5": "correction_smaller_than_the_prior",
    "M0": "stream_ceiling",
    "M1": "anchor_swizzle_invariant",
    "M2": "anchor_rate_in_band",
    "M3": "slope_independent",
    "M4": "measured_bracket_ordered",
    "M5": "every_cell_measured",
    "M6": "pin_took_effect",
}


@dataclass
class Gate:
    number: str
    kind: str            # VALIDITY or CLAIM, in exit_codes' spelling
    claim: str
    verdict: str         # PASS, FAIL or UNKNOWN, in exit_codes' spelling
    measured: str
    threshold: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The token on the RESULT line. Missing from `GATE_NAMES` is a bug at
        the definition site, not something to paper over with a fallback: a
        gate whose name changed shape would quietly leave a driver's summary."""
        return GATE_NAMES[self.number]

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`' vocabulary."""
        return self.kind, self.name, self.verdict

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        `exit_codes.result_line` renders it and `parse_result_lines` reads it
        back, anchored at column zero. The human `GATE V1 ...` block below is
        for a reader; only this line is the machine contract, and nothing else
        this file prints begins with `RESULT: `. The old summary grep matched
        free text (`floor|sigma`) and printed a refused arm's imported constants
        as measured output; that is what this format exists against.
        """
        detail = f"{self.claim} | measured {self.measured} | gate {self.threshold}"
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        """The gate as a block of lines, RESULT line first. Always.

        NO SUPPRESSION SWITCH, and the one that used to sit here was worse than
        useless: it documented `--self-test` as the caller that passed False,
        and `self_test` has never called this method at all. It prints its own
        `[PASS] <world>` lines, which is what actually keeps a planted verdict
        out of the machine format -- a driver that grepped a plant out of a
        self-test log would be reading it as a measurement, the same defect in
        the other direction as a refused arm's log matching a gate regex. A flag
        would have made that guarantee something a caller has to remember to
        ask for; having no flag is the same guarantee with nothing to forget.
        """
        out = [self.result_line()]
        out += [f"GATE {self.number:3s} {self.kind:8s} {self.verdict:7s} {self.claim}",
                f"                        measured {self.measured}   gate {self.threshold}"]
        out += [f"                        {line}" for line in self.lines]
        if self.verdict != PASS and self.invalidates:
            out.append(f"                        INVALIDATES: {self.invalidates}")
        return out


def gate_v1_non_vacuity(fits, refusals) -> Gate:
    cards = {f.card for f in fits}
    ok = len(fits) >= MIN_SCORED_FITS and len(cards) >= 1
    return Gate(
        "V1", VALIDITY, "the run actually scored something",
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
        return Gate("V2", VALIDITY, "OLS on the published prefix returns the "
                    "published slope", FAIL, "no fits to reproduce",
                    f"<= {SLOPE_REPRODUCTION_REL:.0e}", "the correction table")
    ok = worst <= SLOPE_REPRODUCTION_REL
    return Gate(
        "V2", VALIDITY, "OLS on the published prefix returns the published slope",
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
        "V3", VALIDITY, "every scored fit carries a MEASURED n=1 tread",
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
        "V4", VALIDITY, "the poisoned-compute-reference check fires on the known case",
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
        return Gate("V5", VALIDITY, "the bracket is ordered", FAIL, "no fits", "lo <= hi",
                    "everything below")
    ratio = tight.bw_anchor_gbps / tight.bw_ceiling_gbps
    ok = all(f.alpha_lo <= f.alpha_hi + 1e-12 for f in fits) and ratio <= 1.0
    return Gate(
        "V5", VALIDITY, "the anchor rate never exceeds the card's measured ceiling",
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
        "V6", VALIDITY, "ASSUMPTION A: no branch runs slower than its own anchor",
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
        "C1", CLAIM, "every published alpha implies a bandwidth the card has",
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
        "C2", CLAIM, "every published alpha lies inside its own anchor bracket",
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

    Scored against each card's OWN ridge -- the reports' own stamp is reported
    by the census above, never used here -- and through the CORRECTED cap.

    THE CAP THIS GATE USED TO SCORE WAS THE RETRACTED ONE. Until 2026-09-02 this
    docstring said the study's `2 BM / (b alpha)` "agrees with" the corrected
    formula "once the alpha is the fitted composite". It does not.
    `moe/bench/ai_model.py` shows a `B / (A + B)` fit returns
    `(alpha_b + phi) / (1 + phi + delta)`, so a cap read off one is high by
    `ai_model.lin_overstatement`, and a cap read off a bracket end -- which is
    an alpha_b, with `Act1` already subtracted -- is high by
    `(alpha_b + phi) / alpha_b`, which is larger. `Bracket.ai_cap` now goes
    through `ai_model.cap_from_fitted`; `lin_over_ridge_at_lo` keeps the
    retracted number beside it, and the detail lines print both so a reader can
    see how much of this verdict is the correction and how much is the data.

    ONE CONSERVATISM SURVIVES and it is the one that matters: the cap is taken
    at the bracket's LOW alpha, the LARGEST ceiling the anchor ambiguity allows,
    so a tile that still cannot reach the ridge there was not pushed under the
    line by the anchor. The second conservatism this docstring claimed was the
    retracted reading itself.
    """
    small = [f for f in fits if f.block_m <= 64]
    if not small:
        return Gate("C3", CLAIM, "BLOCK_M <= 64 caps below the ridge", FAIL,
                    "no BLOCK_M <= 64 fits scored", "< 1.0 for every fit",
                    "the study's one surviving result, which cannot be checked here")
    worst = max(small, key=lambda f: f.cap_over_ridge_at_lo)
    ok = worst.cap_over_ridge_at_lo < 1.0
    by_bm: dict[int, list[float]] = {}
    lin_by_bm: dict[int, list[float]] = {}
    for f in small:
        by_bm.setdefault(f.block_m, []).append(f.cap_over_ridge_at_lo)
        lin_by_bm.setdefault(f.block_m, []).append(f.lin_over_ridge_at_lo)
    return Gate(
        "C3", CLAIM, "at the bracket's LOWEST alpha, BLOCK_M <= 64 still caps below the ridge",
        PASS if ok else FAIL,
        f"worst {worst.cap_over_ridge_at_lo:.3f} of the ridge at {worst.arm[:26]}/"
        f"{worst.model} G={worst.group_m} BM={worst.block_m}",
        "< 1.000 for every BLOCK_M <= 64 fit",
        "the one finding that survived the adversarial evaluation. A FAIL here "
        "means the cap was an artefact of the anchor and not a property of the tile.",
        [f"  BM={bm:3d}  cap/ridge in [{min(v):.3f}, {max(v):.3f}] over {len(v)} "
         f"fits; the RETRACTED 2BM/(b alpha) reads "
         f"[{min(lin_by_bm[bm]):.3f}, {max(lin_by_bm[bm]):.3f}]"
         for bm, v in sorted(by_bm.items())])


def gate_c4_pooled_alpha(fits) -> Gate:
    """SURFACE.txt's `0 of 12 fits within 0.05 of 0.558`, re-scored."""
    hits = [f for f in fits if f.contains_pooled]
    return Gate(
        "C4", CLAIM, f"no anchor bracket admits the pooled alpha {POOLED_ALPHA}",
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
    """How big the correction is, as a gate so it cannot be read as a footnote.

    THE THRESHOLD IS A PRIOR AND THE GATE SAYS SO. `C5_SHIFT_PRIOR` is the
    precision a published alpha is quoted at, not a quantity derived from the
    arm's noise; the plan prints it beside `mde_alpha()` so a reader can see
    that it is about 3x the smallest shift the instrument could resolve, and
    can therefore tell a claim about REPORTING from a claim about MEASUREMENT.
    A threshold whose origin is not printed is a number the next reader has to
    take on trust, which is how this literal survived unexamined until the
    2026-09-02 audit.
    """
    if not fits:
        return Gate("C5", CLAIM, "the anchor correction is small", FAIL, "no fits",
                    f"median |published - bracket midpoint| <= {C5_SHIFT_PRIOR} "
                    "(a PRIOR)", "the whole table")
    shifts = [abs(f.alpha_published_corrected - 0.5 * (f.alpha_lo + f.alpha_hi))
              for f in fits]
    widths = [f.alpha_hi - f.alpha_lo for f in fits]
    med = statistics.median(shifts)
    ok = med <= C5_SHIFT_PRIOR
    return Gate(
        "C5", CLAIM, "re-anchoring moves the published alpha by less than the "
        f"{C5_SHIFT_PRIOR} it is quoted to",
        PASS if ok else FAIL,
        f"median shift {med:.3f} (max {max(shifts):.3f}); median bracket width "
        f"{statistics.median(widths):.3f} (max {max(widths):.3f})",
        f"median shift <= {C5_SHIFT_PRIOR}, a PRIOR (the quoted precision), not "
        f"noise-derived; the anchor MDE is {mde_alpha():.3f}",
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


#: WHEN PUBLISHED, the re-scoring lands as two FILES beside
#: `CALIBRATION_PROVENANCE.md` and `NOISE_FLOOR.json`, not as a dated directory.
#: `results/published/*/` is the namespace for measurement ARMS:
#: `moe/bench/published.py` and `tests/test_calibration_provenance.py` both
#: enumerate arms with `is_dir()`, so a `2026-09-01-anchor-rescore/` directory
#: would be counted as an eleventh arm and asked for a calibration provenance it
#: does not have. It is a cross-arm artefact, and the stable names also make a
#: re-run idempotent instead of leaving one directory per day.
RESCORE_STEM = "ANCHOR_RESCORE"


def repo_relative(path: Path) -> str:
    """`results/published`, not `/Users/somebody/Desktop/moe-kernels/results/...`.

    An absolute path is a fact about ONE laptop, and this one was being written
    into a tracked file: `--rescore` rewrote `results/published/ANCHOR_RESCORE.txt`
    with the author's home directory in it on every run, including from the
    session driver's own `--dry-run`, so the tree was dirty from arm one and
    44,872 of 100,144 published rows carry `git_dirty=True`. Falls back to the
    absolute form for a path outside the repo, which is every pod path, because
    there the absolute name is the only true one.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def report_output_paths(out_dir: Path, lines: list[str], payload: dict) -> list[str]:
    """Write the pair and say, per file, what git will do with it.

    THE DEFAULT DESTINATION IS UNTRACKED, and `--publish` is the only way into
    the tree. Nothing here decides that; `main` picks the directory. What this
    function guarantees is that the operator is TOLD which of the two they got,
    because the failure it is named against is a tracked file rewritten by a
    command nobody thought was a write.

    HOW THE COMMITTED PAIR IS MEANT TO BE REGENERATED, because the JSON stamps
    the tree it was written from and a stamp naming no committed state is worth
    nothing:

        git status --porcelain            # must be EMPTY before the run
        .venv/bin/python scripts/memory_branch_anchor.py --rescore --publish
        git add results/published/ANCHOR_RESCORE.txt results/published/ANCHOR_RESCORE.json

    RUN IT ON A CLEAN TREE, ALWAYS. The published JSON carried
    `git_dirty: true, git_dirty_files: 2` for one commit, at a `git_sha` two
    commits behind the branch tip, so it named neither the tree it came from
    nor any tree in the history. Regenerating from a clean checkout fixes the
    dirty flag; nothing fixes the SHA lag, because a file cannot contain the
    hash of the commit that carries it. The lag is one commit and the stamped
    SHA is the PARENT of the commit that carries the pair, which is a real
    ancestor a reader can check out.

    WHAT MAKES THE LAG HARMLESS is that the content does not depend on it.
    `--rescore` reads only the committed reports and the committed
    calibrations, so a re-run at any later HEAD whose analysis code is
    unchanged reproduces this pair byte for byte apart from `git_sha`,
    `git_dirty*` and `utc`. That is the check to run when the stamp looks stale:
    re-run into a scratch `--out-dir` and diff, rather than trusting or
    distrusting the SHA alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{RESCORE_STEM}.txt"
    js = out_dir / f"{RESCORE_STEM}.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    notes = []
    for path in (txt, js):
        ig = git_ignored(path)
        state = ("untracked; git will not commit it" if ig else
                 "TRACKED: this write changes the repository" if ig is False else
                 "git could not answer; outside the work tree")
        notes.append(f"  {path}  [{state}]")
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
        "cap/ridge is 2BM/(b (alpha_lo + phi)) over the card's own ridge; "
        "lin/ridge is the RETRACTED",
        "2BM/(b alpha_lo) over the same ridge, printed so the correction's size "
        "is on the page.",
        "",
        f"  {'card':6s} {'model':10s} {'G':>3s} {'BN':>4s} {'st':>3s} {'BM':>4s} "
        f"{'a_pub':>6s} {'lo':>6s} {'hi':>6s} {'hi@pin':>7s} {'elev':>7s} "
        f"{'BW_pub':>7s} {'BW_1':>6s} {'in?':>4s} {'cap/ridge':>9s} "
        f"{'lin/ridge':>9s}",
        "  " + "-" * 118,
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
            f"{f.bw_anchor_gbps:6.0f} {flag:>4s} {f.cap_over_ridge_at_lo:9.3f} "
            f"{f.lin_over_ridge_at_lo:9.3f}"
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
                "best case for a small tile reaching its roof, and through the "
                "CORRECTED cap 2BM/(b (alpha_b + phi)) rather than the retracted "
                "2BM/(b alpha), and it still caps below the ridge on every fit. "
                "The correction moves this verdict the SAFE way: it lowers every "
                "cap, so the claim is easier to keep than it was, and the "
                "lin/ridge column says by how much."]
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
    say(f"published root : {repo_relative(args.published)}")
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
    say(f"  C5 correction size        median |published - bracket midpoint| <= "
        f"{C5_SHIFT_PRIOR} (a PRIOR)")
    say("")
    for line in render_mde(MODEL_CONFIGS["mixtral-8x7b"], "bf16", 32, args.noise):
        say(line)
    say("")

    if args.dry_run:
        reports = sorted(args.published.glob("*/*.report.json"))
        say(f"DRY RUN. {len(reports)} report(s) would be scored, at no GPU cost:")
        for r in reports:
            say(f"  {r.relative_to(REPO)}")
        say("")
        say("Nothing was measured and nothing was written. "
            + exit_codes.describe(exit_codes.REFUSED))
        return exit_codes.REFUSED

    fits, refusals, cals = scan_published(args.published)
    say("CARD CALIBRATIONS USED (the bracket's upper end, and the ridge for C3)")
    for slug in sorted(cals):
        say(f"  {cals[slug].describe()}")
    say("")
    for line in render_ridge_census(ridge_census(args.published, cals), cals):
        say(line)
    say("")
    if refusals:
        say(f"REFUSED, {len(refusals)} fit(s) -- listed, never defaulted to a number")
        for r in refusals:
            say(f"  {r.arm[:30]:30s} {r.model[:14]:14s} G={r.group_m:2d} BN={r.block_n:3d} "
                f"BM={r.block_m:3d}")
            say(f"      {r.reason}")
        say("")
    if not fits:
        say("NOTHING SCORED. No report carried an identifiable ladder with an "
            "anchor. " + exit_codes.describe(exit_codes.REFUSED))
        return exit_codes.REFUSED

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
        "published_root": repo_relative(args.published),
        "pooled_alpha": POOLED_ALPHA,
        "noise_assumption_rel": args.noise,
        "c5_shift_prior": C5_SHIFT_PRIOR,
        "mde_alpha": mde_alpha(POOLED_ALPHA, 0.0, args.noise),
        "calibrations": {s: asdict(c) for s, c in cals.items()},
        "gates": [asdict(g) for g in gates],
        "refusals": [asdict(r) for r in refusals],
        "fits": [asdict(f) for f in fits],
    }
    payload = PV.provenance_block(
        # NOTHING WAS TIMED HERE and the field says so in words. The numbers
        # scored above were timed by the committed reports' own instrument,
        # which is not this repository's current one; a run of `--measure`
        # carries `timing.TIMING_BASIS` instead, and the difference between the
        # two strings is exactly the open question the audit left on the table.
        instrument="imported: the committed reports' own timings; --rescore "
                   "times nothing",
        # Per-card, so no single number belongs at the top level. The source is
        # given without a number, which `provenance_block` accepts; a number
        # without a source is what it lists as missing.
        ridge_source="each card's own moe/bench/hardware/measured_<slug>.yaml; "
                     "see calibrations",
        bandwidth_source="each card's own measured_<slug>.yaml, largest "
                         "demonstrated pattern; see calibrations",
    ).stamp(payload)
    say("")
    say("WROTE")
    for note in report_output_paths(args.out_dir, lines, payload):
        print(note)
    if not args.publish:
        print("  (default destination, untracked. --publish rewrites the "
              f"committed {repo_relative(PUBLISHED)}/{RESCORE_STEM}.txt/.json.)")

    rc = exit_codes.classify(g.scored() for g in gates)
    print(exit_codes.describe(rc))
    return rc


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


#: Plan field -> the short knob name `run_id` hashes and shows. Every field
#: except the card appears here; `MeasurePlan.run_id` raises `KeyError` on a
#: field that does not, which is the point. The short names are what makes a
#: directory readable in `ls`; the completeness is what stops two different
#: runs sharing one.
ID_KNOBS = {
    "model": "model", "dtype": "dtype", "block_sizes": "bm",
    "group_sizes": "g", "slope_tiles": "treads", "block_n": "n",
    "block_k": "k", "num_warps": "w", "num_stages": "s", "seed": "seed",
    "warmup_ms": "warmup", "cell_budget_ms": "budget", "trials": "trials",
    "l2_flush": "flush",
}


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
    #: MILLISECONDS of delivered GPU load, not a call count. `--iters` and
    #: `--warmup 5` are retired: `timing.time_kernel` warms for a DURATION and
    #: sizes its own iteration count from `cell_budget_ms`, and a count is the
    #: wrong unit for a warmup anyway. The old default warmed this arm for 5
    #: calls while the ladders it re-anchors warmed for 20, and the two were
    #: then compared tread for tread.
    warmup_ms: float
    #: Target KERNEL time per trial; the instrument sizes `iters` from it.
    cell_budget_ms: float
    trials: int
    #: Whether every timed iteration starts on a cold L2. In the id because a
    #: flushed and an unflushed cell are different measurements and must never
    #: share a directory, and the ladders are flushed.
    l2_flush: bool

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
        """EVERY field of this plan is in the key, by construction.

        `moe.bench.provenance.run_id` builds it as of 2026-09-02, not a private
        hash here: that function raises `NoCard` on a missing card and
        `UnresolvedKnob` on a None or empty value, hashes the knobs in sorted
        order so the id does not depend on the order they were named in, and
        puts the card slug at the FRONT where `ls` shows it. Three scripts had
        each re-implemented a subset of it and each had left a different knob
        out; the collisions that cost are in that module's docstring, and two of
        them are this arm's own hazards -- GROUP_SIZE_M, which lost a whole G=16
        arm to a resumed G=1 directory, and THE CARD, which the operator sweeps
        by moving to another pod while the results root is a network volume that
        outlives it.

        The knob names come from `ID_KNOBS`, and a field missing from that map
        raises `KeyError` here rather than dropping quietly out of the key. That
        is the whole point of the indirection: the failure mode being defended
        against is a knob that was added to the plan and never added to the id.
        """
        knobs = {ID_KNOBS[f.name]: getattr(self, f.name)
                 for f in fields(self) if f.name != "card"}
        return PV.run_id(card=self.card, **knobs)

    def estimated_seconds(self, compile_s: float = 12.0) -> float:
        """Timed work plus one Triton compile per distinct (BLOCK_M, GROUP_SIZE_M).

        Deliberately crude and stated as such. Its job is to stop a run being
        started without a wall-clock number attached, not to be accurate. The
        per-cell term is now the instrument's OWN budget -- one warmup plus
        `trials` trials of `cell_budget_ms` -- rather than a flat 6 seconds,
        because those are the two knobs that set it and a cost estimate that
        ignored them would not move when the operator doubled the budget.
        """
        settings = len(self.block_sizes) * len(self.group_sizes)
        per_cell_s = (self.warmup_ms + self.trials * self.cell_budget_ms) / 1e3
        # +1 cell for the stream check, which is timed by the same instrument.
        return (len(self.cells) + 1) * per_cell_s + settings * compile_s


def render_mde(cfg, dtype: str, block_m: int, spread_rel: float) -> list[str]:
    """The smallest difference this grid can call, from a STATED assumption.

    B14: no arm in this study stated one, so every threshold in it read as a
    number the author liked. These four lines say what noise is assumed, where
    the assumption came from, what it implies for the two quantities this file
    gates on, and which of its thresholds is a prior rather than a derivation.
    A reader who disagrees with the assumption can change it with `--noise` and
    read the consequences off the same block.
    """
    w_bytes, act1 = anchor_bytes(cfg, dtype, block_m)
    c = act1 / w_bytes
    ratio = mde_ratio(spread_rel)
    alpha_mde = mde_alpha(POOLED_ALPHA, c, spread_rel)
    return [
        "  MINIMUM DETECTABLE EFFECT, from a stated assumption and not from habit:",
        f"    noise assumption   sigma = {spread_rel:.2%} relative spread on a "
        "repeated cell timing",
        f"                       ({CELL_SPREAD_REL:.2%} is the median of the "
        "published H200 replicates, which run 0.76-1.82%; --noise overrides it. "
        "It is an",
        "                       ASSUMPTION about a pod this arm has not run on, "
        "not a measurement of one.)",
        f"    two-sided, 5% size, 80% power: z = {MDE_Z}",
        f"    smallest anchor RATIO callable   {ratio:.2%}   (z sigma sqrt2, two "
        "independent cells)",
        f"      M1 gates the anchor spread at {ANCHOR_INVARIANCE_SMALL_G:.1%} "
        f"across G<=16 and {ANCHOR_INVARIANCE_ALL_G:.1%} across all G, so both "
        "sit above the MDE and can decide.",
        f"    smallest ALPHA shift callable    {alpha_mde:.3f}   (that ratio x "
        f"(alpha + Act1/W) at alpha={POOLED_ALPHA}, BLOCK_M={block_m})",
        "      A LOWER BOUND: it propagates the anchor's noise only, and the "
        "branch slope carries its own.",
        f"    C5's <= {C5_SHIFT_PRIOR} is a PRIOR, not derived from this noise: "
        f"it is {C5_SHIFT_PRIOR / alpha_mde:.1f}x the MDE and is the precision a",
        "      published alpha is QUOTED at. Read it as a claim about reporting, "
        "not about resolution.",
    ]


def render_plan(plan: MeasurePlan, out_dir: Path,
                spread_rel: float = CELL_SPREAD_REL) -> list[str]:
    cells = plan.cells
    cfg = MODEL_CONFIGS[plan.model]
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
        f"  instrument           {timing_basis() or 'NOT NAMEABLE HERE (no torch)'}",
        f"  timing               {plan.warmup_ms:.0f} ms warmup, {plan.trials} "
        f"trials of {plan.cell_budget_ms:.0f} ms, L2 flush "
        f"{'on' if plan.l2_flush else 'OFF'}, seed {plan.seed}",
        "                       iters is NOT a knob: the instrument sizes it per "
        "cell from the budget and the",
        "                       warmup's own queue-deep per-call time, and every "
        "cell records what it used.",
        "  stream check         one cell's worth of the same instrument over the "
        "real w1/w2 buffers",
        "  pin assay            per-cell tile read-back + fresh Triton artefacts "
        "per (BLOCK_M, GROUP_SIZE_M)",
        "",
        f"  cells                {len(cells)} "
        f"({len(plan.block_sizes)} BLOCK_M x {len(plan.group_sizes)} G x "
        f"{1 + len(plan.slope_tiles)} treads)",
        f"  estimated wall time  {plan.estimated_seconds() / 60.0:.1f} min "
        f"(crude: {len(cells)} cells at "
        f"{(plan.warmup_ms + plan.trials * plan.cell_budget_ms) / 1e3:.1f} s "
        "plus one compile per setting)",
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
        "",
    ]
    lines += render_mde(cfg, plan.dtype, min(plan.block_sizes), spread_rel)
    return lines


# `find_override_config` lived here until 2026-09-02 and is GONE, not moved. It
# probed the same three module names as two other scripts, in a third order, and
# then entered the hook it found WITHOUT reading get_config() back -- so a
# context that was entered and did not take looked exactly like one that did.
# `moe.baselines._framework_config.forcing_tile_config` probes and verifies, and
# `run_measure` uses it. The duplication was defended here on the grounds that
# an arm should not change because someone refactored a helper; what actually
# happened is that the arm ran eight GPU minutes at a time with no assay at all
# while every sibling arm carried one.


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
    #: The same two extra columns `ScoredFit` carries, and for the same reason:
    #: a measured cap printed without the factor it used to be high by is a
    #: number a reader cannot compare with the published one.
    phi: float
    lin_over_ridge_at_lo: float
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
            phi=br.phi,
            lin_over_ridge_at_lo=br.lin_cap(dtype_bytes(model_dtype), br.lo) / cal.ridge,
            residuals=residual_profile(allpts, a_free, b_free)))
    return fits, refusals


#: The `tile_config_source` a cell measured inside `override_config` has to
#: carry. The string duplicates `moe.bench.force_tile.TILE_SOURCE_OVERRIDE`,
#: which cannot be imported at module scope here without pulling torch in
#: through `moe.quant` and breaking the laptop `--rescore`/`--self-test` path.
#: The two are pinned to each other by
#: `tests/test_memory_branch_anchor.py::test_the_pin_source_string_has_not_drifted`,
#: so the duplicate cannot drift in silence.
PIN_SOURCE_OVERRIDE = "vllm_override"

#: The six constants a cell asks `override_config` for, and the six it has to be
#: SHOWN to have run. Not four: BLOCK_SIZE_K, num_warps and num_stages are
#: pinned too, and a vLLM that honoured the two this arm sweeps while silently
#: substituting the other four would still be running a kernel nobody asked for.
PIN_KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
            "num_warps", "num_stages")


def pin_disagreement(row: dict) -> str:
    """"" when this cell showed the tile it asked for, else what went wrong.

    Four distinguishable failures, and they are kept apart because they have
    different fixes: no assay was recorded at all (a cell from before the assay
    existed, or a synthetic one); the recorder saw nothing (vLLM memoised the
    lookup, or the call never reached it); the source was not the override (the
    hook stored the config and the kernel took its own); and a constant came
    back different from the one requested (the override was honoured in part).
    A cell that cannot produce the evidence is not a smaller failure than one
    that produces contradicting evidence: both leave a row that CLAIMS a tile
    it cannot show, which is worse than an unpinned row, because an unpinned row
    at least records vLLM's own choice honestly.
    """
    pin = row.get("pin")
    if not isinstance(pin, dict):
        return "no pin assay recorded for this cell"
    requested, observed = pin.get("requested") or {}, pin.get("observed") or {}
    if not requested:
        return "the cell recorded no requested tile, so nothing can be compared"
    for key, column in (("BLOCK_SIZE_M", "block_m"), ("GROUP_SIZE_M", "group_m")):
        if column in row and requested.get(key) != row.get(column):
            return (f"the row is labelled {column}={row.get(column)} and its "
                    f"requested tile says {key}={requested.get(key)}")
    if not observed:
        return ("nothing was read back out of vLLM during this cell, so which "
                "tile ran is unrecorded")
    source = pin.get("source")
    if source != PIN_SOURCE_OVERRIDE:
        return (f"the config vLLM handed the kernel came from {source!r}, not "
                f"{PIN_SOURCE_OVERRIDE!r}: the override was entered and the "
                "kernel took its own tile")
    wrong = [f"{k}: asked {requested.get(k)}, ran {observed.get(k)}"
             for k in PIN_KEYS if observed.get(k) != requested.get(k)]
    return "; ".join(wrong)


def gate_m6_pin(cells) -> Gate:
    """VALIDITY. Did the tile pin reach the kernel, at every cell.

    THE GATE THAT DECIDES WHETHER THE OTHER SIX MEAN ANYTHING, and the one this
    arm ran eight GPU minutes at a time without. Its three legs are described in
    the module docstring; here is why the verdict is one gate and not three.
    A cell needs all of them to be quotable -- an echoed-back override with no
    fresh kernel, or a fresh kernel whose config was never read back, each
    leaves the same question open -- so a single verdict carrying both counts in
    `measured` is the honest shape, and the failing side is named in the lines.
    The first leg cannot reach this gate as a verdict at all: a cell whose hook
    did not take is never timed, so it arrives as a `failed` row and gate M5
    counts it.

    THE COMPILE COUNT IS PER SETTING, NOT PER CELL. One (BLOCK_SIZE_M,
    GROUP_SIZE_M) is one Triton specialisation: its first cell compiles and its
    other fifteen legitimately do not. So the count is taken as the maximum over
    a setting's cells and the gate asks for at least one artefact per setting.

    A RESUMED CELL CARRIES ITS OWN ASSAY. The evidence is written into
    `cells.json` when the cell is measured, so a run that resumes every cell
    inherits it rather than going UNKNOWN for want of an experiment it cannot
    repeat. That is the difference from `block_m_crossing_sweep.gate_0_override`,
    which keeps its counts in memory and must go UNDECIDED on a full resume.
    The cells a resumed session RE-measures are the other half of that, and they
    need the opposite treatment: an inherited count is evidence, an inherited
    Triton cache is not, because a warm one makes a legitimate compile
    invisible and reads out here as "compiled nothing new". `session_cache_root`
    is why the count of a re-measured cell still means something.
    """
    ok = [c for c in cells if c.get("status") == "ok"]
    if not ok:
        return Gate("M6", VALIDITY, "the forced tile is the tile that ran", FAIL,
                    "no cell measured, so no cell was assayed", "every ok cell "
                    "shows its requested tile, and every setting compiled",
                    "every gate below, which would be scoring an empty set")
    bad = [(c, why) for c in ok if (why := pin_disagreement(c))]
    settings: dict[tuple[int, int], int] = {}
    for c in ok:
        key = (int(c.get("block_m", 0)), int(c.get("group_m", 0)))
        fresh = ((c.get("pin") or {}).get("fresh_artefacts")
                 if isinstance(c.get("pin"), dict) else None)
        settings[key] = max(settings.get(key, 0), int(fresh or 0))
    cold = sorted(k for k, n in settings.items() if n < 1)
    lines = [f"  BM={c.get('block_m')} G={c.get('group_m')} n={c.get('tiles')}: {why}"
             for c, why in bad[:10]]
    if cold:
        lines.append("  compiled nothing new: "
                     + ", ".join(f"BM={bm} G={g}" for bm, g in cold))
    return Gate(
        "M6", VALIDITY, "the forced tile is the tile that ran",
        PASS if not bad and not cold else FAIL,
        f"{len(ok) - len(bad)} of {len(ok)} cells showed their requested tile; "
        f"{len(settings) - len(cold)} of {len(settings)} settings compiled at "
        "least one fresh Triton artefact",
        "every ok cell shows its requested tile, and every setting compiled",
        "EVERY M GATE. A run in which the override never reached the kernel is "
        "one kernel measured at every setting, so M1's 'the anchor does not "
        "depend on the swizzle' passes by construction and the report reads as "
        "a tidy null result. That world was executed against this file's scorer "
        "on 2026-09-02: M0-M5 all returned PASS.", lines)


def gate_m0_stream(stream: dict | None, cal: Calibration) -> Gate:
    """Is the committed ceiling actually above what these buffers achieve."""
    if not stream:
        return Gate("M0", VALIDITY, "the committed ceiling is above the measured "
                    "read rate on the real weight buffers", FAIL,
                    "the stream check did not run", "measured <= ceiling",
                    "the bracket's upper end, which would then rest on a ceiling "
                    "this run never checked")
    ratio = stream["gbps"] / cal.ceiling_gbps
    return Gate(
        "M0", VALIDITY, "the committed ceiling is above the measured read rate "
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
        "M1", CLAIM, "P1: the anchor t(1) does not depend on the swizzle",
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
        return Gate("M2", CLAIM, f"P2: the BLOCK_M={block_m} anchor rate is in band",
                    FAIL, f"no BLOCK_M={block_m} cells measured",
                    f"{ANCHOR_RATE_BAND[0]:.0%}-{ANCHOR_RATE_BAND[1]:.0%} of pin",
                    "the carry-across to the committed arms, which is scored at "
                    f"BLOCK_M={block_m}")
    lo = min(f.anchor_rate_of_pin for f in sel)
    hi = max(f.anchor_rate_of_pin for f in sel)
    ok = ANCHOR_RATE_BAND[0] <= lo and hi <= ANCHOR_RATE_BAND[1]
    return Gate(
        "M2", CLAIM, f"P2: the BLOCK_M={block_m} anchor rate is in band",
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
        return Gate("M3", CLAIM, "P3: the slope does not depend on the anchor",
                    FAIL, "no cells", f"<= {SLOPE_INDEPENDENCE_REL:.1%}",
                    "the independence of the bracket's two ends")
    worst = max(fits, key=lambda f: f.slope_shift)
    ok = worst.slope_shift <= SLOPE_INDEPENDENCE_REL
    return Gate(
        "M3", CLAIM, "P3: the slope does not depend on the anchor",
        PASS if ok else FAIL,
        f"worst {worst.slope_shift:.2%} at BM={worst.block_m} G={worst.group_m} "
        f"over {worst.treads} treads",
        f"<= {SLOPE_INDEPENDENCE_REL:.1%} (calibrated on 16- and 33-tread ladders)",
        "B / t(1) as a BOUND. If the slope moves when the anchor leaves the fit, "
        "the low end of the bracket is partly a restatement of the anchor")


def gate_m4_bracket_order(fits, cal: Calibration) -> Gate:
    """P4, scored."""
    if not fits:
        return Gate("M4", VALIDITY, "P4: the bracket is ordered", FAIL, "no cells",
                    "anchor rate <= ceiling", "every interval below")
    tight = max(fits, key=lambda f: f.bw_anchor_gbps)
    ratio = tight.bw_anchor_gbps / cal.ceiling_gbps
    ok = ratio <= 1.0 and all(f.alpha_lo <= f.alpha_hi + 1e-12 for f in fits)
    return Gate(
        "M4", VALIDITY, "P4: the anchor rate never exceeds the measured ceiling",
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
        "M5", VALIDITY, "every planned cell measured",
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
           f"{'cap/ridge':>9s} {'lin/ridge':>9s}",
           "  " + "-" * 92]
    for f in sorted(fits, key=lambda f: (f.block_m, f.group_m)):
        out.append(f"  {f.block_m:4d} {f.group_m:4d} {f.anchor_ms:9.4f} "
                   f"{f.slope_no_anchor:8.4f} {f.slope_shift:6.2%} "
                   f"{f.bw_anchor_gbps:7.0f} {f.anchor_rate_of_pin:6.1%} "
                   f"{f.alpha_lo:8.3f} {f.alpha_hi:8.3f} "
                   f"{f.cap_over_ridge_at_lo:9.3f} "
                   f"{f.lin_over_ridge_at_lo:9.3f}")
    return out


def score_measured(cells, cfg, dtype: str, block_n: int, cal: Calibration,
                   stream: dict | None, planned: int
                   ) -> tuple[list[MeasuredFit], list[Refusal], list[Gate], list[str]]:
    """The GPU arm's whole verdict path, with no device in it."""
    fits, refusals = fits_from_cells(cells, cfg, dtype, block_n, cal)
    gates = [
        # M6 FIRST, whatever its number: it is the gate that decides whether the
        # rest mean anything, and a reader who stops at the first FAIL has to
        # meet it before meeting a null result it would explain.
        gate_m6_pin(cells),
        gate_m5_completeness(cells, planned),
        gate_m0_stream(stream, cal),
        gate_m4_bracket_order(fits, cal),
        gate_m1_anchor_invariance(fits),
        gate_m2_anchor_rate(fits, cal),
        gate_m3_slope_independence(fits),
    ]
    lines = render_measured_table(fits)
    return fits, refusals, gates, lines


# --------------------------------------------------------------------------
# Planted worlds. The whole measured verdict path, off-GPU, with the FAIL branch
# of every M gate planted -- including the world the 2026-09-02 audit executed
# against this file's scorer and got exit 0 out of.
# --------------------------------------------------------------------------

#: The card the planted worlds run on: the A100's own committed calibration,
#: transcribed so a self-test does not depend on a yaml being present. The pin
#: rate and the write-pattern ceiling are what M2 and M4 are scored against, so
#: they are the numbers that decide which plant lands inside a band and which
#: outside; changing them changes what the worlds mean.
SELF_TEST_CALIBRATION = Calibration(
    slug="nvidia_a100_sxm4_80gb", name="A100-SXM4-80GB (planted)",
    checked_on="2026-09-02", measured_commit="planted",
    patterns={"triad": 1799.4, "write": 1879.1}, ceiling_pattern="write",
    ceiling_gbps=1879.1, pin_gbps=2039.0,
    dense_tflops={"bf16": 262.3712016979615}, ridge=145.81,
    reference_clock_mhz=1410.0,
    reference_clock_source="planted: the A100 calibration's GEMM clock")


def plant_ladder(alpha: float, *, bw_anchor_gbps: float, bw_branch_gbps: float,
                 fixed_ms: float, cfg, dtype: str = "bf16", block_m: int = 32,
                 treads: int = 16) -> list[tuple[int, float]]:
    """`t(n)` from a stated alpha, with the ANCHOR allowed its own bandwidth.

    The anchor running slower than the branch is the pathology in the committed
    data -- the measured n=1 tread stands above the fitted line in 12 of 12 A100
    fits -- so the plant has to be able to express it, or every world would be
    one in which nothing is wrong. `treads` counts the anchor: `treads=16` is
    n = 1..16, the anchor plus the fifteen branch treads the CLI defaults to.
    """
    w, act1 = anchor_bytes(cfg, dtype, block_m)
    out = []
    for n in range(1, treads + 1):
        bw = bw_anchor_gbps if n == 1 else bw_branch_gbps
        bytes_n = w * (1.0 + alpha * (n - 1)) + act1 * n
        out.append((n, fixed_ms + bytes_n / (bw * 1e9) * 1e3))
    return out


#: How a planted cell records its pin assay. The four modes are the four
#: distinguishable states gate M6 exists to separate, and every one of them
#: except "ok" was indistinguishable from "ok" before that gate existed.
PIN_MODES = ("ok", "default_tile", "missing", "warm_cache")


def plant_pin(mode: str, requested: dict) -> dict | None:
    """The `pin` record a planted cell carries, or None for "no record at all".

    `default_tile` is vLLM's hardcoded fallback ladder at these row counts
    (BLOCK_SIZE_M 64, GROUP_SIZE_M 1, BLOCK_SIZE_N 64) reached through the tuned
    lookup: the shape a cell has when `override_config` was entered and the
    kernel took its own tile anyway. `warm_cache` is the other half of the
    assay: the right tile, read back correctly, and no kernel built for it,
    which is either a cache serving a previous run or an override that changed
    nothing.
    """
    if mode == "missing":
        return None
    if mode == "default_tile":
        observed = dict(requested)
        observed.update({"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1})
        return {"requested": requested, "observed": observed,
                "source": "vllm_default", "hook": "planted", "fresh_artefacts": 0}
    return {"requested": requested, "observed": dict(requested),
            "source": PIN_SOURCE_OVERRIDE, "hook": "planted",
            "fresh_artefacts": 0 if mode == "warm_cache" else 17}


def plant_cells(cfg, *, alpha: float = POOLED_ALPHA,
                anchor_bw_by_g: dict[int, float] | None = None,
                bw_branch_gbps: float = 1750.0, fixed_ms: float = 0.05,
                block_m: int = 32, block_n: int = 64, block_k: int = 64,
                num_warps: int = 8, num_stages: int = 3, treads: int = 16,
                pin: str = "ok") -> list[dict]:
    """A whole `--measure` grid as `cells.json` rows, with its pin assay.

    `anchor_bw_by_g` is the lever every M gate's world is built with: one
    bandwidth per GROUP_SIZE_M for the n=1 tread. Equal values across G are what
    a HEALTHY run looks like (the anchor does not depend on the swizzle, which
    is P1) AND what a run whose pin never reached the kernel looks like (one
    kernel measured four times). The two are separated by the pin record and by
    nothing else in the file, which is the whole reason gate M6 exists.
    """
    if pin not in PIN_MODES:
        raise ValueError(f"pin mode {pin!r} is not one of {PIN_MODES}")
    anchor_bw_by_g = anchor_bw_by_g or {1: 1450.0, 8: 1450.0, 16: 1450.0,
                                        64: 1450.0}
    rows: list[dict] = []
    for g, bw_anchor in sorted(anchor_bw_by_g.items()):
        points = plant_ladder(alpha, bw_anchor_gbps=bw_anchor,
                              bw_branch_gbps=bw_branch_gbps, fixed_ms=fixed_ms,
                              cfg=cfg, block_m=block_m, treads=treads)
        requested = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
                     "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": g,
                     "num_warps": num_warps, "num_stages": num_stages}
        for n, ms in points:
            row = {"block_m": block_m, "group_m": g, "tiles": n,
                   "rows_per_expert": block_m * n, "tokens": 0,
                   "ms_p50": ms, "ms_p90": ms, "ms_min": ms, "ms_stdev": 0.0,
                   "instrument": "planted: no kernel ran", "warmup_ms": 0.0,
                   "iters": 0, "trials": 0, "l2_flush": True,
                   "sm_clock_load_mhz": None, "clock_level_ok": None,
                   "clock_level_side": "",
                   "clock_drift_ok": None, "host_bound": None,
                   "host_enqueue_ms": None, "clock_note": "", "host_note": "",
                   "status": "ok", "detail": ""}
            record = plant_pin(pin, requested)
            if record is not None:
                row["pin"] = record
            rows.append(row)
    return rows


@dataclass(frozen=True)
class PlantedWorld:
    """One world, its registered exit code, and what it is for.

    `must_fail` and `must_pass` are named gates, not counts: a world that
    reached the right exit code through the wrong gate has not exercised the
    branch it claims to. `must_pass` is what makes the `pin_failed` world an
    argument rather than an assertion -- it requires M0 through M5 to PASS while
    M6 fails, which is the state the audit produced and called exit 0.
    """

    name: str
    what: str
    expect: int
    must_fail: tuple[str, ...]
    must_pass: tuple[str, ...] = ()
    cells: dict = field(default_factory=dict)
    stream_gbps: float | None = 1500.0
    planned_delta: int = 0


ALL_M_GATES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")

SELF_TEST_WORLDS: tuple[PlantedWorld, ...] = (
    PlantedWorld(
        "clean", "a run that obeys the model at every swizzle, correctly pinned",
        exit_codes.DONE, (), ALL_M_GATES),
    PlantedWorld(
        "pin_failed",
        "THE AUDIT'S WORLD: one kernel measured at every GROUP_SIZE_M because "
        "the override never reached it. Numerically identical to `clean`; the "
        "only difference on the page is the tile vLLM handed the kernel",
        exit_codes.INVALID, ("M6",), tuple(g for g in ALL_M_GATES if g != "M6"),
        cells={"pin": "default_tile"}),
    PlantedWorld(
        "unassayed",
        "the 128 synthetic cells the refuter fed this scorer: no pin record at "
        "all. A cell that cannot show its tile is not a smaller failure than "
        "one that shows the wrong tile",
        exit_codes.INVALID, ("M6",), tuple(g for g in ALL_M_GATES if g != "M6"),
        cells={"pin": "missing"}),
    PlantedWorld(
        "warm_cache",
        "the right tile read back, and no Triton artefact built for it: a cache "
        "serving a previous run, or an override that changed no constant",
        exit_codes.INVALID, ("M6",), tuple(g for g in ALL_M_GATES if g != "M6"),
        cells={"pin": "warm_cache"}),
    PlantedWorld(
        "swizzle_dependent_anchor",
        "P1 refuted: t(1) is 45% slower at G=16 than at G=1, so it is a "
        "condition-specific number and not a condition-free bound on L",
        exit_codes.CLAIM_FAIL, ("M1",), ("M0", "M4", "M5", "M6"),
        cells={"anchor_bw_by_g": {1: 1450.0, 16: 1000.0}}),
    PlantedWorld(
        "anchor_out_of_band",
        "P2 refuted: the anchor rate is 59% of pin, below the 64-78% band the "
        "committed arms measured, so this is not the same physical event and "
        "the brackets may not be carried across to them",
        exit_codes.CLAIM_FAIL, ("M2",), ("M0", "M4", "M5", "M6"),
        cells={"anchor_bw_by_g": {1: 1200.0, 16: 1200.0}}),
    PlantedWorld(
        "slope_depends_on_anchor",
        "P3 refuted: an anchor far off the branch moves the fitted slope by "
        "more than 1.5% when it is dropped, so B / t(1) is partly a restatement "
        "of t(1) and not a bound on it",
        exit_codes.CLAIM_FAIL, ("M3",), ("M0", "M4", "M5", "M6"),
        cells={"anchor_bw_by_g": {1: 700.0, 16: 700.0}}),
    PlantedWorld(
        "anchor_above_ceiling",
        "P4 refuted: the n=1 tread moves bytes faster than the card's largest "
        "demonstrated pattern, so the ceiling is not one and the bracket inverts",
        exit_codes.INVALID, ("M4",), ("M5", "M6"),
        cells={"anchor_bw_by_g": {1: 2000.0, 16: 2000.0}}),
    PlantedWorld(
        "ceiling_below_the_data",
        "the committed ceiling sits BELOW the read rate these very buffers "
        "achieved, so every alpha_hi computed from it is too low",
        exit_codes.INVALID, ("M0",), ("M1", "M4", "M5", "M6"),
        stream_gbps=1900.0),
    PlantedWorld(
        "cells_missing",
        "the grid the plan printed is not the grid that ran: a bracket built on "
        "a different set of cells from the one the report describes",
        exit_codes.INVALID, ("M5",), ("M0", "M1", "M4", "M6"),
        planned_delta=1),
)


def self_test(verbose: bool = True) -> int:
    """Score every planted world and compare with what this file registered.

    Returns DONE when every world agrees and INVALID when one does not: a
    scorer that misreads a world whose answer is known cannot be trusted with a
    world whose answer is not, and that is an instrument failure rather than a
    refuted claim.

    NO `RESULT:` LINES ARE PRINTED HERE. A planted world's verdict is not a
    result about this machine, and a driver that grepped one out of a self-test
    log would be reading a plant as a measurement -- the same defect, from the
    other side, as the refused arm whose log matched a summary regex 18 times
    and printed an imported constant under the heading "floor".

    The proof that this function can return non-zero is not in here: it is in
    `tests/test_memory_branch_anchor.py::test_the_self_test_fails_when_a_world_is_mis_registered`,
    which plants a wrong expectation and asserts the INVALID.
    """
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    bad = 0
    for world in SELF_TEST_WORLDS:
        cells = plant_cells(cfg, **world.cells)
        stream = {"gbps": world.stream_gbps} if world.stream_gbps else None
        planned = len(cells) + world.planned_delta
        _, _, gates, _ = score_measured(cells, cfg, "bf16", 64,
                                        SELF_TEST_CALIBRATION, stream, planned)
        rc = exit_codes.classify(g.scored() for g in gates)
        verdicts = {g.number: g.verdict for g in gates}
        wrong_fail = [n for n in world.must_fail if verdicts[n] == PASS]
        wrong_pass = [n for n in world.must_pass if verdicts[n] != PASS]
        ok = rc == world.expect and not wrong_fail and not wrong_pass
        bad += not ok
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {world.name:26s} "
                  f"-> {exit_codes.CODE_NAMES[rc]:10s} "
                  f"(registered {exit_codes.CODE_NAMES[world.expect]})")
            print(f"           {world.what}")
            print("           " + "  ".join(f"{n}:{verdicts[n]}"
                                            for n in sorted(verdicts)))
            if wrong_fail:
                print(f"           EXPECTED TO FAIL AND PASSED: {wrong_fail}")
            if wrong_pass:
                print(f"           EXPECTED TO PASS AND DID NOT: {wrong_pass}")
    if verbose:
        print()
        print(f"{len(SELF_TEST_WORLDS) - bad} of {len(SELF_TEST_WORLDS)} planted "
              "worlds scored as registered.")
        print("Nothing was measured: every number above is planted, and no "
              "RESULT line was printed for that reason.")
    return exit_codes.DONE if bad == 0 else exit_codes.INVALID


def clock_side_of(t) -> str:
    """Which side of the LEVEL band the instrument saw a cell on.

    `timing.LEVEL_LOW`, `timing.LEVEL_HIGH`, or "" for level or undetermined.
    The record's own `clock_level_side` is preferred. A record that failed
    LEVEL and carries no side (a fake built before the field existed on
    2026-09-03) has it derived from its own load and reference, the rule
    `moe.bench.driver` applies to the same records; one with neither answers
    "", which `clock_excluded` reads as the one-sided era's False, below.
    """
    from moe.bench import timing

    side = getattr(t, "clock_level_side", "") or ""
    if not side and getattr(t, "clock_level_ok", None) is False:
        side = timing.level_side(getattr(t, "sm_clock_load_mhz", None),
                                 getattr(t, "reference_clock_mhz", None)) or ""
    return side


def clock_excluded(level_ok: bool | None, side: str,
                   drift_ok: bool | None) -> bool:
    """Do a cell's clock verdicts exclude it. LOW or DRIFT do; HIGH does not.

    THE FIFTEENTH INSTANCE OF A FIX LANDING AT ONE OF TWO CALL SITES. Commit
    03df2d4 made `timing.clock_flags` two-sided at the producer, so a cell
    boosted to 1980 MHz against the 1515 MHz bf16-GEMM reference now fails
    LEVEL with `clock_level_side == "high"`. Until 2026-09-08 this file read
    `clock_level_ok is False` alone, the one-sided era's test, which takes
    that cell for one that ran cold. On the H200 the HIGH side is the NORMAL
    state of a memory-shaped cell: the committed calibration holds 1980 MHz
    under memory load for 30 s against a 1515 MHz GEMM plateau, so the old
    test flagged exactly the cells the memory branch is made of.

    HIGH means the fixed-roof fraction is not comparable and the per-row
    `roof_at_cell_clock` is the number to read. The time itself is a time at
    one clock and stays. This is the `throttled` rule `moe.bench.driver`
    writes on its own rows, restated because the rows this file writes carry
    the verdicts and not that column. A False with no side is the one-sided
    era's meaning, below, and stays excluded. None is not determined, and an
    exclusion has to be positively established.
    """
    from moe.bench import timing

    if drift_ok is False:
        return True
    return level_ok is False and side != timing.LEVEL_HIGH


def clock_state(rows: list[dict]) -> dict:
    """How many timed cells sat where against the roof's clock, in one block.

    Counts and never a verdict: this arm drops no cell for its clock (the
    scorer reads `status == "ok"`), so the block is what a reader of the
    payload has to decide whether the ladders were timed at the clock the
    roof was. `low` and `drift` are the excluded-shaped states
    `clock_excluded` names; `high` is kept and counted apart from them,
    because on the H200 it is the ordinary state of a memory-bound tread and
    this arm's every tread past the anchor is one. `unknown` is the cells
    whose LEVEL was not determined, separate because a run that could not
    read its clocks and a run whose clocks were fine are not the same state.
    """
    from moe.bench import timing

    timed = [r for r in rows if r.get("status") == "ok"]
    return {
        "timed": len(timed),
        "level": sum(1 for r in timed if r.get("clock_level_ok") is True),
        "low": sum(1 for r in timed if r.get("clock_level_ok") is False
                   and r.get("clock_level_side") != timing.LEVEL_HIGH),
        "high": sum(1 for r in timed if r.get("clock_level_ok") is False
                    and r.get("clock_level_side") == timing.LEVEL_HIGH),
        "drift": sum(1 for r in timed if r.get("clock_drift_ok") is False),
        "unknown": sum(1 for r in timed if r.get("clock_level_ok") is None),
        "excluded_shaped": sum(
            1 for r in timed
            if clock_excluded(r.get("clock_level_ok"),
                              r.get("clock_level_side") or "",
                              r.get("clock_drift_ok"))),
        "rule": "LOW or DRIFT excludes; HIGH is kept, its fixed-roof fraction "
                "is not comparable and roof_at_cell_clock is the number to "
                "read; this arm drops no cell for its clock",
    }


def clock_state_lines(state: dict) -> list[str]:
    """The printed form of `clock_state`, saying which side each count is."""
    return [
        f"clock state: {state['timed']} timed cells: {state['level']} level, "
        f"{state['low']} LOW (below the band, excluded-shaped), "
        f"{state['high']} HIGH (boosted above the band, kept: the fixed-roof "
        "fraction is not comparable, read roof_at_cell_clock), "
        f"{state['drift']} DRIFT failed (excluded-shaped), "
        f"{state['unknown']} with LEVEL not determined",
        "  this arm drops no cell for its clock; the counts are for a reader "
        "deciding whether to believe it, and None means NOT DETERMINED, never "
        "fine",
    ]


def timing_columns(t) -> dict:
    """The `KernelTiming` columns every timed row of this arm carries.

    ALL of them, including the three verdicts and the side. A row that
    recorded a time and not the clock it was taken at cannot be compared with
    the roof, and this study has 13,460 driver rows that drift 100 MHz inside
    one cell and nine session scripts that record no clock at all.
    `clock_level_ok` and `clock_drift_ok` are Optional and None means "not
    determined", never "fine"; `clock_note` says which. `clock_level_side`
    says which way a LEVEL failure went, because since 03df2d4 a failure can
    be a boost above the reference and on the H200 every memory-bound tread
    of this arm is one; `clock_excluded` is the reader that tells the two
    apart.
    """
    return {
        "ms_p50": t.ms_p50, "ms_p90": t.ms_p90, "ms_min": t.ms_min,
        "ms_stdev": t.ms_std,
        "instrument": t.instrument, "warmup_ms": t.warmup_ms, "iters": t.iters,
        "trials": t.trials, "l2_flush": t.l2_flush,
        "sm_clock_load_mhz": t.sm_clock_load_mhz,
        "clock_level_ok": t.clock_level_ok, "clock_drift_ok": t.clock_drift_ok,
        "clock_level_side": clock_side_of(t),
        "host_bound": t.host_bound, "host_enqueue_ms": t.host_enqueue_ms,
        "clock_note": t.clock_note, "host_note": t.host_note,
    }


def arm_triton_cache(root: Path, block_m: int, group_m: int) -> Path:
    """Point Triton at a fresh directory for THIS setting, before it compiles.

    Set before the first compile of the setting, because Triton reads the
    variable at compile time. Within one process each (BLOCK_SIZE_M,
    GROUP_SIZE_M) is a distinct specialisation and so a distinct cache entry
    anyway; the per-setting directory is what makes "did this setting compile
    anything" a countable question instead of an assumption.

    Six lines transcribed from `scripts/block_m_crossing_sweep.py` rather than
    imported, for the reason `find_override_config` used to give: importing one
    script from another makes an arm's measurement depend on an unrelated file's
    refactor. The keyed setting is the pair, not the tile alone, because this
    arm sweeps the swizzle and the swizzle is the constant its headline gate is
    about.
    """
    directory = root / f"bm{block_m}-g{group_m}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(directory)
    return directory


def session_cache_root(out_dir: Path) -> Path:
    """An EMPTY Triton cache directory belonging to this session alone.

    The artefact count is only evidence if the cache it counts into started
    empty. A single `out_dir/triton-cache` is not that: `out_dir` is the resume
    directory, so a second session finds the first session's compiled artefacts
    already on disk, the per-setting baseline absorbs them, every re-measured
    cell loads from the warm cache and records `fresh_artefacts = 0`, and gate
    M6 fails "compiled nothing new" on a run that is otherwise sound. That is
    the shape of the M0-forever-fail-on-resume defect the persisted stream check
    was written to remove -- the stream check was carried across sessions, the
    artefact count was not, and the count cannot be carried because it is a
    statement about a directory rather than about a cell. So the directory
    moves instead, and each session compiles once per setting into its own.

    `mkdtemp` rather than a timestamp: two calls in one second in one process
    would collide on a stamp, and the whole point of the directory is that
    nothing has written into it. Earlier sessions' directories are LEFT ALONE.
    They are that session's evidence, and a script handed `--out-dir` by a
    human has no business deleting what it finds there.
    """
    root = out_dir / "triton-cache"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="session-", dir=root))


def count_new(root: Path, seen: set[Path]) -> int:
    """How many files have appeared under `root` since the last call."""
    fresh = [p for p in root.rglob("*") if p.is_file() and p not in seen]
    seen.update(fresh)
    return len(fresh)


def observed_pin(capture) -> tuple[dict, str]:
    """What vLLM handed the kernel this cell, and where that config came from.

    `capture` is a `TileCapture` filled by `recording_tile_config` around ONE
    call. The first recorded call is the one that describes the time (vLLM
    re-derives the config only per chunk above VLLM_FUSED_MOE_CHUNK_SIZE, and
    the first chunk is the full-size one).

    THE SOURCE IS DERIVED FROM WHAT THE LOOKUP DID, not asserted from the fact
    that a context was entered. `tile_meta_from_capture(override_active=True)`
    labels any capture "vllm_override" because its caller said so, which is the
    tautology this gate exists to avoid. vLLM's `try_get_optimal_moe_config`
    consults `get_config()` first and returns the override WITHOUT reaching
    `get_moe_configs`; so a call that skipped the tuned-file lookup took the
    override, and a call that ran the lookup went to vLLM's own tuned file or
    its hardcoded fallback ladder, which is the override failing.

    KNOWN LIMIT, and it is WIDER THAN "the recorder found nothing to wrap".
    `lookup_observed` is False in three worlds, not two, and this function can
    only see that it is False:

      * the override took, and the lookup was skipped because it was;
      * vLLM exposed no `get_moe_configs` binding, so nothing watched the
        lookup that did run;
      * vLLM memoised `try_get_optimal_moe_config` itself, so the observation
        call hit that cache and never re-entered the lookup. That is the
        degradation `recording_tile_config` names in its own docstring, and it
        returns real tile ints with no observation behind them.

    The last two are labelled `vllm_override` here. That is deliberate and it is
    the opposite of what `tile_meta_from_capture` does with the same input,
    where an unobserved lookup writes "unrecorded": that function labels rows
    for a tile-source CSV in which nobody asserted a tile, so the conservative
    answer is to name no source. Here the caller HAS entered an override and the
    only question is whether it took; writing "unrecorded" would make the source
    leg unable to return PASS in the healthy world, which is not conservatism,
    it is deleting the leg.

    WHAT BOUNDS THE RESIDUAL is that the label alone passes nothing. Gate M6
    needs `pin_disagreement` to find the six PIN_KEYS read back EQUAL to the six
    requested, and it needs the setting to have compiled a fresh Triton
    artefact; neither depends on the recorder's reach. In both degraded worlds
    the config in hand is vLLM's own choice, so the six constants agree with the
    six requested only where vLLM would have chosen the pinned tile anyway --
    a cell that measures the requested kernel either way.
    """
    if not capture.calls:
        return {}, "unrecorded"
    call = capture.calls[0]
    conf = {k: v for k, v in (call.config or {}).items() if k in PIN_KEYS}
    if call.lookup_observed:
        return conf, ("vllm_default" if call.tuned_keys is None else "vllm_tuned")
    return conf, PIN_SOURCE_OVERRIDE


def stream_check(weights, plan: MeasurePlan, reference_clock_mhz: float | None) -> dict:
    """Read the real weight buffers and report GB/s.

    A REDUCTION along the contiguous axis, which is what
    `moe/bench/calibrate.py` documents as the read pattern: one pass, every byte
    touched once, and the per-row outputs keep the global combine out of the
    number. This is a LOWER bound on the machine's read rate by construction,
    which is the direction that makes it a valid CHECK on the ceiling: if this
    lower bound exceeds the committed ceiling, the ceiling is wrong.

    Timed by the same `time_kernel` and at the same warmup, budget, trial count
    and flush setting as the cells, because a ceiling checked with a different
    instrument from the data it bounds is not a check on that data.

    No custom kernel. The probe kernel in `moe/bench/read_probe.py` would be a
    tighter instrument and it belongs to another workflow; this arm only needs
    to know whether the ceiling is above the floor.
    """
    from moe.bench import timing

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

    t = timing.time_kernel(once, warmup_ms=plan.warmup_ms,
                           target_ms=plan.cell_budget_ms, trials=plan.trials,
                           l2_flush=plan.l2_flush,
                           reference_clock_mhz=reference_clock_mhz)
    return {"bytes": total, **timing_columns(t),
            "gbps": total / (t.ms_p50 * 1e-3) / 1e9,
            "gbps_from_min": total / (t.ms_min * 1e-3) / 1e9}


class ResumeRefused(RuntimeError):
    """A `cells.json` this run may not resume into.

    Raised rather than started over: silently discarding a measured file is its
    own way to lose an arm, and the two cases here are both cases where the
    operator has to look at the directory before any more pod minutes are spent.
    """


def restore_cells(path: Path, card: str) -> tuple[list[dict], set, dict | None]:
    """`(rows, done, stream_check)` from a previous session's `cells.json`.

    Pure enough to test off-GPU, which is the point: every branch below is a
    refusal or a carry-forward that used to live inside `run_measure` and could
    therefore only be exercised on a rented pod.

    THE RESUME IS GUARDED ON THE CARD, belt as well as braces. The card is in
    the run id, so a second card lands in a different directory and cannot
    normally reach a foreign `cells.json` at all. This check is what catches the
    ways it could anyway: an explicit `--out-dir` pointing both runs at one
    place, a directory copied between pods, or a file written before the card
    entered the id. The legacy shape -- a bare list, with no record of which
    card wrote it -- is refused for the same reason: it is exactly the unknown
    the guard exists for, so it is not assumed to be ours.

    THE STREAM CHECK COMES BACK WITH THE CELLS. It is measured once, on the
    first freshly timed cell, so a run that resumes every cell measures none and
    used to hand gate M0 a None: VALIDITY FAIL, exit 2 under this file's old
    inverted table, REFUSED in the driver's ledger, and the arm could never
    reach DONE however many times it was resumed. The check belongs to the
    session that measured the cells, and it is stored with them.

    ONLY CELLS THAT SUCCEEDED COUNT AS DONE. A failure is retried, because the
    common ones here are a lost device and a shared-memory rejection, and a real
    failure fails again in milliseconds.
    """
    if not path.exists():
        return [], set(), None
    stored = json.loads(path.read_text())
    if isinstance(stored, dict):
        written_by = str(stored.get("card") or "")
        rows = list(stored.get("cells") or [])
        stream = stored.get("stream_check")
    else:
        written_by, rows, stream = "", list(stored), None
    if written_by != card:
        raise ResumeRefused(
            f"will not resume {path}: it was written by card "
            f"{written_by or '<unrecorded, pre-card-in-id>'!r} and this run is "
            f"{card!r}. Resuming would publish one card's timings under the "
            "other's calibration, which is the exact defect the card in the run "
            "id closes. Move or delete that file deliberately. Nothing measured.")
    done = {(int(r["block_m"]), int(r["group_m"]), int(r["tiles"]))
            for r in rows if r.get("status") == "ok"}
    return rows, done, stream


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
        return exit_codes.REFUSED
    plan = MeasurePlan(
        card=card, model=args.model, dtype=args.dtype,
        block_sizes=tuple(int(v) for v in args.tiles.split(",")),
        group_sizes=tuple(int(v) for v in args.group_m.split(",")),
        slope_tiles=tuple(int(v) for v in args.slope_tiles.split(",")),
        block_n=args.block_n, block_k=args.block_k, num_warps=args.num_warps,
        num_stages=args.num_stages, seed=args.seed, warmup_ms=args.warmup,
        cell_budget_ms=args.cell_budget_ms, trials=args.trials,
        l2_flush=not args.no_l2_flush)
    out_dir = args.out_dir / plan.run_id()

    for line in render_plan(plan, out_dir, args.noise):
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
        print("DRY RUN. Nothing was measured, nothing was written, no GPU was "
              f"used. {exit_codes.describe(exit_codes.REFUSED)}")
        return exit_codes.REFUSED

    try:
        import torch
    except ImportError:
        print("REFUSED: torch is not installed. Nothing measured.")
        return exit_codes.REFUSED
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device. The anchor is a measured time and there is "
              "nothing here to measure it on. Nothing measured.")
        return exit_codes.REFUSED

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
        return exit_codes.REFUSED
    slugs = available_calibrations()
    match = calibration_slug_for(slug, slugs) or (slug if slug in slugs else None)
    if match is None:
        print(f"REFUSED: no measured calibration for {gpu!r} (slug {slug!r}). "
              f"Known: {', '.join(slugs) or 'none'}. The bracket's upper end must "
              "be a measured ceiling for THIS card. Nothing measured.")
        return exit_codes.REFUSED
    cal = load_calibration(match)
    print(f"device: {gpu}")
    print(f"ceiling: {cal.describe()}")
    print("reference clock: "
          + (f"{cal.reference_clock_mhz:.0f} MHz, {cal.reference_clock_source}"
             if cal.reference_clock_mhz else
             f"NOT RESOLVED ({cal.reference_clock_source}); every cell's clock "
             "LEVEL verdict will be None and no cell can be excluded for it"))
    print()

    cfg = MODEL_CONFIGS[plan.model]
    out_dir.mkdir(parents=True, exist_ok=True)

    # BEFORE vLLM is imported. Triton may snapshot this variable at import in
    # some versions, and a warm cache compiles and dumps nothing -- the bug that
    # cost this project its A100 PTX dump. Pointing it at this SESSION's own
    # empty directory first makes the count fresh whatever the per-setting
    # redirect below manages, and the count is taken over the whole root so the
    # assay works either way. Per session, not per out_dir: see
    # `session_cache_root`, or a resumed run reports zero fresh artefacts for
    # every setting it re-measures and fails M6 for ever.
    cache_root = session_cache_root(out_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import (
        ForceTileNotHonoured,
        TileCapture,
        forcing_tile_config,
        recording_tile_config,
        vllm_call_kwargs,
    )
    from moe.bench import timing

    print(f"instrument: {timing.TIMING_BASIS}")
    print(f"triton cache: {cache_root} (this session's own, empty)")

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
    #
    # THE STREAM CHECK IS PART OF THE RESUMED STATE. It is measured once, on the
    # first freshly timed cell, so a run that resumes every cell measures none
    # and used to hand gate M0 a None -- VALIDITY FAIL, exit 2 under the old
    # table, REFUSED in the ledger, and the arm could never reach DONE however
    # many times it was resumed. The check belongs to the session that measured
    # the cells and is stored with them.
    try:
        rows, done, stream = restore_cells(cells_path, plan.card)
    except ResumeRefused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    if cells_path.exists():
        print(f"resuming: {len(done)} of {len(plan.cells)} cells already measured"
              + (f", stream check {stream['gbps']:.1f} GB/s carried forward"
                 if stream else ", no stream check stored"))
    inputs: dict[int, tuple] = {}
    seen_files: set[Path] = set()
    started = time.time()

    from moe.routing.distributions import realize_counts

    def write_cells() -> None:
        cells_path.write_text(json.dumps(
            {"card": plan.card, "run_id": plan.run_id(), "stream_check": stream,
             "cells": rows}, indent=1) + "\n")

    for bm in plan.block_sizes:
        for g in plan.group_sizes:
            arm_triton_cache(cache_root, bm, g)
            count_new(cache_root, seen_files)
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
                pin = {"requested": conf, "observed": {}, "source": "unrecorded",
                       "hook": "", "fresh_artefacts": 0}

                def call(_x=x, _w=weights, _ids=ids, _tw=w, _kw=kw):
                    return fused_experts(hidden_states=_x, w1=_w.w1, w2=_w.w2,
                                         topk_weights=_tw, topk_ids=_ids, **_kw)

                timed: dict = {}
                try:
                    # `forcing_tile_config`, not a bare `override_config`: it
                    # probes the hook the way this file used to and then READS
                    # get_config() BACK, refusing a context that was entered and
                    # did not take. That is the first of gate M6's three legs;
                    # the recorder below is the second (what vLLM handed the
                    # kernel) and the artefact count the third (whether a kernel
                    # was built for this setting at all).
                    with forcing_tile_config(conf) as hook:
                        pin["hook"] = hook
                        capture = TileCapture()
                        with recording_tile_config(capture):
                            call()
                        torch.cuda.synchronize()
                        pin["fresh_artefacts"] = count_new(cache_root, seen_files)
                        pin["observed"], pin["source"] = observed_pin(capture)
                        t = timing.time_kernel(
                            call, warmup_ms=plan.warmup_ms,
                            target_ms=plan.cell_budget_ms, trials=plan.trials,
                            l2_flush=plan.l2_flush,
                            reference_clock_mhz=cal.reference_clock_mhz)
                    timed = timing_columns(t)
                    status, detail = "ok", ""
                    # THE SIDE IS READ HERE, NOT ONLY THE VERDICT. On the H200
                    # `clock_level_ok is False` alone fires on every
                    # memory-shaped tread (1980 MHz against the 1515 MHz
                    # reference), which is every tread of this arm past the
                    # anchor; LOW, DRIFT and host-bound get the
                    # exclusion-shaped marker, HIGH is named as kept.
                    if (clock_excluded(t.clock_level_ok,
                                       timed["clock_level_side"],
                                       t.clock_drift_ok) or t.host_bound):
                        print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
                    elif timed["clock_level_side"] == timing.LEVEL_HIGH:
                        print(f"  ^ kept (LEVEL high is not an exclusion): "
                              f"{t.clock_note or ''}".rstrip())
                except ForceTileNotHonoured as exc:
                    # Named separately from the generic failure because it is
                    # the one this arm was blind to for its whole life: the pin
                    # did not reach the kernel, so the cell is not a slow cell
                    # or an OOM, it is a cell that would have measured the wrong
                    # kernel and reported it as the right one.
                    pin["source"] = "not_honoured"
                    status, detail = "failed", f"PIN NOT HONOURED: {exc}"
                except Exception as exc:                       # noqa: BLE001
                    status, detail = "failed", f"{type(exc).__name__}: {exc}"
                if stream is None and status == "ok":
                    stream = stream_check(weights, plan, cal.reference_clock_mhz)
                row = {"block_m": bm, "group_m": g, "tiles": n,
                       "rows_per_expert": rows_per_expert, "tokens": tokens,
                       "ms_p50": 0.0, "ms_min": 0.0, "ms_stdev": 0.0,
                       **timed, "status": status, "detail": detail, "pin": pin}
                rows.append(row)
                write_cells()
                print(f"  BM={bm:3d} G={g:3d} n={n:3d} r={rows_per_expert:5d} "
                      f"{row['ms_p50']:9.4f} ms  {status}"
                      f"{('  ' + detail) if detail else ''}")

    elapsed = time.time() - started
    print()
    print(f"measured {len(rows)} cells in {elapsed / 60.0:.1f} min")
    clocks = clock_state(rows)
    for line in clock_state_lines(clocks):
        print(line)
    if stream:
        print(f"stream check on the real weight buffers: {stream['gbps']:.1f} GB/s "
              f"(ceiling {cal.ceiling_gbps:.1f})")
    fits, refusals, gates, table = score_measured(
        rows, cfg, plan.dtype, plan.block_n, cal, stream, len(plan.cells))
    timed_iters = [int(r["iters"]) for r in rows
                   if r.get("status") == "ok" and r.get("iters")]
    payload = {"plan": asdict(plan), "run_id": plan.run_id(), "gpu": gpu,
               "calibration": asdict(cal), "stream_check": stream, "cells": rows,
               "clock_state": clocks,
               "elapsed_s": elapsed, "fits": [asdict(f) for f in fits],
               "refusals": [asdict(r) for r in refusals],
               "gates": [asdict(g) for g in gates],
               "noise_assumption_rel": args.noise}
    payload = PV.provenance_block(
        instrument=timing.TIMING_BASIS,
        ridge=cal.ridge,
        ridge_source=f"measured_{cal.slug}.yaml, this card's own calibration",
        bandwidth=cal.ceiling_gbps,
        bandwidth_source=f"measured_{cal.slug}.yaml:{cal.ceiling_pattern}",
        warmup_ms=plan.warmup_ms, target_ms=plan.cell_budget_ms,
        # The MEDIAN the cells were actually timed at, not a flag: `time_kernel`
        # sizes the count per cell from the budget, so a knob's default here
        # would contradict every row.
        iters=(int(statistics.median(timed_iters)) if timed_iters else None),
    ).stamp(payload)
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
    rc = exit_codes.classify(g.scored() for g in gates)
    print(exit_codes.describe(rc))
    return rc
def run_score_measured(args) -> int:
    """Score a `measure.json` a pod already wrote, on any machine.

    The pod is rented by the hour and the scoring is free. Splitting them means
    a run whose verdict path has a bug does not have to be paid for twice.
    """
    path = args.score_measured
    if not path.exists():
        print(f"REFUSED: no such file {path}. Nothing scored.")
        return exit_codes.REFUSED
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
    # NO `if not fits: return REFUSED` HERE ANY MORE. A measure.json with no
    # anchorable cell is a run that MEASURED and produced nothing scoreable,
    # which is what gates M5 and M6 are for; calling it REFUSED would file a
    # spent arm under "free, nothing attempted" and hide the failing gate.
    rc = exit_codes.classify(g.scored() for g in gates)
    print(exit_codes.describe(rc))
    return rc


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
    mode.add_argument("--self-test", action="store_true",
                      help="score the planted worlds in SELF_TEST_WORLDS, one "
                           "per FAIL branch of every M gate plus the world in "
                           "which the tile pin silently failed. No GPU, no "
                           "device, and no RESULT lines: nothing is measured")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan and its cost, measure nothing, exit 3")
    ap.add_argument("--published", type=Path, default=PUBLISHED,
                    help="root of the committed reports")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory the report lands in. Defaults to an "
                         "UNTRACKED session path under the results root for "
                         "both modes; --publish is the only thing that writes "
                         "the committed results/published/ANCHOR_RESCORE pair")
    ap.add_argument("--publish", action="store_true",
                    help="write the rescore to the tracked "
                         "results/published/ANCHOR_RESCORE.txt/.json. Until "
                         "2026-09-02 every --rescore did this, including the "
                         "one the session driver runs under --dry-run, so the "
                         "tree was dirty from arm one and the file carried the "
                         "author's home directory in it")
    ap.add_argument("--noise", type=float, default=CELL_SPREAD_REL,
                    metavar="REL",
                    help="assumed relative spread of a repeated cell timing, "
                         "the one number every MDE in the plan output is "
                         "derived from. Default is the median of the published "
                         "H200 replicates; they run 0.76%% to 1.82%%, so a "
                         "reader who wants the pessimistic end passes 0.0182")
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
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. THE SWEEP'S OWN DEFAULT, because "
                         "the anchor has to be the same physical event the "
                         "ladders measured and they warm for 300 ms; this arm "
                         "warmed for 5 CALLS against their 20 while the two "
                         "were compared tread for tread")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its own iteration count from it. "
                         "`--iters` is retired: it was a count this arm chose "
                         "and the ladders derived, so the two could not be the "
                         "same measurement")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per cell; the percentiles are over "
                         "iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the roof and the ladders are flushed "
                         "and a warm-L2 cell is not comparable with either. In "
                         "the run id, so a flushed and an unflushed run can "
                         "never share a directory")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.score_measured is not None:
        return run_score_measured(args)
    if not args.measure:
        args.rescore = True
    if args.out_dir is None:
        if args.measure:
            args.out_dir = results_root() / "memory_branch_anchor"
        elif args.publish:
            # results/published itself, NOT a dated subdirectory: see RESCORE_STEM.
            args.out_dir = PUBLISHED
        else:
            # UNTRACKED BY DEFAULT. `results/*` is git-ignored with only
            # `!results/published/` excepted, so this path cannot dirty the tree
            # however often it is re-run -- which the session driver does on
            # every dry run.
            args.out_dir = results_root() / "anchor_rescore"
    # A BAD FLAG IS A REFUSAL, not a crash and not a claim failure. `raise
    # SystemExit(msg)` exits 1, which is CLAIM_FAIL in the shared table, so a
    # mistyped --slope-tiles used to be recorded as "measured, and a
    # pre-registered claim was refuted".
    branch = [int(t) for t in args.slope_tiles.split(",")]
    if len(branch) < MIN_BRANCH_TREADS:
        print(
            f"REFUSED: --slope-tiles gives {len(branch)} branch treads and P3 needs "
            f"at least {MIN_BRANCH_TREADS}. On a planted ladder the slope moves 3.2% "
            "when the anchor is dropped at 8 treads, 1.4% at 12 and 0.8% at 16, "
            f"against a {SLOPE_INDEPENDENCE_REL:.1%} threshold taken from the "
            "committed 16- and 33-tread fits. A short branch would fail P3 for a "
            "reason that is about the grid and not about the kernel. Nothing "
            "measured.")
        return exit_codes.REFUSED
    if any(t < 2 for t in branch):
        print("REFUSED: --slope-tiles must all be >= 2: the anchor may not be "
              "inside the slope it is compared against. Nothing measured.")
        return exit_codes.REFUSED
    return run_measure(args) if args.measure else run_rescore(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:                                   # noqa: BLE001
        # ERROR (4), not the interpreter's 1. An unhandled exception exiting 1
        # would be read as CLAIM_FAIL -- "measured, and a pre-registered claim
        # was refuted" -- by the one table the driver reads, and a traceback is
        # the opposite of a result. 4 is RETRY in the ledger, which is what a
        # crash and an interrupt both deserve.
        traceback.print_exc()
        sys.exit(exit_codes.ERROR)
