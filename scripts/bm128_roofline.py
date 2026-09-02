#!/usr/bin/env python
"""At BLOCK_M=128, does achieved throughput PLATEAU below the card's own roof?

    python scripts/bm128_roofline.py --dry-run       # plan, predictions, cost. No GPU
    python scripts/bm128_roofline.py --self-test     # three planted worlds, off GPU
    python scripts/bm128_roofline.py                 # the pod run, ~2 min of H200

THE ONE MEASUREMENT IN THIS STUDY THAT GOES THROUGH NO FIT. Everything the study
currently claims travels through alpha, an anchor, an estimator and a ladder.
Each of those four has been attacked and some of the attacks landed: alpha 0.10
was an estimator artefact, the "crossing" was a tile-quantisation staircase, the
A100's whole BLOCK_M=128 row rests on one ladder that runs backwards at its last
tread and clears its own tolerance by 1.0e-4. This script asks the question those
four were built to answer, and asks it with a stopwatch:

    force BLOCK_SIZE_M = 128, sweep the batch from below the multi-tile onset to
    as deep as the card allows, and divide the achieved TFLOP/s by the dense bf16
    rate THIS card was measured at.

Two outcomes, and they are the study's fork:

  * throughput rises and then PLATEAUS below the roof -> the ceiling is real and
    it binds in the regime production actually runs.
  * throughput REACHES the roof -> the ceiling is not binding, and the study's
    central claim is about a regime that does not occur.

WHY 128 AND NOT A --block-m FLAG. 128 is the only production-relevant regime.
From the one published arm that RECORDS the tile vLLM chose:

    BLOCK_M    cells run multi-tile     max M-tiles per expert
       16          0 of 24                      1
       32          0 of  5                      1
       64          0 of 16                      1
      128         59 of 87                     32

The re-read term only exists when there is more than one M-tile per expert. At
16, 32 and 64 vLLM never runs multi-tile, so the caps computed for those tiles
are real and NEVER APPROACHED; at 128 it runs up to 32 tiles per expert and the
cap sits near 150 FLOP/byte against a ridge near 163. A `--block-m` flag would
let a run answer a different question under this script's name, so there is not
one. The tile that IS a parameter is the control, `--control`, and it is a
parameter because which tile can serve as a control depends on the card.

WHAT MULTI-TILE ONSET IS, COMPUTED AND NOT HARDCODED. One M-tile per expert
holds `BLOCK_M` rows, an expert receives `r = T k / E` rows under balanced
routing, so the second tile appears at

    r > BLOCK_M,  i.e.  T > BLOCK_M E / k

which is T > 512 on mixtral (E=8, k=2) and T > 1024 on qwen2 (E=64, k=8). The
grid is derived from that per model rather than written down, because the whole
point of the sweep is to straddle it and a hardcoded 512 straddles nothing on
qwen2. TEMPO (arXiv:2608.13057) states the tile term is INACTIVE in decode
because each expert receives at most 128 tokens and therefore one tile -- "the
tile-aware and tile-blind solutions coincide" -- which holds below roughly 256
tokens per expert and fails above it. Contesting that sentence with a measurement
is what this script is for, and it is why the grid deliberately spends cells
BELOW the onset as well as above: the pre-onset points are the regime TEMPO
describes, and they are plotted beside the ones it does not cover.

THE GRID IS DOUBLINGS, because a plateau is a claim about a DERIVATIVE and a
derivative needs a step of a stated size. Rows per expert run 32, 64, 128, 256,
... 4096, so every interior step is exactly one doubling of the batch and the
last point is 32 M-tiles per expert -- the depth the observed arm reaches. Every
point at or above 128 rows is an EXACTLY FULL tile stack, so padding is zero and
useful throughput is padded throughput; the two pre-onset points are partial
tiles and their padding is reported rather than hidden. Only full-stack
multi-tile points feed the plateau gate.

THE ROOF IS THIS CARD'S, MEASURED, OR THERE IS NO RUN. The denominator is the
dense bf16 rate `scripts/calibrate_hardware.py` measured on the ATTACHED device
(712.3 TFLOP/s on the H200 it calibrated on 2026-09-02), never a datasheet
figure and never another machine's file. A measured run with no calibration for
its own device REFUSES: seven published A100 reports were scored against 160.3
Op/B, a stale H200 ridge, and nothing in their output said so. `--dry-run` and
`--self-test` may assume the committed H200 calibration because nothing there is
measured, and gate V0 marks any report built that way as unquotable.

WHAT THE FRACTION IS AND IS NOT. The numerator counts GEMM flops only; the
denominator is a pure GEMM. The timed call is vLLM's whole fused layer, which
also aligns, permutes, activates and scatters. So the fraction UNDERSTATES the
GEMM's efficiency by whatever the non-GEMM work costs, and a subject that tops
out at 0.70 of the roof might be a binding AI ceiling or might be a layer that
spends 30% of itself outside the GEMMs. THAT is what the control is for, and it
is why the control is not optional.

THE POSITIVE CONTROL, and the two things it buys. The same sweep at BLOCK_M=256,
at the SAME token counts, on the SAME card, in the SAME session: a tile whose AI
cap is far above the ridge and which therefore has no ceiling to hit. Without it
a plateau below the roof could be anything -- the fused layer's fixed cost, the
kernel's occupancy, the card's clocks. With it the study gets a DIFFERENCE, and
a difference cancels every term the two tiles share. If both tiles plateau at the
same fraction, the shortfall belongs to the layer and not to the tile, and that
is a result this script reports rather than a failure it hides.

256 is not free: one CTA needs 4 x (256x64 + 64x64) x 2 B = 160 KiB of shared
memory, which fits sm_90's 227 KiB and fits sm_80's 163 KiB by three, and its
64x64 fp32 accumulator is 64 registers per thread against a ceiling of 255. Both
bills are computed from `block_m_crossing_sweep.tile_resources` and REFUSED
before a pod is rented, because a spilled kernel still returns a time and that
time still plots.

A PLATEAU IS A DERIVATIVE, SO IT IS GATED AS ONE. "It flattened out" is not a
measurement. The gate is: the throughput gain over the last `--plateau-doublings`
doublings, expressed PER DOUBLING, must fall below 2%. If it has not, the honest
answer printed is STILL RISING AT THE LARGEST BATCH MEASURED, and no sentence
containing the word plateau is earned. The gate also refuses when it cannot
RESOLVE 2%: with `--reps` repeats and a relative replication spread `s`, the
ratio of two per-tread medians carries about `s sqrt(2) / sqrt(reps)`, and when
that exceeds the threshold the verdict is UNKNOWN rather than a plateau read off
noise. That is why `--reps` defaults to 3 and not to 1.

THROTTLING INVALIDATES A THROUGHPUT MEASUREMENT MORE THAN A LATENCY ONE. A cell
timed while the SM clock sagged reports less throughput for a reason that has
nothing to do with tiles, and a sag that happens to land on the deep end of the
sweep manufactures a plateau. Every cell therefore records the SM clock and
temperature either side of its own timed interval; a cell whose clock drops more
than 5% across it, or which runs more than 5% below the session's own modal
clock, is EXCLUDED from every gate and printed with an x on the plot. The
exclusions are counted, and a run that excluded so much that the doubling chain
no longer spans the required distance says so instead of scoring.

AND THE CLOCK THE ROOF WAS MEASURED AT IS PART OF THE ROOF. The H200's dense
bf16 ceiling was taken at 1515 MHz -- a dense GEMM pulls the clock down from
1980 -- so a MoE layer running at 1800 MHz is being compared against a
measurement made at a different operating point. The direction is stated rather
than corrected: a run clocked ABOVE the roof's clock has its fraction
OVERSTATED, which is conservative for a claim that the fraction stays below 1;
a run clocked BELOW it has the fraction understated, and a plateau found there
could be the clocks. V3 scores it.

WHAT THE FIT IS STILL USED FOR, exactly and only: choosing 256 as the control
(its cap must clear the ridge by 30%) and PRE-REGISTERING a predicted plateau
band. No verdict below is computed from alpha, and the predicted band is wide on
purpose -- alpha at BLOCK_M=128 and GROUP_SIZE_M=1 has been measured anywhere
from 0.625 to 1.02 across this study's arms, which puts the predicted plateau
between 0.77 and 1.00 of the roof. A prediction that spans a quarter of the
range it is predicting is not a prediction, and that width IS the reason this
experiment exists.

A NOTE ON REUSE. The geometry, the pinned constants, the cell model, the timing
loop, the compile assay and the resource bill are IMPORTED from
`scripts/block_m_crossing_sweep.py` and called, never copied: this measurement
has to be commensurable with the sweep the study publishes, and a private copy
would drift. `_load_sweep` probes SIGNATURES as well as names, because that file
is under active edit and a renamed argument must produce a sentence on a laptop
rather than a TypeError two minutes into a metered pod session.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
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

from moe.bench.roofline import HARDWARE_DIR  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402


def _load_sweep():
    """Load `block_m_crossing_sweep` BY PATH, and name what is missing.

    `scripts/` is not a package, so a bare import works only when this file is
    the entry point and silently fails when a test loads it by path. The symbol
    and signature checks are not defensive noise: that file is under active edit
    by another workstream, and this one calls into it on the GPU path, where a
    rename costs a metered pod session rather than a laptop second.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    needed = ("FIXED", "COMPUTE_BOUND_FRACTION", "ALPHA", "ALPHA_BY_BLOCK_M",
              "PASS", "FAIL", "UNDECIDED", "ai_cap", "make_cell", "model_ms",
              "tokens_for_rows", "rows_quantum", "rows_step", "results_root",
              "scaled_iters", "useful_flops", "tiles_per_expert",
              "find_override", "arm_triton_cache", "count_new", "time_call",
              "balanced_ids", "missing_gpu_stack", "tile_resource_plan",
              "resolve_capability", "gate_0_override", "_measured_yaml")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/block_m_crossing_sweep.py no longer exports "
            f"{', '.join(missing)}. This script is deliberately scored by that "
            "file's geometry and timing rather than a private copy, so the two "
            "move together. Re-point the import; do not fork it.")
    import inspect
    for name, required in (("make_cell", ("sm_count", "block_n")),
                           ("model_ms", ("alpha", "ridge", "bandwidth_gbps")),
                           ("tile_resource_plan",
                            ("pinned", "block_sizes", "dtype_bytes",
                             "capability")),
                           ("gate_0_override",
                            ("compiles", "executed", "block_sizes"))):
        params = inspect.signature(getattr(module, name)).parameters
        gone = [p for p in required if p not in params]
        if gone:
            raise SystemExit(
                f"block_m_crossing_sweep.{name} no longer takes "
                f"{', '.join(gone)}. That file is under active edit and this "
                "one calls it on the pod path; re-check the call sites in "
                "measure_setting and analyse before spending GPU time.")
    return module


SWEEP = _load_sweep()


# --------------------------------------------------------------------------
# The thresholds this script is arguing about, all stated before any code.
# --------------------------------------------------------------------------

#: The subject. Not a parameter, for the reason the docstring gives: every
#: sentence in this file is about the tile vLLM runs multi-tile, and that is 128
#: in 59 of the 87 cells of the one arm that records what it ran.
SUBJECT_BLOCK_M = 128

#: The control: a tile whose AI cap sits far ABOVE the ridge, so it has no
#: ceiling to hit and any shortfall it shows belongs to the layer rather than to
#: the tile. A parameter because which tiles can run pinned depends on the card
#: (`--control 128 --num-stages 3` is the A100-safe fallback, and it is a WEAKER
#: control, not an equal one: 128 is the subject).
DEFAULT_CONTROL_BLOCK_M = 256

#: How far above the ridge a control's cap must sit before it may be called a
#: control at all. 1.30 is not tuned: at the pooled alpha the BLOCK_M=256 cap is
#: 2.8x the H200 ridge and 3.1x the A100's, so the margin is cleared by more
#: than double and a control that only just clears it is one whose own ceiling
#: is part of the argument.
CONTROL_CAP_MARGIN = 1.30

#: Throughput at or above this fraction of the measured roof counts as REACHING
#: the roof. Imported rather than restated so "compute bound" means one thing
#: across this study.
ROOF_REACHED = SWEEP.COMPUTE_BOUND_FRACTION

#: How much of the roof the control must beat the subject by before the subject's
#: shortfall may be attributed to its TILE rather than to the fused layer both
#: tiles run inside. 0.10 of the roof is 71 TFLOP/s on the H200, which is 5x the
#: worst per-cell timing spread this harness has produced and about 3x the
#: largest cross-session drift in its published corpus.
CONTROL_SEPARATION = 0.10

#: A plateau is a derivative: gain per doubling of the batch, below this, over
#: the span `--plateau-doublings` asks for. 2% per doubling against a harness
#: whose replicated per-tread spread runs 0.2-1.0% -- so the threshold is
#: resolvable at `--reps 3`, and the gate CHECKS that rather than assuming it.
PLATEAU_GAIN_PER_DOUBLING = 0.02

#: Doublings of the batch the gain is measured over. Two, because one doubling
#: of a noisy pair is a difference of two numbers and two doublings is a
#: direction.
DEFAULT_PLATEAU_DOUBLINGS = 2

#: Full-stack multi-tile points a subject needs before any verdict. Two points
#: make a slope with no residual; three can disagree with themselves.
MIN_MULTI_TILE_POINTS = 3

#: Percent drop in SM clock ACROSS a cell's own timed interval that marks it
#: throttled. The same 5% `moe.bench.timing.clock_drift` uses, so one number
#: means one thing in this repo.
THROTTLE_DRIFT_PCT = 5.0

#: Fraction of this session's cells that may throttle before the session itself
#: is unreadable. A single sagging cell is EXCLUDED and the run goes on -- that
#: is what the exclusion machinery is for, and killing a 39-cell run over one of
#: them would be a gate that fails on weather. A tenth of them sagging is a box
#: whose surviving medians are not trustworthy either, and that is a refusal.
THROTTLED_CELL_FRACTION = 0.10

#: How far below the session's modal SM clock a cell may start and still be
#: read. A cell that began 10% cold is not measuring the same silicon as its
#: neighbours, and a cold cell at the deep end manufactures a plateau.
CLOCK_FLOOR_FRACTION = 0.95

#: How far this run's clocks may sit from the clock the ROOF was measured at
#: before the fraction is a comparison of two operating points. The H200's dense
#: bf16 ceiling was taken at 1515 MHz against an idle 1980, so this is not a
#: hypothetical: it is the normal state of a dense GEMM.
ROOF_CLOCK_TOLERANCE = 0.10

#: The alphas this study has actually fitted at BLOCK_M=128 with GROUP_SIZE_M=1.
#: The low end is the published surface's own 128 row; the high end is the top of
#: the 0.92-1.02 range `block_m_crossing_sweep`'s --group-m help quotes for that
#: swizzle. The band is carried rather than a point estimate because the
#: PREDICTED plateau it implies spans 0.77 to 1.00 of the roof, and a prediction
#: that wide is the reason this experiment measures instead of computing.
ALPHA_128_BAND = (SWEEP.ALPHA_BY_BLOCK_M.get(SUBJECT_BLOCK_M, SWEEP.ALPHA), 1.02)

#: Rows per expert the grid starts at, and the deepest it goes. 32 is a quarter
#: of one tile, so the pre-onset points show the padding regime TEMPO describes;
#: 4096 is 32 M-tiles per expert, the depth the observed arm reaches.
DEFAULT_R_MIN = 32
DEFAULT_R_MAX = 4096

#: The card slug a run id carries when NO device is attached, i.e. every
#: --dry-run on a laptop. Visible rather than blank, so a dry run cannot be
#: mistaken for printing the path a pod will really write to.
UNKNOWN_CARD_SLUG = "nocard"

#: Used ONLY by --dry-run and --self-test, where nothing is measured and so
#: nothing can be mislabelled. Every one of these is stamped HYPOTHESIS in the
#: report and fails gate V0. A measured run REFUSES instead.
HYPOTHESIS_HARDWARE_STEM = "measured_nvidia_h200"
HYPOTHESIS_ROOF_TFLOPS = 712.259
HYPOTHESIS_RIDGE = 162.809
HYPOTHESIS_BANDWIDTH_GBPS = 4374.763
HYPOTHESIS_ROOF_CLOCK_MHZ = 1515
HYPOTHESIS_NOTE = ("HYPOTHESIS: the 2026-09-02 H200 calibration committed in "
                   "this repo. NO DEVICE IS ATTACHED, so this is a costing and "
                   "not a ceiling; gate V0 refuses to let it stand in a verdict")


# --------------------------------------------------------------------------
# Geometry. Pure arithmetic: no torch, no GPU, no CSV.
# --------------------------------------------------------------------------

def onset_rows(block_m: int) -> int:
    """Rows per expert ABOVE which a second M-tile exists. Trivially BLOCK_M.

    Named rather than inlined because the token form below is the one every
    sentence in the docstring quotes, and the two must not drift.
    """
    return block_m


def onset_tokens(cfg, block_m: int) -> int:
    """Largest token count that still fits in ONE M-tile per expert.

    `T = r E / k` at `r = BLOCK_M`. Multi-tile begins at the next legal token
    count above this, which is this plus `rows_step(cfg)`. Computed per model:
    it is 512 on mixtral and 1024 on qwen2, and a hardcoded 512 would put the
    whole qwen2 grid on the wrong side of the question.
    """
    return SWEEP.tokens_for_rows(cfg, onset_rows(block_m))


def doubling_rows(cfg, r_min: int, r_max: int, block_m: int) -> list[int]:
    """`r_min, 2 r_min, 4 r_min, ...` up to `r_max`, refusing an illegal row.

    Every interior step is exactly one doubling of the batch, which is what makes
    the plateau gate's per-doubling gain a quantity and not a curve fit. Rows at
    or above `block_m` are exactly-full tile stacks and rows below it are the
    single partial tile of the pre-onset regime; both are returned, and the
    caller separates them by `regime`.

    REFUSES a row the model's routing cannot express rather than nudging it: a
    nudged row is not a full tile stack, and a throughput read off a partly
    filled stack is a throughput divided by padding nobody recorded.
    """
    if r_min <= 0 or r_max < r_min:
        raise SystemExit(f"--r-min {r_min} and --r-max {r_max} do not describe "
                         "a grid: r_min must be positive and no larger than r_max.")
    if r_min & (r_min - 1):
        raise SystemExit(
            f"--r-min {r_min} is not a power of two, so the grid it generates "
            "is not a chain of doublings and the plateau gate's per-doubling "
            "gain would be a gain per something-else.")
    quantum = SWEEP.rows_quantum(cfg)
    rows, r = [], r_min
    while r <= r_max:
        if r % quantum:
            raise SystemExit(
                f"{cfg.name}: E={cfg.num_experts} at top-k {cfg.top_k} needs "
                f"rows per expert to be a multiple of {quantum}, and {r} is "
                "not. This model cannot express the doubling grid; choose "
                "another --model.")
        if r >= block_m and r % block_m:
            raise SystemExit(
                f"{r} rows per expert is not an exact multiple of BLOCK_M="
                f"{block_m}, so it is a partly-filled tile stack above the "
                "onset. Every point above the onset must be exactly full.")
        rows.append(r)
        r *= 2
    if not rows:
        raise SystemExit(f"--r-min {r_min} is above --r-max {r_max}; nothing "
                         "to measure.")
    return rows


def regime_of(rows: int, block_m: int) -> str:
    """`pre-onset` (one partial tile), `onset` (one full tile) or `multi-tile`."""
    if rows < block_m:
        return "pre-onset"
    if rows == block_m:
        return "onset"
    return "multi-tile"


def predicted_plateau_band(ridge: float, b: int, block_m: int = SUBJECT_BLOCK_M,
                           band: tuple[float, float] = ALPHA_128_BAND
                           ) -> tuple[float, float]:
    """The plateau the STUDY'S OWN ARITHMETIC predicts, as a fraction of the roof.

    `min(cap / ridge, 1)` at each end of the measured alpha band, clamped: a
    kernel cannot exceed the roof however large its arithmetic intensity, and
    leaving the clamp out is how a ceiling of 3.9 once made a gate unfailable.

    THIS IS A PRE-REGISTRATION AND NOT A CRITERION. Nothing downstream compares
    a measurement against it; it is printed before the run so that a reader can
    see the study committing itself, and so that the width -- 0.77 to 1.00 on
    the H200 -- is on the page next to the number that settles it.
    """
    lo_alpha, hi_alpha = min(band), max(band)
    high = min(SWEEP.ai_cap(block_m, lo_alpha, b) / ridge, 1.0)
    low = min(SWEEP.ai_cap(block_m, hi_alpha, b) / ridge, 1.0)
    return low, high


# --------------------------------------------------------------------------
# The roof. This card's own, or a refusal.
# --------------------------------------------------------------------------

class RoofUnavailable(RuntimeError):
    """No compute roof this run is entitled to use, and no constant may stand in.

    Raised rather than defaulted, for the reason `resolve_ridge` in the sibling
    sweep is raised rather than defaulted: `--ridge` used to fall back to a
    module constant and seven published A100 reports were scored against a stale
    H200 figure, invisibly.
    """


@dataclass(frozen=True)
class Roof:
    """The denominator, its provenance, and the clock it was measured at."""

    tflops: float
    ridge: float
    bandwidth_gbps: float
    #: SM clock the dense GEMM ceiling was taken at, MHz. 0 when the calibration
    #: did not record one, which makes V3 UNKNOWN rather than PASS.
    clock_mhz: int
    device: str
    source: str
    #: True only when this came from the ATTACHED device's own calibration.
    #: Every gate that quotes a fraction reads this.
    attached: bool

    def lines(self) -> list[str]:
        out = [f"roof         {self.tflops:.1f} TFLOP/s dense bf16, "
               + ("MEASURED on the attached card" if self.attached
                  else "NOT MEASURED HERE -- see the source below"),
               f"             {self.source}",
               f"             ridge {self.ridge:.2f} Op/B over "
               f"{self.bandwidth_gbps:.1f} GB/s"]
        out.append(f"             measured at {self.clock_mhz} MHz"
                   if self.clock_mhz else
                   "             the calibration recorded no GEMM clock, so V3 "
                   "cannot check this run against it")
        return out


def _hypothesis_roof(note: str) -> Roof:
    """The committed H200 calibration, for --dry-run and --self-test only."""
    try:
        from moe.bench.roofline import load_hardware
        hw = load_hardware(HYPOTHESIS_HARDWARE_STEM, directory=HARDWARE_DIR)
        detail = (__import__("yaml").safe_load(
            (HARDWARE_DIR / f"{HYPOTHESIS_HARDWARE_STEM}.yaml").read_text())
            or {}).get("detail") or {}
        return Roof(hw.peak("bf16") / 1e12, hw.ridge_point("bf16"),
                    hw.bandwidth_bytes_s / 1e9,
                    int(detail.get("gemm_clock_mhz") or 0), hw.name,
                    f"{note} ({HYPOTHESIS_HARDWARE_STEM}.yaml)", attached=False)
    except Exception:                                   # noqa: BLE001
        # The committed file is not on this checkout. The constants below are
        # that same calibration transcribed, and they are labelled twice over.
        return Roof(HYPOTHESIS_ROOF_TFLOPS, HYPOTHESIS_RIDGE,
                    HYPOTHESIS_BANDWIDTH_GBPS, HYPOTHESIS_ROOF_CLOCK_MHZ,
                    "NVIDIA H200 (measured)",
                    f"{note}; the yaml is absent on this checkout, so these are "
                    "module constants", attached=False)


def resolve_roof(dtype: str, *, synthetic: bool) -> Roof:
    """This card's measured dense rate, or a refusal.

    Order, and each step is a different kind of claim:

      1. THE ATTACHED DEVICE'S OWN CALIBRATION, which is the only one a verdict
         may quote. `peak(dtype)` is the achieved dense rate
         `scripts/calibrate_hardware.py` measured on this box, not a datasheet
         peak: the H200 calibrates at 712 TFLOP/s against a 989 marketing
         figure, and scoring against the second would understate every fraction
         by 28% and turn a real plateau into a fake one.
      2. For `--dry-run` and `--self-test` ONLY, the committed H200 calibration
         as a stated HYPOTHESIS, which gate V0 then refuses to let stand.
      3. Otherwise REFUSE.
    """
    from moe.bench import roofline
    gpu_name = roofline.current_gpu_name()
    hw = None
    try:
        hw = roofline.load_measured(gpu_name or None)
    except roofline.HardwareMismatch as exc:
        raise RoofUnavailable(str(exc)) from exc
    if hw is not None:
        try:
            tflops = hw.peak(dtype) / 1e12
            ridge = hw.ridge_point(dtype)
        except ValueError as exc:
            raise RoofUnavailable(
                f"{hw.name} has a measured bandwidth but no verified {dtype} "
                f"peak, so it states no compute roof: {exc}") from exc
        detail = SWEEP._measured_yaml(gpu_name).get("detail") or {}
        doc = SWEEP._measured_yaml(gpu_name)
        stamp = (SWEEP.calibration_stamp_line(doc)
                 if hasattr(SWEEP, "calibration_stamp_line") else "")
        return Roof(tflops, ridge, hw.bandwidth_bytes_s / 1e9,
                    int(detail.get("gemm_clock_mhz") or 0), hw.name,
                    f"measured on this device: {hw.name}, {tflops:.1f} TFLOP/s "
                    f"{dtype} over {hw.bandwidth_bytes_s / 1e9:.1f} GB/s"
                    + (f"; {stamp}" if stamp else ""), attached=True)
    if synthetic:
        return _hypothesis_roof(HYPOTHESIS_NOTE)
    raise RoofUnavailable(
        f"no calibration for this device ({gpu_name or 'no CUDA device'}), so "
        "this run has no compute roof it is entitled to divide by.\n"
        "    EVERY number this script produces is a fraction of that roof. "
        "Against another machine's ceiling the fraction is wrong by the ratio "
        "of two parts, and nothing in the output would say so -- which is how "
        "seven published A100 reports came to quote an H200 ridge.\n"
        "    Run:  python scripts/calibrate_hardware.py\n"
        "    Off GPU, --dry-run and --self-test may assume the committed H200 "
        "calibration and are marked unquotable for it.")


# --------------------------------------------------------------------------
# One timing.
# --------------------------------------------------------------------------

@dataclass
class Timing:
    """One (block_m, rows, rep) measurement and the clocks around it.

    RESUME IS KEYED ON ROWS, NOT ON TILES, and that is not a detail. The three
    pre-onset points at 32, 64 and 128 rows all have `tiles == 1` at BLOCK_M=128,
    so a manifest keyed on the tile count would find the first of them present
    and skip the other two, silently reporting a quarter-full tile's throughput
    under a full one's label. The sibling depth script keys on tiles safely only
    because it measures nothing below one full stack.
    """

    block_m: int
    rows_per_expert: int
    tiles: int
    tokens: int
    rep: int
    ms_p50: float
    ms_min: float
    ms_stdev: float
    iters: int
    sm_clock_start_mhz: int = 0
    sm_clock_end_mhz: int = 0
    temp_start_c: int = 0
    temp_end_c: int = 0
    clock_drift_pct: float = 0.0
    throttled: bool = False
    status: str = "ok"
    detail: str = ""

    @property
    def clock_seen(self) -> bool:
        """Did anything actually read a clock for this cell?

        NON-VACUITY at the level of one row. `ClockState.sample` returns zeros
        when NVML is absent and when nvidia-smi is restricted, and a throttle
        check over rows like that examines nothing and reports no failures.
        """
        return self.sm_clock_start_mhz > 0 and self.sm_clock_end_mhz > 0


TIMING_FIELDS = list(Timing.__dataclass_fields__)


def append_timing(path: Path, row: Timing) -> None:
    """One row, flushed. An abort costs the cell in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMING_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))
        fh.flush()


def read_timings(path: Path) -> tuple[set[tuple[int, int, int]], list[Timing]]:
    """Timings already on disk, so a re-run resumes rather than repeats."""
    if not path.exists():
        return set(), []
    out: list[Timing] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Timing(
                block_m=int(row["block_m"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tiles=int(row["tiles"]), tokens=int(row["tokens"]),
                rep=int(row["rep"]), ms_p50=float(row["ms_p50"]),
                ms_min=float(row["ms_min"]), ms_stdev=float(row["ms_stdev"]),
                iters=int(row["iters"]),
                sm_clock_start_mhz=int(float(row.get("sm_clock_start_mhz") or 0)),
                sm_clock_end_mhz=int(float(row.get("sm_clock_end_mhz") or 0)),
                temp_start_c=int(float(row.get("temp_start_c") or 0)),
                temp_end_c=int(float(row.get("temp_end_c") or 0)),
                clock_drift_pct=float(row.get("clock_drift_pct") or 0.0),
                throttled=str(row.get("throttled", "")).lower() in ("true", "1"),
                status=row.get("status", "ok"), detail=row.get("detail", "")))
    # Only SUCCEEDED timings count as done: the common failure here is a pod
    # that lost its device, which a re-run can leave behind, and a real failure
    # fails again in milliseconds.
    return ({(t.block_m, t.rows_per_expert, t.rep) for t in out
             if t.status == "ok"}, out)


# --------------------------------------------------------------------------
# The curve: one point per (block_m, rows), with its exclusions attached.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    """One plotted point: the median over repeats, and whether it is readable."""

    block_m: int
    rows_per_expert: int
    tiles: int
    tokens: int
    regime: str
    tile_eff: float
    reps: int
    ms_p50: float
    #: Relative spread of the per-repeat medians. None with a single repeat,
    #: which is what makes the plateau gate refuse to resolve anything.
    spread: float | None
    useful_tflops: float
    roof_fraction: float
    sm_clock_mhz: int
    throttled_reps: int
    retained: bool
    excluded_why: str

    @property
    def aligned(self) -> bool:
        return self.rows_per_expert % self.block_m == 0


def modal_clock(timings: list[Timing]) -> int:
    """The clock this session mostly ran at, or 0 when none was ever read.

    The MEDIAN of every start sample rather than the maximum: one spuriously
    high reading taken during a ramp would put every other cell below the floor
    and exclude the whole run.
    """
    seen = [t.sm_clock_start_mhz for t in timings
            if t.status == "ok" and t.clock_seen]
    return int(statistics.median(seen)) if seen else 0


def build_points(timings: list[Timing], cfg, block_m: int, roof: Roof,
                 *, sm_count: int, block_n: int, clock_ref: int) -> list[Point]:
    """Collapse repeats into one point per row count, and mark the unreadable.

    The median across REPEATS, not the single-pass median: a repeat is a fresh
    call at a fresh point in the pod's thermal history, and a throughput read
    from one pass cannot tell a sagging clock from a tile effect.

    A point is dropped when EVERY repeat of it was throttled, and kept on the
    unthrottled ones otherwise -- with the count of what was dropped carried on
    the point, because a point standing on one of five repeats is not the same
    measurement as one standing on five.
    """
    by: dict[int, list[Timing]] = {}
    for t in timings:
        if t.block_m == block_m and t.status == "ok" and t.ms_p50 > 0:
            by.setdefault(t.rows_per_expert, []).append(t)
    out: list[Point] = []
    for rows in sorted(by):
        group = by[rows]
        cold = [t for t in group
                if clock_ref and t.clock_seen
                and t.sm_clock_start_mhz < clock_ref * CLOCK_FLOOR_FRACTION]
        bad = {id(t) for t in group if t.throttled} | {id(t) for t in cold}
        good = [t for t in group if id(t) not in bad]
        used = good or group
        ms = statistics.median([t.ms_p50 for t in used])
        spread = (statistics.pstdev([t.ms_p50 for t in used]) / ms
                  if len(used) > 1 and ms > 0 else None)
        cell = SWEEP.make_cell(cfg, rows, block_m, ms, sm_count=sm_count,
                               block_n=block_n)
        clocks = [t.sm_clock_start_mhz for t in used if t.clock_seen]
        why = ""
        if not good:
            why = (f"every one of {len(group)} repeats was throttled or ran "
                   f"below {CLOCK_FLOOR_FRACTION:.0%} of the session's "
                   f"{clock_ref} MHz modal clock")
        out.append(Point(
            block_m=block_m, rows_per_expert=rows, tiles=cell.tiles_per_expert,
            tokens=cell.tokens, regime=regime_of(rows, block_m),
            tile_eff=cell.tile_eff, reps=len(used), ms_p50=ms, spread=spread,
            useful_tflops=cell.useful_tflops,
            roof_fraction=(cell.useful_tflops / roof.tflops
                           if roof.tflops > 0 else 0.0),
            sm_clock_mhz=int(statistics.median(clocks)) if clocks else 0,
            throttled_reps=len(bad), retained=bool(good), excluded_why=why))
    return out


def multi_tile(points: list[Point]) -> list[Point]:
    """Retained, exactly-full, more than one tile per expert. The gated set."""
    return [p for p in points
            if p.retained and p.aligned and p.regime == "multi-tile"]


# --------------------------------------------------------------------------
# A plateau is a claim about a derivative, so it is measured as one.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plateau:
    """Gain per doubling over the deep end, and whether 2% was resolvable."""

    required_doublings: float
    span_doublings: float | None
    start_tiles: int | None
    end_tiles: int | None
    start_tflops: float | None
    end_tflops: float | None
    total_gain: float | None
    gain_per_doubling: float | None
    #: Relative spread of the per-tread medians over the window, and the
    #: smallest gain per doubling those spreads can tell from zero.
    spread: float | None
    resolution: float | None
    threshold: float
    reason: str = ""

    @property
    def resolvable(self) -> bool | None:
        if self.resolution is None:
            return None
        return self.resolution <= self.threshold

    @property
    def plateaued(self) -> bool | None:
        """True, False, or None for "this cannot be decided from these points"."""
        if self.gain_per_doubling is None or self.span_doublings is None:
            return None
        if self.span_doublings < self.required_doublings - 1e-9:
            return None
        if self.resolvable is not True:
            return None
        return self.gain_per_doubling <= self.threshold

    def line(self) -> str:
        if self.gain_per_doubling is None:
            return f"no derivative: {self.reason}"
        return (f"{self.gain_per_doubling:+.3%} per doubling over "
                f"{self.span_doublings:.0f} doublings "
                f"(n={self.start_tiles} -> {self.end_tiles} tiles, "
                f"{self.start_tflops:.1f} -> {self.end_tflops:.1f} TFLOP/s, "
                f"{self.total_gain:+.2%} in total); resolvable to "
                + (f"{self.resolution:.3%}" if self.resolution is not None
                   else "UNKNOWN (one repeat: no spread to resolve against)"))


def plateau_of(points: list[Point], *, doublings: float,
               threshold: float = PLATEAU_GAIN_PER_DOUBLING) -> Plateau:
    """The gain per doubling over the SHORTEST suffix spanning `doublings`.

    A suffix rather than the whole curve, because the question is whether it has
    flattened by the largest batch measured, not whether it was ever flat. The
    span is measured in log2 of the TILE COUNT and not in grid positions, so a
    point excluded for a sagging clock shortens the chain honestly instead of
    letting two non-adjacent points pass as one doubling.

    RESOLUTION IS PART OF THE ANSWER. The gain is a ratio of two per-tread
    medians, each carrying a relative spread `s` over `reps` repeats, so the
    ratio carries about `s sqrt(2) / sqrt(reps)`. When that exceeds the
    threshold the honest verdict is that a 2% gain could not be told from zero,
    which is UNKNOWN and never a plateau.
    """
    ordered = sorted(points, key=lambda p: p.tiles)
    if len(ordered) < 2:
        return Plateau(doublings, None, None, None, None, None, None, None,
                       None, None, threshold,
                       f"{len(ordered)} retained multi-tile point(s); a "
                       "derivative needs at least two")
    last = ordered[-1]
    if last.useful_tflops <= 0:
        return Plateau(doublings, None, None, None, None, None, None, None,
                       None, None, threshold,
                       "the deepest retained point has no throughput")
    start = ordered[0]
    for candidate in ordered[:-1]:
        if math.log2(last.tiles / candidate.tiles) >= doublings - 1e-9:
            start = candidate
    span = math.log2(last.tiles / start.tiles)
    if start.useful_tflops <= 0 or span <= 0:
        return Plateau(doublings, span or None, start.tiles, last.tiles,
                       start.useful_tflops, last.useful_tflops, None, None,
                       None, None, threshold,
                       "the window collapsed to a single tile count")
    total = last.useful_tflops / start.useful_tflops - 1.0
    per = (1.0 + total) ** (1.0 / span) - 1.0
    window = [p for p in ordered if start.tiles <= p.tiles <= last.tiles]
    spreads = [p.spread for p in window if p.spread is not None]
    reps = min((p.reps for p in window), default=1)
    spread = max(spreads) if spreads else None
    resolution = (spread * math.sqrt(2.0) / math.sqrt(max(reps, 1))
                  if spread is not None else None)
    reason = ""
    if span < doublings - 1e-9:
        reason = (f"the retained multi-tile points span only {span:.2f} "
                  f"doublings, under the {doublings:g} this gate needs; sweep "
                  "deeper with --r-max or recover the excluded points")
    return Plateau(doublings, span, start.tiles, last.tiles, start.useful_tflops,
                   last.useful_tflops, total, per, spread, resolution,
                   threshold, reason)


# --------------------------------------------------------------------------
# The picture, drawn in the log so a reader of the transcript sees the shape.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Series:
    label: str
    marker: str
    points: list[Point]


def ascii_plot(series: list[Series], roof: Roof, *, height: int = 18,
               col: int = 7) -> list[str]:
    """Roof fraction against batch, in characters.

    The y axis is a fraction of the MEASURED roof and the roof itself is drawn as
    a dashed line, so the question the script exists to ask -- does the subject
    touch that line -- is answered by looking. The x axis is the token grid,
    evenly spaced, which is a log axis because the grid is doublings.

    Excluded points are drawn as `x` at the height they would have had. They are
    on the plot and out of the gates, which is the only combination that lets a
    reader see what was thrown away.
    """
    cols = sorted({p.tokens for s in series for p in s.points})
    if not cols:
        return ["(no points to plot)"]
    # THE ROOF LANDS ON A GRIDLINE, EXACTLY. The row grid is built so that 1.00
    # is `height - 1` steps above zero and any headroom above it is a whole
    # number of the same steps, so the row the dashes are drawn on is the row
    # whose printed label reads 1.00. An axis whose top row is labelled 1.02 and
    # dashed as the roof is an axis that has to be read twice.
    step = 1.0 / (height - 1)
    peak = max(p.roof_fraction for s in series for p in s.points)
    extra = max(0, math.ceil((peak - 1.0) / step - 1e-9))
    total = height + extra
    top = 1.0 + extra * step
    rows = [[" "] * (len(cols) * col) for _ in range(total)]

    def row_of(fraction: float) -> int:
        return min(total - 1, max(0, round((top - fraction) / step)))

    roof_row = extra
    for i in range(len(cols) * col):
        rows[roof_row][i] = "-"
    for s in series:
        for p in s.points:
            r = row_of(p.roof_fraction)
            c = cols.index(p.tokens) * col + col // 2
            mark = s.marker if p.retained else "x"
            rows[r][c] = "*" if rows[r][c] not in (" ", "-") else mark

    out = [f"fraction of the {roof.tflops:.1f} TFLOP/s measured dense bf16 roof"]
    for i, line in enumerate(rows):
        tail = "  <- the roof" if i == roof_row else ""
        out.append(f"  {top - i * step:4.2f} |{''.join(line).rstrip()}{tail}")
    out.append("       +" + "-" * (len(cols) * col))
    out.append(("    T   " + "".join(f"{t:^{col}d}" for t in cols)).rstrip())
    out.append("        " + "  ".join(
        f"{s.marker} BLOCK_M={s.points[0].block_m}" for s in series if s.points)
        + "   x excluded (throttled or off-clock)   * both")
    return out


def point_table(series: list[Series]) -> list[str]:
    """The numbers behind the picture, because 18 rows of characters lose them."""
    out = [f"{'BM':>4s} {'T':>7s} {'rows':>6s} {'n':>4s} {'regime':>10s} "
           f"{'tile_eff':>8s} {'ms':>10s} {'TFLOP/s':>9s} {'of roof':>8s} "
           f"{'MHz':>5s} {'reps':>4s}  note".rstrip()]
    for s in series:
        for p in s.points:
            note = p.excluded_why or ("" if p.retained else "excluded")
            if p.throttled_reps and p.retained:
                note = f"{p.throttled_reps} repeat(s) excluded"
            out.append(
                f"{p.block_m:4d} {p.tokens:7d} {p.rows_per_expert:6d} "
                f"{p.tiles:4d} {p.regime:>10s} {p.tile_eff:8.3f} "
                f"{p.ms_p50:10.4f} {p.useful_tflops:9.1f} "
                f"{p.roof_fraction:8.3f} {p.sm_clock_mhz:5d} "
                f"{p.reps:4d}  {note}".rstrip())
    return out


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
    out += ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN"]
    return out


def gate_v0_roof(roof: Roof) -> Gate:
    """The denominator belongs to the card that produced the numerator."""
    return Gate(
        VALIDITY, "V0 roof provenance",
        "the roof is the ATTACHED card's own measured dense bf16 rate",
        "the calibration names this device and was written by "
        "scripts/calibrate_hardware.py on it",
        roof.attached, roof.source,
        "every fraction in this report. A roof from another machine, or a "
        "datasheet peak, is wrong by the ratio of two parts and nothing "
        "downstream would say so",
        ["The H200 calibrates at 712 TFLOP/s dense bf16 against a 989 "
         "marketing figure: the datasheet denominator would understate every "
         "fraction by 28% and manufacture a plateau."])


def gate_v1_pin(compiles: dict[int, int], executed: dict[int, int],
                block_sizes) -> Gate:
    """Did override_config actually change the kernel, at BOTH tiles.

    ADOPTED from `block_m_crossing_sweep.gate_0_override` rather than
    reimplemented: it is the same assay over the same Triton cache, and a second
    copy of it would be a second thing to keep in step with the sweep.
    """
    inner = SWEEP.gate_0_override(compiles, executed, block_sizes)
    passed = {SWEEP.PASS: True, SWEEP.FAIL: False,
              SWEEP.UNDECIDED: None}[inner.verdict]
    return Gate(
        VALIDITY, "V1 tile pinned",
        f"the forced tile really ran at each of {tuple(block_sizes)}",
        f"{inner.threshold} fresh Triton artefacts, per tile",
        passed, inner.measured,
        "the whole comparison. If the override silently failed, both tiles ran "
        "one kernel, the control-minus-subject difference is zero by "
        "construction, and the report reads as a tidy null result",
        list(inner.lines))


def gate_v2_non_vacuity(subject: list[Point], control: list[Point],
                        planned_multi_tile: int, *, onset_tokens_value: int
                        ) -> Gate:
    """Did this run measure the thing it is about.

    A check that examined nothing also reports zero failures. Three counts, and
    the third is the one that matters: crossing the multi-tile onset is the
    entire premise, so a run whose deepest subject point is still one tile per
    expert has measured the regime TEMPO already describes and nothing else.
    """
    deep = multi_tile(subject)
    reached = max((p.tiles for p in deep), default=0)
    shared = len({p.tokens for p in multi_tile(control)}
                 & {p.tokens for p in deep})
    ok = (len(deep) >= MIN_MULTI_TILE_POINTS and reached >= 4 and shared >= 2)
    return Gate(
        VALIDITY, "V2 non-vacuity",
        "the sweep crossed the multi-tile onset and measured both tiles there",
        f">= {MIN_MULTI_TILE_POINTS} retained full-stack multi-tile subject "
        "points, >= 4 M-tiles per expert reached, >= 2 token counts shared with "
        "the control",
        ok,
        f"{len(deep)} retained subject points of {planned_multi_tile} "
        f"multi-tile batches planned, deepest "
        f"{reached} M-tiles per expert, {shared} token counts shared with the "
        "control",
        "every claim gate. Below the onset there is one tile per expert, the "
        "re-read term does not exist, and this script has measured the regime "
        "the prior work already agrees about",
        [f"multi-tile begins above T={onset_tokens_value} for this model, "
         "which is where the second M-tile per expert appears."])


def gate_v3_clocks(timings: list[Timing], points: list[Point], roof: Roof,
                   clock_ref: int) -> Gate:
    """Was this measured at one operating point, and at the roof's operating point.

    Throttling invalidates a throughput measurement more than a latency one: a
    latency is wrong by the clock ratio, while a throughput read at a sagging
    clock is wrong AND looks exactly like the flattening this script is trying
    to detect.
    """
    ok_rows = [t for t in timings if t.status == "ok"]
    with_clock = [t for t in ok_rows if t.clock_seen]
    if not with_clock:
        return Gate(
            VALIDITY, "V3 clocks", "every cell was timed at one clock",
            f"no cell throttled past {THROTTLE_DRIFT_PCT:.0f}%, and the "
            f"session clock within {ROOF_CLOCK_TOLERANCE:.0%} of the roof's",
            None,
            f"no clock was read on any of {len(ok_rows)} cells; NVML and "
            "nvidia-smi both returned nothing",
            "the throttle exclusions and the roof comparison. A throttle check "
            "over rows with no clocks examined nothing and would report no "
            "failures",
            ["NON-VACUITY: this gate refuses to PASS on an instrument that "
             "never took a reading."])
    throttled = [t for t in with_clock if t.throttled]
    dropped = [p for p in points if not p.retained]
    lines = [f"{len(with_clock)} of {len(ok_rows)} cells carry a clock; "
             f"session modal clock {clock_ref} MHz",
             f"{len(throttled)} cells drifted past {THROTTLE_DRIFT_PCT:.0f}% "
             f"across their own timed interval; {len(dropped)} points lost "
             "every repeat and are excluded"]
    if not roof.clock_mhz:
        return Gate(
            VALIDITY, "V3 clocks", "every cell was timed at one clock",
            f"no cell throttled past {THROTTLE_DRIFT_PCT:.0f}%, and the "
            "session clock within "
            f"{ROOF_CLOCK_TOLERANCE:.0%} of the clock the roof was measured at",
            None, f"{len(throttled)} throttled cells, but the calibration "
                  "records no GEMM clock to compare this session against",
            "the direction of the roof-fraction bias. Without the roof's clock "
            "a fraction below 1 cannot be told from a card running slower now "
            "than it did when it was calibrated", lines)
    rel = clock_ref / roof.clock_mhz - 1.0
    lines.append(
        f"this session ran {rel:+.1%} against the {roof.clock_mhz} MHz the roof "
        "was measured at, which is "
        + ("inside" if abs(rel) <= ROOF_CLOCK_TOLERANCE else "OUTSIDE")
        + f" the {ROOF_CLOCK_TOLERANCE:.0%} this gate allows. "
        + ("Direction, to the extent this layer's rate follows the clock: the "
           "fraction is OVERSTATED, which is the conservative direction for a "
           "claim that it stays below 1."
           if rel > 0 else
           "Direction, to the extent this layer's rate follows the clock: the "
           "fraction is UNDERSTATED, so a plateau found on a session well below "
           "the roof's clock could be the clocks rather than the tile. The "
           "magnitude is not stated because a memory-bound tread does not "
           "follow the SM clock at all and a compute-bound one nearly does."))
    share = len(throttled) / len(with_clock)
    ok = share <= THROTTLED_CELL_FRACTION and abs(rel) <= ROOF_CLOCK_TOLERANCE
    if dropped:
        lines.append(
            f"points that lost EVERY repeat and left the gated set: "
            f"{[(p.block_m, p.rows_per_expert) for p in dropped]}. Losing the "
            "deepest one shortens the doubling chain, which C2 reports as a "
            "span too short rather than as a plateau.")
    return Gate(
        VALIDITY, "V3 clocks", "every cell was timed at one clock, and at the "
        "clock the roof was measured at",
        f"<= {THROTTLED_CELL_FRACTION:.0%} of cells drifting past "
        f"{THROTTLE_DRIFT_PCT:.0f}%, and the session clock within "
        f"{ROOF_CLOCK_TOLERANCE:.0%} of {roof.clock_mhz} MHz",
        ok, f"{len(throttled)} of {len(with_clock)} cells throttled "
            f"({share:.1%}); session {clock_ref} MHz, {rel:+.1%} against the "
            "roof's",
        "the fractions, in the direction stated below. Individual sagging cells "
        "are already excluded; a whole session off-clock, or a box that sags "
        "this often, is not something an exclusion can fix", lines)


def gate_v4_control_ran(control: list[Point], subject: list[Point],
                        control_block_m: int) -> Gate:
    """Did the control produce a comparable curve at all.

    Separated from the CLAIM the control supports, on purpose: whether the
    instrument did the work is a fact about this session, and what the work
    showed is a fact about the hardware. Merging them is how "the control did not
    run" gets reported as "the tiles did not differ".
    """
    deep = multi_tile(control)
    shared = sorted({p.tokens for p in deep} & {p.tokens for p in multi_tile(subject)})
    ok = len(deep) >= MIN_MULTI_TILE_POINTS and len(shared) >= MIN_MULTI_TILE_POINTS
    return Gate(
        VALIDITY, "V4 control ran",
        f"the control tile BLOCK_M={control_block_m} ran the same batches",
        f">= {MIN_MULTI_TILE_POINTS} retained full-stack control points at "
        "token counts the subject also ran",
        ok, f"{len(deep)} control points, {len(shared)} of them at shared token "
            f"counts {shared[:6]}",
        "gate C3, and with it every attribution of the subject's shortfall to "
        "its TILE rather than to the fused layer both tiles run inside",
        ["Without a control a plateau below the roof could be the layer's "
         "permute, activation and scatter rather than an AI ceiling: an "
         "absence recorded by an instrument never shown to detect a presence."])


def gate_c1_roof(subject: list[Point], plateau: Plateau, roof: Roof,
                 predicted: tuple[float, float]) -> Gate:
    """THE HEADLINE. Does BLOCK_M=128 reach the roof where production runs.

    Fit-free on both sides: the numerator is a stopwatch over a known flop
    count, the denominator is a stopwatch over a dense GEMM on the same card.
    The predicted band is printed for the record and compared against nothing.
    """
    deep = multi_tile(subject)
    if not deep:
        return Gate(
            CLAIM, "C1 roof", f"BLOCK_M={SUBJECT_BLOCK_M} stays below the "
            "compute roof in the multi-tile regime",
            f"peak roof fraction < {ROOF_REACHED:.2f}", None,
            "no retained full-stack multi-tile points", "",
            ["Nothing was measured above the onset, so there is no regime here "
             "to be below the roof in."])
    peak = max(p.roof_fraction for p in deep)
    last = max(deep, key=lambda p: p.tiles)
    rising = plateau.plateaued is not True
    lines = [
        f"predicted plateau band, from the study's own alpha at BLOCK_M=128 "
        f"(GROUP_SIZE_M=1): {predicted[0]:.3f} to {predicted[1]:.3f} of the "
        f"roof, i.e. {predicted[0] * roof.tflops:.0f} to "
        f"{predicted[1] * roof.tflops:.0f} TFLOP/s. PRE-REGISTERED AND SCORED "
        "AGAINST NOTHING: a band that wide is a statement about the fit, not "
        "about the hardware.",
        f"the deepest point measured is n={last.tiles} M-tiles per expert at "
        f"T={last.tokens}, reaching {last.roof_fraction:.3f} of the roof "
        f"({last.useful_tflops:.1f} of {roof.tflops:.1f} TFLOP/s), and it was "
        + ("STILL RISING there." if rising else "flat there."),
        "the fraction UNDERSTATES the GEMM's own efficiency: the numerator "
        "counts GEMM flops and the denominator is a dense GEMM, while the timed "
        "call is the whole fused layer. C3 is what removes that term.",
    ]
    if peak >= ROOF_REACHED:
        return Gate(
            CLAIM, "C1 roof",
            f"BLOCK_M={SUBJECT_BLOCK_M} stays below the compute roof in the "
            "multi-tile regime",
            f"peak roof fraction < {ROOF_REACHED:.2f}", False,
            f"peak {peak:.3f} of the roof at n={last.tiles} M-tiles", "",
            [f"BLOCK_M={SUBJECT_BLOCK_M} REACHED the roof. The AI ceiling is "
             "not binding in the regime production runs, and the study's "
             "central claim is about a regime that does not occur."] + lines)
    return Gate(
        CLAIM, "C1 roof",
        f"BLOCK_M={SUBJECT_BLOCK_M} stays below the compute roof in the "
        "multi-tile regime",
        f"peak roof fraction < {ROOF_REACHED:.2f}", True,
        f"peak {peak:.3f} of the roof at n={last.tiles} M-tiles", "", lines)


def gate_c2_plateau(plateau: Plateau, subject: list[Point]) -> Gate:
    """The derivative. A plateau is a claim about one, so it is gated as one."""
    deep = multi_tile(subject)
    last = max(deep, key=lambda p: p.tiles) if deep else None
    honest = ("still rising at the largest batch measured"
              if plateau.plateaued is False else
              "cannot be decided from these points")
    lines = [plateau.line() if plateau.gain_per_doubling is not None
             else f"no derivative: {plateau.reason}"]
    if plateau.reason and plateau.gain_per_doubling is not None:
        lines.append(plateau.reason)
    if last is not None:
        lines.append(
            f"the last point reached {last.roof_fraction:.3f} of the roof and "
            + ("was still rising there." if plateau.plateaued is False else
               "was flat there." if plateau.plateaued else
               "cannot be called either way."))
        if plateau.plateaued is not True:
            lines.append(f"THE HONEST SENTENCE IS '{honest}'. Nothing in this "
                         "report may say 'plateaued'.")
    if plateau.resolvable is False:
        lines.append(
            f"REFUSED on resolution: the replicated spread puts "
            f"{plateau.resolution:.3%} on the ratio of two treads, over the "
            f"{plateau.threshold:.1%} this gate has to resolve. Raise --reps; "
            "a plateau read off noise is not a plateau.")
    return Gate(
        CLAIM, "C2 plateau",
        f"throughput has stopped rising by n={last.tiles if last else '?'} "
        "M-tiles per expert",
        f"gain <= {PLATEAU_GAIN_PER_DOUBLING:.1%} per doubling over the last "
        f"{plateau.required_doublings:g} doublings, and the spread must resolve "
        "that",
        plateau.plateaued,
        (f"{plateau.gain_per_doubling:+.3%} per doubling"
         if plateau.gain_per_doubling is not None else plateau.reason),
        "the word 'plateau'. Without it the reading is 'still rising at the "
        "largest batch measured', which is a statement about this sweep's depth "
        "and not about the hardware", lines)


def gate_c3_attribution(subject: list[Point], control: list[Point],
                        control_plateau: Plateau, control_block_m: int,
                        roof: Roof) -> Gate:
    """Is the shortfall the TILE's, or the fused layer's.

    The difference of two roof fractions measured on the same card, in the same
    session, at the same token counts, through the same fused layer, differing
    only in the forced tile. Every term the two share cancels -- the permute,
    the activation, the scatter, the clocks -- which is what makes this the one
    number in the report that survives not knowing what the layer costs.
    """
    deep_s, deep_c = multi_tile(subject), multi_tile(control)
    shared = sorted({p.tokens for p in deep_s} & {p.tokens for p in deep_c})
    if not shared:
        return Gate(
            CLAIM, "C3 tile attribution",
            f"the shortfall belongs to BLOCK_M={SUBJECT_BLOCK_M}, not to the "
            "fused layer",
            f"control peak - subject peak >= {CONTROL_SEPARATION:.2f} of the roof",
            None, "the two tiles share no multi-tile token count", "",
            ["Nothing cancels when nothing is paired."])
    peak_s = max(p.roof_fraction for p in deep_s if p.tokens in shared)
    peak_c = max(p.roof_fraction for p in deep_c if p.tokens in shared)
    gap = peak_c - peak_s
    lines = [
        f"over the {len(shared)} shared token counts {shared[:6]}: "
        f"BLOCK_M={SUBJECT_BLOCK_M} peaks at {peak_s:.3f} of the roof, "
        f"BLOCK_M={control_block_m} at {peak_c:.3f}, a gap of {gap:+.3f} "
        f"({gap * roof.tflops:+.0f} TFLOP/s)",
        f"the control's own derivative: {control_plateau.line()}",
    ]
    if peak_c >= ROOF_REACHED:
        lines.append(
            f"the control REACHED the roof ({peak_c:.3f} >= "
            f"{ROOF_REACHED:.2f}), so this instrument is demonstrably able to "
            "see a tile reach it, and an absence at the subject is evidence of "
            "absence rather than a shrug.")
    else:
        lines.append(
            f"the control did NOT reach the roof either ({peak_c:.3f} < "
            f"{ROOF_REACHED:.2f}). The gap below is then the whole of the "
            "evidence: what the two tiles share has cancelled, but the height "
            "of the ceiling has not been demonstrated.")
    if control_plateau.plateaued is True and abs(gap) < CONTROL_SEPARATION:
        lines.append(
            "BOTH TILES PLATEAU TOGETHER, at fractions this gate cannot "
            "separate. That is a result: the shortfall is a property of the "
            "fused layer, and whatever binds it is not the tile-dependent AI "
            "ceiling this study proposed.")
    return Gate(
        CLAIM, "C3 tile attribution",
        f"the shortfall belongs to BLOCK_M={SUBJECT_BLOCK_M}, not to the fused "
        "layer both tiles run inside",
        f"control peak - subject peak >= {CONTROL_SEPARATION:.2f} of the roof",
        gap >= CONTROL_SEPARATION, f"gap {gap:+.3f} of the roof", "", lines)


# --------------------------------------------------------------------------
# The verdict, which is the two outcomes the experiment was designed around.
# --------------------------------------------------------------------------

BINDING = "CEILING BINDING AT THE PRODUCTION TILE"
NOT_BINDING = "CEILING NOT BINDING"
STILL_RISING = "STILL RISING AT THE LARGEST BATCH MEASURED"
NOT_TILE = "PLATEAU IS NOT TILE-ATTRIBUTABLE"
UNSETTLED = "NOT SETTLED"


def verdict(gates: list[Gate]) -> tuple[str, list[str]]:
    """One sentence, derived from the gates and from nothing else."""
    by = {g.name.split()[0]: g for g in gates}
    invalid = [g for g in gates if g.kind == VALIDITY and g.passed is not True]
    c1, c2, c3 = by.get("C1"), by.get("C2"), by.get("C3")
    if invalid:
        return UNSETTLED, [
            "The instrument did not qualify: "
            + "; ".join(f"{g.name} ({'FAIL' if g.passed is False else 'UNKNOWN'})"
                        for g in invalid),
            "No claim gate below may be quoted. What each failure invalidates "
            "is printed with it."]
    if c1 is not None and c1.passed is False:
        return NOT_BINDING, [
            f"BLOCK_M={SUBJECT_BLOCK_M} reached the compute roof in the "
            "multi-tile regime, so the arithmetic-intensity ceiling is not "
            "binding where production runs.",
            "The study's central claim is about a regime that does not occur, "
            "and that is this run's result rather than its failure."]
    if c2 is not None and c2.passed is False:
        return STILL_RISING, [
            "Throughput had not stopped rising at the largest batch measured, "
            "so no sentence in this report may contain the word plateau.",
            "This is a statement about the DEPTH of this sweep, not about the "
            "hardware. Re-run with a larger --r-max."]
    if c2 is not None and c2.passed is None:
        return UNSETTLED, [
            "The derivative could not be computed or could not be resolved; "
            "see C2. A plateau read off noise is not a plateau."]
    if c3 is not None and c3.passed is False:
        return NOT_TILE, [
            f"Throughput plateaued below the roof at BLOCK_M={SUBJECT_BLOCK_M}, "
            "and the control tile plateaued with it. What the two tiles share "
            "has cancelled and the gap did not survive.",
            "The shortfall is a property of the fused layer, not of the tile, "
            "and it is not the ceiling this study proposed.",
            "ONE CONFOUND THIS GATE CANNOT REMOVE. The subject and the control "
            "differ in OCCUPANCY as well as in tile: at the production pin of "
            "num_stages=4 the resident-block ratio is 2:1, at 2 stages it is "
            "still 2:1, at 3 it is 3:1, and the only matched setting is 5 "
            "stages, which sm_80 cannot run and which unpins the comparison "
            "from every published arm. So no --num-stages choice disambiguates "
            "a C3 result here. The residency effect is measured DIRECTLY by "
            "scripts/occupancy_vs_swizzle.py, and its size is what to subtract "
            "before reading this verdict as tile-attributable or not."]
    if c1 is not None and c1.passed and c2 is not None and c2.passed and \
            c3 is not None and c3.passed:
        return BINDING, [
            f"At BLOCK_M={SUBJECT_BLOCK_M}, the tile vLLM actually runs "
            "multi-tile, throughput rose and then stopped below this card's own "
            "measured dense rate, while a tile whose cap is far above the ridge "
            "did materially better on the same batches.",
            "The ceiling is real and it binds in the regime production runs. "
            "Nothing in this sentence came from a fit."]
    return UNSETTLED, ["The claim gates did not resolve; read them individually."]


# --------------------------------------------------------------------------
# Predictions, registered and printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def predictions_text(cfg, roof: Roof, b: int, control_block_m: int,
                     rows: list[int], doublings: float) -> str:
    lo, hi = predicted_plateau_band(roof.ridge, b)
    cap_s = SWEEP.ai_cap(SUBJECT_BLOCK_M, max(ALPHA_128_BAND), b)
    cap_c = SWEEP.ai_cap(control_block_m, SWEEP.ALPHA, b)
    onset = onset_tokens(cfg, SUBJECT_BLOCK_M)
    deep = [r for r in rows if r > SUBJECT_BLOCK_M]
    return "\n".join([
        "## Registered predictions, with numbers, before anything is measured",
        "",
        f"  the roof            {roof.tflops:.1f} TFLOP/s dense bf16, "
        f"{'MEASURED on the attached card' if roof.attached else 'HYPOTHESIS'}",
        f"  multi-tile onset    T > {onset} on {cfg.name} "
        f"(r > {SUBJECT_BLOCK_M} rows per expert, from T = r E / k with "
        f"E={cfg.num_experts}, k={cfg.top_k})",
        f"  deepest point       {max(rows)} rows per expert = "
        f"{max(rows) // SUBJECT_BLOCK_M} M-tiles per expert, "
        f"{len(deep)} multi-tile points",
        "",
        f"  P1  the control BLOCK_M={control_block_m} beats the subject by at "
        f"least {CONTROL_SEPARATION:.2f} of the roof",
        f"      ({CONTROL_SEPARATION * roof.tflops:.0f} TFLOP/s). Its cap is "
        f"{cap_c:.0f} Op/B against a ridge of {roof.ridge:.1f}, "
        f"{cap_c / roof.ridge:.1f}x, so it has no ceiling to hit.",
        f"  P2  the subject plateaus between {lo:.3f} and {hi:.3f} of the roof "
        f"({lo * roof.tflops:.0f} to {hi * roof.tflops:.0f} TFLOP/s), which is "
        "what the",
        f"      study's own alpha band at BLOCK_M=128 predicts "
        f"(alpha {min(ALPHA_128_BAND):.3f} to {max(ALPHA_128_BAND):.3f}, cap "
        f"{cap_s:.0f} Op/B at the top). THE BAND SPANS A QUARTER OF THE RANGE",
        "      IT PREDICTS. It is registered so the fit is on the record, and "
        "it is compared against nothing.",
        f"  P3  the subject's gain falls below "
        f"{PLATEAU_GAIN_PER_DOUBLING:.1%} per doubling over the last "
        f"{doublings:g} doublings, by "
        f"n={max(rows) // SUBJECT_BLOCK_M} M-tiles.",
        "  P4  the control does NOT plateau below the roof over the same span.",
        "",
        "  THE FORK. Throughput rises and plateaus below the roof -> the "
        "ceiling is real and binds where",
        "  production runs. Throughput reaches the roof -> the ceiling is not "
        "binding and the study's central",
        "  claim is about a regime that does not occur. Both are results; only "
        "one is the study's.",
    ])


# --------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    model: str
    dtype: str
    pinned: dict
    control_block_m: int
    subject_rows: list[int]
    control_rows: list[int]
    reps: int
    iters: int
    warmup: int
    cell_budget_ms: float
    estimated_seconds: float
    cells: int
    resources: dict
    refusals: dict

    def lines(self, cfg) -> list[str]:
        return [
            f"model        {cfg.name} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       {self.pinned}",
            f"subject      BLOCK_M={SUBJECT_BLOCK_M} at rows {self.subject_rows}",
            f"             tokens "
            f"{[SWEEP.tokens_for_rows(cfg, r) for r in self.subject_rows]}",
            f"control      BLOCK_M={self.control_block_m} at rows "
            f"{self.control_rows} -- the same batches, a tile with no ceiling",
            f"repeats      {self.reps} round-robin passes per tile, so a "
            "throttled cell can be told from a tile effect",
            f"timing       {self.warmup} warmup + up to {self.iters} iters, cut "
            f"to keep a cell inside {self.cell_budget_ms:.0f} ms",
            f"cells        {self.cells} timings",
            f"estimate     {self.estimated_seconds:.0f} s of GPU at the model's "
            "own timings, excluding compiles and allocation",
        ] + [f"resources    {r.render().strip()}"
             for r in self.resources.values()]


def build_plan(args, cfg, b: int, roof: Roof, capability) -> Plan:
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                  BLOCK_SIZE_N=args.block_n, BLOCK_SIZE_K=args.block_k)
    subject_rows = doubling_rows(cfg, args.r_min, args.r_max, SUBJECT_BLOCK_M)
    control_rows = [r for r in subject_rows if r % args.control == 0]
    tiles = (SUBJECT_BLOCK_M, args.control)
    resources, refusals = SWEEP.tile_resource_plan(pinned, tiles, b, capability)
    total = 0.0
    for bm, rows in ((SUBJECT_BLOCK_M, subject_rows), (args.control, control_rows)):
        for r in rows:
            ms = SWEEP.model_ms(cfg, r, bm, alpha=args.alpha, ridge=roof.ridge,
                                bandwidth_gbps=roof.bandwidth_gbps, b=b)
            iters = SWEEP.scaled_iters(ms, args.iters, args.cell_budget_ms)
            total += args.reps * ms * (args.warmup + iters)
    return Plan(args.model, args.dtype, pinned, args.control, subject_rows,
                control_rows, args.reps, args.iters, args.warmup,
                args.cell_budget_ms, total / 1e3,
                args.reps * (len(subject_rows) + len(control_rows)),
                resources, refusals)


def check_control(control_block_m: int, alpha: float, ridge: float, b: int
                  ) -> str:
    """Empty when this tile may serve as a control, else why it may not.

    The ONE place a fit enters the design. A control has to be a tile with no
    ceiling of its own, and "no ceiling" is a statement about `cap / ridge`,
    which needs an alpha. It is checked here, before any GPU time, and it
    decides which tile is measured -- never what the measurement means.
    """
    if control_block_m == SUBJECT_BLOCK_M:
        return (f"--control {control_block_m} is the SUBJECT. A control has to "
                "be a different tile, or the difference C3 measures is zero by "
                "construction.")
    if control_block_m < SUBJECT_BLOCK_M:
        return (f"--control {control_block_m} is below the subject's "
                f"{SUBJECT_BLOCK_M}, so its cap is LOWER and its own ceiling is "
                "tighter. A control must have more headroom than the subject, "
                "not less.")
    cap = SWEEP.ai_cap(control_block_m, alpha, b)
    if cap < ridge * CONTROL_CAP_MARGIN:
        return (f"--control {control_block_m} caps at {cap:.1f} Op/B against a "
                f"ridge of {ridge:.1f}, only {cap / ridge:.2f}x, under the "
                f"{CONTROL_CAP_MARGIN:.2f}x a control needs. Its own ceiling "
                "would be part of the argument it is supposed to settle.")
    return ""


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def measure_setting(args, cfg, block_m: int, rows: list[int], csv_path: Path,
                    cache_root: Path, pinned: dict, done, timings: list[Timing]
                    ) -> tuple[int, int]:
    """Time one tile, `--reps` round-robin passes over its batches.

    ROUND ROBIN INSIDE THE TILE, not batch-by-batch to completion. Running the
    small batch fifty times and then the large batch fifty times puts every
    point at a different place in the pod's thermal history, and the resulting
    monotone drift IS the derivative this script measures. One pass over the
    whole grid per repeat spreads that drift across the curve instead of
    aligning it with the x axis.

    The clocks are sampled immediately either side of the RECORDED timing, not
    around the compile and the calibration call as well, so the drift on a row
    is the drift during the milliseconds that row reports.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.bench.timing import ClockState, clock_drift
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
            tiles = SWEEP.tiles_per_expert(r, block_m)
            if (block_m, r, rep) in done:
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
                    start = ClockState.sample()
                    ms, mn, sd = SWEEP.time_call(call, args.warmup, iters)
                    end = ClockState.sample()
                drift, throttled = clock_drift(start, end)
                row = Timing(block_m, r, tiles, tokens, rep, ms, mn, sd, iters,
                             start.sm_clock_mhz, end.sm_clock_mhz,
                             start.temp_c, end.temp_c, drift, throttled)
            except Exception as exc:                    # noqa: BLE001
                row = Timing(block_m, r, tiles, tokens, rep, 0.0, 0.0, 0.0, 0,
                             status="failed",
                             detail=f"{type(exc).__name__}: {exc}")
                print(f"  BM={block_m} r={r} rep={rep} FAILED {row.detail}")
                if "shared memory" in str(exc).lower():
                    print(f"  ^ re-run the WHOLE experiment with --num-stages "
                          f"{max(1, pinned['num_stages'] - 1)}. Dropping stages "
                          "for one tile alone would unpin the comparison, which "
                          "is the only thing C3 has.")
            timings.append(row)
            append_timing(csv_path, row)
            print(f"  BM={block_m:3d} r={r:5d} n={tiles:3d} T={tokens:7d} "
                  f"rep={rep:2d}  {row.ms_p50:9.4f} ms  "
                  f"{row.sm_clock_start_mhz:4d}->{row.sm_clock_end_mhz:4d} MHz"
                  + ("  THROTTLED" if row.throttled else ""))
    return compiles, executed


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

def figure_rows(series: list[Series], roof: Roof, card: str) -> list[dict]:
    """The plot's data, one row per point, self-describing.

    The roof, its source and the card ride on EVERY row. A figure gets drawn
    months later on a laptop from a committed CSV, and a fraction whose
    denominator is not in the file is a fraction that will be redrawn against
    whatever roof the plotting script happens to load.
    """
    out = []
    for s in series:
        for p in s.points:
            out.append({
                "card": card, "roof_tflops": f"{roof.tflops:.4f}",
                "roof_source": roof.source, "roof_clock_mhz": roof.clock_mhz,
                "block_m": p.block_m, "tokens": p.tokens,
                "rows_per_expert": p.rows_per_expert, "tiles": p.tiles,
                "regime": p.regime, "tile_eff": f"{p.tile_eff:.6f}",
                "reps": p.reps, "ms_p50": f"{p.ms_p50:.6f}",
                "ms_spread_rel": "" if p.spread is None else f"{p.spread:.6f}",
                "useful_tflops": f"{p.useful_tflops:.4f}",
                "roof_fraction": f"{p.roof_fraction:.6f}",
                "sm_clock_mhz": p.sm_clock_mhz,
                "throttled_reps": p.throttled_reps,
                "retained": p.retained, "excluded_why": p.excluded_why,
            })
    return out


FIGURE_FIELDS = ["card", "roof_tflops", "roof_source", "roof_clock_mhz",
                 "block_m", "tokens", "rows_per_expert", "tiles", "regime",
                 "tile_eff", "reps", "ms_p50", "ms_spread_rel", "useful_tflops",
                 "roof_fraction", "sm_clock_mhz", "throttled_reps", "retained",
                 "excluded_why"]


def write_figure_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIGURE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyse(timings: list[Timing], cfg, roof: Roof, *, control_block_m: int,
            b: int, sm_count: int, block_n: int, doublings: float,
            compiles: dict[int, int], executed: dict[int, int],
            planned_multi_tile: int) -> tuple[list[str], list[Gate], dict,
                                              list[Series]]:
    """Everything that is read off the timings, and nothing that is not."""
    clock_ref = modal_clock(timings)
    subject = build_points(timings, cfg, SUBJECT_BLOCK_M, roof,
                           sm_count=sm_count, block_n=block_n,
                           clock_ref=clock_ref)
    control = build_points(timings, cfg, control_block_m, roof,
                           sm_count=sm_count, block_n=block_n,
                           clock_ref=clock_ref)
    series = [Series(f"BLOCK_M={control_block_m} (control)", "o", control),
              Series(f"BLOCK_M={SUBJECT_BLOCK_M} (subject)", "#", subject)]
    sub_plateau = plateau_of(multi_tile(subject), doublings=doublings)
    ctl_plateau = plateau_of(multi_tile(control), doublings=doublings)
    predicted = predicted_plateau_band(roof.ridge, b)

    gates = [
        gate_v0_roof(roof),
        gate_v1_pin(compiles, executed, (SUBJECT_BLOCK_M, control_block_m)),
        gate_v2_non_vacuity(subject, control, planned_multi_tile,
                            onset_tokens_value=onset_tokens(cfg, SUBJECT_BLOCK_M)),
        gate_v3_clocks(timings, subject + control, roof, clock_ref),
        gate_v4_control_ran(control, subject, control_block_m),
        gate_c1_roof(subject, sub_plateau, roof, predicted),
        gate_c2_plateau(sub_plateau, subject),
        gate_c3_attribution(subject, control, ctl_plateau, control_block_m, roof),
    ]
    call, why = verdict(gates)

    lines = ["", "## The curve", ""] + ascii_plot(series, roof) + ["", ""] \
        + point_table(series) + ["", "## The derivative", "",
                                 f"  subject BLOCK_M={SUBJECT_BLOCK_M}: "
                                 f"{sub_plateau.line()}",
                                 f"  control BLOCK_M={control_block_m}: "
                                 f"{ctl_plateau.line()}",
                                 "", "## Verdict", "", f"  {call}", ""] \
        + [f"  {w}" for w in why]

    payload = {
        "verdict": call,
        "verdict_why": why,
        "roof": asdict(roof),
        "clock_ref_mhz": clock_ref,
        "predicted_plateau_band": list(predicted),
        "predicted_band_is_scored_against":
            "nothing. It is the study's own arithmetic, pre-registered so the "
            "fit is on the record. Every verdict here comes from the stopwatch.",
        "subject_plateau": asdict(sub_plateau),
        "control_plateau": asdict(ctl_plateau),
        "points": figure_rows(series, roof, roof.device),
        "gates": [asdict(g) | {"verdict": {True: "PASS", False: "FAIL",
                                           None: "UNKNOWN"}[g.passed]}
                  for g in gates],
    }
    return lines, gates, payload, series


# --------------------------------------------------------------------------
# Self test: plant three worlds, check the gates tell them apart.
# --------------------------------------------------------------------------

def planted_timings(cfg, roof: Roof, b: int, rows_by_tile: dict[int, list[int]],
                    *, alpha: float, overhead_ms: float, reps: int,
                    noise: float, seed: int, clock_mhz: int = 1500,
                    throttle: tuple[int, int] | None = None) -> list[Timing]:
    """Timings generated FROM the study's own model, so the gates have an answer.

    `throttle` plants ONE (block_m, rows) cell with a collapsed end clock, which
    is how the exclusion machinery gets exercised without a pod that overheats
    on demand.
    """
    rng = random.Random(seed)
    out: list[Timing] = []
    for block_m, rows in rows_by_tile.items():
        for rep in range(1, reps + 1):
            for r in rows:
                ms = SWEEP.model_ms(cfg, r, block_m, alpha=alpha,
                                    ridge=roof.ridge,
                                    bandwidth_gbps=roof.bandwidth_gbps, b=b,
                                    overhead_ms=overhead_ms)
                if noise:
                    ms *= math.exp(rng.gauss(0.0, noise))
                end = clock_mhz
                if throttle == (block_m, r):
                    end = int(clock_mhz * 0.80)
                drift = (clock_mhz - end) / clock_mhz * 100.0
                out.append(Timing(
                    block_m, r, SWEEP.tiles_per_expert(r, block_m),
                    SWEEP.tokens_for_rows(cfg, r), rep, ms, ms, ms * noise,
                    0, clock_mhz, end, 50, 50, drift,
                    drift > THROTTLE_DRIFT_PCT))
    return out


#: The three worlds, their planted parameters, and the gate verdict each MUST
#: produce. A gate that answers the same in every world cannot settle this
#: experiment, which is the only thing this self test is for.
SELF_TEST_WORLDS = (
    # cap(128) = 128 Op/B against a ridge of ~163: the memory branch binds at
    # every tread and the curve is flat below the roof. cap(256) = 256 Op/B is
    # above the ridge, so the control is compute bound and reaches it.
    ("capped     alpha 1.00", 1.00, 0.05, "C2", True),
    # The retracted world. cap(128) = 1280 Op/B, ten times the H200 ridge, so
    # nothing is memory bound and the subject reaches the roof.
    ("uncapped   alpha 0.10", 0.10, 0.05, "C1", False),
    # A fused layer with a 5 ms fixed cost. Nothing is capped; throughput climbs
    # toward the roof and is still climbing at the deepest point, which is what
    # a derivative gate has to be able to say.
    ("overhead   alpha 0.10, D = 5 ms", 0.10, 5.00, "C2", False),
)


def self_test(cfg, roof: Roof, b: int, *, r_min: int, r_max: int,
              control_block_m: int, doublings: float, reps: int = 3,
              noise: float = 0.002, seed: int = 0
              ) -> tuple[list[str], list[Gate]]:
    """Three planted worlds and the verdicts they must produce.

    The point is not that the code runs; it is that the gates DISCRIMINATE. Each
    world is named with the gate it is planted to move, and a world whose gate
    agrees with another world's is a world this test would have missed.
    """
    subject_rows = doubling_rows(cfg, r_min, r_max, SUBJECT_BLOCK_M)
    control_rows = [r for r in subject_rows if r % control_block_m == 0]
    grid = {SUBJECT_BLOCK_M: subject_rows, control_block_m: control_rows}
    planned = len(subject_rows)
    planned_multi = len([r for r in subject_rows if r > SUBJECT_BLOCK_M])
    compiles = {SUBJECT_BLOCK_M: 1, control_block_m: 1}
    executed = {SUBJECT_BLOCK_M: planned, control_block_m: len(control_rows)}

    out = ["", "## Self test: planted worlds, real gates", "",
           "  Every world below runs on the HYPOTHESIS roof, so V0 fails in all "
           "three and the real",
           "  verdict is NOT SETTLED in all three -- which is the point of V0 "
           "and is asserted below.",
           "  The `would-be` column is `verdict()` applied to the CLAIM gates "
           "alone, i.e. what this",
           "  run would have concluded had it been measured on a calibrated "
           "card.", "",
           f"{'world':34s} {'subject peak':>12s} {'gain/doubling':>14s} "
           f"{'C1':>8s} {'C2':>8s} {'C3':>8s}  would-be verdict"]
    gates: list[Gate] = []
    verdicts: set[str] = set()
    triples: set[tuple] = set()
    real_verdicts: set[str] = set()
    for name, alpha, overhead, moves, expect in SELF_TEST_WORLDS:
        timings = planted_timings(cfg, roof, b, grid, alpha=alpha,
                                  overhead_ms=overhead, reps=reps, noise=noise,
                                  seed=seed)
        _, world_gates, payload, series = analyse(
            timings, cfg, roof, control_block_m=control_block_m, b=b,
            sm_count=SWEEP.DEFAULT_SM_COUNT, block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
            doublings=doublings, compiles=compiles, executed=executed,
            planned_multi_tile=planned_multi)
        by = {g.name.split()[0]: g for g in world_gates}
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}
        subject = [s for s in series if s.marker == "#"][0]
        peak = max((p.roof_fraction for p in multi_tile(subject.points)),
                   default=0.0)
        plateau = payload["subject_plateau"]["gain_per_doubling"]
        # `verdict()` over the CLAIM gates alone: what this world would have
        # concluded on a calibrated card. The real verdict is collected too,
        # because "a hypothesis roof cannot produce one" is itself an invariant.
        would_be, _ = verdict([g for g in world_gates if g.kind == CLAIM])
        out.append(
            f"{name:34s} {peak:12.3f} "
            + (f"{plateau:+13.2%} " if plateau is not None else f"{'n/a':>14s}")
            + f"{tag[by['C1'].passed]:>8s} {tag[by['C2'].passed]:>8s} "
              f"{tag[by['C3'].passed]:>8s}  {would_be}")
        verdicts.add(would_be)
        real_verdicts.add(payload["verdict"])
        triples.add((by["C1"].passed, by["C2"].passed, by["C3"].passed))
        got = by[moves].passed
        gates.append(Gate(
            VALIDITY, f"S {name.split()[0]}",
            f"the planted world moves {moves} to "
            f"{'PASS' if expect else 'FAIL'}",
            f"{moves} verdict is {'PASS' if expect else 'FAIL'}", got is expect,
            f"{moves} came out {tag[got]}",
            "the self test itself: gates that answer the same in every world "
            "cannot settle this experiment"))

    # NON-VACUITY over the worlds themselves. Three planted worlds that all
    # produce one reading prove nothing about the instrument, however many of
    # them there are. Scored on the claim-gate TRIPLE as well as on the verdict,
    # because a verdict function that collapsed three different triples onto one
    # word would pass a check that only looked at the word.
    gates.append(Gate(
        VALIDITY, "S discrimination",
        "the worlds do not all land on one reading",
        f"{len(SELF_TEST_WORLDS)} distinct (C1, C2, C3) triples and "
        f"{len(SELF_TEST_WORLDS)} distinct would-be verdicts across "
        f"{len(SELF_TEST_WORLDS)} worlds",
        len(triples) == len(SELF_TEST_WORLDS)
        and len(verdicts) == len(SELF_TEST_WORLDS),
        f"{len(triples)} distinct triples, {len(verdicts)} distinct verdicts: "
        f"{sorted(verdicts)}",
        "the self test itself"))

    # AND THE REFUSAL. A synthetic run stands on a roof no device produced, so
    # every world's REAL verdict must be NOT SETTLED however clean its claim
    # gates look. This is what stops a laptop report being quotable.
    gates.append(Gate(
        VALIDITY, "S hypothesis roof refused",
        "a run on a roof no attached device measured reaches no verdict",
        f"every world's real verdict is {UNSETTLED!r}",
        real_verdicts == {UNSETTLED},
        f"real verdicts: {sorted(real_verdicts)}",
        "the self test itself: if a hypothesis roof could produce a verdict, "
        "every --dry-run and --self-test on a laptop would be quotable"))

    # And the exclusion machinery, which no world above exercises: plant one
    # throttled cell and check it leaves the gated set rather than being
    # averaged into it.
    timings = planted_timings(cfg, roof, b, grid, alpha=1.0, overhead_ms=0.05,
                              reps=1, noise=0.0, seed=seed,
                              throttle=(SUBJECT_BLOCK_M, r_max))
    points = build_points(timings, cfg, SUBJECT_BLOCK_M, roof,
                          sm_count=SWEEP.DEFAULT_SM_COUNT,
                          block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
                          clock_ref=modal_clock(timings))
    dropped = [p for p in points if not p.retained]
    gates.append(Gate(
        VALIDITY, "S throttle exclusion",
        "a throttled cell leaves the gated set",
        "the one planted throttled point is the one excluded point",
        len(dropped) == 1 and dropped[0].rows_per_expert == r_max,
        f"{len(dropped)} excluded: "
        f"{[p.rows_per_expert for p in dropped]} rows per expert",
        "the self test itself: an exclusion path never exercised is an "
        "exclusion path that does not work"))
    return out, gates


# --------------------------------------------------------------------------
# Output paths, and whether git will keep them.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file.

    `.gitignore` ignores `results/*` and re-includes only `results/published/`,
    so a run that writes anywhere else under the repo produces files `git add -A`
    silently drops. This project has already lost every published plot that way,
    and the figure CSV this script writes is exactly such a file. Checked with
    `git check-ignore` rather than by re-implementing the pattern rules, because
    the pattern rules are what got it wrong.
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
                "results/published/<date>-<gpu>-bm128-roofline")
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
    return re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name(0).lower()
                  ).strip("_")


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter AND the card, so two runs cannot collide.

    This repo has lost an arm to this twice. Once to a run id that omitted
    GROUP_SIZE_M -- the second run resumed into the first's directory, found
    every cell present, skipped all of them, and printed the first run's timings
    under the second's heading. Once to a run id that omitted THE CARD, and the
    proof is in the tree: two published directories, one A100 and one H200,
    contain a report file of the same name.

    The card is the sharpest of them here, because every number this script
    prints is a fraction of a PER-CARD measured roof -- 712 TFLOP/s on the H200
    against a different figure on the A100 -- and `results_root()` prefers a
    network volume that outlives the pod.

    `--alpha` stays OUT: it selects the control and prints a prediction, and it
    re-reads a set of timings rather than changing one, so two analyses of one
    sweep belong in one directory.
    """
    key = json.dumps({"card": card, "model": args.model, "dtype": args.dtype,
                      "r_min": args.r_min, "r_max": args.r_max,
                      "reps": args.reps, "iters": args.iters,
                      "warmup": args.warmup, "budget": args.cell_budget_ms,
                      "seed": args.seed, "group_m": args.group_m,
                      "block_n": args.block_n, "block_k": args.block_k,
                      "num_stages": args.num_stages, "num_warps": args.num_warps,
                      "subject": SUBJECT_BLOCK_M, "control": args.control},
                     sort_keys=True)
    return (f"{card}-{args.model}-{args.dtype}-r{args.r_min}_{args.r_max}"
            f"-c{args.control}-g{args.group_m}-n{args.block_n}-k{args.block_k}"
            f"-s{args.num_stages}-w{args.num_warps}-x{args.reps}-"
            f"{hashlib.sha1(key.encode()).hexdigest()[:6]}")


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 puts the multi-tile onset "
                         "at T=512 and 32 M-tiles per expert at T=16384, which "
                         "is the depth the one tile-recording arm reached")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="not fp8: halving the weight bytes doubles the AI cap "
                         "and moves the question off the regime the study is "
                         "about, and the fp8 call path needs a quant config "
                         "this script does not build")
    ap.add_argument("--control", type=int, default=DEFAULT_CONTROL_BLOCK_M,
                    help="the positive control tile. Must be larger than the "
                         "subject and cap at least "
                         f"{CONTROL_CAP_MARGIN:.2f}x the ridge, or it is "
                         "refused. 256 needs 160 KiB of shared memory at 4 "
                         "stages, which fits sm_90 and fits sm_80 by 3 KiB; "
                         "--control 128 --num-stages 3 is refused because 128 "
                         "IS the subject")
    ap.add_argument("--r-min", type=int, default=DEFAULT_R_MIN,
                    help="smallest rows per expert. A power of two, and below "
                         "BLOCK_M so the sweep starts in the single-partial-tile "
                         "regime TEMPO describes")
    ap.add_argument("--r-max", type=int, default=DEFAULT_R_MAX,
                    help="largest rows per expert. 4096 is 32 M-tiles per "
                         "expert at BLOCK_M=128, the depth the observed arm "
                         "reaches. Raise it if C2 says STILL RISING")
    ap.add_argument("--plateau-doublings", type=float,
                    default=DEFAULT_PLATEAU_DOUBLINGS,
                    help="doublings of the batch the plateau gate measures its "
                         "gain over. Two: one doubling of a noisy pair is a "
                         "difference, two is a direction")
    ap.add_argument("--reps", type=int, default=3,
                    help="round-robin passes per tile. Below 2 there is no "
                         "spread, and with no spread the plateau gate cannot "
                         "say whether it resolves 2%% -- so it says UNKNOWN")
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"],
                    help="the swizzle width, applied to BOTH tiles. 1 is what "
                         "vLLM's fallback ladder holds across the decode range, "
                         "so it is what a deployment without a tuned file runs")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"],
                    help="the N tile, applied to BOTH tiles. It is in the run "
                         "id because it changes the shared-memory bill and the "
                         "activation re-read ratio")
    ap.add_argument("--block-k", type=int, default=SWEEP.FIXED["BLOCK_SIZE_K"])
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"],
                    help="pipeline stages, applied to BOTH tiles. Lower it here "
                         "and both move together; lowering it for one tile "
                         "alone unpins the comparison C3 rests on")
    ap.add_argument("--num-warps", type=int, default=SWEEP.FIXED["num_warps"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="iterations are cut so one cell stays inside this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sm-count", type=int, default=0,
                    help="0 asks the driver; only needed off-GPU")
    ap.add_argument("--capability", default="",
                    help="compute capability as MAJOR.MINOR, e.g. 8.0 for the "
                         "A100 or 9.0 for the H200. Empty asks the driver; give "
                         "it off-GPU to get the shared-memory verdict for the "
                         "control tile in the plan")
    ap.add_argument("--alpha", type=float, default=SWEEP.ALPHA,
                    help="used ONLY to cost the run and to check that the "
                         "control tile has headroom. No verdict is computed "
                         "from it")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--card", default="",
                    help="card slug the run id is built from. Read from the "
                         "attached device by default and REFUSED if it "
                         "contradicts one; its only use is printing a pod's "
                         "real path from a laptop dry run")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--require-git-visible", action="store_true",
                    help="refuse to run when the output path is git-ignored")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, then stop")
    ap.add_argument("--self-test", action="store_true",
                    help="plant three worlds from the study's own model and "
                         "check the gates tell them apart, off GPU")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero unless every gate passes. Off by "
                         "default: this experiment has two outcomes and only "
                         "one of them is the study's, so a FAIL here is a "
                         "result and not an error")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    synthetic = args.dry_run or args.self_test

    try:
        roof = resolve_roof(args.dtype, synthetic=synthetic)
    except RoofUnavailable as exc:
        print(f"REFUSED: {exc}")
        return 2

    refusal = check_control(args.control, args.alpha, roof.ridge, b)
    if refusal:
        print(f"REFUSED: {refusal}")
        return 2

    capability = SWEEP.resolve_capability(args, synthetic=synthetic)
    plan = build_plan(args, cfg, b, roof, capability)
    if plan.refusals:
        print("REFUSED before any GPU time, from the pinned constants alone:")
        for bm, why in sorted(plan.refusals.items()):
            print(f"  BLOCK_M={bm}: {why}")
        print("A spilled or oversized kernel still returns a time, and that "
              "time still plots. Change --num-stages, --block-n or --control.")
        return 2

    lines = [f"experiment  bm128_roofline: at BLOCK_M={SUBJECT_BLOCK_M}, does "
             "achieved throughput plateau BELOW", "            this card's own "
             "measured dense bf16 rate, in the multi-tile regime?", "",
             predictions_text(cfg, roof, b, args.control, plan.subject_rows,
                              args.plateau_doublings)]

    detected = detect_card_slug()
    card = args.card or detected or UNKNOWN_CARD_SLUG
    if args.card and detected and args.card != detected:
        print("\n".join(lines))
        print(f"\nREFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return 2
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bm128_roofline" / run_id
    csv_path = out_dir / "cells.csv"
    figure_path = out_dir / "figure.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    lines += ["", "## The plan", ""] + plan.lines(cfg) + roof.lines() + [
        f"card         {card}" + ("" if detected else
                                  f"  (NO DEVICE ATTACHED: the id above is the "
                                  f"{UNKNOWN_CARD_SLUG!r} one and is not what a "
                                  "pod derives; pass --card <slug> for that)"),
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per repeat, flushed), figure.csv (the "
        "plot's data), CARD, report.txt, report.json, triton-cache/"]

    if args.self_test:
        more, gates = self_test(cfg, roof, b, r_min=args.r_min, r_max=args.r_max,
                                control_block_m=args.control,
                                doublings=args.plateau_doublings,
                                reps=max(2, args.reps), seed=args.seed)
        lines += more + ["", "## Gates", ""] + render_gates(gates)
        print("\n".join(lines))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    if args.dry_run:
        lo, hi = predicted_plateau_band(roof.ridge, b)
        lines += ["", "## What is settled before the run, and what is not", "",
                  f"  The control's cap is "
                  f"{SWEEP.ai_cap(args.control, args.alpha, b) / roof.ridge:.2f}x"
                  f" the ridge, so it has no ceiling of its own to hit.",
                  f"  The subject's predicted plateau spans {hi - lo:.3f} of "
                  "the roof, which is why this run measures instead of",
                  "  computing. Nothing below the onset is evidence about the "
                  "re-read term: there is one tile there.",
                  "",
                  "  Invocation for a session script:",
                  f"    python scripts/bm128_roofline.py --model {args.model} "
                  f"--dtype {args.dtype} --control {args.control} \\",
                  f"        --r-min {args.r_min} --r-max {args.r_max} --reps "
                  f"{args.reps} --plateau-doublings "
                  f"{args.plateau_doublings:g}",
                  "  Without --fail-on-gate, deliberately: this experiment has two",
                  "  outcomes and a C1 FAIL (the tile REACHED the roof) is one of",
                  "  them. Gate the arm on exit code 2 and the ## Verdict line."]
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
        print(f"\n{missing.split('.')[0]}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --self-test  three planted worlds, checking the gates "
              "discriminate\n"
              "  --dry-run    the pod plan, the grid, the predictions and the "
              "cost")
        return 2

    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    print("\n".join(lines))

    # THE RESUME GUARD, belt as well as braces. The card is already in the run
    # id, so another card lands in another directory and cannot normally reach
    # this cells.csv. This catches the ways it could anyway: an explicit --out
    # or --run-id aiming two cards at one place, or a directory copied between
    # pods. It REFUSES rather than starting over, because silently discarding
    # measured cells is its own way to lose an arm.
    if csv_path.exists():
        written_by = card_path.read_text().strip() if card_path.exists() else ""
        if written_by != card:
            raise SystemExit(
                f"REFUSED to resume {csv_path}: written by card "
                f"{written_by or '<unrecorded>'!r} and this run is {card!r}. "
                "Resuming would divide one card's timings by the other's "
                "measured roof. Move or delete that directory deliberately. "
                "Nothing measured.")
    card_path.write_text(card + "\n")

    sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
    done, timings = read_timings(csv_path)
    compiles: dict[int, int] = {}
    executed: dict[int, int] = {}
    started = time.time()

    # THE CONTROL FIRST. It is the cheaper half (five batches against eight) and
    # it is the half that decides whether the subject's numbers can be read at
    # all: if the control cannot be pinned or cannot run, the subject's cells
    # buy nothing, and finding that out after paying for them is the expensive
    # order.
    print(f"\n-- control tile, BLOCK_M={args.control} --")
    c, e = measure_setting(args, cfg, args.control, plan.control_rows, csv_path,
                           cache_root, plan.pinned, done, timings)
    compiles[args.control], executed[args.control] = c, e

    print(f"\n-- subject tile, BLOCK_M={SUBJECT_BLOCK_M} --")
    c, e = measure_setting(args, cfg, SUBJECT_BLOCK_M, plan.subject_rows,
                           csv_path, cache_root, plan.pinned, done, timings)
    compiles[SUBJECT_BLOCK_M], executed[SUBJECT_BLOCK_M] = c, e
    print(f"\nmeasured in {time.time() - started:.0f} s")

    more, gates, payload, series = analyse(
        timings, cfg, roof, control_block_m=args.control, b=b,
        sm_count=sm_count, block_n=args.block_n,
        doublings=args.plateau_doublings, compiles=compiles, executed=executed,
        planned_multi_tile=len([r for r in plan.subject_rows
                                if r > SUBJECT_BLOCK_M]))
    payload["gpu"] = torch.cuda.get_device_name(0)
    payload["run_id"] = run_id
    write_figure_csv(figure_path, figure_rows(series, roof, card))

    text = "\n".join(lines + more + ["", "## Gates", ""] + render_gates(gates))
    print("\n".join(more + ["", "## Gates", ""] + render_gates(gates)))
    (out_dir / "report.txt").write_text(text)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path), ("figure", figure_path),
                        ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        # Every path, not just the directory: `.gitignore` re-includes
        # `results/published/` under a blanket `results/*` exclusion, and this
        # repo has already lost every published figure to a pattern that matched
        # at a depth nobody checked.
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
