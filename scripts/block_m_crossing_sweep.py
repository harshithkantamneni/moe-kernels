#!/usr/bin/env python
"""Which BLOCK_SIZE_M can reach the compute roof at all, and where does it cross?

    python scripts/block_m_crossing_sweep.py --self-test 0.558   # no GPU needed
    python scripts/block_m_crossing_sweep.py --self-test 0.10    # the retracted world
    python scripts/block_m_crossing_sweep.py --self-test 0.90    # nothing crosses
    python scripts/block_m_crossing_sweep.py --self-test-world low-clock
    python scripts/block_m_crossing_sweep.py --dry-run           # print the grid and stop
    python scripts/block_m_crossing_sweep.py                     # the pod run
    python scripts/block_m_crossing_sweep.py --model qwen2-57b-a14b --r-max 1536

WHY THIS EXISTS AND WHY `scripts/tile_sweep.py` CANNOT ANSWER IT. That script
runs deepseek-v3 at T=16/64/256, which is 0.5 to 8 rows per expert. At 0.5 to 8
rows EVERY expert is one M-tile at EVERY BLOCK_SIZE_M, so the tile count is
identical across the whole sweep, no weight re-read can be saved, and the only
thing a bigger tile can move is occupancy. It measured bigger tiles as slower,
and that measurement is sound -- of the single-tile regime. It says nothing
about the regime this study's claims live in. THIS sweep runs the MULTI-TILE
regime, where `ceil(rows_per_expert / BLOCK_M)` actually varies with BLOCK_M, and
sweeps far enough to bracket a crossing that is at a DIFFERENT batch for each
block size.

THE MODEL BEING FALSIFIED. One expert holding `r` rows is scheduled as
`n = ceil(r / BLOCK_M)` M-tiles. The first tile reads that expert's weights in
full; each extra tile re-reads them, discounted by L2 to a fraction `alpha`:

    Q(n) = 1 + alpha (n - 1)        AI(r) = (2 r / b) / Q(n)

Two consequences, and they are the whole experiment:

  * AI is BOUNDED, and the bound is NOT `2 BM / (alpha b)` for a fitted alpha.
    A ladder fit returns `alpha_fitted = (alpha_b + phi) / (1 + phi + delta)`
    (`moe.bench.ai_model`, form EXA), so a cap read as `2 BM / (alpha_fitted
    b)` is HIGH by `(1 + phi + delta)`: about 1.3x at BLOCK_M=128 on mixtral
    at BLOCK_N=64. The cap this file prints beside each block size is that
    uncorrected reading, labelled with its overstatement factor. And because
    `alpha_a` is unmeasured, any cap quoted here is a BRACKET, not a point. A
    block size whose corrected cap sits below the hardware ridge cannot be
    compute bound at any batch; whether 128's does is what the arm measures.
  * the crossing solves `R = ridge b Q(R) / 2`, a step function on both sides,
    so it moves in jumps of `Q` as the block size changes which tread it lands
    in.

ALPHA WAS REFIT FROM 0.10 TO 0.558 (90% band 0.529-0.588, 10,813 rows, placebo
-0.002). The 0.10 was an estimator artefact. The two values are not a
quantitative disagreement, they are two different worlds, and this sweep is
built to tell them apart:

    BLOCK_M   AI cap    alpha=0.558                 alpha=0.10 (retracted)
       32       57.3    NO CROSSING EVER            crosses at R=309.3
       64      114.7    NO CROSSING EVER            crosses at R=211.7
      128      229.4    crosses at R=253.7          crosses at R=179.1
      256      458.8    crosses at R=162.8          crosses at R=162.8

at the H200's own ridge of 162.8 FLOP/byte (`measured_nvidia_h200.yaml`,
712.3 TFLOP/s over 4374.8 GB/s; `predict_tile(bm, alpha, 162.81)`, recomputed
2026-09-03). The "AI cap" column is the uncorrected `2 BM / (alpha b)` reading,
high by `(1 + phi + delta)` as above. This table used to be scored at 160.3,
the low end of a withdrawn band (md5 4d84542b) that was no card's ridge. At
0.558 the 128-to-256 crossing ratio is 1.558 and two of the four block sizes
never cross. At 0.10 the ratio is 1.10 and all four cross. Everything below is
arranged so the data has to pick one.

WHAT IS PINNED AND WHY. `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`,
`num_warps` and `num_stages` are held at the values `tile_sweep.py` pins, so the
two experiments are comparable and so BLOCK_SIZE_M is the only thing that can
move. GROUP_SIZE_M=1 in particular is load bearing twice over: it is the swizzle
setting whose L2 behaviour the refit's GROUP_SIZE_M=1 slice describes
(alpha 0.570 there against 0.558 pooled, a 2% difference), and leaving it free
would let the swizzle change the very re-read fraction being measured.

WHAT IS NOT PINNED AND CANNOT BE. `alpha` itself drifts with BLOCK_M -- 0.466 at
64, 0.625 at 128 -- which is what a swizzle-for-L2-reuse mechanism predicts and
which this sweep MEASURES per block size rather than assuming. The drift does not
rescue the retracted value: at 0.466 the BLOCK_M=64 cap is 137.3, still below the
ridge, so gate 4's prediction survives its own worst case.

ROUTING IS EXACTLY BALANCED, not sampled uniform, and that is a design decision
rather than a convenience. Sampled uniform routing at T=1024 on mixtral puts
about 15 rows of spread on a mean of 256, which smears every tile step across
+/-60 tokens and makes gate 1 unfalsifiable. `realize_counts` builds an
assignment whose per-expert histogram is EXACTLY `T k / E`, so rows per expert is
an integer the script knows in advance, tile steps land on a token count that can
be named before the run, and the sub-saturation hazard that forces
`crossing.py` to carry a `min_tokens` floor cannot arise: every expert holds the
same number of rows at every point of the grid.

WHAT IT WRITES, AND WHERE IT SURVIVES TEARDOWN. Everything lands under
`$MOE_RESULTS_DIR`, or `/workspace/results` when that exists (the RunPod network
volume, which outlives the pod), or `<repo>/results` on a laptop:

    <results>/block_m_crossing/<run-id>/cells.csv     one row per (BLOCK_M, T)
    <results>/block_m_crossing/<run-id>/report.json   predictions and gate verdicts
    <results>/block_m_crossing/<run-id>/report.txt    exactly what was printed
    <results>/block_m_crossing/<run-id>/triton-cache/ per-setting compile evidence

The exact path is printed at start AND at the end. `cells.csv` is appended and
flushed cell by cell, and a re-run with the same arguments resumes it, so
aborting costs only the cell in flight. The default run id is derived from the
arguments, so "the same experiment" means "the same directory" without anyone
having to remember an id.

THE COMPILE ASSAY, which is a gate and not a detail. If `override_config` fails
to take effect, all four settings run the SAME kernel, every gate reads a
difference of zero, and the report looks like a clean null result. The assay
against that is to count the Triton artefacts that appear during each setting: a
setting that compiled nothing new either did not change the kernel or was served
from a warm cache. Both are fatal to the experiment and both are silent
otherwise -- a stale cache is what cost this project its A100 PTX dump -- so
`TRITON_CACHE_DIR` is pointed at a fresh directory under the results path before
vLLM is imported, and at a per-setting subdirectory before each setting runs.
Gate 0 refuses to let the other four be read if a setting that ran cells
compiled nothing. A setting that ran NO cells, because a previous session
already measured them, is a different state: the assay belongs to that session
and gate 0 says so rather than scoring it.

ONE INSTRUMENT, AND WHAT IT COSTS TO HAVE HAD TWO. Until 2026-09-02 every cell
here was timed by a private `time_call` that created its CUDA events inside the
loop, recorded the start event on a stream it had just synchronised, and
synchronised again after every iteration, with no L2 flush and no clock read.
The roof every one of those cells is scored against was measured queue-deep
(`moe/bench/timing.time_eager`). The audit bounded the host prefix the old loop
exposed at 0.18 ms per fused_experts call on the H200 and 0.30 ms on the A100,
a bias in alpha of 8-16% at the smallest ladder cells and DIFFERENT PER CARD,
which is the same size as the cross-card effect this study registered. Every
cell is now timed by `moe.bench.timing.time_kernel` under `TIMING_BASIS`, and
the instrument name, the warmup duration, the iteration and trial counts, the
SM clock sampled UNDER LOAD, its two verdicts and the flush state are columns on
every row. LEVEL is two-sided since 2026-09-03 and the SIDE is a column too.
SINCE 2026-09-09 A CELL IS EXCLUDED IFF `clock_drift_ok` IS FALSE: the clock
MOVED inside the timed region, so the median is a blend of two operating points
and the time belongs to neither, which no rescaling repairs. NEITHER LEVEL SIDE
EXCLUDES ANYTHING. Both are KEPT, counted on their side, and printed with the
direction they move the fixed-roof fraction in, because on a 700 W-capped card
the under-load clock is an outcome of the CELL, set per tile by the kernel's
own power draw: the 2026-09-09 H200 session holds BM=128/BN=64 at a median 1395
MHz and BM=256 at 1650 against a 1485 MHz calibration GEMM, and a memory-shaped
cell sits at 1950-1980. The old rule excluded the LOW side, and on that card it
removed bm128_depth's entire BM=128 subject and every multi-tile BM=128 cell of
the roofline arm, five of them low by 0.75 MHz -- half of one NVML step. What
an off-band steady clock costs is not the measurement but its fraction of the
FIXED roof, which is off by the clock ratio; the driver's
`roof_at_cell_clock_tflops` is the roof to read that beside. Until this tree
every consumer read `clock_level_ok is False` as "ran cold" and would have
dropped every boosted tread from the fit whose purpose is to find the memory
branch.

EXIT CODES AND THE ONE GREPPABLE LINE. `moe.bench.exit_codes` owns both. Every
scored gate prints exactly one `RESULT: KIND NAME VERDICT detail` line, the
process exit code comes from `exit_codes.classify` over the same gates, and a
refusal exits REFUSED before anything is measured. Nothing else in the output is
a gate result. NOTHING IS FOLDED: `--fail-on-gate` is retired, accepted and
ignored, because a CLAIM_FAIL reported as DONE is a log and a process saying two
different things about one run, and 1 is already the code that tells the ledger
a claim was refuted rather than that the apparatus broke. An unplanned
exception exits ERROR (4), not the interpreter's 1, for the same reason.

OFF-GPU. `--self-test ALPHA` generates the cells from the physical model at that
alpha and runs the entire analysis on them, so the gates, the fits and the
report are exercised on a laptop, and so the claim "these gates can tell 0.558
from 0.10" is checkable rather than asserted. `--self-test-world` plants the
four worlds that are not a single alpha: a drifting-clock tread, whose median
is a blend of two operating points and which must be excluded; a low-clock and
a high-clock tread, the two steady off-band states a power-capped card holds
per tile, both of which must be KEPT and counted on their side; and a memory
branch parallel to the compute branch, which must come out UNDECIDED.
A planted run whose directory this script NAMES writes under a
`synthetic-` directory carrying one `plant` token that spells the alpha, the
noise and the world, so one self-test can never overwrite another; a metered run
carries no plant knob at all, so nothing about the self-tests changes the name a
paid run writes under. A planted run whose directory is NAMED FOR IT, by
`--run-id`, is REFUSED unless that name already begins `synthetic-`, because the
one automated caller that runs this script's self-test on a pod is exactly the
caller that always supplies a name and a rehearsal was landing on the paid
replicate's own `report.json`. Those two rules together are what makes "no plant
can land in a metered run's directory" true rather than true of the default
path; `resolve_run_id` holds the second one and says what it costs.
`--dry-run` prints the grid, the predictions and the cost estimate without
touching a GPU, and REFUSES, because it measured nothing. Absent
torch, CUDA or vLLM the script says which one is missing and what to run
instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.weights import (  # noqa: E402
    WeightSetRefused,
    WeightStreamSlope,
    routed_expert_weight_bytes,
    weight_stream_ms,
    weight_streams_per_tile,
)
from moe.spec import MODEL_CONFIGS, MoEConfig, dtype_bytes  # noqa: E402

#: The five names above are imported INDIVIDUALLY and not as the
#: `moe.bench.weights` module, because `weights` is already a local in
#: `model_ms` (a byte count) and in `run_sweep` (a pair of expert tensors); a
#: module bound to that name at file scope would be shadowed inside both.

#: `moe.bench.timing` is imported LAZILY, everywhere, and this comment is the
#: reason. That module imports torch at module scope; this one is documented to
#: run `--dry-run` and `--self-test` on a laptop with no torch at all, and an
#: import here would turn that documented path into an ImportError before
#: argparse ever ran. `exit_codes`, `provenance`, `ai_model` and
#: `weights` import nothing heavier than the standard library, so they are
#: imported normally above.


def timing_basis() -> str | None:
    """The name of the instrument this file times with, or None off-torch.

    None is not a default: it says the instrument could not be NAMED on this
    machine because torch is absent, which is exactly the laptop `--dry-run` and
    `--self-test` case where nothing was measured either. Every row a pod
    produces carries the string, and a row without it is a row from before
    2026-09-02 or a row from a laptop.
    """
    try:
        from moe.bench.timing import TIMING_BASIS
    except Exception:                                     # noqa: BLE001
        # Broad: a torch that is INSTALLED and broken raises OSError on a
        # missing libcudart rather than ImportError (moe/bench/provenance.py
        # records the same pod failure), and naming the instrument is never
        # worth taking the whole report down for.
        return None
    return TIMING_BASIS


def observed_iters(prov, cells):
    """The run's provenance block with the iteration count the cells were
    actually timed at, or unchanged when nothing was timed.

    THE FIELD USED TO CARRY A NUMBER NO CELL USED. `main` passed
    `iters=args.iters`, the argparse default 50, on the same day `--iters` was
    retired as a timing knob; on a pod `time_kernel` sizes the count per cell
    from `--cell-budget-ms` and it is hundreds for a 1 ms kernel, so
    `provenance.iters` contradicted every row of `cells.csv`. The median over
    the cells that were timed is a number a reader can check against those
    rows, and the report prints the range beside it so a median standing for a
    spread of 20 to 900 cannot pass for a constant.

    NOTHING TIMED MEANS NOTHING RECORDED. A `--self-test` plants `iters=0` on
    every cell, so there is no median to take and the block keeps its None and
    its "supplied as None" reason. Inventing one there would be the same defect
    with a different number in it.
    """
    counts = sorted(c.iters for c in cells
                    if c.status == "ok" and c.iters and c.iters > 0)
    if not counts:
        return prov
    missing = {k: v for k, v in prov.missing.items() if k != "iters"}
    return replace(prov, iters=int(statistics.median(counts)), missing=missing)


def iters_line(cells) -> str:
    """One line naming the instrument's iteration counts, or saying there were
    none. Printed in the report so `provenance.iters` is never read as a knob
    somebody set."""
    counts = sorted(c.iters for c in cells
                    if c.status == "ok" and c.iters and c.iters > 0)
    if not counts:
        return ("iterations per trial: none recorded (nothing was timed; a "
                "self-test's cells carry iters=0)")
    return (f"iterations per trial: median {int(statistics.median(counts))} "
            f"over {len(counts)} timed cells, range {counts[0]}-{counts[-1]}. "
            "Sized per cell by the instrument from --cell-budget-ms, not by "
            "--iters, which is retired as a timing knob.")

# --------------------------------------------------------------------------
# The numbers this script is arguing about. All of them stated up front so a
# reader can check what was assumed without reading the code.
# --------------------------------------------------------------------------

#: Refit 2026-08-31 against the derived tile with a per-group intercept.
ALPHA = 0.558
ALPHA_BAND = (0.529, 0.588)

#: The published value this refit replaces. Kept because every gate here is
#: designed to DISCRIMINATE against it, and a gate that cannot name what it
#: rules out is a gate nobody can check.
RETRACTED_ALPHA = 0.10

#: alpha measured per BLOCK_M. The scalar above is the pooled fit; these are the
#: slices, and they are what gate 4's worst case is run against.
#:
#: WHAT THE DRIFT WITH BLOCK_M IS, since this used to be filed as an unexplained
#: wobble that "cannot be pinned". These are `LadderFit.alpha` readings, and
#: that estimator is B/(A+B): the per-tile slope over the fitted level at one
#: tile. `moe/bench/ai_model.py` derives what that returns on three-term traffic
#: and the form has a name, EXA:
#:
#:     alpha_fitted = (alpha_b + phi) / (1 + phi + delta)
#:
#: with `alpha_b` the weight miss fraction the study NAMES, `phi` one M-tile's
#: activation-and-output cost in units of one full weight read, and `delta` the
#: fused layer's fixed cost in the same units. `phi` grows with BM/BN -- an
#: extra M-tile re-reads activations per N-tile, in exactly that ratio -- so
#: alpha_fitted rises with BLOCK_M at fixed BLOCK_N even when alpha_b does not
#: move at all. At BN=64 on mixtral, phi is 0.16 at BM=64 and 0.32 at BM=128,
#: which is the direction and roughly the size of the 0.466 -> 0.625 step here.
#: So these two numbers are FITTED QUANTITIES of a particular tiling, not two
#: measurements of one miss fraction that disagree, and 0.625 is not "a larger
#: fraction of the weights missed". Reading either as a miss fraction is what
#: `ai_model.alpha_b_from_fitted` exists to undo, and any cap taken as
#: 2*BM/(alpha_fitted*b) is HIGH by `ai_model.lin_overstatement` = 1 + phi +
#: delta, which this report prints beside every such cap.
ALPHA_BY_BLOCK_M = {64: 0.466, 128: 0.625}

#: THE WITHDRAWN H200 RIDGE BAND, kept by name as history. Its two ends are two
#: calibrations of the same card disagreeing by 9.9% on the COMPUTE term while
#: bandwidth reproduced to 0.06%: 160.3 is calibration md5 `4d84542b` (701.6
#: TFLOP/s over 4377.2 GB/s, the `fp8-three-kernel` / `v2lite` arms) and 176.2
#: is 770.9 over 4374.5 (`fp8-refixed` / `whole-layer`). It is NOT "the
#: 2026-08-26 band", which this comment used to call it: the arms of that date
#: quote 160.4 and 162.8 (`scripts/rescore_published_reports.py`,
#: `WITHDRAWN_RIDGE_WHY`). So it measures how badly the compute ceiling
#: reproduces, not any device's ridge, and the two ends do not merely widen a
#: band: they change which TREAD the BLOCK_M=128 crossing lands in (2 at
#: 160.3, 3 at 176.2). It was withdrawn from all 26 published reports on
#: 2026-09-02.
#:
#: IT IS NOT A DEFAULT AND MUST NEVER BECOME ONE AGAIN. `--ridge` used to default
#: to `RIDGE_BAND[0]` and `scripts/cross_card_surface.sh` never passed `--ridge`,
#: so all 7 published A100 reports carried ridge=160.3 and ridge_band=[160.3,
#: 176.2] -- a band belonging to NEITHER card. The A100's own contemporaneous
#: calibration is 262.371/1.79936 = 145.8 and the H200's is 712.259/4.37476 =
#: 162.8, so every printed `ridge x bandwidth` on the A100 was a hybrid of two
#: machines. `resolve_ridge` reads the ATTACHED device's calibration and
#: REFUSES when there is none; this constant survives only as the hypothesis a
#: laptop planning run (--dry-run / --self-test) is allowed to assume, where
#: nothing was measured and so nothing can be mislabelled, and as the value
#: `scripts/tile_cap_test.py` imports. The name is unchanged because that file
#: and its tests import it by name and are not this slice's to edit.
RIDGE_BAND = (160.3, 176.2)

#: What a report says when its ridge is this constant rather than a measurement.
#: Carried into `report.json` so the provenance cannot be lost between the
#: printout and the file, which is how 160.3 reached seven A100 reports unnoticed.
HYPOTHESIS_RIDGE_SOURCE = (
    "HYPOTHESIS: the withdrawn H200 band 160.3-176.2 (md5 4d84542b at the low "
    "end, two compute calibrations 9.9% apart), which belongs to no attached "
    "device")

#: The card slug a run id carries when no device is attached: every --dry-run
#: and every --self-test on a laptop. Visible rather than blank, so a laptop
#: directory cannot be mistaken for the one a pod would write to.
NO_CARD_SLUG = "nocard"


def detect_card_slug() -> str:
    """Slug for the ATTACHED device, or `NO_CARD_SLUG`.

    Read at run-id time, because the card belongs IN the id. See
    `default_run_id` for the collision it prevents.
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
        # A driver present but unusable is not a card identity, and naming it
        # `nocard` keeps the run out of a real card's directory.
        return NO_CARD_SLUG
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or NO_CARD_SLUG

#: Held fixed so BLOCK_SIZE_M is the only thing that can move. Same values as
#: `scripts/tile_sweep.py` pins, so the two experiments compose. num_warps=8
#: satisfies Triton's `BLOCK_M % 64 == 0 AND num_warps % 4 == 0` warpgroup
#: predicate at every setting that can reach it, leaving BLOCK_SIZE_M as the
#: only thing that flips the instruction.
FIXED = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
         "num_warps": 8, "num_stages": 4}

DEFAULT_BLOCK_SIZES = (32, 64, 128, 256)

#: H200. Only used to turn a block count into waves, and printed with its
#: source so a wave count is never mistaken for something the driver reported.
DEFAULT_SM_COUNT = 132

#: WHAT GATE 3 ACTUALLY TESTS. The midpoint between the retracted alpha (0.10)
#: and the refit one (0.558), rounded to the value the published gate used.
#:
#: The gate used to be phrased as a CROSSING RATIO and scored `1 + alpha_hat`
#: against 1.33. That "measured" field was an algebraic restatement of a fitted
#: alpha imported from a DIFFERENT BLOCK_M -- checked against all 22 published
#: reports that carry one, where measured == 1 + alpha_measured in 22 of 22 and
#: differs in 0 -- so the ratio was never an observation. Worse, the identity
#: `R_cross(128)/R_cross(256) = 1 + alpha` holds only if the 128 crossing lands
#: in tread 2, which at the A100's own alphas it does not in 3 of 6 arms; and
#: across all of `results/published` the ladder field `crosses` is False 41
#: times, null 61 times and True ZERO times. No crossing has ever been observed
#: by this study, so no gate may be phrased as though one was.
#:
#: RETIRED AS A VERDICT ON 2026-09-02, KEPT AS AN INFORMATIONAL LINE. Scoring
#: `alpha_hat > 0.33` is ONE-SIDED, and one-sided against the value the study
#: had already retracted. Every one of the 41 committed surface fits reads
#: 0.6-1.0 and every one of them passes this, while NOT ONE is within 0.05 of
#: the 0.558 the same reports predict; the one published sweep measured
#: 0.989/0.923 against a prediction of 0.558 and printed "it is where the refit
#: put it". A test that a pre-registered number cannot fail from above is not a
#: test of that number. `gate_3_alpha_discriminates` now scores the fitted
#: alpha's interval against `ALPHA_BAND` in BOTH directions and prints this
#: threshold beside the verdict as the retired discriminator it is.
GATE3_ALPHA_DISCRIMINATOR = 0.33

#: Resamples in the memory-branch bootstrap. 400 is enough for a 90% percentile
#: interval to be stable to about a third of a percent of alpha, and the whole
#: bootstrap runs in milliseconds on a ladder of at most a few dozen treads, so
#: there is nothing to buy by cutting it.
ALPHA_BOOTSTRAP_TRIALS = 400

#: Coverage of the percentile interval the bootstrap reports. 90% rather than
#: 95% because the interval is UNIONED with the two systematic ends below and a
#: wider random part would mostly widen an interval that is already dominated by
#: the systematic bracket at low noise.
ALPHA_BOOTSTRAP_COVERAGE = 0.90

#: Treads the bootstrap needs before it will resample at all. Below this a
#: resample draws the same one or two points over and over and reports an
#: interval of width zero, which would read as a precise measurement. The same
#: floor `MIN_MEMORY_TREADS` sets for quoting an alpha, for the same reason.
MIN_BOOTSTRAP_TREADS = 3

#: The same threshold in the ratio units the published gate printed. Kept so the
#: old verdict can be recomputed from a new report, and so `1 + alpha` never has
#: to be reconstructed by a reader guessing at what was compared.
GATE3_DISCRIMINATOR = 1.0 + GATE3_ALPHA_DISCRIMINATOR

#: Gate 2 asks a direction, and a direction needs a magnitude or noise answers
#: it. 1.5x at the largest common rows-per-expert is 5x the worst per-cell
#: timing spread this harness has produced.
GATE2_RATIO = 1.5

#: LEGACY, AND A UNIT ERROR. Gate 4 used to score `top > 0.85` where `top` is a
#: fraction of the ARM'S OWN measured plateau, while the 0.85's stated rationale
#: (cap/ridge = 0.716, "leaves room for the fused layer's non-GEMM work") is a
#: fraction of PEAK COMPUTE. Those are not the same denominator: across the 14
#: s3 reports the plateau is 50.5-71.8% of ridge x bandwidth and swings
#: 145.7-198.4 TFLOP/s inside ONE A100 session, so in the gate's own units the
#: model ceiling for the arm that "failed" it is 1.42 and 0.891 was never
#: evidence against anything. The two FAILs it ever produced are both qwen2 at
#: GROUP_SIZE_M=64, where the BLOCK_M=256 reference -- the plateau, the
#: DENOMINATOR -- drops to 41.36 rows/ms against 45.1-45.7 elsewhere while
#: BLOCK_M=64 is unchanged. That is a denominator artefact, not a crossing.
#:
#: `gate_4_no_crossing` now works in fractions of `ridge x bandwidth` on BOTH
#: sides and derives its threshold per run from the two worlds it separates.
#: This constant is retained only because `scripts/tile_cap_test.py` imports it.
GATE4_ROOF_FRACTION = 0.85

#: The two worlds gate 4 separates must actually be apart before a verdict means
#: anything. At alpha=0.558 the BLOCK_M=64 ceiling is cap/ridge = 0.70 (H200) or
#: 0.79 (A100) and the retracted alpha=0.10 puts it at 1.00, a gap of 0.21-0.30.
#: Below this the gate is not discriminating between them and says so instead of
#: scoring.
GATE4_MIN_SEPARATION = 0.10

#: A tread whose top throughput is this close to the plateau is compute bound.
COMPUTE_BOUND_FRACTION = 0.95

#: Treads a ladder needs before its alpha may be quoted in a verdict.
#:
#: Two points make a line with no residual, so a two-tread fit cannot notice
#: that one of its points was wrong -- and the tread most likely to be wrong is
#: the last one, which sits right where the two branches meet. At 1% timing
#: spread a two-tread fit at BLOCK_M=128 reported alpha 0.486 on cells planted
#: at 0.10, and gate 3 passed on it. Three is the fewest treads that can
#: disagree with themselves. Ladders below the threshold are still PRINTED,
#: because "this block size saw two memory-bound treads" is information; they
#: are just not allowed to decide anything.
MIN_MEMORY_TREADS = 3


# --------------------------------------------------------------------------
# The model. Pure arithmetic: no torch, no GPU, no CSV.
# --------------------------------------------------------------------------

def q_of_tiles(tiles: int, alpha: float) -> float:
    """`1 + alpha (n - 1)`: weight traffic in units of one full read."""
    return 1.0 + alpha * (tiles - 1)


def ai_cap(block_m: int, alpha: float, b: int = 2) -> float:
    """`2 BM / (alpha b)`: the UNCORRECTED cap reading for a fitted alpha.

    It is the cap only when `alpha` is the true miss fraction `alpha_b`. For a
    ladder alpha, which is `(alpha_b + phi) / (1 + phi + delta)` (EXA), this
    number is HIGH by `ai_model.lin_overstatement` = `1 + phi + delta`, and
    the report prints that factor beside every cap it quotes; `ai_model.
    cap_from_fitted` is the corrected reading. Kept in this form because the
    published reports and `scripts/tile_cap_test.py` quote it, and a cap
    printed without its factor is the defect the audit found.

    Why alpha is not a nuisance parameter: if the CORRECTED cap sits below the
    ridge, the block size cannot be compute bound at any batch size that
    exists. Infinite at alpha <= 0, which is the correct reading and not a
    guard: with no re-read cost `2r/b` is exact and unbounded.
    """
    return math.inf if alpha <= 0 else 2.0 * block_m / (alpha * b)


@dataclass(frozen=True)
class TilePrediction:
    """What one BLOCK_SIZE_M is predicted to do, before anything runs."""

    block_m: int
    alpha: float
    ridge: float
    dtype_bytes: int
    ai_cap: float
    #: The first M-tile count at which the expert is compute bound. None means
    #: NO CROSSING AT ALL, which is a prediction and not a missing value.
    first_compute_tread: int | None
    #: `ridge b Q(n*) / 2`, rows per expert. None when there is no crossing.
    crossing_rows: float | None

    @property
    def crosses(self) -> bool:
        return self.crossing_rows is not None

    def crossing_tokens(self, num_experts: int, top_k: int) -> float | None:
        """Rows per expert back into tokens: `R E / k`, saturated routing."""
        if self.crossing_rows is None:
            return None
        return self.crossing_rows * num_experts / top_k


def predict_tile(block_m: int, alpha: float, ridge: float, b: int = 2,
                 max_tiles: int = 4096) -> TilePrediction:
    """Solve `n BM >= ridge b Q(n) / 2` for the first tread that is compute bound.

    Scanned rather than solved in closed form because both sides step: the left
    is the padded row count, which only takes multiples of BM, and the right
    moves in units of alpha. A closed form would have to round, and rounding is
    where the two ends of the ridge band stop agreeing about which tread the
    crossing lands in.

    Terminates for the right reason. The right-hand side grows by
    `ridge b alpha / 2` per tread and the left by `BM`, so a solution exists iff
    `BM > ridge b alpha / 2`, which is exactly `ai_cap > ridge`. When the cap is
    below the ridge the gap widens forever and `max_tiles` only decides how long
    the script is willing to demonstrate that; None is then the answer, not a
    timeout.
    """
    cap = ai_cap(block_m, alpha, b)
    if cap <= ridge:
        return TilePrediction(block_m, alpha, ridge, b, cap, None, None)
    for n in range(1, max_tiles + 1):
        rows = ridge * b * q_of_tiles(n, alpha) / 2.0
        if n * block_m >= rows:
            return TilePrediction(block_m, alpha, ridge, b, cap, n, rows)
    raise AssertionError(                             # pragma: no cover
        f"cap {cap:.1f} exceeds ridge {ridge} but no tread crossed in "
        f"{max_tiles} tiles; the scan and the cap disagree")


def predictions(block_sizes, alpha: float, ridge: float, b: int = 2
                ) -> dict[int, TilePrediction]:
    return {bm: predict_tile(bm, alpha, ridge, b) for bm in block_sizes}


def crossing_ratio(preds: dict[int, TilePrediction], lo: int, hi: int
                   ) -> float | None:
    """`R_cross(lo) / R_cross(hi)`, gate 3's quantity. None if either misses."""
    a, c = preds.get(lo), preds.get(hi)
    if a is None or c is None or a.crossing_rows is None or c.crossing_rows is None:
        return None
    return a.crossing_rows / c.crossing_rows


# --------------------------------------------------------------------------
# Geometry: rows, tiles, waves, bytes, flops. Everything a cell knows before
# it is timed, so a report can be checked against the grid without the GPU.
# --------------------------------------------------------------------------

def rows_step(cfg) -> int:
    """Token step that keeps rows-per-expert an exact integer.

    `R = T k / E`, so T must be a multiple of `E / gcd(E, k)`. Stated as a
    function because getting it wrong does not raise: it produces a target
    histogram `realize_counts` refuses, halfway through a metered run.
    """
    return cfg.num_experts // math.gcd(cfg.num_experts, cfg.top_k)


def rows_quantum(cfg) -> int:
    """Rows-per-expert must be a multiple of this or the token count is not one.

    `T = R E / k`, so R must be a multiple of `k / gcd(E, k)`. It is 1 for
    mixtral (E=8, k=2) and for qwen2 (E=64, k=8), which is why nothing noticed
    -- and 3 for deepseek-v2-lite (E=64, k=6), whose default grid starts at
    r=28 and stepped by 32, hitting a legal row only by accident.

    build_grid did not consult it, so the model with the SMALLEST per-expert
    footprint -- the low-phi anchor of the whole alpha-versus-L2 curve -- died
    three seconds into an unattended run with `28 rows per expert is not an
    integer token count`, twice, while every other arm passed. A geometry
    constraint that excludes one model from a study is worse than a crash,
    because the surface still plots.
    """
    return cfg.top_k // math.gcd(cfg.num_experts, cfg.top_k)


def tokens_for_rows(cfg, rows: int) -> int:
    tokens = rows * cfg.num_experts / cfg.top_k
    if abs(tokens - round(tokens)) > 1e-9:
        raise ValueError(f"{rows} rows per expert is not an integer token count "
                         f"for E={cfg.num_experts} k={cfg.top_k}")
    return int(round(tokens))


def rows_for_tokens(cfg, tokens: int) -> float:
    return tokens * cfg.top_k / cfg.num_experts


def tiles_per_expert(rows: float, block_m: int) -> int:
    return max(1, math.ceil(rows / block_m))


def weight_bytes_per_expert(cfg, b: int) -> int:
    """up `[H, 2F]` plus down `[F, H]`, which is `3 F H` elements."""
    return 3 * cfg.intermediate_size * cfg.hidden_size * b


def activation_bytes_per_row(cfg, act_b: int = 2) -> int:
    """x_perm, h_up, h_act, y_perm: the traffic that grows WITH the batch.

    Named and counted because it is affine in the tile count too, so it lands
    in the fitted memory-branch slope and inflates the measured alpha. At
    mixtral it is 102 KB against a 352 MB weight read per expert, so the
    inflation runs from 1.7% of alpha at BLOCK_M=32 to 13% at 256 -- small, but
    not nothing, and the report's `alpha-corrected` column subtracts it.
    """
    cfg_terms = 2 * cfg.hidden_size + 3 * cfg.intermediate_size
    return cfg_terms * act_b


def useful_flops(cfg, rows_total: float) -> float:
    """`6 F H` per row: up is `2 F H` MACs, down is `F H`, two flops each."""
    return 6.0 * rows_total * cfg.intermediate_size * cfg.hidden_size


def waves(cfg, rows_per_expert: float, block_m: int, block_n: int,
          sm_count: int) -> tuple[float, float]:
    """CTA waves for the up and down GEMMs, one CTA resident per SM assumed.

    vLLM pads EACH EXPERT to a multiple of BLOCK_SIZE_M in
    `moe_align_block_size`, so the M-tile count is `E ceil(r / BM)` and not
    `ceil(E r / BM)`. Reported per cell because a time difference across a tile
    step is only about traffic if occupancy was saturated on BOTH sides of it,
    and one resident CTA per SM is the conservative reading: a real kernel
    fitting two would halve these and still be saturated.
    """
    m_tiles = cfg.num_experts * tiles_per_expert(rows_per_expert, block_m)
    up = m_tiles * math.ceil(2 * cfg.intermediate_size / block_n)
    down = m_tiles * math.ceil(cfg.hidden_size / block_n)
    return up / sm_count, down / sm_count


def model_ms(cfg, rows_per_expert: float, block_m: int, *, alpha: float,
             ridge: float, bandwidth_gbps: float, b: int = 2,
             overhead_ms: float = 0.0, activations: bool = True) -> float:
    """Predicted milliseconds: `overhead + max(traffic, padded compute)`.

    The generator for `--self-test`, and the source of every "predicted" column
    in the report. It is the model under test, so nothing that reads it may be
    read as evidence FOR it -- its job is to say what the data would look like
    in each of the two worlds, and the gates then ask which one arrived.

    The compute side is charged on PADDED rows. A tile computes `BM` rows
    whether or not they are useful, so a half-empty tile costs a full one, and
    that is why time is flat along a tread and steps at the tread boundary.
    """
    tiles = tiles_per_expert(rows_per_expert, block_m)
    bw = bandwidth_gbps * 1e9
    peak = ridge * bw
    weights = cfg.num_experts * weight_bytes_per_expert(cfg, b)
    traffic = weights * q_of_tiles(tiles, alpha)
    if activations:
        traffic += (cfg.num_experts * rows_per_expert
                    * activation_bytes_per_row(cfg))
    padded_rows = cfg.num_experts * tiles * block_m
    compute_s = useful_flops(cfg, padded_rows) / peak
    return overhead_ms + 1e3 * max(traffic / bw, compute_s)


# --------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------

def build_grid(cfg, block_sizes, r_max: int, row_step: int, step_probes: int
               ) -> list[int]:
    """Rows-per-expert to measure, as a sorted list.

    Two overlaid designs, because the four gates need different things.

    The BACKGROUND is every multiple of `row_step` (the smallest block size by
    default) up to `r_max`. That puts an EXACTLY-FULL tile stack -- `r = n BM`,
    zero padding -- on the grid for every block size, which is what the ladder
    fit reads, and it puts a common point for all four block sizes on every
    multiple of the largest one, which is what gate 2 compares across.

    The PROBES bracket each of the first `step_probes` tile boundaries per block
    size at `n BM +/- max(2, BM/8)`, because gate 1 is about where the step is
    and a grid that only samples tread tops cannot see a step at all: it has one
    point per tread and every interval it can form spans a boundary.
    """
    # Every point is snapped DOWN to a legal row count, then the illegal ones
    # are dropped rather than nudged: a probe at `edge - gap` that got rounded
    # onto `edge` would stop bracketing the boundary it exists to bracket, and
    # gate 1 would be reading a step that no point straddles.
    q = rows_quantum(cfg)
    row_step = max(row_step, q)
    if row_step % q:
        row_step += q - (row_step % q)
    grid = set(range(row_step, r_max + 1, row_step))
    for bm in block_sizes:
        gap = max(2, bm // 8)
        for n in range(1, step_probes + 1):
            edge = n * bm
            if edge > r_max:
                break
            # The point ABOVE the boundary is allowed one gap past `r_max`. A
            # boundary landing exactly on `r_max` -- which the largest block
            # size's last tread always does -- would otherwise have nothing
            # above it, and an unbracketed boundary is invisible to gate 1
            # however dense the rest of the grid is.
            for r in (edge - gap, edge, edge + gap):
                if 1 <= r <= r_max + gap and r % q == 0:
                    grid.add(r)
    return sorted(grid)


#: `moe.bench.timing.iters_for`'s clamp, mirrored here and ONLY here. That
#: module imports torch at module scope and this file is documented to plan a
#: run on a laptop that has none, so `--dry-run` cannot call the real function
#: and cannot be allowed to guess a different rule either: the estimate an
#: operator buys pod time with has to be the sizing the pod will use.
#: `planned_iters` prefers the real function whenever it imports, and
#: `test_the_mirrored_iters_clamp_is_the_instruments_own` asserts these two
#: numbers against it on any machine where it does.
ITERS_FOR_LO, ITERS_FOR_HI = 10, 2000


def planned_iters(per_call_ms: float, target_ms: float) -> int:
    """`timing.iters_for` when torch imports, its arithmetic when it does not.

    The fallback is a MIRROR and is documented as one, not a second policy: the
    instrument sizes a trial to hold `target_ms` of kernel time, clamped to
    `[ITERS_FOR_LO, ITERS_FOR_HI]`, and a planner that used any other rule would
    be pricing a run nobody is going to make.
    """
    try:
        from moe.bench.timing import iters_for
    except Exception:                                     # noqa: BLE001
        # Broad for the reason `timing_basis` is: an installed-and-broken torch
        # raises OSError on a missing libcudart, and a cost estimate is never
        # worth taking the plan down for.
        return max(ITERS_FOR_LO,
                   min(ITERS_FOR_HI, int(target_ms / max(per_call_ms, 1e-4))))
    return iters_for(per_call_ms, target_ms)


def estimated_seconds(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                      bandwidth_gbps: float, b: int, cell_budget_ms: float,
                      warmup_ms: float | None = None, trials: int | None = None,
                      iters: int | None = None,
                      warmup: int | None = None) -> float:
    """Cost of the whole sweep at the model's own prediction, for --dry-run.

    THE UNIT BUG THIS FIXES, because it is the only number an operator buys pod
    time with. `--warmup` became MILLISECONDS of delivered load on 2026-09-02
    and this function was not migrated with it: it went on charging
    `ms * (warmup + scaled_iters(...))`, so the default 300.0 was billed as 300
    CALLS. That is a duration added to an iteration count, it was 15x the old
    20-call term, and `--trials` -- which multiplies the real cost by three --
    did not enter at all. The mixtral default grid printed 278 s for a run whose
    honest figure is about 414 s, and a 1 ms cell and an 11 ms cell were priced
    700 ms and 3696 ms when the instrument charges nearly the same for both.

    THE RULE, WHICH IS THE RUNNER'S. `time_kernel` warms for a FIXED DURATION
    and then runs `trials` trials, each sized by `iters_for` to hold
    `cell_budget_ms` of kernel time. So one cell costs
    `warmup_ms + trials * ms * planned_iters(ms, cell_budget_ms)`, which is
    about `warmup_ms + trials * cell_budget_ms` and is nearly INDEPENDENT of the
    per-call time -- exactly the property the old formula did not have. The
    clamp is why it is not exactly independent: a cell slower than
    `cell_budget_ms / ITERS_FOR_LO` still pays for ten iterations.

    TWO INSTRUMENTS, NAMED BY THEIR KEYWORDS, because this file does not own
    every caller. `warmup_ms=` and `trials=` price the instrument above.
    `iters=` and `warmup=` price the RETIRED one, where `warmup` is a CALL COUNT
    and `iters` a fixed sample size -- which is what `scripts/tile_cap_test.py`
    still passes and still means, so its `--dry-run` keeps costing the run it
    will actually make. Mixing the two pairs is a refusal rather than a
    precedence rule: the two answers differ by 5x and picking one silently is
    how the wrong one got printed for a fortnight.
    """
    new = warmup_ms is not None or trials is not None
    old = iters is not None or warmup is not None
    if new and old:
        raise ValueError(
            "estimated_seconds got both instruments: warmup_ms/trials size a "
            "time_kernel run and iters/warmup size the retired per-call loop. "
            "They price different runs and there is no conversion between a "
            "duration of warmup and a count of warmup calls; pass one pair.")
    if not new and not old:
        raise ValueError(
            "estimated_seconds needs an instrument to price: warmup_ms= and "
            "trials= for time_kernel, or iters= and warmup= for the retired "
            "per-call loop. There is no default because the two differ by 5x.")
    total = 0.0
    for bm in block_sizes:
        for r in grid:
            ms = model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                          bandwidth_gbps=bandwidth_gbps, b=b)
            if new:
                total += (warmup_ms or 0.0) + (trials or 1) * ms * planned_iters(
                    ms, cell_budget_ms)
            else:
                total += ms * ((warmup or 0)
                               + scaled_iters(ms, iters or 1, cell_budget_ms))
    return total / 1e3


def scaled_iters(ms: float, iters: int, cell_budget_ms: float,
                 floor: int = 5) -> int:
    """Iterations that keep one cell inside its time budget.

    An 11 ms cell at BLOCK_M=32 and 1024 rows per expert costs 50x what a 0.7 ms
    one does, and the sweep is 4 x 80 cells. Recorded per cell in the CSV, so a
    reader can see which numbers rest on 50 samples and which on 5.
    """
    if ms <= 0:
        return iters
    return max(floor, min(iters, int(cell_budget_ms / ms)))


# --------------------------------------------------------------------------
# A measured cell.
# --------------------------------------------------------------------------

#: The two sides a failed LEVEL verdict can have, mirrored from
#: `moe.bench.timing.LEVEL_LOW` / `LEVEL_HIGH` rather than imported, because
#: that module imports torch at module scope and this file is documented to
#: plan a run and replay a CSV on a laptop that has none;
#: `tests/test_block_m_crossing_sweep.py` asserts the mirror against the
#: instrument wherever torch imports. The instrument writes them; this file
#: only reads them, and SINCE 2026-09-09 NEITHER SIDE EXCLUDES: the side is a
#: record on the row and `clock_drift_ok` alone decides membership. See
#: `Cell.clock_excluded` for the 750-cell measurement that settled it.
LEVEL_LOW = "low"
LEVEL_HIGH = "high"
LEVEL_SIDES = ("", LEVEL_LOW, LEVEL_HIGH)


def check_level_side(clock_level_ok: bool | None, clock_level_side: str) -> None:
    """REFUSE a clock verdict and side that contradict each other.

    A side is the direction a FAILED verdict went. A side on a passing or
    undetermined verdict is not a state the instrument produces (`timing.
    time_kernel` derives both from one `level_side` call and `driver` blanks
    the side unless the verdict is failed), so a row carrying one was built by
    hand from two sources and one of them is wrong; reading it either way
    would be a default. A side outside `LEVEL_SIDES` is a typo that would
    otherwise be read as "not high", which is LOW, which excludes. Both are
    raised here, at construction, rather than discovered as an exclusion count
    that does not add up.

    AND A FAILED VERDICT WITHOUT A SIDE IS REFUSED TOO. Since 2026-09-03
    `timing.clock_flags` derives the verdict FROM the side, so a failed LEVEL
    with a blank side is not a state the instrument produces either: it is a
    caller that copied `KernelTiming.clock_level_ok` and dropped
    `KernelTiming.clock_level_side`, which is exactly the shape of the defect
    this column closes (2026-09-08: `bn_decomposition` and
    `occupancy_vs_swizzle` did that through `make_cell`, and
    `span_extent_separation` in its own row class; every boosted tread they
    timed would have been read as LOW and dropped). Reading the blank as LOW
    drops every memory-bound tread on an H200; reading it as HIGH admits a
    sagged card into the fit; both are defaults. No cells.csv in the tree
    carries such a row (the verdict column
    is from 2026-09-02 and the side column from 2026-09-08, and no pod ran in
    between), so a file that does was written by a sibling that dropped the
    side, and its failed rows have to be re-measured.
    """
    if clock_level_side not in LEVEL_SIDES:
        raise ValueError(
            f"clock_level_side {clock_level_side!r} is not one of "
            f"{LEVEL_SIDES}; the instrument writes only those")
    if clock_level_side and clock_level_ok is not False:
        raise ValueError(
            f"clock_level_side {clock_level_side!r} with clock_level_ok="
            f"{clock_level_ok!r}: a side is the direction a FAILED verdict "
            "went, and this verdict did not fail")
    if clock_level_ok is False and not clock_level_side:
        raise ValueError(
            "clock_level_ok=False with no clock_level_side: a failed LEVEL "
            "verdict has carried a side since 2026-09-03 (timing.level_side) "
            "and without it the row cannot be told from a sagged card or a "
            "boosted one; pass KernelTiming.clock_level_side, or re-measure "
            "a cells.csv whose failed rows were written without the column")


@dataclass
class Cell:
    """One (BLOCK_SIZE_M, tokens) measurement, and everything derivable from it.

    `aligned` marks `rows_per_expert == n BLOCK_M` exactly, where padding is
    zero and useful throughput and padded throughput coincide. The ladder fit
    reads only aligned cells, because a partially-filled tread reports a
    throughput that depends on where in the tread it was sampled.

    THE STATE THE CELL WAS TIMED IN IS A COLUMN, not a property of the session.
    `instrument`, `warmup_ms`, `iters`, `trials`, `sm_clock_load_mhz`,
    `clock_level_ok`, `clock_level_side`, `clock_drift_ok` and `l2_flush` are
    what `moe.bench.timing.time_kernel` reports about the measurement it just
    made, and they are written per row because they are what makes a row
    comparable with the roof or not. Before 2026-09-02 none of them existed
    here: the published cells carry an iteration count and nothing else, so a
    reader cannot tell a cell timed on a card at 1980 MHz from one timed at
    1500, and every one of them was timed by a different instrument from the
    roof.

    THE THREE CLOCK FIELDS ARE OPTIONAL AND None MEANS "NOT DETERMINED", never
    "fine". A container without NVML, a trial too short for the poller to land a
    sample, and a laptop replay all produce None, and a filter that reads None
    as True would quietly re-admit exactly the rows this column exists to keep
    out.

    AND A FAILED LEVEL HAS A SIDE, WHICH IS A RECORD AND NOT AN EXCLUSION.
    `clock_level_side` is `LEVEL_LOW`, `LEVEL_HIGH` or "" (inside the band, or
    not determined), the instrument's own word for which way the loaded clock
    left the band around the reference. Until 2026-09-09 `ladder_treads`
    excluded on the LOW side; it no longer excludes on either. The side is
    carried on the row, printed beside the count, and read by nothing that
    decides membership. `Cell.clock_excluded` states the measurement that
    settled it. A LEVEL failure carrying no side is still REFUSED at
    construction (`check_level_side`): the instrument derives the verdict from
    the side, so a failed verdict without one was copied by a caller that
    dropped the side, and reading it either way is a default.
    """

    block_m: int
    tokens: int
    rows_per_expert: float
    tiles_per_expert: int
    padded_rows: int
    tile_eff: float
    aligned: bool
    waves_up: float
    waves_down: float
    ms_p50: float
    ms_min: float = 0.0
    ms_stdev: float = 0.0
    iters: int = 0
    useful_tflops: float = 0.0
    padded_tflops: float = 0.0
    status: str = "ok"
    detail: str = ""
    #: `moe.bench.timing.TIMING_BASIS` of the loop that produced `ms_p50`.
    #: Empty means a row from before the instrument had a name.
    instrument: str = ""
    #: Milliseconds of DELIVERED GPU load the warmup ran for, not a call count.
    warmup_ms: float = 0.0
    trials: int = 0
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    #: Which way a failed LEVEL went: `LEVEL_LOW`, `LEVEL_HIGH`, or "" when the
    #: verdict passed or was not determined. RECORDED, never an exclusion: see
    #: `clock_excluded`.
    clock_level_side: str = ""
    l2_flush: bool = False

    def __post_init__(self) -> None:
        check_level_side(self.clock_level_ok, self.clock_level_side)

    @property
    def rel_spread(self) -> float:
        return self.ms_stdev / self.ms_p50 if self.ms_p50 > 0 else 0.0

    @property
    def clock_excluded(self) -> bool:
        """Did the clock MOVE inside the timed region: `clock_drift_ok` failed.

        THE ONE CLOCK STATE A LADDER FIT EXCLUDES, since 2026-09-09. A median
        taken across two clock states is a blend and the time belongs to
        neither, and no rescaling repairs that. A STEADY clock, on either side
        of the band, is a measurement at a known clock: its side is recorded
        (`clock_level_side`, `clock_sagged`, `clock_boosted`) and its fraction
        of the roof at that clock (`roofline.roof_at_clock`) is printed beside
        the fixed-roof one.

        WHAT THE RULE WAS AND WHAT MEASUREMENT RETIRED IT. Until this date the
        rule was LEVEL-LOW excludes, and the H200 session of 2026-09-09 showed
        across 750 cells that on a 700 W-capped card the under-load clock is an
        outcome of the CELL, set per tile by the kernel's own power draw:
        BM=128/BN=64 holds a median 1395 MHz over 215 cells, BM=256 1650 over
        311, BM=32 1736, and memory-shaped cells boost to 1950-1980, while the
        8192^3 calibration GEMM sits at 1485 MHz at 691 W, near the LOW end of
        what dense tensor work does on this card. LEVEL-LOW was therefore a
        rule against a TILE: it removed every BM=128/BN=64 and BM=64/G=1 cell
        in the session -- the study's two primary tiles -- from measurability
        on that card, on every rerun, while removing nothing on the memory
        side. Every one of the session's 135 DRIFT verdicts, by contrast, is
        the governor settling on the FIRST cell of a repeat after a workload
        change, which is a fault of the instrument's warmup and is fixed there.

        False for None on purpose: an EXCLUSION has to be positively
        established. A row with no clock is a row whose comparability is
        unknown, and the report says how many of those there are rather than
        throwing them away.
        """
        return self.clock_drift_ok is False

    @property
    def clock_sagged(self) -> bool:
        """Did LEVEL fail on the LOW side: the cell ran below the band around
        the roof's clock. KEPT since 2026-09-09 and counted, because on a
        power-capped card that is the steady state of a dense tile and not a
        throttled box; what the fixed roof does not describe is its fraction,
        which is understated by the clock ratio."""
        return self.clock_level_side == LEVEL_LOW

    @property
    def clock_boosted(self) -> bool:
        """Did LEVEL fail on the HIGH side: the cell ran above the band around
        the roof's clock. KEPT in every fit; its fixed-roof fraction is inflated
        by the clock ratio and the driver's `roof_at_cell_clock_tflops` is the
        roof it should be read against."""
        return self.clock_level_side == LEVEL_HIGH


def make_cell(cfg, rows: float, block_m: int, ms: float, *, sm_count: int,
              block_n: int, ms_min: float = 0.0, ms_stdev: float = 0.0,
              iters: int = 0, status: str = "ok", detail: str = "",
              instrument: str = "", warmup_ms: float = 0.0, trials: int = 0,
              sm_clock_load_mhz: float | None = None,
              clock_level_ok: bool | None = None,
              clock_drift_ok: bool | None = None,
              clock_level_side: str | None = "",
              l2_flush: bool = False) -> Cell:
    """One cell, with the state it was timed in.

    SIGNATURE EXTENDED 2026-09-02, never narrowed: every new argument has a
    default and the four sibling scripts that call this
    (`bm128_roofline`, `bm128_depth`, `bn_decomposition`, `occupancy_vs_swizzle`)
    keep working unchanged. A cell built without them carries the empty
    instrument and three None clock verdicts, which is the honest record of a
    row whose timing state nobody wrote down.

    EXTENDED AGAIN 2026-09-08 with `clock_level_side`, the instrument's
    `KernelTiming.clock_level_side`. A caller that passes a FAILED verdict and
    not the side is REFUSED by `check_level_side` at construction: the only
    readings of that row are LOW (drops every boosted tread, the defect this
    column closes) and HIGH (admits a sagged card), and both are defaults. The
    sibling scripts that copy `t.clock_level_ok` into this call must copy
    `t.clock_level_side` beside it. None (the instrument's "no comparison was
    made", which only accompanies a verdict that is not False) is stored as "".
    """
    tiles = tiles_per_expert(rows, block_m)
    padded = cfg.num_experts * tiles * block_m
    rows_total = cfg.num_experts * rows
    up, down = waves(cfg, rows, block_m, block_n, sm_count)
    secs = ms * 1e-3
    return Cell(
        block_m=block_m, tokens=tokens_for_rows(cfg, int(rows)),
        rows_per_expert=float(rows), tiles_per_expert=tiles, padded_rows=padded,
        tile_eff=rows_total / padded if padded else 0.0,
        aligned=(rows % block_m == 0), waves_up=up, waves_down=down,
        ms_p50=ms, ms_min=ms_min, ms_stdev=ms_stdev, iters=iters,
        useful_tflops=(useful_flops(cfg, rows_total) / secs / 1e12) if secs > 0 else 0.0,
        padded_tflops=(useful_flops(cfg, padded) / secs / 1e12) if secs > 0 else 0.0,
        status=status, detail=detail, instrument=instrument,
        warmup_ms=warmup_ms, trials=trials,
        sm_clock_load_mhz=sm_clock_load_mhz, clock_level_ok=clock_level_ok,
        clock_drift_ok=clock_drift_ok, clock_level_side=clock_level_side or "",
        l2_flush=l2_flush)


#: What the synthetic cells claim as their instrument. A planted world is not a
#: measurement and must never carry `TIMING_BASIS`: a reader who greps a
#: cells.csv for the instrument has to be able to tell a pod row from a
#: generated one, and a self-test that stamped the real basis on its own
#: fabrications would make that impossible.
SYNTHETIC_INSTRUMENT = "synthetic/model-generated/not-measured"

#: What a planted run calls its CARD, in the run id and nowhere else.
#: `detect_card_slug` returns the ATTACHED device, and on a pod that is a real
#: card that measured nothing here: a free `--self-test 0.10` landed in the
#: metered run's own directory and its `report.json` write replaced the paid
#: arm's only machine-readable artefact with a synthetic one carrying the
#: retracted alpha. `run_id` puts the card slug at the FRONT, so naming the card
#: `synthetic` is also what prefixes the directory, and `ls` sorts every planted
#: world away from every measurement. The attached card is not lost: it travels
#: as the `synthhost` knob, because the planted world's ridge and bandwidth are
#: resolved from that card's own calibration and two pods' self-tests are
#: therefore two different worlds.
SYNTHETIC_CARD_SLUG = "synthetic"

#: Clock the synthetic cells claim to have run at, and the reference they are
#: scored against. Two numbers rather than one so the low-clock world below can
#: move the first without moving the second.
SYNTHETIC_CLOCK_MHZ = 1980.0

#: How far above its GEMM reference a boosted memory-shaped cell runs on the
#: H200: the committed calibration's `bandwidth_settle.final_mhz` 1980 over its
#: `gemm_clock_mhz` 1515 (moe/bench/hardware/measured_nvidia_h200.yaml). The
#: high-clock world applies this RATIO to the planted reference, so the planted
#: cell fails LEVEL on the HIGH side by the margin a real H200 cell does, and
#: not by a margin chosen to clear the band.
H200_BOOST_RATIO = 1980.0 / 1515.0


def synthetic_cells(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                    bandwidth_gbps: float, b: int, sm_count: int,
                    overhead_ms: float = 0.03, noise: float = 0.0,
                    seed: int = 0, low_clock: tuple[int, int] | None = None,
                    warmup_ms: float = 0.0, trials: int = 0,
                    l2_flush: bool = True,
                    high_clock: tuple[int, int] | None = None,
                    drifting: tuple[int, int] | None = None) -> list[Cell]:
    """Cells generated FROM the model, so the analysis has a known answer.

    This is what makes the whole report testable on a laptop, and it is what
    makes "these gates can tell 0.558 from 0.10" a check rather than a claim:
    generate at one alpha, read the gates, generate at the other, read them
    again. `noise` multiplies each cell by a lognormal draw so the gates are
    exercised against spread and not only against a clean curve.

    THE TIMING COLUMNS ARE PLANTED TOO, and that is not decoration. A cell whose
    `clock_drift_ok` is False is excluded from every ladder fit, and an
    exclusion path that only ever runs on a pod is a path nobody has watched
    work. `drifting=(BLOCK_M, tiles)` plants exactly one tread whose clock
    MOVED across its own trials, so the self-test can assert it was dropped,
    counted and named.

    THE TWO LEVEL SIDES ARE PLANTED AS KEPT PATHS. `low_clock=(BLOCK_M, tiles)`
    plants one tread that failed LEVEL LOW and `high_clock=(BLOCK_M, tiles)`
    one that failed LEVEL HIGH at `H200_BOOST_RATIO` times the reference. Since
    2026-09-09 BOTH must be KEPT, counted on their side, and must leave every
    number the clean world reports unchanged: on a power-capped card the
    under-load clock is set per tile by the kernel's own power draw, so
    excluding a side is excluding a tile. The LOW world was an exclusion path
    until that date, which is what made bm128_depth's whole subject ladder
    unmeasurable on the H200. Their milliseconds are the model's, unchanged,
    because a memory-bound tread's time does not follow the SM clock.
    Everything else carries `clock_level_ok=True` and `SYNTHETIC_INSTRUMENT`,
    which is deliberately NOT `TIMING_BASIS`: these rows were not measured.
    """
    rng = random.Random(seed)
    out = []
    for bm in block_sizes:
        for r in grid:
            ms = model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                          bandwidth_gbps=bandwidth_gbps, b=b,
                          overhead_ms=overhead_ms)
            if noise:
                ms *= math.exp(rng.gauss(0.0, noise))
            slow = (low_clock is not None and bm == low_clock[0]
                    and tiles_per_expert(r, bm) == low_clock[1])
            boosted = (high_clock is not None and bm == high_clock[0]
                       and tiles_per_expert(r, bm) == high_clock[1])
            moved = (drifting is not None and bm == drifting[0]
                     and tiles_per_expert(r, bm) == drifting[1])
            if slow:
                clock, side = SYNTHETIC_CLOCK_MHZ * 0.7, LEVEL_LOW
            elif boosted:
                clock, side = SYNTHETIC_CLOCK_MHZ * H200_BOOST_RATIO, LEVEL_HIGH
            else:
                clock, side = SYNTHETIC_CLOCK_MHZ, ""
            out.append(make_cell(
                cfg, r, bm, ms, sm_count=sm_count,
                block_n=FIXED["BLOCK_SIZE_N"], ms_min=ms,
                ms_stdev=ms * noise, iters=0,
                instrument=SYNTHETIC_INSTRUMENT, warmup_ms=warmup_ms,
                trials=trials, l2_flush=l2_flush,
                sm_clock_load_mhz=clock,
                clock_level_ok=not (slow or boosted),
                clock_drift_ok=not moved,
                clock_level_side=side))
    return out


# --------------------------------------------------------------------------
# The ladder: time per tread, which is where every gate but the first is read.
# --------------------------------------------------------------------------

#: What a ladder fit CONCLUDED, as a fixed token. Six states, and the whole
#: point of having six is that the four that are not `IDENTIFIED` used to arrive
#: as the same blank. `crosses=None` was printed for the BLOCK_M=128 row of
#: every H200 arm and read as "the sweep lacked treads", when the causes were
#: variously a membership rule anchored at n=1, a memory branch parallel to the
#: compute branch, and a compute reference 44x too steep.
IDENTIFIED = "identified"
#: The treads were there, the branch was fitted, and its slope is within
#: `PARALLEL_BRANCH_TOLERANCE` of the compute branch. `B / C = ridge / ai_cap`,
#: so this says the tile's cap sits ON the ridge: the sweep cannot decide, and a
#: roofline arm at this tile can. NOT None, and never imported over.
UNDECIDED_PARALLEL_BRANCH = "undecided_parallel_branch"
#: Enough treads were dropped for a DRIFTING clock that what remains is below
#: `MIN_MEMORY_TREADS`. The governor moving mid-cell, not the tile, is what
#: this ladder measured. Named `UNDECIDED_LOW_CLOCK` / "undecided_low_clock"
#: until 2026-09-09, when LEVEL-LOW stopped excluding anything and DRIFT became
#: the only exclusion: the old token named a cause that can no longer produce
#: this outcome.
UNDECIDED_DRIFTING_CLOCK = "undecided_drifting_clock"
#: Fewer memory-bound treads than `MIN_MEMORY_TREADS`, with nothing excluded and
#: no reference problem: the tile really is compute bound this early.
NOT_IDENTIFIED_TOO_FEW = "too_few_memory_treads"
#: This ladder IS the compute reference, so by the assumption that qualified it
#: there is no memory branch here to fit.
NOT_IDENTIFIED_IS_REFERENCE = "is_the_reference_ladder"
#: No usable tread at all.
NOT_IDENTIFIED_NO_TREADS = "no_usable_treads"

LADDER_OUTCOMES = (IDENTIFIED, UNDECIDED_PARALLEL_BRANCH, UNDECIDED_DRIFTING_CLOCK,
                   NOT_IDENTIFIED_TOO_FEW, NOT_IDENTIFIED_IS_REFERENCE,
                   NOT_IDENTIFIED_NO_TREADS)


@dataclass(frozen=True)
class LadderFit:
    """A max-affine fit of `t(n)` over exactly-full tile stacks at one BLOCK_M.

    THE MEASUREMENT THIS SCRIPT IS BUILT AROUND. At `r = n BM` the layer's time
    is, under the model,

        t(n) = D + max(L Q(n), n M) = D + max(A + B n, C n)

    with `L` one full weight read, `M` one tile's compute, and `D` the fused
    layer's fixed cost. BOTH branches are affine in `n`, so the whole curve is
    two lines and the fit is a split point. That matters because the two lines
    are the two mechanisms, separated:

      * `B = L alpha` is the marginal cost of one more M-tile. `alpha = B / L`
        is then a DIRECT measurement, needing no crossing and no ridge.
      * `C = M` is the marginal cost of one more tile's arithmetic, and is
        proportional to BLOCK_M by construction, which is checkable across
        settings and is checked.
      * whether the block size crosses AT ALL is `C > B`, one comparison of two
        fitted slopes, which is algebraically the same statement as
        `ai_cap > ridge` and is measured instead of assumed.

    WHICH TREADS ARE ON THE MEMORY BRANCH IS NOT DECIDED BY THE RESIDUAL, and
    that is the most load-bearing decision in this file. A free split search
    picks the split that fits best, and at BLOCK_M=128 the best-fitting split
    puts treads 1 and 2 on the memory branch and reports alpha = 0.597 -- from
    data generated at alpha=0.558 AND from data generated at alpha=0.10, because
    tread 2 is compute bound in both worlds and a line drawn through one memory
    point and one compute point answers the same thing whatever the memory
    branch was doing. A number that comes out at 0.6 under every hypothesis is
    unfalsifiable, and it would have arrived looking like a confirmation.

    So membership is decided by the COMPUTE branch instead. `C = 2 BM N / peak`
    is proportional to BLOCK_M with no free parameter, so a compute branch
    measured at one block size gives the compute branch at every block size, and
    a tread is memory bound when its time stands above that line by more than
    the margin. Treads that merely lie ON it carry no information about the
    re-read fraction and are excluded from the alpha fit rather than averaged
    into it.

    WHAT THAT COSTS, said here because the gate that needs it will otherwise
    look broken. At BLOCK_M=128 exactly ONE tread stands above the compute
    branch (tread 1; tread 2 is 256 padded rows against a 249.7 row crossing, a
    2.5% margin), so alpha at 128 is NOT IDENTIFIABLE from this sweep, in either
    world. `memory_points < 2` is that state and gate 3 says out loud that it
    imported alpha from a block size where the measurement exists. The block
    sizes that cannot cross are the ones that measure alpha best, which is a
    pleasing shape for an experiment about a ceiling.
    """

    block_m: int
    points: tuple[tuple[int, float], ...]
    #: Treads standing above the compute branch, which is the count that decides
    #: whether alpha is identifiable here at all.
    memory_points: int
    #: `A + B n`, the memory branch. Both None when fewer than 2 treads are on it.
    intercept: float | None
    slope_memory: float | None
    #: `C n` fitted through the origin on this block size's own compute-bound
    #: treads. None when it has none of its own.
    slope_compute: float | None
    #: `C` scaled from the reference block size by `C ~ BLOCK_M`. Present even
    #: where this block size never reaches its own compute branch, which is
    #: exactly the case gate 4 is about.
    slope_compute_ref: float | None
    mean_rel_err: float
    overhead_ms: float
    basis: str
    #: One of `LADDER_OUTCOMES`. `IDENTIFIED` is the only one that entitles the
    #: ladder's alpha to decide anything; every other value is a NAMED failure
    #: to identify, and the difference between them is what a reader needs.
    #: Defaulted so the four sibling scripts that build a `LadderFit` through
    #: `fit_ladder` keep working without naming it.
    outcome: str = ""
    #: Prose for `outcome`, empty when the outcome is `IDENTIFIED`. This is the
    #: sentence the report and report.json both print, so the text a reader sees
    #: and the text a table generator reads cannot drift apart.
    outcome_reason: str = ""
    #: Cells dropped from this ladder because their clock MOVED while they were
    #: timed (`clock_drift_ok` failed), so the median is a blend of two
    #: operating points. Counted rather than silently missing. Called
    #: `excluded_low_clock` until 2026-09-09, when LEVEL-LOW stopped excluding.
    excluded_drifted: int = 0
    #: Treads KEPT in this ladder whose LEVEL failed LOW: on a power-capped card
    #: the steady state of a dense tile. Their times are in the fit; their
    #: fraction of the fixed roof is UNDERSTATED by the clock ratio.
    kept_low_clock: int = 0
    #: Treads KEPT in this ladder whose LEVEL failed HIGH: the boosted
    #: memory-shaped state. Their times are in the fit; their fraction of the
    #: fixed roof is OVERSTATED by the clock ratio, and the report says so
    #: beside this count.
    kept_high_clock: int = 0
    #: Index into `points` of the FIRST tread on the memory branch. Non-zero
    #: means the lowest tread(s) sat inside the margin and the branch starts
    #: above them; see `memory_branch_members` for why that is allowed and why
    #: the n=1 tread is the one it usually happens to.
    branch_start: int = 0
    #: The model and dtype whose expert weight set `weight_streams` divides by,
    #: and the rate it divides at. All four default to absent, which makes
    #: `weight_streams` None: the four sibling scripts build a `LadderFit`
    #: through `fit_ladder` without naming them, and a w computed against a
    #: guessed model or a guessed bandwidth would be worse than no w at all.
    #:
    #: A `MoEConfig` OR ITS NAME, and callers that hold the config should pass
    #: the config. Called `model_name` and typed `str` until 2026-09-10, when
    #: `analyse` was found handing the string down while every other number in
    #: the same report came off the `cfg` object beside it: a caller whose two
    #: disagreed got a w divided by one geometry and a roof computed from
    #: another, silently, which is the drift `moe/bench/weights.py` exists to
    #: prevent. A name is still accepted and is resolved through
    #: `MODEL_CONFIGS`; a config is used exactly as handed in.
    model: str | MoEConfig = ""
    dtype: str = ""
    #: The rate `w` names. Required to have come from a measurement; see
    #: `moe.bench.weights._check_bandwidth` for why there is no default.
    bandwidth_gbps: float | None = None
    #: Where that rate came from, in the caller's words, so the printed w says
    #: which bandwidth it is a fraction of.
    bandwidth_source: str = ""

    @property
    def undecided(self) -> bool:
        """The two states where the sweep looked and could not say.

        Distinct from "not identifiable for want of treads": UNDECIDED means the
        treads were there and the QUESTION could not be settled by this
        instrument, which is a different thing to report and points at a
        different next experiment.
        """
        return self.outcome in (UNDECIDED_PARALLEL_BRANCH, UNDECIDED_DRIFTING_CLOCK)

    @property
    def load_ms(self) -> float | None:
        """`L = A + B`, one full weight read, the value the memory branch takes
        at a single tile."""
        if self.intercept is None or self.slope_memory is None:
            return None
        return self.intercept + self.slope_memory

    @property
    def alpha(self) -> float | None:
        """`B / (A + B)`, the estimator this study calls alpha. EXA, not a miss
        fraction.

        WHAT THIS NUMBER IS. `moe/bench/ai_model.py` runs this exact estimator
        over three-term traffic and reports what comes back:

            alpha = B/(A+B) = (alpha_b + phi) / (1 + phi + delta)          (EXA)

        `alpha_b` is the weight miss fraction the study NAMES; `phi` is one
        M-tile's activation-and-output cost in units of one full weight read;
        `delta` is the fused layer's fixed cost in the same units. The level
        `A + B` is one full weight read PLUS one tile's activations and output
        PLUS the fixed cost, and every part of that lands in the denominator.
        So this is a FITTED QUANTITY of a particular tiling, not a fraction of
        the weights that missed, and it can be read as one only after
        `ai_model.alpha_b_from_fitted` has taken phi and delta back out.

        THE TWO BIASES, both named, because they are the two extra terms above
        and their signs differ. `delta` is inside the level, which pushes this
        DOWN; `phi` is inside both the slope and the level, and since phi < 1
        its net effect through the slope pushes this UP. Only the activation
        part of phi is correctable per block size, by `activation_slope_ms`, and
        the report prints that correction beside this number as
        `alpha-corrected`. What is NOT done is subtracting an extrapolated fixed
        cost first: on a 4-tread reference ladder under 1% timing spread that
        extrapolation wandered enough to move alpha from 0.56 to 0.70 on data
        planted at 0.558. `alpha_upper` carries that end, and no gate is scored
        on it alone; `alpha_interval` scores both ends at once.

        WHAT A CAP FROM THIS NUMBER COSTS. `2 BM / (alpha b)` treats the level
        as one full weight read, which it is not, so a cap taken that way is
        HIGH by `ai_model.lin_overstatement` = 1 + phi + delta. That factor
        is a BRACKET and not a number, because phi depends on alpha_a, the
        activation re-read fraction, which has no measurement anywhere in this
        repository: `ai_model.overstatement_bracket` gives its two ends, at
        alpha_a = 0 and alpha_a = 1, and with BN=64 on mixtral at delta = 0
        they are 1.02 to 2.02 at BM=64, 1.04 to 3.03 at BM=128 and 1.07 to
        5.06 at BM=256. The 16% / 32% / 64% this docstring once quoted as
        points were the values at alpha_a = 0.143, the withdrawn (LIN)-solved
        pair. Even the LOW end at BM=128, the tile this study's claim is about,
        is of the order of the cap-to-ridge gap the cap was being used to
        decide. The report prints the bracket beside every cap it derives from
        this number (`cap_overstatement`).

        AND SINCE 2026-09-10 IT IS NOT THE ONLY STATISTIC ON THIS FIT.
        `weight_streams` divides the SAME slope `B` by a measured stream time
        instead of by this fitted level, so it carries no intercept, no fixed
        cost and no extrapolation to n = 0. Both are printed and persisted on
        every row. This one is kept, unchanged, because the 100,144 published
        rows were scored on it and have to stay readable; it is not, and was
        never, a weight miss fraction.
        """
        load = self.load_ms
        if load is None or load <= 0 or self.slope_memory is None:
            return None
        return self.slope_memory / load

    @property
    def alpha_upper(self) -> float | None:
        """`B / (L - D)`: alpha with the fused layer's fixed cost taken out.

        The high end of the range. Reported and never gated on, because `D` is
        an extrapolation to zero tiles and a 4-tread ladder under 1% timing
        spread extrapolates it to anywhere between 0.03 and 0.16 ms.

        IT EXCEEDS 1 EXACTLY WHEN `D > A`, which is arithmetic between two
        fitted numbers and not a statement about traffic:
        `B > A + B - D` is `D > A`. `fixed_cost_above_intercept` is that
        condition, computed and printed and written to the report so the state
        is named rather than left for a reader to infer from an out-of-range
        value here. On the 2026-09-10 H200 session it held in four of
        bn_decomposition's six cells and in no others, and those four drove its
        pooled alpha_a to -0.8143.
        """
        load = self.load_ms
        if load is None or self.slope_memory is None:
            return None
        net = load - self.overhead_ms
        return self.slope_memory / net if net > 0 else None

    @property
    def weight_streams(self) -> WeightStreamSlope | None:
        """`w = B / (ms to stream the expert weight set once)`, or None.

        THE SECOND ESTIMATOR, BESIDE `alpha` AND NEVER INSTEAD OF IT. `alpha`
        divides the slope `B` by a fitted LEVEL, which is the ladder
        extrapolated back to zero tiles across up to 44 treads; this divides
        the same `B` by a MEASURED time, the layer's expert weight set over a
        bandwidth the caller supplies. No intercept, no delta, no D, and no
        part of the fit but the slope. On the 2026-09-10 H200 session the two
        say different things about the same ladders and only one of them ever
        returned a value above 1.0 for a quantity bounded by 1.

        Returns a `moe.bench.weights.WeightStreamSlope`, which carries the rate
        and its source alongside the number, because w scales 1:1 in the rate:
        halve the assumed bandwidth and every w halves with it, since the
        stream it counts then takes twice as long. None in THREE cases, all of
        them named on the line by `w_note`: this ladder has no memory branch to
        take a slope from; the caller did not name a model, a dtype and a rate
        (`fit_ladder` leaves all four empty by default, so the sibling scripts
        that do not pass them get None rather than a w against a guessed card);
        or `weights` REFUSED the denominator, for a geometry that is not
        `verified`, an unknown dtype or a rate that is not a rate. The third
        case reached this property on 2026-09-10, when the refusal was found
        travelling out of `analyse` and killing the whole report over one
        supplementary column. `_weight_streams` is where all three are decided,
        once, for every reader of this property.

        WHAT IT IS NOT. Not alpha_b. Under the three-term model this is
        `alpha_b + phi` (`ai_model.slope_weight_streams`), so at the rate the
        weights really stream at it is an UPPER BOUND on the weight miss
        fraction, and phi is the gap: one M-tile's activation and output
        traffic, a function of the unmeasured alpha_a. At any OTHER rate it is
        an upper bound on nothing, and the two rates a card publishes are not
        close: on the H200 of 2026-09-10 the `read_stream` pattern is 5.4%
        above the triad ceiling these reports divide by, and a weight stream is
        a pure read. The report legend carries that condition since 2026-09-10;
        before then the one line a reader actually saw asserted the bound
        flatly, which is the recurring defect on the docstrings around it.
        """
        return self._weight_streams()[0]

    def _weight_streams(self) -> tuple[WeightStreamSlope | None, str]:
        """`(w, reason it is absent)`. Exactly one of the two is filled.

        A REFUSAL IS A BLANK COLUMN, NOT A DEAD REPORT, and one place decides
        which. `weights.weight_streams_per_tile` raises `WeightSetRefused` for
        a model this repository holds no verified geometry for, for an unknown
        dtype and for a rate that is not a rate. Those are the right refusals
        at that layer: its whole contract is that it never guesses a
        denominator. But `w` is a SUPPLEMENTARY column on a report whose other
        forty numbers do not depend on it, and until 2026-09-10 the exception
        travelled all the way out of `analyse`, so a `MoEConfig` added to this
        repository with `verified` still at its default False would have made
        the sweep time every cell on the pod and then die instead of writing
        report.txt and report.json.

        Caught HERE and not at the print sites, because `weight_streams` has
        five reader sites in this file -- the ladder table's w column, three
        per-row payload keys, and the report-level `weight_streams_measured` --
        and `w_note` prints it on three more lines. A fix applied to the two or
        three that were easy to find is this repository's recurring defect. The
        one reader that still needs a catch of its own is the report's legend
        line, which calls `weight_stream_ms` and `routed_expert_weight_bytes`
        directly rather than through a fit, and it has one. The reason string is
        the refusal's own words, so the report says WHY the column is blank
        rather than leaving a reader to guess whether the ladder had no slope or
        the model had no geometry.
        """
        if self.slope_memory is None:
            return None, "no memory branch, so no slope to divide"
        if not self.model or not self.dtype or self.bandwidth_gbps is None:
            return None, ("the caller named no model, dtype and measured "
                          "bandwidth, and there is no default rate")
        try:
            return weight_streams_per_tile(
                self.slope_memory, self.model, self.dtype,
                self.bandwidth_gbps,
                bandwidth_source=self.bandwidth_source), ""
        except WeightSetRefused as exc:
            return None, str(exc)

    @property
    def fixed_cost_above_intercept(self) -> bool | None:
        """`D > A`: the reference fixed cost stands above this ladder's own
        fitted intercept. None when either is missing.

        THE DIAGNOSTIC THAT EXPLAINS EVERY alpha_upper ABOVE 1. `alpha_upper`
        is `B / (A + B - D)`, so `alpha_upper > 1` is exactly `B > A + B - D`,
        which is exactly `D > A`. It is an inequality between two fitted
        numbers, not a statement about traffic: on the 2026-09-10 H200 session
        it held in four of bn_decomposition's six cells and in no others, and
        those four are precisely the cells whose alpha_upper exceeded 1 and
        drove the pooled alpha_a to -0.8143.

        LABELLED, NOT REFUSED. Dropping a ladder in this state would delete
        four of the six cells the BLOCK_N decomposition was fitted over and
        leave two cells for two parameters. The ladder's milliseconds are
        measurements either way; what is not a miss fraction is what
        `alpha_upper` reads on them, and `weight_streams` is the statistic that
        does not depend on either A or D at all.
        """
        if self.intercept is None:
            return None
        return self.overhead_ms > self.intercept

    def w_note(self) -> str:
        """One clause naming `w` and its rate, or naming why there is none.

        ONE RENDERING, THREE CALL SITES. Gate 3's per-BLOCK_M list prints it on
        both of its branches (alpha identified and not) and the line naming the
        alpha gate 3 is scored on prints it a third time; the ladder table
        prints the same fit's `weight_streams` as its own column. A w added at
        one of those and not the rest is this repository's recurring defect
        (eighteen instances) reappearing on the statistic written to fix an
        estimator. This docstring said TWO until 2026-09-10, which is the same
        defect in miniature: a count that stopped matching the file.

        THE D > A CLAUSE IS STATED ON THE SIDE OF THE GUARD IT IS TRUE ON.
        `alpha_upper > 1` is `D > A`, but only where `alpha_upper` exists at
        all: `alpha_upper` returns None when `D >= A + B`, so a ladder with a
        near-zero or negative fitted intercept and a large reference fixed cost
        prints alpha-hi as "n/a" while `fixed_cost_above_intercept` is True.
        Until 2026-09-10 this note said "alpha-hi is above 1" on exactly those
        rows, describing a number the table did not print. The session produced
        three negative intercepts, so the state is reachable in published data.
        """
        w, reason = self._weight_streams()
        if w is None:
            return f"w n/a: {reason}"
        if not self.fixed_cost_above_intercept:
            flag = ""
        elif self.alpha_upper is None:
            flag = ("; D > A, and D exceeds the whole level A + B, so alpha-hi "
                    "is not defined at all")
        else:
            flag = "; D > A, so alpha-hi is above 1 by arithmetic"
        return f"w {w.render()}{flag}"

    @property
    def compute_slope(self) -> float | None:
        """This block size's `C`, measured if it has one and scaled if not."""
        return self.slope_compute if self.slope_compute else self.slope_compute_ref

    @property
    def crosses(self) -> bool | None:
        """`C > B`: does the compute branch ever overtake the memory branch.

        The same statement as `ai_cap > ridge`, made from two fitted slopes
        instead of from two assumed constants. None when a slope is missing,
        which is a different answer from "no" and is reported as a different one.
        """
        c = self.compute_slope
        if self.slope_memory is None or c is None:
            return None
        return c > self.slope_memory

    @property
    def first_compute_tread(self) -> int | None:
        return self.memory_points + 1 if self.memory_points < len(self.points) else None


def _line(xs, ys) -> tuple[float, float]:
    """Ordinary least squares `y = a + b x`. Two points give an exact line."""
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    return my - b * mx, b


def _through_origin(xs, ys) -> float:
    sxx = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys, strict=True)) / sxx if sxx else 0.0


def ladder_treads(cells, block_m: int) -> tuple[list[tuple[int, float]], int]:
    """`(points, cells excluded for a DRIFTING clock)` at exactly-full stacks.

    Aligned only. A tread sampled at 60% fill reports the same TIME as its top
    -- time is flat along a tread -- but a different throughput, and mixing the
    two is how a padding artefact enters a fit that is about traffic.

    AND A CELL WHOSE CLOCK MOVED WHILE IT WAS TIMED IS NOT ON THIS LADDER. Its
    median is a blend of two operating points and the time belongs to neither,
    which no rescaling repairs. `Cell.clock_excluded`, `clock_drift_ok` failed,
    is the only exclusion; None (no NVML, too short a trial, a laptop replay)
    is NOT an exclusion, because an exclusion has to be positively
    established, and the count returned here is what the report says out loud
    so a ladder that lost half its treads cannot look like a ladder that never
    had them.

    A CELL THAT SAT STEADILY OFF THE BAND, ON EITHER SIDE, IS ON THIS LADDER.
    Until 2026-09-09 the LOW side was excluded here. The H200 session showed
    the under-load clock is set per tile by the kernel's own power draw under
    the 700 W cap, so LEVEL-LOW named the study's own dense tiles (BM=128/BN=64
    at a median 1395 MHz, BM=64/G=1 at 1358) and excluding on it removed them
    from measurability on that card; see `Cell.clock_excluded`. Both sides are
    now kept, `off_band_treads` counts them per side for the report, and what
    is not comparable is the fixed-roof FRACTION, not the milliseconds. The
    signature is unchanged so the four sibling scripts that unpack
    `(points, excluded)` keep working.
    """
    pts: dict[int, float] = {}
    excluded = 0
    for c in cells:
        if c.block_m != block_m or not c.aligned or c.status != "ok" or c.ms_p50 <= 0:
            continue
        if c.clock_excluded:
            excluded += 1
            continue
        pts[c.tiles_per_expert] = c.ms_p50
    return sorted(pts.items()), excluded


def ladder_points(cells, block_m: int) -> list[tuple[int, float]]:
    """`ladder_treads` without the exclusion count. The four sibling scripts
    call this name and take a bare list; it stays exactly that."""
    return ladder_treads(cells, block_m)[0]


def off_band_treads(cells, block_m: int) -> tuple[int, int]:
    """`(treads kept LEVEL-low, treads kept LEVEL-high)` on this ladder.

    The counts that go beside `ladder_treads`' exclusion count, over the same
    membership (aligned, ok, timed, not drifted). BOTH SIDES, since 2026-09-09:
    the LOW side used to be excluded and so had nothing to count, and a report
    that names only the HIGH side after both became keepable says of a sagged
    tread neither that it was dropped nor that it was kept. It exists so a
    report can say "N treads ran off the band around the roof's clock, on this
    side; their times are in the fit and their fixed-roof fractions are off by
    the clock ratio in this direction" instead of either dropping them (the
    defect this closes) or keeping them silently (the defect that would replace
    it: a fraction of the fixed roof printed for a cell the fixed roof does not
    describe).
    """
    kept = [c for c in cells
            if c.block_m == block_m and c.aligned and c.status == "ok"
            and c.ms_p50 > 0 and not c.clock_excluded]
    return (sum(1 for c in kept if c.clock_sagged),
            sum(1 for c in kept if c.clock_boosted))


def boosted_treads(cells, block_m: int) -> int:
    """The HIGH half of `off_band_treads`. Kept under its own name because the
    four sibling scripts call it and take an int."""
    return off_band_treads(cells, block_m)[1]


def sagged_treads(cells, block_m: int) -> int:
    """The LOW half of `off_band_treads`, kept and counted since 2026-09-09."""
    return off_band_treads(cells, block_m)[0]


#: A tread has to stand this far above the compute branch to be called memory
#: bound. Below it the two mechanisms predict the same time and the tread
#: carries no information about the re-read fraction, whichever it is. A FLOOR,
#: not the value used: `analyse` raises it to three times the measured timing
#: spread, because the reference slope carries that spread too and a compute
#: branch estimated 2% low makes every compute-bound tread look memory bound.
MEMORY_BRANCH_MARGIN = 0.02

#: How far `B / C` has to sit from 1 before the two branches are two mechanisms.
#:
#: `B / C = ridge / ai_cap` exactly, so a memory branch running parallel to the
#: compute branch is a block size sitting precisely on its own crossing -- and
#: is far more often a fit that took a stretch of the compute branch for a
#: memory branch. At 2% timing spread that happened on 2 seeds in 20 at
#: BLOCK_M=128, reporting alpha 0.80 on cells planted at 0.10 and passing gate 3
#: with it. Rejecting the parallel case costs only block sizes whose ceiling
#: lands within 15% of the ridge, where the answer is undecidable anyway; the
#: four this sweep runs sit at ratios of 2.80, 1.40, 0.70 and 0.35.
PARALLEL_BRANCH_TOLERANCE = 0.15


#: How far ABOVE `ridge x bandwidth` a qualified compute reference is allowed to
#: sit, as a fraction of the roof.
#:
#: A compute branch cannot run faster than the compute roof, so the physical
#: bound is 1.0 and everything past it is slack for the ruler. The slack is
#: taken from the study's own ridge band: three calibrations of the same H200
#: disagreed by 9.9% on the compute term, so a reference within 10% of the roof
#: is inside the calibration's disagreement with itself. Beyond that the arm is
#: claiming to beat its card, which means the roof belongs to a different
#: machine or the FLOP count is wrong -- and both make every roof fraction, every
#: membership decision and every alpha in the report meaningless.
REFERENCE_ROOF_CEILING = 1.10

#: How much SLOWER than the fastest smaller block size, at a matched
#: exactly-full row count, a compute reference may be.
#:
#: At `r = n BM` there is no padding at any block size on the grid, so at a
#: matched `r` every block size does IDENTICAL useful arithmetic, and the larger
#: tile additionally does strictly FEWER weight re-reads (`Q(n)` falls as `n`
#: falls). Under the model the larger tile can therefore only be faster. Slower
#: at all is the model failing; the tolerance covers the timing spread and the
#: coarser wave quantum a big tile leaves at the end of a launch. 1.25 is set
#: above the worst value any sound published reference reaches -- 1.094, over 24
#: references on two cards -- and 4.6x below the smallest corrupt one, 5.723.
REFERENCE_LEVEL_TOLERANCE = 1.25


#: Maximum shared memory ONE thread block may opt into, in bytes, by compute
#: capability. Not the per-SM total: a CTA cannot exceed these however much the
#: SM holds. sm_80 has 164 KiB per SM and lets a block take 163; sm_90 has 228
#: and lets a block take 227. UNKNOWN CAPABILITIES ARE NOT DEFAULTED. A missing
#: entry makes the shared-memory verdict "unknown", never "fits", because a
#: guessed limit that is too generous is exactly the failure this table exists
#: to catch.
SMEM_PER_BLOCK_BYTES: dict[tuple[int, int], int] = {
    (7, 0): 98304,      # V100
    (7, 5): 65536,      # T4
    (8, 0): 166912,     # A100
    (8, 6): 101376,     # A10 / A40 / RTX 30
    (8, 9): 101376,     # L4 / L40S / RTX 40
    (9, 0): 232448,     # H100 / H200
    (10, 0): 232448,    # B200
}

#: A CUDA thread may address 255 registers, on every architecture this study can
#: run on. It is not a tuning parameter and there is no opt-in past it.
MAX_REGISTERS_PER_THREAD = 255


@dataclass(frozen=True)
class TileResources:
    """What one CTA of a given tile setting must be given, against what exists.

    THE CHECK THAT WOULD HAVE STOPPED THE BN=256 ARM BEFORE IT WAS TIMED. Two
    hard limits, computed from the pinned constants alone, so both are knowable
    off-GPU before a single cell runs:

      * SHARED MEMORY. Triton multi-buffers the K loop, so one CTA holds
        `num_stages` copies of the `BM x BK` A tile and the `BK x BN` B tile:
        `num_stages (BM BK + BK BN) b` bytes. At BM=BN=256, BK=64, bf16 and 3
        stages that is 192 KiB, which fits an H200's 227 KiB per-block ceiling
        and does NOT fit an A100's 163 KiB. At 4 stages it is 256 KiB and fits
        neither -- and in the two published num_stages=4 BN=256 arms the
        BLOCK_M=256 ladder is simply ABSENT, which is that cliff.

      * REGISTERS. `tl.dot` accumulates in fp32, so one CTA holds a `BM x BN`
        fp32 accumulator in registers: `BM BN / (32 num_warps)` registers per
        thread. At BM=BN=256 with num_warps=8 that is 256 per thread against a
        hardware maximum of 255, so the accumulator ALONE cannot be held and the
        kernel spills to local memory. This one is card-independent, which is
        why the setting is slow on BOTH cards; the A100 is 43.6x slow and the
        H200 only 3.92x because only the A100 also blows the shared-memory
        limit.

    WHY THIS IS A REFUSAL AND NOT A WARNING. A spilled kernel still returns a
    time, that time still fits a straight line through the origin, and that line
    still qualified as this study's compute reference at 0.2% error. Every tread
    in the arm was then classified against a compute branch 44x too steep. The
    setting has to be refused where it is chosen, not diagnosed afterwards.
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    dtype_bytes: int
    smem_bytes: int
    acc_registers_per_thread: float
    smem_limit_bytes: int | None
    capability: tuple[int, int] | None

    @property
    def registers_fit(self) -> bool:
        return self.acc_registers_per_thread <= MAX_REGISTERS_PER_THREAD

    @property
    def smem_fits(self) -> bool | None:
        """None means the device is unknown, which is not the same as True."""
        if self.smem_limit_bytes is None:
            return None
        return self.smem_bytes <= self.smem_limit_bytes

    @property
    def refusal(self) -> str:
        """Empty when the setting may be timed; otherwise why it may not be."""
        why = []
        if not self.registers_fit:
            why.append(
                f"the {self.block_m}x{self.block_n} fp32 accumulator needs "
                f"{self.acc_registers_per_thread:.0f} registers per thread at "
                f"num_warps={self.num_warps}, against a hardware maximum of "
                f"{MAX_REGISTERS_PER_THREAD}. The accumulator alone does not "
                "fit, so the kernel spills to local memory and its time is not "
                "the time of the tiling this sweep is about")
        if self.smem_fits is False:
            why.append(
                f"one CTA needs {self.smem_bytes / 1024:.0f} KiB of shared "
                f"memory ({self.num_stages} stages x ({self.block_m}x"
                f"{self.block_k} + {self.block_k}x{self.block_n}) x "
                f"{self.dtype_bytes} B) against sm_{self.capability[0]}"
                f"{self.capability[1]}'s {self.smem_limit_bytes / 1024:.0f} KiB "
                "per-block ceiling")
        return "; ".join(why)

    def render(self) -> str:
        smem = f"{self.smem_bytes / 1024:6.0f} KiB"
        limit = ("      ?" if self.smem_limit_bytes is None
                 else f"{self.smem_limit_bytes / 1024:6.0f} KiB")
        return (f"  BLOCK_M={self.block_m:4d}  smem {smem} of {limit}  "
                f"acc {self.acc_registers_per_thread:5.0f} reg/thread of "
                f"{MAX_REGISTERS_PER_THREAD}  "
                + ("REFUSED" if self.refusal else "ok"))


def tile_resources(pinned: dict, block_m: int, dtype_bytes: int,
                   capability: tuple[int, int] | None) -> TileResources:
    """One CTA's shared-memory and accumulator-register bill for a setting.

    Pure arithmetic on the pinned constants, so `--dry-run` on a laptop prints
    the same numbers the pod would refuse on.
    """
    bn = pinned["BLOCK_SIZE_N"]
    bk = pinned["BLOCK_SIZE_K"]
    stages = pinned["num_stages"]
    warps = pinned["num_warps"]
    smem = stages * (block_m * bk + bk * bn) * dtype_bytes
    # fp32 accumulator, one register per element, spread over the CTA's threads.
    acc = block_m * bn / (32.0 * warps)
    return TileResources(
        block_m=block_m, block_n=bn, block_k=bk, num_stages=stages,
        num_warps=warps, dtype_bytes=dtype_bytes, smem_bytes=smem,
        acc_registers_per_thread=acc,
        smem_limit_bytes=SMEM_PER_BLOCK_BYTES.get(capability)
        if capability else None,
        capability=capability)


def parse_capability(text: str) -> tuple[int, int] | None:
    """`"9.0"` -> `(9, 0)`. Empty gives None, which means UNKNOWN, not "fine"."""
    if not text:
        return None
    major, _, minor = text.partition(".")
    return int(major), int(minor or 0)


def resolve_capability(args, *, synthetic: bool) -> tuple[int, int] | None:
    """The attached device's capability, or the one named on the command line.

    A synthetic run has no device, so the shared-memory limit is genuinely
    unknown there and is REPORTED as unknown. The register ceiling is 255 on
    every architecture this can run on, so the check that catches the BN=256
    accumulator still fires on a laptop.
    """
    named = parse_capability(args.capability)
    if named is not None or synthetic:
        return named
    try:
        import torch
        return tuple(torch.cuda.get_device_capability(0))
    except Exception:                                   # noqa: BLE001
        return None


def tile_resource_plan(pinned: dict, block_sizes, dtype_bytes: int,
                       capability: tuple[int, int] | None
                       ) -> tuple[dict[int, TileResources], dict[int, str]]:
    """Every block size's bill, and the refusals among them.

    Returned rather than printed so `main` can print it in the plan, drop the
    refused settings from the sweep, and carry the refusals into the report --
    three places that must agree, from one computation.
    """
    plan = {bm: tile_resources(pinned, bm, dtype_bytes, capability)
            for bm in block_sizes}
    return plan, {bm: r.refusal for bm, r in plan.items() if r.refusal}


@dataclass(frozen=True)
class ComputeReference:
    """The compute branch, and the fused layer's fixed cost, from one ladder.

    TWO THINGS THAT CANNOT BE MEASURED INSIDE A MEMORY-BOUND LADDER, taken from
    a ladder that is compute bound throughout instead.

    The FIXED COST `D` -- router, align, activation, launch -- enters every
    point of a memory-bound ladder identically, so `D` and the memory branch's
    intercept `A` are one number there and `alpha = B / (A + B)` comes out low
    by whatever fraction of the level `D` is. On a ladder that is compute bound
    at every tread, `t(n) = D + C n` has two free parameters and three or more
    points, so `D` separates.

    The COMPUTE SLOPE `C` is what decides membership for every other ladder, and
    `C = 2 BM N_w / peak` is proportional to BLOCK_M with no free parameter, so
    one measurement covers all of them.

    THE ASSUMPTION IS THAT THE LARGEST BLOCK SIZE IS COMPUTE BOUND THROUGHOUT,
    which is a prediction of the model under test, so it is checked rather than
    taken: the ladder has to be straight through a non-negative intercept. When
    it is not, `block_m` is None, alpha downstream becomes a LOWER bound, and
    the membership test falls back to the split search. Degrading to "cannot
    say" is the point; a reference taken from a ladder that was secretly memory
    bound would put `C` at the memory slope and quietly make every block size
    look identifiable.

    SHAPE IS NOT LEVEL, AND THE ORIGINAL QUALIFICATION ONLY TESTED SHAPE. A line
    44x too steep is still perfectly proportional. In the BN=256 arm the
    BLOCK_M=256 reference took 249.765 ms for one tile on the A100 against 5.724
    ms for the identical setting in its BN=64 twin, and qualified at 0.2% mean
    error, because through-origin residual is scale free. Every tread in the arm
    was then classified against that branch, no tread could stand above it, all
    8 cells came out "not identifiable", and the arm printed as a boring null.
    `refusals` carries the LEVEL checks that now have to pass as well, and
    `refused_block_m` names the ladder they rejected, so a reader can tell a
    refused reference from a sweep that never had a candidate.
    """

    block_m: int | None
    overhead_ms: float
    slope_per_tile: float | None
    mean_rel_err: float
    note: str
    #: The ladder the level checks REJECTED, when they rejected one. None both
    #: when a reference qualified and when no candidate existed at all, which is
    #: why `refusals` and not this field is what says a refusal happened.
    refused_block_m: int | None = None
    #: One line per failed level check, empty when the reference is usable.
    refusals: tuple[str, ...] = ()
    #: `C` at the roof over `C` measured: the fraction of `ridge x bandwidth`
    #: the reference ladder actually reached. 1.0 is the roof, and 0.013 is what
    #: the corrupt A100 BN=256 reference reached.
    roof_fraction: float | None = None
    #: `C` scaled to the smallest swept block size, over one full weight read.
    #: At or above 1 the reference makes memory-boundness impossible everywhere.
    #: On the FUSED footing (`fused_roof_band` given) both sides of that
    #: comparison are put on the layer's own roof rather than the dense GEMM's;
    #: `vacuity_basis` says which footing produced the number.
    vacuity_ratio: float | None = None
    #: `dense` or `fused`: which roof the non-vacuity comparison stands on.
    vacuity_basis: str = "dense"
    #: The reference's own fraction of `ridge x bandwidth`, when the fused
    #: footing was asked for: the CONTROL'S MEASURED PLATEAU, which is the
    #: fused layer's roof on this card. None when the footing was not asked for.
    fused_roof_fraction: float | None = None
    #: The band that measured plateau has to land in for the fused footing to
    #: be usable, `(floor, ceiling)` of `ridge x bandwidth`. None when the
    #: footing was not asked for.
    fused_roof_band: tuple[float, float] | None = None
    #: The floor the reference's roof fraction had to clear, in roof units, on
    #: whichever footing was used. Recorded so the page can print the
    #: derivation rather than only the verdict.
    vacuity_floor: float | None = None
    #: One line per candidate ladder that was passed over before a reference
    #: was reached or refused, and why. Distinct from `refusals`, which is the
    #: LEVEL verdict on the candidate that was actually tried: a ladder skipped
    #: for want of treads was never judged, and a report that prints neither
    #: leaves a reader unable to say which ladder the reference is.
    skipped: tuple[str, ...] = ()
    #: Worst `t_reference(r) / min t_smaller(r)` over matched exactly-full row
    #: counts, and how many such comparisons there were. A zero count means the
    #: cross-ladder level check EXAMINED NOTHING and must not read as a pass.
    level_ratio: float | None = None
    level_comparisons: int = 0

    @property
    def refused(self) -> bool:
        """A candidate existed and the level checks threw it out."""
        return bool(self.refusals)

    def slope_for(self, block_m: int) -> float | None:
        """`C` at another block size, by `C ~ BLOCK_M`."""
        if self.slope_per_tile is None or self.block_m is None:
            return None
        return self.slope_per_tile * block_m / self.block_m

    def render(self) -> list[str]:
        """The qualification, as numbers against thresholds.

        Printed whether it passed or failed. The defect this exists to catch was
        invisible precisely because a refused reference and an uninformative
        sweep printed the same blanks downstream.
        """
        out = [f"  compute reference: {self.note}"]
        if self.roof_fraction is not None:
            out.append(
                f"    LEVEL roof fraction   {self.roof_fraction:8.3f}   "
                f"gate <= {REFERENCE_ROOF_CEILING:.2f} of ridge x bandwidth "
                f"(a compute branch cannot beat the roof)")
        if self.vacuity_ratio is not None:
            # THE LABEL SAYS WHICH FOOTING THE RATIO STANDS ON. `_level_checks`
            # appends ", on the fused layer's own roof," to the refusal text
            # when `fused_roof_band` is in force, and until 2026-09-09 this
            # line -- the one a PASSING reference prints, and so the one most
            # readers meet -- carried the dense-footing wording at both
            # footings. On the fused footing the number is not a fraction of
            # one full weight read taken against the dense GEMM roof.
            footing = (", on the fused layer's own roof,"
                       if self.vacuity_basis == "fused" else ",")
            out.append(
                f"    LEVEL non-vacuity     {self.vacuity_ratio:8.3f}   "
                f"gate <  1.00 of one full weight read{footing} scaled to the "
                "smallest block size (at or above, NO tread anywhere can be "
                "memory bound and every alpha is unidentifiable by "
                "construction)")
            out += self.vacuity_derivation()
        if self.level_comparisons:
            out.append(
                f"    LEVEL vs smaller BM   {self.level_ratio:8.3f}   "
                f"gate <= {REFERENCE_LEVEL_TOLERANCE:.2f} at matched "
                f"exactly-full rows, over {self.level_comparisons} comparison(s)")
        else:
            out.append(
                "    LEVEL vs smaller BM   NOT CHECKED   no smaller ladder "
                "shares an exactly-full row count with the reference, so this "
                "check examined nothing and is not a pass")
        for why in self.skipped:
            out.append(f"    PASSED OVER: {why}")
        for why in self.refusals:
            out.append(f"    REFUSED: {why}")
        return out

    def vacuity_derivation(self) -> list[str]:
        """The non-vacuity floor written out, not just scored.

        PRINTED WHETHER IT PASSED OR FAILED. The bm128_depth arm was
        pre-registered INVALID for a whole session because its report said only
        "its LEVEL is wrong" over a vacuity refusal, and a reader could not see
        that the floor being demanded was 83.8% of the DENSE GEMM roof from a
        fused layer that has never exceeded 75.6% of it in 26 published
        reports. The arithmetic below is what makes that visible.
        """
        if self.vacuity_ratio is None:
            return []
        out = []
        if self.vacuity_floor is not None:
            out.append(
                f"      floor              {self.vacuity_floor:8.3f}   "
                "of the roof, = 2 BM_min / (b x ridge): the smallest swept "
                "tile's own AI cap at alpha=1, over the ridge")
        if self.vacuity_basis == "fused" and self.fused_roof_band:
            lo, hi = self.fused_roof_band
            frac = self.fused_roof_fraction
            out.append(
                "      footing            FUSED. The compute branch and the "
                "memory branch are both measured through the same fused layer, "
                "so the comparison is made on that layer's own roof, "
                + (f"{frac:.3f}" if frac is not None else "unknown")
                + " x ridge x bandwidth (the CONTROL'S MEASURED PLATEAU), and "
                f"not on the dense GEMM roof. That plateau has to land in "
                f"[{lo:.3f}, {hi:.3f}]: the floor is the lowest fused plateau "
                "in this study's 26 published reports, the ceiling is the "
                "dense peak plus the tolerance a run generated AT the roof "
                "needs, and a plateau above it says the ruler belongs to "
                "another machine. A reference outside the band is refused "
                "here. The 26 published plateaus themselves run 0.465 to "
                "0.756, well inside it; the band is what may be ADMITTED and "
                "not where they sit.")
        elif self.vacuity_basis == "dense":
            out.append(
                "      footing            DENSE. The memory branch is compared "
                "against one full weight read at the measured pin rate, so the "
                "floor is a fraction of the dense GEMM roof. A fused layer "
                "counting only its two GEMMs' FLOPs has never exceeded 0.756 "
                "of that roof, so a floor above 0.756 cannot be met by any "
                "fused-layer reference on any card.")
        return out


def _level_checks(cells, block_sizes, bm: int, c: float, *, cfg, ridge: float,
                  bandwidth_gbps: float, b: int, pinned: dict | None,
                  capability, fused_roof_band: tuple[float, float] | None = None
                  ) -> tuple[list[str], dict]:
    """Is the candidate's per-tile slope the RIGHT SIZE, not just the right shape.

    Three independent readings of the same number, returned with the numbers so
    the report can print them whether they passed or not.

      1. ROOF CEILING. `C` at the roof is `6 E BM F H / (ridge x bandwidth)`.
         The measured slope cannot be smaller than that: nothing runs faster
         than the compute roof. Above `REFERENCE_ROOF_CEILING` the ruler belongs
         to another machine or the FLOP count is wrong.

      2. NON-VACUITY, and this is the one that catches the BN=256 arm. Scale `C`
         to the SMALLEST swept block size and compare it with ONE FULL WEIGHT
         READ `L = E 3 F H b / bandwidth`. A tread is memory bound when it
         stands above the compute branch, and the highest a memory branch can
         ever sit is `alpha = 1`, one full re-read per tile. So if the scaled
         compute branch is already at or above `L`, NO tread at ANY block size
         can be classified memory bound -- the report's blanks are then a
         property of the reference, not a measurement. Equivalently in roof
         units the floor is `2 BM_min / (b ridge)`, the smallest block size's
         own AI cap at alpha=1 over the ridge. The A100 BN=256 reference sits at
         15.9x this bound and the H200 one at 1.6x; every sound published
         reference sits between 0.31 and 0.52.

         WHICH ROOF THAT COMPARISON STANDS ON, and the defect fixed on
         2026-09-09. `L` is one weight read at the measured PIN RATE and `C` is
         a fused layer's measured slope, so the ratio silently asks the fused
         layer to reach a fraction of the DENSE GEMM roof: `2 BM_min / (b
         ridge)` = 0.838 for BM_min=128 on the H200. No fused layer in this
         study's 26 published reports has exceeded 0.756 of the dense roof, so
         with a {128, 256} pairing the check refused its reference on every
         card and every clock rule, and bm128_depth was pre-registered INVALID
         before a cell ran. `fused_roof_band` opts into the FUSED footing: the
         reference's own measured plateau IS the fused layer's roof on this
         card, both branches are put on it, and the ratio becomes
         `2 BM_min / (b ridge)` outright -- a statement about whether the
         sweep's smallest tile can be memory bound AT ALL, which is what
         non-vacuity always claimed to be. The measurement check does not
         disappear with it: on that footing the measured plateau must itself
         land inside `fused_roof_band`, the corpus interval
         `tile_cap_test.FUSED_PLATEAU_BAND[0]` to the dense peak plus its
         tolerance, which the corrupt A100 BN=256 reference (0.013) misses by
         two orders of magnitude. Left as None the dense footing is unchanged,
         so the sibling sweeps score exactly as before.

      3. AGAINST THE SWEEP'S OWN SMALLER LADDERS. At `r = n BM` nothing is
         padded at any block size, so a matched `r` is identical useful
         arithmetic with strictly fewer weight re-reads for the bigger tile: the
         reference can only be faster. `REFERENCE_LEVEL_TOLERANCE` is the slack.
         This check needs no roof and no calibration at all, which is what makes
         it worth having beside the other two -- and when the reference is the
         smallest ladder swept there is nothing to compare against, which is
         reported as NOT CHECKED rather than silently as a pass.

    A tile setting that cannot physically run is checked here too, because a
    spilled kernel produces a time that is proportional to its tile count and
    therefore sails through the shape test -- which is exactly how 249.765 ms
    became this study's compute branch.
    """
    why: list[str] = []
    nums: dict = {"roof_fraction": None, "vacuity_ratio": None,
                  "level_ratio": None, "level_comparisons": 0,
                  "vacuity_basis": "dense", "fused_roof_fraction": None,
                  "fused_roof_band": None, "vacuity_floor": None}

    if pinned is not None:
        res = tile_resources(pinned, bm, b, capability)
        if res.refusal:
            why.append(
                f"BLOCK_M={bm} cannot run as pinned: {res.refusal}. Its timing "
                "is not a measurement of this tiling and must not become the "
                "compute branch every other ladder is classified against")

    roof_flops = ridge * bandwidth_gbps * 1e9
    if roof_flops > 0:
        c_roof = 1e3 * useful_flops(cfg, cfg.num_experts * bm) / roof_flops
        nums["roof_fraction"] = c_roof / c
        if nums["roof_fraction"] > REFERENCE_ROOF_CEILING:
            why.append(
                f"BLOCK_M={bm} runs at {nums['roof_fraction']:.3f} of "
                f"ridge x bandwidth, past the {REFERENCE_ROOF_CEILING:.2f} "
                "ceiling. A compute branch cannot beat the compute roof, so "
                "either the roof was calibrated on a different machine or the "
                "FLOP count is wrong; either way nothing downstream is scored "
                "against a real ceiling")

    if bandwidth_gbps > 0 and block_sizes:
        bm_min = min(block_sizes)
        full_read_ms = (1e3 * cfg.num_experts * weight_bytes_per_expert(cfg, b)
                        / (bandwidth_gbps * 1e9))
        nums["vacuity_ratio"] = (c * bm_min / bm) / full_read_ms
        nums["vacuity_floor"] = 2.0 * bm_min / (b * ridge) if ridge > 0 else None
        scaled_note = (f"BLOCK_M={bm}'s compute branch scaled to "
                       f"BLOCK_M={bm_min}")
        if fused_roof_band is not None:
            # THE FUSED FOOTING. `roof_fraction` above is this reference's own
            # measured plateau as a fraction of ridge x bandwidth, which for a
            # ladder measured through `fused_experts` IS that layer's roof on
            # this card. Deflating one full weight read to the same footing
            # multiplies the ratio by it, and the product is exactly
            # 2 BM_min / (b ridge): the smallest swept tile's AI cap at alpha=1
            # over the ridge, with the reference's own level dropping out. The
            # level is not thereby unchecked -- it is checked HERE, against the
            # corpus band, where a 1.4%-of-roof reference is caught by two
            # orders of magnitude instead of by a bound no fused layer meets.
            lo, hi = fused_roof_band
            plateau = nums["roof_fraction"]
            nums["vacuity_basis"] = "fused"
            nums["fused_roof_band"] = (lo, hi)
            nums["fused_roof_fraction"] = plateau
            if plateau is None:
                why.append(
                    f"BLOCK_M={bm} has no measured plateau (no roof was "
                    "resolvable), so the fused footing has nothing to stand on "
                    "and the non-vacuity floor cannot be derived")
            elif not lo <= plateau <= hi:
                why.append(
                    f"BLOCK_M={bm}'s measured plateau is {plateau:.3f} of "
                    f"ridge x bandwidth, outside the [{lo:.3f}, {hi:.3f}] this "
                    "arm admits for a fused layer's roof: the floor is the "
                    "lowest of the 26 published fused plateaus and the ceiling "
                    "is the dense peak plus its tolerance. Below the floor "
                    "nothing here reached any roof, fused or dense, and above "
                    "the ceiling the ridge, the "
                    "bandwidth or the FLOP count belongs to another machine; "
                    "either way this ladder cannot be the compute branch every "
                    "other one is classified against")
            else:
                nums["vacuity_ratio"] *= plateau
                scaled_note += ", on the fused layer's own roof,"
        if nums["vacuity_ratio"] >= 1.0:
            why.append(
                f"{scaled_note} is {nums['vacuity_ratio']:.3f} of one full "
                "weight read. A memory branch cannot exceed one full re-read "
                "per tile (alpha <= 1), so no tread at any block size in this "
                "sweep could stand above this line: every 'not identifiable' "
                "below would be a property of the reference and not a "
                "measurement")

    ratios = []
    smaller = [s for s in block_sizes if s < bm]
    if smaller:
        ref_pts = dict(ladder_points(cells, bm))
        others = {s: dict(ladder_points(cells, s)) for s in smaller}
        for n, t_ref in ref_pts.items():
            rows = n * bm
            peers = [pts[rows // s] for s, pts in others.items()
                     if rows % s == 0 and (rows // s) in pts]
            if peers and t_ref > 0:
                ratios.append(t_ref / min(peers))
    if ratios:
        nums["level_ratio"] = max(ratios)
        nums["level_comparisons"] = len(ratios)
        if nums["level_ratio"] > REFERENCE_LEVEL_TOLERANCE:
            why.append(
                f"BLOCK_M={bm} is {nums['level_ratio']:.3f}x slower than the "
                f"best smaller block size at a matched exactly-full row count, "
                f"past the {REFERENCE_LEVEL_TOLERANCE:.2f} tolerance, over "
                f"{len(ratios)} comparison(s). At a matched full stack the "
                "bigger tile does the same arithmetic and strictly fewer weight "
                "re-reads, so under the model it cannot be slower at all")
    return why, nums


#: The machine-readable half of "why is this cell blank". Kept as fixed tokens
#: rather than prose so a downstream table can branch on them; the prose is in
#: `LadderFit.basis` and in `ComputeReference.render`.
NOT_IDENTIFIABLE_REFERENCE_REFUSED = "reference_refused"
NOT_IDENTIFIABLE_NO_REFERENCE = "no_compute_reference"
NOT_IDENTIFIABLE_TOO_FEW_TREADS = "too_few_memory_treads"
NOT_IDENTIFIABLE_IS_REFERENCE = "is_the_reference_ladder"


def _why_not_identifiable(fit, ref) -> str:
    """Empty when the fit IS identifiable, else the reason it is not.

    THE CONFLATION THIS EXISTS TO BREAK. All 8 cells of the BN=256 arm printed
    as blanks under a caption reading "fewer than 3 memory-bound treads". The
    treads were there; the compute reference was 44x too steep, so nothing could
    stand above it. Those two states have to be told apart at the row level,
    because the arm's whole defect was invisible while they were not.
    """
    if fit.memory_points >= MIN_MEMORY_TREADS and fit.alpha is not None:
        return ""
    if ref is not None and ref.refused:
        return NOT_IDENTIFIABLE_REFERENCE_REFUSED
    if ref is not None and ref.block_m == fit.block_m:
        return NOT_IDENTIFIABLE_IS_REFERENCE
    if ref is None or ref.block_m is None:
        return NOT_IDENTIFIABLE_NO_REFERENCE
    return NOT_IDENTIFIABLE_TOO_FEW_TREADS


def compute_reference(cells, block_sizes, max_err: float = 0.05, *,
                      cfg, ridge: float, bandwidth_gbps: float, b: int,
                      pinned: dict | None = None, capability=None,
                      candidates=None,
                      fused_roof_band: tuple[float, float] | None = None
                      ) -> ComputeReference:
    """Qualify the largest ladder as a compute branch, or decline.

    THE QUALIFICATION IS PROPORTIONALITY, not the sign of an extrapolated
    intercept. `t = C n` through the origin is one parameter and stays put under
    noise; `t = D + C n` is two, and on the four treads BLOCK_M=256 gets at
    r_max=1024 a 3% timing spread pushed `D` to -0.097 ms and the slope 7% high.
    Rejecting on that sign threw the reference away exactly when the data was
    noisy, which sent every ladder to the split search and let BLOCK_M=256's own
    straight line be read as a memory branch with alpha 1.097.

    So `C` comes from the through-origin fit and the residual of THAT decides
    whether the ladder is a compute branch. It discriminates: a memory-bound
    ladder `A + B n` with a real intercept cannot be described by a line through
    the origin and misses by 11% on this grid, well past `max_err`.

    `C` then absorbs a little of the fixed cost, which biases the compute branch
    slightly HIGH and so makes membership slightly conservative: a tread has to
    clear a marginally higher line to be called memory bound. That is the
    direction to be wrong in, since every failure this file defends against is a
    tread wrongly called memory bound.

    PROPORTIONALITY IS NECESSARY AND NOWHERE NEAR SUFFICIENT, and believing
    otherwise cost this study 8 published cells. It is a test of SHAPE and a
    line 44x too steep has the right shape. `_level_checks` is the test of
    LEVEL, it runs after the shape test on the same candidate, and a candidate
    that fails it is REFUSED rather than warned about, because the reference is
    what every other ladder in the report is classified against: a bad one does
    not add noise, it decides the answer.

    A REFUSAL DOES NOT FALL THROUGH TO THE NEXT BLOCK SIZE. The next-largest
    ladder is measured under the same pinned constants on the same card, so the
    thing that broke the level is very likely still there; taking the runner-up
    would replace a loud refusal with a quiet, differently-wrong reference.

    AND NEITHER DOES A SHORT ONE, WHICH IS WHAT `candidates` IS FOR. Until
    2026-09-09 the "too few treads" arm of that loop was a bare `continue`, so
    a two-tile experiment whose CONTROL had under 3 treads fell through to the
    next ladder down -- the memory-bound SUBJECT of the experiment. In the
    2026-09-09 cap_test arm that was BLOCK_M=16: it fitted at 0.72% error and
    was then refused on non-vacuity at 1.044, which is physically correct (a
    per-tile slope equal to one full weight read IS alpha ~ 1) and was rendered
    as "BLOCK_M=16 ... its LEVEL is wrong" with the 1.044 never printed, while
    the control's two treads were never mentioned. `candidates` names which
    ladders may BE the reference; `block_sizes` still names every ladder the
    level checks compare against, so the two roles cannot be conflated again.
    Every ladder passed over is recorded in `skipped` and printed by `render`.
    `fused_roof_band` is handed to `_level_checks`; see there.
    """
    tried = tuple(sorted(candidates if candidates is not None else block_sizes,
                         reverse=True))
    skipped: list[str] = []
    for bm in tried:
        pts = ladder_points(cells, bm)
        if len(pts) < 3:
            skipped.append(
                f"BLOCK_M={bm} has {len(pts)} aligned tread(s), under the 3 a "
                "through-origin fit needs to be a qualification rather than a "
                "line through two points; it was NOT tried as the reference "
                "and it was NOT refused")
            continue
        xs = [float(n) for n, _ in pts]
        ys = [ms for _, ms in pts]
        c = _through_origin(xs, ys)
        err = statistics.fmean(abs(c * x - y) / y
                               for x, y in zip(xs, ys, strict=True))
        if c <= 0 or err > max_err:
            return ComputeReference(
                None, 0.0, None, err,
                f"BLOCK_M={bm} ladder is not proportional to its tile count "
                f"({err:.1%} mean error against a line through the origin), so "
                "it is not compute bound throughout and cannot provide a "
                "compute branch. Membership falls back to a split search and NO "
                "alpha may decide a verdict",
                skipped=tuple(skipped))
        why, nums = _level_checks(
            cells, block_sizes, bm, c, cfg=cfg, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
            capability=capability, fused_roof_band=fused_roof_band)
        if why:
            return ComputeReference(
                None, 0.0, None, err,
                f"BLOCK_M={bm} ladder is proportional to {err:.1%} but its "
                "LEVEL is wrong, so it is REFUSED as a compute branch. "
                "Membership falls back to a split search, NO alpha may decide a "
                "verdict, and every 'not identifiable' in this report is "
                "CAUSED BY THIS REFUSAL rather than by a sweep that lacked "
                "treads",
                refused_block_m=bm, refusals=tuple(why),
                skipped=tuple(skipped), **nums)
        # Reported, and used only for `alpha_upper` and to shift the compute
        # branch. Clamped at zero because a negative fixed cost is a fitting
        # artefact and subtracting one would inflate every alpha.
        intercept, _ = _line(xs, ys)
        return ComputeReference(
            bm, max(0.0, intercept), c, err,
            f"BLOCK_M={bm} ladder, {len(pts)} treads, proportional to "
            f"{err:.1%}: compute branch {c:.4f} ms per tile, fixed cost "
            f"{max(0.0, intercept):.4f} ms",
            skipped=tuple(skipped), **nums)
    return ComputeReference(
        None, 0.0, None, math.inf,
        "no candidate ladder ("
        + ", ".join(f"BLOCK_M={bm}" for bm in tried)
        + ") had the 3 treads needed to qualify a compute branch. "
        "Membership falls back to a split search and NO alpha may decide a "
        "verdict",
        skipped=tuple(skipped))


def memory_branch_members(xs, ys, c_ref: float, overhead: float,
                          margin: float) -> tuple[int, int, list[bool]]:
    """`(start index, count, per-tread verdicts)` of the memory branch.

    THE RULE THAT REPLACED THE PREFIX RULE ON 2026-09-02, and the row it was
    costing. A tread is memory bound when it stands more than `margin` above
    `overhead + C n`. The old rule then took the LEADING RUN FROM n=1 and
    stopped at the first tread that did not qualify, so a single tread at n=1
    inside the margin set `k = 0` and threw the entire branch away. In the H200
    mixtral G=1 arm the n=1 tread sits 3.30% above the scaled compute line
    against a 4.20% margin -- a miss of 0.003 ms -- while treads 2 through 8 all
    sit 7.3-9.3% above it. So `k = 0`, alpha was imported from BLOCK_M=64, and
    `crosses` came out None for BLOCK_M=128 in EVERY arm: the one tile the
    paper's claim is about has never been identified by its own ladder.

    WHY n=1 IS THE TREAD THAT GOES MISSING, and why it is right to let the
    branch start above it. The n=1 cell is the shortest kernel in the ladder and
    therefore the one carrying the largest HOST prefix as a fraction of its
    time: the audit bounded that prefix at 0.18 ms per fused_experts call on the
    H200 and 0.30 ms on the A100, which is 8-16% of a small tread and is a
    per-card constant, not traffic. It is also the tread whose elevation is
    G-dependent in a way the constant prefix does not explain, which is the
    thing `scripts/memory_branch_anchor.py` exists to measure. A rule that lets
    that one tread veto the other seven is a rule that hands the answer to the
    least trustworthy point on the ladder.

    So the branch is the LONGEST CONTIGUOUS RUN of memory-bound treads, ties
    going to the lowest n. Contiguity is kept because memory-boundness really is
    a prefix property of the underlying curve -- `Q` grows by alpha per tile and
    the compute branch by 1, so once compute is on top it stays there -- and a
    scattered subset would be noise picking its own points. What is dropped is
    only the requirement that n=1 be in the run. Treads BELOW the run's start
    are neither on the branch nor evidence against it, and the caller is told
    where the run began.

    WHY THE LONGEST RUN AND NOT THE FIRST ONE, which is the rule this replaced
    on 2026-09-02 and the second half of the same defect. `above` is a noisy
    reading of a prefix property, and when it is NOT a clean prefix the first
    run is whatever the lowest treads happened to do. Planted directly:
    `[1,0,1,1,1,1,1,1]` returned `start=0, k=1` and threw six memory-bound
    treads away on the strength of one, which is the single-tread veto arriving
    from the other side; `[1,1,0,1,1,1,1,1]` returned `k=2` off the short run
    and ignored the run of five. Under the 3x-noise margin `analyse` uses, one
    tread sitting just outside it at low n is exactly how those shapes arise.
    The longest run is the reading that does not hand the answer to the least
    trustworthy point, in either direction; the tie goes low because the low
    treads are where a memory branch is if there is one.
    """
    above = [y > overhead + c_ref * x * (1.0 + margin)
             for x, y in zip(xs, ys, strict=True)]
    best_start, best_count = 0, 0
    run_start = None
    for i, flag in enumerate([*above, False]):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            # STRICTLY greater, so the earliest run wins a tie: `>=` would walk
            # the answer up to the last equal-length run, which is the treads
            # furthest from where a memory branch lives.
            if i - run_start > best_count:
                best_start, best_count = run_start, i - run_start
            run_start = None
    return best_start, best_count, above


def fit_ladder(points, block_m: int, ref: ComputeReference | None = None,
               margin: float = MEMORY_BRANCH_MARGIN,
               excluded_drifted: int = 0,
               kept_high_clock: int = 0,
               kept_low_clock: int = 0,
               *,
               model: str | MoEConfig = "",
               dtype: str = "",
               bandwidth_gbps: float | None = None,
               bandwidth_source: str = "") -> LadderFit:
    """Split the ladder into a memory branch and a compute branch.

    Membership comes from the reference compute branch when there is one, by the
    majority-and-contiguity rule `memory_branch_members` documents: the memory
    branch is the run of treads standing more than `margin` above `C n`,
    starting wherever the lowest such tread is rather than at n=1.

    Without a reference (`ref` absent or unusable) it falls back to searching
    every split for the smallest residual, which is what a reader would do by
    eye and carries the failure mode `LadderFit`'s docstring describes. The
    `basis` field says which happened, because the two answers are not
    interchangeable and one of them can invent an alpha.

    THE MEMORY BRANCH IS FITTED ON RAW TIMES, fixed cost included, and that is
    deliberate. Subtracting an extrapolated fixed cost before fitting hands its
    error straight to alpha: a 4-tread reference ladder under 1% timing spread
    put the extrapolated cost anywhere from 0.03 to 0.16 ms, which moved alpha
    from 0.56 to 0.70 on data planted at 0.558. Leaving it in makes
    `LadderFit.alpha` a LOWER BOUND with a known sign -- the fixed cost inflates
    the denominator and nothing else -- and every gate is scored on the bound.
    `alpha_upper` carries the other end for a reader who wants the range.

    SIGNATURE EXTENDED, never narrowed. `excluded_drifted` is optional and
    defaults to 0, so the four sibling scripts that call
    `SWEEP.fit_ladder(points, bm, ref[, margin])` are unaffected; pass it and
    the fit can tell a ladder that never had treads from one whose treads were
    dropped for a drifting clock, which is the difference between
    `NOT_IDENTIFIED_TOO_FEW` and `UNDECIDED_DRIFTING_CLOCK`. `kept_high_clock`
    and `kept_low_clock` are the same kind of count for the two sides of LEVEL,
    carried onto the fit so the ladder row can say how many of its treads sat
    off the band and in which direction; neither changes an outcome, because a
    steady clock off the band is a tread.

    `model`, `dtype`, `bandwidth_gbps` and `bandwidth_source` are KEYWORD-ONLY
    and all four default to absent. Together they are the denominator of
    `LadderFit.weight_streams`, the slope in units of one complete stream of
    the layer's ROUTED expert weight set. Absent, that property is None, which
    is the right answer for a caller that did not say which model's weights or
    which measured rate: w scales 1:1 in the bandwidth, so a w against a
    guessed card is a number with no meaning. None of the four touches the fit,
    the branch membership or any outcome.

    `model` TAKES THE CONFIG YOU ALREADY HOLD, or its name. It was called
    `model_name` and typed `str` until 2026-09-10; a caller that has a
    `MoEConfig` in scope should pass the object, so that the w and the rest of
    its report divide by one geometry rather than by whatever `MODEL_CONFIGS`
    returns for a string beside it.

    THAT RENAME IS THE ONE NARROWING THIS SIGNATURE HAS TAKEN, and it is called
    out because the paragraph above promises the opposite. `model_name=` is no
    longer accepted. Nothing outside this file passed it: the four sibling arms
    build their fits positionally as `SWEEP.fit_ladder(points, bm, ref[,
    margin])` and get no w at all, which is the open item this rename is meant
    to make easy for them to close.
    """
    overhead = ref.overhead_ms if ref else 0.0
    pts = [(n, ms) for n, ms in points if ms > 0]
    if not pts:
        return LadderFit(block_m, tuple(points), 0, None, None, None, None,
                         math.inf, overhead, "no usable treads",
                         outcome=NOT_IDENTIFIED_NO_TREADS,
                         outcome_reason=(
                             "no exactly-full tile stack at this block size "
                             "carried a usable time"
                             + (f"; {excluded_drifted} cell(s) were excluded "
                                "for a drifting clock" if excluded_drifted
                                else "")),
                         excluded_drifted=excluded_drifted,
                         kept_high_clock=kept_high_clock,
                         kept_low_clock=kept_low_clock,
                         model=model, dtype=dtype,
                         bandwidth_gbps=bandwidth_gbps,
                         bandwidth_source=bandwidth_source)
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    c_ref = ref.slope_for(block_m) if ref else None
    start = 0

    if ref is not None and ref.block_m == block_m:
        # The reference ladder has NO memory branch, by the assumption that
        # qualified it as the reference. Letting it test its own points against
        # its own fitted line lets timing noise push the low treads above it:
        # at 1% spread BLOCK_M=256 reported two memory-bound treads and an alpha
        # of 0.96, which then won the "largest identifiable block size" contest
        # and turned gate 3 into a PASS on data planted at 0.10.
        k = 0
        basis = "the reference ladder itself: compute bound at every tread"
    elif c_ref:
        start, k, _above = memory_branch_members(xs, ys, c_ref, overhead, margin)
        basis = (f"membership from the compute branch scaled off "
                 f"BLOCK_M={ref.block_m}"
                 + (f"; the branch starts at tread n={int(xs[start])}, not at "
                    "the first tread, which sits inside the margin"
                    if start else ""))
    else:
        # THE TWO WAYS TO HAVE NO REFERENCE ARE NOT THE SAME STATE and printing
        # them the same way is what hid the BN=256 corruption: 8 cells read as
        # "not identifiable" under a caption blaming tread count, when the
        # actual cause was a compute branch 44x too steep. Say which happened.
        k = _best_split(xs, ys, overhead)
        basis = ("split search: the compute reference was REFUSED at "
                 f"BLOCK_M={ref.refused_block_m} on its level, so this ladder "
                 "is unidentifiable BECAUSE OF THE REFERENCE and not for want "
                 "of treads"
                 if ref is not None and ref.refused
                 else "split search: no usable compute reference")

    stop = start + k
    a = b = None
    if k >= 2:
        a, b = _line(xs[start:stop], ys[start:stop])
    c_own = (_through_origin(xs[stop:], [y - overhead for y in ys[stop:]])
             if stop < len(pts) else None)
    c_eff = c_own if c_own else c_ref
    outcome, reason = "", ""
    if b is not None and c_eff and abs(b / c_eff - 1.0) <= PARALLEL_BRANCH_TOLERANCE:
        # A memory branch parallel to the compute branch is not a second
        # mechanism. `B / C = ridge / ai_cap`, so this says the ceiling sits on
        # the ridge -- or, far more often, that the fit ran the prefix into the
        # compute branch and is about to report that branch's slope as alpha.
        #
        # THE OUTCOME IS UNDECIDED AND IS NAMED. Until 2026-09-02 this branch
        # produced `alpha=None, crosses=None`, which downstream read as "the
        # sweep lacked treads" and let gate 3 IMPORT an alpha from another block
        # size over the top of it. Those are three different statements and only
        # one of them is true here: the ladder was measured, the branch was
        # fitted, and its slope sits on the ridge to within the tolerance. That
        # question is answerable, just not by this sweep.
        gap = abs(b / c_eff - 1.0)
        a, b = None, None
        k = 0
        outcome = UNDECIDED_PARALLEL_BRANCH
        reason = (
            f"UNDECIDED at BLOCK_M={block_m}: the tile's cap is within "
            f"{gap:.1%} of the ridge (the fitted memory slope B and the compute "
            f"slope C differ by that much, and B/C = ridge/ai_cap exactly), "
            f"inside the {PARALLEL_BRANCH_TOLERANCE:.0%} tolerance. This sweep "
            "cannot decide whether this tile crosses; a roofline arm at this "
            "tile can, because it measures the roof fraction directly instead "
            "of separating two slopes that are the same line. This is NOT a "
            "shortage of treads and NOT a null, and no alpha may be imported "
            "over it.")
        basis += (f"; memory branch DISCARDED, its slope was within "
                  f"{PARALLEL_BRANCH_TOLERANCE:.0%} of the compute branch and "
                  "the two are then the same line")
    err = _max_affine_error(xs, ys, a, b, c_eff, overhead)
    made = LadderFit(block_m, tuple(points), k, a, b, c_own, c_ref, err,
                     overhead, basis, outcome=outcome, outcome_reason=reason,
                     excluded_drifted=excluded_drifted,
                     kept_high_clock=kept_high_clock,
                     kept_low_clock=kept_low_clock,
                     branch_start=start if k else 0,
                     model=model, dtype=dtype,
                     bandwidth_gbps=bandwidth_gbps,
                     bandwidth_source=bandwidth_source)
    if outcome:
        return made
    outcome, reason = _ladder_outcome(made, ref, excluded_drifted)
    return LadderFit(block_m, tuple(points), k, a, b, c_own, c_ref, err,
                     overhead, basis, outcome=outcome, outcome_reason=reason,
                     excluded_drifted=excluded_drifted,
                     kept_high_clock=kept_high_clock,
                     kept_low_clock=kept_low_clock,
                     branch_start=start if k else 0,
                     model=model, dtype=dtype,
                     bandwidth_gbps=bandwidth_gbps,
                     bandwidth_source=bandwidth_source)


def _ladder_outcome(fit: LadderFit, ref, excluded: int) -> tuple[str, str]:
    """Name what this ladder concluded, in the vocabulary of `LADDER_OUTCOMES`.

    Order matters. The clock exclusion is tested BEFORE the tread count,
    because a ladder that lost treads to a drifting governor and a ladder that
    never had them report the same count and mean opposite things: the first
    says the clock moved mid-cell and the arm should be re-timed on a settling
    instrument, the second says the tile really is compute bound this early.
    Reporting them as one number is how an instrument fault becomes a physical
    finding.
    """
    if fit.memory_points >= MIN_MEMORY_TREADS and fit.alpha is not None:
        return IDENTIFIED, ""
    if excluded and fit.memory_points < MIN_MEMORY_TREADS:
        return UNDECIDED_DRIFTING_CLOCK, (
            f"UNDECIDED at BLOCK_M={fit.block_m}: {excluded} cell(s) were "
            "excluded because their SM clock MOVED while they were timed, so "
            "each median is a blend of two operating points, leaving "
            f"{fit.memory_points} memory-bound tread(s) against the "
            f"{MIN_MEMORY_TREADS} a verdict needs. What this ladder measured is "
            "the governor settling, not the tile. Re-time the arm on an "
            "instrument that warms until the clock settles; do not read the "
            "shortfall as a property of the tiling.")
    if ref is not None and ref.block_m == fit.block_m:
        return NOT_IDENTIFIED_IS_REFERENCE, (
            f"BLOCK_M={fit.block_m} is the compute reference, so by the "
            "assumption that qualified it there is no memory branch here to fit")
    return NOT_IDENTIFIED_TOO_FEW, (
        f"BLOCK_M={fit.block_m} has {fit.memory_points} memory-bound tread(s), "
        f"under the {MIN_MEMORY_TREADS} a verdict needs")


def _best_split(xs, ys, overhead: float = 0.0) -> int:
    """Fallback membership: the split with the smallest max-affine residual."""
    best = None
    for k in range(0, len(xs) + 1):
        a = b = None
        if k >= 2:
            a, b = _line(xs[:k], ys[:k])
        c = (_through_origin(xs[k:], [y - overhead for y in ys[k:]])
             if k < len(xs) else None)
        if b is None and c is None:
            continue
        score = _max_affine_error(xs, ys, a, b, c, overhead)
        if best is None or score < best[0]:
            best = (score, k)
    return best[1] if best else 0


def _max_affine_error(xs, ys, a, b, c, overhead: float = 0.0) -> float:
    """Mean relative error of `max(a + b n, D + c n)` against the ladder.

    The fixed cost is added to the COMPUTE branch only. The memory branch is
    fitted on raw times and already carries it inside its intercept, which is
    what makes `LadderFit.alpha` a lower bound rather than a noisy point.
    """
    errs = []
    for x, y in zip(xs, ys, strict=True):
        branches = []
        if b is not None:
            branches.append(a + b * x)
        if c:
            branches.append(overhead + c * x)
        if not branches or y <= 0:
            return math.inf
        errs.append(abs(max(branches) - y) / y)
    return statistics.fmean(errs) if errs else math.inf


def activation_slope_ms(cfg, block_m: int, bandwidth_gbps: float) -> float:
    """The part of the per-tile slope that is activations, not weight re-reads.

    An extra tile carries `BM` more rows, and those rows move
    `activation_bytes_per_row` of x_perm / h_up / h_act / y_perm each. That is
    affine in `n` exactly like the re-read term, so it lands inside `B` and
    inflates the fitted alpha. Subtracting it is what the report's
    `alpha-corrected` column is, and gate 3 is scored on that column.
    """
    per_tile = cfg.num_experts * block_m * activation_bytes_per_row(cfg)
    return 1e3 * per_tile / (bandwidth_gbps * 1e9)


def cap_overstatement(cfg, block_m: int, block_n: int, b: int
                      ) -> tuple[float, float]:
    """`(lo, hi)` on `ai_model.lin_overstatement` for this tile. R8's label.

    THE FACTOR EVERY CAP IN THIS REPORT IS HIGH BY. `ai_cap` computes
    `2 BM / (alpha b)` from a FITTED alpha, and `moe/bench/ai_model.py` shows
    that a B/(A+B) fit returns `(alpha_b + phi) / (1 + phi + delta)`, so the cap
    that division gives is the exact cap times `1 + phi + delta`. At BM=128,
    BN=64 on mixtral at delta = 0 the bracket is 1.04 to 3.03, a 3.6% to 203%
    overstatement, and even its low end is of the order of the cap-to-ridge
    gap the cap is being used to decide. The "about 1.32" this docstring once
    quoted was the point at alpha_a = 0.143, the withdrawn (LIN)-solved pair;
    it sits inside the bracket and is not a measurement.

    IT IS A BRACKET AND NOT A NUMBER, and the reason is the honest one:
    `alpha_a`, the miss fraction on the ACTIVATION re-read, has no measurement
    anywhere in this repository. `phi` depends on it, so this returns the factor
    at alpha_a = 0 and at alpha_a = 1 -- no re-read and a full one -- and the
    report prints both ends. `delta`, the fused layer's fixed cost, is taken as
    zero here, which makes both ends LOWER bounds on the overstatement; the
    report's own `overhead_ms` is the measured stand-in for delta and is printed
    beside them.

    ONE DEFINITION. This delegates to `ai_model.overstatement_bracket`, which
    owns the alpha_a-in-{0, 1} loop, rather than carrying a second copy of
    it: two copies of one bracket are how a point re-enters at one of them
    (this file printed the withdrawn 1.32 point for two commits after the
    module it cites had replaced it with the bracket).

    The single-GEMM shape this is evaluated on is the up-projection,
    `N = 2 F` and `K = H`, because that is the GEMM whose B operand is the
    weight slab whose re-read the whole study is about.
    """
    n, k = 2 * cfg.intermediate_size, cfg.hidden_size
    return ai_model.overstatement_bracket(n, k, block_m=block_m,
                                          block_n=block_n, b=b, delta=0.0)


def cap_note(cfg, block_m: int, block_n: int, b: int) -> str:
    """One labelled clause naming the overstatement beside a cap."""
    lo, hi = cap_overstatement(cfg, block_m, block_n, b)
    return (f"cap is 2*BM/(alpha*b) from a FITTED alpha and is HIGH by "
            f"ai_model.lin_overstatement = 1+phi+delta, {lo:.2f}-{hi:.2f}x here "
            f"(alpha_a unmeasured, delta taken as 0, so a lower bound)")


@dataclass(frozen=True)
class AlphaInterval:
    """What a fitted alpha is worth, as an interval, so it can be tested BOTH ways.

    A one-sided gate needs only a point estimate. A two-sided one needs to know
    how wide the estimate is, and this carries the two independent widths that
    apply, kept separate because they answer different objections:

      * RANDOM. `boot_lo`/`boot_hi` are a percentile interval from resampling
        the memory branch's own treads with replacement and refitting. It is
        zero-width on noiseless synthetic cells, which is correct: nothing about
        those treads is random.
      * SYSTEMATIC. `sys_lo`/`sys_hi` are the two ends the report has always
        printed. `alpha-corrected` subtracts the activation traffic that inflates
        the slope; `alpha-hi` subtracts the fused layer's fixed cost that
        inflates the level. `LadderFit.alpha` sits between them BY CONSTRUCTION
        and the two corrections point in opposite directions, so the pair is a
        bracket rather than an error bar.

    `lo` and `hi` are the union, and the union is the honest object to score a
    pre-registered band against: an interval that carried only the random part
    would be a two-sided test resting on the assumption that both named biases
    are zero, which is the assumption the two corrections exist because it is
    false. On mixtral at BN=64 the systematic bracket alone is about 0.05 wide
    at alpha 0.55, which is the size of `ALPHA_BAND` itself.
    """

    point: float
    lo: float
    hi: float
    sys_lo: float
    sys_hi: float
    boot_lo: float | None
    boot_hi: float | None
    treads: int
    trials: int

    def overlaps(self, band: tuple[float, float]) -> bool:
        return self.lo <= max(band) and min(band) <= self.hi

    def direction(self, band: tuple[float, float]) -> str:
        """"ABOVE", "BELOW" or "" for an interval that overlaps the band."""
        if self.overlaps(band):
            return ""
        return "ABOVE" if self.lo > max(band) else "BELOW"

    def render(self) -> str:
        boot = ("no bootstrap: fewer than "
                f"{MIN_BOOTSTRAP_TREADS} treads to resample"
                if self.boot_lo is None
                else f"bootstrap [{self.boot_lo:.3f}, {self.boot_hi:.3f}] over "
                     f"{self.trials} resamples of {self.treads} treads")
        return (f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]  "
                f"(systematic [{self.sys_lo:.3f}, {self.sys_hi:.3f}]: "
                f"activation-corrected to fixed-cost-corrected; {boot})")


def alpha_interval(fit: LadderFit, cfg, bandwidth_gbps: float, *,
                   trials: int = ALPHA_BOOTSTRAP_TRIALS,
                   coverage: float = ALPHA_BOOTSTRAP_COVERAGE,
                   seed: int = 0) -> AlphaInterval | None:
    """The fitted alpha with both its widths, or None when it has no alpha.

    The bootstrap resamples the memory branch's `(tiles, ms)` treads WITH
    REPLACEMENT and refits the line on each draw, which is the resampling that
    matches how the estimate was made: the treads are the observations and the
    line is the statistic. Draws that land on fewer than two distinct treads are
    discarded rather than fitted, because `_line` returns a zero slope on a
    degenerate x and a zero slope is an alpha of zero, which would drag the low
    end of the interval to somewhere no ladder ever suggested.

    The point estimate is the ACTIVATION-CORRECTED alpha, the same column gate 3
    has always scored, so the interval is centred on the number the report
    prints rather than on a second quantity computed only here.
    """
    if fit.alpha is None or fit.slope_memory is None or not fit.load_ms:
        return None
    act = activation_slope_ms(cfg, fit.block_m, bandwidth_gbps)
    point = (fit.slope_memory - act) / fit.load_ms
    upper = fit.alpha_upper
    sys_lo, sys_hi = (point, point if upper is None else max(point, upper))

    treads = [(n, ms) for n, ms in fit.points if ms > 0]
    # The branch is the run the fit used, and `branch_start` says where it
    # began: the run may start above the lowest tread, so slicing from zero
    # would resample points the fit deliberately left off the branch.
    branch = (treads[fit.branch_start:fit.branch_start + fit.memory_points]
              if fit.memory_points else [])
    boot_lo = boot_hi = None
    if len(branch) >= MIN_BOOTSTRAP_TREADS:
        rng = random.Random(seed)
        draws: list[float] = []
        for _ in range(trials):
            sample = [branch[rng.randrange(len(branch))]
                      for _ in range(len(branch))]
            xs = [float(n) for n, _ in sample]
            if len(set(xs)) < 2:
                continue
            a, slope = _line(xs, [ms for _, ms in sample])
            level = a + slope
            if level > 0:
                draws.append((slope - act) / level)
        if len(draws) >= max(20, trials // 10):
            draws.sort()
            tail = (1.0 - coverage) / 2.0
            boot_lo = draws[int(tail * (len(draws) - 1))]
            boot_hi = draws[int((1.0 - tail) * (len(draws) - 1))]
    lo = min([sys_lo] + ([boot_lo] if boot_lo is not None else []))
    hi = max([sys_hi] + ([boot_hi] if boot_hi is not None else []))
    return AlphaInterval(point=point, lo=lo, hi=hi, sys_lo=sys_lo,
                         sys_hi=sys_hi, boot_lo=boot_lo, boot_hi=boot_hi,
                         treads=len(branch), trials=trials)


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

PASS, FAIL, UNDECIDED = "PASS", "FAIL", "UNDECIDED"

#: A gate about the INSTRUMENT: whether the run is readable at all. Gate 0 is
#: the only one, and a non-PASS there voids every claim gate below it.
VALIDITY = "VALIDITY"
#: A gate about the WORLD: it can be FAILed by hardware without the run being
#: broken, and a FAIL is a result rather than an error.
CLAIM = "CLAIM"

#: Where the number in `measured` came from. The distinction exists because
#: gate 3 printed a "measured" crossing ratio for 22 published reports that was
#: an algebraic restatement of a fitted alpha imported from a different
#: BLOCK_M, and the serialized JSON kept only {number, claim, verdict, measured,
#: gate} -- so nothing in the published file said the ratio was never observed.
OBSERVED = "OBSERVED"       # read off a timing in this run
DERIVED = "DERIVED"         # computed from a fit over this run's own timings
IMPORTED = "IMPORTED"       # computed from a fit over a DIFFERENT setting


#: The one-token name each gate answers to on its `RESULT:` line. A name is one
#: run of non-whitespace by `moe.bench.exit_codes.result_line`'s own rule, and
#: these are the words a driver greps for, so they are fixed here rather than
#: derived from the claim text.
GATE_NAMES = {0: "override_took_effect", 1: "tile_steps", 2: "time_falls_with_block_m",
              3: "alpha_in_band", 4: "no_crossing_at_the_null_tile"}


@dataclass
class Gate:
    number: int
    claim: str
    verdict: str
    measured: str
    threshold: str
    lines: list[str] = field(default_factory=list)
    #: VALIDITY or CLAIM. Defaults to CLAIM because four of the five are.
    kind: str = CLAIM
    #: OBSERVED / DERIVED / IMPORTED, for the number in `measured`.
    basis: str = OBSERVED
    #: Machine-readable provenance, serialized beside the verdict. Anything a
    #: reader would need to tell an observation from a restatement goes here.
    #:
    #: The reason given here used to be that the printed detail lines do not
    #: survive into report.json. They have since 2026-09-02 (5a49285): `payload`
    #: serializes `"detail": list(g.lines)` on every gate, and says so in a
    #: comment beside it. The description outlived the behaviour, which is this
    #: repository's recurring defect, and it is load-bearing in both directions
    #: in this one file now that a test reads a w out of those detail lines.
    #: The standing reason is the other half: `lines` is PROSE, written for a
    #: reader, and a driver that had to regex a float out of a sentence would
    #: break the first time the sentence was reworded. This dict is the shape a
    #: program reads.
    provenance: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return GATE_NAMES.get(self.number, f"gate{self.number}")

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        `moe.bench.exit_codes.result_line` renders it and `parse_result_lines`
        reads it back, anchored at column zero on the `RESULT: ` prefix. The
        `GATE n VERDICT` line below is for a human and for
        `scripts/pod_session.sh`, whose `gate_from_log` has matched that exact
        shape since 2026-09-01; both are kept because they have different
        readers, and only this one is the machine contract. Nothing else in this
        file's output starts with `RESULT: `.
        """
        detail = f"[{self.kind}/{self.basis}] {self.claim} | measured " \
                 f"{self.measured} | gate {self.threshold}"
        return exit_codes.result_line(
            *self.scored(), " ".join(detail.split()))

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`'s vocabulary.

        This file has said UNDECIDED since it was written and the shared table
        says UNKNOWN; they are the same state and the table's spelling wins at
        the boundary, because `classify` refuses a verdict it does not know
        rather than letting it fall through a comparison and be scored as
        whatever the fallthrough happened to be. Both names mean "this gate did
        not decide", and both count AGAINST the gate.
        """
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.name,
                exit_codes.UNKNOWN if self.verdict == UNDECIDED else self.verdict)

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"GATE {self.number}  {self.verdict:9s} [{self.kind}/"
               f"{self.basis}] {self.claim}",
               f"          measured {self.measured}   gate {self.threshold}"]
        out += [f"          {line}" for line in self.lines]
        return out


def gate_0_override(compiles: dict[int, int], executed: dict[int, int],
                    block_sizes) -> Gate:
    """Did `override_config` actually change the kernel.

    THE GATE THAT DECIDES WHETHER THE OTHER FOUR MEAN ANYTHING. If the override
    silently failed, all four settings ran one kernel, every difference is zero,
    and the report reads as a tidy null result. A setting that changed the tile
    constants MUST have compiled a new Triton specialisation, so counting the
    artefacts that appear while a setting runs is a direct assay. Zero means
    either the override did nothing or the cache served a previous run, and both
    are fatal in the same way.

    A RESUMED SETTING IS NOT A FAILED ONE. A run that finds every cell already
    in `cells.csv` executes nothing and therefore compiles nothing, which is not
    evidence that the override is broken -- it is the absence of the experiment
    that would have tested it. Those settings are named and the gate goes
    UNDECIDED, because the assay belongs to the session that ran the cells and
    this session cannot inherit it.
    """
    ran = [bm for bm in block_sizes if executed.get(bm, 0) > 0]
    resumed = [bm for bm in block_sizes if executed.get(bm, 0) <= 0]
    missing = [bm for bm in ran if compiles.get(bm, 0) <= 0]
    counts = ", ".join(f"BM={bm}:{compiles.get(bm, 0)}" for bm in block_sizes)
    if missing:
        return Gate(
            0, "override_config changed the kernel at every setting", FAIL,
            f"fresh Triton artefacts per setting: {counts}", ">= 1 per setting",
            [f"BLOCK_SIZE_M {missing} ran cells and compiled nothing new.",
             "Either override_config did not take effect, or TRITON_CACHE_DIR "
             "was warm.",
             "Every gate below is then a comparison of one kernel with itself. "
             "Do not read them."], kind=VALIDITY)
    if resumed:
        return Gate(
            0, "override_config changed the kernel at every setting", UNDECIDED,
            f"fresh Triton artefacts per setting: {counts}", ">= 1 per setting",
            [f"BLOCK_SIZE_M {resumed} ran no cells this session: every one was "
             "already in cells.csv.",
             "The compile assay belongs to the session that measured them and "
             "cannot be inherited. Delete cells.csv and re-run to assay it "
             "again, or read the gates knowing this one was not repeated."],
            kind=VALIDITY)
    return Gate(0, "override_config changed the kernel at every setting", PASS,
                f"fresh Triton artefacts per setting: {counts}",
                ">= 1 per setting", [], kind=VALIDITY)


def gate_1_steps(cells, cfg, preds, *, alpha: float, ridge: float,
                 bandwidth_gbps: float, b: int, noise: float) -> Gate:
    """Do the time steps land at `T = n BLOCK_M E / k`.

    Read as a jump against a tread, not as a slope. A slope over the 4-token
    interval that separates a full tile stack from the next tread would divide a
    0.5% timing difference by `log(1.004)` and report 1.2, so the log-log
    detector `crossing.py` uses is the wrong instrument at this resolution. The
    quantity here is the RATIO across the boundary against the ratio just below
    it, both over intervals of the same width, which needs no logarithm.
    """
    rows = {}
    for c in cells:
        if c.status == "ok" and c.ms_p50 > 0:
            rows.setdefault(c.block_m, {})[c.rows_per_expert] = c
    detail = []
    jumps, treads, predicted = [], [], []
    misplaced = 0
    for bm in sorted(rows):
        grid = sorted(rows[bm])
        for n in range(1, 64):
            edge = float(n * bm)
            if edge > max(grid):
                break
            at = rows[bm].get(edge)
            if at is None:
                continue
            below = max((r for r in grid if (n - 1) * bm < r < edge), default=None)
            above = min((r for r in grid if r > edge), default=None)
            if below is None or above is None:
                continue
            jump = rows[bm][above].ms_p50 / at.ms_p50 - 1.0
            tread = at.ms_p50 / rows[bm][below].ms_p50 - 1.0
            pj = (model_ms(cfg, above, bm, alpha=alpha, ridge=ridge,
                           bandwidth_gbps=bandwidth_gbps, b=b)
                  / model_ms(cfg, edge, bm, alpha=alpha, ridge=ridge,
                             bandwidth_gbps=bandwidth_gbps, b=b) - 1.0)
            jumps.append(jump)
            treads.append(tread)
            predicted.append(pj)
            if jump <= tread:
                misplaced += 1
            detail.append(
                f"BM={bm:3d} n={n:2d} T={tokens_for_rows(cfg, int(edge)):6d} "
                f"r={edge:6.0f}  tread {tread:+7.2%}  STEP {jump:+7.2%}  "
                f"(model {pj:+7.2%})  waves {at.waves_up:6.1f}/"
                f"{rows[bm][above].waves_up:6.1f}")
    if len(jumps) < 3:
        return Gate(1, "time steps at T = n x BLOCK_M x E/k", UNDECIDED,
                    f"{len(jumps)} bracketed boundaries", ">= 3",
                    ["No boundary had a point below, at and above it. Raise "
                     "--step-probes or lower --row-step."])
    measured = statistics.median(j - t for j, t in zip(jumps, treads, strict=True))
    gate = 0.5 * statistics.median(predicted)
    verdict = PASS if (measured >= gate and measured > 3 * noise) else FAIL
    return Gate(
        1, "time steps at T = n x BLOCK_M x E/k", verdict,
        f"median step minus tread {measured:+.2%}",
        f">= {gate:+.2%} (half the model's step) and > 3x noise ({3 * noise:.2%})",
        [f"{len(jumps)} boundaries bracketed, {misplaced} where the tread moved "
         f"at least as much as the step",
         f"median step {statistics.median(jumps):+.2%}, median tread "
         f"{statistics.median(treads):+.2%}, model step "
         f"{statistics.median(predicted):+.2%}",
         "waves are printed on both sides of every step so occupancy can be "
         "ruled out as the cause:"] + detail)


def gate_2_direction(cells, cfg, *, alpha: float, retracted: float, ridge: float,
                     bandwidth_gbps: float, b: int, block_sizes) -> Gate:
    """Does time move UP or DOWN with bigger BLOCK_M in the MULTI-TILE regime.

    Compared only at rows-per-expert that are an exact multiple of EVERY block
    size, where all four run zero-padding tile stacks. Anywhere else the
    comparison carries a padding difference of up to 8x and stops being about
    traffic.

    Both worlds predict a direction and they predict different magnitudes: at
    alpha=0.558 the smallest tile pays 18 weight reads where the largest pays a
    compute-bound 6.4, and at alpha=0.10 the smallest tile is ALSO compute bound
    by then and the ratio is 1.0. So this gate is not only a direction, it is a
    second, independent test of the same disagreement.
    """
    lo, hi = min(block_sizes), max(block_sizes)
    step = math.lcm(*block_sizes)
    by = {}
    for c in cells:
        if c.status == "ok" and c.aligned and c.ms_p50 > 0:
            by.setdefault(c.rows_per_expert, {})[c.block_m] = c
    common = sorted(r for r, d in by.items()
                    if r % step == 0 and all(bm in d for bm in block_sizes))
    if not common:
        return Gate(2, "time falls with BLOCK_M at equal rows per expert",
                    UNDECIDED, "no rows-per-expert common to every block size",
                    f"a multiple of {step}",
                    ["Raise --r-max to at least one multiple of the largest "
                     "block size."])
    detail = ["rows/expert  " + "  ".join(f"BM={bm:<9d}" for bm in block_sizes)
              + "   ms(lo)/ms(hi)"]
    for r in common:
        row = "  ".join(f"{by[r][bm].ms_p50:9.4f} ms" for bm in block_sizes)
        detail.append(f"{r:11.0f}  {row}   "
                      f"{by[r][lo].ms_p50 / by[r][hi].ms_p50:.3f}x")
    top = common[-1]
    ratio = by[top][lo].ms_p50 / by[top][hi].ms_p50
    def predicted(a):
        return (model_ms(cfg, top, lo, alpha=a, ridge=ridge,
                         bandwidth_gbps=bandwidth_gbps, b=b)
                / model_ms(cfg, top, hi, alpha=a, ridge=ridge,
                           bandwidth_gbps=bandwidth_gbps, b=b))
    direction = ("DOWN with BLOCK_M, as traffic requires" if ratio > 1.01 else
                 "UP with BLOCK_M, as occupancy would" if ratio < 0.99 else
                 "not at all with BLOCK_M, which is what a block size already "
                 "compute bound at the smallest tile looks like")
    return Gate(
        2, "time falls with BLOCK_M at equal rows per expert (traffic, not occupancy)",
        PASS if ratio >= GATE2_RATIO else FAIL,
        f"ms(BM={lo}) / ms(BM={hi}) = {ratio:.3f}x at {top:.0f} rows per expert",
        f">= {GATE2_RATIO:.2f}x",
        [f"time moves {direction}",
         f"model at alpha={alpha:.3f}: {predicted(alpha):.3f}x   "
         f"model at the retracted alpha={retracted:.2f}: {predicted(retracted):.3f}x",
         f"a refutation here is the interesting answer: it says the padded "
         f"arithmetic or the lost occupancy outweighs "
         f"{math.ceil(top / lo)} weight re-reads"]
        + detail)


def no_crossing_reason(pred: TilePrediction) -> str:
    """Why this tile never crosses, when it never crosses. Empty when it does.

    "NEVER CROSSES AT THIS ALPHA" IS AN OUTCOME, NOT A MISSING VALUE, and until
    2026-09-02 this file did not have a sentence for it. `predict_tile` returned
    `crossing_rows=None` exactly when `ai_cap <= ridge`, which is the whole
    claim the study is making, and the report then formatted that None with
    `:.0f`. `--self-test 0.90` and `--self-test 1.0` -- the two worlds
    mixtral's G=1 ladders (0.95-1.02 on both cards; qwen2 and deepseek read
    0.62-0.84) actually describe -- died with a TypeError before
    report.json was written, so the analysis crashed in precisely the world the
    data pointed at.
    """
    if pred.crossing_rows is not None:
        return ""
    return (f"BLOCK_M={pred.block_m} never crosses at alpha={pred.alpha:.3f}: "
            f"its AI cap 2*BM/(alpha*b) = {pred.ai_cap:.1f} Op/B sits at or "
            f"below the ridge {pred.ridge:.1f} Op/B, so no batch size makes it "
            "compute bound. This is the study's own claim arriving, not a "
            "missing number")


def gate_3_alpha_discriminates(fits, preds_lo, preds_hi, cfg, *, lo: int,
                               hi: int, alpha_source: str,
                               alpha_hat: float | None,
                               alpha_source_bm: int | None,
                               ridge_band: tuple[float, float],
                               interval: AlphaInterval | None = None,
                               band: tuple[float, float] = ALPHA_BAND) -> Gate:
    """Does the fitted alpha's interval overlap the pre-registered ALPHA_BAND.

    WHAT THIS GATE USED TO CLAIM, AND WHY THAT WAS WITHDRAWN. It was phrased as
    "the BLOCK_M=128 crossing sits Q above the BLOCK_M=256 one" and its
    `measured` field was `1.0 + alpha_hat`. That is not a measurement of a
    crossing ratio; it is the model's identity `R_cross(lo)/R_cross(hi) = 1 +
    alpha` evaluated at a fitted alpha, and the alpha is usually IMPORTED from a
    different BLOCK_M because tread 2 at 128 is compute bound under both worlds
    and so carries no information about re-read. Three things make the old
    phrasing indefensible rather than merely loose:

      * it is ALGEBRA, not evidence. Recomputed over every published report that
        carries both fields, `measured == 1 + alpha_measured` in 22 of 22 and
        differs in 0. The gate restated its own input.
      * the identity needs the 128 crossing to land in tread 2. At the A100's
        own alphas the model predicts NO CROSSING AT ALL for 3 of 6 arms, so
        for those the ratio does not exist and 1 + alpha stands for nothing.
      * NO CROSSING HAS EVER BEEN OBSERVED. Across all of `results/published`
        the ladder field `crosses` is False 41 times, null 61 times and True
        zero times. A gate must not be phrased as though one was seen.

    AND WHAT REPLACED "alpha > 0.33" ON 2026-09-02. The one-sided form was still
    a gate a pre-registered number could not fail from above. `alpha_hat > 0.33`
    passes for every value in 0.33..1.0, so all 41 committed surface fits (0.6
    to 1.0) pass it while NOT ONE of them is within 0.05 of the 0.558 those same
    reports predict; the one published sweep measured 0.989 and 0.923 against a
    prediction of 0.558, passed, and printed "it is where the refit put it". A
    threshold at the midpoint of two hypotheses tests which of the two you are
    nearer, not whether either is right, and this study's registered number is
    the band and not the midpoint.

    THE TEST IS NOW TWO-SIDED AND IS SCORED ON AN INTERVAL. `alpha_interval`
    carries the estimate's random width (a bootstrap over the memory branch's
    own treads) unioned with its systematic bracket (the activation correction
    on one side, the fixed-cost correction on the other -- the two ends this
    report has always printed and never scored). PASS is that interval
    OVERLAPPING `ALPHA_BAND`; FAIL is disjoint, and the verdict says which
    DIRECTION, because "measured 0.99 [0.96, 1.02] is ABOVE the band
    [0.53, 0.59]" and "measured 0.10 is BELOW it" are opposite findings and the
    retired gate rendered the first as a confirmation.

    0.33 SURVIVES AS AN INFORMATIONAL LINE, not as the verdict, so a reader can
    still recompute what the retired gate would have said from a new report.

    The crossing ratio the old gate printed is still computed and still
    reported, under `provenance["restated_crossing_ratio"]`, so an old verdict
    can be recomputed from a new report. It is labelled as a restatement.
    """
    pred_lo = crossing_ratio(preds_lo, lo, hi)
    pred_hi = crossing_ratio(preds_hi, lo, hi)
    retracted_ratio = crossing_ratio(
        predictions((lo, hi), RETRACTED_ALPHA, ridge_band[0]), lo, hi)
    band_lines = []
    for ridge_end, value in zip(ridge_band, (pred_lo, pred_hi), strict=True):
        band_lines.append(f"ridge {ridge_end:.1f}: {value:.3f}x" if value
                          else f"ridge {ridge_end:.1f}: no crossing")
    # An ABSENCE stated from the data rather than from the model, because the
    # gate's old wording implied a crossing had been watched. `LadderFit.crosses`
    # is `C > B` read off two FITTED slopes: whether this ladder's own numbers
    # say a crossing exists at all. It is the weakest form of the claim -- it
    # does not require the crossing to have been reached -- and even so it has
    # never once been true: across all of results/published it is False 41
    # times, null 61 times and True zero times.
    crossed = sorted(bm for bm, f in fits.items() if f.crosses)
    undecided_cross = sorted(bm for bm, f in fits.items() if f.crosses is None)
    observed = (
        f"ladders whose own fitted slopes imply a crossing exists (C > B): "
        f"{crossed}; slopes missing at {undecided_cross}" if crossed else
        "NO ladder's fitted slopes imply a crossing exists (C > B is met "
        f"nowhere; slopes missing at {undecided_cross}), so nothing here is a "
        "crossing measurement and no crossing has been observed")
    claim = (f"the fitted alpha's interval overlaps the pre-registered band "
             f"{band[0]}-{band[1]}")
    threshold = (f"interval overlaps [{band[0]}, {band[1]}] "
                 f"(the refit's 90% band; a FAIL names the direction)")
    one_sided = (None if alpha_hat is None
                 else alpha_hat > GATE3_ALPHA_DISCRIMINATOR)
    provenance = {
        "tests": "alpha interval overlaps ALPHA_BAND, both sides",
        "alpha_band": list(band),
        "alpha_interval": (None if interval is None else asdict(interval)),
        "retired_one_sided_threshold": GATE3_ALPHA_DISCRIMINATOR,
        "retired_one_sided_verdict": one_sided,
        "retired_one_sided_note": (
            "alpha_hat > 0.33, the midpoint of the refit and the retracted "
            "value. INFORMATIONAL. It is one-sided and passes for everything "
            "from 0.33 to 1.0, which is why it is no longer the verdict."),
        "not_an_observed_crossing": True,
        "observed_crossing_ratio": None,
        "ladders_whose_slopes_imply_a_crossing": crossed,
        "ladders_with_a_missing_slope": undecided_cross,
        "alpha_hat": alpha_hat,
        "alpha_source": alpha_source,
        "alpha_source_block_m": alpha_source_bm,
        "imported_from_another_block_m": (
            None if alpha_source_bm is None else alpha_source_bm != lo),
        "target_tile_outcome": (None if fits.get(lo) is None
                                else fits[lo].outcome),
        "target_tile_outcome_reason": (None if fits.get(lo) is None
                                       else fits[lo].outcome_reason),
        "restated_crossing_ratio": (
            None if alpha_hat is None else 1.0 + alpha_hat),
        "restated_crossing_ratio_note": (
            "1 + alpha_hat, the model's identity evaluated at the fitted alpha. "
            "It is what the retired ratio gate printed as 'measured'. It is not "
            "a measurement and it is only the crossing ratio at all when the "
            f"BLOCK_M={lo} crossing lands in tread 2."),
        "model_ratio_ridge_lo": pred_lo,
        "model_ratio_ridge_hi": pred_hi,
        "model_ratio_retracted": retracted_ratio,
        "ridge_band": list(ridge_band),
    }
    if alpha_hat is None or interval is None:
        return Gate(3, claim, UNDECIDED,
                    "alpha not identifiable at any block size", threshold,
                    ["No ladder had enough memory-bound treads, so no block "
                     "size measured the re-read fraction.",
                     "Lower --r-max is not the fix; a block size whose cap is "
                     "below the ridge is. 32 and 64 are those.",
                     observed,
                     "model's crossing ratio, for reference only: "
                     + "   ".join(band_lines)],
                    basis=DERIVED, provenance=provenance)

    where = interval.direction(band)
    own = fits.get(lo)
    imported = alpha_source_bm is not None and alpha_source_bm != lo
    # THE TILE'S OWN FIT CAN VETO THIS GATE, and until 2026-09-02 it could not.
    # When `fit_ladder` finds the BLOCK_M=128 memory branch running parallel to
    # the compute branch it returns UNDECIDED and says in as many words that
    # "no alpha may be imported over it" -- and this gate then imported one from
    # BLOCK_M=64 and returned PASS, two lines above printing that same sentence.
    # One report cannot both refuse to answer for a tile and answer for it. The
    # verdict is now UNDECIDED, which `exit_codes.classify` scores as a claim
    # gate that did not pass, and the roofline arm the fit names is the way to
    # settle it.
    #
    # IT BLOCKS THE PASS AND NOT THE FAIL, and the asymmetry is the argument.
    # A PASS here would say the pre-registered alpha holds AT BLOCK_M=128, on
    # the strength of a number fitted at another tiling, for a tile whose own
    # ladder just refused to answer: that is the direction the fit's reason
    # forbids. A FAIL says the alpha that WAS fitted lies outside the band, is
    # reported `[CLAIM/IMPORTED]` so a reader knows which tiling it belongs to,
    # and does not need the undecided tile to be true. Refusing to affirm and
    # allowing to refute is also what keeps `--self-test 0.85` a FAIL: a gate
    # that answered UNDECIDED to every planted world would discriminate
    # nothing.
    blocked = (imported and not where and own is not None
               and own.outcome == UNDECIDED_PARALLEL_BRANCH)
    verdict = FAIL if where else (UNDECIDED if blocked else PASS)
    # WHICH ALPHA WAS SCORED, in the verdict's own provenance line, because the
    # gate is about BLOCK_M=128 and the number is usually not from BLOCK_M=128.
    if blocked:
        provenance_line = (
            f"NOT SCORED. BLOCK_M={lo}'s own ladder came back "
            f"{own.outcome}, and the alpha above is BLOCK_M={alpha_source_bm}'s, "
            f"shown for reference only. {own.outcome_reason} An imported alpha "
            "cannot decide a tile whose own fit says this sweep cannot, so the "
            "verdict is UNDECIDED rather than the PASS the overlap would "
            "otherwise read.")
    elif own is not None and own.memory_points >= MIN_MEMORY_TREADS and not imported:
        crossing = preds_lo[lo].crossing_rows
        fragility = (
            f"Tread 2 there sits within a few percent of the compute branch -- "
            f"{2 * lo} padded rows against a {crossing:.0f} row crossing -- so "
            "it only just qualified as memory bound, and this alpha is the most "
            "fragile number in the report."
            if crossing is not None else
            f"At this alpha BLOCK_M={lo} has no crossing at all "
            f"(cap {preds_lo[lo].ai_cap:.1f} Op/B at or below the ridge "
            f"{preds_lo[lo].ridge:.1f}), so every tread on it is memory bound "
            "and the branch is as long as the sweep.")
        provenance_line = (
            f"SCORED ON BLOCK_M={lo}'s OWN alpha, over {own.memory_points} "
            f"memory-bound treads. {fragility} Compare it with the ladders "
            "below.")
    else:
        # THE REASON COMES FROM THE FIT, not from a tread count this gate
        # recomputes. A discarded branch leaves `memory_points` at 0, so the
        # old wording printed "0 tread(s) stand above the compute branch" for a
        # ladder whose eight treads all did -- a false count in the one line a
        # reader checks the import against.
        why = (f"there is no BLOCK_M={lo} ladder in this sweep" if own is None
               else own.outcome_reason or
               f"{own.memory_points} tread(s) stand above the compute branch, "
               f"and a verdict needs {MIN_MEMORY_TREADS}")
        provenance_line = (
            f"SCORED ON AN alpha IMPORTED from BLOCK_M={alpha_source_bm}: it is "
            f"not identifiable at BLOCK_M={lo} on this sweep ({why}). "
            "ALPHA_BY_BLOCK_M records a drift of about +/-25% across block "
            "sizes, and that drift is `phi` growing with BM/BN rather than a "
            "different miss fraction, so an imported alpha is an alpha of "
            "ANOTHER TILING and the import is a bound, not a substitution.")
    lines = [
        f"measured is the FITTED alpha ({interval.render()}), {alpha_source}. "
        "It is not a crossing ratio and no crossing was measured to produce it.",
        (f"the interval is DISJOINT from the band and lies {where} it: "
         f"measured {interval.point:.3f} [{interval.lo:.3f}, {interval.hi:.3f}] "
         f"is {where} the band [{band[0]}, {band[1]}]"
         if where else
         f"the interval [{interval.lo:.3f}, {interval.hi:.3f}] overlaps the "
         f"band [{band[0]}, {band[1]}]"),
        (f"INFORMATIONAL, the retired one-sided gate: alpha "
         f"{'>' if one_sided else '<='} {GATE3_ALPHA_DISCRIMINATOR:.2f} would "
         f"have read {'PASS' if one_sided else 'FAIL'}. It is not the verdict: "
         "it passes for everything from 0.33 to 1.0, so it cannot fail from "
         "above and every committed surface fit clears it."),
        observed,
        f"the same alpha restates the model's crossing ratio as "
        f"1 + {alpha_hat:.3f} = {1.0 + alpha_hat:.3f}x, which is what the "
        "retired ratio gate printed as its 'measured' value. That number is "
        "algebra over this line, not a second observation.",
        "model's crossing ratio: " + "   ".join(band_lines),
        f"the retracted alpha={RETRACTED_ALPHA} would put that ratio at "
        + (f"{retracted_ratio:.3f}x" if retracted_ratio else "no crossing"),
        provenance_line]
    # BOTH ESTIMATORS ON EVERY ROW HERE TOO. This is the second place in the
    # file that prints a per-BLOCK_M alpha; the ladder table is the first. A
    # statistic added to one of two print sites is this repository's recurring
    # defect, so both take their w from `LadderFit.w_note` and there is one
    # rendering of it.
    for bm, fit in sorted(fits.items()):
        if fit.alpha is not None:
            lines.append(f"  BLOCK_M={bm:3d}  alpha {fit.alpha:.3f} from "
                         f"{fit.memory_points} memory-bound treads, fit error "
                         f"{fit.mean_rel_err:.2%}; {fit.w_note()}")
        else:
            lines.append(f"  BLOCK_M={bm:3d}  alpha not identifiable "
                         f"({fit.memory_points} memory-bound tread(s)): "
                         f"{fit.outcome_reason or fit.outcome}; "
                         f"{fit.w_note()}")
    # `measured` stays the bare fitted alpha, which is the correction the
    # retired ratio gate already received and what a reader compares across
    # reports. The INTERVAL and the DIRECTION go in `threshold`, so the one
    # greppable RESULT line still carries both.
    threshold = (f"interval [{interval.lo:.3f}, {interval.hi:.3f}] against "
                 f"[{band[0]}, {band[1]}]"
                 + (f" -- DISJOINT, {where} the band" if where
                    else " -- overlaps"))
    if blocked:
        # The overlap is still stated, and it is still not the verdict. A
        # threshold line reading "overlaps" beside an UNDECIDED verdict would
        # be the same one-line contradiction this gate was just corrected for.
        threshold += (f"; NOT SCORED, BLOCK_M={lo} is "
                      f"{UNDECIDED_PARALLEL_BRANCH}")
    provenance["blocked_by_target_tile"] = blocked
    # `measured` stays the bare fitted alpha. Which tile it belongs to is in
    # `basis` (IMPORTED), in the threshold and in the provenance line;
    # `tests/test_gate_units_and_ridge.py` pins this field's exact shape and
    # that file is not this one's to edit.
    return Gate(3, claim, verdict, f"alpha {alpha_hat:.3f}", threshold, lines,
                basis=IMPORTED if imported else DERIVED, provenance=provenance)


@dataclass(frozen=True)
class Bracketing:
    """Was the sweep long enough for "no crossing" to be evidence.

    An unbracketed sweep reporting an absence is worthless, so the three things
    that make the absence mean something are computed and printed rather than
    implied:

      * a POSITIVE CONTROL, some other block size that did cross inside the same
        grid, which proves the instrument can see a crossing;
      * a HORIZON, the largest rows-per-expert at which any competing hypothesis
        places this block size's crossing, times a safety factor;
      * SATURATION, the fraction of the modelled AI ceiling the last tread
        reached, since a curve still climbing has not finished rising.

    THE CONTROL IS SCORED AGAINST `ridge x bandwidth`, NOT AGAINST THE RUN'S OWN
    PLATEAU, and that is the change that makes it a control at all. Against the
    plateau -- the arm's own maximum -- something always reaches 1.00 by
    construction, because the plateau IS the maximum, so the control could never
    fail and the check examined nothing. Against the absolute roof it is a real
    question, and in all 26 published reports the answer is no: the plateau is
    46.5-75.6% of that card's own `ridge x bandwidth`, so nothing in any of
    those sweeps reached a compute roof and none of them was ever entitled to
    read an absence at BLOCK_M=64 as evidence about BLOCK_M=64.
    """

    reached_rows: float
    reached_tiles: int
    horizon_rows: float
    positive_control: int | None
    saturation: float
    last_gain: float
    #: The best roof fraction any OTHER block size reached, in units of
    #: `ridge x bandwidth`. Recorded even when it fails the control threshold,
    #: because "the best anything managed was 0.53" is the diagnosis.
    best_other_roof_fraction: float = 0.0
    #: `ridge x bandwidth` in TFLOP/s, the denominator every fraction here uses.
    roof_tflops: float = 0.0

    @property
    def sufficient(self) -> bool:
        return (self.reached_rows >= self.horizon_rows
                and self.positive_control is not None
                and self.saturation >= 0.90)

    def lines(self) -> list[str]:
        control = (f"BLOCK_M={self.positive_control} reached "
                   f"{self.best_other_roof_fraction:.2f} of ridge x bandwidth "
                   "inside this same grid"
                   if self.positive_control is not None
                   else "NOTHING reached the compute roof in this grid (best "
                        f"other block size {self.best_other_roof_fraction:.2f} "
                        f"of ridge x bandwidth = {self.roof_tflops:.0f} "
                        "TFLOP/s), so the sweep never demonstrated it can "
                        "detect a crossing at all")
        return [
            f"swept to {self.reached_rows:.0f} rows per expert "
            f"({self.reached_tiles} M-tiles)",
            f"horizon {self.horizon_rows:.0f} rows: 2x the crossing the "
            f"retracted alpha={RETRACTED_ALPHA} predicts for this block size",
            f"positive control: {control}",
            f"modelled AI at the last tread is {self.saturation:.1%} of the "
            f"ceiling, and the last tread gained {self.last_gain:.1%} of "
            "throughput over the one before it"]


def bracketing(cells, block_m: int, alpha: float, ridge: float, b: int,
               fits, roof_tflops: float, safety: float = 2.0) -> Bracketing:
    """`roof_tflops` is `ridge x bandwidth`, NOT the run's own plateau.

    Passing the plateau here is what made the positive control vacuous: the
    plateau is the maximum over the same cells the control is read from, so
    some block size always scores 1.00 against it.
    """
    pts = ladder_points(cells, block_m)
    reached_tiles = pts[-1][0] if pts else 0
    reached = float(reached_tiles * block_m)
    retracted = predict_tile(block_m, RETRACTED_ALPHA, ridge, b)
    horizon = safety * (retracted.crossing_rows or 0.0)
    # The control is MEASURED, not fitted: some other block size actually got
    # to the roof inside this same grid. A fitted `crosses` would make the
    # control depend on the same branch assignment gate 4 is arguing about, and
    # a fit that lost its memory branch to noise would silently remove the
    # control and turn a real absence into UNDECIDED.
    control = None
    best_other = 0.0
    for bm in sorted({c.block_m for c in cells}):
        if bm == block_m:
            continue
        tp_other = _throughput_ladder(cells, bm, roof_tflops)
        if not tp_other:
            continue
        best = max(v for _, v in tp_other)
        best_other = max(best_other, best)
        if best >= COMPUTE_BOUND_FRACTION:
            control = bm
    cap = ai_cap(block_m, alpha, b)
    ai = (2.0 * reached / b) / q_of_tiles(max(reached_tiles, 1), alpha) if reached else 0.0
    gain = 0.0
    tp = _throughput_ladder(cells, block_m, roof_tflops)
    if len(tp) >= 2 and tp[-2][1] > 0:
        gain = tp[-1][1] / tp[-2][1] - 1.0
    return Bracketing(reached, reached_tiles, horizon, control,
                      ai / cap if cap else 0.0, gain,
                      best_other_roof_fraction=best_other,
                      roof_tflops=roof_tflops)


def _throughput_ladder(cells, block_m: int, denominator: float):
    """`(tiles, useful throughput as a fraction of `denominator`)` per tread.

    The caller chooses the denominator and OWNS what the fraction then means.
    Gate 4 and `bracketing` pass `ridge x bandwidth`, so their fractions are
    fractions of peak compute and are comparable across arms and cards. A
    caller that passes the run's own plateau gets an ARM-RELATIVE number whose
    denominator moved 145.7-198.4 TFLOP/s inside one A100 session, which is not
    a quantity any threshold can be stated against.

    A DRIFTED CELL IS EXCLUDED HERE FOR THE REASON `ladder_treads` EXCLUDES IT.
    The two builders disagreed for one commit on 2026-09-09: this one kept
    every cell that ran, `ladder_treads` dropped the drifted ones, and
    `tile_cap_test.analyse` reads BOTH on one page, so its V2 flatness figure
    came off a BLOCK_M=256 cell whose `clock_drift_ok` is False while the
    reference fit on the same page refused that exact cell. Neither pool was
    wrong; having two of them silently was. A clock that moved across a cell's
    own trials makes its median a blend of two operating points, and no
    denominator repairs that.
    """
    out = []
    for c in sorted((c for c in cells
                     if c.block_m == block_m and c.aligned and c.status == "ok"
                     and not c.clock_excluded),
                    key=lambda c: c.tiles_per_expert):
        if denominator > 0:
            out.append((c.tiles_per_expert, c.useful_tflops / denominator))
    return out


def null_block_m(block_sizes) -> int:
    """The tile gate 4 asks its absence question about: 64, or the smallest.

    One function rather than the same conditional in three places, because the
    predictions block registers the threshold, `bracketing` measures the horizon
    and `gate_4_no_crossing` scores the verdict, and those three reading
    different tiles would be a gate scored against a threshold registered for
    something else.
    """
    return 64 if 64 in block_sizes else min(block_sizes)


def gate_4_roof_fraction(*, block_m: int, alpha: float, ridge: float, b: int
                         ) -> tuple[float, float, float]:
    """`(model ceiling, retracted ceiling, threshold)`, all as fractions of peak.

    Both worlds cap the block size's attainable throughput at `ai_cap / ridge`
    of `ridge x bandwidth`, CLAMPED AT 1.0 -- a kernel cannot exceed the roof no
    matter how large its arithmetic intensity, and the retracted alpha puts the
    BLOCK_M=64 ceiling at 640/163 = 3.9 if the clamp is left out, which would
    make the midpoint 2.3 and the gate unfailable.

    The threshold is the midpoint of the two, so it is a DISCRIMINATOR in the
    same sense as gate 3's 0.33: above it the refit's ceiling is violated,
    below it the retracted world's is not reached.
    """
    model = min(ai_cap(block_m, alpha, b) / ridge, 1.0)
    retracted = min(ai_cap(block_m, RETRACTED_ALPHA, b) / ridge, 1.0)
    return model, retracted, 0.5 * (model + retracted)


def gate_4_no_crossing(cells, fits, *, block_m: int, plateau: float,
                       alpha: float, ridge: float, b: int, brack: Bracketing,
                       roof_tflops: float) -> Gate:
    """Does BLOCK_M=64 fail to reach the compute roof.

    BOTH SIDES ARE NOW FRACTIONS OF `ridge x bandwidth`, and that is the fix.
    The retired form scored `top > 0.85` where `top` was a fraction of the ARM'S
    OWN measured plateau, while the 0.85's stated rationale (`cap/ridge` = 0.716
    plus room for the fused layer's non-GEMM work) is a fraction of PEAK
    COMPUTE. Those denominators differ by the plateau's own shortfall, which
    across the 14 s3 reports runs 50.5-71.8% of `ridge x bandwidth`; in the
    gate's own units the model ceiling for the arm that "failed" it is 1.42, so
    0.891 was never evidence against anything. Both FAILs it ever produced are
    qwen2 at GROUP_SIZE_M=64, where the BLOCK_M=256 reference that SETS the
    plateau falls to 41.36 rows/ms against 45.1-45.7 elsewhere while BLOCK_M=64
    does not move: a denominator artefact, and the same config scored 0.871 FAIL
    on one run and 0.841 PASS on another.

    WHAT WAS NARROWED AS WELL AS RESCALED. A PASS is only allowed when the sweep
    demonstrated it could have produced a FAIL -- some OTHER block size actually
    reached `COMPUTE_BOUND_FRACTION` of `ridge x bandwidth` in this same grid.
    Without that the gate is a check that examined nothing: an absence measured
    by an instrument never shown to detect a presence. In every published report
    to date the plateau itself is 46.5-75.6% of that card's own roof, so no such
    control existed and the honest verdict there is UNDECIDED, not PASS.

    The arm-relative number the old gate scored is still printed, labelled, as a
    diagnostic. It is not the verdict.
    """
    fit = fits.get(block_m)
    tp = _throughput_ladder(cells, block_m, roof_tflops)
    model_ceiling, retracted_ceiling, threshold = gate_4_roof_fraction(
        block_m=block_m, alpha=alpha, ridge=ridge, b=b)
    separates = retracted_ceiling - model_ceiling >= GATE4_MIN_SEPARATION
    claim = (f"BLOCK_M={block_m} never reaches the compute roof "
             "(ridge x bandwidth)")
    gate_text = (f"< {COMPUTE_BOUND_FRACTION:.2f} of ridge x bandwidth"
                 + (f", and <= {threshold:.3f} (the midpoint of this run's "
                    f"ceiling {model_ceiling:.3f} and the rival world's "
                    f"{retracted_ceiling:.3f})" if separates else
                    f"; the ceiling test is SKIPPED because this run's ceiling "
                    f"{model_ceiling:.3f} and the rival's "
                    f"{retracted_ceiling:.3f} do not separate"))
    provenance = {
        "units": "fraction of ridge x bandwidth (peak compute)",
        "roof_tflops": roof_tflops,
        "plateau_tflops": plateau,
        "plateau_over_roof": (plateau / roof_tflops) if roof_tflops else None,
        "model_ceiling": model_ceiling,
        "retracted_ceiling": retracted_ceiling,
        "threshold": threshold,
        "positive_control_block_m": brack.positive_control,
        "best_other_roof_fraction": brack.best_other_roof_fraction,
        "retired_denominator": "the arm's own plateau; see GATE4_ROOF_FRACTION",
    }
    if not tp or fit is None or roof_tflops <= 0:
        why = ("no aligned cells at this block size" if not tp or fit is None
               else "no roof to measure against: ridge x bandwidth is zero")
        return Gate(4, claim, UNDECIDED, why, gate_text,
                    ["The sweep produced no exactly-full tile stack here."
                     if not tp or fit is None else
                     "ridge x bandwidth resolved to zero, so every fraction "
                     "below would divide by nothing. REFUSED."],
                    provenance=provenance)
    top = max(v for _, v in tp)
    provenance["peak_roof_fraction"] = top
    arm_top = (max(v for _, v in _throughput_ladder(cells, block_m, plateau))
               if plateau > 0 else None)
    provenance["peak_arm_relative_fraction"] = arm_top
    lines = [
        f"measured in fractions of ridge x bandwidth ({roof_tflops:.0f} "
        f"TFLOP/s); the refit caps this block size at cap/ridge = "
        f"{ai_cap(block_m, alpha, b):.1f}/{ridge:.1f} = {model_ceiling:.3f} "
        f"and the retracted alpha={RETRACTED_ALPHA} at {retracted_ceiling:.3f}",
        "throughput per tread as a fraction of that roof: "
        + ", ".join(f"n={n}:{v:.2f}" for n, v in tp),
        f"this arm's own plateau is {plateau:.1f} TFLOP/s = "
        + (f"{plateau / roof_tflops:.1%}" if roof_tflops else "n/a")
        + " of that roof. Against the plateau this block size peaks at "
        + (f"{arm_top:.3f}" if arm_top is not None else "n/a")
        + ", which is the number the retired gate scored against 0.85. It is "
          "ARM-RELATIVE and it is not the verdict.",
    ]
    if fit.slope_memory is not None and fit.compute_slope:
        c = fit.compute_slope
        src = ("measured on this block size's own compute-bound treads"
               if fit.slope_compute else "scaled from the reference by C ~ BLOCK_M")
        lines.append(
            f"per-tile slopes: memory B={fit.slope_memory:.4f} ms against "
            f"compute C={c:.4f} ms ({src}). C > B is the condition for a "
            f"crossing to exist at all, and it is "
            f"{'MET' if c > fit.slope_memory else 'NOT met'}.")
    lines += brack.lines()

    provenance["worlds_separate"] = separates
    provenance["reached_roof"] = top >= COMPUTE_BOUND_FRACTION
    provenance["exceeded_ceiling_midpoint"] = separates and top > threshold
    if not separates:
        # Reported, not scored around. Two ceilings that coincide say the run's
        # own alpha and the rival's put this block size in the same place, so
        # the ceiling comparison cannot discriminate. The DIRECT question --
        # did it reach the roof -- still can, and it is asked below.
        lines.insert(1, f"THE CEILING TEST IS SKIPPED AT BLOCK_M={block_m}: "
                        f"this run's ceiling {model_ceiling:.3f} and the "
                        f"retracted world's {retracted_ceiling:.3f} differ by "
                        f"{retracted_ceiling - model_ceiling:.3f}, under the "
                        f"{GATE4_MIN_SEPARATION:.2f} needed to tell them "
                        "apart. Only the direct roof question is scored.")

    # FAIL first, on the DIRECT question, judged by the same criterion the
    # positive control is judged by so that "reached the roof" means one thing
    # in this file. A block size that reached the roof crossed, and bracketing
    # governs an ABSENCE only: a sweep that watched the roof being reached is
    # bracketed by demonstration whatever the horizon says.
    if top >= COMPUTE_BOUND_FRACTION:
        return Gate(4, claim, FAIL, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    [f"BLOCK_M={block_m} DID reach the roof, so it crosses and "
                     "the AI ceiling is not where this study put it."] + lines,
                    provenance=provenance)
    # The weaker falsification, and the one that needs the two worlds to be
    # apart: the block size did not reach the ROOF but did pass the ceiling its
    # own alpha puts on it, by more than half the distance to the rival world.
    if separates and top > threshold:
        return Gate(4, claim, FAIL, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    [f"BLOCK_M={block_m} stayed below the roof but passed "
                     f"{threshold:.3f}, the midpoint between its modelled "
                     f"ceiling {model_ceiling:.3f} and the rival world's "
                     f"{retracted_ceiling:.3f}. The ceiling is not where this "
                     "run's alpha puts it."] + lines,
                    provenance=provenance)
    if not brack.sufficient:
        # NON-VACUITY. Without a control this gate is an absence reported by an
        # instrument never shown to detect a presence, and every published
        # report to date is in exactly this state.
        return Gate(4, claim, UNDECIDED, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    ["THE SWEEP IS NOT BRACKETED, so an absence here is not "
                     "evidence of absence and this gate refuses to score it."]
                    + lines,
                    provenance=provenance)
    return Gate(4, claim, PASS, f"peak {top:.3f} of ridge x bandwidth",
                gate_text, lines, provenance=provenance)


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

@dataclass
class Report:
    lines: list[str]
    gates: list[Gate]
    payload: dict

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def analyse(cells, cfg, *, block_sizes, alpha: float, ridge: float,
            bandwidth_gbps: float, b: int, model_name: str, dtype: str,
            compiles: dict[int, int], executed: dict[int, int],
            sm_count: int, sm_source: str, pinned: dict | None = None,
            ridge_band: tuple[float, float] | None = None,
            ridge_source: str = "", ridge_band_source: str = "",
            capability=None, card: str = NO_CARD_SLUG,
            bandwidth_source: str = "", prov=None) -> Report:
    """Everything between the timings and the verdicts. No GPU, no I/O.

    Kept pure and passed only cells so that `--self-test` and the test suite
    exercise the SAME code the pod run prints, rather than a second
    implementation that agrees with it until it does not.

    `ridge_band` IS NOT DEFAULTED TO `RIDGE_BAND`. That module constant is the
    withdrawn gap between two H200 compute calibrations, and defaulting to it
    is exactly how all 7 published A100 reports came to carry a band belonging
    to neither card. When
    the caller does not state a band, the band is this run's own single ridge
    twice over, and the report says so -- a degenerate band is honest about
    being one calibration, a borrowed band is not.

    SIGNATURE EXTENDED, never narrowed. `bandwidth_source` and `prov` are
    optional: `bandwidth_source` names where the DENOMINATOR of every predicted
    millisecond came from, which `resolve_bandwidth` now refuses to leave
    unstated, and `prov` is the `moe.bench.provenance.Provenance` block that
    `stamp`s the payload with the commit, the card, the instrument and the two
    sources. Callers that pass neither (the test suite, and any script that
    only wants the analysis) get a report whose provenance says "not supplied
    by caller" for those fields, which is the honest record rather than a
    fabricated one.

    `cfg` AND `model_name` MUST NAME THE SAME MODEL, and since 2026-09-10 that
    is checked rather than assumed. Every computed number in the report comes
    off `cfg`; `model_name` is only the label the header and the payload's
    "model" key carry. While the weight-stream denominator was resolved from
    the STRING the two could disagree and produce a w against one geometry
    inside a report built from another. The denominator now comes off `cfg`, so
    the only thing a disagreement could still do is mislabel the report, and a
    mislabelled report is the thing this repository spends its time undoing.
    Every caller in the tree passes `cfg.name`, so this refuses nothing that
    exists; it refuses the next caller that gets it wrong.
    """
    if model_name and cfg.name and model_name != cfg.name:
        raise ValueError(
            f"model_name={model_name!r} but cfg.name={cfg.name!r}: the report's "
            "label and the geometry every number in it is computed from would "
            "name two different models. Pass the config you want measured and "
            "its own name")
    if ridge_band is None:
        ridge_band = (ridge, ridge)
    ridge_band = (min(ridge_band), max(ridge_band))
    lines: list[str] = []
    timed = [c for c in cells if c.status == "ok" and c.ms_p50 > 0]
    # THE DRIFT EXCLUSION REACHES EVERY GATE, not just the ladder fit.
    # R5 asked only that a mis-clocked cell stay out of the memory-branch fit,
    # and `ladder_treads` does that. But `plateau` and gates 1, 2 and 4 were
    # handed the unfiltered list, and gate 4's claim is an ABSENCE --
    # "BLOCK_M=64 never reaches the compute roof" -- scored against `plateau`.
    # A cell whose clock moved mid-measurement carries a median that belongs to
    # neither operating point, so it biases a maximum in whichever direction
    # the governor happened to move. `ok` below is therefore the SCORED set:
    # everything timed, less the cells whose clock DRIFTED. `timed` is kept
    # only to count what was dropped, and the ladder fits still read `timed` so
    # they can report their own exclusions per block size. None is not an
    # exclusion anywhere; see `ladder_treads`.
    ok = [c for c in timed if not c.clock_excluded]
    excluded_from_gates = len(timed) - len(ok)
    # BOTH SIDES OF LEVEL ARE IN `ok` AND ARE COUNTED, NOT DROPPED. Their
    # milliseconds are measurements; what the fixed roof does not describe is
    # their fraction of that roof, so the report names how many cells the gates
    # read that way and which direction the fixed-roof fraction is off in.
    boosted_from_gates = sum(1 for c in timed if c.clock_boosted)
    sagged_from_gates = sum(1 for c in timed if c.clock_sagged)
    aligned = [c for c in ok if c.aligned]
    plateau = max((c.useful_tflops for c in aligned), default=0.0)
    noise = statistics.median([c.rel_spread for c in ok]) if ok else 0.0

    # The roof and the pinned tile constants are passed in because the
    # qualification is a LEVEL test as well as a shape test, and level cannot be
    # judged without knowing what the card can do. There is no default: a
    # reference qualified against an unknown ceiling is the defect, not the fix.
    ref = compute_reference(ok, block_sizes, cfg=cfg, ridge=ridge,
                            bandwidth_gbps=bandwidth_gbps, b=b,
                            pinned=pinned or FIXED, capability=capability)
    # The margin scales with what the timing actually did. A fixed 2% was too
    # small at 2% spread: the reference slope carries the same spread, and a
    # compute branch estimated 2% low makes every compute-bound tread stand
    # "above" it, which is how a BLOCK_M=128 ladder planted at alpha=0.10
    # reported 0.80.
    margin = max(MEMORY_BRANCH_MARGIN, 3.0 * noise)
    # `timed`, not `ok`: `ladder_treads` does its own exclusion and RETURNS THE
    # COUNT, which is what lets each block size say how many treads it lost.
    # Handing it the already-filtered list would report every ladder as having
    # lost nothing.
    treads = {bm: ladder_treads(timed, bm) for bm in block_sizes}
    off_band = {bm: off_band_treads(timed, bm) for bm in block_sizes}
    # `cfg`, `dtype` and the bandwidth go in here so that every fit carries the
    # denominator of its own weight-stream slope. The rate is this run's own
    # `bandwidth_gbps` with the caller's `bandwidth_source` string attached, so
    # a w printed below says which measured rate it is a fraction of rather
    # than leaving a reader to assume the card's datasheet.
    #
    # `cfg` AND NOT `model_name`, since 2026-09-10. This function takes both,
    # and every other number in the report -- the roof, the predictions, the
    # activation slope, the reference -- is computed from the `cfg` OBJECT.
    # Handing the STRING down made the weight set be re-resolved through
    # `MODEL_CONFIGS`, so a caller whose two arguments disagreed got a w
    # divided by one geometry inside a report built from another, with nothing
    # on the page to say so. It also made an unregistered or unverified name
    # abort the whole report; a config that is already in hand cannot.
    fits = {bm: fit_ladder(pts, bm, ref, margin, excluded_drifted=dropped,
                           kept_low_clock=off_band[bm][0],
                           kept_high_clock=off_band[bm][1],
                           model=cfg, dtype=dtype,
                           bandwidth_gbps=bandwidth_gbps,
                           bandwidth_source=bandwidth_source)
            for bm, (pts, dropped) in treads.items()}
    fits = {bm: f for bm, f in fits.items() if f.points}
    excluded_total = sum(dropped for _, dropped in treads.values())
    boosted_total = sum(hi for _, hi in off_band.values())
    sagged_total = sum(lo for lo, _ in off_band.values())

    preds_lo = predictions(block_sizes, alpha, ridge_band[0], b)
    preds_hi = predictions(block_sizes, alpha, ridge_band[1], b)

    lines.append("")
    lines.append("PREDICTIONS, stated before the run and not adjusted after it")
    lines.append(f"  alpha {alpha:.3f} (band {ALPHA_BAND[0]}-{ALPHA_BAND[1]}), "
                 f"{dtype} at {b} bytes, ridge {ridge:.1f} Op/B, band "
                 f"{ridge_band[0]:.1f}-{ridge_band[1]:.1f} Op/B")
    lines.append(f"  ridge source: {ridge_source or 'NOT STATED by the caller'}")
    lines.append(f"  ridge band source: "
                 f"{ridge_band_source or 'NOT STATED by the caller'}")
    # THE DENOMINATOR OF EVERY PREDICTED MILLISECOND, named beside the ridge and
    # not below it. A ridge measured on this card divided by a bandwidth
    # inherited from a published H200 triad is a HYBRID roof, and until
    # 2026-09-02 `resolve_bandwidth` produced one silently whenever the
    # calibration was missing or unreadable.
    lines.append(f"  bandwidth source: {bandwidth_gbps:.1f} GB/s, "
                 f"{bandwidth_source or 'NOT STATED by the caller'}")
    # Beside the two rulers, because it is the third thing a reader has to know
    # to judge a number here and `provenance.iters` is a single median standing
    # for it. See `observed_iters`.
    lines.append("  " + iters_line(cells))
    if ridge_band[0] == ridge_band[1]:
        lines.append("  the band is DEGENERATE: one calibration, so which tread "
                     "a crossing lands in is not bracketed by this run")
    lines.append(f"  BLOCK_M   AI cap   crossing @ridge {ridge_band[0]:<10.1f} "
                 f" crossing @ridge {ridge_band[1]:.1f}")
    for bm in block_sizes:
        p, ph = preds_lo[bm], preds_hi[bm]
        def fmt(pred):
            if pred.crossing_rows is None:
                return "NO CROSSING EVER    "
            tok = pred.crossing_tokens(cfg.num_experts, cfg.top_k)
            return f"r={pred.crossing_rows:7.1f} T={tok:8.0f} n={pred.first_compute_tread}"
        lines.append(f"  {bm:7d} {p.ai_cap:8.1f}   {fmt(p)}   {fmt(ph)}")
        why = no_crossing_reason(p)
        if why:
            lines.append(f"          {why}")
    r_lo = crossing_ratio(preds_lo, 128, 256)
    r_hi = crossing_ratio(preds_hi, 128, 256)
    if r_lo and r_hi:
        lines.append(f"  128-over-256 crossing ratio: {r_lo:.3f}x at the low "
                     f"ridge, {r_hi:.3f}x at the high one; the retracted "
                     f"alpha={RETRACTED_ALPHA} says 1.100x. MODEL, not "
                     "measurement: no crossing has been observed by this study.")
    # REGISTERED HERE, above the measurement, because both thresholds are
    # derived from this run's ridge and alpha rather than hardcoded, and a
    # derived threshold printed only beside its own verdict is a threshold a
    # reader cannot tell from a threshold chosen after the fact.
    null_bm = null_block_m(block_sizes)
    g4_model, g4_retracted, g4_threshold = gate_4_roof_fraction(
        block_m=null_bm, alpha=alpha, ridge=ridge, b=b)
    # LABELLED, PER TILE, BESIDE THE CAPS THEMSELVES. `ai_cap` is
    # `2 BM / (alpha b)` computed from a FITTED alpha, and a B/(A+B) fit returns
    # `(alpha_b + phi)/(1 + phi + delta)`, so every cap in the table above is
    # high by `1 + phi + delta`. At BM=128 that is larger than the cap-to-ridge
    # gap the cap is being used to decide, which is why it is printed here and
    # not left in a module docstring.
    bn = (pinned or FIXED)["BLOCK_SIZE_N"]
    lines.append(
        "  every AI cap above is 2*BM/(alpha*b) from a FITTED alpha and is HIGH "
        "by ai_model.lin_overstatement = 1+phi+delta (alpha_a is unmeasured "
        "anywhere in this repo and delta is taken as 0, so each range is a "
        "bracket and a lower bound): "
        + ", ".join(f"BM={bm} {lo:.2f}-{hi:.2f}x" for bm, (lo, hi) in
                    ((bm, cap_overstatement(cfg, bm, bn, b))
                     for bm in block_sizes)))
    lines.append(f"  GATE 3 will test alpha > {GATE3_ALPHA_DISCRIMINATOR:.2f} "
                 f"(midpoint of {alpha:.3f} and {RETRACTED_ALPHA}) as an "
                 f"INFORMATIONAL line; the VERDICT is whether the fitted "
                 f"alpha's interval overlaps the band "
                 f"[{ALPHA_BAND[0]}, {ALPHA_BAND[1]}], in both directions")
    lines.append(
        f"  GATE 4 will test BLOCK_M={null_bm} against the roof in fractions "
        f"of ridge x bandwidth: this run's ceiling {g4_model:.3f}, the "
        f"retracted world's {g4_retracted:.3f}, so the gate is "
        f"{COMPUTE_BOUND_FRACTION:.2f} (reached the roof) and "
        f"{g4_threshold:.3f} (passed its own ceiling)")

    lines.append("")
    lines.append(f"MEASURED  {model_name} {dtype}  {len(ok)} cells, "
                 f"{len(aligned)} of them exactly-full tile stacks")
    model_roof = ridge * bandwidth_gbps * 1e9 / 1e12
    lines.append(f"  compute plateau {plateau:.1f} TFLOP/s useful, taken as the "
                 "roof every roof fraction below is against")
    # The plateau is the sweep's own maximum, so it is a roof only if something
    # in the sweep actually reached one. `ridge x bandwidth` is what the roof
    # should be; a plateau far below it means nothing here is compute bound and
    # every roof fraction is against a ceiling that does not exist.
    lines.append(f"  that plateau is {plateau / model_roof:.1%} of "
                 f"ridge x bandwidth ({model_roof:.0f} TFLOP/s). Far below 100% "
                 "means nothing in the sweep reached a roof and every roof "
                 "fraction is relative to something that is not one.")
    lines.append(f"  per-cell timing spread, median {noise:.2%}")
    lines += ref.render()
    lines.append(f"  {sm_count} SMs ({sm_source}), one resident CTA per SM "
                 "assumed for every wave count")
    if ok:
        lines.append(
            f"  up-GEMM waves run {min(c.waves_up for c in ok):.1f} to "
            f"{max(c.waves_up for c in ok):.1f} across the sweep. The smallest "
            "is the smallest tile stack, and even it is many waves deep, so no "
            "cell here is a partial-wave measurement and occupancy cannot be "
            "the thing that moved between two adjacent cells.")

    lines.append("")
    lines.append("THE LADDER: milliseconds per exactly-full tile stack, which is "
                 "where gates 2, 3 and 4 are read")
    lines.append("  alpha is a LOWER bound (the fused layer's fixed cost sits "
                 "in the denominator); alpha-hi takes that cost out, and "
                 "alpha-corrected also removes activation traffic")
    lines.append("  alpha here is B/(A+B), which is (alpha_b+phi)/(1+phi+delta) "
                 "and NOT a weight miss fraction: see moe/bench/ai_model.py "
                 "and LadderFit.alpha")
    # THE SECOND ESTIMATOR, BESIDE THE FIRST. Every alpha column above divides
    # B by a fitted LEVEL, which is the ladder extrapolated back to n=0; `w`
    # divides the same B by a MEASURED time. Both are printed on every row, the
    # way both roof fractions are, because the 100,144 published rows were
    # scored on B/(A+B) and stay readable exactly as they are.
    #
    # THE RATE CONDITION IS ON THIS LINE, since 2026-09-10. `w` is
    # `alpha_b + phi` in units of one weight read only AT THE RATE THE MEMORY
    # BRANCH ACHIEVED, and this line divides by whatever `bandwidth_gbps` the
    # run resolved -- on the H200 session the calibrated TRIAD figure, 4374.30
    # GB/s, while the same card's committed `read_stream` pattern is 4612.25,
    # 5.4% faster, and a weight stream is a pure read. THE PRINTED LINE READS
    # THAT GAP OFF THE ATTACHED CARD (`read_versus_denominator`) instead of
    # carrying this paragraph's H200 figure into every report: on the committed
    # A100 profile the read pattern is 3.06% BELOW its triad, so the typed
    # guidance pointed backwards there. Divided by the faster
    # rate, tile_cap's BM=16 ladder reads 1.1085 where this line prints 1.0514,
    # so calling the printed number an upper bound on alpha_b can be FALSE by
    # more than the gap it is bounding. `weights.py`, `ai_model.py` and
    # `LadderFit.weight_streams` all carried the clause; the one place a reader
    # actually sees did not, which is the recurring defect.
    #
    # ROUTED ONLY, AND SAID SO. `--model` accepts qwen2-57b-a14b and the three
    # deepseek entries, whose layers also carry a shared expert.
    # `weights.layer_weight_bytes` REFUSES those models outright rather than
    # return the routed set under a whole-layer name; this line printed the
    # routed set under the unqualified name "the expert weight set", the same
    # distinction with the opposite care, two files apart.
    try:
        stream = weight_stream_ms(cfg, dtype, bandwidth_gbps)
        gb = routed_expert_weight_bytes(cfg, dtype) / 1e9
        denominator = (f"{gb:.4f} GB in {stream:.4f} ms at "
                       f"{bandwidth_gbps:.1f} GB/s "
                       f"({bandwidth_source or 'rate NOT STATED by the caller'})")
    except WeightSetRefused as exc:
        # The same rule the ladder rows follow: a weight set this repository
        # will not guess blanks the w column, it does not kill the report.
        denominator = f"NOT AVAILABLE: {exc}"
    lines.append(
        f"  w is B divided by one full stream of the ROUTED expert weight set "
        f"(the set the fused grouped GEMM streams; a shared expert, where the "
        f"model has one, is a different kernel and is not in this count): "
        f"{denominator}. No fitted level, no intercept, no fixed cost. It "
        "scales 1:1 in that rate, and under the three-term model it is "
        "alpha_b + phi AT THIS RATE, so it bounds the weight miss fraction "
        "from above only if the memory branch achieved this bandwidth, which "
        "this run did not measure. A weight stream is a pure read and the rate "
        "above is whichever pattern the source names, so this line reads the "
        "card's own bandwidth_patterns rather than quoting one card's gap: "
        + read_versus_denominator(card, bandwidth_gbps) + ".")
    lines.append(
        "  D>A marks a ladder whose reference fixed cost stands above its own "
        "fitted intercept. alpha-hi = B/(A+B-D) exceeds 1 exactly when D>A and "
        "the column has a value at all; where D also exceeds the whole level "
        "A+B the alpha-hi column reads n/a, because B/(A+B-D) is then a "
        "division by a level that is zero or negative and is not reported. "
        "Either way those "
        "rows are arithmetic about an extrapolation and not a miss fraction. "
        "Such a ladder is LABELLED and kept: refusing it would have deleted "
        "four of bn_decomposition's six cells on 2026-09-10. w does not depend "
        "on A or D at all.")
    if excluded_total or excluded_from_gates:
        lines.append(
            f"  {excluded_from_gates} cell(s) excluded for a DRIFTING clock: "
            "the SM clock moved while they were timed, so each median is a "
            "blend of two operating points and the time belongs to neither.")
        # WHICH GATES THE EXCLUSION REACHED, named, because an exclusion whose
        # extent a reader has to infer is an exclusion nobody can check.
        lines.append(
            f"    Reached: the plateau, the compute reference, the noise "
            f"estimate that sets the margin, and gates 1, 2, 3 and 4. "
            f"{excluded_total} of them were aligned treads and are counted per "
            "ladder below.")
    if boosted_from_gates or boosted_total or sagged_from_gates or sagged_total:
        # BOTH SIDES OF LEVEL, SAID OUT LOUD. Neither is excluded since
        # 2026-09-09, and the reason is stated beside each count so a reader
        # cannot take "kept" for "comparable with the fixed roof".
        lines.append(
            f"  {boosted_from_gates} cell(s) KEPT with LEVEL failed HIGH and "
            f"{sagged_from_gates} KEPT with LEVEL failed LOW: their SM clock "
            "under load sat steadily outside the band around the clock the "
            "roof was measured at. On a power-capped card that band is not a "
            "health check: the under-load clock is set per tile by the "
            "kernel's own power draw (H200, 700 W: BM=128/BN=64 a median 1395 "
            "MHz, BM=256 1650, memory-shaped 1950-1980, against a 1485 MHz "
            "calibration GEMM). Their milliseconds are measurements and stay "
            "in every fit; their fraction of the FIXED roof is off by the "
            "clock ratio -- OVERSTATED on the high side, UNDERSTATED on the "
            "low -- and the roof at their own clock (the driver's "
            "roof_at_cell_clock_tflops) is the one to read them against.")
        lines.append(
            "    The plateau gates 1, 2 and 4 read is still the fixed-roof "
            "fraction, so on a boosted cell it is HIGH and on a sagged cell "
            "LOW: gate 4's absence claim (the null tile never reaches the "
            "roof) is biased toward FAIL by the first and toward PASS by the "
            f"second. {boosted_total} high and {sagged_total} low were aligned "
            "treads and are counted per ladder below.")
    if ref.refused:
        # Said BEFORE the table, because the table is all n/a and a reader who
        # meets the blanks first will reach for the tread count -- which is what
        # happened to the BN=256 arm across two cards and eight published cells.
        lines.append("  EVERY BLANK BELOW IS CAUSED BY THE REFUSED REFERENCE "
                     f"ABOVE (BLOCK_M={ref.refused_block_m}), not by a shortage "
                     "of memory-bound treads. Withdraw this arm; do not table "
                     "it beside arms whose reference qualified.")
    lines.append("  BLOCK_M  treads  memory-bound  alpha   alpha-corrected  "
                 "alpha-hi  w streams/tile  D>A  B ms/tile  C ms/tile  fit err")
    alpha_hat, alpha_source, alpha_source_bm = None, "", None
    alpha_corrected: dict[int, float] = {}
    for bm in sorted(fits):
        f = fits[bm]
        corr = None
        if f.alpha is not None and f.load_ms:
            corr = ((f.slope_memory - activation_slope_ms(cfg, bm, bandwidth_gbps))
                    / f.load_ms)
            alpha_corrected[bm] = corr
        hi = f.alpha_upper
        w = f.weight_streams
        lines.append(
            f"  {bm:7d}  {len(f.points):6d}  {f.memory_points:12d}  "
            + (f"{f.alpha:5.3f}" if f.alpha is not None else "  n/a")
            + "   " + (f"{corr:13.3f}" if corr is not None else "          n/a")
            + "  " + (f"{hi:8.3f}" if hi is not None else "     n/a")
            + "  " + (f"{w.streams:13.4f}" if w is not None else "          n/a")
            + "  " + ("yes" if f.fixed_cost_above_intercept
                      else ("  ." if f.fixed_cost_above_intercept is False
                            else "n/a"))
            + "  " + (f"{f.slope_memory:9.4f}" if f.slope_memory is not None else "      n/a")
            + "  " + (f"{f.slope_compute:9.4f}" if f.slope_compute is not None else "      n/a")
            + f"  {f.mean_rel_err:6.2%}"
            + (f"   [{f.excluded_drifted} excluded for a drifting clock]"
               if f.excluded_drifted else "")
            + (f"   [{f.kept_high_clock} kept with LEVEL high; fixed-roof "
               "fraction overstated by the clock ratio]"
               if f.kept_high_clock else "")
            + (f"   [{f.kept_low_clock} kept with LEVEL low; fixed-roof "
               "fraction understated by the clock ratio]"
               if f.kept_low_clock else ""))
        if f.outcome_reason:
            lines.append(f"           {f.outcome_reason}")
        # Last eligible ladder wins, and the loop runs in ascending order, so
        # this is the LARGEST block size that measured alpha over enough treads
        # -- the closest to the 128 gate 3 has to import it to, and so the
        # shortest extrapolation across the drift `ALPHA_BY_BLOCK_M` records.
        eligible = (corr is not None
                    and ref.block_m is not None
                    and bm != ref.block_m
                    and f.memory_points >= MIN_MEMORY_TREADS)
        if eligible:
            alpha_hat, alpha_source_bm = corr, bm
            alpha_source = (
                f"measured at BLOCK_M={bm} over {f.memory_points} memory-bound "
                "treads, activation traffic subtracted")

    # The interval the two-sided gate 3 is scored on, built from the SAME ladder
    # the point estimate came from so the two cannot describe different fits.
    interval = (alpha_interval(fits[alpha_source_bm], cfg, bandwidth_gbps)
                if alpha_source_bm is not None else None)
    if interval is not None:
        lines.append(f"  alpha scored by gate 3: {interval.render()}")
        # THE THIRD PLACE THIS FILE PRINTS THE SOURCE LADDER'S alpha, and the
        # gate is scored on it. `gate.measured` itself is left alone: it is the
        # greppable RESULT token that 22 published reports carry and that
        # `exit_codes` parses, so w goes on the line beside it rather than
        # inside it.
        lines.append(f"  the same ladder in weight-stream units: "
                     f"{fits[alpha_source_bm].w_note()}")

    consistency = _compute_slope_consistency(fits)
    if consistency:
        lines.append("  consistency: " + consistency)

    gates = [
        gate_0_override(compiles, executed, block_sizes),
        gate_1_steps(ok, cfg, preds_lo, alpha=alpha, ridge=ridge,
                     bandwidth_gbps=bandwidth_gbps, b=b, noise=noise),
        gate_2_direction(ok, cfg, alpha=alpha, retracted=RETRACTED_ALPHA,
                         ridge=ridge, bandwidth_gbps=bandwidth_gbps, b=b,
                         block_sizes=block_sizes),
        gate_3_alpha_discriminates(fits, preds_lo, preds_hi, cfg, lo=128,
                                   hi=256, alpha_source=alpha_source,
                                   alpha_hat=alpha_hat,
                                   alpha_source_bm=alpha_source_bm,
                                   ridge_band=ridge_band, interval=interval),
    ]
    # `model_roof`, not `plateau`: see gate_4_no_crossing and Bracketing on why
    # scoring an absence against the run's own maximum examines nothing.
    brack = bracketing(ok, null_bm, alpha, ridge, b, fits, model_roof)
    gates.append(gate_4_no_crossing(ok, fits, block_m=null_bm, plateau=plateau,
                                    alpha=alpha, ridge=ridge, b=b, brack=brack,
                                    roof_tflops=model_roof))

    lines.append("")
    lines.append("GATES")
    for g in gates:
        lines += g.render()
        lines.append("")

    verdicts = {g.verdict for g in gates}
    if ref.refused:
        # Ranked ABOVE gate 0, because a refused reference is not a failed
        # prediction: it means the instrument, not the hypothesis, is what the
        # arm measured. Nothing in it is a result either way.
        lines.append(
            "READING IT. THE COMPUTE REFERENCE WAS REFUSED ON ITS LEVEL, so "
            "every ladder below was classified against a compute branch that "
            "is the wrong size. No alpha here is a measurement, no blank here "
            "is a null, and the gates are being scored against an instrument "
            "rather than against the hardware. WITHDRAW THIS ARM.")
    elif gates[0].verdict != PASS:
        lines.append("READING IT. Gate 0 failed, so the four settings may not "
                     "have been four kernels. Nothing below gate 0 is evidence.")
    elif verdicts == {PASS}:
        lines.append("READING IT. Every gate passed at alpha "
                     f"{alpha:.3f}. The AI ceiling is real, it is where the "
                     "refit put it, and BLOCK_M 32 and 64 cannot reach the "
                     "compute roof at any batch size.")
    else:
        failed = [g.number for g in gates if g.verdict != PASS]
        lines.append(f"READING IT. Gates {failed} did not pass. A FAIL here is "
                     "a result: it falsifies the tile-corrected roofline at "
                     f"alpha={alpha:.3f} in a specific, named place, which is "
                     "what the gate was built to do.")

    payload = {
        "alpha": alpha, "alpha_band": list(ALPHA_BAND),
        "retracted_alpha": RETRACTED_ALPHA, "ridge": ridge,
        "ridge_band": list(ridge_band),
        "ridge_source": ridge_source or "NOT STATED by the caller",
        "ridge_band_source": ridge_band_source or "NOT STATED by the caller",
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_source": (bandwidth_source
                             or "NOT STATED by the caller"),
        # RENAMED 2026-09-09 with the rule: the exclusion is DRIFT and the two
        # LEVEL sides are records. `cells_excluded_for_clock_level` named a
        # cause that can no longer produce an exclusion.
        "cells_excluded_for_drift": excluded_from_gates,
        "cells_excluded_for_drift_from_ladders": excluded_total,
        # BOTH SIDES, kept and counted. A reader of report.json can tell a
        # ladder whose memory treads all boosted (the H200's normal state) and
        # one whose dense treads all sagged (its normal state for BM=128/BN=64
        # under a power cap) from one whose fixed-roof fractions mean what they
        # say.
        "cells_kept_level_high": boosted_from_gates,
        "cells_kept_level_high_from_ladders": boosted_total,
        "cells_kept_level_low": sagged_from_gates,
        "cells_kept_level_low_from_ladders": sagged_total,
        "clock_exclusion_reaches": [
            "plateau", "compute_reference", "noise_margin",
            "gate_1", "gate_2", "gate_3", "gate_4", "ladder_fits"],
        "ridge_band_degenerate": ridge_band[0] == ridge_band[1],
        "model_roof_tflops": model_roof, "dtype_bytes": b,
        "model": model_name, "dtype": dtype, "fixed": pinned or FIXED,
        "plateau_tflops": plateau, "timing_spread_median": noise,
        "overhead_ms": ref.overhead_ms, "compute_reference": asdict(ref),
        "sm_count": sm_count, "sm_source": sm_source,
        # THE CARD, IN THE ONLY MACHINE-READABLE ARTEFACT. `Cell` carries no
        # device column, so without this a cells.csv on a shared network volume
        # is unattributable after the pod is gone -- and this study's central
        # comparison is between two cards.
        "card": card,
        "alpha_measured": alpha_hat, "alpha_source": alpha_source,
        # The same ladder's slope in weight-stream units, beside the alpha the
        # gates are scored on, with the rate named. TWO NULL CONDITIONS, not
        # one, and the second was added by the refusal catch on 2026-09-10
        # while this comment still named only the first: (1) no ladder was
        # eligible to source an alpha, which is the same condition that leaves
        # `alpha_measured` null; and (2) a sourcing ladder whose own `w` is
        # blank, which `LadderFit._weight_streams` decides and names -- no
        # memory branch, no model/dtype/rate on the fit, or a `WeightSetRefused`
        # for a geometry this repository will not guess. So `alpha_measured` can
        # carry a number where this key is null, which the one-condition wording
        # said could not happen. `LadderFit.w_note` is where the reason for (2)
        # is written, in the refusal's own words, and it is what the ladder
        # table's "w n/a: ..." line prints.
        "weight_streams_measured": (
            None if alpha_source_bm is None
            or fits[alpha_source_bm].weight_streams is None
            else fits[alpha_source_bm].weight_streams.streams),
        "weight_streams_bandwidth_gbps": bandwidth_gbps,
        "weight_streams_bandwidth_source": (
            bandwidth_source or "NOT STATED by the caller"),
        # "NEVER CROSSES AT THIS ALPHA" IS AN EXPLICIT OUTCOME HERE. `crosses`
        # is a bool and never absent, `crossing_rows` is null when there is no
        # crossing, and `no_crossing_reason` says in words that the cap sits at
        # or below the ridge. Before 2026-09-02 the only record of that state
        # was a null in `crossing_rows_ridge_lo`, which a report generator read
        # as a missing measurement -- and which this file formatted with `:.0f`
        # and crashed on at every alpha above about 0.79, i.e. at every alpha
        # mixtral's measured G=1 ladders (0.95-1.02) actually describe.
        "predictions": {
            str(bm): {"ai_cap": preds_lo[bm].ai_cap,
                      "crossing_rows_ridge_lo": preds_lo[bm].crossing_rows,
                      "crossing_rows_ridge_hi": preds_hi[bm].crossing_rows,
                      "first_compute_tread": preds_lo[bm].first_compute_tread,
                      "crosses": preds_lo[bm].crosses,
                      "crossing_rows": preds_lo[bm].crossing_rows,
                      "no_crossing_reason": no_crossing_reason(preds_lo[bm]),
                      "cap_overstatement": list(cap_overstatement(
                          cfg, bm, (pinned or FIXED)["BLOCK_SIZE_N"], b)),
                      "cap_overstatement_note": cap_note(
                          cfg, bm, (pinned or FIXED)["BLOCK_SIZE_N"], b)}
            for bm in block_sizes},
        "ladder": {str(bm): {"points": list(f.points),
                             "memory_points": f.memory_points,
                             "alpha": f.alpha,
                             "alpha_corrected": alpha_corrected.get(bm),
                             "alpha_upper": f.alpha_upper,
                             # THE SECOND ESTIMATOR, PERSISTED BESIDE THE
                             # FIRST, never in place of it: every published row
                             # was scored on B/(A+B) and stays readable. `w` is
                             # the same slope over a MEASURED stream time
                             # rather than over a fitted level, and the four
                             # fields beside it are the denominator, so a
                             # reader of the file alone can check the division
                             # and can see which rate it is a fraction of.
                             # NULL FOR MORE THAN A MISSING BRANCH, and since
                             # the refusal catch landed on 2026-09-10 that is
                             # three states, not one: no memory branch to take
                             # a slope from, no model/dtype/rate on the fit, or
                             # a `WeightSetRefused` for a geometry this
                             # repository holds no verified shape for.
                             # `LadderFit._weight_streams` decides all three in
                             # one place and returns the reason in the refusal's
                             # own words.
                             "weight_streams_per_tile": (
                                 None if f.weight_streams is None
                                 else f.weight_streams.streams),
                             "weight_stream_ms": (
                                 None if f.weight_streams is None
                                 else f.weight_streams.stream_ms),
                             "weight_set_bytes": (
                                 None if f.weight_streams is None
                                 else f.weight_streams.weight_bytes),
                             "weight_stream_bandwidth_gbps": f.bandwidth_gbps,
                             "weight_stream_bandwidth_source": (
                                 f.bandwidth_source),
                             # THE DIAGNOSTIC. `alpha_upper > 1` is exactly
                             # `D > A`, and this says which rows are in that
                             # state instead of leaving a reader to rediscover
                             # it from an out-of-range alpha_upper. Labelled,
                             # not refused: the four bn_decomposition cells in
                             # this state on 2026-09-10 are four of its six.
                             "fixed_cost_above_intercept": (
                                 f.fixed_cost_above_intercept),
                             "intercept": f.intercept,
                             "overhead_ms": f.overhead_ms,
                             "slope_memory": f.slope_memory,
                             "slope_compute": f.slope_compute,
                             "slope_compute_ref": f.slope_compute_ref,
                             "crosses": f.crosses,
                             "basis": f.basis,
                             # A blank alpha has two causes and they are not
                             # interchangeable. Written per ladder so a reader
                             # of one row -- or a table generator like
                             # `scripts/alpha_surface.py` -- cannot print a
                             # refused reference as a sweep that lacked treads.
                             "identifiable": (f.memory_points
                                              >= MIN_MEMORY_TREADS
                                              and f.alpha is not None),
                             "unidentifiable_reason": _why_not_identifiable(
                                 f, ref),
                             # THE NAMED OUTCOME, beside the older reason
                             # token. `undecided` is True only where the sweep
                             # LOOKED and could not say -- a branch parallel to
                             # the ridge, or treads lost to a DRIFTING clock,
                             # which is what `undecided_drifting_clock` names
                             # since 2026-09-09 -- which is a different report
                             # from "too few treads" and points at a different
                             # next experiment.
                             "outcome": f.outcome,
                             "outcome_reason": f.outcome_reason,
                             "undecided": f.undecided,
                             "branch_start": f.branch_start,
                             "excluded_drifted": f.excluded_drifted,
                             "kept_high_clock": f.kept_high_clock,
                             "kept_low_clock": f.kept_low_clock,
                             "mean_rel_err": f.mean_rel_err}
                   for bm, f in fits.items()},
        "bracketing": asdict(brack),
        # THE DETAIL LINES AND THE PROVENANCE ARE SERIALIZED. They used not to
        # be, and that is how 22 published reports asserted a crossing ratio
        # with nothing in the file to say it was a restatement of a fitted
        # alpha imported from another BLOCK_M. A verdict a reader cannot trace
        # is a verdict a reader cannot check.
        "gates": [{"number": g.number, "claim": g.claim, "kind": g.kind,
                   "basis": g.basis, "verdict": g.verdict,
                   "measured": g.measured, "gate": g.threshold,
                   "detail": list(g.lines), "provenance": g.provenance}
                  for g in gates],
    }
    # ONE PROVENANCE BLOCK, LAST, AND IT STAMPS RATHER THAN MERGES. `stamp`
    # raises `ProvenanceCollision` if any of the five audited top-level keys is
    # already present with a different value, so two provenance blocks can never
    # be layered over one report without someone noticing -- which is why the
    # caller passes `ridge_source` and `bandwidth_source` strings that ALREADY
    # begin with the block's own token (`main` builds them as
    # "<kind>: <prose>"), so the two writers agree and the stamp is a no-op on
    # those two keys rather than a conflict.
    if prov is not None:
        payload = prov.stamp(payload)
    return Report(lines, gates, payload)


def _compute_slope_consistency(fits) -> str:
    """`C ~ BLOCK_M` across settings, which gate 4 leans on.

    A cross-check and not a gate: if the compute branch does not scale with the
    tile height, the scaled `C` gate 4 compares against is wrong, and a reader
    should be told before the verdict rather than after it.
    """
    have = [(bm, f.slope_compute) for bm, f in sorted(fits.items())
            if f.slope_compute]
    if len(have) < 2:
        return ""
    parts = []
    for (bm0, c0), (bm1, c1) in zip(have, have[1:], strict=False):
        parts.append(f"C({bm1})/C({bm0}) = {c1 / c0:.2f}x against "
                     f"{bm1 / bm0:.2f}x predicted")
    return "compute branch should scale with BLOCK_M -- " + "; ".join(parts)


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def find_override():
    """vLLM's own tuning hook, probed rather than assumed.

    `try_get_optimal_moe_config` consults `get_config()` first and a truthy
    value bypasses the tuned file and the default ladder both. The import path
    has moved between versions, so a wrong guess would silently sweep nothing --
    which is precisely the failure gate 0 exists to catch, and this is the first
    line of that defence.
    """
    import importlib
    candidates = ["vllm.model_executor.layers.fused_moe",
                  "vllm.model_executor.layers.fused_moe.fused_moe",
                  "vllm.model_executor.layers.fused_moe.config"]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, name
    raise SystemExit(
        "could not find vLLM's override_config in any of:\n  "
        + "\n  ".join(candidates)
        + "\nCheck the installed vLLM version; try_get_optimal_moe_config reads "
          "it via get_config(), so the hook exists under some name.")


def arm_triton_cache(root: Path, block_m: int) -> Path:
    """Point Triton at a fresh directory for THIS setting, before it compiles.

    Set before the first compile of the setting, which is the first timed call
    at that BLOCK_SIZE_M, because Triton reads these at compile time. Within one
    process each block size is a distinct specialisation and so a distinct cache
    entry anyway; the per-setting directory is what makes "did this setting
    compile anything" a countable question instead of an assumption. A warm
    cache dumping nothing is what cost this project its A100 PTX dump.
    """
    directory = root / f"bm{block_m}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(directory)
    return directory


def count_new(root: Path, seen: set[Path]) -> int:
    fresh = [p for p in root.rglob("*") if p.is_file() and p not in seen]
    seen.update(fresh)
    return len(fresh)


class RetiredInstrument(SystemExit):
    """`time_call` was called. It no longer times anything, by design.

    A `SystemExit`, AND THAT IS THE WHOLE MECHANISM. All four sibling arms
    time inside a per-cell `except Exception` (`scripts/bm128_roofline.py:1564`,
    `bm128_depth.py:1620`, `bn_decomposition.py:2303`,
    `occupancy_vs_swizzle.py:1309`), which turns any per-cell failure into a
    `status="failed"` row and moves on. As a `RuntimeError` this refusal was
    swallowed by every one of them: the arm compiled and ran its whole grid,
    recorded a failed row per cell and only then reached its gates, so
    `bm128_roofline` -- the arm scheduled to settle the BLOCK_M=128 question --
    would have burned a pod allocation to produce no usable cell.

    `SystemExit` derives from `BaseException`, not `Exception`, so it is outside
    every one of those per-cell handlers and the first cell any arm tries to
    time stops that arm with the fix in the message. It is `SystemExit` rather
    than a bare `BaseException` for the delivery: an arm's top-level
    `except Exception` does not swallow it, the interpreter prints nothing and
    exits with `code`, and a caller that wants to handle it can name it. A bare
    `BaseException` escaped the top-level handlers too, which turned a refusal
    with a remedy in its message into a traceback and an exit code that means
    something else. That is the same delivery defect `BandwidthUnavailable` was
    fixed for, one exception over.
    """

    code = exit_codes.REFUSED


def time_call(fn, warmup: int, iters: int):
    """RETIRED 2026-09-02. Raises `RetiredInstrument`; call `time_kernel`.

    WHAT THIS USED TO DO AND WHY IT MAY NOT DO IT AGAIN. It created a fresh
    CUDA event pair inside the loop, recorded the start event on a stream it had
    just synchronised, ran one call, and synchronised again -- once per
    iteration, with no L2 flush and no clock read. The ROOF every number it
    produced is scored against was measured by `moe.bench.timing.time_eager`,
    which is queue-deep with pre-primed events. Two instruments, one comparison.
    The audit bounded the host prefix this one let inside the measured interval
    at about 0.18 ms per fused_experts call on the H200 pod and 0.30 ms on the
    A100 pod, which is a bias in the fitted alpha of 8-16% at the smallest
    ladder cells, DIFFERENT PER CARD, and of the same order as the cross-card
    effect this study registered (+0.0117, MDE 0.030).

    WHY A REFUSAL AND NOT A WRAPPER. A wrapper would have to invent the two
    things `time_kernel` needs and this signature cannot supply: `warmup` here
    is a COUNT of calls and the instrument warms for a DURATION of delivered GPU
    load, and there is no honest conversion between them; and a caller passing a
    fixed `iters` has already decided a sample size that the instrument sizes
    from the warmup's own queue-deep per-call time. Guessing either would put a
    number in a report that no instrument produced.

    WHY THE NAME SURVIVES. `scripts/bm128_roofline.py`,
    `scripts/bm128_depth.py`, `scripts/bn_decomposition.py` and
    `scripts/occupancy_vs_swizzle.py` load this module by path and check
    `hasattr(module, "time_call")` before they will run at all. Deleting the
    symbol would abort those four at import with "no longer exports time_call",
    which says nothing about what actually changed. They keep importing, and the
    first cell any of them tries to time raises this, with the fix in the
    message.

    WHAT "RAISES THIS" ACTUALLY DOES TO THEM, corrected 2026-09-02. All four
    time inside a per-cell `except Exception`, so while `RetiredInstrument` was
    a `RuntimeError` this refusal was caught per cell: the arm went on to
    compile and run every setting in its grid, wrote `status="failed"` and
    `ms_p50=0.0` for each and printed one FAILED line per cell before its gates
    saw an empty sweep. Nothing fabricated a number -- all four filter on
    `status == "ok" and ms_p50 > 0` and their non-vacuity gates refuse to PASS
    on zero cells -- but the arm burned its whole allocation to learn one fact
    it could have learned at cell 1. `RetiredInstrument` is now a
    `BaseException`, outside those handlers, and the arm stops at the first
    timed cell as this docstring always claimed. Migrating them onto
    `time_kernel` is their own phase; until then their numbers were never
    comparable with the roof, which is the finding, not a regression introduced
    here.
    """
    raise RetiredInstrument(
        "time_call has been retired: it timed with per-iteration synchronises "
        "and events created inside the loop, which is NOT the instrument the "
        "compute roof was measured with, and it exposed 0.18-0.30 ms of host "
        "enqueue time per call inside the measured interval (a per-card bias of "
        "8-16% in the fitted alpha at the smallest ladder cells).\n"
        "    Use moe.bench.timing.time_kernel(fn, warmup_ms=..., "
        "target_ms=..., trials=..., l2_flush=..., reference_clock_mhz=...), "
        "which is queue-deep, flushes L2 per iteration, samples the SM clock "
        "under load and returns a KernelTiming carrying its own instrument "
        "name.\n"
        f"    warmup={warmup} is a CALL COUNT and time_kernel warms for a "
        f"DURATION of delivered GPU load; iters={iters} is a sample size "
        "time_kernel derives from that warmup. Neither converts, which is why "
        "this refuses instead of wrapping.")


def reference_clock_mhz(gpu_name: str = "") -> tuple[float | None, str]:
    """The SM clock this card's roof was measured at, and where it came from.

    `timing.clock_flags` needs a reference before it can say anything about
    LEVEL: a level is relative to something, and `time_kernel` will not invent
    the something. That something is the clock the CALIBRATION ran its GEMM at,
    because the roof every cell here is scored against is that GEMM's number.

    Three places, in falling order of directness, all inside the card's own
    `measured_*.yaml`: the full `LoadedClock` record's median (samples taken
    WHILE the GEMM ran), the scalar `gemm_clock_mhz` every older consumer reads,
    and the compute settle's final plateau. `calibrate.clock_established` exists
    because those three have disagreed: eleven committed calibrations of one
    H200 recorded 1485-1935 MHz for the scalar while their own settle histories
    sat at 1455-1515. The order here prefers the number measured under the load
    that set the roof, and the returned string says which one was used, so a
    LEVEL exclusion can always be traced to a field in a file.

    Returns `(None, reason)` when the card has no calibration. That is not a
    failure: `clock_level_ok` is then None on every cell, which means "not
    determined", and `ladder_treads` excludes nothing. A guessed reference would
    exclude real treads.
    """
    try:
        from moe.bench.roofline import current_gpu_name
    except Exception as exc:                            # noqa: BLE001
        return None, f"roofline unavailable: {type(exc).__name__}: {exc}"
    name = gpu_name or current_gpu_name()
    detail = _measured_detail(name)
    if not detail:
        return None, (f"no calibration yaml for {name or 'this device'}, so the "
                      "clock the roof was measured at is not known here")
    median = (detail.get("gemm_clock") or {}).get("median_mhz")
    if median:
        return float(median), (f"{name}: median of the samples taken while the "
                               "calibration's dense GEMM ran")
    scalar = detail.get("gemm_clock_mhz")
    if scalar:
        return float(scalar), (f"{name}: gemm_clock_mhz, the scalar the "
                               "calibration published for its dense GEMM")
    plateau = (detail.get("settle") or {}).get("final_mhz")
    if plateau:
        return float(plateau), (f"{name}: the compute settle's final plateau; "
                                "the calibration recorded no GEMM clock")
    return None, (f"{name}: the calibration carries no clock at all, so LEVEL "
                  "cannot be scored against it")


def _make_call(fused_experts, x, weights, w, ids, kw):
    """Bind explicitly rather than closing over the loop variables (ruff B023)."""
    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=w, topk_ids=ids, **kw)
    return call


def balanced_ids(cfg, tokens: int, device: str):
    """Top-k ids whose per-expert histogram is EXACTLY `T k / E`.

    Not `sample_topk_ids(uniform)`. Sampled uniform routing puts about 15 rows
    of spread on a mean of 256 at mixtral T=1024, which smears every tile step
    across 60 tokens and makes gate 1 a matter of opinion. `realize_counts`
    builds the histogram exactly, so rows per expert is an integer this script
    knew before the pod was rented and every tile step lands where the grid was
    built to look for it.
    """
    from moe.routing.distributions import realize_counts
    per = tokens * cfg.top_k // cfg.num_experts
    if per * cfg.num_experts != tokens * cfg.top_k:
        raise ValueError(f"T={tokens} does not divide evenly over "
                         f"E={cfg.num_experts} at k={cfg.top_k}")
    return realize_counts([per] * cfg.num_experts, tokens, cfg.top_k,
                          device=device)


def run_sweep(args, cfg, grid, block_sizes, csv_path: Path, cache_root: Path,
              b: int, pinned: dict, prov=None) -> tuple[list[Cell],
                                                        dict[int, int],
                                                        dict[int, int]]:
    """The metered part. Appends every cell as it lands, so aborting keeps it.

    BLOCK_SIZE_M IS THE OUTER LOOP, which is a trade. It makes the Triton cache
    attribution exact -- everything that compiles while a setting runs belongs
    to that setting -- and it makes a resumed run finish a setting before
    starting the next. It also means gate 2's comparison of four settings at the
    same rows-per-expert spans the whole sweep rather than a few seconds, so it
    carries whatever the clocks did in between. The sweep is about a minute of
    GPU on mixtral and the effect gate 2 is looking for is a factor of three, so
    that is a trade worth making; on a card that throttles it would not be.
    """
    import torch

    from moe.bench import timing
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    # THE CLOCK THE ROOF WAS MEASURED AT, resolved once, before any cell. It is
    # the reference the LEVEL flag compares against; without it every cell's
    # `clock_level_ok` is None, which means "not determined" and excludes
    # nothing. Resolved here rather than per cell so the whole arm is scored
    # against one number and a mid-sweep yaml rewrite cannot move it.
    reference_clock, clock_source = reference_clock_mhz()
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every cell's clock LEVEL "
                  "verdict will be None and no cell can be excluded for it"))

    # BEFORE vLLM is imported. Triton may snapshot this variable at import in
    # some versions, and a warm cache dumps and compiles nothing -- the bug that
    # cost this project its A100 PTX dump. Pointing it at this run's own
    # directory first guarantees freshness relative to previous runs whatever
    # the per-setting redirect below manages, and the artefact count is taken
    # over the whole root so the assay works either way.
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    override_config, where = find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs

    print(f"override hook: {where}.override_config")
    print(f"triton cache: {cache_root} (fresh for this run)")
    done, cells = read_cells(csv_path)
    seen_files: set[Path] = set()
    compiles: dict[int, int] = {}
    executed: dict[int, int] = {}
    sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count

    inputs: dict[int, tuple] = {}
    for bm in block_sizes:
        arm_triton_cache(cache_root, bm)
        count_new(cache_root, seen_files)
        compiles.setdefault(bm, 0)
        executed.setdefault(bm, 0)
        for rows in grid:
            tokens = tokens_for_rows(cfg, rows)
            if (bm, tokens) in done:
                continue
            if tokens not in inputs:
                spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                                 routing=RoutingSpec("uniform", 0.0),
                                 seed=args.seed)
                x, weights = make_inputs(spec, device="cuda")
                ids = balanced_ids(cfg, tokens, "cuda")
                w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                               device="cuda")
                kw = vllm_call_kwargs(spec)
                kw["activation"] = MoEActivation(kw["activation"])
                inputs = {tokens: (x, weights, ids, w, kw)}   # one cell live at a time
            x, weights, ids, w, kw = inputs[tokens]
            executed[bm] += 1
            conf = dict(pinned, BLOCK_SIZE_M=bm)
            call = _make_call(fused_experts, x, weights, w, ids, kw)
            try:
                with override_config(conf):
                    call()
                    torch.cuda.synchronize()
                    compiles[bm] += count_new(cache_root, seen_files)
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=reference_clock)
                cell = make_cell(cfg, rows, bm, t.ms_p50, sm_count=sm_count,
                                 block_n=pinned["BLOCK_SIZE_N"],
                                 ms_min=t.ms_min, ms_stdev=t.ms_std,
                                 iters=t.iters, instrument=t.instrument,
                                 warmup_ms=t.warmup_ms, trials=t.trials,
                                 sm_clock_load_mhz=t.sm_clock_load_mhz,
                                 clock_level_ok=t.clock_level_ok,
                                 clock_drift_ok=t.clock_drift_ok,
                                 # THE SIDE TRAVELS WITH THE VERDICT. A failed
                                 # LEVEL without it is refused by make_cell,
                                 # because read as LOW it drops every boosted
                                 # tread.
                                 clock_level_side=t.clock_level_side,
                                 l2_flush=t.l2_flush)
                if t.clock_level_ok is False or t.host_bound:
                    print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
                    if t.clock_level_side == LEVEL_HIGH:
                        print("  ^ LEVEL HIGH: kept in every fit; its fraction "
                              "of the fixed roof is not comparable")
            except timing.TimingRefused:
                # THE INSTRUMENT'S OWN REFUSAL IS NOT ONE CELL'S ERROR. TimingRefused
                # subclasses RuntimeError, so the handler below would file "no CUDA",
                # "trials=0" or "warmup too short" as a failed cell and move on: the
                # arm then walks its whole grid writing zeroed rows and exits DONE.
                # Reproduced on the driver path on 2026-09-03; same door here.
                raise
            except Exception as exc:                    # noqa: BLE001
                cell = make_cell(cfg, rows, bm, 0.0, sm_count=sm_count,
                                 block_n=pinned["BLOCK_SIZE_N"], status="failed",
                                 detail=f"{type(exc).__name__}: {exc}")
                print(f"  BM={bm} T={tokens} FAILED  {cell.detail}")
                if "shared memory" in str(exc).lower():
                    print(f"  ^ re-run the WHOLE sweep with --num-stages "
                          f"{max(1, pinned['num_stages'] - 1)}. Dropping stages "
                          "for one setting alone would unpin the thing this "
                          "sweep holds fixed.")
            cells.append(cell)
            append_cell(csv_path, cell, prov)
            print(f"  BM={bm:3d} T={cell.tokens:6d} r={rows:6d} "
                  f"n={cell.tiles_per_expert:3d} waves {cell.waves_up:7.1f} "
                  f"{cell.ms_p50:9.4f} ms  {cell.useful_tflops:7.1f} TFLOP/s")
    return cells, compiles, executed


# --------------------------------------------------------------------------
# Persistence.
# --------------------------------------------------------------------------

CSV_FIELDS = [f for f in Cell.__dataclass_fields__]

#: Provenance columns appended to every cells.csv row, after the measurement
#: columns. `Provenance.as_columns` prefixes every one with `prov_`, so a
#: provenance column can never collide with a measurement column that happens
#: to share its name (`iters`, `gpu_name`). Written per ROW rather than once per
#: file because a resumed cells.csv is written by two processes on two days and
#: possibly two commits, and a header cannot say that.
PROVENANCE_COLUMNS = sorted(
    PV.Provenance().as_columns())


def append_cell(path: Path, cell: Cell, prov=None) -> None:
    """One row, flushed. An abort costs the cell in flight and nothing else.

    SIGNATURE EXTENDED, never narrowed: `prov` is optional and a caller that
    omits it writes the measurement columns alone, which is what the four
    sibling scripts and the test suite do. When it is given, the row also
    carries the commit, the card, the instrument and the two ruler sources that
    made it, so a cells.csv on a shared network volume stays attributable after
    the pod is gone.
    """
    new = not path.exists()
    fields = CSV_FIELDS + (PROVENANCE_COLUMNS if prov is not None else [])
    row = asdict(cell)
    if prov is not None:
        row.update(prov.as_columns())
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _cell_value(type_name: str, raw: str):
    """One CSV field back into its declared type.

    THE OPTIONAL FORMS ARE HANDLED SEPARATELY AND THE EMPTY STRING IS None, not
    zero and not False. `sm_clock_load_mhz`, `clock_level_ok` and
    `clock_drift_ok` are tri-state by design: a cell timed on a host without
    NVML has no clock, and reading that blank back as `False` would mark it
    excluded while reading it as `True` would mark it comparable. Both are
    claims the row does not make. `clock_level_side` is a plain string whose
    blank IS a value ("inside the band, or not determined"). A cells.csv
    written before the column existed reads back with it blank on every row,
    which is correct for every row whose verdict is not False and is REFUSED
    by `Cell` for a failed one, because that row's side cannot be recovered
    from the file and reading it either way would be a default.
    """
    optional = type_name.replace(" ", "").endswith("|None")
    base = type_name.replace(" ", "").removesuffix("|None")
    if optional and raw == "":
        return None
    if base == "int":
        return int(float(raw or 0))
    if base == "float":
        return float(raw or 0.0)
    if base == "bool":
        return raw == "True"
    return raw


def read_cells(path: Path) -> tuple[set[tuple[int, int]], list[Cell]]:
    """Cells already measured, so a re-run resumes rather than repeats."""
    if not path.exists():
        return set(), []
    cells: list[Cell] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            kw = {name: _cell_value(str(f.type), row.get(name, "") or "")
                  for name, f in Cell.__dataclass_fields__.items()}
            cells.append(Cell(**kw))
    # Only the cells that SUCCEEDED count as done. A cell that failed is
    # retried on the next run, because the common failure here is a setting
    # that ran out of shared memory or a pod that lost its device, and both are
    # states a re-run can leave behind. A failure that is real fails again in
    # milliseconds.
    return {(c.block_m, c.tokens) for c in cells if c.status == "ok"}, cells


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    Same order `scripts/run_all.sh` resolves it in, so this experiment lands
    beside every other arm on the volume that outlives the pod.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


#: THE ARGPARSE DESTINATIONS THAT MAY BE ABSENT FROM `default_run_id`, each
#: with the reason it cannot change a measured value. Every other destination
#: this parser defines MUST move the derived id, and
#: `tests/test_block_m_crossing_sweep.py` checks that by enumerating the parser
#: rather than by reading a list somebody wrote out by hand.
#:
#: WHY A TABLE AND NOT A SENTENCE. `default_run_id`'s docstring twice announced
#: that the last omission had been closed while a knob was still outside the
#: key: first the planted world, then `--sm-count` and `--capability`, the
#: second time in the same paragraph that cited the commit about headers
#: describing a state the code is not in. Prose cannot be executed. This table
#: can, and a flag added tomorrow either lands in the key or is classified here
#: on purpose, with no third state where it is silently forgotten.
ID_EXEMPT_DESTS = {
    "help": "argparse's own flag; it names nothing about a run",
    "ridge": "re-analyses a set of cells rather than changing one, so two "
             "analyses of one sweep belong in one directory and must share "
             "cells.csv or the resume path re-measures identical cells",
    "ridge_band": "an interval around --ridge, analysis for the same reason",
    "alpha": "the PREDICTION the measured cells are scored against; it decides "
             "no cell",
    "bandwidth_gbps": "the roof's other half, analysis for the same reason",
    "run_id": "IS the id; a key on itself is not a key. `resolve_run_id` is "
              "what stops a supplied one from naming a planted run after a "
              "metered directory",
    "out": "where the directory is rooted, not what is inside it",
    "dry_run": "measures nothing, writes nothing and REFUSES, so it has no "
               "directory to collide with",
    "fail_on_gate": "RETIRED, accepted and ignored; it never reached a cell "
                    "and no longer reaches the exit code either",
}


def default_run_id(args, card: str) -> str:
    """Derived from the arguments, so "the same experiment" resumes itself.

    A random id would make every re-run a new directory and turn the resume
    path into dead code the first time anyone used it.

    EVERY pinned knob has to be in this key. It used to omit GROUP_SIZE_M,
    BLOCK_SIZE_N and num_stages, which was safe only while all three were
    unreachable constants. The moment --group-m existed it became a silent
    overwrite: a G=16 run would derive the SAME id as the G=1 run, resume into
    its directory, find every cell already on disk, skip all of them, and report
    G=1's timings under a G=16 heading. Nothing would have looked wrong, because
    the report prints `pinned` from argv rather than from the cells it read. The
    knobs are in the visible name too, so two runs are distinguishable in `ls`
    and not only by a hash nobody can invert.

    THE SAME OMISSION SURVIVED IN FOUR MORE FIELDS UNTIL 2026-09-02, and the
    first of them had already been committed:

      * THE CARD. It is not swept by this script, it is swept by the operator
        moving to another pod, and `results_root()` prefers `$MOE_RESULTS_DIR`
        then `/workspace/results`, the network volume the runbook uses BECAUSE
        it outlives the pod. Two cards therefore derived one id, and the proof
        is in the repo: `results/published/2026-09-01-nvidia_h200-cross-card-s3`
        and `results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`
        both contain `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json`, for
        `sm_count` 132 and 108. `read_cells` keys resume on `(block_m, tokens)`
        and `Cell` carries no device column, so nothing downstream would notice:
        the second card finds all cells present, skips them, spends no GPU time,
        and prints the first card's timings scored against its own ridge --
        145.7 against 162.8. `scripts/replicate_noise_floor.py:run_id_for`
        documents this defect and works around it locally; fixing it here is
        what stops the next caller inheriting it.
      * `--iters`, `--warmup` and `--cell-budget-ms`. These are not analysis
        knobs: they set the measured milliseconds of every cell. A `--iters 200`
        re-run after `--iters 50` landed in the same directory and printed the
        50-iteration numbers under the 200-iteration label, invisibly, because
        the report renders `pinned` and the arguments from argv rather than from
        the cells it read.

    `--ridge`, `--ridge-band`, `--alpha` and `--bandwidth-gbps` stay OUT of the
    key on purpose: they re-analyse a set of cells rather than change one, so
    two analyses of one sweep belong in one directory.

    THE ID IS BUILT BY `moe.bench.provenance.run_id` AS OF 2026-09-02, not by a
    private hash here. That function raises `NoCard` on a missing card and
    `UnresolvedKnob` on a None or empty value, hashes the knobs in sorted order
    so the id does not depend on the order they were named in, and puts the card
    slug at the FRONT where `ls` shows it. Three scripts had each re-implemented
    a subset of this and each had left a different knob out; the collisions that
    cost are in that module's docstring. Two knobs are new to the key here and
    both set the measured milliseconds: `warmup` (now a duration of sustained
    load) and `l2_flush`, which changes whether every timed iteration starts
    with a cold L2.

    THE PLANTED WORLD IS PART OF THE KEY, AND IT WAS THE LAST OF THE SELF-TEST
    OMISSIONS. It was not the last omission, and the sentence that said so is
    the reason the paragraph below this one exists. Until
    2026-09-02 `--self-test`, `--self-test-noise` and `--self-test-world` named
    nothing: `--self-test 0.2`, `--self-test 0.9` and
    `--self-test 0.2 --self-test-noise 0.5` all derived one id, so each planted
    world overwrote the last and a self-test could not be compared with the
    self-test before it. On a pod it was worse than that. The card came from
    `detect_card_slug`, which returns the ATTACHED device whether or not
    anything was measured, so one free `--self-test 0.10` on the metered machine
    landed in the metered run's directory and the unconditional
    `report.json` write replaced the paid arm's only machine-readable artefact
    with a synthetic one carrying the retracted alpha. `scripts/tile_cap_test.py`
    documented this exact failure and keyed on `planted`/`plantnoise`; this file
    did not get the same treatment until now.

    THE PLANTED WORLD IS ONE KNOB, `plant`, AND ONLY A PLANTED RUN CARRIES IT.
    The first fix put three of them, `planted`/`plantnoise`/`plantworld`, in the
    key of EVERY run, which cost the metered runs the thing the visible name
    exists for. `run_id` renders knobs in name order and truncates the visible
    part at 96 characters, so on a measurement the three contributed the
    constant 19 characters `plantedmeasured-pla` and evicted `probes`, `r` and
    `routing` off the end: two pod runs at different `--r-max` stopped being
    distinguishable in `ls` and differed only in the hash. They buy nothing
    there, because a planted run is already the one whose card slug is
    `synthetic`. So the plant travels on the planted branch alone, as a single
    token, and a measured run's id is byte-identical to what it was before any
    of this. `plant_tag` writes that token and says what it costs.

    A PLANTED RUN'S CARD IS `SYNTHETIC_CARD_SLUG`, and that is also the
    `synthetic-` prefix: `run_id` renders the card slug FIRST, so the directory
    already begins `synthetic-` and emitting the word twice would be two names
    for one fact. The attached card stays in the key as `synthhost`, because
    `resolve_ridge` and `resolve_bandwidth` read that card's calibration even
    under `--self-test`, so the same planted alpha is a different world on two
    pods and the two must not share a directory either. Its NAME is chosen to
    sort after `plant`: `run_id` truncates the visible part at 96 characters in
    name order, and what a reader needs in `ls` is which world was planted, not
    which machine generated it.

    `--self-test-world` IMPLIES `--self-test` HERE THE SAME WAY `main` IMPLIES
    IT, so the id does not depend on which of the two flags turned the run
    synthetic. A caller that reaches this function before `main` has applied
    that rule would otherwise get a measured run's id for a planted run.
        `--sm-count` AND `--capability` WERE THE TWO STILL MISSING WHEN THE
    PARAGRAPH ABOVE CLAIMED THE KEY WAS CLOSED, and both decide a measured
    value. `--sm-count` is baked into every persisted row through `waves`, so a
    run that asserts 108 on a 132-SM card writes different `waves` and different
    `tail_fraction` for identical timings. `--capability` PRUNES the tile set:
    `tile_resource_plan` refuses a block size whose pipeline will not fit the
    stated shared memory, so `--capability 8.0` measures three tiles where the
    unpruned run measures four, and `cells.csv` is resumed by run id. Both
    derived an id byte-identical to no flag at all until 2026-09-02.

    THEY ARE KEYED ONLY WHEN GIVEN, which is the bargain `plant` makes and for
    the same reason. Both default to "ask the driver", the driver's answer is a
    function of the card, and the card is already the first component of the id.
    Writing `capdriver-smdriver` into every metered run's name would spend 19
    characters of a 96-character visible budget on a constant, which is exactly
    the regression the three `plant` knobs caused two paragraphs up. An override
    is a fact only the overriding run has, so only that run pays for it. The
    cost of the choice is a FALSE SPLIT and never a collision: `--sm-count 132`
    on a 132-SM card is a second directory for the same measurement, and that is
    a re-measure rather than a report written over another report.

    HOW THE KEY IS KEPT COMPLETE, WHICH IS NOT THE SAME AS CLAIMING IT IS.
    This docstring has now twice said the last omission was closed while a knob
    was still outside, so the completeness is no longer asserted in prose.
    `ID_EXEMPT_DESTS` names every argparse destination that may be absent and
    why, and `tests/test_block_m_crossing_sweep.py` ENUMERATES THIS PARSER,
    perturbs each remaining destination away from its default, and requires the
    derived id to move. A flag added tomorrow is therefore either in the key or
    classified as exempt on purpose; there is no third state in which it is
    quietly forgotten, which is the state `--sm-count` and `--capability` were
    in for as long as anyone had been reading this docstring.
    """
    planted = args.self_test
    if planted is None and args.self_test_world:
        planted = ALPHA
    swept = dict(
        model=args.model,
        dtype=args.dtype,
        tiles=tuple(int(v) for v in args.tiles.split(",")),
        g=args.group_m,
        n=args.block_n,
        stages=args.num_stages,
        r=args.r_max,
        rowstep=args.row_step,
        probes=args.step_probes,
        seed=args.seed,
        iters=args.iters,
        warmup=args.warmup,
        budget=args.cell_budget_ms,
        flush=not args.no_l2_flush,
        trials=args.trials,
        # Not a swept knob today and in the key anyway: `balanced_ids` builds an
        # EXACT per-expert histogram and `run_sweep` never asks for anything
        # else, so the day a sampled-routing arm appears its cells must not land
        # in a balanced arm's directory and be skipped as already measured.
        routing="balanced",
    )
    # CONDITIONAL, AND THE DOCSTRING SAYS WHY AT LENGTH. Both flags default to
    # "ask the driver", whose answer is a function of the card the id already
    # leads with, so an unset knob names nothing a metered run needs and an
    # override is a fact only the overriding run has. `--capability ""` and
    # `--sm-count 0` are also exactly the values `provenance.run_id` REFUSES as
    # unresolved, so passing them through would raise rather than name anything.
    # `sm` sorts after `routing` and so falls off the far side of the
    # 96-character visible cut on the default grid, where `cap` does not. That
    # costs legibility and never separation: the HASH is over the full key, so
    # two --sm-count values are always two directories whatever `ls` shows.
    if args.sm_count:
        swept["sm"] = int(args.sm_count)
    if args.capability:
        swept["cap"] = str(args.capability)
    if planted is None:
        # No plant knob on this branch, deliberately: see the docstring. A
        # measured run's id must not pay a character of its visible name for a
        # fact that is already carried by the card slug of the runs it separates
        # it from.
        return PV.run_id(card=card, **swept)
    # `card_slug` here and not below, so an empty card still raises `NoCard`
    # rather than `UnresolvedKnob`: the caller's mistake is the same one either
    # way and it should get the same sentence.
    return PV.run_id(card=SYNTHETIC_CARD_SLUG, synthhost=PV.card_slug(card),
                     plant=plant_tag(planted, args.self_test_noise,
                                     args.self_test_world),
                     **swept)


#: What every planted directory begins with. `provenance.run_id` renders the
#: card slug FIRST and a planted run's card slug is `SYNTHETIC_CARD_SLUG`, so
#: this prefix is not a second naming rule, it is that one read back.
SYNTHETIC_DIR_PREFIX = SYNTHETIC_CARD_SLUG + "-"


class SuppliedRunIdIsNotPlanted(RuntimeError):
    """A planted run was handed a name that does not say it was planted.

    Raised rather than rewritten. Silently prefixing an operator's own name
    would make the directory the sweep prints differ from the one its caller
    computed, which is how `replicate_noise_floor` would then fail to find a
    report it did generate; refusing says so before anything runs.
    """


def resolve_run_id(args, card: str) -> str:
    """The directory this run writes under, `--run-id` included.

    THE DEFECT THIS CLOSES, AND WHY `default_run_id` COULD NOT CLOSE IT. The
    fix for W1 put a planted run under `SYNTHETIC_CARD_SLUG`, so a DERIVED name
    always begins `synthetic-` and can never be a metered run's. A SUPPLIED
    `--run-id` bypasses that function entirely, and the module header went on
    claiming that every planted run writes under a `synthetic-` directory as if
    it did not.

    IT IS NOT A HYPOTHETICAL OPERATOR. The one in-repo automated caller that
    runs this script's self-test on a pod is exactly the caller that supplies a
    name: `scripts/replicate_noise_floor.py`'s `Arm.sweep_argv` ALWAYS emits
    `--run-id <run_id>`, and `--rehearse` appends `--self-test`. A rehearsal
    replicate's `report.json` path was therefore byte-identical to the paid
    replicate's for the same arm, index and `--out`, so a free plumbing test
    on the metered machine replaced the paid arm's only machine-readable
    artefact with a synthetic one, detectable afterwards only by reading
    `provenance.instrument` out of it.

    REFUSED, AND NOT PREFIXED. The refusal names the second way out, which is
    to supply a name that already begins `synthetic-`: the caller then still
    chooses the directory, and the caller's own idea of where the report landed
    still matches the sweep's. Rewriting the name here instead would have
    closed the overwrite by opening a `no report.json at <path>` on every
    rehearsal replicate, which is this rebuild's recurring error rather than a
    fix for it. `scripts/replicate_noise_floor.py:run_replicate` takes the
    second way out and prefixes its own id when it rehearses.

    A MEASURED RUN IS UNTOUCHED, EXCEPT FOR THE HOLE THE REFUSAL ITSELF OPENS.
    `--run-id` beside no plant is an operator naming an experiment, which is
    what the flag is for. But the refusal above hands out a name beginning
    `synthetic-` and tells someone to use it, and the next command that name
    gets pasted into may be the metered one, at which point a PAID run writes
    into a directory whose name says nothing was measured. Every reader of this
    corpus, `ls` included, treats that prefix as "generated, not measured", so
    the mirror is refused too. Both directions are the same rule: the prefix
    means planted, and a run whose directory disagrees with what it did is the
    defect W1 is about, whichever way round it points.
    """
    if not args.run_id:
        return default_run_id(args, card)
    planted = args.self_test is not None or bool(args.self_test_world)
    if not planted and args.run_id.startswith(SYNTHETIC_DIR_PREFIX):
        raise SuppliedRunIdIsNotPlanted(
            f"--run-id {args.run_id!r} begins {SYNTHETIC_DIR_PREFIX!r} and this "
            "run plants nothing, so it is about to MEASURE into a directory "
            "whose name says it did not.\n"
            f"    {SYNTHETIC_DIR_PREFIX!r} is what `default_run_id` gives a "
            "planted run and what every reader of this corpus takes to mean "
            "generated rather than measured, so a paid arm under that name is "
            "a report nobody will quote and a directory a self-test may later "
            "resume into.\n"
            "    Drop the prefix, or add --self-test if this run was meant to "
            "plant one.")
    if planted and not args.run_id.startswith(SYNTHETIC_DIR_PREFIX):
        raise SuppliedRunIdIsNotPlanted(
            f"--run-id {args.run_id!r} was supplied beside a planted run and it "
            f"does not begin {SYNTHETIC_DIR_PREFIX!r}.\n"
            "    A DERIVED id cannot reach this state: `default_run_id` gives a "
            f"planted run the card slug {SYNTHETIC_CARD_SLUG!r} and `run_id` "
            "renders the card first. A supplied one bypasses that function, and "
            "the one automated caller that runs this script's --self-test is the "
            "caller that always supplies one "
            "(`scripts/replicate_noise_floor.py`: `Arm.sweep_argv` emits "
            "--run-id for every replicate and --rehearse appends --self-test), "
            "so a rehearsal writes its synthetic report.json at the paid "
            "replicate's exact path.\n"
            "    Either drop --run-id and let the plant name itself, or supply "
            f"{SYNTHETIC_DIR_PREFIX + args.run_id!r} and keep the plant out of "
            "the metered run's directory.")
    return args.run_id


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

#: One tread of the null tile's ladder is planted below the roof's clock, at
#: 0.7x the reference: the steady-low state a dense tile holds on a power-capped
#: card. Since 2026-09-09 it must be KEPT by `ladder_treads`, counted on the
#: ladder row as kept LOW, and the ladder must come out exactly as the clean
#: world's does. It was an EXCLUSION world until that date, which is the rule
#: that removed bm128_depth's entire BM=128 subject from measurability.
LOW_CLOCK_WORLD = "low-clock"
#: One tread of the null tile's ladder is planted ABOVE the band around the
#: roof's clock, at `H200_BOOST_RATIO` times the reference: the boosted
#: memory-shaped state. It must be KEPT by `ladder_treads`, counted on the
#: ladder row as kept, and the ladder must come out exactly as the clean
#: world's does. This is the world the 2026-09-08 fix is for: before it,
#: `clock_level_ok is False` was read as "ran cold" whatever the side said, so
#: these boosted treads were excluded and the same cells landed on the token
#: that rule produced, `UNDECIDED_LOW_CLOCK`. That token no longer exists and
#: no LEVEL verdict can produce an exclusion outcome at all.
HIGH_CLOCK_WORLD = "high-clock"
#: One tread of the null tile's ladder is planted with `clock_drift_ok` False:
#: the clock MOVED across its own trials, so its median is a blend of two
#: operating points. It must be excluded by `ladder_treads`, counted on the
#: ladder row, and if that takes the ladder under `MIN_MEMORY_TREADS` the
#: outcome must be `UNDECIDED_DRIFTING_CLOCK`. This is the exclusion world; it
#: replaced `low-clock` in that role on 2026-09-09.
DRIFT_WORLD = "drifting-clock"
#: The subject tile's memory branch is planted ON the compute branch, which is
#: the regime `PARALLEL_BRANCH_TOLERANCE` refuses to decide. The outcome must be
#: `UNDECIDED_PARALLEL_BRANCH` with its reason, not a blank and not an import.
PARALLEL_WORLD = "parallel-branch"
SELF_TEST_WORLDS = (LOW_CLOCK_WORLD, HIGH_CLOCK_WORLD, DRIFT_WORLD,
                    PARALLEL_WORLD)

#: How each planted world spells itself INSIDE A RUN ID and nowhere else. The
#: flag values keep their hyphens, which `run_id` would render as underscores
#: and which cost characters the visible name does not have. Distinct by
#: construction and covered for every world in `SELF_TEST_WORLDS`; both
#: properties are asserted by a test, because a world that fell out of this map
#: would either collide with another world's directory or raise mid-run.
WORLD_ID_TAGS = {LOW_CLOCK_WORLD: "lowclock", HIGH_CLOCK_WORLD: "highclock",
                 DRIFT_WORLD: "driftclock", PARALLEL_WORLD: "parallel"}


def plant_tag(alpha: float, noise: float, world: str) -> str:
    """The whole planted world as ONE run-id knob value: `0.9n0.5wparallel`.

    WHY ONE TOKEN AND NOT THREE KNOBS. `run_id` sorts knobs by name and cuts the
    visible part of the id at 96 characters, and 75 of them are spent before
    anything starting with `p` is reached. Three knobs spelled
    `planted0.9-plantnoise0.5-plantworldparallel_branch` and the cut landed
    inside the SECOND one, so `--self-test 0.2` and
    `--self-test 0.2 --self-test-noise 0.5` wrote two directories whose names
    differed only in the trailing hash and `--self-test-world` never appeared in
    a name at all. One token spends the name on the three facts that separate
    one planted world from another instead of on repeating the word `plant`.

    WHAT IT OMITS AND WHAT THAT MEANS. A zero noise and an empty world are left
    out rather than written as `n0` and `wnone`, so the common
    `--self-test 0.558` reads `plant0.558`; the encoding is still injective,
    because the alpha never contains `n` or `w`. The tail can still clip: the
    longest combination, an alpha and a noise and a world, spends 24 characters
    against the 21 left after the default grid's knobs and loses two off the end
    of the world tag, which stays readable (`wparal`) and stays distinct from
    every other world's: the four tags differ inside their first six characters
    (`lowclo`, `highcl`, `driftc`, `parall`). The HASH always carries all three
    whatever the visible
    name shows, so two planted worlds are always two directories; the cap costs
    legibility, never separation.

    REFUSES an unknown world rather than naming it `w` and nothing: an id that
    silently dropped the world would put two worlds in one directory, which is
    the defect this whole key exists to prevent.
    """
    tag = f"{float(alpha):g}"
    if noise:
        tag += f"n{float(noise):g}"
    if world:
        if world not in WORLD_ID_TAGS:
            raise ValueError(
                f"planted world {world!r} has no entry in WORLD_ID_TAGS; add one "
                "before planting it, or its runs share a directory with another "
                "world's")
        tag += f"w{WORLD_ID_TAGS[world]}"
    return tag


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 makes the whole "
                         "rows-per-expert range reachable at four times the "
                         "token count, where deepseek-v3 needs thirty-two")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="bf16 or fp16. Not fp8: the alpha refit, the ridge "
                         "band and the crossing table above are all bf16 "
                         "statements, and the fp8 call path needs a quant "
                         "config this sweep does not build")
    ap.add_argument("--tiles", default="32,64,128,256")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="largest rows per expert. 1024 is 16 M-tiles at "
                         "BLOCK_M=64, which is 95%% of that block size's AI "
                         "ceiling and 4.9x the crossing the retracted alpha "
                         "predicts for it")
    ap.add_argument("--row-step", type=int, default=32)
    ap.add_argument("--num-stages", type=int, default=FIXED["num_stages"],
                    help="pipeline stages, applied to EVERY setting. The one "
                         "pinned parameter with a hard limit behind it: "
                         "BLOCK_SIZE_M=256 at 4 stages asks for about 164 KB of "
                         "shared memory, and a card that refuses it fails that "
                         "setting alone, which would unpin the sweep. Lower it "
                         "here and every setting moves together")
    ap.add_argument("--group-m", type=int, default=FIXED["GROUP_SIZE_M"],
                    help="the swizzle width, applied to EVERY setting. Pinned to "
                         "1 by default and by design: 1 is the setting vLLM's "
                         "FALLBACK ladder holds across the whole decode range, so "
                         "it is the one a deployment without a tuned file "
                         "actually runs. It is NOT a neutral choice, because "
                         "GROUP_SIZE_M is what groups consecutive M-tiles onto "
                         "one weight read, and alpha measured here is therefore "
                         "alpha AT THIS SWIZZLE rather than a property of the "
                         "kernel. Sweeping it is the point: on the published "
                         "surfaces the identifiable G=1 ladders read 0.62-1.02 "
                         "across models (mixtral 0.95-1.02 on both cards, qwen2 "
                         "0.65-0.84, deepseek-v2-lite 0.62) against 0.63-0.78 "
                         "at 8 and above, so the uncorrected ceiling "
                         "2*BM/(alpha*b) -- a BRACKET, and high by (1+phi+delta) "
                         "-- and therefore whether a given tile can EVER reach "
                         "the compute roof, moves with this number and with "
                         "the model")
    ap.add_argument("--block-n", type=int, default=FIXED["BLOCK_SIZE_N"],
                    help="the N tile, applied to EVERY setting. Exists to bound "
                         "the ACTIVATION confound rather than to tune anything. "
                         "An extra M-tile re-reads activations as well as "
                         "weights, in the ratio BLOCK_M/BLOCK_N, so at the "
                         "default 64 a BLOCK_M=64 setting re-reads them one for "
                         "one and a BLOCK_M=256 setting four times over -- which "
                         "means alpha fitted across that grid is NOT bounded by "
                         "the 0.25 this study quotes elsewhere. Raise this to "
                         "256 and the ratio at BLOCK_M=64 falls to 0.25; if "
                         "alpha does not move, the weight-traffic reading holds")
    ap.add_argument("--step-probes", type=int, default=6,
                    help="tile boundaries per block size to bracket for gate 1")
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob on 2026-09-02 and kept in "
                         "the run id. moe.bench.timing.time_kernel sizes the "
                         "iteration count from --cell-budget-ms and the "
                         "warmup's own queue-deep per-call time, which is the "
                         "only sizing that can hold one trial to a duration. "
                         "It stays in the id because cells measured under a "
                         "different count exist on disk and a directory must "
                         "not be resumed into across that change, and it still "
                         "caps --dry-run's cost estimate")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED 2026-09-02: a count "
                         "is the wrong unit and the ladders that compared cells "
                         "warmed at 5 against cells warmed at 20 were comparing "
                         "clock states. A 1 ms kernel needs hundreds of calls "
                         "before the governor reacts and a 30 ms GEMM needs "
                         "one, so the instrument warms for a duration measured "
                         "with the same events the trials use")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per cell; the percentiles are over "
                         "iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the roof was measured flushed and a "
                         "warm-L2 cell is not comparable with it. Recorded per "
                         "cell and in the run id, so a flushed and an unflushed "
                         "sweep can never share a directory")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its iteration count from it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sm-count", type=int, default=0,
                    help="0 asks the driver; only needed off-GPU")
    ap.add_argument("--capability", default="",
                    help="compute capability as MAJOR.MINOR, e.g. 8.0 for the "
                         "A100 or 9.0 for the H200. Empty asks the driver; "
                         "give it off-GPU to get the shared-memory verdict in "
                         "the tile-resource plan. The register check needs no "
                         "device and runs either way")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="Op/B. 0 (the default) reads the ATTACHED DEVICE's "
                         "own calibration and REFUSES if there is none. It used "
                         "to default to 160.3, an H200 figure, and "
                         "cross_card_surface.sh never passed it, so all 7 "
                         "published A100 reports were scored against a ridge "
                         "belonging to neither card")
    ap.add_argument("--ridge-band", default="",
                    help="LO,HI in Op/B, only meaningful with --ridge. Without "
                         "it an operator-asserted ridge gets a DEGENERATE band "
                         "and the report says so, rather than inheriting a "
                         "width measured on some other machine")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--bandwidth-gbps", "--bandwidth", type=float, default=0.0,
                    dest="bandwidth_gbps", metavar="GBPS",
                    help="GB/s. 0 (the default) reads the ATTACHED DEVICE's "
                         "own calibration and REFUSES if there is none. It used "
                         "to fall back to 4374.5, an H200 triad figure, on a "
                         "missing OR unreadable calibration, so --ridge 145.8 "
                         "(an A100 ridge) built a roof of 145.8 x 4374.5 out of "
                         "two machines. Required alongside --ridge off a "
                         "calibrated card, because that is exactly the pairing "
                         "that produced the hybrid")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid, the predictions and the cost, then stop")
    ap.add_argument("--self-test", type=float, default=None, metavar="ALPHA",
                    help="generate the cells from the model at this alpha and "
                         "run the whole analysis on them, off GPU")
    ap.add_argument("--self-test-noise", type=float, default=0.0,
                    help="lognormal sigma applied to every synthetic cell")
    ap.add_argument("--self-test-world", default="", choices=("",) + SELF_TEST_WORLDS,
                    help="a planted world that is not a single alpha. "
                         f"{DRIFT_WORLD} moves one tread's clock across its "
                         "own trials, which must be excluded and counted; "
                         f"{LOW_CLOCK_WORLD} puts one tread steadily below the "
                         "roof's clock and "
                         f"{HIGH_CLOCK_WORLD} one steadily above it, the two "
                         "off-band states a power-capped card holds per tile, "
                         "both of which must be KEPT and counted on their "
                         "side; "
                         f"{PARALLEL_WORLD} puts the memory branch on the "
                         "compute branch, which must come out UNDECIDED rather "
                         "than blank. Implies --self-test at the refit alpha "
                         "unless one is given")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="RETIRED 2026-09-02 and accepted so old driver lines "
                         f"still parse. A failed CLAIM gate now always exits "
                         f"{exit_codes.CLAIM_FAIL} CLAIM_FAIL, which the ledger "
                         "reads as a finished result rather than a retry; "
                         "folding it into 0 made the log disagree with the "
                         "process")
    return ap


def self_test_cells(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                    bandwidth_gbps: float, b: int, sm_count: int,
                    noise: float = 0.0, seed: int = 0, world: str = "",
                    warmup_ms: float = 0.0, trials: int = 0,
                    l2_flush: bool = True) -> list[Cell]:
    """The synthetic cells, plus the two worlds that are not a single alpha.

    A SELF-TEST THAT CANNOT FAIL IS NOT ONE, and until 2026-09-02 this script's
    self-test could only plant an alpha. The two paths that decide whether a
    BLOCK_M=128 row exists at all were therefore never exercised off-GPU:

      `drifting-clock` sets ONE tread of the null tile's `clock_drift_ok` False:
      its clock moved across its own trials, so its median is a blend of two
      operating points. `ladder_treads` must drop it, the ladder row must say
      how many were dropped, and if that takes the ladder under
      `MIN_MEMORY_TREADS` the outcome must be `UNDECIDED_DRIFTING_CLOCK` and
      not "too few treads".

      `low-clock` puts ONE tread of the null tile steadily below the clock the
      roof was measured at, `clock_level_ok` False with the side LOW at 0.7x.
      Since 2026-09-09 `ladder_treads` must KEEP it and the ladder row must say
      one tread was kept that way; it was an exclusion world until then, and
      that rule removed every BM=128/BN=64 cell of the 2026-09-09 H200 session.

      `high-clock` puts the SAME tread above the band, `clock_level_ok` False
      with the side HIGH at `H200_BOOST_RATIO` times the reference.
      `ladder_treads` must KEEP it, the ladder row must say one tread was kept
      that way, and every number the clean world reports must come back
      unchanged: the boosted memory-shaped cell is a measurement of the memory
      branch, and until 2026-09-08 this file dropped it.

      `parallel-branch` puts the SUBJECT tile's whole ladder on a line whose
      slope is the compute slope, which is the regime
      `PARALLEL_BRANCH_TOLERANCE` refuses to decide. The outcome must be
      `UNDECIDED_PARALLEL_BRANCH` carrying its reason, rather than the `None`
      that used to be indistinguishable from a sweep that lacked treads.

    Every world is built by MODIFYING cells the model generated, not by
    hand-writing a ladder, so everything else in the report stays the world the
    alpha describes and only the planted thing differs.
    """
    if world and world not in SELF_TEST_WORLDS:
        raise ValueError(f"unknown self-test world {world!r}; "
                         f"choices are {list(SELF_TEST_WORLDS)}")
    null_bm = null_block_m(block_sizes)
    low = (null_bm, 2) if world == LOW_CLOCK_WORLD else None
    high = (null_bm, 2) if world == HIGH_CLOCK_WORLD else None
    moved = (null_bm, 2) if world == DRIFT_WORLD else None
    cells = synthetic_cells(cfg, grid, block_sizes, alpha=alpha, ridge=ridge,
                            bandwidth_gbps=bandwidth_gbps, b=b,
                            sm_count=sm_count, noise=noise, seed=seed,
                            low_clock=low, high_clock=high, drifting=moved,
                            warmup_ms=warmup_ms, trials=trials,
                            l2_flush=l2_flush)
    if world != PARALLEL_WORLD:
        return cells
    # The compute branch scales as C ~ BLOCK_M off the reference ladder, so a
    # subject ladder whose time is the reference's time at the same ROW COUNT is
    # a ladder sitting exactly on the compute branch. Built from the reference's
    # own cells rather than from a formula, so it stays parallel whatever the
    # model does to the reference.
    subject = 128 if 128 in block_sizes else null_bm
    reference_bm = max(block_sizes)
    if subject == reference_bm:
        return cells
    by_rows = {c.rows_per_expert: c for c in cells if c.block_m == reference_bm}
    out = []
    for c in cells:
        peer = by_rows.get(c.rows_per_expert)
        if c.block_m != subject or peer is None or peer.ms_p50 <= 0:
            out.append(c)
            continue
        # A hair ABOVE the compute branch, so the tread qualifies as memory
        # bound and the fit gets far enough to notice the two slopes are one
        # line. Sitting ON it would be caught earlier, by membership, and would
        # test a different branch of the code.
        out.append(make_cell(
            cfg, c.rows_per_expert, subject, peer.ms_p50 * 1.10,
            sm_count=sm_count, block_n=FIXED["BLOCK_SIZE_N"],
            ms_min=peer.ms_p50 * 1.10, ms_stdev=peer.ms_p50 * 1.10 * noise,
            instrument=SYNTHETIC_INSTRUMENT, warmup_ms=warmup_ms,
            trials=trials, l2_flush=l2_flush,
            sm_clock_load_mhz=SYNTHETIC_CLOCK_MHZ, clock_level_ok=True,
            clock_drift_ok=True))
    return out


def missing_gpu_stack() -> str:
    """Which half of the stack is absent, and what to run instead.

    One function rather than three checks inline so the message a laptop gets
    is the same one the pod would get, and so the test suite can assert on it
    without a GPU. Empty string means the sweep can run.
    """
    try:
        import torch
    except ImportError:
        return ("no torch on this machine. --self-test 0.558 runs the whole "
                "analysis off GPU; --dry-run prints the grid.")
    if not torch.cuda.is_available():
        return ("no CUDA device. --self-test 0.558 runs the whole analysis off "
                "GPU; --dry-run prints the grid.")
    try:
        import vllm  # noqa: F401
    except ImportError:
        return ("vLLM is not importable in this environment, and it owns the "
                "kernel this sweep overrides.\nOn the pod: source the vllm venv "
                "(scripts/setup_runpod.sh vllm). Off GPU: --self-test 0.558.")
    return ""


class RidgeUnavailable(RuntimeError):
    """No ridge this run is entitled to use, and no constant may stand in.

    Raised rather than defaulted. `--ridge` used to default to `RIDGE_BAND[0]`
    and `scripts/cross_card_surface.sh` never passed it, so seven A100 reports
    were written against 160.3 Op/B -- a withdrawn H200 figure, calibration
    md5 4d84542b -- and every
    `ridge x bandwidth` they printed was a hybrid of two machines. Nothing
    about that failure was visible in the output, which is precisely why the
    replacement refuses instead of choosing.
    """


@dataclass(frozen=True)
class ResolvedRidge:
    """The ridge, its band, and where each came from. Provenance travels with it."""

    ridge: float
    band: tuple[float, float]
    source: str
    band_source: str
    device: str
    #: `cli`, `calibration` or `hypothesis`: the SHORT token, beside the prose.
    #: The prose says which yaml and which session; the token is what a
    #: provenance block records and what an audit gate compares, because a gate
    #: that has to pattern-match a sentence is a gate that breaks on a reword.
    #: Defaulted so the three sibling scripts that read `ResolvedRidge` are
    #: unaffected.
    source_kind: str = ""


#: The phrase `moe/bench/calibrate.py` writes onto a bandwidth pattern it has
#: disowned. A ridge built from a disowned denominator is a ridge built from a
#: number the calibration itself says is not a ceiling.
DISOWNED_NOTE = "not a valid ceiling"


def _measured_yaml(gpu_name: str) -> dict:
    """THIS device's calibration yaml as a dict, or {}.

    Read directly rather than through `roofline.Hardware`, which carries only
    the headline peaks. The candidate order mirrors `load_measured` so this
    cannot end up describing a different file from the one the ridge came from,
    and the device name inside the file is checked again here: a detail block
    from another machine would put another machine's pattern spread on this
    run's band, which is the same class of error the whole change is about.
    """
    try:
        import yaml

        from moe.bench.roofline import HARDWARE_DIR, measured_slug
    except ImportError:                                   # pragma: no cover
        return {}
    stems = ([measured_slug(gpu_name)] if gpu_name else []) + ["measured"]
    for stem in stems:
        path = HARDWARE_DIR / f"{stem}.yaml"
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:                                 # noqa: BLE001
            return {}
        named = str((data.get("detail") or {}).get("gpu_name") or "")
        if named and gpu_name:
            norm = lambda t: "".join(c for c in t.lower() if c.isalnum())  # noqa: E731
            if norm(named) not in norm(gpu_name) and norm(gpu_name) not in norm(named):
                return {}
        return data
    return {}


def _measured_detail(gpu_name: str) -> dict:
    """Just the `detail` block, which is where the per-pattern ridges live."""
    return _measured_yaml(gpu_name).get("detail") or {}


def calibration_stamp_line(doc: dict) -> str:
    """Which calibration SESSION the ridge came from, named in the report.

    The device match this replaces a hardcoded constant with is necessary and
    not sufficient: `measured_*.yaml` is keyed by DEVICE NAME, so a second pod
    of the same part inherits the first pod's ceilings, and the H200's dense
    bf16 moved 7.1% between 2026-08-28 and 2026-09-01 while its bandwidth
    reproduced to 0.014%. The ridge is compute over bandwidth, so that drift
    lands entirely on the ridge. Naming the session does not stop the reuse --
    `scripts/cross_card_surface.sh` recalibrating first is what stops it -- but
    it makes a report that inherited one say whose it was.
    """
    when = doc.get("checked_on")
    commit = str(doc.get("measured_commit") or "")[:8]
    dirty = " DIRTY TREE" if doc.get("measured_dirty") else ""
    if not when and not commit:
        return "calibration session unstamped"
    return f"calibrated {when or 'undated'} at {commit or 'no commit'}{dirty}"


def ridge_band_from_detail(detail: dict, ridge: float
                           ) -> tuple[tuple[float, float], str]:
    """A ridge band from THIS device's own bandwidth patterns, or a degenerate one.

    The band is the same silicon measured against several DRAM rulers, which is
    the honest width of a single calibration: the compute term is common to
    every end, so the spread is carried as a RATIO against the ceiling pattern
    and applied to `ridge`. Carrying the ratio rather than the stored
    `ridge_by_pattern` values keeps this correct for a dtype whose peak is not
    the bf16 peak those values were computed from.

    Patterns the calibration disowned are excluded: `calibrate.py` marks a
    pattern that came in below triad, which a read cannot legitimately do, and
    a band built from one is a band built from a number the file has already
    withdrawn.

    Returns a DEGENERATE band when fewer than two rulers survive. Degenerate is
    the honest answer for one calibration; borrowing another machine's band is
    not, and that is what this function exists to stop.
    """
    by_pattern = detail.get("ridge_by_pattern") or {}
    ceiling = detail.get("ceiling_pattern")
    base = by_pattern.get(ceiling)
    if not base or not ridge:
        return (ridge, ridge), ("degenerate: the calibration records no "
                                "per-pattern ridges, so this run has one "
                                "denominator and the band is one number twice")
    disowned = {p.get("pattern") for p in (detail.get("bandwidth_patterns") or [])
                if DISOWNED_NOTE in str(p.get("note") or "")}
    kept = {k: v for k, v in by_pattern.items() if k not in disowned and v}
    if len(kept) < 2:
        return (ridge, ridge), ("degenerate: fewer than two bandwidth patterns "
                                "survived this calibration's own disowning")
    scaled = sorted(ridge * v / base for v in kept.values())
    names = ", ".join(sorted(kept))
    return (scaled[0], scaled[-1]), (
        f"this device's own bandwidth patterns ({names}), carried as a ratio "
        f"against the {ceiling} ceiling; {sorted(disowned)} excluded as "
        "disowned by the calibration" if disowned else
        f"this device's own bandwidth patterns ({names}), carried as a ratio "
        f"against the {ceiling} ceiling")


#: The pattern names a calibration may use for a PURE READ, in the order this
#: file will accept them. The H200 profile writes `read_stream`; the A100
#: profile, written by an older calibration, writes `read`. A weight stream is
#: a pure read, so this is the pattern a `w` would be divided by if the memory
#: branch achieved the read rate rather than the ceiling the report divides by.
READ_PATTERN_NAMES = ("read_stream", "read")


def read_versus_denominator(card: str, bandwidth_gbps: float) -> str:
    """This card's own pure-read pattern against the rate `w` divides by.

    THE SENTENCE THAT TOLD THE READER TO CONSULT THE CARD TYPED ONE CARD'S
    NUMBER. Until 2026-09-10 the `w` legend ended "on the H200 of 2026-09-10
    read_stream is 5.4% above the triad figure these reports divide by, and
    every w rises by that much against it", in the same breath as telling the
    reader to read the card's own `bandwidth_patterns`. The clause scoped it to
    the H200, so the sentence was not false; on the committed A100 profile the
    read pattern is 1744.3 GB/s against a triad of 1799.4, 3.06% BELOW, so w
    FALLS at the read rate there and stays an upper bound. The guidance pointed
    backwards on the only other card this study has run.

    The run already holds the profile the sentence sends the reader to, so it
    is read here. The denominator is the run's OWN resolved bandwidth rather
    than the file's triad entry, because that is the number every `w` on the
    page was actually divided by; a run given `--bandwidth` on the command line
    is compared against what it was told, not against what the card measured.

    Returns a clause for the legend, never a refusal: a card this tree holds no
    profile for is a reason to say so on the line, not to blank the `w`
    column.
    """
    if not bandwidth_gbps or bandwidth_gbps <= 0:
        return ("this run resolved no bandwidth, so there is no denominator to "
                "compare a read rate against")
    data = _measured_yaml(card) if card and card != NO_CARD_SLUG else {}
    patterns = {p.get("pattern"): p.get("gbps")
                for p in ((data.get("detail") or {}).get("bandwidth_patterns") or [])
                if p.get("gbps")}
    name = next((n for n in READ_PATTERN_NAMES if patterns.get(n)), "")
    if not name:
        return (f"this tree holds no measured bandwidth_patterns for "
                f"card={card or NO_CARD_SLUG}, so the read-versus-ceiling gap "
                "cannot be read here; run scripts/calibrate_hardware.py on the "
                "box, or read detail.bandwidth_patterns in that card's own file")
    read = float(patterns[name])
    gap = read / bandwidth_gbps - 1.0
    if gap >= 0:
        return (f"on card={card} the measured {name} pattern is {read:.1f} GB/s, "
                f"{gap * 100:.2f}% ABOVE the {bandwidth_gbps:.1f} GB/s this "
                f"report divides by, so every w below RISES by that much at the "
                "read rate and the printed number is not an upper bound there")
    return (f"on card={card} the measured {name} pattern is {read:.1f} GB/s, "
            f"{-gap * 100:.2f}% BELOW the {bandwidth_gbps:.1f} GB/s this report "
            f"divides by, so every w below FALLS by that much at the read rate "
            "and the printed number stays above it")


def resolve_ridge(args, *, synthetic: bool) -> ResolvedRidge:
    """The ridge THIS run is entitled to quote, or a refusal.

    Order, and each step is a different kind of claim:

      1. `--ridge` (with optional `--ridge-band`), which makes the number the
         operator's assertion and puts it in the run's own command line.
      2. THE ATTACHED DEVICE'S OWN CALIBRATION: `peak(dtype) / bandwidth`, from
         the yaml `scripts/calibrate_hardware.py` wrote for this GPU. This is
         the default, and it is the whole point of the change: the A100's
         contemporaneous calibration puts its ridge at 145.8 and the H200's at
         162.8, and the reports that quoted 160.3 on the A100 were quoting
         neither.
      3. For `--dry-run` and `--self-test` ONLY, where nothing was measured and
         so nothing can be mislabelled, the module's H200 band as a stated
         HYPOTHESIS. The source string says so and is written into the report.
      4. Otherwise REFUSE. A measured run with no calibration for its own
         device does not get a ridge from anywhere else.
    """
    band_arg = getattr(args, "ridge_band", "") or ""
    if args.ridge:
        band = (args.ridge, args.ridge)
        band_source = ("degenerate: --ridge given as one number and no "
                       "--ridge-band with it")
        if band_arg:
            ends = tuple(float(v) for v in band_arg.split(","))
            if len(ends) != 2:
                raise RidgeUnavailable(
                    f"--ridge-band wants LO,HI; got {band_arg!r}")
            band = (min(ends), max(ends))
            band_source = "given on the command line"
        return ResolvedRidge(args.ridge, band, "given on the command line",
                             band_source, "", source_kind="cli")

    from moe.bench import roofline
    gpu_name = roofline.current_gpu_name()
    try:
        hw = roofline.load_measured(gpu_name or None)
    except roofline.HardwareMismatch as exc:
        raise RidgeUnavailable(str(exc)) from exc
    if hw is not None:
        try:
            ridge = hw.ridge_point(args.dtype)
        except ValueError as exc:
            raise RidgeUnavailable(
                f"{hw.name} has a measured bandwidth but no verified "
                f"{args.dtype} peak, so it cannot state a ridge: {exc}") from exc
        doc = _measured_yaml(gpu_name)
        band, band_source = ridge_band_from_detail(doc.get("detail") or {}, ridge)
        return ResolvedRidge(
            ridge, band,
            f"measured on this device: {hw.name}, "
            f"{hw.peak(args.dtype) / 1e12:.1f} TFLOP/s {args.dtype} over "
            f"{hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
            f"({hw.ceiling_pattern or 'unnamed'} pattern); "
            f"{calibration_stamp_line(doc)}",
            band_source, hw.name or gpu_name, source_kind="calibration")

    if synthetic:
        return ResolvedRidge(RIDGE_BAND[0], RIDGE_BAND, HYPOTHESIS_RIDGE_SOURCE,
                             HYPOTHESIS_RIDGE_SOURCE, gpu_name,
                             source_kind="hypothesis")

    raise RidgeUnavailable(
        f"no calibration for this device ({gpu_name or 'no CUDA device'}), so "
        "this run has no ridge it is entitled to quote.\n"
        "    Every roof fraction, every AI cap comparison and every crossing "
        "prediction below would be scored against another machine's ceiling: "
        f"the module constant is {RIDGE_BAND[0]} Op/B, which is a withdrawn "
        "H200 figure (calibration md5 4d84542b) and belongs to no attached "
        "device.\n"
        "    Run:  python scripts/calibrate_hardware.py\n"
        "    or state the assertion yourself:  --ridge <Op/B> "
        "[--ridge-band LO,HI]\n"
        "    off GPU, --dry-run and --self-test may assume the H200 band and "
        "say so in the report.")


class BandwidthUnavailable(SystemExit, RuntimeError):
    """No bandwidth this run is entitled to use, and no constant may stand in.

    IT IS A `SystemExit` AS WELL AS A `RuntimeError`, AND THAT IS THE DELIVERY,
    not a curiosity. `scripts/tile_cap_test.py:1760` calls `resolve_bandwidth`
    OUTSIDE any try and cannot be edited from here, so a plain `RuntimeError`
    arrived there as an uncaught traceback and exit 1 -- and 1 is `CLAIM_FAIL`
    in the very table this study adopted, so a driver would have ledgered a
    REFUSAL as a measured refutation. That is the audit's own defect, made by
    the fix for it. As a `SystemExit` carrying `code = REFUSED` the same
    uncaught refusal leaves the process at 2 with no traceback, which is what it
    means, and `except (RidgeUnavailable, BandwidthUnavailable)` in `main` still
    catches it by name because it is still a `RuntimeError` too.

    BEING A `RuntimeError` MEANS A BLANKET `except Exception` DOES CATCH IT, and
    that is the price of the compatible path, not a protection this class has.
    The MRO is `SystemExit` then `RuntimeError` then `Exception`, so a caller
    that swallows everything still swallows this. Two things make that survivable
    and neither is the class: the reason is printed at the raise site before it
    propagates, and no caller in this repository wraps `resolve_bandwidth` in a
    blanket handler. If one ever does, the refusal becomes a silent default
    again, so the check belongs in review of the caller.

    THE MESSAGE IS PRINTED AT THE RAISE SITE, once, by `_refuse_bandwidth`,
    because an unhandled `SystemExit` whose code is an int prints nothing at
    all. `main` therefore returns REFUSED without re-printing.

    THE HYBRID ROOF, WHICH `resolve_ridge` ALREADY REFUSED AND THIS DID NOT.
    Every predicted millisecond, every `ridge x bandwidth` roof and therefore
    every roof fraction, every gate-4 verdict and the whole compute-reference
    LEVEL check divide by this number. Until 2026-09-02 `resolve_bandwidth`
    returned the published H200 triad figure, 4374.5 GB/s, whenever
    `load_measured` returned None OR raised ANY exception -- and `--ridge X`
    took the operator's assertion for the numerator while leaving that constant
    in the denominator. So `--ridge 145.8` on an A100 with no calibration
    produced `model_roof = 145.8 x 4374.5`: an A100 ridge times an H200
    bandwidth, printed as this card's roof, with a source line that said
    "published H200 triad ceiling" in a field nobody was reading.

    The two halves of a roof have to come from the same machine, or the roof is
    not a machine's. So this refuses on exactly the terms `RidgeUnavailable`
    does, and `--ridge` now requires `--bandwidth` beside it.
    """

    def __init__(self, message: str):
        super().__init__(message)
        # `SystemExit.__init__` has just set `code = message`, which the
        # interpreter would print and then exit 1 with. The code is the whole
        # point of the class being a SystemExit, so it is overwritten here;
        # `str(exc)` still returns the message, and `_refuse_bandwidth` has
        # already printed it.
        self.code = exit_codes.REFUSED


def _refuse_bandwidth(message: str) -> BandwidthUnavailable:
    """Print the refusal once, then hand back the exception to raise.

    A refusal nobody can read is not a refusal. This function exists because the
    exception is a `SystemExit`: a caller with no try around
    `resolve_bandwidth` gets exit 2 and, without this, a silent one. Printing
    here rather than in `__init__` keeps the side effect at the one place that
    refuses, and keeps `raise` a `raise`.
    """
    print(f"REFUSED: {message}")
    return BandwidthUnavailable(message)


@dataclass(frozen=True)
class ResolvedBandwidth:
    """The bandwidth, and where it came from. Provenance travels with it.

    `source` is the short token (`cli`, `calibration`, `hypothesis`) a
    provenance block records and an audit gate compares; `detail` is the prose
    that names the yaml and the pattern. Two fields because a gate that has to
    pattern-match a sentence breaks on a reword, and a report that prints only
    the token tells a reader nothing about which file it came from.

    THE COMPATIBLE PATH. This unpacks as the `(gbps, source_prose)` pair
    `resolve_bandwidth` used to return, so `scripts/tile_cap_test.py:1760`'s
    `bandwidth, bw_source = SWEEP.resolve_bandwidth(args)` keeps working
    unchanged. New callers take the fields by name and get the token as well.
    """

    gbps: float
    source: str
    detail: str
    device: str

    def __iter__(self):
        yield self.gbps
        yield self.detail


#: The figure that used to stand in silently. Kept ONLY so the refusal below can
#: name what it is refusing to substitute, and so a reader of an older report can
#: recognise the number. It is an H200 triad ceiling and belongs to no other
#: card; nothing in this file may return it as a value.
PUBLISHED_H200_TRIAD_GBPS = 4374.5


#: What a report says when its bandwidth is the module constant rather than a
#: measurement. Only reachable on a laptop plan or replay whose RIDGE is the
#: matching hypothesis, so the two halves of the roof still come from one
#: machine: 4374.5 GB/s is the bandwidth behind the withdrawn band's 176.2 end
#: (770.9 TFLOP/s over 4374.5), not a 2026-08-26 figure as this used to say.
HYPOTHESIS_BANDWIDTH_SOURCE = (
    "HYPOTHESIS: the H200 triad ceiling behind the withdrawn ridge band, which "
    "belongs to no attached device, paired with the H200 hypothesis ridge")


def resolve_bandwidth(args, *, synthetic: bool | None = None) -> ResolvedBandwidth:
    """This card's measured bandwidth, the operator's assertion, or a refusal.

    Order, mirroring `resolve_ridge` step for step:

      1. `--bandwidth` / `--bandwidth-gbps`, the operator's assertion, source
         `cli`.
      2. THE ATTACHED DEVICE'S OWN CALIBRATION, source `calibration`.
      3. For `--dry-run` and `--self-test` ONLY, AND ONLY WHEN `--ridge` WAS NOT
         GIVEN, the module's H200 triad as a stated HYPOTHESIS, source
         `hypothesis`. The second condition is the whole point: a hypothesis
         bandwidth is safe only while the ridge beside it is the MATCHING
         hypothesis, the withdrawn H200 band, so the roof is one machine's.
         The moment the operator asserts a ridge for another card, pairing it
         with this constant is the hybrid roof, and the escape closes.
      4. Otherwise REFUSE, on exactly the terms `resolve_ridge` refuses.

    RETURN TYPE CHANGED on 2026-09-02, from `(float, str)` to
    `ResolvedBandwidth`, so a caller cannot take the number and drop the source:
    that is how 4374.5 reached reports for a card whose own triad is 1799.4.
    The one in-repo caller outside this file monkeypatches this function in a
    test (`tests/test_tile_cap.py`), which passes its own callable and is
    unaffected by the shape.

    HOW THE REFUSAL IS DELIVERED, since one caller cannot catch it.
    `BandwidthUnavailable` is a `SystemExit` carrying `code = REFUSED`, and
    `_refuse_bandwidth` prints the reason before it is raised. So
    `scripts/tile_cap_test.py --ridge 145.8 --dry-run`, which calls this outside
    any try, now ends with the refusal on stdout and exit 2 instead of a
    traceback and exit 1. It is still a `RuntimeError`, so `main`'s named
    `except` is unchanged.
    """
    if synthetic is None:
        # INFERRED, not defaulted to False. `scripts/tile_cap_test.py` calls
        # this with one argument, and defaulting to False would refuse its
        # documented laptop `--dry-run` -- a script this one does not own,
        # broken by a keyword it does not pass.
        synthetic = bool(getattr(args, "dry_run", False)
                         or getattr(args, "self_test", None) is not None)
    if args.bandwidth_gbps:
        return ResolvedBandwidth(
            args.bandwidth_gbps, "cli",
            "given on the command line as --bandwidth/--bandwidth-gbps", "")
    from moe.bench import roofline
    gpu_name = roofline.current_gpu_name()
    hw = None
    try:
        hw = roofline.load_measured(gpu_name or None)
    except Exception as exc:                            # noqa: BLE001
        # Broad, and it REFUSES rather than substituting. A yaml this process
        # cannot parse is not a licence to use another machine's ceiling; it is
        # a reason to stop and say which file failed.
        raise _refuse_bandwidth(
            f"this device's calibration could not be read "
            f"({type(exc).__name__}: {exc}), so this run has no bandwidth it is "
            "entitled to quote.\n"
            "    Run:  python scripts/calibrate_hardware.py\n"
            "    or state it yourself:  --bandwidth <GB/s>") from exc
    if hw is not None:
        return ResolvedBandwidth(
            hw.bandwidth_bytes_s / 1e9, "calibration",
            f"measured on this device: {hw.name}, "
            f"{hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
            f"({hw.ceiling_pattern or 'unnamed'} pattern)",
            hw.name or gpu_name)
    ridge_arg = getattr(args, "ridge", 0.0)
    if synthetic and not ridge_arg:
        return ResolvedBandwidth(PUBLISHED_H200_TRIAD_GBPS, "hypothesis",
                                 HYPOTHESIS_BANDWIDTH_SOURCE, gpu_name)
    if ridge_arg:
        raise _refuse_bandwidth(
            f"--ridge {ridge_arg} Op/B was given and there is no bandwidth to "
            f"pair it with on this device ({gpu_name or 'no CUDA device'}).\n"
            "    The ridge is then YOUR assertion, for some card, and the only "
            f"bandwidth left is the {PUBLISHED_H200_TRIAD_GBPS} GB/s H200 "
            "constant: their product is a roof built out of two machines. That "
            "is the defect exactly -- --ridge 145.8 is an A100 figure and "
            "145.8 x 4374.5 is not any card's roof.\n"
            "    Every predicted millisecond, the ridge x bandwidth roof, every "
            "roof fraction and the compute reference's LEVEL check divide by "
            "this number.\n"
            "    Run:  python scripts/calibrate_hardware.py\n"
            "    or state the assertion yourself:  --bandwidth <GB/s>")
    # NO RIDGE EITHER, so no roof can be assembled and there is nothing to make
    # a hybrid OUT OF. The honest record is a bandwidth of zero with the source
    # `unresolved`, never the constant: zero is not a plausible ceiling, so any
    # arithmetic on it produces an obvious infinity rather than a roof that
    # looks right. `main` walls it below and refuses before anything is
    # measured, and `resolve_ridge` refuses first in this same state, so the
    # wall is a second lock on a door that is already shut.
    #
    # WHY THIS ONE IS RETURNED AND NOT RAISED, said plainly.
    # `scripts/tile_cap_test.py:1760` calls this function OUTSIDE a try and
    # then refuses on its own ridge two lines later. Its refusal is the better
    # message here -- it names the ridge as well -- so this state stays a
    # return and lets that one speak. A zero it never reads costs nothing.
    # Every state where a hybrid roof COULD form -- an asserted ridge with no
    # bandwidth, an unreadable calibration -- raises above, and since
    # 2026-09-02 those raises land in that same untried caller as exit 2
    # REFUSED with the reason printed, not as a traceback and exit 1. Exit 1 is
    # CLAIM_FAIL, and a driver reading one would have written a refusal into
    # the ledger as a measured refutation.
    return ResolvedBandwidth(
        0.0, "unresolved",
        f"NO BANDWIDTH: this device ({gpu_name or 'no CUDA device'}) has no "
        "calibration, --bandwidth was not given, and this is not a --dry-run "
        "or --self-test. There is no ridge either, so no roof can be formed. "
        "Run scripts/calibrate_hardware.py, or state --bandwidth <GB/s>",
        gpu_name)


def _main(argv=None) -> int:
    """The run itself. `main` wraps it; every return here is an exit code.

    Every return is a member of `moe.bench.exit_codes`'s table: REFUSED (2)
    before anything is measured, and otherwise `classify` over the scored gates
    with nothing folded. `--dry-run` returns REFUSED, which is what the
    repository's census settled on: it measured nothing, so it scores no gate
    and its own log classifies as a refusal. The dry-run branch records what had
    to move with the integer, because this arm's plan is priced by a program
    rather than only read by a person.

    `--fail-on-gate` is retired: a CLAIM_FAIL is returned as 1 whether or not it
    is passed, because a falsified pre-registered claim is a successful
    experiment and 1 is already the code that says so to the ledger.
    A VALIDITY failure is INVALID either way, since nothing on the page may be
    quoted after one. ERROR (4) is `main`'s to return and not this function's.
    """
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    block_sizes = tuple(int(v) for v in args.tiles.split(","))
    b = dtype_bytes(args.dtype)
    if args.self_test_world and args.self_test is None:
        args.self_test = ALPHA
    synthetic = bool(args.dry_run or args.self_test is not None)
    # BOTH HALVES OF THE ROOF ARE RESOLVED BEFORE ANY GPU TIME IS SPENT, and
    # before the plan is printed, so a run that has no ruler it may quote costs
    # nothing and says why. Bandwidth refuses on the same terms the ridge does:
    # they are the numerator and the denominator of one roof and a run is not
    # entitled to take one from this card and the other from a constant.
    try:
        rr = resolve_ridge(args, synthetic=synthetic)
        rb = resolve_bandwidth(args, synthetic=synthetic)
    except BandwidthUnavailable:
        # NOT re-printed. `_refuse_bandwidth` printed it at the raise site,
        # because that exception is a SystemExit and reaches sibling scripts
        # that have no try to print it for them. Printing it twice here would
        # make one refusal look like two.
        return exit_codes.REFUSED
    except RidgeUnavailable as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    if rb.source == "unresolved":
        # The one state `resolve_bandwidth` reports rather than raises, because
        # a sibling script calls it outside a try. Nothing in THIS file may
        # proceed on it; see that function's tail comment for why.
        print(f"REFUSED: {rb.detail}")
        return exit_codes.REFUSED
    bandwidth, bw_source = rb.gbps, rb.detail
    grid = build_grid(cfg, block_sizes, args.r_max, args.row_step,
                      args.step_probes)
    step = rows_step(cfg)
    pinned = dict(FIXED, num_stages=args.num_stages,
                  GROUP_SIZE_M=args.group_m, BLOCK_SIZE_N=args.block_n)

    card = detect_card_slug()
    try:
        run_id = resolve_run_id(args, card)
    except SuppliedRunIdIsNotPlanted as exc:
        # BEFORE `out_dir` EXISTS, which is the whole point: the refusal has to
        # land before the path is computed, or the mkdir has already put a
        # planted run's directory next to the metered one it was named after.
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    out_dir = (args.out or results_root()) / "block_m_crossing" / run_id
    csv_path = out_dir / "cells.csv"
    cache_root = out_dir / "triton-cache"

    print(f"experiment  block_m_crossing / {run_id}")
    print(f"card        {card}"
          + ("   (no CUDA device: a plan or a replay, not a measurement)"
             if card == NO_CARD_SLUG else ""))
    if args.self_test is not None:
        # THE HOST, NOT THE SUBJECT. On a pod this line names a real card that
        # measured nothing here, which is how a planted report came to look like
        # that card's. The id says so where it cannot be missed.
        print(f"            the line above is the HOST of a planted run. The "
              f"run id's card component is {SYNTHETIC_CARD_SLUG!r} and the "
              f"attached card travels as `synthhost`.")
    print(f"model       {args.model} E={cfg.num_experts} k={cfg.top_k}  "
          f"{args.dtype} ({b} bytes)")
    print(f"pinned      {pinned}")
    print(f"grid        {len(grid)} rows-per-expert x {len(block_sizes)} block "
          f"sizes = {len(grid) * len(block_sizes)} cells")
    print(f"            r in [{grid[0]}, {grid[-1]}], tokens step {step}, "
          f"T in [{tokens_for_rows(cfg, grid[0])}, "
          f"{tokens_for_rows(cfg, grid[-1])}]")
    print(f"bandwidth   {bandwidth:.1f} GB/s, source={rb.source}: {bw_source}")
    print(f"ridge       {rr.ridge:.2f} Op/B, source={rr.source_kind}: {rr.source}")
    print(f"ridge band  {rr.band[0]:.2f}-{rr.band[1]:.2f} Op/B, {rr.band_source}")
    if rr.source_kind != rb.source:
        # Not a refusal: `--ridge` beside a calibrated bandwidth is a deliberate
        # assertion and the operator may make it. It IS said out loud, because
        # the two halves of the roof then came from different places and the
        # audit's finding was that nobody could see that from the output.
        print(f"            NOTE: the roof's two halves have different sources "
              f"(ridge {rr.source_kind}, bandwidth {rb.source})")
    # The run id deliberately does NOT include the ridge: it names the
    # MEASUREMENT, and the ridge changes only the analysis over it. Two runs of
    # the same grid at two ridges must share cells.csv, or the resume path
    # re-measures identical cells. What must not happen is a ridge reaching the
    # report without being named, which is what the two lines above prevent.
    print(f"WRITES TO   {out_dir}")
    print("            cells.csv (appended per cell), report.txt, report.json, "
          "triton-cache/")

    # THE PLAN'S RESOURCE BILL, PRINTED BEFORE ANY TIMING. A setting that cannot
    # hold its accumulator in registers, or its pipeline in shared memory, still
    # RUNS -- it spills, and returns a time proportional to its tile count that
    # sails through the compute reference's shape test. That is how a 249.765 ms
    # tile became this study's compute branch and cost 8 published cells. The
    # setting is refused here, where it is chosen.
    capability = resolve_capability(args, synthetic=synthetic)
    plan, tile_refusals = tile_resource_plan(pinned, block_sizes, b, capability)
    print("\nTILE RESOURCE PLAN, one CTA, at "
          + (f"sm_{capability[0]}{capability[1]}" if capability
             else "an UNKNOWN device (--capability MAJOR.MINOR gives the "
                  "shared-memory verdict; the register check runs regardless)"))
    for bm in block_sizes:
        print(plan[bm].render())
    if tile_refusals:
        for bm, why in tile_refusals.items():
            print(f"  REFUSED BLOCK_M={bm}: {why}")
        block_sizes = tuple(bm for bm in block_sizes if bm not in tile_refusals)
        if not block_sizes:
            print("REFUSED: every block size in --tiles is unrunnable as "
                  "pinned. Nothing to measure.")
            return exit_codes.REFUSED
        print(f"  sweeping {list(block_sizes)} only. The refused settings are "
              "NOT missing data: they are settings this hardware cannot run, "
              "and a timing taken from one would not be a measurement of the "
              "tiling this sweep is about.")
        # THE GRID IS NOT REBUILT, deliberately. `build_grid` derives its row
        # counts and its step probes from the block sizes, so rebuilding it here
        # would make the SAME run id mean two different grids on two cards --
        # sm_80 refuses this tile where sm_90 does not -- and `cells.csv` is
        # resumed by run id. The refused settings simply contribute no rows.

    if args.dry_run:
        # PRICED AS `time_kernel` WILL CHARGE, not as the retired loop did:
        # a fixed warmup DURATION plus `--trials` trials of `--cell-budget-ms`
        # each. `--iters` is deliberately not passed; it is retired as a timing
        # knob and passing it here is what made this line read a duration as a
        # call count.
        secs = estimated_seconds(cfg, grid, block_sizes, alpha=args.alpha,
                                 ridge=rr.ridge, bandwidth_gbps=bandwidth,
                                 b=b, warmup_ms=args.warmup,
                                 trials=args.trials,
                                 cell_budget_ms=args.cell_budget_ms)
        cells_planned = len(grid) * len(block_sizes)
        print(f"\nestimated GPU time {secs:.0f} s at the model's own timings, "
              "excluding compiles and allocation")
        print(f"  {cells_planned} cells x ({args.warmup:.0f} ms warmup + "
              f"{args.trials} trials x {args.cell_budget_ms:.0f} ms of kernel "
              "time), which is what the instrument charges; a cell slower than "
              f"{args.cell_budget_ms / ITERS_FOR_LO:.0f} ms costs more because "
              f"the iteration count floors at {ITERS_FOR_LO}")
        preds = predictions(block_sizes, args.alpha, rr.ridge, b)
        for bm in block_sizes:
            p = preds[bm]
            where = ("NO CROSSING EVER" if p.crossing_rows is None else
                     f"crosses at r={p.crossing_rows:.1f} "
                     f"(T={p.crossing_tokens(cfg.num_experts, cfg.top_k):.0f}), "
                     f"in the grid: {p.crossing_rows <= args.r_max}")
            print(f"  BLOCK_M={bm:3d} cap {p.ai_cap:7.1f}  {where}")
        # REFUSED (2), AS OF 2026-09-02, AND THE CENSUS IS NOW UNANIMOUS.
        # The census is at `scripts/bm128_depth.py`'s own dry-run branch: six
        # scripts returned DONE from `--dry-run` -- this file and
        # `ruler_rebaseline` as the bare literal 0 -- and seven returned
        # REFUSED, and REFUSED won the argument. A dry run scores no gate and
        # prints no RESULT line, so `exit_codes.classify_text` over this log
        # raises `NoGatesScored`, which that module documents as what a REFUSED
        # log looks like from there, while DONE means "measured; every VALIDITY
        # and CLAIM gate PASSED" and nothing here was measured. A plan whose
        # banner says REFUSED and whose process says DONE is the same defect
        # twice, and this file was the last one still saying both.
        #
        # WHAT MOVED WITH IT, BECAUSE THIS ARM'S PLAN IS READ BY A PROGRAM.
        # `scripts/replicate_noise_floor.py:sweep_cost` runs exactly this
        # branch as a subprocess to price a pod session, and it dropped the
        # whole "TOTAL ... min of GPU" budget line -- not "cost unknown", the
        # line -- on any non-zero code. Flipping this integer alone would have
        # traded a stated inconsistency for a silently missing cost on a rented
        # pod, which is the shape of defect this rebuild keeps producing while
        # closing another. So `sweep_cost` moved in the same commit: it now
        # accepts DONE and REFUSED from the probe, because REFUSED is what a
        # question about a plan is answered with, and keeps returning None for
        # every other code. The shell driver needed nothing: `dry_state` in
        # `scripts/h200_gaps_session.sh` has always mapped 0 to PLANNED and 2
        # to PLAN_REFUSED and re-queues neither.
        print()
        print("=" * 72)
        print("REFUSED. Nothing was measured and nothing was written.")
        print("  reason: --dry-run was given")
        print("  Everything above is a PLAN: the grid, the tile resource bill, "
              "the registered")
        print("  predictions and the cost this run would charge. No gate was "
              "scored, so no")
        print("  RESULT line was printed and none of it is a result. Run "
              "--self-test 0.558")
        print("  for the planted worlds, or the bare command on the pod.")
        print("=" * 72)
        return exit_codes.REFUSED

    if args.self_test is None:
        missing = missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return exit_codes.REFUSED

    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE PROVENANCE BLOCK PER RUN, built here and handed to both writers: the
    # report payload and every cells.csv row. Built AFTER the rulers resolve so
    # it carries their sources, and before any measurement so every row of one
    # run carries one block.
    # ONE STRING PER SOURCE, TOKEN FIRST, HANDED TO BOTH WRITERS. `stamp` puts
    # `ridge_source` and `bandwidth_source` at the payload's top level and
    # RAISES if the payload already carries a different value for either, so
    # the block and the report have to agree; making them literally the same
    # string is how they agree, and leading with the token
    # (`cli` / `calibration` / `hypothesis`) is what lets an audit gate compare
    # a source without pattern-matching a sentence.
    ridge_src = f"{rr.source_kind}: {rr.source}"
    bw_src = f"{rb.source}: {rb.detail}"
    # `instrument` IS THE SELF-TEST'S TOO, and it is the synthetic name.
    # `instrument` is one of the five keys `Provenance.stamp` puts at the
    # payload's top level and a publish gate checks, and report.json outlives
    # every log. A --self-test report carrying the real instrument's name
    # satisfied that gate while describing an instrument the run never touched;
    # `SYNTHETIC_INSTRUMENT` is the same string the planted cells carry, so the
    # report and its rows now agree about having measured nothing.
    #
    # `iters` IS None HERE, NOT `args.iters`. `--iters` is retired as a timing
    # knob: on a pod each cell's count comes from `time_kernel`, sized from
    # --cell-budget-ms, and is hundreds for a 1 ms kernel. Recording the dead
    # argparse default would have put `provenance.iters: 50` in every report
    # while every row of cells.csv said something else, and a reader could not
    # tell which was the instrument's. None reaches the block as the honest
    # "supplied as None"; the report's copy below carries the count the
    # instrument actually used.
    prov = PV.provenance_block(
        instrument=(SYNTHETIC_INSTRUMENT if args.self_test is not None
                    else timing_basis()),
        ridge=rr.ridge, ridge_source=ridge_src,
        bandwidth=bandwidth, bandwidth_source=bw_src,
        warmup_ms=args.warmup, iters=None,
        target_ms=args.cell_budget_ms)

    if args.self_test is not None:
        alpha = args.self_test
        sm_count = args.sm_count or DEFAULT_SM_COUNT
        cells = self_test_cells(cfg, grid, block_sizes, alpha=alpha,
                                ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                                sm_count=sm_count, noise=args.self_test_noise,
                                seed=args.seed, world=args.self_test_world,
                                warmup_ms=args.warmup, trials=args.trials,
                                l2_flush=not args.no_l2_flush)
        compiles = {bm: 1 for bm in block_sizes}
        executed = dict(compiles)
        print(f"\nSELF TEST: cells GENERATED from the model at alpha={alpha}"
              + (f", world={args.self_test_world}" if args.self_test_world else "")
              + ". Nothing here was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
    else:
        import torch
        alpha = args.alpha
        sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
        started = time.time()
        cells, compiles, executed = run_sweep(
            args, cfg, grid, block_sizes, csv_path, cache_root, b, pinned,
            prov=prov)
        print(f"\nswept in {time.time() - started:.0f} s")

    # THE REPORT'S BLOCK CARRIES THE INSTRUMENT'S OWN ITERATION COUNT; the one
    # the CSV rows were stamped with does not, because each row already has its
    # own `iters` column and a run-wide median would contradict most of them.
    report_prov = observed_iters(prov, cells)

    sm_source = ("given on the command line" if args.sm_count
                 else "reported by the driver" if args.self_test is None
                 else f"assumed H200 default {DEFAULT_SM_COUNT}")
    report = analyse(cells, cfg, block_sizes=block_sizes, alpha=alpha,
                     ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                     model_name=args.model, dtype=args.dtype, compiles=compiles,
                     executed=executed, sm_count=sm_count, sm_source=sm_source,
                     pinned=pinned, ridge_band=rr.band, ridge_source=ridge_src,
                     ridge_band_source=rr.band_source, capability=capability,
                     card=card, bandwidth_source=bw_src, prov=report_prov)
    print(report.text())

    (out_dir / "report.txt").write_text(report.text())
    (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
    print(f"cells    {csv_path}")
    print(f"report   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")
    print("These survive pod teardown when the results root is on the network "
          "volume, which is what the WRITES TO line above says.")

    # THE EXIT CODE COMES FROM THE SHARED TABLE, over the SAME gate objects that
    # printed the RESULT lines, so `exit_codes.classify_text` on this log
    # recomputes the code the process returned. Two integers meaning two
    # different things in two files is the defect that module is named against.
    #
    # NOTHING IS FOLDED INTO DONE ANY MORE, and `--fail-on-gate` is why this is
    # a paragraph and not a branch. Until 2026-09-02 a CLAIM_FAIL was described
    # in words and RETURNED AS 0 unless the flag was passed, so
    # `--self-test 0.85` printed a `RESULT: CLAIM ... FAIL` line that
    # `classify_text` reads as CLAIM_FAIL while the process said DONE -- the
    # exact log-versus-exit-code split the comment above claims cannot happen,
    # in the file that prints it. The masking was obsolete once the shared table
    # landed: CLAIM_FAIL (1) is in `FINISHED_CODES` and `ledger_state(1)` is
    # "CLAIM_FAIL", so 1 already tells the driver "this is a result, do not
    # retry it" and 0 protects nothing. The flag is accepted and ignored so an
    # old driver line still parses; `scripts/pod_session.sh` passes it and its
    # comment that the flag is REQUIRED is now belt and braces rather than the
    # thing that makes the code right.
    rc = exit_codes.classify(g.scored() for g in report.gates)
    print(f"exit     {exit_codes.describe(rc)}")
    if rc == exit_codes.CLAIM_FAIL:
        print("         a claim that did not pass is a RESULT and the arm is "
              "FINISHED, not broken.")
    return rc


def main(argv=None) -> int:
    """`_main`, with the two exits the interpreter would otherwise get wrong.

    A `SystemExit` CARRYING A STRING IS A REFUSAL. `raise SystemExit(<str>)`
    sets `SystemExit.code` to the string and the interpreter turns that into
    exit 1 -- CLAIM_FAIL, "measured; a pre-registered claim was refuted" -- so
    the missing `override_config` export, a refusal about the installed vLLM
    that measured nothing, exited with the same code a run that MEASURED and
    then failed a claim gate would have. The session driver cannot tell them
    apart, and this script's own contract says 2 means refused.

    Caught here rather than at every raise site so the contract holds for a
    caller of `main()` as well as for the CLI, and so a refusal added later
    cannot reintroduce the bug by forgetting the code.

    AN UNPLANNED CRASH IS ERROR, WHICH IS THE ONLY RETRYABLE CODE. Left to
    propagate, an unexpected exception exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it is in
    FINISHED_CODES, the driver records it, and it is never retried. A torch OOM
    or a truncated report would then be filed as one of this experiment's
    registered outcomes. ERROR (4) is outside FINISHED_CODES precisely so the
    driver can tell "the apparatus broke" from "the claim did not hold". The
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
        print("ERROR: block_m_crossing_sweep crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim failing, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: "
              "the traceback above is the thing to fix, and the arm may be "
              "re-run.", file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
