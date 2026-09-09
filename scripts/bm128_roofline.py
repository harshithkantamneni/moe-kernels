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

Three outcomes, and they are the study's fork:

  * throughput rises and then PLATEAUS below the roof, AND the control tile
    REACHES the roof -> the ceiling is real, it binds in the regime production
    actually runs, and the instrument has been shown able to see a tile arrive
    at the roof it is being said not to reach.
  * throughput REACHES the roof -> the ceiling is not binding, and the study's
    central claim is about a regime that does not occur.
  * throughput plateaus below the roof and the control beats it there, but the
    control does not reach the roof EITHER -> TILE-DEPENDENT GAP, CEILING
    UNLOCATED. A difference between two tiles is established and its height is
    not: something below the roof binds both of them, this measurement does not
    say what, and the study may quote the gap and not the ceiling. That third
    outcome is not hypothetical. No tile in this repository's published corpus
    reaches 0.95 of ridge x bandwidth: the BLOCK_M=256 control peaks at
    0.48-0.54 of the roof on the H200 and 0.56-0.64 on the A100, so on today's
    evidence this is the outcome the pod arm should expect.

WHY 128 AND NOT A --block-m FLAG. 128 is the only production-relevant regime.
From the one published arm that RECORDS the tile vLLM chose
(`2026-09-01-nvidia_h200-alpha-0558/merged.csv`, `tile_config_source` in
{vllm_default, vllm_tuned}, uniform routing only, a cell = one (model, tokens,
impl) over its seven seeds), counted on `load_max_rows`, which is what
`moe_align_block_size` actually pads to, and recounted 2026-09-03:

    BLOCK_M    cells run multi-tile     max M-tiles per expert
       16          1 of 24                      2
       32          0 of  5                      1
       64          2 of 16                      2
      128         66 of 87                     34

On `load_mean_rows` the same cells read 0 / 0 / 0 / 59 of 87 and 1 / 1 / 1 / 32,
which is the table this paragraph used to print as "16, 32 and 64 never run
multi-tile". The re-read term only exists when there is more than one M-tile
per expert. At 16, 32 and 64 it fires in isolated cells and never past two
tiles, so the caps computed for those tiles are real and barely approached; at
128 it runs up to 34 tiles at the busiest expert. SKEWED ROUTINGS WERE NEVER
COUNTED and production routing is skewed, so this is a statement about uniform
routing and nothing else. A `--block-m` flag would let a run answer a different
question under this script's name, so there is not one. The tile that IS a
parameter is the control, `--control`, and it is a parameter because which tile
can serve as a control depends on the card.

WHAT THE FIT SAYS ABOUT 128, AND WHY NO CAP IS QUOTED HERE. This paragraph used
to read "the cap sits near 150 FLOP/byte against a ridge near 163", computed as
2 BM / (alpha b) at alpha 0.85. Both halves are retracted, for two independent
reasons, and the retraction is the reason this script exists.

  * THE ESTIMATOR. Every alpha in this study comes from `LadderFit.alpha`, which
    is B/(A+B): the per-tile slope over the fitted level at n=1.
    `moe.bench.ai_model` writes out what that estimator returns over the real
    three-term traffic,

        alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                (EXA)

    with `phi` one M-tile's activation-and-output traffic in units of one full
    weight read and `delta` the fused layer's fixed cost in the same units. So
    2 BM / (alpha_fitted b) is NOT the cap: it is the cap multiplied by
    (1 + phi + delta) (`ai_model.lin_overstatement`), which at BLOCK_M=128 with
    BLOCK_SIZE_N=64 on mixtral is about a third too high. `ai_model.cap_from_
    fitted` is the corrected reading and it REFUSES a fitted alpha the three-term
    model cannot produce, which several of this study's published alphas are.
  * THE ALPHA. 0.85 is not what the ladders read, and no single figure is. At
    the swizzle this script pins by default (GROUP_SIZE_M=1) the identifiable
    direct ladders read 0.62 to 1.02 across models and cards, and NONE of them
    is at BLOCK_M=128, where no G=1 ladder was identifiable on either card
    (published SURFACE.txt of the s4 H200 and s3 A100 arms): mixtral reads
    0.95-1.02 on both cards, qwen2 0.72/0.71 on the H200 and 0.65/0.84 on the
    A100, deepseek-v2-lite 0.62. So the UNCORRECTED cap 2 BM / (alpha b) at
    128 is a BRACKET, 125 to 207 Op/B against the H200's own ridge of 162.8
    (`measured_nvidia_h200.yaml`): below it at mixtral's alphas, above it at
    qwen2's and deepseek's, and lower again by (1 + phi + delta) through (EXA).
    This paragraph used to say "0.92-1.02 on both cards, BELOW a ridge of
    163.7, not near it"; that was mixtral's number quoted for every model. At
    GROUP_SIZE_M=16 the same ladders give 0.63-0.66 and the uncorrected cap
    clears the ridge. The sign of the answer is a function of the model and the
    swizzle, which is what `--group-m` is for and why it is in the run id.

So this file quotes no cap and computes no verdict from one. It measures the
fraction of a MEASURED roof with a stopwatch, and the only place a fitted alpha
enters is choosing the control tile and printing a pre-registered band that is
scored against nothing.

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
at the SAME token counts, on the SAME card, in the SAME session: a tile with more
headroom than the subject, and ideally one whose AI cap is far above the ridge so
that it has no ceiling of its own to hit. THE SECOND HALF IS NOT AVAILABLE ON
THIS HARDWARE, which is the finding below and is why it is worded as an ideal
here. Without it
a plateau below the roof could be anything -- the fused layer's fixed cost, the
kernel's occupancy, the card's clocks. With it the study gets a DIFFERENCE, and
a difference cancels every term the two tiles share. If both tiles plateau at the
same fraction, the shortfall belongs to the layer and not to the tile, and that
is a result this script reports rather than a failure it hides.

AND THE CONTROL HAS TO ARRIVE, or the word "ceiling" is not earned. A difference
of 0.10 of the roof between two tiles that both stop at 0.5 says the tiles differ
and says nothing about where the ceiling is: the instrument has not been shown
able to see ANY tile reach the roof, so an absence at the subject is an absence
recorded by an instrument never shown to detect a presence. Gate C4 therefore
requires the control to reach ROOF_REACHED before the headline may be issued, and
without it the verdict is TILE-DEPENDENT GAP, CEILING UNLOCATED. This was added
on 2026-09-02: the gate set could previously declare the ceiling BINDING on the
gap alone, and no tile in the published corpus has ever reached 0.95 of ridge x
bandwidth, so the headline would have been published on a positive control that
never went positive.

THE ONE CONFOUND NO GATE HERE REMOVES, stated in every verdict this script can
reach and not only in the ones it embarrasses. The subject and the control differ
in OCCUPANCY as well as in tile: at the production pin of num_stages=4 the
resident-block ratio is 2:1, at 2 stages it is still 2:1, at 3 it is 3:1, and the
only matched setting is 5 stages, which sm_80 cannot run and which unpins the
comparison from every published arm. No `--num-stages` choice disambiguates a C3
result. `scripts/occupancy_vs_swizzle.py` measures the residency effect directly,
and its size is what has to be subtracted from any gap before the gap is read as
tile-attributable.

256 is not free: at BLOCK_SIZE_N=64 one CTA needs 4 x (256x64 + 64x64) x 2 B =
160 KiB of shared memory, which fits sm_90's 227 KiB and fits sm_80's 163 KiB by
three, and its 256x64 fp32 accumulator is 64 registers per thread against a
ceiling of 255. Both bills are computed from
`block_m_crossing_sweep.tile_resources` and REFUSED before a pod is rented,
because a spilled kernel still returns a time and that time still plots.

AND AT vLLM'S SHIPPED BLOCK_SIZE_N THERE IS NO CONTROL AT ALL. THE STUDY CANNOT
CONFIRM ITS HEADLINE AT ANY BLOCK_SIZE_N ON sm_90. This is the file's own
finding, 2026-09-03, and it is not a defect in the arm: it is the design meeting
the hardware. It was first written naming BLOCK_SIZE_N=256, which was true and
too weak; `ridge_reaching_tiles` computes the stronger form. Of 56 power-of-two
tiles up to 2048x1024, on both models this study measures and at every alpha it
has measured, NOT ONE whose cap clears the H200's ridge fits the per-block
register file: the smallest accumulator among them is 65536 registers at alpha
0.558, which is the whole file, and 131072 at alpha 1. C4 asks whether the
control reaches the roof, a tile reaches a compute roof only if its cap clears
the ridge, so C4 is pre-registered to fail for every control this hardware can
hold and CEILING BINDING is out of reach everywhere -- at 64 and 128 there is a
control that can never arrive, at 256 there is no control at all. The best
verdict any arm of this experiment can reach on sm_90 is TILE-DEPENDENT GAP,
CEILING UNLOCATED. The two limits very nearly coincide, and that coincidence is
the result: the register file runs out exactly where the arithmetic intensity
would have become enough. vLLM's tuned entry for mixtral and for qwen2 on the
H200 is BLOCK_SIZE_M=128 with BLOCK_SIZE_N=256 at every batch from 512 tokens up
(checked in `moe/bench/hardware/vllm_configs`, not assumed). At a FIXED
BLOCK_SIZE_N the only way to buy a tile headroom is a larger BLOCK_SIZE_M; the
only power of two above 128 Triton will pin is 256; and a 256x256 fp32
accumulator is 65536 32-bit registers, which is the ENTIRE per-block register
file of every architecture from sm_70 to sm_100. num_warps redistributes that
total across more threads and does not shrink it, num_stages and BLOCK_SIZE_K do
not touch it, and there is no card to move to. So four of the five escapes are
arithmetic dead ends and the fifth concedes the question:

  * a larger BM at FEWER STAGES -- stages are shared memory, they are not the
    accumulator, and 256x256 is 256 registers per thread at 8 warps at 1 stage
    exactly as at 4;
  * a larger BM at MORE WARPS -- 16 warps reads 128 registers per thread and
    passes the per-thread check, which is why `control_resource_hint` used to
    print "DOES FIT, at --num-warps 16 --num-stages 3"; the block still asks for
    all 65536 registers and has none left for a pointer;
  * a control with a BLOCK_SIZE_N OF ITS OWN, 128, so that 256x128 fits -- this
    one looks right and is the trap. `moe.bench.ai_model.cap` is
    2 / (b (alpha_b/BM + alpha_a/BN + 1/K)), SYMMETRIC in the two tile
    dimensions, so 256x128 has EXACTLY the subject 128x256's cap, 83.59 Op/B
    against 83.59, along with the same accumulator, the same shared memory and
    the same residency. It is the subject's own ceiling wearing a larger M, and
    `block_m_crossing_sweep.ai_cap` would have called it twice the headroom
    because that form is 2 BM / (alpha b) and BLOCK_SIZE_N is not in it;
  * the SUBJECT at BLOCK_SIZE_N=128, where a BLOCK_M=256 control does fit at 8
    warps and 4 stages -- buildable, and unlike the entry above it does buy
    headroom, 147.4 Op/B against the 128x128 subject's 111.6 at alpha 0.558. It
    is still not the escape: 147.4 is 0.91x the ridge, so that control is memory
    bound by construction, C4 fails before the run and the arm lands
    GAP_UNLOCATED. And vLLM ships no BLOCK_SIZE_M=128 entry at BLOCK_SIZE_N=128
    for either model, on either dtype, so it is a gap at a configuration nobody
    ships and it may not be quoted as the production claim. The 2026-09-03
    handoff called this "the one to schedule if the owner wants an arm that CAN
    reach BINDING"; it is not that arm, and no arm is;
  * `--control none`, the subject alone. THE ONLY ONE A SESSION CAN BUY AT
    BLOCK_SIZE_N=256, and what it buys is asymmetric: an uncontrolled run
    REACHING the roof kills the study's central claim outright, because a
    refutation needs no positive control -- the subject is one. An uncontrolled
    run stopping below the roof is a fraction of the roof and is NOT a ceiling,
    and the verdict it reaches says so in its own words (`UNCONTROLLED`), with
    C3 and C4 not scored, omitted rather than left UNKNOWN, so nothing
    downstream can read a gate that examined nothing as a gate that found no
    failure. IT IS REFUSED WHERE A CONTROL FITS: offered at BLOCK_SIZE_N=64,
    where the search names eighteen pins, it would let an operator drop three
    gates by choice and receive a page saying the register file forced it.

AND THE UNCONTROLLED VERDICT IS A SCORED GATE, not a paragraph. The session
driver summarises an arm by grepping `^RESULT: ` and nothing else. Until this
gate existed, `--control none` printed six RESULT lines, every one PASS, and
exited 0 DONE: a BINDING run and an arm that could attribute nothing differed,
in the summary, by two MISSING lines, and the word UNCONTROLLED lived only in
the report body. `gate_cu_uncontrolled` is the one line that carries it, it can
never PASS, and the arm exits CLAIM_FAIL -- measured, VALIDITY clean, a
registered claim not established, which is a result and never a retry.

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
sweep manufactures a plateau. Every cell therefore carries the SM clock sampled
UNDER LOAD by `moe.bench.timing.time_kernel`, and the two verdicts that clock
supports: LEVEL (the loaded clock is inside the band 95% to 105% of the clock
the ROOF was measured at, and `clock_level_side` says which way it left it) and
DRIFT (the first and last under-load samples agree within 5%). A cell that
DRIFTED or that failed LEVEL on the LOW side is EXCLUDED from every gate and
printed with an x on the plot. A cell that failed LEVEL on the HIGH side is
KEPT: on the H200 that is the normal state of a memory-shaped cell, 1980 MHz
under memory load against the 1515 MHz bf16 GEMM the roof was measured at, its
milliseconds are a measurement, and what the fixed roof does not describe is
its FRACTION of that roof, so every point also carries the fraction of the roof
at its own clock (`roofline.roof_at_clock`) and V3 is scored on that. The
exclusions are counted, and a run that excluded so much that the doubling chain
no longer spans the required distance says so instead of scoring.

WHY UNDER LOAD, AND WHY AGAINST THE ROOF'S CLOCK. Until 2026-09-02 this file
sampled the clock with `ClockState.sample()` immediately after a synchronise --
an IDLE INSTANT -- took the session's modal value of those samples, and compared
it to the roof's under-load 1515 MHz within 10%. The H200 idles at 1980 and runs
a dense GEMM at about 1515, so the verdict depended on which of the two modes
the post-sync sample happened to land in: an idle-instant session reads 1980,
which is +31% against the roof's clock and FAILS a 10% gate that nothing was
wrong with, while a genuinely cold session that also idles high PASSES. Two
operating points were being compared through a sample taken at neither. The
instrument now polls the clock on a background thread WHILE the kernels run and
reports the median of those samples, and `timing.clock_flags` scores it against
the clock the calibration's own dense GEMM ran at. Both numbers are then under
load and the comparison means something.

AND THE CLOCK THE ROOF WAS MEASURED AT IS PART OF THE ROOF. The H200's dense
bf16 ceiling was taken at 1515 MHz -- a dense GEMM pulls the clock down from
1980 -- so a MoE layer running at 1800 MHz under load is being compared against
a measurement made at a different operating point. The direction is stated rather
than corrected: a run clocked ABOVE the roof's clock has its fraction
OVERSTATED, which is conservative for a claim that the fraction stays below 1;
a run clocked BELOW it has the fraction understated, and a plateau found there
could be the clocks. V3 scores it.

WHAT THE FIT IS STILL USED FOR, exactly and only: DISCLOSING what each candidate
control's ceiling is, refusing a control with no headroom on the subject, and
PRE-REGISTERING a predicted plateau band. It used to CHOOSE the control by
requiring its cap to clear the ridge by 30%, and that check ran on
`SWEEP.ai_cap` = 2 BM / (alpha b) -- the scalar form this file retracts, which
overstates a 256x256 tile by 3.7x. Priced through `moe.bench.ai_model.cap` no
buildable tile clears the ridge at all, so the check would have refused every
geometry: a gate that can only fail decides as little as one that can only pass.
It is a disclosure now, and `symmetric_cap` is the one cap any decision reads.
No verdict below is computed from alpha, and the predicted band is wide on
purpose -- alpha at BLOCK_M=128 and GROUP_SIZE_M=1 has been measured anywhere
from 0.625 to 1.02 across this study's arms, which puts the predicted plateau
between 0.77 and 1.00 of the roof. A prediction that spans a quarter of the
range it is predicting is not a prediction, and that width IS the reason this
experiment exists.

A NOTE ON REUSE. The geometry, the pinned constants, the cell model, the compile
assay, the cost model and the resource bill are IMPORTED from
`scripts/block_m_crossing_sweep.py` and called, never copied: this measurement
has to be commensurable with the sweep the study publishes, and a private copy
would drift. `_load_sweep` probes SIGNATURES as well as names, because that file
is under active edit and a renamed argument must produce a sentence on a laptop
rather than a TypeError two minutes into a metered pod session. The TIMING comes
from one level lower still: `moe.bench.timing.time_kernel`, the same queue-deep,
L2-flushing, clock-sampling loop the compute roof itself was measured with. It
replaced a private per-iteration-synchronise loop (`SWEEP.time_call`, retired the
same day) that created its events inside the loop and let 0.18-0.30 ms of host
enqueue time per call inside the measured interval -- a per-card bias, of the
same order as the cross-card effect this study registered, in a number that is
divided by a roof measured the other way. Every cell records the instrument's own
name, its warmup duration, its iteration and trial counts, its flush state and
its under-load clock verdicts, because those columns are what make a row
comparable with the roof or not.

THE EXIT CODE IS THE CONTRACT. `moe.bench.exit_codes` owns the table: 0 DONE, 1
CLAIM_FAIL (measured, a pre-registered claim did not hold -- a RESULT, never a
retry), 2 REFUSED (nothing measured, nothing spent), 3 INVALID (measured and a
VALIDITY gate failed, so nothing on the page may be quoted), 4 ERROR. Every
scored gate prints exactly one `RESULT: KIND NAME VERDICT detail` line at column
zero and nothing else here starts with that prefix, so a driver reads the gates
without reading prose.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import ai_model as AI  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.roofline import HARDWARE_DIR, roof_at_clock  # noqa: E402
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
    # `time_call` is deliberately NOT here. It was retired on 2026-09-02 and
    # this script no longer calls it; requiring the symbol would tie a laptop
    # `--dry-run` to a name the sweep keeps only so its three other consumers
    # abort on their first timed cell rather than at import.
    needed = ("FIXED", "COMPUTE_BOUND_FRACTION", "ALPHA", "ALPHA_BY_BLOCK_M",
              "PASS", "FAIL", "UNDECIDED", "ai_cap", "make_cell", "model_ms",
              "tokens_for_rows", "rows_quantum", "rows_step", "results_root",
              "planned_iters", "estimated_seconds", "useful_flops",
              "tiles_per_expert", "timing_basis", "reference_clock_mhz",
              "SYNTHETIC_INSTRUMENT",
              "find_override", "arm_triton_cache", "count_new",
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
    for name, required in (("make_cell", ("sm_count", "block_n", "instrument",
                                          "warmup_ms", "trials",
                                          "sm_clock_load_mhz", "clock_level_ok",
                                          "clock_drift_ok", "clock_level_side",
                                          "l2_flush")),
                           ("model_ms", ("alpha", "ridge", "bandwidth_gbps")),
                           ("estimated_seconds", ("warmup_ms", "trials",
                                                  "cell_budget_ms")),
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

#: How far above the ridge a control's cap would have to sit for the control to
#: have NO ceiling of its own -- the tile the design asks for, and the one this
#: hardware cannot supply.
#:
#: IT IS A DISCLOSURE AND NO LONGER A REFUSAL, changed 2026-09-03. The comment
#: here used to read "1.30 is not tuned: at the pooled alpha the BLOCK_M=256 cap
#: is 2.8x the H200 ridge", and that 2.8x came from `SWEEP.ai_cap` = 2 BM /
#: (alpha b), the scalar form this file retracts: under `symmetric_cap` the same
#: tile is 0.55x at BLOCK_SIZE_N=64 and 1.33x at 256, where it cannot be built.
#: Priced correctly no tile a thread block can hold clears the ridge at all, so
#: a refusal on this number would refuse every geometry -- a gate that can only
#: fail, which decides as little as one that can only pass. `check_control`
#: therefore refuses on the register file and on headroom over the subject, and
#: this constant names, in the plan, the distance between the control the design
#: wants and the control the card allows.
CONTROL_CAP_MARGIN = 1.30

#: The spelling `--control none` takes, and the one the run id carries for it.
#: A word and not an empty string, because `moe.bench.provenance.run_id` refuses
#: an empty value and because an id reading `c-none` says which arm this was.
CONTROL_NONE = "none"

#: 32-bit registers ONE THREAD BLOCK may use. The CUDA technical specifications
#: list 64 K per SM and 64 K per block alike for every compute capability in
#: `block_m_crossing_sweep.SMEM_PER_BLOCK_BYTES`, 7.0 through 10.0, so this is
#: not a per-card number and there is no opt-in past it.
#:
#: WHY IT LIVES HERE AND NOT IN THE SWEEP'S RESOURCE MODEL. The sweep bills the
#: accumulator PER THREAD -- `BM BN / (32 num_warps)` against 255 -- which is the
#: limit that binds at num_warps=8 and DISSOLVES as warps are added: sixteen
#: warps halve the per-thread count and the same accumulator passes. The
#: per-block total does not move with warps at all, because a `BM x BN` fp32
#: accumulator is `BM x BN` registers however many threads hold them. Billing
#: only per thread is what let `control_resource_hint` print "THE REQUESTED
#: CONFIGURATION DOES FIT, at --num-warps 16 --num-stages 3" on this laptop for
#: a 256x256 accumulator that is 65536 registers -- the whole block-level file,
#: with nothing left for a pointer, an index or the K loop. That was the remedy
#: the pod would have rejected, stated unqualified, in the one sentence an
#: operator was meant to act on. Choosing the control is this file's job, so the
#: arithmetic that decides it is this file's too.
REGISTERS_PER_BLOCK = 65536

#: Tile heights a control may be searched over. Triton pins block shapes to
#: powers of two, so 128's successors are these three and there is no 192 to
#: reach for when 256 does not fit.
CONTROL_CANDIDATE_BLOCK_M = (256, 512, 1024)

#: Warp and stage counts the search will try, widest first for the warps
#: because more warps is the only knob that moves the PER-THREAD half of the
#: register bill, and deepest first for the stages because fewer stages is the
#: only knob that moves shared memory.
CONTROL_WARP_COUNTS = (8, 16, 32)
CONTROL_STAGE_COUNTS = (5, 4, 3, 2)

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

#: Percent disagreement between the FIRST and LAST under-load SM clock samples
#: of a cell that marks it throttled. Mirrored from
#: `moe.bench.timing.DRIFT_FRACTION` rather than imported, because that module
#: imports torch at module scope and this file is documented to plan a run on a
#: laptop that has none; `test_the_mirrored_clock_constants_are_the_instruments_
#: own` asserts the two against it wherever torch imports. The verdict itself is
#: the instrument's `clock_drift_ok`, not a comparison recomputed here: this
#: number exists so the printed sentence can say what the instrument used.
THROTTLE_DRIFT_PCT = 5.0

#: How far below the clock the ROOF WAS MEASURED AT a cell's under-load clock
#: may sit and still be read. Mirrored from `moe.bench.timing.LEVEL_FRACTION`,
#: and the reference moved on 2026-09-02 from the session's own modal clock to
#: the roof's: a session that ran uniformly cold has a cold modal clock and
#: excludes nothing, which is exactly the session whose fractions are wrong. The
#: verdict is the instrument's `clock_level_ok`, and since 2026-09-03 it is
#: TWO-SIDED: the band has a ceiling too, below, and the instrument's
#: `clock_level_side` says which edge a failing cell crossed.
#:
#: NEITHER EDGE EXCLUDES SINCE 2026-09-09. This is a RECORDING band: it says
#: which way a cell's fixed-roof fraction is off and by how much, and DRIFT is
#: the only exclusion. The floor excluded until that date, and on the H200 what
#: it excluded was every multi-tile BLOCK_M=128 cell of this arm at 1380-1410
#: MHz -- the study's own subject -- because on a 700 W-capped card the
#: under-load clock is set per tile by the kernel's own power draw and a dense
#: BM=128/BN=64 tile holds 1395 MHz against a 1485 MHz calibration GEMM. Five
#: of those cells were low by 0.75 MHz, half of one 15 MHz NVML step.
CLOCK_FLOOR_FRACTION = 0.95

#: How far ABOVE the roof's clock a cell may sit before its fixed-roof fraction
#: is flagged as inflated. Mirrored from `moe.bench.timing.LEVEL_HIGH_FRACTION`
#: for the same reason as the floor. A cell over this edge is `Timing.boosted`:
#: it is KEPT in every gate, because its time is a measurement, and its
#: fraction of the roof at its own clock (`roofline.roof_at_clock`) is printed
#: beside the fixed one as the issue efficiency. THE GATES STILL READ THE FIXED
#: ROOF: the calibration GEMM and every cell here ran under the same 700 W cap,
#: so delivered throughput against one roof is the comparison that cap makes
#: fair, and the rescaled column answers the other question rather than
#: replacing the first. On the H200 a memory-shaped cell at 1980 MHz against a
#: 1515 MHz reference is 1.31x over this edge and is the NORMAL state; a file
#: that mirrored only the floor could not say so.
CLOCK_CEILING_FRACTION = 1.05

#: Fraction of this session's cells whose clock may MOVE mid-measurement before
#: the session itself is unreadable. A single drifted cell is EXCLUDED and the
#: run goes on -- that is what the exclusion machinery is for, and killing a
#: 39-cell run over one of them would be a gate that fails on weather. A tenth
#: of them drifting is not weather: it is a warmup that starts timing before
#: the governor has settled, so the surviving medians were taken by the same
#: loop and are not trustworthy either. That is a refusal, and it points at the
#: instrument. UNCHANGED AT 0.10 on 2026-09-09 when the LEVEL sides came out of
#: the share: the 2026-09-09 arm sits at 4 of 39 = 10.3% and still FAILS here,
#: which is the honest reading of a pre-v4 instrument and not a bar to move.
THROTTLED_CELL_FRACTION = 0.10

#: The alphas this study has actually fitted with GROUP_SIZE_M=1, as a band.
#: The low end is the published surface's own 128 row (0.625, fitted at G=1 on
#: the 2026-09-01 alpha-0558 arm); the high end is the largest G=1 ladder alpha
#: on the published surfaces, mixtral BLOCK_M=32 at 1.016 (H200 s4) and 1.018
#: (A100 s3). No BLOCK_M=128 ladder at G=1 was identifiable on either surface,
#: so this is a bracket over other tile heights, not a measurement at 128. The
#: band is carried rather than a point estimate because the PREDICTED plateau
#: it implies spans 0.77 to 1.00 of the roof, and a prediction that wide is the
#: reason this experiment measures instead of computing.
ALPHA_128_BAND = (SWEEP.ALPHA_BY_BLOCK_M.get(SUBJECT_BLOCK_M, SWEEP.ALPHA), 1.02)

#: THE NOISE THIS ARM'S PLAN IS SIZED AGAINST, as a relative spread on one
#: cell's milliseconds. 0.0140 is `timing_spread_median` from the one published
#: arm that recorded one, `results/published/2026-09-01-nvidia_h200-alpha-
#: surface-s4/mixtral-8x7b-bf16-r1024-g1-n64-d66ad3.report.json`, measured with
#: the RETIRED instrument -- so it is an upper bound on what `time_kernel`'s
#: queue-deep loop should produce, which is the safe direction for sizing.
#: `--plan-noise` overrides it, and the plan prints which number it used.
#: It sizes an MDE and nothing else: every verdict below uses the spread this
#: run actually measured, and gate C2 refuses when that spread cannot resolve
#: the threshold it is asked to score.
DEFAULT_PLAN_NOISE_REL = 0.0140

#: Two-sided 5% at 80% power, the convention `scripts/replicate_noise_floor.py`
#: uses for every MDE in this study. Named here so the plan line can say what
#: test it is quoting rather than printing a number with no design attached.
MDE_LEVEL, MDE_POWER = 0.05, 0.80

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


def instrument_name() -> str:
    """The name of the loop that will time these cells, or why it has none here.

    `SWEEP.timing_basis()` returns `moe.bench.timing.TIMING_BASIS` when torch
    imports and None when it does not, which is exactly the laptop `--dry-run`
    and `--self-test` case where nothing is measured either. A sentence rather
    than a blank, because a plan that printed an empty instrument would read as
    a plan whose instrument nobody wrote down, and this repository has already
    published 26 reports of that shape.
    """
    return SWEEP.timing_basis() or (
        "NOT NAMEABLE HERE: torch does not import on this machine, so the "
        "instrument cannot be named. A pod names it on every row it writes")


def _load_power():
    """`scripts/replicate_noise_floor.py`'s power arithmetic, loaded by path.

    ADOPTED rather than reimplemented, for the reason `gate_v1_pin` adopts the
    sweep's override assay: this repository has exactly one definition of what a
    minimum detectable effect is, that file owns it, and a second t quantile
    written here would be a second thing to keep in step. It is stdlib-only, so
    loading it costs a laptop nothing.

    REFUSES rather than falling back on a normal approximation. An MDE quoted
    from a z where the design has 4 degrees of freedom is 35% too generous, and
    the whole point of printing one is to say what this run CANNOT resolve.
    """
    spec = importlib.util.spec_from_file_location(
        "replicate_noise_floor", ROOT / "scripts" / "replicate_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(
            "REFUSED: scripts/replicate_noise_floor.py, which owns this "
            f"study's power arithmetic, did not import ({type(exc).__name__}: "
            f"{exc}). The plan's MDE line comes from its two-sample t and there "
            "is no approximation of it this file is allowed to substitute: a "
            "normal quantile at these degrees of freedom is 35% too generous "
            "and would advertise a resolution this run does not have.") from exc
    missing = [n for n in ("mde_two_sample", "replicates_for") if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "REFUSED: scripts/replicate_noise_floor.py no longer exports "
            f"{', '.join(missing)}, so the MDE this plan prints cannot be "
            "computed from the study's own definition. Re-point the call; do "
            "not inline a second one.")
    return module


def mde_lines(*, reps: int, noise_rel: float, noise_source: str,
              gap_threshold: float = CONTROL_SEPARATION,
              gain_threshold: float = PLATEAU_GAIN_PER_DOUBLING,
              plateau_fraction: float = 0.5) -> list[str]:
    """What this run could resolve, from ONE stated noise assumption. B14.

    The audit's finding was that no arm in this study stated an MDE, so every
    threshold in every gate was a prior with nothing behind it and a reader
    could not tell a gate this run can settle from one it cannot. Two thresholds
    here are worth sizing before a pod is rented:

      * GATE C3's separation, `CONTROL_SEPARATION` of the roof. Two tiles, each
        the median of `--reps` repeats at one batch, compared as a difference:
        a two-sample t at n = reps per condition on a standard deviation of
        `noise_rel * plateau_fraction` of the roof, because a relative spread on
        milliseconds is the same relative spread on a throughput and the
        fractions this compares sit near `plateau_fraction`.
      * GATE C2's gain per doubling, `PLATEAU_GAIN_PER_DOUBLING`. That is a
        RATIO of two per-tread medians, so its spread is
        `noise_rel * sqrt(2) / sqrt(reps)` -- the same arithmetic `plateau_of`
        applies to the spread this run actually measures, quoted here in advance
        against the assumption instead.

    Both are printed with the verdict RESOLVABLE or NOT RESOLVABLE and, when not,
    the `--reps` that would be. Neither is a gate: the gates score the measured
    spread, and C2 already refuses when it cannot resolve its own threshold.
    """
    power = _load_power()
    sd = noise_rel * plateau_fraction
    out = [f"MDE          noise assumption {noise_rel:.2%} relative spread per "
           f"cell, {noise_source}",
           f"             two-sided {MDE_LEVEL:.0%} at {MDE_POWER:.0%} power, "
           "the convention scripts/replicate_noise_floor.py uses"]
    if reps < 2:
        out.append(f"             at --reps {reps} there is no within-cell "
                   "spread and no two-sample test exists, so NOTHING here is "
                   "resolvable and C2 will refuse. Raise --reps.")
        return out
    gap_mde = power.mde_two_sample(sd, reps, level=MDE_LEVEL, power=MDE_POWER)
    need = power.replicates_for(gap_threshold, sd, level=MDE_LEVEL, power=MDE_POWER)
    out.append(
        f"             C3 gap: at --reps {reps} the MDE on the control-minus-"
        f"subject difference is {gap_mde:.4f} of the roof against a gate of "
        f"{gap_threshold:.2f}: "
        + ("RESOLVABLE." if gap_mde <= gap_threshold else
           "NOT RESOLVABLE -- "
           + (f"--reps {need} would be." if need else
              "no --reps under 500 would be, at this spread.")))
    ratio_sd = noise_rel * math.sqrt(2.0) / math.sqrt(reps)
    out.append(
        f"             C2 gain: the ratio of two per-tread medians carries "
        f"{ratio_sd:.3%} at --reps {reps} against a gate of "
        f"{gain_threshold:.1%}: "
        + ("RESOLVABLE." if ratio_sd <= gain_threshold else
           f"NOT RESOLVABLE -- --reps "
           f"{math.ceil(2.0 * (noise_rel / gain_threshold) ** 2)} would be.")
        + " Scored on the spread this run MEASURES, not on this assumption.")
    return out


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

    AND IT KEEPS `SWEEP.ai_cap`, DELIBERATELY, where `control_feasibility` and
    `check_control` were moved off it on 2026-09-03. Those two DECIDE things and
    a decision made on a retracted model is a defect. This one records what the
    study predicted, in the arithmetic the study predicted it with; re-deriving
    a pre-registration under a better model after the fact is not a correction,
    it is a rewritten prediction. The band is compared against nothing, and
    `symmetric_cap` is where a reader should go for a cap that is used.
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
# What the arm ALREADY published for the configuration being asked for.
# --------------------------------------------------------------------------

#: Where `published_prediction` looks. Every committed arm writes its
#: `<model>-<dtype>-r<r>-g<G>-n<BN>-<hash>.report.json` under a dated,
#: card-slugged directory, and the ladder inside carries `(n_tiles, ms)` per
#: BLOCK_M -- which is a throughput once it is divided by a roof.
PUBLISHED_DIR = ROOT / "results" / "published"


@dataclass(frozen=True)
class PriorArm:
    """One published ladder pair, read back as roof fractions. R2 / audit A8.

    THE POINT IS TO MAKE AN ARM PREDICTABLE BEFORE IT IS RENTED. The audit's
    finding was that the scheduled headline run at BLOCK_SIZE_N=64,
    GROUP_SIZE_M=1 has an outcome already computable from arms in this
    repository -- subject 0.468, control 0.526, a gap of 0.058 under a
    CONTROL_SEPARATION of 0.10, so "PLATEAU IS NOT TILE-ATTRIBUTABLE" -- and
    that no output said so. A run whose result is known is not an experiment; it
    is a confirmation, and the operator is entitled to read that off the plan
    rather than off the invoice.
    """

    path: str
    fixed: dict
    subject_peak: float | None
    control_peak: float | None
    roof_tflops: float
    control_block_m: int

    @property
    def gap(self) -> float | None:
        if self.subject_peak is None or self.control_peak is None:
            return None
        return self.control_peak - self.subject_peak

    def outcome(self) -> str:
        """The verdict these numbers would produce, in this script's own words."""
        if self.gap is None:
            return UNSETTLED
        if self.subject_peak is not None and self.subject_peak >= ROOF_REACHED:
            return NOT_BINDING
        if self.gap < CONTROL_SEPARATION:
            return NOT_TILE
        if self.control_peak is not None and self.control_peak < ROOF_REACHED:
            return GAP_UNLOCATED
        return BINDING


def _ladder_peak(ladder: dict, cfg, block_m: int, key: str) -> float | None:
    """Best useful TFLOP/s on one published ladder, or None when it has none.

    The ladder is `(n_tiles, ms)` at exactly-full stacks, so rows per expert is
    `n * BLOCK_M` and the flop count is the sweep's own `useful_flops` over
    `E * rows`. Nothing is inferred: a BLOCK_M the arm did not run has no entry
    and this returns None rather than a zero, because a zero would plot.
    """
    entry = ladder.get(key)
    points = (entry or {}).get("points") or []
    best = None
    for item in points:
        try:
            tiles, ms = int(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if ms <= 0 or tiles <= 0:
            continue
        rows = tiles * block_m
        tflops = SWEEP.useful_flops(cfg, cfg.num_experts * rows) / (ms * 1e-3) / 1e12
        best = tflops if best is None else max(best, tflops)
    return best


def roof_card_slug(roof: Roof) -> str:
    """The card the roof belongs to, as the published directories spell it.

    `load_hardware` names a calibrated card `"NVIDIA H200 (measured)"`, and the
    directories are slugged from the driver's `"NVIDIA H200"`. The suffix is
    stripped here rather than the slug being taken from `--card`, because the
    condition that has to hold is that the ladders and the denominator come from
    ONE card: matching arms to the roof itself is that condition, and matching
    them to a flag is a different and weaker one.
    """
    return PV.card_slug(roof.device).removesuffix("_measured")


def published_prediction(cfg, roof: Roof, *, block_n: int, group_m: int,
                         control_block_m: int | None,
                         published_dir: Path | None = None) -> tuple[PriorArm | None, str]:
    """The outcome this configuration's own published arms already imply.

    Returns `(arm, why)`. `arm` is None when nothing in the corpus matches, and
    `why` ALWAYS says what was looked for and what was found, because "it cannot"
    is a legitimate and common answer here: the corpus has no BLOCK_M=256 ladder
    at BLOCK_SIZE_N=256 at all, since that tile cannot be pinned there (its
    256x256 fp32 accumulator needs 256 registers per thread against a hardware
    maximum of 255), and an operator asking for the production swizzle deserves
    to be told that the control's half of the prediction does not exist rather
    than to be shown a half table.

    Matched on model, BLOCK_SIZE_N, GROUP_SIZE_M and the CARD -- the card
    because every number here is a fraction of a per-card roof, and dividing an
    A100 ladder by an H200 roof is the exact defect that put a stale ridge into
    seven published reports.

    SEVERAL ARMS USUALLY MATCH, AND THE ONE PICKED IS THE LEAST FLATTERING TO
    THE STUDY. The scheduled headline configuration matches TWO committed H200
    arms today, alpha-surface-s4 (0.468 / 0.526, gap 0.058) and cross-card-s3
    (0.482 / 0.502, gap 0.019). The selection rule is `max` on
    `(gap is not None, subject_peak)`: an arm with BOTH ladders always beats one
    with a hole, and among those the HIGHEST subject peak wins. That second half
    is a deliberate bias and not a tie-break. When the controls sit close, the
    highest subject peak is the SMALLEST gap, so the arm named is the one LEAST
    likely to predict a tile effect, and a plan that says "not worth renting"
    says it on the evidence most hostile to that conclusion. Both of today's
    arms predict NOT_TILE, so nothing turns on the choice yet; a later corpus
    need not agree, which is why the count and the SPAN of the gaps go into
    `why` rather than only the winner.
    """
    directory = published_dir or PUBLISHED_DIR
    slug = roof_card_slug(roof)
    want = (f"model={cfg.name}, BLOCK_SIZE_N={block_n}, GROUP_SIZE_M={group_m}, "
            f"card={slug}")
    if not directory.is_dir():
        return None, f"no published corpus at {directory} to predict {want} from"
    hits: list[PriorArm] = []
    scanned = 0
    for path in sorted(directory.glob(f"*{slug}*/*.report.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        scanned += 1
        fixed = doc.get("fixed") or {}
        if doc.get("model") != cfg.name:
            continue
        if fixed.get("BLOCK_SIZE_N") != block_n or fixed.get("GROUP_SIZE_M") != group_m:
            continue
        ladder = doc.get("ladder") or {}
        hits.append(PriorArm(
            path=_corpus_path(path), fixed=fixed,
            subject_peak=_frac(_ladder_peak(ladder, cfg, SUBJECT_BLOCK_M,
                                            str(SUBJECT_BLOCK_M)), roof),
            control_peak=None if control_block_m is None else
            _frac(_ladder_peak(ladder, cfg, control_block_m,
                               str(control_block_m)), roof),
            roof_tflops=roof.tflops, control_block_m=control_block_m))
    if not hits:
        return None, (f"no published arm matches {want} ({scanned} report(s) "
                      "read on this card), so this configuration's outcome "
                      "cannot be predicted from the corpus and the run is a "
                      "real experiment")
    # Documented in this function's docstring: complete arms beat holed ones,
    # and among the complete ones the highest subject peak wins BECAUSE it is
    # the smallest gap, i.e. the reading least favourable to a tile effect.
    arm = max(hits, key=lambda a: (a.gap is not None, a.subject_peak or 0.0))
    among = _selection_phrase(hits, want)
    if control_block_m is None:
        return arm, (f"{arm.path}, {among}. --control none was given, so there "
                     "is no CONTROL half to predict and no outcome to predict "
                     "either; only the subject's peak is comparable")
    if arm.control_peak is None:
        return arm, (f"{arm.path}, {among}, has no BLOCK_M={control_block_m} "
                     "ladder, so the CONTROL half of the prediction does not "
                     "exist and the outcome cannot be predicted; only the "
                     "subject's peak is known")
    return arm, f"predicted from {arm.path}, {among}"


def _corpus_path(path: Path) -> str:
    """A published report as the repository spells it, or as it actually is.

    `relative_to(ROOT)` RAISES on a path outside the tree, and `published_dir=`
    is a parameter exactly so a test or an operator can point the prediction at
    a corpus kept elsewhere. Losing a whole plan to a ValueError raised while
    formatting the path it was about to quote is a cosmetic call taking down a
    real one, so the absolute path is the fallback rather than the crash.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _selection_phrase(hits: list[PriorArm], want: str) -> str:
    """How many arms matched, over what span of gaps, and why THIS one.

    Exists because "the published arm at <want>" read as a uniqueness claim in
    the one plan line an operator uses to decide an arm is not worth renting,
    while two arms matched and their gaps differed by 3x. Silence about the
    other arm is the failure this prevents: the span is printed so a reader can
    see the disagreement, and the rule is named so they can see it is the
    conservative end of it.
    """
    if len(hits) == 1:
        return f"the ONE published arm at {want}"
    gaps = [a.gap for a in hits if a.gap is not None]
    span = (f"gaps {min(gaps):+.3f} to {max(gaps):+.3f}" if gaps
            else "no arm among them carries both ladders")
    return (f"the arm with the HIGHEST subject peak of {len(hits)} matching "
            f"{want} ({span}); the highest subject peak is the SMALLEST gap "
            "when the controls sit close, so this is the arm least likely to "
            "predict a tile effect")


def prior_arm_lines(cfg, roof: Roof, *, block_n: int, group_m: int,
                    control_block_m: int | None,
                    published_dir: Path | None = None) -> list[str]:
    """The plan's PREDICTED OUTCOME section: what the corpus already says.

    Printed before a pod is rented, because an arm whose result is computable
    from committed data is a confirmation and not an experiment, and the
    operator is entitled to decide which one they are buying. When the corpus
    cannot answer -- no arm at this swizzle, or an arm with no control ladder --
    that is said in the same place, in those words.
    """
    arm, why = published_prediction(cfg, roof, block_n=block_n,
                                    group_m=group_m,
                                    control_block_m=control_block_m,
                                    published_dir=published_dir)
    out = ["PREDICTION   from arms already in this repository, at "
           f"BLOCK_SIZE_N={block_n}, GROUP_SIZE_M={group_m}, "
           f"{roof_card_slug(roof)}",
           f"             {why}"]
    if arm is None:
        out.append("             THIS ARM IS NOT PREDICTABLE FROM THE CORPUS. "
                   "That is the good case: it is an experiment.")
        return out
    sub = ("not measured" if arm.subject_peak is None
           else f"{arm.subject_peak:.3f} of the roof")
    ctl = ("NOT MEASURED -- the corpus has no ladder for this tile at this "
           "BLOCK_SIZE_N" if arm.control_peak is None
           else f"{arm.control_peak:.3f} of the roof")
    out.append(f"             BLOCK_M={SUBJECT_BLOCK_M} peaked at {sub}"
               + ("; this arm has no control" if control_block_m is None
                  else f"; BLOCK_M={control_block_m} {ctl}"))
    if arm.gap is None:
        out.append("             so the gap, and with it the outcome, cannot "
                   "be predicted. Only the subject's half is known.")
        return out
    out.append(f"             gap {arm.gap:+.3f} against a "
               f"CONTROL_SEPARATION of {CONTROL_SEPARATION:.2f} and a "
               f"ROOF_REACHED of {ROOF_REACHED:.2f}: this run's expected "
               f"verdict is {VERDICT_NAMES[arm.outcome()]} "
               f"({arm.outcome()}).")
    out.append("             A PREDICTION AND NOT A RESULT. It is computed "
               "from ladders timed by the RETIRED per-call instrument, so it "
               "is what the corpus says, not what this run will measure.")
    return out


def _frac(tflops: float | None, roof: Roof) -> float | None:
    """A TFLOP/s as a fraction of this roof, or None for a ladder that has none."""
    if tflops is None or roof.tflops <= 0:
        return None
    return tflops / roof.tflops


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

    THE STATE THE ROW WAS TIMED IN IS A COLUMN, exactly as it is on the sweep's
    `Cell`. `instrument`, `warmup_ms`, `iters`, `trials`, `sm_clock_load_mhz`,
    `clock_level_ok`, `clock_level_side`, `clock_drift_ok` and `l2_flush` are
    what `moe.bench.timing.time_kernel` reports about the measurement it just made,
    and they are written per row because they are what makes a row comparable
    with the roof or not. Before 2026-09-02 this file carried an iteration count
    and a pair of idle-instant clock samples, and a reader could not tell a cell
    timed at 1980 MHz from one timed at 1500.

    THE THREE CLOCK VERDICTS ARE OPTIONAL AND None MEANS "NOT DETERMINED", never
    "fine". A container without NVML, a trial too short for the poller to land a
    sample and a replayed CSV all produce None, and a filter that read None as
    True would re-admit exactly the rows these columns exist to keep out.

    AND A FAILED LEVEL HAS A SIDE, WHICH IS RECORDED AND EXCLUDES NOTHING.
    `clock_level_side` is "low", "high" or "" (passed, or not determined), the
    instrument's own word. `cold` is the LOW side and `boosted` the HIGH side;
    since 2026-09-09 both are KEPT, with the row's fixed-roof fraction flagged
    as off by the clock ratio in the named direction, and `throttled` (DRIFT)
    is the only exclusion. `SWEEP.check_level_side`
    refuses, at construction, a side on a verdict that did not fail AND a
    failed verdict with no side: the instrument derives the verdict from the
    side, so the second shape is a caller that dropped the side, and reading
    its blank as LOW (the pre-2026-09-08 rule) is what excluded every boosted
    cell.
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
    #: Median of the SM clock samples taken WHILE this cell's kernels ran. None
    #: is "no usable sample", which is not zero and not a pass.
    sm_clock_load_mhz: float | None = None
    #: First and last under-load samples, which is what DRIFT is computed over.
    sm_clock_start_mhz: float | None = None
    sm_clock_end_mhz: float | None = None
    #: EVERY under-load sample, space separated, or "" when the instrument
    #: reported none. The first and last decide DRIFT and this is the shape
    #: they came from: without it a drifted row cannot say whether the clock
    #: ramped once and settled (the governor after a workload change, which the
    #: warmup can absorb) or oscillated throughout (which it cannot). Written
    #: as one column rather than parsed into a list, because a CSV cell holds
    #: text and a reader that has to trust a parser is a reader who cannot
    #: check. Empty on every row written before the instrument kept the list.
    clock_samples_mhz: str = ""
    #: Milliseconds the warmup spent WAITING FOR THE CLOCK TO SETTLE beyond
    #: `warmup_ms`, and how many extra warm calls that cost. None on a basis
    #: before `SETTLED_CLOCK_BASIS_VERSION`, which does not wait.
    settle_ms: float | None = None
    settle_calls: int | None = None
    #: `timing.clock_flags`: loaded clock inside [CLOCK_FLOOR_FRACTION,
    #: CLOCK_CEILING_FRACTION] of the clock the ROOF was measured at, and
    #: first-to-last agreement within THROTTLE_DRIFT_PCT.
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    #: `timing.level_side`: which edge a failed LEVEL crossed, "low" or "high";
    #: "" when it passed or was not determined.
    clock_level_side: str = ""
    #: `moe.bench.timing.TIMING_BASIS` of the loop that produced `ms_p50`, or
    #: `SWEEP.SYNTHETIC_INSTRUMENT` on a planted row. Empty means a row from
    #: before the instrument had a name.
    instrument: str = ""
    #: Milliseconds of DELIVERED GPU load the warmup ran for, not a call count.
    warmup_ms: float = 0.0
    trials: int = 0
    l2_flush: bool = False
    status: str = "ok"
    detail: str = ""

    def __post_init__(self) -> None:
        SWEEP.check_level_side(self.clock_level_ok, self.clock_level_side)

    @property
    def clock_seen(self) -> bool:
        """Did anything actually sample a clock under load for this cell?

        NON-VACUITY at the level of one row. The poller lands no sample when
        NVML is absent, when the container forbids it, and when a trial is
        shorter than one poll interval; a throttle check over rows like that
        examines nothing and reports no failures.
        """
        return self.sm_clock_load_mhz is not None

    @property
    def throttled(self) -> bool:
        """Did the clock move across this cell's own trials.

        Derived rather than stored, and False for None on purpose: an EXCLUSION
        has to be positively established, and a row whose clock was never read
        is unknown rather than bad. `excluded` is the union this file filters
        on; V3 counts these two separately because they are different faults.
        """
        return self.clock_drift_ok is False

    @property
    def cold(self) -> bool:
        """Did this cell run BELOW the band around the clock the ROOF was
        measured at: LEVEL failed on the LOW side.

        KEPT since 2026-09-09, and this is the rule change that half this
        file's 2026-09-09 session turned on. Until that date `cold` was the
        exclusion, and on the H200 the 15 cells it excluded were every
        multi-tile BLOCK_M=128 subject cell of the arm, at 1380-1410 MHz
        against a 1410.75 MHz floor -- five of them low by 0.75 MHz, half of
        one NVML step. Across 750 cells of that session the under-load clock is
        an outcome of the CELL, set per tile by the kernel's own power draw
        under the 700 W cap: BM=128/BN=64 holds a median 1395 MHz, BM=256
        boosts to 1650, memory-shaped cells to 1950-1980, and the 8192^3
        calibration GEMM sits at 1485 MHz at 691 W, near the LOW end of what
        dense tensor work does on this card. So LEVEL-LOW was a rule against a
        TILE, and under it this arm's subject could not be measured on this
        card on any rerun. The side stays on the row, is counted by V3, and
        says which way `roof_fraction` is off; `roof_fraction_at_clock` is the
        rescaled number printed beside it."""
        return self.clock_level_side == SWEEP.LEVEL_LOW

    @property
    def boosted(self) -> bool:
        """Did this cell run ABOVE the band: LEVEL failed on the HIGH side.
        Kept; its fixed-roof fraction is inflated by the clock ratio and
        `Point.roof_fraction_at_clock` is the rescaled number printed beside
        it."""
        return self.clock_level_side == SWEEP.LEVEL_HIGH

    @property
    def excluded(self) -> bool:
        """The ONE clock state no rescaling repairs: the clock MOVED across
        this cell's own trials, so its median is a blend of two operating
        points and the time belongs to neither. A steady clock on either side
        of the band is a measurement at a known clock and is kept."""
        return self.throttled


TIMING_FIELDS = list(Timing.__dataclass_fields__)

#: Provenance columns appended to every cells.csv row, after the measurement
#: columns. `Provenance.as_columns` prefixes every one with `prov_`, so a
#: provenance column can never collide with a measurement column that happens to
#: share its name. Written per ROW rather than once per file, because a resumed
#: cells.csv is written by two processes on two days and possibly two commits,
#: and a header cannot say that.
PROVENANCE_COLUMNS = sorted(PV.Provenance().as_columns())


def _opt_float(text) -> float | None:
    """A CSV cell back to a float, or None for the empty one None was written as."""
    return None if text in (None, "", "None") else float(text)


def _samples_text(t) -> str:
    """`KernelTiming`'s under-load sample list as one CSV cell, or "".

    THROUGH getattr, on purpose. `moe.bench.timing.time_kernel` keeps only the
    COUNT of usable samples today and gains the list itself with the
    settle-on-clock warmup; this writer must persist it the moment it exists
    and must keep running until then, because every committed row of the
    2026-09-09 session was written by the basis that has no list. A missing
    list is written blank, which reads back as "not reported" and never as an
    empty measurement.
    """
    samples = getattr(t, "clock_samples_mhz", None)
    if not samples:
        return ""
    return " ".join(f"{float(v):.0f}" for v in samples)


def _opt_bool(text) -> bool | None:
    """A CSV cell back to a three-state flag.

    None round-trips as the empty string and is READ BACK as None, never as
    False: a replayed row whose clock verdict was "not determined" must not
    become "determined, and fine".
    """
    if text in (None, "", "None"):
        return None
    return str(text).strip().lower() in ("true", "1", "yes")


def append_timing(path: Path, row: Timing, prov=None) -> None:
    """One row, flushed. An abort costs the cell in flight and nothing else.

    SIGNATURE EXTENDED, never narrowed: `prov` is optional and a caller that
    omits it writes the measurement columns alone, which is what the tests do.
    When it is given, the row also carries the commit, the card, the instrument
    and the two ruler sources that made it, so a cells.csv on a network volume
    that outlives the pod stays attributable to a code version.
    """
    new = not path.exists()
    fields = TIMING_FIELDS + (PROVENANCE_COLUMNS if prov is not None else [])
    payload = asdict(row)
    if prov is not None:
        payload.update(prov.as_columns())
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(payload)
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
                sm_clock_load_mhz=_opt_float(row.get("sm_clock_load_mhz")),
                sm_clock_start_mhz=_opt_float(row.get("sm_clock_start_mhz")),
                sm_clock_end_mhz=_opt_float(row.get("sm_clock_end_mhz")),
                clock_samples_mhz=(row.get("clock_samples_mhz") or "").strip(),
                settle_ms=_opt_float(row.get("settle_ms")),
                settle_calls=(None if row.get("settle_calls") in (None, "", "None")
                              else int(float(row["settle_calls"]))),
                clock_level_ok=_opt_bool(row.get("clock_level_ok")),
                clock_drift_ok=_opt_bool(row.get("clock_drift_ok")),
                # Blank on every row written before the column existed, and
                # blank is a value: "inside the band, or not determined".
                clock_level_side=(row.get("clock_level_side") or "").strip(),
                instrument=row.get("instrument", ""),
                warmup_ms=float(row.get("warmup_ms") or 0.0),
                trials=int(float(row.get("trials") or 0)),
                l2_flush=bool(_opt_bool(row.get("l2_flush"))),
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
    #: The state the surviving repeats were timed in, carried from `Timing` so
    #: the figure CSV says which instrument drew it. Defaulted because the tests
    #: and the plotting helpers build points without a pod.
    instrument: str = ""
    warmup_ms: float = 0.0
    iters: int = 0
    trials: int = 0
    l2_flush: bool = False
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    clock_level_side: str = ""
    #: Repeats that failed LEVEL on the HIGH side and were KEPT. Counted so the
    #: table can say a point stands on boosted repeats.
    boosted_reps: int = 0
    #: Repeats that failed LEVEL on the LOW side and were KEPT since
    #: 2026-09-09. Counted for the same reason as `boosted_reps`: on a
    #: power-capped card a steady-low clock is a dense tile's own operating
    #: point, so it is kept, and what it moves is the fixed-roof fraction.
    cold_reps: int = 0
    #: `useful_tflops` over the roof AT THIS POINT'S OWN CLOCK
    #: (`roofline.roof_at_clock`), the number a boosted cell is comparable on.
    #: None when the point carries no clock or the roof carries none; never the
    #: fixed-roof fraction standing in for it.
    roof_fraction_at_clock: float | None = None

    @property
    def aligned(self) -> bool:
        return self.rows_per_expert % self.block_m == 0


#: The first `moe.bench.timing.TIMING_BASIS` version whose warmup keeps warming
#: until two consecutive NVML clock reads agree within one 15 MHz step, instead
#: of delivering a fixed duration of load and then starting the poller.
#:
#: WHY A NUMBER AND NOT A STRING. The basis is written per ROW, so a resumed
#: cells.csv can hold rows from two instruments, and this gate has to be able to
#: say which side of the fix each drifted cell came from. Matching the exact v4
#: string would make this file break the moment the instrument renames anything
#: else in the basis; the trailing `/vN` is the part that carries the contract.
SETTLED_CLOCK_BASIS_VERSION = 4


def instrument_settles_clock(instrument: str) -> bool | None:
    """Did the loop that wrote this row wait for the clock to settle.

    None when the row names no instrument or names one with no version, which
    is not a yes: a planted row and a pre-2026-09-02 row both land there and
    neither settled anything.
    """
    tail = (instrument or "").rsplit("/", 1)[-1]
    if not tail.startswith("v") or not tail[1:].isdigit():
        return None
    return int(tail[1:]) >= SETTLED_CLOCK_BASIS_VERSION


def per_tile_clocks(timings: list[Timing]) -> dict[int, int]:
    """Median under-load clock per BLOCK_M, which is the clock that exists.

    THE NUMBER THAT REPLACED `modal_clock` IN THE GATE TEXT. On the 2026-09-09
    H200 session `modal_clock` returned 1590 MHz: the median of a BIMODAL set,
    18 subject cells at 1380-1425 and 21 control and small cells at 1575-1785,
    and the cell sitting at that median is a single BM=128 r=64 repeat between
    the two modes. "+7.1% against the roof's clock" described no operating
    point the card ever ran at. Per tile the same cells read 1410 (BM=128) and
    1665 (BM=256), which are two states the card really held.
    """
    out: dict[int, int] = {}
    for bm in sorted({t.block_m for t in timings
                      if t.status == "ok" and t.clock_seen}):
        seen = [t.sm_clock_load_mhz for t in timings
                if t.block_m == bm and t.status == "ok" and t.clock_seen]
        out[bm] = int(statistics.median(seen))
    return out


def modal_clock(timings: list[Timing]) -> int:
    """The clock this session's KERNELS ran at, or 0 when none was ever read.

    The median of the per-cell UNDER-LOAD medians. Two properties, and the audit
    found the old version had neither. It is a median rather than a maximum, so
    one spuriously high reading taken during a ramp cannot put every other cell
    below the floor and exclude the whole run. And it is taken under load rather
    than at the idle instant after a synchronise: the H200 idles at 1980 MHz and
    runs a dense GEMM at about 1515, so a session of post-sync samples reported
    1980 and was then compared with the roof's 1515 -- a 31% disagreement
    manufactured entirely by which of the card's two operating points the sample
    landed in.

    0 when nothing was sampled, which V3 reads as UNKNOWN rather than as a pass.

    NO LONGER THE NUMBER V3 PRINTS. It is a session-wide median, and when a
    session runs two tiles at two operating points -- which on a power-capped
    card is the normal case, not the exception -- the median is a value neither
    tile held. `per_tile_clocks` is what the gate prints; this stays on the
    payload as `clock_ref_mhz` because it is what `build_points` carries onto
    every point and what earlier reports quoted.
    """
    seen = [t.sm_clock_load_mhz for t in timings
            if t.status == "ok" and t.clock_seen]
    return int(statistics.median(seen)) if seen else 0


def build_points(timings: list[Timing], cfg, block_m: int, roof: Roof,
                 *, sm_count: int, block_n: int, clock_ref: int) -> list[Point]:
    """Collapse repeats into one point per row count, and mark the unreadable.

    The median across REPEATS, not the single-pass median: a repeat is a fresh
    call at a fresh point in the pod's thermal history, and a throughput read
    from one pass cannot tell a sagging clock from a tile effect.

    A point is dropped when EVERY repeat of it was excluded, and kept on the
    surviving ones otherwise -- with the count of what was dropped carried on
    the point, because a point standing on one of five repeats is not the same
    measurement as one standing on five.

    THE EXCLUSION RULE IS THE INSTRUMENT'S, not a comparison recomputed here,
    and since 2026-09-09 it is ONE flag: `clock_drift_ok is False`, the clock
    moved across the cell's own trials, so its median is a blend of two
    operating points. NEITHER LEVEL SIDE EXCLUDES. `Timing.cold` and
    `Timing.boosted` are recorded, counted on the point as `cold_reps` and
    `boosted_reps`, and the point carries `roof_fraction_at_clock`, its
    throughput over the roof rescaled to the clock it ran at, which is the
    issue efficiency printed beside the fixed-roof fraction. `clock_ref` is the
    session's own under-load modal clock; it is carried onto the point and
    printed, and on this arm it is the median of a BIMODAL set (BM=128 at 1410,
    BM=256 at 1665) and names no operating point, which is why V3 prints per
    tile.
    """
    by: dict[int, list[Timing]] = {}
    for t in timings:
        if t.block_m == block_m and t.status == "ok" and t.ms_p50 > 0:
            by.setdefault(t.rows_per_expert, []).append(t)
    out: list[Point] = []
    for rows in sorted(by):
        group = by[rows]
        bad = {id(t) for t in group if t.excluded}
        good = [t for t in group if id(t) not in bad]
        used = good or group
        ms = statistics.median([t.ms_p50 for t in used])
        spread = (statistics.pstdev([t.ms_p50 for t in used]) / ms
                  if len(used) > 1 and ms > 0 else None)
        # The timing STATE the point carries is the first surviving repeat's.
        # Every repeat of one point is timed by one instrument at one warmup and
        # one flush setting -- those come from argv, not from the cell -- so the
        # only fields that could differ across repeats are the clock verdicts,
        # and a repeat whose verdicts differ has already been excluded above.
        state = used[0]
        cell = SWEEP.make_cell(
            cfg, rows, block_m, ms, sm_count=sm_count, block_n=block_n,
            ms_min=min(t.ms_min for t in used),
            ms_stdev=statistics.pstdev([t.ms_p50 for t in used]) if len(used) > 1 else 0.0,
            iters=state.iters, instrument=state.instrument,
            warmup_ms=state.warmup_ms, trials=state.trials,
            sm_clock_load_mhz=state.sm_clock_load_mhz,
            clock_level_ok=state.clock_level_ok,
            clock_drift_ok=state.clock_drift_ok,
            clock_level_side=state.clock_level_side, l2_flush=state.l2_flush)
        clocks = [t.sm_clock_load_mhz for t in used if t.clock_seen]
        why = ""
        if not good:
            n_drift = sum(1 for t in group if t.throttled)
            why = (f"every one of {len(group)} repeats was excluded: "
                   f"{n_drift} drifted more than {THROTTLE_DRIFT_PCT:.0f}% "
                   "across their own trials, so each median blends two "
                   "operating points")
        # THE ROOF AT THIS POINT'S OWN CLOCK. `roof_at_clock` refuses (None)
        # without a clock on the point or on the roof; the fixed fraction is
        # never written into this column in its place.
        point_clock = statistics.median(clocks) if clocks else None
        own_roof = roof_at_clock(roof.tflops, roof.clock_mhz or None, point_clock)
        out.append(Point(
            block_m=block_m, rows_per_expert=rows, tiles=cell.tiles_per_expert,
            tokens=cell.tokens, regime=regime_of(rows, block_m),
            tile_eff=cell.tile_eff, reps=len(used), ms_p50=ms, spread=spread,
            useful_tflops=cell.useful_tflops,
            roof_fraction=(cell.useful_tflops / roof.tflops
                           if roof.tflops > 0 else 0.0),
            sm_clock_mhz=int(statistics.median(clocks)) if clocks else 0,
            throttled_reps=len(bad), retained=bool(good), excluded_why=why,
            instrument=state.instrument, warmup_ms=state.warmup_ms,
            iters=state.iters, trials=state.trials, l2_flush=state.l2_flush,
            clock_level_ok=state.clock_level_ok,
            clock_drift_ok=state.clock_drift_ok,
            clock_level_side=state.clock_level_side,
            boosted_reps=sum(1 for t in used if t.boosted),
            cold_reps=sum(1 for t in used if t.cold),
            roof_fraction_at_clock=(cell.useful_tflops / own_roof
                                    if own_roof else None)))
    return out


def multi_tile(points: list[Point]) -> list[Point]:
    """Retained, exactly-full, more than one tile per expert. The gated set."""
    return [p for p in points
            if p.retained and p.aligned and p.regime == "multi-tile"]


def own_clock_text(p: Point) -> str:
    """This point's fraction of the roof AT ITS OWN CLOCK, as a printable.

    THE COLUMN THAT GOES BESIDE EVERY GATED FRACTION, never instead of it. The
    CLAIM gates read `Point.roof_fraction`, the FIXED roof, because the
    calibration GEMM and every cell here ran under the same 700 W cap and the
    delivered-throughput comparison is the fair one: a tile that spends its
    power budget on a clock it cannot hold has delivered less, and that is the
    result. The rescaled fraction is the ISSUE EFFICIENCY -- what the tile did
    with the clock it actually got -- and it is a different question, so it is
    printed and never scored.

    NEITHER NUMBER MAY STAND ALONE ON THIS ARM. On the 2026-09-09 H200 session
    the control ran at 1605-1725 MHz and the subject at 1380-1425, so the two
    fractions disagree by 10-12% in OPPOSITE directions and the sign of C3
    flips between them: +0.053 of the roof fixed, -0.032 at own clock. A report
    printing one of them without the other lets a reader take a gate's verdict
    for a fact about the tile.
    """
    return ("own clock n/a" if p.roof_fraction_at_clock is None
            else f"own clock {p.roof_fraction_at_clock:.3f}")


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
        + "   x excluded (drifted or LEVEL-low)   * both")
    return out


def point_table(series: list[Series]) -> list[str]:
    """The numbers behind the picture, because 18 rows of characters lose them."""
    out = [f"{'BM':>4s} {'T':>7s} {'rows':>6s} {'n':>4s} {'regime':>10s} "
           f"{'tile_eff':>8s} {'ms':>10s} {'TFLOP/s':>9s} {'of roof':>8s} "
           f"{'@own clk':>8s} {'MHz':>5s} {'reps':>4s}  note".rstrip(),
           "        of roof = the FIXED roof, which is what every CLAIM gate "
           "reads; @own clk = the same throughput over the roof at this "
           "point's own clock, the issue efficiency, scored by nothing"]
    for s in series:
        for p in s.points:
            note = p.excluded_why or ("" if p.retained else "excluded")
            if p.throttled_reps and p.retained:
                note = f"{p.throttled_reps} repeat(s) excluded for DRIFT"
            # Kept, and said, on BOTH sides: an off-band point's `of roof`
            # column is the fixed-roof fraction and is off by the clock ratio
            # in the named direction. The rescaled number is its own column, so
            # it is on every row and not only on the ones with a note.
            for n, side, way in ((p.boosted_reps, "high", "over"),
                                 (p.cold_reps, "low", "under")):
                if n and p.retained:
                    note = (note + "; " if note else "") + (
                        f"{n} repeat(s) LEVEL-{side}, kept; fixed-roof "
                        f"fraction {way}stated by the clock ratio")
            at = ("       -" if p.roof_fraction_at_clock is None
                  else f"{p.roof_fraction_at_clock:8.3f}")
            out.append(
                f"{p.block_m:4d} {p.tokens:7d} {p.rows_per_expert:6d} "
                f"{p.tiles:4d} {p.regime:>10s} {p.tile_eff:8.3f} "
                f"{p.ms_p50:10.4f} {p.useful_tflops:9.1f} "
                f"{p.roof_fraction:8.3f} {at} {p.sm_clock_mhz:5d} "
                f"{p.reps:4d}  {note}".rstrip())
    return out


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

#: Taken from `moe.bench.exit_codes` rather than spelled again, because
#: `classify` refuses a kind it does not know rather than letting it fall
#: through a comparison and be scored as whatever the fallthrough happened to
#: be. The two names were already identical; now they cannot drift apart.
VALIDITY, CLAIM = exit_codes.VALIDITY, exit_codes.CLAIM


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

    @property
    def token(self) -> str:
        """The one-word name this gate answers to on its `RESULT:` line.

        A slug of `name`, because `exit_codes.result_line` requires a name with
        no whitespace and a hand-kept table of tokens is a second thing to keep
        in step with the names. `"C1 roof"` becomes `C1_roof` and
        `"S hypothesis roof refused"` becomes `S_hypothesis_roof_refused`, so a
        driver's grep is stable for as long as the gate's name is.
        """
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")
        if not slug:
            raise exit_codes.MalformedResultLine(
                f"gate name {self.name!r} slugs to nothing, so it cannot be "
                "named on a RESULT line")
        return slug

    @property
    def verdict(self) -> str:
        """PASS / FAIL / UNKNOWN in `exit_codes`'s vocabulary.

        The property `exit_codes.classify` reads off this object directly, so a
        gate list can be handed to it without a translation step that could
        disagree with what was printed.
        """
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.passed]

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        Rendered by `exit_codes.result_line` and read back by
        `parse_result_lines`, anchored at column zero on the `RESULT: ` prefix.
        The bracketed line below it is for a human and for the session driver's
        existing regex; both are kept because they have different readers, and
        only this one is the machine contract. Nothing else this file prints
        starts with `RESULT: `.
        """
        detail = f"{self.prediction} | saw {self.observed} | gate {self.rule}"
        return exit_codes.result_line(self.kind, self.token, self.verdict,
                                      " ".join(detail.split()))

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"[{self.verdict}] {self.kind:8s} {self.name}  {self.prediction}",
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


def gate_v2_non_vacuity(subject: list[Point], control: list[Point] | None,
                        planned_multi_tile: int, *, onset_tokens_value: int
                        ) -> Gate:
    """Did this run measure the thing it is about.

    A check that examined nothing also reports zero failures. Three counts, and
    the third is the one that matters: crossing the multi-tile onset is the
    entire premise, so a run whose deepest subject point is still one tile per
    expert has measured the regime TEMPO already describes and nothing else.

    `control=None` is `--control none`, and the SHARED-TOKEN count is then
    dropped rather than scored as zero. An uncontrolled arm shares its batches
    with nothing by design, so scoring it there would fail this gate for the
    condition the operator asked for -- and a VALIDITY failure voids the two
    claim gates such a run exists to produce. The first two counts still apply
    and can still fail.
    """
    deep = multi_tile(subject)
    reached = max((p.tiles for p in deep), default=0)
    shared = (None if control is None
              else len({p.tokens for p in multi_tile(control)}
                       & {p.tokens for p in deep}))
    ok = (len(deep) >= MIN_MULTI_TILE_POINTS and reached >= 4
          and (shared is None or shared >= 2))
    return Gate(
        VALIDITY, "V2 non-vacuity",
        "the sweep crossed the multi-tile onset and measured "
        + ("the subject there" if control is None else "both tiles there"),
        f">= {MIN_MULTI_TILE_POINTS} retained full-stack multi-tile subject "
        "points, >= 4 M-tiles per expert reached"
        + (", and no control to share them with"
           if control is None else ", >= 2 token counts shared with the "
                                   "control"),
        ok,
        f"{len(deep)} retained subject points of {planned_multi_tile} "
        f"multi-tile batches planned, deepest "
        f"{reached} M-tiles per expert, "
        + ("no control in this arm" if shared is None else
           f"{shared} token counts shared with the control"),
        "every claim gate. Below the onset there is one tile per expert, the "
        "re-read term does not exist, and this script has measured the regime "
        "the prior work already agrees about",
        [f"multi-tile begins above T={onset_tokens_value} for this model, "
         "which is where the second M-tile per expert appears."])


def gate_v3_clocks(timings: list[Timing], points: list[Point], roof: Roof,
                   clock_ref: int) -> Gate:
    """Was this measured at one operating point, and is every retained point
    scored against the roof at the clock it ran.

    Throttling invalidates a throughput measurement more than a latency one: a
    latency is wrong by the clock ratio, while a throughput read at a sagging
    clock is wrong AND looks exactly like the flattening this script is trying
    to detect.

    EVERY NUMBER HERE IS SAMPLED UNDER LOAD. `clock_ref` is the median of the
    per-cell under-load medians and `roof.clock_mhz` is the clock the
    calibration's own dense GEMM ran at, so the comparison printed at the end
    is between two measurements of the same kind. The audit's finding was that
    it was not: the session number was a median of post-synchronise idle
    instants, which on an H200 reads 1980 MHz against a roof measured at 1515,
    and the 10% gate then turned on which of the card's two operating points
    the sample happened to catch rather than on anything about this run.

    AND THE SESSION IS NOT FAILED FOR RUNNING HIGH. Until 2026-09-08 this gate
    also required the session median within 10% of the roof's clock in EITHER
    direction, and `tests/test_bm128_roofline.py` asserted that a session at
    1980 against a 1515 roof FAILS. That is the state every memory-shaped cell
    on the H200 is in: the committed calibration's memory load settles at 1980
    MHz for its whole 30 s against a 1455-1515 MHz GEMM plateau. An honest
    roofline arm therefore exited INVALID, `scripts/h200_gaps_session.sh`
    latched it, and no resume re-ran it. What is wrong about a boosted cell is
    not its time but its fraction of the FIXED roof, which is inflated by the
    clock ratio; the correction is the roof at the cell's own clock
    (`roofline.roof_at_clock`, the driver's `roof_at_cell_clock_tflops`), and
    `build_points` writes it on every point as `roof_fraction_at_clock`. So
    this gate scores PER CELL.

    AND SINCE 2026-09-09 IT IS NOT FAILED FOR RUNNING LOW EITHER, which is the
    other half of the same fix. The share used to count DRIFT and LEVEL-LOW; on
    the 2026-09-09 H200 session the LOW cells were all 15 multi-tile
    BLOCK_M=128 subject cells, at 1380-1410 MHz against a 1410.75 MHz floor
    (five of them low by 0.75 MHz, half of one NVML step), and the same
    session's 750 cells show the under-load clock is set PER TILE by the
    kernel's own power draw under the 700 W cap: BM=128/BN=64 a median 1395
    MHz, BM=256 1650, memory-shaped 1950-1980, against a calibration GEMM
    holding 1485 MHz at 691 W. Failing on that side is failing the arm for
    measuring its own subject. The share now counts DRIFT alone -- the one
    state no rescaling repairs, because the median blends two operating points
    -- and both LEVEL sides are kept, counted, and printed with their rescaled
    fraction beside the fixed one.

    WHAT A FAILING SHARE NOW POINTS AT. Every readable DRIFT in that session is
    the governor settling on the FIRST cell of a repeat after a workload
    change, which is a property of a warmup that delivers a fixed duration and
    then starts timing. That is an INSTRUMENT fault and the fix is in the
    instrument: warm until two consecutive clock reads agree within one NVML
    step. The gate says so in its lines and names the instrument the drifted
    cells were timed by, so a FAIL sends a reader to the warmup rather than to
    the card.

    REFUSES (UNKNOWN) without a clock on any cell, or without the roof's
    clock: the first is a check that examined nothing, the second can form
    neither the LEVEL verdict nor the per-cell roof.
    """
    ok_rows = [t for t in timings if t.status == "ok"]
    with_clock = [t for t in ok_rows if t.clock_seen]
    claim = ("every cell was timed at one clock, and every retained point is "
             "scored against the roof at the clock it ran")
    rule = (f"<= {THROTTLED_CELL_FRACTION:.0%} of clocked cells drifting past "
            f"{THROTTLE_DRIFT_PCT:.0f}% across their own trials. A STEADY "
            f"clock on either side of the band (under "
            f"{CLOCK_FLOOR_FRACTION:.0%} or over {CLOCK_CEILING_FRACTION:.0%} "
            "of the roof's clock) is KEPT, its side recorded, and its fraction "
            "of the roof at its own clock printed beside the fixed one")
    if not with_clock:
        return Gate(
            VALIDITY, "V3 clocks", claim, rule, None,
            f"no under-load clock sample landed on any of {len(ok_rows)} "
            "cells; the poller reported none and NVML may be absent",
            "the throttle exclusions and the roof comparison. A throttle check "
            "over rows with no clocks examined nothing and would report no "
            "failures",
            ["NON-VACUITY: this gate refuses to PASS on an instrument that "
             "never took a reading."])
    drifted = [t for t in with_clock if t.throttled]
    low = [t for t in with_clock if t.cold]
    high = [t for t in with_clock if t.boosted]
    # KEPT, not merely off-band. A cell that both drifted and sat off the band
    # is EXCLUDED, so counting it as "kept" is the defect this split closes:
    # the 2026-09-09 line read "21 high and kept" over 17 kept and 4 excluded.
    low_kept = [t for t in low if not t.excluded]
    high_kept = [t for t in high if not t.excluded]
    dropped = [p for p in points if not p.retained]
    per_tile = per_tile_clocks(ok_rows)
    lines = [f"{len(with_clock)} of {len(ok_rows)} cells carry an under-load "
             "clock; MEDIAN PER TILE, which is the clock a cell of that tile "
             "actually held: "
             + ", ".join(f"BM={bm} {mhz} MHz" for bm, mhz in per_tile.items())
             + ". No session median is quoted: on this arm the two tiles sit "
             "at two operating points and a median over both names neither.",
             f"{len(drifted)} cells drifted past {THROTTLE_DRIFT_PCT:.0f}% "
             "between their first and last under-load sample and are the only "
             f"exclusion; {len(low_kept)} of {len(low)} steady-LOW (under "
             f"{CLOCK_FLOOR_FRACTION:.0%} of the roof's clock) and "
             f"{len(high_kept)} of {len(high)} steady-HIGH (over "
             f"{CLOCK_CEILING_FRACTION:.0%} of it) were KEPT, the rest of each "
             f"having also drifted; {len(dropped)} points lost every repeat "
             "and are excluded"]
    if not roof.clock_mhz:
        return Gate(
            VALIDITY, "V3 clocks", claim, rule, None,
            f"{len(drifted)} drifted and {len(low)} low cells, but the "
            "calibration records no GEMM clock to compare this session "
            "against or to rescale the roof by",
            "the direction of the roof-fraction bias. Without the roof's clock "
            "a fraction below 1 cannot be told from a card running slower now "
            "than it did when it was calibrated, and no per-cell roof can be "
            "formed", lines)
    parts = []
    if high:
        ratio = max(t.sm_clock_load_mhz / roof.clock_mhz for t in high)
        parts.append(
            f"a steady-HIGH cell's fraction of the FIXED roof is OVERSTATED by "
            f"its clock ratio, up to {ratio:.2f}x here")
    if low:
        ratio = min(t.sm_clock_load_mhz / roof.clock_mhz for t in low)
        parts.append(
            f"a steady-LOW cell's fraction of the FIXED roof is UNDERSTATED by "
            f"its clock ratio, down to {ratio:.2f}x here")
    if parts:
        direction = (
            "Direction: " + "; and ".join(parts) + ". Both are KEPT and both "
            "are SCORED against the fixed roof, because the calibration GEMM "
            "and every cell here ran under the same power cap and delivered "
            "throughput is the comparison that cap makes fair. `of roof at own "
            "clock`, which divides by roofline.roof_at_clock(roof, roof clock, "
            "cell clock), is printed beside every one of them as the ISSUE "
            "EFFICIENCY -- what the tile did with the clock it got -- and it "
            "is never a gate input.")
    else:
        direction = ("every clocked cell sat inside the band, so the fixed "
                     "roof is the roof these cells ran under and the two "
                     "fraction columns agree to the rescale's rounding.")
    lines.append(
        "per tile against the "
        f"{roof.clock_mhz} MHz the roof was measured at: "
        + ", ".join(f"BM={bm} {mhz / roof.clock_mhz - 1.0:+.1%}"
                    for bm, mhz in per_tile.items())
        + f". {direction}")
    # THE SHARE COUNTS ONE FAULT: a clock that MOVED while the cell was timed,
    # so the median is a blend of two operating points and the time belongs to
    # neither. NEITHER LEVEL SIDE IS IN THE SHARE since 2026-09-09: on a
    # power-capped card a steady clock off the band is the tile's own operating
    # point, the side is recorded, and the rescaled fraction is printed beside
    # the fixed one.
    excluded = [t for t in with_clock if t.excluded]
    share = len(excluded) / len(with_clock)
    ok = share <= THROTTLED_CELL_FRACTION
    # WHICH INSTRUMENT THE DRIFTED CELLS CAME FROM, named, because the fix for
    # a settling governor is in the warmup and not on the card. A basis before
    # `SETTLED_CLOCK_BASIS_VERSION` delivers a fixed duration of load and then
    # starts the poller, so the first cell of a repeat after a workload change
    # is timed while the governor is still moving -- which is where all 4 of
    # the 2026-09-09 drifts on this arm sit.
    if drifted:
        # THE DIRECTION EACH DRIFT WENT, from the two under-load samples the
        # verdict was formed on. Settling UPWARD after a workload change is the
        # governor catching up and is what a warmup that waits for the clock
        # absorbs; dropping is a card giving up power; and a row carrying
        # neither sample cannot say, which is its own finding about the writer.
        ways: dict[str, int] = {}
        for row in drifted:
            first, last = row.sm_clock_start_mhz, row.sm_clock_end_mhz
            if first is None or last is None:
                ways["direction not recorded"] = \
                    ways.get("direction not recorded", 0) + 1
            else:
                way = ("settling upward" if last > first
                       else "dropping" if last < first else "unchanged ends")
                ways[way] = ways.get(way, 0) + 1
        lines.append(
            "drift direction, first under-load sample to last: "
            + ", ".join(f"{n} {way}" for way, n in sorted(ways.items())))
        bases = sorted({t.instrument or "<unnamed>" for t in drifted})
        unsettled = [t for t in drifted
                     if instrument_settles_clock(t.instrument) is not True]
        lines.append(
            f"the {len(drifted)} drifted cell(s) were timed by: "
            + ", ".join(bases)
            + (f". {len(unsettled)} of them by a basis BEFORE "
               f"v{SETTLED_CLOCK_BASIS_VERSION}, the first that keeps warming "
               "until two consecutive NVML reads agree within one 15 MHz step. "
               "A pre-v4 loop delivers a fixed warmup duration and then starts "
               "timing, so the first cell of a repeat after a workload change "
               "is measured while the governor is still settling: that is an "
               "INSTRUMENT fault, it is where every readable drift in the "
               "2026-09-09 session sits, and re-running on the same instrument "
               "reproduces it." if unsettled else
               ". Every one settles the clock before timing, so a drift here "
               "is the card and not the warmup."))
    rescored = [p for p in points
                if p.retained and p.roof_fraction_at_clock is not None]
    if rescored:
        lines.append(
            "fraction of roof, FIXED (what every CLAIM gate reads) -> at the "
            "point's own clock (issue efficiency, scored by nothing), per "
            "retained point: "
            + ", ".join(f"BM={p.block_m} r={p.rows_per_expert} "
                        f"{p.roof_fraction:.3f}->{p.roof_fraction_at_clock:.3f}"
                        for p in rescored))
    if dropped:
        lines.append(
            f"points that lost EVERY repeat and left the gated set: "
            f"{[(p.block_m, p.rows_per_expert) for p in dropped]}. Losing the "
            "deepest one shortens the doubling chain, which C2 reports as a "
            "span too short rather than as a plateau.")
    return Gate(
        VALIDITY, "V3 clocks", claim, rule, ok,
        f"{len(excluded)} of {len(with_clock)} cells excluded ({share:.1%}), "
        f"all for DRIFT; {len(low_kept)} steady-low and {len(high_kept)} "
        "steady-high kept, side recorded, scored against the fixed roof with "
        "the own-clock fraction printed beside; per-tile medians "
        + ", ".join(f"BM={bm} {mhz} MHz" for bm, mhz in per_tile.items()),
        "the fractions, in the direction stated below. A cell whose clock moved "
        "mid-measurement is already excluded; a session that drifts this often "
        "was timed by a warmup that does not wait for the clock, which no "
        "exclusion can fix", lines)


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
    # THE GATE READS THE FIXED ROOF; the own-clock fraction is printed beside
    # it and scored by nothing. The calibration GEMM and every cell here ran
    # under the same 700 W cap, so delivered throughput against a fixed roof is
    # the comparison that cap makes fair: a tile that spends its power budget
    # on a clock it cannot hold has delivered less, and that IS the result. The
    # rescaled fraction answers a different question -- what the tile did with
    # the clock it got -- and on this arm the two disagree by 10-12% in
    # opposite directions between subject and control.
    best_fixed = max(deep, key=lambda p: p.roof_fraction)
    last = max(deep, key=lambda p: p.tiles)
    rising = plateau.plateaued is not True
    lines = [
        f"issue efficiency beside it: the peak point (BM={best_fixed.block_m} "
        f"n={best_fixed.tiles}) reads {own_clock_text(best_fixed)} at "
        f"{best_fixed.sm_clock_mhz} MHz, and the deepest ({last.tiles} tiles) "
        f"{own_clock_text(last)} at {last.sm_clock_mhz} MHz. NOT A GATE INPUT.",
        f"predicted plateau band, from the study's own alpha at BLOCK_M=128 "
        f"(GROUP_SIZE_M=1): {predicted[0]:.3f} to {predicted[1]:.3f} of the "
        f"roof, i.e. {predicted[0] * roof.tflops:.0f} to "
        f"{predicted[1] * roof.tflops:.0f} TFLOP/s. PRE-REGISTERED AND SCORED "
        "AGAINST NOTHING: a band that wide is a statement about the fit, not "
        "about the hardware.",
        f"the deepest point measured is n={last.tiles} M-tiles per expert at "
        f"T={last.tokens}, reaching {last.roof_fraction:.3f} of the fixed "
        f"roof ({last.useful_tflops:.1f} of {roof.tflops:.1f} TFLOP/s, "
        f"{own_clock_text(last)}), and it was "
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
            f"peak {peak:.3f} of the fixed roof at n={best_fixed.tiles} "
            f"M-tiles ({own_clock_text(best_fixed)})", "",
            [f"BLOCK_M={SUBJECT_BLOCK_M} REACHED the roof. The AI ceiling is "
             "not binding in the regime production runs, and the study's "
             "central claim is about a regime that does not occur."] + lines)
    return Gate(
        CLAIM, "C1 roof",
        f"BLOCK_M={SUBJECT_BLOCK_M} stays below the compute roof in the "
        "multi-tile regime",
        f"peak roof fraction < {ROOF_REACHED:.2f}", True,
        f"peak {peak:.3f} of the fixed roof at n={best_fixed.tiles} M-tiles "
        f"({own_clock_text(best_fixed)})", "", lines)


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
            f"the last point reached {last.roof_fraction:.3f} of the fixed "
            f"roof ({own_clock_text(last)} at {last.sm_clock_mhz} MHz) and "
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
    # THE GATE READS THE FIXED ROOF, and on this arm that choice has a SIGN.
    # The control ran at 1605-1725 MHz and the subject at 1380-1425, so the two
    # fractions disagree by 10-12% in opposite directions: over the 2026-09-09
    # session's 4 shared token counts the gap is +0.053 of the fixed roof and
    # -0.032 at own clock. The fixed roof is the gate input because both tiles
    # and the calibration GEMM ran under the same 700 W cap and delivered
    # throughput is what that cap makes comparable; the own-clock pair is
    # printed under it so a reader can see the sign flip rather than inherit
    # it. Neither number is the whole answer on its own.
    subj = [p for p in deep_s if p.tokens in shared]
    ctl = [p for p in deep_c if p.tokens in shared]
    best_s = max(subj, key=lambda p: p.roof_fraction)
    best_c = max(ctl, key=lambda p: p.roof_fraction)
    peak_s, peak_c = best_s.roof_fraction, best_c.roof_fraction
    gap = peak_c - peak_s
    own_s = best_s.roof_fraction_at_clock
    own_c = best_c.roof_fraction_at_clock
    lines = [
        f"over the {len(shared)} shared token counts {shared[:6]}: "
        f"BLOCK_M={SUBJECT_BLOCK_M} peaks at {peak_s:.3f} of the FIXED roof, "
        f"BLOCK_M={control_block_m} at {peak_c:.3f}, a gap of {gap:+.3f} "
        f"({gap * roof.tflops:+.0f} TFLOP/s)",
        "issue efficiency beside it, scored by nothing: at those same two "
        f"points BLOCK_M={SUBJECT_BLOCK_M} reads "
        + (f"{own_s:.3f}" if own_s is not None else "n/a")
        + f" at {best_s.sm_clock_mhz} MHz and BLOCK_M={control_block_m} "
        + (f"{own_c:.3f}" if own_c is not None else "n/a")
        + f" at {best_c.sm_clock_mhz} MHz, "
        + ("a gap of "
           f"{own_c - own_s:+.3f}. THE TWO GAPS CAN DISAGREE IN SIGN, and "
           "when they do neither may be quoted without the other: one says "
           "which tile delivered more under the cap, the other which tile used "
           "its own clock better."
           if own_s is not None and own_c is not None else
           "so no own-clock gap can be formed."),
        f"the control's own derivative: {control_plateau.line()}",
    ]
    if peak_c >= ROOF_REACHED:
        lines.append(
            f"the control REACHED the roof ({peak_c:.3f} >= "
            f"{ROOF_REACHED:.2f}), so this instrument is demonstrably able to "
            "see a tile reach it, and an absence at the subject is evidence of "
            "absence rather than a shrug. Gate C4 scores that separately.")
    else:
        lines.append(
            f"the control did NOT reach the roof either ({peak_c:.3f} < "
            f"{ROOF_REACHED:.2f}). The gap below is then the whole of the "
            "evidence: what the two tiles share has cancelled, but the height "
            "of the ceiling has not been demonstrated. Gate C4 is where that "
            "is scored, and the verdict it produces is TILE-DEPENDENT GAP, "
            "CEILING UNLOCATED rather than the study's headline.")
    if control_plateau.plateaued is True and abs(gap) < CONTROL_SEPARATION:
        lines.append(
            "BOTH TILES PLATEAU TOGETHER, at fractions this gate cannot "
            "separate. That is a result: the shortfall is a property of the "
            "fused layer, and whatever binds it is not the tile-dependent AI "
            "ceiling this study proposed.")
    # NAMED WHATEVER THIS GATE SAYS, not only when it FAILs. The audit's finding
    # was that the residency confound appeared in the NOT_TILE branch alone, so
    # a PASS -- the branch on which the study's headline is written -- printed a
    # difference of two tiles with no mention that the two also differ 2:1 in
    # resident blocks. A confound that can produce the gap has to be beside the
    # gap, and most of all when the gap is the result being quoted.
    lines += RESIDENCY_CONFOUND
    return Gate(
        CLAIM, "C3 tile attribution",
        f"the shortfall belongs to BLOCK_M={SUBJECT_BLOCK_M}, not to the fused "
        "layer both tiles run inside",
        f"control peak - subject peak >= {CONTROL_SEPARATION:.2f} of the roof",
        gap >= CONTROL_SEPARATION,
        f"gap {gap:+.3f} of the fixed roof"
        + ("" if own_s is None or own_c is None
           else f" ({own_c - own_s:+.3f} at own clocks, not a gate input)"),
        "", lines)


def gate_c4_ceiling_located(control: list[Point], control_block_m: int,
                            roof: Roof) -> Gate:
    """THE POSITIVE CONTROL, scored. Did any tile in this run reach the roof.

    WHY THIS IS A GATE AND NOT A PRINTED LINE. Until 2026-09-02 the control's
    peak was printed inside C3 and gated on nothing, so the headline verdict
    "the ceiling is real and it binds" could be issued with the subject at 0.58
    and the control at 0.70, both stopped far below the roof for reasons this
    study's model does not describe. That is a tile-dependent GAP and it is not
    a ceiling: naming a mechanism from a difference, without the positive
    control ever going positive, is the benchmarking crime this gate exists to
    make impossible.

    IT IS EXPECTED TO FAIL ON TODAY'S EVIDENCE, and that is the point of
    pre-registering it. No tile in this repository's published corpus reaches
    0.95 of ridge x bandwidth: BLOCK_M=256 peaks at 0.48-0.54 of the roof on the
    H200 and 0.56-0.64 on the A100. `docs/FINDINGS.md`'s "structurally incapable
    of reaching its compute roof" has no positive control behind it, and a FAIL
    here is this run saying so in its own output rather than a reviewer saying
    it later.

    A FAIL does not void the gap. It changes the sentence the gap earns, from
    CEILING BINDING to TILE-DEPENDENT GAP, CEILING UNLOCATED.
    """
    deep = multi_tile(control)
    if not deep:
        return Gate(
            CLAIM, "C4 ceiling located",
            f"the control BLOCK_M={control_block_m} reaches this card's roof, "
            "so the roof is somewhere this instrument can see",
            f"control peak roof fraction >= {ROOF_REACHED:.2f}", None,
            "no retained full-stack multi-tile control points", "",
            ["Nothing reached anything, so nothing located the ceiling. V4 says "
             "whether the control ran at all."])
    # THE FIXED ROOF, as in C1 and C3: same cap, same GEMM, delivered
    # throughput. The own-clock fraction is printed beside and gates nothing.
    peak = max(p.roof_fraction for p in deep)
    best = max(deep, key=lambda p: p.roof_fraction)
    lines = [
        f"the control's best point is n={best.tiles} M-tiles per expert at "
        f"T={best.tokens}: {best.useful_tflops:.1f} of {roof.tflops:.1f} "
        f"TFLOP/s, {peak:.3f} of the fixed roof, {own_clock_text(best)} at "
        f"{best.sm_clock_mhz} MHz (issue efficiency, scored by nothing).",
        "the fraction UNDERSTATES the GEMM's efficiency by whatever the fused "
        "layer spends outside its GEMMs, so a control below the roof is not "
        "proof that no tile can reach it -- it is a failure to DEMONSTRATE "
        "that this instrument can see one arrive, which is all this gate "
        "claims.",
    ]
    if peak < ROOF_REACHED:
        lines.append(
            f"the published corpus agrees: BLOCK_M=256 peaks at 0.48-0.54 of "
            "the roof on the H200 and 0.56-0.64 on the A100, so a FAIL here is "
            "the expected reading and the verdict below is "
            f"{GAP_UNLOCATED!r}.")
    lines += RESIDENCY_CONFOUND
    return Gate(
        CLAIM, "C4 ceiling located",
        f"the control BLOCK_M={control_block_m} reaches this card's roof, so "
        "the roof is somewhere this instrument can see a tile arrive",
        f"control peak roof fraction >= {ROOF_REACHED:.2f}",
        peak >= ROOF_REACHED,
        f"control peak {peak:.3f} of the fixed roof ({own_clock_text(best)})",
        "", lines)


#: The one CLAIM gate an uncontrolled arm scores, and the name the driver greps.
#: Its prefix is neither C3 nor C4 on purpose: those two are ABSENT in this mode
#: and `_verdict_call` reads their absence, so a gate wearing their number would
#: have made the verdict function believe a control ran.
UNCONTROLLED_GATE = "CU UNCONTROLLED attribution"


def gate_cu_uncontrolled(subject: list[Point]) -> Gate:
    """THE ONE LINE THAT SAYS THE READING IS UNCONTROLLED, in the one format a
    driver reads.

    WHY IT EXISTS. `scripts/h200_gaps_session.sh` summarises an arm by grepping
    `^RESULT: ` and nothing else -- it was rewritten that way precisely because
    the old summary grepped prose. Until this gate, `--control none` printed six
    RESULT lines, every one of them PASS, and exited 0 DONE; the word
    UNCONTROLLED appeared only in the report body's `## Verdict` block, which
    nothing machine-readable emits. A controlled run that CONFIRMED the headline
    and an uncontrolled run that cannot address it differed, in the session
    summary, by two missing lines -- and an ABSENCE was how the reader was meant
    to learn the arm attributed nothing. That is the same shape as a check that
    examined nothing reporting zero failures.

    WHY IT NEVER PASSES, and why that is not the defect it resembles. The
    forbidden shape is a gate that cannot FAIL: it launders an unexamined claim
    into an exit code of 0. This one is the mirror and it is safe in the
    direction that matters -- it can never contribute a PASS, so it can never
    turn CLAIM_FAIL into DONE. Its two reachable verdicts are both readings of
    the world and both are planted in `--self-test`:

      FAIL     the subject REACHED the roof, so there is no shortfall and the
               registered claim that the shortfall is the tile's is REFUTED. An
               uncontrolled arm can say that: a refutation needs no positive
               control, because the subject is one.
      UNKNOWN  the subject stopped below the roof, and nothing ran that could
               say whether the shortfall is the tile's or the fused layer's.
               `moe.bench.exit_codes`: on a CLAIM gate UNKNOWN means the claim
               was not established, which is CLAIM_FAIL and is a RESULT.

    So the arm exits 1 in both directions, the ledger row reads CLAIM_FAIL
    rather than DONE, and the one line the driver greps carries the word.
    """
    deep = multi_tile(subject)
    peak = max((p.roof_fraction for p in deep), default=None)
    reached = peak is not None and peak >= ROOF_REACHED
    lines = [
        "This arm ran with --control none. Gates C3 and C4 are ABSENT rather "
        "than UNKNOWN, because a gate that examined nothing is not a gate; this "
        "one is present so their absence has a line of its own.",
        "THE ASYMMETRY IS THE MODE. An uncontrolled run reaching the roof kills "
        "the study's central claim outright. An uncontrolled run stopping below "
        "it has measured a fraction of the roof, which is not a ceiling and may "
        "not be quoted as one.",
        "What that costs the reading is spelled under the verdict, once, in "
        "NO_CONTROL_CAVEAT; it is not repeated here.",
    ]
    return Gate(
        CLAIM, UNCONTROLLED_GATE,
        f"the subject's shortfall below the roof belongs to "
        f"BLOCK_M={SUBJECT_BLOCK_M} rather than to the fused layer it runs "
        "inside",
        "a control tile must run and C3 must score the difference; "
        "--control none scores neither, so this gate can be REFUTED by the "
        "subject alone and can never be confirmed",
        False if reached else None,
        ("no control ran; the subject REACHED the roof at "
         f"{peak:.3f} >= {ROOF_REACHED:.2f}, so there is no shortfall to "
         "attribute and the claim is refuted" if reached else
         "no control ran, so nothing cancelled and the shortfall is "
         + (f"unattributed (subject peak {peak:.3f} of the roof)"
            if peak is not None else
            "unattributed (no retained multi-tile subject points)")),
        "", lines)


# --------------------------------------------------------------------------
# The verdict, which is the two outcomes the experiment was designed around.
# --------------------------------------------------------------------------

BINDING = "CEILING BINDING AT THE PRODUCTION TILE"
NOT_BINDING = "CEILING NOT BINDING"
STILL_RISING = "STILL RISING AT THE LARGEST BATCH MEASURED"
NOT_TILE = "PLATEAU IS NOT TILE-ATTRIBUTABLE"
#: The third outcome, added 2026-09-02. A gap between the tiles is established
#: and the height of the ceiling is not, because the positive control never went
#: positive. It is a RESULT and the study may quote the gap; it is not the
#: headline and the study may not quote a ceiling.
GAP_UNLOCATED = "TILE-DEPENDENT GAP, CEILING UNLOCATED"
#: The fifth outcome, added 2026-09-03 with `--control none`. The subject
#: stopped below the roof and NOTHING cancelled, because no control ran. It is a
#: measurement of the subject's fraction of the roof and it is not a ceiling; a
#: run reaching it may quote the fraction and may quote the derivative, and may
#: not quote an attribution. It exists as its own word rather than as NOT SETTLED
#: because "we measured the production configuration and it stopped at 0.5 of
#: this card's dense rate, uncontrolled" is a result, and filing it under the
#: same word as a broken instrument would lose it.
UNCONTROLLED = "BELOW THE ROOF, UNCONTROLLED: NOT A CEILING"
UNSETTLED = "NOT SETTLED"

#: Printed under EVERY verdict this script can reach. The audit found it under
#: one of them: the NOT_TILE branch, where it embarrassed nobody, and not under
#: BINDING, where the study's headline sentence is written and where a confound
#: able to produce the whole gap belongs most.
RESIDENCY_CONFOUND = [
    "ONE CONFOUND NO GATE HERE REMOVES. The subject and the control differ in "
    "OCCUPANCY as well as in tile: at the production pin of num_stages=4 the "
    "resident-block ratio is 2:1, at 2 stages it is still 2:1, at 3 it is 3:1, "
    "and the only matched setting is 5 stages, which sm_80 cannot run and which "
    "unpins the comparison from every published arm. So no --num-stages choice "
    "disambiguates a result here, in either direction: a gap can be residency "
    "rather than tile, and an absent gap can be residency cancelling a tile "
    "effect. scripts/occupancy_vs_swizzle.py measures the residency effect "
    "directly, and its size is what to subtract before this verdict is read as "
    "being about the tile at all.",
]


#: Printed under every verdict an UNCONTROLLED arm can reach, in place of
#: `RESIDENCY_CONFOUND`. The residency paragraph is a statement about how the
#: subject and the control differ, and printing it where no control ran would
#: describe a comparison that did not happen. What replaces it is not a smaller
#: caveat but a larger one.
NO_CONTROL_CAVEAT = [
    "NO CONTROL RAN IN THIS ARM, so nothing cancelled. The align, the permute, "
    "the activation, the scatter and the clocks are all inside the timed call "
    "and none of them is subtracted: the subject's fraction of the roof is a "
    "fraction of the roof, and it is not a ceiling. The residency confound the "
    "controlled arms carry does not apply here because there is no second tile "
    "to differ from; what replaces it is the larger hole, which is that the "
    "shortfall has no attribution at all. WHY there was no control is in this "
    "run's own plan, in the CONTROL SEARCH block, computed at the BLOCK_SIZE_N "
    "this run was pinned to; it is not restated here, because this paragraph "
    "is a constant and the search is not. The constant used to end 'a control "
    "at this BLOCK_SIZE_N does not exist', which is true at 256 and false at "
    "64, and it was printed at both.",
]


def verdict(gates: list[Gate]) -> tuple[str, list[str]]:
    """One sentence, derived from the gates and from nothing else.

    Every branch returns through one place at the bottom, which is how the
    caveat reaches every one of them: a confound appended by hand in each branch
    is a confound that goes missing from the branch nobody revisits, and that is
    exactly what happened here.

    WHICH CAVEAT, decided from the PRESENCE of `gate_cu_uncontrolled` and not
    from a flag. It used to be decided from the ABSENCE of C3, which was true
    and was inferred: a fact read off a missing thing changes meaning the moment
    anything else can be missing. The CU gate is emitted by `analyse` under
    `--control none` and under nothing else, so its presence is the fact and it
    is also the line the session driver greps.
    """
    call, why = _verdict_call(gates)
    uncontrolled = any(g.name == UNCONTROLLED_GATE for g in gates)
    return call, why + (NO_CONTROL_CAVEAT if uncontrolled
                        else RESIDENCY_CONFOUND)


def _verdict_call(gates: list[Gate]) -> tuple[str, list[str]]:
    """The branch table itself. See `verdict`, which is the entry point."""
    by = {g.name.split()[0]: g for g in gates}
    invalid = [g for g in gates if g.kind == VALIDITY and g.passed is not True]
    c1, c2, c3, c4 = by.get("C1"), by.get("C2"), by.get("C3"), by.get("C4")
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
            "and it is not the ceiling this study proposed."]
    if c3 is None and c1 is not None and c2 is not None:
        # `analyse` omits C3 and C4 only under `--control none`. Both remaining
        # claim gates PASSED, which for C1 means the subject stayed BELOW the
        # roof: the reading that needs a control and does not have one. The two
        # branches above have already taken the two readings that do not need
        # one -- C1 FAIL is a refutation and C2 FAIL is a sweep too shallow --
        # so this arm is asymmetric by construction and says so.
        return UNCONTROLLED, [
            f"At BLOCK_M={SUBJECT_BLOCK_M} throughput rose and then stopped "
            "below this card's own measured dense bf16 rate, at the "
            "configuration vLLM ships. THAT IS WHERE THE READING STOPS.",
            "No control ran, so the shortfall is not attributed: it may be an "
            "arithmetic-intensity ceiling on the tile, and it may be whatever "
            "share of the fused layer is spent outside its GEMMs. This run "
            "does not distinguish them and may not be quoted as if it did.",
            "THIS ARM CAN REFUTE THE HEADLINE AND CANNOT CONFIRM IT. Had the "
            "subject reached the roof, C1 would have FAILED and the study's "
            "central claim would be dead with no control needed. It did not, "
            "so the claim survives unconfirmed, which is not the same as "
            "supported.",
        ]
    if c1 is not None and c1.passed and c2 is not None and c2.passed and \
            c3 is not None and c3.passed:
        if c4 is None:
            return UNSETTLED, [
                "C1, C2 and C3 all passed, and C4 -- whether the control ever "
                "reached the roof -- was not scored. The headline may not be "
                "issued on a positive control that was never read: without C4 "
                "this is a gap of unknown height."]
        if c4.passed is not True:
            return GAP_UNLOCATED, [
                f"At BLOCK_M={SUBJECT_BLOCK_M} throughput rose and then stopped "
                "below this card's own measured dense rate, and the control "
                "tile did materially better on the same batches, so the "
                "difference is real and it is tile-dependent.",
                "But the control did not reach the roof either, so the height "
                "of the ceiling is NOT located: something below the roof binds "
                "both tiles, this measurement does not say what, and the study "
                "may quote the gap and may not quote a ceiling.",
                "This is a result and not a failure. It is also the outcome the "
                "published corpus predicts, where no tile exceeds 0.54 of the "
                "roof on the H200 or 0.64 on the A100."]
        return BINDING, [
            f"At BLOCK_M={SUBJECT_BLOCK_M}, the tile vLLM actually runs "
            "multi-tile, throughput rose and then stopped below this card's own "
            "measured dense rate, while a tile whose cap is far above the ridge "
            "did materially better on the same batches AND reached the roof, so "
            "the roof is somewhere this instrument was shown able to see a tile "
            "arrive.",
            "The ceiling is real and it binds in the regime production runs. "
            "Nothing in this sentence came from a fit."]
    return UNSETTLED, ["The claim gates did not resolve; read them individually."]


# --------------------------------------------------------------------------
# Predictions, registered and printed with numbers before anything is measured.
# --------------------------------------------------------------------------

def predictions_text(cfg, roof: Roof, b: int, control_block_m: int | None,
                     rows: list[int], doublings: float, *, block_n: int) -> str:
    """The registered predictions, and the three that do not exist without a control.

    P1, P4 and P5 are all statements about the control, so `--control none`
    replaces them with the one sentence that is true instead of printing three
    predictions nothing will score. A registered prediction with no measurement
    behind it is how a report comes to look complete while its headline is
    unreachable.
    """
    lo, hi = predicted_plateau_band(roof.ridge, b)
    cap_s = SWEEP.ai_cap(SUBJECT_BLOCK_M, max(ALPHA_128_BAND), b)
    onset = onset_tokens(cfg, SUBJECT_BLOCK_M)
    deep = [r for r in rows if r > SUBJECT_BLOCK_M]
    if control_block_m is None:
        control_predictions = [
            "  P1  THERE IS NO CONTROL IN THIS ARM, so P1, P4 and P5 -- the "
            "separation, the control's",
            "      derivative and whether it reaches the roof -- do not exist "
            "and are not registered.",
            "      Gates C3 and C4 are not scored, and the headline cannot be "
            "issued by this run at all.",
        ]
        fork = [
            "  THE FORK, in two ways, and only two. Throughput reaches the "
            "roof -> the ceiling is not binding and",
            "  the study's central claim is about a regime that does not "
            "occur, which needs no control to say.",
            "  It stops below the roof -> that is a fraction of the roof and "
            "NOT a ceiling: nothing here separates",
            "  the tile from the fused layer it runs inside. THIS ARM CAN "
            "REFUTE THE CLAIM AND CANNOT CONFIRM IT.",
        ]
    else:
        cap_c = symmetric_cap(cfg, control_block_m, block_n, SWEEP.ALPHA, b)
        cap_sub = symmetric_cap(cfg, SUBJECT_BLOCK_M, block_n, SWEEP.ALPHA, b)
        control_predictions = [
            f"  P1  the control BLOCK_M={control_block_m} beats the subject by "
            f"at least {CONTROL_SEPARATION:.2f} of the roof",
            f"      ({CONTROL_SEPARATION * roof.tflops:.0f} TFLOP/s). Its cap "
            f"is {cap_c:.1f} Op/B against the subject's {cap_sub:.1f} and a "
            f"ridge of {roof.ridge:.1f}: {cap_c / roof.ridge:.2f}x, so it has "
            f"headroom on the subject and",
            f"      {'still ' if cap_c < roof.ridge else ''}"
            + (f"a ceiling of its own at {cap_c / roof.ridge:.2f}x the ridge, "
               f"under the {CONTROL_CAP_MARGIN:.2f}x a control with none would "
               "need. P5 below is registered to FAIL on that arithmetic alone."
               if cap_c < roof.ridge else
               "no ceiling of its own to hit."),
        ]
        fork = [
            "  THE FORK, in three ways. Throughput reaches the roof -> the "
            "ceiling is not binding and the study's",
            "  central claim is about a regime that does not occur. It "
            "plateaus below the roof and the control",
            "  plateaus with it -> the shortfall is the fused layer's, not the "
            "tile's. It plateaus below the roof,",
            "  the control beats it by the separation, and the control reaches "
            "the roof -> the ceiling is real and",
            "  binds where production runs; the control beats it and does NOT "
            "reach the roof -> a tile-dependent",
            "  gap with the ceiling unlocated. All four are results; only one "
            "is the study's.",
        ]
    tail = ([
        "  P4  the control does NOT plateau below the roof over the same span.",
        f"  P5  the control REACHES the roof ({ROOF_REACHED:.2f} of "
        f"{roof.tflops:.1f} = {ROOF_REACHED * roof.tflops:.0f} TFLOP/s), which "
        "is what makes the subject's",
        "      shortfall a located ceiling rather than a gap of unknown "
        "height. REGISTERED AS THE ONE THIS",
        "      STUDY EXPECTS TO FAIL: no tile in the published corpus reaches "
        "0.95 of ridge x bandwidth,",
        "      the BLOCK_M=256 control peaking at 0.48-0.54 of the roof on the "
        "H200 and 0.56-0.64 on the A100,",
        f"      and this control's own cap is "
        f"{symmetric_cap(cfg, control_block_m, block_n, SWEEP.ALPHA, b) / roof.ridge:.2f}"
        "x the ridge, so the arithmetic registers the same failure the corpus "
        "does.",
    ] if control_block_m is not None else [])
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
    ] + control_predictions + [
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
    ] + tail + [""] + fork)


# --------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    model: str
    dtype: str
    pinned: dict
    control_block_m: int | None
    subject_rows: list[int]
    control_rows: list[int]
    reps: int
    warmup_ms: float
    trials: int
    l2_flush: bool
    cell_budget_ms: float
    estimated_seconds: float
    cells: int
    resources: dict
    refusals: dict
    instrument: str

    def lines(self, cfg) -> list[str]:
        return [
            f"model        {cfg.name} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       {self.pinned}",
            f"subject      BLOCK_M={SUBJECT_BLOCK_M} at rows {self.subject_rows}",
            f"             tokens "
            f"{[SWEEP.tokens_for_rows(cfg, r) for r in self.subject_rows]}",
            (f"control      BLOCK_M={self.control_block_m} at rows "
             f"{self.control_rows} -- the same batches, a tile with no ceiling"
             if self.control_block_m is not None else
             "control      NONE. --control none was given: this arm measures "
             "the subject alone, so it can REFUTE the claim and can never "
             "confirm it"),
            f"repeats      {self.reps} round-robin passes per tile, so a "
            "throttled cell can be told from a tile effect",
            f"timing       {self.instrument}",
            f"             {self.warmup_ms:.0f} ms of delivered GPU load to "
            f"warm, then {self.trials} queue-deep trials each sized by the "
            f"instrument to hold {self.cell_budget_ms:.0f} ms of kernel time",
            "             L2 " + ("flushed between iterations, as the roof was"
                                  if self.l2_flush else
                                  "NOT flushed, so these cells are not "
                                  "comparable with the roof"),
            f"cells        {self.cells} timings",
            f"estimate     {self.estimated_seconds:.0f} s of GPU at the model's "
            "own timings, excluding compiles and allocation",
        ] + [f"resources    {r.render().strip()}"
             for r in self.resources.values()]


def build_plan(args, cfg, b: int, roof: Roof, capability) -> Plan:
    """The grid, the resource bill and the cost, at the instrument that will run.

    THE COST IS THE SWEEP'S OWN `estimated_seconds`, called and not copied, and
    it is called with `warmup_ms=` and `trials=` -- the keywords that price
    `time_kernel` -- rather than with `iters=`/`warmup=`, which price the
    retired per-call loop. That function REFUSES a call that mixes the two
    because the two answers differ by about 5x, and the whole reason an operator
    reads this number is to decide whether to rent the pod.
    """
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  num_warps=args.num_warps, GROUP_SIZE_M=args.group_m,
                  BLOCK_SIZE_N=args.block_n, BLOCK_SIZE_K=args.block_k)
    subject_rows = doubling_rows(cfg, args.r_min, args.r_max, SUBJECT_BLOCK_M)
    control_rows = ([] if args.control is None
                    else [r for r in subject_rows if r % args.control == 0])
    tiles = ((SUBJECT_BLOCK_M,) if args.control is None
             else (SUBJECT_BLOCK_M, args.control))
    resources, refusals = SWEEP.tile_resource_plan(pinned, tiles, b, capability)
    # THE PER-BLOCK REGISTER FILE, merged in beside the sweep's per-thread bill
    # rather than replacing it. The two catch different tiles: the sweep's
    # catches 256x256 at 8 warps and lets it through at 16, and this one catches
    # it at every warp count. A refusal from either is a refusal.
    for tile in tiles:
        why = register_file_refusal(tile, args.block_n)
        if why:
            refusals[tile] = "; ".join(filter(None, (refusals.get(tile), why)))
    total = 0.0
    for bm, rows in ((SUBJECT_BLOCK_M, subject_rows),
                     (args.control, control_rows)):
        if bm is None:
            continue
        total += args.reps * SWEEP.estimated_seconds(
            cfg, rows, [bm], alpha=args.alpha, ridge=roof.ridge,
            bandwidth_gbps=roof.bandwidth_gbps, b=b,
            cell_budget_ms=args.cell_budget_ms, warmup_ms=args.warmup,
            trials=args.trials)
    return Plan(args.model, args.dtype, pinned, args.control, subject_rows,
                control_rows, args.reps, args.warmup, args.trials,
                not args.no_l2_flush, args.cell_budget_ms, total,
                args.reps * (len(subject_rows) + len(control_rows)),
                resources, refusals, instrument_name())


def symmetric_cap(cfg, block_m: int, block_n: int, alpha: float, b: int) -> float:
    """This tile's arithmetic-intensity ceiling, in Op/B, under the model this
    file believes.

    `moe.bench.ai_model.cap` is

        2 / (b (alpha_b/BM + alpha_a/BN + 1/K))

    SYMMETRIC in the two tile dimensions and carrying the output write. The
    scalar form the sibling sweep quotes, `2 BM / (alpha b)`, has neither
    BLOCK_SIZE_N nor 1/K in it.

    WHY THIS FUNCTION EXISTS, and it is an audit finding rather than a
    refactor. Until 2026-09-03 `control_feasibility` and `check_control` priced
    every candidate with the scalar form, twenty lines above a paragraph on the
    same printed page that named that form as the trap. The two disagree by 3.7x
    at the geometry the study is about: a 256x256 tile printed as `cap 459 Op/B
    = 2.82x the ridge` caps at 124.12 Op/B = 0.76x under this one, and 512x256,
    printed as 5.64x, is 163.84 = 1.01x. The overstated number was not
    decoration: it was what the CONTROL_CAP_MARGIN gate ran on, so the check
    that decided WHICH TILE MAY BE A CONTROL was the one the file argued was
    invalid.

    ALPHA IN BOTH SLOTS, and that is a stated approximation rather than a
    measurement. `alpha_b` is the weight re-read fraction and `alpha_a` the
    activation re-read fraction; this study measures neither separately, it
    measures a FITTED alpha that blends them (EXA), and `ai_model.cap_from_
    fitted` exists because 2BM/(alpha_fitted b) is not a cap. Putting the same
    alpha in both slots is the one substitution that keeps the SHAPE of the
    model while refusing to invent a second number. It is used to CHOOSE and to
    DISCLOSE, never to score: no gate in this file reads a cap.

    A miss fraction outside [0, 1] is refused by `ai_model` rather than clamped,
    and `_main` refuses `--alpha` outside that range before anything is priced.
    """
    return AI.cap(cfg.intermediate_size, cfg.hidden_size, block_m=block_m,
                  block_n=block_n, alpha_b=alpha, alpha_a=alpha, b=b)


#: Tile dimensions a search over "could ANY tile reach the roof" runs across.
#: Powers of two because Triton pins block shapes to them, and up to 2048 x 1024
#: because the answer has to be an exhaustive one to be worth printing: a search
#: that stopped at the candidates this file already tries would be reporting its
#: own candidate list back as a fact about the hardware.
TILE_SEARCH_BLOCK_M = (16, 32, 64, 128, 256, 512, 1024, 2048)
TILE_SEARCH_BLOCK_N = (16, 32, 64, 128, 256, 512, 1024)


def predicted_separation(control_cap: float, subject_cap: float,
                         ridge: float) -> float:
    """What the cap model predicts C3 will read, as a fraction of the roof.

    CLAMPED AT THE ROOF AT EACH END, the same clamp `predicted_plateau_band`
    applies and for the same reason: a kernel cannot exceed the roof however
    large its arithmetic intensity, so a cap of twice the ridge does not predict
    twice the roof. Unclamped, a 1024x256 tile printed a predicted separation of
    +1.163 of the roof -- more throughput than the card has -- which is the
    shape that once made a gate unfailable.

    It is a PREDICTION and it gates nothing: `check_control` refuses on the
    register file and on headroom, never on this. It is here so that the number
    C3 will report has something on the record to be compared with.
    """
    return min(control_cap / ridge, 1.0) - min(subject_cap / ridge, 1.0)


def ridge_reaching_tiles(cfg, ridge: float, alpha: float, b: int
                         ) -> tuple[list[tuple[int, int, float, int]],
                                    list[tuple[int, int, float, int]]]:
    """`(reach, buildable)`: every tile whose cap clears the ridge, and which of
    them a thread block can hold.

    THE COMPUTATION BEHIND THIS FILE'S HARDEST SENTENCE. C4 -- the control
    REACHES the roof -- is a necessary condition for the study's headline, and a
    tile can only reach a compute roof if its AI cap clears the ridge. So the
    question "can this experiment reach CEILING BINDING at all" is decidable
    before any pod is rented, by asking whether ANY tile both clears the ridge
    and fits the per-block register file.

    On both models this study measures, at every alpha it has measured, the
    answer is no, and the two limits coincide almost exactly: the smallest
    accumulator among the tiles that clear the H200's ridge is 65536 registers
    at alpha 0.558, which is the whole per-block file, and 131072 at alpha 1.
    That is why the finding is stated at every BLOCK_SIZE_N and not only at 256:
    at 256 there is no control at all, and at 64 and 128 there is one that can
    never reach the roof.

    The shared-memory half is deliberately NOT consulted. It depends on
    num_stages and on the card, and a tile refused on the register file is
    refused on every card this study can reach, so the stronger and simpler
    statement is the one worth printing.
    """
    reach: list[tuple[int, int, float, int]] = []
    for block_m in TILE_SEARCH_BLOCK_M:
        for block_n in TILE_SEARCH_BLOCK_N:
            cap = symmetric_cap(cfg, block_m, block_n, alpha, b)
            if cap >= ridge:
                reach.append((block_m, block_n, cap, block_m * block_n))
    buildable = [t for t in reach if t[3] < REGISTERS_PER_BLOCK]
    return reach, buildable


def accumulator_registers(block_m: int, block_n: int) -> int:
    """One CTA's fp32 accumulator, in 32-bit registers: `BM x BN`.

    `tl.dot` accumulates in fp32 and the accumulator is register resident, so a
    thread block holds one 32-bit register per output element it owns. WARP
    COUNT DOES NOT APPEAR, and that is the whole reason this sits beside the
    sweep's per-thread bill rather than inside it: warps redistribute this
    total across more threads, they do not reduce it.
    """
    return block_m * block_n


def register_file_refusal(block_m: int, block_n: int) -> str:
    """Empty when a CTA's accumulator leaves room for the rest of the kernel.

    THE LIMIT THE PER-THREAD CHECK CANNOT SEE. `block_m_crossing_sweep`
    refuses a tile whose accumulator needs more than 255 registers per thread,
    and at num_warps=8 that catches 256x256 by one register. It stops catching
    it at num_warps=16, where the same 65536 registers are spread over 512
    threads and read 128 each -- comfortably inside 255, and still the entire
    per-block register file. A block that spends 100% of the file on its
    accumulator has none left for its pointers, its indices or its K loop, so
    the compiler spills and the timing is the timing of a spilled kernel.

    STRICTLY LESS THAN, and no tuned margin above it. The exact boundary is a
    fact; where the real kernel starts spilling is somewhere below it and is
    not knowable from here. So this is a NECESSARY condition and not a
    sufficient one, and it is stated that way rather than padded to a number
    that would look like a measurement.
    """
    acc = accumulator_registers(block_m, block_n)
    if acc < REGISTERS_PER_BLOCK:
        return ""
    return (f"the {block_m}x{block_n} fp32 accumulator is {acc} 32-bit "
            f"registers per thread block against a per-block file of "
            f"{REGISTERS_PER_BLOCK}, so the accumulator ALONE is "
            f"{acc / REGISTERS_PER_BLOCK:.0%} of the file and nothing is left "
            "for the pointers, the indices or the K loop. No num_warps fixes "
            "this: warps divide the total across more threads, they do not "
            "shrink it, and num_stages and BLOCK_SIZE_K do not touch it at "
            "all. The kernel spills to local memory and its time is not the "
            "time of the tiling this sweep is about")


def control_feasibility(cfg, *, block_n: int, block_k: int, dtype_bytes: int,
                        capability, alpha: float, ridge: float
                        ) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Every tile that could serve as a control at this BLOCK_SIZE_N, with its bill.

    Returns `(fits, lines)`: the `(BLOCK_M, num_warps, num_stages)` triples under
    which BOTH tiles can be pinned, and the arithmetic for a reader. An empty
    `fits` is an answer and usually the interesting one.

    BOTH TILES, NOT THE CONTROL ALONE. A pin that fits the control and not the
    subject buys a measurement that cannot be compared, so each candidate is
    billed through `SWEEP.tile_resource_plan` over the pair, which is the same
    function the plan's own refusals come from.

    WITHOUT A CAPABILITY ONLY HALF THE BILL EXISTS. `tile_resources` leaves
    `smem_limit_bytes` unset off a device and `smem_fits` is then None, which
    this treats as NOT a fit -- never as one. The register half is decidable
    everywhere, so a tile refused on registers is refused from a laptop with no
    flag, which is how the finding below is reachable without renting anything.

    WHAT DECIDES A FIT IS THE HARDWARE, AND ONLY THE HARDWARE. Until 2026-09-03
    a candidate whose cap sat under `CONTROL_CAP_MARGIN` was skipped here before
    its bill was ever computed, on a cap from the scalar form -- so the search
    both priced with the retracted model and let that price decide which tiles
    an operator was even shown. The cap is now printed beside each candidate
    under `symmetric_cap` and it disqualifies nothing: whether a tile can be
    HELD is a fact about the register file, whether it can REACH THE ROOF is
    what gate C4 measures, and a search that refused the second in advance would
    be refusing an arm for the answer it is expected to give. C4 is
    pre-registered to fail; that is a prediction on the record, not a reason to
    cancel the measurement.
    """
    where = (f"sm_{capability[0]}{capability[1]}" if capability else
             "NO --capability GIVEN, so the shared-memory half is undecidable "
             "here and nothing will be called a fit on the register half alone")
    smem_limit = SWEEP.SMEM_PER_BLOCK_BYTES.get(capability) if capability else None
    lines = [
        f"CONTROL SEARCH at BLOCK_SIZE_N={block_n}, BLOCK_SIZE_K={block_k}, "
        f"{dtype_bytes} B per element, {where}",
        "  the bills: accumulator BLOCK_M x BLOCK_SIZE_N fp32 registers per "
        f"block against {REGISTERS_PER_BLOCK} and "
        f"{SWEEP.MAX_REGISTERS_PER_THREAD} per thread; shared memory "
        f"num_stages x (BLOCK_M x BLOCK_K + BLOCK_K x BLOCK_N) x {dtype_bytes} "
        "B against "
        + (f"{smem_limit / 1024:.0f} KiB" if smem_limit else "an unknown limit"),
    ]
    subject_cap = symmetric_cap(cfg, SUBJECT_BLOCK_M, block_n, alpha, dtype_bytes)
    lines.append(
        f"  the subject BLOCK_M={SUBJECT_BLOCK_M} caps at {subject_cap:.1f} "
        f"Op/B = {subject_cap / ridge:.2f}x the ridge, at alpha {alpha:.3f} in "
        "both slots of moe.bench.ai_model.cap. A control has to sit ABOVE that "
        "line and be held by one block")
    fits: list[tuple[int, int, int]] = []
    for block_m in CONTROL_CANDIDATE_BLOCK_M:
        acc = accumulator_registers(block_m, block_n)
        cap = symmetric_cap(cfg, block_m, block_n, alpha, dtype_bytes)
        lines.append(
            f"  BLOCK_M={block_m}: cap {cap:.1f} Op/B = {cap / ridge:.2f}x the "
            f"ridge ({cap - subject_cap:+.1f} Op/B on the subject, a predicted "
            f"separation of {predicted_separation(cap, subject_cap, ridge):+.3f}"
            f" of the roof); accumulator {block_m}x{block_n} = {acc} registers "
            f"per block, {acc / REGISTERS_PER_BLOCK:.0%} of the file")
        if cap < ridge:
            lines.append(
                f"      ITS OWN CEILING IS BELOW THE RIDGE ({cap / ridge:.2f}x "
                f"of the {CONTROL_CAP_MARGIN:.2f}x a control with no ceiling "
                "would need), so C4 is pre-registered to FAIL and the best "
                f"verdict this pairing can reach is {GAP_UNLOCATED!r}. NOT a "
                "refusal: the gap it measures is a result, and cancelling an "
                "arm for its expected answer is not how a claim is registered")
        if register_file_refusal(block_m, block_n):
            lines.append(
                "      REFUSED AT EVERY WARP AND STAGE COUNT: "
                + ", ".join(f"{w} warps = {acc / (32 * w):.0f} reg/thread"
                            for w in CONTROL_WARP_COUNTS)
                + f", and {acc} of {REGISTERS_PER_BLOCK} per block in all of "
                "them. Warps divide this total, they do not shrink it; stages "
                "and BLOCK_SIZE_K do not touch it.")
            continue
        for warps in CONTROL_WARP_COUNTS:
            per_thread = acc / (32.0 * warps)
            if per_thread > SWEEP.MAX_REGISTERS_PER_THREAD:
                continue
            for stages in CONTROL_STAGE_COUNTS:
                pinned = dict(SWEEP.FIXED, num_stages=stages, num_warps=warps,
                              BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k)
                tiles = (SUBJECT_BLOCK_M, block_m)
                bills, refusals = SWEEP.tile_resource_plan(
                    pinned, tiles, dtype_bytes, capability)
                if refusals or any(register_file_refusal(t, block_n)
                                   for t in tiles):
                    continue
                if any(bills[t].smem_fits is not True for t in tiles):
                    continue
                fits.append((block_m, warps, stages))
                lines.append(
                    f"      FITS at --num-warps {warps} --num-stages {stages}: "
                    f"{per_thread:.0f} reg/thread, "
                    f"{bills[block_m].smem_bytes / 1024:.0f} KiB of shared "
                    f"memory against {bills[block_m].smem_limit_bytes / 1024:.0f}"
                    " KiB, and the subject fits the same pin")
    if not fits:
        lines.append(
            f"  NO TILE ABOVE BLOCK_M={SUBJECT_BLOCK_M} CAN BE PINNED AT "
            f"BLOCK_SIZE_N={block_n}" + ("." if capability else
                                         " on the register bill alone."))
    lines += binding_reachability(cfg, ridge, alpha, dtype_bytes)
    return fits, lines


def binding_reachability(cfg, ridge: float, alpha: float, b: int) -> list[str]:
    """Whether ANY tile could reach the roof on this card, at any BLOCK_SIZE_N.

    Printed under every control search, because the question the search answers
    -- which tile can be pinned here -- is the smaller of the two, and an
    operator who reads only the smaller one comes away thinking a different
    BLOCK_SIZE_N would rescue the headline. It would not. This is the sentence
    the 2026-09-03 finding got half right: it named BLOCK_SIZE_N=256 as the
    place the study cannot confirm its headline, when the register file and the
    ridge coincide so closely that no BLOCK_SIZE_N is.
    """
    reach, buildable = ridge_reaching_tiles(cfg, ridge, alpha, b)
    if buildable:
        best = min(buildable, key=lambda t: t[3])
        return [
            f"  A TILE THAT COULD REACH THE ROOF DOES EXIST: {best[0]}x{best[1]}"
            f" caps at {best[2]:.1f} Op/B against a ridge of {ridge:.2f} and "
            f"needs {best[3]} accumulator registers of {REGISTERS_PER_BLOCK}. "
            "C4 is reachable at that geometry and the headline is not "
            "pre-refuted."]
    smallest = min((t[3] for t in reach), default=0)
    return [
        f"  AND NO ARM OF THIS EXPERIMENT REACHES {BINDING!r} ON THIS CARD, AT "
        "ANY BLOCK_SIZE_N. Over "
        f"{len(TILE_SEARCH_BLOCK_M) * len(TILE_SEARCH_BLOCK_N)} power-of-two "
        f"tiles up to {max(TILE_SEARCH_BLOCK_M)}x{max(TILE_SEARCH_BLOCK_N)}, "
        f"{len(reach)} clear the ridge of {ridge:.2f} Op/B at alpha "
        f"{alpha:.3f} and the smallest accumulator among them is {smallest} "
        f"registers, against a per-block file of {REGISTERS_PER_BLOCK}. So no "
        "tile a thread block can hold is capable of reaching a compute roof "
        "here, C4 fails for every control that can be built, and the best "
        f"verdict any pairing can reach is {GAP_UNLOCATED!r}. The two limits "
        "very nearly coincide, which is the finding: the register file runs "
        "out exactly where the arithmetic intensity would have become enough."]


#: The BLOCK_SIZE_N at which a BLOCK_M=256 control both fits one thread block
#: and sits above the subject: alternative 3 below, and the only geometry in
#: this study where a control can be built at all on sm_90. Named rather than
#: written into a sentence, because two places quote it and a literal in prose
#: is a literal that goes stale in one of them.
CONTROL_FITS_BLOCK_N = 128


def no_control_finding(cfg, ridge: float, alpha: float, b: int, *,
                       block_n: int, fits: list, capability) -> list[str]:
    """WHY THE HEADLINE CANNOT BE CONFIRMED HERE, and what would confirm it.

    A FUNCTION OF THE GEOMETRY, and that is the 2026-09-03 audit's finding
    against the 2026-09-03 finding. This was a module constant whose first
    sentence read "at BLOCK_SIZE_N=256 there is no positive control", and it was
    printed under every refusal at every BLOCK_SIZE_N -- including 64, on a page
    that had just listed twelve pins at which a BLOCK_M=256 control FITS. The
    search on that page said a control exists and the paragraph under it said
    none does. Both are computed now, from the same numbers, so they cannot
    disagree.

    THREE HEADS AND ONE BODY. Whether a control fits here is a fact about this
    BLOCK_SIZE_N and can be undecidable off a device; whether any tile could
    reach the roof is a fact about the card and is decidable everywhere. The
    first is the head, the second is the body, and the body is why the answer
    does not change when the head does.

    AND THE REGISTER HALF IS DECIDABLE WITHOUT A CAPABILITY, which is why the
    undecidable head is the LAST branch and not the first. At BLOCK_SIZE_N=256
    every candidate is refused on the per-block register file alone, on every
    card from sm_70 to sm_100, so "there is no control here" is a fact a laptop
    can state -- and the session driver's own dry run of that arm passes no
    --capability. A head that went undecidable there would have made the
    strongest sentence in this file conditional on a flag nothing supplies.
    """
    reach, buildable = ridge_reaching_tiles(cfg, ridge, alpha, b)
    subject_cap = symmetric_cap(cfg, SUBJECT_BLOCK_M, block_n, alpha, b)
    register_refuses_all = all(register_file_refusal(bm, block_n)
                               for bm in CONTROL_CANDIDATE_BLOCK_M)
    if register_refuses_all:
        head = (
            "THE STUDY CANNOT CONFIRM ITS HEADLINE AT vLLM'S SHIPPED "
            "CONFIGURATION ON sm_90. At a fixed BLOCK_SIZE_N the only way to "
            "buy a tile more headroom is a larger BLOCK_SIZE_M, the only power "
            "of two above 128 that Triton will pin is 256, and a 256x256 fp32 "
            f"accumulator is {accumulator_registers(256, 256)} 32-bit "
            "registers: the entire per-block register file of every "
            "architecture from sm_70 to sm_100. No num_warps, no num_stages and "
            f"no BLOCK_SIZE_K changes that total. So at BLOCK_SIZE_N={block_n} "
            "there is no positive control, and the shortfall a subject-only run "
            "measures has nothing to be attributed against. This half needs no "
            "--capability: the register file is the same on every card this "
            "study can reach.")
    elif capability is None:
        head = (
            f"WHETHER A CONTROL FITS AT BLOCK_SIZE_N={block_n} IS UNDECIDABLE "
            "FROM HERE: no --capability was given, so the shared-memory half of "
            "the bill has no limit to be checked against and nothing below is a "
            "statement that the hardware left this arm without a control. The "
            "register half does NOT refuse every candidate at this "
            "BLOCK_SIZE_N, so the answer turns on the half that is missing. "
            "Pass --capability to decide it.")
    elif fits:
        head = (
            f"A CONTROL DOES FIT AT BLOCK_SIZE_N={block_n}: the search above "
            f"names {len(fits)} pin(s), the first being BLOCK_M={fits[0][0]} at "
            f"--num-warps {fits[0][1]} --num-stages {fits[0][2]}. So the "
            "hardware is NOT what leaves this arm without a control, and no "
            "sentence here may be quoted as though it were.")
    else:
        head = (
            f"NO CONTROL FITS AT BLOCK_SIZE_N={block_n}, and the bill that "
            "refuses it is the SHARED MEMORY one rather than the register file: "
            "some candidate above passes the accumulator check and no pin in "
            f"{CONTROL_STAGE_COUNTS} x {CONTROL_WARP_COUNTS} brings its stages "
            "inside this card's per-block limit. That is a statement about this "
            "--num-stages and this card, not the architecture-wide one the "
            "BLOCK_SIZE_N=256 refusal makes.")
    control_cap = symmetric_cap(cfg, DEFAULT_CONTROL_BLOCK_M,
                                CONTROL_FITS_BLOCK_N, alpha, b)
    fits_subject_cap = symmetric_cap(cfg, SUBJECT_BLOCK_M,
                                     CONTROL_FITS_BLOCK_N, alpha, b)
    transposed = symmetric_cap(cfg, DEFAULT_CONTROL_BLOCK_M, block_n // 2,
                               alpha, b) if block_n >= 2 else 0.0
    return [head] + [
        "WHAT WOULD CONFIRM IT, in the order of how much each concedes:",
        "  1. A card whose per-block register file exceeds "
        f"{REGISTERS_PER_BLOCK * 4 // 1024} KiB. No NVIDIA architecture this "
        "study can reach has one; the file has been 64 K registers per SM and "
        "per block since sm_70.",
        "  2. A kernel whose accumulator is not resident at full BM x BN -- a "
        "split-N or a two-pass formulation. vLLM's fused_moe kernel is not one, "
        "so such a run would measure a different kernel and could not be quoted "
        "about the one production runs.",
        f"  3. The subject at BLOCK_SIZE_N={CONTROL_FITS_BLOCK_N}, where a "
        f"BLOCK_M={DEFAULT_CONTROL_BLOCK_M} control fits at 8 warps and 4 "
        "stages (128 registers per thread, 192 KiB of shared memory) and does "
        f"have headroom on the subject there: {control_cap:.1f} Op/B against "
        f"{fits_subject_cap:.1f}, a predicted separation of "
        f"{predicted_separation(control_cap, fits_subject_cap, ridge):+.3f} of "
        "the roof. IT "
        f"STILL DOES NOT CONFIRM THE HEADLINE: {control_cap:.1f} is "
        f"{control_cap / ridge:.2f}x the ridge, so that control is memory bound "
        "by construction, C4 fails before the run and the arm lands "
        f"{GAP_UNLOCATED!r}. And CHECKED IN THE SHIPPED FILES, NOT ASSUMED: "
        "moe/bench/hardware/vllm_configs holds no BLOCK_SIZE_M=128 entry at "
        f"BLOCK_SIZE_N={CONTROL_FITS_BLOCK_N} for either model this study "
        "measures, on either dtype -- every 128 entry on the H200 ships "
        "BLOCK_SIZE_N=256. So that arm buys a GAP at a configuration nobody "
        "ships, and it may not be quoted as the production claim.",
        f"  4. A control at BLOCK_M={DEFAULT_CONTROL_BLOCK_M} with a "
        f"BLOCK_SIZE_N of its OWN, {block_n // 2}, against this arm's subject "
        f"at {block_n}. REFUSED, and for a reason that is NOT item 3's: "
        "moe.bench.ai_model.cap is 2 / (b (alpha_b/BM + alpha_a/BN + 1/K)), "
        f"SYMMETRIC in the two tile dimensions, so {DEFAULT_CONTROL_BLOCK_M}x"
        f"{block_n // 2} has EXACTLY this subject's cap -- {transposed:.2f} "
        f"Op/B against {subject_cap:.2f} on {cfg.name} at alpha {alpha:.3f} in "
        "both slots -- along with the same accumulator, the same shared memory "
        "and the same residency. Its headroom is zero, so C3's difference is "
        "zero by construction; item 3 changes the SUBJECT too and does buy "
        "headroom. block_m_crossing_sweep.ai_cap would have called this one 2x "
        "the headroom, because that form is 2 BM / (alpha b) and BLOCK_SIZE_N "
        "does not appear in it.",
        "  5. `--control none`: the subject alone, which can REFUTE this claim "
        "and can never confirm it. That is the only one of the five this "
        f"session can buy at BLOCK_SIZE_N={block_n}, and what it buys is stated "
        "where it is offered.",
        "NONE OF THE FIVE CONFIRMS THE HEADLINE ON THIS CARD, AND NO "
        "BLOCK_SIZE_N DOES EITHER. Items 1 and 2 need hardware or a kernel this "
        "study cannot reach, item 3 buys a gap and not a ceiling, item 4 buys "
        "nothing, and item 5 concedes the question. The reachability line in "
        "the CONTROL SEARCH above is why the list has no sixth entry: of the "
        f"{len(reach)} power-of-two tiles whose cap clears this card's ridge, "
        f"the smallest accumulator is {min(t[3] for t in reach)} registers "
        f"against a per-block file of {REGISTERS_PER_BLOCK}"
        + (", so C4 is pre-registered to FAIL for every control this hardware "
           "can hold. The registered outcome of this experiment on sm_90 is a "
           "fraction of the roof, with or without a control."
           if not buildable else
           f", and {len(buildable)} of them do fit, so a geometry that can "
           "reach C4 exists after all -- the search names it."),
    ]


def control_ceiling_line(cfg, control_block_m: int, alpha: float, ridge: float,
                         b: int, *, block_n: int) -> str:
    """The dry run's closing sentence about the control, and it used to be wrong.

    It read "The control's cap is 2.82x the ridge, so it has no ceiling of its
    own to hit", from `SWEEP.ai_cap`, in the last section an operator reads
    before deciding to rent. Under `symmetric_cap` the same tile at
    BLOCK_SIZE_N=64 is 0.55x, which is the opposite sentence: the control has a
    ceiling, it is below the ridge, and C4 will fail. Printing the true number
    here is what turns a surprise on the pod into a decision on a laptop.
    """
    cap = symmetric_cap(cfg, control_block_m, block_n, alpha, b)
    if cap >= ridge:
        return (f"  The control's cap is {cap:.1f} Op/B = {cap / ridge:.2f}x "
                "the ridge, so it has no ceiling of its own to hit.")
    return (f"  The control's cap is {cap:.1f} Op/B = {cap / ridge:.2f}x the "
            f"ridge, so it is MEMORY BOUND BY CONSTRUCTION and C4 -- the "
            f"control reaches the roof -- fails before the run. The best "
            f"verdict this arm can reach is {GAP_UNLOCATED!r}.")


def check_control(cfg, control_block_m: int | None, alpha: float,
                  ridge: float, b: int, *, block_n: int, fits: list) -> str:
    """Empty when this tile may serve as a control, else why it may not.

    The ONE place a fit enters the design, and the one place the hardware does.
    A control has to be a tile the hardware can HOLD and a tile with more
    headroom than the subject; both are arithmetic on the pinned constants and
    on `symmetric_cap`, both are decided before any GPU time, and both decide
    which tile is measured -- never what the measurement means.

    WHAT STOPPED BEING A REFUSAL ON 2026-09-03, and why that is not a
    loosening. This function used to refuse a control whose cap did not clear
    the ridge by `CONTROL_CAP_MARGIN`, on a cap from `SWEEP.ai_cap` -- the
    scalar form this file argues is the wrong reading, overstating a 256x256
    tile by 3.7x. Priced correctly, NO tile a thread block can hold clears the
    ridge at all (see `binding_reachability`), so keeping the refusal would have
    turned it into a gate that can only ever fail, refusing every geometry and
    leaving the study with no controlled arm anywhere. That is the mirror image
    of the rubber stamp this project keeps finding: a check that can only pass
    examines nothing, and a check that can only fail decides nothing. The cap is
    now DISCLOSED at every candidate and pre-registers C4's failure, which is
    what a registered prediction is for. A control is refused for being
    unbuildable or for having no headroom, and never for the answer it is
    expected to give.

    `None` is `--control none`, and it is refused WHEREVER A CONTROL FITS. The
    mode exists because at BLOCK_SIZE_N=256 the register file leaves no tile to
    compare against; offered where a control does fit it would let an operator
    drop V4, C3 and C4 and receive a page saying the hardware forced it. REFUSE
    rather than default: the plan's own CONTROL SEARCH decides, and `fits` is
    that search's answer.

    BLOCK_SIZE_N IS A PARAMETER BECAUSE THE ACCUMULATOR IS TWO-DIMENSIONAL.
    Until 2026-09-03 this function asked only whether the control's cap cleared
    the ridge, the tile's runnability was left to the plan's resource bill one
    call later, and the bill it consulted was per thread -- so the answer to
    "may 256 be the control at BLOCK_SIZE_N=256" was yes at 16 warps, which is
    false at every warp count.
    """
    if control_block_m is None:
        if fits:
            return (
                f"--control none at BLOCK_SIZE_N={block_n}, where a control "
                f"DOES fit: the search names {len(fits)} pin(s), the first "
                f"being BLOCK_M={fits[0][0]} at --num-warps {fits[0][1]} "
                f"--num-stages {fits[0][2]}. Running uncontrolled here drops "
                "V4, C3 and C4 by choice and the report would carry the "
                "no-control caveat, which says the arm had no control to run -- "
                "true at BLOCK_SIZE_N=256 and false here. Pass --control "
                f"{fits[0][0]}, or change --block-n to a geometry where no "
                "control exists.")
        return ""
    if control_block_m == SUBJECT_BLOCK_M:
        return (f"--control {control_block_m} is the SUBJECT. A control has to "
                "be a different tile, or the difference C3 measures is zero by "
                "construction.")
    if control_block_m < SUBJECT_BLOCK_M:
        return (f"--control {control_block_m} is below the subject's "
                f"{SUBJECT_BLOCK_M}, so its cap is LOWER and its own ceiling is "
                "tighter. A control must have more headroom than the subject, "
                "not less.")
    cap = symmetric_cap(cfg, control_block_m, block_n, alpha, b)
    subject_cap = symmetric_cap(cfg, SUBJECT_BLOCK_M, block_n, alpha, b)
    if cap <= subject_cap:
        return (f"--control {control_block_m} at BLOCK_SIZE_N={block_n} caps at "
                f"{cap:.2f} Op/B against the subject's {subject_cap:.2f}, so it "
                "has NO headroom on the tile it is supposed to control and C3's "
                "difference is zero by the model. moe.bench.ai_model.cap is "
                "symmetric in BLOCK_M and BLOCK_SIZE_N, which is how a larger M "
                "buys nothing.")
    why = register_file_refusal(control_block_m, block_n)
    if why:
        return (f"--control {control_block_m} at BLOCK_SIZE_N={block_n} cannot "
                f"be held by one thread block: {why}. THE ARM AS DESIGNED "
                "CANNOT RUN AT THIS GEOMETRY, and no pin rescues it; the "
                "search and the alternatives are printed below.")
    return ""


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def measure_setting(args, cfg, block_m: int, rows: list[int], csv_path: Path,
                    cache_root: Path, pinned: dict, done, timings: list[Timing],
                    *, reference_clock: float | None, prov=None
                    ) -> tuple[int, int]:
    """Time one tile, `--reps` round-robin passes over its batches.

    ROUND ROBIN INSIDE THE TILE, not batch-by-batch to completion. Running the
    small batch fifty times and then the large batch fifty times puts every
    point at a different place in the pod's thermal history, and the resulting
    monotone drift IS the derivative this script measures. One pass over the
    whole grid per repeat spreads that drift across the curve instead of
    aligning it with the x axis.

    ONE INSTRUMENT, AND IT IS THE ROOF'S. `moe.bench.timing.time_kernel` warms
    for a DURATION of delivered GPU load, sizes its own iteration count from
    that warmup's queue-deep per-call time, flushes L2 before every call, and
    polls the SM clock on a background thread WHILE the kernels run. The roof
    this cell's throughput is divided by was measured by the same loop. The
    private per-iteration-synchronise loop this replaced created its events
    inside the timed region and let 0.18-0.30 ms of host enqueue time per call
    into the measured interval, which is a per-card bias in a number that is
    then compared across cards.

    `reference_clock` is the clock the CALIBRATION's dense GEMM ran at, resolved
    once for the whole arm before any cell so a mid-run yaml rewrite cannot move
    it. Without it every cell's `clock_level_ok` is None, which means "not
    determined" and excludes nothing; the caller prints that it happened.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.bench import timing
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
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=reference_clock)
                row = Timing(block_m, r, tiles, tokens, rep, t.ms_p50, t.ms_min,
                             t.ms_std, t.iters,
                             sm_clock_load_mhz=t.sm_clock_load_mhz,
                             sm_clock_start_mhz=t.sm_clock_start_mhz,
                             sm_clock_end_mhz=t.sm_clock_end_mhz,
                             # THE SAMPLE LIST AND THE SETTLE COST TRAVEL TOO,
                             # read through getattr because the instrument gains
                             # them in its own tree: a writer that referenced
                             # them directly would refuse to run against the
                             # basis that produced every committed row. Absent
                             # is written as blank, which reads back as "the
                             # instrument did not report this", never as zero.
                             clock_samples_mhz=_samples_text(t),
                             settle_ms=getattr(t, "settle_ms", None),
                             settle_calls=getattr(t, "settle_calls", None),
                             clock_level_ok=t.clock_level_ok,
                             clock_drift_ok=t.clock_drift_ok,
                             # THE SIDE TRAVELS WITH THE VERDICT: a failed
                             # LEVEL without it is refused by Timing, because
                             # read as LOW it excluded every boosted cell.
                             clock_level_side=t.clock_level_side,
                             instrument=t.instrument, warmup_ms=t.warmup_ms,
                             trials=t.trials, l2_flush=t.l2_flush)
                if t.clock_level_ok is False or t.host_bound:
                    print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
            except timing.TimingRefused:
                # THE INSTRUMENT'S OWN REFUSAL IS NOT ONE CELL'S ERROR. TimingRefused
                # subclasses RuntimeError, so the handler below would file "no CUDA",
                # "trials=0" or "warmup too short" as a failed cell and move on: the
                # arm then walks its whole grid writing zeroed rows and exits DONE.
                # Reproduced on the driver path on 2026-09-03; same door here.
                raise
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
            append_timing(csv_path, row, prov)
            load = ("    ? MHz" if row.sm_clock_load_mhz is None
                    else f"{row.sm_clock_load_mhz:5.0f} MHz")
            print(f"  BM={block_m:3d} r={r:5d} n={tiles:3d} T={tokens:7d} "
                  f"rep={rep:2d}  {row.ms_p50:9.4f} ms  {load} under load"
                  + ("  DRIFTED" if row.throttled else "")
                  + ("  BELOW THE ROOF'S CLOCK" if row.cold else "")
                  + ("  ABOVE THE ROOF'S CLOCK (kept; fixed-roof fraction "
                     "not comparable)" if row.boosted else ""))
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
                "instrument": p.instrument, "warmup_ms": p.warmup_ms,
                "iters": p.iters, "trials": p.trials, "l2_flush": p.l2_flush,
                "clock_level_ok": p.clock_level_ok,
                "clock_drift_ok": p.clock_drift_ok,
                "clock_level_side": p.clock_level_side,
                "boosted_reps": p.boosted_reps,
                "cold_reps": p.cold_reps,
                "roof_fraction_at_clock": (
                    "" if p.roof_fraction_at_clock is None
                    else f"{p.roof_fraction_at_clock:.6f}"),
            })
    return out


#: The instrument columns ride on every plotted row for the same reason the roof
#: does: a figure redrawn months later from a committed CSV must be able to say
#: which loop produced the milliseconds, and a mixed-instrument figure is the
#: defect the whole 2026-09-02 timing rebuild was about.
FIGURE_FIELDS = ["card", "roof_tflops", "roof_source", "roof_clock_mhz",
                 "block_m", "tokens", "rows_per_expert", "tiles", "regime",
                 "tile_eff", "reps", "ms_p50", "ms_spread_rel", "useful_tflops",
                 "roof_fraction", "sm_clock_mhz", "throttled_reps", "retained",
                 "excluded_why", "instrument", "warmup_ms", "iters", "trials",
                 "l2_flush", "clock_level_ok", "clock_drift_ok",
                 "clock_level_side", "boosted_reps", "cold_reps",
                 "roof_fraction_at_clock"]


def write_figure_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIGURE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def control_derivative_line(control_block_m: int | None,
                            ctl_plateau) -> str:
    """The control's row of the derivative section, or the sentence instead of it.

    A line and not a blank: a report whose derivative section silently has one
    row where every other report has two reads as a truncated file rather than
    as an arm that ran without a control.
    """
    if control_block_m is None:
        return ("  control      NONE. No control ran, so no derivative of one "
                "exists and gates C3 and C4 are not in the table below.")
    return f"  control BLOCK_M={control_block_m}: {ctl_plateau.line()}"


def analyse(timings: list[Timing], cfg, roof: Roof, *,
            control_block_m: int | None,
            b: int, sm_count: int, block_n: int, doublings: float,
            compiles: dict[int, int], executed: dict[int, int],
            planned_multi_tile: int) -> tuple[list[str], list[Gate], dict,
                                              list[Series]]:
    """Everything that is read off the timings, and nothing that is not.

    `control_block_m=None` is `--control none`, and the three control gates are
    then OMITTED rather than scored UNKNOWN. A gate that can only ever say
    UNKNOWN is a gate that examined nothing reporting no failures, which is the
    shape `moe/bench/exit_codes.py` is named against; and V4 is a VALIDITY gate,
    so an UNKNOWN there would void C1 and C2 -- the only two readings such a run
    exists to produce.

    AND ONE GATE IS ADDED IN THEIR PLACE, `gate_cu_uncontrolled`. Omitting three
    gates left the mode with six RESULT lines, all PASS, and an exit code of 0
    DONE, so the session driver -- which greps `^RESULT: ` and nothing else --
    saw the production arm as a confirmation. The word UNCONTROLLED was in the
    report body only. The CU gate is the one machine-readable line that says the
    reading is uncontrolled, and it carries the exit code with it.
    """
    clock_ref = modal_clock(timings)
    subject = build_points(timings, cfg, SUBJECT_BLOCK_M, roof,
                           sm_count=sm_count, block_n=block_n,
                           clock_ref=clock_ref)
    control = ([] if control_block_m is None else
               build_points(timings, cfg, control_block_m, roof,
                            sm_count=sm_count, block_n=block_n,
                            clock_ref=clock_ref))
    series = ([] if control_block_m is None else
              [Series(f"BLOCK_M={control_block_m} (control)", "o", control)]) \
        + [Series(f"BLOCK_M={SUBJECT_BLOCK_M} (subject)", "#", subject)]
    sub_plateau = plateau_of(multi_tile(subject), doublings=doublings)
    ctl_plateau = (None if control_block_m is None
                   else plateau_of(multi_tile(control), doublings=doublings))
    predicted = predicted_plateau_band(roof.ridge, b)
    tiles = ((SUBJECT_BLOCK_M,) if control_block_m is None
             else (SUBJECT_BLOCK_M, control_block_m))

    gates = [
        gate_v0_roof(roof),
        gate_v1_pin(compiles, executed, tiles),
        gate_v2_non_vacuity(subject,
                            None if control_block_m is None else control,
                            planned_multi_tile,
                            onset_tokens_value=onset_tokens(cfg, SUBJECT_BLOCK_M)),
        gate_v3_clocks(timings, subject + control, roof, clock_ref),
    ]
    if control_block_m is not None:
        gates.append(gate_v4_control_ran(control, subject, control_block_m))
    gates += [
        gate_c1_roof(subject, sub_plateau, roof, predicted),
        gate_c2_plateau(sub_plateau, subject),
    ]
    if control_block_m is not None:
        gates += [
            gate_c3_attribution(subject, control, ctl_plateau, control_block_m,
                                roof),
            gate_c4_ceiling_located(control, control_block_m, roof),
        ]
    else:
        gates.append(gate_cu_uncontrolled(subject))
    call, why = verdict(gates)

    lines = ["", "## The curve", ""] + ascii_plot(series, roof) + ["", ""] \
        + point_table(series) + ["", "## The derivative", "",
                                 f"  subject BLOCK_M={SUBJECT_BLOCK_M}: "
                                 f"{sub_plateau.line()}",
                                 control_derivative_line(control_block_m,
                                                         ctl_plateau),
                                 "", "## Verdict", "", f"  {call}", ""] \
        + [f"  {w}" for w in why]

    payload = {
        "verdict": call,
        "verdict_why": why,
        "roof": asdict(roof),
        "clock_ref_mhz": clock_ref,
        # THE NUMBER THE GATE PRINTS, beside the session median it replaced.
        # A consumer reading `clock_ref_mhz` alone on a two-tile arm reads a
        # median of a bimodal set; these are the operating points that exist.
        "clock_per_tile_mhz": per_tile_clocks(timings),
        "predicted_plateau_band": list(predicted),
        "predicted_band_is_scored_against":
            "nothing. It is the study's own arithmetic, pre-registered so the "
            "fit is on the record. Every verdict here comes from the stopwatch.",
        "control_block_m": control_block_m,
        "subject_plateau": asdict(sub_plateau),
        "control_plateau": None if ctl_plateau is None else asdict(ctl_plateau),
        "points": figure_rows(series, roof, roof.device),
        "gates": [asdict(g) | {"verdict": g.verdict, "name_token": g.token,
                               "result_line": g.result_line()}
                  for g in gates],
        "exit_code": exit_codes.classify(gates),
        "exit_code_meaning": exit_codes.describe(exit_codes.classify(gates)),
    }
    return lines, gates, payload, series


# --------------------------------------------------------------------------
# Self test: plant three worlds, check the gates tell them apart.
# --------------------------------------------------------------------------

def planted_timings(cfg, roof: Roof, b: int, rows_by_tile: dict[int, list[int]],
                    *, alpha: float, overhead_ms: float, reps: int,
                    noise: float, seed: int, clock_mhz: int = 1500,
                    gemm_share: float = 1.0,
                    throttle: tuple[int, int] | None = None) -> list[Timing]:
    """Timings generated FROM the study's own model, so the gates have an answer.

    `throttle` plants ONE (block_m, rows) cell whose clock DRIFTED across its
    own trials, which is how the exclusion machinery gets exercised without a
    pod that overheats on demand.

    `gemm_share` is the fraction of the fused layer's time its GEMMs get, and it
    divides BOTH tiles' milliseconds by the same number. It is the only knob
    that can plant the world the audit found missing: a layer that spends part
    of itself outside the GEMMs holds both tiles to the same fraction of the
    roof, which is a plateau below the roof that is NOT tile-attributable, and
    at a value below 1 it also keeps the control off the roof, which is a gap
    whose ceiling is unlocated. `model_ms` cannot express either, because
    `overhead_ms` is a FIXED cost and a fixed cost is outgrown: it makes a curve
    that is still rising, which is a third and different world.

    The rows carry `SWEEP.SYNTHETIC_INSTRUMENT`, never `TIMING_BASIS`. A reader
    who greps a cells.csv for the instrument has to be able to tell a pod row
    from a generated one, and a self-test that stamped the real basis on its own
    fabrications would make that impossible.
    """
    rng = random.Random(seed)
    out: list[Timing] = []
    for block_m, rows in rows_by_tile.items():
        for rep in range(1, reps + 1):
            for r in rows:
                ms = SWEEP.model_ms(cfg, r, block_m, alpha=alpha,
                                    ridge=roof.ridge,
                                    bandwidth_gbps=roof.bandwidth_gbps, b=b,
                                    overhead_ms=overhead_ms) / gemm_share
                if noise:
                    ms *= math.exp(rng.gauss(0.0, noise))
                end = float(clock_mhz)
                if throttle == (block_m, r):
                    end = clock_mhz * 0.80
                out.append(Timing(
                    block_m, r, SWEEP.tiles_per_expert(r, block_m),
                    SWEEP.tokens_for_rows(cfg, r), rep, ms, ms, ms * noise, 0,
                    sm_clock_load_mhz=float(clock_mhz),
                    sm_clock_start_mhz=float(clock_mhz), sm_clock_end_mhz=end,
                    clock_level_ok=True,
                    clock_drift_ok=abs(clock_mhz - end) / clock_mhz
                    <= THROTTLE_DRIFT_PCT / 100.0,
                    instrument=SWEEP.SYNTHETIC_INSTRUMENT, warmup_ms=0.0,
                    trials=0, l2_flush=True))
    return out


#: The worlds, their planted parameters, the gate each MUST move, and the
#: verdict each MUST reach. A gate that answers the same in every world cannot
#: settle this experiment, which is the only thing this self test is for; and a
#: VERDICT that no world reaches is a branch nobody has executed, which is how
#: the (C1 PASS, C2 PASS, C3 FAIL) path stayed unexercised until 2026-09-02.
#:
#: Each row is `(name, alpha, overhead_ms, gemm_share, gate, expected, verdict)`.
SELF_TEST_WORLDS = (
    # cap(128) = 128 Op/B against a ridge of ~163: the memory branch binds at
    # every tread and the curve is flat below the roof. cap(256) = 256 Op/B is
    # above the ridge, so the control is compute bound, reaches the roof, and
    # C4 locates the ceiling. This is the study's own hypothesis.
    ("capped      alpha 1.00", 1.00, 0.05, 1.00, "C2", True, BINDING),
    # The retracted world. cap(128) = 1280 Op/B, ten times the H200 ridge, so
    # nothing is memory bound and the subject reaches the roof.
    ("uncapped    alpha 0.10", 0.10, 0.05, 1.00, "C1", False, NOT_BINDING),
    # A fused layer with a 5 ms FIXED cost. Nothing is capped; throughput climbs
    # toward the roof and is still climbing at the deepest point, which is what
    # a derivative gate has to be able to say.
    ("overhead    alpha 0.10, D = 5 ms", 0.10, 5.00, 1.00, "C2", False,
     STILL_RISING),
    # THE (P, P, F) WORLD, planted 2026-09-02. Nothing is capped, but the fused
    # layer keeps a quarter of its own time outside the GEMMs, so BOTH tiles
    # plateau at 0.75 of the roof and the gap between them is nothing. C3 has to
    # be able to say that the shortfall is the layer's, and until this row
    # existed the branch that says it was never executed.
    ("layer-bound alpha 0.10, GEMMs get 0.75", 0.10, 0.05, 0.75, "C3", False,
     NOT_TILE),
    # THE WORLD THE PUBLISHED CORPUS DESCRIBES. Capped at 128 and the layer
    # keeps 45% of itself, so the subject sits at 0.42 and the control at 0.55:
    # a real, tile-dependent gap of 0.13, with the control nowhere near the
    # roof. C4 FAILs and the verdict is the third outcome. The published H200
    # arm reads 0.468 and 0.526 at BLOCK_SIZE_N=64, GROUP_SIZE_M=1.
    ("unlocated   alpha 1.00, GEMMs get 0.55", 1.00, 0.05, 0.55, "C4", False,
     GAP_UNLOCATED),
)

#: The verdict strings, by the name the source spells them. Printed beside the
#: sentence so `--self-test` output contains the IDENTIFIER a driver or a test
#: greps for (NOT_TILE, GAP_UNLOCATED) and not only the prose it renders as.
VERDICT_NAMES = {BINDING: "BINDING", NOT_BINDING: "NOT_BINDING",
                 STILL_RISING: "STILL_RISING", NOT_TILE: "NOT_TILE",
                 GAP_UNLOCATED: "GAP_UNLOCATED", UNCONTROLLED: "UNCONTROLLED",
                 UNSETTLED: "UNSETTLED"}


def self_test(cfg, roof: Roof, b: int, *, r_min: int, r_max: int,
              control_block_m: int | None, doublings: float, reps: int = 3,
              noise: float = 0.002, seed: int = 0
              ) -> tuple[list[str], list[Gate]]:
    """Planted worlds and the verdicts they must produce.

    The point is not that the code runs; it is that the gates DISCRIMINATE. Each
    world is named with the gate it is planted to move, and a world whose gate
    agrees with another world's is a world this test would have missed. Each is
    ALSO scored on the whole verdict it reaches, because a gate table that
    discriminates and a verdict function that collapses its outcomes onto one
    word are two different things and only the second is what gets published.

    `--control none` DOES NOT SHRINK THIS TEST. The five worlds are about
    whether the CONTROLLED gate set discriminates, and that question does not
    stop existing because one arm was asked to run without a control, so an
    uncontrolled invocation still plants them against the default control. The
    uncontrolled gate set gets its own two worlds at the end, and they run in
    every invocation for the same reason.
    """
    control_block_m = (DEFAULT_CONTROL_BLOCK_M if control_block_m is None
                       else control_block_m)
    subject_rows = doubling_rows(cfg, r_min, r_max, SUBJECT_BLOCK_M)
    control_rows = [r for r in subject_rows if r % control_block_m == 0]
    grid = {SUBJECT_BLOCK_M: subject_rows, control_block_m: control_rows}
    planned = len(subject_rows)
    planned_multi = len([r for r in subject_rows if r > SUBJECT_BLOCK_M])
    compiles = {SUBJECT_BLOCK_M: 1, control_block_m: 1}
    executed = {SUBJECT_BLOCK_M: planned, control_block_m: len(control_rows)}

    out = ["", "## Self test: planted worlds, real gates", "",
           "  Every world below runs on the HYPOTHESIS roof, so V0 fails in all "
           "of them and the real",
           "  verdict is NOT SETTLED in all of them -- which is the point of V0 "
           "and is asserted below.",
           "  The `would-be` column is `verdict()` applied to the CLAIM gates "
           "alone, i.e. what this",
           "  run would have concluded had it been measured on a calibrated "
           "card.", "",
           f"{'world':38s} {'subject peak':>12s} {'control peak':>12s} "
           f"{'gain/doubling':>14s} {'C1':>7s} {'C2':>7s} {'C3':>7s} {'C4':>7s}"
           "  would-be verdict"]
    gates: list[Gate] = []
    verdicts: set[str] = set()
    quads: set[tuple] = set()
    real_verdicts: set[str] = set()
    for name, alpha, overhead, share, moves, expect, want in SELF_TEST_WORLDS:
        timings = planted_timings(cfg, roof, b, grid, alpha=alpha,
                                  overhead_ms=overhead, reps=reps, noise=noise,
                                  seed=seed, gemm_share=share)
        _, world_gates, payload, series = analyse(
            timings, cfg, roof, control_block_m=control_block_m, b=b,
            sm_count=SWEEP.DEFAULT_SM_COUNT, block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
            doublings=doublings, compiles=compiles, executed=executed,
            planned_multi_tile=planned_multi)
        by = {g.name.split()[0]: g for g in world_gates}
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}
        subject = [s for s in series if s.marker == "#"][0]
        control = [s for s in series if s.marker == "o"][0]
        peak = max((p.roof_fraction for p in multi_tile(subject.points)),
                   default=0.0)
        peak_c = max((p.roof_fraction for p in multi_tile(control.points)),
                     default=0.0)
        plateau = payload["subject_plateau"]["gain_per_doubling"]
        # `verdict()` over the CLAIM gates alone: what this world would have
        # concluded on a calibrated card. The real verdict is collected too,
        # because "a hypothesis roof cannot produce one" is itself an invariant.
        would_be, _ = verdict([g for g in world_gates if g.kind == CLAIM])
        out.append(
            f"{name:38s} {peak:12.3f} {peak_c:12.3f} "
            + (f"{plateau:+13.2%} " if plateau is not None else f"{'n/a':>14s}")
            + f"{tag[by['C1'].passed]:>7s} {tag[by['C2'].passed]:>7s} "
              f"{tag[by['C3'].passed]:>7s} {tag[by['C4'].passed]:>7s}  "
              f"{VERDICT_NAMES[would_be]} ({would_be})")
        verdicts.add(would_be)
        real_verdicts.add(payload["verdict"])
        quads.add((by["C1"].passed, by["C2"].passed, by["C3"].passed,
                   by["C4"].passed))
        got = by[moves].passed
        gates.append(Gate(
            VALIDITY, f"S {name.split()[0]}",
            f"the planted world moves {moves} to "
            f"{'PASS' if expect else 'FAIL'} and reaches "
            f"{VERDICT_NAMES[want]}",
            f"{moves} verdict is {'PASS' if expect else 'FAIL'} and the "
            f"would-be verdict is {VERDICT_NAMES[want]}",
            got is expect and would_be == want,
            f"{moves} came out {tag[got]} and the verdict was "
            f"{VERDICT_NAMES[would_be]}",
            "the self test itself: gates that answer the same in every world "
            "cannot settle this experiment, and a verdict branch no world "
            "reaches is a branch nobody has executed"))

    # NON-VACUITY over the worlds themselves. Planted worlds that all produce
    # one reading prove nothing about the instrument, however many of them there
    # are. Scored on the claim-gate QUADRUPLE as well as on the verdict, because
    # a verdict function that collapsed several different quadruples onto one
    # word would pass a check that only looked at the word.
    gates.append(Gate(
        VALIDITY, "S discrimination",
        "the worlds do not all land on one reading",
        f"{len(SELF_TEST_WORLDS)} distinct (C1, C2, C3, C4) quadruples and "
        f"{len(SELF_TEST_WORLDS)} distinct would-be verdicts across "
        f"{len(SELF_TEST_WORLDS)} worlds",
        len(quads) == len(SELF_TEST_WORLDS)
        and len(verdicts) == len(SELF_TEST_WORLDS),
        f"{len(quads)} distinct quadruples, {len(verdicts)} distinct verdicts: "
        f"{sorted(VERDICT_NAMES[v] for v in verdicts)}",
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

    # And the clock machinery, which no world above exercises: ONE exclusion
    # and TWO kept states, planted separately because a run can have any of
    # them without the others. A clock that MOVED during the trials is the
    # exclusion; a clock that sat steadily off the band, on either side, is a
    # measurement at a known clock and is kept with its side recorded.
    timings = planted_timings(cfg, roof, b, grid, alpha=1.0, overhead_ms=0.05,
                              reps=1, noise=0.0, seed=seed,
                              throttle=(SUBJECT_BLOCK_M, r_max))
    points = build_points(timings, cfg, SUBJECT_BLOCK_M, roof,
                          sm_count=SWEEP.DEFAULT_SM_COUNT,
                          block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
                          clock_ref=modal_clock(timings))
    dropped = [p for p in points if not p.retained]
    gates.append(Gate(
        VALIDITY, "S drift exclusion",
        "a cell whose clock moved across its own trials leaves the gated set",
        "the one planted drifting point is the one excluded point",
        len(dropped) == 1 and dropped[0].rows_per_expert == r_max,
        f"{len(dropped)} excluded: "
        f"{[p.rows_per_expert for p in dropped]} rows per expert",
        "the self test itself: an exclusion path never exercised is an "
        "exclusion path that does not work"))

    # The LEVEL branch, LOW side, and since 2026-09-09 it must be KEPT. The
    # instrument sets `clock_level_ok` False with the side "low" on a cell it
    # timed below the roof's clock; here that flag is planted directly, because
    # what is being tested is that this file KEEPS it and rescales it, not that
    # `timing.clock_flags` computes it (which `tests/test_timing.py` owns). It
    # was an exclusion until that date, and on the H200 the cells it excluded
    # were every multi-tile BLOCK_M=128 cell of the real arm.
    steady_cold = planted_timings(cfg, roof, b, {SUBJECT_BLOCK_M: [r_max]},
                                  alpha=1.0, overhead_ms=0.05, reps=1,
                                  noise=0.0, seed=seed)
    cold_rows = [replace(t, clock_level_ok=False,
                         clock_level_side=SWEEP.LEVEL_LOW) for t in steady_cold]
    cold_points = build_points(cold_rows, cfg, SUBJECT_BLOCK_M, roof,
                               sm_count=SWEEP.DEFAULT_SM_COUNT,
                               block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
                               clock_ref=modal_clock(cold_rows))
    cold_kept = [p for p in cold_points if p.retained]
    cold_rescaled_ok = bool(cold_kept) and all(
        p.roof_fraction_at_clock is not None and roof.clock_mhz
        and abs(p.roof_fraction_at_clock / p.roof_fraction
                - roof.clock_mhz / cold_rows[0].sm_clock_load_mhz) < 1e-9
        for p in cold_kept)
    gates.append(Gate(
        VALIDITY, "S low side kept",
        "a cell timed steadily below the roof's clock stays in the gated set, "
        "and its fraction is rescaled to the clock it ran at",
        "the planted cold point is retained with no excluded repeat, it is "
        "counted as a LEVEL-low repeat, and its of-roof-at-own-clock is the "
        "fixed fraction over the clock ratio",
        len(cold_kept) == len(cold_points) == 1
        and cold_kept[0].throttled_reps == 0 and cold_kept[0].cold_reps == 1
        and cold_rescaled_ok,
        f"{len(cold_kept)} of {len(cold_points)} retained; LEVEL-low repeats "
        f"{[p.cold_reps for p in cold_points]}; fixed -> own-clock fraction "
        + ", ".join(f"{p.roof_fraction:.3f}->"
                    + (f"{p.roof_fraction_at_clock:.3f}"
                       if p.roof_fraction_at_clock is not None else "n/a")
                    for p in cold_points),
        "the self test itself: this path EXCLUDED until 2026-09-09, and under "
        "that rule the arm's own BLOCK_M=128 subject could not be measured on "
        "a 700 W-capped H200 at all"))

    # The LEVEL branch, HIGH side, and it must be KEPT. The same cell, planted
    # at `H200_BOOST_RATIO` times its clock with the side "high": the boosted
    # memory-shaped state that is every memory-bound cell on an H200. It stays
    # in the gated set, it is not counted as excluded, and its fraction of the
    # roof at its own clock is the fixed fraction over the ratio. Until
    # 2026-09-08 this path did not exist: `Timing.cold` was `clock_level_ok is
    # False` and this cell left the set with the cold one.
    high_rows = [replace(t, clock_level_ok=False,
                         clock_level_side=SWEEP.LEVEL_HIGH,
                         sm_clock_load_mhz=t.sm_clock_load_mhz * SWEEP.H200_BOOST_RATIO,
                         sm_clock_start_mhz=t.sm_clock_start_mhz * SWEEP.H200_BOOST_RATIO,
                         sm_clock_end_mhz=t.sm_clock_end_mhz * SWEEP.H200_BOOST_RATIO)
                 for t in steady_cold]
    high_points = build_points(high_rows, cfg, SUBJECT_BLOCK_M, roof,
                               sm_count=SWEEP.DEFAULT_SM_COUNT,
                               block_n=SWEEP.FIXED["BLOCK_SIZE_N"],
                               clock_ref=modal_clock(high_rows))
    kept = [p for p in high_points if p.retained]
    rescaled_ok = bool(kept) and all(
        p.roof_fraction_at_clock is not None and roof.clock_mhz
        and abs(p.roof_fraction_at_clock / p.roof_fraction
                - roof.clock_mhz / high_rows[0].sm_clock_load_mhz) < 1e-9
        for p in kept)
    gates.append(Gate(
        VALIDITY, "S high side kept",
        "a cell that ran above the band around the roof's clock stays in the "
        "gated set, and its fraction is rescaled to the clock it ran at",
        "the planted boosted point is retained with no excluded repeat, and "
        "its of-roof-at-own-clock is the fixed fraction over the clock ratio",
        len(kept) == len(high_points) == 1 and kept[0].throttled_reps == 0
        and kept[0].boosted_reps == 1 and rescaled_ok,
        f"{len(kept)} of {len(high_points)} retained; boosted repeats "
        f"{[p.boosted_reps for p in high_points]}; fixed -> own-clock fraction "
        + ", ".join(f"{p.roof_fraction:.3f}->"
                    + (f"{p.roof_fraction_at_clock:.3f}"
                       if p.roof_fraction_at_clock is not None else "n/a")
                    for p in high_points),
        "the self test itself: the kept path for a boosted cell was never "
        "executed off GPU, and until 2026-09-08 it did not exist, so an "
        "honest H200 session excluded its memory-shaped cells and failed V3"))

    # THE UNCONTROLLED GATE SET, planted 2026-09-03 alongside `--control none`.
    # TWO worlds and not one, because the whole claim being made about that mode
    # is an ASYMMETRY -- it can REFUTE the headline and it can never confirm it
    # -- and one world would demonstrate half of that while reading as the
    # whole. The capped world is the half that must NOT become a ceiling; the
    # uncapped world is the half that must still be able to kill the claim with
    # no control anywhere in the run. Each also asserts that C3 and C4 are
    # ABSENT rather than UNKNOWN: a gate that examined nothing and reported no
    # failure is the shape this repository keeps relearning.
    for label, planted_alpha, want in (
            ("capped", 1.00, UNCONTROLLED),
            ("uncapped", 0.10, NOT_BINDING)):
        solo = planted_timings(cfg, roof, b, {SUBJECT_BLOCK_M: subject_rows},
                               alpha=planted_alpha, overhead_ms=0.05,
                               reps=reps, noise=noise, seed=seed)
        _, solo_gates, _, _ = analyse(
            solo, cfg, roof, control_block_m=None, b=b,
            sm_count=SWEEP.DEFAULT_SM_COUNT,
            block_n=SWEEP.FIXED["BLOCK_SIZE_N"], doublings=doublings,
            compiles={SUBJECT_BLOCK_M: 1},
            executed={SUBJECT_BLOCK_M: planned},
            planned_multi_tile=planned_multi)
        solo_call, _ = verdict([g for g in solo_gates if g.kind == CLAIM])
        present = {g.name.split()[0] for g in solo_gates}
        gates.append(Gate(
            VALIDITY, f"S uncontrolled {label}",
            f"with no control, the {label} world reaches "
            f"{VERDICT_NAMES[want]} and scores no control gate",
            f"the would-be verdict is {VERDICT_NAMES[want]} and neither C3 nor "
            "C4 appears in the gate table",
            solo_call == want and not ({"C3", "C4"} & present),
            f"reached {VERDICT_NAMES[solo_call]} with gates "
            f"{sorted(present)}",
            "the self test itself: --control none is a mode that can refute "
            "this study's headline and can never confirm it, and a mode whose "
            "two directions are not both planted is a mode nobody has run"))
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

    `moe.bench.provenance.run_id` builds it, and this function's only job is to
    decide WHICH knobs are swept. That split is the point: the id format, the
    card slug, the sort-before-hash and the refusal on a None or empty value are
    one implementation for the whole repository, and the three collisions its
    docstring records -- GROUP_SIZE_M, the card, and the timing knobs -- are
    each a knob that a private id builder left out.

    This repo has lost an arm to two of them. Once to an id that omitted
    GROUP_SIZE_M: the second run derived the first's directory, found every cell
    present, skipped all of them, and printed the first run's timings under the
    second's heading. Once to an id that omitted THE CARD, and the proof is in
    the tree, where two published directories, one A100 and one H200, contain a
    report file of the same name. The card is the sharpest of them here, because
    every number this script prints is a fraction of a PER-CARD measured roof
    and `results_root()` prefers a network volume that outlives the pod.

    THE TIMING KNOBS ARE SWEPT KNOBS. `--warmup`, `--trials`, `--cell-budget-ms`
    and the L2 flush set the measured milliseconds of every cell, so a directory
    must not be resumed into across a change to any of them. There is no
    `--iters` to include: the instrument sizes the iteration count per cell from
    the warmup's own queue-deep per-call time, and a flag that named a sample
    size nothing uses would be a knob in the id and nowhere else.

    `--alpha` stays OUT: it selects the control and prints a prediction, and it
    re-reads a set of timings rather than changing one, so two analyses of one
    sweep belong in one directory. `SUBJECT_BLOCK_M` stays out because it is a
    constant and not a knob; a value that cannot vary is not part of an identity.

    THE KEYS ARE ONE OR TWO LETTERS, and that is a decision about the VISIBLE
    part rather than about the key. `run_id` truncates the readable prefix at 96
    characters and hashes the whole key regardless, so long names cost nothing in
    correctness and cost the operator the ability to read `--num-stages` off an
    `ls`. The table, once:

        m  model      d  dtype       c  --control      g  GROUP_SIZE_M
        n  BLOCK_N    k  BLOCK_K     s  num_stages     w  num_warps
        lo --r-min    hi --r-max     x  --reps         e  --seed
        u  --warmup (ms)             t  --trials       b  --cell-budget-ms
        f  L2 flushed
    """
    return PV.run_id(
        card=card, m=args.model, d=args.dtype, lo=args.r_min, hi=args.r_max,
        x=args.reps, u=args.warmup, t=args.trials, f=not args.no_l2_flush,
        b=args.cell_budget_ms, e=args.seed, g=args.group_m, n=args.block_n,
        k=args.block_k, s=args.num_stages, w=args.num_warps,
        # THE INT STAYS AN INT. `provenance._canonical` keeps a string a string
        # and an int an int, so `c=256` and `c='256'` hash differently while
        # rendering the same visible prefix: two directories for one plan,
        # differing only in the digest, with nothing anywhere saying why. That
        # is the collision shape `moe/bench/provenance.py` exists to prevent,
        # and passing `control_key()` here -- a spelling built for the CLI echo
        # -- moved the controlled arms' ids for nothing on 2026-09-03. Only the
        # ABSENCE is spelled, because `run_id` refuses a None.
        c=CONTROL_NONE if args.control is None else args.control)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def parse_control(text: str) -> int | None:
    """`"256"` -> 256, `"none"` -> None, anything else refused by argparse.

    A WORD AND NOT A FLAG. `--no-control` would be a second thing to keep in
    step with `--control`, and the two could disagree; one option with one value
    cannot. The word also survives into the run id (`c-none`), so an
    uncontrolled arm can never resume into a controlled arm's directory.
    """
    if text.strip().lower() == CONTROL_NONE:
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--control takes a BLOCK_M or the word {CONTROL_NONE!r}, not "
            f"{text!r}. There is no default for a missing control: dropping it "
            "drops three gates and the headline with them, so it is asked for "
            "by name.") from None


def control_key(control_block_m: int | None) -> str:
    """How `--control` spells this control ON A COMMAND LINE.

    THE ID DOES NOT COME THROUGH HERE, and that is a scar. `default_run_id`
    called this for its `c=` key, which turned the int 256 into the string
    "256"; `provenance._canonical` distinguishes the two, so every controlled
    arm's digest moved with nothing else about the arm changing, while the
    visible prefix stayed byte-identical. The id now passes the int and spells
    only the absence, which is the one value `run_id` refuses.
    """
    return CONTROL_NONE if control_block_m is None else str(control_block_m)


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
    ap.add_argument("--control", type=parse_control,
                    default=DEFAULT_CONTROL_BLOCK_M,
                    help="the positive control tile. Must be larger than the "
                         "subject, have MORE HEADROOM than it under "
                         "moe.bench.ai_model.cap, and fit one thread block at "
                         "this --block-n, or it is refused. It is NOT required "
                         f"to clear the ridge by {CONTROL_CAP_MARGIN:.2f}x: no "
                         "tile a block can hold does, so that check could only "
                         "ever refuse, and C4 is pre-registered to fail "
                         "instead. 256 needs 160 KiB of shared memory at 4 "
                         "stages and BLOCK_SIZE_N=64, which fits sm_90 and fits "
                         "sm_80 by 3 KiB. AT --block-n 256 NO CONTROL EXISTS at "
                         "all: a 256x256 fp32 accumulator is the entire "
                         f"per-block register file ({REGISTERS_PER_BLOCK} "
                         "registers) at every warp count. `--control none` is "
                         "the explicit opt-out THERE: it measures the subject "
                         "alone, which can REFUTE the claim and can never "
                         "confirm it, drops gates V4, C3 and C4, scores "
                         "CU_UNCONTROLLED_attribution in their place, and is "
                         "itself refused wherever a control does fit")
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
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. THE UNIT IS THE INSTRUMENT'S: a 1 "
                         "ms kernel needs hundreds of calls before the clock "
                         "governor reacts and a 30 ms one needs a handful, so a "
                         "count warms the two cells of one curve to different "
                         "clock states and the difference lands in the "
                         "derivative this script measures")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per cell; the percentiles are over "
                         "iters x trials samples, and the iteration count is "
                         "sized by the instrument from --cell-budget-ms")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the roof was measured flushed and a "
                         "warm-L2 cell is not comparable with it. Recorded per "
                         "cell and in the run id, so a flushed and an unflushed "
                         "sweep can never share a directory")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its iteration count from it")
    ap.add_argument("--plan-noise", type=float, default=DEFAULT_PLAN_NOISE_REL,
                    metavar="REL",
                    help="relative per-cell timing spread the plan's MDE line "
                         "is sized against. It sizes an ESTIMATE of what this "
                         "run could resolve and nothing else: every gate scores "
                         "the spread the run actually measured. The default is "
                         "the one published arm that recorded a spread")
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
                    help="ACCEPTED AND REDUNDANT since 2026-09-02, kept because "
                         "scripts/h200_gaps_session.sh documents it. Exit codes "
                         "now come from moe.bench.exit_codes.classify: 0 every "
                         "gate PASSED, 1 a CLAIM gate did not (a RESULT, not an "
                         "error and not a retry), 2 nothing was measured, 3 a "
                         "VALIDITY gate failed after measuring. There is no "
                         "longer a mode in which a failed gate exits 0")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    synthetic = args.dry_run or args.self_test

    # A RE-READ FRACTION OUTSIDE [0, 1] IS NOT AN ALPHA. `ai_model` refuses one
    # rather than clamping it, for the reason its own docstring gives, and the
    # refusal has to happen HERE so it costs a sentence instead of a traceback:
    # the scalar form this file moved off accepted --alpha 3.0 and returned a
    # cap of 17 Op/B without a word.
    if not 0.0 <= args.alpha <= 1.0:
        print(f"REFUSED: --alpha {args.alpha} is outside [0, 1]. It is a MISS "
              "FRACTION -- what share of a tile's weight reads miss cache -- so "
              "a value above 1 means more traffic than a full re-read and a "
              "value below 0 means negative traffic. moe.bench.ai_model.cap "
              "refuses both. Nothing measured.")
        return exit_codes.REFUSED

    try:
        roof = resolve_roof(args.dtype, synthetic=synthetic)
    except RoofUnavailable as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED

    # THE CAPABILITY FIRST, because every refusal below wants to print the
    # control search and half that search is undecidable without one.
    capability = SWEEP.resolve_capability(args, synthetic=synthetic)
    fits, search_lines = control_feasibility(
        cfg, block_n=args.block_n, block_k=args.block_k, dtype_bytes=b,
        capability=capability, alpha=args.alpha, ridge=roof.ridge)
    finding = no_control_finding(cfg, roof.ridge, args.alpha, b,
                                 block_n=args.block_n, fits=fits,
                                 capability=capability)

    refusal = check_control(cfg, args.control, args.alpha, roof.ridge, b,
                            block_n=args.block_n, fits=fits)
    if refusal:
        print(f"REFUSED: {refusal}")
        print("\n".join(search_lines))
        print("\n".join(finding))
        return exit_codes.REFUSED

    plan = build_plan(args, cfg, b, roof, capability)
    if plan.refusals:
        print("REFUSED before any GPU time, from the pinned constants alone:")
        for bm, why in sorted(plan.refusals.items()):
            print(f"  BLOCK_M={bm}: {why}")
        print("A spilled or oversized kernel still returns a time, and that "
              "time still plots. Change --num-stages, --block-n or --control.")
        print("\n".join(search_lines))
        return exit_codes.REFUSED

    lines = [f"experiment  bm128_roofline: at BLOCK_M={SUBJECT_BLOCK_M}, does "
             "achieved throughput plateau BELOW", "            this card's own "
             "measured dense bf16 rate, in the multi-tile regime?", "",
             predictions_text(cfg, roof, b, args.control, plan.subject_rows,
                              args.plateau_doublings, block_n=args.block_n)]

    detected = detect_card_slug()
    card = args.card or detected or UNKNOWN_CARD_SLUG
    if args.card and detected and args.card != detected:
        print("\n".join(lines))
        print(f"\nREFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return exit_codes.REFUSED
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "bm128_roofline" / run_id
    csv_path = out_dir / "cells.csv"
    figure_path = out_dir / "figure.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    noise_source = ("--plan-noise" if args.plan_noise != DEFAULT_PLAN_NOISE_REL
                    else "timing_spread_median of the published "
                         "alpha-surface-s4 mixtral g1/n64 arm")
    # THE SEARCH IN EVERY PLAN, not only in the refusals. An operator reading a
    # plan that HAS a control is entitled to see which other tiles could have
    # served and at what bill, and an operator reading an uncontrolled plan is
    # entitled to see the arithmetic that says none could. It is the same
    # computation either way, so it is printed either way.
    lines += ["", "## The plan", ""] + plan.lines(cfg) + [""] + search_lines \
        + ([""] + finding if args.control is None else []) \
        + roof.lines() + [
        f"card         {card}" + ("" if detected else
                                  f"  (NO DEVICE ATTACHED: the id above is the "
                                  f"{UNKNOWN_CARD_SLUG!r} one and is not what a "
                                  "pod derives; pass --card <slug> for that)"),
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per repeat, flushed, with the "
        "instrument and provenance columns), figure.csv (the",
        "             plot's data), CARD, report.txt, report.json, "
        "triton-cache/"] + mde_lines(
        reps=args.reps, noise_rel=args.plan_noise,
        noise_source=noise_source) + prior_arm_lines(
        cfg, roof, block_n=args.block_n, group_m=args.group_m,
        control_block_m=args.control)

    if args.self_test:
        more, gates = self_test(cfg, roof, b, r_min=args.r_min, r_max=args.r_max,
                                control_block_m=args.control,
                                doublings=args.plateau_doublings,
                                reps=max(2, args.reps), seed=args.seed)
        lines += more + ["", "## Gates", ""] + render_gates(gates)
        print("\n".join(lines))
        # Through the same table as a measured run, and unconditionally. The
        # self test's gates are all VALIDITY -- they are the claim that this
        # instrument discriminates -- so a failure is INVALID (3) and never 0.
        # It used to exit 0 unless --fail-on-gate was passed, which is a broken
        # self test reporting success to anything that did not know to ask.
        return exit_codes.classify(gates)

    if args.dry_run:
        lo, hi = predicted_plateau_band(roof.ridge, b)
        lines += ["", "## What is settled before the run, and what is not", "",
                  (control_ceiling_line(cfg, args.control, args.alpha,
                                        roof.ridge, b, block_n=args.block_n)
                   if args.control is not None else
                   "  THERE IS NO CONTROL IN THIS ARM. It can refute the claim "
                   "and cannot confirm it; see CONTROL SEARCH above."),
                  f"  The subject's predicted plateau spans {hi - lo:.3f} of "
                  "the roof, which is why this run measures instead of",
                  "  computing. Nothing below the onset is evidence about the "
                  "re-read term: there is one tile there.",
                  "",
                  "  Invocation for a session script:",
                  f"    python scripts/bm128_roofline.py --model {args.model} "
                  f"--dtype {args.dtype} --control "
                  f"{control_key(args.control)} \\",
                  f"        --r-min {args.r_min} --r-max {args.r_max} --reps "
                  f"{args.reps} --plateau-doublings "
                  f"{args.plateau_doublings:g} \\",
                  f"        --group-m {args.group_m} --block-n {args.block_n} "
                  f"--num-stages {args.num_stages} --num-warps {args.num_warps} "
                  f"--warmup {args.warmup:g} --trials {args.trials}",
                  "  READ THE EXIT CODE, not the prose: "
                  + exit_codes.describe(exit_codes.DONE) + "; "
                  + exit_codes.describe(exit_codes.CLAIM_FAIL) + ";",
                  "  " + exit_codes.describe(exit_codes.REFUSED) + "; "
                  + exit_codes.describe(exit_codes.INVALID) + ".",
                  "  A CLAIM_FAIL is one of this experiment's registered "
                  "outcomes and must never be queued for a retry."]
        print("\n".join(lines))
        return exit_codes.DONE

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return exit_codes.REFUSED

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        print(f"\n{missing.split('.')[0]}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --self-test  three planted worlds, checking the gates "
              "discriminate\n"
              "  --dry-run    the pod plan, the grid, the predictions and the "
              "cost")
        return exit_codes.REFUSED

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

    # THE CLOCK THE ROOF WAS MEASURED AT, resolved ONCE before any cell, so the
    # whole arm is scored against one number and a mid-sweep yaml rewrite cannot
    # move it. Without it every cell's LEVEL verdict is None, which means "not
    # determined" and excludes nothing -- which is a state this run says out
    # loud rather than one it discovers in the gates.
    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    print("\nreference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every cell's clock LEVEL "
                  "verdict will be None, no cell can be excluded for running "
                  "cold, and V3 will say so"))

    # THE PROVENANCE BLOCK, built once and written onto every cells.csv row and
    # into report.json. It NEVER raises: what it could not determine is None and
    # `missing[field]` says why, because a block that guessed would be worse
    # than one with a hole in it.
    prov = PV.provenance_block(
        repo_root=ROOT, instrument=instrument_name(), ridge=roof.ridge,
        ridge_source=roof.source, bandwidth=roof.bandwidth_gbps,
        bandwidth_source=roof.source, warmup_ms=args.warmup,
        target_ms=args.cell_budget_ms)

    # THE CONTROL FIRST. It is the cheaper half (five batches against eight) and
    # it is the half that decides whether the subject's numbers can be read at
    # all: if the control cannot be pinned or cannot run, the subject's cells
    # buy nothing, and finding that out after paying for them is the expensive
    # order.
    if args.control is None:
        print("\n-- no control tile: --control none was given, so gates V4, C3 "
              "and C4 will not be scored and the headline cannot be issued by "
              "this run --")
    else:
        print(f"\n-- control tile, BLOCK_M={args.control} --")
        c, e = measure_setting(args, cfg, args.control, plan.control_rows,
                               csv_path, cache_root, plan.pinned, done,
                               timings, reference_clock=reference_clock,
                               prov=prov)
        compiles[args.control], executed[args.control] = c, e

    print(f"\n-- subject tile, BLOCK_M={SUBJECT_BLOCK_M} --")
    c, e = measure_setting(args, cfg, SUBJECT_BLOCK_M, plan.subject_rows,
                           csv_path, cache_root, plan.pinned, done, timings,
                           reference_clock=reference_clock, prov=prov)
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
    payload["reference_clock_mhz"] = reference_clock
    payload["reference_clock_source"] = clock_source
    # `stamp` puts the block under "provenance" and the five keys the audit's
    # publish gate checks at the TOP level, and raises rather than layering a
    # second block over a first.
    payload = prov.stamp(payload)
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
    code = exit_codes.classify(gates)
    print(f"\nexit {exit_codes.describe(code)}")
    return code


def main(argv=None) -> int:
    """Convert a string SystemExit into REFUSED, and an unplanned crash to ERROR.

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
        print("ERROR: bm128_roofline crashed before it could reach a verdict. "
              "This is the apparatus failing, not a claim failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
