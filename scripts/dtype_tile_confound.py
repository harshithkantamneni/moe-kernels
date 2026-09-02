#!/usr/bin/env python3
"""How much of the fp8/bf16 crossing shift is DTYPE and how much is the TILE?

    python scripts/dtype_tile_confound.py --dry-run     # laptop, free, decides C1+C2
    python scripts/dtype_tile_confound.py --self-test 2.033 --self-test-alpha 0.2
    python scripts/dtype_tile_confound.py --self-test 2.400 --self-test-alpha 0.2
    python scripts/dtype_tile_confound.py               # the pod run
    python scripts/dtype_tile_confound.py --dtypes bf16 # a card with no fp8 units

THE OPEN ITEM THIS CLOSES. `docs/STUDY.md:99` and `docs/FINDINGS.md:259-276` both
carry the same unpaid debt: the measured fp8/bf16 ridge-crossing ratio is
**1.149 +/- 0.069** against a corrected prediction of 1.00, and vLLM resolves a
DIFFERENT tile for the two dtypes on the same shape, so the published number
varied the tile along with the format. FINDINGS says it in one line -- "1.15 is
not a pure dtype measurement, and separating them needs a run with
`BLOCK_SIZE_M` pinned equal across both dtypes, which `override_config` can do
and this study has not done". This is that run.

THE OBSTACLE WAS NEVER THE fp8 PATH; IT WAS ONE SCRIPT'S CALL CONSTRUCTION.
`scripts/block_m_crossing_sweep.py` refuses fp8 with "the fp8 call path needs a
quant config this sweep does not build", and that is true OF THAT SCRIPT: it
calls `fused_experts` from an inlined `vllm_call_kwargs(spec)`, whose
`quant_config` is hardcoded None. The path itself is built, tested and has
already produced 19,908 published rows:

    moe/quant.py                      per-expert quantisation, `q * scale`
    moe/reference/torch_ref.make_inputs   draws fp32, quantises, keeps the scales
    moe/baselines/vllm_fused_moe.quant_config
                                      builds fp8_w8a8_moe_quant_config from them
    moe/baselines/_framework_config.vllm_quant_spec
                                      per_act_token_quant=False, block_shape=None
    moe/quant.fp8_hardware_support    refuses fp8 on silicon without the units

and the rows are at
`results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/` (schema v3,
`dtype=fp8_e4m3`, `impl=vllm_fused_experts`), written by `moe/bench/driver.py`
through that span. So this script REUSES `vllm_fused_moe.quant_config` rather
than building a second, unreviewed one -- a reimplementation that passed the
scales in the reciprocal direction would run, return the right shape, and time a
different layer.

WHAT THE DERIVATION ALREADY SAYS, BEFORE ANY GPU TIME, and it corrects the docs.
`moe/bench/tile_resolve.resolve_tile` reproduces vLLM v0.27.1's lookup exactly,
so the two dtypes' configs can be compared on a laptop. FINDINGS states the
mechanism as "the fp8 arm ran on taller tiles than the bf16 arm THROUGHOUT".
That is true only at low batch:

    mixtral-8x7b   BLOCK_SIZE_M differs (16-32 vs 64) only for M <= 96;
                   from M = 128 upward both dtypes take 64 and then 128
    qwen2-57b-a14b differs (16-32 vs 64) only for M <= 192;
                   from M = 256 upward both take 64 and then 128

The published bf16 crossings are 454 (mixtral) and 810 (qwen2) tokens, and the
fp8 ones 568 and 900. EVERY one of those four batches sits in the region where
BLOCK_SIZE_M AGREES between the dtypes. What still differs there is
BLOCK_SIZE_K (64 bf16 vs 128 fp8, forced by the `dtype_selector == "fp8_w8a8"`
branch of `get_default_config` and by the tuned files) and GROUP_SIZE_M (16 vs
32 on mixtral at M = 448-768; 16 vs 1 on qwen2 above M = 896). GROUP_SIZE_M is
the swizzle knob this project has MEASURED alpha against -- 0.84 / 0.73 / 0.68 /
0.67 at G = 1 / 8 / 16 / 64 on both cards -- so it moves weight re-read traffic
directly. The confound is real; the docs name the wrong knob for it. C2 below
is that correction, and it is decided off-GPU.

The other two models in the published table take the FALLBACK LADDER in both
dtypes, and there BLOCK_SIZE_M is identical at every M by construction --
`get_default_config`'s M ladder does not read the dtype selector at all. Their
published ratios are 1.06/1.16 (deepseek-v2-lite) and 1.07/1.23 (deepseek-v3),
against 1.25/1.16 and 1.11/1.15 for the two tuned shapes. A confound that is
absent in half the cells and produces the same answer there is not the whole
explanation, whatever this run measures.

THE DESIGN: A 2 x 3 FACTORIAL, DTYPE INNERMOST.

Per (model, token count) three arms are timed IN EACH DTYPE:

    native     no override; vLLM resolves its own config per dtype.
               The CONFOUNDED arm, and the one a deployment runs.
    cfg_bf16   the config vLLM natively picks for BF16 here, FORCED on both.
    cfg_fp8    the config vLLM natively picks for FP8 here, FORCED on both.

Two matched arms rather than one, because "pin the tile" leaves open WHICH tile,
and a dtype effect that only exists under one config is not a dtype effect. Both
are reported and the gate reads the pair.

Two placebos fall out for free and cost no extra arm: `native` and `cfg_bf16`
are the SAME config in bf16 (one resolved, one forced), as are `native` and
`cfg_fp8` in fp8. Their ratio is the noise floor, and it is also the only check
that `override_config` forces what `resolve_tile` derived.

THE ESTIMAND IS NOT A CROSSING, and that is deliberate. A crossing read off
`d(log ms)/d(log T)` is a staircase reader: `all_crossings_from_points`
documents 8 of 16 canonical cells crossing 0.5 more than once, and this study's
own headline moved 59% depending on which upcrossing was taken. Worse, at a
MATCHED forced tile the two dtypes' staircases step at the SAME token counts by
construction, so a matched-tile crossing ratio would be pinned near 1.000 by the
grid rather than by the physics -- a gate that cannot fail.

So the headline is the TILT of the ratio curve instead, and it needs no crossing
and no ridge:

    r(T)  = ms(fp8_e4m3, T) / ms(bf16, T)   at MATCHED model, tokens and arm
    tilt  = median r over the LOWEST third of the token grid
            / median r over the HIGHEST third

Why that is the right generalisation of the published figure. Below the ridge
time is flat at `A` and above it linear at `C T`, so the crossing is `T0 = A/C`;
scaling the flat branch by `r_low` and the linear one by `r_high` moves it to
`T0 r_low / r_high`. Where the sweep straddles the transition the tilt IS the
crossing ratio, recovered from twenty-eight paired cells rather than from one
interpolation on a flat curve. Where it does not, the tilt is still exactly
"over these batches the fp8/bf16 ratio moved by this much", which is what moves
ANY threshold defined on those curves -- staircase included.

The branch-based `rm / rc` is computed too, from branches read off the measured
bf16 curve rather than assumed from the byte model, and printed beside every
tilt. It is often None, and that is the point: see below.

AT THE MEASURED ALPHA THE COMPUTE ROOF IS OUT OF REACH FOR THE CONFIGS vLLM
PICKS, which this script's own `--dry-run` established before any GPU time and
which shapes everything above. Arithmetic intensity is bounded at
`2 BM / (alpha b)`. At GROUP_SIZE_M = 1, where alpha measures 0.84, and
BLOCK_SIZE_M = 128, that ceiling is 152.4 FLOP/byte against a measured ridge of
162.8: the configuration vLLM's fallback ladder holds across the whole decode
range cannot be compute bound at any batch. Where the tuned file lifts the
swizzle to 16 -- mixtral above M=448 -- alpha falls to 0.68, the ceiling rises
to 188, and the compute branch IS reachable. So whether a "crossing" exists at
all is a property of the CONFIG, not of the model, and a published crossing
ratio that assumed otherwise was reading tile steps. That is the same fact that
retracted C5.

WHAT THE MODEL PREDICTS, from THIS card's own same-session calibration.
`moe/bench/hardware/measured_nvidia_h200.yaml` (measured 2026-09-02, commit
63de5b9) now carries an fp8_e4m3 ceiling next to the bf16 one -- FINDINGS calls
that "a precondition for publishing this" and it is why this run is possible
now. Read from that file at run time, not transcribed:

    bandwidth 4374.8 GB/s   bf16 712.3 TFLOP/s   fp8_e4m3 1447.7 TFLOP/s
    ridge_bf16 162.8        ridge_fp8 330.9      achieved fp8/bf16 2.033

Compute time therefore scales by 1/2.033 = 0.492, not by the datasheet's 0.500.
Weight bytes halve exactly. Activation traffic is the term that does not:
`spec.activation_dtype` keeps activations at bf16 in an fp8 cell because vLLM's
`fused_experts` asserts it and quantises them itself, so how much of that stream
is actually 8-bit inside the kernel is not something this harness sets. Both
ends are computed and the prediction is a BAND. Run `--dry-run` for the table;
over the two default models and the two matched arms it is

    matched config, activations quantised      tilt 1.000 to 1.016
    matched config, activations stay bf16      tilt 0.928 to 1.005

against a published, CONFOUNDED figure of 1.149. That gap is what C3 tests, and
the two ends of the band differ by less than the gap does.

WHAT THE MODEL CANNOT PREDICT, and why the run is necessary rather than
decorative. `predicted_ms` reads BLOCK_SIZE_M and GROUP_SIZE_M and nothing else.
The knob the two dtypes still disagree about at EVERY crossing-bracketing cell
is BLOCK_SIZE_K -- 64 for bf16, 128 for fp8, forced both by the tuned files and
by `get_default_config`'s `dtype_selector == "fp8_w8a8"` branch -- and no term in
the byte model reads it. So C4, the share of the confounded excess that the
config carries, has no predicted value at all. It can only be measured, and
`--self-test` says so rather than pretending to exercise it.

PREDICTIONS ARE REGISTERED BELOW AND PRINTED BEFORE ANY MEASUREMENT, with their
numbers, split into VALIDITY gates (a FAIL means nothing on the page may be
quoted) and CLAIM gates (a FAIL is a result, and exits 0 unless
`--fail-on-claim`). C1 and C2 are decided by `--dry-run` on a laptop; C3 and C4
need the box.

`--self-test RATIO` generates every cell from the model at a planted fp8/bf16
FLOP ratio and runs the entire analysis on it, so "C3 can tell a small dtype
effect from a large one" is checkable on a laptop. It has to be run with
`--self-test-alpha` to mean anything, and the reason is the ceiling above: at
the measured alpha the compute term never binds, so the planted FLOP ratio
changes nothing and `--self-test 2.033` and `--self-test 2.400` produce
numerically identical reports. With `--self-test-alpha 0.2` the ceiling clears
the ridge and the three worlds separate cleanly:

    --self-test 2.033 --self-test-alpha 0.2    tilt 1.023   C3 PASS
    --self-test 2.400 --self-test-alpha 0.2    tilt 1.208   C3 FAIL, "it is dtype"
    --self-test 1.000 --self-test-alpha 0.2    tilt 0.503   C3 FAIL, "no fp8 rate"

Every VALIDITY gate but V0 reads UNKNOWN in a self test, because no kernel ran
and there is nothing of this machine's to check. An earlier version filled in an
observed config and a weight dtype on the synthetic rows, and V1, V3 and V4 then
PASSED on a laptop with no GPU in the room.

WHAT IT WRITES, AND WHERE IT SURVIVES TEARDOWN. Everything lands under
`$MOE_RESULTS_DIR`, else `/workspace/results` when that exists (the RunPod
network volume, which outlives the pod), else `<repo>/results`:

    <results>/dtype_tile_confound/<run-id>/timings.csv   one row per (cell, arm, dtype)
    <results>/dtype_tile_confound/<run-id>/report.md     exactly what was printed
    <results>/dtype_tile_confound/<run-id>/summary.json  gates and headline
    <results>/dtype_tile_confound/<run-id>/plan.json     the derivation, pre-run

The absolute path is printed at the START as well as the end. Rows are flushed
per (cell, arm, dtype); the run id is a hash of the whole plan, so re-running
the same command RESUMES and a changed parameter lands in a different directory
instead of quietly overwriting the last one.

EXIT CODES. 0 every gate passed, 1 a gate FAILED, 3 nothing was measured.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench.crossing import all_crossings_from_points  # noqa: E402
from moe.bench.roofline import (  # noqa: E402
    HARDWARE_DIR,
    Hardware,
    UnverifiedHardware,
    device_matches,
    load_hardware,
)
from moe.bench.tile_resolve import (  # noqa: E402
    VLLM_TAG,
    DerivedTile,
    SnapshotMissing,
    TileNotDerivable,
    resolve_tile,
)
from moe.spec import MODEL_CONFIGS, activation_dtype, dtype_bytes  # noqa: E402

# --------------------------------------------------------------------------
# The two formats, and nothing else. Widening this is not a flag.
# --------------------------------------------------------------------------

#: The bf16 side. fp16 would do as well arithmetically and is not offered: every
#: published crossing this script is explaining was measured in bf16, and a
#: second float format would double the grid to answer a question nobody asked.
BF16 = "bf16"

#: The fp8 side. e4m3 ONLY. vLLM's w8a8 path is built on that dtype and e5m2
#: weights under an e4m3 flag would run, return the right shape, and compute a
#: different layer -- `sglang_quant_kwargs` raises on exactly this.
FP8 = "fp8_e4m3"

DTYPES: tuple[str, str] = (BF16, FP8)

#: The two shapes vLLM v0.27.1 ships a tuned file for on H200 in BOTH dtypes, so
#: they are the only ones where the tuned-file half of the confound exists at
#: all. deepseek-v2-lite and deepseek-v3 take the fallback ladder in both dtypes
#: and are runnable here (`--models`) as the negative control: their
#: BLOCK_SIZE_M is identical across dtypes at every M by construction.
DEFAULT_MODELS: tuple[str, ...] = ("mixtral-8x7b", "qwen2-57b-a14b")

#: Geometric, every entry a multiple of 32 so that `T k / E` is an exact integer
#: for every model in MODEL_CONFIGS (the largest `E / gcd(E, k)` is 32, at
#: deepseek-v2-lite and deepseek-v3). Balanced routing refuses a token count
#: that is not one, three seconds into an unattended run.
#:
#: Fourteen points spanning 256x, which is what the TILT needs: at least
#: MIN_REGION_CELLS in each third, with the transition between them. mixtral
#: crosses near T=454 on the published rows and qwen2 near T=810, so both
#: published crossings sit in the middle third and neither third is measuring
#: the same side of the transition as the other.
DEFAULT_TOKENS: tuple[int, ...] = (32, 64, 128, 256, 384, 512, 768, 1024,
                                   1536, 2048, 3072, 4096, 6144, 8192)

#: The three arms, in the order they are timed inside one repeat.
ARMS: tuple[str, ...] = ("native", "cfg_bf16", "cfg_fp8")

#: The arm that forces nothing.
NATIVE_ARM = "native"

#: The two arms that force ONE config on both dtypes. Two rather than one,
#: because "pin the tile" leaves open WHICH tile, and a dtype effect that only
#: exists under one config is not a dtype effect. They are reported separately
#: and never averaged into a single "matched" number without saying so.
MATCHED_ARMS: tuple[str, ...] = ("cfg_bf16", "cfg_fp8")

#: Which arm forces the config each dtype would have chosen anyway. `native` and
#: this arm are the same kernel in that dtype, so their ratio is a placebo AND a
#: check that override_config forced what resolve_tile derived.
PLACEBO_PARTNER: dict[str, str] = {BF16: "cfg_bf16", FP8: "cfg_fp8"}

#: Where this machine's ceilings come from. A NAME, resolved through
#: `moe.bench.roofline.load_hardware`, so both numbers carry the file they were
#: measured into rather than being typed into this script.
DEFAULT_CALIBRATION = "measured_nvidia_h200"


# --------------------------------------------------------------------------
# Numbers quoted from elsewhere. Every one of them names where it came from,
# and none of them is presented as something this script measured.
# --------------------------------------------------------------------------

#: The published fp8/bf16 CROSSING ratios, per model and kernel. QUOTED from the
#: table in docs/FINDINGS.md (section "Measured", regenerated by
#: `scripts/crossing_report.py` over the arms
#: `2026-08-28-nvidia_h200-h200-fp8-three-kernel` and `-fp8-refixed` at ridge
#: 160.3). They are the thing being explained; nothing here re-measures them,
#: and no gate is scored against them.
PUBLISHED_CROSSING_RATIO: dict[str, dict[str, float]] = {
    "mixtral-8x7b": {"vllm": 1.25, "sglang": 1.16},
    "qwen2-57b-a14b": {"vllm": 1.11, "sglang": 1.15},
    "deepseek-v2-lite": {"vllm": 1.06, "sglang": 1.16},
    "deepseek-v3": {"vllm": 1.07, "sglang": 1.23},
}

#: The pooled headline and its spread over those eight measurements, same source.
PUBLISHED_SHIFT = 1.149
PUBLISHED_SHIFT_SD = 0.069

#: The published bf16 crossing per model, in tokens, vLLM column, same source.
#: Used ONLY to say which token counts the confound has to be checked at (C2)
#: and to place the grid's midpoint in the plan printout. Never as a prediction.
PUBLISHED_BF16_CROSSING: dict[str, float] = {
    "mixtral-8x7b": 454.0, "qwen2-57b-a14b": 810.0,
    "deepseek-v2-lite": 922.0, "deepseek-v3": 3240.0,
}

#: alpha, the fraction of a weight re-read that misses L2, MEASURED against the
#: swizzle width GROUP_SIZE_M on 2026-09-01/02, monotone and saturating, and
#: reproduced on both the H200 and the A100. This is the only place the tile
#: enters the predicted time other than through the M-tile count, and it is why
#: GROUP_SIZE_M -- not BLOCK_SIZE_M -- is the knob the surviving confound runs
#: through at the crossing.
ALPHA_BY_GROUP_M: dict[int, float] = {1: 0.84, 8: 0.73, 16: 0.68, 64: 0.67}


# --------------------------------------------------------------------------
# Gate thresholds. All of them numbers, all of them fixed before the run.
# --------------------------------------------------------------------------

#: V2. A placebo pair is the SAME config timed twice, once resolved and once
#: forced. Wider than this and the box cannot resolve the ~5% effect the model
#: predicts, whatever the medians come out at. Same threshold
#: `tuned_vs_fallback.py` uses, for the same reason and on the same hardware.
PLACEBO_BAND = 0.03

#: Fewest cells a median may be taken over, and it governs both reductions: a
#: grid third for the TILT and a branch for the shift. Below it the median is one
#: or two cells wearing a costume, so `build_shift` REFUSES the tilt outright and
#: `classify_branches` declines the branch rather than reporting either thin.
MIN_REGION_CELLS = 3

#: What a branch has to look like on the MEASURED bf16 curve. Below the ridge
#: time barely moves with the batch and `d(log ms)/d(log T)` tends to 0; above it
#: time tracks work and the slope tends to 1. These are the two ends, with a wide
#: dead band between them so a stretch has to commit.
#:
#: NOT USED TO SELECT THE HEADLINE. The headline is the TILT, which is defined by
#: the grid. These decide only whether the curve has the two branches a crossing
#: ratio presupposes, and on this kernel they will often say it does not: the
#: memory branch carries `Q(n) = 1 + alpha (n - 1)` weight reads, so at the
#: measured alpha it is itself close to linear past a few M-tiles.
MEMORY_SLOPE_MAX = 0.40
COMPUTE_SLOPE_MIN = 0.60

#: V4. Percent SM clock drop across the run before the ratios stop being
#: comparable. `timing.clock_drift` calls anything past 5% throttled.
MAX_CLOCK_DRIFT_PCT = 5.0

#: V4. Per-timing relative spread. A ratio of two medians inherits both spreads,
#: and the effect under test is ~5%.
MAX_TIMING_SPREAD = 0.03

#: C2. The fraction of crossing-bracketing token counts at which BLOCK_SIZE_M
#: must AGREE between the two dtypes for the docs' stated mechanism to be wrong.
#: Derived off-GPU from vLLM's shipped configs, so this gate is decided in
#: `--dry-run` and the pod cannot change it.
BLOCK_M_AGREEMENT_MIN = 0.90

#: C2. How far either side of a published crossing counts as "at the crossing".
#: A factor of two, which covers both dtypes' crossings and the whole transition
#: between them.
CROSSING_BRACKET = 2.0

#: C3. The pure-dtype tilt, two sided, and the numbers are set from the model's
#: own band on the default grid rather than by taste. Run `--dry-run`: over the
#: two default models and the two matched arms the model predicts 0.928 to 1.016,
#: the two ends being the two answers to how much of the activation stream is
#: 8-bit inside the kernel. This window adds about six points of slack on each
#: side for the per-call fixed cost FINDINGS already measured in the fp8 path.
#:
#: The published CONFOUNDED figure is 1.149 and sits clearly outside it, which is
#: what makes this gate a test rather than a formality. A tilt above the window
#: says the format itself moves the crossing and the config was never the
#: explanation; below it says fp8 is not getting the FLOP ratio the calibration
#: measured.
PURE_DTYPE_SHIFT_LO = 0.90
PURE_DTYPE_SHIFT_HI = 1.08

#: C4. What fraction of the confounded arm's EXCESS over 1.0 the config has to
#: carry before "the tile confound explains it" is the right sentence.
CONFIG_SHARE_MIN = 0.50

#: V1. Two configs that differ compute different schedules, so their outputs
#: differ in the last bits of the K reduction; what this catches is an override
#: that changed the COMPUTATION. Same value as `tuned_vs_fallback.OUTPUT_REL_TOL`.
OUTPUT_REL_TOL = 2e-2

#: Bootstrap for the headline band. Seeded, so two readers of one CSV agree.
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_BAND = 0.90

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
#: Exit codes. A FAILED CLAIM IS NOT A FAILED RUN -- it is the result -- so only
#: a VALIDITY failure exits non-zero by default, and `--fail-on-claim` is the
#: opt-in for a caller that wants a falsified claim to stop a pipeline.
EXIT_OK, EXIT_GATE_FAILED, EXIT_NOT_MEASURED = 0, 1, 3

CSV_COLUMNS = (
    "run_id", "utc", "gpu_name", "lookup_gpu", "torch_version", "vllm_version",
    "vllm_tag", "model", "num_experts", "top_k", "num_tokens",
    "rows_per_expert", "dtype", "arm", "routing", "seed",
    "config_origin", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
    "GROUP_SIZE_M", "num_warps", "num_stages",
    "observed_config", "override_verified", "weight_torch_dtype",
    "quant_config_kind", "correctness_rel_err", "correctness_budget",
    "ms_median", "ms_mean", "ms_stdev", "ms_min", "n_samples",
    "sm_clock_start_mhz", "sm_clock_end_mhz", "error",
)

CONFIG_KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
               "num_warps", "num_stages")


# --------------------------------------------------------------------------
# Typed refusals. Every quantity this script cannot measure raises one of
# these; none of them has a fallback value, and none is caught to be replaced
# by a plausible number.
# --------------------------------------------------------------------------

class ConfoundRefusal(RuntimeError):
    """Base class, so a caller can catch every refusal without catching bugs."""


class Fp8PathUnavailable(ConfoundRefusal):
    """This machine cannot run an honest fp8 cell.

    Raised rather than degrading to bf16-only, because the bf16-only run has a
    name -- it is the tile sweep at fixed dtype -- and it must be ASKED for
    (`--dtypes bf16`) rather than arrived at by a silent fallback. `moe/quant.py`
    documents the failure this prevents: vLLM would accept an fp8 cell on Ampere,
    dequantise to bf16, and write rows labelled fp8_e4m3 that never touched an
    fp8 unit.
    """


class CalibrationIncomplete(ConfoundRefusal):
    """The ceilings this run needs are not both measured on this machine.

    An fp8 prediction against a bf16-only calibration is the exact defect
    docs/FINDINGS.md flags on the published fp8 arm: `achieved_peak_tflops = 0.0`,
    the efficiency column empty, and the headline reconstructed by hand. The
    same-session fp8 ceiling is a precondition, so its absence is a refusal.
    """


class RegimeNotResolved(ConfoundRefusal):
    """A regime holds too few cells, or the measured slope contradicts its label.

    `shift = rm / rc` is only the crossing ratio if `rm` really is the flat
    branch and `rc` really is the linear one. Two cells and a hope is not that.
    """


class UnpairableComparison(ConfoundRefusal):
    """Two things were asked to be compared that are not at matched levels.

    Averaging over a ragged grid is how this project produced confident wrong
    numbers before; saying "unpairable" is the alternative to averaging it away.
    """


# --------------------------------------------------------------------------
# The physical model. Pure arithmetic: no torch, no GPU, no CSV. It generates
# the predictions AND the --self-test cells, so nothing that reads it may be
# read as evidence FOR it.
# --------------------------------------------------------------------------

def alpha_for_group(group_m: int) -> float:
    """The measured L2 miss fraction at this swizzle width.

    Interpolated in log(GROUP_SIZE_M) between the four MEASURED points and
    clamped outside them, because the curve is monotone and saturating and an
    extrapolation past G=64 would invent a regime nothing has measured. G=32,
    which mixtral's fp8 tuned file selects across the whole crossing region, is
    between two measured points and is exactly why this is a curve and not a
    lookup.
    """
    if group_m <= 0:
        raise ValueError(f"GROUP_SIZE_M must be positive, got {group_m}")
    known = sorted(ALPHA_BY_GROUP_M)
    if group_m <= known[0]:
        return ALPHA_BY_GROUP_M[known[0]]
    if group_m >= known[-1]:
        return ALPHA_BY_GROUP_M[known[-1]]
    for lo, hi in zip(known, known[1:], strict=False):
        if lo <= group_m <= hi:
            span = math.log(hi) - math.log(lo)
            frac = (math.log(group_m) - math.log(lo)) / span
            return (ALPHA_BY_GROUP_M[lo]
                    + frac * (ALPHA_BY_GROUP_M[hi] - ALPHA_BY_GROUP_M[lo]))
    raise AssertionError(f"{group_m} fell out of a bracketed scan")  # pragma: no cover


def q_of_tiles(tiles: int, alpha: float) -> float:
    """`1 + alpha (n - 1)`: weight traffic in units of one full read."""
    return 1.0 + alpha * (tiles - 1)


def tiles_per_expert(rows: float, block_m: int) -> int:
    """vLLM pads EACH EXPERT to a multiple of BLOCK_SIZE_M in
    `moe_align_block_size`, so this is per expert and not over the whole batch."""
    return max(1, math.ceil(rows / block_m))


def rows_per_expert(cfg, num_tokens: int) -> float:
    """`T k / E`. Exact here because routing is realised balanced, not sampled."""
    return num_tokens * cfg.top_k / cfg.num_experts


def rows_step(cfg) -> int:
    """Token step that keeps rows-per-expert an exact integer.

    `R = T k / E`, so T must be a multiple of `E / gcd(E, k)`. Stated as a
    function because getting it wrong does not raise here: it produces a target
    histogram `realize_counts` refuses, halfway through a metered run.
    """
    return cfg.num_experts // math.gcd(cfg.num_experts, cfg.top_k)


def weight_bytes_per_expert(cfg, b: int) -> int:
    """up `[H, 2F]` plus down `[F, H]`, which is `3 F H` elements."""
    return 3 * cfg.intermediate_size * cfg.hidden_size * b


def activation_bytes_per_row(cfg, b_act: int) -> int:
    """x_perm, h_up, h_act, y_perm: the traffic that grows WITH the batch.

    Counted separately from the weights because it is the ONE term that does not
    halve when the weight format does. `moe/spec.py` keeps activations at bf16 in
    an fp8 cell -- vLLM's `fused_experts` asserts the hidden states are
    fp32/fp16/bf16 and quantises them itself from a run-time scale -- so how much
    of this stream is 8-bit inside the kernel is not something the harness sets.
    `predicted_shift` therefore evaluates BOTH ends and reports a band.
    """
    return (2 * cfg.hidden_size + 3 * cfg.intermediate_size) * b_act


def useful_flops(cfg, rows_total: float) -> float:
    """`6 F H` per row: up is `2 F H` MACs, down is `F H`, two flops each."""
    return 6.0 * rows_total * cfg.intermediate_size * cfg.hidden_size


@dataclass(frozen=True)
class Ceilings:
    """Bandwidth and per-dtype peak, READ from a calibration file.

    Not typed into this script. `source` and `measured_on` travel with them so a
    predicted column can never pass for one measured against another machine --
    which is the defect docs/FINDINGS.md flags on the published fp8 arm, whose
    calibration measured no fp8 ceiling at all.
    """

    name: str
    bandwidth_bytes_s: float
    peak_flops: dict[str, float]
    source: str
    measured_on: str
    path: str

    def peak(self, dtype: str) -> float:
        value = self.peak_flops.get(dtype)
        if not value:
            raise CalibrationIncomplete(
                f"{self.path} has no measured peak for {dtype!r}. Predicting an "
                f"fp8 shift against a bf16-only calibration is the defect "
                f"FINDINGS flags on the published fp8 arm. Run "
                f"`python scripts/calibrate_hardware.py` on this box first.")
        return value

    def ridge(self, dtype: str) -> float:
        return self.peak(dtype) / self.bandwidth_bytes_s

    @property
    def fp8_over_bf16(self) -> float:
        return self.peak(FP8) / self.peak(BF16)


def load_ceilings(name: str, directory: Path | None = None) -> Ceilings:
    """Read a calibration by NAME and keep its provenance attached."""
    import yaml

    path = (directory or HARDWARE_DIR) / f"{name}.yaml"
    hw: Hardware = load_hardware(name, directory=directory)
    raw = yaml.safe_load(path.read_text())
    return Ceilings(
        name=hw.name, bandwidth_bytes_s=hw.bandwidth_bytes_s,
        peak_flops=dict(hw.peak_flops), source=hw.source,
        measured_on=str(raw.get("checked_on", "unrecorded")), path=str(path))


def predicted_ms(cfg, num_tokens: int, config: dict[str, int], dtype: str, *,
                 ceilings: Ceilings, act_bytes: int, overhead_ms: float = 0.0,
                 alpha: float | None = None) -> float:
    """Predicted milliseconds: `overhead + max(traffic, padded compute)`.

    The compute side is charged on PADDED rows. A tile computes `BLOCK_SIZE_M`
    rows whether or not they are useful, so a half-empty tile costs a full one:
    time is flat along a tread and steps at the tread boundary.

    This is the model under test. It is the generator for `--self-test` and the
    source of every "predicted" column in the report, so no number it produces
    is evidence for it -- its job is to say what each world would look like, and
    the gates then ask which one arrived.

    WHAT THIS MODEL CANNOT SEE, said here because a gate depends on it. Only
    BLOCK_SIZE_M and GROUP_SIZE_M enter -- the first through the M-tile count,
    the second through the measured `alpha_for_group` curve. BLOCK_SIZE_K,
    num_warps and num_stages appear nowhere, and BLOCK_SIZE_K is exactly the knob
    the two dtypes still disagree about at every crossing-bracketing cell. So
    the model predicts almost no config effect at those cells, and C4 has no
    prediction to be scored against: the config effect has to be MEASURED, which
    is the argument for running this at all rather than deriving it.

    `alpha` overrides the swizzle curve, for `--self-test` only. At the MEASURED
    alpha the compute term never binds on this grid -- the ceiling
    `2 BM / (alpha b)` is 152 at G=1 against a ridge of 163 -- so a synthetic
    world generated at the measured alpha is insensitive to the planted fp8 FLOP
    rate, and a gate exercised only there could not tell the two worlds apart.
    """
    rows = rows_per_expert(cfg, num_tokens)
    block_m = config["BLOCK_SIZE_M"]
    tiles = tiles_per_expert(rows, block_m)
    if alpha is None:
        alpha = alpha_for_group(config["GROUP_SIZE_M"])
    weights = weight_bytes_per_expert(cfg, dtype_bytes(dtype))
    traffic = cfg.num_experts * (weights * q_of_tiles(tiles, alpha)
                                 + rows * activation_bytes_per_row(cfg, act_bytes))
    padded_rows = cfg.num_experts * tiles * block_m
    compute_s = useful_flops(cfg, padded_rows) / ceilings.peak(dtype)
    return overhead_ms + 1e3 * max(traffic / ceilings.bandwidth_bytes_s, compute_s)


#: The two ends of the activation question, as (label, bytes-per-element in the
#: fp8 cell). bf16 activations are the harness's documented behaviour; the
#: quantised end is what vLLM does internally with `a1_scale=None`, and nothing
#: in a timing can separate them, so both are predicted.
ACT_MODES: tuple[tuple[str, int], ...] = (("activations bf16", 2),
                                          ("activations quantised", 1))


# --------------------------------------------------------------------------
# The plan: which cells, which configs, derived before anything runs.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (model, token count), and the config each dtype natively resolves.

    Both configs are DERIVED by `tile_resolve` from vLLM v0.27.1's shipped tree.
    They are checked against what the run observes (gate V1), and the check is
    what makes the `native`/forced pair a placebo rather than two different
    kernels wearing the same label.
    """

    model: str
    num_tokens: int
    lookup_gpu: str
    tiles: dict[str, DerivedTile]
    configs: dict[str, dict[str, int]]

    @property
    def key(self) -> tuple[str, int]:
        return (self.model, self.num_tokens)

    @property
    def rows(self) -> float:
        return rows_per_expert(MODEL_CONFIGS[self.model], self.num_tokens)

    @property
    def block_m_agrees(self) -> bool:
        """Do the two dtypes resolve the SAME M tile here?

        The whole of gate C2. FINDINGS states the confound as the fp8 arm
        running taller tiles throughout; this is the per-cell fact that settles
        whether that is true where the crossing lives.
        """
        return (self.configs[BF16]["BLOCK_SIZE_M"]
                == self.configs[FP8]["BLOCK_SIZE_M"])

    @property
    def differing_keys(self) -> tuple[str, ...]:
        return tuple(k for k in CONFIG_KEYS
                     if self.configs[BF16][k] != self.configs[FP8][k])

    @property
    def configs_differ(self) -> bool:
        return bool(self.differing_keys)


def arm_config(cell: Cell, arm: str) -> dict[str, int] | None:
    """The config an arm forces, or None where it forces nothing.

    `native` returns None because it must go through vLLM's own resolution: an
    arm that FORCED the native config would still be a valid baseline but would
    no longer be what a deployment runs, and the confounded number this script
    is decomposing is a deployment number.
    """
    if arm == NATIVE_ARM:
        return None
    if arm == "cfg_bf16":
        return dict(cell.configs[BF16])
    if arm == "cfg_fp8":
        return dict(cell.configs[FP8])
    raise KeyError(f"unknown arm {arm!r}; known arms are {ARMS}")


def arm_is_redundant(cell: Cell, arm: str) -> bool:
    """Would this arm compile the very same kernel as `cfg_bf16`?

    True for `cfg_fp8` when the two dtypes resolved identical configs. Such an
    arm is not timed: its contribution is zero BY CONSTRUCTION, and paying a
    Triton compile for it would buy a third placebo rather than a measurement.
    Recorded rather than skipped silently, because "the configs agreed here" is
    the finding at every cell above the crossing.
    """
    return arm == "cfg_fp8" and not cell.configs_differ


def tile_config(tile: DerivedTile) -> dict[str, int]:
    return {"BLOCK_SIZE_M": tile.block_m_derived,
            "BLOCK_SIZE_N": tile.block_n_derived,
            "BLOCK_SIZE_K": tile.block_k_derived,
            "GROUP_SIZE_M": tile.group_m_derived,
            "num_warps": tile.num_warps_derived,
            "num_stages": tile.num_stages_derived}


def format_config(cfg: dict[str, int] | None) -> str:
    """One compact, always-FULL config string. Never a partial one.

    Reporting `BLOCK_SIZE_M=64` alone is what made this confound invisible for
    four days, so there is no helper here that prints a subset.
    """
    if not cfg:
        return "(resolved by vLLM)"
    return (f"M{cfg['BLOCK_SIZE_M']:<4}N{cfg['BLOCK_SIZE_N']:<4}"
            f"K{cfg['BLOCK_SIZE_K']:<4}G{cfg['GROUP_SIZE_M']:<3}"
            f"w{cfg['num_warps']} s{cfg['num_stages']}")


def plan_cells(models: list[str], tokens: list[int], lookup_gpu: str
               ) -> tuple[list[Cell], list[str]]:
    """Every cell that can be planned, plus a note per model that cannot.

    A model is dropped with its reason rather than skipped silently. The two
    reasons that matter are `SnapshotMissing` -- a tuned file ships upstream and
    this repo has not vendored it, so resolving it as the ladder would report a
    tile the run did not use -- and a token count that balanced routing cannot
    realise, which would otherwise fail mid-run.
    """
    cells: list[Cell] = []
    notes: list[str] = []
    for model in models:
        if model not in MODEL_CONFIGS:
            notes.append(f"{model}: not in MODEL_CONFIGS, skipped")
            continue
        cfg = MODEL_CONFIGS[model]
        experts, _, intermediate = cfg.w2_shape
        step = rows_step(cfg)
        illegal = [t for t in tokens if t % step]
        if illegal:
            notes.append(
                f"{model}: E={cfg.num_experts} k={cfg.top_k} needs token counts "
                f"that are multiples of {step} for an exact balanced histogram; "
                f"{illegal} are not, and are dropped from this model only")
        for tok in sorted(t for t in tokens if t % step == 0):
            try:
                tiles = {d: resolve_tile(experts, intermediate, d, lookup_gpu, tok)
                         for d in DTYPES}
            except (TileNotDerivable, SnapshotMissing) as exc:
                notes.append(f"{model} T={tok}: {exc}")
                continue
            cells.append(Cell(model=model, num_tokens=tok, lookup_gpu=lookup_gpu,
                              tiles=tiles,
                              configs={d: tile_config(t) for d, t in tiles.items()}))
    return cells, notes


def crossing_bracket_cells(cells: list[Cell], bracket: float = CROSSING_BRACKET
                           ) -> list[Cell]:
    """Cells within a factor of `bracket` of the model's published bf16 crossing.

    C2 is a statement about the tile AT THE CROSSING, not over the whole grid.
    Away from it BLOCK_SIZE_M does differ across the dtypes, and pooling the
    decode cells in would answer a different question -- the one the docs already
    answered correctly.
    """
    out = []
    for cell in cells:
        centre = PUBLISHED_BF16_CROSSING.get(cell.model)
        if centre is None:
            continue
        if centre / bracket <= cell.num_tokens <= centre * bracket:
            out.append(cell)
    return out


# --------------------------------------------------------------------------
# Regimes, read off the MEASURED curve rather than assumed from the model.
#
# THE FIRST DRAFT OF THIS SCRIPT CLASSIFIED CELLS BY `AI = 2R/b` AGAINST THE
# MEASURED RIDGE, AND ITS OWN DRY RUN REFUTED THAT. Arithmetic intensity is
# BOUNDED at `2 BM / (alpha b)`, and at the swizzle vLLM actually picks --
# GROUP_SIZE_M = 1, where alpha measures 0.84 -- BLOCK_SIZE_M = 128 caps at
# 152.4 FLOP/byte against a ridge of 162.8. That configuration cannot be
# compute bound at any batch, so cells the byte model labelled "compute" were
# still on the memory branch, `rc` was a second `rm`, and the predicted shift
# came out at a meaningless 1.000. Assuming the label is how a gate stops being
# able to fail.
#
# So the label comes from the data. A branch is a stretch of the measured bf16
# curve whose pooled `d(log ms)/d(log T)` is flat (memory) or tracks work
# (compute), and an arm whose grid never reaches a compute branch is REFUSED a
# shift rather than given one -- which is itself the tile-corrected roofline's
# prediction and not a failure of the run.
# --------------------------------------------------------------------------

def log_log_slope(points: list[tuple[float, float]]) -> float | None:
    """Pooled `d(log ms)/d(log T)` by least squares, or None under two points.

    POOLED over the stretch rather than taken per adjacent pair. The curve is a
    staircase -- `all_crossings_from_points` documents the slope spiking above
    0.5 at every tile step and sagging on every tread -- so one interval's slope
    says where the grid happened to sample, while a fit across a stretch says
    which branch the stretch is on.
    """
    usable = [(t, ms) for t, ms in points if t > 0 and ms > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(t) for t, _ in usable]
    ys = [math.log(ms) for _, ms in usable]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx


def classify_branches(series: list[tuple[float, float]]
                      ) -> tuple[list[float], list[float]]:
    """(memory tokens, compute tokens) read off one measured `(T, ms)` curve.

    A SPLIT SEARCH, not a greedy prefix. Every `(k1, k2)` with `k1 <= k2` is
    tried: cells below `k1` are the candidate memory branch, cells from `k2` up
    are the candidate compute branch, and the cells between are the transition,
    excluded rather than assigned. A pair qualifies when the POOLED slope of the
    low stretch is at most `MEMORY_SLOPE_MAX` and the pooled slope of the high
    stretch is at least `COMPUTE_SLOPE_MIN`, with `MIN_REGION_CELLS` on each
    side. Among the qualifying pairs the one using the most cells wins, ties
    broken by the widest slope separation.

    A GREEDY LONGEST PREFIX WAS THE FIRST VERSION AND ITS OWN DRY RUN KILLED IT:
    it swallowed cells that were plainly on the compute branch, because pooling
    twelve very flat cells with two steep ones still averages under the
    threshold. The split search cannot do that -- adding a steep cell to the low
    stretch raises its pooled slope, which is the test.

    EMPTY IS AN ANSWER AND IT IS THE LIKELY ONE HERE. On the memory branch
    traffic is `Q(n) = 1 + alpha (n - 1)` full weight reads, so for `n` well past
    `1/alpha` the memory branch is itself nearly LINEAR in the batch: at the
    measured alpha of 0.68 to 0.84 it reaches slope 0.8 by four M-tiles. A
    flat-versus-steep test therefore cannot separate the two branches in this
    regime at all, which is the same fact that retracted C5 and the same fact
    `all_crossings_from_points` documents as a staircase. That is why the TILT
    and not this is what every gate reads; this exists to say out loud when a
    curve does not have the two branches a crossing ratio presupposes.
    """
    points = sorted((t, ms) for t, ms in series if t > 0 and ms > 0)
    n = len(points)
    best: tuple[int, float, list[float], list[float]] | None = None
    for k1 in range(MIN_REGION_CELLS, n - MIN_REGION_CELLS + 1):
        low = points[:k1]
        low_slope = log_log_slope(low)
        if low_slope is None or low_slope > MEMORY_SLOPE_MAX:
            continue
        for k2 in range(k1, n - MIN_REGION_CELLS + 1):
            high = points[k2:]
            high_slope = log_log_slope(high)
            if high_slope is None or high_slope < COMPUTE_SLOPE_MIN:
                continue
            score = (len(low) + len(high), high_slope - low_slope)
            if best is None or score > (best[0], best[1]):
                best = (score[0], score[1], [t for t, _ in low],
                        [t for t, _ in high])
    if best is None:
        return [], []
    return best[2], best[3]


def grid_thirds(tokens: list[float]) -> tuple[list[float], list[float]]:
    """The lowest and highest thirds of a token grid, by position not by value.

    The TILT is read across these, and they are defined by the grid rather than
    by any model, which is the whole point: the tilt is computable in every run,
    on the same cells for every arm, and it cannot be steered by an assumption
    about where the ridge is. Where the sweep does straddle the transition the
    tilt IS the crossing ratio; where it does not, the tilt is still the honest
    statement "over these batches the ratio moved by this much", and the
    branch-based shift beside it is the one that gets refused.
    """
    ordered = sorted(tokens)
    if len(ordered) < 2 * MIN_REGION_CELLS:
        return [], []
    size = max(MIN_REGION_CELLS, len(ordered) // 3)
    return ordered[:size], ordered[-size:]


# --------------------------------------------------------------------------
# Persistence. Append-only, flushed per row, re-read on resume.
# --------------------------------------------------------------------------

def default_out_dir() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    The same order `scripts/run_all.sh` resolves it in. A pod's container disk
    dies with the pod and the network volume does not, so a results path that
    defaults to the checkout is a results path that defaults to being lost.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    if Path(os.environ.get("WORKSPACE", "/workspace")).is_dir():
        return Path(os.environ.get("WORKSPACE", "/workspace")) / "results"
    return Path(__file__).resolve().parents[1] / "results"


def gitignore_note(path: Path) -> str:
    """Whether the output path is inside the repo and whether git will keep it.

    THE RULE THAT HAS ALREADY EATEN OUTPUT HERE is `results/*` with only
    `!results/published/` excepted, and an unanchored `plots/` that silently
    swallowed every published figure. So this asks `git check-ignore` rather than
    reasoning about the patterns, and prints the answer next to the path at the
    START of the run. IGNORED is the correct and expected state for a raw run --
    `scripts/publish_results.sh` is what promotes a kept result into
    `results/published/` -- and the point is that it is stated rather than
    discovered later by someone whose `git add` did nothing.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(repo)
    except ValueError:
        return "outside the repo, so git has no opinion about it"
    try:
        done = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=repo, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); status unknown"
    if done.returncode == 0:
        return ("inside the repo and GIT-IGNORED, which is correct for a raw "
                "run: scripts/publish_results.sh promotes a kept result into "
                "results/published/")
    if done.returncode == 1:
        return "inside the repo and NOT ignored, so `git status` will show it"
    return f"git check-ignore exited {done.returncode}; status unknown"


def plan_run_id(payload: dict) -> str:
    """A hash of the WHOLE plan, so re-running the same command resumes.

    EVERY swept parameter is in this key, and that is not a style point. The
    sibling sweep shipped a run id that omitted GROUP_SIZE_M: a second setting
    derived the same id, resumed the first's directory, found every cell already
    on disk, skipped all of them, and printed the first's timings under the
    second's heading. Nothing looked wrong. So the arms, the dtypes, the token
    grid, the calibration name and the lookup device are all in here, and the
    visible part of the name carries the models and the token count so two runs
    are distinguishable in `ls` and not only by a hash nobody can invert.
    """
    key = json.dumps(payload, sort_keys=True)
    models = "+".join(payload["models"])[:40]
    return (f"{models}-{len(payload['tokens'])}t-"
            f"{'.'.join(payload['dtypes'])}-"
            f"{hashlib.sha256(key.encode()).hexdigest()[:10]}")


@dataclass
class ArmResult:
    """One (cell, arm, dtype) timing, or the reason there is no timing.

    `error` non-empty means no number was produced -- a Triton compile that
    overran shared memory is the expected case, since one dtype's config forced
    on the other dtype can name a shape vLLM would never emit for it. Such an arm
    is excluded from every median and named in the report, never silently treated
    as equal to its partner.
    """

    model: str
    num_tokens: int
    arm: str
    dtype: str
    config: dict[str, int] | None = None
    config_origin: str = ""
    redundant: bool = False
    ms_median: float | None = None
    ms_mean: float | None = None
    ms_stdev: float | None = None
    ms_min: float | None = None
    n_samples: int = 0
    observed_config: dict | None = None
    override_verified: bool | None = None
    weight_torch_dtype: str = ""
    quant_config_kind: str = ""
    correctness_rel_err: float | None = None
    correctness_budget: float | None = None
    sm_clock_start_mhz: int = 0
    sm_clock_end_mhz: int = 0
    error: str = ""

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.model, self.num_tokens, self.arm, self.dtype)

    @property
    def spread(self) -> float | None:
        if not self.ms_median or self.ms_stdev is None:
            return None
        return self.ms_stdev / self.ms_median


def summarise_samples(result: ArmResult, samples: list[float]) -> ArmResult:
    result.ms_median = statistics.median(samples)
    result.ms_mean = statistics.fmean(samples)
    result.ms_stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    result.ms_min = min(samples)
    result.n_samples = len(samples)
    return result


class Store:
    """Append-only CSV, flushed per arm, re-read on resume.

    Per arm and not per cell because the unit of loss on a killed pod should be
    the smallest thing that took real time, and here that is one arm's compile
    plus its repeats.
    """

    def __init__(self, path: Path, fresh: bool = False):
        self.path = path
        self.done: dict[tuple[str, int, str, str], dict] = {}
        if fresh and path.exists():
            path.unlink()
        if path.exists():
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        key = (row["model"], int(row["num_tokens"]), row["arm"],
                               row["dtype"])
                    except (KeyError, ValueError):
                        continue
                    self.done[key] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(CSV_COLUMNS),
                                      extrasaction="ignore")
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def restore(self, key: tuple[str, int, str, str]) -> ArmResult | None:
        row = self.done.get(key)
        if row is None:
            return None

        def num(name, cast=float):
            try:
                return cast(row.get(name, ""))
            except (TypeError, ValueError):
                return None

        cfg = {k: int(row[k]) for k in CONFIG_KEYS if row.get(k) not in (None, "")}
        return ArmResult(
            model=row["model"], num_tokens=int(row["num_tokens"]),
            arm=row["arm"], dtype=row["dtype"], config=cfg or None,
            config_origin=row.get("config_origin", ""),
            # str() because a row read back from CSV holds "1" where a row
            # written this session holds the int 1, and both mean the same thing.
            redundant=str(row.get("config_origin")) == "redundant",
            ms_median=num("ms_median"), ms_mean=num("ms_mean"),
            ms_stdev=num("ms_stdev"), ms_min=num("ms_min"),
            n_samples=num("n_samples", int) or 0,
            observed_config=json.loads(row["observed_config"])
            if row.get("observed_config") else None,
            override_verified=None if str(row.get("override_verified", "")) == ""
            else str(row["override_verified"]) == "1",
            weight_torch_dtype=row.get("weight_torch_dtype", ""),
            quant_config_kind=row.get("quant_config_kind", ""),
            correctness_rel_err=num("correctness_rel_err"),
            correctness_budget=num("correctness_budget"),
            sm_clock_start_mhz=num("sm_clock_start_mhz", int) or 0,
            sm_clock_end_mhz=num("sm_clock_end_mhz", int) or 0,
            error=row.get("error", ""))

    def write(self, result: ArmResult, cell: Cell, meta: dict) -> None:
        cfg = result.config or {}
        model = MODEL_CONFIGS[result.model]
        row = {
            "run_id": meta["run_id"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_name": meta["gpu_name"], "lookup_gpu": cell.lookup_gpu,
            "torch_version": meta["torch_version"],
            "vllm_version": meta["vllm_version"], "vllm_tag": VLLM_TAG,
            "model": result.model, "num_experts": model.num_experts,
            "top_k": model.top_k, "num_tokens": result.num_tokens,
            "rows_per_expert": f"{cell.rows:.4f}", "dtype": result.dtype,
            "arm": result.arm, "routing": meta["routing"], "seed": meta["seed"],
            "config_origin": result.config_origin,
            "observed_config": json.dumps(result.observed_config, sort_keys=True)
            if result.observed_config else "",
            "override_verified": "" if result.override_verified is None
            else int(result.override_verified),
            "weight_torch_dtype": result.weight_torch_dtype,
            "quant_config_kind": result.quant_config_kind,
            "correctness_rel_err": "" if result.correctness_rel_err is None
            else f"{result.correctness_rel_err:.4e}",
            "correctness_budget": "" if result.correctness_budget is None
            else f"{result.correctness_budget:.4e}",
            "ms_median": "" if result.ms_median is None else f"{result.ms_median:.6f}",
            "ms_mean": "" if result.ms_mean is None else f"{result.ms_mean:.6f}",
            "ms_stdev": "" if result.ms_stdev is None else f"{result.ms_stdev:.6f}",
            "ms_min": "" if result.ms_min is None else f"{result.ms_min:.6f}",
            "n_samples": result.n_samples,
            "sm_clock_start_mhz": result.sm_clock_start_mhz,
            "sm_clock_end_mhz": result.sm_clock_end_mhz,
            "error": result.error,
        }
        row.update({k: cfg.get(k, "") for k in CONFIG_KEYS})
        self._writer.writerow(row)
        self._fh.flush()
        # The row itself, not a placeholder: `restore` reads this dict, and a
        # placeholder would make a same-session restore raise.
        self.done[result.key] = row

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def find_vllm_hooks():
    """vLLM's `override_config`, and `get_config` beside it if it is there.

    Probed across every module path the import has lived at, exactly as
    `scripts/tile_sweep.find_override` probes: a wrong guess forces nothing,
    every arm quietly becomes the native one, and the script reports a
    beautifully tight 1.000 that means the experiment did not happen. V1 is the
    gate against that and this is its first line.
    """
    import importlib

    from moe.baselines._framework_config import VLLM_CONFIG_MODULES

    for name in VLLM_CONFIG_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(module, "override_config", None)
        if fn is not None:
            return fn, getattr(module, "get_config", None), name
    raise SystemExit(
        "vLLM is importable but exposes no override_config in any of "
        f"{VLLM_CONFIG_MODULES}. Without it every arm would run the native "
        "config and the whole comparison would be vacuous.")


def preflight_fp8(dtypes: list[str]) -> str:
    """Refuse an fp8 run this machine cannot make honestly. Returns a note.

    Three separate things have to hold, and each has its own silent failure:

      the SILICON has fp8 tensor cores. Without them vLLM accepts the cell,
        most likely dequantises to bf16, and writes rows labelled fp8_e4m3 that
        never touched an fp8 unit -- indistinguishable, once merged, from real
        ones. `moe/quant.fp8_hardware_support` is the existing check.
      TORCH has the dtype. An older build has no `float8_e4m3fn` and
        `quantize_per_expert` would raise mid-run instead of before it.
      vLLM has the quant-config builder. `fp8_w8a8_moe_quant_config` is imported
        lazily inside the span, so its absence surfaces on the first fp8 cell of
        a metered session rather than here.
    """
    if FP8 not in dtypes:
        return "fp8 not requested; nothing to preflight"
    from moe.quant import fp8_hardware_support, torch_fp8_dtype

    verdict = fp8_hardware_support()
    if verdict is None:
        raise Fp8PathUnavailable(
            "no CUDA device, so fp8 support cannot be established. --dry-run "
            "prints the plan and --self-test runs the analysis off GPU.")
    if not verdict.supported:
        raise Fp8PathUnavailable(verdict.reason)
    try:
        resolved = torch_fp8_dtype(FP8)
    except ValueError as exc:
        raise Fp8PathUnavailable(f"torch cannot represent {FP8}: {exc}") from exc
    try:
        from vllm.model_executor.layers.fused_moe.config import (  # noqa: F401
            fp8_w8a8_moe_quant_config,
        )
    except ImportError as exc:
        raise Fp8PathUnavailable(
            f"vLLM exposes no fp8_w8a8_moe_quant_config ({exc}), so an fp8 cell "
            "would have to be called with quant_config=None, which resolves to "
            "FUSED_MOE_UNQUANTIZED_CONFIG and computes a different layer.") from exc
    return (f"fp8 preflight OK: silicon has fp8 tensor cores, torch has "
            f"{resolved}, vLLM has fp8_w8a8_moe_quant_config")


def build_model_inputs(model: str, dtypes: list[str], seed: int, device: str):
    """Weights per dtype and one shared router gate, held resident together.

    WHY NOT `make_inputs` PER CALL. Its weight cache holds ONE entry and evicts
    on every miss -- deliberately, since expert weights are the largest
    allocation in a cell -- so alternating bf16 and fp8 through it would redraw
    and requantise mixtral's 3.8 GB fp32 draw on every alternation. Both dtypes
    are built once here and kept, which is what lets the dtype be the INNERMOST
    loop and therefore lets a ratio be immune to whatever the clocks did.

    THE ROUTER GATE IS SHARED ACROSS DTYPES, and this is not a micro-optimisation.
    `wg` is drawn from the same generator as the expert weights, so the two
    dtypes would otherwise route on different logits, give different combine
    weights, and stop being the same cell. Sharing it makes routing, gate scores
    and the combine literally identical on both sides. The expert weights
    themselves still differ -- an fp8 draw is not a quantisation of the bf16 one
    -- which is irrelevant to time and is stated so nobody looks for it later.
    """
    from dataclasses import replace as dc_replace

    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    cfg = MODEL_CONFIGS[model]
    out: dict[str, tuple] = {}
    shared_wg = None
    for dtype in dtypes:
        spec = BenchSpec(cfg, num_tokens=1, dtype=dtype,
                         routing=RoutingSpec("uniform", 0.0), seed=seed)
        # reuse_weights=False so nothing is written into the module cache that a
        # later model would evict, and so the two dtypes coexist.
        _, weights = make_inputs(spec, device=device, reuse_weights=False)
        if shared_wg is None:
            shared_wg = weights.wg
        else:
            weights = dc_replace(weights, wg=shared_wg)
        out[dtype] = weights
    return out


def balanced_ids(cfg, tokens: int, device: str):
    """Top-k ids whose per-expert histogram is EXACTLY `T k / E`.

    Not sampled uniform. Sampled routing puts about 15 rows of spread on a mean
    of 256 at mixtral T=1024, and every ratio here is between two dtypes at ONE
    token count, so a sampled histogram would have to be redrawn identically for
    both or the two sides would not be the same cell. Realising the histogram
    exactly makes them the same cell by construction, and makes rows-per-expert
    an integer the plan knew before the pod was rented.
    """
    from moe.routing.distributions import realize_counts

    per = tokens * cfg.top_k // cfg.num_experts
    if per * cfg.num_experts != tokens * cfg.top_k:
        raise ValueError(f"T={tokens} does not divide evenly over "
                         f"E={cfg.num_experts} at k={cfg.top_k}")
    return realize_counts([per] * cfg.num_experts, tokens, cfg.top_k, device=device)


def time_calls(fn, warmup: int, iters: int) -> list[float]:
    """Per-iteration milliseconds from CUDA events. No L2 flush.

    No flush on purpose. The arms differ only in the schedule and the format and
    run on identical routing, and `L2Flusher`'s own docstring records the H200
    hazard that a flush can make a microsecond kernel FASTER by holding clocks
    up -- which is precisely the small-batch regime the memory-bound half of this
    comparison lives in.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def _make_call(fused_experts, x, weights, topk_w, ids, kwargs):
    """Bind explicitly rather than closing over loop variables (ruff B023)."""
    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=topk_w, topk_ids=ids, **kwargs)
    return call


def measure_cell(cell: Cell, weights_by_dtype: dict, dtypes: list[str], args,
                 store: Store, meta: dict, hooks, check_correctness: bool
                 ) -> dict[tuple[str, str], ArmResult]:
    """Time every (arm, dtype) of one cell, round-robin, dtype innermost.

    ROUND-ROBIN WITH DTYPE INNERMOST IS THE WHOLE POINT OF THE LOOP ORDER. Every
    number this script reports is a ratio between two dtypes at one token count.
    Running all of bf16 and then all of fp8 would put whatever the clocks did in
    between into that ratio; interleaving at the innermost level means a thermal
    excursion lands on both sides of every ratio, and the placebo pairs measure
    whatever is left.
    """
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import (
        TileCapture,
        recording_tile_config,
        vllm_call_kwargs,
    )

    # The TESTED builder, not a second one written here. A reimplementation that
    # passed the scales in the reciprocal direction would run, return the right
    # shape, and time a different layer -- which is the failure `quant_config`'s
    # own docstring names. This module imports vLLM at module scope, so it is
    # imported here rather than at the top of the file.
    from moe.baselines.vllm_fused_moe import quant_config as build_quant_config
    from moe.bench.timing import ClockState
    from moe.bench.tolerance import relative_error, tolerance
    from moe.reference.torch_ref import golden_forward, weights_for_forced_ids
    from moe.spec import BenchSpec, RoutingSpec

    override_config, get_config, _ = hooks
    cfg = MODEL_CONFIGS[cell.model]

    # ONE x for both dtypes. `activation_dtype` is bf16 for bf16 AND for
    # fp8_e4m3 -- vLLM asserts the hidden states are float and quantises them
    # itself -- so the two dtypes can and should be handed the identical tensor.
    # Asserted rather than assumed: if a future format changed this, the two
    # sides would silently be fed different data.
    for dtype in dtypes:
        if activation_dtype(dtype) != BF16:
            raise UnpairableComparison(
                f"{dtype} draws activations as {activation_dtype(dtype)}, not "
                f"{BF16}, so the two dtypes cannot share one input tensor and "
                "every ratio here would carry a data difference as well as a "
                "format one")
    # Drawn in fp32 and cast, exactly as `make_inputs` does. `normal_` on a
    # low-precision tensor is a different draw, and an x that differed between
    # this script and the rest of the harness would make a time here
    # incomparable with a published row for no reason worth having.
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + cell.num_tokens)
    x = (torch.empty((cell.num_tokens, cfg.hidden_size), device="cuda",
                     dtype=torch.float32)
         .normal_(0.0, 1.0, generator=generator).to(torch.bfloat16))
    ids = balanced_ids(cfg, cell.num_tokens, "cuda")
    # The router's OWN combine weights, not a uniform 1/k. They cost nothing at
    # run time and they make `golden_forward(..., forced_topk_ids=ids)` an exact
    # oracle for this call, which is what turns V0 from an assertion about dtypes
    # into a check on the arithmetic.
    logits = x.float() @ weights_by_dtype[dtypes[0]].wg.float().t()
    topk_w = weights_for_forced_ids(logits, ids, cfg)

    specs, calls, quant_kinds = {}, {}, {}
    for dtype in dtypes:
        spec = BenchSpec(cfg, num_tokens=cell.num_tokens, dtype=dtype,
                         routing=RoutingSpec(args.routing, 0.0), seed=args.seed)
        weights = weights_by_dtype[dtype]
        kwargs = vllm_call_kwargs(spec)
        kwargs["activation"] = MoEActivation(kwargs["activation"])
        # The tested builder, not a second one written here. A reimplementation
        # that passed the scales in the reciprocal direction would run, return
        # the right shape, and time a different layer.
        quant = build_quant_config(spec, weights)
        kwargs["quant_config"] = quant
        quant_kinds[dtype] = type(quant).__name__ if quant is not None else "none"
        specs[dtype] = spec
        calls[dtype] = _make_call(fused_experts, x, weights, topk_w, ids, kwargs)

    results: dict[tuple[str, str], ArmResult] = {}
    pending: list[tuple[str, str]] = []
    for arm in ARMS:
        for dtype in dtypes:
            key = (cell.model, cell.num_tokens, arm, dtype)
            restored = None if args.fresh else store.restore(key)
            if restored is not None:
                results[(arm, dtype)] = restored
                continue
            if arm_is_redundant(cell, arm):
                # Its config IS cfg_bf16's, so its contribution is zero by
                # construction. Recorded, not timed: a Triton compile here would
                # buy a third placebo rather than a measurement.
                result = ArmResult(cell.model, cell.num_tokens, arm, dtype,
                                   config=arm_config(cell, arm),
                                   config_origin="redundant", redundant=True)
                results[(arm, dtype)] = result
                store.write(result, cell, meta)
                continue
            pending.append((arm, dtype))

    def context(arm: str):
        forced = arm_config(cell, arm)
        return contextlib.nullcontext() if forced is None else override_config(forced)

    # Untimed prologue: what config did vLLM really use, what dtype did the
    # kernel really see, and did it compute the right layer. All three are about
    # the FIRST call, and the recorder deep-copies a dict per call, so none of it
    # can land inside a timed region.
    for arm, dtype in pending:
        forced = arm_config(cell, arm)
        result = ArmResult(cell.model, cell.num_tokens, arm, dtype,
                           config=forced,
                           config_origin="forced" if forced else "observed",
                           weight_torch_dtype=str(weights_by_dtype[dtype].w1.dtype),
                           quant_config_kind=quant_kinds[dtype])
        capture = TileCapture()
        try:
            with context(arm), recording_tile_config(capture):
                out = calls[dtype]()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            # Broad on purpose: one dtype's config forced on the other can name a
            # tile whose shared-memory footprint that path refuses, and one arm
            # failing to compile must not take the cell, or the session, down.
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            results[(arm, dtype)] = result
            continue
        seen = capture.calls[0].config if capture.calls else None
        result.observed_config = seen
        if forced is not None:
            if seen is not None:
                result.override_verified = all(seen.get(k) == v
                                               for k, v in forced.items())
            elif get_config is not None:
                # The recorder saw nothing -- a vLLM that memoised its own lookup
                # would do that -- so ask the hook directly. Weaker evidence,
                # since it proves the override is SET rather than that the kernel
                # read it, but it is the difference between UNKNOWN and a check.
                with context(arm):
                    live = get_config()
                result.override_verified = bool(live) and all(
                    live.get(k) == v for k, v in forced.items())
        if check_correctness and arm == NATIVE_ARM:
            budget = tolerance(specs[dtype])
            golden = golden_forward(specs[dtype], weights_by_dtype[dtype], x,
                                    forced_topk_ids=ids)
            result.correctness_rel_err = relative_error(out, golden)
            result.correctness_budget = budget.rel_max
            del golden
        results[(arm, dtype)] = result
        del out

    usable = [k for k in pending if not results[k].error]
    samples: dict[tuple[str, str], list[float]] = {k: [] for k in usable}
    clocks_start = ClockState.sample()
    for _ in range(args.reps):
        for arm in ARMS:
            for dtype in dtypes:
                key = (arm, dtype)
                if key not in samples:
                    continue
                try:
                    with context(arm):
                        samples[key].extend(
                            time_calls(calls[dtype], args.warmup, args.iters))
                except Exception as exc:  # noqa: BLE001
                    results[key].error = f"{type(exc).__name__}: {exc}"[:300]
                    samples.pop(key, None)
    clocks_end = ClockState.sample()

    for key in pending:
        got = samples.get(key)
        if got:
            summarise_samples(results[key], got)
        results[key].sm_clock_start_mhz = clocks_start.sm_clock_mhz
        results[key].sm_clock_end_mhz = clocks_end.sm_clock_mhz
        store.write(results[key], cell, meta)
    del x, ids, topk_w, logits, calls
    return results


# --------------------------------------------------------------------------
# The synthetic world, so the whole analysis is exercisable on a laptop.
# --------------------------------------------------------------------------

def synthetic_results(cells: list[Cell], dtypes: list[str], ceilings: Ceilings,
                      *, fp8_flop_ratio: float, act_bytes_fp8: int,
                      overhead_ms: float, noise: float, seed: int,
                      alpha: float | None = None
                      ) -> dict[tuple[str, int], dict[tuple[str, str], ArmResult]]:
    """Cells GENERATED from the model at a planted fp8/bf16 FLOP ratio.

    This is what makes "C3 can tell a small dtype effect from a large one" a
    check rather than a claim. The tile effect is not planted separately: it
    falls out of the configs themselves, because `predicted_ms` reads
    BLOCK_SIZE_M through the M-tile count and GROUP_SIZE_M through the MEASURED
    `alpha_for_group` curve. So the synthetic world's tile effect is grounded in
    the swizzle measurement rather than invented.

    TO EXERCISE C3 THE SELF TEST HAS TO LEAVE THE MEASURED ALPHA, and its first
    version did not, which is how this comment came to exist. At alpha 0.68-0.84
    the ceiling `2 BM / (alpha b)` sits below the ridge for every config vLLM
    picks, the compute term never binds, and the generated times are pure byte
    counts -- so `--self-test 2.033` and `--self-test 2.400` produced numerically
    IDENTICAL reports and C3 passed in both. `--self-test-alpha 0.2` lifts the
    ceiling clear of the ridge and the planted FLOP ratio then moves the answer,
    which is the world the gate has to be able to fail in. That the measured
    alpha makes the fp8 FLOP rate unobservable in this kernel is a finding, not
    a defect of the harness.
    """
    rng = random.Random(seed)
    planted = Ceilings(
        name=f"{ceilings.name} (planted fp8 ratio {fp8_flop_ratio})",
        bandwidth_bytes_s=ceilings.bandwidth_bytes_s,
        peak_flops={BF16: ceilings.peak(BF16),
                    FP8: ceilings.peak(BF16) * fp8_flop_ratio},
        source="SYNTHETIC: generated by --self-test, nothing was measured",
        measured_on="n/a", path="n/a")
    out: dict[tuple[str, int], dict[tuple[str, str], ArmResult]] = {}
    for cell in cells:
        cfg = MODEL_CONFIGS[cell.model]
        per_cell: dict[tuple[str, str], ArmResult] = {}
        for arm in ARMS:
            for dtype in dtypes:
                forced = arm_config(cell, arm)
                config = forced or cell.configs[dtype]
                # NOTHING THAT A VALIDITY GATE READS IS FILLED IN HERE. An
                # earlier version set observed_config, override_verified and a
                # weight dtype, and V1, V3 and V4 then PASSED on a laptop with no
                # kernel, no vLLM and no GPU -- three validity gates certifying
                # evidence this function had just invented. They now read UNKNOWN
                # in a self test, which is the honest state: the self test
                # exercises the REDUCTION, and says nothing about the box.
                result = ArmResult(cell.model, cell.num_tokens, arm, dtype,
                                   config=forced,
                                   config_origin="forced" if forced else "observed")
                if arm_is_redundant(cell, arm):
                    result.config_origin, result.redundant = "redundant", True
                    per_cell[(arm, dtype)] = result
                    continue
                act = 2 if dtype == BF16 else act_bytes_fp8
                ms = predicted_ms(cfg, cell.num_tokens, config, dtype,
                                  ceilings=planted, act_bytes=act,
                                  overhead_ms=overhead_ms, alpha=alpha)
                draws = [ms * math.exp(rng.gauss(0.0, noise)) if noise else ms
                         for _ in range(max(2, 8))]
                summarise_samples(result, draws)
                per_cell[(arm, dtype)] = result
        out[cell.key] = per_cell
    return out


# --------------------------------------------------------------------------
# Analysis.
# --------------------------------------------------------------------------

def bootstrap_interval(values: list[float], band: float = BOOTSTRAP_BAND,
                       reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
                       ) -> tuple[float, float] | None:
    """Percentile bootstrap of the MEDIAN, resampling CELLS.

    Cells and not timing samples: the noise inside one cell is already summarised
    by its own median, and the uncertainty that matters for a headline is whether
    another set of cells would have said the same.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(statistics.median([values[rng.randrange(n)] for _ in range(n)])
                     for _ in range(reps))
    lo = medians[int((1 - band) / 2 * reps)]
    hi = medians[min(reps - 1, int((1 + band) / 2 * reps))]
    return lo, hi


@dataclass(frozen=True)
class Shift:
    """One (model, arm) dtype effect: the ratio curve, its tilt, its branches.

    TWO STATISTICS, and the difference between them is the whole reason this
    dataclass is not just a float.

    `tilt` is the median `r` over the lowest third of the token grid divided by
    the median over the highest third. It is defined by the GRID, so it exists
    in every run, is the same instrument for every arm, and no assumption about
    where the ridge sits can steer it. Every gate reads this one.

    `shift` is `rm / rc` over branches read off the measured bf16 curve. For a
    two-branch curve that IS the crossing ratio: below the ridge time is flat at
    `A`, above it linear at `C T`, so scaling the branches by `rm` and `rc` moves
    the intersection `A/C` to `A rm / (C rc)`. It is None whenever the curve does
    not show two branches -- an arm whose config caps arithmetic intensity below
    the ridge never reaches a compute branch, and that is a prediction of the
    tile-corrected roofline rather than a failed measurement.

    Where the sweep straddles the transition the two agree, and the report
    prints them side by side so a reader can see whether they did.
    """

    arm: str
    model: str
    points: tuple[tuple[int, float, float], ...]   # (tokens, bf16_ms, fp8_ms)
    low_tokens: tuple[float, ...]
    high_tokens: tuple[float, ...]
    r_low: float
    r_high: float
    memory_tokens: tuple[float, ...]
    compute_tokens: tuple[float, ...]
    rm: float | None
    rc: float | None
    memory_slope_bf16: float | None
    compute_slope_bf16: float | None
    log_slope: float | None
    unpaired: tuple[int, ...]

    @property
    def tilt(self) -> float:
        return self.r_low / self.r_high

    @property
    def shift(self) -> float | None:
        if self.rm is None or not self.rc:
            return None
        return self.rm / self.rc

    @property
    def branch_note(self) -> str:
        """Why there is no `shift`, in the words a reader needs."""
        if self.shift is not None:
            return ""
        missing = []
        if self.rm is None:
            missing.append(f"no flat stretch (pooled bf16 slope never <= "
                           f"{MEMORY_SLOPE_MAX})")
        if self.rc is None:
            missing.append(f"no steep stretch (pooled bf16 slope never >= "
                           f"{COMPUTE_SLOPE_MIN}), which is what "
                           f"2*BLOCK_M/(alpha*b) < ridge predicts")
        return "; ".join(missing)


def median_ratio(pairs: list[tuple[float, float]]) -> float | None:
    """Median of PER-CELL ratios, never a ratio of pooled medians.

    Per cell first, because the grid spans three orders of magnitude in absolute
    time and a pooled ratio would be the T=8192 cell wearing a costume.
    """
    ratios = [a / b for a, b in pairs if b > 0]
    return statistics.median(ratios) if ratios else None


def build_shift(model: str, arm: str, series: list[tuple[int, float, float]],
                unpaired: list[int]) -> Shift:
    """Reduce one arm's paired `(T, bf16_ms, fp8_ms)` series. Refuses when thin.

    Raises `RegimeNotResolved` when the grid holds too few paired cells for a
    tilt at all -- the one statistic every gate reads, so a thin grid must stop
    the number rather than shrink it. The branch-based `shift` is allowed to be
    None on the same series, because "this arm never reached a compute branch"
    is an answer.
    """
    ordered = sorted(series)
    tokens = [float(t) for t, _, _ in ordered]
    if len(ordered) < 2 * MIN_REGION_CELLS:
        raise RegimeNotResolved(
            f"{model} arm {arm}: {len(ordered)} paired cells, below the "
            f"{2 * MIN_REGION_CELLS} a tilt needs ({MIN_REGION_CELLS} in each "
            f"third). Widen --tokens, or look at the failed arms below."
            + (f" Token counts with only one dtype: {sorted(unpaired)}."
               if unpaired else ""))
    low, high = grid_thirds(tokens)
    by_token = {float(t): (bf16, fp8) for t, bf16, fp8 in ordered}
    r_low = median_ratio([(by_token[t][1], by_token[t][0]) for t in low])
    r_high = median_ratio([(by_token[t][1], by_token[t][0]) for t in high])
    if not r_low or not r_high:                               # pragma: no cover
        raise RegimeNotResolved(f"{model} arm {arm}: a grid third held no time")

    # The branches come off the bf16 curve, and the SAME token sets are then used
    # for both dtypes. Classifying each dtype separately would let the two sides
    # of a ratio be taken over different cells, which is the ragged-grid pooling
    # this project keeps having to retract.
    memory, compute = classify_branches([(t, by_token[t][0]) for t in tokens])
    rm = median_ratio([(by_token[t][1], by_token[t][0]) for t in memory]) \
        if len(memory) >= MIN_REGION_CELLS else None
    rc = median_ratio([(by_token[t][1], by_token[t][0]) for t in compute]) \
        if len(compute) >= MIN_REGION_CELLS else None
    return Shift(
        arm=arm, model=model, points=tuple(ordered),
        low_tokens=tuple(low), high_tokens=tuple(high),
        r_low=r_low, r_high=r_high,
        memory_tokens=tuple(memory), compute_tokens=tuple(compute),
        rm=rm, rc=rc,
        memory_slope_bf16=log_log_slope([(t, by_token[t][0]) for t in memory]),
        compute_slope_bf16=log_log_slope([(t, by_token[t][0]) for t in compute]),
        log_slope=log_log_slope([(t, by_token[t][1] / by_token[t][0])
                                 for t in tokens]),
        unpaired=tuple(sorted(unpaired)))


def cell_ms(results: dict[tuple[str, str], ArmResult], arm: str, dtype: str,
            cell: Cell) -> float | None:
    """One arm's median time, resolving the redundant arm to its twin.

    A `cfg_fp8` arm that was never timed because it is identical to `cfg_bf16`
    is not missing data: it is the same kernel, and reading its twin's time is
    exact rather than an approximation. Anything else -- a compile failure, an
    absent row -- returns None and is excluded from every median.
    """
    result = results.get((arm, dtype))
    if result is not None and result.redundant and arm == "cfg_fp8":
        result = results.get(("cfg_bf16", dtype))
    if result is None or result.error or not result.ms_median:
        return None
    if result.num_tokens != cell.num_tokens or result.model != cell.model:
        # The dict is keyed by (arm, dtype) within one cell, so this can only
        # fire if two cells' results were merged. Every ratio downstream assumes
        # the pairing is by (model, tokens, arm), and a merge would make it a
        # comparison of two different batches wearing one label.
        raise UnpairableComparison(
            f"a result for {result.model} T={result.num_tokens} was stored under "
            f"{cell.model} T={cell.num_tokens}; the pairing is broken")
    return result.ms_median


def compute_shift(cells: list[Cell], results, model: str, arm: str,
                  dtypes: list[str]) -> Shift:
    """Pair one (model, arm) across the two dtypes, then reduce it.

    A cell enters only when BOTH dtypes produced a time for it, and one that did
    not is named as unpaired rather than dropped quietly. Matching on
    (model, tokens, arm) is what makes every ratio here a comparison at matched
    levels of everything it is not about.
    """
    if set(dtypes) != set(DTYPES):
        raise UnpairableComparison(
            f"a dtype shift needs both {DTYPES}; this run measured "
            f"{tuple(dtypes)}. `--dtypes bf16` prices the CONFIG effect at fixed "
            f"dtype and nothing else, and every dtype number is refused.")
    series: list[tuple[int, float, float]] = []
    unpaired: list[int] = []
    for cell in cells:
        if cell.model != model:
            continue
        per_cell = results.get(cell.key, {})
        bf16 = cell_ms(per_cell, arm, BF16, cell)
        fp8 = cell_ms(per_cell, arm, FP8, cell)
        if bf16 is None or fp8 is None:
            if bf16 is not None or fp8 is not None:
                unpaired.append(cell.num_tokens)
            continue
        series.append((cell.num_tokens, bf16, fp8))
    return build_shift(model, arm, series, unpaired)


def predicted_shift(model: str, cells: list[Cell], ceilings: Ceilings,
                    act_bytes_fp8: int, arm: str) -> Shift | None:
    """The model's own answer for one arm, through the SAME reduction.

    Generated as a `(T, bf16_ms, fp8_ms)` series and handed to `build_shift`,
    rather than solved in closed form, so a prediction and a measurement that
    disagree disagree about the world and not about the reduction. Returns None
    where the grid is too thin for the reduction to run at all.
    """
    cfg = MODEL_CONFIGS[model]
    series: list[tuple[int, float, float]] = []
    for cell in cells:
        if cell.model != model:
            continue
        forced = arm_config(cell, arm)
        bf16 = predicted_ms(cfg, cell.num_tokens, forced or cell.configs[BF16],
                            BF16, ceilings=ceilings, act_bytes=2)
        fp8 = predicted_ms(cfg, cell.num_tokens, forced or cell.configs[FP8],
                           FP8, ceilings=ceilings, act_bytes=act_bytes_fp8)
        series.append((cell.num_tokens, bf16, fp8))
    try:
        return build_shift(model, arm, series, [])
    except ConfoundRefusal:
        return None


def config_share(tilt_native: float, tilt_matched: float) -> float | None:
    """What fraction of the confounded arm's EXCESS the config carries.

    Both are ratios, so the excesses are `x - 1` and the share is the ratio of
    those. None when the confounded arm has no excess to apportion, rather than
    dividing by something near zero and printing a share of 4000%.
    """
    if tilt_native <= 1.0:
        return None
    return (tilt_native - tilt_matched) / (tilt_native - 1.0)


@dataclass
class Analysis:
    """Everything the report prints and every gate reads, in one place."""

    shifts: dict[tuple[str, str], Shift] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)
    #: (model, arm, activation-mode) -> the model's own answer, reduced by the
    #: same code path as the measurement.
    predicted: dict[tuple[str, str, str], Shift] = field(default_factory=dict)
    placebo_deviations: list[float] = field(default_factory=list)
    placebo_worst: str = ""
    override_checked: int = 0
    override_failures: list[str] = field(default_factory=list)
    derivation_checked: int = 0
    derivation_mismatches: list[str] = field(default_factory=list)
    distinct_observed_configs: int = 0
    correctness: list[tuple[str, float, float]] = field(default_factory=list)
    fp8_weight_dtypes: set[str] = field(default_factory=set)
    fp8_quant_kinds: set[str] = field(default_factory=set)
    spreads: list[float] = field(default_factory=list)
    clock_drift_pct: float | None = None
    clock_throttled: bool = False
    timed_arms: int = 0
    failed_arms: list[str] = field(default_factory=list)
    crossings: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    block_m_agreement: tuple[int, int] = (0, 0)
    configs_differ: tuple[int, int] = (0, 0)
    #: Every model the run PLANNED, not only the ones that produced a number. A
    #: model whose every arm was refused still has to appear in the report, or
    #: the page silently narrows to whatever happened to work.
    models: list[str] = field(default_factory=list)

    def shift_of(self, model: str, arm: str) -> Shift | None:
        return self.shifts.get((model, arm))


def analyse(cells: list[Cell], results, ceilings: Ceilings, dtypes: list[str]
            ) -> Analysis:
    """Reduce the run. No printing, no torch, so it runs against synthetic arms.

    Every aggregate here is taken at MATCHED levels of everything it is not
    about: a shift is per (model, arm), a placebo is per (cell, dtype), and a
    ratio only exists where both dtypes produced a time for the same cell and
    the same arm. Nothing is pooled across models, because the two default models
    differ by a factor of eight in `E/k` and their ridges land at different token
    counts.
    """
    analysis = Analysis()
    models = list(dict.fromkeys(c.model for c in cells))
    analysis.models = models

    for model in models:
        for arm in ARMS:
            try:
                analysis.shifts[(model, arm)] = compute_shift(
                    cells, results, model, arm, dtypes)
            except ConfoundRefusal as exc:
                analysis.refusals.append(f"{type(exc).__name__}: {exc}")
        for label, act in ACT_MODES:
            for arm in ARMS:
                try:
                    got = predicted_shift(model, cells, ceilings, act, arm)
                except ConfoundRefusal as exc:
                    # A REFUSAL IS RECORDED, NOT RAISED THROUGH THE REDUCTION.
                    # The predicted band is corroboration beside C3's measured
                    # tilt, not an input to any gate, so a calibration that
                    # cannot state one must cost the band and nothing else.
                    # `--dtypes bf16` is the live case and it is the fallback
                    # the fp8 refusal above POINTS AT: the A100 has no fp8
                    # units and no fp8 peak on file, so `predicted_ms` raised
                    # CalibrationIncomplete out of `analyse` and took the whole
                    # run with it -- the named escape hatch crashed.
                    analysis.refusals.append(
                        f"predicted band for {model} {arm} {label}: "
                        f"{type(exc).__name__}: {exc}")
                    continue
                if got is not None:
                    analysis.predicted[(model, arm, label)] = got

    observed_configs: set[tuple] = set()
    worst_placebo = 0.0
    for cell in cells:
        per_cell = results.get(cell.key, {})
        for arm in ARMS:
            for dtype in dtypes:
                result = per_cell.get((arm, dtype))
                if result is None:
                    continue
                where = f"{cell.model} T={cell.num_tokens} {arm} {dtype}"
                if result.error:
                    analysis.failed_arms.append(f"{where}: {result.error}")
                    continue
                if result.redundant:
                    continue
                if result.ms_median:
                    analysis.timed_arms += 1
                spread = result.spread
                if spread is not None:
                    analysis.spreads.append(spread)
                if result.observed_config:
                    observed_configs.add(tuple(sorted(
                        (k, v) for k, v in result.observed_config.items()
                        if k in CONFIG_KEYS)))
                if result.override_verified is not None:
                    analysis.override_checked += 1
                    if not result.override_verified:
                        analysis.override_failures.append(
                            f"{where}: forced {format_config(result.config)} but "
                            f"vLLM ran {result.observed_config}")
                if arm == NATIVE_ARM and result.observed_config:
                    analysis.derivation_checked += 1
                    derived = cell.configs[dtype]
                    differing = {k: (v, result.observed_config.get(k))
                                 for k, v in derived.items()
                                 if result.observed_config.get(k) != v}
                    if differing:
                        analysis.derivation_mismatches.append(
                            f"{where}: DERIVED {format_config(derived)} but the "
                            f"run OBSERVED {differing} (derived, observed)")
                if dtype == FP8:
                    analysis.fp8_weight_dtypes.add(result.weight_torch_dtype)
                    analysis.fp8_quant_kinds.add(result.quant_config_kind)
                if result.correctness_rel_err is not None:
                    analysis.correctness.append(
                        (where, result.correctness_rel_err,
                         result.correctness_budget or 0.0))
        # The placebo: `native` against the arm that FORCES the config this dtype
        # resolves natively. Same kernel, one resolved and one forced.
        for dtype in dtypes:
            partner = PLACEBO_PARTNER.get(dtype)
            if partner is None:
                continue
            native = cell_ms(per_cell, NATIVE_ARM, dtype, cell)
            forced = cell_ms(per_cell, partner, dtype, cell)
            if native and forced:
                deviation = abs(forced / native - 1.0)
                analysis.placebo_deviations.append(deviation)
                if deviation > worst_placebo:
                    worst_placebo = deviation
                    analysis.placebo_worst = (
                        f"{cell.model} T={cell.num_tokens} {dtype}: "
                        f"{partner}/{NATIVE_ARM} = {forced / native:.4f}")
    analysis.distinct_observed_configs = len(observed_configs)

    # Crossings, as corroboration only. Matched by upcrossing INDEX, and the
    # `min_tokens` floor is the model's own saturation batch: below E/k tokens
    # not every expert is active, weight traffic grows with the batch, and the
    # slope crosses 0.5 for a reason that has nothing to do with the ridge.
    for model in models:
        cfg = MODEL_CONFIGS[model]
        floor = float(cfg.num_experts) / cfg.top_k
        for arm in ARMS:
            for dtype in dtypes:
                points = []
                for cell in cells:
                    if cell.model != model:
                        continue
                    ms = cell_ms(results.get(cell.key, {}), arm, dtype, cell)
                    if ms:
                        points.append((float(cell.num_tokens), ms))
                if len(points) >= 3:
                    analysis.crossings[(model, arm, dtype)] = \
                        all_crossings_from_points(sorted(points), min_tokens=floor)

    bracket = crossing_bracket_cells(cells)
    analysis.block_m_agreement = (sum(1 for c in bracket if c.block_m_agrees),
                                  len(bracket))
    analysis.configs_differ = (sum(1 for c in bracket if c.configs_differ),
                               len(bracket))

    # PER CELL, and the worst one, not first-start against last-end. A resumed
    # run holds rows from two sessions in one CSV, and differencing the first
    # session's opening clock against the second's closing clock is a number
    # about nothing. Each cell carries the clocks that bracket its own timing
    # block, which is the window every ratio in that cell was measured inside.
    drifts = [(r.sm_clock_start_mhz - r.sm_clock_end_mhz)
              / r.sm_clock_start_mhz * 100.0
              for per in results.values() for r in per.values()
              if r.sm_clock_start_mhz > 0 and r.sm_clock_end_mhz > 0]
    if drifts:
        worst = max(drifts, key=abs)
        analysis.clock_drift_pct = worst
        analysis.clock_throttled = abs(worst) > MAX_CLOCK_DRIFT_PCT
    return analysis


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `verdict` is UNKNOWN when the run could not evaluate it, and UNKNOWN is never
    printed as a pass. A gate nobody could check is the state this project's
    retractions were written in.
    """

    name: str
    kind: str            # VALIDITY | CLAIM
    prediction: str
    rule: str
    verdict: str
    observed: str
    invalidates: str = ""

    def render(self) -> str:
        out = [f"[{self.verdict:7s}] {self.kind:8s} {self.name}  {self.prediction}",
               f"                    gate: {self.rule}",
               f"                    saw:  {self.observed}"]
        if self.verdict == FAIL and self.invalidates:
            out.append(f"                    a FAIL here: {self.invalidates}")
        return "\n".join(out)


def _verdict(value: bool | None) -> str:
    return UNKNOWN if value is None else (PASS if value else FAIL)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: every value printed was actually measured."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q * len(ordered) + 0.5))))
    return ordered[rank - 1]


def build_gates(analysis: Analysis, ceilings: Ceilings, dtypes: list[str],
                fp8_note: str, synthetic: bool = False) -> list[Gate]:
    """Six validity gates and four claim gates, in that order.

    Validity first so a reader hits the reasons to distrust the page before the
    page's conclusion.
    """
    gates: list[Gate] = []
    fp8_measured = FP8 in dtypes

    # ---- VALIDITY ----------------------------------------------------------
    has_both = all(ceilings.peak_flops.get(d) for d in DTYPES)
    gates.append(Gate(
        "V0 ceilings", "VALIDITY",
        "one calibration on THIS machine carries both dtypes' ceilings",
        f"{ceilings.path} has a measured peak for both {DTYPES}",
        _verdict(has_both),
        (f"bf16 {ceilings.peak_flops.get(BF16, 0) / 1e12:.1f} TFLOP/s, "
         f"fp8_e4m3 {ceilings.peak_flops.get(FP8, 0) / 1e12:.1f} TFLOP/s, "
         f"bandwidth {ceilings.bandwidth_bytes_s / 1e9:.1f} GB/s, measured "
         f"{ceilings.measured_on}") if has_both else
        f"{sorted(ceilings.peak_flops)} measured; one of {DTYPES} is missing",
        "every predicted shift on this page was computed against a ceiling this "
        "machine did not measure, which is the exact defect FINDINGS flags on "
        "the published fp8 arm"))

    weight_dtypes = analysis.fp8_weight_dtypes - {""}
    quant_kinds = analysis.fp8_quant_kinds - {"", "none"}
    fp8_real = None
    if not fp8_measured:
        observed = "fp8 was not measured in this run"
    elif not weight_dtypes:
        observed = "no fp8 arm recorded a weight dtype, so nothing was checked"
    else:
        fp8_real = (all("float8" in d for d in weight_dtypes) and bool(quant_kinds))
        observed = (f"fp8 weights arrived as {sorted(weight_dtypes)}, quant config "
                    f"{sorted(quant_kinds) or ['NONE']}; {fp8_note}")
    gates.append(Gate(
        "V1 fp8 is fp8", "VALIDITY",
        "the fp8 arms handed the kernel fp8 weights AND a quant config",
        "every fp8 arm's weight dtype is a torch float8 type and its quant "
        "config is not None",
        _verdict(fp8_real), observed,
        "the fp8 rows were computed by a path that dequantised to another "
        "format, and every ratio on this page is bf16 against bf16"))

    worst = max((rel for _, rel, _ in analysis.correctness), default=None)
    over = [w for w, rel, budget in analysis.correctness if budget and rel > budget]
    gates.append(Gate(
        "V2 correctness", "VALIDITY",
        "each dtype computes the layer the fp32 oracle computes",
        "relative error against golden_forward within moe.bench.tolerance's budget",
        _verdict(None if worst is None else not over),
        "no cell was checked against the oracle. The check belongs to the "
        "session that RAN the cell, so a fully resumed run inherits no oracle "
        "comparison and this reads UNKNOWN rather than PASS"
        if worst is None
        else (f"{len(analysis.correctness)} checks, worst rel err {worst:.3e}, "
              f"{len(over)} over budget"
              + ("" if not over else f"; first {over[0]}")),
        "one of the two sides is not computing the MoE layer, so the ratio "
        "between them is not a time ratio for the same work"))

    override_ok = (None if analysis.override_checked == 0
                   else not analysis.override_failures)
    distinct = analysis.distinct_observed_configs
    # Only bites once something WAS watched. "Fewer than two distinct configs"
    # and "nothing looked at a config" are different states and this project has
    # confused them before; the second is UNKNOWN, decided by the clause above.
    if override_ok and distinct < 2:
        # Every arm ran ONE config. That is what a silently-failing override
        # looks like, and it is fatal in exactly the way the gate exists to
        # catch, so it demotes the pass rather than being a footnote.
        override_ok = False
    gates.append(Gate(
        "V3 override", "VALIDITY",
        "override_config forces what it is given, and the arms are not one kernel",
        "zero arms running a config they were not given, and >= 2 distinct "
        "observed configs across the run",
        _verdict(override_ok),
        "no forced arm was watched, so the override is still unchecked"
        if analysis.override_checked == 0
        else (f"{len(analysis.override_failures)} of {analysis.override_checked} "
              f"forced arms ran a config they were not given; "
              f"{distinct} distinct observed configs"
              + ("" if not analysis.override_failures
                 else "; first: " + analysis.override_failures[0])),
        "the arms are comparisons of one kernel with itself and every ratio "
        "below is a placebo wearing a label"))

    derived_ok = (None if analysis.derivation_checked == 0
                  else not analysis.derivation_mismatches)
    gates.append(Gate(
        "V4 derivation", "VALIDITY",
        f"vLLM {VLLM_TAG} loads the config tile_resolve DERIVES",
        "zero native cells where the observed config differs from the derived one",
        _verdict(derived_ok),
        "no native config was observed, so the derivation is unchecked"
        if analysis.derivation_checked == 0
        else (f"{len(analysis.derivation_mismatches)} mismatches over "
              f"{analysis.derivation_checked} observed native arms"
              + ("" if not analysis.derivation_mismatches
                 else "; first: " + analysis.derivation_mismatches[0])),
        "the plan printed above describes a different experiment than the one "
        "that ran, and the placebo pairs are not placebos"))

    band = percentile(analysis.placebo_deviations, 0.90)
    spread = percentile(analysis.spreads, 0.90)
    drift = analysis.clock_drift_pct
    noise_ok = None
    parts = []
    if band is not None:
        parts.append(f"p90 placebo deviation {band:.2%} over "
                     f"{len(analysis.placebo_deviations)} pairs")
    if spread is not None:
        parts.append(f"p90 per-timing spread {spread:.2%}")
    if drift is not None:
        parts.append(f"worst per-cell SM clock drift {drift:+.1f}%")
    if band is not None:
        noise_ok = (band < PLACEBO_BAND
                    and (spread is None or spread < MAX_TIMING_SPREAD)
                    and not analysis.clock_throttled)
    gates.append(Gate(
        "V5 noise floor", "VALIDITY",
        "the box can resolve an effect the size of the one being measured",
        f"p90 |placebo - 1| < {PLACEBO_BAND:.0%}, p90 timing spread < "
        f"{MAX_TIMING_SPREAD:.0%}, |clock drift| <= {MAX_CLOCK_DRIFT_PCT:.0f}%",
        _verdict(noise_ok),
        "no placebo pair was timed" if band is None
        else "; ".join(parts) + (f"; worst {analysis.placebo_worst}"
                                 if analysis.placebo_worst else ""),
        "the model's predicted band is 1.016 to 1.053, about 4 points wide, and "
        "the box cannot see 4 points, so C3 cannot be read either way"))

    # ---- CLAIM -------------------------------------------------------------
    differ, total = analysis.configs_differ
    gates.append(Gate(
        "C1 confound exists", "CLAIM",
        "the two dtypes resolve DIFFERENT configs at the crossing",
        "the configs differ in at least one key at every crossing-bracketing cell",
        _verdict(None if total == 0 else differ == total),
        "no cell brackets a published crossing for any model in this run"
        if total == 0 else
        f"{differ} of {total} cells within {CROSSING_BRACKET}x of the published "
        f"bf16 crossing have differing configs",
        "there was no config confound to separate at the crossing, and the "
        "published 1.15 has to be explained by something else entirely"))

    agree, total = analysis.block_m_agreement
    fraction = agree / total if total else None
    gates.append(Gate(
        "C2 not the M tile", "CLAIM",
        "BLOCK_SIZE_M AGREES across the dtypes where the crossing lives",
        f"agreement fraction >= {BLOCK_M_AGREEMENT_MIN:.0%} over cells within "
        f"{CROSSING_BRACKET}x of the published bf16 crossing",
        _verdict(None if fraction is None
                 else fraction >= BLOCK_M_AGREEMENT_MIN),
        "no cell brackets a published crossing" if fraction is None
        else (f"BLOCK_SIZE_M agrees in {agree} of {total} bracketing cells "
              f"({fraction:.0%}); this is DERIVED from vLLM's shipped configs "
              f"and is decided off GPU"),
        "FINDINGS' stated mechanism -- the fp8 arm running taller M tiles -- "
        "does hold at the crossing after all, and this script's premise is wrong"))

    matched = [analysis.shift_of(m, arm) for m in analysis.models
               for arm in MATCHED_ARMS]
    matched_values = [s.tilt for s in matched if s is not None]
    pure = statistics.median(matched_values) if matched_values else None
    interval = bootstrap_interval(matched_values)
    pure_ok = (None if pure is None
               else PURE_DTYPE_SHIFT_LO <= pure <= PURE_DTYPE_SHIFT_HI)
    band = sorted(got.tilt for (_, arm, _), got in analysis.predicted.items()
                  if arm in MATCHED_ARMS)
    gates.append(Gate(
        "C3 pure dtype", "CLAIM",
        "at a MATCHED config the format barely tilts the fp8/bf16 ratio",
        f"median matched tilt in [{PURE_DTYPE_SHIFT_LO}, {PURE_DTYPE_SHIFT_HI}]; "
        + (f"the model at the MEASURED alpha predicts [{band[0]:.3f}, "
           f"{band[-1]:.3f}] over the same cells"
           if band else "the model produced no band on this grid"),
        _verdict(pure_ok),
        "no matched arm produced a tilt" if pure is None
        else (f"median matched tilt {pure:.3f} over {len(matched_values)} "
              f"(model, arm) pairs"
              + ("" if interval is None
                 else f", {BOOTSTRAP_BAND:.0%} interval "
                      f"[{interval[0]:.3f}, {interval[1]:.3f}]")
              + f"; the published confounded figure is {PUBLISHED_SHIFT:.3f} "
                f"+/- {PUBLISHED_SHIFT_SD:.3f}"),
        "the FORMAT itself moves the crossing, the config was never the "
        "explanation, and C2 of the study needs a correction rather than the "
        "1.15 needing a caveat"))

    shares = []
    for model in analysis.models:
        native = analysis.shift_of(model, NATIVE_ARM)
        for arm in MATCHED_ARMS:
            pinned = analysis.shift_of(model, arm)
            if native is None or pinned is None:
                continue
            share = config_share(native.tilt, pinned.tilt)
            if share is not None:
                shares.append(share)
    share_median = statistics.median(shares) if shares else None
    gates.append(Gate(
        "C4 config share", "CLAIM",
        "the config carries the majority of the confounded arm's excess",
        f"median (tilt_native - tilt_matched) / (tilt_native - 1) >= "
        f"{CONFIG_SHARE_MIN:.0%}",
        _verdict(None if share_median is None
                 else share_median >= CONFIG_SHARE_MIN),
        "the native arm showed no excess over 1.0 to apportion, so there is "
        "nothing for the config to explain" if share_median is None
        else f"median config share {share_median:.1%} over {len(shares)} pairs",
        "the confounded excess is mostly NOT the config, so pinning it does not "
        "close the 1.15 and the residual needs its own experiment"))

    if synthetic:
        # EVERY VALIDITY GATE THAT READS THE BOX IS DEMOTED TO UNKNOWN, with the
        # value it computed still printed so the reduction is visibly exercised.
        # V0 is exempt: it reads the calibration file, which is as real on a
        # laptop as on the pod. Without this the self test certified its own
        # invented rows -- see the note in `synthetic_results`.
        demoted = []
        for gate in gates:
            if gate.kind != "VALIDITY" or gate.name.startswith("V0"):
                demoted.append(gate)
                continue
            demoted.append(Gate(
                gate.name, gate.kind, gate.prediction, gate.rule, UNKNOWN,
                "SELF TEST, synthetic rows: no kernel ran, so this gate has "
                "nothing of this machine's to read. Computed anyway: "
                + gate.observed,
                gate.invalidates))
        gates = demoted
    return gates


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------

SIGN_BANNER = """\
THE SIGN, stated once so it cannot be misread. Every ratio below is

    r = time(fp8_e4m3) / time(bf16)          at MATCHED model, tokens and arm
    shift = rm / rc                          memory-region r over compute-region r

r < 1 means fp8 is FASTER, which it must be: it moves half the weight bytes and
runs tensor cores at 2.03x. shift > 1 means the crossing moves LATER in fp8,
which is the direction the published 1.149 reports and the direction the
corrected theory says should not happen. Every shift printed here is followed by
the words LATER or EARLIER."""


def shift_sentence(shift: float) -> str:
    """The shift as a sentence naming which way the crossing moved.

    A table of bare ratios is exactly how a direction gets misread, and reversing
    this one would flip the study's conclusion between "fp8 crosses where bf16
    does" and "quantising moves the ridge".
    """
    if shift > 1.0:
        return f"fp8 crosses {100 * (shift - 1):.1f}% LATER than bf16"
    if shift < 1.0:
        return f"fp8 crosses {100 * (1 / shift - 1):.1f}% EARLIER than bf16"
    return "fp8 and bf16 cross at exactly the same batch"


def render_derivation(cells: list[Cell], notes: list[str], models: list[str]) -> str:
    """The per-cell config table, DERIVED, printed before anything runs.

    This is the half of the experiment that needs no GPU, and C1 and C2 are both
    decided here. A reader sees what is being compared before seeing a number
    that might make them want a different comparison.
    """
    lines = ["## The confound, DERIVED from vLLM's shipped configs before any run",
             "",
             f"vLLM {VLLM_TAG}. `resolve_tile` reproduces `get_config_file_name` +",
             "the nearest-key lookup + `get_default_config`; nothing below was "
             "observed on a GPU.",
             ""]
    for note in notes:
        lines.append(f"  dropped -- {note}")
    if notes:
        lines.append("")
    lines.append("| model | T | rows/E | bf16 config | fp8_e4m3 config | "
                 "BLOCK_M | differs in |")
    lines.append("|---|---:|---:|---|---|---|---|")
    bracket = {c.key for c in crossing_bracket_cells(cells)}
    for cell in cells:
        mark = " *" if cell.key in bracket else ""
        lines.append(
            f"| {cell.model}{mark} | {cell.num_tokens} | {cell.rows:.1f} | "
            f"`{format_config(cell.configs[BF16])}` | "
            f"`{format_config(cell.configs[FP8])}` | "
            f"{'SAME' if cell.block_m_agrees else 'DIFFERS'} | "
            f"{', '.join(cell.differing_keys) or 'nothing'} |")
    lines.append("")
    lines.append("`*` marks a cell within a factor of "
                 f"{CROSSING_BRACKET} of that model's published bf16 crossing "
                 "-- the only cells C1 and C2 read.")
    for model in models:
        rows = [c for c in cells if c.model == model]
        marked = [c for c in rows if c.key in bracket]
        if not rows:
            continue
        agree = sum(1 for c in marked if c.block_m_agrees)
        lines.append(
            f"  {model}: BLOCK_SIZE_M agrees in {agree}/{len(marked)} bracketing "
            f"cells and {sum(1 for c in rows if c.block_m_agrees)}/{len(rows)} "
            f"over the whole grid; the configs differ somewhere in "
            f"{sum(1 for c in rows if c.configs_differ)}/{len(rows)}")
    return "\n".join(lines)


def render_predictions(cells: list[Cell], models: list[str], ceilings: Ceilings,
                       dtypes: list[str]) -> str:
    """Every registered number, with the ceiling file each one came from.

    Takes `dtypes` rather than assuming both. In a `--dtypes bf16` run there is
    no fp8 ceiling to quote and no tilt to predict, and printing an fp8 ridge
    that the calibration does not carry would either crash or invent one.
    """
    lines = ["## Predictions, registered before the run", "",
             f"Ceilings read from {ceilings.path}",
             f"  measured {ceilings.measured_on} -- {ceilings.source}",
             f"  bandwidth {ceilings.bandwidth_bytes_s / 1e9:.1f} GB/s",
             f"  bf16 {ceilings.peak_flops.get(BF16, 0) / 1e12:.1f} TFLOP/s  "
             f"-> ridge {ceilings.ridge(BF16):.1f} FLOP/byte"]
    if FP8 not in dtypes:
        lines += [
            "",
            "NO DTYPE PREDICTION. This run measures bf16 only, which is the "
            "largest",
            "defensible subset on a card with no fp8 tensor cores: it prices "
            "what letting",
            "each dtype pick its own CONFIG is worth, at fixed dtype, and every "
            "dtype",
            "number -- the tilt, the shift, C3 and C4 -- is REFUSED rather than "
            "computed",
            "from one side.",
            "",
            f"QUOTED, not measured here: the published confounded ratio is "
            f"{PUBLISHED_SHIFT:.3f} +/- {PUBLISHED_SHIFT_SD:.3f}",
            "(docs/FINDINGS.md). Nothing in this run bears on it.",
        ]
        return "\n".join(lines)
    lines += [
        f"  fp8  {ceilings.peak_flops.get(FP8, 0) / 1e12:.1f} TFLOP/s  "
        f"-> ridge {ceilings.ridge(FP8):.1f} FLOP/byte",
        f"  achieved fp8/bf16 = {ceilings.fp8_over_bf16:.3f}, so the compute "
        f"branch scales by rc = {1 / ceilings.fp8_over_bf16:.3f}",
        f"     (the DATASHEET relationship is 2.000 exactly; this card "
        f"measures {ceilings.fp8_over_bf16:.3f}, and the difference is "
        f"{abs(ceilings.fp8_over_bf16 - 2.0) / 2.0:.1%} of the prediction)",
        "",
        "PREDICTED TILT -- the median fp8/bf16 time ratio over the lowest "
        "third of the",
        "token grid divided by the median over the highest third, produced "
        "by running the",
        "model's own timings through the SAME reduction the measurement "
        "uses. `branches`",
        "is what the classifier finds on the predicted bf16 curve; `none` "
        "there means",
        "the model says this arm never reaches a compute branch, which is "
        "the",
        "`2*BLOCK_M/(alpha*b) < ridge` ceiling and not a defect of the grid.",
        "",
        "| model | arm | tilt, activations bf16 | tilt, activations "
        "quantised | branches (mem/comp) | model shift |",
        "|---|---|---:|---:|---|---:|"]
    for model in models:
        for arm in ARMS:
            values, branches, shift = [], "--", "--"
            for _, act in ACT_MODES:
                got = predicted_shift(model, cells, ceilings, act, arm)
                values.append("--" if got is None else f"{got.tilt:.3f}")
                if got is not None and act == 2:
                    branches = (f"{len(got.memory_tokens)}/"
                                f"{len(got.compute_tokens)}")
                    shift = ("none" if got.shift is None else f"{got.shift:.3f}")
            lines.append(f"| {model} | {arm} | {values[0]} | {values[1]} | "
                         f"{branches} | {shift} |")
    lines += [
        "",
        "Read the two tilt columns as a BAND. `moe/spec.py` keeps activations at "
        "bf16 in an",
        "fp8 cell because vLLM's `fused_experts` asserts it, but vLLM quantises "
        "them",
        "internally with `a1_scale=None`, and no timing can separate the two. "
        "The band is",
        "what the model can honestly say.",
        "",
        f"QUOTED, not measured here: the published confounded ratio is "
        f"{PUBLISHED_SHIFT:.3f} +/- {PUBLISHED_SHIFT_SD:.3f}",
        "over eight measurements from two kernels (docs/FINDINGS.md, from the "
        "arms",
        "2026-08-28-nvidia_h200-h200-fp8-three-kernel and -fp8-refixed). Per "
        "model, vLLM:",
        "  " + ",  ".join(
            f"{m} {PUBLISHED_CROSSING_RATIO[m]['vllm']:.2f}"
            for m in models if m in PUBLISHED_CROSSING_RATIO),
    ]
    return "\n".join(lines)


def render_gate_summary(gates: list[Gate]) -> str:
    lines = [g.render() for g in gates]
    counts = {v: sum(1 for g in gates if g.verdict == v)
              for v in (PASS, FAIL, UNKNOWN)}
    validity_failed = [g.name for g in gates
                       if g.kind == "VALIDITY" and g.verdict == FAIL]
    lines.append("")
    lines.append(f"{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[UNKNOWN]} UNKNOWN")
    if validity_failed:
        lines.append(f"VALIDITY GATES FAILED: {validity_failed}. No number on "
                     "this page may be quoted.")
    return "\n".join(lines)


def render_shifts(analysis: Analysis) -> str:
    """The measurement, arm by arm, with the branch statistic beside the tilt.

    Both are printed, always. The TILT is what every gate reads and it exists in
    every run; the branch SHIFT is the same quantity read as a crossing ratio and
    it is None whenever the measured curve does not show two branches. Printing
    only the tilt would hide that some arms never reached a compute branch;
    printing only the shift would leave whole rows blank for a reason a reader
    could not see.
    """
    lines = ["## The measurement: the fp8/bf16 ratio, its tilt, and its branches",
             "",
             "| model | arm | r (low third) | r (high third) | TILT | reads as | "
             "cells | branches mem/comp | rm | rc | branch shift |",
             "|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    for model in analysis.models:
        for arm in ARMS:
            got = analysis.shift_of(model, arm)
            if got is None:
                lines.append(f"| {model} | {arm} | -- | -- | -- | REFUSED, see "
                             f"below | -- | -- | -- | -- | -- |")
                continue
            branch = ("--" if got.shift is None else f"{got.shift:.4f}")
            lines.append(
                f"| {model} | {arm} | {got.r_low:.4f} | {got.r_high:.4f} | "
                f"{got.tilt:.4f} | {shift_sentence(got.tilt)} | "
                f"{len(got.points)} | "
                f"{len(got.memory_tokens)}/{len(got.compute_tokens)} | "
                + ("--" if got.rm is None else f"{got.rm:.4f}") + " | "
                + ("--" if got.rc is None else f"{got.rc:.4f}") + " | "
                + f"{branch} |")
    for model in analysis.models:
        for arm in ARMS:
            got = analysis.shift_of(model, arm)
            if got is not None and got.branch_note:
                lines.append(f"  no branch shift for {model} {arm}: "
                             f"{got.branch_note}")
    if analysis.refusals:
        lines += ["", "REFUSED (no number produced, and no substitute):"]
        lines += [f"  - {line}" for line in analysis.refusals]
    return "\n".join(lines)


def render_ratio_curves(analysis: Analysis) -> str:
    """Every per-cell ratio, so the tilt can be checked against the cells.

    The tilt is two medians over two thirds of a grid, and a reader has to be
    able to see whether the curve actually tilts or whether one cell moved it.
    The `L` and `H` marks say which third each cell fell in.
    """
    lines = ["## The ratio curve, cell by cell", ""]
    for model in analysis.models:
        for arm in ARMS:
            got = analysis.shift_of(model, arm)
            if got is None:
                continue
            low, high = set(got.low_tokens), set(got.high_tokens)
            cells = []
            for tokens, bf16, fp8 in got.points:
                mark = "L" if float(tokens) in low else (
                    "H" if float(tokens) in high else " ")
                cells.append(f"{tokens}:{fp8 / bf16:.3f}{mark}")
            slope = ("--" if got.log_slope is None else f"{got.log_slope:+.4f}")
            lines.append(f"  {model:16s} {arm:9s} dlog(r)/dlog(T) = {slope}")
            lines.append("    " + "  ".join(cells))
            if got.unpaired:
                lines.append(f"    UNPAIRED (one dtype only, excluded): "
                             f"{list(got.unpaired)}")
    lines += ["",
              "`L` marks the lowest third of the grid and `H` the highest; the "
              "tilt is the",
              "median over `L` divided by the median over `H`. "
              "`dlog(r)/dlog(T)` is the same",
              "tendency as a single fitted number over every cell, and a tilt "
              "that disagrees",
              "in sign with it is one cell doing the work."]
    return "\n".join(lines)


def render_decomposition(analysis: Analysis) -> str:
    """The confounded arm against the two matched ones, per model.

    The decomposition is exact and multiplicative:

        tilt(native) = config_effect x tilt(matched)

    so `tilt(native) / tilt(matched)` is what the config is worth, and
    `config_share` is that excess as a fraction of the confounded one. The two
    matched arms are NOT averaged away: both are printed, and if they disagree
    the dtype effect depends on which config it is measured at, which is a
    finding rather than a nuisance.
    """
    lines = ["## Decomposition: how much of the confounded tilt is the config",
             "",
             "| model | tilt(native) | tilt(cfg_bf16) | tilt(cfg_fp8) | "
             "config effect | config share |",
             "|---|---:|---:|---:|---:|---:|"]
    for model in analysis.models:
        native = analysis.shift_of(model, NATIVE_ARM)
        columns = []
        for arm in ARMS:
            got = analysis.shift_of(model, arm)
            columns.append("--" if got is None else f"{got.tilt:.4f}")
        effect, share = "--", "--"
        pinned = [analysis.shift_of(model, a) for a in MATCHED_ARMS]
        pinned_values = [got.tilt for got in pinned if got is not None]
        if native is not None and pinned_values:
            matched = statistics.median(pinned_values)
            effect = f"{native.tilt / matched:.4f}"
            got_share = config_share(native.tilt, matched)
            share = "--" if got_share is None else f"{got_share:.1%}"
        lines.append(f"| {model} | {columns[0]} | {columns[1]} | {columns[2]} | "
                     f"{effect} | {share} |")
    lines += ["",
              "`config effect` is tilt(native) / median(tilt of the two matched "
              "arms): what",
              "letting each dtype pick its own config is worth, at fixed dtype "
              "physics.",
              "`config share` is (tilt(native) - tilt(matched)) / "
              "(tilt(native) - 1): the",
              "fraction of the confounded arm's EXCESS that pinning the config "
              "removes. It is",
              "undefined, and printed as --, when the confounded arm shows no "
              "excess to split."]
    return "\n".join(lines)


def render_crossings(analysis: Analysis, dtypes: list[str]) -> str:
    """Crossings as corroboration, matched by upcrossing index, never pooled.

    Kept out of every gate on purpose. `all_crossings_from_points` documents 8 of
    16 canonical cells crossing 0.5 more than once, and at a matched forced tile
    the two dtypes' staircases step at identical token counts by construction --
    so a crossing ratio here is pinned near 1.000 by the grid and would be a gate
    that cannot fail. It is printed because it is the instrument the published
    1.149 was read with, and a reader is entitled to see what it says.
    """
    lines = ["## Corroboration: the crossing detector, matched by upcrossing index",
             "",
             "| model | arm | bf16 upcrossings | fp8 upcrossings | per-index ratio |",
             "|---|---|---|---|---|"]
    for model in analysis.models:
        for arm in ARMS:
            bf16 = analysis.crossings.get((model, arm, BF16))
            fp8 = analysis.crossings.get((model, arm, FP8)) if FP8 in dtypes else None
            if bf16 is None or fp8 is None:
                lines.append(f"| {model} | {arm} | "
                             f"{'--' if bf16 is None else bf16} | "
                             f"{'--' if fp8 is None else fp8} | not both measured |")
                continue
            if len(bf16) != len(fp8):
                lines.append(
                    f"| {model} | {arm} | {[f'{v:.0f}' for v in bf16]} | "
                    f"{[f'{v:.0f}' for v in fp8]} | UNPAIRABLE: "
                    f"{len(bf16)} vs {len(fp8)} upcrossings |")
                continue
            ratios = [f"{b / a:.3f}" for a, b in zip(bf16, fp8, strict=True) if a > 0]
            lines.append(f"| {model} | {arm} | {[f'{v:.0f}' for v in bf16]} | "
                         f"{[f'{v:.0f}' for v in fp8]} | {', '.join(ratios) or '--'} |")
    return "\n".join(lines)


def render_headline(analysis: Analysis, gates: list[Gate]) -> str:
    """One paragraph, with both sides named and the validity state first."""
    failed = [g.name for g in gates if g.kind == "VALIDITY" and g.verdict == FAIL]
    if failed:
        return ("## Headline\n\nNONE. Validity gates " + ", ".join(failed)
                + " failed, so nothing on this page is a measurement of what it "
                  "claims to measure.")
    matched = [got.tilt for (_, arm), got in analysis.shifts.items()
               if arm in MATCHED_ARMS]
    native = [got.tilt for (_, arm), got in analysis.shifts.items()
              if arm == NATIVE_ARM]
    if not matched:
        return ("## Headline\n\nNothing was measured, so there is no headline. "
                "That is not a null result.")
    pure = statistics.median(matched)
    lines = ["## Headline", ""]
    if native:
        confounded = statistics.median(native)
        lines.append(
            f"Confounded (each dtype on its own config, which is what a "
            f"deployment runs): tilt {confounded:.3f} -- "
            f"{shift_sentence(confounded)}.")
        share = config_share(confounded, pure)
        if share is not None:
            lines.append(
                f"Pinning the config to one value for both formats removes "
                f"{share:.0%} of that excess.")
    lines += [
        f"At a MATCHED config the tilt is {pure:.3f} -- {shift_sentence(pure)}.",
        "",
        "TILT is the median fp8/bf16 time ratio over the lowest third of the "
        "token grid",
        "divided by the median over the highest third. Where the sweep straddles "
        "the",
        "transition that IS the crossing ratio; the branch-based shift beside it "
        "in the",
        "table above says whether it did.",
        "",
        f"The published, confounded figure this decomposes is "
        f"{PUBLISHED_SHIFT:.3f} +/- {PUBLISHED_SHIFT_SD:.3f}",
        "(docs/FINDINGS.md, eight measurements, two kernels). It is QUOTED, not "
        "re-measured here.",
    ]
    return "\n".join(lines)


def render_report(header: str, analysis: Analysis, gates: list[Gate],
                  dtypes: list[str], stopped: str = "") -> str:
    """The exact text written to report.md, assembled in one testable place."""
    body = "\n\n".join([
        header,
        render_shifts(analysis),
        render_ratio_curves(analysis),
        render_decomposition(analysis),
        render_crossings(analysis, dtypes),
        render_headline(analysis, gates),
        "## Gates\n\n```\n" + render_gate_summary(gates) + "\n```",
    ])
    if stopped:
        body += f"\n\nPARTIAL RUN: {stopped}."
    if analysis.failed_arms:
        body += ("\n\n## Arms that produced no timing\n\n"
                 + "\n".join(f"- {line}" for line in analysis.failed_arms))
    return body


# --------------------------------------------------------------------------
# Cost, so --dry-run prices the run before it is paid for.
# --------------------------------------------------------------------------

def estimated_seconds(cells: list[Cell], dtypes: list[str], ceilings: Ceilings,
                      reps: int, iters: int, warmup: int) -> float:
    """GPU seconds at the model's own timings, compiles and allocation excluded.

    Excluded because they do not scale with the grid and they dominate a short
    run: every distinct forced config is one Triton specialisation, and the count
    of those is printed beside this number rather than folded into it.
    """
    total_ms = 0.0
    for cell in cells:
        cfg = MODEL_CONFIGS[cell.model]
        for arm in ARMS:
            if arm_is_redundant(cell, arm):
                continue
            for dtype in dtypes:
                config = arm_config(cell, arm) or cell.configs[dtype]
                total_ms += predicted_ms(cfg, cell.num_tokens, config, dtype,
                                         ceilings=ceilings, act_bytes=2) * (
                    reps * (warmup + iters) + warmup + 1)
    return total_ms / 1e3


def distinct_compiles(cells: list[Cell], dtypes: list[str]) -> int:
    """How many Triton specialisations the run will ask for.

    One per (dtype, config) pair actually launched. A specialisation is a compile
    of a few seconds and they dominate a grid this cheap, so the estimate above
    would mislead without this beside it.
    """
    seen = set()
    for cell in cells:
        for arm in ARMS:
            if arm_is_redundant(cell, arm):
                continue
            for dtype in dtypes:
                config = arm_config(cell, arm) or cell.configs[dtype]
                seen.add((dtype, tuple(sorted(config.items()))))
    return len(seen)


# --------------------------------------------------------------------------
# Environment.
# --------------------------------------------------------------------------

def detect_environment() -> dict:
    """What this machine can run, and the name of whatever is missing.

    Returns rather than raises: "no GPU" is a supported mode here. The
    derivation, the predictions and the cost are arithmetic over vLLM's shipped
    configs and this repo's calibration, and they are worth printing on a laptop
    the day before the pod goes up.
    """
    env = {"torch": False, "cuda": False, "vllm": False, "gpu_name": None,
           "torch_version": "", "vllm_version": "", "missing": []}
    try:
        import torch
        env["torch"] = True
        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = True
            env["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            env["missing"].append("no CUDA device (torch.cuda.is_available() is False)")
    except ImportError as exc:
        env["missing"].append(f"torch is not importable: {exc}")
    try:
        import vllm
        env["vllm"] = True
        env["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except ImportError as exc:
        env["missing"].append(f"vllm is not importable: {exc}")
    return env


def version_warning(env: dict) -> str:
    """Say so loudly when the installed vLLM is not the one derived against."""
    installed = env.get("vllm_version") or ""
    if not installed or installed.lstrip("v") == VLLM_TAG.lstrip("v"):
        return ""
    return (f"WARNING: vLLM {installed} is installed but every config on this "
            f"page is derived from {VLLM_TAG}. V4 checks the derivation against "
            f"what actually loaded, so read a V4 failure as a version difference "
            f"first.")


def calibration_warning(ceilings: Ceilings, gpu_name: str | None) -> str:
    """Refuse-by-warning when the calibration describes a different card.

    Scoring this run against another machine's ceilings produces predictions that
    look entirely plausible and are wrong by the ratio of two parts. Printed
    rather than raised because `--dry-run` on a laptop has no device to match
    against and is a supported mode.
    """
    if not gpu_name:
        return ""
    if device_matches(Hardware(ceilings.name, ceilings.bandwidth_bytes_s,
                               ceilings.peak_flops, ceilings.source), gpu_name):
        return ""
    return (f"WARNING: {ceilings.path} was measured on {ceilings.name!r} but this "
            f"machine reports {gpu_name!r}. Every predicted number on this page "
            f"is against the wrong card. Run scripts/calibrate_hardware.py, or "
            f"pass --calibration for this device.")


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma list. The two defaults are the only shapes with "
                         "a tuned H200 file in BOTH dtypes, which is where the "
                         "tuned half of the confound lives. deepseek-v2-lite and "
                         "deepseek-v3 take the fallback ladder in both dtypes and "
                         "are the negative control: their BLOCK_SIZE_M is "
                         "identical across dtypes at every M by construction")
    ap.add_argument("--tokens", default=",".join(str(t) for t in DEFAULT_TOKENS),
                    help="every entry must be a multiple of E/gcd(E,k) for each "
                         "model, or balanced routing cannot realise it")
    ap.add_argument("--dtypes", default=",".join(DTYPES),
                    help="both by default. 'bf16' alone is the LARGEST "
                         "DEFENSIBLE SUBSET when a card has no fp8 units: it "
                         "prices the config effect at fixed dtype, which is the "
                         "half of the decomposition that does not need fp8, and "
                         "every dtype gate then reads UNKNOWN rather than PASS")
    ap.add_argument("--routing", default="uniform",
                    help="recorded on every row. The histogram is REALISED "
                         "exactly balanced whatever this says, because both "
                         "sides of every ratio must be the same cell; this names "
                         "the regime for the CSV and for `routing_domain`")
    ap.add_argument("--reps", type=int, default=3,
                    help="round-robin repeats. Arms and dtypes are interleaved "
                         "inside each, so a thermal excursion lands on both "
                         "sides of every ratio")
    ap.add_argument("--iters", type=int, default=15,
                    help="timed calls per (arm, dtype) per repeat")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-name", default=None,
                    help="override the device name used for the CONFIG LOOKUP. "
                         "Off a GPU this is what the plan is derived for")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION,
                    help="name of the hardware yaml under moe/bench/hardware/ "
                         "that both ceilings are read from")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"defaults to {default_out_dir()}/dtype_tile_confound")
    ap.add_argument("--run-id", default=None,
                    help="defaults to a hash of the whole plan, so re-running "
                         "the same command RESUMES rather than starting over")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and overwrite any rows already on disk")
    ap.add_argument("--dry-run", "--plan-only", dest="dry_run",
                    action="store_true",
                    help="print the derivation, the predictions and the cost, "
                         "then stop. Needs no GPU and no vLLM")
    ap.add_argument("--self-test", type=float, default=None, metavar="FP8_RATIO",
                    help="generate every cell from the model at this fp8/bf16 "
                         "FLOP ratio and run the whole analysis off GPU. 2.033 "
                         "is what this card measures; 2.400 is the world in "
                         "which the published 1.15 is pure dtype")
    ap.add_argument("--self-test-noise", type=float, default=0.0,
                    help="lognormal sigma applied to every synthetic timing")
    ap.add_argument("--self-test-alpha", type=float, default=None,
                    help="override the measured swizzle alpha in the synthetic "
                         "world. Needed to exercise C3 at all: at the measured "
                         "0.68-0.84 the compute term never binds, so the planted "
                         "FLOP ratio changes nothing and two different worlds "
                         "generate identical reports. 0.2 lifts the ceiling "
                         "2*BLOCK_M/(alpha*b) clear of the ridge")
    ap.add_argument("--self-test-fp8-activations", action="store_true",
                    help="generate the synthetic fp8 cells with QUANTISED "
                         "activations, the low end of the predicted band")
    ap.add_argument("--self-test-overhead-ms", type=float, default=0.0,
                    help="fixed per-call cost planted on every synthetic cell. "
                         "FINDINGS measures a real one in the fp8 path, and a "
                         "fixed cost inflates rm and rc together, so this checks "
                         "the gates survive it")
    ap.add_argument("--fail-on-claim", action="store_true",
                    help="exit non-zero when a CLAIM gate fails. Off by default "
                         "because a falsified prediction is a successful run; a "
                         "VALIDITY failure exits non-zero either way")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this long and report what exists")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    tokens = sorted({int(t) for t in args.tokens.split(",") if t})
    dtypes = [d for d in args.dtypes.split(",") if d]
    unknown = [d for d in dtypes if d not in DTYPES]
    if unknown:
        raise SystemExit(f"--dtypes accepts only {DTYPES}; got {unknown}. e5m2 "
                         "under an e4m3 flag would run and compute a different "
                         "layer, so it is refused rather than mapped.")
    if BF16 not in dtypes:
        raise SystemExit("--dtypes must include bf16: it is the denominator of "
                         "every ratio on the page.")

    env = detect_environment()
    lookup_gpu = args.gpu_name or env["gpu_name"] or "NVIDIA H200"
    try:
        ceilings = load_ceilings(args.calibration)
    except (FileNotFoundError, ValueError, KeyError, UnverifiedHardware) as exc:
        raise SystemExit(
            f"could not read ceilings from moe/bench/hardware/"
            f"{args.calibration}.yaml: {exc}\nEvery prediction on this page is "
            f"computed from a measured bandwidth and TWO measured peaks, and "
            f"there is no default to fall back to.") from exc

    # BEFORE anything reads a peak. Every prediction, the cost estimate and the
    # activation band all call `ceilings.peak(dtype)`, which refuses rather than
    # substituting a datasheet 2x -- and a typed refusal that arrives as a
    # traceback in the middle of a printed plan is not a usable refusal. Asked
    # here, once, with the fallback named. The A100 is the live case: its
    # calibration has no fp8 entry because the card has no fp8 tensor cores.
    missing_peaks = [d for d in dtypes if not ceilings.peak_flops.get(d)]
    # BF16 is always needed: it is the denominator of every ratio and of the
    # regime classification, so a calibration without it is refused whatever
    # --dtypes says.
    if BF16 not in missing_peaks and not ceilings.peak_flops.get(BF16):
        missing_peaks.append(BF16)
    if missing_peaks:
        print("\n".join([
            "REFUSED. Nothing was planned and nothing was measured.",
            f"  CalibrationIncomplete: {ceilings.path} carries no measured peak "
            f"for {missing_peaks}.",
            f"  It measured {sorted(ceilings.peak_flops)} on "
            f"{ceilings.measured_on}.",
            "  Substituting the datasheet's 2x for a missing fp8 ceiling is the "
            "exact defect",
            "  docs/FINDINGS.md flags on the published fp8 arm, so there is no "
            "fallback value.",
            "  On a card WITH fp8 units: run scripts/calibrate_hardware.py, "
            "which writes both",
            "  peaks in one session. On a card WITHOUT them (the A100 has none): "
            "run",
            "  `--dtypes bf16`, which prices the CONFIG effect at fixed dtype "
            "and refuses",
            "  every dtype number."]))
        return EXIT_NOT_MEASURED

    cells, notes = plan_cells(models, tokens, lookup_gpu)
    if not cells:
        print("NOT A RESULT: no cell could be planned.")
        for note in notes:
            print(f"  {note}")
        return EXIT_NOT_MEASURED

    payload = {"models": models, "tokens": tokens, "dtypes": dtypes,
               "arms": list(ARMS), "reps": args.reps, "iters": args.iters,
               "warmup": args.warmup, "seed": args.seed,
               "routing": args.routing, "lookup_gpu": lookup_gpu,
               "calibration": args.calibration, "vllm_tag": VLLM_TAG,
               # EVERY self-test knob, not just the headline one. Two synthetic
               # worlds that derived the same id would resume each other's
               # directory and print the first's numbers under the second's
               # label, which is the resume failure this project has already had.
               "self_test": args.self_test,
               "self_test_alpha": args.self_test_alpha,
               "self_test_noise": args.self_test_noise,
               "self_test_overhead_ms": args.self_test_overhead_ms,
               "self_test_fp8_activations": args.self_test_fp8_activations}
    run_id = args.run_id or plan_run_id(payload)
    out_dir = (args.out_dir or (default_out_dir() / "dtype_tile_confound")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, report_path = out_dir / "timings.csv", out_dir / "report.md"

    compiles = distinct_compiles(cells, dtypes)
    seconds = estimated_seconds(cells, dtypes, ceilings, args.reps, args.iters,
                                args.warmup)
    header_lines = [
        "# Is the fp8/bf16 crossing shift the DTYPE or the TILE?",
        "",
        f"run id {run_id}",
        f"config lookup device `{lookup_gpu}`   dtypes {','.join(dtypes)}   "
        f"routing {args.routing} (histogram realised exactly balanced)   "
        f"seed {args.seed}",
        f"{args.reps} repeats x {args.iters} timed calls per (arm, dtype), "
        f"round-robin with dtype innermost, {args.warmup} warmup each",
        "",
        f"EVERYTHING IS SAVED TO  {out_dir}",
        f"  git     {gitignore_note(out_dir)}",
        f"  rows    {csv_path}",
        f"  report  {report_path}",
        f"  plan    {out_dir / 'plan.json'}",
        "Re-run the same command to resume; completed (cell, arm, dtype) "
        "triples are skipped.",
        "",
        f"COST  {len(cells)} cells x {len(ARMS)} arms x {len(dtypes)} dtypes; "
        f"{compiles} distinct Triton specialisations;",
        f"      {seconds:.0f} s of timed kernel at the model's own timings, "
        f"excluding compiles and allocation.",
        "",
        SIGN_BANNER,
        "",
        render_derivation(cells, notes, models),
        "",
        render_predictions(cells, models, ceilings, dtypes),
    ]
    for warning in (version_warning(env), calibration_warning(ceilings, env["gpu_name"])):
        if warning:
            header_lines += ["", warning]
    header = "\n".join(header_lines)
    print(header)

    (out_dir / "plan.json").write_text(json.dumps({
        "run_id": run_id, **payload,
        "ceilings": {"path": ceilings.path, "measured_on": ceilings.measured_on,
                     "bandwidth_bytes_s": ceilings.bandwidth_bytes_s,
                     "peak_flops": ceilings.peak_flops},
        "estimated_timed_seconds": seconds, "distinct_compiles": compiles,
        "cells": [{"model": c.model, "num_tokens": c.num_tokens,
                   "rows_per_expert": c.rows,
                   "config_bf16": c.configs[BF16], "config_fp8": c.configs[FP8],
                   "block_m_agrees": c.block_m_agrees,
                   "differing_keys": list(c.differing_keys)} for c in cells],
        "dropped": notes}, indent=2))

    if args.dry_run:
        # C1 AND C2 ARE RENDERED, NOT ASSERTED TO HAVE BEEN DECIDED. This branch
        # used to return here having printed the raw inputs -- the DIFFERS/SAME
        # table and "BLOCK_SIZE_M agrees in 4/4 bracketing cells" -- and a
        # closing banner claiming "C1 and C2 are DECIDED here", with no PASS or
        # FAIL anywhere in 110 lines of output. Every gate in this project is a
        # number against a threshold printed as PASS or FAIL; a script that says
        # three times that a gate has been settled and renders no verdict is the
        # shape a reader trusts without checking. C1 is the PREMISE the whole
        # pod run rests on -- if it FAILED there would be no run worth buying --
        # and the free half of the experiment is exactly where that has to be
        # visible, before the box is rented.
        #
        # The verdicts come from the SHIPPED gate builder over an analysis with
        # no measurements in it, not from a second implementation here: both
        # gates read `crossing_bracket_cells`, which is derived from the config
        # lookup and needs no timing, and every other gate on that analysis
        # reads UNKNOWN because nothing was measured. A duplicate evaluation
        # would agree with the real one until it did not.
        derived = build_gates(analyse(cells, {}, ceilings, dtypes), ceilings,
                              dtypes, fp8_note="", synthetic=True)
        decided = [g for g in derived
                   if g.name.startswith(("C1", "C2")) and g.verdict != UNKNOWN]
        undecidable = [g.name for g in derived if g.verdict == UNKNOWN]
        print("\n".join(["", "=" * 72,
                         "DECIDED OFF GPU. These two are settled by vLLM's own "
                         "shipped config tree,",
                         "which `tile_resolve` reproduces exactly, so a pod "
                         "would not change them:", ""]))
        for g in decided:
            print(g.render())
        if not decided:
            # NON-VACUITY. No bracketing cell means no cell within 2x of a
            # published crossing, so C1 and C2 examined nothing and the claim
            # that they were decided here would be false.
            print("  NEITHER C1 NOR C2 COULD BE DECIDED: no planned cell falls "
                  f"within {CROSSING_BRACKET}x of any model's published bf16 "
                  "crossing, so both gates read UNKNOWN and this grid cannot "
                  "settle the premise the pod run rests on. Widen --tokens or "
                  "pick models with a published crossing.")
        print("\n".join(["", "=" * 72,
                         "NOT A RESULT. Nothing was measured.",
                         "  reason: --dry-run was given",
                         "  Everything above is arithmetic over vLLM's shipped "
                         "config tree and this",
                         "  repo's calibration. The gates NOT listed above "
                         f"({', '.join(undecidable)})",
                         "  read UNKNOWN and need the box; UNKNOWN is never "
                         "printed as a pass.",
                         f"  The plan was written to {out_dir / 'plan.json'}.",
                         "=" * 72]))
        return EXIT_NOT_MEASURED

    fp8_note = ""
    stopped = ""
    meta = {"run_id": run_id, "gpu_name": env["gpu_name"] or "synthetic",
            "torch_version": env["torch_version"],
            "vllm_version": env["vllm_version"], "routing": args.routing,
            "seed": args.seed}

    if args.self_test is not None:
        act = 1 if args.self_test_fp8_activations else 2
        results = synthetic_results(
            cells, dtypes, ceilings, fp8_flop_ratio=args.self_test,
            act_bytes_fp8=act, overhead_ms=args.self_test_overhead_ms,
            noise=args.self_test_noise, seed=args.seed,
            alpha=args.self_test_alpha)
        generated = sum(1 for per in results.values() for r in per.values()
                        if r.ms_median)
        fp8_note = (f"SYNTHETIC: fp8 peak planted at {args.self_test}x bf16, "
                    f"activations at {act} bytes, alpha "
                    + ("from the measured swizzle curve"
                       if args.self_test_alpha is None
                       else f"planted at {args.self_test_alpha}")
                    + f"; {generated} timings generated")
        print("\n".join([
            "", "=" * 72,
            f"SELF TEST: every cell GENERATED from the model at an fp8/bf16 FLOP "
            f"ratio of {args.self_test}.",
            "Nothing here was measured. The gates below are run against a world "
            "this script",
            "constructed, which tests the gates and not the hardware.",
            f"  {fp8_note}",
            "  NON-VACUITY: a self test that generated nothing would report zero "
            "failures too.",
            "  V1 and V2 read UNKNOWN here by construction: no kernel ran, so "
            "there is no",
            "  weight dtype and no oracle comparison to check.",
            "  C4 is NOT exercised: the surviving confound runs through "
            "BLOCK_SIZE_K, which",
            "  `predicted_ms` does not read, so the synthetic config effect is "
            "~1.000 and the",
            "  share it apportions is ~0. Only the box can settle C4.",
            "=" * 72]))
    else:
        if not (env["cuda"] and env["vllm"]):
            print("\n".join([
                "", "=" * 72,
                "NOT A RESULT. Nothing was measured.",
                "  reason: " + "; ".join(env["missing"]),
                "  --dry-run prints the derivation, the predictions and the cost.",
                "  --self-test 2.033 runs the entire analysis off GPU.",
                "=" * 72]))
            return EXIT_NOT_MEASURED
        try:
            fp8_note = preflight_fp8(dtypes)
        except Fp8PathUnavailable as exc:
            print("\n".join([
                "", "=" * 72,
                "REFUSED. Nothing was measured.",
                f"  Fp8PathUnavailable: {exc}",
                "  The largest defensible subset on this machine is "
                "`--dtypes bf16`, which",
                "  prices the CONFIG effect at fixed dtype. Every dtype gate then "
                "reads",
                "  UNKNOWN rather than PASS, which is the honest state.",
                "=" * 72]))
            return EXIT_NOT_MEASURED
        print(f"\n{fp8_note}")
        hooks = find_vllm_hooks()
        print(f"override hook: {hooks[2]}.override_config"
              + ("" if hooks[1] else "   (no get_config in that module)"))

        store = Store(csv_path, fresh=args.fresh)
        results = {}
        started = time.time()
        try:
            for model in models:
                model_cells = [c for c in cells if c.model == model]
                if not model_cells:
                    continue
                # A fully resumed model must not pay for its weights again.
                # deepseek-v3 draws 33 GB across both dtypes and quantises half
                # of it, all to answer questions already on disk.
                wanted = {(c.model, c.num_tokens, arm, dtype)
                          for c in model_cells for arm in ARMS
                          for dtype in dtypes}
                if not args.fresh and wanted <= set(store.done):
                    print(f"\n== {model}: every (cell, arm, dtype) is already in "
                          f"{csv_path.name}; not redrawing its weights ==",
                          flush=True)
                    for cell in model_cells:
                        restored = {(arm, dtype): store.restore(
                            (cell.model, cell.num_tokens, arm, dtype))
                            for arm in ARMS for dtype in dtypes}
                        results[cell.key] = {k: v for k, v in restored.items()
                                             if v is not None}
                    continue
                print(f"\n== {model}: building weights for "
                      f"{','.join(dtypes)} ==", flush=True)
                weights = build_model_inputs(model, dtypes, args.seed, "cuda")
                try:
                    for index, cell in enumerate(model_cells, 1):
                        if args.max_minutes and (
                                time.time() - started) / 60 >= args.max_minutes:
                            stopped = (f"stopped after {args.max_minutes} minutes "
                                       f"inside {model}")
                            break
                        print(f"  [{index}/{len(model_cells)}] T="
                              f"{cell.num_tokens:5d} r={cell.rows:8.1f}  bf16 "
                              f"{format_config(cell.configs[BF16])}  fp8 "
                              f"{format_config(cell.configs[FP8])}", flush=True)
                        results[cell.key] = measure_cell(
                            cell, weights, dtypes, args, store, meta, hooks,
                            check_correctness=(index == 1))
                finally:
                    # Explicit, and inside a finally: expert weights are the
                    # largest allocation in the run (33 GB at deepseek-v3 across
                    # both dtypes) and the next model allocates its own before
                    # Python would collect these.
                    import torch
                    weights.clear()
                    torch.cuda.empty_cache()
                if stopped:
                    break
        except KeyboardInterrupt:
            stopped = ("interrupted; every (cell, arm, dtype) finished before the "
                       "interrupt is on disk and the same command resumes")
        finally:
            store.close()

    analysis = analyse(cells, results, ceilings, dtypes)
    gates = build_gates(analysis, ceilings, dtypes, fp8_note,
                        synthetic=args.self_test is not None)
    report = render_report(header, analysis, gates, dtypes, stopped)
    report_path.write_text(report + "\n")
    (out_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id, "gpu_name": meta["gpu_name"],
        "vllm_version": meta["vllm_version"], "vllm_tag": VLLM_TAG,
        "self_test_fp8_ratio": args.self_test,
        "sign": "r = fp8_time / bf16_time; shift = rm/rc; shift > 1 means fp8 "
                "crosses LATER",
        "ceilings": {"path": ceilings.path, "measured_on": ceilings.measured_on,
                     "peak_flops": ceilings.peak_flops,
                     "bandwidth_bytes_s": ceilings.bandwidth_bytes_s},
        "shifts": {f"{model}|{arm}": {
            "tilt": got.tilt, "r_low": got.r_low, "r_high": got.r_high,
            "log_slope": got.log_slope, "n_paired": len(got.points),
            "low_tokens": list(got.low_tokens),
            "high_tokens": list(got.high_tokens),
            "memory_tokens": list(got.memory_tokens),
            "compute_tokens": list(got.compute_tokens),
            "rm": got.rm, "rc": got.rc, "branch_shift": got.shift,
            "branch_note": got.branch_note,
            "memory_slope_bf16": got.memory_slope_bf16,
            "compute_slope_bf16": got.compute_slope_bf16,
            "unpaired_tokens": list(got.unpaired),
            "ratios": [{"num_tokens": t, "bf16_ms": b, "fp8_ms": f, "r": f / b}
                       for t, b, f in got.points]}
            for (model, arm), got in analysis.shifts.items()},
        "predicted": {f"{m}|{a}|{label}": {"tilt": got.tilt,
                                           "branch_shift": got.shift}
                      for (m, a, label), got in analysis.predicted.items()},
        "refusals": analysis.refusals,
        "published_confounded_shift": PUBLISHED_SHIFT,
        "block_m_agreement_at_crossing": list(analysis.block_m_agreement),
        "timed_arms": analysis.timed_arms,
        "failed_arms": analysis.failed_arms,
        "partial": stopped,
        "gates": [{"name": g.name, "kind": g.kind, "verdict": g.verdict,
                   "rule": g.rule, "observed": g.observed} for g in gates],
    }, indent=2))

    print("\n" + render_shifts(analysis))
    print("\n" + render_ratio_curves(analysis))
    print("\n" + render_decomposition(analysis))
    print("\n" + render_crossings(analysis, dtypes))
    print("\n" + render_headline(analysis, gates))
    print("\n## Gates\n")
    print(render_gate_summary(gates))
    if stopped:
        print(f"\nPARTIAL RUN: {stopped}.")
    print(f"\nEVERYTHING IS SAVED TO {out_dir}")
    print(f"  rows    {csv_path}\n  report  {report_path}\n"
          f"  summary {out_dir / 'summary.json'}")
    if analysis.timed_arms == 0:
        print("\nNON-VACUITY: zero arms produced a timing, so every gate above "
              "examined nothing.")
        return EXIT_NOT_MEASURED
    if args.self_test is not None:
        print(f"\nNON-VACUITY: {analysis.timed_arms} SYNTHETIC timings were "
              f"generated over {len(cells)} planned cells and reduced by the "
              f"same code the pod run uses. No config was observed and no "
              f"kernel ran, which is why every validity gate but V0 reads "
              f"UNKNOWN.")
    else:
        print(f"\nNON-VACUITY: {analysis.timed_arms} arms produced a timing "
              f"over {len(cells)} planned cells; {analysis.derivation_checked} "
              f"native configs were read back out of vLLM and "
              f"{analysis.override_checked} forced ones were verified.")
    validity_failed = [g.name for g in gates
                       if g.kind == "VALIDITY" and g.verdict == FAIL]
    claims_failed = [g.name for g in gates
                     if g.kind == "CLAIM" and g.verdict == FAIL]
    if validity_failed:
        return EXIT_GATE_FAILED
    if claims_failed:
        print(f"CLAIM gates {claims_failed} failed. That is a RESULT, not a "
              f"broken run, so this exits 0; pass --fail-on-claim to make a "
              f"falsified claim non-zero.")
        return EXIT_GATE_FAILED if args.fail_on_claim else EXIT_OK
    return EXIT_OK


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
