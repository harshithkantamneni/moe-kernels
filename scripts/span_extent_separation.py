#!/usr/bin/env python3
"""How much of the 0.563 separation is span EXTENT and how much is KERNEL?

    python scripts/span_extent_separation.py --dry-run          # free, no GPU
    python scripts/span_extent_separation.py --self-test kernel --fail-on-world
    python scripts/span_extent_separation.py --self-test extent --fail-on-world
    python scripts/span_extent_separation.py                    # the pod run
    python scripts/span_extent_separation.py --models mixtral-8x7b --tokens 256,512

WHY THIS EXISTS. The study reports a separation of 0.563 between the five-stage
fused baselines and the one-stage grouped GEMM: a five-stage span crosses the
ridge at 56% of the token count a one-stage span does. `docs/FINDINGS.md:778-786`
calls the confound it carries "the largest single piece of work standing between
this result and a defensible claim", because the five-stage side is Triton
`fused_moe` with a tile that varies with batch and the one-stage side is CUTLASS
`grouped_mm` with a tile fixed at 64. The number is span extent CONVOLVED with
Triton-versus-CUTLASS and variable-versus-fixed tile, and nothing separates them.
`docs/FINDINGS.md:1286-1289` names the two routes out and says neither exists.

THE TWO ROUTES, AND WHICH ONE IS REACHABLE.

  ROUTE B, fuse the harness's own spans: NOT REACHABLE, and not for want of
  effort. `moe/kernels/` holds `.gitkeep` and `TEMPLATE.md`; the harness owns no
  fused kernel. Its own spans are the `ref_*` torch stages and
  `torch_grouped_mm_*`, one canonical stage each. A five-stage PIPELINE can be
  built out of them today and it would not answer this question: it is five
  launches with every intermediate materialised, and the published traffic table
  scores it at 12.43x its compulsory bytes against the fused span's 1.16x. That
  is a different implementation QUALITY, which is the very thing being
  controlled for. Writing the fused kernel is the study's own open work and it
  is the user's, not this script's.

  ROUTE A, run a fused implementation at a single-stage extent: REACHABLE, and
  it is reachable because vLLM's `fused_experts` is not one kernel. It is five
  launches:

      moe_align_block_size      the sort and the per-expert BLOCK_M padding
      invoke_fused_moe_kernel   the UP grouped GEMM, gather fused in      <- GEMM
      silu_and_mul              SwiGLU over the [gate | up] halves
      invoke_fused_moe_kernel   the DOWN grouped GEMM, combine weight applied <- GEMM
      moe_sum                   the top-k reduction back to [T, H]

  `invoke_fused_moe_kernel` IS the Triton `fused_moe_kernel`, with the same tile
  config the fused call resolved, restricted to one grouped GEMM. Handing it the
  alignment metadata built in an UNTIMED prologue gives the same kernel at a
  one-launch extent. That is the missing corner.

THE 2x2, THREE CORNERS OF WHICH EXIST.

                    | Triton fused_moe_kernel | CUTLASS grouped_mm
    five-launch     | `fused` (published)     | MISSING: needs route B
    one GEMM launch | `gemm_up`, `gemm_down`  | `cutlass_up`, `cutlass_down`
                    |   NEW, this script      |   (published)

  With three corners the published separation factors EXACTLY:

      separation = five_Triton / one_CUTLASS
                 = (five_Triton / one_Triton) * (one_Triton / one_CUTLASS)
                 =        EXTENT             *          KERNEL

  Exact because the middle term cancels, and every corner is measured in ONE
  session on ONE card at MATCHED cells, so no cross-session ruler enters. What is
  NOT measured is the interaction: whether extent would cost CUTLASS what it
  costs Triton is exactly the missing corner, and the report says so rather than
  assuming additivity in silence.

AND THE PREDICTED CROSSING CANCELS TOO. The published 0.563 is
`mean(measured/predicted)` over five-stage spans divided by the same over
one-stage spans, and `ridge.crossing_batch_full` takes a MODEL, not a span, so
per model the prediction is one number appearing in both numerator and
denominator. Every ratio below is therefore a ratio of two MEASURED crossings.
No ridge, no byte model and no calibration enters the separation or either
factor. That is why the 0.563 was ridge-independent in the first place, and it
is why this script can decompose it without inheriting the 9.9% ridge band.

THE PREDICTION, registered here with its mechanism and its arithmetic before any
measurement. THE SEPARATION IS THE KERNEL, NOT THE EXTENT.

  The extra stages are cheap. At mixtral's measured crossing (T=316, 79 rows per
  expert) the permute, activation and combine traffic is about 62 MB against 2.8
  GB of weights, 2.2%; at deepseek-v3's (T=3010) it is about 1.07 GB against
  22.5 GB, 4.8%. Three extra launches at a few microseconds each are flat in T,
  and a flat term FLATTENS a log-log slope, which pushes a crossing LATER rather
  than earlier. Nothing there moves a crossing by 1.8x.

  The padding is not cheap, and only one of the two kernels pays it.
  `moe_align_block_size` pads EVERY expert to a multiple of BLOCK_SIZE_M and the
  kernel COMPUTES the padded rows; `grouped_mm` takes ragged M and computes none.
  Padding inflates the compute side by `p` and leaves weight traffic alone, so
  the crossing moves to `1/p` of its predicted batch. At the published five-stage
  crossings, with the ladder's BLOCK_M:

      mixtral   T=316  r=79.0  -> 128 padded rows  p=1.62   1/p=0.62
      qwen2     T=787  r=98.4  -> 128              p=1.30   1/p=0.77
      v2-lite   T=931  r=87.3  -> 128              p=1.47   1/p=0.68
      dsv3      T=3010 r=94.1  -> 128              p=1.36   1/p=0.74
                                             mean  p=1.44   1/p=0.70

  against a measured five-stage mean of 0.63 and a one-stage mean of 1.13. The
  magnitude lands; the per-model ORDERING does not, and gate C3 is written to
  accept the first and not to claim the second.

  So the one-launch Triton arm, which pads identically, is predicted to cross
  where the five-launch one does, and the CUTLASS arm, which does not pad, is
  predicted to be the odd one out.

THE GATES. Split into VALIDITY, where a FAIL means no number on the page may be
quoted, and CLAIM, where a FAIL is a result:

  V0 assembly      the five launches recompute fused_experts' own output
  V1 same tile     the tile was READ OUT of the fused call, complete, unforced
  V2 reconstruct   the five launch times sum to the fused time
  V3 placebo       re-timing the SAME fused call moves nothing
  V4 non-vacuity   cells, samples and Triton compiles were all nonzero
  V5 comparable    this session's five-launch crossing is the published one
  V6 estimator     the up and down one-launch crossings agree with each other
  V7 one instrument  every timed arm carries `timing.TIMING_BASIS` and no other
                     value; a ratio between two rulers is not a ratio

  C1 extent is not the story     EXTENT in [0.85, 1.18]
  C2 the kernel is the story     KERNEL <= 0.75
  C3 mechanism is padding        the Triton/CUTLASS time ratio carries the
                                 padding factor above the crossing and not
                                 below, within 15% -- and reports UNKNOWN
                                 rather than PASS on a grid where the null
                                 sits inside its own band
  C4 the extra stages are cheap  non-GEMM share of the fused time < 20%
  C5 it is the published number  in-session separation within 33% of 0.563

V0 AND V1 READ THE REFUSALS, not just the numbers, and that is what lets them
FAIL. The prologue raises rather than timing an assembly that is not the fused
path, so a cell whose assembly is wrong records an error and no relative error
at all; a gate reading only the errors that got recorded as numbers would have
two reachable states, PASS and UNKNOWN. This repo has shipped a check in exactly
that shape before -- it compiled a probe kernel by piping source to stdin, which
`@jit` refuses, so it had never passed on any machine and the error was
suppressed.

WHEN THE CROSSING REFUSES, THE BOUND IS STILL DELIVERED. A crossing is read off
a slope and `docs/FINDINGS.md` records that the measured curve is a tile
STAIRCASE: taking the last upcrossing instead of the first moves the published
separation from 0.560 to 0.889. So every crossing quantity here is reported at
BOTH ends, and where no crossing can be located the script falls back to the
time-domain budget, which needs no estimator at all:

    EXTENT_time = fused / (gemm_up + gemm_down)     same kernel, two extents
    KERNEL_time = (gemm_up + gemm_down) / (cutlass_up + cutlass_down)

WHAT THE BOUND LICENSES AND WHAT IT DOES NOT. It licenses: for any comparison of
a five-stage MILLISECOND against a one-stage millisecond at the same cell, the
extra stages account for at most `1 - GEMM/total` of the gap. It does NOT
license any statement about the 0.563, because a crossing is a property of the
SLOPE and a term that is 3% of the level can still dominate the derivative. The
two are reported apart, and the bound never stands in for the ratio.

THE RESIDUAL ASYMMETRY, stated once because it bounds C2 rather than the run.
The Triton one-launch arm gathers its rows through `sorted_token_ids` and
computes padded rows; the CUTLASS one takes a dense pre-permuted matrix and
computes none. So KERNEL as measured CONTAINS the gather and the padding. That
is deliberate -- padding is the mechanism C3 names -- but it means KERNEL is not
"Triton's schedule against CUTLASS's schedule at identical work". The padding
factor is recorded per cell from the ACTUAL routing histogram so a reader can
see how much of KERNEL it could be.

WHAT SURVIVES TEARDOWN. Everything lands under `--out-dir`, defaulting to
`$MOE_RESULTS_DIR`, else `/workspace/results` when `/workspace` exists (the
RunPod network volume, which outlives the pod), else `<repo>/results`, in
`span_extent_separation/<run-id>/`. The absolute path and its gitignore status
are printed at the START as well as the end. Rows are flushed per arm, the run
id is a hash of the plan, and re-running the same command RESUMES: completed
(model, tokens, arm) triples are skipped.

OFF THE BOX. `--dry-run` prints the route judgement, the corner table, the cell
grid, the arms, the instrument's own plan, the MDE, the registered predictions
and the cost, and measures nothing. `--self-test WORLD` generates every arm time
from an explicit model in a named world and runs the WHOLE analysis on it, so
"these gates can tell a kernel world from an extent world" is checkable on a
laptop rather than asserted. Both are hermetic: neither reads the device, so a
replay is identical on every machine.

AND THE SELF TEST NOW ASSERTS SOMETHING. `WORLD_EXPECTATIONS` registers the
verdict every gate must produce in every world, `--fail-on-world` scores those S
gates and exits through the shared table on them, and a world that does not
reproduce its row is INVALID rather than a wall of text nobody diffs. Before
2026-09-02 the self test asserted nothing per world at all: on the grid the
session driver ran first, the `kernel` world -- the world where the claim is
TRUE -- gave C2 FAIL, C5 FAIL and C3 UNKNOWN, and produced the SAME claim-gate
row as the `neither` world, in which no mechanism exists. Nothing said so.

AND A NOTE FOR WHOEVER MERGES THIS. `--densify` being the default makes the
session driver's two span arms -- one bare, one `--densify` -- derive the SAME
run id and the same output directory, so the second restores every row the first
wrote, times nothing, and still lands DONE in the ledger. The data is right and
the ledger is wrong, the fix is deleting the bare arm in the driver, and that
file is not this one. What this file does instead is refuse to be quiet about
it: see `MeasurementTally`, the `REPLAY:` line and `arms_measured_here` /
`arms_restored` / `replay` in `summary.json`.

WHICH IS WHY `--densify` IS NOW THE DEFAULT, and the plan REFUSES on a grid that
cannot answer. On the published powers-of-two grid every expert holds a
power-of-two number of rows, BLOCK_M is a power of two, and the padding factor
is therefore exactly 1.00 at every point: the mechanism C2 names is invisible
where the grid looks. `c2_grid_power` plants the kernel world on the PLANNED
cells before anything is timed and refuses with "grid too sparse for C2; use
--densify" rather than spending 45 metered minutes to reach a FAIL that could
not have been anything else.

EXIT CODES AND THE ONE GREPPABLE LINE, both owned by `moe.bench.exit_codes`.
Every scored gate prints exactly one `RESULT: <KIND> <NAME> <VERDICT> <detail>`
line, the process exit code comes from `exit_codes.classify` over the same
gates, and every refusal exits REFUSED before anything is measured. This file
used to exit 3 for its `find_pieces` refusals and 2 for its missing-stack one,
against a driver that read 2 as REFUSED and everything else as RETRY, so a
genuine refusal was queued for another attempt. It also exited 1 for "a gate
FAILED", which collapsed a refuted claim and a broken instrument onto one
integer and scored UNKNOWN as a pass.

THE INSTRUMENT is `moe.bench.timing.time_kernel` under `TIMING_BASIS`, and the
`KernelTiming` columns are on every row. `--warmup-ms` is a duration of
delivered GPU load and `--target-ms` a trial length the instrument sizes `iters`
from; the old `--warmup 8` / `--iters 15` counted calls and there is no honest
conversion, so the flags were replaced rather than reinterpreted. See `time_arm`
for the one place this script's instrument is configured differently from the
one the roof was measured with, and why.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.crossing import all_crossings_from_points  # noqa: E402
from moe.bench.ridge import saturation_batch  # noqa: E402
from moe.bench.tile_resolve import config_dtype_selector, default_config  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

#: `moe.bench.timing` is imported LAZILY, everywhere, and this comment is the
#: reason. That module imports torch at module scope; this one documents
#: `--dry-run` and `--self-test` as things a laptop with no torch can run, and
#: `missing_gpu_stack` exists to say WHICH half of the stack is absent. A module
#: import here would turn both documented paths into an ImportError before
#: argparse ever ran. `exit_codes` and `provenance` import nothing heavier than
#: the standard library, so they are imported normally above.


def timing_basis() -> str | None:
    """`timing.TIMING_BASIS`, or None when torch is not importable here.

    None is not a default: it says the instrument could not be NAMED on this
    machine, which is exactly the laptop case where nothing was measured either.
    Every row a pod produces carries the string; a row without one came from a
    laptop or from before 2026-09-02, and V7 is the gate that reads it.
    """
    try:
        from moe.bench import timing
    except Exception:                                     # noqa: BLE001
        return None
    return timing.TIMING_BASIS


#: What `timing.iters_for` clamps to, for a machine that cannot import it to
#: ask: a laptop with no torch, where nothing is measured anyway. One place, and
#: it is a fallback rather than the source, for the reason `iters_clamp` gives.
ITERS_CLAMP_FALLBACK = (10, 2000)


def iters_clamp() -> tuple[int, int]:
    """The `[lo, hi]` `timing.iters_for` really clamps `iters` to, READ OFF IT.

    A COPY OF A BOUND DRIFTS, AND THIS ONE HAD. Until 2026-09-02 two docstrings
    here said the instrument clamps to [10, 10000] while `iters_for` clamps at
    `hi=2000`, and the gap is not cosmetic. At the default `--target-ms 200` the
    lower threshold is `target_ms / hi`: 0.1 ms with the real bound, 0.02 ms
    with the stated one. 261 of the 756 arms in the default plan are modelled
    under 0.1 ms, so a third of the grid underruns its budget and the plan said
    nothing about it, while `estimated_seconds` charged all of them the full
    `warmup_ms + trials * target_ms` and was therefore an OVER-estimate there.

    So the plan asks the shared function for its own defaults rather than
    restating them, and a future edit to `iters_for` moves this file's threshold
    with it instead of silently invalidating a sentence.
    """
    try:
        from moe.bench import timing
    except Exception:                                     # noqa: BLE001
        return ITERS_CLAMP_FALLBACK
    params = inspect.signature(timing.iters_for).parameters
    return int(params["lo"].default), int(params["hi"].default)


# --------------------------------------------------------------------------
# The numbers this script argues about, every one with its provenance, so a
# reader can check what was assumed without reading the code.
# --------------------------------------------------------------------------

#: The separation this script decomposes. `docs/STUDY.md:198-202` and
#: `docs/FINDINGS.md:760-772`: five-stage over one-stage, full byte model, and
#: identical at both ends of the ridge band because the prediction cancels.
PUBLISHED_SEPARATION = 0.563

#: The same quantity on the FIRST upcrossing, uniform routing only
#: (`docs/FINDINGS.md:766-770`), and on the LAST
#: (`moe/bench/crossing.all_crossings_from_points` docstring). Two ends of the
#: staircase, not two estimates of one number.
PUBLISHED_SEPARATION_FIRST = 0.5602
PUBLISHED_SEPARATION_LAST = 0.8889

#: Published five-stage measured/predicted, and one-stage. `docs/STUDY.md:150-155`.
PUBLISHED_FIVE_STAGE_RATIO = 0.63
PUBLISHED_ONE_STAGE_RATIO = 1.13

#: Published five-stage crossings in TOKENS, uniform routing, H200, re-scored
#: 2026-09-01 (`docs/FINDINGS.md:860-866`). Used ONLY by V5 to ask whether this
#: session is the same session the 0.563 came from. Nothing below is derived
#: from them.
PUBLISHED_FIVE_STAGE_CROSSING: dict[str, float] = {
    "mixtral-8x7b": 316.0,
    "qwen2-57b-a14b": 787.0,
    "deepseek-v2-lite": 931.0,
    "deepseek-v3": 3010.0,
}

#: THE COST MODEL'S CEILINGS ARE RESOLVED, NOT WRITTEN DOWN. Until 2026-09-03
#: this file carried `RIDGE_BAND = (160.3, 176.2)` as "the measured H200 ridge
#: band, both ends" and `BANDWIDTH_GBPS = 4374.5`, and `--ridge` defaulted to
#: the band's low end. That band is two H200 compute calibrations disagreeing
#: by 9.9% (160.3 is calibration md5 `4d84542b`), no card's ridge, withdrawn
#: from all 26 published reports on 2026-09-02. No ratio on this page touches
#: the ridge -- the predicted crossing cancels -- but the cost model divides by
#: `ridge x bandwidth`, and a plan priced against a withdrawn number is a plan
#: nobody can reproduce. `resolve_cost_ceilings` is where both now come from.
WITHDRAWN_RIDGE_BAND_WHY = (
    "160.3-176.2 FLOP/byte was two H200 compute calibrations 9.9% apart (md5 "
    "4d84542b at the low end), no card's ridge; withdrawn 2026-09-02")

#: The calibration a laptop plan is allowed to PRICE against when no device is
#: attached: the committed H200 file, read through `roofline` so the number is
#: the file's and not a literal here. It is a stated hypothesis and the
#: provenance block says so; a run with a real card never reads it.
HYPOTHESIS_CALIBRATION = "measured_nvidia_h200"


@dataclasses.dataclass(frozen=True)
class CostCeilings:
    """The ridge and bandwidth the cost model divides by, each with its source.

    `ridge_source` and `bandwidth_source` are prose, and both start with
    "COST MODEL ONLY" because that is the whole scope of these two numbers in
    this script: `render_instrument_plan` and the synthetic worlds price and
    generate against `ridge x bandwidth`, and no measured ratio consults
    either. `kind` is the short token: `cli`, `calibration` or `hypothesis`.
    """

    ridge: float
    bandwidth_gbps: float
    ridge_source: str
    bandwidth_source: str
    kind: str


def resolve_cost_ceilings(ridge_arg: float, bandwidth_arg: float, card: str,
                          dtype: str) -> CostCeilings:
    """The ceilings THIS run is entitled to price against, or a refusal.

    Order, and each step is a different kind of claim:

      1. `--ridge` / `--bandwidth-gbps`, each independently: the operator's
         assertion, in the run's own command line.
      2. THE ATTACHED DEVICE'S OWN CALIBRATION, through `roofline.load_measured`:
         `peak(dtype) / bandwidth` off the yaml `scripts/calibrate_hardware.py`
         wrote for this GPU. A card with no calibration REFUSES rather than
         being priced against another machine's file, and a calibration for a
         different device (`HardwareMismatch`) refuses the same way.
      3. NO DEVICE (`NO_CARD`: a laptop plan, a dry run, a self test), where
         nothing is measured and so nothing can be mislabelled: the committed
         `HYPOTHESIS_CALIBRATION`, labelled HYPOTHESIS in both source strings.

    The refusal is a string `SystemExit`, which `main` files as REFUSED (2),
    never CLAIM_FAIL: a plan that could not name its ceiling measured nothing.
    """
    from moe.bench import roofline

    if ridge_arg > 0.0 and bandwidth_arg > 0.0:
        return CostCeilings(
            ridge_arg, bandwidth_arg,
            "COST MODEL ONLY: --ridge, given on the command line",
            "COST MODEL ONLY: --bandwidth-gbps, given on the command line",
            "cli")

    if card == NO_CARD:
        try:
            hw = roofline.load_hardware(HYPOTHESIS_CALIBRATION)
        except (FileNotFoundError, ValueError, KeyError,
                roofline.UnverifiedHardware) as exc:
            raise SystemExit(
                f"REFUSED: no device is attached and the committed "
                f"{HYPOTHESIS_CALIBRATION}.yaml cannot be read ({exc}), so the "
                "cost model has no ceiling to price against. State one: "
                "--ridge <FLOP/byte> --bandwidth-gbps <GB/s>. Nothing measured.") \
                from exc
        label = (f"HYPOTHESIS, no device attached: {hw.name} from "
                 f"{HYPOTHESIS_CALIBRATION}.yaml, which belongs to no card in "
                 "this run")
        kind = "hypothesis"
    else:
        try:
            hw = roofline.load_measured(card)
        except roofline.HardwareMismatch as exc:
            raise SystemExit(f"REFUSED: {exc}\nNothing measured.") from exc
        if hw is None:
            raise SystemExit(
                f"REFUSED: no calibration for this device ({card}), so this run "
                "has no ceiling it is entitled to price against; the withdrawn "
                f"constant ({WITHDRAWN_RIDGE_BAND_WHY}) does not stand in.\n"
                "    Run:  python scripts/calibrate_hardware.py\n"
                "    or state the assertion yourself:  --ridge <FLOP/byte> "
                "--bandwidth-gbps <GB/s>\nNothing measured.")
        label = f"measured on this device: {hw.name}"
        kind = "calibration"

    try:
        ridge = ridge_arg if ridge_arg > 0.0 else hw.ridge_point(dtype)
    except ValueError as exc:
        raise SystemExit(
            f"REFUSED: {hw.name} has a measured bandwidth but no verified "
            f"{dtype} peak, so it cannot state a ridge: {exc}\nNothing "
            "measured.") from exc
    bandwidth = bandwidth_arg if bandwidth_arg > 0.0 else hw.bandwidth_bytes_s / 1e9
    ridge_source = ("COST MODEL ONLY: --ridge, given on the command line"
                    if ridge_arg > 0.0 else
                    f"COST MODEL ONLY: {label}; {dtype} ridge {ridge:.1f} FLOP/byte "
                    "= peak over bandwidth. No ratio on this page touches it; "
                    "the predicted crossing cancels out of every one")
    bandwidth_source = ("COST MODEL ONLY: --bandwidth-gbps, given on the command line"
                        if bandwidth_arg > 0.0 else
                        f"COST MODEL ONLY: {label}; {bandwidth:.1f} GB/s")
    return CostCeilings(ridge, bandwidth, ridge_source, bandwidth_source,
                        "cli" if ridge_arg > 0.0 and bandwidth_arg > 0.0 else kind)

#: The models the published separation was computed over.
DEFAULT_MODELS: tuple[str, ...] = ("mixtral-8x7b", "qwen2-57b-a14b",
                                   "deepseek-v2-lite", "deepseek-v3")

#: The published token grid, powers of two. MATCHED on purpose: V5 asks whether
#: this session reproduces the published crossing, and a denser grid would move
#: the crossing for a reason that has nothing to do with the session.
DEFAULT_TOKENS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                                   2048, 4096, 8192)

#: The card name a run id carries when no device is attached, so a laptop plan
#: and a replay never land in a real card's directory. `provenance.card_slug`
#: turns it into `nocard`; the same sentinel `scripts/block_m_crossing_sweep.py`
#: uses, spelled the same way, because two names for "no card" would sort into
#: two directories.
NO_CARD = "nocard"

#: What the run id's `case` knob records when nothing was generated. A sentinel
#: and not an empty string: `provenance.run_id` raises `UnresolvedKnob` on an
#: empty value, and rightly, because "" reads as "not filled in" and this is a
#: filled-in answer.
MEASURED_NOT_GENERATED = "measured"


def run_case(self_test: str | None, self_test_noise: float) -> str:
    """WHERE THIS RUN'S ARM TIMES CAME FROM, as one value of the run id.

    `measured-noise0`, or `kernel-noise0`, `extent-noise0.05` and so on. One
    knob rather than two, and named `case` rather than `self_test`, for a reason
    that is entirely about `ls`: `provenance.run_id` sorts the knobs by name and
    truncates the VISIBLE part at 96 characters, so anything sorting after
    `routing` is cut off this script's ids and survives only in the hash. A
    directory whose world can only be recovered by opening `summary.json` is
    what let three worlds overwrite one report in the first place.

    THE NOISE IS ALWAYS IN IT, including on a measured run where it generates
    nothing. That over-splits directories for a knob a pod run never passes, and
    it is deliberate: an id that is conditional on which knobs "matter" has a
    branch in it, and a knob quietly not mattering is the exact failure the key
    exists to prevent.
    """
    return f"{self_test or MEASURED_NOT_GENERATED}-noise{self_test_noise:g}"


# --------------------------------------------------------------------------
# Arms.
# --------------------------------------------------------------------------

#: What each arm times, in one place, printed in the plan. `fused` and
#: `fused_replica` are the placebo pair and straddle every other arm inside each
#: repeat, so a thermal excursion lands on all arms and the pair measures what
#: is left.
ARM_DESCRIPTIONS: dict[str, str] = {
    "fused": "vLLM fused_experts, five stages, one call -- the published span",
    "align": "moe_align_block_size alone: the sort and the BLOCK_M padding",
    "gemm_up": "invoke_fused_moe_kernel on w1 -- the SAME Triton kernel, one GEMM",
    "act": "silu_and_mul alone: SwiGLU over the [gate | up] halves",
    "gemm_down": "invoke_fused_moe_kernel on w2 -- the SAME Triton kernel, one GEMM",
    "sum": "moe_sum alone: the top-k reduction back to [T, H]",
    "cutlass_up": "torch grouped_mm on w1 -- CUTLASS, one GEMM, dense ragged M",
    "cutlass_down": "torch grouped_mm on w2 -- CUTLASS, one GEMM, dense ragged M",
    "fused_replica": "fused_experts again, at the far end of the repeat: the placebo",
}

ARM_ORDER: tuple[str, ...] = ("fused", "align", "gemm_up", "act", "gemm_down",
                              "sum", "cutlass_up", "cutlass_down",
                              "fused_replica")

#: The six keys a complete vLLM tile config carries. An observed config missing
#: one of them was not fully read back, and a row that recorded four of six
#: would look like a measurement of the tile.
TILE_KEYS: tuple[str, ...] = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                              "GROUP_SIZE_M", "num_warps", "num_stages")

#: The five launches `fused_experts` is made of. Their times must sum to the
#: fused time (V2) or the budget is not a budget.
LAUNCH_ARMS: tuple[str, ...] = ("align", "gemm_up", "act", "gemm_down", "sum")

#: The two of those five that are grouped GEMMs. Everything else is the extent.
GEMM_ARMS: tuple[str, ...] = ("gemm_up", "gemm_down")
NON_GEMM_ARMS: tuple[str, ...] = ("align", "act", "sum")
CUTLASS_ARMS: tuple[str, ...] = ("cutlass_up", "cutlass_down")

# --------------------------------------------------------------------------
# Gate thresholds. Registered, so they cannot be chosen after seeing a number.
# --------------------------------------------------------------------------

#: V0. Different tile shapes reduce the K loop in a different order, so bf16
#: outputs differ in the last bits legitimately. This catches an assembly that
#: computed a DIFFERENT LAYER, not one that rounded differently.
OUTPUT_REL_TOL = 2e-2

#: V2. The five launches are timed one at a time and the fused call runs them
#: back to back, so the sum carries five launch gaps the fused call does not and
#: loses whatever the fused path overlaps. A band rather than a point, and a
#: FAIL means the launch-by-launch budget is not a budget for this path.
RECONSTRUCTION_BAND = (0.85, 1.20)

#: V3. Same gate and same value as `scripts/tuned_vs_fallback.py`, for the same
#: reason: no ratio smaller than the noise floor means anything.
PLACEBO_BAND = 0.03

#: V4. Non-vacuity. A check that examined nothing reports zero failures too.
MIN_MEASURED_CELLS = 4
MIN_DECOMPOSED_MODELS = 2
MIN_SAMPLES_PER_ARM = 5

#: V5. How far this session's five-stage crossing may sit from the published one
#: before the two are not the same measurement. Wide, because the published
#: crossing came from a different profile with different iteration counts, and
#: because the staircase means a crossing can jump a whole tread.
COMPARABILITY_BAND = (0.60, 1.60)

#: V6. `docs/FINDINGS.md:786-791` records the one-stage up and down crossings
#: disagreeing by about 2.3x on two models, on what is the same arithmetic over
#: the same cells. A model whose two one-launch crossings disagree by more than
#: this is DROPPED from the decomposition by name rather than averaged in.
ESTIMATOR_AGREEMENT_MAX = 2.0

#: C1. "No material extent effect" as a band rather than a point. 1.18 and its
#: reciprocal 0.85 are one grid step's worth of crossing on a powers-of-two grid
#: at the slopes this study measures.
EXTENT_BAND = (0.85, 1.18)

#: C2. `ln(0.75)/ln(0.563) = 0.50`, so KERNEL <= 0.75 is exactly "the kernel
#: carries at least half the published separation in log terms".
KERNEL_MAX = 0.75

#: C3. How close the measured contrast must sit to the measured padding factor
#: for padding to be called the mechanism.
#:
#: CHOSEN FOR POWER, not from data, and the distinction matters. C3's null is
#: "the kernel gap is a constant that does not care about the regime", which
#: predicts a contrast of 1 against a prediction of `p`, i.e. a ratio of `1/p`.
#: The achievable `p` on the compute-bound side of any grid this study can
#: afford is 1.1 to 1.3, so `1/p` is 0.77 to 0.91 and a tolerance of 0.25 would
#: accept the null and the hypothesis alike -- which it did: all three
#: `--self-test` worlds passed C3 identically at 0.25. At 0.15 the worlds
#: separate, the kernel world recovering 0.999 and the extent world 0.80.
#: Nothing measured entered this choice; it was made against worlds whose truth
#: this script plants.
PADDING_TOLERANCE = 0.15

#: C4. The non-GEMM share of the fused time at the cells nearest each model's
#: crossing. A FAIL means the extra stages are expensive in TIME even though the
#: byte model says they are cheap in TRAFFIC, which would be worth knowing.
NON_GEMM_SHARE_MAX = 0.20

#: C5. How far this session's own separation may sit from the published 0.563
#: before the decomposition, however exact internally, is not a decomposition OF
#: the published claim.
SEPARATION_BAND = (0.75 * PUBLISHED_SEPARATION, 1.33 * PUBLISHED_SEPARATION)

#: Bootstrap for the two factors. Seeded, so two readers of one CSV agree.
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_BAND = 0.90

#: At or below this many models the percentile bootstrap of a median is
#: DEGENERATE and the interval is a restatement of the points rather than a
#: sampling distribution: resampling n=2 with replacement gives three possible
#: multisets and two possible medians, and n=3 gives four. The interval is still
#: printed, because "here are the two points" is information, but it is printed
#: with its scope and its n beside it so nobody reads it as an error bar.
#: `MIN_DECOMPOSED_MODELS` is 2, so the default grid can and does land here.
BOOTSTRAP_DEGENERATE_N = 3

#: THE NOISE ASSUMPTION every MDE on this page is derived from, and it is an
#: ASSUMPTION, not a measurement of this script's own arms. It is
#: `scripts/replicate_noise_floor.py:PRIOR_SD`, 0.0323 / sqrt(2) = 0.0228: the
#: s3-vs-s4 paired sd over 11 matched cells, divided by sqrt(2) to turn a
#: difference-of-two sd into a per-arm one. It CONFOUNDS num_stages with rerun
#: noise, so it is an upper bound on the per-arm replicate spread, which is what
#: makes it safe to size a design against and unsafe to quote as a floor.
#:
#: It is a spread on a per-arm log time, and the quantities gated here are
#: per-model log RATIOS of two crossings, each read off a curve of many cells.
#: Propagating it as if a ratio carried sqrt(2) times one arm's spread is the
#: crudest defensible step and it is stated rather than hidden; the honest
#: reading of the MDE line is "the design cannot see an effect smaller than
#: this", never "the design can see an effect larger than this".
PRIOR_ARM_SD_LOG = 0.0323 / math.sqrt(2.0)

#: Two-sided 5% at 80% power: `z(0.975) + z(0.80)`. The same convention
#: `scripts/replicate_noise_floor.py` states for every MDE it prints, so one
#: number means one thing across the study.
MDE_Z_SUM = 1.959964 + 0.841621

CSV_COLUMNS = (
    "run_id", "utc", "gpu_name", "torch_version", "triton_version",
    "vllm_version", "model", "num_experts", "top_k", "dtype", "routing", "seed",
    "num_tokens", "arm", "ms_median", "ms_mean", "ms_stdev", "ms_min",
    "ms_p90", "n_samples", "block_m", "block_n", "block_k", "group_m",
    "num_warps", "num_stages", "tile_config_source", "rows_total",
    "padded_rows", "padding_source", "active_experts", "rel_err_vs_fused",
    "triton_artifacts",
    # The instrument's own state, one column each, on EVERY row. Until
    # 2026-09-02 this script timed with a private copy of the retired
    # `time_call` loop and recorded none of this, so a reader could not tell a
    # row measured queue-deep from a row measured one synchronize per iteration,
    # nor a row timed while the card sat at 1425 MHz from one at 1980.
    "instrument", "warmup_ms", "iters", "trials", "sm_clock_load_mhz",
    "clock_level_ok", "clock_drift_ok", "l2_flush", "host_bound",
    "host_enqueue_ms", "clock_note", "host_note",
    "error",
)

#: `prov_*`, derived from the dataclass so the two cannot drift. Every row
#: carries the git sha, the card, the versions and the ruler that produced it;
#: an unattributable row is what the audit found in all 26 published reports.
PROVENANCE_COLUMNS: tuple[str, ...] = tuple(PV.Provenance().as_columns())

ALL_CSV_COLUMNS: tuple[str, ...] = CSV_COLUMNS + PROVENANCE_COLUMNS


# --------------------------------------------------------------------------
# Typed refusals. A quantity that cannot be measured raises with a name; it
# never returns 0.0 and never substitutes a plausible value.
# --------------------------------------------------------------------------

class SeparationRefusal(RuntimeError):
    """Base for everything this script refuses to answer."""


class PieceMissing(SeparationRefusal):
    """A launch of vLLM's fused path is not where this script looked for it.

    Raised rather than worked around. A torch stand-in for `silu_and_mul` would
    time a DIFFERENT kernel and the budget would then be a budget of something
    nobody runs, which is worse than no budget.
    """


class SignatureDrifted(SeparationRefusal):
    """A vLLM entry point wants a parameter this script has no value for.

    The failure this prevents: filling an unknown parameter with a plausible
    default (None, False, the first enum member) produces a call that runs,
    returns the right shape, and times a different schedule.
    """


class AssemblyMismatch(SeparationRefusal):
    """The five launches did not reproduce `fused_experts`' own output.

    Fatal to the whole script and not to one cell: if the assembly is not the
    fused path, then the one-launch arms are not the fused path's kernel and
    EXTENT is a comparison between two unrelated things.
    """


class ConfigUnobserved(SeparationRefusal):
    """The recorder did not see the tile the fused call resolved.

    Without it the one-launch arms would have to DERIVE a config, and a derived
    config that differs from the live one turns EXTENT into a tile comparison
    wearing a span-extent costume.
    """


# --------------------------------------------------------------------------
# The plan: cells, and the arithmetic every cell knows before it is timed.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (model, token count) comparison. Every arm runs on this one cell."""

    model: str
    num_tokens: int
    dtype: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.model, self.num_tokens)

    @property
    def cfg(self):
        return MODEL_CONFIGS[self.model]

    @property
    def rows_per_expert(self) -> float:
        """`T k / E`, the SATURATED ratio. True on average under sampled uniform
        routing and exactly true under a balanced realisation; the per-cell
        padding factor is computed from the real histogram instead, and says so
        through `padding_source`."""
        cfg = self.cfg
        return self.num_tokens * cfg.top_k / cfg.num_experts


#: Multiples of a model's published crossing that `--densify` adds, roughly
#: half a power of two apart. They bracket the crossing on both sides and put
#: three points in the 80-to-200 rows-per-expert window where the padding factor
#: is 1.3 to 1.8 -- which is the only place C3's mechanism is observable, and
#: which the powers-of-two grid steps straight over.
DENSIFY_MULTIPLIERS: tuple[float, ...] = (0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0)


def token_step(cfg) -> int:
    """Token step that keeps `T k / E` an exact integer.

    `R = T k / E`, so T must be a multiple of `E / gcd(E, k)`. It is 4 for
    mixtral and 32 for deepseek-v2-lite, and a densified point that ignored it
    would put the padding arithmetic on a fractional row count for one model
    only -- which is how a geometry constraint quietly excludes a model from a
    study while the table still renders.
    """
    return cfg.num_experts // math.gcd(cfg.num_experts, cfg.top_k)


def densified_tokens(model: str) -> list[int]:
    """Extra token counts around this model's published crossing.

    Anchored on `PUBLISHED_FIVE_STAGE_CROSSING`, which is a DOCUMENTED number
    and not a measurement of this session. That is legitimate here in a way it
    would not be in a result: it decides where to LOOK, and every crossing the
    run reports is still read off this session's own curve. A model with no
    published anchor gets no extra points rather than a guessed anchor.
    """
    cfg = MODEL_CONFIGS[model]
    anchor = PUBLISHED_FIVE_STAGE_CROSSING.get(model)
    if not anchor:
        return []
    step = token_step(cfg)
    out = set()
    for mult in DENSIFY_MULTIPLIERS:
        snapped = int(round(anchor * mult / step)) * step
        if snapped >= step:
            out.add(snapped)
    return sorted(out)


def plan_cells(models: list[str], tokens: list[int], dtype: str,
               densify: bool = False) -> tuple[list[Cell], list[str]]:
    """Every cell, plus a note per model that cannot be planned.

    A model missing from MODEL_CONFIGS is NAMED rather than skipped: the
    published separation is a mean over four models and a run that silently
    measured three would decompose a different number.
    """
    cells: list[Cell] = []
    notes: list[str] = []
    for model in models:
        if model not in MODEL_CONFIGS:
            notes.append(f"{model}: not in MODEL_CONFIGS, so it cannot be planned")
            continue
        grid = sorted(set(tokens) | (set(densified_tokens(model)) if densify
                                     else set()))
        if densify and not densified_tokens(model):
            notes.append(f"{model}: --densify given but there is no published "
                         "crossing to anchor extra points on, so it keeps the "
                         "base grid and C3 may have no power on it")
        cells.extend(Cell(model, tok, dtype) for tok in grid)
    return cells, notes


def ladder_config(tokens: int, num_experts: int, dtype: str) -> dict[str, int]:
    """The whole config vLLM's fallback ladder would choose for this cell.

    All six keys, never a subset. A config carrying only BLOCK_SIZE_M is what a
    row looks like when the recorder half-read it, and V1 exists to catch
    exactly that -- so the synthetic world must not manufacture the shape it is
    supposed to reject.
    """
    return default_config(max(1, tokens), num_experts,
                          config_dtype_selector(dtype))


def ladder_block_m(tokens: int, num_experts: int, dtype: str) -> int:
    """The BLOCK_SIZE_M vLLM's fallback ladder would choose for this cell.

    Used by the PLAN and by `--self-test`, never by a measurement: on the box
    the block size is OBSERVED out of the fused call. Two sources for one
    quantity is how a documentation constant ends up presented as a measurement,
    so the CSV carries which one produced each row.
    """
    return ladder_config(tokens, num_experts, dtype)["BLOCK_SIZE_M"]


def padded_rows_saturated(cfg, tokens: int, block_m: int) -> float:
    """Rows the kernel COMPUTES, on the saturated routing assumption.

    `moe_align_block_size` pads every expert to a multiple of BLOCK_SIZE_M and
    the kernel computes the padded rows, so this is `E ceil(r / BM) BM` and not
    `ceil(E r / BM) BM`. Getting that wrong understates the padding by a factor
    of E at decode, which is the regime the whole study lives in.
    """
    rows = tokens * cfg.top_k / cfg.num_experts
    return cfg.num_experts * math.ceil(rows / block_m) * block_m


def padding_factor(padded: float, rows_total: float) -> float | None:
    """Padded rows over real rows. None when there are no rows to pad.

    None rather than 1.0: "this cell has no rows" and "this cell has no padding"
    are different, and only the second is a measurement.
    """
    if rows_total <= 0:
        return None
    return padded / rows_total


def modelled_padding_at(cfg, tokens: float, block_m: int) -> float | None:
    """`p` at an arbitrary, possibly INTERPOLATED, token count.

    ARITHMETIC, not a measurement, and the report labels it so. It exists
    because C3's mechanism is a fixed point: the crossing is where padded
    compute meets weight traffic, so `p` has to be evaluated AT the crossing,
    and a crossing read off a log-spaced grid almost never lands on a grid
    point. mixtral's published crossing is T=316, which is 79 rows per expert
    and pads to 128 at BLOCK_M=64 for p=1.62; the nearest measured cells are
    T=256 (r=64, p=1.00 exactly) and T=512 (r=128, p=1.00 exactly), so reading
    `p` off either of them would report NO padding at a crossing the mechanism
    says exists entirely because of padding.

    ON THE BALANCED ASSUMPTION, and it is a LOWER BOUND. `r = T k / E` is the
    mean; under sampled uniform routing the per-expert counts scatter around it
    and `sum_e ceil(n_e / BM) BM` is at least `E ceil(r / BM) BM`. So the real
    padding is at least this and usually more, and the CSV carries the measured
    one from the histogram beside it.
    """
    if tokens <= 0:
        return None
    rows = tokens * cfg.top_k / cfg.num_experts
    if rows <= 0 or block_m <= 0:
        return None
    return math.ceil(rows / block_m) * block_m / rows


# --------------------------------------------------------------------------
# The cost model. Arithmetic only: no torch, no GPU. Every number it produces
# is labelled as the byte model's own prediction wherever it is printed.
# --------------------------------------------------------------------------

def weight_bytes(cfg, stage: str, b: int) -> float:
    """Compulsory weight bytes for one grouped GEMM over all experts."""
    if stage == "up_gemm":
        return cfg.num_experts * 2.0 * cfg.intermediate_size * cfg.hidden_size * b
    if stage == "down_gemm":
        return cfg.num_experts * cfg.intermediate_size * cfg.hidden_size * b
    raise KeyError(f"no weight bytes for stage {stage!r}")


def stage_traffic(cfg, rows_total: float, tokens: int, stage: str, b: int) -> float:
    """Activation bytes one stage moves. Weights are added by the caller."""
    h, f, k = cfg.hidden_size, cfg.intermediate_size, cfg.top_k
    if stage == "align":
        return (tokens * k * 4.0) * 2          # ids in, sorted ids out, int32
    if stage == "up_gemm":
        return rows_total * (h + 2.0 * f) * b
    if stage == "act":
        return rows_total * 3.0 * f * b        # read [gate|up], write one half
    if stage == "down_gemm":
        return rows_total * (f + h) * b
    if stage == "sum":
        return rows_total * h * b + tokens * h * b
    raise KeyError(f"no traffic model for stage {stage!r}")


def stage_flops(cfg, rows: float, stage: str) -> float:
    if stage == "up_gemm":
        return 2.0 * rows * cfg.hidden_size * 2.0 * cfg.intermediate_size
    if stage == "down_gemm":
        return 2.0 * rows * cfg.intermediate_size * cfg.hidden_size
    return 0.0


#: A kernel launch that does nothing still costs this. Used only by the cost
#: estimate and by `--self-test`; it is a round number, not a measurement, and
#: every place it appears says so.
LAUNCH_MS = 0.006


def modelled_ms(cfg, tokens: int, stage: str, *, b: int, ridge: float,
                bandwidth_gbps: float, compute_rows: float | None = None
                ) -> float:
    """`launch + max(traffic / BW, flops(compute_rows) / peak)` for one launch.

    `compute_rows` is the rows the kernel actually COMPUTES, which is the whole
    mechanism C3 names: `moe_align_block_size` pads every expert to a multiple
    of BLOCK_SIZE_M and the Triton kernel computes the padded rows, while
    `grouped_mm` takes ragged M and computes none. Defaults to the real row
    count, so a caller that does not say gets no padding rather than a padding
    it did not ask for.

    THIS IS THE MODEL UNDER TEST. Nothing that reads it may be read as evidence
    FOR it: its job is to price the run for `--dry-run` and to generate the
    worlds `--self-test` discriminates between.
    """
    rows_total = tokens * cfg.top_k
    bw = bandwidth_gbps * 1e9
    peak = ridge * bw
    traffic = stage_traffic(cfg, rows_total, tokens, stage, b)
    if stage in ("up_gemm", "down_gemm"):
        traffic += weight_bytes(cfg, stage, b)
        compute_s = stage_flops(
            cfg, rows_total if compute_rows is None else compute_rows,
            stage) / peak
    else:
        compute_s = 0.0
    return LAUNCH_MS + 1e3 * max(traffic / bw, compute_s)


def modelled_arm_ms(cell: Cell, arm: str, *, ridge: float,
                    bandwidth_gbps: float) -> float:
    """One arm's predicted milliseconds, from the same model for every arm."""
    cfg = cell.cfg
    b = dtype_bytes(cell.dtype)
    bm = ladder_block_m(cell.num_tokens, cfg.num_experts, cell.dtype)
    padded = padded_rows_saturated(cfg, cell.num_tokens, bm)
    kw = {"b": b, "ridge": ridge, "bandwidth_gbps": bandwidth_gbps}
    if arm in ("fused", "fused_replica"):
        return sum(modelled_ms(cfg, cell.num_tokens, s,
                               compute_rows=padded if s.endswith("gemm") else None,
                               **kw)
                   for s in ("align", "up_gemm", "act", "down_gemm", "sum"))
    if arm == "gemm_up":
        return modelled_ms(cfg, cell.num_tokens, "up_gemm", compute_rows=padded, **kw)
    if arm == "gemm_down":
        return modelled_ms(cfg, cell.num_tokens, "down_gemm", compute_rows=padded, **kw)
    if arm == "cutlass_up":
        return modelled_ms(cfg, cell.num_tokens, "up_gemm", **kw)
    if arm == "cutlass_down":
        return modelled_ms(cfg, cell.num_tokens, "down_gemm", **kw)
    return modelled_ms(cfg, cell.num_tokens, arm, **kw)


def estimated_seconds(cells: list[Cell], arms: list[str], *, reps: int,
                      target_ms: float, warmup_ms: float, trials: int) -> float:
    """GPU seconds at the byte model's own timings, compiles excluded.

    THE COST STOPS DEPENDING ON THE CELL'S OWN SPEED BETWEEN THE TWO CLAMPS,
    and that is a property of the instrument rather than of this arithmetic.
    `time_kernel` warms for `warmup_ms` of delivered load and then sizes `iters`
    so a trial lasts `target_ms`, so an arm whose modelled per-call time lands
    inside `[target_ms / hi, target_ms / lo]` costs
    `warmup_ms + trials * target_ms` per repeat whatever it is. The old estimate
    multiplied a modelled millisecond by a call count and therefore said a
    T=8192 deepseek-v3 cell cost a thousand times a T=1 mixtral one, which the
    instrument makes false.

    OUTSIDE THE CLAMPS THIS IS A BOUND, NOT THE COST, and the plan says which
    way. `iters_clamp` reads the real `[lo, hi]` off `iters_for` -- [10, 2000],
    not the [10, 10000] this docstring claimed until 2026-09-02. An arm slower
    than `target_ms / lo` cannot fit `lo` calls in the budget and OVERRUNS it,
    so this UNDER-estimates it. An arm faster than `target_ms / hi` is capped at
    `hi` calls and delivers less measured time than the budget, so this
    OVER-estimates it -- and at the default target that is 261 of the 756 arms
    in the default plan, a third of the grid, the fastest modelled at 0.006 ms.
    Neither is modelled here: this returns the design's own budget, both
    departures from it are counted by name in `render_instrument_plan`, and
    `--max-minutes` is what actually stops the run.
    """
    per_arm_ms = warmup_ms + trials * target_ms
    return len(cells) * len(arms) * reps * per_arm_ms / 1e3


# --------------------------------------------------------------------------
# Results, and the reduction. Kept free of torch so the whole analysis runs on
# a laptop against synthetic arms, which is the only way the SIGN of the answer
# gets a test before a pod exists.
# --------------------------------------------------------------------------

@dataclass
class ArmResult:
    """One timed arm of one cell, or the reason there is no timing."""

    model: str
    num_tokens: int
    arm: str
    ms_median: float | None = None
    ms_mean: float | None = None
    ms_stdev: float | None = None
    ms_min: float | None = None
    n_samples: int = 0
    config: dict[str, int] | None = None
    tile_config_source: str = ""
    rows_total: float = 0.0
    padded_rows: float = 0.0
    padding_source: str = ""
    active_experts: int = 0
    rel_err_vs_fused: float | None = None
    triton_artifacts: int = 0
    error: str = ""
    # ---- the instrument's own state, copied off `timing.KernelTiming` ----
    # Carried on the ROW and not in a run-level header, because a resumed run
    # mixes rows written by two invocations and only a per-row record can say
    # which ruler produced which number. `instrument` empty means "not
    # determined", the state a synthetic world and a pre-2026-09-02 CSV are
    # both in, and V7 refuses to score a run that mixes it with a real one.
    ms_p90: float | None = None
    instrument: str = ""
    warmup_ms: float | None = None
    iters: int | None = None
    trials: int | None = None
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    l2_flush: bool | None = None
    host_bound: bool | None = None
    host_enqueue_ms: float | None = None
    clock_note: str = ""
    host_note: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model, self.num_tokens, self.arm)

    def with_timing(self, t) -> ArmResult:
        """Copy the whole `timing.KernelTiming` record onto this row.

        One place, so a field added to the instrument is added to the CSV by
        editing `CSV_COLUMNS` and this function and nothing else. Takes the
        record rather than the numbers so a caller cannot pass the p50 of one
        measurement beside the clocks of another.
        """
        return dataclasses.replace(
            self, ms_p90=t.ms_p90, instrument=t.instrument, warmup_ms=t.warmup_ms,
            iters=t.iters, trials=t.trials, sm_clock_load_mhz=t.sm_clock_load_mhz,
            clock_level_ok=t.clock_level_ok, clock_drift_ok=t.clock_drift_ok,
            l2_flush=t.l2_flush, host_bound=t.host_bound,
            host_enqueue_ms=t.host_enqueue_ms, clock_note=t.clock_note,
            host_note=t.host_note)

    @property
    def timed(self) -> bool:
        return not self.error and bool(self.ms_median) and self.ms_median > 0.0

    def row(self, cell: Cell, meta: dict) -> dict:
        """One CSV row: the timing, the tile, the instrument and the provenance.

        `meta["provenance_columns"]` is `Provenance.as_columns()` for the whole
        run, written onto EVERY row. Per row and not once per file because a
        resumed CSV holds rows from two invocations at two commits, and a
        file-level header would attribute all of them to the last one.
        """
        cfg = cell.cfg
        conf = self.config or {}

        def flag(value: bool | None) -> str:
            # "" is a third state and means "not determined". Writing False for
            # it would turn "NVML was unavailable" into "the clock was wrong".
            return "" if value is None else str(bool(value))

        def num(value, spec: str) -> str:
            return "" if value is None else format(value, spec)

        return {
            **meta.get("provenance_columns", {}),
            "run_id": meta["run_id"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_name": meta["gpu_name"], "torch_version": meta["torch_version"],
            "triton_version": meta["triton_version"],
            "vllm_version": meta["vllm_version"], "model": self.model,
            "num_experts": cfg.num_experts, "top_k": cfg.top_k,
            "dtype": cell.dtype, "routing": meta["routing"], "seed": meta["seed"],
            "num_tokens": self.num_tokens, "arm": self.arm,
            "ms_median": "" if self.ms_median is None else f"{self.ms_median:.6f}",
            "ms_mean": "" if self.ms_mean is None else f"{self.ms_mean:.6f}",
            "ms_stdev": "" if self.ms_stdev is None else f"{self.ms_stdev:.6f}",
            "ms_min": "" if self.ms_min is None else f"{self.ms_min:.6f}",
            "n_samples": self.n_samples,
            "block_m": conf.get("BLOCK_SIZE_M", ""),
            "block_n": conf.get("BLOCK_SIZE_N", ""),
            "block_k": conf.get("BLOCK_SIZE_K", ""),
            "group_m": conf.get("GROUP_SIZE_M", ""),
            "num_warps": conf.get("num_warps", ""),
            "num_stages": conf.get("num_stages", ""),
            "tile_config_source": self.tile_config_source,
            "rows_total": f"{self.rows_total:.1f}",
            "padded_rows": f"{self.padded_rows:.1f}",
            "padding_source": self.padding_source,
            "active_experts": self.active_experts,
            "rel_err_vs_fused": ("" if self.rel_err_vs_fused is None
                                 else f"{self.rel_err_vs_fused:.3e}"),
            "triton_artifacts": self.triton_artifacts,
            "ms_p90": num(self.ms_p90, ".6f"),
            "instrument": self.instrument,
            "warmup_ms": num(self.warmup_ms, ".3f"),
            "iters": "" if self.iters is None else self.iters,
            "trials": "" if self.trials is None else self.trials,
            "sm_clock_load_mhz": num(self.sm_clock_load_mhz, ".0f"),
            "clock_level_ok": flag(self.clock_level_ok),
            "clock_drift_ok": flag(self.clock_drift_ok),
            "l2_flush": flag(self.l2_flush),
            "host_bound": flag(self.host_bound),
            "host_enqueue_ms": num(self.host_enqueue_ms, ".6f"),
            "clock_note": self.clock_note,
            "host_note": self.host_note,
            "error": self.error,
        }


Results = dict[tuple[str, int], dict[str, ArmResult]]

#: How `arm_ruler` spells a flush state that no row recorded: a synthetic world,
#: or a CSV from before the column existed. Not folded into "off", because "we
#: do not know" and "we know it was off" are different rows to a reader.
FLUSH_UNRECORDED = "l2_flush=unrecorded"


def arm_ruler(arm: ArmResult) -> str:
    """The instrument string AND the flush state, which is what a ratio needs.

    WHY THE FLUSH IS PART OF THE RULER AND NOT A FOOTNOTE. `KernelTiming` stamps
    `timing.TIMING_BASIS` on every row it produces, unconditionally, whatever
    `l2_flush` was; `timing.py`'s own contract (its module docstring) is that a
    row is comparable with the ROOF only if it carries that string. This script
    times with the flush OFF on purpose -- `time_arm` says why, and the reason
    is V2 -- so its rows carry the basis of an instrument configured differently
    from the one that measured the roof. Reading the basis alone, a downstream
    consumer would divide these milliseconds by a roof they were never
    commensurable with.

    So V7 asks its question of THIS string rather than of the column, and
    `roof_comparable` answers the roof question separately and in the negative.
    Two rows with the same ruler are divisible by each other, which is the only
    thing every ratio on this page needs; two rows with the same instrument and
    different flush states are not, and until 2026-09-02 V7 called them one.
    """
    flush = {True: "l2_flush=on", False: "l2_flush=off"}.get(
        arm.l2_flush, FLUSH_UNRECORDED)
    return f"{arm.instrument or '(none recorded)'} {flush}"


def roof_comparable(rulers: dict[str, int], basis: str | None) -> bool | None:
    """May these rows be divided by the ROOF? None when this host cannot say.

    True needs both halves: the instrument named by `timing.TIMING_BASIS`, and
    the flush ON, because the roof was measured queue-deep WITH the flush. This
    script's default answer is therefore False, deliberately and per its own
    design, and it is stamped in `summary.json` so that the negative travels
    with the numbers instead of living in a docstring here. Nothing on this page
    is scored against the roof -- every ratio is measured over measured and the
    predicted crossing cancels -- so False costs this script nothing and costs a
    reader who assumed otherwise a great deal.
    """
    if basis is None or not rulers:
        return None
    return set(rulers) == {f"{basis} l2_flush=on"}


def arm_ms(results: Results, cell_key: tuple[str, int], arm: str) -> float | None:
    """One arm's median, or None. Never 0.0 standing in for a missing timing."""
    got = results.get(cell_key, {}).get(arm)
    return got.ms_median if got is not None and got.timed else None


def summed_ms(results: Results, cell_key: tuple[str, int],
              arms: tuple[str, ...]) -> float | None:
    """Sum of several arms, or None if ANY of them is missing.

    None rather than a partial sum. A partial sum of a two-GEMM curve is a
    one-GEMM curve wearing the other's label, and it would land in a crossing
    with nothing to mark it.
    """
    total = 0.0
    for arm in arms:
        got = arm_ms(results, cell_key, arm)
        if got is None:
            return None
        total += got
    return total


def curve(cells: list[Cell], results: Results, model: str,
          arms: tuple[str, ...]) -> list[tuple[float, float]]:
    """`(tokens, ms)` for one model, summing `arms` per cell."""
    points = []
    for cell in cells:
        if cell.model != model:
            continue
        total = summed_ms(results, cell.key, arms)
        if total is not None:
            points.append((float(cell.num_tokens), total))
    return sorted(points)


@dataclass(frozen=True)
class Crossings:
    """Where one curve crosses slope 0.5, at both ends of the staircase.

    BOTH ends, always. `moe/bench/crossing.all_crossings_from_points` records
    that 8 of 16 canonical cells cross more than once and that taking the last
    instead of the first moves the published separation from 0.560 to 0.889.
    A single number here would be a choice this script is not entitled to make.
    """

    model: str
    label: str
    tokens: tuple[float, ...]
    n_points: int

    @property
    def first(self) -> float | None:
        return self.tokens[0] if self.tokens else None

    @property
    def last(self) -> float | None:
        return self.tokens[-1] if self.tokens else None

    def at(self, end: str) -> float | None:
        return self.first if end == "first" else self.last


def crossings_of(points: list[tuple[float, float]], model: str,
                 label: str) -> Crossings:
    """Every upcrossing above the model's saturation batch.

    The floor is not optional. Below `E/k` tokens a batch does not touch every
    expert, so weight traffic grows WITH the batch and the slope crosses 0.5 for
    a reason that has nothing to do with the ridge; without it mixtral reported
    a crossing at 5 tokens against a predicted 641.
    """
    floor = saturation_batch(model)
    found = all_crossings_from_points(points, min_tokens=floor)
    return Crossings(model, label, tuple(found), len(points))


@dataclass
class ModelDecomposition:
    """One model's three crossings and the two factors they produce."""

    model: str
    five: Crossings
    one_triton: Crossings
    one_cutlass: Crossings
    #: The two one-launch curves taken singly, for V6's self-consistency check.
    triton_up: Crossings | None = None
    triton_down: Crossings | None = None
    cutlass_up: Crossings | None = None
    cutlass_down: Crossings | None = None
    excluded: str = ""

    def factors(self, end: str) -> tuple[float | None, float | None, float | None]:
        """`(extent, kernel, separation)` at one end of the staircase."""
        c5, c1t, c1c = (self.five.at(end), self.one_triton.at(end),
                        self.one_cutlass.at(end))
        extent = c5 / c1t if c5 and c1t else None
        kernel = c1t / c1c if c1t and c1c else None
        separation = c5 / c1c if c5 and c1c else None
        return extent, kernel, separation

    def estimator_spread(self, end: str) -> float | None:
        """Worst up-versus-down disagreement across both one-launch pairs.

        The quantity `docs/FINDINGS.md:786-791` records at about 2.3x. Returns
        None when neither pair has both halves, which is UNKNOWN and not
        agreement.
        """
        worst = None
        for up, down in ((self.triton_up, self.triton_down),
                         (self.cutlass_up, self.cutlass_down)):
            if up is None or down is None:
                continue
            a, c = up.at(end), down.at(end)
            if not a or not c:
                continue
            ratio = max(a / c, c / a)
            worst = ratio if worst is None else max(worst, ratio)
        return worst


@dataclass
class CellBudget:
    """One cell's launch-by-launch time budget. No estimator, no model."""

    cell: Cell
    fused_ms: float | None
    launch_ms: dict[str, float | None]
    padding: float | None
    padding_source: str
    block_m: int | None

    @property
    def parts_total(self) -> float | None:
        got = [v for v in self.launch_ms.values() if v is not None]
        if len(got) != len(LAUNCH_ARMS):
            return None
        return sum(got)

    @property
    def reconstruction(self) -> float | None:
        total, fused = self.parts_total, self.fused_ms
        if total is None or not fused:
            return None
        return total / fused

    @property
    def gemm_ms(self) -> float | None:
        got = [self.launch_ms.get(a) for a in GEMM_ARMS]
        return None if any(v is None for v in got) else sum(got)

    @property
    def non_gemm_share(self) -> float | None:
        """Share of the RECONSTRUCTED total, not of the fused time.

        Of the reconstructed total on purpose: dividing by the fused time would
        fold the reconstruction residual into the share and make an arm's
        overhead look like an extra stage.
        """
        total, gemm = self.parts_total, self.gemm_ms
        if total is None or gemm is None or total <= 0:
            return None
        return (total - gemm) / total

    @property
    def extent_time(self) -> float | None:
        """`fused / (gemm_up + gemm_down)`: the same kernel at two extents."""
        gemm, fused = self.gemm_ms, self.fused_ms
        if gemm is None or not fused or gemm <= 0:
            return None
        return fused / gemm


def build_budgets(cells: list[Cell], results: Results) -> list[CellBudget]:
    out = []
    for cell in cells:
        launches = {arm: arm_ms(results, cell.key, arm) for arm in LAUNCH_ARMS}
        fused = results.get(cell.key, {}).get("fused")
        block_m = None
        padding = None
        source = ""
        if fused is not None:
            block_m = (fused.config or {}).get("BLOCK_SIZE_M")
            padding = padding_factor(fused.padded_rows, fused.rows_total)
            source = fused.padding_source
        out.append(CellBudget(cell, arm_ms(results, cell.key, "fused"), launches,
                              padding, source, block_m))
    return out


def kernel_time_ratio(results: Results, cell: Cell) -> float | None:
    """`(gemm_up + gemm_down) / (cutlass_up + cutlass_down)` at one cell."""
    triton = summed_ms(results, cell.key, GEMM_ARMS)
    cutlass = summed_ms(results, cell.key, CUTLASS_ARMS)
    if triton is None or cutlass is None or cutlass <= 0:
        return None
    return triton / cutlass


def nearest_cell(cells: list[Cell], model: str, tokens: float) -> Cell | None:
    """The measured cell closest in log-token distance to a crossing."""
    got = [c for c in cells if c.model == model and c.num_tokens > 0]
    if not got or tokens is None or tokens <= 0:
        return None
    return min(got, key=lambda c: abs(math.log(c.num_tokens) - math.log(tokens)))


def bootstrap_interval(values: list[float], band: float = BOOTSTRAP_BAND,
                       reps: int = BOOTSTRAP_REPS,
                       seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """Percentile bootstrap of the MEDIAN, resampling MODELS.

    Models and not timing samples: the uncertainty that matters for a headline
    is "would another set of models have said the same", which is between-model.
    None below two values, because a bootstrap of one point is that point.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(statistics.median([values[rng.randrange(n)] for _ in range(n)])
                     for _ in range(reps))
    lo = int((1 - band) / 2 * reps)
    hi = min(reps - 1, int((1 + band) / 2 * reps))
    return medians[lo], medians[hi]


def mde_log_ratio(models: int, sd_log: float = PRIOR_ARM_SD_LOG) -> float | None:
    """Smallest log-ratio this design can resolve, at 5% two-sided and 80% power.

    `(z(0.975) + z(0.80)) * sqrt(2) * sd / sqrt(n)`. The `sqrt(2)` is the ratio:
    every quantity gated here is one arm's crossing over another's, so two
    independent per-arm spreads enter. The `sqrt(n)` is the n MODELS the median
    is taken over, which is also what the bootstrap resamples.

    THREE APPROXIMATIONS, named rather than hidden, and every one of them makes
    this number SMALLER than the truth, so it is a lower bound on what the
    design can miss and never a promise about what it can see.

      * The median of n is treated as a mean of n. At n = 2 to 4 the median is
        the less efficient estimator, so the real interval is wider.
      * `sd_log` is a per-ARM replicate spread; a crossing is read off a whole
        curve through `all_crossings_from_points`, and a slope estimator can
        amplify or damp the spread of the points it is fitted through by a
        factor this script does not know.
      * The models are treated as independent draws. They are four fixed
        architectures, not a sample from a population, so `sqrt(n)` is a
        convenient fiction; what the bootstrap and this line both really say is
        "the four models we have disagree by about this much or less".

    None below two models, because there is no n to divide by.
    """
    if models < 2:
        return None
    if sd_log <= 0:
        raise ValueError(f"an MDE needs a positive sd, got {sd_log}")
    return MDE_Z_SUM * math.sqrt(2.0) * sd_log / math.sqrt(models)


def log_share(factor: float | None, separation: float | None) -> float | None:
    """What fraction of the separation one factor carries, in log terms.

    Multiplicative, so log space is the only apportionment that adds up: the two
    shares sum to exactly 1 by construction. None where there is nothing to
    apportion, rather than dividing by a log near zero and printing 4000%.
    """
    if factor is None or separation is None:
        return None
    if factor <= 0 or separation <= 0:
        return None
    denom = math.log(separation)
    if abs(denom) < 1e-9:
        return None
    return math.log(factor) / denom



@dataclass(frozen=True)
class ModelProbe:
    """C3's mechanism test for ONE model: does the kernel gap appear only where
    compute bites?

    THE PROBLEM THIS SOLVES. The obvious form of the padding test -- evaluate
    `p` at the crossing and compare it to `1/KERNEL` -- cannot work on a
    log-spaced grid. `p` is a step function of rows per expert, the crossing is
    interpolated between two grid points, and mixtral's one-launch crossing at
    T=520 sits 1.6% above T=512 where `p` steps from 1.00 to 1.48. The estimator
    then swings by 50% on a 1.6% move in the crossing and reports whichever side
    of a tile boundary the interpolation happened to land on.

    THE DIFFERENTIAL FORM. Padding costs COMPUTE and not TRAFFIC. So the
    Triton-over-CUTLASS time ratio should carry the padding factor ABOVE the
    crossing, where compute sets the time, and should not carry it BELOW, where
    traffic does. The contrast

        q = median(KERNEL_time above) / median(KERNEL_time below)

    is predicted to be the measured padding factor on the compute-bound side,
    and any constant "Triton is generally slower than CUTLASS" factor divides
    out of it. Both halves are measured at matched cells against the real
    routing histogram; nothing is interpolated and nothing is modelled.

    ITS LIMIT, stated because a gate that cannot be wrong is not a gate. The
    nuisance constant cancels only insofar as Triton's efficiency relative to
    CUTLASS is the same in the two regimes. It need not be: one comparison is
    bandwidth and the other is tensor core. A FAIL is therefore "padding does
    not explain the gap OR the two efficiencies differ", and this script does
    not claim to tell those apart.
    """

    model: str
    compute_cells: int
    memory_cells: int
    kernel_time_compute: float | None
    kernel_time_memory: float | None
    padding_compute: float | None

    @property
    def contrast(self) -> float | None:
        above, below = self.kernel_time_compute, self.kernel_time_memory
        if not above or not below:
            return None
        return above / below

    @property
    def discriminates(self) -> bool:
        """Can this model's cells tell the padding hypothesis from the null?

        The null is `q = 1`: the kernel gap is a constant that does not care
        about the regime. If `1` already sits inside the acceptance band around
        the measured padding factor then a PASS would be reported whether the
        mechanism is there or not, and the check examined nothing.

        THIS IS NOT HYPOTHETICAL AND IT IS THE REASON THE CHECK EXISTS. Above
        the crossing the rows per expert are large and `p` decays toward 1 like
        `1 + BM/2r`; on the default powers-of-two grid the compute-bound cells
        measure about 1.07, seven percent from the null inside a twenty-five
        percent band, and all three self-test worlds passed C3 identically.
        Padding bites in a narrow window AROUND the crossing, at 80 to 200 rows
        per expert, and a grid that does not sample it cannot see the mechanism.
        """
        p = self.padding_compute
        if p is None or p <= 0 or self.contrast is None:
            return False
        return abs(1.0 - p) > PADDING_TOLERANCE * p

    @property
    def ratio_to_prediction(self) -> float | None:
        """`contrast / p`, predicted 1.0 if padding is the whole mechanism."""
        q, p = self.contrast, self.padding_compute
        if q is None or p is None or p <= 0:
            return None
        return q / p


#: C3 needs this many cells on EACH side of a model's crossing before that
#: model may speak. Two, because a median of one is that one.
MIN_PROBE_CELLS = 2

#: The guard band, as a multiple of the crossing. A cell inside it is on neither
#: side of the transition, and averaging it into both turns a contrast between
#: two regimes into a contrast between one regime and itself.
PROBE_GUARD = 1.5


@dataclass(frozen=True)
class PaddingProbe:
    """Every model's mechanism test, and the verdict over the ones with power.

    Models WITHOUT power are named rather than averaged in. A model whose
    compute-bound cells carry no padding contributes a contrast of 1 against a
    prediction of 1, which would drag the pooled answer toward agreement no
    matter what the mechanism is doing -- the ragged-grid pooling mistake, in
    its most flattering form.
    """

    per_model: tuple[ModelProbe, ...]

    @property
    def powered(self) -> tuple[ModelProbe, ...]:
        return tuple(m for m in self.per_model if m.discriminates)

    @property
    def median_ratio(self) -> float | None:
        got = [m.ratio_to_prediction for m in self.powered
               if m.ratio_to_prediction is not None]
        return statistics.median(got) if got else None

    @property
    def agrees(self) -> bool | None:
        """PASS, FAIL, or UNKNOWN. UNKNOWN whenever no model had power."""
        ratio = self.median_ratio
        if ratio is None:
            return None
        return abs(ratio - 1.0) <= PADDING_TOLERANCE

    def describe(self) -> str:
        if not self.per_model:
            return "no model produced a crossing to place cells against"
        parts = []
        for m in self.per_model:
            if m.contrast is None or m.padding_compute is None:
                parts.append(f"{m.model}: {m.compute_cells} above / "
                             f"{m.memory_cells} below, too few to compare")
                continue
            tag = "" if m.discriminates else "  NO POWER (p is too near 1)"
            parts.append(
                f"{m.model}: contrast {m.contrast:.3f} against p "
                f"{m.padding_compute:.3f} -> {m.ratio_to_prediction:.3f} "
                f"({m.compute_cells}/{m.memory_cells} cells){tag}")
        ratio = self.median_ratio
        head = ("no model had power; --densify puts cells at 80-200 rows per "
                "expert where padding bites"
                if ratio is None
                else f"median contrast/p over {len(self.powered)} powered "
                     f"models {ratio:.3f}, predicted 1.000")
        return head + "; " + " | ".join(parts)


def padding_probe(cells: list[Cell], results: Results,
                  per_model: list[ModelDecomposition]) -> PaddingProbe:
    """Build C3's contrast per model, each against its OWN crossing.

    Per model and only then pooled, because a cell's regime is defined by its
    own model's crossing: deepseek-v3 at T=2048 is memory-bound and mixtral at
    T=2048 is compute-bound, and a pool built on token count alone would put
    them on the same side.
    """
    probes = []
    budgets = {b.cell.key: b for b in build_budgets(cells, results)}
    for dec in per_model:
        if dec.excluded or dec.one_triton.first is None:
            continue
        crossing = dec.one_triton.first
        floor = saturation_batch(dec.model)
        above, below, padding = [], [], []
        for cell in cells:
            if cell.model != dec.model:
                continue
            ratio = kernel_time_ratio(results, cell)
            if ratio is None:
                continue
            if cell.num_tokens >= PROBE_GUARD * crossing:
                above.append(ratio)
                pad = budgets[cell.key].padding
                if pad is not None:
                    padding.append(pad)
            elif floor <= cell.num_tokens <= crossing / PROBE_GUARD:
                below.append(ratio)
        probes.append(ModelProbe(
            model=dec.model, compute_cells=len(above), memory_cells=len(below),
            kernel_time_compute=(statistics.median(above)
                                 if len(above) >= MIN_PROBE_CELLS else None),
            kernel_time_memory=(statistics.median(below)
                                if len(below) >= MIN_PROBE_CELLS else None),
            padding_compute=(statistics.median(padding)
                             if len(padding) >= MIN_PROBE_CELLS else None)))
    return PaddingProbe(tuple(probes))


@dataclass
class Analysis:
    """Everything the report prints and every gate reads, computed in one place."""

    per_model: list[ModelDecomposition]
    budgets: list[CellBudget]
    #: end -> factor name -> median over models
    medians: dict[str, dict[str, float | None]]
    #: end -> {"extent", "kernel"} -> median over models of that model's OWN log
    #: share. Taken per model and then medianed, never from the medians: the
    #: three factors are exact per model, and `median(extent) * median(kernel)`
    #: is not `median(separation)` on a ragged set of models.
    shares: dict[str, dict[str, float | None]]
    intervals: dict[str, dict[str, tuple[float, float] | None]]
    kernel_time_median: float | None
    extent_time_median: float | None
    reconstruction_median: float | None
    non_gemm_share_at_crossing: float | None
    #: `p` MEASURED from the real routing histogram at the cell nearest each
    #: model's five-launch crossing. A diagnostic; C3 reads the probe instead.
    padding_measured_near_crossing: float | None
    #: C3's differential mechanism test.
    probe: PaddingProbe
    #: Refusal class name -> the cells that raised it. A cell whose rig refused
    #: writes an errored row and no timings, so without this list V0 and V1
    #: would both be gates that CANNOT FAIL: the only run in which the assembly
    #: is wrong is the run in which no assembly was ever compared.
    refusals: dict[str, list[str]]
    #: Cells whose observed tile config was missing one of the six keys.
    incomplete_configs: list[str]
    placebo_p90: float | None
    max_rel_err: float | None
    cells_measured: int
    arms_timed: int
    min_samples: int | None
    triton_artifacts: int
    config_mismatches: list[str]
    configs_observed: int
    comparability: dict[str, float]
    estimator_spreads: dict[str, float]
    failures: list[str]
    #: Instrument string -> how many timed arms carry it. A run whose rows came
    #: from two instruments is a run whose ratios compare two rulers, and until
    #: 2026-09-02 every row here carried no instrument at all.
    instruments: dict[str, int]
    #: RULER string -> how many timed arms carry it, where a ruler is the
    #: instrument TOGETHER WITH the L2 flush state. V7 reads this one, and the
    #: two are separate fields because they answer different questions: the
    #: instrument column says which code timed the row, the ruler says whether
    #: two rows are divisible by each other. `arm_ruler` says why the flush has
    #: to be in it.
    rulers: dict[str, int]
    #: Timed arms whose recorded clock verdict was False. Counted rather than
    #: gated: every ratio here is between two arms of the SAME round-robin
    #: repeat, so a card that sat low all session moves neither, and what the
    #: counts are for is a reader deciding whether to believe a 5% effect.
    clock_level_bad: int
    clock_drift_bad: int
    #: Timed arms whose queue had drained before the host finished enqueueing,
    #: so their intervals include host time and bound the kernel from above.
    #: At T=1 a `moe_sum` over 8 rows is host-bound by construction, which is
    #: why this is reported and not gated.
    host_bound_arms: int

    @property
    def decomposed(self) -> list[ModelDecomposition]:
        return [m for m in self.per_model if not m.excluded]

    @property
    def interval_n(self) -> int:
        """How many models each interval in `intervals` was resampled over.

        The SCOPE of every interval this script prints, and the reason it is a
        property rather than a comment: `MIN_DECOMPOSED_MODELS` is 2, so a run
        that satisfies V4 can still produce a 90% "interval" out of two points.
        """
        return len(self.decomposed)

    @property
    def interval_scope(self) -> str:
        """One sentence naming what the intervals resample and over what n."""
        n = self.interval_n
        base = (f"{BOOTSTRAP_BAND:.0%} percentile bootstrap of the MEDIAN, "
                f"resampling the {n} decomposed model(s) with replacement, "
                f"{BOOTSTRAP_REPS} reps, seed {BOOTSTRAP_SEED}. It is "
                f"between-MODEL uncertainty and carries nothing about timing "
                f"noise within a cell.")
        if n < 2:
            return base + (" No interval was produced: a bootstrap of one point "
                           "is that point.")
        if n <= BOOTSTRAP_DEGENERATE_N:
            return base + (f" DEGENERATE at n={n}: resampling {n} values with "
                           f"replacement gives only a handful of distinct "
                           f"medians, so this interval is a restatement of the "
                           f"{n} points and NOT a sampling distribution. Read "
                           f"the per-model table instead.")
        return base


def analyse(cells: list[Cell], results: Results) -> Analysis:
    """Reduce a whole run to the numbers the gates and the headline read.

    Everything is computed per MODEL first and then medianed, never pooled. The
    cells span three orders of magnitude in absolute time and a pooled ratio
    would be the T=8192 cell wearing a costume; this is the same mistake
    `alpha_refit.py` documents in its estimator note.
    """
    models = list(dict.fromkeys(c.model for c in cells))
    per_model: list[ModelDecomposition] = []
    for model in models:
        five = crossings_of(curve(cells, results, model, ("fused",)), model,
                            "five-launch Triton")
        one_t = crossings_of(curve(cells, results, model, GEMM_ARMS), model,
                             "one-launch Triton")
        one_c = crossings_of(curve(cells, results, model, CUTLASS_ARMS), model,
                             "one-launch CUTLASS")
        dec = ModelDecomposition(
            model, five, one_t, one_c,
            triton_up=crossings_of(curve(cells, results, model, ("gemm_up",)),
                                   model, "gemm_up"),
            triton_down=crossings_of(curve(cells, results, model, ("gemm_down",)),
                                     model, "gemm_down"),
            cutlass_up=crossings_of(curve(cells, results, model, ("cutlass_up",)),
                                    model, "cutlass_up"),
            cutlass_down=crossings_of(curve(cells, results, model, ("cutlass_down",)),
                                      model, "cutlass_down"))
        missing = [c.label for c in (five, one_t, one_c) if not c.tokens]
        if missing:
            dec.excluded = ("no crossing located on: " + ", ".join(missing)
                            + "; the grid does not bracket one for this model")
        else:
            spread = dec.estimator_spread("first")
            if spread is not None and spread > ESTIMATOR_AGREEMENT_MAX:
                dec.excluded = (
                    f"the up and down one-launch crossings disagree by "
                    f"{spread:.2f}x, over the {ESTIMATOR_AGREEMENT_MAX:.1f}x "
                    f"limit; the estimator is not resolving this model")
        per_model.append(dec)

    medians: dict[str, dict[str, float | None]] = {}
    intervals: dict[str, dict[str, tuple[float, float] | None]] = {}
    shares: dict[str, dict[str, float | None]] = {}
    for end in ("first", "last"):
        collected: dict[str, list[float]] = {"extent": [], "kernel": [],
                                             "separation": []}
        share_of: dict[str, list[float]] = {"extent": [], "kernel": []}
        for dec in per_model:
            if dec.excluded:
                continue
            extent, kernel, separation = dec.factors(end)
            for name, value in (("extent", extent), ("kernel", kernel),
                                ("separation", separation)):
                if value is not None:
                    collected[name].append(value)
            for name, value in (("extent", extent), ("kernel", kernel)):
                got = log_share(value, separation)
                if got is not None:
                    share_of[name].append(got)
        medians[end] = {k: (statistics.median(v) if v else None)
                        for k, v in collected.items()}
        intervals[end] = {k: bootstrap_interval(v) for k, v in collected.items()}
        shares[end] = {k: (statistics.median(v) if v else None)
                       for k, v in share_of.items()}

    budgets = build_budgets(cells, results)
    recon = [b.reconstruction for b in budgets if b.reconstruction is not None]
    extent_times = [b.extent_time for b in budgets if b.extent_time is not None]
    kernel_times = [r for r in (kernel_time_ratio(results, c) for c in cells)
                    if r is not None]

    # The non-GEMM share and the padding factor are read at the cell NEAREST
    # each model's own five-launch crossing, not pooled over the grid. At T=1
    # the share is launch overhead and at T=8192 it is rounding, and neither is
    # the regime the separation lives in.
    nongemm_shares, measured_p = [], []
    by_cell = {b.cell.key: b for b in budgets}
    for dec in per_model:
        if dec.excluded or dec.five.first is None:
            continue
        near = nearest_cell(cells, dec.model, dec.five.first)
        budget = by_cell.get(near.key) if near else None
        if budget is None:
            continue
        if budget.non_gemm_share is not None:
            nongemm_shares.append(budget.non_gemm_share)
        if budget.padding is not None:
            measured_p.append(budget.padding)

    placebos, rel_errs, mismatches, failures = [], [], [], []
    refusals: dict[str, list[str]] = {}
    incomplete: list[str] = []
    configs_observed = arms_timed = artifacts = 0
    instruments: dict[str, int] = {}
    rulers: dict[str, int] = {}
    clock_level_bad = clock_drift_bad = host_bound_arms = 0
    sample_counts: list[int] = []
    fused_config: dict[tuple[str, int], dict] = {}
    for cell in cells:
        arms = results.get(cell.key, {})
        fused = arms.get("fused")
        if fused is not None and fused.config:
            fused_config[cell.key] = fused.config
            configs_observed += 1
        for name, arm in arms.items():
            where = f"{cell.model} T={cell.num_tokens} {name}"
            if arm.config and name in LAUNCH_ARMS + ("fused",):
                absent = [k for k in TILE_KEYS if k not in arm.config]
                if absent:
                    incomplete.append(f"{where}: config lacks {absent}")
            if arm.error:
                failures.append(f"{where}: {arm.error}")
                kind = arm.error.split(":", 1)[0].strip()
                refusals.setdefault(kind, []).append(where)
                continue
            if arm.timed:
                arms_timed += 1
                sample_counts.append(arm.n_samples)
                # The empty string is a real state and gets its own key: a row
                # with no instrument came from a laptop or from before the
                # instrument column existed, and folding it in with the rows
                # that name one is how a retired ruler survives a resume.
                instruments[arm.instrument] = instruments.get(arm.instrument, 0) + 1
                ruler = arm_ruler(arm)
                rulers[ruler] = rulers.get(ruler, 0) + 1
                clock_level_bad += arm.clock_level_ok is False
                clock_drift_bad += arm.clock_drift_ok is False
                host_bound_arms += arm.host_bound is True
            artifacts += arm.triton_artifacts
            if arm.rel_err_vs_fused is not None:
                rel_errs.append(arm.rel_err_vs_fused)
            # Within one session this compares a value with itself -- the
            # launches are handed the fused call's config -- and it is kept for
            # the case where it does not: a RESUMED run restores launch rows
            # written by an earlier session and times the fused arm fresh. If
            # the two disagree then the run id let two different settings share
            # a directory, or the installed vLLM changed under the CSV, and
            # either way the rows on disk are not describing one experiment.
            if name in LAUNCH_ARMS and arm.config and cell.key in fused_config:
                want = fused_config[cell.key]
                differing = {k: (want.get(k), v) for k, v in arm.config.items()
                             if want.get(k) != v}
                if differing:
                    mismatches.append(
                        f"{where}: the fused arm observed {want} but this "
                        f"launch row carries {differing} (fused, launch) -- a "
                        f"resumed row from a different setting or a different "
                        f"vLLM")
        a, b = arm_ms(results, cell.key, "fused"), arm_ms(results, cell.key,
                                                          "fused_replica")
        if a and b:
            placebos.append(abs(b / a - 1.0))

    comparability = {}
    for dec in per_model:
        published = PUBLISHED_FIVE_STAGE_CROSSING.get(dec.model)
        if published and dec.five.first:
            comparability[dec.model] = dec.five.first / published

    spreads = {}
    for dec in per_model:
        spread = dec.estimator_spread("first")
        if spread is not None:
            spreads[dec.model] = spread

    measured = sum(1 for c in cells
                   if arm_ms(results, c.key, "fused") is not None)
    return Analysis(
        per_model=per_model, budgets=budgets, medians=medians,
        intervals=intervals, shares=shares,
        kernel_time_median=statistics.median(kernel_times) if kernel_times else None,
        extent_time_median=statistics.median(extent_times) if extent_times else None,
        reconstruction_median=statistics.median(recon) if recon else None,
        non_gemm_share_at_crossing=(statistics.median(nongemm_shares)
                                    if nongemm_shares else None),
        padding_measured_near_crossing=(statistics.median(measured_p)
                                        if measured_p else None),
        probe=padding_probe(cells, results, per_model),
        placebo_p90=percentile(placebos, 0.90),
        max_rel_err=max(rel_errs) if rel_errs else None,
        cells_measured=measured, arms_timed=arms_timed,
        min_samples=min(sample_counts) if sample_counts else None,
        triton_artifacts=artifacts, config_mismatches=mismatches,
        configs_observed=configs_observed, comparability=comparability,
        estimator_spreads=spreads, failures=failures, refusals=refusals,
        incomplete_configs=incomplete, instruments=instruments,
        rulers=rulers,
        clock_level_bad=clock_level_bad, clock_drift_bad=clock_drift_bad,
        host_bound_arms=host_bound_arms)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, so every number printed was actually observed."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q * len(ordered) + 0.5))))
    return ordered[rank - 1]


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

#: The two kinds, taken FROM the shared table rather than spelled again here.
#: `exit_codes.classify` raises on a kind it does not know rather than letting
#: it fall through a comparison and be scored as whatever the fallthrough
#: happened to be, so a local "Validity" would have been a crash, not a silent
#: pass -- but only if the two spellings are the same object, which is what this
#: line guarantees.
VALIDITY, CLAIM = exit_codes.VALIDITY, exit_codes.CLAIM

#: The ONE token each gate answers to on its `RESULT:` line. A name is one run
#: of non-whitespace by `moe.bench.exit_codes.result_line`'s own rule, so it is
#: fixed here rather than sliced out of the human title: "C4 extra stages" has
#: two words after the number and slicing would have silently produced "extra".
#: These are the words a driver greps for, so they are part of the contract and
#: change only when the gate changes.
GATE_TOKENS: dict[str, str] = {
    "V0": "assembly", "V1": "same_tile", "V2": "reconstruction",
    "V3": "placebo", "V4": "non_vacuity", "V5": "comparable",
    "V6": "estimator_agreement", "V7": "one_instrument",
    "C1": "extent_is_not_the_story", "C2": "kernel_is_the_story",
    "C3": "mechanism_is_padding", "C4": "extra_stages_are_cheap",
    "C5": "reproduces_the_published_separation",
}


@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` is UNKNOWN and never a pass. A gate nobody could evaluate is
    the state this project's retractions were written in, and
    `moe.bench.exit_codes` scores UNKNOWN against the gate for the same reason.
    """

    name: str
    kind: str                 # VALIDITY or CLAIM
    prediction: str
    rule: str
    passed: bool | None
    observed: str
    invalidates: str

    @property
    def tag(self) -> str:
        """PASS / FAIL / UNKNOWN, in `moe.bench.exit_codes`'s spelling."""
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.passed]

    @property
    def token(self) -> str:
        """The one-word name on the RESULT line. Raises on an unregistered gate.

        A missing token would otherwise be filled in with something derived, and
        a derived name that changed when the title changed is a gate that
        silently disappears from a driver's summary.
        """
        key = self.name.split()[0]
        if key not in GATE_TOKENS:
            raise KeyError(f"gate {self.name!r} has no token in GATE_TOKENS; "
                           "a RESULT line's name is part of the driver contract "
                           "and is registered, not derived")
        return GATE_TOKENS[key]

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`'s vocabulary."""
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.token, self.tag)

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        The `[PASS] V0 assembly ...` block below is for a human and for
        `scripts/h200_gaps_session.sh`, whose span gate regex has matched that
        exact shape since before this change; both are kept because they have
        different readers, and only this one is the machine contract. Nothing
        else this file prints starts with `RESULT: `.
        """
        detail = f"[{self.kind}] {self.prediction} | saw {self.observed} | " \
                 f"gate {self.rule}"
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> str:
        lines = [f"[{self.tag}] {self.name}  {self.prediction}",
                 f"         gate: {self.rule}",
                 f"         saw:  {self.observed}"]
        if self.passed is not True:
            lines.append(f"         a FAIL means: {self.invalidates}")
        return "\n".join(lines)


def render_gates(gates: list[Gate]) -> str:
    lines = [g.render() for g in gates]
    passed = sum(1 for g in gates if g.passed is True)
    failed = sum(1 for g in gates if g.passed is False)
    unknown = sum(1 for g in gates if g.passed is None)
    lines += ["", f"{passed} PASS, {failed} FAIL, {unknown} UNKNOWN"]
    return "\n".join(lines)


def in_band(value: float | None, band: tuple[float, float]) -> bool | None:
    return None if value is None else band[0] <= value <= band[1]


def build_gates(analysis: Analysis) -> list[Gate]:
    """The twelve pre-registered gates, evaluated against the run.

    VALIDITY first, so a reader hits the reasons to distrust the page before the
    page's conclusion.
    """
    gates: list[Gate] = []
    void = ("no number on this page may be quoted: the one-launch arms are not "
            "demonstrably the fused path's own kernel")

    err = analysis.max_rel_err
    # THE REFUSALS ARE WHAT MAKES THIS GATE ABLE TO FAIL. `build_rig` raises
    # AssemblyMismatch rather than spending metered GPU time timing an assembly
    # that is not the fused path, so a cell whose assembly is wrong records an
    # ERROR and no rel_err at all. Reading only `max_rel_err` would therefore
    # produce a gate whose only two states are PASS and UNKNOWN -- the exact
    # shape of the check in this repo that had never passed on any machine.
    mismatched = analysis.refusals.get("AssemblyMismatch", [])
    v0 = None
    if mismatched:
        v0 = False
    elif err is not None:
        v0 = err <= OUTPUT_REL_TOL
    gates.append(Gate(
        "V0 assembly", "VALIDITY",
        "the five launches recompute fused_experts' own output",
        f"zero cells refused for assembly mismatch, and max relative error of "
        f"the assembly vs the fused call <= {OUTPUT_REL_TOL:g}",
        v0,
        "no assembly was compared" if err is None and not mismatched
        else (f"{len(mismatched)} cells REFUSED for mismatch"
              + (f" (first: {mismatched[0]})" if mismatched else "")
              + (f"; max rel err elsewhere {err:.2e}" if err is not None else "")),
        void))

    # V1 does NOT compare the launches' config against the fused call's: the
    # launches are HANDED that config, so such a check compares a value with
    # itself and can only pass. What can go wrong, and what this asks, is
    # whether the tile was observable at all -- a vLLM that memoises its own
    # lookup records nothing, a chunked call resolves two configs, and a stale
    # override_config context makes the observed tile not vLLM's choice. Each
    # raises in the prologue and lands here as a refusal.
    tile_refusals = (analysis.refusals.get("ConfigUnobserved", [])
                     + analysis.refusals.get("SignatureDrifted", [])
                     + analysis.refusals.get("PieceMissing", []))
    v1 = None
    if tile_refusals or analysis.incomplete_configs or analysis.config_mismatches:
        v1 = False
    elif analysis.configs_observed:
        v1 = True
    gates.append(Gate(
        "V1 same tile", "VALIDITY",
        "the tile the launches ran was READ OUT of the fused call, not derived",
        "every measured cell observed exactly one complete tile config, with no "
        "override in force, no chunking, and no resumed row carrying a "
        "different one",
        v1,
        "no fused config was observed, so the tile is still unchecked"
        if v1 is None
        else (f"{analysis.configs_observed} cells observed a complete config, "
              f"{len(tile_refusals)} refused, "
              f"{len(analysis.incomplete_configs)} incomplete"
              + (f"; first refusal: {tile_refusals[0]}" if tile_refusals else "")
              + (f", {len(analysis.config_mismatches)} resumed rows disagree"
                 if analysis.config_mismatches else "")
              + (f"; first incomplete: {analysis.incomplete_configs[0]}"
                 if analysis.incomplete_configs else "")
              + (f"; first disagreement: {analysis.config_mismatches[0]}"
                 if analysis.config_mismatches else "")),
        "EXTENT is a tile comparison wearing a span-extent costume"))

    recon = analysis.reconstruction_median
    gates.append(Gate(
        "V2 reconstruction", "VALIDITY",
        "the five launch times sum to the fused time",
        f"median (sum of launches) / fused in "
        f"[{RECONSTRUCTION_BAND[0]}, {RECONSTRUCTION_BAND[1]}]",
        in_band(recon, RECONSTRUCTION_BAND),
        "no cell had all five launches timed" if recon is None
        else f"median reconstruction {recon:.3f} over "
             f"{len(analysis.budgets)} cells",
        "the launch-by-launch budget is not a budget for this path, and the "
        "time-domain bound may not be quoted"))

    band = analysis.placebo_p90
    gates.append(Gate(
        "V3 placebo", "VALIDITY",
        "re-timing the SAME fused call moves the answer by almost nothing",
        f"p90 of |replica/fused - 1| < {PLACEBO_BAND:.0%}",
        None if band is None else band < PLACEBO_BAND,
        "no placebo pair timed" if band is None
        else f"p90 placebo deviation {band:.2%}",
        "the box is too noisy for a ratio this size to mean anything"))

    enough = (analysis.cells_measured >= MIN_MEASURED_CELLS
              and len(analysis.decomposed) >= MIN_DECOMPOSED_MODELS
              and (analysis.min_samples or 0) >= MIN_SAMPLES_PER_ARM
              and analysis.triton_artifacts > 0)
    gates.append(Gate(
        "V4 non-vacuity", "VALIDITY",
        "real work happened: cells, models, samples and Triton compiles",
        f">= {MIN_MEASURED_CELLS} cells, >= {MIN_DECOMPOSED_MODELS} decomposed "
        f"models, >= {MIN_SAMPLES_PER_ARM} samples on every arm, > 0 Triton "
        f"artefacts",
        enough,
        f"{analysis.cells_measured} cells, {len(analysis.decomposed)} decomposed "
        f"models, min samples {analysis.min_samples}, "
        f"{analysis.triton_artifacts} Triton artefacts",
        "the run examined too little to have found anything; a check that "
        "examined nothing reports zero failures too"))

    comp = list(analysis.comparability.values())
    comp_ok = None
    if comp:
        inside = sum(1 for v in comp if COMPARABILITY_BAND[0] <= v <= COMPARABILITY_BAND[1])
        comp_ok = inside * 2 >= len(comp)
    gates.append(Gate(
        "V5 comparable", "VALIDITY",
        "this session's five-launch crossing is the published one",
        f"at least half the models inside "
        f"[{COMPARABILITY_BAND[0]}, {COMPARABILITY_BAND[1]}]x the published crossing",
        comp_ok,
        "no model had both a crossing and a published anchor" if not comp
        else "; ".join(f"{m} {v:.2f}x" for m, v in sorted(analysis.comparability.items())),
        "this session did not reproduce the published crossing, so whatever it "
        "decomposes is not the published 0.563"))

    # A DECOMPOSED MODEL WITH NO MEASURED SPREAD IS NOT A MODEL THAT AGREED.
    # This used to read `.get(d.model, 1.0)`, substituting perfect agreement for
    # a model whose up/down pair was never measured -- and the `observed` string
    # was built only from the models that HAD a spread, so the substituted one
    # was not named anywhere on the page. One measured model was then enough to
    # report PASS over any number of unmeasured ones. V6 is the gate that
    # decides whether the crossing estimator resolved these curves at all, so a
    # vacuous PASS there gets the EXTENT/KERNEL decomposition quoted on models
    # the estimator was never shown to work on.
    spreads = list(analysis.estimator_spreads.values())
    unmeasured = sorted(d.model for d in analysis.decomposed
                        if d.model not in analysis.estimator_spreads)
    disagreeing = sorted(
        d.model for d in analysis.decomposed
        if analysis.estimator_spreads.get(d.model, 0.0) > ESTIMATOR_AGREEMENT_MAX)
    v6_verdict: bool | None
    if not spreads or unmeasured:
        # None is this file's UNDECIDED: the gate could not ask, and a check
        # that examined nothing reports zero failures too.
        v6_verdict = None
    else:
        v6_verdict = not disagreeing
    observed = "; ".join(f"{m} {v:.2f}x"
                         for m, v in sorted(analysis.estimator_spreads.items()))
    if not spreads:
        observed = "no model had both halves of a one-launch pair"
    if unmeasured:
        observed = ((observed + "; ") if spreads else "") + (
            f"NO up/down spread measured for {unmeasured}, which is "
            "decomposed and would have been scored as perfect agreement")
    gates.append(Gate(
        "V6 estimator", "VALIDITY",
        "the up and down one-launch crossings agree with each other",
        f"every decomposed model has a MEASURED up/down spread and it is <= "
        f"{ESTIMATOR_AGREEMENT_MAX:.1f}x",
        v6_verdict, observed,
        "the crossing estimator is not resolving these curves, so both factors "
        "are noise; read the time-domain bound instead"))

    # V7. ONE RULER FOR EVERY ARM. Every ratio on this page is one arm's
    # milliseconds over another's, so the one thing that must be true of the
    # timings is that they were produced by the same instrument IN THE SAME
    # CONFIGURATION. It is a reachable FAIL in four real ways and each has
    # happened or could happen here: a resumed CSV holding rows written by the
    # retired `time_call` loop beside rows written by `time_kernel`; a
    # `--self-test` world, whose arms carry `synthetic:<world>` and were
    # produced by no instrument at all; a laptop replay where torch would not
    # import so the basis could not even be named; and a CSV mixing flushed with
    # unflushed rows, which `KernelTiming` stamps with the SAME basis string and
    # which this gate called one ruler until 2026-09-02. `arm_ruler` is where
    # the flush joined the key. The gate reads the strings on the ROWS, not a
    # run-level header, because a resume is exactly the case a header gets
    # wrong.
    basis = timing_basis()
    found = analysis.instruments
    rulers = analysis.rulers
    v7_verdict: bool | None
    if not found:
        v7_verdict = None
    elif len(rulers) > 1:
        # A definite mix, and nameable or not it is one: two rulers in one CSV
        # is a FAIL this host can see without knowing what either should be.
        v7_verdict = False
    elif basis is None:
        # Nothing to compare against: torch is absent here, so this process
        # cannot say what the one ruler on disk should have said.
        v7_verdict = None
    else:
        v7_verdict = set(found) == {basis}
    named = ", ".join(f"{name}: {n} arm(s)" for name, n in sorted(rulers.items()))
    roof_ok = roof_comparable(rulers, basis)
    # The clause below is about ONE case: rows this instrument produced, with
    # the flush off. A synthetic world is not roof-comparable either, and saying
    # "carries the basis string" about `synthetic:kernel` would be false.
    basis_unflushed = (basis is not None and set(found) == {basis}
                       and roof_ok is False)
    gates.append(Gate(
        "V7 one instrument", VALIDITY,
        "every timed arm was measured by the one instrument the study times "
        "with, in one configuration of it",
        f"every timed arm's `instrument` column is {basis or 'TIMING_BASIS'} "
        f"and its `l2_flush` column agrees with every other row's",
        v7_verdict,
        ("no arm was timed, so no instrument was recorded" if not found
         else (f"observed {named}"
               + ("; torch is not importable here so this process cannot name "
                  "the instrument to compare against"
                  if basis is None else "")
               + ("; NOT ROOF-COMPARABLE: these rows carry the basis string but "
                  "were not flushed, and the roof was, so nothing here may be "
                  "divided by it -- nothing here is (see `roof_comparable`)"
                  if basis_unflushed else "")
               + (f"; clock LEVEL bad on {analysis.clock_level_bad}, DRIFT bad "
                  f"on {analysis.clock_drift_bad}, host-bound on "
                  f"{analysis.host_bound_arms} of {analysis.arms_timed} timed "
                  f"arms (reported, not gated: every ratio here is between two "
                  f"arms of one round-robin repeat)"))),
        "the ratios on this page divide a millisecond measured one way by a "
        "millisecond measured another, which is the defect the whole apparatus "
        "was rebuilt to remove"))

    first = analysis.medians["first"]
    last = analysis.medians["last"]

    extent = first.get("extent")
    gates.append(Gate(
        "C1 extent", "CLAIM",
        "span extent does NOT move the crossing",
        f"median EXTENT (five-launch / one-launch, both Triton) in "
        f"[{EXTENT_BAND[0]}, {EXTENT_BAND[1]}] on the first crossing",
        in_band(extent, EXTENT_BAND),
        "no model produced both crossings" if extent is None
        else (f"EXTENT {extent:.3f} first, "
              + (f"{last['extent']:.3f} last" if last.get("extent") else "-- last")),
        "span extent DOES move the crossing and part of the published 0.563 is "
        "the extra stages after all -- which is a result, not a broken run"))

    kernel = first.get("kernel")
    gates.append(Gate(
        "C2 kernel", "CLAIM",
        "the Triton-versus-CUTLASS kernel difference carries the separation",
        f"median KERNEL (one-launch Triton / one-launch CUTLASS) <= {KERNEL_MAX}, "
        f"i.e. at least half the published separation in log terms",
        None if kernel is None else kernel <= KERNEL_MAX,
        "no model produced both crossings" if kernel is None
        else (f"KERNEL {kernel:.3f} first, "
              + (f"{last['kernel']:.3f} last" if last.get("kernel") else "-- last")
              + (f", log share {analysis.shares['first']['kernel']:.0%}"
                 if analysis.shares["first"].get("kernel") is not None else "")),
        "neither factor carries the separation, so the published number is not "
        "explained by extent OR by these two kernels, and the missing CUTLASS "
        "five-launch corner is where the rest of it lives"))

    probe = analysis.probe
    gates.append(Gate(
        "C3 mechanism", "CLAIM",
        "the mechanism is PADDING: the kernel gap appears only where compute bites",
        f"the Triton/CUTLASS time ratio ABOVE the crossing over the same ratio "
        f"BELOW it is the measured padding factor, within {PADDING_TOLERANCE:.0%}",
        probe.agrees,
        probe.describe(),
        "padding does not explain the kernel gap, OR Triton's efficiency "
        "relative to CUTLASS differs between the bandwidth-bound and the "
        "compute-bound regime; this script cannot tell those apart"))

    share = analysis.non_gemm_share_at_crossing
    gates.append(Gate(
        "C4 extra stages", "CLAIM",
        "the four non-GEMM stages are cheap in TIME as well as in traffic",
        f"median non-GEMM share of the reconstructed fused time at the crossing "
        f"cells < {NON_GEMM_SHARE_MAX:.0%}",
        None if share is None else share < NON_GEMM_SHARE_MAX,
        "no crossing cell had all five launches timed" if share is None
        else f"non-GEMM share {share:.1%}",
        "the extra stages cost real time even though they move few bytes, and "
        "any five-stage-versus-one-stage MILLISECOND is partly extent"))

    sep = first.get("separation")
    gates.append(Gate(
        "C5 the published number", "CLAIM",
        f"this session reproduces the published separation of {PUBLISHED_SEPARATION}",
        f"median in-session separation in "
        f"[{SEPARATION_BAND[0]:.3f}, {SEPARATION_BAND[1]:.3f}]",
        in_band(sep, SEPARATION_BAND),
        "no model produced both end crossings" if sep is None
        else (f"separation {sep:.3f} first "
              f"(published {PUBLISHED_SEPARATION_FIRST}), "
              + (f"{last['separation']:.3f} last "
                 f"(published {PUBLISHED_SEPARATION_LAST})"
                 if last.get("separation") else "-- last")),
        "the decomposition is exact but it decomposes THIS session's separation, "
        "not the published one"))
    return gates


# --------------------------------------------------------------------------
# The synthetic worlds. Hermetic: nothing here reads the device, so a replay is
# identical on every machine.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class World:
    """A world the gates must be able to tell apart.

    `extra_stage_scale` inflates the non-GEMM launches; `triton_pads` and
    `cutlass_pads` decide which kernel computes padded rows. The real hardware
    is claimed to be `kernel`: Triton pads, CUTLASS does not, and the extra
    stages cost what the byte model says they cost.
    """

    name: str
    extra_stage_scale: float
    triton_pads: bool
    cutlass_pads: bool
    summary: str


WORLDS: dict[str, World] = {
    "kernel": World(
        "kernel", 1.0, True, False,
        "padding only: Triton pads the sampled histogram to BLOCK_M, CUTLASS "
        "does not, extra stages cost what the byte model says. Signature: "
        "EXTENT near 1, KERNEL below 1."),
    "extent": World(
        # The scale is SOLVED before the world is generated; see
        # `extent_scale_for_separation`. 1.0 is a placeholder that is never used.
        "extent", 1.0, True, True,
        "extent only: both kernels pad identically so there is no kernel "
        "difference, and the extra stages are inflated by whatever factor it "
        "takes for span extent ALONE to produce the published separation. "
        "Signature: EXTENT at the published separation, KERNEL at 1 -- and the "
        "factor itself is the interesting number."),
    "neither": World(
        "neither", 1.0, False, False,
        "neither: no padding anywhere and cheap extra stages. Signature: both "
        "factors near 1 and no separation to decompose."),
}


#: WHAT EACH WORLD MUST PRODUCE, gate by gate, on the grid the driver runs.
#:
#: WHY THIS TABLE EXISTS. Until 2026-09-02 `--self-test WORLD` generated a world,
#: ran the whole analysis on it, printed twelve verdicts and asserted NOTHING
#: about any of them; it exited 1 if any gate failed, so the only outcome it
#: could distinguish was "all green". On the grid the session driver actually
#: scheduled first, the `kernel` world -- the world in which the claim is TRUE --
#: produced C2 FAIL (0.815 against a 0.75 gate), C5 FAIL and C3 UNKNOWN, and the
#: `neither` world produced the IDENTICAL claim-gate row. Forty-five minutes of
#: pod were booked on a grid where the primary gate could not pass even when its
#: hypothesis held, and nothing in the code said so, because nothing in the code
#: had an opinion about what a world should produce.
#:
#: HOW TO READ A ROW. Every entry is what the gate MUST say in that world. A
#: PASS here is not "the code works", it is "the world was built so this gate
#: has this answer, and if it does not, either the analysis changed or the world
#: did". FAIL entries are as load-bearing as PASS entries and there are more of
#: them: `extent` fails four claim gates BECAUSE the mechanism is in the other
#: factor, and a run where those four came out PASS would mean the gates cannot
#: tell the two worlds apart, which is the only thing this self test is for.
#:
#: V7 IS FAIL IN ALL THREE, on purpose and not for want of a fourth world. A
#: synthetic arm time was produced by no instrument, so it carries
#: `synthetic:<world>` and V7 refuses it. That is what makes a `--self-test`
#: page unquotable by the same rule that makes a pod page quotable, and it is
#: why the self test is scored on the S gates below rather than on these twelve.
#:
#: V5 IS FAIL IN `neither`, which is also a fact about the world rather than a
#: defect: with no padding anywhere the five-launch crossings land nowhere near
#: the published ones, and V5 asks exactly that question.
WORLD_EXPECTATIONS: dict[str, dict[str, str]] = {
    "kernel": {
        "V0": exit_codes.PASS, "V1": exit_codes.PASS, "V2": exit_codes.PASS,
        "V3": exit_codes.PASS, "V4": exit_codes.PASS, "V5": exit_codes.PASS,
        "V6": exit_codes.PASS, "V7": exit_codes.FAIL,
        "C1": exit_codes.PASS, "C2": exit_codes.PASS, "C3": exit_codes.PASS,
        "C4": exit_codes.PASS, "C5": exit_codes.PASS,
    },
    "extent": {
        "V0": exit_codes.PASS, "V1": exit_codes.PASS, "V2": exit_codes.PASS,
        "V3": exit_codes.PASS, "V4": exit_codes.PASS, "V5": exit_codes.PASS,
        "V6": exit_codes.PASS, "V7": exit_codes.FAIL,
        "C1": exit_codes.FAIL, "C2": exit_codes.FAIL, "C3": exit_codes.FAIL,
        "C4": exit_codes.FAIL, "C5": exit_codes.PASS,
    },
    "neither": {
        "V0": exit_codes.PASS, "V1": exit_codes.PASS, "V2": exit_codes.PASS,
        "V3": exit_codes.PASS, "V4": exit_codes.PASS, "V5": exit_codes.FAIL,
        "V6": exit_codes.PASS, "V7": exit_codes.FAIL,
        "C1": exit_codes.PASS, "C2": exit_codes.FAIL, "C3": exit_codes.FAIL,
        "C4": exit_codes.PASS, "C5": exit_codes.FAIL,
    },
}

#: The gates whose registered row has to differ between worlds for the
#: experiment to be able to settle anything. The VALIDITY gates are about the
#: apparatus and are allowed to agree everywhere; these are the claim.
DISCRIMINATING_GATES: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5")


def world_triples() -> dict[str, tuple[str, ...]]:
    """Each world's registered claim-gate row, for the discrimination check."""
    return {name: tuple(row[g] for g in DISCRIMINATING_GATES)
            for name, row in WORLD_EXPECTATIONS.items()}


#: (model, tokens, seed) -> per-expert row histogram. The solver below rebuilds
#: the whole synthetic world 24 times, and drawing 400k rows per build would
#: make an off-GPU self-test slower than the analysis it exercises.
_COUNT_CACHE: dict[tuple[str, int, int], list[int]] = {}


def synthetic_counts(cfg, tokens: int, rng: random.Random) -> list[int]:
    """A per-expert row histogram, drawn the way sampled uniform routing draws.

    NOT the balanced histogram, and that is the whole point. Under balanced
    routing on a powers-of-two token grid every expert holds a power-of-two
    number of rows and BLOCK_M is a power of two, so `ceil(n/BM) BM == n` and
    the padding the `kernel` world exists to plant would be exactly ZERO at
    every grid point. Sampled routing scatters the counts, an expert one row
    over a tile boundary pays a whole extra tile, and at mixtral T=256 that is
    worth about 1.5x on its own.

    Rows are drawn with replacement, which is a simplification: real top-k picks
    `k` DISTINCT experts per token, so this slightly overstates the variance for
    large `k`. It is a generator for a self-test, never a measurement, and the
    real run reads the real histogram off the device.

    Seeded from the caller's Random, so a replay is identical on every machine.
    """
    counts = [0] * cfg.num_experts
    for _ in range(tokens * cfg.top_k):
        counts[rng.randrange(cfg.num_experts)] += 1
    return counts


def padded_from_counts(counts: list[int], block_m: int) -> float:
    """`sum_e ceil(n_e / BM) BM`, the rows `moe_align_block_size` emits.

    Per expert and not globally. Padding globally would understate the cost by a
    factor of E at decode, which is the regime this whole study lives in.
    """
    return float(sum(math.ceil(n / block_m) * block_m for n in counts if n > 0))


def synthetic_results(cells: list[Cell], world: World, *, ridge: float,
                      bandwidth_gbps: float, noise: float,
                      seed: int) -> Results:
    """Every arm's time, generated from the model in one named world.

    The generator is `modelled_ms`, the same arithmetic `--dry-run` prices the
    run with, so a self-test exercises the analysis rather than a second model
    written to agree with the first.
    """
    out: Results = {}
    for cell in cells:
        cfg = cell.cfg
        b = dtype_bytes(cell.dtype)
        bm = ladder_block_m(cell.num_tokens, cfg.num_experts, cell.dtype)
        # Seeded per CELL, not per run, so a cell's histogram does not depend on
        # how many cells were planned before it. A grid that changed every
        # earlier cell's routing when a token count was added would make two
        # self-tests incomparable for a reason nobody would look for.
        # sha256 and not hash(): PYTHONHASHSEED randomises str hashing, so a
        # replay seeded from hash() would differ between two runs on the same
        # machine, which is the one thing a hermetic self-test may not do.
        rng = random.Random(int(hashlib.sha256(
            f"{seed}:{cell.model}:{cell.num_tokens}".encode()
        ).hexdigest()[:12], 16))
        cache_key = (cell.model, cell.num_tokens, seed)
        if cache_key not in _COUNT_CACHE:
            _COUNT_CACHE[cache_key] = synthetic_counts(cfg, cell.num_tokens, rng)
        counts = _COUNT_CACHE[cache_key]
        rows_total = float(sum(counts))
        padded = padded_from_counts(counts, bm)
        kw = {"b": b, "ridge": ridge, "bandwidth_gbps": bandwidth_gbps}
        triton_rows = padded if world.triton_pads else rows_total
        cutlass_rows = padded if world.cutlass_pads else rows_total
        gemm_t = {
            "gemm_up": modelled_ms(cfg, cell.num_tokens, "up_gemm",
                                   compute_rows=triton_rows, **kw),
            "gemm_down": modelled_ms(cfg, cell.num_tokens, "down_gemm",
                                     compute_rows=triton_rows, **kw),
        }
        cut_t = {
            "cutlass_up": modelled_ms(cfg, cell.num_tokens, "up_gemm",
                                      compute_rows=cutlass_rows, **kw),
            "cutlass_down": modelled_ms(cfg, cell.num_tokens, "down_gemm",
                                        compute_rows=cutlass_rows, **kw),
        }
        extra = {name: world.extra_stage_scale
                 * modelled_ms(cfg, cell.num_tokens, name, **kw)
                 for name in NON_GEMM_ARMS}
        fused = sum(gemm_t.values()) + sum(extra.values())
        times = dict(gemm_t, **cut_t, **extra)
        times["fused"] = fused
        times["fused_replica"] = fused
        arms: dict[str, ArmResult] = {}
        for arm, ms in times.items():
            jitter = math.exp(rng.gauss(0.0, noise)) if noise else 1.0
            arms[arm] = ArmResult(
                cell.model, cell.num_tokens, arm, ms_median=ms * jitter,
                ms_mean=ms * jitter, ms_stdev=0.0, ms_min=ms * jitter,
                n_samples=MIN_SAMPLES_PER_ARM,
                config=(ladder_config(cell.num_tokens, cfg.num_experts,
                                      cell.dtype)
                        if arm in LAUNCH_ARMS + ("fused",) else None),
                tile_config_source="synthetic_ladder",
                rows_total=rows_total, padded_rows=padded,
                padding_source="synthetic_histogram",
                active_experts=cfg.num_experts,
                rel_err_vs_fused=0.0 if arm == "gemm_up" else None,
                triton_artifacts=1 if arm == "gemm_up" else 0,
                # NO INSTRUMENT PRODUCED THIS NUMBER, and the row says so in the
                # field a reader filters on. V7 then FAILS in every world, which
                # is what stops a laptop `--self-test` page being quotable --
                # the same job `bm128_roofline`'s hypothesis-roof refusal does
                # for a synthetic roof. It is also why the self test is scored
                # on its S gates rather than on the twelve below.
                instrument=synthetic_instrument(world))
        out[cell.key] = arms
    return out


def synthetic_instrument(world: World) -> str:
    """`synthetic:<world>`: the string a generated arm time carries.

    Deliberately not empty. An empty `instrument` is what a row from a laptop
    or from before 2026-09-02 carries, and the two states have to be
    distinguishable in a CSV that could hold both.
    """
    return f"synthetic:{world.name}"


@dataclass(frozen=True)
class ExtentSolve:
    """What span extent ALONE can and cannot do, solved rather than asserted."""

    #: The inflation of the four extra stages the world was generated at.
    scale: float
    #: The separation that inflation produced.
    separation: float | None
    #: Did it reach the published number?
    reached: bool
    #: The lowest separation any inflation reached while every model still had
    #: a crossing to read. Beyond that the fused curve is linear from the first
    #: grid point, its slope never RISES through 0.5, and the estimator has no
    #: answer at all -- an estimator limit, not a proof of impossibility.
    floor: float | None
    models_at_floor: int


def extent_scale_for_separation(cells: list[Cell], target: float, *,
                                ridge: float, bandwidth_gbps: float, seed: int,
                                ladder_points: int = 25,
                                max_scale: float = 1e5) -> ExtentSolve:
    """How expensive would the four extra stages have to be to produce `target`?

    THE QUESTION THIS ANSWERS, and it is the point of the `extent` world. The
    published separation is 0.563. If span extent alone caused it, the permute,
    activation and combine stages would have to cost some amount, and that
    amount is either plausible or it is not. Rather than assert "the extra
    stages are cheap", this SOLVES for the cost the extent explanation requires
    and prints it, so a reader can compare it against the byte model's own
    figure of a few percent.

    Both kernels pad identically in this world, so the KERNEL factor is pinned
    at 1 and every bit of the separation has to come from extent.

    A GEOMETRIC LADDER rather than a bisection, because the answer is not
    guaranteed to be monotone all the way down: past some inflation the fused
    curve is linear from the first grid point, its slope never rises THROUGH
    0.5, `all_crossings_from_points` correctly returns nothing, and the model
    drops out of the median. A bisection would walk into that region and report
    whatever the surviving models happened to say. The ladder keeps only the
    scales at which every model that crossed at scale 1 still crosses, and the
    lowest separation among those is the FLOOR: the most span extent can do to
    this number while the estimator can still read it.
    """
    def evaluate(scale: float) -> tuple[float | None, int]:
        world = World("extent", scale, True, True, "")
        results = synthetic_results(cells, world, ridge=ridge,
                                    bandwidth_gbps=bandwidth_gbps, noise=0.0,
                                    seed=seed)
        analysis = analyse(cells, results)
        return (analysis.medians["first"].get("separation"),
                len(analysis.decomposed))

    baseline_sep, baseline_models = evaluate(1.0)
    best_scale, best_sep = 1.0, baseline_sep
    reached_scale = None
    for i in range(ladder_points):
        scale = max_scale ** (i / max(ladder_points - 1, 1))
        sep, models = evaluate(scale)
        if sep is None or models < baseline_models:
            continue
        if best_sep is None or sep < best_sep:
            best_scale, best_sep = scale, sep
        if reached_scale is None and sep <= target:
            reached_scale = scale
    if reached_scale is not None:
        sep, _ = evaluate(reached_scale)
        return ExtentSolve(reached_scale, sep, True, best_sep, baseline_models)
    return ExtentSolve(best_scale, best_sep, False, best_sep, baseline_models)


# --------------------------------------------------------------------------
# Can this grid answer at all? And did the world it was checked in behave?
# --------------------------------------------------------------------------

def run_world(cells: list[Cell], world: World, *, ridge: float,
              bandwidth_gbps: float, noise: float, seed: int
              ) -> tuple[Analysis, list[Gate]]:
    """Generate a world over `cells` and run the SHIPPED analysis and gates on it.

    One function, so the grid-power check before a run and the self test after
    one are the same code path. A power check that used its own reduced
    analysis would answer about the check rather than about the experiment.
    """
    results = synthetic_results(cells, world, ridge=ridge,
                                bandwidth_gbps=bandwidth_gbps, noise=noise,
                                seed=seed)
    analysis = analyse(cells, results)
    return analysis, build_gates(analysis)


@dataclass(frozen=True)
class GridPower:
    """Whether the planned grid can answer C2 in the world where C2 is TRUE.

    Not a statistical power calculation and it does not pretend to be: it plants
    the mechanism C2 claims, at zero noise, and asks whether the gate as written
    comes out PASS on THIS grid. A grid that cannot pass a gate in a world built
    to satisfy it cannot pass it on hardware either, whatever the hardware does.
    """

    can_answer: bool
    kernel: float | None
    verdict: str
    #: What to run instead. Empty when the grid can answer.
    remedy: str

    @property
    def detail(self) -> str:
        seen = ("no KERNEL factor was recoverable at all" if self.kernel is None
                else f"planted-kernel world gives KERNEL {self.kernel:.3f} "
                     f"against the gate's {KERNEL_MAX}")
        return f"C2 {self.verdict} in the planted kernel world: {seen}"


#: The remedy named in the refusal, in the words an operator has to type. It is
#: a single string because the refusal, the plan line and the test all have to
#: say the same thing, and three copies of a sentence drift.
DENSIFY_REMEDY = ("grid too sparse for C2; use --densify (it is the default; "
                  "--no-densify is what removed the points C2 needs)")


def c2_grid_power(cells: list[Cell], *, ridge: float, bandwidth_gbps: float,
                  seed: int) -> GridPower:
    """Ask, BEFORE measuring, whether this grid could ever answer C2.

    THE FINDING THIS IS THE FIX FOR. The session driver booked 45 minutes for
    the sparse grid and 35 for the dense one, sparse first. On the sparse grid
    the `kernel` world -- padding in Triton, none in CUTLASS, the exact
    mechanism C2 names -- produces KERNEL 0.815 against a gate of 0.75, i.e. C2
    FAIL, and produces the SAME claim-gate row as the `neither` world, in which
    there is no mechanism at all. A pod run on that grid could return C2 FAIL
    and it would have meant nothing, because C2 FAIL was the only thing it could
    return.

    The reason is arithmetic rather than noise: on a powers-of-two token grid
    every expert holds a power-of-two number of rows and BLOCK_M is a power of
    two, so `ceil(r/BM) BM == r` and the padding factor is exactly 1.00 at every
    grid point. The mechanism is invisible where the grid looks. `--densify`
    adds points at 0.5x to 4x each model's published crossing, where r lands at
    80-200 and p is 1.3-1.8, and there the same world gives 12 gates PASS.

    Zero noise on purpose. This asks whether the DESIGN can answer, not whether
    a particular draw would; the MDE line in the plan is where noise enters.
    """
    analysis, gates = run_world(cells, WORLDS["kernel"], ridge=ridge,
                                bandwidth_gbps=bandwidth_gbps, noise=0.0,
                                seed=seed)
    c2 = next(g for g in gates if g.name.startswith("C2"))
    ok = c2.passed is True
    return GridPower(ok, analysis.medians["first"].get("kernel"), c2.tag,
                     "" if ok else DENSIFY_REMEDY)


def self_test_gates(world: World, gates: list[Gate]) -> list[Gate]:
    """The S gates: did this world produce the verdicts registered for it?

    ONE GATE PER GATE, not one for the world. A world that got eleven of twelve
    right and the twelfth wrong has to name the twelfth; a single "the world
    matched" gate would report one FAIL for any number of causes and a reader
    would have to diff two tables to find out which.

    Plus a discrimination gate over the registered table itself. Every world
    reproducing its own row proves nothing unless the rows DIFFER: three worlds
    that all registered the same claim-gate answers would each pass their own S
    gates and the experiment would still be unable to settle anything. That
    check is over the table rather than over three live runs because a run
    scores one world at a time; the test suite runs all three rows, and the
    combination of "each world reproduces its row" and "the rows differ" is the
    proof.

    These are VALIDITY gates. A world that did not behave means nothing on the
    page can be quoted, which is exactly what `exit_codes.INVALID` says.
    """
    expected = WORLD_EXPECTATIONS.get(world.name)
    out: list[Gate] = []
    if expected is None:
        out.append(Gate(
            "S-registered self test", VALIDITY,
            "the world under test has a registered row of expected verdicts",
            f"world {world.name!r} appears in WORLD_EXPECTATIONS",
            False,
            f"world {world.name!r} has no registered row, so nothing it "
            f"produced could be wrong",
            "a self test with no expectation is a self test that cannot fail, "
            "which is the finding this table was added for"))
        return out

    by_key = {g.name.split()[0]: g for g in gates}
    missing = sorted(set(expected) - set(by_key))
    unregistered = sorted(set(by_key) - set(expected))
    out.append(Gate(
        "S-registered self test", VALIDITY,
        "every gate the run scored has a registered expectation, and vice versa",
        f"the gate set and WORLD_EXPECTATIONS[{world.name!r}] name the same gates",
        not missing and not unregistered,
        ("every one of the "
         f"{len(expected)} registered gates was scored" if not missing
         and not unregistered
         else f"registered but not scored: {missing or 'none'}; scored but not "
              f"registered: {unregistered or 'none'}"),
        "a gate added without an expectation is a gate no world constrains, and "
        "a gate removed without editing the table is an expectation about "
        "nothing"))
    for key in sorted(expected):
        want = expected[key]
        got = by_key.get(key)
        out.append(Gate(
            f"S-{key} world expectation", VALIDITY,
            f"in the {world.name!r} world, {key} comes out {want}",
            f"{key} verdict == {want}",
            None if got is None else got.tag == want,
            (f"{key} was not scored in this run" if got is None
             else f"{key} came out {got.tag}: {got.observed}"),
            f"either the analysis moved or the world did; a {key} that answers "
            f"the same in every world cannot settle this experiment"))
    triples = world_triples()
    distinct = len(set(triples.values()))
    out.append(Gate(
        "S-discrimination across worlds", VALIDITY,
        "the registered worlds do not all land on one claim-gate row",
        f"{len(triples)} worlds register {len(triples)} distinct "
        f"{DISCRIMINATING_GATES} rows",
        distinct == len(triples),
        f"{distinct} distinct rows across {len(triples)} worlds: "
        + "; ".join(f"{name} {row}" for name, row in sorted(triples.items())),
        "the gates cannot tell these worlds apart, so a verdict from the pod "
        "would not be evidence about which world we are in"))
    return out


def _register_self_test_tokens() -> None:
    """Give every S gate a RESULT name, derived once from the gate it mirrors.

    A driver greps a name, so an S gate needs one as much as a C gate does, and
    deriving it here rather than writing thirteen more literals keeps
    `self_test_kernel_is_the_story` in step with `kernel_is_the_story` when
    either is renamed. Run at import: a token missing at print time raises, and
    a gate that raises while rendering its own result line is worse than one
    with an ugly name.
    """
    GATE_TOKENS["S-registered"] = "self_test_registered"
    GATE_TOKENS["S-discrimination"] = "self_test_discrimination"
    for key, token in [(k, v) for k, v in GATE_TOKENS.items()
                       if not k.startswith("S-")]:
        GATE_TOKENS[f"S-{key}"] = f"self_test_{token}"


_register_self_test_tokens()


# --------------------------------------------------------------------------
# The measurement. The only part that needs the box.
# --------------------------------------------------------------------------

#: Where each launch of vLLM's fused path has lived. Probed rather than
#: assumed: a wrong guess would silently time nothing, or time a stand-in.
PIECE_MODULES: dict[str, tuple[str, ...]] = {
    # `vllm._custom_ops` is DELIBERATELY not a candidate here even though it
    # exports a symbol of this name. That one is the raw CUDA op and takes the
    # OUTPUT tensors as arguments; the python wrapper allocates them and returns
    # them. Two different functions sharing a name is exactly the silent
    # substitution this file exists to refuse, and `bind_call` would reject the
    # raw op anyway -- but on a parameter name, with a message about signature
    # drift, which would send a reader looking for the wrong thing.
    "moe_align_block_size": (
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.moe_align_block_size",
    ),
    "invoke_fused_moe_kernel": (
        "vllm.model_executor.layers.fused_moe.fused_moe",
    ),
    "silu_and_mul": (
        "vllm._custom_ops",
    ),
    "moe_sum": (
        "vllm._custom_ops",
        "vllm.model_executor.layers.fused_moe.fused_moe",
    ),
}


def find_pieces() -> dict:
    """Every launch of vLLM's fused path, by name, or a typed refusal.

    Refuses on the first missing piece and names it. There is deliberately no
    fallback: a torch stand-in for any of these would time a kernel nobody
    serves, and the budget would then describe a path that does not exist.
    """
    import importlib

    found = {}
    for attr, modules in PIECE_MODULES.items():
        for name in modules:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            fn = getattr(module, attr, None)
            if fn is not None:
                found[attr] = fn
                break
        else:
            raise PieceMissing(
                f"vLLM exposes no {attr} in any of {modules}. Without it the "
                "fused path cannot be taken apart, and a stand-in would time a "
                "different kernel. Run scripts/probe_baseline_api.py in this "
                "venv and say what it prints.")
    return found


def bind_call(fn, values: dict) -> dict:
    """Keyword arguments for `fn`, refusing rather than guessing.

    Supplies only names the signature has, and RAISES for any parameter that has
    no default and no value here. Filling an unknown required parameter with a
    plausible None produces a call that runs and times a different schedule,
    which is the failure this whole file is built around.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise SignatureDrifted(
            f"{getattr(fn, '__name__', fn)} has no introspectable signature, so "
            f"its arguments cannot be bound by name: {exc}") from None

    kwargs, unknown = {}, []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Parameter.empty:
            unknown.append(name)
    if unknown:
        raise SignatureDrifted(
            f"{getattr(fn, '__name__', fn)} requires {unknown}, which this "
            f"script has no value for. The installed vLLM's signature has moved "
            f"away from the one this experiment was written against; filling "
            f"them with defaults would time a different schedule.")
    return kwargs


def compute_type_for(dtype: str):
    """Triton's dtype for the accumulator/output, or a refusal."""
    import triton.language as tl

    table = {"bf16": tl.bfloat16, "fp16": tl.float16, "fp32": tl.float32}
    if dtype not in table:
        raise SignatureDrifted(
            f"no Triton compute type for dtype {dtype!r}; this experiment is "
            f"bf16 and the quantised paths take a different kernel entirely")
    return table[dtype]


@dataclass
class CellRig:
    """Everything one cell needs, built once in an UNTIMED prologue.

    The prologue is where the alignment metadata, the intermediate buffers, the
    permuted activations and the correctness comparison all happen, so that
    every timed region contains one launch and nothing else.
    """

    cell: Cell
    calls: dict            # arm -> zero-argument callable
    config: dict
    rows_total: float
    padded_rows: float
    active_experts: int
    rel_err: float


def build_rig(cell: Cell, args, pieces: dict) -> CellRig:
    """Take vLLM's fused path apart for one cell, and prove the parts are it.

    The order matters and is the whole safety argument:

      1. run `fused_experts` under the tile recorder, keeping its output;
      2. take the config it RESOLVED -- not one derived from the ladder;
      3. build the alignment metadata at that config's BLOCK_SIZE_M, because a
         mismatch there makes the sorted-token blocks disagree with the tiles
         and the output would be wrong rather than merely differently scheduled;
      4. run the five launches by hand into the same buffers;
      5. compare the assembly against the fused output, and REFUSE if it differs.

    Step 5 is what makes the one-launch arms evidence about the fused path
    rather than about some other kernel with a similar name.
    """
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts

    from moe.baselines._framework_config import (
        TileCapture,
        recording_tile_config,
        vllm_call_kwargs,
        vllm_override_active,
    )
    from moe.bench.tolerance import relative_error
    from moe.reference.torch_ref import build_permutation, make_inputs, swiglu
    from moe.routing.distributions import sample_topk_ids
    from moe.spec import BenchSpec, RoutingSpec

    cfg = cell.cfg
    spec = BenchSpec(cfg, num_tokens=cell.num_tokens, dtype=cell.dtype,
                     routing=RoutingSpec(args.routing, 0.0), seed=args.seed)
    x, weights = make_inputs(spec, device="cuda")
    ids = sample_topk_ids(spec.routing, cell.num_tokens, cfg.num_experts,
                          cfg.top_k, seed=args.seed, device="cuda")
    topk_w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                        device="cuda")
    kw = vllm_call_kwargs(spec)
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        kw["activation"] = MoEActivation(kw["activation"])
    except ImportError:
        # Older vLLM took the activation as a plain string. Left as-is rather
        # than guessing an enum that may not exist.
        pass

    def fused_call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=topk_w, topk_ids=ids, **kw)

    capture = TileCapture()
    with recording_tile_config(capture):
        y_fused = fused_call()
    torch.cuda.synchronize()
    if not capture.calls or not capture.calls[0].config:
        raise ConfigUnobserved(
            f"{cell.model} T={cell.num_tokens}: the recorder saw no tile config "
            "for the fused call, so the one-launch arms would have to derive "
            "one. A derived config that differs from the live one turns EXTENT "
            "into a tile comparison.")
    if len(capture.calls) != 1:
        raise ConfigUnobserved(
            f"{cell.model} T={cell.num_tokens}: the fused call resolved "
            f"{len(capture.calls)} configs, which means it CHUNKED. Every "
            "launch below assumes one config for the whole layer, and the "
            "budget would be of the first chunk only.")
    if vllm_override_active():
        raise ConfigUnobserved(
            f"{cell.model} T={cell.num_tokens}: an override_config context is "
            "in force, so the observed tile is one somebody FORCED and not the "
            "one vLLM would choose for this batch. A leaked context from "
            "another script in the same process is enough; nothing here sets "
            "one, which is why this is a refusal rather than a warning.")
    config = dict(capture.calls[0].config)
    absent = [k for k in TILE_KEYS if k not in config]
    if absent:
        raise ConfigUnobserved(
            f"{cell.model} T={cell.num_tokens}: the observed config is missing "
            f"{absent}. A launch handed four of six keys runs a tile vLLM did "
            "not choose for the other two, and the row would still look like a "
            "measurement of the tile.")
    block_m = int(config["BLOCK_SIZE_M"])

    align = pieces["moe_align_block_size"]
    invoke = pieces["invoke_fused_moe_kernel"]
    silu = pieces["silu_and_mul"]
    moe_sum = pieces["moe_sum"]

    align_kwargs = bind_call(align, {
        "topk_ids": ids, "block_size": block_m,
        "num_experts": cfg.num_experts, "expert_map": None,
        "pad_sorted_ids": False, "expert_map_or_none": None,
    })
    aligned = align(**align_kwargs)
    if not isinstance(aligned, tuple) or len(aligned) < 3:
        raise SignatureDrifted(
            f"moe_align_block_size returned {type(aligned).__name__} with "
            f"{len(aligned) if isinstance(aligned, tuple) else 1} values; this "
            "experiment expects (sorted_token_ids, expert_ids, "
            "num_tokens_post_padded)")
    sorted_ids, expert_ids, num_post_pad = aligned[0], aligned[1], aligned[2]

    m, top_k = cell.num_tokens, cfg.top_k
    n_up = weights.w1.shape[1]
    hidden = weights.w2.shape[1]
    dt = x.dtype
    cache1 = torch.empty((m, top_k, n_up), device="cuda", dtype=dt)
    cache2 = torch.empty((m * top_k, n_up // 2), device="cuda", dtype=dt)
    cache3 = torch.empty((m, top_k, hidden), device="cuda", dtype=dt)
    out = torch.empty((m, hidden), device="cuda", dtype=dt)
    compute_type = compute_type_for(cell.dtype)

    shared = {
        "A_scale": None, "B_scale": None, "B_zp": None,
        "topk_weights": topk_w, "sorted_token_ids": sorted_ids,
        "expert_ids": expert_ids, "num_tokens_post_padded": num_post_pad,
        "config": config, "compute_type": compute_type,
        "use_fp8_w8a8": False, "use_int8_w8a8": False, "use_int8_w8a16": False,
        "use_int4_w4a16": False, "use_mxfp4_w4a4": False,
        "per_channel_quant": False, "block_shape": None, "B_bias": None,
    }
    up_kwargs = bind_call(invoke, dict(
        shared, A=x, B=weights.w1, C=cache1, mul_routed_weight=False,
        top_k=top_k))
    down_kwargs = bind_call(invoke, dict(
        shared, A=cache2, B=weights.w2, C=cache3, mul_routed_weight=True,
        top_k=1))
    silu_kwargs_names = bind_call(silu, {"out": cache2, "x": cache1.view(-1, n_up),
                                         "input": cache1.view(-1, n_up)})
    sum_kwargs = bind_call(moe_sum, {"input": cache3, "output": out})

    def call_align():
        return align(**align_kwargs)

    def call_up():
        return invoke(**up_kwargs)

    def call_act():
        return silu(**silu_kwargs_names)

    def call_down():
        return invoke(**down_kwargs)

    def call_sum():
        return moe_sum(**sum_kwargs)

    # Run the assembly once, in the prologue, and prove it is the fused path.
    call_up()
    call_act()
    call_down()
    call_sum()
    torch.cuda.synchronize()
    rel = relative_error(out, y_fused)
    if rel > OUTPUT_REL_TOL:
        raise AssemblyMismatch(
            f"{cell.model} T={cell.num_tokens}: the five launches reproduce the "
            f"fused output to {rel:.3e}, over the {OUTPUT_REL_TOL:g} budget. "
            "The pieces this script assembled are not the path fused_experts "
            "runs, so nothing measured from them is evidence about it.")

    # CUTLASS arms: dense, pre-permuted, ragged M. Built here so no permute or
    # activation lands inside the timed region -- the point of the comparison is
    # the GEMM, and the published torch_grouped_mm_* spans are timed the same
    # way with ref_permute upstream.
    offsets, perm = build_permutation(ids, cfg.num_experts)
    x_perm = x[perm.long() // top_k]
    h_act = swiglu(cache1.view(-1, n_up)[perm.long()])
    offs = offsets[1:].to(torch.int32)
    grouped_mm = (getattr(torch.nn.functional, "grouped_mm", None)
                  or getattr(torch, "_grouped_mm", None))

    calls = {
        "fused": fused_call, "fused_replica": fused_call,
        "align": call_align, "gemm_up": call_up, "act": call_act,
        "gemm_down": call_down, "sum": call_sum,
    }
    if grouped_mm is not None:
        w1_t, w2_t = weights.w1.transpose(1, 2), weights.w2.transpose(1, 2)

        def call_cutlass_up():
            return grouped_mm(x_perm, w1_t, offs=offs)

        def call_cutlass_down():
            return grouped_mm(h_act, w2_t, offs=offs)

        calls["cutlass_up"] = call_cutlass_up
        calls["cutlass_down"] = call_cutlass_down

    counts = torch.bincount(ids.reshape(-1).to(torch.int64),
                            minlength=cfg.num_experts)
    padded = float((counts.float() / block_m).ceil().sum().item() * block_m)
    return CellRig(cell=cell, calls=calls, config=config,
                   rows_total=float(counts.sum().item()), padded_rows=padded,
                   active_experts=int((counts > 0).sum().item()), rel_err=rel)


def time_arm(fn, args, reference_clock_mhz: float | None = None):
    """Time one arm through `moe.bench.timing.time_kernel`. Returns a KernelTiming.

    THE INSTRUMENT IS NOT THIS FILE'S ANY MORE, and that is the point. Until
    2026-09-02 this script carried a private `time_calls`, one of six verbatim
    copies of the retired `time_call`: a fresh CUDA event pair created inside
    the loop, one `synchronize` per iteration, no L2 flush, no clock read. The
    audit bounded the host prefix that loop let inside the measured interval at
    about 0.18 ms per `fused_experts` call on the H200 pod and 0.30 ms on the
    A100, DIFFERENT PER CARD and of the order of the cross-card effect the study
    registered. Every ratio here is between two arms, so a constant additive
    prefix does not cancel: it inflates the SMALL arm most, and `align`, `act`
    and `sum` are the small arms, which is exactly what C4 and V2 read.

    WARMUP IS A DURATION NOW, not a count of calls. `--warmup-ms` is
    milliseconds of delivered GPU load, and `iters` is sized by the instrument
    from the warmup's own queue-deep per-call time so that a trial lasts
    `--target-ms`. There is no honest conversion from the old `--warmup 8`, so
    the flags were replaced rather than reinterpreted, and both are in the run
    id.

    THE L2 FLUSH IS OFF BY DEFAULT HERE, and this is the one place this study's
    instrument is configured differently from the one the ROOF was measured
    with. The reason is V2: it compares a launch timed ALONE against the same
    launch inside a five-launch sequence, where `silu_and_mul` reads an
    intermediate `gemm_up` wrote microseconds earlier. Flushing before every
    isolated call charges the isolated arm a cold cache the fused path never
    pays, which does not make the budget more honest, it makes it a budget of a
    different thing. Nothing here is scored against the roof -- every ratio on
    this page is measured/measured and the predicted crossing cancels -- so the
    comparison the flush protects is one this script does not make. `--l2-flush`
    turns it on, it is in the run id so the two can never share a directory, and
    `l2_flush` is on every row either way.

    THE LEVEL FLAG IS None UNLESS `--reference-clock-mhz` IS GIVEN. A LEVEL is
    relative to the clock the roof was measured at, `time_kernel` will not
    invent that something, and this script has no roof. What threatens a ratio
    between two arms timed minutes apart in one round-robin is DRIFT, which
    `clock_drift_ok` scores with no reference at all.
    """
    from moe.bench import timing

    return timing.time_kernel(
        fn, warmup_ms=args.warmup_ms, target_ms=args.target_ms,
        trials=args.trials, l2_flush=args.l2_flush,
        reference_clock_mhz=reference_clock_mhz)


def count_new(root: Path, seen: set[Path]) -> int:
    fresh = [p for p in root.rglob("*") if p.is_file() and p not in seen]
    seen.update(fresh)
    return len(fresh)


def measure_cell(cell: Cell, arms: list[str], args, store, meta: dict,
                 pieces: dict, cache_root: Path, seen: set[Path]
                 ) -> dict[str, ArmResult]:
    """Time every arm of one cell, round-robin across repeats.

    ROUND-ROBIN IS LOAD-BEARING. Running one arm to completion and then the next
    puts every slow-clock minute of the session into whichever arm ran during
    it, and the whole answer is a ratio between arms. Interleaving at the repeat
    level spreads a thermal excursion over all of them, and the fused/replica
    pair measures what is left.
    """
    results: dict[str, ArmResult] = {}
    pending: list[str] = []
    for arm in arms:
        restored = store.restore((cell.model, cell.num_tokens, arm))
        if restored is not None and not args.fresh:
            results[arm] = restored
        else:
            pending.append(arm)
    if not pending:
        return results

    rig = build_rig(cell, args, pieces)
    compiled = count_new(cache_root, seen)
    missing = [a for a in pending if a not in rig.calls]
    for arm in missing:
        results[arm] = ArmResult(
            cell.model, cell.num_tokens, arm,
            error="no callable for this arm in this venv: torch.grouped_mm is "
                  "absent, so the CUTLASS corner cannot be measured here")
        store.write(results[arm], cell, meta)
    pending = [a for a in pending if a in rig.calls]

    timings: dict[str, list] = {arm: [] for arm in pending}
    for _ in range(args.reps):
        for arm in list(timings):
            try:
                timings[arm].append(
                    time_arm(rig.calls[arm], args, meta.get("reference_clock_mhz")))
            except Exception as exc:  # noqa: BLE001
                # Broad on purpose: one arm running out of memory or hitting a
                # shape it cannot take must not end the cell, or the session.
                # `timing.TimingRefused` arrives here too and is recorded with
                # its own class name, so a refusal reads as a refusal in the CSV
                # rather than as an arm that happened to produce nothing.
                results[arm] = ArmResult(
                    cell.model, cell.num_tokens, arm,
                    error=f"{type(exc).__name__}: {exc}"[:300])
                timings.pop(arm, None)

    for arm in pending:
        got = timings.get(arm)
        if not got:
            results.setdefault(arm, ArmResult(
                cell.model, cell.num_tokens, arm,
                error="every repeat of this arm failed"))
            store.write(results[arm], cell, meta)
            continue
        p50s = [t.ms_p50 for t in got]
        result = ArmResult(
            cell.model, cell.num_tokens, arm,
            ms_median=statistics.median(p50s), ms_mean=statistics.fmean(p50s),
            # The spread ACROSS round-robin repeats, and only that. A pooled
            # within-trial spread would be the noise of one moment repeated,
            # and the thing the round-robin exists to expose is the drift
            # BETWEEN moments.
            ms_stdev=statistics.pstdev(p50s) if len(p50s) > 1 else 0.0,
            ms_min=min(t.ms_min for t in got),
            n_samples=sum(t.samples for t in got),
            config=dict(rig.config) if arm in LAUNCH_ARMS + ("fused",) else None,
            tile_config_source="vllm_observed" if arm in LAUNCH_ARMS + ("fused",)
            else "not_a_triton_tile",
            rows_total=rig.rows_total, padded_rows=rig.padded_rows,
            padding_source="histogram", active_experts=rig.active_experts,
            rel_err_vs_fused=rig.rel_err if arm == "gemm_up" else None,
            triton_artifacts=compiled if arm == "gemm_up" else 0)
        results[arm] = result.with_timing(representative_timing(got))
        store.write(results[arm], cell, meta)
    return results


def representative_timing(timings: list):
    """The repeat whose p50 IS the reported median, with the WORST flags of all.

    Two rules, and they pull in opposite directions on purpose.

    The instrument columns -- `iters`, `warmup_ms`, `trials`, `l2_flush`, the
    clock the card actually ran at -- come from the repeat whose p50 is the
    median, so the numbers describing the measurement describe the SAME
    measurement the row's `ms_median` came from. Averaging `iters` across
    repeats would put a count on the row that no trial ran.

    The three verdict flags are the worst across every repeat instead, and
    "worst" is spelled out per flag because the polarities differ:
    `clock_level_ok` and `clock_drift_ok` are False when something is wrong, so
    one False wins; `host_bound` is True when something is wrong, so one True
    wins. None survives only when no repeat determined the flag at all. A reader
    filtering rows on a clock verdict must not be shown the best of three, and a
    row that was host-bound in one repeat out of three is a row whose median is
    bounding the kernel from above.
    """
    if not timings:
        raise ValueError("representative_timing needs at least one repeat")
    order = sorted(timings, key=lambda t: t.ms_p50)
    chosen = order[(len(order) - 1) // 2]

    def worst(name: str, bad: bool) -> bool | None:
        seen = [getattr(t, name) for t in timings]
        if bad in seen:
            return bad
        return (not bad) if (not bad) in seen else None

    return dataclasses.replace(
        chosen, clock_level_ok=worst("clock_level_ok", False),
        clock_drift_ok=worst("clock_drift_ok", False),
        host_bound=worst("host_bound", True))


# --------------------------------------------------------------------------
# Persistence, and where a killed run picks up again.
# --------------------------------------------------------------------------

def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    Same order `scripts/run_all.sh` resolves it in. A pod's container disk dies
    with the pod and the network volume does not, so a results path that
    defaults to the checkout is a results path that defaults to being lost.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


def gitignore_note(path: Path) -> str:
    """Whether git would silently drop what this run writes, and by which rule.

    Asked and PRINTED rather than assumed. `.gitignore` carries `results/*` with
    only `!results/published/` excepted, and this repo has already lost every
    published figure to an unanchored rule. Knowing the output is ignored is
    fine; not knowing is how a result disappears between the pod and the commit.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-v", str(path)],
                              capture_output=True, text=True, timeout=10,
                              cwd=str(Path(__file__).resolve().parents[1]))
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not ask git whether this path is ignored: {exc}"
    if proc.returncode == 0 and proc.stdout.strip():
        rule = proc.stdout.strip().replace("\t", " ")
        return (f"GITIGNORED by [{rule}] -- expected for a raw run; publish with "
                f"scripts/publish_results.sh to get it into git")
    return "not gitignored"


def detect_card() -> str:
    """torch's name for the attached device, or `NO_CARD`.

    Read at run-id time, because the card belongs IN the id. Never raises: a
    driver present but unusable is not a card identity, and naming it `NO_CARD`
    keeps a laptop plan out of a real card's directory rather than aborting the
    plan that was the point of running on a laptop.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return NO_CARD
        return torch.cuda.get_device_name(0) or NO_CARD
    except Exception:                                     # noqa: BLE001
        return NO_CARD


def plan_run_id(card: str, models: list[str], tokens: list[int], dtype: str,
                routing: str, seed: int, reps: int, target_ms: float,
                warmup_ms: float, trials: int, l2_flush: bool,
                arms: list[str], densify: bool, self_test: str | None,
                self_test_noise: float) -> str:
    """A run id that is a HASH OF THE PLAN, so a rerun resumes by default.

    EVERY swept or pinned parameter is in the key, THE CARD FIRST. The failure
    this prevents is specific and this project has hit it: two settings deriving
    the same id, the second resuming the first, skipping every completed cell,
    and printing the first's numbers under the second's label.

    THE CARD WAS THE ONE THAT WAS MISSING, until 2026-09-02. It is not swept by
    this script; it is swept by the operator moving to another pod, while
    `results_root()` prefers `$MOE_RESULTS_DIR` and then `/workspace/results`,
    the network volume the runbook uses BECAUSE it outlives the pod. So a second
    card derived this run's id exactly, found every `(model, tokens, arm)`
    triple already on disk, measured nothing, and reported the first card's
    numbers. `Store` now carries the card in its resume key as well; that
    docstring says why one defence is not enough.

    BUILT BY `moe.bench.provenance.run_id` and not by a private hash here. That
    function raises `NoCard` on a missing card and `UnresolvedKnob` on a None or
    empty value, sorts the knobs before hashing so the id does not depend on the
    order they were named in, and puts the card slug at the FRONT where `ls`
    shows it. Three scripts had each re-implemented a subset of this and each
    had left a different knob out.

    THE THREE KNOBS THAT REPLACED `iters` AND `warmup` are here for the same
    reason those two were: they set the measured milliseconds of every cell.
    `target_ms` is the trial length the instrument sizes `iters` from,
    `warmup_ms` is warmup as a duration of delivered GPU load rather than a call
    count, and `l2_flush` decides whether every timed iteration starts with a
    cold L2. `--ridge` and `--bandwidth-gbps` stay OUT: they price the plan and
    touch no ratio, so two costings of one sweep belong in one directory.

    AND `--self-test` AND `--self-test-noise`, WHICH WERE THE OTHER HALF OF THE
    SAME DEFECT, together in the `case` knob `run_case` builds. They are not
    swept on a pod, they are swept on a laptop, and until 2026-09-02 they were
    not in the key: the `kernel`, `extent` and `neither` worlds all derived one
    id, so three runs wrote one `report.md` and one `summary.json` and the
    survivor was whichever ran last, labelled by its own `synthetic_world` field
    and by nothing else. Worse on the box, where the card is detected
    identically for a generated page and a measured one: the runbook's own
    off-GPU check line would have overwritten a measured run's report and
    summary IN PLACE, leaving a synthetic page beside a real `timings.csv` and
    carrying the real run's provenance block.
    """
    return PV.run_id(
        card=card, models=named_or_listed(models, DEFAULT_MODELS, "published4"),
        tokens=named_or_listed(tokens, DEFAULT_TOKENS, "pow2grid"),
        arms=named_or_listed(arms, ARM_ORDER, "all"),
        dtype=dtype, routing=routing, seed=seed, reps=reps, target=target_ms,
        warmup=warmup_ms, trials=trials, flush=l2_flush, densify=densify,
        case=run_case(self_test, self_test_noise))


def named_or_listed(chosen, default, name: str):
    """`name` when `chosen` is exactly `default`, else `chosen` spelled out.

    PURELY FOR THE VISIBLE PART OF THE ID, and it changes no information: the
    default set and its name are in bijection, so two different sets can still
    never derive one id. Without it `provenance.run_id` renders the four model
    names, fourteen token counts and nine arm names in sorted-key order and
    truncates at 96 characters, so every default run's directory is called
    `nocard-armsfused_align_gemm_up_act_gemm_down_...` and nothing a reader
    scanning `ls` needs survives the cut.
    """
    return name if tuple(chosen) == tuple(default) else tuple(chosen)


@dataclass(frozen=True)
class ResumeSurvey:
    """What is already on disk for this run id, and what of it is usable HERE.

    Computed without opening the file for writing, so `--dry-run` can print it
    and a plan can say what a resume would skip before anything is measured.
    """

    path: Path
    gpu_name: str
    #: Timed rows this card may restore.
    restorable: int
    #: Card as written in the row -> rows carried under it, for every card that
    #: is not this one.
    foreign: dict[str, int]
    #: Rows that recorded an error; they are retried rather than restored.
    errored: int

    @property
    def reason(self) -> str:
        """Why the foreign rows will not be restored, in one sentence, or ""."""
        if not self.foreign:
            return ""
        where = ", ".join(f"{name or '(no gpu_name recorded)'}: {n} row(s)"
                          for name, n in sorted(self.foreign.items()))
        return (f"{sum(self.foreign.values())} row(s) in {self.path.name} were "
                f"written on a different card and will NOT be restored on "
                f"{self.gpu_name or '(no card here)'} -- {where}. A millisecond "
                f"measured on another part is not a resumable row for this one, "
                f"and restoring one would print that card's timings under this "
                f"card's heading.")


def survey_resume(path: Path, gpu_name: str) -> ResumeSurvey:
    """Read an existing timings CSV and say what a resume would and would not use.

    Read-only and tolerant: a file from before the `gpu_name` resume key existed
    still parses, and its rows land under whatever card they recorded, which is
    exactly the comparison that has to happen.
    """
    restorable, errored = 0, 0
    foreign: dict[str, int] = {}
    if path.exists():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if "model" not in row or "arm" not in row:
                    continue
                if row.get("error"):
                    errored += 1
                    continue
                theirs = (row.get("gpu_name") or "").strip()
                if theirs == (gpu_name or "").strip():
                    restorable += 1
                else:
                    foreign[theirs] = foreign.get(theirs, 0) + 1
    return ResumeSurvey(path, gpu_name, restorable, foreign, errored)


@dataclass(frozen=True)
class MeasurementTally:
    """How many arms this process TIMED, against how many it restored from disk.

    THE FINDING THIS EXISTS FOR, and it is a finding about the SESSION DRIVER
    rather than about this script. The driver books two span arms, one bare and
    one `--densify`, 45 minutes and 35. `--densify` has been the default since
    2026-09-02 -- the sparse grid cannot answer C2 even in the world where C2 is
    true -- so both arms derive the SAME run id and the same directory, and the
    second restores every row the first wrote, measures nothing, and lands DONE
    in the ledger having spent zero minutes. The data is right; the ledger is
    not. The fix is one line in the driver, deleting the bare arm, and that file
    is not this slice's to edit.

    What IS this script's is refusing to be silent about it. A resumed run and a
    duplicate arm are indistinguishable from inside -- both are "the rows are
    already there", and a resume after a killed pod is a feature -- so this does
    not change the exit code, which would break the resume. It counts, says so
    in one unmissable line, and puts both numbers in `summary.json` where the
    ledger's own bookkeeping can read them.
    """

    measured: int
    restored: int
    path: Path

    @property
    def replay(self) -> bool:
        """Every arm came off disk and nothing was timed here."""
        return self.measured == 0 and self.restored > 0

    @property
    def note(self) -> str:
        """The one line, whichever case this is."""
        if self.replay:
            return (f"REPLAY: this run TIMED NOTHING. All {self.restored} "
                    f"arm(s) were restored from {self.path}, written by an "
                    f"earlier run with this run id on this card, and no GPU "
                    f"minute was spent here. That is correct for a resume after "
                    f"an interrupted run. If a session driver booked this as a "
                    f"SECOND arm expecting fresh minutes, its two arms derive "
                    f"one run id and only the first one spends anything")
        return (f"MEASURED: {self.measured} arm row(s) written by this run, "
                f"{self.restored} restored from {self.path}")


class Store:
    """Append-only CSV of arm results, flushed per arm, re-read on resume.

    Per arm and not per cell: the unit of loss on a killed pod should be the
    smallest thing that took real time, and here that is one arm's repeats.

    THE CARD IS IN THE KEY, and this is the second of two defences against one
    defect. Until 2026-09-02 the resume key was `(model, tokens, arm)` and the
    run id was card-free, so a second card pointed at the same
    `$MOE_RESULTS_DIR` -- which the runbook uses BECAUSE the network volume
    outlives the pod -- derived the same directory, found every triple present,
    measured ZERO cells, and printed the first card's timings under the second
    card's heading. Worse, the tile-config mismatch check then compared two
    restored first-card rows with each other and found them identical, so the
    one check that could have noticed passed.

    The first defence is `plan_run_id`, which now puts the card in the id. It is
    not enough on its own: `--run-id` overrides it, and an operator who pins one
    to resume a killed run gets the old behaviour back. So the key carries the
    card too, and a row from another card is counted and named rather than
    quietly skipped.
    """

    def __init__(self, path: Path, gpu_name: str, fresh: bool = False):
        self.path = path
        self.gpu_name = (gpu_name or "").strip()
        self.done: dict[tuple[str, str, int, str], dict] = {}
        #: Arms this process restored from disk, and arms it wrote itself. The
        #: pair is the whole content of `replay_note`: a run that restored every
        #: arm and wrote none spent no GPU minutes, whatever its exit code says.
        self.restored_arms = 0
        self.written_arms = 0
        #: Card -> rows on disk under it, for every card that is not this one.
        self.foreign: dict[str, int] = {}
        if fresh and path.exists():
            path.unlink()
        if path.exists():
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        theirs = (row.get("gpu_name") or "").strip()
                        key = (theirs, row["model"], int(row["num_tokens"]),
                               row["arm"])
                    except (KeyError, ValueError):
                        continue
                    if theirs != self.gpu_name:
                        self.foreign[theirs] = self.foreign.get(theirs, 0) + 1
                    self.done[key] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(ALL_CSV_COLUMNS),
                                      extrasaction="ignore")
        if new:
            self._writer.writeheader()
            self._fh.flush()

    @property
    def foreign_reason(self) -> str:
        """Why rows from another card were not restored, or "" when there are none."""
        return survey_resume(self.path, self.gpu_name).reason if self.foreign else ""

    def restore(self, key: tuple[str, int, str]) -> ArmResult | None:
        """Rebuild an ArmResult from a row THIS CARD wrote. Never another card's.

        `key` is `(model, tokens, arm)`; the card is the store's own, so a caller
        cannot ask for another one's row by accident. A row that carries an ERROR
        is not restored either: the common failures here are a pod that lost its
        device and an arm that ran out of memory behind a since-finished
        neighbour, and both are states a re-run can leave behind. A failure that
        is real fails again in milliseconds.
        """
        row = self.done.get((self.gpu_name, *key))
        if row is None or row.get("error"):
            return None
        self.restored_arms += 1

        def num(name, cast=float):
            try:
                return cast(row.get(name, ""))
            except (TypeError, ValueError):
                return None

        def flag(name) -> bool | None:
            # Three states round-trip: "True", "False" and "" for "not
            # determined". Anything else is also None, because a column that
            # says something this reader does not understand has not told it
            # the clock was fine.
            text = str(row.get(name, "")).strip()
            return {"True": True, "False": False}.get(text)

        config = {k: int(row[v]) for k, v in
                  (("BLOCK_SIZE_M", "block_m"), ("BLOCK_SIZE_N", "block_n"),
                   ("BLOCK_SIZE_K", "block_k"), ("GROUP_SIZE_M", "group_m"),
                   ("num_warps", "num_warps"), ("num_stages", "num_stages"))
                  if str(row.get(v, "")).strip() not in ("", "None")}
        return ArmResult(
            model=row["model"], num_tokens=int(row["num_tokens"]),
            arm=row["arm"], ms_median=num("ms_median"), ms_mean=num("ms_mean"),
            ms_stdev=num("ms_stdev"), ms_min=num("ms_min"),
            n_samples=num("n_samples", int) or 0, config=config or None,
            tile_config_source=row.get("tile_config_source", ""),
            rows_total=num("rows_total") or 0.0,
            padded_rows=num("padded_rows") or 0.0,
            padding_source=row.get("padding_source", ""),
            active_experts=num("active_experts", int) or 0,
            rel_err_vs_fused=num("rel_err_vs_fused"),
            triton_artifacts=num("triton_artifacts", int) or 0,
            ms_p90=num("ms_p90"), instrument=row.get("instrument", ""),
            warmup_ms=num("warmup_ms"), iters=num("iters", int),
            trials=num("trials", int),
            sm_clock_load_mhz=num("sm_clock_load_mhz"),
            clock_level_ok=flag("clock_level_ok"),
            clock_drift_ok=flag("clock_drift_ok"), l2_flush=flag("l2_flush"),
            host_bound=flag("host_bound"),
            host_enqueue_ms=num("host_enqueue_ms"),
            clock_note=row.get("clock_note", ""),
            host_note=row.get("host_note", ""),
            error=row.get("error", ""))

    def write(self, result: ArmResult, cell: Cell, meta: dict) -> None:
        row = result.row(cell, meta)
        self._writer.writerow(row)
        self._fh.flush()
        self.written_arms += 1
        self.done[(self.gpu_name, *result.key)] = row

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------

ROUTE_TEXT = """\
## The two routes, and which one this script takes

ROUTE B -- fuse the harness's own spans -- IS NOT REACHABLE. `moe/kernels/` holds
`.gitkeep` and `TEMPLATE.md`; the harness owns no fused kernel. Its own spans are
the `ref_*` torch stages and `torch_grouped_mm_*`, one canonical stage each. A
five-stage PIPELINE of them can be built today and would not answer the question:
it is five launches with every intermediate materialised, scored at 12.43x its
compulsory bytes against the fused span's 1.16x, so it differs from the fused
span in implementation QUALITY -- the very thing being controlled for. Writing the
fused kernel is the study's own open work.

ROUTE A -- run a fused implementation at a single-stage extent -- IS REACHABLE,
because vLLM's `fused_experts` is five launches and one of them is the Triton
grouped GEMM on its own:

    moe_align_block_size      the sort and the per-expert BLOCK_M padding
    invoke_fused_moe_kernel   the UP grouped GEMM, gather fused in       <- GEMM
    silu_and_mul              SwiGLU over the [gate | up] halves
    invoke_fused_moe_kernel   the DOWN grouped GEMM, combine applied     <- GEMM
    moe_sum                   the top-k reduction back to [T, H]

The one-launch arms hand `invoke_fused_moe_kernel` the config the fused call
RESOLVED and the alignment metadata built in an untimed prologue, so the kernel,
the tile and the data are all held fixed and only the extent moves."""

CORNER_TEXT = f"""\
## The 2x2, three corners of which exist

                    | Triton fused_moe_kernel  | CUTLASS grouped_mm
    five-launch     | `fused`      (published) | MISSING -- needs route B
    one GEMM launch | `gemm_*`     (NEW, here) | `cutlass_*` (published)

    separation = five_Triton / one_CUTLASS
               = (five_Triton / one_Triton) * (one_Triton / one_CUTLASS)
               =        EXTENT              *          KERNEL

Exact: the middle term cancels. The INTERPRETATION assumes no interaction --
that extent would cost CUTLASS what it costs Triton -- and that assumption is
exactly the missing corner. It is not measured here and is not claimed.

The predicted crossing cancels too. `ridge.crossing_batch_full` takes a MODEL,
not a span, so per model one prediction sits in both numerator and denominator.
Every ratio below is a ratio of two MEASURED crossings: no ridge, no byte model,
no calibration. Published separation to decompose: {PUBLISHED_SEPARATION} \
(first-crossing {PUBLISHED_SEPARATION_FIRST}, last-crossing {PUBLISHED_SEPARATION_LAST})."""

PREDICTIONS_TEXT = f"""\
## Predictions, registered before the run

THE CLAIM: the separation is the KERNEL, not the EXTENT.

MECHANISM, with its arithmetic. `moe_align_block_size` pads every expert to a
multiple of BLOCK_SIZE_M and the Triton kernel COMPUTES the padded rows;
`grouped_mm` takes ragged M and computes none. Padding inflates the compute side
by `p` and leaves weight traffic alone, so the crossing moves to `1/p` of its
predicted batch. At the published five-stage crossings:

    mixtral   T=316  r=79.0  -> 128 padded rows   p=1.62   1/p=0.62
    qwen2     T=787  r=98.4  -> 128               p=1.30   1/p=0.77
    v2-lite   T=931  r=87.3  -> 128               p=1.47   1/p=0.68
    dsv3      T=3010 r=94.1  -> 128               p=1.36   1/p=0.74
                                            mean  p=1.44   1/p=0.70

against a measured five-stage {PUBLISHED_FIVE_STAGE_RATIO} and one-stage \
{PUBLISHED_ONE_STAGE_RATIO}. The magnitude lands; the per-model ORDERING does
not, and C3 accepts the first without claiming the second.

The competing explanation is priced and found small. At mixtral's crossing the
permute, activation and combine traffic is ~62 MB against 2.8 GB of weights
(2.2%); at deepseek-v3's, ~1.07 GB against 22.5 GB (4.8%). Three extra launches
are flat in T, and a flat term FLATTENS a slope, pushing a crossing LATER.

VALIDITY -- a FAIL means no number on the page may be quoted.
  V0  the five launches recompute fused_experts' output   rel err <= {OUTPUT_REL_TOL:g}
  V1  the tile was READ OUT of the fused call             zero refusals
  V2  the launch times sum to the fused time              in {RECONSTRUCTION_BAND}
  V3  the placebo band is small                           p90 < {PLACEBO_BAND:.0%}
  V4  real work happened                                  cells/samples/compiles > 0
  V5  this session reproduces the published crossing      in {COMPARABILITY_BAND}x
  V6  up and down one-launch crossings agree              <= {ESTIMATOR_AGREEMENT_MAX:.1f}x

CLAIM -- a FAIL is a result, not a broken run.
  C1  extent does NOT move the crossing    EXTENT in {EXTENT_BAND}
  C2  the kernel carries the separation    KERNEL <= {KERNEL_MAX}
  C3  the mechanism is padding             |KERNEL - 1/p| <= {PADDING_TOLERANCE:.0%} of 1/p
  C4  the extra stages are cheap in time   non-GEMM share < {NON_GEMM_SHARE_MAX:.0%}
  C5  it is the published number           separation in \
[{SEPARATION_BAND[0]:.3f}, {SEPARATION_BAND[1]:.3f}]

EVERY CROSSING IS REPORTED AT BOTH ENDS OF THE STAIRCASE. The measured curve
crosses 0.5 more than once on half the canonical cells, and taking the last
instead of the first moves the published separation from 0.560 to 0.889. A
single number here would be a choice this script is not entitled to make."""

BOUND_TEXT = f"""\
## What the time-domain bound licenses, and what it does not

    EXTENT_time = fused / (gemm_up + gemm_down)          same kernel, two extents
    KERNEL_time = (gemm_up + gemm_down) / (cutlass_up + cutlass_down)

LICENSES: for any comparison of a five-stage MILLISECOND against a one-stage
millisecond at the same cell, the four extra stages account for at most the
non-GEMM share printed above. That is a statement about LEVELS and it holds
whether or not any crossing could be located.

DOES NOT LICENSE: anything about the {PUBLISHED_SEPARATION} separation. A
crossing is a property of the SLOPE, and a term that is 3% of the level can
still dominate the derivative -- which is precisely why the crossing
decomposition is the primary result and this is the fallback. The two are never
substituted for one another."""


def render_mde(cells: list[Cell]) -> list[str]:
    """The MDE line, and the two other things the plan can say before measuring.

    THE NOISE ASSUMPTION IS STATED, NOT MEASURED HERE, and the line says so. No
    arm of this experiment has a replicate spread yet, so it is sized against
    `PRIOR_ARM_SD_LOG`, the study's own upper-bound prior. An arm that quotes an
    effect smaller than this line has quoted noise.
    """
    models = len({c.model for c in cells})
    mde = mde_log_ratio(models)
    out = ["", "MDE, BEFORE ANYTHING RUNS. Every gate below is a threshold on a "
           "median log ratio over models, so the design has a smallest "
           "resolvable effect and it is stated here rather than discovered "
           "after the fact:"]
    out.append(f"  noise assumption  per-arm replicate sd {PRIOR_ARM_SD_LOG:.4f} "
               f"in log ms (scripts/replicate_noise_floor.py PRIOR_SD: the "
               f"s3-vs-s4 paired sd over 11 cells / sqrt(2), an UPPER bound, "
               f"and not this script's own arms)")
    if mde is None:
        out.append(f"  MDE               NOT COMPUTABLE at {models} model(s): a "
                   f"median over one model has no n to divide by, so this run "
                   f"can resolve nothing and its gates are descriptions of one "
                   f"curve")
        return out
    out.append(f"  MDE               {mde:.4f} in log terms, a factor of "
               f"{math.exp(mde):.3f}, at 5% two-sided and 80% power over "
               f"{models} models")
    for name, margin, what in (
            ("C1", abs(math.log(EXTENT_BAND[1])),
             f"the EXTENT band's upper edge {EXTENT_BAND[1]}"),
            ("C2", abs(math.log(KERNEL_MAX)), f"the KERNEL gate {KERNEL_MAX}"),
            ("C5", abs(math.log(SEPARATION_BAND[1] / PUBLISHED_SEPARATION)),
             "the C5 band's half width")):
        verdict = ("resolvable" if margin > mde else
                   "INSIDE THE MDE: this gate cannot distinguish its threshold "
                   "from no effect at this design")
        out.append(f"  {name} margin         {margin:.4f} in log terms "
                   f"({what}) -- {verdict}")
    out.append("  Read this as an upper bound on the design's power: the median "
               "is treated as a mean, the per-arm sd is propagated through a "
               "crossing estimator that can amplify it, and four fixed "
               "architectures are treated as four independent draws. "
               "`mde_log_ratio` names all three.")
    return out


def render_instrument_plan(cells: list[Cell], arms: list[str], *,
                           target_ms: float, warmup_ms: float, trials: int,
                           l2_flush: bool, ridge: float,
                           bandwidth_gbps: float) -> list[str]:
    """What the instrument will do to each arm, including where it clamps.

    BOTH SIDES OF THE CLAMP, because both are real and this plan used to print
    one. `iters_clamp` reads `[lo, hi]` off `timing.iters_for` itself. An arm
    modelled slower than `target_ms / lo` cannot fit `lo` calls into the budget
    and OVERRUNS it; an arm faster than `target_ms / hi` is capped at `hi` calls
    and UNDERRUNS it, delivering less measured kernel time than the budget
    bought and a wider interval than the plan implies. At the default
    `--target-ms 200` the underrun threshold is 0.1 ms and a third of the
    default grid sits below it, which the old text -- written against a stated
    clamp of 10000 that the shared function does not have -- put at 0.02 ms and
    therefore reported as nobody.

    Naming both here is cheaper than discovering either at minute 40 of a
    45-minute arm.
    """
    basis = timing_basis() or ("TIMING_BASIS -- torch is not importable here, so "
                               "the instrument cannot be named on this machine")
    out = ["", f"INSTRUMENT: {basis}",
           f"  warmup {warmup_ms:g} ms of delivered load per arm per repeat, "
           f"then {trials} trial(s) of about {target_ms:g} ms each, iters sized "
           f"by the instrument from the warmup's own queue-deep per-call time",
           f"  L2 flush {'ON' if l2_flush else 'OFF'}"
           + ("" if l2_flush else " -- see `time_arm` for why this arm, alone "
              "in the study, does not flush: V2 compares an isolated launch "
              "against the same launch inside a five-launch sequence")]
    lo, hi = iters_clamp()
    slow, fast, fastest = [], [], None
    for cell in cells:
        for arm in arms:
            ms = modelled_arm_ms(cell, arm, ridge=ridge,
                                 bandwidth_gbps=bandwidth_gbps)
            where = f"{cell.model} T={cell.num_tokens} {arm}"
            if ms * lo > target_ms:
                slow.append(f"{where} ~{ms:.1f} ms")
            elif ms * hi < target_ms:
                fast.append(f"{where} ~{ms:.3f} ms")
                fastest = ms if fastest is None else min(fastest, ms)
    out.append(f"  iters clamp [{lo}, {hi}], read off `timing.iters_for` "
               f"itself, so a trial holds {target_ms:g} ms of kernel time only "
               f"for an arm between {target_ms / hi:.3f} and "
               f"{target_ms / lo:.1f} ms per call")
    if slow:
        out.append(f"  {len(slow)} arm(s) are modelled slower than "
                   f"{target_ms:g} ms / {lo}, the instrument's minimum of {lo} "
                   f"calls per trial, so their trials will OVERRUN the budget: "
                   + "; ".join(slow[:4])
                   + (f"; and {len(slow) - 4} more" if len(slow) > 4 else ""))
    else:
        out.append(f"  no arm is modelled slower than {target_ms:g} ms / {lo}, "
                   f"so no trial should overrun the budget")
    if fast:
        out.append(f"  {len(fast)} of {len(cells) * len(arms)} arm(s) are "
                   f"modelled faster than {target_ms:g} ms / {hi}, the "
                   f"instrument's maximum of {hi} calls per trial, so their "
                   f"trials UNDERRUN the budget -- the fastest at "
                   f"{fastest:.3f} ms delivers about "
                   f"{fastest * hi:.1f} ms of kernel time per trial, not "
                   f"{target_ms:g}, and the cost estimate above OVER-charges "
                   f"them: " + "; ".join(fast[:4])
                   + (f"; and {len(fast) - 4} more" if len(fast) > 4 else ""))
    else:
        out.append(f"  no arm is modelled faster than {target_ms:g} ms / {hi}, "
                   f"so no trial should underrun the budget")
    return out


def render_plan(cells: list[Cell], arms: list[str], notes: list[str],
                seconds: float) -> str:
    lines = ["## The plan", ""]
    for note in notes:
        lines.append(f"  dropped -- {note}")
    if notes:
        lines.append("")
    lines.append("| arm | what it times |")
    lines.append("|---|---|")
    for arm in arms:
        lines.append(f"| `{arm}` | {ARM_DESCRIPTIONS[arm]} |")
    lines += ["",
              "| model | T | rows/expert | ladder BLOCK_M | padded rows | p (lower bound) |",
              "|---|---:|---:|---:|---:|---:|"]
    for cell in cells:
        cfg = cell.cfg
        bm = ladder_block_m(cell.num_tokens, cfg.num_experts, cell.dtype)
        rows = cell.num_tokens * cfg.top_k
        padded = padded_rows_saturated(cfg, cell.num_tokens, bm)
        p = padding_factor(padded, rows)
        lines.append(f"| {cell.model} | {cell.num_tokens} | "
                     f"{cell.rows_per_expert:.1f} | {bm} | {padded:.0f} | "
                     + ("--" if p is None else f"{p:.2f}") + " |")
    lines += ["",
              f"{len(cells)} cells x {len(arms)} arms = {len(cells) * len(arms)} "
              f"timed arms.",
              "BLOCK_M and p above are DERIVED from vLLM's fallback ladder for "
              "the plan only; on the box both are OBSERVED, the block size out "
              "of the fused call and p out of the real routing histogram, and "
              "the CSV records which through `padding_source`.",
              "p above is a LOWER BOUND twice over. It assumes every expert "
              "holds exactly `T k / E` rows, and under sampled uniform routing "
              "the counts scatter, so `sum_e ceil(n_e/BM) BM` exceeds "
              "`E ceil(r/BM) BM`. And on a powers-of-two grid r is a power of "
              "two and BM is too, so p reads exactly 1.00 wherever r >= BM -- "
              "the crossings sit BETWEEN those points, at r near 80-100, which "
              "is why C3 evaluates p at the interpolated crossing rather than "
              "at a grid cell.",
              f"Estimated KERNEL time {seconds:.0f} s at the byte model's own "
              f"timings. That is a PREDICTION from arithmetic, not a "
              f"measurement, and it is NOT the wall clock."]
    # The wall clock is dominated by what the timing estimate excludes, so the
    # estimate is printed next to the things that dwarf it rather than alone. A
    # cost line that says 34 s for a session that takes half an hour is worse
    # than no cost line.
    configs = {}
    for cell in cells:
        cfg = MODEL_CONFIGS[cell.model]
        key = tuple(sorted(ladder_config(cell.num_tokens, cfg.num_experts,
                                         cell.dtype).items()))
        configs.setdefault(cell.model, set()).add(key)
    distinct = sum(len(v) for v in configs.values())
    heaviest = max(cells, key=lambda c: (c.cfg.num_experts * 3
                                         * c.cfg.intermediate_size
                                         * c.cfg.hidden_size))
    gb = (heaviest.cfg.num_experts * 3 * heaviest.cfg.intermediate_size
          * heaviest.cfg.hidden_size * dtype_bytes(heaviest.dtype) / 1e9)
    lines += ["",
              f"WALL CLOCK IS NOT THAT NUMBER. On top of it: up to {distinct} "
              f"distinct Triton specialisations to compile across "
              f"{len(configs)} models (a fused_moe compile is tens of seconds, "
              f"and the ladder changes the config four times over this token "
              f"grid), and one weight build per model, the largest being "
              f"{heaviest.model} at {gb:.1f} GB. Budget for those, not for the "
              f"timings."]
    lines += ["", "C3's POWER ON THIS GRID, before anything runs. The mechanism "
              "is only visible where the padding factor is materially above 1 "
              "on the COMPUTE-BOUND side, and the null it is tested against is "
              f"p = 1.00 inside a {PADDING_TOLERANCE:.0%} band:"]
    for model in dict.fromkeys(c.model for c in cells):
        cfg = MODEL_CONFIGS[model]
        anchor = PUBLISHED_FIVE_STAGE_CROSSING.get(model)
        above = [c for c in cells if c.model == model and anchor
                 and c.num_tokens >= 2 * anchor]
        if not above:
            lines.append(f"  {model}: no planned cell above 2x its published "
                         "crossing, so C3 will report UNKNOWN")
            continue
        cell = min(above, key=lambda c: c.num_tokens)
        bm = ladder_block_m(cell.num_tokens, cfg.num_experts, cell.dtype)
        p_low = modelled_padding_at(cfg, float(cell.num_tokens), bm)
        verdict = ("has power" if p_low and abs(1 - p_low) > PADDING_TOLERANCE * p_low
                   else "NO POWER at the balanced lower bound; the routing "
                        "scatter may still supply some, and --densify supplies "
                        "more")
        lines.append(f"  {model}: first compute-bound cell T={cell.num_tokens}, "
                     f"BLOCK_M {bm}, p >= "
                     + ("--" if p_low is None else f"{p_low:.3f}")
                     + f" -- {verdict}")
    return "\n".join(lines)


def render_crossings(analysis: Analysis) -> str:
    lines = ["## Crossings, per model, at both ends of the staircase", "",
             "| model | five-launch Triton | one-launch Triton | one-launch "
             "CUTLASS | EXTENT | KERNEL | separation |",
             "|---|---|---|---|---:|---:|---:|"]
    for dec in analysis.per_model:
        if dec.excluded:
            lines.append(f"| {dec.model} | EXCLUDED: {dec.excluded} | | | | | |")
            continue

        def ends(cross):
            if not cross.tokens:
                return "--"
            if len(cross.tokens) == 1:
                return f"{cross.tokens[0]:.0f}"
            return f"{cross.first:.0f} .. {cross.last:.0f} ({len(cross.tokens)})"

        for end in ("first", "last"):
            extent, kernel, sep = dec.factors(end)
            label = dec.model if end == "first" else ""
            lines.append(
                f"| {label} ({end}) | "
                + " | ".join([ends(dec.five), ends(dec.one_triton),
                              ends(dec.one_cutlass)])
                + " | " + " | ".join(
                    "--" if v is None else f"{v:.3f}"
                    for v in (extent, kernel, sep)) + " |")
    lines.append("")
    for end in ("first", "last"):
        med = analysis.medians[end]
        pieces = []
        for name in ("extent", "kernel", "separation"):
            value = med.get(name)
            interval = analysis.intervals[end].get(name)
            text = "--" if value is None else f"{value:.3f}"
            if interval:
                text += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
            pieces.append(f"{name} {text}")
        lines.append(f"  {end:>5} crossing, median over models: " + ",  ".join(pieces))
    for end in ("first", "last"):
        e_share = analysis.shares[end].get("extent")
        k_share = analysis.shares[end].get("kernel")
        if e_share is None or k_share is None:
            continue
        lines.append(f"  {end:>5}: of the separation, EXTENT carries "
                     f"{e_share:.0%} and KERNEL {k_share:.0%} in log terms. "
                     f"Taken per model and then medianed, because the three "
                     f"factors are exact per model and the MEDIANS of them are "
                     f"not: median(EXTENT) x median(KERNEL) is not "
                     f"median(separation) on a ragged set of models. The "
                     f"interaction is the missing CUTLASS five-launch corner "
                     f"and is in NEITHER share.")
    return "\n".join(lines)


def render_budget(analysis: Analysis) -> str:
    lines = ["## The time budget: what a five-launch span is made of", "",
             "| model | T | fused ms | align | gemm_up | act | gemm_down | sum "
             "| sum/fused | non-GEMM | p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for budget in analysis.budgets:
        if budget.fused_ms is None:
            continue
        cols = []
        for arm in LAUNCH_ARMS:
            value = budget.launch_ms.get(arm)
            cols.append("--" if value is None else f"{value:.4f}")
        recon = budget.reconstruction
        share = budget.non_gemm_share
        lines.append(
            f"| {budget.cell.model} | {budget.cell.num_tokens} | "
            f"{budget.fused_ms:.4f} | " + " | ".join(cols) + " | "
            + ("--" if recon is None else f"{recon:.3f}") + " | "
            + ("--" if share is None else f"{share:.1%}") + " | "
            + ("--" if budget.padding is None else f"{budget.padding:.2f}") + " |")
    lines.append("")
    lines.append(f"  {analysis.arms_timed} arms carry a timing over "
                 f"{analysis.cells_measured} cells with a fused arm")
    if analysis.reconstruction_median is not None:
        lines.append(f"  median (sum of five launches) / fused: "
                     f"{analysis.reconstruction_median:.3f}")
    if analysis.extent_time_median is not None:
        lines.append(f"  EXTENT_time, fused / (gemm_up + gemm_down), median over "
                     f"cells: {analysis.extent_time_median:.3f}")
    if analysis.kernel_time_median is not None:
        lines.append(f"  KERNEL_time, Triton GEMMs / CUTLASS GEMMs, median over "
                     f"cells: {analysis.kernel_time_median:.3f}")
    if analysis.non_gemm_share_at_crossing is not None:
        lines.append(f"  non-GEMM share at the crossing cells: "
                     f"{analysis.non_gemm_share_at_crossing:.1%}")
    if analysis.padding_measured_near_crossing is not None:
        lines.append(f"  padding factor MEASURED from the routing histogram at "
                     f"the crossing cells: "
                     f"{analysis.padding_measured_near_crossing:.3f}")
    lines.append("  p is MEASURED per cell from the real routing histogram "
                 "(`padding_source=histogram` in the CSV), never from the "
                 "balanced formula, which is a lower bound.")
    return "\n".join(lines)


def render_headline(analysis: Analysis, gates: list[Gate]) -> str:
    """The answer in a full sentence, with both factors named and the caveat."""
    med = analysis.medians["first"]
    extent, kernel, sep = (med.get("extent"), med.get("kernel"),
                           med.get("separation"))
    if extent is None or kernel is None:
        return ("## Headline\n\nNo crossing decomposition was produced. That is "
                "not a null result -- it means the grid did not bracket a "
                "crossing on all three curves for enough models. What survives "
                "is the time-domain bound below, and it answers a different "
                "question: see the licensing note.")
    void = [g.name for g in gates if g.kind == "VALIDITY" and g.passed is not True]
    warn = ("" if not void else
            f"\n\nDO NOT QUOTE ANY OF THIS. Validity gates {', '.join(void)} did "
            f"not pass, and each names what it invalidates.")
    e_share = analysis.shares["first"].get("extent")
    k_share = analysis.shares["first"].get("kernel")
    return (
        f"## Headline\n\n"
        f"On {len(analysis.decomposed)} models, first crossing:\n\n"
        f"    separation  {sep:.3f}   (published {PUBLISHED_SEPARATION_FIRST})\n"
        f"    EXTENT      {extent:.3f}   the SAME Triton kernel, five launches "
        f"over one\n"
        f"    KERNEL      {kernel:.3f}   one launch, Triton over CUTLASS\n\n"
        f"In words: of the separation between a five-stage span and a one-stage "
        f"span, "
        + (f"EXTENT carries {e_share:.0%} and KERNEL {k_share:.0%} in log terms"
           if e_share is not None and k_share is not None
           else "the two factors multiply to it exactly")
        + ". The decomposition is EXACT -- the middle term cancels -- but reading "
          "it as two separable causes assumes the extent effect would be the "
          "same for CUTLASS as for Triton, and that is the missing fourth "
          "corner of the 2x2. It is not measured here." + warn)


def provenance_text(prov) -> str:
    """The provenance block as a printable section.

    Printed at the TOP of every page, before any number, and written into
    `summary.json` as a nested block plus the five top-level keys the audit's
    publish gate checks. None means "could not determine" and the reason is
    beside it; a laptop plan therefore says `gpu_name: null` with
    `no CUDA device` next to it rather than leaving the field out.
    """
    block = prov.as_dict()
    missing = block.pop("missing", {})
    lines = ["## Provenance", ""]
    for key, value in block.items():
        note = missing.get(key)
        lines.append(f"  {key:20s} "
                     + ("--" if value is None else str(value))
                     + (f"   ({note})" if note else ""))
    return "\n".join(lines)


def render_report(header: str, analysis: Analysis, gates: list[Gate],
                  stopped: str = "", self_gates: list[Gate] | None = None) -> str:
    """The exact text written to report.md, assembled in one testable place."""
    body = "\n\n".join([
        header,
        render_crossings(analysis),
        render_budget(analysis),
        BOUND_TEXT,
        render_headline(analysis, gates),
        "## Gates\n\n```\n" + render_gates(gates) + "\n```",
        f"## Intervals\n\n{analysis.interval_scope}",
    ])
    if self_gates:
        body += ("\n\n## Self-test gates: did this world produce its registered "
                 "verdicts?\n\n```\n" + render_gates(self_gates) + "\n```")
    if stopped:
        body += f"\n\nPARTIAL RUN: {stopped}."
    if analysis.failures:
        body += ("\n\n## Arms that produced no timing\n\n"
                 + "\n".join(f"- {line}" for line in analysis.failures))
    return body


# --------------------------------------------------------------------------
# Environment, argument parsing, main.
# --------------------------------------------------------------------------

def missing_gpu_stack() -> str:
    """Which half of the stack is absent, and what to run instead.

    One function so a laptop gets the same message the pod would, and so the
    test suite can assert on it without a GPU.
    """
    try:
        import torch
    except ImportError:
        return ("no torch on this machine. --self-test kernel runs the whole "
                "analysis off GPU; --dry-run prints the plan.")
    if not torch.cuda.is_available():
        return ("no CUDA device. --self-test kernel runs the whole analysis off "
                "GPU; --dry-run prints the plan.")
    try:
        import vllm  # noqa: F401
    except ImportError:
        return ("vLLM is not importable here, and it owns both the five-launch "
                "span and the one-launch kernel this script compares.\n"
                "On the pod: source the vllm venv (scripts/setup_runpod.sh "
                "vllm). Off GPU: --self-test kernel.")
    return ""


def environment_meta() -> dict:
    meta = {"gpu_name": "", "torch_version": "", "triton_version": "",
            "vllm_version": ""}
    with contextlib.suppress(Exception):
        import torch
        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
    with contextlib.suppress(Exception):
        import triton
        meta["triton_version"] = getattr(triton, "__version__", "")
    with contextlib.suppress(Exception):
        import vllm
        meta["vllm_version"] = getattr(vllm, "__version__", "")
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tokens", default=",".join(str(t) for t in DEFAULT_TOKENS),
                    help="MATCHED to the published grid on purpose; a denser "
                         "grid moves the crossing for reasons unrelated to the "
                         "session, which is what V5 is checking")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--routing", default="uniform",
                    help="the regime `2R/b` is defined for and the one the "
                         "published separation was restricted to")
    ap.add_argument("--arms", default="all",
                    help=f"comma list from {','.join(ARM_ORDER)}, or 'all'")
    ap.add_argument("--reps", type=int, default=3,
                    help="round-robin repeats; arms are interleaved inside each")
    ap.add_argument("--target-ms", type=float, default=200.0,
                    help="how long one trial of one arm should take; the "
                         "instrument sizes iters from it. Replaces --iters, "
                         "which set a call count the instrument now owns")
    ap.add_argument("--warmup-ms", type=float, default=200.0,
                    help="milliseconds of DELIVERED GPU load before timing, per "
                         "arm per repeat. Replaces --warmup, which counted "
                         "calls; there is no honest conversion between the two, "
                         "which is why the flag was replaced and not reused")
    ap.add_argument("--trials", type=int, default=3,
                    help="trials inside one timed arm, each one synchronize")
    ap.add_argument("--l2-flush", action="store_true",
                    help="flush the L2 before every timed call. OFF by default "
                         "here and nowhere else in the study: see `time_arm`. "
                         "It is in the run id, so a flushed and an unflushed "
                         "run can never share a directory")
    ap.add_argument("--reference-clock-mhz", type=float, default=None,
                    help="the SM clock the roof was measured at, if you want "
                         "the LEVEL flag scored. Without it clock_level_ok is "
                         "None on every row, which means NOT DETERMINED; this "
                         "script scores no ratio against the roof, so DRIFT is "
                         "the flag that matters here and it needs no reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="COST MODEL ONLY; no ratio in this script touches it. "
                         "FLOP/byte. 0 (the default) reads the attached "
                         "device's own calibration and REFUSES if there is "
                         "none; with no device it prices against the committed "
                         "H200 calibration and labels that a HYPOTHESIS. It "
                         "used to default to 160.3, a withdrawn figure that "
                         "was no card's ridge")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0,
                    help="COST MODEL ONLY, same reason and same resolution; "
                         "0 (the default) reads the same calibration as --ridge")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-id", default=None,
                    help="defaults to a hash of the plan, so re-running the "
                         "same command RESUMES rather than starting over")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and overwrite any rows already on disk")
    ap.add_argument("--densify", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="cells around each model's PUBLISHED crossing, at "
                         "80-200 rows per expert where the padding factor is "
                         "1.3-1.8. ON BY DEFAULT since 2026-09-02: on the "
                         "powers-of-two grid alone the padding factor is "
                         "exactly 1.00 at every point, C2 FAILS in the world "
                         "where its own hypothesis is TRUE, and the kernel and "
                         "neither worlds produce the identical claim-gate row. "
                         "--no-densify keeps the published grid, which V5's "
                         "grid-matching argument slightly prefers, and the run "
                         "then REFUSES before measuring rather than spending 45 "
                         "minutes to reach a FAIL that means nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the routes, the plan, the predictions and the "
                         "cost; measure nothing")
    ap.add_argument("--self-test", choices=sorted(WORLDS), default=None,
                    help="generate every arm time from the model in this world "
                         "and run the WHOLE analysis on it, off GPU")
    ap.add_argument("--self-test-noise", type=float, default=0.0,
                    help="lognormal sigma applied to every synthetic time")
    ap.add_argument("--fail-on-world", action="store_true",
                    help="score the S gates -- did this world produce the "
                         "verdicts registered for it in WORLD_EXPECTATIONS -- "
                         "and exit through moe.bench.exit_codes.classify over "
                         "them. Without it a --self-test exits REFUSED, because "
                         "a self test measured no device and its twelve gates "
                         "are a world's planted answers rather than results")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this long and report what exists")
    return ap


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    tokens = sorted({int(t) for t in args.tokens.split(",") if t})
    arms = list(ARM_ORDER) if args.arms == "all" else [
        a for a in ARM_ORDER if a in set(args.arms.split(","))]
    for required in ("fused", "gemm_up", "gemm_down"):
        if required not in arms:
            raise SystemExit(
                f"--arms must include {required}: the three corners of the 2x2 "
                "are the whole experiment, and dropping one leaves a comparison "
                "with nothing to separate. Nothing measured.")

    cells, notes = plan_cells(models, tokens, args.dtype, args.densify)
    card = detect_card()
    run_id = args.run_id or plan_run_id(
        card, models, tokens, args.dtype, args.routing, args.seed, args.reps,
        args.target_ms, args.warmup_ms, args.trials, args.l2_flush, arms,
        args.densify, args.self_test, args.self_test_noise)
    out_dir = ((args.out_dir or (results_root() / "span_extent_separation"))
               / run_id)
    csv_path, report_path = out_dir / "timings.csv", out_dir / "report.md"
    seconds = estimated_seconds(cells, arms, reps=args.reps,
                                target_ms=args.target_ms,
                                warmup_ms=args.warmup_ms, trials=args.trials)
    gpu_name = "" if card == NO_CARD else card
    resume = survey_resume(csv_path, gpu_name)
    ceilings = resolve_cost_ceilings(args.ridge, args.bandwidth_gbps, card,
                                     args.dtype)
    args.ridge, args.bandwidth_gbps = ceilings.ridge, ceilings.bandwidth_gbps
    prov = PV.provenance_block(
        instrument=timing_basis(), ridge=ceilings.ridge,
        ridge_source=ceilings.ridge_source,
        bandwidth=ceilings.bandwidth_gbps,
        bandwidth_source=ceilings.bandwidth_source,
        warmup_ms=args.warmup_ms, target_ms=args.target_ms)

    header = "\n\n".join([
        "# Span extent or kernel quality? Decomposing the 0.563 separation",
        (f"run id {run_id}\n"
         f"card   {card}"
         + ("   (no CUDA device: a plan or a replay, never a measurement)"
            if card == NO_CARD else "")
         + f"\ndtype {args.dtype}   routing {args.routing}   seed {args.seed}\n"
         f"reps {args.reps}, round-robin over {len(arms)} arms\n\n"
         f"EVERYTHING IS SAVED TO  {out_dir}\n"
         f"  rows   {csv_path}\n"
         f"  report {report_path}\n"
         f"  {gitignore_note(out_dir)}\n"
         "Re-run the same command to resume; completed (card, model, tokens, "
         "arm) rows are skipped.\n"
         f"RESUME: {resume.restorable} row(s) on disk for this card, "
         f"{resume.errored} errored row(s) that will be retried"
         + (f"\n  {resume.reason}" if resume.reason else "")),
        provenance_text(prov),
        ROUTE_TEXT,
        CORNER_TEXT,
        render_plan(cells, arms, notes, seconds),
        "\n".join(render_instrument_plan(
            cells, arms, target_ms=args.target_ms, warmup_ms=args.warmup_ms,
            trials=args.trials, l2_flush=args.l2_flush, ridge=args.ridge,
            bandwidth_gbps=args.bandwidth_gbps) + render_mde(cells)),
        PREDICTIONS_TEXT,
    ])
    print(header)

    # CAN THIS GRID ANSWER, before a single millisecond is spent. See
    # `c2_grid_power`: on the powers-of-two grid alone C2 cannot PASS even in
    # the world where its hypothesis is true, so a FAIL from the pod would carry
    # no information and 45 minutes would buy nothing.
    power = c2_grid_power(cells, ridge=args.ridge,
                          bandwidth_gbps=args.bandwidth_gbps, seed=args.seed)
    print(f"\nGRID POWER: {power.detail}")

    if args.dry_run:
        print("\n".join([
            "", "=" * 72,
            "NOT A RESULT. Nothing was measured.",
            "  reason: --dry-run was given",
            "  Everything above is arithmetic over vLLM's config ladder and the",
            "  byte model. It says what WOULD be compared and what it would",
            "  cost. It does not say what anything is."
            + ("" if power.can_answer else
               f"\n  AND THIS GRID WOULD REFUSE: {power.remedy}"),
            "=" * 72]))
        return exit_codes.REFUSED

    if not power.can_answer:
        print("\n".join([
            "", "=" * 72,
            f"REFUSED: {power.remedy}",
            "  Nothing was measured and nothing was spent. This grid gives the",
            "  primary claim gate one reachable answer, and the `kernel` and",
            "  `neither` worlds the identical claim-gate row, so a verdict from",
            "  it would not be evidence about which world we are in.",
            "=" * 72]))
        return exit_codes.REFUSED

    stopped = ""
    self_gates: list[Gate] = []
    # The self-test path times nothing and restores nothing; an empty tally is
    # the true statement about it, and `synthetic_world` in the summary is what
    # says the page is generated.
    tally = MeasurementTally(0, 0, csv_path)
    if args.self_test is not None:
        world = WORLDS[args.self_test]
        solved = ""
        if world.name == "extent":
            solve = extent_scale_for_separation(
                cells, PUBLISHED_SEPARATION_FIRST, ridge=args.ridge,
                bandwidth_gbps=args.bandwidth_gbps, seed=args.seed)
            world = World(world.name, solve.scale, world.triton_pads,
                          world.cutlass_pads, world.summary)
            if solve.reached:
                solved = (f"\n  SOLVED: for span extent ALONE to produce "
                          f"{PUBLISHED_SEPARATION_FIRST}, the four extra stages "
                          f"would have to cost {solve.scale:.0f}x what the byte "
                          f"model says they cost. Compare that against their "
                          f"measured share in the budget below.")
            else:
                solved = (
                    f"\n  NOT REACHED. No inflation up to 1e5x drives the "
                    f"separation to {PUBLISHED_SEPARATION_FIRST} while every "
                    f"model still has a crossing to read. The floor is "
                    f"{solve.floor:.3f} at {solve.scale:.0f}x, and the world "
                    f"below is generated there. Read that as an ESTIMATOR limit "
                    f"as much as a physical one: past this inflation the fused "
                    f"curve is linear from the first grid point, its slope never "
                    f"rises THROUGH 0.5, and there is no upcrossing to find.")
        results = synthetic_results(cells, world, ridge=args.ridge,
                                    bandwidth_gbps=args.bandwidth_gbps,
                                    noise=args.self_test_noise, seed=args.seed)
        print(f"\nSELF TEST, world '{world.name}': every arm time was GENERATED "
              f"from the byte model.\n  {world.summary}{solved}\n"
              "Nothing here was measured. The gates below are being run against "
              "a world we constructed, which tests the gates and not the "
              "hardware. A world that reproduces the published separation shows "
              "its mechanism is SUFFICIENT in the byte model; only the "
              "measurement can say whether it is what happens.")
    else:
        missing = missing_gpu_stack()
        if missing:
            print(f"\nREFUSED: {missing}")
            return exit_codes.REFUSED
        results, stopped, tally = run_measurement(
            cells, arms, args, out_dir, csv_path, run_id, gpu_name, prov)
        if results is None:
            # `find_pieces` could not locate a launch of vLLM's fused path, or
            # its signature drifted. Nothing was measured and nothing was spent,
            # which is REFUSED and not INVALID; the two used to be the same
            # integer here and the driver read it as a retry.
            return exit_codes.REFUSED

    analysis = analyse(cells, results)
    gates = build_gates(analysis)
    if args.self_test is not None:
        self_gates = self_test_gates(world, gates)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(header, analysis, gates, stopped, self_gates) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(prov.stamp({
        "run_id": run_id,
        "card": card,
        "synthetic_world": args.self_test,
        "published_separation": PUBLISHED_SEPARATION,
        "definition": ("separation = five-launch Triton crossing / one-launch "
                       "CUTLASS crossing; EXTENT = five/one both Triton; "
                       "KERNEL = one-launch Triton / one-launch CUTLASS; "
                       "EXTENT * KERNEL == separation exactly"),
        "medians": analysis.medians,
        "log_shares": analysis.shares,
        "intervals": {end: {k: (list(v) if v else None) for k, v in inner.items()}
                      for end, inner in analysis.intervals.items()},
        # THE SCOPE AND THE n TRAVEL WITH THE NUMBERS. An interval in a JSON
        # file with neither beside it is a pair of floats a reader will treat as
        # an error bar; at n=2, which `MIN_DECOMPOSED_MODELS` permits, it is a
        # restatement of two points.
        "interval_scope": analysis.interval_scope,
        "interval_n": analysis.interval_n,
        "mde_log_ratio": mde_log_ratio(analysis.interval_n),
        "mde_noise_assumption": (
            f"per-arm replicate sd {PRIOR_ARM_SD_LOG:.4f} in log ms, an UPPER "
            f"bound imported from scripts/replicate_noise_floor.py PRIOR_SD; "
            f"not measured from this run's own arms"),
        "instruments": analysis.instruments,
        # THE RULER, AND THE ROOF ANSWER, BESIDE THE NUMBERS. `instruments` is
        # the column; `rulers` is the column together with the flush state, and
        # it is what V7 scores. `roof_comparable` is False here by design and is
        # published as False so a consumer applying `timing.py`'s "carries
        # TIMING_BASIS, therefore roof-comparable" rule to these rows finds the
        # contradiction in the file rather than in a docstring.
        "rulers": analysis.rulers,
        "roof_comparable": roof_comparable(analysis.rulers, timing_basis()),
        "roof_comparable_note": (
            "the roof was measured with the L2 flush ON and these rows are "
            "timed with it OFF (see `time_arm`: V2 compares an isolated launch "
            "against the same launch inside a five-launch sequence, and "
            "flushing charges the isolated arm a cold cache the fused path "
            "never pays). Every ratio on this page is measured over measured "
            "and the predicted crossing cancels, so no number here is scored "
            "against the roof"),
        "clock_level_bad_arms": analysis.clock_level_bad,
        "clock_drift_bad_arms": analysis.clock_drift_bad,
        "host_bound_arms": analysis.host_bound_arms,
        "extent_time_median": analysis.extent_time_median,
        "kernel_time_median": analysis.kernel_time_median,
        "reconstruction_median": analysis.reconstruction_median,
        "non_gemm_share_at_crossing": analysis.non_gemm_share_at_crossing,
        "padding_measured_near_crossing": analysis.padding_measured_near_crossing,
        "padding_probe": {
            "median_contrast_over_prediction": analysis.probe.median_ratio,
            "powered_models": [m.model for m in analysis.probe.powered],
            "per_model": [{
                "model": m.model, "contrast": m.contrast,
                "measured_padding_above": m.padding_compute,
                "kernel_time_above": m.kernel_time_compute,
                "kernel_time_below": m.kernel_time_memory,
                "cells_above": m.compute_cells, "cells_below": m.memory_cells,
                "has_power": m.discriminates,
            } for m in analysis.probe.per_model],
        },
        "cells_measured": analysis.cells_measured,
        "arms_timed": analysis.arms_timed,
        # `arms_timed` counts the rows the ANALYSIS read, restored ones
        # included. These two count what this PROCESS did, which is the pair a
        # ledger needs to tell a run that spent 45 minutes from one that spent
        # none. See `MeasurementTally`.
        "arms_measured_here": tally.measured,
        "arms_restored": tally.restored,
        "replay": tally.replay,
        "refusals": analysis.refusals,
        "models_decomposed": [d.model for d in analysis.decomposed],
        "models_excluded": {d.model: d.excluded for d in analysis.per_model
                            if d.excluded},
        "missing_corner": ("CUTLASS at a five-launch extent; it needs the fused "
                           "harness kernel that does not exist, so the "
                           "interaction between extent and kernel is not "
                           "measured"),
        "partial": stopped,
        "gates": [{"name": g.name, "kind": g.kind, "token": g.token,
                   "passed": g.passed, "verdict": g.tag, "rule": g.rule,
                   "observed": g.observed, "invalidates": g.invalidates}
                  for g in gates],
        "self_test_gates": [{"name": g.name, "kind": g.kind, "token": g.token,
                             "passed": g.passed, "verdict": g.tag,
                             "rule": g.rule, "observed": g.observed}
                            for g in self_gates],
        "world_expectations": WORLD_EXPECTATIONS,
    }), indent=2))

    print("\n" + render_crossings(analysis))
    print("\n" + render_budget(analysis))
    print("\n" + BOUND_TEXT)
    print("\n" + render_headline(analysis, gates))
    print("\n## Gates\n")
    print(render_gates(gates))
    if self_gates:
        print("\n## Self-test gates: did this world produce its registered "
              "verdicts?\n")
        print(render_gates(self_gates))
    if stopped:
        print(f"\nPARTIAL RUN: {stopped}.")
    if tally.replay:
        # Twice on purpose, and the second time at the bottom, because the
        # bottom of stdout is what a human reads and what a driver logs.
        print("\n" + "=" * 72 + f"\n{tally.note}.\n" + "=" * 72)
    print(f"\nEVERYTHING IS SAVED TO {out_dir}")
    print(f"  rows    {csv_path}\n  report  {report_path}\n"
          f"  summary {out_dir / 'summary.json'}\n  {gitignore_note(out_dir)}")

    # WHICH GATES DECIDE THE EXIT, and there are two answers because there are
    # two kinds of run.
    #
    # A POD RUN is scored on its own twelve gates through
    # `exit_codes.classify`: a VALIDITY gate that did not PASS is INVALID, a
    # CLAIM gate that did not is CLAIM_FAIL, and CLAIM_FAIL is a RESULT that the
    # driver must mark finished rather than queue for another attempt. That
    # replaces "exit 1 if any gate FAILED", which collapsed a refuted claim and
    # a broken instrument onto one integer and left UNKNOWN scored as a pass.
    #
    # A SELF TEST is scored on its S gates instead, and never on the twelve. Its
    # twelve are a world's PLANTED answers: V7 fails in every world by
    # construction because no instrument produced the numbers, so classifying on
    # them would report INVALID for a self test that worked perfectly. What a
    # self test can be right or wrong about is whether the world produced the
    # verdicts registered for it, and that is what the S gates ask. Without
    # `--fail-on-world` a self test exits REFUSED, because nothing was measured.
    if args.self_test is not None:
        if not args.fail_on_world:
            print("\n".join([
                "", "=" * 72,
                "NOT A RESULT. Nothing was measured.",
                f"  reason: --self-test {args.self_test} was given; every arm "
                f"time above was GENERATED",
                "  The twelve gates above are this world's planted answers. Pass",
                "  --fail-on-world to score the S gates and exit through the",
                "  shared exit-code table instead.",
                "=" * 72]))
            return exit_codes.REFUSED
        for gate_ in self_gates:
            print(gate_.result_line())
        rc = exit_codes.classify(g.scored() for g in self_gates)
        print(f"exit     {exit_codes.describe(rc)}")
        return rc

    for gate_ in gates:
        print(gate_.result_line())
    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def run_measurement(cells: list[Cell], arms: list[str], args, out_dir: Path,
                    csv_path: Path, run_id: str, gpu_name: str, prov):
    """The metered part. Returns `(results, stopped, tally)`.

    `results` is None when `find_pieces` refused, and then nothing was measured
    and the tally is empty. `MeasurementTally` says why the third element is
    there at all.

    TRITON_CACHE_DIR is pointed at this run's own directory BEFORE vLLM is
    imported. A warm cache compiles and dumps nothing, which makes V4's compile
    assay vacuous; that exact bug cost this project its A100 PTX dump.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "triton-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    try:
        pieces = find_pieces()
    except SeparationRefusal as exc:
        print(f"\nNOT A RESULT: {exc}")
        return None, str(exc), MeasurementTally(0, 0, csv_path)

    # `gpu_name` is forced to the card the Store keys on rather than re-probed.
    # The row's `gpu_name` column IS the resume key, so a row whose column and
    # whose store disagreed would be written under one card and looked up under
    # another, which is the defect the key was added for wearing a new hat.
    meta = dict(environment_meta(), gpu_name=gpu_name, run_id=run_id,
                routing=args.routing, seed=args.seed,
                reference_clock_mhz=args.reference_clock_mhz,
                provenance_columns=prov.as_columns())
    print(f"\ntriton cache: {cache_root} (fresh for this run)")
    print("pieces: " + ", ".join(sorted(pieces)))
    print(f"device: {meta['gpu_name']}   torch {meta['torch_version']}   "
          f"triton {meta['triton_version']}   vllm {meta['vllm_version']}")
    print("reference clock: "
          + (f"{args.reference_clock_mhz:.0f} MHz, given by "
             "--reference-clock-mhz" if args.reference_clock_mhz
             else "NOT GIVEN, so clock_level_ok is None on every row and no row "
                  "is excluded for it. See `time_arm`: a LEVEL is relative to "
                  "the clock the roof was measured at, and this script scores "
                  "nothing against the roof"))

    store = Store(csv_path, gpu_name, fresh=args.fresh)
    if store.foreign:
        print("\n" + store.foreign_reason)
    results: Results = {}
    seen: set[Path] = set()
    count_new(cache_root, seen)
    started, stopped = time.time(), ""
    try:
        for index, cell in enumerate(cells, 1):
            if args.max_minutes and (time.time() - started) / 60 >= args.max_minutes:
                stopped = (f"stopped after {args.max_minutes} minutes with "
                           f"{index - 1} of {len(cells)} cells done")
                break
            print(f"  [{index}/{len(cells)}] {cell.model} T={cell.num_tokens}",
                  flush=True)
            try:
                results[cell.key] = measure_cell(cell, arms, args, store, meta,
                                                 pieces, cache_root, seen)
            except Exception as exc:  # noqa: BLE001
                # A refusal is about this CELL's rig, not about the session, so
                # it is recorded against the cell and the sweep continues. The
                # gates then see a cell with no arms rather than a run that
                # ended quietly, and V0 and V1 read the recorded class name.
                #
                # Broad on purpose beyond the typed refusals: deepseek-v3 at
                # T=8192 allocates about 26 GB of caches on top of 22.5 GB of
                # weights, and one cell running out of memory must not end a
                # metered session that has already paid for the rest.
                print(f"      REFUSED: {type(exc).__name__}: {exc}")
                results[cell.key] = {
                    "fused": ArmResult(cell.model, cell.num_tokens, "fused",
                                       error=f"{type(exc).__name__}: {exc}"[:300])}
                store.write(results[cell.key]["fused"], cell, meta)
    except KeyboardInterrupt:
        stopped = ("interrupted; every arm finished before the interrupt is on "
                   "disk and the same command resumes")
    finally:
        store.close()
    tally = MeasurementTally(store.written_arms, store.restored_arms, csv_path)
    print("\n" + tally.note)
    return results, stopped, tally


def main(argv: list[str] | None = None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it
    is in FINISHED_CODES, so the driver files the arm as finished, skips it on
    every resume, leaves RETRY_ARMS at zero and exits the session 0 over an arm
    that never measured. A torch OOM, a truncated report or a drifted import
    would be published as one of this experiment's registered outcomes. ERROR
    (4) is outside FINISHED_CODES precisely so the driver can tell "the
    apparatus broke" from "the claim did not hold", and the traceback is
    printed first rather than swallowed, because a code without one tells an
    operator nothing about what to fix.

    Wrapped around `_main` rather than installed at the `__main__` guard so the
    contract holds for a caller of `main()` -- the tests, and anything that
    imports this file -- as well as for the CLI. `SystemExit` is a
    `BaseException`, so the `except Exception` arm cannot catch it: a refusal is
    not a crash.

    A STRING REFUSAL IS REFUSED (2) AND NOT CLAIM_FAIL. `raise SystemExit("some
    sentence")` sets `SystemExit.code` to the STRING, and the interpreter turns
    a non-integer code into exit ONE. One is CLAIM_FAIL, which the paragraph
    above spends itself explaining is a RESULT: `--arms fused,cutlass_up`
    dropped a corner of the 2x2 and measured NOTHING, and the driver filed that
    as a refutation of the claim and never re-ran the arm. The sibling arms
    `occupancy_vs_swizzle` and `bn_decomposition` already convert here; this
    file had only the crash half of the handler, so its one string refusal at
    `_main` still exited one. Caught at the handler rather than at the raise
    site so a refusal added later cannot reintroduce it by forgetting the code,
    and an INTEGER code is re-raised untouched, because argparse's own
    `SystemExit(2)` is already the right number and is not ours to relabel.
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
        print("ERROR: span_extent_separation crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim "
              "failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
