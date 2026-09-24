#!/usr/bin/env python3
"""Is there a route to a DRAM counter, and what would it settle if there were?

    python scripts/dram_counter_route.py --dry-run          # the plan and its cost, off GPU
    python scripts/dram_counter_route.py --bracket          # a result, today, off GPU
    python scripts/dram_counter_route.py --self-test        # the estimator and the parser, off GPU
    python scripts/dram_counter_route.py --probe            # which route is open on THIS box
    python scripts/dram_counter_route.py --run --out c.json # TAKE the measurement, needs the GPU
    python scripts/dram_counter_route.py --analyse c.json   # score ONE counter run
    python scripts/dram_counter_route.py --contrast a.json b.json  # score the RATIO across runs

    python scripts/dram_counter_route.py --dry-run --family r3-arms   # R3's arms: plan, price
    python scripts/dram_counter_route.py --probe --family r3-arms     # every metric its pages ask
    python scripts/dram_counter_route.py --run --family r3-arms --census-only --out census.json
    python scripts/dram_counter_route.py --run --family r3-arms --group-m 4 \
                                         --census census.json --out r3c-g4.json
    python scripts/dram_counter_route.py --analyse r3c-g1.json r3c-g4.json  # alpha(G)

WHY THIS EXISTS. Every alpha in this study is `B / L`, the fraction of one full
weight read that a second M-tile costs, and `L` is an EXTRAPOLATION of the fitted
memory branch back to a single tile. The 2026-09 adversarial evaluation showed
`L` is not identified by the ladder: three defensible anchors give 0.452, 0.647
and 0.705 for the SAME cell (mixtral, GROUP_SIZE_M=16, BLOCK_SIZE_M=32, A100),
and refitting on `n >= 3` raises alpha in 12 of 12 A100 fits. The evaluation
named one experiment as decisive -- a DRAM-traffic measurement of the n=1 cell --
and called it "the only experiment that unblocks a numeric alpha".

THAT EXPERIMENT NEEDS A HARDWARE COUNTER, WHICH THIS PROJECT HAS NEVER HAD. So
this file does three separate things, and keeps them separate on purpose:

  --bracket  bounds alpha WITHOUT any counter, from published data, today. Two
             inequalities that need no new measurement at all (see
             `physical_bracket`). On the A100 surface they EXCLUDE four of the
             twelve published alphas and all four of their `n >= 3` refits.
             This is the part that produces a result rather than a plan.

  --probe    asks the machine it is running on which counter route is open, and
             distinguishes the four failures that look identical from a log:
             no ncu, ncu blocked by the host module flag, ncu blocked but
             fixable with a container capability, and nsys present but unable to
             write a report because its IMPORTER half was never installed.

  --dry-run  registers the counter experiment: the exact cell, the exact metric,
             the value predicted under each of the three candidate anchors, the
             margin between them, and the gates that would score the result.
             Item 4 of the brief: if a counter run ever happens it must not be
             improvised.

  --run      DRIVES ncu over one registered cell and writes the JSON --analyse
             consumes. Until 2026-09-10 this file planned, bracketed,
             self-tested, probed and scored, and there was no way to take the
             measurement, and the probe said OPEN. That OPEN was a false
             positive (see "--probe" above), so this mode has still never run
             on a box whose counters were shown readable. See "THE PER-CALL
             TRAP" below, which is the whole reason this is a mode and not a
             shell one-liner.

  --contrast scores the RATIO of dR/dn across two or more runs, which is the
             reading the extended plan exists to take and which no single
             payload contains. `--analyse` reads ONE payload and scores that
             cell; the discriminator between the two rivals for the missing
             1/BLOCK_N term is 1.871 against 1.000 ACROSS two cells, and until
             this mode existed the operator came off the pod with five scored
             cells and had to do that arithmetic by hand against the printed
             predictions. Pairs are matched inside one cache mode, and a pair
             nobody ran is not scored at all rather than reported UNKNOWN.

WHAT IS NOT HERE. No kernel. The cell profiled is the one
`scripts/block_m_crossing_sweep.py` already runs -- vLLM's `fused_experts` under
`override_config` -- and this file only prints the command line that wraps it.
Writing a second kernel to measure the first one's traffic would measure the
second kernel.

TWO CELL FAMILIES (`--family`). Everything above is the `ladder` family, the
default, and it behaves exactly as it did before the second family landed.
The `r3-arms` family (2026-09-24) profiles R3's three arms
(`scripts/private_weight_reference.py`) under the counter, one GROUP_SIZE_M
per invocation, the arm GEMMs only, through R3's own `--counter-child`, and
scores the bytes against a card-free model of the kernel's schedule; see
"THE R3 ARMS UNDER A DRAM COUNTER" below and docs/COUNTERS.md section 6. It
reads no ridge, no bandwidth and no calibration, and every page it writes
names the live card on its first line.

THE ONE ARITHMETIC IDEA, stated once because everything below depends on it.
At `n` M-tiles per expert the model says DRAM read traffic is

        R(n) = W (1 + alpha (n - 1)) + a n

with `W` the compulsory weight read (every active expert, up and down, once) and
`a` the activation bytes an extra tile carries. That is AFFINE IN n, so a counter
that reports `R` at four or more values of `n` gives

        alpha = (dR/dn - a) / W

with NO bandwidth constant, NO extrapolation to zero tiles and NO timing model.
The quantity the ladder can only infer is the quantity a counter reads directly,
which is exactly why the evaluation called this experiment decisive.

AND THE TRAP IT AVOIDS. `R(1)` is the SAME under all three anchors -- they differ
only in slope -- so an n=1 measurement alone settles nothing. It is a VALIDITY
check on the byte model, not a claim. The claim needs the slope, which needs
several n. Anyone who profiles one launch and reports an alpha has measured the
byte model's intercept and called it the answer.

THE PER-CALL TRAP, which is what `--run` exists to get right. The instrument
under the profiler is `moe.bench.timing`, which warms up for a DURATION and then
sizes its own iteration count: the profiled launch count is warmup + iters x
trials fused_experts calls, each of several kernel launches. A counter therefore
integrates MANY calls and every byte field must be divided by the number of
calls the instrument actually made, counted from the profile's own launch list.
Dividing by an assumed 1 inflates every reading by an order of magnitude and
would still fit an affine line, so nothing downstream would notice. `--run`
counts the calls, `--self-test` plants a known count and checks the division,
and a count that does not exceed the cell's own `iters x trials` is REFUSED.

WHY THE PLAN IS ON THE H200 AND NOT THE A100. It was registered against the
A100 mixtral G=16 cell, because that is where the three anchors disagree most.
It moved to the H200 on 2026-09-10 because the probe read OPEN there and
nowhere else. THAT OPEN WAS A FALSE POSITIVE and the move outlived it: the
probe profiled `/bin/true` and so never asked for a counter at all, and on
2026-09-15 a rented H200 refused the read outright with ERR_NVGPUCTRPERM. The
plan stays on the H200 anyway, because the H200 is the card this study rents
and the card whose calibration and session corpus are in the tree, and NOT
because any box has been shown to allow a counter. It registers the H200 cell
and reads every prediction out of
`moe/bench/hardware/measured_nvidia_h200.yaml` and the committed session
corpus. The honest consequence is stated where it lands: on the H200 the three
anchors agree to 0.04 and the anchor contrast (C1) is worth about 5% in
traffic, so it is no longer the reason to rent a box. The BLOCK_N contrast is:
its two rival readings are 87% apart. See `contrast_plan`.

AND C1 IS RE-REGISTERED ON THAT CELL, because for one commit it was not. The
gate asked "does exactly one anchor survive a window of 0.05" while the H200
anchors span 0.0393, so a PERFECT measurement of the registered cell left all
three standing and scored CLAIM C1 FAIL with exit 1. The file disclosed the
collapse in prose and left the gate calibrated to the cell it had left, which
is this repository's most-repeated defect exactly. `c1_registration` now
decides which question the cell can carry, --dry-run prints that decision
before anything runs, and a successful measurement of a clustered cell passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The shared exit-code table and the shared provenance block, both imported
# rather than approximated here: this file spent its life returning integers it
# chose for itself and writing JSON that named no commit and no machine.
from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import counter_probe_kernel as PK  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402

# Imported, never re-derived. Two byte models for one study is how the padding
# tax survived three months: `weight_bytes_per_expert` and
# `activation_bytes_per_row` are the SAME functions the ladder fit is scored
# against, so if they move, this file's predictions move with them and
# tests/test_dram_counter_route.py pins the numbers so the move is visible.
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402  (after sys.path insert)
from scripts.block_m_crossing_sweep import FIXED as SWEEP_FIXED  # noqa: E402
from scripts.block_m_crossing_sweep import (  # noqa: E402
    activation_bytes_per_row,
    ai_cap,
    gemm_operand_read_bytes_per_row,
    q_of_tiles,
    tokens_for_rows,
    weight_bytes_per_expert,
)

PASS, FAIL, REFUSE = "PASS", "FAIL", "REFUSE"

#: The card label a mode that touches no GPU carries. `--bracket` reads
#: published CSVs and `--analyse` scores a JSON someone else measured, so
#: neither has a live card, and `provenance.run_id` refuses an id without one.
#: A name no `nvidia-smi` can produce, so it can never be read as a real card.
NO_CARD = "no-card-nothing-measured"

#: The instrument of the modes that measure NOTHING, and deliberately NOT
#: `moe.bench.timing.TIMING_BASIS`. `--bracket` and `--analyse` are ARITHMETIC
#: over numbers other runs measured. Stamping the timing basis on either would
#: describe an apparatus that never ran, which is the exact defect the audit
#: found in two sibling `--synthetic` paths. `--probe` and `--run` carry their
#: own strings below: since 2026-09-10 this file is no longer one where nothing
#: touches a device, and one constant covering every mode would have said so.
INSTRUMENT = "arithmetic-over-published-rows/no-kernel-timed"

#: `--probe` times nothing, but since 2026-09-15 it is no longer true that it
#: touches no device: it launches ONE kernel under ncu, because a counter probe
#: that launches none cannot reach the permission check it exists to reach. The
#: string says both halves. Nothing it measures is reported as a measurement --
#: the only question asked of the profile is whether a number came back.
PROBE_INSTRUMENT = ("machine-configuration-probe/one-probe-kernel-profiled/"
                    "nothing-timed")

#: `--contrast` is arithmetic over payloads `--run` measured, and the arithmetic
#: is a RATIO of two counter slopes: no bandwidth constant, no byte model and no
#: kernel of its own. Naming the counter apparatus here would claim this mode
#: profiled something.
CONTRAST_INSTRUMENT = ("arithmetic-over-counter-payloads/no-kernel-timed; "
                       "a ratio of two measured slopes, no bandwidth constant")

#: `--run` is the ONE mode of this file that measures a device, and what it
#: measures is TRAFFIC, not time. It is still not `timing.TIMING_BASIS`: under
#: `--replay-mode kernel` every launch is replayed, so any duration in the
#: payload is replay time and is not comparable with the unprofiled ladder's
#: wall clock. Saying so in the instrument string is the difference between a
#: reader joining the two and a reader knowing not to.
RUN_INSTRUMENT = ("nsight-compute/dram-counters/replay-mode-kernel/"
                  "per-call-normalised; NOT timing.TIMING_BASIS")

# --------------------------------------------------------------------------
# Constants that are quoted rather than measured, each with its source.
# --------------------------------------------------------------------------

#: DATASHEET peak DRAM bandwidth, GB/s. Used only as a HARD physical ceiling in
#: `physical_bracket`: a fitted memory branch that implies more than this is
#: impossible, not merely surprising. Deliberately the datasheet number and not
#: the calibrated one -- the calibrated triad figure is a STREAM pattern and a
#: weight stream is read-dominated, so using it as a ceiling would over-claim.
#: The measured figure is reported beside it as the softer bound.
#: A100 80GB SXM4: 2039 GB/s, NVIDIA A100 datasheet. There is no a100 profile
#: yaml in this repo (only measured_nvidia_a100_sxm4_80gb.yaml), so this one is
#: a quoted constant and is flagged as such wherever it is printed.
#: H200: 4800 GB/s, and that one IS in the repo -- moe/bench/hardware/h200_sxm.yaml
#: `memory.bandwidth_tb_s: 4.8`, checked against datasheet 3512650.Nov24.
DATASHEET_PEAK_GBPS: dict[str, float] = {
    "nvidia_a100_sxm4_80gb": 2039.0,
    "nvidia_h200": 4800.0,
}

#: The metrics a counter run collects, in the order they are asked for.
#: `dram__bytes_read.sum` is the one the evaluation named. The other three are
#: not decoration:
#:   * writes, because the byte model charges read+write as one traffic total
#:     and a large write share would mean the model is missing a term rather
#:     than that alpha is high;
#:   * the L2 read hit rate, because "alpha is the fraction of a re-read that
#:     MISSES L2" is the definition on every published surface and this is the
#:     only direct measurement of it this project could ever take;
#:   * the kernel duration, so bytes and time come from the SAME launch and the
#:     achieved bandwidth can be computed without joining two runs.
NCU_METRICS: tuple[str, ...] = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_op_read_hit_rate.pct",
    "gpu__time_duration.sum",
)

#: The unit each metric is reduced to before anything is summed, and every
#: unit ncu is allowed to have reported it in.
#:
#: ncu RESCALES. `--csv --page raw` prints whatever unit keeps the number
#: readable, so the same metric comes back as `byte` on one launch and `Mbyte`
#: on another in the same file, and the prefixes are decimal (Kbyte = 1000 B),
#: not binary. Summing the raw column across launches therefore adds
#: megabytes to bytes and lands a factor of a million out, which is a wrong
#: number and not a crash. Each metric is converted to ONE canonical unit here
#: and an unrecognised unit REFUSES: this table is the list of units this
#: parser has been shown, not a guess at ncu's whole vocabulary, so a new one
#: must be added deliberately rather than silently scaled by 1.
#:
#: THE EMPTY UNIT IS NOT IN ANY TABLE, and it was, until 2026-09-10. Each table
#: carried `"": 1.0`, and `parse_ncu_csv` filled `unit` with `""` whenever the
#: `Metric Unit` column was absent from the header, so a CSV without that column
#: took every value at face value: a profile whose bytes ncu had already
#: rescaled to Mbyte was read as bytes, a factor of 1e6 low, still perfectly
#: affine in n and therefore invisible to every gate below. The column is now
#: REQUIRED the way `ID` is, and a blank unit on a metric that has one refuses.
NCU_METRIC_UNITS: dict[str, tuple[str, dict[str, float]]] = {
    "dram__bytes_read.sum": ("byte", {
        "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}),
    "dram__bytes_write.sum": ("byte", {
        "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}),
    "gpu__time_duration.sum": ("nsecond", {
        "nsecond": 1.0, "usecond": 1e3, "msecond": 1e6, "second": 1e9}),
    "lts__t_sector_op_read_hit_rate.pct": ("%", {"%": 1.0, "percent": 1.0}),
    # THE SECTOR METRICS, the r3-arms family's (`--family r3-arms`). An L2
    # sector is 32 bytes; ncu prints the count in sectors and rescales it with
    # the same decimal prefixes as bytes. `--print-units base` asks for the
    # bare "sector", and the prefixed spellings are here because the CSV of a
    # capture made without that flag rescales per launch like every other
    # metric does.
    **{m: ("sector", {"sector": 1.0, "Ksector": 1e3, "Msector": 1e6,
                      "Gsector": 1e9})
       for m in ("lts__t_sectors_srcunit_tex_op_read.sum",
                 "lts__d_sectors_fill_device.sum",
                 "lts__t_sectors_op_read_lookup_miss.sum",
                 "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
                 "lts__t_sectors_srcunit_ltcfabric.sum")},
}

#: THE ONE PLACE AN EMPTY UNIT IS ACCEPTED, and it is a separate table so the
#: rule above ("no byte, time, rate or sector table carries the empty unit")
#: stays true of `NCU_METRIC_UNITS` and is still tested there. Every metric
#: here is a `launch__*` launch attribute that ncu prints either with no unit
#: at all (a count or a ratio) or with the one unit listed: a grid size, a
#: wave count, an occupancy limit in blocks, registers per thread. None of
#: them is rescaled, so an empty unit means "unitless" here and cannot mean
#: "ncu dropped the prefix". A unit not listed still REFUSES, loudly, at the
#: probe; `tests/test_dram_counter_route.py` holds that every metric with an
#: empty entry is a `launch__*` metric.
UNITLESS = ""
NCU_LAUNCH_UNITS: dict[str, tuple[str, dict[str, float]]] = {
    "launch__grid_size": (UNITLESS, {UNITLESS: 1.0}),
    "launch__waves_per_multiprocessor": (UNITLESS, {UNITLESS: 1.0}),
    "launch__registers_per_thread": ("register/thread", {
        "register/thread": 1.0, UNITLESS: 1.0}),
    **{m: ("block", {"block": 1.0, UNITLESS: 1.0})
       for m in ("launch__occupancy_limit_blocks",
                 "launch__occupancy_limit_registers",
                 "launch__occupancy_limit_shared_mem",
                 "launch__occupancy_limit_warps")},
}


def unit_table(metric: str) -> tuple[str, dict[str, float]]:
    """`(canonical unit, accepted units)` for a registered metric. One lookup
    over both tables, so the parser never scales a metric it has no table for."""
    if metric in NCU_METRIC_UNITS:
        return NCU_METRIC_UNITS[metric]
    return NCU_LAUNCH_UNITS[metric]


def registered_metric(metric: str) -> bool:
    return metric in NCU_METRIC_UNITS or metric in NCU_LAUNCH_UNITS

#: The byte metrics, the ones divided by the call count. A rate and a duration
#: are not; the rate is a weighted mean and the duration is per call because it
#: is a sum over launches like the bytes are.
NCU_BYTE_METRICS: tuple[str, ...] = ("dram__bytes_read.sum", "dram__bytes_write.sum")

#: The kernel whose launch count IS the fused_experts call count. vLLM's
#: `fused_experts` runs `moe_align_block_size` exactly once per call, before
#: either GEMM, so counting it counts calls without assuming how many GEMM or
#: reduction launches a version of vLLM emits per call. `--call-marker`
#: overrides it, because a kernel rename must cost one flag on the pod and not
#: a lost hour; a marker that matches NOTHING refuses and prints the kernel
#: names the profile actually contained.
CALL_MARKER = "moe_align_block_size"

#: The GEMM itself, which runs twice per call (up and down). Not used to count
#: calls: it is the cross-check that the count is the count it looks like.
GEMM_MARKER = "fused_moe_kernel"

#: The session corpus the contrast predictions are derived from, ladder by
#: ladder. Never transcribed: `corpus_slope` re-fits the committed cells.
GAPS_SESSION = (REPO / "results" / "published"
                / "2026-09-10-nvidia_h200-gaps-session" / "results")

#: The tile counts to profile. Four is the fewest that lets the OLS line have a
#: residual at all (three points and two parameters leaves one degree of
#: freedom); six buys a sharper slope. NOT "for eight seconds of GPU time",
#: which is what stood here and was the unprofiled sweep's cost: under ncu each
#: tile count is its own invocation and the replay of a 2.8 GB weight set is
#: minutes, so a tile count is minutes and the cost line prices it that way.
DEFAULT_TILES = (1, 2, 3, 4, 6, 8)

#: The A100 mixtral G=16 cell, which the 2026-09 evaluation singled out because
#: its three anchors are 0.452 / 0.647 / 0.705, the widest disagreement
#: anywhere on the surface. It was this file's default cell until 2026-09-10 and
#: it is still the cell `docs/COUNTERS.md` section 4 registers, so it keeps a
#: name here rather than disappearing into history. It is NOT the plan's cell
#: any more: no counter route has ever been SHOWN open on an A100 this study
#: can rent, and as of 2026-09-15 none has been shown open on an H200 either.
A100_REPORT = (REPO / "results" / "published"
               / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
               / "mixtral-8x7b-bf16-r1024-g16-n64-b156b5.report.json")

#: The cell the plan is written for, by default: the H200 twin of the same arm.
#: Same model, same dtype, same GROUP_SIZE_M=16, same BLOCK_SIZE_N=64, so the
#: counter still answers the ladder rather than a different question. The H200
#: was chosen on 2026-09-10 because its `--probe` came back OPEN; that reading
#: was retracted on 2026-09-15 and the choice now rests on this being the card
#: this study rents and calibrates, not on a counter anyone has read.
DEFAULT_REPORT = (REPO / "results" / "published"
                  / "2026-09-01-nvidia_h200-alpha-surface-s4"
                  / "mixtral-8x7b-bf16-r1024-g16-n64-69f35a.report.json")

#: Bit 21 of the Linux capability mask. CAP_SYS_ADMIN is what NVIDIA's own
#: ERR_NVGPUCTRPERM page names as the container-side alternative to changing the
#: host module parameter, so a probe that does not check it cannot tell
#: "this container could profile" from "this host refuses".
CAP_SYS_ADMIN_BIT = 21

#: Bit 38, CAP_PERFMON, and it was missing until 2026-09-15. Since Linux 5.8 and
#: driver R450 the counter gate accepts CAP_PERFMON as well as CAP_SYS_ADMIN, so
#: a container can profile while holding neither SYS_ADMIN nor root. A probe
#: that reads only bit 21 cannot tell "no capability at all" from "the weaker
#: capability that would have sufficed", which is the difference between two
#: different asks to a provider -- and providers refuse `--cap-add=SYS_ADMIN`
#: far more often than `--cap-add=PERFMON`. THE PUBLISHED PAYLOADS SAY NOTHING
#: ABOUT THE FIELD'S WIDTH, and a first draft of this comment claimed they did:
#: the 2026-09-09 and 2026-09-10 payloads record `cap_eff` as `0xa80425fb`
#: because `probe_capabilities` stored `hex(mask)`, and `hex` strips leading
#: zeros. Linux renders `CapEff` in sixteen zero-padded hex digits, so those
#: eight digits are Python's formatting and not the mask's width. What the
#: payloads DO establish is the VALUE: 0xa80425fb is below 2**38, so bit 38 was
#: clear on both pods. That is a measurement of the bit, which is the stronger
#: statement anyway.
CAP_PERFMON_BIT = 38

#: The metric `--probe` asks for, and deliberately the SAME metric `--run`'s
#: first registered metric is: a probe that proves a different counter readable
#: proves the wrong thing. It is read through `parse_ncu_csv`, the same parser
#: `--run` uses, for the same reason.
NCU_PROBE_METRIC = NCU_METRICS[0]

#: The probe kernel `--probe` profiles: a real file on disk, launched in a child
#: interpreter under ncu. See that module for why `/bin/true` was not one.
NCU_PROBE_KERNEL = REPO / "moe" / "bench" / "counter_probe_kernel.py"

#: Seconds the profiled probe child may take. The old no-op probe had 90 and
#: finished in under one; this one imports torch and creates a CUDA context in a
#: child under a profiler, and this repo's own measured figure for a
#: torch-importing child under a profiler is "roughly fifteen seconds a rung ...
#: the child's torch import dominates, not the profiling"
#: (`scripts/nsys_dram_probe.py`). 180 is twelve times that, which is slack for
#: a cold page cache on a fresh pod. IT IS A CEILING AND NOT A BUDGET, and it is
#: deliberately LARGER than the arm's own booking rather than inside it: the
#: session books `counter_plan` one minute, which prices the EXPECTED 15 s, and
#: a ceiling set inside that booking would turn a pod that is merely slow into a
#: pod this file reports as hung. A child that actually reaches 180 s has wedged
#: on CUDA init, which is the MooseFS-stall shape this study has already hit, and
#: `_run` kills its whole process group when it does. This comment read "an
#: eighth of the one minute" until 2026-09-15, which inverted the comparison it
#: was making: 180 s is three times sixty, not an eighth of it.
NCU_PROBE_TIMEOUT_S = 180

#: THE CELL FAMILIES `--family` chooses between. `ladder` is the cell this
#: file registered before 2026-09-24: the whole-layer `fused_experts` call
#: under the timed sweep's instrument, one tile count per ncu invocation. It
#: is the default and every mode behaves byte-identically under it. `r3-arms`
#: is R3's three arms (`scripts/private_weight_reference.py`) under the
#: counter, one GROUP_SIZE_M per invocation; see "THE R3 ARMS UNDER A DRAM
#: COUNTER" below and docs/COUNTERS.md section 6.
LADDER_FAMILY = "ladder"
R3_FAMILY = "r3-arms"
FAMILIES: tuple[str, str] = (LADDER_FAMILY, R3_FAMILY)


# --------------------------------------------------------------------------
# The three anchors, and the bracket that does not need a counter.
# --------------------------------------------------------------------------

def ols(xs, ys) -> tuple[float, float]:
    """Ordinary least squares `y = a + b x`. Same estimator as the sweep's."""
    if len(xs) < 2:
        raise ValueError("OLS needs at least two points; refusing to invent a line")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("all x are equal; the slope is not identified")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    return my - b * mx, b


@dataclass(frozen=True)
class Anchors:
    """One ladder's alpha under each of the three defensible anchors for `L`.

    They are not variants of an estimator. They are three different answers to
    "what is the time of ONE full weight read", and the ladder cannot choose:

      published  `L = A + B`, the fitted branch evaluated at n=1. What every
                 published report prints.
      t1         `L = t(1)`, the MEASURED single-tile time. Defensible because
                 at n=1 there is exactly one M-tile per expert and no re-read is
                 geometrically possible, so t(1) IS a full weight read plus
                 non-negative extras.
      n3         drop n=1 and n=2 and refit. Defensible because the low treads
                 are the ones whose L2-reuse condition differs from the branch.
    """

    block_m: int
    t1_ms: float
    intercept: float
    slope: float
    intercept_n3: float
    slope_n3: float

    @property
    def published(self) -> float:
        return self.slope / (self.intercept + self.slope)

    @property
    def t1(self) -> float:
        return self.slope / self.t1_ms

    @property
    def n3(self) -> float:
        return self.slope_n3 / (self.intercept_n3 + self.slope_n3)

    def as_dict(self) -> dict[str, float]:
        return {"published": self.published, "t1": self.t1, "n3": self.n3}


def anchors_from_points(points, memory_points: int) -> Anchors:
    """Refit a published ladder's memory branch and evaluate all three anchors.

    `points` is the report's `ladder[bm].points`, a list of `(n, ms)`. Only the
    leading `memory_points` treads are on the memory branch; the rest are
    compute bound and belong to a different line.
    """
    mem = [(int(n), float(ms)) for n, ms in points][:memory_points]
    if len(mem) < 3:
        raise ValueError(f"{len(mem)} memory-bound treads; a branch needs 3")
    a, b = ols([n for n, _ in mem], [ms for _, ms in mem])
    tail = [(n, ms) for n, ms in mem if n >= 3]
    if len(tail) < 2:
        raise ValueError("fewer than two treads at n>=3; the n3 anchor is not defined")
    a3, b3 = ols([n for n, _ in tail], [ms for _, ms in tail])
    return Anchors(block_m=0, t1_ms=mem[0][1], intercept=a, slope=b,
                   intercept_n3=a3, slope_n3=b3)


def physical_bracket(slope_ms: float, t1_ms: float, weight_bytes: int,
                     act_bytes_1: int, peak_gbps: float) -> tuple[float, float]:
    """Two inequalities that bound alpha with NO counter and NO new run.

    LOWER, `alpha >= B / t(1)`. alpha is `B / L`. At n=1 there is one M-tile per
    expert, so the kernel reads the weight set exactly once and `t(1)` is that
    read PLUS non-negative extras -- launch, low occupancy, the tail. Extras are
    never negative, so `L <= t(1)`, so `alpha >= B / t(1)`. Nothing about the
    fit enters; only the measured single-tile time.

    UPPER, `alpha <= B * peak / (W + a)`. The traffic at one tile is at least
    the compulsory weight read plus one tile of activations, and no traffic
    moves faster than the pin rate, so `L >= (W + a) / peak`, so
    `alpha <= B peak / (W + a)`.

    WHAT MAKES THIS WORTH HAVING: the two bounds come from opposite directions
    and neither uses the fitted intercept, which is the quantity the evaluation
    showed is unidentified. A published alpha OUTSIDE this interval is not
    uncertain, it is impossible -- it asserts a memory branch that moves the
    compulsory bytes faster than the memory system can move them.

    WHICH ALPHA IT BOUNDS, because the reports print two. Both bounds are on the
    UNCORRECTED `alpha = B / L`, the report's `alpha` column, because `B` is the
    raw fitted slope and the activation share sits inside it. Subtracting that
    share -- the `alpha-corrected` column -- needs a bandwidth constant to turn
    activation BYTES into milliseconds, and the whole point of this bracket is
    that it assumes no such constant. `alpha_corrected` runs 1-3% below `alpha`
    on these fits, so a bracket violation of a few percent decides nothing and
    the caller tests against the fit's own residual.

    WHAT IT IS NOT: a measurement of alpha. The interval is wide exactly where
    the kernel runs far from peak, because that is where "time" and "traffic"
    stop being the same statement. Its width is
    `alpha_lo * (peak / achieved(1) - 1)`, so a cell at 70% of peak carries a
    43% relative bracket. Tightening it is what the counter is for.
    """
    if peak_gbps <= 0 or weight_bytes <= 0 or t1_ms <= 0:
        raise ValueError("refusing to bracket with a non-positive constant")
    lo = slope_ms / t1_ms
    l_min_ms = 1e3 * (weight_bytes + act_bytes_1) / (peak_gbps * 1e9)
    hi = slope_ms / l_min_ms
    return lo, hi


# --------------------------------------------------------------------------
# The counter experiment: what would be measured, and what it would settle.
# --------------------------------------------------------------------------

def weight_bytes_total(cfg, b: int = 2) -> int:
    """`E * 3 F H * b`: every active expert's up and down weights, once.

    Every expert is active in this cell by construction -- the sweep routes with
    `balanced_ids`, an exact histogram -- so `active_experts == E` and there is
    no routing term to argue about.
    """
    return cfg.num_experts * weight_bytes_per_expert(cfg, b)


def activation_bytes_per_tile(cfg, block_m: int, act_b: int = 2) -> int:
    """`E * BM * (2H + 3F) * act_b`: what one more M-tile per expert carries."""
    return cfg.num_experts * block_m * activation_bytes_per_row(cfg, act_b)


def predicted_read_bytes(cfg, block_m: int, n: int, alpha: float, b: int = 2) -> float:
    """`W q(n) + a n`, the model's traffic at `n` tiles per expert."""
    return (weight_bytes_total(cfg, b) * q_of_tiles(n, alpha)
            + activation_bytes_per_tile(cfg, block_m) * n)


def layer_gemm_shapes(cfg) -> tuple[tuple[int, int], tuple[int, int]]:
    """`((N, K) of the up GEMM, (N, K) of the down GEMM)` for one expert layer.

    `moe/bench/ai_model.py` is a single-GEMM model and a fused expert layer is
    two of them: up, `C[M, 2F] = A[M, H] @ W13[H, 2F]`, and down,
    `C[M, H] = A[M, F] @ W2[F, H]`. A per-tile activation cost `phi` is stated
    per GEMM, so a bracket over the layer takes the widest of the two.
    """
    return ((2 * cfg.intermediate_size, cfg.hidden_size),
            (cfg.hidden_size, cfg.intermediate_size))


def cap_from_counter(cfg, block_m: int, alpha: float, b: int = 2) -> tuple[float, float]:
    """`(cap, uncorrected)`: the AI ceiling a COUNTER alpha implies, and the
    study's `2 BM / (alpha b)` reading of the same number, which is an upper
    bound on it.

    WHY THIS IS NOT `ai_model.cap_from_fitted`. That function inverts a
    B/(A+B) LADDER fit, whose alpha is `(alpha_b + phi) / (1 + phi + delta)`:
    the first tread's activation, output and fixed cost sit in the denominator
    and the cap read as `2 BM / (alpha b)` is high by exactly that level. The
    counter's alpha is a different quantity with no level in it at all:
    `alpha_from_counters` returns `(dR/dn - a) / W`, a TRAFFIC SLOPE in
    weight-read units with the once-read activation share `a` taken out.
    Applying the level correction to it would divide by a denominator the
    number never had.

    What the counter alpha does need is the share it removed put back. The
    intensity at `n` tiles is FLOPs over bytes, both affine in `n`, so the
    ceiling is the ratio of slopes: `2 BM W / (b dR/dn) = 2 BM / (b (alpha +
    a/W))`. The study's `2 BM / (alpha b)` drops `a/W` (0.009 at BLOCK_M=32 on
    mixtral, 0.019 at 64) and is therefore an UPPER BOUND on the cap, not the
    cap; the difference flips the verdict for an alpha within `a/W` of the
    threshold, which is why the gate scores the corrected number.

    No alpha_a bracket is needed here, and that is a property of the
    instrument: a DRAM counter measures the activation re-read and the weight
    re-read alike, so whichever operand the extra traffic is on, it is in the
    slope. The counter cannot SPLIT the slope into alpha_b and alpha_a, which
    is why C1's comparison with the ladder anchors is in different units, and
    that is stated on the C3 gate rather than hidden.

    Read traffic only: `dram_bytes_read` is the metric the evaluation named,
    the write slope is not in `alpha`, and the gate says the cap is high by
    the write share.
    """
    # At alpha = 0 the slope is the once-read share alone and the cap is
    # FINITE, unlike `ai_cap`'s infinity: a tile still carries its own
    # activations and output whatever the weight re-read costs.
    W = weight_bytes_total(cfg, b)
    a = activation_bytes_per_tile(cfg, block_m)
    slope = alpha + a / W
    if slope <= 0.0:
        raise ValueError(
            f"alpha={alpha} plus the once-read share {a / W:.4f} is not positive; "
            "a non-positive traffic slope is not a ladder and has no cap")
    return 2.0 * block_m / (b * slope), ai_cap(block_m, alpha, b)


@dataclass(frozen=True)
class CapBracket:
    """The cap a LADDER alpha implies, as a bracket over the unmeasured alpha_a.

    `low` is the alpha_a = 1 end (every N-tile re-reads the activation slab,
    the largest per-tile cost, the lowest cap) and `high` the alpha_a = 0 end;
    both at delta = 0, the smallest level correction, so even the high end is
    an upper bound on the corrected cap. An end is None when
    `ai_model.cap_from_fitted` REFUSED it, because the fitted alpha is below
    the floor `phi / (1 + phi)` that alpha_b = 0 gives at that alpha_a: the
    three-term model cannot have produced this reading with that much
    activation re-read, and `refused` says so per end rather than clamping.
    """

    alpha_fitted: float
    uncorrected: float
    low: float | None
    high: float | None
    refused: dict[str, str]

    def render(self) -> str:
        def end(v):
            return f"{v:.1f}" if v is not None else "REFUSED"
        return (f"2BM/(alpha b) = {self.uncorrected:.1f} (upper bound); corrected "
                f"[{end(self.low)} at alpha_a=1, {end(self.high)} at alpha_a=0] at delta=0"
                + ("" if not self.refused else
                   "; " + "; ".join(f"{k}: {v}" for k, v in self.refused.items())))


def anchor_cap_bracket(cfg, block_m: int, block_n: int, alpha_fitted: float,
                       b: int = 2) -> CapBracket:
    """`ai_model.cap_from_fitted` over alpha_a in {0, 1}, both layer GEMMs.

    The registered anchors (`published`, `t1`, `n3`) are B/(A+B) ladder fits,
    so the cap each one implies is the study's `2 BM / (alpha b)` divided by
    `(1 + phi + delta)`, retraction (a). `alpha_a` has no measurement anywhere
    in this repository, so `phi` is a bracket: its alpha_a = 0 end is the
    smaller of the two GEMMs' once-read costs and its alpha_a = 1 end the
    larger of their full re-read costs, which is the widest honest interval.
    `delta` is unmeasured too and only lowers the cap, so it is held at 0 and
    the bracket is labelled as the generous end.
    """
    uncorrected = ai_cap(block_m, alpha_fitted, b)
    ends: dict[str, float | None] = {}
    refused: dict[str, str] = {}
    for label, alpha_a, pick in (("alpha_a=0", 0.0, min), ("alpha_a=1", 1.0, max)):
        phi = pick(ai_model.phi(N, K, block_m=block_m, block_n=block_n,
                                alpha_a=alpha_a, b=b)
                   for N, K in layer_gemm_shapes(cfg))
        try:
            ends[label] = ai_model.cap_from_fitted(
                alpha_fitted, block_m=block_m, b=b, phi=phi, delta=0.0)
        except ai_model.AIModelRefused as exc:
            ends[label] = None
            refused[label] = str(exc).split(":")[0] + f" (phi {phi:.3f})"
    return CapBracket(alpha_fitted, uncorrected, ends["alpha_a=1"], ends["alpha_a=0"],
                      refused)


def anchor_cap_lines(cfg, block_m: int, block_n, anchors: dict, ridge: float | None,
                     b: int = 2) -> list[str]:
    """One line per registered anchor: the cap it implies, as a bracket.

    Refuses in words when the payload names no BLOCK_N: `phi` needs it and 64
    is the sweep's pin, not this payload's, so it is not assumed.
    """
    if not anchors:
        return ["no anchors registered in the payload, so no ladder cap to bracket"]
    if not block_n:
        return ["anchor caps NOT bracketed: the payload names no block_n and phi "
                "needs BLOCK_N; REFUSED rather than assumed 64"]
    out = ["what each registered LADDER anchor implies for the cap, through "
           "ai_model.cap_from_fitted, alpha_a unmeasured so a bracket:"]
    for name, value in sorted(anchors.items()):
        br = anchor_cap_bracket(cfg, block_m, int(block_n), float(value), b)
        verdict = ""
        if ridge is not None and br.low is not None and br.high is not None:
            verdict = ("  both ends below the ridge" if br.high < ridge else
                       "  both ends at or above the ridge" if br.low >= ridge else
                       "  the bracket straddles the ridge; alpha_a decides")
        out.append(f"  {name:<10} alpha {value:.3f}: {br.render()}{verdict}")
    return out


def alpha_from_counters(rows, cfg, block_m: int, b: int = 2) -> tuple[float, float, float]:
    """`(alpha, intercept_bytes, max_relative_residual)` from measured traffic.

    The whole point of the experiment in three lines: fit `R = R0 + dR n`, take
    out the activation share of the slope, divide by the compulsory weight read.
    No bandwidth constant appears, which is what makes this independent of every
    calibration the study argues about.
    """
    ns = [int(r["n"]) for r in rows]
    ys = [float(r["dram_bytes_read"]) for r in rows]
    if len(set(ns)) < 3:
        raise ValueError(f"{len(set(ns))} distinct tile counts; refusing to fit a slope")
    r0, dr = ols(ns, ys)
    a = activation_bytes_per_tile(cfg, block_m)
    alpha = (dr - a) / weight_bytes_total(cfg, b)
    resid = max(abs((r0 + dr * n) - y) / y for n, y in zip(ns, ys, strict=True))
    return alpha, r0, resid


def discrimination(cfg, block_m: int, tiles, alphas: dict[str, float]) -> list[dict]:
    """The prediction table, registered before anything runs.

    One row per tile count, one column per candidate anchor, plus the SPREAD --
    the gap between the extreme predictions as a fraction of the smallest. That
    last column is the whole design argument: it is the counter accuracy the
    experiment needs, and if it were 1% the experiment would not be worth
    renting a box for.
    """
    out = []
    for n in tiles:
        vals = {k: predicted_read_bytes(cfg, block_m, n, a) for k, a in alphas.items()}
        lo, hi = min(vals.values()), max(vals.values())
        out.append({"n": n, "bytes": vals, "spread_frac": (hi - lo) / lo if lo > 0 else math.inf})
    return out


# --------------------------------------------------------------------------
# THE CONTRAST. Two rivals for the missing term, and the reading that separates
# them. Everything here is re-derived from the committed session corpus.
# --------------------------------------------------------------------------

class CorpusMissing(FileNotFoundError):
    """A committed ladder the plan predicts from is not in this tree."""


class CounterRunRefused(RuntimeError):
    """A profile that cannot be reduced to a per-call reading anyone may quote."""


def one_run_dir(arm: Path) -> Path:
    """The single run directory under a published arm, or a refusal.

    Two run directories mean two different apparatus settings and picking the
    first would make the prediction depend on the sort order of a filesystem.
    """
    if not arm.exists():
        raise CorpusMissing(f"{arm} is not in this tree; the contrast has no corpus")
    runs = sorted(p for p in arm.iterdir() if p.is_dir())
    if len(runs) != 1:
        raise CorpusMissing(
            f"{arm.name} holds {len(runs)} run directories; refusing to pick one, "
            "because two runs are two apparatus settings")
    return runs[0]


def _column_matches(row: dict, key: str, want) -> bool:
    """Compare a CSV cell with a wanted value, numerically when both are numbers.

    `block_m` is written "64" by one arm and "64.0" by another, and a string
    compare silently selects nothing, which is a ladder of zero treads and an
    OLS that raises somewhere far away from the cause.
    """
    if key not in row:
        raise CorpusMissing(f"the corpus has no column {key!r}; columns are {sorted(row)}")
    got = row[key]
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return str(got) == str(want)


def corpus_slope(cells_csv: Path, select: dict, *, tile_key: str = "tiles",
                 time_key: str = "ms_p50") -> tuple[float, float, int, int]:
    """`(slope ms per tile, intercept ms, treads, rows)` for one committed ladder.

    The SAME estimator the synthesis used (`s8_common_currency.py`): median
    `ms_p50` per tile count, then OLS on the medians. Median first, because a
    tread carries up to 17 repeats and one clock excursion in a repeat should
    move the tread by nothing; OLS second, because the ladder is affine and the
    slope is the only thing read off it.

    Rows whose `status` is not `ok` are dropped. A ladder with fewer than three
    treads raises rather than returning a two-point line with no residual.
    """
    if not cells_csv.exists():
        raise CorpusMissing(f"{cells_csv} is not in this tree")
    import csv as _csv
    by_tile: dict[int, list[float]] = {}
    with cells_csv.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            if not all(_column_matches(row, k, v) for k, v in select.items()):
                continue
            by_tile.setdefault(int(float(row[tile_key])), []).append(float(row[time_key]))
    if len(by_tile) < 3:
        raise CorpusMissing(
            f"{cells_csv.parent.parent.name} at {select} has {len(by_tile)} treads; "
            "a slope needs three")
    tiles = sorted(by_tile)
    med = [statistics.median(by_tile[t]) for t in tiles]
    a, b = ols(tiles, med)
    return b, a, len(tiles), sum(len(v) for v in by_tile.values())


def corpus_knobs(cells_csv: Path, select: dict, keys: tuple[str, ...]) -> dict[str, int]:
    """The compile knobs the SELECTED rows were actually run at, read off the row.

    THE RECIPE IS A KNOB LIST AND A TYPED KNOB LIST IS A GUESS. The BLOCK_N
    cells read GROUP_SIZE_M, num_stages and num_warps out of their arm's own
    report.json; the schedule pair used to have them typed from the setting
    string (`s3w8g1` read as num_stages 3, num_warps 8, group 1), even though
    the occupancy arm's cells.csv carries `num_stages`, `num_warps` and
    `group_m` as columns. The typed values matched the corpus, so nothing was
    wrong, and nothing would have said so if a republish had moved a depth: the
    plan would have printed one recipe and predicted from another arm's rows.

    Refuses a key the rows do not carry, and refuses a key whose selected rows
    disagree: a ladder run at two depths is two ladders and its slope is not
    one cell's.
    """
    if not cells_csv.exists():
        raise CorpusMissing(f"{cells_csv} is not in this tree")
    import csv as _csv
    seen: dict[str, set[str]] = {k: set() for k in keys}
    rows = 0
    with cells_csv.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            if not all(_column_matches(row, k, v) for k, v in select.items()):
                continue
            rows += 1
            for k in keys:
                if k not in row:
                    raise CorpusMissing(
                        f"the corpus has no column {k!r}; columns are {sorted(row)}")
                seen[k].add(row[k])
    if rows == 0:
        raise CorpusMissing(f"{cells_csv} has no ok row at {select}; no knobs to read")
    out: dict[str, int] = {}
    for k, values in seen.items():
        distinct = {int(float(v)) for v in values}
        if len(distinct) != 1:
            raise CorpusMissing(
                f"{cells_csv.parent.parent.name} at {select} ran at {sorted(distinct)} "
                f"for {k!r}; that is two cells and their rows may not share one slope")
        out[k] = distinct.pop()
    return out


def occupancy_block_n(report: dict) -> tuple[int, str]:
    """The BLOCK_SIZE_N the occupancy arm pinned, from the arm's own report.

    That arm records the pin inside the text of the gate that checks it
    ("block sizes measured are exactly {64, 256} at BLOCK_SIZE_N=64, ...") and
    nowhere as a field, so this reads every `BLOCK_SIZE_N=<n>` in the report and
    accepts the value only when the report is unanimous. Otherwise it falls back
    to the sweep's own FIXED value, which is the default that arm's `--block-n`
    carries, and SAYS which of the two it used rather than presenting a constant
    as a reading.
    """
    found = {int(v) for v in re.findall(r"BLOCK_SIZE_N=(\d+)", json.dumps(report))}
    if len(found) == 1:
        return found.pop(), "the arm's own report.json"
    return (int(SWEEP_FIXED["BLOCK_SIZE_N"]),
            "block_m_crossing_sweep.FIXED, the arm's own --block-n default; its "
            f"report states {sorted(found) or 'no'} BLOCK_SIZE_N")


def weight_stream_ms(weight_bytes: int, gbps: float) -> float:
    """Milliseconds to stream `weight_bytes` once at `gbps`. The denominator of
    the common currency, and the only place the card's rate enters a prediction."""
    if gbps <= 0:
        raise ValueError("refusing to divide by a non-positive bandwidth")
    return 1e3 * weight_bytes / (gbps * 1e9)


@dataclass(frozen=True)
class ContrastCell:
    """One cell of the extended plan, with the ladder it is predicted from."""

    name: str
    role: str
    block_m: int
    block_n: int
    group_m: int
    num_stages: int
    num_warps: int
    arm: str
    cells_csv: str
    slope_ms: float
    treads: int
    reps: int
    #: Where the four compile knobs above came from. A recipe is a knob list,
    #: and a reader has to be able to tell a knob read off the committed row
    #: from a knob this file typed.
    knob_source: str = "unstated"

    def w(self, stream_ms: float) -> float:
        """Weight-streams per extra M-tile. The statistic with no fitted level,
        no intercept, no delta and no extrapolation to zero tiles in it."""
        return self.slope_ms / stream_ms

    def traffic_bytes_per_tile(self, stream_ms: float, weight_bytes: int) -> float:
        """`dR/dn` if the whole per-M-tile time is DRAM traffic at the card's own
        rate. This is the TRAFFIC rival's prediction and nothing else."""
        return self.w(stream_ms) * weight_bytes


def contrast_plan(block_m: int, *, model: str = "mixtral-8x7b") -> list[ContrastCell]:
    """The cells the extended plan profiles, each with its corpus ladder.

    WHY THESE CELLS. Section 2 of the 2026-09-10 analysis leaves one question
    open that this grid cannot answer: the measured BLOCK_N dependence is a cost
    going as `1/BLOCK_N` and NOT with BLOCK_M, and whether it is TRAFFIC or TIME
    decides whether a traffic model can contain it at all. Two cells at
    BLOCK_N=32 and BLOCK_N=128 at one BLOCK_M separate the rivals by the full
    ratio of their per-M-tile slopes. The BLOCK_N=64 cell in between is the
    plan's own primary cell and is profiled anyway, so it is a third point on
    the same curve for free.

    AND A GROUP_SIZE_M=1 CELL, because the analysis's statement (3) is that the
    per-M-tile cost is a function of the SCHEDULE: G=1 to G=16 moves it 24% with
    byte-identical shared memory and an identical instruction census. If that
    24% is traffic the swizzle is changing what the kernel reads; if it is time
    it is changing only when. The occupancy arm pinned BLOCK_M=64 and
    BLOCK_SIZE_N=64 for every setting, so the schedule pair is always read at
    BLOCK_M=64 whatever `block_m` the BLOCK_N contrast uses, and the printed
    plan says so rather than implying one BLOCK_M throughout.
    """
    del model  # every ladder below is the mixtral arm; the parameter documents that
    bn_run = one_run_dir(GAPS_SESSION / "bn_decomposition")
    bn_pinned = json.loads((bn_run / "report.json").read_text())["pinned"]
    occ_run = one_run_dir(GAPS_SESSION / "occupancy_vs_swizzle")
    occ_report = json.loads((occ_run / "report.json").read_text())
    occ_bn, occ_bn_src = occupancy_block_n(occ_report)

    cells: list[ContrastCell] = []
    for block_n, role in ((32, "BN-low"), (64, "BN-hinge, the plan's primary cell"),
                          (128, "BN-high")):
        slope, _a, treads, reps = corpus_slope(
            bn_run / "cells.csv", {"block_n": block_n, "block_m": block_m})
        cells.append(ContrastCell(
            name=f"bn{block_n}-g{bn_pinned['GROUP_SIZE_M']}-m{block_m}", role=role,
            block_m=block_m, block_n=block_n, group_m=int(bn_pinned["GROUP_SIZE_M"]),
            num_stages=int(bn_pinned["num_stages"]), num_warps=int(bn_pinned["num_warps"]),
            arm="bn_decomposition",
            cells_csv=str((bn_run / "cells.csv").relative_to(REPO)),
            slope_ms=slope, treads=treads, reps=reps,
            knob_source="GROUP_SIZE_M, num_stages and num_warps from the arm's "
                        "report.json 'pinned' block"))
    # The occupancy arm names its settings s<stages>w<warps>g<group>. s3w8g1 and
    # s3w8g16 are the matched pair the analysis's statement (3) is about: one
    # num_stages, one num_warps, one BLOCK_M, one BLOCK_SIZE_N, one swizzle apart.
    # The three knobs are READ OFF THE ROWS, not decoded from the setting name:
    # the name is a label the arm chose and the columns are what it ran.
    for setting, role in (("s3w8g1", "schedule: no swizzle"),
                          ("s3w8g16", "schedule: the shipped swizzle")):
        select = {"setting": setting, "block_m": 64}
        slope, _a, treads, reps = corpus_slope(occ_run / "cells.csv", select)
        knobs = corpus_knobs(occ_run / "cells.csv", select,
                             ("group_m", "num_stages", "num_warps"))
        cells.append(ContrastCell(
            name=f"{setting}-m64", role=role, block_m=64, block_n=occ_bn,
            group_m=knobs["group_m"], num_stages=knobs["num_stages"],
            num_warps=knobs["num_warps"], arm="occupancy_vs_swizzle",
            cells_csv=str((occ_run / "cells.csv").relative_to(REPO)),
            slope_ms=slope, treads=treads, reps=reps,
            knob_source=f"group_m, num_stages and num_warps from the arm's own "
                        f"cells.csv rows; BLOCK_SIZE_N from {occ_bn_src}"))
    return cells


#: How near a measured ratio must sit to a rival's prediction before the
#: contrast is scored as that rival, relative. Registered here, once, so the
#: plan that prints the separation and the mode that scores it use one number.
#: 5% is the accuracy a DRAM counter is quoted at; contrast A's two rivals are
#: 87% apart and contrast B's 31%, so this window decides both with room.
CONTRAST_TOLERANCE = 0.05

#: The measured ratio of one rival to the other under TIME: the same bytes,
#: moved less well, read identically by a counter.
TIME_RATIO = 1.0


@dataclass(frozen=True)
class ContrastPair:
    """Two cells, the two rival predictions for their ratio, and the reading.

    A CONTRAST IS A RATIO OF TWO CELLS AND NO SINGLE PAYLOAD CONTAINS ONE. The
    plan registers the ratio and `--contrast` measures it; both build the pair
    here, so a pair cannot be printed with one partner and scored with another.
    """

    label: str
    lo: ContrastCell
    hi: ContrastCell
    title: str
    decides: str

    @property
    def traffic_ratio(self) -> float:
        """dR/dn(lo) / dR/dn(hi) if the whole per-M-tile time is DRAM traffic.

        A ratio of two measured times: no bandwidth constant enters it, which
        is why a recalibration moves the absolute columns beside it and never
        moves this.
        """
        return self.lo.slope_ms / self.hi.slope_ms

    def separates(self, tolerance: float = CONTRAST_TOLERANCE) -> bool:
        """Can any measured ratio satisfy only one rival, rather than both?

        The two acceptance windows are the rivals' predictions widened by
        `tolerance`, and this asks whether they are disjoint. The same question
        `c1_registration` asks of the anchors, asked here for the same reason: a
        gate whose two answers both fit inside its own window cannot decide, and
        finding that out after the pod is rented is what this file spent
        2026-09-10 fixing on C1.
        """
        lo_t, hi_t = sorted((self.traffic_ratio * (1 - tolerance),
                             self.traffic_ratio * (1 + tolerance)))
        lo_i, hi_i = TIME_RATIO * (1 - tolerance), TIME_RATIO * (1 + tolerance)
        return lo_t > hi_i or hi_t < lo_i

    def read(self, ratio: float,
             tolerance: float = CONTRAST_TOLERANCE) -> tuple[str, str]:
        """`(verdict, which rival)` for a measured ratio. Registered, not fitted."""
        near_traffic = abs(ratio - self.traffic_ratio) <= tolerance * self.traffic_ratio
        near_time = abs(ratio - TIME_RATIO) <= tolerance * TIME_RATIO
        if near_traffic and not near_time:
            return PASS, "TRAFFIC"
        if near_time and not near_traffic:
            return PASS, "TIME"
        if near_traffic and near_time:
            return REFUSE, "both, which means this pair does not separate them"
        return FAIL, "neither: both rivals are refuted as stated and this is a split"


def contrast_pairs(cells: list[ContrastCell]) -> list[ContrastPair]:
    """The pairs the extended plan registers, built from the corpus ladders."""
    by_name = {c.name: c for c in cells}
    lo_bn = next(c for c in cells if c.block_n == 32 and c.arm == "bn_decomposition")
    hi_bn = next(c for c in cells if c.block_n == 128 and c.arm == "bn_decomposition")
    return [
        ContrastPair(
            "A", lo_bn, hi_bn,
            f"CONTRAST A, the one section 2 turns on. BLOCK_M={lo_bn.block_m} fixed, "
            "GROUP_SIZE_M fixed, BLOCK_N 32 against 128.",
            "a ratio near the TRAFFIC value means the missing 1/BLOCK_N term is "
            "bytes and belongs in the byte model; a ratio near 1.000 means it is "
            "time and the three-term traffic model cannot hold it. Anything in "
            "between refutes both as stated and is reported as the split it is."),
        ContrastPair(
            "B", by_name["s3w8g1-m64"], by_name["s3w8g16-m64"],
            "CONTRAST B, the schedule. BLOCK_M=64 and BLOCK_SIZE_N=64 fixed by the "
            "occupancy arm, num_stages=3, num_warps=8, GROUP_SIZE_M 1 against 16.",
            "a ratio near the TRAFFIC value means the swizzle changes what the "
            "kernel reads and alpha_b is a function of the schedule, which is what "
            "statement (3) asserts and no counter has ever tested; a ratio near "
            "1.000 means the swizzle changes only WHEN the same bytes move, and the "
            "24% it moves the per-M-tile cost is latency, not traffic."),
    ]


def contrast_lines(cells: list[ContrastCell], stream_ms: float, weight_bytes: int,
                   act_bytes_per_tile: int, bandwidth_source: str) -> list[str]:
    """The pre-registered contrast, both rivals, and which outcome means which.

    The DISCRIMINATOR IS A RATIO of two measured slopes, so it needs no assumed
    bandwidth: that is the point. The absolute columns beside it are the traffic
    rival's own predictions, which DO carry the card's measured rate, and they
    are labelled as such so the ratio is not read as resting on them.
    """
    by_name = {c.name: c for c in cells}
    out = [
        "THE CONTRAST, registered. Two rivals for the term the ladder cannot name.",
        "  TRAFFIC  the BLOCK_N dependence is bytes: a smaller N-tile re-reads more, so",
        "           dR/dn scales with the measured per-M-tile time and the two BLOCK_N",
        "           cells read different amounts. alpha_b is then itself a function of",
        "           BLOCK_N and the three-term model can be repaired inside a byte count.",
        "  TIME     the BLOCK_N dependence is schedule: the same bytes, moved less well.",
        "           dR/dn is then IDENTICAL at both BLOCK_N and the three-term traffic",
        "           model cannot contain the term at all, because alpha is measuring a",
        "           schedule and not a byte ratio.",
        "",
        f"  denominator, read not typed: {bandwidth_source}",
        f"  one full weight stream = {weight_bytes / 1e9:.4f} GB = {stream_ms:.4f} ms "
        "at this card's own measured rate",
        f"  once-read activation share a/W = {act_bytes_per_tile / weight_bytes:.4f} "
        "per M-tile. It is ADDITIVE in dR/dn, so it",
        "  cancels in a DIFFERENCE and not in a ratio, and the line that stood here "
        "said it cancelled in",
        "  the ratio: a right number with the wrong reason under it. It does not enter "
        "the discriminator",
        "  either way. The TRAFFIC ratio is the two cells' measured times divided, "
        "which neither adds nor",
        "  subtracts it, and under TIME it is the same term on both sides of a ratio "
        "of 1.000.",
        "",
        f"  {'cell':<16}{'BM':>4}{'BN':>5}{'G':>4}{'ms/tile':>10}{'w':>8}"
        f"{'TRAFFIC dR/dn':>16}{'TIME dR/dn':>13}  ladder",
    ]
    for c in cells:
        out.append(
            f"  {c.name:<16}{c.block_m:>4}{c.block_n:>5}{c.group_m:>4}"
            f"{c.slope_ms:>10.5f}{c.w(stream_ms):>8.4f}"
            f"{c.traffic_bytes_per_tile(stream_ms, weight_bytes) / 1e9:>15.4f}G"
            f"{'same for all':>13}  {c.arm} {c.treads} treads / {c.reps} reps")
    out.append("")

    def render(p: ContrastPair) -> list[str]:
        lines = [
            f"  {p.title}",
            f"    TRAFFIC predicts dR/dn({p.lo.name}) / dR/dn({p.hi.name}) = "
            f"{p.traffic_ratio:.3f}, that is "
            f"{p.lo.traffic_bytes_per_tile(stream_ms, weight_bytes) / 1e9:.2f} GB against "
            f"{p.hi.traffic_bytes_per_tile(stream_ms, weight_bytes) / 1e9:.2f} GB per M-tile",
            "    TIME    predicts the ratio 1.000, the two reads equal to the "
            "counter's own accuracy",
            f"    the two rivals are {(p.traffic_ratio - 1.0) * 100:.0f}% apart on a "
            "quantity a counter reads to a few percent",
            f"    {p.decides}",
        ]
        verdict = ("SEPARATING" if p.separates() else
                   "NOT DISCRIMINATING, and --contrast registers it as such rather "
                   "than failing a gate that cannot pass")
        lines.append(f"    scored by --contrast at +/-{CONTRAST_TOLERANCE * 100:.0f}% "
                     f"of each rival: {verdict}")
        return lines

    pairs = contrast_pairs(cells)
    out += render(pairs[0])
    out.append("")
    out += render(pairs[1])
    hinge = next(c for c in cells if c.block_n == 64 and c.arm == "bn_decomposition")
    g16 = by_name["s3w8g16-m64"]
    out += [
        "",
        "  THE TWO ROWS AT BM=64 BN=64 G=16 ARE NOT ONE CELL and the table does not",
        f"  pretend they are: {hinge.name} ran at num_stages={hinge.num_stages} in the "
        f"bn_decomposition arm and",
        f"  {g16.name} at num_stages={g16.num_stages} in the occupancy arm, which is why "
        f"their slopes are",
        f"  {hinge.slope_ms:.5f} and {g16.slope_ms:.5f} ms per tile. Each contrast is "
        "read WITHIN one arm, so the",
        "  pipeline depth is fixed inside A and inside B and the two ratios are not "
        "compared with each other.",
        "",
        "  WHAT NEITHER CONTRAST CAN DO. A DRAM counter sees the weight re-read and the",
        "  activation re-read in one number, so it cannot split dR/dn into alpha_b and",
        "  alpha_a. It settles whether the term is traffic; it does not name the operand.",
    ]
    return out


# --------------------------------------------------------------------------
# C1's registration, which depends on the cell and not on the run.
# --------------------------------------------------------------------------

#: The window C1 calls a match, in alpha. One constant, because the gate and
#: the registration-time report have to use the same number or the plan
#: promises a decision the scorer cannot make.
C1_TOLERANCE = 0.05


@dataclass(frozen=True)
class C1Registration:
    """What C1 may ask of a cell, decided by the cell's own anchors.

    WHY THIS EXISTS. C1 asked one question from the day it was written: does
    EXACTLY ONE of the registered anchors survive a window of `C1_TOLERANCE`
    around the measured alpha. That question is answerable only when no two
    anchors are within the window of each other, because a measurement that
    agrees with anchor A leaves anchor B standing whenever |A - B| <= tol.

    On the A100 cell this plan used to hold, the anchors are 0.4522 / 0.6473 /
    0.7047 and the closest pair is 0.0574 apart, just outside the window: the
    question is answerable and the gate is a CLAIM about the world. When the
    plan moved to the H200 cell on 2026-09-10 the anchors became 0.6202 /
    0.6583 / 0.6595, a closest pair of 0.0011 and a total spread of 0.0393,
    narrower than the window itself. The gate was left as it was, so a PERFECT
    measurement of that cell, landing exactly on an anchor, left all three
    standing and scored CLAIM C1 FAIL with the diagnosis "the counter did not
    separate them and the cell was badly chosen". True, and it was known before
    the run: the file's own --dry-run says the anchors agree to 0.04. A gate
    that cannot pass is not a gate, and the operator would have read a
    successful measurement as a refuted claim on the metered box.

    So the registration is computed from the anchors, printed by --dry-run and
    used by the scorer, and it says which of two questions this cell can carry:

      SEPARATING   every pair of anchors is more than `tolerance` apart. C1 is
                   the original claim: exactly one anchor survives.
      CLUSTERED    some pair is not. Separating them was never on this cell's
                   ballot, so C1 asks the question this cell CAN answer: does
                   the counter land inside the cluster at all. A FAIL there is
                   every anchor refuted at once, which is a bigger result than
                   the one the separating question asks for, and the reason
                   this stays a CLAIM gate rather than becoming a VALIDITY one:
                   scored as VALIDITY, that refutation would exit INVALID and
                   the run's own numbers would be unquotable, which is the
                   opposite of what it earned.
    """

    anchors: dict[str, float]
    tolerance: float
    min_gap: float

    @property
    def separating(self) -> bool:
        return len(self.anchors) >= 2 and self.min_gap > self.tolerance

    @property
    def claim(self) -> str:
        return ("exactly one of the registered anchors matches the counter"
                if self.separating else
                "the counter lands inside the anchor cluster this cell cannot separate")

    @property
    def threshold(self) -> str:
        return (f"exactly 1 anchor within {self.tolerance:.2f}" if self.separating else
                f"at least 1 anchor within {self.tolerance:.2f}; this cell cannot be "
                "asked for exactly one")

    @property
    def invalidates(self) -> str:
        if self.separating:
            return ("the anchor choice. Zero survivors means all of them are wrong and "
                    "the memory-branch model needs replacing, not re-anchoring; more "
                    "than one means the counter did not separate them and the cell was "
                    "badly chosen")
        return ("the ladder's agreement with the counter. A FAIL here is every "
                "registered anchor refuted at once: the fitted memory branch and the "
                "measured traffic slope are not the same quantity, and no published "
                "alpha on this card survives it")

    def verdict(self, survivors: list[str]) -> str:
        if self.separating:
            return PASS if len(survivors) == 1 else FAIL
        return PASS if survivors else FAIL

    def lines(self) -> list[str]:
        gaps = (f"closest pair {self.min_gap:.4f} apart against a tolerance of "
                f"{self.tolerance:.2f}")
        if self.separating:
            return [f"SEPARATING cell: {gaps}, so exactly one anchor can survive and "
                    "this gate is that claim"]
        return [
            f"NOT DISCRIMINATING: {gaps}, so a measurement that agrees with one anchor "
            "agrees with its neighbour too",
            "registered as such BEFORE the run, not diagnosed after it: this gate asks "
            "the question the cell can answer (is the counter inside the cluster) and "
            "never the one it cannot (which anchor is right)",
            "the claim this cell DOES carry is the BLOCK_N contrast, whose two rivals "
            "are 87% apart; see --dry-run's CONTRAST block and --contrast",
        ]


def c1_registration(anchors: dict, tolerance: float = C1_TOLERANCE) -> C1Registration:
    """The C1 question this set of anchors can be asked. One call site for the
    plan and one for the scorer, both this function."""
    values = sorted(float(v) for v in anchors.values())
    gaps = [b - a for a, b in zip(values, values[1:], strict=False)]
    return C1Registration(anchors={k: float(v) for k, v in anchors.items()},
                          tolerance=tolerance,
                          min_gap=min(gaps) if gaps else math.inf)


# --------------------------------------------------------------------------
# Gates. Validity gates say whether the run may be read at all; claim gates say
# what it decided. A validity FAIL voids every claim gate below it, and each
# gate names what its own failure invalidates.
# --------------------------------------------------------------------------

@dataclass
class Gate:
    number: str
    kind: str          # "VALIDITY" or "CLAIM"
    claim: str
    verdict: str
    measured: str
    threshold: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`'s vocabulary.

        This file has said REFUSE since it was written, for "the payload did not
        carry what this gate needs", and the shared table spells that state
        UNKNOWN. They are the same state and the table's spelling wins at the
        boundary, because `classify` refuses a verdict it does not recognise
        rather than letting it fall through every branch and be scored as
        whatever the fallthrough happens to be. UNKNOWN counts AGAINST the gate:
        a check that examined nothing reports no failures.
        """
        return (self.kind, self.number,
                exit_codes.UNKNOWN if self.verdict == REFUSE else self.verdict)

    def result_line(self) -> str:
        """The ONE line the session driver may grep for this gate.

        Rendered by `moe.bench.exit_codes.result_line`, so the prefix, the field
        order and the UNKNOWN spelling are the shared table's. Until 2026-09-03
        this file rendered every gate as `GATE B1  CLAIM  FAIL ...` and nothing
        else, so `--bracket` exited 1 with ZERO RESULT lines, `classify_text`
        raised `NoGatesScored` on a log whose process said CLAIM_FAIL, and the
        arm was the log/code disagreement shape by construction in every mode
        (exit-code review, finding 4). The `GATE` block stays beside it as
        prose for a human.
        """
        kind, name, verdict = self.scored()
        detail = f"{self.claim}: {self.measured}".replace("\n", " ")[:160]
        return exit_codes.result_line(kind, name, verdict, detail)

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"GATE {self.number:<3} {self.kind:<8} {self.verdict:<6} {self.claim}",
               f"              measured {self.measured}   gate {self.threshold}"]
        if self.verdict != PASS and self.invalidates:
            out.append(f"              a {self.verdict} here invalidates: {self.invalidates}")
        out += [f"              {line}" for line in self.lines]
        return out


def score_counter_run(payload: dict) -> tuple[list[Gate], dict]:
    """Score a measured counter run against the pre-registered gates.

    `payload` is the JSON a counter run writes; see `COUNTER_SCHEMA_TEXT`.
    Returns the gates and a summary dict. Refuses rather than defaults: a
    missing field raises, it does not become zero.
    """
    for key in ("device", "model", "block_m", "cache_control", "rows"):
        if key not in payload:
            raise KeyError(f"counter payload has no '{key}'; refusing to score a partial run")
    cfg = MODEL_CONFIGS[payload["model"]]
    bm = int(payload["block_m"])
    rows = sorted(payload["rows"], key=lambda r: int(r["n"]))
    gates: list[Gate] = []

    # V1 NON-VACUITY. A scorer that examined nothing also reports no failures.
    launched = [r for r in rows if int(r.get("launches", 0)) > 0
                and float(r.get("dram_bytes_read", 0)) > 0]
    distinct = len({int(r["n"]) for r in launched})
    gates.append(Gate(
        "V1", "VALIDITY", "the run actually profiled something at four or more tile counts",
        PASS if distinct >= 4 else FAIL,
        f"{distinct} tile counts with a launch and non-zero read bytes",
        ">= 4",
        "everything below; with fewer points the slope is not identified and no "
        "alpha may be quoted from this run"))
    if distinct < 4:
        return gates, {"alpha": None, "reason": "non-vacuity gate failed"}

    W = weight_bytes_total(cfg)
    a1 = activation_bytes_per_tile(cfg, bm)
    r1 = next((float(r["dram_bytes_read"]) for r in launched if int(r["n"]) == 1), None)

    # V2 THE BYTE MODEL ITSELF. This is the gate that has never been run. If
    # R(1) is not one compulsory weight read then `alpha` is not a re-read
    # fraction and no published alpha means what its caption says.
    if r1 is None:
        gates.append(Gate("V2", "VALIDITY", "n=1 traffic is one compulsory weight read",
                          REFUSE, "no n=1 row in the payload", "|R(1)/(W+a) - 1| <= 0.10",
                          "the byte model is unchecked; alpha keeps its units only by assumption"))
    else:
        err = abs(r1 / (W + a1) - 1.0)
        gates.append(Gate(
            "V2", "VALIDITY", "n=1 traffic is one compulsory weight read",
            PASS if err <= 0.10 else FAIL,
            f"R(1)={r1 / 1e9:.4f} GB against W+a={(W + a1) / 1e9:.4f} GB, {err * 100:.1f}% off",
            "<= 10%",
            "the units of alpha. If R(1) is not one weight read then B/L is not a "
            "re-read fraction and every published alpha, on every card, is "
            "uninterpretable rather than merely uncertain",
            ["this is a VALIDITY gate and not a claim: R(1) is identical under all "
             "three anchors, so it discriminates none of them"]))

    # V3 MONOTONICITY. Traffic that does not grow with tiles is not traffic.
    ys = [float(r["dram_bytes_read"]) for r in launched]
    mono = all(y1 < y2 for y1, y2 in zip(ys, ys[1:], strict=False))
    gates.append(Gate("V3", "VALIDITY", "read traffic increases with tile count",
                      PASS if mono else FAIL,
                      "strictly increasing" if mono else "not monotone",
                      "strictly increasing in n",
                      "the affine model. A non-monotone ladder means the profiled "
                      "launches are not all the same kernel or the cache state moved"))

    alpha, r0, resid = alpha_from_counters(launched, cfg, bm)
    gates.append(Gate("V4", "VALIDITY", "the traffic ladder is affine in the tile count",
                      PASS if resid <= 0.03 else FAIL,
                      f"max relative residual {resid * 100:.2f}%", "<= 3%",
                      "the single-slope reading. A curved ladder means alpha varies "
                      "with n and no scalar describes it"))

    # C1 WHICH ANCHOR SURVIVES, or, on a cell whose anchors are closer together
    # than the gate's own window, whether the counter is inside the cluster.
    # `c1_registration` decides which of the two questions this cell carries and
    # --dry-run prints the same decision before anything runs.
    cand = payload.get("anchors", {})
    if not cand:
        # REFUSE, like C2 and C3 do when the payload carries nothing for them.
        # This branch used to fall into the survivors arithmetic and score FAIL
        # with "survivors none", which reads as three refuted anchors when what
        # happened is that nobody registered any.
        survivors = []
        reg = c1_registration({})
        gates.append(Gate("C1", "CLAIM", "the counter decides between the registered anchors",
                          REFUSE, "no anchors in the payload",
                          f"at least one anchor within {C1_TOLERANCE:.2f}",
                          "the comparison with the ladder. Nothing was registered to "
                          "compare against and this gate examined nothing"))
    else:
        reg = c1_registration(cand)
        survivors = [k for k, v in cand.items() if abs(float(v) - alpha) <= reg.tolerance]
        lines = [f"{k:<10} {float(v):.3f}   |{float(v) - alpha:+.3f}|   "
                 f"{'SURVIVES' if abs(float(v) - alpha) <= reg.tolerance else 'REFUTED'}"
                 for k, v in sorted(cand.items())] + reg.lines()
        gates.append(Gate("C1", "CLAIM", reg.claim, reg.verdict(survivors),
                          f"alpha_measured {alpha:.4f}; survivors {survivors or 'none'}; "
                          f"closest anchor pair {reg.min_gap:.4f}",
                          reg.threshold, reg.invalidates, lines))

    # C2 THE COUNTER MUST LAND INSIDE THE COUNTER-FREE BRACKET.
    br = payload.get("bracket")
    if br is None:
        gates.append(Gate("C2", "CLAIM", "the counter agrees with the physical bracket",
                          REFUSE, "no bracket in the payload", "lo <= alpha <= hi",
                          "the cross-check between timing and traffic"))
    else:
        lo, hi = float(br[0]), float(br[1])
        gates.append(Gate("C2", "CLAIM", "the counter agrees with the physical bracket",
                          PASS if lo <= alpha <= hi else FAIL,
                          f"{alpha:.4f} against [{lo:.4f}, {hi:.4f}]", "inside",
                          "one of the two measurements. The bracket uses only measured "
                          "time and the pin rate; a counter outside it means either the "
                          "timing or the counter is wrong, and the run cannot say which"))

    # C3 THE STUDY'S ONE SURVIVING RESULT. Scored on the cap the measured
    # traffic slope implies (`cap_from_counter`), NOT on `2 BM / (alpha b)`,
    # which drops the once-read share and is an upper bound: until 2026-09-03
    # this gate printed that upper bound as "ai_cap" and compared it with the
    # ridge, so a counter alpha within a/W of BM/ridge was scored on the wrong
    # side. The registered ladder anchors are bracketed beside it through
    # `ai_model.cap_from_fitted`, since THOSE are the readings retraction (a)
    # is about.
    ridge = payload.get("ridge")
    if ridge is None:
        gates.append(Gate("C3", "CLAIM", "this BLOCK_M still cannot reach the compute roof",
                          REFUSE, "no ridge in the payload", "cap < ridge",
                          "the tile-cap result, which is scored against this card's ridge"))
    else:
        ridge = float(ridge)
        cap, uncorrected = cap_from_counter(cfg, bm, alpha)
        # cap < ridge  <=>  alpha + a/W > BM/ridge  <=>  alpha > BM/ridge - a/W
        thresh = bm / ridge - a1 / W
        lines = [
            "the counter's alpha is a TRAFFIC SLOPE, (dR/dn - a)/W, with no level in it, "
            "so the (1+phi+delta) correction a LADDER alpha needs (moe/bench/ai_model.py) "
            "does not apply; the cap is 2 BM W / (b dR/dn), the once-read share a/W "
            f"= {a1 / W:.4f} restored, and the study's 2 BM/(alpha b) = {uncorrected:.1f} "
            "drops that share and is an UPPER BOUND on the cap, not the cap",
            "read traffic only: dram_bytes_write is not in the slope, so the cap above is "
            "high by the write share; the counter cannot split the slope into alpha_b and "
            "alpha_a, which is why C1 compares it with ladder anchors in different units",
            *anchor_cap_lines(cfg, bm, payload.get("block_n"), payload.get("anchors", {}),
                              ridge),
        ]
        gates.append(Gate("C3", "CLAIM", "this BLOCK_M still cannot reach the compute roof",
                          PASS if cap < ridge else FAIL,
                          f"cap {cap:.1f} FLOP/byte from the measured slope (uncorrected "
                          f"upper bound {uncorrected:.1f}) against ridge {ridge:.2f}; "
                          f"alpha {alpha:.4f} against the threshold {thresh:.4f}",
                          f"cap < ridge, i.e. alpha > BM/ridge - a/W = {thresh:.4f}",
                          "the ONE result the 2026-09 evaluation did not kill. A cap at or "
                          "above the ridge would mean this tile height CAN reach the roof "
                          "and the cap claim must be withdrawn",
                          lines))

    return gates, {"alpha": alpha, "intercept_bytes": r0, "residual": resid,
                   "survivors": survivors,
                   "c1_mode": "separating" if reg.separating else "clustered",
                   "c1_min_anchor_gap": reg.min_gap}


#: The keys a counter payload must carry, and the ONE place they are named.
#:
#: The schema block below is prose and `build_counter_payload` is code, and a
#: page that describes a file nobody writes is this repository's most-repeated
#: defect: it had already happened here, the block saying num_stages 3 while
#: `block_m_crossing_sweep.FIXED` said 4 and the printed recipe passed no
#: --num-stages at all. So the writer checks against these tuples, and
#: `tests/test_dram_counter_route.py` checks that every one of them appears
#: literally in the block. A key added to one and not the other fails a test
#: instead of misleading a reader.
COUNTER_TOP_KEYS: tuple[str, ...] = (
    "device", "model", "dtype", "group_m", "block_n", "block_k", "num_warps",
    "num_stages", "block_m", "cache_control", "ridge", "anchors", "bracket", "rows")

#: Per row. `calls` and `calls_floor` are the normalisation and its gate, and
#: they are REQUIRED so that a reader can redo the division that produced every
#: other number in the row.
COUNTER_ROW_KEYS: tuple[str, ...] = (
    "n", "calls", "calls_floor", "launches", "dram_bytes_read", "dram_bytes_write",
    "l2_read_hit_pct", "gpu_time_ns", "by_kernel")

COUNTER_SCHEMA_TEXT = """\
{{
  "device": "{device}",       # the card slug, matching the calibration file
  "model": "{model}", "dtype": "{dtype}",
  "group_m": {group_m}, "block_n": {block_n}, "block_k": {block_k},
  "num_warps": {num_warps}, "num_stages": {num_stages}, "block_m": {block_m},
  "cache_control": "{cache}",           # ncu --cache-control; "all" or "none"
  "ridge": {ridge},                # THIS card's own calibration, never 152.8
  "anchors":  {anchors},
  "bracket":  {bracket},
  "rows": [ {{"n": 1, "calls": 11, "calls_floor": 10, "launches": 55,
             "dram_bytes_read": 2.84e9,
             "dram_bytes_write": 1.1e8, "l2_read_hit_pct": 4.2,
             "gpu_time_ns": 1.95e6,
             "by_kernel": {{"fused_moe_kernel": 2.81e9, "moe_sum": 3.0e7}} }}, ... ]
}}

  Every key is required. `--analyse` raises on a missing one rather than
  defaulting it: a zero that was never measured is the failure mode this whole
  file exists to avoid, and `--run` refuses a metric ncu did not return rather
  than writing a zero into the row. `calls` is the number of fused_experts
  calls the instrument made (warmup + iters x trials), COUNTED from the
  profile's own launch list as the launches of `{marker}`, which vLLM runs
  exactly once per call; `calls_floor` is that cell's own iters x trials, read
  back from the row the sweep wrote, and `calls` must EXCEED it because warmup
  adds calls on top. `launches` is every profiled kernel launch. The byte
  fields are PER CALL, the sums over all launches divided by `calls`, never by
  an assumed one.

  `block_k` and `num_warps` are the sweep's own FIXED values, imported rather
  than typed: this block said num_stages 3 while `block_m_crossing_sweep.FIXED`
  said 4 and the recipe passed no --num-stages, so the schema described a cell
  the recipe would not have run.
"""


# --------------------------------------------------------------------------
# --probe: which route is open on THIS machine.
# --------------------------------------------------------------------------

def _run(argv, timeout=60) -> tuple[int, str, str]:
    """Run one child, bounded, and on a timeout kill EVERYTHING IT STARTED.

    THE PROCESS GROUP IS THE POINT. `subprocess.run(..., timeout=)` kills the
    child it spawned and waits for that one only. Every caller here spawns
    `ncu`, and ncu's own child is what does the work: the probe interpreter
    that imports torch and creates a CUDA context, or a whole vLLM sweep at
    `profile_one_tile_count`. SIGKILL cannot be caught, so a killed ncu tears
    nothing down, and the grandchild is reparented to init still holding a CUDA
    context on a card this session is paying for. That was harmless while the
    probe target was `/bin/true`; since 2026-09-15 it is a torch process, and a
    torch child wedged on CUDA init is exactly the failure that reaches the
    timeout in the first place.

    So the child gets its own session (`start_new_session`), which makes it a
    process-group leader, and the timeout path signals the GROUP.

    AND CTRL-C SIGNALS IT TOO, which is the half `start_new_session` would
    otherwise take away. A child in the terminal's own process group receives
    the SIGINT the terminal sends; one in a session of its own does not, so an
    operator interrupting a wedged probe would have kept the wedged probe. The
    KeyboardInterrupt path therefore kills the group before re-raising, and it
    is the SAME kill as the timeout's, written once.
    """
    def kill_group() -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            # Every holder of the pipes is now signalled, so this returns; the
            # bound is there so a kill the kernel somehow did not deliver
            # cannot turn a timeout into a hang.
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
    except FileNotFoundError:
        return 127, "", f"{argv[0]}: not found"
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        kill_group()
        return 124, "", f"{argv[0]}: timed out after {timeout}s"
    except BaseException:
        kill_group()
        raise


def probe_capabilities() -> dict:
    """Which of the two counter capabilities does THIS process hold?

    NVIDIA's ERR_NVGPUCTRPERM page says a container may profile either because
    the host enabled it or because the container "was started with the
    appropriate permissions by passing --cap-add=SYS_ADMIN". Those are two
    different asks to a provider and a log cannot tell them apart, so the probe
    reads the capability mask rather than guessing from a failure message.

    BOTH BITS, since 2026-09-15. CAP_PERFMON also opens the gate on any driver
    from R450 on, and it is the capability a provider will actually grant, so a
    probe that reported only SYS_ADMIN sent the operator to ask for the one
    answer that is usually no.

    `cap_eff_field` IS THE RAW TOKEN and `cap_eff_bits` is measured off it,
    because the width of what the kernel printed is not recoverable from the
    value. `cap_eff` is `hex(mask)`, which strips leading zeros, and reading a
    width off THAT is a mistake this file made once already: the published
    `0xa80425fb` is ten characters because the value is small, not because the
    kernel printed a 32-bit field. Linux renders `CapEff` in sixteen digits.

    THIS IS RECORDED DETAIL, NOT A VERDICT. Neither bit decides whether the
    route is open: `probe_ncu` decides that by reading a counter. These are
    what a human needs in order to FIX a refusal, and they are reported whether
    the route is open or shut.
    """
    path = Path("/proc/self/status")
    if not path.exists():
        return {"available": False, "why": "no /proc/self/status; not a Linux container"}
    for line in path.read_text().splitlines():
        if line.startswith("CapEff:"):
            field = line.split()[1]
            mask = int(field, 16)
            return {"available": True, "cap_eff": hex(mask),
                    "cap_eff_field": field, "cap_eff_bits": 4 * len(field),
                    "sys_admin": bool(mask >> CAP_SYS_ADMIN_BIT & 1),
                    "perfmon": bool(mask >> CAP_PERFMON_BIT & 1)}
    return {"available": False, "why": "CapEff not present in /proc/self/status"}


def probe_module_flag() -> dict:
    """The host's `NVreg_RestrictProfilingToAdminUsers`, read not assumed.

    `/proc/driver/nvidia/params` is readable from inside an unprivileged
    container and reports the parameters the HOST loaded the module with. It is
    the difference between "this provider could turn counters on with a reboot"
    and "counters are already on and something else is wrong".
    """
    path = Path("/proc/driver/nvidia/params")
    if not path.exists():
        return {"available": False, "why": "no /proc/driver/nvidia/params; no NVIDIA module here"}
    for line in path.read_text().splitlines():
        if "RestrictProfilingToAdminUsers" in line:
            m = re.search(r":\s*(\d+)", line)
            if not m:
                return {"available": False, "why": f"unparsable line: {line!r}"}
            return {"available": True, "restrict": int(m.group(1)), "line": line.strip()}
    return {"available": False, "why": "the parameter is not listed by this driver"}


def ncu_probe_argv(binary: str, log_file: Path,
                   metrics: tuple[str, ...] = (NCU_PROBE_METRIC,)) -> list[str]:
    """The probe invocation: the registered metrics over one real kernel.

    The ladder family asks its one metric. The r3-arms family asks its WHOLE
    list (`r3_probe_metrics`): a probe that proves one counter readable says
    nothing about the sector and launch metrics the R3 pages are gated on.

    Four flags and each is load-bearing.

      `--launch-count 1`   ncu profiles the FIRST kernel launch and stops, and
                           one launch is all a permission check needs. WHICH
                           launch that is matters and this comment said "the
                           child launches exactly one kernel" until 2026-09-15,
                           which was false: `torch.ones` on CUDA is `empty` plus
                           a `fill_` kernel, so the profiled launch was the
                           fill and not the add the probe module described.
                           `counter_probe_kernel` now builds its buffer on the
                           host and copies it, which is a memcpy and not a
                           launch, so the in-place add IS launch zero. The
                           verdict does not depend on that -- the caller asserts
                           that a NUMBER came back and never its value -- but a
                           payload naming one kernel beside a count off another
                           is a page that cannot be read.
      `--target-processes all`  ncu attaches to the interpreter's children as
                           well, the way `ncu_argv` does; a torch that forks
                           would otherwise be profiled by nobody.
      `--csv --page raw`   the SAME page `--run` parses, because the probe is
                           read by `parse_ncu_csv`, the same parser. A probe
                           that proves a different output shape parseable
                           proves the wrong thing.
      `--log-file`         ncu's CSV and the child's own stdout otherwise
                           interleave in one stream, and the child's stdout is
                           where the probe kernel's marker line lands.
    """
    return [binary, "--metrics", ",".join(metrics),
            "--launch-count", "1", "--target-processes", "all",
            "--csv", "--page", "raw", "--log-file", str(log_file),
            "--", sys.executable, str(NCU_PROBE_KERNEL)]


def probe_kernel_word(blob: str) -> tuple[str, str]:
    """The probe child's own marker line, or `("", "")` if it printed none.

    Parsed rather than inferred from a return code: the child and ncu both
    contribute to that code and only the child knows which of its four worlds
    it landed in.
    """
    for line in blob.splitlines():
        line = line.strip()
        if not line.startswith(PK.MARKER):
            continue
        rest = line[len(PK.MARKER):].strip().split(" ", 1)
        word = rest[0] if rest else ""
        if word in PK.WORDS:
            return word, (rest[1] if len(rest) > 1 else "")
    return "", ""


def probe_ncu(family: str = LADDER_FAMILY) -> dict:
    """Is ncu installed, and CAN IT READ A COUNTER ON THIS BOX?

    THE DEFECT THIS REPLACES, measured on a rented H200 on 2026-09-15. This
    function used to run

        ncu --metrics dram__bytes_read.sum /bin/true

    and call rc 0 "attached with no permission error". `/bin/true` launches no
    CUDA kernel, so ncu attached, found nothing to profile, printed
    `==WARNING== No kernels were profiled.` and exited 0 WITHOUT EVER
    ATTEMPTING A COUNTER READ. The permission error cannot appear on that path.
    Every OPEN this probe ever returned, 2026-09-09 and 2026-09-10 included,
    was that false positive: both published payloads carry ncu's own
    "No kernels were profiled" in `output_head`, beside `cause` "attached".
    A session booked two 120-minute counter arms on the word and both died in
    35 seconds with ERR_NVGPUCTRPERM.

    SO THIS PROFILES A REAL KERNEL AND READS THE VALUE BACK. The four worlds it
    must keep apart, because a log cannot:

      no ncu on PATH          -- the tool is not installed. `present` False.
      ERR_NVGPUCTRPERM        -- ncu ran, a kernel launched, the counter read
                                 was REFUSED. THIS IS A FACT ABOUT THE POD and
                                 not a broken instrument: the host module flag
                                 or the container's capabilities, both named in
                                 the recorded detail beside this.
      counters readable       -- a kernel launched and ncu returned a NUMBER
                                 for the registered metric, parsed by the same
                                 `parse_ncu_csv` `--run` uses.
      no CUDA device          -- the probe child could not launch a kernel at
                                 all (no torch, no card, or CUDA refused to
                                 initialise). Nothing is known about counters
                                 here, and saying otherwise is the old defect.

    WHY IT IS STILL CHEAP. One child interpreter, one 4 MiB host buffer copied
    to the card and one in-place add over it, which is the first and only
    kernel the child launches. The cost is the child's torch import, which this
    repo has measured at roughly fifteen seconds; the profiling itself is a
    single launch. Against an arm the session books a minute for and counter
    arms it books two hours for, fifteen seconds to find out whether those two
    hours can happen at all is the cheapest thing in the session.
    
    THE r3-arms FAMILY ASKS MORE (`--probe --family r3-arms`, and `--run
    --family r3-arms`, which calls this). It first asks ncu which metrics this
    chip offers (`query_metric_names`), refuses when a STRICT one is absent,
    drops the cross-check and recorded ones it lacks, and then asks the probe
    kernel for every metric left. OPEN then means every STRICT metric came
    back as a number; which cross-check and recorded metrics came back is
    recorded (`metrics_proven`), and those are the only ones a page gates on.

    ONE UNKNOWN NAME MUST NOT COST THE VERDICT. A name this ncu does not know
    refuses the whole invocation, and a readable metric list does not rule
    that out: the `launch__*` names pass the list check unverified
    (`r3_probe_metrics`). So whenever the whole ask read no counter for a
    reason other than ERR_NVGPUCTRPERM, whatever the list said, the probe asks
    again: first STRICT plus the cross-check and recorded metrics the list
    verified (no `launch__*` optional), then STRICT alone, skipping an ask it
    has already made and stopping at the first ask that reads a counter or is
    refused permission. `attempts` records every ask and `first_attempt`
    says why the earlier ones read nothing; an optional metric the last ask
    left out is listed in `metrics_unproven`, so no page gates on it. Until
    2026-09-24 the retry ran only when the list could not be read, so one
    `launch__` name this ncu does not know read REFUSE on a box whose
    counters work, and `--run --family r3-arms` refused with it.
    """
    binary = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if not binary:
        return {"present": False, "counters_read": False, "why": "no ncu on PATH"}
    rc, out, err = _run([binary, "--version"], timeout=30)
    version = (out or err).strip().splitlines()[-1] if (out or err) else ""
    if family != R3_FAMILY:
        return _probe_once(binary, version, (NCU_PROBE_METRIC,), ())
    names, query = query_metric_names(binary)
    strict, optional, dropped, refused = r3_probe_metrics(names)
    if refused:
        return {"present": True, "binary": binary, "version": version,
                "family": R3_FAMILY, "returncode": None, "metric": strict,
                "probe_kernel": "NOT_RUN", "probe_kernel_detail": "",
                "counters_read": False, "permission_refused": False,
                "metric_value": None, "metrics_query": query,
                "metrics_dropped": dropped, "cause": refused, "output_head": ""}
    verified = () if names is None else tuple(m for m in optional
                                              if not m.startswith("launch__"))
    asks: list[tuple[str, ...]] = []
    for ask in (optional, verified, ()):
        if ask not in asks:
            asks.append(ask)
    attempts: list[dict] = []
    info: dict = {}
    for ask in asks:
        info = _probe_once(binary, version, strict, ask, family=R3_FAMILY)
        error = next((ln.strip() for ln in str(info.get("output_head") or "").splitlines()
                      if "==ERROR==" in ln), "")
        attempts.append({"metrics": list(strict + ask),
                         "counters_read": bool(info["counters_read"]),
                         "permission_refused": bool(info["permission_refused"]),
                         "ncu_error": error, "cause": info.get("cause")})
        if info["counters_read"] or info["permission_refused"]:
            break
    if len(attempts) > 1:
        info["first_attempt"] = "; ".join(
            f"attempt {i + 1} ({len(a['metrics'])} metrics) read no counter: "
            + (f"ncu said {a['ncu_error']!r}; " if a["ncu_error"] else "") + str(a["cause"])
            for i, a in enumerate(attempts[:-1]))
        last = set(attempts[-1]["metrics"])
        unproven = info.setdefault("metrics_unproven", {})
        for m in optional:
            if m not in last:
                unproven[m] = (f"asked by attempt 1, which read no counter, and left "
                               f"out of attempt {len(attempts)}, the one this verdict "
                               "is read off")
    info["attempts"] = attempts
    info["metrics_query"] = query
    info["metrics_dropped"] = dropped
    return info


def _probe_once(binary: str, version: str, strict: tuple[str, ...],
                optional: tuple[str, ...], family: str = LADDER_FAMILY) -> dict:
    """One profiled launch of the probe kernel asking `strict + optional`."""
    with tempfile.TemporaryDirectory(prefix="ncu-counter-probe-") as tmp:
        log_file = Path(tmp) / "probe.csv"
        rc2, out2, err2 = _run(ncu_probe_argv(binary, log_file, strict + optional),
                               timeout=NCU_PROBE_TIMEOUT_S)
        log_text = log_file.read_text() if log_file.exists() else ""
    return probe_reading(binary, version, rc2, out2, err2, log_text,
                         strict=strict, optional=optional, family=family)


def query_metric_names(binary: str) -> tuple[frozenset[str] | None, str]:
    """`(the base metric names this ncu offers on this box, how that was
    read)`, or `(None, why not)`.

    `ncu --query-metrics` lists the metrics of the attached chip, one base
    name per row (`dram__bytes_read`, with its `.sum` rollup listed beside it
    rather than in the name). Parsed as every token shaped like a metric
    name, so a column reordering in a future ncu cannot empty the set; an
    empty set, or a nonzero exit, is `None` and the caller asks the whole
    list instead.
    """
    rc, out, err = _run([binary, "--query-metrics"], timeout=120)
    names = frozenset(re.findall(r"\b[a-z][a-z0-9]*__[a-z0-9_]+\b", out or ""))
    if rc != 0 or not names:
        return None, (f"ncu --query-metrics exited {rc} and listed {len(names)} "
                      f"metric names ({(err or out or '').strip()[:160]}); the "
                      "whole list is asked of the probe kernel instead")
    return names, f"ncu --query-metrics listed {len(names)} base metric names"


def probe_reading(binary: str, version: str, returncode: int,
                  stdout: str, stderr: str, log_text: str, *,
                  strict: tuple[str, ...] = (NCU_PROBE_METRIC,),
                  optional: tuple[str, ...] = (),
                  family: str = LADDER_FAMILY) -> dict:
    """What `probe_ncu` CONCLUDES, separated from what it RUNS. Pure.

    Split out on 2026-09-15 so `--self-test` can plant the four worlds and
    check the verdict and the exit code they produce without a GPU, a driver or
    an ncu. This file already holds that a parser exercised for the first time
    on a rented box is an assertion and not a check (`do_self_test`); the logic
    below is what decides whether two 120-minute arms are bookable, and it was
    in exactly that position until this split. There is ONE copy: `probe_ncu`
    runs the child and hands its bytes here, `self_test_probe` plants bytes and
    hands them here, and neither restates a rule the other applies.

    `strict` are the metrics OPEN needs as numbers on one profiled launch;
    `optional` are asked and recorded, never required. The ladder family
    passes its one metric and nothing optional, which is the reading this
    function has always made. The r3-arms family passes its STRICT class and
    everything else it asked, and its payload gains the proven and unproven
    lists and the CSV layout the parser read.
    """
    rc2, out2, err2 = returncode, stdout, stderr
    # ncu's own errors go to the log file when one is given, and the child's
    # marker line goes to stdout. Both must be read or the diagnosis is the
    # half that happened to be in the stream someone looked at.
    blob = "\n".join(part for part in (out2, err2, log_text) if part)
    word, detail = probe_kernel_word(blob)
    info = {"present": True, "binary": binary, "version": version,
            "returncode": rc2, "metric": NCU_PROBE_METRIC,
            "probe_kernel": word or "NO_MARKER",
            "probe_kernel_detail": detail,
            "counters_read": False, "permission_refused": False,
            "metric_value": None,
            "output_head": blob.strip()[:600]}
    # ORDER MATTERS AND IS THE POINT. The permission refusal is checked first
    # because it is the one world where a kernel demonstrably launched and the
    # COUNTER was the thing that was refused. Then the counter value is read,
    # and a value that came back is the whole of OPEN: it is direct evidence,
    # and nothing the child printed can add to it or take it away. Only when no
    # value came back does the child's marker get consulted, and then only to
    # SAY WHY. The old probe had this exactly inverted: it concluded from ncu's
    # return code, which is a fact about the process, that a counter was
    # readable, which is a fact about the driver.
    if "ERR_NVGPUCTRPERM" in blob:
        info["permission_refused"] = True
        info["cause"] = ("ERR_NVGPUCTRPERM: a kernel launched and the COUNTER READ was "
                         "REFUSED by this box. That is a fact about the pod -- its host "
                         "module flag or its container capabilities, both recorded beside "
                         "this -- and not a broken instrument")
        return info
    if family == R3_FAMILY:
        info.update({"family": R3_FAMILY, "metric": list(strict),
                     "metrics_asked": list(strict) + list(optional),
                     "metrics_proven": [], "metrics_unproven": {}})
    try:
        launches = parse_ncu_csv(log_text, soft=frozenset(optional))
        first = next((lz for lz in launches if all(m in lz.metrics for m in strict)),
                     None)
        if first is None:
            raise CounterRunRefused(
                f"ncu profiled {len(launches)} launch(es) and none reported "
                + (strict[0] if len(strict) == 1 else f"every one of {list(strict)}"))
        info["counters_read"] = True
        info["metric_value"] = first.metrics[strict[0]]
        if family != R3_FAMILY:
            info["cause"] = (f"counters readable: {NCU_PROBE_METRIC} came back as "
                             f"{info['metric_value']:.6g} on a profiled launch of the "
                             "probe kernel")
            return info
        proven = [m for m in (*strict, *optional) if m in first.metrics]
        info["metrics_proven"] = proven
        info["metrics_unproven"] = {m: first.unreadable.get(m, "not reported")
                                    for m in optional if m not in first.metrics}
        info["csv_layout"], info["csv_header"] = ncu_csv_layout(log_text)
        info["cause"] = (f"counters readable: all {len(strict)} STRICT metrics of the "
                         f"r3-arms family came back as numbers on a profiled launch of "
                         f"the probe kernel ({strict[0]} {info['metric_value']:.6g}); "
                         f"{len(proven) - len(strict)} of {len(optional)} cross-check "
                         f"and recorded metrics proven; the CSV was "
                         f"{info['csv_layout'].upper()}")
        return info
    except CounterRunRefused as exc:
        unreadable = str(exc)
    if word == PK.NO_TORCH:
        info["cause"] = (f"no CUDA device reached: the probe interpreter has no torch "
                         f"({detail}), so no kernel launched and nothing was ever asked "
                         "of a counter")
    elif word == PK.NO_CUDA_DEVICE:
        info["cause"] = (f"no CUDA device at all: {detail}. ncu had nothing to profile, "
                         "so this box has said NOTHING about counter permission")
    elif word == PK.LAUNCH_FAILED:
        info["cause"] = (f"the probe kernel failed to run ({detail}), so counter "
                         "permission is UNTESTED here: no launch retired")
    elif not word:
        info["cause"] = (f"the probe kernel printed no {PK.MARKER} line, so it is not "
                         f"known whether a kernel launched at all; ncu exited {rc2} and "
                         f"returned no counter ({unreadable})")
    elif "No kernels were profiled" in blob:
        # The exact string the 2026-09-09 and 2026-09-10 payloads carried in
        # `output_head` while reporting "attached with no permission error". It
        # now decides, and it decides against.
        info["cause"] = ("the probe kernel ran and ncu profiled NO kernels, so no counter "
                         "was read; this is the shape the /bin/true probe mistook for "
                         "permission")
    else:
        info["cause"] = (f"the probe kernel ran and ncu returned no readable "
                         f"{NCU_PROBE_METRIC}: {unreadable}")
    return info


def counter_route_is_open(ncu: dict) -> bool:
    """THE test for "a counter can be read here", written once.

    It is a function and not two inlined string comparisons because it WAS two.
    `route_verdict` and `do_run` each spelled out `cause.startswith("attached")`
    separately, so the probe's verdict and the runner's pre-flight were one
    rule at two call sites, which is this repository's most-repeated defect and
    the reason a broken premise reached a rented box twice. Anything that needs
    to know whether the route is open asks here.

    The test is `counters_read`, a boolean set by the one branch of `probe_ncu`
    that actually parsed a number out of a profile. It is NOT the prose in
    `cause`: prose is for the operator, and the old rule keyed on prose.
    """
    return bool(ncu.get("present")) and bool(ncu.get("counters_read"))


def probe_nsys() -> dict:
    """Is nsys installed, and is its IMPORTER half present?

    THE FAILURE THIS EXISTS TO NAME. On the 2026-09-01 H200 pod every nsys
    attempt -- including the CONTROL that requested no GPU metrics at all --
    died with "The importer binary and its dependencies were not found", and the
    session's own report concluded "no invocation of this nsys sampled a DRAM
    metric on this device". That conclusion does not follow: a control that
    needs no counters failed the same way, so the ladder discriminated nothing
    about counter permission. The importer lives in `host-linux-x64/` and the
    pod had only the target half installed.
    """
    binary = shutil.which("nsys")
    if not binary:
        return {"present": False, "why": "no nsys on PATH"}
    rc, out, err = _run([binary, "--version"], timeout=30)
    version = (out or err).strip()
    real = Path(binary).resolve()
    roots = [real.parent, real.parent.parent, *Path("/opt/nvidia/nsight-systems").glob("*")]
    found = []
    for root in roots:
        for cand in (root / "host-linux-x64" / "QdstrmImporter",
                     root / "QdstrmImporter"):
            if cand.exists():
                found.append(str(cand))
    return {"present": True, "binary": binary, "resolved": str(real),
            "version": version, "importers": sorted(set(found)),
            "importer_present": bool(found)}


def route_verdict(caps: dict, flag: dict, ncu: dict, nsys: dict) -> tuple[str, list[str]]:
    """One of five verdicts, and the specific next action for each.

    REFUSE is a verdict here. "Probably blocked" is what has kept this question
    open for two weeks; a probe that cannot tell says so.
    """
    notes = []
    if counter_route_is_open(ncu) and ncu.get("family") == R3_FAMILY:
        unproven = ncu.get("metrics_unproven") or {}
        dropped = ncu.get("metrics_dropped") or []
        return "OPEN", [
            f"ncu read every STRICT metric of the r3-arms family off a profiled launch "
            f"of the probe kernel, from a {str(ncu.get('csv_layout')).upper()} CSV: the "
            "R3 counter run can happen on this box. Next: --run --family r3-arms "
            "--census-only, then one --run --family r3-arms --group-m G --census "
            "<census.json> per registered G (--dry-run --family r3-arms prints them "
            "and their price).",
            f"proven beyond STRICT: {ncu.get('metrics_proven', [])[len(R3_STRICT_METRICS):]}"
            f"; not proven, so never gated: {sorted(unproven) or 'none'}; dropped by "
            f"the metric query: {dropped or 'none'}"
            + (f"; {ncu['first_attempt']}" if ncu.get("first_attempt") else "")]
    if counter_route_is_open(ncu):
        return "OPEN", [f"ncu read {ncu.get('metric')} off a profiled launch of the probe "
                        f"kernel ({ncu.get('metric_value')}): a counter is READABLE on "
                        "this box, which is the thing the two counter arms need and the "
                        "thing the /bin/true probe never checked. Read the plan with "
                        "--dry-run and then take it with --run, which drives ncu over one "
                        "cell and writes the JSON --analyse scores. Run every cell of a "
                        "registered pair and score the RATIO with --contrast: the "
                        "discriminator is across cells and --analyse sees one. This is "
                        "the decisive route."]
    if ncu.get("present") and ncu.get("permission_refused"):
        # A FLAG SET BY THE PROBE, not a substring of its prose. Keying this on
        # `"ERR_NVGPUCTRPERM" in cause` read BLOCKED off any message that merely
        # MENTIONED the error -- including `parse_ncu_csv`'s own refusal, which
        # now tells the operator to grep the log for exactly that token. A
        # verdict that moves when a sentence is rewritten is not a verdict.
        # RECORDED DETAIL, NOT GATES. ERR_NVGPUCTRPERM already decided the
        # verdict; none of the lines below can change it. They exist so the next
        # ask of the provider is the right ask, which is the one thing a tenant
        # can act on.
        if caps.get("available") and not caps.get("sys_admin") and not caps.get("perfmon"):
            notes.append("this process holds NEITHER CAP_SYS_ADMIN NOR CAP_PERFMON. "
                         "Either opens the gate on a driver from R450 on and PERFMON is "
                         "the narrower ask, so ask the provider for --cap-add=PERFMON "
                         "first, --cap-add=SYS_ADMIN second, and a host reboot last.")
        elif caps.get("available") and not caps.get("sys_admin"):
            notes.append("this process holds CAP_PERFMON but NOT CAP_SYS_ADMIN and the "
                         "counters were still refused. PERFMON suffices only from driver "
                         "R450 on, so read the driver version before asking for a "
                         "capability this container already has.")
        elif caps.get("available") and not caps.get("perfmon"):
            notes.append("this process holds CAP_SYS_ADMIN but NOT CAP_PERFMON and the "
                         "counters were still refused. SYS_ADMIN is the capability "
                         "NVIDIA's own page names, so this is not a capability problem "
                         "and the raw ncu output needs reading.")
        if caps.get("available"):
            # THE MASK ITSELF, so the three capability lines above can be
            # checked rather than believed. There was a "the field was too
            # narrow to carry bit 38" note here until this commit and it could
            # never fire: `cap_eff_bits` is the width of the RAW /proc field,
            # and Linux prints sixteen digits whatever the value. `perfmon` is
            # a real reading of bit 38 on any box that reaches this line.
            notes.append(f"the effective capability mask read "
                         f"{caps.get('cap_eff_field', caps.get('cap_eff'))} "
                         f"({caps.get('cap_eff_bits')} bits as the kernel printed it), "
                         f"so CAP_PERFMON (bit {CAP_PERFMON_BIT}) is "
                         f"{'set' if caps.get('perfmon') else 'clear'} and CAP_SYS_ADMIN "
                         f"(bit {CAP_SYS_ADMIN_BIT}) is "
                         f"{'set' if caps.get('sys_admin') else 'clear'} by measurement.")
        if flag.get("available") and flag.get("restrict") == 1:
            notes.append("the host loaded the module with "
                         "RestrictProfilingToAdminUsers=1; the host-side fix is a module "
                         "parameter change and a reload, which a tenant cannot do.")
        if flag.get("available") and flag.get("restrict") == 0:
            notes.append("the host ALREADY allows unprivileged profiling "
                         "(RestrictProfilingToAdminUsers=0) yet ncu still refused. That "
                         "combination is not explained by the module flag and needs the "
                         "raw ncu output read, not another retry.")
        if not flag.get("available"):
            notes.append(f"the host module parameter could not be read "
                         f"({flag.get('why')}), so 'the host forbids it' and 'this "
                         "container lacks the capability' cannot be separated from "
                         "inside; the capability line above is the side a tenant can act "
                         "on.")
        return "BLOCKED", notes
    if ncu.get("present") and ncu.get("probe_kernel") in (
            PK.NO_TORCH, PK.NO_CUDA_DEVICE, PK.LAUNCH_FAILED, "NO_MARKER"):
        # ncu is here and the box never got a kernel, so NOTHING is known about
        # counter permission. This is REFUSE and not BLOCKED, and the difference
        # is the whole defect: BLOCKED is a measured refusal that the ledger
        # files as a finding, and "we could not ask" is not a finding.
        notes.append(f"ncu is installed but the probe launched no kernel to count "
                     f"({ncu.get('probe_kernel')}: {ncu.get('probe_kernel_detail')}). "
                     "Counter permission is UNTESTED here, not open and not blocked: on "
                     "a box with no CUDA device there is no counter question to answer, "
                     "and on one with a device this means the probe interpreter cannot "
                     "reach it. Fix the interpreter, then probe again.")
    if not ncu.get("present"):
        notes.append("no ncu here. It installs from the public CUDA apt tree as a plain "
                     "file: the nsight-compute-* debs sit beside the nsight-systems-* ones "
                     "and `dpkg -x` unpacks either without root.")
    if nsys.get("present") and not nsys.get("importer_present"):
        notes.append("nsys is installed WITHOUT its importer, which is the 2026-09-01 pod's "
                     "failure exactly. Any capture here writes a .qdstrm that this machine "
                     "cannot convert. See docs/COUNTERS.md for the version-matched fix.")
    return "REFUSE", notes or ["not enough evidence on this machine to name the route"]


def probe_gates(verdict: str, ncu: dict) -> list[Gate]:
    """P1, or NO GATE AT ALL when the route is REFUSE. Written once.

    REFUSE means the box did not say enough to name a route: nothing was
    established, so there is nothing to score and `exit_codes` documents a log
    with zero RESULT lines beside exit 2 as exactly that shape. OPEN and
    BLOCKED are both MEASURED, so both print P1 -- PASS and FAIL -- and the
    ledger files either as finished.
    """
    if verdict == REFUSE:
        return []
    return [Gate(
        "P1", "CLAIM", "a DRAM counter can be READ on this box",
        PASS if verdict == "OPEN" else FAIL,
        f"route {verdict}: {ncu.get('cause', ncu.get('why', 'no ncu'))}",
        "OPEN",
        "the counter experiment on this box; the plan stands and needs "
        "another box or a provider-side change named in the notes above")]


def probe_exit(verdict: str, gates: list[Gate]) -> int:
    """The code `--probe` returns. REFUSED before any gate, else the table's."""
    if verdict == REFUSE:
        return exit_codes.REFUSED
    return exit_codes.classify(g.scored() for g in gates)


def do_probe(args) -> int:
    family = getattr(args, "family", LADDER_FAMILY)
    caps, flag = probe_capabilities(), probe_module_flag()
    ncu, nsys = probe_ncu(family), probe_nsys()
    verdict, notes = route_verdict(caps, flag, ncu, nsys)
    print("ROUTE PROBE" + (f"  family {R3_FAMILY}: every metric its pages ask"
                           if family == R3_FAMILY else ""))
    print(f"  host      {os.uname().sysname} {os.uname().machine}")
    print(f"  caps      {caps}")
    print(f"  module    {flag}")
    print(f"  ncu       {ncu.get('cause', ncu.get('why'))}"
          + (f"  [{ncu.get('version', '')}]" if ncu.get("present") else ""))
    if ncu.get("present"):
        # Printed separately from the cause because it is the half the old probe
        # never had: which world the CHILD landed in is what decides whether the
        # ncu line above is about counters at all.
        print(f"  kernel    {ncu.get('probe_kernel')}  {ncu.get('probe_kernel_detail', '')}")
    if family == R3_FAMILY and ncu.get("present"):
        print(f"  metrics   query: {ncu.get('metrics_query', 'not run')}")
        print(f"            proven {ncu.get('metrics_proven', [])}")
        print(f"            unproven {sorted(ncu.get('metrics_unproven') or {})}"
              f"  dropped {ncu.get('metrics_dropped') or []}")
        if ncu.get("csv_layout"):
            print(f"  csv       {ncu['csv_layout']}: {str(ncu.get('csv_header'))[:160]}")
    if nsys.get("present"):
        # "importer MISSING" is only meaningful when nsys is here at all; printing it
        # for a machine with no nsys would report the pod's failure on a laptop.
        print(f"  nsys      importer={'present' if nsys['importer_present'] else 'MISSING'}"
              f"  {nsys.get('version', '')}")
    else:
        print(f"  nsys      {nsys.get('why', 'absent')}")
    print(f"\n  VERDICT   {verdict}")
    for n in notes:
        print(f"            - {n}")
    # THE PROBE IS A SCORED GATE, so its log carries the one line the driver
    # greps and `classify_text` over it recomputes the code the process
    # returns. Until 2026-09-03 this mode scored nothing and returned DONE for
    # both OPEN and BLOCKED: a log with zero RESULT lines beside exit 0, which
    # `exit_codes` documents as the shape a REFUSED log has, and the driver's
    # summary printed "NOT scored" beside a finished arm. The pre-registered
    # expectation is that a counter can be READ; BLOCKED is that expectation
    # refuted by the box, a CLAIM_FAIL, which the table defines as "measured,
    # the world disagreed": a RESULT the ledger files as finished and the
    # summary prints as the finding, never a retry. It is the answer this arm
    # was written to obtain and it now says so in the greppable line. REFUSE
    # stays REFUSED (2), before any gate: the box did not say enough to name a
    # route, nothing was established, and there is no gate to print.
    gates = probe_gates(verdict, ncu)
    if gates:
        print()
        for g in gates:
            for line in g.render():
                print(line)
    body = {"verdict": verdict, "notes": notes, "capabilities": caps,
            "module_flag": flag, "ncu": ncu, "nsys": nsys,
            "gates": [asdict(g) for g in gates]}
    if family == R3_FAMILY:
        body["family"] = R3_FAMILY
    payload = stamped(body, mode="probe", args=args, card=live_card(),
                      instrument=PROBE_INSTRUMENT)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n  wrote {out}")
        print(f"  git   {git_visibility(out)}")
    return probe_exit(verdict, gates)


# --------------------------------------------------------------------------
# --run: drive ncu, reduce the profile PER CALL, write the schema.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Launch:
    """One profiled kernel launch and the registered metrics it reported."""

    launch_id: str
    kernel: str
    metrics: dict[str, float]
    #: A SOFT metric (asked, never gated) whose cell could not be read, and
    #: why. Empty unless the caller named soft metrics: every other metric
    #: still refuses the whole parse on an unreadable cell.
    unreadable: dict[str, str] = field(default_factory=dict)


def _metric_value(metric: str, unit: str, raw: str) -> float:
    """One CSV cell, in the metric's canonical unit, or a refusal.

    THREE WAYS THIS CELL IS NOT A NUMBER AND ALL THREE MUST REFUSE.
    ncu writes `n/a` for a metric the device or the replay could not supply, it
    writes an empty cell for a metric it did not collect at all, and it groups
    thousands with a comma under a locale that does. The first two are the
    failure this file exists to avoid: a zero that was never measured fits an
    affine line as happily as a real one and nothing downstream can tell. The
    third is a parse bug that turns 2,818,572,288 into 2.
    """
    text = (raw or "").strip().replace(",", "")
    if text == "" or text.lower() in {"n/a", "na", "nan", "-"}:
        raise CounterRunRefused(
            f"ncu returned {raw!r} for {metric}; a metric that was not measured "
            "is REFUSED and never defaulted to 0.0")
    try:
        value = float(text)
    except ValueError as exc:
        raise CounterRunRefused(f"{metric}: cannot read {raw!r} as a number") from exc
    canonical, table = unit_table(metric)
    key = (unit or "").strip()
    if key not in table:
        raise CounterRunRefused(
            f"{metric} came back in unit {unit!r}, which this parser has never been "
            f"shown; known units are {sorted(table)} reduced to {canonical}. Add it "
            "deliberately rather than scaling it by 1. An EMPTY unit is refused here "
            "too, and used to be scaled by 1: ncu rescales per launch, so a metric "
            "with no unit beside it cannot be reduced")
    return value * table[key]


#: The two shapes ncu's CSV is parsed in. LONG is one row per (launch,
#: metric) under `ID`, `Kernel Name`, `Metric Name`, `Metric Unit`, `Metric
#: Value`, which is the shape this parser was written against. WIDE is one row
#: per launch with one column per metric and a units row under the header.
#: NO LIVE ncu CSV HAS EVER BEEN CAPTURED IN THIS REPOSITORY (the only raw ncu
#: output committed is an ERR_NVGPUCTRPERM, `profiles/q2_kernel_names.txt`),
#: NVIDIA's CLI documentation does not state which shape `--csv --page raw`
#: prints, and the recollection this file's r3-arms family was designed on is
#: that it is WIDE. A parser that read one shape would turn a box whose
#: counters work into a probe that reads REFUSE ("no ncu CSV header"), which
#: is the gate that decides a booking. So both are read, the probe records
#: which one it parsed (`ncu_csv_layout`), and the r3-arms family keeps the
#: `.ncu-rep` so the CSV can be regenerated off the box.
CSV_LONG = "long"
CSV_WIDE = "wide"


def _ncu_csv_header(text: str) -> tuple[str, list[str], list[list[str]]]:
    """`(layout, header, rows after the header)`, or the no-header refusal.

    Everything before the header is skipped rather than parsed. ncu prefixes
    its own progress with `==PROF==` and the profiled process writes its own
    stdout into the same stream when no `--log-file` is given; `--run` always
    passes one, and this still skips, because a parser that trusts line 1 is
    a parser that breaks the first time ncu prints a warning.
    """
    import csv as _csv
    rows = list(_csv.reader(text.splitlines()))
    for i, row in enumerate(rows):
        cols = set(row)
        if {"Kernel Name", "Metric Name", "Metric Value"} <= cols:
            return CSV_LONG, row, rows[i + 1:]
        if "Kernel Name" in cols and any(registered_metric(c) for c in cols):
            return CSV_WIDE, row, rows[i + 1:]
    raise CounterRunRefused(
        "no ncu CSV header in this output: expected a row carrying 'Kernel "
        "Name', 'Metric Name' and 'Metric Value' (one row per launch and "
        "metric) or 'Kernel Name' beside a registered metric's own column (one "
        "row per launch). THIS IS NOT A FLAG "
        "PROBLEM -- `ncu_argv` already passes --csv --page raw --log-file, and "
        "telling the operator to pass them is where this message used to send "
        "them. A header-less log means ncu collected nothing: either no kernel "
        "was launched inside the profiled process, or the counter read was "
        "refused (grep the log for ERR_NVGPUCTRPERM). `--probe` separates those "
        "two and a re-run with the same flags will not")


def ncu_csv_layout(text: str) -> tuple[str, str]:
    """`(layout, the header row as ncu printed it)`, for the probe's record:
    the first box whose counters work closes the question of which shape
    `--csv --page raw` prints, and this is where the answer is written down."""
    layout, header, _rows = _ncu_csv_header(text)
    return layout, ",".join(f'"{c}"' for c in header)


def _no_id_refusal(header: list[str]) -> CounterRunRefused:
    return CounterRunRefused(
        f"the ncu CSV has no 'ID' column (columns: {header}). Without a launch "
        "id two launches of one kernel cannot be told apart and the per-call "
        "division would be wrong by their count")


def _read_value(launch_id: str, metric: str, unit: str, raw: str, soft,
                metrics: dict, unreadable: dict) -> None:
    """One cell into `metrics`, or, for a SOFT metric only, its refusal into
    `unreadable`. A hard metric's refusal propagates and refuses the file."""
    try:
        metrics[metric] = _metric_value(metric, unit, raw)
    except CounterRunRefused as exc:
        if metric not in soft:
            raise
        unreadable[metric] = f"launch {launch_id}: {exc}"


def parse_ncu_csv(text: str, *, soft=frozenset()) -> list[Launch]:
    """`ncu --csv --page raw` output to one `Launch` per profiled launch.

    BOTH LAYOUTS (`CSV_LONG`, `CSV_WIDE`; see there for why). In the long one
    the launches are recovered by grouping on the launch ID column; in the
    wide one each row is a launch and the row under the header carries the
    units. The ID is REQUIRED in both: without it the only other way to tell
    two launches of the same kernel apart is the kernel time string, which
    repeats, and merging two launches into one halves the traffic the
    reduction then divides by the call count. The units are REQUIRED in both,
    as the `Metric Unit` column or as the units row: ncu rescales per launch.

    `soft` names metrics that are asked and never gated (the r3-arms family's
    RECORDED class, and at the probe every metric outside the STRICT class):
    an unreadable cell of one of those lands in `Launch.unreadable` instead of
    refusing the file. Every other metric refuses on an unreadable cell, as
    it always has.
    """
    layout, header, rows = _ncu_csv_header(text)
    if "ID" not in header:
        raise _no_id_refusal(header)
    if layout == CSV_WIDE:
        return _parse_wide(header, rows, frozenset(soft))
    return _parse_long(header, rows, frozenset(soft))


def _parse_long(header: list[str], rows: list[list[str]], soft) -> list[Launch]:
    if "Metric Unit" not in header:
        # REQUIRED, and it was optional until 2026-09-10: the column was picked
        # up `if name in header` and its absence left `unit` empty, which the
        # byte tables then scaled by 1.0. A file ncu had rescaled to Mbyte was
        # read as bytes, 1e6 low, and still affine in n, so no gate below could
        # see it. The unit table is the only defence against ncu's per-launch
        # rescaling and a parser that will run without it has no defence.
        raise CounterRunRefused(
            f"the ncu CSV has no 'Metric Unit' column (columns: {header}). ncu "
            "rescales per launch, so a value without its unit cannot be reduced "
            "to bytes and would be off by whatever prefix ncu chose. AND THE FIX IS "
            "NOT --csv --page raw, which is what this message used to advise and "
            "what `ncu_argv` already passes: a raw page that emits the other columns "
            "and not this one is a different ncu from the one this parser was "
            "written against, so read the log's own header and the ncu version "
            "before changing any flag")
    idx = {name: header.index(name) for name in
           ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")}
    order: list[str] = []
    seen: dict[str, tuple[str, dict[str, float], dict[str, str]]] = {}
    for row in rows:
        if len(row) != len(header) or row == header:
            continue
        metric = row[idx["Metric Name"]].strip()
        if not registered_metric(metric):
            continue
        launch_id = row[idx["ID"]].strip()
        kernel = row[idx["Kernel Name"]].strip()
        if launch_id not in seen:
            order.append(launch_id)
            seen[launch_id] = (kernel, {}, {})
        known_kernel, metrics, unreadable = seen[launch_id]
        if metric in metrics or metric in unreadable:
            raise CounterRunRefused(
                f"launch {launch_id} reports {metric} twice; the ID column is not "
                "unique in this file and the launches cannot be separated")
        if kernel != known_kernel:
            raise CounterRunRefused(
                f"launch {launch_id} is named both {known_kernel!r} and {kernel!r}")
        _read_value(launch_id, metric, row[idx["Metric Unit"]],
                    row[idx["Metric Value"]], soft, metrics, unreadable)
    if not order:
        raise CounterRunRefused(
            "the ncu CSV carries no row for any registered metric. A profile that "
            "measured nothing also reports no failures")
    return [Launch(i, seen[i][0], seen[i][1], seen[i][2]) for i in order]


def _parse_wide(header: list[str], rows: list[list[str]], soft) -> list[Launch]:
    id_at, kernel_at = header.index("ID"), header.index("Kernel Name")
    columns = [(i, name) for i, name in enumerate(header) if registered_metric(name)]
    body = [r for r in rows if len(r) == len(header) and r != header]
    if not body or body[0][id_at].strip():
        # THE UNITS ROW IS THE WIDE LAYOUT'S `Metric Unit` COLUMN, and it is
        # required for the same reason: ncu rescales per launch.
        raise CounterRunRefused(
            f"the wide ncu CSV has no units row under its header (columns: "
            f"{header}); the first row under it carries launch id "
            f"{body[0][id_at]!r}" if body else
            f"the wide ncu CSV has a header and no rows under it (columns: "
            f"{header})")
    units, launches = body[0], body[1:]
    order: list[str] = []
    seen: dict[str, Launch] = {}
    for row in launches:
        launch_id = row[id_at].strip()
        if not launch_id:
            continue
        if launch_id in seen:
            raise CounterRunRefused(
                f"launch {launch_id} appears twice; the ID column is not unique in "
                "this file and the launches cannot be separated")
        metrics: dict[str, float] = {}
        unreadable: dict[str, str] = {}
        for i, name in columns:
            _read_value(launch_id, name, units[i], row[i], soft, metrics, unreadable)
        seen[launch_id] = Launch(launch_id, row[kernel_at].strip(), metrics, unreadable)
        order.append(launch_id)
    if not order:
        raise CounterRunRefused(
            "the wide ncu CSV carries a header and a units row and no launch. A "
            "profile that measured nothing also reports no failures")
    return [seen[i] for i in order]


def normalise_per_call(launches: list[Launch], *, calls_floor: int,
                       call_marker: str = CALL_MARKER,
                       gemm_marker: str = GEMM_MARKER) -> dict:
    """Sum the profile and divide the byte fields by the calls it actually made.

    THE TRAP THIS FUNCTION IS. The instrument under the profiler is
    `moe.bench.timing`, which warms up for a DURATION and sizes `iters` from
    that load, so one profiled invocation is `warmup + iters x trials`
    fused_experts calls, each of several kernel launches. The counter integrates
    all of them. Dividing the byte total by an assumed 1 inflates every reading
    by that count and STILL FITS AN AFFINE LINE, because the count is the same
    at every tile count, so the residual gate passes, the monotonicity gate
    passes, and only the n=1 validity gate would notice, against a bound of
    10%, when the error is a factor of eleven.

    So `calls` is COUNTED, from the launch list, as the number of launches whose
    kernel name contains `call_marker`. vLLM runs `moe_align_block_size` exactly
    once per `fused_experts` call, before either GEMM, so it counts calls without
    assuming how many GEMM or reduction launches a given vLLM emits per call.

    `calls_floor` is the cell's own `iters x trials`, read from the row the sweep
    wrote. The count must EXCEED it, because warmup runs at least one more call
    on top; a count at or below the floor means the marker matched something
    other than one-per-call and the division is not the division it looks like.

    Two aggregations that are not sums, and are labelled rather than hidden:
      * the L2 read hit RATE is a ratio and cannot be added across launches. It
        is returned as a duration-weighted mean, with the range beside it, and
        it is NOT a sector-weighted rate: the sector counts were not among the
        four registered metrics. Read the range before quoting the mean.
      * `gpu_time_ns` is a sum over launches divided by calls, like the bytes,
        so it is per call. Under `--replay-mode kernel` it is replay time and it
        is not comparable with the unprofiled ladder's wall clock.
    """
    missing = {m: [ln.launch_id for ln in launches if m not in ln.metrics]
               for m in NCU_METRICS}
    missing = {m: ids for m, ids in missing.items() if ids}
    if missing:
        raise CounterRunRefused(
            "these registered metrics are absent from launches "
            + "; ".join(f"{m} on launch(es) {ids[:5]}" for m, ids in missing.items())
            + ". Every registered metric is required on every profiled launch: a "
            "missing one is REFUSED, never taken as 0.0")
    calls = sum(1 for ln in launches if call_marker in ln.kernel)
    if calls == 0:
        names = sorted({ln.kernel for ln in launches})
        raise CounterRunRefused(
            f"no profiled kernel name contains {call_marker!r}, so the call count "
            f"cannot be counted and nothing may be divided by it. Kernels in this "
            f"profile: {names}. Pass --call-marker with the name this vLLM uses")
    if calls <= calls_floor:
        raise CounterRunRefused(
            f"counted {calls} {call_marker!r} launches against a floor of "
            f"iters x trials = {calls_floor}. The instrument runs warmup calls on "
            "top of the timed ones, so the count must EXCEED the floor; a count at "
            "or below it means the marker is not one-per-call and the per-call "
            "division would be wrong by the difference")
    gemm = sum(1 for ln in launches if gemm_marker in ln.kernel)
    read = sum(ln.metrics["dram__bytes_read.sum"] for ln in launches)
    write = sum(ln.metrics["dram__bytes_write.sum"] for ln in launches)
    time_ns = sum(ln.metrics["gpu__time_duration.sum"] for ln in launches)
    by_kernel: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ln in launches:
        by_kernel[ln.kernel] = by_kernel.get(ln.kernel, 0.0) + ln.metrics["dram__bytes_read.sum"]
        counts[ln.kernel] = counts.get(ln.kernel, 0) + 1
    hits = [ln.metrics["lts__t_sector_op_read_hit_rate.pct"] for ln in launches]
    weights = [ln.metrics["gpu__time_duration.sum"] for ln in launches]
    total_w = sum(weights)
    hit = (sum(h * w for h, w in zip(hits, weights, strict=True)) / total_w
           if total_w > 0 else statistics.fmean(hits))
    return {
        "calls": calls,
        "calls_floor": calls_floor,
        "launches": len(launches),
        "dram_bytes_read": read / calls,
        "dram_bytes_write": write / calls,
        "l2_read_hit_pct": hit,
        "l2_read_hit_pct_range": [min(hits), max(hits)],
        "l2_read_hit_pct_basis": "duration-weighted mean over profiled launches; "
                                 "NOT a sector-weighted rate",
        "gpu_time_ns": time_ns / calls,
        "by_kernel": {k: v / calls for k, v in sorted(by_kernel.items())},
        "launches_by_kernel": dict(sorted(counts.items())),
        "gemm_launches": gemm,
        "gemm_launches_per_call": gemm / calls,
    }


def sweep_argv(args, n: int, out_dir: Path) -> list[str]:
    """The unprofiled sweep invocation `--run` wraps, one tile count.

    `--no-l2-flush` IS PASSED AND THE PRINTED RECIPE DID NOT PASS IT. The
    instrument's software flush is a 240 MB ATen reduction executed once per
    timed iteration (`moe.bench.timing.L2Flusher`, 4x this card's 60 MB L2), and
    under a profiler it is a launch like any other, so its DRAM reads land in
    the same `dram__bytes_read.sum` total as the kernel under test: at ten
    iterations that is 2.4 GB against a 2.82 GB weight read, an 85% inflation of
    R(n) that is CONSTANT in n and therefore invisible to the slope gate, the
    monotonicity gate and the residual gate, and would fail only the n=1 byte
    model gate, which would be read as the byte model being wrong. Under the
    counter, ncu's own `--cache-control` sets the cache state, which is why that
    is the swept parameter; the software flush would be a second, unmeasured one.
    """
    return [sys.executable, str(REPO / "scripts" / "block_m_crossing_sweep.py"),
            "--model", args.model, "--dtype", args.dtype,
            "--tiles", str(args.block_m),
            "--group-m", str(args.group_m), "--block-n", str(args.block_n),
            "--num-stages", str(args.num_stages),
            "--r-max", str(n * args.block_m), "--row-step", str(n * args.block_m),
            "--step-probes", "0", "--warmup", "1", "--trials", "1",
            "--cell-budget-ms", "1", "--no-l2-flush", "--out", str(out_dir)]


def ncu_common_flags(cache_control: str) -> list[str]:
    """The replay and cache flags every profiled invocation in this file
    passes, written once: the ladder family's `ncu_argv` and the r3-arms
    family's `r3_ncu_argv` both take them from here. Kernel replay, so every
    profiled launch is replayed once per metric pass, and ncu's cache control,
    which with "all" flushes every cache before each pass."""
    return ["--replay-mode", "kernel", "--cache-control", cache_control]


def ncu_argv(binary: str, cache_control: str, log_file: Path) -> list[str]:
    """The profiler wrapper. `--log-file` is not optional here: without it ncu's
    CSV and the profiled process's own stdout interleave in one stream."""
    return [binary, "--metrics", ",".join(NCU_METRICS),
            *ncu_common_flags(cache_control),
            "--csv", "--page", "raw", "--target-processes", "all",
            "--log-file", str(log_file)]


def sweep_cell_row(out_dir: Path) -> dict:
    """The one row the wrapped sweep wrote, or a refusal.

    TWO REFUSALS, both of which would otherwise be a silently wrong reading.
    More than one `cells.csv` means more than one sweep wrote under this
    directory and the iters the profile is divided against would come from
    whichever one sorted first. More than one `ok` row means the profiled
    process measured more than one cell, so the counter integrated two
    geometries into one byte total and the tile count the row claims is not the
    only tile count in the profile.
    """
    found = sorted(out_dir.rglob("cells.csv"))
    if len(found) != 1:
        raise CounterRunRefused(
            f"{len(found)} cells.csv under {out_dir}; expected exactly one. The "
            "call floor comes from the row the sweep wrote and there must be one row")
    import csv as _csv
    with found[0].open(newline="") as fh:
        rows = [r for r in _csv.DictReader(fh) if r.get("status") == "ok"]
    if len(rows) != 1:
        raise CounterRunRefused(
            f"{len(rows)} ok rows in {found[0]}; expected exactly one. More than one "
            "cell under one profile means the byte total spans two geometries")
    return rows[0]


def profile_one_tile_count(args, n: int, binary: str, profile_dir: Path) -> dict:
    """Profile one tile count and reduce it to one schema row.

    Returns the row `--analyse` scores, with the reduction's own bookkeeping
    beside it: the call count it divided by, the floor that count had to clear,
    and the per-kernel split, so a reader can redo the division.
    """
    log_file = profile_dir / f"counters-n{n}-cc{args.cache_control}.csv"
    out_dir = profile_dir / f"sweep-n{n}-cc{args.cache_control}"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ncu_argv(binary, args.cache_control, log_file) + sweep_argv(args, n, out_dir)
    rc, out, err = _run(argv, timeout=args.ncu_timeout)
    if not log_file.exists():
        raise CounterRunRefused(
            f"ncu exited {rc} and wrote no {log_file.name}. stderr head: "
            f"{(err or out).strip()[:400]}")
    launches = parse_ncu_csv(log_file.read_text())
    cell = sweep_cell_row(out_dir)
    calls_floor = int(float(cell["iters"])) * int(float(cell["trials"]))
    row = normalise_per_call(launches, calls_floor=calls_floor,
                             call_marker=args.call_marker)
    row.update({"n": n, "rows_per_expert": n * args.block_m,
                "ncu_returncode": rc, "ncu_log": str(log_file.relative_to(REPO))
                if str(log_file).startswith(str(REPO)) else str(log_file),
                "sweep_iters": int(float(cell["iters"])),
                "sweep_trials": int(float(cell["trials"])),
                "sweep_ms_p50": float(cell["ms_p50"])})
    return row


def build_counter_payload(args, *, ridge: float, ridge_source: str, anchors: dict,
                          bracket: tuple[float, float], contrast: dict | None,
                          rows: list[dict]) -> dict:
    """The payload `--run` writes, checked against the schema it advertises.

    Refuses on a key the schema names and this function does not produce. That
    is the same rule `score_counter_run` applies from the other side, applied at
    the WRITE site as well as the read site: a rule enforced at one of two call
    sites is not enforced, and every field here is one `--analyse` will later
    refuse to default.
    """
    report = Path(args.report)
    try:
        report_name = str(report.resolve().relative_to(REPO))
    except ValueError:
        report_name = str(report)
    payload = {
        "device": args.card, "model": args.model, "dtype": args.dtype,
        "group_m": args.group_m, "block_n": args.block_n,
        "block_k": SWEEP_FIXED["BLOCK_SIZE_K"], "num_warps": SWEEP_FIXED["num_warps"],
        "num_stages": args.num_stages, "block_m": args.block_m,
        "cache_control": args.cache_control,
        "ridge": ridge, "ridge_source": ridge_source,
        "anchors": anchors, "bracket": [bracket[0], bracket[1]],
        "anchor_report": report_name,
        "call_marker": args.call_marker,
        "contrast": contrast,
        "rows": rows,
    }
    absent = sorted(k for k in COUNTER_TOP_KEYS if payload.get(k) is None)
    if absent:
        raise CounterRunRefused(
            f"the schema names {absent} and this payload has none; refusing to write "
            "a file --analyse would then have to default")
    for row in rows:
        missing = sorted(set(COUNTER_ROW_KEYS) - set(row))
        if missing:
            raise CounterRunRefused(
                f"row n={row.get('n')} is missing {missing}, which the schema names. "
                "A metric that was not measured is REFUSED, never written as 0.0")
    return payload


def do_run(args) -> int:
    """Take the measurement, write the schema, score it.

    Refuses before touching the GPU when the route is not open, when `--out` is
    absent, or when this card has no committed calibration: the ridge the C3
    gate scores against is that file's, and a run whose ridge is guessed decides
    the study's one surviving result against a number nobody measured.
    """
    if not args.out:
        print("REFUSE: --run needs --out <path>; a measurement nobody wrote down "
              "is not a measurement.")
        return exit_codes.REFUSED
    ncu = probe_ncu()
    if not counter_route_is_open(ncu):
        # THE SAME predicate `route_verdict` uses, called rather than restated.
        # Until 2026-09-15 this line spelled out `cause.startswith("attached")`
        # for itself, so the probe and the runner were one rule at two call
        # sites and both read a premise that had never been checked.
        print("REFUSE: no counter could be read on this box. --probe says: "
              f"{ncu.get('cause', ncu.get('why'))}")
        return exit_codes.REFUSED
    ridge, ridge_src = measured_ridge(args.card)
    if ridge is None:
        print(f"REFUSE: {ridge_src}. The C3 gate is scored against this card's own "
              "ridge and it is not in this tree.")
        return exit_codes.REFUSED
    peak = DATASHEET_PEAK_GBPS.get(args.card)
    cfg = MODEL_CONFIGS[args.model]
    bm = args.block_m
    W, a = weight_bytes_total(cfg), activation_bytes_per_tile(cfg, bm)
    anc, _rep = anchors_from_report(Path(args.report), bm)
    anchors = anc.as_dict()
    if args.anchor:
        anchors.update(dict(args.anchor))
    bracket = physical_bracket(anc.slope, anc.t1_ms, W, a, peak)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.profile_dir) if args.profile_dir else \
        out.parent / f"{out.stem}.profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"DRAM COUNTER RUN  id={plan_id(args)}")
    print(f"  ncu             {ncu.get('binary')}  [{ncu.get('version', '')}]")
    print(f"  cell            {args.model} {args.dtype} BLOCK_SIZE_M={bm} "
          f"BLOCK_SIZE_N={args.block_n} GROUP_SIZE_M={args.group_m} "
          f"num_stages={args.num_stages}")
    print(f"  cache-control   {args.cache_control}")
    print(f"  profiles        {profile_dir}")
    print()
    rows = []
    for n in args.tiles:
        row = profile_one_tile_count(args, n, ncu["binary"], profile_dir)
        rows.append(row)
        print(f"  n={n:<3} {row['launches']:>4} launches / {row['calls']:>3} calls "
              f"(floor {row['calls_floor']})  read {row['dram_bytes_read'] / 1e9:8.4f} "
              f"GB per call  L2 hit {row['l2_read_hit_pct']:5.1f}%")
    print()

    cells = None
    try:
        cells = contrast_plan(bm, model=args.model)
    except CorpusMissing as exc:
        print(f"  contrast prediction NOT stamped: {exc}")
    contrast = None
    if cells is not None:
        gbps, gbps_src = measured_bandwidth_gbps(args.card)
        if gbps is not None:
            stream_ms = weight_stream_ms(W, gbps)
            # num_stages is part of the match, not decoration: the BLOCK_N
            # hinge and the schedule pair's G=16 cell are the same geometry at
            # two pipeline depths, and matching on geometry alone would stamp
            # one arm's prediction on the other arm's cell.
            mine = next((c for c in cells
                         if (c.block_m, c.block_n, c.group_m, c.num_stages)
                         == (bm, args.block_n, args.group_m, args.num_stages)), None)
            contrast = {
                "weight_stream_ms": stream_ms, "bandwidth_source": gbps_src,
                "cells": [dict(asdict(c),
                               w=c.w(stream_ms),
                               traffic_bytes_per_tile=c.traffic_bytes_per_tile(stream_ms, W))
                          for c in cells],
                "this_cell": mine.name if mine else None,
                "traffic_prediction_bytes_per_tile":
                    mine.traffic_bytes_per_tile(stream_ms, W) if mine else None,
                "time_prediction": "dR/dn identical across BLOCK_N at fixed BLOCK_M",
            }

    payload = build_counter_payload(args, ridge=ridge, ridge_source=ridge_src,
                                    anchors=anchors, bracket=bracket,
                                    contrast=contrast, rows=rows)
    out.write_text(json.dumps(
        stamped(payload, mode="run", args=args, card=live_card(),
                instrument=RUN_INSTRUMENT), indent=2))
    print(f"wrote {out}")
    print(f"git   {git_visibility(out)}")
    print()
    gates, summary = score_counter_run(payload)
    for g in gates:
        for line in g.render():
            print(line)
    if summary.get("alpha") is not None:
        print(f"\n  alpha measured directly from DRAM traffic: {summary['alpha']:.4f}")
    return exit_codes.classify(g.scored() for g in gates)


# --------------------------------------------------------------------------
# Where the output lands, asked of git rather than assumed.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """ASK GIT whether it would keep `--out`. Never assert it, never skip it.

    `.gitignore` ignores `results/*` and re-includes only `results/published/`,
    so `--out results/bracket.json` is silently dropped by `git add -A` and the
    operator is told nothing. This repo has already lost every published plot of
    ten arms that way. `--bracket` is the mode this file's own header calls "the
    part that produces a result rather than a plan", and `--run` is the mode
    that takes the measurement on a metered box, so both write output that is
    exactly the kind meant to be kept and both are checked.

    rc 0 ignored, rc 1 kept, anything else UNVERIFIED and said so. rc 128 is
    what `git check-ignore` returns for a path outside the work tree -- the pod
    default `/workspace/...` -- and calling that "tracked" is the same loss in
    the other direction.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=str(REPO), capture_output=True, timeout=15,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path UNVERIFIED"
    if proc.returncode == 0:
        return ("IGNORED by git: `git add -A` will not pick this up. Write it "
                "under results/published/<arm>/ if it is meant to be kept.")
    if proc.returncode == 1:
        return "git WILL KEEP this path."
    return (f"git check-ignore exited {proc.returncode}; path UNVERIFIED "
            f"({proc.stderr.decode(errors='replace').strip()})")


# --------------------------------------------------------------------------
# Provenance: which commit, which machine, which knobs produced this JSON.
# --------------------------------------------------------------------------

def live_card() -> str:
    """The card this box has, or `NO_CARD`.

    Never raises and never guesses: a torch that is installed and broken raises
    OSError on a missing libcudart rather than ImportError, and a probe report
    is not worth taking down over the name of a device it did not use.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            if name:
                return str(name)
    except Exception:                                     # noqa: BLE001
        return NO_CARD
    return NO_CARD


def run_id_for(mode: str, args, card: str) -> str:
    """A run id naming the card and the knobs THIS mode consumes.

    IT IS A FIELD IN THE JSON, NOT THE PATH. `--out` is an operator-chosen file
    path and the session driver passes one (`--out $SESSION/counter_route.json`),
    so deriving a directory here would break the caller; what the audit found
    missing is not a directory, it is the record of WHICH knobs produced the
    file, since two `--bracket` runs over different `--published` sets and two
    `--analyse` runs with different `--anchor` overrides overwrite each other in
    silence. With the id inside, a reader of the overwritten file can at least
    tell that it is not the one they wrote.

    Each mode hashes only what it reads. `--bracket` never looks at `--model` or
    `--tiles`, and putting them in would make two ids differ where the rows
    cannot, which teaches a reader that a difference means nothing.
    """
    if mode == "probe":
        knobs: dict = {"mode": mode}
        if getattr(args, "family", LADDER_FAMILY) == R3_FAMILY:
            knobs["family"] = R3_FAMILY
    elif mode == "bracket":
        knobs = {"mode": mode,
                 "published": sorted(str(p) for p in (args.published or []))}
    elif mode == "analyse":
        # ONE payload keeps the id it always had; several are the r3-arms
        # family's summary, keyed on the set of pages it joined.
        paths = list(args.analyse) if isinstance(args.analyse, list) else [args.analyse]
        knobs = {"mode": mode,
                 "payload": str(paths[0]) if len(paths) == 1 else sorted(map(str, paths)),
                 "anchor": dict(args.anchor or {})}
        if getattr(args, "timed_reference", None):
            knobs["timed_reference"] = sorted(map(str, args.timed_reference))
    elif mode == "contrast":
        # The payloads compared and the window that scores them. Two contrasts
        # over different payload sets, or the same set at a different tolerance,
        # are two readings and must not share an id.
        knobs = {"mode": mode, "payloads": sorted(str(p) for p in args.contrast),
                 "tolerance": CONTRAST_TOLERANCE}
    elif mode == "self-test":
        knobs = {"mode": mode}
    elif mode in ("r3-run", "r3-census"):
        # The r3-arms family's page and census: the cell (model, dtype, the
        # pinned tile, the G), the treads, and the call schedule. `--card` is
        # not a knob here: the page's card is the live device, in `card=`.
        knobs = {"mode": mode, "family": R3_FAMILY, "model": args.model,
                 "dtype": args.dtype, "group_m": args.group_m,
                 "block_n": args.block_n, "block_m": args.block_m,
                 "num_stages": args.num_stages, "tiles": list(args.tiles),
                 "calls": R3_CALLS_PER_CELL, "warmups": R3_WARMUP_CALLS}
    elif mode == "run":
        # Everything the profiler consumed, and the two knobs a timed run does
        # not have: the cache-control mode and the marker the call count was
        # counted with. Two runs that differ in either are two different
        # measurements and their ids must differ.
        knobs = {"mode": mode, "calibration_card": args.card, "model": args.model,
                 "dtype": args.dtype,
                 "group_m": args.group_m, "block_n": args.block_n,
                 "block_m": args.block_m, "num_stages": args.num_stages,
                 "tiles": list(args.tiles), "cache": args.cache_control,
                 "call_marker": args.call_marker, "metrics": list(NCU_METRICS),
                 "report": str(args.report), "anchor": dict(args.anchor or {})}
    else:
        # `calibration_card` is the --card knob, which selects the ridge, the
        # pin rate and the streaming rate every prediction is derived from. It
        # is NOT the `card=` argument beside it: that one is the device this
        # process found, and a plan mode finds none.
        knobs = {"mode": mode, "calibration_card": args.card, "model": args.model,
                 "dtype": args.dtype,
                 "group_m": args.group_m, "block_n": args.block_n,
                 "block_m": args.block_m, "tiles": list(args.tiles),
                 "cache": args.cache_control, "report": str(args.report),
                 "anchor": dict(args.anchor or {})}
    return PV.run_id(card=card, **knobs)


def stamped(payload: dict, *, mode: str, args, card: str,
            instrument: str, **known) -> dict:
    """`payload` with a provenance block, the audit's five top-level keys and
    the run id.

    EVERY JSON THIS SCRIPT WRITES GOES THROUGH HERE. Before 2026-09-02 the
    `--probe` payload had keys
    `['capabilities','module_flag','ncu','notes','nsys','verdict']` and the
    `--bracket` payload had `['gates','rows']`: no commit, no card, no
    timestamp, nothing. `--bracket` is the mode this file's own header calls
    "the part that produces a result rather than a plan", and it was
    unattributable; `--probe` describes a MACHINE and never named the machine,
    so two pods' probes were indistinguishable.
    """
    prov = PV.provenance_block(instrument=instrument, **known)
    return prov.stamp({**payload, "run_id": run_id_for(mode, args, card)})


# --------------------------------------------------------------------------
# --bracket: the counter-free result, computed over published reports.
# --------------------------------------------------------------------------

def card_key(directory: Path) -> str | None:
    """The device slug inside a published run directory's name.

    Published directories are `<date>-<device_slug>-<label>`, and the device
    slug is what selects the datasheet ceiling. Returns None rather than a
    default: bracketing an A100 fit against an H200 pin rate would silently
    widen every bound by 2.4x, which is the same class of mistake as the stale
    ridge already in the reports.
    """
    name = directory.name
    for key in DATASHEET_PEAK_GBPS:
        if key in name:
            return key
    return None


def bracket_directory(directory: Path, *, skip_block_n: tuple[int, ...] = (256,)) -> list[dict]:
    """Every identifiable fit in one published run, with its bracket.

    `skip_block_n` defaults to the BLOCK_SIZE_N=256 arm, whose eight cells were
    withdrawn by the 2026-09 evaluation: its compute reference at BLOCK_M=256
    took 249.765 ms against 5.724 ms for the identical setting in its BN=64
    twin, and the qualification test passed it because that test checks
    PROPORTIONALITY and never checks LEVEL. Nothing derived from those cells is
    read here.
    """
    key = card_key(directory)
    if key is None:
        raise ValueError(f"{directory.name}: no known device slug; refusing to pick a pin rate")
    peak = DATASHEET_PEAK_GBPS[key]
    out = []
    for path in sorted(directory.glob("*.report.json")):
        rep = json.loads(path.read_text())
        cfg = MODEL_CONFIGS[rep["model"]]
        bn = int(rep["fixed"]["BLOCK_SIZE_N"])
        if bn in skip_block_n:
            continue
        for bm_s, ladder in sorted(rep["ladder"].items(), key=lambda kv: int(kv[0])):
            if ladder.get("alpha") is None or ladder["memory_points"] < 3:
                continue
            bm = int(bm_s)
            try:
                anc = anchors_from_points(ladder["points"], ladder["memory_points"])
            except ValueError:
                continue
            W = weight_bytes_total(cfg)
            a1 = activation_bytes_per_tile(cfg, bm)
            lo, hi = physical_bracket(anc.slope, anc.t1_ms, W, a1, peak)
            achieved = (W + a1) / (anc.t1_ms * 1e-3) / 1e9
            out.append({
                "card": key, "peak_gbps": peak, "run": directory.name,
                "model": rep["model"], "group_m": int(rep["fixed"]["GROUP_SIZE_M"]),
                "block_n": bn, "block_m": bm, "treads": ladder["memory_points"],
                "published": anc.published, "t1": anc.t1, "n3": anc.n3,
                "lo": lo, "hi": hi, "achieved_frac_peak": achieved / peak,
                "reported_alpha": float(ladder["alpha"]),
                "fit_err": float(ladder["mean_rel_err"]),
            })
    return out


def do_bracket(args) -> int:
    roots = [Path(p) for p in args.published] if args.published else \
        sorted((REPO / "results" / "published").glob("*alpha-surface*"))
    if not roots:
        print("REFUSE: no published alpha-surface directories found and none given")
        return exit_codes.REFUSED
    rows: list[dict] = []
    for r in roots:
        rows.extend(bracket_directory(r))
    if not rows:
        print("REFUSE: examined the directories and found no identifiable fit. "
              "A check that examined nothing also reports no failures.")
        return exit_codes.REFUSED

    print("THE COUNTER-FREE BRACKET")
    print()
    print("  alpha >= B/t(1)                 t(1) is one weight read PLUS non-negative extras")
    print("  alpha <= B*peak/(W + a)         no traffic moves faster than the pin rate")
    print("  Neither bound uses the fitted intercept, which is the unidentified quantity.")
    print("  `peak` is the DATASHEET pin rate, quoted not measured; the softer measured")
    print("  ceiling would tighten every bound below and is not used.")
    print()
    print("  THE TWO VIOLATIONS ARE NOT THE SAME FINDING.")
    print("    ABOVE  refutes: the fit asserts traffic moving faster than the pin rate.")
    print("    below  does not: it only says the fitted branch sits above the measured")
    print("           n=1 tread, which is a fit and noise question, not a physics one.")
    print("    A violation smaller than the fit's own mean relative error decides nothing")
    print("    and is not counted.")
    print()
    hdr = (f"  {'card':<6}{'model':<15}{'G':>3}{'BM':>5}{'pub':>7}{'t1':>7}{'n>=3':>7}"
           f"{'lo':>7}{'hi':>7}  {'pub':<7}{'n>=3':<7}{'%pk':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_pub_above = n_n3_above = n_pub_below = n_n3_below = 0

    def _mark(value: float, r: dict) -> str:
        tol = r["fit_err"]      # the fit's own residual; a violation inside it decides nothing
        if value > r["hi"] * (1 + tol):
            return "ABOVE"
        if value < r["lo"] * (1 - tol):
            return "below"
        return "in"

    for r in rows:
        pub_m, n3_m = _mark(r["published"], r), _mark(r["n3"], r)
        n_pub_above += pub_m == "ABOVE"
        n_n3_above += n3_m == "ABOVE"
        n_pub_below += pub_m == "below"
        n_n3_below += n3_m == "below"
        card = "A100" if "a100" in r["card"] else "H200"
        print(f"  {card:<6}{r['model']:<15}{r['group_m']:>3}{r['block_m']:>5}"
              f"{r['published']:>7.3f}{r['t1']:>7.3f}{r['n3']:>7.3f}"
              f"{r['lo']:>7.3f}{r['hi']:>7.3f}  "
              f"{pub_m:<7}{n3_m:<7}{r['achieved_frac_peak'] * 100:>6.1f}")
    print()
    widths = [r["hi"] - r["lo"] for r in rows]
    gates = [
        Gate("B0", "VALIDITY", "the bracket examined real fits",
             PASS if len(rows) >= 8 else FAIL, f"{len(rows)} identifiable fits", ">= 8",
             "every line below; a scan over nothing reports no violations"),
        Gate("B1", "CLAIM", "every published alpha is physically possible",
             PASS if n_pub_above == 0 else FAIL,
             f"{n_pub_above} of {len(rows)} ABOVE their own pin-rate bound", "0 above",
             "the published alpha for each ABOVE row. Those fits assert a memory branch "
             "that moves the compulsory weight bytes faster than the pin rate",
             ["registered prediction: FAIL, on the A100 BLOCK_M=32 fits with a swizzle",
              f"{n_pub_below} row(s) sit below the lower bound; that is a fit question, "
              f"not a physics one, and is not counted here"]),
        Gate("B2", "CLAIM", "the n>=3 refit anchor is physically possible",
             PASS if n_n3_above == 0 else FAIL,
             f"{n_n3_above} of {len(rows)} ABOVE their own pin-rate bound", "0 above",
             "the n>=3 anchor wherever it is ABOVE. Dropping the low treads raises alpha "
             "in 12 of 12 A100 fits and on these cells it raises it past the pin rate",
             ["registered prediction: FAIL, by a wider margin than B1",
              f"{n_n3_below} row(s) below the lower bound, again not counted"]),
        Gate("B3", "CLAIM", "the bracket is tight enough to replace a counter",
             PASS if statistics.median(widths) <= 0.10 else FAIL,
             f"median width {statistics.median(widths):.3f} alpha", "<= 0.10",
             "nothing already published, but a FAIL is the argument FOR the counter run: "
             "the bracket excludes anchors, it does not pin a number"),
    ]
    for g in gates:
        for line in g.render():
            print(line)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(
            stamped({"rows": rows, "gates": [asdict(g) for g in gates]},
                    mode="bracket", args=args, card=NO_CARD,
                    instrument=INSTRUMENT), indent=2))
        print(f"\nwrote {out}")
        print(f"git   {git_visibility(out)}")
    # `return 0 if <condition> else 0` is what stood here: both branches were
    # DONE, so B0's non-vacuity check and the three CLAIM gates could not reach
    # the exit code at all. The shared table decides it now, and B1/B2 are
    # REGISTERED to fail, so the honest code for this mode today is CLAIM_FAIL
    # (1) -- a result, never a retry.
    return exit_codes.classify(g.scored() for g in gates)


# --------------------------------------------------------------------------
# --dry-run: the registered plan.
# --------------------------------------------------------------------------

def calibration_path(card: str) -> Path:
    """Where this card's measured calibration lives. One place, so a mode that
    reads the ridge and a mode that reads the bandwidth cannot read two files."""
    return REPO / "moe" / "bench" / "hardware" / f"measured_{card}.yaml"


def measured_bandwidth_gbps(card: str) -> tuple[float, str] | tuple[None, str]:
    """This card's OWN achieved streaming rate, GB/s, from its OWN calibration.

    THE DENOMINATOR OF THE CONTRAST, and the one number in this file that a
    stale constant would silently corrupt. The 2026-09-10 recalibration moved
    this card's triad figure and with it the ridge (152.8 -> 155.93) and the
    bf16 peak (668.5 -> 682.09), so a per-M-tile cost quoted in weight-stream
    units against a transcribed rate is wrong by that ratio and looks fine.
    Read, never typed. `detail.bandwidth_patterns` carries the other patterns
    for anyone who wants a different denominator, and the pattern this returns
    is named in the source string beside the number.
    """
    path = calibration_path(card)
    if not path.exists():
        return None, f"no calibration at {path.relative_to(REPO)}"
    import yaml
    d = yaml.safe_load(path.read_text())
    gbps = float(d["memory"]["bandwidth_tb_s"]) * 1000.0
    pattern = d.get("detail", {}).get("ceiling_pattern", "unstated")
    return gbps, f"{gbps:.1f} GB/s, '{pattern}' pattern, from {path.name}"


def measured_ridge(card: str) -> tuple[float, str] | tuple[None, str]:
    """This card's OWN ridge, from its OWN calibration file.

    NOT `RIDGE_BAND[0]`. All seven published A100 reports CARRIED ridge=160.3
    until `scripts/rescore_published_reports.py` rewrote them on 2026-09-02,
    because the sweep's `--ridge` default was the H200's band and
    cross_card_surface.sh never passed it; the A100's own contemporaneous
    calibration is 262.371/1.79936 = 145.81, which is what they carry now.
    Repeating the old default here would put a withdrawn H200 number in the
    gate that scores the study's one surviving result, so this reads the yaml
    and REFUSES when it is absent.
    """
    import yaml
    path = calibration_path(card)
    if not path.exists():
        return None, f"no calibration at {path.relative_to(REPO)}"
    d = yaml.safe_load(path.read_text())
    tf = d["compute_dense_tflops"]["bf16"]
    bw = d["memory"]["bandwidth_tb_s"]
    return tf / bw, f"{tf:.3f} TFLOP/s / {bw:.5f} TB/s from {path.name}"


def plan_id(args) -> str:
    """Every swept parameter, or two settings collide and the second reports
    the first's numbers. The knobs here are the CARD, the cell (model, dtype, G,
    BN, BM and the pipeline depth), the ladder (the tile list), and the two
    things a counter run varies that a timed run does not: the cache-control
    mode and the metric set.

    `num_stages` joined this on 2026-09-10 with the extended plan. It was not a
    knob while the plan held one cell at the sweep's FIXED depth; the schedule
    contrast runs the occupancy arm's pair at depth 3 against the BLOCK_N cells
    at depth 4, and without it those two ids are the same string.

    `card` joined it in the same round's repair, one line late. `--card` became
    a first-class knob on 2026-09-10 (two choices, with the ridge, the pin rate
    and the streaming rate all read per card and the default moved a100 ->
    h200), and it was left out of the id, so the same cell planned for the A100
    and for the H200 produced the identical string while every prediction under
    it differed. Same shape as the `num_stages` omission repaired above it.
    """
    return (f"{args.card}-{args.model}-{args.dtype}-g{args.group_m}-n{args.block_n}"
            f"-m{args.block_m}-s{args.num_stages}"
            f"-t{'.'.join(str(t) for t in args.tiles)}"
            f"-cc{args.cache_control}-{len(NCU_METRICS)}metrics")


def anchors_from_report(path: Path, block_m: int) -> tuple[Anchors, dict]:
    """Read the three anchors, t(1) and the slope out of a PUBLISHED report.

    Registering the plan against numbers typed into this file would let the plan
    and the run drift apart, which is the failure that produced a gate whose
    "measured" field was an algebraic restatement of its own input. Everything
    the plan predicts therefore comes from the report the run will be compared
    against, and the report path is printed with the prediction.
    """
    rep = json.loads(path.read_text())
    ladder = rep["ladder"].get(str(block_m))
    if ladder is None or ladder.get("alpha") is None or ladder["memory_points"] < 3:
        raise ValueError(f"{path.name} has no identifiable BLOCK_M={block_m} ladder")
    return anchors_from_points(ladder["points"], ladder["memory_points"]), rep


def do_dry_run(args) -> int:
    cfg = MODEL_CONFIGS[args.model]
    bm = args.block_m
    W = weight_bytes_total(cfg)
    a = activation_bytes_per_tile(cfg, bm)
    card = args.card
    ridge, ridge_src = measured_ridge(card)
    peak = DATASHEET_PEAK_GBPS.get(card)

    print(f"DRAM COUNTER RUN -- PLAN  id={plan_id(args)}")
    print()
    print("THE CELL, which is not a new one. It is the cell the alpha surface already")
    print("measured, so the counter answers the ladder rather than a different question.")
    ridge_note = (f"   ridge {ridge:.2f} FLOP/byte  ({ridge_src})" if ridge
                  else f"   RIDGE REFUSED: {ridge_src}")
    print(f"  card            {card}{ridge_note}")
    print(f"  model           {args.model} {args.dtype}  E={cfg.num_experts} k={cfg.top_k} "
          f"H={cfg.hidden_size} F={cfg.intermediate_size}")
    print(f"  pinned          GROUP_SIZE_M={args.group_m} BLOCK_SIZE_N={args.block_n} "
          f"BLOCK_SIZE_M={bm}")
    print(f"  tiles per expert {list(args.tiles)}   rows per expert "
          f"{[t * bm for t in args.tiles]}")
    print(f"  W  compulsory   {W / 1e9:.5f} GB   ({cfg.num_experts} experts x 3FH x 2 B)")
    print(f"  a  per tile     {a / 1e6:.3f} MB   (E x BM x (2H+3F) x 2 B)")
    print()
    print(f"THE METRIC: {NCU_METRICS[0]}, plus {len(NCU_METRICS) - 1} companions.")
    for m in NCU_METRICS:
        print(f"  {m}")
    print()

    if peak is None:
        print(f"REFUSE: no datasheet pin rate for {card}; the bracket cannot be stated.")
        return exit_codes.REFUSED

    # The anchors come from the PUBLISHED report the run will be compared
    # against, never from constants typed here, so the plan and the run cannot
    # drift apart. --anchor overrides them only for a what-if.
    bracket = None
    if args.report:
        anc, _rep = anchors_from_report(Path(args.report), bm)
        anchors = anc.as_dict()
        bracket = physical_bracket(anc.slope, anc.t1_ms, W, a, peak)
        print(f"ANCHORS, read from {Path(args.report).name}")
        print(f"  memory branch   t(1) = {anc.t1_ms:.5f} ms   A = {anc.intercept:.5f} ms   "
              f"B = {anc.slope:.6f} ms/tile")
        print(f"  counter-free bracket  [{bracket[0]:.4f}, {bracket[1]:.4f}]  "
              f"(width {bracket[1] - bracket[0]:.4f})")
    else:
        print("REFUSE: --dry-run needs --report <published report json> so the plan is "
              "registered against the numbers the run will be scored against.")
        return exit_codes.REFUSED
    if args.anchor:
        anchors.update(dict(args.anchor))
    order = [k for k, _ in sorted(anchors.items(), key=lambda kv: kv[1])]
    print()
    # C1, REGISTERED, AND IT WAS NOT. The gate asked for exactly one surviving
    # anchor on a cell whose anchors are 0.0393 apart end to end, inside its own
    # 0.05 window: a perfect measurement scored CLAIM C1 FAIL and exited 1 on
    # the metered box. What a cell can be asked is a property of its anchors, so
    # it is decided here, before the run, by the same function the scorer calls.
    reg = c1_registration(anchors)
    print("C1, REGISTERED: what this cell can be asked about the anchors, decided")
    print("  from the anchors themselves and printed before anything runs.")
    print(f"  anchors        {', '.join(f'{k} {anchors[k]:.4f}' for k in order)}")
    print(f"  closest pair   {reg.min_gap:.4f} against C1's own window "
          f"{reg.tolerance:.2f}")
    print(f"  registered as  {'SEPARATING' if reg.separating else 'NOT DISCRIMINATING'}: "
          f"{reg.claim}")
    for line in reg.lines():
        print(f"    {line}")
    print()
    print("C3, REGISTERED: what each anchor implies for this tile's AI cap, and how")
    print("  the counter will score it. A ladder anchor is a B/(A+B) fit, so the")
    print("  study's 2 BM/(alpha b) reading is HIGH by (1 + phi + delta) (retraction")
    print("  (a), moe/bench/ai_model.py); alpha_a is unmeasured, so the corrected cap")
    print("  is a BRACKET over alpha_a in [0, 1] at delta = 0, the generous end.")
    print("  The counter's own alpha is a traffic slope with no level in it and is")
    print("  scored on 2 BM W / (b dR/dn) instead; see the C3 gate.")
    for line in anchor_cap_lines(cfg, bm, args.block_n, anchors, ridge)[1:]:
        print(line)
    if ridge:
        print(f"  the counter's cap crosses ridge {ridge:.2f} at alpha = BM/ridge - a/W "
              f"= {bm / ridge - a / W:.4f}; 2 BM/(alpha b) alone would say {bm / ridge:.4f}")
    print()
    print("PREDICTIONS, registered here and not adjusted afterwards.")
    print("The three anchors predict the SAME traffic at n=1 and diverge in the slope,")
    print("so n=1 is a validity check and the slope is the claim.")
    print()
    print(f"  {'n':>3}{'rows/exp':>10}"
          + "".join(f"{f'{k} {anchors[k]:.3f}':>16}" for k in order) + f"{'spread':>9}")
    for row in discrimination(cfg, bm, args.tiles, anchors):
        cells = "".join(f"{row['bytes'][k] / 1e9:>15.4f}G" for k in order)
        print(f"  {row['n']:>3}{row['n'] * bm:>10}{cells}{row['spread_frac'] * 100:>8.1f}%")
    print()
    print("  READ THE SPREAD COLUMN AS THE ACCURACY THE EXPERIMENT NEEDS. A counter good")
    print("  to a few percent separates these with an order of magnitude to spare.")
    print("  AND READ IT AS THE REASON THE PLAN NO LONGER RESTS ON IT. On the A100 cell")
    print("  this plan used to register, the three anchors were 0.452 / 0.647 / 0.705 and")
    print("  the anchor contrast was the argument for renting a box. On the H200, the")
    print("  card whose route is actually open, they agree to 0.04 and the spread above")
    print("  is single-digit percent. The contrast below is 87% and is now the claim.")
    print()

    contrast_cells: list[ContrastCell] | None = None
    gbps, gbps_src = measured_bandwidth_gbps(card)
    if gbps is None:
        print(f"CONTRAST REFUSED: {gbps_src}. The per-M-tile cost is quoted in units of "
              "one weight stream at this card's own measured rate and that file is not "
              "in this tree.")
        print()
    else:
        try:
            contrast_cells = contrast_plan(bm, model=args.model)
        except CorpusMissing as exc:
            print(f"CONTRAST REFUSED: {exc}")
            print()
    if contrast_cells is not None:
        stream_ms = weight_stream_ms(W, gbps)
        for line in contrast_lines(contrast_cells, stream_ms, W, a, gbps_src):
            print(line)
        print()

    print("THE COMMAND, on a box where --probe says OPEN.")
    print()
    print("  # 1. the timed ladder, unprofiled, so time and traffic come from the same")
    print("  #    cell. It KEEPS the software L2 flush, because this one is the timed")
    print("  #    apparatus and the flush is part of it; only the profiled runs in step 2")
    print("  #    turn it off, where ncu's --cache-control sets the cache state instead")
    print("  #    and the flush would otherwise put its own 240 MB into the counter.")
    print("  #    Only the primary cell needs this: the contrast cells' timed ladders are")
    print("  #    already in results/published/2026-09-10-nvidia_h200-gaps-session and are")
    print("  #    what the predictions above were fitted from.")
    print(f"  python scripts/block_m_crossing_sweep.py --model {args.model} "
          f"--dtype {args.dtype} \\")
    print(f"      --tiles {bm} --group-m {args.group_m} --block-n {args.block_n} \\")
    print(f"      --num-stages {args.num_stages} \\")
    print(f"      --r-max {max(args.tiles) * bm} --row-step {bm} --step-probes 0")
    print()
    print("  # 2. the counter itself. --run drives ncu, counts the calls out of the")
    print("  #    profile's own launch list and writes the schema below. It is a MODE")
    print("  #    and not a shell loop because of that count: the instrument")
    print("  #    (moe/bench/timing.time_kernel) warms up for a DURATION, --warmup in")
    print("  #    MILLISECONDS, and sizes its own iteration count from that load, so the")
    print("  #    profiled launch count is warmup + iters x trials fused_experts calls of")
    print("  #    several kernels each and every byte field must be divided by it.")
    print("  #    Dividing by an assumed 1 still fits an affine line and still passes the")
    print("  #    residual and monotonicity gates, so nothing downstream would catch it.")
    cells_to_run = contrast_cells or []
    for c in cells_to_run:
        print(f"  python scripts/dram_counter_route.py --run --card {card} \\")
        print(f"      --model {args.model} --dtype {args.dtype} --block-m {c.block_m} "
              f"--block-n {c.block_n} \\")
        print(f"      --group-m {c.group_m} --num-stages {c.num_stages} "
              f"--tiles {','.join(str(t) for t in args.tiles)} \\")
        print(f"      --cache-control {args.cache_control} "
              f"--out counters-{c.name}-cc{args.cache_control}.json   # {c.role}")
    if not cells_to_run:
        print(f"  python scripts/dram_counter_route.py --run --card {card} \\")
        print(f"      --block-m {bm} --block-n {args.block_n} --group-m {args.group_m} \\")
        print(f"      --cache-control {args.cache_control} --out counters.json")
    print()
    print("  # 3. score ONE cell. --run scores its own file as it writes it; --analyse")
    print("  #    rescores one anybody wrote, and refuses a payload with a key missing")
    print("  #    rather than defaulting it to a zero that was never measured.")
    print("  python scripts/dram_counter_route.py --analyse counters-<cell>.json")
    print()
    print("  # 4. score the CONTRAST, which is the reading this plan exists to take and")
    print("  #    which no single payload contains: the discriminator is a RATIO of")
    print("  #    dR/dn across two cells and step 3 scores one cell at a time. Pairs are")
    print("  #    matched inside one cache mode, the measured ratio is compared with the")
    print("  #    registered rivals above, and a pair nobody ran is not scored.")
    if cells_to_run:
        names = " ".join(f"counters-{c.name}-cc{args.cache_control}.json"
                         for c in cells_to_run)
        print(f"  python scripts/dram_counter_route.py --contrast {names}")
    else:
        print("  python scripts/dram_counter_route.py --contrast counters-*.json")
    print()
    print("CACHE CONTROL, and why it is a swept parameter rather than a default.")
    print("  ncu's default is Flush All: all GPU caches are flushed before each replay")
    print("  iteration. That is a DIFFERENT cache state from the timed sweep, which never")
    print("  flushes. The LEVEL it can move is bounded by L2, and L2 is 40-60 MB against a")
    print(f"  {W / 1e9:.2f} GB weight read -- at most {60e6 / W * 100:.1f}% of R(1). The SLOPE,")
    print("  which is the claim, differences that term away entirely. Run both modes")
    print("  anyway: two agreeing numbers close the question, and two disagreeing ones")
    print("  are themselves the result.")
    print()
    n_max = max(args.tiles)
    # The instrument's floor: 1 warmup call at 1 ms of a millisecond-scale
    # kernel, iters floored at 10, one trial. Eleven calls of about five
    # kernels per invocation. A bracket, not a promise: a sub-millisecond
    # kernel warms in more than one call.
    #
    # NO FLUSH LAUNCHES ARE PRICED ANY MORE, and that is a fix and not an
    # omission. `--run` passes `--no-l2-flush`, because the instrument's
    # software flush is a 240 MB read that a profiler counts into the same
    # dram__bytes_read total as the kernel under test (see `sweep_argv`). The
    # cost line priced ten of them per invocation while the recipe it printed
    # would have let their traffic into the answer.
    calls_per_invocation = 1 + 10
    n_cells = len(contrast_cells) if contrast_cells else 1
    invocations = n_cells * len(args.tiles) * 2
    launches = invocations * calls_per_invocation * 5
    # Five minutes per profiled invocation, which is the rate the single-cell
    # plan implied when it budgeted an hour for twelve. It is a rate and not a
    # constant, so it is written down and multiplied rather than restated.
    minutes_each = 5.0
    print("COST, of the plan as extended.")
    print(f"  {n_cells} cells x {len(args.tiles)} tile counts x 2 cache modes = "
          f"{invocations} profiled invocations")
    print(f"  of at least {calls_per_invocation} fused_experts calls each, about "
          f"{launches} profiled kernel launches.")
    print(f"  The largest cell is n={n_max} ({n_max * bm} rows per expert), which the timed")
    print("  sweep measures in single-digit milliseconds; ncu replay and its save/restore")
    print(f"  of the {W / 1e9:.1f} GB weight buffers dominate, at seconds per launch.")
    print(f"  At {minutes_each:.0f} minutes per profiled invocation that is "
          f"{invocations * minutes_each / 60:.1f} GPU-hours for the whole")
    print("  extended plan, against 1.0 for the single cell the plan used to hold, and")
    print("  about half again in pod time. Not the fifteen minutes the one-launch recipe")
    print("  used to promise. At H200 spot rates that is tens of dollars. The cost of")
    print("  this experiment has never been the money.")
    a_only = len(args.tiles) * 2 * 3
    print(f"  DROP TO {a_only} INVOCATIONS ({a_only * minutes_each / 60:.1f} GPU-hours) "
          "by running contrast A alone: the")
    print("  three BLOCK_N cells settle section 2. Contrast B is the schedule question and")
    print("  is worth its own two cells only once A has an answer.")
    print()
    print("WHICH KERNELS TO KEEP, and why the profile is NOT name-filtered.")
    print("  All of W moves inside the two `fused_moe_kernel` launches; the auxiliary")
    print("  launches (align, reduction) move activation-sized traffic only. Filtering to")
    print("  the GEMM would still be defensible -- but then the measured level would no")
    print("  longer be comparable to the byte model, which charges the whole layer. So")
    print("  profile everything, record bytes PER KERNEL NAME, and let the analysis")
    print("  report both totals. A filter applied at capture cannot be undone afterwards.")
    print()
    print("SCHEMA the run must write, so --analyse can score it:")
    print(COUNTER_SCHEMA_TEXT.format(
        device=card, model=args.model, dtype=args.dtype, group_m=args.group_m,
        block_n=args.block_n, block_m=bm, cache=args.cache_control,
        block_k=SWEEP_FIXED["BLOCK_SIZE_K"], num_warps=SWEEP_FIXED["num_warps"],
        num_stages=args.num_stages, marker=args.call_marker,
        ridge=f"{ridge:.2f}" if ridge else "null",
        anchors=json.dumps({k: round(v, 4) for k, v in anchors.items()}),
        bracket=f"[{bracket[0]:.4f}, {bracket[1]:.4f}]"))
    # A REGISTERED PLAN IS THIS ARM'S PRODUCT. The session driver runs this mode
    # as `counter_plan`, and the plan -- the cell, the metric, the predictions
    # under each anchor, the gates that would score them -- is what it is asked
    # for. Nothing is measured and nothing is claimed, so there are no gates to
    # classify; DONE is named rather than spelled 0.
    return exit_codes.DONE


# --------------------------------------------------------------------------
# --self-test: does the estimator recover a planted alpha?
# --------------------------------------------------------------------------

def canned_ncu_csv(*, calls: int, read_per_call: float, write_per_call: float,
                   hit_pct: float, ns_per_call: float, gemms_per_call: int = 2,
                   drop_metric: str | None = None, marker: str = CALL_MARKER,
                   read_unit: str = "byte") -> str:
    """A profile with a KNOWN call count, in ncu's own CSV shape.

    Everything the parser has to survive is in here on purpose: a `==PROF==`
    preamble before the header, a mixture of units on the same metric across
    launches (`read_unit` scales one launch's row so the reducer has to convert
    rather than add), and a launch id column that is the only thing separating
    two launches of one kernel. `drop_metric` removes one metric from one launch,
    which is the planted hole the reducer must refuse rather than read as zero.

    The per-call totals are exact by construction: each call emits one marker
    launch, `gemms_per_call` GEMM launches and one reduction launch, and the
    bytes are split across them so their sum is `read_per_call` exactly.
    """
    head = ["==PROF== Connected to process 1 (python)",
            "==PROF== Profiling \"fused_moe_kernel\": 0%....50%....100% - 1 pass",
            '"ID","Process ID","Process Name","Kernel Name","Metric Name",'
            '"Metric Unit","Metric Value"']
    scale = {"byte": 1.0, "Kbyte": 1e-3, "Mbyte": 1e-6, "Gbyte": 1e-9}[read_unit]
    lines = list(head)
    launch = 0
    per_launch = 1 + gemms_per_call + 1
    for call in range(calls):
        # one marker, `gemms_per_call` GEMMs, one reduction: the shape a
        # fused_experts call actually has.
        shares = [("moe_align_block_size_kernel", 0.0)]
        shares += [(f"{GEMM_MARKER}_up_down", 1.0 / gemms_per_call * 0.98)
                   for _ in range(gemms_per_call)]
        shares += [("moe_sum_kernel", 0.02)]
        shares[0] = (marker, 0.0)
        for kernel, share in shares:
            launch += 1
            unit = read_unit if launch == 2 else "byte"
            factor = scale if launch == 2 else 1.0
            values = {
                "dram__bytes_read.sum": (unit, read_per_call * share * factor),
                "dram__bytes_write.sum": ("byte", write_per_call * share),
                "lts__t_sector_op_read_hit_rate.pct": ("%", hit_pct),
                "gpu__time_duration.sum": ("nsecond", ns_per_call / per_launch),
            }
            for metric, (metric_unit, value) in values.items():
                if drop_metric == metric and launch == 2:
                    continue
                lines.append(f'"{launch}","1","python","{kernel}","{metric}",'
                             f'"{metric_unit}","{value!r}"')
        del call
    return "\n".join(lines) + "\n"


def synthetic_counter_payload(cell: ContrastCell, dr_dn: float, *,
                              cache_control: str = "all",
                              model: str = "mixtral-8x7b",
                              device: str = "nvidia_h200",
                              tiles=DEFAULT_TILES,
                              stamp: bool = True,
                              stamped_cell: str | None = None) -> dict:
    """A counter payload for `cell` whose traffic slope is EXACTLY `dr_dn`.

    Used by `--self-test` and by the tests to exercise the contrast scorer with
    a known answer. The rows are the byte model's own, so the payload passes
    V2 (R(1) is one weight read) by construction and the only thing under test
    is the ratio the scorer forms from the two slopes.
    """
    cfg = MODEL_CONFIGS[model]
    W = weight_bytes_total(cfg)
    a = activation_bytes_per_tile(cfg, cell.block_m)
    alpha = (dr_dn - a) / W
    rows = [{"n": n, "calls": 11, "calls_floor": 10, "launches": 44,
             "dram_bytes_read": predicted_read_bytes(cfg, cell.block_m, n, alpha),
             "dram_bytes_write": 1.1e8, "l2_read_hit_pct": 4.2,
             "gpu_time_ns": 1.95e6, "by_kernel": {}}
            for n in tiles]
    payload = {"device": device, "model": model, "dtype": "bf16",
               "group_m": cell.group_m, "block_n": cell.block_n,
               "block_k": SWEEP_FIXED["BLOCK_SIZE_K"], "num_warps": cell.num_warps,
               "num_stages": cell.num_stages, "block_m": cell.block_m,
               "cache_control": cache_control, "rows": rows}
    if stamp:
        payload["contrast"] = {"this_cell": stamped_cell or cell.name,
                               "cells": [asdict(cell)]}
    return payload


def self_test_contrast() -> list[tuple[str, bool, str]]:
    """`(label, passed, detail)` for the contrast scorer, on planted worlds.

    THE SCORER IS THE HALF THE POD CANNOT REDO. `--run` writes five payloads and
    the answer to section 2 is a RATIO across two of them, so the arithmetic
    that turns five files into an answer has to be exercised before the box is
    rented, exactly like the parser below it. Three planted worlds:

      1. TRAFFIC: the two cells read the corpus's own predicted slopes, so the
         ratio is the registered TRAFFIC value and must be read as TRAFFIC.
      2. TIME: the two cells read the SAME slope, ratio 1.000, which must be
         read as TIME and not as a failed TRAFFIC prediction.
      3. a payload whose stamped cell name disagrees with its own knobs, which
         must REFUSE: the stamp is read back rather than trusted or ignored.
    """
    import tempfile
    out: list[tuple[str, bool, str]] = []
    try:
        cells = contrast_plan(64)
    except CorpusMissing as exc:
        return [("the contrast scorer on planted worlds", False,
                 f"could not be exercised: {exc}")]
    pair = contrast_pairs(cells)[0]
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    W = weight_bytes_total(cfg)
    gbps, _src = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(W, gbps) if gbps else None
    if stream is None:
        return [("the contrast scorer on planted worlds", False,
                 "no calibration for nvidia_h200 in this tree")]

    lo_dr = pair.lo.traffic_bytes_per_tile(stream, W)
    hi_dr = pair.hi.traffic_bytes_per_tile(stream, W)
    for label, lo_v, hi_v, want in (
            ("planted TRAFFIC world", lo_dr, hi_dr, "TRAFFIC"),
            ("planted TIME world", hi_dr, hi_dr, "TIME")):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for cell, value in ((pair.lo, lo_v), (pair.hi, hi_v)):
                p = Path(tmp) / f"{cell.name}.json"
                p.write_text(json.dumps(synthetic_counter_payload(cell, value)))
                paths.append(p)
            rows, _cells = contrast_rows(paths)
            by = {r["cell"].name: r for r in rows}
            ratio = by[pair.lo.name]["dr_dn"] / by[pair.hi.name]["dr_dn"]
            verdict, which = pair.read(ratio)
            good = verdict == PASS and which == want
            out.append((label, good,
                        f"ratio {ratio:.4f} read as {which} against TRAFFIC "
                        f"{pair.traffic_ratio:.4f} and TIME {TIME_RATIO:.3f}"))

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mislabelled.json"
        p.write_text(json.dumps(synthetic_counter_payload(
            pair.lo, lo_dr, stamped_cell=pair.hi.name)))
        try:
            contrast_rows([p, p])
            out.append(("planted payload whose stamp names another cell", False,
                        "the scorer took a ratio under a name the knobs contradict"))
        except CounterRunRefused as exc:
            out.append(("planted payload whose stamp names another cell", True,
                        f"REFUSED as required: {str(exc)[:90]}"))
    return out


def self_test_parser() -> list[tuple[str, bool, str]]:
    """`(label, passed, detail)` for each planted profile.

    THREE PLANTED WORLDS, and two of them must REFUSE.
      1. a good profile with a known call count: the reducer must divide by that
         count and return the planted per-call bytes exactly, including the
         launch whose unit ncu rescaled.
      2. a metric missing from one launch: the reducer must refuse. Reading it
         as 0.0 is the failure this whole file exists to avoid, and it would not
         show up anywhere downstream: the total would be short by one launch's
         bytes, still monotone, still affine, still inside every gate.
      3. a call count at or below the cell's own iters x trials: the reducer must
         refuse. The instrument runs warmup calls on top of the timed ones, so a
         count that does not exceed the floor means the marker is not
         one-per-call and every byte field is wrong by the ratio.
    """
    out: list[tuple[str, bool, str]] = []
    planted = {"calls": 11, "read_per_call": 2.84e9, "write_per_call": 1.1e8,
               "hit_pct": 4.2, "ns_per_call": 1.95e6}
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**planted, read_unit="Mbyte")),
                             calls_floor=10)
    good = (row["calls"] == planted["calls"]
            and abs(row["dram_bytes_read"] - planted["read_per_call"]) < 1.0
            and abs(row["dram_bytes_write"] - planted["write_per_call"]) < 1.0
            and abs(row["gpu_time_ns"] - planted["ns_per_call"]) < 1.0
            and abs(row["l2_read_hit_pct"] - planted["hit_pct"]) < 1e-9
            and row["launches"] == planted["calls"] * 4
            and row["gemm_launches_per_call"] == 2.0)
    out.append((f"planted {planted['calls']} calls, one launch rescaled to Mbyte", good,
                f"counted {row['calls']} calls over {row['launches']} launches; "
                f"read {row['dram_bytes_read'] / 1e9:.6f} GB per call against a planted "
                f"{planted['read_per_call'] / 1e9:.6f}"))
    # And the division is the thing being tested, so check it moves: the same
    # per-launch bytes over twice the calls must halve the per-call figure.
    twice = normalise_per_call(
        parse_ncu_csv(canned_ncu_csv(**{**planted, "calls": 22})), calls_floor=10)
    halves = abs(twice["dram_bytes_read"] - planted["read_per_call"]) < 1.0
    out.append(("22 calls of the same per-call traffic still reads per-call", halves,
                f"{twice['dram_bytes_read'] / 1e9:.6f} GB per call over "
                f"{twice['calls']} calls"))

    try:
        normalise_per_call(
            parse_ncu_csv(canned_ncu_csv(**planted,
                                         drop_metric="dram__bytes_write.sum")),
            calls_floor=10)
        out.append(("planted MISSING metric on one launch", False,
                    "the reducer returned a number for a profile with a hole in it"))
    except CounterRunRefused as exc:
        out.append(("planted MISSING metric on one launch", True,
                    f"REFUSED as required: {str(exc)[:90]}"))

    try:
        normalise_per_call(parse_ncu_csv(canned_ncu_csv(**{**planted, "calls": 8})),
                           calls_floor=10)
        out.append(("planted WRONG call count, 8 against a floor of 10", False,
                    "the reducer divided by a count below the cell's own iters x trials"))
    except CounterRunRefused as exc:
        out.append(("planted WRONG call count, 8 against a floor of 10", True,
                    f"REFUSED as required: {str(exc)[:90]}"))

    try:
        normalise_per_call(parse_ncu_csv(canned_ncu_csv(**planted, marker="not_the_marker")),
                           calls_floor=10)
        out.append(("planted profile with no marker kernel at all", False,
                    "the reducer counted calls it could not see"))
    except CounterRunRefused as exc:
        out.append(("planted profile with no marker kernel at all", True,
                    f"REFUSED as required: {str(exc)[:90]}"))
    return out


def self_test_probe() -> list[tuple[str, bool, str]]:
    """The probe worlds, planted, scored end to end off any GPU.

    WHY THIS SECTION EXISTS. `do_self_test`'s own rule is that a stage first
    exercised on a rented box is an assertion and not a check, and until
    2026-09-15 the probe's verdict logic was in exactly that position: the
    pytest suite planted `_run`, the pod never runs pytest, and `arm_verify
    counter_plan` named only `--dry-run` and `--bracket`, neither of which
    touches it. What the logic decides is whether two 120-minute arms are
    bookable, so it is the most expensive thing in this file to get wrong.

    Each world is planted as the BYTES ncu and the child would produce and is
    pushed through the same `probe_reading` -> `route_verdict` ->
    `probe_gates` -> `probe_exit` chain `--probe` runs. The VERDICT AND THE
    EXIT CODE are both scored, because the ledger reads the code and an OPEN
    that exited 2 would retire the arms as surely as a BLOCKED.
    """
    header = ('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n')
    launched = f"{PK.MARKER} {PK.LAUNCHED} NVIDIA H200: one add_ over 1048576 fp32"
    nsys_ok = {"present": True, "importer_present": True}
    caps_none = {"available": True, "cap_eff": "0xa80425fb",
                 "cap_eff_field": "00000000a80425fb", "cap_eff_bits": 64,
                 "sys_admin": False, "perfmon": False}
    worlds = (
        ("a counter that came back with a NUMBER", "OPEN", exit_codes.DONE,
         dict(returncode=0, stdout=launched, stderr="",
              log_text=header + '"0","probe","dram__bytes_read.sum","byte","4194304"\n')),
        ("a counter that honestly read ZERO", "OPEN", exit_codes.DONE,
         dict(returncode=0, stdout=launched, stderr="",
              log_text=header + '"0","probe","dram__bytes_read.sum","byte","0"\n')),
        ("ERR_NVGPUCTRPERM on a launched kernel", "BLOCKED", exit_codes.CLAIM_FAIL,
         dict(returncode=1, stdout=launched,
              stderr="==ERROR== ERR_NVGPUCTRPERM - The user does not have permission "
                     "to access NVIDIA GPU Performance Counters on the target device 0.",
              log_text="")),
        ("the /bin/true shape: ncu profiled nothing", REFUSE, exit_codes.REFUSED,
         dict(returncode=0, stdout="", stderr="",
              log_text="==WARNING== No kernels were profiled.\n")),
        ("no CUDA device for the probe child", REFUSE, exit_codes.REFUSED,
         dict(returncode=1,
              stdout=f"{PK.MARKER} {PK.NO_CUDA_DEVICE} torch.cuda.is_available() is False",
              stderr="", log_text="")),
    )
    out = []
    for label, want_verdict, want_exit, planted in worlds:
        ncu = probe_reading("/planted/ncu", "Version 2025.1.1.0", **planted)
        verdict, _notes = route_verdict(caps_none, {"available": False, "why": "planted"},
                                        ncu, nsys_ok)
        code = probe_exit(verdict, probe_gates(verdict, ncu))
        good = verdict == want_verdict and code == want_exit
        out.append((label, good,
                    f"verdict {verdict} exit {code}, wanted {want_verdict} "
                    f"exit {want_exit}; counters_read={ncu['counters_read']}"))
    return out


def do_self_test(args) -> int:
    """Plant an alpha, synthesise the counter rows the model implies, and check
    the estimator returns it. Then plant a ladder and check the bracket contains
    the alpha that generated it. Then plant two anchor sets, one separating and
    one clustered, and check C1 registers the question each cell can carry.
    Then plant a TRAFFIC world and a TIME world and check the contrast scorer
    names the right rival. Then plant a PROFILE with a known call count and
    check `--run`'s parser and its per-call division return it. Then plant the
    PROBE worlds and check the verdict and the exit code each produces.

    This is the check that the analysis half is not itself the source of a
    number. `block_m_crossing_sweep.py` has the same shape for the same reason:
    an estimator that has never been run against a known answer is an assertion.
    The parser half was added on 2026-09-10 with `--run`: a runner whose parsing
    has never been exercised is the same assertion one layer down, and it would
    have been exercised for the first time on a rented box. The probe half was
    added on 2026-09-15 for the same reason and a larger bill: it is the gate
    that decides whether 240 booked minutes are spent.
    """
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    bm, ok = 32, True
    for planted in (0.10, 0.4522, 0.558, 0.705, 1.0):
        rows = [{"n": n, "launches": 5,
                 "dram_bytes_read": predicted_read_bytes(cfg, bm, n, planted)}
                for n in DEFAULT_TILES]
        got, _, resid = alpha_from_counters(rows, cfg, bm)
        good = abs(got - planted) < 1e-9 and resid < 1e-9
        ok &= good
        print(f"  planted alpha {planted:.4f} -> recovered {got:.6f}  "
              f"residual {resid:.2e}  {'PASS' if good else 'FAIL'}")

    # And the bracket: a ladder generated at a known alpha and a known bandwidth
    # must be bracketed by bounds that contain the UNCORRECTED alpha that ladder
    # would report -- `(W alpha + a) / (W + a)`, since the activation share is
    # inside the fitted slope. Comparing against the planted WEIGHT re-read
    # fraction instead is the bug this line exists to have already found: it is
    # off by the activation term and would make the bracket look broken.
    W = weight_bytes_total(cfg)
    a1 = activation_bytes_per_tile(cfg, bm)
    peak, achieved = 2039.0, 1450.0
    for planted in (0.30, 0.558, 0.90):
        t = [1e3 * predicted_read_bytes(cfg, bm, n, planted) / (achieved * 1e9)
             for n in (1, 2, 3, 4)]
        _, slope = ols([1, 2, 3, 4], t)
        lo, hi = physical_bracket(slope, t[0], W, a1, peak)
        uncorrected = (W * planted + a1) / (W + a1)
        good = lo - 1e-12 <= uncorrected <= hi + 1e-12
        ok &= good
        print(f"  planted alpha {planted:.4f} (uncorrected {uncorrected:.4f}) at "
              f"{achieved:.0f} GB/s -> bracket [{lo:.4f}, {hi:.4f}]  "
              f"{'PASS' if good else 'FAIL'}")

    # C1's REGISTRATION, planted both ways. The gate that was pre-determined to
    # FAIL on the cell this plan registers was not caught by any test of the
    # estimator: the arithmetic was right and the question was unanswerable. So
    # the self test plants an anchor set of each shape and checks that a
    # measurement landing ON an anchor passes in both.
    print("\n  C1's REGISTRATION, on planted anchor sets.")
    for label, anchors, alpha, want_sep in (
            ("separating anchors (the A100 cell's 0.452 / 0.647 / 0.705)",
             {"t1": 0.4522, "published": 0.6473, "n3": 0.7047}, 0.4522, True),
            ("clustered anchors (this plan's H200 cell, 0.0393 end to end)",
             {"t1": 0.6202, "n3": 0.6583, "published": 0.6595}, 0.6583, False)):
        reg = c1_registration(anchors)
        survivors = [k for k, v in anchors.items() if abs(v - alpha) <= reg.tolerance]
        good = reg.separating == want_sep and reg.verdict(survivors) == PASS
        ok &= good
        print(f"  {label:<58} {'PASS' if good else 'FAIL'}  "
              f"{'SEPARATING' if reg.separating else 'NOT DISCRIMINATING'}, closest pair "
              f"{reg.min_gap:.4f}, a measurement at {alpha:.4f} leaves "
              f"{len(survivors)} and scores {reg.verdict(survivors)}")

    print("\n  THE CONTRAST SCORER, on planted TRAFFIC and TIME worlds.")
    contrast_ok = True
    for label, passed, detail in self_test_contrast():
        contrast_ok &= passed
        print(f"  {label:<58} {'PASS' if passed else 'FAIL'}  {detail}")
    ok &= contrast_ok

    print("\n  THE RUNNER'S PARSER AND ITS PER-CALL DIVISION, on canned ncu output.")
    parser_ok = True
    for label, passed, detail in self_test_parser():
        parser_ok &= passed
        print(f"  {label:<58} {'PASS' if passed else 'FAIL'}  {detail}")
    ok &= parser_ok

    print("\n  THE PROBE'S VERDICT AND EXIT CODE, on planted ncu and child output.")
    probe_ok = True
    for label, passed, detail in self_test_probe():
        probe_ok &= passed
        print(f"  {label:<58} {'PASS' if passed else 'FAIL'}  {detail}")
    ok &= probe_ok

    print("\n  THE R3-ARMS FAMILY, on planted pages through --run's own reduction.")
    r3_ok = True
    for label, passed, detail in self_test_r3():
        r3_ok &= passed
        print(f"  {label:<58} {'PASS' if passed else 'FAIL'}  {detail}")
    ok &= r3_ok
    print(f"\n  SELF TEST {'PASS' if ok else 'FAIL'}")
    # A SELF TEST IS A VALIDITY GATE ON THE ANALYSIS HALF, so its failure is
    # INVALID (3) and not CLAIM_FAIL (1): an estimator that cannot recover a
    # planted alpha has not refuted anything about the world, it has said that
    # nothing this file computes may be quoted. It returned 1, which the driver
    # files as a finished result and never retries. The gate prints its RESULT
    # line like every other gate in this file, so the log and the code agree.
    gate = Gate("S1", "VALIDITY",
                "the estimator, the bracket, C1's registration, the contrast scorer, "
                "the runner's parser, the probe's verdict and the r3-arms family's "
                "group model, attribution and gates recover planted alphas, ratios, "
                "call counts, refusals, routes and worlds",
                PASS if ok else FAIL, "every planted row above",
                "all rows PASS", "everything this file computes, everything --run "
                "would write, everything --contrast would read and the gate that "
                "decides whether the counter arms are bookable at all")
    print(gate.result_line())
    return exit_codes.classify([gate.scored()])


# --------------------------------------------------------------------------
# --contrast: the ratio ACROSS two counter runs, which is the reading the
# extended plan exists to take and which no single payload contains.
# --------------------------------------------------------------------------

def measured_dr_dn(payload: dict) -> tuple[float, float]:
    """`(dR/dn bytes per M-tile, max relative residual)` from one counter run.

    The same OLS the alpha estimator uses, stopped one step earlier: the
    contrast is a ratio of two SLOPES and never converts either into an alpha,
    so no byte model, no weight total and no card rate enters it.
    """
    rows = [r for r in payload.get("rows", [])
            if int(r.get("launches", 0)) > 0 and float(r.get("dram_bytes_read", 0)) > 0]
    ns = [int(r["n"]) for r in rows]
    ys = [float(r["dram_bytes_read"]) for r in rows]
    if len(set(ns)) < 3:
        raise CounterRunRefused(
            f"{len(set(ns))} distinct tile counts with a launch and non-zero bytes; "
            "a slope needs three and a ratio of slopes needs two of them")
    r0, dr = ols(ns, ys)
    resid = max(abs((r0 + dr * n) - y) / y for n, y in zip(ns, ys, strict=True))
    return dr, resid


def payload_contrast_cell(payload: dict, cells: list[ContrastCell]) -> ContrastCell:
    """Which registered cell a payload profiled, and a refusal if it is not one.

    Matched on the four knobs that make a cell (BLOCK_M, BLOCK_N, GROUP_SIZE_M,
    num_stages), then CROSS-CHECKED against the name `--run` stamped into the
    file. Until 2026-09-10 `build_counter_payload` stamped `contrast` into every
    payload and nothing ever read it back, so a run whose knobs and whose stamp
    disagreed would have been scored under whichever the reader trusted.
    """
    need = ("block_m", "block_n", "group_m", "num_stages")
    absent = [k for k in need if payload.get(k) is None]
    if absent:
        raise CounterRunRefused(
            f"this payload names no {absent}; a contrast is a ratio between two "
            "identified cells and an unidentified one cannot enter it")
    key = tuple(int(payload[k]) for k in need)
    mine = next((c for c in cells
                 if (c.block_m, c.block_n, c.group_m, c.num_stages) == key), None)
    if mine is None:
        raise CounterRunRefused(
            f"BLOCK_M/BLOCK_N/GROUP_SIZE_M/num_stages {key} is not a cell the plan "
            f"registers; the registered cells are "
            f"{[(c.name, c.block_m, c.block_n, c.group_m, c.num_stages) for c in cells]}")
    stamped_name = (payload.get("contrast") or {}).get("this_cell")
    if stamped_name and stamped_name != mine.name:
        raise CounterRunRefused(
            f"this payload's knobs are cell {mine.name} and its stamped contrast says "
            f"{stamped_name}; one of the two is wrong and the ratio may not be taken "
            "under either name")
    return mine


def contrast_rows(paths: list[Path]) -> tuple[list[dict], list[ContrastCell]]:
    """Reduce each counter payload to one contrast row, refusing what cannot enter.

    Each row carries the cell the payload profiled, its cache mode, its measured
    slope and the verdicts its OWN gates returned: a ratio over a payload whose
    validity gates failed is a ratio over numbers nobody may quote.
    """
    loaded = [(p, json.loads(p.read_text())) for p in paths]
    block_ms = {int(d.get("block_m", 0)) for _p, d in loaded} - {0}
    cells: list[ContrastCell] = []
    for bm in sorted(block_ms):
        for c in contrast_plan(bm):
            if c.name not in {x.name for x in cells}:
                cells.append(c)
    rows = []
    for path, payload in loaded:
        cell = payload_contrast_cell(payload, cells)
        dr, resid = measured_dr_dn(payload)
        gates, _summary = score_counter_run(payload)
        bad = [g.number for g in gates if g.kind == "VALIDITY" and g.verdict != PASS]
        stamped = payload.get("contrast") or {}
        drift = None
        for stamped_cell in stamped.get("cells", []):
            if stamped_cell.get("name") == cell.name:
                was = float(stamped_cell["slope_ms"])
                if abs(was - cell.slope_ms) > 1e-9 * max(1.0, abs(cell.slope_ms)):
                    drift = (was, cell.slope_ms)
        rows.append({"path": str(path), "cell": cell, "cache": payload.get("cache_control"),
                     "device": payload.get("device"), "model": payload.get("model"),
                     "dtype": payload.get("dtype"), "dr_dn": dr, "residual": resid,
                     "invalid_gates": bad, "stamped": bool(stamped), "drift": drift})
    return rows, cells


def do_contrast(args) -> int:
    """Score the RATIO the plan registered, across the payloads --run wrote.

    WHY THIS IS A MODE. Until 2026-09-10 `--analyse` read ONE payload and scored
    V1-V4 and C1-C3 on that cell alone, while the reading the extended plan
    exists to take is a RATIO of dR/dn across two cells (1.871 under TRAFFIC,
    1.000 under TIME). Nothing read two payloads and nothing compared a measured
    ratio with the registered rivals, so after five runs on a metered box the
    operator had five scored cells and section 2's arithmetic to do by hand,
    off the predictions, which is the improvisation the plan exists to prevent.

    A pair nobody ran is NOT scored. A gate for a contrast that has no payloads
    would be UNKNOWN, which counts against the run, so a plan run half through
    would report CLAIM_FAIL for the half it never took: the same shape as the C1
    defect this file fixed the same day. Zero complete pairs REFUSES instead,
    before any gate, and names what is missing.
    """
    paths = [Path(p) for p in args.contrast]
    if len(paths) < 2:
        print("REFUSE: --contrast needs at least two counter payloads. A contrast is "
              "a ratio between two cells; one file is one cell and --analyse scores it.")
        return exit_codes.REFUSED
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"REFUSE: no such payload: {missing}")
        return exit_codes.REFUSED

    rows, cells = contrast_rows(paths)
    pairs = contrast_pairs(cells)
    gbps, gbps_src = measured_bandwidth_gbps(args.card)

    print(f"COUNTER CONTRAST  {len(rows)} payloads, tolerance "
          f"+/-{CONTRAST_TOLERANCE * 100:.0f}% of each rival")
    print()
    print(f"  {'cell':<16}{'cache':>6}{'BM':>4}{'BN':>5}{'G':>4}{'s':>3}"
          f"{'measured dR/dn':>17}{'resid':>8}  file")
    for r in rows:
        c = r["cell"]
        print(f"  {c.name:<16}{str(r['cache']):>6}{c.block_m:>4}{c.block_n:>5}"
              f"{c.group_m:>4}{c.num_stages:>3}"
              f"{r['dr_dn'] / 1e9:>16.4f}G{r['residual'] * 100:>7.2f}%  "
              f"{Path(r['path']).name}")
    print()

    by_key = {}
    collisions = []
    for r in rows:
        key = (r["cell"].name, r["cache"])
        if key in by_key:
            collisions.append(key)
        by_key[key] = r

    # X0, the validity of the whole comparison: every payload sound on its own
    # terms, identified, and predicted from the corpus the tree holds NOW.
    lines = []
    for r in rows:
        note = "sound" if not r["invalid_gates"] else \
            f"VALIDITY {r['invalid_gates']} did not pass"
        stamp = "stamped prediction read back and matched" if r["stamped"] else \
            "no stamped prediction; identified by its knobs alone"
        if r["drift"]:
            stamp = (f"STAMPED SLOPE {r['drift'][0]:.5f} ms/tile against the corpus's "
                     f"{r['drift'][1]:.5f} today: the ladders moved under the run")
        lines.append(f"{r['cell'].name:<16} {note}; {stamp}")
    for key in collisions:
        lines.append(f"TWO payloads for cell {key[0]} at cache-control {key[1]}; the "
                     "second overwrote the first and a ratio would be taken over one "
                     "of two measurements chosen by argument order")
    # A ratio across two cards, two models or two dtypes is not a contrast: the
    # cells differ in more than the one knob the pair varies, and the ratio
    # would carry that difference silently. Checked here rather than per pair,
    # because a payload from another card has no business in this comparison at
    # all.
    apparatus = {k: sorted({str(r[k]) for r in rows}) for k in ("device", "model", "dtype")}
    mixed = {k: v for k, v in apparatus.items() if len(v) > 1}
    for k, v in mixed.items():
        lines.append(f"the payloads carry {len(v)} values of {k} ({v}); a ratio across "
                     "them varies more than the pair's own knob")
    unsound = [r for r in rows if r["invalid_gates"] or r["drift"]]
    gates = [Gate("X0", "VALIDITY",
                  "every payload in the comparison is sound, identified, from one "
                  "apparatus, and predicted from the corpus this tree holds now",
                  PASS if not unsound and not collisions and not mixed else FAIL,
                  f"{len(rows)} payloads, {len(unsound)} unsound or drifted, "
                  f"{len(collisions)} collisions, {len(mixed)} mixed apparatus fields",
                  "all sound, none drifted, one device / model / dtype, no two payloads "
                  "for one cell and cache mode",
                  "every ratio below: a ratio over numbers whose own validity gates "
                  "failed, or over two apparatuses, is a ratio nobody may quote", lines)]

    scored_pairs = 0
    for p in pairs:
        for cache in sorted({r["cache"] for r in rows}, key=str):
            lo = by_key.get((p.lo.name, cache))
            hi = by_key.get((p.hi.name, cache))
            if lo is None or hi is None:
                continue
            scored_pairs += 1
            ratio = lo["dr_dn"] / hi["dr_dn"]
            if p.separates():
                verdict, which = p.read(ratio)
                claim = (f"contrast {p.label} reads as exactly one of TRAFFIC and TIME")
            else:
                near = (abs(ratio - p.traffic_ratio) <= CONTRAST_TOLERANCE * p.traffic_ratio
                        or abs(ratio - TIME_RATIO) <= CONTRAST_TOLERANCE)
                verdict, which = (PASS if near else FAIL), "the two rivals, which this "\
                    "pair's own predictions do not separate"
                claim = (f"contrast {p.label} agrees with its rivals, which this pair "
                         "cannot tell apart")
            detail = [
                f"TRAFFIC predicted {p.traffic_ratio:.3f}, TIME predicted "
                f"{TIME_RATIO:.3f}, measured {ratio:.3f}",
                f"{p.lo.name} {lo['dr_dn'] / 1e9:.4f} GB per M-tile against "
                f"{p.hi.name} {hi['dr_dn'] / 1e9:.4f} GB, both at cache-control {cache}",
                "the ratio carries NO bandwidth constant, so a recalibration moves "
                "neither prediction; the absolute columns of --dry-run do carry one "
                f"({gbps_src if gbps else 'no calibration in this tree'})",
                p.decides,
            ]
            gates.append(Gate(f"X{p.label}-{cache}", "CLAIM", claim, verdict,
                              f"ratio {ratio:.3f}; reads as {which}",
                              f"within {CONTRAST_TOLERANCE * 100:.0f}% of exactly one "
                              "rival",
                              f"section 2 of the 2026-09-10 analysis for contrast "
                              f"{p.label}: whether the missing 1/BLOCK_N term is traffic "
                              "or time stays open", detail))

    if scored_pairs == 0:
        have = sorted(f"{name}@{cache}" for name, cache in by_key)
        want = [f"{p.label}: {p.lo.name} and {p.hi.name} at one cache mode" for p in pairs]
        print("REFUSE: no registered pair is complete in these payloads, so there is no "
              "ratio to score.")
        print(f"  have {have}")
        for w in want:
            print(f"  want {w}")
        print("  A gate for a contrast nobody ran would report UNKNOWN, which counts "
              "against the run; a plan run half through is not a refuted claim.")
        return exit_codes.REFUSED

    for g in gates:
        for line in g.render():
            print(line)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(stamped(
            {"rows": [dict(r, cell=asdict(r["cell"])) for r in rows],
             "pairs": [{"label": p.label, "lo": p.lo.name, "hi": p.hi.name,
                        "traffic_ratio": p.traffic_ratio, "time_ratio": TIME_RATIO,
                        "separates": p.separates()} for p in pairs],
             "tolerance": CONTRAST_TOLERANCE,
             "gates": [asdict(g) for g in gates]},
            mode="contrast", args=args, card=NO_CARD,
            instrument=CONTRAST_INSTRUMENT), indent=2))
        print(f"\nwrote {out}")
        print(f"git   {git_visibility(out)}")
    return exit_codes.classify(g.scored() for g in gates)


def do_analyse(args) -> int:
    path = args.analyse[0] if isinstance(args.analyse, list) else args.analyse
    payload = json.loads(Path(path).read_text())
    gates, summary = score_counter_run(payload)
    print(f"COUNTER RUN  {payload.get('device')}  {payload.get('model')}  "
          f"BLOCK_M={payload.get('block_m')}  cache-control={payload.get('cache_control')}")
    print()
    for g in gates:
        for line in g.render():
            print(line)
    print()
    if summary.get("alpha") is not None:
        print(f"  alpha measured directly from DRAM traffic: {summary['alpha']:.4f}")
    # THE TWO CODES WERE THE WRONG WAY ROUND: this returned 1 for a VALIDITY
    # failure and 3 for a CLAIM failure, which is the shared table inverted. 1
    # is CLAIM_FAIL, a result the driver files as finished, so an unsound
    # counter run was recorded as a refuted claim; 3 is INVALID, which would
    # have latched a perfectly good refutation.
    return exit_codes.classify(g.scored() for g in gates)


# ==========================================================================
# THE R3 ARMS UNDER A DRAM COUNTER (`--family r3-arms`).
#
# alpha(G) is the fraction of an expert's weight set each extra M-tile
# re-reads from DRAM, G being Triton's GROUP_SIZE_M swizzle. R3
# (`scripts/private_weight_reference.py`) builds three arms over one tread
# ladder: NATIVE (vLLM as shipped), SHARED (72 declared slots over one copy,
# reuse possible) and PRIVATE (one copy per M-tile, no reuse possible). Its
# timing could not identify alpha at G >= 4 on the H200, because an on-chip
# floor hides the traffic, and brackets alpha(1) in [0.915, 1.0]. A DRAM
# counter reads the traffic itself. This family profiles R3's OWN calls under
# ncu, one G per invocation, and scores the bytes against a card-free model of
# the kernel's schedule (`group_reads`).
#
# WHY IT LIVES HERE AND NOT IN A NEW SCRIPT. This file already owns the
# permission probe, the CSV parser and its unit tables, Gate and the exit
# codes, provenance stamping and git visibility. A parallel script would fork
# the parser and the probe, which is the "one rule at two call sites" defect
# this file keeps documenting. R3 is imported LAZILY (`_r3`), inside family
# functions, so the ladder family's import graph does not change.
#
# WHAT IT DOES NOT DO. It uses no ridge, no bandwidth and no calibration, so
# `--card` is not read and `measured_ridge` does not gate it. Its bytes are
# the ATTACHED card's: every page's first line names that card
# (`card_line`), and none of them is the study's H200.
# ==========================================================================

#: The metrics the family asks, in three classes, and the reason for each.
#:
#: STRICT. The run refuses without them, and V2 fails a page missing one on
#: any launch. `dram__bytes_read.sum` is the traffic; `dram__bytes_write.sum`
#: says whether a write share is hiding in the read model; the L2 read sectors
#: the SMs request (`lts__t_sectors_srcunit_tex_op_read.sum`) are what V6 holds
#: equal across the three arms, which issue identical loads; `launch__grid_size`
#: attributes every launch to its cell (`attribute_launches`); the duration is
#: replay time at the base clock and is never compared with a ladder.
R3_STRICT_METRICS: tuple[str, ...] = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "launch__grid_size",
    "gpu__time_duration.sum",
)
#: CROSS-CHECK. Asked when this chip offers them, and a gate reads one only
#: when the probe PROVED it readable (`metrics_proven`); the dry run says so.
#: V8 holds DRAM bytes against the L2 sectors filled from DRAM.
R3_CROSSCHECK_METRICS: tuple[str, ...] = (
    "lts__d_sectors_fill_device.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sector_op_read_hit_rate.pct",
)
#: RECORDED, never gated. The four occupancy limits set the co-residency
#: window C1 reads at G=1 (`coresident_window`); if they were not proven, that
#: claim is not asked. An unreadable cell of one of these never refuses a
#: page: they are parsed SOFT.
R3_RECORDED_METRICS: tuple[str, ...] = (
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "lts__t_sectors_srcunit_ltcfabric.sum",
)
R3_OCCUPANCY_LIMITS: tuple[str, ...] = R3_RECORDED_METRICS[:4]
R3_ALL_METRICS: tuple[str, ...] = (R3_STRICT_METRICS + R3_CROSSCHECK_METRICS
                                   + R3_RECORDED_METRICS)

#: The page field each summed or averaged metric lands in. `launch__grid_size`
#: lands in `grid_size`, per GEMM, and the recorded ones in `recorded`.
R3_FIELDS: dict[str, str] = {
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "lts__t_sectors_srcunit_tex_op_read.sum": "l2_tex_read_sectors",
    "gpu__time_duration.sum": "gpu_time_ns",
    "lts__d_sectors_fill_device.sum": "l2_fill_device_sectors",
    "lts__t_sectors_op_read_lookup_miss.sum": "l2_read_miss_sectors",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "l2_tex_read_hit_sectors",
    "lts__t_sector_op_read_hit_rate.pct": "l2_read_hit_pct",
}
#: The per-GEMM fields V2 requires as numbers on every cell.
R3_STRICT_FIELDS: tuple[str, ...] = ("dram_bytes_read", "dram_bytes_write",
                                     "l2_tex_read_sectors", "gpu_time_ns",
                                     "grid_size")
#: A rate, which is weighted across the two GEMMs rather than summed.
R3_RATE_FIELDS: tuple[str, ...] = ("l2_read_hit_pct",)

#: Bytes per L2 sector. V8 converts sector counts to bytes with it.
L2_SECTOR_BYTES = 32

#: THE REGISTERED G, IN THE ORDER THEY RUN. 64 first because it has the
#: sharpest prediction (one read per weight byte) and carries the private
#: arm's +15.8% timed cost at fixed bytes; 1 is the traffic-bound regime; 4 is
#: the first G whose timing is floored; 2 is the regime locator. 16 is
#: optional (`R3_OPTIONAL_GROUP`): it bridges session 5's timed set and tests
#: the group model's 1.3 / 1.075 / 1.0 at G = 4 / 16 / 64.
R3_GROUPS: tuple[int, ...] = (64, 1, 4, 2)
R3_OPTIONAL_GROUP = 16

#: THE TREADS. n=1 is the identity tread (SHARED and PRIVATE are one call);
#: n=6 is R3's deepest; n=3 is required (four points for an OLS residual, the
#: first tread where G=2 pays a second group, G=4's mid-step); n=4 is where
#: tiles align with groups at G=4 and G=16 and the group model predicts a DROP
#: from n=3, which no per-tile scalar alpha can produce. n=5 is omitted: its
#: q equals q(6) at G=2 and G=4.
R3_TREADS: tuple[int, ...] = (1, 2, 3, 4, 6)
#: The census's mini plan: NATIVE at these treads, one warmup, one call.
R3_CENSUS_TREADS: tuple[int, ...] = (1, 6)

#: K, the measured calls per cell (V3 reads their spread), and U, the warmup
#: calls per cell ahead of the profiled window (the first compiles).
R3_CALLS_PER_CELL = 3
R3_WARMUP_CALLS = 2

#: ncu's kernel filter: the fused MoE GEMM only, by function name.
R3_KERNEL_FILTER = "regex:^fused_moe_kernel$"

#: The script whose `--counter-child` mode ncu runs.
R3_CHILD = REPO / "scripts" / "private_weight_reference.py"

#: The card the study's timing pages were measured on. Every page and
#: summary says whether it is this one; none of the counter boxes is.
STUDY_CARD = "nvidia_h200"

#: The keys a card block carries, and the ones V0 requires non-empty.
R3_CARD_KEYS: tuple[str, ...] = ("name", "slug", "uuid", "sm_count", "l2_bytes",
                                 "capability", "memory_bytes", "driver",
                                 "study_card", "same_card_as_study")
R3_CARD_REQUIRED: tuple[str, ...] = ("name", "uuid", "sm_count", "l2_bytes",
                                     "capability", "driver")

#: VALIDITY thresholds, each a tolerance on a comparison and none a quantity
#: derived from a calibration.
R3_REPEAT_TOL = 0.01              # V3: (max - min) / median of a cell's K calls
R3_IDENTITY_FLOOR = 0.005         # V4: |R_S(1) / R_P(1) - 1| at least this wide
R3_SPREAD_FACTOR = 3.0            # V4, V7: the floor widens to 3x the repeat spread
R3_Q1_BAND: tuple[float, float] = (0.97, 1.03)     # V4: every arm's q(1)
R3_PRIVATE_BAND: tuple[float, float] = (0.97, 1.5)  # V5: q_P(n) / n and its slope
R3_REQUEST_TOL = 0.005            # V6: requested L2 sectors across arms
R3_DECLARATION_FLOOR = 0.01       # V7: native against shared DRAM reads
R3_COUNTER_TOL = 0.02             # V8: DRAM bytes against 32 x L2 fill sectors
#: CLAIM thresholds.
R3_GROUP_TOL = 0.05               # C1: w1 within 5% of group_reads, G >= 2
R3_FULL_REREAD_MIN = 0.95         # C1 at G=1: q_S,w1(n) >= 0.95 n
R3_W2_CEILING = 1.05              # C2: q_S,w2(n) <= 1.05 group_reads
R3_PRIVATE_EXCESS = 0.03          # C6: q_P,total(n) <= 1.03 n
R3_ALPHA1_CEILING = 1.03          # C5 at G=1: the bracket's upper edge, widened by V4's band

#: GPU TIME, priced by these registered constants and multiplied by the dry
#: run; the page re-prices the rest from the first G's own per-launch time.
#: Per G on 1x H100 SXM5: the child's torch and vLLM import, the weight build,
#: the Triton compiles, ncu's per-launch overhead (a device-side save of the
#: ~26 GB footprint included), and the proof and reduction.
R3_COST_S: dict[str, float] = {"child_start": 25.0, "weight_build": 2.0,
                               "compile": 60.0, "proof_and_reduce": 12.0}
R3_COST_PER_LAUNCH_S = 1.5
R3_COST_PER_LAUNCH_PESSIMISTIC_S = 5.0
#: The metric query, the probe and the census, once per box.
R3_COST_PREFLIGHT_S = 180.0
#: The VM booking the dry run recommends: venv downloads, measurement, one
#: parser or door debug loop, and exfiltration.
R3_VM_BOOKING_H = 1.5

#: The instruments the family's modes stamp. None is `timing.TIMING_BASIS`.
R3_RUN_INSTRUMENT = ("nsight-compute/dram-counters/replay-mode-kernel/cache-control-all/"
                     "clock-control-base/r3-arms-fused_moe_kernel-launches-attributed-"
                     "by-order-and-grid; NOT timing.TIMING_BASIS")
R3_CENSUS_INSTRUMENT = ("nsight-compute/launch-census/no-skip-no-cap/fused_moe_kernel-only; "
                        "counts launches and reads grids, times nothing")
R3_ANALYSE_INSTRUMENT = "arithmetic-over-r3-counter-pages/no-kernel-timed"

R3_SCHEMA_VERSION = 1

#: The keys an r3-arms page carries, the ONE place they are named. The writer
#: (`check_r3_page`, called on every page `--run` writes) and the schema block
#: below are both checked against these by the test suite.
R3_TOP_KEYS: tuple[str, ...] = (
    "family", "schema", "run_id", "provenance", "card", "stack", "ncu", "design",
    "byte_model", "census", "proof", "cells", "group_model", "estimates", "gates")
R3_CELL_KEYS: tuple[str, ...] = (
    "arm", "n", "tokens", "declared", "calls", "launches", "grid", "per_call",
    "per_gemm", "per_call_values", "spread_rel", "recorded")

R3_SCHEMA_TEXT = """\
{
  "family": "r3-arms", "schema": 1, "run_id": "<live card slug>-...",
  "provenance": {...},     # commit, dirty, host: the PV stamp
  "card": {"name", "slug", "uuid", "sm_count", "l2_bytes", "capability",
           "memory_bytes", "driver", "study_card": "nvidia_h200",
           "same_card_as_study"},        # THE LIVE DEVICE, never --card
  "stack": {"torch", "triton", "vllm", "python"},
  "ncu": {"binary", "version", "argv", "replay_mode": "kernel",
          "cache_control": "all", "clock_control": "base", "report",
          "report_sha256", "csv", "csv_layout": "wide"|"long",
          "metrics_asked", "metrics_dropped", "capture_commit"},
  "design": {"model", "dtype", "block_m", "block_n", "block_k", "num_warps",
             "num_stages", "group_m", "treads", "arms", "copies_declared",
             "declared_reason", "declared_by_arm", "calls_per_cell",
             "warmup_calls", "gemms_per_call", "launch_skip", "launch_count",
             "seed"},
  "byte_model": {"W", "W_w1", "W_w2", "operand_per_tile_w1",
                 "operand_per_tile_w2", "source"},
  "census": {"path", "sha256", "gemms_per_call"},
  "proof": {"parts", "detail", "verdict", "tread"},   # R3's five-part proof
  "cells": [ {"arm": "shared", "n": 3, "tokens": 384, "declared": 72,
              "calls": 3, "launches": 6, "grid": {"w1": ..., "w2": ...},
              "per_call": {"dram_bytes_read", "dram_bytes_write",
                           "l2_tex_read_sectors", "gpu_time_ns", ...},
              "per_gemm": {"w1": {..., "grid_size"}, "w2": {...}},
              "per_call_values": [K reads], "spread_rel": 0.001,
              "recorded": {"w1": {occupancy ...}, "w2": {...}} }, ... ],
  "group_model": {"model", "q": {"n": group_reads(E, n, G)}},
  "estimates": {"alpha_slope": {"total", "w1", "w2"}, "residual",
                "alpha_ratio", "alpha_diff", "q_S", "q_P", "q_N"},
  "gates": [asdict(Gate), ...]
}

  PER CALL means one fused_experts call: its w1 launch plus its w2 launch,
  the mean over the cell's K measured calls. `per_call_values` are the K
  individual per-call DRAM reads the repeat gate reads. The byte fields sum
  the two GEMMs; `l2_read_hit_pct` is the two GEMMs' rates weighted by their
  requested L2 read sectors. Cross-check fields are present only when the
  probe proved their metric; recorded ones never gate. `grid` is the grid the
  child derived from vLLM's own sorted-id buffer, and `per_gemm[g].grid_size`
  is the one ncu read off every launch.
"""


def _r3():
    """R3's module, imported on first use by a family function and never by
    the ladder family, so `--family ladder`'s import graph is what it was."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import private_weight_reference as r3
    return r3


# --------------------------------------------------------------------------
# The card-free model of the kernel's schedule.
# --------------------------------------------------------------------------

def group_reads(num_experts: int, n: int, group_m: int) -> float:
    """Weight reads per `fused_experts` call, in units of one weight set, if L2
    serves every re-read INSIDE a GROUP_SIZE_M group and none across groups.

    FROM vLLM v0.27.1's pid mapping in `fused_moe_kernel`: `group_id = pid //
    (GROUP_SIZE_M x num_pid_n)`, `first_pid_m = group_id x GROUP_SIZE_M`, and
    pid_m varies fastest within a group. The sorted tiles are expert-major in
    every arm (`private_topk_ids` keeps PRIVATE's copies of expert e
    contiguous), and the dead tiles the wider declaration adds sit at the
    tail, so expert e owns M-tiles [e n, e n + n) and every group starts at
    pid_m 0. Each (expert, N-tile) weight slab is read once per group its
    expert's tiles fall in, so

        q(n) = (1/E) sum_e ( floor((e n + n - 1) / G) - floor(e n / G) + 1 )

    CARD-FREE: nothing here reads a cache size or an SM count. Where the card
    enters (co-resident CTAs sharing w2 slabs across groups) the claims read
    the page's own recorded occupancy (`coresident_window`).
    """
    if num_experts < 1 or n < 1 or group_m < 1:
        raise ValueError(f"group_reads needs E, n and G >= 1, got {num_experts}, "
                         f"{n}, {group_m}")
    groups = sum((e * n + n - 1) // group_m - (e * n) // group_m + 1
                 for e in range(num_experts))
    return groups / num_experts


def pid_mapping_reads(num_experts: int, n: int, group_m: int, *,
                      num_pid_n: int = 3, dead_tiles: int = 5) -> float:
    """`group_reads` by brute force: walk every pid of the launch grid through
    vLLM's pid mapping, dead tail tiles included, and count the distinct
    (expert, group) pairs whose weights a CTA reads. `--self-test` holds the
    closed form to this; the test suite holds both to its own walk."""
    num_pid_m = num_experts * n + dead_tiles
    in_group = group_m * num_pid_n
    seen: list[set[int]] = [set() for _ in range(num_experts)]
    for pid in range(num_pid_m * num_pid_n):
        group_id = pid // in_group
        first = group_id * group_m
        size = min(num_pid_m - first, group_m)
        pid_m = first + ((pid % in_group) % size)
        if pid_m < num_experts * n:
            seen[pid_m // n].add(group_id)
    return sum(len(s) for s in seen) / num_experts


def coresident_window(sm_count: int, ctas_per_sm: int) -> int:
    """CTAs in flight at once: SMs times the CTAs one SM holds, the latter the
    smallest of the four occupancy limits the page recorded."""
    return int(sm_count) * int(ctas_per_sm)


def pid_n_count(n_cols: int, block_n: int) -> int:
    """`num_pid_n`, the N-tiles per M-row: 2F / BLOCK_N for w1, H / BLOCK_N
    for w2."""
    return -(-int(n_cols) // int(block_n))


def ols_slope(xs, ys) -> float:
    return ols(list(xs), list(ys))[1]


def metric_base(metric: str) -> str:
    """`dram__bytes_read.sum` -> `dram__bytes_read`: the name ncu's metric
    query lists, with the rollup suffix off."""
    return metric.split(".", 1)[0]


def r3_probe_metrics(names) -> tuple[tuple[str, ...], tuple[str, ...], list[str], str]:
    """`(strict, optional, dropped, refusal)` for the probe, from the chip's
    metric list. `names` None (the query could not be read) asks everything;
    a STRICT name the chip lacks is a refusal; a cross-check or recorded name
    it lacks is DROPPED and listed, never asked.

    THE `launch__*` ATTRIBUTES ARE NOT HELD TO THE LIST. They are launch
    statistics ncu records itself rather than hardware counters, and whether
    `--query-metrics` lists them is not verified here; holding them to it
    would let a list that omits them refuse a box whose counters work. The
    probe's own launch proves them or does not, and one this ncu does not
    know costs `probe_ncu` a retry without them, never the verdict.
    """
    rest = R3_CROSSCHECK_METRICS + R3_RECORDED_METRICS
    if names is None:
        return R3_STRICT_METRICS, rest, [], ""

    def offered(m: str) -> bool:
        return m.startswith("launch__") or metric_base(m) in names
    missing = [m for m in R3_STRICT_METRICS if not offered(m)]
    if missing:
        return (R3_STRICT_METRICS, (), [],
                f"this ncu's metric list for the attached chip does not offer the "
                f"STRICT metric(s) {missing}, and every r3-arms page is gated on "
                "them; the route is not open for this family on this box. Read "
                "`ncu --query-metrics` and the ncu version before booking a run")
    return (R3_STRICT_METRICS, tuple(m for m in rest if offered(m)),
            [m for m in rest if not offered(m)], "")


# --------------------------------------------------------------------------
# The card, which is the live device and never a flag.
# --------------------------------------------------------------------------

def live_card_block() -> dict | None:
    """The attached card, read off torch and nvidia-smi, or None with no card.

    The r3-arms page's card is the device it ran on: `--card` is a calibration
    knob of the ladder family and is not read here.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:                                     # noqa: BLE001
        return None
    name = str(props.name)
    driver, _why = PV._nvidia_smi_driver_version()
    slug = PV.card_slug(name)
    return {"name": name, "slug": slug,
            "uuid": str(getattr(props, "uuid", "") or "") or None,
            "sm_count": int(props.multi_processor_count),
            "l2_bytes": int(getattr(props, "L2_cache_size", 0) or 0) or None,
            "capability": f"{props.major}.{props.minor}",
            "memory_bytes": int(props.total_memory), "driver": driver,
            "study_card": STUDY_CARD, "same_card_as_study": slug == STUDY_CARD}


def card_line(card) -> str:
    """THE FIRST LINE OF EVERY PAGE AND SUMMARY: which card every number on it
    belongs to, and that it is not the study's H200 unless it is."""
    if not isinstance(card, dict) or not card.get("name"):
        return ("CARD none: this page carries no card block, so nothing on it names "
                "the card it came from, and V0 fails it; the study's timing pages "
                f"are {STUDY_CARD}.")
    cc = str(card.get("capability") or "?").replace(".", "")
    l2 = card.get("l2_bytes")
    l2_text = f"{l2 / 2 ** 20:g}" if l2 else "?"
    return (f"CARD {card['name']} ({card.get('slug')}, UUID {card.get('uuid')}, "
            f"sm_{cc}, {card.get('sm_count')} SMs, {l2_text} MiB L2): every number "
            f"here is THIS card's; the study's timing pages are {STUDY_CARD}.")


def r3_stack_versions() -> dict:
    """torch, triton and vLLM as this interpreter has them, and the Python."""
    import platform
    from importlib import metadata
    out: dict = {"python": platform.python_version()}
    for dist in ("torch", "triton", "vllm"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _sha256(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------
# The plan, the byte model, and the argv, all off GPU.
# --------------------------------------------------------------------------

def r3_byte_model(cfg, dtype: str, block_m: int) -> dict:
    """W and the per-tread GEMM operand reads, from their owners.

    `W_w1`, `W_w2` from `moe.bench.weights.routed_expert_weight_bytes_by_gemm`;
    the operand terms are `gemm_operand_read_bytes_per_row` times E x BLOCK_M,
    the A-operand bytes one more M-tile per expert makes each GEMM read. They
    are charged as cold reads, because ncu flushes every cache before each
    profiled launch.
    """
    from moe.bench import weights as WEIGHTS
    by = WEIGHTS.routed_expert_weight_bytes_by_gemm(cfg, dtype)
    row = gemm_operand_read_bytes_per_row(cfg, dtype_bytes(dtype))
    return {"W": by["w1"] + by["w2"], "W_w1": by["w1"], "W_w2": by["w2"],
            "operand_per_tile_w1": cfg.num_experts * block_m * row["w1"],
            "operand_per_tile_w2": cfg.num_experts * block_m * row["w2"],
            "source": ("moe.bench.weights.routed_expert_weight_bytes_by_gemm; "
                       "block_m_crossing_sweep.gemm_operand_read_bytes_per_row x E "
                       "x BLOCK_M")}


def r3_plan(*, model: str, dtype: str, block_m: int, block_n: int, num_stages: int,
            group_m: int, treads, kind: str, arms, calls: int, warmups: int,
            profile_dir: Path, stem: str) -> dict:
    """The plan the child runs, validated by R3 before anything touches a box.

    The declaration is R3's own (`counter_declaration`, over R3's whole ladder
    and not over this subset), and `validate_counter_plan` and
    `counter_schedule` refuse it here exactly as the child would.
    """
    r3 = _r3()
    cfg = MODEL_CONFIGS[model]
    copies, reason = r3.counter_declaration(cfg, block_m)
    treads = [int(n) for n in treads]
    plan = {"family": R3_FAMILY, "kind": kind, "model": model, "dtype": dtype,
            "block_m": int(block_m), "block_n": int(block_n),
            "num_stages": int(num_stages), "group_m": int(group_m),
            "treads": treads, "arms": list(arms),
            "cells": [[a, n] for a in arms for n in treads],
            "copies_declared": copies, "declared_reason": reason,
            "calls_per_cell": int(calls), "warmup_calls": int(warmups),
            "gemms_per_call": r3.GEMMS_PER_CALL, "seed": 0,
            "manifest": str(Path(profile_dir) / f"{stem}.manifest.json"),
            "triton_cache": str(Path(profile_dir) / f"{stem}.triton-cache")}
    r3.validate_counter_plan(plan)
    r3.counter_schedule(plan)
    return plan


def r3_design(plan: dict) -> dict:
    """The page's `design` block, read off the plan and R3's own pin."""
    r3 = _r3()
    cfg = MODEL_CONFIGS[plan["model"]]
    sched = r3.counter_schedule(plan)
    pinned = r3.pinned_config(plan["block_n"], plan["group_m"], plan["num_stages"])
    copies = int(plan["copies_declared"])
    return {"model": plan["model"], "dtype": plan["dtype"],
            "block_m": int(plan["block_m"]), "block_n": pinned["BLOCK_SIZE_N"],
            "block_k": pinned["BLOCK_SIZE_K"], "num_warps": pinned["num_warps"],
            "num_stages": pinned["num_stages"], "group_m": pinned["GROUP_SIZE_M"],
            "treads": list(plan["treads"]), "arms": list(plan["arms"]),
            "copies_declared": copies, "declared_reason": plan.get("declared_reason"),
            "declared_by_arm": {a: r3.declared_experts(a, cfg.num_experts, copies)
                                for a in r3.ARMS},
            "calls_per_cell": int(plan["calls_per_cell"]),
            "warmup_calls": int(plan["warmup_calls"]),
            "gemms_per_call": int(plan["gemms_per_call"]),
            "launch_skip": sched.launch_skip, "launch_count": sched.launch_count,
            "seed": int(plan["seed"])}


def r3_ncu_argv(binary: str, plan_path: Path, report_path: Path, metrics, *,
                launch_skip: int, launch_count: int | None,
                python: str | None = None, child: Path = R3_CHILD) -> list[str]:
    """ncu over R3's child: the arm GEMMs only, at a cold L2, a base clock,
    and exactly the planned launch window.

      `--cache-control all`   (from `ncu_common_flags`) flushes every cache
                              before each replay pass of every profiled launch,
                              so each GEMM starts cold. `none` is NOT run:
                              under kernel replay ncu's first-pass save of all
                              accessible memory streams through L2 just before
                              the kernel, so "none" is "the L2 ncu's save left".
      `--clock-control base`  passed and recorded: the documented default has
                              moved between versions, and bytes should not
                              care, so no default is trusted.
      `-k regex:^fused_moe_kernel$ --kernel-name-base function`
                              the GEMM launches only; the alignment, the
                              activation and the reduction run unprofiled.
      `--launch-skip`, `--launch-count`
                              from `counter_schedule`: the warmups are
                              skipped, and exactly K calls per cell are kept.
      `--nvtx`                the child's ranges, for a human; nothing filters
                              on them.

    `python` is this interpreter unless the dry run prints a placeholder: the
    child runs under whichever interpreter runs this file, so --run is run
    from the vLLM venv.
    """
    argv = [binary, "--target-processes", "all", *ncu_common_flags("all"),
            "--clock-control", "base", "--nvtx", "-k", R3_KERNEL_FILTER,
            "--kernel-name-base", "function", "--launch-skip", str(launch_skip)]
    if launch_count is not None:
        argv += ["--launch-count", str(launch_count)]
    return argv + ["--metrics", ",".join(metrics), "--export", str(report_path),
                   "--force-overwrite", "--", python or sys.executable, str(child),
                   "--counter-child", str(plan_path)]


def r3_import_argv(binary: str, report_path: Path) -> list[str]:
    """The reduction, split from the capture: a parser defect costs a laptop
    fix and not a re-rent, because the `.ncu-rep` is kept."""
    return [binary, "--import", str(report_path), "--csv", "--page", "raw",
            "--print-units", "base"]


# --------------------------------------------------------------------------
# Attribution and reduction: every launch to its (cell, call, GEMM).
# --------------------------------------------------------------------------

def r3_launch_sequence(manifest: dict) -> list[tuple[str, int, str, bool]]:
    """`(cell key, call, gemm, warmup)` for every launch ncu profiled, in order.

    A page profiles the K measured calls per cell, cells in manifest order,
    each call's w1 launch then its w2 launch. A census (no launch count)
    profiles the warmups too, all of them first, as the child makes them.
    """
    gemms = _r3().GEMMS
    if int(manifest["gemms_per_call"]) != len(gemms):
        raise CounterRunRefused(
            f"the manifest says {manifest['gemms_per_call']} GEMMs per call and the "
            f"attribution knows {len(gemms)}")
    order = [(str(a), int(n)) for a, n in manifest["order"]]
    seq: list[tuple[str, int, str, bool]] = []
    if manifest.get("launch_count") is None:
        for a, n in order:
            for u in range(int(manifest["warmup_calls"])):
                seq += [(f"{a}/{n}", u, g, True) for g in gemms]
    for a, n in order:
        for k in range(int(manifest["calls_per_cell"])):
            seq += [(f"{a}/{n}", k, g, False) for g in gemms]
    return seq


def _launch_order(launches: list[Launch]) -> list[Launch]:
    """By ncu's launch ID when every ID is an integer, else as printed."""
    try:
        return sorted(launches, key=lambda ln: int(ln.launch_id))
    except ValueError:
        return list(launches)


def attribute_launches(launches: list[Launch], manifest: dict) -> list[dict]:
    """Every profiled launch mapped to the manifest's (cell, call, GEMM).

    EXACT, not a floor and not "at least": the profile must hold exactly the
    launches the manifest planned, launch i is the i-th entry of
    `r3_launch_sequence`, and every launch's `launch__grid_size` must equal the
    grid the child derived for that arm, tread and GEMM from vLLM's own
    sorted-id buffer. Any extra, missing or mis-gridded launch raises
    `CounterRunRefused`, which exits INVALID: a byte total that cannot be
    attributed launch by launch is the per-call trap in another form.
    """
    seq = r3_launch_sequence(manifest)
    ordered = _launch_order(launches)
    if len(ordered) != len(seq):
        raise CounterRunRefused(
            f"the profile holds {len(ordered)} fused_moe_kernel launches and the "
            f"manifest planned EXACTLY {len(seq)} ({len(manifest['order'])} cells x "
            f"{manifest['calls_per_cell']} calls x {manifest['gemms_per_call']} "
            "GEMMs); an extra or a missing launch shifts every attribution after "
            "it, so nothing may be divided")
    out = []
    for i, (ln, (key, call, gemm, warm)) in enumerate(zip(ordered, seq, strict=True)):
        if GEMM_MARKER not in ln.kernel:
            raise CounterRunRefused(
                f"launch {i} (ID {ln.launch_id}) is {ln.kernel!r}, not {GEMM_MARKER}; "
                "the kernel filter let through a launch the manifest never planned")
        grid = ln.metrics.get("launch__grid_size")
        want = int(manifest["grids"][key][gemm])
        if grid is None or int(round(grid)) != want:
            raise CounterRunRefused(
                f"launch {i} (ID {ln.launch_id}) is attributed to {key} call {call} "
                f"{gemm} and ran a grid of {grid}, where the child derived {want} from "
                "vLLM's own sorted-id buffer; the launch order and the manifest "
                "disagree, so no launch after it can be attributed")
        out.append({"key": key, "call": call, "gemm": gemm, "warmup": warm,
                    "launch": ln})
    return out


def r3_spread(values) -> float | None:
    """(max - min) / median of a cell's per-call reads, or None when fewer
    than two readings exist or any is missing."""
    vals = list(values)
    if len(vals) < 2 or any(v is None for v in vals):
        return None
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med > 0 else None


def r3_reduce_cells(attributed: list[dict], manifest: dict, metrics_asked) -> list[dict]:
    """The attributed launches as page cells: per GEMM, per call, the K values.

    PER CALL is one fused_experts call, its w1 launch plus its w2 launch,
    averaged over the cell's K measured calls. K is read off the manifest and
    is also the number of calls counted from the attributed launches; the two
    must agree or the cell refuses.
    """
    gemms = _r3().GEMMS
    k = int(manifest["calls_per_cell"])
    asked = set(metrics_asked)
    by_key: dict[str, dict[int, dict[str, Launch]]] = {}
    for rec in attributed:
        if rec["warmup"]:
            continue
        by_key.setdefault(rec["key"], {}).setdefault(rec["call"], {})[rec["gemm"]] = \
            rec["launch"]
    cells = []
    for arm, n in manifest["order"]:
        key = f"{arm}/{n}"
        calls = by_key.get(key, {})
        if sorted(calls) != list(range(k)) or any(set(c) != set(gemms)
                                                   for c in calls.values()):
            raise CounterRunRefused(
                f"cell {key} counted {len(calls)} complete calls against the "
                f"manifest's {k}; K is read off the manifest AND off the launch list, "
                "and the two must agree")
        per_gemm: dict[str, dict] = {}
        recorded: dict[str, dict] = {}
        for g in gemms:
            runs = [calls[i][g] for i in range(k)]
            vals: dict = {}
            for metric, name in R3_FIELDS.items():
                if metric not in asked:
                    continue
                xs = [ln.metrics.get(metric) for ln in runs]
                vals[name] = (statistics.fmean(xs) if all(x is not None for x in xs)
                              else None)
            grids = {int(round(ln.metrics["launch__grid_size"])) for ln in runs}
            vals["grid_size"] = grids.pop() if len(grids) == 1 else None
            per_gemm[g] = vals
            rec = {}
            for metric in R3_RECORDED_METRICS:
                xs = [ln.metrics[metric] for ln in runs if metric in ln.metrics]
                if len(xs) == k:
                    rec[metric] = statistics.fmean(xs)
            recorded[g] = rec
        per_call_values = []
        for i in range(k):
            xs = [calls[i][g].metrics.get("dram__bytes_read.sum") for g in gemms]
            per_call_values.append(None if any(x is None for x in xs) else sum(xs))
        per_call: dict = {}
        for name in per_gemm[gemms[0]]:
            if name == "grid_size":
                continue
            xs = [per_gemm[g].get(name) for g in gemms]
            if any(x is None for x in xs):
                per_call[name] = None
            elif name in R3_RATE_FIELDS:
                w = [per_gemm[g].get("l2_tex_read_sectors") or 0.0 for g in gemms]
                per_call[name] = (sum(x * wi for x, wi in zip(xs, w, strict=True)) / sum(w)
                                  if sum(w) > 0 else statistics.fmean(xs))
            else:
                per_call[name] = sum(xs)
        cells.append({
            "arm": str(arm), "n": int(n),
            "tokens": int(manifest["tokens"][str(n)]),
            "declared": int(manifest["declared_by_arm"][arm]),
            "calls": k, "launches": k * len(gemms),
            "grid": {g: int(manifest["grids"][key][g]) for g in gemms},
            "per_call": per_call, "per_gemm": per_gemm,
            "per_call_values": per_call_values,
            "spread_rel": r3_spread(per_call_values), "recorded": recorded})
    return cells


# --------------------------------------------------------------------------
# The estimators: no bandwidth, no ridge, no intercept, no calibration.
# --------------------------------------------------------------------------

def _cell_map(payload: dict) -> dict[tuple[str, int], dict]:
    return {(str(c["arm"]), int(c["n"])): c for c in payload["cells"]}


def r3_q(payload: dict, byte_model: dict | None = None) -> dict:
    """`{arm: {"total"|"w1"|"w2": {n: q}}}`, q(n) = (R(n) - n x operand) / W.

    R is `dram__bytes_read.sum` per call, per GEMM or both GEMMs together;
    the operand term is the GEMM's A-operand read per extra tread and is
    0.33% of W on mixtral at BLOCK_M 32.
    """
    design = payload["design"]
    bm = byte_model or r3_byte_model(MODEL_CONFIGS[design["model"]], design["dtype"],
                                     int(design["block_m"]))
    op = {"w1": bm["operand_per_tile_w1"], "w2": bm["operand_per_tile_w2"]}
    weight = {"w1": bm["W_w1"], "w2": bm["W_w2"], "total": bm["W"]}
    out: dict = {}
    for (arm, n), c in _cell_map(payload).items():
        q = out.setdefault(arm, {"total": {}, "w1": {}, "w2": {}})
        q["total"][n] = (c["per_call"]["dram_bytes_read"] - n * (op["w1"] + op["w2"])) \
            / weight["total"]
        for g in ("w1", "w2"):
            q[g][n] = (c["per_gemm"][g]["dram_bytes_read"] - n * op[g]) / weight[g]
    return out


def r3_estimates(payload: dict) -> dict:
    """The page's estimates, recomputed from its cells.

    The PRIMARY product is q_S(n) per tread beside `group_reads`: the model
    predicts a staircase, so per-tread counts are the result and alpha is a
    summary of them. `alpha_slope` is the OLS slope of q_S over n, printed
    with its max relative residual and labelled a scalar summary, meaningful
    where the ladder is affine. `alpha_ratio` is slope(R_S) / slope(R_P), the
    byte analogue of R3's timed ratio. `alpha_diff` is 1 - (slope_P -
    slope_S) / W, which cancels any activation term the two arms share.
    """
    design = payload["design"]
    cfg = MODEL_CONFIGS[design["model"]]
    bm = r3_byte_model(cfg, design["dtype"], int(design["block_m"]))
    q = r3_q(payload, bm)
    shared, private = "shared", "private"
    treads = sorted(q[shared]["total"])
    cells = _cell_map(payload)

    def slope(arm: str, part: str) -> float:
        return ols_slope(treads, [q[arm][part][n] for n in treads])

    a_s, b_s = ols(treads, [q[shared]["total"][n] for n in treads])
    resid = max(abs((a_s + b_s * n) - q[shared]["total"][n]) / abs(q[shared]["total"][n])
                for n in treads)
    raw_s = ols_slope(treads, [cells[(shared, n)]["per_call"]["dram_bytes_read"]
                               for n in treads])
    raw_p = ols_slope(treads, [cells[(private, n)]["per_call"]["dram_bytes_read"]
                               for n in treads])
    return {
        "alpha_slope": {p: slope(shared, p) for p in ("total", "w1", "w2")},
        "residual": resid,
        "residual_note": "a scalar summary; meaningful where the ladder is affine",
        "alpha_slope_private": {p: slope(private, p) for p in ("total", "w1", "w2")},
        "alpha_ratio": raw_s / raw_p if raw_p else None,
        "alpha_diff": 1.0 - (raw_p - raw_s) / bm["W"],
        "q_S": {p: {str(n): v for n, v in q[shared][p].items()} for p in q[shared]},
        "q_P": {p: {str(n): v for n, v in q[private][p].items()} for p in q[private]},
        "q_N": ({p: {str(n): v for n, v in q["native"][p].items()} for p in q["native"]}
                if "native" in q else None),
    }


# --------------------------------------------------------------------------
# The gates. VALIDITY fails exit INVALID and void the page; CLAIM fails exit
# CLAIM_FAIL and are results.
# --------------------------------------------------------------------------

#: The design a timed report must share with a counter page before C5 may
#: compare them: the kernel R3 timed, G aside (C5 matches G itself). The
#: timed report's key, and the page design key it is held to.
R3_TIMED_DESIGN: tuple[tuple[str, str], ...] = (
    ("model", "model"), ("dtype", "dtype"), ("block_m", "block_m"),
    ("pinned.BLOCK_SIZE_N", "block_n"), ("pinned.BLOCK_SIZE_K", "block_k"),
    ("pinned.num_warps", "num_warps"), ("pinned.num_stages", "num_stages"))


def _timed_value(rep: dict, key: str):
    if key.startswith("pinned."):
        return (rep.get("pinned") or {}).get(key.split(".", 1)[1])
    return rep.get(key)


def load_timed_reference(paths) -> dict[int, dict]:
    """R3's timed report.json files, grouped by GROUP_SIZE_M: the ratio each
    reads, their mean and sd across seeds, the card they were timed on, and
    each run's duty and fit window. READ, never typed: the 0.915 / 0.706 /
    0.680 / 0.617 of session 5 are whatever these files hold.

    REFUSES what R3 itself would not pool as a replicate. Until 2026-09-24
    this read only the experiment, the ratio, G, the card, the seed and the
    window, so C5 scored a page's bytes against a planted report, an INVALID
    one, or one from another tile, model, dtype or duty. Now a report that is
    not R3's, formed no ratio, is planted (`synthetic`), or whose own VALIDITY
    gates did not all PASS is refused; the runs pooled at one G must share
    R3's design (`R3_TIMED_DESIGN`, the duty and the fit window), because a
    mean and sd over two designs is two estimators in one envelope; and the
    page's own design is held to them by `timed_reference_mismatch`.
    """
    r3 = _r3()
    by_g: dict[int, list[dict]] = {}
    for p in paths or ():
        rep = json.loads(Path(p).read_text())
        if rep.get("experiment") != "private_weight_reference" or rep.get("ratio") is None:
            raise CounterRunRefused(
                f"{p} is not an R3 report with a ratio; --timed-reference reads "
                "private_weight_reference report.json files")
        if rep.get("synthetic"):
            raise CounterRunRefused(f"{p} is a planted (--self-test) R3 report; C5 "
                                    "compares bytes with a measured timing")
        gates = rep.get("gates") or []
        broken = [f"{g.get('tag') or g.get('number')} {g.get('verdict')}" for g in gates
                  if g.get("kind") == "VALIDITY" and g.get("verdict") != PASS]
        if not gates or broken:
            raise CounterRunRefused(
                f"{p} is not a VALID R3 page ("
                + (f"VALIDITY gates {broken}" if broken else "it carries no gates")
                + "); its ratio is not quotable, so C5 does not compare with it")
        g = int(rep["pinned"]["GROUP_SIZE_M"])
        by_g.setdefault(g, []).append({
            "path": str(p), "ratio": float(rep["ratio"]), "card": rep.get("card"),
            "seed": rep.get("seed"),
            "duty": float(r3.design_value(rep, "duty")),
            "claim_min_tread": int(r3.design_value(rep, "claim_min_tread")),
            "design": {k: _timed_value(rep, k) for k, _page_key in R3_TIMED_DESIGN}})
    out = {}
    for g, rows in by_g.items():
        for key in ("design", "duty", "claim_min_tread"):
            seen = {json.dumps(r[key], sort_keys=True) for r in rows}
            if len(seen) > 1:
                raise CounterRunRefused(
                    f"the timed reports at G={g} differ in {key} ({sorted(seen)}); a "
                    "mean and sd over two designs pools two estimators, which R3 "
                    "refuses for its own replicates")
        ratios = [r["ratio"] for r in rows]
        out[g] = {"G": g, "mean": statistics.fmean(ratios),
                  "sd": statistics.stdev(ratios) if len(ratios) >= 2 else 0.0,
                  "runs": rows, "cards": sorted({str(r["card"]) for r in rows}),
                  "design": rows[0]["design"], "duty": rows[0]["duty"],
                  "windows": sorted({int(r["claim_min_tread"]) for r in rows})}
    return out


def timed_reference_mismatch(ref: dict, design: dict) -> list[str]:
    """How a G's timed reports differ from a counter page's design, G aside:
    `[]` when C5 may compare them. Called by `--analyse` before any page is
    scored (it refuses the join) and by C5 itself, so one rule decides both."""
    return [f"{key} {ref['design'].get(key)!r} against the page's {page_key} "
            f"{design.get(page_key)!r}"
            for key, page_key in R3_TIMED_DESIGN
            if ref["design"].get(key) != design.get(page_key)]


def _worst(items) -> str:
    return "; ".join(items[:4]) + (f"; and {len(items) - 4} more" if len(items) > 4 else "")


def score_r3_page(payload: dict, *, timed: dict | None = None) -> tuple[list[Gate], dict]:
    """Score one r3-arms page. Pure over the payload, recomputing every number
    from its cells; nothing stored under `estimates` is trusted.

    WHAT IS DELIBERATELY NOT HERE: the ladder family's V3 (monotone) and V4
    (affine) are not applied to SHARED. The group model predicts non-monotone,
    non-affine shared ladders at G = 2, 4 and 16 (q(4) = 1 < q(3) = 1.5 at
    G=4), and those two gates would void a correct page. A test holds that a
    planted G=4 staircase with its n=4 drop is VALID.
    """
    for key in ("family", "design", "cells"):
        if key not in payload:
            raise KeyError(f"r3 counter page has no '{key}'; refusing to score a "
                           "partial page")
    if payload["family"] != R3_FAMILY:
        raise KeyError(f"family {payload['family']!r} is not {R3_FAMILY!r}")
    r3 = _r3()
    design = payload["design"]
    cfg = MODEL_CONFIGS[design["model"]]
    e = cfg.num_experts
    g_m = int(design["group_m"])
    treads = sorted(int(n) for n in design["treads"])
    arms = list(design["arms"])
    cells = _cell_map(payload)
    k = int(design["calls_per_cell"])
    gemms = r3.GEMMS
    gates: list[Gate] = []
    summary: dict = {"group_m": g_m, "estimates": None, "not_asked": []}

    # V0 THE CARD.
    card = payload.get("card")
    absent = [c for c in R3_CARD_REQUIRED
              if not isinstance(card, dict) or card.get(c) in (None, "")]
    gates.append(Gate(
        "V0", "VALIDITY", "the page names the live card it was measured on",
        PASS if not absent else FAIL,
        card_line(card) if not absent else f"card block missing {absent}",
        f"a card block with {list(R3_CARD_REQUIRED)}",
        "every number on the page: bytes without the card they came from cannot be "
        "compared with anything, least of all the study's H200"))

    # V1 COUNT AND ATTRIBUTION, and the census that proved GEMMS_PER_CALL.
    want = [(a, n) for a in arms for n in treads]
    missing = [f"{a}/{n}" for a, n in want if (a, n) not in cells]
    extra = [f"{a}/{n}" for a, n in cells if (a, n) not in want]
    problems = []
    if missing or extra:
        problems.append(f"cells missing {missing}, extra {extra}")
    if len(treads) < 4:
        problems.append(f"{len(treads)} treads; the scorer needs four for an OLS "
                        "residual")
    if set(arms) != set(r3.ARMS):
        problems.append(f"arms {arms}, not R3's {list(r3.ARMS)}")
    total = 0
    for (a, n), c in cells.items():
        if int(c.get("calls", -1)) != k:
            problems.append(f"{a}/{n} made {c.get('calls')} calls against K={k}")
        if int(c.get("launches", -1)) != k * int(design["gemms_per_call"]):
            problems.append(f"{a}/{n} holds {c.get('launches')} launches")
        total += int(c.get("launches", 0) or 0)
        for g in gemms:
            got = ((c.get("per_gemm") or {}).get(g) or {}).get("grid_size")
            planned = (c.get("grid") or {}).get(g)
            if got is None or planned is None or int(got) != int(planned):
                problems.append(f"{a}/{n} {g} grid {got} against {planned}")
    if design.get("launch_count") is None or total != int(design["launch_count"]):
        problems.append(f"{total} launches attributed against the planned "
                        f"{design.get('launch_count')}")
    census = payload.get("census") or {}
    if int(design["gemms_per_call"]) != r3.GEMMS_PER_CALL or \
            census.get("gemms_per_call") != r3.GEMMS_PER_CALL:
        problems.append(f"GEMMs per call: design {design['gemms_per_call']}, census "
                        f"{census.get('gemms_per_call')}, cited {r3.GEMMS_PER_CALL}")
    gates.append(Gate(
        "V1", "VALIDITY", "every launch is attributed: exact count, every grid "
        "equal to the child's, and a census that measured GEMMS_PER_CALL",
        PASS if not problems else FAIL,
        _worst(problems) if problems else
        f"{total} launches over {len(cells)} cells, every grid matched",
        f"exact count; grids equal; census GEMMS_PER_CALL = {r3.GEMMS_PER_CALL}",
        "every byte on the page: a launch counted twice or into the wrong cell "
        "is the per-call trap in another form, and it still fits a line"))
    if missing or len(treads) < 4 or set(arms) != set(r3.ARMS):
        summary["reason"] = "V1 failed on the cell set; nothing else is scored"
        return gates, summary

    # V2 METRICS.
    holes = []
    for (a, n), c in sorted(cells.items()):
        for g in gemms:
            for f in R3_STRICT_FIELDS:
                v = ((c.get("per_gemm") or {}).get(g) or {}).get(f)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    holes.append(f"{a}/{n} {g} {f}={v}")
        if any(not isinstance(v, (int, float)) for v in c.get("per_call_values") or [None]):
            holes.append(f"{a}/{n} per-call reads {c.get('per_call_values')}")
    gates.append(Gate(
        "V2", "VALIDITY", "every STRICT metric is a number on every launch",
        PASS if not holes else FAIL,
        _worst(holes) if holes else f"{len(R3_STRICT_FIELDS)} fields x "
                                    f"{len(cells) * len(gemms)} GEMM cells",
        "no missing value, never a 0.0 default",
        "every estimate: a metric that was not measured is not a zero"))
    if holes:
        summary["reason"] = "V2 failed; no estimate is formed over a missing metric"
        return gates, summary

    # V3 REPEAT.
    spreads = {key: r3_spread(c["per_call_values"]) for key, c in cells.items()}
    over = [f"{a}/{n} " + ("unformed" if s is None else f"{s:.4f}")
            for (a, n), s in sorted(spreads.items()) if s is None or s > R3_REPEAT_TOL]
    worst_spread = max((s for s in spreads.values() if s is not None), default=math.nan)
    gates.append(Gate(
        "V3", "VALIDITY", "each cell's K calls read the same bytes",
        PASS if not over else FAIL,
        f"worst (max - min) / median {worst_spread:.4%}" + (f"; over: {_worst(over)}"
                                                            if over else ""),
        f"<= {R3_REPEAT_TOL:.0%} in every cell",
        "the L2 state or the attribution: with a cold cache per launch the same "
        "call reads the same bytes, and a spread says it did not"))

    bm = r3_byte_model(cfg, design["dtype"], int(design["block_m"]))
    q = r3_q(payload, bm)
    est = r3_estimates(payload)
    summary["estimates"] = est
    gm = {n: group_reads(e, n, g_m) for n in treads}
    summary["group_model"] = gm

    def spread_of(*keys) -> float:
        return max((spreads.get(x) or 0.0) for x in keys)

    # V4 IDENTITY at n = 1.
    lines = []
    if 1 not in treads:
        v4, measured = FAIL, "no n=1 tread on the page"
    else:
        rs = cells[("shared", 1)]["per_call"]["dram_bytes_read"]
        rp = cells[("private", 1)]["per_call"]["dram_bytes_read"]
        tol = max(R3_IDENTITY_FLOOR,
                  R3_SPREAD_FACTOR * spread_of(("shared", 1), ("private", 1)))
        q1 = {a: q[a]["total"][1] for a in arms}
        off = abs(rs / rp - 1.0)
        lo, hi = R3_Q1_BAND
        v4 = PASS if off <= tol and all(lo <= v <= hi for v in q1.values()) else FAIL
        measured = (f"|R_S(1)/R_P(1) - 1| = {off:.4%} against {tol:.4%}; q(1) "
                    + ", ".join(f"{a} {v:.4f}" for a, v in q1.items()))
        lines.append("at n=1 SHARED and PRIVATE route every tile to copy 0: they are "
                     "the same call, so their bytes must agree")
    gates.append(Gate(
        "V4", "VALIDITY", "at n=1 the arms are one call and each reads one weight set",
        v4, measured,
        f"<= max({R3_IDENTITY_FLOOR:.1%}, {R3_SPREAD_FACTOR:g} x the repeat spread); "
        f"q(1) in [{R3_Q1_BAND[0]}, {R3_Q1_BAND[1]}]",
        "the byte model's intercept: if one call reads other than W at n=1, q is "
        "not a re-read fraction", lines))

    # V5 THE PRIVATE CONTROL.
    lo, hi = R3_PRIVATE_BAND
    bad = [f"n={n} {p} q_P {q['private'][p][n]:.4f}" for n in treads
           for p in ("w1", "w2", "total")
           if not (lo * n <= q["private"][p][n] <= hi * n)]
    slope_p = est["alpha_slope_private"]["total"]
    if not (lo <= slope_p <= hi):
        bad.append(f"slope q_P,total {slope_p:.4f}")
    gates.append(Gate(
        "V5", "VALIDITY", "the PRIVATE arm re-reads the whole weight set per tile",
        PASS if not bad else FAIL,
        _worst(bad) if bad else f"every q_P(n)/n in [{lo}, {hi}]; slope {slope_p:.4f}",
        f"{lo} n <= q_P(n) <= {hi} n per GEMM; slope of q_P,total in [{lo}, {hi}]",
        "the apparatus. Below the floor, weight bytes are MISSING, which is "
        "impossible for an arm whose every tile reads its own copy: the counter, "
        "the attribution or the relabelling is broken (V9 should agree). Above "
        f"{hi} n is a per-call or double-count error"))

    # V6 REQUESTED IDENTITY.
    mism = []
    for n in treads:
        for g in gemms:
            vals = [cells[(a, n)]["per_gemm"][g]["l2_tex_read_sectors"] for a in arms]
            rel = max(vals) / min(vals) - 1.0 if min(vals) > 0 else math.inf
            if rel > R3_REQUEST_TOL:
                mism.append(f"n={n} {g} {rel:.4%}")
    gates.append(Gate(
        "V6", "VALIDITY", "the three arms request the same L2 read sectors",
        PASS if not mism else FAIL,
        _worst(mism) if mism else "every (n, GEMM) within tolerance",
        f"max/min - 1 <= {R3_REQUEST_TOL:.1%} across arms",
        "the arms' identity: they issue identical loads (the dead CTAs the wider "
        "72-slot declaration adds each load one expert id and exit), so a "
        "difference in requests is a different call"))

    # V7 DECLARATION.
    decl = []
    for n in treads:
        tol = max(R3_DECLARATION_FLOOR,
                  R3_SPREAD_FACTOR * spread_of(("native", n), ("shared", n)))
        for g in gemms:
            rn = cells[("native", n)]["per_gemm"][g]["dram_bytes_read"]
            rs = cells[("shared", n)]["per_gemm"][g]["dram_bytes_read"]
            if abs(rn / rs - 1.0) > tol:
                decl.append(f"n={n} {g} {rn / rs - 1.0:+.4%} against {tol:.4%}")
    gates.append(Gate(
        "V7", "VALIDITY", "NATIVE and SHARED read the same DRAM bytes",
        PASS if not decl else FAIL,
        _worst(decl) if decl else "every (n, GEMM) within tolerance",
        f"|R_N/R_S - 1| <= max({R3_DECLARATION_FLOOR:.0%}, {R3_SPREAD_FACTOR:g} x the "
        "repeat spread)",
        "SHARED as the study's call: if the declaration moved the bytes, SHARED "
        "measures a call vLLM does not make"))

    # V8 COUNTER CONSISTENCY, asked only when its metric was proven.
    asked = set((payload.get("ncu") or {}).get("metrics_asked") or [])
    v8 = None
    if "lts__d_sectors_fill_device.sum" in asked:
        off8 = []
        for (a, n), c in sorted(cells.items()):
            for g in gemms:
                fill = c["per_gemm"][g].get("l2_fill_device_sectors")
                r = c["per_gemm"][g]["dram_bytes_read"]
                if not fill or abs(r / (L2_SECTOR_BYTES * fill) - 1.0) > R3_COUNTER_TOL:
                    off8.append(f"{a}/{n} {g} fill {fill}")
        v8 = (PASS if not off8 else FAIL, _worst(off8) if off8 else
              "every GEMM cell's DRAM bytes within tolerance of 32 x fill sectors",
              f"|dram_bytes_read / (32 x lts__d_sectors_fill_device) - 1| <= "
              f"{R3_COUNTER_TOL:.0%}")
    elif "lts__t_sectors_op_read_lookup_miss.sum" in asked:
        off8 = []
        for (a, n), c in sorted(cells.items()):
            for g in gemms:
                miss = c["per_gemm"][g].get("l2_read_miss_sectors")
                r = c["per_gemm"][g]["dram_bytes_read"]
                if miss is None or r > (1 + R3_COUNTER_TOL) * L2_SECTOR_BYTES * miss:
                    off8.append(f"{a}/{n} {g} miss {miss}")
        v8 = (PASS if not off8 else FAIL, _worst(off8) if off8 else
              "every GEMM cell's DRAM bytes under 1.02 x 32 x L2 read misses",
              f"dram_bytes_read <= {1 + R3_COUNTER_TOL:g} x 32 x "
              "lts__t_sectors_op_read_lookup_miss")
    if v8 is None:
        summary["not_asked"].append("V8: neither L2 fill nor L2 miss sectors were "
                                    "proven readable on this box")
    else:
        gates.append(Gate("V8", "VALIDITY", "the DRAM counter and the L2 counters agree",
                          v8[0], v8[1], v8[2],
                          "the counter itself: DRAM bytes and the L2 sectors filled "
                          "from DRAM are two readings of one traffic"))

    # V9 R3's BUFFER PROOF, through R3's own gate.
    proof = payload.get("proof") or {}
    bp = r3.BufferProof(parts=dict(proof.get("parts") or {}),
                        detail=dict(proof.get("detail") or {}),
                        synthetic=bool(proof.get("synthetic", False)))
    g9 = r3.gate_v2_distinct_buffers(bp)
    gates.append(Gate("V9", "VALIDITY", g9.claim, g9.verdict, g9.measured, g9.threshold,
                      g9.consequence, list(g9.lines)))

    # C1 GROUP ARITHMETIC ON w1.
    q_sw1 = q["shared"]["w1"]
    if g_m >= 2:
        off1 = {n: abs(q_sw1[n] - gm[n]) / gm[n] for n in treads}
        gates.append(Gate(
            "C1", "CLAIM", "w1 reads exactly what the group model counts",
            PASS if all(v <= R3_GROUP_TOL for v in off1.values()) else FAIL,
            ", ".join(f"n={n} {q_sw1[n]:.4f}/{gm[n]:.4f}" for n in treads),
            f"|q_S,w1(n) - group_reads| <= {R3_GROUP_TOL:.0%} x group_reads at every n",
            "the pid-order reuse model: w1 has 2F/BN N-tiles per M-row, far more "
            "than the CTAs in flight, so nothing is shared across groups"))
    else:
        window, why = _r3_window(payload, cells, cfg)
        n_w1 = pid_n_count(2 * cfg.intermediate_size, int(design["block_n"]))
        if window is None:
            summary["not_asked"].append(f"C1 at G=1: {why}")
        elif window >= n_w1:
            summary["not_asked"].append(
                f"C1 at G=1: the co-residency window {window} is not below "
                f"num_pid_n(w1) = {n_w1}, so a full w1 re-read is not predicted")
        else:
            gates.append(Gate(
                "C1", "CLAIM", "at G=1, w1 is re-read whole at every tread",
                PASS if all(q_sw1[n] >= R3_FULL_REREAD_MIN * n for n in treads) else FAIL,
                ", ".join(f"n={n} {q_sw1[n]:.4f}" for n in treads),
                f"q_S,w1(n) >= {R3_FULL_REREAD_MIN} n, asked because the window "
                f"{window} < num_pid_n(w1) = {n_w1}",
                "the co-residency reading of G=1: w1's M-row outlasts the CTAs in "
                "flight, so no w1 slab survives to the next tile"))

    # C2 w2 NEVER ABOVE THE GROUP MODEL.
    q_sw2 = q["shared"]["w2"]
    gates.append(Gate(
        "C2", "CLAIM", "w2 never reads more than the group model counts",
        PASS if all(q_sw2[n] <= R3_W2_CEILING * gm[n] for n in treads) else FAIL,
        ", ".join(f"n={n} {q_sw2[n]:.4f}/{gm[n]:.4f}" for n in treads),
        f"q_S,w2(n) <= {R3_W2_CEILING} x group_reads at every n",
        "the model's ceiling: co-residency can only remove reads, never add them"))

    # C3 AT G=1, w2 SHARED BY CO-RESIDENT TILES.
    if g_m == 1:
        a1, a2 = est["alpha_slope"]["w1"], est["alpha_slope"]["w2"]
        gates.append(Gate(
            "C3", "CLAIM", "at G=1, w2 is re-read less than w1",
            PASS if a2 < a1 else FAIL, f"alpha_w2 {a2:.4f} against alpha_w1 {a1:.4f}",
            "alpha_w2 < alpha_w1",
            "the co-residency reading of G=1's reuse: w1 has 2F/H = 7x more N-tiles "
            "per M-row than w2, so fewer of an expert's M-tiles are in flight together "
            "on w1"))

    # C6 THE PRIVATE EXCESS.
    q_pt = q["private"]["total"]
    gates.append(Gate(
        "C6", "CLAIM", "PRIVATE reads no more than one weight set per tile",
        PASS if all(q_pt[n] <= (1 + R3_PRIVATE_EXCESS) * n for n in treads) else FAIL,
        ", ".join(f"n={n} {q_pt[n] / n - 1:+.4f}" for n in treads)
        + "  (q_P/n - 1; w1 " + ", ".join(f"{q['private']['w1'][n] / n - 1:+.4f}"
                                          for n in treads)
        + "; w2 " + ", ".join(f"{q['private']['w2'][n] / n - 1:+.4f}" for n in treads)
        + ")",
        f"q_P,total(n) <= {1 + R3_PRIVATE_EXCESS:g} n at every n",
        "a FAIL is a finding: part of the private arm's timed G-cost is BYTES, and "
        "the per-GEMM split localises it (w2 is what the activation-thrash reading "
        "predicts)"))

    # C5 CROSS-CARD, only with --timed-reference.
    ref = (timed or {}).get(g_m)
    mismatch = timed_reference_mismatch(ref, design) if ref is not None else []
    if timed is not None and ref is None:
        summary["not_asked"].append(f"C5: no timed reference page at G={g_m}")
    elif mismatch:
        summary["not_asked"].append(
            f"C5: the timed reports at G={g_m} are another kernel ({'; '.join(mismatch)}), "
            "so their ratio is not this page's to compare with")
    elif ref is not None:
        cross = [f"CROSS-CARD: the timed pages are {ref['cards']}, this page is "
                 f"{(card or {}).get('slug')}; the same sm_90 kernel on 132 SMs where "
                 "both are H100/H200, a different card either way",
                 "timed ratios read from "
                 + "; ".join(f"{r['path']} (seed {r['seed']}, duty {r['duty']:g}, fit "
                             f"window n >= {r['claim_min_tread']})" for r in ref["runs"])
                 + f"; timed at duty {ref['duty']:g}, where this page's bytes carry no "
                 "duty: a timed ratio belongs to its duty's operating point"]
        if g_m == 1:
            got = est["alpha_slope"]["total"]
            lo_edge = ref["mean"] - ref["sd"]
            gates.append(Gate(
                "C5", "CLAIM", "alpha(1) from bytes lies inside the timed bracket",
                PASS if lo_edge <= got <= R3_ALPHA1_CEILING else FAIL,
                f"alpha(1) {got:.4f} against [{ref['mean']:.4f} - sd {ref['sd']:.4f}, "
                "1.0]",
                f"timed mean - sd <= alpha(1) <= {R3_ALPHA1_CEILING}",
                "inside refutes co-residency sharing as the source of G=1 reuse; "
                "below says the timed lower edge's equal-rate assumption fails on "
                "this architecture (the private arm pays a per-copy rate cost)",
                cross))
        elif g_m >= 4:
            got = est["alpha_ratio"]
            edge = ref["mean"] - ref["sd"]
            gates.append(Gate(
                "C5", "CLAIM", "the byte ratio sits below the timed ratio",
                PASS if got is not None and got < edge else FAIL,
                f"byte ratio {got:.4f} against timed {ref['mean']:.4f} (sd "
                f"{ref['sd']:.4f})",
                "slope(R_S)/slope(R_P) < timed mean - its seed sd",
                "finding 4.2's floor reading: a byte ratio equal to the timed ratio "
                "refutes it, and says the timed ratios at G >= 4 are traffic "
                "fractions after all", cross))
        else:
            summary["not_asked"].append(f"C5: no registered C5 at G={g_m}")
    return gates, summary


def _r3_window(payload: dict, cells: dict, cfg) -> tuple[int | None, str]:
    """The co-residency window from the page's own recorded occupancy, or
    `(None, why)` when the four limits were not proven readable."""
    card = payload.get("card") or {}
    sm = card.get("sm_count")
    rec = (cells.get(("shared", 1)) or next(iter(cells.values())))["recorded"].get("w1") or {}
    limits = [rec.get(m) for m in R3_OCCUPANCY_LIMITS]
    if sm is None or any(v is None for v in limits):
        return None, ("the occupancy limits were not proven readable on this box, so "
                      "the co-residency window is unknown and the claim is not asked")
    return coresident_window(int(sm), int(min(limits))), ""


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------

def build_r3_page(*, plan: dict, manifest: dict, cells: list[dict], card, stack: dict,
                  ncu: dict, census: dict) -> dict:
    """The page body, before scoring and stamping."""
    cfg = MODEL_CONFIGS[plan["model"]]
    g_m = int(plan["group_m"])
    page = {"family": R3_FAMILY, "schema": R3_SCHEMA_VERSION, "card": card,
            "stack": stack, "ncu": ncu, "design": r3_design(plan),
            "byte_model": r3_byte_model(cfg, plan["dtype"], int(plan["block_m"])),
            "census": census, "proof": manifest.get("proof"), "cells": cells,
            "group_model": {"model": "group_reads: vLLM v0.27.1's pid mapping, a pure "
                                     "L2 inside a GROUP_SIZE_M group and none across",
                            "q": {str(n): group_reads(cfg.num_experts, int(n), g_m)
                                  for n in plan["treads"]}},
            "estimates": None, "gates": []}
    try:
        page["estimates"] = r3_estimates(page)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        page["estimates"] = {"refused": f"{type(exc).__name__}: {exc}"}
    return page


def check_r3_page(payload: dict) -> None:
    """The WRITE SITE's check against the schema: refuses a page missing a key
    `R3_TOP_KEYS` or `R3_CELL_KEYS` names, before it is written."""
    absent = [k for k in R3_TOP_KEYS if k not in payload]
    if absent:
        raise CounterRunRefused(f"the r3-arms schema names {absent} and this page has "
                                "none; refusing to write a page --analyse would default")
    for c in payload["cells"]:
        gone = [k for k in R3_CELL_KEYS if k not in c]
        if gone:
            raise CounterRunRefused(f"cell {c.get('arm')}/{c.get('n')} is missing {gone}")


def r3_page_lines(payload: dict, gates: list[Gate], summary: dict) -> list[str]:
    """What a human reads: the card first, the per-tread q beside the group
    model, the estimates, and every gate."""
    d = payload["design"]
    out = [card_line(payload.get("card")),
           f"R3 ARMS UNDER A DRAM COUNTER  G={d['group_m']}  {d['model']} {d['dtype']} "
           f"BLOCK_M={d['block_m']} BLOCK_N={d['block_n']} BLOCK_K={d['block_k']} "
           f"num_warps={d['num_warps']} num_stages={d['num_stages']}",
           f"  arms {d['arms']} x treads {d['treads']}; {d['calls_per_cell']} measured "
           f"calls per cell after {d['warmup_calls']} warmups; {d['launch_count']} "
           "profiled launches; declared "
           f"{d['copies_declared']} copies ({d['declared_by_arm']})"]
    est = summary.get("estimates")
    gm = summary.get("group_model")
    if est and gm:
        out += ["", "  q(n) = (R(n) - n x operand) / W, one weight set per unit; the "
                "group model beside it",
                f"  {'n':>3}{'model':>8}{'q_S':>9}{'q_S,w1':>9}{'q_S,w2':>9}"
                f"{'q_N':>9}{'q_P':>9}{'q_P,w1':>9}{'q_P,w2':>9}"]
        for n in sorted(gm):
            s = str(n)
            out.append(f"  {n:>3}{gm[n]:>8.4f}{est['q_S']['total'][s]:>9.4f}"
                       f"{est['q_S']['w1'][s]:>9.4f}{est['q_S']['w2'][s]:>9.4f}"
                       f"{est['q_N']['total'][s]:>9.4f}{est['q_P']['total'][s]:>9.4f}"
                       f"{est['q_P']['w1'][s]:>9.4f}{est['q_P']['w2'][s]:>9.4f}")
        a = est["alpha_slope"]
        out += ["",
                f"  alpha(G={d['group_m']}) = {a['total']:.4f} (w1 {a['w1']:.4f}, w2 "
                f"{a['w2']:.4f}), max relative residual {est['residual']:.2%}: "
                f"{est['residual_note']}",
                f"  alpha_ratio slope(R_S)/slope(R_P) = {est['alpha_ratio']:.4f}; "
                f"alpha_diff 1 - (slope_P - slope_S)/W = {est['alpha_diff']:.4f}; none "
                "uses a bandwidth, a ridge, an intercept or a calibration"]
    for why in summary.get("not_asked") or []:
        out.append(f"  NOT ASKED  {why}")
    out.append("")
    for g in gates:
        out += g.render()
    return out


# --------------------------------------------------------------------------
# --dry-run --family r3-arms: the plan, its predictions and its price.
# --------------------------------------------------------------------------

def r3_group_rows(num_experts: int, treads, groups) -> list[tuple[int, list[float], float]]:
    """`(G, [group_reads at each tread], OLS slope)` for the dry run's table."""
    rows = []
    for g in groups:
        qs = [group_reads(num_experts, int(n), int(g)) for n in treads]
        rows.append((int(g), qs, ols_slope(list(treads), qs)))
    return rows


def r3_cost_s(launch_count: int, per_launch_s: float = R3_COST_PER_LAUNCH_S) -> float:
    """One G's GPU seconds at the registered constants."""
    return sum(R3_COST_S.values()) + launch_count * per_launch_s


def do_dry_run_r3(args) -> int:
    r3 = _r3()
    cfg = MODEL_CONFIGS[args.model]
    bm, treads = args.block_m, list(args.tiles)
    groups = list(R3_GROUPS) + [R3_OPTIONAL_GROUP]
    try:
        plans = {g: r3_plan(model=args.model, dtype=args.dtype, block_m=bm,
                            block_n=args.block_n, num_stages=args.num_stages,
                            group_m=g, treads=treads, kind="measure", arms=r3.ARMS,
                            calls=R3_CALLS_PER_CELL, warmups=R3_WARMUP_CALLS,
                            profile_dir=Path("$R") / f"r3c-g{g}.profiles",
                            stem=f"g{g}") for g in groups}
    except r3.CounterPlanRefused as exc:
        print(f"REFUSE: {exc}")
        return exit_codes.REFUSED
    first = plans[groups[0]]
    byte = r3_byte_model(cfg, args.dtype, bm)
    pinned = r3.pinned_config(args.block_n, groups[0], args.num_stages)
    e = cfg.num_experts
    a = byte["operand_per_tile_w1"] + byte["operand_per_tile_w2"]
    print(f"DRAM COUNTER RUN -- PLAN  family {R3_FAMILY}: R3's three arms under the "
          "counter, one GROUP_SIZE_M per ncu invocation")
    print()
    print("CARD: DECIDED ON THE BOX. The page's card is the live device, read off torch")
    print("  and nvidia-smi; --card is not read. The target is 1x H100 SXM5 (80 GB, 132")
    print("  SMs, 50 MB L2), with an optional A100 40 GB shake-out first. Neither is the")
    print(f"  study's {STUDY_CARD}: every alpha a page prints is the attached card's own, and")
    print("  every page's first line says which card that is.")
    print()
    print(f"  model           {args.model} {args.dtype}  E={e} k={cfg.top_k} "
          f"H={cfg.hidden_size} F={cfg.intermediate_size}")
    print(f"  pinned          BLOCK_SIZE_M={bm} BLOCK_SIZE_N={pinned['BLOCK_SIZE_N']} "
          f"BLOCK_SIZE_K={pinned['BLOCK_SIZE_K']} num_warps={pinned['num_warps']} "
          f"num_stages={pinned['num_stages']} (R3's pinned_config)")
    print(f"  declared        {first['copies_declared']} copies, "
          f"{e * first['copies_declared']} slots: {first['declared_reason']}")
    print(f"  arms            {list(r3.ARMS)}; declared experts "
          f"{ {x: r3.declared_experts(x, e, first['copies_declared']) for x in r3.ARMS} }")
    print(f"  treads          {treads} (n=5 omitted: its q equals q(6) at G=2 and G=4)")
    mem = r3.memory_plan(cfg, args.dtype, dtype_bytes(args.dtype),
                         first["copies_declared"],
                         tokens_for_rows(cfg, max(treads) * bm), None,
                         "decided on the box",
                         flush=(0, "the counter child times nothing, so no flush buffer"),
                         copies_read=max(treads))
    print(f"  memory          R3's memory_plan, no flush buffer: predicted peak "
          f"{mem.predicted_peak_bytes / 1e9:.1f} GB ({mem.copies} copies at "
          f"{mem.per_copy_bytes / 1e9:.4f} GB); fits a card with at least "
          f"{mem.predicted_peak_bytes / mem.headroom / 1e9:.1f} GB free at the "
          f"{mem.headroom:.0%} headroom. The child refuses otherwise.")
    print(f"  G               {list(R3_GROUPS)} in this order, then optionally "
          f"{R3_OPTIONAL_GROUP}")
    print()
    print("THE BYTE MODEL, from its owners; no bandwidth, ridge or calibration anywhere.")
    print(f"  W   = {byte['W'] / 1e9:.4f} GB  (W_w1 {byte['W_w1'] / 1e9:.4f}, W_w2 "
          f"{byte['W_w2'] / 1e9:.4f})  routed_expert_weight_bytes_by_gemm")
    print(f"  a   = {a / 1e6:.3f} MB per tread (w1 {byte['operand_per_tile_w1'] / 1e6:.3f}, "
          f"w2 {byte['operand_per_tile_w2'] / 1e6:.3f}), {a / byte['W']:.2%} of W: the "
          "GEMMs' A-operand reads, charged cold")
    print("  q(n) = (R(n) - n a) / W, R = dram__bytes_read.sum per fused_experts call "
          "(w1 launch + w2 launch)")
    print()
    print("THE GROUP MODEL, registered before anything runs: q(n) = group_reads(E, n, G),")
    print("  from vLLM v0.27.1's pid mapping, a pure L2 inside a GROUP_SIZE_M group and")
    print("  none across groups. Card-free. PRIVATE reads q(n) = n at every G by")
    print("  construction; SHARED and NATIVE are predicted to read these:")
    print("  " + f"{'G':>4}" + "".join(f"{f'n={n}':>9}" for n in treads) + f"{'slope':>10}")
    for g, qs, slope in r3_group_rows(e, treads, groups):
        print(f"  G={g:<2}" + "".join(f"{v:>9.4f}" for v in qs) + f"{slope:>10.4f}")
    ratios = "  ".join(f"G={g} {(slope * byte['W'] + a) / (byte['W'] + a):.3f}"
                       for g, _qs, slope in r3_group_rows(e, treads, groups))
    print(f"  the byte ratio slope(R_S)/slope(R_P) this predicts, (slope W + a)/(W + a): "
          f"{ratios}")
    print("  At G=4 and G=16 the ladder DROPS at n=4, where the tiles align with the")
    print("  groups: no per-tile scalar alpha produces a drop, which is why n=4 is on")
    print("  the ladder and why the ladder family's monotone and affine gates are NOT")
    print("  applied to SHARED here.")
    print(f"  w1 has 2F/BN = {pid_n_count(2 * cfg.intermediate_size, args.block_n)} N-tiles "
          "per M-row, so the model binds w1 at every G >= 2; w2 has "
          f"{pid_n_count(cfg.hidden_size, args.block_n)} and may read LESS where the")
    print("  co-residency window (SMs x CTAs per SM, the page's own occupancy) exceeds")
    print("  G x that; C2 holds w2 under the model, never at it.")
    print()
    print("PREDICTIONS IN BYTES per call, GB: FULL RE-READ R = n (W + a) against the")
    print("  GROUP MODEL R = q W + n a.")
    for g, qs, _slope in r3_group_rows(e, treads, groups):
        full = [n * (byte["W"] + a) / 1e9 for n in treads]
        grp = [(qv * byte["W"] + n * a) / 1e9 for n, qv in zip(treads, qs, strict=True)]
        print(f"  G={g:<2} full   " + " ".join(f"{v:8.3f}" for v in full))
        print("        group  " + " ".join(f"{v:8.3f}" for v in grp))
    print()
    print("GATES, registered. VALIDITY fails exit INVALID and no alpha may be quoted:")
    print("  V0 live card block; V1 exact launch count, every grid, census "
          f"GEMMS_PER_CALL = {r3.GEMMS_PER_CALL}; V2 every STRICT metric a number;")
    print(f"  V3 K calls within {R3_REPEAT_TOL:.0%}; V4 at n=1 |R_S/R_P - 1| <= "
          f"max({R3_IDENTITY_FLOOR:.1%}, {R3_SPREAD_FACTOR:g} x spread) and q(1) in "
          f"{list(R3_Q1_BAND)};")
    print(f"  V5 {R3_PRIVATE_BAND[0]} n <= q_P(n) <= {R3_PRIVATE_BAND[1]} n per GEMM; V6 "
          f"requested L2 sectors equal across arms within {R3_REQUEST_TOL:.1%}; V7 NATIVE "
          f"= SHARED within max({R3_DECLARATION_FLOOR:.0%}, {R3_SPREAD_FACTOR:g} x spread);")
    print(f"  V8 DRAM bytes = 32 x L2 fill sectors within {R3_COUNTER_TOL:.0%}, asked only "
          "if proven; V9 R3's five-part buffer proof.")
    print("  The ladder family's monotone and affine gates are NOT applied to SHARED.")
    print(f"CLAIMS, a failure is a result: C1 w1 within {R3_GROUP_TOL:.0%} of the group "
          "model at every n for G >= 2; at G=1 q_S,w1(n) >= "
          f"{R3_FULL_REREAD_MIN} n, asked only when the page's")
    print("  co-residency window is below num_pid_n(w1), and NOT ASKED when the occupancy "
          f"limits were not proven readable; C2 w2 <= {R3_W2_CEILING} x the model;")
    print(f"  C3 at G=1 alpha_w2 < alpha_w1; C6 q_P,total(n) <= {1 + R3_PRIVATE_EXCESS:g} n; "
          "C5 only with --timed-reference, labelled cross-card.")
    print()
    print("METRICS, three classes. The box asks only what its probe proved readable.")
    print(f"  STRICT, refused without:   {', '.join(R3_STRICT_METRICS)}")
    print(f"  CROSS-CHECK, gated if proven: {', '.join(R3_CROSSCHECK_METRICS)}")
    print(f"  RECORDED, never gated:     {', '.join(R3_RECORDED_METRICS)}")
    print()
    sched = r3.counter_schedule(first)
    cells = len(first["cells"])
    print("LAUNCH ARITHMETIC, from counter_schedule, the rule the child runs:")
    print(f"  {cells} cells ({len(r3.ARMS)} arms x {len(treads)} treads); U = "
          f"{R3_WARMUP_CALLS} warmups per cell first (they compile), then K = "
          f"{R3_CALLS_PER_CELL} calls per cell")
    print(f"  GEMMS_PER_CALL = {r3.GEMMS_PER_CALL} (cited from vLLM v0.27.1, measured by "
          "the census)")
    print(f"  --launch-skip {sched.launch_skip} = {r3.GEMMS_PER_CALL} x {R3_WARMUP_CALLS} x "
          f"{cells};  --launch-count {sched.launch_count} = {r3.GEMMS_PER_CALL} x "
          f"{R3_CALLS_PER_CELL} x {cells}")
    print("  grids PREDICTED from vLLM's documented sorted-id buffer (numel + declared x "
          "(BM - 1)); the child reads the real one and every launch is held to it:")
    for arm in r3.ARMS:
        declared = r3.declared_experts(arm, e, first["copies_declared"])
        grids = []
        for n in treads:
            tokens = tokens_for_rows(cfg, n * bm)
            em = r3.predicted_sorted_ids(tokens * cfg.top_k, declared, bm)
            gr = r3.expected_grids(cfg, em=em, tokens=tokens, block_m=bm,
                                   block_n=args.block_n)
            grids.append(f"n={n} {gr['w1']}/{gr['w2']}")
        print(f"    {arm:<8} " + "  ".join(grids))
    print()
    print("THE COMMANDS, on a box where --probe --family r3-arms says OPEN, from the vLLM")
    print("  venv so the child is sys.executable ($S the session directory, $R results):")
    print(f"  python scripts/dram_counter_route.py --probe --family {R3_FAMILY} "
          "--out $S/probe.json")
    print(f"  $PY_VLLM scripts/dram_counter_route.py --run --family {R3_FAMILY} "
          "--census-only --out $S/census.json")
    for g in groups:
        tag = "" if g != R3_OPTIONAL_GROUP else "   # optional"
        print(f"  $PY_VLLM scripts/dram_counter_route.py --run --family {R3_FAMILY} "
              f"--group-m {g} --census $S/census.json --out $R/r3c-g{g}.json{tag}")
    print("  python scripts/dram_counter_route.py --analyse "
          + " ".join(f"$R/r3c-g{g}.json" for g in R3_GROUPS)
          + " [--timed-reference <R3 report.json ...>] [--out summary.json]")
    print("  a page refused on the parser is rebuilt off the box after the fix, from the")
    print("  kept profiles, with no card and no child:")
    print(f"  python scripts/dram_counter_route.py --run --family {R3_FAMILY} --reduce-only "
          "--group-m G --census $S/census.json --out $R/r3c-gG.json")
    print("  publish each box's pages, census and summary under")
    print("  results/published/<date>-<slug>-r3-counters/, <slug> being the one every "
          "page's CARD line names")
    print("  each --run profiles through this argv, and reduces the kept .ncu-rep with")
    print("  ncu --import <rep> --csv --page raw --print-units base > <csv>:")
    for g in groups:
        p = plans[g]
        s = r3.counter_schedule(p)
        prof = Path(p["manifest"]).parent
        argv = r3_ncu_argv("ncu", prof / f"g{g}.plan.json", prof / f"g{g}.ncu-rep",
                           R3_ALL_METRICS, launch_skip=s.launch_skip,
                           launch_count=s.launch_count, python="$PY_VLLM",
                           child=R3_CHILD.relative_to(REPO))
        print(f"  G={g}: " + " ".join(argv))
    print()
    per_g = r3_cost_s(sched.launch_count)
    four = len(R3_GROUPS) * per_g
    print("COST, at the registered constants (R3_COST_S, per launch "
          f"{R3_COST_PER_LAUNCH_S:g} s); the first G's measured per-launch time re-prices")
    print("  the rest.")
    print(f"  per G: child start {R3_COST_S['child_start']:.0f} s + weight build "
          f"{R3_COST_S['weight_build']:.0f} s + compiles {R3_COST_S['compile']:.0f} s + "
          f"{sched.launch_count} profiled launches x {R3_COST_PER_LAUNCH_S:g} s + proof "
          f"and reduce {R3_COST_S['proof_and_reduce']:.0f} s = {per_g / 60:.1f} min")
    print(f"  {len(R3_GROUPS)} G: {four / 60:.0f} min; with G={R3_OPTIONAL_GROUP}: "
          f"{(four + per_g) / 60:.0f} min; preflight (metric query, probe, census) "
          f"{R3_COST_PREFLIGHT_S / 60:.0f} min")
    pess = r3_cost_s(sched.launch_count, R3_COST_PER_LAUNCH_PESSIMISTIC_S)
    print(f"  at a pessimistic {R3_COST_PER_LAUNCH_PESSIMISTIC_S:g} s per launch: "
          f"{len(R3_GROUPS)} G {len(R3_GROUPS) * pess / 60:.0f} min, with "
          f"G={R3_OPTIONAL_GROUP} {(len(R3_GROUPS) + 1) * pess / 60:.0f} min")
    print(f"  book {R3_VM_BOOKING_H:g} h of VM: venv downloads, measurement, one parser "
          "or door debug loop, and exfiltration")
    print("  an A100 40 GB shake-out at G in {1, 64}: ncu's first-pass save of the ~26 GB")
    print("  footprint goes to HOST memory there (device free < footprint), about 1 s per")
    print("  launch over PCIe, and needs ~30 GB of host RAM")
    print()
    print("SCHEMA each page is written in, so --analyse can score it:")
    print(R3_SCHEMA_TEXT)
    return exit_codes.DONE


# --------------------------------------------------------------------------
# --run --family r3-arms: the census and one G's page.
# --------------------------------------------------------------------------

def r3_commit() -> tuple[str | None, str]:
    """`(sha, "")` for the tree this file runs from, or `(None, why)`.

    A CENSUS LICENSES A PAGE BY COMMIT, and `None == None` is not a match.
    `provenance._git` returns no sha on any git failure, and the expected one
    on a counter box is git refusing to run as root in a repository another
    user owns ("detected dubious ownership"), which is exactly what the sudo
    counter door (`sudo -E env PATH=... HOME=...`) does. Until 2026-09-24 the
    census stored `commit: None`, the page compared `None != None` as a match,
    and `--analyse` joined pages over the commit set `{'None'}`: a page could be
    written and joined with no commit at all. Every r3-arms write site now
    asks here and refuses without a sha, printing provenance's own reason and
    the remedy for the root-in-a-user-repository case.
    """
    prov = PV.provenance_block(instrument=R3_RUN_INSTRUMENT)
    if prov.git_sha:
        return prov.git_sha, ""
    reason = prov.missing.get("git_sha") or "provenance named no git sha and no reason"
    remedy = ("run from a git checkout of this repository" if "dubious" not in reason else
              "git refuses to run as this user in a repository another user owns, which "
              "is what the sudo counter door does: add the checkout to safe.directory in "
              "the gitconfig the launcher's HOME points at (as the login user, `git "
              f"config --global --add safe.directory {REPO}`), then run again")
    return None, (f"this tree has no commit git could name ({reason}), and every "
                  f"r3-arms census and page is matched by commit: {remedy}")


def _r3_capture(argv: list[str], log_path: Path, timeout: float) -> tuple[int, str]:
    rc, out, err = _run(argv, timeout=timeout)
    log_path.write_text((out or "") + (err or ""))
    return rc, ((err or out or "").strip()[-600:])


def _r3_reduce(binary: str, report: Path, csv_path: Path) -> str:
    rc, out, err = _run(r3_import_argv(binary, report), timeout=600)
    csv_path.write_text(out or "")
    if rc != 0 or not out:
        raise CounterRunRefused(
            f"ncu --import {report.name} exited {rc} and printed no CSV ({(err or '')[:300]}); "
            "the .ncu-rep is kept, so the reduction can be redone off the box")
    return out


def do_run_r3(args) -> int:
    """The census or one G's page. REFUSED before the child runs when the route
    is not open for this family, when there is no card, when git cannot name
    this tree's commit (`r3_commit`), when a page is asked for without its G
    or its census, and when the census is another card's, another commit's or
    another vLLM's, or names no commit."""
    if not args.out:
        print("REFUSE: --run needs --out <path>; a measurement nobody wrote down is "
              "not a measurement.")
        return exit_codes.REFUSED
    if args.census_only and args.census:
        print("REFUSE: --census-only writes a census and reads none; drop --census")
        return exit_codes.REFUSED
    if not args.census_only:
        if not getattr(args, "group_m_given", False):
            print(f"REFUSE: a page is one G, and --group-m names it: the registered G "
                  f"are {list(R3_GROUPS)} in that order, then optionally "
                  f"{R3_OPTIONAL_GROUP}. It is not defaulted.")
            return exit_codes.REFUSED
        if not args.census:
            print("REFUSE: a page needs --census <census.json>, the census THIS card, "
                  "commit and vLLM wrote. Run --census-only first.")
            return exit_codes.REFUSED
    ncu = probe_ncu(R3_FAMILY)
    if not counter_route_is_open(ncu):
        print("REFUSE: the r3-arms family's counters could not be read on this box. "
              f"--probe --family {R3_FAMILY} says: {ncu.get('cause', ncu.get('why'))}")
        return exit_codes.REFUSED
    card = live_card_block()
    if card is None or not card.get("uuid"):
        print("REFUSE: no card, or a card whose UUID could not be read; every page "
              "names its card and a census is matched to it by UUID")
        return exit_codes.REFUSED
    stack = r3_stack_versions()
    commit, no_commit = r3_commit()
    if commit is None:
        print(f"REFUSE: {no_commit}")
        return exit_codes.REFUSED
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profiles = Path(args.profile_dir) if args.profile_dir else \
        out.parent / f"{out.stem}.profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    metrics = tuple(ncu.get("metrics_proven") or R3_STRICT_METRICS)
    r3 = _r3()
    if args.census_only:
        return r3_census(args, ncu, card, stack, commit, out, profiles)

    census_path = Path(args.census)
    try:
        census = json.loads(census_path.read_text())
    except (OSError, ValueError) as exc:
        print(f"REFUSE: cannot read the census {census_path}: {exc}")
        return exit_codes.REFUSED
    why = []
    if census.get("family") != R3_FAMILY or census.get("kind") != "census":
        why.append("it is not an r3-arms census")
    if (census.get("card") or {}).get("uuid") != card["uuid"]:
        why.append(f"card {(census.get('card') or {}).get('uuid')} is not this card "
                   f"{card['uuid']}")
    if not census.get("commit"):
        why.append("it names no commit, and a census licenses a page by commit")
    elif census.get("commit") != commit:
        why.append(f"commit {census.get('commit')} is not this tree's {commit}")
    if (census.get("stack") or {}).get("vllm") != stack.get("vllm"):
        why.append(f"vLLM {(census.get('stack') or {}).get('vllm')} is not "
                   f"{stack.get('vllm')}")
    if census.get("gemms_per_call_measured") != r3.GEMMS_PER_CALL \
            or census.get("verdict") != PASS:
        why.append(f"it measured {census.get('gemms_per_call_measured')} GEMMs per call "
                   f"with verdict {census.get('verdict')}")
    if why:
        print(f"REFUSE: the census {census_path} cannot license this page: "
              + "; ".join(why))
        return exit_codes.REFUSED

    g_m, stem = args.group_m, f"g{args.group_m}"
    try:
        plan = r3_plan(model=args.model, dtype=args.dtype, block_m=args.block_m,
                       block_n=args.block_n, num_stages=args.num_stages, group_m=g_m,
                       treads=args.tiles, kind="measure", arms=r3.ARMS,
                       calls=R3_CALLS_PER_CELL, warmups=R3_WARMUP_CALLS,
                       profile_dir=profiles, stem=stem)
    except r3.CounterPlanRefused as exc:
        print(f"REFUSE: {exc}")
        return exit_codes.REFUSED
    sched = r3.counter_schedule(plan)
    plan_path = profiles / f"{stem}.plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    report = profiles / f"{stem}.ncu-rep"
    csv_path = profiles / f"{stem}.csv"
    argv = r3_ncu_argv(ncu["binary"], plan_path, report, metrics,
                       launch_skip=sched.launch_skip, launch_count=sched.launch_count)
    print(card_line(card))
    print(f"R3 COUNTER RUN  G={g_m}  {len(plan['cells'])} cells, {sched.launch_count} "
          f"profiled launches after {sched.launch_skip} skipped")
    print(f"  ncu        {ncu.get('binary')}  [{ncu.get('version', '')}]")
    print(f"  metrics    {len(metrics)} asked: {', '.join(metrics)}")
    print(f"  profiles   {profiles}")
    rc, tail = _r3_capture(argv, profiles / f"{stem}.ncu.log", args.ncu_timeout)
    manifest_path = Path(plan["manifest"])
    absent = [str(p) for p in (manifest_path, report) if not p.exists()]
    if absent:
        raise CounterRunRefused(
            f"ncu exited {rc} and the capture left no {absent}; the child writes the "
            f"manifest only after its last call. ncu's log ends: {tail}")
    # THE CAPTURE'S OWN RECORD, written before anything is parsed: the page a
    # later `--reduce-only` rebuilds from these profiles names the card, the
    # stack, the commit and the argv of THIS capture, not the reducing tree's.
    capture = {"argv": argv, "binary": ncu.get("binary"), "version": ncu.get("version"),
               "returncode": rc, "metrics_asked": list(metrics),
               "metrics_dropped": sorted(set(ncu.get("metrics_dropped") or [])
                                         | set(ncu.get("metrics_unproven") or {})),
               "card": card, "stack": stack, "commit": commit}
    capture_path = profiles / f"{stem}.capture.json"
    capture_path.write_text(json.dumps(capture, indent=2))
    csv_text = _r3_reduce(ncu["binary"], report, csv_path)
    return _r3_write_page(args, plan=plan, capture=capture, census_path=census_path,
                          census=census, report=report, csv_path=csv_path,
                          csv_text=csv_text, out=out)


def _r3_write_page(args, *, plan: dict, capture: dict, census_path: Path, census: dict,
                   report: Path, csv_path: Path, csv_text: str, out: Path) -> int:
    """Parse, attribute, reduce, score, stamp and write one G's page. The one
    path both `--run` and `--reduce-only` take after the capture."""
    manifest = json.loads(Path(plan["manifest"]).read_text())
    card = capture["card"]
    if (manifest.get("device") or {}).get("uuid") != card["uuid"]:
        raise CounterRunRefused(
            f"the child ran on {(manifest.get('device') or {}).get('uuid')} and this "
            f"page's card is {card['uuid']}")
    metrics = capture["metrics_asked"]
    launches = parse_ncu_csv(csv_text, soft=frozenset(R3_RECORDED_METRICS))
    cells = r3_reduce_cells(attribute_launches(launches, manifest), manifest, metrics)
    layout, _header = ncu_csv_layout(csv_text)
    page = build_r3_page(
        plan=plan, manifest=manifest, cells=cells, card=card, stack=capture["stack"],
        ncu={"binary": capture.get("binary"), "version": capture.get("version"),
             "argv": capture["argv"], "replay_mode": "kernel", "cache_control": "all",
             "clock_control": "base", "report": str(report),
             "report_sha256": _sha256(report), "csv": str(csv_path),
             "csv_layout": layout, "metrics_asked": list(metrics),
             "metrics_dropped": list(capture.get("metrics_dropped") or []),
             "capture_commit": capture.get("commit")},
        census={"path": str(census_path), "sha256": _sha256(census_path),
                "gemms_per_call": census.get("gemms_per_call_measured")})
    gates, summary = score_r3_page(page)
    page["gates"] = [asdict(g) for g in gates]
    payload = stamped(page, mode="r3-run", args=args, card=card["name"],
                      instrument=R3_RUN_INSTRUMENT)
    check_r3_page(payload)
    out.write_text(json.dumps(payload, indent=2))
    for line in r3_page_lines(payload, gates, summary):
        print(line)
    print(f"wrote {out}")
    print(f"git   {git_visibility(out)}")
    return exit_codes.classify(g.scored() for g in gates)


def do_reduce_r3(args) -> int:
    """`--run --family r3-arms --reduce-only`: rebuild one G's page from the
    profiles a capture left, with no card, no probe and no child.

    THE CAPTURE AND THE REDUCTION ARE SPLIT so that a parser defect found on
    the box costs a laptop fix and not a re-rent: the `.ncu-rep`, the plan, the
    manifest and the capture's own record (`g<G>.capture.json`) are kept under
    `<out>.profiles/`. With an ncu on PATH the CSV is regenerated from the
    `.ncu-rep`; without one the CSV the capture reduced is read. The card, the
    stack, the argv and the commit on the page are the CAPTURE's, from its
    record, and the reducing tree's commit is the provenance stamp beside them.
    """
    if not args.out or not getattr(args, "group_m_given", False) or not args.census:
        print("REFUSE: --reduce-only rebuilds one G's page: it needs --group-m, the "
              "--census the capture was licensed by, and the --out whose .profiles "
              "directory the capture wrote")
        return exit_codes.REFUSED
    out = Path(args.out)
    profiles = Path(args.profile_dir) if args.profile_dir else \
        out.parent / f"{out.stem}.profiles"
    stem = f"g{args.group_m}"
    need = {name: profiles / f"{stem}.{name}" for name in
            ("plan.json", "capture.json", "manifest.json")}
    absent = [str(p) for p in need.values() if not p.exists()]
    if absent:
        print(f"REFUSE: no capture to reduce: {absent} do not exist")
        return exit_codes.REFUSED
    plan = json.loads(need["plan.json"].read_text())
    capture = json.loads(need["capture.json"].read_text())
    if not capture.get("commit"):
        print(f"REFUSE: the capture {need['capture.json']} names no commit, so the page "
              "it would rebuild could be joined with nothing")
        return exit_codes.REFUSED
    here, no_commit = r3_commit()
    if here is None:
        print(f"REFUSE: {no_commit}")
        return exit_codes.REFUSED
    census_path = Path(args.census)
    census = json.loads(census_path.read_text())
    if (census.get("card") or {}).get("uuid") != (capture.get("card") or {}).get("uuid"):
        print(f"REFUSE: the census is card {(census.get('card') or {}).get('uuid')} and "
              f"the capture is card {(capture.get('card') or {}).get('uuid')}")
        return exit_codes.REFUSED
    report, csv_path = profiles / f"{stem}.ncu-rep", profiles / f"{stem}.csv"
    binary = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if binary and report.exists():
        csv_text = _r3_reduce(binary, report, csv_path)
    elif csv_path.exists() and csv_path.read_text().strip():
        csv_text = csv_path.read_text()
    else:
        print(f"REFUSE: no ncu on PATH to import {report} and no CSV at {csv_path}")
        return exit_codes.REFUSED
    print(card_line(capture.get("card")))
    print(f"R3 COUNTER REDUCTION  G={args.group_m}  from {profiles}, captured at "
          f"commit {capture.get('commit')}; nothing is measured here")
    return _r3_write_page(args, plan=plan, capture=capture, census_path=census_path,
                          census=census, report=report, csv_path=csv_path,
                          csv_text=csv_text, out=out)


def r3_census(args, ncu: dict, card: dict, stack: dict, commit, out: Path,
              profiles: Path) -> int:
    """PROVE GEMMS_PER_CALL AND THE GRIDS BEFORE ANY PAGE. A mini plan (NATIVE
    at n in {1, 6}, one warmup, one call) under the same kernel filter with no
    skip and no cap: the report must hold exactly GEMMS_PER_CALL x 4 launches
    and their grids must equal the child's vLLM-derived ones. The child
    builds R3's whole declaration, so its memory plan is checked too. A census
    with no commit licenses nothing, so none is written (`r3_commit`)."""
    if not commit:
        print(f"REFUSE: {r3_commit()[1]}")
        return exit_codes.REFUSED
    r3 = _r3()
    g_m = args.group_m if getattr(args, "group_m_given", False) else R3_GROUPS[0]
    try:
        plan = r3_plan(model=args.model, dtype=args.dtype, block_m=args.block_m,
                       block_n=args.block_n, num_stages=args.num_stages, group_m=g_m,
                       treads=R3_CENSUS_TREADS, kind="census", arms=(r3.NATIVE,),
                       calls=1, warmups=1, profile_dir=profiles, stem="census")
    except r3.CounterPlanRefused as exc:
        print(f"REFUSE: {exc}")
        return exit_codes.REFUSED
    sched = r3.counter_schedule(plan)
    plan_path = profiles / "census.plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    report = profiles / "census.ncu-rep"
    argv = r3_ncu_argv(ncu["binary"], plan_path, report, R3_STRICT_METRICS,
                       launch_skip=sched.launch_skip, launch_count=sched.launch_count)
    print(card_line(card))
    print(f"R3 COUNTER CENSUS  {len(plan['cells'])} cells, "
          f"{len(sched.warmups) + len(sched.measured)} calls, no skip and no cap")
    rc, tail = _r3_capture(argv, profiles / "census.ncu.log", args.ncu_timeout)
    manifest_path = Path(plan["manifest"])
    if not manifest_path.exists() or not report.exists():
        raise CounterRunRefused(f"ncu exited {rc} and the census child left no manifest "
                                f"or no report: {tail}")
    manifest = json.loads(manifest_path.read_text())
    launches = _launch_order(parse_ncu_csv(
        _r3_reduce(ncu["binary"], report, profiles / "census.csv")))
    seq = r3_launch_sequence(manifest)
    calls = len(sched.warmups) + len(sched.measured)
    seen = [int(round(ln.metrics.get("launch__grid_size", -1))) for ln in launches]
    want = [int(manifest["grids"][key][g]) for key, _c, g, _w in seq]
    per_call = len(launches) / calls
    gates = [
        Gate("CEN1", "VALIDITY", f"one fused_experts call makes exactly "
             f"{r3.GEMMS_PER_CALL} fused_moe_kernel launches",
             PASS if len(launches) == r3.GEMMS_PER_CALL * calls else FAIL,
             f"{len(launches)} launches over {calls} calls = {per_call:g} per call",
             f"{r3.GEMMS_PER_CALL} x {calls} = {r3.GEMMS_PER_CALL * calls}",
             "every page's skip and count, which are GEMMS_PER_CALL multiples"),
        Gate("CEN2", "VALIDITY", "every launch's grid is the grid the child derived "
             "from vLLM's own sorted-id buffer", PASS if seen == want else FAIL,
             f"seen {seen}", f"want {want}",
             "the attribution every page makes, launch by launch"),
        Gate("CEN3", "VALIDITY", "the child ran on this card",
             PASS if (manifest.get("device") or {}).get("uuid") == card["uuid"] else FAIL,
             str((manifest.get("device") or {}).get("uuid")), card["uuid"],
             "the census's card, which every page is matched to"),
    ]
    rc_all = exit_codes.classify(g.scored() for g in gates)
    body = {"family": R3_FAMILY, "kind": "census", "card": card, "stack": stack,
            "commit": commit, "plan": plan, "calls": calls, "launches": len(launches),
            "gemms_per_call_measured": (int(per_call) if per_call == int(per_call)
                                        else per_call),
            "grids_expected": want, "grids_seen": seen,
            "memory_plan": manifest.get("memory_plan"),
            "ncu": {"binary": ncu.get("binary"), "version": ncu.get("version"),
                    "argv": argv, "report": str(report),
                    "report_sha256": _sha256(report)},
            "gates": [asdict(g) for g in gates],
            "verdict": PASS if rc_all == exit_codes.DONE else FAIL}
    payload = stamped(body, mode="r3-census", args=args, card=card["name"],
                      instrument=R3_CENSUS_INSTRUMENT)
    out.write_text(json.dumps(payload, indent=2))
    for g in gates:
        for line in g.render():
            print(line)
    print(f"wrote {out}")
    print(f"git   {git_visibility(out)}")
    return rc_all


# --------------------------------------------------------------------------
# --analyse over r3-arms pages: one page's gates, or the alpha(G) table.
# --------------------------------------------------------------------------

def _page_commit(d: dict):
    return d.get("git_sha") or (d.get("provenance") or {}).get("git_sha")


def do_analyse_r3(args, loaded: list[tuple[Path, dict]]) -> int:
    """Score each page; with several, print the alpha(G) table and REFUSE a
    join across two card UUIDs, two commits, two vLLM versions or two designs
    (the design with its G taken out)."""
    families = {d.get("family") for _p, d in loaded}
    if families != {R3_FAMILY}:
        print(f"REFUSED: --analyse was given {sorted(map(str, families))}; an r3-arms "
              "summary joins r3-arms pages only")
        return exit_codes.REFUSED
    if len(loaded) > 1:
        bare = [str(p) for p, d in loaded if not _page_commit(d)]
        if bare:
            print(f"REFUSED: {bare} name no commit; an alpha(G) table joins pages of one "
                  "commit, and pages that name none cannot show they share one")
            return exit_codes.REFUSED
        apparatus = {
            "card UUID": lambda d: (d.get("card") or {}).get("uuid"),
            "commit": _page_commit,
            "vLLM": lambda d: (d.get("stack") or {}).get("vllm"),
            "design": lambda d: json.dumps({k: v for k, v in d["design"].items()
                                            if k != "group_m"}, sort_keys=True),
        }
        for name, read in apparatus.items():
            values = sorted({str(read(d)) for _p, d in loaded})
            if len(values) > 1:
                print(f"REFUSED: the pages carry {len(values)} values of {name} "
                      f"({values}); an alpha(G) table across them compares two "
                      "apparatuses, not two G")
                return exit_codes.REFUSED
        gs = [int(d["design"]["group_m"]) for _p, d in loaded]
        if len(set(gs)) != len(gs):
            print(f"REFUSED: two pages at one G ({sorted(gs)}); the table has one row per G")
            return exit_codes.REFUSED
    try:
        timed = (load_timed_reference(args.timed_reference)
                 if getattr(args, "timed_reference", None) else None)
    except (OSError, ValueError, KeyError, CounterRunRefused) as exc:
        print(f"REFUSED: --timed-reference: {exc}")
        return exit_codes.REFUSED
    for g_ref, ref in sorted((timed or {}).items()):
        mismatch = timed_reference_mismatch(ref, loaded[0][1]["design"])
        if mismatch:
            print(f"REFUSED: --timed-reference: the reports at G={g_ref} timed another "
                  f"kernel than these pages measured ({'; '.join(mismatch)}); C5 compares "
                  "one kernel configuration's bytes with its own timing")
            return exit_codes.REFUSED
    loaded = sorted(loaded, key=lambda pd: int(pd[1]["design"]["group_m"]))
    scored = []
    for path, d in loaded:
        gates, summary = score_r3_page(d, timed=timed)
        scored.append((path, d, gates, summary))
    card = loaded[0][1].get("card")
    all_gates: list[Gate] = []
    if len(scored) == 1:
        path, d, gates, summary = scored[0]
        for line in r3_page_lines(d, gates, summary):
            print(line)
        all_gates = gates
    else:
        print(card_line(card))
        print(f"ALPHA(G) OVER {len(scored)} PAGES, one card, one commit, one vLLM, one "
              "design")
        print(f"  {'G':>4}{'alpha':>9}{'w1':>9}{'w2':>9}{'resid':>9}{'ratio':>9}"
              f"{'diff':>9}  exit")
        for _path, d, gates, summary in scored:
            est = summary.get("estimates")
            rc = exit_codes.classify(g.scored() for g in gates)
            if est:
                a = est["alpha_slope"]
                ratio = est["alpha_ratio"]
                print(f"  {d['design']['group_m']:>4}{a['total']:>9.4f}{a['w1']:>9.4f}"
                      f"{a['w2']:>9.4f}{est['residual']:>9.2%}"
                      + (f"{ratio:>9.4f}" if ratio is not None else f"{'none':>9}")
                      + f"{est['alpha_diff']:>9.4f}  {exit_codes.describe(rc)}")
            else:
                print(f"  {d['design']['group_m']:>4}  no estimate: "
                      f"{summary.get('reason')}  {exit_codes.describe(rc)}")
        print()
        print("  q per (G, arm, GEMM) beside the group model:")
        for _path, d, _gates, summary in scored:
            est, gm = summary.get("estimates"), summary.get("group_model")
            if not est or not gm:
                continue
            for label, key in (("shared", "q_S"), ("native", "q_N"), ("private", "q_P")):
                for part in ("total", "w1", "w2"):
                    vals = est[key][part]
                    print(f"  G={d['design']['group_m']:<3}{label:<8}{part:<6}"
                          + " ".join(f"{vals[str(n)]:8.4f}" for n in sorted(gm)))
            print(f"  G={d['design']['group_m']:<3}{'model':<14}"
                  + " ".join(f"{gm[n]:8.4f}" for n in sorted(gm)))
        print()
        for path, d, gates, summary in scored:
            tag = f"g{d['design']['group_m']}"
            print(f"PAGE {path}  G={d['design']['group_m']}")
            for why in summary.get("not_asked") or []:
                print(f"  NOT ASKED  {why}")
            for g in gates:
                renamed = Gate(f"{tag}.{g.number}", g.kind, g.claim, g.verdict,
                               g.measured, g.threshold, g.invalidates, g.lines)
                for line in renamed.render():
                    print(line)
                all_gates.append(renamed)
            print()
    if args.out:
        out = Path(args.out)
        body = {"family": R3_FAMILY, "kind": "summary", "card": card,
                "commit": _page_commit(loaded[0][1]),
                "pages": [{"path": str(p), "group_m": d["design"]["group_m"],
                           "exit": exit_codes.classify(g.scored() for g in gates),
                           "estimates": s.get("estimates"),
                           "group_model": {str(n): v for n, v in
                                           (s.get("group_model") or {}).items()},
                           "not_asked": s.get("not_asked")}
                          for p, d, gates, s in scored],
                "timed_reference": timed,
                "gates": [asdict(g) for g in all_gates]}
        out.write_text(json.dumps(stamped(
            body, mode="analyse", args=args,
            card=(card or {}).get("name") or NO_CARD,
            instrument=R3_ANALYSE_INSTRUMENT), indent=2))
        print(f"wrote {out}")
        print(f"git   {git_visibility(out)}")
    return exit_codes.classify(g.scored() for g in all_gates)


def do_analyse_any(args) -> int:
    """`--analyse` over one or more payloads, dispatched on the `family` key.
    One ladder payload is scored exactly as it always was; several ladder
    payloads are refused (their ratio is --contrast's); r3-arms pages go to
    `do_analyse_r3`."""
    paths = [Path(p) for p in args.analyse]
    if len(paths) == 1 and not paths[0].exists():
        return do_analyse(args)
    loaded = []
    for p in paths:
        if not p.exists():
            print(f"REFUSED: no such payload: {p}")
            return exit_codes.REFUSED
        loaded.append((p, json.loads(p.read_text())))
    if any(d.get("family") == R3_FAMILY for _p, d in loaded):
        return do_analyse_r3(args, loaded)
    if len(loaded) > 1:
        print("REFUSED: several ladder-family payloads; --analyse scores one, and the "
              "ratio across ladder runs is --contrast's")
        return exit_codes.REFUSED
    return do_analyse(args)


# --------------------------------------------------------------------------
# The family, planted: a manifest, ncu CSV and a page, off any GPU, pushed
# through the SAME reduction and scorer `--run` uses.
# --------------------------------------------------------------------------

#: A card no nvidia-smi can print, for planted pages only. Its SM count and
#: occupancy make the co-residency window 132 x 1 = 132, below w1's 448
#: N-tiles, so C1 is asked at G=1.
R3_PLANTED_CARD: dict = {
    "name": "PLANTED-CARD-not-a-device", "slug": "planted_card_not_a_device",
    "uuid": "planted-0000", "sm_count": 132, "l2_bytes": 50 * 2 ** 20,
    "capability": "9.0", "memory_bytes": 80 * 10 ** 9, "driver": "planted",
    "study_card": STUDY_CARD, "same_card_as_study": False}

#: The planted worlds: why each exists and the exit it must score.
R3_WORLDS: dict[str, tuple[str, int, str | None]] = {
    "group": ("SHARED and NATIVE read the group model (w2 shared by co-resident "
              "tiles at G=1), PRIVATE reads n: the prediction come true",
              exit_codes.DONE, None),
    "no-reuse": ("SHARED re-reads the whole set at every G: the group model refuted",
                 exit_codes.CLAIM_FAIL, "C1"),
    "private-reads-copy-0": ("PRIVATE reads like SHARED: the relabelling did not take",
                             exit_codes.INVALID, "V5"),
    "declaration": ("NATIVE reads 5% more than SHARED", exit_codes.INVALID, "V7"),
    "noisy": ("the K calls of every cell spread by 2%", exit_codes.INVALID, "V3"),
    "request-mismatch": ("PRIVATE requests 1% more L2 sectors", exit_codes.INVALID, "V6"),
    "uncarded": ("the page carries no card block", exit_codes.INVALID, "V0"),
}

#: The co-resident w2 sharing the "group" world plants at G=1: w2's q grows
#: at half the rate of w1's, so alpha_w2 < alpha_w1 and C3 passes.
R3_PLANTED_W2_SHARE = 0.5


def planted_r3_manifest(plan: dict, *, device_uuid: str) -> dict:
    """What the child would write for `plan`, from arithmetic alone: the grids
    come from vLLM's documented buffer size and the proof is planted."""
    r3 = _r3()
    cfg = MODEL_CONFIGS[plan["model"]]
    sched = r3.counter_schedule(plan)
    copies, bm = int(plan["copies_declared"]), int(plan["block_m"])
    declared = {a: r3.declared_experts(a, cfg.num_experts, copies) for a in r3.ARMS}
    tokens = {int(n): tokens_for_rows(cfg, int(n) * bm) for n in plan["treads"]}
    grids = {}
    for arm, n in plan["cells"]:
        em = r3.predicted_sorted_ids(tokens[int(n)] * cfg.top_k, declared[arm], bm)
        grids[f"{arm}/{n}"] = r3.expected_grids(cfg, em=em, tokens=tokens[int(n)],
                                                block_m=bm, block_n=int(plan["block_n"]))
    return {"family": R3_FAMILY, "kind": plan["kind"], "order": plan["cells"],
            "calls_per_cell": int(plan["calls_per_cell"]),
            "warmup_calls": int(plan["warmup_calls"]),
            "gemms_per_call": r3.GEMMS_PER_CALL, "launch_skip": sched.launch_skip,
            "launch_count": sched.launch_count,
            "tokens": {str(n): t for n, t in tokens.items()},
            "copies_declared": copies, "declared_by_arm": declared,
            "grids": grids, "pinned": r3.pinned_config(plan["block_n"], plan["group_m"],
                                                       plan["num_stages"]),
            "proof": {"parts": {name: True for name, _ in r3.PROOF_PARTS},
                      "detail": {name: "planted" for name, _ in r3.PROOF_PARTS},
                      "verdict": PASS, "tread": max(int(n) for n in plan["treads"]),
                      "synthetic": True},
            "memory_plan": None, "versions": {"planted": True},
            "device": {"name": "planted", "uuid": device_uuid}}


def r3_world_launch(world: str, cfg, *, block_m: int, group_m: int, arm: str, n: int,
                    gemm: str, call: int, calls: int, grid: int) -> dict[str, float]:
    """One planted launch's metrics in `world`, in canonical units."""
    byte = r3_byte_model(cfg, "bf16", block_m)
    weight = byte["W_w1"] if gemm == "w1" else byte["W_w2"]
    op = byte[f"operand_per_tile_{gemm}"]
    gr = group_reads(cfg.num_experts, n, group_m)
    if arm == "private":
        q = gr if world == "private-reads-copy-0" else float(n)
    elif world == "no-reuse":
        q = float(n)
    else:
        q = gr
        if gemm == "w2" and group_m == 1:
            q = 1.0 + (gr - 1.0) * R3_PLANTED_W2_SHARE
    read = q * weight + n * op
    if world == "declaration" and arm == "native":
        read *= 1.05
    centre = (calls - 1) / 2.0
    wobble = (0.02 / max(calls - 1, 1)) if world == "noisy" else 1e-5
    read *= 1.0 + wobble * (call - centre)
    requested = (n * weight + n * op) / L2_SECTOR_BYTES
    if world == "request-mismatch" and arm == "private":
        requested *= 1.01
    fill = read / L2_SECTOR_BYTES
    return {"dram__bytes_read.sum": read, "dram__bytes_write.sum": 4096.0 * n,
            "lts__t_sectors_srcunit_tex_op_read.sum": requested,
            "launch__grid_size": float(grid),
            "gpu__time_duration.sum": 1e9 * read / 3.0e12,
            "lts__d_sectors_fill_device.sum": fill,
            "lts__t_sectors_op_read_lookup_miss.sum": fill,
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": max(requested - fill, 0.0),
            "lts__t_sector_op_read_hit_rate.pct": 100.0 * max(1.0 - fill / requested, 0.0),
            "launch__occupancy_limit_blocks": 32.0,
            "launch__occupancy_limit_registers": 1.0,
            "launch__occupancy_limit_shared_mem": 1.0,
            "launch__occupancy_limit_warps": 8.0,
            "launch__registers_per_thread": 128.0,
            "launch__waves_per_multiprocessor": grid / 132.0,
            "lts__t_sectors_srcunit_ltcfabric.sum": 0.0}


def canned_r3_csv(manifest: dict, world: str, *, model: str = "mixtral-8x7b",
                  block_m: int = 32, group_m: int, layout: str = CSV_WIDE,
                  drop_launch: int | None = None, extra_launch: bool = False,
                  swap_grid_at: int | None = None) -> str:
    """ncu's CSV for a planted run of `manifest` in `world`, in either layout.

    The three knobs at the end plant the attribution's failures: a launch
    dropped, a launch added, and one launch's grid swapped with its partner's.
    """
    cfg = MODEL_CONFIGS[model]
    seq = r3_launch_sequence(manifest)
    k = int(manifest["calls_per_cell"])
    launches = []
    for key, call, gemm, _warm in seq:
        arm, n = key.split("/")
        launches.append((key, r3_world_launch(
            world, cfg, block_m=block_m, group_m=group_m, arm=arm, n=int(n), gemm=gemm,
            call=call, calls=k, grid=int(manifest["grids"][key][gemm]))))
    if swap_grid_at is not None:
        i = swap_grid_at
        a, b = launches[i][1], launches[i + 1][1]
        a["launch__grid_size"], b["launch__grid_size"] = (b["launch__grid_size"],
                                                           a["launch__grid_size"])
    if drop_launch is not None:
        launches.pop(drop_launch)
    if extra_launch:
        launches.append(launches[-1])
    head = ["==PROF== Connected to process 1 (python)",
            "==PROF== Disconnected from process 1"]
    units = {m: unit_table(m)[0] for m in R3_ALL_METRICS}
    if layout == CSV_WIDE:
        fixed = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
                 "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"]
        rows = [",".join(f'"{c}"' for c in fixed + list(R3_ALL_METRICS)),
                ",".join(['""'] * len(fixed) + [f'"{units[m]}"' for m in R3_ALL_METRICS])]
        for i, (_key, vals) in enumerate(launches):
            lead = [str(i), "1", "python", "127.0.0.1", GEMM_MARKER, "1", "7",
                    "(256, 1, 1)", f"({int(vals['launch__grid_size'])}, 1, 1)", "0", "9.0"]
            rows.append(",".join(f'"{c}"' for c in lead)
                        + "," + ",".join(f'"{vals[m]!r}"' for m in R3_ALL_METRICS))
        return "\n".join(head + rows) + "\n"
    rows = ['"ID","Process ID","Process Name","Kernel Name","Metric Name",'
            '"Metric Unit","Metric Value"']
    for i, (_key, vals) in enumerate(launches):
        for m in R3_ALL_METRICS:
            rows.append(f'"{i}","1","python","{GEMM_MARKER}","{m}","{units[m]}",'
                        f'"{vals[m]!r}"')
    return "\n".join(head + rows) + "\n"


def planted_r3_plan(group_m: int, *, kind: str = "measure") -> dict:
    """The registered plan at `group_m`, as `--run` would write it."""
    r3 = _r3()
    fixed = r3.SWEEP.FIXED
    if kind == "census":
        return r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=r3.DEFAULT_BLOCK_M,
                       block_n=fixed["BLOCK_SIZE_N"], num_stages=fixed["num_stages"],
                       group_m=group_m, treads=R3_CENSUS_TREADS, kind="census",
                       arms=(r3.NATIVE,), calls=1, warmups=1,
                       profile_dir=Path("planted"), stem="census")
    return r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=r3.DEFAULT_BLOCK_M,
                   block_n=fixed["BLOCK_SIZE_N"], num_stages=fixed["num_stages"],
                   group_m=group_m, treads=R3_TREADS, kind="measure", arms=r3.ARMS,
                   calls=R3_CALLS_PER_CELL, warmups=R3_WARMUP_CALLS,
                   profile_dir=Path("planted"), stem=f"g{group_m}")


def planted_r3_page(world: str, group_m: int, *, layout: str = CSV_WIDE) -> dict:
    """A whole page in `world` at `group_m`: plan, manifest, ncu CSV, parse,
    attribution, reduction, page and gates, every step the code `--run`
    runs, with only the child and ncu planted."""
    r3 = _r3()
    plan = planted_r3_plan(group_m)
    manifest = planted_r3_manifest(plan, device_uuid=R3_PLANTED_CARD["uuid"])
    text = canned_r3_csv(manifest, world, block_m=int(plan["block_m"]), group_m=group_m,
                         layout=layout)
    launches = parse_ncu_csv(text, soft=frozenset(R3_RECORDED_METRICS))
    cells = r3_reduce_cells(attribute_launches(launches, manifest), manifest,
                            R3_ALL_METRICS)
    page = build_r3_page(
        plan=plan, manifest=manifest, cells=cells,
        card=None if world == "uncarded" else dict(R3_PLANTED_CARD),
        stack={"torch": "planted", "triton": "planted", "vllm": "planted",
               "python": "planted"},
        ncu={"binary": "/planted/ncu", "version": "planted", "argv": [],
             "replay_mode": "kernel", "cache_control": "all", "clock_control": "base",
             "report": "planted", "report_sha256": None, "csv": "planted",
             "csv_layout": layout, "metrics_asked": list(R3_ALL_METRICS),
             "metrics_dropped": []},
        census={"path": "planted", "sha256": None, "gemms_per_call": r3.GEMMS_PER_CALL})
    gates, _summary = score_r3_page(page)
    page["gates"] = [asdict(g) for g in gates]
    return page


def self_test_r3() -> list[tuple[str, bool, str]]:
    """The family's arithmetic, parser, attribution and gates on planted
    pages, off any GPU: the group model against vLLM's pid mapping walked by
    brute force, both CSV layouts to identical cells, the attribution's three
    refusals, and every world scoring the exit it was registered to score."""
    out: list[tuple[str, bool, str]] = []
    worst = 0.0
    for e in (8, 64):
        for n in range(1, 9):
            for g in (1, 2, 3, 4, 8, 16, 64):
                worst = max(worst, abs(group_reads(e, n, g) - pid_mapping_reads(e, n, g)))
    out.append(("group_reads against vLLM's pid mapping by brute force", worst < 1e-12,
                f"largest difference {worst:.2e} over E in {{8, 64}}, n 1..8, 7 G"))
    wide = planted_r3_page("group", 4)
    long = planted_r3_page("group", 4, layout=CSV_LONG)
    same = [c["per_call"] for c in wide["cells"]] == [c["per_call"] for c in long["cells"]]
    out.append(("both ncu CSV layouts reduce to the same cells", same,
                f"{len(wide['cells'])} cells from a wide and a long planted CSV"))
    plan = planted_r3_plan(4)
    manifest = planted_r3_manifest(plan, device_uuid="planted")
    for label, knob in (("one launch more than planted", {"extra_launch": True}),
                        ("one launch fewer than planted", {"drop_launch": 3}),
                        ("a w1 launch's grid swapped with its w2", {"swap_grid_at": 0})):
        text = canned_r3_csv(manifest, "group", group_m=4, **knob)
        try:
            attribute_launches(parse_ncu_csv(text), manifest)
            out.append((f"attribution refuses {label}", False, "it attributed them"))
        except CounterRunRefused as exc:
            out.append((f"attribution refuses {label}", True,
                        f"REFUSED as required: {str(exc)[:70]}"))
    for world, (_why, want, gate) in R3_WORLDS.items():
        for g_m in ((1, 4) if world == "group" else (4,)):
            page = planted_r3_page(world, g_m)
            gates = [Gate(**d) for d in page["gates"]]
            rc = exit_codes.classify(g.scored() for g in gates)
            named = [g for g in gates if g.number == gate] if gate else []
            good = rc == want and (gate is None or (named and named[0].verdict == FAIL))
            # Lower case on purpose: a self-test log carries the word FAIL only
            # when a row of the self-test itself failed.
            out.append((f"planted {world} world at G={g_m}", good,
                        f"exit {rc}, wanted {want}"
                        + (f"; {gate} reads {named[0].verdict.lower() if named else 'absent'}"
                           ", as registered" if gate else "")))
    return out


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the registered plan, its predictions and its cost. No GPU.")
    ap.add_argument("--bracket", action="store_true",
                    help="bound alpha from published data with no counter at all. No GPU.")
    ap.add_argument("--probe", action="store_true",
                    help="ask THIS machine which counter route is open. Needs the box.")
    ap.add_argument("--self-test", action="store_true",
                    help="plant an alpha, a profile and a call count, and check the "
                         "estimator and the parser return them. No GPU.")
    ap.add_argument("--run", action="store_true",
                    help="drive ncu over the chosen cell and write the JSON --analyse "
                         "consumes. Needs the box and an open route; refuses otherwise")
    ap.add_argument("--analyse", nargs="+", metavar="JSON", default=None,
                    help="score a measured counter run, dispatched on the payload's "
                         "family. ONE ladder payload is scored exactly as it always "
                         "was (the ratio ACROSS ladder runs is --contrast). r3-arms "
                         "pages: one page's gates, or with several the alpha(G) "
                         "table, refusing pages from two card UUIDs, two commits, two "
                         "vLLM versions or two designs")
    ap.add_argument("--contrast", nargs="+", metavar="JSON", default=None,
                    help="score the RATIO across two or more counter runs, which is "
                         "the reading the extended plan takes and the one no single "
                         "payload contains. Pairs are matched inside one cache mode; "
                         "a pair nobody ran is not scored")
    ap.add_argument("--published", nargs="*", default=None,
                    help="published run directories for --bracket (default: every "
                         "results/published/*alpha-surface*)")
    ap.add_argument("--family", default=LADDER_FAMILY, choices=FAMILIES,
                    help="which registered cell family. 'ladder' (the default, "
                         "unchanged) is the timed sweep's whole-layer call, one tile "
                         "count per ncu invocation. 'r3-arms' is R3's three arms "
                         "(native, shared, private) under the counter, one "
                         "GROUP_SIZE_M per invocation, the arm GEMMs only: its "
                         "--block-m, --block-n, --num-stages and --tiles default to "
                         "R3's (32, 64, 4, treads 1 2 3 4 6), --card is not read (the "
                         "page's card is the live device), and no ridge, bandwidth or "
                         "calibration enters it. See docs/COUNTERS.md section 6")
    ap.add_argument("--census-only", action="store_true",
                    help="with --run --family r3-arms: profile the census mini plan "
                         "(native at n 1 and 6, one warmup and one call each, no "
                         "skip and no cap) and write census.json: it must hold "
                         "exactly GEMMS_PER_CALL launches per call at the child's "
                         "grids, and every page needs it")
    ap.add_argument("--census", default="",
                    help="with --run --family r3-arms: the census.json this card, "
                         "commit and vLLM wrote; a page refuses any other")
    ap.add_argument("--reduce-only", action="store_true",
                    help="with --run --family r3-arms --group-m G --census C --out "
                         "P: rebuild the page from the profiles a capture left under "
                         "P's .profiles directory (plan, manifest, capture record, "
                         ".ncu-rep or CSV), with no card, no probe and no child, so a "
                         "parser fix never needs the box again")
    ap.add_argument("--timed-reference", nargs="+", default=None, metavar="REPORT",
                    help="with --analyse over r3-arms pages: R3's timed report.json "
                         "files, whose ratios C5 compares the bytes with, labelled "
                         "cross-card. Read, never typed")
    ap.add_argument("--card", default="nvidia_h200",
                    choices=sorted(DATASHEET_PEAK_GBPS),
                    help="which card the plan is written for. The H200 by default, "
                         "because it is the card this study rents and calibrates. It "
                         "was the A100 until 2026-09-10, where the three anchors are "
                         "furthest apart. NOT because any --probe has read a counter: "
                         "the 2026-09-09 and 2026-09-10 OPEN readings profiled "
                         "/bin/true and never attempted one, and the 2026-09-15 pod "
                         "refused the read")
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--group-m", type=int, default=16,
                    help="16, not 1, for the PRIMARY cell: the anchor disagreement is a "
                         "G>1 phenomenon, because at n=1 there is one M-tile per expert, "
                         "so a G>1 swizzle group spans G different experts and can reuse "
                         "nothing, which is the mechanism that puts the n=1 tread above "
                         "the branch it anchors. The extended plan also runs a G=1 cell: "
                         "statement (3) of the 2026-09-10 analysis says the per-M-tile "
                         "cost is a function of the schedule, so an all-G=16 plan cannot "
                         "see it. See the CONTRAST block of --dry-run")
    ap.add_argument("--block-n", type=int, default=64,
                    help="64, the shipped BLOCK_SIZE_N and the hinge of the BLOCK_N "
                         "contrast, which reads 32 and 128 on either side of it")
    ap.add_argument("--block-m", type=int, default=64,
                    help="64 by default: it is the BLOCK_M the 2026-09-10 analysis "
                         "states the BLOCK_N separation at, and the only BLOCK_M the "
                         "occupancy arm ran, so both contrasts are read at one tile "
                         "height. It was 32, for the A100 cell this plan has left")
    ap.add_argument("--num-stages", type=int, default=SWEEP_FIXED["num_stages"],
                    help="the pipeline depth the profiled cell compiles at. The "
                         "sweep's own FIXED value by default; pass 3 for the "
                         "occupancy arm's schedule pair, which ran at 3")
    ap.add_argument("--tiles", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=DEFAULT_TILES, help="tile counts per expert to profile")
    ap.add_argument("--cache-control", default="all", choices=("all", "none"),
                    help="ncu --cache-control. 'all' is ncu's own default and flushes "
                         "every cache before each replay pass")
    ap.add_argument("--call-marker", default=CALL_MARKER,
                    help="the kernel name substring whose launch count IS the "
                         "fused_experts call count. A marker that matches nothing "
                         "REFUSES and prints the names the profile did contain, so a "
                         "vLLM rename costs one flag and not an hour of pod time")
    ap.add_argument("--profile-dir", default="",
                    help="where --run keeps the ncu CSVs and the wrapped sweep's own "
                         "output. Defaults to <out>.profiles beside --out")
    ap.add_argument("--ncu-timeout", type=float, default=3600.0,
                    help="seconds one profiled invocation may take. ncu replay saves "
                         "and restores the weight buffers per pass, so this is minutes "
                         "per launch and not seconds")
    ap.add_argument("--report", default=str(DEFAULT_REPORT),
                    help="the PUBLISHED report the plan registers its predictions "
                         "against. Defaults to the H200 mixtral G=16 BN=64 arm, the "
                         "twin of the A100 cell, on the card whose route is open")
    ap.add_argument("--anchor", nargs=2, action="append", metavar=("NAME", "VALUE"),
                    type=str, default=None,
                    help="override a registered anchor, e.g. --anchor t1 0.4522")
    ap.add_argument("--out", default="",
                    help="write the JSON result here. Its git visibility is "
                         "CHECKED and printed beside the path: results/* is "
                         "ignored with only results/published/ excepted, and a "
                         "--bracket result dropped by `git add -A` is the loss "
                         "this repo has already taken once")
    return ap


def _given(argv: list[str], flag: str) -> bool:
    """Whether the operator typed `flag`, as opposed to argparse defaulting it."""
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def resolve_r3_defaults(args, argv: list[str]) -> None:
    """The r3-arms family's defaults for the knobs the operator did not type:
    R3's BLOCK_M, its pinned BLOCK_N and num_stages, and the family's treads.
    The ladder family's defaults stay the parser's, which `build_parser()
    .parse_args([])` still returns unchanged."""
    r3 = _r3()
    if not _given(argv, "--block-m"):
        args.block_m = r3.DEFAULT_BLOCK_M
    if not _given(argv, "--block-n"):
        args.block_n = r3.SWEEP.FIXED["BLOCK_SIZE_N"]
    if not _given(argv, "--num-stages"):
        args.num_stages = r3.SWEEP.FIXED["num_stages"]
    if not _given(argv, "--tiles"):
        args.tiles = R3_TREADS


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.anchor:
        args.anchor = [(k, float(v)) for k, v in args.anchor]
    args.group_m_given = _given(argv, "--group-m")
    if args.family == R3_FAMILY:
        resolve_r3_defaults(args, argv)
    chosen = [args.dry_run, args.bracket, args.probe, args.self_test, args.run,
              bool(args.analyse), bool(args.contrast)]
    if sum(bool(c) for c in chosen) != 1:
        print("REFUSE: pick exactly one of --dry-run / --bracket / --probe / "
              "--self-test / --run / --analyse / --contrast. Running two would "
              "interleave a plan with a result and this study has been burned by "
              "exactly that.")
        return exit_codes.REFUSED
    if (args.census_only or args.census or args.reduce_only) and not (
            args.run and args.family == R3_FAMILY):
        print("REFUSE: --census-only, --census and --reduce-only belong to --run "
              "--family r3-arms")
        return exit_codes.REFUSED
    if args.reduce_only and args.census_only:
        print("REFUSE: --reduce-only rebuilds a page; a census is never reduced apart "
              "from its capture")
        return exit_codes.REFUSED
    if args.timed_reference and not args.analyse:
        print("REFUSE: --timed-reference is read by --analyse over r3-arms pages")
        return exit_codes.REFUSED
    if args.family == R3_FAMILY and (args.bracket or args.contrast):
        print("REFUSE: --bracket and --contrast are the ladder family's; the r3-arms "
              "family's reading across pages is --analyse over several pages")
        return exit_codes.REFUSED
    if args.contrast:
        try:
            return do_contrast(args)
        except (CounterRunRefused, CorpusMissing) as exc:
            # Same code as --run's refusal and for the same reason: the payloads
            # exist, the comparison examined them and refused. Nothing about the
            # world was decided and no retry of the same files can help.
            print(f"REFUSED: {exc}")
            print(exit_codes.result_line("VALIDITY", "X0", exit_codes.FAIL,
                                         str(exc).replace("\n", " ")[:160]))
            return exit_codes.INVALID
    if args.analyse:
        return do_analyse_any(args)
    if args.probe:
        return do_probe(args)
    if args.bracket:
        return do_bracket(args)
    if args.self_test:
        return do_self_test(args)
    if args.run:
        try:
            if args.family == R3_FAMILY:
                return do_reduce_r3(args) if args.reduce_only else do_run_r3(args)
            return do_run(args)
        except (CounterRunRefused, CorpusMissing) as exc:
            # INVALID (3), not CLAIM_FAIL and not ERROR. The profiler ran and the
            # reduction refused it: nothing about the world was decided, and
            # nothing this run produced may be quoted. CLAIM_FAIL would file it
            # as a refuted prediction, and ERROR would put it in the retry queue
            # when a retry cannot help: a wrong call marker or a missing metric
            # needs a flag changed, not the same command again.
            print(f"REFUSED: {exc}")
            print(exit_codes.result_line("VALIDITY", "R1", exit_codes.FAIL,
                                         str(exc).replace("\n", " ")[:160]))
            return exit_codes.INVALID
    if args.family == R3_FAMILY:
        return do_dry_run_r3(args)
    return do_dry_run(args)


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
        # crash and an interrupt both deserve. `memory_branch_anchor.py` has
        # had this since 2026-09-02 and this arm is the last one without it.
        traceback.print_exc()
        sys.exit(exit_codes.ERROR)
