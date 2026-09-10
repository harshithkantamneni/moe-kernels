#!/usr/bin/env python3
"""Is there a route to a DRAM counter, and what would it settle if there were?

    python scripts/dram_counter_route.py --dry-run          # the plan and its cost, off GPU
    python scripts/dram_counter_route.py --bracket          # a result, today, off GPU
    python scripts/dram_counter_route.py --self-test        # the estimator and the parser, off GPU
    python scripts/dram_counter_route.py --probe            # which route is open on THIS box
    python scripts/dram_counter_route.py --run --out c.json # TAKE the measurement, needs the GPU
    python scripts/dram_counter_route.py --analyse c.json   # score ONE counter run
    python scripts/dram_counter_route.py --contrast a.json b.json  # score the RATIO across runs

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
             measurement: the route came back OPEN on the H200 box and the arm
             had no runner. See "THE PER-CALL TRAP" below, which is the whole
             reason this is a mode and not a shell one-liner.

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
The route came back OPEN on the H200 box (ncu 2025.1.1.0, attached, no
permission error, no CAP_SYS_ADMIN needed) and on no other, so the plan now
registers the H200 cell and reads every prediction out of
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
import statistics
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The shared exit-code table and the shared provenance block, both imported
# rather than approximated here: this file spent its life returning integers it
# chose for itself and writing JSON that named no commit and no machine.
from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402

# Imported, never re-derived. Two byte models for one study is how the padding
# tax survived three months: `weight_bytes_per_expert` and
# `activation_bytes_per_row` are the SAME functions the ladder fit is scored
# against, so if they move, this file's predictions move with them and
# tests/test_dram_counter_route.py pins the numbers so the move is visible.
from moe.spec import MODEL_CONFIGS  # noqa: E402  (after sys.path insert)
from scripts.block_m_crossing_sweep import FIXED as SWEEP_FIXED  # noqa: E402
from scripts.block_m_crossing_sweep import (  # noqa: E402
    activation_bytes_per_row,
    ai_cap,
    q_of_tiles,
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

#: `--probe` measures nothing at all: it asks the box what it is.
PROBE_INSTRUMENT = "machine-configuration-probe/nothing-timed"

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
}

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
#: any more: no counter route has ever been open on an A100 this study can rent.
A100_REPORT = (REPO / "results" / "published"
               / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
               / "mixtral-8x7b-bf16-r1024-g16-n64-b156b5.report.json")

#: The cell the plan is written for, by default: the H200 twin of the same arm,
#: on the card whose `--probe` came back OPEN. Same model, same dtype, same
#: GROUP_SIZE_M=16, same BLOCK_SIZE_N=64, so the counter still answers the
#: ladder rather than a different question, on the box that can answer it.
DEFAULT_REPORT = (REPO / "results" / "published"
                  / "2026-09-01-nvidia_h200-alpha-surface-s4"
                  / "mixtral-8x7b-bf16-r1024-g16-n64-69f35a.report.json")

#: Bit 21 of the Linux capability mask. CAP_SYS_ADMIN is what NVIDIA's own
#: ERR_NVGPUCTRPERM page names as the container-side alternative to changing the
#: host module parameter, so a probe that does not check it cannot tell
#: "this container could profile" from "this host refuses".
CAP_SYS_ADMIN_BIT = 21


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
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]}: timed out after {timeout}s"


def probe_capabilities() -> dict:
    """Is CAP_SYS_ADMIN held by THIS process?

    NVIDIA's ERR_NVGPUCTRPERM page says a container may profile either because
    the host enabled it or because the container "was started with the
    appropriate permissions by passing --cap-add=SYS_ADMIN". Those are two
    different asks to a provider and a log cannot tell them apart, so the probe
    reads the capability mask rather than guessing from a failure message.
    """
    path = Path("/proc/self/status")
    if not path.exists():
        return {"available": False, "why": "no /proc/self/status; not a Linux container"}
    for line in path.read_text().splitlines():
        if line.startswith("CapEff:"):
            mask = int(line.split()[1], 16)
            return {"available": True, "cap_eff": hex(mask),
                    "sys_admin": bool(mask >> CAP_SYS_ADMIN_BIT & 1)}
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


def probe_ncu() -> dict:
    """Is ncu installed, and if so what does a minimal invocation actually say?

    The minimal invocation profiles `true`, which launches no kernel. That is
    deliberate: the permission check happens at profiler ATTACH, before any
    kernel runs, so ERR_NVGPUCTRPERM surfaces in under a second with no GPU work
    and no risk of a long profile on a metered box.
    """
    binary = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if not binary:
        return {"present": False, "why": "no ncu on PATH"}
    rc, out, err = _run([binary, "--version"], timeout=30)
    version = (out or err).strip().splitlines()[-1] if (out or err) else ""
    rc2, out2, err2 = _run([binary, "--metrics", NCU_METRICS[0], "/bin/true"], timeout=90)
    blob = f"{out2}\n{err2}"
    if "ERR_NVGPUCTRPERM" in blob:
        cause = "ERR_NVGPUCTRPERM: counters gated by the host module flag or a missing capability"
    elif rc2 == 0:
        cause = "attached with no permission error"
    else:
        cause = f"failed with rc={rc2} and no permission marker; read the raw output"
    return {"present": True, "binary": binary, "version": version,
            "returncode": rc2, "cause": cause, "output_head": blob.strip()[:600]}


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
    if ncu.get("present") and ncu.get("cause", "").startswith("attached"):
        return "OPEN", ["ncu attached with no permission error: read the plan with "
                        "--dry-run and then take it with --run, which drives ncu over "
                        "one cell and writes the JSON --analyse scores. Run every cell "
                        "of a registered pair and score the RATIO with --contrast: the "
                        "discriminator is across cells and --analyse sees one. This is "
                        "the decisive route."]
    if ncu.get("present") and "ERR_NVGPUCTRPERM" in ncu.get("cause", ""):
        if caps.get("available") and not caps.get("sys_admin"):
            notes.append("this process does NOT hold CAP_SYS_ADMIN. NVIDIA's own "
                         "ERR_NVGPUCTRPERM page names --cap-add=SYS_ADMIN as the "
                         "container-side fix, so ask the provider for that capability "
                         "before asking for a host reboot.")
        if flag.get("available") and flag.get("restrict") == 1:
            notes.append("the host loaded the module with "
                         "RestrictProfilingToAdminUsers=1; the host-side fix is a module "
                         "parameter change and a reload, which a tenant cannot do.")
        if flag.get("available") and flag.get("restrict") == 0:
            notes.append("the host ALREADY allows unprivileged profiling "
                         "(RestrictProfilingToAdminUsers=0) yet ncu still refused. That "
                         "combination is not explained by the module flag and needs the "
                         "raw ncu output read, not another retry.")
        return "BLOCKED", notes
    if not ncu.get("present"):
        notes.append("no ncu here. It installs from the public CUDA apt tree as a plain "
                     "file: the nsight-compute-* debs sit beside the nsight-systems-* ones "
                     "and `dpkg -x` unpacks either without root.")
    if nsys.get("present") and not nsys.get("importer_present"):
        notes.append("nsys is installed WITHOUT its importer, which is the 2026-09-01 pod's "
                     "failure exactly. Any capture here writes a .qdstrm that this machine "
                     "cannot convert. See docs/COUNTERS.md for the version-matched fix.")
    return "REFUSE", notes or ["not enough evidence on this machine to name the route"]


def do_probe(args) -> int:
    caps, flag = probe_capabilities(), probe_module_flag()
    ncu, nsys = probe_ncu(), probe_nsys()
    verdict, notes = route_verdict(caps, flag, ncu, nsys)
    print("ROUTE PROBE")
    print(f"  host      {os.uname().sysname} {os.uname().machine}")
    print(f"  caps      {caps}")
    print(f"  module    {flag}")
    print(f"  ncu       {ncu.get('cause', ncu.get('why'))}"
          + (f"  [{ncu.get('version', '')}]" if ncu.get("present") else ""))
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
    # expectation is that a counter route is OPEN; BLOCKED is that expectation
    # refuted by the box, a CLAIM_FAIL, which the table defines as "measured,
    # the world disagreed": a RESULT the ledger files as finished and the
    # summary prints as the finding, never a retry. It is the answer this arm
    # was written to obtain and it now says so in the greppable line. REFUSE
    # stays REFUSED (2), before any gate: the box did not say enough to name a
    # route, nothing was established, and there is no gate to print.
    gates: list[Gate] = []
    if verdict != REFUSE:
        gates.append(Gate(
            "P1", "CLAIM", "a counter route is open on this box",
            PASS if verdict == "OPEN" else FAIL,
            f"route {verdict}: {ncu.get('cause', ncu.get('why', 'no ncu'))}",
            "OPEN",
            "the counter experiment on this box; the plan stands and needs "
            "another box or a provider-side change named in the notes above"))
        print()
        for g in gates:
            for line in g.render():
                print(line)
    payload = stamped({"verdict": verdict, "notes": notes, "capabilities": caps,
                       "module_flag": flag, "ncu": ncu, "nsys": nsys,
                       "gates": [asdict(g) for g in gates]},
                      mode="probe", args=args, card=live_card(),
                      instrument=PROBE_INSTRUMENT)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n  wrote {out}")
        print(f"  git   {git_visibility(out)}")
    if verdict == REFUSE:
        return exit_codes.REFUSED
    return exit_codes.classify(g.scored() for g in gates)


# --------------------------------------------------------------------------
# --run: drive ncu, reduce the profile PER CALL, write the schema.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Launch:
    """One profiled kernel launch and the registered metrics it reported."""

    launch_id: str
    kernel: str
    metrics: dict[str, float]


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
    canonical, table = NCU_METRIC_UNITS[metric]
    key = (unit or "").strip()
    if key not in table:
        raise CounterRunRefused(
            f"{metric} came back in unit {unit!r}, which this parser has never been "
            f"shown; known units are {sorted(table)} reduced to {canonical}. Add it "
            "deliberately rather than scaling it by 1. An EMPTY unit is refused here "
            "too, and used to be scaled by 1: ncu rescales per launch, so a metric "
            "with no unit beside it cannot be reduced")
    return value * table[key]


def parse_ncu_csv(text: str) -> list[Launch]:
    """`ncu --csv --page raw` output to one `Launch` per profiled launch.

    The raw page is one row per (launch, metric), so the launches are recovered
    by grouping on the launch ID column. The ID is REQUIRED: without it the only
    other way to tell two launches of the same kernel apart is the kernel time
    string, which repeats, and merging two launches into one halves the traffic
    the reduction then divides by the call count.

    Everything before the header is skipped rather than parsed. ncu prefixes its
    own progress with `==PROF==` and the profiled process writes its own stdout
    into the same stream when no `--log-file` is given; `--run` always passes
    one, and this still skips, because a parser that trusts line 1 is a parser
    that breaks the first time ncu prints a warning.
    """
    import csv as _csv
    rows = list(_csv.reader(text.splitlines()))
    header = None
    for i, row in enumerate(rows):
        if {"Kernel Name", "Metric Name", "Metric Value"} <= set(row):
            header, rows = row, rows[i + 1:]
            break
    if header is None:
        raise CounterRunRefused(
            "no ncu CSV header in this output: expected a row carrying "
            "'Kernel Name', 'Metric Name' and 'Metric Value'. Profile with "
            "--csv --page raw and read the log file, not the console")
    if "ID" not in header:
        raise CounterRunRefused(
            f"the ncu CSV has no 'ID' column (columns: {header}). Without a launch "
            "id two launches of one kernel cannot be told apart and the per-call "
            "division would be wrong by their count")
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
            "to bytes and would be off by whatever prefix ncu chose. Profile with "
            "--csv --page raw, which emits the column")
    idx = {name: header.index(name) for name in
           ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")}
    order: list[str] = []
    seen: dict[str, tuple[str, dict[str, float]]] = {}
    for row in rows:
        if len(row) != len(header) or row == header:
            continue
        metric = row[idx["Metric Name"]].strip()
        if metric not in NCU_METRIC_UNITS:
            continue
        launch_id = row[idx["ID"]].strip()
        kernel = row[idx["Kernel Name"]].strip()
        value = _metric_value(metric, row[idx["Metric Unit"]], row[idx["Metric Value"]])
        if launch_id not in seen:
            order.append(launch_id)
            seen[launch_id] = (kernel, {})
        known_kernel, metrics = seen[launch_id]
        if metric in metrics:
            raise CounterRunRefused(
                f"launch {launch_id} reports {metric} twice; the ID column is not "
                "unique in this file and the launches cannot be separated")
        if kernel != known_kernel:
            raise CounterRunRefused(
                f"launch {launch_id} is named both {known_kernel!r} and {kernel!r}")
        metrics[metric] = value
    if not order:
        raise CounterRunRefused(
            "the ncu CSV carries no row for any registered metric. A profile that "
            "measured nothing also reports no failures")
    return [Launch(i, seen[i][0], seen[i][1]) for i in order]


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


def ncu_argv(binary: str, cache_control: str, log_file: Path) -> list[str]:
    """The profiler wrapper. `--log-file` is not optional here: without it ncu's
    CSV and the profiled process's own stdout interleave in one stream."""
    return [binary, "--metrics", ",".join(NCU_METRICS),
            "--replay-mode", "kernel", "--cache-control", cache_control,
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
    if not (ncu.get("present") and ncu.get("cause", "").startswith("attached")):
        print("REFUSE: no open counter route on this box. --probe says: "
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
    elif mode == "bracket":
        knobs = {"mode": mode,
                 "published": sorted(str(p) for p in (args.published or []))}
    elif mode == "analyse":
        knobs = {"mode": mode, "payload": str(args.analyse),
                 "anchor": dict(args.anchor or {})}
    elif mode == "contrast":
        # The payloads compared and the window that scores them. Two contrasts
        # over different payload sets, or the same set at a different tolerance,
        # are two readings and must not share an id.
        knobs = {"mode": mode, "payloads": sorted(str(p) for p in args.contrast),
                 "tolerance": CONTRAST_TOLERANCE}
    elif mode == "self-test":
        knobs = {"mode": mode}
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


def do_self_test(args) -> int:
    """Plant an alpha, synthesise the counter rows the model implies, and check
    the estimator returns it. Then plant a ladder and check the bracket contains
    the alpha that generated it. Then plant two anchor sets, one separating and
    one clustered, and check C1 registers the question each cell can carry.
    Then plant a TRAFFIC world and a TIME world and check the contrast scorer
    names the right rival. Then plant a PROFILE with a known call count and
    check `--run`'s parser and its per-call division return it.

    This is the check that the analysis half is not itself the source of a
    number. `block_m_crossing_sweep.py` has the same shape for the same reason:
    an estimator that has never been run against a known answer is an assertion.
    The parser half was added on 2026-09-10 with `--run`: a runner whose parsing
    has never been exercised is the same assertion one layer down, and it would
    have been exercised for the first time on a rented box.
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
    print(f"\n  SELF TEST {'PASS' if ok else 'FAIL'}")
    # A SELF TEST IS A VALIDITY GATE ON THE ANALYSIS HALF, so its failure is
    # INVALID (3) and not CLAIM_FAIL (1): an estimator that cannot recover a
    # planted alpha has not refuted anything about the world, it has said that
    # nothing this file computes may be quoted. It returned 1, which the driver
    # files as a finished result and never retries. The gate prints its RESULT
    # line like every other gate in this file, so the log and the code agree.
    gate = Gate("S1", "VALIDITY",
                "the estimator, the bracket, C1's registration, the contrast scorer "
                "and the runner's parser recover planted alphas, ratios, call counts "
                "and refusals",
                PASS if ok else FAIL, "every planted row above",
                "all rows PASS", "everything this file computes, everything --run "
                "would write and everything --contrast would read")
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
    payload = json.loads(Path(args.analyse).read_text())
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
    ap.add_argument("--analyse", metavar="JSON",
                    help="score ONE measured counter run. The ratio ACROSS runs, which "
                         "is what the contrast turns on, is --contrast")
    ap.add_argument("--contrast", nargs="+", metavar="JSON", default=None,
                    help="score the RATIO across two or more counter runs, which is "
                         "the reading the extended plan takes and the one no single "
                         "payload contains. Pairs are matched inside one cache mode; "
                         "a pair nobody ran is not scored")
    ap.add_argument("--published", nargs="*", default=None,
                    help="published run directories for --bracket (default: every "
                         "results/published/*alpha-surface*)")
    ap.add_argument("--card", default="nvidia_h200",
                    choices=sorted(DATASHEET_PEAK_GBPS),
                    help="which card the plan is written for. The H200 by default, "
                         "because that is the only box whose --probe has ever come "
                         "back OPEN. It was the A100 until 2026-09-10, where the three "
                         "anchors are furthest apart and no counter route exists")
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.anchor:
        args.anchor = [(k, float(v)) for k, v in args.anchor]
    chosen = [args.dry_run, args.bracket, args.probe, args.self_test, args.run,
              bool(args.analyse), bool(args.contrast)]
    if sum(bool(c) for c in chosen) != 1:
        print("REFUSE: pick exactly one of --dry-run / --bracket / --probe / "
              "--self-test / --run / --analyse / --contrast. Running two would "
              "interleave a plan with a result and this study has been burned by "
              "exactly that.")
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
        return do_analyse(args)
    if args.probe:
        return do_probe(args)
    if args.bracket:
        return do_bracket(args)
    if args.self_test:
        return do_self_test(args)
    if args.run:
        try:
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
