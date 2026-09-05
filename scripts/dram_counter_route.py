#!/usr/bin/env python3
"""Is there a route to a DRAM counter, and what would it settle if there were?

    python scripts/dram_counter_route.py --dry-run          # the plan and its cost, off GPU
    python scripts/dram_counter_route.py --bracket          # a result, today, off GPU
    python scripts/dram_counter_route.py --self-test        # the estimator, off GPU
    python scripts/dram_counter_route.py --probe            # which route is open on THIS box
    python scripts/dram_counter_route.py --analyse c.json   # score a counter run

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

#: This file's instrument, and it is deliberately NOT
#: `moe.bench.timing.TIMING_BASIS`: nothing here times a kernel. `--bracket`
#: and `--analyse` are ARITHMETIC over numbers other runs measured, and
#: `--probe` reads the machine's configuration. Stamping the timing basis on
#: any of them would describe an apparatus that never ran, which is the exact
#: defect the audit found in two sibling `--synthetic` paths.
INSTRUMENT = "arithmetic-over-published-rows/no-kernel-timed"

#: `--probe` measures nothing at all: it asks the box what it is.
PROBE_INSTRUMENT = "machine-configuration-probe/nothing-timed"

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

#: The tile counts to profile. Four is the fewest that lets the OLS line have a
#: residual at all (three points and two parameters leaves one degree of
#: freedom); eight buys a sharper slope for about eight seconds of GPU time.
DEFAULT_TILES = (1, 2, 3, 4, 6, 8)

#: The cell the plan is written for, by default. It is the A100 mixtral arm at
#: GROUP_SIZE_M=16, which the 2026-09 evaluation singled out because its three
#: anchors are 0.452 / 0.647 / 0.705 -- the widest disagreement anywhere on the
#: surface, and the one the evaluation used to say alpha is unidentified.
DEFAULT_REPORT = (REPO / "results" / "published"
                  / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
                  / "mixtral-8x7b-bf16-r1024-g16-n64-b156b5.report.json")

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

    # C1 WHICH ANCHOR SURVIVES.
    cand = payload.get("anchors", {})
    lines = [f"{k:<10} {v:.3f}   |{v - alpha:+.3f}|   "
             f"{'SURVIVES' if abs(v - alpha) <= 0.05 else 'REFUTED'}"
             for k, v in sorted(cand.items())]
    survivors = [k for k, v in cand.items() if abs(v - alpha) <= 0.05]
    gates.append(Gate("C1", "CLAIM", "exactly one of the three anchors matches the counter",
                      PASS if len(survivors) == 1 else FAIL,
                      f"alpha_measured {alpha:.4f}; survivors {survivors or 'none'}",
                      "exactly 1 anchor within 0.05",
                      "the anchor choice. Zero survivors means all three are wrong and "
                      "the memory-branch model needs replacing, not re-anchoring; more "
                      "than one means the counter did not separate them and the cell "
                      "was badly chosen",
                      lines))

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
                   "survivors": survivors}


COUNTER_SCHEMA_TEXT = """\
{{
  "device": "{device}",       # the card slug, matching the calibration file
  "model": "{model}", "dtype": "{dtype}",
  "group_m": {group_m}, "block_n": {block_n}, "block_k": 64,
  "num_warps": 8, "num_stages": 3, "block_m": {block_m},
  "cache_control": "{cache}",           # ncu --cache-control; "all" or "none"
  "ridge": {ridge},                # THIS card's own calibration, never 160.3
  "anchors":  {anchors},
  "bracket":  {bracket},
  "rows": [ {{"n": 1, "calls": 11, "launches": 55, "dram_bytes_read": 2.84e9,
             "dram_bytes_write": 1.1e8, "l2_read_hit_pct": 4.2,
             "gpu_time_ns": 1.95e6,
             "by_kernel": {{"fused_moe_kernel": 2.81e9, "moe_sum": 3.0e7}} }}, ... ]
}}

  Every key is required. `--analyse` raises on a missing one rather than
  defaulting it: a zero that was never measured is the failure mode this whole
  file exists to avoid. `calls` is the number of fused_experts calls the
  instrument made (warmup + iters x trials), counted from the profile's own
  launch list; `launches` is every profiled kernel launch; the byte fields are
  PER CALL, the sums over all launches divided by `calls`, never by an
  assumed one.
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
        return "OPEN", ["ncu attached with no permission error: run the plan under --dry-run "
                        "and then profile. This is the decisive route."]
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
# Where the output lands, asked of git rather than assumed.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """ASK GIT whether it would keep `--out`. Never assert it, never skip it.

    `.gitignore` ignores `results/*` and re-includes only `results/published/`,
    so `--out results/bracket.json` is silently dropped by `git add -A` and the
    operator is told nothing. This repo has already lost every published plot of
    ten arms that way. `--bracket` is the mode this file's own header calls "the
    part that produces a result rather than a plan", so its JSON is exactly the
    kind of output meant to be kept.

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
    elif mode == "self-test":
        knobs = {"mode": mode}
    else:
        knobs = {"mode": mode, "model": args.model, "dtype": args.dtype,
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
    path = REPO / "moe" / "bench" / "hardware" / f"measured_{card}.yaml"
    if not path.exists():
        return None, f"no calibration at {path.relative_to(REPO)}"
    d = yaml.safe_load(path.read_text())
    tf = d["compute_dense_tflops"]["bf16"]
    bw = d["memory"]["bandwidth_tb_s"]
    return tf / bw, f"{tf:.3f} TFLOP/s / {bw:.5f} TB/s from {path.name}"


def plan_id(args) -> str:
    """Every swept parameter, or two settings collide and the second reports
    the first's numbers. The knobs here are the cell (model, dtype, G, BN, BM),
    the ladder (the tile list), and the two things a counter run varies that a
    timed run does not: the cache-control mode and the metric set."""
    return (f"{args.model}-{args.dtype}-g{args.group_m}-n{args.block_n}-m{args.block_m}"
            f"-t{'.'.join(str(t) for t in args.tiles)}-cc{args.cache_control}"
            f"-{len(NCU_METRICS)}metrics")


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
    print("  to a few percent separates these with an order of magnitude to spare, which")
    print("  is why the cell is worth renting a box for and why a one-launch probe is not.")
    print()

    print("THE COMMAND, on a box where --probe says OPEN.")
    print()
    tiles = ",".join(str(t) for t in args.tiles)
    print("  # 1. the timed ladder, unprofiled, so time and traffic come from the same cell")
    print(f"  python scripts/block_m_crossing_sweep.py --model {args.model} "
          f"--dtype {args.dtype} \\")
    print(f"      --tiles {bm} --group-m {args.group_m} --block-n {args.block_n} \\")
    print(f"      --r-max {max(args.tiles) * bm} --row-step {bm} --step-probes 0")
    print()
    print("  # 2. the same cell under the counter, one tile count per invocation.")
    print("  #    The instrument (moe/bench/timing.time_kernel) warms up for a DURATION")
    print("  #    of sustained load, --warmup in MILLISECONDS, and sizes its own")
    print("  #    iteration count from that load: there is no --iters any more (the")
    print("  #    flag is retired and ignored) and a zero warmup is REFUSED outright")
    print("  #    (timing.warm_until). So the profiled launch count is never one; it is")
    print("  #    warmup_calls + iters x trials fused_experts calls of about five kernels")
    print("  #    each, plus one L2-flush kernel per timed iteration. The smallest")
    print("  #    settings the instrument accepts keep that small (1 ms of warmup is one")
    print("  #    call of a millisecond-scale kernel; iters floors at 10; one trial), and")
    print("  #    the cell row the sweep writes carries the iters and trials it used.")
    print(f"  for n in {tiles.replace(',', ' ')}; do")
    print(f"    ncu --metrics {','.join(NCU_METRICS)} \\")
    print(f"        --replay-mode kernel --cache-control {args.cache_control} \\")
    print("        --csv --page raw --target-processes all \\")
    print(f"        --log-file counters-n$n-cc{args.cache_control}.csv \\")
    print(f"        python scripts/block_m_crossing_sweep.py --model {args.model} \\")
    print(f"          --tiles {bm} --group-m {args.group_m} --block-n {args.block_n} \\")
    print(f"          --r-max $(( n * {bm} )) --row-step $(( n * {bm} )) --step-probes 0 \\")
    print("          --warmup 1 --trials 1 --cell-budget-ms 1")
    print("  done")
    print()
    print("  # 3. reduce, then score. Sum every byte metric PER KERNEL NAME over all")
    print("  #    profiled launches and divide by the ACTUAL call count: `calls` is the")
    print("  #    number of moe_align_block_size launches in the CSV (exactly one per")
    print("  #    fused_experts call; fused_moe_kernel runs twice per call, up and down),")
    print("  #    and it must equal the row's iters x trials plus the warmup's calls, so")
    print("  #    REFUSE a CSV whose count is not above iters x trials. Write the per-call")
    print("  #    figures as dram_bytes_read etc. and the count as `calls`. Never divide")
    print("  #    by an assumed 1: the instrument decides the count, the row records it.")
    print("  python scripts/dram_counter_route.py --analyse counters.json")
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
    # kernels, plus ten flush kernels, per invocation. A bracket, not a
    # promise: a sub-millisecond kernel warms in more than one call.
    calls_per_invocation = 1 + 10
    launches = len(args.tiles) * 2 * calls_per_invocation * 5
    flushes = len(args.tiles) * 2 * 10
    print("COST.")
    print(f"  {len(args.tiles)} tile counts x 2 cache modes = {len(args.tiles) * 2} profiled "
          f"invocations of at least {calls_per_invocation} fused_experts calls each,")
    print(f"  about {launches} profiled kernel launches plus {flushes} L2-flush launches.")
    print(f"  The largest cell is n={n_max} ({n_max * bm} rows per expert), which the timed")
    print("  sweep measures in single-digit milliseconds; ncu replay and its save/restore")
    print(f"  of the {W / 1e9:.1f} GB weight buffers dominate, at seconds per launch. Budget")
    print("  an hour of GPU time and two pod-hours end to end, not the fifteen minutes")
    print("  the one-launch recipe used to promise. At A100 spot rates that is still a")
    print("  few dollars. The cost of this experiment has never been the money.")
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

def do_self_test(args) -> int:
    """Plant an alpha, synthesise the counter rows the model implies, and check
    the estimator returns it. Then plant a ladder and check the bracket contains
    the alpha that generated it.

    This is the check that the analysis half is not itself the source of a
    number. `block_m_crossing_sweep.py` has the same shape for the same reason:
    an estimator that has never been run against a known answer is an assertion.
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
    print(f"\n  SELF TEST {'PASS' if ok else 'FAIL'}")
    # A SELF TEST IS A VALIDITY GATE ON THE ANALYSIS HALF, so its failure is
    # INVALID (3) and not CLAIM_FAIL (1): an estimator that cannot recover a
    # planted alpha has not refuted anything about the world, it has said that
    # nothing this file computes may be quoted. It returned 1, which the driver
    # files as a finished result and never retries. The gate prints its RESULT
    # line like every other gate in this file, so the log and the code agree.
    gate = Gate("S1", "VALIDITY", "the estimator and the bracket recover planted alphas",
                PASS if ok else FAIL, "every planted row above",
                "all rows PASS", "everything this file computes")
    print(gate.result_line())
    return exit_codes.classify([gate.scored()])


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
                    help="plant an alpha and check the estimator returns it. No GPU.")
    ap.add_argument("--analyse", metavar="JSON", help="score a measured counter run")
    ap.add_argument("--published", nargs="*", default=None,
                    help="published run directories for --bracket (default: every "
                         "results/published/*alpha-surface*)")
    ap.add_argument("--card", default="nvidia_a100_sxm4_80gb",
                    choices=sorted(DATASHEET_PEAK_GBPS),
                    help="which card the plan is written for. The A100 by default, "
                         "because that is where the three anchors are furthest apart")
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--group-m", type=int, default=16,
                    help="16, not 1. The anchor disagreement is a G>1 phenomenon: at n=1 "
                         "there is one M-tile per expert, so a G>1 swizzle group spans G "
                         "different experts and can reuse nothing, which is the mechanism "
                         "that puts the n=1 tread above the branch it anchors")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-m", type=int, default=32,
                    help="32 by default: the four A100 fits whose published alpha is "
                         "physically impossible are all BLOCK_M=32 with a swizzle")
    ap.add_argument("--tiles", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=DEFAULT_TILES, help="tile counts per expert to profile")
    ap.add_argument("--cache-control", default="all", choices=("all", "none"),
                    help="ncu --cache-control. 'all' is ncu's own default and flushes "
                         "every cache before each replay pass")
    ap.add_argument("--report", default=str(DEFAULT_REPORT),
                    help="the PUBLISHED report the plan registers its predictions "
                         "against. Defaults to the A100 mixtral G=16 arm, the cell where "
                         "the three anchors are furthest apart (0.452 / 0.647 / 0.705)")
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
    chosen = [args.dry_run, args.bracket, args.probe, args.self_test, bool(args.analyse)]
    if sum(bool(c) for c in chosen) != 1:
        print("REFUSE: pick exactly one of --dry-run / --bracket / --probe / "
              "--self-test / --analyse. Running two would interleave a plan with a "
              "result and this study has been burned by exactly that.")
        return exit_codes.REFUSED
    if args.analyse:
        return do_analyse(args)
    if args.probe:
        return do_probe(args)
    if args.bracket:
        return do_bracket(args)
    if args.self_test:
        return do_self_test(args)
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
