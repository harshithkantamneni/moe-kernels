#!/usr/bin/env python3
"""Measure `alpha` by ABLATION, without the byte model. STUDY.md item 4.

    python scripts/alias_ablation.py                    # plan + prediction, no GPU
    python scripts/alias_ablation.py --run              # the pod run
    python scripts/alias_ablation.py --replay <dir>     # re-report, no GPU
    python scripts/alias_ablation.py --synthetic refit  # exercise the gates, no GPU
    python scripts/alias_ablation.py --run --compute dot --replicates 15

WHY THIS EXISTS. `alpha` is the cost of an extra M-tile as a fraction of a fresh
weight read. One expert holding `r` rows is scheduled as `ceil(r/BM)` M-tiles;
the first reads that expert's weight block in full and each later one costs
`alpha` of a fresh read, because L2 absorbs part of the re-read:

    Q(r) = 1 + alpha (ceil(r/BM) - 1),   AI(r) = (2r/b) / Q(r) -> 2 BM / (alpha b)

That bound is why alpha is not a nuisance parameter: at 0.558 it puts BLOCK_M of
16, 32 and 64 BELOW the measured ridge band of 160.3-176.2, and vLLM runs
BLOCK_M=16 through the whole decode range, so a decode-configured MoE kernel can
never reach its compute roof. The whole tile-corrected roofline rests on it.

AND IT RESTS ON ONE REGRESSION AGAINST A BYTE MODEL THAT HAS NO TILE TERM.
`scripts/alpha_refit.py` fits alpha from `implied_traffic_ratio`, which is
time x bandwidth / COMPULSORY BYTES, and C4 (`docs/FINDINGS.md`) is a confirmed
finding that the compulsory-byte ruler was wrong by 1.85% until it was fixed.
A number that decides a paper's headline should not depend on one estimator over
one derived column. This script measures the same physical quantity by a route
that touches neither: no compulsory bytes, no calibrated bandwidth, no ridge, no
fitted intercept, no `implied_traffic_ratio`.

THE METHOD, quoted from `docs/STUDY.md` order-of-work item 4, never done since
August: "alias B by taking the tile offset modulo so every iteration reloads the
same tile (loads execute, L2 hits, no HBM traffic, nothing folds since values
are runtime); acc += tl.sum(b) + tl.sum(a) to keep loads live on the compute
side."

WHY THAT MEASURES ALPHA. Run the same grouped-GEMM access pattern twice at n
M-tiles per expert. NORMAL reads every expert's weight block, once per M-tile,
so its HBM weight traffic is `W (1 + alpha (n-1))` in units of one full pass.
ALIASED points every weight load at ONE resident tile, so its weight loads all
hit L2 and its HBM weight traffic is essentially zero. Everything else --
activation reads, output writes, arithmetic, launch, the grid, the instruction
stream -- is IDENTICAL, so it cancels in the difference:

    D(n) = T_normal(n) - T_aliased(n) = W (1 + alpha (n-1))
    D(1) = W                                       <- one full weight read
    alpha = (D(n)/D(1) - 1) / (n-1)

The n=1 rung supplies the denominator directly, which is what makes this
self-calibrating: W never has to be predicted from bytes and a bandwidth, it is
measured in the same units, in the same session, by the same clock.

AND THAT SUBTRACTION IS ONLY VALID IF L2 AND HBM COSTS ADD, WHICH THEY MAY NOT.
The aliased variant issues exactly the same loads; what it does not do is MISS.
So both variants push `n W` bytes through L2, and whether that common cost
cancels depends on how the two units compose. If they add, it cancels and the
line above is exact. If the kernel instead runs at `max(L2, HBM)`, which is what
a streaming kernel with enough parallelism does, then

    D(n) = W(1 + alpha(n-1)) - r W n,   fitting to   (alpha - r)/(1 - r)

where `r` is the aliased ladder's per-tile cost as a fraction of one weight
read. On an H200, whose L2 bandwidth is only about twice HBM's, `r` is plausibly
0.4 to 0.6 -- and at r = 0.55 a true alpha of 0.558 fits to 0.018. THAT IS THE
SHAPE OF A CONFIDENT WRONG NUMBER, and it lands on top of the retracted 0.10.
This trap is not hypothetical arithmetic; it is what the naive version of this
experiment would have reported.

SO THE ANSWER IS A BRACKET, FROM TWO ESTIMATORS THAT FAIL IN OPPOSITE
DIRECTIONS. The second one fits `T_normal(n)` directly, taking the fixed launch
cost from the ALIASED ladder's own n=0 intercept rather than as a free
parameter, and is exact under max() while biased UP to `(alpha + r)/(1 + r)`
under addition. For any alpha <= 1:

    (alpha - r)/(1 - r)  <=  alpha  <=  (alpha + r)/(1 + r)
      DIFFERENCE estimator          DIRECT estimator

Both are reported, the interval between them is what the report quotes, and `r`
is measured and printed because it is the only thing that sets the width. A run
whose interval is too wide to separate 0.10 from 0.33 from 0.558 says NOT
TESTABLE and names the reason, rather than picking one.

HOW THE ALIASING IS DONE, and why nothing folds. NOT with a constexpr flag and
NOT with a compile-time modulo. The two variants are THE SAME COMPILED KERNEL,
launched with three different runtime integers, each of which is a STRIDE:

    stride_be_eff      N*K  ->  0    every M-tile reads expert 0's block
    stride_bn_blk_eff  BLOCK_N*K -> 0   every N-tile reads column block 0
    b_k_advance        BLOCK_K -> 0    the K loop never advances the B pointer

In NORMAL those three are the real strides; in ALIASED every weight load lands
on the same BLOCK_K x BLOCK_N tile. The kernel emits one add per loop iteration
either way, so the two variants are byte-identical machine code and differ only
in the VALUES of three scalars the compiler cannot see. Dead-code elimination
cannot apply to one and not the other, because there is only one code path. A
modulo would have cost an integer remainder per iteration inside the K loop,
which is real work charged to both sides and is avoidable; the pointer-advance
form costs nothing.

THEY ARE STRIDES AND NOT 1/0 FLAGS FOR A REASON, and it is a trap that would
have silently destroyed the experiment. Triton SPECIALISES integer kernel
arguments: an argument whose value is exactly 1 is compiled in as a constant,
and an argument divisible by 16 gets a divisibility hint. Written the obvious
way -- `b_expert_scale` of 1 for normal and 0 for aliased -- the two variants
would land in DIFFERENT specialisations, the normal one would fold `off_e * 1`
at compile time, and the "same compiled kernel" guarantee this whole design
rests on would be false while every table still printed numbers. Every one of
the three scalars is therefore a large stride in the normal case and 0 in the
aliased case, so both take the divisible-by-16 path and neither takes the
equal-to-1 path. The ISA gate CHECKS that they compiled to one kernel rather
than trusting this paragraph.

THE HAZARD IS NAMED IN THE REPO AND IS NOT NEGOTIABLE. Aliasing a load is
exactly the shape a compiler folds: hoist the invariant load out of the loop and
the measurement reports the optimiser rather than the cache, and it reports it
as a beautiful clean alpha of nearly zero. Three independent checks, all of
which must pass before a number is quoted:

  1. ONE KERNEL. The PTX of the launch used for NORMAL and the PTX of the launch
     used for ALIASED are compared hash for hash, and the global-memory
     instruction counts are printed side by side. They are equal by
     construction, which is a stronger guarantee than two compilations that
     happen to agree, and the gate still checks it rather than asserting it.
  2. THE FOLD IS DEMONSTRATED, not assumed away. A THIRD kernel is compiled with
     the aliasing as `tl.constexpr` -- the naive way to write this experiment --
     and its instruction counts are printed in the same table. If the constexpr
     variant issues fewer global loads than the runtime one, that is the fold
     happening in front of the reader, and it is the evidence that the runtime
     design was necessary. If it does NOT fold, that is reported too.
  3. THE OUTPUT IS CHECKED, both ways. NORMAL must reproduce a torch reference,
     which proves it really read every expert's whole weight block. ALIASED must
     reproduce a DIFFERENT closed-form reference -- `(K/BLOCK_K)` copies of the
     one aliased tile -- which proves the aliasing did exactly what was intended
     and is not a silent no-op that would make D(n) pure noise.

Counting `ld.global` alone would have read ZERO on this kernel. Triton pipelines
global-to-shared copies as `cp.async.cg.shared.global` at num_stages > 1, and on
Hopper can use `cp.async.bulk.tensor`. Every global-memory mnemonic is counted
and reported separately, and the gate is on the total.

THE PREDICTION, stated here so it can fail, and printed before anything runs:

  P1  alpha = 0.558, 90% band 0.529-0.588 (today's refit, 10,813 rows).
      The retracted repo value is 0.10 and TEMPO (arXiv:2608.13057) fits 0.33.
      The report says PASS or FAIL against P1 and names which of the three
      candidate values this ablation supports.

  P2  THE MECHANISM, and it is the more interesting half. At GROUP_SIZE_M=1 the
      reuse distance between M-tile i and M-tile i+1 of one expert is exactly
      one pass over that expert's weight block, so alpha should be a function of
      PER-EXPERT BYTES against L2 and not a universal constant:

          mixtral-8x7b      235.0 MB/expert   >> L2  ->  alpha near 1
          deepseek-v3        58.7 MB/expert   ~= L2  ->  alpha intermediate
          qwen2-57b-a14b     36.7 MB/expert    < L2  ->  alpha small
          deepseek-v2-lite   11.5 MB/expert   << L2  ->  alpha near 0

      All four are run, because those four models ARE the pool the 0.558 was
      fitted over. If P2 holds then 0.558 is a pool average of a step function
      and 0.10, 0.33 and 0.558 can all be right about different pools, which
      would explain a threefold disagreement that has stood since August.

WHAT ELSE DIFFERS BETWEEN ALIASED AND NORMAL, being adversarial about it, since
D(n) charges the whole difference to weight traffic:

  * L2 SERVICE, and the cache-set distribution that makes it worse. Both
    variants push `n W` bytes through L2 and the aliased variant pins every one
    of them to a single BLOCK_K x BLOCK_N tile, so its loads land on a handful
    of L2 slices and are served more slowly than the aggregate L2 bandwidth
    would suggest. BOUNDED BY THE BRACKET above: whatever it costs appears in
    the aliased ladder's own slope, which is `r`, and the two estimators bracket
    the truth for any composition between a sum and a max. Narrowing it means
    spreading the aliased tile across more slices, which this design does not do
    because a spread alias has no closed form to check the output against.
  * L2 CAPACITY. The aliased variant leaves L2 almost entirely to activations
    and outputs. Bounded by design rather than hoped away: at BLOCK_M=16 the
    activation re-stream is `n BLOCK_M / N` of the weight stream, which the plan
    prints per rung and which is under 5% at every shipped rung. The output is
    write traffic and is identical in both variants.
  * TLB AND PAGE BEHAVIOUR. The aliased variant touches one page of B where the
    normal one touches the whole tensor. What bounds it is the L2-RESIDENT
    CONTROL (`--control`): the same ladder on a geometry whose PER-EXPERT block
    fits in L2, so NORMAL has no HBM re-read to save and the tile-count slope
    has nothing legitimate to be. Its alpha must come out consistent with zero,
    and whatever it does come out at is the size of every extra-tile cost that
    is not weight traffic.
  * ANY COST THAT SCALES WITH THE EXTRA-TILE COUNT AND IS NOT A WEIGHT RE-READ
    is absorbed into alpha by construction, because that is the regressor. Said
    plainly because `scripts/group_m_alpha_sweep.py` had to say it too.

WHAT THIS IS NOT. It is not vLLM's `fused_moe_kernel`. It is a kernel with
vLLM's B-pointer arithmetic, vLLM's GROUP_SIZE_M swizzle, vLLM's [E, N, K]
weight layout and vLLM's tile constants, whose reduction is replaced by
`acc += tl.sum(a) + tl.sum(b)` on the study's own instruction. That replacement
is deliberate and it is what makes the estimator unbiased: with a real `tl.dot`
the aliased variant becomes compute-bound while the normal one stays
memory-bound, so D(n) loses one copy of the per-tile compute cost and the fitted
alpha is biased DOWN. `--compute dot` runs it that way anyway, prints the bound
and refuses to quote the result as P1's answer. The accumulator keeps its full
BLOCK_M x BLOCK_N float32 shape in both modes, so register pressure, occupancy
and therefore the L2 working set are unchanged between them.

EXIT CODES, because a refutation is a result and not an error:
    0  the run completed and every runnable gate passed
    1  the run completed and a gate FAILED: the prediction is refuted
    2  usage error
    3  cannot run here (no GPU, no triton, not enough memory); nothing measured
    4  the run completed but the design did not identify alpha: not testable
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.spec import MODEL_CONFIGS  # noqa: E402

# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------

#: M-tiles per expert. Powers of two so rows per expert is an exact multiple of
#: BLOCK_M at every rung and the padding term is EXACTLY zero, which is what
#: leaves the tile count as the only thing that changes between rungs. The
#: ladder is the one STUDY.md asks for; four rungs is also the minimum that can
#: show D(n) is affine in (n-1) rather than merely fitted through two points.
TILE_LADDER = (1, 2, 4, 8)

#: The four models the published alpha was fitted over, so a pooled number here
#: is comparable with a pooled number there. They also span 20x in per-expert
#: bytes, which is what makes P2 testable at all.
DEFAULT_MODELS = ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                  "deepseek-v3")

#: What vLLM actually runs through the decode range, and the only tile at which
#: the multi-tile and memory-bound windows overlap generously: multi-tile needs
#: rows per expert above BLOCK_M, memory bound needs rows per expert below the
#: ridge, and at BLOCK_M=16 the whole ladder (16, 32, 64, 128 rows per expert)
#: sits inside 160.3.
DEFAULT_BLOCK_M = 16

#: Everything else about the tile, held fixed so the ONLY thing that varies
#: between the two variants is three runtime integers. GROUP_M=1 is the
#: load-bearing entry and is chosen, not inherited: at 1 the reuse distance
#: between consecutive M-tiles of one expert is exactly one pass over that
#: expert's weight block, which is what makes P2's per-expert-bytes prediction a
#: prediction rather than a hope. It is also the setting the published pool
#: mostly sits at -- `alpha_refit` split by GROUP_SIZE_M reads 0.570 at 1 and
#: 0.488 at 16 -- so 0.570 is the closer comparison and the report says so.
FIXED_TILE = {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1,
              "num_warps": 8, "num_stages": 3}

#: How the loads are kept live. "sum" is what STUDY.md item 4 prescribes and is
#: the unbiased estimator; "dot" is the real GEMM reduction and is biased low.
COMPUTE_MODES = ("sum", "dot")

#: Timed launches per (rung, variant). Nine because the estimator divides two
#: differences of medians, so the noise on alpha is roughly four times the noise
#: on one timing, and nine samples put a median's standard error at about a
#: third of a single sample's.
DEFAULT_REPLICATES = 9

#: The L2-resident control geometry, and every number in it is chosen.
#:
#: 16.0 MiB PER EXPERT, comfortably inside any L2 this study has run on (H200
#: 50 MiB, A100 40 MiB), so at GROUP_SIZE_M=1 the re-read between one expert's
#: consecutive M-tiles HITS and NORMAL has no HBM re-read to save. Its alpha
#: must therefore come out near zero, and whatever it does come out at is the
#: size of everything that scales with the tile count and is not weight traffic:
#: TLB and page behaviour, cache-set distribution, and any code-path difference.
#:
#: 512 MiB IN TOTAL, not L2-resident, because the control has to keep a real
#: first-pass W. A control whose whole tensor fitted in L2 would have no W to
#: divide by and no alpha at all.
#:
#: N = 4096 so the activation stream stays at 3% of the weight stream at the top
#: rung, which is the same bound the real models get. A narrower control fails
#: the activation-fraction preflight, which is the gate catching a control that
#: could not have controlled for anything.
CONTROL_MODEL = "control-l2-resident"
CONTROL_EXPERTS = 32
CONTROL_K = 2048
CONTROL_N = 4096


# --------------------------------------------------------------------------
# what the answer is being scored against
# --------------------------------------------------------------------------

#: Today's refit and its 90% band, from `docs/FINDINGS.md` "The tile-corrected
#: roofline" and the 2026-09-01 rewrite of C1: 0.558 over 10,813 rows.
REFIT_ALPHA = 0.558
REFIT_BAND = (0.529, 0.588)

#: The two rivals. Cross-checked against `scripts/alpha_refit.py`'s own
#: constants at import time, and by a test, so the three numbers this report
#: scores against cannot drift away from the estimator's.
REPO_RETRACTED_ALPHA = 0.10
TEMPO_ALPHA = 0.33

#: The GROUP_SIZE_M=1 split of the published refit. Printed beside the pooled
#: 0.558 because this ablation runs at GROUP_M=1 and the pooled figure does not.
REFIT_ALPHA_AT_GROUP_M_1 = 0.570

CANDIDATES = (
    ("this repo, retracted", REPO_RETRACTED_ALPHA),
    ("TEMPO arXiv:2608.13057", TEMPO_ALPHA),
    ("today's refit", REFIT_ALPHA),
)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

#: The placebo. Two launches of the IDENTICAL configuration must differ by less
#: than this fraction of D(1), or the timing noise floor is a large enough part
#: of the signal that no alpha read off it means anything.
PLACEBO_MAX_FRACTION = 0.10

#: D(1) must be at least this fraction of T_normal(1). Below it the weight read
#: is not what the kernel is doing, and the difference is measuring something
#: else with a weight-read label on it.
MIN_SIGNAL_FRACTION = 0.25

#: D(n) must be affine in (n-1) to at least this R^2, or `W(1 + alpha(n-1))` is
#: the wrong functional form and its slope-over-intercept is not alpha.
MIN_LINEARITY_R2 = 0.97

#: A reported interval wider than this cannot separate the three candidate
#: values, which span 0.10 to 0.558. Half the widest gap between adjacent
#: candidates (0.228) is the loosest interval that could still put exactly one
#: of them inside. It is a NOT-TESTABLE threshold and not a failure: an
#: unresolved measurement is not a refutation.
MAX_BAND_WIDTH = 0.11

#: How far the L2-resident control's BRACKET may sit from zero. Its weight
#: re-reads hit L2, so there is no HBM re-read for the aliasing to remove and
#: its alpha must be consistent with zero. Anything beyond this is the size of
#: every extra-tile-scaling difference that is NOT weight traffic, and it is
#: charged against the real alphas.
CONTROL_MAX_ALPHA = 0.15

#: Relative RMS error allowed between a variant's output and its closed-form
#: reference. Both sides sum the same bf16 values in a different order, so the
#: floor is summation order, not correctness; 1e-3 is three orders above it.
CORRECTNESS_RTOL = 1e-3

#: Activation re-stream as a fraction of the weight stream, above which the L2
#: capacity confound stops being negligible. n*BLOCK_M/N at the shipped rungs is
#: under 0.01 everywhere, so this is a tripwire for someone changing the shape.
MAX_ACTIVATION_FRACTION = 0.05

#: Clock drift across a rung, above which the rung is flagged. Matches
#: `moe/bench/timing.clock_drift`'s own threshold.
CLOCK_DRIFT_LIMIT = 0.05


# --------------------------------------------------------------------------
# the rungs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung:
    """One (model, tile count) cell, with everything the report needs.

    `rows_per_expert` is an EXACT multiple of BLOCK_M by construction: the
    M-tile-to-expert map is built directly rather than sampled from a routing
    distribution, so padding is exactly zero and the tile count is the only
    thing that moves across the ladder. `moe/bench/published.py`'s rows cannot
    do that -- uniform routing is SAMPLED per replicate and its tile count
    varies within a cell, which is the correction FINDINGS.md had to make to the
    staircase table on 2026-09-01.
    """

    model: str
    tiles: int
    block_m: int
    experts: int
    k: int
    n: int
    block_n: int
    block_k: int
    group_m: int
    control: bool = False

    @property
    def key(self) -> str:
        return f"{self.model}|t{self.tiles}|bm{self.block_m}"

    @property
    def rows_per_expert(self) -> int:
        return self.tiles * self.block_m

    @property
    def total_rows(self) -> int:
        return self.experts * self.rows_per_expert

    @property
    def per_expert_bytes(self) -> int:
        """One expert's weight block, in bytes. The reuse distance at GROUP_M=1
        and therefore the quantity P2 says alpha is a function of."""
        return self.n * self.k * 2

    @property
    def weight_bytes(self) -> int:
        return self.experts * self.per_expert_bytes

    @property
    def activation_bytes(self) -> int:
        return self.total_rows * self.k * 2

    @property
    def output_bytes(self) -> int:
        """float32, so the correctness check is not fighting bf16 rounding."""
        return self.total_rows * self.n * 4

    @property
    def footprint_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes + self.output_bytes

    @property
    def activation_fraction(self) -> float:
        """The L2-capacity confound's bound: activation stream over weight
        stream. Equals n*BLOCK_M/N once the E and K terms cancel."""
        return self.activation_bytes / self.weight_bytes

    @property
    def arith_intensity(self) -> float:
        """Compulsory FLOP/byte in `dot` mode, which is `2 r / b` = rows per
        expert in bf16. Meaningless in `sum` mode, where there is no matmul and
        the kernel is memory bound by construction; reported anyway because the
        `dot` mode gate reads it."""
        return float(self.rows_per_expert)

    @property
    def num_pid_m(self) -> int:
        return self.experts * self.tiles

    @property
    def num_pid_n(self) -> int:
        return self.n // self.block_n

    @property
    def programs(self) -> int:
        return self.num_pid_m * self.num_pid_n

    @property
    def k_iters(self) -> int:
        return self.k // self.block_k


def rung_for(model: str, tiles: int, block_m: int, tile: dict) -> Rung:
    """A rung from a real model geometry, shaped as vLLM's up-projection.

    N is `2 * intermediate_size` and K is `hidden_size`, which is w1's shape in
    vLLM's `fused_experts`: the gate and up halves are one tensor. Using the
    real w1 rather than a square stand-in matters because P2 is a claim about
    PER-EXPERT BYTES against L2, and a stand-in would be a claim about a shape
    no model has.
    """
    if model == CONTROL_MODEL:
        experts, k, n = CONTROL_EXPERTS, CONTROL_K, CONTROL_N
    else:
        cfg = MODEL_CONFIGS[model]
        experts, k, n = cfg.num_experts, cfg.hidden_size, 2 * cfg.intermediate_size
    return Rung(model=model, tiles=tiles, block_m=block_m, experts=experts,
                k=k, n=n, block_n=tile["BLOCK_N"], block_k=tile["BLOCK_K"],
                group_m=tile["GROUP_M"], control=(model == CONTROL_MODEL))


@dataclass(frozen=True)
class Design:
    models: tuple[str, ...]
    tiles: tuple[int, ...]
    block_m: int
    tile: dict
    compute: str
    replicates: int
    rungs: tuple[Rung, ...]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({
            "models": list(self.models), "tiles": list(self.tiles),
            "block_m": self.block_m, "tile": self.tile,
            "compute": self.compute, "replicates": self.replicates,
            # THE GEOMETRIES, not just the model names. `--replay` and the
            # resume path key on `rung.key`, which is model|tiles|block_m, so a
            # change to the control constants would silently reuse records
            # measured on a different shape under the same key. The fingerprint
            # is what sends a changed design to a different directory.
            "shapes": [[r.model, r.experts, r.k, r.n] for r in self.rungs],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    def rungs_for(self, model: str) -> list[Rung]:
        return [r for r in self.rungs if r.model == model]


def build_design(args) -> Design:
    models = tuple(args.models)
    if args.control:
        models = models + (CONTROL_MODEL,)
    tile = dict(FIXED_TILE)
    tile["GROUP_M"] = args.group_m
    tile["BLOCK_N"] = args.block_n
    tile["BLOCK_K"] = args.block_k
    rungs = tuple(rung_for(m, t, args.block_m, tile)
                  for m in models for t in args.tiles)
    return Design(models=models, tiles=tuple(args.tiles), block_m=args.block_m,
                  tile=tile, compute=args.compute, replicates=args.replicates,
                  rungs=rungs)


# --------------------------------------------------------------------------
# gates as a value, not as an exception
# --------------------------------------------------------------------------

@dataclass
class Gate:
    """One numeric verdict. `ok=None` means the data cannot answer it.

    The bool coercion is not cosmetic and the bug it fixes was live in
    `scripts/group_m_alpha_sweep.py` for one run: a numpy bool is not `False`,
    so `g.ok is False` skipped a gate that had printed FAIL, and the verdict
    disagreed with the table above it. Anything derived from a fitted number can
    arrive here as a numpy scalar.
    """

    name: str
    ok: bool | None
    detail: str

    def __post_init__(self) -> None:
        self.ok = None if self.ok is None else bool(self.ok)

    @property
    def label(self) -> str:
        return {True: "PASS", False: "FAIL", None: "NOT TESTABLE"}[self.ok]


def preflight(design: Design, l2_bytes: int) -> list[Gate]:
    """Refuse a design that cannot answer the question, before it is paid for."""
    gates: list[Gate] = []

    bad_shape = [r.key for r in design.rungs
                 if r.n % r.block_n or r.k % r.block_k
                 or r.num_pid_m % r.group_m]
    gates.append(Gate(
        "every rung divides exactly, so no mask and no padding",
        not bad_shape,
        "N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero"
        if not bad_shape else
        f"{len(bad_shape)} rungs do not divide: {bad_shape[:3]}. A masked tile "
        "loads a partial block and the two variants would stop being the same "
        "amount of work."))

    gates.append(Gate(
        "the ladder has at least three rungs, so D(n) can be shown affine",
        len(design.tiles) >= 3,
        f"tile ladder {list(design.tiles)}; two rungs fit a line through two "
        "points and can never contradict the form"))

    gates.append(Gate(
        "the ladder starts at one tile, which is the only source of W",
        min(design.tiles) == 1,
        f"smallest rung is {min(design.tiles)} tiles. D(1) is the denominator "
        "of every alpha here; without it W has to come from a byte model, "
        "which is the thing this experiment exists to avoid"))

    worst = max((r.activation_fraction for r in design.rungs), default=0.0)
    gates.append(Gate(
        "the activation stream is small against the weight stream",
        worst <= MAX_ACTIVATION_FRACTION,
        f"worst rung streams {worst * 100:.2f}% as many activation bytes as "
        f"weight bytes (limit {MAX_ACTIVATION_FRACTION * 100:.0f}%). This "
        "bounds the L2-capacity confound: the aliased variant frees L2, and "
        "what it could free it for is the activation re-stream"))

    if design.compute == "dot":
        over = [r.key for r in design.rungs
                if not r.control and r.arith_intensity >= 160.3]
        gates.append(Gate(
            "in dot mode every rung stays below the ridge band",
            not over,
            "all rungs below 160.3 FLOP/byte" if not over else
            f"{len(over)} rungs are compute bound: {over[:3]}. A compute-bound "
            "rung pays for extra tiles in padded arithmetic, not in traffic"))

    if l2_bytes:
        spread = sorted({r.per_expert_bytes for r in design.rungs
                         if not r.control})
        both = spread and spread[0] < l2_bytes < spread[-1]
        gates.append(Gate(
            "P2 is testable: the models straddle L2",
            bool(both),
            f"per-expert weight blocks run {spread[0] / 2**20:.1f} to "
            f"{spread[-1] / 2**20:.1f} MiB against an L2 of "
            f"{l2_bytes / 2**20:.1f} MiB"
            if spread else "no non-control rungs"))
    return gates


# --------------------------------------------------------------------------
# the estimator. Deliberately NOT alpha_refit's.
# --------------------------------------------------------------------------

@dataclass
class AlphaFit:
    """alpha from the ablation, as a BRACKET, and everything that qualifies it.

    TWO ESTIMATORS, BECAUSE ONE OF THEM IS ALWAYS WRONG. The aliased variant
    issues exactly the same loads as the normal one; what it does not do is miss
    L2. So both variants pay the L2 service cost of `n W` bytes, and whether
    that common cost CANCELS in the difference depends on how the two units
    compose:

        if L2 and HBM service ADD          D(n) = W(1 + alpha(n-1)) exactly,
                                           and the DIFFERENCE estimator is right
        if the kernel runs at max(L2, HBM) D(n) = W(1+alpha(n-1)) - r W n,
                                           and the difference estimator returns
                                           (alpha - r)/(1 - r), biased DOWN

    where `r` is the aliased ladder's per-tile cost as a fraction of one weight
    read. A streaming kernel is closer to max() than to a sum, and r on an H200
    is plausibly 0.4 to 0.6, at which the difference estimator returns about
    0.02 for a true alpha of 0.558. THAT IS THE SHAPE OF A CONFIDENT WRONG
    NUMBER, and it would land on top of the retracted 0.10.

    The DIRECT estimator is the other extreme: fit `T_normal(n)` itself, with
    the fixed launch cost taken from the aliased ladder's own n=0 intercept. It
    is exact under max() and biased UP to (alpha + r)/(1 + r) under addition.

    For any alpha <= 1 the two bracket the truth:

        (alpha - r)/(1 - r)  <=  alpha  <=  (alpha + r)/(1 + r)

    so this dataclass reports both and the report quotes the interval. The
    bracket is narrow exactly when the aliased ladder is cheap, which is
    MEASURED rather than assumed, and a run whose bracket is too wide to
    separate the candidates says NOT TESTABLE rather than picking one.

    `alpha` remains the difference estimator, so it is always the LOW end.
    """

    alpha: float | None
    w_ms: float | None
    alpha_direct: float | None = None
    fixed_ms: float | None = None
    aliased_slope: float | None = None
    per_rung: dict[int, float] = field(default_factory=dict)
    r2: float | None = None
    diffs: dict[int, float] = field(default_factory=dict)
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.alpha is not None

    @property
    def bracket(self) -> tuple[float, float] | None:
        if self.alpha is None:
            return None
        if self.alpha_direct is None:
            return (self.alpha, self.alpha)
        return (min(self.alpha, self.alpha_direct),
                max(self.alpha, self.alpha_direct))

    @property
    def width(self) -> float:
        span = self.bracket
        return 0.0 if span is None else span[1] - span[0]

    @property
    def l2_share(self) -> float | None:
        """The aliased ladder's per-tile cost over one weight read. This is `r`,
        and it is what sets the bracket's width."""
        if self.aliased_slope is None or not self.w_ms:
            return None
        return self.aliased_slope / self.w_ms


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float, float] | None:
    """Least squares of y on x = tiles - 1. Returns (intercept, slope, R^2).

    x is `tiles - 1` and not `tiles` so that the intercept IS the one-tile
    value, which is the quantity every estimator here divides by.
    """
    if len(points) < 2:
        return None
    xs = [float(n - 1) for n, _ in points]
    ys = [float(v) for _, v in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2
                 for x, y in zip(xs, ys, strict=True))
    return intercept, slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def fit_alpha(diffs: dict[int, float]) -> AlphaFit:
    """The DIFFERENCE estimator: alpha from `D(n) = W (1 + alpha (n-1))`.

    Slope over intercept, and both come out of the same fit, so the units of
    time cancel: no bandwidth, no byte count and no calibration enters. That is
    the entire point of the ablation route.

    Returns an unfitted result rather than raising when the intercept is not
    positive. A non-positive W means the aliased variant was not faster at one
    tile, so there is no weight read to be a fraction OF, and a ratio computed
    through it would be a large confident number with no meaning.
    """
    points = sorted((n, d) for n, d in diffs.items() if d == d)
    line = _fit_line(points)
    if line is None:
        return AlphaFit(None, None,
                        why="fewer than two rungs, or one tile count only")
    intercept, slope, r2 = line
    if intercept <= 0:
        return AlphaFit(
            None, None, diffs=dict(points),
            why=f"fitted W is {intercept:.4f} ms, not positive: the aliased "
                "variant was not faster, so there is no weight read to divide by")
    base = diffs.get(1)
    per_rung = {}
    if base and base > 0:
        per_rung = {n: (d / base - 1.0) / (n - 1) for n, d in points if n > 1}
    return AlphaFit(alpha=slope / intercept, w_ms=intercept, per_rung=per_rung,
                    r2=r2, diffs=dict(points))


def fit_bracket(samples: dict[int, dict[str, list[float]]]) -> AlphaFit:
    """Both estimators, so the answer is an interval that contains the truth.

    The fixed cost is the ALIASED ladder's n=0 intercept, `I_a - S_a`, which is
    launch and dispatch and nothing else: the aliased variant's own per-tile
    cost is its slope, and extrapolating it out leaves what a zero-tile launch
    would have cost. Taking it from the aliased ladder rather than fitting it as
    a free parameter of the normal one is what keeps the direct estimator from
    being "a fitted intercept", which is the thing this whole experiment exists
    to avoid.
    """
    fit = fit_alpha(differences(samples))
    if not fit.ok:
        return fit
    aliased = sorted((n, statistics.median(v["aliased"]))
                     for n, v in samples.items() if v.get("aliased"))
    normal = sorted((n, statistics.median(v["normal"]))
                    for n, v in samples.items() if v.get("normal"))
    line_a, line_n = _fit_line(aliased), _fit_line(normal)
    if line_a is None or line_n is None:
        return fit
    fixed = line_a[0] - line_a[1]
    w_direct = line_n[0] - fixed
    fit.fixed_ms = fixed
    fit.aliased_slope = line_a[1]
    if w_direct > 0:
        fit.alpha_direct = line_n[1] / w_direct
    return fit


def differences(samples: dict[int, dict[str, list[float]]]) -> dict[int, float]:
    """D(n) = median(normal) - median(aliased), per rung."""
    out: dict[int, float] = {}
    for tiles, by_variant in samples.items():
        normal = by_variant.get("normal") or []
        aliased = by_variant.get("aliased") or []
        if normal and aliased:
            out[tiles] = statistics.median(normal) - statistics.median(aliased)
    return out


def bootstrap_alpha(samples: dict[int, dict[str, list[float]]], draws: int,
                    seed: int) -> tuple[float, float] | None:
    """90% interval on the alpha BRACKET, by resampling the timings.

    Resampling replicates WITHIN each (rung, variant) is the right unit: the
    rungs are a fixed designed ladder and are not a sample of anything, while
    the replicates are repeated draws of the same quantity and are exactly what
    the band is supposed to describe.
    """
    rng = random.Random(seed)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(draws):
        fit = fit_bracket(_resample(samples, rng))
        span = fit.bracket
        if span is not None:
            lows.append(span[0])
            highs.append(span[1])
    if len(lows) < max(20, draws // 10):
        return None
    lows.sort()
    highs.sort()
    # The 5th of the LOW end and the 95th of the HIGH end, so the reported
    # interval carries both the model ambiguity and the timing noise. Taking
    # percentiles of a single number would have quietly dropped the first.
    return (_percentile(lows, 5.0), _percentile(highs, 95.0))


def _resample(samples: dict[int, dict[str, list[float]]], rng):
    return {tiles: {name: [rng.choice(v) for _ in v] if v else []
                    for name, v in by_variant.items()}
            for tiles, by_variant in samples.items()}


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def pooled_band(per_model: dict[str, dict[int, dict[str, list[float]]]],
                draws: int, seed: int) -> tuple[float, float] | None:
    """An interval on the MEDIAN of the per-model alpha brackets.

    The median rather than the mean because P2 says alpha is a step function of
    per-expert bytes against L2, and a mean of a step function over four models
    is a number about this particular four models. The median is quoted as the
    pool statistic and every per-model alpha is printed beside it, so a reader
    who cares about a specific geometry is never made to read the pool.
    """
    rng = random.Random(seed)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(draws):
        low, high = [], []
        for samples in per_model.values():
            span = fit_bracket(_resample(samples, rng)).bracket
            if span is not None:
                low.append(span[0])
                high.append(span[1])
        if low:
            lows.append(statistics.median(low))
            highs.append(statistics.median(high))
    if len(lows) < max(20, draws // 10):
        return None
    lows.sort()
    highs.sort()
    return (_percentile(lows, 5.0), _percentile(highs, 95.0))


def supported_candidate(band: tuple[float, float]) -> tuple[list[str], str]:
    """Which of the three published values this band contains, and the verdict.

    Returns the names inside the band and a sentence naming the nearest one when
    the band contains none, because "supports none of them" is the answer that
    matters most and is the easiest to leave unsaid.
    """
    inside = [name for name, value in CANDIDATES if band[0] <= value <= band[1]]
    if inside:
        return inside, (
            f"the interval {band[0]:.3f}-{band[1]:.3f} contains "
            + " and ".join(inside))
    mid = 0.5 * (band[0] + band[1])
    name, value = min(CANDIDATES, key=lambda c: abs(c[1] - mid))
    width = max(band[1] - band[0], 1e-9)
    return [], (
        f"the interval {band[0]:.3f}-{band[1]:.3f} contains NONE of the "
        "three. "
        f"Nearest is {name} at {value:.3f}, "
        f"{abs(value - mid) / width:.1f} interval-widths away")


# --------------------------------------------------------------------------
# the kernel, built lazily so this file imports on a laptop
# --------------------------------------------------------------------------

class CannotRunHere(RuntimeError):
    """No GPU, no triton, or not enough memory. Named so `main` can exit 3."""


def build_kernel():
    """The ablation kernel. Imports triton, so it is never called off-GPU.

    THE SHAPE IS vLLM's, not a convenience. `b_ptrs` is
    `b + e*stride_be + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn`
    over a [E, N, K] weight tensor with K contiguous, advanced by
    `BLOCK_K * stride_bk` per iteration, under the same
    `GROUP_SIZE_M` swizzle -- which is `fused_moe_kernel` line for line on the
    B side. The A side and the reduction are the parts that differ, and the
    docstring at the top of this file says why.

    THE THREE RUNTIME SCALARS ARE THE ABLATION. `stride_be_eff`,
    `stride_bn_blk_eff` and `b_k_advance` are ordinary int arguments, so the
    compiler sees three unknown values and cannot fold, hoist or specialise on
    any of them. NORMAL passes the real strides; ALIASED passes (0, 0, 0), which
    pins every weight load to `B[0, 0:BLOCK_N, 0:BLOCK_K]`. Same kernel, same
    instructions, same counts. They are strides rather than 1/0 flags because
    Triton compiles an integer argument of exactly 1 in as a constant, which
    would have put the two variants in different specialisations; the top of
    this file says what that would have cost.

    `CONSTEXPR_ALIAS` exists ONLY to demonstrate the hazard. It is the naive way
    to write this experiment and it is compiled, counted and reported so that a
    reader can see whether the fold the repo warned about actually happens.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - needs the box
        raise CannotRunHere(
            "triton is not importable in this interpreter. Run inside the vllm "
            "venv on the pod: /workspace/venvs/vllm/bin/python "
            "scripts/alias_ablation.py --run") from exc

    @triton.jit
    def _ablation_kernel(
        a_ptr, b_ptr, c_ptr, expert_of_tile_ptr,
        EM, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_be_eff, stride_bn_blk_eff, b_k_advance,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, COMPUTE_DOT: tl.constexpr,
        CONSTEXPR_ALIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

        off_e = tl.load(expert_of_tile_ptr + pid_m).to(tl.int64)
        if CONSTEXPR_ALIAS:
            # The naive experiment: the aliasing is a compile-time constant, so
            # the compiler can see the whole loop reads one address. Compiled
            # and counted, never timed as the answer.
            b_base = off_e * 0
            k_advance = 0
        else:
            b_base = (off_e * stride_be_eff
                      + pid_n.to(tl.int64) * stride_bn_blk_eff)
            k_advance = b_k_advance
        offs_bn = tl.arange(0, BLOCK_N).to(tl.int64)
        b_ptrs = (b_ptr + b_base
                  + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if COMPUTE_DOT:
                acc += tl.dot(a, b)
            else:
                # STUDY.md item 4's own prescription. Both loads are consumed,
                # so neither can be eliminated, and the accumulator keeps its
                # full BLOCK_M x BLOCK_N float32 shape so register pressure and
                # therefore occupancy match the dot-mode kernel.
                acc += (tl.sum(tl.sum(a.to(tl.float32), axis=1), axis=0)
                        + tl.sum(tl.sum(b.to(tl.float32), axis=1), axis=0))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += k_advance

        offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, acc)

    return _ablation_kernel


# --------------------------------------------------------------------------
# the ISA check
# --------------------------------------------------------------------------

#: Every PTX mnemonic that moves data between global memory and the SM. Counting
#: `ld.global` ALONE would read zero on this kernel: Triton pipelines its
#: global-to-shared copies as `cp.async.cg.shared.global` whenever num_stages is
#: above one, and on Hopper can issue `cp.async.bulk.tensor` instead. A fold
#: check that counts the wrong mnemonic passes silently and is worse than none.
GLOBAL_OPS = {
    "ld.global": r"\bld\.global[a-z0-9_.]*",
    "cp.async": r"\bcp\.async[a-z0-9_.]*",
    "st.global": r"\bst\.global[a-z0-9_.]*",
}

#: Reported beside the load counts, because a change in the reduction would show
#: up here first and would mean the two variants are not the same work.
COMPUTE_OPS = {
    "mma.sync": r"\bmma\.sync[a-z0-9_.]*",
    "wgmma": r"\bwgmma[a-z0-9_.]*",
    "ld.shared": r"\bld\.shared[a-z0-9_.]*",
}


def count_ops(ptx: str) -> dict[str, int]:
    counts = {name: len(re.findall(pattern, ptx))
              for name, pattern in {**GLOBAL_OPS, **COMPUTE_OPS}.items()}
    counts["global_loads"] = counts["ld.global"] + counts["cp.async"]
    return counts


def extract_ptx(launch_result, jit_fn, seen: set) -> tuple[str | None, str]:
    """The PTX of the kernel that was just launched, however this Triton keeps it.

    Probed rather than assumed. `JITFunction.run` returns the CompiledKernel in
    Triton 3.x, but the attribute has moved before and a wrong guess would make
    the fold check silently unavailable at exactly the moment it matters. Every
    route is tried and the one that worked is reported, so a report never says
    "counts equal" when it means "counts unavailable".

    THE CACHE FALLBACK TAKES THE NEWLY ADDED ENTRY, not the first one it finds.
    `JITFunction.cache` holds one entry per specialisation and accumulates
    across rungs and across the constexpr variant, so "the first entry with
    PTX" would hand back whichever kernel happened to compile first and the fold
    check would compare a kernel against itself. `seen` carries the keys already
    accounted for; an ambiguous fallback -- more than one new entry, or none --
    reports unavailable rather than guessing, because a wrong PTX here is worse
    than no PTX.
    """
    asm = getattr(launch_result, "asm", None)
    if isinstance(asm, dict) and asm.get("ptx"):
        return asm["ptx"], "launch return value"
    cache = getattr(jit_fn, "cache", None)
    fresh = []
    if isinstance(cache, dict):
        for device, device_cache in cache.items():
            if not isinstance(device_cache, dict):
                continue
            for key, compiled in device_cache.items():
                if (device, key) in seen:
                    continue
                seen.add((device, key))
                asm = getattr(compiled, "asm", None)
                if isinstance(asm, dict) and asm.get("ptx"):
                    fresh.append(asm["ptx"])
    if len(fresh) == 1:
        return fresh[0], "JITFunction.cache, newly compiled entry"
    if fresh:
        return None, (f"{len(fresh)} kernels compiled during that launch, so "
                      "which PTX belongs to it is ambiguous")
    return None, ("no PTX reachable from the launch result, and nothing new in "
                  "JITFunction.cache; the fold check cannot run on this Triton")


@dataclass
class IsaReading:
    variant: str
    source: str
    digest: str
    counts: dict[str, int]


def isa_gate(records: list[dict]) -> Gate:
    """Did the aliased launch issue the same global loads as the normal one?

    A silent fold is the failure this whole design is arranged around, and it
    would produce a clean alpha of nearly zero that looked like a triumph.

    COMPARED WITHIN A RUNG, NEVER POOLED, and that is not fussiness. Triton
    specialises on each rung's own K, N and strides, so mixtral's kernel and
    deepseek-v3's kernel are legitimately different compilations with different
    digests. A gate that pooled every reading and compared digests would fail on
    every multi-model run for a reason that has nothing to do with folding, and
    a gate that fails for the wrong reason gets disabled.

    The constexpr reading is reported beside the pair and is NOT gated: it is
    allowed -- expected, even -- to fold, and its folding is the evidence that
    the runtime form was necessary.
    """
    name = "ISA: the aliased kernel issued the same global loads"
    pairs = []
    for row in records:
        by_variant = {r["variant"]: r for r in row.get("isa", [])}
        if "normal" in by_variant and "aliased" in by_variant:
            pairs.append((row["id"], by_variant))
    if not pairs:
        return Gate(name, None,
                    "PTX was not reachable for both runtime variants on any "
                    "rung, so the fold check did not run. No alpha from this "
                    "run should be quoted until it does")
    bad_loads, bad_digest = [], []
    for rid, by_variant in pairs:
        normal, aliased = by_variant["normal"], by_variant["aliased"]
        if normal["counts"]["global_loads"] != aliased["counts"]["global_loads"]:
            bad_loads.append((rid, normal, aliased))
        elif normal["digest"] != aliased["digest"]:
            bad_digest.append(rid)
    first = pairs[0][1]
    detail = (f"{len(pairs)} rungs compared. First, {pairs[0][0]}: normal "
              f"{first['normal']['counts']['global_loads']} global-load "
              f"instructions (ld.global {first['normal']['counts']['ld.global']}, "
              f"cp.async {first['normal']['counts']['cp.async']}); aliased "
              f"{first['aliased']['counts']['global_loads']} "
              f"(ld.global {first['aliased']['counts']['ld.global']}, "
              f"cp.async {first['aliased']['counts']['cp.async']})")
    if bad_loads:
        rid, normal, aliased = bad_loads[0]
        detail += (f". {len(bad_loads)} rungs DIFFER, first {rid}: "
                   f"{normal['counts']['global_loads']} against "
                   f"{aliased['counts']['global_loads']}. The aliased launch "
                   "did not issue the same loads, so its time is the optimiser "
                   "and not the cache, and every number below is void")
    elif bad_digest:
        detail += (f". {len(bad_digest)} rungs match on counts but not on PTX "
                   f"digest, first {bad_digest[0]}. Same counts from DIFFERENT "
                   "code is weaker than this design intends: the two launches "
                   "were supposed to be one compiled kernel driven by three "
                   "runtime scalars, and Triton's equal-to-1 argument "
                   "specialisation is the usual reason they are not")
    else:
        detail += ". Every rung's two launches are ONE compiled kernel"
    constexpr = next((v["constexpr-alias"] for _, v in pairs
                      if "constexpr-alias" in v), None)
    if constexpr is not None:
        folded = (constexpr["counts"]["global_loads"]
                  < first["normal"]["counts"]["global_loads"])
        detail += (f". The constexpr-aliased kernel, the naive way to write "
                   f"this, issues {constexpr['counts']['global_loads']} and "
                   + ("DID fold, which is the hazard happening in front of you"
                      if folded else
                      "did NOT fold on this Triton, so the hazard is real in "
                      "principle and did not bite here"))
    return Gate(name, not bad_loads and not bad_digest, detail)


# --------------------------------------------------------------------------
# measuring, which is the only part that needs the box
# --------------------------------------------------------------------------

def ablation_scalars(rung: Rung) -> dict[str, dict[str, int]]:
    """The three runtime integers that ARE the ablation, per variant.

    Pure arithmetic over the rung's shape, so the values the pod will pass can
    be checked on a laptop. `B` is a contiguous [E, N, K] tensor, matching
    vLLM's w1 layout, so its strides are (N*K, K, 1); `measure_rung` asserts the
    allocated tensor really has those strides rather than assuming it, because a
    non-contiguous B would change the address arithmetic under both variants
    while every table still printed numbers.

    NO ENTRY MAY EVER BE 1. Triton compiles an integer argument of exactly 1 in
    as a constant, which would put NORMAL and ALIASED in different
    specialisations and destroy the one-compiled-kernel guarantee the fold check
    rests on. Every entry is a large stride or 0, so both take the
    divisible-by-16 path. A test pins this for every shipped rung.
    """
    stride_be, stride_bn, stride_bk = rung.n * rung.k, rung.k, 1
    return {
        "normal": {"stride_be_eff": stride_be,
                   "stride_bn_blk_eff": rung.block_n * stride_bn,
                   "b_k_advance": rung.block_k * stride_bk},
        "aliased": {"stride_be_eff": 0, "stride_bn_blk_eff": 0,
                    "b_k_advance": 0},
    }


def check_output(rung: Rung, a, b, c, compute: str, aliased: bool,
                 torch) -> float | None:
    """Relative RMS error of a variant's output against its own closed form.

    BOTH DIRECTIONS ARE CHECKED AND BOTH ARE LOAD BEARING. NORMAL reproducing
    the reference proves it really read every expert's whole weight block, which
    is what W is supposed to be the cost of. ALIASED reproducing a DIFFERENT
    closed form -- `K/BLOCK_K` copies of the single aliased tile -- proves the
    aliasing took effect, which is the failure mode where the three runtime
    scalars do nothing, D(n) is pure noise, and alpha comes out near zero for
    the second time in this project's history.

    NOTHING IS MATERIALISED AT FULL SIZE. deepseek-v3's weight tensor is 15 GiB
    in bf16, so a float64 copy of it is 60 GiB and an obvious `b.to(float64)`
    would turn a correctness check into an out-of-memory crash on the one model
    that matters most. Every reduction passes `dtype=` so torch accumulates in
    float64 without a cast, and the comparison is against the ONE distinct value
    each (M-tile, N-block) cell holds rather than against an expanded copy of
    the output.

    In `dot` mode there is no cheap closed form, so three experts -- first,
    middle and last -- are checked against a real matmul. Three rather than one
    because the failure this catches is "normal did not read every expert".
    """
    if compute == "dot":
        if aliased:
            return None
        picks = sorted({0, rung.experts // 2, rung.experts - 1})
        worst = 0.0
        for e in picks:
            lo = e * rung.rows_per_expert
            hi = lo + rung.rows_per_expert
            want = (a[lo:hi].to(torch.float32) @ b[e].to(torch.float32).T)
            got = c[lo:hi]
            scale = want.abs().mean().item()
            err = (got - want).abs().mean().item()
            worst = max(worst, err / scale if scale else err)
        return float(worst)

    # sum mode: every (M-tile, N-block) cell of the output holds ONE value.
    a_tile = a.view(rung.num_pid_m, rung.block_m, rung.k).sum(
        dim=(1, 2), dtype=torch.float64)
    if aliased:
        one = b[0, :rung.block_n, :rung.block_k].sum(dtype=torch.float64)
        cell = a_tile[:, None] + one * rung.k_iters
        cell = cell.expand(rung.num_pid_m, rung.num_pid_n)
    else:
        b_block = b.view(rung.experts, rung.num_pid_n, rung.block_n,
                         rung.k).sum(dim=(2, 3), dtype=torch.float64)
        tile_expert = (torch.arange(rung.num_pid_m, device=a.device)
                       // rung.tiles)
        cell = a_tile[:, None] + b_block[tile_expert]

    got = c.view(rung.num_pid_m, rung.block_m, rung.num_pid_n, rung.block_n)
    # Within a cell every element is the same computation, so any spread is a
    # kernel bug and is folded into the error rather than averaged away.
    spread = (got.amax(dim=(1, 3)) - got.amin(dim=(1, 3))).to(torch.float64)
    err = (got[:, 0, :, 0].to(torch.float64) - cell).abs() + spread
    scale = cell.abs().mean().item()
    return float(err.mean().item() / scale) if scale else float(err.mean().item())


def measure_rung(kernel, rung: Rung, design: Design, replicates: int,
                 flusher, order_seed: int, torch, seen_kernels: set) -> dict:
    """Time both variants of one rung, interleaved, on the same tensors.

    THE INTERLEAVING IS THE POINT. Normal, aliased and a SECOND normal are timed
    in a shuffled order inside every replicate, on tensors allocated once, so a
    clock that drifts during the rung hits all three roughly equally and the
    second normal measures how much drift is left. That second pass is the
    placebo: two launches of an identical configuration, differing in nothing,
    whose difference is the noise floor D(n) has to beat.
    """
    from moe.bench.timing import ClockState, clock_drift

    device = "cuda"
    torch.manual_seed(0)
    # `empty().uniform_()` rather than `torch.rand(float32).to(bfloat16)`: the
    # float32 intermediate for deepseek-v3's weights is 30 GiB and would run the
    # card out of memory before a single kernel launched.
    a = torch.empty((rung.total_rows, rung.k), device=device,
                    dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    b = torch.empty((rung.experts, rung.n, rung.k), device=device,
                    dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    c = torch.zeros((rung.total_rows, rung.n), device=device, dtype=torch.float32)
    expert_of_tile = (torch.arange(rung.num_pid_m, device=device,
                                   dtype=torch.int32) // rung.tiles)

    grid = (rung.programs,)
    common = dict(
        EM=rung.total_rows, N=rung.n, K=rung.k,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bn=b.stride(1), stride_bk=b.stride(2),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=rung.block_m, BLOCK_N=rung.block_n, BLOCK_K=rung.block_k,
        GROUP_M=rung.group_m, COMPUTE_DOT=(design.compute == "dot"),
    )
    if tuple(b.stride()) != (rung.n * rung.k, rung.k, 1):
        raise CannotRunHere(
            f"the weight tensor is not contiguous [E, N, K]: strides "
            f"{tuple(b.stride())}. `ablation_scalars` computes the three "
            "runtime integers from the shape, and a different layout would "
            "alias something other than one BLOCK_K x BLOCK_N tile while every "
            "table still printed numbers.")
    scalars = ablation_scalars(rung)

    def launch(variant: str, constexpr_alias: bool = False):
        return kernel[grid](
            a, b, c, expert_of_tile,
            CONSTEXPR_ALIAS=constexpr_alias, num_warps=design.tile["num_warps"],
            num_stages=design.tile["num_stages"],
            **scalars[variant], **common)

    readings: list[IsaReading] = []
    correctness: dict[str, float | None] = {}
    for variant in ("normal", "aliased"):
        c.zero_()
        handle = launch(variant)
        torch.cuda.synchronize()
        ptx, source = extract_ptx(handle, kernel, seen_kernels)
        if ptx is not None:
            readings.append(IsaReading(
                variant=variant, source=source,
                digest=hashlib.sha256(ptx.encode()).hexdigest()[:12],
                counts=count_ops(ptx)))
        correctness[variant] = check_output(
            rung, a, b, c, design.compute, variant == "aliased", torch)
    with contextlib.suppress(Exception):
        c.zero_()
        handle = launch("normal", constexpr_alias=True)
        torch.cuda.synchronize()
        ptx, source = extract_ptx(handle, kernel, seen_kernels)
        if ptx is not None:
            readings.append(IsaReading(
                variant="constexpr-alias", source=source,
                digest=hashlib.sha256(ptx.encode()).hexdigest()[:12],
                counts=count_ops(ptx)))

    for _ in range(3):
        launch("normal")
        launch("aliased")
    torch.cuda.synchronize()

    rng = random.Random(order_seed)
    samples: dict[str, list[float]] = {"normal": [], "aliased": [], "placebo": []}
    clock_start = ClockState.sample()
    for _ in range(replicates):
        passes = ["normal", "aliased", "placebo"]
        rng.shuffle(passes)
        for name in passes:
            flusher.flush()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch("normal" if name == "placebo" else name)
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end))
    clock_end = ClockState.sample()
    drift, throttled = clock_drift(clock_start, clock_end)

    # NOT `del a, b, c` here: `launch` closes over them, and deleting a name a
    # nested function reads is a live bug the moment anyone adds a call below.
    # The tensors die with the frame and the caller drops the cache.
    return {
        "kind": "rung", "id": rung.key, "model": rung.model,
        "tiles": rung.tiles, "block_m": rung.block_m,
        "rows_per_expert": rung.rows_per_expert,
        "experts": rung.experts, "k": rung.k, "n": rung.n,
        "per_expert_bytes": rung.per_expert_bytes,
        "weight_bytes": rung.weight_bytes,
        "activation_fraction": rung.activation_fraction,
        "programs": rung.programs, "control": rung.control,
        "compute": design.compute,
        "ms": {name: values for name, values in samples.items()},
        "correctness": correctness,
        "isa": [{"variant": r.variant, "source": r.source, "digest": r.digest,
                 "counts": r.counts} for r in readings],
        "sm_clock_start": clock_start.sm_clock_mhz,
        "sm_clock_end": clock_end.sm_clock_mhz,
        "clock_drift": drift, "throttled": bool(throttled),
        "provenance": "measured",
    }


def measure(design: Design, args, out_dir: Path, done: set[str]) -> tuple[list[dict], dict]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - needs the box
        raise CannotRunHere("torch is not importable in this interpreter") from exc
    if not torch.cuda.is_available():
        raise CannotRunHere(
            "no CUDA device. The plan, the prediction and the estimator's "
            "self-test above all ran; only the timings need the box.")

    from moe.bench.timing import L2Flusher, flush_mb_for_device, runtime_info

    kernel = build_kernel()
    seen_kernels: set = set()
    meta = runtime_info()
    free_bytes = torch.cuda.mem_get_info()[0]
    flusher = L2Flusher(flush_mb_for_device() if args.l2_flush else 0)

    records: list[dict] = []
    path = out_dir / "cells.jsonl"
    for index, rung in enumerate(design.rungs):
        if rung.key in done:
            continue
        need = rung.footprint_bytes + 512 * 2 ** 20
        if need > free_bytes:
            record = {"kind": "rung", "id": rung.key, "model": rung.model,
                      "tiles": rung.tiles, "ms": {}, "skipped": True,
                      "provenance": "measured",
                      "why": f"needs {need / 2**30:.1f} GiB, "
                             f"{free_bytes / 2**30:.1f} GiB free"}
            print(f"[alias] SKIP {rung.key}: {record['why']}")
            _append(path, record)
            records.append(record)
            continue
        record = measure_rung(kernel, rung, design, args.replicates, flusher,
                              args.seed + index, torch, seen_kernels)
        # deepseek-v3's rung holds 16 GiB; the next model cannot be allocated
        # until the caching allocator gives it back.
        torch.cuda.empty_cache()
        _append(path, record)
        records.append(record)
        med = {k: (statistics.median(v) if v else float("nan"))
               for k, v in record["ms"].items()}
        print(f"[alias] {rung.key:44s} normal {med['normal']:9.4f} ms  "
              f"aliased {med['aliased']:9.4f} ms  "
              f"D {med['normal'] - med['aliased']:9.4f} ms")
    return records, meta


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves a partial last line. Dropping it is
            # right; failing the whole replay because of it is not.
            continue
        if row.get("kind") == "rung":
            out.append(row)
    return out


# --------------------------------------------------------------------------
# synthetic measurements, so the gates can be exercised without a GPU
# --------------------------------------------------------------------------

#: Each law exists to show one gate can bite. A pod script that always prints
#: PASS is worth nothing.
SYNTHETIC_LAWS = {
    "refit": "alpha = 0.558, L2 and HBM costs ADD; every gate should pass",
    "retracted": "alpha = 0.10, the value this repo retracted; P1 must FAIL",
    "tempo": "alpha = 0.33, TEMPO's value; P1 must FAIL",
    "folded": "alpha = 0.558 but the aliased kernel issued half the global "
              "loads; the ISA gate must FAIL and nothing may be quoted",
    "l2-step": "alpha is a step function of per-expert bytes against a 50 MiB "
               "L2, which is P2's mechanism; the per-model bands must separate",
    "noise": "alpha = 0.558 with a placebo drift as large as the signal; the "
             "placebo gate must FAIL",
    "max-model": "alpha = 0.558 but the kernel runs at max(L2, HBM), so the "
                 "DIFFERENCE estimator is biased down and only the bracket "
                 "still contains the truth",
    "l2-heavy": "alpha = 0.558 at max(L2, HBM) with r = 0.45, which is the H200 "
                "end of plausible; the bracket is too wide to pick a candidate "
                "and the run must say NOT TESTABLE rather than choose",
}

#: Laws that compose L2 and HBM service as `max` rather than as a sum. A
#: streaming kernel is closer to this end, which is why two of the eight are
#: here and why the report brackets rather than reporting one number.
SYNTHETIC_MAX_LAWS = ("max-model", "l2-heavy")

#: `r`: the aliased ladder's per-tile cost as a fraction of one weight read.
#: 0.12 for most laws so the shipped gates have something to resolve, and 0.45
#: for `l2-heavy`, which is what an L2 only about twice HBM's bandwidth would
#: give and is the reason the bracket exists at all.
SYNTHETIC_L2_RATIO = 0.12
SYNTHETIC_L2_RATIO_HEAVY = 0.45

SYNTHETIC_L2_BYTES = 50 * 2 ** 20


def synthesise(design: Design, law: str, seed: int,
               noise: float = 0.004) -> list[dict]:
    """Timings generated from a STATED law, so a verdict can be checked.

    The generator is the ablation's own physics and nothing more: a fixed cost,
    an activation and output stream that scales with the tile count, and a
    weight stream of `W (1 + alpha (n-1))` that the aliased variant does not
    pay. If the estimator cannot recover a planted alpha from that, it cannot
    recover a real one either.
    """
    if law not in SYNTHETIC_LAWS:
        raise ValueError(f"unknown law {law!r}; expected one of {sorted(SYNTHETIC_LAWS)}")
    rng = random.Random(seed)
    bandwidth = 4.4e12
    records = []
    for rung in design.rungs:
        if law in ("l2-heavy", "max-model"):
            alpha = REFIT_ALPHA
        elif law == "l2-step":
            alpha = 0.95 if rung.per_expert_bytes > SYNTHETIC_L2_BYTES else 0.05
        elif law == "retracted":
            alpha = REPO_RETRACTED_ALPHA
        elif law == "tempo":
            alpha = TEMPO_ALPHA
        else:
            alpha = REFIT_ALPHA
        if rung.control:
            alpha = 0.0
        w_ms = rung.weight_bytes / bandwidth * 1e3
        other = (rung.activation_bytes + rung.output_bytes) / bandwidth * 1e3
        fixed = 0.01
        # THE ALIASED LADDER IS NOT FREE, and pretending it was is what made an
        # earlier version of this generator unable to exercise the bracket at
        # all. Both variants issue n W bytes of loads; the aliased one hits L2
        # for all of them, at `r` of a weight read per tile.
        ratio = (SYNTHETIC_L2_RATIO_HEAVY if law == "l2-heavy"
                 else SYNTHETIC_L2_RATIO)
        l2_cost = ratio * w_ms * rung.tiles
        base_alias = fixed + other + l2_cost
        hbm_cost = w_ms * (1.0 + alpha * (rung.tiles - 1))
        if law in SYNTHETIC_MAX_LAWS:
            base_normal = fixed + other + max(l2_cost, hbm_cost)
        else:
            base_normal = base_alias + hbm_cost
        drift = 0.0
        if law == "noise":
            drift = 0.30 * w_ms

        def draw(mean, spread=noise, rng=rng):
            return [mean * (1.0 + rng.gauss(0.0, spread))
                    for _ in range(design.replicates)]

        counts_normal = {"ld.global": 0, "cp.async": 64, "st.global": 1,
                         "mma.sync": 0, "wgmma": 0, "ld.shared": 128,
                         "global_loads": 64}
        counts_alias = dict(counts_normal)
        if law == "folded":
            counts_alias["cp.async"] = 32
            counts_alias["global_loads"] = 32
        isa = [{"variant": "normal", "source": "synthetic", "digest": "aaaaaaaaaaaa",
                "counts": counts_normal},
               {"variant": "aliased", "source": "synthetic",
                "digest": "aaaaaaaaaaaa" if law != "folded" else "bbbbbbbbbbbb",
                "counts": counts_alias},
               {"variant": "constexpr-alias", "source": "synthetic",
                "digest": "cccccccccccc",
                "counts": dict(counts_normal, cp_async=1, global_loads=1)}]
        records.append({
            "kind": "rung", "id": rung.key, "model": rung.model,
            "tiles": rung.tiles, "block_m": rung.block_m,
            "rows_per_expert": rung.rows_per_expert, "experts": rung.experts,
            "k": rung.k, "n": rung.n,
            "per_expert_bytes": rung.per_expert_bytes,
            "weight_bytes": rung.weight_bytes,
            "activation_fraction": rung.activation_fraction,
            "programs": rung.programs, "control": rung.control,
            "compute": design.compute,
            "ms": {"normal": draw(base_normal), "aliased": draw(base_alias),
                   "placebo": draw(base_normal + drift)},
            "correctness": {"normal": 1e-7, "aliased": 1e-7},
            "isa": isa, "sm_clock_start": 1500, "sm_clock_end": 1500,
            "clock_drift": 0.0, "throttled": False,
            "provenance": "synthetic", "law": law,
        })
    return records


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

class Report:
    """Everything printed, kept so it can also be written beside the data.

    A report that exists only in a terminal scrollback does not survive a pod
    teardown, and the whole point of the output directory is that the analysis
    leaves with the numbers.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def report_prediction(say) -> None:
    say("## the prediction, before anything runs")
    say()
    say(f"  P1  alpha = {REFIT_ALPHA:.3f}, 90% band "
        f"{REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}  (today's refit, 10,813 rows)")
    say(f"      TEMPO arXiv:2608.13057 fits {TEMPO_ALPHA:.2f}; this repo's "
        f"retracted value is {REPO_RETRACTED_ALPHA:.2f}.")
    say(f"      This ablation runs at GROUP_SIZE_M=1, where the refit's own "
        f"split reads {REFIT_ALPHA_AT_GROUP_M_1:.3f},")
    say("      so that is the closer comparison and both are printed.")
    say()
    say("  P2  alpha is not a scalar: at GROUP_SIZE_M=1 the reuse distance is "
        "one pass over")
    say("      one expert's weight block, so alpha should track PER-EXPERT "
        "BYTES against L2.")
    say()
    say("  PASS/FAIL below is against P1's band, and the report names which of "
        "the three")
    say("  candidate values the measured band actually supports.")


def report_design(say, design: Design, gates: list[Gate], l2_bytes: int) -> None:
    say()
    say("## the design")
    say()
    say(f"compute mode {design.compute}   BLOCK_M {design.block_m}   "
        f"tile {design.tile}   replicates {design.replicates}")
    if design.compute == "dot":
        say("*** dot mode is BIASED LOW and is not P1's answer. The aliased "
            "variant becomes")
        say("*** compute bound while the normal one stays memory bound, so "
            "D(n) loses one copy")
        say("*** of the per-tile compute cost and the fitted alpha falls "
            "toward zero.")
    say()
    say("  model                   E      K      N   MiB/expert  weight GiB  "
        "rows/expert   act frac")
    for model in design.models:
        for rung in design.rungs_for(model):
            say(f"  {model:20s} {rung.experts:4d} {rung.k:6d} {rung.n:6d} "
                f"{rung.per_expert_bytes / 2**20:11.1f} "
                f"{rung.weight_bytes / 2**30:11.2f} "
                f"{rung.rows_per_expert:12d} {rung.activation_fraction:10.4f}")
    if l2_bytes:
        say()
        say(f"  L2 on the attached card: {l2_bytes / 2**20:.1f} MiB. P2 says "
            "alpha is near 1 above it and near 0 below.")
    say()
    for gate in gates:
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")


def samples_from(records: list[dict]) -> dict[str, dict[int, dict[str, list[float]]]]:
    out: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in records:
        if row.get("skipped") or not row.get("ms"):
            continue
        out.setdefault(row["model"], {})[row["tiles"]] = {
            name: list(values) for name, values in row["ms"].items()}
    return out


def report_measurements(say, records: list[dict], design: Design) -> None:
    say()
    say("## the measurements")
    say()
    say("  model                  tiles   rows/e     normal    aliased          D"
        "    placebo")
    for model in design.models:
        for row in sorted((r for r in records if r.get("model") == model),
                          key=lambda r: r.get("tiles", 0)):
            if row.get("skipped"):
                say(f"  {model:20s} {row.get('tiles', 0):7d}   SKIPPED: "
                    f"{row.get('why', '')}")
                continue
            ms = row["ms"]
            normal = statistics.median(ms["normal"])
            aliased = statistics.median(ms["aliased"])
            placebo = statistics.median(ms["placebo"]) if ms.get("placebo") else float("nan")
            say(f"  {model:20s} {row['tiles']:7d} {row['rows_per_expert']:8d} "
                f"{normal:10.4f} {aliased:10.4f} {normal - aliased:10.4f} "
                f"{placebo - normal:10.4f}")
    say()
    say("  D is the ablation difference and is the HBM cost of that rung's "
        "weight reads.")
    say("  placebo is a SECOND normal launch minus the first: two identical "
        "configurations,")
    say("  so it is the noise floor D has to beat and nothing else.")


def report_isa(say, records: list[dict]) -> None:
    say()
    say("## the ISA check: did the aliased kernel really issue the loads?")
    say()
    say("  rung                          variant           PTX      "
        "ld.global   cp.async   global loads   mma.sync")
    seen = set()
    for row in records:
        for reading in row.get("isa", []):
            key = (reading["variant"], reading["digest"])
            if key in seen:
                continue
            seen.add(key)
            counts = reading["counts"]
            say(f"  {row['id']:28s}  {reading['variant']:16s} "
                f"{reading['digest']:>12s} {counts.get('ld.global', 0):10d} "
                f"{counts.get('cp.async', 0):10d} "
                f"{counts.get('global_loads', 0):14d} "
                f"{counts.get('mma.sync', 0):10d}")
    say()
    say("  ld.global ALONE would read zero here: Triton pipelines its "
        "global-to-shared copies")
    say("  as cp.async at num_stages > 1, so the gate is on the SUM. normal and "
        "aliased are")
    say("  the same compiled kernel driven by three runtime scalars, so equal "
        "counts are a")
    say("  property of the design; constexpr-alias is the naive form and is "
        "shown folding or not.")


@dataclass
class ModelResult:
    model: str
    fit: AlphaFit
    band: tuple[float, float] | None
    per_expert_mib: float
    control: bool


def analyse_models(design: Design, records: list[dict], draws: int,
                   seed: int) -> list[ModelResult]:
    per_model = samples_from(records)
    sizes = {r["model"]: r.get("per_expert_bytes", 0) for r in records}
    controls = {r["model"]: bool(r.get("control")) for r in records}
    out = []
    for model in design.models:
        samples = per_model.get(model)
        if not samples:
            continue
        fit = fit_bracket(samples)
        band = bootstrap_alpha(samples, draws, seed) if fit.ok else None
        out.append(ModelResult(model=model, fit=fit, band=band,
                               per_expert_mib=sizes.get(model, 0) / 2 ** 20,
                               control=controls.get(model, False)))
    return out


def report_alphas(say, results: list[ModelResult], pooled: tuple[float, float] | None,
                  pooled_bracket: tuple[float, float] | None) -> None:
    say()
    say("## alpha, from the ablation alone, as a bracket")
    say()
    say("  Two estimators, because exactly one of them is right and which one "
        "depends on how")
    say("  L2 service and HBM service compose. DIFFERENCE subtracts the aliased "
        "ladder and is")
    say("  exact if the two costs ADD; DIRECT fits the normal ladder with the "
        "fixed cost taken")
    say("  from the aliased ladder's own n=0 intercept and is exact if the "
        "kernel runs at")
    say("  max(L2, HBM). For any alpha <= 1 they BRACKET the truth, and the "
        "bracket is narrow")
    say("  exactly when r, the aliased ladder's per-tile cost over one weight "
        "read, is small.")
    say()
    say("  model                  MiB/expert    W (ms)   fixed  difference  "
        "direct       r      R^2")
    for res in results:
        fit = res.fit
        if not fit.ok:
            say(f"  {res.model:20s} {res.per_expert_mib:11.1f}   NOT FITTED: "
                f"{fit.why}")
            continue
        direct = "none" if fit.alpha_direct is None else f"{fit.alpha_direct:.3f}"
        share = "none" if fit.l2_share is None else f"{fit.l2_share:.3f}"
        tag = "  (L2-RESIDENT CONTROL)" if res.control else ""
        say(f"  {res.model:20s} {res.per_expert_mib:11.1f} {fit.w_ms:9.4f} "
            f"{(fit.fixed_ms or 0.0):7.4f} {fit.alpha:11.3f} {direct:>7s} "
            f"{share:>7s} {fit.r2:8.4f}{tag}")
    say()
    say("  model                  bracket           90% interval        "
        "per-rung alphas (difference)")
    for res in results:
        fit = res.fit
        if not fit.ok:
            continue
        span = fit.bracket
        band = (f"{res.band[0]:.3f} to {res.band[1]:.3f}" if res.band else "none")
        rungs = "  ".join(f"n={n}:{a:.3f}" for n, a in sorted(fit.per_rung.items()))
        say(f"  {res.model:20s} {span[0]:.3f} to {span[1]:.3f}   "
            f"{band:>18s}   {rungs}")
    say()
    say("  W is the measured time of ONE full pass over every expert's weight "
        "block, read off")
    say("  the n=1 rung. Both alphas are a fitted slope over that intercept, so "
        "every unit of")
    say("  time cancels and no bandwidth, byte count or ridge enters either "
        "number.")
    if pooled_bracket is not None:
        say()
        say(f"  POOLED (median over the non-control models): alpha is in "
            f"{pooled_bracket[0]:.3f} to {pooled_bracket[1]:.3f}"
            + (f", 90% interval {pooled[0]:.3f} to {pooled[1]:.3f}"
               if pooled else ", no interval"))
        say(f"  against the refit's {REFIT_ALPHA:.3f} "
            f"({REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}) pooled and "
            f"{REFIT_ALPHA_AT_GROUP_M_1:.3f} at GROUP_SIZE_M=1.")


def report_confounds(say, design: Design, results: list[ModelResult]) -> None:
    say()
    say("## what else differs between aliased and normal")
    say()
    worst = max((r.activation_fraction for r in design.rungs), default=0.0)
    say("  BOUNDED. L2 capacity. The aliased variant leaves L2 to the "
        "activations, whose stream")
    say(f"  is {worst * 100:.2f}% of the weight stream at the worst rung, so "
        "the most the freed")
    say("  capacity can be worth is that fraction of D. The output write is "
        "identical in both.")
    say()
    control = next((r for r in results if r.control), None)
    if control is not None and control.fit.ok:
        say("  BOUNDED BY MEASUREMENT. TLB, page behaviour and code path. The "
            "L2-resident control")
        say("  ran the same ladder on a geometry whose PER-EXPERT block fits "
            "in L2, so NORMAL")
        say(f"  has no HBM re-read to save. Its W is {control.fit.w_ms:.4f} ms "
            f"and its alpha brackets "
            f"{control.fit.bracket[0]:.3f} to {control.fit.bracket[1]:.3f}.")
        say("  Whatever survives there is not weight traffic.")
    else:
        say("  NOT BOUNDED. TLB, page behaviour and code path. The aliased "
            "variant touches one")
        say("  page of B where the normal one touches the whole tensor, and "
            "this run has no")
        say("  L2-resident control to bound it. Re-run with --control.")
    say()
    say("  BOUNDED BY THE BRACKET. L2 service, and the cache-set distribution "
        "that makes it")
    say("  worse. Both variants push n W bytes through L2, and the aliased "
        "variant pins every")
    say("  one of them to a single BLOCK_K x BLOCK_N tile, so its loads land on "
        "a handful of L2")
    say("  slices and are served more slowly than the aggregate L2 bandwidth "
        "would suggest.")
    say("  Whatever that costs shows up in the ALIASED ladder's own slope, "
        "which is r above,")
    say("  and the two estimators bracket the truth for any composition of L2 "
        "and HBM service")
    say("  between a sum and a max. Narrowing it means spreading the aliased "
        "tile across more")
    say("  slices, which this design does not do because a spread alias has no "
        "closed form to")
    say("  check the output against.")
    say()
    say("  ABSORBED BY CONSTRUCTION. Any cost that scales with the extra-tile "
        "count and is not a")
    say("  weight re-read lands in alpha, because (n-1) is the regressor. That "
        "is a property of")
    say("  the estimator and no control can remove it.")


# --------------------------------------------------------------------------
# gates on the result
# --------------------------------------------------------------------------

def correctness_gate(records: list[dict], compute: str) -> Gate:
    checked = [(r["id"], name, value)
               for r in records
               for name, value in (r.get("correctness") or {}).items()
               if value is not None]
    if not checked:
        return Gate("correctness: each variant reproduced its closed form", None,
                    "no rung recorded a correctness check")
    bad = [(rid, name, value) for rid, name, value in checked
           if not (value == value) or value > CORRECTNESS_RTOL]
    detail = (f"{len(checked)} checks at relative RMS <= {CORRECTNESS_RTOL:.0e}; "
              f"worst {max(v for _, _, v in checked):.2e}")
    if compute == "dot":
        detail += (". In dot mode the ALIASED side has no cheap closed form and "
                   "is UNCHECKED, so nothing here says the aliasing took "
                   "effect; the normal side is checked against a real matmul on "
                   "the first, middle and last expert")
    if bad:
        detail += (f". {len(bad)} failed, first {bad[0][0]} {bad[0][1]} at "
                   f"{bad[0][2]:.2e}. A failing NORMAL means it did not read "
                   "every expert's block; a failing ALIASED means the three "
                   "runtime scalars did nothing and D is noise")
    return Gate("correctness: each variant reproduced its closed form",
                not bad, detail)


def placebo_gate(records: list[dict]) -> Gate:
    """Two launches of an identical configuration, against the signal.

    THE CONTROL RUNG IS EXCLUDED and that is not a convenience. Its D is near
    zero by construction, so the ratio drift/D is a ratio of two noise floors
    and reads as a catastrophic failure on a perfectly clean run -- observed at
    542% on the first synthetic pass of this gate. The question the gate asks is
    whether the noise floor is small against the signal that alpha is FITTED
    from, and no alpha is fitted from the control.
    """
    worst = None
    for row in records:
        ms = row.get("ms") or {}
        if row.get("control"):
            continue
        if not ms.get("normal") or not ms.get("placebo") or not ms.get("aliased"):
            continue
        normal = statistics.median(ms["normal"])
        diff = normal - statistics.median(ms["aliased"])
        drift = abs(statistics.median(ms["placebo"]) - normal)
        if diff <= 0:
            continue
        ratio = drift / diff
        if worst is None or ratio > worst[0]:
            worst = (ratio, row["id"], drift, diff)
    if worst is None:
        return Gate("placebo: two identical launches differ by far less than D",
                    None, "no rung has both a placebo pass and a positive D")
    ratio, rid, drift, diff = worst
    return Gate("placebo: two identical launches differ by far less than D",
                ratio <= PLACEBO_MAX_FRACTION,
                f"worst rung {rid}: two identical configurations differ by "
                f"{drift:.4f} ms against a D of {diff:.4f} ms, "
                f"{ratio * 100:.1f}% (limit {PLACEBO_MAX_FRACTION * 100:.0f}%)")


def signal_gate(records: list[dict]) -> Gate:
    fractions = []
    for row in records:
        if row.get("control") or row.get("tiles") != 1:
            continue
        ms = row.get("ms") or {}
        if not ms.get("normal") or not ms.get("aliased"):
            continue
        normal = statistics.median(ms["normal"])
        if normal <= 0:
            continue
        fractions.append(((normal - statistics.median(ms["aliased"])) / normal,
                          row["id"]))
    if not fractions:
        return Gate("signal: the weight read is most of what the kernel does",
                    None, "no one-tile rung was timed")
    worst, rid = min(fractions)
    return Gate("signal: the weight read is most of what the kernel does",
                worst >= MIN_SIGNAL_FRACTION,
                f"weakest one-tile rung {rid} has D(1) at {worst * 100:.1f}% of "
                f"its own time (limit {MIN_SIGNAL_FRACTION * 100:.0f}%). Below "
                "that the difference is measuring something the weight-read "
                "label does not cover")


def linearity_gate(results: list[ModelResult]) -> Gate:
    fitted = [r for r in results if r.fit.ok and not r.control]
    if not fitted:
        return Gate("form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires",
                    None, "no model produced a fit")
    worst = min(fitted, key=lambda r: r.fit.r2)
    return Gate("form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires",
                worst.fit.r2 >= MIN_LINEARITY_R2,
                f"worst R^2 {worst.fit.r2:.4f} on {worst.model} "
                f"(limit {MIN_LINEARITY_R2}). Below it the functional form is "
                "wrong and slope-over-intercept is not alpha")


def control_gate(results: list[ModelResult]) -> Gate:
    """The control bounds everything that scales with the tile count and is not
    weight traffic.

    IT IS THE CONTROL'S ALPHA THAT MATTERS, NOT ITS W. An earlier version of
    this gate compared the control's W against the real models' and would have
    failed on a correct run: the control still pays a full first pass over its
    512 MiB tensor, so its W is legitimately the same order as theirs. What the
    control does NOT have is an HBM re-read, because 16 MiB per expert fits in
    L2, so the tile-count SLOPE is where its emptiness has to show up. If TLB
    pressure, page behaviour or a code-path difference were driving D(n), they
    would drive it here too and the control's alpha would not be near zero.
    """
    name = "control: an L2-resident expert shows no extra-tile cost"
    control = next((r for r in results if r.control and r.fit.ok), None)
    real = [r for r in results if not r.control and r.fit.ok]
    if control is None or not real:
        return Gate(name, None,
                    "no L2-resident control in this run; re-run with --control. "
                    "Without it, TLB and page behaviour are named but unbounded")
    smallest = min(r.fit.bracket[1] for r in real)
    span = control.fit.bracket
    # CONSISTENT WITH ZERO, not "small in the difference estimator". The
    # control carries the same L2-versus-HBM ambiguity as everything else, so
    # its difference estimator reads -r/(1-r) and its direct one r/(1+r) for a
    # true alpha of zero. Gating either end alone would fail a clean control.
    ok = span[0] <= CONTROL_MAX_ALPHA and span[1] >= -CONTROL_MAX_ALPHA
    return Gate(name, ok,
                f"the control's bracket is {span[0]:.3f} to {span[1]:.3f} on a "
                f"W of {control.fit.w_ms:.4f} ms, against a top end of "
                f"{smallest:.3f} for the weakest real model. It must be "
                f"consistent with zero to within {CONTROL_MAX_ALPHA}: its "
                "re-reads hit L2, so anything it does show is the size of an "
                "extra-tile cost that is not weight traffic")


def band_gate(pooled: tuple[float, float] | None,
              l2_shares: list[float]) -> Gate:
    """Can the reported interval pick one of the three candidates at all?

    NOT TESTABLE rather than FAIL when it cannot. A wide interval is not a
    refutation of anything, it is a measurement that did not resolve, and the
    difference matters: this project has a standing habit of reading a wide
    number as a finding. The detail names WHY it is wide, because r is
    measured and is the only thing that sets it.
    """
    name = "resolution: the interval can separate the three candidates"
    if pooled is None:
        return Gate(name, None, "no pooled interval was produced")
    width = pooled[1] - pooled[0]
    detail = (f"90% interval is {width:.3f} wide (limit {MAX_BAND_WIDTH}). The "
              f"candidates span {REPO_RETRACTED_ALPHA} to {REFIT_ALPHA}, and an "
              "interval wider than half the largest gap between adjacent "
              "candidates cannot pick one")
    if l2_shares:
        detail += (f". r, the aliased ladder's per-tile cost over one weight "
                   f"read, is {min(l2_shares):.3f} to {max(l2_shares):.3f}, and "
                   "r is what sets the bracket's width: the two estimators "
                   "differ by roughly 2r/(1-r^2). Narrowing this needs a "
                   "cheaper aliased ladder, which means spreading the aliased "
                   "tile across more L2 slices rather than pinning every load "
                   "to one")
    # NOT TESTABLE, not FAIL. A wide interval refutes nothing; it says the run
    # did not resolve, and this project has a standing habit of reading a wide
    # number as a finding.
    return Gate(name, True if width <= MAX_BAND_WIDTH else None, detail)


def prediction_gate(pooled: tuple[float, float] | None,
                    pooled_bracket: tuple[float, float] | None,
                    compute: str) -> Gate:
    name = f"P1: the ablation agrees with the refit, alpha = {REFIT_ALPHA}"
    if pooled is None or pooled_bracket is None:
        return Gate(name, None, "no pooled interval; nothing to score")
    if compute == "dot":
        return Gate(name, None,
                    f"dot mode puts alpha in {pooled[0]:.3f} to "
                    f"{pooled[1]:.3f}, and dot mode is biased LOW by one copy "
                    "of the per-tile compute cost on top of everything the "
                    "bracket already carries. It is a lower bound, not P1's "
                    "answer. Re-run in sum mode")
    overlap = pooled[0] <= REFIT_BAND[1] and REFIT_BAND[0] <= pooled[1]
    _, sentence = supported_candidate(pooled)
    return Gate(name, overlap,
                f"measured alpha is in {pooled_bracket[0]:.3f} to "
                f"{pooled_bracket[1]:.3f} before noise and {pooled[0]:.3f} to "
                f"{pooled[1]:.3f} after it, against the refit's "
                f"{REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}: "
                f"{'they overlap' if overlap else 'they are DISJOINT'}. "
                f"{sentence}")


def mechanism_note(say, results: list[ModelResult], l2_bytes: int) -> None:
    """P2, reported and never gated.

    It is not gated because a step in four points is a pattern and not a test,
    and this project has already had to retract one monotone-in-expert-count
    pattern that was an artefact of pooling. Printed with the sizes beside it so
    a reader can see the ordering rather than be told about it.
    """
    fitted = [r for r in results if r.fit.ok and not r.control]
    if len(fitted) < 2:
        return
    say()
    say("## P2, reported and deliberately NOT gated")
    say()
    say("  A step in four points is a pattern, not a test. This study has "
        "already retracted one")
    say("  monotone-in-expert-count reading that was an artefact of pooling, so "
        "the ordering is")
    say("  printed and left for a design that varies the footprint "
        "continuously.")
    say()
    for res in sorted(fitted, key=lambda r: r.per_expert_mib):
        side = "?" if not l2_bytes else (
            "above L2" if res.per_expert_mib * 2 ** 20 > l2_bytes else "below L2")
        span = res.fit.bracket
        say(f"  {res.model:20s} {res.per_expert_mib:8.1f} MiB/expert  "
            f"{side:9s}  alpha {span[0]:.3f} to {span[1]:.3f}")


def verdict(say, gates: list[Gate]) -> int:
    say()
    say("## gates")
    say()
    for gate in gates:
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")
    say()
    failed = [g for g in gates if g.ok is False]
    untested = [g for g in gates if g.ok is None]
    if failed:
        say(f"VERDICT: REFUTED or VOID. {len(failed)} gate(s) failed: "
            + "; ".join(g.name for g in failed))
        return 1
    if untested:
        say(f"VERDICT: NOT TESTABLE. {len(untested)} gate(s) had no evidence: "
            + "; ".join(g.name for g in untested))
        return 4
    say("VERDICT: the ablation agrees with the refit. alpha measured without "
        "the byte model,")
    say("without a calibrated bandwidth and without the ridge lands inside the "
        "refit's band,")
    say("so the number the tile-corrected roofline rests on has independent "
        "support.")
    return 0


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def results_root() -> Path:
    """Where output goes so that it survives the pod being terminated.

    Same rule as `scripts/run_all.sh` and `scripts/group_m_alpha_sweep.py`:
    `$MOE_RESULTS_DIR`, else the network volume at `/workspace/results` when
    there is one, else the repo's own `results/`. The pod's container disk dies
    with the pod and the volume does not.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return ROOT / "results"


def git_head() -> str:
    with contextlib.suppress(Exception):
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return ""


def l2_bytes_here() -> int:
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(getattr(props, "L2_cache_size", 0))
    return 0


def cross_check_candidates() -> str:
    """The two published rivals, checked against `scripts/alpha_refit.py`.

    Loaded by path because `scripts/` is not a package. This does NOT import the
    estimator -- the whole value of this experiment is that it fits alpha by a
    route `alpha_refit` does not touch -- but the three numbers being scored
    against must not drift away from the ones the estimator prints.
    """
    import importlib.util
    path = ROOT / "scripts" / "alpha_refit.py"
    try:
        spec = importlib.util.spec_from_file_location("alpha_refit_constants", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a missing rival is not fatal
        return f"could not read {path.name} ({type(exc).__name__}); using literals"
    mismatches = []
    if getattr(module, "REPO_PUBLISHED_ALPHA", None) != REPO_RETRACTED_ALPHA:
        mismatches.append("REPO_PUBLISHED_ALPHA")
    if getattr(module, "TEMPO_ALPHA", None) != TEMPO_ALPHA:
        mismatches.append("TEMPO_ALPHA")
    if mismatches:
        return (f"DISAGREES with alpha_refit.py on {', '.join(mismatches)}. "
                "One of the two files has drifted and the scoring below is "
                "against this file's literals")
    return "agrees with alpha_refit.py on both rival values"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--tiles", default=",".join(str(t) for t in TILE_LADDER))
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=FIXED_TILE["BLOCK_N"])
    parser.add_argument("--block-k", type=int, default=FIXED_TILE["BLOCK_K"])
    parser.add_argument("--group-m", type=int, default=FIXED_TILE["GROUP_M"])
    parser.add_argument("--compute", choices=COMPUTE_MODES, default="sum",
                        help="how the loads are kept live. sum is STUDY.md's "
                             "prescription and is unbiased; dot is the real "
                             "GEMM reduction and is biased low")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--control", action="store_true", default=True,
                        help="include the L2-resident control rung (default on)")
    parser.add_argument("--no-control", dest="control", action="store_false")
    parser.add_argument("--run", action="store_true",
                        help="measure on the GPU; without it this plans only")
    parser.add_argument("--replay", type=Path, default=None,
                        help="re-report an existing output directory, no GPU")
    parser.add_argument("--synthetic", choices=sorted(SYNTHETIC_LAWS), default=None,
                        help="generate timings from a stated law and run the "
                             "gates on them, so the gates are testable off-GPU")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any rungs already on disk and start over")
    parser.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                        help="time with L2 warm; the default flushes, because "
                             "an ablation of cache behaviour must control the "
                             "cache state it starts from")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.models = tuple(v for v in str(args.models).split(",") if v)
    args.tiles = tuple(sorted(int(v) for v in str(args.tiles).split(",") if v))
    unknown = [m for m in args.models
               if m != CONTROL_MODEL and m not in MODEL_CONFIGS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from "
                     f"{sorted(MODEL_CONFIGS)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    design = build_design(args)
    suffix = f"-synthetic-{args.synthetic}" if args.synthetic else ""
    out_dir = args.replay or args.out or (
        results_root() / "alias_ablation"
        / f"{design.compute}-bm{design.block_m}-{design.fingerprint}{suffix}")

    say = Report()
    say(f"# alpha by ablation, without the byte model   ({git_head() or 'no git'})")
    say()
    say(f"output directory: {out_dir}")
    say("Everything below is written there as report.md, beside plan.json and "
        "cells.jsonl.")
    say(f"candidate values: {cross_check_candidates()}")
    if args.synthetic:
        say()
        say(f"*** SYNTHETIC ({args.synthetic}): {SYNTHETIC_LAWS[args.synthetic]}.")
        say("*** Nothing here was measured. These rows come from a stated law "
            "and exist to show")
        say("*** the gates can see an effect and can miss its absence.")
    say()
    report_prediction(say)

    l2 = l2_bytes_here() or (SYNTHETIC_L2_BYTES if args.synthetic else 0)
    pre = preflight(design, l2)
    report_design(say, design, pre, l2)
    if any(g.ok is False for g in pre):
        say()
        say("VERDICT: the design is refused before spending anything. Fix the "
            "failed preflight gate above.")
        _save(out_dir, say)
        return 1

    records: list[dict] = []
    if args.replay:
        records = read_records(out_dir / "cells.jsonl")
        say()
        say(f"## replay: {len(records)} rungs read from disk, nothing measured")
    elif args.synthetic:
        records = synthesise(design, args.synthetic, args.seed)
        with contextlib.suppress(OSError):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "cells.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records))
    elif args.run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            (out_dir / "cells.jsonl").unlink(missing_ok=True)
        existing = read_records(out_dir / "cells.jsonl")
        done = {r["id"] for r in existing}
        (out_dir / "plan.json").write_text(json.dumps(
            {"fingerprint": design.fingerprint, "argv": sys.argv[1:],
             "git": git_head(), "models": list(design.models),
             "tiles": list(design.tiles), "block_m": design.block_m,
             "tile": design.tile, "compute": design.compute,
             "replicates": design.replicates}, indent=2))
        say()
        say(f"## measuring: {len(design.rungs) - len(done)} rungs to do, "
            f"{len(done)} already on disk")
        try:
            fresh, _ = measure(design, args, out_dir, done)
        except CannotRunHere as exc:
            say()
            say(f"CANNOT RUN HERE: {exc}")
            say("The plan, the prediction and the preflight above are still "
                "valid and cost nothing;")
            say("re-run with --run on the pod, or --synthetic to exercise the "
                "gates.")
            _save(out_dir, say)
            return 3
        except KeyboardInterrupt:
            say()
            say("## aborted; reporting on what reached disk")
            fresh = []
        records = existing + fresh
    else:
        say()
        say("Nothing was measured. Add --run on the pod, --synthetic to "
            "exercise the gates,")
        say("or --replay <dir> to re-report a finished run.")
        _save(out_dir, say)
        return 0

    return _analyse(say, design, records, args, out_dir, l2)


def _analyse(say, design: Design, records: list[dict], args, out_dir: Path,
             l2: int) -> int:
    known = {r.key for r in design.rungs}
    stray = [r for r in records if r.get("id") not in known]
    synthetic = bool(args.synthetic) or any(
        r.get("provenance") == "synthetic" for r in records)
    if synthetic and not args.synthetic:
        say()
        say("*** SYNTHETIC. These records were generated from a stated law "
            f"({records[0].get('law', 'unknown')}) and nothing here was "
            "measured on any hardware.")
    if stray:
        say()
        say(f"  {len(stray)} records name a rung this design does not contain "
            f"and are IGNORED. First: {stray[0].get('id')}. That means the "
            "flags differ from the ones that")
        say("  produced the file; re-run --replay with the argv in plan.json.")
    records = [r for r in records if r.get("id") in known]
    timed = [r for r in records if r.get("ms")]
    if not timed:
        say()
        say("VERDICT: NOT TESTABLE. Nothing was timed.")
        _save(out_dir, say)
        return 4

    throttled = [r["id"] for r in timed if r.get("throttled")]
    if throttled:
        say()
        say(f"  {len(throttled)} rungs drifted more than "
            f"{CLOCK_DRIFT_LIMIT * 100:.0f}% in SM clock: {throttled[:3]}. The "
            "interleaved order is what")
        say("  protects a paired difference from that, and the placebo gate is "
            "what measures it.")

    report_measurements(say, timed, design)
    report_isa(say, timed)

    results = analyse_models(design, timed, args.bootstrap, args.seed)
    by_model = samples_from(timed)
    real = {r.model: by_model[r.model]
            for r in results if not r.control and r.model in by_model}
    brackets = [r.fit.bracket for r in results if r.fit.ok and not r.control]
    pooled_bracket = (
        (statistics.median(b[0] for b in brackets),
         statistics.median(b[1] for b in brackets)) if brackets else None)
    pooled = pooled_band(real, args.bootstrap, args.seed) if real else None
    l2_shares = [r.fit.l2_share for r in results
                 if r.fit.ok and not r.control and r.fit.l2_share is not None]
    report_alphas(say, results, pooled, pooled_bracket)
    mechanism_note(say, results, l2)
    report_confounds(say, design, results)

    gates = [
        isa_gate(timed),
        correctness_gate(timed, design.compute),
        placebo_gate(timed),
        signal_gate(timed),
        linearity_gate(results),
        control_gate(results),
        band_gate(pooled, l2_shares),
        prediction_gate(pooled, pooled_bracket, design.compute),
    ]
    code = verdict(say, gates)
    _save(out_dir, say)
    return code


def _save(out_dir: Path, say: Report) -> None:
    with contextlib.suppress(OSError):
        out_dir.mkdir(parents=True, exist_ok=True)
        say.save(out_dir / "report.md")
        print()
        print(f"[alias] report written to {out_dir / 'report.md'}")
        print(f"[alias] measurements at  {out_dir / 'cells.jsonl'}")


if __name__ == "__main__":
    raise SystemExit(main())
