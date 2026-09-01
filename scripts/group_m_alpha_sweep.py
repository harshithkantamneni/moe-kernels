#!/usr/bin/env python3
"""Is `alpha` a scalar? Sweep GROUP_SIZE_M, the swizzle width, and refit it.

    python scripts/group_m_alpha_sweep.py                  # plan + power, no GPU
    python scripts/group_m_alpha_sweep.py --run            # the pod run
    python scripts/group_m_alpha_sweep.py --replay <dir>   # re-report, no GPU
    python scripts/group_m_alpha_sweep.py --synthetic monotone   # gates, no GPU
    python scripts/group_m_alpha_sweep.py --run \
        --model qwen2-57b-a14b --tokens 32,64,128,256,512,768,1024   # second arm

WHY THIS EXISTS. `alpha` is the discount L2 applies to an expert's weight
re-read when that expert spans more than one M-tile:

    Q(r) = 1 + alpha (ceil(r/BM) - 1),   AI(r) = (2r/b) / Q(r)

Today's refit put it at 0.558 (90% band 0.529-0.588) over 10,813 published rows,
and split it by GROUP_SIZE_M it read 0.570 at 1 and 0.488 at 16 -- the direction
the mechanism predicts, since GROUP_SIZE_M is exactly how many M-tiles the Triton
swizzle groups so they reuse one weight block out of L2 before moving on.

THE PUBLISHED POOL CANNOT TAKE THAT ANY FURTHER, and that is the whole reason
for a pod run. GROUP_SIZE_M 32 and 64 have ZERO discriminating rows in it, so the
trend is untested rather than established above 16. Worse, the 1-versus-16 split
is CONFOUNDED WITH BATCH: `get_default_config` sets GROUP_SIZE_M to 16 only when
`M // E > 128`, so every g=16 row in the pool is also a large-batch row, and no
published row varies g at a fixed batch. This sweep forces the axis instead of
observing it, which is the only way to break that confound.

PREDICTION, stated before anything runs, and gated numerically below:

  P1  alpha FALLS between the smallest and the largest GROUP_SIZE_M, with
      disjoint 90% bands.
  P2  no adjacent inversion: alpha is non-increasing across the ladder, to
      within the bands.
  P3  a FLOOR once the group covers one expert's M-tiles. The knee is at
      g* = tiles per expert, which runs 1.1 to 8.4 across this design's batch
      ladder with a median of 4.4, so 8, 16, 32 and 64 should agree with each
      other. The knee is a median over a ladder and not a sharp threshold, and
      the gate says so where it prints.
  P4  the effect is SPECIFIC to the multi-tile regime. In the single-tile rung,
      where no expert spans two tiles and no weight re-read is possible, the
      GROUP_SIZE_M time effect must be at most a third of the multi-tile one.
  P5  a placebo: permuting the response inside each intercept group must
      collapse every fitted alpha to near zero.

THE ESTIMATOR IS IMPORTED, NEVER REIMPLEMENTED. `scripts/alpha_refit.py` is
loaded by path and its `Observation`, `cell_key` and `fit_alpha` are used
verbatim. If it cannot be loaded this script REFUSES to run: a second estimator
that can disagree with the first would make every number here unattributable,
which is the exact failure the refit was written to end.

WHY A BATCH LADDER RATHER THAN ONE BATCH. The instruction for this experiment
said fixed batch, and one batch cannot identify alpha under this estimator. The
intercept is per (model, dtype, card, impl, timing mode, TOKEN COUNT), so a
single token count is a single intercept, and everything a single x-level says
about the LEVEL of the traffic ratio is absorbed exactly. What is left is
curvature: log(1 + alpha x) has to be consistent across cells at DIFFERENT x, and
x is set by the tile count, which is set by the batch. Simulated on this design's
own x values at 0.5% timing noise, the top rung alone returns a 90% band of
0.373-0.756 and the seven-rung ladder returns 0.552-0.580, a fourteenfold
difference in width against a published GROUP_SIZE_M effect of 0.082. The ladder
is held IDENTICAL across every GROUP_SIZE_M setting, so the comparison across
settings is still at fixed design; `--tokens 448` collapses it to one rung for
anyone who wants to watch the band blow up. That power simulation runs, and
prints, before a cent is spent.

WHAT WOULD CONFOUND THE FIT, checked where it can be:

  1. ACTIVATIONS, not just weights. An extra M-tile re-reads its expert's whole
     weight matrix (N*K elements) AND re-reads its own activation tile once per
     N-tile (BLOCK_M*K*num_pid_n elements). The ratio is exactly
     BLOCK_M / BLOCK_N, so at this design's 16/64 at most 20% of any fitted alpha
     is activation traffic. GROUP_SIZE_M moves the two terms in OPPOSITE
     directions -- a swizzle that shortens the weight reuse distance lengthens
     the activation one -- so a falling alpha is a NET statement about traffic
     per extra tile and not a pure weight-re-read measurement. The script prints
     the bound for the config it ran.
  2. LAUNCH ORDER AND WAVE QUANTISATION. GROUP_SIZE_M does not change the grid
     size, only the map from program id to (pid_m, pid_n), but it does change
     which tiles are co-resident, and the last group is ragged whenever
     num_pid_m % GROUP_SIZE_M != 0. Both are reported per cell, and P4's
     single-tile rung is the control: identical kernel config, identical weight
     set, identical routing, the only difference being that no expert spans two
     tiles. A GROUP_SIZE_M effect that survives there is not a weight re-read.
  3. WHAT THE CONTROL CANNOT CATCH, said plainly: any cost that scales with the
     extra-tile count and is not a weight re-read is absorbed into alpha by
     construction, because that is the regressor. Item 1 bounds the largest
     known such term.
  4. THE REGIME. `implied_traffic_ratio` is only a traffic bound while the cell
     is memory bound, and a compute-bound cell would pay for extra tiles in
     PADDED ARITHMETIC, which GROUP_SIZE_M cannot change -- so it would report a
     flat alpha and look like a clean refutation. Every planned cell is gated
     against the measured ridge band before anything runs.
  5. DRIFT. Settings are timed in a randomised order inside each cell, on the
     same tensors, so the GROUP_SIZE_M comparison is paired; clock and
     temperature are sampled per cell and a throttled cell is flagged.
  6. THE OVERRIDE ITSELF. vLLM's config dict is recorded from inside the call
     and checked against the forced one, so "the sweep swept nothing" is a
     failure the report names rather than a silent flat line.

WHAT IT WRITES, and it survives pod teardown. Everything lands in
`$MOE_RESULTS_DIR/group_m_alpha/<plan fingerprint>/` (default
`/workspace/results/...`, the network volume; the repo's own `results/` when
there is no `/workspace`): `plan.json`, `cells.jsonl` flushed per cell, and
`report.md`. The path is printed at the start and at the end. Re-running the
same plan RESUMES: completed cells are skipped by id, so a Ctrl-C costs one
cell, and a plan that differs in any way gets a different directory rather than
mixing two designs in one file.

EXIT CODES, because a refutation is a result and not an error:
    0  the run completed and every runnable gate passed
    1  the run completed and a gate FAILED: the prediction is refuted
    2  usage error
    3  cannot run here (no GPU, no vLLM, no estimator); nothing was measured
    4  the run completed but the design did not identify alpha: not testable
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import bytes_model as BM  # noqa: E402
from moe.routing.imbalance import expert_load, padded_rows, tile_efficiency  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec  # noqa: E402
from moe.stages import contract_for, exposed_writes  # noqa: E402

# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------

#: The swept axis. 32 and 64 are the two settings the published pool cannot
#: speak about at all, and 64 is not exotic: vLLM's own tuned mixtral H200 file
#: selects GROUP_SIZE_M 64 at its M=64 entry.
GROUP_M_LADDER = (1, 8, 16, 32, 64)

#: mixtral-8x7b, because the mechanism needs L2 PRESSURE. One expert's up-GEMM
#: weights are 235 MB against an H200's 50 MB L2, so at GROUP_SIZE_M=1, where a
#: weight block is not touched again until the next M-tile comes round, the
#: re-read is guaranteed to reach DRAM. On a 64-expert model the same pass is
#: 37 MB and can sit in L2 whatever the swizzle does, which would test nothing.
DEFAULT_MODEL = "mixtral-8x7b"

#: Forced on every cell. BLOCK_SIZE_M=16 is what vLLM actually runs through the
#: decode range, and at this study's ridge it is also the only way to be in the
#: MULTI-TILE and MEMORY-BOUND regimes at once: multi-tile needs rows per expert
#: above BLOCK_M, memory bound needs compulsory intensity below the ridge, and
#: those two windows only overlap generously at a small tile.
DEFAULT_BLOCK_M = 16

#: Everything except BLOCK_SIZE_M and the swept GROUP_SIZE_M, copied verbatim
#: from vLLM v0.27.1's tuned entry for this exact shape and card,
#: `E=8,N=14336,device_name=NVIDIA_H200.json` key 16, which reads
#: {M 16, N 64, K 256, GROUP 16, warps 4, stages 3}. Copied rather than chosen
#: so that the swept configs are configs vLLM would really run at a decode
#: batch, with GROUP_SIZE_M the single departure. `scripts/tile_sweep.py` pins a
#: set of its own for a different question and an earlier version of it wrongly
#: claimed they came from a shipped file; these do, and the file is vendored
#: under `moe/bench/hardware/vllm_configs/`.
FIXED_TILE = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "num_warps": 4, "num_stages": 3}
FIXED_TILE_SOURCE = "E=8,N=14336,device_name=NVIDIA_H200.json key 16"

#: The batch ladder. rows per expert is T*k/E, so on mixtral these are
#: 4, 8, 16, 32, 64, 96, 112 rows per expert, i.e. 1 to 7 M-tiles per expert at
#: BLOCK_M=16. The bottom rungs are the SINGLE-TILE CONTROL for P4 and the top
#: rung is the one the swizzle should help most.
#:
#: THE TOP IS CAPPED BY THE RIDGE, and the cap is tighter than the mean says.
#: T=512 looks safe on a uniform draw at 127.6 FLOP/byte, but a dirichlet draw
#: that leaves an expert empty cuts the compulsory weight bytes by an eighth and
#: pushes that same cell to 145.8, over the 90% of 160.3 the preflight allows.
#: The ladder is set by the WORST realisation it contains, not by the mean, and
#: the preflight recomputes that rather than trusting this comment.
DEFAULT_TOKENS = (16, 32, 64, 128, 256, 384, 448)

#: Routing realisations per token count. They are the ONLY thing that varies the
#: tile count inside an intercept group, so they are what identifies alpha at
#: all: `cell_key` holds model, dtype, card, impl, timing mode and token count
#: fixed, and routing is deliberately not in it.
DEFAULT_ROUTINGS = ("uniform", "zipf:0.5", "zipf:1.0", "hot:0.25", "hot:0.4",
                    "dirichlet:0.5", "dirichlet:1.0")
DEFAULT_ROUTING_SEEDS = 4

#: The two GEMM stages plus what vLLM's `fused_experts` fuses around them. It
#: restates `VllmFusedExperts.covers`, which cannot be imported without vLLM
#: installed, and `--run` checks the two agree the moment vLLM is importable.
VLLM_COVERS = ("permute", "up_gemm", "act", "down_gemm", "unpermute")
VLLM_IMPL = "vllm_fused_experts"

#: Bandwidth used to turn a time into `implied_traffic_ratio` when no
#: calibration for the attached card is on disk. THE FITTED ALPHA DOES NOT
#: DEPEND ON IT: the ratio is time x bandwidth / compulsory bytes, the fit is in
#: logs, and a constant factor is absorbed exactly by the group intercept. A
#: test pins that invariance. It is still reported, because the RATIO's absolute
#: level is read by a human and that level does depend on it.
NOMINAL_BANDWIDTH_BYTES_S = 4.8e12


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

#: A setting with fewer discriminating rows than this has not measured alpha,
#: whatever number the optimiser returns. Matches `alpha_refit._split_line`'s
#: own refusal to print a split's alpha below 10, doubled because a forced sweep
#: has no excuse for being thin.
MIN_DISCRIMINATING = 20

#: Fewer intercept groups than this and only the level is identified, not the
#: curvature. Two is the arithmetic minimum; three is the smallest number that
#: can show the curvature is consistent rather than merely fitted.
MIN_INTERCEPT_GROUPS = 3

#: A 90% band wider than this cannot resolve the effect this experiment is for:
#: the published GROUP_SIZE_M split differs by 0.082 between g=1 and g=16.
MAX_BAND_WIDTH = 0.15

#: P4. The single-tile rung's paired time effect must be at most this fraction
#: of the multi-tile rung's, or the GROUP_SIZE_M effect is not specific to the
#: regime where weight re-reads exist.
CONTROL_SPECIFICITY = 1.0 / 3.0

#: P5. A response permuted inside its intercept group must not fit an alpha
#: bigger than this.
PLACEBO_MAX_ALPHA = 0.10

#: P4 is a RATIO of two effect sizes, so it needs a floor: below this the
#: paired time table shows no multi-tile effect at all and there is nothing for
#: the control to be specific about. Without it, a run where GROUP_SIZE_M does
#: nothing anywhere fails P4 on the ratio of two noise floors and reads as
#: "the effect is an artefact" when the honest answer is "there is no effect".
MIN_ATTRIBUTABLE_EFFECT = 0.01

#: P3's slack, in units of one setting's 90% band. TWO, not one, because the
#: gate compares the RANGE of several independent estimates and a range is
#: wider than any one band: on the synthetic plateau, four settings whose bands
#: are 0.013 wide scattered over 0.018, which a one-band slack calls a violation
#: of a floor that was planted flat by construction.
FLOOR_SLACK_BANDS = 2.0

#: Cells whose compulsory arithmetic intensity is above this fraction of the
#: LOW end of the ridge band are refused: `implied_traffic_ratio` is only a
#: traffic bound below the ridge, and a compute-bound cell pays for extra tiles
#: in padded arithmetic, which GROUP_SIZE_M cannot move.
MEMORY_BOUND_MARGIN = 0.90

#: Timing noise assumed by the power simulation, as a fraction of log time.
#: `moe/bench/crossing.py` records that repeated measurements of one cell
#: reproduce to about 0.2%; 0.5% is the pessimistic end of that.
POWER_NOISE = 0.005

#: The effect the design has to be able to see, from the published split
#: (0.570 at GROUP_SIZE_M=1 against 0.488 at 16).
PUBLISHED_GROUP_M_EFFECT = 0.082


# --------------------------------------------------------------------------
# the estimator, imported
# --------------------------------------------------------------------------

class EstimatorMissing(RuntimeError):
    """`scripts/alpha_refit.py` could not be loaded, so there is no fit to run.

    Raised rather than falling back to a local implementation. The point of this
    experiment is that five fitted alphas are comparable with each other AND
    with the 0.558 the refit published; a second estimator, however carefully
    written, makes every one of those comparisons unattributable.
    """


def load_alpha_refit(path: Path | None = None):
    """`scripts/alpha_refit.py` as a module.

    By path because `scripts/` is not a package and never has been, which is the
    same shape `tests/test_alpha_refit.py` needs. Registered in `sys.modules`
    BEFORE execution because `@dataclass` resolves its annotations through
    `sys.modules[cls.__module__]`, and a module that is not there yet fails
    inside the decorator with an AttributeError about NoneType.
    """
    path = path or (ROOT / "scripts" / "alpha_refit.py")
    spec = importlib.util.spec_from_file_location("alpha_refit", path)
    if spec is None or spec.loader is None:
        raise EstimatorMissing(f"no importable estimator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any failure means no fit
        raise EstimatorMissing(
            f"{path} did not import ({type(exc).__name__}: {exc}). This script "
            "will not substitute a second estimator: the whole comparison "
            "depends on these alphas being fitted by the same code as the "
            "published 0.558.") from exc
    for name in ("Observation", "fit_alpha", "cell_key", "RIDGE_BAND"):
        if not hasattr(module, name):
            raise EstimatorMissing(f"{path} has no {name}; it is not the estimator")
    return module


# --------------------------------------------------------------------------
# the cost model for one cell
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Span:
    """Duck-types `StageSpan` for `bytes_model.span_cost`, without vLLM.

    `moe/baselines/vllm_fused_moe.py` imports vLLM at module scope on purpose,
    so the real span object cannot be built on a laptop and the plan could not
    be costed off the GPU box. Reads and writes come from the same
    `contract_for` / `exposed_writes` the real class uses, so this is the same
    span by construction and not a second opinion about what fused_experts
    covers.
    """

    covers: tuple[str, ...] = VLLM_COVERS
    name: str = VLLM_IMPL

    @property
    def reads(self) -> frozenset[str]:
        return contract_for(self.covers).reads

    @property
    def writes(self) -> frozenset[str]:
        return exposed_writes(self.covers)


@dataclass(frozen=True)
class Cell:
    """One routing realisation at one token count, costed. No timing in here.

    Everything on this object is computable on a laptop, which is what makes the
    whole design -- its tile counts, its regime, its identification -- checkable
    before the pod is rented.
    """

    model: str
    tokens: int
    dtype: str
    routing_label: str
    routing_seed: int
    block_m: int
    counts: tuple[int, ...]
    active_experts: int
    total_rows: int
    max_rows: int
    m_tiles: float
    tile_eff: float
    compulsory_bytes: float
    per_expert_bytes: float
    flops: float

    @property
    def extra_tiles(self) -> float:
        return max(self.m_tiles - self.active_experts, 0.0)

    @property
    def x(self) -> float:
        """The regressor: extra tile weight bytes over the compulsory total."""
        return self.per_expert_bytes * self.extra_tiles / self.compulsory_bytes

    @property
    def tiles_per_expert(self) -> float:
        return self.m_tiles / max(self.active_experts, 1)

    @property
    def single_tile(self) -> bool:
        """Every active expert fits in ONE M-tile, so no weight re-read exists.

        The P4 control. Tested on the realisation rather than on the mean,
        because a uniform draw at 8 mean rows per expert still puts 14 on one of
        them, and a cell that spans two tiles is not a control.
        """
        return self.extra_tiles == 0.0

    @property
    def arith_intensity(self) -> float:
        return self.flops / max(self.compulsory_bytes, 1.0)

    @property
    def key(self) -> str:
        return (f"{self.model}|T{self.tokens}|{self.dtype}|bm{self.block_m}"
                f"|{self.routing_label}|s{self.routing_seed}")


def routing_from_label(label: str) -> RoutingSpec:
    """`"zipf:1.0"` -> `RoutingSpec("zipf", 1.0)`, matching `RoutingSpec.label`.

    Parsed here rather than taking a kind and a param as two flags so that the
    label written into every record round-trips back to the spec that made it.
    """
    kind, _, param = label.partition(":")
    return RoutingSpec(kind, float(param) if param else 0.0)


def build_cell(model: str, tokens: int, dtype: str, routing_label: str,
               routing_seed: int, block_m: int) -> Cell:
    """Sample one routing realisation and cost it. CPU only.

    The routing seed is separate from `BenchSpec.seed` on purpose. `make_inputs`
    caches weights on (model, dtype, seed, device, scale), so folding the
    routing seed into the spec would redraw 2.8 GB of mixtral weights for every
    cell of the sweep -- minutes of metered time, and a different weight set per
    cell, which would put a nuisance axis inside the comparison.
    """
    try:
        import torch

        from moe.routing.distributions import sample_topk_ids
    except ImportError as exc:  # pragma: no cover - torch is a hard dep here
        raise SystemExit(
            f"planning needs torch to draw a routing realisation ({exc}). It is "
            "CPU work; install torch or run this on the pod.") from exc
    cfg = MODEL_CONFIGS[model]
    routing = routing_from_label(routing_label)
    spec = BenchSpec(cfg, num_tokens=tokens, dtype=dtype, routing=routing)
    ids = sample_topk_ids(routing, tokens, cfg.num_experts, cfg.top_k,
                          seed=routing_seed, device="cpu")
    counts = tuple(int(v) for v in
                   torch.bincount(ids.flatten(), minlength=cfg.num_experts).tolist())
    load = expert_load(counts)
    cost = BM.pipeline_cost([_Span()], spec, load.active_experts)
    per_expert = float(sum(BM.weight_bytes_for_stage(spec, s, 1)
                           for s in ("up_gemm", "down_gemm")))
    return Cell(
        model=model, tokens=tokens, dtype=dtype, routing_label=routing_label,
        routing_seed=routing_seed, block_m=block_m, counts=counts,
        active_experts=load.active_experts, total_rows=load.total_rows,
        max_rows=load.max_rows,
        # From the real histogram, which the published rows do not keep. This is
        # exactly sum(ceil(rows_e / BLOCK_M)) and not a reconstruction, so it
        # stays right where an expert spans many tiles -- the regime this whole
        # sweep lives in, and the one `crossing.m_tiles_for_row` has to refuse.
        m_tiles=padded_rows(counts, block_m) / block_m,
        tile_eff=tile_efficiency(counts, block_m),
        compulsory_bytes=float(cost.bytes_total), per_expert_bytes=per_expert,
        flops=float(cost.flops))


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    model: str
    dtype: str
    block_m: int
    tokens: tuple[int, ...]
    group_m: tuple[int, ...]
    routings: tuple[str, ...]
    seeds: int
    passes: int
    cells: tuple[Cell, ...]
    fixed_tile: dict

    @property
    def fingerprint(self) -> str:
        """Stable hash of everything that defines the experiment.

        The output directory is named after it, so re-running the same plan
        resumes into the same file and a plan that differs anywhere gets a fresh
        one. Mixing two designs in one cells.jsonl would be undetectable
        afterwards, which is why this is not just a timestamp.
        """
        payload = json.dumps({
            "model": self.model, "dtype": self.dtype, "block_m": self.block_m,
            "tokens": list(self.tokens), "group_m": list(self.group_m),
            "routings": list(self.routings), "seeds": self.seeds,
            "passes": self.passes, "fixed": self.fixed_tile,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def multi(self) -> list[Cell]:
        return [c for c in self.cells if not c.single_tile]

    @property
    def control(self) -> list[Cell]:
        return [c for c in self.cells if c.single_tile]

    @property
    def n_measurements(self) -> int:
        return len(self.cells) * len(self.group_m) * self.passes


def build_plan(args) -> Plan:
    cells = tuple(
        build_cell(args.model, tokens, args.dtype, routing, seed, args.block_m)
        for tokens in args.tokens
        for routing in args.routings
        for seed in range(args.seeds))
    return Plan(model=args.model, dtype=args.dtype, block_m=args.block_m,
                tokens=tuple(args.tokens), group_m=tuple(args.group_m),
                routings=tuple(args.routings), seeds=args.seeds,
                passes=args.passes, cells=cells, fixed_tile=dict(FIXED_TILE))


@dataclass
class Gate:
    """One numeric verdict. `ok=None` means the data cannot answer it.

    THE BUG THIS COERCION FIXES, which was live for one run of this script.
    `alpha_refit.fit_alpha` returns a numpy float, so every comparison built
    from a fitted alpha is a `numpy.bool_`. `numpy.bool_(False) is False` is
    FALSE, while `{True: ..., False: ...}[numpy.bool_(False)]` succeeds, so a
    failed gate printed "[FAIL]" on its own line and was then invisible to the
    `g.ok is False` scan that decides the verdict: the report said PREDICTION
    HELD underneath a gate that had failed. A verdict must not be able to
    disagree with the table above it.
    """

    name: str
    ok: bool | None
    detail: str

    def __post_init__(self) -> None:
        self.ok = None if self.ok is None else bool(self.ok)

    @property
    def label(self) -> str:
        return {True: "PASS", False: "FAIL", None: "NOT TESTABLE"}[self.ok]


def preflight(plan: Plan, ridge_low: float) -> list[Gate]:
    """Refuse a design that cannot answer the question, before it is paid for.

    Every one of these has a way of passing silently and producing a confident
    wrong answer: a compute-bound cell fits alpha to padded arithmetic, a
    single-tile-everywhere plan fits it to nothing, and a plan with no control
    rung cannot tell an L2 effect from a launch-order one.
    """
    gates: list[Gate] = []
    worst = max(plan.cells, key=lambda c: c.arith_intensity)
    limit = MEMORY_BOUND_MARGIN * ridge_low
    gates.append(Gate(
        "regime: every cell is memory bound",
        worst.arith_intensity <= limit,
        f"max compulsory AI {worst.arith_intensity:.1f} at T={worst.tokens} "
        f"against {MEMORY_BOUND_MARGIN:.0%} of ridge {ridge_low} = {limit:.1f}"))

    multi = plan.multi
    deepest = max((c.tiles_per_expert for c in multi), default=0.0)
    gates.append(Gate(
        "regime: the multi-tile rung has re-reads to save",
        len(multi) >= MIN_DISCRIMINATING and deepest >= 2.0,
        f"{len(multi)} of {len(plan.cells)} cells span more than one tile per "
        f"expert, deepest {deepest:.2f} tiles/expert"))

    control = plan.control
    gates.append(Gate(
        "control: a single-tile rung exists",
        len(control) >= 3,
        f"{len(control)} cells where every active expert is exactly one tile "
        f"(tokens {sorted({c.tokens for c in control})})"))

    knee = statistics.median([c.tiles_per_expert for c in multi]) if multi else 0.0
    gates.append(Gate(
        "design: the predicted floor is inside the swept ladder",
        bool(multi) and min(plan.group_m) < knee < max(plan.group_m),
        f"predicted knee g* = {knee:.1f} tiles per expert, ladder "
        f"{list(plan.group_m)}"))

    spread = ({round(c.x, 3) for c in plan.cells})
    gates.append(Gate(
        "design: the tile count varies inside a token count",
        _within_group_x_spread(plan) > 0.0,
        f"{len(spread)} distinct x values; within-token spread "
        f"{_within_group_x_spread(plan):.3f}"))
    return gates


def _within_group_x_spread(plan: Plan) -> float:
    """Largest spread of the regressor inside any one token count.

    Zero means no cell can move alpha: the intercept absorbs the level and there
    is nothing else. This is the one preflight number that is about
    IDENTIFICATION rather than about the physics.
    """
    by_tokens: dict[int, list[float]] = collections.defaultdict(list)
    for c in plan.cells:
        by_tokens[c.tokens].append(c.x)
    return max((max(v) - min(v) for v in by_tokens.values()), default=0.0)


def confound_bound(plan: Plan) -> float:
    """Activation traffic per extra M-tile, over weight traffic per extra M-tile.

    An extra M-tile reads its expert's whole weight matrix once (N*K elements,
    summed over the N-tiles it visits) and reads its own activation tile once
    per N-tile (BLOCK_M*K per N-tile, num_pid_n of them). Dividing,
    everything but BLOCK_M / BLOCK_N cancels. So this ratio, and not a hand
    wave, bounds how much of a fitted alpha could be activation rather than
    weight traffic -- and it is the term GROUP_SIZE_M moves the OPPOSITE way,
    since grouping M-tiles lengthens the activation reuse distance.
    """
    return plan.block_m / float(plan.fixed_tile["BLOCK_SIZE_N"])


# --------------------------------------------------------------------------
# turning measurements into the estimator's own Observations
# --------------------------------------------------------------------------

def observation(AR, cell: Cell, ms: float, group_m: int, gpu: str,
                bandwidth: float, l2_flush: bool):
    """One `alpha_refit.Observation`, built from a measured cell.

    `routing` carries the routing label AND the seed, packed, because the
    Observation has nowhere else to put the seed and the bootstrap below
    resamples the routing REALISATION -- label plus seed -- as its cluster.
    """
    from moe.bench.calibrate import implied_traffic_ratio
    ratio = implied_traffic_ratio(cell.compulsory_bytes, ms, bandwidth)
    return AR.Observation(
        traffic_ratio=ratio,
        compulsory_bytes=cell.compulsory_bytes,
        per_expert_bytes=cell.per_expert_bytes,
        active_experts=float(cell.active_experts),
        m_tiles=cell.m_tiles,
        block_m=cell.block_m,
        group_m=group_m,
        tile_provenance="vllm_override_forced",
        model=cell.model, dtype=cell.dtype, gpu=gpu, impl=VLLM_IMPL,
        tokens=cell.tokens,
        routing=f"{cell.routing_label}/s{cell.routing_seed}",
        l2_flush=l2_flush, cuda_graph=False,
        tile_columns=(("load_total_rows", str(cell.total_rows)),
                      ("load_active_experts", str(cell.active_experts)),
                      ("load_max_rows", str(cell.max_rows))))


def band(AR, observations, draws: int, seed: int,
         quantiles: tuple[float, float] = (0.05, 0.95)) -> tuple[float, float] | None:
    """90% band by resampling ROUTING REALISATIONS inside each intercept group.

    NOT `alpha_refit.bootstrap_band`, and the difference is the resampling unit
    rather than a disagreement about method. That function clusters on the
    intercept group because in the published pool the rows inside one group are
    replicates of ONE cell measured at several seeds and trials -- six views of
    one thermal state, not six measurements. Here the rows inside one group are
    DIFFERENT routing realisations, deliberately drawn to vary the tile count,
    and they are the independently sampled unit. Clustering on the group instead
    would resample five clusters and report a band over nothing.

    Resampling is stratified: each group keeps its own number of realisations,
    so a draw cannot empty a token count and destroy the curvature the fit needs.
    Intercept ids follow `cell_key`, never the draw index, because two copies of
    one drawn realisation ARE the same physical cell and must share its level.
    """
    groups: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for o in observations:
        groups[AR.cell_key(o)][o.routing].append(o)
    if sum(len(v) for v in groups.values()) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        rows, ids = [], []
        for index, (_, realisations) in enumerate(sorted(groups.items(),
                                                         key=lambda kv: str(kv[0]))):
            keys = list(realisations)
            for _ in range(len(keys)):
                members = realisations[rng.choice(keys)]
                rows.extend(members)
                ids.extend([index] * len(members))
        try:
            samples.append(AR.fit_alpha(rows, group_ids=ids))
        except ValueError:
            continue
    if len(samples) < 2:
        return None
    lo = float(np.quantile(samples, quantiles[0]))
    hi = float(np.quantile(samples, quantiles[1]))
    return lo, hi


def residual_rms(AR, observations, alpha: float) -> float:
    """Within-group RMS of `log ratio - log(1 + alpha x)`, in log units.

    A MODEL-ADEQUACY number, and it is not the same thing as the bootstrap band.
    The band says how much the fitted alpha moves under resampling; this says
    whether the one-parameter model describes the rows at all. If it is several
    times the timing repeatability then the residual is misspecification rather
    than noise, the resampling band is optimistic, and no interval printed here
    should be read as a confidence statement about the physics.
    """
    if len(observations) < 2:
        return float("nan")
    ids = {}
    residuals = []
    groups: dict = collections.defaultdict(list)
    for o in observations:
        key = AR.cell_key(o)
        ids.setdefault(key, len(ids))
        groups[key].append(math.log(o.traffic_ratio) - math.log1p(alpha * o.x))
    for values in groups.values():
        mean = statistics.fmean(values)
        residuals.extend(v - mean for v in values)
    return math.sqrt(statistics.fmean(r * r for r in residuals))


def placebo_alpha(AR, observations, seed: int) -> float | None:
    """Refit after permuting the response inside each intercept group.

    The same placebo `alpha_refit` runs on the published pool, for the same
    reason: it breaks the pairing between a row's traffic ratio and its tile
    count while leaving both marginals and the whole group structure alone. A
    fit that survives it is fitting the group structure and not the tile.
    """
    import dataclasses as _dc
    rng = random.Random(seed)
    groups: dict = collections.defaultdict(list)
    for o in observations:
        groups[AR.cell_key(o)].append(o)
    shuffled = []
    for members in groups.values():
        ratios = [o.traffic_ratio for o in members]
        rng.shuffle(ratios)
        shuffled.extend(_dc.replace(o, traffic_ratio=r)
                        for o, r in zip(members, ratios, strict=True))
    try:
        return float(AR.fit_alpha(shuffled))
    except ValueError:
        return None


@dataclass
class SettingFit:
    group_m: int
    n: int
    n_disc: int
    n_groups: int
    alpha: float | None
    interval: tuple[float, float] | None
    rms: float
    placebo: float | None

    @property
    def identified(self) -> bool:
        return (self.alpha is not None
                and self.n_disc >= MIN_DISCRIMINATING
                and self.n_groups >= MIN_INTERCEPT_GROUPS
                and self.interval is not None
                and (self.interval[1] - self.interval[0]) <= MAX_BAND_WIDTH)

    @property
    def width(self) -> float:
        return (self.interval[1] - self.interval[0]) if self.interval else float("inf")


def fit_per_setting(AR, records, plan: Plan, gpu: str, bandwidth: float,
                    l2_flush: bool, draws: int, seed: int) -> list[SettingFit]:
    """One fit per GROUP_SIZE_M, over the multi-tile rungs only.

    The single-tile rung is excluded from the FIT and not from the run: its
    cells sit at x = 0, so they contribute an intercept and nothing else, and
    leaving them in would inflate `n` in a table whose whole job is to say how
    much evidence there is. They are used, in full, by the P4 control.
    """
    cells = {c.key: c for c in plan.cells}
    by_g: dict[int, list] = collections.defaultdict(list)
    for rec in records:
        cell = cells.get(rec["cell"])
        if cell is None or cell.single_tile or rec.get("ms_p50", 0.0) <= 0.0:
            continue
        by_g[int(rec["group_m"])].append(
            observation(AR, cell, float(rec["ms_p50"]), int(rec["group_m"]),
                        gpu, bandwidth,
                        # From the RECORD, not from this process's flags: a
                        # replay run with a different --no-l2-flush would
                        # otherwise relabel the timing mode of rows it did not
                        # measure.
                        bool(rec.get("l2_flush", l2_flush))))
    out = []
    for group_m in sorted(by_g):
        obs = by_g[group_m]
        n_disc = sum(1 for o in obs if o.discriminating)
        n_groups = len({AR.cell_key(o) for o in obs})
        try:
            # float(), because fit_alpha answers in numpy and a numpy bool built
            # from it is not `is False`. See Gate.__post_init__.
            alpha = float(AR.fit_alpha(obs))
        except ValueError:
            alpha = None
        out.append(SettingFit(
            group_m=group_m, n=len(obs), n_disc=n_disc, n_groups=n_groups,
            alpha=alpha,
            interval=band(AR, obs, draws, seed) if alpha is not None else None,
            rms=residual_rms(AR, obs, alpha) if alpha is not None else float("nan"),
            placebo=placebo_alpha(AR, obs, seed) if alpha is not None else None))
    return out


# --------------------------------------------------------------------------
# the paired time table, which is what the control gate reads
# --------------------------------------------------------------------------

def paired_ratios(records, cells: dict, reference_g: int,
                  want_single_tile: bool) -> dict[int, float]:
    """Geometric mean of ms(g) / ms(reference g), over cells measured at both.

    PAIRED, because every GROUP_SIZE_M setting is timed on the same tensors
    inside the same cell, so the cell's own level cancels exactly. An unpaired
    mean over an aborted run would compare a different set of cells at each
    setting and read a composition change as an effect.
    """
    by_cell: dict[tuple, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for rec in records:
        cell = cells.get(rec["cell"])
        if cell is None or cell.single_tile != want_single_tile:
            continue
        if rec.get("ms_p50", 0.0) > 0.0:
            by_cell[(rec["cell"], rec.get("pass", 0))][int(rec["group_m"])].append(
                float(rec["ms_p50"]))
    logs: dict[int, list[float]] = collections.defaultdict(list)
    for settings in by_cell.values():
        if reference_g not in settings:
            continue
        base = statistics.fmean(settings[reference_g])
        for group_m, values in settings.items():
            logs[group_m].append(math.log(statistics.fmean(values) / base))
    return {g: math.exp(statistics.fmean(v)) for g, v in sorted(logs.items()) if v}


def effect_size(ratios: dict[int, float]) -> float:
    """How much the fastest setting beats the reference, as a fraction.

    Signed away deliberately: the control's effect is read as a magnitude,
    because a launch-order artefact that makes things SLOWER with g is just as
    disqualifying as one that makes them faster.
    """
    if not ratios:
        return 0.0
    return max(abs(1.0 - r) for r in ratios.values())


# --------------------------------------------------------------------------
# measuring, which is the only part that needs the box
# --------------------------------------------------------------------------

class CannotRunHere(RuntimeError):
    """No GPU, no vLLM, or no override hook. Named so `main` can exit 3."""


def find_override_config():
    """vLLM's `override_config` context manager, wherever this version keeps it.

    Found through `_framework_config.bindings_of`, the repo's own module probe,
    rather than a second hardcoded import path: `try_get_optimal_moe_config`
    consults `get_config()` first and a truthy value bypasses both the tuned file
    and the default ladder, so this is the hook that makes the sweep a sweep, and
    a wrong guess would silently force nothing while every setting still printed
    a number.
    """
    import importlib

    from moe.baselines._framework_config import (
        VLLM_CONFIG_MODULES,
        bindings_of,
    )
    for module in bindings_of("override_config"):
        return module.override_config, module.__name__
    for name in VLLM_CONFIG_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise CannotRunHere(
            f"vLLM is installed and {name} imports, but nothing there exposes "
            "override_config. Without that hook GROUP_SIZE_M cannot be forced "
            "and this sweep would time one config five times and call it flat.")
    raise CannotRunHere(
        "vLLM is not importable in this interpreter. Run inside the vllm venv: "
        "/workspace/venvs/vllm/bin/python scripts/group_m_alpha_sweep.py --run")


def measure(plan: Plan, args, out_dir: Path, done: set[str]) -> tuple[list[dict], dict]:
    """Time every (cell, GROUP_SIZE_M) not already on disk. Appends as it goes.

    Order is deliberate. The outer loop is the cell, so one routing realisation
    is drawn once and every setting is timed on the SAME tensors; the inner loop
    over settings is shuffled per cell, so a clock that drifts during the run
    cannot align with the swept axis. Each result is flushed before the next
    cell starts, which is what makes a Ctrl-C cost one cell.
    """
    import torch

    from moe.baselines._framework_config import (
        TileCapture,
        recording_tile_config,
        vllm_override_active,
    )
    from moe.bench.timing import ClockState, clock_drift, time_eager
    if not torch.cuda.is_available():
        raise CannotRunHere("no CUDA device; --run needs the pod")
    override_config, where = find_override_config()
    try:
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    except ImportError as exc:
        # The override hook resolved and the entry point did not, which is a
        # version skew rather than an absent vLLM, so it gets its own message
        # instead of a traceback out of the middle of a metered session.
        raise CannotRunHere(
            f"vLLM's fused_experts entry point did not import ({exc}). The "
            "config hook was found, so this is a version skew, not a missing "
            "install.") from exc

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.reference.torch_ref import make_inputs
    from moe.routing.distributions import sample_topk_ids

    gpu = torch.cuda.get_device_properties(0).name
    meta = {"gpu": gpu, "override_hook": f"{where}.override_config"}
    print(f"[group_m] override hook {meta['override_hook']}  device {gpu}")

    cfg = MODEL_CONFIGS[plan.model]
    rng = random.Random(args.seed)
    written: list[dict] = []
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    path = out_dir / "cells.jsonl"

    for pass_index in range(plan.passes):
        for cell in plan.cells:
            todo = [g for g in plan.group_m
                    if _record_id(cell, g, pass_index) not in done]
            if not todo:
                continue
            if deadline and time.time() > deadline:
                print("[group_m] --max-minutes reached; stopping cleanly")
                return written, meta
            spec = BenchSpec(cfg, num_tokens=cell.tokens, dtype=plan.dtype,
                             routing=routing_from_label(cell.routing_label))
            x, weights = make_inputs(spec, device="cuda")
            ids = sample_topk_ids(routing_from_label(cell.routing_label),
                                  cell.tokens, cfg.num_experts, cfg.top_k,
                                  seed=cell.routing_seed, device="cuda")
            topk_w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                                device="cuda")
            kw = vllm_call_kwargs(spec)
            kw["activation"] = MoEActivation(kw["activation"])

            def call(_x=x, _w=weights, _tw=topk_w, _ids=ids, _kw=kw):
                return fused_experts(hidden_states=_x, w1=_w.w1, w2=_w.w2,
                                     topk_weights=_tw, topk_ids=_ids, **_kw)

            # GROUP_SIZE_M REORDERS PROGRAM IDS AND MUST NOT CHANGE THE
            # ANSWER. Each program still owns the same (pid_m, pid_n) block and
            # accumulates over K in the same order, so every setting has to
            # return the identical tensor, bit for bit. Checking it costs one
            # comparison per setting and catches the two ways this experiment
            # could be measuring something other than the same computation: a
            # forced config that is not actually legal for the shape, and a
            # fused_experts that mutates `hidden_states` in place, which would
            # let every later setting run on decayed input and time a different
            # problem.
            x_before = x.detach().clone()
            reference, reference_g = None, None
            rng.shuffle(todo)
            for group_m in todo:
                conf = dict(plan.fixed_tile, BLOCK_SIZE_M=plan.block_m,
                            GROUP_SIZE_M=group_m)
                before = ClockState.sample()
                capture = TileCapture()
                try:
                    with override_config(conf):
                        with recording_tile_config(capture):
                            out = call()
                        observed_override = vllm_override_active()
                        input_unchanged = bool(torch.equal(x, x_before))
                        if reference is None:
                            reference, reference_g = out.detach().clone(), group_m
                            matches, max_diff = True, 0.0
                        else:
                            matches = bool(torch.equal(out, reference))
                            max_diff = float((out.float() - reference.float())
                                             .abs().max().item())
                        del out
                        timing = time_eager(call, warmup=args.warmup,
                                            trials=args.trials,
                                            l2_flush=args.l2_flush)
                except Exception as exc:  # noqa: BLE001 - one cell must not end the run
                    record = _record(cell, group_m, pass_index, plan, math.nan,
                                     error=f"{type(exc).__name__}: {exc}")
                    print(f"[group_m] FAILED {record['id']}: {record['error']}")
                    _append(path, record)
                    written.append(record)
                    continue
                after = ClockState.sample()
                drift, throttled = clock_drift(before, after)
                seen = capture.calls[0].config if capture.calls else None
                record = _record(
                    cell, group_m, pass_index, plan, timing.ms_p50,
                    ms_std=timing.ms_std, jitter=timing.jitter_p90_over_p50,
                    samples=timing.samples, l2_flush=timing.l2_flush,
                    clock_drift_pct=drift, throttled=throttled,
                    override_active=bool(observed_override),
                    observed_config=seen,
                    matches_reference=matches, max_abs_diff=max_diff,
                    reference_group_m=reference_g,
                    input_unchanged=input_unchanged)
                _append(path, record)
                written.append(record)
    return written, meta


def _record_id(cell: Cell, group_m: int, pass_index: int) -> str:
    return f"{cell.key}|g{group_m}|p{pass_index}"


def _record(cell: Cell, group_m: int, pass_index: int, plan: Plan, ms: float,
            **extra) -> dict:
    """One measured row, with everything an analysis or an audit needs.

    The cell's derived quantities are copied in rather than recomputed at read
    time so that `--replay` reports exactly the numbers the run used, even if
    the cost model or the sampler changes underneath it.
    """
    record = {
        "kind": "cell", "id": _record_id(cell, group_m, pass_index),
        "cell": cell.key, "pass": pass_index, "group_m": group_m,
        "block_m": plan.block_m, "model": cell.model, "tokens": cell.tokens,
        "dtype": cell.dtype, "routing": cell.routing_label,
        "routing_seed": cell.routing_seed,
        "ms_p50": None if ms != ms else float(ms),
        "active_experts": cell.active_experts, "total_rows": cell.total_rows,
        "max_rows": cell.max_rows, "m_tiles": cell.m_tiles,
        "tile_eff": cell.tile_eff, "single_tile": cell.single_tile,
        "x": cell.x, "compulsory_bytes": cell.compulsory_bytes,
        "per_expert_bytes": cell.per_expert_bytes,
        "arith_intensity": cell.arith_intensity,
    }
    record.update(extra)
    return record


def _append(path: Path, record: dict) -> None:
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
        if row.get("kind") == "cell":
            out.append(row)
    return out


# --------------------------------------------------------------------------
# synthetic measurements, so the gates can be exercised without a GPU
# --------------------------------------------------------------------------

#: The three laws `--synthetic` can generate. Each exists to show one gate
#: doing its job before the pod is rented.
SYNTHETIC_LAWS = ("monotone", "flat", "order")


def synthesise(plan: Plan, law: str, seed: int, noise: float = POWER_NOISE,
               alpha_hi: float = 0.60, alpha_lo: float = 0.12) -> list[dict]:
    """Records generated FROM a stated law, labelled synthetic everywhere.

    `monotone`  alpha falls with GROUP_SIZE_M and floors at the knee. Every gate
                should pass, which is what "the gates can see the effect" means.
    `flat`      alpha is a scalar. P1 must FAIL. Without this the gates could be
                passing because they always pass.
    `order`     alpha is a scalar AND a per-setting level shift of the same size
                as the real effect is applied to BOTH rungs. P1 still fails and
                the P4 control fails too, which is the shape of a GROUP_SIZE_M
                result that is really a launch-order artefact.

    Nothing here touches a GPU, and every record carries provenance "synthetic"
    so a report built from one cannot be quoted as a measurement.
    """
    if law not in SYNTHETIC_LAWS:
        raise ValueError(f"unknown law {law!r}; known: {SYNTHETIC_LAWS}")
    rng = random.Random(seed)
    knee = statistics.median([c.tiles_per_expert for c in plan.multi] or [8.0])
    out = []
    for pass_index in range(plan.passes):
        for cell in plan.cells:
            base = cell.compulsory_bytes / NOMINAL_BANDWIDTH_BYTES_S * 1e3
            for group_m in plan.group_m:
                reach = min(1.0, math.log2(group_m) / math.log2(max(knee, 2.0)))
                if law == "monotone":
                    alpha, level = alpha_hi - (alpha_hi - alpha_lo) * reach, 1.0
                elif law == "flat":
                    alpha, level = alpha_hi, 1.0
                else:
                    alpha, level = alpha_hi, 1.0 - 0.25 * reach
                ms = base * level * (1.0 + alpha * cell.x) * math.exp(
                    rng.gauss(0.0, noise))
                out.append(_record(
                    cell, group_m, pass_index, plan, ms,
                    provenance="synthetic", law=law, planted_alpha=alpha,
                    l2_flush=False,
                    # A law defines ONE computation, so the cross-setting
                    # comparison a real run makes is satisfied by construction.
                    # Emitted rather than left absent so that the correctness
                    # gate is exercised in the same shape it will meet on the
                    # pod; a test flips it to prove the gate bites.
                    matches_reference=True, max_abs_diff=0.0,
                    input_unchanged=True))
    return out


def power_band(AR, plan: Plan, alpha: float, noise: float, draws: int,
               seed: int) -> tuple[float, float] | None:
    """The band this DESIGN would return, on its own x values, at a known alpha.

    A power analysis rather than a result, and it is the number that decides
    whether the pod session is worth booking: if the band is wider than the
    effect being looked for, the run cannot answer the question however clean
    the hardware is. Uses the real planted-alpha simulation and the real
    estimator, so it is not an analytic approximation of either.
    """
    rng = random.Random(seed)
    obs = []
    levels = {t: rng.uniform(0.5, 2.0) for t in plan.tokens}
    for cell in plan.multi:
        ratio = levels[cell.tokens] * (1.0 + alpha * cell.x) * math.exp(
            rng.gauss(0.0, noise))
        obs.append(AR.Observation(
            traffic_ratio=ratio, compulsory_bytes=cell.compulsory_bytes,
            per_expert_bytes=cell.per_expert_bytes,
            active_experts=float(cell.active_experts), m_tiles=cell.m_tiles,
            block_m=cell.block_m, group_m=1, tile_provenance="simulated",
            model=cell.model, dtype=cell.dtype, gpu="simulated", impl=VLLM_IMPL,
            tokens=cell.tokens,
            routing=f"{cell.routing_label}/s{cell.routing_seed}",
            l2_flush=False, cuda_graph=False, tile_columns=()))
    return band(AR, obs, draws, seed)


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


def estimated_seconds(plan: Plan, warmup: int, trials: int,
                      target_ms: float = 200.0) -> float:
    """Rough wall clock for the whole sweep, from the byte model. NOT a promise.

    Printed because this runs on a metered box and "how long is this" should not
    require starting it. `time_eager` calibrates its iteration count so that one
    trial is about `target_ms` of kernel, so the timed part is nearly constant
    per cell and the warmup is what scales with the cell's own time.
    """
    per_cell = 0.0
    for cell in plan.cells:
        ms = cell.compulsory_bytes * (1.0 + 0.558 * cell.x) / NOMINAL_BANDWIDTH_BYTES_S * 1e3
        per_cell += len(plan.group_m) * (trials * target_ms + warmup * ms) / 1e3
    return per_cell * plan.passes


def report_plan(say, plan: Plan, gates: list[Gate], ridge: tuple[float, float],
                warmup: int = 10, trials: int = 3) -> None:
    say("## the design")
    say()
    say(f"model {plan.model}  dtype {plan.dtype}  BLOCK_SIZE_M {plan.block_m} FORCED")
    say(f"other tile constants held fixed from {FIXED_TILE_SOURCE}: {plan.fixed_tile}")
    say(f"GROUP_SIZE_M ladder {list(plan.group_m)}")
    say(f"{len(plan.cells)} routing realisations x {len(plan.group_m)} settings "
        f"x {plan.passes} pass(es) = {plan.n_measurements} timings")
    say(f"rough wall clock, from the byte model: "
        f"{estimated_seconds(plan, warmup, trials) / 60:.0f} min of kernel plus "
        f"compilation")
    say()
    say("| T | rows/expert | M-tiles | tiles/expert | x (min..max) | AI | rung |")
    say("|---:|---:|---:|---:|---|---:|---|")
    for tokens in plan.tokens:
        rung = [c for c in plan.cells if c.tokens == tokens]
        xs = [c.x for c in rung]
        tpe = [c.tiles_per_expert for c in rung]
        kind = ("CONTROL (one tile per expert)" if all(c.single_tile for c in rung)
                else "multi-tile" if all(not c.single_tile for c in rung)
                else "MIXED")
        say(f"| {tokens} | {rung[0].total_rows / max(rung[0].active_experts, 1):.0f} "
            f"| {min(c.m_tiles for c in rung):.0f}..{max(c.m_tiles for c in rung):.0f} "
            f"| {min(tpe):.2f}..{max(tpe):.2f} | {min(xs):.2f}..{max(xs):.2f} "
            f"| {rung[0].arith_intensity:.0f} | {kind} |")
    say()
    say("SATURATION, which is why the M-tile count is in the table above. The "
        "swizzle groups")
    say("GROUP_SIZE_M consecutive M-tiles, so once g reaches a rung's own "
        "num_pid_m the map")
    say("is plain column-major and every larger g is the SAME order:")
    for tokens in plan.tokens:
        rung = [c for c in plan.cells if c.tokens == tokens]
        pid_m = min(c.m_tiles for c in rung)
        saturated = [g for g in plan.group_m if g >= pid_m]
        if saturated:
            say(f"  T={tokens}: num_pid_m {pid_m:.0f}, so GROUP_SIZE_M "
                f"{saturated} are ONE setting there, not {len(saturated)}")
    say("  It bites the single-tile CONTROL rungs hardest, since they hold the "
        "fewest")
    say("  M-tiles. The control can still separate g=1 from the grouped orders, "
        "which is")
    say("  what P4 asks of it, but it cannot resolve the grouped ones from each "
        "other.")
    say("  THE SECOND ARM THAT FIXES IT, and it is worth running both because "
        "they fail in")
    say("  opposite directions:")
    say("    --model qwen2-57b-a14b --tokens 32,64,128,256,512,768,1024")
    say("  E=64 and k=8 put num_pid_m near 550 at the top rung, so nothing "
        "saturates, and")
    say("  the tiles-per-expert ladder is the same 1 to 9. The cost is that one "
        "expert's")
    say("  up-GEMM weights are 37 MB against mixtral's 235 MB, so on qwen2 L2 "
        "may already")
    say("  absorb the re-read at GROUP_SIZE_M=1 and leave the swizzle nothing to "
        "save. A")
    say("  large effect on mixtral and a small one on qwen2 is the mechanism; a "
        "large one")
    say("  on both, with the control flat, is stronger still.")
    say()
    say(f"measured ridge band {ridge[0]}-{ridge[1]} FLOP/byte; every cell above "
        f"is below {MEMORY_BOUND_MARGIN:.0%} of the low end, so implied_traffic_ratio")
    say("is a traffic bound rather than a statement about padded arithmetic.")
    say()
    bound = confound_bound(plan)
    say("CONFOUND BOUND: activation re-reads per extra M-tile over weight "
        "re-reads per extra")
    say(f"M-tile is exactly BLOCK_M / BLOCK_N = {plan.block_m}/"
        f"{plan.fixed_tile['BLOCK_SIZE_N']} = {bound:.2f}, so at most "
        f"{bound / (1 + bound):.0%} of any alpha")
    say("fitted here is activation traffic. GROUP_SIZE_M moves that term the "
        "OPPOSITE way,")
    say("so a falling alpha is a NET traffic statement and not a pure weight one.")
    say()
    say("### preflight")
    say()
    for gate in gates:
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")


def report_power(say, AR, plan: Plan, draws: int, seed: int) -> None:
    say()
    say("## can this design see the effect at all")
    say()
    say("Planted alpha, this design's own x values, the same estimator, "
        f"{POWER_NOISE:.1%} timing noise.")
    say()
    full = power_band(AR, plan, 0.558, POWER_NOISE, draws, seed)
    say(f"  whole ladder ({len(plan.tokens)} token counts): "
        + (f"band {full[0]:.3f}..{full[1]:.3f}  width {full[1] - full[0]:.3f}"
           if full else "no band"))
    if len(plan.tokens) > 1:
        top = max(plan.tokens)
        single = [c for c in plan.multi if c.tokens == top]
        one_rung = Plan(**{**plan.__dict__, "cells": tuple(single),
                           "tokens": (top,)})
        got = power_band(AR, one_rung, 0.558, POWER_NOISE, draws, seed)
        say(f"  one token count (T={top}) alone:      "
            + (f"band {got[0]:.3f}..{got[1]:.3f}  width {got[1] - got[0]:.3f}"
               if got else "no band"))
        say()
        say("  That gap is why the batch is a LADDER here. One token count is one")
        say("  intercept, the level is absorbed exactly, and only curvature across")
        say("  x is left to identify alpha.")
    say()
    if full:
        width = full[1] - full[0]
        ok = width <= PUBLISHED_GROUP_M_EFFECT
        say(f"  [{'PASS' if ok else 'FAIL'}] the band ({width:.3f}) is at most the "
            f"published GROUP_SIZE_M effect")
        say(f"          ({PUBLISHED_GROUP_M_EFFECT:.3f}, 0.570 at g=1 against 0.488 "
            f"at g=16). A wider band can still")
        say("          resolve a LARGER effect, which is what a forced sweep expects "
            "to find.")


def report_fits(say, fits: list[SettingFit]) -> None:
    say()
    say("## alpha per GROUP_SIZE_M")
    say()
    say("| GROUP_SIZE_M | n | discriminating | intercepts | alpha | 90% band | "
        "residual RMS | placebo |")
    say("|---:|---:|---:|---:|---:|---|---:|---:|")
    for fit in fits:
        interval = (f"{fit.interval[0]:.3f}..{fit.interval[1]:.3f}"
                    if fit.interval else "n/a")
        alpha = f"{fit.alpha:.3f}" if fit.alpha is not None else "n/a"
        placebo = f"{fit.placebo:.3f}" if fit.placebo is not None else "n/a"
        say(f"| {fit.group_m} | {fit.n} | {fit.n_disc} | {fit.n_groups} | {alpha} "
            f"| {interval} | {fit.rms:.4f} | {placebo} |")
    say()
    say("A setting is IDENTIFIED when it has at least "
        f"{MIN_DISCRIMINATING} discriminating rows,")
    say(f"{MIN_INTERCEPT_GROUPS} intercepts and a band no wider than "
        f"{MAX_BAND_WIDTH}. Residual RMS is model")
    say("adequacy in log units, not sampling error: if it is far above the "
        "timing")
    say("repeatability the bands are optimistic and only the ORDER of the "
        "alphas survives.")


def report_time(say, multi: dict[int, float], control: dict[int, float]) -> None:
    say()
    say("## the paired time table, which is what the control gate reads")
    say()
    say("| GROUP_SIZE_M | multi-tile ms / ms(ref) | single-tile ms / ms(ref) |")
    say("|---:|---:|---:|")
    for group_m in sorted(set(multi) | set(control)):
        say(f"| {group_m} | {multi.get(group_m, float('nan')):.4f} "
            f"| {control.get(group_m, float('nan')):.4f} |")
    say()
    say("Paired inside each cell on identical tensors, so the cell's own level "
        "cancels.")


# --------------------------------------------------------------------------
# gates over the result
# --------------------------------------------------------------------------

def result_gates(fits: list[SettingFit], multi: dict[int, float],
                 control: dict[int, float], knee: float,
                 knee_range: tuple[float, float] = (0.0, 0.0)) -> list[Gate]:
    """P1 to P5, each as a number against a threshold.

    Every gate that cannot be answered returns `ok=None` rather than False. A
    design that failed to identify alpha has not refuted the prediction, and
    reporting it as a refutation would be the most expensive kind of wrong
    answer available here.
    """
    gates: list[Gate] = []
    usable = [f for f in fits if f.identified]
    gates.append(Gate(
        "identification: every setting fitted a usable alpha",
        len(usable) == len(fits) and bool(fits),
        f"{len(usable)} of {len(fits)} settings identified"
        + ("" if len(usable) == len(fits) else
           "; the rest had too few discriminating rows, too few intercepts, "
           "or too wide a band")))

    if len(usable) < 2:
        gates.append(Gate("P1 alpha falls from the smallest to the largest "
                          "GROUP_SIZE_M", None,
                          "fewer than two identified settings"))
        gates.append(Gate("P2 no adjacent inversion", None, "same"))
        gates.append(Gate("P3 a floor above the knee", None, "same"))
    else:
        lo, hi = usable[0], usable[-1]
        disjoint = lo.interval[0] > hi.interval[1]
        gates.append(Gate(
            f"P1 alpha falls from GROUP_SIZE_M={lo.group_m} to {hi.group_m}",
            lo.alpha > hi.alpha and disjoint,
            f"alpha {lo.alpha:.3f} [{lo.interval[0]:.3f},{lo.interval[1]:.3f}] "
            f"-> {hi.alpha:.3f} [{hi.interval[0]:.3f},{hi.interval[1]:.3f}]; "
            f"bands {'disjoint' if disjoint else 'OVERLAP'}, "
            f"drop {lo.alpha - hi.alpha:+.3f}"))

        inversions = []
        for first, second in zip(usable, usable[1:], strict=False):
            # The band on a DIFFERENCE of two independent estimates, not either
            # band on its own: comparing a gap against one setting's band calls
            # ordinary sampling scatter an inversion.
            slack = math.hypot(first.width, second.width)
            if second.alpha > first.alpha + slack:
                inversions.append(f"{first.group_m}->{second.group_m} "
                                  f"({first.alpha:.3f}->{second.alpha:.3f})")
        gates.append(Gate(
            "P2 no adjacent inversion beyond the bands", not inversions,
            "none" if not inversions else "; ".join(inversions)))

        above = [f for f in usable if f.group_m >= knee]
        if len(above) < 2:
            gates.append(Gate("P3 a floor above the knee", None,
                              f"knee g* = {knee:.1f}; fewer than two settings "
                              "above it are identified"))
        else:
            spread = max(f.alpha for f in above) - min(f.alpha for f in above)
            slack = FLOOR_SLACK_BANDS * statistics.fmean([f.width for f in above])
            gates.append(Gate(
                "P3 alpha floors once the group covers one expert's tiles",
                spread <= slack,
                f"knee g* = {knee:.1f} tiles per expert (MEDIAN over the ladder, "
                f"which spans {knee_range[0]:.1f} to {knee_range[1]:.1f}, so this "
                f"is a statement about the ladder as a whole and not a sharp "
                f"threshold); settings {[f.group_m for f in above]} spread "
                f"{spread:.3f} against {FLOOR_SLACK_BANDS:g} x the mean band "
                f"width, {slack:.3f}"))

    multi_effect, control_effect = effect_size(multi), effect_size(control)
    if not multi or not control:
        gates.append(Gate("P4 the effect is specific to the multi-tile regime",
                          None, "one of the two rungs has no paired timings"))
    elif multi_effect < MIN_ATTRIBUTABLE_EFFECT:
        gates.append(Gate(
            "P4 the effect is specific to the multi-tile regime", None,
            f"the multi-tile time effect is {multi_effect:.1%}, below the "
            f"{MIN_ATTRIBUTABLE_EFFECT:.0%} floor, so there is no effect for "
            f"the control ({control_effect:.1%}) to be specific about"))
    else:
        ok = control_effect <= CONTROL_SPECIFICITY * multi_effect
        gates.append(Gate(
            "P4 the effect is specific to the multi-tile regime", ok,
            f"multi-tile time effect {multi_effect:.1%}, single-tile control "
            f"{control_effect:.1%}; the control must be at most "
            f"{CONTROL_SPECIFICITY:.0%} of it"
            + ("" if ok else ". A GROUP_SIZE_M effect that survives where no "
                             "expert spans two tiles is launch order, wave "
                             "quantisation or occupancy, NOT a weight re-read")))

    placebos = [f.placebo for f in fits if f.placebo is not None]
    if not placebos:
        gates.append(Gate("P5 the placebo collapses", None, "nothing to permute"))
    else:
        worst = max(abs(p) for p in placebos)
        gates.append(Gate(
            "P5 permuting the response inside each group collapses alpha",
            worst <= PLACEBO_MAX_ALPHA,
            f"largest permuted alpha {worst:.3f} against a limit of "
            f"{PLACEBO_MAX_ALPHA:.2f}"))
    return gates


def swizzle_integrity_gate(records) -> Gate:
    """Did reordering the program ids change the answer?

    A forced tile is a claim that the kernel still computes the same thing, and
    nothing in a timing table would show that it does not: an illegal-for-the-
    shape config or an in-place `fused_experts` both produce numbers, and the
    faster ones would read as a win. Rows that predate the check answer None
    rather than True, because "not measured" and "measured equal" are the two
    states this gate exists to keep apart.
    """
    checked = [r for r in records if r.get("matches_reference") is not None]
    if not checked:
        return Gate("correctness: the swizzle did not change the result", None,
                    "no record carries the cross-setting comparison")
    bad = [r for r in checked if not r["matches_reference"]]
    mutated = [r for r in checked if r.get("input_unchanged") is False]
    detail = (f"{len(checked)} settings compared against their cell's reference "
              f"setting; {len(bad)} differed")
    if bad:
        detail += (f". Worst |diff| {max(r['max_abs_diff'] for r in bad):.3e} at "
                   f"{bad[0]['id']}")
    if mutated:
        detail += (f". {len(mutated)} cells had their input mutated in place, "
                   "so every later setting timed a different problem")
    return Gate("correctness: the swizzle did not change the result",
                not bad and not mutated, detail)


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
        say(f"VERDICT: PREDICTION REFUTED. {len(failed)} gate(s) failed: "
            + "; ".join(g.name for g in failed))
        return 1
    if untested:
        say(f"VERDICT: NOT TESTABLE. {len(untested)} gate(s) had no evidence: "
            + "; ".join(g.name for g in untested))
        return 4
    say("VERDICT: PREDICTION HELD. alpha is not a scalar: it falls with the "
        "swizzle width,")
    say("floors where the group covers one expert's M-tiles, and the effect "
        "does not")
    say("survive in the single-tile control.")
    return 0


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def results_root() -> Path:
    """Where output goes so that it survives the pod being terminated.

    Same rule as `scripts/run_all.sh`: `$MOE_RESULTS_DIR`, else the network
    volume at `/workspace/results` when there is one, else the repo's own
    `results/`. The pod's container disk dies with the pod and the volume does,
    so this is not a cosmetic preference.
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


def bandwidth_for(gpu_name: str) -> tuple[float, str]:
    """The card's measured read ceiling, or the nominal one, and which it was.

    Stated because the RATIO's level is read by a human. The fitted alpha is
    invariant to it: a constant factor on every ratio is an additive constant in
    logs and the group intercept absorbs it exactly.
    """
    with contextlib.suppress(Exception):
        from moe.bench.roofline import load_measured
        hardware = load_measured(gpu_name or None)
        if hardware is not None:
            return hardware.bandwidth_bytes_s, f"measured ({hardware.name})"
    return NOMINAL_BANDWIDTH_BYTES_S, "nominal, no calibration on disk"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--tokens", default=",".join(str(t) for t in DEFAULT_TOKENS))
    parser.add_argument("--group-m", default=",".join(str(g) for g in GROUP_M_LADDER))
    parser.add_argument("--routings", default=",".join(DEFAULT_ROUTINGS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_ROUTING_SEEDS,
                        help="routing realisations per (token count, routing)")
    parser.add_argument("--passes", type=int, default=1,
                        help="repeat the whole grid, for a drift check")
    parser.add_argument("--run", action="store_true",
                        help="measure on the GPU; without it this plans only")
    parser.add_argument("--replay", type=Path, default=None,
                        help="re-report an existing output directory, no GPU")
    parser.add_argument("--synthetic", choices=SYNTHETIC_LAWS, default=None,
                        help="generate measurements from a stated law and run "
                             "the gates on them, so the gates are testable "
                             "without a GPU")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any cells already on disk and start over")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                        help="time with L2 warm; the default flushes, because a "
                             "swizzle is a cache claim")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.tokens = tuple(int(v) for v in str(args.tokens).split(",") if v)
    args.group_m = tuple(sorted(int(v) for v in str(args.group_m).split(",") if v))
    args.routings = tuple(v for v in str(args.routings).split(",") if v)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        AR = load_alpha_refit()
    except EstimatorMissing as exc:
        print(f"[group_m] {exc}", file=sys.stderr)
        return 3

    plan = build_plan(args)
    # A synthetic run gets its OWN directory, suffixed with the law. Nothing
    # generated from a stated law may ever land in the file a measured run
    # resumes from, and a suffix in the path is a stronger guarantee of that
    # than a field inside the records.
    suffix = f"-synthetic-{args.synthetic}" if args.synthetic else ""
    out_dir = args.replay or args.out or (
        results_root() / "group_m_alpha"
        / f"{plan.model}-bm{plan.block_m}-{plan.fingerprint}{suffix}")
    say = Report()
    say(f"# GROUP_SIZE_M sweep: is alpha a scalar?   ({git_head() or 'no git'})")
    say()
    say(f"output directory: {out_dir}")
    say("Everything below is written there as report.md, beside plan.json and "
        "cells.jsonl.")
    if args.synthetic:
        say()
        say(f"*** SYNTHETIC ({args.synthetic}). Nothing here was measured. These "
            "rows come from a")
        say("*** stated law and exist to show the gates can see an effect and "
            "can miss its absence.")
    say()

    ridge = AR.RIDGE_BAND
    gates = preflight(plan, ridge[0])
    report_plan(say, plan, gates, ridge, args.warmup, args.trials)
    if any(g.ok is False for g in gates):
        say()
        say("VERDICT: the design is refused before spending anything. Fix the "
            "failed preflight gate above.")
        _save(out_dir, say)
        return 1

    records: list[dict] = []
    meta: dict = {"gpu": "", "override_hook": ""}
    if args.replay:
        records = read_records(out_dir / "cells.jsonl")
        say()
        say(f"## replay: {len(records)} cells read from disk, nothing measured")
    elif args.synthetic:
        records = synthesise(plan, args.synthetic, args.seed)
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
        say()
        say(f"## measuring: {plan.n_measurements - len(done)} timings to do, "
            f"{len(done)} already on disk")
        (out_dir / "plan.json").write_text(json.dumps(
            {"fingerprint": plan.fingerprint, "argv": sys.argv[1:],
             "git": git_head(), "model": plan.model, "block_m": plan.block_m,
             "tokens": list(plan.tokens), "group_m": list(plan.group_m),
             "routings": list(plan.routings), "seeds": plan.seeds,
             "passes": plan.passes, "fixed_tile": plan.fixed_tile}, indent=2))
        try:
            fresh, meta = measure(plan, args, out_dir, done)
        except CannotRunHere as exc:
            say()
            say(f"CANNOT RUN HERE: {exc}")
            say("The plan and the preflight above are still valid and cost "
                "nothing; re-run with")
            say("--run on the pod, or --synthetic to exercise the gates.")
            _save(out_dir, say)
            return 3
        except KeyboardInterrupt:
            say()
            say("## aborted; reporting on what reached disk")
            fresh = []
        records = existing + fresh
    else:
        report_power(say, AR, plan, args.bootstrap, args.seed)
        say()
        say("Nothing was measured. Add --run on the pod, --synthetic to "
            "exercise the gates,")
        say("or --replay <dir> to re-report a finished run.")
        _save(out_dir, say)
        return 0

    return _analyse(say, AR, plan, records, meta, args, out_dir)


def _analyse(say, AR, plan: Plan, records: list[dict], meta: dict, args,
             out_dir: Path) -> int:
    known = {c.key for c in plan.cells}
    timed = [r for r in records if r.get("ms_p50")]
    # A REPLAY OF SYNTHETIC ROWS MUST NOT READ AS A MEASUREMENT. `--replay` does
    # not carry `--synthetic`, so without this the banner printed at the top of
    # a synthetic run disappears on the way back in and the report looks like a
    # pod result. Provenance travels in the records for exactly this path.
    synthetic = bool(args.synthetic) or any(
        r.get("provenance") == "synthetic" for r in records)
    if synthetic and not args.synthetic:
        say()
        say("*** SYNTHETIC. These records were generated from a stated law "
            f"({records[0].get('law', 'unknown')}) and")
        say("*** nothing here was measured on any hardware.")
    stray = [r for r in records if r.get("cell") not in known]
    say()
    say("## the measurements")
    say()
    say(f"  {len(records)} records, {len(timed)} timed, "
        f"{len(records) - len(timed)} failed or unmeasured")
    if stray:
        say(f"  {len(stray)} records name a cell this plan does not contain and "
            "are IGNORED. That")
        say("  means the flags differ from the ones that produced the file; "
            "re-run --replay with")
        say(f"  the argv recorded in plan.json. First stray: {stray[0]['cell']}")
    if not timed:
        say()
        say("VERDICT: NOT TESTABLE. Nothing was timed.")
        _save(out_dir, say)
        return 4
    throttled = [r for r in timed if r.get("throttled")]
    if throttled:
        say(f"  {len(throttled)} cells drifted more than 5% in SM clock and are "
            "flagged; their")
        say("  time is not the kernel's and a paired comparison across settings "
            "is what")
        say("  protects the result from them.")
    forced = [r for r in timed if r.get("observed_config")]
    if forced:
        wrong = [r for r in forced
                 if r["observed_config"].get("GROUP_SIZE_M") != r["group_m"]
                 or r["observed_config"].get("BLOCK_SIZE_M") != r["block_m"]]
        say(f"  {len(forced)} cells recorded the config vLLM actually resolved; "
            f"{len(wrong)} disagreed")
        say("  with what was forced. A disagreement means the sweep swept "
            "nothing.")
        if wrong:
            say(f"  FIRST DISAGREEMENT: {wrong[0]['id']} -> "
                f"{wrong[0]['observed_config']}")
    elif not synthetic:
        say("  NO cell recorded vLLM's own config, so nothing here confirms the "
            "override took")
        say("  effect. Read every flat line below as unexplained rather than as "
            "evidence.")

    bandwidth, source = bandwidth_for(meta.get("gpu", ""))
    say(f"  bandwidth for implied_traffic_ratio: {bandwidth / 1e12:.2f} TB/s "
        f"({source}).")
    say("  The fitted alpha does not depend on it; the ratio's level does.")

    fits = fit_per_setting(AR, timed, plan,
                           meta.get("gpu") or ("synthetic" if synthetic else "unknown"),
                           bandwidth, args.l2_flush, args.bootstrap, args.seed)
    report_fits(say, fits)

    cells = {c.key: c for c in plan.cells}
    reference = min(plan.group_m)
    multi = paired_ratios(timed, cells, reference, want_single_tile=False)
    control = paired_ratios(timed, cells, reference, want_single_tile=True)
    report_time(say, multi, control)

    tpe = [c.tiles_per_expert for c in plan.multi] or [0.0]
    knee = statistics.median(tpe)
    gates = result_gates(fits, multi, control, knee, (min(tpe), max(tpe)))
    gates.insert(0, swizzle_integrity_gate(timed))
    code = verdict(say, gates)
    _save(out_dir, say)
    return code


def _save(out_dir: Path, say: Report) -> None:
    with contextlib.suppress(OSError):
        out_dir.mkdir(parents=True, exist_ok=True)
        say.save(out_dir / "report.md")
        print()
        print(f"[group_m] report written to {out_dir / 'report.md'}")
        print(f"[group_m] measurements at  {out_dir / 'cells.jsonl'}")


if __name__ == "__main__":
    raise SystemExit(main())
