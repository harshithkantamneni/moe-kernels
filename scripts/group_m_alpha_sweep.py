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
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import bytes_model as BM  # noqa: E402
from moe.bench import exit_codes, timing  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
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

#: A setting's 90% band must be no wider than this MULTIPLE of the effect the
#: experiment is for, and the effect is `PUBLISHED_GROUP_M_EFFECT` below.
#:
#: IT WAS AN ABSOLUTE 0.15 UNTIL 2026-09-02, against an effect of 0.082: a band
#: 1.8x the effect, so a setting could be called "identified" while its interval
#: contained both zero and twice the number being claimed (S37, Hoefler & Belli
#: rule 7). Expressed as a multiple now, so the two cannot drift apart when
#: either is revised, and set to 1.0: a band that does not fit inside the effect
#: has not resolved it. On the shipped design the fitted bands run around 0.03
#: to 0.05, so this tightens a threshold nothing was using rather than
#: retiring settings that were passing.
BAND_WIDTH_OVER_EFFECT = 1.0

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

#: Timing noise the power simulation plants, as a fraction of log time.
#: MEASURED, NOT ASSUMED, as of 2026-09-02: it is the MEDIAN of
#: `timing_spread_median` over the 26 published `*.report.json` files under
#: `results/published`, which is every arm in this repository that records one.
#: The corpus runs 0.0039 to 0.0182 with a median of 0.0077.
#:
#: IT WAS 0.005, sourced to `moe/bench/crossing.py`'s note that one cell
#: reproduces to about 0.2% and described as "the pessimistic end of that"
#: (S37). It sat BELOW the entire measured range, so the power line overstated
#: what the design can see. The median is the honest single number; the maximum
#: is `POWER_NOISE_MAX` and `report_power` prints the band at BOTH, because the
#: design's answer differs between them and one number would hide which.
POWER_NOISE = 0.0077

#: The worst per-cell spread in the same corpus. The mixtral H200 arms -- this
#: sweep's own model on its own card -- are the top of that range (0.0140 to
#: 0.0182), so this is not a tail case for this design; it is the case.
POWER_NOISE_MAX = 0.0182

#: The effect the design has to be able to see, from the published split
#: (0.570 at GROUP_SIZE_M=1 against 0.488 at 16).
#:
#: IT IS A POOLED, UNPAIRED NUMBER and is labelled one (idx 36): it is the
#: `alpha_refit` marginal split over an unbalanced pool, not a paired
#: comparison at matched levels. It sizes the band gate and the power line,
#: neither of which is a claim about the world; nothing is scored against it.
PUBLISHED_GROUP_M_EFFECT = 0.082


def max_band_width() -> float:
    """The absolute band a setting must fit inside to count as identified.

    A function rather than a constant so the two numbers it is built from stay
    visibly connected: a revision to either has to move this, and a threshold
    that silently stops tracking the effect it is about is how a band 1.8x the
    effect came to read as "resolved".
    """
    return BAND_WIDTH_OVER_EFFECT * PUBLISHED_GROUP_M_EFFECT


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
        # NOT `raise SystemExit(<str>)`, which sets `SystemExit.code` to the
        # STRING and leaves the interpreter to exit 1 -- CLAIM_FAIL, "measured;
        # a pre-registered claim was refuted" -- from a planner that had drawn
        # no routing and timed nothing. `CannotRunHere` is caught in `main` and
        # returns REFUSED.
        raise CannotRunHere(
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
    #: `--cell-budget-ms`: the target duration of ONE trial, from which
    #: `time_kernel` derives the iteration count. It is part of the DESIGN and
    #: not of the analysis, because it sets the measured milliseconds of every
    #: cell, so it belongs in the fingerprint beside the tokens and the ladder.
    #: NO DEFAULT: a default here is a Plan that can be built without saying how
    #: long a trial ran, and the whole point of this field is that the answer
    #: reaches the fingerprint and the directory name.
    cell_budget_ms: float

    @property
    def fingerprint(self) -> str:
        """Stable hash of everything that defines the experiment.

        The output directory is named after it, so re-running the same plan
        resumes into the same file and a plan that differs anywhere gets a fresh
        one. Mixing two designs in one cells.jsonl would be undetectable
        afterwards, which is why this is not just a timestamp.

        `cell_budget_ms` JOINED IT 2026-09-02. It had been in neither the
        fingerprint nor the run id, which is collision 3 of
        `moe.bench.provenance`'s docstring verbatim: a `--cell-budget-ms 800`
        re-run after a 400 run derived the same directory, `measure` skipped
        every cell by `r["id"]`, and the report printed the 400 ms timings under
        the 800 ms heading. Every sibling sweep already had it.
        """
        payload = json.dumps({
            "model": self.model, "dtype": self.dtype, "block_m": self.block_m,
            "tokens": list(self.tokens), "group_m": list(self.group_m),
            "routings": list(self.routings), "seeds": self.seeds,
            "passes": self.passes, "fixed": self.fixed_tile,
            "cell_budget_ms": self.cell_budget_ms,
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
                passes=args.passes, cells=cells, fixed_tile=dict(FIXED_TILE),
                cell_budget_ms=float(args.cell_budget_ms))


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

    @property
    def token(self) -> str:
        """The gate's name as ONE whitespace-free token, for `RESULT:`.

        THE WHOLE NAME IS SLUGGED, not just its first word, and the reason is in
        this script's own gate list: `regime: every cell is memory bound` and
        `regime: the multi-tile rung has re-reads to save` are two different
        gates whose first word is the same. A driver keying on `regime` would
        see one of them and silently lose the other, which is the shape of
        failure `exit_codes` exists to stop rather than to reproduce.
        """
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.name).strip("-")
        return slug[:56] or "gate"

    @property
    def kind(self) -> str:
        """VALIDITY for the preflight gates, CLAIM for the result gates.

        The preflight gates are about whether the DESIGN can answer the
        question and are scored before a kernel runs; the result gates are the
        pre-registered predictions. `exit_codes` needs the distinction because a
        failed prediction is a RESULT and a design that cannot answer is not.
        The preflight names are lower-case words and the result names all start
        with P and a digit, which is the discriminator.
        """
        first = self.name.split()[0].rstrip(":")
        return (exit_codes.CLAIM
                if first[:1] == "P" and first[1:].isdigit()
                else exit_codes.VALIDITY)

    @property
    def verdict(self) -> str:
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.ok]

    def result_line(self) -> str:
        """The ONE line the session driver may grep for this gate.

        Rendered by `moe.bench.exit_codes.result_line`, so the prefix, the
        field order and the refusals are identical in every script. The
        `[PASS] name` block beside it is prose: a line that merely contains
        "PASS" is not a result, which is what the driver's old free-text grep
        was reading pre-registered expectations out of.
        """
        return exit_codes.result_line(self.kind, self.token, self.verdict,
                                      self.detail.replace("\n", " ")[:160])

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.token, self.verdict)


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
                and (self.interval[1] - self.interval[0]) <= max_band_width())

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
    """No GPU, no vLLM, or no override hook. Named so `main` can exit REFUSED.

    IT USED TO SAY "so `main` can exit 3", and `main` did. 3 is INVALID in
    `moe.bench.exit_codes`: "measured; a VALIDITY gate failed after measuring;
    nothing quotable", which tells the driver there is a directory of cells that
    must not be scored and that the arm must NOT be retried. Every raise of this
    class happens before a single cell is timed. It is REFUSED (2): free,
    nothing measured, and retryable on a box that has the thing that was
    missing.
    """


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
    from moe.bench.timing import ClockState, clock_drift, time_kernel
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
    # RESOLVED ONCE PER RUN, not per cell: the answer is a property of this box
    # and the calibration on it, and a per-cell lookup would read a yaml off
    # disk for every cell of a metered sweep.
    ref = reference_clock_for(gpu)
    meta = {"gpu": gpu, "override_hook": f"{where}.override_config",
            "reference_clock_mhz": ref.mhz,
            "reference_clock_source": ref.source}
    print(f"[group_m] override hook {meta['override_hook']}  device {gpu}")
    print("[group_m] LEVEL reference: "
          + (f"{ref.mhz:.0f} MHz, {ref.source}" if ref.mhz else
             f"NOT RESOLVED ({ref.source}). clock_level_ok is None on every "
             "cell this run writes, so no row can be excluded on the clock it "
             "ran at and the alpha below carries no clock evidence"))

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
                        # ONE INSTRUMENT (A7). This was `time_eager`, which
                        # is queue-deep and flushes but takes a warmup as a
                        # CALL COUNT and reads no clock under load: cells timed
                        # here at `--warmup 10` were compared with a roof warmed
                        # for hundreds of milliseconds, and the only clock
                        # evidence was the two idle-instant samples `clock_drift`
                        # takes, which the audit showed detect whether the START
                        # sample caught the idle boost rather than throttling.
                        measured = time_kernel(
                            call, warmup_ms=args.warmup,
                            target_ms=args.cell_budget_ms,
                            trials=args.trials, l2_flush=args.l2_flush,
                            reference_clock_mhz=ref.mhz)
                except timing.TimingRefused:
                    # THE SECOND DOOR INTO THE SAME ROOM, and it was open.
                    # `TimingRefused` subclasses RuntimeError, so the handler
                    # below caught every refusal the INSTRUMENT ITSELF raises --
                    # no CUDA and no injected fakes, trials=0, a warmup that
                    # makes the measurement meaningless. Each of those is a fact
                    # about the RUN and identical for every cell, so this sweep
                    # wrote one `ms_p50=nan` record per cell, reached "nothing
                    # was timed" with a page of them on disk, and reported a
                    # refusal whose stated reason was the wrong one: the records
                    # blamed the last cell's exception, not the instrument that
                    # refused all of them. Re-raised to `main`, which exits
                    # REFUSED. `driver.run_cell` has the same clause for the
                    # same reason, and names the BASE class as this does, so a
                    # refusal added to the instrument later cannot reintroduce
                    # the bug by forgetting to add itself here.
                    raise
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
                    cell, group_m, pass_index, plan, measured.ms_p50,
                    ms_std=measured.ms_std,
                    jitter=(measured.ms_p90 / measured.ms_p50
                            if measured.ms_p50 else None),
                    samples=measured.samples, l2_flush=measured.l2_flush,
                    # The state the cell ran in, from `KernelTiming`. The two
                    # clock flags are tri-state and BOTH have to be read: the
                    # `clock_drift_pct`/`throttled` pair beside them is the old
                    # idle-instant measurement, kept so a resumed jsonl still
                    # parses and so the two can be compared on the next pod.
                    instrument=measured.instrument,
                    warmup_ms=measured.warmup_ms, iters=measured.iters,
                    trials=measured.trials,
                    sm_clock_load_mhz=measured.sm_clock_load_mhz,
                    clock_level_ok=measured.clock_level_ok,
                    # The number LEVEL was scored against, on the row it scored,
                    # so a replay a week later can tell a row that PASSED the
                    # flag from one that had nothing to be level against. The
                    # tri-state alone cannot: both read None.
                    reference_clock_mhz=ref.mhz,
                    clock_drift_ok=measured.clock_drift_ok,
                    host_bound=measured.host_bound,
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


def estimated_seconds(plan: Plan, warmup: float, trials: int,
                      target_ms: float = 200.0) -> float:
    """Rough wall clock for the whole sweep, from the byte model. NOT a promise.

    Printed because this runs on a metered box and "how long is this" should not
    require starting it. `time_kernel` sizes its iteration count so one trial is
    about `target_ms` of kernel, so the timed part is nearly constant per cell.

    THE WARMUP TERM IS NOW CONSTANT, and it is the units change that made it so:
    `warmup` is milliseconds of delivered load rather than a call count, so it
    costs the same wall clock at every cell instead of scaling with the cell's
    own time. The old term (`warmup * ms`) priced a call count and is what a
    reader comparing this estimate against an old ARMS.tsv will find changed.
    """
    per_setting = (trials * target_ms + warmup) / 1e3
    return len(plan.cells) * len(plan.group_m) * per_setting * plan.passes


def report_plan(say, plan: Plan, gates: list[Gate], ridge: tuple[float, float],
                warmup: float = 300.0, trials: int = 3) -> None:
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
    # NO `RESULT:` LINE HERE, and that is the whole of the 2026-09-02 fix to
    # this block. A bare `group_m_alpha_sweep.py` measures nothing and used to
    # print five `RESULT: VALIDITY ... PASS` lines and exit 0, so
    # `exit_codes.classify_text` recomputed DONE for a run that spent nothing --
    # the shape a REFUSED log must never have. The preflight gates print their
    # RESULT lines in `verdict`, which runs only on a page that has timings on
    # it; a run that stops before that prints none, `classify_text` raises
    # `NoGatesScored`, and the process returns REFUSED to agree with it.
    for gate in gates:
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")


def report_power(say, AR, plan: Plan, draws: int, seed: int) -> None:
    say()
    say("## can this design see the effect at all")
    say()
    say("Planted alpha, this design's own x values, the same estimator, and a "
        "timing noise")
    say(f"MEASURED rather than assumed: {POWER_NOISE:.2%} is the median "
        f"`timing_spread_median` over the 26")
    say(f"published reports and {POWER_NOISE_MAX:.2%} is the worst of them. "
        "This line used to assume")
    say("0.5%, which sits below the whole measured range.")
    say()
    say("MDE. Both ends are priced, because the answer differs between them "
        "and one number")
    say("would hide which:")
    for label, noise in (("median", POWER_NOISE), ("worst", POWER_NOISE_MAX)):
        got = power_band(AR, plan, 0.558, noise, draws, seed)
        if got is None:
            say(f"  at the {label:<6} spread {noise:.2%}: no band; the design "
                "does not identify alpha at all")
            continue
        width = got[1] - got[0]
        say(f"  at the {label:<6} spread {noise:.2%}: band width {width:.3f}"
            + (f" <= the {PUBLISHED_GROUP_M_EFFECT:.3f} effect, RESOLVED"
               if width <= PUBLISHED_GROUP_M_EFFECT
               else f" > the {PUBLISHED_GROUP_M_EFFECT:.3f} effect, "
                    "CANNOT RESOLVE IT"))
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
        say(f"  [{'PASS' if ok else 'FAIL'}] at the MEDIAN measured spread, the "
            f"band ({width:.3f}) is at most the")
        say("          published GROUP_SIZE_M effect")
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
        f"{max_band_width():.3f} -- {BAND_WIDTH_OVER_EFFECT:g}x the "
        f"{PUBLISHED_GROUP_M_EFFECT:.3f} effect it")
    say("has to resolve, where it used to be an absolute 0.15, i.e. 1.8x that "
        "effect. Residual RMS is model")
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
    """Print every gate's RESULT line and the human table, and return the code.

    THE CODE COMES FROM `exit_codes.classify` OVER THE SAME GATE OBJECTS THAT
    PRINTED THE LINES, so `classify_text` over this function's output recomputes
    the integer the process returns and a disagreement between the two is itself
    a defect. Until 2026-09-02 it did not: the rule here was "any failed gate ->
    1, any undecided gate -> 4", and the table's rule over the very same RESULT
    lines is "any failed or undecided VALIDITY gate -> 3 INVALID, else any
    failed or undecided CLAIM gate -> 1 CLAIM_FAIL". They parted on every
    VALIDITY failure, which is most of this script's gates: `regime`, `control`,
    `design`, `identification` and `correctness` are VALIDITY; only `P1` to `P5`
    are CLAIMs. A failed apparatus gate is not a finding about the world. It
    means nothing on the page may be quoted, which is INVALID, and reporting it
    as 1 told the ledger a pre-registered prediction had been refuted.

    IT WAS A LIVE PATH AND NOT AN UNTESTED ONE. `result_gates` builds
    `identification: every setting fitted a usable alpha` from the fits, so a
    real pod run whose fits fail reached this branch, returned 1, and printed
    RESULT lines that classify to 3.

    THE THREE `VERDICT:` LINES STAY, and they are prose. They are what a human
    reads, they are not what the driver greps, and the exit code no longer comes
    from the branch that prints them: the same three sentences are chosen the
    same way, and `classify` answers separately over the gates. One `EXIT:` line
    beside them names the code in words, so the transcript says which of the
    five states this run ended in without anyone counting brackets.

    AN EMPTY GATE LIST RAISES `NoGatesScored` rather than returning DONE, which
    is `classify`'s own rule and the right one here: `_analyse` only reaches
    this call once records exist, so no gates at all means the gate builders
    silently produced nothing, and "a check that examined nothing reports zero
    failures" is this project's documented failure shape.
    """
    say()
    say("## gates")
    say()
    for gate in gates:
        say(gate.result_line())
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")
    say()
    failed = [g for g in gates if g.ok is False]
    untested = [g for g in gates if g.ok is None]
    code = exit_codes.classify(g.scored() for g in gates)
    if failed:
        say(f"VERDICT: PREDICTION REFUTED. {len(failed)} gate(s) failed: "
            + "; ".join(g.name for g in failed))
    elif untested:
        say(f"VERDICT: NOT TESTABLE. {len(untested)} gate(s) had no evidence: "
            + "; ".join(g.name for g in untested))
    else:
        say("VERDICT: PREDICTION HELD. alpha is not a scalar: it falls with "
            "the swizzle width,")
        say("floors where the group covers one expert's M-tiles, and the "
            "effect does not")
        say("survive in the single-tile control.")
    say(f"EXIT: {exit_codes.describe(code)}")
    return code


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


#: The card label a run that touches no GPU carries. `--synthetic` generates its
#: rows from a stated law and `--replay` re-reports a finished directory, so
#: neither has a card, and `provenance.run_id` refuses an id without one. A name
#: no `nvidia-smi` can produce, so it can never be read as a real card.
NO_CARD = "no-card-nothing-measured"


def resolve_card(args) -> str:
    """The card this run is about, or `NO_CARD` when there is not one.

    `--card`, else the live device, else `NO_CARD`. The measuring path
    (`--run`) needs a real one and `CannotRunHere` already refuses without a
    GPU, so `NO_CARD` reaches the id only on the two paths that measure nothing.
    """
    if getattr(args, "card", None):
        return str(args.card)
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            if name:
                return str(name)
    return NO_CARD


def default_run_id(args, card: str, plan: Plan) -> str:
    """The output directory's name: card first, then every knob that moves a row.

    THE FOUR OMISSIONS (A5/P5), all of them live before 2026-09-02:

      * THE CARD. It is not swept by this script, it is swept by the operator
        moving to another pod, and `results_root()` prefers `$MOE_RESULTS_DIR`
        then `/workspace/results`, a network volume the runbook uses BECAUSE it
        outlives the pod. Two cards therefore derived one directory, and the
        resume key is `r["id"]`, which carries no device: the second card would
        read the first's `cells.jsonl`, find every measurement present, spend no
        GPU time, and report the first card's timings under its own heading.
        The sibling sweep has the proof in the repo -- one report filename under
        both `2026-09-01-nvidia_h200-cross-card-s3` and
        `2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`.
      * `--warmup`, `--trials` and `--no-l2-flush`. Each sets the measured
        milliseconds of every row. The flush one is the sharpest here: this
        experiment is ABOUT a cache claim, so a warm-L2 run and a flushed run
        are different experiments, and they shared a directory.

    AND THE FIFTH, FOUND 2026-09-02: `--cell-budget-ms`. Its own help says it is
    the target duration of ONE trial and that `time_kernel` derives the
    iteration count from it, so it sets the measured milliseconds of every cell
    exactly the way `--iters` did in collision 3, and it was in neither this key
    nor `plan.fingerprint`. `tile_sweep`, `alias_ablation`, `block_m_crossing_sweep`
    and `tuned_vs_fallback` all carried it; this sweep, converted last, did not.
    It is in the fingerprint now AND named here as `7budget`, because a knob
    that reaches the id only through a hash cannot be read off `ls` and the
    operator comparing two directories is the reader who needs it most.

    `plan.fingerprint` still carries the design (model, dtype, block_m, tokens,
    group_m ladder, routings, seeds, passes, fixed tile, cell budget), so a
    change to any of those still lands elsewhere; it is passed through rather
    than re-listed, so the two cannot drift. `--bootstrap` and `--seed` stay
    OUT: they re-analyse a set of measurements rather than change one, and two
    analyses of one sweep belong in one directory.
    """
    return PV.run_id(card=card, **{
        "1model": plan.model, "2bm": plan.block_m,
        "3plan": plan.fingerprint, "4wm": args.warmup,
        "5tr": args.trials, "6flush": bool(args.l2_flush),
        "7budget": plan.cell_budget_ms})


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


def reference_clock_for(gpu_name: str):
    """The clock this card's roof was measured at, which LEVEL is scored against.

    THE FLAG HAD NO LEFT-HAND SIDE HERE. `timing.clock_flags` leaves
    `clock_level_ok` None unless it is handed the clock the roof was measured
    at, and this arm called `time_kernel` without one, so every row it wrote
    recorded the LEVEL column undetermined while carrying the column. The only
    clock evidence left in its records was the pair of IDLE-INSTANT samples
    `clock_drift` takes either side of a cell, which is the measurement
    `TIMING_BASIS` moved to v2 to RETIRE: it detects whether the first sample
    caught the idle boost, not whether the card throttled under the load. This
    arm and `alias_ablation` produce the two alphas `pod_session.sh` reconciles
    as the last thing on the screen, so a card that sagged during either one
    could not be seen from the numbers it is reconciled by.

    Resolved through `roofline.reference_clock` rather than as a fourth copy of
    the three-field rule. This tree already reads those fields in
    `block_m_crossing_sweep`, `dtype_tile_confound` and `memory_branch_anchor`,
    and copies of one rule are how the driver came to believe no calibration
    recorded a clock while the sweep read 1515 MHz out of the committed yaml;
    `roofline` is the copy the driver's `RunConfig` now uses, and the one that
    asks `load_hardware` whether the roof came from the file the clock is being
    read out of.

    NOT A REFUSAL, unlike the driver's, and that is a decision. The driver is
    the general path and refuses a card with no reference because a sweep of
    undetermined rows costs the same rental as one that can report a clock.
    Here, `pod_session.sh` runs `calibrate_hardware.py --publish` at step 1 and
    treats a refusal there as FATAL, so by step 3 either the reference exists or
    the session has already stopped; and every sibling arm resolves-and-says. An
    arm that refused alone would turn a missing yaml into a lost hour of card
    for a run whose correctness gates and paired ratios are unaffected by it. So
    the absence is printed in the words the operator can act on, carried into
    `meta`, and counted in the report.

    Returns the `ReferenceClock`: one `source` string that says where the number
    came from when there is one and why there is none when there is not, so a
    reason and a provenance cannot drift apart.
    """
    from moe.bench.roofline import reference_clock
    return reference_clock(gpu_name or None)


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
    parser.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                        dest="warmup", metavar="MS",
                        help="MILLISECONDS of delivered GPU load to warm up "
                             "for, not a call count. UNITS CHANGED 2026-09-02: "
                             "this sweep's cells span 16 to 448 tokens, so a "
                             "fixed count of 10 calls delivered two orders of "
                             "magnitude of different warmup across the grid "
                             "whose LEVELS the fit compares")
    parser.add_argument("--cell-budget-ms", type=float, default=200.0,
                        help="target duration of ONE trial; time_kernel derives "
                             "the iteration count from it")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                        help="time with L2 warm; the default flushes, because a "
                             "swizzle is a cache claim")
    parser.add_argument("--card", default=None,
                        help="the card this run measures, as nvidia-smi names "
                             "it. Defaults to the live device; --synthetic and "
                             "--replay touch no GPU and are labelled "
                             f"{NO_CARD!r}")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.tokens = tuple(int(v) for v in str(args.tokens).split(",") if v)
    args.group_m = tuple(sorted(int(v) for v in str(args.group_m).split(",") if v))
    args.routings = tuple(v for v in str(args.routings).split(",") if v)
    return args


def main(argv: list[str] | None = None) -> int:
    """`_main` with the escapes that were exiting ONE, which is CLAIM_FAIL.

    AN UNPLANNED CRASH IS ERROR, WHICH IS THE ONLY RETRYABLE CODE. Left to
    propagate, an unexpected exception exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it is in
    FINISHED_CODES, the session driver records it, and it is never retried. This
    arm fits one of the two alphas `pod_session.sh` reconciles as the last thing
    on the screen, and it is the longest of them; a torch OOM in its last cell
    would be filed as its registered answer and the session would never re-run
    it. ERROR (4) is outside FINISHED_CODES precisely so the driver can tell
    "the apparatus broke" from "the claim did not hold". The traceback is
    printed first and not swallowed, because a code without one tells an
    operator nothing about what to fix.

    A STRING `SystemExit` IS A REFUSAL. `raise SystemExit(<str>)` sets
    `SystemExit.code` to the STRING and leaves the interpreter to exit ONE as
    well. This file raises none today, on purpose -- see `build_cell`, which
    says so -- and the branch is here anyway so that one added later, or one
    raised by a library this imports, cannot land as a refuted claim.

    A `TimingRefused` IS A REFUSAL TOO, and it is the one the per-cell handler
    in `measure` now lets past it. It is the same fact for every cell, so
    nothing was measured and nothing was spent, which is REFUSED and not ERROR.
    `cli._main` catches the same base class around `driver.run_sweep` and exits
    the same code, so the two entry points agree.
    """
    try:
        return _main(argv)
    except timing.TimingRefused as exc:
        print(f"[group_m] REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = (exc.code if exc.code.startswith("REFUS")
                   else f"REFUSED: {exc.code}")
            print(f"[group_m] {msg}", file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        print("ERROR: group_m_alpha_sweep crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim failing, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: "
              "the traceback above is the thing to fix, the cells already on "
              "disk are under the run directory the plan printed, and the arm "
              "may be re-run.", file=sys.stderr)
        return exit_codes.ERROR


def _main(argv: list[str] | None = None) -> int:
    """The one place an exit code is chosen for a run that reached a verdict.

    Every return is a member of `moe.bench.exit_codes`'s table. This wrapper
    catches `CannotRunHere` raised out of PLANNING -- `build_cell` needs torch
    on the CPU to draw a routing realisation -- so a missing dependency exits
    REFUSED (2) rather than escaping as a traceback. The measuring path catches
    it separately, where it also has a report to save first.
    """
    try:
        return _run(argv)
    except CannotRunHere as exc:
        print(f"[group_m] REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED


def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        AR = load_alpha_refit()
    except EstimatorMissing as exc:
        # REFUSED (2), NOT INVALID (3). It returned 3 until 2026-09-02, and 3
        # means "measured, then a VALIDITY gate failed", which tells the driver
        # there is a directory of cells that must not be scored. There is no
        # directory: the estimator this sweep is scored against could not be
        # imported, which is a precondition, and nothing has run.
        print(f"[group_m] REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED

    plan = build_plan(args)
    # A synthetic run gets its OWN directory, suffixed with the law. Nothing
    # generated from a stated law may ever land in the file a measured run
    # resumes from, and a suffix in the path is a stronger guarantee of that
    # than a field inside the records.
    suffix = f"-synthetic-{args.synthetic}" if args.synthetic else ""
    card = resolve_card(args)
    out_dir = args.replay or args.out or (
        results_root() / "group_m_alpha"
        / f"{default_run_id(args, card, plan)}{suffix}")
    # PROVENANCE, built once and written into every artefact this run leaves.
    # None of the ten gaps-session scripts wrote a commit, a card or an
    # instrument, so not one of the 26 published reports can be attributed to a
    # code version (A5). `instrument` is the string `time_kernel` stamps on
    # every row it produces, so a reader can tell at a glance which apparatus
    # made a number.
    prov = PV.provenance_block(instrument=timing.TIMING_BASIS,
                               warmup_ms=args.warmup,
                               target_ms=args.cell_budget_ms)

    say = Report()
    say(f"# GROUP_SIZE_M sweep: is alpha a scalar?   ({git_head() or 'no git'})")
    say()
    say(f"card: {card}   instrument: {timing.TIMING_BASIS}")
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
        say("REFUSED: the design is refused before spending anything. Fix the "
            "failed preflight gate above.")
        _save(out_dir, say, prov)
        # REFUSED (2), NOT CLAIM_FAIL (1). It returned 1 until 2026-09-02, and 1
        # means "measured; VALIDITY passed; a pre-registered claim did not" --
        # a statement about the world, made by a run that had not started. The
        # preflight gates print no RESULT line here on purpose (they are scored
        # in `_analyse`, on a page that HAS timings), so the log carries none at
        # all and `exit_codes.classify_text` raises `NoGatesScored`, which is
        # the REFUSED shape. Nothing was spent and nothing was measured.
        return exit_codes.REFUSED

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
        (out_dir / "plan.json").write_text(json.dumps(prov.stamp(
            {"fingerprint": plan.fingerprint, "argv": sys.argv[1:],
             "git": git_head(), "card": card, "run_id": out_dir.name,
             "model": plan.model, "block_m": plan.block_m,
             "tokens": list(plan.tokens), "group_m": list(plan.group_m),
             "routings": list(plan.routings), "seeds": plan.seeds,
             "passes": plan.passes, "fixed_tile": plan.fixed_tile,
             "warmup_ms": args.warmup, "cell_budget_ms": args.cell_budget_ms,
             "trials": args.trials, "l2_flush": bool(args.l2_flush)}),
            indent=2))
        try:
            fresh, meta = measure(plan, args, out_dir, done)
        except CannotRunHere as exc:
            say()
            say(f"REFUSED. CANNOT RUN HERE: {exc}")
            say("The plan and the preflight above are still valid and cost "
                "nothing; re-run with")
            say("--run on the pod, or --synthetic to exercise the gates.")
            _save(out_dir, say, prov)
            # REFUSED (2), NOT INVALID (3). It returned 3 until 2026-09-02, and
            # 3 means "measured, then a VALIDITY gate failed: nothing quotable,
            # and do NOT retry it". Nothing was measured here and there is no
            # gate in the log to have failed; the missing thing is a GPU, which
            # is a precondition, and the arm is free to retry on a box that has
            # one. The driver treats the two oppositely, which is why they are
            # two codes.
            return exit_codes.REFUSED
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
        _save(out_dir, say, prov)
        # REFUSED, NOT DONE. This path prints a plan, a preflight, a power
        # analysis and an MDE, and times nothing; it used to return 0, so a
        # driver that asked for the arm and got a bare invocation logged it DONE
        # and never ran it. There are no RESULT lines above either, so
        # `exit_codes.classify_text` raises `NoGatesScored` on this log, which
        # is what a REFUSED log looks like from there: the two agree.
        return exit_codes.REFUSED

    return _analyse(say, AR, plan, records, meta, args, out_dir, prov, gates)


def _analyse(say, AR, plan: Plan, records: list[dict], meta: dict, args,
             out_dir: Path, prov=None, pre: list[Gate] | None = None) -> int:
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
        say("REFUSED. NOT TESTABLE: nothing was timed, so no gate below could "
            "have examined anything.")
        _save(out_dir, say, prov)
        # REFUSED (2), NOT ERROR (4). It returned 4 until 2026-09-02, and 4 is
        # "crashed; an exception the script did not plan for", which the ledger
        # reads as RETRY with a traceback to go and find. This path is planned
        # and there is no traceback: every record that reached disk carried no
        # `ms_p50`, so nothing was measured. No gate is scored below it either,
        # so the log carries no RESULT line and `exit_codes.classify_text`
        # raises `NoGatesScored`, which is the REFUSED shape.
        return exit_codes.REFUSED
    # THE TWO CLOCK VERDICTS, AND THEY ARE NOT THE SAME MEASUREMENT. LEVEL asks
    # whether the card sat at the clock the ROOF was measured at while the cell
    # ran, sampled under load; `throttled` below is the retired idle-instant
    # pair, kept so a resumed jsonl still parses and so the two can be compared
    # on the next pod. LEVEL is printed FIRST because it is the one the
    # instrument is at v2 for, and because "undetermined on every row" is a
    # fact about the apparatus that a reader has to have before the alpha.
    level_failed = [r for r in timed if r.get("clock_level_ok") is False]
    level_blind = [r for r in timed if r.get("clock_level_ok") is None]
    if not synthetic and timed:
        # THE APPARATUS BEFORE THE FINDING: whether the column could have said
        # anything is a different question from what it said, and a reader who
        # sees "0 flagged" without the first is reading silence as evidence.
        if level_blind:
            say(f"  {len(level_blind)} of {len(timed)} cells could not be "
                "examined on the clock at all")
            why = meta.get("reference_clock_source") or (
                "the rows carry no reference; they predate the LEVEL flag")
            say(f"  (clock_level_ok undetermined): {why}.")
        else:
            against = timed[0].get("reference_clock_mhz") or meta.get(
                "reference_clock_mhz")
            say("  every timed cell was scored against "
                + (f"{float(against):.0f} MHz" if against else "a reference")
                + ", so a clock problem could have been seen.")
        if level_failed:
            say(f"  {len(level_failed)} of them ran below the clock this card's "
                "roof was measured at and")
            say("  are flagged LEVEL failed; their time is the governor's, not "
                "the kernel's.")
    throttled = [r for r in timed if r.get("throttled")]
    if throttled:
        say(f"  {len(throttled)} cells drifted more than 5% in SM clock by the "
            "RETIRED idle-instant")
        say("  check and are flagged; their time is not the kernel's and a "
            "paired comparison")
        say("  across settings is what protects the result from them.")
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
    # THE PREFLIGHT GATES ARE SCORED HERE AND NOWHERE ELSE. They are decided
    # before a cell is timed, but their RESULT lines belong to a page that HAS
    # timings on it: printed at plan time they let a run that measured nothing
    # classify as DONE. `main` only reaches this call once `records` exist.
    code = verdict(say, list(pre or []) + gates)
    _save(out_dir, say, prov)
    return code


def _save(out_dir: Path, say: Report, prov=None) -> None:
    with contextlib.suppress(OSError):
        out_dir.mkdir(parents=True, exist_ok=True)
        say.save(out_dir / "report.md")
        if prov is not None:
            # Beside the markdown, not inside it: the report is prose a human
            # reads and the block is fields a script reads, and a run whose
            # report.md was hand-edited must still carry an unedited record of
            # the commit, the card and the instrument.
            (out_dir / "provenance.json").write_text(
                json.dumps(prov.stamp({"run_id": out_dir.name}), indent=2))
        print()
        print(f"[group_m] report written to {out_dir / 'report.md'}")
        print(f"[group_m] measurements at  {out_dir / 'cells.jsonl'}")


if __name__ == "__main__":
    raise SystemExit(main())
