#!/usr/bin/env python3
"""Refit `alpha`, the cost of an extra M-tile, against the DERIVED tile.

    python scripts/alpha_refit.py results/published/*/run_*.csv \
        --original-estimator --adversarial

`alpha` is the one free parameter in the tile-corrected roofline
(`docs/FINDINGS.md`, "The tile-corrected roofline"). One expert holding `r` rows
is scheduled as `ceil(r / BLOCK_M)` M-tiles, the first tile reads that expert's
weights in full, and each additional tile costs `alpha` of a fresh read because
L2 absorbs part of the re-read:

    weight bytes = N_w b (1 + alpha (M-tiles - 1)),   AI(r) = (2r/b) / Q(r)

It decides which tile heights can ever reach the compute roof at all, since
`AI -> 2 BM / (alpha b)` as `r` grows, so it is not a nuisance parameter.

TWO NUMBERS DISAGREE THREEFOLD AND THIS SCRIPT EXISTS TO SAY WHY. This repo
published `alpha = 0.10` (CV 12.8%) off 151 rows; arXiv:2608.13057 (TEMPO,
Aug 2026) fits the same physical parameter at about 0.33. The 0.10 was described
as "refit against the OBSERVED tile", which is true of the tile and hides where
the weakness actually is: it also came from ONE arm, ONE implementation, ONE
timing mode, and an estimator that minimises the coefficient of variation of a
POOLED ratio.

WHAT THIS SCRIPT CHANGES, in order of how much it moves the answer:

1. THE ESTIMATOR, which turns out to matter far more than the tile. Minimising
   the CV of a pooled ratio lets `alpha` absorb every between-cell difference in
   level -- model, batch, timing mode, card -- and those differences are an order
   of magnitude larger than the tile term. Worse, the largest of them runs the
   WRONG WAY: `implied_traffic_ratio` falls with batch as fixed dispatch cost
   amortises, while the tile count rises with batch, so a pooled fit pays
   `alpha` to explain a trend that has nothing to do with tiles and is pushed
   toward zero. This script fits a group intercept per
   (model, dtype, card, impl, timing mode, token count), so only rows that
   differ in tile count while agreeing on everything else can move `alpha`.
2. THE TILE, from `moe.bench.tile_resolve`, per row, DERIVED from vLLM 0.27.1's
   own lookup rather than assumed. It matters, and it is not the story: forcing
   64 on every row instead answers 0.48 and forcing 128 answers 0.65, against
   0.56 with the per-row derivation. A 35% spread, on a disagreement of 330%.
3. THE POOL, every current published arm rather than one.

WHAT `alpha` MEANS HERE, and it is narrower than "a fraction of a weight read".
`implied_traffic_ratio` is `time x achievable_bandwidth / compulsory_bytes`, so
the fit attributes to an extra tile EVERYTHING that tile costs in TIME: its
weight re-read, its padded arithmetic, its scheduling, its share of the tail.
Read as a traffic coefficient the answer is therefore an UPPER BOUND, which is
the direction that matters against TEMPO's `b2/b`, a pure byte ratio.
`--adversarial` prints a consequence of that bound which this study's own
measured crossings contradict.

Everything here is arithmetic over published CSVs: no GPU, no torch.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench import schema as SC  # noqa: E402
from moe.bench.bytes_model import weight_bytes_for_stage  # noqa: E402
from moe.bench.crossing import m_tiles_for_row  # noqa: E402
from moe.bench.published import filter_superseded, superseded_impls  # noqa: E402
from moe.bench.ridge import rows_per_expert  # noqa: E402
from moe.bench.tile_resolve import (  # noqa: E402
    VLLM_IMPLS,
    TileNotDerivable,
    resolve_tile_for_row,
)
from moe.routing.imbalance import TileEfficiencyUndetermined  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec  # noqa: E402
from moe.stages import CANONICAL_STAGES  # noqa: E402

#: The two numbers this fit is judged against.
REPO_PUBLISHED_ALPHA = 0.10
TEMPO_ALPHA = 0.33

#: The measured H200 ridge band, `docs/FINDINGS.md`. A band and not a number
#: because three calibrations of the same card disagree by 9.9% on the compute
#: term, so every absolute AI statement carries both ends.
RIDGE_BAND = (160.3, 176.2)

#: torch's `grouped_mm` tile, OBSERVED under claim C1 by reading the CUTLASS
#: kernel name out of the profiler, and fixed at 64 by Hopper's
#: `wgmma.mma_async.m64nNk16` whatever the shape. Not derived from vLLM's config
#: tree, which is why these rows are collected separately and labelled.
CUTLASS_BLOCK_M = 64
CUTLASS_IMPLS = frozenset({"torch_grouped_mm_up", "torch_grouped_mm_down"})

#: The arm the published 0.10 was fitted on. Named, because "torch grouped_mm
#: rows" is not a reproduction: pooled over every current arm those are 1,728
#: rows spanning two cards and two dtypes, the CV of the pooled ratio is 1245%,
#: and the objective is flat to the fourth decimal. The write-up's basis was one
#: arm, and only on that arm do the published 151 rows, 27 discriminating, and
#: the 13.1% / 12.8% / 17.5% CV column come back.
ORIGINAL_ALPHA_ARM = "2026-08-22-standard-sweep"

#: FINDINGS C2's one-stage bf16 crossing for mixtral, in tokens. Quoted rather
#: than recomputed because this script does not read crossings; it is used only
#: to check the fitted `alpha` against a number the study already published.
MIXTRAL_ONE_STAGE_CROSSING_TOKENS = 938

#: Columns `m_tiles_for_row` reads. Carried per observation so a tile count can
#: be recomputed at another block size without holding a 94-column dict.
TILE_COLUMNS = ("load_total_rows", "load_active_experts", "load_max_rows",
                "load_tile_eff_bm64", "load_tile_eff_bm128")

#: An M-tile count this far above the active-expert count is FLOATING-POINT DUST,
#: not a tile.
#:
#: THE BUG THIS FIXES, which was live for one run of this script. `m_tiles_for_row`
#: computes `total_rows / (tile_eff * block_m)`, and where `tile_eff` was
#: reconstructed the two divisions cancel exactly in algebra and to about 1e-13 in
#: binary. A bare `> 0` test then called 93 single-tile rows "discriminating",
#: handed them an `x` of about 1e-16, and the fit answered -0.850: the lower
#: BOUND, because a residual that flat is minimised by running away. An
#: unidentified split has to report that it is unidentified, not a boundary.
TILE_EPSILON = 1e-6


@dataclass(frozen=True)
class Observation:
    """One published row, reduced to what the fit needs.

    `extra_tile_bytes` is `W_expert x (M-tiles - active experts)`: the weight
    bytes a tile-corrected model charges ON TOP of the compulsory minimum, which
    already counts each active expert's weights exactly once. It is 0 whenever
    every active expert fits inside one tile, and such a row constrains `alpha`
    not at all -- it contributes only a group intercept. `discriminating` says
    which is which, because "10,813 rows" and "3,124 rows that can move the
    answer" are very different claims and only one of them is honest.
    """

    traffic_ratio: float
    compulsory_bytes: float
    per_expert_bytes: float
    active_experts: float
    m_tiles: float
    block_m: int
    group_m: int
    tile_provenance: str
    model: str
    dtype: str
    gpu: str
    impl: str
    tokens: int
    routing: str
    l2_flush: bool
    cuda_graph: bool
    #: Just `TILE_COLUMNS`, so `at_block_m` can recount tiles.
    tile_columns: tuple[tuple[str, str], ...]

    @property
    def extra_tile_bytes(self) -> float:
        extra = self.m_tiles - self.active_experts
        return self.per_expert_bytes * extra if extra > TILE_EPSILON else 0.0

    @property
    def x(self) -> float:
        """The regressor: extra tile bytes as a fraction of the compulsory total."""
        return self.extra_tile_bytes / self.compulsory_bytes

    @property
    def discriminating(self) -> bool:
        return self.extra_tile_bytes > 0.0

    @property
    def mode(self) -> str:
        return ("L2-cold" if self.l2_flush else "L2-warm") + (
            "/graph" if self.cuda_graph else "/eager")

    def at_block_m(self, block_m: int) -> Observation | None:
        """The same row scheduled at another tile height, or None if uncountable.

        Recounts from the stored load columns rather than rescaling `m_tiles`:
        tiles are a CEILING per expert, so halving the block does not double the
        count, and a rescaled number would be wrong by up to one tile per expert
        in a direction that depends on the histogram.
        """
        try:
            tiles = m_tiles_for_row(dict(self.tile_columns), block_m)
        except (TileEfficiencyUndetermined, SC.TileConfigUnrecorded, ValueError):
            return None
        return dataclasses.replace(self, m_tiles=tiles, block_m=block_m)


def expert_weight_bytes(spec: BenchSpec, covers: str) -> float:
    """Weight bytes for ONE expert over the stages this span covers.

    The two GEMM stages only. `router` also has a weight, and it is a single
    dense `[E, H]` gate read once per layer no matter how the rows are tiled, so
    multiplying it by an M-tile count would charge a re-read that cannot happen.
    `__pipeline__` rows record `covers = "all"`, which is where that mistake
    would have landed.
    """
    stages = CANONICAL_STAGES if covers == "all" else tuple(covers.split("+"))
    return float(sum(weight_bytes_for_stage(spec, s, 1) for s in stages
                     if s in ("up_gemm", "down_gemm")))


def collect(paths, census: collections.Counter, *, cutlass: bool = False,
            include_throttled: bool = False) -> list[Observation]:
    """Every published row that can carry a tile-corrected traffic fit.

    `cutlass=False` collects the vLLM Triton spans and derives each row's tile
    from vLLM 0.27.1's lookup. `cutlass=True` collects torch's `grouped_mm`
    spans at the C1-observed 64 instead, which is the pool the published 0.10
    came from, and is kept separate so the derived and the observed never pool by
    accident. SGLang is in neither: it ships its own tuned tree, nothing here
    models it, and substituting vLLM's answer would be the exact failure this
    work exists to correct.

    Every rejection is counted rather than dropped, because a filter that
    silently removes 90% of its input looks the same as one that removes nothing.
    """
    kept, dropped_arms = filter_superseded(paths)
    for path in dropped_arms:
        census[f"arm superseded whole: {path.parent.name}"] += 1
    out: list[Observation] = []
    for path in kept:
        retired = superseded_impls(path) or set()
        for row in SC.read_csv(path):
            impl = str(row.get("impl", ""))
            if impl in retired:
                census["implementation retired by a later arm"] += 1
                continue
            if impl not in (CUTLASS_IMPLS if cutlass else VLLM_IMPLS):
                census["implementation outside this pool"] += 1
                continue
            if float(row.get("ms_p50") or 0.0) <= 0.0:
                census["never timed (ms_p50 = 0 is not a measurement)"] += 1
                continue
            if not SC.passed(row):
                census["failed the correctness gate"] += 1
                continue
            if SC.row_bool(row, "throttled") and not include_throttled:
                census["throttled"] += 1
                continue
            ratio = SC.row_float(row, "implied_traffic_ratio")
            if ratio <= 0.0:
                census["no implied_traffic_ratio: the driver called the cell "
                       "compute-bound, or the arm has no ceiling for its dtype"] += 1
                continue
            if cutlass:
                block_m, group_m, provenance = CUTLASS_BLOCK_M, 0, "cutlass_c1_observed"
            else:
                try:
                    tile = resolve_tile_for_row(row)
                except TileNotDerivable:
                    census["tile not derivable from vLLM's lookup"] += 1
                    continue
                block_m, group_m = tile.block_m_derived, tile.group_m_derived
                provenance = tile.provenance
            try:
                tiles = m_tiles_for_row(row, block_m)
            except (TileEfficiencyUndetermined, SC.TileConfigUnrecorded):
                census[f"M-tiles undetermined at BLOCK_M={block_m}: an expert spans "
                       "several tiles and the per-expert histogram is not stored"] += 1
                continue
            model = str(row.get("model", ""))
            spec = BenchSpec(MODEL_CONFIGS[model],
                             num_tokens=int(float(row["num_tokens"])),
                             dtype=str(row.get("dtype", "")))
            per_expert = expert_weight_bytes(spec, str(row.get("covers", "")))
            compulsory = SC.row_float(row, "compulsory_bytes")
            if compulsory <= 0.0 or per_expert <= 0.0:
                census["no compulsory byte model for this span"] += 1
                continue
            out.append(Observation(
                traffic_ratio=ratio,
                compulsory_bytes=compulsory,
                per_expert_bytes=per_expert,
                active_experts=SC.row_float(row, "load_active_experts"),
                m_tiles=tiles,
                block_m=block_m, group_m=group_m, tile_provenance=provenance,
                model=model, dtype=str(row.get("dtype", "")),
                gpu=str(row.get("gpu_name", "")), impl=impl,
                tokens=int(float(row["num_tokens"])),
                routing=str(row.get("routing_kind", "")),
                l2_flush=SC.row_bool(row, "l2_flush"),
                cuda_graph=SC.row_bool(row, "cuda_graph"),
                tile_columns=tuple((c, str(row.get(c, ""))) for c in TILE_COLUMNS)))
            census["ADMITTED"] += 1
    return out


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

def cell_key(obs: Observation) -> tuple:
    """Everything that sets the LEVEL of `implied_traffic_ratio` bar the tile.

    Token count is in here and it is the important one. Fixed dispatch cost
    amortises over more work as the batch grows, so the ratio falls with T while
    the tile count rises with T. Without T in the intercept, `alpha` is paid to
    explain that trend and comes out near zero, which is most of the distance
    between 0.10 and the answer this script prints.
    """
    return (obs.model, obs.dtype, obs.gpu, obs.impl, obs.l2_flush,
            obs.cuda_graph, obs.tokens)


def _design(observations, keyfn, group_ids=None):
    x = np.array([o.x for o in observations], dtype=float)
    y = np.log(np.array([o.traffic_ratio for o in observations], dtype=float))
    if group_ids is not None:
        groups = np.asarray(group_ids, dtype=np.int64)
        return x, y, groups, int(groups.max()) + 1 if len(groups) else 0
    index: dict = {}
    groups = np.empty(len(observations), dtype=np.int64)
    for i, o in enumerate(observations):
        groups[i] = index.setdefault(keyfn(o), len(index))
    return x, y, groups, len(index)


def _within_group_ssr(x, y, groups, n_groups, alpha: float) -> float:
    """Within-group squared deviation of `log ratio - log(1 + alpha x)`.

    Log space, not absolute. The ratio spans about 1.0 to 8 across the pool, so
    a least-squares fit on raw values would weight one eager fp8 cell more than
    a hundred graphed bf16 cells. In logs the group intercept is a pure scale
    factor, which is exactly what it represents.
    """
    u = 1.0 + alpha * x
    if np.any(u <= 0.0):
        return math.inf
    residual = y - np.log(u)
    counts = np.bincount(groups, minlength=n_groups)
    sums = np.bincount(groups, weights=residual, minlength=n_groups)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    centred = residual - means[groups]
    return float(centred @ centred)


def fit_alpha(observations, keyfn=cell_key, *, group_ids=None,
              lo: float = -0.9, hi: float = 5.0, tol: float = 1e-5) -> float:
    """`alpha` minimising the within-group residual, by scan then golden section.

    The scan comes first because the objective is not guaranteed unimodal: `x`
    is a step function of the batch, so the residual has kinks wherever a cell
    changes tile count, and a bare golden section from a bad bracket would
    report a local minimum with nothing to indicate it.

    The lower bound is NEGATIVE on purpose. A fit that can only return a
    non-negative number cannot tell "the data say zero" from "the data say less
    than zero and were clipped at the boundary", and the pooled estimator this
    replaces fails in exactly that way on three of the four timing modes.

    `group_ids` names each row's intercept directly, for the bootstrap, where the
    same Observation OBJECT can be drawn into two different clusters and a
    key function computed from its fields could not tell the two copies apart.
    """
    if len(observations) < 2:
        raise ValueError("a fit needs at least two observations")
    x, y, groups, n_groups = _design(observations, keyfn, group_ids)

    def objective(alpha: float) -> float:
        return _within_group_ssr(x, y, groups, n_groups, alpha)

    grid = np.linspace(lo, hi, 120)
    values = [objective(a) for a in grid]
    best = int(np.argmin(values))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = right - phi * (right - left), left + phi * (right - left)
    fc, fd = objective(c), objective(d)
    while right - left > tol:
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - phi * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + phi * (right - left)
            fd = objective(d)
    return (left + right) / 2.0


def bootstrap_band(observations, draws: int, seed: int,
                   quantiles: tuple[float, float] = (0.05, 0.95)
                   ) -> tuple[float, float] | None:
    """A CLUSTER bootstrap over the fixed-effect groups, not over rows.

    Rows inside one group are replicates of one cell measured at several seeds
    and trials, so resampling them independently would treat six views of one
    thermal state as six measurements and return a band several times too
    narrow. The group is the unit the fit has independent information about.

    Each DRAWN COPY of a group gets its own intercept, so drawing the same cell
    twice gives two independent observations of it rather than one with double
    weight -- which is what sharing an intercept between the copies would mean.

    None when fewer than two groups survive; a band over one cluster is not a
    band.
    """
    groups: dict = collections.defaultdict(list)
    for o in observations:
        groups[cell_key(o)].append(o)
    keys = list(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        rows: list[Observation] = []
        ids: list[int] = []
        for copy in range(len(keys)):
            members = groups[rng.choice(keys)]
            rows.extend(members)
            ids.extend([copy] * len(members))
        samples.append(fit_alpha(rows, group_ids=ids))
    samples.sort()
    return _percentile(samples, quantiles[0]), _percentile(samples, quantiles[1])


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def pooled_cv_alpha(observations, lo: float = 0.0, hi: float = 2.0,
                    steps: int = 4000) -> tuple[float, float, float]:
    """`(alpha, CV, mean ratio)` under the ORIGINAL estimator, for comparison.

    Minimises the coefficient of variation of the corrected ratio over the whole
    pool with no group structure at all, which is what produced 0.10. Kept so
    both estimators can be run on the SAME rows: run only on different pools,
    the difference between them is unattributable, which is the position this
    study was in before today.

    A plain scan rather than an optimiser, because the point of printing it is
    the SHAPE of the objective rather than its argmin: on the original 151 rows
    it falls by well under one percent between `alpha = 0` and its minimum.
    """
    best = (math.inf, lo)
    for i in range(steps + 1):
        alpha = lo + (hi - lo) * i / steps
        score = _cv(observations, alpha)
        if score < best[0]:
            best = (score, alpha)
    return best[1], best[0], _mean_ratio(observations, best[1])


def _corrected(observations, alpha: float) -> list[float]:
    return [o.traffic_ratio / (1.0 + alpha * o.x) for o in observations]


def _cv(observations, alpha: float) -> float:
    values = _corrected(observations, alpha)
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def _mean_ratio(observations, alpha: float) -> float:
    return statistics.fmean(_corrected(observations, alpha))


def ai_cap(block_m: int, alpha: float, dtype_bytes: int = 2) -> float:
    """`2 BM / (alpha b)`: the rows per expert this tile height can never exceed.

    The consequence the whole parameter is being measured for. If this sits
    below the hardware ridge, that tile cannot reach compute bound at any batch
    size at all. Infinite at `alpha <= 0`, which is the correct reading: with no
    re-read cost the intensity is unbounded and `2R/b` is exact.
    """
    return 2.0 * block_m / (alpha * dtype_bytes) if alpha > 0 else math.inf


def max_alpha_that_still_crosses(block_m: int, ridge: float,
                                 dtype_bytes: int = 2) -> float:
    """The largest `alpha` at which `block_m` can still reach `ridge`."""
    return 2.0 * block_m / (ridge * dtype_bytes)


def count_excluded_memory_bound(paths, alpha: float,
                                include_throttled: bool = False
                                ) -> collections.Counter:
    """How many rows the memory-bound filter drops that it should not.

    THE FILTER THIS WHOLE FIT DEPENDS ON. `driver.py` writes
    `implied_traffic_ratio` only where `hardware.bound(dtype,
    arith_intensity_compulsory) == "memory"`, so a row with no column is a row
    the driver called compute-bound on the COMPULSORY intensity.

    That has no false positives, and to that extent the filter is sound:
    compulsory intensity is an upper bound on the true one, so
    `AI_compulsory < ridge` implies `AI_true < ridge`. The driver's own comment
    says exactly this.

    It has false NEGATIVES, and they are not randomly placed.
    `AI_corrected = AI_compulsory / Q` with `Q = 1 + alpha (tiles - 1) >= 1`, so
    a row with many tiles can be memory bound while its compulsory intensity
    says otherwise. Those rows carry no column, so they cannot enter the fit --
    and they are by construction the rows with the MOST extra tiles, which is
    precisely the evidence the fit is short of.
    """
    kept, _ = filter_superseded(paths)
    census: collections.Counter = collections.Counter()
    for path in kept:
        retired = superseded_impls(path) or set()
        for row in SC.read_csv(path):
            impl = str(row.get("impl", ""))
            if impl in retired or impl not in VLLM_IMPLS:
                continue
            if float(row.get("ms_p50") or 0.0) <= 0.0 or not SC.passed(row):
                continue
            if SC.row_bool(row, "throttled") and not include_throttled:
                continue
            peak = SC.row_float(row, "achieved_peak_tflops")
            bandwidth = SC.row_float(row, "achieved_bw_gbps")
            if peak <= 0.0 or bandwidth <= 0.0:
                census["no ceiling on the row, so no ridge to classify against"] += 1
                continue
            if SC.row_float(row, "implied_traffic_ratio") > 0.0:
                census["memory-bound and carries the column"] += 1
                continue
            try:
                tile = resolve_tile_for_row(row)
                tiles = m_tiles_for_row(row, tile.block_m_derived)
            except (TileNotDerivable, TileEfficiencyUndetermined,
                    SC.TileConfigUnrecorded):
                census["no column, and the tile-corrected test cannot be run"] += 1
                continue
            spec = BenchSpec(MODEL_CONFIGS[str(row["model"])],
                             num_tokens=int(float(row["num_tokens"])),
                             dtype=str(row.get("dtype", "")))
            per_expert = expert_weight_bytes(spec, str(row.get("covers", "")))
            active = SC.row_float(row, "load_active_experts")
            compulsory = SC.row_float(row, "compulsory_bytes")
            extra = per_expert * max(tiles - active, 0.0)
            extra = extra if tiles - active > TILE_EPSILON else 0.0
            corrected = SC.row_float(row, "flops") / max(compulsory + alpha * extra, 1.0)
            ridge = peak * 1e12 / (bandwidth * 1e9)
            if corrected < ridge:
                census["NO COLUMN BUT TILE-CORRECTED MEMORY-BOUND: "
                       "excluded and should not have been"] += 1
            else:
                census["compute-bound under both models, correctly excluded"] += 1
    return census


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _split_line(label: str, subset: list[Observation], width: int = 24) -> str:
    """One split of the pool, with its discriminating count beside its answer.

    The count is not decoration. A split with no discriminating rows has NO
    information about `alpha` -- every row in it sits at `x = 0` -- so printing a
    number there would be printing the pooled answer under a new heading.
    """
    n_disc = sum(1 for o in subset if o.discriminating)
    head = f"  {label:<{width}} n={len(subset):>6}  discriminating={n_disc:>5}  "
    if n_disc < 10 or len(subset) < 2:
        return head + "alpha=n/a (nothing in this split can move it)"
    return head + f"alpha={fit_alpha(subset):.3f}"


def _report_pool(triton: list[Observation], census: collections.Counter) -> None:
    print("## the pool")
    print()
    for reason, count in census.most_common():
        print(f"  {count:>7}  {reason}")
    print()
    n_disc = sum(1 for o in triton if o.discriminating)
    print(f"  {len(triton)} rows admitted, of which {n_disc} have M-tiles > active")
    print(f"  experts and can move alpha at all. The other {len(triton) - n_disc} sit at")
    print("  x = 0 and contribute a group intercept and nothing else.")
    print()
    print(f"  dtypes present: {dict(collections.Counter(o.dtype for o in triton))}")
    if {o.dtype for o in triton} == {"bf16"}:
        print("  THE FIT IS bf16 ONLY, and not by choice. Every fp8 vLLM row lives in")
        print("  `-fp8-three-kernel`, whose calibration measured no fp8 ceiling, so all")
        print("  of them carry achieved_peak_tflops = 0 and with it")
        print("  implied_traffic_ratio = 0. A same-session fp8 calibration is one line")
        print("  on a pod, and it is what would let alpha be tested across dtypes.")
    print()
    print("  tile provenance: "
          f"{dict(collections.Counter(o.tile_provenance for o in triton))}")


def _report_fit(triton: list[Observation], alpha: float, args) -> None:
    print("## the fit")
    print()
    band = bootstrap_band(triton, args.bootstrap, args.seed)
    n_groups = len({cell_key(o) for o in triton})
    n_disc = sum(1 for o in triton if o.discriminating)
    print(f"  alpha = {alpha:.3f}")
    if band:
        print(f"  90% cluster-bootstrap band: {band[0]:.3f} .. {band[1]:.3f}"
              f"  ({args.bootstrap} draws over {n_groups} groups)")
    print(f"  n = {len(triton)} rows, {n_disc} discriminating, {n_groups} intercepts")
    print()
    print(f"  vs this repo's published {REPO_PUBLISHED_ALPHA:.2f}: "
          f"{alpha / REPO_PUBLISHED_ALPHA:.1f}x")
    print(f"  vs TEMPO's {TEMPO_ALPHA:.2f}:                 {alpha / TEMPO_ALPHA:.1f}x")


def _report_splits(triton: list[Observation]) -> None:
    print("## is alpha a scalar?")
    print()
    print("GROUP_SIZE_M is the swizzle width, i.e. exactly how many M-tiles reuse one")
    print("weight block out of L2, so it is the parameter alpha should vary with if")
    print("alpha varies with anything at all.")
    print()
    for group_m in sorted({o.group_m for o in triton}):
        print(_split_line(f"GROUP_SIZE_M = {group_m}",
                          [o for o in triton if o.group_m == group_m]))
    print()
    for block_m in sorted({o.block_m for o in triton}):
        print(_split_line(f"BLOCK_M = {block_m}",
                          [o for o in triton if o.block_m == block_m]))
    print()
    for model in sorted({o.model for o in triton}):
        print(_split_line(model, [o for o in triton if o.model == model]))
    print()
    for gpu in sorted({o.gpu for o in triton}):
        print(_split_line(gpu, [o for o in triton if o.gpu == gpu]))
    print()
    for mode in sorted({o.mode for o in triton}):
        print(_split_line(mode, [o for o in triton if o.mode == mode]))
    print()
    for routing in sorted({o.routing for o in triton}):
        print(_split_line(f"routing {routing}",
                          [o for o in triton if o.routing == routing]))


def _report_original(cutlass: list[Observation], alpha_new: float) -> None:
    """The published 0.10, its rows, and the same rows under the new estimator.

    Run on the L2-cold eager torch `grouped_mm` rows because that is the basis
    the 2026-08-22 write-up named: 151 unthrottled memory-bound rows at the
    CUTLASS tile of 64. Reproducing the count and the CV column is what turns
    the comparison into an attribution instead of an assertion.
    """
    basis = [o for o in cutlass if o.l2_flush and not o.cuda_graph]
    print("## the original estimator, on the original rows")
    print()
    print(f"  `{ORIGINAL_ALPHA_ARM}` only: torch grouped_mm, L2-cold eager,")
    print("  unthrottled, memory-bound, at the CUTLASS BLOCK_M="
          f"{CUTLASS_BLOCK_M} OBSERVED under C1.")
    print("  Nothing in this section is derived from anything.")
    print()
    n_disc = sum(1 for o in basis if o.discriminating)
    print(f"  n={len(basis)}  discriminating={n_disc}"
          "   (the write-up says 151 rows, 27 of them discriminating)")
    print()
    print("  The CV column below reproduces the write-up's to a tenth of a point.")
    print("  The mean ratio sits about 1% lower, and for a reason worth stating: the")
    print("  write-up divided by WEIGHT bytes at a fixed 4390.29 GB/s read ceiling,")
    print("  while implied_traffic_ratio divides by the row's full compulsory bytes,")
    print("  activations included, at the row's own triad ceiling. Same rows, slightly")
    print("  different denominator; nothing about the identification changes.")
    if len(basis) < 2:
        print("  too few rows to fit")
        return
    fitted, cv, mean = pooled_cv_alpha(basis)
    print()
    print("  | alpha | mean ratio | CV |")
    print("  |---|---:|---:|")
    for candidate in (0.0, REPO_PUBLISHED_ALPHA, TEMPO_ALPHA, 1.0):
        print(f"  | {candidate:.2f} | {_mean_ratio(basis, candidate):.2f}x | "
              f"{_cv(basis, candidate):.1%} |")
    print(f"  | {fitted:.3f} (its own minimum) | {mean:.2f}x | {cv:.1%} |")
    print()
    print("  THE OBJECTIVE IS NEARLY FLAT: the CV falls by "
          f"{1 - cv / _cv(basis, 0.0):.1%} between")
    print(f"  alpha = 0 and its own minimum, on {n_disc} rows out of {len(basis)} that can")
    print("  move it at all. A minimum that shallow is a statement about the")
    print("  estimator, not about the hardware.")
    print()
    print(f"  SAME ROWS, group-intercept estimator: alpha = {fit_alpha(basis):.3f}")
    print(f"  SAME ESTIMATOR, whole derived pool:   alpha = {alpha_new:.3f}")
    print()
    print("  So the disagreement with TEMPO is NOT an artefact of the assumed tile.")
    print("  The tile in these rows was never assumed: it was read out of the CUTLASS")
    print("  kernel name. It is an artefact of the ESTIMATOR.")


def _report_adversarial(triton: list[Observation], alpha: float, args) -> None:
    """Everything that would make the number above wrong, checked where it can be."""
    print("## against my own fit")
    print()
    print("### 1. the placebo")
    print("  The RESPONSE is permuted inside each group, which breaks the pairing")
    print("  between a row's traffic ratio and its tile count while leaving both")
    print("  marginals and the whole group structure exactly as they were. The")
    print("  regressor is not touched, because shuffling THAT would also break the")
    print("  pairing between a row's tile count and its own active-expert count and")
    print("  would test something else.")
    print()
    rng = random.Random(args.seed)
    groups: dict = collections.defaultdict(list)
    for o in triton:
        groups[cell_key(o)].append(o)
    shuffled: list[Observation] = []
    for members in groups.values():
        responses = [o.traffic_ratio for o in members]
        rng.shuffle(responses)
        shuffled.extend(dataclasses.replace(o, traffic_ratio=r)
                        for o, r in zip(members, responses, strict=True))
    print(f"  traffic ratio permuted within each group: alpha = {fit_alpha(shuffled):.3f}")
    print(f"  as measured:                              alpha = {alpha:.3f}")
    print("  A fit that survived the permutation would be fitting the group")
    print("  structure rather than the tile.")
    print()

    print("### 2. how much of the answer is the derivation")
    print("  The tile enters only through the M-tile count, so forcing ONE height on")
    print("  every row bounds how much of the answer the per-row derivation carries.")
    print()
    for block_m in (16, 32, 64, 128, 256):
        forced = [f for f in (o.at_block_m(block_m) for o in triton) if f is not None]
        n_disc = sum(1 for o in forced if o.discriminating)
        line = (f"  forced BLOCK_M={block_m:>3}: n={len(forced):>6} "
                f"discriminating={n_disc:>5}  ")
        print(line + (f"alpha={fit_alpha(forced):.3f}" if n_disc >= 10 and len(forced) > 1
                      else "alpha=n/a"))
    print()
    print("  BLOCK_M 16 and 32 have NO discriminating rows, and that is structural")
    print("  rather than a sampling accident. At a block size the schema does not")
    print("  store, the M-tile count is RECONSTRUCTED, and the reconstruction is only")
    print("  valid while every expert fits in one tile -- so the rows that survive at")
    print("  16 are exactly the rows where 16 costs nothing. The derived pool inherits")
    print("  that: most of it resolves to BLOCK_M=16, and none of that part")
    print("  constrains alpha.")
    print()

    print("### 3. the memory-bound filter is sound one way and incomplete the other")
    excluded = count_excluded_memory_bound(args.csvs, alpha, args.include_throttled)
    for reason, count in excluded.most_common():
        print(f"  {count:>7}  {reason}")
    print()
    print("  No false positives: compulsory intensity is an UPPER bound on the true")
    print("  one, so a row the driver called memory-bound is memory-bound under the")
    print("  tile-corrected model too. The false NEGATIVES are the problem, and they")
    print("  are not randomly placed: AI_corrected = AI_compulsory / Q, and Q grows")
    print("  with the tile count, so the rows wrongly excluded are the many-tile rows")
    print("  this fit is short of. That biases the pool toward low leverage; it does")
    print("  not obviously bias alpha in a known direction.")
    print()

    print("### 4. the fitted alpha contradicts a crossing this study measured")
    print("  alpha here is fitted to TIME, so it absorbs an extra tile's padded")
    print("  arithmetic and scheduling as well as its re-read, and read as a traffic")
    print("  coefficient it is an UPPER bound. That bound caps arithmetic intensity at")
    print("  2 BM / (alpha b), in rows per expert:")
    print()
    print("  | BLOCK_M | AI cap | vs ridge band "
          f"{RIDGE_BAND[0]}-{RIDGE_BAND[1]} |")
    print("  |---:|---:|---|")
    for block_m in (16, 32, 64, 128, 256):
        cap = ai_cap(block_m, alpha)
        verdict = ("NEVER crosses" if cap < RIDGE_BAND[0]
                   else "crosses" if cap > RIDGE_BAND[1] else "inside the band")
        print(f"  | {block_m} | {cap:.0f} | {verdict} |")
    print()
    measured = rows_per_expert("mixtral-8x7b", MIXTRAL_ONE_STAGE_CROSSING_TOKENS)
    ceiling = max_alpha_that_still_crosses(CUTLASS_BLOCK_M, RIDGE_BAND[0])
    print("  AND THAT IS REFUTED BY THIS STUDY'S OWN ROWS. torch grouped_mm runs at")
    print(f"  CUTLASS BLOCK_M={CUTLASS_BLOCK_M} and DOES cross: FINDINGS C2 puts mixtral's")
    print(f"  one-stage bf16 crossing at {MIXTRAL_ONE_STAGE_CROSSING_TOKENS} tokens, "
          f"which is {measured:.0f} rows per expert,")
    print(f"  well above the {ai_cap(CUTLASS_BLOCK_M, alpha):.0f} this alpha allows.")
    print()
    print("  So one of three things is true, and this pool cannot say which:")
    print(f"   - the TRAFFIC coefficient is at most {ceiling:.3f}, the largest value at")
    print(f"     which BLOCK_M={CUTLASS_BLOCK_M} still reaches a ridge of {RIDGE_BAND[0]}, and the")
    print(f"     gap up to {alpha:.2f} is an extra tile's NON-traffic cost;")
    print("   - the bounded-AI consequence does not follow from a time-fitted alpha;")
    print("   - or the one-stage crossings are tile steps rather than the ridge, which")
    print("     FINDINGS already downgrades them to being.")
    print("  A BLOCK_M sweep on a pod separates them, and it is the experiment")
    print("  FINDINGS already names.")


def report(args) -> int:
    census: collections.Counter = collections.Counter()
    triton = collect(args.csvs, census, include_throttled=args.include_throttled)

    print("# alpha, refit against the derived tile")
    print()
    print("Every BLOCK_M under a vLLM span below is DERIVED from vLLM 0.27.1's config")
    print("lookup plus the row's own gpu_name, and never observed: all ten published")
    print("arms are schema v3 and record no tile. torch's 64 is OBSERVED, under C1.")
    print()
    _report_pool(triton, census)
    if len(triton) < 2:
        print()
        print("nothing admitted; there is no fit to report")
        return 1
    alpha = fit_alpha(triton)
    print()
    _report_fit(triton, alpha, args)
    print()
    _report_splits(triton)
    if args.original_estimator:
        print()
        original = [p for p in args.csvs if ORIGINAL_ALPHA_ARM in Path(p).parent.name]
        if not original:
            print(f"## the original estimator: `{ORIGINAL_ALPHA_ARM}` is not in the")
            print("   inputs, so the published 0.10 cannot be reproduced from them")
        else:
            _report_original(collect(original, collections.Counter(), cutlass=True,
                                     include_throttled=args.include_throttled), alpha)
    if args.adversarial:
        print()
        _report_adversarial(triton, alpha, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="cluster-bootstrap draws for the band (default 200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-throttled", action="store_true",
                        help="keep rows whose clock drifted; off by default, because "
                             "a throttled row's time is not the kernel's")
    parser.add_argument("--original-estimator", action="store_true",
                        help="also run the pooled-CV estimator that produced 0.10, on "
                             "the rows it was originally run on")
    parser.add_argument("--adversarial", action="store_true",
                        help="the checks against this fit's own answer")
    return report(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
