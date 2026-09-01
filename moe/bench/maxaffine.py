"""The other estimator: fit the whole curve, do not walk it.

`crossing.py` finds the memory-to-compute transition by walking adjacent slopes
and returning the first one that passes 0.5. That detector has a known failure
mode, documented at length in `all_crossings_from_points`: the measured curve is
a STAIRCASE in M-tiles per expert, so the slope spikes above 0.5 at every tile
step and sags below it on every tread, and 8 of the 16 canonical uniform cells
cross 0.5 upward more than once. qwen2 reaches a log-log slope of 1.720, which no
roofline transition can produce. A first-passage rule then answers "which step did
the token grid happen to sample first", not "where is the ridge".

arXiv:2608.13057 (TEMPO, Aug 2026) locates the same inflection differently, with
a GLOBAL max-affine fit rather than a local threshold:

    t = max(a + b G,  c + beta N)

`G` is the memory-side term (activated expert replicas) and `N` the compute-side
term (tokens, optionally rounded up to M-tiles). Two affine branches fitted over
the whole curve at once. THE POINT OF PORTING IT HERE is that this estimator
CANNOT have our failure mode: two affine pieces intersect once, so it returns one
inflection by construction and there is no step for it to trip on.

That is also the reason to distrust it. An estimator that cannot report a
staircase will not report one when the staircase is real, so a max-affine
inflection on a stepped curve is a smoothed-over answer rather than a corrected
one. Which is why every fit here carries its own relative error: if max-affine
describes these curves, its residuals are small and its single answer is worth
more than our two; if it is being forced onto a staircase, the residuals say so
and the single answer is an artefact of the model rather than a property of the
hardware. `MaxAffineFit.mean_rel_err` and `.p95_rel_err` are properties and not
an optional extra so that a reader never has to take the inflection without them.

WHAT IT DID ON THIS STUDY'S ROWS, measured on the canonical four-arm bf16 H200
pool at uniform routing, 16 cells, `N = tokens`, relative weighting:

 - it fits the SMOOTH curves and not the stepped ones. The two one-stage CUTLASS
   spans, whose BLOCK_M is fixed at 64 by the instruction set, fit to a mean
   relative error of 6.3 to 14.2% (median 9.0%). The two five-stage Triton spans,
   whose tile varies with batch, fit to 14.7 to 51.4% (median 24.0%) with a p95
   of up to 263%. A model that misses by 260% is not locating anything.
 - its single answer is EARLY, and on the five-stage cells it is not the ridge.
   Rows per expert at the inflection is 145.8 (CV 6%) on the one-stage cells,
   just under the measured ridge band of 160.3 to 176.2, and 48.2 (CV 30%) on the
   five-stage cells, three to nine times below it. The slope detector's LAST
   crossing gives 175.8 (CV 21%), inside the band.
 - "one crossing by construction" IS FALSE ALONG A MEASURED GRID. 14 of the 16
   cells have the fitted compute branch on top at T=1 and T=2, below it by T=4
   and on top again at the inflection, because `G` rises steeply at tiny batches
   while `N` has barely moved. The model crosses once in the (G, N) plane and
   twice along the path the sweep actually walks. `MaxAffineFit.reversals`
   counts it, and `single_crossing` is the honest predicate.

So on the cells where the slope detector is ambiguous, max-affine removes the
ambiguity and replaces it with an answer three times below the ridge, at a fit
error that says the model is not describing the curve. THE STAIRCASE IS THE
PHENOMENON, NOT THE DETECTOR. That is the result, and it is a negative one.

WHAT IS MEASURED AND WHAT IS DERIVED. `G` comes from `load_active_experts`, a
recorded column, and `N = tokens` from `num_tokens`, also recorded. `N = padded
rows` is DERIVED: it needs a BLOCK_M, every published arm is schema v3 and
records no tile configuration, so the tile height is read off vLLM's source plus
the row's `gpu_name` and is an assumption the caller states. `ComputeSide.derived`
carries that distinction into the return value so a report cannot print a derived
number in the same shape as an observed one.

ONE THING THE MEMORY BRANCH GETS FOR FREE, and it is worth naming because the
slope detector had to be told: `crossing_from_points` needs `min_tokens =
saturation_batch(model)`, because below `E/k` tokens a batch does not reach every
expert, weight traffic grows with the batch, and the slope crosses 0.5 for a
reason unrelated to the ridge. Regressing on `G` instead of on `T` removes the
need for that floor: `G` grows below saturation and flatlines above it, so a
single affine branch in `G` describes both regions and the sub-saturation points
are data rather than a hazard.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..routing.imbalance import TileEfficiencyUndetermined
from .crossing import m_tiles_for_row
from .schema import TileConfigUnrecorded, row_float

#: The recorded column that counts the experts whose weights the kernel must
#: read. TEMPO's `G` is "activated expert replicas", which on one GPU holding
#: every expert is exactly this: one replica per active expert, no sharding.
MEMORY_SIDE_COLUMN = "load_active_experts"

#: `N = num_tokens`, straight off the row. TEMPO's own compute-side term.
TOKENS = "tokens"

#: `N = active experts x M-tiles x BLOCK_M`, the rows the kernel actually
#: multiplies once each expert's rows are padded up to a whole tile. Needs a
#: BLOCK_M and is therefore DERIVED for every published arm.
#:
#: IT FITS BETTER AND MEANS LESS, which is why it is not the default. Below
#: saturation every expert holds fewer rows than one tile, so padded rows is
#: EXACTLY `BLOCK_M x active experts`: the compute-side regressor is then a
#: constant multiple of the memory-side one, the two branches are collinear, and
#: their intersection is set by whichever fits a rounding difference. Measured:
#: median mean relative error falls from 24.0% to 4.1% on the five-stage cells
#: while all eight of their inflections collapse to between 1.9 and 58.5 tokens,
#: every one of them BELOW that model's saturation batch, where `G` is still
#: growing and no layer is compute bound. A better residual bought by making the
#: parameter unidentifiable is not a better estimate.
#:
#: AND THE ANSWER IS THEN SET BY THE ASSUMPTION. Refitting the same five-stage
#: cells at BLOCK_M 64 instead of 128 moves mixtral's inflection from 2 tokens to
#: 288 and deepseek-v3's from 44 to 2601, while the one-stage cells move by at
#: most 11%. A number that swings 144x with a tile height NO PUBLISHED ROW
#: RECORDS is a reading of vLLM's config tree, not of the sweep.
PADDED_ROWS = "padded_rows"

COMPUTE_SIDES = (TOKENS, PADDED_ROWS)

#: Weight each residual by `1 / ms^2`, so the objective is relative error.
RELATIVE = "relative"

#: Weight every residual equally, the textbook least-squares objective.
ABSOLUTE = "absolute"

WEIGHTINGS = (RELATIVE, ABSOLUTE)

#: Two points fit a line exactly, so a branch with fewer says nothing, and a
#: split that leaves one side with fewer is not a fit of two branches.
MIN_BRANCH_POINTS = 2


@dataclass(frozen=True)
class ComputeSide:
    """Which compute-side regressor a fit used, and whether it was observed.

    A named pair rather than a bare string because the two available regressors
    differ in PROVENANCE and not just in value: `tokens` is a recorded column
    and `padded_rows` is computed from a BLOCK_M no published row states. Every
    tile-derived number in this study that got quoted as a measurement did so
    because the value travelled without its provenance, so here they travel
    together.
    """

    kind: str
    block_m: int | None

    @property
    def derived(self) -> bool:
        """True when the regressor needed a tile height the rows do not record."""
        return self.kind == PADDED_ROWS

    @property
    def label(self) -> str:
        if not self.derived:
            return "N = tokens (observed column)"
        return (f"N = padded rows at BLOCK_M {self.block_m} "
                "(DERIVED from vLLM's source, not recorded by these rows)")


@dataclass(frozen=True)
class Observation:
    """One token count, its two regressors and its median time.

    Medians across replicates for the same reason `crossing_report` medians
    them: uniform routing is redrawn per replicate, so a `G` taken off one row
    beside a time taken off eight would be describing two different draws.
    """

    tokens: float
    g: float
    n: float
    ms: float


@dataclass(frozen=True)
class Branch:
    """One affine piece, `value = intercept + slope * regressor`.

    `degenerate` marks a branch whose regressor never moved across the points
    assigned to it, which is not a fitting failure but the expected outcome on
    this data: `G` saturates at `E` once the batch reaches `E/k`, so a memory
    branch fitted entirely above saturation is a horizontal line and its `b` is
    unidentifiable. Reporting a slope of 0 without saying it was never
    constrained would be inventing a measurement of `b`.
    """

    intercept: float
    slope: float
    degenerate: bool

    def at(self, x: float) -> float:
        return self.intercept + self.slope * x


@dataclass(frozen=True)
class MaxAffineFit:
    """A fitted `max(a + b G, c + beta N)`, its inflection and its residuals.

    `inflections` is a tuple and not a scalar deliberately. Two affine branches
    intersect once in the plane, but the fit is evaluated along the measured
    token grid where `G(T)` and `N(T)` are both empirical, so a second crossing
    is arithmetically possible and the claim "one crossing by construction" is
    worth CHECKING on the data rather than asserting from the algebra. On the
    canonical pool it holds; the tuple is what makes that a result instead of an
    assumption.

    `reversals` counts the crossings in the other direction, where the memory
    branch retakes the compute branch. On this study's rows that count is 1 on
    14 of 16 canonical cells, so the check earns its place: the fitted model has
    a compute-dominant region at T=1 to 2, which is the most memory-bound corner
    of the entire dataset.
    """

    memory: Branch
    compute: Branch
    compute_side: ComputeSide
    weighting: str
    observations: tuple[Observation, ...]
    split_tokens: float
    n_memory: int
    n_compute: int
    inflections: tuple[float, ...]
    reversals: int

    def predict(self, obs: Observation) -> float:
        return max(self.memory.at(obs.g), self.compute.at(obs.n))

    @property
    def relative_errors(self) -> tuple[float, ...]:
        return tuple(abs(self.predict(o) - o.ms) / o.ms
                     for o in self.observations)

    @property
    def mean_rel_err(self) -> float:
        return statistics.fmean(self.relative_errors)

    @property
    def p95_rel_err(self) -> float:
        return _percentile(sorted(self.relative_errors), 0.95)

    @property
    def max_rel_err(self) -> float:
        return max(self.relative_errors)

    @property
    def inflection(self) -> float | None:
        """The single inflection, or None when the grid does not give exactly one.

        None rather than the first of several, because the whole reason to fit
        this model is that it is supposed to produce ONE answer. Silently
        returning the first would rebuild the first-passage reduction this
        module exists to avoid.
        """
        return self.inflections[0] if len(self.inflections) == 1 else None

    @property
    def single_crossing(self) -> bool:
        """Does the FITTED model cross this token grid exactly once?

        Not the same question as `inflection is not None`, and the difference is
        the one this module was built to check. The algebra gives one
        intersection of two planes; the measured grid is a path through the
        (G, N) plane along which the max can flip more than once, and on 13 of
        16 canonical cells it does. Quoting the paper's "one crossing by
        construction" without this predicate would repeat exactly the error this
        study keeps finding: a property of a model reported as a property of the
        data.
        """
        return len(self.inflections) == 1 and self.reversals == 0


def memory_side(rows: Sequence[Mapping]) -> float:
    """Median active experts over a token count's replicate rows."""
    return statistics.median(row_float(r, MEMORY_SIDE_COLUMN) for r in rows)


def compute_side(rows: Sequence[Mapping], side: ComputeSide) -> float:
    """Median compute-side regressor over a token count's replicate rows.

    Raises rather than falling back to tokens when a padded count cannot be
    determined. A curve with one point silently switched to a different
    regressor is exactly the kind of hole this project has spent a day finding
    in its own tables.
    """
    if not side.derived:
        return statistics.median(row_float(r, "num_tokens") for r in rows)
    block_m = side.block_m
    if block_m is None:                      # pragma: no cover - guarded in build
        raise ValueError("padded rows need a BLOCK_M")
    return statistics.median(m_tiles_for_row(r, block_m) * block_m for r in rows)


def observations(rows_by_tokens: Mapping[int, Sequence[Mapping]],
                 side: ComputeSide) -> list[Observation]:
    """One `Observation` per token count, sorted, medianed across replicates.

    Propagates `TileEfficiencyUndetermined` / `TileConfigUnrecorded` from the
    padded regressor instead of dropping the token count: a curve missing its
    largest T fits a different model than the one the caller asked for, and it
    would fit it without complaining.
    """
    out = []
    for tokens, rows in sorted(rows_by_tokens.items()):
        rows = [r for r in rows if row_float(r, "ms_p50") > 0.0]
        if not rows:
            continue
        out.append(Observation(
            tokens=float(tokens),
            g=memory_side(rows),
            n=compute_side(rows, side),
            ms=statistics.median(row_float(r, "ms_p50") for r in rows)))
    return out


def fit(obs: Sequence[Observation],
        side: ComputeSide,
        weighting: str = RELATIVE) -> MaxAffineFit | None:
    """Fit `t = max(a + b G, c + beta N)` over every point at once.

    THE SEARCH IS EXHAUSTIVE OVER CONTIGUOUS SPLITS, which is a choice and not
    an approximation to a harder problem. Max-affine regression in general is
    fitted by alternating between assigning points to whichever plane is larger
    and refitting, which is a local method and lands in a different optimum from
    a different start. Here both regressors are non-decreasing in T, so a point
    assigned to the memory branch above a point assigned to the compute branch
    would need the model to cross twice, which the model cannot do. That reduces
    the assignment to a single split index, and there are at most a dozen of
    them, so the global optimum is reachable by enumeration and no seed or
    tolerance enters the answer.

    EVERY CANDIDATE IS SCORED ON THE MAX, not on its two branches separately.
    Fitting each side to its own points and adding the two residuals rewards a
    split whose branches cross in the wrong place, because a branch is never
    charged for exceeding the other one where it should not. Scoring
    `max(...)` against every point charges exactly that.

    `weighting=RELATIVE` is the default and weights each residual by `1/ms^2`.
    Unweighted least squares on these curves is a fit to the largest token count
    alone: one canonical cell runs 0.13 ms at T=1 and 27.9 ms at T=8192, so the
    absolute residual at the top is 200x the entire time at the bottom, and the
    memory branch -- the branch every decode claim in this study lives on --
    would be fitted to nothing. `ABSOLUTE` is available so the choice can be
    shown to matter or not.

    None when there are too few points to give both branches
    `MIN_BRANCH_POINTS`, which is a real answer: a four-point curve does not
    determine two lines and an inflection between them.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    pts = sorted(obs, key=lambda o: o.tokens)
    if len({o.tokens for o in pts}) != len(pts):
        raise ValueError(
            "duplicate token counts: median to one observation per token count "
            "before fitting, rather than letting the fit weight one T twice")
    if len(pts) < 2 * MIN_BRANCH_POINTS:
        return None

    best: MaxAffineFit | None = None
    best_cost = math.inf
    for split in range(MIN_BRANCH_POINTS, len(pts) - MIN_BRANCH_POINTS + 1):
        lo, hi = pts[:split], pts[split:]
        memory = _wls([o.g for o in lo], [o.ms for o in lo],
                      _weights(lo, weighting))
        compute = _wls([o.n for o in hi], [o.ms for o in hi],
                       _weights(hi, weighting))
        cost = 0.0
        for o, w in zip(pts, _weights(pts, weighting), strict=True):
            resid = max(memory.at(o.g), compute.at(o.n)) - o.ms
            cost += w * resid * resid
        if cost < best_cost:
            up, down = _crossings(pts, memory, compute)
            best_cost = cost
            best = MaxAffineFit(
                memory=memory, compute=compute, compute_side=side,
                weighting=weighting, observations=tuple(pts),
                split_tokens=hi[0].tokens, n_memory=len(lo), n_compute=len(hi),
                inflections=tuple(up), reversals=down)
    return best


def _weights(pts: Sequence[Observation], weighting: str) -> list[float]:
    if weighting == ABSOLUTE:
        return [1.0] * len(pts)
    return [1.0 / (o.ms * o.ms) for o in pts]


def _wls(xs: Sequence[float], ys: Sequence[float],
         ws: Sequence[float]) -> Branch:
    """Weighted least squares on one regressor, degenerate case marked.

    A zero-variance regressor is the NORMAL case for the memory branch above
    saturation, not an error: every expert is already active, `G` is pinned at
    `E`, and the normal equations are singular. Returning the weighted mean with
    `degenerate=True` says "this branch is a horizontal line and its slope was
    never constrained", which is the truth. Solving anyway with a pseudo-inverse
    would return a slope of 0 that reads like a measurement.
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs, strict=True)) / sw
    my = sum(w * y for w, y in zip(ws, ys, strict=True)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs, strict=True))
    if sxx <= 0.0:
        return Branch(intercept=my, slope=0.0, degenerate=True)
    sxy = sum(w * (x - mx) * (y - my)
              for w, x, y in zip(ws, xs, ys, strict=True))
    slope = sxy / sxx
    return Branch(intercept=my - slope * mx, slope=slope, degenerate=False)


def _crossings(pts: Sequence[Observation], memory: Branch,
               compute: Branch) -> tuple[list[float], int]:
    """`(compute overtakes memory, count of the reverse)`, along the token grid.

    Interpolated LINEARLY IN T, and deliberately not in log T the way
    `crossing.upcrossings` does. That is not an inconsistency: a slope is a
    log-log derivative stamped at a geometric midpoint, so log space is where it
    is linear, while these two branches are affine in their regressors and the
    difference between them is affine in T whenever `N` is the token count. With
    `a + b G` at 0.05 + 0.01 x 64 against `0.02 + 0.003 N`, the model puts the
    crossing at exactly 223.33 tokens, and linear interpolation over a grid
    bracketing it at 128 and 256 returns exactly that where log-space
    interpolation returns 214.5. A four percent error that is purely a
    convention showing up in an answer.
    """
    up: list[float] = []
    down = 0
    tokens = [o.tokens for o in pts]
    diffs = [compute.at(o.n) - memory.at(o.g) for o in pts]
    for i in range(len(pts) - 1):
        t0, t1 = tokens[i], tokens[i + 1]
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 < 0.0 <= d1:
            f = (0.0 - d0) / (d1 - d0)
            up.append(t0 + f * (t1 - t0))
        elif d0 >= 0.0 > d1:
            down += 1
    return up, down


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics, matching `crossing.py`.

    Duplicated rather than imported because `crossing._percentile` is private to
    a module whose percentiles are over Monte Carlo draws; the two happening to
    agree today is not a contract either one owes the other.
    """
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


@dataclass(frozen=True)
class Comparison:
    """The two estimators' answers for one cell, side by side.

    Built so the head-to-head is computed once and printed or asserted from the
    same object. `slope_crossings` is EVERY upcrossing rather than the first,
    because the disagreement worth quantifying is not "two numbers differ" but
    "one estimator returns a set and the other returns a point", and a reduction
    to the first would hide that.
    """

    slope_crossings: tuple[float, ...]
    fit: MaxAffineFit | None

    @property
    def slope_ambiguous(self) -> bool:
        """The slope detector offers more than one answer for this cell."""
        return len(self.slope_crossings) > 1

    @property
    def affine_single(self) -> bool:
        """Max-affine offers exactly one, which is its whole claim."""
        return self.fit is not None and self.fit.inflection is not None

    @property
    def affine_monotone(self) -> bool:
        """Exactly one inflection and no reversal: a genuinely single answer.

        The strict reading of `affine_single`. Both are reported because they
        disagree on most cells, and a comparison that quoted only the loose one
        would credit max-affine with a property it does not have here.
        """
        return self.fit is not None and self.fit.single_crossing

    @property
    def resolves_ambiguity(self) -> bool:
        """A cell where the slope detector is ambiguous and max-affine is not.

        The claim under test, stated as a boolean so a test can count the cells
        where it holds instead of reading it off a printed table. RESOLVING THE
        AMBIGUITY IS NOT THE SAME AS BEING RIGHT: on the eight cells where this
        is true, the single answer max-affine returns sits below the slope
        detector's FIRST crossing on six, and on the six five-stage cells among
        them it sits at 39 to 63 rows per expert against a measured ridge band of
        160.3 to 176.2. Read it with `fit.mean_rel_err` beside it.
        """
        return self.slope_ambiguous and self.affine_single

    def ratio_to(self, crossing: float | None) -> float | None:
        """Max-affine's inflection over one of the slope detector's, or None.

        None when either side has no answer, so a caller aggregating ratios
        never divides by a crossing that was not found.
        """
        if not self.affine_single or crossing is None or crossing <= 0.0:
            return None
        return self.fit.inflection / crossing

    @property
    def first(self) -> float | None:
        return self.slope_crossings[0] if self.slope_crossings else None

    @property
    def last(self) -> float | None:
        return self.slope_crossings[-1] if self.slope_crossings else None


def build_compute_side(kind: str, block_m: int | None) -> ComputeSide:
    """Validate the regressor choice at the edge, once.

    A bad `kind` deep inside a fit surfaces as a wrong number rather than an
    error, since `compute_side` would silently take the tokens branch for
    anything that is not `PADDED_ROWS`.
    """
    if kind not in COMPUTE_SIDES:
        raise ValueError(f"compute side must be one of {COMPUTE_SIDES}")
    if kind == PADDED_ROWS and not block_m:
        raise ValueError("padded rows need a BLOCK_M, which no published row "
                         "records: name it rather than defaulting")
    return ComputeSide(kind=kind, block_m=block_m if kind == PADDED_ROWS else None)


def fit_rows(rows_by_tokens: Mapping[int, Sequence[Mapping]],
             kind: str = TOKENS,
             block_m: int | None = None,
             weighting: str = RELATIVE) -> MaxAffineFit | None:
    """`observations` then `fit`, for a caller that has rows and wants an answer.

    Returns None when the padded regressor cannot be determined for some token
    count, rather than fitting the curve that is left: a fit over a curve with a
    hole in it is not a fit of the curve the caller named.
    """
    side = build_compute_side(kind, block_m)
    try:
        obs = observations(rows_by_tokens, side)
    except (TileEfficiencyUndetermined, TileConfigUnrecorded):
        return None
    return fit(obs, side, weighting)
