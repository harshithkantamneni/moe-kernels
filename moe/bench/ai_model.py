"""Arithmetic intensity of one tiled GEMM, with BOTH operand re-reads, and what a
ladder fit actually returns when it is run over those bytes.

WHAT THIS IS. docs/FINDINGS.md states the model as

    AI(r) = (2r/b) / Q(r),   Q(r) = 1 + alpha*(ceil(r/BLOCK_M) - 1)

which NAMES only the weight re-read. This module writes out all three terms of
the traffic (weights, activations, output), and then does the thing the first
version of this file did not do: it runs the study's own ladder estimator over
those bytes and reports what comes back. The two halves are separate on purpose.
`traffic`, `arithmetic_intensity`, `ideal_intensity` and `cap` describe BYTES
and are right. `ladder`, `decompose`, `fitted_alpha`, `alpha_b_from_fitted`,
`cap_from_fitted` and `lin_overstatement` describe an ESTIMATOR, and the
estimator is not the bytes.

THE THREE TERMS, derived from the standard GEMM basis, which is right and kept.
For C[M,N] = A[M,K] @ B[K,N] the compulsory traffic is MK + KN + MN elements and
the arithmetic is 2MNK, so

    AI_ideal = 2MNK / (b * (MK + KN + MN))

Now tile it. The grid is n = ceil(M/BM) by m = ceil(N/BN), and EACH tile reads a
BM x K slice of A and a K x BN slice of B. So A is read m times and B is read n
times. The weight re-read the study NAMES is the B side, and there is a
SYMMETRIC re-read on the A side that its fitted alpha carries without naming:

    A bytes = M*K*b * (1 + alpha_a*(m - 1))     activations, re-read per N-tile
    B bytes = K*N*b * (1 + alpha_b*(n - 1))     weights,     re-read per M-tile
    C bytes = M*N*b                             written once; K is a loop, not
                                                a split, so there is no
                                                accumulator round-trip

alpha_b is the study's `alpha`. alpha_a is its counterpart on the activations.
The ratio of the two re-read costs is exactly BM/BN, which the study already
knows as the "activation confound" bound on the alpha FIT. At the sweep's own
pinned BLOCK_N=64 with BLOCK_M=64 that ratio is 1.0: the activation term is as
large as the weight term.

WHAT A LADDER FIT RETURNS, which is the part this file got wrong on 2026-09-02
and the audit corrected the same day. A timing ladder at BLOCK_M is t(n) for
n = 1, 2, 3, ... full M-tiles. If time is bytes over bandwidth, then in units
of one full weight read W = K*N*b the byte ladder is

    t(n) = (1 - alpha_b + delta) + n*(alpha_b + phi)

with `phi` the activation-plus-output cost of ONE M-tile in weight-read units
(`phi()` below; at BN | N it is alpha_a*BM/BN + BM/K + (1-alpha_a)*BM/N) and
`delta` the fused layer's fixed cost in the same units. The estimator this
repository uses, scripts/block_m_crossing_sweep.py `LadderFit.alpha`, is
B/(A+B): the per-tile slope divided by the fitted LEVEL at n=1. Read off the
line above, that is

    alpha_fitted = (alpha_b + phi) / (1 + phi + delta)                    (EXA)

The level contains one weight read PLUS one tile's activation and output
traffic PLUS the fixed cost, and all of that lands in the denominator. The
first version of this module asserted instead

    alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K                        (LIN)

which is what a fit would return if it divided the slope by the WEIGHT BYTES
alone. No estimator in this repository does that. (LIN) is (EXA)'s NUMERATOR,
the slope alpha_b + phi with the level taken as 1 and the once-read slab
(1-alpha_a)*BM/N dropped. It is not even (EXA)'s first-order expansion, which
is alpha_b + (1-alpha_b)*phi; the two agree only as phi -> 0, and at the tiles
this study argues about phi is not small: 0.16, 0.32 and 0.64 at BM = 64, 128,
256 with BN = 64 on mixtral. scripts/bn_decomposition.py derived (EXA) at the
same HEAD and said the two are not interchangeable here. It was right.

WHAT THAT CHANGES. Under (LIN), 2*BM/(alpha_fitted*b) is the three-term cap
exactly, and commit f732035 and the first version of this docstring said so and
concluded that the published caps stand. Under (EXA) it is not:

    2*BM/(alpha_fitted*b) = (1 + phi + delta) * 2*BM/(b*(alpha_b + phi))

so every cap computed as 2*BM/(alpha*b) from a `LadderFit` alpha is HIGH by the
factor (1 + phi + delta): 16% at BM=64, 32% at BM=128 and 64% at BM=256 with
BN=64 on mixtral (15.6 / 31.3 / 62.9% against `cap()`, which drops a 0.4-0.8%
term, see its docstring). At BM=128, the one tile the study says matters, the
overstatement is larger than the cap-to-ridge gap it was being used to decide.
The (0.307, 0.143) pair this module used to quote was solved from
ALPHA_BY_BLOCK_M through (LIN); the same two points solved through (EXA) give
alpha_b = 0.07 and alpha_a = 0.72, and ALPHA_BY_BLOCK_M comes from a third
estimator (scripts/alpha_refit.py) that matches neither form. Two points cannot
choose a reading. And the "alpha_b = 0.307 agrees with TEMPO's b2/b = 0.311 to
2%" line was a unit artefact of reading through (LIN): through (EXA), with
alpha_a held at the (LIN)-era 0.143, the same G=1 ladders read alpha_b near
0.92 (scripts/bn_decomposition.py, LADDER column), which is nowhere near TEMPO
and is the number that has to be explained. Held is the operative word: solved
jointly under the FUSED-LAYER byte model of scripts/bn_decomposition.py (which
carries the layer's own W, and is not the single-GEMM model written here), the
BN=64 and BN=256 points give alpha_a = 4.7, impossible for a miss fraction, so
that pair has no consistent (alpha_b, alpha_a) under that model at all. Two
BN values cannot separate the two readings; three can, which is the experiment
scripts/bn_decomposition.py exists to run.

HISTORY. Derived 2026-09-02 as (LIN) and pinned by a test that rearranged its
own definition; no fit was ever called. Corrected the same day by the audit
(AUDIT_REPORT B1, findings X57 and X62), which fed `traffic()` through the real
estimator and got 0.403 / 0.475 / 0.578 where (LIN) predicted 0.466 / 0.625 /
0.942. `test_ai_model.py` now runs `LadderFit` on a `traffic()` ladder and
asserts (EXA) against it, so the identity is measured against the estimator
rather than against itself. Nothing here is a GPU measurement; `alpha_a` still
has no measurement anywhere in this repo, and every function takes it as an
argument rather than assuming one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class AIModelRefused(ValueError):
    """A shape, tile or fitted value that does not describe a GEMM this model can score."""


@dataclass(frozen=True)
class Traffic:
    """Bytes moved by one tiled GEMM, split by operand so a term can be checked.

    Kept separate rather than summed because the whole defect this module fixes
    was a total that silently omitted one of its parts.
    """

    activation_bytes: float
    weight_bytes: float
    output_bytes: float
    m_tiles: int
    n_tiles: int

    @property
    def total(self) -> float:
        return self.activation_bytes + self.weight_bytes + self.output_bytes

    def share(self) -> dict[str, float]:
        t = self.total
        return {"activations": self.activation_bytes / t,
                "weights": self.weight_bytes / t,
                "output": self.output_bytes / t}


def _check_positive(**dims: int) -> None:
    """Every shape, tile and byte width named here is a count; zero or less is
    not a GEMM and is refused by name rather than divided by."""
    for name, v in dims.items():
        # `not v > 0` rather than `v <= 0`: NaN fails every comparison, so
        # `nan <= 0` is False and a NaN dimension used to pass this check and
        # come out the far end as a NaN cap. The arguments are typed int, but a
        # float NaN arrives easily from a division upstream.
        if not v > 0:
            raise AIModelRefused(f"{name}={v} must be positive")


def _check_alphas(alpha_b: float, alpha_a: float) -> None:
    # alpha is a MISS FRACTION on a re-read. Outside [0, 1] it is not that
    # quantity, and a fit that returns one is reporting something else, which
    # is the whole reason this module exists. Refuse rather than extrapolate.
    for name, a in (("alpha_b", alpha_b), ("alpha_a", alpha_a)):
        if not 0.0 <= a <= 1.0:
            raise AIModelRefused(
                f"{name}={a} is outside [0, 1]. A miss fraction cannot exceed a "
                "full re-read; a value above 1 means the quantity being divided "
                "by the operand bytes contains more than that operand's traffic")


def _check(M: int, N: int, K: int, block_m: int, block_n: int,
           alpha_b: float, alpha_a: float, b: int) -> None:
    _check_positive(M=M, N=N, K=K, block_m=block_m, block_n=block_n, b=b)
    _check_alphas(alpha_b, alpha_a)


def traffic(M: int, N: int, K: int, *, block_m: int, block_n: int,
            alpha_b: float, alpha_a: float, b: int = 2) -> Traffic:
    """Bytes for C[M,N] = A[M,K] @ B[K,N] tiled BM x BN, with both re-reads."""
    _check(M, N, K, block_m, block_n, alpha_b, alpha_a, b)
    n_tiles = math.ceil(M / block_m)      # M-tiles: how often B is re-read
    m_tiles = math.ceil(N / block_n)      # N-tiles: how often A is re-read
    return Traffic(
        activation_bytes=M * K * b * (1.0 + alpha_a * (m_tiles - 1)),
        weight_bytes=K * N * b * (1.0 + alpha_b * (n_tiles - 1)),
        output_bytes=M * N * b,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
    )


def arithmetic_intensity(M: int, N: int, K: int, *, block_m: int, block_n: int,
                         alpha_b: float, alpha_a: float, b: int = 2) -> float:
    """FLOP per byte. At alpha_b = alpha_a = 0 this is the textbook GEMM value."""
    return 2.0 * M * N * K / traffic(
        M, N, K, block_m=block_m, block_n=block_n,
        alpha_b=alpha_b, alpha_a=alpha_a, b=b).total


def ideal_intensity(M: int, N: int, K: int, b: int = 2) -> float:
    """2MNK / (b(MK + KN + MN)). Compulsory traffic, every operand read once."""
    if min(M, N, K, b) <= 0:
        raise AIModelRefused("all of M, N, K, b must be positive")
    return 2.0 * M * N * K / (b * (M * K + K * N + M * N))


def cap(N: int, K: int, *, block_m: int, block_n: int,
        alpha_b: float, alpha_a: float, b: int = 2) -> float:
    """The three-term ceiling on AI as M grows without bound.

        cap = 2 / (b * (alpha_b/BM + alpha_a/BN + 1/K))

    M cancels because every term in the denominator grows linearly in it: more
    rows bring more arithmetic AND more weight re-reads AND more activation and
    output traffic, in fixed proportion. So batching stops helping, and where it
    stops is set by the two tile dimensions and the output width, not by M.

    THE THREE TERMS ARE THREE SEPARATE CEILINGS and the largest term wins:
      alpha_b/BM   weight re-reads, the part the study NAMES
      alpha_a/BN   activation re-reads, carried inside alpha_fitted unnamed
      1/K          the output write, which caps AI at 2K/b even with PERFECT
                   caching. A reader who mistakes alpha_fitted for alpha_b and
                   sets it to zero would conclude infinity.

    WHAT THIS FORM DROPS, stated because a test elsewhere pins it. The exact
    limit of `arithmetic_intensity` carries a fourth term, (1-alpha_a)/N: the
    once-read BM x K activation slab of each new M-tile, which is per-tile
    traffic that is neither a re-read nor an output write. On mixtral shapes it
    is 0.4-0.8% of the denominator (audit X66). `exact_cap()` keeps it; this
    function keeps the three-term form the study quotes, because
    tests/test_memory_branch_anchor.py pins `memory_branch_anchor.ai_cap` to it
    and the quoted form should stay where the study quotes it. Wherever
    "exactly" is claimed below, it is claimed against `exact_cap`, never
    against this.

    WHAT TO PUT IN THE alpha_b SLOT. A `LadderFit` alpha is NOT alpha_b, and
    2*BM/(alpha_fitted*b) is NOT this cap; see `cap_from_fitted`.
    """
    _check(1, N, K, block_m, block_n, alpha_b, alpha_a, b)
    denom = alpha_b / block_m + alpha_a / block_n + 1.0 / K
    return 2.0 / (b * denom)


def phi(N: int, K: int, *, block_m: int, block_n: int, alpha_a: float,
        b: int = 2) -> float:
    """The activation-plus-output cost of ONE M-tile, in units of one full
    weight read W = K*N*b.

    Taken from `traffic()` at M = BM rather than from a closed form so that it
    is, by construction, the per-tile cost the byte ladder actually carries. At
    BN | N it equals alpha_a*BM/BN + BM/K + (1-alpha_a)*BM/N: the activation
    re-read per N-tile, the output write, and the once-read activation slab.
    alpha_b does not enter: phi is the part of the slope that is NOT weights.

    This is the quantity that separates (EXA) from (LIN): (LIN) is the slope
    alpha_b + phi with the level 1 + phi + delta replaced by 1, so the two agree
    only as phi -> 0. On mixtral at BN=64 phi is 0.16 / 0.32 / 0.64 at
    BM = 64 / 128 / 256, which is why that replacement is not an approximation
    this study may use.
    """
    one_tile = traffic(block_m, N, K, block_m=block_m, block_n=block_n,
                       alpha_b=0.0, alpha_a=alpha_a, b=b)
    return (one_tile.activation_bytes + one_tile.output_bytes) / (K * N * b)


def exact_cap(N: int, K: int, *, block_m: int, block_n: int,
              alpha_b: float, alpha_a: float, b: int = 2) -> float:
    """The limit `arithmetic_intensity` actually approaches: 2*BM / (b*(alpha_b + phi)).

    This is `cap()` with the (1-alpha_a)/N term restored, and it is the
    denominator that makes every identity in this module exact rather than
    close. It is the cap a byte ladder's own slope implies, and the number a
    corrected `LadderFit` cap should be compared against.
    """
    _check(1, N, K, block_m, block_n, alpha_b, alpha_a, b)
    slope = alpha_b + phi(N, K, block_m=block_m, block_n=block_n, alpha_a=alpha_a, b=b)
    return 2.0 * block_m / (b * slope)


def _delta(fixed_bytes: float, N: int, K: int, b: int) -> float:
    if fixed_bytes < 0.0:
        raise AIModelRefused(
            f"fixed_bytes={fixed_bytes} cannot be negative: it is the fused "
            "layer's fixed cost, which a ladder pays on every tread")
    return fixed_bytes / (K * N * b)


def ladder(N: int, K: int, *, block_m: int, block_n: int, alpha_b: float,
           alpha_a: float, b: int = 2, n_max: int = 8,
           fixed_bytes: float = 0.0) -> list[tuple[int, float]]:
    """The byte ladder t(n), n = 1..n_max, in weight-read units.

    Each tread is `traffic(M = n*BM).total + fixed_bytes` divided by K*N*b. This
    is what a timing ladder IS if time is bytes over bandwidth, and it is the
    thing to hand to `LadderFit` when asking what that estimator returns under
    the three-term model. Under the model it is affine in n,

        t(n) = (1 - alpha_b + delta) + n*(alpha_b + phi)

    which `decompose` checks rather than assumes. Every tread is an exactly-full
    stack (M = n*BM), matching `ladder_points` in the sweep, which admits only
    aligned treads for the same reason: time is flat along a tread and a
    partial tile would put a padding artefact into a fit about traffic.
    """
    if n_max < 2:
        raise AIModelRefused(
            f"n_max={n_max}: a ladder is a slope, and one tread has none")
    delta = _delta(fixed_bytes, N, K, b)
    W = K * N * b
    return [(n, traffic(n * block_m, N, K, block_m=block_m, block_n=block_n,
                        alpha_b=alpha_b, alpha_a=alpha_a, b=b).total / W + delta)
            for n in range(1, n_max + 1)]


@dataclass(frozen=True)
class FittedAlpha:
    """What a B/(A+B) ladder fit returns on a three-term byte ladder, in pieces.

    `alpha_fitted` is the estimator's output. `slope` and `level` are the two
    numbers it divides, in weight-read units: slope = alpha_b + phi is what one
    more M-tile costs, level = 1 + phi + delta is what the first tread costs.
    `phi` and `delta` are exposed so a reader can see which part of the level
    is not weights, because that part is the whole difference between (EXA) and
    (LIN) and the factor by which a LIN cap is high.
    """

    alpha_fitted: float
    phi: float
    delta: float
    slope: float
    level: float

    @property
    def lin_overstatement(self) -> float:
        """(1 + phi + delta): what 2*BM/(alpha_fitted*b) overstates the exact cap by."""
        return self.level


#: Relative tolerance on the byte ladder being affine in n. It is affine by
#: construction of `traffic()`; this exists so that a future change to
#: `traffic()` that breaks the identity is REFUSED rather than silently read
#: through a two-point fit that no longer means (EXA).
_AFFINE_TOL = 1e-9


def decompose(N: int, K: int, *, block_m: int, block_n: int, alpha_b: float,
              alpha_a: float, b: int = 2, fixed_bytes: float = 0.0) -> FittedAlpha:
    """Run the two-point B/(A+B) fit on the byte ladder and return its pieces.

    The slope and level are READ OFF `ladder()`, not written down from the
    closed form, so that `alpha_fitted` here is what the estimator does to the
    bytes and the (EXA) closed form is something a test can check it against.
    `phi` and `delta` are computed independently; if the ladder is not the
    affine line they predict, the byte model and the estimator have parted
    company and this refuses rather than returning a number that is neither.
    """
    pts = ladder(N, K, block_m=block_m, block_n=block_n, alpha_b=alpha_b,
                 alpha_a=alpha_a, b=b, n_max=2, fixed_bytes=fixed_bytes)
    (_, t1), (_, t2) = pts
    slope = t2 - t1
    level = t1
    p = phi(N, K, block_m=block_m, block_n=block_n, alpha_a=alpha_a, b=b)
    d = _delta(fixed_bytes, N, K, b)
    expect_slope = alpha_b + p
    expect_level = 1.0 + p + d
    if (not math.isclose(slope, expect_slope, rel_tol=_AFFINE_TOL)
            or not math.isclose(level, expect_level, rel_tol=_AFFINE_TOL)):
        raise AIModelRefused(
            f"the byte ladder is not affine in n: slope {slope:.9g} vs "
            f"alpha_b + phi = {expect_slope:.9g}, level {level:.9g} vs "
            f"1 + phi + delta = {expect_level:.9g}. `traffic()` no longer "
            "grows one M-tile at a time, so B/(A+B) on it is not (EXA)")
    return FittedAlpha(alpha_fitted=slope / level, phi=p, delta=d,
                       slope=slope, level=level)


def fitted_alpha(N: int, K: int, *, block_m: int, block_n: int, alpha_b: float,
                 alpha_a: float, b: int = 2, fixed_bytes: float = 0.0) -> float:
    """What `LadderFit.alpha` returns on the three-term byte ladder: (EXA).

        alpha_fitted = (alpha_b + phi) / (1 + phi + delta)

    NOT the study's (LIN) blend, and never above 1 for legal inputs, because the
    slope alpha_b + phi can at most equal the level's 1 + phi. A fit reading
    above 1 is therefore a slope carrying traffic this model does not name, not
    a large BM/BN ratio as the first version of this module claimed.
    """
    return decompose(N, K, block_m=block_m, block_n=block_n, alpha_b=alpha_b,
                     alpha_a=alpha_a, b=b, fixed_bytes=fixed_bytes).alpha_fitted


def lin_blend(K: int, *, block_m: int, block_n: int, alpha_b: float,
              alpha_a: float) -> float:
    """(LIN): alpha_b + alpha_a*(BM/BN) + BM/K. APPROXIMATE, kept only to be
    compared against.

    What a fit would return if it divided the per-tile slope by the weight
    bytes alone and the once-read activation slab were free: (EXA)'s numerator
    with its level taken as 1, which agrees with (EXA) only as phi -> 0. No
    estimator in this repository does that. Handing this to 2*BM/(x*b)
    reproduces `cap()`, which is the identity commit f732035 mistook for a
    statement about `LadderFit`. It is a statement about this function.
    """
    _check_positive(K=K, block_m=block_m, block_n=block_n)
    _check_alphas(alpha_b, alpha_a)
    return alpha_b + alpha_a * (block_m / block_n) + block_m / K


def lin_overstatement(*, phi: float, delta: float) -> float:
    """(1 + phi + delta): the factor by which 2*BM/(alpha_fitted*b) computed
    from a B/(A+B) ladder alpha overstates the exact cap 2*BM/(b*(alpha_b + phi)).

    Reports that print a cap from a `LadderFit` alpha should divide by this,
    and print it. On mixtral at BN=64 with no fixed cost it is 1.16 / 1.32 /
    1.64 at BM = 64 / 128 / 256.
    """
    if phi < 0.0:
        raise AIModelRefused(f"phi={phi} cannot be negative: it is a byte count")
    if delta < 0.0:
        raise AIModelRefused(f"delta={delta} cannot be negative: it is a byte count")
    return 1.0 + phi + delta


#: How far past 0 or 1 a recovered alpha_b may land by rounding before it is a
#: refusal rather than a wall. The one multiply and one subtract that produce
#: it round at about 1e-16 for values near 1, so 1e-9 is seven orders of
#: magnitude above the rounding it absorbs and seven below any alpha this
#: study reports to four places. Widening it would turn a wall into a silent
#: fallback; tests/test_ai_model.py pins both sides of it.
_WALL_TOL = 1e-9


def alpha_b_from_fitted(alpha_fitted: float, *, phi: float, delta: float) -> float:
    """The (EXA) inverse: alpha_b = alpha_fitted*(1 + phi + delta) - phi.

    Recovers the weight miss fraction from what a B/(A+B) ladder fit returned,
    given the per-tile activation-and-output cost and the fixed cost in
    weight-read units. This replaces the (LIN) inverse the first version of
    this module carried, which subtracted alpha_a*BM/BN + BM/K and was the
    wrong operation for every estimator in the repository.

    REFUSES a result outside [0, 1], naming which input is implausible. On a
    ladder with this phi and delta, alpha_b = 0 reads phi/(1+phi+delta) and
    alpha_b = 1 reads (1+phi)/(1+phi+delta); a fitted value outside that band
    cannot have come from the three-term model with these inputs.
    """
    # NaN fails every comparison below, so it would fall through the whole
    # wall-and-refusal ladder and be RETURNED as an alpha_b. A degenerate fit
    # produces one easily: zero-variance treads give a 0/0 slope. Refuse here,
    # before any comparison, or the gate has a hole exactly where the numbers
    # are least trustworthy.
    # AND NOT ONLY NaN. An infinite phi or delta multiplies through to
    # inf - inf = NaN one line later, which then walks the same unbounded path
    # the NaN check above it was written to close: the review that found it
    # sent phi=inf through and got NaN back. isfinite closes both ends at once.
    for name, value in (("alpha_fitted", alpha_fitted), ("phi", phi), ("delta", delta)):
        if not math.isfinite(value):
            kind = "NaN" if math.isnan(value) else "infinite"
            raise AIModelRefused(
                f"{name} is {kind}, which no comparison can bound. A NaN here is a "
                "degenerate ladder fit (a zero-variance or single-point branch) and "
                "an infinity is a shape or cost that no ladder produced: fix the "
                "input rather than reading a cap from it")
    level = lin_overstatement(phi=phi, delta=delta)
    floor = phi / level
    ceiling = (1.0 + phi) / level
    alpha_b = alpha_fitted * level - phi
    # A reading AT a wall is legal (alpha_b = 0 is no re-read, alpha_b = 1 a
    # full one) and the multiply-subtract above can land a hair past it by
    # rounding. That hair is not a refusal; anything wider is.
    if -_WALL_TOL <= alpha_b < 0.0:
        return 0.0
    if 1.0 < alpha_b <= 1.0 + _WALL_TOL:
        return 1.0
    if alpha_b < 0.0:
        raise AIModelRefused(
            f"alpha_fitted={alpha_fitted:.4f} is below the floor {floor:.4f} "
            f"that alpha_b=0 gives with phi={phi:.4f}, delta={delta:.4f}: the "
            "fit's per-tile slope is smaller than one tile's activation and "
            "output traffic alone, so phi overstates the per-tile cost for this "
            "ladder or alpha_fitted is not a B/(A+B) reading")
    if alpha_b > 1.0:
        raise AIModelRefused(
            f"alpha_fitted={alpha_fitted:.4f} is above the ceiling {ceiling:.4f} "
            f"that alpha_b=1 gives with phi={phi:.4f}, delta={delta:.4f}: the "
            "fit's per-tile slope exceeds one full weight read plus one tile's "
            "activation and output traffic, so the slope carries traffic this "
            "model does not name, or delta understates the fixed cost in the level")
    return alpha_b


def cap_from_fitted(alpha_fitted: float, *, block_m: int, b: int,
                    phi: float, delta: float) -> float:
    """The cap a B/(A+B) ladder alpha actually implies.

        cap = 2*BM / (b * (alpha_b + phi)),   alpha_b = alpha_fitted*(1+phi+delta) - phi
            = 2*BM / (alpha_fitted * b) / (1 + phi + delta)

    which is `exact_cap()` at the recovered alpha_b, and the study's
    2*BM/(alpha_fitted*b) divided by `lin_overstatement`. Refuses, through
    `alpha_b_from_fitted`, whenever the recovered miss fraction is not one.
    """
    _check_positive(block_m=block_m, b=b)
    alpha_b = alpha_b_from_fitted(alpha_fitted, phi=phi, delta=delta)
    return 2.0 * block_m / (b * (alpha_b + phi))
