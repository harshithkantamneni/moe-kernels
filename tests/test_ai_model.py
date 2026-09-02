"""The AI model with BOTH operand re-reads, and what a ladder fit returns on it.

WHAT THIS IS AND IS NOT. docs/FINDINGS.md draws its roofline from
AI(r) = (2r/b)/Q(r), which names only the weight re-read. `moe/bench/ai_model`
writes out all three terms, and this file tests two different things about
them and keeps them apart:

  * the BYTES: traffic, intensity, the three-term cap and its limit. These are
    arithmetic and the tests below are arithmetic checks.
  * the ESTIMATOR: what `LadderFit.alpha = B/(A+B)` returns when run over those
    bytes. That is (EXA), alpha_fitted = (alpha_b + phi)/(1 + phi + delta), and
    the decisive test here BUILDS a byte ladder with `traffic()` and FEEDS IT
    TO THE REAL `fit_ladder` from scripts/block_m_crossing_sweep.py. The first
    version of this file instead defined alpha_fitted by the (LIN) formula and
    checked its own rearrangement; no fit was called and the test could not
    fail whatever the estimator did (audit X57).

The consequence that matters: a cap computed as 2*BM/(alpha*b) from a ladder
alpha is HIGH by (1 + phi + delta), 31% at BM=128 / BN=64 on mixtral. The (LIN)
identity the earlier test pinned holds only for a fit that divides by weight
bytes, which no estimator in this repository does.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from moe.bench.ai_model import (
    AIModelRefused,
    alpha_b_from_fitted,
    arithmetic_intensity,
    cap,
    cap_from_fitted,
    decompose,
    exact_cap,
    fitted_alpha,
    ideal_intensity,
    ladder,
    lin_blend,
    lin_overstatement,
    phi,
    traffic,
)
from moe.spec import MODEL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIX = MODEL_CONFIGS["mixtral-8x7b"]
K = MIX.hidden_size                 # 4096
N = 2 * MIX.intermediate_size       # 28672, gate and up

#: The parameters the audit reproduced its numbers at: this module's own
#: (LIN)-solved pair, at the sweep's pinned BLOCK_N, with no fixed cost.
AUDIT_ALPHA_B, AUDIT_ALPHA_A, AUDIT_BN = 0.307, 0.143, 64


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The bytes. These were right before the audit and are unchanged.
# --------------------------------------------------------------------------

def test_zero_alpha_is_exactly_the_textbook_gemm_intensity():
    """2MNK / (b(MK + KN + MN)). If the model does not reduce to this when
    nothing is re-read, it is not a model of a GEMM."""
    for m in (64, 256, 4096):
        for bm, bn in ((16, 64), (128, 256), (256, 64)):
            got = arithmetic_intensity(m, N, K, block_m=bm, block_n=bn,
                                       alpha_b=0.0, alpha_a=0.0, b=2)
            assert got == pytest.approx(ideal_intensity(m, N, K, 2), rel=1e-12)


def test_intensity_converges_to_the_cap():
    """The cap is a limit, so it has to be one. A closed form that the finite
    calculation does not approach is a different formula, not a shortcut.

    `cap()` is approached to 2% (it drops the (1-alpha_a)/N slab term, audit
    X66); `exact_cap()` is approached to the residual of the finite M, which at
    M = 4e8 rows is a part in 1e5."""
    for bm, bn, ab, aa in ((64, 64, 0.31, 0.14), (128, 256, 0.31, 0.14),
                           (16, 64, 0.9, 0.5)):
        c = cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa)
        e = exact_cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa)
        big = arithmetic_intensity(400_000_000, N, K, block_m=bm, block_n=bn,
                                   alpha_b=ab, alpha_a=aa)
        assert big == pytest.approx(c, rel=0.02)
        assert big == pytest.approx(e, rel=1e-4)
        assert big < e < c, (
            "AI approaches the exact cap from below; the three-term cap sits "
            "above it by the dropped slab term")


def test_the_three_term_cap_drops_the_slab_term_by_under_one_percent():
    """Audit X66, stated rather than hidden. `cap()` omits (1-alpha_a)/N and is
    kept in that form because tests/test_memory_branch_anchor.py pins
    `ai_cap` to it. On mixtral shapes the gap is 0.4-0.8%, always in the
    direction of `cap()` being the higher number."""
    for bm in (64, 128, 256):
        c = cap(N, K, block_m=bm, block_n=AUDIT_BN,
                alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        e = exact_cap(N, K, block_m=bm, block_n=AUDIT_BN,
                      alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        assert 1.003 < c / e < 1.01, (bm, c / e)


def test_plugging_a_component_where_the_slope_belongs_overstates_the_cap():
    """The two-term form 2*BM/(b*x) wants x = alpha_b + phi, the whole per-tile
    slope in weight-read units. Hand it alpha_b alone, a component, and at
    BLOCK_M = BLOCK_N, where the activation term equals the weight term, the
    cap comes out 2x too high. That is not a defect in the formula; it is what
    happens when a component is read as the whole. The error grows as BLOCK_N
    shrinks against BLOCK_M."""
    component_in_slope_slot = 2 * 64 / (0.95 * 2)       # alpha_b where alpha_b + phi goes
    full = cap(N, K, block_m=64, block_n=64, alpha_b=0.95, alpha_a=0.95)
    assert full < component_in_slope_slot / 1.9
    # and it is the BM/BN ratio that governs, exactly as the study's own
    # "activation confound" bound says
    wide = cap(N, K, block_m=64, block_n=256, alpha_b=0.95, alpha_a=0.95)
    assert wide > full, "a wider N tile must reduce the activation re-read"


def test_the_output_write_caps_intensity_even_with_perfect_caching():
    """A third ceiling the published form does not have. At alpha = 0 it says
    infinity; the truth is 2K/b, because every output element is still written."""
    perfect = cap(N, K, block_m=1024, block_n=1024, alpha_b=0.0, alpha_a=0.0)
    assert perfect == pytest.approx(2.0 * K / 2, rel=1e-9)


def test_the_smallest_tile_sets_the_ceiling():
    """Three additive terms in the denominator means three ceilings, and the
    binding one is whichever is largest. Widening the tile that is not binding
    buys almost nothing, which is a design statement the weights-only form
    cannot make."""
    narrow_m = cap(N, K, block_m=16, block_n=256, alpha_b=0.5, alpha_a=0.5)
    narrow_n = cap(N, K, block_m=256, block_n=16, alpha_b=0.5, alpha_a=0.5)
    assert narrow_m == pytest.approx(narrow_n, rel=1e-9), (
        "the two re-read terms enter symmetrically; only BM/BN distinguishes them")


def test_a_miss_fraction_above_one_is_refused():
    """A value above 1 means the numerator carried traffic that is not that
    operand's, which is exactly how the contamination was found. Refuse rather
    than compute a cap from it."""
    with pytest.raises(AIModelRefused, match="outside"):
        cap(N, K, block_m=64, block_n=64, alpha_b=1.6, alpha_a=0.1)
    with pytest.raises(AIModelRefused, match="outside"):
        arithmetic_intensity(256, N, K, block_m=64, block_n=64,
                             alpha_b=0.5, alpha_a=-0.1)


def test_degenerate_shapes_are_refused_rather_than_returning_a_number():
    for kwargs in ({"M": 0}, {"N": 0}, {"K": 0}):
        args = {"M": 256, "N": N, "K": K} | kwargs
        with pytest.raises(AIModelRefused, match="positive"):
            arithmetic_intensity(args["M"], args["N"], args["K"],
                                 block_m=64, block_n=64,
                                 alpha_b=0.5, alpha_a=0.1)


def test_traffic_splits_by_operand_so_a_dropped_term_is_visible():
    """The whole defect was a total that omitted one of its parts, so the parts
    are reported separately and must sum to the total."""
    t = traffic(1024, N, K, block_m=64, block_n=64, alpha_b=0.31, alpha_a=0.14)
    assert t.total == pytest.approx(
        t.activation_bytes + t.weight_bytes + t.output_bytes)
    assert t.n_tiles == 16 and t.m_tiles == N // 64
    shares = t.share()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["activations"] > 0.05, (
        "at BLOCK_N=64 the activation traffic is a large share, which is the "
        "reason the weights-only formula is wrong here")


# --------------------------------------------------------------------------
# The estimator. What B/(A+B) returns on those bytes is (EXA), not (LIN).
# --------------------------------------------------------------------------

def test_phi_is_the_one_tile_cost_that_is_not_weights():
    """At BN | N, phi = alpha_a*BM/BN + BM/K + (1-alpha_a)*BM/N, and it does not
    depend on alpha_b. The closed form is checked against the byte count so
    that a reader can trust the pieces the docstrings quote: 0.16 / 0.32 / 0.64
    at BM = 64 / 128 / 256 on mixtral at BN=64."""
    for bm, want in ((64, 0.1605), (128, 0.3211), (256, 0.6422)):
        p = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=AUDIT_ALPHA_A)
        closed = (AUDIT_ALPHA_A * bm / AUDIT_BN + bm / K
                  + (1 - AUDIT_ALPHA_A) * bm / N)
        assert p == pytest.approx(closed, rel=1e-12)
        assert p == pytest.approx(want, abs=5e-4)


def test_exa_the_real_ladder_fit_on_a_traffic_ladder_returns_exa_not_lin():
    """THE DECISIVE ONE. Build the byte ladder with `traffic()`, hand it to the
    sweep's own `fit_ladder`, and read `LadderFit.alpha` back. It must be
    (alpha_b + phi)/(1 + phi + delta) to a part in 1e9, and it must NOT be the
    (LIN) blend: the two differ by the factor (1 + phi + delta), 32% at
    BM=128 / BN=64. Run with and without a fixed cost, because delta is the
    term that only shows up in the level.

    Membership goes through the production path: a `ComputeReference` whose
    scaled compute branch sits well below every tread, so all eight are memory
    bound and the fit is the sweep's own `_line` over all of them. The
    no-reference split search is NOT used, and the reason is recorded: on an
    affine ladder a through-origin line over the last tread alone is exact, so
    the search ties between "all memory" and "all but one", takes the first,
    and then discards the memory branch as parallel to a one-point compute
    branch. That is a property of the fallback, not of B/(A+B), and the
    reference path is the one every published ladder was read through."""
    sweep = _load_sweep()
    for bm, fixed in ((64, 0.0), (128, 0.0), (256, 0.0),
                      (128, 0.05 * K * N * 2), (32, 0.2 * K * N * 2)):
        pts = ladder(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                     alpha_a=AUDIT_ALPHA_A, b=2, n_max=8, fixed_bytes=fixed)
        delta = fixed / (K * N * 2)
        # a compute branch at a tenth of a weight read per tile at BM=1024,
        # scaled by C ~ BLOCK_M: far under every tread at every bm here
        ref = sweep.ComputeReference(1024, delta, 0.1, 0.0, "planted, byte ladder")
        fit = sweep.fit_ladder(pts, bm, ref)
        assert fit.memory_points == len(pts), fit.basis
        assert "membership from the compute branch" in fit.basis
        d = decompose(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                      alpha_a=AUDIT_ALPHA_A, b=2, fixed_bytes=fixed)
        exa = (AUDIT_ALPHA_B + d.phi) / (1.0 + d.phi + d.delta)
        assert fit.alpha == pytest.approx(exa, rel=1e-9), (bm, fixed)
        assert fit.alpha == pytest.approx(
            fitted_alpha(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                         alpha_a=AUDIT_ALPHA_A, b=2, fixed_bytes=fixed), rel=1e-9)
        # the LIN blend is NOT what came back, and the gap is the factor
        lin = lin_blend(K, block_m=bm, block_n=AUDIT_BN,
                        alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        factor = lin_overstatement(phi=d.phi, delta=d.delta)
        assert (AUDIT_ALPHA_B + d.phi) / fit.alpha == pytest.approx(factor, rel=1e-9)
        assert lin / fit.alpha == pytest.approx(factor, rel=1e-2), (
            "LIN is alpha_b + phi with the slab term dropped, so it sits a "
            "factor (1+phi+delta) above the fit, to under 1%")
        assert abs(lin - fit.alpha) / fit.alpha > 0.10, (
            "if LIN and the fit agreed to 10% the correction would not matter; "
            "at these tiles it is 16% to 64%")


def test_exa_reproduces_the_audits_numbers_where_lin_predicted_others():
    """Audit X57 fed this module's own parameters through the real estimator
    and got 0.403 / 0.475 / 0.578 at BM = 64 / 128 / 256 where (LIN) predicted
    0.466 / 0.625 / 0.942. Pinned so the module cannot drift back."""
    for bm, want_exa, want_lin in ((64, 0.4029, 0.4656), (128, 0.4754, 0.6242),
                                   (256, 0.5780, 0.9415)):
        got = fitted_alpha(N, K, block_m=bm, block_n=AUDIT_BN,
                           alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        assert got == pytest.approx(want_exa, abs=5e-4)
        assert lin_blend(K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                         alpha_a=AUDIT_ALPHA_A) == pytest.approx(want_lin, abs=5e-4)


def test_fitted_alpha_converges_to_alpha_b_as_phi_vanishes():
    """At delta = 0 and phi -> 0 (BN as wide as N so there is no activation
    re-read, K and N huge so the slab and output terms vanish) the estimator
    reads alpha_b itself. That is the regime in which (LIN) and (EXA) agree,
    and it is not the regime this study runs in."""
    ab = 0.31
    errs = []
    for scale in (12, 16, 20, 24):
        big = 2 ** scale
        got = fitted_alpha(big, big, block_m=16, block_n=big, alpha_b=ab, alpha_a=0.9)
        errs.append(abs(got - ab))
    assert errs == sorted(errs, reverse=True), "the error must fall monotonically"
    assert errs[-1] < 1e-5
    assert errs[0] > 1e-3, "and it must have been visible before it vanished"


def test_a_fixed_cost_only_ever_lowers_the_fitted_alpha():
    """delta sits in the level and nowhere else. The sweep's docstring says the
    fused layer's fixed cost pushes `LadderFit.alpha` DOWN and calls the raw
    fit a lower bound; (EXA) says the same in closed form, so check it."""
    base = fitted_alpha(N, K, block_m=128, block_n=64, alpha_b=0.5, alpha_a=0.2)
    prev = base
    for frac in (0.01, 0.05, 0.2, 1.0):
        got = fitted_alpha(N, K, block_m=128, block_n=64, alpha_b=0.5, alpha_a=0.2,
                           fixed_bytes=frac * K * N * 2)
        assert got < prev
        prev = got
    with pytest.raises(AIModelRefused, match="negative"):
        fitted_alpha(N, K, block_m=128, block_n=64, alpha_b=0.5, alpha_a=0.2,
                     fixed_bytes=-1.0)


def test_alpha_b_from_fitted_inverts_fitted_alpha_exactly_over_a_grid():
    """The (EXA) inverse recovers the alpha_b that generated the ladder, at
    every tile, every alpha and with or without a fixed cost, to rounding."""
    for bm in (16, 32, 64, 128, 256):
        for bn in (32, 64, 128, 256):
            for ab in (0.0, 0.05, 0.307, 0.558, 0.92, 1.0):
                for aa in (0.0, 0.143, 0.5, 1.0):
                    for fixed in (0.0, 0.1 * K * N * 2):
                        d = decompose(N, K, block_m=bm, block_n=bn, alpha_b=ab,
                                      alpha_a=aa, fixed_bytes=fixed)
                        back = alpha_b_from_fitted(d.alpha_fitted, phi=d.phi,
                                                   delta=d.delta)
                        assert back == pytest.approx(ab, abs=1e-12), (bm, bn, ab, aa)


def test_cap_from_fitted_is_the_exact_cap_and_lin_overstates_it_by_the_factor():
    """Three statements, in decreasing exactness, and the last is the audit's.

    1. `cap_from_fitted` at a ladder alpha equals `exact_cap` at the alpha_b
       that generated the ladder, to rounding.
    2. The study's 2*BM/(alpha_fitted*b) overstates that by EXACTLY
       (1 + phi + delta), with and without a fixed cost.
    3. Against `cap()`, which drops the slab term, the audit measured the
       overstatement as 15.6 / 31.3 / 62.9% at BM = 64 / 128 / 256 (mixtral,
       BN=64, no fixed cost). Reproduced to 0.2 points; the exact factor is
       16.1 / 32.1 / 64.2% and the difference is X66, not this defect."""
    b = 2
    for bm, bn, ab, aa, fixed in ((64, 64, 0.307, 0.143, 0.0),
                                  (128, 64, 0.307, 0.143, 0.0),
                                  (128, 256, 0.307, 0.143, 0.0),
                                  (128, 64, 0.92, 0.146, 0.05 * K * N * b),
                                  (256, 32, 0.9, 0.9, 0.0),
                                  (32, 64, 0.5, 0.2, 0.3 * K * N * b)):
        d = decompose(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa, b=b,
                      fixed_bytes=fixed)
        corrected = cap_from_fitted(d.alpha_fitted, block_m=bm, b=b,
                                    phi=d.phi, delta=d.delta)
        assert corrected == pytest.approx(
            exact_cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa, b=b),
            rel=1e-12)
        lin_cap = 2 * bm / (d.alpha_fitted * b)
        assert lin_cap / corrected == pytest.approx(
            lin_overstatement(phi=d.phi, delta=d.delta), rel=1e-12)
        assert lin_cap / corrected == pytest.approx(d.lin_overstatement, rel=1e-12)
    for bm, want_pct in ((64, 15.6), (128, 31.3), (256, 62.9)):
        d = decompose(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                      alpha_a=AUDIT_ALPHA_A, b=b)
        lin_cap = 2 * bm / (d.alpha_fitted * b)
        three_term = cap(N, K, block_m=bm, block_n=AUDIT_BN,
                         alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A, b=b)
        assert (lin_cap / three_term - 1.0) * 100 == pytest.approx(want_pct, abs=0.2)
        assert (d.lin_overstatement - 1.0) * 100 > want_pct, (
            "the exact factor is larger still, because cap() is already high")


def test_lin_cap_of_the_lin_blend_is_the_three_term_cap_and_that_is_all_it_is():
    """The identity commit f732035 pinned is TRUE of `lin_blend` and of nothing
    else: 2*BM/(lin_blend*b) == cap() to rounding. It says what a fit that
    divides by weight bytes would return. No estimator in the repository does,
    so the identity is a statement about the (LIN) formula, not about any
    published cap."""
    b = 2
    for bm, bn, ab, aa in ((32, 64, 0.307, 0.143), (128, 64, 0.307, 0.143),
                           (64, 256, 0.5, 0.2), (256, 32, 0.9, 0.9)):
        lin = lin_blend(K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa)
        assert 2 * bm / (lin * b) == pytest.approx(
            cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa, b=b), rel=1e-12)
        # and the estimator does not return it
        exa = fitted_alpha(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa, b=b)
        assert exa < lin


def test_alpha_b_from_fitted_refuses_a_result_outside_zero_one():
    """Both walls, each naming the floor or ceiling the inputs imply. A fitted
    alpha below phi/(1+phi+delta) cannot come from any alpha_b >= 0 on that
    ladder; one above (1+phi)/(1+phi+delta) cannot come from any alpha_b <= 1."""
    p = phi(N, K, block_m=128, block_n=64, alpha_a=0.143)
    floor = p / (1 + p)
    ceiling = (1 + p) / (1 + p)
    assert alpha_b_from_fitted(floor, phi=p, delta=0.0) == pytest.approx(0.0, abs=1e-12)
    assert alpha_b_from_fitted(ceiling, phi=p, delta=0.0) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(AIModelRefused, match="below the floor"):
        alpha_b_from_fitted(floor - 1e-3, phi=p, delta=0.0)
    with pytest.raises(AIModelRefused, match="above the ceiling"):
        alpha_b_from_fitted(1.0 + 1e-3, phi=p, delta=0.0)
    # a fixed cost lowers the ceiling below 1: a fit reading 1.0 with delta > 0
    # cannot be a miss fraction either
    with pytest.raises(AIModelRefused, match="delta understates"):
        alpha_b_from_fitted(1.0, phi=p, delta=0.2)
    # and the byte counts themselves cannot be negative
    with pytest.raises(AIModelRefused, match="phi=.*negative"):
        alpha_b_from_fitted(0.5, phi=-0.1, delta=0.0)
    with pytest.raises(AIModelRefused, match="delta=.*negative"):
        lin_overstatement(phi=0.1, delta=-0.1)
    # cap_from_fitted refuses through the same wall rather than printing a cap
    with pytest.raises(AIModelRefused, match="above the ceiling"):
        cap_from_fitted(1.2, block_m=128, b=2, phi=p, delta=0.0)
    with pytest.raises(AIModelRefused, match="positive"):
        cap_from_fitted(0.5, block_m=0, b=2, phi=p, delta=0.0)


def test_the_wall_tolerance_absorbs_rounding_and_nothing_wider():
    """`alpha_b_from_fitted` computes alpha_fitted*level - phi, and that
    multiply-subtract can land a rounding hair past 0 or 1 when the true value
    sits on the wall. The module clamps a hair (1e-9) to the wall and refuses
    anything wider. The exact-wall assertions in the test above pass whether
    or not that clamp exists, so this test plants both sides of it: 5e-10 past
    each wall must come back as EXACTLY the wall, and 2e-9 past it must
    refuse. A widened tolerance, which is a silent fallback wearing a
    rounding costume, fails the second half."""
    p = phi(N, K, block_m=128, block_n=64, alpha_a=0.143)
    level = 1.0 + p

    def fitted_for(alpha_b):
        # the (EXA) forward map, so the inverse lands at alpha_b to ~1e-16
        return (alpha_b + p) / level

    # inside the hair: clamped to the wall, not returned as a negative or a
    # value above one that would then be handed to cap_from_fitted
    assert alpha_b_from_fitted(fitted_for(-5e-10), phi=p, delta=0.0) == 0.0
    assert alpha_b_from_fitted(fitted_for(1.0 + 5e-10), phi=p, delta=0.0) == 1.0
    # outside the hair: refused, naming the wall
    with pytest.raises(AIModelRefused, match="below the floor"):
        alpha_b_from_fitted(fitted_for(-2e-9), phi=p, delta=0.0)
    with pytest.raises(AIModelRefused, match="above the ceiling"):
        alpha_b_from_fitted(fitted_for(1.0 + 2e-9), phi=p, delta=0.0)
    # and a value well inside the band is returned untouched, so the clamp is
    # a wall treatment and not a rounding of every result
    assert alpha_b_from_fitted(fitted_for(0.25), phi=p, delta=0.0) == pytest.approx(0.25, abs=1e-12)


def test_a_ladder_fit_cannot_exceed_one_so_a_reading_above_one_is_a_missing_term():
    """The earlier version of this test said fits above 1.0 were 'not
    impossible for the quantity actually being reported'. Under (EXA) they
    are: alpha_fitted <= (1+phi)/(1+phi+delta) <= 1 for every legal alpha_b,
    because the slope alpha_b + phi can at most match the level's 1 + phi.
    (LIN) exceeds 1 at BM=256 / BN=64, which is one more way it is not what the
    estimator returns. A published ladder alpha above 1 is therefore a slope
    carrying traffic the three-term model does not name."""
    ab, aa = 1.0, 1.0
    assert lin_blend(K, block_m=256, block_n=64, alpha_b=0.9, alpha_a=0.9) > 1.0
    for bm, bn in ((256, 64), (256, 32), (128, 64)):
        assert fitted_alpha(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa) <= 1.0
        assert fitted_alpha(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa,
                            fixed_bytes=0.1 * K * N * 2) < 1.0


def test_the_two_point_solve_of_alpha_by_block_m_depends_on_the_reading():
    """scripts/block_m_crossing_sweep.py records ALPHA_BY_BLOCK_M =
    {64: 0.466, 128: 0.625}. The first version of this file solved those two
    points through (LIN) for (alpha_b, alpha_a) = (0.307, 0.143) and quoted the
    0.307 against TEMPO's 0.311. Solve the SAME two points through (EXA): phi is
    linear in alpha_a, so alpha_b = f*(1+phi) - phi is linear in alpha_a at
    each BM, and equating the two lines gives a different pair. Both pairs
    reproduce the two inputs by construction; neither is evidence, and the
    published pair comes from a third estimator (alpha_refit) that is neither
    form. What this pins is that two points cannot choose a reading, so the
    TEMPO comparison built on the (LIN) pair was a unit artefact."""
    points = {64: 0.466, 128: 0.625}

    def line(bm, f):
        # alpha_b(alpha_a) = p + q*alpha_a, from phi at alpha_a = 0 and 1
        p0 = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=0.0)
        p1 = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=1.0)
        c1 = p1 - p0
        return f * (1 + p0) - p0, c1 * (f - 1)

    (p_64, q_64), (p_128, q_128) = line(64, points[64]), line(128, points[128])
    exa_a = (p_64 - p_128) / (q_128 - q_64)
    exa_b = p_64 + q_64 * exa_a
    assert 0.0 <= exa_a <= 1.0 and 0.0 <= exa_b <= 1.0, "a legal pair, so it cannot be excluded"
    assert exa_b == pytest.approx(0.073, abs=0.005)
    assert exa_a == pytest.approx(0.72, abs=0.01)
    assert abs(exa_b - AUDIT_ALPHA_B) > 0.2, "the LIN reading and the EXA reading disagree by 3x"
    # both pairs reproduce the two points through their own reading
    for bm, f in points.items():
        assert fitted_alpha(N, K, block_m=bm, block_n=AUDIT_BN,
                            alpha_b=exa_b, alpha_a=exa_a) == pytest.approx(f, abs=1e-9)
        assert lin_blend(K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B,
                         alpha_a=AUDIT_ALPHA_A) == pytest.approx(f, abs=2e-3)


def test_the_g1_ladders_read_near_0_92_through_exa_not_0_307():
    """The H200 mixtral G=1 ladder at BN=64 reports alpha_fitted 0.9327 and at
    BN=256 0.8235 (scripts/bn_decomposition.py). Through (EXA) at BM=32 with the
    (LIN)-era alpha_a of 0.143 those read alpha_b 0.93 and 0.82: three times
    the 0.307 that was compared with TEMPO's 0.311, and on the other side of
    that number. The agreement was an artefact of the reading."""
    for bn, f, want in ((64, 0.9327, 0.927), (256, 0.8235, 0.819)):
        p = phi(N, K, block_m=32, block_n=bn, alpha_a=AUDIT_ALPHA_A)
        got = alpha_b_from_fitted(f, phi=p, delta=0.0)
        assert got == pytest.approx(want, abs=0.005)
        assert got > 2.5 * AUDIT_ALPHA_B


def test_the_byte_ladder_is_affine_and_a_non_affine_one_is_refused(monkeypatch):
    """`decompose` reads slope and level off the ladder and checks them against
    the closed form, so that a `traffic()` that stopped growing one M-tile at a
    time would be refused rather than read as (EXA). Plant that world."""
    import moe.bench.ai_model as m

    pts = ladder(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=8)
    diffs = [t2 - t1 for (_, t1), (_, t2) in zip(pts[:-1], pts[1:], strict=True)]
    for dd in diffs:
        assert dd == pytest.approx(diffs[0], rel=1e-12)
    with pytest.raises(AIModelRefused, match="slope"):
        ladder(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=1)

    real = m.traffic

    def bent(M, N_, K_, **kw):
        t = real(M, N_, K_, **kw)
        # a second tile costs 1.5x what the model says: not affine any more
        return m.Traffic(t.activation_bytes, t.weight_bytes * (1.0 + 0.5 * (t.n_tiles > 1)),
                         t.output_bytes, t.m_tiles, t.n_tiles)

    monkeypatch.setattr(m, "traffic", bent)
    with pytest.raises(AIModelRefused, match="not affine"):
        m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)
