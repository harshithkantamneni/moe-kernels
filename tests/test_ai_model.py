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
alpha is HIGH by (1 + phi + delta). That factor is a BRACKET over the
unmeasured alpha_a, 3.6% to 203% at BM=128 / BN=64 on mixtral
(`overstatement_bracket`); the 31% this docstring once quoted as a point is
the value at alpha_a = 0.143, the withdrawn (LIN)-solved pair, and every
point figure in this file is pinned at that pair BY NAME (AUDIT_ALPHA_A) and
checked to lie inside the bracket. The (LIN) identity the earlier test pinned
holds only for a fit that divides by weight bytes, which no estimator in this
repository does.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import statistics
import sys
from dataclasses import replace
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
    overstatement_bracket,
    phi,
    slope_weight_streams,
    traffic,
)
from moe.bench.weights import (
    WeightSetRefused,
    layer_weight_bytes,
    routed_expert_weight_bytes,
    weight_stream_ms,
    weight_streams_per_tile,
)
from moe.spec import MODEL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIX = MODEL_CONFIGS["mixtral-8x7b"]
K = MIX.hidden_size                 # 4096
N = 2 * MIX.intermediate_size       # 28672, gate and up

#: The parameters the audit reproduced its numbers at: this module's own
#: (LIN)-solved pair, at the sweep's pinned BLOCK_N, with no fixed cost. The
#: pair is WITHDRAWN as evidence (two points cannot choose a reading, and
#: alpha_a is unmeasured); it is kept here as the named point at which every
#: reproduced figure in this file was computed, so that no figure below is a
#: point quoted without the alpha_a it assumes.
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


def test_the_three_term_cap_gap_is_a_bracket_over_alpha_a_not_a_point():
    """Audit X66, stated as what it is. `cap()` omits (1-alpha_a)/N, so it
    sits above `exact_cap` by ((1-alpha_a)/N) / (alpha_b/BM + alpha_a/BN + 1/K),
    always in the direction of `cap()` being the higher number. The "0.4-0.8%"
    this test's name once carried was that gap at the withdrawn (0.307, 0.143)
    pair; over the unmeasured alpha_a at alpha_b = 0.307 it is 0.00% (alpha_a
    = 1, the slab is a re-read and cap() counts it) to 2.42% (alpha_a = 0 at
    BM=256), and over the whole legal square its ceiling is K/N = 14.3% at the
    perfect-caching corner, at every BM. The reason the module gives for
    keeping `cap()` is no longer the memory_branch_anchor pin (removed in
    c0efa7d; `ai_cap` goes through `cap_from_fitted`), so this test does not
    say it is."""
    worst = 0.0
    for bm in (64, 128, 256):
        c = cap(N, K, block_m=bm, block_n=AUDIT_BN,
                alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        e = exact_cap(N, K, block_m=bm, block_n=AUDIT_BN,
                      alpha_b=AUDIT_ALPHA_B, alpha_a=AUDIT_ALPHA_A)
        assert 1.003 < c / e < 1.01, ("at the audit's own pair", bm, c / e)
        # the ends of the bracket over alpha_a at that alpha_b
        lo = cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B, alpha_a=1.0) \
            / exact_cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B, alpha_a=1.0)
        hi = cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B, alpha_a=0.0) \
            / exact_cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=AUDIT_ALPHA_B, alpha_a=0.0)
        assert lo == pytest.approx(1.0, abs=1e-12), "at alpha_a = 1 nothing is dropped"
        assert lo < c / e < hi, "the audit's point lies inside the bracket"
        worst = max(worst, hi - 1.0)
    assert worst == pytest.approx(0.0242, abs=5e-4), "2.42% at BM=256, alpha_a=0"
    # the whole-square ceiling: alpha_b = alpha_a = 0 leaves the slab as the
    # only per-tile activation traffic, and the gap is K/N at any tile
    for bm in (16, 128, 1024):
        corner = cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=0.0, alpha_a=0.0) \
            / exact_cap(N, K, block_m=bm, block_n=AUDIT_BN, alpha_b=0.0, alpha_a=0.0)
        assert corner - 1.0 == pytest.approx(K / N, rel=1e-9), (bm, corner)
    assert K / N == pytest.approx(0.1429, abs=1e-4)


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
    at BM = 64 / 128 / 256 on mixtral at BN=64 AT alpha_a = 0.143, the
    withdrawn pair, and the ends 0.018 to 1.02 / 0.036 to 2.03 / 0.071 to 4.06
    over alpha_a in [0, 1], between which every point figure has to lie."""
    for bm, want, want_lo, want_hi in ((64, 0.1605, 0.0179, 1.0156),
                                       (128, 0.3211, 0.0357, 2.0312),
                                       (256, 0.6422, 0.0714, 4.0625)):
        p = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=AUDIT_ALPHA_A)
        closed = (AUDIT_ALPHA_A * bm / AUDIT_BN + bm / K
                  + (1 - AUDIT_ALPHA_A) * bm / N)
        assert p == pytest.approx(closed, rel=1e-12)
        assert p == pytest.approx(want, abs=5e-4)
        lo = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=0.0)
        hi = phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=1.0)
        assert lo == pytest.approx(want_lo, abs=5e-4)
        assert hi == pytest.approx(want_hi, abs=5e-4)
        assert lo < p < hi


def test_exa_the_real_ladder_fit_on_a_traffic_ladder_returns_exa_not_lin():
    """THE DECISIVE ONE. Build the byte ladder with `traffic()`, hand it to the
    sweep's own `fit_ladder`, and read `LadderFit.alpha` back. It must be
    (alpha_b + phi)/(1 + phi + delta) to a part in 1e9, and it must NOT be the
    (LIN) blend: the two differ by the factor (1 + phi + delta), 32% at
    BM=128 / BN=64 at the audit's alpha_a = 0.143 (3.6% to 203% over the
    unmeasured alpha_a). Run with and without a fixed cost, because delta is
    the term that only shows up in the level.

    Membership goes through the production path: a `ComputeReference` whose
    scaled compute branch sits well below every tread, so all eight are memory
    bound and the fit is the sweep's own `_line` over all of them. The
    no-reference split search is NOT used, for two reasons. It is not the
    path any published ladder was read through, so a test of "what the
    estimator returns" that went through it would be testing a different
    estimator. And its membership is an OUTCOME of a search rather than an
    input: on these five ladders it returns `undecided_parallel_branch` and
    no alpha at all on the fixed-cost BM=128 case (review 2026-09-03,
    review_fallback.py), which is a property of the fallback's split logic
    and not of B/(A+B). The first version of this docstring said the search
    ties, takes the first split, and discards the memory branch on every
    ladder; it does not, and that description was never executed."""
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
            "at these tiles and the audit's alpha_a = 0.143 it is 16% to 64%")


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
       BN=64, no fixed cost, AT alpha_a = 0.143). Reproduced to 0.2 points;
       the exact factor at that alpha_a is 16.1 / 32.1 / 64.2% and the
       difference is X66, not this defect. Both are points inside the
       `overstatement_bracket` over the unmeasured alpha_a, and the last
       assertion says so rather than letting the point stand alone."""
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
        lo, hi = overstatement_bracket(N, K, block_m=bm, block_n=AUDIT_BN, b=b)
        assert lo < 1.0 + want_pct / 100 < d.lin_overstatement < hi, (
            "both the audit's point and the exact point are POINTS inside the "
            "bracket over alpha_a; neither is the overstatement")


def test_overstatement_bracket_has_exact_ends_and_every_point_figure_sits_inside():
    """The overstatement is a bracket because alpha_a is unmeasured, and the
    bracket's ends are exact because phi is linear and increasing in alpha_a.
    Three things are pinned: the ends ARE `lin_overstatement` at alpha_a = 0
    and 1; every interior alpha_a lands between them; and the two ends at
    BM=128 / BN=64 on mixtral are 1.036 and 3.031, the 3.6% to 203% that
    replaces the 32% the docstrings once quoted as a point. delta shifts both
    ends together, and the FAIL branches are a negative delta and a shape
    that is not a GEMM, refused through the functions the bracket is built
    from rather than by a check of its own."""
    b = 2
    for bm, want_lo, want_hi in ((64, 1.0179, 2.0156), (128, 1.0357, 3.0312),
                                 (256, 1.0714, 5.0625)):
        lo, hi = overstatement_bracket(N, K, block_m=bm, block_n=AUDIT_BN, b=b)
        assert lo == pytest.approx(want_lo, abs=5e-4), bm
        assert hi == pytest.approx(want_hi, abs=5e-4), bm
        assert lo == lin_overstatement(
            phi=phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=0.0, b=b), delta=0.0)
        assert hi == lin_overstatement(
            phi=phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=1.0, b=b), delta=0.0)
        for aa in (0.05, AUDIT_ALPHA_A, 0.5, 0.95):
            inside = lin_overstatement(
                phi=phi(N, K, block_m=bm, block_n=AUDIT_BN, alpha_a=aa, b=b), delta=0.0)
            assert lo < inside < hi, (bm, aa)
    # the retracted headline: 32% was the point at alpha_a = 0.143, inside
    # a bracket whose ends differ by a factor of 56 in the overstatement
    lo, hi = overstatement_bracket(N, K, block_m=128, block_n=64, b=b)
    assert (lo - 1.0) * 100 == pytest.approx(3.6, abs=0.05)
    assert (hi - 1.0) * 100 == pytest.approx(203.1, abs=0.1)
    assert lo < 1.32 < hi
    # a fixed cost lifts both ends by the same delta
    lo_d, hi_d = overstatement_bracket(N, K, block_m=128, block_n=64, b=b, delta=0.05)
    assert lo_d == pytest.approx(lo + 0.05, rel=1e-12)
    assert hi_d == pytest.approx(hi + 0.05, rel=1e-12)
    # FAIL branches
    with pytest.raises(AIModelRefused, match="delta=.*negative"):
        overstatement_bracket(N, K, block_m=128, block_n=64, b=b, delta=-0.01)
    with pytest.raises(AIModelRefused, match="positive"):
        overstatement_bracket(N, K, block_m=0, block_n=64, b=b)
    with pytest.raises(AIModelRefused, match="positive"):
        overstatement_bracket(N, math.nan, block_m=128, block_n=64, b=b)


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
    """`decompose` fits slope and level over an eight-tread ladder and checks
    EVERY tread against the closed form, so that a `traffic()` that stopped
    growing one M-tile at a time would be refused rather than read as (EXA).
    Plant that world twice: bent at the second tile, which the two-tread guard
    this function had until 2026-09-03 could see, and bent at the THIRD, which
    it could not. The review that found the hole (review_fallback.py) bent the
    weights at n_tiles > 2, got the unbent (EXA) 0.3737 back from `decompose`,
    and 0.5433 from the real eight-tread `fit_ladder` on the same bytes. The
    docstrings said such a `traffic()` "is refused"; it was not. `n_max=2`
    reproduces the old guard so its blind spot stays executable."""
    import moe.bench.ai_model as m

    pts = ladder(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=8)
    diffs = [t2 - t1 for (_, t1), (_, t2) in zip(pts[:-1], pts[1:], strict=True)]
    for dd in diffs:
        assert dd == pytest.approx(diffs[0], rel=1e-12)
    with pytest.raises(AIModelRefused, match="slope"):
        ladder(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=1)
    # on the unbent ladder the eight-tread OLS and the two-tread read agree,
    # so widening the guard changed no number
    full = m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)
    two = m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=2)
    assert full.alpha_fitted == pytest.approx(two.alpha_fitted, rel=1e-12)
    unbent = full.alpha_fitted

    real = m.traffic

    def bent_after(first_bent_tile):
        def bent(M, N_, K_, **kw):
            t = real(M, N_, K_, **kw)
            # every tile from `first_bent_tile` on costs 1.5x what the model
            # says: not affine any more
            factor = 1.0 + 0.5 * (t.n_tiles >= first_bent_tile)
            return m.Traffic(t.activation_bytes, t.weight_bytes * factor,
                             t.output_bytes, t.m_tiles, t.n_tiles)
        return bent

    # bent at tile 2: the case the old guard saw, still refused
    monkeypatch.setattr(m, "traffic", bent_after(2))
    with pytest.raises(AIModelRefused, match="not affine.*tread 2"):
        m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)

    # bent at tile 3: THE FAIL BRANCH the old guard could not see
    monkeypatch.setattr(m, "traffic", bent_after(3))
    with pytest.raises(AIModelRefused, match="not affine.*tread 3"):
        m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)
    with pytest.raises(AIModelRefused, match="not affine"):
        m.fitted_alpha(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)
    # and the two-tread form returns the unbent number as if nothing happened,
    # which is exactly what the review caught
    blind = m.decompose(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=2)
    assert blind.alpha_fitted == pytest.approx(unbent, rel=1e-12)
    assert unbent == pytest.approx(0.3737, abs=5e-4)
    # the real estimator on the same bent bytes returns a different number,
    # so the two-tread guard was hiding a real disagreement
    sweep = _load_sweep()
    bent_pts = m.ladder(N, K, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1, n_max=8)
    monkeypatch.setattr(m, "traffic", real)
    fit = sweep.fit_ladder(bent_pts, 64, sweep.ComputeReference(1024, 0.0, 1e-6, 0.0, "planted"))
    assert fit.memory_points == 8
    assert fit.alpha == pytest.approx(0.5433, abs=5e-4)
    assert abs(fit.alpha - unbent) > 0.1


def test_a_nan_is_refused_before_it_can_pass_the_wall_comparisons():
    """THE HOLE THE AUDIT'S SECOND REVIEW FOUND. NaN fails every comparison,
    so `alpha_b < 0.0`, `alpha_b > 1.0` and both wall tolerances are all False
    and the value falls through to `return alpha_b` as if it were in band. A
    degenerate ladder produces NaN easily (a zero-variance or single-point
    memory branch gives a 0/0 slope), which is precisely when a cap must not
    be read. Every NaN input is refused before the ladder of comparisons."""
    for kwargs in ({"alpha_fitted": math.nan, "phi": 0.3, "delta": 0.0},
                   {"alpha_fitted": 0.5, "phi": math.nan, "delta": 0.0},
                   {"alpha_fitted": 0.5, "phi": 0.3, "delta": math.nan}):
        with pytest.raises(AIModelRefused, match="NaN"):
            alpha_b_from_fitted(**kwargs)


def test_an_infinite_alpha_fitted_is_refused_by_the_finiteness_guard():
    """The sibling case, kept beside it. Until ee83978 this docstring said an
    infinite alpha_fitted "is caught by the band refusal rather than the NaN
    guard", because infinity does order against the walls. ee83978 widened
    the guard to `isfinite` for phi and delta, and alpha_fitted went through
    the same loop, so an infinite alpha_fitted is now refused as "infinite"
    BEFORE any wall comparison; the sentence outlived the code it described
    by one commit. Pin the message so the site that refuses is the one the
    docstring names."""
    with pytest.raises(AIModelRefused, match="alpha_fitted is infinite"):
        alpha_b_from_fitted(alpha_fitted=math.inf, phi=0.3, delta=0.0)
    with pytest.raises(AIModelRefused, match="alpha_fitted is infinite"):
        alpha_b_from_fitted(alpha_fitted=-math.inf, phi=0.3, delta=0.0)


def test_an_infinite_phi_or_delta_is_refused_rather_than_returning_nan():
    """THE HOLE BESIDE THE ONE d601442 CLOSED. The NaN check tested each input
    with isnan; an infinite phi or delta passed it, multiplied through to
    inf - inf = NaN on the next line, and that NaN then walked the same
    unbounded path. Found in review by sending phi=inf through. Every
    non-finite input is refused before any arithmetic, and the message says
    which kind it was."""
    for kwargs, kind in (({"alpha_fitted": 0.5, "phi": math.inf, "delta": 0.0}, "infinite"),
                         ({"alpha_fitted": 0.0, "phi": 0.3, "delta": math.inf}, "infinite"),
                         ({"alpha_fitted": 0.5, "phi": math.nan, "delta": 0.0}, "NaN")):
        with pytest.raises(AIModelRefused, match=kind):
            alpha_b_from_fitted(**kwargs)
    with pytest.raises(AIModelRefused, match="infinite"):
        cap_from_fitted(0.5, block_m=128, b=2, phi=math.inf, delta=0.0)


def test_a_nan_dimension_is_refused_by_the_positivity_check():
    """`v <= 0` is False for NaN, so a NaN block_m, K or b used to pass the
    positivity check and surface as a NaN cap. `not v > 0` catches it."""
    with pytest.raises(AIModelRefused, match="positive"):
        cap_from_fitted(0.5, block_m=math.nan, b=2, phi=0.3, delta=0.0)
    with pytest.raises(AIModelRefused, match="positive"):
        cap(N, math.nan, block_m=64, block_n=64, alpha_b=0.3, alpha_a=0.1)


# --------------------------------------------------------------------------
# THE SECOND ESTIMATOR: the slope in weight-stream units.
#
# `moe/bench/weights.py` divides a ladder's per-tile slope by a MEASURED time
# instead of by a fitted level. Two kinds of test below and they are kept
# apart. The first kind is arithmetic about the byte count and the refusals.
# The second kind is a REPLAY over
# results/published/2026-09-10-nvidia_h200-gaps-session: it re-fits the
# committed cells and pins the numbers the 2026-09-10 synthesis published, so
# a change to the byte count or to the division has to move a figure a paper
# quotes before it can pass.
# --------------------------------------------------------------------------

#: The card's own calibrated triad rate, as the published cells record it in
#: `prov_bandwidth`. Read from the corpus rather than typed here: the
#: 2026-09-10 calibration moved this card's ridge from 152.8 to 155.9 and its
#: bf16 peak from 668.5 to 682.1, and a test that pins a rate instead of
#: reading the file it came from is stale by construction the next time the
#: card is calibrated.
SESSION = ROOT / "results" / "published" / "2026-09-10-nvidia_h200-gaps-session"


def _corpus_rows(arm: str):
    """`(rows, prov_bandwidth_gbps)` for one arm of the published session."""
    runs = sorted((SESSION / "results" / arm).glob("*/cells.csv"))
    assert len(runs) == 1, f"{arm}: expected one published run, found {runs}"
    with open(runs[0], newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    assert rows, f"{arm}: no ok cells"
    return rows, float(rows[0]["prov_bandwidth"])


def _ols(xs, ys):
    """The same ordinary least squares the sweep's `_line` and
    `analysis/synth/s8_common_currency.py` both fit. Returns (intercept, slope)."""
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    return my - slope * mx, slope


def _ladder_slope_ms(rows, tread_key, keep):
    """The per-M-tile slope of one ladder: OLS through the per-tread MEDIAN of
    `ms_p50` over the repeats, which is s8's recipe and the sweep's line."""
    treads: dict[int, list[float]] = {}
    for r in rows:
        if not keep(r):
            continue
        treads.setdefault(int(float(r[tread_key])), []).append(float(r["ms_p50"]))
    ns = sorted(treads)
    return _ols([float(n) for n in ns], [statistics.median(treads[n]) for n in ns])[1]


def test_the_mixtral_bf16_weight_set_is_the_2_8186_gb_every_w_is_quoted_against():
    """The denominator of every weight-stream figure in the 2026-09-10
    synthesis. `analysis/synth/s8_common_currency.py` reaches it as
    `weight_elements(cfg) * num_experts * 2`; `moe.spec.MoEConfig.weight_bytes`
    reaches it through the two slab shapes; this module is the documented name
    for it. All three must be the same integer, because a study whose two
    halves divide by two denominators cannot compare its own ladders."""
    got = routed_expert_weight_bytes("mixtral-8x7b", "bf16")
    assert got == 2_818_572_288
    assert got / 1e9 == pytest.approx(2.8186, abs=5e-5)
    assert got == MIX.weight_bytes("bf16")
    # s8's own route: 3FH elements per expert, E experts, 2 bytes.
    assert got == 3 * MIX.intermediate_size * MIX.hidden_size * MIX.num_experts * 2


def test_the_weight_set_scales_with_the_dtype_and_nothing_else():
    """Per dtype, as the function's contract says. fp8 halves it, fp32 doubles
    it, and the geometry is untouched."""
    bf16 = routed_expert_weight_bytes("mixtral-8x7b", "bf16")
    assert routed_expert_weight_bytes("mixtral-8x7b", "fp16") == bf16
    assert routed_expert_weight_bytes("mixtral-8x7b", "fp8_e4m3") * 2 == bf16
    assert routed_expert_weight_bytes("mixtral-8x7b", "fp32") == 2 * bf16


def test_a_tp_entry_returns_the_set_one_device_streams():
    """`intermediate_size` is the PER-SHARD width, so a TP=8 entry is an eighth
    of the whole model's expert weights and that is the set the kernel on that
    device re-reads. Reporting the unsharded figure would divide a measured
    slope by eight times the bytes the measured kernel moved."""
    whole = routed_expert_weight_bytes("mixtral-8x7b", "bf16")
    assert routed_expert_weight_bytes("mixtral-8x7b-tp8", "bf16") * 8 == whole


def test_the_whole_layer_is_refused_on_a_model_with_a_shared_expert():
    """THE GEOMETRY THIS MODULE CANNOT RESOLVE. `MoEConfig` carries
    `shared_experts` as a count, not a width: qwen2-57b-a14b's shared expert is
    20480 wide where its routed experts are 2560. Returning the routed set
    under the name `layer_weight_bytes` would understate that layer silently.
    Mixtral has no shared expert, so there the two functions agree exactly."""
    for name in ("qwen2-57b-a14b", "deepseek-v3", "deepseek-v2-lite"):
        with pytest.raises(WeightSetRefused, match="shared expert"):
            layer_weight_bytes(name, "bf16")
    assert (layer_weight_bytes("mixtral-8x7b", "bf16")
            == routed_expert_weight_bytes("mixtral-8x7b", "bf16"))


def test_an_unknown_model_or_dtype_is_refused_by_name():
    """A byte count for a geometry this repository does not hold is a guess,
    and a guessed denominator is the failure mode the whole statistic exists to
    avoid. The message lists what IS known so the caller can pick."""
    with pytest.raises(WeightSetRefused, match="unknown model"):
        routed_expert_weight_bytes("mixtral-8x22b", "bf16")
    with pytest.raises(WeightSetRefused, match="unknown dtype"):
        routed_expert_weight_bytes("mixtral-8x7b", "int4")


def test_an_unverified_geometry_is_refused():
    """`verified` flips per model once the geometry has been checked against
    the upstream config.json. An unverified one multiplies out to a byte count
    that looks exactly as authoritative as a checked one."""
    guess = replace(MIX, name="mixtral-guess", verified=False)
    with pytest.raises(WeightSetRefused, match="not `verified`"):
        routed_expert_weight_bytes(guess, "bf16")


def test_a_bandwidth_of_none_or_zero_is_refused_rather_than_defaulted():
    """THE POINT OF THE STATISTIC IS THAT IT NAMES THE RATE IT WAS DIVIDED BY.
    w scales exactly 1:1 in the rate, so a module that supplied its own would
    be quoting the card's datasheet as the memory branch's achieved bandwidth
    which is the confound of statement (5) of the 2026-09-10 synthesis,
    dressed as a convenience. Zero, negative and NaN are refused for the same reason: they
    are broken calibration reads, not slow cards."""
    for bad in (None, 0.0, -4374.3, math.nan, math.inf):
        with pytest.raises(WeightSetRefused):
            weight_streams_per_tile(0.5, "mixtral-8x7b", "bf16", bad)


def test_w_scales_exactly_one_to_one_in_the_rate_it_names():
    """w rises with the assumed rate, exactly 1:1, because a faster rate makes
    one stream take less time and the same slope is then more streams. The test
    exists so the confound is a property the code demonstrates rather than a
    sentence in a docstring, and so a w quoted without its rate can be seen to
    be meaningless. The DIRECTION is pinned too: the first version of this test
    asserted the reciprocal and the module docstring beside it said "doubling
    the bandwidth halves w", which is the ratio upside down."""
    triad, read = 4374.299702465323, 4612.253362489001
    a = weight_streams_per_tile(0.881234, "mixtral-8x7b", "bf16", triad)
    b = weight_streams_per_tile(0.881234, "mixtral-8x7b", "bf16", read)
    assert b.streams / a.streams == pytest.approx(read / triad, rel=1e-12)
    assert b.streams > a.streams
    half = weight_streams_per_tile(0.881234, "mixtral-8x7b", "bf16", triad / 2)
    assert half.streams == pytest.approx(a.streams / 2, rel=1e-12)
    assert a.bandwidth_gbps == triad and "4374" in a.render()


def test_a_stream_of_the_mixtral_weight_set_is_0_6443_ms_at_the_triad_rate():
    """The denominator the synthesis quotes, to the place it quotes it, at the
    rate the published cells record."""
    _, bw = _corpus_rows("bn_decomposition")
    assert weight_stream_ms("mixtral-8x7b", "bf16", bw) == pytest.approx(
        0.6443, abs=5e-5)


def test_a_negative_slope_is_labelled_and_not_refused():
    """A ladder whose fitted slope FALLS with another M-tile has said something
    about its branch membership, and dropping it deletes that. `descending` is
    the label, `render` says on the line that the number is not a fraction of a
    stream, and the sweep prints the negative. A non-finite slope is a
    different thing, a degenerate fit rather than a measurement, and is
    refused."""
    w = weight_streams_per_tile(-0.6422, "mixtral-8x7b", "bf16", 4374.3)
    assert w.streams < 0 and w.descending
    assert "NEGATIVE" in w.render()
    assert not weight_streams_per_tile(0.5, "mixtral-8x7b", "bf16",
                                       4374.3).descending
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(WeightSetRefused, match="not finite"):
            weight_streams_per_tile(bad, "mixtral-8x7b", "bf16", 4374.3)


def test_the_model_says_w_is_alpha_b_plus_phi_in_this_gemms_own_weight_unit():
    """`ai_model.slope_weight_streams` is the three-term model's prediction for
    the quantity `weights.weight_streams_per_tile` measures, and the unit it is
    in is ONE FULL READ OF THIS GEMM'S B OPERAND, not the layer's expert set.
    Both halves are pinned: the identity, and the conversion factor between the
    two units, which on mixtral's up-projection is exactly 1.5E = 12. A
    docstring that claimed the two were the same number would be out by that
    factor."""
    for bm, bn, aa in ((64, 64, 0.0), (64, 64, 1.0), (128, 256, 0.143)):
        got = slope_weight_streams(N, K, block_m=bm, block_n=bn,
                                   alpha_b=AUDIT_ALPHA_B, alpha_a=aa)
        want = AUDIT_ALPHA_B + phi(N, K, block_m=bm, block_n=bn, alpha_a=aa)
        assert got == pytest.approx(want, rel=1e-12)
        # phi >= 0, so the statistic is an UPPER bound on the miss fraction.
        assert got >= AUDIT_ALPHA_B
    assert (routed_expert_weight_bytes("mixtral-8x7b", "bf16") / (K * N * 2)
            == pytest.approx(1.5 * MIX.num_experts))


def test_exact_cap_takes_its_denominator_from_slope_weight_streams():
    """One definition of `alpha_b + phi`, two readers. The cap and the
    weight-stream statistic cannot come to describe two different slopes."""
    for bm, bn, ab, aa in ((64, 64, 0.31, 0.14), (128, 256, 0.9, 0.5)):
        slope = slope_weight_streams(N, K, block_m=bm, block_n=bn,
                                     alpha_b=ab, alpha_a=aa)
        assert exact_cap(N, K, block_m=bm, block_n=bn, alpha_b=ab,
                         alpha_a=aa) == pytest.approx(2.0 * bm / (2 * slope),
                                                      rel=1e-12)


# --- the replay over the published session ------------------------------

#: The weight-stream slopes the 2026-09-10 synthesis publishes in section 1,
#: at the card's own calibrated triad rate: `(BLOCK_N, BLOCK_M) -> w`. These
#: are the numbers a paper would quote, so they are pinned to the place the
#: synthesis prints them.
BN_G16_W_AT_TRIAD = {(32, 32): 1.254, (32, 64): 1.368,
                     (64, 32): 0.863, (64, 64): 0.897,
                     (128, 32): 0.683, (128, 64): 0.731}


def test_replay_bn_g16_weight_stream_slopes_from_the_published_cells():
    """RECOMPUTED FROM results/published/2026-09-10-nvidia_h200-gaps-session,
    not asserted. Every one of the six ladders the BLOCK_N decomposition was
    fitted over, scored on the statistic that has no fitted level in it.

    The shape is the whole refutation of the three-term model's activation
    term: the model's only BLOCK_N-dependent term is proportional to BLOCK_M,
    so the drop from BN=32 to BN=128 must DOUBLE when BLOCK_M doubles. Measured
    here it is 1.254 - 0.683 = 0.571 at BM=32 against 1.368 - 0.731 = 0.637 at
    BM=64, a ratio of 1.115 where the model requires 2.000."""
    rows, bw = _corpus_rows("bn_decomposition")
    stream = weight_stream_ms("mixtral-8x7b", "bf16", bw)
    got = {}
    for (bn, bm), want in BN_G16_W_AT_TRIAD.items():
        slope = _ladder_slope_ms(
            rows, "tiles",
            lambda r, bn=bn, bm=bm: (int(r["block_n"]) == bn
                                     and int(r["block_m"]) == bm))
        got[(bn, bm)] = slope / stream
        assert got[(bn, bm)] == pytest.approx(want, abs=5e-4), (bn, bm)
    drop32 = got[(32, 32)] - got[(128, 32)]
    drop64 = got[(32, 64)] - got[(128, 64)]
    assert drop64 / drop32 == pytest.approx(1.115, abs=5e-3)


def test_replay_cap_test_bm16_g1_ladder_is_one_full_weight_stream_per_tile():
    """cap_test's BLOCK_M=16, GROUP_SIZE_M=1 ladder, over exactly-full treads
    from n=3 up: 1.0514 weight-streams per extra M-tile at the triad rate. The
    per-M-tile cost of this kernel at that schedule is one complete re-read of
    the layer's expert weights, and no part of that sentence went through a
    fitted level, an intercept or an extrapolated fixed cost."""
    rows, bw = _corpus_rows("tile_cap")
    slope = _ladder_slope_ms(
        rows, "tiles_per_expert",
        lambda r: (int(float(r["block_m"])) == 16
                   and float(r["tile_eff"]) >= 1.0
                   and int(float(r["tiles_per_expert"])) >= 3))
    w = weight_streams_per_tile(slope, "mixtral-8x7b", "bf16", bw,
                                bandwidth_source="triad, published cells")
    assert w.streams == pytest.approx(1.0514, abs=5e-5)
    assert w.streams > 1.0
    assert "triad, published cells" in w.render()


def test_the_published_session_ran_at_the_2026_09_10_calibration():
    """The rate the replays divide by is the card's own, as the cells record
    it. Pinned here so that the two replays above cannot quietly start reading
    a different arm's bandwidth: they agree to the last digit because both arms
    ran against one calibration."""
    _, bn_bw = _corpus_rows("bn_decomposition")
    _, cap_bw = _corpus_rows("tile_cap")
    assert bn_bw == cap_bw == pytest.approx(4374.299702465323, rel=1e-12)
