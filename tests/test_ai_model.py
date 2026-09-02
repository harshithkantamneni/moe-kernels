"""The AI model with BOTH operand re-reads, and the ceilings it implies.

WHAT THIS IS AND IS NOT. docs/FINDINGS.md draws its roofline from
AI(r) = (2r/b)/Q(r), which names only the weight re-read. This module writes out
all three terms -- weight re-read, activation re-read, output write -- and
shows they are the SAME formula once you know what the fitted quantity is:

    alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K
    2*BM/(alpha_fitted*b)  ==  2/(b*(alpha_b/BM + alpha_a/BN + 1/K))   exactly

So every cap the study published is arithmetically correct. The error was one
of NAMING: alpha_fitted was called an L2 miss fraction, compared against
published miss fractions, and its drift with BLOCK_M called unexplainable when
a blend of three terms containing BM twice must drift. The value of the
decomposition is mechanism and prior-art comparison -- which knob moves which
term, and that alpha_b lands on TEMPO's b2/b -- not a correction to the cap.
"""
from __future__ import annotations

import pytest

from moe.bench.ai_model import (
    AIModelRefused,
    arithmetic_intensity,
    cap,
    decompose_fitted_alpha,
    ideal_intensity,
    traffic,
)
from moe.spec import MODEL_CONFIGS

MIX = MODEL_CONFIGS["mixtral-8x7b"]
K = MIX.hidden_size                 # 4096
N = 2 * MIX.intermediate_size       # 28672, gate and up


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
    calculation does not approach is a different formula, not a shortcut."""
    for bm, bn, ab, aa in ((64, 64, 0.31, 0.14), (128, 256, 0.31, 0.14),
                           (16, 64, 0.9, 0.5)):
        c = cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa)
        big = arithmetic_intensity(4_000_000, N, K, block_m=bm, block_n=bn,
                                   alpha_b=ab, alpha_a=aa)
        assert big == pytest.approx(c, rel=0.02)
        assert big < c, "AI approaches the cap from below and never exceeds it"


def test_plugging_a_component_where_the_blend_belongs_overstates_the_cap():
    """THE NAMING ERROR, made concrete. The two-term form wants alpha_fitted.
    Hand it alpha_b instead -- a component -- and at BLOCK_M = BLOCK_N, where
    the activation term equals the weight term, the cap comes out 2x too high.
    That is not a defect in the formula; it is what happens when a component is
    read as the whole. The error grows as BLOCK_N shrinks against BLOCK_M."""
    component_in_blend_slot = 2 * 64 / (0.95 * 2)       # alpha_b where alpha_fitted goes
    full = cap(N, K, block_m=64, block_n=64, alpha_b=0.95, alpha_a=0.95)
    assert full < component_in_blend_slot / 1.9
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


def test_a_fitted_alpha_decomposes_consistently_across_block_m():
    """alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K.

    scripts/block_m_crossing_sweep.py:144 records ALPHA_BY_BLOCK_M =
    {64: 0.466, 128: 0.625} and calls the drift unexplained. Solving those two
    points for the two unknowns gives alpha_a = 0.143 and alpha_b = 0.307, and
    the SAME alpha_b must come back from both block sizes.

    This is two equations in two unknowns, so agreement is by construction and
    is NOT evidence. It is recorded because the decisive test follows from it:
    alpha_fitted must be linear in 1/BN at fixed BM, with slope alpha_a*BM.
    """
    alpha_a = 0.143
    b64 = decompose_fitted_alpha(0.466, block_m=64, block_n=64, K=K, alpha_a=alpha_a)
    b128 = decompose_fitted_alpha(0.625, block_m=128, block_n=64, K=K, alpha_a=alpha_a)
    assert b64 == pytest.approx(b128, abs=0.002)
    assert b64 == pytest.approx(0.307, abs=0.005)


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


def test_the_two_term_cap_with_a_fitted_alpha_is_exactly_the_three_term_cap():
    """THE RESULT THAT MAKES EVERY PUBLISHED CAP NUMBER CORRECT.

    The study computes cap = 2*BM/(alpha*b) and the corrected model says
    cap = 2/(b*(alpha_b/BM + alpha_a/BN + 1/K)). These looked like different
    formulas, and for two hours on 2026-09-02 the second was described as a
    correction to the first. It is not. What a ladder fit returns is

        alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K

    and substituting that into the two-term form and dividing through by BM
    reproduces the three-term form EXACTLY. The "missing" terms were never
    missing; they were inside the fitted quantity in the right proportion,
    because the fit divides every per-tile cost by the same weight bytes.

    So the published caps stand. What was wrong was naming: alpha_fitted was
    called an L2 miss fraction, compared against published miss fractions, and
    its drift with BLOCK_M called unexplainable when it must drift. This test
    pins the equivalence so that distinction cannot be lost again.
    """
    b = 2
    for bm, bn, ab, aa in ((32, 64, 0.307, 0.143), (64, 64, 0.307, 0.143),
                           (128, 64, 0.307, 0.143), (128, 256, 0.307, 0.143),
                           (64, 256, 0.5, 0.2), (256, 32, 0.9, 0.9)):
        alpha_fitted = ab + aa * (bm / bn) + bm / K
        two_term = 2 * bm / (alpha_fitted * b)
        three_term = cap(N, K, block_m=bm, block_n=bn, alpha_b=ab, alpha_a=aa, b=b)
        assert two_term == pytest.approx(three_term, rel=1e-12), (
            f"BM={bm} BN={bn}: two-term {two_term} vs three-term {three_term}")


def test_alpha_fitted_is_not_bounded_by_one_even_though_its_components_are():
    """A blend of three terms can exceed 1.0; a miss fraction cannot. Fits above
    1.0 were read as impossible during the study. They are not impossible for
    the quantity actually being reported -- they are a large BM/BN ratio."""
    ab, aa = 0.9, 0.9                       # both legal miss fractions
    blend = ab + aa * (256 / 64) + 256 / K   # BM=256, BN=64
    assert blend > 1.0
    # and the cap computed from it is still finite, positive and correct
    assert 2 * 256 / (blend * 2) == pytest.approx(
        cap(N, K, block_m=256, block_n=64, alpha_b=ab, alpha_a=aa), rel=1e-12)
