"""The AI model with BOTH operand re-reads, and the ceilings it implies.

The defect being fixed: docs/FINDINGS.md draws its roofline from
AI(r) = (2r/b)/Q(r), which counts the weight re-read and drops the activation
re-read and the output write entirely. At the sweep's own pinned BLOCK_N=64 with
BLOCK_M=64 the dropped activation term is the SAME SIZE as the one kept.
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


def test_the_activation_term_is_not_negligible_at_the_swept_tiles():
    """The reason this module exists. At BLOCK_M = BLOCK_N the two re-read costs
    are EQUAL, so the study's weights-only cap is 2x too high there -- and the
    error grows as BLOCK_N shrinks relative to BLOCK_M."""
    weights_only = 2 * 64 / (0.95 * 2)                  # the published form
    full = cap(N, K, block_m=64, block_n=64, alpha_b=0.95, alpha_a=0.95)
    assert full < weights_only / 1.9
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
