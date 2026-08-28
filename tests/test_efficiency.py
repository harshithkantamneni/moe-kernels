"""Does achieved-versus-peak explain the crossing offset?

THE OPEN ITEM. Measured crossings sit consistently BELOW `2R/b`'s prediction:
0.63x in bf16, 0.71x in fp8, across four models. One multiplicative factor that
stable is structure, not scatter.

THE HYPOTHESIS, and it is falsifiable. A crossing is where AI meets the ridge,
and the ridge is `peak_FLOPS / bandwidth`. Datasheet peaks are not what a kernel
reaches. If it attains fraction `f` of peak FLOPs and `g` of peak bandwidth, the
ridge it actually meets is `(f/g) x nominal`. Since the crossing is proportional
to the ridge:

    measured_crossing / predicted_crossing  ==  effective_ridge / nominal_ridge

The left side is already measured, at 0.63 and 0.71. The right side comes from
columns the sweep has been writing all along. If they agree, the offset is
explained and C2 needs no extra term. If they do not, something else is missing
and the activation-traffic candidate comes back into play.

Both sides of the effective ridge come from the SAME model -- `flops` and
`compulsory_bytes` -- so it is a roofline in the byte model's own units rather
than a mix of measured traffic and modelled work.
"""
from __future__ import annotations

import pytest

from moe.bench.efficiency import Efficiency, efficiency_from_rows


def row(tflops, gbps, ms="1.0"):
    return {"tflops": str(tflops), "compulsory_gbps": str(gbps), "ms_p50": ms}


def test_the_ridge_is_flops_over_bytes_in_consistent_units():
    """1000 TFLOP/s against 1000 GB/s is 1e15 FLOP over 1e12 byte = 1000."""
    eff = efficiency_from_rows([row(1000.0, 1000.0)])
    assert eff.effective_ridge == pytest.approx(1000.0)


def test_it_takes_the_PEAK_of_each_not_the_average():
    """A kernel reaches its FLOP peak at large batch and its bandwidth peak at
    small batch, never both in one cell. Averaging would describe neither."""
    rows = [row(50.0, 4000.0),    # memory-bound end
            row(700.0, 300.0)]    # compute-bound end
    eff = efficiency_from_rows(rows)
    assert eff.peak_tflops == pytest.approx(700.0)
    assert eff.peak_gbps == pytest.approx(4000.0)
    assert eff.effective_ridge == pytest.approx(700.0 / 4000.0 * 1000)


def test_untimed_rows_are_excluded():
    """Same trap as the crossing report: ms_p50 0.0 means the cell never ran,
    and its tflops/gbps are 0.0 defaults, not measurements."""
    good = [row(700.0, 4000.0)]
    with_junk = good + [row(0.0, 0.0, ms="0.0")]
    assert efficiency_from_rows(good) == efficiency_from_rows(with_junk)


def test_rows_with_no_numbers_do_not_crash_it():
    assert efficiency_from_rows([{"tflops": "", "compulsory_gbps": "x",
                                  "ms_p50": "1.0"}]) is None


def test_no_rows_gives_None_rather_than_a_zero_ridge():
    """A zero ridge would predict a crossing of zero and look like a finding."""
    assert efficiency_from_rows([]) is None


def test_the_hypothesis_is_expressible_as_a_ratio():
    """The whole point: effective/nominal must be comparable to the 0.63 that
    measured/predicted gave, or the analysis cannot confirm anything."""
    eff = efficiency_from_rows([row(700.0, 4000.0)])
    nominal = 160.3
    assert eff.ratio_against(nominal) == pytest.approx(
        eff.effective_ridge / nominal)


def test_a_nonpositive_nominal_ridge_is_refused():
    eff = efficiency_from_rows([row(700.0, 4000.0)])
    with pytest.raises(ValueError):
        eff.ratio_against(0.0)


def test_it_counts_what_it_used_so_a_thin_sample_is_visible():
    """A ridge built from three rows and one from three thousand should not
    print identically."""
    eff = efficiency_from_rows([row(700.0, 4000.0), row(650.0, 3900.0)])
    assert isinstance(eff, Efficiency) and eff.n_rows == 2
