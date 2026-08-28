"""Figures must describe the dtype the rows were measured in.

MEASURED, 2026-08-28. `results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/`
contains 19,908 rows, every one of them fp8_e4m3, and shipped with eight figures
named `*_bf16.png`. `plot.py` takes `--dtype`, defaulting to bf16, and
`run_all.sh` never passes one, so an fp8 sweep rendered bf16 charts from whatever
bf16 rows happened to be lying in the results directory.

Nothing crashed and nothing was labelled wrong -- the filenames say bf16 quite
honestly. The failure is that a reader opening an fp8 result set finds figures
about a different format, which is the same silent-substitution shape this
harness guards elsewhere.

The fix is to plot what is THERE rather than what a default assumes.
"""
from __future__ import annotations

from scripts.plot import dtypes_present


def rows(*dtypes):
    return [{"dtype": d, "scope": "span", "ms_p50": "1.0"} for d in dtypes]


def test_it_finds_the_one_dtype_in_a_single_format_set():
    assert dtypes_present(rows("fp8_e4m3", "fp8_e4m3")) == ["fp8_e4m3"]


def test_a_mixed_set_yields_both():
    got = dtypes_present(rows("bf16", "fp8_e4m3", "bf16"))
    assert got == ["bf16", "fp8_e4m3"]


def test_an_explicit_choice_still_wins():
    """A caller asking for one dtype gets exactly that, even from a mixed set."""
    assert dtypes_present(rows("bf16", "fp8_e4m3"), requested="bf16") == ["bf16"]


def test_asking_for_a_dtype_that_is_not_there_yields_nothing():
    """Better an empty list than eight figures built from zero rows, which is
    how a bf16 default produced charts for an all-fp8 sweep."""
    assert dtypes_present(rows("fp8_e4m3"), requested="bf16") == []


def test_rows_without_a_dtype_column_are_ignored():
    assert dtypes_present([{"scope": "span"}, {"dtype": "bf16"}]) == ["bf16"]


def test_the_order_is_stable():
    """Figure filenames derive from this, so it must not depend on set order."""
    a = dtypes_present(rows("fp8_e4m3", "bf16", "fp16"))
    b = dtypes_present(rows("bf16", "fp16", "fp8_e4m3"))
    assert a == b
