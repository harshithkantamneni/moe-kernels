"""Measure the fp8 compute roof rather than assuming it is twice bf16.

WHY THIS MATTERS TO THE STUDY. STUDY.md's C2 rests on `ridge_fp8 = 2 x
ridge_bf16`, taken from the datasheet ratio of fp8 to bf16 tensor-core rates.
Every fp8 crossing prediction in it inherits that assumption.

The bf16 calibration already showed why a datasheet ratio is not a measurement:
on this H200 the cuBLAS GEMM settles at ~1455 MHz under sustained dense load,
not the 1830 MHz the 989.5 TFLOP/s headline assumes, and the achieved figure is
701.6 TFLOP/s against a 989.4 datasheet peak. If fp8 attains a different
fraction of ITS peak, the real ridge ratio is not 2.

It also unblocks the fp8 roofline, which currently refuses to draw: the measured
H200 profile has no verified fp8 peak, and `roofline` will not invent one.

fp8 matmul is `torch._scaled_mm`, not `torch.mm`: fp8 tensors carry a scale and
the plain path does not take one.
"""
from __future__ import annotations

import pytest


def test_the_flop_rate_table_knows_fp8_for_hopper_and_ada():
    """Same shape as the bf16 table, pinned by reproducing vendor headlines."""
    from moe.bench.calibrate import _DENSE_FP8_FLOP_PER_SM_CLK as TBL
    # (capability, SMs, boost MHz, published dense FP8 TFLOP/s)
    for cap, sms, mhz, headline in [
        ((9, 0), 132, 1830, 1978.9),     # H200 SXM, 2x its bf16 989.4
    ]:
        per_clk = TBL.get(cap)
        assert per_clk is not None, f"no fp8 constant for sm_{cap[0]}{cap[1]}"
        implied = sms * per_clk * mhz * 1e6 / 1e12
        assert implied == pytest.approx(headline, rel=0.01), (
            f"sm_{cap[0]}{cap[1]}: implied {implied:.1f}, published {headline}")


def test_fp8_is_twice_bf16_per_sm_per_clock():
    """The generational relationship the study currently ASSUMES. Pinned here
    so the assumption is visible, and measured separately so it can be wrong."""
    from moe.bench.calibrate import (
        _DENSE_BF16_FLOP_PER_SM_CLK as BF,
    )
    from moe.bench.calibrate import (
        _DENSE_FP8_FLOP_PER_SM_CLK as FP,
    )
    assert FP[(9, 0)] == 2 * BF[(9, 0)]


def test_ampere_has_no_fp8_entry_at_all():
    """sm_80 has no fp8 tensor cores. An entry would imply a peak that does not
    exist, and sustained_peak_tflops would hand back a number for silicon that
    cannot run the format."""
    from moe.bench.calibrate import _DENSE_FP8_FLOP_PER_SM_CLK as FP
    assert (8, 0) not in FP


def test_sustained_fp8_peak_is_none_where_the_hardware_has_none(monkeypatch):
    import torch

    from moe.bench.calibrate import sustained_peak_tflops_fp8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert sustained_peak_tflops_fp8(1410) is None


def test_sustained_fp8_peak_scales_with_the_clock(monkeypatch):
    """SM count comes from the device too, so it is simulated alongside the
    capability. Patching only `is_available` leaves get_device_properties
    asserting on a machine with no CUDA, which is a test that only runs on a
    pod -- the mistake that stopped an H200 sweep earlier today."""
    import torch

    from moe.bench.calibrate import sustained_peak_tflops_fp8

    class _Props:
        multi_processor_count = 132

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: _Props())
    a, b = sustained_peak_tflops_fp8(1000), sustained_peak_tflops_fp8(2000)
    assert a is not None and b == pytest.approx(2 * a)
    # 132 SMs x 8192 FLOP/clk x 1830 MHz reproduces the H200's 1978.9 headline.
    assert sustained_peak_tflops_fp8(1830) == pytest.approx(1978.9, rel=0.01)


def test_the_measurement_uses_scaled_mm_not_mm():
    """fp8 tensors carry a scale; torch.mm takes none and would refuse or, worse,
    silently take a different path."""
    import inspect

    from moe.bench.calibrate import measure_fp8_gemm
    src = inspect.getsource(measure_fp8_gemm)
    assert "_scaled_mm" in src
    assert "torch.mm(" not in src
