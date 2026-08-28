"""Settling the clock for the wrong workload is the same as not settling.

MEASURED, H200 SXM 2026-08-27. `calibrate()` settled once under a dense bf16
matmul, converged at 1470 MHz and 64 C, and then ran the bandwidth patterns at
1980 MHz and 52 C. The chip COOLED and BOOSTED, because streaming memory draws
less power than saturating tensor cores, so the two workloads have different
steady states. The settle was real; it was just the steady state of the wrong
load.

That is one of two reasons the recorded `read` ceiling came out at 4389.4 GB/s
when the same `torch.sum(dim=0)` call measures 4463.0 fresh, and it makes every
percent-of-ceiling figure in the study pessimistic.

The convergence rule is pure and tested here. Which load induces the plateau is
a separate decision, tested only to the extent that it can be off-GPU.
"""
from __future__ import annotations

import pytest

from moe.bench.calibrate import SETTLE_LOADS, clocks_settled


def test_it_needs_three_samples_before_deciding():
    """Two points can agree by accident on a ramp; three is the minimum that
    says anything about a plateau."""
    assert not clocks_settled([], 2.0)
    assert not clocks_settled([1980], 2.0)
    assert not clocks_settled([1980, 1980], 2.0)
    assert clocks_settled([1980, 1980, 1980], 2.0)


def test_a_flat_plateau_settles():
    assert clocks_settled([1470, 1980, 1975, 1980, 1980], 2.0)


def test_a_climbing_ramp_does_not():
    """The failure this whole function exists to prevent."""
    assert not clocks_settled([840, 1200, 1600, 1980], 2.0)


def test_only_the_last_three_count():
    """Early ramp samples must not hold a settled reading hostage."""
    assert clocks_settled([840, 1200, 1975, 1980, 1978], 2.0)


def test_the_tolerance_is_respected_in_both_directions():
    # 1980 -> 1900 is 4.0% of the max, outside a 2% tolerance and inside 5%.
    assert not clocks_settled([1980, 1900, 1980], 2.0)
    assert clocks_settled([1980, 1900, 1980], 5.0)


def test_a_zero_reading_is_not_treated_as_settled():
    """nvidia-smi can return 0 when the query fails. Three zeros have a spread
    of zero and would otherwise look like the most settled clock imaginable."""
    assert not clocks_settled([0, 0, 0], 2.0)
    assert not clocks_settled([1980, 0, 1980], 2.0)


def test_both_workload_kinds_exist_and_differ():
    """Bandwidth must settle under a memory load and the GEMM under a compute
    one, because their steady states differ by 500 MHz on this hardware."""
    assert set(SETTLE_LOADS) == {"compute", "memory"}
    assert SETTLE_LOADS["compute"] is not SETTLE_LOADS["memory"]


def test_an_unknown_workload_is_refused():
    from moe.bench.calibrate import settle_clocks
    with pytest.raises(KeyError):
        settle_clocks(load="magnetic-tape", max_seconds=0.0)


def test_the_dense_bf16_rate_reproduces_each_vendor_headline():
    """The per-SM-per-clock constants are pinned by reproducing the published
    figure from SM count and boost clock, so a wrong entry is caught here rather
    than silently normalising a ceiling against the wrong silicon.

    sm_80 was added when the A100 run found the table had only sm_90, which made
    `sustained_peak_tflops` return None and failed a test on hardware that was
    working correctly."""
    from moe.bench.calibrate import _DENSE_BF16_FLOP_PER_SM_CLK as TBL

    # (capability, SMs, boost MHz, published dense BF16 TFLOP/s)
    published = [
        ((9, 0), 132, 1830, 989.4),    # H200 SXM
        ((8, 0), 108, 1410, 312.0),    # A100 SXM4-80GB
    ]
    for cap, sms, mhz, headline in published:
        per_clk = TBL.get(cap)
        assert per_clk is not None, f"no constant for sm_{cap[0]}{cap[1]}"
        implied = sms * per_clk * mhz * 1e6 / 1e12
        assert implied == pytest.approx(headline, rel=0.01), (
            f"sm_{cap[0]}{cap[1]}: {sms} SM x {per_clk} x {mhz} MHz = "
            f"{implied:.1f} TFLOP/s, published {headline}")


def test_ampere_does_half_of_hopper_per_sm_per_clock():
    """Not a coincidence worth losing: it is the generational tensor-core step,
    and a table entry that broke it would be a typo."""
    from moe.bench.calibrate import _DENSE_BF16_FLOP_PER_SM_CLK as TBL
    assert TBL[(9, 0)] == 2 * TBL[(8, 0)]


@pytest.mark.parametrize("name,clk_mhz,published_gbps", [
    ("NVIDIA H200", 3201, 4800),                # spec 4.8 TB/s, already derated
    ("NVIDIA A100-SXM4-80GB", 1593, 2039),      # spec 2039 GB/s
    ("NVIDIA H100 80GB HBM3", 2619, 3350),      # spec 3.35 TB/s
])
def test_the_pin_rate_reproduces_each_vendor_bandwidth(name, clk_mhz, published_gbps):
    """The pin rate is the only hard physical bound in the calibration, and
    every "nothing exceeded it" argument rests on it. It hardcoded 6144 bits for
    every device, which is the H200's bus, so an A100 was reported at 2446.8
    GB/s instead of 2038.8: 20% high, on the one number that cannot be wrong.

    Each width is checked by reproducing the vendor figure. The derived rate
    should sit at or slightly above the published one, since published numbers
    are already derated.
    """
    from scripts.calibrate_hardware import _memory_bus_bits

    bits = _memory_bus_bits(name)
    assert bits is not None, f"no bus width for {name!r}"
    derived = clk_mhz * 2 * bits / 8 / 1000
    assert published_gbps <= derived <= published_gbps * 1.05, (
        f"{name}: {clk_mhz} MHz x 2 x {bits} bits / 8 = {derived:.1f} GB/s "
        f"against a published {published_gbps}")


def test_an_unknown_device_gets_no_pin_rate_rather_than_a_guess():
    """Returning the H200's width for an unrecognised card is how the A100 bug
    happened. None is the correct answer, and the caller records that."""
    from scripts.calibrate_hardware import _memory_bus_bits
    assert _memory_bus_bits("NVIDIA GeForce RTX 5090") is None
    assert _memory_bus_bits("") is None
