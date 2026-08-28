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
