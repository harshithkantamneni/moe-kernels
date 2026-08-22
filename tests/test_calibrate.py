"""The counter-free substitutes for Nsight Compute.

Nsight needs GPU performance counters, which need a host permission a rented pod
does not grant. These are the inferences that replace a direct DRAM-traffic
measurement; the arithmetic is testable on CPU, the measurement itself is in
tests/test_gpu.py.
"""
import pytest

from moe.bench.calibrate import implied_traffic_ratio, l2_absorbed_bytes

BW = 4.0e12   # 4 TB/s achievable


def test_a_kernel_moving_exactly_the_minimum_scores_one():
    compulsory = 4.0e9          # 4 GB
    ms = compulsory / BW * 1e3  # exactly the roofline minimum
    assert implied_traffic_ratio(compulsory, ms, BW) == pytest.approx(1.0)


def test_re_reading_every_tile_twice_scores_two():
    compulsory = 4.0e9
    ms = 2 * compulsory / BW * 1e3
    assert implied_traffic_ratio(compulsory, ms, BW) == pytest.approx(2.0)


def test_ratio_is_monotonic_in_measured_time():
    c = 1.0e9
    ratios = [implied_traffic_ratio(c, ms, BW) for ms in (0.25, 0.5, 1.0, 2.0)]
    assert ratios == sorted(ratios)


def test_faster_than_the_compulsory_minimum_reads_below_one():
    """Below 1.0 means the compulsory model over-counted, usually because the
    working set was cache resident. Not an error; a signal to read the row's
    l2_flush column."""
    c = 4.0e9
    assert implied_traffic_ratio(c, 0.5 * c / BW * 1e3, BW) == pytest.approx(0.5)


@pytest.mark.parametrize("args", [(0, 1.0, BW), (1e9, 0, BW), (1e9, 1.0, 0)])
def test_degenerate_inputs_return_zero_not_infinity(args):
    assert implied_traffic_ratio(*args) == 0.0


# --- L2 absorption from the flush axis --------------------------------------

def test_l2_absorption_is_the_flush_delta_times_bandwidth():
    absorbed = l2_absorbed_bytes(ms_flushed=1.5, ms_warm=1.0, achieved_bw_bytes_s=BW)
    assert absorbed == pytest.approx(0.5e-3 * BW)


def test_no_absorption_when_flushing_does_not_slow_the_kernel():
    """A kernel whose working set never fit L2 is unaffected by the flush."""
    assert l2_absorbed_bytes(1.0, 1.0, BW) == 0.0


def test_a_faster_flushed_run_is_clamped_to_zero_not_negative():
    """Noise can put the flushed run marginally ahead; negative absorbed bytes
    would be meaningless."""
    assert l2_absorbed_bytes(0.99, 1.0, BW) == 0.0
