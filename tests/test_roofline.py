import pytest

from moe.bench import roofline as RL


def test_refuses_unverified_hardware():
    with pytest.raises(RL.UnverifiedHardware, match="verified: false"):
        RL.load_hardware("h200_nvl")


def test_error_message_points_at_the_source_and_the_sparsity_trap():
    with pytest.raises(RL.UnverifiedHardware) as e:
        RL.load_hardware("h200_nvl")
    assert "nvidia.com" in str(e.value)
    assert "DENSE" in str(e.value)


def test_missing_hardware_file():
    with pytest.raises(FileNotFoundError):
        RL.load_hardware("gb200_nvl72")


def test_ridge_and_bound_classification():
    hw = RL.load_hardware("h200_nvl", allow_unverified=True)
    ridge = hw.ridge_point("bf16")
    assert ridge > 1.0
    assert hw.bound("bf16", ridge * 0.5) == "memory"
    assert hw.bound("bf16", ridge * 2.0) == "compute"


def test_attainable_is_the_min_of_the_two_roofs():
    hw = RL.load_hardware("h200_nvl", allow_unverified=True)
    low, high = 1.0, 1e6
    assert hw.attainable("bf16", low) == pytest.approx(low * hw.bandwidth_bytes_s)
    assert hw.attainable("bf16", high) == hw.peak("bf16")


def test_unfilled_dtype_peak_is_an_error_not_a_zero():
    hw = RL.load_hardware("h200_nvl", allow_unverified=True)
    with pytest.raises(ValueError, match="no verified peak"):
        hw.peak("fp32")


def test_memory_bound_kernel_can_be_highly_efficient():
    """A kernel at 4% of peak compute but 95% of its roofline is a good kernel.
    The efficiency helper must say so."""
    hw = RL.load_hardware("h200_nvl", allow_unverified=True)
    ai = 2.0  # deep in the memory-bound regime
    achieved = 0.95 * hw.attainable("bf16", ai)
    assert RL.efficiency(hw, "bf16", ai, achieved) == pytest.approx(0.95)
    assert achieved / hw.peak("bf16") < 0.05
