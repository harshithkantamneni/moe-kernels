import pytest

from moe.bench import roofline as RL

UNVERIFIED_YAML = """
name: Fake Part
verified: false
source: https://www.nvidia.com/en-us/data-center/h200/
memory:
  bandwidth_tb_s: 4.8
compute_dense_tflops:
  bf16: 835.5
"""


@pytest.fixture
def unverified_dir(tmp_path):
    (tmp_path / "fake_part.yaml").write_text(UNVERIFIED_YAML)
    return tmp_path


def test_refuses_unverified_hardware(unverified_dir):
    with pytest.raises(RL.UnverifiedHardware, match="verified: false"):
        RL.load_hardware("fake_part", directory=unverified_dir)


def test_error_message_points_at_the_source_and_the_sparsity_trap(unverified_dir):
    with pytest.raises(RL.UnverifiedHardware) as e:
        RL.load_hardware("fake_part", directory=unverified_dir)
    assert "nvidia.com" in str(e.value)
    assert "DENSE" in str(e.value)


def test_escape_hatch_loads_an_unverified_file(unverified_dir):
    hw = RL.load_hardware("fake_part", allow_unverified=True, directory=unverified_dir)
    assert hw.name == "Fake Part"


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


# --- now that the file is verified against the datasheet --------------------

def test_verified_hardware_loads_without_the_escape_hatch():
    hw = RL.load_hardware("h200_nvl")
    assert hw.bandwidth_bytes_s == pytest.approx(4.8e12)
    assert hw.peak("bf16") == pytest.approx(835.5e12)


def test_dense_not_sparse_peaks():
    """NVIDIA publishes 1671 TFLOP/s BF16 for H200 NVL *with sparsity*. Using
    that as a roofline roof would halve every reported efficiency."""
    hw = RL.load_hardware("h200_nvl")
    assert hw.peak("bf16") < 1000e12, "looks like a sparsity figure crept in"
    assert hw.peak("fp8_e4m3") == pytest.approx(2 * hw.peak("bf16"), rel=1e-3)


def test_not_confused_with_the_sxm_part():
    """H200 NVL and SXM share bandwidth but not compute."""
    hw = RL.load_hardware("h200_nvl")
    assert hw.peak("bf16") != pytest.approx(989.5e12)


def test_h200_ridge_point_is_in_the_expected_range():
    hw = RL.load_hardware("h200_nvl")
    ridge = hw.ridge_point("bf16")
    assert 150 < ridge < 200, f"unexpected ridge point {ridge:.1f} FLOP/byte"
