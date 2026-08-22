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


# --- more than one part, selected by device rather than hardcoded -----------

def test_sxm_and_nvl_are_distinct_profiles():
    """Same memory, different compute. Using the wrong one moves the ridge
    point by 18% and misstates every efficiency number."""
    sxm = RL.load_hardware("h200_sxm")
    nvl = RL.load_hardware("h200_nvl")
    assert sxm.bandwidth_bytes_s == nvl.bandwidth_bytes_s
    assert sxm.peak("bf16") > nvl.peak("bf16")
    assert sxm.peak("bf16") == pytest.approx(989.5e12)
    assert sxm.ridge_point("bf16") == pytest.approx(206.1, abs=1.0)
    assert nvl.ridge_point("bf16") == pytest.approx(174.1, abs=1.0)


def test_sxm_peaks_are_dense_not_sparsity():
    sxm = RL.load_hardware("h200_sxm")
    assert sxm.peak("bf16") < 1100e12, "1979 would be the sparsity figure"
    assert sxm.peak("fp8_e4m3") == pytest.approx(2 * sxm.peak("bf16"), rel=1e-3)


@pytest.mark.parametrize("device,expected", [
    ("NVIDIA H200 SXM", "h200_sxm"),
    ("NVIDIA H200 NVL", "h200_nvl"),
    ("NVIDIA H100 NVL", None),
    ("NVIDIA A100-SXM4-80GB", None),
])
def test_profile_is_selected_from_the_device_name(device, expected):
    assert RL.for_device(device) == expected


def test_device_mismatch_is_detected():
    """The guard that stops an H200 run being plotted against an H100 roof."""
    sxm = RL.load_hardware("h200_sxm")
    assert RL.device_matches(sxm, "NVIDIA H200 SXM")
    assert not RL.device_matches(sxm, "NVIDIA H100 NVL")
    assert RL.device_matches(sxm, ""), "no gpu_name recorded means nothing to check"


def test_an_ambiguous_device_name_refuses_to_choose():
    """torch reports an H200 SXM as plain "NVIDIA H200", which is a substring of
    both H200 profiles. Picking the first match chose NVL and would have
    understated every efficiency number by 18%."""
    assert RL.for_device("NVIDIA H200") is None
    assert RL.ambiguous_for_device("NVIDIA H200") == ["h200_nvl", "h200_sxm"]


def test_an_unambiguous_name_still_resolves():
    assert RL.for_device("NVIDIA H200 SXM") == "h200_sxm"
    assert RL.for_device("NVIDIA H200 NVL") == "h200_nvl"
    assert RL.ambiguous_for_device("NVIDIA H200 SXM") == []


def test_no_match_is_distinguishable_from_ambiguity():
    assert RL.for_device("NVIDIA H100 NVL") is None
    assert RL.ambiguous_for_device("NVIDIA H100 NVL") == []


def test_power_limit_breaks_the_h200_name_tie():
    """torch calls both parts "NVIDIA H200". Board power does not: SXM is
    700 W, NVL is 600 W. One nvidia-smi field resolves what the name cannot."""
    assert RL.for_device("NVIDIA H200") is None
    assert RL.for_device("NVIDIA H200", tdp_w=700.0) == "h200_sxm"
    assert RL.for_device("NVIDIA H200", tdp_w=600.0) == "h200_nvl"


def test_an_unrecognised_power_limit_still_refuses():
    assert RL.for_device("NVIDIA H200", tdp_w=350.0) is None


def test_profiles_declare_their_tdp():
    assert RL.load_hardware("h200_sxm").tdp_w == 700
    assert RL.load_hardware("h200_nvl").tdp_w == 600


def test_the_datasheet_figure_is_below_the_derived_pin_rate():
    """Measured on the device: clocks.max.memory 3201 MHz, so the pin rate is
    3201 x 2 (DDR) x 6144 / 8 = 4916.7 GB/s. NVIDIA publishes 4.8 TB/s, which is
    2.4% lower, so the datasheet number is already derated. Any published
    percentage must say which denominator it used."""
    pin = 3201 * 2 * 6144 / 8 / 1000
    assert pin == pytest.approx(4916.7, abs=0.1)
    spec = RL.load_hardware("h200_sxm").bandwidth_bytes_s / 1e9
    assert spec < pin
    assert 100 * spec / pin == pytest.approx(97.6, abs=0.2)
