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


# --- device-aware calibration -------------------------------------------------
#
# One harness, one calibration file per device. `measured.yaml` was a single
# committed file, so calibrating on a second GPU overwrote the first and any
# later re-plot of the published sweep silently used the wrong roof.

MEASURED_A100 = """
name: NVIDIA A100-SXM4-80GB (measured)
verified: true
source: measured on this machine by scripts/calibrate_hardware.py
memory:
  bandwidth_tb_s: 1.935
compute_dense_tflops:
  bf16: 267.1
"""

MEASURED_H200 = """
name: NVIDIA H200 (measured)
verified: true
source: measured on this machine by scripts/calibrate_hardware.py
memory:
  bandwidth_tb_s: 4.3756
compute_dense_tflops:
  bf16: 729.99
"""


def test_measured_slug_is_a_filesystem_safe_per_device_name():
    assert RL.measured_slug("NVIDIA H200") == "measured_nvidia_h200"
    assert RL.measured_slug("NVIDIA A100-SXM4-80GB") == "measured_nvidia_a100_sxm4_80gb"


def test_measured_prefers_the_file_for_this_device(tmp_path):
    (tmp_path / "measured_nvidia_a100_sxm4_80gb.yaml").write_text(MEASURED_A100)
    (tmp_path / "measured.yaml").write_text(MEASURED_H200)
    hw = RL.load_measured("NVIDIA A100-SXM4-80GB", directory=tmp_path)
    assert hw is not None
    assert "A100" in hw.name, "picked the H200 file for an A100 run"


def test_measured_refuses_a_calibration_from_another_device(tmp_path):
    """The gap that would have tainted a whole sweep: measured.yaml ships with
    H200 ceilings, so an A100 run would have been scored against 4375 GB/s."""
    (tmp_path / "measured.yaml").write_text(MEASURED_H200)
    with pytest.raises(RL.HardwareMismatch, match="A100"):
        RL.load_measured("NVIDIA A100-SXM4-80GB", directory=tmp_path)


def test_measured_accepts_the_bare_file_when_it_matches_this_device(tmp_path):
    (tmp_path / "measured.yaml").write_text(MEASURED_H200)
    hw = RL.load_measured("NVIDIA H200", directory=tmp_path)
    assert hw is not None and "H200" in hw.name


def test_measured_is_none_when_no_calibration_has_been_run(tmp_path):
    assert RL.load_measured("NVIDIA H200", directory=tmp_path) is None


def test_per_device_calibrations_are_not_datasheet_profiles(tmp_path):
    """`for_device` picks a DATASHEET profile. A measured file is this machine's
    own calibration, not a spec sheet, and letting one into that search makes
    every device ambiguous with itself the moment it is calibrated."""
    (tmp_path / "h200_sxm.yaml").write_text(
        "name: NVIDIA H200 SXM\nverified: true\nsource: nvidia.com\n"
        "tdp_w: 700\nmemory:\n  bandwidth_tb_s: 4.8\n"
        "compute_dense_tflops:\n  bf16: 989.5\n")
    (tmp_path / "measured_nvidia_h200.yaml").write_text(MEASURED_H200)
    import moe.bench.roofline as _RL
    old, _RL.HARDWARE_DIR = _RL.HARDWARE_DIR, tmp_path
    try:
        # torch reports an H200 SXM as plain "NVIDIA H200", which is a substring
        # of "NVIDIA H200 (measured)". Without an explicit skip, calibrating the
        # box makes its own datasheet profile unresolvable.
        assert RL.for_device("NVIDIA H200", tdp_w=700.0) == "h200_sxm"
        assert RL.ambiguous_for_device("NVIDIA H200") == []
    finally:
        _RL.HARDWARE_DIR = old


def test_another_devices_calibration_does_not_count_as_this_ones(tmp_path):
    """The contract run_all.sh depends on: a repo carrying measured_<other>.yaml
    must still report 'no calibration' on this device. Returning None rather
    than raising matters, because an absent calibration is a normal first-run
    state while a MISMATCHED one is an error."""
    (tmp_path / "measured_nvidia_h200.yaml").write_text(MEASURED_H200)
    assert RL.load_measured("NVIDIA H100 80GB HBM3", directory=tmp_path) is None
