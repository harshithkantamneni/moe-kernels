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


def test_measured_resolves_against_the_rows_being_plotted(tmp_path):
    """"measured" stopped being a filename when calibrations went per-device,
    and every caller passing that literal broke. plot.py died on it AFTER a
    three-hour sweep, with all the rows already on disk.

    Resolution has to use the device the ROWS were measured on, not the device
    running the plot: figures get drawn on a laptop from a committed CSV."""
    (tmp_path / "measured_nvidia_h200.yaml").write_text(MEASURED_H200)
    rows = [{"gpu_name": "NVIDIA H200", "impl": "x"}]
    hw = RL.hardware_for_rows("measured", rows, directory=tmp_path)
    assert "H200" in hw.name


def test_a_named_profile_still_loads_directly(tmp_path):
    (tmp_path / "h200_sxm.yaml").write_text(
        "name: NVIDIA H200 SXM\nverified: true\nsource: nvidia.com\ntdp_w: 700\n"
        "memory:\n  bandwidth_tb_s: 4.8\ncompute_dense_tflops:\n  bf16: 989.5\n")
    hw = RL.hardware_for_rows("h200_sxm", [], directory=tmp_path)
    assert hw.peak("bf16") == 989.5e12


def test_missing_measured_calibration_names_the_device(tmp_path):
    rows = [{"gpu_name": "NVIDIA A100-SXM4-80GB"}]
    with pytest.raises(FileNotFoundError, match="A100"):
        RL.hardware_for_rows("measured", rows, directory=tmp_path)


# --- the reference clock is per GEMM, and the per-row roof rule lives here ---

def test_a_dtype_maps_to_the_gemm_whose_roof_it_is_scored_against():
    """Two GEMMs, two clocks. Every fp8 weight format is the fp8 GEMM's;
    everything else the bf16 one, which is also fp16's roof in the yaml."""
    for dtype in ("fp8_e4m3", "fp8_e5m2"):
        assert RL.reference_family(dtype) == RL.FP8_FAMILY
    for dtype in ("bf16", "fp16", "fp32", ""):
        assert RL.reference_family(dtype) == RL.BF16_FAMILY
    assert set(RL.CLOCK_FIELDS_BY_FAMILY) == set(RL.REFERENCE_FAMILIES)


def test_the_committed_h200_resolves_a_different_clock_per_family():
    """The file the pod will read: bf16 GEMM at 1515, fp8 GEMM at 1905, both
    the idle scalar today. One number for the two was the defect: an fp8 cell
    at 1905 is AT its roof's clock and was filed LEVEL-failed HIGH."""
    bf16 = RL.reference_clock("NVIDIA H200")
    fp8 = RL.reference_clock("NVIDIA H200", family=RL.FP8_FAMILY)
    assert (bf16.mhz, bf16.family) == (1515.0, RL.BF16_FAMILY)
    assert (fp8.mhz, fp8.family) == (1905.0, RL.FP8_FAMILY)
    assert "fp8_gemm_clock_mhz" in fp8.source and "fp8 GEMM" in fp8.source
    assert "gemm_clock_mhz" in bf16.source and "(bf16 GEMM)" in bf16.source
    assert bf16.grade == fp8.grade == RL.REFERENCE_IDLE_SCALAR
    assert not fp8.usable_for_roof
    # THE FAIL BRANCH: the A100 has no fp8 tensor cores, its calibration wrote
    # fp8_gemm_clock_mhz: 0, and that is "no fp8 reference" with the reason,
    # never the bf16 number in the fp8 family's coat.
    a100 = RL.reference_clock("NVIDIA A100-SXM4-80GB", family=RL.FP8_FAMILY)
    assert a100.mhz is None and a100.family == RL.FP8_FAMILY
    assert "no fp8 GEMM clock" in a100.source
    assert RL.reference_clock("NVIDIA A100-SXM4-80GB").mhz == 1335.0
    with pytest.raises(ValueError, match="not a reference family"):
        RL.reference_clock("NVIDIA H200", family="fp8_e4m3")


def test_the_fp8_family_has_no_settle_plateau_fallback(tmp_path):
    """The settle ran under the bf16 GEMM; the committed H200 puts the fp8 GEMM
    435 MHz above its plateau. A file with the plateau and no fp8 field has no
    fp8 reference, while the bf16 family may still fall back to it."""
    import yaml

    card = "NVIDIA H200"
    (tmp_path / f"{RL.measured_slug(card)}.yaml").write_text(yaml.safe_dump({
        "name": "NVIDIA H200 (measured)", "verified": True,
        "memory": {"bandwidth_tb_s": 4.37},
        "compute_dense_tflops": {"bf16": 700.0, "fp8_e4m3": 1400.0},
        "detail": {"gpu_name": card, "settle": {"final_mhz": 1470}}}))
    bf16 = RL.reference_clock(card, directory=tmp_path)
    assert bf16.mhz == 1470.0 and bf16.grade == RL.REFERENCE_SETTLE_PLATEAU
    fp8 = RL.reference_clock(card, directory=tmp_path, family=RL.FP8_FAMILY)
    assert fp8.mhz is None and "no fp8 GEMM clock" in fp8.source


def test_a_measured_profile_carries_its_reference_clocks_and_a_datasheet_none():
    """`Hardware` used to drop everything under `detail`, which is how the
    recompute came to rebuild the fixed roof without the clock the per-row
    roof needs. Read from the SAME file as the peaks, by the same walk the
    driver resolves with."""
    measured = RL.load_hardware("measured_nvidia_h200")
    assert measured.reference_clocks[RL.BF16_FAMILY].mhz == 1515.0
    assert measured.reference_clocks[RL.FP8_FAMILY].mhz == 1905.0
    assert measured.reference_clocks[RL.BF16_FAMILY] == RL.reference_clock("NVIDIA H200")
    a100 = RL.load_hardware("measured_nvidia_a100_sxm4_80gb")
    assert set(a100.reference_clocks) == {RL.BF16_FAMILY}
    assert RL.load_hardware("h200_nvl", allow_unverified=True).reference_clocks == {}


def test_the_per_row_roof_rule_can_pass_and_can_refuse_by_name():
    """ONE STATEMENT, TWO WRITERS. `driver._apply_cost` and
    `recompute.ceiling_columns` both call this; the pre-fix recompute had no
    rule at all. Each refusal is named so `roof_note` says which."""
    ok = RL.roof_scale_refusal(1980.0, 1515.0, RL.REFERENCE_UNDER_LOAD)
    assert ok == ""
    assert RL.cell_clock_roof(800.0, 1980.0, 1515.0, RL.REFERENCE_UNDER_LOAD) == (
        pytest.approx(800 * 1980 / 1515), "")
    assert RL.cell_clock_roof(800.0, 1980.0, 1515.0, RL.REFERENCE_UNDER_LOAD)[0] == (
        pytest.approx(1045.5, abs=0.05))
    # A LOW cell's roof is lower, and written: the roof is not the exclusion.
    assert RL.cell_clock_roof(700.0, 1400.0, 1515.0, RL.REFERENCE_UNDER_LOAD)[0] == (
        pytest.approx(700 * 1400 / 1515))
    for load, ref, grade, word in (
            (0.0, 1515.0, RL.REFERENCE_UNDER_LOAD, "no under-load clock"),
            (None, 1515.0, RL.REFERENCE_UNDER_LOAD, "no under-load clock"),
            (1980.0, None, "", "dtype family"),
            (1980.0, 0.0, "", "dtype family"),
            (1980.0, 1515.0, RL.REFERENCE_IDLE_SCALAR, "idle-scalar"),
            (1980.0, 1515.0, RL.REFERENCE_SETTLE_PLATEAU, "settle-plateau"),
            (1980.0, 1515.0, "caller", "'caller'")):
        why = RL.roof_scale_refusal(load, ref, grade)
        assert word in why, (load, ref, grade, why)
        assert RL.cell_clock_roof(800.0, load, ref, grade) == (0.0, why)


def test_an_fp8_cell_at_its_own_roof_s_clock_is_level_and_roofed_unchanged():
    """The number the one-reference design got wrong, restated as arithmetic:
    the fp8 roof of 1447.7 was measured at 1905 MHz. A cell at 1905 is level
    against 1905 and its roof is 1447.7; against the bf16 GEMM's 1515 it was
    HIGH and `roof_at_clock` gave 1820."""
    from moe.bench import timing as T

    assert T.level_side(1905.0, 1905.0) == ""
    assert T.level_side(1905.0, 1515.0) == T.LEVEL_HIGH
    assert RL.cell_clock_roof(1447.7, 1905.0, 1905.0, RL.REFERENCE_UNDER_LOAD)[0] == (
        pytest.approx(1447.7))
    assert RL.roof_at_clock(1447.7, 1515.0, 1905.0) == pytest.approx(1820.4, abs=0.1)


def test_a_script_with_a_private_walk_gets_each_cell_its_own_reference_from_one_call():
    """THE DROP-IN FOR THE THIRD COPY OF THE WALK. `scripts/dtype_tile_confound.py`
    resolves one bf16 clock and hands it to its fp8 cells; on the committed
    H200 that files an fp8 cell at its own roof's 1905 MHz as LEVEL-failed
    HIGH (1905/1515 = 1.26) and its any-flag noise gate then fails the arm.
    This pins the call it should make instead, per cell, against the file the
    pod reads, with the three cells the fix must plant: fp8 at 1905 level and
    KEPT, bf16 at 1980 HIGH and KEPT (the roof moves, the row does not), bf16
    at 1400 LOW and EXCLUDED. `.mhz` and `.source` are the pair the private
    walk returned, so the swap is a rename, not a rewrite."""
    import yaml

    from moe.bench import timing as T

    raw = yaml.safe_load((RL.HARDWARE_DIR / "measured_nvidia_h200.yaml").read_text())
    by_dtype = {d: RL.reference_clock_from_doc(raw, "NVIDIA H200", RL.reference_family(d))
                for d in ("bf16", "fp8_e4m3")}
    assert (by_dtype["bf16"].mhz, by_dtype["fp8_e4m3"].mhz) == (1515.0, 1905.0)
    for rc in by_dtype.values():
        assert isinstance(rc.mhz, float) and rc.source.startswith(raw["name"])
    # Planted through the instrument's verdict, one cell per row of the rule.
    cells = (("fp8_e4m3", 1905.0, True, "", False),
             ("bf16", 1980.0, False, T.LEVEL_HIGH, False),
             ("bf16", 1400.0, False, T.LEVEL_LOW, True))
    for dtype, mhz, level_ok, side, excluded in cells:
        ref = by_dtype[dtype].mhz
        assert T.clock_flags(mhz, mhz, mhz, ref)[0] is level_ok, (dtype, mhz)
        assert T.level_side(mhz, ref) == side, (dtype, mhz)
        # LOW or DRIFT excludes; HIGH is "use roof_at_cell_clock", never exclusion.
        assert (T.level_side(mhz, ref) == T.LEVEL_LOW) is excluded, (dtype, mhz)
    # THE FAIL BRANCH the private walk takes today: one bf16 clock for all.
    one_clock = by_dtype["bf16"].mhz
    assert T.clock_flags(1905.0, 1905.0, 1905.0, one_clock) == (False, True)
    assert T.level_side(1905.0, one_clock) == T.LEVEL_HIGH



def test_the_decorated_measured_name_slugs_to_the_same_file_as_the_device_name():
    """2026-09-09: bm128_depth handed `load_measured`'s decorated name to the
    slug and looked for measured_nvidia_h200_measured.yaml, which no card
    has, so it refused a calibration that was right there."""
    from moe.bench import roofline as RF
    assert RF.measured_slug("NVIDIA H200 (measured)") == RF.measured_slug("NVIDIA H200")
    assert RF.measured_slug("NVIDIA H200" + RF.MEASURED_DECORATION) == "measured_nvidia_h200"
    assert RF.measured_slug("NVIDIA A100-SXM4-80GB (measured)") == "measured_nvidia_a100_sxm4_80gb"
