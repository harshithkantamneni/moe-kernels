"""Two gates that were not evidence, and a ridge that belonged to neither card.

WHAT WAS MEASURED, on the committed reports, before any of this was written:

  * GATE 3 WAS CIRCULAR. Its `measured` field was `1.0 + alpha_hat`, so the
    "measured crossing ratio" was an algebraic restatement of a fitted alpha
    imported from a DIFFERENT BLOCK_M. Recomputed over every published report
    that carries both fields, `measured == 1 + alpha_measured` in 22 of 22 and
    differs in 0. The serializer kept only {number, claim, verdict, measured,
    gate}, so the published JSON asserted a ratio with no provenance at all.
    And across all of `results/published` the ladder field `crosses` is False
    41 times, null 61 times and True ZERO times: no crossing has ever been
    observed by this study, so no gate may be phrased as though one was.

  * GATE 4 COMPARED DIFFERENT UNITS. Its verdict was `top > 0.85` where `top`
    is a fraction of the ARM'S OWN measured plateau, while the 0.85's stated
    rationale (cap/ridge = 0.716) is a fraction of PEAK COMPUTE. Across the 26
    published reports the plateau is 46.5-75.6% of that card's own
    `ridge x bandwidth`, so the two denominators differ by up to a factor of
    two; in the gate's own units the model ceiling for the arm that "failed" it
    is 1.42. Both FAILs it ever produced are qwen2 at GROUP_SIZE_M=64, and the
    same config scored 0.871 FAIL on one run and 0.841 PASS on another.

  * THE RIDGE BELONGED TO NEITHER CARD. `--ridge` defaulted to `RIDGE_BAND[0]`
    and `scripts/cross_card_surface.sh` never passed it, so all 26 published
    reports -- the 7 A100 ones included -- carry ridge=160.3 and
    ridge_band=[160.3, 176.2]. The A100's own contemporaneous calibration puts
    its ridge at 262.371/1.79936 = 145.8 and the H200's at 712.259/4.37476 =
    162.8. Every `ridge x bandwidth` printed on the A100 was a hybrid of two
    machines, and nothing in the output said so.

Everything here runs off GPU, against the committed calibration yamls and
against cells generated from the model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BM = _load_script()

from moe.bench import roofline  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (32, 64, 128, 256)
RIDGE = 160.3
RIDGE_BAND = (160.3, 176.2)
BANDWIDTH = 4374.5
ROOF = RIDGE * BANDWIDTH * 1e9 / 1e12        # ridge x bandwidth, TFLOP/s
REFIT = 0.558
RETRACTED = 0.10

A100_YAML = ROOT / "moe" / "bench" / "hardware" / "measured_nvidia_a100_sxm4_80gb.yaml"
H200_YAML = ROOT / "moe" / "bench" / "hardware" / "measured_nvidia_h200.yaml"


def cells_at(alpha: float, *, slowdown: float = 1.0, r_max: int = 1024,
             tiles=TILES):
    """Cells from the model, optionally scaled to a slower machine.

    `slowdown` multiplies every time, which divides every absolute throughput
    and leaves every ARM-RELATIVE fraction untouched. That is exactly the state
    the published reports are in -- a plateau at half the roof, with the
    plateau-relative ladder unchanged -- and it is what separates the two
    denominators in a test.
    """
    grid = BM.build_grid(MIXTRAL, tiles, r_max, 32, 6)
    cells = BM.synthetic_cells(MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE,
                               bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)
    if slowdown == 1.0:
        return cells
    return [BM.make_cell(MIXTRAL, c.rows_per_expert, c.block_m,
                         c.ms_p50 * slowdown, sm_count=132, block_n=64)
            for c in cells]


def report_from(cells, *, alpha: float, tiles=TILES):
    return BM.analyse(
        cells, MIXTRAL, block_sizes=tiles, alpha=alpha, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name=MIXTRAL.name, dtype="bf16",
        compiles={bm: 1 for bm in tiles}, executed={bm: 1 for bm in tiles},
        sm_count=132, sm_source="test", ridge_band=RIDGE_BAND,
        ridge_source="stated by the test", ridge_band_source="stated by the test")


def gate(report, number: int):
    return next(g for g in report.gates if g.number == number)


# --------------------------------------------------------------------------
# (a) Gate 3 is not an observation and must not read as one.
# --------------------------------------------------------------------------

def test_gate_3_no_longer_reports_a_crossing_ratio_it_never_measured():
    """The retired form printed `1 + alpha_hat` as `measured`, which matched
    `1 + alpha_measured` in all 22 published reports that carry both -- because
    it IS that, evaluated rather than observed. The new `measured` is the fitted
    alpha itself, so a reader cannot mistake algebra for a second reading."""
    report = report_from(cells_at(REFIT), alpha=REFIT)
    g = gate(report, 3)
    alpha_hat = report.payload["alpha_measured"]
    assert alpha_hat is not None, "this test needs an identifiable alpha"
    assert g.measured == f"alpha {alpha_hat:.3f}"
    assert "x" not in g.measured, "a ratio-looking measured field is back"
    # The retired number is still recoverable, and still labelled.
    assert g.provenance["restated_crossing_ratio"] == pytest.approx(1.0 + alpha_hat)
    assert g.provenance["not_an_observed_crossing"] is True
    assert g.provenance["observed_crossing_ratio"] is None


def test_gate_3_claim_says_alpha_and_not_a_crossing():
    """"the BLOCK_M=128 crossing sits Q above the BLOCK_M=256 one" claims an
    observation. The identity behind it needs the 128 crossing to land in tread
    2, which at the A100's own alphas it does not in 3 of 6 arms, and no
    crossing has ever been observed at all."""
    g = gate(report_from(cells_at(REFIT), alpha=REFIT), 3)
    assert "alpha" in g.claim
    assert "crossing" not in g.claim
    assert g.basis in (BM.DERIVED, BM.IMPORTED), "a fitted alpha is not OBSERVED"


def test_gate_3_says_out_loud_that_no_crossing_was_observed():
    """`crosses` is True zero times across all of results/published. A gate that
    never saw one has to print that rather than imply the opposite."""
    lines = gate(report_from(cells_at(REFIT), alpha=REFIT), 3).lines
    assert any("no crossing has been observed" in ln for ln in lines)
    assert any("not a crossing ratio" in ln for ln in lines)


def test_gate_3_threshold_is_the_alpha_midpoint_and_still_discriminates():
    """Renaming a gate must not weaken it. `measured > 1.33` reduced to
    `alpha_hat > 0.33`, so the discrimination between the refit 0.558 and the
    retracted 0.10 is unchanged and has to survive the rename."""
    assert BM.GATE3_ALPHA_DISCRIMINATOR == pytest.approx(
        BM.GATE3_DISCRIMINATOR - 1.0)
    assert RETRACTED < BM.GATE3_ALPHA_DISCRIMINATOR < REFIT
    assert gate(report_from(cells_at(REFIT), alpha=REFIT), 3).verdict == "PASS"
    assert gate(report_from(cells_at(RETRACTED), alpha=RETRACTED),
                3).verdict == "FAIL"


def test_gate_3_names_the_block_size_its_alpha_came_from_in_the_json():
    """The detail lines said "imported"; the JSON did not, and the JSON is what
    survives into results/published. An import whose source is only in the text
    is an import a machine-readable reader cannot see."""
    g = gate(report_from(cells_at(REFIT), alpha=REFIT), 3)
    assert g.provenance["alpha_source_block_m"] in (32, 64, 128)
    assert g.provenance["imported_from_another_block_m"] is True
    assert "BLOCK_M=" in g.provenance["alpha_source"]


def test_the_serialized_gates_carry_their_detail_and_provenance():
    """The retired serializer kept {number, claim, verdict, measured, gate} and
    dropped every detail line, so a published verdict could not be traced at
    all. Anything a reader needs to tell an observation from a restatement now
    survives into report.json."""
    payload = report_from(cells_at(REFIT), alpha=REFIT).payload
    blob = json.loads(json.dumps(payload))          # it has to be serializable
    for entry in blob["gates"]:
        assert entry["kind"] in (BM.VALIDITY, BM.CLAIM)
        assert entry["basis"] in (BM.OBSERVED, BM.DERIVED, BM.IMPORTED)
        assert "detail" in entry and "provenance" in entry
    three = next(e for e in blob["gates"] if e["number"] == 3)
    assert three["provenance"]["not_an_observed_crossing"] is True
    assert three["detail"], "gate 3 shipped without its provenance lines"
    zero = next(e for e in blob["gates"] if e["number"] == 0)
    assert zero["kind"] == BM.VALIDITY, "the one instrument gate is not labelled"


# --------------------------------------------------------------------------
# (b) Gate 4 compared different units, and could not have failed.
# --------------------------------------------------------------------------

def test_gate_4_measures_and_thresholds_in_the_same_units():
    """Both sides are fractions of `ridge x bandwidth` now. The retired form put
    a fraction of the arm's own plateau against a threshold whose rationale was
    a fraction of peak compute."""
    report = report_from(cells_at(REFIT), alpha=REFIT)
    g = gate(report, 4)
    assert "ridge x bandwidth" in g.measured
    assert "ridge x bandwidth" in g.threshold
    assert g.provenance["units"] == "fraction of ridge x bandwidth (peak compute)"
    roof = report.payload["model_roof_tflops"]
    assert g.provenance["roof_tflops"] == pytest.approx(roof)
    # And the retired arm-relative number is still printed, as a diagnostic.
    assert g.provenance["peak_arm_relative_fraction"] is not None
    assert any("ARM-RELATIVE and it is not the verdict" in ln for ln in g.lines)


def test_the_two_denominators_differ_by_the_plateaus_own_shortfall():
    """THE ARITHMETIC OF THE UNIT ERROR, on cells built to have the published
    reports' shape: a plateau at 60% of the roof with the plateau-relative
    ladder unchanged. The same run reads 0.66 against the plateau and 0.39
    against peak compute, and 0.85 was stated about the second."""
    report = report_from(cells_at(REFIT, slowdown=1 / 0.6), alpha=REFIT)
    g = gate(report, 4)
    roof = report.payload["model_roof_tflops"]
    assert report.payload["plateau_tflops"] / roof == pytest.approx(0.6, abs=0.02)
    absolute = g.provenance["peak_roof_fraction"]
    arm = g.provenance["peak_arm_relative_fraction"]
    assert arm / absolute == pytest.approx(1 / 0.6, rel=0.05)
    assert absolute < arm, "the two denominators stopped differing"


def test_gate_4_refuses_to_pass_when_nothing_in_the_grid_reached_the_roof():
    """NON-VACUITY, and the state every published report is in. An absence
    reported by an instrument never shown to detect a presence is not evidence,
    and against the arm's own plateau the control can never fail, because the
    plateau IS the maximum over the same cells."""
    report = report_from(cells_at(REFIT, slowdown=1 / 0.6), alpha=REFIT)
    g = gate(report, 4)
    assert g.verdict == "UNDECIDED"
    assert g.provenance["positive_control_block_m"] is None
    assert g.provenance["best_other_roof_fraction"] < BM.COMPUTE_BOUND_FRACTION
    assert any("NOT BRACKETED" in ln for ln in g.lines)
    # The same cells, scored the retired way, would have had a control at 1.00.
    plateau = report.payload["plateau_tflops"]
    arm_relative = BM._throughput_ladder(
        [c for c in cells_at(REFIT, slowdown=1 / 0.6) if c.block_m == 256],
        256, plateau)
    assert max(v for _, v in arm_relative) == pytest.approx(1.0, abs=0.02)


def test_the_positive_control_is_vacuous_against_the_plateau_and_real_against_the_roof():
    """The control is the whole non-vacuity argument, so the difference between
    the two denominators is pinned directly rather than through a gate."""
    cells = cells_at(REFIT, slowdown=1 / 0.6)
    fits = {bm: BM.fit_ladder(BM.ladder_points(cells, bm), bm, None)
            for bm in TILES}
    plateau = max(c.useful_tflops for c in cells if c.aligned)
    vacuous = BM.bracketing(cells, 64, REFIT, RIDGE, 2, fits, plateau)
    real = BM.bracketing(cells, 64, REFIT, RIDGE, 2, fits, ROOF)
    assert vacuous.positive_control is not None, "the trap stopped firing"
    assert real.positive_control is None
    assert real.best_other_roof_fraction == pytest.approx(0.6, abs=0.05)


def test_gate_4_still_fails_when_the_block_size_really_did_reach_the_roof():
    """The rescaling must not cost the gate its teeth. At the retracted alpha
    BLOCK_M=64 crosses at 208 rows and the grid goes to 1024, so it reaches the
    roof in absolute units and the gate has to say so."""
    g = gate(report_from(cells_at(RETRACTED), alpha=RETRACTED), 4)
    assert g.verdict == "FAIL"
    assert g.provenance["reached_roof"] is True
    assert g.provenance["peak_roof_fraction"] >= BM.COMPUTE_BOUND_FRACTION


def test_gate_4_skips_the_ceiling_test_when_the_two_worlds_coincide():
    """At alpha=0.10 the BLOCK_M=64 ceiling is 640/160.3, clamped to 1.0, which
    is also the rival's. A midpoint between two identical numbers discriminates
    nothing, so the ceiling test is skipped and said to be skipped rather than
    quietly scored."""
    model, retracted, threshold = BM.gate_4_roof_fraction(
        block_m=64, alpha=RETRACTED, ridge=RIDGE, b=2)
    assert model == retracted == 1.0, "the clamp stopped applying"
    assert threshold == 1.0
    g = gate(report_from(cells_at(RETRACTED), alpha=RETRACTED), 4)
    assert g.provenance["worlds_separate"] is False
    assert any("CEILING TEST IS SKIPPED" in ln for ln in g.lines)


def test_the_retracted_alpha_ceiling_is_clamped_at_the_roof():
    """Without the clamp `ai_cap(64, 0.10)/ridge` is 3.99 and the midpoint 2.35,
    a threshold no throughput can ever exceed. A gate that cannot fail is not a
    gate."""
    assert BM.ai_cap(64, RETRACTED, 2) / RIDGE > 3.0
    _, retracted, threshold = BM.gate_4_roof_fraction(
        block_m=64, alpha=REFIT, ridge=RIDGE, b=2)
    assert retracted == 1.0
    assert threshold < 1.0


def test_gate_4_at_the_refit_alpha_passes_only_with_a_demonstrated_control():
    """The other half: when something in the grid did reach the roof and this
    block size did not, the absence is evidence and the gate scores it."""
    report = report_from(cells_at(REFIT), alpha=REFIT)
    g = gate(report, 4)
    assert g.verdict == "PASS"
    assert g.provenance["positive_control_block_m"] in (128, 256)
    assert g.provenance["peak_roof_fraction"] < g.provenance["threshold"]


# --------------------------------------------------------------------------
# (c) The ridge default.
# --------------------------------------------------------------------------

class _Args:
    """The three fields `resolve_ridge` reads. A namespace rather than a parsed
    argv so a test can state one impossible combination at a time."""

    def __init__(self, ridge=0.0, ridge_band="", dtype="bf16"):
        self.ridge, self.ridge_band, self.dtype = ridge, ridge_band, dtype


def test_the_ridge_no_longer_defaults_to_a_constant_on_the_command_line():
    """`--ridge` defaulted to `RIDGE_BAND[0]`, and cross_card_surface.sh never
    passed it. That single default is how a stale H200 number reached seven
    A100 reports."""
    args = BM.build_parser().parse_args([])
    assert args.ridge == 0.0
    assert args.ridge_band == ""


def test_a_run_with_no_calibration_for_its_device_refuses_rather_than_defaults(
        monkeypatch):
    """REFUSE rather than default. The message has to name the fix and has to
    name the constant it is refusing, or the next reader re-adds it."""
    monkeypatch.setattr(roofline, "current_gpu_name", lambda: "NVIDIA MADE-UP")
    monkeypatch.setattr(roofline, "load_measured", lambda *a, **k: None)
    with pytest.raises(BM.RidgeUnavailable) as exc:
        BM.resolve_ridge(_Args(), synthetic=False)
    assert "calibrate_hardware.py" in str(exc.value)
    assert "160.3" in str(exc.value)
    assert "--ridge" in str(exc.value)


def test_a_planning_run_may_assume_the_h200_band_and_must_say_so(monkeypatch):
    """--dry-run and --self-test measure nothing, so nothing can be mislabelled
    -- but the assumption still has to travel into the report, because the
    printout is not what gets published."""
    monkeypatch.setattr(roofline, "current_gpu_name", lambda: "")
    monkeypatch.setattr(roofline, "load_measured", lambda *a, **k: None)
    rr = BM.resolve_ridge(_Args(), synthetic=True)
    assert rr.ridge == BM.RIDGE_BAND[0]
    assert "HYPOTHESIS" in rr.source
    assert "belongs to no attached device" in rr.source


def test_the_device_calibration_is_what_a_run_on_that_device_gets(monkeypatch):
    """THE FIX, on the two calibrations actually committed. The A100's own
    ridge is 145.8 and the H200's is 162.8, and 160.3 is neither."""
    for path, expected in ((A100_YAML, 145.8), (H200_YAML, 162.8)):
        data = yaml.safe_load(path.read_text())
        hw = roofline.load_hardware(path.stem)
        monkeypatch.setattr(roofline, "current_gpu_name",
                            lambda name=data["detail"]["gpu_name"]: name)
        monkeypatch.setattr(roofline, "load_measured", lambda *a, hw=hw, **k: hw)
        rr = BM.resolve_ridge(_Args(), synthetic=False)
        assert rr.ridge == pytest.approx(expected, abs=0.1)
        assert "measured on this device" in rr.source
        assert data["detail"]["gpu_name"].split()[1][:4].lower() in rr.device.lower()
        # WHICH calibration session, not merely which device. measured_*.yaml
        # is keyed by device NAME, so a second pod of the same part inherits
        # the first pod's ceilings, and the H200's dense bf16 moved 7.1%
        # between two sessions while its bandwidth reproduced to 0.014%.
        assert str(data["checked_on"]) in rr.source
        assert data["measured_commit"][:8] in rr.source


def test_the_module_constant_is_not_the_ridge_of_either_committed_card():
    """The reason the default was a defect and not a rounding difference: 160.3
    is 9.9% above the A100's own ridge and 1.5% below the H200's, and it is a
    third machine-session's number."""
    a100 = roofline.load_hardware(A100_YAML.stem).ridge_point("bf16")
    h200 = roofline.load_hardware(H200_YAML.stem).ridge_point("bf16")
    assert a100 == pytest.approx(145.81, abs=0.05)
    assert h200 == pytest.approx(162.81, abs=0.05)
    assert abs(BM.RIDGE_BAND[0] - a100) / a100 > 0.09
    assert BM.RIDGE_BAND[0] != pytest.approx(h200, abs=1.0)


def test_the_band_comes_from_the_devices_own_bandwidth_patterns():
    """A single calibration has a real width -- the same silicon against several
    DRAM rulers -- and that width is this card's, not another card's. The A100's
    `read` pattern is disowned by its own calibration ("not a valid ceiling")
    and must not set an end of the band."""
    detail = BM._measured_detail("NVIDIA A100-SXM4-80GB")
    assert detail["ceiling_pattern"] == "triad"
    ridge = roofline.load_hardware(A100_YAML.stem).ridge_point("bf16")
    band, source = BM.ridge_band_from_detail(detail, ridge)
    by_pattern = detail["ridge_by_pattern"]
    assert band[0] < ridge < band[1], "the ceiling pattern is not inside its band"
    assert band[1] == pytest.approx(by_pattern["copy"], abs=0.3)
    assert band[1] != pytest.approx(by_pattern["read"], abs=0.3), \
        "a disowned pattern set the top of the band"
    assert "disowned" in source and "read" in source


def test_an_operator_asserted_ridge_gets_a_degenerate_band_and_says_so():
    """One number is one number. Inheriting a WIDTH measured on another machine
    is the same error as inheriting the point, and it is the width that decides
    which tread a crossing lands in."""
    rr = BM.resolve_ridge(_Args(ridge=200.0), synthetic=False)
    assert rr.band == (200.0, 200.0)
    assert "degenerate" in rr.band_source
    wide = BM.resolve_ridge(_Args(ridge=200.0, ridge_band="190,210"),
                            synthetic=False)
    assert wide.band == (190.0, 210.0)
    with pytest.raises(BM.RidgeUnavailable):
        BM.resolve_ridge(_Args(ridge=200.0, ridge_band="190"), synthetic=False)


def test_a_calibration_with_no_pattern_detail_gets_a_degenerate_band():
    """REFUSE rather than invent. A band this run cannot measure is reported as
    one number twice, which is honest about being one calibration."""
    band, source = BM.ridge_band_from_detail({}, 145.8)
    assert band == (145.8, 145.8)
    assert "degenerate" in source


def test_analyse_never_borrows_the_module_band_when_the_caller_states_none():
    """The defect in one line: a default is what put an H200 band on an A100
    report. With no band stated the report gets this run's own ridge twice and
    prints that the band is degenerate."""
    cells = cells_at(REFIT)
    report = BM.analyse(
        cells, MIXTRAL, block_sizes=TILES, alpha=REFIT, ridge=145.81,
        bandwidth_gbps=1799.36, b=2, model_name=MIXTRAL.name, dtype="bf16",
        compiles={bm: 1 for bm in TILES}, executed={bm: 1 for bm in TILES},
        sm_count=108, sm_source="test")
    assert report.payload["ridge_band"] == [145.81, 145.81]
    assert report.payload["ridge_band_degenerate"] is True
    assert list(BM.RIDGE_BAND) != report.payload["ridge_band"]
    assert "DEGENERATE" in report.text()
    assert report.payload["ridge_source"] == "NOT STATED by the caller"


def test_both_derived_thresholds_are_registered_before_the_measurement():
    """A threshold derived from this run's ridge and alpha, printed only beside
    its own verdict, is indistinguishable from one chosen after the fact. Both
    are stated in the PREDICTIONS block, above the first measured number."""
    text = report_from(cells_at(REFIT), alpha=REFIT).text()
    predictions, _, measured = text.partition("MEASURED  ")
    assert measured, "the report stopped having a MEASURED section"
    assert f"GATE 3 will test alpha > {BM.GATE3_ALPHA_DISCRIMINATOR:.2f}" in predictions
    assert "GATE 4 will test BLOCK_M=64" in predictions
    _, _, threshold = BM.gate_4_roof_fraction(
        block_m=64, alpha=REFIT, ridge=RIDGE, b=2)
    assert f"{threshold:.3f}" in predictions


def test_the_report_carries_the_ridge_and_its_source_into_the_json():
    """Provenance that only exists on stdout is provenance that does not survive
    publication, which is how 160.3 sat in 26 committed reports unremarked."""
    payload = report_from(cells_at(REFIT), alpha=REFIT).payload
    assert payload["ridge"] == RIDGE
    assert payload["ridge_source"] == "stated by the test"
    assert payload["ridge_band_source"] == "stated by the test"
    assert payload["model_roof_tflops"] == pytest.approx(ROOF)
    assert "ridge source:" in report_from(cells_at(REFIT), alpha=REFIT).text()


def test_the_dry_run_prints_the_ridge_and_its_source_before_any_gpu_time(capsys):
    """A plan that does not say which card's ceiling it is planning against is
    the plan that produced seven hybrid reports. `--dry-run` has to run off GPU
    and has to print the ridge, its band and where both came from."""
    assert BM.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "\nridge  " in out and "ridge band " in out
    assert "HYPOTHESIS" in out, "a planning ridge that does not say it is one"
    assert "estimated GPU time" in out, "the plan printed no cost"


def test_a_measured_run_with_no_calibration_stops_before_it_spends_gpu_time(
        monkeypatch, capsys, tmp_path):
    """The refusal has to happen at the top of `main`, not after the sweep: a
    run that cannot state a ridge produces nothing worth the pod minutes."""
    monkeypatch.setattr(roofline, "current_gpu_name", lambda: "NVIDIA MADE-UP")
    monkeypatch.setattr(roofline, "load_measured", lambda *a, **k: None)
    assert BM.main(["--out", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert out.startswith("REFUSED:")
    assert "calibrate_hardware.py" in out
    assert not list(tmp_path.rglob("report.json")), "it wrote a report anyway"
