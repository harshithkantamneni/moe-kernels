"""The four apparatus repairs to `scripts/tile_cap_test.py`, each pinned.

`tests/test_tile_cap.py` is the experiment's own suite: the depth argument, the
two worlds, the demotion from a production claim, the run-id collisions. This
file is narrower and is about the APPARATUS the 2026-09-02 audit found below the
bar, one section per finding, so that a later reader can see which check exists
because of which defect:

  R1  V3 asked a FUSED layer to reach the DENSE cuBLAS peak. No published arm
      exceeds 0.54 (H200) or 0.64 (A100), so the gate failed on every card, the
      FAIL voided the page and the arm was INVALID by construction: nine pod
      minutes whose verdict was computable from the calibration alone.
  R2  Both planted worlds were noiseless, so every noise-floored threshold was
      exercised at the one spread no pod produces; nothing asserted a verdict
      per world; and C2's FAIL branch was unreachable in either world.
  R3  The file knew `B/(A+B)` biases the cap and said so in prose, with no
      number. `moe.bench.ai_model` names the factor exactly.
  R4  The instrument, the provenance block, the exit-code table and the run-id
      rule were the script's own rather than the repository's.

Everything here runs off GPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAP = _load_script("tile_cap_test")
SWEEP = CAP.SWEEP

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (16, 256)
RIDGE, RIDGE_HI = 160.3, 176.2
BAND = (RIDGE, RIDGE_HI)
BANDWIDTH = 4374.5
ROOF = RIDGE * BANDWIDTH * 1e9 / 1e12


def _cells(alpha: float, *, r_max: int = 2112, noise: float = 0.0, seed: int = 0):
    grid = SWEEP.build_grid(MIXTRAL, TILES, r_max, 32, 6)
    return grid, SWEEP.synthetic_cells(
        MIXTRAL, grid, TILES, alpha=alpha, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
        b=2, sm_count=132, noise=noise, seed=seed)


def _report(alpha: float, **kw):
    grid, cells = _cells(alpha, **kw)
    return CAP.analyse(
        cells, MIXTRAL, cap_tile=16, control_tile=256, alpha=alpha, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name="mixtral-8x7b", dtype="bf16",
        compiles={16: 1, 256: 1}, executed={16: 1, 256: 1}, sm_count=132,
        sm_source="test", depth=CAP.required_depth(16, b=2, ridge_band=BAND),
        planned_cells=len(grid) * len(TILES), header=[])


def _tp(fraction: float, treads: int = 4):
    """A control throughput ladder pinned at one fraction of the dense peak."""
    return [(n, fraction) for n in range(1, treads + 1)]


# --------------------------------------------------------------------------
# R1. The ruler V3 scores against.
# --------------------------------------------------------------------------

def test_the_published_fused_layer_band_is_where_the_v3_floor_comes_from():
    """The floor is the band's low end, not a number someone liked.

    46.5-75.6% of `ridge x bandwidth` is what the sibling's `bracketing`
    docstring counted across 26 published reports. If that band is ever
    re-measured, the floor must move with it rather than being edited
    separately, which is what this pins.
    """
    lo, hi = CAP.FUSED_PLATEAU_BAND
    assert CAP.FUSED_ROOF_FLOOR == lo
    assert lo < hi <= CAP.FUSED_ROOF_CEILING
    # The dense-compute-bound fraction the sibling uses is ABOVE the whole
    # band, which is exactly why asking a fused layer for it never passed.
    assert SWEEP.COMPUTE_BOUND_FRACTION > hi


@pytest.mark.parametrize("fraction", [0.465, 0.51, 0.54, 0.64, 0.756, 0.95])
def test_v3_passes_everywhere_a_real_fused_layer_has_ever_landed(fraction):
    """THE DEFECT, STATED AS A RANGE. Every published arm sits in here."""
    gate = CAP.gate_v3_control_roof(_tp(fraction), control_tile=256,
                                    roof_tflops=ROOF, plateau=fraction * ROOF)
    assert gate.verdict == CAP.PASS, gate.measured


@pytest.mark.parametrize("fraction", [0.0001, 0.10, 0.30, 0.46])
def test_v3_still_fails_when_nothing_reached_any_roof(fraction):
    """The FAIL branch is reachable, which is what makes the PASS mean anything."""
    gate = CAP.gate_v3_control_roof(_tp(fraction), control_tile=256,
                                    roof_tflops=ROOF, plateau=fraction * ROOF)
    assert gate.verdict == CAP.FAIL
    assert "C2 SURVIVES" in gate.consequence


def test_v3_refuses_a_control_that_beat_the_dense_peak():
    """A broken ruler is not a strong kernel, and UNDECIDED is not PASS."""
    gate = CAP.gate_v3_control_roof(_tp(1.40), control_tile=256,
                                    roof_tflops=ROOF, plateau=1.40 * ROOF)
    assert gate.verdict == CAP.UNDECIDED
    assert "ABOVE the dense peak" in " ".join(gate.lines)


def test_the_upper_wall_widens_with_the_measured_spread():
    """`top` is a MAXIMUM over treads, so it carries the spread upward.

    The planted world lands at exactly 1.000 of the dense peak at zero noise and
    at 1.02 / 1.04 / 1.07 at spreads of 1 / 2 / 3%. A fixed wall would turn the
    self-test's own noise into a refusal, so the wall widens the way V2's
    flatness gate does.
    """
    quiet = CAP.gate_v3_control_roof(_tp(1.07), control_tile=256,
                                     roof_tflops=ROOF, plateau=ROOF, noise=0.0)
    noisy = CAP.gate_v3_control_roof(_tp(1.07), control_tile=256,
                                     roof_tflops=ROOF, plateau=ROOF, noise=0.03)
    assert quiet.verdict == CAP.UNDECIDED
    assert noisy.verdict == CAP.PASS


def test_the_gate_and_the_report_share_one_fused_roof():
    """One function owns the number, so a threshold cannot be registered for
    one quantity and scored against another."""
    report = _report(CAP.ALPHA)
    assert (report.payload["fused_layer_roof_tflops"]
            == pytest.approx(CAP.fused_layer_roof(
                report.payload["model_roof_tflops"])))
    v3 = next(g for g in report.gates if g.tag == "V3")
    assert f"{report.payload['fused_layer_roof_tflops']:.0f}" in v3.threshold


def test_a_roof_of_zero_refuses_instead_of_returning_zero():
    with pytest.raises(CAP.Unmeasurable):
        CAP.fused_layer_roof(0.0)


def test_the_dense_peak_fraction_is_still_reported_and_is_labelled_diagnostic():
    """It is the number that compares across cards and against the sibling's
    gate 4, and it is the number that must never again be the verdict."""
    gate = CAP.gate_v3_control_roof(_tp(0.52), control_tile=256,
                                    roof_tflops=ROOF, plateau=0.52 * ROOF)
    text = " ".join(gate.lines)
    assert "DIAGNOSTIC, not the verdict" in text
    assert "0.520" in text


# --------------------------------------------------------------------------
# R2. Planted noise, planted worlds, and a registered verdict per world.
# --------------------------------------------------------------------------

def test_plant_noise_defaults_to_the_published_spread():
    args = CAP.build_parser().parse_args([])
    assert args.plant_noise == CAP.PUBLISHED_CELL_SPREAD
    assert 0.0076 <= CAP.PUBLISHED_CELL_SPREAD <= 0.0182, (
        "the default must sit inside the published H200 ladder spread range; "
        "outside it, it is a number rather than an assumption")


def test_the_retired_spelling_still_reaches_the_same_knob():
    """`--self-test-noise` is the name the sibling and the docs used. It must
    not silently become a second, separately-defaulting knob."""
    args = CAP.build_parser().parse_args(["--self-test-noise", "0.02"])
    assert args.plant_noise == 0.02


@pytest.mark.parametrize("alpha", sorted(CAP.SELF_TEST_WORLDS))
def test_every_registered_world_returns_its_registered_verdicts(alpha):
    """The registration IS the test. A self-test that only runs is a smoke
    test, and both worlds this file used to plant asserted nothing at all."""
    world = CAP.SELF_TEST_WORLDS[alpha]
    report = _report(alpha, noise=CAP.PUBLISHED_CELL_SPREAD)
    assert world.check(report) == []


def test_the_worlds_between_them_reach_every_branch_of_c2():
    """C2's FAIL branch was unreachable before `C2_FAIL_ALPHA` existed.

    At 0.558 C2 passes; at the retracted 0.10 the cap tile's memory branch runs
    parallel to the compute branch (ridge/cap = 1.002, inside the tolerance) so
    the fit declines to name an alpha and C2 is UNDECIDED. Neither can fail it.
    """
    seen = {w.expect.get("C2") for w in CAP.SELF_TEST_WORLDS.values()}
    assert {CAP.PASS, CAP.FAIL, CAP.UNDECIDED} <= seen


def test_the_c2_fail_world_is_where_the_arithmetic_says_it_is():
    """Derived, not tuned. Two conditions have to hold at once and the
    docstring's numbers are recomputed here rather than quoted."""
    cap = SWEEP.ai_cap(CAP.CAP_TILE, CAP.C2_FAIL_ALPHA, 2)
    disc = CAP.cap_discriminator(CAP.CAP_TILE, RIDGE, 2)
    assert cap / RIDGE > disc, "below the discriminator C2 would PASS"
    assert abs(RIDGE / cap - 1.0) > SWEEP.PARALLEL_BRANCH_TOLERANCE, (
        "inside the tolerance the fit refuses to name an alpha and C2 goes "
        "UNDECIDED, which is the state this world exists to escape")


def test_a_world_that_breaks_its_registration_is_an_error_and_not_a_result(
        tmp_path, monkeypatch):
    """A self-test coming out other than registered is a broken apparatus.

    ERROR rather than INVALID or CLAIM_FAIL, because both of those invite a
    reader to interpret the run. Planted by registering a verdict the refit
    world cannot produce.
    """
    broken = CAP.PlantedWorld(CAP.ALPHA, "a registration that cannot hold",
                              {"C1": CAP.FAIL})
    monkeypatch.setitem(CAP.SELF_TEST_WORLDS, CAP.ALPHA, broken)
    rc = CAP.main(["--self-test", str(CAP.ALPHA), "--out", str(tmp_path)])
    assert rc == exit_codes.ERROR


def test_a_registration_is_not_asserted_on_a_design_it_was_not_made_for():
    """`--r-max 512` fails V1 and V2 BY DESIGN, and that is not a defect.

    A check that fires on the one thing it is not about is worse than no check,
    because the way to silence it is to weaken the registration.
    """
    world = CAP.SELF_TEST_WORLDS[CAP.ALPHA]
    assert world.applies(tiles=(16, 256), reached_rows=2112,
                         needed_rows=2112) == ""
    shallow = world.applies(tiles=(16, 256), reached_rows=512, needed_rows=2112)
    assert "512" in shallow
    other = world.applies(tiles=(32, 128), reached_rows=2112, needed_rows=2112)
    assert "tile pair" in other


def test_a_gate_named_in_a_registration_that_the_report_omits_is_a_mismatch():
    """A registration that silently matches nothing is the
    check-that-examined-nothing shape one level up."""
    world = CAP.PlantedWorld(CAP.ALPHA, "names a gate that does not exist",
                             {"V9": CAP.PASS})
    bad = world.check(_report(CAP.ALPHA))
    assert len(bad) == 1 and "no gate V9" in bad[0]


def test_the_audit_s_own_command_reports_v3_pass(tmp_path, capsys):
    """`--self-test 0.558 --plant-noise 0.015 | grep '^RESULT: VALIDITY V3'`,
    which is the check the audit asked for, run as the audit wrote it."""
    CAP.main(["--self-test", "0.558", "--plant-noise", "0.015",
              "--out", str(tmp_path)])
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("RESULT: VALIDITY V3")]
    assert len(lines) == 1 and " PASS " in lines[0]


# --------------------------------------------------------------------------
# R3. The EXA identity, and the factor a LIN cap is high by.
# --------------------------------------------------------------------------

def test_c2_prints_the_exa_form_and_the_overstatement_beside_the_cap():
    report = _report(CAP.ALPHA)
    c2 = next(g for g in report.gates if g.tag == "C2")
    text = " ".join(c2.lines)
    assert "(alpha_b + phi)/(1 + phi + delta)" in text
    assert "exact cap implied" in text
    assert "lin_overstatement" in text


def test_the_printed_overstatement_is_ai_models_own_number():
    """Recomputed from `ai_model` here, so a drift in either direction is a
    failure rather than two files quietly disagreeing."""
    report = _report(CAP.ALPHA)
    lo, hi = report.payload["lin_overstatement"]
    ends = []
    for alpha_a in (0.0, 1.0):
        p = ai_model.phi(2 * MIXTRAL.intermediate_size, MIXTRAL.hidden_size,
                         block_m=CAP.CAP_TILE, block_n=64, alpha_a=alpha_a, b=2)
        ends.append(ai_model.lin_overstatement(phi=p, delta=0.0))
    assert (lo, hi) == pytest.approx((min(ends), max(ends)))
    assert lo > 1.0, "a factor of 1.0 would say the LIN cap is exact"


def test_the_overstatement_grows_with_the_tile_and_is_smallest_where_it_is_tested():
    """WHY BLOCK_M=16 IS WHERE THE FORMULA IS TESTABLE, as a number.

    phi is the activation re-read in units of one weight read and scales with
    BLOCK_M/BLOCK_N, so the bracket widens fast with the tile: at BM=128 its top
    end is over 3x, far larger than the cap-to-ridge gap the study is deciding,
    while the whole bracket at BM=16 stays under 1.26.
    """
    brackets = [SWEEP.cap_overstatement(MIXTRAL, bm, 64, 2)
                for bm in (16, 64, 128, 256)]
    assert [lo for lo, _ in brackets] == sorted(lo for lo, _ in brackets)
    assert [hi for _, hi in brackets] == sorted(hi for _, hi in brackets)
    assert all(lo > 1.0 for lo, _ in brackets), (
        "a factor of 1.0 would say a LIN cap is exact at some tile height")
    assert brackets[0][1] < 1.26, "the cap tile's whole bracket is narrow"
    assert brackets[2][1] > 3.0, "and BM=128's is not"


def test_a_gate_with_no_model_config_says_the_factor_is_unstated():
    """REFUSE RATHER THAN DEFAULT. A cap printed with no factor beside it is
    the shape that put a 31% overstatement into the study's headline table."""
    report = _report(CAP.ALPHA)
    fit = report.payload["ladder"]["16"]
    gate = CAP.gate_c2_measured_cap(
        _fit_of(report), report.payload["alpha_corrected"], cap_tile=16,
        ridge=RIDGE, b=2, discriminator=0.589, cfg=None, block_n=0)
    assert fit["alpha"] is not None
    assert "NOT stated here" in " ".join(gate.lines)


def _fit_of(report):
    """The cap tile's `LadderFit`, refitted from the same cells the report used."""
    _, cells = _cells(CAP.ALPHA)
    ok = [c for c in cells if c.status == "ok" and c.ms_p50 > 0]
    ref = SWEEP.compute_reference(ok, TILES, cfg=MIXTRAL, ridge=RIDGE,
                                  bandwidth_gbps=BANDWIDTH, b=2)
    return SWEEP.fit_ladder(SWEEP.ladder_points(ok, 16), 16, ref,
                            SWEEP.MEMORY_BRANCH_MARGIN)


# --------------------------------------------------------------------------
# R4. The instrument, the provenance block, the exit table, the run id.
# --------------------------------------------------------------------------

def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does(
        tmp_path, capsys):
    CAP.main(["--self-test", "0.558", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    parsed = exit_codes.parse_result_lines(out)
    report = _report(CAP.ALPHA, noise=CAP.PUBLISHED_CELL_SPREAD)
    assert [r.name for r in parsed] == [g.tag for g in report.gates]
    raw = [ln for ln in out.splitlines() if ln.startswith("RESULT: ")]
    assert len(raw) == len(parsed), "a RESULT line the parser cannot read back"


@pytest.mark.parametrize("alpha,code", [
    (0.558, exit_codes.DONE),
    (0.10, exit_codes.DONE),          # CLAIM_FAIL, reported as DONE
])
def test_the_log_recomputes_the_code_the_process_returned(alpha, code,
                                                          tmp_path, capsys):
    """`classify_text` over the RESULT lines against the returned integer.

    They may differ in exactly one direction and for one reason: without
    `--fail-on-gate` a CLAIM_FAIL is REPORTED as DONE. Anything else is the
    defect `moe.bench.exit_codes` is named against -- a script printing one
    thing and exiting another.
    """
    rc = CAP.main(["--self-test", str(alpha), "--out", str(tmp_path)])
    assert rc == code
    implied = exit_codes.classify_text(capsys.readouterr().out)
    assert implied in (rc, exit_codes.CLAIM_FAIL)


def test_fail_on_gate_returns_claim_fail_where_the_default_returns_done(tmp_path):
    assert CAP.main(["--self-test", "0.10",
                     "--out", str(tmp_path / "a")]) == exit_codes.DONE
    assert CAP.main(["--self-test", "0.10", "--fail-on-gate",
                     "--out", str(tmp_path / "b")]) == exit_codes.CLAIM_FAIL


def test_the_report_carries_a_provenance_block_with_the_instrument_named(
        tmp_path):
    CAP.main(["--self-test", "0.558", "--out", str(tmp_path)])
    payload = json.loads(next(tmp_path.rglob("report.json")).read_text())
    for key in PV.TOP_LEVEL_KEYS:
        assert key in payload, key
    # A SYNTHETIC RUN NAMES THE SYNTHETIC INSTRUMENT. A planted report carrying
    # the real instrument's name satisfies a presence check while describing an
    # instrument the run never touched.
    assert payload["instrument"] == SWEEP.SYNTHETIC_INSTRUMENT
    assert payload["provenance"]["ridge_source"] == payload["ridge_source"]
    assert payload["provenance"]["bandwidth"] == pytest.approx(BANDWIDTH)


def test_a_direct_analyse_call_says_it_has_no_provenance_rather_than_looking_clean():
    payload = _report(CAP.ALPHA).payload
    assert payload["provenance"] is None
    assert "names no git sha" in payload["provenance_note"]


def test_the_run_id_is_the_repositorys_rule_and_not_a_local_hash():
    """One run-id scheme in the repository. Two is how two settings came to
    derive one directory in the first place."""
    args = CAP.build_parser().parse_args([])
    got = CAP.default_run_id(args, 2112, "NVIDIA H200")
    assert got.startswith(PV.card_slug("NVIDIA H200") + "-")
    # The hash is provenance's, over the whole canonical key, so any knob moving
    # moves the tail.
    other = CAP.default_run_id(
        CAP.build_parser().parse_args(["--trials", "9"]), 2112, "NVIDIA H200")
    assert other != got


def test_the_run_id_refuses_a_knob_that_was_never_resolved():
    """`provenance.run_id` raises on None, and it is right to: a run named
    without saying what a knob was is a directory two settings can share."""
    with pytest.raises(PV.UnresolvedKnob):
        PV.run_id(card="NVIDIA H200", model="mixtral-8x7b", planted=None)


def test_the_timing_knobs_are_the_instruments_units_and_are_in_the_id():
    """`--warmup` is MILLISECONDS of delivered load, because `run_sweep` hands
    it to `time_kernel` as a duration. A count there would be a warmup of 20 ms
    where 300 was meant, silently."""
    parser = CAP.build_parser()
    assert parser.parse_args([]).warmup == 300.0
    assert parser.parse_args(["--warmup-ms", "50"]).warmup == 50.0
    base = CAP.default_run_id(parser.parse_args([]), 2112, "NVIDIA H200")
    for knob in (["--warmup", "50"], ["--trials", "5"], ["--no-l2-flush"]):
        moved = CAP.default_run_id(parser.parse_args(knob), 2112, "NVIDIA H200")
        assert moved != base, knob


def test_the_plan_states_an_mde_from_the_stated_noise_assumption(capsys):
    CAP.main(["--dry-run", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert "MINIMUM DETECTABLE EFFECT" in out
    assert f"{CAP.PUBLISHED_CELL_SPREAD:.2%}" in out
    for tag in ("C1", "C2", "V2"):
        assert f"  {tag}  " in out


def test_the_mde_refuses_a_zero_spread_rather_than_reporting_omniscience(capsys):
    """At sigma = 0 every effect is detectable, which is a statement about the
    planted world and not about any pod."""
    CAP.main(["--dry-run", "--capability", "9.0", "--plant-noise", "0"])
    out = capsys.readouterr().out
    assert "MINIMUM DETECTABLE EFFECT: not stateable" in out


def test_the_mde_shrinks_with_depth_and_grows_with_the_spread():
    """Both directions, because a power calculation that moves the wrong way is
    invisible in a single printed number."""
    def slope(treads, spread):
        ys = [SWEEP.model_ms(MIXTRAL, n * 16, 16, alpha=CAP.ALPHA, ridge=RIDGE,
                             bandwidth_gbps=BANDWIDTH, b=2)
              for n in range(1, treads + 1)]
        return CAP.slope_relative_se(ys, spread)

    assert slope(40, 0.015) < slope(10, 0.015)
    assert slope(10, 0.03) == pytest.approx(2.0 * slope(10, 0.015))


def test_a_single_tread_has_no_slope_and_says_so():
    with pytest.raises(CAP.Unmeasurable):
        CAP.slope_relative_se([1.0], 0.015)
