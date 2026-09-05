"""The calibration: where it writes, what it refuses, and what it says it is.

Two things went wrong here and both were side effects rather than decisions.

The first is where the file landed. `calibrate_hardware.py` wrote
`moe/bench/hardware/measured_<device>.yaml`, which is TRACKED, so the first
metered step of every session modified the checkout: `pod_session.sh` P1 went
from PASS to FAIL and `driver.py` stamped `git_dirty=True` on every row measured
afterwards. All 3,696 rows of the 2026-09-01 alpha-0558 arm carry it, and 44,872
of the 100,144 published rows in total. The default output is now an untracked
session path and the tracked copy is `--publish`, so a run that does not ask for
it cannot cause it.

The second is what the file claimed. Nothing recorded which loop timed the
ceilings, while the ladders every alpha is fitted from were timed by a different
one; the audit put the resulting bias at 12-16% on the H200's smallest cells,
which is the size of the cross-card effect the study registers. So the yaml
carries a provenance block whose `instrument` names the function that actually
ran, and this file asserts that it refuses to claim `timing.TIMING_BASIS` while
`moe.bench.calibrate` is still on `time_eager`.

Every gate is exercised in BOTH directions against a synthetic `Calibration`,
which needs no GPU: a gate whose FAIL branch has never run is a gate nobody has
tested.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_hardware as CH  # noqa: E402

from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench.calibrate import BandwidthResult, Calibration  # noqa: E402


def pattern(name: str, gbps: float, note: str = "") -> BandwidthResult:
    return BandwidthResult(pattern=name, bytes_moved=8 << 30, ms_p50=2.0,
                           ms_min=1.9, gbps=gbps, gbps_peak_min=gbps * 1.02,
                           note=note, sm_clock_start_mhz=1755,
                           sm_clock_end_mhz=1755)


def calibration(**over) -> Calibration:
    """An H200-shaped calibration whose every gate passes, so a test that wants
    one gate to fail changes exactly one thing and nothing else can be blamed."""
    fields = dict(
        gpu_name="NVIDIA H200",
        achieved_bandwidth_gbps=4374.0,
        ceiling_pattern="triad",
        achieved_bf16_tflops=716.0,
        bandwidth_patterns=(pattern("triad", 4374.0), pattern("read_stream", 4200.0),
                            pattern("copy", 4100.0), pattern("write", 3900.0)),
        gemm_shape=(8192, 8192, 8192),
        gemm_clock_mhz=1755,
        # The idle pair the retired flag was built from stays in the record;
        # the UNDER-LOAD verdict beside it is what `not_throttled` scores.
        clocks={"sm_start_mhz": 1755, "sm_end_mhz": 1750, "temp_start_c": 40,
                "temp_end_c": 45, "throttled": False,
                "sm_clock_load_mhz": 1755.0, "reference_clock_mhz": 1755.0,
                "clock_level_ok": True, "clock_drift_ok": True},
        settle={"settled": True, "final_mhz": 1755},
        gemm_clock={"median_mhz": 1755, "samples": 5, "spread_pct": 0.4,
                    "after_idle_mhz": 1980},
    )
    fields.update(over)
    return Calibration(**fields)


#: The shape `calibrate()` wrote until the under-load verdict landed: the
#: idle-instant pair and the retired `throttled` flag, nothing sampled under
#: load. What every committed calibration in this tree carries today.
LEGACY_CLOCKS = {"sm_start_mhz": 1755, "sm_end_mhz": 1200, "temp_start_c": 40,
                 "temp_end_c": 85, "drift_pct": -31.6, "throttled": True}


#: The derived pin rate for an H200: 3201 MHz x 2 x 6144 bits / 8.
H200_PIN = 4916.7


# --------------------------------------------------------------------------
# the gates, both branches of each
# --------------------------------------------------------------------------

def verdicts(cal, pin=H200_PIN) -> dict[str, str]:
    return {name: verdict for _kind, name, verdict, _detail in CH.score(cal, pin)}


def test_a_sound_calibration_passes_every_gate_and_exits_done():
    """All six, named here so a gate added later cannot slip in untested: the
    docstring on `score` claimed five while emitting six, and the sixth had no
    FAIL branch anywhere in this file."""
    cal = calibration()
    scored = CH.score(cal, H200_PIN)
    assert {n for _k, n, _v, _d in scored} == {
        "clock_established", "ceiling_pattern_measured",
        "no_pattern_exceeds_the_pin_rate", "write_rate_is_a_store_rate",
        "clock_steady_across_patterns", "not_throttled"}
    assert {v for _k, _n, v, _d in scored} == {EX.PASS}
    assert EX.classify([(k, v) for k, _n, v, _d in scored]) == EX.DONE


def test_a_clock_that_was_never_established_is_invalid_not_done():
    """`sustained_peak_tflops` and `gemm_efficiency_pct` are normalised by that
    clock and `moe.bench.calibrate` refuses to quote them without it. INVALID is
    the table's word for measured-and-unquotable, and it is not REFUSED: the
    minutes were spent and the yaml exists."""
    cal = calibration(settle={"settled": True, "final_mhz": 1400},
                      gemm_clock={"median_mhz": 1755, "samples": 5,
                                  "spread_pct": 22.0, "after_idle_mhz": 1980})
    assert cal.clock_established is False
    assert verdicts(cal)["clock_established"] == EX.FAIL
    assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN)]) == EX.INVALID


def test_no_settle_leaves_the_clock_unknown_which_still_counts_against_it():
    """UNKNOWN is not PASS. `--no-settle` measures from idle and walks the clock
    ramp across the patterns, so there is nothing to establish the clock
    against, and "could not decide" must not read as "fine"."""
    cal = calibration(settle={}, gemm_clock={})
    assert cal.clock_established is None
    assert verdicts(cal)["clock_established"] == EX.UNKNOWN
    assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN)]) == EX.INVALID


def test_a_disowned_ceiling_pattern_fails_its_validity_gate():
    """`moe.bench.calibrate` marks a pattern it does not stand behind with the
    DISOWNED tail. Publishing a ceiling the measurement disowned is quoting a
    number its own instrument refused."""
    from moe.bench.calibrate import DISOWNED
    cal = calibration(bandwidth_patterns=(
        pattern("triad", 4374.0, note=f"reduction-limited, {DISOWNED}"),))
    assert verdicts(cal)["ceiling_pattern_measured"] == EX.FAIL


def test_a_pattern_above_the_pin_rate_is_a_failed_claim():
    """The pin rate is the one hard physical bound in the file. A figure above
    it means the byte accounting is wrong, not that the hardware exceeded its
    specification."""
    cal = calibration(bandwidth_patterns=(pattern("triad", 5200.0),))
    assert verdicts(cal)["no_pattern_exceeds_the_pin_rate"] == EX.FAIL
    assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN)]) == EX.CLAIM_FAIL


def test_a_write_rate_above_the_pin_rate_fails_its_own_gate():
    """The gate whose FAIL branch nobody had ever run.

    `write_rate_is_a_store_rate` is the arithmetic check on the store path: a
    write that reports N bytes per second when the hardware must move 2N for a
    read-for-ownership is measuring its own accounting, not the card. Every
    fixture here put write at 3900 GB/s against a 4916.7 pin rate, and the one
    fixture that goes over the pin drops every pattern but `triad`, so this gate
    was not even emitted in it. Both are now covered: the gate is emitted, and
    it fails.
    """
    cal = calibration(bandwidth_patterns=(pattern("triad", 4374.0),
                                          pattern("write", 5200.0)))
    scored = verdicts(cal)
    assert scored["write_rate_is_a_store_rate"] == EX.FAIL
    # and the general pin gate is failed by the same row, which is the point:
    # a write over the pin rate is a pattern over the pin rate.
    assert scored["no_pattern_exceeds_the_pin_rate"] == EX.FAIL
    assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN)]) == EX.CLAIM_FAIL
    detail = {n: d for _k, n, _v, d in CH.score(cal, H200_PIN)}
    assert "read-for-ownership" in detail["write_rate_is_a_store_rate"]


def test_a_calibration_with_no_write_pattern_omits_the_store_rate_gate():
    """A gate that was not scored must not read as one that passed.

    This is the shape that hid the missing FAIL branch above: the over-the-pin
    fixture had no write pattern, so `write_rate_is_a_store_rate` was silently
    absent from its verdicts and a test asking only about the gates present saw
    nothing wrong. Asserted directly so the conditional stays visible.
    """
    cal = calibration(bandwidth_patterns=(pattern("triad", 4374.0),))
    assert "write_rate_is_a_store_rate" not in verdicts(cal)
    assert "write_rate_is_a_store_rate" in verdicts(calibration())


def test_an_unknown_bus_width_leaves_the_pin_claim_untested_not_passed():
    """A card outside the memory-bus table has no derived pin rate, so the
    claim that nothing exceeded it was never tested. That is UNKNOWN, and
    UNKNOWN counts against the gate."""
    cal = calibration()
    scored = dict((n, v) for _k, n, v, _d in CH.score(cal, None))
    assert scored["no_pattern_exceeds_the_pin_rate"] == EX.UNKNOWN
    assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, None)]) == EX.CLAIM_FAIL


def test_a_clock_that_moved_across_the_patterns_fails():
    """Patterns measured in different clock states are not comparable with each
    other, and the ceiling is a comparison between them."""
    cal = calibration(bandwidth_patterns=(
        BandwidthResult("triad", 8 << 30, 2.0, 1.9, 4374.0, 4460.0, "",
                        sm_clock_start_mhz=1755, sm_clock_end_mhz=1755),
        BandwidthResult("write", 8 << 30, 2.0, 1.9, 3900.0, 3980.0, "",
                        sm_clock_start_mhz=1200, sm_clock_end_mhz=1200)))
    assert cal.clock_ramped
    assert verdicts(cal)["clock_steady_across_patterns"] == EX.FAIL


def test_a_card_below_its_reference_clock_fails_the_throttle_claim_on_level():
    """LEVEL is the flag: the clock sampled WHILE the GEMM ran against the
    clock the roof is quoted at. The gate text names it and the numbers."""
    cal = calibration(clocks={**LEGACY_CLOCKS, "throttled": False,
                              "sm_clock_load_mhz": 1400.0,
                              "reference_clock_mhz": 1755.0,
                              "clock_level_ok": False, "clock_drift_ok": True})
    gates = {name: (verdict, detail)
             for _k, name, verdict, detail in CH.score(cal, H200_PIN)}
    verdict, detail = gates["not_throttled"]
    assert verdict == EX.FAIL
    assert "scored LEVEL (clock_level_ok=False)" in detail
    assert "1400.0 MHz under load against reference 1755.0 MHz" in detail
    assert "ceilings are low" in detail
    # The retired flag is NOT what decided it: it says the opposite here.
    assert cal.clocks["throttled"] is False


def test_the_under_load_verdict_passes_a_card_at_its_reference_clock():
    """The PASS branch of the same flag, with the retired flag planted TRUE so
    that a gate still reading `throttled` would fail here."""
    cal = calibration(clocks={**LEGACY_CLOCKS, "clock_level_ok": True,
                              "clock_drift_ok": True, "sm_clock_load_mhz": 1755.0,
                              "reference_clock_mhz": 1755.0})
    gates = {name: (verdict, detail)
             for _k, name, verdict, detail in CH.score(cal, H200_PIN)}
    verdict, detail = gates["not_throttled"]
    assert verdict == EX.PASS and "scored LEVEL (clock_level_ok=True)" in detail
    assert cal.clocks["throttled"] is True


def test_drift_is_scored_only_when_level_has_no_reference():
    """A calibration that sampled under load but had no reference to level
    against still has DRIFT, and the gate says that is what it scored."""
    for drift, want in ((True, EX.PASS), (False, EX.FAIL)):
        cal = calibration(clocks={**LEGACY_CLOCKS, "clock_level_ok": None,
                                  "clock_drift_ok": drift,
                                  "sm_clock_start_mhz": 1755, "sm_clock_end_mhz": 1600})
        gates = {name: (verdict, detail)
                 for _k, name, verdict, detail in CH.score(cal, H200_PIN)}
        verdict, detail = gates["not_throttled"]
        assert verdict == want, drift
        assert f"scored DRIFT (clock_drift_ok={drift})" in detail
        assert "LEVEL undetermined" in detail


def test_a_calibration_that_predates_the_under_load_verdict_is_refused_not_scored():
    """The legacy shape: only the idle-instant pair and the retired flag.

    The gate scored `clocks["throttled"]` until 2026-09-03, and that field is
    `timing.clock_drift` over two samples taken between loads: it detects
    whether the first sample caught the idle boost, not throttling. The gate
    now REFUSES it (UNKNOWN, with the reason), and UNKNOWN counts against a
    CLAIM gate, so a sound-looking legacy calibration exits CLAIM_FAIL rather
    than DONE. Both legacy values of the retired flag land the same way,
    because neither is evidence.
    """
    for throttled in (True, False):
        cal = calibration(clocks={**LEGACY_CLOCKS, "throttled": throttled})
        gates = {name: (verdict, detail)
                 for _k, name, verdict, detail in CH.score(cal, H200_PIN)}
        verdict, detail = gates["not_throttled"]
        assert verdict == EX.UNKNOWN, throttled
        assert "predates the under-load clock verdict" in detail
        assert "NOT scored" in detail
        assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN)]) \
            == EX.CLAIM_FAIL


def test_flags_present_but_none_are_unknown_not_passed():
    """No usable under-load sample is not a pass, and the retired flag is not
    reached for as a substitute."""
    cal = calibration(clocks={**LEGACY_CLOCKS, "throttled": False,
                              "clock_level_ok": None, "clock_drift_ok": None})
    verdict, detail = CH.under_load_clock_verdict(cal)
    assert verdict == EX.UNKNOWN
    assert "both None" in detail and "NOT substituted" in detail


def test_the_under_load_verdict_is_read_from_the_gemm_clock_block_too():
    """`calibrate.py` may carry the loaded-clock record inside the GEMM's own
    `LoadedClock` block rather than beside the idle pair; both are read, in a
    stated order, and the detail names which."""
    cal = calibration(clocks={**LEGACY_CLOCKS},
                      gemm_clock={"median_mhz": 1755, "samples": 5, "spread_pct": 0.4,
                                  "after_idle_mhz": 1980, "clock_level_ok": False,
                                  "clock_drift_ok": True, "sm_clock_load_mhz": 1755.0,
                                  "reference_clock_mhz": 1900.0})
    verdict, detail = CH.under_load_clock_verdict(cal)
    assert verdict == EX.FAIL and "from gemm_clock." in detail


def test_every_gate_prints_one_result_line_and_nothing_else_does(capsys):
    """The driver greps `RESULT: ` at column zero and nothing else. A line that
    merely contains PASS is prose, and prose is what the pre-2026-09-02 summary
    grep was reading pre-registered expectations out of."""
    for kind, name, verdict, detail in CH.score(calibration(), H200_PIN):
        print(EX.result_line(kind, name, verdict, detail))
    printed = capsys.readouterr().out
    lines = EX.parse_result_lines(printed)
    assert len(lines) == len(CH.score(calibration(), H200_PIN))
    assert {ln.verdict for ln in lines} == {EX.PASS}


# --------------------------------------------------------------------------
# where it writes
# --------------------------------------------------------------------------

def test_the_default_output_path_is_untracked_and_carries_the_card_and_the_knobs():
    """R4. The path is under the results root, which `.gitignore` excludes, and
    its run id carries every swept knob so two settings of the same card cannot
    land on each other."""
    import argparse

    args = argparse.Namespace(buffer_gb=8.0, gemm_n=8192, ceiling="triad",
                              settle=True, settle_seconds=30.0,
                              results_root=REPO / "results")
    out = CH.session_out(args, "NVIDIA H200")
    assert "results" in out.parts and "calibration" in out.parts
    assert out.name == "measured_nvidia_h200.yaml"
    assert "nvidia_h200" in out.parent.name
    assert "buffer_gb8.0" in out.parent.name and "ceilingtriad" in out.parent.name
    other = CH.session_out(argparse.Namespace(**{**vars(args), "gemm_n": 4096}),
                           "NVIDIA H200")
    assert other.parent != out.parent, "a knob that changes the numbers must change the id"


def test_a_run_without_publish_leaves_the_tree_clean(tmp_path):
    """The requirement, tested the way it failed: run the script and ask git.

    Off a GPU it refuses before measuring, which is the only branch a laptop can
    take; the assertion that matters is the same one either way, that nothing
    was written into the checkout.
    """
    before = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "calibrate_hardware.py"),
                        "--results-root", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode in (EX.REFUSED, EX.DONE, EX.CLAIM_FAIL, EX.INVALID)
    after = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    assert after == before, "the calibration modified the working tree"


def test_without_a_gpu_it_refuses_rather_than_inventing_a_ceiling(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "calibrate_hardware.py"),
                        "--results-root", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode == EX.REFUSED:
        assert "REFUSED" in r.stderr
    else:                                       # pragma: no cover - needs a card
        pytest.skip("a GPU is present, so the refusal branch cannot be taken")


def test_the_dry_run_prints_the_plan_the_paths_and_an_mde(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "calibrate_hardware.py"),
                        "--dry-run", "--card", "NVIDIA H200",
                        "--results-root", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == EX.REFUSED, r.stdout + r.stderr
    assert "would write" in r.stdout
    assert "measured_nvidia_h200.yaml" in r.stdout
    assert "MDE:" in r.stdout and "sigma" in r.stdout
    assert not (tmp_path / "calibration").exists(), "a dry run wrote a file"


def test_the_dry_run_says_the_run_id_cannot_be_formed_without_a_card(tmp_path):
    """`provenance.run_id` raises `NoCard` rather than naming a run after a
    machine it cannot identify, and the plan says so instead of guessing."""
    import torch
    if torch.cuda.is_available():               # pragma: no cover - needs no card
        pytest.skip("a GPU is present, so the card is always visible here")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "calibrate_hardware.py"),
                        "--dry-run", "--results-root", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert "NoCard" in r.stdout, r.stdout


# --------------------------------------------------------------------------
# what it says it is
# --------------------------------------------------------------------------

def test_the_instrument_is_named_and_is_not_the_ladders_instrument():
    """Two instruments must not travel under one label. `time_kernel` times the
    cells and `moe.bench.calibrate` times the roof; the audit bounded the bias
    that difference introduces at 12-16% on the H200's smallest ladders, which
    is the size of the cross-card effect the study registers."""
    from moe.bench import calibrate as calibrate_mod
    from moe.bench import timing

    name = CH.instrument_name()
    assert name
    if getattr(calibrate_mod, "TIMING_BASIS", None) is None:
        assert timing.TIMING_BASIS not in name
        assert "time_eager" in name
    else:                                       # pragma: no cover - future phase
        assert name == calibrate_mod.TIMING_BASIS


def test_the_mde_refuses_when_there_is_no_measured_noise_level(tmp_path):
    """A minimum detectable effect computed from a number nobody measured is a
    prior dressed as a bound, which is what B14 found across the arms."""
    sigma, rows, why = CH.published_sigma_pct(tmp_path)
    assert sigma is None and rows == 0
    line = CH.mde_line(sigma, 3, why)
    assert line.startswith("MDE: REFUSED")


def test_the_mde_comes_from_the_published_rows_and_says_so():
    sigma, rows, why = CH.published_sigma_pct(REPO / "results" / "published")
    assert sigma and rows > 1000
    line = CH.mde_line(sigma, 3, why)
    assert "ms_std/ms_p50" in line
    assert "LOWER bound" in line, "within-cell spread understates the real noise"
    # 2.80 sigma / sqrt(trials), stated rather than asserted.
    assert f"{CH.MDE_Z * sigma / 3 ** 0.5:.2f}%" in line


def test_the_cells_written_beside_the_yaml_carry_the_instrument_and_no_invented_state(tmp_path):
    """`find results/published -name cells.csv` returned zero across every arm.
    The columns this instrument cannot report are EMPTY rather than filled with
    a plausible default, so a reader sees the gap instead of inheriting a
    guess."""
    import csv as _csv

    from moe.bench.provenance import provenance_block
    prov = provenance_block(instrument=CH.instrument_name(), iters=None)
    path = CH.write_cells(tmp_path / "cells.csv", calibration(), prov)
    rows = list(_csv.DictReader(path.open(newline="")))
    assert len(rows) == 4
    assert rows[0]["instrument"] == CH.instrument_name()
    assert rows[0]["l2_flush"] == "True"
    for unreported in ("warmup_ms", "iters", "trials", "clock_level_ok",
                       "clock_drift_ok", "sm_clock_load_mhz"):
        assert rows[0][unreported] == "", unreported
    assert rows[0]["prov_instrument"] == CH.instrument_name()
