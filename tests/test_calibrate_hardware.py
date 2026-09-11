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
from moe.bench import timing as T  # noqa: E402
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

#: The maximum SM clock the FLOOR term of `not_throttled` is a fraction OF,
#: for the synthetic H200 fixture above. It is a PROPERTY OF THE PART, read
#: from the card by `timing.max_sm_clock_mhz` on a real run and never derived
#: from any calibration, which is why it may stand as a literal here where a
#: ridge or a ceiling may not. What must not be a literal is the FLOOR: it is
#: always `THERMAL_FLOOR_FRACTION` of whatever the device reported, and the
#: tests below assert that relation rather than 660.
H200_MAX_SM = 1980.0


# --------------------------------------------------------------------------
# the gates, both branches of each
# --------------------------------------------------------------------------

def verdicts(cal, pin=H200_PIN, max_sm=H200_MAX_SM) -> dict[str, str]:
    return {name: verdict
            for _kind, name, verdict, _detail in CH.score(cal, pin, max_sm)}


def details(cal, pin=H200_PIN, max_sm=H200_MAX_SM) -> dict[str, tuple[str, str]]:
    return {name: (verdict, detail)
            for _kind, name, verdict, detail in CH.score(cal, pin, max_sm)}


def test_a_sound_calibration_passes_every_gate_and_exits_done():
    """All six, named here so a gate added later cannot slip in untested: the
    docstring on `score` claimed five while emitting six, and the sixth had no
    FAIL branch anywhere in this file."""
    cal = calibration()
    scored = CH.score(cal, H200_PIN, H200_MAX_SM)
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
    assert EX.classify([(k, v) for k, _n, v, _d in
                        CH.score(cal, H200_PIN, H200_MAX_SM)]) == EX.INVALID


def test_no_settle_leaves_the_clock_unknown_which_still_counts_against_it():
    """UNKNOWN is not PASS. `--no-settle` measures from idle and walks the clock
    ramp across the patterns, so there is nothing to establish the clock
    against, and "could not decide" must not read as "fine"."""
    cal = calibration(settle={}, gemm_clock={})
    assert cal.clock_established is None
    assert verdicts(cal)["clock_established"] == EX.UNKNOWN
    assert EX.classify([(k, v) for k, _n, v, _d in
                        CH.score(cal, H200_PIN, H200_MAX_SM)]) == EX.INVALID


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
    assert EX.classify([(k, v) for k, _n, v, _d in
                        CH.score(cal, H200_PIN, H200_MAX_SM)]) == EX.CLAIM_FAIL


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
    assert EX.classify([(k, v) for k, _n, v, _d in
                        CH.score(cal, H200_PIN, H200_MAX_SM)]) == EX.CLAIM_FAIL
    detail = {n: d for _k, n, _v, d in CH.score(cal, H200_PIN, H200_MAX_SM)}
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
    scored = dict((n, v) for _k, n, v, _d in CH.score(cal, None, H200_MAX_SM))
    assert scored["no_pattern_exceeds_the_pin_rate"] == EX.UNKNOWN
    # And only that one: an absent bus width says nothing about the clock.
    assert scored["not_throttled"] == EX.PASS
    assert EX.classify([(k, v) for k, _n, v, _d
                        in CH.score(cal, None, H200_MAX_SM)]) == EX.CLAIM_FAIL


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
    verdict, detail = details(cal)["not_throttled"]
    assert verdict == EX.FAIL
    assert ("LEVEL FAIL (clock_level_ok=False, 1400.0 MHz against reference "
            "1755.0 MHz)") in detail
    assert "ceilings are low" in detail
    # 1400 of 1980 is well above the thermal floor, so the FLOOR term PASSES
    # here and LEVEL is what failed. The two terms are independent questions
    # and the detail names both.
    assert "FLOOR PASS" in detail and "DRIFT PASS" in detail
    # The retired flag is NOT what decided it: it says the opposite here.
    assert cal.clocks["throttled"] is False


def test_the_under_load_verdict_passes_a_card_at_its_reference_clock():
    """The PASS branch of the same flag, with the retired flag planted TRUE so
    that a gate still reading `throttled` would fail here."""
    cal = calibration(clocks={**LEGACY_CLOCKS, "clock_level_ok": True,
                              "clock_drift_ok": True, "sm_clock_load_mhz": 1755.0,
                              "reference_clock_mhz": 1755.0})
    verdict, detail = details(cal)["not_throttled"]
    assert verdict == EX.PASS and "LEVEL PASS (clock_level_ok=True" in detail
    assert cal.clocks["throttled"] is True


def test_floor_and_drift_carry_the_gate_when_level_has_no_reference():
    """A calibration has no reference to LEVEL against -- it IS the reference --
    so on everything `calibrate.py` writes the gate rests on the other two
    terms, and the detail names both of them and their numbers.

    UNTIL 2026-09-11 IT RESTED ON DRIFT ALONE, and the DRIFT row below is
    exactly the shape that let a floored card through: a steady clock passes
    however low it is. FLOOR is scored beside it now, so the PASS row here is
    a card that is both steady AND clocking, which is what the ceilings need.
    """
    for drift, want in ((True, EX.PASS), (False, EX.FAIL)):
        cal = calibration(clocks={**LEGACY_CLOCKS, "clock_level_ok": None,
                                  "clock_drift_ok": drift,
                                  "sm_clock_load_mhz": 1470.0,
                                  "sm_clock_start_mhz": 1755, "sm_clock_end_mhz": 1600})
        verdict, detail = details(cal)["not_throttled"]
        assert verdict == want, drift
        assert f"DRIFT {'PASS' if drift else 'FAIL'} (clock_drift_ok={drift}" in detail
        assert "1755 -> 1600 MHz" in detail
        # FLOOR is a fraction of the card's own maximum, never a literal.
        assert "FLOOR PASS (1470.0 MHz median under load against a floor of " in detail
        assert f"{T.thermal_floor_mhz(H200_MAX_SM):.0f} MHz" in detail
        assert "LEVEL" not in detail, "a reference nobody supplied is not a term"


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
                 for _k, name, verdict, detail in CH.score(cal, H200_PIN, H200_MAX_SM)}
        verdict, detail = gates["not_throttled"]
        assert verdict == EX.UNKNOWN, throttled
        assert "predates the under-load clock verdict" in detail
        assert "NOT scored" in detail
        assert EX.classify([(k, v) for k, _n, v, _d in CH.score(cal, H200_PIN, H200_MAX_SM)]) \
            == EX.CLAIM_FAIL


def test_flags_present_but_none_are_unknown_not_passed():
    """No usable under-load sample is not a pass, and the retired flag is not
    reached for as a substitute. Neither FLOOR nor DRIFT can be scored from a
    record with no samples in it, and the detail says so of each."""
    cal = calibration(clocks={**LEGACY_CLOCKS, "throttled": False,
                              "clock_level_ok": None, "clock_drift_ok": None})
    verdict, detail = CH.under_load_clock_verdict(cal, H200_MAX_SM)
    assert verdict == EX.UNKNOWN
    assert "FLOOR NOT SCORED" in detail and "DRIFT NOT SCORED" in detail
    assert "was not tested" in detail and "NOT substituted" in detail


def test_a_calibration_with_no_maximum_clock_leaves_the_floor_untested():
    """THE ARGUMENT IS OPTIONAL AND ITS ABSENCE IS NEVER A PASS. A caller that
    does not supply the card's maximum has not asked the one question that
    catches a floored card, and an untested claim has to be visible in the
    exit code: UNKNOWN, which `classify` counts against the gate. This is the
    same rule `no_pattern_exceeds_the_pin_rate` takes for a missing bus width.
    """
    cal = calibration()
    verdict, detail = CH.under_load_clock_verdict(cal)          # no maximum
    assert verdict == EX.UNKNOWN
    assert "maximum SM clock could not be read" in detail
    assert "LEVEL PASS" in detail and "DRIFT PASS" in detail, (
        "the other two terms still passed; it is the untested one that decides")
    assert CH.under_load_clock_verdict(cal, H200_MAX_SM)[0] == EX.PASS


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
    for kind, name, verdict, detail in CH.score(calibration(), H200_PIN, H200_MAX_SM):
        print(EX.result_line(kind, name, verdict, detail))
    printed = capsys.readouterr().out
    lines = EX.parse_result_lines(printed)
    assert len(lines) == len(CH.score(calibration(), H200_PIN, H200_MAX_SM))
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


# --------------------------------------------------------------------------
# the producer writes the block the scorer reads (2026-09-09, the pod's arm 0)
# --------------------------------------------------------------------------


def test_the_loaded_clock_record_carries_the_under_load_verdict_the_gate_scores():
    """2026-09-09, first H200 session on the merged tree: arm 0 measured the
    clock under load (1470 MHz, five NVML samples, spread 2.0%) and its own
    `not_throttled` gate came back UNKNOWN, 'predates the under-load clock
    verdict', because `LoadedClock.as_dict` wrote samples, median and spread
    and never the `clock_drift_ok` / `clock_level_ok` keys the scorer reads.
    CLAIM_FAIL, session refused, on a calibration that was sound: the
    sixteenth instance of a fix at one of two sites. The record now carries
    the verdict in `timing.clock_flags`' vocabulary: DRIFT scored from the
    first and last under-load samples, LEVEL None because this record IS the
    reference a roof is quoted at. Planted in the pod's exact shape: the
    legacy idle pair in `clocks`, the LoadedClock block in `gemm_clock`."""
    from moe.bench.calibrate import LoadedClock

    steady = LoadedClock(label="bf16 GEMM", samples=(1485, 1485, 1455, 1470, 1470),
                         median_mhz=1470, spread_pct=2.0, after_idle_mhz=1470)
    d = steady.as_dict()
    assert d["clock_drift_ok"] is True and d["clock_level_ok"] is None
    assert (d["sm_clock_load_mhz"], d["sm_clock_start_mhz"],
            d["sm_clock_end_mhz"]) == (1470, 1485, 1470)
    cal = calibration(clocks={**LEGACY_CLOCKS}, gemm_clock=d)
    verdict, detail = details(cal)["not_throttled"]
    assert verdict == EX.PASS
    assert "from gemm_clock." in detail
    assert "DRIFT PASS (clock_drift_ok=True" in detail and "1485 -> 1470 MHz" in detail
    assert "FLOOR PASS (1470 MHz median under load" in detail
    # Scored, not refused: no gate is UNKNOWN. (clock_established FAILs on the
    # legacy fixture pair; on the pod it PASSED, and it is not this gate.)
    assert all(v != EX.UNKNOWN for _k, _n, v, _d in CH.score(cal, H200_PIN, H200_MAX_SM))


def test_a_loaded_clock_that_drifted_under_load_fails_the_gate_and_an_empty_one_is_unknown():
    """The FAIL branch the same record can take, and the no-sample record,
    which the scorer must read as UNKNOWN and never as a pass."""
    from moe.bench.calibrate import LoadedClock

    drifted = LoadedClock(label="bf16 GEMM", samples=(1980, 1800, 1600, 1500, 1470),
                          median_mhz=1600, spread_pct=26.0, after_idle_mhz=1470)
    cal = calibration(clocks={**LEGACY_CLOCKS}, gemm_clock=drifted.as_dict())
    verdict, detail = CH.under_load_clock_verdict(cal, H200_MAX_SM)
    assert verdict == EX.FAIL
    assert "1980 -> 1470 MHz" in detail and "blend of two states" in detail
    empty = LoadedClock(label="bf16 GEMM", samples=(), median_mhz=0,
                        spread_pct=0.0, after_idle_mhz=0)
    d = empty.as_dict()
    assert d["clock_drift_ok"] is None and d["clock_level_ok"] is None
    cal = calibration(clocks={**LEGACY_CLOCKS}, gemm_clock=d)
    verdict, detail = CH.under_load_clock_verdict(cal, H200_MAX_SM)
    assert verdict == EX.UNKNOWN
    assert "FLOOR NOT SCORED" in detail and "DRIFT NOT SCORED" in detail



def test_the_pin_rate_is_built_from_the_enabled_bus_nvml_reports():
    """2026-09-09 H200 pod: NVML reported a 6016-bit bus, 6144 x 141/144, the
    same harvest as the 141-of-144 GB capacity. 3201 MHz x 2 x 6016 / 8 =
    4814.3 GB/s is NVIDIA's 4.8 TB/s to 0.3%. The table's 6144 gave 4916.7,
    and this file then called the datasheet "already derated". The datasheet
    was the pin rate. NVML first, the table as the fallback with its source
    named, and a disagreement recorded beside the device's number, never
    averaged and never resolved in the table's favour."""
    bits, source, table = CH.resolve_memory_bus_bits("NVIDIA H200", 6016)
    assert (bits, source) == (6016, "nvml")
    assert CH.pin_rate_gbps(3201, bits) == 4814.3
    assert abs(CH.pin_rate_gbps(3201, bits) - 4800.0) / 4800.0 < 0.005
    assert CH.pin_rate_gbps(3201, 6144) == 4916.7
    bits, source, table = CH.resolve_memory_bus_bits("NVIDIA H200", None)
    assert (bits, source, table) == (CH._MEMORY_BUS_BITS["H200"], "table", None)
    assert CH._MEMORY_BUS_BITS["H200"] == 6016
    bits, source, table = CH.resolve_memory_bus_bits("NVIDIA A100-SXM4-80GB", 5120)
    assert (bits, source, table) == (5120, "nvml", None)
    bits, source, table = CH.resolve_memory_bus_bits("NVIDIA A100-SXM4-80GB", 5000)
    assert (bits, source, table) == (5000, "nvml", 5120)
    assert CH.resolve_memory_bus_bits("Unknown GPU", None) == (None, "none", None)
    # Off a GPU the NVML reader answers None and never raises.
    got = CH._nvml_memory_bus_bits()
    assert got is None or (isinstance(got, int) and got > 0)


# --------------------------------------------------------------------------
# the card that ran flat at its floor (2026-09-11) and the term that catches it
# --------------------------------------------------------------------------


def floored_calibration():
    """The 2026-09-11 pod, planted in the shape `calibrate.py` would have
    written it: the GEMM sampled five times at the card's 345 MHz floor.

    Nothing here is invented. That pod boosted to 1980 MHz, collapsed inside
    ~30 s of sustained bf16 GEMM, stayed at 345 and drew ~240 W of a 700 W
    limit while climbing from 87 C to 93 C."""
    from moe.bench.calibrate import LoadedClock

    floored = LoadedClock(label="bf16 GEMM", samples=(345, 345, 345, 345, 345),
                          median_mhz=345, spread_pct=0.0, after_idle_mhz=345,
                          temp_c=93, power_w=240.0)
    # THE SETTLE COLLAPSED WITH IT, which is the whole difficulty: every other
    # gate on the page agrees with itself. `clock_established` compares the
    # GEMM's median against the settle plateau, and on a card that fell before
    # the settle finished they are both 345, so that VALIDITY gate PASSES and
    # the page looks complete. Planting a healthy settle here would let this
    # test pass for the wrong reason -- INVALID on `clock_established` rather
    # than CLAIM_FAIL on the term this slice added.
    return calibration(clocks={**LEGACY_CLOCKS}, gemm_clock=floored.as_dict(),
                       settle={"settled": True, "final_mhz": 345},
                       gemm_clock_mhz=345)


def test_a_card_flat_at_its_floor_fails_the_gate_that_used_to_pass_it():
    """THE MOTIVATING EVENT, both halves in one test.

    The OLD rule is planted first and shown to pass, because a claim that the
    gate had a hole is worth nothing unless the hole is demonstrated: DRIFT
    over this record is True, since first and last samples are both 345. That
    is what `not_throttled` scored, and it is why a pod that could not hold a
    clock published a tracked ruler whose ridge read 73.6 against a real ~156.

    The FLOOR term is what fails it, and a FAIL is a RESULT: CLAIM_FAIL, which
    the session driver latches and refuses on, not INVALID and not a retry."""
    cal = floored_calibration()
    # The hole, reproduced: DRIFT alone says this card is fine.
    assert cal.gemm_clock["clock_drift_ok"] is True
    assert cal.gemm_clock["clock_level_ok"] is None
    assert T.clock_flags(345, 345, 345, None) == (None, True)

    verdict, detail = CH.under_load_clock_verdict(cal, H200_MAX_SM)
    assert verdict == EX.FAIL
    assert "FLOOR FAIL (345 MHz median under load against a floor of " in detail
    assert "DRIFT PASS" in detail, "the term that used to decide still passes"
    assert "pinned near its own clock floor" in detail
    assert "the ridge derived from them is wrong" in detail
    scored = CH.score(cal, H200_PIN, H200_MAX_SM)
    by_name = dict((n, v) for _k, n, v, _d in scored)
    assert by_name["not_throttled"] == EX.FAIL
    # EVERY OTHER GATE ON THIS PAGE PASSES, which is the point. The pod that
    # motivated this ran to completion and published; only the term added here
    # separates it from a sound calibration.
    assert by_name["clock_established"] == EX.PASS
    assert {n for n, v in by_name.items() if v != EX.PASS} == {"not_throttled"}
    assert EX.classify([(k, v) for k, _n, v, _d in scored]) == EX.CLAIM_FAIL
    # And with the term absent -- a run that could not read the maximum -- the
    # page is UNKNOWN, which still counts against the gate. Never PASS.
    assert CH.under_load_clock_verdict(cal)[0] == EX.UNKNOWN


def test_the_floor_is_a_fraction_of_the_cards_own_maximum_and_never_a_literal():
    """R2. The same 345 MHz record is a FAIL on a 1980 MHz part and a PASS on
    a part whose maximum is low enough, because the floor moves with the card.
    A gate carrying 660 as a number would be a gate that works on one part."""
    cal = floored_calibration()
    # THE FRACTION IS THE CONSTANT AND NOT THE DIGIT 3. These two lines read
    # `1980.0 / 3` until 2026-09-11, which pinned THERMAL_FLOOR_FRACTION inside
    # the one test named "and never a literal": retune it anywhere in its own
    # admissible window (0.1742 < f <= 0.6439) and this test failed on correct
    # code, pointing at the arithmetic instead of at the retune.
    for maximum in (1980.0, 1410.0):
        assert T.thermal_floor_mhz(maximum) == T.snap_to_clock_step(
            maximum * T.THERMAL_FLOOR_FRACTION)
    assert T.thermal_floor_mhz(1980.0) > T.thermal_floor_mhz(1410.0)
    fault, fault_max = T.THERMAL_FAULT_OBSERVED_MHZ
    assert CH.under_load_clock_verdict(cal, fault_max)[0] == EX.FAIL
    # A hypothetical part small enough that the SAME 345 MHz record clears its
    # floor, derived so it stays a PASS at any admissible fraction: a maximum
    # whose floor is 0.9 x the recorded fault clock.
    low_part = 0.9 * fault / T.THERMAL_FLOOR_FRACTION
    assert T.thermal_floor_mhz(low_part) < fault
    assert CH.under_load_clock_verdict(cal, low_part)[0] == EX.PASS
    # Every edge NVML can report is a clock NVML can report.
    for maximum in (1410.0, 1755.0, 1980.0, 2100.0):
        assert T.thermal_floor_mhz(maximum) % T.CLOCK_STEP_MHZ == 0


def test_the_floor_fraction_separates_the_published_rows_from_the_fault():
    """WHERE THE FRACTION CAME FROM, re-derived from the committed corpus on
    every run rather than believed from the constant's own comment.

    The binding healthy end is the LOWEST per-cell `sm_clock_load_mhz` median
    in `results/published`, as a fraction of the highest clock the same corpus
    ever recorded, which on this card is its maximum. The fault end is the
    2026-09-11 pair, recorded in `timing.THERMAL_FAULT_OBSERVED_MHZ` and never
    read at gate time. The fraction has to sit strictly between them with real
    margin on both sides, or it either refuses healthy cards or admits the
    fault."""
    import csv
    import hashlib

    published = REPO / "results" / "published"
    if not published.is_dir():
        pytest.skip("no published corpus in this checkout")
    seen, loads = set(), []
    for path in sorted(published.rglob("cells.csv")):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:          # the 2026-09-10 session duplicates its tree
            continue
        seen.add(digest)
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    mhz = float(row.get("sm_clock_load_mhz") or 0)
                except ValueError:
                    continue
                if mhz > 0:
                    loads.append(mhz)
    assert len(loads) > 1000, f"too few rows to derive a bound from: {len(loads)}"
    healthy = min(loads) / max(loads)
    fault_mhz, fault_max = T.THERMAL_FAULT_OBSERVED_MHZ
    fault = fault_mhz / fault_max
    assert fault < T.THERMAL_FLOOR_FRACTION < healthy, (fault, healthy)
    # Margin, as a ratio, on each side. 1.5x is well inside the 1.93 and 1.91
    # the constant's comment claims, so this fails on a drift rather than on
    # a rounding.
    assert healthy / T.THERMAL_FLOOR_FRACTION > 1.5, healthy
    assert T.THERMAL_FLOOR_FRACTION / fault > 1.5, fault
    # And no published row would have been refused by it.
    floor = T.thermal_floor_mhz(max(loads))
    assert min(loads) > floor, (min(loads), floor)
    # THE RECORDED HEALTHY PAIR IS THAT SAME BINDING END, so the margin
    # `thermal_acceptance` prints is the margin this corpus actually holds.
    assert T.THERMAL_HEALTHY_OBSERVED_MHZ == (min(loads), max(loads))
    # NOR WOULD ANY REJECTED FRACTION HAVE REFUSED ONE, which the constant's
    # comment claimed of f = 0.60 until 2026-09-11. The reason to reject 0.60
    # is its margin, not a cell it would have failed, and a comment that gives
    # the wrong reason is the sentence a later retune gets argued from.
    for rejected in (0.25, 0.50, 0.60):
        refused = [m for m in loads
                   if m < T.snap_to_clock_step(rejected * max(loads))]
        assert refused == [], (rejected, len(refused), min(refused))


def test_the_a100_margin_in_the_constants_comment_comes_out_of_the_a100_file():
    """R2, asked of the one cross-card number the derivation quotes.

    `THERMAL_FLOOR_FRACTION`'s comment reaches for a SECOND part to show the
    fraction is not H200-shaped, and the clock it quoted was 1230 MHz, which
    appears in no committed file: the A100's own calibration records a
    compute-settle history whose minimum is 1245, so the margin was 2.68x and
    not the 2.65x written down. A derived quantity typed as a literal, drifted
    from its source before it was ever committed."""
    import yaml as yaml_mod

    path = (REPO / "moe" / "bench" / "hardware"
            / "measured_nvidia_a100_sxm4_80gb.yaml")
    if not path.exists():
        pytest.skip("no committed A100 calibration in this checkout")
    history = (yaml_mod.safe_load(path.read_text())["detail"]["settle"]
               or {}).get("clock_history_mhz") or []
    assert history, "the committed A100 file carries no compute-settle history"
    a100_max = 1410.0          # a property of the PART, never of a calibration
    margin = min(history) / T.thermal_floor_mhz(a100_max)
    source = (REPO / "moe" / "bench" / "timing.py").read_text()
    assert f"{min(history):.0f} MHz" in source, min(history)
    assert f"{margin:.2f}x" in source, margin
    # and the number it used to carry is gone from the tree entirely.
    assert "1230 MHz" not in source


def test_the_floor_is_scored_on_the_median_and_not_on_a_sample():
    """A healthy card posts individual samples far below the floor during one
    drain-and-ramp: the published corpus holds entries at 405 MHz on a part
    whose maximum is 1980, inside cells whose medians and DRIFT verdicts are
    sound. `clock_floor_ok` takes the median, and a record built from such a
    trace passes."""
    from moe.bench.calibrate import LoadedClock

    excursion = LoadedClock(
        label="bf16 GEMM", samples=(1425, 1410, 405, 825, 1365, 1410, 1425),
        median_mhz=1410, spread_pct=71.6, after_idle_mhz=1425)
    d = excursion.as_dict()
    assert min(d["samples"]) < T.thermal_floor_mhz(H200_MAX_SM)
    assert T.clock_floor_ok(d["sm_clock_load_mhz"], H200_MAX_SM) is True


def test_both_call_sites_of_the_verdict_are_given_the_maximum_clock():
    """RULE 3, asked of the source. `under_load_clock_verdict` is called twice
    in this file -- once by `score` for the RESULT line and once by `main` for
    the printed page -- and a fix landing at one of two call sites is this
    repository's recurring defect, nineteen instances deep. A call that
    forgets the maximum is not a crash, it is a page that silently scores one
    term fewer than the line beside it."""
    import ast

    source = (REPO / "scripts" / "calibrate_hardware.py").read_text()
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "under_load_clock_verdict"]
    assert len(calls) == 2, [ast.unparse(c) for c in calls]
    for call in calls:
        assert len(call.args) + len(call.keywords) == 2, ast.unparse(call)
    # and the printed page's FAIL advice names the floored card, not a settle.
    assert "THIS CARD CANNOT HOLD A CLOCK" in source
    assert "let it settle and re-run" in source, "the DRIFT advice still stands"


def test_the_committed_calibration_in_this_tree_still_passes_the_new_gate():
    """THE OTHER DIRECTION, and it is the expensive one to get wrong: a term
    added to catch a sick card must not start refusing the healthy ones this
    repository has already published against.

    Read out of `moe/bench/hardware/measured_nvidia_h200.yaml` rather than
    asserted here, which is R2: this card's GEMM clock has read different
    numbers in different sessions, and a test that carried one of them would
    fail the next time the card is recalibrated for a reason that has nothing
    to do with this gate."""
    import types

    import yaml as yaml_mod

    path = REPO / "moe" / "bench" / "hardware" / "measured_nvidia_h200.yaml"
    if not path.exists():
        pytest.skip("no committed H200 calibration in this checkout")
    loaded = yaml_mod.safe_load(path.read_text())
    detail = loaded["detail"]
    cal = types.SimpleNamespace(clocks=detail.get("clocks") or {},
                                gemm_clock=detail.get("gemm_clock") or {})
    median = cal.gemm_clock.get("sm_clock_load_mhz")
    assert median, "the committed file carries no under-load median to score"
    # THE MAXIMUM COMES OUT OF THE SAME FILE WHEN THE FILE HAS IT. Only the
    # median was read here until 2026-09-11, and the denominator of the whole
    # threshold was the module literal -- in a test whose docstring says it
    # reads the file "which is R2". `calibrate_hardware` now writes
    # `observed.clocks_max_sm_mhz` on every run, so the next recalibration
    # supplies it; the committed file predates that key, hence the fallback,
    # and the literal is admissible only because a maximum SM clock is a
    # property of the PART and never derived from a calibration. It matters on
    # an H200 NVL, which torch names "NVIDIA H200" identically and which
    # h200_nvl.yaml documents as a clock-cut part: a recalibration there lands
    # in this same filename against another SKU's maximum.
    max_sm = (loaded.get("observed") or {}).get("clocks_max_sm_mhz") or H200_MAX_SM
    verdict, detail_text = CH.under_load_clock_verdict(cal, max_sm)
    assert verdict == EX.PASS, detail_text
    # The relation, not the number: whatever that file says, it is above a
    # third of the card's maximum with the margin the constant was sized for.
    assert median >= T.thermal_floor_mhz(max_sm)
    assert median / T.thermal_floor_mhz(max_sm) > 1.5, median
