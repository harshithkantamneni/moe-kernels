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
        clocks={"sm_start_mhz": 1755, "sm_end_mhz": 1750, "temp_start_c": 40,
                "temp_end_c": 45, "throttled": False},
        settle={"settled": True, "final_mhz": 1755},
        gemm_clock={"median_mhz": 1755, "samples": 5, "spread_pct": 0.4,
                    "after_idle_mhz": 1980},
    )
    fields.update(over)
    return Calibration(**fields)


#: The derived pin rate for an H200: 3201 MHz x 2 x 6144 bits / 8.
H200_PIN = 4916.7


# --------------------------------------------------------------------------
# the gates, both branches of each
# --------------------------------------------------------------------------

def verdicts(cal, pin=H200_PIN) -> dict[str, str]:
    return {name: verdict for _kind, name, verdict, _detail in CH.score(cal, pin)}


def test_a_sound_calibration_passes_every_gate_and_exits_done():
    cal = calibration()
    scored = CH.score(cal, H200_PIN)
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


def test_a_throttling_card_fails_its_claim():
    cal = calibration(clocks={"sm_start_mhz": 1755, "sm_end_mhz": 1200,
                              "temp_start_c": 40, "temp_end_c": 85,
                              "throttled": True})
    assert verdicts(cal)["not_throttled"] == EX.FAIL


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
