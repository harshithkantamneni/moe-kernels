"""The surface must be sized against the floor the card measured, not the proxy.

`scripts/alpha_surface.py` prints one MDE before every paired lever comparison,
and that MDE is only as honest as the sigma under it. Until 2026-09-03 the
sigma came from `payload["prior_sd"]` read straight out of
`results/published/NOISE_FLOOR.json`. That field is the s3-vs-s4 PROXY: a
paired sd over 11 matched cells divided by sqrt(2), an upper bound because the
two arms differ in `num_stages` as well as in nothing. Part (a) of the
noise-floor arm spends 120 minutes of card measuring a real between-replicate
spread and publishes it into `replicate_floor` of the SAME file, leaving
`prior_sd` untouched -- so this script would have gone on printing the
assumption, and every "below the detection limit" verdict it issues would have
been scored against the wrong denominator.

`replicate_noise_floor.sizing_sigma` is the one accessor that prefers the
measurement and falls back to the proxy with the word ASSUMED. THE TESTS BELOW
PLANT BOTH BRANCHES, because a consumer that only ever reads one of them proves
nothing: a measured floor must come back MEASURED and move the MDE, a file with
no floor must come back ASSUMED at the declared number, and a REHEARSAL floor
must not be mistaken for either.

BOTH BRANCHES ARE PLANTED, INCLUDING THE ASSUMED ONE. The first version of this
file asserted the ASSUMED state off the TRACKED
`results/published/NOISE_FLOOR.json`, which reads ASSUMED only because part (a)
has not run yet. That is a fact about the calendar, not about this script:
three of these tests would have gone red the instant the arm they exist to
serve published its result, and the owner would have met them on return from
the pod. The tracked file is now read for the DECLARED PRIOR it carries, which
is pinned and does not move, and the floor state under test is grafted in both
directions by `_measured_floor_file` and `_unmeasured_floor_file`.

The script is loaded by path, because `scripts/` is not a package.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "published"
H200_S4 = PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"
TRACKED_FLOOR = PUBLISHED / "NOISE_FLOOR.json"

#: The floor the reviewer simulated: well below the 0.0229 proxy, so a reader
#: cannot confuse the two numbers by eye and neither can an assertion.
MEASURED_SD = 0.0091


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec, for the reason `test_bn_decomposition` gives:
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AS = _load("alpha_surface", "alpha_surface.py")


def _write(where: Path, doc: dict) -> Path:
    """Each graft in its OWN directory under the same basename.

    Two grafts writing one `tmp_path / "NOISE_FLOOR.json"` means the second
    silently overwrites the first, and a test holding both then passes only
    because Python evaluates arguments left to right. The basename stays real
    because the error paths print it.
    """
    where.mkdir(parents=True, exist_ok=True)
    path = where / "NOISE_FLOOR.json"
    path.write_text(json.dumps(doc))
    return path


def _measured_floor_file(tmp_path: Path, *, sd: float = MEASURED_SD,
                         synthetic: bool = False) -> Path:
    """The tracked file with a replicate floor grafted into it.

    Built FROM the tracked document rather than from a hand-written stub, so
    the schema, the declared prior and its source are the real ones and the
    only thing under test is which of the two this script reads.
    """
    doc = json.loads(TRACKED_FLOOR.read_text())
    doc["replicate_floor"] = {
        "n_replicates": 5, "cache_mode": "flush", "gpu_name": "NVIDIA H200",
        "provenance": "simulated part (a), planted by the test suite",
        "instrument": "queue-deep/l2-flush/clock-under-load/v2",
        "scope": None, "synthetic": synthetic,
        "per_field": {"alpha_corrected": {
            "sd": sd, "df": 8, "upper95": sd * 1.6, "pooled": True,
            "reason": "planted", "cells": 3, "per_cell": []}},
    }
    return _write(tmp_path / "measured", doc)


def _unmeasured_floor_file(tmp_path: Path) -> Path:
    """The tracked file with its replicate floor NULLED. The other graft.

    THE STATE, NOT THE CALENDAR. Reading the tracked file directly for the
    ASSUMED branch tests "part (a) has not run yet", which stops being true the
    day the arm publishes and takes three assertions with it. The declared
    prior is what these tests are actually about and it is carried through
    unchanged, so `prior_sd` and `prior_sd_source` are still the repo's real
    ones and the number asserted below is still read from the tracked file.
    """
    doc = json.loads(TRACKED_FLOOR.read_text())
    doc["replicate_floor"] = None
    return _write(tmp_path / "unmeasured", doc)


#: The declared prior the ASSUMED branch must fall back to, read from the
#: tracked file rather than typed here: it is pinned, so it is safe to read.
DECLARED_SD = json.loads(TRACKED_FLOOR.read_text())["prior_sd"]


# --------------------------------------------------------------------------
# Which sigma, and the word that says which.
# --------------------------------------------------------------------------

def test_a_measured_floor_is_read_and_labelled_measured(tmp_path):
    """The 120 minutes of card have to reach this script, or they bought it
    nothing."""
    sd, basis, source = AS.prior_sd(_measured_floor_file(tmp_path))
    assert basis == "MEASURED"
    assert sd == pytest.approx(MEASURED_SD)
    assert "simulated part (a)" in source


def test_no_floor_falls_back_to_the_declared_proxy_and_says_assumed(tmp_path):
    """A document with no measured floor falls back, and must SAY so.

    Planted rather than read off the tracked file: this asserts the state, and
    the tracked file's state changes the day part (a) publishes.
    """
    sd, basis, source = AS.prior_sd(_unmeasured_floor_file(tmp_path))
    assert basis == "ASSUMED"
    assert sd == pytest.approx(DECLARED_SD)
    assert "s3-vs-s4" in source


def test_a_rehearsal_floor_is_not_a_measurement(tmp_path):
    """`--publish` can write a synthetic floor to test the plumbing. Reading it
    as MEASURED would be the worst of the three outcomes: a number nobody
    measured, wearing the word that says somebody did."""
    sd, basis, _ = AS.prior_sd(_measured_floor_file(tmp_path, synthetic=True))
    assert basis == "ASSUMED"
    assert sd != pytest.approx(MEASURED_SD)


def test_a_missing_file_states_no_mde_rather_than_inventing_one(tmp_path):
    sd, basis, source = AS.prior_sd(tmp_path / "absent.json")
    assert (sd, basis) == (None, "NONE")
    assert "does not exist" in source


def test_a_truncated_floor_file_leaves_the_table_readable(tmp_path):
    """`sizing_sigma` parses the JSON before its own guards run, so a half
    written file arrives as a decode error rather than as its own exception.
    A surface that dies on it prints no table at all."""
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text('{"schema": "moe-kernels/noise-floor/2", "prior_')
    sd, basis, source = AS.prior_sd(path)
    assert (sd, basis) == (None, "NONE")
    assert "unreadable" in source


# --------------------------------------------------------------------------
# The word has to reach the page. An operator reads the printout, not the JSON.
# --------------------------------------------------------------------------

def test_the_printed_mde_carries_the_word_in_both_states(tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(AS, "NOISE_FLOOR", _measured_floor_file(tmp_path))
    AS.print_mde(3)
    measured = capsys.readouterr().out
    assert "MEASURED sd of one alpha 0.0091" in measured
    assert "ASSUMED" not in measured

    monkeypatch.setattr(AS, "NOISE_FLOOR", _unmeasured_floor_file(tmp_path))
    AS.print_mde(3)
    assumed = capsys.readouterr().out
    assert f"ASSUMED sd of one alpha {DECLARED_SD:.4f}" in assumed
    assert "and it is an upper bound" in assumed
    assert "MEASURED" not in assumed


def test_a_measured_floor_moves_the_detection_limit(tmp_path, monkeypatch,
                                                    capsys):
    """The word alone is not the point. A tighter floor resolves smaller lever
    effects, and the number on the page has to move with it."""
    def mde_from(path):
        monkeypatch.setattr(AS, "NOISE_FLOOR", path)
        AS.print_mde(3)
        line = capsys.readouterr().out.splitlines()[0]
        return float(line.split("MDE")[1].split()[0])

    assert (mde_from(_measured_floor_file(tmp_path))
            < mde_from(_unmeasured_floor_file(tmp_path)))


def test_no_noise_model_is_stated_as_such_and_no_number_is_printed(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(AS, "NOISE_FLOOR", tmp_path / "absent.json")
    AS.print_mde(3)
    out = capsys.readouterr().out
    assert "MDE: NOT STATED" in out
    assert "ASSUMED" not in out and "MEASURED" not in out


def test_the_whole_table_says_which_floor_it_was_read_against(monkeypatch,
                                                              capsys, tmp_path):
    """End to end over a committed arm: the surface an operator actually reads."""
    monkeypatch.setattr(AS, "NOISE_FLOOR", _measured_floor_file(tmp_path))
    monkeypatch.setattr(sys, "argv", ["alpha_surface.py", str(H200_S4)])
    assert AS.main() == 0
    out = capsys.readouterr().out
    assert "MEASURED sd of one alpha" in out
    assert "ALPHA SURFACE" in out


# --------------------------------------------------------------------------
# The two defects the file's own docstring names, kept under test.
# --------------------------------------------------------------------------

def test_both_committed_report_layouts_are_found():
    """The published arms hold `<cell>.report.json`; a single-cell run writes a
    bare `report.json`. Globbing for the second alone printed "no report.json"
    on every published arm."""
    found = AS.report_paths(H200_S4)
    assert found
    assert len(set(found)) == len(found)
    assert all(p.name.endswith("report.json") for p in found)


def test_levels_sort_numerically_so_the_extremes_are_the_real_ones():
    """Sorted as text, GROUP_SIZE_M ran 1, 16, 64, 8 and the paired-change line
    named 1 -> 8 as the swizzle's extremes when the sweep runs 1 -> 64."""
    assert sorted([1, 16, 64, 8], key=AS.level_sort_key) == [1, 8, 16, 64]
    mixed = sorted([64, "mixtral", None, 1], key=AS.level_sort_key)
    assert mixed == [1, 64, "mixtral", None]


# --------------------------------------------------------------------------
# The weight-stream columns. Added 2026-09-10: `report.json` had carried `w`
# per ladder since the estimator landed and no table printed it, so the one
# statistic with no fitted level in it was invisible to every reader of a
# surface.
# --------------------------------------------------------------------------

def _corpus(tmp_path: Path, ladders: dict) -> Path:
    """One synthetic arm holding one report, with the ladder rows given."""
    root = tmp_path / "arm"
    root.mkdir(parents=True)
    (root / "cell.report.json").write_text(json.dumps({
        "model": "mixtral-8x7b",
        "fixed": {"GROUP_SIZE_M": 16, "BLOCK_SIZE_N": 64},
        "compute_reference": {"refusals": [], "refused_block_m": None},
        "ladder": ladders,
    }))
    return root


def _surface(root: Path) -> str:
    got = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "alpha_surface.py"), str(root)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert got.returncode == 0, got.stderr[-2000:]
    return got.stdout


def test_the_ladder_table_prints_w_and_keeps_its_two_absences_apart(tmp_path):
    """THREE FACTS, THREE RENDERINGS, and they were one blank.

    A `w` key ABSENT means the report predates the statistic. A `w` key present
    and null means that fit HAS no w, for a reason its own report records. A
    blank in the alpha columns means the fit was not identifiable. Rendering
    the first as the third is how a corpus that simply predates a column comes
    to read as a corpus of unidentifiable fits.

    `w` is also NOT gated on identifiability, and that is the point of it: it
    has no fitted level, no intercept, no delta and no D, so a ladder whose
    alpha is blank for want of treads can still carry one. The BM=64 row below
    is exactly that case.
    """
    out = _surface(_corpus(tmp_path, {
        # Identifiable, and carries a w.
        "32": {"memory_points": 8, "alpha": 0.62, "alpha_corrected": 0.60,
               "alpha_upper": 0.94, "mean_rel_err": 0.004,
               "weight_streams_per_tile": 1.2537,
               "weight_stream_bandwidth_gbps": 4374.3,
               "fixed_cost_above_intercept": True},
        # NOT identifiable (too few treads) and still carries a w.
        "64": {"memory_points": 1, "alpha": None, "alpha_corrected": None,
               "alpha_upper": None, "mean_rel_err": None,
               "weight_streams_per_tile": 1.3676,
               "weight_stream_bandwidth_gbps": 4374.3,
               "fixed_cost_above_intercept": False},
        # The key is THERE and null: this fit has no w.
        "128": {"memory_points": 8, "alpha": 0.55, "alpha_corrected": 0.53,
                "alpha_upper": 0.80, "mean_rel_err": 0.006,
                "weight_streams_per_tile": None,
                "weight_stream_bandwidth_gbps": None,
                "fixed_cost_above_intercept": None},
        # The keys are ABSENT: written before the statistic existed.
        "256": {"memory_points": 8, "alpha": 0.41, "alpha_corrected": 0.40,
                "alpha_upper": 0.62, "mean_rel_err": 0.005},
    }))
    header = next(ln for ln in out.splitlines() if ln.strip().startswith("model"))
    assert "w GB/s" in header and header.strip().endswith("A/D"), header
    rows = {ln.split()[3]: ln.split() for ln in out.splitlines()
            if ln.startswith("  mixtral-8x7b ")}
    # Identifiable, w printed with its rate and the D-versus-A label.
    assert rows["32"][-3:] == ["1.2537", "4374.3", "D>A"]
    # NOT identifiable: the alpha columns blank, the w column NOT.
    assert rows["64"][5:9] == ["--", "--", "--", "--"], rows["64"]
    assert rows["64"][-3:] == ["1.3676", "4374.3", "D<A"]
    # Key present and null: n/a, which is not the alpha columns' blank.
    assert rows["128"][-3:] == ["n/a", "n/a", "n/a"]
    # Key absent: nothing at all, and the legend says what nothing means.
    assert rows["256"] == rows["256"][:9], rows["256"]
    assert "A w column left BLANK means the report predates the statistic" in out
    assert "HAS no w, for a reason its own report" in out
    assert "not identifiable" in out


def test_a_corpus_with_no_w_anywhere_prints_no_w_columns(tmp_path):
    """THE PUBLISHED SURFACES HAVE TO STILL REBUILD BYTE FOR BYTE.

    All three committed `SURFACE.txt` files predate the statistic, and
    `tests/test_analysis_tools.py` requires this script to regenerate them
    exactly: a published summary a stranger cannot rebuild is not evidence.
    Three empty columns and a paragraph explaining their emptiness would have
    cost that for a corpus with nothing to say. So the columns appear when the
    corpus holds a w and not otherwise, and one report carrying one is enough.
    """
    without = _surface(_corpus(tmp_path / "a", {
        "32": {"memory_points": 8, "alpha": 0.62, "alpha_corrected": 0.60,
               "alpha_upper": 0.94, "mean_rel_err": 0.004}}))
    assert "w GB/s" not in without
    assert "A w column left BLANK" not in without
    with_one = _surface(_corpus(tmp_path / "b", {
        "32": {"memory_points": 8, "alpha": 0.62, "alpha_corrected": 0.60,
               "alpha_upper": 0.94, "mean_rel_err": 0.004,
               "weight_streams_per_tile": 1.2537,
               "weight_stream_bandwidth_gbps": 4374.3,
               "fixed_cost_above_intercept": True}}))
    assert "w GB/s" in with_one
    # And the committed surfaces are the no-w case, which is why they still
    # regenerate. The byte-for-byte comparison itself lives in
    # tests/test_analysis_tools.py; what belongs here is the reason the branch
    # exists, which is that these three files hold no w to print.
    assert "w GB/s" not in _surface(H200_S4)
