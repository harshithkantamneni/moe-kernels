"""scripts/paper/make_paper.py: every planned figure's data and every table
regenerates from committed files to what docs/paper/ holds."""
import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import make_paper as MP  # noqa: E402

DOC = ROOT / "docs/paper"
FILES = {
    "F1": ["data/F1.csv"], "F4": ["data/F4.csv"], "F5": ["data/F5.csv"], "F6": ["data/F6.csv"],
    "F7": ["data/F7.csv"], "F8": ["data/F8.csv"], "F9": ["data/F9.csv"],
    "F10": ["data/F10.csv", "data/F10_fits.csv"], "F11": ["data/F11.csv"],
    "T1": ["T1.md", "T1.csv"], "T4": ["T4.md", "T4.csv"], "T6": ["T6.md", "T6.csv"],
    "T7": ["T7.md", "T7.csv"],
}


def test_every_item_has_a_generator():
    assert set(MP.ITEMS) == {*FILES, "F2", "F3", "T3"}


@pytest.mark.parametrize("name", sorted(FILES))
def test_item_regenerates_byte_identical(name, tmp_path):
    MP.ITEMS[name](tmp_path)
    for f in FILES[name]:
        assert (tmp_path / f).read_bytes() == (DOC / f).read_bytes(), f


def _rms(rows, key):
    v = [float(r[key]) / float(r["measured_ms"]) - 1 for r in rows]
    return 100 * math.sqrt(sum(x * x for x in v) / len(v))


def test_f2_soft_against_hard_max(tmp_path):
    """The knee's numbers (FINDINGS 2026-09-27: rms 0.53 -> 0.28%, G=3 0.95 -> 0.26%)."""
    MP.F2(tmp_path)
    for f in ("data/F2.csv", "data/F2_mf.csv"):
        assert (tmp_path / f).read_bytes() == (DOC / f).read_bytes()
    rows = [r for r in csv.DictReader((DOC / "data/F2.csv").open()) if r["fitted"] == "1"]
    assert (round(_rms(rows, "pinf_ms"), 2), round(_rms(rows, "p14_ms"), 2)) == (0.53, 0.28)
    g3 = [r for r in rows if r["G"] == "3"]
    assert (round(_rms(g3, "pinf_ms"), 2), round(_rms(g3, "p14_ms"), 2)) == (0.95, 0.26)


def test_f3_ladders(tmp_path):
    MP.F3(tmp_path)
    assert (tmp_path / "data/F3.csv").read_bytes() == (DOC / "data/F3.csv").read_bytes()
    rows = list(csv.DictReader((DOC / "data/F3.csv").open()))
    assert sorted({int(r["G"]) for r in rows}) == [2, 3, 4, 8, 32]


def test_t3_commits_and_registrations(tmp_path):
    MP.T3(tmp_path)
    new = list(csv.reader((tmp_path / "T3.csv").open()))
    old = list(csv.reader((DOC / "T3.csv").open()))
    assert [(r[0], r[1], r[4]) for r in new] == [(r[0], r[1], r[4]) for r in old]
    have_history = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "0f77622^{commit}"],
                                  capture_output=True).returncode == 0
    if have_history:
        assert new == old


def test_plot_draws(tmp_path):
    for f in FILES["F5"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes((DOC / f).read_bytes())
    MP.plot("F5", tmp_path)
    assert (tmp_path / "figures/F5.png").stat().st_size > 1000
