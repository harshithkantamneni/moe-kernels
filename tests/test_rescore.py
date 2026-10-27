"""scripts/scoring/rescore.py: the census scorers re-score the published tileA and tileA2 trees to the committed
files under PYTHONHASHSEED=0 (score_census.py's occupancy key order follows the hash seed; the file is pinned), and
the settle rental S1's registered reader re-reads its published session to the committed settleS1.read files, and the
calibration rental C2's registered reader its own to the committed cal2.read files (where the knee round's frozen chain is
on this machine: the reader refuses without it), and the calibration rental C3's registered reader its own to the committed
cal3.read files (where the run's commit is in this clone's history: the reader checks the freeze against it)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rescore", REPO / "scripts" / "scoring" / "rescore.py")
RS = importlib.util.module_from_spec(spec)
sys.modules["rescore"] = RS
spec.loader.exec_module(RS)


def test_the_scorer_subprocess_runs_with_hash_seed_zero():
    assert RS.scorer_env()["PYTHONHASHSEED"] == "0"


@pytest.mark.parametrize("stage", ["tileA", "tileA2"])
def test_the_published_census_rescores_to_the_committed_files(stage, tmp_path):
    ran = RS.rescore(stage, tmp_path, only=["census"])
    assert ran and all(rc == 0 for _s, rc in ran), ran
    rows = dict(RS.compare(stage, tmp_path))
    assert rows == {"census.score.json": "EQUAL", "census.score.txt": "EQUAL"}, rows


def test_the_published_settleS1_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("settleS1") == "results/published/2026-10-21-nvidia_gh200_480gb-settleS1-session"
    ran = RS.rescore("settleS1", tmp_path)
    assert ran == [("scripts/scoring/settleS1/read_s1.py", 0)], ran
    rows = dict(RS.compare("settleS1", tmp_path))
    assert rows == {"settleS1.read.json": "EQUAL", "settleS1.read.txt": "EQUAL"}, rows


_c2spec = importlib.util.spec_from_file_location("read_c2_for_rescore", REPO / "scripts" / "scoring" / "cal2" / "read_c2.py")
_C2 = importlib.util.module_from_spec(_c2spec)
_c2spec.loader.exec_module(_C2)
HAVE_C2_CHAIN = all((_C2.WORK / f).is_file() for f in _C2.FROZEN_FILES) and \
    (Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results").is_dir()


@pytest.mark.skipif(not HAVE_C2_CHAIN, reason="the knee round's work folder or A2's exfil tree is not here")
def test_the_published_cal2_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("cal2") == "results/published/2026-10-21-nvidia_gh200_480gb-cal2-session"
    ran = RS.rescore("cal2", tmp_path)
    assert ran == [("scripts/scoring/cal2/read_c2.py", 0)], ran
    rows = dict(RS.compare("cal2", tmp_path))
    assert rows == {"cal2.read.json": "EQUAL", "cal2.read.txt": "EQUAL"}, rows


C3_RUN_COMMIT = "e10fda1b9e8cad4f815d549184f76cddaab781e1"
HAVE_C3_COMMIT = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{C3_RUN_COMMIT}^{{commit}}"],
                                capture_output=True).returncode == 0


@pytest.mark.skipif(not HAVE_C3_COMMIT, reason="C3's run commit is not in this clone's history")
def test_the_published_cal3_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("cal3") == "results/published/2026-10-27-nvidia_gh200_480gb-cal3-session"
    ran = RS.rescore("cal3", tmp_path)
    assert ran == [("scripts/scoring/cal3/read_c3.py", 0)], ran
    rows = dict(RS.compare("cal3", tmp_path))
    assert rows == {"cal3.read.json": "EQUAL", "cal3.read.txt": "EQUAL"}, rows


def test_a_changed_score_file_reads_differs(tmp_path):
    RS.rescore("tileA", tmp_path, only=["census"])
    p = tmp_path / "census.score.txt"
    p.write_text(p.read_text() + "x")
    assert dict(RS.compare("tileA", tmp_path))["census.score.txt"] == "DIFFERS"
