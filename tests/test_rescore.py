"""scripts/scoring/rescore.py: the census scorers re-score the published tileA and tileA2 trees to the committed
files under PYTHONHASHSEED=0 (score_census.py's occupancy key order follows the hash seed; the file is pinned), and
the settle rental S1's registered reader re-reads its published session to the committed settleS1.read files."""
from __future__ import annotations

import importlib.util
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


def test_a_changed_score_file_reads_differs(tmp_path):
    RS.rescore("tileA", tmp_path, only=["census"])
    p = tmp_path / "census.score.txt"
    p.write_text(p.read_text() + "x")
    assert dict(RS.compare("tileA", tmp_path))["census.score.txt"] == "DIFFERS"
