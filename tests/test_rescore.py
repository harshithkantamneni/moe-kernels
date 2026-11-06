"""scripts/scoring/rescore.py: the census scorers re-score the published tileA and tileA2 trees to the committed
files under PYTHONHASHSEED=0 (score_census.py's occupancy key order follows the hash seed; the file is pinned), and
the settle rental S1's registered reader re-reads its published session to the committed settleS1.read files, and the
calibration rental C2's registered reader its own to the committed cal2.read files (where the knee round's frozen chain is
on this machine: the reader refuses without it), and the calibration rental C3's registered reader its own to the committed
cal3.read files (where the run's commit is in this clone's history: the reader checks the freeze against it), and the
calibration rental C4's registered reader its own to the committed cal4.read files (where round 4's frozen chain is on
this machine: the reader refuses without it), and the decode stage's CAL scorer under the second addendum its published
session to the committed scripts/scoring/decode/cal.score files (it reads only the published tree and A2's committed
score directory, so it needs nothing off the repo), and the decode CAL2 scorer under the third addendum its published
session to the committed cal2.score files (scored on the fetched tree before the blind finalisation: equal but for the
power file's path), and the decode blind scorer under the third addendum its published session, against the committed
decodeB.finalised.json, to the committed blind.score files, and the decode settling rental DS's registered reader
under the fourth addendum its published session to the committed ds.readings files (read from the published tree, so
equal byte for byte), and tile rental B3's registered scorer its published session to the committed tileB3.score files
(where the round-5 frozen chain is on this machine and B3's run commit is in this clone's history: the scorer refuses
without either), and the committed B3 score names no model CERTIFIED at its top level."""
from __future__ import annotations

import importlib.util
import json
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


_c4spec = importlib.util.spec_from_file_location("read_c4_for_rescore", REPO / "scripts" / "scoring" / "cal4" / "read_c4.py")
_C4 = importlib.util.module_from_spec(_c4spec)
_c4spec.loader.exec_module(_C4)
HAVE_C4_CHAIN = all((_C4.WORK / f).is_file() for f in _C4.FROZEN_FILES) and \
    (Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results").is_dir()


@pytest.mark.skipif(not HAVE_C4_CHAIN, reason="round 4's work folder or A2's exfil tree is not here")
def test_the_published_cal4_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("cal4") == "results/published/2026-10-27-nvidia_gh200_480gb-cal4-session"
    ran = RS.rescore("cal4", tmp_path)
    assert ran == [("scripts/scoring/cal4/read_c4.py", 0)], ran
    rows = dict(RS.compare("cal4", tmp_path))
    assert rows == {"cal4.read.json": "EQUAL", "cal4.read.txt": "EQUAL"}, rows


def test_the_published_decode_cal_session_rescores_to_the_committed_files(tmp_path):
    assert RS.tree_of("decodeCAL") == "results/published/2026-10-29-nvidia_gh200_480gb-decodeCAL-session/results"
    ran = RS.rescore("decodeCAL", tmp_path)
    assert ran == [("scripts/scoring/decode/score_cal_d2.py", 0)], ran
    rows = dict(RS.compare("decodeCAL", tmp_path))
    assert rows == {"cal.score.json": "EQUAL", "cal.score.txt": "EQUAL"}, rows
    txt = (REPO / "scripts/scoring/decode/cal.score.txt").read_text().splitlines()
    assert "power: READ (700.0 W)" in txt and "STOP: STOP" in txt


def test_the_published_decode_cal2_session_rescores_to_the_committed_files(tmp_path):
    session = "results/published/2026-11-03-nvidia_gh200_480gb-decodeCAL2-session"
    assert RS.tree_of("decodeCAL2") == f"{session}/results"
    ran = RS.rescore("decodeCAL2", tmp_path)
    assert ran == [("scripts/scoring/decode/score_cal_d3.py", 0)], ran
    rows = dict(RS.compare("decodeCAL2", tmp_path))
    assert rows == {"cal.score.json": "EQUAL", "cal.score.txt": "EQUAL"}, rows
    # the .txt is byte-identical; the JSON differs from the committed file only in the power file's path
    assert (tmp_path / "cal.score.txt").read_bytes() == (REPO / "scripts/scoring/decode/cal2.score.txt").read_bytes()
    got = (tmp_path / "cal.score.json").read_text().splitlines()
    want = (REPO / "scripts/scoring/decode/cal2.score.json").read_text().splitlines()
    diff = [(g, w) for g, w in zip(got, want) if g != w]
    assert len(got) == len(want) and len(diff) == 1, diff
    assert diff[0][0].strip() == f'"file": "{session}/session/power-set.txt",', diff
    txt = (REPO / "scripts/scoring/decode/cal2.score.txt").read_text().splitlines()
    assert "power: READ (700.0 W)" in txt and "STOP (addendum 3, G3-D label DESCRIPTIVE): GO" in txt


def test_the_published_decode_blind_session_rescores_to_the_committed_files(tmp_path):
    assert RS.tree_of("decodeBlind") == "results/published/2026-11-04-nvidia_gh200_480gb-decodeBlind-session/results"
    ran = RS.rescore("decodeBlind", tmp_path)
    assert ran == [("scripts/scoring/decode/score_blind_d3.py", 0)], ran
    rows = dict(RS.compare("decodeBlind", tmp_path))
    assert rows == {"blind.score.json": "EQUAL", "blind.score.txt": "EQUAL"}, rows
    txt = (REPO / "scripts/scoring/decode/blind.score.txt").read_text().splitlines()
    assert "power: READ (700.0 W); 27 descriptive cells out of E1-DEC" in txt
    assert sum(" -> NOT CERTIFIED" in t for t in txt) == 5 and not any(" -> CERTIFIED" in t for t in txt)


def test_the_published_decode_ds_session_rereads_to_the_committed_files(tmp_path):
    session = "results/published/2026-11-04-nvidia_gh200_480gb-decodeDS-session"
    assert RS.tree_of("decodeDS") == f"{session}/results"
    ran = RS.rescore("decodeDS", tmp_path)
    assert ran == [("scripts/scoring/decode/read_ds.py", 0)], ran
    rows = dict(RS.compare("decodeDS", tmp_path))
    assert rows == {"ds.readings.json": "EQUAL", "ds.readings.txt": "EQUAL"}, rows
    for name in ("ds.readings.json", "ds.readings.txt"):
        assert (tmp_path / name).read_bytes() == (REPO / "scripts/scoring/decode" / name).read_bytes(), name
    txt = (REPO / "scripts/scoring/decode/ds.readings.txt").read_text().splitlines()
    assert "power: READ" in txt and "verdict: READ" in txt and "NOT IDENTIFIED: none" in txt
    assert "CENSUS CROSS-CHECK (DS's census against CAL2's, reported): SAME over 48 entries" in txt
    assert any(t.startswith("REFUSED ROWS ") and t.endswith(": 0") for t in txt)
    assert f'"{session}/session/power-set.txt"' in (REPO / "scripts/scoring/decode/ds.readings.json").read_text()


def _b3_frozen_files() -> tuple:
    """read_b3.FROZEN_FILES, read without importing the reader (it puts its own directories on sys.path)."""
    import ast
    tree = ast.parse((REPO / "scripts/scoring/tileB3/read_b3.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FROZEN_FILES":
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("no FROZEN_FILES in read_b3.py")


B3_WORK = Path.home() / "code" / "moe-kernels-work"
B3_EXFIL = Path.home() / "moe-kernels-exfil" / "runs"
B3_RUN_COMMIT = "a56080792f7404ce56f62eafc70335e66d9a6af1"
HAVE_B3_CHAIN = all((B3_WORK / f).is_file() for f in _b3_frozen_files()) and all(
    (B3_EXFIL / d / "tree/vm/results").is_dir() for d in ("gh200-tile2-20261020T2242Z", "gh200-tile2-20261021T1842Z",
                                                          "gh200-tile1-20261021T0704Z", "gh200-tile4-20261027T2238Z")) \
    and subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{B3_RUN_COMMIT}^{{commit}}"],
                       capture_output=True).returncode == 0
B3_SESSION = "results/published/2026-11-05-nvidia_gh200_480gb-tileB3-session"
B3_SCORES = REPO / "scripts/scoring/tileB3"


@pytest.mark.skipif(not HAVE_B3_CHAIN, reason="round 5's frozen chain, the exfil trees it reads or B3's run commit is not here")
def test_the_published_tileB3_session_rescores_to_the_committed_files(tmp_path):
    assert RS.tree_of("tileB3") == B3_SESSION
    ran = RS.rescore("tileB3", tmp_path)
    assert ran == [("scripts/scoring/tileB3/read_b3.py", 0)], ran
    rows = dict(RS.compare("tileB3", tmp_path))
    assert rows == {"tileB3.score.json": "EQUAL", "tileB3.score.txt": "EQUAL"}, rows
    # scored on the published tree: equal byte for byte, and the tree is recorded repo-relative
    for name in ("tileB3.score.json", "tileB3.score.txt"):
        assert (tmp_path / name).read_bytes() == (B3_SCORES / name).read_bytes(), name
    txt = (B3_SCORES / "tileB3.score.txt").read_text().splitlines()
    assert f"  pages: {B3_SESSION}" in txt
    assert "E1-TILE (rms <= 3%, every cell within 8%, |mean| <= 1.5%): FAILS" in txt


def test_the_committed_tileB3_score_certifies_no_model():
    """The registered certification (b3common.certification): CERTIFIED only when E1-TILE HOLDS. E1-TILE FAILS, so the
    top level names every blind model NOT CERTIFIED. The CLEAN view's own per-model block (views.addendum.CLEAN.E1-TILE)
    labels Qwen1.5 and JetMoE "CERTIFIED": b3common.e1_rule (tcommon.e1tile_rule, B2's view rule, with the bar as an
    argument) writes that label inside one view from the model's own bar and its page count only, before rule 3
    combines the views and without the view's pooled verdict; b3common.certification overrides it at the top level.
    The scorer is pinned and is not edited: this test holds the reading to the top level."""
    d = json.loads((B3_SCORES / "tileB3.score.json").read_text())
    assert d["E1-TILE"]["verdict"] == "FAILS"
    assert d["views"]["addendum"]["verdicts"] and d["E1-TILE"]["pooled"]["verdict"] == "FAILS"
    cert = d["certification"]
    assert set(cert) == {"phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b"}, cert
    assert all(v.startswith("NOT CERTIFIED") for v in cert.values()), cert
    assert "deepseek-v2-lite" not in cert
    assert "never CERTIFIED" in d["E1-TILE"]["per_model"]["deepseek-v2-lite"]["label"]
    txt = (B3_SCORES / "tileB3.score.txt").read_text().splitlines()
    line = next(t for t in txt if t.startswith("certification: "))
    assert '"CERTIFIED' not in line and line.count("NOT CERTIFIED") == 3, line
    # the view-local label the top level overrides (the defect described in docs/registered/README.md)
    view = d["views"]["addendum"]["CLEAN"]["E1-TILE"]
    assert view["verdict"] == "FAILS"
    assert {m: p.get("certification") for m, p in view["per_model"].items()} == {
        "deepseek-v2-lite": None, "jetmoe-8b": "CERTIFIED", "qwen1.5-moe-a2.7b": "CERTIFIED",
        "phi-3.5-moe": "NOT CERTIFIED (3 pages in CLEAN, bar FAILS)"}


def test_a_changed_score_file_reads_differs(tmp_path):
    RS.rescore("tileA", tmp_path, only=["census"])
    p = tmp_path / "census.score.txt"
    p.write_text(p.read_text() + "x")
    assert dict(RS.compare("tileA", tmp_path))["census.score.txt"] == "DIFFERS"
