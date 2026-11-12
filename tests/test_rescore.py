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
without either), and the committed B3 score names no model CERTIFIED at its top level, and rental PC's registered
reader its published session to the committed pc.read files (where PC's run commit is in this clone's history: the
reader checks the freeze against it; read from the published tree, so equal byte for byte), and the committed PC
reading is DESCRIPTIVE with H1 NOT IDENTIFIED, and rental PC2's registered reader its published session to the
committed pc2.read files (where PC2's run commit is in this clone's history; read from the published tree, so equal
byte for byte), and the committed PC2 reading is DESCRIPTIVE with v6d's term 3 DOES NOT HOLD on both 4-per-SM tiles, and
rental RR's registered reader its published session to the committed rr.read files (where RR's run commit is in
this clone's history; read from the published tree, so equal byte for byte), and the committed RR reading is
DESCRIPTIVE with RR-R4 NOT IDENTIFIED, and RR's br unit refused on its plan line's arms against the registered page."""
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


PC_RUN_COMMIT = "c9c92f1edede773a19079f900ef7a237ee7de37e"
HAVE_PC_COMMIT = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{PC_RUN_COMMIT}^{{commit}}"],
                                capture_output=True).returncode == 0
PC_SESSION = "results/published/2026-11-06-nvidia_gh200_480gb-pc-session"
PC_READS = REPO / "scripts/scoring/pc"


@pytest.mark.skipif(not HAVE_PC_COMMIT, reason="rental PC's run commit is not in this clone's history")
def test_the_published_pc_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("pc") == PC_SESSION
    ran = RS.rescore("pc", tmp_path)
    assert ran == [("scripts/scoring/pc/read_pc.py", 0)], ran
    rows = dict(RS.compare("pc", tmp_path))
    assert rows == {"pc.read.json": "EQUAL", "pc.read.txt": "EQUAL"}, rows
    # read on the published tree: equal byte for byte, and the tree is recorded repo-relative
    for name in ("pc.read.json", "pc.read.txt"):
        assert (tmp_path / name).read_bytes() == (PC_READS / name).read_bytes(), name
    assert f"  pages: {PC_SESSION}" in (PC_READS / "pc.read.txt").read_text().splitlines()


def test_the_committed_pc_reading_is_descriptive_and_h1_not_identified():
    """PC certifies nothing; H1 reads NOT IDENTIFIED (beta_f at G 8 and the G 2 plateau rung NOT READ); the sd unit is
    refused (its regcheck FAIL); the run commit and the board are the registered ones."""
    d = json.loads((PC_READS / "pc.read.json").read_text())
    assert d["status"].startswith("MEASUREMENT, DESCRIPTIVE")
    assert d["pc"] == PC_SESSION and d["registration"] == "2026-11-06-pc-gh200"
    assert d["commit"] == {"checked": True, "files": 68, "sha": PC_RUN_COMMIT}
    assert d["board"]["valid"] and d["board"]["g_band"] == "PASS"
    h1 = d["H1"]
    assert h1["label"] == "NOT IDENTIFIED" and h1["why"] == ["unread: beta_f, plateau_spread, plateau_ratio"]
    assert h1["inputs"]["beta_f"] is None and h1["inputs"]["plateau_ratio"] is None
    assert d["PC-R4"]["C-G8 / A-G8"]["status"] == "NOT IDENTIFIED: a twin not READ"
    a_g2 = next(r for r in d["PC-R1"]["A (D64, 1710)"] if r["cell"] == "A-G2")
    assert a_g2["status"].startswith("NOT READ: overhead")
    sd = d["PC-R0"]["sd"]
    assert sd["valid"] is False and sd["regcheck"] == "FAIL" and sd["launches"] == 0
    assert d["PC-R1"]["D (D16, 1710)"] == []
    assert d["PC-R7"]["status_sat"] == "READ" and abs(d["PC-R7"]["X_sat"] - 3612.42) < 0.01
    assert d["PC-R8"]["does the curve price the CAL2 cell (Mixtral T128 A4)"]["reading"].startswith("CARRIES")


PC2_RUN_COMMIT = "e7786f4eff48a65d017ac073a2ac10d576dd19a7"
HAVE_PC2_COMMIT = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{PC2_RUN_COMMIT}^{{commit}}"],
                                 capture_output=True).returncode == 0
PC2_SESSION = "results/published/2026-11-10-nvidia_gh200_480gb-pc2-session"
PC2_READS = REPO / "scripts/scoring/pc2"


@pytest.mark.skipif(not HAVE_PC2_COMMIT, reason="rental PC2's run commit is not in this clone's history")
def test_the_published_pc2_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("pc2") == PC2_SESSION
    ran = RS.rescore("pc2", tmp_path)
    assert ran == [("scripts/scoring/pc2/read_pc2.py", 0)], ran
    rows = dict(RS.compare("pc2", tmp_path))
    assert rows == {"pc2.read.json": "EQUAL", "pc2.read.txt": "EQUAL"}, rows
    # read on the published tree: equal byte for byte, and the tree is recorded repo-relative
    for name in ("pc2.read.json", "pc2.read.txt"):
        assert (tmp_path / name).read_bytes() == (PC2_READS / name).read_bytes(), name
    assert f"  pages: {PC2_SESSION}" in (PC2_READS / "pc2.read.txt").read_text().splitlines()


def test_the_committed_pc2_reading_is_descriptive_and_term3_does_not_hold():
    """PC2 certifies nothing; every unit is valid (sd3 CLEAN, the rest PERTURBED); v6d's term 3 DOES NOT HOLD on both
    4-per-SM tiles (G 528 SATURATED, kappa 528 / 396 MOVED); c0 on 16x64x128 at G 128 is NOT IDENTIFIED, so the L
    table is too; X_sat on 16x64x128 is READ; c0 on 64x128x64 reads at G 8 only; PLAIN is present on every cell."""
    d = json.loads((PC2_READS / "pc2.read.json").read_text())
    assert d["status"].startswith("MEASUREMENT, DESCRIPTIVE")
    assert d["pc2"] == PC2_SESSION and d["registration"] == "2026-11-10-pc2-gh200"
    assert d["commit"] == {"checked": True, "files": 72, "sha": PC2_RUN_COMMIT}
    assert d["board"]["valid_decode"] is True
    r0 = d["PC2-R0"]
    assert all(u["valid"] and u["occupancy_leg"] == "PASS" and u["regcheck"] == "PASS" for u in r0.values()), r0
    assert {k: u["page_label"] for k, u in r0.items()} == {
        "sd": "PERTURBED", "sd3": "CLEAN", "sn": "PERTURBED", "sr": "PERTURBED", "ss": "PERTURBED", "sw": "PERTURBED"}
    r1 = d["PC2-R1"]
    assert r1["R1g n_if"]["sd"] == {"16x64x128w4s4": 3}
    assert r1["R1b F ladders"]["G 128"]["status"].startswith("NOT IDENTIFIED")
    assert r1["R1b F ladders"]["G 8"]["status"] == "READ"
    assert all(row["status"].startswith("NOT IDENTIFIED") for row in r1["R1c L table (c0 = R1b G 128)"])
    assert r1["R1d X_sat"]["status"] == "READ" and abs(r1["R1d X_sat"]["X_sat"] - 3781.89) < 0.01
    r2a = d["PC2-R2"]["R2a c0 per G (F 256 to 14336)"]
    assert {G: v["status"] == "READ" for G, v in r2a.items()} == {"G 8": True, "G 32": False, "G 128": False,
                                                                   "G 256": False}
    r3 = d["PC2-R3"]
    for tile in ("32x64x128w4s3", "32x128x64w4s3"):
        assert r3[tile]["v6d term 3"]["label"] == "DOES NOT HOLD", tile
        assert r3[tile]["G 528"]["status"] == "SATURATED" and r3[tile]["kappa 528 / kappa 396"]["status"] == "MOVED"
    plain = [v for k, v in r3.items() if k.startswith("R3p PLAIN")][0]
    assert len(plain) == 58 and all(v["plain_us"] is not None for v in plain.values())
    assert d["v6d terms"].startswith("read off as registered")


RR_RUN_COMMIT = "51d33d9b09301e73cab5a938baecb27012c37b83"
HAVE_RR_COMMIT = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{RR_RUN_COMMIT}^{{commit}}"],
                                capture_output=True).returncode == 0
RR_SESSION = "results/published/2026-11-12-nvidia_gh200_480gb-rr-session"
RR_READS = REPO / "scripts/scoring/rr"
RR_FORMS = ("F1 step at 42.5 MiB, between starts", "F2 step at 45.0 MiB, between mid-lives",
            "F3 step at 30.0 MiB, between starts (one partition's capacity)",
            "F4 random replacement 1 - exp(-d / 62.5 MiB), between starts")


@pytest.mark.skipif(not HAVE_RR_COMMIT, reason="rental RR's run commit is not in this clone's history")
def test_the_published_rr_session_rereads_to_the_committed_files(tmp_path):
    assert RS.tree_of("rr") == RR_SESSION
    ran = RS.rescore("rr", tmp_path)
    assert ran == [("scripts/scoring/rr/read_rr.py", 0)], ran
    rows = dict(RS.compare("rr", tmp_path))
    assert rows == {"rr.read.json": "EQUAL", "rr.read.txt": "EQUAL"}, rows
    # read on the published tree: equal byte for byte, and the tree is recorded repo-relative
    for name in ("rr.read.json", "rr.read.txt"):
        assert (tmp_path / name).read_bytes() == (RR_READS / name).read_bytes(), name
    assert f"  pages: {RR_SESSION}" in (RR_READS / "rr.read.txt").read_text().splitlines()


def test_the_committed_rr_reading_is_descriptive_and_the_replicate_not_identified():
    """RR certifies nothing; every stamps and counter unit is valid (the three stamps units PERTURBED, timing gate only);
    RR-R1 reads 38 of 40 cells on 128x128x64 and 32 of 40 on 64x128x64, every NOT READ on the 3% per-call spread;
    RR-R2 reads 69 cells on both units (56 fresh), F2 the lowest rms on the fresh cells and on both tiles; the
    descriptive refit reads 45 and 47.5 MiB; RR-R3 is numbers only; RR-R4 is NOT IDENTIFIED: the br unit wrote no
    counter page."""
    d = json.loads((RR_READS / "rr.read.json").read_text())
    assert d["status"].startswith("MEASUREMENT, DESCRIPTIVE")
    assert d["rr"] == RR_SESSION and d["registration"] == "2026-11-11-rr-gh200"
    assert d["commit"] == {"checked": True, "files": 79, "sha": RR_RUN_COMMIT}
    assert d["board"]["valid"] is True
    assert d["census"]["occupancy"] == {"128x128x64w8s3": 2, "64x128x64w8s3": 3}
    r0 = d["RR-R0"]
    assert all(u["valid"] for u in {**r0["stamps"], **r0["counters"]}.values()), r0
    assert {k: (u["page_label"], u["hard"], u["regcheck"]) for k, u in r0["stamps"].items()} == {
        k: ("PERTURBED", "PASS", "PASS") for k in ("s128", "s64", "sr")}
    r1 = d["RR-R1"]
    for unit, n_read in (("c128", 38), ("c64", 32)):
        cells = r1[unit]
        assert len(cells) == 40 and sum(c["status"] == "READ" for c in cells.values()) == n_read, unit
        assert all(c["status"].startswith("NOT READ: the calls' w1 DRAM reads spread")
                   for c in cells.values() if c["status"] != "READ"), unit
    r2 = d["RR-R2"]
    assert [sum(c["status"] == "READ" for c in r2[t]["cells"]) for t in ("128x128x64w8s3", "64x128x64w8s3")] == [37, 32]
    fresh, both = r2["pooled, FRESH synthetic cells (P not 100 or 224)"], r2["pooled, both tiles"]
    assert (fresh["cells"], both["cells"]) == (56, 69)
    assert [round(fresh[f]["rms"], 3) for f in RR_FORMS] == [0.030, 0.024, 0.052, 0.060]
    assert fresh["lowest rms (DESCRIPTIVE)"] == both["lowest rms (DESCRIPTIVE)"] == RR_FORMS[1]
    for k in ("refit (DESCRIPTIVE), FRESH cells", "refit (DESCRIPTIVE), both tiles"):
        assert r2[k]["step between starts"]["C_star_mib"] == 45.0
        assert r2[k]["step between mid-lives"]["C_star_mib"] == 47.5
    assert all(r3["status"] == "READ" for t, r3 in d["RR-R3"].items() if t.endswith("w8s3"))
    r4 = d["RR-R4"]
    assert r4["status"] == "NOT IDENTIFIED: no cell read on both"
    assert r4["br unit"]["valid"] is False and r4["br unit"]["dir"] is None
    assert all(c["counted"] == "NOT READ: not counted" for c in r4["cells"].values())


def test_rr_br_refused_on_its_plan_lines_arms_against_the_registered_page(monkeypatch, tmp_path):
    """Why RR's br unit exited 2 within 5 s and wrote no page: its registered plan line (rrcommon.py PLAN, the br row)
    runs --arms native on docs/registered/2026-10-14-tileB-hist/phi-3.5-moe-128g1.json, whose every cell names its own
    arms [native, shared]; cell_arms (scripts/private_weight_reference.py) refuses a cell asking an arm the run does
    not run, before any launch. PC's and B3's byte units on the same page ran arms native,shared. The published log
    holds exactly this refusal."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import dram_counter_route as dcr
    import private_weight_reference as r3
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "scoring" / "rr"))
    import rrcommon as RRC
    plan = (REPO / "scripts/plans/rentalRR-2026-11.plan").read_text().splitlines()
    br = [ln for ln in plan if " bytes label=br " in ln]
    assert len(br) == 1 and " arms=native " in br[0], br
    # the registered source of that line: rrcommon.UNITS' br row
    rows = [u for u in RRC.UNITS if u[1] == "bytes" and u[2] == "br"]
    assert len(rows) == 1 and " arms=native " in f" {rows[0][3]} ", rows
    # PC's and B3's byte units on the same page ran both arms
    for plan_file, line in (("rentalPC-2026-11.plan", 34), ("rentalB3-tile-2026-11.plan", 62)):
        ln = (REPO / "scripts/plans" / plan_file).read_text().splitlines()[line - 1]
        assert "phi-3.5-moe-128g1.json" in ln and " arms=native,shared " in ln, (plan_file, ln)
    page = "docs/registered/2026-10-14-tileB-hist/phi-3.5-moe-128g1.json"
    assert all(c["arms"] == ["native", "shared"] for c in json.loads((REPO / page).read_text())["cells"])
    hp = r3.load_histogram_page(REPO / page, r3.MODEL_CONFIGS["phi-3.5-moe"], 128, "phi-3.5-moe")
    assert len(r3.histogram_cells(hp, ("native", "shared"))) == 24
    with pytest.raises(r3.HistogramRefused) as e:
        r3.histogram_cells(hp, ("native",))
    msg = "cell PT n=1 asks arms ['shared'] that the run's --arms native does not run"
    assert str(e.value) == msg
    # the counter route's own plan build (dram_counter_route.r3_plan, as do_run_r3 calls it) refuses the same way
    def r3_plan(arms):
        return dcr.r3_plan(model="phi-3.5-moe", dtype="bf16", block_m=128, block_n=128, num_stages=3, group_m=1,
                           treads=hp.treads, kind="measure", arms=arms, histogram=hp, calls=dcr.R3_CALLS_PER_CELL,
                           warmups=dcr.R3_WARMUP_CALLS, profile_dir=tmp_path, stem="g1", block_k=64)
    assert len(r3_plan(("native", "shared"))["cells"]) == 24
    with pytest.raises(r3.CounterPlanRefused) as e:
        r3_plan(("native",))
    assert str(e.value) == msg
    log = REPO / RR_SESSION / "session/logs/r3c-phi-3.5-moe-br-g1-lock1410.log"
    assert log.read_text().strip() == f"REFUSE: {msg}"
    assert (REPO / RR_SESSION / "session/logs/lock1410-pages-phi-3.5-moe-br.status").read_text().startswith(
        "G=1 exit 2 ")
    assert not list((REPO / RR_SESSION / "results").glob("*-phi-3.5-moe-br-r3-counters"))


def test_a_changed_score_file_reads_differs(tmp_path):
    RS.rescore("tileA", tmp_path, only=["census"])
    p = tmp_path / "census.score.txt"
    p.write_text(p.read_text() + "x")
    assert dict(RS.compare("tileA", tmp_path))["census.score.txt"] == "DIFFERS"
