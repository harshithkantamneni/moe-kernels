"""Model v7's prefill branch over round 5's chain (scripts/round5_chain.py, scripts/round5_frozen), checked through
tests/v7_chain_job.py in its own interpreter: the vendored work files are B3's frozen ones; with RR off v7 is round 5
exactly (B3's 84 committed predictions to 6 decimals and their reads to the byte; round 5's held-out table as
tileB3.score.json preflight records it); with RR on only the 48 G 1 cells move, only in w1, and D6FIX moves none; the
chain opens nothing outside the repository. No gate statistic of v7 is computed (the gate runs after its
registration is public)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import round5_chain as R5  # noqa: E402

PUB = REPO / "results" / "published"
HAVE = all((PUB / s / "results").is_dir() for s in R5.RUNS.values())
need = pytest.mark.skipif(not HAVE, reason="the A2, S1, C2 or C4 session is not published in this checkout")


def test_the_vendored_files_are_b3s_frozen_files_byte_for_byte():
    want = json.loads(R5.B3_REG.read_text())["frozen"]["files"]
    assert set(want) == set(R5.FROZEN_FILES) and len(want) == 32
    for f in R5.FROZEN_FILES:
        assert hashlib.sha256((R5.FROZEN_ROOT / f).read_bytes()).hexdigest() == want[f], f
    assert sorted(str(p.relative_to(R5.FROZEN_ROOT)) for p in R5.FROZEN_ROOT.rglob("*") if p.is_file()) == sorted(R5.FROZEN_FILES)


def test_verify_refuses_a_changed_vendored_file(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "frozen"
    shutil.copytree(R5.FROZEN_ROOT, root)
    monkeypatch.setattr(R5, "FROZEN_ROOT", root)
    R5.verify()
    p = root / R5.FROZEN_FILES[0]
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(R5.Refused):
        R5.verify()


def test_the_frozen_home_maps_the_work_root_the_repo_and_the_four_runs(monkeypatch):
    monkeypatch.setattr(R5, "_HOME", {})
    h = R5.frozen_home()
    assert (h / "code" / "moe-kernels-work").resolve() == R5.FROZEN_ROOT.resolve()
    assert (h / "code" / "moe-kernels").resolve() == R5.REPO.resolve()
    for run, session in R5.RUNS.items():
        res = h / "moe-kernels-exfil" / "runs" / run / "tree" / "vm" / "results"
        assert res.is_dir() and not res.is_symlink()
        src = R5.PUB / session / "results"
        if src.is_dir():
            n = sum(1 for p in src.rglob("*") if p.is_file())
            assert sum(1 for p in res.rglob("*") if p.is_symlink()) == n


@pytest.fixture(scope="module")
def job():
    if not HAVE:
        pytest.skip("the A2, S1, C2 or C4 session is not published in this checkout")
    r = subprocess.run([sys.executable, str(REPO / "tests" / "v7_chain_job.py")], capture_output=True, text=True,
                       cwd=str(REPO), timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@need
def test_with_rr_off_v7_prices_b3s_84_cells_as_b3_committed_them(job):
    off = job["b3_off"]
    assert len(off) == 84
    assert [c["ms"] for c in off] == [c["want_ms"] for c in off]
    assert [c["reads"] for c in off] == [c["want_reads"] for c in off]


@need
def test_with_rr_off_v7_reproduces_round_5s_held_out_table(job):
    assert job["r5_off"] == job["r5_want"]
    assert job["r5_off"] == {**R5.ROUND5_MEANS, R5.M_C2: R5.ROUND5_M_C2}


@need
def test_rr_moves_only_the_g1_cells_and_only_their_w1_and_d6fix_moves_no_prefill_cell(job):
    on = job["b3_on"]
    moved = on["moved"]
    assert len(moved) == on["g1_cells"] == 48
    assert {(m["tile"], m["G"]) for m in moved} == {("128x128x64w8s3", 1), ("64x128x64w8s3", 1)}
    assert all(m["w2_same"] and m["rr"]["form"] == "F2" and m["rr"]["C_mib"] == 47.5 for m in moved)
    assert {m["rr"]["sigma"] for m in moved if m["tile"].startswith("128")} == {0.08}
    assert {m["rr"]["sigma"] for m in moved if m["tile"].startswith("64")} == {0.06}
    assert on["ms"] == on["ms_rr_without_d6fix"]


@need
def test_the_chain_opens_nothing_outside_the_repository(job):
    assert job["trace"] == [], job["trace"][:10]
