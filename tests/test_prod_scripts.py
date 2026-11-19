"""scripts/prod/{smoke,main,frontier_setup}.sh: shell syntax, and the --dry-run plans (every step printed, nothing
created, the ERNIE 60 s gate before any long run, the locks per branch)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"


@pytest.mark.parametrize("name", ["smoke.sh", "main.sh", "kitlib.sh", "frontier_setup.sh", "sag1410.sh", "offline.sh"])
def test_shell_syntax(name):
    assert subprocess.run(["bash", "-n", str(PROD / name)]).returncode == 0


def _dry(name, tmp_path, *extra):
    env = dict(os.environ, PY_VLLM="python3", HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    p = subprocess.run(["bash", str(PROD / name), "--dry-run", "--results", str(tmp_path / "res"), *extra],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "res").exists() and not (tmp_path / "moe").exists()
    return [line for line in p.stdout.splitlines() if line.startswith("STEP ")]


def _soft(step: str) -> bool:
    return step.split()[2] == "[soft]"


def test_smoke_dry_run_lists_the_smoke_sequence(tmp_path):
    steps = _dry("smoke.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    for want in ("env", "price-check", "capture-graphs-nsys", "nsys-export", "extract-gate-A", "sequences-A",
                 "capture-eager-torchprof", "extract-gate-B", "sag1410", "replay-decode", "replay-prefill",
                 "replay-wall", "replay-check-decode", "replay-check-prefill", "frontier-setup",
                 "frontier-profile-fidelity", "score"):
        assert want in names, want
    by = dict(zip(names, steps, strict=True))
    # review 1: the prefill replay, the sag par, the sequences and the replay-vs-serving check are HARD in the smoke
    for hard in ("replay-prefill", "sag1410", "sequences-A", "replay-check-prefill", "price-check"):
        assert not _soft(by[hard]), hard
    assert "--min-big-T 8256" in by["replay-sample-prefill"]
    assert "--sag" in by["score"] and "sag/par.json" in by["score"]
    cap = next(s for s in steps if s.split()[1] == "capture-graphs-nsys")
    assert "--cuda-graph-trace=node" in cap and "olmoe-1b-7b" in cap and "--max-seconds 60" in cap
    assert "--allow-unpinned" in next(s for s in steps if s.split()[1] == "prompts-fetch")


def test_main_dry_run_gates_first_and_locks_each_branch(tmp_path):
    steps = _dry("main.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    by = dict(zip(names, steps, strict=True))
    assert names.index("gate60-extract") < names.index("A1710-capture") < names.index("A1410-capture")
    # review 1 MAJOR 5: gate60, (the sag par), A1710, A1410, replay C, Frontier, then the secondary runs, all soft
    order = ["gate60-extract", "sag1410", "A1710-capture", "A1410-capture", "replay-decode", "replay-prefill",
             "frontier-profile-primary-uniform", "Adefault-capture", "Aon-capture", "Beager-capture"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    for n in ("Adefault-capture", "Adefault-extract", "Aon-lock", "Aon-capture", "Aon-extract", "Beager-capture",
              "Beager-extract", "replay-offset-nsys"):
        assert _soft(by[n]), n
    for n in ("A1710-capture", "A1410-extract", "replay-prefill", "sag1410", "price-check", "gate60-sequences",
              "revision-pinned", "replay-wall-present"):
        assert not _soft(by[n]), n
    # review 1 MAJOR 7: no CPU scoring or Frontier predict on the rental; the replay budget reads the smoke's walls
    assert not any(n == "score" or n.startswith("frontier-predict") for n in names)
    assert "--wall" in by["replay-sample-decode"] and "--wall" in by["replay-sample-prefill"]
    assert all("--require-pinned" in by[n] for n in ("gate60-capture", "A1710-capture", "A1410-capture"))
    assert all("--allow-unpinned" not in s for s in steps)
    assert "kit_lock 1710" in steps[names.index("A1710-lock")] and "kit_lock 1410" in steps[names.index("A1410-lock")]
    rd = steps[names.index("replay-decode")]
    rp = steps[names.index("replay-prefill")]
    assert "--locks 1710" in rd and "--branch decode" in rd and "--locks 1410" in rp and "--branch prefill" in rp
    aon = steps[names.index("Aon-capture")]
    assert "--shared-stream on" in aon
    assert re.search(r"--shared-stream off", steps[names.index("A1710-capture")])
    assert "--torch-profiler" in steps[names.index("Beager-capture")]
    assert not any("variant-all4" in n for n in _dry("main.sh", tmp_path / "x", "--skip-frontier-variant"))


def test_offline_dry_run_predicts_and_scores_on_the_laptop(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--results", str(tmp_path / "res")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    steps = [x for x in p.stdout.splitlines() if x.startswith("STEP ")]
    names = [x.split()[1] for x in steps]
    assert names[-1] == "score" and "frontier-predict-primary-uniform" in names and not (tmp_path / "res").exists()
    sc = steps[-1]
    assert "--sag" in sc and "--workers 4" in sc and "ernie-4.5-21b-a3b" in sc


def test_the_sag_plan_is_the_v7_prefill_blinds_own_units_and_the_driver_accepts_it(tmp_path):
    plan = [x for x in (PROD / "sag1410.plan").read_text().splitlines() if x and not x.startswith("#")]
    reg = (REPO / "scripts/plans/rentalV7P-prefill-blind-2026-11.plan").read_text().splitlines()
    assert plan[0] == "- prelude lock-mhz=1410 power-limit-w=700 est=2 cap=15" and plan[0] in reg
    assert all(x in reg for x in plan) and len(plan) == 4
    assert [x.split()[1] for x in plan] == ["prelude", "bytes", "bytes", "gate"] and plan[-1].endswith("cap=5")
    env = dict(os.environ, MOE_HOME=str(tmp_path / "moe"), HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "sag1410.sh"), "--out", str(tmp_path / "sag"), "--dry-run"],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "PASS sag1410 plan" in p.stdout and "gate b3sag" in p.stdout and not (tmp_path / "sag").exists()


def test_frontier_setup_dry_run(tmp_path):
    env = dict(os.environ, PY_VLLM="python3")
    p = subprocess.run(["bash", str(PROD / "frontier_setup.sh"), "--dry-run", "--dest", str(tmp_path / "fr")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "checkout --quiet --detach db5a85b" in p.stdout and "vllm==0.10.2" in p.stdout
    assert "--no-deps -e" in p.stdout and not (tmp_path / "fr").exists()
