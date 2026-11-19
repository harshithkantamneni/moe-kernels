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


@pytest.mark.parametrize("name", ["smoke.sh", "main.sh", "kitlib.sh", "frontier_setup.sh"])
def test_shell_syntax(name):
    assert subprocess.run(["bash", "-n", str(PROD / name)]).returncode == 0


def _dry(name, tmp_path, *extra):
    env = dict(os.environ, PY_VLLM="python3", HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    p = subprocess.run(["bash", str(PROD / name), "--dry-run", "--results", str(tmp_path / "res"), *extra],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "res").exists() and not (tmp_path / "moe").exists()
    return [line for line in p.stdout.splitlines() if line.startswith("STEP ")]


def test_smoke_dry_run_lists_the_smoke_sequence(tmp_path):
    steps = _dry("smoke.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    for want in ("env", "capture-graphs-nsys", "nsys-export", "extract-gate-A", "capture-eager-torchprof",
                 "extract-gate-B", "replay-decode", "frontier-setup", "frontier-profile-fidelity", "score"):
        assert want in names, want
    cap = next(s for s in steps if s.split()[1] == "capture-graphs-nsys")
    assert "--cuda-graph-trace=node" in cap and "olmoe-1b-7b" in cap and "--max-seconds 60" in cap
    assert "--allow-unpinned" in next(s for s in steps if s.split()[1] == "prompts-fetch")


def test_main_dry_run_gates_first_and_locks_each_branch(tmp_path):
    steps = _dry("main.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    assert names.index("gate60-extract") < names.index("A1710-capture") < names.index("A1410-capture")
    assert all("--allow-unpinned" not in s for s in steps)
    assert "kit_lock 1710" in steps[names.index("A1710-lock")] and "kit_lock 1410" in steps[names.index("A1410-lock")]
    rd = steps[names.index("replay-decode")]
    rp = steps[names.index("replay-prefill")]
    assert "--locks 1710" in rd and "--branch decode" in rd and "--locks 1410" in rp and "--branch prefill" in rp
    aon = steps[names.index("Aon-capture")]
    assert "--shared-stream on" in aon
    assert re.search(r"--shared-stream off", steps[names.index("A1710-capture")])
    assert "--torch-profiler" in steps[names.index("Beager-capture")]
    assert "ernie-4.5-21b-a3b" in steps[names.index("score")]
    assert not any("variant-all4" in n for n in _dry("main.sh", tmp_path / "x", "--skip-frontier-variant"))


def test_frontier_setup_dry_run(tmp_path):
    env = dict(os.environ, PY_VLLM="python3")
    p = subprocess.run(["bash", str(PROD / "frontier_setup.sh"), "--dry-run", "--dest", str(tmp_path / "fr")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "checkout --quiet --detach db5a85b" in p.stdout and "vllm==0.10.2" in p.stdout
    assert "--no-deps -e" in p.stdout and not (tmp_path / "fr").exists()
