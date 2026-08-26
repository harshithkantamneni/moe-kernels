"""One sweep must leave one run id behind, not one per venv.

`run_env` builds `python -m moe.bench.cli --env <env> <cli_args>` and hands the
SAME cli_args to every environment, so `--run-id X` is what makes the arms of a
sweep identifiable as one experiment afterwards. Without it each venv falls
through to `driver.RunConfig.run_id`, a fresh uuid4 per process, and three CSVs
that are one result become three unrelated ones on disk. That is what left the
2026-08-26 three-way sweep publishable only one arm at a time.

The fake interpreters here are shell scripts, not mocks: run_env really does
spawn them, really does parse the JSON line they print, and really does fail the
same way if the contract changes.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

from moe.runner.subproc import run_env, run_envs

FAKE = """#!/bin/sh
# Stands in for a venv interpreter: record the argv it was invoked with, then
# emit the JSON summary line the real CLI ends with.
echo "$@" >> "{log}"
run_id=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --env) env="$2"; shift 2 ;;
    *) shift ;;
  esac
done
csv="{out}/run_${{run_id}}_${{env}}.csv"
: > "$csv"
echo "{{\\"csv\\": \\"$csv\\", \\"manifest\\": \\"$csv\\", \\"run_id\\": \\"$run_id\\"}}"
"""


def make_venv(root: Path, env: str, log: Path, out: Path) -> None:
    bindir = root / env / "bin"
    bindir.mkdir(parents=True)
    py = bindir / "python"
    py.write_text(FAKE.format(log=log, out=out))
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def argv_per_env(log: Path) -> list[list[str]]:
    return [line.split() for line in log.read_text().splitlines() if line.strip()]


def test_the_same_run_id_reaches_every_environment(tmp_path):
    root, log, out = tmp_path / "venvs", tmp_path / "argv.log", tmp_path / "out"
    out.mkdir()
    for env in ("base", "vllm", "sglang"):
        make_venv(root, env, log, out)

    _, results = run_envs(["base", "vllm", "sglang"], ["--run-id", "sweep01"],
                          out / "merged.csv", root=root, cwd=tmp_path)

    assert [r.run_id for r in results] == ["sweep01"] * 3
    for argv in argv_per_env(log):
        assert "--run-id" in argv and argv[argv.index("--run-id") + 1] == "sweep01"
    # The point of the shared id: one glob finds the whole sweep.
    assert sorted(p.name for p in out.glob("run_sweep01_*.csv")) == [
        "run_sweep01_base.csv", "run_sweep01_sglang.csv", "run_sweep01_vllm.csv"]


def test_each_environment_is_told_which_one_it_is(tmp_path):
    """`--env` is per-process even though everything else is shared; the `env`
    field on every span is what pipeline validation checks against."""
    root, log, out = tmp_path / "venvs", tmp_path / "argv.log", tmp_path / "out"
    out.mkdir()
    for env in ("base", "vllm"):
        make_venv(root, env, log, out)

    run_env("base", ["--run-id", "x"], root=root, cwd=tmp_path, echo=False)
    run_env("vllm", ["--run-id", "x"], root=root, cwd=tmp_path, echo=False)

    envs = [argv[argv.index("--env") + 1] for argv in argv_per_env(log)]
    assert envs == ["base", "vllm"]


RUN_ALL = Path(__file__).resolve().parents[1] / "scripts" / "run_all.sh"


def dry_run_id(*args: str) -> str:
    """run_all.sh --dry-run needs no GPU, so the default path is testable."""
    proc = subprocess.run(["bash", str(RUN_ALL), "--dry-run", *args],
                          cwd=RUN_ALL.parents[1], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    m = re.search(r"sweep run id\s+(\S+)", proc.stdout)
    assert m, f"run_all.sh printed no sweep run id:\n{proc.stdout}"
    return m.group(1)


def test_run_all_gives_a_sweep_one_run_id_when_none_was_asked_for():
    """The default path is the one that matters: nobody passes --run-id by hand,
    and it is the default that produced three unrelated uuids."""
    first, second = dry_run_id(), dry_run_id()
    assert re.fullmatch(r"[0-9a-f]{12}", first), first
    assert first != second, "a run id that repeats would collide across sweeps"


def test_an_explicit_run_id_is_used_verbatim():
    """`--run-id` is also how an interrupted sweep resumes, so it must not be
    replaced by a generated one."""
    assert dry_run_id("--run-id", "resume0abcdef") == "resume0abcdef"


def test_a_missing_environment_costs_only_that_arm(tmp_path):
    """A framework that is not installed must not take the rest of the sweep
    down: the session is billed by the minute."""
    root, log, out = tmp_path / "venvs", tmp_path / "argv.log", tmp_path / "out"
    out.mkdir()
    make_venv(root, "base", log, out)

    _, results = run_envs(["base", "sglang"], ["--run-id", "sweep02"],
                          out / "merged.csv", root=root, cwd=tmp_path)

    by_env = {r.env: r for r in results}
    assert by_env["base"].ok
    assert not by_env["sglang"].ok
    assert by_env["sglang"].returncode == 127
    assert os.path.exists(out / "run_sweep02_base.csv")
