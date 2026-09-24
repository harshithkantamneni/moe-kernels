"""scripts/setup_vm.sh and the setup_runpod.sh it calls, checked without a VM.

A Lambda VM is a plain Ubuntu box with root and no network volume: its driver
is unknown until it boots, ncu may be absent, the counter door is the host's
module flag rather than a container capability, and everything on its disk is
lost at termination. `setup_vm.sh` exists for the jobs a RunPod pod never has
(clone at a pinned commit, pick the torch wheel index off the driver, install
ncu without moving the driver, pick a counter door, prove a counter readable)
and hands venv building to `setup_runpod.sh`, which it calls and never copies.

These run the real scripts as subprocesses with stub binaries on PATH
(nvidia-smi, sudo, apt-get, dpkg, uv, ncu) and HOME in a temporary directory,
in the style of tests/test_shell_gates.py. A stub that must never run
(sudo and apt-get under --dry-run) writes a marker and exits 99, so "the plan
invokes neither" is asserted, not assumed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETUP_RUNPOD = REPO / "scripts" / "setup_runpod.sh"
PY = sys.executable


def tree_state() -> str:
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                          capture_output=True, text=True).stdout


def _stub(bindir: Path, name: str, body: str) -> Path:
    bindir.mkdir(parents=True, exist_ok=True)
    path = bindir / name
    path.write_text(body)
    path.chmod(0o755)
    return path


# --------------------------------------------------------------------------
# setup_runpod.sh, as the VM calls it
# --------------------------------------------------------------------------

#: A uv that builds nothing: `uv venv DIR` makes DIR/bin/python an interpreter
#: that runs this test's python WITHOUT site-packages (`-S`), which is enough
#: for setup_runpod.sh's own resolved-set check and makes its report block's
#: `import torch` fail fast; every other call is logged and succeeds, unless
#: UV_FAIL_INSTALL names a word that appears in the call.
UV_STUB = f"""#!/usr/bin/env bash
printf 'uv %s\\n' "$*" >> "${{UV_LOG:-/dev/null}}"
if [[ "$1" == venv ]]; then
  dir="${{@: -1}}"
  mkdir -p "$dir/bin"
  printf '#!/bin/sh\\nexec {PY} -S "$@"\\n' > "$dir/bin/python"
  chmod +x "$dir/bin/python"
  exit 0
fi
if [[ -n "${{UV_FAIL_INSTALL:-}}" && "$*" == *"$UV_FAIL_INSTALL"* ]]; then
  echo "uv stub: failing on purpose ($UV_FAIL_INSTALL)" >&2
  exit 1
fi
exit 0
"""


def _runpod_world(tmp_path: Path) -> dict:
    """A requirements dir whose resolved set agrees with its input, a stub uv,
    and a workspace in tmp. Returns the env to run setup_runpod.sh under."""
    req = tmp_path / "requirements"
    req.mkdir()
    (req / "base.txt").write_text("numpy\n")
    (req / "resolved-base.txt").write_text("numpy==2.1.0\ntorch==2.13.0\n")
    (req / "vllm.txt").write_text("vllm==0.27.1\n")
    (req / "resolved-vllm.txt").write_text("vllm==0.27.1\ntorch==2.13.0\n")
    bindir = tmp_path / "bin"
    _stub(bindir, "uv", UV_STUB)
    ws = tmp_path / "ws"
    return {"PATH": f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path / "home"), "WORKSPACE": str(ws),
            "MOE_VENV_ROOT": str(ws / "venvs"), "MOE_REQUIREMENTS_DIR": str(req),
            "MOE_PYTHON": PY, "UV_LOG": str(tmp_path / "uv.log")}


def runpod(env: dict, *args: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SETUP_RUNPOD), *args], cwd=REPO, text=True,
                          capture_output=True, env={**env, **extra})


def test_a_second_run_with_nothing_changed_skips_on_its_stamp(tmp_path):
    """THE STAMP WAS READ AND NEVER WRITTEN. setup_runpod.sh compares
    `$VENVS/.stamp-<env>` with the requirements file's hash and skips an
    unchanged environment, and 75bd12a dropped the one line that wrote it, so
    every run rebuilt every environment. setup_vm.sh's idempotence rests on
    this skip, so the second run must print it."""
    env = _runpod_world(tmp_path)
    first = runpod(env, "base", "vllm")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "base: building" in first.stdout and "vllm: building" in first.stdout
    second = runpod(env, "base", "vllm")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "base: unchanged, skipping" in second.stdout, second.stdout
    assert "vllm: unchanged, skipping" in second.stdout, second.stdout


def test_the_stamp_keys_on_the_torch_pin_and_the_interpreter_too(tmp_path):
    """A stamp keyed on the requirements file alone would call a venv built
    against the cu128 index "unchanged" when the next run asks for cu130, or a
    framework venv on one interpreter "unchanged" when the next asks for
    another. Both are rebuilds; the interpreter change recreates the venv."""
    env = _runpod_world(tmp_path)
    pin = {"MOE_BASE_TORCH": "torch==2.13.0",
           "MOE_TORCH_INDEX": "https://download.pytorch.org/whl/cu128"}
    assert runpod(env, "base", **pin).returncode == 0
    same = runpod(env, "base", **pin)
    assert "base: unchanged, skipping" in same.stdout, same.stdout
    moved = runpod(env, "base", **{**pin, "MOE_TORCH_INDEX":
                                   "https://download.pytorch.org/whl/cu130"})
    assert "base: building" in moved.stdout, moved.stdout
    assert runpod(env, "vllm").returncode == 0
    other = runpod(env, "vllm", MOE_FRAMEWORK_PYTHON="3.12")
    assert "vllm: building" in other.stdout, other.stdout
    assert "recreating it" in other.stdout, other.stdout


def test_a_failed_install_leaves_no_stamp_behind(tmp_path):
    """setup_env runs under `||`, where bash suspends errexit, so a failed
    `uv pip install` used to fall through to the editable install and the
    freeze. With a stamp written at the end that would mark a broken venv as
    built; the next run must build it again."""
    env = _runpod_world(tmp_path)
    bad = runpod(env, "base", UV_FAIL_INSTALL="-r ")
    assert "environment(s) FAILED: base" in bad.stdout, bad.stdout + bad.stderr
    assert not (Path(env["MOE_VENV_ROOT"]) / ".stamp-base").exists()
    again = runpod(env, "base")
    assert "base: building" in again.stdout, again.stdout


def test_the_framework_venvs_take_the_interpreter_they_are_asked_for():
    """The resolved sets were frozen under the pod's /usr/bin/python3.12
    (profiles/q2_kernel_names.txt), and the framework venvs took whatever
    interpreter uv found: python3.10 on an Ubuntu 22.04 VM.
    MOE_FRAMEWORK_PYTHON pins vllm's and sglang's, and the plan says so."""
    r = subprocess.run(["bash", str(SETUP_RUNPOD), "--dry-run", "vllm"], cwd=REPO,
                       text=True, capture_output=True,
                       env={**os.environ, "MOE_PYTHON": PY, "MOE_FRAMEWORK_PYTHON": "3.12",
                            "WORKSPACE": "/tmp/moe-vm-plan"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "vllm: uv venv --python 3.12 /tmp/moe-vm-plan/venvs/vllm" in r.stdout, r.stdout


def test_a_vm_is_told_its_disk_dies_with_the_instance(tmp_path):
    """The closing block said "*** NOT A MOUNTED VOLUME ***" and "lost when the
    pod is terminated", which is a RunPod warning; on a VM the local disk is
    the whole design and survives a reboot. MOE_HOST_KIND=vm says what is
    true there, including where the results have to go."""
    env = _runpod_world(tmp_path)
    empty = tmp_path / "empty-req"
    empty.mkdir()
    r = runpod(env, "base", MOE_REQUIREMENTS_DIR=str(empty), MOE_HOST_KIND="vm")
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("local disk: survives reboot, lost at termination; exfiltrate results"
            in r.stdout), r.stdout
    assert "NOT A MOUNTED VOLUME" not in r.stdout
    pod = runpod(env, "base", MOE_REQUIREMENTS_DIR=str(empty))
    assert "NOT A MOUNTED VOLUME" in pod.stdout, pod.stdout
    refused = runpod(env, "base", MOE_REQUIREMENTS_DIR=str(empty), MOE_HOST_KIND="vn")
    assert refused.returncode == 2 and "MOE_HOST_KIND" in refused.stderr
