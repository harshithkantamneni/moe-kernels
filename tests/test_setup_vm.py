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

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


# --------------------------------------------------------------------------
# setup_vm.sh: the stubbed box
# --------------------------------------------------------------------------

SETUP_VM = REPO / "scripts" / "setup_vm.sh"
HEAD = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()
H100 = "NVIDIA H100 80GB HBM3"
UUID = "GPU-0ffa33b8-1111-2222-3333-444455556666"


def smi_stub(driver: str = "580.95.05", cuda: str = "13.0", name: str = H100,
             free_mib: int = 81000, apps: str = "") -> str:
    """An nvidia-smi that answers the three questions setup_vm.sh and the
    preflight ask: --query-gpu=<fields>, --query-compute-apps, and the plain
    header with its CUDA version."""
    return f"""#!{PY}
import sys
args = sys.argv[1:]
fields = {{"name": {name!r}, "memory.total": "81559", "driver_version": {driver!r},
          "uuid": {UUID!r}, "memory.free": "{free_mib}"}}
q = [a for a in args if a.startswith("--query-gpu=")]
if q:
    print(", ".join(fields[k] for k in q[0].split("=", 1)[1].split(",")))
elif any(a.startswith("--query-compute-apps") for a in args):
    print({apps!r}, end="")
else:
    print("| NVIDIA-SMI {driver}   Driver Version: {driver}   CUDA Version: {cuda} |")
"""


#: A binary that must not run: it leaves a mark and fails loudly.
LOUD = '#!/bin/sh\necho "$(basename "$0") $*" >> "$LOUD_MARK"\nexit 99\n'


def _vm_world(tmp_path: Path, *, driver: str = "580.95.05", cuda: str = "13.0",
              restrict: int = 1, loud: bool = True, smi: bool = True) -> dict:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if smi:
        _stub(bindir, "nvidia-smi", smi_stub(driver, cuda))
    if loud:
        for name in ("sudo", "apt-get", "dpkg"):
            _stub(bindir, name, LOUD)
    fx = tmp_path / "fixtures"
    fx.mkdir(exist_ok=True)
    (fx / "os-release").write_text('ID=ubuntu\nVERSION_ID="22.04"\n')
    (fx / "params").write_text(f"ResmanDebugLevel: 4294967295\n"
                               f"RestrictProfilingToAdminUsers: {restrict}\n")
    (fx / "status").write_text("Name:\tbash\nCapEff:\t000001ffffffffff\n")
    (fx / "meminfo").write_text("MemTotal:       230000000 kB\nMemAvailable:   220000000 kB\n")
    return {"PATH": f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp"),
            "MOE_OS_RELEASE": str(fx / "os-release"), "MOE_NVIDIA_PARAMS": str(fx / "params"),
            "MOE_PROC_STATUS": str(fx / "status"), "MOE_MEMINFO": str(fx / "meminfo"),
            "LOUD_MARK": str(tmp_path / "loud.log"), "NCU_SEARCH": str(tmp_path / "no-ncu/*/ncu")}


def vm(env: dict, *args: str, **extra: str) -> subprocess.CompletedProcess:
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", str(SETUP_VM), *args], cwd=REPO, text=True,
                          capture_output=True, env={**env, **extra}, timeout=600)


def test_the_laptop_dry_run_prints_its_plan_and_refuses_cleanly(tmp_path):
    """On this laptop there is no GPU: the plan still prints every stage, the
    refusal names why, and nothing is written, installed or invoked."""
    env = _vm_world(tmp_path, smi=False)
    before = tree_state()
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "REFUSED: no GPU" in r.stderr
    for stage in ("S0 DETECT", "S1 REPO", "S2 DECIDE", "S3 SYSTEM", "S4 VENVS", "S5 NCU",
                  "S6 COUNTER DOOR", "S7 ENV", "S8 PREFLIGHT", "PLAN REFUSED"):
        assert stage in out, stage
    assert not Path(env["LOUD_MARK"]).exists(), Path(env["LOUD_MARK"]).read_text()
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").iterdir())
    assert tree_state() == before


@pytest.mark.parametrize("driver,cuda,extra,index,package", [
    ("580.95.05", "13.0", [], "cu130", "cuda-nsight-compute-13-0"),
    ("575.57.08", "12.9", ["--torch-index", "cu128"], "cu128", "cuda-nsight-compute-12-8"),
])
def test_the_driver_picks_the_wheel_index_and_the_ncu_package(tmp_path, driver, cuda, extra,
                                                              index, package):
    """pod_session.sh P2c's rule, applied as a decision: r580+ takes cu130,
    570-579 cu128. On 570-579 only an explicit --torch-index cu128 gets this
    far, and the plan says it is the untested path whose vLLM venv (a cu13
    torch) fails PF2, because it will."""
    env = _vm_world(tmp_path, driver=driver, cuda=cuda)
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run",
           *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"torch index         https://download.pytorch.org/whl/{index}" in r.stdout
    assert f"MOE_TORCH_INDEX=https://download.pytorch.org/whl/{index}" in r.stdout
    assert f"ncu package         {package}" in r.stdout
    assert ("OPT-IN: --torch-index cu128" in r.stdout) == (index == "cu128")
    assert ("untested in this repo" in r.stdout) == (index == "cu128")
    assert not Path(env["LOUD_MARK"]).exists()


def test_a_570_to_579_driver_is_refused_under_auto_before_the_venv_build(tmp_path, tiny):
    """S2 knew on a 575 driver that the vLLM venv's torch (resolved-vllm.txt's
    cu13 build) needs r580+, printed a WARNING, and spent the whole venv build
    before PF2 failed it with exit 1. Under --torch-index auto that box is a
    refusal: exit 2, the remedy named, nothing cloned and uv never run."""
    env = _vm_world(tmp_path, driver="575.57.08", cuda="12.9")
    plan = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    assert plan.returncode == 2, plan.stdout + plan.stderr
    assert "driver 575.57.08 is r570-r579" in plan.stderr, plan.stderr
    assert "rent an instance whose driver is r580+" in plan.stderr
    assert "pass --torch-index cu128" in plan.stderr
    assert not Path(env["LOUD_MARK"]).exists()

    bundle, tip, _ = tiny
    (tmp_path / "real").mkdir()
    real = _real_world(tmp_path / "real", APT_NCU_ONLY)
    _stub(tmp_path / "real" / "bin", "nvidia-smi", smi_stub("575.57.08", "12.9"))
    r = vm(real, "--commit", tip, "--bundle", str(bundle))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "driver 575.57.08 is r570-r579" in r.stderr
    assert not Path(real["UV_LOG"]).exists(), Path(real["UV_LOG"]).read_text()
    assert not (tmp_path / "real" / "home" / "moe" / "repo").exists()


def test_a_driver_below_570_is_refused_with_the_remedy(tmp_path):
    env = _vm_world(tmp_path, driver="565.57.01", cuda="12.7")
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "driver 565.57.01 is below r570" in r.stderr
    assert "upgrade the driver yourself" in r.stderr


def test_asking_for_cu130_on_a_pre_r580_driver_is_refused(tmp_path):
    env = _vm_world(tmp_path, driver="575.57.08", cuda="12.9")
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run",
           "--torch-index", "cu130")
    assert r.returncode == 2 and "a cu130 wheel needs r580+" in r.stderr, r.stderr


@pytest.mark.parametrize("args", [[], ["--commit", HEAD[:12]], ["--commit", "HEAD"]],
                         ids=["missing", "short", "a-ref"])
def test_the_commit_must_be_a_full_sha(tmp_path, args):
    env = _vm_world(tmp_path)
    r = vm(env, *args, "--repo", "https://example.invalid/moe.git", "--dry-run")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "--commit" in r.stderr
    assert not Path(env["LOUD_MARK"]).exists()


def test_the_dry_run_places_nothing_under_workspace_and_invokes_no_sudo_or_apt(tmp_path):
    """The VM has no /workspace, and a plan that installs nothing invokes
    neither sudo nor apt, not even `sudo -n true`: both are stubs that fail
    loudly here."""
    env = _vm_world(tmp_path)
    before = tree_state()
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "/workspace" not in r.stdout + r.stderr
    install = ("would run as root: apt-get install -y --no-install-recommends "
               "cuda-nsight-compute-13-0")
    assert install in r.stdout
    assert not Path(env["LOUD_MARK"]).exists(), Path(env["LOUD_MARK"]).read_text()
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").iterdir())
    assert tree_state() == before


def test_the_plan_names_env_sh_and_the_vllm_interpreter_in_it(tmp_path):
    env = _vm_world(tmp_path)
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    home = tmp_path / "home"
    assert f'export PY_VLLM="{home}/moe/venvs/vllm/bin/python"' in r.stdout, r.stdout
    assert f'export MOE_RESULTS_DIR="{home}/moe/results"' in r.stdout
    assert 'export MOE_CARD="nvidia_h100_80gb_hbm3"' in r.stdout
    assert "rsync -avz ubuntu@<instance ip>:" in r.stdout


@pytest.mark.parametrize("restrict,door", [(0, "door open"), (1, "door sudo")])
def test_the_counter_door_follows_the_module_flag(tmp_path, restrict, door):
    """RestrictProfilingToAdminUsers=0 needs no admin; =1 with sudo on PATH
    takes the sudo door, whose launcher the plan prints. The dry run decides
    the door without invoking sudo."""
    env = _vm_world(tmp_path, restrict=restrict)
    r = vm(env, "--commit", HEAD, "--repo", "https://example.invalid/moe.git", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert door in r.stdout, r.stdout
    launcher = 'MOE_COUNTER_LAUNCHER="sudo -E env PATH=$PATH HOME=$HOME"'
    assert (launcher in r.stdout) == (restrict == 1)
    assert not Path(env["LOUD_MARK"]).exists()


def _tiny_repo(root: Path) -> tuple[Path, str, str]:
    """A git repo holding a copy of what stage 2 runs (scripts/, moe/,
    requirements/), committed twice and bundled from its branch. Returns
    `(bundle, tip, parent)`."""
    import shutil
    src = root / "src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("scripts", "moe", "requirements"):
        shutil.copytree(REPO / d, src / d, ignore=ignore)
    for f in ("pyproject.toml", ".gitignore"):
        shutil.copy(REPO / f, src / f)

    def git(*a):
        return subprocess.run(["git", "-C", str(src), *a], capture_output=True, text=True,
                              check=True).stdout.strip()
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    parent = git("rev-parse", "HEAD")
    (src / "NOTE").write_text("two\n")
    git("add", "NOTE")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "two")
    tip = git("rev-parse", "HEAD")
    bundle = root / "moe.bundle"
    git("bundle", "create", str(bundle), "main")
    return bundle, tip, parent


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _tiny_repo(tmp_path_factory.mktemp("tiny"))


def test_a_bundle_is_accepted_at_a_sha_it_holds_and_refused_at_a_fake_one(tmp_path, tiny):
    bundle, tip, parent = tiny
    env = _vm_world(tmp_path)
    r = vm(env, "--commit", tip, "--bundle", str(bundle), "--dry-run")
    assert r.returncode == 0 and "is one of its heads" in r.stdout, r.stdout + r.stderr
    r = vm(env, "--commit", parent, "--bundle", str(bundle), "--dry-run")
    assert r.returncode == 0 and f"contains {parent}" in r.stdout, r.stdout + r.stderr
    fake = "0123456789abcdef0123456789abcdef01234567"
    r = vm(env, "--commit", fake, "--bundle", str(bundle), "--dry-run")
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"does not contain {fake}" in r.stderr
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").iterdir())


# --------------------------------------------------------------------------
# setup_vm.sh for real, on a stubbed box: clone, venvs, ncu, door, preflight
# --------------------------------------------------------------------------

NCU_STUB = """#!/bin/sh
case "$*" in
  *--version*)
    echo "NVIDIA (R) Nsight Compute Command Line Profiler"
    echo "Version 2025.3.1.0 (build 1)";;
  *--list-chips*) echo "ga100, gh100";;
  *collection*)   echo "launch__grid_size launch__registers_per_thread";;
  *--query-metrics*)
    echo "dram__bytes_read dram__bytes_write gpu__time_duration"
    echo "lts__t_sectors_srcunit_tex_op_read";;
  *) echo "ncu stub: $*" >&2; exit 1;;
esac
"""


def _real_world(tmp_path: Path, apt_sim: str) -> dict:
    """The stubbed box for a real (non-dry) run: sudo runs what it is given,
    apt-get simulates `apt_sim` and "installs" an ncu into the searched glob,
    dpkg-query says the CUDA keyring is already installed, uv builds stub venvs."""
    env = _vm_world(tmp_path, loud=False)
    bindir = tmp_path / "bin"
    ncu_home = tmp_path / "nsight" / "2025.3.1"
    _stub(bindir, "uv", UV_STUB)
    _stub(tmp_path / "ncu-src", "ncu", NCU_STUB)
    _stub(bindir, "sudo", '#!/bin/sh\necho "sudo $*" >> "$SUDO_LOG"\n'
          'while [ $# -gt 0 ]; do case "$1" in -n|-E) shift;; *) break;; esac; done\n'
          'exec "$@"\n')
    (tmp_path / "apt-sim.txt").write_text(apt_sim)
    _stub(bindir, "apt-get", '#!/bin/sh\necho "apt-get $*" >> "$APT_LOG"\ncase "$*" in\n'
          f'  "-s install"*) cat "{tmp_path / "apt-sim.txt"}";;\n'
          f'  "install"*) mkdir -p "{ncu_home}" && cp "{tmp_path / "ncu-src" / "ncu"}" '
          f'"{ncu_home}/ncu";;\nesac\nexit 0\n')
    _stub(bindir, "dpkg-query", '#!/bin/sh\ncase "$*" in\n'
          '  *cuda-keyring*) echo "install ok installed";;\n  *) exit 1;;\nesac\n')
    env.update({"NCU_SEARCH": str(tmp_path / "nsight" / "*" / "ncu"),
                "SUDO_LOG": str(tmp_path / "sudo.log"), "APT_LOG": str(tmp_path / "apt.log"),
                "UV_LOG": str(tmp_path / "uv.log")})
    return env


APT_NCU_ONLY = ("NOTE: This is only a simulation!\n"
                "Inst nsight-compute-2025.3.1 (2025.3.1.4-1 cuda)\n"
                "Inst cuda-nsight-compute-13-0 (13.0.85-1 cuda)\n"
                "Conf nsight-compute-2025.3.1 (2025.3.1.4-1 cuda)\n")
APT_MOVES_DRIVER = APT_NCU_ONLY + ("Inst nvidia-driver-580 [575.57.08-0ubuntu1] "
                                   "(580.95.05-0ubuntu1 cuda)\n"
                                   "Inst libnvidia-compute-580 (580.95.05-0ubuntu1 cuda)\n")


def test_an_apt_transaction_that_would_move_the_driver_is_refused(tmp_path, tiny):
    bundle, tip, _ = tiny
    env = _real_world(tmp_path, APT_MOVES_DRIVER)
    r = vm(env, "--commit", tip, "--bundle", str(bundle))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "would touch the driver" in r.stderr
    assert "nvidia-driver-580" in r.stderr and "libnvidia-compute-580" in r.stderr
    apt = Path(env["APT_LOG"]).read_text()
    assert "-s install --no-install-recommends cuda-nsight-compute-13-0" in apt
    assert "install -y" not in apt, "the transaction ran after the simulation refused it"


def test_a_real_run_installs_ncu_and_its_preflight_refuses_success_without_a_counter(tmp_path,
                                                                                     tiny):
    """The whole stage 1 -> stage 2 path on a stubbed box: clone from the
    bundle, venvs through setup_runpod.sh, ncu installed after a clean
    simulation, the sudo door, env.sh, and the preflight. The stub venvs have
    no torch and this checkout's dram_counter_route.py has no r3-arms family,
    so no counter can be read: the run must NOT say READY, and must exit 1
    with PREFLIGHT.txt naming PF5. The second run is idempotent: every stage
    that acted the first time says it is skipping."""
    bundle, tip, _ = tiny
    env = _real_world(tmp_path, APT_NCU_ONLY)
    home = tmp_path / "home" / "moe"
    r = vm(env, "--commit", tip, "--bundle", str(bundle))
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "stage 1 done: exec" in out and "setup_vm.sh stage 2" in out
    assert "apt-get -s: cuda-nsight-compute-13-0 touches no driver package" in out
    assert "READY: a counter was read" not in out and "NOT READY" in out
    assert subprocess.run(["git", "-C", str(home / "repo"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip() == tip
    envsh = (home / "env.sh").read_text()
    assert f'export PY_VLLM="{home}/venvs/vllm/bin/python"' in envsh
    assert f"{tmp_path}/nsight/2025.3.1:" in envsh, "ncu's directory goes first on PATH"
    assert 'MOE_COUNTER_DOOR="sudo"' in envsh
    text = (home / "session" / "PREFLIGHT.txt").read_text()
    assert re.search(r"^PF5 FAIL .*has no --family", text, re.M), text
    assert re.search(r"^PF6 SKIP ", text, re.M), text
    assert "NOT READY" in text
    payload = json.loads((home / "session" / "PREFLIGHT.json").read_text())
    assert payload["all_pass"] is False and payload["exit"] == 1
    assert "Version 2025.3.1.0" in (home / "session" / "ncu.txt").read_text()
    assert "--base-python 3.12" in (home / "session" / "logs" / "setup_runpod.log").read_text() \
        or "uv venv --python 3.12" in Path(env["UV_LOG"]).read_text()

    again = vm(env, "--commit", tip, "--bundle", str(bundle))
    out2 = again.stdout + again.stderr
    assert again.returncode == 1, out2
    for marker in ("S1 skip: ", "S3 skip: ", "base: unchanged, skipping",
                   "vllm: unchanged, skipping", "S5 skip install: ncu at "):
        assert marker in out2, marker
    assert Path(env["APT_LOG"]).read_text().count("install -y") == 1, "ncu installed twice"

    # --check installs nothing and still asks the box; --preflight-only asks only
    check = vm(env, "--commit", tip, "--check")
    assert check.returncode == 1, check.stdout + check.stderr
    assert "venv base: built and stamped" in check.stdout
    assert "venv vllm: built and stamped" in check.stdout
    assert f"ncu: {tmp_path}/nsight/2025.3.1/ncu" in check.stdout
    assert "S4 VENVS" not in check.stdout and "S5 NCU" not in check.stdout
    only = vm(env, "--commit", tip, "--preflight-only")
    assert only.returncode == 1, only.stdout + only.stderr
    assert "S8 PREFLIGHT" in only.stdout and "S2 DECIDE" not in only.stdout
    assert Path(env["APT_LOG"]).read_text().count("install -y") == 1


#: What nvidia-smi prints on an Ubuntu VM after an unattended upgrade moved the
#: NVIDIA userspace under the loaded kernel module.
NVML_MISMATCH = ("#!/bin/sh\necho 'Failed to initialize NVML: Driver/library version mismatch'\n"
                 "echo 'NVML library version: 580.95'\nexit 18\n")


@pytest.mark.parametrize("extra", [[], ["--torch-index", "cu130"]], ids=["auto", "cu130"])
def test_a_driver_nvidia_smi_cannot_reach_is_refused_before_anything_is_spent(tmp_path, tiny,
                                                                             extra):
    """S0 read the NVML error's two lines as two cards named "Failed to
    initialize NVML: ...", read no driver, and went on: it cloned, handed
    runpod_env's placeholder to uv as an --index-url and built the venvs
    before anything refused. A box whose driver nvidia-smi cannot reach is the
    header's "no GPU" refusal, exit 2, with nothing cloned and uv never run,
    whatever --torch-index says."""
    bundle, tip, _ = tiny
    env = _real_world(tmp_path, APT_NCU_ONLY)
    _stub(tmp_path / "bin", "nvidia-smi", NVML_MISMATCH)
    r = vm(env, "--commit", tip, "--bundle", str(bundle), *extra)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "REFUSED: no GPU reachable: nvidia-smi exited 18" in r.stderr, out
    assert "Driver/library version mismatch" in r.stderr and "a reboot" in r.stderr
    assert "2 GPUs" not in out and "gpu                 0x none" in r.stdout, out
    assert not Path(env["UV_LOG"]).exists(), Path(env["UV_LOG"]).read_text()
    assert not (tmp_path / "home" / "moe" / "repo").exists()
    plan = vm(env, "--commit", tip, "--bundle", str(bundle), "--dry-run", *extra)
    assert plan.returncode == 2 and "no GPU reachable" in plan.stderr, plan.stdout + plan.stderr


def test_a_checkout_at_another_sha_is_refused_not_moved(tmp_path, tiny):
    bundle, tip, parent = tiny
    env = _real_world(tmp_path, APT_NCU_ONLY)
    first = vm(env, "--commit", parent, "--bundle", str(bundle), "--no-ncu-install")
    assert first.returncode == 1, first.stdout + first.stderr
    r = vm(env, "--commit", tip, "--bundle", str(bundle))
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"is at {parent}, not {tip}" in r.stderr


# --------------------------------------------------------------------------
# the rules setup_vm.sh shares with other scripts, held to them
# --------------------------------------------------------------------------

def _sh_constant(name: str) -> str:
    m = re.search(rf'^{name}="?([^"\n]*)"?$', SETUP_VM.read_text(), re.M)
    assert m, name
    return m.group(1)


def test_the_cu130_threshold_is_pod_session_p2cs():
    """One rule at two call sites is this repo's most-repeated defect, so the
    decision here and the check in pod_session.sh are held to one number."""
    pod = (REPO / "scripts" / "pod_session.sh").read_text()
    m = re.search(r'"\$\{major:-0\}" -ge (\d+) \]\]; verdict P2c', pod)
    assert m, "pod_session.sh P2c moved; re-point this test"
    assert _sh_constant("CU130_MIN_DRIVER") == m.group(1)


def test_the_exit_codes_are_the_tables():
    from moe.bench import exit_codes
    for name, code in (("EXIT_DONE", exit_codes.DONE), ("EXIT_CLAIM_FAIL", exit_codes.CLAIM_FAIL),
                       ("EXIT_REFUSED", exit_codes.REFUSED), ("EXIT_ERROR", exit_codes.ERROR)):
        assert int(_sh_constant(name)) == code, name


def test_the_ncu_search_covers_every_place_the_chain_looks():
    chain = (REPO / "scripts" / "alpha_g_chain.sh").read_text()
    m = re.search(r'^NCU_SEARCH="\$\{NCU_SEARCH:-([^}]*)\}"', chain, re.M)
    assert m, "the chain's NCU_SEARCH default moved"
    assert set(m.group(1).split()) <= set(_sh_constant("NCU_SEARCH_DEFAULT").split())
