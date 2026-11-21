"""scripts/prod/env_preflight.py (smoke rental 2026-11-19): every environment fault the smoke met, checked in the
first step with its reason. The commands it calls (nvidia-smi, sudo, nsys, ncu, ninja) are fakes on PATH; git is
real, on a throwaway repo; the HF cache is a fake one read offline; Frontier is a local repo at a known commit."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"
sys.path.insert(0, str(PROD))
import prodlib as PL  # noqa: E402

FAKE_SMI = r"""#!/bin/bash
echo "$*" >> "$FAKE_LOG"
case "$*" in
  *enforced.power.limit*) echo "${FAKE_ENFORCED:-700.00}" ;;
  *power.limit*) echo "900.00" ;;
  *query-compute-apps*) [[ -n "${FAKE_APPS:-}" ]] && echo "$FAKE_APPS" ;;
  -lgc*) if [[ -n "${FAKE_LGC_FAIL:-}" ]]; then echo "Setting locked GPU clocks is not supported"; exit 3; fi
         f="${2%%,*}"; echo "GPU clocks set to \"(gpuClkMin $f, gpuClkMax $f)\" for GPU 00000009:01:00.0" ;;
  -rgc*) echo "All done." ;;
  -pm*) echo "Persistence mode is already Enabled" ;;
esac
exit 0
"""
FAKE_SUDO = '#!/bin/bash\n[[ "$1" == -n ]] && shift\nexec "$@"\n'
FAKE_NSYS = r"""#!/bin/bash
echo "TMPDIR=$TMPDIR $*" >> "$FAKE_LOG"
if [[ -n "${FAKE_NSYS_FAIL:-}" ]]; then echo "Failed to create directory /tmp/nvidia/nsight_systems"; exit 1; fi
exit 0
"""


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                            GIT_COMMITTER_EMAIL="t@t"))
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def box(tmp_path):
    """A fake VM: fake tools on PATH, a clean checkout, a Frontier repo, a partial HF cache, setup_vm READY."""
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in (("nvidia-smi", FAKE_SMI), ("sudo", FAKE_SUDO), ("nsys", FAKE_NSYS),
                       ("ncu", "#!/bin/bash\nexit 0\n"), ("ninja", "#!/bin/bash\nexit 0\n")):
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "f.txt").write_text("x")
    _git("init", "-q", cwd=repo)
    _git("add", "f.txt", cwd=repo)
    head = _git("commit", "-q", "-m", "c", cwd=repo)
    fr = tmp_path / "frontier-origin"
    fr.mkdir()
    (fr / "g.txt").write_text("y")
    _git("init", "-q", cwd=fr)
    _git("add", "g.txt", cwd=fr)
    fsha = _git("commit", "-q", "-m", "c", cwd=fr)
    m = PL.MODELS["olmoe-1b-7b"]
    rev = PL.hf_revision("olmoe-1b-7b")
    snap = tmp_path / "hf" / ("models--" + m.hf_repo.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True)
    for f in ("config.json", "model.safetensors.index.json"):          # only what vLLM fetches
        (snap / f).write_text("{}")
    ses = tmp_path / "session"
    ses.mkdir()
    (ses / "PREFLIGHT.txt").write_text("THE VM PREFLIGHT\nPF1 PASS\nverdict READY (exit 0: done)\n")
    blob = tmp_path / "shard.bin"                                       # P12 times a local "download"
    blob.write_bytes(b"\0" * (4 << 20))
    gitdir = str(Path(shutil.which("git")).parent)
    env = dict(os.environ, PATH=f"{b}:{gitdir}:/usr/bin:/bin", FAKE_LOG=str(tmp_path / "calls.log"),
               HF_HUB_CACHE=str(tmp_path / "hf"), HF_HUB_OFFLINE="1", HOME=str(tmp_path / "home"),
               MOE_HOME=str(tmp_path / "moe"))
    env.pop("PROD_EXPECT_COMMIT", None)
    args = ["--repo", str(repo), "--model", "olmoe-1b-7b", "--expect-commit", head, "--nsys", str(b / "nsys"),
            "--tmpdir", str(tmp_path / "moe" / "tmp"), "--session", str(ses), "--frontier-url", str(fr),
            "--frontier-commit", fsha, "--frontier-src", str(tmp_path / "moe" / "frontier" / "src"),
            "--disk", str(tmp_path), "--min-free-gb", "0", "--rate-url", blob.as_uri(), "--out", str(tmp_path / "pf.json")]
    return {"tmp": tmp_path, "env": env, "args": args, "repo": repo, "ses": ses, "head": head}


def _run(box, *extra, **env_extra):
    env = dict(box["env"], **env_extra)
    p = subprocess.run([sys.executable, str(PROD / "env_preflight.py"), *box["args"], *extra], capture_output=True,
                       text=True, env=env, timeout=120)
    res = json.loads((box["tmp"] / "pf.json").read_text())
    return p, {c["id"]: c for c in res["checks"]}


def test_every_check_passes_on_a_ready_box_and_the_card_is_left_unlocked(box):
    p, c = _run(box)
    assert p.returncode == 0, p.stdout + p.stderr
    assert all(v["verdict"] == "PASS" for v in c.values()), c
    assert "PASS env-preflight" in p.stdout and "resolves to the pin" in c["P8"]["detail"]
    assert "MB/s" in c["P12"]["detail"] and "measured" in c["P12"]["detail"]
    log = (box["tmp"] / "calls.log").read_text().splitlines()
    # P3 read the ENFORCED limit; P4 set each lock and restored the default clocks after each one
    assert any("enforced.power.limit" in x for x in log)
    lgc = [x for x in log if x.startswith("-lgc") or x.startswith("-rgc")]
    assert lgc == ["-lgc 1710,1710", "-rgc", "-lgc 1410,1410", "-rgc"]
    # P7: nsys profiled /bin/true with node tracing under the kit's TMPDIR
    nsys = next(x for x in log if x.startswith("TMPDIR="))
    assert f"TMPDIR={box['tmp'] / 'moe' / 'tmp'} " in nsys and "--cuda-graph-trace=node" in nsys and "/bin/true" in nsys


@pytest.mark.parametrize("mutate,cid,why", [
    ("dirty", "P1", "DIRTY"),
    ("commit", "P1", "is not the expected commit"),
    ("busy", "P2", "the GPU is busy"),
    ("power", "P3", "enforced power limit 900 W, want 700 W"),
    ("lock", "P4", "did not take"),
    ("nsys", "P7", "Failed to create directory"),
    ("ninja", "P6", "missing on PATH: ninja"),
    ("hf", "P8", "config.json"),
    ("frontier", "P9", "has no commit"),
    ("disk", "P10", "need >"),
    ("setup", "P11", "NOT READY"),
    ("rate", "P12", "under the 1e+12 MB/s floor"),
    ("rate-url", "P12", "URLError"),
])
def test_each_fault_fails_its_own_check_with_its_reason(box, mutate, cid, why):
    env, extra = {}, []
    if mutate == "dirty":
        (box["repo"] / "f.txt").write_text("patched live")
    elif mutate == "commit":                       # the checkout moved on past the commit the run expects
        (box["repo"] / "f.txt").write_text("next")
        _git("commit", "-q", "-am", "next", cwd=box["repo"])
    elif mutate == "busy":
        env["FAKE_APPS"] = "4242, python3"
    elif mutate == "power":
        env["FAKE_ENFORCED"] = "900.00"
    elif mutate == "lock":
        env["FAKE_LGC_FAIL"] = "1"
    elif mutate == "nsys":
        env["FAKE_NSYS_FAIL"] = "1"
    elif mutate == "ninja":
        (box["tmp"] / "bin" / "ninja").unlink()
    elif mutate == "hf":
        env["HF_HUB_CACHE"] = str(box["tmp"] / "empty-hf")
    elif mutate == "frontier":
        extra = ["--frontier-commit", "0" * 40]
    elif mutate == "disk":
        extra = ["--min-free-gb", "1e12"]
    elif mutate == "rate":                         # a slow link: the measured rate is reported with the floor
        extra = ["--min-mbps", "1e12"]
    elif mutate == "rate-url":
        extra = ["--rate-url", (box["tmp"] / "nope.bin").as_uri()]
    elif mutate == "setup":
        (box["ses"] / "PREFLIGHT.txt").write_text("verdict NOT READY (exit 1: claim failed)\n")
    p, c = _run(box, *extra, **env)
    assert p.returncode == 3 and "FAIL env-preflight" in p.stdout
    assert c[cid]["verdict"] == "FAIL" and why in c[cid]["detail"], c[cid]
    others = [k for k, v in c.items() if v["verdict"] == "FAIL" and k != cid]
    assert others == ([] if mutate != "busy" else ["P4"]), others     # a busy GPU is never locked


def test_the_expected_commit_falls_back_to_setup_vms_env_sh(box, tmp_path):
    import env_preflight as EP
    moe = tmp_path / "m"
    moe.mkdir()
    assert EP.expected_commit(None, moe) == (None, "none")
    (moe / "env.sh").write_text("# Written by scripts/setup_vm.sh at abc1234. Source it: . x\nexport A=1\n")
    assert EP.expected_commit(None, moe) == ("abc1234", str(moe / "env.sh"))
    assert EP.expected_commit("def", moe)[0] == "def"
    # frontier_setup.sh's own pin is the default (one source for the commit)
    assert EP.frontier_pin(PROD / "frontier_setup.sh") == ("https://github.com/NetX-lab/Frontier", "db5a85b")
