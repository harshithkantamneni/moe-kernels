"""Rental RR's plan, driver and launcher off the GPU (scripts/plans/rentalRR-2026-11.plan, scripts/gh200_rr_session.sh,
scripts/vm_run_rr.sh): the parse-only dry run listing every unit, held to the registration; the plan generated from
rrcommon; the driver against PC2's pinned driver and the launcher against PC2's pinned launcher (every changed line marked
RR); the new key value (kernels=rr) and the '-' model on a counter unit accepted only together; the plan's variants
accepted by RR's probe; the session on the fake box (every probe unit on RR's probe, every counter unit on RR's counter
route with its page and the 1410 lock, every lock read back after its tool; the regcheck drop rule dropping the probe
units while the counter units still run; the deadline's cut order); `vm_run_rr.sh start` delivering the driver, the plan
and every page byte for byte; and the driver-upgrade step (FLAGGED FOR THE OWNER) run through a stub VM whose apt, dpkg,
pgrep, fuser, sudo and timeout are stubs: a lock held then freed (waits, names the holder, installs), a lock never freed
(refuses within its bound, installs nothing), an install that hangs (the VM's timeout ends it, the fallback runs, then
prepare refuses), and every apt command's stdin /dev/null (a prompt cannot wait on the laptop)."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "rr", REPO / "scripts" / "scoring" / "pc2",
          REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rrcommon as P  # noqa: E402
import test_cal2_plan as TC2  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_pc_driver as TPC  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / P.PLAN
DRIVER = REPO / P.DRIVER
PC2_DRIVER = REPO / P.PARENT_DRIVER
LAUNCHER = REPO / P.LAUNCHER
PC2_LAUNCHER = REPO / P.PARENT_LAUNCHER
REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
PC2_REG = json.loads((REPO / "docs" / "registered" / "2026-11-10-pc2-gh200.json").read_text())
PG = P.PAGES_DIR


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_plan_is_generated_and_its_dry_run_lists_every_unit():
    assert PLAN.read_text() == P.plan_text()
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = REG["units"]["rows"]
    assert len(units) == len(reg) == len(P.UNITS) == 10 and "a PLAN of 10 units" in got.stdout
    import test_tileA2_plan as TA2
    import test_tile_plan as TT
    for u, r in zip(units, reg, strict=True):
        assert (u[2], u[1], TT._label(u) or "", int(u[3]), int(u[4]), TA2._drop(u)) == (
            r["step"], r["model"], r["label"], r["est"], r["cap"], r["drop"]), u
    figs = REG["plan"]
    assert sum(int(u[3]) for u in units) == figs["units_min"] == 43 and "43 min of units" in got.stdout
    assert (figs["wall_min"], figs["cost_usd"], figs["deadline_after_launch_min"], figs["cost_at_deadline_usd"],
            figs["hard_cap_after_launch_min"], figs["cost_at_hard_cap_usd"]) == (68, 2.6, 85, 3.24, 100, 3.82)
    assert (figs["nodrop_wall_min"], figs["nodrop_cost_usd"]) == (60, 2.29)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == figs["sha256"] and figs["drop_order_last_first"] == ["rep"]
    assert "u5-s128 u7-s64 u9-sr" in got.stdout and "the floor lock" not in got.stdout
    assert got.stdout.count("RR counter page at the 1410 lock") == 2
    assert got.stdout.count("(scripts/instr_probe_rr.py)") == 3 and "instr_probe_pc2.py" not in got.stdout


def test_every_unit_runs_at_the_session_lock_and_only_the_replicate_drops():
    rows = {u["label"]: u for u in REG["units"]["rows"]}
    assert not any("lock-mhz" in u["keys"] for u in REG["units"]["rows"] if u["step"] != "prelude")
    assert [u["label"] for u in REG["units"]["rows"] if u["drop"] != "nodrop"] == ["sr", "br"]
    assert rows["pt"]["keys"]["gates"] == "s128,s64,sr"
    for s, c in P.PAIRED.items():
        assert rows[s]["keys"]["page"] == rows[c]["keys"]["histogram"] and rows[c]["keys"]["kernels"] == "rr"


def test_the_plans_variants_are_the_ones_rrs_probe_accepts(tmp_path):
    import instr_probe_rr as PR
    got = subprocess.run(["bash", str(DRIVER), "--print-pc-variants", "--plan", str(PLAN)], capture_output=True, text=True,
                         timeout=60, env={**os.environ, "HOME": "/nonexistent"})
    f = tmp_path / "v.json"
    f.write_text(got.stdout)
    vs = PR.load_variants(f)
    assert [v["id"] for v in vs] == list(P.VARIANTS.values()) == ["u5-s128", "u7-s64", "u9-sr"]
    assert [len(v["cells"]) for v in vs] == [40, 40, 12]


def _held_to_parent(parent: Path, child: Path, header_end: str | None, mark: str = "RR") -> list[str]:
    a = parent.read_text().splitlines()
    b = child.read_text().splitlines()
    end = b.index(header_end) if header_end else 0
    changed = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        assert tag in ("replace", "insert") or j1 < end, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < end or mark in b[j], (j + 1, b[j])
            if j >= end:
                changed.append(b[j])
    for ln in b:
        assert chr(0x2014) not in ln and chr(0x2013) not in ln
    return changed


def test_the_rr_driver_is_the_pc2_driver_plus_lines_marked_rr():
    changed = _held_to_parent(PC2_DRIVER, DRIVER, "set -uo pipefail")
    assert len(changed) == 8, changed
    b, a = DRIVER.read_text().splitlines(), PC2_DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    assert b[0] == a[0] and "RENTAL RR'S DRIVER" in b[1]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE PC2 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    body = "\n".join(b[header:])
    assert body.count("scripts/instr_probe_rr.py") == 4 and "scripts/instr_probe_pc2.py" not in body
    assert body.count("scripts/dram_counter_route_rr.py") == 1
    assert hashlib.sha256(PC2_DRIVER.read_bytes()).hexdigest() == PC2_REG["code_pins"][P.PARENT_DRIVER] \
        == REG["code_pins"][P.PARENT_DRIVER]
    assert (DRIVER.stat().st_mode & 0o777) == (PC2_DRIVER.stat().st_mode & 0o777)


def test_the_rr_launcher_is_vm_run_pc2_plus_lines_marked_rr():
    changed = _held_to_parent(PC2_LAUNCHER, LAUNCHER, None)
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)", LAUNCHER.read_text(), re.M).group(1).split()
    assert drivers == ["gh200_rr_session.sh"]
    assert sum('local drv="${DRIVER_ARG:-gh200_rr_session.sh}"' in x for x in changed) == 1
    assert sum("ADOPTED BY THE OWNER" in x for x in changed) == 1
    assert hashlib.sha256(PC2_LAUNCHER.read_bytes()).hexdigest() == PC2_REG["code_pins"][P.PARENT_LAUNCHER] \
        == REG["code_pins"][P.PARENT_LAUNCHER]
    assert (LAUNCHER.stat().st_mode & 0o777) == (PC2_LAUNCHER.stat().st_mode & 0o777)


@pytest.mark.parametrize("body, why", [
    ("- prelude\n- decodebytes label=c histogram=" + PG + "/rr-128.json\n", "'-' is not a model name"),
    ("- prelude\nmixtral-8x7b decodebytes label=c histogram=" + PG + "/rr-128.json kernels=rrr\n", "kernels rrr: gemm, wide or rr"),
    ("- prelude\n- regcheck label=r probe=pc kernels=rr\n", "kernels belongs to a decodebytes unit"),
    ("- prelude\n- decodebytes label=c kernels=rr\n", "needs label= and histogram="),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG
     + "/nope.json\n", "no such file"),
])
def test_rrs_keys_are_refused_where_they_do_not_belong(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

FAKE_RR = r'''
if tool == "dram_counter_route_rr":
    if "--dry-run" in args:
        print("COUNTER ROUTE rr-counter (planted)")
        print("REFUSED. Nothing was measured: --dry-run was given")
        sys.exit(2)
    o = Path(val("--out")); o.parent.mkdir(parents=True, exist_ok=True); o.write_text("{}")
    print("RESULT: VALIDITY grid PASS planted")
    sys.exit(0)
'''


def fakepy_rr() -> str:
    body = TPC.fakepy_pc()
    assert body.count('if tool == "instr_probe_pc":') == 1
    body = body.replace('if tool == "instr_probe_pc":', 'if tool == "instr_probe_rr":')
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_RR + "sys.exit(0)" + tail


def rrbox(tmp_path, scenario, plan=None, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_rr())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "rr.plan"
    f.write_text(plan or PLAN.read_text())
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


def starts(led: str) -> list[str]:
    return re.findall(r"^\S+ (\w+) START", led, re.M)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return rrbox(tmp_path_factory.mktemp("rrclean"), {})


def test_every_unit_runs_on_rrs_tools_at_the_session_lock(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led[-3000:]
    assert starts(led) == ["prelude", "tilecensus", "regcheck", "perturb", "stamps", "decodebytes", "stamps",
                           "decodebytes", "stamps", "bytes"]
    pr = box.tool("instr_probe_rr", dry=False)
    assert [(TG.val(a, "--mode"), TG.val(a, "--variant") if "--variant" in a else TG.val(a, "--only") if "--only" in a
             else "") for a in pr] == [("regcheck", ""), ("perturb", "u5-s128,u7-s64,u9-sr"), ("stamps", "u5-s128"),
                                       ("stamps", "u7-s64"), ("stamps", "u9-sr")]
    assert not box.tool("instr_probe_pc2") and not box.tool("instr_probe_pc")
    cr = box.tool("dram_counter_route_rr", dry=False)
    assert [TG.val(a, "--page") for a in cr] == [P.PAGE_OF["s128"], P.PAGE_OF["s64"]]
    assert all(TG.val(a, "--page-lock-mhz") == "1410" and TG.val(a, "--out").endswith("lock1410/r3d.json") for a in cr)
    assert not box.tool("dram_counter_route_decode3") and not box.tool("dram_counter_route_decode")
    D = box.session / "gh200-driver"
    locks = (D / "pc-locks.env").read_text().splitlines()
    assert [ln.split("=")[0] for ln in locks] == ["PC_LOCK_mixtral_8x7b_pt", "PC_LOCK_mixtral_8x7b_s128",
                                                  "PC_LOCK_mixtral_8x7b_c128", "PC_LOCK_mixtral_8x7b_s64",
                                                  "PC_LOCK_mixtral_8x7b_c64", "PC_LOCK_phi_3_5_moe_sr"]
    assert all("=1410 before=1410 after=1410 exit=0" in ln for ln in locks), locks
    assert "PC DROP RULE" not in led and "DROPPED" not in led


@pytest.mark.parametrize("scenario, rc", [({"pc_regcheck_refuse": True}, "2"), ({"pc_regcheck_crash": 4}, "4")])
def test_a_failed_start_drops_every_probe_unit_and_the_counter_units_still_run(tmp_path, scenario, rc):
    box, got = rrbox(tmp_path, scenario)
    led = box.ledger()
    assert f"PC DROP RULE: the probe=pc regcheck read '{rc}'" in led, led[-3000:]
    for lab in P.PC_UNIT_LABELS:
        assert re.search(rf"DROPPED unit \d+/10 \S+ {lab} ", led), (lab, led[-3000:])
    assert starts(led) == ["prelude", "tilecensus", "regcheck", "decodebytes", "decodebytes", "bytes"]
    assert len(box.tool("dram_counter_route_rr", dry=False)) == 2


@pytest.mark.parametrize("minutes, dropped", [(85, []), (60, ["br", "sr"])])
def test_the_deadline_cuts_the_replicate_only(tmp_path, minutes, dropped):
    box, got = rrbox(tmp_path, {}, deadline=int(time.time()) + (minutes - 25) * 60)
    got_d = re.findall(r"DROPPED unit \d+/10 \S+ (\S+) ", box.ledger())
    assert got_d == dropped, box.ledger()[-3000:]


# --------------------------------------------------------------------------
# vm_run_rr.sh start --driver gh200_rr_session.sh
# --------------------------------------------------------------------------

def test_start_with_rr_delivers_the_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    pages = sorted(set(re.findall(r"(?:page|histogram)=(\S+)", PLAN.read_text())))
    assert len(pages) == 3
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN, lap.clone / P.PLAN)
    for f in (DRIVER, LAUNCHER, PC2_LAUNCHER):
        shutil.copy(f, lap.clone / "scripts" / f.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    launcher = lap.clone / "scripts" / LAUNCHER.name
    env = {**os.environ, "MOE_VM_SSH": str(lap.bin / "ssh"), "MOE_VM_SCP": str(lap.bin / "scp"),
           "MOE_VM_GH": str(lap.bin / "gh"), "MOE_VM_STATE_ROOT": str(lap.state), "MOE_VM_POLL_S": "0",
           "MOE_VM_WAIT_S": "5", "FAKE_VM": str(lap.vm), "FAKE_GH": str(lap.gh)}

    def run(*args, script=launcher):
        return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", P.PLAN, "--driver", DRIVER.name)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / DRIVER.name) == sha(DRIVER)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_rr_session.sh --dry-run" in cmds and "gh200_rr_session.sh --setup" in cmds
    pinned = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", P.PLAN,
                 "--driver", DRIVER.name, script=lap.clone / "scripts" / PC2_LAUNCHER.name)
    assert pinned.returncode == exit_codes.REFUSED and "--driver gh200_rr_session.sh: one of" in pinned.stdout + \
        pinned.stderr   # PC2's launcher does not start RR's driver


# --------------------------------------------------------------------------
# the driver-upgrade step against a held package lock (FLAGGED FOR THE OWNER)
# --------------------------------------------------------------------------

#: the VM's side: a command naming apt, dpkg, pgrep or fuser runs in a local bash whose PATH holds ONLY the stub tools
#: and a few basics (so a VM without fuser can be played); the rest is the model session's fake ssh
EXEC_SSH = r"""#!__PY__ -S
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
cmd = " ".join(args[1:])
st_path = Path(os.environ["FAKE_VM"])
if any(w in cmd for w in ("apt-get", "dpkg", "pgrep", "fuser")):
    st = json.loads(st_path.read_text()); st.setdefault("cmds", []).append(cmd); st_path.write_text(json.dumps(st))
    r = subprocess.run(["bash", "-c", cmd], cwd=os.environ["FAKE_VMHOME"],
                       env={**os.environ, "PATH": os.environ["FAKE_STUBS"]})
    sys.exit(r.returncode)
os.execv(os.environ["FAKE_BASE_SSH"], [os.environ["FAKE_BASE_SSH"], *sys.argv[1:]])
"""

STUBS = {
    "sudo": '#!/bin/bash\n[[ "$1" == -n ]] && shift\nwhile [[ "$1" == *=* ]]; do export "$1"; shift; done\nexec "$@"\n',
    # GNU timeout's contract: [--kill-after=N] DURATION CMD...; the command in its own process group, the GROUP signalled
    # (TERM, then KILL N s later) on expiry, exit 124
    "timeout": ('#!__PY__ -S\nimport os, signal, subprocess, sys\na = sys.argv[1:]\nka = 30.0\n'
                'while a[0].startswith("--"):\n    if a[0].startswith("--kill-after="): ka = float(a[0].split("=")[1])\n'
                '    a = a[1:]\nd = float(a[0])\np = subprocess.Popen(a[1:], start_new_session=True)\ntry:\n'
                '    sys.exit(p.wait(timeout=d))\nexcept subprocess.TimeoutExpired:\n'
                '    os.killpg(p.pid, signal.SIGTERM)\n    try:\n        p.wait(timeout=min(ka, 2.0))\n'
                '    except subprocess.TimeoutExpired:\n        os.killpg(p.pid, signal.SIGKILL); p.wait()\n'
                '    sys.exit(124)\n'),
    # pgrep honours -x and its pattern against the planted process names ($FAKE_PROCS: JSON {names, checks}); a check
    # without -a spends one of `checks` ('forever' never ends); no name matching, or checks spent: exit 1
    "pgrep": ('#!__PY__ -S\nimport json, os, re, sys\na = sys.argv[1:]\nlist_ = "-a" in a\n'
              'pat = [x for x in a if not x.startswith("-")][-1]\nf = os.environ["FAKE_PROCS"]\n'
              'st = json.load(open(f)) if os.path.exists(f) else {"names": [], "checks": 0}\n'
              'live = st["checks"] == "forever" or st["checks"] > 0\n'
              'hit = [n for n in st["names"] if ("-x" in a and re.fullmatch(pat, n)) or ("-x" not in a and re.search(pat, n))]\n'
              'if live and hit and not list_ and st["checks"] != "forever":\n'
              '    st["checks"] -= 1; json.dump(st, open(f, "w"))\n'
              'if live and hit:\n    print("\\n".join(f"4242 {n}" if list_ else "4242" for n in hit)); sys.exit(0)\n'
              'sys.exit(1)\n'),
    # fuser honours its file arguments against the planted held lock files ($FAKE_FILES: JSON {files, checks})
    "fuser": ('#!__PY__ -S\nimport json, os, sys\nf = os.environ["FAKE_FILES"]\n'
              'st = json.load(open(f)) if os.path.exists(f) else {"files": [], "checks": 0}\n'
              'held = [x for x in sys.argv[1:] if x in st["files"]]\n'
              'if held and (st["checks"] == "forever" or st["checks"] > 0):\n'
              '    if st["checks"] != "forever": st["checks"] -= 1; json.dump(st, open(f, "w"))\n'
              '    print(" ".join(f"{x}: 4243" for x in held), file=sys.stderr); sys.exit(0)\n'
              'sys.exit(1)\n'),
    # dpkg: --configure -a needs --force-confdef and --force-confold, and clears the interrupted state
    "dpkg": ('#!/bin/bash\necho "dpkg $*" >> "$FAKE_VMHOME/tools.log"\n'
             'if [[ " $* " == *" --configure -a "* || " $* " == *" --configure "* ]]; then\n'
             '  [[ " $* " == *" --force-confdef "* && " $* " == *" --force-confold "* ]] || exit 2\n'
             '  rm -f "$FAKE_VMHOME/dpkg-interrupted"; fi\nexit 0\n'),
    # apt-get: records itself and whether its stdin is /dev/null; refuses while dpkg is interrupted; 'hang' marks dpkg
    # interrupted, starts a child (its pid kept) and sleeps; 'prompt' reads stdin
    "apt-get": ('#!__PY__ -S\nimport os, subprocess, sys, time\nst = os.fstat(0); dn = os.stat("/dev/null")\n'
                'null = (st.st_rdev == dn.st_rdev and st.st_ino == dn.st_ino)\nh = os.environ["FAKE_VMHOME"]\n'
                'open(h + "/tools.log", "a").write("apt-get %s stdin_null=%s env=%s\\n" % ('
                '" ".join(sys.argv[1:]), null, os.environ.get("DEBIAN_FRONTEND")))\n'
                'mode = os.environ.get("FAKE_APT", "ok")\n'
                'if "install" in sys.argv:\n'
                '    if os.path.exists(h + "/dpkg-interrupted"):\n'
                '        print("E: dpkg was interrupted, you must manually run dpkg --configure -a"); sys.exit(100)\n'
                '    if mode == "hang":\n'
                '        open(h + "/dpkg-interrupted", "w").write("x")\n'
                '        c = subprocess.Popen(["sleep", "300"]); open(h + "/child.pid", "a").write(str(c.pid) + "\\n")\n'
                '        time.sleep(300)\n'
                '    if mode == "prompt": sys.stdin.read()\n'
                'sys.exit(0)\n'),
}
BASICS = ("bash", "sh", "sleep", "head", "tr", "cat", "rm")


def lock_laptop(tmp_path, procs=None, files=None, apt="ok", budget="20", fuser=True, interrupted=False):
    lap = TG.Laptop(tmp_path)
    shutil.copy(LAUNCHER, lap.clone / "scripts" / LAUNCHER.name)
    shutil.copy(DRIVER, lap.clone / "scripts" / DRIVER.name)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "rr launcher"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    stubs, home = tmp_path / "stubs", tmp_path / "vmhome2"
    stubs.mkdir()
    home.mkdir()
    for n, body in STUBS.items():
        if n == "fuser" and not fuser:
            continue
        TG._exe(stubs / n, body.replace("__PY__", sys.executable))
    for b in BASICS:
        os.symlink(shutil.which(b), stubs / b)
    if interrupted:
        (home / "dpkg-interrupted").write_text("x")
    (tmp_path / "procs.json").write_text(json.dumps(procs or {"names": [], "checks": 0}))
    (tmp_path / "files.json").write_text(json.dumps(files or {"files": [], "checks": 0}))
    (lap.bin / "ssh").rename(lap.bin / "ssh.base")
    TG._exe(lap.bin / "ssh", EXEC_SSH.replace("__PY__", sys.executable))
    env = {**os.environ, "MOE_VM_SSH": str(lap.bin / "ssh"), "MOE_VM_SCP": str(lap.bin / "scp"),
           "MOE_VM_GH": str(lap.bin / "gh"), "MOE_VM_STATE_ROOT": str(lap.state), "MOE_VM_POLL_S": "0",
           "MOE_VM_WAIT_S": "5", "MOE_VM_APT_BUDGET_S": budget, "MOE_VM_APT_MIN_S": "1", "FAKE_VM": str(lap.vm),
           "FAKE_GH": str(lap.gh), "FAKE_STUBS": str(stubs), "FAKE_VMHOME": str(home),
           "FAKE_PROCS": str(tmp_path / "procs.json"), "FAKE_FILES": str(tmp_path / "files.json"),
           "FAKE_BASE_SSH": str(lap.bin / "ssh.base"), "FAKE_APT": apt}
    t0 = time.time()
    got = subprocess.run(["bash", str(lap.clone / "scripts" / LAUNCHER.name), "prepare", "--ip", "10.0.0.1", "--run-id",
                          "r", "--branch", "run-gh200-t"], env=env, capture_output=True, text=True, timeout=120)
    log = (home / "tools.log").read_text() if (home / "tools.log").exists() else ""
    return got, log, time.time() - t0, lap, home


def _installed_with_the_fix(log: str) -> None:
    lines = log.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith("apt-get") and " install " in ln)
    assert any(ln.startswith("dpkg --force-confdef --force-confold --configure -a") for ln in lines[:i]), lines
    for ln in lines:
        if ln.startswith("apt-get"):
            assert "stdin_null=True" in ln and "--force-confold" in ln and "--force-confdef" in ln, ln


def test_a_process_holding_the_lock_is_waited_for_and_named_then_the_driver_installs(tmp_path):
    got, log, _s, lap, _h = lock_laptop(tmp_path, procs={"names": ["unattended-upgr", "sshd"], "checks": 3})
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.count("the package lock is held (") == 3 and "4242 unattended-upgr" in got.stdout
    assert "sshd" not in got.stdout and "locks are free" in got.stdout
    assert "install -y nvidia-driver-580-server-open=580.105.08" in log
    _installed_with_the_fix(log)
    installs = [c for c in lap.vm_cmds() if "install -y" in c]
    assert len(installs) == 1 and "timeout --kill-after=30" in installs[0] and "NEEDRESTART_MODE=a" in installs[0]


def test_a_lock_file_holder_alone_is_waited_for_and_named(tmp_path):
    got, log, _s, _lap, _h = lock_laptop(tmp_path, files={"files": ["/var/lib/dpkg/lock-frontend"], "checks": 4})
    assert got.returncode == 0, got.stdout + got.stderr
    assert "/var/lib/dpkg/lock-frontend: 4243" in got.stdout and "locks are free" in got.stdout
    _installed_with_the_fix(log)


def test_a_process_whose_name_does_not_match_the_pattern_holds_nothing(tmp_path):
    got, log, _s, _lap, _h = lock_laptop(tmp_path, procs={"names": ["apt-cacher", "dpkg-query-x"], "checks": "forever"})
    assert got.returncode == 0 and "lock is held" not in got.stdout, got.stdout


def test_a_lock_never_freed_refuses_inside_the_budget_and_installs_nothing(tmp_path):
    got, log, secs, lap, _h = lock_laptop(tmp_path, procs={"names": ["unattended-upgr"], "checks": "forever"}, budget="4")
    assert got.returncode == exit_codes.REFUSED and "held through the upgrade step's 4 s budget" in got.stdout, got.stdout
    assert "apt-get" not in log and secs < 60 and not any("reboot" in c for c in lap.vm_cmds())


def test_a_vm_without_fuser_fails_closed(tmp_path):
    got, log, _s, _lap, _h = lock_laptop(tmp_path, fuser=False, budget="4")
    assert got.returncode == exit_codes.REFUSED and "no fuser on the VM (psmisc): read as held" in got.stdout
    assert "apt-get" not in log


def test_an_interrupted_dpkg_left_before_the_step_is_finished_before_the_install(tmp_path):
    got, log, _s, _lap, home = lock_laptop(tmp_path, interrupted=True)
    assert got.returncode == 0, got.stdout + got.stderr
    assert not (home / "dpkg-interrupted").exists()
    _installed_with_the_fix(log)


def test_an_install_that_hangs_is_killed_with_its_process_group_and_prepare_refuses(tmp_path):
    got, log, secs, lap, home = lock_laptop(tmp_path, apt="hang", budget="6")
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "did not install (exit 124: the upgrade step's 6 s budget ended it, dpkg included)" in got.stdout
    assert "budget is spent" in got.stdout and "terminate the instance" in got.stdout and secs < 60
    for pid in (home / "child.pid").read_text().split():   # the hung apt-get's child died with its group
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)
    assert (home / "dpkg-interrupted").exists()            # the killed attempt left dpkg interrupted: not reused
    assert not any("reboot" in c for c in lap.vm_cmds())


def test_a_prompt_cannot_wait_on_the_laptop(tmp_path):
    got, log, _s, _lap, _h = lock_laptop(tmp_path, apt="prompt")
    assert got.returncode == 0 and "stdin_null=True" in log, got.stdout + got.stderr
