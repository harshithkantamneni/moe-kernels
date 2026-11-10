"""Rental PC2's plan, driver and launcher off the GPU (scripts/plans/rentalPC2-2026-11.plan, scripts/gh200_pc2_session.sh,
scripts/vm_run_pc2.sh): the parse-only dry run listing every unit, held to the registration; the plan generated from
pc2common; the driver against PC's pinned driver and the launcher against the pinned vm_run.sh (every changed line
marked PC2); PC's keys still refused where they do not belong; the plan's variants accepted by PC2's probe; then the
session on the fake box: every probe unit on PC2's probe at its own 1710 lock, read back before and after; the regcheck
drop rule (a refused or crashed start drops every probe unit, the census still ran); a variant's regcheck FAIL refusing
its stamps unit only; the deadline's cut order (rep, d3, w); and `vm_run_pc2.sh start --driver gh200_pc2_session.sh`
delivering the driver, the plan and every page byte for byte, through a stub ssh that drains its stdin."""
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
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc2", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pc2common as P  # noqa: E402
import test_cal2_plan as TC2  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_pc_driver as TPC  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / P.PLAN
DRIVER = REPO / P.DRIVER
PC_DRIVER = REPO / P.PARENT_DRIVER
LAUNCHER = REPO / P.LAUNCHER
VM_RUN = REPO / P.PARENT_LAUNCHER
REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
PC_REG = json.loads((REPO / "docs" / "registered" / "2026-11-06-pc-gh200.json").read_text())
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
    assert sum(int(u[3]) for u in units) == figs["units_min"] == 35 and "35 min of units" in got.stdout
    assert (figs["wall_min"], figs["cost_usd"], figs["deadline_after_launch_min"], figs["cost_at_deadline_usd"],
            figs["hard_cap_after_launch_min"], figs["cost_at_hard_cap_usd"]) == (60, 2.29, 75, 2.86, 90, 3.44)
    assert figs["deadline_after_launch_min"] - P.BOOT_MIN - P.RESERVE_MIN >= figs["units_min"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == figs["sha256"]
    assert figs["drop_order_last_first"] == ["rep", "d3", "w"]
    assert "instrumented units, each run only on its perturb unit's hard legs and regcheck (PERTURBED on a timing FAIL " \
           "alone): u5-sd u6-ss u7-sn u8-sw u9-sd3 u10-sr" in got.stdout
    assert "the prelude also probes the floor lock(s) 1710 MHz" in got.stdout
    assert "probe=pc units run scripts/instr_probe_pc2.py" in got.stdout
    assert got.stdout.count("(scripts/instr_probe_pc2.py)") == 6 and "instr_probe_pc.py" not in got.stdout


def test_every_probe_unit_carries_the_1710_lock_and_the_core_is_nodrop():
    rows = {u["label"]: u for u in REG["units"]["rows"]}
    for lab in P.PC_UNIT_LABELS:
        assert rows[lab]["keys"]["lock-mhz"] == "1710" and rows[lab]["keys"]["probe"] == "pc", lab
    assert [u["label"] for u in REG["units"]["rows"] if u["drop"] == "nodrop"] == ["", "pc2", "rcp", "pt", "sd", "ss", "sn"]
    assert rows["pt"]["keys"]["gates"] == "sd,ss,sn,sw,sd3,sr"


def test_the_plans_variants_are_the_ones_pc2s_probe_accepts(tmp_path):
    import instr_probe_pc2 as PR
    got = subprocess.run(["bash", str(DRIVER), "--print-pc-variants", "--plan", str(PLAN)], capture_output=True, text=True,
                         timeout=60, env={**os.environ, "HOME": "/nonexistent"})
    assert got.returncode == exit_codes.REFUSED, got.stderr
    f = tmp_path / "v.json"
    f.write_text(got.stdout)
    vs = PR.load_variants(f)
    assert [v["id"] for v in vs] == list(P.VARIANTS.values()) == ["u5-sd", "u6-ss", "u7-sn", "u8-sw", "u9-sd3", "u10-sr"]
    assert [v["page"] for v in vs] == [P.PAGE_OF[k] for k in P.VARIANTS]


def _held_to_parent(parent: Path, child: Path, header_end: str | None, mark: str = "PC2") -> list[str]:
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


def test_the_pc2_driver_is_the_pc_driver_plus_lines_marked_pc2():
    changed = _held_to_parent(PC_DRIVER, DRIVER, "set -uo pipefail")
    assert len(changed) == 4 and all("instr_probe_pc2.py" in x for x in changed)
    b = DRIVER.read_text().splitlines()
    a = PC_DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    assert b[0] == a[0] and "RENTAL PC2'S DRIVER" in b[1]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE PC DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    body = "\n".join(b[header:])
    assert body.count("scripts/instr_probe_pc2.py") == 4 and "echo scripts/instr_probe_pc.py" not in body
    assert hashlib.sha256(PC_DRIVER.read_bytes()).hexdigest() == PC_REG["code_pins"][P.PARENT_DRIVER] \
        == REG["code_pins"][P.PARENT_DRIVER]
    assert (DRIVER.stat().st_mode & 0o777) == (PC_DRIVER.stat().st_mode & 0o777)


def test_the_pc2_launcher_is_vm_run_plus_lines_marked_pc2():
    changed = _held_to_parent(VM_RUN, LAUNCHER, None)
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)", LAUNCHER.read_text(), re.M).group(1).split()
    assert drivers == ["gh200_pc2_session.sh"]
    assert sum(1 for x in changed if x.startswith("SESSION_DRIVERS=(")) == 1 and len(changed) == 7
    assert sum('local drv="${DRIVER_ARG:-gh200_pc2_session.sh}"' in x for x in changed) == 1
    assert hashlib.sha256(VM_RUN.read_bytes()).hexdigest() == PC_REG["code_pins"][P.PARENT_LAUNCHER] \
        == REG["code_pins"][P.PARENT_LAUNCHER]
    assert (LAUNCHER.stat().st_mode & 0o777) == (VM_RUN.stat().st_mode & 0o777)
    usage = subprocess.run(["bash", str(LAUNCHER), "--help"], capture_output=True, text=True, timeout=30)
    assert "THE LAPTOP'S SIDE OF AN UNATTENDED VM RUN" in usage.stdout


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=f probe=pc byte-groups=64\n", "probe belongs to a regcheck, perturb or stamps"),
    ("- prelude\n- regcheck label=r probe=pc page=" + PG + "/pc2-d.json\n", "page belongs to a probe=pc stamps unit"),
    ("- prelude\n- regcheck label=r probe=pc lock-mhz=1710\n", "lock-mhz belongs to the prelude"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=x\n- stamps label=s probe=pc page=" + PG
     + "/pc2-d.json\n", "needs a probe=pc perturb unit before it naming it"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc\n", "names the stamps units it gates"),
    ("- prelude\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG + "/pc2-d.json\n",
     "needs a probe=pc regcheck unit before it"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s,t\n- stamps label=s probe=pc page=" + PG
     + "/pc2-d.json\n", "gates=t names no probe=pc stamps unit"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG
     + "/nope.json\n", "no such file"),
    ("- prelude\n- regcheck label=r probe=pc2\n", "probe pc2: pc or c3"),
])
def test_pcs_keys_are_still_refused_where_they_do_not_belong(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

def fakepy_pc2() -> str:
    body = TPC.fakepy_pc()
    assert body.count('if tool == "instr_probe_pc":') == 1
    return body.replace('if tool == "instr_probe_pc":', 'if tool == "instr_probe_pc2":')


def pc2box(tmp_path, scenario, plan=None, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_pc2())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "pc2.plan"
    f.write_text(plan or PLAN.read_text())
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


def starts(led: str) -> list[str]:
    return re.findall(r"^\S+ (\w+) START", led, re.M)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return pc2box(tmp_path_factory.mktemp("pc2clean"), {})


def test_every_unit_runs_on_pc2s_probe_at_its_own_lock(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led[-3000:]
    assert starts(led) == ["prelude", "tilecensus", "regcheck", "perturb"] + ["stamps"] * 6
    D = box.session / "gh200-driver"
    pcv = json.loads((D / "pc-variants.json").read_text())
    assert [v["id"] for v in pcv] == list(P.VARIANTS.values())
    pr = box.tool("instr_probe_pc2", dry=False)
    assert [(TG.val(a, "--mode"), TG.val(a, "--variant") if "--variant" in a else TG.val(a, "--only") if "--only" in a
             else "") for a in pr] == [("regcheck", ""), ("perturb", ",".join(P.VARIANTS.values()))] + \
        [("stamps", v) for v in P.VARIANTS.values()]
    assert all(TG.val(a, "--variants").endswith("pc-variants.json") for a in pr)
    assert not box.tool("instr_probe_pc") and not box.tool("instr_probe_c3")
    locks = (D / "pc-locks.env").read_text().splitlines()
    assert [ln.split("=")[0] for ln in locks] == [f"PC_LOCK_mixtral_8x7b_{lab}" for lab in P.PC_UNIT_LABELS]
    assert all("=1710 before=1710 after=1710 exit=0" in ln for ln in locks), locks
    assert "LOCK_PROBE_1710=held" in (D / "locks.env").read_text()
    assert "PC DROP RULE" not in led and "DROPPED" not in led


@pytest.mark.parametrize("scenario, rc", [({"pc_regcheck_refuse": True}, "2"), ({"pc_regcheck_crash": 4}, "4"),
                                          ({"pc_regcheck_crash": 1}, "1")])
def test_a_regcheck_that_refuses_or_crashes_drops_every_probe_unit(tmp_path, scenario, rc):
    box, got = pc2box(tmp_path, scenario)
    led = box.ledger()
    assert f"PC DROP RULE: the probe=pc regcheck read '{rc}'" in led, led[-3000:]
    for lab in P.PC_UNIT_LABELS:
        assert re.search(rf"DROPPED unit \d+/10 \S+ {lab} .*owner decision 2", led), (lab, led[-3000:])
    assert starts(led) == ["prelude", "tilecensus", "regcheck"]
    assert [TG.val(a, "--mode") for a in box.tool("instr_probe_pc2", dry=False)] == (["regcheck"] if rc != "2" else [])


def test_a_variants_regcheck_fail_refuses_that_stamps_unit_only(tmp_path):
    box, got = pc2box(tmp_path, {"pc_regcheck_fail": ["u8-sw"]})
    led = box.ledger()
    assert "PC DROP RULE" not in led
    st = [TG.val(a, "--variant") for a in box.tool("instr_probe_pc2", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert st == [v for v in P.VARIANTS.values() if v != "u8-sw"], led[-3000:]
    assert "stamps REFUSED: its dry run exited 2 (REFUSED: planted)" in led
    per = [TG.val(a, "--only") for a in box.tool("instr_probe_pc2", dry=False) if TG.val(a, "--mode") == "perturb"]
    assert per == [",".join(P.VARIANTS.values())]   # its cells keep their PLAIN medians


@pytest.mark.parametrize("minutes, dropped", [
    (75, []),                       # the registered deadline: nothing dropped
    (66, None),                     # near the edge: a non-empty prefix of the cut order (rep first)
    (60, ["sr", "sd3", "sw"]),      # rep, d3, then w; the nodrop core runs
])
def test_the_deadline_cuts_in_the_registered_order(tmp_path, minutes, dropped):
    box, got = pc2box(tmp_path, {}, deadline=int(time.time()) + (minutes - 25) * 60)
    led = box.ledger()
    got_d = re.findall(r"DROPPED unit \d+/10 \S+ (\S+) ", led)
    if dropped is None:
        assert got_d and got_d == ["sr", "sd3", "sw"][:len(got_d)], led[-4000:]
    else:
        assert got_d == dropped, led[-4000:]
    assert "sd" not in got_d and "sn" not in got_d and "ss" not in got_d


# --------------------------------------------------------------------------
# vm_run_pc2.sh start --driver gh200_pc2_session.sh
# --------------------------------------------------------------------------

def test_start_with_pc2_delivers_the_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    pages = sorted(set(re.findall(r"page=(\S+)", PLAN.read_text())))
    assert len(pages) == 6
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN, lap.clone / P.PLAN)
    shutil.copy(DRIVER, lap.clone / "scripts" / DRIVER.name)
    shutil.copy(LAUNCHER, lap.clone / "scripts" / LAUNCHER.name)
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
           "MOE_VM_WAIT_S": "5", "FAKE_VM": str(lap.vm), "FAKE_GH": str(lap.gh)}   # Laptop.run's, on this launcher

    def run(*args, script=launcher):
        return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')   # the stub drains its stdin
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
    assert "gh200_pc2_session.sh --dry-run" in cmds and "gh200_pc2_session.sh --setup" in cmds
    assert "driver_script=gh200_pc2_session.sh" in (lap.state / "r" / "run.env").read_text()
    pinned = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", P.PLAN,
                 "--driver", DRIVER.name, script=lap.clone / "scripts" / "vm_run.sh")
    assert pinned.returncode == exit_codes.REFUSED and "--driver gh200_pc2_session.sh: one of" in pinned.stdout + \
        pinned.stderr   # the pinned vm_run.sh does not start PC2's driver


def test_the_pre_setup_dry_run_prints_the_pc2_plan_from_the_files_the_launcher_copies(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(DRIVER, home / DRIVER.name)
    shutil.copy(PLAN, home / PLAN.name)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    args = ["bash", str(home / DRIVER.name), "--dry-run", "--plan", P.PLAN]
    bare = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert "no such file" in bare.stdout + bare.stderr
    for p in sorted(set(re.findall(r"page=(\S+)", PLAN.read_text()))):
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    got = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 10 units" in got.stdout, got.stdout[-2000:] + got.stderr
