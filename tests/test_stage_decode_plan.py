"""The decode CAL plan and the decode stage's driver off the GPU (scripts/plans/rentalD-cal-2026-10.plan,
scripts/gh200_decode_session.sh): the parse-only dry run, its minutes, nodrop units and drop order;
the decode keys and where they are refused; the driver against the tile driver (every changed line
marked DECODE); the units reaching their tools on the fake box (r3_decode.py through locked_r3's
--r3-script at G 1, the counter wrapper under the lock); and `vm_run.sh start --driver
gh200_decode_session.sh` with the real plan against a stub VM whose ssh drains stdin, every delivered
file compared by sha256."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_gh200_model_session as TG  # noqa: E402
import test_tile_plan as TT  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalD-cal-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_decode_session.sh"
TILE_DRIVER = TT.TILE_DRIVER
PAGES = "docs/registered/2026-10-14-decode-pages"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def test_the_cal_plan_parses_in_106_minutes_and_drops_rep_decl_ncu_e240_last_first():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    assert len(units) == 13 and "a PLAN of 13 units" in got.stdout
    assert sum(int(u[3]) for u in units) == 106 and "106 min of units" in got.stdout
    assert [u[2] for u in units[:3]] == ["prelude", "calibrate", "tilecensus"]
    nodrop = {(u[1], u[2], TT._label(u)) for u in units if "[nodrop]" in u[5]}
    assert nodrop == {("mixtral-8x7b", "calibrate", "dcal"), ("-", "tilecensus", "dec"),
                      ("olmoe-1b-7b", "r3decode", "dm"), ("mixtral-8x7b", "r3decode", "dm"),
                      ("olmoe-1b-7b", "r3decode", "dc"), ("mixtral-8x7b", "r3decode", "dc")}
    seen = []
    for u in reversed(units):
        g = TT._group(u)
        if g and g not in seen:
            seen.append(g)
    assert seen == ["rep", "decl", "ncu", "e240"]
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == ["13", "12", "11", "10", "9", "8"]
    dirs = [TG._unit_dir(u[5].replace(" [nodrop]", "")) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b r3decode label=a histogram=PAGE slip-policy=cell\n", "names its instrument: inst="),
    ("- prelude\nolmoe-1b-7b r3decode label=a histogram=PAGE inst=GRR\n", "runs slip-policy=cell"),
    ("- prelude\nolmoe-1b-7b r3decode label=a inst=GRR\n", "names its decode page: histogram="),
    ("- prelude\nolmoe-1b-7b r3decode label=a histogram=PAGE inst=GR2 slip-policy=cell\n", "inst GR2: GRR, GR1 or E240"),
    ("- prelude\nolmoe-1b-7b r3decode label=a histogram=PAGE inst=GRR rot=0 slip-policy=cell\n", "rot 0: auto"),
    ("- prelude\nolmoe-1b-7b timed label=a histogram=PAGE inst=GRR declared-copies=9\n", "inst belongs to an r3decode unit"),
    ("- prelude\nolmoe-1b-7b decodebytes label=a histogram=PAGE inst=GRR\n", "inst belongs to an r3decode unit"),
    ("- prelude\nolmoe-1b-7b decodebytes label=a histogram=PAGE slip-policy=cell\n", "slip-policy belongs to a timed or r3decode unit"),
    ("- prelude\nolmoe-1b-7b r3decode histogram=PAGE inst=GRR slip-policy=cell\n", "names its directory: label="),
])
def test_the_decode_keys_are_refused_where_they_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body.replace("PAGE", f"{PAGES}/olmoe-1b-7b-main.json"))
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_tile_driver_refuses_the_decode_steps():
    got = dry("--plan", str(PLAN), driver=TILE_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "no step 'r3decode'" in got.stderr


def test_the_decode_driver_is_the_tile_driver_plus_lines_marked_decode():
    a = TILE_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "DECODE" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION")
    assert len(b) - len(a) < 120


def test_the_lambda_stop_instruction_names_the_decode_driver():
    text = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_decode_session\.sh)'", text).group(1)
    assert re.search(pat, "bash gh200_decode_session.sh --plan x") and not re.search(pat, "gh200_tile_session.sh")


FAKE_DECODE = r'''
if tool == "r3_decode":
    if "--dry-run" in args:
        print("DECODE PAGE (planted)")
        print("session     " + str(val("--session-tag")))
        print("REFUSED. Nothing was measured and nothing was written.")
        sys.exit(2)
    sys.exit(99)   # the driver never runs it itself: locked_r3.py does
if tool == "dram_counter_route_decode":
    if "--dry-run" in args:
        print("COUNTER ROUTE r3-decode (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}")
    print("RESULT: VALIDITY grid PASS planted")
    sys.exit(0)
'''


def fakepy_decode() -> str:
    body = TT.fakepy_tile()
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_DECODE + "sys.exit(0)" + tail


PLAN_BOX = f"""- prelude
mixtral-8x7b calibrate label=dcal nodrop=1 est=2 cap=5
- tilecensus label=dec tiles=16x64x128w4s4,32x64x128w4s3 models=olmoe-1b-7b nodrop=1 est=2 cap=5
olmoe-1b-7b r3decode label=dm histogram={PAGES}/olmoe-1b-7b-main.json inst=GRR slip-policy=cell g1-rule=r7-median-run3 est=2 cap=5
mixtral-8x7b r3decode label=dc histogram={PAGES}/mixtral-8x7b-ctl.json inst=GR1 rot=auto slip-policy=cell g1-rule=r7-median-run3 est=2 cap=5
olmoe-1b-7b decodebytes label=dn histogram={PAGES}/olmoe-1b-7b-ctl.json est=2 cap=5
"""


@pytest.fixture(scope="module")
def decodebox(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("decodebox")
    box = TG.Box(tmp, {"retracted": 40})
    TG._exe(box.bin / "fakepy", fakepy_decode())
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp / "decode.plan"
    f.write_text(PLAN_BOX)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(), capture_output=True, text=True, timeout=600)
    return box, got


def test_the_decode_units_run_and_reach_their_tools(decodebox):
    box, got = decodebox
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert starts == ["prelude", "calibrate", "tilecensus", "r3decode", "r3decode", "decodebytes"]
    lr = box.tool("locked_r3")
    assert len(lr) == 2
    for argv, page, inst in ((lr[0], "olmoe-1b-7b-main.json", "GRR"), (lr[1], "mixtral-8x7b-ctl.json", "GR1")):
        i = argv.index("--")
        mine, r3 = argv[:i], argv[i + 1:]
        assert TG.val(mine, "--r3-script") == "scripts/r3_decode.py" and mine[mine.index("--groups") + 1:mine.index("--groups") + 2] == ["1"] and mine[mine.index("--groups") + 2].startswith("--")
        assert TG.val(mine, "--slip-policy") == "cell" and TG.val(mine, "--min-clean-repeats") == "0"
        assert TG.val(mine, "--locks") == "1710"
        assert TG.val(r3, "--histogram") == f"{PAGES}/{page}" and TG.val(r3, "--inst") == inst
        assert TG.val(r3, "--rot") == "auto" and TG.val(r3, "--repeats") == "9" and TG.val(r3, "--duty") == "0.25"
        import locked_r3 as LR
        assert LR.build_parser().parse_args(mine).r3_script == "scripts/r3_decode.py"
    dry_ = box.tool("r3_decode", dry=True)
    assert len(dry_) == 2 and all(TG.val(a, "--group-m") == "1" and TG.val(a, "--lock-mhz") == "1710" for a in dry_)
    dd = [a for a in box.tool("dram_counter_route_decode") if "--run" in a]
    assert len(dd) == 1 and TG.val(dd[0], "--family") == "r3-decode" and TG.val(dd[0], "--page-lock-mhz") == "1710"
    assert TG.val(dd[0], "--out").endswith("-olmoe-1b-7b-dn-r3-counters/lock1710/r3d.json")
    assert "decodebytes: RESULT: VALIDITY grid PASS planted" in led
    # the counter page ran under the lock, and the lock was reset after
    smi = [" ".join(a) for a in box.smi()]
    assert any("-lgc 1710,1710" in s for s in smi) and any("-rgc" in s for s in smi)


def test_the_real_r3_decode_dry_run_of_a_unit_passes_the_drivers_checks(tmp_path):
    body = (f"- prelude\nmixtral-8x7b calibrate label=dcal nodrop=1 est=2 cap=5\n"
            f"olmoe-1b-7b r3decode label=dm histogram={PAGES}/olmoe-1b-7b-main.json inst=GRR slip-policy=cell "
            "g1-rule=r7-median-run3 nodrop=1 est=2 cap=5\n")
    box = TG.Box(tmp_path, {"calibrate": {"rc": 4, "results": []}})
    py = fakepy_decode().replace('if tool == "r3_decode":\n    if "--dry-run" in args:',
                                 'if tool == "r3_decode":\n    if "--dry-run" in args and os.environ.get("FAKE_REAL_DRY"):\n'
                                 '        real("r3_decode.py")\n    if "--dry-run" in args:')
    TG._exe(box.bin / "fakepy", py)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "t.plan"
    f.write_text(body)
    subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                   env=box.env(FAKE_REAL_DRY=1), capture_output=True, text=True, timeout=900)
    logs = sorted((box.session / "logs").glob("r3decode-olmoe-1b-7b-dm-dry.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert text.startswith("DECODE PAGE") and "reason: --dry-run was given" in text and "REFUSED:" not in text, text[-2000:]
    led = box.ledger()
    assert "r3decode REFUSED" not in led, led


def test_start_with_the_cal_plan_delivers_the_decode_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 7
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN, lap.clone / plan_rel)
    shutil.copy(DRIVER, lap.clone / "scripts" / DRIVER.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-d"], check=True, capture_output=True)
    lap.make_branch(branch="run-gh200-d")
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-d").returncode == 0
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    log = tmp_path / "scp.log"
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        f'printf "%s %s\\n" "$src" "$dst" >> "{log}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", plan_rel,
            "--driver", DRIVER.name)
    sent = [ln.split(" ", 1) for ln in log.read_text().splitlines()] if log.exists() else []
    got = {(Path(src).resolve(), dst) for src, dst in sent}
    assert (lap.clone.resolve() / plan_rel, "ubuntu@1.2.3.4:") in got, sent
    assert (lap.clone.resolve() / "scripts" / DRIVER.name, "ubuntu@1.2.3.4:") in got, sent
    for p in pages:
        assert (lap.clone.resolve() / p, f"ubuntu@1.2.3.4:{p}") in got, (p, sent)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / DRIVER.name) == sha(DRIVER)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_decode_session.sh --dry-run" in cmds and "gh200_decode_session.sh --setup" in cmds
    assert "driver_script=gh200_decode_session.sh" in (lap.state / "r" / "run.env").read_text()


def test_the_pre_setup_dry_run_prints_the_cal_plan_from_the_files_vm_run_copies(tmp_path):
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(DRIVER, home / DRIVER.name)
    shutil.copy(PLAN, home / PLAN.name)
    for p in pages:
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    got = subprocess.run(["bash", str(home / DRIVER.name), "--dry-run", "--plan", f"scripts/plans/{PLAN.name}"],
                         capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 13 units" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
    assert got.stdout.startswith("THE GH200 MODEL-TEST SESSION")
