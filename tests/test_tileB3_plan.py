"""Tile rental B3's plan and driver off the GPU (scripts/plans/rentalB3-tile-2026-11.plan,
scripts/gh200_tileB3_session.sh): the parse-only dry run, its units held to the registration
(docs/registered/2026-11-04-tileB3-gh200 'plan': step, model, label, est, cap, nodrop or drop group, the totals, wall,
cost, deadline and hard cap, the plan's sha256), the timed and byte units held to B2's frozen unit list
(finalise.plan_b's lines) and B2's pages by sha256, the f64w8 units held to A2's, the driver against C2's pinned
driver (every changed line marked B3), vm_run.sh and LAMBDA.md naming the driver; then the session on the fake box
(C2's power and lock stops, the B3 census gate, G-BAND, the B3 sag gate and the 64x128 sag gate before any timed unit,
each FAIL a STOP) and
`vm_run.sh start --driver gh200_tileB3_session.sh` against a stub VM, every file by sha256."""
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
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "tileA", REPO / "scripts" / "scoring" / "tileB3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_cal2_plan as TC2  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_settleS1_plan as TS1  # noqa: E402
import test_tile_plan as TT  # noqa: E402
import test_tileA2_plan as TA2  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalB3-tile-2026-11.plan"
DRIVER = REPO / "scripts" / "gh200_tileB3_session.sh"
C2_DRIVER = TC2.DRIVER
REG = json.loads((REPO / "docs" / "registered" / "2026-11-04-tileB3-gh200.json").read_text())
B2 = json.loads((REPO / "docs" / "registered" / "2026-10-20-tileB2-e1tile-gh200.json").read_text())
C2_REG = TC2.REG
TILE = TC2.TILE
HB = "docs/registered/2026-10-14-tileB-hist"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def comments(text: str) -> str:
    return " ".join(ln[1:].strip() for ln in text.splitlines() if ln.startswith("#"))


def body_lines(text: str) -> list[str]:
    return [ln for ln in (x.split("#", 1)[0].strip() for x in text.splitlines()) if ln]


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_b3_plan_parses_and_every_unit_is_the_registrations():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = REG["plan"]
    assert len(units) == reg["units_count"] == 23 and "a PLAN of 23 units" in got.stdout
    for u, r in zip(units, reg["units"], strict=True):
        label = TT._label(u) or ""
        assert (u[2], u[1], label, int(u[3]), int(u[4]), TA2._drop(u)) == (r["step"], r["model"], r["label"], r["est"],
                                                                             r["cap"], r["drop"]), u
    total = sum(int(u[3]) for u in units)
    assert total == reg["est_total_min"] == 143 and f"{total} min of units" in got.stdout
    nodrop = sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop")
    assert nodrop == reg["est_nodrop_min"] == 112 and reg["drop_order_last_first"] == ["byt", "srep", "dsv"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    c = comments(PLAN.read_text())
    assert "143 of units estimated (112 nodrop)" in c
    assert 186 - 25 - 8 >= total + 10   # every unit fits the deadline less boot and the driver's reserve, 10 min spare
    assert "--deadline <launch + 186 min, epoch s>" in c and "guardian hard cap launch + 201 min ($7.67 at most)" in c
    assert reg["wall_min"] == 168 and reg["cost_usd"] == 6.41 and reg["deadline_after_launch_min"] == 186
    assert reg["hard_cap_after_launch_min"] == 201 and reg["cost_at_hard_cap_usd"] == 7.67 and reg["rate_usd_per_h"] == 2.29
    assert "about 168 min ($6.41 at $2.29/h)" in c
    body = "\n".join(body_lines(PLAN.read_text()))
    assert "- prelude lock-mhz=1410 power-limit-w=700" in body and "power-limit-w=900" not in body
    assert sorted(set(re.findall(r"lock-mhz=(\d+)", body))) == ["1410"]
    for absent in ("lockprobe", "powerproof", "replay=", "stall-metrics", "stamps", "regcheck", "perturb", "unless-lock"):
        assert absent not in body, absent
    assert "the session lock 1410 MHz" in got.stdout


def test_every_stop_comes_before_the_first_timed_unit():
    u = REG["plan"]["units"]
    first_timed = next(i for i, x in enumerate(u) if x["step"] == "timed")
    gates = [(i, x["keys"]["gate"]) for i, x in enumerate(u) if x["step"] == "gate"]
    assert [g for _i, g in gates] == ["b3census", "band", "b3sag", "b3sag64"]
    assert all(i < first_timed for i, _g in gates)
    idx = {(x["model"], x["step"], x["label"]): i for i, x in enumerate(u)}
    assert idx[("-", "tilecensus", "tb")] < idx[("-", "gate", "gcen")] < idx[("olmoe-1b-7b", "bytes", "f128")]
    assert "launchfloor" not in {x["step"] for x in u} and "b3" + "align" not in json.dumps(u)
    assert idx[("olmoe-1b-7b", "bytes", "f128")] < idx[("mixtral-8x7b", "bytes", "f128")] < idx[("-", "gate", "gband")] \
        < idx[("-", "gate", "gsag")] < idx[("olmoe-1b-7b", "bytes", "f64w8")] < idx[("mixtral-8x7b", "bytes", "f64w8")] \
        < idx[("-", "gate", "gsag64")] < first_timed
    for x in u[:first_timed + 7]:   # the prelude, the census, every gate, both f128 pages, the seven blind timed pages
        assert x["drop"] == "nodrop", x
    assert REG["stop_rule"]["order"][-1] == "the timed units"


def test_the_timed_and_byte_units_are_b2s_frozen_lines_unchanged():
    import finalise as FN
    want = [ln for ln in FN.plan_b().splitlines() if not ln.startswith("#") and (" timed " in ln or " bytes " in ln)]
    got = [ln for ln in body_lines(PLAN.read_text()) if (" timed " in ln or " bytes label=b" in ln)]
    assert got == want and len(want) == 11
    k = {(x["model"], x["label"]): x["keys"] for x in REG["plan"]["units"]}
    c2 = {(x["model"], x["label"]): x["keys"] for x in C2_REG["plan"]["units"]}
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):   # C2's (and C4's) f128 pages, unchanged
        assert k[(m, "f128")] == c2[(m, "f128")] and k[(m, "f128")]["byte-groups"] == "64"
    assert k[("olmoe-1b-7b", "f128r")] == c2[("olmoe-1b-7b", "f128")]
    assert k[("-", "gband")] == c2[("-", "gband")] == {"gate": "band"}
    a2 = {}
    for ln in body_lines((REPO / "scripts" / "plans" / "rentalA2-tile-2026-10.plan").read_text()):
        w = ln.split()
        kv = dict(x.split("=", 1) for x in w[2:])
        if kv.get("label") == "f64w8":
            a2[w[0]] = {x: v for x, v in kv.items() if x not in ("label", "est", "cap", "nodrop", "drop-group")}
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):   # A2's f64w8 pages, unchanged
        assert k[(m, "f64w8")] == a2[m] and k[(m, "f64w8")]["block-m"] == "64" and k[(m, "f64w8")]["byte-groups"] == "64"
    assert k[("-", "tb")] == {"tiles": "128x128x64w8s3,64x128x64w8s3",
                              "models": "deepseek-v2-lite,jetmoe-8b,phi-3.5-moe,qwen1.5-moe-a2.7b"}


def test_the_pages_the_plan_names_are_b2s_registered_pages_at_their_sha256():
    body = "\n".join(body_lines(PLAN.read_text()))
    files = set(re.findall(r"histogram=(\S+)", body))
    b2 = {v["path"]: v["sha256"] for v in B2["pages"]["files"].values()}
    reg = {v["path"]: v["sha256"] for v in REG["pages"]["files"].values()}
    assert files == set(b2) == set(reg) and len(files) == 8 and reg == b2
    for f in files:
        assert f.startswith(HB) and hashlib.sha256((REPO / f).read_bytes()).hexdigest() == b2[f]
    assert hashlib.sha256((REPO / "docs/registered/2026-10-20-tileB2-e1tile-gh200.json").read_bytes()).hexdigest() == \
        REG["pages"]["B2 registration"]["sha256"]


def test_the_b3_driver_is_the_c2_driver_with_only_b3_marked_lines_changed():
    a = C2_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    assert b[header:].count("set -uo pipefail") == 1
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    body = [h for h in hunks if h[3] >= header]
    # the gate's description; the gate's dispatch (the b3 branch before C2's call, the fi after it); the gate values
    assert len(body) == 4, body
    for tag, i1, i2, j1, j2 in body:
        assert tag in ("replace", "insert"), (tag, a[i1:i2])
        for j in range(j1, j2):
            assert "B3" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and "TILE RENTAL B3'S DRIVER" in b[1]
    assert "# THE C2 DRIVER'S OWN HEADER, UNCHANGED:" in b[:header]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE C2 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    assert len(b) - len(a) < 40
    assert hashlib.sha256(C2_DRIVER.read_bytes()).hexdigest() == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert REG["code_pins"]["scripts/gh200_cal2_session.sh"] == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert (DRIVER.stat().st_mode & 0o777) == (C2_DRIVER.stat().st_mode & 0o777)
    assert REG["code_pins"]["scripts/gh200_tileB3_session.sh"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    text = DRIVER.read_text()
    assert 'scripts/tileB3_gate.py "${GATE_NAME#b3}"' in text and "b3census|b3sag|b3sag64" in text and "b3" + "align" not in text


def test_c2s_driver_refuses_the_b3_gates_and_the_b3_driver_parses_c2s_plan():
    got = dry("--plan", str(PLAN), driver=C2_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of" not in got.stdout
    assert "gate b3census" in got.stdout + got.stderr
    got = dry("--plan", str(TC2.PLAN))
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 10 units" in got.stdout


def test_vm_run_and_lambda_name_the_b3_driver_and_nothing_else_moved():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:9] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh",
                           "gh200_tileA2_session.sh", "gh200_settleS1_session.sh", "gh200_cal2_session.sh",
                           "gh200_cal3_session.sh", "gh200_cal4_session.sh", "gh200_decode3_session.sh"]
    assert "gh200_tileB3_session.sh" in drivers
    line = vm.splitlines()[5]
    assert "|gh200_cal4_session.sh|gh200_tileB3_session.sh|gh200_decode3_session.sh]" in line
    lam = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_tileB3_session\.sh)'", lam).group(1)
    assert re.search(pat, "bash gh200_tileB3_session.sh --plan x") and not re.search(pat, "gh200_tileA2_session.sh")
    for other in (r"[g]h200_tileA2_session.sh", r"[g]h200_cal2_session.sh", r"[g]h200_(model|tile)_session.sh"):
        assert not re.search(other, "gh200_tileB3_session.sh")
    assert "tile rental B3's copy (2026-11-04) with" in lam


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

FAKE_B3 = r'''
if tool == "tileB3_gate":
    key = "b3" + args[0]
    bad = key in SC.get("gate_fail", [])
    rec = val("--record")
    if rec:
        Path(rec).parent.mkdir(parents=True, exist_ok=True)
        Path(rec).write_text(json.dumps({"gate": args[0], "verdict": "FAIL" if bad else "PASS"}))
    print(f"RESULT: GATE {key} {'FAIL planted' if bad else 'PASS'}" + {"sag": " par A 48.0000 us tau 500 us",
          "sag64": " par A 30.3650 us tau 300 us"}.get(args[0], ""))
    sys.exit(3 if bad else 0)
'''


def fakepy_b3() -> str:
    body = TS1.fakepy_s1()
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_B3 + "sys.exit(0)" + tail


H = HB
PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=700 est=1 cap=5
mixtral-8x7b calibrate label=tb nodrop=1 est=1 cap=5
- tilecensus label=tb tiles=128x128x64w8s3,64x128x64w8s3 models=jetmoe-8b,phi-3.5-moe nodrop=1 est=1 cap=5
- gate label=gcen gate=b3census nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f128 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gband gate=band nodrop=1 est=1 cap=5
- gate label=gsag gate=b3sag nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f64w8 block-m=64 block-n=128 block-k=64 num-warps=8 num-stages=3 byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gsag64 gate=b3sag64 nodrop=1 est=1 cap=5
jetmoe-8b timed label=t128g1 {TILE} histogram={H}/jetmoe-8b-128g1.json arms=native,shared declared-copies=9 timed-groups=1 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=1 cap=5
phi-3.5-moe bytes label=b64g1 block-m=64 block-n=128 block-k=64 num-warps=8 num-stages=3 histogram={H}/phi-3.5-moe-64g1.json arms=native,shared byte-groups=1 drop-group=byt est=1 cap=5
"""


def b3box(tmp_path, scenario, plan=PLAN_BOX, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_b3())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "b3.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return b3box(tmp_path_factory.mktemp("b3clean"), {})


def test_every_unit_runs_at_700_w_and_1410_mhz_and_both_b3_gates_run_through_their_file(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "gate", "bytes", "gate",
                                                          "gate", "bytes", "gate", "timed", "bytes"]
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "700", "-sc", "0"]]
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_W=700\n"
    assert "LOCK_PROBE_1410=held" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert [(a[0], TG.val(a, "--lock")) for a in box.tool("tileA2_gate")] == [("band", "1410")]
    b3 = box.tool("tileB3_gate")
    assert [(a[0], TG.val(a, "--lock")) for a in b3] == [("census", "1410"), ("sag", "1410"), ("sag64", "1410")]
    assert all(TG.val(a, "--record").endswith(f"gate-b3{a[0]}-1410.json") for a in b3)
    assert "gate b3sag at 1410 MHz: RESULT: GATE b3sag PASS par A 48.0000 us tau 500 us (exit 0)" in led
    assert "gate b3sag64 at 1410 MHz: RESULT: GATE b3sag64 PASS par A 30.3650 us tau 300 us (exit 0)" in led
    assert not box.tool("launch_floor")
    assert "gate b3census: scripts/tileB3_gate.py census at the 1410 lock" in led
    lr = [a for a in box.tool("locked_r3") if f"{H}/jetmoe-8b-128g1.json" in a]
    assert len(lr) == 1 and TG.val(lr[0], "--locks") == "1410" and TG.val(lr[0], "--model") == "jetmoe-8b"
    for u in re.findall(r"^\S+ (?:bytes|timed) START: .*$", led, re.M):
        assert "1410" in u and "1710" not in u, u


@pytest.mark.parametrize("gate, last", [
    ("b3census", ["prelude", "calibrate", "tilecensus", "gate"]),
    ("b3sag", ["prelude", "calibrate", "tilecensus", "gate", "bytes", "gate", "gate"]),
    ("b3sag64", ["prelude", "calibrate", "tilecensus", "gate", "bytes", "gate", "gate", "bytes", "gate"])])
def test_a_b3_gate_fail_ends_the_session_before_any_timed_unit(tmp_path, gate, last):
    box, got = b3box(tmp_path, {"gate_fail": [gate]})
    led = box.ledger()
    assert got.returncode == exit_codes.INVALID, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == last
    assert f"GATE {gate} FAIL planted" in led and "STOPPED" in led
    assert not box.tool("locked_r3", dry=False) or not [a for a in box.tool("locked_r3") if "--histogram" in a]


def test_a_g_band_fail_ends_the_session_before_the_sag_gate(tmp_path):
    box, got = b3box(tmp_path, {"gate_fail": ["band"]})
    led = box.ledger()
    assert got.returncode == exit_codes.INVALID, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "gate", "bytes", "gate"]
    assert [a[0] for a in box.tool("tileB3_gate")] == ["census"]


@pytest.mark.parametrize("scenario", [{"enforced_w": 650.0}, {"lock_reads": "1395"}])
def test_a_power_or_lock_refusal_stops_before_any_page(tmp_path, scenario):
    box, got = b3box(tmp_path, scenario, MOE_DRIVER_PL_WAIT_S=4)
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert not box.tool("tileB3_gate") and not box.tool("dram_counter_route_tile")


def test_the_full_plan_runs_whole_at_launch_plus_186_after_25_min_of_boot(tmp_path):
    """The full plan on the fake box with the deadline the driver sees after 25 min of boot: 161 min left against 143
    estimated and the driver's 8 min reserve, so nothing is dropped and all twenty-three units run."""
    box, got = b3box(tmp_path, {}, plan=PLAN.read_text(), deadline=int(time.time()) + (186 - 25) * 60)
    led = box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert got.returncode == exit_codes.DONE and "DROPPED" not in led and len(starts) == 23, led[-3000:]
    assert [a[0] for a in box.tool("tileB3_gate")] == ["census", "sag", "sag64"]
    timed = [TG.val(a, "--model") for a in box.tool("locked_r3") if "--histogram" in a and "--dry-run" not in a]
    assert timed == ["phi-3.5-moe", "phi-3.5-moe", "phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b", "qwen1.5-moe-a2.7b",
                     "jetmoe-8b", "deepseek-v2-lite"], timed


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_tileB3_session.sh
# --------------------------------------------------------------------------

def test_start_with_b3_delivers_the_b3_driver_the_plan_and_the_pages_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 8
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
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
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
    lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", plan_rel,
            "--driver", DRIVER.name)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / DRIVER.name) == sha(DRIVER)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_tileB3_session.sh --dry-run" in cmds and "gh200_tileB3_session.sh --setup" in cmds
    assert "driver_script=gh200_tileB3_session.sh" in (lap.state / "r" / "run.env").read_text()
