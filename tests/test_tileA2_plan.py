"""Tile rental A2's plan and driver off the GPU (scripts/plans/rentalA2-tile-2026-10.plan,
scripts/gh200_tileA2_session.sh): the parse-only dry run, its units held to the registration (tileA2-gates
'plan': step, model, label, est, cap, nodrop or drop group, the totals, the plan's sha256), every page it
names registered, the driver against the pinned tile driver (every changed line marked A2), the A2 keys
and where they are refused, and the session on the fake box: the 900 W power limit refused when the card's
max is under it, the lock-hold probe's ladder (a failed rung moves every later unit to the next), a failed
gate ending the session at once, a timed unit's own lock, the census through tile_census_a2; then
`vm_run.sh start --driver gh200_tileA2_session.sh` against a stub VM, every delivered file by sha256."""
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

PLAN = REPO / "scripts" / "plans" / "rentalA2-tile-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_tileA2_session.sh"
TILE_DRIVER = TT.TILE_DRIVER
H = "docs/registered/2026-10-20-tileA2-hist"
GATES = json.loads((REPO / "docs" / "registered" / "2026-10-20-tileA2-gates-gh200.json").read_text())
TILE = "block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def _drop(u):
    if "[nodrop]" in u[5] or u[2] == "prelude":
        return "nodrop"
    m = re.search(r"\[drop-group ([\w-]+)\]", u[5])
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_a2_plan_parses_and_every_unit_is_the_registrations():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = GATES["plan"]
    assert len(units) == reg["units_count"] == 32 and "a PLAN of 32 units" in got.stdout
    for u, r in zip(units, reg["units"], strict=True):
        label = TT._label(u) or ""
        assert (u[2], u[1], label, int(u[3]), int(u[4]), _drop(u)) == (r["step"], r["model"], r["label"], r["est"], r["cap"], r["drop"]), u
    total = sum(int(u[3]) for u in units)
    assert total == reg["est_total_min"] == 168 and f"{total} min of units" in got.stdout
    assert sum(int(u[3]) for u in units if _drop(u) == "nodrop") == reg["est_nodrop_min"] == 142
    seen = []
    for u in reversed(units):
        g = _drop(u)
        if g != "nodrop" and g not in seen:
            seen.append(g)
    assert seen == reg["drop_order_last_first"] == ["dk", "ms", "w4", "rep"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    assert "the prelude also probes the floor lock(s) 1710 1410 1305 MHz" in got.stdout
    assert "168 of units estimated (142 nodrop)" in PLAN.read_text() and "launch + 220 min" in PLAN.read_text()
    assert "hard cap launch + 235 min" in PLAN.read_text()
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    assert "ladder=1710,1410,1305" in body and "- prelude lock-mhz=1710 power-limit-w=900" in body
    assert sorted(re.findall(r"label=(cv\w+) unless-lock=1710 ", body)) == ["cvm1", "cvm1t", "cvm8", "cvm8t", "cvo1", "cvo1t"]
    assert "lp1710" not in body and "drop-group=p1710" not in body


def test_the_gates_sit_after_the_units_they_read_and_the_probe_before_any_timed_or_counter_page():
    full = [(u[2], u[1], TT._label(u)) for u in TG._units(dry("--plan", str(PLAN)).stdout)]
    units = [(s, lab) for s, _m, lab in full]
    assert units[:6] == [("prelude", None), ("calibrate", "ta2"), ("gate", "gcal"), ("tilecensus", "ta2"), ("gate", "gcen"),
                         ("lockprobe", "lp")]
    i = units.index(("gate", "gband"))
    assert full[i - 1] == ("bytes", "mixtral-8x7b", "f128") and full[i - 2] == ("bytes", "olmoe-1b-7b", "f128")
    j = units.index(("gate", "g64"))
    assert full[j - 1] == ("bytes", "mixtral-8x7b", "f64w8") and full[j - 2] == ("bytes", "olmoe-1b-7b", "f64w8")
    first_measured = min(k for k, (s, _l) in enumerate(units) if s in ("bytes", "timed"))
    assert first_measured > units.index(("lockprobe", "lp"))
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    assert re.findall(r"gate=(\w+)", body) == ["calibrate", "census", "band", "f64w8"]


def test_every_page_the_plan_names_is_registered_at_its_sha256():
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    files = set(re.findall(r"histogram=(\S+)", body)) | set(re.findall(r":(docs/[^:]+\.json):", body))
    reg = {}
    for name in ("2026-10-20-tileA2-gates-gh200", "2026-10-20-tileA2-convl-gh200", "2026-10-20-tileA2-e1cal-gh200",
                 "2026-10-20-tileA2-bytes-gh200", "2026-10-20-tileA2-dead-gh200"):
        d = json.loads((REPO / "docs" / "registered" / f"{name}.json").read_text())
        blocks = [d.get("pages", {}).get("files", {}), (d.get("G-PROBE (the lock-hold probe; DESIGN-A2 6 risk 2, REVIEW-A2 1a, 1c, 6)") or {}).get("pages", {})]
        for b in blocks:
            reg.update({v["path"]: v["sha256"] for v in b.values()})
    assert len(files) == 11
    for f in files:
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f], f


@pytest.mark.parametrize("body, why", [
    ("- prelude\n- gate label=g gate=bogus\n", "gate bogus: calibrate, census, band or f64w8"),
    ("- prelude\n- gate gate=band\n", "a gate unit needs label= and gate="),
    ("- prelude\nolmoe-1b-7b bytes label=a gate=band\n", "gate belongs to a gate unit, not bytes"),
    ("- prelude\nolmoe-1b-7b bytes label=a lock-mhz=1410\n", "lock-mhz belongs to the prelude"),
    ("- prelude\n- tilecensus label=c tiles=128x128x64w8s3 models=olmoe-1b-7b lock-mhz=1410\n", "lock-mhz belongs to the prelude"),
    ("- prelude\n- lockprobe label=lp ladder=1410\n", "a lockprobe unit needs probes=..."),
    ("- prelude\n- lockprobe label=lp ladder=1410 probes=mixtral-8x7b:x.json:16:m\n", "a lockprobe unit needs block-m=..."),
    ("- prelude\n- lockprobe label=lp ladder=fast probes=mixtral-8x7b:x.json:16:m\n", "ladder fast: locks in MHz"),
    ("- prelude\n- lockprobe label=lp ladder=1410 probes=mixtral-8x7b:../x.json:16:m\n", "probes mixtral-8x7b:../x.json:16:m"),
    ("- prelude power-limit-w=800\n", "power-limit-w 800: 700 or 900"),
    ("- prelude\nolmoe-1b-7b timed label=a power-limit-w=900\n", "power-limit-w belongs to the prelude"),
    ("- prelude\nolmoe-1b-7b timed label=a ladder=1410\n", "ladder belongs to a lockprobe unit"),
    ("- prelude\nolmoe-1b-7b bytes label=a unless-lock=1710\n", "unless-lock belongs to a timed unit"),
    ("- prelude\nolmoe-1b-7b timed label=a unless-lock=fast\n", "unless-lock fast: a clock in MHz"),
])
def test_the_a2_keys_are_refused_where_they_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_tile_driver_refuses_the_a2_plan():
    got = dry("--plan", str(PLAN), driver=TILE_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "no key 'lock-mhz'" in got.stderr


def test_the_a2_driver_is_the_tile_driver_plus_lines_marked_a2():
    a = TILE_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "A2" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and len(b) - len(a) < 170
    text = DRIVER.read_text()
    assert "POWER_LIMIT_W=\"${MOE_DRIVER_POWER_LIMIT_W:-900}\"" in text and "\nLOCK_TIMED=1710\n" in text
    assert "SKIPPED BY RULE" in text and "REFUSED (pre-setup, tileA2-gates P-W)" in text
    assert "power.max_limit" in text and "no fallback to 700 W" in text and "scripts/tile_census_a2.py" in text


def test_the_lambda_stop_instruction_and_vm_run_name_the_a2_driver():
    text = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_tileA2_session\.sh)'", text).group(1)
    assert re.search(pat, "bash gh200_tileA2_session.sh --plan x") and not re.search(pat, "gh200_tile_session.sh")
    assert not re.search(r"[g]h200_(model|tile)_session.sh", "gh200_tileA2_session.sh")
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:4] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh", "gh200_tileA2_session.sh"]


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

SMI_A2 = TG.SMI.replace('''        elif fld == "clocks.max.sm":''', '''        elif fld == "power.max_limit":
            vals.append(f"{float(SC.get('pl_max', 900.0)):.2f}" + (" W" if units else ""))
        elif fld == "clocks.max.sm":''')
FAKE_A2 = r'''
if tool == "tileA2_gate":
    key = args[0] + (":" + str(int(float(val("--lock")))) if args[0] == "probe" else "")
    bad = key in SC.get("gate_fail", [])
    print(f"RESULT: GATE {args[0]} {'FAIL planted' if bad else 'PASS'}")
    sys.exit(3 if bad else 0)
if tool == "tile_census_a2":
    if "--dry-run" in args:
        print("TILE CENSUS (A2): compile only (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tiles": val("--tiles"), "models": val("--models"), "entries": [], "path_mismatch": []}))
    print("RESULT: VALIDITY census_path PASS 0 configs, 0 path mismatches, 0 errors")
    sys.exit(0)
'''


def fakepy_a2() -> str:
    body = TT.fakepy_tile()
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_A2 + "sys.exit(0)" + tail


PLAN_BOX = f"""- prelude lock-mhz=1710 power-limit-w=900 est=1 cap=5
mixtral-8x7b calibrate label=ta2 nodrop=1 est=1 cap=5
- gate label=gcal gate=calibrate nodrop=1 est=1 cap=5
- tilecensus label=ta2 tiles=128x128x64w8s3 models=olmoe-1b-7b nodrop=1 est=1 cap=5
- lockprobe label=lp ladder=1710,1410,1305 probes=mixtral-8x7b:{H}/mixtral-8x7b-128g16-probe.json:16:m,qwen2-57b-a14b:{H}/qwen2-57b-a14b-128g1-probe.json:1:q {TILE} nodrop=1 est=2 cap=10
olmoe-1b-7b bytes label=f128 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gband gate=band nodrop=1 est=1 cap=5
mixtral-8x7b timed label=cvm1t lock-mhz=1710 unless-lock=1710 histogram={H}/convl-mixtral-8x7b-g1.json arms=native declared-copies=9 timed-groups=1 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=1 cap=5
"""


def a2box(tmp_path, scenario):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_a2())
    TG._exe(box.bin / "nvidia-smi", SMI_A2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "a2.plan"
    f.write_text(PLAN_BOX)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    return a2box(tmp_path_factory.mktemp("ladder"), {"gate_fail": ["probe:1710"]})


def test_the_prelude_sets_900_w_and_a_failed_1710_rung_moves_every_later_unit_to_1410(ladder):
    box, got = ladder
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    pl = [a for a in box.smi() if "-pl" in a]
    assert pl == [["-pl", "900", "-sc", "0"]]
    assert "power.max_limit 900.00 W, at or over the registered 900 W" in led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "gate", "tilecensus", "lockprobe", "bytes", "gate", "timed"]
    assert "lockprobe: rung 1710 MHz: RESULT: GATE probe FAIL planted (exit 3)" in led
    assert "lockprobe: rung 1410 MHz HOLDS: the session's lock is 1410 MHz" in led
    assert (box.session / "gh200-driver" / "a2-lock.env").read_text() == "A2_LOCK=1410\n"
    lr = box.tool("locked_r3")
    probe = [x for x in lr if f"{H}/mixtral-8x7b-128g16-probe.json" in x]
    assert [TG.val(x, "--locks") for x in probe] == ["1710", "1410"]
    # the early stop: the failing 1710 rung runs no Qwen2-57B page and no counter page
    q = [x for x in lr if f"{H}/qwen2-57b-a14b-128g1-probe.json" in x]
    assert [TG.val(x, "--locks") for x in q] == ["1410"] and all(TG.val(x, "--declared-copies") == "9" for x in q)
    assert "lp1710m fails the rule, the rung's remaining pages are not run (early stop)" in led
    dcr = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    outs = [TG.val(a, "--out") for a in dcr]
    assert len(dcr) == 2 and all(TG.val(a, "--page-lock-mhz") == "1410" for a in dcr), outs
    assert "-olmoe-1b-7b-lp1410f-r3-counters/lock1410/r3c-g64.json" in outs[0] and "-olmoe-1b-7b-f128-r3-counters/lock1410/" in outs[1]
    assert TG.val(dcr[0], "--group-m") == "64" and TG.val(dcr[0], "--block-m") == "128" and TG.val(dcr[0], "--num-warps") == "8"
    partial = [a for a in box.tool("tileA2_gate") if "--partial" in a]
    assert [TG.val(a, "--lock") for a in partial] == ["1710", "1410", "1410", "1410"]
    gates = {(a[0], TG.val(a, "--lock")) for a in box.tool("tileA2_gate") if "--partial" not in a}
    assert gates == {("calibrate", "1710"), ("probe", "1710"), ("probe", "1410"), ("band", "1410")}
    cv = [x for x in lr if f"{H}/convl-mixtral-8x7b-g1.json" in x]
    assert len(cv) == 1 and TG.val(cv[0], "--locks") == "1710" and TG.val(cv[0], "--arms") == "native"
    assert len(box.tool("tile_census_a2", dry=False)) == 1 and not box.tool("tile_census")
    assert [TG.val(a, "--lock-mhz") for a in box.tool("hw_rulers", dry=False)] == ["1710", "1410"]
    probes = {ln.split("=")[0] for ln in (box.session / "gh200-driver" / "locks.env").read_text().split() if "PROBE" in ln}
    assert {"LOCK_PROBE_1410", "LOCK_PROBE_1305", "LOCK_PROBE_1710"} <= probes


def test_when_1710_holds_no_rung_below_runs_and_conv_l_is_skipped_by_rule(tmp_path):
    box, got = a2box(tmp_path, {})
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + led
    assert (box.session / "gh200-driver" / "a2-lock.env").read_text() == "A2_LOCK=1710\n"
    assert {TG.val(a, "--lock") for a in box.tool("tileA2_gate") if a[0] == "probe"} == {"1710"}
    assert "DROPPED" not in led   # the ruled-out CONV-L unit is not budgeted
    assert not [x for x in box.tool("locked_r3") if f"{H}/convl-mixtral-8x7b-g1.json" in x]
    assert "timed SKIPPED BY RULE: the session's lock is 1710 MHz" in led
    dcr = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    assert TG.val(dcr[0], "--page-lock-mhz") == "1710"


def test_the_pre_setup_dry_run_refuses_a_card_under_900_w_before_any_setup(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    TG._exe(fake / "nvidia-smi", "#!/bin/bash\necho 700.00\n")
    env = {**os.environ, "HOME": "/nonexistent", "PATH": f"{fake}:{os.environ['PATH']}"}
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN)], capture_output=True, text=True,
                         timeout=60, env=env)
    assert got.returncode == exit_codes.REFUSED and "REFUSED (pre-setup, tileA2-gates P-W)" in got.stdout
    assert "THE GH200 MODEL-TEST SESSION" not in got.stdout
    TG._exe(fake / "nvidia-smi", "#!/bin/bash\necho 'Failed to initialize NVML: Driver/library version mismatch'\nexit 18\n")
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN)], capture_output=True, text=True,
                         timeout=60, env=env)
    assert got.returncode == exit_codes.REFUSED and got.stdout.startswith("THE GH200 MODEL-TEST SESSION")
    assert "the prelude's P-W check decides after setup" in got.stderr
    TG._exe(fake / "nvidia-smi", "#!/bin/bash\necho 900.00\n")
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN)], capture_output=True, text=True,
                         timeout=60, env=env)
    assert got.returncode == exit_codes.REFUSED and got.stdout.startswith("THE GH200 MODEL-TEST SESSION")


@pytest.mark.parametrize("scenario, last, rc", [
    ({"gate_fail": ["band"]}, "gate", exit_codes.INVALID),
    ({"gate_fail": ["probe:1710", "probe:1410", "probe:1305"]}, "lockprobe", exit_codes.INVALID),
    ({"gate_fail": ["calibrate"]}, "gate", exit_codes.INVALID),
    ({"pl_max": 700.0}, "prelude", exit_codes.REFUSED),
])
def test_a_failed_gate_or_a_card_under_900_w_ends_the_session_at_once(tmp_path, scenario, last, rc):
    box, got = a2box(tmp_path, scenario)
    led = box.ledger()
    assert got.returncode == rc, got.stdout[-2000:] + led
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert starts[-1] == last and "timed" not in starts, starts
    assert "STOPPED" in led
    if "pl_max" in scenario:
        assert "power.max_limit reads '700.00' W, under the registered 900 W" in led
        assert not [a for a in box.smi() if "-pl" in a]


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_tileA2_session.sh
# --------------------------------------------------------------------------

def test_start_with_rental_a2_delivers_the_a2_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 9
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

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / DRIVER.name) == sha(DRIVER)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_tileA2_session.sh --dry-run" in cmds and "gh200_tileA2_session.sh --setup" in cmds
    assert "driver_script=gh200_tileA2_session.sh" in (lap.state / "r" / "run.env").read_text()


def test_the_pre_setup_dry_run_prints_rental_a2_from_the_files_vm_run_copies(tmp_path):
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
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 32 units" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
