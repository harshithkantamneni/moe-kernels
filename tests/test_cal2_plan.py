"""The calibration rental C2's plan and driver off the GPU (scripts/plans/rentalC2-cal-2026-10.plan,
scripts/gh200_cal2_session.sh): the parse-only dry run, its units held to the registration
(docs/registered/2026-10-22-cal2-gh200 'plan': step, model, label, est, cap, nodrop or drop group, the totals, the
plan's sha256), every page it names registered, the driver against S1's pinned driver (every changed line marked C2),
the 700 W read-back with no fallback (the refusal stops the session and the ledger names the file it wrote), the
session lock's probe as a stop, G-BAND after the f128 pages, the ledger's power limit, and the session on the fake
box; then `vm_run.sh start --driver gh200_cal2_session.sh` against a stub VM, every delivered file by sha256."""
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
import test_settleS1_plan as TS1  # noqa: E402
import test_tile_plan as TT  # noqa: E402
import test_tileA2_plan as TA2  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalC2-cal-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_cal2_session.sh"
S1_DRIVER = TS1.DRIVER
H = "docs/registered/2026-10-22-cal2-hist"
HA = "docs/registered/2026-10-14-tileA-hist"
REG = json.loads((REPO / "docs" / "registered" / "2026-10-22-cal2-gh200.json").read_text())
S1_REG = json.loads((REPO / "docs" / "registered" / "2026-10-21-settleS1-gh200.json").read_text())
TILE = "block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_c2_plan_parses_and_every_unit_is_the_registrations():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = REG["plan"]
    assert len(units) == reg["units_count"] == 10 and "a PLAN of 10 units" in got.stdout
    for u, r in zip(units, reg["units"], strict=True):
        label = TT._label(u) or ""
        assert (u[2], u[1], label, int(u[3]), int(u[4]), TA2._drop(u)) == (r["step"], r["model"], r["label"], r["est"], r["cap"], r["drop"]), u
    total = sum(int(u[3]) for u in units)
    assert total == reg["est_total_min"] == 48 and f"{total} min of units" in got.stdout
    nodrop = sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop")
    assert nodrop == reg["est_nodrop_min"] == 37 and reg["drop_order_last_first"] == ["m7"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    text = PLAN.read_text()
    assert "48 of units estimated (37 nodrop)" in text
    assert 90 - 25 - 8 >= total   # every unit, m7 included, fits the deadline less boot and the driver's reserve
    assert "--deadline <launch + 90 min" in text and "hard\n# cap launch + 105 min ($4.01 at most)" in text
    assert reg["wall_min"] == 73 and reg["cost_usd"] == 2.79 and reg["deadline_after_launch_min"] == 90
    assert reg["hard_cap_after_launch_min"] == 105 and reg["cost_at_hard_cap_usd"] == 4.01
    assert "about 73 min ($2.79 at $2.29/h)" in text
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    assert "- prelude lock-mhz=1410 power-limit-w=700" in body and "power-limit-w=900" not in body
    assert "lockprobe" not in body and "powerproof" not in body and "replay=" not in body and "stall-metrics" not in body
    assert sorted(re.findall(r"lock-mhz=(\d+)", body)) == ["1410"]
    # the ledger names the plan's limit, not the driver's default 900 W
    assert "3c.1 supported clocks, persistence, 700 W (power.max_limit read first), the session lock 1410 MHz" in got.stdout


def test_the_units_are_the_registered_cells_in_the_registered_order():
    u = REG["plan"]["units"]
    assert [(x["model"], x["step"], x["label"]) for x in u] == [
        ("-", "prelude", ""), ("mixtral-8x7b", "calibrate", "tc2"), ("-", "tilecensus", "tc2"),
        ("olmoe-1b-7b", "bytes", "f128"), ("mixtral-8x7b", "bytes", "f128"), ("-", "gate", "gband"),
        ("mixtral-8x22b", "timed", "t128g1"), ("mixtral-8x22b", "bytes", "b128g1"), ("mixtral-8x7b", "timed", "t128g1m"),
        ("mixtral-8x7b", "bytes", "b128g1")]
    assert [x["drop"] for x in u] == ["nodrop"] * 8 + ["m7", "m7"]
    k = {(x["model"], x["label"]): x["keys"] for x in u}
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):
        assert k[(m, "f128")]["byte-groups"] == "64" and "histogram" not in k[(m, "f128")]
    assert k[("-", "gband")]["gate"] == "band"
    t, b, m7 = k[("mixtral-8x22b", "t128g1")], k[("mixtral-8x22b", "b128g1")], k[("mixtral-8x7b", "b128g1")]
    mt = k[("mixtral-8x7b", "t128g1m")]
    assert {a: v for a, v in mt.items() if a != "histogram"} == {a: v for a, v in t.items() if a != "histogram"}
    assert mt["histogram"] == m7["histogram"]
    assert t["histogram"] == b["histogram"] == f"{H}/mixtral-8x22b-128g1.json"
    assert m7["histogram"] == f"{HA}/mixtral-8x7b-128g1.json"
    assert t["arms"] == b["arms"] == m7["arms"] == "native,shared"
    assert (t["declared-copies"], t["timed-groups"], t["slip-policy"], t["g1-rule"]) == ("9", "1", "cell", "r7-median-run3")
    assert b["byte-groups"] == m7["byte-groups"] == "1"
    for x in (t, b, m7, k[("olmoe-1b-7b", "f128")]):
        assert (x["block-m"], x["block-n"], x["block-k"], x["num-warps"], x["num-stages"]) == ("128", "128", "64", "8", "3")
    assert k[("-", "tc2")] == {"tiles": "128x128x64w8s3", "models": "mixtral-8x22b"}
    # every registered unit block names plan units
    names = {f"{x['model']} {x['step']} {x['label']}".strip() for x in u} | {"- prelude"}
    for blk in REG["units"].values():
        for nm in blk["units"]:
            assert nm in names, nm
    # the a2 E1-CAL Mixtral 128 G 1 file is the one A2's timed page ran
    a2 = json.loads((REPO / "docs/registered/2026-10-20-tileA2-e1cal-gh200.json").read_text())["pages"]["files"]
    assert REG["pages"]["files"]["mixtral-8x7b-128g1.json"] == {**a2["mixtral-8x7b-128g1.json"], "model": "mixtral-8x7b"}


def test_every_page_the_plan_names_is_registered_at_its_sha256_and_parses_under_r3():
    import private_weight_reference as R3
    from moe.spec import MODEL_CONFIGS
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    files = set(re.findall(r"histogram=(\S+)", body))
    reg = {v["path"]: v["sha256"] for v in REG["pages"]["files"].values()}
    assert files == set(reg) and len(files) == 2
    for f in files:
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f], f
        doc = json.loads((REPO / f).read_text())
        page = R3.load_histogram_page(REPO / f, MODEL_CONFIGS[doc["model"]], int(doc["block_m"]), doc["model"])
        assert all(c.arms == ("native", "shared") for c in page.cells), f
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/cal2/hist_c2.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and "same" in got.stdout, got.stdout + got.stderr


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=a byte-treads=0\n", "byte-treads 0: treads 1..9"),
    ("- prelude\nolmoe-1b-7b bytes label=a replay=kernel\n", "replay kernel: application"),
    ("- prelude\n- tilecensus label=c tiles=128x128x64w8s3 models=mixtral-8x22b lock-mhz=1410\n", "lock-mhz belongs to the prelude"),
])
def test_s1s_key_refusals_hold_in_the_c2_driver(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_c2_driver_is_the_s1_driver_plus_lines_marked_c2():
    a = S1_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "C2" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and len(b) - len(a) < 60
    removed = [ln for tag, i1, i2, j1, j2 in hunks if tag == "replace" for ln in a[i1:i2]]
    assert any("S1_FALLBACK_W" in ln for ln in removed)   # the fallback is not taken
    assert '      printf \'%s\\n\' "$q" > "$S/power-set-unsettled.txt"   # S1' in removed
    text = DRIVER.read_text()
    assert "s1-power.env" not in "\n".join(ln for ln in b[header:] if "C2" in ln)
    assert 'c2-power.env' in text and "lock_and_check \"$LOCK_TIMED\"" in text
    # the S1 driver is S1's pinned bytes
    assert hashlib.sha256(S1_DRIVER.read_bytes()).hexdigest() == S1_REG["code_pins"]["scripts/gh200_settleS1_session.sh"]
    assert REG["code_pins"]["scripts/gh200_settleS1_session.sh"] == S1_REG["code_pins"]["scripts/gh200_settleS1_session.sh"]


def test_the_ledger_names_the_unsettled_file_the_prelude_writes():
    """S1's open defect: its log named power-set-unsettled.txt for the file power-set-unsettled-900.txt."""
    b = DRIVER.read_text()
    writes = re.findall(r'> "\$S/(power-set-unsettled[^"]*)"', b)
    named = re.findall(r"\((power-set-unsettled[^)]*\.txt)\)", b)
    assert writes == ["power-set-unsettled-${POWER_LIMIT_W}.txt"]
    assert set(named) == {"power-set-unsettled-${POWER_LIMIT_W}.txt"} and "mv \"$S/power-set-unsettled" not in b


def test_the_s1_driver_refuses_nothing_the_c2_plan_needs_and_the_a2_driver_refuses_it():
    got = dry("--plan", str(PLAN), driver=S1_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 10 units" in got.stdout   # S1 parses it (no new keys)
    assert "900 W (power.max_limit read first)" in got.stdout   # but S1's dry run names its default limit
    got = dry("--plan", str(PLAN), driver=REPO / "scripts" / "gh200_tile_session.sh")
    assert got.returncode == exit_codes.REFUSED


def test_vm_run_names_the_c2_driver_and_the_s1_test_still_holds():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:6] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh",
                           "gh200_tileA2_session.sh", "gh200_settleS1_session.sh", "gh200_cal2_session.sh"]
    assert "|gh200_settleS1_session.sh|gh200_cal2_session.sh" in vm.splitlines()[5]
    assert not re.search(r"[g]h200_settleS1_session.sh", "gh200_cal2_session.sh")
    assert not re.search(r"[g]h200_(model|tile)_session.sh", "gh200_cal2_session.sh")
    lam = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_cal2_session\.sh)'", lam).group(1)
    assert re.search(pat, "bash gh200_cal2_session.sh --plan x") and not re.search(pat, "gh200_settleS1_session.sh")


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

#: S1's fake nvidia-smi, plus a lock that does not read back: clocks.sm reads `lock_reads` under any -lgc
SMI_C2 = TS1.SMI_S1.replace('''            vals.append(str(st["locked"] or "345").split(",")[0] + (" MHz" if units else ""))''',
                            '''            vals.append(str((SC.get("lock_reads") if st["locked"] else None) or st["locked"] or "345").split(",")[0] + (" MHz" if units else ""))''')
assert "lock_reads" in SMI_C2

PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=700 est=1 cap=5
mixtral-8x7b calibrate label=tc2 nodrop=1 est=1 cap=5
- tilecensus label=tc2 tiles=128x128x64w8s3 models=mixtral-8x22b nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f128 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gband gate=band nodrop=1 est=1 cap=5
mixtral-8x22b timed label=t128g1 {TILE} histogram={H}/mixtral-8x22b-128g1.json arms=native,shared declared-copies=9 timed-groups=1 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=1 cap=5
mixtral-8x22b bytes label=b128g1 {TILE} histogram={H}/mixtral-8x22b-128g1.json arms=native,shared byte-groups=1 nodrop=1 est=1 cap=5
"""


def c2box(tmp_path, scenario, plan=PLAN_BOX, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", TS1.fakepy_s1())
    TG._exe(box.bin / "nvidia-smi", SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "c2.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return c2box(tmp_path_factory.mktemp("c2clean"), {})


def test_every_unit_runs_at_700_w_and_1410_mhz_with_its_registered_keys(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "bytes", "gate", "timed", "bytes"]
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "700", "-sc", "0"]]
    assert "prelude START: 3c.1 supported clocks, persistence, 700 W" in led
    assert ("the power limit read back after 0 s: enforced.power.limit 700.00 W, -q -d POWER GPU scope Current 700.00 W "
            "(Requested 700.00 W), power.limit 700.00 W") in led
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_W=700\n"
    assert not (box.session / "gh200-driver" / "s1-power.env").exists()
    assert "Current Power Limit               : 700.00 W" in (box.session / "power-set.txt").read_text()
    assert "lock probe: the session lock 1410 MHz held (C2)" in led
    assert "LOCK_PROBE_1410=held" in (box.session / "gh200-driver" / "locks.env").read_text()
    smi = box.smi()
    i = next(k for k, a in enumerate(smi) if a[:1] == ["-lgc"] and a[1] == "1410,1410")
    assert ["-rgc"] in smi[i:]   # probed, then reset
    gates = [a for a in box.tool("tileA2_gate")]
    assert [(a[0], TG.val(a, "--lock")) for a in gates] == [("band", "1410")]
    lr = [a for a in box.tool("locked_r3") if f"{H}/mixtral-8x22b-128g1.json" in a]
    assert len(lr) == 1 and TG.val(lr[0], "--locks") == "1410" and TG.val(lr[0], "--declared-copies") == "9"
    assert TG.val(lr[0], "--model") == "mixtral-8x22b"
    dcr = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    assert [TG.val(a, "--page-lock-mhz") for a in dcr] == ["1410", "1410"]
    assert TG.val(dcr[1], "--histogram") == f"{H}/mixtral-8x22b-128g1.json" and TG.val(dcr[1], "--arms") == "native,shared"
    for u in re.findall(r"^\S+ (?:bytes|timed) START: .*$", led, re.M):
        assert "1410" in u and "1710" not in u, u


@pytest.mark.parametrize("scenario", [{"enforced_w": 650.0}, {"q_stuck_w": 900.0}, {"enforced_raw": "[N/A]"}])
def test_a_700_w_read_back_refusal_stops_the_session_and_names_its_file(tmp_path, scenario):
    box, got = c2box(tmp_path, scenario, MOE_DRIVER_PL_WAIT_S=4)
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "700", "-sc", "0"]]   # no fallback, no other scope
    assert "not both 700 W (power-set-unsettled-700.txt)" in led
    assert ("prelude REFUSED: the power limit never read back as 700 W on both readings within 4 s (C2: no fallback "
            "limit, the session stops; power-set-unsettled-700.txt)") in led
    assert (box.session / "power-set-unsettled-700.txt").is_file() and not (box.session / "power-set-unsettled.txt").exists()
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_700=refused\n"
    assert not (box.session / "power-set.txt").exists()
    assert not box.tool("dram_counter_route_tile") and not box.tool("locked_r3")


def test_a_session_lock_that_does_not_read_back_stops_before_any_page(tmp_path):
    box, got = c2box(tmp_path, {"lock_reads": "1395"})
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert "lock probe: the session lock 1410 MHz refused (C2)" in led
    assert "prelude REFUSED: the session lock 1410 MHz did not read back after -lgc" in led
    assert "LOCK_PROBE_1410=refused" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert not box.tool("calibrate_hardware") and not box.tool("dram_counter_route_tile")


def test_a_g_band_fail_ends_the_session_before_the_8x22b_pages(tmp_path):
    box, got = c2box(tmp_path, {"gate_fail": ["band"]})
    led = box.ledger()
    assert got.returncode == exit_codes.INVALID, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "bytes", "gate"]
    assert not [a for a in box.tool("locked_r3") if f"{H}/mixtral-8x22b-128g1.json" in a]


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_cal2_session.sh
# --------------------------------------------------------------------------

def test_start_with_c2_delivers_the_c2_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 2
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
    assert "gh200_cal2_session.sh --dry-run" in cmds and "gh200_cal2_session.sh --setup" in cmds
    assert "driver_script=gh200_cal2_session.sh" in (lap.state / "r" / "run.env").read_text()


@pytest.mark.parametrize("minutes, dropped", [(90 - 25, False), (80 - 25, True)])
def test_m7_is_not_dropped_at_the_start_at_launch_plus_90_after_25_min_of_boot(tmp_path, minutes, dropped):
    """The full plan on the fake box with the deadline the driver sees after 25 min of boot: at launch + 90 nothing is
    dropped (57 min left against 48 estimated); at the old launch + 80 the driver drops m7 at once."""
    import time
    box = TG.Box(tmp_path, {"retracted": 40})
    TG._exe(box.bin / "fakepy", TS1.fakepy_s1())
    TG._exe(box.bin / "nvidia-smi", SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(int(time.time()) + minutes * 60), "--plan", str(PLAN), "--no-push"],
                         env=box.env(), capture_output=True, text=True, timeout=600)
    led = box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    if dropped:
        assert "DROPPED" in led and len(starts) == 8, led
    else:
        assert got.returncode == exit_codes.DONE and "DROPPED" not in led and len(starts) == 10, led[-3000:]
        lr = [a for a in box.tool("locked_r3") if f"{HA}/mixtral-8x7b-128g1.json" in a]
        assert len(lr) == 1 and TG.val(lr[0], "--model") == "mixtral-8x7b" and TG.val(lr[0], "--locks") == "1410"
