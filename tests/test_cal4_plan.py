"""The calibration rental C4's plan and driver off the GPU (scripts/plans/rentalC4-cal-2026-10.plan,
scripts/gh200_cal4_session.sh): the parse-only dry run, its units held to the registration
(docs/registered/2026-10-27-cal4-gh200 'plan': step, model, label, est, cap, nodrop or drop group, the totals, wall,
cost, deadline and hard cap, the plan's sha256), the tp2 page it names registered and parsed by R3, the driver against
C2's pinned driver (no line past the header changed, C2's header kept whole), C2's driver parsing the C4 plan,
vm_run.sh and LAMBDA.md naming the driver; then the session on the fake box (C2's power and lock stops, G-BAND after
the f128 pages, the census on tp2, tp2's timed and byte pages at 1410 on the registered file) and
`vm_run.sh start --driver gh200_cal4_session.sh` against a stub VM, every file by sha256."""
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
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_cal2_plan as TC2  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_settleS1_plan as TS1  # noqa: E402
import test_tile_plan as TT  # noqa: E402
import test_tileA2_plan as TA2  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalC4-cal-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_cal4_session.sh"
C2_DRIVER = TC2.DRIVER
H4 = "docs/registered/2026-10-27-cal4-hist"
PAGE = f"{H4}/mixtral-8x7b-tp2-128g1.json"
REG = json.loads((REPO / "docs" / "registered" / "2026-10-27-cal4-gh200.json").read_text())
C2_REG = TC2.REG
TILE = TC2.TILE
MODEL = "mixtral-8x7b-tp2"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def comments(text: str) -> str:
    """The plan's comment lines joined into one string (wrapping removed)."""
    return " ".join(ln[1:].strip() for ln in text.splitlines() if ln.startswith("#"))


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_c4_plan_parses_and_every_unit_is_the_registrations():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = REG["plan"]
    assert len(units) == reg["units_count"] == 8 and "a PLAN of 8 units" in got.stdout
    for u, r in zip(units, reg["units"], strict=True):
        label = TT._label(u) or ""
        assert (u[2], u[1], label, int(u[3]), int(u[4]), TA2._drop(u)) == (r["step"], r["model"], r["label"], r["est"],
                                                                             r["cap"], r["drop"]), u
    total = sum(int(u[3]) for u in units)
    assert total == reg["est_total_min"] == 25 and f"{total} min of units" in got.stdout
    nodrop = sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop")
    assert nodrop == reg["est_nodrop_min"] == 25 and reg["drop_order_last_first"] == []
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    text = PLAN.read_text()
    c = comments(text)
    assert "25 of units estimated (25 nodrop)" in c
    assert 65 - 25 - 8 >= total   # every unit fits the deadline less boot and the driver's reserve
    assert "--deadline <launch + 65 min, epoch s>" in c and "guardian hard cap launch + 80 min ($3.05 at most)" in c
    assert reg["wall_min"] == 50 and reg["cost_usd"] == 1.91 and reg["deadline_after_launch_min"] == 65
    assert reg["hard_cap_after_launch_min"] == 80 and reg["cost_at_hard_cap_usd"] == 3.05 and reg["rate_usd_per_h"] == 2.29
    assert "about 50 min ($1.91 at $2.29/h)" in c
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    assert "- prelude lock-mhz=1410 power-limit-w=700" in body and "power-limit-w=900" not in body
    assert sorted(re.findall(r"lock-mhz=(\d+)", body)) == ["1410"]
    for absent in ("lockprobe", "powerproof", "replay=", "stall-metrics", "stamps", "regcheck", "perturb", "drop-group"):
        assert absent not in body, absent
    assert "3c.1 supported clocks, persistence, 700 W (power.max_limit read first), the session lock 1410 MHz" in got.stdout


def test_the_units_are_the_registered_cells_in_the_registered_order():
    u = REG["plan"]["units"]
    assert [(x["model"], x["step"], x["label"]) for x in u] == [
        ("-", "prelude", ""), ("mixtral-8x7b", "calibrate", "tc4"), ("-", "tilecensus", "tc4"),
        ("olmoe-1b-7b", "bytes", "f128"), ("mixtral-8x7b", "bytes", "f128"), ("-", "gate", "gband"),
        (MODEL, "timed", "t128g1"), (MODEL, "bytes", "b128g1")]
    assert [x["drop"] for x in u] == ["nodrop"] * 8
    k = {(x["model"], x["label"]): x["keys"] for x in u}
    c2 = {(x["model"], x["label"]): x["keys"] for x in C2_REG["plan"]["units"]}
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):   # C2's f128 pages, unchanged
        assert k[(m, "f128")] == c2[(m, "f128")] and k[(m, "f128")]["byte-groups"] == "64"
    assert k[("-", "gband")] == c2[("-", "gband")] == {"gate": "band"}
    assert k[("-", "tc4")] == {"tiles": "128x128x64w8s3", "models": MODEL}
    t, b = k[(MODEL, "t128g1")], k[(MODEL, "b128g1")]
    t2, b2 = c2[("mixtral-8x22b", "t128g1")], c2[("mixtral-8x22b", "b128g1")]
    # C2's 8x22B timed and byte keys, the histogram file the only change
    assert {a: v for a, v in t.items() if a != "histogram"} == {a: v for a, v in t2.items() if a != "histogram"}
    assert {a: v for a, v in b.items() if a != "histogram"} == {a: v for a, v in b2.items() if a != "histogram"}
    assert t["histogram"] == b["histogram"] == PAGE == REG["pages"]["files"]["mixtral-8x7b-tp2-128g1.json"]["path"]
    assert t["arms"] == b["arms"] == "native,shared" and t["declared-copies"] == "9" and b["byte-groups"] == "1"
    names = {f"{x['model']} {x['step']} {x['label']}".strip() for x in u} | {"- prelude"}
    for blk in REG["units"].values():
        for nm in blk["units"]:
            assert nm in names, nm


def test_the_page_the_plan_names_is_registered_at_its_sha256_and_parses_under_r3():
    import private_weight_reference as R3
    from moe.spec import MODEL_CONFIGS
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    files = set(re.findall(r"histogram=(\S+)", body))
    reg = {v["path"]: v["sha256"] for v in REG["pages"]["files"].values()}
    assert files == set(reg) == {PAGE}
    assert hashlib.sha256((REPO / PAGE).read_bytes()).hexdigest() == reg[PAGE]
    doc = json.loads((REPO / PAGE).read_text())
    page = R3.load_histogram_page(REPO / PAGE, MODEL_CONFIGS[MODEL], 128, MODEL)
    assert all(c.arms == ("native", "shared") for c in page.cells)
    assert [[c.label, c.n] for c in page.ordered()] == [[c["label"], c["n"]] for c in doc["cells"]]
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/cal4/hist_c4.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and "same" in got.stdout, got.stdout + got.stderr


def test_the_c4_driver_is_the_c2_driver_with_its_own_header_only():
    a = C2_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:   # every hunk is in the header; a body hunk would have to be marked C4
        assert tag == "insert" and j2 <= header, (tag, a[i1:i2], b[j1:j2])
    assert b[header:] == a[a.index("set -uo pipefail"):]
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and "C4'S DRIVER" in b[1]
    assert len(b) - len(a) < 60
    assert "# THE C2 DRIVER'S OWN HEADER, UNCHANGED:" in b[:header]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE C2 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    assert hashlib.sha256(C2_DRIVER.read_bytes()).hexdigest() == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert REG["code_pins"]["scripts/gh200_cal2_session.sh"] == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert (DRIVER.stat().st_mode & 0o777) == (C2_DRIVER.stat().st_mode & 0o777)
    assert REG["code_pins"]["scripts/gh200_cal4_session.sh"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()


def test_c2s_driver_parses_the_c4_plan_and_the_c4_driver_parses_c2s():
    got = dry("--plan", str(PLAN), driver=C2_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 8 units" in got.stdout, got.stderr
    got = dry("--plan", str(TC2.PLAN))
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 10 units" in got.stdout


def test_vm_run_and_lambda_name_the_c4_driver_and_nothing_else_moved():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh",
                       "gh200_tileA2_session.sh", "gh200_settleS1_session.sh", "gh200_cal2_session.sh",
                       "gh200_cal3_session.sh", "gh200_cal4_session.sh"]
    assert vm.splitlines()[5].endswith("|gh200_cal2_session.sh|gh200_cal3_session.sh|gh200_cal4_session.sh]")
    for other in (r"[g]h200_cal3_session.sh", r"[g]h200_cal2_session.sh", r"[g]h200_settleS1_session.sh",
                  r"[g]h200_(model|tile)_session.sh"):
        assert not re.search(other, "gh200_cal4_session.sh")
    lam = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_cal4_session\.sh)'", lam).group(1)
    assert re.search(pat, "bash gh200_cal4_session.sh --plan x") and not re.search(pat, "gh200_cal3_session.sh")
    assert "and the calibration rental C4's copy (2026-10-27) with" in lam


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=700 est=1 cap=5
mixtral-8x7b calibrate label=tc4 nodrop=1 est=1 cap=5
- tilecensus label=tc4 tiles=128x128x64w8s3 models={MODEL} nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f128 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gband gate=band nodrop=1 est=1 cap=5
{MODEL} timed label=t128g1 {TILE} histogram={PAGE} arms=native,shared declared-copies=9 timed-groups=1 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=1 cap=5
{MODEL} bytes label=b128g1 {TILE} histogram={PAGE} arms=native,shared byte-groups=1 nodrop=1 est=1 cap=5
"""


def c4box(tmp_path, scenario, plan=PLAN_BOX, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", TS1.fakepy_s1())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "c4.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return c4box(tmp_path_factory.mktemp("c4clean"), {})


def test_every_unit_runs_at_700_w_and_1410_mhz_with_its_registered_keys(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "bytes", "gate", "timed", "bytes"]
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "700", "-sc", "0"]]
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_W=700\n"
    assert "lock probe: the session lock 1410 MHz held (C2)" in led
    assert "LOCK_PROBE_1410=held" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert [(a[0], TG.val(a, "--lock")) for a in box.tool("tileA2_gate")] == [("band", "1410")]
    cen = [a for a in box.tool("tile_census_a2") if "--models" in a]
    assert cen and all(TG.val(a, "--models") == MODEL and TG.val(a, "--tiles") == "128x128x64w8s3" for a in cen)
    lr = [a for a in box.tool("locked_r3") if PAGE in a]
    assert len(lr) == 1 and TG.val(lr[0], "--locks") == "1410" and TG.val(lr[0], "--declared-copies") == "9"
    assert TG.val(lr[0], "--model") == MODEL
    dcr = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    assert [TG.val(a, "--page-lock-mhz") for a in dcr] == ["1410", "1410"]
    assert TG.val(dcr[1], "--histogram") == PAGE and TG.val(dcr[1], "--arms") == "native,shared"
    assert TG.val(dcr[1], "--model") == MODEL
    for u in re.findall(r"^\S+ (?:bytes|timed) START: .*$", led, re.M):
        assert "1410" in u and "1710" not in u, u


@pytest.mark.parametrize("scenario", [{"enforced_w": 650.0}, {"enforced_raw": "[N/A]"}])
def test_a_700_w_refusal_stops_before_any_page(tmp_path, scenario):
    box, got = c4box(tmp_path, scenario, MOE_DRIVER_PL_WAIT_S=4)
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_700=refused\n"
    assert not box.tool("dram_counter_route_tile") and not box.tool("locked_r3")


def test_a_lock_that_does_not_read_back_stops_before_any_page(tmp_path):
    box, got = c4box(tmp_path, {"lock_reads": "1395"})
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert "LOCK_PROBE_1410=refused" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert not box.tool("calibrate_hardware") and not box.tool("dram_counter_route_tile")


def test_a_g_band_fail_ends_the_session_before_the_tp2_pages(tmp_path):
    box, got = c4box(tmp_path, {"gate_fail": ["band"]})
    led = box.ledger()
    assert got.returncode == exit_codes.INVALID, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "calibrate", "tilecensus", "bytes", "gate"]
    assert not [a for a in box.tool("locked_r3") if PAGE in a]


def test_the_full_plan_runs_whole_at_launch_plus_65_after_25_min_of_boot(tmp_path):
    """The full plan on the fake box with the deadline the driver sees after 25 min of boot: 40 min left against 25
    estimated and the driver's 8 min reserve, so nothing is dropped and all eight units run."""
    box, got = c4box(tmp_path, {}, plan=PLAN.read_text(), deadline=int(time.time()) + (65 - 25) * 60)
    led = box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert got.returncode == exit_codes.DONE and "DROPPED" not in led and len(starts) == 8, led[-3000:]
    lr = [a for a in box.tool("locked_r3") if PAGE in a]
    assert len(lr) == 1 and TG.val(lr[0], "--model") == MODEL and TG.val(lr[0], "--locks") == "1410"


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_cal4_session.sh
# --------------------------------------------------------------------------

def test_start_with_c4_delivers_the_c4_driver_the_plan_and_the_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert pages == [PAGE]
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
    assert "gh200_cal4_session.sh --dry-run" in cmds and "gh200_cal4_session.sh --setup" in cmds
    assert "driver_script=gh200_cal4_session.sh" in (lap.state / "r" / "run.env").read_text()
