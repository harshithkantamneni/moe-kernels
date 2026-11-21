"""The calibration rental C3's plan and driver off the GPU (scripts/plans/rentalC3-cal-2026-10.plan,
scripts/gh200_cal3_session.sh): the parse-only dry run, its units held to the registration
(docs/registered/2026-10-26-cal3-gh200 'plan': step, model, label, est, cap, nodrop or drop group, the totals, wall,
cost, deadline and hard cap, the plan's sha256), every histogram page it names registered and A2's E1-CAL file, the
driver against C2's pinned driver (every changed line marked C3), the plan's variants accepted by the C3 probe, the
new keys refused where they do not belong, vm_run.sh and LAMBDA.md naming the driver; then the session on the fake box
(C2's power and lock stops kept, every instrumented step on scripts/instr_probe_c3.py, a stamps unit refused on a FAIL
gate or regcheck) and `vm_run.sh start --driver gh200_cal3_session.sh` against a stub VM, every file by sha256."""
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

PLAN = REPO / "scripts" / "plans" / "rentalC3-cal-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_cal3_session.sh"
C2_DRIVER = TC2.DRIVER
HA = "docs/registered/2026-10-14-tileA-hist"
REG = json.loads((REPO / "docs" / "registered" / "2026-10-26-cal3-gh200.json").read_text())
C2_REG = TC2.REG
TILE = TC2.TILE


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def print_variants(plan=PLAN, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--print-variants", "--plan", str(plan)], capture_output=True, text=True,
                          timeout=60, env={**os.environ, "HOME": "/nonexistent"})


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_c3_plan_parses_and_every_unit_is_the_registrations():
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
    assert total == reg["est_total_min"] == 30 and f"{total} min of units" in got.stdout
    nodrop = sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop")
    assert nodrop == reg["est_nodrop_min"] == 26 and reg["drop_order_last_first"] == ["q"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    text = PLAN.read_text()
    assert "30 of units estimated (26 nodrop)" in text
    assert 70 - 25 - 8 >= total   # every unit, q included, fits the deadline less boot and the driver's reserve
    assert "--deadline <launch + 70 min" in text and "guardian hard cap\n# launch + 85 min ($3.24 at most)" in text
    assert reg["wall_min"] == 55 and reg["cost_usd"] == 2.10 and reg["deadline_after_launch_min"] == 70
    assert reg["hard_cap_after_launch_min"] == 85 and reg["cost_at_hard_cap_usd"] == 3.24 and reg["rate_usd_per_h"] == 2.29
    assert "about 55 min ($2.10 at $2.29/h)" in text
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    assert "- prelude lock-mhz=1410 power-limit-w=700" in body and "power-limit-w=900" not in body
    assert sorted(re.findall(r"lock-mhz=(\d+)", body)) == ["1410"]
    for absent in ("lockprobe", "powerproof", "timed ", "calibrate", "instr-variants", "instr-evict", "stamp-treads"):
        assert absent not in body, absent
    assert "3c.1 supported clocks, persistence, 700 W (power.max_limit read first), the session lock 1410 MHz" in got.stdout
    assert ("instrumented units, each run only on its perturb unit's hard legs and regcheck (PERTURBED on a timing "
            "FAIL alone): u7-st7 u8-stq") in got.stdout


def test_the_units_are_the_registered_cells_in_the_registered_order():
    u = REG["plan"]["units"]
    assert [(x["model"], x["step"], x["label"]) for x in u] == [
        ("-", "prelude", ""), ("olmoe-1b-7b", "bytes", "f128"), ("mixtral-8x7b", "bytes", "f128"), ("-", "gate", "gband"),
        ("-", "regcheck", "rc3"), ("-", "perturb", "pt3"), ("mixtral-8x7b", "stamps", "st7"),
        ("qwen2-57b-a14b", "stamps", "stq")]
    assert [x["drop"] for x in u] == ["nodrop"] * 7 + ["q"]
    k = {(x["model"], x["label"]): x["keys"] for x in u}
    tile = {"block-m": "128", "block-n": "128", "block-k": "64", "num-warps": "8", "num-stages": "3"}
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):   # C2's f128 pages, unchanged
        c2 = {(x["model"], x["label"]): x["keys"] for x in C2_REG["plan"]["units"]}[(m, "f128")]
        assert k[(m, "f128")] == c2 and k[(m, "f128")]["byte-groups"] == "64"
    assert k[("-", "gband")] == {"gate": "band"} and k[("-", "rc3")] == {} and k[("-", "pt3")] == {}
    for m, lab in (("mixtral-8x7b", "st7"), ("qwen2-57b-a14b", "stq")):
        x = k[(m, lab)]
        assert {a: x[a] for a in tile} == tile
        assert x["histogram"] == f"{HA}/{m}-128g1.json" == REG["pages"][m]["path"]
        assert (x["instr-stamps"], x["stamp-groups"], x["stamp-arms"]) == ("ends", "1", "native")
    names = {f"{x['model']} {x['step']} {x['label']}".strip() for x in u} | {"- prelude"}
    for blk in REG["units"].values():
        for nm in blk["units"]:
            assert nm in names, nm


def test_every_page_the_plan_names_is_a2s_e1cal_file_at_its_registered_sha256():
    import private_weight_reference as R3
    from moe.spec import MODEL_CONFIGS
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    files = set(re.findall(r"histogram=(\S+)", body))
    reg = {v["path"]: v["sha256"] for v in REG["pages"].values()}
    a2 = json.loads((REPO / "docs/registered/2026-10-20-tileA2-e1cal-gh200.json").read_text())["pages"]["files"]
    assert files == set(reg) and len(files) == 2
    for f in files:
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f] == a2[Path(f).name]["sha256"], f
        doc = json.loads((REPO / f).read_text())
        page = R3.load_histogram_page(REPO / f, MODEL_CONFIGS[doc["model"]], 128, doc["model"])
        assert [[c.label, c.n] for c in page.ordered()] == REG["pages"][doc["model"]]["cells"]


def test_the_plans_variants_are_the_ones_the_c3_probe_accepts(tmp_path):
    import instr_probe_c3 as P
    got = print_variants()
    assert got.returncode == exit_codes.REFUSED, got.stderr
    vs = json.loads(got.stdout)
    assert [v["id"] for v in vs] == ["u7-st7", "u8-stq"]
    assert [v["id"] for v in vs] == [REG["pages"][m]["variant"] for m in ("mixtral-8x7b", "qwen2-57b-a14b")]
    f = tmp_path / "v.json"
    f.write_text(got.stdout)
    loaded = P.load_variants(f)
    assert [len(v["cells"]) for v in loaded] == [4, 8]
    assert all(v["tile"] == (128, 128, 64, 8, 3) and v["spec_obj"].stamps == 3 and v["arms"] == ["native"] for v in loaded)
    for mode, extra in (("regcheck", []), ("perturb", []), ("stamps", ["--variant", "u7-st7"])):
        r = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "instr_probe_c3.py"), "--mode", mode,
                            "--variants", str(f), "--out", str(tmp_path / "o"), "--dry-run", *extra],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == exit_codes.REFUSED, r.stdout + r.stderr
        if mode == "stamps":   # no gate file given: refused before anything
            assert "REFUSED: the perturbation gate's hard legs for u7-st7 read nothing" in r.stdout
        else:
            assert "REFUSED" not in r.stdout and "2 variant(s)" in r.stdout


@pytest.mark.parametrize("body, why", [
    ("- prelude\n- regcheck label=r\n- perturb label=p\nmixtral-8x7b stamps label=s instr-stamps=ends stamp-groups=1 "
     "stamp-arms=native histogram=" + HA + "/mixtral-8x7b-128g1.json\n", "a histogram stamps unit needs block-m=..."),
    ("- prelude\n- regcheck label=r\n- perturb label=p\nmixtral-8x7b stamps label=s instr-stamps=ends stamp-groups=1 "
     "stamp-treads=2 stamp-arms=native " + TILE + " histogram=" + HA + "/mixtral-8x7b-128g1.json\n",
     "stamp-treads on a histogram stamps unit"),
    ("- prelude\n- perturb label=p\nmixtral-8x7b stamps label=s instr-stamps=ends stamp-groups=1 stamp-arms=native "
     + TILE + " histogram=" + HA + "/mixtral-8x7b-128g1.json\n", "a histogram stamps unit needs a regcheck unit before it"),
    ("- prelude\n- regcheck label=r\n- perturb label=p\nmixtral-8x7b stamps label=s instr-stamps=sample stamp-groups=1 "
     "stamp-arms=native " + TILE + " histogram=" + HA + "/mixtral-8x7b-128g1.json\n", "instr-stamps sample: cta, iter or ends"),
    ("- prelude\nmixtral-8x7b gate label=g gate=band histogram=" + HA + "/mixtral-8x7b-128g1.json\n",
     "histogram belongs to a timed, bytes or stamps unit"),
    ("- prelude\n- regcheck label=r block-m=128\n", "belongs to a timed, bytes, lockprobe or stamps unit"),
])
def test_c3s_keys_are_refused_where_they_do_not_belong(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_c3_driver_is_the_c2_driver_plus_lines_marked_c3():
    a = C2_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "C3" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and "C3'S DRIVER" in b[1]
    assert len(b) - len(a) < 60
    # the C2 driver's own header is kept whole under C3's
    assert "# THE C2 DRIVER'S OWN HEADER, UNCHANGED:" in b[:header]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE C2 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    body = "\n".join(b[header:])
    assert "scripts/instr_probe.py" not in body and body.count("scripts/instr_probe_c3.py") == 5
    assert hashlib.sha256(C2_DRIVER.read_bytes()).hexdigest() == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert REG["code_pins"]["scripts/gh200_cal2_session.sh"] == C2_REG["code_pins"]["scripts/gh200_cal2_session.sh"]
    assert (DRIVER.stat().st_mode & 0o777) == (C2_DRIVER.stat().st_mode & 0o777)


def test_the_c2_driver_refuses_the_c3_plan_and_the_c3_driver_parses_c2s():
    got = dry("--plan", str(PLAN), driver=C2_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "instr-stamps ends" in got.stderr
    got = dry("--plan", str(TC2.PLAN))
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 10 units" in got.stdout


def test_vm_run_and_lambda_name_the_c3_driver_and_nothing_else_moved():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:7] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh",
                           "gh200_tileA2_session.sh", "gh200_settleS1_session.sh", "gh200_cal2_session.sh",
                           "gh200_cal3_session.sh"]
    assert "|gh200_settleS1_session.sh|gh200_cal2_session.sh|gh200_cal3_session.sh" in vm.splitlines()[5]
    for other in (r"[g]h200_cal2_session.sh", r"[g]h200_settleS1_session.sh", r"[g]h200_(model|tile)_session.sh"):
        assert not re.search(other, "gh200_cal3_session.sh")
    lam = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_cal3_session\.sh)'", lam).group(1)
    assert re.search(pat, "bash gh200_cal3_session.sh --plan x") and not re.search(pat, "gh200_cal2_session.sh")
    assert "the calibration rental C3's copy (2026-10-26) with" in lam


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

FAKE_C3 = r'''
if tool == "instr_probe_c3":
    vid = val("--variant")
    def line(path, prefix):
        for ln in Path(path).read_text().splitlines() if path and Path(path).exists() else []:
            k, _, w = ln.partition("=")
            if k == prefix + "_" + re.sub("[^A-Za-z0-9]", "_", vid):
                return w.split()[0]
        return ""
    # as the real probe: a stamps unit's gate and regcheck lines are read before its dry run returns
    if val("--mode") == "stamps" and (line(val("--gate"), "HARD") != "PASS" or line(val("--regcheck"), "REGCHECK") != "PASS"):
        print("REFUSED: planted")
        sys.exit(2)
    if "--dry-run" in args:
        print("INSTRUMENTED KERNEL C3 (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "cache-dir.txt").write_text(os.environ.get("TRITON_CACHE_DIR", ""))
    vs = json.loads(Path(val("--variants")).read_text())
    key = lambda v: re.sub("[^A-Za-z0-9]", "_", v["id"])
    if val("--mode") == "regcheck":
        bad = SC.get("regcheck_fail", [])
        (out / "regcheck.env").write_text("".join(
            f"REGCHECK_{key(v)}={'FAIL 1 CTAs/SM against 2' if v['id'] in bad else 'PASS'}\n" for v in vs))
        sys.exit(1 if bad else 0)
    if val("--mode") == "perturb":
        bad = SC.get("gate_fail", [])
        hard = SC.get("hard_fail", [])
        (out / "gate.env").write_text("".join(
            f"GATE_{key(v)}={'FAIL' if v['id'] in bad + hard else 'PASS'} median=0.001 worst=0.002\n"
            f"HARD_{key(v)}={'FAIL' if v['id'] in hard else 'PASS'}\n" for v in vs))
        (out / "perturb.json").write_text("{}")
        sys.exit(1 if bad + hard else 0)
    (out / "stamps.json").write_text("{}")
    sys.exit(0)
'''


def fakepy_c3() -> str:
    body = TS1.fakepy_s1()
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_C3 + "sys.exit(0)" + tail


PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=700 est=1 cap=5
olmoe-1b-7b bytes label=f128 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
- gate label=gband gate=band nodrop=1 est=1 cap=5
- regcheck label=rc3 nodrop=1 est=1 cap=5
- perturb label=pt3 nodrop=1 est=1 cap=5
mixtral-8x7b stamps label=st7 {TILE} histogram={HA}/mixtral-8x7b-128g1.json instr-stamps=ends stamp-groups=1 stamp-arms=native nodrop=1 est=1 cap=5
qwen2-57b-a14b stamps label=stq {TILE} histogram={HA}/qwen2-57b-a14b-128g1.json instr-stamps=ends stamp-groups=1 stamp-arms=native drop-group=q est=1 cap=5
"""


def c3box(tmp_path, scenario, plan=PLAN_BOX, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_c3())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "c3.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return c3box(tmp_path_factory.mktemp("c3clean"), {})


def test_every_unit_runs_at_700_w_and_1410_mhz_and_the_probe_is_c3s(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "bytes", "gate", "regcheck", "perturb", "stamps",
                                                            "stamps"]
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "700", "-sc", "0"]]
    assert (box.session / "gh200-driver" / "c2-power.env").read_text() == "C2_POWER_W=700\n"
    assert "LOCK_PROBE_1410=held" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert [(a[0], TG.val(a, "--lock")) for a in box.tool("tileA2_gate")] == [("band", "1410")]
    assert not box.tool("instr_probe")
    runs = box.tool("instr_probe_c3", dry=False)
    assert [TG.val(a, "--mode") for a in runs] == ["regcheck", "perturb", "stamps", "stamps"]
    st = [a for a in runs if TG.val(a, "--mode") == "stamps"]
    assert [TG.val(a, "--variant") for a in st] == ["u6-st7", "u7-stq"]   # this box plan's own unit numbers
    for a in st:
        assert TG.val(a, "--gate").endswith("instr-gate.env") and TG.val(a, "--regcheck").endswith("instr-regcheck.env")
    vs = json.loads((box.session / "gh200-driver" / "instr-variants.json").read_text())
    assert [(v["id"], v["histogram"], v["spec"], v["block_m"], v["num_warps"], v["num_stages"]) for v in vs] == [
        ("u6-st7", f"{HA}/mixtral-8x7b-128g1.json", "stamps=ends", 128, 8, 3),
        ("u7-stq", f"{HA}/qwen2-57b-a14b-128g1.json", "stamps=ends", 128, 8, 3)]
    for u in re.findall(r"^\S+ (?:stamps|perturb) START: .*$", led, re.M):
        assert "1410" in u and "1710" not in u, u
    # every timed instrumented step runs under the session lock; regcheck is compile only
    smi = box.smi()
    assert sum(1 for a in smi if a[:2] == ["-lgc", "1410,1410"]) >= 4


@pytest.mark.parametrize("scenario, step", [({"hard_fail": ["u6-st7"]}, "hard"), ({"regcheck_fail": ["u6-st7"]}, "regcheck")])
def test_a_hard_leg_or_regcheck_fail_refuses_that_stamps_unit_only(tmp_path, scenario, step):
    box, got = c3box(tmp_path, scenario)
    led = box.ledger()
    st = [a for a in box.tool("instr_probe_c3", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert [TG.val(a, "--variant") for a in st] == ["u7-stq"], led
    if step == "hard":
        assert "stamps REFUSED: the gate's hard legs for u6-st7 read 'FAIL'" in led
    else:   # the driver's hard line reads PASS; the probe's own dry run refuses on the regcheck line
        assert "stamps REFUSED: its dry run exited 2 (REFUSED: planted)" in led


def test_a_timing_only_gate_fail_runs_the_stamps_perturbed(tmp_path):
    """Owner OD-1 (2026-10-26): a gate that fails on its timing legs alone keeps the unit."""
    box, got = c3box(tmp_path, {"gate_fail": ["u6-st7"]})
    led = box.ledger()
    st = [a for a in box.tool("instr_probe_c3", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert [TG.val(a, "--variant") for a in st] == ["u6-st7", "u7-stq"], led
    assert "stamps: the overhead gate for u6-st7 reads 'FAIL median=0.001 worst=0.002': the pages run PERTURBED" in led


def test_a_700_w_refusal_stops_before_any_page(tmp_path):
    box, got = c3box(tmp_path, {"enforced_w": 650.0}, MOE_DRIVER_PL_WAIT_S=4)
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert not box.tool("instr_probe_c3") and not box.tool("dram_counter_route_tile")


def test_a_lock_that_does_not_read_back_stops_before_any_page(tmp_path):
    box, got = c3box(tmp_path, {"lock_reads": "1395"})
    assert got.returncode == exit_codes.REFUSED and "prelude REFUSED: the session lock 1410 MHz did not read back" in box.ledger()
    assert not box.tool("instr_probe_c3")


def test_a_g_band_fail_ends_the_session_before_the_instrumented_units(tmp_path):
    box, got = c3box(tmp_path, {"gate_fail": ["band"]})
    led = box.ledger()
    assert got.returncode == exit_codes.INVALID, led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "bytes", "gate"]
    assert not box.tool("instr_probe_c3", dry=False)


@pytest.mark.parametrize("minutes, dropped", [(70 - 25, False), (60 - 25, True)])
def test_q_is_not_dropped_at_the_start_at_launch_plus_70_after_25_min_of_boot(tmp_path, minutes, dropped):
    """The full plan on the fake box with the deadline the driver sees after 25 min of boot: at launch + 70 nothing is
    dropped (45 min left against 30 estimated); at launch + 60 the driver drops q."""
    box, got = c3box(tmp_path, {}, plan=PLAN.read_text(), deadline=int(time.time()) + minutes * 60)
    led = box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    if dropped:
        assert "DROPPED" in led and len(starts) == 7, led
    else:
        assert got.returncode == exit_codes.DONE and "DROPPED" not in led and len(starts) == 8, led[-3000:]


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_cal3_session.sh
# --------------------------------------------------------------------------

def test_start_with_c3_delivers_the_c3_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
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
    assert "gh200_cal3_session.sh --dry-run" in cmds and "gh200_cal3_session.sh --setup" in cmds
    assert "driver_script=gh200_cal3_session.sh" in (lap.state / "r" / "run.env").read_text()
