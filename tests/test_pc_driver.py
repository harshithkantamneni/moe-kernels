"""Rental PC's plan and driver off the GPU (scripts/plans/rentalPC-2026-11.plan, scripts/gh200_pc_session.sh): the
parse-only dry run listing every unit, held to the registration; the plan generated from pccommon; the driver against
C3's pinned driver (every changed line marked PC); every new key refused where it does not belong; the plan's variants
accepted by the probes; then the session on the fake box: the unit locks set, read back before and after and the
session restored, both probes on their own variants, the regcheck drop rule (owner decision 2), a G-BAND FAIL dropping
the Phi group only (owner decision 5), the deadline's cut order with E1 and E2 together, a unit lock the prelude could
not hold; and `vm_run.sh start --driver gh200_pc_session.sh` delivering the driver, the plan and every page byte for
byte, through a stub ssh that drains its stdin."""
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
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pccommon as P  # noqa: E402
import test_cal2_plan as TC2  # noqa: E402
import test_cal3_plan as TC3  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / P.PLAN
DRIVER = REPO / P.DRIVER
C3_DRIVER = REPO / P.PARENT_DRIVER
REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
T128 = P.tile_keys(P.P128)
T64 = P.tile_keys(P.P64)
PG = P.PAGES_DIR


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def units_of(stdout: str) -> list[list[str]]:
    return TG._units(stdout)


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_plan_is_generated_and_its_dry_run_lists_every_unit():
    assert PLAN.read_text() == P.plan_text()
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = units_of(got.stdout)
    reg = REG["units"]["rows"]
    assert len(units) == len(reg) == len(P.UNITS) == 20 and "a PLAN of 20 units" in got.stdout
    import test_tileA2_plan as TA2
    import test_tile_plan as TT
    for u, r in zip(units, reg, strict=True):
        assert (u[2], u[1], TT._label(u) or "", int(u[3]), int(u[4]), TA2._drop(u)) == (
            r["step"], r["model"], r["label"], r["est"], r["cap"], r["drop"]), u
    figs = REG["plan"]
    assert sum(int(u[3]) for u in units) == figs["units_min"] == 84 and f"{figs['units_min']} min of units" in got.stdout
    assert (figs["wall_min"], figs["cost_usd"], figs["deadline_after_launch_min"], figs["hard_cap_after_launch_min"],
            figs["cost_at_hard_cap_usd"]) == (109, 4.16, 125, 140, 5.34)
    assert figs["deadline_after_launch_min"] - P.BOOT_MIN - P.RESERVE_MIN >= figs["units_min"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == figs["sha256"]
    assert figs["drop_order_last_first"] == ["rep", "b16", "cnt", "d16", "phi"]
    assert "drop order when time runs short: the last unit first" in got.stdout
    assert "instrumented units, each run only on its perturb unit's hard legs and regcheck (PERTURBED on a timing FAIL " \
           "alone): u5-sa u7-sc u13-se1 u14-se2 u17-sd u20-sr" in got.stdout
    assert "the prelude also probes the floor lock(s) 1710 MHz" in got.stdout


def test_e1_and_e2_share_one_drop_group_and_b128g16_depends_on_it():
    rows = {u["label"]: u for u in REG["units"]["rows"]}
    assert rows["se1"]["drop"] == rows["se2"]["drop"] == "phi"
    assert [k for k, u in rows.items() if u["depends"]] == ["b128g16"] and rows["b128g16"]["depends"] == "phi"
    assert REG["units"]["rows"][2]["cap"] == 20
    assert [rows[k]["drop"] for k in ("b128g1", "b64g1", "b128g16", "sd", "cnt", "sr")] == ["phi", "phi", "b16", "d16",
                                                                                           "cnt", "rep"]
    assert [u["label"] for u in REG["units"]["rows"] if u["drop"] == "nodrop"] == ["", "pc", "rcp", "ptp", "sa", "ptc", "sc"]


def test_the_plans_variants_are_the_ones_each_probe_accepts(tmp_path):
    import instr_probe_c3 as C3
    import instr_probe_pc as PR
    for flag, probe, ids in (("--print-variants", C3, ["u13-se1"]),
                             ("--print-pc-variants", PR, ["u5-sa", "u7-sc", "u14-se2", "u17-sd", "u20-sr"])):
        got = subprocess.run(["bash", str(DRIVER), flag, "--plan", str(PLAN)], capture_output=True, text=True, timeout=60,
                             env={**os.environ, "HOME": "/nonexistent"})
        assert got.returncode == exit_codes.REFUSED, got.stderr
        f = tmp_path / f"{probe.__name__}.json"
        f.write_text(got.stdout)
        vs = probe.load_variants(f)
        assert [v["id"] for v in vs] == ids
    assert ids == [P.VARIANTS[k] for k in ("sa", "sc", "se2", "sd", "sr")]


def test_the_pc_driver_is_the_c3_driver_plus_lines_marked_pc():
    a = C3_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "PC" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and "RENTAL PC'S DRIVER" in b[1]
    assert "# THE C3 DRIVER'S OWN HEADER, UNCHANGED:" in b[:header]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE C3 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    body = "\n".join(b[header:])
    assert body.count("scripts/instr_probe_pc.py") == 4 and "scripts/dram_counter_route_decode3.py" in body
    c3reg = json.loads((REPO / "docs/registered/2026-10-26-cal3-gh200.json").read_text())
    assert hashlib.sha256(C3_DRIVER.read_bytes()).hexdigest() == c3reg["code_pins"]["scripts/gh200_cal3_session.sh"] \
        == REG["code_pins"]["scripts/gh200_cal3_session.sh"]
    assert (DRIVER.stat().st_mode & 0o777) == (C3_DRIVER.stat().st_mode & 0o777)
    for ln in b:
        assert chr(0x2014) not in ln and chr(0x2013) not in ln


def test_the_ported_decode_step_is_the_d3_drivers_with_the_unit_lock():
    d3 = (REPO / "scripts/gh200_decode3_session.sh").read_text()
    pc = DRIVER.read_text()
    def take(src):
        lines = [ln.split("   #")[0].rstrip() for ln in src[src.index("step_decodebytes() {"):].splitlines()[1:]]
        return lines[:lines.index("}") + 1]
    d3_body = take(d3)
    pc_body = [x for x in take(pc) if "pc_lock_ok" not in x]
    assert [x.replace("pc_locked_run", "locked_run") for x in pc_body] == d3_body and len(d3_body) == 13


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=f probe=pc byte-groups=64\n", "probe belongs to a regcheck, perturb or stamps"),
    ("- prelude\n- regcheck label=r probe=pc page=" + PG + "/pc-a.json\n", "page belongs to a probe=pc stamps unit"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG
     + "/pc-a.json gates=s\n", "gates belongs to a probe=pc perturb unit"),
    ("- prelude\nolmoe-1b-7b bytes label=f kernels=wide byte-groups=64\n", "kernels belongs to a decodebytes unit"),
    ("- prelude\n- regcheck label=r probe=pc lock-mhz=1710\n", "lock-mhz belongs to the prelude"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s\n",
     "'- stamps' is a page stamps unit's"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=x\n- stamps label=s probe=pc page=" + PG
     + "/pc-a.json\n", "needs a probe=pc perturb unit before it naming it"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc\n", "names the stamps units it gates"),
    ("- prelude\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG + "/pc-a.json\n",
     "needs a probe=pc regcheck unit before it"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s,t\n- stamps label=s probe=pc page=" + PG
     + "/pc-a.json\n", "gates=t names no probe=pc stamps unit"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\nphi-3.5-moe stamps label=s probe=pc "
     "page=" + PG + "/pc-a.json\n", "a page stamps unit's model is its cells'"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG
     + "/pc-a.json block-m=64\n", "a tile is whole"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\n- stamps label=s probe=pc page=" + PG
     + "/nope.json\n", "no such file"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=s\nphi-3.5-moe stamps label=s " + T128
     + " histogram=" + P.PHI_128 + " instr-stamps=ends stamp-groups=1 stamp-arms=native\n",
     "a C3 histogram stamps unit needs a probe=c3 regcheck and perturb"),
    ("- prelude\nmixtral-8x7b decodebytes label=c kernels=wide\n", "a decodebytes unit needs label= and histogram="),
    ("- prelude\n- regcheck label=r probe=x\n", "probe x: pc or c3"),
    ("- prelude\n- regcheck label=r probe=pc\n- perturb label=p probe=pc gates=S\n", "gates S: stamps unit labels"),
])
def test_pcs_keys_are_refused_where_they_do_not_belong(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_c3_driver_refuses_the_pc_plan_and_the_pc_driver_parses_c3s():
    got = dry("--plan", str(PLAN), driver=C3_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "no key 'probe'" in got.stderr
    got = dry("--plan", str(TC3.PLAN))
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 8 units" in got.stdout


def test_vm_run_and_lambda_name_the_pc_driver_and_nothing_else_moved():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:10] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh",
                            "gh200_tileA2_session.sh", "gh200_settleS1_session.sh", "gh200_cal2_session.sh",
                            "gh200_cal3_session.sh", "gh200_cal4_session.sh", "gh200_decode3_session.sh",
                            "gh200_tileB3_session.sh"] and drivers[10:] == ["gh200_pc_session.sh"]
    assert vm.splitlines()[5].endswith("|gh200_tileB3_session.sh|gh200_decode3_session.sh]")
    assert "--driver gh200_pc_session.sh" in vm.splitlines()[6]
    assert "page=[^[:space:]]+" in vm
    lam = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_pc_session\.sh)'", lam).group(1)
    assert re.search(pat, "bash gh200_pc_session.sh --plan x") and not re.search(pat, "gh200_cal3_session.sh")
    for other in (r"[g]h200_cal3_session.sh", r"[g]h200_tileB3_session.sh", r"[g]h200_(model|tile)_session.sh"):
        assert not re.search(other, "gh200_pc_session.sh")
    assert "rental PC's copy (2026-11-06) with" in lam


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

FAKE_PC = r'''
if tool == "instr_probe_pc":
    vid = val("--variant")
    def line(path, prefix):
        for ln in Path(path).read_text().splitlines() if path and Path(path).exists() else []:
            k, _, w = ln.partition("=")
            if k == prefix + "_" + re.sub("[^A-Za-z0-9]", "_", vid):
                return w.split()[0]
        return ""
    if val("--mode") == "stamps" and (line(val("--gate"), "HARD") != "PASS" or line(val("--regcheck"), "REGCHECK") != "PASS"):
        print("REFUSED: planted")
        sys.exit(2)
    if val("--mode") == "regcheck" and SC.get("pc_regcheck_refuse"):
        print("REFUSED: planted regcheck refusal")
        sys.exit(2)
    if "--dry-run" in args:
        print("INSTRUMENTED KERNEL PC (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    vs = json.loads(Path(val("--variants")).read_text())
    key = lambda i: re.sub("[^A-Za-z0-9]", "_", i)
    if val("--mode") == "regcheck":
        if SC.get("pc_regcheck_crash"):
            sys.exit(int(SC["pc_regcheck_crash"]))
        bad = SC.get("pc_regcheck_fail", [])
        (out / "regcheck.env").write_text("".join(
            f"REGCHECK_{key(v['id'])}={'FAIL 2 CTAs/SM against 3' if v['id'] in bad else 'PASS'}\n" for v in vs))
        sys.exit(1 if bad else 0)
    if val("--mode") == "perturb":
        only = val("--only").split(",")
        (out / "only.txt").write_text(val("--only"))
        (out / "gate.env").write_text("".join(
            f"GATE_{key(i)}=PASS median=0.001 worst=0.002\nHARD_{key(i)}=PASS\n" for i in only))
        (out / "perturb.json").write_text("{}")
        sys.exit(0)
    (out / "stamps.json").write_text(json.dumps({"variant": vid}))
    sys.exit(0)

if tool == "dram_counter_route_decode3":
    if "--dry-run" in args:
        print("DECODE COUNTER ROUTE (planted)")
        sys.exit(2)
    o = Path(val("--out")); o.parent.mkdir(parents=True, exist_ok=True); o.write_text("{}")
    print("RESULT: VALIDITY V_ALIGN_grid PASS planted")
    sys.exit(0)
'''


def fakepy_pc() -> str:
    body = TC3.fakepy_c3()
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_PC + "sys.exit(0)" + tail


PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=700 est=1 cap=5
- regcheck label=rcp probe=pc nodrop=1 est=1 cap=5
- perturb label=ptp probe=pc gates=sa,sr lock-mhz=1710 nodrop=1 est=1 cap=5
- stamps label=sa probe=pc page={PG}/pc-a.json lock-mhz=1710 nodrop=1 est=1 cap=5
- perturb label=ptc probe=pc gates=se2 nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f128 {T128} byte-groups=64 drop-group=phi est=1 cap=5
- gate label=gband gate=band drop-group=phi est=1 cap=5
- regcheck label=rc3 probe=c3 drop-group=phi est=1 cap=5
- perturb label=pt3 probe=c3 drop-group=phi est=1 cap=5
phi-3.5-moe stamps label=se1 {T128} histogram={P.PHI_128} instr-stamps=ends stamp-groups=1 stamp-arms=native drop-group=phi est=1 cap=5
phi-3.5-moe stamps label=se2 probe=pc {T64} histogram={P.PHI_64} instr-stamps=ends stamp-groups=1 stamp-arms=native drop-group=phi est=1 cap=5
mixtral-8x7b decodebytes label=cnt histogram={PG}/{P.CNT_PAGE}.json kernels=wide lock-mhz=1710 drop-group=cnt est=1 cap=5
- stamps label=sr probe=pc page={PG}/pc-r.json lock-mhz=1710 drop-group=rep est=1 cap=5
"""


def pcbox(tmp_path, scenario, plan=PLAN_BOX, deadline=None, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_pc())
    TG._exe(box.bin / "nvidia-smi", TC2.SMI_C2)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "pc.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(deadline or TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


def starts(led: str) -> list[str]:
    return re.findall(r"^\S+ (\w+) START", led, re.M)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return pcbox(tmp_path_factory.mktemp("pcclean"), {})


def test_every_unit_runs_with_its_probe_its_variants_and_its_lock(clean):
    box, got = clean
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert starts(led) == ["prelude", "regcheck", "perturb", "stamps", "perturb", "bytes", "gate", "regcheck", "perturb",
                           "stamps", "stamps", "decodebytes", "stamps"]
    D = box.session / "gh200-driver"
    pcv = json.loads((D / "pc-variants.json").read_text())
    c3v = json.loads((D / "instr-variants.json").read_text())
    assert [v["id"] for v in pcv] == ["u4-sa", "u11-se2", "u13-sr"] and [v["id"] for v in c3v] == ["u10-se1"]
    assert pcv[0] == {"id": "u4-sa", "kind": "stamps", "probe": "pc", "page": f"{PG}/pc-a.json", "spec": "stamps=ends"}
    assert (pcv[1]["histogram"], pcv[1]["block_m"], pcv[1]["num_warps"]) == (P.PHI_64, 64, 8)
    pc = box.tool("instr_probe_pc", dry=False)
    assert [(TG.val(a, "--mode"), TG.val(a, "--variant") if "--variant" in a else TG.val(a, "--only") if "--only" in a
             else "") for a in pc] == [("regcheck", ""), ("perturb", "u4-sa,u13-sr"), ("stamps", "u4-sa"),
                                       ("perturb", "u11-se2"), ("stamps", "u11-se2"), ("stamps", "u13-sr")]
    assert all(TG.val(a, "--variants").endswith("pc-variants.json") for a in pc)
    c3 = box.tool("instr_probe_c3", dry=False)
    assert [TG.val(a, "--mode") for a in c3] == ["regcheck", "perturb", "stamps"]
    assert all(TG.val(a, "--variants").endswith("instr-variants.json") for a in c3)
    dcr = box.tool("dram_counter_route_decode3", dry=False)
    assert len(dcr) == 1 and TG.val(dcr[0], "--kernels") == "wide" and TG.val(dcr[0], "--page-lock-mhz") == "1710"
    assert TG.val(dcr[0], "--histogram") == f"{PG}/{P.CNT_PAGE}.json"
    # the unit locks: 1710 on ptp, sa, cnt and sr, read back after each tool; the session's 1410 on the rest
    locks = (D / "pc-locks.env").read_text().splitlines()
    assert [ln.split("=")[0] for ln in locks] == ["PC_LOCK_mixtral_8x7b_ptp", "PC_LOCK_mixtral_8x7b_sa",
                                                  "PC_LOCK_mixtral_8x7b_ptc", "PC_LOCK_mixtral_8x7b_pt3",
                                                  "PC_LOCK_phi_3_5_moe_se2", "PC_LOCK_mixtral_8x7b_cnt",
                                                  "PC_LOCK_mixtral_8x7b_sr"]
    want = {"ptp": 1710, "sa": 1710, "ptc": 1410, "pt3": 1410, "se2": 1410, "cnt": 1710, "sr": 1710}
    for ln in locks:
        lab = ln.split("=")[0].split("_")[-1]
        assert f"={want[lab]} before={want[lab]} after={want[lab]} exit=0" in ln, ln
    assert "LOCK_PROBE_1710=held" in (D / "locks.env").read_text()
    lgc = [a[1] for a in box.smi() if a[:1] == ["-lgc"]]
    assert lgc.count("1710,1710") >= 5 and lgc[-1] == "1710,1710"
    for u in re.findall(r"^\S+ (?:perturb|stamps|decodebytes) START: .*$", led, re.M):
        lab = re.search(r"\[unit \d+/13 \S+ (\S+) ", u).group(1)
        assert ("the 1710 lock" in u) == (lab in ("ptp", "sa", "cnt", "sr")), u
    assert "PC DROP RULE" not in led and "DROPPED" not in led


def test_after_a_1710_unit_the_next_unit_runs_at_the_session_lock(clean):
    box, _ = clean
    calls = [c for c in box.calls() if c.get("exe") == "nvidia-smi" and c["argv"][:1] in (["-lgc"], ["-rgc"])
             or c.get("tool") in ("instr_probe_pc", "instr_probe_c3", "dram_counter_route_tile") and not c.get("dry")]
    seq = []
    for c in calls:
        if c.get("exe") == "nvidia-smi":
            seq.append(c["argv"][1] if c["argv"][0] == "-lgc" else "reset")
        else:
            seq.append(c["tool"] + ":" + (c["argv"][c["argv"].index("--mode") + 1] if "--mode" in c["argv"] else "run"))
    i = seq.index("instr_probe_pc:stamps")
    j = seq.index("instr_probe_pc:perturb", i)
    assert "reset" in seq[i:j] and seq[j - 1] == "1410,1410", seq[i:j + 1]


def test_a_regcheck_that_refuses_drops_every_pc_unit_and_the_existing_tools_still_run(tmp_path):
    box, got = pcbox(tmp_path, {"pc_regcheck_refuse": True})
    led = box.ledger()
    assert "PC DROP RULE: the probe=pc regcheck read '2'" in led, led
    for lab in ("ptp", "sa", "ptc", "se2", "sr"):
        assert re.search(rf"DROPPED unit \d+/13 \S+ {lab} .*owner decision 2", led), (lab, led)
    assert not box.tool("instr_probe_pc", dry=False)
    assert starts(led) == ["prelude", "regcheck", "bytes", "gate", "regcheck", "perturb", "stamps", "decodebytes"]
    assert [TG.val(a, "--mode") for a in box.tool("instr_probe_c3", dry=False)] == ["regcheck", "perturb", "stamps"]
    assert box.tool("dram_counter_route_decode3", dry=False)
    assert got.returncode == exit_codes.INVALID


@pytest.mark.parametrize("rc", [4, 1])
def test_a_regcheck_that_crashes_drops_every_pc_unit_too(tmp_path, rc):
    """Review MAJOR-1: an uncaught Python error exits 1 with no regcheck line; the drop rule reads the lines too."""
    box, got = pcbox(tmp_path, {"pc_regcheck_crash": rc})
    led = box.ledger()
    assert f"PC DROP RULE: the probe=pc regcheck read '{rc}'" in led
    if rc == 1:
        assert "and wrote no line for u4-sa u11-se2 u13-sr" in led
    assert [TG.val(a, "--mode") for a in box.tool("instr_probe_pc", dry=False)] == ["regcheck"]
    assert box.tool("instr_probe_c3", dry=False) and box.tool("dram_counter_route_decode3", dry=False)


def test_a_variants_regcheck_fail_refuses_that_stamps_unit_only(tmp_path):
    box, got = pcbox(tmp_path, {"pc_regcheck_fail": ["u4-sa"]})
    led = box.ledger()
    assert "PC DROP RULE" not in led
    st = [TG.val(a, "--variant") for a in box.tool("instr_probe_pc", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert st == ["u11-se2", "u13-sr"], led
    assert "stamps REFUSED: its dry run exited 2 (REFUSED: planted)" in led


def test_a_g_band_fail_drops_the_phi_group_and_the_session_goes_on(tmp_path):
    box, got = pcbox(tmp_path, {"gate_fail": ["band"]})
    led = box.ledger()
    assert "PC GATE: " in led and "drop-group phi" in led
    for lab in ("rc3", "pt3", "se1", "se2"):
        assert re.search(rf"DROPPED unit \d+/13 \S+ {lab} .*drop-group phi did not pass", led), (lab, led)
    assert not box.tool("instr_probe_c3", dry=False)
    st = [TG.val(a, "--variant") for a in box.tool("instr_probe_pc", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert st == ["u4-sa", "u13-sr"] and box.tool("dram_counter_route_decode3", dry=False)
    assert starts(led)[-2:] == ["decodebytes", "stamps"] and "STOPPED: " not in led


def test_a_perturb_unit_gates_no_dropped_unit(tmp_path):
    """Review MINOR-10: --only names the gated units still in the plan; a G-BAND drop removes se2 from ptc's list."""
    plan = PLAN_BOX.replace("- perturb label=ptc probe=pc gates=se2 nodrop=1 est=1 cap=5\n", "")
    plan = plan.replace("- gate label=gband gate=band drop-group=phi est=1 cap=5\n",
                        "- gate label=gband gate=band drop-group=phi est=1 cap=5\n- perturb label=ptc probe=pc gates=se2,sr nodrop=1 est=1 cap=5\n")
    box, got = pcbox(tmp_path, {"gate_fail": ["band"]}, plan=plan)
    only = [TG.val(a, "--only") for a in box.tool("instr_probe_pc", dry=False) if TG.val(a, "--mode") == "perturb"]
    assert only == ["u4-sa,u13-sr", "u13-sr"], (only, box.ledger()[-3000:])


def test_a_unit_lock_the_prelude_could_not_hold_refuses_its_units_only(tmp_path):
    box, got = pcbox(tmp_path, {"clocks": [c for c in range(1980, 344, -15) if c != 1710]})
    led = box.ledger()
    assert "LOCK_PROBE_1710=unsupported" in (box.session / "gh200-driver" / "locks.env").read_text()
    assert "perturb REFUSED: the unit's lock 1710 MHz reads 'unsupported'" in led
    assert "decodebytes REFUSED: the unit's lock 1710 MHz reads 'unsupported'" in led
    st = [TG.val(a, "--variant") for a in box.tool("instr_probe_pc", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert st == ["u11-se2"] and not box.tool("dram_counter_route_decode3", dry=False)


@pytest.mark.parametrize("minutes, dropped", [
    (125, []),                                                  # the registered deadline: nothing dropped
    (110, ["sr", "b128g16"]),                                   # 77 min left against 84: rep, then b16
    (80, ["sr", "b128g16", "cnt", "sd", "b64g1", "b128g1", "se2", "se1", "pt3", "rc3", "gband", "f128", "f128"]),
])
def test_the_deadline_cuts_in_the_registered_order_e1_and_e2_together(tmp_path, minutes, dropped):
    """The full plan on the fake box with the deadline the driver sees after 25 min of boot."""
    box, got = pcbox(tmp_path, {}, plan=PLAN.read_text(), deadline=int(time.time()) + (minutes - 25) * 60)
    led = box.ledger()
    got_d = re.findall(r"DROPPED unit \d+/20 \S+ (\S+) ", led)
    assert sorted(got_d) == sorted(dropped), led[-4000:]
    if dropped:
        assert got_d[:2] == ["sr", "b128g16"]
    if "se1" in dropped:
        assert "se2" in dropped


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_pc_session.sh
# --------------------------------------------------------------------------

def test_start_with_pc_delivers_the_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = P.PLAN
    pages = sorted(set(re.findall(r"(?:histogram|page)=(\S+)", PLAN.read_text())))
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
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')   # the stub drains its stdin
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
    assert "gh200_pc_session.sh --dry-run" in cmds and "gh200_pc_session.sh --setup" in cmds
    assert "driver_script=gh200_pc_session.sh" in (lap.state / "r" / "run.env").read_text()


def test_the_pre_setup_dry_run_prints_the_pc_plan_from_the_files_vm_run_copies(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(DRIVER, home / DRIVER.name)
    shutil.copy(PLAN, home / PLAN.name)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    args = ["bash", str(home / DRIVER.name), "--dry-run", "--plan", P.PLAN]
    bare = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert "no such file" in bare.stdout + bare.stderr
    for p in sorted(set(re.findall(r"(?:histogram|page)=(\S+)", PLAN.read_text()))):
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    got = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 20 units" in got.stdout, got.stdout[-2000:] + got.stderr
