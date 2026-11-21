"""The settle rental S1's plan and driver off the GPU (scripts/plans/rentalS1-settle-2026-10.plan,
scripts/gh200_settleS1_session.sh): the parse-only dry run, its units held to the registration
(docs/registered/2026-10-21-settleS1-gh200 'plan': step, model, label, est, cap, nodrop or drop group, the totals, the
plan's sha256), every page it names registered, the driver against A2's pinned driver (every changed line marked S1),
the S1 keys and where they are refused, the power read-back (a stale 700 W first reading no longer reaches
power-set.txt; a limit that never reads back refuses the session), and the session on the fake box: each unit
reaches its tool with its lock, treads, replay mode and stall metrics; then `vm_run.sh start --driver
gh200_settleS1_session.sh` against a stub VM, every delivered file by sha256."""
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
import test_tileA2_plan as TA2  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalS1-settle-2026-10.plan"
DRIVER = REPO / "scripts" / "gh200_settleS1_session.sh"
A2_DRIVER = TA2.DRIVER
H = "docs/registered/2026-10-21-settleS1-hist"
REG = json.loads((REPO / "docs" / "registered" / "2026-10-21-settleS1-gh200.json").read_text())
TILE = "block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3"


def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


# --------------------------------------------------------------------------
# the plan, parse only, held to the registration
# --------------------------------------------------------------------------

def test_the_s1_plan_parses_and_every_unit_is_the_registrations():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    reg = REG["plan"]
    assert len(units) == reg["units_count"] == 14 and "a PLAN of 14 units" in got.stdout
    for u, r in zip(units, reg["units"], strict=True):
        label = TT._label(u) or ""
        assert (u[2], u[1], label, int(u[3]), int(u[4]), TA2._drop(u)) == (r["step"], r["model"], r["label"], r["est"], r["cap"], r["drop"]), u
    total = sum(int(u[3]) for u in units)
    assert total == reg["est_total_min"] == 47 and f"{total} min of units" in got.stdout
    assert sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop") == reg["est_nodrop_min"] == 40
    assert reg["drop_order_last_first"] == ["st"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == reg["sha256"]
    assert "the prelude also probes the floor lock(s) 1305 1005 1410 MHz" in got.stdout
    text = PLAN.read_text()
    assert "47 of units estimated (40 nodrop" in text
    assert [int(u[4]) for u in units if TT._label(u) == "ar"] == [10, 10]
    assert 85 - 25 - 8 >= sum(int(u[3]) for u in units if TA2._drop(u) == "nodrop")   # nodrop fits the 52 min and "launch + 85 min" in text and "hard cap launch + 100 min" in text
    assert reg["wall_min"] == 72 and reg["cost_usd"] == 2.75 and reg["deadline_after_launch_min"] == 85
    assert reg["hard_cap_after_launch_min"] == 100 and "about 72 min ($2.75 at $2.29/h)" in text
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    assert "- prelude lock-mhz=1410 power-limit-w=900" in body
    assert [u[2] for u in units[:3]] == ["prelude", "powerproof", "bytes"]
    for u in units:   # the ledger names the lock each unit runs at (A2's printed 1710 for every unit)
        assert "at the 1710 lock" not in u[5]
    assert "gate" not in body and "lockprobe" not in body and "timed" not in body and "calibrate" not in body
    assert sorted(re.findall(r"lock-mhz=(\d+)", body)) == ["1005", "1305", "1410", "1410"]


def test_each_measurement_runs_the_registered_cells_and_keys():
    units = {(u["model"], u["label"]): u for u in REG["plan"]["units"]}
    for model in ("olmoe-1b-7b", "qwen2-57b-a14b"):
        k, a = units[(model, "kr")]["keys"], units[(model, "ar")]["keys"]
        assert k["histogram"] == a["histogram"] and a["replay"] == "application" and "replay" not in k
        assert k["arms"] == a["arms"] == "native"
    for label, lock in (("f1305", "1305"), ("f1005", "1005"), ("f1410", "1410")):
        u = units[("olmoe-1b-7b", label)]["keys"]
        assert u["lock-mhz"] == lock and u["byte-groups"] == "64" and "histogram" not in u and "byte-treads" not in u
    assert units[("mixtral-8x7b", "p128g16")]["keys"]["byte-treads"] == "3,4,8"
    assert units[("mixtral-8x7b", "p128g16")]["keys"]["byte-groups"] == "16"
    p64 = units[("mixtral-8x7b", "p64g1")]["keys"]
    assert p64["byte-treads"] == "2" and p64["block-m"] == "64" and p64["num-warps"] == "8"
    for model in ("qwen2-57b-a14b", "olmoe-1b-7b"):
        assert units[(model, "st")]["keys"]["stall-metrics"] == "1" and units[(model, "st")]["drop"] == "st"
    assert units[("olmoe-1b-7b", "f1410")]["drop"] == "nodrop"
    assert set(REG["measurements"]) == {"M0", "M1", "M2", "M3", "M4", "M5"}
    for m, x in REG["measurements"].items():
        assert x["read_by"].startswith("scripts/scoring/settleS1/read_s1.py ") and x["constrains"], m
        for u in x["units"]:
            model, label = u.split(" ")[:2]
            assert (model, label) in units, (m, u)


def test_every_page_the_plan_names_is_registered_at_its_sha256_and_parses_under_r3():
    import private_weight_reference as R3
    from moe.spec import MODEL_CONFIGS
    body = "\n".join(ln.split("#", 1)[0] for ln in PLAN.read_text().splitlines())
    files = set(re.findall(r"histogram=(\S+)", body))
    reg = {v["path"]: v["sha256"] for v in REG["pages"]["files"].values()}
    assert files == set(reg) and len(files) == 5
    for f in files:
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f], f
        doc = json.loads((REPO / f).read_text())
        page = R3.load_histogram_page(REPO / f, MODEL_CONFIGS[doc["model"]], int(doc["block_m"]), doc["model"])
        assert all(c.arms == ("native",) for c in page.cells), f
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/settleS1/hist_s1.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stdout + got.stderr


def test_the_m1_cells_are_the_e1cal_cells_byte_for_byte():
    src = {d: json.loads((REPO / "docs/registered/2026-10-14-tileA-hist" / f"{d}-128g1.json").read_text())
           for d in ("olmoe-1b-7b", "qwen2-57b-a14b")}
    for model, want in (("olmoe-1b-7b", {(1, "PT"), (4, "PT")}), ("qwen2-57b-a14b", {(2, "PT")})):
        doc = json.loads((REPO / H / f"{model}-128g1-rp.json").read_text())
        assert {(c["n"], c["label"]) for c in doc["cells"]} == want
        assert doc["shuffle_seed"] == src[model]["shuffle_seed"]
        for c in doc["cells"]:
            s = next(x for x in src[model]["cells"] if (x["n"], x["label"]) == (c["n"], c["label"]))
            assert c["counts"] == s["counts"]


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=a byte-treads=0\n", "byte-treads 0: treads 1..9"),
    ("- prelude\nolmoe-1b-7b bytes label=a byte-treads=3 histogram=docs/registered/2026-10-21-settleS1-hist/olmoe-1b-7b-128g1-rp.json\n",
     "byte-treads is a ladder page's"),
    ("- prelude\nolmoe-1b-7b timed label=a byte-treads=3\n", "byte-treads belongs to a bytes unit"),
    ("- prelude\nolmoe-1b-7b bytes label=a replay=kernel\n", "replay kernel: application"),
    ("- prelude\nolmoe-1b-7b bytes label=a replay=application\n", "replay needs histogram="),
    ("- prelude\nolmoe-1b-7b bytes label=a stall-metrics=1\n", "stall-metrics needs histogram="),
    ("- prelude\nolmoe-1b-7b bytes label=a stall-metrics=2\n", "stall-metrics 2: 1 or leave it out"),
    ("- prelude\n- tilecensus label=c tiles=128x128x64w8s3 models=olmoe-1b-7b stall-metrics=1\n", "stall-metrics belongs to a bytes unit"),
    ("- prelude\n- tilecensus label=c tiles=128x128x64w8s3 models=olmoe-1b-7b lock-mhz=1410\n", "lock-mhz belongs to the prelude"),
])
def test_the_s1_keys_are_refused_where_they_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_a_bytes_unit_takes_its_own_lock_and_the_prelude_probes_it(tmp_path):
    f = tmp_path / "p.plan"
    f.write_text("- prelude lock-mhz=1410\nolmoe-1b-7b bytes label=f lock-mhz=1005 byte-groups=64\n")
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and "the prelude also probes the floor lock(s) 1005 MHz" in got.stdout


def test_the_dry_run_names_each_units_own_lock():
    got = dry("--plan", str(PLAN))
    what = {TT._label(u): got.stdout.split("\n")[i + 1] for i, ln in enumerate(got.stdout.split("\n"))
            for u in TG._units(ln) if u[2] == "bytes"}
    assert "at the 1305 lock" in what["f1305"] and "at the 1005 lock" in what["f1005"] and "at the 1410 lock" in what["kr"]


def test_the_a2_driver_refuses_the_s1_plan():
    got = dry("--plan", str(PLAN), driver=A2_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "no step 'powerproof'" in got.stderr


def test_the_s1_driver_is_the_a2_driver_plus_lines_marked_s1():
    a = A2_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "S1" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and b[1].startswith("# THE GH200 MODEL-TEST SESSION") and len(b) - len(a) < 130
    removed = [ln for tag, i1, i2, j1, j2 in hunks if tag == "replace" for ln in a[i1:i2]]
    assert '  nvidia-smi -q -d POWER > "$S/power-set.txt"' in removed
    text = DRIVER.read_text()
    assert "POWER_LIMIT_W=\"${MOE_DRIVER_POWER_LIMIT_W:-900}\"" in text and "s1_power_settle ||" in text
    assert "scripts/dram_counter_route_s1.py" in text and "REFUSED (pre-setup, tileA2-gates P-W)" in text


def test_the_lambda_stop_instruction_and_vm_run_name_the_s1_driver():
    text = (REPO / "docs" / "LAMBDA.md").read_text()
    pat = re.search(r"pkill -TERM -f '(\[g\]h200_settleS1_session\.sh)'", text).group(1)
    assert re.search(pat, "bash gh200_settleS1_session.sh --plan x") and not re.search(pat, "gh200_tileA2_session.sh")
    for other in (r"[g]h200_(model|tile)_session.sh", r"[g]h200_tileA2_session.sh", r"[g]h200_decode_session.sh"):
        assert not re.search(other, "gh200_settleS1_session.sh")
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)$", vm, re.M).group(1).split()
    assert drivers[:5] == ["gh200_model_session.sh", "gh200_tile_session.sh", "gh200_decode_session.sh", "gh200_tileA2_session.sh",
                           "gh200_settleS1_session.sh"]
    assert "|gh200_settleS1_session.sh" in vm.splitlines()[5]


# --------------------------------------------------------------------------
# the session on the fake box
# --------------------------------------------------------------------------

#: A2's fake nvidia-smi, plus `-q -d POWER` in the card's real layout (GPU and module scopes) and
#: enforced.power.limit. After -pl the GPU-scope Current and Requested limits read the set value, except: the first
#: `pl_q_stale` -q reads after -pl read `pl_q_stale_w` (a stale read), `q_stuck_w` reads forever (A2's session: 700 W
#: from -q, 900 W from power.limit); enforced.power.limit reads `enforced_w` when given, else the set value.
SMI_S1 = TA2.SMI_A2.replace('''if args[:2] == ["-q", "-d"]:''', '''if args[:3] == ["-q", "-d", "POWER"]:
    w = 700.0
    if st.get("pl_set"):
        st["q_after_pl"] = st.get("q_after_pl", 0) + 1; save()
        w = st["pl"] if st["q_after_pl"] > SC.get("pl_q_stale", 0) else float(SC.get("pl_q_stale_w", 700.0))
        w = float(SC.get("q_stuck_w", w))
    print("==============NVSMI LOG==============\\nAttached GPUs : 1\\nGPU 00000000:DD:00.0")
    print("    GPU Power Readings\\n        Instantaneous Power Draw          : 86.96 W")
    print(f"        Current Power Limit               : {w:.2f} W\\n        Requested Power Limit             : {w:.2f} W")
    print("        Default Power Limit               : 900.00 W\\n        Max Power Limit                   : 900.00 W")
    print("    GPU Memory Power Readings \\n        Average Power Draw                : 0.00 W")
    print("    Module Power Readings\\n        Current Power Limit               : 1000.00 W")
    sys.exit(0)
if args[:2] == ["-q", "-d"]:''').replace('''st["pl"] = float(SC.get("pl_reads", args[args.index("-pl") + 1])); save(); sys.exit(0)''',
                                         '''st["pl"] = float(SC.get("pl_reads", args[args.index("-pl") + 1])); st["pl_set"] = True; save(); sys.exit(0)''').replace(
    '''        elif fld == "clocks.max.sm":''', '''        elif fld == "enforced.power.limit":
            e = float(SC.get("enforced_w", st["pl"])) if st.get("pl_set") else 700.0
            vals.append(SC["enforced_raw"] if "enforced_raw" in SC else f"{e:.2f}" + (" W" if units else ""))
        elif fld == "clocks.max.sm":''', 1)
assert "pl_q_stale" in SMI_S1 and 'st["pl_set"] = True' in SMI_S1 and "enforced_w" in SMI_S1


def fakepy_s1() -> str:
    body = TA2.fakepy_a2()
    old = 'if tool in ("dram_counter_route", "dram_counter_route_tile"):'
    assert old in body
    body = body.replace(old, 'if tool in ("dram_counter_route", "dram_counter_route_tile", "dram_counter_route_s1"):')
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_PP + "sys.exit(0)" + tail


FAKE_PP = r'''
if tool == "power_proof_s1":
    if "--dry-run" in args:
        print("POWER PROOF (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"batches": []}))
    time.sleep(1.5)   # the load runs while the background nvidia-smi -lms logs (a real GEMM runs 90 s)
    print("RESULT: 0 batches")
    sys.exit(SC.get("pp_rc", 0))
'''


PLAN_BOX = f"""- prelude lock-mhz=1410 power-limit-w=900 est=1 cap=5
- powerproof label=pp nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=kr {TILE} histogram={H}/olmoe-1b-7b-128g1-rp.json arms=native byte-groups=1 nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=ar {TILE} histogram={H}/olmoe-1b-7b-128g1-rp.json arms=native byte-groups=1 replay=application nodrop=1 est=1 cap=5
olmoe-1b-7b bytes label=f1005 lock-mhz=1005 {TILE} byte-groups=64 nodrop=1 est=1 cap=5
mixtral-8x7b bytes label=p128g16 {TILE} byte-groups=16 byte-treads=3,4,8 nodrop=1 est=1 cap=5
qwen2-57b-a14b bytes label=st {TILE} histogram={H}/qwen2-57b-a14b-128g1-n.json arms=native byte-groups=1 stall-metrics=1 drop-group=st est=1 cap=5
"""


def s1box(tmp_path, scenario, plan=PLAN_BOX, **env):
    box = TG.Box(tmp_path, {"retracted": 40, **scenario})
    TG._exe(box.bin / "fakepy", fakepy_s1())
    TG._exe(box.bin / "nvidia-smi", SMI_S1)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "s1.plan"
    f.write_text(plan)
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(**env), capture_output=True, text=True, timeout=600)
    return box, got


@pytest.fixture(scope="module")
def stale(tmp_path_factory):
    return s1box(tmp_path_factory.mktemp("s1stale"), {"pl_q_stale": 2})


def test_every_unit_reaches_its_tool_with_its_lock_treads_replay_and_metrics(stale):
    box, got = stale
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "powerproof", "bytes", "bytes", "bytes", "bytes", "bytes"]
    starts = re.findall(r"^\S+ bytes START: 3c.2 byte pages at the (\d+) lock.*\[unit \d+/\d+ \S+ (\w+)", led, re.M)
    assert [(lab, lock) for lock, lab in starts] == [("kr", "1410"), ("ar", "1410"), ("f1005", "1005"),
                                                     ("p128g16", "1410"), ("st", "1410")], starts
    pp = box.tool("power_proof_s1", dry=False)
    assert len(pp) == 1 and TG.val(pp[0], "--seconds") == "90"
    lms = [a for a in box.smi() if "-lms" in a]
    assert len(lms) == 1 and "enforced.power.limit" in lms[0][0] and "clocks_event_reasons.active" in lms[0][0]
    pp_at = next(i for i, a in enumerate(box.smi()) if "-lms" in a)
    assert ["-rgc"] in box.smi()[:pp_at] and not any("-lgc" in a for a in box.smi()[pp_at - 2:pp_at])
    tile = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    s1 = [a for a in box.tool("dram_counter_route_s1") if "--run" in a]
    assert [TG.val(a, "--out").split("-r3-counters")[0].rsplit("-", 1)[-1] for a in tile] == ["kr", "f1005", "p128g16"]
    assert [TG.val(a, "--out").split("-r3-counters")[0].rsplit("-", 1)[-1] for a in s1] == ["ar", "st"]
    kr, f1005, p16 = tile
    ar, st = s1
    assert TG.val(kr, "--page-lock-mhz") == "1410" and "--s1-replay" not in kr and "--s1-stall-metrics" not in kr
    assert TG.val(ar, "--page-lock-mhz") == "1410" and TG.val(ar, "--s1-replay") == "application"
    assert TG.val(ar, "--histogram") == TG.val(kr, "--histogram") == f"{H}/olmoe-1b-7b-128g1-rp.json"
    assert TG.val(ar, "--arms") == "native" and "--s1-stall-metrics" not in ar
    assert TG.val(f1005, "--page-lock-mhz") == "1005" and TG.val(f1005, "--tiles") == "1,2,3,4,5,6,7,8,9"
    assert TG.val(f1005, "--group-m") == "64" and TG.val(f1005, "--block-m") == "128"
    assert TG.val(p16, "--tiles") == "3,4,8" and TG.val(p16, "--group-m") == "16" and "--histogram" not in p16
    assert "--s1-stall-metrics" in st and "--s1-replay" not in st and TG.val(st, "--page-lock-mhz") == "1410"
    assert TG.val(st, "--num-warps") == "8" and TG.val(st, "--block-n") == "128"
    locks = (box.session / "gh200-driver" / "locks.env").read_text()
    assert "LOCK_PROBE_1005=held" in locks and "A2_LOCK" not in locks
    assert (box.session / "gh200-driver" / "a2-lock.env").read_text() == "A2_LOCK=1410\n"
    lgc = [a[a.index("-lgc") + 1] for a in box.smi() if "-lgc" in a]
    assert "1005,1005" in lgc and "1410,1410" in lgc
    assert not box.tool("tileA2_gate") and not box.tool("hw_rulers") and not box.tool("locked_r3")


def test_a_stale_first_power_reading_is_polled_until_both_readings_agree(stale):
    box, got = stale
    led = box.ledger()
    assert got.returncode == exit_codes.DONE
    q = [a for a in box.smi() if a[:3] == ["-q", "-d", "POWER"]]
    pl = [i for i, a in enumerate(box.smi()) if "-pl" in a]
    assert len(pl) == 1 and box.smi()[pl[0]] == ["-pl", "900", "-sc", "0"]
    after = [a for a in box.smi()[pl[0]:] if a[:3] == ["-q", "-d", "POWER"]]
    assert len(after) >= 3 and len(q) >= 4   # power-at-start, two stale reads, the agreeing one (and the power proof's)
    ps = (box.session / "power-set.txt").read_text()
    assert re.search(r"GPU Power Readings\n(.*\n)*?\s+Current Power Limit\s+: 900\.00 W", ps) and "700.00" not in ps
    assert ("the power limit read back after 4 s: enforced.power.limit 900.00 W, -q -d POWER GPU scope Current 900.00 W "
            "(Requested 900.00 W), power.limit 900.00 W") in led
    assert "900.00 W" in (box.session / "clocks-at-start.txt").read_text()
    assert not (box.session / "power-set-unsettled.txt").exists()
    assert (box.session / "gh200-driver" / "s1-power.env").read_text() == "S1_POWER_W=900\n"


@pytest.mark.parametrize("scenario, why", [
    ({"q_stuck_w": 700.0}, "enforced.power.limit reads '900.00' W and -q -d POWER GPU scope Current '700.00' W (Requested '700.00' W; power.limit '900.00' W)"),
    ({"enforced_w": 700.0}, "enforced.power.limit reads '700.00' W and -q -d POWER GPU scope Current '900.00' W (Requested '900.00' W; power.limit '900.00' W)"),
])
def test_a_refusal_at_900_w_reruns_the_session_at_700_w(tmp_path, scenario, why):
    """A2's session (-q 700 W, power.limit 900 W throughout), or an enforced limit under the request: the 900 W
    refusal is recorded, the same session sets 700 W, both readings agree there, and every unit runs."""
    box, got = s1box(tmp_path, scenario, MOE_DRIVER_PL_WAIT_S=6)
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + led
    assert why in led, led
    assert "the power limit never read back as 900 W on both readings within 6 s (S1); recorded" in led
    assert "the owner's fallback (2026-10-21): every unit runs at 700 W" in led
    assert "read back after 0 s: enforced.power.limit 700.00 W, -q -d POWER GPU scope Current 700.00 W" in led
    assert [a for a in box.smi() if "-pl" in a] == [["-pl", "900", "-sc", "0"], ["-pl", "700", "-sc", "0"]]
    env = (box.session / "gh200-driver" / "s1-power.env").read_text()
    assert env == "S1_POWER_900=refused\nS1_POWER_W=700\n"
    assert "700.00" in (box.session / "power-set-unsettled-900.txt").read_text() or scenario.get("enforced_w")
    assert "Current Power Limit               : 700.00 W" in (box.session / "power-set.txt").read_text()
    assert "700.00 W" in (box.session / "clocks-at-start.txt").read_text()
    assert "persistence Enabled, power.limit 700.00 W" in led
    assert re.findall(r"^\S+ (\w+) START", led, re.M)[:3] == ["prelude", "powerproof", "bytes"]
    assert not any("-sc" in a and a[a.index("-sc") + 1] not in ("0",) for a in box.smi())


@pytest.mark.parametrize("scenario", [{"enforced_w": 650.0}, {"enforced_raw": "[N/A]"},
                                      {"enforced_raw": 'Field "enforced.power.limit" is not a valid field to query.'}])
def test_a_refusal_at_900_and_at_700_w_stops_the_session(tmp_path, scenario):
    """An enforced limit below both, or one nvidia-smi does not return as a number: refused at 900 and 700 W, stop."""
    box, got = s1box(tmp_path, scenario, MOE_DRIVER_PL_WAIT_S=6)
    led = box.ledger()
    assert got.returncode == exit_codes.REFUSED, got.stdout[-2000:] + led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude"]
    assert "prelude REFUSED: the power limit never read back as 700 W on both readings within 6 s (S1, fallback): the session stops" in led
    assert (box.session / "gh200-driver" / "s1-power.env").read_text() == "S1_POWER_900=refused\nS1_POWER_700=refused\n"
    assert not (box.session / "power-set.txt").exists() and not (box.session / "clocks-at-start.txt").exists()
    assert (box.session / "power-set-unsettled-900.txt").exists() and (box.session / "power-set-unsettled.txt").exists()
    assert len([a for a in box.smi() if a[:3] == ["-q", "-d", "POWER"]]) == 1 + 4 + 4
    assert not box.tool("dram_counter_route_tile") and not box.tool("power_proof_s1")


def test_a_reading_that_agrees_at_once_writes_power_set_without_waiting(tmp_path):
    box, got = s1box(tmp_path, {}, plan="- prelude lock-mhz=1410 power-limit-w=900 est=1 cap=5\n"
                     f"olmoe-1b-7b bytes label=f1005 lock-mhz=1005 {TILE} byte-groups=64 est=1 cap=5\n")
    assert got.returncode == exit_codes.DONE, box.ledger()
    assert "read back after 0 s" in box.ledger()
    assert len([a for a in box.smi() if a[:3] == ["-q", "-d", "POWER"]]) == 2


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_settleS1_session.sh
# --------------------------------------------------------------------------

def test_start_with_s1_delivers_the_s1_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 5
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
    assert "gh200_settleS1_session.sh --dry-run" in cmds and "gh200_settleS1_session.sh --setup" in cmds
    assert "driver_script=gh200_settleS1_session.sh" in (lap.state / "r" / "run.env").read_text()


def test_the_pre_setup_dry_run_prints_s1_from_the_files_vm_run_copies(tmp_path):
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
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 14 units" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
