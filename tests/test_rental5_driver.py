"""Rental 5's driver scaffolding off the GPU (scripts/plans/rental5-2026-10.plan): the regcheck
step, histogram= and arms= on timed and bytes units, block-k / num-stages on timed units,
instr-variants with the first-passing choice, and depends=. The units run on
test_gh200_model_session's FAKE BOX, its interpreter extended with a fake regcheck."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_rental4_scaffolding as T4  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.bench import exit_codes  # noqa: E402

PLAN5 = REPO / "scripts" / "plans" / "rental5-2026-10.plan"
DRIVER = TG.DRIVER
H = "docs/registered/2026-10-07-rental5-skew-hist"


def dry(*args):
    return T4.dry(*args)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def test_the_rental5_plan_parses_in_the_review_order_and_fits_three_hours():
    got = dry("--plan", str(PLAN5))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "a PLAN of 22 units" in got.stdout
    units = TG._units(got.stdout)
    steps = [s for _, _m, s, *_r in units]
    assert steps[:3] == ["prelude", "calibrate", "regcheck"], steps   # regcheck before any timed unit
    assert steps[3:12] == ["timed", "timed", "bytes"] * 2 + ["timed", "timed", "timed"]
    assert steps[12:18] == ["perturb"] + ["stamps"] * 5
    assert steps[18:20] == ["bytes", "bytes"] and steps[20:] == ["timed"] * 2
    total = sum(int(e) for _, _m, _s, e, *_r in units)
    assert f"{total} min of units" in got.stdout and total == 152 and total <= 180
    tags = [m.group(1) if (m := re.search(r"\[drop-group ([\w-]+)\]", u[5])) else None for u in units]
    assert tags[2:12] == ["rc", "skm", "skm", "skm", "sko", "sko", "sko", "skq", "skq", "c15"]
    assert tags[12:18] == ["st", "st", "stt", "stt", "std", "std"]
    # the bottom of the file drops first: BK 128, then second K (tp2 CUT by the owner)
    assert tags[18:] == ["k2", "k2", "bk", "bk"] and "mixtral-8x7b-tp2" not in got.stdout
    deps = [m.group(1) if (m := re.search(r"\[depends ([\w-]+)\]", u[5])) else None for u in units]
    assert deps[12] == "rc" and deps[14:18] == ["st"] * 4
    dirs = [TG._unit_dir(u[5]) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)
    for u in units:
        if "histogram=" in u[5]:
            assert u[2] in ("timed", "bytes")
            f = re.search(r"histogram=(\S+)", u[5]).group(1)
            assert (REPO / f).is_file() and f.startswith(H)
        m = re.search(r"(?<![\w-])arms=(\S+)", u[5])
        if m:
            assert "histogram=" in u[5] and "private" not in m.group(1)
    assert "instr-stamps" not in PLAN5.read_text() and "pid mod 17" in PLAN5.read_text()


def test_every_histogram_the_plan_names_is_registered():
    reg = json.loads((REPO / "docs" / "registered" / "2026-10-07-rental5-skew-gh200.json").read_text())
    named = set(re.findall(r"histogram=(\S+)", PLAN5.read_text()))
    assert named and {Path(n).name for n in named} <= set(reg["histograms"]["files"])


def test_the_drivers_variant_texts_are_the_registered_ones():
    body = DRIVER.read_text()
    got = dict(re.findall(r'^    (v[123])\) echo "(stamps=[^"]+)" ;;$', body, re.M))
    assert got == I.R5_VARIANTS
    for v, t in I.R5_VARIANTS.items():
        assert I.parse_spec(t).text() and "mod=16" not in t
        assert ("_mod=17" in t) == (v != "v1")


def test_the_plans_variants_expand_in_preference_order_and_pass_the_probes_checks(tmp_path):
    got = subprocess.run(["bash", str(DRIVER), "--print-variants", "--plan", str(PLAN5)], capture_output=True,
                         text=True, timeout=120)
    assert got.returncode == exit_codes.REFUSED, got.stderr
    vs = json.loads(got.stdout)
    assert [v["id"] for v in vs] == ["u14-stf-v1", "u14-stf-v2", "u14-stf-v3", "u15-stt8x22-v1", "u15-stt8x22-v2",
                                     "u16-stt-v1", "u16-stt-v2", "u17-std4-v2", "u17-std4-v3", "u18-std2-v2",
                                     "u18-std2-v3"]
    assert all(v["spec"] == I.R5_VARIANTS[v["id"].rsplit("-", 1)[1]] for v in vs)
    f = tmp_path / "v.json"
    f.write_text(got.stdout)
    loaded = IP.load_variants(f)
    assert {v["id"]: v["num_stages"] for v in loaded}["u18-std2-v3"] == 8


@pytest.mark.parametrize("body, why", [
    (f"- prelude\nolmoe-1b-7b deep label=a histogram={H}/olmoe-1b-7b-A.json\n", "histogram belongs to a timed or bytes unit"),
    ("- prelude\nolmoe-1b-7b timed label=a arms=native\n", "arms needs histogram="),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H}/olmoe-1b-7b-A.json arms=native,private declared-copies=9\n",
     "PRIVATE cannot run a histogram page"),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H}/olmoe-1b-7b-A.json\n", "needs declared-copies"),
    ("- prelude\nolmoe-1b-7b timed label=a histogram=docs/none.json declared-copies=9\n", "no such file"),
    ("- prelude\nolmoe-1b-7b timed label=a histogram=/etc/x.json declared-copies=9\n", "a repo-relative .json path"),
    ("- prelude\nolmoe-1b-7b timed label=a\n- regcheck label=rc instr-variants=v1\n", "regcheck runs before any timed or deep unit"),
    ("- prelude\n- perturb\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-treads=2 stamp-arms=native instr-variants=v1\n",
     "needs a regcheck unit before it"),
    ("- prelude\n- regcheck label=rc\n- perturb\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-treads=2 stamp-arms=native instr-variants=v4\n",
     "instr-variants v4"),
    ("- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 instr-variants=v1\n", "instr-variants belongs to a regcheck, perturb or stamps unit"),
    ("- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 depends=zz\n", "depends=zz names no drop-group of an earlier unit"),
    ("- prelude\nolmoe-1b-7b deep label=a block-k=32\n", "block-k belongs to a bytes unit (or a stamps or timed unit), not deep"),
    ("- prelude\n- regcheck label=rc\nolmoe-1b-7b timed label=a\n", "a regcheck unit and no instrumented unit"),
])
def test_rental5_keys_are_refused_off_their_step(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# the units on the fake box
# --------------------------------------------------------------------------

FAKE_REGCHECK = r'''
if tool == "instr_probe" and val("--mode") == "regcheck" and "--dry-run" not in args:
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    vs = json.loads(Path(val("--variants")).read_text())
    bad = SC.get("regcheck_fail", [])
    (out / "regcheck.env").write_text("".join(
        f"REGCHECK_{re.sub('[^A-Za-z0-9]', '_', v['id'])}={'FAIL 5 CTAs/SM against 4' if v['id'] in bad else 'PASS'}\n"
        for v in vs))
    (out / "regcheck.json").write_text("{}")
    sys.exit(1 if bad else 0)
'''


def fakepy5() -> str:
    body = T4.fakepy4()
    old = 'T, G = int(val("--treads")), val("--group-m")'
    assert old in body
    body = body.replace(old, 'T = int(val("--treads")) if val("--treads") else max(int(c["n"]) for c in '
                             'json.load(open(val("--histogram")))["cells"]); G = val("--group-m")')
    head, tail = body.split('if tool == "instr_probe":', 1)
    return head + FAKE_REGCHECK + 'if tool == "instr_probe":' + tail


def box5(tmp_path, body, scenario=None, **kw):
    box = TG.Box(tmp_path, scenario or {})
    TG._exe(box.bin / "fakepy", fakepy5())
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "test.plan"
    f.write_text(body)
    return box, box.run("--plan", str(f), **kw)


PLAN_SMALL = f"""- prelude
mixtral-8x7b calibrate label=r5 est=2 cap=5
- regcheck label=rc instr-variants=v1,v2,v3 drop-group=rc est=1 cap=5
olmoe-1b-7b timed label=ska histogram={H}/olmoe-1b-7b-A.json arms=native,shared declared-copies=9 timed-groups=8 est=2 cap=5
olmoe-1b-7b bytes label=skc histogram={H}/olmoe-1b-7b-C.json arms=native,shared byte-groups=8 est=2 cap=5
olmoe-1b-7b timed label=bk128 histogram={H}/olmoe-1b-7b-bk.json arms=native declared-copies=9 timed-groups=64 block-k=128 num-stages=4 est=2 cap=5
- perturb label=pt instr-variants=v1,v2,v3 drop-group=st est=2 cap=5
olmoe-1b-7b stamps label=std4 stamp-groups=8 stamp-treads=2,4 stamp-arms=native,shared instr-variants=v1,v2,v3 drop-group=st est=2 cap=5
"""


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return box5(tmp_path_factory.mktemp("r5"), PLAN_SMALL, {"regcheck_fail": ["u8-std4-v1"], "retracted": 40})


def test_the_units_run_and_the_stamps_unit_takes_the_first_variant_passing_both_gates(small):
    box, got = small
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout + got.stderr + led
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert starts == ["prelude", "calibrate", "regcheck", "timed", "bytes", "timed", "perturb", "stamps"]
    assert "stamps: u8-std4-v2 chosen (the first of u8-std4-v1 u8-std4-v2 u8-std4-v3 passing regcheck and perturb)" in led
    st = [a for a in box.tool("instr_probe", dry=False) if TG.val(a, "--mode") == "stamps"]
    assert len(st) == 1 and TG.val(st[0], "--variant") == "u8-std4-v2" and "--regcheck" in st[0]
    chosen = list(box.results.rglob("CHOSEN_VARIANT.txt"))
    assert len(chosen) == 1 and chosen[0].read_text() == f"id=u8-std4-v2\nspec={I.R5_VARIANTS['v2']}\n"
    # regcheck is compile only: no lock is set around it
    rc = [a for a in box.tool("instr_probe", dry=False) if TG.val(a, "--mode") == "regcheck"]
    assert len(rc) == 1


def test_histogram_units_hand_r3_and_the_counter_route_their_page_and_arms(small):
    box, _got = small
    lr = box.tool("locked_r3")
    ska = [a for a in lr if f"{H}/olmoe-1b-7b-A.json" in a]
    assert len(ska) == 1 and TG.val(ska[0], "--arms") == "native,shared" and "--treads" not in ska[0]
    assert TG.val(ska[0], "--declared-copies") == "9"
    bk = [a for a in lr if f"{H}/olmoe-1b-7b-bk.json" in a][0]
    assert TG.val(bk, "--block-k") == "128" and TG.val(bk, "--num-stages") == "4" and TG.val(bk, "--arms") == "native"
    by = [a for a in box.tool("dram_counter_route") if "--histogram" in a]
    assert by and all(TG.val(a, "--histogram") == f"{H}/olmoe-1b-7b-C.json" and TG.val(a, "--arms") == "native,shared"
                      for a in by)


def test_no_variant_passing_both_gates_refuses_the_stamps_unit(tmp_path):
    body = PLAN_SMALL.replace("instr-variants=v1,v2,v3 drop-group=st est=2", "instr-variants=v1 drop-group=st est=2")
    box, got = box5(tmp_path, body, {"regcheck_fail": ["u8-std4-v1"], "retracted": 40})
    assert "stamps REFUSED: none of u8-std4-v1 reads PASS on both its regcheck and perturb lines" in box.ledger()
    assert not [a for a in box.tool("instr_probe", dry=False) if TG.val(a, "--mode") == "stamps"]


def test_a_unit_that_depends_on_a_dropped_group_goes_with_it(tmp_path):
    body = """- prelude
olmoe-1b-7b bytes label=a byte-groups=1 est=10 cap=20
mixtral-8x7b bytes label=g1a byte-groups=1 drop-group=st est=4 cap=20
mixtral-8x7b bytes label=g2 byte-groups=1 drop-group=stt depends=st est=1 cap=20
mixtral-8x7b bytes label=g1b byte-groups=2 drop-group=st est=4 cap=20
"""
    now = 1_900_000_000
    # 25 min less the 8-minute reserve: 17 against 22 planned. Unit 5 goes, its drop-group st
    # takes unit 3, and unit 4 depends on st, so it goes too: 13 left fits
    box, _got = TG._plan_box(tmp_path, body, deadline=now + 25 * 60, MOE_DRIVER_NOW=now)
    led = box.ledger()
    assert re.findall(r"DROPPED unit (\d+)/", led) == ["5", "3", "4"], led
    assert "drop-group st goes with unit 5" in led and "it depends on drop-group st" in led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "bytes"]
