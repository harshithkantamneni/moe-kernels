"""Rental RR's registration (docs/registered/2026-11-11-rr-gh200.{json,txt}) held to its generator and to the files it
names: `register_rr.py --check` reads same; every earlier registration RR reads, imports or copies still reads same under
its own --check; the plan figures; the status (DESCRIPTIVE, no form, no gate, no blind); the forms stated as found
post-hoc on seen pages; the launcher's change flagged for the owner; every repo module the reader, the probe, the counter
route and the registration load pinned with its sha256; the pinned files byte-identical; no em-dash and no AI trailer in
any new file; and no __pycache__ left in the new directories."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "rr", REPO / "scripts" / "scoring" / "pc2",
          REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rrcommon as P  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
TXT = (REPO / "docs" / "registered" / f"{P.NAME}.txt").read_text()
NEW = [P.PROBE, P.DRIVER, P.LAUNCHER, P.COUNTER, "scripts/scoring/rr/rrcommon.py", "scripts/scoring/rr/gen_rr_pages.py",
       "scripts/scoring/rr/register_rr.py", "scripts/scoring/rr/read_rr.py", P.PLAN, P.SHAPES_FILE,
       f"docs/registered/{P.NAME}.json", f"docs/registered/{P.NAME}.txt", "tests/test_instr_probe_rr.py",
       "tests/test_rr_driver.py", "tests/test_read_rr.py", "tests/test_rr_registration.py", "tests/rr_fakes.py",
       "tests/test_dram_counter_route_rr.py", *(f"{P.PAGES_DIR}/{n}" for n in P.page_files())]
ENV = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def test_the_registration_reads_same_under_its_generator():
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/rr/register_rr.py"), "--check"],
                         capture_output=True, text=True, timeout=300, env=ENV)
    assert got.returncode == 0 and got.stdout.count(": same") == 4, got.stdout + got.stderr


@pytest.mark.parametrize("script, n", [("scripts/scoring/pc2/register_pc2.py", 4), ("scripts/scoring/pc/register_pc.py", 4),
                                       ("scripts/scoring/cal3/register_c3.py", 2),
                                       ("scripts/scoring/decode/register_decode_d4.py", 4),
                                       ("scripts/scoring/decode/register_decode_d5.py", 3)])
def test_every_earlier_registration_rr_leans_on_still_reads_same(script, n):
    got = subprocess.run([sys.executable, "-B", str(REPO / script), "--check"], capture_output=True, text=True,
                         timeout=600, env=ENV)
    assert got.returncode == 0 and "DIFFERS" not in got.stdout and got.stdout.count(": same") >= n, got.stdout + got.stderr


def test_the_plan_figures_and_the_cut_order_are_the_registered_ones():
    r = REG["plan"]
    assert r == {**r, **P.rental()} and r["units_count"] == len(P.UNITS) == 10
    assert (r["units_min"], r["nodrop_min"], r["wall_min"], r["cost_usd"]) == (43, 35, 68, 2.6)
    assert (r["deadline_after_launch_min"], r["cost_at_deadline_usd"], r["hard_cap_after_launch_min"],
            r["cost_at_hard_cap_usd"]) == (85, 3.24, 100, 3.82)
    assert (r["nodrop_wall_min"], r["nodrop_cost_usd"]) == (60, 2.29)
    assert r["drop_order_last_first"] == list(P.CUT_ORDER) == ["rep"] and r["sha256"] == sha(P.PLAN)
    assert "wall about 68 min with 25 min of boot and setup: $2.60 at $2.29/h; deadline launch + 85 min ($3.24); " \
           "guardian hard cap launch + 100 min ($3.82 at most)" in TXT
    stop = REG["stop rules"]["the deadline and the upgrade step's budget (review MEDIUM 3)"]
    assert "ONE total budget of 600 s" in stop and "a slow lock cannot skip a nodrop unit" in stop
    assert r["deadline_after_launch_min"] - 48 == 37 and "(37 min at the registered deadline)" in stop


def test_the_status_registers_no_form_no_gate_no_blind_and_states_the_post_hoc_origin():
    assert "MEASUREMENT, DESCRIPTIVE" in REG["status"]
    for w in ("no model form", "no gate", "no blind"):
        assert w in REG["status"], w
    rd = REG["readings"]
    assert rd["every reading"].startswith("DESCRIPTIVE") and "NOT IDENTIFIED" in rd["every reading"]
    assert "FOUND POST-HOC ON SEEN PAGES" in rd["STATED PLAINLY"] and "FRESH synthetic cells" in rd["STATED PLAINLY"]
    for k in ("RR-R0", "RR-R1", "RR-R2", "RR-R3", "RR-R4", "STAND-IN"):
        assert any(x.startswith(k) for x in rd), k
    forms = REG["forms (registered; constants POST-HOC on SEEN pages)"]
    assert [v["C_mib"] for v in forms.values()] == [42.5, 45.0, 30.0, 62.5]
    r3 = rd[next(k for k in rd if k.startswith("RR-R3"))]
    assert r3.startswith("NUMBERS ONLY") and "No label and no band" in r3 and "AS FOUND" in r3   # the disclosure only
    assert "set with PC's published pages in view" in r3 and "removed on the owner's decision" in r3
    assert "as_found_band" not in REG["constants"] and REG["stand-in wave numbers (SEEN pages, frozen)"] == P.STANDIN_WAVE
    r4 = rd[next(k for k in rd if k.startswith("RR-R4"))]
    assert "RR-R1's rules" in r4 and "no read-back after its tool" in r4 and "An invalid unit reads no number" in r4
    od = REG["owner_decisions (2026-11-11)"]
    assert "NO blind" in od[0] and "no GPU kernel is written or edited" in od[3]
    assert od[4:] == ["the apt-lock launcher fix is ADOPTED (vm_run_rr.sh's driver-upgrade step)",
                      "the Phi-3.5 128g1 replicate is KEPT (droppable, cut first)",
                      "the wave-start spread reading is NUMBERS ONLY: no label and no band"]


def test_the_build_decisions_flag_the_launcher_and_name_what_was_not_built():
    bd = " ".join(REG["build decisions (to design-P1's sketch, before any page)"])
    assert "ADOPTED BY THE OWNER" in bd and "unattended-upgrade" in bd and "ONE budget of 600 s" in bd
    assert "THE BOUND KILLS DPKG" in bd and "RECOVERY" in bd and "dpkg --force-confdef --force-confold --configure -a" in bd
    assert "not built" in bd and "align slope by numel" in bd and "L-TABLE" in bd
    assert "the replicate" in bd and "$2.29" in bd


def test_every_repo_module_the_reader_the_probe_and_the_registration_load_is_pinned():
    code = ("import sys, json; sys.path[:0] = ['scripts', 'scripts/scoring/rr', 'scripts/scoring/pc2', "
            "'scripts/scoring/pc', 'tests']\n"
            "import instr_probe_rr, read_rr, register_rr, gen_rr_pages, rrcommon as P, dram_counter_route_rr\n"
            "import dram_counter_route\n"
            "P.check_page_doc(P.page_doc('rr-128'), P.load_shapes())\n"
            "from pathlib import Path\n"
            "root = Path('.').resolve()\n"
            "out = sorted({str(Path(m.__file__).resolve().relative_to(root)) for m in list(sys.modules.values())\n"
            "              if isinstance(getattr(m, '__file__', None), str) and Path(m.__file__).is_absolute()\n"
            "              and str(Path(m.__file__).resolve()).startswith(str(root) + '/')\n"
            "              and not any(x in Path(m.__file__).resolve().relative_to(root).parts\n"
            "                          for x in ('.venv', 'venv', 'site-packages', 'tests'))})\n"
            "print(json.dumps(out))\n")
    got = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert got.returncode == 0, got.stderr
    loaded = json.loads(got.stdout.strip().splitlines()[-1])
    missing = [m for m in loaded if m not in REG["code_pins"]]
    assert not missing, missing
    for rel, h in REG["code_pins"].items():
        assert sha(rel) == h, rel


def test_every_pinned_file_of_pc2_is_byte_identical_and_pinned_again_here():
    pc2 = json.loads((REPO / "docs/registered/2026-11-10-pc2-gh200.json").read_text())["code_pins"]
    for rel, h in pc2.items():
        assert sha(rel) == h == REG["code_pins"][rel], rel
    for e in REG["earlier registrations kept byte-identical"]:
        doc = json.loads((REPO / f"docs/registered/{e}.json").read_text())
        for rel, h in (doc.get("code_pins") or {}).items():
            if (REPO / rel).is_file() and rel in REG["code_pins"]:
                assert sha(rel) == h, (e, rel)


def test_no_new_file_carries_an_em_dash_or_an_ai_trailer_and_no_bytecode_is_left():
    for rel in NEW:
        t = (REPO / rel).read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, rel
        assert "Co-Authored" + "-By" not in t and "Generated" + " with" not in t, rel
        assert "Cla" + "ude" not in t, rel
    assert not list((REPO / "scripts" / "scoring" / "rr").glob("**/__pycache__"))


def test_the_seen_block_states_the_guard_and_the_loaded_models():
    s = REG["seen"]
    assert s["reserved (owner decision, 2026-11-06, kept)"]["models"] == list(P.RESERVED)
    assert s["loaded by name (both SEEN)"] == ["mixtral-8x7b", "phi-3.5-moe"]
    assert "never a refusal" in next(v for k, v in s.items() if k.startswith("blind candidates"))
    assert REG["census and occupancy"]["registered occupancy"] == P.OCCUPANCY == {"128x128x64w8s3": 2,
                                                                                   "64x128x64w8s3": 3}
