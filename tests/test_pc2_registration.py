"""Rental PC2's registration (docs/registered/2026-11-10-pc2-gh200.{json,txt}) held to its generator and to the files it
names: `register_pc2.py --check` reads same; every earlier registration PC2 reads, imports or copies still reads same
under its own --check (B3's is held by tests/test_tileB3_reader.py); the plan figures; the status (DESCRIPTIVE, no form,
no constant); how each of v6d's open terms will be read, the naive-path disclosure stated plainly; every repo module the
reader, the probe and the registration load pinned with its sha256; the pinned files byte-identical; no em-dash and no
AI trailer in any new file; and no __pycache__ left in the new directories."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc2", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pc2common as P  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
TXT = (REPO / "docs" / "registered" / f"{P.NAME}.txt").read_text()
NEW = [P.PROBE, P.DRIVER, P.LAUNCHER, "scripts/scoring/pc2/pc2common.py", "scripts/scoring/pc2/gen_pc2_pages.py",
       "scripts/scoring/pc2/register_pc2.py", "scripts/scoring/pc2/read_pc2.py", P.PLAN, P.SHAPES_FILE,
       f"docs/registered/{P.NAME}.json", f"docs/registered/{P.NAME}.txt", "tests/test_instr_probe_pc2.py",
       "tests/test_pc2_driver.py", "tests/test_read_pc2.py", "tests/test_pc2_registration.py", "tests/pc2_fakes.py",
       *(f"{P.PAGES_DIR}/{n}" for n in P.page_files())]
ENV = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def test_the_registration_reads_same_under_its_generator():
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/pc2/register_pc2.py"), "--check"],
                         capture_output=True, text=True, timeout=300, env=ENV)
    assert got.returncode == 0 and got.stdout.count(": same") == 4, got.stdout + got.stderr


@pytest.mark.parametrize("script, n", [("scripts/scoring/pc/register_pc.py", 4), ("scripts/scoring/cal3/register_c3.py", 2),
                                       ("scripts/scoring/decode/register_decode_d4.py", 4),
                                       ("scripts/scoring/decode/register_decode_d5.py", 3)])
def test_every_earlier_registration_pc2_leans_on_still_reads_same(script, n):
    got = subprocess.run([sys.executable, "-B", str(REPO / script), "--check"], capture_output=True, text=True,
                         timeout=600, env=ENV)
    assert got.returncode == 0 and "DIFFERS" not in got.stdout and got.stdout.count(": same") >= n, got.stdout + got.stderr


def test_the_plan_figures_and_the_cut_order_are_the_registered_ones():
    r = REG["plan"]
    assert r == {**r, **P.rental()} and r["units_count"] == len(P.UNITS) == 10
    assert (r["units_min"], r["nodrop_min"], r["wall_min"], r["cost_usd"]) == (35, 27, 60, 2.29)
    assert (r["deadline_after_launch_min"], r["cost_at_deadline_usd"], r["hard_cap_after_launch_min"],
            r["cost_at_hard_cap_usd"]) == (75, 2.86, 90, 3.44)
    assert r["drop_order_last_first"] == list(P.CUT_ORDER) == ["rep", "d3", "w"] and r["sha256"] == sha(P.PLAN)
    assert "wall about 60 min with 25 min of boot and setup: $2.29 at $2.29/h; deadline launch + 75 min ($2.86); " \
           "guardian hard cap launch + 90 min ($3.44 at most)" in TXT


def test_the_status_registers_no_form_no_constant_and_every_reading_is_descriptive():
    assert "MEASUREMENT, DESCRIPTIVE" in REG["status"] and "no model form" in REG["status"] and "no constant" in REG["status"]
    assert REG["readings"]["every reading"].startswith("DESCRIPTIVE") and "NOT IDENTIFIED" in REG["readings"]["every reading"]
    od = REG["owner_decisions (2026-11-10)"]
    assert od[:3] == ["run PC2 before building v6d's addendum 6", "v6d stays as frozen, with the duty factor",
                      "Granite-3.0-1B stays in the blind (PC2 reaches its regime with synthetic shapes, never the model)"]
    assert "no GPU kernel is written or edited" in od[3]


def test_each_v6d_open_term_has_its_reading_and_the_naive_term_is_disclosed_plainly():
    t = REG["v6d terms (how each open term will be read off PC2's pages)"]
    keys = [k for k in t if k[0].isdigit()]
    assert [k[0] for k in keys] == ["1", "2", "3", "4"]
    assert "16x64x128" in keys[0] and "short slabs" in keys[1] and "big-single-wave" in keys[2] and "naive" in keys[3]
    for k, ref in zip(keys, ("PC2-R1", "PC2-R2", "PC2-R3", "PC2-R4"), strict=True):
        assert ref in t[k], k
    n = t[keys[3]]
    assert "WITH THE SEEN DECODE BLIND IN VIEW" in n and "DISCLOSED AS SUCH" in n and "never a blind test" in n
    assert "no form and no value" in t["status"] and "v6d stays as frozen" in t["status"]
    for k in ("PC2-R1", "PC2-R2", "PC2-R3", "PC2-R4"):
        assert any(x.startswith(k) for x in REG["readings"]), k
    assert "never used by a reading" in "".join(t)


def test_the_build_decisions_name_the_limits_and_what_was_not_built():
    bd = " ".join(REG["build decisions (to design-D6's sketch, before any page)"])
    assert "(3 = 3, both GEMMs) and s3 (5 = 5)" in bd and "w2: 8 against 7" in bd
    assert "is NOT BUILT" in bd and "T k >= 1024" in bd and "caps T at 128" in bd
    assert "no per-CTA life, no c0 split, no kappa" in bd
    assert "vm_run_pc2.sh" in bd and "gemm_guard" in bd


def test_every_repo_module_the_reader_the_probe_and_the_registration_load_is_pinned():
    code = ("import sys, json; sys.path[:0] = ['scripts', 'scripts/scoring/pc2', 'scripts/scoring/pc', 'tests']\n"
            "import instr_probe_pc2, read_pc2, register_pc2, gen_pc2_pages, pc2common as P\n"
            "P.check_page_doc(P.page_doc('pc2-n'), P.load_shapes())\n"
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


def test_every_pinned_file_of_pc_is_byte_identical_and_pinned_again_here():
    pc = json.loads((REPO / "docs/registered/2026-11-06-pc-gh200.json").read_text())["code_pins"]
    for rel, h in pc.items():
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
    assert not list((REPO / "scripts" / "scoring" / "pc2").glob("**/__pycache__"))


def test_the_seen_block_states_the_guard_and_the_reserved_models():
    s = REG["seen"]
    assert s["reserved (owner decision, 2026-11-06, kept 2026-11-10)"]["models"] == list(P.RESERVED)
    assert "either GEMM of a reserved model" in s["reserved (owner decision, 2026-11-06, kept 2026-11-10)"]["rule"]
    assert "never its shape and never its GEMMs" in s["Granite-1B's regime"]
    assert REG["census and occupancy"]["registered occupancy"] == P.OCCUPANCY
