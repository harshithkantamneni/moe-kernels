"""Rental PC's registration (docs/registered/2026-11-06-pc-gh200.{json,txt}) held to its generator and to the files it
names: `register_pc.py --check` reads same; the plan figures; H1's bands; the by-value Phi inputs against their
published sources; the Phi histogram pages against B3's registration; every repo module the reader and the probe load
pinned; the frozen 'unseen at prefill tiles' literal recomputed over the sessions published by 2026-11-05; the reader's
freeze refusals; and no em-dash in any new file."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pccommon as P  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
NEW = ["scripts/instr_probe_pc.py", P.DRIVER, "scripts/scoring/pc/pccommon.py", "scripts/scoring/pc/gen_pc_pages.py",
       "scripts/scoring/pc/register_pc.py", "scripts/scoring/pc/read_pc.py", P.PLAN, P.SHAPES_FILE, P.PRICED,
       f"docs/registered/{P.NAME}.json", f"docs/registered/{P.NAME}.txt", "tests/test_instr_probe_pc.py",
       "tests/test_pc_driver.py", "tests/test_read_pc.py", "tests/test_pc_registration.py", "tests/pc_fakes.py"]


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def test_the_registration_reads_same_under_its_generator():
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/pc/register_pc.py"), "--check"],
                         capture_output=True, text=True, timeout=300, env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
    assert got.returncode == 0 and got.stdout.count(": same") == 4, got.stdout + got.stderr


def test_the_plan_figures_and_the_cut_order_are_the_registered_ones():
    r = REG["plan"]
    assert r == {**r, **P.rental()} and r["units_count"] == len(P.UNITS) == 20
    assert (r["units_min"], r["nodrop_min"], r["wall_min"], r["cost_usd"]) == (84, 30, 109, 4.16)
    assert (r["deadline_after_launch_min"], r["cost_at_deadline_usd"], r["hard_cap_after_launch_min"],
            r["cost_at_hard_cap_usd"]) == (125, 4.77, 140, 5.34)
    assert r["drop_order_last_first"] == list(P.CUT_ORDER) == ["rep", "b16", "cnt", "d16", "phi"]
    assert r["sha256"] == sha(P.PLAN)
    text = (REPO / f"docs/registered/{P.NAME}.txt").read_text()
    assert "wall about 109 min with 25 min of boot and setup: $4.16 at $2.29/h; deadline launch + 125 min ($4.77); " \
           "guardian hard cap launch + 140 min ($5.34 at most)" in text
    assert "separate Phi 64g1 step is gone" in text


def test_h1s_bands_are_registered_before_any_page_and_are_the_readers():
    assert REG["H1"]["bands"] == json.loads(json.dumps(P.H1))
    for k in ("FALSIFIED when any read falsifier holds", "SUPPORTED when every band holds", "otherwise", "label"):
        assert REG["H1"][k]
    assert "owner decision 4" in REG["H1"]["label"] and "DESCRIPTIVE" in REG["status"]
    assert "no model form" in REG["status"] and "no constant" in REG["status"]


def test_the_phi_inputs_are_the_published_numbers_by_value():
    doc = json.loads((REPO / P.PRICED).read_text())
    assert REG["priced"]["sha256"] == sha(P.PRICED) and len(doc["rows"]) == 18
    for rel, h in doc["sources_sha256"].items():
        if not rel.startswith("~"):
            assert sha(rel) == h, rel
    pages = {"128g1": [k for k in doc["sources_sha256"] if "b128g1" in k][0],
             "64g1": [k for k in doc["sources_sha256"] if "b64g1" in k][0]}
    for cls, rel in pages.items():
        page = json.loads((REPO / rel).read_text())
        got = {(int(c["n"]), str(c.get("histogram") or "balanced")): c["per_gemm"]["w1"]["dram_bytes_read"]
               for c in page["cells"] if c["arm"] == "native"}
        for r in doc["rows"]:
            if r["cls"] == cls:
                assert r["counted_bytes"]["w1"] == got[(r["n"], r["label"])], (cls, r["label"])
    pred = {(c["page"], c["n"], c["label"]): c["ms"] for c in
            json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())["cells"]}
    for r in doc["rows"]:
        assert r["call_pred_ms"] == pred[(r["page"], r["n"], r["label"])]


def test_the_phi_pages_are_b3s_registered_histograms():
    b3 = json.loads((REPO / "docs/registered/2026-11-04-tileB3-gh200.json").read_text())
    blob = json.dumps(b3)
    for rel in (P.PHI_128, P.PHI_64, P.PHI_128G16):
        assert REG["pages"][rel]["sha256"] == sha(rel) and sha(rel) in blob, rel


def test_every_repo_module_the_reader_and_the_probe_load_is_pinned():
    code = ("import sys, json; sys.path[:0] = ['scripts', 'scripts/scoring/pc', 'tests']\n"
            "import instr_probe_pc, read_pc, register_pc, gen_pc_pages, pccommon as P\n"
            "P.check_page_doc(P.page_doc('pc-a'), P.load_shapes())\n"
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


def test_the_unseen_at_prefill_literal_recomputes_over_the_sessions_published_by_2026_11_05():
    from moe.spec import MODEL_CONFIGS
    pub = sorted(p for p in (REPO / "results/published").iterdir() if p.is_dir())
    pre = [p for p in pub if re.search(r"(tile|cal[234]|settleS1)", p.name) and "decode" not in p.name
           and p.name[:10] <= "2026-11-05"]
    assert len(pre) == 7
    seen = set()
    for p in pre:
        for d in (p / "results").rglob("*"):
            if d.is_dir():
                seen |= {m for m in MODEL_CONFIGS if re.search(rf"(^|-){re.escape(m)}(-|$)", d.name)}
    assert sorted(set(MODEL_CONFIGS) - seen - {"toy"}) == sorted(P.UNSEEN_AT_PREFILL_6C800F84)
    assert REG["seen"]["still unseen at vLLM's prefill tiles (COMPUTED at 6c800f84, FROZEN)"] == \
        list(P.UNSEEN_AT_PREFILL_6C800F84)


def test_the_reader_refuses_a_tree_whose_commit_is_not_on_its_ledger_or_is_not_this_tree(tmp_path):
    import pc_fakes as F
    import read_pc as R
    pc = F.tree(tmp_path)
    with pytest.raises(SystemExit, match="the rental's commit is not on its ledger"):
        R.read(pc)
    (pc / "session" / "gh200-driver" / "status").write_text(
        "2026-11-06T00:00:00Z driver start: commit 6c800f845af59e50e118083760cd28bb5a548cea, steps of the plan x\n")
    with pytest.raises(SystemExit, match="the working tree is not the rental's commit"):
        R.read(pc)


def test_no_new_file_carries_an_em_dash_or_an_ai_trailer():
    for rel in NEW:
        t = (REPO / rel).read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, rel
        assert "Co-Authored" + "-By" not in t and "Generated" + " with" not in t, rel
