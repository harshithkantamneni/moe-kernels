"""Rental RR's probe and rules off the GPU (scripts/instr_probe_rr.py, scripts/scoring/rr/rrcommon.py): the probe is PC2's
pinned probe plus lines marked RR; the shapes and pages are generated and registered; every cell's geometry is the hand
value and r3_decode's readback; every refusal of a shape, a cell, a page and a variant; the skew layouts recomputed from
the published pages they cite; the axis coverage (W_c / P, m_e, waves, F_stag across the 42.5 MiB step, series B's 2 MiB
slab, series C's E 64); the reserved-model guard and the blind candidates' GEMMs avoided; a cell built on the CPU; and the
pinned files the probe imports left byte-identical."""
from __future__ import annotations

import argparse
import difflib
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "rr", REPO / "scripts" / "scoring" / "pc2",
          REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import instr_probe_rr as PR  # noqa: E402
import pc2common as PC2C  # noqa: E402
import r3_decode as RD  # noqa: E402
import rrcommon as P  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS, MoEConfig  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
PC2_REG = json.loads((REPO / "docs" / "registered" / f"{PC2C.NAME}.json").read_text())
PUB = REPO / "results" / "published"


def shapes():
    return P.load_shapes()


def cells(page):
    return {c["label"]: c for c in P.check_page_doc(P.page_doc(page), shapes())}


def cell(**kw) -> dict:
    c = {"label": "X", "shape": "rrs-e8-h4096-p20", "T": 256, "counts": [256, 256, 0, 0, 0, 0, 0, 0],
         "config": P.config_of(P.T128), "group_m": 1, "flush": True, "read": "w1", "path": "aligned"}
    c.update(kw)
    return c


# --------------------------------------------------------------------------
# the probe is PC2's probe plus lines marked RR
# --------------------------------------------------------------------------

def test_the_probe_is_pc2s_probe_plus_lines_marked_rr():
    a = (REPO / P.PARENT_PROBE).read_text().splitlines()
    b = (REPO / P.PROBE).read_text().splitlines()
    start = b.index("from __future__ import annotations")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        assert tag in ("replace", "insert") or j1 < start, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < start or "RR" in b[j], (j + 1, b[j])
    changed = [x for x in b[start:] if "RR" in x and x not in a]
    assert len(changed) == 6 and any("import rrcommon as PCC" in x for x in changed)
    assert sum('"tool": "scripts/instr_probe_rr.py --mode' in x for x in changed) == 3
    assert PC2_REG["code_pins"][P.PARENT_PROBE] == hashlib.sha256((REPO / P.PARENT_PROBE).read_bytes()).hexdigest()
    assert "PC2'S DOCSTRING, UNCHANGED:" in (REPO / P.PROBE).read_text()


def test_the_probe_writes_no_kernel_and_adds_no_stamp_point():
    for f in (P.PROBE, P.COUNTER):
        src = (REPO / f).read_text()
        for word in ("@triton.jit", "tl.load", "tl.store", "evict_first", "evict_last", "stamps=iter", "stamps=sample"):
            assert word not in src, (f, word)
    assert PR.LEVEL == 3 and PR.PCC is P


# --------------------------------------------------------------------------
# shapes and pages
# --------------------------------------------------------------------------

def test_the_shapes_are_registered_none_is_a_models_and_none_carries_a_reserved_gemm():
    doc = json.loads((REPO / P.SHAPES_FILE).read_text())
    assert doc == P.shapes_doc() and hashlib.sha256((REPO / P.SHAPES_FILE).read_bytes()).hexdigest() == \
        REG["shapes"]["sha256"]
    sh = P.check_shapes_doc(doc)
    real, res = P.model_shapes(), P.reserved_gemms()
    assert len(sh) == 11
    for name, cfg in sh.items():
        key = P.shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k)
        assert key not in real.values(), name
        assert not any(tuple(P.gemm_dims(cfg, g)) in res for g in ("w1", "w2")), name
    mx = MODEL_CONFIGS["mixtral-8x7b"]
    assert (mx.num_experts, mx.hidden_size, mx.intermediate_size, mx.top_k) == (8, 4096, 64 * 224, 2)


@pytest.mark.parametrize("bad, why", [
    ({"name": "pcs-x", "E": 8, "H": 4096, "F": 1280, "k": 2}, "an rrs- name"),
    ({"name": "rrs-mx", "E": 8, "H": 4096, "F": 14336, "k": 2}, "mixtral-8x7b's shape"),
    ({"name": "rrs-g", "E": 40, "H": 1024, "F": 512, "k": 8}, "granite-3.0-1b-a400m's w1 GEMM"),
    ({"name": "rrs-y", "E": 8, "H": "x", "F": 1, "k": 2}, "whole numbers"),
])
def test_a_shape_off_the_rules_is_refused(bad, why):
    with pytest.raises(P.Refused, match=why):
        P.check_shapes_doc({"schema": P.SHAPES_SCHEMA, "shapes": [bad]})


def test_the_pages_are_generated_byte_for_byte_and_registered():
    import gen_rr_pages as GEN
    for rel, text in GEN.files().items():
        assert (REPO / rel).read_text() == text, rel
    for n in P.page_files():
        rel = f"{P.PAGES_DIR}/{n}"
        assert REG["pages"][rel]["sha256"] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/rr/gen_rr_pages.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and got.stdout.count(": same") == 3, got.stdout


def test_the_blocks_are_the_designs():
    for page, tile in (("rr-128", P.T128), ("rr-64", P.T64)):
        cs = cells(page)
        assert len(cs) == 40 and {P.tile_key(c["tile"]) for c in cs.values()} == {P.tile_key(tile)}
        a = [c for c in cs.values() if c["series"] == "A"]
        assert len(a) == 32 and sorted({c["P"] for c in a}) == list(P.SERIES_A_P)
        assert {c["layout"] for c in a} == {"m2", "m4", "m8", "skew"}
        assert sorted({c["P"] for c in cs.values() if c["series"] == "B"}) == [24, 40]
        assert sorted({c["P"] for c in cs.values() if c["series"] == "C"}) == [36, 64]
        for c in cs.values():
            g = P.geometry(c["cfg"], c["counts"], c["tile"])
            assert g["P"] == c["P"] and g["R_bytes"] > 0 and c["read"] == "w1" and c["flush"] is True
            if c["layout"].startswith("m"):
                assert g["m_e"] == [int(c["layout"][1:])] * c["cfg"].num_experts
            assert c["T"] * c["cfg"].top_k == sum(c["counts"]) and c["T"] <= 4096
        assert cs[f"A{tile[0]}-P224-m8"]["shape"] == "mixtral-8x7b"
        assert sum(1 for c in cs.values() if c["P"] in P.ANCHOR_P) == 8


def test_every_cells_geometry_is_the_hand_value_and_r3_decodes_readback():
    c = cells("rr-128")["A128-P100-m4"]
    g = P.geometry(c["cfg"], c["counts"], c["tile"])
    We = 12800 * 4096 * 2
    assert (g["P"], g["W_c"], g["pairs"], g["live_w1"], g["slab_bytes"]) == (100, 264, 8 * 3 * 100, 3200, 1048576)
    assert g["C_u_bytes"] == 8 * We + 2048 * 4096 * 2 and g["R_bytes"] == 8 * 3 * We
    assert g["F_stag_MiB"] == 100.0
    for page in P.BLOCKS:
        for c in cells(page).values():
            conf = PR.conf_of(c["tile"])
            dc = RD.DecodeCell(label=c["label"], T=c["T"], counts=tuple(c["counts"]), arms=(RD.NATIVE,))
            ids = RD.cell_ids(c["cfg"], dc, P.SHUFFLE_SEED)
            exp = RD.expected_readback(c["cfg"], c["T"], c["counts"], ids.reshape(-1).tolist(),
                                       declared=c["cfg"].num_experts, copies=1, conf=conf)
            assert exp["path"] != RD.NAIVE and exp["live_tiles"] == sum(P.m_tiles(c["counts"], c["tile"][0]))
            rec = {"gemms": [{"sorted": list(range(exp["numel"])) + [exp["numel"]] * (exp["sorted_len"] - exp["numel"]),
                              "expert_ids": exp["expert_order"], "post_padded": exp["post_padded"], "grid": gr,
                              "use_td": False, "config": dict(conf)} for gr in exp["grid"]]}
            assert RD.check_readback(rec, exp) == [], c["label"]


def test_the_skew_layouts_are_the_published_pages_counts():
    for vec, (sess, unit, lab, n) in ((P.SKEW_E8, P.SKEW_E8_SOURCE), (P.SKEW_E64, P.SKEW_E64_SOURCE)):
        fs = glob.glob(str(PUB / sess / "results" / f"*-{unit}" / "lock*" / "r3c-g1.json"))
        if not fs:
            pytest.skip("published pages absent")
        page = json.loads(Path(fs[0]).read_text())
        got = next(c["bincount"] for c in page["cells"] if c["arm"] == "native" and c.get("histogram") == lab
                   and int(c["n"]) == n)
        assert tuple(got) == vec
    r = P.rescale(P.SKEW_E8, 4096)
    assert sum(r) == 4096 and r == [432, 456, 530, 506, 548, 542, 572, 510]
    assert sum(P.rescale(P.SKEW_E64, 16384)) == 16384


def test_axis_coverage_spans_the_step_and_the_candidates_regimes():
    rows = [P.geometry(c["cfg"], c["counts"], c["tile"]) | {"tile": P.tile_key(c["tile"]), "series": c["series"]}
            for page in P.BLOCKS for c in cells(page).values()]
    for t, lo, hi in (("128x128x64w8s3", 1.18, 13.2), ("64x128x64w8s3", 1.77, 19.8)):
        wp = [r["W_c_over_P"] for r in rows if r["tile"] == t]
        assert min(wp) <= lo + 0.01 and max(wp) >= hi - 0.01
    # design-P1's candidates sit inside: deepseek-v3 8.25 / 12.4, Mixtral-8x7B-tp4 4.7 / 7.1
    for t, x in (("128x128x64w8s3", 8.25), ("128x128x64w8s3", 4.7), ("64x128x64w8s3", 12.4), ("64x128x64w8s3", 7.1)):
        wp = [r["W_c_over_P"] for r in rows if r["tile"] == t]
        assert min(wp) < x < max(wp)
    fs = [r["F_stag_MiB"] for r in rows]
    assert min(fs) < 42.5 < max(fs) and max(fs) == 224
    assert max(r["m_e_max"] for r in rows) >= 8 and max(r["waves_w1"] for r in rows) > 50
    assert {r["slab_bytes"] for r in rows if r["series"] == "B"} == {2 * 2 ** 20}
    assert all(r["A"] == 64 for r in rows if r["series"] == "C")


@pytest.mark.parametrize("kw, why", [
    ({"counts": [128, 128, 0, 0, 0, 0, 0, 0], "T": 128}, "no expert holds two M-tiles"),
    ({"config": P.config_of((64, 128, 64, 4, 3))}, "is not one of the registered"),
    ({"read": "w2"}, "RR reads w1"),
    ({"flush": False}, "cold L2"),
    ({"group_m": 16}, "G 1"),
    ({"T": 5000, "counts": [5000, 5000, 0, 0, 0, 0, 0, 0]}, "1 to 4096"),
    ({"T": 1, "counts": [1, 1, 0, 0, 0, 0, 0, 0]}, "aligned path"),
    ({"counts": [256, 255, 1, 0, 0, 0, 0, 0], "T": 256}, None),
    ({"counts": [300, 212, 0, 0, 0, 0, 0, 0]}, "V-BIN"),
    ({"label": "a b"}, "letters"),
    ({"shape": "mixtral-8x22b"}, "reserved"),
    ({"shape": "deepseek-v3"}, "not one RR may load"),
])
def test_a_page_cell_off_the_design_is_refused(kw, why):
    if why is None:
        assert P.check_cell(cell(**kw), shapes())["label"] == "X"
        return
    with pytest.raises(P.Refused, match=why):
        P.check_cell(cell(**kw), shapes())


def test_a_page_with_two_tiles_a_repeated_label_or_no_cells_is_refused():
    two = [cell(), cell(label="Y", config=P.config_of(P.T64), counts=[128, 128, 0, 0, 0, 0, 0, 0], T=128)]
    with pytest.raises(P.Refused, match="one tile"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": two}, shapes())
    with pytest.raises(P.Refused, match="twice"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": [cell(), cell()]}, shapes())
    with pytest.raises(P.Refused, match="no cells"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": []}, shapes())
    with pytest.raises(P.Refused, match="schema"):
        P.check_page_doc({"schema": PC2C.SCHEMA, "shuffle_seed": 1, "cells": [cell()]}, shapes())


# --------------------------------------------------------------------------
# variants and modes
# --------------------------------------------------------------------------

def pv(**kw) -> dict:
    v = {"id": "u5-s128", "kind": "stamps", "probe": "pc", "page": P.PAGE_OF["s128"], "spec": "stamps=ends"}
    v.update(kw)
    return v


HIST = {"id": "u9-sr", "kind": "stamps", "probe": "pc", "model": "phi-3.5-moe", "groups": [1], "arms": ["native"],
        "block_m": 128, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3, "histogram": P.PHI_128,
        "spec": "stamps=ends"}


@pytest.mark.parametrize("kw, why", [
    ({"spec": "stamps=cta"}, "ends only"),
    ({"probe": "c3"}, "this probe runs PC stamps units"),
    ({"page": "/etc/passwd"}, "repo-relative"),
    ({"page": f"{P.PAGES_DIR}/none.json"}, "cannot read the page"),
    ({"page": f"{PC2C.PAGES_DIR}/pc2-d.json"}, "schema"),
])
def test_a_variant_off_rrs_design_is_refused(kw, why):
    with pytest.raises(PR.REFUSALS, match=why):
        PR.check_variant(pv(**kw))


@pytest.mark.parametrize("kw, why", [({"model": "mixtral-8x22b"}, "reserved"), ({"block_m": 16, "block_k": 128}, "tile"),
                                     ({"arms": ["native", "shared"]}, "NATIVE only")])
def test_the_replicates_histogram_variant_and_its_refusals(kw, why):
    v = PR.check_variant(dict(HIST))
    assert v["kind_"] == "hist" and len(v["cells"]) == 12
    with pytest.raises(PR.REFUSALS, match=why):
        PR.check_variant({**HIST, **kw})


def run_probe(*args, timeout=180):
    return subprocess.run([sys.executable, "-B", str(REPO / P.PROBE), *args], capture_output=True, text=True,
                          timeout=timeout)


def _variants(tmp_path) -> Path:
    f = tmp_path / "pc-variants.json"
    f.write_text(json.dumps([pv(id=P.VARIANTS["s128"]), pv(id=P.VARIANTS["s64"], page=P.PAGE_OF["s64"]), HIST]))
    return f


def test_each_modes_dry_run_lists_every_cell_and_exits_2(tmp_path):
    f = _variants(tmp_path)
    got = run_probe("--mode", "regcheck", "--variants", str(f), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and "REFUSED" not in got.stdout and "3 variant(s)" in got.stdout
    assert got.stdout.count("cells 40 ") == 2 and "cells 12 " in got.stdout
    assert "INSTRUMENTED KERNEL RR (regcheck)" in got.stdout and not (tmp_path / "o").exists()


@pytest.mark.parametrize("gate, rc, want", [
    ("GATE_u5_s128=PASS median=0.001 worst=0.002\nHARD_u5_s128=PASS\n", "REGCHECK_u5_s128=PASS\n", "DRY RUN"),
    ("GATE_u5_s128=FAIL median=0.02 worst=0.03\nHARD_u5_s128=PASS\n", "REGCHECK_u5_s128=PASS\n", "PERTURBED"),
    ("GATE_u5_s128=FAIL median=0.02 worst=0.03\nHARD_u5_s128=FAIL\n", "REGCHECK_u5_s128=PASS\n",
     "REFUSED: the perturbation gate's hard legs for u5-s128 read FAIL"),
    ("GATE_u5_s128=PASS median=0.001 worst=0.002\nHARD_u5_s128=PASS\n", "REGCHECK_u5_s128=FAIL 1 against 2\n",
     "REFUSED: the regcheck line for u5-s128 reads FAIL"),
])
def test_a_stamps_unit_runs_only_on_its_hard_legs_and_regcheck(tmp_path, gate, rc, want):
    g, r = tmp_path / "gate.env", tmp_path / "rc.env"
    g.write_text(gate)
    r.write_text(rc)
    got = run_probe("--mode", "stamps", "--variants", str(_variants(tmp_path)), "--variant", "u5-s128", "--gate", str(g),
                    "--regcheck", str(r), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and want in got.stdout, got.stdout


def test_with_no_cuda_the_probe_refuses_and_an_uncaught_error_exits_4(tmp_path, monkeypatch):
    monkeypatch.setattr(IP, "run_refusal", lambda: "no CUDA device")
    assert PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")]) == \
        exit_codes.REFUSED
    monkeypatch.setattr(IP, "run_refusal", lambda: "")
    monkeypatch.setattr(PR, "regcheck", lambda args, vs: (_ for _ in ()).throw(KeyError("planted")))
    assert PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")]) == 4


# --------------------------------------------------------------------------
# the guard, a cell on the CPU, the pinned files
# --------------------------------------------------------------------------

def test_nothing_in_the_plan_loads_a_reserved_model_and_no_gemm_is_a_candidates():
    loaded = {c["shape"] for page in P.BLOCKS for c in cells(page).values()}
    plan = (REPO / P.PLAN).read_text()
    body = [ln.split("#", 1)[0].split() for ln in plan.splitlines() if ln.split("#", 1)[0].strip()]
    loaded |= {w[0] for w in body if w[0] != "-"}
    loaded |= {m for w in body for kv in w[2:] if kv.startswith("models=") for m in kv[7:].split(",")}
    assert loaded <= set(P.REAL_ALLOWED) | {s["name"] for s in P.SHAPES} and not loaded & set(P.RESERVED)
    cand = {tuple(P.gemm_dims(MODEL_CONFIGS[m], g)) for m in P.CANDIDATE_MODELS for g in ("w1", "w2")}
    for m in loaded:
        cfg = P.shape_cfg(m)
        assert not {tuple(P.gemm_dims(cfg, g)) for g in ("w1", "w2")} & cand, m
    for word in ("8x22b", "qwen2-57b-a14b\"", "granite-3.0-1b"):
        assert word not in plan


@pytest.mark.parametrize("name", P.RESERVED)
def test_the_probe_refuses_a_reserved_model_before_any_weight_build(name, monkeypatch):
    built = []
    import private_weight_reference as PW
    monkeypatch.setattr(PW, "build_private_weights", lambda *a, **k: built.append(a))
    card = PR.Card.__new__(PR.Card)
    card.shapes, card.model = P.load_shapes(), None
    with pytest.raises(P.Refused, match="reserved"):
        card.load_shape(name)
    assert built == []
    g = MODEL_CONFIGS["granite-3.0-1b-a400m"]
    fake = {"rrs-g": MoEConfig(name="rrs-g", hidden_size=g.hidden_size, intermediate_size=g.intermediate_size,
                               num_experts=g.num_experts + 8, top_k=g.top_k)}
    with pytest.raises(P.Refused, match="granite-3.0-1b-a400m's"):
        P.shape_cfg("rrs-g", fake)


def test_the_smallest_cell_builds_its_ids_and_call_on_the_cpu():
    torch = pytest.importorskip("torch")
    card = PR.Card.__new__(PR.Card)
    card.device, card.torch, card.model, card.shapes = "cpu", torch, None, P.load_shapes()
    card.args = argparse.Namespace(seed=0)
    got = []
    card.fused_experts = lambda **kw: got.append(kw) or "out"
    card.activation = lambda: "silu"
    v = PR.check_variant(pv(id="u7-s64", page=P.PAGE_OF["s64"]))
    c = next(x for x in v["cells"] if x["label"] == "A64-P20-m2")
    fn, ids = card.cell_fn(v, c)
    assert torch.bincount(ids.reshape(-1).long(), minlength=8).tolist() == c["counts"] == [128] * 8
    assert fn() == "out" and got[0]["global_num_experts"] == 8 and tuple(got[0]["hidden_states"].shape) == (512, 4096)
    assert tuple(card.w1.shape) == (8, 2 * 1280, 4096)


def test_the_pinned_files_the_probe_imports_are_untouched():
    import test_pinned_files as TP
    for rel in ("moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == TP.BASELINE[rel], rel
    for rel in ("scripts/instr_probe.py", "scripts/instr_probe_c3.py", "scripts/instr_probe_pc2.py",
                "scripts/scoring/pc2/pc2common.py", "scripts/scoring/pc2/read_pc2.py", "scripts/gh200_pc2_session.sh",
                "scripts/vm_run_pc2.sh", "scripts/vm_run.sh", "scripts/r3_decode.py", "scripts/dram_counter_route.py",
                "scripts/private_weight_reference.py", "moe/spec.py"):
        h = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        assert PC2_REG["code_pins"][rel] == h == REG["code_pins"][rel], rel
    assert REG["instrument"]["copy_sha256"] == TP.BASELINE["moe/instrumented/fused_moe_instr.py"]
