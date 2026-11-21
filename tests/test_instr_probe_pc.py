"""Rental PC's probe off the GPU (scripts/instr_probe_pc.py, scripts/scoring/pc/pccommon.py): the shapes and pages, the
geometry of every registered cell against hand values and r3_decode, every refusal of a page, a shape, a variant and a
mode, the unseen-model guard (nothing in the plan loads Mixtral-8x22B, Qwen2-57B or Granite-3.0-1B, and the probe
refuses them before any weight build), the loads-in-flight reading on C3's published PTX, the regcheck path leg, the
readback composition, a decode cell built on the CPU, and the pinned files the probe imports left byte-identical."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import instr_probe_c3 as C3  # noqa: E402
import instr_probe_pc as PR  # noqa: E402
import pccommon as P  # noqa: E402
import r3_decode as RD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
C3_PTX = sorted((REPO / "results/published/2026-10-27-nvidia_gh200_480gb-cal3-session/results/"
                 "2026-10-27-nvidia_gh200_480gb-instr-mixtral-8x7b-st7/triton-cache").glob("*/fused_moe_kernel.ptx"))


def shapes():
    return P.load_shapes()


def cells(page):
    return {c["label"]: c for c in P.check_page_doc(P.page_doc(page), shapes())}


def cell(**kw) -> dict:
    c = {"label": "X", "shape": "pcs-e8-h512", "T": 64, "counts": [64, 64, 0, 0, 0, 0, 0, 0],
         "config": P.config_of(P.D64), "group_m": 1, "flush": True, "read": "w2"}
    c.update(kw)
    return c


# --------------------------------------------------------------------------
# shapes and pages
# --------------------------------------------------------------------------

def test_the_shapes_file_is_the_registered_one_and_no_shape_is_a_models():
    doc = json.loads((REPO / P.SHAPES_FILE).read_text())
    assert doc == P.shapes_doc() and hashlib.sha256((REPO / P.SHAPES_FILE).read_bytes()).hexdigest() == REG["shapes"]["sha256"]
    sh = P.check_shapes_doc(doc)
    real = P.model_shapes()
    for name, cfg in sh.items():
        key = P.shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k)
        assert key not in real.values() and name not in MODEL_CONFIGS and cfg.top_k == 2, name
    assert {s["name"] for s in P.SHAPES} == set(sh) and len(sh) == 9


@pytest.mark.parametrize("bad, why", [
    ({"name": "mixtral-8x7b", "E": 8, "H": 64, "F": 64, "k": 2}, "a pcs- name"),
    ({"name": "pcs-mx", "E": 8, "H": 4096, "F": 14336, "k": 2}, "is mixtral-8x7b's shape"),
    ({"name": "pcs-q", "E": 64, "H": 3584, "F": 2560, "k": 8}, "is qwen2-57b-a14b's shape"),
    ({"name": "pcs-x", "E": 8, "H": "a", "F": 64, "k": 2}, "whole numbers"),
])
def test_a_shape_naming_or_equal_to_a_model_is_refused(bad, why):
    doc = P.shapes_doc()
    doc["shapes"] = [bad]
    with pytest.raises(P.Refused, match=why):
        P.check_shapes_doc(doc)
    with pytest.raises(P.Refused, match="schema"):
        P.check_shapes_doc({"schema": "x"})


def test_the_pages_are_generated_byte_for_byte_and_registered():
    for rel, text in __import__("gen_pc_pages").files().items():
        assert (REPO / rel).read_text() == text, rel
    for page, n in (("pc-a", 38), ("pc-c", 9), ("pc-d", 15), ("pc-r", 3)):
        assert len(cells(page)) == n
        rel = f"{P.PAGES_DIR}/{page}.json"
        assert REG["pages"][rel]["sha256"] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/pc/gen_pc_pages.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and "DIFFERS" not in got.stdout, got.stdout + got.stderr


#: hand values: the read GEMM's live CTAs = A x ceil(N / BLOCK_N), and w1's = A x 2F / BLOCK_N
HAND = {("pc-a", "A-G2"): (448, 2), ("pc-a", "A-G8"): (448, 8), ("pc-a", "A-G32"): (448, 32),
        ("pc-a", "A-G64"): (448, 64), ("pc-a", "A-G1024"): (7168, 1024), ("pc-a", "A-MX-T128-A4"): (896, 128),
        ("pc-a", "A-JM-T128-A4"): (352, 64), ("pc-a", "A-MX-T128-A8"): (1792, 256), ("pc-a", "A-F1792-G8"): (56, 8),
        ("pc-a", "A-TP8-G128"): (112, 128), ("pc-a", "A-G384-s4k64"): (2688, 384), ("pc-d", "D-G4"): (896, 4),
        ("pc-d", "D-G1024"): (7168, 1024), ("pc-d", "D-MX-T2-A2"): (896, 128), ("pc-c", "C-G8"): (448, 8)}


def test_every_cells_geometry_is_the_hand_value_and_r3_decodes():
    for (page, label), (w1, w2) in HAND.items():
        c = cells(page)[label]
        assert (P.live_ctas(c["cfg"], c["counts"], c["tile"], "w1"), P.live_ctas(c["cfg"], c["counts"], c["tile"], "w2")) \
            == (w1, w2), label
    for page in P.BLOCKS:
        for c in cells(page).values():
            conf = PR.conf_of(c["tile"])
            g = RD.geometry(c["cfg"], c["T"], c["counts"], c["cfg"].num_experts, conf)
            assert list(g.live_ctas) == [P.live_ctas(c["cfg"], c["counts"], c["tile"], x) for x in ("w1", "w2")]
            assert g.path != RD.NAIVE and g.multi_tile_experts == 0 and max(c["counts"]) <= c["tile"][0]
            assert sum(c["counts"]) == c["T"] * c["cfg"].top_k


def test_the_blocks_are_the_designs():
    a = cells("pc-a")
    assert [k for k in a if k.startswith("A-G") and k[3:].isdigit()] == [
        "A-G2", "A-G4", "A-G8", "A-G16", "A-G32", "A-G64", "A-G128", "A-G160", "A-G192", "A-G256", "A-G288", "A-G384",
        "A-G416", "A-G512", "A-G768", "A-G1024"]
    assert all(P.tile_key(c["tile"]) == "64x128x64w4s3" for k, c in a.items() if "-s" not in k and "-w8" not in k)
    assert {P.tile_key(a[f"A-G8-{k}"]["tile"]) for k in P.FACT} == {P.tile_key(t) for t in P.FACT.values()}
    assert a["A-G8-noflush"]["flush"] is False and sum(1 for c in a.values() if not c["flush"]) == 1
    d = cells("pc-d")
    assert all(c["tile"][:4] == (16, 64, 128, 4) for c in d.values())
    assert [c["T"] for k, c in d.items() if k.startswith("D-MX")] == [2, 4, 16]
    c = cells("pc-c")
    assert {k[2:] for k in c} <= {k[2:] for k in a}
    r = cells("pc-r")
    for k, x in r.items():
        y = a["A-" + k[2:]]
        assert (x["shape"], x["counts"], x["tile"]) == (y["shape"], y["counts"], y["tile"])
    assert P.stamp_calls(2) == 6 and P.stamp_calls(4) == 3 and P.stamp_calls(1024) == 3


@pytest.mark.parametrize("kw, why", [
    ({"T": 2, "counts": [2, 2, 0, 0, 0, 0, 0, 0], "shape": "pcs-e32-h4096"}, "counts must be 32"),
    ({"T": 1, "counts": [1, 1, 0, 0, 0, 0, 0, 0]}, "naive path"),
    ({"T": 128, "counts": [128, 128, 0, 0, 0, 0, 0, 0]}, "past BLOCK_M 64"),
    ({"config": P.config_of((128, 128, 64, 8, 3))}, "is not one of the registered"),
    ({"config": {"BLOCK_SIZE_M": 64}}, "is not BLOCK_SIZE_M, N, K"),
    ({"shape": "pcs-nope"}, "not in the registered shapes file"),
    ({"shape": "deepseek-v3"}, "is not one PC may load"),
    ({"shape": "mixtral-8x22b"}, "is reserved"),
    ({"shape": "qwen2-57b-a14b"}, "is reserved"),
    ({"shape": "granite-3.0-1b-a400m"}, "is reserved"),
    ({"T": 63}, "V-BIN"),
    ({"flush": "yes"}, "true or false"),
    ({"read": "w3"}, "w1 or w2"),
    ({"group_m": 8}, "G 1"),
    ({"label": "a b"}, "letters, digits"),
])
def test_a_page_cell_off_the_design_is_refused(kw, why):
    with pytest.raises(P.Refused, match=why):
        P.check_cell(cell(**kw), shapes())


def test_a_page_with_a_repeated_label_or_no_cells_is_refused():
    with pytest.raises(P.Refused, match="twice"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": [cell(), cell()]}, shapes())
    with pytest.raises(P.Refused, match="no cells"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": []}, shapes())
    with pytest.raises(P.Refused, match="schema"):
        P.check_page_doc({"schema": RD.SCHEMA, "cells": [cell()]}, shapes())
    with pytest.raises(P.Refused, match="shuffle_seed"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": -1, "cells": [cell()]}, shapes())


def test_the_counter_page_is_a_decode_page_of_the_stamped_mixtral_cells():
    doc = json.loads((REPO / P.PAGES_DIR / f"{P.CNT_PAGE}.json").read_text())
    page = RD.page_from(doc, MODEL_CONFIGS["mixtral-8x7b"], "mixtral-8x7b")
    by = {c.label: c for c in page.cells}
    for lab, (pg, stem) in P.CNT_MATCH.items():
        c = cells(pg)[stem]
        assert list(by[lab].counts) == c["counts"] and by[lab].T == c["T"], lab
        assert RD.geometry(MODEL_CONFIGS["mixtral-8x7b"], c["T"], c["counts"]).tile == P.tile_key(c["tile"]), lab


# --------------------------------------------------------------------------
# variants and modes
# --------------------------------------------------------------------------

def pv(**kw) -> dict:
    v = {"id": "u5-sa", "kind": "stamps", "probe": "pc", "page": f"{P.PAGES_DIR}/pc-a.json", "spec": "stamps=ends"}
    v.update(kw)
    return v


def hv(**kw) -> dict:
    v = {"id": "u14-se2", "kind": "stamps", "probe": "pc", "model": "phi-3.5-moe", "groups": [1], "arms": ["native"],
         "block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3, "histogram": P.PHI_64,
         "spec": "stamps=ends"}
    v.update(kw)
    return v


def test_a_page_variant_and_a_histogram_variant_read_their_cells():
    v = PR.check_variant(pv())
    assert len(v["cells"]) == 38 and v["kind_"] == "page" and v["source_sha256"] == REG["pages"][v["page"]]["sha256"]
    h = PR.check_variant(hv())
    assert [(c["label"], c["n"]) for c in h["cells"]] == [("PT", 1), ("PW", 1), ("PT2", 1), ("PW2", 1), ("uniform", 1),
                                                          ("balanced", 1)]
    assert h["tile"] == P.P64 and all(c["arm"] == "native" for c in h["cells"])
    assert h["source_sha256"] == REG["pages"][P.PHI_64]["sha256"]
    o = PR.order(v["cells"])
    seen, firsts = [], []
    for c in o:
        if c["shape"] not in seen:
            seen.append(c["shape"])
            firsts.append(c["shape"])
        assert c["shape"] == seen[-1] or c["shape"] not in seen[:-1], "a shape returns after another"
    assert sorted(c["label"] for c in o) == sorted(c["label"] for c in v["cells"])


@pytest.mark.parametrize("fn, kw, why", [
    (pv, {"spec": "stamps=cta"}, "ends only"),
    (pv, {"spec": "stamps=ends,evict_b=first"}, "ends only"),
    (pv, {"probe": "c3"}, "this probe runs PC stamps units"),
    (pv, {"kind": "bytes"}, "this probe runs PC stamps units"),
    (pv, {"page": "/etc/passwd"}, "repo-relative"),
    (pv, {"page": f"{P.PAGES_DIR}/../x.json"}, "repo-relative"),
    (pv, {"page": f"{P.PAGES_DIR}/none.json"}, "cannot read the page"),
    (pv, {"histogram": P.PHI_64}, "one of them"),
    (hv, {"block_m": 128, "num_warps": 8}, "is not one of the registered"),
    (hv, {"arms": ["native", "shared"]}, "NATIVE only"),
    (hv, {"groups": [16]}, "G 1"),
    (hv, {"model": "qwen2-57b-a14b", "histogram": "docs/registered/2026-10-14-tileA-hist/qwen2-57b-a14b-128g1.json"},
     "is reserved"),
    (hv, {"model": "mixtral-8x22b"}, "is reserved"),
    (hv, {"histogram": P.PHI_128}, "counts sum to"),
])
def test_a_variant_off_pcs_design_is_refused(fn, kw, why):
    with pytest.raises(PR.REFUSALS, match=why):
        PR.check_variant(fn(**kw))


def test_a_variants_file_that_is_not_a_list_or_repeats_an_id_is_refused(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"id": "x"}))
    with pytest.raises(PR.Refused, match="a JSON list"):
        PR.load_variants(f)
    f.write_text(json.dumps([pv(), pv()]))
    with pytest.raises(PR.Refused, match="duplicate variant ids"):
        PR.load_variants(f)


def run_probe(*args, timeout=180):
    return subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "instr_probe_pc.py"), *args],
                          capture_output=True, text=True, timeout=timeout)


def _variants(tmp_path) -> Path:
    f = tmp_path / "pc-variants.json"
    f.write_text(json.dumps([pv(), pv(id="u17-sd", page=f"{P.PAGES_DIR}/pc-d.json"), hv()]))
    return f


def test_each_modes_dry_run_lists_every_cell_and_exits_2(tmp_path):
    f = _variants(tmp_path)
    got = run_probe("--mode", "regcheck", "--variants", str(f), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and "REFUSED" not in got.stdout and "3 variant(s)" in got.stdout
    assert "cells 38" in got.stdout and "cells 15" in got.stdout and "cells 6" in got.stdout and "DRY RUN" in got.stdout
    got = run_probe("--mode", "perturb", "--variants", str(f), "--only", "u5-sa,u14-se2", "--out", str(tmp_path / "o"),
                    "--dry-run")
    assert got.returncode == exit_codes.REFUSED and "REFUSED" not in got.stdout and "u17-sd" not in got.stdout
    assert "every cell of ['u14-se2', 'u5-sa']" in got.stdout


@pytest.mark.parametrize("args, why", [
    (["--mode", "perturb"], "--only None: the perturb unit names variants"),
    (["--mode", "perturb", "--only", "u9-x"], "--only 'u9-x'"),
    (["--mode", "regcheck", "--only", "u5-sa"], "--only belongs to perturb"),
    (["--mode", "stamps", "--variant", "u9-x"], "--variant 'u9-x' is not one of"),
])
def test_a_mode_with_the_wrong_ids_is_refused(tmp_path, args, why):
    got = run_probe(*args, "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and f"REFUSED: {why}" in got.stdout, got.stdout


@pytest.mark.parametrize("gate, rc, want", [
    ("GATE_u5_sa=PASS median=0.001 worst=0.002\nHARD_u5_sa=PASS\n", "REGCHECK_u5_sa=PASS\n", "DRY RUN"),
    ("GATE_u5_sa=FAIL median=0.02 worst=0.03\nHARD_u5_sa=PASS\n", "REGCHECK_u5_sa=PASS\n", "PERTURBED: the overhead gate"),
    ("GATE_u5_sa=FAIL median=0.02 worst=0.03\nHARD_u5_sa=FAIL\n", "REGCHECK_u5_sa=PASS\n",
     "REFUSED: the perturbation gate's hard legs for u5-sa read FAIL"),
    ("", "REGCHECK_u5_sa=PASS\n", "REFUSED: the perturbation gate's hard legs for u5-sa read nothing"),
    ("GATE_u5_sa=PASS median=0.001 worst=0.002\nHARD_u5_sa=PASS\n", "REGCHECK_u5_sa=FAIL 2 against 3\n",
     "REFUSED: the regcheck line for u5-sa reads FAIL"),
])
def test_a_stamps_unit_runs_only_on_its_hard_legs_and_regcheck(tmp_path, gate, rc, want):
    g, r = tmp_path / "gate.env", tmp_path / "rc.env"
    g.write_text(gate)
    r.write_text(rc)
    got = run_probe("--mode", "stamps", "--variants", str(_variants(tmp_path)), "--variant", "u5-sa", "--gate", str(g),
                    "--regcheck", str(r), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and want in got.stdout, got.stdout


def test_with_no_cuda_the_probe_refuses_before_anything_runs(tmp_path, monkeypatch):
    """run_refusal is the probe's own door; the test plants its answer, so it holds on any host (trap 7)."""
    monkeypatch.setattr(IP, "run_refusal", lambda: "no CUDA device")
    rc = PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")])
    assert rc == exit_codes.REFUSED and not (tmp_path / "o").exists()
    monkeypatch.setattr(IP, "run_refusal", lambda: "vLLM or Triton is not importable here")
    assert PR.main(["--mode", "perturb", "--only", "u5-sa", "--variants", str(_variants(tmp_path)), "--out",
                    str(tmp_path / "o")]) == exit_codes.REFUSED


# --------------------------------------------------------------------------
# the unseen-model guard (owner decision 3)
# --------------------------------------------------------------------------

def test_nothing_in_the_plan_loads_a_reserved_model_or_its_shape():
    reserved = set(P.RESERVED)
    rshape = P.reserved_shapes()
    loaded = set()
    for page in P.BLOCKS:
        loaded |= {c["shape"] for c in cells(page).values()}
    doc = json.loads((REPO / P.PAGES_DIR / f"{P.CNT_PAGE}.json").read_text())
    loaded.add(doc["model"])
    plan = (REPO / P.PLAN).read_text()
    body = [ln.split("#", 1)[0].split() for ln in plan.splitlines() if ln.split("#", 1)[0].strip()]
    loaded |= {w[0] for w in body if w[0] != "-"}
    loaded |= {m for w in body for kv in w[2:] if kv.startswith("models=") for m in kv[7:].split(",")}
    for kv in (kv for w in body for kv in w[2:] if kv.startswith("histogram=")):
        loaded.add(json.loads((REPO / kv[10:]).read_text())["model"])
    assert loaded and not loaded & reserved, loaded & reserved
    for m in loaded:
        cfg = P.shape_cfg(m)
        assert P.shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) not in rshape, m
    assert loaded <= set(P.REAL_ALLOWED) | {s["name"] for s in P.SHAPES}
    for word in ("8x22b", "qwen2-57b", "granite-3.0-1b"):
        assert word not in plan and word not in json.dumps(REG["cells (COMPUTED)"]), word


@pytest.mark.parametrize("name", P.RESERVED)
def test_the_probe_refuses_a_reserved_model_before_any_weight_build(name, monkeypatch):
    built = []
    import private_weight_reference as PW
    monkeypatch.setattr(PW, "build_private_weights", lambda *a, **k: built.append(a))
    card = PR.Card.__new__(PR.Card)
    card.shapes, card.model = P.load_shapes(), None
    with pytest.raises(P.Refused, match="reserved"):
        card.load(name)
    with pytest.raises(P.Refused, match="reserved"):
        card.load_shape(name)
    assert built == []
    with pytest.raises(P.Refused, match="reserved"):
        P.shape_cfg(name)


def test_a_synthetic_shape_equal_to_a_reserved_models_is_refused_at_build():
    from moe.spec import MoEConfig
    q = MODEL_CONFIGS["qwen2-57b-a14b"]
    fake = {"pcs-q": MoEConfig(name="pcs-q", hidden_size=q.hidden_size, intermediate_size=q.intermediate_size,
                               num_experts=q.num_experts, top_k=q.top_k)}
    with pytest.raises(P.Refused, match="reserved model's"):
        P.shape_cfg("pcs-q", fake)


# --------------------------------------------------------------------------
# the compiled kernel, regcheck, readback
# --------------------------------------------------------------------------

def test_loads_in_flight_on_c3s_published_ptx_are_num_stages_minus_one():
    assert len(C3_PTX) == 2
    for f in C3_PTX:
        got = P.inflight_stages(f.read_text())
        assert (got["n_if"], got["wait"], got["groups"]) == (2, 2, 2), got
        assert got["wait"] + 1 != 2   # the design's "immediate + 1" would read 3 here
    assert P.q_bytes(P.D64, 2) == 2 * 64 * 128 * 2 and P.q_bytes(P.D64, None) is None


def _ptx(wait: int, groups: int, loop: bool = True) -> str:
    body = ["$L__BB0_5:                             // =>This Inner Loop Header: Depth=1", f"\tcp.async.wait_group \t{wait};"]
    body += ["\tcp.async.commit_group;"] * groups + ["\t@%p9 bra \t$L__BB0_5;" if loop else "\tret;",
                                                     "\tcp.async.wait_group \t0;"]
    return "\n".join(body) + "\n"


@pytest.mark.parametrize("wait, groups, loop, n_if, why", [
    (0, 2, True, 1, ""), (2, 2, True, 2, ""), (4, 2, True, 3, ""), (1, 1, True, 2, ""),
    (3, 2, True, None, "not whole groups"), (2, 2, False, None, "no loop"), (2, 0, True, None, "no loop"),
])
def test_loads_in_flight_on_planted_ptx(wait, groups, loop, n_if, why):
    got = P.inflight_stages(_ptx(wait, groups, loop))
    assert got["n_if"] == n_if and why in got["why"]
    assert P.inflight_stages(None)["n_if"] is None


def _facts(regs, ctas):
    return {"regs": regs, "shared": 73728, "occupancy": {"ctas_per_sm": ctas}}


def test_regcheck_reads_the_census_predicates_path():
    own = "stamps=ends,evict_a=none,evict_b=none"
    k64 = (64, 128, 64, 1, 3, 4, False, 2)
    k16 = (16, 64, 128, 1, 4, 4, False, 2)
    rows = IP.regcheck_rows({k64: _facts(90, 3), k16: _facts(60, 3)}, {k64: _facts(90, 3), k16: _facts(60, 3)},
                            {own: {k64: _facts(94, 3), k16: _facts(64, 3)}})
    ok_paths = {"plain": {k64: C3.ptx_path("wgmma.mma_async"), k16: C3.ptx_path("mma.sync.aligned")},
                "copy": {k64: C3.ptx_path("wgmma.mma_async"), k16: C3.ptx_path("mma.sync.aligned")}}
    assert PR.regcheck_pc(rows, own, ok_paths) == (True, "")
    bad = {"plain": {k64: C3.ptx_path("wgmma.mma_async"), k16: C3.ptx_path("wgmma.mma_async")}, "copy": ok_paths["copy"]}
    ok, why = PR.regcheck_pc(rows, own, bad)
    assert not ok and "on the wgmma path, the predicate's is mma.sync" in why
    occ = IP.regcheck_rows({k64: _facts(90, 3)}, {k64: _facts(90, 3)}, {own: {k64: _facts(130, 2)}})
    assert PR.regcheck_pc(occ, own, ok_paths)[0] is False
    assert PR.regcheck_pc(rows, own, {"plain": {}, "copy": {}})[0] is False
    assert P.mma_path(64, 4) == "wgmma" and P.mma_path(16, 4) == "mma.sync" and P.mma_path(64, 8) == "wgmma"


def test_the_config_is_vllms_override_at_the_cells_tile():
    for t in list(P.TILES.values()) + [P.P128]:
        c = PR.conf_of(t)
        assert P.tile_of(c) == tuple(t) and c["GROUP_SIZE_M"] == 1
    v = C3.check_variant({"id": "u7", "kind": "stamps", "model": "mixtral-8x7b", "groups": [1], "arms": ["native"],
                          "block_m": 128, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3,
                          "histogram": "docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json",
                          "spec": "stamps=ends"})
    assert PR.conf_of(P.P128) == C3.conf(v)


def test_the_readback_composes_r3_decodes_rules_at_the_cells_config():
    for page in P.BLOCKS:
        for c in cells(page).values():
            conf = PR.conf_of(c["tile"])
            dc = RD.DecodeCell(label=c["label"], T=c["T"], counts=tuple(c["counts"]), arms=(RD.NATIVE,))
            ids = RD.cell_ids(c["cfg"], dc, P.SHUFFLE_SEED)
            exp = RD.expected_readback(c["cfg"], c["T"], c["counts"], ids.reshape(-1).tolist(),
                                       declared=c["cfg"].num_experts, copies=1, conf=conf)
            rec = {"gemms": [{"sorted": [exp["numel"]] * exp["sorted_len"] if exp["sorted_len"] else None,
                              "expert_ids": exp["expert_order"], "post_padded": exp["post_padded"], "grid": g,
                              "use_td": False, "config": dict(conf)} for g in exp["grid"]]}
            for r in rec["gemms"]:   # the padded rows: post_padded - numel sentinels within the live range
                r["sorted"] = list(range(exp["numel"])) + [exp["numel"]] * (exp["sorted_len"] - exp["numel"])
            assert RD.check_readback(rec, exp) == [], c["label"]
            bad = json.loads(json.dumps(rec))
            bad["gemms"][1]["config"]["num_stages"] = 9
            assert any("GD5" in w for w in RD.check_readback(bad, exp))


def test_a_decode_cell_builds_its_ids_and_call_on_the_cpu():
    torch = pytest.importorskip("torch")
    card = PR.Card.__new__(PR.Card)
    card.device, card.torch, card.model, card.shapes = "cpu", torch, None, P.load_shapes()
    card.args = argparse.Namespace(seed=0)
    got = []
    card.fused_experts = lambda **kw: got.append(kw) or "out"
    card.activation = lambda: "silu"
    v = PR.check_variant(pv())
    c = next(x for x in v["cells"] if x["label"] == "A-G2")
    fn, ids = card.cell_fn(v, c)
    assert torch.bincount(ids.reshape(-1).long(), minlength=8).tolist() == c["counts"]
    assert all(len(set(r.tolist())) == 2 for r in ids)
    assert fn() == "out" and got[0]["global_num_experts"] == 8 and tuple(got[0]["hidden_states"].shape) == (64, 128)
    assert tuple(card.w1.shape) == (8, 2 * 14336, 128) and tuple(card.w2.shape) == (8, 128, 14336)
    assert got[0]["topk_ids"] is ids and float(got[0]["topk_weights"][0, 0]) == 0.5
    with pytest.raises(P.Refused, match="reserved"):
        card.load_shape("granite-3.0-1b-a400m")


# --------------------------------------------------------------------------
# the pinned files
# --------------------------------------------------------------------------

def test_the_pinned_files_the_probe_imports_are_untouched():
    import test_pinned_files as TP
    for rel in ("moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == TP.BASELINE[rel], rel
    c3 = json.loads((REPO / "docs/registered/2026-10-26-cal3-gh200.json").read_text())["code_pins"]
    for rel in ("scripts/instr_probe.py", "scripts/instr_probe_c3.py", "moe/instrumented/fused_moe_instr.py",
                "scripts/gh200_cal3_session.sh", "scripts/scoring/cal3/read_c3.py", "scripts/scoring/cal2/read_c2.py",
                "scripts/private_weight_reference.py", "scripts/launch_floor.py"):
        assert c3[rel] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == REG["code_pins"][rel], rel
    d4 = json.loads((REPO / "docs/registered/2026-11-04-decode-addendum4-gh200.json").read_text())["code_pins"]
    for rel in ("scripts/r3_decode.py", "moe/spec.py", "scripts/dram_counter_route_decode3.py"):
        assert d4[rel] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == REG["code_pins"][rel], rel
    assert REG["instrument"]["copy_sha256"] == TP.BASELINE["moe/instrumented/fused_moe_instr.py"]


def test_the_probe_writes_no_kernel_and_adds_no_stamp_point():
    src = (REPO / "scripts" / "instr_probe_pc.py").read_text()
    for word in ("@triton.jit", "tl.load", "tl.store", "evict_first", "evict_last", "stamps=iter", "stamps=sample"):
        assert word not in src, word
    assert PR.LEVEL == 3 and "fused_moe_instr.fused_moe_kernel" in src


def test_one_variant_that_raises_costs_only_its_own_lines(tmp_path, monkeypatch):
    """Review MAJOR-2: each variant's regcheck and gate lines are written as it finishes; one that raises reads FAIL."""
    vs = PR.load_variants(_variants(tmp_path))
    monkeypatch.setattr(PR, "Card", lambda args: object())

    def rc(card, args, v):
        if v["id"] == "u17-sd":
            raise RuntimeError("planted")
        return True, "", {"ok": True}
    monkeypatch.setattr(PR, "regcheck_variant", rc)
    args = argparse.Namespace(out=tmp_path)
    assert PR.regcheck(args, vs) == exit_codes.CLAIM_FAIL
    env = (tmp_path / "regcheck.env").read_text().splitlines()
    assert env[0] == "REGCHECK_u5_sa=PASS" and env[1].startswith("REGCHECK_u17_sd=FAIL the variant raised RuntimeError")
    assert env[2] == "REGCHECK_u14_se2=PASS"


def test_an_uncaught_error_exits_4_never_0_or_1(tmp_path, monkeypatch):
    monkeypatch.setattr(IP, "run_refusal", lambda: "")
    monkeypatch.setattr(PR, "regcheck", lambda args, vs: (_ for _ in ()).throw(KeyError("planted")))
    rc = PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")])
    assert rc == exit_codes.ERROR == 4


def test_the_regcheck_line_lists_excluded_cells_and_readers_take_its_first_word(tmp_path):
    line = PR.regcheck_env_line("u5-sa", True, "", {"A-G8": ["x"], "A-G2": ["y"]})
    assert line == "REGCHECK_u5_sa=PASS excluded 2: A-G2,A-G8"
    f = tmp_path / "regcheck.env"
    f.write_text(line + "\n")
    assert IP.read_regcheck(f, "u5-sa") == "PASS" and C3.read_env(f, "REGCHECK", "u5-sa") == "PASS"
    assert PR.regcheck_env_line("u5-sa", True, "", {}) == "REGCHECK_u5_sa=PASS"


def test_writes_are_atomic(tmp_path):
    np = pytest.importorskip("numpy")
    PR.atomic_write_text(tmp_path / "stamps.json", "{}")
    PR.atomic_save(tmp_path / "x-call0-w1.npy", np.arange(3))
    assert (tmp_path / "stamps.json").read_text() == "{}" and np.load(tmp_path / "x-call0-w1.npy").tolist() == [0, 1, 2]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stamps.json", "x-call0-w1.npy"]   # no temporary left
    src = (REPO / "scripts" / "instr_probe_pc.py").read_text()
    body = src[src.index("def stamps("):src.index("def build_parser(")]
    assert "np.save(" not in body and ".write_text(" not in body


def test_every_distinct_compile_is_kept_for_its_sass_and_loads_in_flight():
    class CK:
        def __init__(self, h):
            self.hash, self.asm, self.metadata, self.n_regs = h, {"ptx": "p" + h}, {"shared": 1, "num_warps": 4}, 90
    rec = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_stages": 3,
           "num_warps": 4, "MUL_ROUTED_WEIGHT": False, "top_k": 2}
    got = PR.distinct_compiles([{**rec, "compiled": CK("a")}, {**rec, "compiled": CK("a")},
                                {**rec, "compiled": CK("b")}], {})
    assert sorted(h for _k, h in got) == ["a", "b"] and len({k for k, _h in got}) == 1
