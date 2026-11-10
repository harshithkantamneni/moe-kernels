"""Rental PC2's probe and rules off the GPU (scripts/instr_probe_pc2.py, scripts/scoring/pc2/pc2common.py): the probe is
PC's pinned probe plus lines marked PC2; the shapes and pages are generated and registered; every cell's geometry is
r3_decode's (naive and aligned); every refusal of a shape, a cell, a twin or pair, a variant and a mode; the unseen-model
guard (nothing in the plan loads Mixtral-8x22B, Qwen2-57B, its tp8 shard or Granite-3.0-1B, by name, by shape or by
GEMM, and the probe refuses them before any weight build); the axis coverage (the cells reach every regime design-D6
section 7 names, Granite-1B's included, with synthetic shapes); the frozen literal of PC's decode-tile regcheck
recomputed from PC's published page; the readback composition on naive cells; a naive cell built on the CPU; and the
pinned files the probe imports left byte-identical."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "pc2", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import instr_probe_c3 as C3  # noqa: E402
import instr_probe_pc2 as PR  # noqa: E402
import pc2common as P  # noqa: E402
import pccommon as PCC  # noqa: E402
import r3_decode as RD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS, MoEConfig  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())
PC_REG = json.loads((REPO / "docs" / "registered" / f"{PCC.NAME}.json").read_text())
PC_RCP = REPO / ("results/published/2026-11-06-nvidia_gh200_480gb-pc-session/results/"
                 "2026-11-06-nvidia_gh200_480gb-regcheck-rcp/regcheck.json")


def shapes():
    return P.load_shapes()


def cells(page):
    return {c["label"]: c for c in P.check_page_doc(P.page_doc(page), shapes())}


def cell(**kw) -> dict:
    c = {"label": "X", "shape": "pcs-e8-h512", "T": 64, "counts": [64, 64, 0, 0, 0, 0, 0, 0],
         "config": P.config_of(P.D64), "group_m": 1, "flush": True, "read": "w2", "path": "aligned"}
    c.update(kw)
    return c


# --------------------------------------------------------------------------
# the probe is PC's probe plus lines marked PC2
# --------------------------------------------------------------------------

def test_the_probe_is_pcs_probe_plus_lines_marked_pc2():
    a = (REPO / P.PARENT_PROBE).read_text().splitlines()
    b = (REPO / P.PROBE).read_text().splitlines()
    start = b.index("from __future__ import annotations")
    assert a.index("from __future__ import annotations") < len(a)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        assert tag in ("replace", "insert") or j1 < start, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < start or "PC2" in b[j], (j + 1, b[j])
    changed = [x for x in b[start:] if "PC2" in x]
    assert len(changed) == 6 and any("import pc2common as PCC" in x for x in changed)
    assert sum('"tool": "scripts/instr_probe_pc2.py --mode' in x for x in changed) == 3
    pa = a[a.index("from __future__ import annotations"):]
    assert [x for x in pa if x not in b] == [x for x in pa if "pccommon" in x or "INSTRUMENTED KERNEL PC (" in x
                                             or x.startswith("for p in (str(REPO), str(HERE)")
                                             or '"tool": "scripts/instr_probe_pc.py --mode' in x]
    assert PC_REG["code_pins"][P.PARENT_PROBE] == hashlib.sha256((REPO / P.PARENT_PROBE).read_bytes()).hexdigest()
    assert "PC'S DOCSTRING, UNCHANGED:" in (REPO / P.PROBE).read_text()


def test_the_probe_writes_no_kernel_and_adds_no_stamp_point():
    src = (REPO / P.PROBE).read_text()
    for word in ("@triton.jit", "tl.load", "tl.store", "evict_first", "evict_last", "stamps=iter", "stamps=sample"):
        assert word not in src, word
    assert PR.LEVEL == 3 and "fused_moe_instr.fused_moe_kernel" in src and PR.PCC is P


# --------------------------------------------------------------------------
# shapes and pages
# --------------------------------------------------------------------------

def test_the_shapes_file_is_the_registered_one_and_no_shape_is_a_models_or_a_reserved_gemm():
    doc = json.loads((REPO / P.SHAPES_FILE).read_text())
    assert doc == P.shapes_doc() and hashlib.sha256((REPO / P.SHAPES_FILE).read_bytes()).hexdigest() == REG["shapes"]["sha256"]
    sh = P.check_shapes_doc(doc)
    real = P.model_shapes()
    res = P.reserved_gemms()
    for name, cfg in sh.items():
        assert P.shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) not in real.values()
        assert name not in MODEL_CONFIGS and name.startswith(("pcs-", "pcs2-"))
        for g in ("w1", "w2"):
            assert tuple(P.gemm_dims(cfg, g)) not in res, (name, g)
    pc = {s["name"]: s for s in PCC.SHAPES}
    for s in doc["shapes"]:
        if s["name"].startswith("pcs-"):
            assert s == pc[s["name"]], s
    assert len(sh) == len(P.SHAPES) == 26
    for m in P.REAL_ALLOWED:   # the SEEN models PC2 loads carry no reserved GEMM either
        for g in ("w1", "w2"):
            assert tuple(P.gemm_dims(MODEL_CONFIGS[m], g)) not in res, (m, g)


@pytest.mark.parametrize("bad, why", [
    ({"name": "mixtral-8x7b", "E": 8, "H": 64, "F": 64, "k": 2}, "a pcs- or pcs2- name"),
    ({"name": "pcs3-x", "E": 8, "H": 64, "F": 64, "k": 2}, "a pcs- or pcs2- name"),
    ({"name": "pcs2-mx", "E": 8, "H": 4096, "F": 14336, "k": 2}, "is mixtral-8x7b's shape"),
    ({"name": "pcs2-q", "E": 64, "H": 3584, "F": 2560, "k": 8}, "is qwen2-57b-a14b's shape"),
    ({"name": "pcs2-g1", "E": 40, "H": 1024, "F": 512, "k": 8}, "is granite-3.0-1b-a400m's"),
    ({"name": "pcs2-g1w1", "E": 8, "H": 1024, "F": 512, "k": 2}, "is granite-3.0-1b-a400m's"),
    ({"name": "pcs2-qw2", "E": 8, "H": 3584, "F": 2560, "k": 2}, "is qwen2-57b-a14b's"),
    ({"name": "pcs2-x22", "E": 4, "H": 6144, "F": 16384, "k": 2}, "is mixtral-8x22b's"),
    ({"name": "pcs-e8-h512", "E": 8, "H": 512, "F": 7168, "k": 2}, "a pcs- name is PC's shape"),
    ({"name": "pcs2-x", "E": 8, "H": "a", "F": 64, "k": 2}, "whole numbers"),
])
def test_a_shape_naming_or_equal_to_a_model_or_carrying_a_reserved_gemm_is_refused(bad, why):
    doc = P.shapes_doc()
    doc["shapes"] = [bad]
    with pytest.raises(P.Refused, match=why):
        P.check_shapes_doc(doc)
    with pytest.raises(P.Refused, match="schema"):
        P.check_shapes_doc({"schema": PCC.SHAPES_SCHEMA})
    with pytest.raises(P.Refused, match="no shape"):
        P.check_shapes_doc({"schema": P.SHAPES_SCHEMA, "shapes": []})


def test_a_gemm_equal_to_either_reserved_gemm_is_refused_whatever_its_role():
    """The stricter guard compares every PC2 GEMM with both GEMMs of every reserved model: H 512, F 512 has w1 (1024,
    512), Granite-1B's w2 (the guard moved PC2's G 8 short-slab ladder to H 256)."""
    with pytest.raises(P.Refused, match=r"w1 GEMM \(N, K\) \(1024, 512\) is granite-3.0-1b-a400m's w2 GEMM"):
        P.gemm_guard("x", MoEConfig(name="x", hidden_size=512, intermediate_size=512, num_experts=8, top_k=2))
    with pytest.raises(P.Refused, match="is granite-3.0-1b-a400m's w1 GEMM"):
        P.gemm_guard("x", MoEConfig(name="x", hidden_size=1024, intermediate_size=512, num_experts=4, top_k=1))
    with pytest.raises(P.Refused, match="is qwen2-57b-a14b's w1 GEMM"):
        P.gemm_guard("x", MoEConfig(name="x", hidden_size=3584, intermediate_size=2560, num_experts=4, top_k=1))
    P.gemm_guard("ok", MoEConfig(name="ok", hidden_size=2048, intermediate_size=1024, num_experts=4, top_k=1))


def test_the_pages_are_generated_byte_for_byte_and_registered():
    import gen_pc2_pages as GEN
    for rel, text in GEN.files().items():
        assert (REPO / rel).read_text() == text, rel
    for page, n in (("pc2-d", 29), ("pc2-s", 20), ("pc2-n", 26), ("pc2-w", 29), ("pc2-d3", 3), ("pc2-r", 3)):
        assert len(cells(page)) == n, page
        rel = f"{P.PAGES_DIR}/{page}.json"
        assert REG["pages"][rel]["sha256"] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts/scoring/pc2/gen_pc2_pages.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert got.returncode == 0 and "DIFFERS" not in got.stdout, got.stdout + got.stderr


#: hand values (w1, w2 live CTAs): D16 w2 = A x H / 64; naive = T k x ceil(N / 64)
HAND = {("pc2-d", "D-G4"): (2 * 448, 4), ("pc2-d", "D-G1024"): (16 * 448, 1024), ("pc2-d", "D-G8-F512"): (2 * 16, 8),
        ("pc2-d", "D-G128-F3584"): (2 * 112, 128), ("pc2-s", "S-G8-F256"): (4 * 4, 8), ("pc2-s", "S-G256"): (8 * 224, 256),
        ("pc2-w", "W64-G528"): (8 * 448, 528), ("pc2-w", "W128-G528"): (16 * 224, 528), ("pc2-w", "W64-G396-F512"): (6 * 16, 396),
        ("pc2-n", "N-MX-T1"): (2 * 448, 2 * 64), ("pc2-n", "N-G3-T1"): (8 * 16, 8 * 24),
        ("pc2-n", "N-PH-T2-A2"): (4 * 200, 4 * 64), ("pc2-n", "N-K8-T2-A16"): (16 * 44, 16 * 32)}


def test_every_cells_geometry_is_the_hand_value_and_r3_decodes():
    for (page, label), want in HAND.items():
        c = cells(page)[label]
        assert tuple(P.live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")) == want, label
    for page in P.BLOCKS:
        for c in cells(page).values():
            conf = PR.conf_of(c["tile"])
            g = RD.geometry(c["cfg"], c["T"], c["counts"], c["cfg"].num_experts, conf)
            assert list(g.live_ctas) == [P.live_ctas(c["cfg"], c["counts"], c["tile"], x) for x in ("w1", "w2")], c["label"]
            assert (g.path == RD.NAIVE) == (c["path"] == "naive") and g.multi_tile_experts == 0
            assert sum(c["counts"]) == c["T"] * c["cfg"].top_k and max(c["counts"]) <= c["tile"][0]


def test_the_blocks_are_the_designs():
    d = cells("pc2-d")
    assert all(P.tile_key(c["tile"]) == "16x64x128w4s4" for c in d.values())
    assert [int(k[3:]) for k in d if k.startswith("D-G") and k[3:].isdigit()] == [4, 8, 16, 32, 64, 128, 192, 256, 320,
                                                                                  384, 512, 1024]
    assert all(P.tile_key(c["tile"]) == "16x64x128w4s3" for c in cells("pc2-d3").values())
    s = cells("pc2-s")
    assert all(P.tile_key(c["tile"]) == "64x128x64w4s3" and max(c["counts"]) == 64 for c in s.values())
    w = cells("pc2-w")
    assert {P.tile_key(c["tile"]) for c in w.values()} == {"32x64x128w4s3", "32x128x64w4s3"}
    n = cells("pc2-n")
    assert all(P.tile_key(c["tile"]) == P.NAIVE_TILE for c in n.values())
    assert sum(1 for c in n.values() if c["path"] == "naive") == 15
    r = cells("pc2-r")
    src = {**cells("pc2-d"), **cells("pc2-s")}
    for k, x in r.items():
        y = src[k[2:]]
        assert (x["shape"], x["counts"], x["tile"]) == (y["shape"], y["counts"], y["tile"])
    assert all(c["flush"] for p in P.BLOCKS for c in cells(p).values())


def test_axis_coverage_every_regime_v6d_cannot_price_is_reached():
    """design-D6 section 7: items 1 to 4, and Granite-1B's reach per tile (frozen from reach6.out) inside PC2's
    single-wave cells with a rung at or beyond each end of both axes."""
    allc = [c for p in ("pc2-d", "pc2-s", "pc2-w") for c in cells(p).values()]

    def box(tile):
        out = []
        for c in allc:
            if P.tile_key(c["tile"]) != tile:
                continue
            n = P.live_ctas(c["cfg"], c["counts"], c["tile"], "w2")
            if n <= P.wave(c["tile"]):
                out.append((n, P.slab_bytes(c["cfg"], c["tile"], "w2") / 1e3))
        return out
    for tile, (n0, n1, s0, s1) in P.GRANITE1B_REACH.items():
        pts = box(tile)
        assert pts, tile
        short = [(n, s) for n, s in pts if s <= s1 * 1.05]
        assert min(s for _n, s in short) <= s0 * 1.05 and max(s for _n, s in short) >= s1 * 0.95, (tile, short)
        W = P.wave(P.TILES[tile])   # the largest whole-64 count inside one wave (Granite-1B's n step is 16 or 33)
        assert min(n for n, _s in short) <= n0 and max(n for n, _s in short) >= min(n1, W - W % 64), (tile, short)
    # item 1: the decode tile from G 4 to one wave and past it, at the long slab and on F ladders at G 8 and 128
    d = [(P.live_ctas(c["cfg"], c["counts"], c["tile"], "w2"), P.slab_bytes(c["cfg"], c["tile"], "w2"))
         for c in cells("pc2-d").values()]
    assert {4, 384, 512, 1024} <= {n for n, s in d if s == 14336 * 128}
    for G in (8, 128):
        assert sorted({s // 128 for n, s in d if n == G}) == [512, 1024, 1792, 3584, 7168, 14336]
    # item 2: 66, 131, 262 KB on 64x128x64 at G 8, 32, 128, 256
    s = [(P.live_ctas(c["cfg"], c["counts"], c["tile"], "w2"), P.slab_bytes(c["cfg"], c["tile"], "w2"))
         for c in cells("pc2-s").values()]
    for G in (8, 32, 128, 256):
        assert {65536, 131072, 262144, 458752, 3670016} == {b for n, b in s if n == G}
    # item 3: single waves of 384 to 528 on both 4-per-SM tiles, and past one wave
    for pre, tile in (("W64", P.W64), ("W128", P.W128)):
        ns = {P.live_ctas(c["cfg"], c["counts"], c["tile"], "w2") for k, c in cells("pc2-w").items() if k.startswith(pre + "-")}
        assert {396, 462, 528, 594, 1056} <= ns and P.wave(tile) == 528
    # item 4: a naive cell with a partner for every SEEN model whose naive cells the published blind holds
    nv = {c["shape"] for c in cells("pc2-n").values() if c["path"] == "naive"}
    assert set(P.BLIND_NAIVE_CELLS) <= nv and "pcs2-e64-h2048-f1408-k8" in nv
    assert any(c["path"] == "naive" and not P.slab_once(c["cfg"], c["counts"]) for c in cells("pc2-n").values())


@pytest.mark.parametrize("kw, why", [
    ({"T": 1, "counts": [1, 1, 0, 0, 0, 0, 0, 0]}, "declares the aligned path, vLLM takes the naive one"),
    ({"path": "naive"}, "declares the naive path, vLLM takes the aligned one"),
    ({"path": "fast"}, "naive or aligned"),
    ({"T": 1, "counts": [1, 1, 0, 0, 0, 0, 0, 0], "path": "naive"}, "a naive cell runs at 16x64x128w4s4"),
    ({"T": 2, "counts": [2, 2, 0, 0, 0, 0, 0, 0], "shape": "pcs-e32-h4096"}, "counts must be 32"),
    ({"T": 128, "counts": [128, 128, 0, 0, 0, 0, 0, 0]}, "past BLOCK_M 64"),
    ({"config": P.config_of((128, 128, 64, 8, 3))}, "is not one of the registered"),
    ({"config": P.config_of((16, 64, 128, 4, 2))}, "is not one of the registered"),
    ({"config": {"BLOCK_SIZE_M": 64}}, "is not BLOCK_SIZE_M, N, K"),
    ({"shape": "pcs2-nope"}, "not in the registered shapes file"),
    ({"shape": "deepseek-v3"}, "is not one PC2 may load"),
    ({"shape": "olmoe-1b-7b"}, "is not one PC2 may load"),
    ({"shape": "mixtral-8x22b"}, "is reserved"),
    ({"shape": "qwen2-57b-a14b"}, "is reserved"),
    ({"shape": "qwen2-57b-a14b-tp8"}, "is reserved"),
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


def _n(label, counts, path, **kw):
    c = {"label": label, "shape": "phi-3.5-moe", "counts": list(counts) + [0] * (16 - len(counts)),
         "config": P.config_of(P.D16), "group_m": 1, "flush": True, "read": "w2", "path": path, **kw}
    c["T"] = sum(c["counts"]) // 2
    return c


@pytest.mark.parametrize("page, why", [
    ([_n("a", [1, 1], "naive")], "a naive cell without followers names a twin"),
    ([_n("a", [2, 2], "naive", twin="b"), _n("b", [3, 3], "aligned")], "a naive cell with followers names a pair"),
    ([_n("a", [1, 1], "naive", twin="z")], "its twin 'z' is not on the page"),
    ([_n("a", [1, 1], "naive", twin="b"), _n("b", [2, 2, 1, 1], "aligned")], "has live CTAs"),
    ([_n("a", [1, 1], "naive", twin="b"), _n("b", [0, 3, 3], "aligned")], "has other active experts"),
    ([_n("a", [1, 1], "naive", twin="b"), _n("b", [1, 1], "naive", twin="a")], "is not on the aligned path"),
    ([_n("a", [2, 2], "naive", pair="b"), _n("b", [2, 2], "naive", pair="a")], "is not a naive cell without followers"),
    ([_n("a", [1, 1], "naive", twin="b"), {**_n("b", [3, 3], "aligned"), "shape": "jetmoe-8b",
                                           "counts": [3, 3] + [0] * 6}], "another shape or tile"),
    ([_n("b", [3, 3], "aligned", twin="a")], "only a naive cell names a twin or a pair"),
])
def test_a_naive_cells_partner_off_the_design_is_refused(page, why):
    with pytest.raises(P.Refused, match=why):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": page}, shapes())


def test_a_page_with_a_repeated_label_or_no_cells_is_refused():
    with pytest.raises(P.Refused, match="twice"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": [cell(), cell()]}, shapes())
    with pytest.raises(P.Refused, match="no cells"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": 1, "cells": []}, shapes())
    with pytest.raises(P.Refused, match="schema"):
        P.check_page_doc({"schema": PCC.SCHEMA, "cells": [cell()]}, shapes())
    with pytest.raises(P.Refused, match="shuffle_seed"):
        P.check_page_doc({"schema": P.SCHEMA, "shuffle_seed": -1, "cells": [cell()]}, shapes())


# --------------------------------------------------------------------------
# variants and modes
# --------------------------------------------------------------------------

def pv(**kw) -> dict:
    v = {"id": "u5-sd", "kind": "stamps", "probe": "pc", "page": f"{P.PAGES_DIR}/pc2-d.json", "spec": "stamps=ends"}
    v.update(kw)
    return v


@pytest.mark.parametrize("kw, why", [
    ({"spec": "stamps=cta"}, "ends only"),
    ({"spec": "stamps=ends,evict_b=first"}, "ends only"),
    ({"probe": "c3"}, "this probe runs PC stamps units"),
    ({"kind": "bytes"}, "this probe runs PC stamps units"),
    ({"page": "/etc/passwd"}, "repo-relative"),
    ({"page": f"{P.PAGES_DIR}/../x.json"}, "repo-relative"),
    ({"page": f"{P.PAGES_DIR}/none.json"}, "cannot read the page"),
    ({"page": f"{PCC.PAGES_DIR}/pc-a.json"}, "schema"),
])
def test_a_variant_off_pc2s_design_is_refused(kw, why):
    with pytest.raises(PR.REFUSALS, match=why):
        PR.check_variant(pv(**kw))


def test_a_page_variant_reads_its_cells_and_orders_them_by_shape():
    v = PR.check_variant(pv(id="u7-sn", page=f"{P.PAGES_DIR}/pc2-n.json"))
    assert len(v["cells"]) == 26 and v["source_sha256"] == REG["pages"][v["page"]]["sha256"]
    o = PR.order(v["cells"])
    seen = []
    for c in o:
        if c["shape"] not in seen:
            seen.append(c["shape"])
        assert c["shape"] == seen[-1], "a shape returns after another"


def run_probe(*args, timeout=180):
    return subprocess.run([sys.executable, "-B", str(REPO / P.PROBE), *args], capture_output=True, text=True,
                          timeout=timeout)


def _variants(tmp_path) -> Path:
    f = tmp_path / "pc-variants.json"
    f.write_text(json.dumps([pv(id=v, page=P.PAGE_OF[lab]) for lab, v in P.VARIANTS.items()]))
    return f


def test_each_modes_dry_run_lists_every_cell_and_exits_2(tmp_path):
    f = _variants(tmp_path)
    got = run_probe("--mode", "regcheck", "--variants", str(f), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and "REFUSED" not in got.stdout and "6 variant(s)" in got.stdout
    for n in (29, 20, 26, 29, 3):
        assert f"cells {n} " in got.stdout
    assert "INSTRUMENTED KERNEL PC2 (regcheck)" in got.stdout and "DRY RUN" in got.stdout
    got = run_probe("--mode", "perturb", "--variants", str(f), "--only", "u5-sd,u7-sn", "--out", str(tmp_path / "o"),
                    "--dry-run")
    assert got.returncode == exit_codes.REFUSED and "u8-sw" not in got.stdout
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("args, why", [
    (["--mode", "perturb"], "--only None: the perturb unit names variants"),
    (["--mode", "perturb", "--only", "u9-x"], "--only 'u9-x'"),
    (["--mode", "regcheck", "--only", "u5-sd"], "--only belongs to perturb"),
    (["--mode", "stamps", "--variant", "u9-x"], "--variant 'u9-x' is not one of"),
])
def test_a_mode_with_the_wrong_ids_is_refused(tmp_path, args, why):
    got = run_probe(*args, "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and f"REFUSED: {why}" in got.stdout, got.stdout


@pytest.mark.parametrize("gate, rc, want", [
    ("GATE_u5_sd=PASS median=0.001 worst=0.002\nHARD_u5_sd=PASS\n", "REGCHECK_u5_sd=PASS\n", "DRY RUN"),
    ("GATE_u5_sd=FAIL median=0.02 worst=0.03\nHARD_u5_sd=PASS\n", "REGCHECK_u5_sd=PASS\n", "PERTURBED: the overhead gate"),
    ("GATE_u5_sd=FAIL median=0.02 worst=0.03\nHARD_u5_sd=FAIL\n", "REGCHECK_u5_sd=PASS\n",
     "REFUSED: the perturbation gate's hard legs for u5-sd read FAIL"),
    ("", "REGCHECK_u5_sd=PASS\n", "REFUSED: the perturbation gate's hard legs for u5-sd read nothing"),
    ("GATE_u5_sd=PASS median=0.001 worst=0.002\nHARD_u5_sd=PASS\n", "REGCHECK_u5_sd=FAIL 7 against 8\n",
     "REFUSED: the regcheck line for u5-sd reads FAIL"),
])
def test_a_stamps_unit_runs_only_on_its_hard_legs_and_regcheck(tmp_path, gate, rc, want):
    g, r = tmp_path / "gate.env", tmp_path / "rc.env"
    g.write_text(gate)
    r.write_text(rc)
    got = run_probe("--mode", "stamps", "--variants", str(_variants(tmp_path)), "--variant", "u5-sd", "--gate", str(g),
                    "--regcheck", str(r), "--out", str(tmp_path / "o"), "--dry-run")
    assert got.returncode == exit_codes.REFUSED and want in got.stdout, got.stdout


def test_with_no_cuda_the_probe_refuses_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(IP, "run_refusal", lambda: "no CUDA device")
    rc = PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")])
    assert rc == exit_codes.REFUSED and not (tmp_path / "o").exists()


def test_an_uncaught_error_exits_4_never_0_or_1(tmp_path, monkeypatch):
    monkeypatch.setattr(IP, "run_refusal", lambda: "")
    monkeypatch.setattr(PR, "regcheck", lambda args, vs: (_ for _ in ()).throw(KeyError("planted")))
    rc = PR.main(["--mode", "regcheck", "--variants", str(_variants(tmp_path)), "--out", str(tmp_path / "o")])
    assert rc == exit_codes.ERROR == 4


def test_one_variant_that_raises_costs_only_its_own_line(tmp_path, monkeypatch):
    vs = PR.load_variants(_variants(tmp_path))
    monkeypatch.setattr(PR, "Card", lambda args: object())

    def rc(card, args, v):
        if v["id"] == "u8-sw":
            raise RuntimeError("planted")
        return True, "", {"ok": True}
    monkeypatch.setattr(PR, "regcheck_variant", rc)
    assert PR.regcheck(argparse.Namespace(out=tmp_path), vs) == exit_codes.CLAIM_FAIL
    env = (tmp_path / "regcheck.env").read_text().splitlines()
    assert env[0] == "REGCHECK_u5_sd=PASS" and env[3].startswith("REGCHECK_u8_sw=FAIL the variant raised RuntimeError")
    assert len(env) == 6


def test_writes_are_atomic(tmp_path):
    np = pytest.importorskip("numpy")
    PR.atomic_write_text(tmp_path / "stamps.json", "{}")
    PR.atomic_save(tmp_path / "x-call0-w1.npy", np.arange(3))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stamps.json", "x-call0-w1.npy"]
    import gen_pc2_pages as GEN
    GEN.atomic_write(tmp_path / "a" / "b.json", "x")
    assert (tmp_path / "a" / "b.json").read_text() == "x" and not list((tmp_path / "a").glob("*.tmp"))


# --------------------------------------------------------------------------
# the unseen-model guard
# --------------------------------------------------------------------------

def test_nothing_in_the_plan_loads_a_reserved_model_its_shape_or_its_gemm():
    loaded = set()
    for page in P.BLOCKS:
        loaded |= {c["shape"] for c in cells(page).values()}
    plan = (REPO / P.PLAN).read_text()
    body = [ln.split("#", 1)[0].split() for ln in plan.splitlines() if ln.split("#", 1)[0].strip()]
    loaded |= {w[0] for w in body if w[0] != "-"}
    loaded |= {m for w in body for kv in w[2:] if kv.startswith("models=") for m in kv[7:].split(",")}
    assert loaded and not loaded & set(P.RESERVED)
    rshape, rg = P.reserved_shapes(), P.reserved_gemms()
    for m in loaded:
        cfg = P.shape_cfg(m)
        assert P.shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) not in rshape, m
        assert not any(tuple(P.gemm_dims(cfg, g)) in rg for g in ("w1", "w2")), m
    assert loaded <= set(P.REAL_ALLOWED) | {s["name"] for s in P.SHAPES} | set(P.CENSUS_MODELS)
    blob = plan + json.dumps(REG["cells (COMPUTED)"]) + json.dumps(REG["pages"])
    for word in ("8x22b", "qwen2-57b", "granite-3.0-1b"):
        assert word not in blob, word


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


def test_a_synthetic_shape_with_a_reserved_gemm_is_refused_at_build():
    g = MODEL_CONFIGS["granite-3.0-1b-a400m"]
    fake = {"pcs2-g": MoEConfig(name="pcs2-g", hidden_size=g.hidden_size, intermediate_size=g.intermediate_size,
                                num_experts=g.num_experts + 8, top_k=g.top_k)}
    with pytest.raises(P.Refused, match="granite-3.0-1b-a400m's"):
        P.shape_cfg("pcs2-g", fake)


# --------------------------------------------------------------------------
# the decode tile's regcheck literal, the readback, a naive cell on the CPU
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PC_RCP.is_file(), reason="PC's published regcheck page is not in this checkout")
def test_pcs_decode_tile_regcheck_literal_recomputes_from_its_published_page():
    v = json.loads(PC_RCP.read_text())["variants"]["u17-sd"]
    got = {}
    for row in v["rows"]:
        bm, bn, bk, _g, st, w, mul, _k = row["config"]
        t = P.tile_key((bm, bn, bk, w, st))
        copy = next(iter(row["specs"].values()))["copy"]
        got.setdefault(t, {})["w2" if mul else "w1"] = (row["plain"]["ctas_per_sm"], copy["ctas_per_sm"])
    assert got == P.PC_D16_REGCHECK
    for t in ("16x64x128w4s4", "16x64x128w4s3"):
        assert all(a == b == P.OCCUPANCY[t] for a, b in got[t].values())


def test_the_readback_composes_r3_decodes_rules_on_naive_and_aligned_cells():
    for page in ("pc2-n", "pc2-d"):
        for c in cells(page).values():
            conf = PR.conf_of(c["tile"])
            dc = RD.DecodeCell(label=c["label"], T=c["T"], counts=tuple(c["counts"]), arms=(RD.NATIVE,))
            ids = RD.cell_ids(c["cfg"], dc, P.SHUFFLE_SEED)
            exp = RD.expected_readback(c["cfg"], c["T"], c["counts"], ids.reshape(-1).tolist(),
                                       declared=c["cfg"].num_experts, copies=1, conf=conf)
            naive = exp["path"] == RD.NAIVE
            assert naive == (c["path"] == "naive"), c["label"]
            rec = {"gemms": [{"sorted": None if naive else list(range(exp["numel"])) + [exp["numel"]] * (
                exp["sorted_len"] - exp["numel"]), "expert_ids": exp["expert_order"], "post_padded": exp["post_padded"],
                "grid": g, "use_td": False, "config": dict(conf)} for g in exp["grid"]]}
            assert RD.check_readback(rec, exp) == [], c["label"]
            if naive:
                assert exp["grid"] == [P.live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")]


def test_a_naive_cell_builds_its_ids_and_call_on_the_cpu():
    torch = pytest.importorskip("torch")
    card = PR.Card.__new__(PR.Card)
    card.device, card.torch, card.model, card.shapes = "cpu", torch, None, P.load_shapes()
    card.args = argparse.Namespace(seed=0)
    got = []
    card.fused_experts = lambda **kw: got.append(kw) or "out"
    card.activation = lambda: "silu"
    v = PR.check_variant(pv(id="u7-sn", page=f"{P.PAGES_DIR}/pc2-n.json"))
    c = next(x for x in v["cells"] if x["label"] == "N-K8-T2-A8")
    fn, ids = card.cell_fn(v, c)
    assert torch.bincount(ids.reshape(-1).long(), minlength=64).tolist() == c["counts"]
    assert all(len(set(r.tolist())) == 8 for r in ids) and tuple(ids.shape) == (2, 8)
    assert fn() == "out" and got[0]["global_num_experts"] == 64
    assert tuple(card.w1.shape) == (64, 2 * 1408, 2048) and tuple(card.w2.shape) == (64, 2048, 1408)


# --------------------------------------------------------------------------
# the pinned files
# --------------------------------------------------------------------------

def test_the_pinned_files_the_probe_imports_are_untouched():
    import test_pinned_files as TP
    for rel in ("moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == TP.BASELINE[rel], rel
    for rel in ("scripts/instr_probe.py", "scripts/instr_probe_c3.py", "scripts/instr_probe_pc.py",
                "scripts/scoring/pc/pccommon.py", "scripts/scoring/pc/read_pc.py", "scripts/gh200_pc_session.sh",
                "scripts/vm_run.sh", "scripts/r3_decode.py", "scripts/private_weight_reference.py", "moe/spec.py"):
        h = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        assert PC_REG["code_pins"][rel] == h == REG["code_pins"][rel], rel
    assert REG["instrument"]["copy_sha256"] == TP.BASELINE["moe/instrumented/fused_moe_instr.py"]
