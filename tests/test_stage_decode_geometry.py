"""The decode stage's geometry and readback rules off the GPU (scripts/r3_decode.py): vLLM v0.27.1's
assignment path, buffer, EM cut, grid, live and dead tiles against design-decode's own table
(decode_geom.txt, its rows typed in here) and against the quoted upstream lines; V-ALIGN and GD5
on planted spy records; the page's refusals; GRR's rotation rule; V-BIN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_fakes as F  # noqa: E402
import r3_decode as RD  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402


def balanced(model, T):
    cfg = MC[model]
    n, E = T * cfg.top_k, cfg.num_experts
    return [n // E + (1 if e < n % E else 0) for e in range(E)]


#: design-decode work/decode_geom.txt (BALANCED rows): (model, T, tile, path, L, emtiles_w1, dead_w1, live_w1)
DESIGN_ROWS = [
    ("mixtral-8x7b", 1, "16x64x128w4s4", "naive", 2, 2, 0, 896),
    ("mixtral-8x7b", 8, "16x64x128w4s4", "small-batch", 8, 9, 448, 3584),
    ("mixtral-8x7b", 96, "32x128x64w4s3", "small-batch", 8, 14, 1344, 1792),
    ("mixtral-8x7b", 128, "64x128x64w4s3", "small-batch", 8, 12, 896, 1792),
    ("olmoe-1b-7b", 2, "16x64x128w4s4", "naive", 16, 16, 0, 512),
    ("olmoe-1b-7b", 16, "16x64x128w4s4", "small-batch", 64, 68, 128, 2048),
    ("olmoe-1b-7b", 128, "64x128x64w4s3", "block-scan", 64, 79, 240, 1024),
    ("qwen1.5-moe-a2.7b", 2, "16x64x128w4s4", "naive", 8, 8, 0, 352),
    ("qwen1.5-moe-a2.7b", 16, "16x64x128w4s4", "small-batch", 60, 61, 44, 2640),
    ("phi-3.5-moe", 32, "16x64x128w4s4", "small-batch", 16, 19, 600, 3200),
    ("jetmoe-8b", 96, "32x128x64w4s3", "small-batch", 8, 14, 528, 704),
]


@pytest.mark.parametrize("model, T, tile, path, L, em1, dead1, live1", DESIGN_ROWS)
def test_geometry_reproduces_the_designs_table(model, T, tile, path, L, em1, dead1, live1):
    g = RD.geometry(MC[model], T, balanced(model, T))
    assert (g.tile, g.path, g.L, g.grid[0] // g.npn[0], g.dead_ctas[0], g.live_ctas[0]) == (tile, path, L, em1, dead1, live1)


def test_the_default_tiles_are_the_four_decode_tiles():
    E = 64
    assert {RD.tile_key(RD.default_config(T, E)) for T in range(1, 129)} == {
        "16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3"}
    assert all(RD.default_config(T, E)["GROUP_SIZE_M"] == 1 for T in range(1, 129))


def test_the_rules_are_the_quoted_upstream_lines():
    up = json.loads(RD.UPSTREAM.read_text())["rules"]
    assert up["NAIVE"]["text"] == "and num_tokens * top_k_num * 4 <= global_num_experts" and RD.NAIVE_FACTOR == 4
    assert up["MAXPAD"]["text"] == "max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)"
    assert up["MAXPAD_CAP"]["text"] == "topk_ids.numel() * block_size, max_num_tokens_padded"
    assert up["SMALL_BATCH"]["text"] == "(topk_ids.numel() < 1024) && (num_experts <= 64);"
    assert up["EM_CUT"]["text"] == 'if A.size(0) < config["BLOCK_SIZE_M"]:'
    assert up["DEAD"]["text"] == "if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:"
    assert up["NAIVE_POST"]["text"] == 'topk_ids.numel() * config["BLOCK_SIZE_M"],'
    # the capped buffer at numel < E, and the naive cutover on each model
    assert RD.max_padded(8, 64, 16) == 8 * 16 and RD.max_padded(16, 8, 16) == 16 + 8 * 15
    naive_T = {m: max(T for T in range(1, 9) if RD.decode_path(T, MC[m].top_k, MC[m].num_experts) == RD.NAIVE)
               for m in ("mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b", "phi-3.5-moe", "jetmoe-8b",
                         "deepseek-v2-lite", "granite-3.0-3b-a800m")}
    assert naive_T == {"mixtral-8x7b": 1, "olmoe-1b-7b": 2, "qwen1.5-moe-a2.7b": 3, "phi-3.5-moe": 2, "jetmoe-8b": 1,
                       "deepseek-v2-lite": 2, "granite-3.0-3b-a800m": 1}


def test_the_upstream_record_matches_the_fetched_files_when_they_are_on_disk():
    rec = json.loads(RD.UPSTREAM.read_text())
    assert rec["files"]["fused_moe.py"]["sha256"] == json.loads(
        (REPO / "moe" / "instrumented" / "UPSTREAM.json").read_text())["file_sha256"]
    import hashlib
    d = Path.home() / "code" / "moe-kernels-work" / "2026-10-14" / "build-decode" / "upstream"
    names = {"fused_moe.py": "vllm_model_executor_layers_fused_moe_fused_moe.py",
             "moe_align_block_size.py": "vllm_model_executor_layers_fused_moe_moe_align_block_size.py",
             "moe_align_sum_kernels.cu": "csrc_libtorch_stable_moe_moe_align_sum_kernels.cu",
             "utils.py": "vllm_model_executor_layers_fused_moe_utils.py"}
    if not d.is_dir():
        pytest.skip("the fetched v0.27.1 sources are not on this machine")
    for k, f in names.items():
        b = (d / f).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["files"][k]["sha256"], k
        lines = b.decode().splitlines()
        for name, r in rec["rules"].items():
            if r["file"] == k:
                assert lines[r["line"] - 1].strip() == r["text"], name


def test_multi_tile_and_dead_ctas_on_mincover_past_bm():
    cfg = MC["mixtral-8x7b"]
    c = [32, 32, 0, 0, 0, 0, 0, 0]
    g = RD.geometry(cfg, 32, c)
    assert g.multi_tile_experts == 2 and g.L == 4 and g.P == 0 and g.A == 2
    assert g.dead_ctas[0] == (g.grid[0] // g.npn[0] - 4) * g.npn[0] > 0


def test_decl_moves_the_path_and_the_dead_ctas_only():
    cfg = MC["mixtral-8x7b"]
    c = balanced("mixtral-8x7b", 32)
    n, d = RD.geometry(cfg, 32, c), RD.geometry(cfg, 32, c, declared=8 * 9)
    assert (n.path, d.path) == ("small-batch", "block-scan") and n.L == d.L and n.P == d.P
    assert d.dead_ctas[0] > n.dead_ctas[0]
    assert RD.geometry(cfg, 4, balanced("mixtral-8x7b", 4), declared=72).path == RD.NAIVE


def _cell(model, T, counts, label="c", arms=("native",)):
    return RD.DecodeCell(label=label, T=T, counts=tuple(counts), arms=arms)


@pytest.mark.parametrize("model, T", [("olmoe-1b-7b", 1), ("olmoe-1b-7b", 16), ("olmoe-1b-7b", 128),
                                      ("mixtral-8x7b", 8), ("mixtral-8x7b", 96)])
def test_a_perfect_readback_passes_and_every_planted_mismatch_is_named(model, T):
    cfg = MC[model]
    cell = _cell(model, T, balanced(model, T))
    ids = RD.cell_ids(cfg, cell, 7)
    flat = ids.reshape(-1).tolist()
    exp = RD.expected_readback(cfg, T, cell.counts, flat, declared=cfg.num_experts, copies=1)
    rec = F.perfect_record(cfg, cell, flat, declared=cfg.num_experts, copies=1)
    assert RD.check_readback(rec, exp) == []
    for mutate, word in ((lambda r: r["gemms"][0].update(post_padded=r["gemms"][0]["post_padded"] + 1), "num_tokens_post_padded"),
                         (lambda r: r["gemms"][1].update(grid=r["gemms"][1]["grid"] + 1), "launch grid"),
                         (lambda r: r["gemms"][0]["config"].update(BLOCK_SIZE_M=2 * r["gemms"][0]["config"]["BLOCK_SIZE_M"]), "GD5"),
                         (lambda r: r["gemms"][0].update(use_td=True), "USE_TD"),
                         (lambda r: r["gemms"][0].update(expert_ids=list(reversed(r["gemms"][0]["expert_ids"]))), "expert_ids"),
                         (lambda r: r.update(tuned="a tuned file"), "tuned")):
        bad = F.perfect_record(cfg, cell, flat, declared=cfg.num_experts, copies=1)
        mutate(bad)
        got = RD.check_readback(bad, exp)
        assert got and any(word in g for g in got), (word, got)
    if exp["path"] != RD.NAIVE:
        bad = F.perfect_record(cfg, cell, flat, declared=cfg.num_experts, copies=1)
        bad["gemms"][0]["sorted"] = None
        assert any("sorted_token_ids is None" in g for g in RD.check_readback(bad, exp))
        bad = F.perfect_record(cfg, cell, flat, declared=cfg.num_experts, copies=1)
        s = bad["gemms"][0]["sorted"]
        s[s.index(exp["numel"])] = 0
        assert any("sentinel" in g for g in RD.check_readback(bad, exp))


def test_cell_ids_realise_the_counts_and_shuffle_only_rows():
    cfg = MC["olmoe-1b-7b"]
    c = balanced("olmoe-1b-7b", 16)
    a = RD.cell_ids(cfg, RD.DecodeCell("a", 16, tuple(c), ("native",), shuffle=True), 3)
    b = RD.cell_ids(cfg, RD.DecodeCell("b", 16, tuple(c), ("native",), shuffle=False), 3)
    import torch
    assert torch.bincount(a.reshape(-1).long(), minlength=64).tolist() == c
    assert sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist())) and not torch.equal(a, b)


@pytest.mark.parametrize("edit, why", [
    (lambda d: d.update(schema="x"), "schema"),
    (lambda d: d.update(model="mixtral-8x7b"), "this run is"),
    (lambda d: d["cells"][0].update(T=200), "not a decode batch"),
    (lambda d: d["cells"][0].update(counts=d["cells"][0]["counts"][:-1]), "whole numbers"),
    (lambda d: d["cells"][1].update(label=d["cells"][0]["label"]), "twice"),
    (lambda d: d["cells"][0].update(arms=["decl"]), "native, and optionally decl"),
    (lambda d: d["cells"][0].update(arms=["native", "decl"]), "decl_copies"),
    (lambda d: d["cells"][3].update(counts=[d["cells"][3]["T"] * 8] + [0] * 63), "V-BIN"),
])
def test_the_page_refuses_what_the_card_would_meet(edit, why):
    doc = json.loads((REPO / "docs/registered/2026-10-14-decode-pages/olmoe-1b-7b-main.json").read_text())
    edit(doc)
    with pytest.raises(RD.DecodeRefused, match=why):
        RD.page_from(doc, MC["olmoe-1b-7b"], "olmoe-1b-7b")


def test_grr_rotation_makes_every_call_cold():
    for model, R in (("mixtral-8x7b", 1), ("olmoe-1b-7b", 3), ("granite-3.0-3b-a800m", 7), ("qwen1.5-moe-a2.7b", 4)):
        cfg = MC[model]
        page = RD.load_page(REPO / f"docs/registered/2026-10-14-decode-pages/{model}-main.json", cfg, model)
        got, calls, _ = RD.rotation(cfg, page)
        a_min = min(sum(1 for v in c.counts if v) for c in page.cells)
        assert got == R and got * a_min * RD.expert_bytes(cfg) >= RD.GRR_COLD_MIB * 2 ** 20
        assert (got - 1) * a_min * RD.expert_bytes(cfg) < RD.GRR_COLD_MIB * 2 ** 20
        assert calls % got == 0 and calls >= RD.GRR_MIN_CALLS
        with pytest.raises(RD.DecodeRefused):
            RD.rotation(cfg, page, override=got - 1) if got > 1 else (_ for _ in ()).throw(RD.DecodeRefused("x"))
