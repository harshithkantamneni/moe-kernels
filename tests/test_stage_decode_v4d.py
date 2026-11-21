"""Model v4d off the GPU (scripts/r3_timing_model_v4d.py): its bytes are A W_e plus activations, not a
function of T; naive repeats fetch once; the rho rival pair; padded rows cost 0; the dead term is 0 on
the naive path; the CAL constants' closed-form fit recovers planted constants; E1-DEC's 1 us floor;
v4 and the pinned v3 modules are restored after every price."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_decode as RD  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r3_timing_model_v4d as V4D  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402

NOM = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})


def balanced(model, T, A=None):
    cfg = MC[model]
    n, E = T * cfg.top_k, cfg.num_experts
    A = A or min(E, n)
    return [n // A + (1 if e < n % A else 0) if e < A else 0 for e in range(E)]


@pytest.mark.parametrize("model, T", [("olmoe-1b-7b", 8), ("olmoe-1b-7b", 32), ("mixtral-8x7b", 16), ("phi-3.5-moe", 64)])
def test_the_weight_bytes_are_a_w_e_at_one_tile_per_expert(model, T):
    p = V4D.price(model, T, balanced(model, T), V4D.DecodeParams(tile=NOM))
    cfg = MC[model]
    W = p["A"] * RD.expert_bytes(cfg)
    # every live CTA fetches its slab once: the read bytes are the weights plus a sliver of activations
    assert W <= sum(p["reads"].values()) <= W * 1.02


def test_the_bytes_do_not_move_with_t_inside_a_tile_band():
    a = V4D.price("mixtral-8x7b", 8, balanced("mixtral-8x7b", 8), V4D.DecodeParams(tile=NOM))
    b = V4D.price("mixtral-8x7b", 32, balanced("mixtral-8x7b", 32), V4D.DecodeParams(tile=NOM))
    assert a["tile"] == b["tile"] and abs(sum(a["reads"].values()) / sum(b["reads"].values()) - 1) < 0.01


def test_naive_repeats_fetch_once_and_the_nodedup_rival_does_not():
    p = V4D.price("olmoe-1b-7b", 1, [1] * 8 + [0] * 56, V4D.DecodeParams(tile=NOM))
    assert p["path"] == RD.NAIVE and p["dead_ctas"] == [0, 0]
    rep = [2] * 8 + [0] * 56              # olmoe T 2 (naive): both tokens on the same 8 experts
    a = V4D.price("olmoe-1b-7b", 2, rep, V4D.DecodeParams(tile=NOM))
    b = V4D.price("olmoe-1b-7b", 2, rep, V4D.DecodeParams(tile=NOM, naive_dedup=False))
    assert a["path"] == RD.NAIVE and b["reads"]["w1"] > 1.9 * a["reads"]["w1"] and b["body_ms"] > a["body_ms"]


def test_rho_scaled_is_twice_fixed_at_the_small_decode_tile():
    t = V4.Tile(16, 64, 128, 4, 4)
    assert V4D.rho_of(t, "scaled") == pytest.approx(2.0 * V4D.rho_of(t, "fixed"))
    assert V4D.rho_of(V4.Tile(32, 64, 64, 8, 4), "scaled") == pytest.approx(V4D.RHO_GBPS)
    # the pair separates on a sub-wave cell and on nothing many-wave
    sub = V4D.price("granite-3.0-3b-a800m", 1, [1] * 8 + [0] * 32, V4D.DecodeParams(tile=NOM))
    assert sub["sub_wave"]
    s2 = V4D.price("granite-3.0-3b-a800m", 1, [1] * 8 + [0] * 32, V4D.DecodeParams(tile=NOM, rho_form="scaled"))
    assert s2["body_ms"] < sub["body_ms"]
    assert TM.CORES_RHO_GBPS == V4D.RHO_GBPS


def test_padded_rows_cost_nothing_unless_the_fill_rival_prices_them():
    a = V4D.price("olmoe-1b-7b", 16, balanced("olmoe-1b-7b", 16), V4D.DecodeParams(tile=NOM))
    b = V4D.price("olmoe-1b-7b", 16, balanced("olmoe-1b-7b", 16), V4D.DecodeParams(tile=NOM, fill_r_us=0.001))
    assert b["body_ms"] - a["body_ms"] == pytest.approx(a["P"] * 1e-6)


def test_the_tile_globals_are_restored_after_a_price():
    keep = (TM.BLOCK_M, TM.BLOCK_N, TM.BLOCK_K, TM.MODEL, dict(TM.CTA_FIXED_KSTEPS), TM.CORES_RHO_GBPS)
    V4D.price("qwen1.5-moe-a2.7b", 64, balanced("qwen1.5-moe-a2.7b", 64), V4D.DecodeParams(tile=NOM, rho_form="scaled"))
    assert (TM.BLOCK_M, TM.BLOCK_N, TM.BLOCK_K, TM.MODEL, dict(TM.CTA_FIXED_KSTEPS), TM.CORES_RHO_GBPS) == keep


def test_the_wgmma_decode_tile_is_not_priced_without_the_tile_stages_constants():
    with pytest.raises(TM.Refused):
        V4D.price("olmoe-1b-7b", 128, balanced("olmoe-1b-7b", 128), V4D.DecodeParams())
    assert V4D.price("olmoe-1b-7b", 64, balanced("olmoe-1b-7b", 64), V4D.DecodeParams())["body_ms"] > 0


def test_the_cal_fit_recovers_planted_constants_exactly():
    T0, sn, sb = 0.031, -0.004, 0.006
    rows = []
    for model, T in (("olmoe-1b-7b", 1), ("olmoe-1b-7b", 2), ("olmoe-1b-7b", 8), ("olmoe-1b-7b", 128),
                     ("mixtral-8x7b", 1), ("mixtral-8x7b", 16), ("mixtral-8x7b", 64), ("olmoe-1b-7b", 32)):
        p = V4D.price(model, T, balanced(model, T), V4D.DecodeParams(tile=NOM))
        const = T0 + {RD.SMALL_BATCH: 0.0, RD.NAIVE: sn, RD.BLOCK_SCAN: sb}[p["path"]]
        rows.append({"meas_ms": p["body_ms"] + const, "body_ms": p["body_ms"], "path": p["path"]})
    f = V4D.fit_constants(rows)
    assert f["verdict"] == "FITTED"
    assert (f["T0_GR_ms"], f["s_naive_GR_ms"], f["s_block_GR_ms"]) == pytest.approx((T0, sn, sb), abs=1e-12)
    only_small = [r for r in rows if r["path"] == RD.SMALL_BATCH]
    g = V4D.fit_constants(only_small)
    assert g["T0_GR_ms"] == pytest.approx(T0) and set(g["not_fitted"]) == {RD.NAIVE, RD.BLOCK_SCAN}


def test_e1_decs_error_floors_a_microsecond_at_zero():
    assert V4D.rel_err(0.0505, 0.0500) == 0.0
    assert V4D.rel_err(0.0520, 0.0500) == pytest.approx(0.04)
    assert V4D.rel_err(0.0480, 0.0500) == pytest.approx(-0.04)


def test_the_eager_rival_is_v3s_eager_constants():
    e = V4D.eager_params(V4D.DecodeParams(T0_GR_ms=0.02, tile=NOM))
    assert e.T0_GR_ms == pytest.approx(V4D.CARD["T0_eager_ms"] + V4D.CARD["s_small_eager_ms"])
    assert e.s_block_GR_ms == pytest.approx(V4D.CARD["s_block_eager_ms"] - V4D.CARD["s_small_eager_ms"])


@pytest.mark.parametrize("model, T, counts", [
    ("olmoe-1b-7b", 16, [16] * 8 + [0] * 56), ("mixtral-8x7b", 16, [16, 16] + [0] * 6), ("mixtral-8x7b", 32, [8] * 8),
    ("olmoe-1b-7b", 8, [1] * 64), ("olmoe-1b-7b", 64, [8] * 64), ("mixtral-8x7b", 8, [2] * 8),
    ("olmoe-1b-7b", 64, [64] * 8 + [0] * 56), ("mixtral-8x7b", 96, [24] * 8)])
def test_v4d_reduces_to_v4_on_full_and_partial_tiles_at_sigma_1_and_g_1(model, T, counts):
    """build-decode review 3: v4d's GEMM terms are v4's own composition (TM.window at G 1, gemm_ms at
    sigma 1, v2's dead term) and its body adds only the decode B_other, to rounding. v4d defines no
    other G: vLLM's decode config has GROUP_SIZE_M 1 at every T (a G-16 cell would need T // E > 128)."""
    cfg = MC[model]
    prm = V4D.DecodeParams(tile=NOM)
    p = V4D.price(model, T, counts, prm)
    t = V4D.tile_of(RD.default_config(T, cfg.num_experts))
    phis = {g: V4.phi(t, g, prm.tile) for g in TM.GEMMS}
    occ = {g: prm.tile.occ(t, g) for g in TM.GEMMS}
    ct = tuple(counts)
    with V4.at_tile(model, t, phis), V4D._rho(V4D.rho_of(t, "fixed")), TM.dead_model("v2"):
        for i, g in enumerate(TM.GEMMS):
            cg = prm.c_ns * V4.c_scale(t, g, prm.tile)
            win = TM.window("native", cfg.num_experts, 1, 1, g, TM.window_width(prm.k_w, prm.sms, occ[g]), ct)
            assert p["gemm"][g]["gemm_ms"] == TM.gemm_ms(win, g, 1.0, cg, prm.bw_gbps, prm.sms, occ=occ[g])
            hidden = TM.V2_KAPPA * occ[g] * TM.floor_ksteps(g) * cg
            assert p["gemm"][g]["dead_ms"] == max(0.0, p["dead_ctas"][i] * TM.V2_DEAD_NS - hidden) * 1e-6
            if T >= t.bm and sum(counts) >= cfg.num_experts:      # no cap, no EM cut: v4's own dead count
                assert p["dead_ctas"][i] == TM.dead_ctas(cfg.num_experts, 1, g, ct)
                assert p["gemm"][g]["dead_ms"] == TM.dead_ms(cfg.num_experts, 1, g, cg, prm.k_w, occ[g], ct)
    assert all(x["live_ctas"] > 0 for x in p["gemm"].values())
    assert RD.default_config(T, cfg.num_experts)["GROUP_SIZE_M"] == 1
