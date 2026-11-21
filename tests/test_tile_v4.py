"""Model v4 off the GPU (scripts/r3_timing_model_v4.py, scripts/wave_split_bytes_v4.py; registration
2026-10-14-tileA-floor): v4 at R3's tile is v3 to 0 on every registered v3 prediction, the c and F
units of the review's fix 1a, the wgmma counts against the SEEN H200 PTX, the accountings' pipe and
Z_wg, the dead-CTA increments of the review's table, and the A keys' floor."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "tileA", REPO / "scripts" / "scoring" / "rental6"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import wave_split_bytes_v4 as WV4  # noqa: E402

R3_PTX = ("results/published/2026-09-10-nvidia_h200-gaps-session/results/bm128_depth/"
          "nvidia_h200-budget400.0-dtypebf16-g1-iters50-k64-l2flushtrue-modelmixtral_8x7b-n64-r2048-referen-3c34b7a0/"
          "triton-cache/bm32/7INWIJHYBZG425JTKZNWSANE4VVXELRKFBVEEC3D4MCGB5QQYANQ/fused_moe_kernel.ptx")


def test_v4_at_r3s_tile_reproduces_every_registered_v3_prediction_to_zero():
    """design-tile-review 1a: equality with the registered numbers (6 decimals) and with rental 6's
    pricer bit for bit, on all 300 predictions of rental6-skew and rental7-skew."""
    got = V4.check()
    assert got["compared"] == 300, got["compared"]
    assert not got["unequal_registered"], got["unequal_registered"][:3]
    assert not got["unequal_pricer"], got["unequal_pricer"][:3]


def test_at_r3s_tile_c_scale_is_exactly_one_and_phi_is_v3s_own():
    prm = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})
    for g in ("w1", "w2"):
        assert V4.c_scale(V4.R3_TILE, g, prm) == 1.0
        assert V4.phi(V4.R3_TILE, g, prm) is TM.CTA_FIXED_KSTEPS[g] or V4.phi(V4.R3_TILE, g, prm) == TM.CTA_FIXED_KSTEPS[g]
        assert V4.c_counter(V4.R3_TILE, g, prm) == TM.C_CYC_PER_KSTEP == 344.1


def test_c_time_uses_the_counter_c_and_f_is_held_in_counter_cycles():
    """c_time = c_card x c_counter / 344.1, PHI = F / c_counter: at the calibration tile both F forms
    give F / c_128; at 64x128 F-step holds PHI and F-cyc holds F."""
    prm = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, variant="W")
    assert V4.c_scale(V4.CAL_TILE, "w1", prm) == pytest.approx(900.0 / 344.1)
    for form in V4.F_FORMS:
        p = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, variant="W", f_form=form)
        assert V4.phi(V4.CAL_TILE, "w1", p) == pytest.approx(520.0 / 900.0)
    c64 = V4.c_counter(V4.T64W8, "w1", prm)
    step = V4.phi(V4.T64W8, "w1", prm)
    cyc = V4.phi(V4.T64W8, "w1", V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, variant="W", f_form="F-cyc"))
    assert step == pytest.approx(520.0 / 900.0) and cyc == pytest.approx(520.0 / c64) and step != cyc
    assert V4.F_FORM_REGISTERED == "F-step"


def test_the_v3own_rival_is_v3s_constants_on_the_tiles_geometry():
    p = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, law="v3own")
    for t in (V4.CAL_TILE, V4.T64W8):
        assert V4.c_scale(t, "w1", p) == 1.0 and V4.phi(t, "w2", p) == TM.CTA_FIXED_KSTEPS["w2"]


def test_the_layout_rule_matches_the_seen_h200_ptx_and_the_bytes_per_kstep():
    import regtile as RG
    import tile_census as TCEN
    seen = {key: TCEN.ptx_counts((REPO / path).read_text(), V4.Tile.parse(key)) for path, key in RG.SEEN_PTX}
    for key, inst in (("128x128x64w8s4", "m64n128k16"), ("64x128x64w8s4", "m64n64k16")):
        t = V4.Tile.parse(key)
        est = V4.wgmma_counts_estimate(t)
        assert est["instruction"] == seen[key]["instruction"] == inst
        assert est["per_warpgroup"] == seen[key]["per_warpgroup"] == 4 and est["warpgroups"] == 2
        assert seen[key]["bar_sync"] == 2 and seen[key]["fence_proxy_async"] == 1 and seen[key]["wgmma_wait_group"] == 1
        assert V4.wgmma_bytes(t, seen[key]) == V4.wgmma_bytes(t, est)
        assert seen[key]["cp_async_bytes_per_thread"] * 256 == V4.wgmma_bytes(t)["writes"]
    assert V4.wgmma_bytes(V4.CAL_TILE)["reads"] == 49152 and V4.wgmma_bytes(V4.CAL_TILE)["writes"] == 32768
    assert V4.wgmma_bytes(V4.T64W8)["reads"] == 32768 and V4.wgmma_bytes(V4.T64W8)["writes"] == 24576
    assert V4.wgmma_bytes(V4.T64W4)["reads"] == 24576
    assert seen["32x64x64w8s4"]["wavefronts"] == 280.0 == V4.wavefronts_estimate(V4.R3_TILE)


def test_the_shared_rule_gives_the_design_occupancies_and_the_paths():
    assert (V4.CAL_TILE.shared_bytes(), V4.CAL_TILE.occ_shared()) == (98304, 2)
    assert (V4.T64W8.shared_bytes(), V4.T64W8.occ_shared()) == (73728, 3)
    assert V4.R3_TILE.shared_bytes() == 36864 and V4.R3_TILE.path == "mma.sync"
    assert [t.path for t in V4.VLLM_TILES] == ["mma.sync"] * 3 + ["wgmma"] * 3
    assert V4.Tile.parse("128x128x64w8s3") == V4.CAL_TILE and V4.CAL_TILE.key == "128x128x64w8s3"


def test_eta_mma_is_luos_table_11_per_clock_and_the_rival_table_9():
    """build-tile-review 1: per clock primary (0.991 carried by Table 9's within-table ratios), the
    chip-relative Table 9 rows the rival."""
    assert V4.ETA_PER_CLOCK_N256 == pytest.approx(0.991, abs=5e-4)
    assert V4.eta_mma_of(V4.CAL_TILE) == pytest.approx(0.983, abs=5e-4)
    assert V4.eta_mma_of(V4.T64W8) == pytest.approx(0.965, abs=5e-4)
    assert V4.eta_mma_of(V4.CAL_TILE, rival=True) == pytest.approx(659.8 / 756.5)
    assert V4.eta_mma_of(V4.T64W8, rival=True) == pytest.approx(648.3 / 756.5)


@pytest.mark.parametrize("variant", V4.VARIANTS)
def test_z_wg_puts_the_calibration_tile_at_c128_and_b_is_unreachable_low(variant):
    prm = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, variant=variant)
    z = V4.z_wg(prm)
    assert z is not None and V4.c_counter(V4.CAL_TILE, "w1", prm) == 900.0
    D = V4.pipe_service(V4.CAL_TILE, variant, V4.eta_mma_of(V4.CAL_TILE))["D"]
    assert V4.mva(D, z, 2) == pytest.approx(900.0, rel=1e-9)
    low = V4.Params(c128=600.0, F_wg={"w1": 520.0, "w2": 979.0}, variant="B")
    assert V4.z_wg(low) is None
    with pytest.raises(TM.Refused):
        V4.c_counter(V4.T64W8, "w1", low)


def test_the_b_accounting_matches_the_reviews_variant_table():
    """design-tile-review work/variant_effect.txt: B at c_128 900 gives 584 (w8) and 515 (w4) at any
    eta (B is not tensor-bound there)."""
    prm = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0}, variant="B")
    assert round(V4.c_counter(V4.T64W8, "w1", prm)) == 584 and round(V4.c_counter(V4.T64W4, "w1", prm)) == 515


def test_the_dead_increments_reproduce_the_reviews_table():
    """design-tile-review work/dead_lever.txt at c_128 600 / 900 / 1100 (OLMoE 128 G 1, occupancy 2)."""
    import regtile as RG
    want = {600.0: {"D": 12.13, "SLOT": 24.69, "kappa 0": 12.13, "d 1.06": 12.92, "D2": 11.95},
            900.0: {"D": 10.02, "SLOT": 24.69, "kappa 0.15": 12.13, "d 1.06": 11.41, "D2": 10.10},
            1100.0: {"D": 7.57, "SLOT": 24.69, "kappa 0": 12.13, "d 1.06": 8.96, "D2": 7.84}}
    for c, laws in want.items():
        got = RG.dead_increments(c)
        for k, v in laws.items():
            assert got[k] == pytest.approx(v, abs=0.006), (c, k, got[k])


def test_clamp_floors_q_at_the_active_weights_once_and_the_row_key_replaces_w1_only():
    q = {"w1": 0.967, "w2": 1.10}
    with V4.at_tile("qwen1.5-moe-a2.7b", V4.CAL_TILE):
        v3 = WV4.keyed_reads("v3", q, 1, 1.0, None)
        cl = WV4.keyed_reads("CLAMP", q, 1, 1.0, None)
        rk = WV4.keyed_reads("ROWKEY60", q, 1, 1.0, 5.0e8)
        W1 = TM.BYTE_MODEL["W_w1"]
    assert cl["w1"] - v3["w1"] == pytest.approx((1.0 - 0.967) * W1) and cl["w2"] == v3["w2"]
    assert rk["w1"] == 5.0e8 and rk["w2"] == v3["w2"]
    assert WV4.keyed_reads("ROWKEY30", q, 1, 1.0, None) is None
    assert WV4.q_ideal("olmoe-1b-7b", 1, [0] * 32 + [1] * 32, 128) == 0.5


def test_at_tile_restores_r3s_tile_and_the_model():
    old = TM.MODEL
    keep = (TM.BLOCK_M, TM.BLOCK_N, TM.BLOCK_K, TM.OUT_TILE, dict(TM.CTA_FIXED_KSTEPS))
    with V4.at_tile("olmoe-1b-7b", V4.CAL_TILE, {"w1": 0.5, "w2": 0.9}):
        assert (TM.BLOCK_M, TM.BLOCK_N, TM.GEOMETRY["w1"].npn, TM.GEOMETRY["w1"].ksteps) == (128, 128, 16, 32)
        assert TM.CTA_FIXED_KSTEPS == {"w1": 0.5, "w2": 0.9}
    assert (TM.BLOCK_M, TM.BLOCK_N, TM.BLOCK_K, TM.OUT_TILE, dict(TM.CTA_FIXED_KSTEPS)) == keep and TM.MODEL == old


def test_geometry_at_r3s_tile_is_the_engines_own_and_deepseek_gets_one_without_r3s_ladder():
    import regtile as RG
    pr = RG.pricer()
    g = pr.eng.geom("olmoe-1b-7b", 9)
    assert WV4.geometry_at(g, V4.R3_TILE, V4.OCC_R3) is g
    t = WV4.geometry_at(g, V4.CAL_TILE, {"w1": 2, "w2": 2})
    assert (t.block_m, t.block_n, dict(t.W_c)["w1"], dict(t.P)["w1"]) == (128, 128, 264, 16)
    d = WV4.base_geometry(pr.eng, "deepseek-v2-lite", 9)
    assert d.experts == 64 and dict(d.declared) == {"native": 64, "private": 576, "shared": 576}
