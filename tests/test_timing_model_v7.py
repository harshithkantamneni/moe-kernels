"""Model v7 off the GPU (scripts/r3_timing_model_v7.py), every check in this process except the prefill chain's
(tests/test_timing_model_v7_prefill.py runs that in its own interpreter). The regime switch refuses a tile outside its
keys; the RR term's miss counter (the owner's piece, miss_fraction) passes the six hand-worked tests, reproduces RR's
reader on every READ launch of RR's published stamps and is unmoved by an absolute time base; the launch layer and
miss_fraction agree while every slot is busy; v7 at R3's tile re-prices the 300 registered v3 predictions bit for bit
with neither new term ever called; at decode, v7 with D6FIX off is v6d exactly (gate_d6's CAL2 and re-score rows),
the fix's CAL2 fit is the committed one, and D6FIX moves only the cells v6d prices by v4d''s naive GEMMs. No gate
statistic of v7 is computed here (the gate runs only after its registration is public)."""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "decode", REPO / "scripts" / "scoring" / "rr"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_timing_model_v7 as V7  # noqa: E402
from r3_timing_model_v7 import MIB, miss_fraction  # noqa: E402

PUB = REPO / "results" / "published"


# ---------------------------------------------------------------- the regime switch

def test_every_declared_tile_has_one_branch_and_any_other_tile_is_refused():
    assert V7.regime_of("32x64x64w8s4") == "r3"
    assert V7.regime_of("128x128x64w8s3") == V7.regime_of("64x128x64w8s3") == "prefill"
    for t in ("16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3"):
        assert V7.regime_of(t) == "decode"
    for t in ("128x64x64w8s3", "64x128x64w4s4", "32x64x64w4s4"):
        with pytest.raises(V7.Refused):
            V7.regime_of(t)
    with pytest.raises(V7.Refused):
        V7.price({"tile": "128x64x64w8s3"})


def test_the_rr_term_acts_only_on_prefill_tiles_at_group_size_m_1():
    assert V7.rr_applies("128x128x64w8s3", 1) and V7.rr_applies("64x128x64w8s3", 1)
    assert not V7.rr_applies("128x128x64w8s3", 16)
    assert not V7.rr_applies("32x64x64w8s4", 1)
    assert not any(V7.rr_applies(t, 1) for t in V7.DECODE_TILES)


def test_the_switches_and_constants_are_the_registered_ones():
    assert V7.V7.as_dict() == {"RR": True, "D6FIX": True} and V7.OFF.as_dict() == {"RR": False, "D6FIX": False}
    assert set(V7.RUNGS) == {"off", "RR alone", "D6FIX alone", "v7 (all)"}
    c = V7.rr_constants()
    assert (c["form"], c["C_mib"], c["sigma"], c["reps"], c["seed"]) == ("F2", 47.5, {"128": 0.08, "64": 0.06}, 6, 7)
    assert c["W_c"] == {"128": 264, "64": 396}
    assert re.fullmatch(r"[0-9a-f]{64}", V7.constants_sha256())


# ---------------------------------------------------------------- miss_fraction: the owner's six tests

def lockstep(waves: int, W: int, life: float = 1.0):
    """W CTAs per wave, wave k lives [k life, (k + 1) life)."""
    s = np.repeat(np.arange(waves) * life, W).astype(float)
    return s, s + life


def test_same_wave_partner_hits():
    # 8 CTAs in one wave, all start at 0: d = 0 for every pair, under either form -> no miss
    s, e = lockstep(1, 8)
    assert miss_fraction(s, e, np.full(8, 1 * MIB), pairs=np.array([4, 5, 6, 7]), P=4, C_mib=3, form="F1") == 0.0
    assert miss_fraction(s, e, np.full(8, 1 * MIB), pairs=np.array([4, 5, 6, 7]), P=4, C_mib=3, form="F2") == 0.0


def test_next_wave_partner_misses():
    # 2 waves of 4 CTAs, life 1 us, slab 1 MiB: the GPU streams 4 MiB per us. Reader i (wave 1) has partner i - 4
    # (wave 0): starts 1 us apart -> d = 4 MiB > 3 -> miss; mid-lives also 1 us apart -> 4 MiB -> miss
    s, e = lockstep(2, 4)
    for form in ("F1", "F2"):
        assert miss_fraction(s, e, np.full(8, 1 * MIB), pairs=np.array([4, 5, 6, 7]), P=4, C_mib=3, form=form) == 1.0


def test_threshold_is_strict():
    # same launch, C exactly 4 MiB: d == C is a HIT (read_rr.law_miss: miss iff d > C)
    s, e = lockstep(2, 4)
    assert miss_fraction(s, e, np.full(8, 1 * MIB), pairs=np.array([4, 5, 6, 7]), P=4, C_mib=4, form="F1") == 0.0


def test_forms_differ_when_lives_differ():
    # partner 0 lives [0, 2), reader 1 lives [1, 2); CTA 2 lives [0, 2) as background; slabs 2, 1, 2 MiB.
    # stream rates: CTA0 1 MiB/us, CTA1 1 MiB/us, CTA2 1 MiB/us.
    # F1: moments 0 and 1 -> bytes in [0, 1) = CTA0 1 + CTA2 1 = 2 MiB.
    # F2: mids 1.0 and 1.5 -> bytes in [1, 1.5) = 3 CTAs x 0.5 = 1.5 MiB.
    s = np.array([0.0, 1.0, 0.0])
    e = np.array([2.0, 2.0, 2.0])
    b = np.array([2, 1, 2]) * MIB
    assert miss_fraction(s, e, b, pairs=np.array([1]), P=1, C_mib=1.75, form="F1") == 1.0   # 2.0 > 1.75
    assert miss_fraction(s, e, b, pairs=np.array([1]), P=1, C_mib=1.75, form="F2") == 0.0   # 1.5 <= 1.75


def test_no_pairs():
    s, e = lockstep(1, 4)
    assert miss_fraction(s, e, np.full(4, MIB), pairs=np.array([], dtype=int), P=4) == 0.0


def test_unknown_form_refused():
    s, e = lockstep(1, 4)
    with pytest.raises(ValueError):
        miss_fraction(s, e, np.full(4, MIB), pairs=np.array([3]), P=1, form="F3")


# ---------------------------------------------------------------- miss_fraction: hygiene and the reader

def test_an_absolute_time_base_does_not_move_the_count():
    """The C3-R1 lesson: differences are taken on centered times. Lives on a 0.25 us grid sit exactly in float64 at a
    1.8e12 us (1.8e18 ns) base, so the shifted launch must read exactly as the unshifted one."""
    rng = np.random.default_rng(1)
    life = 1.0 + 0.25 * rng.integers(0, 4, 64)
    s = np.repeat(np.arange(8) * 1.5, 8)
    e = s + life
    b = np.full(64, 3 * MIB)
    pairs = np.arange(8, 64)
    for form in ("F1", "F2"):
        a = miss_fraction(s, e, b, pairs, P=8, C_mib=20.0, form=form)
        assert miss_fraction(s + 1.8e12, e + 1.8e12, b, pairs, P=8, C_mib=20.0, form=form) == a
    F = V7.bytes_streamed(s + 1.8e12, e + 1.8e12, b)
    assert F(1.8e12) == 0.0 and abs(F(1.8e12 + 100.0) - b.sum()) < 1e-6 * b.sum()


def test_int64_globaltimer_stamps_are_centered_in_integer_arithmetic_before_any_float():
    """Raw int64 globaltimer ns near 1.8e18 sit on a 256 ns float64 grid. Four 100 ns CTAs back to back (slabs 1 MiB,
    P 2, reader 2 against partner 0): F2's mid-lives are 200 ns apart and the GPU streams 0.5 + 1 + 0.5 = 2 MiB
    between them, a miss at C 1.5 MiB, so the count is 1.0 on any base. Cast to float64 before centering (the
    reviewed defect), the lives collapse (some to zero length) and the same launch read 0.0."""
    base = np.int64(1_800_000_000_000_000_037)                   # not a multiple of 256
    s = base + np.arange(4, dtype=np.int64) * 100
    e = s + 100
    b = np.full(4, MIB)
    pairs = np.array([2, 3])
    assert miss_fraction(s - base, e - base, b, pairs, P=2, C_mib=1.5) == 1.0
    assert miss_fraction(s, e, b, pairs, P=2, C_mib=1.5) == 1.0
    assert miss_fraction(s, e, b, pairs, P=2, C_mib=2.5) == 0.0
    assert (e.astype(np.float64) - s.astype(np.float64) <= 0).any()     # the float-first cast does collapse lives

    def float_first(start, end, slab, prs, P, C):                          # the reviewed code path, for contrast
        with np.errstate(divide="ignore", invalid="ignore"):
            ss, ee = np.asarray(start, np.float64), np.asarray(end, np.float64)
            t0 = min(ss.min(), ee.min())
            ss, ee = ss - t0, ee - t0
            F = V7._stream(ss, ee, slab)
            x = ss + (ee - ss) / 2
            return float(np.mean(F(x[prs]) - F(x[prs - P]) > C * MIB))
    assert float_first(s, e, b, pairs, 2, 1.5) == 0.0
    F = V7.bytes_streamed(s, e, b)
    assert F(base) == 0.0 and F(base + 400) == 4 * MIB and F(base + 150) == 1.5 * MIB


def test_a_non_positive_life_is_refused():
    s = np.array([0, 10, 20], dtype=np.int64)
    for e in (np.array([5, 10, 30], dtype=np.int64), np.array([5, 9, 30], dtype=np.int64)):
        with pytest.raises(ValueError):
            miss_fraction(s, e, np.full(3, MIB), pairs=np.array([2]), P=1)
        with pytest.raises(ValueError):
            V7.bytes_streamed(s, e, np.full(3, MIB))


@pytest.mark.skipif(not (PUB / "2026-11-12-nvidia_gh200_480gb-rr-session").is_dir(), reason="RR's session is not here")
def test_miss_fraction_is_rrs_reader_on_every_read_launch_of_rrs_stamps():
    """Fed the real stamps of RR's stamp units (read_rr.stamp_unit, pinned), miss_fraction gives the reader's m_hat for
    every registered step form (F1 at 42.5, F2 at 45.0, F3 at 30.0 MiB) on every READ w1 launch, exactly."""
    import read_rr as RR
    orig, got = RR.pair_read, []

    def pair_read(rows, experts, ntpp, N, K, tile):
        out = orig(rows, experts, ntpp, N, K, tile)
        if out.get("status") == "READ":
            s, e, _life = RR.exact_se(rows)
            bm, bn = int(tile[0]), int(tile[1])
            Pn = -(-int(N) // bn)
            lt = int(ntpp) // bm
            r = RR.RC3.pairs(np.asarray(experts).reshape(-1)[:lt], Pn)
            pid = np.arange(lt * Pn)
            b = int(K) * np.minimum(bn, int(N) - (pid % Pn) * bn) * 2.0
            for f, (dist, law, C, _o) in RR.P.FORMS.items():
                if law == "step":
                    got.append((out["m_hat"][f], miss_fraction(s / 1e3, e / 1e3, b, r, Pn, C,
                                                                "F1" if dist == "start" else "F2")))
        return out
    RR.pair_read = pair_read
    try:
        rr = PUB / "2026-11-12-nvidia_gh200_480gb-rr-session"
        results, session = rr / "results", rr / "session"
        pd, pw = RR.RP.unit_dir(results, RR.OTHER_UNITS[RR.PT])
        perturbs = {RR.PT: {"dir": pd, "why": pw, "json": json.loads((pd / "perturb.json").read_text())
                            if pd and (pd / "perturb.json").is_file() else {}}}
        rcd, rcw = RR.RP.unit_dir(results, RR.OTHER_UNITS[RR.RCP])
        for label in RR.STAMP_UNITS:
            RR.stamp_unit(results, session, label, RR.reg(), perturbs, {RR.RCP: {"dir": rcd, "why": rcw}},
                          RR.P.load_shapes())
    finally:
        RR.pair_read = orig
    assert len(got) >= 600, len(got)
    assert all(a == b for a, b in got), [x for x in got if x[0] != x[1]][:5]


# ---------------------------------------------------------------- the launch layer

def test_lsched_fills_every_slot_at_zero_and_starts_in_pid_order():
    s, life = V7.lsched(1000, 264, 0.08, np.random.default_rng(7))
    assert (s[:264] == 0).all() and (np.diff(s) >= 0).all() and (life >= V7.RR_LIFE_MIN).all()


def test_the_launch_layer_and_miss_fraction_agree_while_every_slot_is_busy():
    """With each CTA streaming its slab over one median life (rate = slab per unit time, bytes = slab x life), the
    GPU streams W_c x slab per unit time while all W_c slots are busy, so miss_fraction's integral equals the design's
    dt x W_c x slab on every pair whose moments both fall before the last start."""
    counts, bm, P, W_c = [600, 300, 900, 128, 1200], 128, 7, 24
    slab = 3 * MIB
    r = V7.reread_pairs(counts, bm, P)
    live = V7.m_tile_experts(counts, bm).size * P
    s, life = V7.lsched(live, W_c, 0.08, np.random.default_rng(7))
    e = s + life
    F = V7.bytes_streamed(s, e, slab * life)
    mid = s + life / 2
    busy = (mid[r] <= s.max()) & (mid[r - P] <= s.max())
    assert busy.sum() > 50
    design = (mid[r] - mid[r - P]) * W_c * slab
    integral = F(mid[r]) - F(mid[r - P])
    assert np.allclose(integral[busy], design[busy], rtol=1e-9, atol=1e-3)


def test_rr_w1_counts_each_active_experts_w1_once_and_only_its_extra_m_tiles_as_re_reads():
    from moe.spec import MODEL_CONFIGS
    m = "olmoe-1b-7b"
    cfg = MODEL_CONFIGS[m]
    We = 2 * cfg.intermediate_size * cfg.hidden_size * 2
    counts = [0] * cfg.num_experts
    counts[:3] = [128, 300, 0]
    t = V7.rr_w1(m, counts, 1, 128)
    assert t["C_u"] == 2 * We + (428 // cfg.top_k) * cfg.hidden_size * 2
    assert t["R"] == 2 * We and t["pairs"] == 2 * t["P"]
    assert t["bytes"] == t["C_u"] + t["m_hat"] * t["R"]
    one = V7.rr_w1(m, [100] * cfg.num_experts, 1, 128)
    assert one["R"] == 0 and one["m_hat"] == 0.0 and one["pairs"] == 0
    assert V7.rr_w1(m, None, 2, 128)["R"] == cfg.num_experts * We       # balanced: n x BLOCK_M rows, 2 M-tiles each


def test_the_launch_layer_is_seeded_and_deterministic():
    counts = [700] * 8
    a = V7.rr_distances(counts, 128, 28, 264, 128 * 4096 * 2, 0.08)
    b = V7.rr_distances(counts, 128, 28, 264, 128 * 4096 * 2, 0.08)
    assert len(a) == V7.RR_REPS and all((x[1] == y[1]).all() for x, y in zip(a, b))


# ---------------------------------------------------------------- r3: v3 exactly, both terms out of domain

def test_v7_at_r3s_tile_is_v3_bit_for_bit_on_all_300_registered_predictions_and_no_new_term_runs(monkeypatch):
    def never(*a, **k):
        raise AssertionError("a v7 term ran at R3's tile")
    monkeypatch.setattr(V7, "rr_w1", never)
    monkeypatch.setattr(V7, "price_prefill", never)
    monkeypatch.setattr(V7, "price_decode_cell", never)
    got = V7.check_r3(switches=V7.V7)
    assert got["compared"] == 300, got["compared"]
    assert not got["unequal_registered"], got["unequal_registered"][:3]
    assert not got["unequal_pricer"], got["unequal_pricer"][:3]


def test_the_r3_check_cli_exits_0():
    assert V7.main(["--check-r3", "--limit", "3"]) == 0
    assert V7.main([]) == 2


# ---------------------------------------------------------------- decode: v6d exactly with D6FIX off

def _decode():
    import d4common as D4
    if not (REPO / D4.CAL2_PUBLISHED / "results").is_dir() or not (REPO / D4.BLIND_PUBLISHED / "results").is_dir():
        pytest.skip("the CAL2 or decode blind session is not published in this checkout")
    import dcommon as DC
    import finalise_decode as FN
    import gate_d6 as G
    import score_blind_d3 as SB3
    import score_cal_d3 as SC3
    return D4, DC, FN, G, SB3, SC3


@pytest.fixture(scope="module")
def dec():
    D4, DC, FN, G, SB3, SC3 = _decode()
    terms, law = G.readings_law(G.readings_dir_committed(Path(tempfile.mkdtemp()) / "r"))
    tp, src = D4.tile_source()
    cal2, blind = REPO / D4.CAL2_PUBLISHED / "results", REPO / D4.BLIND_PUBLISHED / "results"
    X = V7.decode_modules()

    def cal(sw, prm_fix=None, naive_v4dp=False):
        with V7.decode_scorers(terms=terms, law=law, prm_fix=prm_fix, switches=sw, naive_v4dp=naive_v4dp):
            return SC3.score(cal2, DC.registration(DC.REPO, "cal"), tp, src, settle=False)
    return {"terms": terms, "law": law, "tp": tp, "cal2": cal2, "blind": blind, "cal": cal, "X": X, "G": G, "DC": DC,
            "FN": FN, "SB3": SB3}


def test_decode_with_d6fix_off_is_v6d_on_cal2_and_on_the_re_score_row_for_row(dec):
    G, X = dec["G"], dec["X"]
    c0 = dec["cal"](V7.OFF)
    g0 = G.cal_score(dec["terms"], dec["law"], dec["cal2"])
    assert c0["rows"] == g0["rows"] and c0["FIT"] == g0["FIT"] and c0["STOP"]["verdict"] == g0["STOP"]["verdict"]
    committed = json.loads(next(ln for ln in (REPO / "scripts/scoring/decode/gate_d6.out.txt").read_text().splitlines()
                                if ln.startswith("CAL: "))[5:])
    assert c0["E1_CAL_IN_SAMPLE"]["stats"] == committed["E1_CAL_IN_SAMPLE"]
    fz = G.frozen_at(c0)
    fz["model"] = {"name": "v6d", "fixes": X["V4DP"].ALL.as_dict(), "terms": dec["terms"].as_dict(),
                   "law": dec["law"].as_dict()}
    with V7.decode_scorers(terms=dec["terms"], law=dec["law"], switches=V7.OFF), X["D3"].v4dp(X["V4DP"].ALL):
        rows = dec["FN"].predict(fz, models=dec["DC"].BLIND_MODELS)
    with V7.decode_scorers(terms=dec["terms"], law=dec["law"], switches=V7.OFF):
        s0 = dec["SB3"].score(dec["blind"], dec["DC"].registration(dec["DC"].REPO, "blind"), {"frozen": fz, "cells": rows})
    gs = G.rescore(dec["terms"], dec["law"], g0, dec["blind"])
    assert s0["rows"] == gs["rows"] and s0["E1_DEC"]["pooled"] == gs["E1_DEC"]["pooled"]
    assert s0["E1_DEC"]["pooled"]["stats"]["rms"] == 0.045169       # gate_d6.out.txt, addendum 6


def test_the_fix_fit_is_the_committed_cal2_fit_with_v4d_prime_on_every_naive_cell(dec):
    """d6_build/post_gate.out: 'v4d' on every naive cell  scaled T0 -1.87 s_naive +9.55 s_block -0.72 rms 0.0148
    sub-wave 0.0144 (selected)'."""
    c = dec["cal"](V7.V7, naive_v4dp=True)
    f = c["FIT"]["forms"]
    assert c["FIT"]["selected"] == "scaled"
    assert [round(f["scaled"][k] * 1e3, 2) for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")] == [-1.87, 9.55, -0.72]
    assert round(f["scaled"]["in_sample"]["rms"], 4) == 0.0148 and round(f["scaled"]["sub_wave"]["rms"], 4) == 0.0144
    assert round(f["fixed"]["s_naive_GR_ms"] * 1e3, 2) == -1.83


def _gate_prm(dec):
    """v6d's committed CAL2 constants (gate_d6.out.txt, fixed rho)."""
    X = dec["X"]
    line = next(ln for ln in (REPO / "scripts/scoring/decode/gate_d6.out.txt").read_text().splitlines()
                if ln.startswith("CAL: "))
    cst = json.loads(line[5:])["constants"]
    return X["V4D"].params_from(cst, rho_form="fixed", tile=dec["tp"])


def test_d6fix_moves_exactly_the_cells_v6d_prices_by_v4d_primes_naive_gemms_and_rr_moves_no_decode_cell(dec):
    """Per page cell at v6d's committed CAL2 constants (no score is computed): with D6FIX on only the v4d'-branch
    cells move (post_gate.out: CAL2 OLMoE 4, the seen blind's Phi-3.5 2, Qwen1.5 7, deepseek-v2-lite 4), and the
    RR switch changes no decode price."""
    DC = dec["DC"]
    prm = _gate_prm(dec)
    cf = dec["cal"](V7.V7, naive_v4dp=True)
    pf = dec["X"]["V4D"].params_from(cf["FIT"]["forms"]["scaled"], rho_form="scaled", tile=dec["tp"])
    moved, branch = {}, {}
    for models in (DC.CAL_MODELS, DC.BLIND_MODELS):
        for m in models:
            for c in DC.page_doc(m, "main")["cells"]:
                kw = dict(terms=dec["terms"], law=dec["law"], prm_fix=pf)
                off = V7.price_decode_cell(m, c, prm, switches=V7.OFF, **kw)
                rr = V7.price_decode_cell(m, c, prm, switches=V7.Switches(RR=True, D6FIX=False), **kw)
                on = V7.price_decode_cell(m, c, prm, switches=V7.V7, **kw)
                if off is None:
                    assert rr is None and on is None
                    continue
                assert rr["ms"] == off["ms"]
                branch[(m, c["label"])] = V7.v4dp_branch(off)
                if on["ms"] != off["ms"]:
                    moved[(m, c["label"])] = on["v7"]["D6FIX"]["rho_form"]
    want = {("olmoe-1b-7b", x) for x in ("L-T2-A8", "L-T2-A11", "L-T2-A13", "U-T2")}
    want |= {("phi-3.5-moe", x) for x in ("L-T2-A2", "L-T2-A3")}
    want |= {("qwen1.5-moe-a2.7b", x) for x in ("L-T2-A4", "L-T2-A5", "L-T2-A7", "L-T3-A4", "L-T3-A7", "L-T3-A9", "U-T3")}
    want |= {("deepseek-v2-lite", x) for x in ("L-T2-A6", "L-T2-A8", "L-T2-A10", "U-T2")}
    assert {k for k, v in branch.items() if v} == want
    assert set(moved) == want and set(moved.values()) == {"scaled"}


def test_d6fix_needs_a_fitted_prm_so_the_cal_fit_is_unchanged(dec):
    """The CAL fit prices bodies with no constants (T0_GR None): D6FIX never touches them, so the fit and its rho
    selection under D6FIX are v6d's."""
    c0 = dec["cal"](V7.OFF)
    cf = dec["cal"](V7.V7, naive_v4dp=True)
    pf = dec["X"]["V4D"].params_from(cf["FIT"]["forms"]["scaled"], rho_form="scaled", tile=dec["tp"])
    c1 = dec["cal"](V7.V7, prm_fix=pf)
    assert c1["FIT"] == c0["FIT"] and c1["FIT"]["selected"] == "fixed"


def test_price_refuses_a_decode_cell_at_a_tile_v6d_does_not_price_it_at(dec):
    DC = dec["DC"]
    prm = _gate_prm(dec)
    c = next(c for c in DC.page_doc("mixtral-8x7b", "main")["cells"] if int(c["T"]) == 32)
    p = V7.price_decode_cell("mixtral-8x7b", c, prm, terms=dec["terms"], law=dec["law"])
    tile = p["v6d"]["tile"]
    cell = {"model": "mixtral-8x7b", "cell": c, "prm": prm, "terms": dec["terms"], "law": dec["law"]}
    got = V7.price(dict(cell, tile=tile))
    assert got["regime"] == "decode" and got["ms"] == p["ms"]
    with pytest.raises(V7.Refused):
        V7.price(dict(cell, tile=next(t for t in V7.DECODE_TILES if t != tile)))


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_v7s_files():
    paths = [REPO / "scripts" / "r3_timing_model_v7.py", REPO / "scripts" / "round5_chain.py", Path(__file__),
             REPO / "tests" / "test_timing_model_v7_prefill.py", REPO / "tests" / "v7_chain_job.py"]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p
