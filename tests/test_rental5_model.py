"""Rental 5's model changes off the GPU: model v2 (r3_timing_model --dead-model v2), the
per-expert schedule (MODEL M2), the per-expert byte walk (MODEL M3, wave_split_bytes), the
histogram page reader (S7) and the synthetic histograms (scripts/skew_synth.py), with the
registrations' --check (docs/registered/2026-10-07-rental5-*)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cores_heldout_predict as CP  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import skew_synth as K  # noqa: E402
import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

HIST = REPO / "docs" / "registered" / "2026-10-07-rental5-skew-hist"
R5 = REPO / "scripts" / "scoring" / "rental5"


@pytest.fixture(autouse=True)
def _model():
    old = TM.set_model("mixtral-8x7b")
    yield
    TM.set_model(old)


# --------------------------------------------------------------------------
# model v2
# --------------------------------------------------------------------------

def _src(dead_model):
    argv = [*map(str, CP.source_pages()), "--counters", str(CP.C27), "--dead-model", dead_model]
    return TM.build(TM.build_parser().parse_args(argv))["main"].params


def test_v2_refits_t0_to_43_23_us_on_the_cal_pages_and_leaves_c_and_bw():
    m, v2 = _src("m"), _src("v2")
    assert round(m["T0"] * 1e3, 2) == 44.24 and round(v2["T0"] * 1e3, 2) == 43.23
    assert v2["c"] == pytest.approx(m["c"], rel=1e-9) and v2["bw"] == pytest.approx(m["bw"], rel=1e-9)
    assert round(v2["s_small"] * 1e3, 2) == 12.99 and round(v2["s_block"] * 1e3, 2) == -7.07
    assert TM.DEAD_MODEL == "m", "build restores the default"


def test_the_default_dead_term_is_model_m_and_v2_is_d_and_kappa():
    assert TM.DEAD_MODEL == "m" and TM.DEAD_MODELS == ("m", "v2")
    a = TM.dead_ms(72, 1, "w1", 202.55, TM.K_W, 5)
    hid = TM.K_W * 5 * TM.floor_ksteps("w1") * 202.55
    assert a == pytest.approx(max(0, TM.dead_ctas(72, 1, "w1") * TM.DEAD_CTA_NS - hid) * 1e-6)
    with TM.dead_model("v2"):
        b = TM.dead_ms(72, 1, "w1", 202.55, TM.K_W, 5)
        hid2 = TM.V2_KAPPA * 5 * TM.floor_ksteps("w1") * 202.55
        assert b == pytest.approx(max(0, TM.dead_ctas(72, 1, "w1") * TM.V2_DEAD_NS - hid2) * 1e-6)
    assert TM.DEAD_MODEL == "m"
    with pytest.raises(TM.Refused):
        with TM.dead_model("d3"):
            pass
    assert "v2" in TM.MODEL_VERSION and TM.V2_DEAD_NS == 0.995 and TM.V2_KAPPA == 0.325


def test_the_dead_model_flag_is_in_the_parser_and_off_by_default():
    a = TM.build_parser().parse_args(["x"])
    assert a.dead_model is None
    assert TM.build_parser().parse_args(["x", "--dead-model", "v2"]).dead_model == "v2"
    with pytest.raises(SystemExit):
        TM.build_parser().parse_args(["x", "--dead-model", "d"])


# --------------------------------------------------------------------------
# MODEL M2: the per-expert schedule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b"])
def test_the_uniform_histogram_reproduces_the_schedule_bitwise_at_every_tread(model):
    TM.set_model(model)
    for n in range(1, 10):
        u = TM.uniform_counts(n)
        for arm, D in (("native", TM.E), ("shared", 9 * TM.E)):
            assert TM.grid_rows(D, n) == TM.grid_rows(D, n, u)
            assert TM.live_rows(n) == TM.live_rows(n, u)
            for g in TM.GEMMS:
                assert TM.dead_ctas(D, n, g) == TM.dead_ctas(D, n, g, u)
                for G in (1, 8, 64):
                    a, b = TM.schedule(arm, D, G, n, g), TM.schedule(arm, D, G, n, g, u)
                    assert a[1] == b[1] and a[0].tobytes() == b[0].tobytes()
    TM.set_model("mixtral-8x7b")


def test_a_skewed_histogram_counts_partial_tiles_whole_and_empty_experts_own_nothing():
    c = (100, 0, 33, 1, 64, 64, 128, 122)   # 512 rows, n = 2
    assert sum(c) == TM.IDS_PER_TREAD * 2
    assert TM.live_rows(2, c) == 4 + 0 + 2 + 1 + 2 + 2 + 4 + 4
    r, leads = TM.schedule("native", 8, 8, 2, "w2", c)
    P = TM.GEOMETRY["w2"].npn
    assert r.size == TM.live_rows(2, c) * P
    assert TM.grid_rows(8, 2, c) == TM.grid_rows(8, 2)            # numel is the tread's
    assert TM.dead_ctas(8, 2, "w2", c) == (TM.grid_rows(8, 2) - 19) * P
    # the A share of the one-row tile is 1/32 of a full tile's
    share = (r - (r > TM.GEOMETRY["w2"].arow / P * 1.5) * TM.GEOMETRY["w2"].slab)
    assert share.min() == pytest.approx(TM.GEOMETRY["w2"].arow / P / 32)
    with pytest.raises(TM.Refused):
        TM.schedule("private", 72, 8, 2, "w1", c)


def test_call_ms_on_the_uniform_histogram_equals_the_balanced_cell():
    cell = TM.Cell(arm="shared", G=8, n=3, ms=1.0, mhz=None, path=TM.BLOCK_SCAN, declared=72, runs=(),
                   source="x", reads=None, sigma={"w1": 1.0, "w2": 1.0}, fit=False)
    x, ctx = np.array([0.0432, 202.5, 3598.0, 0.013, -0.007]), TM.Context(
        card="c", device="d", sms=132, sms_source="", occupancy={"w1": 5, "w2": 4}, occupancy_source="",
        clock_mhz=1710.0, locked=True, bandwidth_gbps=4000.0, byte_label="")
    import dataclasses
    u = dataclasses.replace(cell, counts=TM.uniform_counts(3))
    for dm in ("m", "v2"):
        with TM.dead_model(dm):
            assert TM.call_ms(x, cell, ctx, 0.5) == TM.call_ms(x, u, ctx, 0.5)


# --------------------------------------------------------------------------
# MODEL M3: the per-expert byte walk
# --------------------------------------------------------------------------

def _same(a, b):
    for f in ("live", "slabs", "in_win1", "in_later"):
        assert getattr(a, f) == getattr(b, f), f
    for f in ("x_D", "x_win1", "x_k", "x_S", "a_D", "a_win1", "a_ws", "a_k"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_the_per_expert_walk_at_uniform_counts_is_the_balanced_walk():
    for E, n, G, P, BM in ((8, 2, 8, 5, 4), (8, 3, 4, 7, 4), (4, 5, 3, 4, 2), (6, 1, 2, 4, 3)):
        npm = E * n + E
        _same(W.walk("shared", G, n, P, 40, npm, E, slab_bytes=7, atile_bytes=3),
              W.walk("shared", G, n, P, 40, npm, E, slab_bytes=7, atile_bytes=3, counts=[n * BM] * E, block_m=BM))


def _lru(G, P, owner, slab, atile, npm):
    live = len(owner)
    gid, pm, pn = W.pid_map(np.arange(npm * P), npm, P, G)
    keep = pm < live
    stream = []
    for g, m, j in zip(gid[keep].tolist(), pm[keep].tolist(), pn[keep].tolist(), strict=True):
        stream += [(("A", m), atile, g), (("S", int(owner[m]), j), slab, g)]
    last, out = {}, []
    for i, (k, _w, g) in enumerate(stream):
        if k in last and k[0] == "S" and stream[last[k]][2] != g:
            seen = {stream[t][0]: stream[t][1] for t in range(last[k] + 1, i)}
            out.append(float(sum(seen.values())))
        last[k] = i
    return sorted(out)


def test_the_per_expert_reuse_distance_is_an_exact_lru_walk_on_skewed_launches():
    rng = np.random.default_rng(3)
    done = 0
    while done < 30:
        E, BM, n = int(rng.integers(2, 7)), int(rng.integers(1, 4)), int(rng.integers(1, 5))
        G, P = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        tot = E * n * BM
        c = rng.multinomial(tot, rng.dirichlet([0.7] * E))
        owner, _rows = W.tile_owners(c, BM)
        npm = -(-(tot + E * (BM - 1)) // BM)
        ev = W.walk("native", G, n, P, 10 ** 6, npm, E, slab_bytes=7, atile_bytes=3, counts=list(c), block_m=BM)
        got = sorted(s for s, k in zip(ev.x_S.tolist(), ev.x_k.tolist(), strict=True) for _ in range(int(k)))
        assert got == _lru(G, P, owner, 7, 3, npm), (E, BM, n, G, P, c)
        done += 1


def test_tile_owners_give_each_expert_its_tiles_in_expert_order():
    owner, rows = W.tile_owners([33, 0, 64, 1], 32)
    assert owner.tolist() == [0, 0, 2, 2, 3] and rows.tolist() == [32, 1, 32, 32, 1]
    with pytest.raises(W.Refused):
        W.walk("private", 2, 1, 2, 10, 8, 4, counts=[8, 8, 8, 8], block_m=8)


def test_the_byte_model_q_at_the_uniform_histogram_equals_the_balanced_q():
    card = W.load_card(CP.C27)
    prm = W.analyse(card).params(W.REGISTERED_VIEW)
    m = W.Model(card.geom)
    for n in (2, 4):
        cells = [("shared", 8, n, g) for g in W.GEMMS]
        a = m.evaluate(prm, cells)
        b = m.evaluate(prm, [c + (tuple([32 * n] * 8),) for c in cells])
        assert [x["q"] for x in a] == [x["q"] for x in b]


# --------------------------------------------------------------------------
# S7: a histogram page is read as a target, never fitted
# --------------------------------------------------------------------------

def test_a_histogram_page_is_read_into_hist_rows_and_never_into_the_fitted_rows(tmp_path):
    d = tmp_path / "private_weight_reference" / "run1"
    d.mkdir(parents=True)
    rows = [{"arm": "native", "tiles": 2, "tokens": 256, "ms_p50": 1.5, "sm_clock_load_mhz": 1710.0,
             "experts_declared": 8, "histogram": "PT", "counts_sha256": "ab"},
            {"arm": "shared", "tiles": 2, "tokens": 256, "ms_p50": 1.2, "sm_clock_load_mhz": 1710.0,
             "experts_declared": 72, "histogram": "balanced", "counts_sha256": "cd"}]
    (d / "report.json").write_text(json.dumps({"experiment": "private_weight_reference", "kind": "histogram-page",
                                               "treads_table": rows, "histogram": {"file_sha256": "f"},
                                               "pinned": {"GROUP_SIZE_M": 8}, "gates": []}))
    p = TM.load_timed_page(d)
    assert p.rows == {} and p.histogram == {"file_sha256": "f"}
    assert p.hist_rows[("native", "PT", 2)] == (1.5, 1710.0, 8, "ab")
    assert p.hist_rows[("shared", "balanced", 2)][2] == 72


# --------------------------------------------------------------------------
# the synthetic histograms (owner decision 3)
# --------------------------------------------------------------------------

def test_the_targets_are_summary_statistics_only():
    t = json.loads(K.TARGETS.read_text())
    assert "no count of any batch" in t["what"]
    for row in t["targets"]:
        assert set(row) == {"model", "n", "B", "shape", "scheme", "layer", "layer_rule", "B_on_grid",
                            "cv", "max_mean", "gini"}
        assert all(isinstance(row[s], float) for s in K.STATS)
    assert len(t["targets"]) == 27


def test_the_draws_reproduce_the_committed_pages_from_the_committed_fit():
    pages = K.pages(json.loads(K.FIT.read_text()))
    for name, doc in pages.items():
        assert (HIST / f"{name}.json").read_text() == K.dump(doc), name


def test_one_fitted_shape_reproduces_from_its_target():
    t = json.loads(K.TARGETS.read_text())["targets"]
    fit = json.loads(K.FIT.read_text())["shapes"]
    i = next(i for i, x in enumerate(t) if x["model"] == "olmoe-1b-7b" and x["n"] == 2 and x["shape"] == "DW")
    got = K.fit_one(t[i])
    assert {k: got[k] for k in ("kind", "param", "loss", "matches")} == {k: fit[i][k] for k in ("kind", "param", "loss", "matches")}


def test_every_page_is_feasible_synthetic_and_carries_its_seed():
    for p in sorted(HIST.glob("*.json")):
        d = json.loads(p.read_text())
        if "cells" not in d:
            continue
        assert d["schema"] == K.SCHEMA and d["shuffle_seed"] == K.SHUFFLE_SEED and "synthetic" in d
        cfg = MODEL_CONFIGS[d["model"]]
        for c in d["cells"]:
            if c["counts"] is None:
                assert c["label"] == "balanced"
                continue
            T = K.tokens_for(d["model"], c["n"])
            assert sum(c["counts"]) == T * cfg.top_k and max(c["counts"]) <= T and len(c["counts"]) == cfg.num_experts
            if c["label"] in ("PT", "PW", "DW") and "provenance" in c:
                assert c["provenance"]["kind"] in ("zipf", "dirichlet") and c["provenance"]["seed"] >= K.DRAW_SEED_BASE


def test_the_skew_registration_commits_every_page_and_count_sha():
    reg = json.loads((REPO / "docs" / "registered" / "2026-10-07-rental5-skew-gh200.json").read_text())
    import hashlib
    for name, ent in reg["histograms"]["files"].items():
        assert hashlib.sha256((HIST / name).read_bytes()).hexdigest() == ent["sha256"]
        doc = json.loads((HIST / name).read_text())
        for c, e in zip(doc.get("cells", []), ent.get("cells", []), strict=True):
            assert K.sha_counts(c["counts"]) == e["counts_sha256"]
    # U predicts no skew effect, and the uniform cell matches the balanced one (delta_M3 = 0)
    for r in reg["predictions"]:
        if "ratio" in r:
            assert r["ratio"]["U"] == 1.0
        if "delta_M3" in r:
            assert r["delta_M3"] == 0.0


def test_register_check_recomputes_every_committed_number():
    got = subprocess.run([sys.executable, str(R5 / "register.py"), "--check"], capture_output=True,
                         text=True, timeout=900)
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.count(": same") == 12 and "DIFFERS" not in got.stdout
