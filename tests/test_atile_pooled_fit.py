"""The PRIVATE A-tile scaffolding of 2026-10-01: wave_split_bytes' content-keyed
w1 A term and duration factor (both off by default), the pooled fitter
(`scripts/atile_pooled_fit.py`) and cross_model_predict's --byte-params /
--bytes-only, held to the published GH200 lock pages and to the registration
docs/registered/2026-10-01-atile-ksteps-gh200.json.

What each can get wrong silently: the content ids are R3's routing or the
credit is fiction; the content-keyed walk must count what a brute-force
first-read / re-read walk counts; with both switches off every registered
number must be unchanged (pinned from the pre-change code at 6d2595d); the
pooled fit must reproduce the registered parameters; a pooled file must reach
cross_model_predict whole.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import atile_pooled_fit as APF  # noqa: E402
import block_m_crossing_sweep as SWEEP  # noqa: E402
import cross_model_predict as X  # noqa: E402
import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

PUB = ROOT / "results" / "published"
CARDS = {
    "mixtral-8x7b": "2026-09-27-nvidia_gh200_480gb-session",
    "mixtral-8x22b": "2026-09-28-nvidia_gh200_480gb-8x22b-session",
    "qwen2-57b-a14b": "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session",
    "olmoe-1b-7b": "2026-09-29-nvidia_gh200_480gb-olmoe-session",
    "qwen1.5-moe-a2.7b": "2026-09-29-nvidia_gh200_480gb-qwen1.5-session",
    "phi-3.5-moe": "2026-09-29-nvidia_gh200_480gb-phi3.5-session",
    "jetmoe-8b": "2026-09-29-nvidia_gh200_480gb-jetmoe-session",
    "granite-3.0-3b-a800m": "2026-09-30-nvidia_gh200_480gb-granite-session",
}
REG = ROOT / "docs" / "registered" / "2026-10-01-atile-ksteps-gh200.json"


def card_dir(model: str) -> Path:
    return next((PUB / CARDS[model] / "results").glob("*r3-counters/lock1710"))


_cache: dict = {}


def card(model: str) -> W.Card:
    if model not in _cache:
        _cache[model] = W.load_card(card_dir(model))
    return _cache[model]


def test_content_ids_are_r3s_routing_in_e_over_k_classes_of_k():
    """Every published model: E n / k distinct tiles at n = 1 and 3, each held
    by exactly k M-tiles, one per expert of a class (the heap greedy's ties)."""
    for m in CARDS:
        cfg = MODEL_CONFIGS[m]
        for n in (1, 3):
            ids = W.tile_content_ids(m, n, 32)
            assert ids.size == cfg.num_experts * n
            u, counts = np.unique(ids, return_counts=True)
            assert u.size == cfg.num_experts * n // cfg.top_k, (m, n)
            assert set(counts.tolist()) == {cfg.top_k}, (m, n)
            # the k holders of one content are k distinct experts, same tile index
            for c in u[:5]:
                where = np.nonzero(ids == c)[0]
                assert len({w // n for w in where}) == cfg.top_k
                assert len({w % n for w in where}) == 1


def test_content_ids_read_the_routing_balanced_ids_builds():
    cfg = MODEL_CONFIGS["olmoe-1b-7b"]
    ids = SWEEP.balanced_ids(cfg, SWEEP.tokens_for_rows(cfg, 64), "cpu").numpy()
    cid = W.tile_content_ids("olmoe-1b-7b", 2, 32)
    toks = {e: np.nonzero((ids == e).any(axis=1))[0] for e in range(cfg.num_experts)}
    for a in range(cfg.num_experts * 2):
        for b in range(cfg.num_experts * 2):
            same = (toks[a // 2][(a % 2) * 32:(a % 2 + 1) * 32]
                    == toks[b // 2][(b % 2) * 32:(b % 2 + 1) * 32]).all()
            assert same == (cid[a] == cid[b])


def _toy_geometry() -> W.Geometry:
    cfg = MODEL_CONFIGS["toy"]
    bm = 32
    P = {"w1": 2 * cfg.intermediate_size // 64, "w2": cfg.hidden_size // 64}
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    Wg = {g: cfg.num_experts * P[g] * 64 * K[g] * 2 for g in W.GEMMS}
    return W.Geometry(card="toy", name="toy", uuid="", capability="9.0", sm_count=4,
                      l2_bytes=1 << 20, model="toy", dtype="bf16", experts=cfg.num_experts,
                      block_m=bm, block_n=64, block_k=64, num_warps=4, num_stages=4,
                      copies_declared=2,
                      declared=(("native", 4), ("private", 8), ("shared", 8)),
                      P=tuple(sorted(P.items())), K=tuple(sorted(K.items())),
                      W=tuple(sorted(Wg.items())), W_c=(("w1", 6), ("w2", 6)),
                      ctas_per_sm=(("w1", 2), ("w2", 2)))


@pytest.mark.parametrize("G", [1, 2, 3])
def test_the_content_keyed_walk_is_a_brute_force_first_read_walk_on_a_toy(G):
    """E 4, k 2, n 2: CTAs in launch order (dead ones skipped) read their tile's
    content; the first read of a content is compulsory, every later one an A
    event at the live-rank distance to the content's previous read."""
    geom = _toy_geometry()
    n, arm = 2, "private"
    base = W.walk(arm, G, n, geom.get("P", "w1"), geom.get("W_c", "w1"),
                  geom.num_pid_m(arm, n), geom.experts,
                  slab_bytes=geom.slab("w1"), atile_bytes=geom.atile("w1"))
    import dataclasses
    base = dataclasses.replace(base, gemm="w1")
    ev = W.content_events(geom, base)
    ctab = W.tile_content_ids("toy", n, 32)
    npm, P = geom.num_pid_m(arm, n), geom.get("P", "w1")
    last, D, first = {}, [], 0
    rank = 0
    for pid in range(npm * P):
        _g, m, _j = (int(v[0]) for v in W.pid_map(np.array([pid]), npm, P, G))
        if m >= geom.experts * n:
            continue
        c = int(ctab[m])
        if c in last:
            D.append(rank - last[c])
        else:
            first += 1
        last[c] = rank
        rank += 1
    got = sorted(d for d, k in zip(ev.a_D, ev.a_k, strict=True) for _ in range(int(k)))
    assert got == sorted(D)
    assert first == np.unique(ctab).size
    off = (first - geom.experts * n) * geom.atile("w1") / geom.slab("w1")
    assert ev.slabs == pytest.approx(base.slabs + off)
    w2 = dataclasses.replace(base, gemm="w2")
    assert W.content_events(geom, w2) is w2   # w2's A is per (token, slot): no sharing


#: q on the 8x7B 09-27 card with fixed parameters, from wave_split_bytes at
#: 6d2595d (before the 2026-10-01 terms existed).
PINNED_CELLS = [("private", 64, 9, "w1"), ("private", 64, 9, "w2"), ("shared", 16, 4, "w2"),
                ("native", 2, 3, "w1"), ("private", 1, 1, "w1")]
PINNED_Q = [9.8135240253, 12.0388306785, 1.2493729305, 2.0028229603, 1.0003862247]
PINNED_PRM = dict(C_A=108.0, beta_A=1.3, theta1_A=0.2, C_B=70.0, beta_B=2.0, theta1=0.5,
                  eps1_w1=0.01, eps1_w2=0.1, view=W.MIX)


def test_with_both_switches_off_every_number_is_the_registered_codes():
    prm = W.Params(**PINNED_PRM)
    got = W.Model(card("mixtral-8x7b").geom).q(prm, PINNED_CELLS)
    assert got == pytest.approx(PINNED_Q, abs=1e-9)
    assert sorted(prm.as_json()) == ['C_A', 'C_A2', 'C_B', 'beta_A', 'beta_A2', 'beta_B',
                                     'eps1_w1', 'eps1_w2', 'theta1', 'theta1_A', 'view']
    # a duration factor of exponent 0 is the same map
    same = W.Model(card("mixtral-8x7b").geom).q(prm.replace(k0_A=40.0, a_A=0.0), PINNED_CELLS)
    assert same == pytest.approx(PINNED_Q, abs=1e-9)
    assert "k0_A" in prm.replace(k0_A=40.0, a_A=1.0).as_json()
    assert W.Params.from_json(prm.as_json()) == prm


def test_the_duration_factor_shortens_every_a_distance_by_its_k_steps():
    """s(ks) multiplies the distance: the factor at k0, a equals a law whose
    C_A is divided by s (the survival depends on X / C)."""
    geom = card("mixtral-8x7b").geom
    prm = W.Params(**PINNED_PRM).replace(theta1_A=1.0)
    k0, a = 100.0, 1.0
    cells = [("private", 32, 6, "w2")]
    ks = geom.get("K", "w2") // geom.block_k
    s = (1 + ks / k0) ** -a
    q_dur = W.Model(geom).q(prm.replace(k0_A=k0, a_A=a), cells)[0]
    q_cap = W.Model(geom).q(prm.replace(C_A=prm.C_A / s), cells)[0]
    assert q_dur == pytest.approx(q_cap, rel=1e-9)
    assert q_dur < W.Model(geom).q(prm, cells)[0]


def test_routing_is_a_design_key_and_every_published_gh200_card_still_loads():
    assert "routing" in W.DESIGN_KEYS
    for m in CARDS:
        assert card(m).geom.model == m


@pytest.mark.parametrize("model", ["qwen2-57b-a14b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b",
                                   "granite-3.0-3b-a800m"])
def test_the_g1_credit_is_the_harness_share_and_the_pages_read_it(model):
    """At G=1 n=1 the content-keyed compulsory term is -(1 - 1/k) BM / (2F) per
    weight set exactly, and every k >= 4 published page reads within 0.0005
    of it."""
    c = card(model)
    cfg = MODEL_CONFIGS[model]
    credit = -(1 - 1 / cfg.top_k) * 32 / (2 * cfg.intermediate_size)
    base = W.Model(c.geom).events("private", 1, 1, "w1")
    ev = W.Model(c.geom, content_a=True).events("private", 1, 1, "w1")
    assert (ev.slabs - base.slabs) / ev.per_set == pytest.approx(credit, rel=1e-9)
    assert c.measured[("private", 1, 1, "w1")] - 1 == pytest.approx(credit, abs=5e-4)


def _cal_cards():
    return [card(m) for m in APF.CALIBRATION_MODELS]


def test_the_pooled_fit_reproduces_the_registered_primary_r1():
    reg = json.loads(REG.read_text())
    want = reg["fits"]["primary (calibration-only: 8x7B 09-27, 8x22B, Qwen2-57B, OLMoE)"]["R1"]
    prm, res = APF.fit_pooled(_cal_cards(), "mix")
    assert prm.C_A == pytest.approx(want["params"]["C_A"], rel=2e-3)
    assert prm.beta_A == pytest.approx(want["params"]["beta_A"], rel=2e-3)
    d = APF.doc_for(_cal_cards(), "mix", prm, res)
    assert d["label"] == "calibration-only" and d["content_a"] is True
    assert d["rms_pct"] == pytest.approx(want["rms_pct"], abs=0.01)
    assert len(res) == 4   # one entry per model, not folded by the shared card slug


def test_a_pool_with_a_diagnosis_model_says_its_pages_were_seen():
    assert APF.label_for([card("mixtral-8x7b"), card("jetmoe-8b")]).startswith("includes")
    assert APF.label_for(_cal_cards()) == "calibration-only"
    with pytest.raises(W.Refused):
        APF.fit_pooled(_cal_cards(), "knee40")


def test_the_registration_holds_every_scoring_cell_and_separates_its_candidates():
    reg = json.loads(REG.read_text())
    cells = reg["cells_private"]
    for t, gs in (("mixtral-8x7b-tp8", (8, 16, 32, 64)), ("mixtral-8x7b-tp4", (16, 32, 64)),
                  ("mixtral-8x7b-tp2", (16, 32, 64)), ("qwen2-57b-a14b", (128,)),
                  ("olmoe-1b-7b", (128,)), ("mixtral-8x7b", (8, 16, 32))):
        for G in gs:
            for n in range(1, 10):
                for g in W.GEMMS:
                    assert set(cells[f"{t}:{G}:{n}:{g}"]) == {"R0", "R1-cal", "R2-cal",
                                                               "R1-all8", "R2-all8"}
    rho = reg["tests"]["T2 duration term (separates R2 from every byte law, R0 and R1)"]["rho"]
    for name in ("rho42", "rho84"):
        assert rho[name]["R1-cal"]["band"][1] < rho[name]["R2-cal"]["band"][0]
    split = reg["split_table_q_n8"]
    q2 = split["qwen2-57b-a14b:128:8:w1"]
    assert q2["R0"] / q2["R1-cal"] > 1.10


@pytest.fixture(scope="module")
def pooled_file(tmp_path_factory):
    prm = W.Params(C_A=159.6, beta_A=1.05, theta1_A=0.0, C_B=1e9, beta_B=1.0, theta1=1.0,
                   eps1_w1=0.0, eps1_w2=0.0, view=W.MIX)
    cards = [card("olmoe-1b-7b")]
    res = {APF.card_key(cards[0]): ([], np.zeros(3))}
    p = tmp_path_factory.mktemp("pooled") / "R1test.json"
    p.write_text(json.dumps(APF.doc_for(cards, "mix", prm, res)))
    return p, prm


def test_byte_params_reach_cross_model_predict_whole(pooled_file):
    path, prm = pooled_file
    got, ca, doc = APF.read_pooled(path)
    assert got == prm and ca is True
    d = X.predict_bytes(card_dir("mixtral-8x7b"), "mixtral-8x7b-tp8", [path], groups=(64,))
    geom = X.target_geometry(card("mixtral-8x7b").geom, "mixtral-8x7b-tp8")
    row = next(r for r in d["cells"] if r["arm"] == "private" and r["n"] == 8)
    want = W.Model(geom, content_a=True).q(prm, [("private", 64, 8, "w1")])[0]
    assert row["q_pooled"]["R1test"]["w1"] == pytest.approx(want, rel=1e-9)
    bare = X.predict_bytes(card_dir("mixtral-8x7b"), "mixtral-8x7b-tp8", (), groups=(64,))
    r0 = next(r for r in bare["cells"] if r["arm"] == "private" and r["n"] == 8)
    assert r0["q"] == row["q"] and "q_pooled" not in r0
    assert d["pooled"]["R1test"]["label"] == "calibration-only"


def test_without_bytes_only_the_timed_pages_are_required(capsys):
    assert X.main(["--target", "mixtral-8x7b-tp8", "--counters",
                   str(card_dir("mixtral-8x7b"))]) == 2
    assert "--timed is required" in capsys.readouterr().out
