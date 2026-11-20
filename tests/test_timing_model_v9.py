"""v9 (scripts/r3_timing_model_v9.py): v8 everywhere except the 64x128x64 prefill call's non-GEMM term."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prodlib as PL  # noqa: E402
import r3_timing_model_v8 as V8  # noqa: E402
import r3_timing_model_v9 as V9  # noqa: E402

SAG128 = (56.725, 562.5)
MODELS = ("ernie-4.5-21b-a3b", "qwen3-30b-a3b")


def _call(model_key: str, T: int, seed: int = 0) -> dict:
    cfg = PL.MODELS[model_key].cfg
    rng = np.random.default_rng(seed * 100003 + T)
    ids = np.argsort(rng.random((T, cfg.num_experts)), axis=1)[:, :cfg.top_k]
    call = {"model": cfg, "T": T, "counts": [int(c) for c in PL.counts_from_ids(ids, cfg.num_experts)],
            "ids": ids.reshape(-1).astype(np.int64)}
    if T > PL.DECODE_MAX_T:
        call.update(PL.prefill_constants(SAG128))
    return call


@pytest.fixture(scope="module")
def priced():
    from moe.spec import MODEL_CONFIGS
    before = set(MODEL_CONFIGS)
    out = {}
    for mk in MODELS:
        for T in (8, 96, 136, 256, 512, 520, 1024):
            c = _call(mk, T)
            out[(mk, T)] = (V8.price(dict(c)), V9.price(dict(c)), V9.price(dict(c), V9.Switches(NONGEMM=False)))
    yield out
    for k in set(MODEL_CONFIGS) - before:      # pricing registered the served models (conftest's rule, module scope)
        del MODEL_CONFIGS[k]


def test_the_constants_are_the_fit_on_the_published_launch_floor_pages():
    if not glob.glob(str(REPO / "results/published/*/results/*launch-floor*/*/traces/parsed.json")):
        pytest.skip("no published launch-floor pages in this checkout")
    import v9_nongemm_fit as FIT
    r = FIT.fit(REPO)
    assert (r["A0_US"], r["AS_US"], r["AS_US_PER_1K_IDS"], r["K0_ACT_US"], r["K0_SUM_US"]) == \
        (V9.A0_US, V9.AS_US, V9.AS_US_PER_1K_IDS, V9.K0_ACT_US, V9.K0_SUM_US)


def test_every_call_off_the_64x128_prefill_tile_is_v8_bit_for_bit(priced):
    seen = set()
    for (mk, T), (a8, a9, _off) in priced.items():
        if a8["regime"] == "prefill" and str(a8["tile"]).startswith(V9.TILE_64):
            continue
        seen.add(a8["tile"])
        assert a9 == a8, (mk, T)
    assert any(t.startswith("128x128") for t in seen) and any(t.startswith("16x64") for t in seen), seen


def test_on_the_64x128_prefill_tile_only_t0_s_b_is_swapped_for_the_kernel_constants(priced):
    n, paths = 0, set()
    for (mk, T), (a8, a9, _off) in priced.items():
        if not (a8["regime"] == "prefill" and str(a8["tile"]).startswith(V9.TILE_64)):
            continue
        n += 1
        t = a9["terms"]["NONGEMM"]
        assert t["v8_ms"] == a8["ms"]
        s_path = {"block-scan": -7.07e-3, "small-batch": 12.99e-3}[t["path"]]           # fit2's s_block, s_small
        assert t["removed_T0_s_b_ms"] == pytest.approx(43.23e-3 + s_path, abs=2e-5)    # fit2's T0 + the path's s
        assert a9["ms"] == pytest.approx(a8["ms"] - t["removed_T0_s_b_ms"] + V9.nongemm_ms(t["path"], a8["numel"]),
                                         rel=0, abs=1e-12)
        paths.add(t["path"])
        assert {k: v for k, v in a9.items() if k not in ("ms", "terms")} == \
            {k: v for k, v in a8.items() if k not in ("ms", "terms")}
    assert n >= 4 and paths == {"block-scan", "small-batch"}, (n, paths)


def test_the_alignment_term_per_path():
    assert V9.nongemm_ms("block-scan", 999) == pytest.approx((V9.A0_US + V9.K0_ACT_US + V9.K0_SUM_US) * 1e-3)
    assert V9.nongemm_ms("small-batch", 1000) == pytest.approx(
        (V9.AS_US + V9.AS_US_PER_1K_IDS + V9.K0_ACT_US + V9.K0_SUM_US) * 1e-3)
    with pytest.raises(V8.Refused):
        V9.nongemm_ms("other", 10)


def test_the_switch_off_is_v8(priced):
    for (mk, T), (a8, _a9, off) in priced.items():
        assert off == a8, (mk, T)


def test_v8s_bindings_are_restored_after_a_v9_call():
    before = {(id(h), a): (h.get(a) if isinstance(h, dict) else vars(h).get(a)) for h, a, _f, _c in V8._plan()}
    V9.price(_call("ernie-4.5-21b-a3b", 256))
    after = {(id(h), a): (h.get(a) if isinstance(h, dict) else vars(h).get(a)) for h, a, _f, _c in V8._plan()}
    assert before == after


def test_a_call_v8_refuses_is_refused_the_same_way():
    c = _call("ernie-4.5-21b-a3b", 256)
    c["counts"] = c["counts"][:-1] + [c["counts"][-1] + 1]
    with pytest.raises(ValueError):
        V8.price(dict(c))
    with pytest.raises(ValueError):
        V9.price(dict(c))
