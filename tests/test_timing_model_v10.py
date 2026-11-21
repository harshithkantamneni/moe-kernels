"""v10 (scripts/r3_timing_model_v10.py): v9 everywhere except the 128x128x64 prefill call's non-GEMM term."""
from __future__ import annotations

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
import r3_timing_model_v10 as V10  # noqa: E402

SAG128 = (56.725, 562.5)
CASES = [("ernie-4.5-21b-a3b", T) for T in (8, 96, 136, 256, 520, 1024, 4096)] + \
        [("qwen3-30b-a3b", T) for T in (8, 136, 520, 2048)] + \
        [("lfm2-8b-a1b", T) for T in (8, 256, 1024, 4200)]      # 4200 >= 129 x 32: the G16 128x128 tile


def _call(model_key: str, T: int, seed: int = 0) -> dict:
    cfg = PL.MODELS[model_key].cfg
    rng = np.random.default_rng(seed * 100003 + T)
    ids = np.argsort(rng.random((T, cfg.num_experts)), axis=1)[:, :cfg.top_k]
    call = {"model": cfg, "T": T, "counts": [int(c) for c in PL.counts_from_ids(ids, cfg.num_experts)],
            "ids": ids.reshape(-1).astype(np.int64)}
    if T > PL.DECODE_MAX_T:
        call.update(PL.prefill_constants(SAG128))
    return call


def _is128(a: dict) -> bool:
    return a.get("regime") == "prefill" and a.get("ms") is not None and str(a.get("tile", "")).startswith(V10.TILE_128)


@pytest.fixture(scope="module")
def priced():
    from moe.spec import MODEL_CONFIGS
    before = set(MODEL_CONFIGS)
    out = {}
    for mk, T in CASES:
        c = _call(mk, T)
        out[(mk, T)] = (V8.price(dict(c)), V9.price(dict(c)), V10.price(dict(c)),
                        V10.price(dict(c), V10.Switches(NONGEMM128=False)))
    yield out
    for k in set(MODEL_CONFIGS) - before:      # pricing registered the served models (conftest's rule, module scope)
        del MODEL_CONFIGS[k]


def test_every_call_off_the_128x128_prefill_tile_is_v9_bit_for_bit(priced):
    seen = set()
    for key, (a8, a9, a10, _off) in priced.items():
        if _is128(a8):
            continue
        seen.add(a8["tile"])
        assert a10 == a9, key
    assert any(t.startswith("64x128") for t in seen) and any(t.startswith("16x64") for t in seen), seen


def test_on_the_128x128_prefill_tile_only_t0_s_b_is_swapped_for_v9s_kernel_constants(priced):
    n, tiles = 0, set()   # tiles: the calls' GROUP_SIZE_M
    for key, (a8, a9, a10, _off) in priced.items():
        if not _is128(a8):
            continue
        n += 1
        tiles.add(int(a8["G"]))
        assert a9 == a8, key                                                           # v9 left this tile alone
        t = a10["terms"]["NONGEMM128"]
        assert t["v8_ms"] == a8["ms"] and t["path"] == "block-scan"
        assert t["removed_T0_s_b_ms"] == pytest.approx(43.23e-3 - 7.07e-3, abs=2e-5)    # fit2's T0 + s_block
        assert t["added_ms"] == V9.nongemm_ms("block-scan", a8["numel"])
        assert a10["ms"] == pytest.approx(a8["ms"] - t["removed_T0_s_b_ms"] + t["added_ms"], rel=0, abs=1e-12)
        assert {k: v for k, v in a10.items() if k not in ("ms", "terms")} == \
            {k: v for k, v in a8.items() if k not in ("ms", "terms")}
    assert n >= 5 and 16 in tiles and len(tiles) >= 2, (n, tiles)                   # G16 and below it


def test_no_new_constant():
    assert not [k for k, v in vars(V10).items() if k.isupper() and isinstance(v, float)]


def test_the_switch_off_is_v9(priced):
    for key, (_a8, a9, _a10, off) in priced.items():
        assert off == a9, key


def test_v8s_bindings_are_restored_after_a_v10_call():
    before = {(id(h), a): (h.get(a) if isinstance(h, dict) else vars(h).get(a)) for h, a, _f, _c in V8._plan()}
    V10.price(_call("ernie-4.5-21b-a3b", 1024))
    after = {(id(h), a): (h.get(a) if isinstance(h, dict) else vars(h).get(a)) for h, a, _f, _c in V8._plan()}
    assert before == after


def test_a_call_v8_refuses_is_refused_the_same_way():
    c = _call("ernie-4.5-21b-a3b", 1024)
    c["counts"] = c["counts"][:-1] + [c["counts"][-1] + 1]
    with pytest.raises(ValueError):
        V8.price(dict(c))
    with pytest.raises(ValueError):
        V10.price(dict(c))
