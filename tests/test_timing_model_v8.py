"""Model v8 (scripts/r3_timing_model_v8.py) off the GPU. Round 5's chain loads generic module names, so every check
that prices a cell runs in its own interpreter (tests/v8_invariant_job.py, v8_checks_job.py, v8_extension_job.py).

  INVARIANT (slow)  v8 == v7 bit for bit, each side in a SEPARATE process (critique M3): the 300 registered R3
                    predictions, B3's 84 frozen predictions (switches off) and the same cells with v7's switches on,
                    the v7 prefill blind's 54 cells at its four reference pars and its finalised par with the round-5
                    DESCRIPTIVE column, the v7 decode blind's 143 cells, and ERNIE's whole-tread anchors (T 4096,
                    12288, 16384) against v7 at the integer tread; the v7 side holds every registered set to its
                    registered numbers; a canary shows the two processes computed independently
  POISON            the Tread raises on every numeric use; frozen sites handed one fail loudly; whole-tread page cells
                    priced through it on both prefill tiles at G 1 and 16; n_eff 3.7 != 3.0; HOPPER_G1_N and v7's
                    64x128 KeyError('reads') are refusals
  IDS               a page cell's own generated ids passed explicitly == ids None, bit for bit, on both branches
  BINDINGS          every binding of every swapped function (scanned) is the declared grep-complete list; restored
  EXTENSION (slow)  ERNIE T 1..16384 on the default tile: every call prices or is refused with a reason; refusals
                    listed; adjacent-T jumps and the within-expert order sensitivity reported, never a condition
  SPEED (slow)      per-tile wall time per call (median, p90), reported
  CAL2 (slow)       CAL2's decode rows (dcommon.CAL_MODELS' main pages) at the frozen constants: v8 == v7 bit for bit
  V7-SUITE (slow)   tests/test_timing_model_v7.py run with v8 imported and its swaps installed around every test
  SKEW (slow)       Zipf and CUDA-graph-padded routing on ERNIE's ladder: every refusal a DECLARED_REFUSALS class
  NO DEFAULTS       a prefill call without sag128, sag64 or knee is a ValueError (review 1, BLOCKER 2)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import round5_chain as R5  # noqa: E402

PUB = REPO / "results" / "published"
HAVE = all((PUB / s / "results").is_dir() for s in R5.RUNS.values())
need = pytest.mark.skipif(not HAVE, reason="the A2, S1, C2 or C4 session is not published in this checkout")
PREFILL_SWAPS = ("Geometry.num_pid_m", "W.walk", "W.cell_events", "Model.domain", "Model.evaluate", "WV4.reads_of",
                 "WV4.price", "V4.make_cell", "V4.native_path_of", "price_parts", "d5.v5_parts", "r4.m4_reads",
                 "k5.cell_ms")


def _last_json(text: str) -> dict:
    return json.loads(text.strip().splitlines()[-1])


def _run(*args, timeout=1800) -> dict:
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=str(REPO),
                       timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return _last_json(r.stdout)


# ---------------------------------------------------------------- in-process, no chain

@pytest.fixture(autouse=True)
def _model_configs_restored():
    """register_model writes MODEL_CONFIGS in this process: put it back, so no other test of this worker (the v7
    prefill blind's census reads every model of moe/spec.py) sees ERNIE."""
    from moe.spec import MODEL_CONFIGS
    keep = dict(MODEL_CONFIGS)
    yield
    MODEL_CONFIGS.clear()
    MODEL_CONFIGS.update(keep)


def test_switches_and_ernie_and_the_default_ladder():
    import r3_timing_model_v8 as V8
    assert V8.V8.as_dict() == {"RR": True, "D6FIX": True, "GEOM": True, "IDS": True}
    e = V8.ERNIE
    assert (e.name, e.hidden_size, e.intermediate_size, e.num_experts, e.top_k, e.gate_fn, e.norm_topk_prob,
            e.shared_experts, e.num_layers, e.first_moe_layer, e.hf_repo, e.verified) == (
        "ernie-4.5-21b-a3b", 2560, 1536, 64, 6, "softmax", True, 2, 28, 1, "baidu/ERNIE-4.5-21B-A3B-PT", True)
    assert V8.default_tile(32, 64) == ("16x64x128w4s4", 1)
    assert V8.default_tile(64, 64) == ("32x64x128w4s3", 1)
    assert V8.default_tile(96, 64) == ("32x128x64w4s3", 1)
    assert V8.default_tile(128, 64) == ("64x128x64w4s3", 1)
    assert V8.default_tile(512, 64) == ("64x128x64w8s3", 1)
    assert V8.default_tile(8255, 64) == ("128x128x64w8s3", 1)
    assert V8.default_tile(8256, 64) == ("128x128x64w8s3", 16)
    assert set(V8.SWAPS) == set(V8.SWAP_BINDINGS)


def test_a_call_whose_counts_are_not_T_times_k_is_refused_before_any_pricing():
    import r3_timing_model_v8 as V8
    with pytest.raises(ValueError):
        V8.price({"model": V8.ERNIE, "T": 10, "counts": [1] * 64})


def test_a_prefill_call_without_its_constants_is_a_value_error_not_a_default():
    import r3_timing_model_v8 as V8
    counts = [96] * 64                                   # ERNIE T 1024: the 128x128 prefill tile
    full = {"model": V8.ERNIE, "T": 1024, "counts": counts, "sag128": (59.815, 612.5), "sag64": (30.365, 300.0),
            "knee": (6.0, 2.75)}
    for k in ("sag128", "sag64", "knee"):
        with pytest.raises(ValueError, match=k):
            V8.price({kk: v for kk, v in full.items() if kk != k})
    with pytest.raises(ValueError, match="a pair"):
        V8.price(dict(full, knee=(6.0,)))
    assert not hasattr(V8, "DEFAULT_SAG128") and not hasattr(V8, "DEFAULT_KNEE")


def test_a_non_native_arm_on_a_tread_is_refused():
    import r3_timing_model_v8 as V8
    n = V8.Tread([128] * 8, 8, 128, label=1)
    with pytest.raises(V8.Refused, match="native arm only"):
        V8.wv4_price(None, "mixtral-8x7b", "shared", n, None, 9, None, None, 1, None)
    with pytest.raises(V8.Refused, match="native arm only"):
        V8.make_cell(None, "mixtral-8x7b", "shared", n, None, 9, 1, None, None, None, None, None)


def test_every_refusal_reason_has_a_declared_class():
    import r3_timing_model_v8 as V8
    assert V8.refusal_class("HOPPER_G1_N: w2 at G 1 with n_eff 6 >= 6 is ILL-POSED") == "HOPPER_G1_N"
    assert V8.refusal_class("Refused: v8 prices the native arm only") == "REFUSED"
    assert V8.refusal_class("KeyError: 'reads'") is None and V8.refusal_class(None) is None


def test_register_model_refuses_a_different_config_under_a_known_name():
    import dataclasses

    import r3_timing_model_v8 as V8
    V8.register_model(V8.ERNIE)
    with pytest.raises(V8.Refused):
        V8.register_model(dataclasses.replace(V8.ERNIE, top_k=8))


# ---------------------------------------------------------------- INVARIANT, two processes

@pytest.fixture(scope="module")
def inv():
    if not HAVE:
        pytest.skip("the A2, S1, C2 or C4 session is not published in this checkout")
    job = REPO / "tests" / "v8_invariant_job.py"
    ps = {s: subprocess.Popen([sys.executable, str(job), s], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              cwd=str(REPO)) for s in ("v7", "v8")}
    out = {}
    for s, p in ps.items():
        so, se = p.communicate(timeout=2400)
        assert p.returncode == 0, f"{s}: {se[-3000:]}"
        out[s] = _last_json(so)
    return out


@need
@pytest.mark.slow
def test_invariant_v8_is_v7_bit_for_bit_on_every_registered_set(inv):
    a, b = inv["v7"], inv["v8"]
    assert len(a["r3"]) == 300 and None not in a["r3"]
    assert len(a["b3"]["off"]) == len(a["b3"]["on"]) == 84
    assert a["pblind_cells"] == 54 and set(a["pblind"]) == {"A2", "C2", "C4", "B3", "finalised"}
    assert len(a["decode"]) == 143 and len(a["ernie"]) == 3
    assert len(a["cal2"]) == len(b["cal2"]) > 0 and a["cal2_naive"] > 0 and None not in a["cal2"]
    for k in ("r3", "b3", "pblind", "decode", "cal2", "ernie"):
        assert a[k] == b[k], k
    # the v7 side reproduces every registered number (so the equality above is to the registrations)
    for k in ("r3_unequal_registered", "b3_unequal_registered", "pblind_unequal_registered",
              "decode_unequal_registered"):
        assert a[k] == 0, (k, a[k])


@need
@pytest.mark.slow
def test_invariant_canary_the_two_sides_computed_in_separate_processes(inv):
    a, b = inv["v7"]["canary"], inv["v8"]["canary"]
    assert a["pid"] != b["pid"]
    assert not a["v8_loaded"] and a["tread_calls"] == {}
    assert b["v8_loaded"] and all(b["tread_calls"].get(k, 0) > 0 for k in PREFILL_SWAPS), b["tread_calls"]
    assert b["tread_calls"].get("WV3.tile_content_ids_counts", 0) > 0     # the ERNIE anchors with their real ids
    assert not a["swaps_in_place"] and not b["swaps_in_place"]


# ---------------------------------------------------------------- POISON, IDS, BINDINGS (one process)

@pytest.fixture(scope="module")
def chk():
    if not HAVE:
        pytest.skip("the A2, S1, C2 or C4 session is not published in this checkout")
    return _run(REPO / "tests" / "v8_checks_job.py")


@need
def test_the_tread_raises_on_every_numeric_use_and_is_only_a_label(chk):
    s = chk["sentinel"]
    assert all(v is True for v in s["raises"].values()), s["raises"]
    assert s["hash_equal"] and s["eq_tread"] and s["str"] == "1" and s["fmt"] == "n1" and s["frac_str"] == "0.78125"


@need
def test_a_frozen_site_handed_the_tread_fails_loudly(chk):
    assert chk["missed"] == {"frozen walk": True, "frozen num_pid_m": True, "v7 price_prefill": True}


@need
def test_whole_tread_page_cells_price_through_the_tread_on_both_prefill_tiles_at_g1_and_g16(chk):
    cells = chk["pages"]["cells"]
    assert {(c["tile"], c["G"]) for c in cells} == {("128x128x64w8s3", 1), ("128x128x64w8s3", 16),
                                                   ("64x128x64w8s3", 1), ("64x128x64w8s3", 16)}
    for c in cells:
        assert c["ms"] is not None and c["tread_calls"] > 0 and c["equal_v7"], c


@need
def test_a_fractional_tread_is_not_truncated(chk):
    f = chk["frac"]
    assert f["n_eff"][0] == 3.0 and abs(f["n_eff"][1] - 3.7) < 1e-3
    assert f["differ"] and None not in f["ms"]


@need
def test_hopper_g1_n_at_n_eff_and_v7s_latent_64x128_defect_are_refusals(chk):
    h = chk["frac"]["hopper_128"]
    assert h["ms"] is None and h["why"].startswith("HOPPER_G1_N")
    d = chk["frac"]["defect_64"]
    assert d["v7"] == "KeyError('reads')"
    assert d["v8_ms"] is None and d["v8_why"].startswith("HOPPER_G1_N")


@need
def test_a_page_cells_own_generated_ids_passed_explicitly_equal_ids_none_on_both_branches(chk):
    i = chk["ids"]
    assert len(i["prefill"]) == 84 and all(x["equal"] for x in i["prefill"])
    assert {x["cls"] for x in i["prefill"]} == {"64g1", "128g1", "128g16"}
    assert any(x["balanced"] for x in i["prefill"])
    assert len(i["decode"]) == 143 and all(x["equal"] for x in i["decode"])
    assert any(x["naive"] for x in i["decode"])          # the cells whose launch order reads the ids
    # NOT VACUOUS (review 1, MAJOR 10): on every naive cell the call's ids reached the naive path (counted), on no
    # other cell; and the same counts in another token order move at least one naive cell's price
    assert all(x["ids_path_calls"] == (1 if x["naive"] else 0) for x in i["decode"]), i["decode"]
    perm = i["decode_permuted"]
    print("decode naive cells whose price moved under permuted ids:", sum(p["differ"] for p in perm), "of", len(perm))
    assert len(perm) == sum(x["naive"] for x in i["decode"]) and any(p["differ"] for p in perm), perm
    assert chk["stats"].get("WV3.tile_content_ids_counts", 0) > 0


@need
def test_every_binding_of_every_swapped_function_is_declared_and_restored(chk):
    b = chk["bindings"]
    assert b["equal"], (b["bindings"], b["declared"])
    assert b["restored"] and b["plan_entries"] == sum(len(v) for v in b["declared"].values())
    assert set(b["declared"]["price_parts"]) == {"common.price_parts", "v5price.price_parts", "kneefit.price_parts",
                                                 "d5.price_parts", "bw.price_parts"}
    # review 1 MINOR: re-scanned per install when the loaded modules change (and not otherwise)
    assert b["rescan_unchanged_skipped"] and b["rescan_on_new_module"]


@need
def test_cleared_caches_reprice_to_the_same_bits(chk):
    c = chk["caches"]
    assert c["equal"] and c["cache_every"] > 0
    assert c["dropped"]["_CONTENT_REAL"] > 0, c["dropped"]


# ---------------------------------------------------------------- EXTENSION and SPEED (ERNIE, no measurement)

@pytest.fixture(scope="module")
def ext():
    if not HAVE:
        pytest.skip("the A2, S1, C2 or C4 session is not published in this checkout")
    return _run(REPO / "tests" / "v8_extension_job.py", timeout=3600)


@need
@pytest.mark.slow
def test_extension_every_ernie_call_prices_or_is_refused_with_a_reason(ext):
    rows = ext["rows"]
    assert ext["calls"] == len(rows) >= 1500
    assert {r["T"] for r in rows} >= set(range(1, 601)) | {4096, 8192, 8255, 8256, 12288, 16384, 16368}
    assert all((r["ms"] is not None) != bool(r["why"]) for r in rows)
    tiles = {r["tile"] for r in rows}
    assert tiles == {"16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3", "64x128x64w8s3",
                     "128x128x64w8s3"}
    refused = ext["refused"]
    print("refused:", json.dumps(refused))
    # the only declared refusal on ERNIE's default ladder: HOPPER_G1_N at 128x128 G 1, 8192 <= T <= 8255
    assert all(r["tile"] == "128x128x64w8s3" and r["G"] == 1 and 8192 <= r["T"] <= 8255
               and r["why"].startswith("HOPPER_G1_N") for r in refused), refused


@need
@pytest.mark.slow
def test_skewed_and_padded_routing_price_or_refuse_with_a_declared_reason(ext):
    skew = ext["skew"]
    assert {r["routing"] for r in skew} == {"zipf", "padded"}
    assert all((r["ms"] is not None) != bool(r["why"]) for r in skew)
    # padding concentrates: the padded rows all sit on the same k experts
    assert any(r["routing"] == "padded" and r["T"] - r["T_real"] >= 2 and r["max_count"] >= r["T"] - r["T_real"]
               for r in skew)
    print("refusal classes:", json.dumps(ext["refusal_classes"]))
    assert ext["undeclared_refusals"] == [], ext["undeclared_refusals"][:5]
    assert sum(1 for r in skew if r["ms"] is not None) > 0.9 * len(skew)


@need
@pytest.mark.slow
def test_v7s_own_suite_passes_with_v8s_swaps_installed_around_every_test(tmp_path):
    import os
    rep = tmp_path / "v7suite.json"
    env = dict(os.environ, V8_V7SUITE_REPORT=str(rep),
               PYTHONPATH=os.pathsep.join([str(REPO / "tests"), str(REPO / "scripts"), str(REPO),
                                           os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "v8_v7suite_plugin",
                        str(REPO / "tests" / "test_timing_model_v7.py")], capture_output=True, text=True,
                       cwd=str(REPO), env=env, timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    print("v7 suite under v8:", tail)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    got = json.loads(rep.read_text())
    assert got["wrapped"] > 0 and got["depth_ok"] == got["wrapped"] and got["restored"] is True, got
    assert got["plan_entries"] > 0 and sum(got["tread_calls"].values()) == 0, got     # v7 never hands a Tread


@need
@pytest.mark.slow
def test_extension_reports_jumps_and_order_sensitivity(ext):
    for k, j in ext["jumps"].items():
        print(k, json.dumps({x: j[x] for x in ("calls", "T_range", "steps_over_3pct", "decreasing_steps")}),
              "largest:", json.dumps(j["largest"][:3]))
    print("order sensitivity max |rel|:", ext["order_max_abs_rel"])
    assert len(ext["order"]) == 30 and ext["order_max_abs_rel"] >= 0


@need
@pytest.mark.slow
def test_speed_per_call_per_tile_is_recorded(ext):
    for k, s in ext["speed"].items():
        print(k, json.dumps(s))
        assert s["calls"] > 0 and 0 < s["median_ms"] <= s["p90_ms"]
