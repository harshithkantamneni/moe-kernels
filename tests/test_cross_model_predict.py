"""scripts/cross_model_predict.py: one card's byte and timing models, fitted on
Mixtral 8x7B, predicting Mixtral 8x22B with every fitted number held.

Only the shapes change, from `moe/spec.py`. These tests hold the shapes to the
config, the prediction's physics to arithmetic that needs no model (at the
floor the SHARED slope scales with CTAs x k-steps; PRIVATE's with its bytes),
and the refusal of a target that would be another experiment."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cross_model_predict as X  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import wave_split_bytes as W  # noqa: E402

SESSION = ROOT / "results" / "published" / "2026-09-27-nvidia_gh200_480gb-session" / "results"
COUNTERS = SESSION / "2026-09-27-nvidia_gh200_480gb-r3-counters" / "lock1710"


def _timed() -> list[Path]:
    return [rep.parent for rep in sorted(SESSION.glob(
                "gaps-*/private_weight_reference/*/report.json"))
            if json.loads(rep.read_text()).get("session_tag", "").endswith("-lock1710")]


@pytest.fixture(scope="module")
def pred():
    return X.predict(_timed(), COUNTERS, "mixtral-8x22b")


def test_the_target_geometry_is_the_configs():
    g = X.target_geometry(W.load_card(COUNTERS).geom, "mixtral-8x22b")
    assert dict(g.K) == {"w1": 6144, "w2": 16384}
    assert dict(g.P) == {"w1": 512, "w2": 96}
    assert dict(g.W) == {"w1": 8 * 32768 * 6144 * 2, "w2": 8 * 6144 * 16384 * 2}
    src = W.load_card(COUNTERS).geom
    assert g.W_c == src.W_c and g.ctas_per_sm == src.ctas_per_sm and g.declared == src.declared


def test_nothing_is_fitted_on_the_target_and_the_source_model_is_restored(pred):
    assert pred["target"] == "mixtral-8x22b" and pred["source_model"] == "mixtral-8x7b"
    assert TM.MODEL == "mixtral-8x7b" and TM.GEOMETRY["w1"].K == 4096
    assert len(pred["cells"]) == 3 * len(X.PLAN_G) * len(X.PLAN_N)


def test_the_floor_slope_scales_with_ctas_times_ksteps(pred):
    """At G >= 8 SHARED sits on the floor; its per-tread cost is (N-tiles x
    k-steps) summed over the GEMMs: 8x22B's is 1.714 of 8x7B's on both GEMMs."""
    ratio = (512 * 96) / (448 * 64)
    assert (96 * 256) / (64 * 224) == pytest.approx(ratio)
    cells = {(c["arm"], c["G"], c["n"]): c["ms"] for c in pred["cells"]}
    slope = (cells[("shared", 8, 6)] - cells[("shared", 8, 2)]) / 4
    fit = TM.build(TM.build_parser().parse_args(
        [*map(str, _timed()), "--counters", str(COUNTERS)]))
    measured_8x7b = {(c.arm, c.G, c.n): c.ms for c in fit["cells"]}
    slope7 = (measured_8x7b[("shared", 8, 6)] - measured_8x7b[("shared", 8, 2)]) / 4
    assert slope / slope7 == pytest.approx(ratio, rel=0.02)


def test_private_rises_by_its_own_bytes_at_the_cards_bandwidth(pred):
    cells = {(c["arm"], c["G"], c["n"]): c["ms"] for c in pred["cells"]}
    per_tread = (cells[("private", 2, 6)] - cells[("private", 2, 2)]) / 4
    W_total = sum(s["W_bytes"] for s in pred["shapes"].values())
    bw = pred["timing_params"]["bw"]
    assert per_tread == pytest.approx(W_total / (bw * 1e6), rel=0.03)


def test_another_expert_count_is_another_experiment():
    with pytest.raises(X.Refused, match="another experiment"):
        X.predict(_timed(), COUNTERS, "deepseek-v2-lite")


def test_the_known_ill_posed_cells_print_no_time(pred):
    g1 = [c for c in pred["cells"] if c["arm"] == "shared" and c["G"] == 1 and c["n"] >= 6]
    assert g1 and all(c["ms"] is None and c["q"]["w2"] is None for c in g1)
    assert all(math.isfinite(c["ms"]) for c in pred["cells"] if c["ms"] is not None)
