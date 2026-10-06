"""Table T2 (docs/paper/T2_parameters.{md,csv}) is generated from the code and the
registrations, and its parameter count is pinned here: a parameter added to or
removed from the timing or byte model must change this test."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("t2_parameters",
                                                  ROOT / "scripts/paper/t2_parameters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_table_is_current():
    assert _load().main(["--check"]) == 0


def test_parameter_count_from_code():
    t2 = _load()
    rs = t2.rows()
    c = t2.counts(rs)
    assert c["timing_per_card"] == 5
    assert c["timing_study"] == 7
    assert c["bytes_per_card"] == 10
    assert c["bytes_per_card_before_stage3"] == 8
    assert c["total"] == 22
    names = [r["parameter"] for r in rs]
    for n in ("T0", "c", "bw", "s_small", "s_block", "k_w", "P_KNEE (p)", "DEAD_CTA_NS",
              "c_floor", "F_w1", "F_w2", "CORES_RHO_GBPS (rho)", "C_A", "beta_A", "theta1_A",
              "C_B", "beta_B", "theta1", "eps1_w1", "eps1_w2", "C_A2", "beta_A2"):
        assert n in names


def test_per_registration_counts():
    t2 = _load()
    per = t2.counts(t2.rows())["per_registration"]
    assert per["8x22B"] == 15
    assert per["Qwen2-57B"] == 17
    assert per["OLMoE"] == 18
    assert per["Qwen1.5"] == per["Phi-3.5"] == per["JetMoE"] == per["Granite-3B"] == 21
    assert per["tp8 floor (CORES)"] == 9


def test_values_read_from_registrations():
    t2 = _load()
    rows = {r["parameter"]: r for r in t2.rows()}
    assert "0.0540 (8x22B)" in rows["T0"]["value"]
    assert "202.55 (Qwen1.5, Phi-3.5, JetMoE, Granite-3B, tp8 floor (CORES))" in rows["c"]["value"]
    assert "72.52 (Qwen2-57B" in rows["C_B"]["value"]
    assert "86.18" in rows["C_A2"]["value"]
    fl = t2.floor_law_fit()
    assert (fl["c"], fl["F_w1"], fl["F_w2"]) == (344.1, 520.0, 979.0)
