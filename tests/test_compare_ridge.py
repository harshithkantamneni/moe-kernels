"""`scripts/compare.py` no longer carries a numeric ridge default.

Until 2026-09-08 it held `RIDGE_DEFAULT = 166.0` (retired) and printed a per-row
COMPUTE/memory regime against it on both cards' tables with no source line:
166 is no card's ridge (H200 162.8, A100 145.8 from the committed
calibrations), so rows/expert between a card's own ridge and 166 read
"memory" when the card's own ruler says compute. Retraction (e), "no --ridge
default is a number" (c42a20c), had reached every other `--ridge` in scripts/
and missed this one: the recurring defect, a fix at one of two call sites.

Three paths, each pinned here PASS and FAIL: the rows' own card resolves the
ridge through `roofline.hardware_for_rows`; rows naming no card get the
column REFUSED by name with the fix in the source line; `--ridge` is printed
as the operator's assertion. The script's own `--self-test` plants the same
three, so the proof is runnable without pytest as well.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe.bench import roofline  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("compare_under_test",
                                                  ROOT / "scripts" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: a frozen dataclass under `from __future__ import
    # annotations` resolves its field types through sys.modules[__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(t: int, gpu: str, dtype: str = "bf16") -> dict:
    return {"impl": "vllm_fused_experts", "num_tokens": str(t), "ms_p50": "1.0",
            "model": "mixtral-8x7b", "dtype": dtype, "routing_kind": "uniform",
            "routing_param": "0", "seed": "0", "l2_flush": "True",
            "cuda_graph": "False", "correctness_passed": "True", "covers": "all",
            "load_active_experts": "8", "load_total_rows": str(2 * t),
            "gpu_name": gpu}


def test_no_numeric_ridge_default_survives_in_the_module():
    mod = _load()
    assert not hasattr(mod, "RIDGE_DEFAULT")
    source = (ROOT / "scripts" / "compare.py").read_text()
    assert "default=RIDGE_DEFAULT" not in source
    assert 'ap.add_argument("--ridge", type=float, default=None' in source


def test_rows_naming_a_calibrated_card_resolve_to_that_cards_own_ridge():
    """PASS side: the number has a file behind it and is neither the retired
    166.0 nor an end of the withdrawn (160.3, 176.2) pair."""
    mod = _load()
    own = roofline.load_hardware("measured_nvidia_h200").ridge_point("bf16")
    res = mod.resolve_ridge([_row(700, "NVIDIA H200")], None)
    assert not res.refused and not res.asserted
    assert res.by_dtype == {"bf16": own}
    # The number is whatever the committed calibration says today (162.8 on
    # the 2026-09-02 file, 152.8 on the 2026-09-09 one). What is pinned is
    # that it CAME from that file and is printed in the source line, not the
    # value: a test that pins the value goes stale on the next calibration
    # and then asserts a ridge no card has.
    assert own == pytest.approx(
        roofline.load_hardware("measured_nvidia_h200").ridge_point("bf16"))
    assert "H200" in res.source and f"{own:.1f}" in res.source
    assert not any(abs(own - w) < 0.05 for w in (166.0, 160.3, 176.2))
    assert mod.regime(own * 1.02, {"bf16"}, res) == "COMPUTE"
    assert mod.regime(own * 0.98, {"bf16"}, res) == "memory"


def test_the_a100_resolves_to_its_own_ridge_not_the_h200s_or_166():
    mod = _load()
    a100 = roofline.load_hardware("measured_nvidia_a100_sxm4_80gb").ridge_point("bf16")
    res = mod.resolve_ridge([_row(600, "NVIDIA A100-SXM4-80GB")], None)
    assert res.by_dtype == {"bf16": a100}
    assert a100 == pytest.approx(145.8, abs=0.05)
    # 150 rows/expert: compute on the A100's own ruler, "memory" against 166.
    assert mod.regime(150.0, {"bf16"}, res) == "COMPUTE"


def test_rows_naming_no_card_get_the_column_refused_by_name():
    """FAIL side, planted: no number at all, and the fix is in the line."""
    mod = _load()
    res = mod.resolve_ridge([_row(700, "")], None)
    assert res.refused and res.by_dtype == {}
    assert res.source.startswith("REFUSED")
    assert "--ridge" in res.source
    assert mod.regime(1e9, {"bf16"}, res) == mod.NO_RIDGE


def test_a_dtype_with_no_verified_peak_is_refused_per_row_not_defaulted():
    mod = _load()
    res = mod.resolve_ridge([_row(700, "NVIDIA H200", dtype="int4")], None)
    assert res.refused and "int4" in res.source


def test_an_asserted_ridge_is_printed_as_an_assertion_and_scored_against():
    mod = _load()
    res = mod.resolve_ridge([_row(700, "")], 150.0)
    assert res.asserted and "ASSERTED" in res.source and "--ridge" in res.source
    assert mod.regime(151.0, {"bf16"}, res) == "COMPUTE"
    assert mod.regime(149.0, {"bf16"}, res) == "memory"
    with pytest.raises(ValueError):
        mod.resolve_ridge([_row(700, "")], 0.0)


def test_two_dtypes_at_one_batch_with_different_ridges_read_mixed():
    mod = _load()
    res = mod.resolve_ridge([_row(700, "NVIDIA H200"),
                             _row(700, "NVIDIA H200", dtype="fp8_e4m3")], None)
    if len(res.by_dtype) < 2:
        pytest.skip("the committed H200 yaml carries no verified fp8 peak")
    assert mod.regime(700.0, {"bf16", "fp8_e4m3"}, res) == mod.MIXED_RIDGE


def test_the_scripts_own_self_test_passes_and_prints_its_result_line():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare.py"),
                          "--self-test"], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "RESULT: compare --self-test PASS" in out.stdout
    assert "FAIL" not in out.stdout.replace("PASS", "")


def test_the_published_a100_arm_prints_its_own_ridge_in_the_source_line():
    """The invocation the audit executed: it used to print a regime column
    with no ridge source line at all."""
    arm = ROOT / "results/published/2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card"
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare.py"),
                          "--results", str(arm), "--model", "mixtral-8x7b",
                          "--routing", "any"], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr
    assert "regime column: ridge from the rows' own card, NVIDIA A100" in out.stdout
    assert "145.8 Op/B" in out.stdout
    assert "166" not in out.stdout.split("regime column")[1].split("\n")[0]
