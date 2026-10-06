"""scripts/paper/regime.py: the regime table behind OUTLINE 2.11 regenerates byte for
byte, and its dead-CTA count matches r3_timing_model's own Qwen2-57B figure."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import regime as RG  # noqa: E402


def test_committed_table_regenerates(tmp_path):
    RG.write(RG.build(), tmp_path)
    for f in ("regime.md", "regime.csv"):
        assert (tmp_path / f).read_bytes() == (ROOT / "docs/paper" / f).read_bytes(), f


def test_rows():
    by = {r["test"]: r for r in RG.build()}
    assert [r["verdict"] for r in by.values()].count("HELD") == 3
    q = by["Qwen2-57B-A14B"]
    # r3_timing_model DEAD comment: 44,640 + 31,248 dead CTAs a call at 576 slots
    assert q["dead_ctas_n1"] == 44640 + 31248
    assert q["meets_all_three"] and q["verdict"] == "FALSIFIED"
    assert (q["ksteps_w1"], q["ksteps_w2"]) == (56, 40)
