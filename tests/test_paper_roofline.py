"""scripts/paper/roofline.py: the POST-HOC whole-call roofline baseline regenerates
byte for byte from committed files, is labelled POST-HOC, and prices exactly the
cells the registered time tests scored."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import roofline as RF  # noqa: E402

DOC = ROOT / "docs/paper"


def test_committed_table_regenerates(tmp_path):
    RF.write(RF.build(), tmp_path)
    for f in ("roofline.md", "roofline.csv", "data/roofline_cells.csv"):
        assert (tmp_path / f).read_bytes() == (DOC / f).read_bytes(), f


def test_labelled_post_hoc():
    assert "POST-HOC" in (DOC / "roofline.md").read_text()
    assert "POST-HOC" in RF.__doc__ and "not a registered prediction" in RF.__doc__


def test_same_cells_and_model_scores_as_the_registered_tests():
    b = RF.build()
    by = {t["test"]: t for t in b["tests"]}
    assert by["Mixtral 8x22B"]["model"]["cells"] == 40
    assert by["Qwen2-57B-A14B"]["model"]["cells"] == 58
    assert sum(t["model"]["cells"] for t in b["tests"]) == 384
    reg = json.loads((ROOT / "scripts/scoring/crossmodel/8x22b.registered.json").read_text())
    assert abs(by["Mixtral 8x22B"]["model"]["rms"]
               - reg["time_predicted_bytes"]["shared+private"]["rms"]) < 1e-12
    rows = list(csv.DictReader((DOC / "data/roofline_cells.csv").open()))
    assert len(rows) == 384 and all(r["roof_counted_ms"] and r["roof_predicted_ms"] for r in rows)
