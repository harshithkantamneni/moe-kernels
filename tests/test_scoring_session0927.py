"""scripts/scoring/session0927/: ledger rows 1 to 9 scored by the procedure the
fe73508/f0a831b and bb979d4 docstrings registered, on the 2026-09-27 pages.

The scorer is rerun here from the git history (`git archive` of the registering
commits, read only) and every committed output must come back as committed. The
verdicts and the differences from the hand scoring are pinned, so a change that
moves one is seen moving it. Needs the history: the repository itself, or a clone
named by MOE_HISTORY_REPO; without the commits the module is skipped."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "scoring" / "session0927"
sys.path.insert(0, str(OUT))

import score as SC  # noqa: E402

REPO = Path(os.environ.get("MOE_HISTORY_REPO", ROOT))


def _have_history() -> bool:
    return all(subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{c}^{{commit}}"],
                              capture_output=True).returncode == 0 for c in SC.C.values())


pytestmark = pytest.mark.skipif(
    not _have_history(), reason=f"{REPO} lacks the registering commits (set MOE_HISTORY_REPO)")

FILES = ("score.txt", "prerun/r3_timing_model.f0a831b.txt", "prerun/r3_timing_model.fe73508.txt",
         "prerun/wave_split_bytes.bb979d4.txt")


@pytest.fixture(scope="module")
def rerun(tmp_path_factory):
    out = tmp_path_factory.mktemp("session0927")
    assert SC.main(["--repo", str(REPO), "--out", str(out)]) == 0
    return out


def _close(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), path
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (path, a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]")
    else:
        assert a == b, (path, a, b)


def test_a_rerun_reproduces_every_committed_output(rerun):
    for f in FILES:
        assert (rerun / f).read_text() == (OUT / f).read_text(), f
    _close(json.loads((rerun / "score.json").read_text()),
           json.loads((OUT / "score.json").read_text()))
    assert sorted(p.name for p in (rerun / "prerun").iterdir()) == sorted(
        p.name for p in (OUT / "prerun").iterdir())


def test_the_procedure_verdicts_and_where_they_part_from_the_hand_scoring():
    doc = json.loads((OUT / "score.json").read_text())
    got = {int(k): v["scorer"] for k, v in doc["verdicts"].items()}
    assert got == {1: "HELD", 2: "FALSIFIED", 3: "HELD", 4: "HELD", 5: "FALSIFIED",
                   6: "UNDECIDED", 7: "HELD", 8: "FALSIFIED", 9: "not answered"}
    assert [int(k) for k, v in doc["verdicts"].items() if not v["agree"]] == [2, 6]
    # Row 2: G=32 n=4->5 reads 0.53754 ms, under the registered 0.538; inside only when rounded.
    g32 = doc["rows"]["2"]["G"]["32"]
    assert list(g32["outside"]) == ["n4->5"]
    assert g32["outside"]["n4->5"] == pytest.approx(0.53754, abs=5e-6)
    assert g32["inside_if_rounded_to_3_decimals"]
    # Row 6: achieved occupancy sits at the limit, which the hand scoring never read.
    r6 = doc["rows"]["6"]
    assert r6["achieved_over_limit_range"]["w1"][0] > 0.97
    assert r6["reading_loose"]["AND form (docstring falsifier list, FALSIFIERS)"] == "HELD"
    assert r6["reading_loose"]["OR form (printed falsified-if line)"] == "FALSIFIED"
    differs = {(c["row"], c["what"]) for c in doc["compare"] if not c["agree"]}
    assert differs == {(1, "G=4 slopes 2-6 at 1410/1500/1605"), (2, "verdict"), (6, "verdict"),
                       (6, "w1 cycles range"), (8, "w2 SHARED G=16 n=5, 7, 8 (% vs registered)"),
                       (8, "w2 SHARED G=4 n=8 (%)"), (8, "w2 SHARED G=2 n=5, 7, 8 (%)"),
                       (8, "every w1 and PRIVATE cell within 1.5% (README, beside the test)")}


def test_group_a_quotes_sit_in_the_registering_docstrings_and_group_b_values_do_not():
    doc = json.loads((OUT / "score.json").read_text())
    for q in doc["quotes"]:
        assert q["lines"], q
        assert q["in_docstring"] == (q["what"] != "printed by the tool (not docstring)"), q
    assert {b["group"] for b in doc["group_b"]} == {"b"}
    # The recomputed registered numbers are the session README's quotations.
    assert all(c["agree"] for c in doc["compare"] if "recomputed" in c["what"])
    assert doc["fe73508_vs_f0a831b"]["only_P1_third_clock_differs"]


def test_the_m3_facts():
    F = json.loads((OUT / "score.json").read_text())["facts"]
    for reg in ("timing", "wsc", "p1_1605"):
        assert F["ancestry"][f"{reg} -> run"] and F["ancestry"][f"{reg} -> pages_0927"]
        assert F["ancestry"][f"pages_0925 -> {reg}"]
    assert not F["ancestry"]["timing -> wsc"] and not F["ancestry"]["wsc -> timing"]
    assert F["ancestry"]["timing -> p1_1605"] and F["ancestry"]["wsc -> p1_1605"]
    assert F["tree_0925_pages_one_tree"] and F["tree_0927_pages"]["run"] is None


def test_the_band_rule_is_inclusive_and_unrounded():
    lad = {n: 0.5 * n for n in range(1, 7)}
    meas = [{"run": "x", "G": G, "label": "VALID", "clocks": [1710.0],
             "shared_ms": {str(n): v for n, v in lad.items()}} for G in (8, 32)]
    band = {"T": [], "inc": [], "step_1_2": 0.0, "slope_2_6": 0.5, "band": [0.5, 0.556]}
    T = {"predictions": {"P2": {"8": band, "32": band}}}
    assert SC.row2(T, meas)["verdict"] == "HELD"
    meas[1]["shared_ms"]["5"] = 2.5 - 0.0001
    r = SC.row2(T, meas)
    assert r["verdict"] == "FALSIFIED" and r["G"]["32"]["inside_if_rounded_to_3_decimals"]
