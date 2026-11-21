"""Rental 7's draws, prices and registrations off the GPU (design-r7 S10, design-r7-review (c), (e)
and (h); owner decisions of 2026-10-13): skew_synth7 --check7, the fresh seeds on rental 6's
treads, the controls as SEEN replicates, the l2open page priced row for row, the power checked on
MPS, the pins of every earlier registration, and the four registrations' --check."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R7DIR = REPO / "scripts" / "scoring" / "rental7"
for p in (REPO, REPO / "scripts", R7DIR, REPO / "scripts" / "scoring" / "rental6"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r7common as C7  # noqa: E402
import skew_synth as SS  # noqa: E402
import skew_synth7 as S7  # noqa: E402

REG = REPO / "docs" / "registered"
H7 = REPO / C7.HIST_DIR
H6 = REPO / "docs" / "registered" / "2026-10-09-rental6-skew-hist"
REGS = {p: C7.registration(REPO, p) for p in C7.PARTS}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_the_rental7_pages_recompute_byte_for_byte_and_rental6s_are_untouched():
    assert S7.main(["--check7"]) == 0
    assert SS.main(["--check6"]) == 0


def test_the_skew_draws_are_rental6s_rule_at_fresh_seeds_on_the_same_treads():
    for m, ns in (("mixtral-8x7b", (3, 6, 16)), ("phi-3.5-moe", (2, 4, 12))):
        for page in ("A", "B", "C"):
            d7 = json.loads((H7 / f"{m}-{page}.json").read_text())
            d6 = json.loads((H6 / f"{m}-{page}.json").read_text())
            assert sorted({c["n"] for c in d7["cells"]}) == list(ns) and d7["shuffle_seed"] == 20261020
            assert [(c["label"], c["n"], c.get("arms")) for c in d7["cells"]] == [(c["label"], c["n"], c.get("arms")) for c in d6["cells"]]
            for c7, c6 in zip(d7["cells"], d6["cells"], strict=True):
                if c7.get("provenance"):
                    p7, p6 = c7["provenance"], c6["provenance"]
                    # rental 6's seed rule at base 20261020: seed(n) is rental 6's seed(n + 1)
                    assert p7["seed"] == p6["seed"] + 10 and (p7["kind"], p7["param"], p7["B"]) == (p6["kind"], p6["param"], p6["B"])
                    assert c7["counts"] != c6["counts"]
                else:
                    assert c7["counts"] == c6["counts"]      # uniform and balanced: rental 6's counts
    lk = REGS["skew"]["leakage"]
    assert lk["skewed_cells"] == 48 and lk["skewed_cells_whose_counts_sha256_is_published"] == 0
    assert lk["control_cells"] == 72 and lk["control_cells_whose_counts_sha256_is_published"] == 72


def test_the_l2open_page_is_the_priced_draw_and_every_row_is_priced():
    doc = json.loads((H7 / "olmoe-1b-7b-l2open.json").read_text())
    assert doc["shuffle_seed"] == 20261030
    cells = [(c["label"], c["n"], arm) for c in doc["cells"] for arm in c["arms"]]
    assert len(cells) == 20 and {c[1] for c in cells} == {4, 8, 16}
    assert all(c["arms"] == ["native"] for c in doc["cells"] if c["n"] == 16)
    for c in doc["cells"]:
        if c.get("provenance"):
            assert c["provenance"]["seed"] == 20261030 + 10 * c["n"] + ("PT", "PW").index(c["label"])
    rk = json.loads((H7 / "l2open_rowkey.json").read_text())
    assert sorted((c["label"], c["n"], c["arm"]) for c in rk["cells"]) == sorted(cells)
    assert rk["registered_C_MiB"] == 60.0 and rk["C_MiB"] == [39.5, 60.0]
    assert rk["device"] == "mps" and rk["cpu_float64_check"]["cells"] == 20
    assert rk["cpu_float64_check"]["max_rel_diff_read_bytes"] < 1e-6
    pred = REGS["l2open"]["byte_predictions"]
    assert sorted((r["label"], r["n"], r["arm"]) for r in pred) == sorted(cells)
    assert REGS["l2open"]["rivals"]["file"]["sha256"] == _sha(H7 / "l2open_rowkey.json")
    # the design's priced NATIVE draws (olmoe_w1.json "discriminating", seed base 20261030) reproduce
    sep = {(r["label"], r["n"]): r["gaps_w1_pct"] for r in pred if r["arm"] == "native" and r["skewed"]}
    assert sep[("PT", 4)]["ROWKEY60_vs_ROWKEY39.5"] == pytest.approx(-14.16, abs=0.05)
    assert sep[("PW", 4)]["ROWKEY60_vs_ROWKEY39.5"] == pytest.approx(-11.72, abs=0.05)
    assert sep[("PT", 16)]["v3_vs_ROWKEY60"] == pytest.approx(29.03, abs=0.05)


def test_every_registered_power_point_agrees_on_mps_float32():
    chk = json.loads((H7 / "power7_check.json").read_text())
    assert chk["max_abs_diff"] < 1e-3
    assert chk["cpu_float64"]["E1"] == REGS["skew"]["power"]["scenarios"]
    assert chk["cpu_float64"]["Q1-H"] == REGS["q1"]["tests"]["Q1-H"]["power"]["points"]
    assert chk["cpu_float64"]["Q1-C"] == REGS["q1"]["tests"]["Q1-C"]["power"]["points"]
    assert REGS["skew"]["power"]["mps_check"]["sha256"] == _sha(H7 / "power7_check.json")


def test_q1h_is_registered_as_a_replicate_with_the_added_clusters_labelled():
    h = REGS["q1"]["tests"]["Q1-H"]
    pairs = h["pairs"]
    assert len(pairs) == 24 and len(h["clusters"]) == 5
    assert sum(p["copies"] == 15 for p in pairs) == 3 and h["label"].startswith("REPLICATE: 21 of its 24 pairs")
    added = [p for p in pairs if p["population"].startswith("ADDED")]
    assert {(p["model"], p["n"]) for p in added} == {("mixtral-8x7b", 6), ("phi-3.5-moe", 4)} and len(added) == 12
    assert all("SEEN-SELECTED" in p["population"] for p in added)
    assert all(p["native_path"] == "block-scan" and p["host_ok"] for p in pairs)
    # rental 6's Q1 predictions are carried verbatim
    assert REGS["q1"]["predictions"] == C7.C6.registration(REPO, "q1")["predictions"]


def test_the_ten_scorer_ambiguities_are_resolved_in_every_registration():
    for p in ("skew", "q1", "l2open"):
        amb = REGS[p]["scorer_ambiguities"]
        assert len(amb) == 10 and all(k.split(" ", 1)[0] == str(i) for i, k in enumerate(amb, 1))


def test_model_v3_is_unchanged_and_no_term_is_added():
    for f, pin in json.loads((REG / "2026-10-09-rental6-v3-gh200.json").read_text())["code_pins"].items():
        if f in ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/r3_timing_model_v3.py",
                 "scripts/wave_split_bytes_v3.py", "scripts/scoring/rental5/skewmodel.py", "scripts/scoring/rental6/r6model.py"):
            assert _sha(REPO / f) == pin, f


def test_every_file_an_earlier_registration_pins_is_byte_identical_to_the_baseline():
    """The owner's preference (2026-10-13): a registered code pin stays byte-identical. Every file a
    rental-6 registration pins, R3 included (sha256 daa40706...), matches its pin, and rental 7
    pins the same R3."""
    for f in sorted(REG.glob("2026-10-09-rental6-*-gh200.json")):
        for path, pin in json.loads(f.read_text())["code_pins"].items():
            assert _sha(REPO / path) == pin, (f.name, path)
    r3 = _sha(REPO / "scripts" / "private_weight_reference.py")
    assert r3.startswith("daa40706")
    for p in C7.PARTS:
        assert REGS[p]["code_pins"]["scripts/private_weight_reference.py"] == r3


def test_the_registrations_recompute():
    got = subprocess.run([sys.executable, str(R7DIR / "register.py"), "--check"], capture_output=True, text=True,
                         timeout=900, cwd=REPO, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert got.returncode == 0, got.stdout + got.stderr[-2000:]
    assert got.stdout.count("same") == 8
