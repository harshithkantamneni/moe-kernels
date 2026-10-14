"""scripts/decode_synth.py: the committed decode pages and coverage table are reproduced byte for
byte; every page passes r3_decode's own check; the ladder, controls and G3-D pairs are what the
registration says; the trace draws are labelled EXTRAPOLATED; --extract writes statistics only and
never inside the repository."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_synth as DS  # noqa: E402
import r3_decode as RD  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402


def test_check_reproduces_every_committed_file():
    files = DS.all_files()
    for name, text in files.items():
        assert (DS.OUT / name).read_text() == text, name
    assert len(files) == len(DS.page_names()) + 1


@pytest.mark.parametrize("model, kind", DS.page_names())
def test_every_page_is_one_r3_decode_runs(model, kind):
    page = RD.load_page(DS.OUT / f"{model}-{kind}.json", MC[model], model)
    assert page.cells and all(1 <= c.T <= 128 for c in page.cells)
    if kind == "decl":
        assert page.decl_copies == DS.DECL_COPIES and all(c.arms == ("native", "decl") for c in page.cells)
    else:
        assert page.decl_copies is None


@pytest.mark.parametrize("model", DS.MODELS)
def test_the_main_page_holds_the_ladder_the_controls_and_every_path(model):
    cfg = MC[model]
    doc = json.loads((DS.OUT / f"{model}-main.json").read_text())
    by = {c["label"]: c for c in doc["cells"]}
    for T in DS.ts_of(model):
        lv = [c for c in doc["cells"] if c["T"] == T and c["label"].startswith("L-")]
        As = sorted(sum(1 for v in c["counts"] if v) for c in lv)
        assert As[0] == cfg.top_k and As[-1] == min(cfg.num_experts, T * cfg.top_k)
        for c in lv:
            nz = [v for v in c["counts"] if v]
            assert max(nz) - min(nz) <= 1                       # balanced over its active set
        assert f"U-T{T}" in by
    paths = {RD.geometry(cfg, c["T"], c["counts"]).path for c in doc["cells"]}
    assert RD.NAIVE in paths and RD.SMALL_BATCH in paths
    for T in DS.G3_LABEL_TS:
        a, b = by[f"MIN-T{T}"], by[f"L-T{T}-A{cfg.top_k}"]
        assert sorted(a["counts"]) == sorted(b["counts"]) and a["counts"] != b["counts"]
    for T in DS.G3_SHUFFLE_TS:
        assert by[f"US-T{T}"]["counts"] == by[f"U-T{T}"]["counts"] and by[f"US-T{T}"]["shuffle"] is False
    dw = [c for c in doc["cells"] if c["role"].startswith("TRACE-DW")]
    assert (len(dw) == len(DS.TRACE_TS)) == (model in DS.dw_shapes())
    assert all("EXTRAPOLATED" in c["role"] for c in dw)


def test_qwen15s_naive_t3_is_on_its_page():
    assert 3 in DS.ts_of("qwen1.5-moe-a2.7b") and 3 not in DS.ts_of("olmoe-1b-7b")


def test_the_coverage_table_is_statistics_only_and_reproduces_the_designs_finding():
    d = json.loads((DS.OUT / "coverage_stats.json").read_text())
    assert "NOT RUN" in d["primary_extract"] and "gone" in d["primary_extract"]
    keys = set().union(*(r.keys() for r in d["rows"]))
    assert not any("count" in k and "p_" not in k for k in keys)
    row = {(r["model"], r["shape"], r["B"]): r for r in d["rows"]}
    assert row[("olmoe-1b-7b", "uniform", 16)]["coverage_p50"] == pytest.approx(0.89, abs=0.01)
    assert row[("olmoe-1b-7b", "DW", 16)]["coverage_p50"] == pytest.approx(0.80, abs=0.01)
    assert row[("mixtral-8x7b", "uniform", 4)]["coverage_p50"] == pytest.approx(0.62, abs=0.01)


def test_extract_reads_a_log_into_statistics_and_refuses_the_repository(tmp_path):
    rng = random.Random(0)
    log = tmp_path / "log.jsonl"
    with log.open("w") as fh:
        for prompt in range(160):
            for pos in range(3):
                for layer in range(2):
                    fh.write(json.dumps({"prompt": prompt, "pos": pos, "layer": layer,
                                         "experts": rng.sample(range(8), 2)}) + "\n")
    got = DS.extract(log, "mixtral-8x7b", batches=20)
    assert got["rows"] and {r["B"] for r in got["rows"]} == set(DS.COVERAGE_BS)
    assert all(0 < r["coverage_p50_median_layer"] <= 1 for r in got["rows"])
    assert "experts" not in json.dumps(got) and "counts" not in json.dumps(got)
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "decode_synth.py"), "--extract", str(log), "--model",
                          "mixtral-8x7b", "--out", str(REPO / "x.json")], capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "outside the repository" in bad.stdout and not (REPO / "x.json").exists()
