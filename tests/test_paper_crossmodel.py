"""scripts/scoring/crossmodel/: the committed score outputs for ledger rows 10 to 43
and the runs that made them (scripts/paper/crossmodel_runs.py MANIFEST.json)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import crossmodel_runs as CR  # noqa: E402
import cross_model_score as X  # noqa: E402
import score_registered as SR  # noqa: E402

OUT = ROOT / "scripts/scoring/crossmodel"
MAN = json.loads((OUT / "MANIFEST.json").read_text())


def test_manifest_is_current_and_complete():
    assert MAN == json.loads(json.dumps(CR.manifest()))
    named = {o for r in MAN["runs"] for o in r["output"]}
    on_disk = {f"scripts/scoring/crossmodel/{p.name}" for p in OUT.iterdir()} - {
        "scripts/scoring/crossmodel/MANIFEST.json"}
    # Granite's time run refused (no VALID page): only its console line exists.
    assert on_disk == named - {"scripts/scoring/crossmodel/granite-3.0-3b-a800m.score.json"}
    for r in MAN["runs"]:
        for a in r["argv"]:
            if a.startswith("results/"):
                assert (ROOT / a).exists(), a
    rows = sorted({n for r in MAN["runs"] for n in r["ledger_rows"]})
    assert set(rows) <= set(range(10, 44))


def _argv(r):
    a = r["argv"][1:]

    def span(k, e):
        return a[a.index(k) + 1:a.index(e)]
    return dict(target=a[a.index("--target") + 1],
                source_timed=span("--source-timed", "--source-counters"),
                source_counters=a[a.index("--source-counters") + 1],
                target_timed=span("--target-timed", "--target-counters"),
                target_counters=a[a.index("--target-counters") + 1])


XM = [r for r in MAN["runs"] if r["scorer"] == "scripts/cross_model_score.py"
      and (ROOT / r["output"][0]).exists()]


@pytest.mark.parametrize("run", XM, ids=[r["output"][0].split("/")[-1] for r in XM])
def test_head_scorer_reproduces_the_registered_commits_output(run):
    """The output was made at the registration's commit; HEAD's scorer with the flags
    that restore that model version prices every cell the same."""
    a = _argv(run)
    flags = run["head_flags"]
    d = X.score(a["source_timed"], ROOT / a["source_counters"], a["target"],
                [ROOT / t for t in a["target_timed"]], ROOT / a["target_counters"],
                no_cta_fixed="--no-cta-fixed" in flags, no_cores="--no-cores" in flags)
    c = json.loads((ROOT / run["output"][0]).read_text())
    assert d["timing_params"] == pytest.approx(c["timing_params"], rel=1e-12)
    assert len(d["cells"]) == len(c["cells"])
    for x, y in zip(d["cells"], c["cells"]):
        assert (x["arm"], x["G"], x["n"]) == (y["arm"], y["G"], y["n"])
        assert x["predicted_ms"] == pytest.approx(y["predicted_ms"], rel=1e-12)
    assert d["sets"]["shared+private"]["rms"] == pytest.approx(
        c["sets"]["shared+private"]["rms"], rel=1e-12)


def test_registered_rms_values():
    """The ledger's time numbers (D/README.md:165-171, :213-240)."""
    want = {"olmoe-1b-7b": (0.0353, 12), "qwen1.5-moe-a2.7b": (0.0187, 0),
            "phi-3.5-moe": (0.0057, 0), "jetmoe-8b": (0.0717, 12)}
    for t, (rms, beyond) in want.items():
        s = json.loads((OUT / f"{t}.score.json").read_text())["sets"]["shared+private"]
        assert round(s["rms"], 4) == rms and s["beyond_5pct"] == beyond


@pytest.mark.parametrize("case", ["8x22b", "jetmoe-floor"])
def test_score_registered_regenerates_byte_identical(case, tmp_path):
    SR.main([case, "--out-dir", str(tmp_path)])
    for ext in ("json", "txt"):
        assert (tmp_path / f"{case}.registered.{ext}").read_bytes() == \
            (OUT / f"{case}.registered.{ext}").read_bytes()


def test_score_registered_key_numbers():
    d = json.loads((OUT / "8x22b.registered.json").read_text())
    t = d["time_predicted_bytes"]["shared+private"]
    assert (t["cells"], round(t["rms"], 4), round(t["worst"], 4)) == (40, 0.0171, -0.0473)
    named = d["bytes"]["named set (w1 S+P, PRIVATE w2, SHARED w2 G<=16 n<=4)"]
    assert (named["cells"], named["beyond_5pct"]) == (240, 5)
    q = json.loads((OUT / "qwen2.registered.json").read_text())
    assert round(q["time_predicted_bytes"]["shared+private"]["rms"], 4) == 0.0271
    assert round(q["bytes"]["w1 shared+private"]["rms"], 3) == 0.143
    g = json.loads((OUT / "granite.registered.json").read_text())["floor"]["r3f-g64.json"]["w2"]
    assert round(g["slope"], 1) == 459.8 and g["verdict"] == "HELD"
