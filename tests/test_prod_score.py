"""scripts/prod/score_prod.py off the GPU: the metrics, the step bootstrap, the per-step totals, the v8 binding's
interface checks, and the whole scorer on synthetic runs (extracted by the real nsys_extract) with a stub pricer
in v8's registered interface."""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nsys_extract as NX  # noqa: E402
import prod_fakes as F  # noqa: E402
import prodlib as PL  # noqa: E402
import score_prod as SP  # noqa: E402

MODEL = PL.MODELS["olmoe-1b-7b"]

STUB = textwrap.dedent('''
    ROOFLINE = {"decode": {"T0_ms": 0.01, "bw_gbps": 3600.0}, "prefill": {"T0_ms": 0.02, "bw_gbps": 3500.0}}

    def price(call):
        T, counts = call["T"], call["counts"]
        assert sum(counts) == T * call["model_cfg"].top_k and call["ids"] is not None
        if 8192 <= T <= 8255:
            return {"ms": None, "tile": "128x128x64", "regime": "prefill", "why": "HOPPER_G1_N refused"}
        A = sum(1 for c in counts if c)
        return {"ms": 0.03 + 0.001 * A + 1e-5 * T, "tile": "t", "regime": "decode" if T <= 128 else "prefill"}
''')


@pytest.fixture()
def stub(tmp_path, monkeypatch):
    d = tmp_path / "stubmod"
    d.mkdir()
    (d / "stub_v8.py").write_text(STUB)
    monkeypatch.syspath_prepend(str(d))
    return "stub_v8"


def test_metrics_on_known_numbers():
    m = SP.metrics([1.1, 0.9, 2.0], [1.0, 1.0, 2.0])
    assert m["n"] == 3 and m["MAPE"] == pytest.approx(100 * 0.2 / 3)
    assert m["wMAPE"] == pytest.approx(100 * 0.2 / 4.0) and m["bias"] == pytest.approx(0.0, abs=1e-12)
    assert m["p50_abs_e"] == pytest.approx(10.0)
    assert SP.metrics([np.nan, 1.0], [1.0, 0.0])["n"] == 0


def test_bootstrap_resamples_whole_steps_and_brackets_the_point():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.05, 0.2, 400)
    p = t * (1 + rng.normal(0.02, 0.05, 400))
    steps = np.repeat(np.arange(40), 10)
    ci = SP.bootstrap_by_step(p, t, steps)
    m = SP.metrics(p, t)
    assert ci["steps"] == 40 and ci["MAPE_ci95"][0] <= m["MAPE"] <= ci["MAPE_ci95"][1]
    assert ci == SP.bootstrap_by_step(p, t, steps)          # seeded


def test_step_totals_sum_every_layer_and_drop_incomplete_steps():
    rows = [{"_run": "a", "step": 0, "p": 1.0, "t": 2.0}, {"_run": "a", "step": 0, "p": 1.0, "t": 2.0},
            {"_run": "a", "step": 1, "p": 1.0, "t": None}, {"_run": "a", "step": 1, "p": 1.0, "t": 1.0}]
    ps, ts, keys = SP.step_totals(rows, "p", "t")
    assert ps == [2.0] and ts == [4.0] and keys == [("a", 0)]


def test_roofline_and_off_lock():
    cfg = MODEL.cfg
    W = 3 * cfg.hidden_size * cfg.intermediate_size * 2
    assert SP.roofline_ms(cfg, [1, 0, 3] + [0] * 61, 0.01, 1000.0) == pytest.approx(0.01 + 2 * W / 1e9)
    assert SP.off_lock(1650, 1710) and not SP.off_lock(1700, 1710) and not SP.off_lock(1980, None)


def test_the_v8_binding_holds_v8_to_its_interface(tmp_path, monkeypatch):
    d = tmp_path / "m"
    d.mkdir()
    (d / "bad_v8.py").write_text("def price(c):\n    return {'ms': 1.0}\n")
    (d / "mute_v8.py").write_text("def price(c):\n    return {'ms': None, 'tile': 'x', 'regime': 'decode'}\n")
    monkeypatch.syspath_prepend(str(d))
    with pytest.raises(PL.KitError, match="registered interface"):
        PL.price_v8(MODEL.cfg, 8, [8] * 8 + [0] * 56, module="bad_v8")
    with pytest.raises(PL.KitError, match="without a why"):
        PL.price_v8(MODEL.cfg, 8, [8] * 8 + [0] * 56, module="mute_v8")


def _extracted(tmp_path, name, steps, off=()):
    run, calls = F.make_run(tmp_path / name, steps=steps)
    ev, marks, _x = F.timeline(MODEL.cfg, calls)
    F.write_clocks(run, len(calls) // 4 + 2, mhz=float(SP.lock_of(name) or 1980), off=off)
    db = F.write_sqlite(tmp_path / f"{name}.sqlite", ev, marks)
    assert NX.write(NX.extract(run, NX.read_sqlite(db), MODEL), run, "") == 0
    return run, calls


def _replay(tmp_path, run, n=6):
    rows = PL.read_jsonl(run / "calls.jsonl")[:n]
    d = tmp_path / "replay"
    d.mkdir()
    with open(d / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "status", "ms_p50", "clock_drift_ok"])
        w.writeheader()
        for r in rows:
            for rep in range(3):
                w.writerow({"label": f"g{r['g']}", "status": "ok", "ms_p50": 0.05 + 0.001 * rep, "clock_drift_ok": True})
    return d


def test_the_scorer_end_to_end(tmp_path, stub):
    a, calls = _extracted(tmp_path, "A1710", ((5, 8), (40, 40), (100, 104), (300, 304)), off=(2,))
    b, _ = _extracted(tmp_path, "Adefault", ((5, 8), (300, 304)))
    rep = _replay(tmp_path, a)
    fr = tmp_path / "fr.jsonl"
    with open(fr, "w") as fh:
        for r in PL.read_jsonl(a / "calls.jsonl"):
            fh.write(json.dumps({"run": "A1710", "g": r["g"], "arm": "primary-uniform", "pred_ms": 0.04}) + "\n")
    out = tmp_path / "score"
    rc = SP.main(["--model", MODEL.key, "--run", f"A1710={a}", "--run", f"Adefault={b}", "--replay", f"decode={rep}",
                  "--frontier", str(fr), "--pricer", stub, "--workers", "1", "--out", str(out)])
    assert rc == 0
    sc = json.loads((out / "prod.score.json").read_text())
    r1 = sc["runs"]["A1710"]
    assert r1["lock"] == 1710 and r1["off_lock_dropped"] == 4           # the 'clk 2' window ran off the lock
    t = r1["tables"]["v8_ms vs kernel_sum_ms"]
    n_dec = sum(1 for c in calls if c["n"] <= 128) - 4
    assert t["all"]["n"] == n_dec and set(t["branch"]) == {"decode", "prefill"} and t["branch"]["prefill"]["n"] == 0
    assert "MAPE_ci95" in t["all"] and t["per_step_total"]["n"] == 3
    assert "roof_ms vs span_ms" in r1["tables"] and "frontier:primary-uniform vs kernel_sum_ms" in r1["tables"]
    assert sc["runs"]["Adefault"]["role"].startswith("production delta only")
    assert sc["runs"]["replay-decode"]["tables"]["v8_ms vs replay_ms"]["all"]["n"] == 6
    assert "Adefault vs A1710" in sc["deltas"]
    assert "DESCRIPTIVE" in (out / "prod.score.txt").read_text()


def test_refused_calls_are_counted_with_their_share_of_moe_time(tmp_path, stub):
    a, _ = _extracted(tmp_path, "A1410", ((8200, 8200), (600, 600)))
    out = tmp_path / "score"
    assert SP.main(["--model", MODEL.key, "--run", f"A1410={a}", "--pricer", stub, "--workers", "2",
                    "--out", str(out)]) == 0
    r = json.loads((out / "prod.score.json").read_text())["runs"]["A1410"]
    assert r["v8_refusals"] == {"HOPPER_G1_N refused": 16}
    assert r["v8_refused_share_of_moe_time"] == pytest.approx(0.5)
    st = r["tables"]["v8_ms vs kernel_sum_ms"]["stratum"]
    assert "refused_window_8192_8255" not in st                       # refused: no prediction, not scored


def test_a_pricer_that_raises_fails_the_score(tmp_path, monkeypatch):
    d = tmp_path / "m"
    d.mkdir()
    (d / "boom_v8.py").write_text("def price(c):\n    raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(d))
    a, _ = _extracted(tmp_path, "A1710", ((5, 8),))
    assert SP.main(["--model", MODEL.key, "--run", f"A1710={a}", "--pricer", "boom_v8", "--workers", "1",
                    "--roofline", str(_roof(tmp_path)), "--out", str(tmp_path / "s")]) == 3


def _roof(tmp_path):
    p = tmp_path / "roof.json"
    p.write_text(json.dumps({"decode": {"T0_ms": 0.01, "bw_gbps": 3600}, "prefill": {"T0_ms": 0.02, "bw_gbps": 3500}}))
    return p
