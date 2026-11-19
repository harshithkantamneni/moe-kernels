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
    def price(call):
        T, counts = call["T"], call["counts"]
        assert sum(counts) == T * call["model"].top_k and call["ids"] is not None
        if T > 128:
            assert len(call["sag128"]) == 2 and call["sag64"] == (30.365, 300.0) and call["knee"] == (6.0, 2.75)
        if 8192 <= T <= 8255:
            return {"ms": None, "tile": "128x128x64", "regime": "prefill", "why": "HOPPER_G1_N refused"}
        A = sum(1 for c in counts if c)
        sag = call.get("sag128", (0.0, 0.0))[0]
        return {"ms": 0.03 + 0.001 * A + 1e-5 * T + 1e-6 * sag, "tile": "t",
                "regime": "decode" if T <= 128 else "prefill"}
''')
BOARD_PAR = (64.325, 612.5)


def _par(tmp_path, par=BOARD_PAR, verdict="PASS"):
    p = tmp_path / "par.json"
    p.write_text(json.dumps({"schema": PL.PAR_SCHEMA, "verdict": verdict, "par": list(par)}))
    return str(p)


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
    c = {"T0_GR_ms": 0.01, "bw_gbps": 1000.0}
    assert PL.roofline_ms(cfg, [1, 0, 3] + [0] * 61, c) == pytest.approx(0.01 + 2 * W / 1e9)
    assert SP.off_lock(1650, 1710) and not SP.off_lock(1700, 1710) and not SP.off_lock(1980, None)


def test_the_roofline_is_the_decode_blinds_registered_rival_on_every_one_of_its_cells():
    """MAJOR 4: prodlib.roofline_ms reproduces the registered roofline of every cell of the v7 decode blind
    (registration predictions.cells ms.roofline, by A) and of every scored row (scripts/scoring/v7blind/
    blind.score.json rows pred['roofline']), bit for bit."""
    from moe.spec import MODEL_CONFIGS
    reg = json.loads(PL.ROOFLINE_REG.read_text())["predictions"]["cells"]
    c = PL.roofline_constants()
    n = 0
    for cell in reg:
        want = (cell.get("ms") or {}).get("roofline")
        if want is None:
            continue
        assert PL.roofline_ms_A(MODEL_CONFIGS[cell["model"]], cell["A"], c) == want, cell["label"]
        n += 1
    assert n == len(reg) == 143
    A = {(x["model"], x["label"], x["T"]): x["A"] for x in reg}
    rows = json.loads((REPO / "scripts/scoring/v7blind/blind.score.json").read_text())["rows"]
    got = [(PL.roofline_ms_A(MODEL_CONFIGS[r["model"]], A[(r["model"], r["label"], r["T"])], c), r["pred"]["roofline"])
           for r in rows if "roofline" in r.get("pred", {})]
    assert len(got) > 100 and all(a == b for a, b in got)


def test_the_registered_prefill_constants_are_their_sources_values():
    import ast
    import re

    import round5_chain as R5
    src = (REPO / "scripts/scoring/v7pblind/v7pcommon.py").read_text()
    a2 = ast.literal_eval(re.search(r"^A2_PAR_64 = (.+)$", src, re.M).group(1))
    ref = ast.literal_eval(re.search(r"^REF_PARS = (\{.+\})$", src, re.M).group(1))
    assert tuple(PL.PREFILL_REGISTERED["sag64"]) == a2 and tuple(PL.PREFILL_REGISTERED["knee"]) == tuple(R5.K1_SEEN)
    assert PL.REF_PARS == ref and PL.REF_PAR_MEDIAN == (48.135000000000005, 506.25)
    with pytest.raises(PL.KitError, match="no 128x128 sag par"):
        PL.prefill_constants(None)


def test_prodlib_sends_v8s_interface_and_reuses_v8s_ernie():
    import r3_timing_model_v8 as V8
    assert PL.ERNIE is V8.ERNIE and PL.MODELS["ernie-4.5-21b-a3b"].cfg is V8.ERNIE
    assert PL.refused_window(PL.ERNIE) == (8192, 8255) == PL.REFUSED_WINDOW
    assert PL.refused_window(MODEL.cfg) == (6144, 8255)
    assert PL.strata(6200, 8, 64) == [PL.REFUSED_STRATUM] and PL.strata(6200, 6, 64) == []


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
                  "--frontier", str(fr), "--pricer", stub, "--workers", "1", "--sag", _par(tmp_path), "--out", str(out)])
    assert rc == 0
    sc = json.loads((out / "prod.score.json").read_text())
    r1 = sc["runs"]["A1710"]
    assert r1["lock"] == 1710 and r1["off_lock_dropped"] == 4           # the 'clk 2' window ran off the lock
    t = r1["tables"]["v8_ms vs kernel_sum_ms"]
    n_dec = sum(1 for c in calls if c["n"] <= 128) - 4
    assert t["all"]["n"] == n_dec and set(t["branch"]) == {"decode", "prefill"} and t["branch"]["prefill"]["n"] == 0
    assert "MAPE_ci95" in t["all"] and t["per_step_total"]["n"] == 3
    assert "roof_ms vs span_ms" in r1["tables"] and "frontier:primary-uniform vs kernel_sum_ms" in r1["tables"]
    assert "v8_refpar_ms vs kernel_sum_ms" in r1["tables"] and sc["constants"]["sag128 (board, measured at 1410)"] == list(BOARD_PAR)
    assert not any("NOT SCORED" in n for n in sc["notes"])
    assert sc["runs"]["Adefault"]["role"].startswith("production delta only")
    assert sc["runs"]["replay-decode"]["tables"]["v8_ms vs replay_ms"]["all"]["n"] == 6
    assert "Adefault vs A1710" in sc["deltas"]
    assert "DESCRIPTIVE" in (out / "prod.score.txt").read_text()


def test_replay_vs_serving_with_and_without_the_api_fills_and_truths_labelled(tmp_path, stub):
    """SMOKE-FIXES 6: --replay-check carries each sampled cell's fills (the replay's functional API path); the score
    reports kernel-sum vs replay with and without them, the fills' share per tile; kernel-sum is the PRIMARY truth,
    span SECONDARY with span / kernel-sum per cudagraph mode."""
    a, _calls = _extracted(tmp_path, "A1710", ((5, 8), (40, 40), (100, 104)))
    rep = _replay(tmp_path, a)
    rows = PL.read_jsonl(a / "calls.jsonl")[:6]
    rc = tmp_path / "replay_check.json"
    rc.write_text(json.dumps({"cells": [{"label": f"g{r['g']}", "replay": {"fills_n": 9, "fills_ns": 9000}}
                                        for r in rows]}))
    out = tmp_path / "score"
    assert SP.main(["--model", MODEL.key, "--run", f"A1710={a}", "--replay", f"decode={rep}", "--replay-check",
                    f"decode={rc}", "--replay-check", f"prefill={tmp_path / 'absent.json'}", "--pricer", stub,
                    "--workers", "1", "--sag", _par(tmp_path), "--out", str(out)]) == 0
    sc = json.loads((out / "prod.score.json").read_text())
    tabs = sc["runs"]["replay-decode"]["tables"]
    w = tabs["kernel_sum_ms (in-serving) vs replay_ms"]["all"]
    wo = tabs["kernel_sum_ms (in-serving) vs replay_ms minus the API fills"]["all"]
    assert w["n"] == wo["n"] == 6 and wo["bias"] > w["bias"]           # the replay minus 9 us is closer to serving
    assert "v8_ms vs replay_ms minus the API fills" in tabs
    fills = sc["runs"]["replay-decode"]["fills"]
    assert all(abs(v["fills_us_median"] - 9.0) < 1e-9 for v in fills.values())
    assert any("PRIMARY" in n and "SECONDARY" in n for n in sc["notes"])
    assert sc["runs"]["A1710"]["span_over_kernel_sum_by_cg_mode"]
    txt = (out / "prod.score.txt").read_text()
    assert "[primary]" in txt and "[secondary]" in txt and "API fills (replay)" in txt and "span / kernel-sum" in txt
    assert SP.load_replay_check(tmp_path / "absent.json") == {}


def test_refused_calls_are_counted_with_their_share_of_moe_time(tmp_path, stub):
    a, _ = _extracted(tmp_path, "A1410", ((8200, 8200), (600, 600)))
    out = tmp_path / "score"
    assert SP.main(["--model", MODEL.key, "--run", f"A1410={a}", "--pricer", stub, "--workers", "2",
                    "--sag", _par(tmp_path), "--out", str(out)]) == 0
    r = json.loads((out / "prod.score.json").read_text())["runs"]["A1410"]
    assert r["v8_refusals"] == {"HOPPER_G1_N refused": 16}
    assert r["v8_refused_share_of_moe_time"] == pytest.approx(0.5)
    st = r["tables"]["v8_ms vs kernel_sum_ms"]["stratum"]
    assert PL.REFUSED_STRATUM not in st                               # refused: no prediction, not scored
    # the prefill rows carry v8 at the board's par and, beside it, at the reference median (DESCRIPTIVE)
    t, u = r["tables"]["v8_ms vs kernel_sum_ms"]["all"], r["tables"]["v8_refpar_ms vs kernel_sum_ms"]["all"]
    assert t["n"] == u["n"] > 0 and t["bias"] != u["bias"]


def test_a_pricer_that_raises_fails_the_score(tmp_path, monkeypatch):
    d = tmp_path / "m"
    d.mkdir()
    (d / "boom_v8.py").write_text("def price(c):\n    raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(d))
    a, _ = _extracted(tmp_path, "A1710", ((5, 8),))
    assert SP.main(["--model", MODEL.key, "--run", f"A1710={a}", "--pricer", "boom_v8", "--workers", "1",
                    "--sag", _par(tmp_path), "--out", str(tmp_path / "s")]) == 3


def test_a_worker_that_raises_a_base_exception_is_recorded_not_lost(tmp_path, monkeypatch, capsys):
    """MINOR: v8's PoisonedTread is a BaseException; a pricing worker catches it, records it as the job's error,
    and the score exits 3 naming it (in a process pool too)."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "poison_v8.py").write_text("class PoisonedTread(BaseException):\n    pass\n\n"
                                    "def price(c):\n    raise PoisonedTread('int() on a Tread')\n")
    monkeypatch.syspath_prepend(str(d))
    monkeypatch.setenv("PYTHONPATH", str(d) + ":" + __import__("os").environ.get("PYTHONPATH", ""))
    got = SP._price_one(("olmoe-1b-7b", 8, [8] * 8 + [0] * 56, None, None, "poison_v8"))
    assert got["ms"] is None and got["error"] == "PoisonedTread: int() on a Tread"
    a, _ = _extracted(tmp_path, "A1710", ((5, 8),))
    for workers in ("1", "2"):
        rc = SP.main(["--model", MODEL.key, "--run", f"A1710={a}", "--pricer", "poison_v8", "--workers", workers,
                      "--sag", _par(tmp_path), "--out", str(tmp_path / f"s{workers}")])
        assert rc == 3 and "PoisonedTread" in capsys.readouterr().out
        sc = json.loads((tmp_path / f"s{workers}" / "prod.score.json").read_text())
        assert sc["pricing"]["errors"] > 0 and sc["pricing"]["first_errors"][0].startswith("PoisonedTread")


def test_the_scorer_refuses_a_missing_or_failed_board_par(tmp_path, stub):
    a, _ = _extracted(tmp_path, "A1410", ((600, 600),))
    rc = SP.main(["--model", MODEL.key, "--run", f"A1410={a}", "--pricer", stub, "--sag",
                  _par(tmp_path, verdict="FAIL"), "--out", str(tmp_path / "s")])
    assert rc == 3


# ---------------------------------------------------------------- the REAL v8 (separate processes: round 5's chain)

PUB = REPO / "results" / "published"
V7P = PUB / "2026-11-18-nvidia_gh200_480gb-v7p-session"


def _have_chain():
    import round5_chain as R5
    return all((PUB / s / "results").is_dir() for s in R5.RUNS.values())


@pytest.mark.skipif(not _have_chain(), reason="round 5's published sessions are not in this checkout")
def test_contract_the_kit_prices_a_real_decode_and_prefill_call_through_the_real_v8(tmp_path):
    """BLOCKER 1: prodlib.price_v8 against scripts/r3_timing_model_v8.py itself (no stub): one decode call (T 8) and
    one prefill call (T 1024) of OLMoE and of ERNIE, real ids, at a board par: both price."""
    import subprocess
    for model in ("olmoe-1b-7b", "ernie-4.5-21b-a3b"):
        r = subprocess.run([sys.executable, str(REPO / "scripts/prod/prodlib.py"), "price-check", "--model", model,
                            "--sag", _par(tmp_path)], capture_output=True, text=True, cwd=str(REPO), timeout=900)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        got = json.loads(r.stdout.strip().splitlines()[-2])
        assert got["8"]["regime"] == "decode" and got["1024"]["regime"] == "prefill"
        assert got["8"]["ms"] > 0 and got["1024"]["ms"] > got["8"]["ms"]


@pytest.mark.skipif(not (_have_chain() and V7P.is_dir()), reason="the v7 prefill blind's session is not published here")
def test_sag_par_finalises_the_v7_prefill_blinds_own_pages_to_its_registered_par(tmp_path):
    """BLOCKER 2: sag_par.py (v7pcommon.finalise_pars, unchanged) on the v7p rental's f128 pages and b3sag record
    gives that blind's finalised par (prefill_blind.score.json), and prodlib.load_board_par reads it."""
    import subprocess
    out = tmp_path / "par.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts/prod/sag_par.py"), "--results", str(V7P / "results"),
                        "--record", str(V7P / "session/gh200-driver/gate-b3sag-1410.json"), "--driver-exit", "0",
                        "--started", "0", "--out", str(out)],
                       capture_output=True, text=True, cwd=str(REPO), timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    want = json.loads((REPO / "scripts/scoring/v7pblind/prefill_blind.score.json").read_text())["finalised"]["par"]
    assert PL.load_board_par(out) == tuple(want) == BOARD_PAR
    r2 = subprocess.run([sys.executable, str(REPO / "scripts/prod/sag_par.py"), "--results", str(tmp_path),
                         "--record", str(tmp_path / "none.json"), "--driver-exit", "3", "--started", "0",
                         "--out", str(tmp_path / "f.json")],
                        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r2.returncode == 3 and "no sag gate record" in r2.stdout
    with pytest.raises(PL.KitError):
        PL.load_board_par(tmp_path / "f.json")


def test_sag_par_needs_the_drivers_exit_0_and_a_record_newer_than_the_step(tmp_path):
    """review 2, MINOR 4: a non-zero driver exit, or a gate record older than the sag1410 step, is a FAIL par."""
    import time

    import sag_par
    rec = tmp_path / "gate-b3sag-1410.json"
    rec.write_text(json.dumps({"verdict": "PASS"}))
    bad = sag_par.finalise(tmp_path, rec, 4, started=0)
    assert bad["verdict"] == "FAIL" and "exited 4" in bad["why"]
    stale = sag_par.finalise(tmp_path, rec, 0, started=time.time() + 3600)
    assert stale["verdict"] == "FAIL" and "stale" in stale["why"]
    # a fresh record with exit 0 reaches the finaliser (here it refuses the empty page tree, but not for staleness)
    fresh = sag_par.finalise(tmp_path, rec, 0, started=time.time() - 3600)
    assert "stale" not in fresh.get("why", "") and "exited" not in fresh.get("why", "")
