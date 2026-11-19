"""scripts/prod/replay.py off the GPU: the registered sample rule, the sample files (V-BIN on load, sha-checked),
and the adapter: r3_decode's own run_page / readback checks / report driven with the calls' REAL ids through a
fake backend (tests/decode_fakes.py), plus the dry run's plan as locked_r3 parses it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import decode_fakes as DF  # noqa: E402
import locked_r3 as LR  # noqa: E402
import prod_fakes as F  # noqa: E402
import prodlib as PL  # noqa: E402
import r3_decode as RD  # noqa: E402
import replay as RP  # noqa: E402

MODEL = PL.MODELS["olmoe-1b-7b"]


def _rows(n_steps=40):
    rows = []
    g = 0
    for s in range(n_steps):
        for layer in range(4):
            T = [1, 40, 70, 120, 300, 900, 9000][s % 7]
            rows.append({"g": g, "step": s, "layer": layer, "T_pad": T, "tile": PL.tile_bin(T), "status": "ok",
                         "warmup": s < 2})
            g += 1
    rows.append({"g": g, "step": 99, "layer": 0, "T_pad": 1, "tile": PL.tile_bin(1), "status": "no moe_sum after w2"})
    return rows


def test_the_sample_rule_is_stratified_deterministic_and_skips_bad_calls():
    rows = _rows()
    a = RP.select_sample(rows, 70)
    assert a == RP.select_sample(rows, 70)
    assert all(r["status"] == "ok" and not r["warmup"] for r in a)
    per = {}
    for r in a:
        per[r["tile"]] = per.get(r["tile"], 0) + 1
    assert set(per) == {lab for lab, _lo, _hi in PL.TILE_BINS}
    assert sorted(per.values()) == [10] * 7
    assert len({r["g"] for r in a}) == len(a)                 # no call twice
    assert RP.select_sample(rows, 70, seed=1) != a
    small = RP.select_sample(rows, 1500)                       # a tile with fewer calls gives them all
    assert len(small) == sum(1 for r in rows if r["status"] == "ok" and not r["warmup"])


def _sample(tmp_path, steps=((5, 8), (100, 112), (300, 304))):
    run, calls = F.make_run(tmp_path / "run", steps=steps)
    rows = [{"g": i, "step": c["step"], "layer": c["layer"], "T_pad": c["n"], "T": c["n"], "tile": PL.tile_bin(c["n"]),
             "status": "ok", "warmup": False} for i, c in enumerate(calls)]
    picked = RP.select_sample(rows, 9)
    p = RP.write_sample(run, MODEL, picked, tmp_path / "sample")
    return p, calls, picked


def test_sample_files_round_trip_with_the_real_ids(tmp_path):
    p, calls, picked = _sample(tmp_path)
    page, ids = RP.load_sample(p, MODEL, "decode")
    assert {c.T for c in page.cells} <= {8, 112} and all(c.arms == (RD.NATIVE,) and not c.shuffle for c in page.cells)
    for c in page.cells:
        g = int(c.label[1:])
        assert np.array_equal(ids[c.label], calls[g]["ids"])
        assert list(c.counts) == PL.counts_from_ids(calls[g]["ids"], 64)
    page_p, _ = RP.load_sample(p, MODEL, "prefill")
    assert {c.T for c in page_p.cells} == {304}


def test_a_tampered_sample_is_refused(tmp_path):
    p, _c, _p = _sample(tmp_path)
    doc = json.loads(p.read_text())
    doc["cells"][0]["counts"][0] += 1
    doc["cells"][0]["counts"][1] -= 1
    p.write_text(json.dumps(doc))
    with pytest.raises(RD.DecodeRefused, match="V-BIN"):
        RP.load_sample(p, MODEL, None)
    ids = p.parent / "sample_ids.npz"
    ids.write_bytes(ids.read_bytes() + b"x")
    with pytest.raises(RD.DecodeRefused, match="sha256"):
        RP.load_sample(p, MODEL, None)


class FakeProd(RP.RealIdsMixin, DF.FakeBackend):
    pass


def test_the_adapter_runs_r3_decodes_page_loop_on_the_real_ids(tmp_path):
    p, calls, _picked = _sample(tmp_path)
    page, real = RP.load_sample(p, MODEL, "decode")
    cfg = MODEL.cfg
    be = FakeProd(cfg, page, truth=lambda c, a: 0.02 + 1e-4 * c.T)
    be.real_ids = real
    seen = []
    orig = DF.FakeBackend.readback

    def spy(self, cell, arm, ids, calls_):
        seen.append((cell.label, ids.reshape(-1).tolist()))
        return orig(self, cell, arm, ids, calls_)
    be.readback = spy.__get__(be)
    out = tmp_path / "pwr" / "run"
    out.mkdir(parents=True)
    store = RD.Store(out / "cells.csv")
    res = RD.run_page(cfg, page, be, inst="GRR", repeats=3, store=store, lock_mhz=1710, min_clean=0, say=lambda m: None)
    store.close()
    for lab, flat in seen:                       # the readback saw the CALL's ids, not realize_counts'
        assert flat == real[lab].reshape(-1).tolist()
    assert not res["refused"]
    args = argparse.Namespace(model=MODEL.key, dtype="bf16", inst="GRR", duty=0.25, repeats=3, group_m=1, seed=0,
                              session_tag="T-lock1710", min_clean_repeats=0, rot="auto")
    rc = RD.write_report(args, cfg, page, res, out_dir=out, run_id="r", card=DF.CARD, R=1, calls=1, copies=1,
                         slip_lock=1710, header=["x"], seconds=1.0)
    rep = json.loads((out / "report.json").read_text())
    assert rc == 0 and len(rep["treads_table"]) == len(page.cells)
    assert rep["histogram_gates"]["V_ALIGN"]["verdict"] == "PASS"


def test_dry_run_plan_is_what_locked_r3_reads(tmp_path, capsys, monkeypatch):
    p, _c, _p = _sample(tmp_path)
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path / "res"))
    rc = RP.main(["--model", MODEL.key, "--histogram", str(p), "--branch", "decode", "--session-tag", "T-lock1710",
                  "--slip-policy", "cell", "--lock-mhz", "1710", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2 and "--dry-run was given" in out
    assert LR.PLAN_SESSION.search(out).group(1) == "T-lock1710"
    assert "prodreplay" in LR.PLAN_WRITES.search(out).group(1) and "-decode-" in LR.PLAN_WRITES.search(out).group(1)
    assert "real ids" in out
    # a branch at the other branch's clock is refused (CRITIQUE B1)
    assert RP.main(["--model", MODEL.key, "--histogram", str(p), "--branch", "prefill", "--slip-policy", "cell",
                    "--lock-mhz", "1710", "--dry-run"]) == 2
    assert "branch clock 1410" in capsys.readouterr().out


# ---------------------------------------------------------------- review 1, MAJOR 6: budget, big cell, wall, memory

def _wall(tiles=None, repeats=3):
    tiles = tiles or {lab: {"median_wall_s": 10.0 * (i + 1), "cells": 3} for i, (lab, _lo, _hi) in enumerate(PL.TILE_BINS)}
    return {"schema": RP.WALL_SCHEMA, "repeats": repeats, "tiles": tiles}


def test_the_budget_rule_scales_every_quota_to_fit_and_keeps_one_per_tile():
    rows = [dict(r, branch=PL.branch(r["T_pad"])) for r in _rows()]
    q = RP.tile_quota([r for r in rows if r["branch"] == "prefill"], 643)
    assert sum(q.values()) == 643 and set(q) == {"64x128x64w8s3", "128x128x64w8s3", "128x128x64w8s3 g16"}
    b = RP.budget_quota(q, _wall(), "prefill", repeats=9, budget_s=3000)
    per = b["per_cell_s"]
    assert per["64x128x64w8s3"] == pytest.approx(1.5 * 50.0 * 10 / 4)       # safety x wall x (9 + 1) / (3 + 1)
    assert b["scale"] < 1 and b["cost_after_s"] <= 3000 and all(v >= 1 for v in b["quota"].values())
    assert RP.budget_quota(q, _wall(), "prefill", repeats=9, budget_s=1e9)["quota"] == q      # fits: unchanged
    # a tile the smoke did not time takes its branch's dearest timed tile
    w = _wall({"64x128x64w8s3": {"median_wall_s": 20.0, "cells": 1}, "128x128x64w8s3": {"median_wall_s": 40.0, "cells": 1}})
    assert RP.budget_quota(q, w, "prefill", repeats=3)["per_cell_s"]["128x128x64w8s3 g16"] == pytest.approx(60.0)
    with pytest.raises(PL.KitError, match="no decode tile"):
        RP.budget_quota({"16x64x128w4s4": 3}, w, "decode", repeats=3)


def test_sample_cli_budgets_and_adds_a_big_cell(tmp_path):
    run, calls = F.make_run(tmp_path / "run", steps=((5, 8), (300, 304), (600, 600)))
    rows = [{"g": i, "step": c["step"], "layer": c["layer"], "T_pad": c["n"], "T": c["n"], "tile": PL.tile_bin(c["n"]),
             "branch": PL.branch(c["n"]), "status": "ok", "warmup": False} for i, c in enumerate(calls)]
    with open(tmp_path / "calls.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    wall = tmp_path / "wall.json"
    wall.write_text(json.dumps(_wall()))
    out = tmp_path / "s"
    rc = RP.main(["sample", "--run", str(run), "--calls", str(tmp_path / "calls.jsonl"), "--model", MODEL.key,
                  "--branch", "prefill", "--n", "20", "--wall", str(wall), "--repeats", "9", "--budget-s", "1000",
                  "--min-big-T", "8256", "--out", str(out)])
    assert rc == 0
    doc = json.loads((out / "sample.json").read_text())
    assert doc["budget"]["scale"] < 1 and doc["budget"]["cost_after_s"] <= 1000
    big = [c for c in doc["cells"] if c["T"] >= 8256]
    assert len(big) == 1 and big[0]["synthetic"] and big[0]["label"] == "s16384"
    page, ids = RP.load_sample(out / "sample.json", MODEL, "prefill")       # V-BIN holds on the synthetic cell
    assert ids["s16384"].shape == (16384, MODEL.cfg.top_k)


def test_the_adapter_clocks_each_cells_wall_and_writes_it(tmp_path):
    p, calls, _picked = _sample(tmp_path)
    page, real = RP.load_sample(p, MODEL, "decode")
    be = FakeProd(MODEL.cfg, page, truth=lambda c, a: 0.02)
    be.real_ids = real
    store = RD.Store(tmp_path / "cells.csv")
    RD.run_page(MODEL.cfg, page, be, inst="GRR", repeats=2, store=store, lock_mhz=1710, min_clean=0, say=lambda m: None)
    store.close()
    w = be.wall()
    assert set(w) == {c.label for c in page.cells} and all(v > 0 for v in w.values())
    f = RP.write_wall(tmp_path, page, w, 2, p)
    doc = json.loads(f.read_text())
    assert doc["repeats"] == 2 and len(doc["cells"]) == len(page.cells)
    rc = RP.wall_merge(["--from", str(tmp_path), "--out", str(tmp_path / "replay_wall.json")])
    assert rc == 0 and json.loads((tmp_path / "replay_wall.json").read_text())["schema"] == RP.WALL_SCHEMA


def test_the_grr_graphs_outputs_are_bounded():
    cfg = MODEL.cfg
    page = type("P", (), {"Ts": (16384,)})()
    assert RP.grr_memory_refusal(cfg, page, "GRR", 8, 10 ** 9, 90 * 10 ** 9) == ""
    assert "GiB" in RP.grr_memory_refusal(cfg, page, "GRR", 200, 10 ** 9, 900 * 10 ** 9)
    assert "free" in RP.grr_memory_refusal(cfg, page, "GRR", 8, 10 ** 9, 10 ** 9)
    assert RP.grr_memory_refusal(cfg, page, "E240", 200, 10 ** 9, 10 ** 9) == ""
