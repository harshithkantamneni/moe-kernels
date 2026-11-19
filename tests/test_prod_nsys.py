"""scripts/prod/nsys_extract.py off the GPU: a synthetic serving run (the real Capture host logic) and the GPU
timeline it would leave in an nsys sqlite export (CUPTI/NVTX tables as nsys names them) or a torch-profiler
trace; the call segmentation (align..moe_sum, router excluded), kernel-sum and span, the joins, and every gate
failing loudly when its invariant is broken."""
from __future__ import annotations

import json
import sys
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

MODEL = PL.MODELS["olmoe-1b-7b"]
CFG = MODEL.cfg
STEPS = ((1, 1), (2, 2), (5, 8), (200, 208), (600, 600))


def _run(tmp_path, steps=STEPS, mutate=None, off=(), **kw):
    run, calls = F.make_run(tmp_path / "run", steps=steps, **kw)
    ev, marks, expect = F.timeline(CFG, calls, mutate=mutate)
    F.write_clocks(run, len(calls) // 4 + 2, off=off)
    db = F.write_sqlite(tmp_path / "serve.sqlite", ev, marks)
    return run, calls, db, expect, ev


def test_classify_names_as_vllm_0271_launches_them():
    assert NX.classify(F.NAMES["gemm"]) == "gemm"
    assert NX.classify(F.NAMES["align"]) == "align" and NX.classify(F.NAMES["small"]) == "align"
    assert NX.classify(F.NAMES["sort"]) == "align"
    assert NX.classify(F.NAMES["act"]) == "act" and NX.classify(F.NAMES["reduce"]) == "reduce"
    assert NX.classify(F.NAMES["fill"]) == "fill" and NX.classify(F.NAMES["marker"]) == "marker"
    for other in ("router", "convert", "attn", "shared"):
        assert NX.classify(F.NAMES[other]) == "other", other
    # a fused_moe_kernel_gptq_awq or similar is not the Triton fused_moe_kernel
    assert NX.classify("fused_moe_kernel_gptq_awq") == "other"
    assert NX.classify("fused_moe_kernel_0d1d2d3d4d5d") == "gemm"


def test_a_clean_run_passes_every_gate_and_times_align_to_moe_sum(tmp_path):
    run, calls, db, expect, _ = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert all(g["verdict"] == "PASS" for g in res["gates"].values()), res["gates"]
    rows = res["rows"]
    assert len(rows) == len(calls) == len(STEPS) * MODEL.moe_layers
    for r, c, x in zip(rows, calls, expect, strict=True):
        assert r["T_pad"] == c["n"] and r["layer"] == c["layer"] and r["status"] == "ok"
        assert r["kernel_sum_ns"] == x["kernel_sum_ns"]          # router, marker, convert, attention excluded
        assert r["span_ns"] == x["span_ns"] and r["align_ns"] == x["align_ns"]
        assert r["intruders"] == 0
        assert r["tile"] == PL.tile_bin(c["n"]) and r["branch"] == PL.branch(c["n"])
        assert r["sm_mhz"] == 1710.0
    assert np.array_equal(res["counts"][0], PL.counts_from_ids(calls[0]["ids"], CFG.num_experts))
    # the step join: unpadded T from the step log
    assert [r["T"] for r in rows[::MODEL.moe_layers]] == [s[0] for s in STEPS]
    assert {r["step"] for r in rows} == set(range(len(STEPS)))


def test_naive_path_takes_the_fill_kernel_as_its_align(tmp_path):
    run, calls, db, expect, _ = _run(tmp_path, steps=((1, 1),))
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-CLASS"]["verdict"] == "PASS"
    assert res["rows"][0]["align_ns"] == F.DUR["fill"]


def test_an_intruder_is_inside_span_and_outside_kernel_sum(tmp_path):
    run, calls, db, expect, _ = _run(tmp_path, mutate=lambda i: {"intruder": True} if i == 3 else None)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    r = res["rows"][3]
    assert r["intruders"] == 1 and r["kernel_sum_ns"] == expect[3]["kernel_sum_ns"]
    assert r["span_ns"] == expect[3]["span_ns"]  # the fake's span already includes the shared GEMM's slot
    assert all(g["verdict"] == "PASS" for g in res["gates"].values())


def test_a_missing_marker_fails_G_REC(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, mutate=lambda i: {"drop_marker": True} if i == 5 else None)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-REC"]["verdict"] == "FAIL"
    assert "ring records" in res["gates"]["G-REC"]["why"]


def test_a_wrong_grid_fails_G_GRID(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, mutate=lambda i: {"bad_grid": True} if i == 40 else None)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-GRID"]["verdict"] == "FAIL" and res["gates"]["G-GRID"]["problems"] == 1


def test_a_chunked_call_fails_G_GEMM_and_G_CLASS(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, mutate=lambda i: {"extra_gemm": True} if i == 7 else None)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-GEMM"]["verdict"] == "FAIL"
    assert res["gates"]["G-CLASS"]["verdict"] == "FAIL" and "chunked" in res["gates"]["G-CLASS"]["why"]


def test_a_step_log_that_disagrees_with_the_ring_fails_G_STEP(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path)
    steps = PL.read_jsonl(run / "steps.jsonl")
    steps[2]["T_pad"] = 16
    (run / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-STEP"]["verdict"] == "FAIL" and "T_pad 16" in res["gates"]["G-STEP"]["why"]


def test_ids_with_a_repeated_expert_fail_G_IDS(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, steps=((5, 8),))
    shard = sorted((run / "ring").glob("shard_*.npz"))[0]
    with np.load(shard) as z:
        ids, recs, meta = z["ids"].copy(), z["recs"], z["meta"]
    ids[1] = ids[0]
    np.savez(shard, ids=ids, recs=recs, meta=meta)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-IDS"]["verdict"] == "FAIL"


def test_no_reset_mark_is_a_kit_error(tmp_path):
    run, calls = F.make_run(tmp_path / "run", steps=((5, 8),))
    ev, marks, _ = F.timeline(CFG, calls)
    db = F.write_sqlite(tmp_path / "s.sqlite", ev, [m for m in marks if m[1] != F.MC_RESET])
    with pytest.raises(PL.KitError):
        NX.extract(run, NX.read_sqlite(db), MODEL)


def test_warmup_markers_before_the_reset_are_excluded(tmp_path):
    run, calls, db, _x, ev = _run(tmp_path, steps=((5, 8),))
    assert sum(1 for e in ev if e["name"] == F.NAMES["marker"]) == len(calls) + 3
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-REC"]["markers"] == len(calls)


def test_off_lock_clock_samples_are_read_per_call(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, steps=((5, 8),), off=(1,))
    rows = NX.extract(run, NX.read_sqlite(db), MODEL)["rows"]
    assert [r["sm_mhz"] for r in rows[:8]] == [1710.0] * 4 + [1650.0] * 4


def test_torch_trace_window_extracts_with_the_window_shard(tmp_path):
    run, calls = F.make_run(tmp_path / "run", steps=((5, 8), (40, 40), (600, 600)), tag_last="profiler-window")
    window = [c for c in calls if c["step"] == 2]
    ev, _marks, expect = F.timeline(CFG, window, warmup_markers=0)
    F.write_torch_trace(tmp_path / "tt", ev, gz=True)
    res = NX.extract(run, NX.read_torch_trace(tmp_path / "tt"), MODEL, shard_tag="profiler-window")
    assert all(g["verdict"] == "PASS" for g in res["gates"].values()), res["gates"]
    assert [r["kernel_sum_ns"] for r in res["rows"]] == [x["kernel_sum_ns"] for x in expect]


def test_cli_writes_outputs_and_exits_3_on_a_failed_gate(tmp_path, capsys):
    run, calls, db, _x, _e = _run(tmp_path, mutate=lambda i: {"bad_grid": True} if i == 0 else None)
    rc = NX.main(["--run", str(run), "--sqlite", str(db), "--model", "olmoe-1b-7b"])
    out = capsys.readouterr().out
    assert rc == 3 and "FAIL  G-GRID" in out
    assert (run / "calls.jsonl").exists() and (run / "counts.npy").exists() and (run / "gates.json").exists()
    run2, _c, db2, _x2, _e2 = _run(tmp_path / "ok")
    assert NX.main(["--run", str(run2), "--sqlite", str(db2), "--model", "olmoe-1b-7b"]) == 0
