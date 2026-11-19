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


# ---------------------------------------------------------------- review 1: pairing, shared experts, fills, smoke checks

def test_records_shifted_by_one_call_fail_G_PAIR_though_the_counts_agree(tmp_path):
    """MAJOR 8: the marker's gridY carries its layer; a trace whose calls are one call off the ring's records (the
    first call's kernels missing, one extra call at the end: equal counts) passes G-REC and fails G-PAIR."""
    run, calls = F.make_run(tmp_path / "run", steps=((5, 8), (200, 208)))
    ev, marks, _x = F.timeline(CFG, calls[1:] + [calls[0]])
    F.write_clocks(run, len(calls) // 4 + 2)
    db = F.write_sqlite(tmp_path / "s.sqlite", ev, marks)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert res["gates"]["G-REC"]["verdict"] == "PASS"
    g = res["gates"]["G-PAIR"]
    assert g["verdict"] == "FAIL" and g["problems"] == len(calls) and "gridY" in g["why"]
    # the aligned trace passes it, call by call
    run2, calls2, db2, _x2, _e2 = _run(tmp_path / "ok")
    ok = NX.extract(run2, NX.read_sqlite(db2), MODEL)["gates"]["G-PAIR"]
    assert ok["verdict"] == "PASS" and ok["paired"] == len(calls2)


def test_the_marker_grid_is_layer_plus_one_and_warm_launches_use_one():
    import moe_capture as MC
    assert MC.marker_grid(1) == (1, 2) and MC.marker_grid(27) == (1, 28) and MC.marker_grid(-1) == (1, 1)


def test_a_shared_expert_silu_on_another_stream_does_not_touch_the_routed_call(tmp_path):
    """MAJOR 5: the shared-experts stream on: the shared MLP (GEMM, SiluAndMul, GEMM) runs on a second stream
    overlapping w1..w2 in time. Classification by stream and sequence leaves every routed call exact."""
    run, calls, db, expect, ev = _run(tmp_path, mutate=lambda i: {"shared_stream": True})
    assert sum(1 for e in ev if e["stream"] == F.SHARED_STREAM and e["role"] == "shared_act") == len(calls)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert all(g["verdict"] == "PASS" for g in res["gates"].values()), res["gates"]
    for r, x in zip(res["rows"], expect, strict=True):
        assert r["kernel_sum_ns"] == x["kernel_sum_ns"] and r["span_ns"] == x["span_ns"] and r["intruders"] == 0
        assert r["stream"] == F.STREAM


def test_a_shared_expert_silu_on_the_same_stream_runs_outside_the_routed_call(tmp_path):
    """Stream off: the shared MLP is its own op on the marker's stream, before the router; never between the GEMMs."""
    run, calls, db, expect, _e = _run(tmp_path, mutate=lambda i: {"shared_same_stream": True})
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert all(g["verdict"] == "PASS" for g in res["gates"].values()), res["gates"]
    assert [r["kernel_sum_ns"] for r in res["rows"]] == [x["kernel_sum_ns"] for x in expect]
    assert all(r["intruders"] == 0 for r in res["rows"])


def test_two_fill_kernels_on_the_naive_path_are_ambiguous_not_a_silent_pick(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path, steps=((1, 1),), mutate=lambda i: {"extra_fill": i == 2})
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    g = res["gates"]["G-CLASS"]
    assert g["verdict"] == "FAIL" and g["problems"] == 1 and "fill kernels before w1" in g["why"]


def test_check_sequences_passes_a_clean_run_per_path_and_fails_a_bad_naive_sequence(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    cs = NX.check_sequences(res["rows"])
    assert cs["verdict"] == "PASS", cs
    assert set(cs["paths"]) == {r["path"] for r in res["rows"]} and "naive" in cs["paths"]
    naive = cs["paths"]["naive"][0]["sig"]
    assert naive.startswith("fill:") and naive.count("gemm:fused_moe_kernel") == 2
    bad = [dict(r) for r in res["rows"]]
    i = next(i for i, r in enumerate(bad) if r["path"] == "naive")
    bad[i]["sig"] = "align:moe_align_block_size_kernel|" + bad[i]["sig"].split("|", 1)[1]
    assert NX.check_sequences(bad)["verdict"] == "FAIL"


def test_replay_check_matches_the_replays_kernels_to_servings_per_tile(tmp_path):
    """MINOR: the replay (no markers) must show each sampled cell's launch grids with a kernel sequence serving
    shows on the same tile; a wrong grid fails."""
    run, calls, db, _x, _e = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    NX.write(res, run, "")
    rows = res["rows"]
    pick = [rows[0], rows[MODEL.moe_layers * 3], rows[-1]]
    cells = [{"label": f"g{r['g']}", "T": r["T_pad"], "tile": r["tile"],
              "counts": [int(x) for x in res["counts"][rows.index(r)]]} for r in pick]
    sample = {"cells": cells}
    rcalls = [dict(calls[rows.index(r)]) for r in pick]
    ev, _m, _x2 = F.timeline(CFG, rcalls, warmup_markers=0, mutate=lambda i: {"drop_marker": True})
    db2 = F.write_sqlite(tmp_path / "replay.sqlite", ev, [])
    rc = NX.replay_check(rows, sample, NX.segment_unmarked(NX.read_sqlite(db2)["events"]), MODEL)
    assert rc["verdict"] == "PASS", rc
    assert all(v["seen"] == v["cells"] == v["sig_ok"] for v in rc["per_tile"].values())
    ev3, _m3, _x3 = F.timeline(CFG, rcalls, warmup_markers=0,
                               mutate=lambda i: {"drop_marker": True, "bad_grid": i == 1})
    db3 = F.write_sqlite(tmp_path / "replay3.sqlite", ev3, [])
    bad = NX.replay_check(rows, sample, NX.segment_unmarked(NX.read_sqlite(db3)["events"]), MODEL)
    assert bad["verdict"] == "FAIL" and "not in the replay trace" in bad["why"]


def _replay_fixture(tmp_path, api_fills=9):
    run, calls, db, _x, _e = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    rows = res["rows"]
    pick = [rows[0], rows[MODEL.moe_layers * 3], rows[MODEL.moe_layers * 3 + 1], rows[-1]]
    cells = [{"label": f"g{r['g']}", "g": r["g"], "T": r["T_pad"], "tile": r["tile"], "branch": PL.branch(r["T_pad"]),
              "counts": [int(x) for x in res["counts"][rows.index(r)]]} for r in pick]
    rcalls = []
    for i, r in enumerate(pick):
        c = dict(calls[rows.index(r)])
        c["step"] = i                       # one cell per "step": the cell marks (gridY = cell index + 1)
        rcalls.append(c)
    return rows, {"cells": cells}, rcalls


def test_replay_check_compares_the_core_sequence_and_counts_the_api_fills(tmp_path):
    """SMOKE-FIXES 6: the replay's functional API launches ~9 fill kernels per call before align; serving's modular
    path does not. The check compares the CORE sequence (align..moe_sum) and grids with the fills stripped, and records
    per call the fills' count and GPU time for the replay and for the serving call; on the naive path the last fill
    is the align (vLLM's torch.full), the others are API fills."""
    rows, sample, rcalls = _replay_fixture(tmp_path)
    ev, _m, x = F.timeline(CFG, rcalls, warmup_markers=0, mutate=lambda i: {"drop_marker": True, "api_fills": 9})
    rcs = NX.segment_unmarked(NX.read_sqlite(F.write_sqlite(tmp_path / "r.sqlite", ev, []))["events"])
    assert [c["fills_n"] for c in rcs] == [9] * 4 and all(c["fills_ns"] == 9 * F.DUR["fill"] for c in rcs)
    assert [c["kernel_sum_ns"] for c in rcs] == [e["kernel_sum_ns"] for e in x]          # the core only
    assert rcs[0]["sig"].startswith("fill:") and rcs[0]["sig_full"].count("fill:") == 10    # naive: core = last fill
    assert all(c["sig"].count("fill:") <= 1 for c in rcs)
    rc = NX.replay_check(rows, sample, rcs, MODEL)
    assert rc["verdict"] == "PASS", rc["why"]
    assert all(c["replay"]["fills_n"] == 9 and c["serving"]["fills_n"] == 0 for c in rc["cells"])
    assert all(v["serving_fills_n_median"] == 0 and v["replay_fills_n_median"] == 9 for v in rc["per_tile"].values())
    assert rc["compared"].startswith("core sequence")
    # the old comparison (fills inside the sequence) is what failed the smoke: the full sequences differ from serving's
    serving_sigs = {r["sig"] for r in rows if r.get("sig")}
    assert any(c["sig_full"] not in serving_sigs for c in rcs)


def test_replay_check_with_cell_marks_gives_each_cell_its_own_calls(tmp_path):
    """replay.py --cell-marks: the step marker (gridY = cell index + 1) before each cell's builds; two cells on the
    same grid are kept apart; a cell whose own calls miss its grid fails even if another cell has that grid."""
    rows, sample, rcalls = _replay_fixture(tmp_path)
    ev, _m, x = F.timeline(CFG, rcalls, warmup_markers=0, mutate=lambda i: {"drop_marker": True, "api_fills": 9},
                           step_ids=list(range(len(rcalls))))
    events = NX.read_sqlite(F.write_sqlite(tmp_path / "m.sqlite", ev, []))["events"]
    marked = NX.segment_marked(events)
    assert sorted(marked) == [0, 1, 2, 3] and all(len(v) == 1 for v in marked.values())
    rc = NX.replay_check(rows, sample, NX.segment_unmarked(events), MODEL, marked=marked)
    assert rc["verdict"] == "PASS" and rc["matched_by"] == "cell marks"
    cells = NX.replay_cells(sample, marked, MODEL)
    assert cells["verdict"] == "PASS" and [c["n"] for c in cells["cells"]] == [1, 1, 1, 1]
    assert [round(c["kernel_sum_ms"]["median"] * 1e6) for c in cells["cells"]] == [e["kernel_sum_ns"] for e in x]
    assert all(c["fills_n"] == 9 for c in cells["cells"])
    swapped = {0: marked[1], 1: marked[0], 2: marked[2], 3: marked[3]}
    if NX.PL.expected_grids(CFG, sample["cells"][0]["T"], sample["cells"][0]["counts"]) != \
            NX.PL.expected_grids(CFG, sample["cells"][1]["T"], sample["cells"][1]["counts"]):
        assert NX.replay_check(rows, sample, NX.segment_unmarked(events), MODEL, marked=swapped)["verdict"] == "FAIL"
    assert NX.segment_marked([e for e in events if NX.classify(e[3]) != "stepmark"]) == {}


def test_serving_extraction_records_zero_fills_on_the_modular_path(tmp_path):
    run, calls, db, _x, _e = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    assert all(r["fills_n"] == 0 and r["fills_ns"] == 0 for r in res["rows"] if r["status"] == "ok")


# ---------------------------------------------------------------- review 2: a pairing shifted by a WHOLE step

SAME = ((8, 8), (8, 8), (8, 8), (8, 8))


def test_a_whole_step_shift_passes_g_pair_and_fails_g_stepmark(tmp_path):
    """Every step at the same T_pad: the trace's calls are the ring's, layer for layer (G-REC, G-PAIR and G-STEP
    hold), but the trace places them one step later than the step log and the ring join do."""
    run, calls = F.make_run(tmp_path / "run", steps=SAME)
    ev, marks, _x = F.timeline(CFG, calls, step_ids=[1, 2, 3, 4])
    db = F.write_sqlite(tmp_path / "s.sqlite", ev, marks)
    F.write_clocks(run, len(calls) // 4 + 2)
    g = NX.extract(run, NX.read_sqlite(db), MODEL)["gates"]
    assert g["G-REC"]["verdict"] == g["G-PAIR"]["verdict"] == g["G-STEP"]["verdict"] == "PASS"
    assert g["G-STEPMARK"]["verdict"] == "FAIL" and g["G-STEPMARK"]["misplaced_calls"] == len(calls)
    # the same run with the trace's own steps passes
    ev, marks, _x = F.timeline(CFG, calls)
    g = NX.extract(run, NX.read_sqlite(F.write_sqlite(tmp_path / "ok.sqlite", ev, marks)), MODEL)["gates"]
    assert g["G-STEPMARK"]["verdict"] == "PASS" and g["G-STEPMARK"]["step_markers"] == len(SAME)


def test_a_missing_step_marker_fails_g_stepmark_by_count(tmp_path):
    run, calls = F.make_run(tmp_path / "run", steps=SAME)
    ev, marks, _x = F.timeline(CFG, calls, drop_step_marks=(2,))
    F.write_clocks(run, len(calls) // 4 + 2)
    g = NX.extract(run, NX.read_sqlite(F.write_sqlite(tmp_path / "s.sqlite", ev, marks)), MODEL)["gates"]
    assert g["G-PAIR"]["verdict"] == "PASS"
    assert g["G-STEPMARK"]["verdict"] == "FAIL" and "call count" in g["G-STEPMARK"]["why"]
    assert NX.classify(F.NAMES["stepmark"]) == "stepmark"


def test_the_extract_writes_calls_in_shards_below_the_cap_and_reads_them_back(tmp_path):
    """review 2, MAJOR A: calls.jsonl and counts.npy are sharded at write time below SHARD_MAX_BYTES (each shard a
    plain file the push carries); every reader gets the whole run back; a missing or edited shard is refused."""
    run, calls, db, _x, _e = _run(tmp_path)
    res = NX.extract(run, NX.read_sqlite(db), MODEL)
    out = tmp_path / "out"
    man = PL.write_calls(out, res["rows"], res["counts"], max_bytes=4096)
    assert len(man["calls"]) > 1 and len(man["counts"]) > 1
    assert all((out / e["file"]).stat().st_size <= 4096 for e in man["calls"] + man["counts"])
    rows = PL.read_calls(out)
    assert [r["g"] for r in rows] == [r["g"] for r in res["rows"]]
    assert np.array_equal(PL.load_counts(out), res["counts"])
    # the default cap leaves the ERNIE-sized outputs under the push's 95 MB
    assert PL.SHARD_MAX_BYTES < 95_000_000
    # vm_results_push.sh's compressed form reads back too
    import lzma
    f = out / man["calls"][0]["file"]
    f.with_name(f.name + ".xz").write_bytes(lzma.compress(f.read_bytes()))
    f.unlink()
    assert len(PL.read_calls(out)) == len(rows)
    (out / man["calls"][1]["file"]).unlink()
    with pytest.raises(PL.KitError, match="missing"):
        PL.read_calls(out)
    cf = out / man["counts"][0]["file"]
    raw = bytearray(cf.read_bytes())
    raw[-1] ^= 1
    cf.write_bytes(bytes(raw))
    with pytest.raises(PL.KitError, match="sha256"):
        PL.load_counts(out)
