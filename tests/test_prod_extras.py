"""scripts/prod/extras.sh and its pieces (extras_lib.py, ncu_bytes.py, the Qwen3 model, prompts build-long, the extras'
plan and score): the dry-run order and caps, the budget skipping a block, the commit switch, the noise-floor matching,
the bytes join, Qwen3's config and pricing smoke. CPU only."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"
sys.path.insert(0, str(PROD))

import extras_lib as XL  # noqa: E402
import ncu_bytes as NB  # noqa: E402
import prodlib as PL  # noqa: E402
import prompts as PR  # noqa: E402

QM = "qwen3-30b-a3b"
#: the Hub's config.json of Qwen/Qwen3-30B-A3B at prodlib.HF_REVISIONS (fetched 2026-11-19), the keys G-CFG reads
QWEN3_HF = {"architectures": ["Qwen3MoeForCausalLM"], "decoder_sparse_step": 1, "hidden_size": 2048,
            "intermediate_size": 6144, "mlp_only_layers": [], "model_type": "qwen3_moe", "moe_intermediate_size": 768,
            "norm_topk_prob": True, "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48}


def _cap(step: str) -> int:
    return int(re.search(r" cap=(\d+) ", step).group(1))


def _soft(step: str) -> bool:
    return step.split()[2] == "[soft]"


def _needs(step: str) -> list[str]:
    m = re.search(r" needs=(\S+) ", step)
    return m.group(1).split(",") if m else []


def _dry(tmp_path, *extra, py=None):
    env = dict(os.environ, PY_VLLM=py or sys.executable, HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    p = subprocess.run(["bash", str(PROD / "extras.sh"), "--dry-run", "--results", str(tmp_path / "res"), *extra],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=180)
    return p


# ------------------------------------------------------------------------------------------------------------
# extras.sh: the dry run
# ------------------------------------------------------------------------------------------------------------

def test_extras_shell_syntax():
    assert subprocess.run(["bash", "-n", str(PROD / "extras.sh")]).returncode == 0


def test_extras_dry_run_order_soft_blocks_and_nothing_created(tmp_path):
    p = _dry(tmp_path)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "res").exists() and not (tmp_path / "moe").exists()
    steps = [x for x in p.stdout.splitlines() if x.startswith("STEP ")]
    names = [s.split()[1] for s in steps]
    by = dict(zip(names, steps, strict=True))
    # X0 first, hard: main finished, the commit switch, then the environment at the extras commit
    assert names[:3] == ["extras-main-finished", "extras-commit-switch", "env-preflight"]
    for n in names[:3]:
        assert not _soft(by[n]) and " always " in by[n]
    assert "--expect-commit" in by["env-preflight"] and "--model ernie-4.5-21b-a3b" in by["env-preflight"]
    # Qwen3's pin, rate and disk: X3's own soft check, so a Hub fault drops X3 only
    hf = by["X3-hf-preflight"]
    assert _soft(hf) and f"--model {QM}" in hf and "--skip P1\\,P2" in hf and "X3-hf-preflight" in _needs(
        by["X3-prompts-fetch"])
    assert _soft(by["price-check-qwen3"])
    # the blocks in order, X1 .. X4, then the final push and the held-back check (the last word)
    firsts = [names.index(f"{b}-budget") for b in ("X1", "X2", "X3", "X4")]
    assert firsts == sorted(firsts) and names[-2:] == ["extras-push-final", "extras-held-back-check"]
    # every block step is soft except the unlocks (as main.sh), and reaches its block's budget through --needs
    for n, s in by.items():
        blk = n.split("-")[0]
        if blk not in ("X1", "X2", "X3", "X4"):
            continue
        if "unlock" in n and not _soft(s):
            assert " always " in s, n
            continue
        assert _soft(s), n
        if n.endswith("-budget") or "-push" in n:
            continue
        seen, todo = set(), list(_needs(s))
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo += _needs(by[x])
        assert f"{blk}-budget" in seen, (n, seen)
    # main's outputs are read, never written: every declared output is under extras/ (or main's data dir, fetched
    # only when absent) or the extras' trace dir
    res, tr = f"{tmp_path}/res", f"{tmp_path}/moe/traces/res/extras"
    for s in steps:
        m = re.search(r" out=(\S+) ", s)
        if m:
            for o in re.split(r"[,|]", m.group(1)):
                assert o.startswith((f"{res}/extras/", tr, f"{res}/data/")), (s.split()[1], o)
    # X1 is main's A1710 verbatim: the same prompt file, levels, N, lock, mode and stream
    cap = by["X1-capture"]
    for want in (f"--prompts {res}/prompts.jsonl", "--concurrency 1\\,8\\,24\\,48\\,80\\,112\\,256\\,1024x3",
                 "--requests-per-level 8", "--mode graphs", "--shared-stream off", f"--out {res}/extras/A1710r"):
        assert want in cap, want
    assert "kit_lock 1710" in by["X1-lock"] and "kit_lock 1410" in by["X4-lock"]
    # X3 runs Qwen3 everywhere, its replay on the smaller budget priced by main's own probe walls
    assert f"--model {QM}" in by["X3-A1710-capture"] and "--budget-s 1200" in by["X3-replay-sample-decode"]
    assert f"--wall {res}/replay-probe-decode/replay_wall.json" in by["X3-replay-probe-sample-decode"]
    assert "--run-cap-s 2500" in by["X3-replay-decode"] and _cap(by["X3-replay-decode"]) > 2500
    assert "--lock-mhz 1710" in by["X2-ncu-decode"] and "--lock-mhz 1410" in by["X2-ncu-prefill"]
    # the expected seconds per block and in total are printed
    out = p.stdout
    assert "PLAN extras blocks:" in out and "PLAN EXTRAS EXPECTED TOTAL" in out and "budget 9000 s" in out


def test_extras_no_long_context_drops_x4_and_results_are_required(tmp_path):
    p = _dry(tmp_path, "--no-long-context")
    names = [x.split()[1] for x in p.stdout.splitlines() if x.startswith("STEP ")]
    assert p.returncode == 0 and not any(n.startswith("X4") for n in names) and "X3-push-frontier" in names
    env = dict(os.environ, PY_VLLM=sys.executable, HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    q = subprocess.run(["bash", str(PROD / "extras.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                       cwd=str(REPO), timeout=60)
    assert q.returncode == 2 and "--results" in q.stderr


def test_every_extras_step_has_a_cap_at_least_twice_its_expected_time_and_fits_the_budget(tmp_path):
    import plan_time as PT
    p = _dry(tmp_path)
    caps = {s.split()[1]: _cap(s) for s in p.stdout.splitlines() if s.startswith("STEP ")}
    plan = PT.extras_plan()
    names = [st for _b, st, _s, _x in plan["steps"]]
    assert set(names) == set(caps), set(names) ^ set(caps)
    for _blk, st, sec, _basis in plan["steps"]:
        assert caps[st] >= 2 * sec, (st, caps[st], sec)
    assert all(plan["fits"].values()) and plan["total_s"] < plan["budget_s"]
    tot = [x for x in p.stdout.splitlines() if x.startswith("PLAN EXTRAS EXPECTED TOTAL")]
    assert tot and f"{plan['total_s']:.0f} s" in tot[0]
    # a short budget skips the later blocks in the plan's own reading
    small = PT.extras_plan(budget_s=plan["blocks"]["X0"] + plan["blocks"]["X1"] + 1)
    assert small["fits"]["X1"] and not small["fits"]["X3"]


# ------------------------------------------------------------------------------------------------------------
# the budget
# ------------------------------------------------------------------------------------------------------------

def test_budget_check_passes_inside_and_fails_outside():
    exp = {"X1": 1000.0}
    ok, why = XL.budget_check("X1", t0=0, budget_s=9000, now=7000, expected=exp)
    assert ok, why
    ok, why = XL.budget_check("X1", t0=0, budget_s=9000, now=8500, expected=exp)
    assert not ok and "BUDGET X1" in why and "skipped" in why


def test_a_block_over_the_budget_is_skipped_step_by_step(tmp_path):
    """The block's budget step FAILS (soft) and every step that --needs it is SKIPPED, not run; the next block whose
    expected time still fits runs."""
    import time
    t0 = int(time.time()) - 8990
    py = sys.executable
    body = (f"step X1-budget --soft --cap 60 -- {py} {PROD / 'extras_lib.py'} budget --block X1\n"
            "step X1-inputs --soft --cap 10 --needs X1-budget -- true\n"
            f"step X1-capture --soft --cap 10 --needs X1-inputs -- touch {tmp_path / 'ran'}\n"
            f"step X2-budget --soft --cap 60 -- {py} {PROD / 'extras_lib.py'} budget --block X2 --budget-s 1e9\n"
            "step X2-inputs --soft --cap 10 --needs X2-budget -- true\n")
    script = tmp_path / "seq.sh"
    script.write_text("set -euo pipefail\n. " + str(PROD / "kitlib.sh") + "\nkit_init " + str(tmp_path / "r")
                      + "\n" + body + "kit_summary\n")
    env = dict(os.environ, EXTRAS_T0=str(t0), EXTRAS_BUDGET_S="9000", PYTHONPATH=str(PROD))
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=180)
    assert p.returncode == 0, p.stderr
    summ = (tmp_path / "r" / "SUMMARY.txt").read_text().splitlines()
    assert summ[0].startswith("FAIL (soft, dropped) X1-budget") and "BUDGET X1" in summ[0]
    assert summ[1].startswith("SKIPPED X1-inputs") and summ[2].startswith("SKIPPED X1-capture")
    assert not (tmp_path / "ran").exists()
    assert summ[3].startswith("PASS X2-budget") and summ[4].startswith("PASS X2-inputs")


# ------------------------------------------------------------------------------------------------------------
# main finished, the commit switch
# ------------------------------------------------------------------------------------------------------------

def _git(d: Path, *args) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=d, check=True, capture_output=True, text=True, env=env).stdout.strip()


def _preflight(res: Path, sha: str) -> None:
    res.mkdir(parents=True, exist_ok=True)
    (res / "env-preflight.json").write_text(json.dumps({"schema": "moe-kernels/prod-env-preflight/1", "checks": [
        {"id": "P1", "what": "git", "verdict": "PASS", "detail": f"clean at {sha} (expected, from env.sh)"}]}))


def test_commit_switch_holds_a_clean_tree_at_the_extras_commit_newer_than_mains(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a").write_text("main\n")
    _git(repo, "add", "a")
    _git(repo, "commit", "-q", "-m", "main")
    main_sha = _git(repo, "rev-parse", "HEAD")
    (repo / "b").write_text("extras\n")
    _git(repo, "add", "b")
    _git(repo, "commit", "-q", "-m", "extras")
    ext = _git(repo, "rev-parse", "HEAD")
    res = tmp_path / "res"
    _preflight(res, main_sha)
    ok, why = XL.commit_switch(repo, res, ext)
    assert ok and "newer" in why, why
    ok, why = XL.commit_switch(repo, res, "")
    assert not ok and "PROD_EXTRAS_COMMIT" in why
    ok, why = XL.commit_switch(repo, res, main_sha)                  # HEAD is not the expected commit
    assert not ok and "git checkout" in why
    (repo / "a").write_text("dirty\n")
    ok, why = XL.commit_switch(repo, res, ext)
    assert not ok and "dirty" in why
    _git(repo, "checkout", "-q", "--", "a")
    _git(repo, "checkout", "-q", "--orphan", "other")                 # main's commit not an ancestor of HEAD
    _git(repo, "commit", "-q", "-m", "unrelated")
    other = _git(repo, "rev-parse", "HEAD")
    ok, why = XL.commit_switch(repo, res, other)
    assert not ok and "not an ancestor" in why
    (res / "env-preflight.json").unlink()
    ok, why = XL.commit_switch(repo, res, other)
    assert not ok and "records no main commit" in why


def test_main_finished_needs_a_summary_and_no_running_main(tmp_path):
    ok, why = XL.main_finished(tmp_path, running=[])
    assert not ok and "SUMMARY.txt" in why
    (tmp_path / "SUMMARY.txt").write_text("PASS env-preflight (3s)\nPASS held-back-check (1s)\n")
    assert not XL.main_finished(tmp_path, running=["main.sh (pid 1)"])[0]
    ok, why = XL.main_finished(tmp_path, running=[])
    assert ok and "PASS held-back-check" in why


def test_inputs_check_names_what_main_lacks(tmp_path):
    ok, why = XL.inputs_check("X2", tmp_path)
    assert not ok and "sample-decode/sample.json" in why
    assert XL.inputs_check("X4", tmp_path)[0]


# ------------------------------------------------------------------------------------------------------------
# X1: the noise floor
# ------------------------------------------------------------------------------------------------------------

def _row(g, layer, T, step, ms, tile="16x64x128w4s4", v8=None, roof=None, **kw):
    return {"g": g, "layer": layer, "T_pad": T, "step": step, "counts_digest": f"d{g}", "kernel_sum_ms": ms,
            "tile": tile, "v8_ms": v8, "roof_ms": roof, **kw}


def test_noise_floor_pairs_call_k_with_call_k_when_the_step_sequences_are_identical():
    """review of the extras M8: identical (step, layer, T_pad) sequences pair by position (whatever the counts
    digests say); a filtered call drops its pair, not the alignment."""
    seq = [(1, 8, 0), (2, 8, 0), (1, 1024, 1), (2, 8, 1)]
    a = [_row(i, L, T, st, ms, "128x128x64w8s3" if T > 512 else "16x64x128w4s4")
         for i, ((L, T, st), ms) in enumerate(zip(seq, (0.10, 0.30, 1.0, 0.20), strict=True))]
    b = [_row(100 + i, L, T, st, ms, "128x128x64w8s3" if T > 512 else "16x64x128w4s4")
         for i, ((L, T, st), ms) in enumerate(zip(seq, (0.11, 0.30, 1.1, 0.5), strict=True))]
    b[3]["warmup"] = True
    pairs = XL.pair_by_step(a, b)
    assert [(x["g"], y["g"]) for x, y in pairs] == [(0, 100), (1, 101), (2, 102), (3, 103)]
    nf = XL.noise_floor(a, b, ok=lambda r: not r.get("warmup"))
    assert nf["pairing"] == "step" and nf["matched_calls"] == 3 and nf["matched"]["all"]["n"] == 3
    t = nf["matched"]["128x128x64w8s3"]
    assert t["n"] == 1 and t["abs_rel_median_pct"] == pytest.approx(100 * 0.1 / 1.05)
    assert t["signed_median_pct"] == pytest.approx(10.0)
    small = nf["matched"]["16x64x128w4s4"]
    assert small["n"] == 2 and small["abs_rel_median_pct"] == pytest.approx(100 * 0.01 / 0.105 / 2)
    med = nf["per_tile_medians"]["16x64x128w4s4"]
    assert med["n_a"] == 3 and med["n_b"] == 2 and med["a_ms"] == pytest.approx(0.20)
    assert nf["ratio_medians"] == {}


def test_noise_floor_falls_back_to_per_tile_ratio_medians_when_the_sequences_differ():
    a = [_row(0, 1, 8, 0, 0.20, v8=0.10, roof=0.05), _row(1, 2, 8, 0, 0.40, v8=0.20, roof=0.10)]
    b = [_row(0, 1, 16, 0, 0.30, v8=0.10, roof=0.10), _row(1, 2, 8, 0, 0.40, v8=0.20, roof=0.10),
         _row(2, 3, 8, 1, 0.66, v8=0.30, roof=0.30)]
    assert XL.pair_by_step(a, b) is None
    nf = XL.noise_floor(a, b)
    assert nf["pairing"] == "ratio" and nf["matched_calls"] == 0 and nf["matched"] == {}
    v8 = nf["ratio_medians"]["v8_ms"]["16x64x128w4s4"]
    assert v8["a"] == pytest.approx(2.0) and v8["b"] == pytest.approx(2.2) and v8["delta_pct"] == pytest.approx(10.0)
    rf = nf["ratio_medians"]["roof_ms"]["all"]
    assert rf["n_a"] == 2 and rf["n_b"] == 3 and rf["a"] == pytest.approx(4.0)


def test_x1_identical_holds_levels_n_seed_mode_and_the_prompt_body(tmp_path):
    base = {"levels": [1, 8, 1024, 1024], "requests_per_level": 8, "seed": 0, "mode": "graphs", "shared_stream": "off",
            "prompts": {"body_sha256": "ab" * 32}}
    for d in ("A1710", "A1710r"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "manifest.json").write_text(json.dumps(base))
    assert XL.identical_files(tmp_path / "A1710", tmp_path / "A1710r")[0]
    for k, v in (("levels", [1, 8]), ("requests_per_level", 16), ("seed", 1), ("mode", "eager"),
                 ("prompts", {"body_sha256": "cd" * 32})):
        ok, why = XL.identical_check(base, dict(base, **{k: v}))
        assert not ok and k in why, k
    assert not XL.identical_files(tmp_path / "A1710", tmp_path / "none")[0]
    assert XL.main(["x1-identical", "--base", str(tmp_path / "A1710"), "--repeat", str(tmp_path / "A1710r")]) == 0


# ------------------------------------------------------------------------------------------------------------
# X2: the subsample, the window and the bytes join
# ------------------------------------------------------------------------------------------------------------

def _main_sample(d: Path, branch: str, tiles: dict, model: str = "ernie-4.5-21b-a3b") -> Path:
    """A replay.py-schema sample of `tiles` {tile: [T, ...]} with seeded uniform ids (ERNIE: E 64, k 6)."""
    cfg = PL.MODELS[model].cfg
    d.mkdir(parents=True, exist_ok=True)
    cells, ids, g = [], {}, 0
    for tile, Ts in tiles.items():
        for T in Ts:
            rng = np.random.default_rng(g)
            a = np.argsort(rng.random((T, cfg.num_experts)), axis=1)[:, :cfg.top_k]
            lab = f"g{g}"
            cells.append({"label": lab, "g": g, "T": T, "T_unpadded": T, "layer": g % 27 + 1, "tile": tile,
                          "branch": branch, "counts": PL.counts_from_ids(a, cfg.num_experts),
                          "ids_digest": PL.digest(a), "step": g})
            ids[lab] = a.astype(np.int16)
            g += 1
    cells.append(dict(cells[-1], label="s16384", g=None, synthetic=True))
    ids["s16384"] = ids[cells[-2]["label"]]
    np.savez_compressed(d / "sample_ids.npz", **ids)
    import hashlib
    doc = {"schema": NB.SAMPLE_SCHEMA, "model": model, "seed": 1, "n_cells": len(cells), "run": "x",
           "ids_file": "sample_ids.npz", "ids_sha256": hashlib.sha256((d / "sample_ids.npz").read_bytes()).hexdigest(),
           "cells": cells}
    (d / "sample.json").write_text(json.dumps(doc))
    return d / "sample.json"


def test_ncu_subsample_is_stratified_by_tile_deterministic_and_never_synthetic(tmp_path):
    dec = _main_sample(tmp_path / "sd", "decode", {"16x64x128w4s4": [4] * 20, "32x64x128w4s3": [40] * 20,
                                                   "32x128x64w4s3": [80] * 3, "64x128x64w4s3": [120] * 20})
    pre = _main_sample(tmp_path / "sp", "prefill", {"64x128x64w8s3": [256] * 12, "128x128x64w8s3": [1024] * 12})
    p = NB.subsample({"decode": dec, "prefill": pre}, "ernie-4.5-21b-a3b", 10, tmp_path / "out")
    doc = json.loads(p.read_text())
    tiles = [c["tile"] for c in doc["cells"]]
    # decode: 10 over 4 tiles = 3, 3, 2 (only 3 cells, takes them all, its 2), 2; prefill: 5 + 5
    assert tiles.count("16x64x128w4s4") == 3 and tiles.count("32x64x128w4s3") == 3
    assert tiles.count("32x128x64w4s3") == 2 and tiles.count("64x128x64w4s3") == 2
    assert tiles.count("64x128x64w8s3") == 5 and tiles.count("128x128x64w8s3") == 5
    assert not any(c.get("synthetic") for c in doc["cells"]) and len({c["label"] for c in doc["cells"]}) == 20
    q = NB.subsample({"decode": dec, "prefill": pre}, "ernie-4.5-21b-a3b", 10, tmp_path / "out2")
    assert json.loads(q.read_text())["cells"] == doc["cells"]
    cells, ids, _sha = NB.load_cells(p, "ernie-4.5-21b-a3b", "prefill")
    assert len(cells) == 10 and all(ids[c["label"]].shape == (c["T"], 6) for c in cells)
    plan = NB.plan_of(p, "ernie-4.5-21b-a3b", "decode", profile_dir=tmp_path, stem="x", calls=3, warmups=2)
    assert NB.window(plan) == (10 * 2 * 2, 10 * 3 * 2)


def _page(cells: list[dict], branch: str) -> dict:
    out = []
    for i, c in enumerate(cells):
        out.append({"label": c["label"], "T": c["T"], "tile": c["tile"], "branch": branch,
                    "readback": {"verdict": "PASS"},
                    "per_gemm": {"w1": {"dram__bytes_read.sum": 2.0e8 + i}, "w2": {"dram__bytes_read.sum": 1.0e8}}})
    return {"schema": NB.PAGE_SCHEMA, "branch": branch, "cells": out}


def test_bytes_join_compares_counted_reads_with_v8s_and_counts_not_available(tmp_path):
    import score_prod as SP
    dec = _main_sample(tmp_path / "sd", "decode", {"16x64x128w4s4": [2, 2], "64x128x64w4s3": [120, 120]})
    pre = _main_sample(tmp_path / "sp", "prefill", {"128x128x64w8s3": [1024, 1024]})
    p = NB.subsample({"decode": dec, "prefill": pre}, "ernie-4.5-21b-a3b", 4, tmp_path / "out")
    cells = json.loads(p.read_text())["cells"]
    pages = [_page([c for c in cells if c["branch"] == b], b) for b in ("decode", "prefill")]
    seen = []

    def pricer(T, counts, ids, sag):
        seen.append((T, sag))
        if T <= 2:                                                       # the naive path: no reads in the answer
            return {"ms": 0.03, "parts": {"reads": None}}
        if T <= 128:
            return {"ms": 0.3, "parts": {"reads": {"w1": 1.0e8, "w2": 1.0e8}}}
        return {"ms": 1.0, "reads": {"w1": 2.0e8, "w2": 2.0e8}}
    rows = SP.bytes_rows(PL.MODELS["ernie-4.5-21b-a3b"], pages, p, sag128=(64.0, 600.0), pricer=pricer)
    assert len(rows) == 6 and all(s == (64.0, 600.0) for T, s in seen if T > 128)
    assert all(s is None for T, s in seen if T <= 128)
    na = [r for r in rows if r["pred_source"] == "NOT AVAILABLE"]
    assert [r["T"] for r in na] == [2, 2]
    summ = SP.bytes_summary(rows)
    assert summ["not_available"] == 2
    w1 = summ["tile"]["64x128x64w4s3"]["w1"]
    assert w1["n"] == 2 and w1["n_predicted"] == 2 and w1["ratio_meas_over_pred_median"] == pytest.approx(2.0, 1e-6)
    w2 = summ["tile"]["128x128x64w8s3"]["w2"]
    assert w2["ratio_meas_over_pred_median"] == pytest.approx(0.5) and w2["pred_bias"] == pytest.approx(100.0)
    assert summ["tile"]["16x64x128w4s4"]["w1"]["n_predicted"] == 0
    assert any("NCU DRAM READ BYTES" in ln for ln in SP.render_extras({"ncu_bytes": {"rows": rows, "summary": summ}}))


def test_predicted_reads_reads_prefill_and_decode_answers_and_refuses_nothing_silently():
    assert PL.predicted_reads({"ms": 1.0, "reads": {"w1": 1.0, "w2": 2.0}}) == {"w1": 1.0, "w2": 2.0, "source": "reads"}
    assert PL.predicted_reads({"ms": 1.0, "parts": {"reads": {"w1": 3, "w2": 4}}})["source"] == "parts.reads"
    assert PL.predicted_reads({"ms": None, "why": "HOPPER_G1_N"}) is None
    assert PL.predicted_reads({"ms": 0.1, "parts": {}}) is None


# ------------------------------------------------------------------------------------------------------------
# X3: Qwen3-30B-A3B
# ------------------------------------------------------------------------------------------------------------

def test_qwen3_config_is_the_hubs_and_pinned():
    m = PL.MODELS[QM]
    c = m.cfg
    assert (c.hidden_size, c.intermediate_size, c.num_experts, c.top_k, c.num_layers) == (2048, 768, 128, 8, 48)
    assert m.moe_layers == 48 and c.shared_experts == 0 and m.hf_repo == "Qwen/Qwen3-30B-A3B"
    assert PL.check_hf_config(m, QWEN3_HF) == []
    assert PL.check_hf_config(m, dict(QWEN3_HF, decoder_sparse_step=2))
    assert PL.check_hf_config(m, dict(QWEN3_HF, mlp_only_layers=[3]))
    assert PL.check_hf_config(m, dict(QWEN3_HF, num_experts_per_tok=6))
    assert re.fullmatch(r"[0-9a-f]{40}", PL.hf_revision(QM))


class _Tf515Qwen3Config:
    """transformers 5.15.1's Qwen3MoeConfig as the VM showed it on 2026-11-20: to_dict() names the expert count
    num_local_experts; num_experts survives only as an attribute_map alias."""
    attribute_map = {"num_experts": "num_local_experts"}

    def to_dict(self):
        d = {k: v for k, v in QWEN3_HF.items() if k != "num_experts"}
        d["num_local_experts"] = QWEN3_HF["num_experts"]
        return d


def test_g_cfg_reads_attribute_map_aliases_that_to_dict_drops():
    m = PL.MODELS[QM]
    cfg = _Tf515Qwen3Config()
    assert PL.check_hf_config(m, cfg.to_dict())          # the X3 stop: num_experts=None
    view = PL.hf_config_view(cfg)
    assert view["num_experts"] == 128 and PL.check_hf_config(m, view) == []


def test_hf_config_view_never_replaces_a_key_to_dict_holds():
    class C:
        attribute_map = {"num_experts": "num_local_experts"}

        def to_dict(self):
            return {"num_experts": 64, "num_local_experts": 128}
    assert PL.hf_config_view(C())["num_experts"] == 64

    class NoMap:
        def to_dict(self):
            return {"moe_num_experts": [64, 64]}
    assert PL.hf_config_view(NoMap()) == {"moe_num_experts": [64, 64]}


def test_prompts_write_creates_its_directory(tmp_path):
    out = tmp_path / "extras" / "long" / "prompts.jsonl"
    PR.write({"header": {"schema": "x"}, "rows": [{"a": 1}]}, out)
    assert out.read_text().splitlines() == ['{"schema": "x"}', '{"a": 1}']
    assert PR.tokenizer_revision("Qwen/Qwen3-30B-A3B") == PL.hf_revision(QM)
    import moe.spec as S
    assert QM not in S.MODEL_CONFIGS or S.MODEL_CONFIGS[QM] == PL.QWEN3    # spec.py untouched: runtime only
    import frontier_op as FO
    assert FO.frontier_config(QM)[0] == "qwen3-30b-a3b-moe"
    assert str(FO.csv_relpath(QM)) == "compute/gh200/qwen3-30b-a3b-moe/moe_kernel_only.csv"


def test_tiles_at_e128_move_g16_past_the_serving_budget():
    assert PL.tile_bins(64) == PL.TILE_BINS
    b128 = PL.tile_bins(128)
    assert b128[-2] == ("128x128x64w8s3", 513, 16511) and b128[-1][1] == 16512
    for T in (1, 32, 33, 96, 128, 129, 512, 513, 8255, 8256, 12288, 16384, 16511, 16512, 20000):
        assert PL.tile_bin(T, 128) == PL.tile_label(T, 128), T
        assert PL.tile_bin(T) == PL.tile_bin(T, 64)
    assert len(PL.tile_bins(128, 16384)) == 6 and len(PL.tile_bins(64, 16384)) == 7
    import serve_capture as SC
    est = SC.estimate_histogram([1, 8], [{"input_len": 40, "max_tokens": 8}], 8, E=128)
    assert "128x128x64w8s3 g16" not in est["per_tile"] and len(est["per_tile"]) == 6
    assert set(SC.estimate_histogram([1], [{"input_len": 40, "max_tokens": 8}], 8)["per_tile"]) == \
        {lab for lab, _lo, _hi in PL.TILE_BINS}
    assert PL.refused_window(PL.MODELS[QM].cfg) == (12288, 16511)
    # the refused stratum is named by the model's own window (review of the extras, MINOR), ERNIE's unchanged
    qs = PL.refused_stratum(8, 128)
    assert qs == "refused_window_12288_16511" and qs in PL.strata(12288, 8, 128) and qs not in PL.strata(12287, 8, 128)
    assert PL.REFUSED_STRATUM not in PL.strata(12288, 8, 128)
    assert PL.strata(8200, 6, 64) == ["refused_window_8192_8255"] == [PL.REFUSED_STRATUM]
    assert "small_batch_align_on_prefill_tile" not in PL.strata(130, 7, 128)


def test_qwen3_prices_through_v8_on_all_six_tiles_and_refuses_only_its_hopper_window():
    """The pricing smoke (its own interpreter: v8 swaps frozen functions in-process): prodlib.price_check, then one
    call per default tile Qwen3's serving reaches plus its HOPPER_G1_N window, each priced or refused by a declared
    class."""
    code = r"""
import json, sys
sys.path[:0] = [sys.argv[1]]
import numpy as np, prodlib as PL, r3_timing_model_v8 as V8
cfg = PL.MODELS["qwen3-30b-a3b"].cfg
out = {"check": {str(k): v["ms"] for k, v in PL.price_check("qwen3-30b-a3b").items()}, "calls": []}
for T in (8, 48, 80, 112, 256, 1024, 12288):
    rng = np.random.default_rng(T)
    ids = np.argsort(rng.random((T, 128)), axis=1)[:, :8]
    a = PL.price_v8(cfg, T, PL.counts_from_ids(ids, 128), ids, sag128=PL.REF_PAR_MEDIAN)
    out["calls"].append({"T": T, "tile": a["tile"], "ms": a["ms"],
                         "cls": None if a["ms"] is not None else V8.refusal_class(a["why"]),
                         "reads": PL.predicted_reads(a)})
print(json.dumps(out, default=str))
"""
    p = subprocess.run([sys.executable, "-c", code, str(PROD)], capture_output=True, text=True, cwd=str(REPO),
                       timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    doc = json.loads(p.stdout.strip().splitlines()[-1])
    assert all(v and v > 0 for v in doc["check"].values())
    by = {c["T"]: c for c in doc["calls"]}
    assert {by[T]["tile"] for T in (8, 48, 80, 112, 256, 1024)} == {PL.tile_label(T, 128) for T in
                                                                    (8, 48, 80, 112, 256, 1024)}
    for T in (8, 48, 80, 112, 256, 1024):
        assert by[T]["ms"] and by[T]["ms"] > 0, by[T]
    assert by[256]["reads"] and by[1024]["reads"]["source"] == "reads"
    assert by[12288]["ms"] is None and by[12288]["cls"] == "HOPPER_G1_N"


# ------------------------------------------------------------------------------------------------------------
# X4: the long-context prompt set
# ------------------------------------------------------------------------------------------------------------

def test_build_long_concatenates_turns_to_lengths_in_range_deterministically(tmp_path):
    data = [{"id": f"c{i}", "conversations": [{"from": "human", "value": " ".join(["w"] * (50 + i))},
                                              {"from": "gpt", "value": "ok"}]} for i in range(400)]
    f = tmp_path / "sg.json"
    f.write_text(json.dumps(data))
    enc = lambda s: [len(w) for w in s.split()]                          # noqa: E731  (one token per word)
    a = PR.build_long(data_path=f, encode=enc, n=6, seed=0, min_input=300, max_input=900, out_dist="fixed:16")
    b = PR.build_long(data_path=f, encode=enc, n=6, seed=0, min_input=300, max_input=900, out_dist="fixed:16")
    assert a == b and a["header"]["kind"] == "sharegpt-long"
    for r in a["rows"]:
        assert 300 <= r["input_len"] <= 900 and len(r["prompt_token_ids"]) == r["input_len"] and r["max_tokens"] == 16
    PR.write(a, tmp_path / "p.jsonl")
    assert PR.load(tmp_path / "p.jsonl")["rows"] == a["rows"]
    with pytest.raises(PR.PromptError):
        PR.build_long(data_path=f, encode=enc, n=1, seed=0, min_input=300, max_input=900, out_dist="sharegpt")
    with pytest.raises(PR.PromptError):
        PR.build_long(data_path=f, encode=enc, n=500, seed=0, min_input=800, max_input=900, out_dist="fixed:4")


# ------------------------------------------------------------------------------------------------------------
# offline: the extras in the same report
# ------------------------------------------------------------------------------------------------------------

def test_offline_with_extras_scores_them_in_the_same_report(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--extras", "--results", str(tmp_path / "res")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    steps = [x for x in p.stdout.splitlines() if x.startswith("STEP ")]
    by = {x.split()[1]: x for x in steps}
    names = [x.split()[1] for x in steps]
    assert names == ["decompress", "expected-outputs", "extras-expected-outputs", "score", "score-lc1410", "score-qwen3",
                     "score-merge-lc1410", "score-merge-qwen3"]
    ex = f"{tmp_path}/res/extras"
    sc = by["score"]
    for want in (f"--noise-floor A1710={ex}/A1710r", f"--ncu-bytes {ex}/ncu-decode/ncu_bytes.json",
                 f"--ncu-bytes {ex}/ncu-prefill/ncu_bytes.json", f"--ncu-sample {ex}/ncu-sample/sample.json",
                 "--model ernie-4.5-21b-a3b"):
        assert want in sc, want
    assert not _soft(sc)
    # review of the extras M5: LC1410 is never in main's score step; its own soft score, merged like Qwen3's
    assert "LC1410" not in sc
    lc = by["score-lc1410"]
    assert _soft(lc) and f"--run LC1410={ex}/LC1410" in lc and "--model ernie-4.5-21b-a3b" in lc
    assert f"--out {tmp_path}/res/score/lc1410" in lc and "--noise-floor" not in lc
    q = by["score-qwen3"]
    assert _soft(q) and f"--model {QM}" in q and f"--sag {tmp_path}/res/sag/par.json" in q
    assert f"--run A1710={ex}/qwen3/A1710" in q and f"--frontier {ex}/qwen3/frontier/predictions" in q
    for lab in ("lc1410", "qwen3"):
        m = by[f"score-merge-{lab}"]
        assert _soft(m) and f"needs=score-{lab}" in m and f"--extra {lab}={tmp_path}/res/score/{lab}" in m


def test_merge_appends_the_qwen3_score_once(tmp_path):
    import score_prod as SP
    main, q = tmp_path / "score", tmp_path / "score" / "qwen3"
    q.mkdir(parents=True)
    (main / "prod.score.json").write_text(json.dumps({"model": "ernie-4.5-21b-a3b", "runs": {}}))
    (main / "prod.score.txt").write_text("PRODUCTION CHECK ernie\n")
    (q / "prod.score.json").write_text(json.dumps({"model": QM, "runs": {"A1710": {}}}))
    (q / "prod.score.txt").write_text("PRODUCTION CHECK qwen3\n")
    assert SP.main(["merge", "--into", str(main), "--extra", f"qwen3={q}"]) == 0
    assert SP.main(["merge", "--into", str(main), "--extra", f"qwen3={q}"]) == 0
    doc = json.loads((main / "prod.score.json").read_text())
    txt = (main / "prod.score.txt").read_text()
    assert doc["extras"]["qwen3"]["model"] == QM and doc["model"] == "ernie-4.5-21b-a3b"
    assert txt.count("== EXTRAS qwen3:") == 1 and txt.startswith("PRODUCTION CHECK ernie") and "qwen3\n" in txt


def test_extras_expected_outputs_hold_each_passed_step_to_its_file(tmp_path):
    ex = tmp_path / "extras"
    ex.mkdir()
    (ex / "SUMMARY.txt").write_text("PASS X1-extract (9s)\nPASS X2-ncu-decode (5s)\nFAIL (soft, dropped) X2-ncu-prefill\n")
    got = XL.expected_outputs(tmp_path)
    assert got["extras"] and len(got["missing"]) == 2
    (ex / "A1710r").mkdir()
    (ex / "A1710r" / "calls.shards.json").write_text("{}")
    (ex / "ncu-decode").mkdir()
    (ex / "ncu-decode" / "ncu_bytes.json.xz").write_text("x")
    got = XL.expected_outputs(tmp_path)
    assert not got["missing"] and set(got["have"]) == {"X1-extract", "X2-ncu-decode"}
    assert XL.expected_outputs(tmp_path / "none")["extras"] is False


# ------------------------------------------------------------------------------------------------------------
# REVIEW-EXTRAS (2026-11-20): one test per item
# ------------------------------------------------------------------------------------------------------------

def _steps(p) -> dict:
    return {x.split()[1]: x for x in p.stdout.splitlines() if x.startswith("STEP ")}


def test_b1_the_commit_switch_is_documented_for_a_repo_clone_and_a_bundle_clone():
    """B1: a bundle-cloned VM has no origin; both forms, then a detached checkout of the sha."""
    for f in (PROD / "README.md", PROD / "extras.sh", PROD / "extras_lib.py"):
        t = f.read_text()
        assert "git fetch origin && git checkout --detach <extras sha>" in t, f
        assert "git fetch <bundle> '+refs/heads/*:refs/remotes/origin/*'" in t, f
        assert "git checkout <extras sha>" not in t, f


def test_m1_main_finished_needs_mains_held_back_check_or_the_logged_override(tmp_path):
    (tmp_path / "SUMMARY.txt").write_text("PASS env-preflight (3s)\nFAIL A1410-capture: exit 3: boom\n")
    ok, why = XL.main_finished(tmp_path, running=[])
    assert not ok and "PASS held-back-check" in why and "--after-main-fail" in why
    ok, why = XL.main_finished(tmp_path, after_main_fail=True, running=[])
    assert ok and why.startswith("OVERRIDE --after-main-fail")
    # a FAIL held-back-check is not an end
    (tmp_path / "SUMMARY.txt").write_text("FAIL held-back-check: exit 3: held back\n")
    assert not XL.main_finished(tmp_path, running=[])[0]
    # refused while main.sh or anything it leaves on the card runs, override or not
    (tmp_path / "SUMMARY.txt").write_text("PASS held-back-check (1s)\n")
    for busy in ("main.sh (pid 7)", "serve_capture (pid 8)", "locked_r3 (pid 9)", "nsys (pid 10)"):
        ok, why = XL.main_finished(tmp_path, after_main_fail=True, running=[busy])
        assert not ok and busy in why
    pats = dict(XL.BUSY_PATTERNS)
    assert pats["main.sh"] == "[m]ain[.]sh" and set(pats) == {"main.sh", "serve_capture", "locked_r3", "nsys"}
    for name, line in (("nsys", "/usr/local/cuda/bin/nsys profile -o x python3 a.py"),
                       ("serve_capture", "python3 scripts/prod/serve_capture.py --model m"),
                       ("locked_r3", "python3 scripts/locked_r3.py --locks 1710"), ("main.sh", "bash scripts/prod/main.sh")):
        assert re.search(pats[name], line), name
    assert not re.search(pats["nsys"], "python3 scripts/prod/nsys_extract.py --run x")


def test_m1_extras_sh_passes_the_override_and_records_it(tmp_path):
    p = _dry(tmp_path, "--after-main-fail")
    assert p.returncode == 0, p.stderr
    assert "--after-main-fail" in _steps(p)["extras-main-finished"]
    assert "--after-main-fail" not in _steps(_dry(tmp_path / "b"))["extras-main-finished"]
    assert 'kit_record "OVERRIDE extras-main-finished: --after-main-fail' in (PROD / "extras.sh").read_text()


def test_m2_the_extras_exit_writes_extras_done_and_never_pushes_final(tmp_path):
    """M2: with KIT_DONE_FILE the EXIT trap writes it and pushes without --final (DRIVER-DONE stays main's); main.sh
    (no KIT_DONE_FILE) still pushes --final RC."""
    stub, got, done = tmp_path / "push_stub.sh", tmp_path / "pushed.txt", tmp_path / "EXTRAS-DONE"
    stub.write_text(f'printf "%s\\n" "$@" > {got}\n')
    script = tmp_path / "seq.sh"
    script.write_text(f"set -euo pipefail\n. {PROD / 'kitlib.sh'}\nkit_init {tmp_path / 'r'}\nKIT_DONE_FILE={done}\n"
                      "kit_trap_unlock --push\nstep boom --cap 10 -- bash -c 'exit 4'\nkit_summary\n")
    env = dict(os.environ, KIT_KILL_AFTER="2", KIT_PUSH_SCRIPT=str(stub))
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 4
    assert done.read_text().startswith("exit 4 at ")
    args = got.read_text().split("\n")
    assert args[0] == "push" and "--final" not in args
    # an inherited KIT_DONE_FILE never reaches main.sh: kitlib resets it when sourced
    env2 = dict(env, KIT_DONE_FILE=str(tmp_path / "leak"))
    script.write_text(f"set -euo pipefail\n. {PROD / 'kitlib.sh'}\nkit_init {tmp_path / 'r2'}\n"
                      "kit_trap_unlock --push\nstep boom --cap 10 -- bash -c 'exit 5'\nkit_summary\n")
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env2, timeout=120)
    args = got.read_text().split("\n")
    assert p.returncode == 5 and args[args.index("--final") + 1] == "5" and not (tmp_path / "leak").exists()
    for f in (PROD / "extras.sh", PROD / "README.md"):
        t = " ".join(f.read_text().replace("#", " ").split())
        assert "TERMINATE ONLY AFTER THE EXTRAS" in t.upper() and "EXTRAS-DONE" in t, f
        assert "PASS extras-held-back-check" in t, f
    assert 'KIT_DONE_FILE="$EX/EXTRAS-DONE"' in (PROD / "extras.sh").read_text()


def test_m3_extras_env_preflight_skips_frontier_setup_and_the_download_rate(tmp_path):
    ep = _steps(_dry(tmp_path))["env-preflight"]
    assert "--skip P9\\,P11\\,P12" in ep and "--expect-commit" in ep


def test_m4_steps_past_the_deadline_are_skipped_capped_and_the_trap_still_runs(tmp_path):
    import time
    stub, got = tmp_path / "push_stub.sh", tmp_path / "pushed.txt"
    stub.write_text(f'printf "%s\\n" "$@" > {got}\n')
    now = int(time.time())
    body = (f"KIT_DEADLINE={now + 4}\nkit_trap_unlock --push\n"
            "step slow --soft --cap 100 -- sleep 30\n"                      # capped at the ~4 s left: TIMEOUT
            f"step late --soft --cap 10 -- touch {tmp_path / 'late'}\n"      # starts past it: SKIPPED, not run
            "step dep --soft --cap 10 --needs late -- true\n"
            f"step final --soft --no-deadline --cap 10 -- touch {tmp_path / 'final'}\n")
    script = tmp_path / "seq.sh"
    script.write_text(f"set -euo pipefail\n. {PROD / 'kitlib.sh'}\nkit_init {tmp_path / 'r'}\n{body}kit_summary\n")
    env = dict(os.environ, KIT_KILL_AFTER="1", KIT_PUSH_SCRIPT=str(stub))
    t0 = time.time()
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=120)
    summ = (tmp_path / "r" / "SUMMARY.txt").read_text().splitlines()
    assert p.returncode == 0 and time.time() - t0 < 25, p.stderr
    assert summ[0].startswith("FAIL (soft, dropped) slow: exit 124: TIMEOUT")
    assert summ[1].startswith("SKIPPED late: past the deadline") and not (tmp_path / "late").exists()
    assert summ[2].startswith("SKIPPED dep: needs late")
    assert summ[3].startswith("PASS final") and (tmp_path / "final").exists()
    assert got.read_text().startswith("push\n")                             # the EXIT trap pushed
    # extras.sh exports the deadline (start + budget + grace) and marks its last two steps --no-deadline
    src = (PROD / "extras.sh").read_text()
    assert "EXTRAS_DEADLINE=$(( EXTRAS_T0 + BUDGET + GRACE ))" in src and 'KIT_DEADLINE="$EXTRAS_DEADLINE"' in src
    by = _steps(_dry(tmp_path / "d"))
    assert " nodeadline " in by["extras-push-final"] and " nodeadline " in by["extras-held-back-check"]
    assert " nodeadline " not in by["X1-capture"]


def test_m5_merge_keeps_mains_report_and_every_extra(tmp_path):
    import score_prod as SP
    main = tmp_path / "score"
    main.mkdir()
    (main / "prod.score.json").write_text(json.dumps({"model": "ernie-4.5-21b-a3b", "runs": {"A1710": {}}}))
    (main / "prod.score.txt").write_text("PRODUCTION CHECK ernie\n")
    for lab, model in (("lc1410", "ernie-4.5-21b-a3b"), ("qwen3", QM)):
        (main / lab).mkdir()
        (main / lab / "prod.score.json").write_text(json.dumps({"model": model, "runs": {lab: {}}}))
        (main / lab / "prod.score.txt").write_text(f"PRODUCTION CHECK {lab}\n")
    assert SP.main(["merge", "--into", str(main), "--extra", f"lc1410={main / 'lc1410'}"]) == 0
    assert SP.main(["merge", "--into", str(main), "--extra", f"qwen3={main / 'qwen3'}"]) == 0
    assert SP.main(["merge", "--into", str(main), "--extra", f"lc1410={main / 'lc1410'}"]) == 0   # again
    txt = (main / "prod.score.txt").read_text()
    assert txt.count("== EXTRAS lc1410:") == 1 and txt.count("== EXTRAS qwen3:") == 1
    assert "concatenated unrelated turns" in txt and txt.startswith("PRODUCTION CHECK ernie")
    doc = json.loads((main / "prod.score.json").read_text())
    assert set(doc["extras"]) == {"lc1410", "qwen3"} and doc["runs"] == {"A1710": {}}
    # main's own report kept, untouched
    assert (main / "prod.score.main.txt").read_text() == "PRODUCTION CHECK ernie\n"
    assert "extras" not in json.loads((main / "prod.score.main.json").read_text())


def test_m6_qwen3_replay_probes_on_qwen3_priced_by_active_bytes(tmp_path):
    import replay as RP
    by = _steps(_dry(tmp_path))
    names = list(by)
    for b in ("decode", "prefill"):
        ps = by[f"X3-replay-probe-sample-{b}"]
        assert "--per-tile 2" in ps and "--scale active-bytes" in ps and f"--model {QM}" in ps
        wm = by[f"X3-replay-probe-wall-{b}"]
        assert f"--model {QM}" in wm and f"replay-probe-{b}/replay_wall.json" in wm
        rest = by[f"X3-replay-sample-{b}"]
        assert f"--probe-wall {tmp_path}/res/extras/qwen3/replay-probe-{b}/replay_wall.json" in rest
        assert names.index(f"X3-replay-probe-{b}") < names.index(f"X3-replay-sample-{b}")
    # the active-bytes rule: per tile, Qwen3's activated bytes over ERNIE's (2x the expert-bytes ratio's 0.4)
    w = {"schema": RP.WALL_SCHEMA, "model": "ernie-4.5-21b-a3b", "repeats": 9, "total_wall_s": 100.0,
         "tiles": {"16x64x128w4s4": {"median_wall_s": 10.0, "cells": 2},
                   "64x128x64w4s3": {"median_wall_s": 20.0, "cells": 2}}}
    q = {"16x64x128w4s4": 2, "32x64x128w4s3": 2, "64x128x64w4s3": 2}
    eb = RP.budget_quota(q, w, "decode", repeats=9, target=QM, budget_s=1e6)
    ab = RP.budget_quota(q, w, "decode", repeats=9, target=QM, budget_s=1e6, scale_rule="active-bytes")
    assert eb["model_scale"] == pytest.approx(0.4) and "scale_rule" not in eb
    s16, s64 = ab["model_scale"]["16x64x128w4s4"], ab["model_scale"]["64x128x64w4s3"]
    assert 0.5 < s16 < 0.65 and 0.75 < s64 < 0.85 and ab["scale_rule"] == "active-bytes"
    assert ab["per_cell_s"]["16x64x128w4s4"] == pytest.approx(RP.WALL_SAFETY * 10.0 * s16)
    assert ab["per_cell_s"]["32x64x128w4s3"] == pytest.approx(RP.WALL_SAFETY * 20.0 * s64)   # untimed: dearest
    assert RP.model_scale({"model": QM}, QM, rule="active-bytes", tile="16x64x128w4s4") == pytest.approx(1.0)


def test_m7_qwen3_serves_without_the_1024_bursts_and_the_estimator_still_covers_every_tile(tmp_path):
    import plan_time as PT
    import serve_capture as SC
    by = _steps(_dry(tmp_path))
    assert "--concurrency 1\\,8\\,24\\,48\\,80\\,112\\,256 " in by["X3-A1710-capture"]
    assert "1024x3" not in by["X3-A1710-capture"] and "1024x3" in by["X1-capture"]
    est = SC.estimate_histogram(SC.parse_levels(PT.QWEN3_LEVELS), PT.length_rows(), 8, E=128)
    assert len(est["per_tile"]) == 6 and min(est["per_tile"].values()) >= SC.MIN_STEPS_PER_TILE, est["per_tile"]


class _Req:
    def __init__(self, rid):
        self.request_id = rid


def test_m7_every_manifest_records_kv_blocks_preemptions_and_realized_concurrency():
    import types

    import serve_capture as SC
    sched = types.SimpleNamespace(running=[], waiting=[])
    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(num_gpu_blocks=4321, block_size=16)),
        engine_core=types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=sched))))
    kv = SC.KVTracker(llm)
    kv.level(0, 3)
    for run, wait in ((["a", "b", "c"], []), (["a", "b"], ["c"]), (["a", "c"], ["b"]), (["c"], [])):
        sched.running, sched.waiting = [_Req(x) for x in run], [_Req(x) for x in wait]
        kv.observe()
    s = kv.summary()
    assert s["num_gpu_blocks"] == 4321 and s["preemptions"] == 2
    lv = s["realized_concurrency"][0]
    assert lv["L"] == 3 and lv["steps"] == 4 and lv["running_max"] == 3 and lv["steps_at_L"] == 1
    assert lv["preemptions"] == 2 and lv["waiting_max"] == 1
    # an engine that exposes nothing is recorded NOT AVAILABLE, never left out
    bare = SC.KVTracker(types.SimpleNamespace(llm_engine=types.SimpleNamespace()))
    bare.level(0, 1)
    bare.observe()
    s = bare.summary()
    assert all(str(s[k]).startswith("NOT AVAILABLE") for k in ("num_gpu_blocks", "preemptions", "realized_concurrency"))
    src = (PROD / "serve_capture.py").read_text()
    assert src.count('manifest["kv"] = kv.summary()') == 2 and "kv.observe()" in src


def test_m8_x1_checks_its_manifest_against_a1710_before_the_extract(tmp_path):
    by = _steps(_dry(tmp_path))
    x = by["X1-identical"]
    assert _soft(x) and "needs=X1-capture" in x and f"--base {tmp_path}/res/A1710" in x
    assert f"--repeat {tmp_path}/res/extras/A1710r" in x
    src = (PROD / "score_prod.py").read_text()
    assert "XL.identical_files(_kv(a.run)[base], rdir)" in src and "XL.pair_by_step" in src


def test_minor_x3_needs_v8_to_price_qwen3_and_runs_frontiers_grid_check(tmp_path):
    by = _steps(_dry(tmp_path))
    assert _soft(by["price-check-qwen3"]) and "needs=price-check-qwen3" in by["X3-budget"]
    assert "price-check-qwen3" not in by["X1-budget"]
    g = by["X3-frontier-grid-check"]
    assert _soft(g) and f"--model {QM}" in g and "grid-check" in g and "needs=X3-frontier-env" in g
    assert "needs=X3-frontier-grid-check" in by["X3-frontier-lock-decode"]


def test_minor_commit_switch_refuses_an_untracked_file_as_p1_does(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a").write_text("x\n")
    _git(repo, "add", "a")
    _git(repo, "commit", "-q", "-m", "m")
    sha = _git(repo, "rev-parse", "HEAD")
    res = tmp_path / "res"
    _preflight(res, sha)
    assert XL.commit_switch(repo, res, sha)[0]
    (repo / "stray.txt").write_text("u\n")
    ok, why = XL.commit_switch(repo, res, sha)
    assert not ok and "dirty" in why and "stray.txt" in why
    import env_preflight as EP
    assert not EP.p1_git(repo, sha, "test")[0]                           # the same verdict


def test_minor_the_budget_prices_the_runs_own_levels(tmp_path, monkeypatch):
    import plan_time as PT
    small = PT.extras_plan(levels="1,8", n=8)["blocks"]["X1"]
    big = PT.extras_plan()["blocks"]["X1"]
    assert small < big
    t0 = 1000.0
    now = t0 + 9000 - (small + big) / 2                                  # fits the small run, not the default
    args = ["budget", "--block", "X1", "--t0", str(t0), "--budget-s", "9000", "--now", str(now)]
    assert XL.main(args + ["--levels", "1,8", "--n", "8"]) == 0
    assert XL.main(args) == 3
    assert "--levels 1\\,8\\,24" in _steps(_dry(tmp_path))["X1-budget"]


def test_minor_p10_need_comes_from_mains_trace_sizes(tmp_path):
    need, why = XL.disk_need_gb(tmp_path)
    assert need is None and "traces.sha256.json" in why
    files = {"A1710/serve.nsys-rep": {"bytes": 30e9}, "A1710/serve.sqlite": {"bytes": 10e9},
             "A1410/serve.nsys-rep": {"bytes": 20e9}, "gate60/serve.nsys-rep": {"bytes": 1e9},
             "replay-nsys-decode/x": {"bytes": 99e9}}
    (tmp_path / "traces.sha256.json").write_text(json.dumps({"files": files}))
    need, why = XL.disk_need_gb(tmp_path)
    # X1 = A1710 (40 GB), X3 = gate60 + A1710 + A1410 + Qwen3's weights, X4 = A1410; replay traces not counted
    assert need == pytest.approx(1.25 * (40 + (1 + 40 + 20 + XL.QWEN3_WEIGHTS_GB) + 20))
    assert XL.disk_need_gb(tmp_path, long_context=False)[0] == pytest.approx(1.25 * (40 + 61 + XL.QWEN3_WEIGHTS_GB))
    ep = _steps(_dry(tmp_path / "d"))["env-preflight"]
    assert "--min-free-gb 200 " in ep                                     # no manifest in the dry run: the default


def test_minor_x3_pushes_each_capture_on_its_own_after_a_size_check(tmp_path):
    by = _steps(_dry(tmp_path))
    names = list(by)
    for lab in ("A1710", "A1410"):
        ps = by[f"X3-{lab}-push-size"]
        assert _soft(ps) and f"needs=X3-{lab}-extract" in ps and f"--dir {tmp_path}/res/extras/qwen3/{lab}" in ps
        assert names.index(f"X3-{lab}-push-size") < names.index(f"X3-push-{lab}") < names.index(f"X3-push-{lab}") + 1
    assert "X3-push-captures" not in by
    d = tmp_path / "run"
    (d / "ring").mkdir(parents=True)
    (d / "ring" / "shard_00000.npz").write_bytes(b"x" * 1000)
    (d / "serve.sqlite").write_bytes(b"x" * 5000)                          # expendable: not pushed
    ok, why = XL.push_size([d], total_max_mb=0.002)
    assert ok and "0 MB" in why
    ok, why = XL.push_size([d], total_max_mb=0.0005)
    assert not ok and "copy by hand" in why


def test_minor_x4_prompts_are_labelled_concatenated_unrelated_turns(tmp_path):
    import score_prod as SP
    data = [{"id": f"c{i}", "conversations": [{"from": "human", "value": " ".join(["w"] * 60)},
                                              {"from": "gpt", "value": "ok"}]} for i in range(40)]
    f = tmp_path / "sg.json"
    f.write_text(json.dumps(data))
    a = PR.build_long(data_path=f, encode=lambda s: [len(w) for w in s.split()], n=2, seed=0, min_input=200, max_input=300,
                      out_dist="fixed:16")
    assert a["header"]["label"] == "concatenated unrelated turns"
    assert "concatenated unrelated turns" in SP.EXTRA_ROLES["LC1410"]


def test_minor_v8s_decode_answers_carry_parts_reads_for_the_bytes_join():
    """The real v8 (its own interpreter): every ERNIE decode call it prices carries parts.reads, so X2's decode rows
    are compared, and bytes_summary counts each branch's sources (NOT AVAILABLE explicit)."""
    import score_prod as SP
    code = r"""
import json, sys
sys.path[:0] = [sys.argv[1]]
import numpy as np, prodlib as PL
cfg = PL.MODELS["ernie-4.5-21b-a3b"].cfg
out = []
for T in (2, 8, 48, 112):
    rng = np.random.default_rng(T)
    ids = np.argsort(rng.random((T, 64)), axis=1)[:, :6]
    rd = PL.predicted_reads(PL.price_v8(cfg, T, PL.counts_from_ids(ids, 64), ids))
    out.append(rd and rd["source"])
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", code, str(PROD)], capture_output=True, text=True, cwd=str(REPO),
                       timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout.strip().splitlines()[-1]) == ["parts.reads"] * 4
    rows = [{"branch": "decode", "tile": "t", "pred_source": "parts.reads", "meas_w1": 1.0, "pred_w1": 1.0},
            {"branch": "decode", "tile": "t", "pred_source": "NOT AVAILABLE", "meas_w1": 1.0, "pred_w1": None}]
    s = SP.bytes_summary(rows)
    assert s["pred_source_by_branch"] == {"decode": {"parts.reads": 1, "NOT AVAILABLE": 1}}
    assert any("decode predicted reads: NOT AVAILABLE 1, parts.reads 1" in ln
               for ln in SP.render_extras({"ncu_bytes": {"rows": rows, "summary": s}}))
