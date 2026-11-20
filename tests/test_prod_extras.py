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
    assert f"--wall {res}/replay-probe-decode/replay_wall.json" in by["X3-replay-sample-decode"]
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
    ok, why = XL.main_finished(tmp_path, running=False)
    assert not ok and "SUMMARY.txt" in why
    (tmp_path / "SUMMARY.txt").write_text("PASS env-preflight (3s)\nPASS held-back-check (1s)\n")
    assert not XL.main_finished(tmp_path, running=True)[0]
    ok, why = XL.main_finished(tmp_path, running=False)
    assert ok and "reached held-back-check" in why


def test_inputs_check_names_what_main_lacks(tmp_path):
    ok, why = XL.inputs_check("X2", tmp_path)
    assert not ok and "sample-decode/sample.json" in why
    assert XL.inputs_check("X4", tmp_path)[0]


# ------------------------------------------------------------------------------------------------------------
# X1: the noise floor
# ------------------------------------------------------------------------------------------------------------

def _row(g, layer, T, dig, ms, tile="16x64x128w4s4"):
    return {"g": g, "layer": layer, "T_pad": T, "counts_digest": dig, "kernel_sum_ms": ms, "tile": tile}


def test_noise_floor_pairs_the_kth_occurrence_of_a_key_and_reports_per_tile():
    a = [_row(0, 1, 8, "d1", 0.10), _row(1, 1, 8, "d1", 0.20), _row(2, 2, 8, "d2", 0.30),
         _row(3, 1, 1024, "d9", 1.0, "128x128x64w8s3"), _row(4, 3, 8, "dx", 0.5)]
    b = [_row(10, 1, 8, "d1", 0.11), _row(11, 2, 8, "d2", 0.30), _row(12, 1, 1024, "d9", 1.1, "128x128x64w8s3"),
         _row(13, 1, 8, "d1", 0.22), _row(14, 1, 8, "d1", 0.5)]
    pairs = XL.match_calls(a, b)
    got = sorted((x["g"], y["g"]) for x, y in pairs)
    # d1 occurs twice in a and three times in b: 1st with 1st (g10), 2nd with 2nd (g13); dx and b's 3rd d1 unmatched
    assert got == [(0, 10), (1, 13), (2, 11), (3, 12)]
    nf = XL.noise_floor(a, b)
    assert nf["matched_calls"] == 4 and nf["matched"]["all"]["n"] == 4
    t = nf["matched"]["128x128x64w8s3"]
    assert t["n"] == 1 and t["abs_rel_median_pct"] == pytest.approx(100 * 0.1 / 1.05)
    assert t["signed_median_pct"] == pytest.approx(10.0)
    small = nf["matched"]["16x64x128w4s4"]
    assert small["n"] == 3 and small["abs_rel_median_pct"] == pytest.approx(100 * 0.01 / 0.105)
    med = nf["per_tile_medians"]["16x64x128w4s4"]
    assert med["n_a"] == 4 and med["n_b"] == 4 and med["a_ms"] == pytest.approx(0.25)


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
    assert PR.tokenizer_revision("Qwen/Qwen3-30B-A3B") == PL.hf_revision(QM)
    import moe.spec as S
    assert QM not in S.MODEL_CONFIGS or S.MODEL_CONFIGS[QM] == c          # spec.py untouched: runtime only
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
    assert PL.REFUSED_STRATUM in PL.strata(12288, 8, 128) and PL.REFUSED_STRATUM not in PL.strata(12287, 8, 128)
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
    assert names == ["decompress", "expected-outputs", "extras-expected-outputs", "score", "score-qwen3", "score-merge"]
    ex = f"{tmp_path}/res/extras"
    sc = by["score"]
    for want in (f"--noise-floor A1710={ex}/A1710r", f"--ncu-bytes {ex}/ncu-decode/ncu_bytes.json",
                 f"--ncu-bytes {ex}/ncu-prefill/ncu_bytes.json", f"--ncu-sample {ex}/ncu-sample/sample.json",
                 f"--run LC1410={ex}/LC1410", "--model ernie-4.5-21b-a3b"):
        assert want in sc, want
    assert not _soft(sc)
    q = by["score-qwen3"]
    assert _soft(q) and f"--model {QM}" in q and f"--sag {tmp_path}/res/sag/par.json" in q
    assert f"--run A1710={ex}/qwen3/A1710" in q and f"--frontier {ex}/qwen3/frontier/predictions" in q
    assert _soft(by["score-merge"]) and "needs=score-qwen3" in by["score-merge"]


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
