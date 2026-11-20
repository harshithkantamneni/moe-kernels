"""scripts/prod blind profile (2026-11-20): ONE new blind model per 1x GH200. The background steps of kitlib.sh, main.sh
--model / --profile blind, G-COV per branch, Frontier's stale models cleared, the blind registry (one dict entry), the
per-instance names, expected outputs without the dropped runs, and v9 scored beside v8 (--also-price)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"
for _p in (REPO, REPO / "scripts", PROD, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frontier_op as FO  # noqa: E402
import prodlib as PL  # noqa: E402
import serve_capture as S  # noqa: E402

SPEC = {"hf_repo": "acme/MoE-16B-A2B", "hidden_size": 2048, "moe_intermediate_size": 1024, "num_experts": 64,
        "top_k": 6, "num_layers": 28, "first_moe_layer": 1, "levels": "1,8,24,48,80,112,256",
        "frontier_config": {"num_attention_heads": 16, "vocab_size": 102400}}


# ---- kitlib.sh: step --bg / step_join -------------------------------------------------------------------------

def _kit(tmp_path, body: str, **env_extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "seq.sh"
    script.write_text("set -euo pipefail\n. " + str(PROD / "kitlib.sh") + "\nkit_init " + str(tmp_path / "r")
                      + "\n" + body + "\nkit_summary\n")
    env = dict(os.environ, KIT_KILL_AFTER="2", **env_extra)
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    summ = tmp_path / "r" / "SUMMARY.txt"
    return p, (summ.read_text().splitlines() if summ.exists() else [])


def test_background_steps_overlap_and_are_recorded_like_foreground_steps(tmp_path):
    t0 = time.time()
    p, summ = _kit(tmp_path, "step a --bg --cap 30 -- sleep 2\nstep b --bg --cap 30 -- sleep 2\n"
                             "step fg --cap 30 -- sleep 2\nstep_join a\nstep_join b")
    dt = time.time() - t0
    assert p.returncode == 0, p.stderr
    assert dt < 5.5, dt                                   # three 2 s steps overlapped, not 6 s in a row
    assert [x.split()[1] for x in summ] == ["fg", "a", "b"] and all(x.startswith("PASS ") for x in summ)
    assert "STARTED a (background" in p.stdout


def test_a_soft_background_failure_is_dropped_and_its_dependents_skipped(tmp_path):
    flag = tmp_path / "ran"
    p, summ = _kit(tmp_path, f"step s --bg --soft --cap 30 -- false\nstep dep --cap 30 --needs s -- touch {flag}\n"
                             "step other --cap 30 -- true")
    assert p.returncode == 0 and not flag.exists()
    # the dependent JOINS the running prerequisite first: its result decides, never a race
    assert summ[0].startswith("FAIL (soft, dropped) s: exit 1")
    assert summ[1] == "SKIPPED dep: needs s (failed or skipped)" and summ[2].startswith("PASS other")


def test_a_hard_background_failure_stops_the_sequence_at_its_join(tmp_path):
    flag = tmp_path / "after"
    p, summ = _kit(tmp_path, f"step h --bg --cap 30 -- bash -c 'exit 5'\nstep fg --cap 30 -- true\nstep_join h\n"
                             f"step never --cap 30 -- touch {flag}")
    assert p.returncode == 5 and not flag.exists()
    assert summ[0].startswith("PASS fg") and summ[1].startswith("FAIL h: exit 5") and len(summ) == 2


def test_kit_summary_joins_and_the_exit_trap_kills_what_still_runs(tmp_path):
    p, summ = _kit(tmp_path, "step late --bg --cap 30 -- sleep 1")
    assert p.returncode == 0 and summ[0].startswith("PASS late")
    flag = tmp_path / "x" / "bg-finished"
    stub = tmp_path / "push.sh"
    stub.write_text("exit 0\n")
    t0 = time.time()
    p, summ = _kit(tmp_path / "x", f"kit_trap_unlock --push\nstep long --bg --cap 60 -- bash -c 'sleep 8; touch {flag}'\n"
                                   "step boom --cap 30 -- bash -c 'exit 4'", KIT_PUSH_SCRIPT=str(stub))
    assert p.returncode == 4 and "killed background step long" in p.stderr
    assert time.time() - t0 < 7
    time.sleep(1.5)
    assert not flag.exists()


def test_a_background_step_past_its_cap_is_a_timeout(tmp_path):
    p, summ = _kit(tmp_path, "step hang --bg --soft --cap 1 -- sleep 30\nstep_join hang")
    assert p.returncode == 0 and summ[0].startswith("FAIL (soft, dropped) hang: exit 124: TIMEOUT")


# ---- main.sh --profile blind ----------------------------------------------------------------------------------

def _dry(tmp_path, *extra, py="python3"):
    env = dict(os.environ, PY_VLLM=py, HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    return subprocess.run(["bash", str(PROD / "main.sh"), "--dry-run", *extra], capture_output=True, text=True,
                          env=env, cwd=str(REPO), timeout=120)


def _steps(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith("STEP ") or x.startswith("JOIN ")]


def test_blind_profile_runs_only_the_lean_plan_with_overlap(tmp_path):
    p = _dry(tmp_path, "--profile", "blind", "--model", "qwen3-30b-a3b", "--utc", "20261121T010203Z")
    assert p.returncode == 0, p.stderr
    lines = _steps(p.stdout)
    steps = [x for x in lines if x.startswith("STEP ")]
    names = [x.split()[1] for x in steps]
    by = dict(zip(names, steps, strict=True))
    order = [x.split()[0] + " " + x.split()[1] for x in lines]
    # dropped: the secondary runs, replay C, the replay offsets, the extras, the partial / hold-out Frontier arms
    for n in names:
        assert not re.match(r"(Adefault|Aon|Beager|replay|graphmode|X\d|frontier-holdout|frontier-.*partial)", n), n
    for want in ("price-check", "price-check-v9", "model-prefetch", "prompts", "frontier-setup", "sag1410",
                 "gate60-capture", "frontier-timer-probe", "A1710-capture", "A1410-capture", "A1710-extract",
                 "A1410-extract", "frontier-timer-decode", "frontier-timer-prefill", "frontier-points", "frontier-csv",
                 "frontier-train", "frontier-simpath", "frontier-csv-fills", "frontier-train-fills", "frontier-predict",
                 "score", "push-final", "held-back-check"):
        assert want in names, want
    assert "--pricer r3_timing_model_v9" in by["price-check-v9"] and "--pricer r3_timing_model_v8" in by["price-check"]
    bg = {n for n, s in by.items() if " bg " in s}
    assert bg == {"price-check", "price-check-v9", "model-prefetch", "prompts", "frontier-setup", "A1710-extract",
                  "A1410-extract", "push-primary", "frontier-train", "frontier-simpath", "frontier-train-fills"}
    # GPU-timed steps are foreground and never inside a background window of another GPU step
    gpu = ("sag1410", "gate60-capture", "frontier-timer-probe", "A1710-capture", "A1410-capture",
           "frontier-timer-decode", "frontier-timer-prefill")
    assert not any(g in bg for g in gpu)
    assert order.index("STEP sag1410") < order.index("JOIN model-prefetch") < order.index("STEP gate60-capture")
    for s in ("model-prefetch", "prompts", "frontier-setup"):        # downloads + venv run DURING the sag par
        assert order.index(f"STEP {s}") < order.index("STEP sag1410")
    assert order.index("STEP A1710-extract") < order.index("STEP A1410-capture") < order.index("JOIN A1710-extract")
    assert order.index("STEP A1410-extract") < order.index("STEP frontier-timer-decode") < order.index("JOIN A1410-extract")
    for t in ("frontier-train", "frontier-simpath", "frontier-train-fills"):
        assert order.index(f"STEP {t}") < order.index("JOIN frontier-train") < order.index("STEP frontier-predict")
    # G-COV per branch; the model everywhere; its own CSV; predictions on the two runs only
    assert "--cov-branch decode" in by["A1710-capture"] and "--cov-branch prefill" in by["A1410-capture"]
    assert "--model qwen3-30b-a3b" in by["A1710-capture"] and "ernie" not in p.stdout.lower()
    assert "qwen3-30b-a3b-moe/moe_kernel_only.csv" in by["frontier-csv"]
    assert "--calls A1710=" in by["frontier-predict"] and "--calls A1410=" in by["frontier-predict"]
    assert "Adefault" not in by["frontier-predict"]
    for arm in ("cli", "simpath", "cli-fills"):
        assert f"--models {arm}=" in by["frontier-predict"]
    # the spec's levels when it names them (Qwen3 has none: the default), per-instance names
    res = f"{tmp_path}/home/moe/results/prod-qwen3-30b-a3b-20261121T010203Z"
    assert f"--results {res} " in by["score"]
    assert "run-gh200-blind-qwen3-30b-a3b-20261121t010203z" in p.stdout
    # scored on the VM: v8 and v9 side by side, nproc workers, the scorer's python
    sc = by["score"]
    assert "offline.sh" in sc and "--also-price v9=r3_timing_model_v9" in sc and "PY_SCORE=" in sc
    assert re.search(r"--workers \d+ ", sc) and int(re.search(r"--workers (\d+) ", sc).group(1)) <= 32
    assert not _soft_of(by["held-back-check"]) and _soft_of(by["score"])
    assert not (tmp_path / "home").exists() and not (tmp_path / "moe").exists()


def _soft_of(step: str) -> bool:
    return step.split()[2] == "[soft]"


def test_blind_profile_refuses_the_opt_in_arms_and_an_unknown_model_or_profile(tmp_path):
    for extra in (("--graphmode-overhead",), ("--frontier-balanced-arm",)):
        p = _dry(tmp_path, "--profile", "blind", *extra)
        assert p.returncode == 2 and "opt-in" in p.stderr
    p = _dry(tmp_path, "--profile", "fast")
    assert p.returncode == 2 and "full|blind" in p.stderr
    p = _dry(tmp_path, "--profile", "blind", "--model", "no-such-moe")
    assert p.returncode == 2 and "no-such-moe" in p.stderr


def test_blind_profile_resumes_from_a_step_and_the_full_profile_gates_each_branch(tmp_path):
    args = ("--profile", "blind", "--results", str(tmp_path / "res"), "--from", "A1410-capture")
    p = _dry(tmp_path, *args)
    assert p.returncode == 2 and "skips prompts, but its output is missing" in p.stderr   # a bg step's output counts
    res, tr = tmp_path / "res", tmp_path / "moe" / "traces" / "res"
    for f in (res / "prompts.jsonl", tmp_path / "moe" / "frontier" / "frontier.env", res / "sag" / "par.json",
              tr / "gate60" / "serve.nsys-rep", res / "gate60" / "manifest.json", res / "gate60" / "calls.jsonl",
              res / "frontier-probe" / "predictions" / "index.json", tr / "A1710" / "serve.nsys-rep",
              res / "A1710" / "manifest.json", res / "A1710" / "calls.jsonl"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    p = _dry(tmp_path, *args)
    assert p.returncode == 0, p.stderr
    out = p.stdout.splitlines()
    for s in ("price-check-v9", "model-prefetch", "sag1410", "A1710-capture", "A1710-extract"):
        assert f"SKIPPED-RESUME {s}" in out, s
    # the A1410 capture is re-entered relocked (the lock is --always); its background extract follows
    names = [x.split()[1] for x in out if x.startswith("STEP ")]
    assert names[names.index("A1410-capture") - 1] == "A1410-lock" and "A1410-extract" in names
    full = _dry(tmp_path / "f", "--results", str(tmp_path / "f" / "res"))
    by = {x.split()[1]: x for x in _steps(full.stdout) if x.startswith("STEP ")}
    assert "--cov-branch decode" in by["A1710-capture"] and "--cov-branch prefill" in by["A1410-capture"]
    assert "--cov-branch decode" in by["Aon-capture"] and "--cov-branch all" in by["Adefault-capture"]
    assert "--model ernie-4.5-21b-a3b" in by["A1710-capture"]


def test_the_run_names_are_per_instance_and_the_branch_passes_vm_runs_rule():
    n = PL.run_names("qwen3-30b-a3b", "20261121T010203Z")
    assert n["results"] == "prod-qwen3-30b-a3b-20261121T010203Z"
    assert n["branch"] == "run-gh200-blind-qwen3-30b-a3b-20261121t010203z"
    assert re.fullmatch(r"run-[a-z0-9][a-z0-9.-]*", n["branch"])          # scripts/vm_run.sh prepare's rule
    for bad in ("", "2026-11-21", "a/b"):
        with pytest.raises(PL.KitError):
            PL.run_names("qwen3-30b-a3b", bad)
    with pytest.raises(PL.KitError):
        PL.run_names("no-such-moe", "20261121T010203Z")
    r = subprocess.run([sys.executable, str(PROD / "prodlib.py"), "run-names", "--model", "qwen3-30b-a3b", "--utc",
                        "20261121T010203Z"], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 0 and "RUN_BRANCH=run-gh200-blind-qwen3-30b-a3b-20261121t010203z" in r.stdout


# ---- G-COV per branch -----------------------------------------------------------------------------------------

def test_gcov_gates_only_the_branch_of_the_run():
    """2026-11-20: the A1710 capture stopped on 25 steps of the PREFILL G16 tile; the 1710 run is gated on the decode
    tiles only, the 1410 run on the prefill tiles only."""
    assert PL.branch_tiles(64, "decode") == ("16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3")
    assert PL.branch_tiles(64, "prefill") == ("64x128x64w8s3", "128x128x64w8s3", "128x128x64w8s3 g16")
    assert PL.branch_tiles(128, "prefill", 16384) == ("64x128x64w8s3", "128x128x64w8s3")
    realized = {"16x64x128w4s4": 400, "32x64x128w4s3": 90, "32x128x64w4s3": 60, "64x128x64w4s3": 50,
                "64x128x64w8s3": 4, "128x128x64w8s3": 3, "128x128x64w8s3 g16": 0}
    dec = S.coverage_gate(S.branch_view(realized, 64, "decode"), 30)
    assert dec["verdict"] == "PASS" and set(dec["per_tile"]) == set(PL.branch_tiles(64, "decode"))
    assert S.coverage_gate(S.branch_view(realized, 64, "prefill"), 30)["verdict"] == "FAIL"
    assert S.coverage_gate(S.branch_view(realized, 64, "all"), 30)["verdict"] == "FAIL"
    realized["64x128x64w4s3"] = 5
    assert S.coverage_gate(S.branch_view(realized, 64, "decode"), 30)["verdict"] == "FAIL"
    a = S.build_parser().parse_args(["--model", "ernie-4.5-21b-a3b", "--prompts", "p", "--out", "o"])
    assert a.cov_branch == "all"


# ---- Frontier: stale models ------------------------------------------------------------------------------------

def test_frontier_train_clears_stale_models_before_training(tmp_path, monkeypatch):
    """2026-11-20: X3's predict found 3 moe_grouped_gemm_*.pkl (want 1) in a dir an earlier training had filled."""
    out = tmp_path / "models"
    out.mkdir()
    for n in ("moe_grouped_gemm_a.pkl", "moe_grouped_gemm_b.pkl", "other.pkl", "versions.json"):
        (out / n).write_text("old")
    (out / "keep.csv").write_text("x")
    stub = tmp_path / "fr_py"
    stub.write_text("#!/usr/bin/env bash\n"
                    "if [[ \"$1\" == -c ]]; then echo '{\"sklearn\": \"0\"}'; exit 0; fi\n"
                    "while [[ $# -gt 0 ]]; do [[ \"$1\" == --out ]] && o=\"$2\"; shift; done\n"
                    "ls \"$o\"/*.pkl >/dev/null 2>&1 && { echo stale; exit 7; }\n"
                    "touch \"$o/moe_grouped_gemm_new.pkl\"\n")
    stub.chmod(0o755)
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setenv("FR_PY_PRIMARY", str(stub))
    rc = FO.main(["train", "--model", "ernie-4.5-21b-a3b", "--src", str(src), "--csv", str(tmp_path / "x.csv"),
                  "--out", str(out)])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["keep.csv", "moe_grouped_gemm_new.pkl", "versions.json"]
    assert FO.clear_models(tmp_path / "absent-is-fine") == []


# ---- the blind registry ----------------------------------------------------------------------------------------

@pytest.fixture()
def registry(monkeypatch):
    monkeypatch.setattr(PL, "MODELS", dict(PL.MODELS))
    monkeypatch.setattr(PL, "HF_REVISIONS", dict(PL.HF_REVISIONS))
    monkeypatch.setattr(PL, "BLIND_SPECS", {"acme-moe-16b-a2b": dict(SPEC)})
    monkeypatch.setattr(FO, "FRONTIER_MODELS", dict(FO.FRONTIER_MODELS))
    monkeypatch.setattr(FO, "PROBE_POINTS", dict(FO.PROBE_POINTS))
    assert PL.register_specs(PL.BLIND_SPECS, {}) == ["acme-moe-16b-a2b"]
    FO._blind_entries()
    return "acme-moe-16b-a2b"


def test_one_dict_entry_registers_a_blind_model_everywhere(registry):
    m = PL.MODELS[registry]
    c = m.cfg
    assert (c.hidden_size, c.intermediate_size, c.num_experts, c.top_k, c.num_layers, c.first_moe_layer) == \
        (2048, 1024, 64, 6, 28, 1)
    assert m.moe_layers == 27 and m.hf_repo == "acme/MoE-16B-A2B" and c.name == registry
    assert PL.HF_REVISIONS[registry] is None and PL.hf_revision(registry) is None      # unpinned until filled
    hf = {"hidden_size": 2048, "moe_intermediate_size": 1024, "num_experts": 64, "num_experts_per_tok": 6,
          "num_hidden_layers": 28}
    assert PL.check_hf_config(m, hf) == [] and PL.check_hf_config(m, dict(hf, num_experts=128))
    name, doc = FO.frontier_config(registry)                                       # its asserts hold
    assert name == "acme-moe-16b-a2b-moe" and doc["num_attention_heads"] == 16 and doc["num_experts"] == 64
    assert str(FO.csv_relpath(registry)) == PL.frontier_csv_rel(registry)
    assert FO.PROBE_POINTS[registry] == PL.DEFAULT_PROBE_POINTS
    info = PL.shell_model_info(registry)
    assert "TOK=acme/MoE-16B-A2B" in info and "SPEC_LEVELS=1,8,24,48,80,112,256" in info
    assert f"CSVREL=compute/gh200/{name}/moe_kernel_only.csv" in info
    import hf_revisions as HR
    assert "acme/MoE-16B-A2B" in HR.tokenizer_repos()
    # every kit model's Frontier CSV path agrees between prodlib and frontier_op
    for k in PL.MODELS:
        assert str(FO.csv_relpath(k)) == PL.frontier_csv_rel(k), k


def test_the_registry_refuses_a_missing_field_and_a_redefinition(registry):
    with pytest.raises(PL.KitError, match="missing"):
        PL.served_from_spec("x", {"hf_repo": "a/b"})
    with pytest.raises(PL.KitError, match="already"):
        PL.register_specs({registry: dict(SPEC)}, {})
    with pytest.raises(PL.KitError, match="already"):
        PL.register_specs({"ernie-4.5-21b-a3b": dict(SPEC)}, {})
    assert PL.BLIND_REVISIONS_PENDING == {} or all(re.fullmatch(r"[0-9a-f]{40}", v or "")
                                                   for v in PL.BLIND_REVISIONS_PENDING.values())
    with pytest.raises(PL.KitError, match="not pinned"):
        PL.prefetch(registry)


def test_a_registered_blind_model_prices_through_v9(registry):
    """The contract a blind model needs at price-check: v9 (and the v8 it wraps) prices a real decode and a real
    prefill call of a model registered only by its spec."""
    pytest.importorskip("r3_timing_model_v9")
    got = PL.price_check(registry, module="r3_timing_model_v9")
    assert got[8]["regime"] == "decode" and got[1024]["regime"] == "prefill" and got[1024]["ms"] > got[8]["ms"] > 0


# ---- expected outputs and the score ---------------------------------------------------------------------------

def _tree(R: Path, profile: str | None, model: str = "ernie-4.5-21b-a3b"):
    R.mkdir(parents=True)
    if profile:
        (R / PL.RUN_PROFILE_FILE).write_text(json.dumps({"profile": profile, "model": model}))
    (R / "SUMMARY.txt").write_text("PASS A1710-extract (1s)\nPASS A1410-extract (1s)\nPASS frontier-csv (1s)\n")
    for lab in ("A1710", "A1410"):
        (R / lab).mkdir()
        (R / lab / "calls.jsonl").write_text("{}\n")
    (R / "sag").mkdir()
    (R / "sag" / "par.json").write_text("{}")


def test_expected_outputs_do_not_require_the_runs_the_blind_profile_drops(tmp_path):
    _tree(tmp_path / "b", "blind", "qwen3-30b-a3b")
    got = PL.expected_outputs(tmp_path / "b")
    assert got["runs"] == ["A1710", "A1410"]
    assert len(got["missing"]) == 1 and "qwen3-30b-a3b-moe/moe_kernel_only.csv" in got["missing"][0]
    csv = tmp_path / "b" / "frontier" / "profile" / PL.frontier_csv_rel("qwen3-30b-a3b")
    csv.parent.mkdir(parents=True)
    csv.write_text("x\n")
    assert PL.expected_outputs(tmp_path / "b")["missing"] == []
    # the full profile (no kit-profile.json: every earlier run) still requires both replays
    _tree(tmp_path / "f", None)
    miss = PL.expected_outputs(tmp_path / "f")["missing"]
    assert sum("replay-" in m for m in miss) == 2
    assert PL.expected_outputs(tmp_path / "f", profile="blind")["missing"] == [m for m in miss if "replay-" not in m]


def test_offline_on_a_blind_tree_skips_decompress_and_passes_no_replay(tmp_path):
    R = tmp_path / "b"
    _tree(R, "blind")
    (R / "SUMMARY.txt").write_text("PASS A1710-extract (1s)\n")
    (R / "A1410" / "calls.jsonl").unlink()
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--results", str(R)], capture_output=True, text=True,
                       env=env, cwd=str(REPO), timeout=300)
    assert p.returncode == 3 and "run A1410: no calls" in p.stdout
    summ = (R / "offline-logs" / "SUMMARY.txt").read_text().splitlines()
    assert summ[0].startswith("FAIL expected-outputs")                         # no .xz: no decompress step
    d = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--results", str(R), "--pricer", "mod_a",
                        "--also-price", "none"], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    sc = [x for x in d.stdout.splitlines() if x.startswith("STEP score ")][0]
    assert "--pricer mod_a " in sc and "--also-price" not in sc
    d = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--results", str(R)], capture_output=True,
                       text=True, env=env, cwd=str(REPO), timeout=60)
    sc = [x for x in d.stdout.splitlines() if x.startswith("STEP score ")][0]
    assert "--pricer r3_timing_model_v8 " in sc and "--also-price v9=r3_timing_model_v9 " in sc


def test_also_price_scores_v9_beside_v8_in_the_same_tables(tmp_path, monkeypatch):
    import textwrap

    import nsys_extract as NX
    import prod_fakes as F
    import score_prod as SP
    d = tmp_path / "mods"
    d.mkdir()
    (d / "fake_v8.py").write_text(textwrap.dedent('''
        def price(call):
            return {"ms": 0.05, "tile": "t", "regime": "decode" if call["T"] <= 128 else "prefill"}
    '''))
    (d / "fake_v9.py").write_text(textwrap.dedent('''
        def price(call):
            if call["T"] > 128:
                return {"ms": None, "tile": "t", "regime": "prefill", "why": "v9 refuses"}
            return {"ms": 0.04, "tile": "t", "regime": "decode"}
    '''))
    monkeypatch.syspath_prepend(str(d))
    model = PL.MODELS["olmoe-1b-7b"]
    run, calls = F.make_run(tmp_path / "A1710", steps=((5, 8), (40, 40), (300, 304)))
    ev, marks, _x = F.timeline(model.cfg, calls)
    F.write_clocks(run, len(calls) // 4 + 2, mhz=1710.0)
    db = F.write_sqlite(tmp_path / "a.sqlite", ev, marks)
    assert NX.write(NX.extract(run, NX.read_sqlite(db), model), run, "") == 0
    par = tmp_path / "par.json"
    par.write_text(json.dumps({"schema": PL.PAR_SCHEMA, "verdict": "PASS", "par": [64.3, 612.5]}))
    out = tmp_path / "score"
    rc = SP.main(["--model", model.key, "--run", f"A1710={run}", "--pricer", "fake_v8", "--also-price", "v9=fake_v9",
                  "--workers", "1", "--sag", str(par), "--out", str(out)])
    assert rc == 0
    sc = json.loads((out / "prod.score.json").read_text())
    t8 = sc["runs"]["A1710"]["tables"]["v8_ms vs kernel_sum_ms"]["all"]
    t9 = sc["runs"]["A1710"]["tables"]["v9_ms vs kernel_sum_ms"]["all"]
    assert t8["n"] == t9["n"] > 0 and t8["MAPE"] != t9["MAPE"]
    assert sc["also_priced"] == {"v9": "fake_v9"} and sc["pricing"]["errors"] == 0
    txt = (out / "prod.score.txt").read_text()
    assert "v9_ms vs kernel_sum_ms [primary]" in txt and "v8_ms vs kernel_sum_ms [primary]" in txt


def test_blind_sh_syntax_and_it_is_only_sourced():
    assert subprocess.run(["bash", "-n", str(PROD / "blind.sh")]).returncode == 0
    assert not (PROD / "blind.sh").read_text().startswith("#!")            # sourced by main.sh, never run alone
    assert '. "$HERE/blind.sh"' in (PROD / "main.sh").read_text()
