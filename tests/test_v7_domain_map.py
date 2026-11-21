"""Model v7's decode domain map (docs/registered/2026-11-18-v7-domain-map-gh200), off the GPU and on SYNTHETIC trees
only: the registration, its 14 pages and its plan recompute to their committed files; the map shapes are synthetic,
collide with no real model and stay out of moe/spec.py (frozen); the VM hook adds them for the pinned decode driver's
processes; the pinned driver's dry run lists every unit; the launcher is vm_run_v7b.sh plus lines marked V7M, delivers
the plan and pages on a stub VM with the hook on the driver's PYTHONPATH, and refuses a DRAFT plan; the rule is
registered verbatim and computed mechanically, including on adversarial trees (a path that never qualifies, shapes that
move X_p under leave-one-shape-out, a significant slab slope, off-lock cells, a failed power check)."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode",
          REPO / "scripts" / "scoring" / "v7blind", REPO / "scripts" / "scoring" / "v7domain"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_fakes as F  # noqa: E402
import map_shapes as MS  # noqa: E402
import read_map as RM  # noqa: E402
import register_v7_domain as RG  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_rr_driver as TRR  # noqa: E402
import v7mcommon as M  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

REGF = REPO / M.REG_JSON
PLAN = REPO / M.PLAN
DRIVER = REPO / "scripts" / M.DRIVER
LAUNCHER = REPO / M.LAUNCHER
PARENT = REPO / M.PARENT_LAUNCHER
VMSITE = REPO / M.VMSITE
PY = sys.executable
SPEC_SHA = "58fb9b7fe2fdbd82867e0ab789cfc02bc03705c01722f9cfa0769c40bf6f87de"


def _reg() -> dict:
    return json.loads(REGF.read_text())


@pytest.fixture(scope="module")
def pred():
    return M.predictions()


# ---------------------------------------------------------------- the registration, pages and plan

def test_the_registration_pages_and_plan_recompute_to_their_committed_files(pred):
    reg = _reg()
    for p, t in pred["pages"].items():
        assert (REPO / p).read_text() == t, p
    doc = json.loads(json.dumps(RG.build(pred), default=str))
    assert RG.close(doc, reg) == []
    assert (REPO / M.REG_TXT).read_text() == "\n".join(RG.text(doc)) + "\n"
    assert PLAN.read_text() == M.plan_text(pred["checks"], pred["frozen"])
    assert reg["rental"]["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert M.verify_freeze(reg) == [] and RM.refusals(reg) == []
    assert reg["status"].startswith("REGISTERED BEFORE ANY PAGE") and "draft" not in reg
    assert "# DRAFT" not in PLAN.read_text()


def test_the_map_is_phase_0s_design_and_v7_prices_every_cell(pred):
    cells = pred["cells"]
    assert len(cells) == 130 and len(pred["pages"]) == 14
    by = lambda k: {x: sum(1 for c in cells if k(c) == x) for x in {k(c) for c in cells}}  # noqa: E731
    assert by(lambda c: c["path"]) == {"naive": 16, "small-batch": 75, "block-scan": 39}
    assert by(lambda c: c["family"]) == {"A": 48, "B": 52, "C": 30}
    assert by(lambda c: c["tile"]) == {"16x64x128w4s4": 86, "32x64x128w4s3": 16, "32x128x64w4s3": 12,
                                       "64x128x64w4s3": 16}
    assert all(c["in_scope"] for c in cells) and all(c["pred_ms"] > 0 for c in cells)
    assert (sum(c["pred_us"] < 60 for c in cells), sum(60 <= c["pred_us"] < 200 for c in cells),
            sum(c["pred_us"] >= 200 for c in cells)) == (55, 40, 35)
    assert {round(c["W_e_MB"], 2) for c in cells if c["family"] == "B"} == {2.36, 4.72, 9.44, 17.69, 28.31, 37.75}
    # Phase 0's spot values (out/map.txt), to 0.05 us
    spot = {("map-a-e8k2-h2048i512", "L-T1-A2"): 15.8, ("map-b-e32k4-h2560i1152", "L-T8-A32"): 174.9,
            ("map-c-e256k8-h4096i1024", "L-T128-A256"): 1823.7, ("map-a-e256k8-h2048i512", "MIN-T16"): 39.8}
    got = {(c["model"], c["label"]): c["pred_us"] for c in cells}
    for k, v in spot.items():
        assert abs(got[k] - v) <= 0.05, (k, got[k])
    for m, ck in pred["checks"].items():
        assert ck["page_from"] == "PASSES" and ck["weights_GB"] < 13
    fz = pred["frozen"]
    assert fz["model"]["switches"] == {"RR": True, "D6FIX": True}
    assert fz["model"]["constants_sha256"] == M.V7.constants_sha256() == \
        M.VB.gate_registration()["freeze"]["constants_sha256"]
    assert round(fz["constants"]["T0_GR_ms"] * 1e3, 2) == -1.87 and fz["rho_form"] == "fixed"


def test_the_rule_is_phase_0s_section_6_verbatim_and_its_points_fix_every_open_choice():
    reg = _reg()
    t = reg["rule"]["text"]
    for s in ("For each decode path p (naive, small-batch, block-scan):", "e = dcommon.e_dec(T_v7, T_meas)",
              "T_meas ∓ 2.306 × 1.2533 × s/√n", "holds at least 10 cells has:",
              "(i) every e_ub ≤ 6%;", "(ii) rms e ≤ 2.5%;", "(iii) |mean e| ≤ 1.0%;",
              "(iv) X_p is the largest of the values this rule gives with each synthetic shape's cells removed in turn",
              "A path with no such X is outside the domain.",
              "D = {decode calls whose path has an X_p, whose T_v7 ≥ X_p, and whose every active expert fits one "
              "M-tile}.", "fit resid_us = a + b·W_e_MB + c·1[T=2]",
              "If b's 95% interval excludes 0 and |b|·35 MB ≥ 1 us, D adds W_e ≥ S, with S chosen by "
              "the same rule; otherwise D has no slab condition."):
        assert s in t, s
    c = reg["rule"]["constants"]
    assert (c["e_ub_max"], c["rms_max"], c["mean_max"], c["min_cells"]) == (0.06, 0.025, 0.010, 10)
    assert (c["t975_8"], c["median_se"], c["slab_span_MB"], c["slab_min_us"]) == (2.306, 1.2533, 35.0, 1.0)
    pts = " ".join(reg["rule"]["points (how read_map.py computes it; fixed before any page)"])
    for s in ("power check", "row_drop", "clean_r7", "no 1 us floor", "neither tested nor eligible",
              "a path where any of them is none has no X_p", "left out", "NOT FITTED", "W_e >= S_p"):
        assert s in pts, s
    assert "Phase 2" in reg["the domain registration this feeds"] and "NOT REGISTERED HERE" in \
        reg["the domain registration this feeds"]
    for m in ("DeepSeek-V3, DeepSeek-V3-tp4, DeepSeek-V3-tp8", "Qwen2-57B-A14B-tp8", "rms 3%, worst 8%, |mean| 1.5%"):
        assert m in reg["the domain registration this feeds"], m


def test_the_registration_discloses_the_reopening_the_post_hoc_origin_the_offline_pricing_and_cost():
    reg = _reg()
    t = " ".join(reg["disclosures"])
    for s in ("second reopening of decode (owner, 2026-11-18)", "POST-HOC", "v7-domain/REPORT.md",
              "domain-phase0/REPORT.md", "Phase 0 priced these map shapes offline before this registration",
              "$2.29/h", "$2.59/h", "not in moe/spec.py"):
        assert s in t, s
    r = reg["rental"]
    assert (r["units_min"], r["wall_min"], r["deadline_after_launch_min"], r["hard_cap_after_launch_min"]) == \
        (76, 115, 130, 145)
    assert (r["cost_usd"], r["cost_usd_at_list"], r["cost_at_hard_cap_usd"], r["cost_at_hard_cap_usd_at_list"]) == \
        (4.39, 4.96, 5.53, 6.26)
    assert reg["label"] == "DESCRIPTIVE" and reg["model v7 (frozen)"]["switches"] == {"RR": True, "D6FIX": True}


# ---------------------------------------------------------------- the shapes and the VM hook

def test_the_map_shapes_are_synthetic_collide_with_nothing_and_stay_out_of_moe_spec():
    import moe.spec as S
    import v7bcommon as VB
    assert hashlib.sha256((REPO / "moe" / "spec.py").read_bytes()).hexdigest() == SPEC_SHA == \
        M.VB.gate_registration()["freeze"]["files"]["moe/spec.py"]
    assert len(MS.NAMES) == len(set(MS.NAMES)) == 14
    real = set(S.MODEL_CONFIGS)
    repos = {c.hf_repo.lower() for c in S.MODEL_CONFIGS.values() if c.hf_repo}
    for n in MS.NAMES:
        assert n.startswith("map-") and n not in real and re.fullmatch(r"map-[abc]-e\d+k\d+-h\d+i\d+", n)
        assert not VB.PAT.search(n) and not any(n in r or r.split("/")[-1] in n for r in repos)
        assert not any(n in x or x in n for x in real)
    # Phase 0's construction: granite-3.0-1b-a400m's config with the geometry replaced, hf_repo None, verified True
    base = S.MODEL_CONFIGS["granite-3.0-1b-a400m"]
    for n, c in MS.configs(S).items():
        want = dataclasses.replace(base, name=n, num_experts=c.num_experts, top_k=c.top_k, hidden_size=c.hidden_size,
                                   intermediate_size=c.intermediate_size, hf_repo=None, verified=True)
        assert c == want
    before = dict(S.MODEL_CONFIGS)
    with MS.installed():
        assert set(S.MODEL_CONFIGS) == real | set(MS.NAMES)
    assert S.MODEL_CONFIGS == before
    S.MODEL_CONFIGS["map-a-e8k2-h2048i512"] = base
    try:
        with pytest.raises(ValueError):
            MS.install(S)
    finally:
        del S.MODEL_CONFIGS["map-a-e8k2-h2048i512"]
    assert S.MODEL_CONFIGS == before


def _py(code: str, hook: bool = True):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if hook:
        env["PYTHONPATH"] = str(VMSITE)
    return subprocess.run([PY, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_the_vm_hook_adds_the_shapes_to_moe_spec_and_the_pinned_r3_decode_takes_a_map_page():
    got = _py("import moe.spec as S, json; print(json.dumps(sorted(m for m in S.MODEL_CONFIGS if m.startswith('map-'))))")
    assert got.returncode == 0, got.stderr
    assert json.loads(got.stdout) == sorted(MS.NAMES)
    off = _py("import moe.spec as S; print(sum(m.startswith('map-') for m in S.MODEL_CONFIGS))", hook=False)
    assert off.stdout.strip() == "0"
    m = "map-c-e256k8-h4096i1024"
    args = ["scripts/r3_decode.py", "--model", m, "--histogram", M.page_path(m), "--inst", "GRR", "--rot", "auto",
            "--repeats", "9", "--duty", "0.25", "--seed", "0", "--group-m", "1", "--dry-run"]
    got = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": str(VMSITE), "PYTHONDONTWRITEBYTECODE": "1",
                              "MOE_RESULTS_DIR": "/nonexistent/r"})
    assert got.returncode == exit_codes.REFUSED and got.stdout.startswith("DECODE PAGE"), got.stdout + got.stderr
    assert "REFUSED:" not in got.stdout and "15 cells" in got.stdout
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    bad = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**env, "PYTHONDONTWRITEBYTECODE": "1"})
    assert bad.returncode != 0 and "invalid choice" in bad.stderr


# ---------------------------------------------------------------- the plan, the driver and the launcher

def test_the_plan_runs_every_page_nodrop_and_the_pinned_driver_lists_every_unit():
    reg = _reg()
    units = [ln for ln in PLAN.read_text().splitlines() if ln and not ln.startswith("#")]
    assert units[:2] == list(M.D4.PRELUDE) and len(units) == 16
    pages = re.findall(r"histogram=(\S+)", PLAN.read_text())
    assert pages == [M.page_path(m) for m in MS.NAMES]
    assert all("nodrop=1" in u and "inst=GRR" in u and "slip-policy=cell" in u for u in units[2:])
    for p in pages:
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == reg["freeze"]["own"][p] == \
            reg["cells"]["pages"][Path(p).name[:-len("-main.json")]]["sha256"]
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN), "--deadline", "2000000000"],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": "/nonexistent"},
                         cwd=REPO)
    assert got.returncode == 2, got.stdout + got.stderr
    assert "a PLAN of 16 units" in got.stdout and "76 min of units" in got.stdout
    assert got.stdout.count(" r3decode ") == 14


def test_the_launcher_is_v7bs_plus_lines_marked_v7m():
    changed = TRR._held_to_parent(PARENT, LAUNCHER, None, mark="V7M")
    assert len(changed) == 10, changed
    t = LAUNCHER.read_text()
    assert re.search(r"^SESSION_DRIVERS=\(([^)]*)\)", t, re.M).group(1).split() == ["gh200_decode_session.sh"]
    assert "V7M_ENV='PYTHONPATH=$HOME/moe/repo/scripts/scoring/v7domain/vmsite '" in t
    assert sum('vm "${V7M_ENV}nohup setsid bash $drv --setup' in x for x in changed) == 1
    assert sum("^# DRAFT" in x and "refuse" in x for x in changed) == 1
    assert (LAUNCHER.stat().st_mode & 0o777) == (PARENT.stat().st_mode & 0o777)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == M.VB.freeze()["own"][M.PARENT_LAUNCHER] == \
        _reg()["freeze"]["own"][M.PARENT_LAUNCHER]


def _laptop(tmp_path, plan_text: str):
    pages = sorted(set(re.findall(r"histogram=(\S+)", plan_text)))
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    (lap.clone / M.PLAN).write_text(plan_text)
    for f in (DRIVER, LAUNCHER):
        shutil.copy(f, lap.clone / "scripts" / f.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    env = {**os.environ, "MOE_VM_SSH": str(lap.bin / "ssh"), "MOE_VM_SCP": str(lap.bin / "scp"),
           "MOE_VM_GH": str(lap.bin / "gh"), "MOE_VM_STATE_ROOT": str(lap.state), "MOE_VM_POLL_S": "0",
           "MOE_VM_WAIT_S": "5", "FAKE_VM": str(lap.vm), "FAKE_GH": str(lap.gh)}

    def run(*args):
        return subprocess.run(["bash", str(lap.clone / M.LAUNCHER), *args], env=env, capture_output=True, text=True,
                              timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    return lap, run, pages


def test_start_delivers_the_plan_and_every_page_and_runs_the_driver_with_the_hook(tmp_path):
    lap, run, pages = _laptop(tmp_path, PLAN.read_text())
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", M.PLAN)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / DRIVER.name) == sha(DRIVER)
    assert len(pages) == 14
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = [json.dumps(c) for c in lap.vm_cmds()]
    setup = [c for c in cmds if "gh200_decode_session.sh --setup" in c]
    assert len(setup) == 1 and \
        "PYTHONPATH=$HOME/moe/repo/scripts/scoring/v7domain/vmsite nohup setsid bash gh200_decode_session.sh --setup" \
        in setup[0]
    assert any("gh200_decode_session.sh --dry-run" in c for c in cmds)


def test_start_refuses_a_draft_plan_and_starts_nothing(tmp_path):
    draft = PLAN.read_text().replace("# Every page is nodrop.", "# DRAFT: NOT LAUNCHABLE.\n# Every page is nodrop.")
    lap, run, _pages = _laptop(tmp_path, draft)
    got = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", M.PLAN)
    assert got.returncode == exit_codes.REFUSED and "carries a DRAFT line" in got.stdout + got.stderr
    assert "2026-11-18-v7-domain-map-gh200" in got.stdout + got.stderr
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "--dry-run" not in cmds and "--setup" not in cmds


# ---------------------------------------------------------------- the rule, pure

def _cell(model, pred, e, path="small-batch", fam="A", W=6.0, T=8, waves=0.5, ub=None):
    return {"model": model, "pred_us": pred, "e": e, "e_ub": abs(e) + 0.001 if ub is None else ub, "path": path,
            "family": fam, "W_e_MB": W, "T": T, "waves": waves, "in_scope": True, "meas_us": pred / (1 + e)}


def test_the_bound_is_the_registered_formula():
    s, n, meas, pred = 0.4, 9, 40.0, 41.0
    h = 2.306 * 1.2533 * s / 3
    assert RM.e_ub(pred, meas, s, n) == pytest.approx(max(abs(pred / (meas - h) - 1), abs(pred / (meas + h) - 1)))
    assert RM.e_ub(pred, meas, None, 1) == math.inf and RM.e_ub(pred, meas, 0.1, 1) == math.inf
    assert RM.e_ub(10.0, 0.5, 5.0, 9) == math.inf


def test_x_is_the_smallest_counted_candidate_past_which_every_counted_candidate_passes():
    good = [_cell(f"s{i % 3}", 10.0 * (i + 1), 0.0) for i in range(20)]
    assert RM.threshold(good)["X"] == 10.0
    # the two shortest cells read 8%: X moves past them
    bad = [dict(c, e=0.08, e_ub=0.081) if c["pred_us"] <= 20 else c for c in good]
    assert RM.threshold(bad)["X"] == 30.0
    # a failing cell blocks every X at or below it; X is the first counted candidate above it
    top = [dict(c, e=0.07, e_ub=0.071) if c["pred_us"] == 100.0 else c for c in good]
    assert RM.threshold(top)["X"] == 110.0
    # one past the last counted candidate (110 us, the last set of 10 cells) leaves no X
    late = [dict(c, e=0.07, e_ub=0.071) if c["pred_us"] == 120.0 else c for c in good]
    r = RM.threshold(late)
    assert r["candidates"][-1] == [110.0, 10, False] and r["X"] is None
    assert RM.threshold(good[:9]) == {"X": None, "candidates": []}
    # rms and mean: 30% of the cells at +3.4% fails |mean| 1.0%, at 2.0% passes everything
    m = [dict(c, e=0.034, e_ub=0.035) if i % 10 < 3 else c for i, c in enumerate(good)]
    assert not RM.passes(m)
    assert RM.passes([dict(c, e=0.02, e_ub=0.021) if i % 10 < 3 else c for i, c in enumerate(good)])


def test_a_path_that_never_qualifies_is_outside_and_d_names_the_others():
    cells = [_cell(f"s{i % 4}", 20.0 + i, 0.0, path="naive") for i in range(16)]
    cells += [_cell(f"s{i % 4}", 30.0 + i, 0.07, path="block-scan", ub=0.071) for i in range(30)]
    cells += [_cell(f"s{i % 4}", 30.0 + i, 0.0) for i in range(30)]
    d = RM.domain(cells, shapes=["s0", "s1", "s2", "s3"])
    assert d["per_path"]["block-scan"]["X_p_us"] is None and "OUTSIDE" in d["per_path"]["block-scan"]["verdict"]
    # (iv): removing s0 (its cells hold the shortest call) moves each X one candidate up
    assert d["per_path"]["naive"]["X_p_us"] == 21.0 and d["per_path"]["small-batch"]["X_p_us"] == 31.0
    assert d["per_path"]["naive"]["rule"]["full"] == 20.0
    assert len(d["D"]) == 2 and "block-scan" not in d["statement"]
    empty = RM.domain([dict(c, e=0.09, e_ub=0.09) for c in cells], shapes=["s0"])
    assert empty["D"] == [] and empty["statement"].startswith("D is EMPTY")


def test_a_shape_whose_cells_hold_the_set_moves_x_under_leave_one_shape_out():
    # shape Z's cells are good and long; six mid-length cells of the others read +4%: with Z the |mean| bar holds
    # from 30 us, without Z it fails until 33 us
    cells = [_cell("Z", 100.0 + i, 0.0) for i in range(12)]
    cells += [_cell(f"o{i % 2}", 30.0 + i, 0.04, ub=0.041) for i in range(6)]
    cells += [_cell(f"o{i % 2}", 50.0 + i, 0.0) for i in range(10)]
    r = RM.loso(cells, ["Z", "o0", "o1"])
    assert r["full"] == 30.0 and r["folds"]["Z"] == 33.0 and r["X"] == 33.0
    assert "Z" in r["shapes that raise X when removed"] and r["folds"]["o0"] == 31.0
    # a fold with no X (fewer than 10 cells without Z) leaves the path without an X_p
    small = [c for c in cells if c["model"] == "Z"] + [_cell("o0", 50.0 + i, 0.0) for i in range(3)]
    rr = RM.loso(small, ["Z", "o0"])
    assert rr["full"] == 50.0 and rr["folds"]["Z"] is None and rr["X"] is None and rr["none in"] == ["Z"]


def test_the_slab_guard_fits_on_family_b_under_a_wave_and_triggers_only_on_a_real_slope():
    Ws = [2.36, 4.72, 9.44, 17.69, 28.31, 37.75]
    flat = [_cell(f"b{j}", 40.0 + j + i, 0.0, fam="B", W=W, T=8, waves=0.5) for j, W in enumerate(Ws) for i in range(3)]
    g = RM.slab_guard(flat)
    assert g["fit"]["verdict"] == "FITTED" and not g["triggered"] and g["fit"]["dropped"] == ["T2"]
    slope = [dict(c, meas_us=c["pred_us"] - (4.0 - 0.12 * c["W_e_MB"]) + 0.01 * (i % 3)) for i, c in enumerate(flat)]
    g = RM.slab_guard(slope)
    assert g["triggered"] and g["fit"]["b_us_per_MB"] == pytest.approx(-0.12, abs=0.01)
    assert g["fit"]["b_ci95"][1] < 0 and g["fit"]["|b| x 35 MB (us)"] >= 1.0
    # significant but small (|b| x 35 under 1 us): no slab condition
    tiny = [dict(c, meas_us=c["pred_us"] - (0.5 - 0.02 * c["W_e_MB"]) + 0.001 * (i % 3)) for i, c in enumerate(flat)]
    g = RM.slab_guard(tiny)
    assert g["fit"]["interval excludes 0"] and not g["triggered"]
    # cells over a wave and other families are not fitted; one slab only: NOT FITTED
    assert RM.slab_guard([dict(c, waves=1.2) for c in slope])["cells"] == 0
    one = [c for c in slope if c["W_e_MB"] == 2.36]
    assert RM.slab_guard(one)["fit"]["verdict"] == "NOT FITTED" and not RM.slab_guard(one)["triggered"]
    # with T2 cells the indicator is fitted
    t2 = slope + [dict(slope[0], T=2, meas_us=slope[0]["pred_us"] - 6.0)]
    assert "T2" in RM.slab_guard(t2)["fit"]["coef"]


# ---------------------------------------------------------------- the readout on synthetic trees

POWER = REPO / "results/published/2026-11-03-nvidia_gh200_480gb-decodeCAL2-session/session/power-set.txt"


def _page(tree: Path, m: str, truth, slip=()):
    """One map page as the VM writes it (decode_fakes.write_page with FakeBackend's slip)."""
    RD = F.RD
    cfg = F.MODEL_CONFIGS[m]
    page = RD.load_page(REPO / M.page_path(m), cfg, m)
    out = tree / f"gaps-{F.CARD}-{m}-dm" / "private_weight_reference" / "run-dm-GRR"
    out.mkdir(parents=True, exist_ok=True)
    store = RD.Store(out / "cells.csv")
    be = F.FakeBackend(cfg, page, truth=truth, noise=0.0005, seed=7, slip=slip)
    res = RD.run_page(cfg, page, be, inst="GRR", repeats=9, store=store, lock_mhz=1710, min_clean=0,
                      say=lambda s: None)
    store.close()
    args = argparse.Namespace(model=m, dtype="bf16", inst="GRR", duty=0.25, repeats=9, group_m=1, seed=0,
                              session_tag="T-lock1710", min_clean_repeats=0, rot="auto")
    RD.write_report(args, cfg, page, res, out_dir=out, run_id=out.name, card=F.CARD, R=1, calls=1, copies=1,
                    slip_lock=1710, header=["DECODE PAGE (fake)"], seconds=1.0)


def _tree(root: Path, meas_us, slip=None, power: Path = POWER) -> Path:
    """A results tree: every map page with measured time meas_us(cell record) in us."""
    reg = _reg()
    rec = {(c["model"], c["label"]): c for c in reg["predictions"]["cells"]}
    (root / "session").mkdir(parents=True)
    shutil.copy(power, root / "session" / "power-set.txt")
    tree = root / "results"
    with MS.installed():
        for m in MS.NAMES:
            _page(tree, m, lambda c, a, m=m: meas_us(rec[(m, c.label)]) * 1e-3, (slip or {}).get(m, ()))
    return tree


ZSHAPE = "map-b-e32k4-h4096i1536"


def _adversarial(c):
    """block-scan 7% over-priced everywhere; small-batch +3% over-priced at 40 to 150 us except on ZSHAPE."""
    if c["path"] == "block-scan":
        return c["pred_us"] / 1.07
    if c["path"] == "small-batch" and 40 <= c["pred_us"] < 150 and c["model"] != ZSHAPE:
        return c["pred_us"] / 1.03
    return c["pred_us"]


def _slab(c):
    """family B under a wave: v7 over-prices by 4.0 - 0.12 W_e us (small slabs over-priced, as Granite)."""
    if c["family"] == "B" and c["waves"] < 1:
        return c["pred_us"] - (4.0 - 0.12 * c["W_e_MB"])
    return c["pred_us"]


OFFLOCK = {"map-b-e32k4-h1536i512": tuple(("MIN-T4", "native", n) for n in range(1, 6)),
           "map-a-e64k8-h2048i512": tuple(("L-T64-A64", "native", n) for n in range(1, 10))}


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    reg = _reg()
    good = _tree(tmp_path_factory.mktemp("good"), lambda c: c["pred_us"])
    adv = _tree(tmp_path_factory.mktemp("adv"), _adversarial, slip=OFFLOCK)
    slab = _tree(tmp_path_factory.mktemp("slab"), _slab)
    low = tmp_path_factory.mktemp("lowpower")
    (low / "session").mkdir()
    (low / "session" / "power-set.txt").write_text(POWER.read_text().replace("Current Power Limit               : 700.00 W",
                                                                                 "Current Power Limit               : 600.00 W"))
    (low / "results").symlink_to(good)
    return {k: (v, RM.read(v, reg)) for k, v in
            (("good", good), ("adv", adv), ("slab", slab), ("low", low / "results"))}


def test_a_tree_v7_prices_reads_every_cell_and_every_path_in_the_domain(trees):
    _t, r = trees["good"]
    assert r["verdict"] == "READ" and r["cells"] == {"registered": 130, "valid": 130, "valid in scope": 130}
    assert set(r["pages"].values()) == {"counted"}
    d = r["domain"]
    assert all(d["per_path"][p]["verdict"] == "IN THE DOMAIN" for p in M.PATHS) and len(d["D"]) == 3
    assert not d["slab_guard"]["triggered"] and d["slab_guard"]["cells"] == 30
    for p in M.PATHS:
        x = d["per_path"][p]
        assert x["X_p_us"] == max(v for v in [x["rule"]["full"], *x["rule"]["folds"].values()])
    assert all(c["n"] == 9 and c["median_check"] for c in r["rows"])
    s = r["descriptive"]
    assert s["pooled"]["cells"] == 130 and abs(s["pooled"]["rms"]) < 0.002
    assert {x["E"] for x in s["aligned step over-price by E (us)"]} == {8, 16, 32, 64, 128, 256}
    assert len(s["slab sweep"]) == 52


def test_adversarial_tree_block_scan_outside_a_shape_moves_small_batch_and_off_lock_cells_are_not_read(trees):
    _t, r = trees["adv"]
    d = r["domain"]
    assert d["per_path"]["block-scan"]["X_p_us"] is None and "OUTSIDE" in d["per_path"]["block-scan"]["verdict"]
    sb = d["per_path"]["small-batch"]
    assert sb["X_p_us"] is not None and sb["X_p_us"] > sb["rule"]["full"]
    assert ZSHAPE in sb["rule"]["shapes that raise X when removed"]
    assert sb["X_p_us"] == sb["rule"]["folds"][ZSHAPE] >= 100
    assert d["per_path"]["naive"]["verdict"] == "IN THE DOMAIN" and "block-scan" not in d["statement"]
    # five off-lock repeats of one cell: that cell is not read; nine of one cell on another page: G1 voids the page
    bad = {(c["model"], c["label"]): c["why"] for c in r["rows"] if not c["valid"]}
    assert bad.pop(("map-b-e32k4-h1536i512", "MIN-T4")) == "4 clean repeats, under 6"
    assert {m for m, _lab in bad} == {"map-a-e64k8-h2048i512"} and len(bad) == 8
    assert all(w.startswith("G1 FAIL") for w in bad.values())
    assert r["pages"]["map-a-e64k8-h2048i512"].startswith("G1 FAIL")
    assert r["cells"]["valid"] == 121 and sum(v == "counted" for v in r["pages"].values()) == 13


def test_a_significant_slab_slope_triggers_the_guard_and_sets_s_per_path(trees):
    _t, r = trees["slab"]
    g = r["domain"]["slab_guard"]
    assert g["triggered"] and g["fit"]["b_us_per_MB"] == pytest.approx(-0.12, abs=0.02)
    assert g["fit"]["b_ci95"][1] < 0 and g["fit"]["dropped"] == ["T2"]
    for p, x in r["domain"]["per_path"].items():
        if x["X_p_us"] is not None:
            assert "slab rule" in x
            if x["S_p_MB"] is not None:
                assert x["verdict"] == "IN THE DOMAIN" and any(f"W_e >= {x['S_p_MB']:.6f} MB" in s
                                                               for s in r["domain"]["D"])
            else:
                assert "no S_p" in x["verdict"]


def test_a_failed_power_check_reads_nothing(trees, tmp_path):
    _t, r = trees["low"]
    assert r["verdict"] == "NOT READ" and "domain" not in r and "600" in r["why"]
    assert RM.lines(r)[0].endswith("not a certification)") and "NOT READ" in RM.lines(r)[0]


def test_the_readout_reads_only_the_committed_registration_and_refuses_a_broken_freeze(trees, tmp_path):
    good, _r = trees["good"]
    with pytest.raises(SystemExit):
        RM.main([str(good), str(tmp_path / "o"), "--registration", str(REGF)])   # no such option
    assert RM.main([str(good), str(tmp_path / "o")]) == 0
    out = json.loads((tmp_path / "o" / "domain.json").read_text())
    assert out["provenance"]["registration_sha256"] == hashlib.sha256(REGF.read_bytes()).hexdigest()
    assert out["provenance"]["registration_path"] == M.REG_JSON
    assert (tmp_path / "o" / "domain.txt").read_text().startswith("MODEL v7 DECODE DOMAIN MAP")
    low, _ = trees["low"]
    assert RM.main([str(low), str(tmp_path / "p")]) == 1
    reg = _reg()
    p = next(iter(reg["freeze"]["own"]))
    reg["freeze"]["own"][p] = "0" * 64
    assert RM.refusals(reg)[0].startswith("the freeze does not hold")
    reg = _reg()
    reg["rental"]["plan_sha256"] = "0" * 64
    assert RM.refusals(reg)[0].endswith("is not the registered plan")
    assert RM.refusals(dict(_reg(), name="another"))[0].startswith("the registration is not")


def test_no_em_dash_and_no_trailer_in_the_map_files():
    files = [REPO / M.REG_JSON, REPO / M.REG_TXT, PLAN, LAUNCHER, Path(__file__), VMSITE / "sitecustomize.py",
             *(REPO / "scripts/scoring/v7domain").glob("*.py"), *(REPO / M.PAGES).glob("*.json")]
    for f in files:
        t = f.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, f
        assert ("Co-" + "Authored") not in t and ("Clau" + "de") not in t, f
