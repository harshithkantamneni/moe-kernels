"""Model v7's prefill blind (docs/registered/2026-11-18-v7-prefill-blind-gh200, a DRAFT until the owner's go), off the
GPU and on SYNTHETIC trees and stand-in boards only: the census (no chosen model has a prefill-tile page; B3's and the
gate's models do), the treads and GD5 offline, the pages and the plan recompute; the registered pricer reproduces the
v7 gate's every-switch-on prefill re-run (prefill_rerun_v7) with the refit4 fix, which is the re-run's; the
registration recomputes; the freeze holds and the scorer refuses a broken one, a DRAFT, a moved fingerprint and a
malformed registration; --final refuses without the owner's go or with a moved DRAFT; the census gate passes and fails
on synthetic censuses; the driver is B3's plus lines marked V7P and its dry run lists every unit; the launcher is
vm_run_v7b.sh plus lines marked V7P, delivers the driver, the plan and every page byte for byte on a stub VM and refuses
a DRAFT plan; the scorer, on a stand-in board made of B3's published sag pages and synthetic timed cells, finalises at
that board's par to the registered predictions and reads a pass, a fail, a model under the minimum, a missing page, a
page voided from CLEAN, G3's homogeneous void and a failed power check as registered."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "v7pblind"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import register_v7p_blind as RG  # noqa: E402
import score_blind_v7p as S  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_rr_driver as TRR  # noqa: E402
import v7pcommon as V  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

REGF = REPO / V.REG_JSON
PLAN = REPO / V.PLAN
DRIVER = REPO / "scripts" / V.DRIVER
PARENT_DRIVER = REPO / "scripts" / V.PARENT_DRIVER
LAUNCHER = REPO / V.LAUNCHER
PARENT = REPO / V.PARENT_LAUNCHER
B3S = REPO / "results" / "published" / "2026-11-05-nvidia_gh200_480gb-tileB3-session"
PY = sys.executable


def _reg() -> dict:
    return json.loads(REGF.read_text())


# ---------------------------------------------------------------- the census, the treads, GD5 and the pages

def test_the_census_finds_no_prefill_tile_page_of_the_chosen_models_and_finds_the_seen_ones():
    reg = _reg()
    sessions = reg["models"]["census (COMPUTED)"]["sessions scanned"]
    cen = json.loads(json.dumps(V.census(sessions), default=str))
    assert RG.close(cen, reg["models"]["census (COMPUTED)"]) == []
    per = cen["per model"]
    for m in V.MODELS:
        assert per[m]["verdict"] == "NO PREFILL-TILE PAGE" and per[m]["unseen at prefill"] is True, m
        assert per[m]["synthetic shapes or models with its (E, H, I, k)"] == []
    for m in ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b", "deepseek-v2-lite", "olmoe-1b-7b", "qwen2-57b-a14b",
              "mixtral-8x7b", "mixtral-8x22b", "mixtral-8x7b-tp2"):
        assert per[m]["verdict"] == "HAS A PREFILL-TILE PAGE", m
    assert per["deepseek-v3"]["unseen at prefill"] is False and per["deepseek-v3"][
        "prefill-size timings on other cards (merged.csv rows past 512 tokens, where vLLM's default is a prefill tile)"]
    assert sorted(m for m, x in per.items() if x["unseen at prefill"]) == sorted(
        ["granite-3.0-1b-a400m", "granite-3.0-3b-a800m", "qwen2-57b-a14b-tp8", "deepseek-v3-tp4", "deepseek-v3-tp8",
         "mixtral-8x7b-tp4", "mixtral-8x7b-tp8"])
    ins = V.instrument_check()
    assert all(ins[m]["runs"] for m in V.MODELS) and not any(ins[m]["runs"] for m in
                                                             ("deepseek-v3", "deepseek-v3-tp4", "deepseek-v3-tp8"))
    # the earlier timings the registration discloses are the census's, other tiles only
    other = {m: per[m]["disclosure: its published records at other tiles, per session"] for m in V.MODELS}
    assert all(not (set(t) & V.PREFILL_TILES) for d in other.values() for t in d.values())
    assert "2026-11-18-nvidia_gh200_480gb-v7b-session" in other["granite-3.0-1b-a400m"]
    assert "2026-09-30-nvidia_gh200_480gb-granite-session" in other["granite-3.0-3b-a800m"]
    assert "2026-10-06-nvidia_gh200_480gb-rental3-session" in other["qwen2-57b-a14b-tp8"]


def test_json_tiles_reads_configs_designs_and_contexts():
    out: dict = {}
    V.json_tiles({"model": "a", "block_m": 128, "pinned": {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8,
                                                            "num_stages": 3},
                  "cells": [{"shape": "s", "config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                                      "num_warps": 4, "num_stages": 3}}],
                  "design": {"block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3}},
                 None, None, out)
    assert out == {"a": {"128x128x64w8s3", "64x128x64w8s3"}, "s": {"64x128x64w4s3"}}
    assert V.names_model("gaps-x-deepseek-v3-tp4-dm", "deepseek-v3-tp4")
    assert not V.names_model("gaps-x-deepseek-v3-tp4-dm", "deepseek-v3")


def test_the_treads_are_the_smallest_vllm_runs_at_the_tile_and_gd5_passes_offline():
    from moe.bench.tile_resolve import default_config
    from moe.spec import MODEL_CONFIGS
    want = {"granite-3.0-1b-a400m": {1: (2, 3), 16: (9,)}, "granite-3.0-3b-a800m": {1: (1, 2), 16: (9,)},
            "qwen2-57b-a14b-tp8": {1: (1, 2), 16: (9,)}}
    assert {m: V.treads(m) for m in V.MODELS} == want
    for m, G, ns, _name in V.page_specs():
        E = MODEL_CONFIGS[m].num_experts
        for n in ns:
            c = default_config(V.tokens_at(m, n), E)
            assert (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], c["num_warps"], c["num_stages"],
                    c["GROUP_SIZE_M"]) == (128, 128, 64, 8, 3, G)
            assert G == 16 or n < V.HOPPER_G1_N
            assert n <= V.COPIES
        # a smaller tread is either another tile or the other G
        for n in range(1, min(ns)):
            c = default_config(V.tokens_at(m, n), E)
            assert (c["BLOCK_SIZE_M"], c["GROUP_SIZE_M"]) != (128, G)
    g = V.gd5()
    assert g["verdict"] == "PASS" and len(g["cells"]) == 9
    assert all(not x["shipped in v0.27.1"] and x["provenance"] == "vllm_default_derived" for x in g["cells"].values())
    assert _reg()["models"]["GD5 offline (vLLM 0.27.1's tuned-config lookup)"] == json.loads(json.dumps(g))


def test_the_pages_recompute_and_their_seeds_repeat_no_earlier_draw():
    for p, t in V.page_texts().items():
        assert (REPO / p).read_text() == t, p
    reg = _reg()
    assert reg["cells"]["pages"]["files"] == json.loads(json.dumps(V.files_block()))
    earlier = set()
    for d in (REPO / "docs" / "registered").glob("*-hist"):
        if d.name == Path(V.PAGES).name:
            continue
        for f in d.glob("*.json"):
            try:
                doc = json.loads(f.read_text())
            except ValueError:
                continue
            for c in (doc.get("cells") or []) if isinstance(doc, dict) else []:
                s = (c.get("provenance") or {}).get("seed")
                if s is not None:
                    earlier.add(s)
    from moe.spec import MODEL_CONFIGS
    seeds = []
    for m, G, ns, name in V.page_specs():
        doc = V.load_page(name)
        assert (doc["model"], doc["block_m"], doc["group_m"], doc["shuffle_seed"]) == (m, 128, G, V.BASE)
        assert [c["label"] for c in doc["cells"]] == [x for _n in ns for x in (*V.LABELS, "uniform", "balanced")]
        for c in doc["cells"]:
            assert c["arms"] == ["native", "shared"]
            if c["counts"] is not None:
                assert sum(c["counts"]) == V.tokens_at(m, c["n"]) * MODEL_CONFIGS[m].top_k
            if c.get("provenance"):
                assert c["provenance"]["source_model"] == "olmoe-1b-7b"
                seeds.append(c["provenance"]["seed"])
    assert len(seeds) == len(set(seeds)) == 36 and not set(seeds) & earlier
    assert sum(1 for _ in V.rows()) == 54


# ---------------------------------------------------------------- the plan, the rental, the driver and the launcher

def test_the_plan_is_the_registrations_and_the_rental_numbers():
    reg = _reg()
    assert reg["draft"] is True and "# DRAFT: NOT LAUNCHABLE" in PLAN.read_text()
    assert PLAN.read_text() == V.plan_text(True)
    assert reg["rental"]["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    r = V.rental()
    assert (r["units_min"], r["wall_min"], r["deadline_after_launch_min"], r["hard_cap_after_launch_min"]) == \
        (84, 125, 140, 155)
    assert (r["cost_usd"], r["cost_usd_at_list"], r["cost_at_hard_cap_usd"], r["cost_at_hard_cap_usd_at_list"]) == \
        (4.77, 5.4, 5.92, 6.69)
    pages = re.findall(r"histogram=(\S+)", PLAN.read_text())
    assert sorted(pages) == sorted(V.page_path(s[3]) for s in V.page_specs())
    body = [ln for ln in PLAN.read_text().splitlines() if ln and not ln.startswith("#")]
    assert all("nodrop=1" in ln for ln in body[1:]) and "drop-group" not in PLAN.read_text()
    gates = [ln for ln in body if " gate " in f" {ln} "]
    first_timed = next(i for i, ln in enumerate(body) if " timed " in ln)
    assert all(body.index(g) < first_timed for g in gates)
    assert ["gate=v7pcensus" in gates[0], "gate=band" in gates[1], "gate=b3sag" in gates[2]] == [True] * 3


def test_the_driver_is_b3s_with_only_v7p_marked_lines_changed():
    changed = TRR._held_to_parent(PARENT_DRIVER, DRIVER, "set -uo pipefail", mark="V7P")
    assert len(changed) == 6, changed
    b, a = DRIVER.read_text().splitlines(), PARENT_DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    assert b[0] == a[0] and "MODEL v7'S PREFILL BLIND DRIVER" in b[1]
    assert a[1:a.index("set -uo pipefail")] == b[b.index("# THE B3 DRIVER'S OWN HEADER, UNCHANGED:") + 1:header]
    body = "\n".join(b[header:])
    assert body.count("scripts/v7p_gate.py") == 2 and "|v7pcensus)$" in body
    assert (DRIVER.stat().st_mode & 0o777) == (PARENT_DRIVER.stat().st_mode & 0o777)
    b3 = json.loads((REPO / "docs/registered/2026-11-04-tileB3-gh200.json").read_text())
    assert hashlib.sha256(PARENT_DRIVER.read_bytes()).hexdigest() == b3["code_pins"]["scripts/gh200_tileB3_session.sh"]


def _dry(driver: Path, plan: Path):
    return subprocess.run(["bash", str(driver), "--dry-run", "--plan", str(plan)], capture_output=True, text=True,
                          cwd=REPO, timeout=120)


def test_the_drivers_dry_run_lists_every_unit_and_b3s_driver_refuses_the_v7p_gate():
    got = _dry(DRIVER, PLAN)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 14 units" in got.stdout, got.stdout[-2000:] + got.stderr
    assert "gate v7pcensus: scripts/v7p_gate.py census" in got.stdout and "gate b3sag: scripts/tileB3_gate.py sag" in got.stdout
    for s in V.page_specs():
        assert V.page_path(s[3]) in got.stdout
    old = _dry(PARENT_DRIVER, PLAN)
    assert old.returncode == exit_codes.REFUSED and "a PLAN of" not in old.stdout
    assert "gate v7pcensus" in old.stdout + old.stderr


def test_the_launcher_is_v7bs_plus_lines_marked_v7p():
    changed = TRR._held_to_parent(PARENT, LAUNCHER, None, mark="V7P")
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)", LAUNCHER.read_text(), re.M).group(1).split()
    assert drivers == ["gh200_v7p_session.sh"]
    assert sum('local drv="${DRIVER_ARG:-gh200_v7p_session.sh}"' in x for x in changed) == 1
    assert sum("^# DRAFT" in x and "refuse" in x and V.NAME in x for x in changed) == 1
    assert len(changed) == 9
    assert (LAUNCHER.stat().st_mode & 0o777) == (PARENT.stat().st_mode & 0o777)
    assert _reg()["freeze"]["own"][V.PARENT_LAUNCHER] == hashlib.sha256(PARENT.read_bytes()).hexdigest()


def _laptop(tmp_path, plan_text: str):
    pages = sorted(set(re.findall(r"histogram=(\S+)", plan_text)))
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    (lap.clone / V.PLAN).write_text(plan_text)
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
        return subprocess.run(["bash", str(lap.clone / V.LAUNCHER), *args], env=env, capture_output=True, text=True,
                              timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    return lap, run, pages


def test_start_delivers_the_driver_the_final_plan_and_every_page_byte_for_byte(tmp_path):
    final = V.plan_text(False)
    assert "# DRAFT" not in final
    lap, run, pages = _laptop(tmp_path, final)
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
    run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == hashlib.sha256(final.encode()).hexdigest() and sha(vm / DRIVER.name) == sha(DRIVER)
    assert len(pages) == 6
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_v7p_session.sh --dry-run" in cmds and "gh200_v7p_session.sh --setup" in cmds


def test_start_refuses_the_draft_plan(tmp_path):
    lap, run, _pages = _laptop(tmp_path, V.plan_text(True))
    got = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)
    assert got.returncode == exit_codes.REFUSED and "carries a DRAFT line" in got.stdout + got.stderr
    assert V.NAME in got.stdout + got.stderr
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "--dry-run" not in cmds and "--setup" not in cmds


# ---------------------------------------------------------------- the census gate

def _census(drift: dict | None = None, mismatch=None, drop=None) -> dict:
    ref = json.loads((REPO / V.B.A2_MERGED_CENSUS).read_text())
    ents = []
    for m in V.MODELS:
        for e in ref["entries"]:
            if e["model"] == V.CENSUS_REF_MODEL and e["tile"] == V.TILE and (m, e["gemm"]) != drop:
                x = copy.deepcopy(e)
                x["model"] = m
                x.update((drift or {}).get((m, e["gemm"]), {}))
                ents.append(x)
    return {"entries": ents, "path_mismatch": list(mismatch or [])}


def test_the_census_rule_passes_olmoes_facts_and_fails_any_drift():
    TCA = V.census_mods()
    ref = json.loads((REPO / V.B.A2_MERGED_CENSUS).read_text())
    ok = V.census_check(ref, _census(), TCA.facts, TCA.FIELDS)
    assert ok["verdict"] == "PASS" and ok["compared"] == 6 and set(ok["ctas_per_sm"].values()) == {2}
    bad = V.census_check(ref, _census({("qwen2-57b-a14b-tp8", "w1"): {"regs": 168}}), TCA.facts, TCA.FIELDS)
    assert bad["verdict"] == "FAIL" and "regs" in bad["why"] and bad["drift"]["qwen2-57b-a14b-tp8"]
    occ = V.census_check(ref, _census({("granite-3.0-1b-a400m", "w2"): {"occupancy": {"ctas_per_sm": 1}}}), TCA.facts,
                         TCA.FIELDS)
    assert occ["verdict"] == "FAIL" and "occupancy" in occ["why"]
    assert V.census_check(ref, _census(mismatch=["x"]), TCA.facts, TCA.FIELDS)["verdict"] == "FAIL"
    miss = V.census_check(ref, _census(drop=("granite-3.0-3b-a800m", "w1")), TCA.facts, TCA.FIELDS)
    assert miss["verdict"] == "FAIL" and "not compiled on this board" in miss["why"]


def _gate(*args):
    return subprocess.run([PY, "-B", str(REPO / V.GATE_SCRIPT), *map(str, args)], capture_output=True, text=True,
                          timeout=300, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_the_census_gate_cli_exit_codes_its_ledger_line_and_record(tmp_path):
    res = tmp_path / "results"
    d = res / f"2026-11-19-{V.B.CARD}-tile-census-mixtral-8x7b-{V.CENSUS_LABEL}"
    d.mkdir(parents=True)
    (d / "tile-census.json").write_text(json.dumps(_census()))
    rec = tmp_path / "gate-v7pcensus-1410.json"
    got = _gate("census", "--results", res, "--record", rec, "--lock", 1410)
    assert got.returncode == 0 and got.stdout.startswith("RESULT: GATE v7pcensus PASS ctas_per_sm 2"), got.stdout + got.stderr
    assert json.loads(rec.read_text())["verdict"] == "PASS"
    (d / "tile-census.json").write_text(json.dumps(_census({("granite-3.0-1b-a400m", "w1"): {"shared": 1}})))
    got = _gate("census", "--results", res)
    assert got.returncode == 3 and "v7pcensus FAIL" in got.stdout and "shared differ" in got.stdout
    (d / "tile-census.json").write_text("not json")
    got = _gate("census", "--results", res)
    assert got.returncode == 3 and "malformed" in got.stdout
    got = _gate("census", "--results", tmp_path / "none")
    assert got.returncode == 3 and "no single tile-census.json" in got.stdout
    assert _gate("census", "--results", res, "--lock", 1710).returncode == 2
    assert _gate("census").returncode == 2


# ---------------------------------------------------------------- the freeze, the refusals and --final

def test_the_freeze_holds_and_a_moved_or_missing_file_breaks_it():
    reg = _reg()
    assert V.verify_freeze(reg) == []
    own = reg["freeze"]["own"]
    for p in (*(V.page_path(s[3]) for s in V.page_specs()), "scripts/scoring/v7pblind/v7pcommon.py", V.GATE_SCRIPT,
              f"scripts/{V.DRIVER}", V.LAUNCHER, "scripts/scoring/v7/prefill_rerun_v7.py",
              "scripts/scoring/v7/gate_v7.out.txt"):
        assert p in own, p
    g = V.gate_registration()["freeze"]["files"]
    assert "scripts/r3_timing_model_v7.py" in g and "scripts/round5_chain.py" in g
    bad = copy.deepcopy(reg)
    bad["freeze"]["own"][V.page_path(V.page_specs()[0][3])] = "0" * 64
    bad["freeze"]["own"]["scripts/no-such-file.py"] = "0" * 64
    why = V.verify_freeze(bad)
    assert len(why) == 2 and "missing" in why[1]
    bad["freeze"]["gate registration"]["constants_sha256"] = "x"
    assert any("constants hash" in w for w in V.verify_freeze(bad))
    assert V.verify_freeze({"freeze": None})


def test_the_scorer_refuses_a_draft_a_foreign_or_malformed_registration_and_a_broken_freeze():
    reg = _reg()
    assert S.refusals(reg) == ["the registration is a DRAFT (the owner's go is not recorded): nothing may be scored "
                               "against it"]
    assert S.refusals([1, 2])[0].startswith("the registration is malformed")
    assert S.refusals(dict(reg, name="x"))[0].startswith("the registration is not")
    broken = copy.deepcopy(reg)
    broken["draft"] = False
    broken["freeze"]["own"][V.GATE_SCRIPT] = "0" * 64
    assert S.refusals(broken)[0].startswith("the freeze does not hold")
    got = subprocess.run([PY, "-B", str(REPO / "scripts/scoring/v7pblind/score_blind_v7p.py"), "/nonexistent",
                          "/tmp/never-written"], capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert got.returncode == 2 and "DRAFT" in got.stdout


def test_final_refuses_without_the_owners_go_or_with_a_moved_draft():
    reg = _reg()
    doc = copy.deepcopy(reg)
    doc.update(draft=False, status="REGISTERED", **{"owner's go": "2026-11-19: go, one blind"})
    go = "2026-11-19: go, one blind"
    assert RG.final_refusals(doc, reg, go) == []
    assert any("owner's go" in w for w in RG.final_refusals(doc, reg, None))
    assert any("owner's go" in w for w in RG.final_refusals(doc, reg, "yes"))
    moved = copy.deepcopy(doc)
    moved["rules"]["minimum cells"] = 5
    assert any("differs from the committed DRAFT" in w for w in RG.final_refusals(moved, reg, go))
    moved = copy.deepcopy(doc)
    moved["predictions"]["cells at A2's par"][0]["ms"] += 1e-3
    assert any("differs from the committed DRAFT" in w for w in RG.final_refusals(moved, reg, go))
    seen = copy.deepcopy(doc)
    seen["models"]["census (COMPUTED)"]["per model"]["granite-3.0-1b-a400m"]["unseen at prefill"] = False
    seen["models"]["census (COMPUTED)"]["per model"]["granite-3.0-1b-a400m"]["verdict"] = "HAS A PREFILL-TILE PAGE"
    assert any("granite-3.0-1b-a400m: HAS A PREFILL-TILE PAGE" in w for w in RG.final_refusals(seen, reg, go))
    assert any("no committed DRAFT" in w for w in RG.final_refusals(doc, None, go))
    assert any("not a DRAFT" in w for w in RG.final_refusals(doc, doc, go))
    # the census may move between the DRAFT and --final (sessions published since): not a refusal
    later = copy.deepcopy(doc)
    later["models"]["census (COMPUTED)"]["sessions scanned"].append("2026-11-20-x-session")
    assert RG.final_refusals(later, reg, go) == []


def test_the_registration_states_the_reopening_the_override_and_the_rules():
    reg = _reg()
    txt = (REPO / V.REG_TXT).read_text()
    head = "\n".join(txt.splitlines()[:12])
    for s in ("THE THIRD REOPENING OF PREFILL", "OVERRIDES the gate's registered PREFILL NO-GO", "P(A) 0.2067",
              "B 0.284", "D 0.1885", "EVERY EARLIER REGISTERED VERDICT STANDS", "B3", "round 5's gate: FAIL",
              "PREFILL: NO-GO: REFUSED", "v7's decode blind"):
        assert s in head, s
    e = reg["the gate's own pass estimate (ESTIMATE)"]
    assert (e["P(A)"], e["B"], e["D"]) == (0.2067, 0.284, 0.1885)
    r = reg["rules"]
    assert r["bars"]["rms_max"] == 0.03 and r["bars"]["max_abs"] == 0.08 and r["bars"]["mean_max"] == 0.015
    assert "a model is CERTIFIED when its own bar and the pooled bar hold" in r["verdict (registered)"]
    assert r["minimum cells"] == 10 and "BELOW THE MINIMUM" in r["minimum"]
    assert "DESCRIPTIVE" in r["B3's E1-TILE (DESCRIPTIVE)"] and "power check first" in r["voids"]
    d = " ".join(reg["disclosures"])
    for s in ("POST-HOC ORIGIN", "v7 UNCHANGED", "third reopening".upper(), "per-call risk", "$2.29/h", "$2.59/h"):
        assert s in d or s.lower() in d.lower(), s
    fx = reg["model v7 (frozen)"]["the fix (part of the frozen model)"]
    assert fx["source (verbatim)"] == V.FIX_SOURCE and 'C["r7"]._rc = lambda: RF' in fx["source (verbatim)"]
    assert reg["model v7 (frozen)"]["equals the committed re-run (scripts/scoring/v7/prefill_rerun_v7.json, every switch on)"] is True
    assert reg["model v7 (frozen)"]["switches"] == {"RR": True, "D6FIX": True}
    assert chr(0x2014) not in txt and chr(0x2013) not in txt


def test_the_fix_is_the_reruns_fix():
    import inspect
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "v7"))
    import prefill_rerun_v7 as PR

    def body(src):
        lines = [ln.strip() for ln in src.splitlines()[1:]]
        lines = [ln for ln in lines if ln and not ln.startswith('"""') and not ln.startswith("import round5_chain")]
        return lines
    assert body(inspect.getsource(PR.apply_fix)) == body(V.FIX_SOURCE)


# ---------------------------------------------------------------- the pure certification

def _res(per: dict, pooled="HOLDS"):
    return {"BARS": {"pooled": {"verdict": pooled}, "per_model": {m: {"verdict": v} for m, v in per.items()}}}


def test_certification_needs_the_own_bar_the_pooled_bar_and_the_minimum():
    allh = {m: "HOLDS" for m in V.MODELS}
    clean = {"BARS": {"cells scored": {m: 18 for m in V.MODELS}}}
    assert S.certify(_res(allh), clean, True) == {m: "CERTIFIED" for m in V.MODELS}
    c = S.certify(_res(dict(allh, **{"granite-3.0-3b-a800m": "FAILS"})), clean, True)
    assert c["granite-3.0-3b-a800m"].startswith("NOT CERTIFIED (own bar FAILS") and c["qwen2-57b-a14b-tp8"] == "CERTIFIED"
    assert all(v.startswith("NOT CERTIFIED") for v in S.certify(_res(allh, "FAILS"), clean, True).values())
    inc = S.certify(_res(dict(allh, **{"qwen2-57b-a14b-tp8": "INCONCLUSIVE (ALL HOLDS; CLEAN FAILS)"})), clean, True)
    assert inc["qwen2-57b-a14b-tp8"].startswith("NOT CERTIFIED")
    few = {"BARS": {"cells scored": dict({m: 18 for m in V.MODELS}, **{"granite-3.0-1b-a400m": 9})}}
    assert S.certify(_res(allh), few, True)["granite-3.0-1b-a400m"].startswith("NOT CERTIFIED: BELOW THE MINIMUM (9")
    assert S.certify(_res(allh), clean, False) == {m: "NOT CERTIFIED (not scored)" for m in V.MODELS}
    g3 = {"BARS": {"verdict": "NOT SCORED: G3 FAILS (homogeneous), so every histogram cell is voided"}}
    assert all("G3 FAILS" in v for v in S.certify(g3, clean, True).values())


# ---------------------------------------------------------------- the frozen chain (slow): the registration, the gate, the scorer

@pytest.fixture(scope="module")
def built():
    fp = V.frozen_predictions()
    rep = V.gate_reproduction()
    return {"fp": fp, "rep": rep}


def test_the_registered_pricer_reproduces_the_v7_gates_every_switch_on_prefill_rerun(built):
    rep = json.loads(json.dumps(built["rep"]))
    assert rep == V.committed_rerun()
    assert rep["estimate"]["A (5 models)"] == 0.2067 and rep["means_pct"]["mixtral-8x7b-tp2"] == -1.99
    assert rep["b3_information"]["pooled"] == {"n": 84, "mean": 0.19, "rms": 1.48, "worst": 3.82}
    C = V.R5.chain()
    assert C["r7"]._rc() is C["RF"]


def test_the_registration_recomputes_to_its_committed_files(built):
    reg = _reg()
    wb = reg["pass estimate"]["at the working-model bar (B3's method, b3common.pass_estimates, the same cells and means; DESCRIPTIVE)"]
    sessions = reg["models"]["census (COMPUTED)"]["sessions scanned"]
    doc = json.loads(json.dumps(RG.build(reg["draft"], sessions=sessions, fp=built["fp"], rep=built["rep"], wb=wb),
                                default=str))
    assert RG.close(doc, reg) == []
    assert (REPO / V.REG_TXT).read_text() == "\n".join(RG.text(doc)) + "\n"
    assert S.fingerprint_refusals(reg, built["fp"]) == []
    moved = copy.deepcopy(reg)
    moved["predictions"]["fingerprint (the scorer recomputes it before reading anything)"]["B3"]["ms"][0] += 1e-6
    assert S.fingerprint_refusals(moved, built["fp"])[0].startswith("the registered pricer does not reproduce")
    fp = built["fp"]
    assert fp["n_cells"] == 54 and fp["unpriced"] == []
    for c in fp["boards"]["A2"]["cells"]:
        assert (c["RR"] is None) == (c["G"] == 16)
        if c["G"] == 16:
            assert c["ms"] == c["round5_ms (DESCRIPTIVE, every switch off)"]


def test_the_working_bar_estimate_is_b3s_method_on_v7s_means(built):
    wb = V.working_bar_estimate(built["rep"]["means_pct"], R=200_000)
    reg = _reg()["pass estimate"]
    assert json.loads(json.dumps(wb)) == reg[
        "at the working-model bar (B3's method, b3common.pass_estimates, the same cells and means; DESCRIPTIVE)"]
    assert 0.5 < wb["A (5 models)"]["E1-TILE"] < 0.7


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    """A stand-in board: B3's published session's two f128 pages, power, lock, G-BAND and b3sag files, and a PASS
    census record from the census rule on a synthetic census of these models."""
    root = tmp_path_factory.mktemp("v7p") / "sess"
    res, drv = root / "results", root / "session" / "gh200-driver"
    res.mkdir(parents=True)
    drv.mkdir(parents=True)
    for d in sorted((B3S / "results").glob("*-f128-r3-counters")):
        os.symlink(d, res / d.name)
    for f in ("c2-power.env", "locks.env", "gate-band-1410.json", "gate-b3sag-1410.json"):
        shutil.copy(B3S / "session" / "gh200-driver" / f, drv / f)
    TCA = V.census_mods()
    ref = json.loads((REPO / V.B.A2_MERGED_CENSUS).read_text())
    (drv / V.GATE_RECORDS["census"]).write_text(V.B.dumps({"gate": "census", **V.census_check(ref, _census(), TCA.facts,
                                                                                              TCA.FIELDS)}))
    return root


@pytest.fixture(scope="module")
def fin(standin):
    rec = json.loads((standin / "session" / "gh200-driver" / V.GATE_RECORDS["sag"]).read_text())
    return S.finalise(standin, rec)


def test_finalise_on_b3s_board_reproduces_the_registered_predictions_at_its_par(fin):
    reg = _reg()
    want = reg["predictions"]["fingerprint (the scorer recomputes it before reading anything)"]["B3"]
    assert fin["par"] == want["par"] == [59.815, 612.5]
    assert [c["ms"] for c in fin["cells"]] == want["ms"] and fin["unpriced"] == []


def test_finalise_refuses_a_failed_or_foreign_sag_record_and_a_blind_page_opened(standin, monkeypatch):
    rec = json.loads((standin / "session" / "gh200-driver" / V.GATE_RECORDS["sag"]).read_text())
    with pytest.raises(S.Refused, match="did not PASS"):
        S.finalise(standin, dict(rec, verdict="FAIL"))
    with pytest.raises(S.Refused, match="par or page sha256"):
        S.finalise(standin, dict(rec, par=[50.0, 600.0]))
    gaps = standin / "results" / "gaps-nvidia_gh200_480gb-granite-3.0-1b-a400m-t128g1"
    gaps.mkdir(exist_ok=True)
    (gaps / "report.json").write_text("{}")
    real = V.finalise_pars

    def peek(results, record):
        (Path(results) / gaps.name / "report.json").read_text()
        return real(results, record)
    monkeypatch.setattr(V, "finalise_pars", peek)
    with pytest.raises(S.Refused, match="opened a blind page"):
        S.finalise(standin, rec)


def _score(standin, fin, monkeypatch, e=None, out_pages=(), clean_out=(), keep=None, g3=None):
    """S.score on the stand-in with the finalised predictions and synthetic timed cells T = T_pred / (1 + e)."""
    reg = copy.deepcopy(_reg())
    reg["draft"] = False
    TC = V.R5.chain()["d5"].TC
    e = e or {}
    monkeypatch.setattr(S, "finalise", lambda sess, rec: fin)

    def timed_page(tree, model, label, G, tv, f, view="registered"):
        if (model, label) in out_pages:
            return None, "NOT SCORED (G5): synthetic"
        if (model, label) in clean_out and tv.name == "CLEAN":
            return None, "out of CLEAN (G1: synthetic)"
        return {"fake": True}, "counted (synthetic)"

    def valid_cells(rep, sub, model, page):
        got = {k: v["ms"] / (1 + e.get(model, 0.005)) for k, v in sub.items()}
        if keep and model in keep:
            got = dict(list(got.items())[:keep[model]])
        return got, [], None
    monkeypatch.setattr(TC, "timed_page", timed_page)
    monkeypatch.setattr(TC, "valid_cells", valid_cells)
    if g3 is not None:
        monkeypatch.setattr(TC.C5, "g3_rule", lambda rows: dict(g3, rows=rows))
    return S.score(standin, reg, check_commit=False)


def test_the_scorer_certifies_every_model_v7_prices(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch)
    assert s["verdict"] == "SCORED" and s["validity"]["valid"] is True
    b = s["BARS (registered: ALL and CLEAN, rule 3)"]
    assert b["pooled"]["verdict"] == "HOLDS" and b["pooled"]["stats"]["cells"] == 54
    assert s["certified"] == {m: "CERTIFIED" for m in V.MODELS}
    assert s["B3's E1-TILE (DESCRIPTIVE)"]["reading"] == "HOLDS"
    assert s["cells scored in CLEAN"] == {m: 18 for m in V.MODELS}


def test_the_scorer_fails_a_mispriced_model_and_keeps_the_others(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, e={"granite-3.0-3b-a800m": 0.03})
    b = s["BARS (registered: ALL and CLEAN, rule 3)"]
    assert b["per_model"]["granite-3.0-3b-a800m"]["verdict"] == "FAILS" and b["pooled"]["verdict"] == "HOLDS"
    assert s["certified"]["granite-3.0-3b-a800m"].startswith("NOT CERTIFIED (own bar FAILS")
    assert s["certified"]["granite-3.0-1b-a400m"] == s["certified"]["qwen2-57b-a14b-tp8"] == "CERTIFIED"
    assert s["B3's E1-TILE (DESCRIPTIVE)"]["reading"] == "FAILS"


def test_a_model_under_the_minimum_is_not_certified_and_still_pooled(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, keep={"qwen2-57b-a14b-tp8": 4})   # 4 a page, 8 in all
    assert s["cells scored in CLEAN"]["qwen2-57b-a14b-tp8"] == 8
    assert s["certified"]["qwen2-57b-a14b-tp8"].startswith("NOT CERTIFIED: BELOW THE MINIMUM (8 cells")
    assert s["BARS (registered: ALL and CLEAN, rule 3)"]["pooled"]["stats"]["cells"] == 44
    assert s["certified"]["granite-3.0-1b-a400m"] == "CERTIFIED"


def test_a_missing_page_leaves_its_model_under_the_minimum(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, out_pages={("granite-3.0-1b-a400m", "t128g1")})
    assert s["cells scored in CLEAN"]["granite-3.0-1b-a400m"] == 6
    assert s["certified"]["granite-3.0-1b-a400m"].startswith("NOT CERTIFIED: BELOW THE MINIMUM (6 cells")
    assert s["views"]["addendum"]["CLEAN"]["pages"]["granite-3.0-1b-a400m-t128g1"].startswith("NOT SCORED (G5)")


def test_a_page_voided_from_clean_counts_in_all_only(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, clean_out={("granite-3.0-3b-a800m", "t128g1")})
    assert s["cells scored in CLEAN"]["granite-3.0-3b-a800m"] == 6
    assert s["BARS (registered: ALL and CLEAN, rule 3)"]["pooled"]["stats"]["cells"] == 54   # ALL's numbers
    assert s["certified"]["granite-3.0-3b-a800m"].startswith("NOT CERTIFIED: BELOW THE MINIMUM (6 cells")
    assert s["certified"]["qwen2-57b-a14b-tp8"] == "CERTIFIED"


def test_g3s_homogeneous_fail_voids_every_histogram_cell(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, g3={"verdict": "FAIL (voids every histogram cell of the board)",
                                              "void_strata": "ALL"})
    assert s["BARS (registered: ALL and CLEAN, rule 3)"]["verdict"].startswith("NOT SCORED: G3 FAILS (homogeneous)")
    assert s["verdict"] == "NOT SCORED" and s["why"].startswith("NOT SCORED: G3 FAILS (homogeneous)")
    assert S.lines(s)[-1].startswith("certified")
    assert all(v.startswith("NOT CERTIFIED (NOT SCORED: G3") for v in s["certified"].values())


def test_a_failed_power_check_reads_not_scored(standin, fin, monkeypatch, tmp_path):
    bad = tmp_path / "sess"
    shutil.copytree(standin, bad, symlinks=True)
    (bad / "session" / "gh200-driver" / "c2-power.env").write_text("C2_POWER_W=900\n")
    s = _score(bad, fin, monkeypatch)
    assert s["verdict"] == "NOT SCORED" and "C2_POWER_W=700" in s["why"]
    assert s["certified"] == {m: "NOT CERTIFIED (not scored)" for m in V.MODELS}
    (bad / "session" / "gh200-driver" / "c2-power.env").write_text("C2_POWER_W=700\n")
    (bad / "session" / "gh200-driver" / V.GATE_RECORDS["census"]).unlink()
    s = _score(bad, fin, monkeypatch)
    assert s["verdict"] == "NOT SCORED" and "gate census reads None" in s["why"]


def test_g3s_heterogeneous_fail_voids_the_failing_strata_non_balanced_cells(standin, fin, monkeypatch):
    s = _score(standin, fin, monkeypatch, g3={"verdict": "FAIL in strata ['0'] (heterogeneous: only those strata's "
                                                         "histogram cells void)", "void_strata": ["0"]})
    assert s["verdict"] == "SCORED"
    for view in (s["views"], s["views"]["addendum"]["CLEAN"]):
        voided = view["voided_cells"]
        assert len(voided) == 30 and not any(c.endswith("/balanced") for c in voided)
        smallest = {name: min(ns) for _m, _G, ns, name in V.page_specs()}
        assert all(int(c.split("/")[2]) == smallest[c.split("/")[1]] for c in voided)   # stratum 0: each page's smallest n
        assert view["DESCRIPTIVE"]["the bars without G3's per-stratum voids"]["pooled"]["stats"]["cells"] == 54
    assert s["BARS (registered: ALL and CLEAN, rule 3)"]["pooled"]["stats"]["cells"] == 24
    assert s["cells scored in CLEAN"] == {m: 8 for m in V.MODELS}
    assert all(v.startswith("NOT CERTIFIED: BELOW THE MINIMUM (8 cells") for v in s["certified"].values())


def _ledger(root: Path, sha: str | None, commit_txt: str | None = None) -> Path:
    d = root / "session" / "gh200-driver"
    d.mkdir(parents=True, exist_ok=True)
    if sha is not None:
        (d / "status").write_text(f"2026-11-19T00:00:00Z driver start: commit {sha}, steps of the plan\n")
    if commit_txt is not None:
        (root / "session" / "commit.txt").write_text(commit_txt + "\n")
    return root


def test_the_scorer_refuses_a_session_whose_commit_does_not_hold_the_freeze(tmp_path):
    reg = _reg()
    assert S.commit_refusals(_ledger(tmp_path / "a", None), reg)[0].startswith("the rental's commit is not on its ledger")
    got = S.commit_refusals(_ledger(tmp_path / "b", "0" * 40), reg)
    assert got[0].startswith("the working tree is not the rental's commit 000000000000") and "absent at" in got[0]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    got = S.commit_refusals(_ledger(tmp_path / "c", head, commit_txt="1" * 40), reg)
    assert "commit.txt reads" in got[0]
    first = subprocess.run(["git", "-C", str(REPO), "rev-list", "--max-parents=0", "HEAD"], capture_output=True,
                           text=True).stdout.split()[0]
    got = S.commit_refusals(_ledger(tmp_path / "d", first), reg)
    assert got and got[0].startswith(f"the working tree is not the rental's commit {first[:12]}")
    files = {V.REG_JSON, V.REG_TXT, V.PLAN, *reg["freeze"]["own"]}
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *sorted(files)], capture_output=True,
                           text=True).stdout.strip()
    if not dirty:   # the committed kit: HEAD holds its own freeze
        assert S.commit_refusals(_ledger(tmp_path / "e", head), reg) == []
    with pytest.raises(S.Refused, match="not on its ledger"):
        S.score(_ledger(tmp_path / "f", None), dict(reg, draft=False))
    assert S.dirty_refusals("") == []
    assert S.dirty_refusals(" M scripts/x.py\n")[0].startswith("the checkout is dirty (git_dirty true)")
    assert S.dirty_refusals(None)[0].startswith("git status could not be read")
