"""main.sh's expected wall time: the smoke rental's MEASURED step seconds (OLMoE-1B-7B, GH200, 2026-11-19), scaled to
ERNIE-4.5-21B-A3B where the work scales. Planning numbers, not measurements; main.sh's dry run prints them.

    python scripts/prod/plan_time.py [--levels 1,8,24,48,80,112,256,1024x3] [--n 8] [--frontier-budget 2700] [--json]

THE SCALING RULES (each line of the plan names its basis):
  ERNIE / OLMoE    expert bytes 23.6 / 12.6 MB = 1.875 (prodlib.expert_bytes), MoE layers 27 / 16; serving work per
                   step and engine start scale by 1.875 x 27/16 = 3.16 (conservative: all of a step's time is taken
                   to scale like the MoE); replay cells by 1.875 (the replay budget's own rule, replay.model_scale);
                   nsys export and extraction by the call count (steps x 27 vs the smoke's 11922 x 16).
  clock            the smoke served at default clocks (up to 1980 MHz): a locked run's serving time x 1980 / lock.
  serving          serve_capture's own estimator (V1's scheduler model; it predicted the smoke's 11917 steps, the
                   smoke logged 11922) on the smoke prompt set's length mix (plan_lengths.json), each step priced by
                   the smoke's median step time in its T band (STEP_MS, the smoke's A/steps.jsonl; it prices the
                   smoke's own run at 44 s, the smoke measured 44 s), x 3.16 x the clock factor.
  replay C         the registered budget (replay.py budget_quota): per branch the probe (PROBE_PER_TILE cells per tile
                   at the scaled smoke walls) plus the rest, which spends (REPLAY_BUDGET_S - probe) / WALL_SAFETY when
                   the walls are right.
  Frontier pages   each cell at the smoke's measured repeats-1 nsys replay wall (prefill a + b T) or repeats-3 wall
                   (decode), x 1.875, plus the step overhead; pages run while elapsed + the dearest page so far fit the
                   budget (main.sh fr_page_step), simulated here in PAGE_ORDER.
  ASSUMED          steps the smoke did not time (env-preflight, pushes, the ERNIE weight download, Frontier's venv
                   build and trainer): stated as assumptions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import frontier_op as FO  # noqa: E402
import prodlib as PL  # noqa: E402
import replay as RP  # noqa: E402
import serve_capture as SC  # noqa: E402

#: the smoke rental's SUMMARY.txt (seconds per step; results/prod-smoke-20261119T204442Z)
SMOKE_S = {"env": 4, "price-check": 81, "prompts-fetch": 3, "prompts-build": 9, "serve-plan": 0,
           "capture-graphs-nsys": 96, "nsys-export": 12, "extract-gate-A": 51, "sequences-A": 2,
           "capture-eager-torchprof": 88, "sag1410": 1048, "replay-decode": 100, "replay-prefill": 130,
           "replay-nsys-decode": 62, "replay-nsys-export-decode": 2, "unlock-after-sag": 1}
#: the smoke's run A: 11922 logged steps in 44 s of serving (manifest.json seconds 44.0), 16 MoE layers, so the
#: capture's other 52 s are the engine start and teardown under nsys
SMOKE_STEPS, SMOKE_SERVE_S, SMOKE_LAYERS = 11922, 44.0, 16
SMOKE_ENGINE_S = SMOKE_S["capture-graphs-nsys"] - SMOKE_SERVE_S
SMOKE_PROMPTS = 1024
#: median wall per engine step (ms) by unpadded T band, the smoke's A/steps.jsonl (graphs, nsys node tracing, default
#: clocks); T 4097..8192 had no step: the mean of its neighbours
STEP_MS = ((1, 2.23), (8, 3.60), (32, 5.02), (64, 5.86), (128, 6.80), (256, 8.54), (512, 12.15), (1024, 39.02),
           (2048, 39.34), (4096, 39.51), (8192, 58.09), (16384, 76.66))
#: replay cell walls (s) from the smoke's cell_wall.json files: repeats 3 decode mean; repeats 1 under nsys,
#: decode mean and prefill least squares a + b T; the replay step's own overhead (step s - sum of cell walls)
WALL_DECODE_R3, WALL_DECODE_R1 = 7.08, 3.99
WALL_PREFILL_R1 = (4.156, 2.164e-3)
REPLAY_STEP_OVERHEAD_S = 15
CLOCK_DEFAULT_MHZ = 1980
#: assumptions (not timed by the smoke)
ASSUMED = {"env-preflight": 120, "revision-pinned": 5, "push-small": 60, "push-big": 120,
           "ernie-download": 220,        # 43.7 GB of bf16 weights at an assumed 200 MB/s
           "frontier-setup": 300, "frontier-train": 180, "frontier-predict": 600, "frontier-points": 20,
           "frontier-csv": 20, "grid-check": 20}


def ernie_scale(model: str = "ernie-4.5-21b-a3b", smoke: str = "olmoe-1b-7b") -> tuple[float, float]:
    """(expert bytes ratio, bytes x layers ratio) ERNIE / OLMoE."""
    eb = PL.expert_bytes(PL.MODELS[model].cfg) / PL.expert_bytes(PL.MODELS[smoke].cfg)
    return eb, eb * PL.MODELS[model].moe_layers / PL.MODELS[smoke].moe_layers


def step_ms(T: int) -> float:
    for hi, ms in STEP_MS:
        if T <= hi:
            return ms
    return STEP_MS[-1][1]


def length_rows(path=HERE / "plan_lengths.json") -> list[dict]:
    doc = json.loads(Path(path).read_text())
    return [{"input_len": a, "max_tokens": b} for a, b in doc["rows"]]


def serving(levels: str, n: int, rows=None) -> dict:
    """{"steps", "serve_s_default" (ERNIE, default clocks), "per_tile"}: the estimator's steps, each priced."""
    rows = rows or length_rows()
    _eb, work = ernie_scale()
    lv = SC.parse_levels(levels)
    i, steps, sec = 0, 0, 0.0
    for L in lv:
        reqs = []
        for _ in range(SC.level_quota(L, n)):
            r = rows[i % len(rows)]
            reqs.append((int(r["input_len"]), int(r["max_tokens"])))
            i += 1
        Ts = SC.simulate_level(L, reqs)
        steps += len(Ts)
        sec += sum(step_ms(T) for T in Ts) / 1e3
    est = SC.estimate_histogram(lv, rows, n)
    return {"steps": steps, "serve_s_default": sec * work, "per_tile": est["per_tile"]}


def clock_factor(lock) -> float:
    return 1.0 if not lock else CLOCK_DEFAULT_MHZ / float(lock)


def replay_branch_s(branch: str, model: str = "ernie-4.5-21b-a3b") -> dict:
    """The registered budget's expected spend on one branch: probe at the scaled smoke walls, then the rest."""
    wall = json.loads((HERE / "replay_wall.json").read_text())
    ms = RP.model_scale(wall, model)
    tiles = [lab for lab, lo, _hi in PL.TILE_BINS if PL.branch(lo) == branch]
    timed = {t: v["median_wall_s"] for t, v in wall["tiles"].items() if t in tiles}
    per = {t: ms * timed.get(t, max(timed.values())) * (9 + 1) / (int(wall["repeats"]) + 1) for t in tiles}
    probe = sum(RP.PROBE_PER_TILE * v for v in per.values())
    rest = max(0.0, RP.REPLAY_BUDGET_S[branch] - probe) / RP.WALL_SAFETY
    cells = sum(int(rest // (RP.WALL_SAFETY * v) // len(per)) for v in per.values()) if per else 0
    return {"probe_s": probe, "rest_s": rest, "rest_cells": cells, "per_cell_s": per}


def frontier_pages(budget: float, pages: int = FO.PAGES, max_tokens: int = FO.MAX_TOKENS) -> dict:
    """Each page's expected seconds and the pages the budget admits (main.sh fr_page_step's rule)."""
    eb, _w = ernie_scale()
    cost = {("decode", 0): eb * WALL_DECODE_R3 * len(FO.page_points("decode", 0, pages, max_tokens))
            + REPLAY_STEP_OVERHEAD_S + 10}
    a, b = WALL_PREFILL_R1
    for pg in range(pages):
        Ts = FO.page_points("prefill", pg, pages, max_tokens)
        cost[("prefill", pg)] = eb * sum(a + b * T for T in Ts) + REPLAY_STEP_OVERHEAD_S + 20
    order = [("decode", 0)] + [("prefill", pg) for pg in FO.page_order(pages)]
    t, dearest, run = 0.0, 0.0, []
    for key in order:
        if run and t + dearest > budget:
            break
        run.append(key)
        t += cost[key]
        dearest = max(dearest, cost[key])
    return {"cost": cost, "run": run, "spent_s": t,
            "points": sum(len(FO.page_points(b_, p_, pages, max_tokens)) for b_, p_ in run)}


def plan(levels: str = "1,8,24,48,80,112,256,1024x3", n: int = 8, frontier_budget: float = 2700,
         model: str = "ernie-4.5-21b-a3b") -> dict:
    """[(block, step, seconds, basis)] in main.sh's order, with the totals."""
    eb, work = ernie_scale(model)
    sv = serving(levels, n)
    calls_ratio = sv["steps"] * PL.MODELS[model].moe_layers / (SMOKE_STEPS * SMOKE_LAYERS)
    engine = SMOKE_ENGINE_S * work
    out = []

    def add(block, step, s, basis):
        out.append((block, step, float(s), basis))

    add("M0", "env-preflight", ASSUMED["env-preflight"], "assumed (not in the smoke)")
    add("M0", "env", SMOKE_S["env"], "smoke")
    add("M0", "power-readback", 1, "smoke")
    add("M0", "unlock-start", 1, "smoke")
    add("M0", "revision-pinned", ASSUMED["revision-pinned"], "assumed")
    add("M0", "replay-wall-present", 1, "trivial")
    add("M0", "price-check", SMOKE_S["price-check"], "smoke (CPU, v8)")
    add("M1", "prompts-fetch", SMOKE_S["prompts-fetch"], "smoke")
    add("M1", "prompts-build", SMOKE_S["prompts-build"] * 4096 / SMOKE_PROMPTS, "smoke x 4096/1024 prompts")
    add("M1", "serve-plan", 1, "smoke")
    add("M2", "gate60-capture", engine + 60 + ASSUMED["ernie-download"],
        "smoke engine start x 3.16 + 60 s + the weight download (assumed 200 MB/s)")
    add("M2", "gate60-export", SMOKE_S["nsys-export"] * 60 / SMOKE_SERVE_S, "smoke x 60/44 s")
    add("M2", "gate60-extract", SMOKE_S["extract-gate-A"] * 60 / SMOKE_SERVE_S, "smoke x 60/44 s")
    add("M2", "gate60-sequences", SMOKE_S["sequences-A"], "smoke")
    add("M2", "push-gate60", ASSUMED["push-small"], "assumed")
    add("M3", "sag1410", SMOKE_S["sag1410"], "smoke (the same OLMoE/Mixtral pages)")
    add("M3", "unlock-after-sag", 1, "smoke")
    for blk, lab, lock in (("M4", "A1710", 1710), ("M5", "A1410", 1410)):
        add(blk, f"{lab}-lock", 1, "trivial")
        add(blk, f"{lab}-capture", engine + sv["serve_s_default"] * clock_factor(lock),
            f"engine start + {sv['steps']} steps x smoke step time x 3.16 x {clock_factor(lock):.3f} (clock)")
        add(blk, f"{lab}-unlock-after", 1, "trivial")
        add(blk, f"{lab}-export", SMOKE_S["nsys-export"] * calls_ratio, f"smoke x calls {calls_ratio:.2f}")
        add(blk, f"{lab}-extract", SMOKE_S["extract-gate-A"] * calls_ratio, f"smoke x calls {calls_ratio:.2f}")
    add("M5", "push-primary", ASSUMED["push-big"], "assumed")
    rb = {}
    for b in ("decode", "prefill"):
        rb[b] = replay_branch_s(b, model)
        add("M6", f"replay-probe-sample-{b}", 3, "smoke (replay-sample)")
        add("M6", f"replay-probe-{b}", rb[b]["probe_s"] + REPLAY_STEP_OVERHEAD_S,
            f"{RP.PROBE_PER_TILE} cells/tile x smoke wall x 1.875 x (9+1)/(3+1)")
        add("M6", f"replay-probe-wall-{b}", 1, "trivial")
        add("M6", f"replay-sample-{b}", 3, "smoke (replay-sample)")
        add("M6", f"replay-{b}", rb[b]["rest_s"] + REPLAY_STEP_OVERHEAD_S,
            f"(REPLAY_BUDGET_S - probe) / {RP.WALL_SAFETY} (the registered budget, walls right)")
    add("M6", "unlock-after-replay", 1, "trivial")
    add("M6", "push-replay", ASSUMED["push-small"], "assumed")
    fp = frontier_pages(frontier_budget)
    add("M7", "frontier-setup", ASSUMED["frontier-setup"], "assumed (uv, the vLLM closure cached by setup_vm)")
    add("M7", "frontier-grid-check", ASSUMED["grid-check"], "assumed")
    order = [("decode", 0)] + [("prefill", pg) for pg in FO.page_order(FO.PAGES)]
    for key in order:
        ran = key in fp["run"]
        add("M7", f"frontier-page-{key[0]}-{key[1]:02d}", fp["cost"][key] if ran else 0,
            ("smoke replay walls x 1.875" if ran else "budget: not run"))
    add("M7", "frontier-points", ASSUMED["frontier-points"], "assumed")
    add("M7", "frontier-csv", ASSUMED["frontier-csv"], "assumed")
    for s in ("frontier-train", "frontier-holdout", "frontier-simpath"):
        add("M7", s, ASSUMED["frontier-train"], "assumed (RF grid search, 27 fits x 10 folds, all cores)")
    add("M7", "push-frontier", ASSUMED["push-small"], "assumed")
    add("M8", "Adefault-unlock", 1, "trivial")
    add("M8", "Adefault-capture", engine + sv["serve_s_default"], "engine start + serving at default clocks")
    add("M8", "Adefault-unlock-after", 1, "trivial")
    add("M8", "Adefault-export", SMOKE_S["nsys-export"] * calls_ratio, "as A1710")
    add("M8", "Adefault-extract", SMOKE_S["extract-gate-A"] * calls_ratio, "as A1710")
    add("M8", "Aon-lock", 1, "trivial")
    add("M8", "Aon-capture", engine + sv["serve_s_default"] * clock_factor(1710), "as A1710")
    add("M8", "Aon-unlock-after", 1, "trivial")
    add("M8", "Aon-export", SMOKE_S["nsys-export"] * calls_ratio, "as A1710")
    add("M8", "Aon-extract", SMOKE_S["extract-gate-A"] * calls_ratio, "as A1710")
    add("M8", "Beager-capture", SMOKE_S["capture-eager-torchprof"] * work,
        "smoke B (one window, <= 90 s of serving) x 3.16")
    add("M8", "Beager-extract", 5, "smoke (extract-gate-B 0 s)")
    for b in ("decode", "prefill"):
        cells = rb[b]["rest_cells"]
        if b == "decode":
            nsys_s = eb * WALL_DECODE_R1 * cells
        else:
            # the prefill sample's mean T: the tiles' quotas are equal, so a mid-tile T per tile (G16: 16384)
            mids = {"64x128x64w8s3": 320, "128x128x64w8s3": 2048, "128x128x64w8s3 g16": 16384}
            per_tile = max(1, cells // 3)
            nsys_s = eb * sum(per_tile * (WALL_PREFILL_R1[0] + WALL_PREFILL_R1[1] * T) for T in mids.values())
        add("M8", f"replay-offset-nsys-{b}", nsys_s + REPLAY_STEP_OVERHEAD_S,
            f"~{cells} cells x the smoke's repeats-1 nsys wall x 1.875")
        add("M8", f"replay-offset-export-{b}", 10, "smoke (2 s) x cells")
        add("M8", f"replay-offset-check-{b}", 10, "assumed")
    add("M8", "replay-offset-unlock", 1, "trivial")
    add("M8", "frontier-predict", ASSUMED["frontier-predict"], "assumed (two models, one batch each)")
    add("M8", "push-final", ASSUMED["push-big"], "assumed")
    add("M8", "held-back-check", 2, "trivial")
    blocks: dict = {}
    for blk, _s, sec, _b in out:
        blocks[blk] = blocks.get(blk, 0.0) + sec
    total = sum(blocks.values())
    return {"steps": out, "blocks": blocks, "total_s": total, "total_h": total / 3600, "serving": sv,
            "frontier": {"budget_s": frontier_budget, "pages_run": len(fp["run"]), "points": fp["points"],
                         "grid_points": len(FO.token_grid()), "spent_s": fp["spent_s"]},
            "levels": levels, "n": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="1,8,24,48,80,112,256,1024x3")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--frontier-budget", type=float, default=2700)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = plan(a.levels, a.n, a.frontier_budget)
    if a.json:
        print(json.dumps(p, indent=1))
        return 0
    for blk, step, sec, basis in p["steps"]:
        print(f"PLAN {blk:<3s} {step:<30s} {sec:8.0f} s  {basis}")
    print("PLAN per tile (estimator): " + ", ".join(f"{k} {v}" for k, v in p["serving"]["per_tile"].items()))
    print("PLAN blocks: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in p["blocks"].items()))
    fr = p["frontier"]
    print(f"PLAN Frontier: {fr['pages_run']} pages, {fr['points']} of {fr['grid_points']} grid points, "
          f"{fr['spent_s']:.0f} s of the {fr['budget_s']:.0f} s budget")
    print(f"PLAN EXPECTED TOTAL {p['total_s']:.0f} s = {p['total_h']:.2f} h (levels {p['levels']}, N {p['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
