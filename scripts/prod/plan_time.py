"""main.sh's expected wall time: the smoke rental's MEASURED step seconds (OLMoE-1B-7B, GH200, 2026-11-19), scaled to
ERNIE-4.5-21B-A3B where the work scales. Planning numbers, not measurements; main.sh's dry run prints them.

    python scripts/prod/plan_time.py [--levels 1,8,24,48,80,112,256,1024x3] [--n 8] [--json]

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
  Frontier timer   every grid point (387, no budget) x its samples (Frontier's default "uniform" routing: 3 seeds
                   per point, 1161 timed samples; the opt-in balanced arm is not in the default plan), per branch
                   pass: the process start, the weight build and the Triton compiles (ASSUMED once per pass) + per
                   sample (warm-up + timed calls) x the call's wall
                   (the larger of the eager host time per call, ASSUMED, and the call's GPU time: the smoke's replay C
                   per-call medians, OLMoE at the branch clock, x 1.875) + a per-sample overhead (ASSUMED: Frontier's
                   generator on the card, the synchronizes, two NVML reads); the nsys export by the kernel count
                   against the smoke's; the extraction (fpoints) ASSUMED. The early probe (M2, frontier-timer-probe):
                   the same pricing on its 3 points x 3 samples, two passes, plus fpoints, the CSV, the narrow
                   trainer and the predictions (ASSUMED).
  ERNIE download   ASSUMED 200 MB/s in the expected total; env-preflight's P12 MEASURES the rate and fails under
                   50 MB/s, so the worst case the run can meet is the floor's (printed beside the total).
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
#: the call's GPU time (ms, OLMoE-1B-7B, one cold call of the GRR replay) from the smoke's replay C cells.csv ms_p50:
#: decode (1710 MHz) 0.04 ms at T 1-2 (the naive path) and 0.21-0.26 ms at T 16-128; prefill (1410 MHz) 0.26 ms at
#: T 152-272, 0.47 at 1262, 0.67 at 2048, 4.15 at 16384: priced as max(floor, a + b T)
CALL_MS_DECODE = 0.26
CALL_MS_PREFILL = (0.17, 2.43e-4, 0.26)
#: smoke nsys export: 12 s for the smoke's A run (11922 steps x 16 MoE calls, ~6 core kernels per call + the rest of
#: the step); the timer's calls carry ~15 kernels each (the functional API's ~9 fills): priced per kernel, floor 10 s
SMOKE_EXPORT_KERNELS = SMOKE_STEPS * SMOKE_LAYERS * 12
#: assumptions (not timed by the smoke)
#: ERNIE's bf16 weights (GB) and the assumed / floor download rates (MB/s; the floor is env-preflight P12's)
ERNIE_WEIGHTS_GB, DOWNLOAD_MBPS, DOWNLOAD_FLOOR_MBPS = 43.7, 200.0, 50.0
ASSUMED = {"env-preflight": 150,         # P1..P11 + P12's timed download (<= 20 s)
           "revision-pinned": 5, "push-small": 60, "push-big": 120,
           "ernie-download": ERNIE_WEIGHTS_GB * 1e3 / DOWNLOAD_MBPS,
           "frontier-setup": 300, "frontier-train": 180, "frontier-predict": 600, "frontier-points": 60,
           "frontier-csv": 20, "grid-check": 20, "frontier-csv-partial": 5, "frontier-train-partial": 5,
           # the probe's Frontier side: fpoints + csv + the narrow trainer (9 rows, 1 x 3 folds) + predict, each a
           # process start in Frontier's venv
           "frontier-probe-frontier": 90,
           # the timer: interpreter + vLLM import + weights built once + Triton's compiles, per pass; the eager host
           # time of one functional fused_experts call under nsys CUDA tracing (~15 launches) plus its synchronize;
           # per point: the routing build, the synchronizes and two NVML reads
           "frontier-timer-start": 120, "frontier-timer-host-call-ms": 1.0, "frontier-timer-point-ms": 30.0}


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


def serving(levels: str, n: int, rows=None, model: str = "ernie-4.5-21b-a3b", step_factor: float = 1.0) -> dict:
    """{"steps", "serve_s_default" (the model, default clocks), "per_tile"}: the estimator's steps, each priced
    (x step_factor: the long-context run's assumed attention cost, extras_plan)."""
    rows = rows or length_rows()
    _eb, work = ernie_scale(model)
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
    est = SC.estimate_histogram(lv, rows, n, E=PL.MODELS[model].cfg.num_experts)
    return {"steps": steps, "serve_s_default": sec * work * step_factor, "per_tile": est["per_tile"]}


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


def call_ms(T: int, model: str = "ernie-4.5-21b-a3b") -> float:
    """The call's GPU ms on the model (ERNIE by default) at its branch clock (the smoke's replay C per-call medians x
    the expert-bytes ratio: ERNIE 1.875)."""
    eb, _w = ernie_scale(model)
    if T <= 128:
        return eb * CALL_MS_DECODE
    a, b, floor = CALL_MS_PREFILL
    return eb * max(floor, a + b * T)


def frontier_timer(max_tokens: int = FO.MAX_TOKENS, warmup: int = FO.WARMUP_STEPS,
                   repeats: int = FO.ACTIVE_STEPS, routing: str = FO.DEFAULT_ROUTING,
                   points: list[int] | None = None, model: str = "ernie-4.5-21b-a3b") -> dict:
    """Per branch pass: {"points", "samples", "timer_s", "export_s", "calls"} for every grid point (no budget; or
    the given `points`: the probe's) x its samples (uniform: 3)."""
    out = {}
    ns = len(FO.samples_for(routing))
    for b in ("decode", "prefill"):
        Ts = FO.branch_points(b, max_tokens) if points is None else [T for T in points if PL.branch(T) == b]
        per = ns * sum((warmup + repeats) * max(ASSUMED["frontier-timer-host-call-ms"], call_ms(T, model))
                       + ASSUMED["frontier-timer-point-ms"] for T in Ts) / 1e3
        calls = ns * len(Ts) * (warmup + repeats)
        out[b] = {"points": len(Ts), "samples": ns * len(Ts), "calls": calls,
                  "timer_s": ASSUMED["frontier-timer-start"] + per,
                  "export_s": max(10.0, SMOKE_S["nsys-export"] * calls * 15 / SMOKE_EXPORT_KERNELS)}
    return out


def frontier_probe_s(model: str = "ernie-4.5-21b-a3b") -> float:
    """The early probe: two timer passes on PROBE_POINTS (3 samples each) with their exports and locks, then
    Frontier's fpoints / csv / narrow train / predict."""
    pr = frontier_timer(points=list(FO.PROBE_POINTS[model]))
    return sum(v["timer_s"] + v["export_s"] + 2 for v in pr.values()) + ASSUMED["frontier-probe-frontier"]


def plan(levels: str = "1,8,24,48,80,112,256,1024x3", n: int = 8, model: str = "ernie-4.5-21b-a3b") -> dict:
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
        f"smoke engine start x 3.16 + 60 s + the weight download ({ERNIE_WEIGHTS_GB} GB at an assumed "
        f"{DOWNLOAD_MBPS:.0f} MB/s; env-preflight P12 measures it, floor {DOWNLOAD_FLOOR_MBPS:.0f})")
    add("M2", "gate60-export", SMOKE_S["nsys-export"] * 60 / SMOKE_SERVE_S, "smoke x 60/44 s")
    add("M2", "gate60-extract", SMOKE_S["extract-gate-A"] * 60 / SMOKE_SERVE_S, "smoke x 60/44 s")
    add("M2", "gate60-sequences", SMOKE_S["sequences-A"], "smoke")
    add("M2", "frontier-setup", ASSUMED["frontier-setup"], "assumed (uv, the vLLM closure cached by setup_vm)")
    add("M2", "frontier-grid-check", ASSUMED["grid-check"], "assumed")
    add("M2", "frontier-timer-probe", frontier_probe_s(model),
        f"{len(FO.PROBE_POINTS[model])} points x 3 samples: two timer passes priced as M7's + "
        f"{ASSUMED['frontier-probe-frontier']} s of Frontier's fpoints/csv/narrow train/predict (assumed)")
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
    ft = frontier_timer()
    for b in ("decode", "prefill"):
        add("M7", f"frontier-timer-lock-{b}", 1, "trivial")
        add("M7", f"frontier-timer-{b}", ft[b]["timer_s"],
            f"start {ASSUMED['frontier-timer-start']} s (assumed) + {ft[b]['points']} points x 3 samples x "
            f"{FO.WARMUP_STEPS}+{FO.ACTIVE_STEPS} calls x max(host {ASSUMED['frontier-timer-host-call-ms']} ms "
            "(assumed), smoke call ms x 1.875)")
        add("M7", f"frontier-timer-export-{b}", ft[b]["export_s"], f"smoke export x kernels ({ft[b]['calls']} calls)")
    add("M7", "frontier-timer-unlock", 1, "trivial")
    add("M7", "frontier-points", ASSUMED["frontier-points"], "assumed")
    add("M7", "frontier-csv", ASSUMED["frontier-csv"], "assumed")
    for s in ("frontier-train", "frontier-holdout", "frontier-simpath"):
        add("M7", s, ASSUMED["frontier-train"], "assumed (RF grid search, 27 fits x 10 folds, all cores)")
    add("M7", "frontier-csv-fills", ASSUMED["frontier-csv"], "assumed")
    add("M7", "frontier-train-fills", ASSUMED["frontier-train"], "assumed (as frontier-train)")
    add("M7", "frontier-csv-partial", ASSUMED["frontier-csv-partial"], "assumed (complete grid: no partial CSV)")
    add("M7", "frontier-train-partial", ASSUMED["frontier-train-partial"], "assumed (complete grid: nothing trained)")
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
    slow = ERNIE_WEIGHTS_GB * 1e3 * (1 / DOWNLOAD_FLOOR_MBPS - 1 / DOWNLOAD_MBPS)
    return {"steps": out, "blocks": blocks, "total_s": total, "total_h": total / 3600, "serving": sv,
            "worst_download_total_s": total + slow, "worst_download_total_h": (total + slow) / 3600,
            "frontier": {"points": sum(v["points"] for v in ft.values()), "grid_points": len(FO.token_grid()),
                         "samples": sum(v["samples"] for v in ft.values()),
                         "timer_s": sum(v["timer_s"] + v["export_s"] for v in ft.values())},
            "levels": levels, "n": n}


# ------------------------------------------------------------------------------------------------------------
# extras.sh (after main, same rental): X1 noise floor, X2 ncu bytes, X3 Qwen3-30B-A3B, X4 long-context prefill
# ------------------------------------------------------------------------------------------------------------

QWEN3 = "qwen3-30b-a3b"
#: Qwen3-30B-A3B's bf16 weights (GB, ASSUMED: 30.5 B parameters x 2 bytes; env-preflight P12 measures the rate)
QWEN3_WEIGHTS_GB = 61.1
#: X3's replay budget per branch (s; the owner's "smaller budget, e.g. 1200 s per branch"), the registered rule priced
#: by main's own ERNIE probe walls x the expert-bytes ratio
QWEN3_REPLAY_BUDGET_S = 1200
#: X4: the long-context serving (levels, N per level, prompt lengths uniform in [LONG_MIN, LONG_MAX], output fixed);
#: ASSUMED: a step of long-context prefill costs LONG_ATTN_FACTOR x the smoke's step time at its T (the smoke's prompts
#: were <= 2048 tokens: attention over 16k-token contexts is not in its step times)
LONG_LEVELS, LONG_N, LONG_MIN, LONG_MAX, LONG_OUT, LONG_PROMPTS = "1,2,4,8", 24, 4096, 16000, 16, 96
LONG_ATTN_FACTOR = 2.0
EXTRAS_ASSUMED = {"check": 2, "push": 60, "push-big": 120, "build-long": 60, "qwen3-predict": 300,
                  "ncu-subsample": 5}


def long_rows(n: int = LONG_PROMPTS) -> list[dict]:
    """The long set's (input, output) lengths for pricing: n lengths evenly over [LONG_MIN, LONG_MAX]."""
    return [{"input_len": int(LONG_MIN + (LONG_MAX - LONG_MIN) * i / max(1, n - 1)), "max_tokens": LONG_OUT}
            for i in range(n)]


def extras_plan(levels: str = "1,8,24,48,80,112,256,1024x3", n: int = 8, budget_s: float = 9000,
                long_context: bool = True) -> dict:
    """extras.sh's steps in order, [(block, step, seconds, basis)], the per-block totals and the budget's reading."""
    import ncu_bytes as NB
    model = "ernie-4.5-21b-a3b"
    out = []

    def add(block, step, sec, basis):
        out.append((block, step, float(sec), basis))

    _eb, work = ernie_scale(model)
    _qb, qwork = ernie_scale(QWEN3)
    sv = serving(levels, n)
    calls_ratio = sv["steps"] * PL.MODELS[model].moe_layers / (SMOKE_STEPS * SMOKE_LAYERS)
    engine, qengine = SMOKE_ENGINE_S * work, SMOKE_ENGINE_S * qwork
    chk = EXTRAS_ASSUMED["check"]
    add("X0", "extras-main-finished", chk, "trivial")
    add("X0", "extras-commit-switch", chk, "trivial (git)")
    add("X0", "env-preflight", ASSUMED["env-preflight"], "assumed (as main's, on ERNIE)")
    add("X0", "env", SMOKE_S["env"], "smoke")
    add("X0", "power-readback", 1, "smoke")
    add("X0", "unlock-start", 1, "smoke")
    add("X0", "revision-pinned-qwen3", ASSUMED["revision-pinned"], "assumed")
    add("X0", "price-check-qwen3", SMOKE_S["price-check"], "smoke (CPU, v8)")
    # X1: main's A1710, again
    add("X1", "X1-budget", chk, "trivial")
    add("X1", "X1-inputs", chk, "trivial")
    add("X1", "X1-lock", 1, "trivial")
    add("X1", "X1-capture", engine + sv["serve_s_default"] * clock_factor(1710), "as main's A1710-capture")
    add("X1", "X1-unlock-after", 1, "trivial")
    add("X1", "X1-export", SMOKE_S["nsys-export"] * calls_ratio, "as main's A1710-export")
    add("X1", "X1-extract", SMOKE_S["extract-gate-A"] * calls_ratio, "as main's A1710-extract")
    add("X1", "X1-push", EXTRAS_ASSUMED["push-big"], "assumed")
    # X2: the counter route on 2 x NB.N_PER_BRANCH cells
    add("X2", "X2-budget", chk, "trivial")
    add("X2", "X2-inputs", chk, "trivial")
    add("X2", "X2-subsample", EXTRAS_ASSUMED["ncu-subsample"], "assumed")
    for b in ("decode", "prefill"):
        add("X2", f"X2-lock-{b}", 1, "trivial")
        add("X2", f"X2-ncu-{b}", NB.expected_s(NB.N_PER_BRANCH),
            f"child start {NB.SETUP_S:.0f} s (assumed) + {NB.N_PER_BRANCH} cells x {NB.CALLS} calls x 2 GEMMs x "
            f"{NB.SEC_PER_LAUNCH} s per profiled launch (RR's counter route on the GH200)")
        add("X2", f"X2-unlock-{b}", 1, "trivial")
    add("X2", "X2-push", EXTRAS_ASSUMED["push"], "assumed")
    # X3: Qwen3-30B-A3B
    qsv = serving(levels, n, model=QWEN3)
    qcalls = qsv["steps"] * PL.MODELS[QWEN3].moe_layers / (SMOKE_STEPS * SMOKE_LAYERS)
    gate_calls = 60 / SMOKE_SERVE_S * PL.MODELS[QWEN3].moe_layers / SMOKE_LAYERS
    add("X3", "X3-budget", chk, "trivial")
    add("X3", "X3-inputs", chk, "trivial")
    add("X3", "X3-hf-preflight", 40, "assumed (P8, P10, P12's timed download <= 20 s)")
    add("X3", "X3-prompts-fetch", SMOKE_S["prompts-fetch"], "smoke (main's file reused when present)")
    add("X3", "X3-prompts-build", SMOKE_S["prompts-build"] * 4096 / SMOKE_PROMPTS, "smoke x 4096/1024 prompts")
    add("X3", "X3-unlock-gate", 1, "trivial")
    add("X3", "X3-gate60-capture", qengine + 60 + QWEN3_WEIGHTS_GB * 1e3 / DOWNLOAD_MBPS,
        f"smoke engine start x {qwork:.2f} (Qwen3 bytes x 48/16 layers) + 60 s + {QWEN3_WEIGHTS_GB} GB at an assumed "
        f"{DOWNLOAD_MBPS:.0f} MB/s")
    add("X3", "X3-gate60-export", SMOKE_S["nsys-export"] * gate_calls, "smoke x 60/44 s x 48/16 layers")
    add("X3", "X3-gate60-extract", SMOKE_S["extract-gate-A"] * gate_calls, "smoke x 60/44 s x 48/16 layers")
    add("X3", "X3-gate60-sequences", SMOKE_S["sequences-A"], "smoke")
    for lab, lock in (("A1710", 1710), ("A1410", 1410)):
        add("X3", f"X3-{lab}-lock", 1, "trivial")
        add("X3", f"X3-{lab}-capture", qengine + qsv["serve_s_default"] * clock_factor(lock),
            f"engine start + {qsv['steps']} steps x smoke step time x {qwork:.2f} x {clock_factor(lock):.3f} (clock)")
        add("X3", f"X3-{lab}-unlock-after", 1, "trivial")
        add("X3", f"X3-{lab}-export", SMOKE_S["nsys-export"] * qcalls, f"smoke x calls {qcalls:.2f}")
        add("X3", f"X3-{lab}-extract", SMOKE_S["extract-gate-A"] * qcalls, f"smoke x calls {qcalls:.2f}")
    add("X3", "X3-push-captures", EXTRAS_ASSUMED["push-big"], "assumed")
    for b in ("decode", "prefill"):
        add("X3", f"X3-replay-sample-{b}", 3, "smoke (replay-sample)")
        add("X3", f"X3-replay-{b}", QWEN3_REPLAY_BUDGET_S / RP.WALL_SAFETY + REPLAY_STEP_OVERHEAD_S,
            f"{QWEN3_REPLAY_BUDGET_S} s budget / {RP.WALL_SAFETY} (main's ERNIE probe walls x the bytes ratio, right)")
    add("X3", "X3-unlock-after-replay", 1, "trivial")
    add("X3", "X3-push-replay", EXTRAS_ASSUMED["push"], "assumed")
    ft = frontier_timer(model=QWEN3)
    add("X3", "X3-frontier-env", chk, "trivial (main's Frontier venv)")
    for b in ("decode", "prefill"):
        add("X3", f"X3-frontier-lock-{b}", 1, "trivial")
        add("X3", f"X3-frontier-timer-{b}", ft[b]["timer_s"], f"as main's M7 on Qwen3's call times ({ft[b]['points']} "
                                                               "points x 3 samples)")
        add("X3", f"X3-frontier-export-{b}", ft[b]["export_s"], f"smoke export x kernels ({ft[b]['calls']} calls)")
    add("X3", "X3-frontier-unlock", 1, "trivial")
    add("X3", "X3-frontier-points", ASSUMED["frontier-points"], "assumed")
    add("X3", "X3-frontier-csv", ASSUMED["frontier-csv"], "assumed")
    add("X3", "X3-frontier-train", ASSUMED["frontier-train"], "assumed (as main's)")
    add("X3", "X3-frontier-csv-fills", ASSUMED["frontier-csv"], "assumed")
    add("X3", "X3-frontier-train-fills", ASSUMED["frontier-train"], "assumed (as main's)")
    add("X3", "X3-frontier-predict", EXTRAS_ASSUMED["qwen3-predict"], "assumed (two models, two runs)")
    add("X3", "X3-push-frontier", EXTRAS_ASSUMED["push"], "assumed")
    if long_context:
        lsv = serving(LONG_LEVELS, LONG_N, rows=long_rows(), step_factor=LONG_ATTN_FACTOR)
        lcalls = lsv["steps"] * PL.MODELS[model].moe_layers / (SMOKE_STEPS * SMOKE_LAYERS)
        add("X4", "X4-budget", chk, "trivial")
        add("X4", "X4-inputs", chk, "trivial")
        add("X4", "X4-prompts-fetch", SMOKE_S["prompts-fetch"], "smoke")
        add("X4", "X4-prompts-build", EXTRAS_ASSUMED["build-long"], "assumed (~1M tokens tokenized)")
        add("X4", "X4-lock", 1, "trivial")
        add("X4", "X4-capture", engine + lsv["serve_s_default"] * clock_factor(1410),
            f"engine start + {lsv['steps']} steps (levels {LONG_LEVELS}, N {LONG_N}, inputs {LONG_MIN}..{LONG_MAX}) x "
            f"smoke step time x 3.16 x {LONG_ATTN_FACTOR} (long attention, assumed) x {clock_factor(1410):.3f}")
        add("X4", "X4-unlock-after", 1, "trivial")
        add("X4", "X4-export", max(10.0, SMOKE_S["nsys-export"] * lcalls), f"smoke x calls {lcalls:.2f}")
        add("X4", "X4-extract", max(10.0, SMOKE_S["extract-gate-A"] * lcalls), f"smoke x calls {lcalls:.2f}")
        add("X4", "X4-push", EXTRAS_ASSUMED["push"], "assumed")
    add("XF", "extras-push-final", EXTRAS_ASSUMED["push-big"], "assumed")
    add("XF", "extras-held-back-check", 2, "trivial")
    blocks: dict = {}
    for blk, _s, sec, _b in out:
        blocks[blk] = blocks.get(blk, 0.0) + sec
    total = sum(blocks.values())
    run, fits = blocks.get("X0", 0.0), {}
    for blk in ("X1", "X2", "X3", "X4"):
        if blk in blocks:
            fits[blk] = run + blocks[blk] <= budget_s
            run += blocks[blk] if fits[blk] else 0.0
    return {"steps": out, "blocks": blocks, "total_s": total, "total_h": total / 3600, "budget_s": float(budget_s),
            "fits": fits, "qwen3_serving": qsv, "long_serving": lsv if long_context else None, "levels": levels, "n": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="1,8,24,48,80,112,256,1024x3")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--extras", action="store_true", help="extras.sh's plan (X0..X4)")
    ap.add_argument("--budget-s", type=float, default=9000)
    ap.add_argument("--no-long-context", action="store_true")
    a = ap.parse_args(argv)
    if a.extras:
        p = extras_plan(a.levels, a.n, a.budget_s, long_context=not a.no_long_context)
        if a.json:
            print(json.dumps(p, indent=1))
            return 0
        for blk, step, sec, basis in p["steps"]:
            print(f"PLAN {blk:<3s} {step:<30s} {sec:8.0f} s  {basis}")
        print("PLAN extras blocks: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in p["blocks"].items()))
        print("PLAN extras budget " + f"{p['budget_s']:.0f} s from extras.sh's start (X0 counted): " + ", ".join(
            f"{k} {'runs' if v else 'SKIPPED by the budget'}" for k, v in p["fits"].items()))
        print(f"PLAN EXTRAS EXPECTED TOTAL {p['total_s']:.0f} s = {p['total_h']:.2f} h")
        return 0
    p = plan(a.levels, a.n)
    if a.json:
        print(json.dumps(p, indent=1))
        return 0
    for blk, step, sec, basis in p["steps"]:
        print(f"PLAN {blk:<3s} {step:<30s} {sec:8.0f} s  {basis}")
    print("PLAN per tile (estimator): " + ", ".join(f"{k} {v}" for k, v in p["serving"]["per_tile"].items()))
    print("PLAN blocks: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in p["blocks"].items()))
    fr = p["frontier"]
    print(f"PLAN Frontier: {fr['points']} of {fr['grid_points']} grid points (the full grid) x 3 samples = "
          f"{fr['samples']} timed samples, timer + export {fr['timer_s']:.0f} s")
    print(f"PLAN EXPECTED TOTAL {p['total_s']:.0f} s = {p['total_h']:.2f} h (levels {p['levels']}, N {p['n']})")
    print(f"PLAN WORST CASE at the {DOWNLOAD_FLOOR_MBPS:.0f} MB/s download floor (env-preflight P12): "
          f"{p['worst_download_total_s']:.0f} s = {p['worst_download_total_h']:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
