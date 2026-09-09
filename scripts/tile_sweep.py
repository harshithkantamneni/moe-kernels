#!/usr/bin/env python
"""Does forcing a bigger tile buy anything, now that we know 16 gives up WGMMA?

    python scripts/tile_sweep.py
    python scripts/tile_sweep.py --model deepseek-v3 --tokens 16,64,256
    python scripts/tile_sweep.py --dump-ptx /workspace/ptx-tiles   # verify the ISA

WHY THIS EXISTS. `check_mma_path.sh` measured, on a real deepseek-v3 T=16 cell,
that vLLM emits zero `wgmma` and 32 `mma.sync.aligned.m16n8k16`. That 16 is not a
tuned choice, and saying so matters: deepseek-v3 is E=256,N=2048 and vLLM
v0.27.1 ships no config file for it on any card or dtype, so the cell fell to
`get_default_config`'s hardcoded bf16 ladder, M<=32 -> 16 / M<=96 -> 32 /
M<=512 -> 64 / else 128. Only 2 of the 8 (model x card) cells in this study have
a tuned file at all, both H200: E=8,N=14336 (mixtral-8x7b) and E=64,N=2560
(qwen2-57b-a14b), and in bf16 both of those pick 16 at the T=16 cell this was
measured on. Not "16 all the way up from M=1": mixtral's file reads
1:16 2:32 4:16 8:16 16:16 24:16 32:16 48:32, so it takes 32 at M=2 and again
from M=48, and check_mma_path.sh prints both ladders in full. Stated to the
entry rather than to a range because a ladder summarised as a range is how the
16-through-256 figure that belongs to E=128,N=512 got attached to a model that
never ran it. So at the measured cell the tile is 16 either way, by two
different mechanisms, and Hopper's warpgroup MMA needs BLOCK_M % 64 == 0. The
obvious question is whether declining it costs anything. Force 64, reach the
warpgroup instruction, and see.

THE PREDICTION, stated before the run so it can fail. If the study's thesis is
right and this regime is memory-bound, then:

  * 16 -> 32   both stay on mma.sync. Padded MACs double. Time should NOT move,
               because those MACs hide under the weight read.
  * 32 -> 64   the instruction should switch to wgmma. Time should still NOT
               improve, because the tensor core was never the constraint, and it
               may get WORSE as the larger accumulator costs occupancy.
  * 64 -> 128  more of the same, more occupancy lost.

A FLAT curve confirms the thesis. A curve that improves at 64 refutes it and
says vLLM's autotuner left performance on the table, which would be the more
interesting result and is worth wanting.

CONFOUND, named rather than hidden. BLOCK_SIZE_M sizes the register accumulator,
so changing it changes occupancy as well as the instruction. A time change is
therefore ambiguous between the two. Only a NULL result is clean evidence, which
is exactly what the thesis predicts, and is why this experiment is worth running
in this direction rather than the other.

Routing is uniform and T is small on purpose: max rows per expert stays far below
16, so every expert is one tile at every setting and weight traffic is identical
across the sweep. Without that the hot expert spills and traffic stops being
flat, which is the objection raised in GPU MODE against the original design.

THE APPARATUS, and what it replaced on 2026-09-02. Every cell is timed by
`moe.bench.timing.time_kernel` under `moe.bench.timing.TIMING_BASIS`: queue-deep,
one synchronise per trial, L2 flushed by default, the SM clock sampled WHILE the
trials run. This file used to carry a verbatim copy of
`block_m_crossing_sweep.time_call` -- events created inside the loop, a
synchronise after every iteration, no flush, no clock -- which is one of the six
copies the audit found (A7) and which exposes ~0.18 ms of host enqueue per
fused_experts call on the H200 pod. That prefix is roughly constant across the
tile settings compared here, so it did not reverse this experiment's answer; it
did make "flat" mean flat-including-a-host-prefix, which is not the claim.

The prediction above is now a scored GATE rather than a paragraph: C1 asks
whether any tile beats the first by more than the noise band, and V1 asks
whether the run measured enough distinct settings to have been able to say. Both
print one `RESULT:` line, the exit code comes from `moe.bench.exit_codes` over
the same gates, and `--dry-run` prints the design's MDE against the measured
timing spread before any GPU is rented.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from moe.bench import exit_codes, timing  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.roofline import reference_clock  # noqa: E402
from moe.reference.torch_ref import make_inputs  # noqa: E402
from moe.routing.distributions import sample_topk_ids  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec  # noqa: E402

#: Held fixed so only M varies, which is the whole design: any time difference
#: has to be attributable to BLOCK_SIZE_M and to nothing else. These values are
#: NOT copied from a shipped config. An earlier comment here said "taken from
#: vLLM's tuned H200 entry at batch 16", and that was wrong: of the two study
#: shapes with a tuned bf16 H200 file, E=8,N=14336 holds
#: {M 16, N 64, K 256, GROUP 16, warps 4, stages 3} at batch 16 and E=64,N=2560
#: holds {M 16, N 128, K 128, GROUP 16, warps 4, stages 5}, so only BLOCK_SIZE_N
#: agrees with either. This sweep therefore prices the tile in isolation and does
#: not reproduce what vLLM would run at any batch. num_warps=8 is the
#: load-bearing entry: Triton takes the warpgroup path only when
#: BLOCK_M % 64 == 0 AND num_warps % 4 == 0, so pinning warps at 8 satisfies the
#: warp half at every setting and leaves BLOCK_SIZE_M as the only thing that can
#: flip the instruction.
FIXED = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
         "num_warps": 8, "num_stages": 4}


def find_override():
    """vLLM's own tuning hook: try_get_optimal_moe_config consults get_config()
    first, and a truthy value bypasses both the tuned file and the default.

    Probed rather than assumed, because the import path has moved between
    versions and a wrong guess would silently sweep nothing.
    """
    candidates = [
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.config",
    ]
    import importlib
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, name
    raise SystemExit(
        "could not find vllm's override_config in any of:\n  "
        + "\n  ".join(candidates)
        + "\nCheck the installed vLLM version; try_get_optimal_moe_config reads "
          "it via get_config(), so the hook exists under some name.")


def arm_ptx_dump(directory: Path) -> None:
    """Make Triton write its generated PTX, and guarantee it recompiles.

    Both halves matter. TRITON_KERNEL_DUMP/TRITON_DUMP_DIR ask for the dump, but
    Triton does not recompile a kernel it has already cached, and a cache hit
    dumps nothing. Pointing TRITON_CACHE_DIR at a fresh directory forces every
    specialisation to be built, so every tile setting produces a file.

    Called before vLLM is imported, since Triton reads these at compile time and
    the first compile happens on the first fused_experts call.
    """
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(directory)
    os.environ["TRITON_CACHE_DIR"] = str(directory / "_cache")


def scan_new_ptx(directory: Path, seen: set[Path]) -> tuple[int, int, list[str]]:
    """Instructions in PTX files that appeared since the last call.

    Returns (wgmma count, mma.sync count, distinct shapes). Each BLOCK_SIZE_M is
    a different Triton specialisation and therefore a different cache entry, so
    the files that appear after a setting ran belong to that setting.
    """
    import re
    fresh = [q for q in directory.rglob("*.ptx") if q not in seen]
    seen.update(fresh)
    w = m = 0
    shapes: set[str] = set()
    for q in fresh:
        try:
            text = q.read_text(errors="ignore")
        except OSError:
            continue
        w += len(re.findall(r"wgmma\.", text))
        m += len(re.findall(r"mma\.sync\.", text))
        # Not `wgmma\.aligned`: the real mnemonics are
        #   mma.sync.aligned.m16n8k16...      and
        #   wgmma.mma_async.sync.aligned.m64n128k16...
        # so the shape is reached through a variable middle section.
        shapes.update(re.findall(
            r"(?:wgmma|mma\.sync)[a-z0-9_.]*?\.m\d+n\d+k\d+", text))
    return w, m, sorted(shapes)


def _make_call(fused_experts, x, weights, w, ids, kw):
    """Bind the arguments explicitly rather than closing over loop variables.

    The closure was safe here because it is invoked inside the same iteration,
    but a `def` inside a loop that captures the loop's variables is one refactor
    away from a silent late-binding bug, and ruff B023 is right to flag it.
    """
    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=w, topk_ids=ids, **kw)
    return call


#: The timing spread this design is sized against, as a fraction of one cell's
#: time. MEASURED, not assumed: the median and the maximum of
#: `timing_spread_median` over the 26 published `*.report.json` files under
#: `results/published`, which is every arm in this repository that records one
#: (min 0.0039, median 0.0077, max 0.0182 on 2026-09-02).
MEASURED_SPREAD_MEDIAN = 0.0077
MEASURED_SPREAD_MAX = 0.0182

#: Two-sided 5% at 80% power, the convention `replicate_noise_floor` uses for
#: every MDE it prints.
MDE_LEVEL = 0.05
MDE_POWER = 0.80

#: C1's band. A tile counts as FASTER than the reference only when it beats it
#: by more than this fraction. Set to the MDE this design has at the worst
#: measured spread rather than to a round number, so the gate cannot claim to
#: have resolved something smaller than the box can show; `render_mde` prints
#: the arithmetic and `--dry-run` prints it before anything is rented.
def improvement_band(trials: int) -> float:
    """The smallest speedup this design can call a speedup.

    One p50 per setting, compared as a ratio against the first setting, so the
    quantity that has to clear the noise is a difference of two log times and a
    ratio inherits both spreads: hence the sqrt(2). Sigma is imported from the
    published corpus rather than estimated inside the run, which makes this a
    known-variance z test -- the only form evaluable at one measurement per
    setting, and the stricter of the two available.
    """
    return mde_of_ratio(MEASURED_SPREAD_MAX, max(1, trials))


def mde_of_ratio(spread: float, reps: int) -> float:
    """Smallest ratio-versus-the-first-tile this design can resolve."""
    if reps < 1:
        raise ValueError(f"an MDE needs at least one repeat, got {reps}")
    if spread <= 0:
        raise ValueError(f"an MDE needs a positive spread, got {spread}")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1.0 - MDE_LEVEL / 2.0) + normal.inv_cdf(MDE_POWER)
    return z * spread * math.sqrt(2.0 / reps)


#: The columns `timing.KernelTiming` contributes to every measured row.
TIMING_CSV_COLUMNS = ("instrument", "warmup_ms", "iters", "trials",
                      "sm_clock_load_mhz", "sm_clock_start_mhz",
                      "sm_clock_end_mhz", "clock_samples_mhz", "power_w",
                      "clock_level_ok", "clock_level_side",
                      "clock_drift_ok", "l2_flush", "host_bound")

#: NOT one of `TIMING_CSV_COLUMNS`, because `KernelTiming` does not carry it:
#: it is the number LEVEL was scored AGAINST, written on the row it scored so a
#: replay can tell a row that PASSED the flag from one that had nothing to be
#: level against. The tri-state alone cannot: both read empty.
CSV_COLUMNS = ("model", "num_tokens", "block_size_m", "active_experts",
               "max_rows_on_one_expert", "ms_p50", "ms_p90", "ms_min",
               "ms_stdev", "ratio_vs_first", "isa_note", "error",
               *TIMING_CSV_COLUMNS, "reference_clock_mhz",
               *PV.Provenance().as_columns())


def _flag(value: bool | None) -> str:
    """A tri-state flag as a CSV cell: "1", "0", or empty for NOT DETERMINED."""
    return "" if value is None else str(int(value))


def _mhz(value: float | None) -> str:
    """An Optional clock as a CSV cell. Empty is NOT DETERMINED, never 0 MHz."""
    return "" if value is None else f"{value:.0f}"


def clock_side_of(t) -> str:
    """Which side of the LEVEL band the instrument saw a cell on.

    `timing.LEVEL_LOW`, `timing.LEVEL_HIGH`, or "" for level or undetermined.
    The record's own `clock_level_side` is preferred. A record that failed
    LEVEL and carries no side (a fake built before the field existed on
    2026-09-03) has it derived from its own load and reference, the rule
    `moe.bench.driver` applies to the same records; one with neither answers
    "", which is "no side recorded" and not "level".

    SINCE 2026-09-09 THE SIDE IS A RECORD AND NOT A FILTER. `clock_excluded`
    below reads DRIFT alone; the side is written on the row, printed beside the
    fixed-roof fraction, and is what `roof_at_cell_clock` is scored from.
    """
    from moe.bench import timing

    side = getattr(t, "clock_level_side", "") or ""
    if not side and getattr(t, "clock_level_ok", None) is False:
        side = timing.level_side(getattr(t, "sm_clock_load_mhz", None),
                                 getattr(t, "reference_clock_mhz", None)) or ""
    return side


def clock_samples_of(t) -> dict:
    """The under-load clock evidence a DRIFT verdict rests on, as row columns.

    THE FIRST AND LAST SAMPLE WERE COMPUTED AND THROWN AWAY. `time_kernel` has
    put `sm_clock_start_mhz` and `sm_clock_end_mhz` on every `KernelTiming`
    since the clock-under-load instrument landed, and six of the seven writers
    in this repo kept only the median. The 2026-09-09 H200 session therefore
    ended with 135 rows that say DRIFT and cannot say which way the clock went:
    "the governor was still settling after a workload change" had to be argued
    from where the drifted cells sat in each rep rather than from the cells.
    Persisted from here on so the next session can be read off its own rows.

    Every field is fetched with `getattr` because the instrument gained
    `clock_samples_mhz` and `power_w` after these rows first existed: a record
    without them writes the column EMPTY, which is NOT DETERMINED and never
    zero. The sample list is space-joined integers, one representation that
    serves a CSV cell and a JSON value alike.
    """
    samples = getattr(t, "clock_samples_mhz", None) or ()
    return {
        "sm_clock_start_mhz": getattr(t, "sm_clock_start_mhz", None),
        "sm_clock_end_mhz": getattr(t, "sm_clock_end_mhz", None),
        "clock_samples_mhz": " ".join(f"{c:.0f}" for c in samples),
        "power_w": getattr(t, "power_w", None),
    }


def clock_excluded(level_ok: bool | None, side: str,
                   drift_ok: bool | None) -> bool:
    """Do a cell's clock verdicts exclude it. DRIFT does; no LEVEL side does.

    THE RULE CHANGED ON 2026-09-09. Until then this file excluded a cell
    whose LEVEL failed LOW, and before 2026-09-08 one whose LEVEL failed at
    all, which took a memory-shaped cell boosted to 1980 MHz for one that had
    run cold. The 750-cell census of the H200 gaps session settled what the LOW
    side is. Under the 700 W cap the under-load clock is an OUTCOME of the
    cell, set per tile by the kernel's own power draw: BLOCK_M=128 at
    BLOCK_N=64 sat at 1380-1410 MHz in every rep and every tread, BLOCK_M=256
    at 1620-1755, memory-shaped cells at 1950-1980, against a calibration
    GEMM that itself held 1485 MHz at 691 W, near the LOW end of what dense
    work does on this card. A band around that GEMM's operating point therefore
    excludes a TILE and not a defect: it dropped 148 cells session-wide,
    every one of them the steady state of one of the two tile families this
    study is about, and it would drop the same ones on every rerun.

    DRIFT survives, because it says something else: the clock MOVED while the
    cell was timed, so the median load is a blend of two clocks and the time
    is not a time at one operating point. All 135 drifts in that session were
    the governor settling on the first cell of a rep after a workload change,
    which is an instrument problem and is fixed at the instrument.

    `level_ok` and `side` are still taken and still written on the row. The
    side is a RECORD of where the cell ran, printed beside the fixed-roof
    fraction and used for `roof_at_cell_clock`, and it excludes nothing. None
    is not determined and an exclusion has to be positively established, so
    only a False DRIFT excludes.
    """
    return drift_ok is False


def clock_state(rows: list[dict]) -> dict:
    """How many timed settings sat where against the roof's clock.

    Counts and never a verdict: C1 reads a flat curve and this sweep drops no
    row for its clock, so the block is what a reader has to decide whether a
    flat curve was timed at one clock. `low` and `drift` are the excluded-
    shaped states `clock_excluded` names; `high` is kept and counted apart,
    because on the H200 it is the ordinary state of a memory-bound cell and
    the decode cells this sweep times are memory-bound by design. `unknown`
    is the rows whose LEVEL was not determined, separate because a run that
    could not read its clocks and a run whose clocks were fine are not the
    same state. Reads the CSV cell strings `timing_columns` wrote, so a replay
    of cells.csv and the live run count the same thing.
    """
    timed = [r for r in rows if not r.get("error")]
    level = [r.get("clock_level_ok", "") for r in timed]
    sides = [r.get("clock_level_side", "") or "" for r in timed]
    drift = [r.get("clock_drift_ok", "") for r in timed]
    low = sum(1 for ok, s in zip(level, sides, strict=True)
              if ok == "0" and s != timing.LEVEL_HIGH)
    high = sum(1 for ok, s in zip(level, sides, strict=True)
               if ok == "0" and s == timing.LEVEL_HIGH)
    return {
        "timed": len(timed),
        "level": sum(1 for ok in level if ok == "1"),
        "low": low,
        "high": high,
        "drift": sum(1 for d in drift if d == "0"),
        "unknown": sum(1 for ok in level if ok == ""),
        "excluded_shaped": sum(
            1 for ok, s, d in zip(level, sides, drift, strict=True)
            if clock_excluded(None if ok == "" else ok == "1", s,
                              None if d == "" else d == "1")),
        "rule": "DRIFT excludes; BOTH LEVEL sides are kept with the side "
                "recorded, because the under-load clock is set per tile by the "
                "kernel's own power draw under the cap; the fixed roof is what "
                "the compute-bound gates score against and roof_at_cell_clock "
                "is printed beside it; this sweep drops no row for its clock",
    }


def clock_state_lines(state: dict) -> list[str]:
    """The printed form of `clock_state`, saying which side each count is."""
    return [
        f"clock state: {state['timed']} timed rows: {state['level']} level, "
        f"{state['low']} steady LOW (kept, side recorded), "
        f"{state['high']} steady HIGH (kept, side recorded), "
        f"{state['drift']} DRIFT failed (excluded-shaped), "
        f"{state['unknown']} with LEVEL not determined",
        "  scored against the fixed roof, with roof_at_cell_clock printed "
        "beside it as issue efficiency; this sweep drops no row for its "
        "clock, and an empty flag means NOT DETERMINED, never fine",
    ]


def timing_columns(t) -> dict:
    """The `KernelTiming` fields that have to reach `cells.csv`.

    Columns rather than a note, because the question a reader asks of a flat
    curve is "was every point at the same clock", and a note cannot be filtered.
    None stays None: an empty cell means NOT DETERMINED, never False.
    `clock_level_side` is the tri-state "low" / "high" / "" beside the LEVEL
    verdict, because since 03df2d4 a False can be a boost above the reference
    and a reader of the CSV has to be able to tell it from a sag.
    """
    samples = clock_samples_of(t)
    return {"instrument": t.instrument, "warmup_ms": f"{t.warmup_ms:.1f}",
            "iters": t.iters, "trials": t.trials,
            "sm_clock_load_mhz": ("" if t.sm_clock_load_mhz is None
                                  else f"{t.sm_clock_load_mhz:.0f}"),
            "sm_clock_start_mhz": _mhz(samples["sm_clock_start_mhz"]),
            "sm_clock_end_mhz": _mhz(samples["sm_clock_end_mhz"]),
            "clock_samples_mhz": samples["clock_samples_mhz"],
            "power_w": ("" if samples["power_w"] is None
                        else f"{samples['power_w']:.1f}"),
            "clock_level_ok": _flag(t.clock_level_ok),
            "clock_level_side": clock_side_of(t),
            "clock_drift_ok": _flag(t.clock_drift_ok),
            "l2_flush": _flag(t.l2_flush),
            "host_bound": _flag(t.host_bound)}


@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    UNKNOWN is never printed as a pass: a gate that could not be evaluated has
    not passed, and on a VALIDITY gate that means the page is not quotable. The
    exit code comes from `exit_codes.classify` over these, so the log and the
    code cannot disagree.
    """

    name: str
    kind: str
    prediction: str
    rule: str
    verdict: str
    observed: str

    def result_line(self) -> str:
        return exit_codes.result_line(self.kind, self.name.split()[0],
                                      self.verdict, self.observed[:160])

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.name.split()[0], self.verdict)

    def render(self, with_result: bool = True) -> str:
        out = [self.result_line()] if with_result else []
        out += [f"[{self.verdict:7s}] {self.kind:8s} {self.name}  "
                f"{self.prediction}",
                f"                    gate: {self.rule}",
                f"                    saw:  {self.observed}"]
        return "\n".join(out)


def build_gates(rows: list[dict], band: float) -> list[Gate]:
    """V1 then C1, over the rows that actually produced a time.

    V1 IS THE NON-VACUITY GATE and it is not decoration. A sweep in which every
    setting but one failed to compile would print a perfectly flat curve, and a
    flat curve is what this experiment reads as confirmation. Two settings are
    the arithmetic minimum for a comparison, so fewer is UNKNOWN rather than a
    pass, and a run in that state cannot reach DONE.
    """
    timed = [r for r in rows if r.get("ms_p50")]
    gates = [Gate(
        "V1 comparable", exit_codes.VALIDITY,
        "at least two tile settings produced a time to compare",
        "timed settings >= 2",
        exit_codes.PASS if len(timed) >= 2 else exit_codes.FAIL,
        f"{len(timed)} of {len(rows)} planned settings produced a time"
        + (f"; failures: {[r['block_size_m'] for r in rows if r.get('error')]}"
           if any(r.get('error') for r in rows) else ""))]

    if len(timed) < 2:
        gates.append(Gate(
            "C1 flat", exit_codes.CLAIM,
            "no tile setting beats the smallest by more than the noise band",
            f"min(ratio vs first) > 1 - {band:.3f}", exit_codes.UNKNOWN,
            "fewer than two settings were timed, so nothing was compared"))
        return gates

    best = min(timed, key=lambda r: r["ratio_vs_first"])
    improved = best["ratio_vs_first"] < 1.0 - band
    gates.append(Gate(
        "C1 flat", exit_codes.CLAIM,
        "no tile setting beats the smallest by more than the noise band",
        f"min(ratio vs first) > 1 - {band:.3f}, the MDE at the worst measured "
        f"timing spread",
        exit_codes.FAIL if improved else exit_codes.PASS,
        f"best is BLOCK_SIZE_M={best['block_size_m']} at "
        f"{best['ratio_vs_first']:.3f}x the first setting"
        + (" -- an improvement the thesis does not allow"
           if improved else " -- inside the band")))
    return gates


def render_mde(trials: int, spreads=None) -> str:
    """What this design can see, printed BEFORE the box is rented.

    B14: no arm in this study stated a minimum detectable effect, so C1 could
    "confirm" a flat curve without anyone knowing how big a step it could have
    seen. `spreads` overrides the measured pair, which is how the
    CANNOT-RESOLVE branch is planted in the tests.
    """
    spreads = spreads or (("median", MEASURED_SPREAD_MEDIAN),
                          ("worst", MEASURED_SPREAD_MAX))
    lines = ["## What this design can see (MDE)", "",
             "Effect under test: a speedup at BLOCK_SIZE_M >= 64, which would "
             "refute the thesis.",
             "Noise assumption: per-cell timing spread, MEASURED over the 26 "
             f"published reports; median {MEASURED_SPREAD_MEDIAN:.2%}, worst "
             f"{MEASURED_SPREAD_MAX:.2%}.",
             f"Design: one time_kernel p50 per setting over {trials} "
             "queue-deep trials, compared as a ratio against the first.", ""]
    for label, spread in spreads:
        mde = mde_of_ratio(spread, trials)
        lines.append(f"  at the {label:<7} spread {spread:.2%}:  MDE "
                     f"{mde:.3f}"
                     + ("   a 1% step would be INVISIBLE" if mde > 0.01
                        else "   resolves a 1% step"))
    lines += ["",
              "C1's band is the WORST-spread number above. A flat curve at a "
              "band of 0.05 says",
              "less than a flat curve at a band of 0.01, and the band is "
              "printed beside the verdict."]
    return "\n".join(lines)


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    The same order `scripts/run_all.sh` resolves it in. A pod's container disk
    dies with the pod and the network volume does not, so a results path that
    defaults to the checkout is a results path that defaults to being lost.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


class NoCardToLabel(ValueError):
    """No card could be named, and none will be invented.

    The run id, the output directory and every row carry a card. There is no
    default: a laptop run labelled `NVIDIA H200` looks exactly like a pod run
    in `ls` and in the CSV, and this script's whole answer is a per-card claim
    about which MMA instruction the tile reaches.
    """


def resolve_card(args) -> str:
    """`--card`, else the live device, else refuse.

    Refusing is cheap here: this script cannot measure anything without a GPU
    anyway, and `--dry-run` is the mode that runs off one. What the refusal
    prevents is the third state -- a machine that has a GPU whose name could
    not be read -- producing rows attributed to nothing.
    """
    if args.card:
        return str(args.card)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        if name:
            return str(name)
    raise NoCardToLabel(
        "no --card and no readable CUDA device name. Pass --card as nvidia-smi "
        "spells it; the run id, the output directory and every row are named "
        "after it, and a wrong label is worse than no run.")


def reference_clock_for(card: str):
    """The clock THIS card's roof was measured at, which LEVEL is scored against.

    THE FLAG HAD NO LEFT-HAND SIDE HERE. `timing.clock_flags` leaves
    `clock_level_ok` None unless it is handed the clock the roof was measured
    at, and this file called `time_kernel` without one while writing a
    `clock_level_ok` column and branching on `t.clock_level_ok is False` below
    (since 2026-09-08 the branch is `clock_excluded`, which reads the side).
    The column was null on every row it has ever written and the branch was
    dead, so the sweep carried the apparatus for a verdict it could not reach.

    Resolved through `roofline.reference_clock` rather than as another copy of
    the three-field rule. Copies of that rule are how the driver came to believe
    no calibration recorded a clock while a sweep read 1515 MHz out of the
    committed yaml, and `roofline` is the copy `driver.RunConfig` and
    `group_m_alpha_sweep` already use.

    Returns the `ReferenceClock`. `mhz` None with `card` set is a pod holding a
    card whose ruler was never measured; `main` refuses on it, and
    `_no_reference_refusal` says why.
    """
    return reference_clock(card or None)


def _no_reference_refusal(card: str, ref) -> str:
    """The refusal text for an attached card with no clock to level against.

    THIS ARM REFUSES WHERE `group_m_alpha_sweep` RESOLVES-AND-SAYS, and the
    difference is not an inconsistency. That one runs as step 3 of
    `pod_session.sh`, which runs `calibrate_hardware.py --publish` at step 1 and
    treats a refusal there as fatal, so by the time it measures, the reference
    exists or the session has already stopped. This file is in no session
    script: it is launched by hand on a fresh pod, where a missing calibration
    is the normal state and nothing upstream has checked for one.

    And its claim is the one that needs the flag most. C1 reads a FLAT curve as
    confirmation. A card that sagged at one tile setting can manufacture that
    flatness or hide a real difference under it, and with `clock_level_ok` null
    on every row nothing in the report can tell either story from the other. A
    null result measured by an apparatus that cannot report a clock problem is
    not a conservative result, it is an unexamined one, and it costs the same
    rental as one that can.
    """
    return "\n".join([
        "=" * 72,
        "REFUSED. Nothing was measured.",
        f"  reason: the attached card {card!r} has no clock to level against: "
        f"{ref.source}.",
        "  Without a reference, timing.clock_flags leaves clock_level_ok None "
        "on every cell,",
        "  the clock_level_ok column is null and the branch that prints the "
        "clock note is dead,",
        "  so a FLAT curve could not be told from a curve flattened by a card "
        "that sagged.",
        "  Two ways out, both cheap: run "
        "`python scripts/calibrate_hardware.py --publish` on",
        "  this box, which measures this card's roof and writes the clock its "
        "dense GEMM ran",
        "  at into moe/bench/hardware/; or pass --card naming the card whose "
        "calibration this",
        "  run is to be scored against, and say in the write-up where the "
        "number came from.",
        "=" * 72,
    ])


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept and device-dependent knob, card first.

    A run id that omits a knob is how two settings come to share a directory
    and the second silently reports the first's numbers: the sibling sweep lost
    a whole arm to one that omitted GROUP_SIZE_M. The card is in it because it
    is swept by the operator moving to another pod while `$MOE_RESULTS_DIR` is
    a network volume that outlives the pod on purpose. The four timing knobs
    are in it because each sets the measured milliseconds of every row.

    `--dump-ptx` is NOT in the key: it adds a column to the report and does not
    change a time, so two runs that differ only there belong together.

    THE KEYS ARE NUMBERED because `run_id` renders them sorted into a name
    capped at 96 characters and truncates the tail. `FIXED` is five knobs whose
    rendered value is longer than everything else put together, and it has no
    flag behind it, so it sorts LAST: it stays in the hash, where a change to
    the constants sends the run to a new directory, and it is the first thing
    the visible name gives up.
    """
    return PV.run_id(card=card, **{
        "1model": args.model,
        "2tok": tuple(int(v) for v in args.tokens.split(",")),
        "3tile": tuple(int(v) for v in args.tiles.split(",")),
        "4seed": args.seed, "5it": args.iters, "6wm": args.warmup,
        "7bd": args.cell_budget_ms, "8tr": args.trials,
        "9flush": not args.no_l2_flush, "zfixed": FIXED})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v3", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--tokens", default="16,64,256")
    ap.add_argument("--tiles", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob on 2026-09-02 and kept in "
                         "the run id. moe.bench.timing.time_kernel sizes the "
                         "iteration count from --cell-budget-ms and the "
                         "warmup's own queue-deep per-call time, which is the "
                         "only sizing that holds a trial to a duration")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED 2026-09-02: the tile "
                         "settings compared here differ in occupancy, so they "
                         "reach the clock governor's operating point at "
                         "different call counts, and a fixed count warms them "
                         "unequally")
    ap.add_argument("--cell-budget-ms", type=float, default=200.0,
                    help="target duration of ONE trial")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per setting")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="time with L2 warm. The default FLUSHES, the opposite "
                         "of what this script did before 2026-09-02. The old "
                         "comment argued a flush 'adds its own variance', which "
                         "is true and is not the point: the roof and every "
                         "other arm in this study are measured flushed, and an "
                         "unflushed curve cannot be read against them")
    ap.add_argument("--card", default=None,
                    help="the card this run is for, as nvidia-smi names it. "
                         "Defaults to the live device; there is no static "
                         "default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"defaults to {results_root()}/tile_sweep")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the MDE and the cost, measure nothing")
    ap.add_argument("--dump-ptx", type=Path, default=None,
                    help="dump and count the emitted ISA per tile setting, so "
                         "the wgmma claim is verified rather than labelled")
    return ap


def _main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    tiles = [int(v) for v in args.tiles.split(",")]
    tokens = [int(v) for v in args.tokens.split(",")]

    try:
        card = resolve_card(args)
    except NoCardToLabel as exc:
        print("\n".join(["REFUSED. Nothing was planned and nothing was measured.",
                         f"  NoCardToLabel: {exc}"]))
        return exit_codes.REFUSED

    run_id = default_run_id(args, card)
    out_dir = (args.out_dir or (results_root() / "tile_sweep")) / run_id
    csv_path = out_dir / "cells.csv"
    report_path = out_dir / "report.json"
    band = improvement_band(args.trials)

    cfg = MODEL_CONFIGS[args.model]
    print("# Does forcing a bigger tile buy anything?")
    print(f"run id  {run_id}")
    print(f"card    {card}   instrument {timing.TIMING_BASIS}")
    print(f"model   {args.model}  E={cfg.num_experts} k={cfg.top_k}  "
          f"fixed {FIXED}")
    print(f"timing  {args.trials} queue-deep trials sized to "
          f"{args.cell_budget_ms:.0f} ms, after {args.warmup:.0f} ms of warmup, "
          f"L2 " + ("flushed" if not args.no_l2_flush else "WARM"))
    print(f"writes  {out_dir}")
    print()
    print(render_mde(args.trials))
    print()

    prov = PV.provenance_block(instrument=timing.TIMING_BASIS,
                               warmup_ms=args.warmup,
                               target_ms=args.cell_budget_ms)

    if args.dry_run:
        print("=" * 72)
        print("NOT A RESULT. Nothing was measured.")
        print("  reason: --dry-run was given")
        print(f"  {len(tokens)} token counts x {len(tiles)} tile settings = "
              f"{len(tokens) * len(tiles)} cells, each "
              f"{args.trials} x {args.cell_budget_ms:.0f} ms of trials plus "
              f"{args.warmup:.0f} ms of warmup;")
        print(f"  {len(tiles)} distinct Triton specialisations to compile.")
        print("  NO `RESULT:` LINE IS PRINTED HERE. Nothing was measured, so "
              "this exits REFUSED, and")
        print("  a REFUSED log carrying result lines would let the driver "
              "recompute DONE from them.")
        print("=" * 72)
        return exit_codes.REFUSED

    if not torch.cuda.is_available():
        print("\n".join(["=" * 72,
                         "REFUSED. Nothing was measured.",
                         "  reason: no CUDA device. --dry-run prints the plan, "
                         "the MDE and the cost.",
                         "=" * 72]))
        return exit_codes.REFUSED

    # RESOLVED ONCE PER RUN and BEFORE the first compile, not per cell: the
    # answer is a property of this box and the calibration on it, a per-cell
    # lookup would read a yaml off disk for every cell, and a refusal is only
    # worth anything while nothing has been spent. Past the CUDA check a card is
    # attached by definition, which is the state `driver.unreferenced_clock`
    # refuses in for the same reason.
    ref = reference_clock_for(card)
    print("clock   LEVEL reference: "
          + (f"{ref.mhz:.0f} MHz, {ref.source}" if ref.mhz else
             f"NOT RESOLVED ({ref.source})"))
    if ref.mhz is None:
        print(_no_reference_refusal(card, ref))
        return exit_codes.REFUSED
    print()

    # Before find_override(), which imports vLLM: Triton reads these at compile
    # time and the first compile happens on the first fused_experts call.
    if args.dump_ptx:
        arm_ptx_dump(args.dump_ptx)

    override_config, where = find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs

    # Across ALL token counts, not per block. Triton specialises the kernel on
    # the tile constants and T only sizes the grid at runtime, so each setting
    # compiles exactly once and every later token block is a legitimate cache
    # hit. Resetting per block made the first row of each later block re-count
    # every file the earlier blocks had produced.
    seen_ptx: set[Path] = set()
    isa_by_tile: dict[int, str] = {}
    rows: list[dict] = []
    print(f"override hook: {where}.override_config\n")

    for tok in tokens:
        spec = BenchSpec(cfg, num_tokens=tok, dtype="bf16",
                         routing=RoutingSpec("uniform", 0.0), seed=args.seed)
        x, weights = make_inputs(spec, device="cuda")
        ids = sample_topk_ids(spec.routing, tok, cfg.num_experts, cfg.top_k,
                              seed=args.seed, device="cuda")
        counts = torch.bincount(ids.flatten(), minlength=cfg.num_experts)
        active = int((counts > 0).sum())
        mx = int(counts.max())
        w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32, device="cuda")

        kw = vllm_call_kwargs(spec)
        kw["activation"] = MoEActivation(kw["activation"])

        print(f"T={tok}: {active} active experts, max {mx} rows on one expert"
              + ("   <-- WARNING: max >= smallest tile, traffic will not be flat"
                 if mx >= min(tiles) else ""))
        head = f"  {'BLOCK_SIZE_M':>13} {'ms p50':>10} {'stdev':>8} {'vs first':>9}   "
        head += "EMITTED" if args.dump_ptx else "note"
        print(head)
        base = None
        for bm in tiles:
            conf = dict(FIXED, BLOCK_SIZE_M=bm)
            call = _make_call(fused_experts, x, weights, w, ids, kw)
            row = {"model": args.model, "num_tokens": tok, "block_size_m": bm,
                   "active_experts": active, "max_rows_on_one_expert": mx,
                   **prov.as_columns()}
            with override_config(conf):
                try:
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=ref.mhz)
                except timing.TimingRefused:
                    # THE SECOND DOOR INTO THE SAME ROOM, and it was open.
                    # `TimingRefused` subclasses RuntimeError, so the handler
                    # below caught every refusal the INSTRUMENT ITSELF raises --
                    # no CUDA and no injected fakes, trials=0, a warmup that
                    # makes the measurement meaningless. Each of those is a fact
                    # about the RUN, identical for every cell, so this sweep
                    # wrote a zeroed FAILED row per cell, scored C1 UNKNOWN and
                    # V1 FAIL over a page of nothing, and reported a verdict on
                    # a run that measured not one setting. Re-raised to `main`,
                    # which exits REFUSED: nothing was measured and nothing was
                    # spent. `driver.run_cell` has the same clause for the same
                    # reason, and names the BASE class as this does, so a
                    # refusal added to the instrument later cannot reintroduce
                    # the bug by forgetting to add itself here.
                    raise
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    rows.append(row)
                    print(f"  {bm:13d} {'FAILED':>10}   {row['error']}")
                    continue
            base = base if base is not None else t.ms_p50
            row.update({"ms_p50": t.ms_p50, "ms_p90": t.ms_p90,
                        "ms_min": t.ms_min, "ms_stdev": t.ms_std,
                        "ratio_vs_first": t.ms_p50 / base, "error": "",
                        "reference_clock_mhz": f"{ref.mhz:.0f}",
                        **timing_columns(t)})
            if args.dump_ptx:
                # Each BLOCK_SIZE_M is a distinct Triton specialisation and so a
                # distinct cache entry, which is why files appearing after this
                # setting ran belong to it.
                w_n, m_n, shapes = scan_new_ptx(args.dump_ptx, seen_ptx)
                if shapes:
                    note = f"wgmma={w_n} mma.sync={m_n}  " + ",".join(shapes)
                    isa_by_tile[bm] = note
                elif bm in isa_by_tile:
                    # Same specialisation, already built and already counted.
                    note = isa_by_tile[bm] + "   [same kernel as an earlier T]"
                else:
                    note = ("no PTX emitted and none recorded for this tile; "
                            "dump not armed, or the kernel was cached before "
                            "TRITON_CACHE_DIR was redirected")
            else:
                # A label, not a measurement, and only correct for the default
                # 16/32/64/128 grid. The real predicate is BLOCK_M % 64 == 0 AND
                # num_warps % 4 == 0 (supportMMA, triton release/3.7.x
                # lib/Analysis/Utility.cpp), so `--tiles 80,96` would print
                # "reachable" here and still compile to mma.sync. On an A100 it
                # would be wrong at every tile, since getMMAVersionSafe offers
                # only {2} below compute capability 9.0. Use --dump-ptx to make
                # this column evidence.
                note = "mma.sync (M<64)" if bm < 64 else "wgmma reachable (M>=64)"
            row["isa_note"] = note
            rows.append(row)
            print(f"  {bm:13d} {t.ms_p50:10.4f} {t.ms_std:8.4f} "
                  f"{row['ratio_vs_first']:8.3f}x   {note}")
            # THE SIDE IS RECORDED HERE AND EXCLUDES NOTHING. Since
            # 2026-09-09 `clock_excluded` reads DRIFT alone, so DRIFT and
            # host-bound get the exclusion-shaped marker and BOTH LEVEL sides
            # are named as kept: every decode cell this sweep times is
            # memory-shaped and boosts above the reference, and a steady LOW
            # would be a hungry tile at its own operating point.
            if (clock_excluded(t.clock_level_ok, row["clock_level_side"],
                               t.clock_drift_ok) or t.host_bound):
                print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
            elif row["clock_level_side"]:
                print(f"  ^ kept (LEVEL {row['clock_level_side']} is "
                      f"recorded, not excluded): "
                      f"{t.clock_note or ''}".rstrip())
        print()

    clocks = clock_state(rows)
    for line in clock_state_lines(clocks):
        print(line)
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    gates = build_gates(rows, band)
    print("## Gates\n")
    for gate in gates:
        print(gate.render())
    print()
    print("READING IT. Flat across all four confirms the thesis: neither the")
    print("padded arithmetic nor the tensor-core instruction is on the critical")
    print("path, because the weight read is. An improvement at 64 refutes it,")
    print(f"and 'improvement' means more than {band:.1%}, which is what this")
    print("design can see at the worst spread the published arms recorded.")
    if args.dump_ptx:
        print(f"ISA counted per setting from {args.dump_ptx}. If wgmma stays 0 at")
        print("M>=64 then Triton is not reaching the warpgroup instruction even when")
        print("the tile allows it, which is a finding in its own right.")
    else:
        print("Re-run with --dump-ptx to verify the instruction actually changed;")
        print("without it the note column asserts nothing about what was emitted.")

    report_path.write_text(json.dumps(prov.stamp({
        "run_id": run_id, "card": card, "model": args.model,
        "tokens": tokens, "tiles": tiles, "fixed": FIXED,
        "improvement_band": band,
        "reference_clock_mhz": ref.mhz,
        "reference_clock_source": ref.source,
        "clock_state": clocks,
        "measured_spread_median": MEASURED_SPREAD_MEDIAN,
        "measured_spread_max": MEASURED_SPREAD_MAX,
        "cells": rows,
        "gates": [{"name": g.name, "kind": g.kind, "verdict": g.verdict,
                   "rule": g.rule, "observed": g.observed} for g in gates],
    }), indent=2, default=str))
    print(f"\ncells    {csv_path}")
    print(f"report   {report_path}")

    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def main(argv: list[str] | None = None) -> int:
    """`_main` with the two escapes that were exiting ONE, which is CLAIM_FAIL.

    AN UNPLANNED CRASH IS ERROR, WHICH IS THE ONLY RETRYABLE CODE. Left to
    propagate, an unexpected exception exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it is in
    FINISHED_CODES, the session driver records it, and it is never retried. A
    torch OOM three tiles into a sweep, or an import that drifted, would be
    filed as this experiment's registered answer to "does a bigger tile buy
    anything". ERROR (4) is outside FINISHED_CODES precisely so the driver can
    tell "the apparatus broke" from "the claim did not hold". The traceback is
    printed first and not swallowed, because a code without one tells an
    operator nothing about what to fix.

    A STRING `SystemExit` IS A REFUSAL, and that is the second half of the same
    defect. `raise SystemExit(<str>)` sets `SystemExit.code` to the STRING and
    leaves the interpreter to exit ONE as well; `find_override` is one of those,
    and it fires on a version skew before a single cell has been timed. REFUSED
    (2) is the table's word for "a precondition was not met and nothing was
    measured".

    A `TimingRefused` IS A REFUSAL TOO, and it is the one the per-cell handler
    now lets past it. It is the same fact for every cell, so nothing was
    measured and nothing was spent, which is REFUSED and not ERROR.
    `cli._main` catches the same base class around `driver.run_sweep` and exits
    the same code, so the two entry points agree.

    Caught here rather than at the raise sites so the contract holds for a
    caller of `main()` as well as for the CLI, and so a refusal added later
    cannot reintroduce the bug by forgetting the code.
    """
    try:
        return _main(argv)
    except timing.TimingRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = (exc.code if exc.code.startswith("REFUS")
                   else f"REFUSED: {exc.code}")
            print(msg, file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        print("ERROR: tile_sweep crashed before it could reach a verdict. This "
              "is the apparatus failing, not a claim failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
