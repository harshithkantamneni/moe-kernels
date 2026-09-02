#!/usr/bin/env python3
"""What does vLLM's FALLBACK config cost, measured on a shape that HAS a tuned one?

    python scripts/tuned_vs_fallback.py                       # the whole thing
    python scripts/tuned_vs_fallback.py --plan-only           # free, no GPU
    python scripts/tuned_vs_fallback.py --models mixtral-8x7b --tokens 1,32,256
    python scripts/tuned_vs_fallback.py --run-id abc123       # resume, idempotent

WHY THIS EXISTS. vLLM ships 327 tuned fused-MoE config files at v0.27.1 and they
are concentrated on H100 and H200. Everything they do not cover falls to
`get_default_config`, a hardcoded ladder whose whole M-dependence is
`M<=32 -> 16, M<=96 -> 32, M<=512 -> 64, else 128` plus four other knobs pinned
by two-branch rules. The study already knows that most realistic
(model, tensor-parallel shard, GPU) combinations land there -- 60 of 288 covered,
so 79.2% take the ladder -- and knows it from the shipped file names, which is a
statement about COVERAGE and not about COST. Nothing here or upstream has priced
the ladder. If it costs a couple of percent the coverage gap is a footnote; if it
costs 15% or more then the sentence is "most MoE deployments run an untuned
kernel and it costs X", and X is a number this script has to produce rather than
estimate.

HOW IT IS PRICED. Take the two shapes that DO have a tuned bf16 H200 file --
mixtral-8x7b at E=8,N=14336 and qwen2-57b-a14b at E=64,N=2560 -- and, at each
token count, time the SAME layer under the config vLLM resolves normally and
under the config the fallback ladder WOULD have produced had no file shipped.
The ladder is not reimplemented here: `moe.bench.tile_resolve.default_config` is
already vLLM's branch transcribed and tested, and `override_config` is vLLM's own
hook for forcing a config, the same one `scripts/tile_sweep.py` uses.

THE SIGN, stated once so it cannot be misread anywhere below. Every ratio in this
script is

        penalty = time(fallback config) / time(tuned config)

so penalty > 1 means the FALLBACK IS SLOWER and running untuned COSTS
(penalty - 1) of the tuned time. penalty < 1 would mean vLLM's tuned file is
WORSE than its own fallback, which is a publishable result in the other
direction. Every printed line carries the word SLOWER or FASTER; no bare ratio is
ever the whole sentence.

THE CONFOUND, and what is done about it. The tuned config differs from the
fallback in far more than BLOCK_SIZE_M. On this card, at bf16, the two sides
already AGREE on BLOCK_SIZE_M in 19 of the 28 default cells; what they disagree
about everywhere is GROUP_SIZE_M (the ladder pins it to 1 until M//E > 128, so
across the entire decode range, while the tuned files use 16, 32 and 64),
BLOCK_SIZE_N, BLOCK_SIZE_K and num_stages. So a two-arm experiment would price
"tuning" and be silently unable to say which knob it priced. Four extra arms fix
that: each takes the TUNED config and moves ONE knob group to its fallback value.
They are one-at-a-time effects from a common baseline and they need NOT sum to
the whole gap; the residual is printed as the interaction term rather than
hidden.

THE PREDICTIONS, registered here before the run and each printed as PASS or FAIL
against a numeric gate:

  G1  the config vLLM resolves natively is the one `tile_resolve` DERIVES.
      Observed with the existing recorder, not inferred from a time. This is the
      first chance the repo has ever had to check that derivation against a run,
      and if it fails every other number on the page is void.
  G2  forcing a config actually changes what runs: the recorder must see the
      forced dict inside every override context. Without this, three arms could
      quietly be the same kernel and the answer would be a confident 1.000.
  G3  PLACEBO. `native` and `replica` are the same config timed at opposite ends
      of each repeat. Their spread is the noise floor, and no penalty smaller
      than it means anything. Gate: the placebo band is under 3%.
  G4  SIGN. The tuned config is not slower than its own fallback:
      median penalty >= 1.00.
  G5  SIZE, the registered claim. The fallback costs at least 15%: the median
      penalty and the LOW end of its 90% bootstrap interval are both >= 1.15.
      A FAIL here is a real result and not a broken run; it demotes the finding
      to "modest" (5-15%) or "footnote" (<5%), and the script says which.
  G6  MECHANISM. Tile height alone explains less than half the gap. It nearly has
      to: in 19 of 28 cells the two sides pick the same BLOCK_SIZE_M, so the
      penalty there is 0% tile height by construction. Stated as a gate anyway,
      because "nearly has to" is how this study has been wrong before.
  G7  MECHANISM. GROUP_SIZE_M alone explains the majority of the gap. This is the
      one prediction with an independent reason: today's refit found alpha
      FALLING with GROUP_SIZE_M (0.570 at 1, 0.488 at 16), which is what a
      swizzle-for-L2-reuse mechanism predicts, and the ladder pins GROUP_SIZE_M
      to 1 across the whole decode range.

WHAT SURVIVES TEARDOWN. Everything is written under `--out-dir`, which defaults
to the network volume (`$MOE_RESULTS_DIR`, else `/workspace/results` when that
exists, else `<repo>/results`) in `tuned_vs_fallback/<run-id>/`. The absolute
path is printed at the START as well as the end, so a session killed in the
middle still tells you where its rows went. Rows are flushed per arm, the run id
defaults to a hash of the plan, and an interrupted run resumes by re-running the
same command: completed (cell, arm) pairs are skipped.

OFF THE BOX. With no CUDA device or no vLLM, the script prints the full plan --
the coverage census, every cell's two configs, which knob groups differ, and the
predictions -- then exits 3 under a banner that says nothing was measured. Exit
codes: 0 every gate passed, 1 a gate FAILED, 3 nothing was measured.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.baselines._framework_config import (  # noqa: E402
    TileCapture,
    recording_tile_config,
    vllm_call_kwargs,
)
from moe.bench import (  # noqa: E402
    exit_codes,
    timing,
)
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.tile_resolve import (  # noqa: E402
    DERIVED_TUNED,
    VLLM_TAG,
    DerivedTile,
    SnapshotMissing,
    TileNotDerivable,
    config_dtype_selector,
    config_file_name,
    default_config,
    device_selector,
    resolve_tile,
    ships,
)
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec  # noqa: E402

#: The two shapes this study benchmarks that vLLM v0.27.1 ships a tuned bf16
#: H200 file for. Everything else in MODEL_CONFIGS takes the ladder on every
#: card, which is the point of the exercise but makes those shapes useless as the
#: MEASUREMENT: with no tuned file there is no second side to compare against.
TUNED_H200_MODELS: tuple[str, ...] = ("mixtral-8x7b", "qwen2-57b-a14b")

#: Decode through prefill, powers of two, plus the two ladder edges. 32 and 96
#: are the M values at which `get_default_config` steps BLOCK_SIZE_M, so a grid
#: that skipped them would price the ladder everywhere except where it changes.
DEFAULT_TOKENS: tuple[int, ...] = (1, 4, 16, 32, 33, 64, 96, 97, 128, 256, 512,
                                   1024, 2048, 4096)

#: The cards a census of "does a tuned file ship" is worth taking over. Names are
#: the DEVICE SELECTOR form vLLM builds, so they can be compared to the shipped
#: listing directly.
CENSUS_GPUS: tuple[str, ...] = (
    "NVIDIA_H200", "NVIDIA_H100_80GB_HBM3", "NVIDIA_B200",
    "NVIDIA_A100-SXM4-80GB", "NVIDIA_L40S", "NVIDIA_L20",
    "NVIDIA_GeForce_RTX_4090", "NVIDIA_A10G",
)

#: The knob groups the tuned config and the fallback can disagree about, each one
#: an arm that moves ONLY that group from the tuned config to its fallback value.
#: Grouped rather than one arm per key because BLOCK_SIZE_N and BLOCK_SIZE_K
#: jointly size the shared-memory tile and a config that moved one without the
#: other is not a configuration vLLM would ever produce.
KNOB_GROUPS: dict[str, tuple[str, ...]] = {
    "bm": ("BLOCK_SIZE_M",),
    "nk": ("BLOCK_SIZE_N", "BLOCK_SIZE_K"),
    "group": ("GROUP_SIZE_M",),
    "warpstages": ("num_warps", "num_stages"),
}

#: Arm order INSIDE one repeat. `native` first and `replica` last, so the placebo
#: pair straddles every other arm and therefore absorbs whatever drift a repeat
#: contains rather than hiding it. `tuned` is the forced twin of `native` and is
#: the baseline the knob arms are read against, since all five of those are
#: forced and compile through the same path.
ARM_ORDER: tuple[str, ...] = ("native", "tuned", "fallback", "bm", "nk",
                              "group", "warpstages", "replica")

#: The two arms that need no `override_config` context at all.
NATIVE_ARMS = frozenset({"native", "replica"})

#: Verdict bands on the median penalty. 1.15 is the threshold the study named in
#: advance as the difference between a footnote and a headline, so it is written
#: down here rather than chosen after seeing the number.
MATERIAL_PENALTY = 1.15
MODEST_PENALTY = 1.05

#: G3's gate. A placebo spread wider than this means the box is too noisy for a
#: 15% claim to be safe, whatever the penalty comes out at.
PLACEBO_BAND = 0.03

#: G6 and G7. Fractions of the measured gap, not absolute ratios.
TILE_HEIGHT_MAX_SHARE = 0.5
GROUP_MIN_SHARE = 0.5

#: G1's tolerance is zero -- a derived config either is or is not the dict vLLM
#: loaded. This is the correctness tolerance for the OUTPUTS instead. Different
#: tile shapes reduce the K loop in a different order, so bf16 outputs differ in
#: the last bits legitimately; what this catches is an override that changed the
#: computation rather than the schedule.
OUTPUT_REL_TOL = 2e-2

#: Bootstrap settings for the headline interval. Seeded, so two readers of the
#: same CSV get the same interval.
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_BAND = 0.90

EXIT_OK, EXIT_GATE_FAILED, EXIT_NOT_MEASURED = 0, 1, 3

#: The columns `timing.KernelTiming` contributes to every measured row. Named
#: as a group so the header and the row builder cannot drift apart.
TIMING_CSV_COLUMNS = ("instrument", "warmup_ms", "iters", "trials",
                      "sm_clock_load_mhz", "clock_level_ok", "clock_drift_ok",
                      "l2_flush", "host_bound")

CSV_COLUMNS = (
    "run_id", "utc", "gpu_name", "vllm_version", "torch_version", "vllm_tag",
    "model", "num_experts", "intermediate_n", "dtype", "routing", "seed",
    "num_tokens", "arm", "config_origin", "identical_to_tuned",
    # `provenance` here is the TILE's provenance, which predates the run-level
    # block and keeps its name; the run-level fields all carry the `prov_`
    # prefix `moe.bench.provenance` gives them, so the two cannot collide.
    "config_file", "config_key", "provenance",
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
    "observed_config", "override_verified",
    "ms_median", "ms_mean", "ms_stdev", "ms_min", "n_samples",
    "rel_err_vs_native",
    *TIMING_CSV_COLUMNS,
    *PV.Provenance().as_columns(),
    "error",
)


# --------------------------------------------------------------------------
# the sign, in words
# --------------------------------------------------------------------------

def penalty_sentence(ratio: float) -> str:
    """The ratio as a sentence that names which side is slow.

    Exists because a table of bare ratios is exactly how a sign gets misread, and
    a reversed sign here would flip the study's conclusion from "the untuned
    kernel costs" to "vLLM's tuning hurts". Every place a penalty is printed goes
    through this function.
    """
    if ratio > 1.0:
        return f"fallback is {100 * (ratio - 1):.1f}% SLOWER than tuned"
    if ratio < 1.0:
        return (f"fallback is {100 * (1 / ratio - 1):.1f}% FASTER than tuned "
                f"(vLLM's tuned file LOST to its own fallback)")
    return "fallback and tuned are exactly equal"


def verdict_of(ratio: float) -> str:
    """Which of the three pre-registered bands the headline lands in."""
    if ratio < 1.0:
        return "INVERTED"
    if ratio < MODEST_PENALTY:
        return "FOOTNOTE"
    if ratio < MATERIAL_PENALTY:
        return "MODEST"
    return "MATERIAL"


# --------------------------------------------------------------------------
# the plan: which cells, which two configs, which knobs differ
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One (model, token count) comparison, and both configs it will time.

    `tuned` is DERIVED at plan time from vLLM's shipped file and is checked
    against what the run observes (gate G1); `fallback` is DERIVED from vLLM's
    ladder and is never observed anywhere, because the whole point is that no
    file ships for the shapes that take it. Both are labelled derived in the CSV
    through `config_origin`.
    """

    model: str
    num_tokens: int
    dtype: str
    gpu_name: str
    tile: DerivedTile
    tuned: dict[str, int]
    fallback: dict[str, int]

    @property
    def key(self) -> tuple[str, int]:
        return (self.model, self.num_tokens)

    @property
    def differing_groups(self) -> tuple[str, ...]:
        return tuple(name for name, keys in KNOB_GROUPS.items()
                     if any(self.tuned[k] != self.fallback[k] for k in keys))

    @property
    def configs_differ(self) -> bool:
        return bool(self.differing_groups)


def arm_config(cell: Cell, arm: str) -> dict[str, int] | None:
    """The config an arm forces, or None where the arm forces nothing.

    `native` and `replica` return None because they must go through vLLM's own
    resolution: an arm that FORCED the tuned config would still be a valid
    baseline for the knob decomposition but would not answer "what does a
    deployment lose", since a deployment never forces anything.
    """
    if arm in NATIVE_ARMS:
        return None
    if arm == "tuned":
        return dict(cell.tuned)
    if arm == "fallback":
        return dict(cell.fallback)
    if arm in KNOB_GROUPS:
        forced = dict(cell.tuned)
        for key in KNOB_GROUPS[arm]:
            forced[key] = cell.fallback[key]
        return forced
    raise KeyError(f"unknown arm {arm!r}; known arms are {ARM_ORDER}")


def arm_is_identical_to_tuned(cell: Cell, arm: str) -> bool:
    """Would this arm compile the very same kernel as `tuned`?

    True for a knob arm whose group already agrees between the two sides. Such an
    arm is not timed: it would be a second placebo rather than a measurement, and
    its contribution to the gap is exactly zero BY CONSTRUCTION, which is a fact
    worth recording and not worth paying a Triton compile for.
    """
    if arm in KNOB_GROUPS:
        return all(cell.tuned[k] == cell.fallback[k] for k in KNOB_GROUPS[arm])
    return False


def plan_cells(models: list[str], tokens: list[int], dtype: str,
               gpu_name: str) -> tuple[list[Cell], list[str]]:
    """Every cell that can be measured, plus a note per cell that cannot.

    A model with no tuned file on this card is DROPPED with a message rather than
    skipped silently: it is the 79.2% case, it is the reason the script exists,
    and it is unmeasurable for exactly that reason -- there is no tuned side to
    compare the ladder against.
    """
    cells: list[Cell] = []
    notes: list[str] = []
    selector = config_dtype_selector(dtype)
    for model in models:
        if model not in MODEL_CONFIGS:
            notes.append(f"{model}: not in MODEL_CONFIGS, skipped")
            continue
        experts, _, intermediate = MODEL_CONFIGS[model].w2_shape
        try:
            probe = resolve_tile(experts, intermediate, dtype, gpu_name, 1)
        except TileNotDerivable as exc:
            notes.append(f"{model}: {exc}")
            continue
        except SnapshotMissing as exc:
            # A tuned file DOES ship for this shape and this repo has not
            # vendored it. Measurable in principle and not from here, and the
            # difference matters: silently treating it as "no tuned file" would
            # price the ladder against the ladder and answer 1.000.
            notes.append(f"{model}: {exc}")
            continue
        if probe.provenance != DERIVED_TUNED:
            notes.append(
                f"{model}: E={experts},N={intermediate} {dtype} has NO tuned "
                f"file on {gpu_name} ({probe.config_file}), so it takes the "
                f"fallback ladder already and there is no tuned side to price "
                f"it against. This is the 79.2% case, not a bug.")
            continue
        for tok in tokens:
            tile = resolve_tile(experts, intermediate, dtype, gpu_name, tok)
            tuned = {"BLOCK_SIZE_M": tile.block_m_derived,
                     "BLOCK_SIZE_N": tile.block_n_derived,
                     "BLOCK_SIZE_K": tile.block_k_derived,
                     "GROUP_SIZE_M": tile.group_m_derived,
                     "num_warps": tile.num_warps_derived,
                     "num_stages": tile.num_stages_derived}
            cells.append(Cell(model=model, num_tokens=tok, dtype=dtype,
                              gpu_name=gpu_name, tile=tile, tuned=tuned,
                              fallback=default_config(tok, experts, selector)))
    return cells, notes


def format_config(cfg: dict[str, int]) -> str:
    """One compact, always-full config string. Never a partial one.

    Reporting "BLOCK_SIZE_M=64" alone is what made the tile confound invisible in
    the first place, so there is no helper here that prints a subset.
    """
    return (f"M{cfg['BLOCK_SIZE_M']:<4}N{cfg['BLOCK_SIZE_N']:<4}"
            f"K{cfg['BLOCK_SIZE_K']:<4}G{cfg['GROUP_SIZE_M']:<3}"
            f"w{cfg['num_warps']} s{cfg['num_stages']}")


# --------------------------------------------------------------------------
# the census that motivates the measurement
# --------------------------------------------------------------------------

def coverage_census(models: list[str], gpus: list[str],
                    dtype: str = "bf16") -> list[tuple[str, str, bool, str]]:
    """(model, gpu, has a tuned file, filename) over a named grid.

    NOT the study's 288-combination census, which spans tensor-parallel shards
    this repo does not all define; this is the smaller grid of the shapes
    `MODEL_CONFIGS` actually carries, computed from the same shipped listing, and
    it is here so the script's own motivation carries a number it can defend
    rather than a number quoted from somewhere else.
    """
    selector = config_dtype_selector(dtype)
    out = []
    for model in models:
        experts, _, intermediate = MODEL_CONFIGS[model].w2_shape
        for gpu in gpus:
            # `ships`, not `resolve_tile`: the question is whether a file EXISTS
            # upstream, and only four of the 327 are vendored into this repo. A
            # census built on the vendored snapshot would report 323 shapes as
            # uncovered when they are merely not copied here.
            name = config_file_name(experts, intermediate, selector,
                                    device_selector(gpu))
            out.append((model, gpu, ships(name), name))
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def median_ratio(pairs: list[tuple[float, float]]) -> float | None:
    """Median of per-cell ratios, not a ratio of pooled medians.

    Per cell first, because the cells span three orders of magnitude in absolute
    time and a pooled ratio would be the T=4096 cell wearing a costume. This is
    the same mistake `alpha_refit.py` documents in its estimator note, in a
    smaller form.
    """
    ratios = [a / b for a, b in pairs if b > 0]
    return statistics.median(ratios) if ratios else None


def bootstrap_interval(values: list[float], band: float = BOOTSTRAP_BAND,
                       reps: int = BOOTSTRAP_REPS,
                       seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """Percentile bootstrap of the MEDIAN, resampling cells.

    Cells and not timing samples: the timing noise inside one cell is small and
    already summarised by its own median, and the uncertainty that matters for a
    headline is "would another set of cells have said the same", which is
    between-cell.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(reps):
        medians.append(statistics.median(
            [values[rng.randrange(n)] for _ in range(n)]))
    medians.sort()
    lo_i = int((1 - band) / 2 * reps)
    hi_i = min(reps - 1, int((1 + band) / 2 * reps))
    return medians[lo_i], medians[hi_i]


def knob_share(knob_penalty: float, total_penalty: float) -> float | None:
    """What fraction of the measured excess time one knob group explains.

    Both arguments are ratios against the tuned baseline, so the excesses are
    `x - 1` and the share is the ratio of those. Returns None when there is no
    gap to apportion, rather than dividing by something near zero and printing a
    share of 4000%.
    """
    if total_penalty <= 1.0:
        return None
    return (knob_penalty - 1.0) / (total_penalty - 1.0)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` means the run could not evaluate it, which is printed as
    UNKNOWN and never as a pass. A gate nobody could check is the state this
    project's retractions were written in.
    """

    name: str
    prediction: str
    rule: str
    passed: bool | None
    observed: str

    @property
    def token(self) -> str:
        """The gate's name as ONE whitespace-free token, for `RESULT:`.

        `name` reads "G5 size" so a human can read the table; a result line's
        name field is one token by the parser's rule, and every name here
        starts with its own G<n>, so two gates cannot collide on it.
        """
        return self.name.split()[0]

    @property
    def kind(self) -> str:
        """VALIDITY for G0 to G3, CLAIM for G4 upwards.

        The split was already in `build_gates`'s docstring and in the order the
        gates are appended in; it had never been a field, so nothing downstream
        could act on it and the process exited 1 for either. It decides the exit
        code now: a VALIDITY failure is INVALID and a CLAIM failure is a RESULT.
        """
        number = self.token.lstrip("G")
        return (exit_codes.VALIDITY
                if number.isdigit() and int(number) <= 3 else exit_codes.CLAIM)

    @property
    def verdict(self) -> str:
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.passed]

    def result_line(self) -> str:
        """The ONE line the session driver may grep for this gate.

        Rendered by `moe.bench.exit_codes.result_line`, so the prefix, the
        field order and the refusals are identical in every script. The human
        block below it is prose: a line that merely contains "PASS" is not a
        result, which is what the driver's old free-text grep was reading
        pre-registered expectations out of.
        """
        return exit_codes.result_line(self.kind, self.token, self.verdict,
                                      self.observed.replace("\n", " ")[:160])

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` for `exit_codes.classify`."""
        return (self.kind, self.token, self.verdict)

    def render(self, with_result: bool = True) -> str:
        out = ([self.result_line()] if with_result else [])
        out.append(f"[{self.verdict}] {self.name}  {self.prediction}\n"
                   f"         gate: {self.rule}\n"
                   f"         saw:  {self.observed}")
        return "\n".join(out)


def render_gates(gates: list[Gate]) -> str:
    lines = [g.render() for g in gates]
    failed = [g for g in gates if g.passed is False]
    unknown = [g for g in gates if g.passed is None]
    lines.append("")
    lines.append(f"{sum(1 for g in gates if g.passed)} PASS, {len(failed)} FAIL, "
                 f"{len(unknown)} UNKNOWN")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# where the rows go, and how a killed run picks them up again
# --------------------------------------------------------------------------

def default_out_dir() -> Path:
    """The same resolution `scripts/run_all.sh` uses, for the same reason.

    A pod's container disk dies with the pod and the network volume does not, so
    a results path that defaults to the repo checkout is a results path that
    defaults to being lost. `$MOE_RESULTS_DIR` wins, then `/workspace/results`
    when `/workspace` exists at all, then the repo.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    if Path("/workspace").is_dir():
        return Path("/workspace/results")
    return Path(__file__).resolve().parents[1] / "results"


def plan_run_id(models: list[str], tokens: list[int], dtype: str,
                gpu_name: str, reps: int, iters: int, seed: int,
                routing: str, *, card: str | None = None,
                warmup: float = 300.0, budget: float = 200.0,
                trials: int = 3, l2_flush: bool = True) -> str:
    """A run id that is a HASH OF THE PLAN UNDER THE CARD, so a rerun resumes.

    An idempotent script whose default run id is random is not idempotent in
    practice: the second invocation writes a second directory and repeats every
    cell. Hashing the plan means "run the same command again" is the resume
    command, and changing any parameter that would invalidate the old rows
    changes the directory instead of silently mixing two experiments.

    THE CARD AND THE FOUR TIMING KNOBS WERE MISSING UNTIL 2026-09-02. The card
    is not swept by the script, it is swept by the operator moving to another
    pod, and the results root prefers `$MOE_RESULTS_DIR` then
    `/workspace/results`, a network volume that outlives a pod on purpose. The
    sibling sweep has the proof of what that costs in the repo: one report
    filename appears under both `2026-09-01-nvidia_h200-cross-card-s3` and
    `2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`, one id and two cards.
    `--warmup`, `--cell-budget-ms`, `--trials` and the flush state each set the
    measured milliseconds of every row, so a re-run at a different one landing
    in the same directory would print the old numbers under the new label.

    `gpu_name` stays in the key beside `card` and they are not the same thing:
    `gpu_name` is the device the CONFIG LOOKUP is derived for, which the
    operator may deliberately set to a card they are about to rent, while
    `card` is what the run is labelled and stored as.

    `card` defaults to `gpu_name` because a caller who named only one card
    named the one this run is about; the two diverge only when the operator
    deliberately derives a plan for a card they are about to rent, and then
    both are in the key. The timing knobs default to the parser's defaults, so
    an older caller that passes eight positionals still gets an id for the
    configuration that caller would have run.

    Built by `moe.bench.provenance.run_id`, which refuses a missing card and a
    None knob, sorts before hashing, and puts the card slug at the front where
    `ls` shows it.
    """
    card = card or gpu_name
    return PV.run_id(card=card, model=models, tokens=tokens, dtype=dtype,
                     lookup=gpu_name, reps=reps, iters=iters, seed=seed,
                     routing=routing, warmup=warmup, budget=budget,
                     trials=trials, flush=l2_flush)


class NoCardToLabel(ValueError):
    """A run that will MEASURE could not name the card it is measuring.

    Raised only on the measuring path. A plan-only run has an answer that is
    not a refusal (`ASSUMED_CARD` below); a run about to time kernels does not,
    because its rows would carry a card name nothing checked.
    """


#: The card a plan-only run off GPU is derived for when the operator names none,
#: AND THE PREFIX IS PART OF IT. The old fallback was the bare literal
#: "NVIDIA H200" (A5/R6), which did two jobs at once and hid both: it named the
#: plan for a card the machine might not be, and, because the lookup device
#: decides which tuned file `resolve_tile` reads, it decided whether there was a
#: tuned side to compare against AT ALL -- the two default models have a tuned
#: H200 file and almost nothing else does, so the default made the premise true
#: by construction on every machine.
#:
#: It is still a default, because a plan-only run must be able to print a plan
#: and there is no card to detect off GPU. What changed is that it can no longer
#: be mistaken for a measurement: the assumption is in the run id, in the
#: directory name, in the header and in `card` on every row, so no off-GPU run
#: writes a bare H200 label anywhere. The measuring path refuses instead.
ASSUMED_CARD = "ASSUMED NVIDIA H200"

#: What `ASSUMED_CARD` assumes, once, for `resolve_tile`: the config lookup
#: needs a real device selector and "ASSUMED ..." is not one.
ASSUMED_LOOKUP_GPU = "NVIDIA H200"


def resolve_lookup_gpu(args, env: dict) -> tuple[str, str, str]:
    """`(lookup device, card label, note)`, in falling order of directness.

    `--gpu-name` overrides the lookup, because deriving a plan for a card you
    are about to rent is a supported and useful thing to do; `--card` names
    what the run is labelled and stored as; the live device answers both when
    there is one. With none of the three the answer is `ASSUMED_CARD`, and the
    note says so in the words the header prints.
    """
    lookup = args.gpu_name or args.card or env.get("gpu_name")
    card = args.card or env.get("gpu_name") or args.gpu_name
    if lookup and card:
        return str(lookup), str(card), ""
    return ASSUMED_LOOKUP_GPU, ASSUMED_CARD, (
        f"NO CARD WAS NAMED. The config lookup below is derived for "
        f"{ASSUMED_LOOKUP_GPU} because nothing on this machine says otherwise, "
        f"and every artefact this run writes is labelled {ASSUMED_CARD!r} so "
        f"it cannot be read as a measurement of one. Pass --card to say which "
        f"card you mean; the measuring path REFUSES without it.")


def require_card_to_measure(card: str) -> None:
    """Refuse to time anything under an assumed card.

    A plan is arithmetic and can say what it assumed. A row of milliseconds
    cannot: it is compared against a per-card ridge and a per-card bandwidth by
    everything downstream, and an assumed label on one is how a stale H200 band
    ended up in seven published A100 reports.
    """
    if card == ASSUMED_CARD:
        raise NoCardToLabel(
            "this run would MEASURE, and no card was named. Pass --card "
            "'NVIDIA H200' (as nvidia-smi spells it). --plan-only runs off GPU "
            f"without one and label everything {ASSUMED_CARD!r}.")


@dataclass
class ArmResult:
    """One timed arm of one cell, or the reason there is no timing.

    `error` non-empty means the arm did not produce a number -- a Triton compile
    that ran out of shared memory is the expected case, since a single-knob arm
    can build a config vLLM would never emit. Such an arm is excluded from every
    median and named in the report, never treated as equal to the baseline.
    """

    model: str
    num_tokens: int
    arm: str
    config: dict[str, int] | None
    config_origin: str
    identical_to_tuned: bool = False
    ms_median: float | None = None
    ms_mean: float | None = None
    ms_stdev: float | None = None
    ms_min: float | None = None
    n_samples: int = 0
    rel_err_vs_native: float | None = None
    observed_config: dict | None = None
    override_verified: bool | None = None
    #: The state the number was measured in, from `timing.KernelTiming`. A row
    #: with no `instrument` is a row from before 2026-09-02, measured by the
    #: retired private loop, and is not comparable with a roof. The three flags
    #: are tri-state and None means NOT DETERMINED; `clock_level_ok` and
    #: `clock_drift_ok` must BOTH be read, since the old drop-only flag is the
    #: defect they replace.
    instrument: str = ""
    warmup_ms: float | None = None
    iters: int = 0
    trials: int = 0
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    l2_flush: bool | None = None
    host_bound: bool | None = None
    error: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.model, self.num_tokens, self.arm)

    def row(self, cell: Cell, meta: dict) -> dict:
        cfg = self.config or {}
        return {
            "run_id": meta["run_id"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                           time.gmtime()),
            "gpu_name": meta["gpu_name"], "vllm_version": meta["vllm_version"],
            "torch_version": meta["torch_version"], "vllm_tag": VLLM_TAG,
            "model": self.model, "num_experts": MODEL_CONFIGS[self.model].num_experts,
            "intermediate_n": MODEL_CONFIGS[self.model].w2_shape[2],
            "dtype": cell.dtype, "routing": meta["routing"], "seed": meta["seed"],
            "num_tokens": self.num_tokens, "arm": self.arm,
            "config_origin": self.config_origin,
            "identical_to_tuned": int(self.identical_to_tuned),
            "config_file": cell.tile.config_file,
            "config_key": cell.tile.config_key_derived,
            "provenance": cell.tile.provenance,
            "BLOCK_SIZE_M": cfg.get("BLOCK_SIZE_M", ""),
            "BLOCK_SIZE_N": cfg.get("BLOCK_SIZE_N", ""),
            "BLOCK_SIZE_K": cfg.get("BLOCK_SIZE_K", ""),
            "GROUP_SIZE_M": cfg.get("GROUP_SIZE_M", ""),
            "num_warps": cfg.get("num_warps", ""),
            "num_stages": cfg.get("num_stages", ""),
            "observed_config": json.dumps(self.observed_config, sort_keys=True)
                               if self.observed_config else "",
            "override_verified": "" if self.override_verified is None
                                 else int(self.override_verified),
            "ms_median": "" if self.ms_median is None else f"{self.ms_median:.6f}",
            "ms_mean": "" if self.ms_mean is None else f"{self.ms_mean:.6f}",
            "ms_stdev": "" if self.ms_stdev is None else f"{self.ms_stdev:.6f}",
            "ms_min": "" if self.ms_min is None else f"{self.ms_min:.6f}",
            "n_samples": self.n_samples,
            "rel_err_vs_native": "" if self.rel_err_vs_native is None
                                 else f"{self.rel_err_vs_native:.3e}",
            "instrument": self.instrument,
            "warmup_ms": "" if self.warmup_ms is None else f"{self.warmup_ms:.1f}",
            "iters": self.iters, "trials": self.trials,
            "sm_clock_load_mhz": ("" if self.sm_clock_load_mhz is None
                                  else f"{self.sm_clock_load_mhz:.0f}"),
            "clock_level_ok": _flag(self.clock_level_ok),
            "clock_drift_ok": _flag(self.clock_drift_ok),
            "l2_flush": _flag(self.l2_flush),
            "host_bound": _flag(self.host_bound),
            # One column per provenance field, under `prov_`. A row read on its
            # own then names the commit, the card and the instrument that made
            # it; the 26 published reports name none of the three.
            **(meta["prov"].as_columns() if meta.get("prov") else {}),
            "error": self.error,
        }


class Store:
    """Append-only CSV of arm results, flushed per arm, re-read on resume.

    Flushed per arm and not per cell because the unit of loss on a killed pod
    should be the smallest thing that took real time, and on this script that is
    one arm's compile plus its repeats.
    """

    def __init__(self, path: Path, fresh: bool = False):
        self.path = path
        self.done: dict[tuple[str, int, str], dict] = {}
        if fresh and path.exists():
            path.unlink()
        if path.exists():
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        key = (row["model"], int(row["num_tokens"]), row["arm"])
                    except (KeyError, ValueError):
                        continue
                    self.done[key] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(CSV_COLUMNS),
                                      extrasaction="ignore")
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def has(self, key: tuple[str, int, str]) -> bool:
        return key in self.done

    def restore(self, key: tuple[str, int, str]) -> ArmResult | None:
        """Rebuild an ArmResult from a CSV row written by an earlier run."""
        row = self.done.get(key)
        if row is None:
            return None

        def num(name, cast=float):
            value = row.get(name, "")
            try:
                return cast(value)
            except (TypeError, ValueError):
                return None

        cfg = {k: int(row[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                        "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                        "num_warps", "num_stages")
               if row.get(k) not in (None, "")}
        return ArmResult(
            model=row["model"], num_tokens=int(row["num_tokens"]),
            arm=row["arm"], config=cfg or None,
            config_origin=row.get("config_origin", ""),
            # str() because a row read back from CSV holds "1" while a row
            # written this session holds the int 1, and both mean the same thing.
            identical_to_tuned=str(row.get("identical_to_tuned")) in ("1", "True"),
            ms_median=num("ms_median"), ms_mean=num("ms_mean"),
            ms_stdev=num("ms_stdev"), ms_min=num("ms_min"),
            n_samples=num("n_samples", int) or 0,
            rel_err_vs_native=num("rel_err_vs_native"),
            observed_config=json.loads(row["observed_config"])
                            if row.get("observed_config") else None,
            override_verified=None if str(row.get("override_verified", "")) == ""
                              else str(row["override_verified"]) == "1",
            # A CSV written before 2026-09-02 has none of these columns, and an
            # absent instrument is the marker that says so. Restored as ""/None
            # rather than defaulted, so a resumed run cannot claim the current
            # instrument for rows the retired one measured.
            instrument=row.get("instrument", ""),
            warmup_ms=num("warmup_ms"),
            iters=num("iters", int) or 0, trials=num("trials", int) or 0,
            sm_clock_load_mhz=num("sm_clock_load_mhz"),
            clock_level_ok=_unflag(row.get("clock_level_ok", "")),
            clock_drift_ok=_unflag(row.get("clock_drift_ok", "")),
            l2_flush=_unflag(row.get("l2_flush", "")),
            host_bound=_unflag(row.get("host_bound", "")),
            error=row.get("error", ""))

    def write(self, result: ArmResult, cell: Cell, meta: dict) -> None:
        row = result.row(cell, meta)
        self._writer.writerow(row)
        self._fh.flush()
        # The row itself, not a placeholder: `restore` reads this dict, and a
        # placeholder would make a same-session restore raise instead of
        # returning what was just written.
        self.done[result.key] = row

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


# --------------------------------------------------------------------------
# the measurement, which is the only part that needs the box
# --------------------------------------------------------------------------

def find_vllm_hooks():
    """vLLM's `override_config` and, if it is there, `get_config`.

    Probed across the module paths the import has lived at, exactly as
    `scripts/tile_sweep.find_override` does: a wrong guess here forces nothing
    and every arm quietly becomes the native one, which would produce a
    beautifully tight 1.000 and mean nothing at all.
    """
    import importlib

    from moe.baselines._framework_config import VLLM_CONFIG_MODULES

    for name in VLLM_CONFIG_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, getattr(mod, "get_config", None), name
    raise SystemExit(
        "vLLM is importable but exposes no override_config in any of "
        f"{VLLM_CONFIG_MODULES}. Without it every arm would run the native "
        "config and the comparison would be vacuous.")


#: RETIRED 2026-09-02. This file carried a verbatim copy of
#: `block_m_crossing_sweep.time_call`: events created inside the loop, a
#: synchronise after every iteration, no L2 flush, no clock read. It was one of
#: six copies, and the audit (A7) bounded the host prefix they expose at
#: ~0.18 ms per fused_experts call on the H200 pod and ~0.30 ms on the A100.
#: A ratio between two arms cancels a CONSTANT prefix, which is why this script
#: survived it better than the ladders did, but it does not cancel one that
#: scales with launches, and the fallback arm at BLOCK_SIZE_M=16 launches more
#: tiles than the tuned one. `timing.time_kernel` is the one instrument, and
#: `timing.TIMING_BASIS` travels in every row it produces.


def summarise_samples(result: ArmResult, samples: list[float]) -> ArmResult:
    """Reduce the per-repeat p50s to the arm's headline numbers.

    THE UNIT CHANGED ON 2026-09-02 AND THE ARITHMETIC DID NOT. `samples` used
    to be individual `time_calls` iterations; it is now one queue-deep p50 per
    round-robin repeat. `ms_stdev` is therefore the BETWEEN-REPEAT spread,
    which is what the G3 placebo band is about, rather than the within-cell
    jitter the column used to hold.
    """
    result.ms_median = statistics.median(samples)
    result.ms_mean = statistics.fmean(samples)
    result.ms_stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    result.ms_min = min(samples)
    result.n_samples = len(samples)
    return result


def summarise_timings(result: ArmResult, timings: list) -> ArmResult:
    """Fold the repeats' `KernelTiming` records into the arm's row.

    One repeat is one `time_kernel` call, so an arm has several; the row has one
    of each column. The reductions are chosen so a filter over the row cannot be
    more permissive than a filter over the repeats: the flags are folded with
    the BAD value dominating and None absorbing, so one throttled repeat makes
    the arm throttled and an undetermined one stays undetermined; `iters`,
    `trials` and `warmup_ms` are per-repeat medians, not totals, so a reader
    can check them against `n_samples`.
    """
    if not timings:
        return result
    result.instrument = timings[0].instrument
    result.l2_flush = bool(timings[0].l2_flush)
    result.iters = int(statistics.median([t.iters for t in timings]))
    result.trials = int(statistics.median([t.trials for t in timings]))
    result.warmup_ms = float(statistics.median([t.warmup_ms for t in timings]))
    clocks = [t.sm_clock_load_mhz for t in timings if t.sm_clock_load_mhz]
    result.sm_clock_load_mhz = statistics.median(clocks) if clocks else None
    result.clock_level_ok = _fold_flag([t.clock_level_ok for t in timings])
    result.clock_drift_ok = _fold_flag([t.clock_drift_ok for t in timings])
    result.host_bound = _fold_flag([t.host_bound for t in timings], bad=True)
    return result


def _fold_flag(values: list, bad: bool = False) -> bool | None:
    """Fold a tri-state flag over repeats so one bad repeat wins, None absorbing.

    `bad` is the DOMINATING value: False for the two clock flags, whose False
    means throttled or drifting, True for `host_bound`, whose True means the
    interval carried host time. One function with the polarity as an argument
    rather than two that differ by a negation, because that difference is how a
    filter comes to pass a row it should have dropped.
    """
    if any(v is bad for v in values):
        return bad
    if any(v is None for v in values):
        return None
    return not bad


def _flag(value: bool | None) -> str:
    """A tri-state flag as a CSV cell: "1", "0", or empty for NOT DETERMINED."""
    return "" if value is None else str(int(value))


def _unflag(cell: str) -> bool | None:
    """`_flag` read back. An empty cell is None, which is NOT False."""
    text = str(cell).strip()
    if text == "":
        return None
    return text not in ("0", "False", "false")


def measure_cell(cell: Cell, arms: list[str], args, store: Store, meta: dict,
                 hooks) -> dict[str, ArmResult]:
    """Time every arm of one cell, round-robin across repeats.

    ROUND-ROBIN IS LOAD-BEARING. Running arm A to completion and then arm B puts
    every slow-clock minute of the session into whichever arm ran during it, and
    the whole answer is a ratio between arms. Interleaving at the repeat level
    means a thermal or neighbour-noise excursion lands on all arms roughly
    equally, and the `native`/`replica` placebo measures whatever is left.
    """
    import torch

    from moe.bench.tolerance import relative_error

    override_config, get_config, _ = hooks
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.reference.torch_ref import make_inputs
    from moe.routing.distributions import sample_topk_ids

    cfg = MODEL_CONFIGS[cell.model]
    spec = BenchSpec(cfg, num_tokens=cell.num_tokens, dtype=cell.dtype,
                     routing=RoutingSpec(args.routing, 0.0), seed=args.seed)
    x, weights = make_inputs(spec, device="cuda")
    ids = sample_topk_ids(spec.routing, cell.num_tokens, cfg.num_experts,
                          cfg.top_k, seed=args.seed, device="cuda")
    topk_w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                        device="cuda")
    call_kwargs = vllm_call_kwargs(spec)
    call_kwargs["activation"] = MoEActivation(call_kwargs["activation"])

    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=topk_w, topk_ids=ids, **call_kwargs)

    def context(arm: str):
        forced = arm_config(cell, arm)
        return contextlib.nullcontext() if forced is None else override_config(forced)

    results: dict[str, ArmResult] = {}
    pending: list[str] = []
    for arm in arms:
        key = (cell.model, cell.num_tokens, arm)
        restored = store.restore(key)
        if restored is not None and not args.fresh:
            results[arm] = restored
            continue
        if arm_is_identical_to_tuned(cell, arm):
            # Not timed. Its config IS the tuned config, so its contribution to
            # the gap is exactly zero by construction and a Triton compile would
            # buy a second placebo rather than a measurement.
            results[arm] = ArmResult(
                cell.model, cell.num_tokens, arm, arm_config(cell, arm),
                config_origin="derived", identical_to_tuned=True,
                error="")
            store.write(results[arm], cell, meta)
            continue
        pending.append(arm)

    # One observation pass before any timing: what config did vLLM really use,
    # and did the arms compute the same layer? Both questions are about the FIRST
    # call, and the recorder deep-copies a dict per call, so it is kept out of
    # the timed loop.
    native_out = None
    if pending and "native" not in pending and "native" in arms:
        # `native` came back from a previous run's CSV, so nothing in this
        # session has its output to compare the new arms against. Run it once,
        # untimed. Without this a resumed run silently loses G0 while still
        # printing G0 as a pass.
        with contextlib.suppress(Exception):
            native_out = call()
            torch.cuda.synchronize()
    for arm in pending:
        forced = arm_config(cell, arm)
        result = ArmResult(cell.model, cell.num_tokens, arm, forced,
                           config_origin="forced" if forced else "observed")
        capture = TileCapture()
        try:
            with context(arm), recording_tile_config(capture):
                out = call()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            # Broad on purpose: a single-knob arm can name a config vLLM would
            # never emit (BLOCK_SIZE_N=256 at num_stages=5 overruns shared
            # memory on some shapes), and one arm failing to compile must not
            # take the cell, or the session, down with it.
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            results[arm] = result
            continue
        seen = capture.calls[0].config if capture.calls else None
        result.observed_config = seen
        if forced is not None:
            if seen is not None:
                result.override_verified = all(
                    seen.get(k) == v for k, v in forced.items())
            elif get_config is not None:
                # The recorder saw nothing -- a vLLM that memoises its own
                # lookup would do that -- so ask the hook directly instead.
                # Weaker evidence, since it proves the override is SET rather
                # than that the kernel read it, but it is the difference between
                # UNKNOWN and a check, and G2 exists precisely so that an
                # override which forces nothing cannot pass unnoticed.
                with context(arm):
                    live = get_config()
                result.override_verified = bool(live) and all(
                    live.get(k) == v for k, v in forced.items())
        if arm == "native":
            native_out = out
        elif native_out is not None:
            result.rel_err_vs_native = relative_error(out, native_out)
        results[arm] = result
        del out

    # Round-robin repeats. Every arm gets its warmup inside every repeat so the
    # repeats are symmetric and the first one is not the only one paying for a
    # cold instruction cache.
    samples: dict[str, list[float]] = {arm: [] for arm in pending
                                       if not results[arm].error}
    #: One `KernelTiming` per arm per repeat. Kept rather than reduced on the
    #: spot because a repeat that throttled has to be able to make the whole
    #: arm's row say so; see `summarise_timings`.
    records: dict[str, list] = {arm: [] for arm in samples}
    for _ in range(args.reps):
        for arm in list(samples):
            try:
                with context(arm):
                    # ONE INSTRUMENT (A7). This used to be a private
                    # `time_calls` whose events were created inside the loop
                    # and which synchronised after every iteration, with no
                    # flush and no clock read. `time_kernel` is the queue-deep
                    # loop the roof was measured with, and it reports the clock
                    # it ran at beside the time.
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=meta.get("reference_clock_mhz"))
                samples[arm].append(t.ms_p50)
                records[arm].append(t)
            except Exception as exc:  # noqa: BLE001
                results[arm].error = f"{type(exc).__name__}: {exc}"[:300]
                samples.pop(arm, None)
                records.pop(arm, None)

    for arm in pending:
        got = samples.get(arm)
        if got:
            summarise_samples(results[arm], got)
            summarise_timings(results[arm], records.get(arm) or [])
        store.write(results[arm], cell, meta)
    del native_out
    return results


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. No interpolation, so every value printed is a
    value that was actually measured somewhere."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q * len(ordered) + 0.5))))
    return ordered[rank - 1]


@dataclass
class CellResult:
    """One cell reduced to the numbers the report and the gates both read."""

    cell: Cell
    penalty: float | None = None            # fallback / native
    penalty_forced: float | None = None     # fallback / tuned, both forced
    placebo: float | None = None            # replica / native
    forced_vs_native: float | None = None   # tuned / native
    knob: dict[str, float | None] = field(default_factory=dict)
    native_ms: float | None = None
    fallback_ms: float | None = None
    excluded: str = ""


def analyse_cell(cell: Cell, arms: dict[str, ArmResult]) -> CellResult:
    """Reduce one cell's arms to ratios, or say why it has none.

    A cell whose two configs are IDENTICAL is excluded rather than counted as a
    penalty of 1.0. Averaging in a comparison that was never a comparison would
    drag the headline toward 1 in proportion to how well vLLM's ladder happens to
    agree with its own tuned file, which is a different question.
    """
    out = CellResult(cell=cell)
    if not cell.configs_differ:
        out.excluded = ("tuned and fallback configs are identical here; there "
                        "is nothing to price")
        return out

    def ms(name: str) -> float | None:
        arm = arms.get(name)
        if arm is None or arm.error or not arm.ms_median:
            return None
        return arm.ms_median

    native, fallback, tuned = ms("native"), ms("fallback"), ms("tuned")
    out.native_ms, out.fallback_ms = native, fallback
    if native is None or fallback is None:
        broken = [f"{n}: {arms[n].error}" for n in ("native", "fallback")
                  if n in arms and arms[n].error]
        out.excluded = "; ".join(broken) or "no timing recorded"
        return out
    out.penalty = fallback / native
    if tuned:
        out.penalty_forced = fallback / tuned
        out.forced_vs_native = tuned / native
    replica = ms("replica")
    if replica:
        out.placebo = replica / native
    for name in KNOB_GROUPS:
        arm = arms.get(name)
        if arm is None:
            out.knob[name] = None
        elif arm.identical_to_tuned:
            out.knob[name] = 1.0            # zero contribution BY CONSTRUCTION
        elif arm.error or not arm.ms_median or not tuned:
            out.knob[name] = None
        else:
            out.knob[name] = arm.ms_median / tuned
    return out


@dataclass
class Analysis:
    """The whole run, reduced to what the gates and the headline need."""

    cells: list[CellResult]
    per_model: dict[str, float]
    headline: float | None
    interval: tuple[float, float] | None
    placebo_band: float | None
    knob_share_median: dict[str, float | None]
    max_rel_err: float | None
    #: How many cells had a native config the recorder actually saw. Zero means
    #: G1 is UNKNOWN rather than passed: "no mismatch found" and "nothing was
    #: looked at" are the two states this repo has confused before.
    natives_observed: int
    #: How many forced arms the recorder actually watched. Zero makes G2
    #: UNKNOWN: an override that forced nothing and an override nobody checked
    #: both produce an empty failure list.
    overrides_checked: int
    derivation_mismatches: list[str]
    override_failures: list[str]
    compile_failures: list[str]

    @property
    def measured(self) -> list[CellResult]:
        return [c for c in self.cells if c.penalty is not None]


def analyse(cells: list[Cell],
            results: dict[tuple[str, int], dict[str, ArmResult]]) -> Analysis:
    """Everything the report prints, computed in one place and testable.

    Kept free of printing and of torch so the whole reduction can be exercised on
    a laptop against synthetic arm results, which is the only way the SIGN of the
    headline gets a test at all before a pod exists.
    """
    per_cell = [analyse_cell(c, results.get(c.key, {})) for c in cells]
    penalties = [c.penalty for c in per_cell if c.penalty is not None]

    per_model: dict[str, float] = {}
    for model in dict.fromkeys(c.cell.model for c in per_cell):
        got = [c.penalty for c in per_cell
               if c.cell.model == model and c.penalty is not None]
        if got:
            per_model[model] = statistics.median(got)

    placebos = [abs(c.placebo - 1.0) for c in per_cell if c.placebo is not None]
    shares: dict[str, float | None] = {}
    for name in KNOB_GROUPS:
        got = [s for s in
               (knob_share(c.knob.get(name), c.penalty_forced) for c in per_cell
                if c.knob.get(name) is not None and c.penalty_forced is not None)
               if s is not None]
        shares[name] = statistics.median(got) if got else None

    mismatches, override_fail, compile_fail, rel_errs = [], [], [], []
    natives_observed = overrides_checked = 0
    for cell in cells:
        for arm_name, arm in results.get(cell.key, {}).items():
            where = f"{cell.model} T={cell.num_tokens} {arm_name}"
            if arm.error:
                compile_fail.append(f"{where}: {arm.error}")
            if arm.rel_err_vs_native is not None:
                rel_errs.append(arm.rel_err_vs_native)
            if arm.override_verified is not None:
                overrides_checked += 1
            if arm.override_verified is False:
                override_fail.append(
                    f"{where}: forced {format_config(arm.config)} but vLLM used "
                    f"{arm.observed_config}")
            if arm_name == "native" and arm.observed_config:
                natives_observed += 1
                differing = {k: (v, arm.observed_config.get(k))
                             for k, v in cell.tuned.items()
                             if arm.observed_config.get(k) != v}
                if differing:
                    mismatches.append(
                        f"{where}: DERIVED {format_config(cell.tuned)} but the "
                        f"run OBSERVED {differing} (derived, observed)")

    return Analysis(
        cells=per_cell, per_model=per_model,
        headline=statistics.median(penalties) if penalties else None,
        interval=bootstrap_interval(penalties),
        placebo_band=percentile(placebos, 0.90),
        knob_share_median=shares,
        max_rel_err=max(rel_errs) if rel_errs else None,
        natives_observed=natives_observed, overrides_checked=overrides_checked,
        derivation_mismatches=mismatches, override_failures=override_fail,
        compile_failures=compile_fail)


def build_gates(analysis: Analysis) -> list[Gate]:
    """The eight pre-registered gates, evaluated against the run.

    Split deliberately into VALIDITY gates (G0 to G3), where a FAIL means the
    numbers are not to be believed, and CLAIM gates (G4 to G7), where a FAIL is a
    result. The rendering keeps them in that order so a reader hits the reasons
    to distrust the page before the page's conclusion.
    """
    gates: list[Gate] = []
    n = len(analysis.measured)

    err = analysis.max_rel_err
    gates.append(Gate(
        "G0 same-layer", "every arm computes the same MoE layer as the native one",
        f"max relative error vs native <= {OUTPUT_REL_TOL:g}",
        None if err is None else err <= OUTPUT_REL_TOL,
        "no arm was compared" if err is None else f"max rel err {err:.2e}"))

    gates.append(Gate(
        "G1 derivation", f"vLLM {VLLM_TAG} loads the config tile_resolve DERIVES",
        "zero cells where the observed native config differs from the derived one",
        None if analysis.natives_observed == 0
        else len(analysis.derivation_mismatches) == 0,
        "no native config was observed, so the derivation is still unchecked"
        if analysis.natives_observed == 0
        else (f"{len(analysis.derivation_mismatches)} mismatches over "
              f"{analysis.natives_observed} observed cells"
              + ("" if not analysis.derivation_mismatches
                 else "; first: " + analysis.derivation_mismatches[0]))))

    gates.append(Gate(
        "G2 override", "override_config actually forces the config it is given",
        "zero arms where the recorder saw a config other than the forced one",
        None if analysis.overrides_checked == 0
        else len(analysis.override_failures) == 0,
        "no forced arm was watched, so the override is still unchecked"
        if analysis.overrides_checked == 0
        else (f"{len(analysis.override_failures)} of "
              f"{analysis.overrides_checked} forced arms ran a config they were "
              f"not given"
              + ("" if not analysis.override_failures
                 else "; first: " + analysis.override_failures[0]))))

    band = analysis.placebo_band
    gates.append(Gate(
        "G3 placebo", "re-timing the SAME config moves the answer by almost nothing",
        f"p90 of |replica/native - 1| < {PLACEBO_BAND:.0%}",
        None if band is None else band < PLACEBO_BAND,
        "no placebo pair timed" if band is None
        else f"p90 placebo deviation {band:.2%} over {n} cells"))

    head = analysis.headline
    lo, hi = analysis.interval if analysis.interval else (None, None)
    gates.append(Gate(
        "G4 sign", "the tuned config is not SLOWER than its own fallback",
        "median penalty >= 1.00",
        None if head is None else head >= 1.0,
        "nothing measured" if head is None
        else f"median penalty {head:.3f} -- {penalty_sentence(head)}"))

    material = None
    if head is not None:
        material = head >= MATERIAL_PENALTY and (lo is None or lo >= MATERIAL_PENALTY)
    gates.append(Gate(
        "G5 size", f"the fallback costs at least {MATERIAL_PENALTY - 1:.0%}",
        f"median penalty and the low end of its {BOOTSTRAP_BAND:.0%} bootstrap "
        f"interval are both >= {MATERIAL_PENALTY}",
        material,
        "nothing measured" if head is None
        else (f"median {head:.3f}"
              + (f", 90% interval [{lo:.3f}, {hi:.3f}]" if lo is not None else "")
              + f" -> verdict {verdict_of(head)}")))

    bm_share = analysis.knob_share_median.get("bm")
    gates.append(Gate(
        "G6 tile height", "BLOCK_SIZE_M alone explains less than half the gap",
        f"median share of the excess attributable to BLOCK_SIZE_M < "
        f"{TILE_HEIGHT_MAX_SHARE:.0%}",
        None if bm_share is None else bm_share < TILE_HEIGHT_MAX_SHARE,
        "no cell could apportion a gap" if bm_share is None
        else f"BLOCK_SIZE_M share {bm_share:.1%}"))

    group_share = analysis.knob_share_median.get("group")
    gates.append(Gate(
        "G7 swizzle", "GROUP_SIZE_M alone explains the majority of the gap",
        f"median share attributable to GROUP_SIZE_M >= {GROUP_MIN_SHARE:.0%}",
        None if group_share is None else group_share >= GROUP_MIN_SHARE,
        "no cell could apportion a gap" if group_share is None
        else f"GROUP_SIZE_M share {group_share:.1%}"))
    return gates


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

SIGN_BANNER = """\
THE SIGN, so it cannot be misread. Every ratio below is

    penalty = time(FALLBACK ladder config) / time(TUNED file config)

penalty > 1  the FALLBACK is SLOWER: running without a tuned file COSTS that much.
penalty < 1  the FALLBACK is FASTER: vLLM's tuned file lost to its own default.

The tuned side is the denominator throughout, and every penalty printed here is
followed by the words SLOWER or FASTER."""


def render_census(rows: list[tuple[str, str, bool, str]]) -> str:
    """The coverage table, which is the motivation and not the result."""
    covered = sum(1 for _, _, has, _ in rows if has)
    lines = ["## Coverage: which of these shapes has a tuned file at all",
             "",
             "The shapes MODEL_CONFIGS defines, crossed with eight serving cards,",
             f"bf16, at vLLM {VLLM_TAG}. Derived from the SHIPPED file listing and",
             "not from the four files this repo vendors, so a `ladder` here means",
             "vLLM ships nothing for that shape rather than that nobody copied it.",
             ""]
    gpus = list(dict.fromkeys(g for _, g, _, _ in rows))
    models = list(dict.fromkeys(m for m, _, _, _ in rows))
    labels = [g.replace("NVIDIA_", "")[:12] for g in gpus]
    widths = [max(len(label), len("ladder")) for label in labels]
    name_width = max((len(m) for m in models), default=8)
    lines.append("| " + "model".ljust(name_width) + " | " + " | ".join(
        label.ljust(w) for label, w in zip(labels, widths, strict=True)) + " |")
    lines.append("|" + "---|" * (len(gpus) + 1))
    by = {(m, g): has for m, g, has, _ in rows}
    for model in models:
        cells = ["tuned" if by[(model, g)] else "ladder" for g in gpus]
        lines.append("| " + model.ljust(name_width) + " | " + " | ".join(
            c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |")
    lines.append("")
    lines.append(f"{covered} of {len(rows)} pairs have a tuned bf16 file "
                 f"({covered / max(len(rows), 1):.1%}). The rest run the ladder, "
                 f"and this script is what the ladder costs.")
    return "\n".join(lines)


#: The timing spread this design is sized against, as a fraction of one cell's
#: time. MEASURED, not assumed: the range and median of `timing_spread_median`
#: over the 26 published `*.report.json` files under `results/published`, which
#: is every arm this repository has that records one (min 0.0039, median 0.0077,
#: max 0.0182 on 2026-09-02).
MEASURED_SPREAD_MEDIAN = 0.0077
MEASURED_SPREAD_MAX = 0.0182

#: Two-sided 5% at 80% power, the convention `replicate_noise_floor` uses for
#: every MDE it prints. Named rather than inlined so a reader can see nothing
#: here was chosen to make a gate pass.
MDE_LEVEL = 0.05
MDE_POWER = 0.80


def mde_of_ratio(spread: float, reps: int) -> float:
    """Smallest fallback/tuned PENALTY this design can resolve, as a fraction.

    The design is one number per arm per cell compared as a ratio, so the
    quantity that must clear the noise is a difference of two log times, each a
    median over `reps` repeats; a ratio inherits both spreads, hence the
    sqrt(2). With sigma imported rather than estimated inside the run the test
    is a known-variance z test, the stricter of the two forms available and the
    only one evaluable at these repeat counts. `statistics.NormalDist` supplies
    the quantiles so no distribution code is written twice in this repository.
    """
    if reps < 1:
        raise ValueError(f"an MDE needs at least one repeat, got {reps}")
    if spread <= 0:
        raise ValueError(f"an MDE needs a positive spread, got {spread}")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1.0 - MDE_LEVEL / 2.0) + normal.inv_cdf(MDE_POWER)
    return z * spread * math.sqrt(2.0 / reps)


def render_mde(args, spreads=None) -> str:
    """What this design can see, printed BEFORE the box is rented.

    B14: no arm in this study stated a minimum detectable effect, so G5 could
    pass or fail without anyone knowing whether the design could resolve the
    difference either way. The effect G5 is about is `MATERIAL_PENALTY - 1`,
    the 15% the study named in advance. Both ends of the measured spread are
    printed because the answer can differ between them and one number would
    hide which. `spreads` overrides the pair, which is how the
    CANNOT-RESOLVE branch is planted in the tests.
    """
    spreads = spreads or (("median", MEASURED_SPREAD_MEDIAN),
                          ("worst", MEASURED_SPREAD_MAX))
    effect = MATERIAL_PENALTY - 1.0
    lines = ["## What this design can see (MDE)", "",
             f"Effect under test: G5's registered {effect:.0%} penalty.",
             "Noise assumption: per-cell timing spread, MEASURED over the 26 "
             f"published reports; median {MEASURED_SPREAD_MEDIAN:.2%}, worst "
             f"{MEASURED_SPREAD_MAX:.2%}.",
             f"Design: {args.reps} round-robin repeats per arm, compared as a "
             "ratio against the tuned arm.", ""]
    for label, spread in spreads:
        mde = mde_of_ratio(spread, args.reps)
        verdict = ("resolves the effect" if mde <= effect
                   else "CANNOT resolve the effect")
        lines.append(f"  at the {label:<7} spread {spread:.2%}:  MDE "
                     f"{mde:.3f}   {verdict}")
    lines += ["",
              "An MDE above the effect does not make a PASS wrong; it makes a "
              "FAIL uninformative,",
              "and it is the number to raise --reps against before spending "
              "pod minutes."]
    return "\n".join(lines)


def render_plan(cells: list[Cell], notes: list[str]) -> str:
    """Both configs of every cell, and which knob groups differ.

    Printed BEFORE any timing so the reader sees what is being compared before
    seeing a number that might make them want a different comparison.
    """
    lines = ["## The plan: two configs per cell, DERIVED, before anything runs",
             ""]
    for note in notes:
        lines.append(f"  dropped -- {note}")
    if notes:
        lines.append("")
    lines.append("| model | T | key | tuned (from file) | fallback (ladder) "
                 "| differs in |")
    lines.append("|---|---|---|---|---|---|")
    for cell in cells:
        groups = ", ".join(cell.differing_groups) or "NOTHING (excluded)"
        lines.append(f"| {cell.model} | {cell.num_tokens} | "
                     f"{cell.tile.config_key_derived} | "
                     f"`{format_config(cell.tuned)}` | "
                     f"`{format_config(cell.fallback)}` | {groups} |")
    same_bm = sum(1 for c in cells
                  if c.tuned["BLOCK_SIZE_M"] == c.fallback["BLOCK_SIZE_M"])
    lines += ["",
              f"{len(cells)} cells. BLOCK_SIZE_M AGREES in {same_bm} of them, so "
              f"in {same_bm} cells any penalty is 0% tile height by "
              f"construction.",
              f"GROUP_SIZE_M differs in "
              f"{sum(1 for c in cells if 'group' in c.differing_groups)}."]
    return "\n".join(lines)


def render_results(analysis: Analysis) -> str:
    """The per-cell table, in time and in ratio, with the sign spelled out."""
    lines = ["## Per cell: what the ladder cost", "",
             "| model | T | tuned ms | fallback ms | penalty | reads as | "
             "placebo |",
             "|---|---|---|---|---|---|---|"]
    for res in analysis.cells:
        if res.penalty is None:
            lines.append(f"| {res.cell.model} | {res.cell.num_tokens} | -- | -- "
                         f"| -- | EXCLUDED: {res.excluded} | -- |")
            continue
        placebo = "--" if res.placebo is None else f"{res.placebo:.3f}"
        lines.append(
            f"| {res.cell.model} | {res.cell.num_tokens} | "
            f"{res.native_ms:.4f} | {res.fallback_ms:.4f} | "
            f"{res.penalty:.3f} | {penalty_sentence(res.penalty)} | {placebo} |")
    lines.append("")
    for model, med in sorted(analysis.per_model.items()):
        lines.append(f"  {model}: median penalty {med:.3f} -- "
                     f"{penalty_sentence(med)}")
    return "\n".join(lines)


def render_decomposition(analysis: Analysis) -> str:
    """One knob group at a time, from the tuned config, with the residual named.

    These are ONE-AT-A-TIME effects from a common baseline. They are not a
    partition and they do not have to sum to the whole gap; whatever is left is
    the interaction between the knobs, and it is printed rather than dropped so
    that a reader cannot mistake four shares for an explanation of 100%.
    """
    lines = ["## Which knob is the cost? One group moved at a time", "",
             "| model | T | full gap | BLOCK_SIZE_M | BLOCK_N/K | GROUP_SIZE_M | "
             "warps/stages | residual |",
             "|---|---|---|---|---|---|---|---|"]
    for res in analysis.cells:
        if res.penalty_forced is None:
            continue
        cols = []
        total = res.penalty_forced - 1.0
        explained = 0.0
        for name in ("bm", "nk", "group", "warpstages"):
            ratio = res.knob.get(name)
            if ratio is None:
                cols.append("--")
                continue
            explained += ratio - 1.0
            share = knob_share(ratio, res.penalty_forced)
            cols.append(f"{ratio:.3f}"
                        + ("" if share is None else f" ({share:.0%})"))
        residual = "--" if total <= 0 else f"{(total - explained) / total:+.0%}"
        lines.append(f"| {res.cell.model} | {res.cell.num_tokens} | "
                     f"{res.penalty_forced:.3f} | " + " | ".join(cols)
                     + f" | {residual} |")
    lines.append("")
    for name, share in analysis.knob_share_median.items():
        keys = "/".join(KNOB_GROUPS[name])
        lines.append(f"  {keys}: median share of the excess "
                     + ("undetermined" if share is None else f"{share:.1%}"))
    return "\n".join(lines)


def render_headline(analysis: Analysis) -> str:
    """The one number, said in a full sentence with both sides named."""
    head = analysis.headline
    if head is None:
        return ("## Headline\n\nNothing was measured, so there is no headline. "
                "That is not a null result.")
    lo_hi = ("" if analysis.interval is None
             else f", {BOOTSTRAP_BAND:.0%} bootstrap interval over cells "
                  f"[{analysis.interval[0]:.3f}, {analysis.interval[1]:.3f}]")
    return (f"## Headline\n\n"
            f"Median over {len(analysis.measured)} cells with a genuine config "
            f"difference: penalty {head:.3f}{lo_hi}.\n\n"
            f"In words: {penalty_sentence(head)}. A deployment on a "
            f"(model, shard, card) with no tuned vLLM config runs the fallback "
            f"ladder and pays {100 * (head - 1):.1f}% more time per MoE layer "
            f"than the same layer on the same card with a tuned file.\n\n"
            f"Verdict band: {verdict_of(head)} "
            f"(FOOTNOTE <{MODEST_PENALTY}, MODEST <{MATERIAL_PENALTY}, "
            f"MATERIAL >={MATERIAL_PENALTY}).")


def render_report(header: str, analysis: Analysis, gates: list[Gate],
                  stopped: str = "") -> str:
    """The exact text written to report.md, assembled in one testable place.

    A function rather than four appends inside `main` so that the file a reader
    finds on the volume after the pod is gone can be checked without a pod. The
    partial-run note and the list of arms that produced no timing come LAST and
    are never omitted: a report that quietly drops the arms that failed is a
    report that overstates how much of the plan actually ran.
    """
    body = "\n\n".join([
        header,
        render_results(analysis),
        render_decomposition(analysis),
        render_headline(analysis),
        "## Gates\n\n```\n" + render_gates(gates) + "\n```",
    ])
    if stopped:
        body += f"\n\nPARTIAL RUN: {stopped}."
    if analysis.compile_failures:
        body += ("\n\n## Arms that produced no timing\n\n"
                 + "\n".join(f"- {line}" for line in analysis.compile_failures))
    return body


PREDICTIONS_TEXT = f"""\
## Predictions, registered before the run

VALIDITY -- a FAIL here means no number on this page may be quoted.
  G0  every arm computes the same layer   max rel err <= {OUTPUT_REL_TOL:g}
  G1  vLLM loads the config we DERIVE     zero observed/derived mismatches
  G2  override_config really forces       zero arms running an unforced config
  G3  placebo band is small               p90 |replica/native - 1| < {PLACEBO_BAND:.0%}

CLAIM -- a FAIL here is a result, not a broken run.
  G4  tuning is not harmful          median penalty >= 1.00
  G5  the ladder costs materially    median and interval low end >= {MATERIAL_PENALTY}
  G6  tile height is NOT the story   BLOCK_SIZE_M share < {TILE_HEIGHT_MAX_SHARE:.0%}
  G7  the swizzle IS the story       GROUP_SIZE_M share >= {GROUP_MIN_SHARE:.0%}

G7 is the only one with an independent reason to believe it: the alpha refit
found alpha falling with GROUP_SIZE_M (0.570 at 1, 0.488 at 16), which is what a
swizzle-for-L2-reuse mechanism predicts, and the ladder pins GROUP_SIZE_M to 1
until M//E > 128, i.e. across the whole decode range."""


# --------------------------------------------------------------------------
# environment detection, so the script is testable off the box
# --------------------------------------------------------------------------

def detect_environment() -> dict:
    """What this machine can actually run, and the name of what is missing.

    Returns rather than raises, because "no GPU" is a supported mode of this
    script and not an error: the plan, the census and the predictions are all
    arithmetic over vLLM's shipped configs and are worth printing on a laptop the
    day before the pod goes up.
    """
    env = {"torch": False, "cuda": False, "vllm": False, "gpu_name": None,
           "torch_version": "", "vllm_version": "", "missing": []}
    try:
        import torch
        env["torch"] = True
        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = True
            env["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            env["missing"].append("no CUDA device (torch.cuda.is_available() is False)")
    except ImportError as exc:
        env["missing"].append(f"torch is not importable: {exc}")
    try:
        import vllm
        env["vllm"] = True
        env["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except ImportError as exc:
        env["missing"].append(f"vllm is not importable: {exc}")
    return env


def version_warning(env: dict) -> str:
    """Say so loudly when the installed vLLM is not the one we derive against.

    `tile_resolve` reproduces v0.27.1's lookup and its snapshot is v0.27.1's
    config tree. Against a different release both the tuned file and the ladder
    may have moved, so G1 would fail for a reason that has nothing to do with the
    derivation being wrong.
    """
    installed = env.get("vllm_version") or ""
    if not installed or installed.lstrip("v") == VLLM_TAG.lstrip("v"):
        return ""
    return (f"WARNING: vLLM {installed} is installed but every config on this "
            f"page is derived from {VLLM_TAG}. G1 checks the derivation against "
            f"what actually loaded, so read a G1 failure as a version "
            f"difference first.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(TUNED_H200_MODELS),
                    help="only shapes with a tuned file on this card can be "
                         "measured; the rest are dropped with a message")
    ap.add_argument("--tokens", default=",".join(str(t) for t in DEFAULT_TOKENS))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--routing", default="uniform",
                    choices=list(RoutingSpec.KINDS[:-1]))
    ap.add_argument("--arms", default="all",
                    help=f"comma list from {','.join(ARM_ORDER)}, or 'all'")
    ap.add_argument("--reps", type=int, default=3,
                    help="round-robin repeats; arms are interleaved inside each")
    ap.add_argument("--iters", type=int, default=15,
                    help="RETIRED as a timing knob on 2026-09-02 and kept in "
                         "the run id. moe.bench.timing.time_kernel sizes the "
                         "iteration count per arm from --cell-budget-ms and "
                         "the warmup's own queue-deep per-call time, which is "
                         "the only sizing that holds a trial to a duration. It "
                         "stays in the id because rows measured at a different "
                         "count exist on disk and must not be resumed into")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED 2026-09-02: a T=1 "
                         "decode cell and a T=4096 prefill cell need three "
                         "orders of magnitude of different call counts to reach "
                         "the same clock, and this grid spans exactly that")
    ap.add_argument("--cell-budget-ms", type=float, default=200.0,
                    help="target duration of ONE trial; the iteration count is "
                         "derived from it and the warmup's per-call time")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per time_kernel call")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="time with L2 warm. The default FLUSHES, the opposite "
                         "of what this script did before 2026-09-02: the arms "
                         "differ in TILE SHAPE, which is exactly what decides "
                         "how much of an expert's weights are still resident "
                         "when the next tile asks for them, so a warm L2 is "
                         "not a constant across the comparison")
    ap.add_argument("--card", default=None,
                    help="the card this run is for, as nvidia-smi names it. "
                         "Used for the run id and the results directory. There "
                         "is no default: a run labelled with a card it did not "
                         "run on is worse than a run with no label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-name", default=None,
                    help="override the device name used for the CONFIG LOOKUP; "
                         "off a GPU this is what the plan is built for")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"defaults to {default_out_dir()}/tuned_vs_fallback")
    ap.add_argument("--run-id", default=None,
                    help="defaults to a hash of the plan, so re-running the same "
                         "command RESUMES rather than starting over")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and overwrite any rows already on disk")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the plan and the predictions, measure nothing")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this long and report what exists")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    tokens = sorted({int(t) for t in args.tokens.split(",") if t})
    arms = list(ARM_ORDER) if args.arms == "all" else [
        a for a in ARM_ORDER if a in set(args.arms.split(","))]
    if "native" not in arms or "fallback" not in arms:
        raise SystemExit("--arms must include at least native and fallback; "
                         "they are the two sides of the comparison")

    env = detect_environment()
    gpu_name, card, card_note = resolve_lookup_gpu(args, env)
    run_id = args.run_id or plan_run_id(models, tokens, args.dtype, gpu_name,
                                        args.reps, args.iters, args.seed,
                                        args.routing, card=card,
                                        warmup=args.warmup,
                                        budget=args.cell_budget_ms,
                                        trials=args.trials,
                                        l2_flush=not args.no_l2_flush)
    out_dir = (args.out_dir or (default_out_dir() / "tuned_vs_fallback")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, report_path = out_dir / "timings.csv", out_dir / "report.md"

    cells, notes = plan_cells(models, tokens, args.dtype, gpu_name)
    census = coverage_census(
        [m for m in MODEL_CONFIGS if m != "toy"], list(CENSUS_GPUS), args.dtype)

    header = [
        "# What does vLLM's fallback config cost?",
        "",
        f"run id {run_id}   config lookup device `{gpu_name}`   dtype "
        f"{args.dtype}   routing {args.routing}   seed {args.seed}",
        f"card `{card}`   instrument {timing.TIMING_BASIS}",
        *(["", card_note] if card_note else []),
        f"reps {args.reps} per arm, round-robin; each repeat is one "
        f"time_kernel call of {args.trials} queue-deep trials sized to "
        f"{args.cell_budget_ms:.0f} ms, after {args.warmup:.0f} ms of warmup, "
        "L2 " + ("flushed" if not args.no_l2_flush else "WARM (--no-l2-flush)"),
        "",
        f"EVERYTHING IS SAVED TO  {out_dir}",
        f"  rows   {csv_path}",
        f"  report {report_path}",
        "Re-run the same command to resume; completed (cell, arm) pairs are "
        "skipped.",
        "",
        SIGN_BANNER,
        "",
        render_census(census),
        "",
        render_plan(cells, notes),
        "",
        render_mde(args),
        "",
        PREDICTIONS_TEXT,
    ]
    warning = version_warning(env)
    if warning:
        header += ["", warning]
    print("\n".join(header))
    prov = PV.provenance_block(instrument=timing.TIMING_BASIS,
                               warmup_ms=args.warmup,
                               target_ms=args.cell_budget_ms)
    (out_dir / "plan.json").write_text(json.dumps(prov.stamp(
        {"run_id": run_id, "card": card, "lookup_gpu": gpu_name,
         "dtype": args.dtype,
         "routing": args.routing, "seed": args.seed, "models": models,
         "tokens": tokens, "arms": arms, "vllm_tag": VLLM_TAG,
         "cells": [{"model": c.model, "num_tokens": c.num_tokens,
                    "tuned": c.tuned, "fallback": c.fallback,
                    "differing_groups": list(c.differing_groups),
                    "config_file": c.tile.config_file,
                    "config_key": c.tile.config_key_derived} for c in cells],
         "dropped": notes}), indent=2))

    if not (args.plan_only or not (env["cuda"] and env["vllm"])):
        # BEFORE the store is opened and before a single kernel runs: a row of
        # milliseconds under an assumed card is what this refuses.
        try:
            require_card_to_measure(card)
        except NoCardToLabel as exc:
            print("\n".join(["", "=" * 72,
                             "REFUSED. Nothing was measured.",
                             f"  NoCardToLabel: {exc}",
                             "=" * 72]))
            return EXIT_NOT_MEASURED

    blocked = args.plan_only or not (env["cuda"] and env["vllm"])
    if blocked:
        why = ("--plan-only was given" if args.plan_only
               else "; ".join(env["missing"]))
        print("\n".join([
            "", "=" * 72,
            "NOT A RESULT. Nothing was measured.",
            f"  reason: {why}",
            "  What is above is arithmetic over vLLM's shipped config tree plus",
            f"  its ladder, all of it DERIVED at {VLLM_TAG}. It says what WOULD be",
            "  compared. It does not say what anything costs.",
            f"  The plan was still written to {out_dir / 'plan.json'}.",
            "=" * 72]))
        return EXIT_NOT_MEASURED
    if not cells:
        print("\nNOT A RESULT: no shape in --models has a tuned config on "
              f"{gpu_name}, so there is no tuned side to price the ladder "
              "against. Pick a card that ships tuned files, or a model that has "
              "one. See the dropped list above.")
        return EXIT_NOT_MEASURED

    hooks = find_vllm_hooks()
    print(f"\noverride hook: {hooks[2]}.override_config"
          + ("" if hooks[1] else "   (no get_config in that module)"))
    meta = {"run_id": run_id, "gpu_name": env["gpu_name"] or gpu_name,
            "vllm_version": env["vllm_version"],
            "torch_version": env["torch_version"], "routing": args.routing,
            "seed": args.seed, "prov": prov,
            # The clock the roof was measured at, so `time_kernel` can score
            # LEVEL. None here rather than a guess: this script reads no
            # calibration, and `clock_flags` returns None for the level when it
            # is not given a reference, which is "not determined" and excludes
            # nothing. Threading it is what lets a caller supply one.
            "reference_clock_mhz": None}
    store = Store(csv_path, fresh=args.fresh)
    results: dict[tuple[str, int], dict[str, ArmResult]] = {}
    started = time.time()
    stopped = ""
    try:
        for index, cell in enumerate(cells, 1):
            if args.max_minutes and (time.time() - started) / 60 >= args.max_minutes:
                stopped = (f"stopped after {args.max_minutes} minutes with "
                           f"{index - 1} of {len(cells)} cells done")
                break
            print(f"  [{index}/{len(cells)}] {cell.model} T={cell.num_tokens}  "
                  f"tuned {format_config(cell.tuned)}  vs  fallback "
                  f"{format_config(cell.fallback)}", flush=True)
            results[cell.key] = measure_cell(cell, arms, args, store, meta, hooks)
    except KeyboardInterrupt:
        stopped = ("interrupted; every arm finished before the interrupt is on "
                   "disk and the same command resumes")
    finally:
        store.close()

    analysis = analyse(cells, results)
    gates = build_gates(analysis)
    report_path.write_text(
        render_report("\n".join(header), analysis, gates, stopped) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(prov.stamp({
        # `gpu_name` belongs to the PROVENANCE block: the device torch reported
        # at run time, or null with a reason. This is what the rows were
        # labelled with, and `stamp` refuses the collision if they share a key.
        "run_id": run_id, "card": card, "row_label_gpu": meta["gpu_name"],
        "vllm_version": meta["vllm_version"], "vllm_tag": VLLM_TAG,
        "sign": "penalty = fallback_time / tuned_time; >1 means fallback SLOWER",
        "headline_median_penalty": analysis.headline,
        "bootstrap_interval": list(analysis.interval) if analysis.interval else None,
        "verdict": None if analysis.headline is None else verdict_of(analysis.headline),
        "per_model_median_penalty": analysis.per_model,
        "placebo_p90_deviation": analysis.placebo_band,
        "knob_share_median": analysis.knob_share_median,
        "cells_measured": len(analysis.measured), "cells_planned": len(cells),
        "partial": stopped,
        "gates": [{"name": g.name, "kind": g.kind, "passed": g.passed,
                   "verdict": g.verdict, "rule": g.rule,
                   "observed": g.observed} for g in gates],
    }), indent=2))

    print("\n" + render_results(analysis))
    print("\n" + render_decomposition(analysis))
    print("\n" + render_headline(analysis))
    print("\n## Gates\n")
    print(render_gates(gates))
    if stopped:
        print(f"\nPARTIAL RUN: {stopped}.")
    print(f"\nEVERYTHING IS SAVED TO {out_dir}")
    print(f"  rows {csv_path}\n  report {report_path}\n"
          f"  summary {out_dir / 'summary.json'}")
    # THE EXIT CODE COMES FROM THE SHARED TABLE, over the SAME gate objects that
    # printed the RESULT lines above, so `exit_codes.classify_text` on this log
    # recomputes the code the process returned. It also finally separates the
    # two outcomes this script used to fold into 1: a VALIDITY failure (G0-G3)
    # is INVALID, its numbers are not to be believed; a CLAIM failure (G4-G7) is
    # a RESULT, and the run is finished rather than broken.
    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
