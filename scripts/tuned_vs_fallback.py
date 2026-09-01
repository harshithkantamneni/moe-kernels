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
import hashlib
import json
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

CSV_COLUMNS = (
    "run_id", "utc", "gpu_name", "vllm_version", "torch_version", "vllm_tag",
    "model", "num_experts", "intermediate_n", "dtype", "routing", "seed",
    "num_tokens", "arm", "config_origin", "identical_to_tuned",
    "config_file", "config_key", "provenance",
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
    "observed_config", "override_verified",
    "ms_median", "ms_mean", "ms_stdev", "ms_min", "n_samples",
    "rel_err_vs_native", "error",
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

    def render(self) -> str:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        return (f"[{tag}] {self.name}  {self.prediction}\n"
                f"         gate: {self.rule}\n"
                f"         saw:  {self.observed}")


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
                routing: str) -> str:
    """A run id that is a HASH OF THE PLAN, so a rerun resumes by default.

    An idempotent script whose default run id is random is not idempotent in
    practice: the second invocation writes a second directory and repeats every
    cell. Hashing the plan means "run the same command again" is the resume
    command, and changing any parameter that would invalidate the old rows
    changes the directory instead of silently mixing two experiments.
    """
    payload = json.dumps({"models": models, "tokens": tokens, "dtype": dtype,
                          "gpu": gpu_name, "reps": reps, "iters": iters,
                          "seed": seed, "routing": routing}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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


def time_calls(fn, warmup: int, iters: int) -> list[float]:
    """Per-iteration milliseconds from CUDA events. No L2 flush.

    No flush on purpose: the arms differ only in the tile schedule and run on
    identical data, and a flush would add a large fixed term to every arm plus
    its own variance, which widens the placebo band without moving the ratio the
    script is trying to resolve.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def summarise_samples(result: ArmResult, samples: list[float]) -> ArmResult:
    result.ms_median = statistics.median(samples)
    result.ms_mean = statistics.fmean(samples)
    result.ms_stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    result.ms_min = min(samples)
    result.n_samples = len(samples)
    return result


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
    for _ in range(args.reps):
        for arm in list(samples):
            try:
                with context(arm):
                    samples[arm].extend(time_calls(call, args.warmup, args.iters))
            except Exception as exc:  # noqa: BLE001
                results[arm].error = f"{type(exc).__name__}: {exc}"[:300]
                samples.pop(arm, None)

    for arm in pending:
        got = samples.get(arm)
        if got:
            summarise_samples(results[arm], got)
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
    ap.add_argument("--iters", type=int, default=15, help="timed calls per arm per rep")
    ap.add_argument("--warmup", type=int, default=8)
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
    gpu_name = args.gpu_name or env["gpu_name"] or "NVIDIA H200"
    run_id = args.run_id or plan_run_id(models, tokens, args.dtype, gpu_name,
                                        args.reps, args.iters, args.seed,
                                        args.routing)
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
        f"reps {args.reps} x {args.iters} timed calls per arm, round-robin, "
        f"{args.warmup} warmup per arm per rep",
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
        PREDICTIONS_TEXT,
    ]
    warning = version_warning(env)
    if warning:
        header += ["", warning]
    print("\n".join(header))
    (out_dir / "plan.json").write_text(json.dumps(
        {"run_id": run_id, "gpu_name": gpu_name, "dtype": args.dtype,
         "routing": args.routing, "seed": args.seed, "models": models,
         "tokens": tokens, "arms": arms, "vllm_tag": VLLM_TAG,
         "cells": [{"model": c.model, "num_tokens": c.num_tokens,
                    "tuned": c.tuned, "fallback": c.fallback,
                    "differing_groups": list(c.differing_groups),
                    "config_file": c.tile.config_file,
                    "config_key": c.tile.config_key_derived} for c in cells],
         "dropped": notes}, indent=2))

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
            "seed": args.seed}
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
    (out_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id, "gpu_name": meta["gpu_name"],
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
        "gates": [{"name": g.name, "passed": g.passed, "rule": g.rule,
                   "observed": g.observed} for g in gates],
    }, indent=2))

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
    return EXIT_GATE_FAILED if any(g.passed is False for g in gates) else EXIT_OK


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
