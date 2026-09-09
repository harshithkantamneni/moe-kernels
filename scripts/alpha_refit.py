#!/usr/bin/env python3
"""Refit `alpha`, the cost of an extra M-tile, against the DERIVED tile.

    python scripts/alpha_refit.py --pinned-set \
        --original-estimator --adversarial

`--pinned-set` reads `results/published/REFIT_SET.txt`, the ten arms
`docs/FINDINGS.md` fitted, and reproduces its 0.558 / 10,813 rows / 3,124
discriminating / 0.529-0.588. Pass CSV paths instead to fit whatever the tree
holds today; the header names which of the two you got, because the documented
glob now reads fourteen arms and answers 0.560 / 11,181 / 3,181 / 0.529-0.593.

`alpha` is the one free parameter in the tile-corrected roofline
(`docs/FINDINGS.md`, "The tile-corrected roofline"). One expert holding `r` rows
is scheduled as `ceil(r / BLOCK_M)` M-tiles, the first tile reads that expert's
weights in full, and each additional tile costs `alpha` of a fresh read because
L2 absorbs part of the re-read:

    weight bytes = N_w b (1 + alpha (M-tiles - 1)),   AI(r) = (2r/b) / Q(r)

It decides which tile heights can ever reach the compute roof at all, since
`AI -> 2 BM / (alpha b)` as `r` grows, so it is not a nuisance parameter. That
reading is an UPPER BOUND on the cap when alpha is a ladder fit: the fit's
denominator carries the first tread's activation, output and fixed cost, so the
exact cap is lower by (1 + phi + delta) (`moe/bench/ai_model.py`), and with
alpha_a unmeasured the factor is a bracket. Section 4 of the report prints it
that way.

TWO NUMBERS DISAGREE THREEFOLD AND THIS SCRIPT EXISTS TO SAY WHY. This repo
published `alpha = 0.10` (CV 12.8%) off 151 rows; arXiv:2608.13057 (TEMPO,
Aug 2026) fits the same physical parameter at about 0.33. The 0.10 was described
as "refit against the OBSERVED tile", which is true of the tile and hides where
the weakness actually is: it also came from ONE arm, ONE implementation, ONE
timing mode, and an estimator that minimises the coefficient of variation of a
POOLED ratio.

WHAT THIS SCRIPT CHANGES, in order of how much it moves the answer:

1. THE ESTIMATOR, which turns out to matter far more than the tile. Minimising
   the CV of a pooled ratio lets `alpha` absorb every between-cell difference in
   level -- model, batch, timing mode, card -- and those differences are an order
   of magnitude larger than the tile term. Worse, the largest of them runs the
   WRONG WAY: `implied_traffic_ratio` falls with batch as fixed dispatch cost
   amortises, while the tile count rises with batch, so a pooled fit pays
   `alpha` to explain a trend that has nothing to do with tiles and is pushed
   toward zero. This script fits a group intercept per
   (model, dtype, card, impl, timing mode, token count), so only rows that
   differ in tile count while agreeing on everything else can move `alpha`.
2. THE TILE, from `moe.bench.tile_resolve`, per row, DERIVED from vLLM 0.27.1's
   own lookup rather than assumed. It matters, and it is not the story: forcing
   64 on every row instead answers 0.48 and forcing 128 answers 0.65, against
   0.56 with the per-row derivation. A 35% spread, on a disagreement of 330%.
3. THE POOL, every current published arm rather than one.

WHAT `alpha` MEANS HERE, and it is narrower than "a fraction of a weight read".
`implied_traffic_ratio` is `time x achievable_bandwidth / compulsory_bytes`, so
the fit attributes to an extra tile EVERYTHING that tile costs in TIME: its
weight re-read, its padded arithmetic, its scheduling, its share of the tail.
Read as a traffic coefficient the answer is therefore an UPPER BOUND, which is
the direction that matters against TEMPO's `b2/b`, a pure byte ratio.
`--adversarial` prints a consequence of that bound which this study's own
measured crossings contradict.

ONE FIT, ONE INSTRUMENT, AND THE POOL IS REFUSED RATHER THAN MIXED. Schema v5
put the timer's name in every row (`schema.instrument_of`), and it had to,
because until 2026-09-02 `moe/bench/driver.py` -- the path every published row
came through -- timed on `time_eager`/`time_graph` while the roof and every
ladder script had moved to `time_kernel`. Those two apparatus differ in level by
construction: one warms for a COUNT of calls and sizes its iteration count from a
single isolated call on an idle GPU, the other warms for a DURATION of sustained
load and sizes it from a queue-deep per-call time, so they do not agree on what
clock the card was at while the cell ran.

That matters HERE more than it would in most analyses, and the reason is the
estimator. `alpha` is identified WITHIN a group intercept keyed on
(model, dtype, card, impl, timing mode, token count). Two rows of one cell
measured on two apparatus would share that key, so any level difference between
the instruments would be charged to `alpha` exactly the way the batch trend was
before token count entered the key -- which is most of the distance between 0.10
and the answer below. The instrument is therefore IN `cell_key`, so an intercept
can never span both. That is necessary and it is not sufficient: an instrument
that shifts a many-tile row differently from a one-tile row moves the SLOPE, and
no arm in this corpus ran one cell both ways, so nothing here can measure
whether it does. So a mixed pool is REFUSED (exit 2), with the count per
instrument printed, and the reader is told to fit each separately.
`--pool-instruments` overrides that and prints the mix beside every headline
number. It is the honest second-best rather than the default, because the
default is what gets quoted. An observation that names NO instrument is refused
by either route (`UNSTATED_INSTRUMENT`): pooling weighs two apparatus a reader
can name, and an absent fact is not one of them.

THE LEVEL VERDICT HAS A SIDE, AND THE GATE READS IT. Since 03df2d4 the
instrument's LEVEL check is a BAND around the reference clock, so a
memory-shaped cell boosted to 1980 MHz against the 1515 MHz bf16-GEMM reference
FAILS LEVEL with `clock_level_side = "high"`. On an H200 that is the normal
state of every memory-bound cell, which is every cell this fit is identified on.
`clock_gate` therefore excludes only LOW (the throttle the flag was built for)
and DRIFT; a HIGH row is ADMITTED and marked, because its time is the kernel's
and only its FIXED-roof fraction is wrong. The rows are pooled with the level
ones because nothing the fit reads depends on the fixed roof; the roof section
of the report prints the fraction against the roof at the cell's own clock
beside the fixed one and says on how many rows it is available.

Everything here is arithmetic over published CSVs: no GPU, no torch. That was
a promise this file could not keep until 2026-09-02: `moe.bench.tile_resolve`
imported `moe.quant`, which imported torch at module scope, and torch is not one
of the four dependencies `pyproject.toml` declares. Both imports are lazy now,
and `tests/test_analysis_tools.py` blocks torch at the import finder and asserts
this path still runs.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench import ai_model  # noqa: E402
from moe.bench import schema as SC  # noqa: E402
from moe.bench.bytes_model import weight_bytes_for_stage  # noqa: E402
from moe.bench.crossing import m_tiles_for_row  # noqa: E402
from moe.bench.exit_codes import REFUSED  # noqa: E402
from moe.bench.published import (  # noqa: E402
    dirty_share,
    dirty_share_line,
    filter_superseded,
    mde_line,
    sd_from_band,
    superseded_impls,
    two_sample_mde,
)
from moe.bench.ridge import rows_per_expert  # noqa: E402
from moe.bench.tile_resolve import (  # noqa: E402
    VLLM_IMPLS,
    TileNotDerivable,
    resolve_tile_for_row,
)
from moe.routing.imbalance import TileEfficiencyUndetermined  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec  # noqa: E402
from moe.stages import CANONICAL_STAGES  # noqa: E402

#: The two numbers this fit is judged against.
REPO_PUBLISHED_ALPHA = 0.10
TEMPO_ALPHA = 0.33

#: THE WITHDRAWN BAND, and what it actually is. Both ends are H200 numbers from
#: two calibrations of the SAME card: 160.3 is 701.6 TFLOP/s over 4377.2 GB/s
#: (`fp8-three-kernel`, `v2lite`) and 176.2 is 770.9 over 4374.5
#: (`fp8-refixed`, `whole-layer`), `docs/INSTRUMENTATION.md`'s six-calibration
#: table. Bandwidth reproduces to 0.06% across all six and the compute term does
#: not, so this band measures how badly the compute ceiling REPRODUCES, not how
#: wide any device's ridge is. It was then quoted on A100 arms, whose own ridge
#: is 145.8, where neither end belongs to the attached card at all. On
#: 2026-09-02 it was withdrawn from all 26 published reports by
#: `scripts/rescore_published_reports.py`.
#:
#: NOTHING IN THIS FILE SCORES AGAINST IT ANY MORE. `card_ridge_bands()` is
#: where a ceiling comes from. `scripts/group_m_alpha_sweep.py` resolved its
#: own ridge from the attached card's calibration on 2026-09-03 and no longer
#: reads this; the name survives only for `tests/test_group_m_sweep.py`'s
#: loader, which imports this module and is not this slice's to edit. Delete
#: it when that test stops loading it.
RIDGE_BAND = (160.3, 176.2)

#: torch's `grouped_mm` tile, OBSERVED under claim C1 by reading the CUTLASS
#: kernel name out of the profiler, and fixed at 64 by Hopper's
#: `wgmma.mma_async.m64nNk16` whatever the shape. Not derived from vLLM's config
#: tree, which is why these rows are collected separately and labelled.
CUTLASS_BLOCK_M = 64
CUTLASS_IMPLS = frozenset({"torch_grouped_mm_up", "torch_grouped_mm_down"})

#: The two words `schema.Row.clock_level_side` carries on a LEVEL failure.
#: They ARE `timing.LEVEL_LOW` and `timing.LEVEL_HIGH`, spelled here rather
#: than imported because `moe/bench/timing.py` imports torch at module scope
#: and this script's header promises never to need it; `tests/test_alpha_refit.py`
#: pins the two spellings to the instrument's own so they cannot drift.
LEVEL_LOW = "low"
LEVEL_HIGH = "high"
#: Every value the column may hold: "" when LEVEL held, was undetermined, or
#: the row predates the side. Any other word is refused, not defaulted.
LEVEL_SIDES = frozenset({"", LEVEL_LOW, LEVEL_HIGH})

#: What a HIGH-side row is marked with when it is admitted. One sentence, kept
#: in one place, because it is printed beside the count and quoted in tests.
HIGH_SIDE_NOTE = ("LEVEL failed HIGH: the cell ran above the band around the "
                  "reference clock, so its fixed-roof fraction is not "
                  "comparable; use pct_of_roof_at_cell_clock")

#: `Observation.roof_note` on a row that predates the per-row roof (schema v6).
ROOF_PREDATES = "not available (v<6 row)"

#: The manifest that pins the pool FINDINGS' numbers were fitted on. See the
#: file itself for why a glob is not a reproducible input set (audit B8).
DEFAULT_REFIT_SET = (Path(__file__).resolve().parent.parent / "results"
                     / "published" / "REFIT_SET.txt")

#: Where a pinned arm's rows live, relative to the manifest's directory.
REFIT_SET_GLOB = "run_*.csv"

#: The arm the published 0.10 was fitted on. Named, because "torch grouped_mm
#: rows" is not a reproduction: pooled over every current arm those are 1,728
#: rows spanning two cards and two dtypes, the CV of the pooled ratio is 1245%,
#: and the objective is flat to the fourth decimal. The write-up's basis was one
#: arm, and only on that arm do the published 151 rows, 27 discriminating, and
#: the 13.1% / 12.8% / 17.5% CV column come back.
ORIGINAL_ALPHA_ARM = "2026-08-22-standard-sweep"

#: FINDINGS C2's one-stage bf16 crossing for mixtral, in tokens. Quoted rather
#: than recomputed because this script does not read crossings; it is used only
#: to check the fitted `alpha` against a number the study already published.
MIXTRAL_ONE_STAGE_CROSSING_TOKENS = 938

#: Columns `m_tiles_for_row` reads. Carried per observation so a tile count can
#: be recomputed at another block size without holding a 94-column dict.
TILE_COLUMNS = ("load_total_rows", "load_active_experts", "load_max_rows",
                "load_tile_eff_bm64", "load_tile_eff_bm128")

#: An M-tile count this far above the active-expert count is FLOATING-POINT DUST,
#: not a tile.
#:
#: THE BUG THIS FIXES, which was live for one run of this script. `m_tiles_for_row`
#: computes `total_rows / (tile_eff * block_m)`, and where `tile_eff` was
#: reconstructed the two divisions cancel exactly in algebra and to about 1e-13 in
#: binary. A bare `> 0` test then called 93 single-tile rows "discriminating",
#: handed them an `x` of about 1e-16, and the fit answered -0.850: the lower
#: BOUND, because a residual that flat is minimised by running away. An
#: unidentified split has to report that it is unidentified, not a boundary.
TILE_EPSILON = 1e-6

#: The `instrument` of an Observation whose builder did not say what timed it.
#:
#: A DEFAULT THAT REFUSES, and it replaced a default that lied. The field used
#: to default to `schema.LEGACY_INSTRUMENT`, justified as convenience for "a
#: synthetic Observation a test builds" -- but the one non-test caller that
#: builds an Observation by hand, `scripts/group_m_alpha_sweep.py:718`, passes
#: seventeen keywords and not that one, so every cell it measured with
#: `timing.time_kernel` claimed the retired apparatus, inside a script whose own
#: provenance block writes `instrument=timing.TIMING_BASIS`. No number moved,
#: because the mislabel was uniform and `cell_key` only needs the component to
#: be constant; the point is that this module exists to end exactly that
#: confusion and a default reintroduced it silently.
#:
#: So the default now names its own absence, and `_report_instruments` refuses
#: a pool containing it -- with or without `--pool-instruments`, because
#: pooling is a decision about two known apparatus and this is not one. `collect`
#: always sets the real name off the row, so no CLI path can meet it.
UNSTATED_INSTRUMENT = "<not stated by the caller>"


@dataclass(frozen=True)
class Observation:
    """One published row, reduced to what the fit needs.

    `extra_tile_bytes` is `W_expert x (M-tiles - active experts)`: the weight
    bytes a tile-corrected model charges ON TOP of the compulsory minimum, which
    already counts each active expert's weights exactly once. It is 0 whenever
    every active expert fits inside one tile, and such a row constrains `alpha`
    not at all -- it contributes only a group intercept. `discriminating` says
    which is which, because "10,813 rows" and "3,124 rows that can move the
    answer" are very different claims and only one of them is honest.
    """

    traffic_ratio: float
    compulsory_bytes: float
    per_expert_bytes: float
    active_experts: float
    m_tiles: float
    block_m: int
    group_m: int
    tile_provenance: str
    model: str
    dtype: str
    gpu: str
    impl: str
    tokens: int
    routing: str
    l2_flush: bool
    cuda_graph: bool
    #: Just `TILE_COLUMNS`, so `at_block_m` can recount tiles.
    tile_columns: tuple[tuple[str, str], ...]
    #: The published arm directory this row came from. A LEVEL OF THE CLUSTER
    #: BOOTSTRAP, not decoration: rows inside one arm share a pod, a session, a
    #: thermal state and one calibration, so they are nowhere near independent
    #: of each other, and a band that resamples cells alone treats fourteen
    #: pods' worth of shared state as 562 independent observations.
    #:
    #: DEFAULTED so a synthetic Observation built by a test does not have to
    #: invent a provenance it does not have. Every observation `collect` builds
    #: carries the real directory name.
    arm: str = ""
    #: `git_dirty` off the row, AS THE RAW COLUMN and not as a bool. 44,872 of
    #: the 100,144 published rows were measured from a tree with uncommitted
    #: changes and no analysis path in this repository read the column (audit
    #: A6). The raw string is kept because "" means the arm predates the column
    #: and cannot answer, which is not the same as clean, and a bool would
    #: silently fold the two together.
    dirty_raw: str = ""
    #: WHICH TIMER produced the `ms_p50` this row's traffic ratio came from,
    #: through `schema.instrument_of`. A CLUSTER OF THE FIT and not a label: it
    #: is in `cell_key`, so an intercept never spans two apparatus, and the
    #: report refuses to pool two of them without being told to.
    #:
    #: DEFAULTED TO A REFUSAL and never to an apparatus; see
    #: `UNSTATED_INSTRUMENT` for the caller that a convenient default
    #: mislabelled. `collect` always sets it from the row itself.
    instrument: str = UNSTATED_INSTRUMENT
    #: Which way this row's LEVEL verdict failed, off the row. `LEVEL_HIGH` for
    #: a boosted cell `clock_gate` ADMITTED (its time is the kernel's; only the
    #: fixed-roof fraction beside it is wrong), `LEVEL_LOW` only when
    #: `--include-throttled` re-admitted a throttled row, "" when LEVEL held,
    #: was undetermined, or the row predates the side. The report counts the
    #: HIGH rows beside the fit so a reader can see how much of the pool the
    #: fixed roof misdescribes.
    clock_level_side: str = ""
    #: `pct_of_achieved_tflops` off the row: `tflops` against the FIXED roof,
    #: the calibration's peak at the calibration's clock, identical on every
    #: row of a run. Inflated by `load / reference` on a HIGH-side row.
    pct_fixed_roof: float = 0.0
    #: `pct_of_roof_at_cell_clock` off the row when the driver scored it
    #: (`schema.has_cell_clock_roof`): `tflops` against the roof AT THE CLOCK
    #: THE CELL RAN. None when it is not on the row, and never 0.0 in its
    #: place, because 0.0 is the driver's "not scored" and a median over it
    #: would be a number about nothing.
    pct_cell_clock_roof: float | None = None
    #: Why the per-row roof is absent, in the driver's words; "" when scored,
    #: `ROOF_PREDATES` on a row from before schema v6.
    roof_note: str = ""

    @property
    def extra_tile_bytes(self) -> float:
        extra = self.m_tiles - self.active_experts
        return self.per_expert_bytes * extra if extra > TILE_EPSILON else 0.0

    @property
    def x(self) -> float:
        """The regressor: extra tile bytes as a fraction of the compulsory total."""
        return self.extra_tile_bytes / self.compulsory_bytes

    @property
    def discriminating(self) -> bool:
        return self.extra_tile_bytes > 0.0

    @property
    def mode(self) -> str:
        return ("L2-cold" if self.l2_flush else "L2-warm") + (
            "/graph" if self.cuda_graph else "/eager")

    @property
    def dirty(self) -> bool:
        """Was this row measured from a tree with uncommitted changes?

        False for a row that never recorded the column, which `dirty_share`
        counts and reports separately; see `dirty_raw`.
        """
        return dirty_share([{"git_dirty": self.dirty_raw}])[0] == 1

    @property
    def context(self) -> tuple:
        """`cell_key` with the two timing-mode flags REMOVED.

        The unit of the paired basis contrast. A context is one physical cell --
        one model, dtype, card, implementation and token count -- observed under
        whichever timing modes that arm happened to run, and pairing on it is
        what turns "alpha differs between modes" into a statement about the
        instrument rather than about which models each mode was pointed at.
        """
        return (self.model, self.dtype, self.gpu, self.impl, self.tokens,
                self.instrument)

    def at_block_m(self, block_m: int) -> Observation | None:
        """The same row scheduled at another tile height, or None if uncountable.

        Recounts from the stored load columns rather than rescaling `m_tiles`:
        tiles are a CEILING per expert, so halving the block does not double the
        count, and a rescaled number would be wrong by up to one tile per expert
        in a direction that depends on the histogram.
        """
        try:
            tiles = m_tiles_for_row(dict(self.tile_columns), block_m)
        except (TileEfficiencyUndetermined, SC.TileConfigUnrecorded, ValueError):
            return None
        return dataclasses.replace(self, m_tiles=tiles, block_m=block_m)


def expert_weight_bytes(spec: BenchSpec, covers: str) -> float:
    """Weight bytes for ONE expert over the stages this span covers.

    The two GEMM stages only. `router` also has a weight, and it is a single
    dense `[E, H]` gate read once per layer no matter how the rows are tiled, so
    multiplying it by an M-tile count would charge a re-read that cannot happen.
    `__pipeline__` rows record `covers = "all"`, which is where that mistake
    would have landed.
    """
    stages = CANONICAL_STAGES if covers == "all" else tuple(covers.split("+"))
    return float(sum(weight_bytes_for_stage(spec, s, 1) for s in stages
                     if s in ("up_gemm", "down_gemm")))


def clock_gate(row: dict, instrument: str) -> str:
    """Why this row's CLOCK STATE disqualifies it, or "" to keep it.

    ASKED OF THE INSTRUMENT THAT WROTE THE ROW, because the two apparatus have
    no flag in common and reading one row's answer out of the other's column is
    how a filter becomes a no-op without anybody noticing.

    A pre-v5 row carries `throttled`: two clock samples taken at IDLE INSTANTS
    either side of the cell, flagged on a >5% DROP. `moe/bench/timing.py`
    documents what that actually detects -- whether the START sample caught the
    idle boost -- and it fired on 91% of vLLM rows above T=4096 while flagged
    and unflagged replicates of the same cell timed at ratio 0.998. It is kept
    as this pool's gate anyway, because it is the only clock evidence those rows
    carry and dropping the gate entirely would admit rows nothing checked. What
    changed is that the drop is now NAMED for what it is, and its skew is stated
    beside the answer: over the pinned set it removes 0.0% of rows at T=1 and
    100% at T=16384, along the very axis alpha is identified on.

    A v5 row carries the three under-load verdicts instead, and a row is dropped
    when any of them FAILED. "undetermined" is kept: the check could not be run
    (no NVML, a trial too short for the poller), which is not evidence against
    the number, and folding it into a failure would silently discard every row
    measured in a container that forbids NVML.

    A LEVEL FAILURE IS READ WITH ITS SIDE, AND ONLY LOW EXCLUDES. Since 03df2d4
    the instrument scores LEVEL as a band around the reference clock, and a
    memory-shaped cell boosted to 1980 MHz against the 1515 MHz bf16-GEMM
    reference fails it with `clock_level_side = "high"`. That is not a
    throttle: the cell's time is the kernel's, and what is wrong is the
    FIXED-roof fraction beside it, inflated by the ratio, which this fit never
    reads. Until 2026-09-08 this gate dropped such a row on the bare verdict,
    which on an H200 is every memory-bound cell, which is every cell alpha is
    identified on: a rental re-measured on the instrument would have had its
    memory-bound half silently removed from the refit, and `--include-throttled`
    could not rescue it because that flag also re-admits the genuinely
    throttled. So: LOW excludes, DRIFT excludes, host-bound excludes; HIGH is
    admitted with "" here and the row is marked (`Observation.clock_level_side`,
    `HIGH_SIDE_NOTE`) so the report can count it. A LEVEL failure carrying no
    side is excluded and says why: a pre-v6 LEVEL was one-sided and could only
    fail low, and a v6 row always carries one. A side word outside
    `LEVEL_SIDES` is refused with an exception, never read as either.

    WHY HIGH ROWS MAY BE POOLED WITH THE LEVEL ONES. The fit's per-row inputs
    are `implied_traffic_ratio`, `compulsory_bytes`, the per-expert weight bytes
    and the M-tile count. The ratio is `driver._apply_cost`'s
    `implied_traffic_ratio(bytes, ms, bandwidth)`: measured time over the
    compulsory bytes at the BANDWIDTH ceiling, which is not rescaled with the SM
    clock (HBM does not run on it; calibrate.py measured 1.7% sensitivity). The
    fixed compute roof enters only the driver's memory-bound CLASSIFICATION,
    which decides whether the column is written at all, and on a boosted card
    that under-classifies (the effective ridge is higher), so it omits rows
    that earned the column rather than writing it on rows that did not. No
    admitted row's regressor or response therefore depends on the fixed roof,
    and one fit over both sides is one fit over one quantity. The report still
    splits the pool by side under "is alpha a scalar?" so the claim is checked
    on the data rather than only argued here.

    AN UNTIMED ROW IS ITS OWN ANSWER, and it used to be nobody's. The driver
    stamps `NO_INSTRUMENT` on every cell it declines or fails, and such a row
    fell into the v5 branch, read three default "undetermined" words and was
    ADMITTED. It reached no fit only because both callers drop `ms_p50 <= 0` a
    few lines earlier, which is an incidental filter doing a gate's job -- the
    accident this function's own first paragraph exists to prevent. So it is
    named here, ahead of the branch, and `schema.timing_verdict` now refuses the
    row underneath as well, so the two cannot drift apart again.
    """
    if instrument == SC.NO_INSTRUMENT:
        return ("never timed: the driver wrote this row for a cell it declined "
                "or failed, so there is no clock evidence to gate on and no "
                "measurement to admit")
    if instrument == SC.LEGACY_INSTRUMENT:
        if SC.row_bool(row, "throttled"):
            return ("throttled (the RETIRED idle-instant flag: it fires when "
                    "the START sample caught the idle boost, and its drop rate "
                    "runs 0% at T=1 to 100% at T=16384)")
        return ""
    failed = [c for c in SC.TIMING_VERDICT_COLUMNS
              if SC.timing_verdict(row, c) == SC.VERDICT_FAILED]
    if "clock_level_ok" in failed:
        side = level_side_of(row)
        failed.remove("clock_level_ok")
        if side == LEVEL_LOW:
            failed.insert(0, "clock_level_ok failed LOW (the card sat below "
                             "the band around the reference clock: the "
                             "throttle the flag was built for)")
        elif side == "":
            failed.insert(0, "clock_level_ok failed with no side recorded (a "
                             "pre-v6 LEVEL was one-sided and could only fail "
                             "low; a v6 row always carries one)")
        # LEVEL_HIGH: admitted. The time is the kernel's; the mark is carried
        # on the Observation and counted by the report.
    if failed:
        return ("under-load check failed on the instrument: "
                + ", ".join(failed))
    return ""


def level_side_of(row: dict) -> str:
    """`clock_level_side` off a row, as one of `LEVEL_SIDES`, or raise.

    "" for an absent column, an empty one, or the UNRECORDED stamp `read_csv`
    puts on a pre-v6 row: none of those is a side. Any other word outside the
    set is refused, because a side no branch of `clock_gate` matches would be
    admitted by falling through, which is the silent default this gate exists
    to prevent.
    """
    value = row.get("clock_level_side")
    if value is None or value == "" or value == SC.UNRECORDED:
        return ""
    side = str(value)
    if side not in LEVEL_SIDES:
        raise ValueError(
            f"clock_level_side {side!r} is not one of {sorted(LEVEL_SIDES)}; "
            "the instrument writes only those and a word it never wrote is "
            "not a verdict")
    return side


def roof_fractions(row: dict) -> tuple[float, float | None, str]:
    """The two compute-side fractions off a row, and why the second is absent.

    `(pct_of_achieved_tflops, pct_of_roof_at_cell_clock or None, roof_note)`.
    The first is the FIXED-roof figure every row since v2 carries; the second
    is against the roof at the clock the cell ran, present only on a v6 row
    the driver scored (`schema.has_cell_clock_roof`). None, not 0.0, when it
    is absent: 0.0 is the driver's "not scored" and a reader that medianed it
    would quote a fraction of nothing. The note is the driver's own reason on
    a v6 row it refused, `ROOF_PREDATES` on a row from before the column, and
    "" when scored.
    """
    fixed = SC.row_float(row, "pct_of_achieved_tflops")
    if SC.has_cell_clock_roof(row):
        return fixed, SC.row_float(row, "pct_of_roof_at_cell_clock"), ""
    note = row.get("roof_note")
    if note is None or note == SC.UNRECORDED:
        return fixed, None, ROOF_PREDATES
    if "roof_at_cell_clock_tflops" not in row:
        return fixed, None, ROOF_PREDATES
    return fixed, None, (str(note) or "not scored, and the driver recorded "
                                      "no reason (a v6 row with an empty "
                                      "roof_note and no roof)")


def instrument_mix(observations) -> collections.Counter:
    """How many admitted rows each timing apparatus produced."""
    return collections.Counter(o.instrument for o in observations)


def collect(paths, census: collections.Counter, *, cutlass: bool = False,
            include_throttled: bool = False) -> list[Observation]:
    """Every published row that can carry a tile-corrected traffic fit.

    `cutlass=False` collects the vLLM Triton spans and derives each row's tile
    from vLLM 0.27.1's lookup. `cutlass=True` collects torch's `grouped_mm`
    spans at the C1-observed 64 instead, which is the pool the published 0.10
    came from, and is kept separate so the derived and the observed never pool by
    accident. SGLang is in neither: it ships its own tuned tree, nothing here
    models it, and substituting vLLM's answer would be the exact failure this
    work exists to correct.

    Every rejection is counted rather than dropped, because a filter that
    silently removes 90% of its input looks the same as one that removes nothing.
    """
    kept, dropped_arms = filter_superseded(paths)
    for path in dropped_arms:
        census[f"arm superseded whole: {path.parent.name}"] += 1
    out: list[Observation] = []
    for path in kept:
        retired = superseded_impls(path) or set()
        for row in SC.read_csv(path):
            impl = str(row.get("impl", ""))
            if impl in retired:
                census["implementation retired by a later arm"] += 1
                continue
            if impl not in (CUTLASS_IMPLS if cutlass else VLLM_IMPLS):
                census["implementation outside this pool"] += 1
                continue
            if float(row.get("ms_p50") or 0.0) <= 0.0:
                census["never timed (ms_p50 = 0 is not a measurement)"] += 1
                continue
            if not SC.passed(row):
                census["failed the correctness gate"] += 1
                continue
            instrument = SC.instrument_of(row)
            gate = clock_gate(row, instrument)
            if gate and not include_throttled:
                census[gate] += 1
                continue
            ratio = SC.row_float(row, "implied_traffic_ratio")
            if ratio <= 0.0:
                census["no implied_traffic_ratio: the driver called the cell "
                       "compute-bound, or the arm has no ceiling for its dtype"] += 1
                continue
            if cutlass:
                block_m, group_m, provenance = CUTLASS_BLOCK_M, 0, "cutlass_c1_observed"
            else:
                try:
                    tile = resolve_tile_for_row(row)
                except TileNotDerivable:
                    census["tile not derivable from vLLM's lookup"] += 1
                    continue
                block_m, group_m = tile.block_m_derived, tile.group_m_derived
                provenance = tile.provenance
            try:
                tiles = m_tiles_for_row(row, block_m)
            except (TileEfficiencyUndetermined, SC.TileConfigUnrecorded):
                census[f"M-tiles undetermined at BLOCK_M={block_m}: an expert spans "
                       "several tiles and the per-expert histogram is not stored"] += 1
                continue
            model = str(row.get("model", ""))
            spec = BenchSpec(MODEL_CONFIGS[model],
                             num_tokens=int(float(row["num_tokens"])),
                             dtype=str(row.get("dtype", "")))
            per_expert = expert_weight_bytes(spec, str(row.get("covers", "")))
            compulsory = SC.row_float(row, "compulsory_bytes")
            if compulsory <= 0.0 or per_expert <= 0.0:
                census["no compulsory byte model for this span"] += 1
                continue
            pct_fixed, pct_cell, roof_note = roof_fractions(row)
            out.append(Observation(
                traffic_ratio=ratio,
                compulsory_bytes=compulsory,
                per_expert_bytes=per_expert,
                active_experts=SC.row_float(row, "load_active_experts"),
                m_tiles=tiles,
                block_m=block_m, group_m=group_m, tile_provenance=provenance,
                model=model, dtype=str(row.get("dtype", "")),
                gpu=str(row.get("gpu_name", "")), impl=impl,
                tokens=int(float(row["num_tokens"])),
                routing=str(row.get("routing_kind", "")),
                l2_flush=SC.row_bool(row, "l2_flush"),
                cuda_graph=SC.row_bool(row, "cuda_graph"),
                tile_columns=tuple((c, str(row.get(c, ""))) for c in TILE_COLUMNS),
                arm=path.parent.name,
                dirty_raw=str(row.get("git_dirty", "")),
                instrument=instrument,
                clock_level_side=level_side_of(row),
                pct_fixed_roof=pct_fixed, pct_cell_clock_roof=pct_cell,
                roof_note=roof_note))
            census["ADMITTED"] += 1
    return out


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

def cell_key(obs: Observation) -> tuple:
    """Everything that sets the LEVEL of `implied_traffic_ratio` bar the tile.

    Token count is in here and it is the important one. Fixed dispatch cost
    amortises over more work as the batch grows, so the ratio falls with T while
    the tile count rises with T. Without T in the intercept, `alpha` is paid to
    explain that trend and comes out near zero, which is most of the distance
    between 0.10 and the answer this script prints.
    """
    return (obs.model, obs.dtype, obs.gpu, obs.impl, obs.l2_flush,
            obs.cuda_graph, obs.tokens, obs.instrument)


def _design(observations, keyfn, group_ids=None):
    x = np.array([o.x for o in observations], dtype=float)
    y = np.log(np.array([o.traffic_ratio for o in observations], dtype=float))
    if group_ids is not None:
        groups = np.asarray(group_ids, dtype=np.int64)
        return x, y, groups, int(groups.max()) + 1 if len(groups) else 0
    index: dict = {}
    groups = np.empty(len(observations), dtype=np.int64)
    for i, o in enumerate(observations):
        groups[i] = index.setdefault(keyfn(o), len(index))
    return x, y, groups, len(index)


def _within_group_ssr(x, y, groups, n_groups, alpha: float) -> float:
    """Within-group squared deviation of `log ratio - log(1 + alpha x)`.

    Log space, not absolute. The ratio spans about 1.0 to 8 across the pool, so
    a least-squares fit on raw values would weight one eager fp8 cell more than
    a hundred graphed bf16 cells. In logs the group intercept is a pure scale
    factor, which is exactly what it represents.
    """
    u = 1.0 + alpha * x
    if np.any(u <= 0.0):
        return math.inf
    residual = y - np.log(u)
    counts = np.bincount(groups, minlength=n_groups)
    sums = np.bincount(groups, weights=residual, minlength=n_groups)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    centred = residual - means[groups]
    return float(centred @ centred)


def fit_alpha(observations, keyfn=cell_key, *, group_ids=None,
              lo: float = -0.9, hi: float = 5.0, tol: float = 1e-5) -> float:
    """`alpha` minimising the within-group residual, by scan then golden section.

    The scan comes first because the objective is not guaranteed unimodal: `x`
    is a step function of the batch, so the residual has kinks wherever a cell
    changes tile count, and a bare golden section from a bad bracket would
    report a local minimum with nothing to indicate it.

    The lower bound is NEGATIVE on purpose. A fit that can only return a
    non-negative number cannot tell "the data say zero" from "the data say less
    than zero and were clipped at the boundary", and the pooled estimator this
    replaces fails in exactly that way on three of the four timing modes.

    `group_ids` names each row's intercept directly, for the bootstrap, where the
    same Observation OBJECT can be drawn into two different clusters and a
    key function computed from its fields could not tell the two copies apart.
    """
    if len(observations) < 2:
        raise ValueError("a fit needs at least two observations")
    x, y, groups, n_groups = _design(observations, keyfn, group_ids)

    def objective(alpha: float) -> float:
        return _within_group_ssr(x, y, groups, n_groups, alpha)

    grid = np.linspace(lo, hi, 120)
    values = [objective(a) for a in grid]
    best = int(np.argmin(values))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = right - phi * (right - left), left + phi * (right - left)
    fc, fd = objective(c), objective(d)
    while right - left > tol:
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - phi * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + phi * (right - left)
            fd = objective(d)
    return (left + right) / 2.0


#: The units a band may resample, and what each one buys.
#:
#: `cell` is the original: one (model, dtype, card, impl, timing mode, token
#: count) cell, which is the unit the FIT has an independent intercept for.
#: `arm`  is a published result directory: one pod, one session, one thermal
#:        state, one calibration. Rows in different cells of one arm share all
#:        of that, and the cell-level band prices none of it.
#: `card` is the physical part. Two levels, and that IS the finding: an
#:        interval built over two clusters is a statement about how little
#:        independent information the corpus holds at that level, not a
#:        narrower answer.
#:
#: All three are reported. Audit S34: the published band resampled cells alone
#: while the model split spanned 0.29 and the routing split 0.19, six to ten
#: times its width, and the text beside it read "stable".
CLUSTER_LEVELS = {
    "cell": cell_key,
    "arm": lambda o: (o.arm,),
    "card": lambda o: (o.gpu,),
}


def bootstrap_band(observations, draws: int, seed: int,
                   quantiles: tuple[float, float] = (0.05, 0.95),
                   level: str = "cell") -> tuple[float, float] | None:
    """A CLUSTER bootstrap at `level`, never over rows.

    Rows inside one cell are replicates of one measurement taken at several
    seeds and trials, so resampling them independently would treat six views of
    one thermal state as six measurements and return a band several times too
    narrow.

    THE FIXED EFFECTS STAY AT THE CELL WHATEVER THE CLUSTER IS. Each DRAWN COPY
    of a cluster gets its own intercept per cell it contains, so drawing the
    same arm twice gives two independent observations of every cell in it rather
    than one with double weight. At `level="cell"` a cluster holds exactly one
    cell and this reduces, identically, to the intercept-per-copy the published
    band was computed with -- which is why `--pinned-set` still prints
    0.529-0.588.

    `KeyError` for an unknown level, deliberately: a mistyped level silently
    falling back to `cell` would print a cell-level band under an arm-level
    heading, which is the exact mislabelling this parameter was added to fix.

    None when fewer than two clusters survive; a band over one cluster is not a
    band, and at `level="card"` on a single-card pool that is the honest answer.
    """
    keyfn = CLUSTER_LEVELS[level]
    groups: dict = collections.defaultdict(list)
    for o in observations:
        groups[keyfn(o)].append(o)
    keys = list(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        rows: list[Observation] = []
        ids: list[int] = []
        intercepts: dict = {}
        for copy in range(len(keys)):
            members = groups[rng.choice(keys)]
            rows.extend(members)
            for o in members:
                ids.append(intercepts.setdefault((copy, cell_key(o)),
                                                 len(intercepts)))
        samples.append(fit_alpha(rows, group_ids=ids))
    samples.sort()
    return _percentile(samples, quantiles[0]), _percentile(samples, quantiles[1])


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def pooled_cv_alpha(observations, lo: float = 0.0, hi: float = 2.0,
                    steps: int = 4000) -> tuple[float, float, float]:
    """`(alpha, CV, mean ratio)` under the ORIGINAL estimator, for comparison.

    Minimises the coefficient of variation of the corrected ratio over the whole
    pool with no group structure at all, which is what produced 0.10. Kept so
    both estimators can be run on the SAME rows: run only on different pools,
    the difference between them is unattributable, which is the position this
    study was in before today.

    A plain scan rather than an optimiser, because the point of printing it is
    the SHAPE of the objective rather than its argmin: on the original 151 rows
    it falls by well under one percent between `alpha = 0` and its minimum.
    """
    best = (math.inf, lo)
    for i in range(steps + 1):
        alpha = lo + (hi - lo) * i / steps
        score = _cv(observations, alpha)
        if score < best[0]:
            best = (score, alpha)
    return best[1], best[0], _mean_ratio(observations, best[1])


def _corrected(observations, alpha: float) -> list[float]:
    return [o.traffic_ratio / (1.0 + alpha * o.x) for o in observations]


def _cv(observations, alpha: float) -> float:
    values = _corrected(observations, alpha)
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def _mean_ratio(observations, alpha: float) -> float:
    return statistics.fmean(_corrected(observations, alpha))


def ai_cap(block_m: int, alpha: float, dtype_bytes: int = 2) -> float:
    """`2 BM / (alpha b)`: an UPPER BOUND on the AI this tile height can reach.

    The consequence the whole parameter is being measured for. If this sits
    below the hardware ridge, that tile cannot reach compute bound at any batch
    size at all, and lowering it only strengthens that. Infinite at
    `alpha <= 0`, which is the correct reading of THIS formula: with no
    weight re-read cost it names no other traffic.

    IT IS NOT THE CAP when `alpha` is a ladder fit, and every alpha this file
    fits is one. A B/(A+B) fit returns `(alpha_b + phi) / (1 + phi + delta)`,
    the per-tile slope over the first tread's level, and the level holds one
    tile's activation and output traffic plus the fixed cost. Read through
    this formula that level cancels the wrong way and the figure is HIGH by
    exactly `(1 + phi + delta)` (`moe/bench/ai_model.py`, `cap_from_fitted`,
    retraction (a)): 1.02 to 3.0 at BLOCK_M=128 with BLOCK_N=64, the width
    being alpha_a, which has no measurement in this repository. So a number
    from here is quoted as a bound, and `print_ai_cap_table` prints the
    corrected cap beside it as a bracket. The rows-per-expert unit is the
    b=2 identity, not an error.
    """
    return 2.0 * block_m / (alpha * dtype_bytes) if alpha > 0 else math.inf


#: The tile width the correction bracket is taken at: vLLM's pin for every
#: swept arm and the BN the study's activation-confound bound uses. CUTLASS's
#: grouped_mm tile is not observed, and the bracket says so where it quotes
#: BLOCK_M=64.
CAP_TABLE_BLOCK_N = 64

#: The models the pooled refit is fitted over, which is what the correction
#: bracket in `print_ai_cap_table` spans when the caller has no observation
#: list to take them from. Stated in the table header, never silent.
CAP_TABLE_MODELS = ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                    "deepseek-v3")


def lin_overstatement_bracket(block_m: int, models=CAP_TABLE_MODELS,
                              block_n: int = CAP_TABLE_BLOCK_N,
                              dtype_bytes: int = 2) -> tuple[float, float]:
    """`(low, high)` of `(1 + phi + delta)` at this tile, alpha_a in [0, 1], delta 0.

    The factor `2 BM / (alpha b)` overstates the exact cap by, as a BRACKET,
    because `phi` is one M-tile's activation-and-output traffic in weight-read
    units and its activation term is `alpha_a * BM / BN`, with alpha_a
    unmeasured. The low end is alpha_a = 0 on the shape with the smallest
    once-read cost, the high end alpha_a = 1 on the shape with the largest
    re-read cost, over the up and down GEMMs of every model in `models`.
    `delta`, the fixed cost, is unmeasured too and can only widen the factor,
    so 0 is the honest floor: the corrected caps this produces are themselves
    upper bounds, and the table says so.
    """
    lo, hi = math.inf, 0.0
    for name in models:
        cfg = MODEL_CONFIGS[name]
        for N, K in ((2 * cfg.intermediate_size, cfg.hidden_size),
                     (cfg.hidden_size, cfg.intermediate_size)):
            for alpha_a in (0.0, 1.0):
                phi = ai_model.phi(N, K, block_m=block_m, block_n=block_n,
                                   alpha_a=alpha_a, b=dtype_bytes)
                factor = ai_model.lin_overstatement(phi=phi, delta=0.0)
                lo, hi = min(lo, factor), max(hi, factor)
    return lo, hi


def corrected_cap_bracket(block_m: int, alpha: float, models=CAP_TABLE_MODELS,
                          block_n: int = CAP_TABLE_BLOCK_N,
                          dtype_bytes: int = 2) -> tuple[float, float]:
    """`(low, high)` cap in FLOP/byte: `ai_cap` divided by the factor bracket.

    Low is the alpha_a = 1 end, high the alpha_a = 0 end. Infinite at
    `alpha <= 0` like `ai_cap`, because dividing infinity by a finite factor
    is still the statement "this formula names no bound".
    """
    cap = ai_cap(block_m, alpha, dtype_bytes)
    lo, hi = lin_overstatement_bracket(block_m, models, block_n, dtype_bytes)
    return cap / hi, cap / lo


def cap_bracket_verdict(low: float, high: float, band: list[float]) -> str:
    """`cap_verdict` over a BRACKET: one word when both ends agree, and when
    they do not, the honest answer is that alpha_a decides and this study has
    not measured it."""
    a, b = cap_verdict(low, band), cap_verdict(high, band)
    if a == b:
        return a
    return f"undecided ({a} at alpha_a=1, {b} at alpha_a=0)"


def max_alpha_that_still_crosses(block_m: int, ridge: float,
                                 dtype_bytes: int = 2) -> float:
    """The largest `alpha` at which `block_m` can still reach `ridge`."""
    return 2.0 * block_m / (ridge * dtype_bytes)


def card_ridge_bands(dtype: str = "bf16") -> list[tuple[str, float, list[float]]]:
    """`(card, ridge, band)` for every card with a committed calibration.

    WHAT FAILURE THIS PREVENTS. The AI-cap table below decided "NEVER crosses"
    against `RIDGE_BAND`, which is the gap between two H200 calibrations of the
    same card and is not any device's ceiling. A cap verdict is a claim about a
    specific card, so it is now scored card by card against each one's own
    ridge, from each one's own `measured_*.yaml`, with the card named in the
    column header. Two cards disagreeing about whether a tile crosses is a
    result; hiding that behind one band belonging to neither was not.

    WHAT THESE BANDS DO NOT CARRY. Each is the spread of ONE calibration's
    surviving DRAM patterns, so it is a bandwidth band. The 9.9% disagreement
    between the H200's own compute ceilings is a separate open number and no
    band here contains it; `docs/INSTRUMENTATION.md` is where that lives.

    RESOLVED THROUGH THE RESCORER RATHER THAN RE-DERIVED.
    `scripts/rescore_published_reports.py` already resolves a card's ridge and
    band for the 26 published reports, and it does so through the sweep's
    `ridge_band_from_detail`, which drops the bandwidth patterns a calibration
    disowned. Recomputing that here would be a second implementation free to
    drift from the one the reports were rescored with.

    Returns an EMPTY list when nothing resolves, and the caller refuses on it
    rather than reaching for a constant.
    """
    import importlib.util

    path = Path(__file__).resolve().with_name("rescore_published_reports.py")
    if not path.exists():                     # pragma: no cover - repo invariant
        return []
    spec = importlib.util.spec_from_file_location("_alpha_refit_rescore", path)
    rescore = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, rescore)
    spec.loader.exec_module(rescore)
    sweep = rescore.load_sweep()

    out = []
    for card, profile in sorted(rescore.measured_profiles().items()):
        try:
            ridge, band, *_ = rescore.calibration(profile, dtype, sweep)
        except (ValueError, KeyError, FileNotFoundError):
            continue
        out.append((card, ridge, band))
    return out


def cap_verdict(cap: float, band: list[float]) -> str:
    """What a tile's AI cap says about one card's band. The whole vocabulary.

    Three answers and no fourth: below the low end the tile cannot reach that
    card's ridge at any batch size, above the high end it always can, and
    between them the calibration's own spread decides and this study cannot.
    """
    lo, hi = band[0], band[-1]
    return ("NEVER crosses" if cap < lo
            else "crosses" if cap > hi else "inside the band")


def count_excluded_memory_bound(paths, alpha: float,
                                include_throttled: bool = False
                                ) -> collections.Counter:
    """How many rows the memory-bound filter drops that it should not.

    THE FILTER THIS WHOLE FIT DEPENDS ON. `driver.py` writes
    `implied_traffic_ratio` only where `hardware.bound(dtype,
    arith_intensity_compulsory) == "memory"`, so a row with no column is a row
    the driver called compute-bound on the COMPULSORY intensity.

    That has no false positives, and to that extent the filter is sound:
    compulsory intensity is an upper bound on the true one, so
    `AI_compulsory < ridge` implies `AI_true < ridge`. The driver's own comment
    says exactly this.

    It has false NEGATIVES, and they are not randomly placed.
    `AI_corrected = AI_compulsory / Q` with `Q = 1 + alpha (tiles - 1) >= 1`, so
    a row with many tiles can be memory bound while its compulsory intensity
    says otherwise. Those rows carry no column, so they cannot enter the fit --
    and they are by construction the rows with the MOST extra tiles, which is
    precisely the evidence the fit is short of.
    """
    kept, _ = filter_superseded(paths)
    census: collections.Counter = collections.Counter()
    for path in kept:
        retired = superseded_impls(path) or set()
        for row in SC.read_csv(path):
            impl = str(row.get("impl", ""))
            if impl in retired or impl not in VLLM_IMPLS:
                continue
            if float(row.get("ms_p50") or 0.0) <= 0.0 or not SC.passed(row):
                continue
            if clock_gate(row, SC.instrument_of(row)) and not include_throttled:
                continue
            peak = SC.row_float(row, "achieved_peak_tflops")
            bandwidth = SC.row_float(row, "achieved_bw_gbps")
            if peak <= 0.0 or bandwidth <= 0.0:
                census["no ceiling on the row, so no ridge to classify against"] += 1
                continue
            if SC.row_float(row, "implied_traffic_ratio") > 0.0:
                census["memory-bound and carries the column"] += 1
                continue
            try:
                tile = resolve_tile_for_row(row)
                tiles = m_tiles_for_row(row, tile.block_m_derived)
            except (TileNotDerivable, TileEfficiencyUndetermined,
                    SC.TileConfigUnrecorded):
                census["no column, and the tile-corrected test cannot be run"] += 1
                continue
            spec = BenchSpec(MODEL_CONFIGS[str(row["model"])],
                             num_tokens=int(float(row["num_tokens"])),
                             dtype=str(row.get("dtype", "")))
            per_expert = expert_weight_bytes(spec, str(row.get("covers", "")))
            active = SC.row_float(row, "load_active_experts")
            compulsory = SC.row_float(row, "compulsory_bytes")
            extra = per_expert * max(tiles - active, 0.0)
            extra = extra if tiles - active > TILE_EPSILON else 0.0
            corrected = SC.row_float(row, "flops") / max(compulsory + alpha * extra, 1.0)
            # The ridge is the compute roof over the bandwidth roof, and on a
            # v6 row the driver scored the compute roof AT THE CLOCK THE CELL
            # RAN. Classifying against it is what the driver's own fixed-ridge
            # call could not do (its docstring says it under-classifies on a
            # boosted card); the bandwidth roof stays fixed, as there.
            if SC.has_cell_clock_roof(row):
                peak = SC.row_float(row, "roof_at_cell_clock_tflops")
            ridge = peak * 1e12 / (bandwidth * 1e9)
            if corrected < ridge:
                census["NO COLUMN BUT TILE-CORRECTED MEMORY-BOUND: "
                       "excluded and should not have been"] += 1
            else:
                census["compute-bound under both models, correctly excluded"] += 1
    return census


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

#: A split needs at least this many discriminating rows before its alpha is a
#: number rather than a restatement of the pool. Named once so `_split_line`'s
#: table and `split_range`'s ranges cannot drift apart on the same pool.
MIN_DISCRIMINATING = 10


def _split_line(label: str, subset: list[Observation], width: int = 24) -> str:
    """One split of the pool, with its discriminating count beside its answer.

    The count is not decoration. A split with no discriminating rows has NO
    information about `alpha` -- every row in it sits at `x = 0` -- so printing a
    number there would be printing the pooled answer under a new heading.
    """
    n_disc = sum(1 for o in subset if o.discriminating)
    head = f"  {label:<{width}} n={len(subset):>6}  discriminating={n_disc:>5}  "
    if n_disc < MIN_DISCRIMINATING or len(subset) < 2:
        return head + "alpha=n/a (nothing in this split can move it)"
    return head + f"alpha={fit_alpha(subset):.3f}"


#: This checkout's root, used only to print paths relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_relative(path) -> str:
    """`results/published/REFIT_SET.txt`, not `/private/tmp/.../results/...`.

    Every path this script prints goes through here. An absolute path is a fact
    about the machine that ran the tool and not about the study, and audit A6
    found one embedded in a committed artefact (`ANCHOR_RESCORE.txt` carries the
    pod's own repo root), which makes the artefact unreproducible for the
    trivial reason that nobody else's checkout is at that path. Falls back to the
    absolute form for a path outside the repository, which is honest: it really
    is somewhere else.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class PinnedSetBroken(SystemExit):
    """A manifest that does not describe the tree it is being read against.

    `SystemExit` so an unhandled one exits without a traceback, and with a
    message, because the only useful response is to fix the manifest or the
    tree. Refusing is the point: a pinned set that quietly drops a missing arm
    would answer with a different pool under the heading that promises this one,
    which is the failure the manifest exists to close.
    """


def read_refit_set(manifest: Path) -> tuple[list[str], list[Path]]:
    """`(arm names, csv paths)` from a pinned-set manifest.

    Blank lines and `#` comments out. Every named arm must exist under the
    manifest's own directory and must hold at least one `run_*.csv`; anything
    else raises `PinnedSetBroken` naming every arm at fault at once, so a
    reviewer fixes one file rather than rediscovering the next failure per run.
    """
    if not manifest.exists():
        raise PinnedSetBroken(
            f"no pinned-set manifest at {manifest}. Write one (see "
            f"{DEFAULT_REFIT_SET.name} in this repository for the format) or "
            "pass the CSVs yourself.")
    root = manifest.parent
    arms = [line.strip() for line in manifest.read_text().splitlines()]
    arms = [a for a in arms if a and not a.startswith("#")]
    if not arms:
        raise PinnedSetBroken(f"{manifest} lists no arms; an empty pinned set "
                              "is not a set")
    missing, empty, csvs = [], [], []
    for arm in arms:
        found = sorted((root / arm).glob(REFIT_SET_GLOB))
        if not (root / arm).is_dir():
            missing.append(arm)
        elif not found:
            empty.append(arm)
        csvs.extend(found)
    if missing or empty:
        parts = []
        if missing:
            parts.append(f"absent from {root}: {', '.join(missing)}")
        if empty:
            parts.append(f"present but holding no {REFIT_SET_GLOB}: "
                         f"{', '.join(empty)}")
        raise PinnedSetBroken(
            f"{manifest} does not describe this tree -- " + "; ".join(parts)
            + ". The pinned set is refused rather than shrunk: a fit over "
              "fewer arms than the manifest names is not the fit the manifest "
              "promises.")
    return arms, csvs


def _report_inputs(args, arms: list[str] | None) -> None:
    """WHICH ROWS THIS RUN READ, at the top, before any number.

    Audit B8: FINDINGS' 0.558 / 10,813 / 3,124 / 0.529-0.588 reproduce on ten
    arms and the documented glob reads fourteen, and nothing in either the
    document or the output said which set was in front of the reader. A number
    whose input set is not printed beside it is not reproducible, however
    deterministic the arithmetic is.
    """
    print("## the input set")
    print()
    if arms is not None:
        print(f"  PINNED by {repo_relative(args.pinned_set)}")
        print(f"  {len(arms)} arm(s), {len(args.csvs)} csv(s):")
        for arm in arms:
            print(f"    {arm}")
        print()
        print("  This is the set `docs/FINDINGS.md` quotes. Run without "
              "--pinned-set to")
        print("  read whatever the tree holds today and watch the two "
              "disagree.")
    else:
        seen: list[str] = []
        for path in args.csvs:
            name = Path(path).parent.name
            if name not in seen:
                seen.append(name)
        print(f"  the caller's own paths: {len(args.csvs)} csv(s) across "
              f"{len(seen)} arm(s):")
        for arm in seen:
            print(f"    {arm}")
        print()
        print(f"  NOT the pinned set. `--pinned-set` reads "
              f"{DEFAULT_REFIT_SET.name} and reproduces")
        print("  the numbers in docs/FINDINGS.md; this run does not claim to.")


def _report_pool(triton: list[Observation], census: collections.Counter) -> None:
    print("## the pool")
    print()
    for reason, count in census.most_common():
        print(f"  {count:>7}  {reason}")
    print()
    n_disc = sum(1 for o in triton if o.discriminating)
    print(f"  {len(triton)} rows admitted, of which {n_disc} have M-tiles > active")
    print(f"  experts and can move alpha at all. The other {len(triton) - n_disc} sit at")
    print("  x = 0 and contribute a group intercept and nothing else.")
    print()
    print(f"  dtypes present: {dict(collections.Counter(o.dtype for o in triton))}")
    if {o.dtype for o in triton} == {"bf16"}:
        print("  THE FIT IS bf16 ONLY, and not by choice. Every fp8 vLLM row lives in")
        print("  `-fp8-three-kernel`, whose calibration measured no fp8 ceiling, so all")
        print("  of them carry achieved_peak_tflops = 0 and with it")
        print("  implied_traffic_ratio = 0. A same-session fp8 calibration is one line")
        print("  on a pod, and it is what would let alpha be tested across dtypes.")
    print()
    print("  tile provenance: "
          f"{dict(collections.Counter(o.tile_provenance for o in triton))}")
    print()
    # THE EXPOSURE, PRINTED WHATEVER IT IS. Audit A6: `calibrate_hardware.py`
    # writes a TRACKED yaml, so a session sweeping right after a calibration
    # stamps every row `git_dirty=True`, and 44,872 of the 100,144 published
    # rows carry it. Until 2026-09-02 no analysis path in this repository read
    # the column, so the number below has never appeared beside an answer.
    rows_for_dirt = [{"git_dirty": o.dirty_raw} for o in triton]
    print(dirty_share_line(rows_for_dirt, "admitted rows"))
    dirty, _unrecorded, total = dirty_share(rows_for_dirt)
    if dirty:
        by_arm = collections.Counter(o.arm for o in triton if o.dirty)
        print("  those rows cannot be tied to a commit, so a stranger cannot "
              "rebuild the")
        print("  code that produced them. They are NOT dropped -- that would "
              "discard most of")
        print("  the study -- and the share is stated so the reader prices "
              "it. By arm: "
              + ", ".join(f"{a} {n}" for a, n in by_arm.most_common()))
    elif total:
        print("  every admitted row was measured from a clean tree, which is a "
              "finding and")
        print("  is printed for that reason rather than omitted as an absence.")


def _pct_summary(values: list[float]) -> str:
    """median and the 10th to 90th percentile span of a list of percentages."""
    ordered = sorted(values)
    return (f"median {statistics.median(ordered):6.2f}%   "
            f"p10-p90 {_percentile(ordered, 0.10):6.2f}% to "
            f"{_percentile(ordered, 0.90):6.2f}%   over {len(ordered)} rows")


def _report_roof(triton: list[Observation]) -> None:
    """THE READER FOR `pct_of_roof_at_cell_clock`, printed beside the fixed one.

    Commit 03df2d4 wrote the per-row roof (`roof_at_cell_clock_tflops`,
    `pct_of_roof_at_cell_clock`) so a boosted cell's efficiency is against the
    roof at its own clock rather than the calibration's, and until 2026-09-08
    nothing under scripts/ read the column: the LEVEL-HIGH clock note told the
    reader to "read roof_at_cell_clock_tflops" and no report printed it. This
    section prints the two fractions together, says on how many admitted rows
    the corrected one exists and why it is absent on the rest (the committed
    corpus is v3 to v5, so what shows there is "not available"), and counts
    the HIGH-side rows the gate admitted, with the note that their fixed-roof
    figure is the one not to quote.

    Nothing in the fit reads either column (see `clock_gate`); this is the
    fraction a reader would otherwise take off a row by hand, printed once with
    its provenance.
    """
    print("## fraction of the compute roof on the admitted rows")
    print()
    fixed = [o.pct_fixed_roof for o in triton if o.pct_fixed_roof > 0.0]
    print("  pct_of_achieved_tflops   (FIXED roof, the calibration's clock):")
    if fixed:
        print("      " + _pct_summary(fixed))
    unscored = len(triton) - len(fixed)
    if unscored:
        print(f"      not scored on {unscored} rows (achieved_peak_tflops = 0: "
              "no ceiling for the dtype)")
    cell = [o.pct_cell_clock_roof for o in triton
            if o.pct_cell_clock_roof is not None]
    print("  pct_of_roof_at_cell_clock (roof AT THE CLOCK THE CELL RAN):")
    if cell:
        print("      " + _pct_summary(cell))
    absent = collections.Counter(o.roof_note for o in triton
                                 if o.pct_cell_clock_roof is None)
    for note, count in absent.most_common():
        print(f"      {note}: {count} rows")
    if not cell:
        print("      no admitted row carries the corrected fraction; every "
              "fraction of roof above")
        print("      is the fixed-roof figure with the bias `schema.Row` "
              "states for it.")
    print()
    high = [o for o in triton if o.clock_level_side == LEVEL_HIGH]
    low = [o for o in triton if o.clock_level_side == LEVEL_LOW]
    print(f"  {len(high)} of {len(triton)} admitted rows are {HIGH_SIDE_NOTE}")
    if high:
        by_arm = collections.Counter(o.arm for o in high)
        print("      kept on purpose: the time is the kernel's, and nothing "
              "the fit reads depends")
        print("      on the fixed roof. By arm: "
              + ", ".join(f"{a} {n}" for a, n in by_arm.most_common()))
    if low:
        print(f"  {len(low)} admitted rows failed LEVEL LOW and are in the pool "
              "only because")
        print("      --include-throttled re-admitted them; their time is NOT "
              "the kernel's.")


def _report_instruments(triton: list[Observation], pool: bool) -> bool:
    """WHICH APPARATUS MEASURED THE ROWS, AND WHETHER THEY MAY BE FITTED TOGETHER.

    Printed for every run, pinned or not, because a number whose instrument is
    not stated beside it is not comparable with any other number in this study:
    the roof, the ladder scripts and (since schema v5) the driver all name the
    timer that produced them, and a report that quoted an alpha without one
    would be the only artefact left that does not.

    Returns True when the fit may proceed. A single instrument always may. Two
    or more may only under `--pool-instruments`, and the module docstring argues
    why: `alpha` is identified inside a group intercept, the instruments differ
    in level by construction, and no arm in this corpus ran one cell both ways,
    so nothing here can measure whether they also differ in slope.

    `UNSTATED_INSTRUMENT` is refused before either branch and `--pool-instruments`
    does NOT lift it: that flag is a decision to pool two apparatus a reader can
    name, and an unstated one cannot be weighed against anything. It reaches
    here only from an Observation built by hand, so the fix is at that call site.
    """
    mix = instrument_mix(triton)
    print("## the instrument that measured the fitted set")
    print()
    for name, count in mix.most_common():
        disc = sum(1 for o in triton
                   if o.instrument == name and o.discriminating)
        print(f"  {count:>7}  {name}")
        print(f"           of which {disc} discriminating")
    print()
    if UNSTATED_INSTRUMENT in mix:
        print(f"  REFUSED. {mix[UNSTATED_INSTRUMENT]} of these observations "
              "were built by a caller that")
        print("  did not say what timed them. That is not a third apparatus to "
              "pool, it is an")
        print("  absent fact: alpha is identified inside a group intercept "
              "keyed on the")
        print("  instrument, so an unstated one either invents a cluster or "
              "collapses two")
        print("  real ones, and nothing here can tell which. Pass "
              "`instrument=` at the site")
        print("  that built them -- `collect` reads it off the row and never "
              "reaches this.")
        return False
    if len(mix) == 1:
        only = next(iter(mix))
        if only == SC.LEGACY_INSTRUMENT:
            print("  ONE INSTRUMENT, AND IT IS THE RETIRED ONE. Every row here was "
                  "timed with a")
            print("  COUNT of warmup calls, an iteration count sized from one "
                  "isolated call on an")
            print("  idle GPU, and two idle-instant clock samples either side of "
                  "the cell. The")
            print("  fit below is therefore internally consistent and is NOT "
                  "comparable, cell for")
            print("  cell, with an alpha measured by "
                  "`moe.bench.timing.time_kernel`; a ladder")
            print("  script's answer and this one are two instruments' answers "
                  "until an arm")
            print("  re-measures these cells on the instrument.")
        else:
            print("  ONE INSTRUMENT, and it is the one `moe/bench/timing.py` "
                  "names. The fit is")
            print("  comparable with the roof and with every ladder script "
                  "measured under it.")
        return True

    print("  TWO OR MORE INSTRUMENTS IN ONE POOL.")
    if not pool:
        print()
        print("  REFUSED. alpha is identified INSIDE a group intercept, the "
              "instrument is in")
        print("  that key so no intercept spans both -- but an instrument that "
              "shifts a")
        print("  many-tile row differently from a one-tile row moves the SLOPE, "
              "and no arm in")
        print("  this corpus ran one cell both ways, so nothing here can measure "
              "whether it")
        print("  does. Fit each instrument separately, or pass "
              "--pool-instruments to fit them")
        print("  together with the counts above printed beside every number.")
        return False
    print()
    print("  POOLED ON PURPOSE (--pool-instruments). Every number below is a "
          "number over")
    print("  the mix printed above, and the between-instrument split in the fit "
          "section is")
    print("  the only evidence here about how much of it that mix is worth.")
    return True


def split_range(triton: list[Observation], keyfn
                ) -> tuple[float, float, int] | None:
    """`(lowest alpha, highest alpha, splits fitted)` over one lever's levels.

    None when fewer than two levels can be fitted at all, which is the honest
    answer for a lever the pool does not vary.

    THIS IS THE NUMBER THE BAND HAS TO BE READ AGAINST. The band says how
    precisely the POOLED alpha is located; this says how far alpha moves when
    one thing about the rows changes. Audit S34: the published band was 0.03
    wide and the model split spanned 0.29, and the sentence beside them called
    alpha stable.
    """
    alphas = []
    for level in sorted({keyfn(o) for o in triton}, key=str):
        subset = [o for o in triton if keyfn(o) == level]
        if sum(1 for o in subset if o.discriminating) < MIN_DISCRIMINATING:
            continue
        if len(subset) < 2:
            continue
        alphas.append(fit_alpha(subset))
    if len(alphas) < 2:
        return None
    return min(alphas), max(alphas), len(alphas)


def _report_fit(triton: list[Observation], alpha: float, args) -> None:
    print("## the fit")
    print()
    n_groups = len({cell_key(o) for o in triton})
    n_disc = sum(1 for o in triton if o.discriminating)
    print(f"  alpha = {alpha:.3f}")
    print(f"  n = {len(triton)} rows, {n_disc} discriminating, {n_groups} intercepts")
    # THE MIX, ON THE SAME LINES AS THE HEADLINE. A reader who quotes the number
    # above quotes these three lines or none of them: the instrument is what
    # makes this alpha comparable with the roof, or not.
    print("  measured on: "
          + "; ".join(f"{count} rows on {name}"
                      for name, count in instrument_mix(triton).most_common()))
    print()

    # THREE BANDS, NOT ONE, AND THE WIDEST IS THE ANSWER. Cells are nested in
    # arms and arms in cards, so each level prices a kind of shared state the
    # one below it treats as independent. The published band was the first row
    # of this table quoted on its own.
    print("  90% cluster-bootstrap bands, by what the draw resamples:")
    bands: dict[str, tuple[float, float] | None] = {}
    for level in CLUSTER_LEVELS:
        n_clusters = len({CLUSTER_LEVELS[level](o) for o in triton})
        band = bootstrap_band(triton, args.bootstrap, args.seed, level=level)
        bands[level] = band
        if band is None:
            print(f"    {level:<5} REFUSED: {n_clusters} cluster(s); a band "
                  "over one cluster is not a band")
            continue
        print(f"    {level:<5} {band[0]:.3f} .. {band[1]:.3f}   "
              f"({args.bootstrap} draws over {n_clusters} cluster(s))")
    if bands.get("card") is not None:
        print("    the card row is TWO clusters wide. Read it as a statement "
              "about how")
        print("    little independent information the corpus holds at that "
              "level, never as")
        print("    a tighter answer than the rows above it.")

    # THE MDE, DERIVED FROM THE BAND THAT WAS JUST PRINTED. Audit B14: no arm in
    # this study states one, so no split-to-split difference below has ever been
    # compared with the smallest difference this design could have found.
    print()
    widest = max((b for b in bands.values() if b is not None),
                 key=lambda b: b[1] - b[0], default=None)
    if widest is None:
        print("  MDE: NOT STATED -- no level of this pool supports a band, so "
              "no split")
        print("  comparison below is comparable with anything.")
    else:
        print(mde_line(widest[0], widest[1], "alpha",
                       "the widest cluster-bootstrap band above"))

    # THE RANGES THE BAND HAS TO BE READ AGAINST, on the same screen as the
    # band. Between-basis is called out by name because FINDINGS quotes it as
    # "stable across timing modes (0.48-0.59)" and the paired contrast below
    # shows most of that spread is composition, not instrument.
    print()
    print("  and what alpha does when one thing about the rows changes:")
    for label, keyfn in (("between model", lambda o: o.model),
                         ("between card", lambda o: o.gpu),
                         ("between basis", lambda o: o.mode),
                         ("between instrument", lambda o: o.instrument),
                         ("between routing", lambda o: o.routing),
                         ("between arm", lambda o: o.arm)):
        rng = split_range(triton, keyfn)
        if rng is None:
            print(f"    {label:<16} only one level fits; not a range")
            continue
        lo, hi, n = rng
        mde = (None if widest is None
               else two_sample_mde(sd_from_band(widest[0], widest[1])))
        verdict = ("no MDE" if mde is None else
                   "READABLE" if hi - lo > mde else
                   "inside the MDE: UNMEASURED")
        print(f"    {label:<16} {lo:.3f} .. {hi:.3f}  (span {hi - lo:.3f} over "
              f"{n} levels)  {verdict}")
    print()
    print(f"  vs this repo's published {REPO_PUBLISHED_ALPHA:.2f}: "
          f"{alpha / REPO_PUBLISHED_ALPHA:.1f}x")
    print(f"  vs TEMPO's {TEMPO_ALPHA:.2f}:                 {alpha / TEMPO_ALPHA:.1f}x")


#: The two axes of the timing basis, each as `(name, accessor, low, high)`.
#: `low`/`high` are the words the report prints for the False and True levels,
#: so the table reads as a comparison rather than as two booleans.
BASIS_AXES = (
    ("cuda_graph", lambda o: o.cuda_graph, "eager", "graph"),
    ("l2_flush", lambda o: o.l2_flush, "L2-warm", "L2-cold"),
)


def paired_basis_contrast(triton: list[Observation], level_of
                          ) -> tuple[list[Observation], list[Observation]]:
    """The two halves of one basis axis, restricted to CONTEXTS present in both.

    A context is `Observation.context`: the physical cell, with the timing mode
    taken out. Keeping only contexts observed at both levels makes the two
    halves comparable by construction, which the marginal split is not.

    WHY THE MARGINAL SPLIT IS NOT A CONTRAST. `docs/FINDINGS.md` reads the
    per-basis fits (L2-cold/eager 0.538, L2-cold/graph 0.480, L2-warm/eager
    0.597, L2-warm/graph 0.505) as "stable across timing modes (0.48-0.59)", and
    the audit's refuter found the split composition-confounded: 98% of the
    graph-mode discriminating rows are deepseek-v2-lite, which is the
    lowest-alpha model in the pool. The split was mostly measuring which models
    each mode was pointed at. Held to shared contexts the basis moves alpha by
    about one band width, not by 1.9 of them.

    Returns `(low rows, high rows)`, both empty when no context holds both.
    """
    by_context: dict = collections.defaultdict(set)
    for o in triton:
        by_context[o.context].add(bool(level_of(o)))
    shared = {c for c, levels in by_context.items() if levels == {False, True}}
    low = [o for o in triton if o.context in shared and not level_of(o)]
    high = [o for o in triton if o.context in shared and level_of(o)]
    return low, high


def _composition(rows: list[Observation]) -> str:
    """The model mix of the DISCRIMINATING rows, which is the confounded part.

    Only the discriminating rows, because those are the only ones that move
    alpha: a split can be balanced in rows and still be one model's answer.
    """
    disc = [o for o in rows if o.discriminating]
    if not disc:
        return "no discriminating rows"
    counts = collections.Counter(o.model for o in disc).most_common()
    return ", ".join(f"{m} {n / len(disc):.0%}" for m, n in counts)


def _report_basis_contrast(triton: list[Observation], args) -> None:
    """The instrument's effect on alpha, once composition is held fixed."""
    print("## does the timing basis move alpha, or move the composition?")
    print()
    print("Marginal first, then paired. The marginal rows are the ones FINDINGS "
          "quotes;")
    print("the paired rows are the same question asked of cells that were "
          "measured BOTH")
    print("ways, so the two halves cannot differ in which models they contain.")
    for name, level_of, low_word, high_word in BASIS_AXES:
        print()
        print(f"  axis {name}: {low_word} against {high_word}")
        marg_low = [o for o in triton if not level_of(o)]
        marg_high = [o for o in triton if level_of(o)]
        for word, rows in ((low_word, marg_low), (high_word, marg_high)):
            n_disc = sum(1 for o in rows if o.discriminating)
            if n_disc < MIN_DISCRIMINATING or len(rows) < 2:
                print(f"    marginal {word:<8} n={len(rows):>6} "
                      f"discriminating={n_disc:>5}  alpha=n/a")
                continue
            print(f"    marginal {word:<8} n={len(rows):>6} "
                  f"discriminating={n_disc:>5}  alpha={fit_alpha(rows):.3f}")
            print(f"      composition: {_composition(rows)}")
        low, high = paired_basis_contrast(triton, level_of)
        n_contexts = len({o.context for o in low})
        if not low or not high:
            print("    PAIRED: no cell in this pool was measured both ways, so "
                  "this axis")
            print("    cannot be contrasted here at all. The marginal numbers "
                  "above are a")
            print("    comparison of two different sets of cells and must not "
                  "be read as one.")
            continue
        alphas = {}
        for word, rows in ((low_word, low), (high_word, high)):
            n_disc = sum(1 for o in rows if o.discriminating)
            if n_disc < MIN_DISCRIMINATING or len(rows) < 2:
                print(f"    paired   {word:<8} n={len(rows):>6} "
                      f"discriminating={n_disc:>5}  alpha=n/a")
                continue
            alphas[word] = fit_alpha(rows)
            print(f"    paired   {word:<8} n={len(rows):>6} "
                  f"discriminating={n_disc:>5}  alpha={alphas[word]:.3f}")
            print(f"      composition: {_composition(rows)}")
        print(f"    over {n_contexts} context(s) measured both ways "
              "(model, dtype, card, impl, tokens)")
        if len(alphas) == 2:
            delta = alphas[high_word] - alphas[low_word]
            band = bootstrap_band(low + high, args.bootstrap, args.seed)
            print(f"    PAIRED SHIFT {high_word} - {low_word} = {delta:+.3f}")
            if band is None:
                print("    no band over the paired pool, so the shift is "
                      "unscored")
            else:
                mde = two_sample_mde(sd_from_band(band[0], band[1]))
                verdict = ("READABLE" if abs(delta) > mde
                           else "inside the MDE: UNMEASURED")
                print(f"    against MDE {mde:.3f} from the paired pool's own "
                      f"cell band {band[0]:.3f}..{band[1]:.3f}: {verdict}")


def _report_splits(triton: list[Observation]) -> None:
    print("## is alpha a scalar?")
    print()
    print("GROUP_SIZE_M is the swizzle width, i.e. exactly how many M-tiles reuse one")
    print("weight block out of L2, so it is the parameter alpha should vary with if")
    print("alpha varies with anything at all.")
    print()
    for group_m in sorted({o.group_m for o in triton}):
        print(_split_line(f"GROUP_SIZE_M = {group_m}",
                          [o for o in triton if o.group_m == group_m]))
    print()
    for block_m in sorted({o.block_m for o in triton}):
        print(_split_line(f"BLOCK_M = {block_m}",
                          [o for o in triton if o.block_m == block_m]))
    print()
    for model in sorted({o.model for o in triton}):
        print(_split_line(model, [o for o in triton if o.model == model]))
    print()
    for gpu in sorted({o.gpu for o in triton}):
        print(_split_line(gpu, [o for o in triton if o.gpu == gpu]))
    print()
    for mode in sorted({o.mode for o in triton}):
        print(_split_line(mode, [o for o in triton if o.mode == mode]))
    print()
    for routing in sorted({o.routing for o in triton}):
        print(_split_line(f"routing {routing}",
                          [o for o in triton if o.routing == routing]))
    print()
    # The check on the pooling argument in `clock_gate`: if a HIGH-side row's
    # traffic ratio were a different quantity from a level row's, this split
    # would show it. Labelled by what the side means, not by the word.
    side_label = {"": "LEVEL held or pre-v6", LEVEL_HIGH: "LEVEL failed HIGH",
                  LEVEL_LOW: "LEVEL failed LOW (re-admitted)"}
    for side in sorted({o.clock_level_side for o in triton}):
        print(_split_line(side_label[side],
                          [o for o in triton if o.clock_level_side == side]))


def _report_original(cutlass: list[Observation], alpha_new: float) -> None:
    """The published 0.10, its rows, and the same rows under the new estimator.

    Run on the L2-cold eager torch `grouped_mm` rows because that is the basis
    the 2026-08-22 write-up named: 151 unthrottled memory-bound rows at the
    CUTLASS tile of 64. Reproducing the count and the CV column is what turns
    the comparison into an attribution instead of an assertion.
    """
    basis = [o for o in cutlass if o.l2_flush and not o.cuda_graph]
    print("## the original estimator, on the original rows")
    print()
    print(f"  `{ORIGINAL_ALPHA_ARM}` only: torch grouped_mm, L2-cold eager,")
    print("  unthrottled, memory-bound, at the CUTLASS BLOCK_M="
          f"{CUTLASS_BLOCK_M} OBSERVED under C1.")
    print("  Nothing in this section is derived from anything.")
    print()
    n_disc = sum(1 for o in basis if o.discriminating)
    print(f"  n={len(basis)}  discriminating={n_disc}"
          "   (the write-up says 151 rows, 27 of them discriminating)")
    print()
    print("  The CV column below reproduces the write-up's to a tenth of a point.")
    print("  The mean ratio sits about 1% lower, and for a reason worth stating: the")
    print("  write-up divided by WEIGHT bytes at a fixed 4390.29 GB/s read ceiling,")
    print("  while implied_traffic_ratio divides by the row's full compulsory bytes,")
    print("  activations included, at the row's own triad ceiling. Same rows, slightly")
    print("  different denominator; nothing about the identification changes.")
    if len(basis) < 2:
        print("  too few rows to fit")
        return
    fitted, cv, mean = pooled_cv_alpha(basis)
    print()
    print("  | alpha | mean ratio | CV |")
    print("  |---|---:|---:|")
    for candidate in (0.0, REPO_PUBLISHED_ALPHA, TEMPO_ALPHA, 1.0):
        print(f"  | {candidate:.2f} | {_mean_ratio(basis, candidate):.2f}x | "
              f"{_cv(basis, candidate):.1%} |")
    print(f"  | {fitted:.3f} (its own minimum) | {mean:.2f}x | {cv:.1%} |")
    print()
    print("  THE OBJECTIVE IS NEARLY FLAT: the CV falls by "
          f"{1 - cv / _cv(basis, 0.0):.1%} between")
    print(f"  alpha = 0 and its own minimum, on {n_disc} rows out of {len(basis)} that can")
    print("  move it at all. A minimum that shallow is a statement about the")
    print("  estimator, not about the hardware.")
    print()
    print(f"  SAME ROWS, group-intercept estimator: alpha = {fit_alpha(basis):.3f}")
    print(f"  SAME ESTIMATOR, whole derived pool:   alpha = {alpha_new:.3f}")
    print()
    print("  So the disagreement with TEMPO is NOT an artefact of the assumed tile.")
    print("  The tile in these rows was never assumed: it was read out of the CUTLASS")
    print("  kernel name. It is an artefact of the ESTIMATOR.")


def _report_adversarial(triton: list[Observation], alpha: float, args) -> None:
    """Everything that would make the number above wrong, checked where it can be."""
    print("## against my own fit")
    print()
    print("### 1. the placebo")
    print("  The RESPONSE is permuted inside each group, which breaks the pairing")
    print("  between a row's traffic ratio and its tile count while leaving both")
    print("  marginals and the whole group structure exactly as they were. The")
    print("  regressor is not touched, because shuffling THAT would also break the")
    print("  pairing between a row's tile count and its own active-expert count and")
    print("  would test something else.")
    print()
    rng = random.Random(args.seed)
    groups: dict = collections.defaultdict(list)
    for o in triton:
        groups[cell_key(o)].append(o)
    shuffled: list[Observation] = []
    for members in groups.values():
        responses = [o.traffic_ratio for o in members]
        rng.shuffle(responses)
        shuffled.extend(dataclasses.replace(o, traffic_ratio=r)
                        for o, r in zip(members, responses, strict=True))
    print(f"  traffic ratio permuted within each group: alpha = {fit_alpha(shuffled):.3f}")
    print(f"  as measured:                              alpha = {alpha:.3f}")
    print("  A fit that survived the permutation would be fitting the group")
    print("  structure rather than the tile.")
    print()

    print("### 2. how much of the answer is the derivation")
    print("  The tile enters only through the M-tile count, so forcing ONE height on")
    print("  every row bounds how much of the answer the per-row derivation carries.")
    print()
    for block_m in (16, 32, 64, 128, 256):
        forced = [f for f in (o.at_block_m(block_m) for o in triton) if f is not None]
        n_disc = sum(1 for o in forced if o.discriminating)
        line = (f"  forced BLOCK_M={block_m:>3}: n={len(forced):>6} "
                f"discriminating={n_disc:>5}  ")
        print(line + (f"alpha={fit_alpha(forced):.3f}" if n_disc >= 10 and len(forced) > 1
                      else "alpha=n/a"))
    print()
    print("  BLOCK_M 16 and 32 have NO discriminating rows, and that is structural")
    print("  rather than a sampling accident. At a block size the schema does not")
    print("  store, the M-tile count is RECONSTRUCTED, and the reconstruction is only")
    print("  valid while every expert fits in one tile -- so the rows that survive at")
    print("  16 are exactly the rows where 16 costs nothing. The derived pool inherits")
    print("  that: most of it resolves to BLOCK_M=16, and none of that part")
    print("  constrains alpha.")
    print()

    print("### 3. the memory-bound filter is sound one way and incomplete the other")
    excluded = count_excluded_memory_bound(args.csvs, alpha, args.include_throttled)
    for reason, count in excluded.most_common():
        print(f"  {count:>7}  {reason}")
    print()
    print("  No false positives: compulsory intensity is an UPPER bound on the true")
    print("  one, so a row the driver called memory-bound is memory-bound under the")
    print("  tile-corrected model too. The false NEGATIVES are the problem, and they")
    print("  are not randomly placed: AI_corrected = AI_compulsory / Q, and Q grows")
    print("  with the tile count, so the rows wrongly excluded are the many-tile rows")
    print("  this fit is short of. That biases the pool toward low leverage; it does")
    print("  not obviously bias alpha in a known direction.")
    print()

    print("### 4. the fitted alpha contradicts a crossing this study measured")
    print("  alpha here is fitted to TIME, so it absorbs an extra tile's padded")
    print("  arithmetic and scheduling as well as its re-read, and read as a traffic")
    print("  coefficient it is an UPPER bound. Read as 2 BM / (alpha b) it gives an")
    print("  UPPER BOUND on the AI cap, not the cap: a ladder fit's level carries one")
    print("  tile's activation, output and fixed cost, so the exact cap is that figure")
    print("  over (1 + phi + delta) (moe/bench/ai_model.py, retraction (a)), and with")
    print("  alpha_a unmeasured the factor is a BRACKET. Both are printed, in rows per")
    print("  expert (the b=2 identity):")
    print()
    print_ai_cap_table(alpha, sorted({o.model for o in triton}) or None)


def print_ai_cap_table(alpha: float, models=None) -> None:
    """The AI-cap table, and the C2 comparison, each against a NAMED card's band.

    A SEPARATE FUNCTION SO ITS REFUSALS CAN BE PLANTED. Both failure branches
    here -- no calibration at all, and no H200 calibration for a paragraph about
    H200 rows -- have to be reachable in a test, and they are not while the only
    way in is a full adversarial run over the corpus.

    TWO NUMBERS PER TILE, AND THE VERDICT IS SCORED ON THE BRACKET. `2 BM /
    (alpha b)` is printed as what it is, an upper bound on the cap for a
    ladder-fitted alpha; beside it is the `(1 + phi + delta)` factor as a
    bracket over alpha_a in [0, 1] at delta = 0 (`lin_overstatement_bracket`,
    over the up and down GEMMs of `models`, BLOCK_N = 64), and the corrected
    cap the bracket gives. Each card's verdict is `cap_bracket_verdict`: one
    word when both ends of the bracket land on the same side of the card's
    band, "undecided" when alpha_a would decide. Until 2026-09-03 this table
    scored the uncorrected figure as the cap, which at BLOCK_M=128 is high by
    1.03 to 3.0.
    """
    models = tuple(models) if models else CAP_TABLE_MODELS
    bands = card_ridge_bands()
    if not bands:
        print("  REFUSED: no committed calibration resolves, so there is no ceiling")
        print("  this table is entitled to score against. It is NOT printed against")
        print("  the withdrawn cross-machine band, which belongs to no device.")
        return
    print("  Scored against EACH CARD'S OWN band, off its own measured_*.yaml. The")
    print("  cross-machine 160.3-176.2 this table used to quote was withdrawn from")
    print("  all 26 published reports on 2026-09-02 and is not a ceiling of anything.")
    print("  Correction bracket: (1 + phi + delta) over alpha_a in [0, 1] at delta = 0,")
    print(f"  the up and down GEMMs of {', '.join(models)}, BLOCK_N = {CAP_TABLE_BLOCK_N}.")
    print("  delta >= 0 is unmeasured and only lowers the cap, so the corrected column")
    print("  is itself an upper bound; alpha_a is unmeasured, so it is a bracket.")
    print()
    print("  | BLOCK_M | 2BM/(alpha b), upper bound | (1+phi+delta) bracket | "
          "cap bracket [alpha_a=1, alpha_a=0] | "
          + " | ".join(f"vs {card} {band[0]}-{band[-1]}"
                       for card, _ridge, band in bands) + " |")
    print("  |---:|---:|---|---|" + "---|" * len(bands))
    for block_m in (16, 32, 64, 128, 256):
        cap = ai_cap(block_m, alpha)
        f_lo, f_hi = lin_overstatement_bracket(block_m, models)
        c_lo, c_hi = corrected_cap_bracket(block_m, alpha, models)
        print(f"  | {block_m} | {cap:.0f} | {f_lo:.2f}-{f_hi:.2f} | "
              f"[{c_lo:.0f}, {c_hi:.0f}] | "
              + " | ".join(cap_bracket_verdict(c_lo, c_hi, band)
                           for _c, _r, band in bands) + " |")
    print()

    # FINDINGS C2's crossing was measured on the H200, so the ceiling it is
    # weighed against has to be the H200's own low end and not whichever card
    # sorts first. No H200 calibration, no paragraph: the alternative is
    # comparing one card's rows with another card's ridge, which is the exact
    # substitution this section now exists to have stopped doing.
    h200 = [entry for entry in bands if "h200" in entry[0]]
    if not h200:
        print("  The C2 comparison below needs the H200's own band and no committed")
        print("  H200 calibration resolved, so it is REFUSED rather than scored")
        print("  against another card.")
        return
    card, _ridge, band = h200[0]
    measured = rows_per_expert("mixtral-8x7b", MIXTRAL_ONE_STAGE_CROSSING_TOKENS)
    ceiling = max_alpha_that_still_crosses(CUTLASS_BLOCK_M, band[0])
    c_lo, c_hi = corrected_cap_bracket(CUTLASS_BLOCK_M, alpha, models)
    print("  AND THAT IS REFUTED BY THIS STUDY'S OWN ROWS. torch grouped_mm runs at")
    print(f"  CUTLASS BLOCK_M={CUTLASS_BLOCK_M} and DOES cross: FINDINGS C2 puts mixtral's")
    print(f"  one-stage bf16 crossing at {MIXTRAL_ONE_STAGE_CROSSING_TOKENS} tokens, "
          f"which is {measured:.0f} rows per expert,")
    print(f"  well above the {ai_cap(CUTLASS_BLOCK_M, alpha):.0f} this alpha allows as an")
    print(f"  upper bound, and further above the corrected [{c_lo:.0f}, {c_hi:.0f}] "
          f"(at BLOCK_N={CAP_TABLE_BLOCK_N}, vLLM's pin;")
    print("  CUTLASS's own tile width is not observed, so that bracket is the sweep's")
    print("  geometry applied to torch's rows). The correction only widens the gap.")
    print(f"  Those rows are {card} rows, and the band below is {card}'s.")
    print()
    print("  So one of three things is true, and this pool cannot say which:")
    print(f"   - the TRAFFIC coefficient is at most {ceiling:.3f}, the largest value at")
    print(f"     which BLOCK_M={CUTLASS_BLOCK_M} still reaches a ridge of {band[0]}, and the")
    print(f"     gap up to {alpha:.2f} is an extra tile's NON-traffic cost;")
    print("   - the bounded-AI consequence does not follow from a time-fitted alpha;")
    print("   - or the one-stage crossings are tile steps rather than the ridge, which")
    print("     FINDINGS already downgrades them to being.")
    print("  A BLOCK_M sweep on a pod separates them, and it is the experiment")
    print("  FINDINGS already names.")


def report(args, arms: list[str] | None = None) -> int:
    census: collections.Counter = collections.Counter()
    triton = collect(args.csvs, census, include_throttled=args.include_throttled)

    print("# alpha, refit against the derived tile")
    print()
    print("Every BLOCK_M under a vLLM span below is DERIVED from vLLM 0.27.1's config")
    print("lookup plus the row's own gpu_name, and never observed: the published")
    print("arms are schema v3 and record no tile. torch's 64 is OBSERVED, under C1.")
    print()
    _report_inputs(args, arms)
    print()
    _report_pool(triton, census)
    if len(triton) < 2:
        print()
        print("nothing admitted; there is no fit to report")
        return 1
    print()
    _report_roof(triton)
    print()
    if not _report_instruments(triton, args.pool_instruments):
        return REFUSED
    alpha = fit_alpha(triton)
    print()
    _report_fit(triton, alpha, args)
    print()
    _report_basis_contrast(triton, args)
    print()
    _report_splits(triton)
    if args.original_estimator:
        print()
        original = [p for p in args.csvs if ORIGINAL_ALPHA_ARM in Path(p).parent.name]
        if not original:
            print(f"## the original estimator: `{ORIGINAL_ALPHA_ARM}` is not in the")
            print("   inputs, so the published 0.10 cannot be reproduced from them")
        else:
            _report_original(collect(original, collections.Counter(), cutlass=True,
                                     include_throttled=args.include_throttled), alpha)
    if args.adversarial:
        print()
        _report_adversarial(triton, alpha, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csvs", nargs="*", type=Path,
                        help="the CSVs to fit. Omit them and pass --pinned-set "
                             "to read the manifest instead; giving both is "
                             "refused, because only one of them can be the "
                             "input set the header then names")
    parser.add_argument("--pinned-set", nargs="?", type=Path,
                        const=DEFAULT_REFIT_SET, default=None,
                        metavar="MANIFEST",
                        help="read the arms named in MANIFEST (default "
                             f"{DEFAULT_REFIT_SET.name} beside the published "
                             "arms) instead of a glob. This is the set "
                             "docs/FINDINGS.md quotes; a glob over the tree "
                             "reads whatever has landed since and answers "
                             "differently")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="cluster-bootstrap draws for the band (default 200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-throttled", action="store_true",
                        help="keep rows the clock gate would drop; off by "
                             "default, because such a row's time is not the "
                             "kernel's. The gate is whichever one the row's own "
                             "instrument recorded: `throttled` on a pre-v5 row, "
                             "the three under-load verdicts on a v5 one, of "
                             "which a LEVEL failure counts only on its LOW "
                             "side (a HIGH-side row is admitted without this "
                             "flag; see clock_gate). The name is the pre-v5 "
                             "flag's and is kept so the option means one thing "
                             "across the boundary")
    parser.add_argument("--pool-instruments", action="store_true",
                        help="fit rows from two or more timing instruments in "
                             "one pool. Off by default, and the run REFUSES "
                             "(exit 2) rather than pooling silently: alpha is "
                             "identified inside a group intercept and the "
                             "apparatus differ in level by construction. With "
                             "it, the count per instrument is printed beside "
                             "every headline number")
    parser.add_argument("--original-estimator", action="store_true",
                        help="also run the pooled-CV estimator that produced 0.10, on "
                             "the rows it was originally run on")
    parser.add_argument("--adversarial", action="store_true",
                        help="the checks against this fit's own answer")
    args = parser.parse_args(argv)
    # REFUSED RATHER THAN MERGED OR SILENTLY PREFERRED. With both given, one of
    # them is not the input set, and the header would name a set that is not
    # what was read -- which is the defect `--pinned-set` was added to close.
    if args.csvs and args.pinned_set is not None:
        parser.error("pass CSVs or --pinned-set, not both: with both given the "
                     "header cannot honestly name the input set")
    arms = None
    if args.pinned_set is not None:
        arms, args.csvs = read_refit_set(args.pinned_set)
    if not args.csvs:
        parser.error("no input: pass CSVs, or --pinned-set to read "
                     f"{DEFAULT_REFIT_SET.name}")
    return report(args, arms)


if __name__ == "__main__":
    raise SystemExit(main())
