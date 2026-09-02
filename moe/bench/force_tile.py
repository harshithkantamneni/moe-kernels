r"""MOE_FORCE_TILE: the pinning hook the sweep path was documented to have.

    MOE_FORCE_TILE='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":64,
                     "GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}' \
        python -m moe.bench.cli --profile crossing-uniform --env vllm \
               --groups baselines --impl vllm_fused_experts

WHY THIS EXISTS. `scripts/pod_session.sh` has set this variable since
2026-09-01 and then gated (S6a) on reading the OBSERVED `tile_block_m` back out
of the CSV, and `docs/POD_RUNBOOK.md` documents it as the way the dense grid is
pinned. NOTHING IN THE SWEEP PATH READ IT. From the 2026-09-01 ledger:

    "S6a tile pinning is honoured FAIL observed tile_block_m = none ... No
     pinning hook is wired into the sweep path, so the dense grid runs on
     vLLM's own ladder. The crossing it produces is then a MIX of tile steps
     and the roofline transition, which is instrument defect 3, and it may not
     be quoted as a tile-pinned crossing."

A 54-minute, 3,696-row sweep therefore produced a crossing that could not be
quoted for the purpose it was run for, and the pinned answer had to come from
`scripts/block_m_crossing_sweep.py` instead -- which forces the tile through the
same `override_config` hook this module now reaches from the main harness.

WHY A PINNED GRID SAYS SOMETHING AN UNPINNED ONE CANNOT. vLLM's fallback ladder
steps BLOCK_SIZE_M 16 -> 32 -> 64 -> 128 with the token count (M<=32, <=96,
<=512, else) and the tuned files step it too: the published 2026-09-01 vLLM arm
carries all four values inside one sweep, 16 at the small end and 128 at the
large one. A throughput curve along an unpinned token grid is therefore a tile
staircase and a roofline transition superimposed, and `crossing_from_points`
returns the FIRST crossing, which is usually a tile step. Pinning removes one of
the two mechanisms; what is left is attributable to the other.

THE THREE RULES, each of them a failure this module refuses to reproduce.

1. MALFORMED REFUSES BEFORE ANYTHING IS SPENT. A config that does not parse, or
   that omits a key the kernel takes as a keyword argument, raises at startup
   rather than falling through to the ladder. "The variable was set" and "the
   kernel ran that tile" are different facts and the gap between them is what
   cost the 2026-09-01 session its pinned crossing.

2. A PATH THAT CANNOT HONOUR THE PIN IS NEVER SILENTLY UNPINNED. Only vLLM's
   fused_moe exposes a hook to force a Triton tile. torch's grouped GEMM is
   CUTLASS, whose tile is fixed at 64 by the wgmma instruction shape (claim C1);
   SGLang runs its own tuning tree this harness has never probed; the reference
   spans have no tile at all. Cells on those paths are NOT measured while a pin
   is active, and each one is recorded in the manifest under a status of its
   own. A row that says it was pinned when it was not is worse than no pinning.

3. THE ROW RECORDS WHAT RAN, NOT WHAT WAS ASKED FOR. The tile columns are still
   filled by the OBSERVER in `_framework_config.recording_tile_config`, which
   reads the config back out of vLLM during a real call. This module then
   compares that observation against what it asked for and refuses the cell when
   they differ, so `tile_config_source == "vllm_override"` on a published row
   means the kernel was SEEN running that tile rather than that an environment
   variable was exported. That is the distinction S6a exists to make, and it is
   why nothing here ever writes a tile column itself.

WHAT IT IS NOT. Not a tuning knob and not a sweep axis: one process runs one
forced tile, because the tile is part of a cell's identity and the resume
manifest is keyed by it (see `CellPin.key_suffix`). Sweeping BLOCK_SIZE_M is
`scripts/block_m_crossing_sweep.py`'s job, which owns its own grid, its own
predictions and its own compile assay.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from ..baselines._framework_config import CONFIG_KEY_TO_COLUMN, ForceTileNotHonoured
from . import schema as SC

#: The variable `scripts/pod_session.sh` sets and `docs/POD_RUNBOOK.md`
#: documents. One spelling: there is deliberately no `--force-tile` flag, for
#: the reason `cli.parse_routing` gives for having one spelling of a routing --
#: two spellings of the same knob drift, and a sweep whose CSV cannot say which
#: one was used is not reproducible.
ENV_VAR = "MOE_FORCE_TILE"

#: Method a span defines to RUN under a tile configuration chosen from outside.
#: Mirrors `driver.TILE_OBSERVER`, and its ABSENCE is load-bearing: a span
#: without this method cannot be pinned, which is a refusal (rule 2) and never a
#: default.
FORCE_TILE_HOOK = "force_tile_config"

#: Exactly the keys a forced config must carry -- the same six the observer maps
#: back into schema columns, so a forced tile and an observed tile are described
#: in one vocabulary and cannot drift apart.
#:
#: EXACTLY, not a subset and not a superset. `try_get_optimal_moe_config`
#: returns the override INSTEAD of the tuned dict rather than merging into it,
#: and `fused_experts_impl` splats what comes back into the kernel launch. So a
#: config missing `num_stages` is a TypeError several minutes into a metered
#: sweep, and a config carrying `BLOCK_M` (the name the kernel's parameter has,
#: not the name the config file uses) is a launch with a different tile than the
#: one the operator typed.
REQUIRED_KEYS: tuple[str, ...] = tuple(CONFIG_KEY_TO_COLUMN)

#: Keys Triton requires to be powers of two. The block sizes index `tl.arange`,
#: which refuses anything else, and `num_warps` is a launch parameter with the
#: same restriction. GROUP_SIZE_M is deliberately absent: it only divides the
#: program id into swizzle groups and any positive value runs. Checked here
#: rather than at the launch, because the launch is inside a metered sweep and
#: this is not.
POWER_OF_TWO_KEYS: frozenset[str] = frozenset(
    {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps"})

#: What `tile_config_source` must read on a row produced under a pin. Checked
#: against the schema at import, structurally rather than in a test, for the
#: reason `tile_resolve` checks its field names that way: a test can be skipped
#: and a typo'd source is not a loud failure, it is a value no group-by matches.
TILE_SOURCE_OVERRIDE = "vllm_override"
if TILE_SOURCE_OVERRIDE not in SC.TILE_SOURCES:              # pragma: no cover
    raise AssertionError(
        f"{TILE_SOURCE_OVERRIDE!r} is not in schema.TILE_SOURCES; a pinned row "
        f"would carry a source no analysis can match")

#: `CellPin.status` values.
OFF = "off"                    # nothing was asked of this cell
PINNED = "pinned"              # a span will run it under the forced config
UNHONOURABLE = "unhonourable"  # a pin was asked for and no span can honour it


class ForceTileMalformed(ValueError):
    """The variable was set to something that is not a runnable tile config.

    Raised at startup, before a GPU is touched, rather than being ignored in
    favour of vLLM's ladder. Ignoring it is what "the sweep ran unpinned and
    nothing said so" looks like from the inside.
    """


@dataclass(frozen=True)
class ForcedTile:
    """One tile configuration, as typed, with its provenance attached."""

    #: `(key, value)` pairs in REQUIRED_KEYS order rather than a dict, so the
    #: object is hashable and cannot be mutated by a consumer into a config that
    #: was never typed.
    items: tuple[tuple[str, int], ...]
    #: Where it came from, for the message a refusal prints. Always ENV_VAR
    #: today; a parameter so a test does not have to lie about it.
    source: str = ENV_VAR

    @property
    def config(self) -> dict[str, int]:
        """A fresh dict per call: vLLM's `fused_experts_impl` MUTATES the config
        it is handed (it overwrites BLOCK_SIZE_M for the tail chunk), so handing
        the same object to two cells would let the first one edit the second."""
        return dict(self.items)

    @property
    def block_m(self) -> int:
        return self.config["BLOCK_SIZE_M"]

    def fingerprint(self) -> str:
        """Short, stable, and legible in a manifest: `bm128-3f2a1c9d`.

        The digest covers EVERY key, not only BLOCK_SIZE_M. Two runs that differ
        only in GROUP_SIZE_M are different experiments -- the 2026-09-01 session
        measured alpha 0.84 at G=1 against 0.67 at G=64 on both cards -- and a
        key that omitted the swizzle would let the second resume the first.
        """
        blob = json.dumps(dict(self.items), sort_keys=True).encode()
        return f"bm{self.block_m}-{hashlib.sha1(blob).hexdigest()[:8]}"

    def describe(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.items) + f"  [{self.source}]"


def parse(raw: str, source: str = ENV_VAR) -> ForcedTile:
    """Read one forced tile, refusing anything that is not exactly one.

    Every branch raises rather than repairing. The repair that suggests itself
    at each of them -- default the missing key, drop the unknown one, round the
    odd block size -- produces a sweep that runs a tile nobody typed and labels
    every row with the tile they did.
    """
    text = raw.strip()
    if not text:
        raise ForceTileMalformed(
            f"{source} is set but empty. That is a variable that was meant to "
            f"carry a tile config and did not (an unset shell variable expanded "
            f"into an assignment, usually). UNSET it to run vLLM's own ladder; "
            f"an empty value is not a way to ask for one.")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ForceTileMalformed(
            f"{source} is not JSON: {e}. It reads {text!r}. Expected an object "
            f"with exactly {list(REQUIRED_KEYS)}, e.g.\n"
            f"  {source}='{_EXAMPLE}'") from None
    if not isinstance(loaded, dict):
        raise ForceTileMalformed(
            f"{source} parsed as {type(loaded).__name__}, not an object. A tile "
            f"config is a JSON object of {list(REQUIRED_KEYS)}.")

    missing = [k for k in REQUIRED_KEYS if k not in loaded]
    unknown = sorted(set(loaded) - set(REQUIRED_KEYS))
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise ForceTileMalformed(
            f"{source}: {', and '.join(parts)}. vLLM's override REPLACES the "
            f"config rather than merging into it and the result is splatted "
            f"into the kernel launch, so a missing key is a TypeError mid-sweep "
            f"and an unknown one is a launch nobody asked for. Give exactly "
            f"{list(REQUIRED_KEYS)}.")

    values: dict[str, int] = {}
    for key in REQUIRED_KEYS:
        value = loaded[key]
        # bools first: `isinstance(True, int)` is True in python, and
        # BLOCK_SIZE_M=true would sail through every check below as a 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ForceTileMalformed(
                f"{source}: {key}={value!r} is {type(value).__name__}, not an "
                f"int. JSON floats included: 128.0 reaches the kernel as a "
                f"float and Triton's constexpr specialisation is not the same "
                f"for it.")
        if value <= 0:
            raise ForceTileMalformed(
                f"{source}: {key}={value} is not positive. There is no tile of "
                f"height zero, and schema.tile_field refuses to read a 0 back "
                f"as a measurement for exactly that reason.")
        if key in POWER_OF_TWO_KEYS and value & (value - 1):
            raise ForceTileMalformed(
                f"{source}: {key}={value} is not a power of two. Triton indexes "
                f"a block with tl.arange, which refuses it -- and it would "
                f"refuse it inside the first timed cell of a metered sweep "
                f"rather than here.")
        values[key] = value
    # In REQUIRED_KEYS order, not alphabetical: that is the order vLLM's config
    # files and every message in this project print a tile in, and BLOCK_SIZE_K
    # ahead of BLOCK_SIZE_M reads as a different tile at a glance. The
    # fingerprint sorts its own keys, so the digest does not depend on this.
    return ForcedTile(tuple((k, values[k]) for k in REQUIRED_KEYS), source=source)


#: Printed in the refusal above, so the fix is on the screen with the error.
#: The values are the ones pod_session.sh's S6a probe sets.
_EXAMPLE = json.dumps({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128,
                       "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                       "num_warps": 8, "num_stages": 4}, separators=(",", ":"))


def from_env(environ: Mapping[str, str] | None = None) -> ForcedTile | None:
    """The forced tile this process was launched with, or None for the ladder.

    None ONLY when the variable is absent. An empty or unparseable value raises:
    see `parse`.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR)
    if raw is None:
        return None
    return parse(raw)


# --------------------------------------------------------------------------
# which cells can honour it
# --------------------------------------------------------------------------

def can_pin(span) -> bool:
    """Does this span expose the hook, i.e. can it run a tile chosen outside it?"""
    return callable(getattr(span, FORCE_TILE_HOOK, None))


def split_by_pinnability(spans: Iterable) -> tuple[list[str], list[str]]:
    """`(can pin, cannot pin)` span names, for the dry run's plan.

    Off the GPU box neither vLLM nor SGLang registers at all, so a laptop dry
    run reports every planned implementation in the second list. That is the
    honest answer and it is also the useful one: it is exactly what the run
    would do on a machine where the framework failed to import.
    """
    yes, no = [], []
    for span in spans:
        (yes if can_pin(span) else no).append(span.name)
    return sorted(yes), sorted(no)


def _why_not(targets) -> str:
    """The refusal message for a cell no span can pin. Names the spans."""
    named = ", ".join(t.name for t in targets) or "no span"
    return (f"{named} exposes no {FORCE_TILE_HOOK}(); only vLLM's fused_moe can "
            f"be forced to a Triton tile (torch's grouped GEMM is CUTLASS with "
            f"the tile fixed by the wgmma shape, SGLang runs its own tuning "
            f"tree this harness has not probed, and the reference spans have no "
            f"tile). Measuring it anyway would put an unpinned row in a pinned "
            f"sweep with nothing in the row to say so.")


@dataclass(frozen=True)
class CellPin:
    """What pinning means for ONE cell: nothing, a context, or a refusal."""

    forced: ForcedTile | None = None
    status: str = OFF
    #: Span that will honour it, "" otherwise.
    target: str = ""
    #: Why it cannot be honoured, "" otherwise.
    reason: str = ""
    #: The bound `force_tile_config` of `target`. Held rather than looked up
    #: again so the span that was ASKED is the span that RUNS it.
    hook: Callable | None = None

    @property
    def key_suffix(self) -> str:
        """What a pin adds to a cell's resume identity.

        THE PROJECT'S FAILURE MODE 2, and it is not hypothetical here: the
        manifest is keyed by cell identity, and the moment a forced tile exists
        it is part of that identity. Without this suffix a pinned run resumed
        into an unpinned run's directory finds every cell already recorded,
        skips all of them before the fp32 oracle, writes nothing, and the report
        prints the UNPINNED numbers under the pinned label. Empty when no pin is
        active, so an ordinary sweep's keys are byte-identical to what they have
        always been and existing manifests still resume.
        """
        if self.forced is None:
            return ""
        return f"|force_tile={self.forced.fingerprint()}"

    def applied(self):
        """The context every timed and untimed call of this cell runs inside.

        A nullcontext when nothing was asked, so the unpinned path pays nothing
        and cannot behave differently.
        """
        if self.status == OFF:
            return contextlib.nullcontext()
        if self.status != PINNED or self.hook is None or self.forced is None:
            raise ForceTileNotHonoured(
                f"this cell cannot be pinned ({self.reason or 'no target'}), so "
                f"it must be recorded rather than run; applied() is not the "
                f"path for it")
        ctx = self.hook(self.forced.config)
        if not (hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")):
            # NON-VACUITY: a hook that returns None enters nothing, and
            # `with None:` fails with an AttributeError that names neither the
            # span nor the tile. Say what is wrong while the names are in hand.
            raise ForceTileNotHonoured(
                f"{self.target}.{FORCE_TILE_HOOK}() returned "
                f"{type(ctx).__name__}, not a context manager, so nothing would "
                f"be forced for the duration of the cell")
        return ctx

    def disagrees_with(self, meta: Mapping) -> str:
        """Does the row this cell produced SHOW the tile that was forced?

        "" when it does, a message naming both when it does not. This is rule 3
        and the only reason the S6a gate means anything: `meta` is what the
        observer read back out of vLLM during a real call, so agreement here is
        evidence the kernel ran the tile, while the environment variable alone
        is evidence only that someone typed it.

        A pinned cell whose observation is UNRECORDED disagrees too. It is not a
        smaller failure: such a row is indistinguishable from an unpinned one in
        every group-by, which is precisely the state instrument defect 3
        describes.
        """
        if self.status != PINNED or self.forced is None:
            return ""
        wanted = self.forced.config
        wrong = [(k, wanted[k], meta.get(CONFIG_KEY_TO_COLUMN[k]))
                 for k in REQUIRED_KEYS
                 if meta.get(CONFIG_KEY_TO_COLUMN[k]) != wanted[k]]
        source = meta.get("tile_config_source")
        if not wrong and source == TILE_SOURCE_OVERRIDE:
            return ""
        detail = "; ".join(f"{k}: forced {want}, observed {got!r}"
                           for k, want, got in wrong)
        return (f"{self.target} was run under {self.forced.source} but the row "
                f"does not show it: source={source!r} (expected "
                f"{TILE_SOURCE_OVERRIDE!r})"
                + (f", {detail}" if detail else "")
                + ". The tile columns are the only evidence a reader has that a "
                  "row is pinned, so a cell that cannot produce them is "
                  "recorded rather than measured.")


def pin_for(forced: ForcedTile | None, spans: Iterable, span) -> CellPin:
    """What pinning means for one (pipeline, target) pair.

    `span` is the span under study, or None for a whole-layer row, and the
    target rule is deliberately the same one `driver.observe_tile_config` uses:
    ask the span itself, or every span of the tiling when the timer wraps the
    whole layer, and take the first that answers. In practice exactly one span
    in a tiling is a framework kernel.
    """
    if forced is None:
        return CellPin()
    targets = [span] if span is not None else list(spans)
    for target in targets:
        if can_pin(target):
            return CellPin(forced=forced, status=PINNED, target=target.name,
                           hook=getattr(target, FORCE_TILE_HOOK))
    return CellPin(forced=forced, status=UNHONOURABLE, reason=_why_not(targets))


# --------------------------------------------------------------------------
# what the sweep did with it
# --------------------------------------------------------------------------

@dataclass
class ForceTileLedger:
    """Cells pinned, cells skipped, cells whose row did not show the pin.

    NON-VACUITY. A sweep that pinned nothing writes no pinned rows and no
    complaint: every check downstream of it examines zero rows and reports zero
    failures. This is the counter that makes "the pin applied to nothing" a
    statement the CLI can refuse on, and it is why `pinned_cells` is incremented
    where the row is written rather than where the pin is planned.
    """

    pinned_cells: int = 0
    #: Cells already complete under THIS pin when the run started. A separate
    #: count, not added to the one above, for the reason
    #: `block_m_crossing_sweep`'s gate 0 refuses to score a setting that ran no
    #: cells: "this session measured nothing pinned" and "this session had
    #: nothing left to measure" are different states, and only the first is a
    #: failure. They are distinguishable at all only because the manifest key
    #: carries the pin's fingerprint, so a key present under a pinned key was
    #: completed under that exact tile.
    resumed_cells: int = 0
    #: impl name -> cells skipped because nothing on that path can be pinned.
    skipped: dict[str, int] = field(default_factory=dict)
    #: impl name -> why that impl could not be pinned. One entry per impl, not
    #: per cell: the reason is a property of the path, not of the token count.
    reasons: dict[str, str] = field(default_factory=dict)
    #: impl name -> the first disagreement seen between the pin and the row.
    unobserved: dict[str, str] = field(default_factory=dict)
    #: Cells refused for that reason. Counted as well as keyed by impl, because
    #: gate F1 is a number against a threshold and "3 implementations" is not
    #: the same number as "how many cells were thrown away".
    unobserved_cells: int = 0

    def record_pinned(self) -> None:
        self.pinned_cells += 1

    def record_resumed(self) -> None:
        self.resumed_cells += 1

    def record_skip(self, impl: str, reason: str) -> bool:
        """Count a skipped cell. True the FIRST time for this impl.

        The caller prints on a True. A dense grid skips thousands of cells on
        the same two implementations, and a line each would bury the summary
        that matters under its own repetition.
        """
        first = impl not in self.skipped
        self.skipped[impl] = self.skipped.get(impl, 0) + 1
        if first:
            self.reasons[impl] = reason
        return first

    def record_unobserved(self, impl: str, detail: str) -> bool:
        first = impl not in self.unobserved
        self.unobserved_cells += 1
        if first:
            self.unobserved[impl] = detail
        return first

    def gates(self) -> list[tuple[str, str, str, str, bool]]:
        """(id, claim, measured, threshold, passed), the project's gate shape.

        Both are VALIDITY gates rather than claim gates: a FAIL does not mean
        the tile behaved unexpectedly, it means the instrument did not do what
        the run was told to do, and no number the run produced may be quoted as
        tile-pinned. They are printed with their thresholds BEFORE the sweep and
        re-printed with their measurements after, so the prediction cannot be
        adjusted to fit what came back.
        """
        return [
            ("F1", "every row produced under the pin shows the forced tile",
             f"{self.unobserved_cells} cells disagreed", "== 0",
             self.unobserved_cells == 0),
            ("F2", "cells of this run stand under the pin",
             f"{self.pinned_cells} measured + {self.resumed_cells} already "
             f"complete, {self.skipped_cells} skipped as unpinnable", ">= 1",
             not self.vacuous()),
        ]

    @property
    def skipped_cells(self) -> int:
        return sum(self.skipped.values())

    def vacuous(self) -> bool:
        """A pin was asked for and not one cell of the run stands under it.

        Counts resumed cells as standing under it: their rows are in the file,
        written under this pin's own manifest key. What this refuses is the
        state the 2026-09-01 session was in without noticing -- a variable set,
        a sweep run, and nothing pinned anywhere in the output.
        """
        return self.pinned_cells == 0 and self.resumed_cells == 0

    def summary(self, forced: ForcedTile | None) -> list[str]:
        """Lines for the end of a run. Empty when no pin was asked for."""
        if forced is None:
            return []
        out = [f"[force-tile] {forced.describe()}",
               f"[force-tile] {self.pinned_cells} cells ran pinned, "
               f"{self.resumed_cells} were already complete under this tile, "
               f"{self.skipped_cells} skipped as unpinnable"]
        for impl, count in sorted(self.skipped.items()):
            out.append(f"[force-tile]   SKIPPED {impl}: {count} cells -- "
                       f"{self.reasons.get(impl, '')}")
        for impl, detail in sorted(self.unobserved.items()):
            out.append(f"[force-tile]   NOT OBSERVED {impl}: {detail}")
        return out
