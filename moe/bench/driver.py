"""Walks the benchmark matrix, gates on correctness, writes rows, and resumes.

Invariants this module enforces:
  - No timing number is written for an implementation that did not pass the
    golden-fp32 oracle on that exact cell in that same run.
  - The timer wraps the span under study, not the whole layer. Timing a full
    pipeline that contains python-loop reference stages would measure the
    reference, not the kernel.
  - Every row is flushed and fsynced before the next cell starts, so a killed
    pod loses at most one cell.
  - A completed unit of work is recorded in a manifest and skipped on re-run,
    so an interrupted sweep resumes instead of restarting.

The timing backend is injectable so the whole control flow above can be tested
on a laptop, before any of it runs on a metered GPU.
"""
from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import torch

from ..pipeline import Pipeline, PipelineError, build
from ..reference.torch_ref import expert_counts, golden_forward, make_inputs
from ..routing.imbalance import counts_from_offsets, expert_load
from ..spec import BenchSpec
from ..stages import BASE_ENV
from ..state import MoEState
from . import bytes_model as BM
from . import force_tile as FT
from . import roofline as RF
from . import schema as SC
from . import timing as T
from .roofline import Hardware
from .tolerance import relative_error, tolerance

# Returns forced top-k expert ids for a cell, or None to let the router decide.
RoutingSource = Callable[[BenchSpec], "torch.Tensor | None"]

#: `impl` value meaning "time the entire tiling end to end" rather than one span.
#: Bare, this is the ALL-REFERENCE whole layer, and 3,528 rows in each published
#: bf16 arm carry exactly this string. It must not be renamed.
PIPELINE_SCOPE = "__pipeline__"


def pipeline_scope_for(span_name: str) -> str:
    """The whole-layer `impl` name for a layer built around `span_name`.

    Distinct from the bare marker on purpose. Analyses key on the `impl` column
    -- compare.py builds by[(tokens, impl)] -- so an all-reference whole layer
    and a vLLM whole layer sharing one label would average a 7.3 ms python loop
    into a 0.588 ms kernel series.
    """
    return f"{PIPELINE_SCOPE}:{span_name}"


def is_pipeline_scope(impl: str) -> bool:
    """Does this `impl` mean "time the whole tiling" rather than one span?"""
    return impl == PIPELINE_SCOPE or impl.startswith(PIPELINE_SCOPE + ":")

def should_time_graph(cost, cfg: RunConfig) -> tuple[bool, str]:
    """Is isolating launch overhead worth a doubled sweep for this cell?

    Predict the roofline-minimum time from compulsory traffic. If a kernel
    launch is a smaller fraction of that than `graph_min_launch_share`, the
    graph/eager delta cannot be a first-order effect and the cell is not worth
    measuring twice. Errs toward measuring: an unknown bandwidth means yes.
    """
    if cfg.graph_min_launch_share <= 0:
        return True, ""
    bw = cfg.hardware.bandwidth_bytes_s if cfg.hardware else None
    if not bw:
        return True, ""
    predicted_ms = (cost.bytes_total / bw) * 1e3
    if predicted_ms <= 0:
        return True, ""
    share = cfg.launch_overhead_ms / predicted_ms
    if share >= cfg.graph_min_launch_share:
        return True, ""
    return False, (f"launch overhead is {share * 100:.2f}% of the "
                   f"{predicted_ms:.3f} ms roofline minimum, below the "
                   f"{cfg.graph_min_launch_share * 100:.1f}% threshold")


def time_kernel_graph(fn: Callable[[], None], *, on_captured=None,
                      **kw) -> T.KernelTiming:
    """Capture `fn` into a CUDA graph and time the REPLAY on the instrument.

    `timing.time_kernel` times a callable; a graph is a different callable made
    out of one, so the capture belongs to the caller and this is the caller. It
    is the graph half of what `timing.time_graph` used to do, minus the timing,
    which `time_kernel` now does for both modes -- the point being that eager and
    graph rows come off ONE apparatus and are therefore comparable with each
    other and with the roof.

    `on_captured` runs after the warmup replays and before the timed trials,
    while the graph is still the thing that produced the output. A replay writes
    into graph-private buffers every replay reuses, so a kernel leaving part of
    its output unwritten would show the PREVIOUS replay's correct values; the
    driver re-earns the correctness verdict there.

    Raises `timing.NotCapturable`, which is a finding rather than a failure: an
    implementation that syncs with the host cannot be used in real MoE
    inference, and the row records that.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except RuntimeError as e:
        raise T.NotCapturable(str(e)) from None

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if on_captured is not None:
        on_captured()
    return T.time_kernel(graph.replay, **kw)


#: The knobs on `RunConfig` that only the RETIRED instrument reads. `warmup` is
#: a COUNT of calls and `iters` a fixed iteration count; `timing.time_kernel`
#: takes neither, warming for a duration (`warmup_ms`) and sizing iterations
#: from the warmup's own queue-deep per-call time.
RETIRED_KNOBS: tuple[str, ...] = ("warmup", "iters")


def instrument_modes(cfg: RunConfig) -> list[str]:
    """Which of this run's timing modes measure on `timing.time_kernel`.

    A mode is on the instrument when the sweep runs it at all (it is in
    `graph_modes`) and no legacy timer is injected for it. Written once and read
    by both config guards, because they are asking the same question -- what
    will this run actually put through the instrument -- and two copies of that
    answer is exactly how a wall ends up built at one of two ways in.
    """
    modes = []
    if False in tuple(cfg.graph_modes) and cfg.timer_eager is None:
        modes.append("eager")
    if True in tuple(cfg.graph_modes) and cfg.timer_graph is None:
        modes.append("graph")
    return modes


def unhonourable_retired_knobs(cfg: RunConfig) -> list[tuple[str, object, str]]:
    """`(knob, value, mode)` for every retired knob this run would DROP.

    A knob is unhonourable when it was set to something other than its own
    field default AND the mode it would apply to measures on the instrument,
    which reads no such argument. Empty for the ordinary sweep, empty when the
    retired timer is injected for that mode, and empty when the value equals
    the default, because a caller that passes the default asked for nothing.

    THE DEFECT IT NAMES. `_instrument_kwargs` passes warmup_ms/target_ms/
    trials/l2_flush/reference_clock_mhz/flusher and nothing else, while
    `moe/bench/cli.py` filled `warmup` and `iters` from the profile on EVERY
    run. Two profiles set them: `smoke` (warmup=5, iters=10) to be quick, and
    `profile-cell` (warmup=5, trials=1, iters=1), whose note read "one cell, one
    launch: the shape ncu can read a counter off" and which four session scripts
    invoke. On the instrument path that one launch silently became
    `iters_for(per_call_ms, 200)`, which is 10 to 2000, and the five-call warmup
    became 300 ms of sustained load. Nothing warned, nothing recorded it, and a
    counter read off the wrong shape is a number that looks exactly like a right
    one.

    NEITHER PROFILE SETS THEM NOW and `Profile` no longer has the fields, so
    this returns empty for every shipped profile and the two say what they meant
    in `warmup_ms`/`target_ms` instead. The check stays because the fields stay
    on `RunConfig`, where a caller that injects a legacy timer still needs them,
    and a caller that injects nothing must not be able to reach the instrument
    with a count in hand.
    """
    fields = RunConfig.__dataclass_fields__
    asked = [(name, getattr(cfg, name)) for name in RETIRED_KNOBS
             if getattr(cfg, name) != fields[name].default]
    if not asked:
        return []
    on_instrument = instrument_modes(cfg)
    return [(name, value, mode)
            for name, value in asked for mode in on_instrument]


def _retired_knob_refusal(cfg: RunConfig, dropped) -> str:
    """The message. Names each dropped knob, its value, and both ways out.

    REFUSES RATHER THAN WARNS because a warning on a metered pod scrolls past
    and the rows it qualifies outlive it. There are exactly two honest
    resolutions and the message states both: express the intent in the units
    the instrument takes, or put that mode back on the retired timer, which
    stamps `schema.LEGACY_INSTRUMENT` into every row it writes so the choice is
    legible in the data afterwards.
    """
    shown = sorted({f"{name}={value!r}" for name, value, _ in dropped})
    names = sorted({mode for _, _, mode in dropped})
    knobs = (f"{', '.join(shown)} are RETIRED instrument knobs"
             if len(shown) > 1 else f"{shown[0]} is a RETIRED instrument knob")
    modes = (f"the {' and '.join(names)} modes of this run measure"
             if len(names) > 1 else f"the {names[0]} mode of this run measures")
    return (
        f"{knobs}, and {modes} on `timing.time_kernel`, which has no such "
        f"parameter: it warms for a DURATION (warmup_ms={cfg.warmup_ms}) and "
        f"sizes iters from the warmup's own per-call time toward target_ms="
        f"{cfg.target_ms}. Passing them here would change nothing and record "
        f"nothing, which is how `profile-cell` asked for one launch and got up "
        f"to 2000 of them. Say it in the instrument's units (warmup_ms, "
        f"target_ms, trials), or set timer_eager/timer_graph to put that mode "
        f"back on the retired timer that does read them.")


class RetiredKnobRefused(T.TimingRefused):
    """This run asked for a knob the instrument it measures on cannot honour.

    A SEPARATE TYPE so `run_sweep` can let it out. That loop catches
    `Exception` per cell and records a crash, which is right for a kernel that
    launched badly and wrong for a configuration error: the config is the same
    for every cell, so swallowing it would print one warning per cell, write no
    rows, and exit 0 -- the loudest possible way to say nothing.
    """


def refuse_dropped_retired_knobs(cfg: RunConfig) -> None:
    """Raise if measuring this cell would silently drop a retired knob.

    CHECKED HERE, per cell and just before the first thing that costs anything,
    rather than in `RunConfig.__init__`. A knob is only dropped when a cell is
    actually measured on the instrument, and a sweep can construct a config it
    never measures with: `tests/test_force_tile.py` drives the CLI purely to
    prove a force-tile plan is refused before anything is spent, and every cell
    in that run is declined by the pin. A refusal at construction would fail
    such a run for a knob no cell would ever have reached.

    AND THE CALLER HAS TO SURVIVE IT. This raises out of `run_sweep` rather than
    being recorded per cell, so the process that called it exits on it;
    `cli.main` catches `TimingRefused` and exits REFUSED with the message, which
    is what makes this a free refusal instead of an uncaught traceback that
    `exit_codes` would read as a measured CLAIM_FAIL.
    """
    dropped = unhonourable_retired_knobs(cfg)
    if dropped:
        raise RetiredKnobRefused(_retired_knob_refusal(cfg, dropped))


class ReferenceClockRefused(T.TimingRefused):
    """This run would measure a real card with no clock to level it against.

    A SEPARATE TYPE for the same reason `RetiredKnobRefused` is one, and it
    escapes `run_sweep` for the same reason: the config is the same for every
    cell, so recording it per cell would print one warning per cell, write a
    sweep of rows whose LEVEL column is undetermined, and exit 0.

    A `TimingRefused` because that is the code path that already means "nothing
    was measured and nothing was spent": `cli._main` catches it around
    `run_sweep` and exits REFUSED(2). A plain exception here would arrive at
    `cli.main`'s catch-all as ERROR(4).
    """


def _missing_key(family: str) -> str:
    """The `RunConfig.missing` key a family's absent reference is recorded
    under. The bf16 GEMM's is the bare field name, as it has been since the
    field existed; the others are suffixed so one map can say which GEMM."""
    if family == RF.BF16_FAMILY:
        return "reference_clock_mhz"
    return f"reference_clock_mhz[{family}]"


def unreferenced_clock(cfg: RunConfig, dtype: str | None = None) -> list[str]:
    """The instrument modes this run would measure with no reference clock.

    Empty unless ALL THREE hold: a mode goes through `time_kernel`, no
    reference was resolved or supplied FOR THIS CELL'S DTYPE FAMILY, and A
    CARD IS ATTACHED. The third is the one that decides what this check is
    for. With a card and no reference, the run is about to spend metered
    minutes writing rows whose LEVEL column can never say anything, which is
    the state that wrote all 100,144 published rows. With no card,
    `time_kernel` already refuses unless the caller injected the fakes that
    stand in for CUDA, and a fake clock sampler has no clock to be level
    against in the first place -- so the reason is recorded in `cfg.missing`
    and the laptop run proceeds.

    PER DTYPE since 2026-09-08. `dtype` None asks about the run's own
    reference, the bf16 GEMM's, which is what every caller before that date
    asked; an fp8 dtype asks about the fp8 GEMM's clock (`RunConfig.
    reference_for`), because levelling an fp8 cell against the bf16 GEMM's
    clock filed a cell running at its own roof's 1905 MHz as LEVEL-failed
    HIGH against 1515.
    """
    if dtype is None or RF.reference_family(dtype) == RF.BF16_FAMILY:
        if cfg.reference_clock_mhz is not None or not cfg.reference_clock_card:
            return []
        return instrument_modes(cfg)
    ref = cfg.reference_for(dtype)
    if ref.mhz is not None or not (ref.card or cfg.reference_clock_card):
        return []
    return instrument_modes(cfg)


def _reference_clock_refusal(cfg: RunConfig, modes: list[str],
                             dtype: str | None = None) -> str:
    """The message. Names the card, the GEMM, the reason from the missing map,
    what is lost, and both ways out.

    The reason is `cfg.missing[...]` under the family's key, quoted rather
    than re-derived, so the sentence the operator reads on the pod is the same
    sentence a report stamping that map would carry.
    """
    family = RF.reference_family(dtype) if dtype is not None else RF.BF16_FAMILY
    which = (f"the {' and '.join(modes)} modes of this run measure"
             if len(modes) > 1 else f"the {modes[0]} mode of this run measures")
    card = cfg.reference_clock_card
    if family != RF.BF16_FAMILY:
        card = cfg.family_references[family].card or card
    if family == RF.BF16_FAMILY:
        way_out = ("or pass RunConfig(reference_clock_mhz=...) with the clock "
                   "you are levelling against and say in the write-up where "
                   "the number came from.")
    else:
        way_out = (f"or inject RunConfig(reference_clock_resolvers={{{family!r}: "
                   "...}}) with a `roofline.ReferenceClock` for that GEMM and "
                   "say in the write-up where the number came from.")
    return (
        f"{which} on `timing.time_kernel` against the card {card!r}, and no "
        f"clock was resolved for its {family} GEMM, the roof a "
        f"{dtype or family} cell is scored against: "
        f"{cfg.missing.get(_missing_key(family), 'reason not recorded')}.\n"
        "    Without a reference, `timing.clock_flags` leaves clock_level_ok "
        "None on every cell and the LEVEL verdict -- the flag that exists "
        "because the retired throttle check detected an idle boost rather than "
        "throttling -- cannot fire. A sweep of rows that CANNOT report a "
        "clock problem is not a conservative sweep, it is an unexamined one, "
        "and it costs the same rental as a sweep that can.\n"
        "    Two ways out, both cheap: run `python "
        "scripts/calibrate_hardware.py --publish` on this box, which measures "
        "this card's roofs and writes the clock each dense GEMM ran at into "
        f"moe/bench/hardware/, {way_out}")


def refuse_unreferenced_clock(cfg: RunConfig, dtype: str | None = None) -> None:
    """Raise if this cell would measure an attached card with no LEVEL reference.

    CHECKED BESIDE `refuse_dropped_retired_knobs`, per cell and just before the
    first thing that costs anything, for the reason that function gives: a
    sweep can construct a config it never measures with, and a refusal at
    construction would fail a run whose every cell is declined by the pin.
    Per cell is also what makes it per DTYPE: the cell's format names the
    GEMM whose clock it is levelled against, and a run may carry both.
    """
    modes = unreferenced_clock(cfg, dtype)
    if modes:
        raise ReferenceClockRefused(_reference_clock_refusal(cfg, modes, dtype))


@dataclass
class RunConfig:
    out_dir: Path = Path("results")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    env_name: str = "base"
    #: Sustained-load warmup, in MILLISECONDS of delivered GPU time. The unit
    #: the instrument takes, and the unit a warmup has to be in: a count settles
    #: a 30 ms GEMM in one call and leaves a 1 ms kernel below the clock
    #: governor's response for hundreds. 300 ms is what every ladder script in
    #: this repository already defaults to, so one number means one thing.
    warmup_ms: float = 300.0
    #: Kernel time one trial should hold, which is what `iters` is sized from.
    target_ms: float = 200.0
    trials: int = 3
    #: The clock the ROOF was measured at, for the LEVEL verdict. RESOLVED from
    #: this card's own calibration in `__post_init__`, not left at None: the
    #: comment that stood here until 2026-09-03 said "no committed calibration
    #: records it yet", and `moe/bench/hardware/measured_nvidia_h200.yaml` has
    #: carried `detail.gemm_clock_mhz: 1515` since 2026-09-02. What is true is
    #: only the parenthetical: `roofline.Hardware` carried bandwidth and peaks
    #: and DROPPED everything under `detail` (it carries the reference clocks
    #: since 2026-09-08, for the recompute mirror; the driver still resolves
    #: its own through the resolver below, which also answers whether a card
    #: is attached), so the driver read the roof through a type that had
    #: already thrown the clock away and concluded the number did not exist.
    #: The cost of that was total, not partial: `clock_level_ok` is None
    #: without a reference, so the LEVEL verdict -- the flag installed because
    #: the retired throttle check detected an idle boost rather than
    #: throttling, and the whole reason `TIMING_BASIS` left v1 -- could never
    #: fire on the path that wrote all 100,144 published rows and that
    #: `scripts/alpha_refit.py` reads for the headline alpha.
    #:
    #: THIS IS THE bf16 GEMM'S CLOCK, the run's primary reference: the roof
    #: for every non-fp8 dtype. An fp8 cell is levelled and roofed against the
    #: fp8 GEMM's clock, resolved per family by `reference_for` below; this
    #: field and its source/grade/card describe the bf16 one only.
    #:
    #: Set it explicitly to a POSITIVE number to override the calibration; that
    #: is recorded in `reference_clock_source` and skips the resolution
    #: entirely. A non-positive one is neither an override nor a reference, so
    #: it is dropped and the resolution still runs, for the CARD, so the
    #: refusal fires on it like any other missing reference. None means
    #: "resolve it", which is why there is no way to ask for no reference at
    #: all: a run that measures a real card with no reference REFUSES (see
    #: `refuse_unreferenced_clock`) rather than writing a sweep of rows whose
    #: LEVEL column is permanently undetermined.
    reference_clock_mhz: float | None = None
    #: Where that number came from, or, when it is None, why it is not known.
    #: One field either way, from `roofline.ReferenceClock.source`.
    reference_clock_source: str = ""
    #: HOW that number was taken, `roofline.ReferenceClock.grade`: one of the
    #: `roofline.REFERENCE_*` constants, "caller" for an explicit number, ""
    #: for none. The per-row roof (`_apply_cost`) is rescaled only against
    #: `REFERENCE_UNDER_LOAD`; against the idle scalar both committed
    #: calibrations carry it is REFUSED with the reason in `roof_note`, because
    #: scaling by `load / reference` puts the reference into every number and
    #: that scalar moved 30% across one card's calibrations. LEVEL is still
    #: scored against a disowned reference, and every row carries the source
    #: so the verdict can be discounted; the alternative, refusing the sweep,
    #: would refuse every sweep against the committed yamls until a
    #: recalibration, and the pod runbook recalibrates at step 1 anyway.
    reference_clock_grade: str = ""
    #: The attached card the resolution saw, "" for none, and SET ON EVERY PATH
    #: that ends without a clock. The refusal turns on it together with the
    #: clock: a card with no usable calibration is a pod misconfiguration, and
    #: no card at all is a laptop, where no clock is being sampled anyway. A
    #: path that dropped the clock and left this "" is a refusal that cannot
    #: fire, which is how the non-positive branch shipped inert.
    reference_clock_card: str = ""
    #: Why a field this config could not determine is None, keyed by field name.
    #: The same contract `provenance.Provenance.missing` states, so a caller
    #: that stamps a report can lift these entries into that block unchanged: a
    #: None here is complete when this map says why, and a guess would not be.
    missing: dict[str, str] = field(default_factory=dict)
    #: THE RETIRED INSTRUMENT'S KNOBS, read only when a legacy timer is injected
    #: below. `warmup` is a CALL COUNT and `iters` a fixed iteration count; both
    #: are ignored on the instrument path, which warms for a duration and sizes
    #: iterations from the warmup's own queue-deep per-call time.
    warmup: int = 25
    iters: int | None = None          # None: derive from FLOPs
    flush_mb: int = T.DEFAULT_FLUSH_MB
    flush_mode: str = "read"
    input_scale: float = 1.0
    #: Reuse expert weights across cells that share (model, dtype, seed).
    #: Values are bit-identical either way; the contract is that no
    #: implementation writes to `weights`. Set False to opt out.
    reuse_weights: bool = True
    l2_modes: tuple[bool, ...] = (True, False)
    graph_modes: tuple[bool, ...] = (False, True)
    calibration: dict | None = None
    #: Optional callable returning per-cell routing provenance, e.g. the trace
    #: fingerprint and the resolved batch/layer slice.
    routing_info: Callable[[BenchSpec], dict] | None = None
    #: Skip CUDA-graph timing when predicted kernel time is so long that launch
    #: overhead cannot matter. Graph mode doubles a metered sweep; at DeepSeek
    #: geometry with a few tokens the roofline minimum is already ~0.8 ms
    #: against a ~5 us launch, so the axis would cost half the session to
    #: measure a sub-1% effect. Set to 0.0 to always time both.
    graph_min_launch_share: float = 0.01
    launch_overhead_ms: float = 0.005
    #: Measured ceilings from scripts/calibrate_hardware.py, as a Hardware.
    #: Without it the efficiency columns stay zero rather than being quoted
    #: against a spec peak this machine may never reach. It already knows its
    #: own ridge point and bound classification, so the driver does not.
    hardware: Hardware | None = None
    validate_shapes: bool = True
    device: str = "cuda"
    #: The tile every cell of this run must be pinned to, from MOE_FORCE_TILE.
    #: None is the ordinary sweep: vLLM picks its own tile per token count and
    #: the rows record which one. Set, it becomes part of a cell's identity
    #: (see _cell_key) and a cell whose implementation cannot honour it is
    #: recorded rather than measured. moe/bench/force_tile.py has the why.
    force_tile: FT.ForcedTile | None = None
    #: Counters the CLI refuses on. Lives on the config because run_sweep hands
    #: cfg to every cell and nothing else is threaded through all of them; a
    #: sweep that pinned NOTHING must not exit 0, which is the state the
    #: 2026-09-01 session shipped in without noticing.
    force_tile_ledger: FT.ForceTileLedger = field(
        default_factory=FT.ForceTileLedger)

    # THE INSTRUMENT. `timing.time_kernel` for an eager cell and
    # `time_kernel_graph` for a captured one, and both write `TIMING_BASIS` into
    # the row. Injectable so the driver's control flow can be verified without
    # CUDA; a fake must return a `timing.KernelTiming`.
    timer: Callable = T.time_kernel
    graph_timer: Callable = time_kernel_graph

    # THE RETIRED INSTRUMENT, AND WHY IT IS STILL REACHABLE. Setting either of
    # these puts that mode back on `time_eager`/`time_graph`: a COUNT of warmup
    # calls, an iteration count sized from one isolated call on an idle GPU, and
    # two idle-instant clock samples around the cell. It is None by default, so a
    # sweep measures on the instrument and nothing has to remember to ask.
    #
    # It is kept, rather than deleted, for the control-flow tests that were
    # written against `TimingResult` and for a session that has to reproduce an
    # old row bit for bit. That is only safe because a row is STAMPED with what
    # measured it: a legacy-seam row carries `schema.LEGACY_INSTRUMENT` in its
    # `instrument` column, so nothing it writes can be pooled with an instrument
    # row by accident, and `scripts/alpha_refit.py` refuses to fit the two
    # together without being told to.
    timer_eager: Callable | None = None
    timer_graph: Callable | None = None
    #: Idle-instant clock sampler. The retired path's, and used only there.
    clock_sampler: Callable = T.ClockState.sample

    #: How `reference_clock_mhz` is resolved when the caller did not set it.
    #: Injectable for the same reason the timers are: the resolution reads the
    #: attached device and a file beside the code, and both of the branches
    #: that matter -- a card with a calibration, and a card without one -- have
    #: to be plantable on a laptop or neither is ever tested before the rental.
    reference_clock_resolver: Callable[[], RF.ReferenceClock] = RF.reference_clock
    #: Resolvers for the OTHER reference families (`roofline.REFERENCE_FAMILIES`
    #: minus the bf16 one), keyed by family. A family absent here is resolved
    #: on first use from the SAME CARD the primary came from, through
    #: `roofline.reference_clock(card, family=...)`; injected per family for
    #: the same reason `reference_clock_resolver` is. The bf16 family is not
    #: accepted here: it is the primary, and two ways to set one reference is
    #: how this project ends up with two references.
    reference_clock_resolvers: dict[str, Callable[[], RF.ReferenceClock]] = field(
        default_factory=dict)
    #: What `reference_for` resolved, by family, so a run resolves each GEMM's
    #: clock once and every cell of that family levels against one number.
    #: Filled lazily: a bf16-only sweep never asks for the fp8 GEMM's clock and
    #: never records it as missing.
    family_references: dict[str, RF.ReferenceClock] = field(default_factory=dict)

    def reference_for(self, dtype: str) -> RF.ReferenceClock:
        """The reference clock a cell of `dtype` is levelled and roofed against.

        WHY PER DTYPE. The calibration measures two GEMMs at two clocks (the
        committed H200 file: bf16 at 1515 MHz, fp8 at 1905), and a row's roof
        is its dtype's GEMM. One reference per run levelled every fp8 cell
        against the bf16 clock, which files a cell at its own roof's clock as
        LEVEL-failed HIGH and, once an under-load fp8 reference exists, would
        rescale the fp8 roof by 1905/1515 on top of a peak already measured
        at 1905. The family is `roofline.reference_family(dtype)`.

        The bf16 family IS the run's primary, returned from the fields
        `__post_init__` resolved, so the two cannot disagree. Any other
        family is resolved here on first use, checked against `hardware` the
        way the primary is (the clock and the roof must come from one file),
        cached in `family_references`, and recorded in `missing` under
        `reference_clock_mhz[<family>]` when it has no number. A resolver that
        hands back a clock of the WRONG family is treated as no clock, with
        the reason, rather than trusted: the whole point of the family is
        that the number belongs to a particular GEMM.

        NEVER RAISES, for the reason `__post_init__` gives: it is reached from
        inside `run_sweep`, where a missing reference is a `ReferenceClockRefused`
        (REFUSED, exit 2) and not a traceback (ERROR, exit 4).
        """
        family = RF.reference_family(dtype)
        if family == RF.BF16_FAMILY:
            return RF.ReferenceClock(
                self.reference_clock_mhz, self.reference_clock_source,
                card=self.reference_clock_card, grade=self.reference_clock_grade,
                family=family)
        if family in self.family_references:
            return self.family_references[family]
        resolver = self.reference_clock_resolvers.get(family)
        if resolver is None:
            ref = RF.reference_clock(self.reference_clock_card or None,
                                     family=family)
        else:
            ref = resolver()
        if ref.family != family:
            ref = replace(
                ref, mhz=None, grade="",
                source=(f"the resolver for the {family} family returned a "
                        f"{ref.family!r}-family clock ({ref.source}); a clock "
                        "that belongs to another GEMM is not this family's "
                        "reference"))
        if (ref.mhz is not None and self.hardware is not None
                and ref.profile and ref.profile != self.hardware.name):
            ref = replace(
                ref, mhz=None, grade="",
                source=(f"{ref.profile} records {ref.mhz:.0f} MHz for its "
                        f"{family} GEMM, but the roof this run is scored "
                        f"against is {self.hardware.name!r}, which is a "
                        "different profile. LEVEL asks whether the card sat "
                        "at the clock THE ROOF was measured at, and a "
                        "datasheet roof was never measured at any clock"))
        self.family_references[family] = ref
        if ref.mhz is None:
            self.missing[_missing_key(family)] = ref.source
        return ref

    def __post_init__(self) -> None:
        """Resolve the reference clock, or record in `missing` why there is none.

        RESOLVED HERE, once per run, rather than per cell: the answer is a
        property of the box and the calibration on it, neither of which changes
        between cells, and a per-cell lookup would read a yaml off disk for
        every cell of a metered sweep. Recorded rather than raised, because
        `RunConfig` is constructed in `cli._main` OUTSIDE the `except
        TimingRefused` that makes a refusal exit REFUSED(2); raising here would
        reach `cli.main`'s catch-all and exit ERROR(4), which is the same defect
        as commit 366b4de, one file over. `refuse_unreferenced_clock` does the
        raising, from inside `run_sweep`, where the handler is.

        AN EXPLICIT NUMBER WINS AND IS SAID TO HAVE WON. A non-positive one is
        not a number: `timing.clock_flags` treats `reference_mhz <= 0` as no
        reference at all, so passing 0 would have bought exactly the silent
        undetermined column this whole change exists to end. It is dropped to
        None here with the reason, and the refusal downstream then names it.

        AND THAT DROP HAS TO ASK WHO IS ATTACHED, which is what the first cut
        of this method did not do. `unreferenced_clock` turns on TWO fields, a
        None clock AND a non-empty card, so a branch that set one and left the
        other at "" built the wall at one of the two ways in: `RunConfig(
        reference_clock_mhz=0)` on a pod dropped the number, recorded the
        reason, refused NOTHING, and wrote a full sweep of
        `clock_level_ok = undetermined` rows -- the exact state this method
        exists to end, reached through the argument documented as the way out
        of it. The resolver is therefore consulted on every path that ends with
        no clock, for the card and not for a number to stand in, and
        `reference_clock_card` is assigned in exactly ONE place below.

        THE CLOCK AND THE ROOF MUST COME FROM ONE FILE. `hardware` and this
        number are two readings of the same calibration. If `hardware` was
        supplied and is not the profile the clock came out of -- a datasheet
        roof, or another box's file -- then "the clock the roof was measured at"
        is not what was resolved, and LEVEL would compare this card's clock
        against a number belonging to something else. That is dropped too, with
        both names in the reason.
        """
        if RF.BF16_FAMILY in self.reference_clock_resolvers:
            # Recorded, not raised, like every other refusal this method makes
            # (see above): the run refuses at its first cell with the reason.
            self.reference_clock_resolvers = {
                k: v for k, v in self.reference_clock_resolvers.items()
                if k != RF.BF16_FAMILY}
            self.missing["reference_clock_resolvers"] = (
                f"a resolver for the {RF.BF16_FAMILY!r} family was passed in "
                "reference_clock_resolvers; the bf16 GEMM's clock is the "
                "primary and is set by reference_clock_resolver or "
                "reference_clock_mhz, so the entry was dropped rather than "
                "read as a second primary")
        supplied = self.reference_clock_mhz
        if supplied is not None and supplied > 0:
            self.reference_clock_source = (
                f"{supplied:.0f} MHz, given by the caller rather than read "
                "from this card's calibration")
            # A caller's number has no grade the code can vouch for, so the
            # per-row roof is not rescaled against it; the LEVEL flag is.
            self.reference_clock_grade = "caller"
            return

        # The one assignment of the card, on the one path that can end with no
        # clock. Anything that returns above this line has a clock, so the
        # refusal never reads the field.
        ref = self.reference_clock_resolver()
        self.reference_clock_card = ref.card
        if supplied is not None:
            self.reference_clock_mhz = None
            self.reference_clock_source = (
                f"the caller passed reference_clock_mhz={supplied!r}, which is "
                "not a clock; `timing.clock_flags` reads any value <= 0 as no "
                "reference and would have left LEVEL undetermined on every row "
                "without saying so. The calibration was consulted for the "
                "CARD, so this refuses like any other missing reference, and "
                "not for a clock to quietly stand in for the one asked for")
            self.missing["reference_clock_mhz"] = self.reference_clock_source
            return

        self.reference_clock_mhz = ref.mhz
        self.reference_clock_source = ref.source
        self.reference_clock_grade = ref.grade if ref.mhz is not None else ""
        if (ref.mhz is not None and self.hardware is not None
                and ref.profile and ref.profile != self.hardware.name):
            self.reference_clock_source = (
                f"{ref.profile} records {ref.mhz:.0f} MHz, but the roof this "
                f"run is scored against is {self.hardware.name!r}, which is a "
                "different profile. LEVEL asks whether the card sat at the "
                "clock THE ROOF was measured at, and a datasheet roof was "
                "never measured at any clock")
            self.reference_clock_mhz = None
            self.reference_clock_grade = ""
        if self.reference_clock_mhz is None:
            self.missing["reference_clock_mhz"] = self.reference_clock_source

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"run_{self.run_id}_{self.env_name}.csv"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / f"run_{self.run_id}_{self.env_name}.manifest.jsonl"


@dataclass
class CorrectnessResult:
    passed: bool
    max_abs_err: float
    rel_err: float          # scale-free: max|got-ref| / max|ref|
    tol_rel_max: float
    calibrated: bool = False


def compare(got, golden, tol) -> CorrectnessResult:
    max_abs = float((got.float() - golden.float()).abs().max())
    rel = relative_error(got, golden)
    return CorrectnessResult(tol.passes(rel), max_abs, rel, tol.rel_max,
                             tol.calibrated)


@torch.no_grad()
def check_correctness(spec: BenchSpec, pipe: Pipeline, x, weights, forced,
                      cfg: RunConfig):
    """Run the tiling once and compare against golden fp32.

    Returns the populated state and the golden output as well. The state becomes
    the prologue for span-scoped timing, so the span under study is timed
    against real inputs produced by the real upstream stages; the golden output
    is reused to re-validate a CUDA-graph replay, and the tolerance is returned
    so callers do not rebuild it.
    """
    tol = tolerance(spec, cfg.calibration)
    st = MoEState(spec=spec, weights=weights, x=x)
    st.forced_topk_ids = forced
    pipe.run(st, validate_shapes=cfg.validate_shapes)
    golden = golden_forward(spec, weights, x, forced_topk_ids=forced)
    return compare(st.y, golden, tol), st, golden, tol


#: Method a span may define to report the tile configuration it actually runs.
#: Optional: a span without one contributes nothing and its rows say
#: "unrecorded", which is the honest answer and not a gap to be filled in.
TILE_OBSERVER = "observe_tile_config"

#: What a row says when nothing observed its tile. An explicit source rather
#: than an empty column: "unrecorded" is a claim about the row, "" is a column
#: nobody got round to.
_TILE_UNOBSERVED = {"tile_config_source": "unrecorded"}


def _validated_tile_meta(meta: dict, where: str) -> dict:
    """Reject a tile record that would vanish silently on its way to the CSV.

    `_apply_meta` drops any key that is not a column, so a typo'd `tile_blockm`
    is written nowhere and reads back as unrecorded -- an observer that appears
    to work and records nothing, which is precisely the failure the tile columns
    were added to end. Same reason TERMINAL_STATUSES is a closed set: a value no
    consumer matches does not fail loudly, it just disappears from every
    group-by.

    Restricted to the v4 provenance columns as well. An observer is not a second
    route into the timing or load columns, and a span that could set `ms_p50`
    from outside the timer is a hole in the harness's headline invariant.
    """
    if not meta:
        return {}
    allowed = set(SC.COLUMNS_ADDED_IN[4])
    stray = sorted(set(meta) - allowed)
    if stray:
        raise ValueError(
            f"{where}.{TILE_OBSERVER} returned {stray}: an observer may only "
            f"set the v4 tile provenance columns {sorted(allowed)}. A key "
            f"outside that set is dropped by _apply_meta and the row reads back "
            f"as unrecorded, which is the failure this whole column set exists "
            f"to end.")
    source = meta.get("tile_config_source")
    if source not in SC.TILE_SOURCES:
        raise ValueError(
            f"{where}.{TILE_OBSERVER} reported tile_config_source={source!r}; "
            f"legal values are {sorted(SC.TILE_SOURCES)}")
    return dict(meta)


def observe_tile_config(pipe: Pipeline, span, st: MoEState) -> dict:
    """Ask the implementation which tile it runs, outside every timed region.

    UNTIMED BY CONSTRUCTION, which is the point of it living here. run_cell
    calls this from the prologue block that already runs the correctness check
    and the fp32 oracle, above the line where the timed `call` is even defined,
    so an observation cannot drift inside a measured interval in a later edit.
    An observer that needs a real call (vLLM's does: it rebinds
    try_get_optimal_moe_config for the duration of one invocation) therefore
    costs one extra call in a prologue that already runs a python-loop fp32
    reference, and nothing in the timed loop.

    For a whole-layer row the target is the pipeline, so every span is asked and
    the first that answers wins. In practice exactly one span in a tiling has an
    observer -- the framework kernel -- and the reference stages around it have
    no tile to report.

    Never raises. A broken observer must not cost a metered cell: the failure is
    printed and the row records "unrecorded", which is true.
    """
    targets = [span] if span is not None else list(pipe.spans)
    for target in targets:
        hook = getattr(target, TILE_OBSERVER, None)
        if hook is None:
            continue
        try:
            meta = _validated_tile_meta(hook(st), target.name)
        except Exception as e:  # noqa: BLE001
            # Deliberately broad. An observer reaches into a framework's
            # internals by design, so it can fail in ways no narrow clause
            # predicts -- a moved import path, a changed signature, a vLLM
            # version that returns something other than a dict.
            print(f"[warn] {target.name}: tile observation failed: "
                  f"{describe_exception(e)}")
            return dict(_TILE_UNOBSERVED)
        if meta:
            return meta
    return dict(_TILE_UNOBSERVED)


def _downstream_of(pipe: Pipeline, span):
    """Spans that run after `span`, needed to turn its output back into a layer
    output that can be compared against golden."""
    if span is None:
        return []
    return list(pipe.spans[pipe.spans.index(span) + 1:])


def _expert_counts(st: MoEState, spec: BenchSpec, forced) -> tuple[list[int], str]:
    """Per-expert row counts, from the most independent source available.

    Precedence matters and is not obvious. The forced routing decision is the
    experimental INPUT; `topk_ids` and `expert_offsets` in state are OUTPUTS of
    whatever implementation is under test. Reading state first meant a kernel
    covering `router` or `permute` derived its own load metrics from its own
    bug, and those metrics feed active_experts -> weight bytes ->
    compulsory_bytes -> arithmetic intensity. A benchmark axis must never be
    computed from the thing being measured while the ground truth is in hand.

    A span covering all six stages (the vLLM/SGLang fused_moe shape) never
    materialises `expert_offsets`, so no source is guaranteed and the chain has
    to end in something. It ends loudly: zeros are reported with a source of
    "unknown" so the row says the load is not known rather than saying it is 0.
    """
    E = spec.model.num_experts
    if forced is not None:
        return expert_counts(forced.cpu(), E).tolist(), "forced routing"
    if st.topk_ids is not None:
        return expert_counts(st.topk_ids.cpu(), E).tolist(), "topk_ids"
    if st.expert_offsets is not None:
        return counts_from_offsets(st.expert_offsets.cpu()), "expert_offsets"
    return [0] * E, "unknown"


def _resolve_target(pipe: Pipeline, impl: str):
    if is_pipeline_scope(impl):
        return None
    for s in pipe.spans:
        if s.name == impl:
            return s
    raise PipelineError(
        f"impl {impl!r} is not part of pipeline {pipe.label}; "
        f"available: {[s.name for s in pipe.spans]}"
    )


#: Columns _base_row sets from its own arguments. Anything the machine-info
#: dict happens to carry under these names is dropped rather than colliding:
#: passing both raises "got multiple values for keyword argument", which
#: run_sweep catches as a crash and turns into a sweep that writes zero rows.
_ROW_OWNED_BY_CALLER = frozenset({
    "run_id", "timestamp", "git_sha", "git_dirty", "env_name", "model",
    "hidden_size", "intermediate_size", "num_experts", "top_k", "num_tokens",
    "rows", "dtype", "routing_kind", "routing_param", "trace_id", "seed",
    "pipeline", "impl", "scope", "covers", "cuda_graph_safe", "input_init",
    "input_scale",
})


def _base_row(spec: BenchSpec, pipe: Pipeline, impl: str, span, cfg: RunConfig,
              info: dict, sha: str, dirty: bool) -> SC.Row:
    m = spec.model
    machine = {k: v for k, v in info.items()
               if k in SC.COLUMNS and k not in _ROW_OWNED_BY_CALLER}
    return SC.Row(
        run_id=cfg.run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=sha,
        git_dirty=dirty,
        env_name=cfg.env_name,
        model=m.name,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_experts=m.num_experts,
        top_k=m.top_k,
        num_tokens=spec.num_tokens,
        rows=spec.rows,
        dtype=spec.dtype,
        routing_kind=spec.routing.kind,
        routing_param=spec.routing.param,
        trace_id=spec.routing.trace_id or "",
        seed=spec.seed,
        pipeline=pipe.label,
        impl=impl,
        scope="pipeline" if span is None else "span",
        covers="+".join(span.covers) if span is not None else "all",
        cuda_graph_safe=(pipe.cuda_graph_safe if span is None
                         else span.cuda_graph_safe),
        input_init="fan_in",
        input_scale=cfg.input_scale,
        **machine,
    )


def _apply_correctness(row: SC.Row, c: CorrectnessResult) -> None:
    row.correctness_passed = c.passed
    row.max_abs_err = c.max_abs_err
    row.rel_err = c.rel_err
    row.tol_rel_max = c.tol_rel_max
    row.tol_calibrated = c.calibrated


def _apply_cost(row: SC.Row, cost, ms: float | None,
                cfg: RunConfig | None = None) -> None:
    """Cost, the FIXED-roof efficiency, and the roof AT THE CLOCK THE CELL RAN.

    THE PER-ROW ROOF HAS TWO WRITERS AND ONE RULE. This function on the pod
    and `recompute.ceiling_columns` off it both write
    `roof_at_cell_clock_tflops` / `pct_of_roof_at_cell_clock` / `roof_note`,
    and both do it through `roofline.cell_clock_roof` with the reference of
    the ROW'S DTYPE FAMILY (`cfg.reference_for(row.dtype)`: an fp8 row is
    roofed at the fp8 GEMM's clock, everything else at the bf16 GEMM's). The
    recompute used to carry no rule and left the per-row roof at the old
    peak's value beside a rewritten fixed roof; the driver used to hand every
    dtype the bf16 clock.

    Two compute-side fractions are written and they are not the same number.
    `pct_of_achieved_tflops` is against `achieved_peak_tflops`, the
    calibration's roof at the calibration's clock, identical on every row of
    the run; it keeps its v2 meaning and its comment in `schema.Row` states
    the bias it carries. `pct_of_roof_at_cell_clock` is against
    `roof_at_cell_clock_tflops`, `roofline.roof_at_clock` applied to THIS row's
    `sm_clock_load_mhz` and the run's under-load reference. On a memory-shaped
    H200 cell boosted to 1980 MHz against a 1515 roof the two differ by 1.31x,
    and the one-sided LEVEL flag used to pass that cell. Read the second; the
    first is there to compare with pre-v6 rows.

    ORDER MATTERS AND IS RELIED ON: `_apply_kernel_timing` has already put
    `sm_clock_load_mhz` on the row when this runs (`_run_modes` applies the
    timing before the cost), so the per-row roof reads it off the row rather
    than being handed it, and a caller that applied cost first would find 0.0
    there and get the refusal named for it. The bandwidth roof is NOT
    rescaled: HBM does not run on the SM clock (1.7% measured sensitivity),
    and the ridge `hw.bound` classifies against therefore stays the fixed
    one, which UNDER-classifies memory-bound cells on a boosted card (the
    effective ridge is higher), so `implied_traffic_ratio` is omitted on some
    cells that earned it rather than written on some that did not.
    """
    row.flops = cost.flops
    row.compulsory_bytes = cost.bytes_total
    row.arith_intensity_compulsory = cost.arithmetic_intensity
    if not ms:
        return
    row.tflops = cost.tflops(ms)
    row.compulsory_gbps = cost.gbps(ms)
    if cfg is None:
        return

    hw = cfg.hardware
    if hw is None:
        return
    row.achieved_bw_gbps = hw.bandwidth_bytes_s / 1e9
    row.bw_ceiling_pattern = hw.ceiling_pattern
    try:
        peak = hw.peak(row.dtype)
    except ValueError:
        peak = 0.0
    if peak:
        row.achieved_peak_tflops = peak / 1e12
        row.pct_of_achieved_tflops = 100.0 * row.tflops / row.achieved_peak_tflops
        ref = cfg.reference_for(row.dtype)
        roof, why = RF.cell_clock_roof(row.achieved_peak_tflops,
                                       row.sm_clock_load_mhz, ref.mhz, ref.grade)
        row.roof_at_cell_clock_tflops = roof
        row.pct_of_roof_at_cell_clock = (100.0 * row.tflops / roof) if roof else 0.0
        row.roof_note = why
    else:
        row.roof_note = f"no measured compute ceiling for dtype {row.dtype!r}"

    # Only sound when the cell is genuinely memory bound. Compulsory intensity
    # is an UPPER bound on true intensity, so compulsory < ridge implies true <
    # ridge and the classification is conservative. Hardware owns that call.
    if peak and hw.bound(row.dtype, cost.arithmetic_intensity) == "memory":
        from .calibrate import implied_traffic_ratio
        row.implied_traffic_ratio = implied_traffic_ratio(
            cost.bytes_total, ms, hw.bandwidth_bytes_s)


def _instrument_kwargs(cfg: RunConfig, l2_flush: bool, dtype: str) -> dict:
    """What `time_kernel` is called with for one mode of one cell.

    THE REFERENCE IS THE CELL'S DTYPE FAMILY'S (`cfg.reference_for`), not the
    run's one number: the instrument scores LEVEL against what it is handed,
    and handing an fp8 cell the bf16 GEMM's clock made it fail HIGH at its own
    roof's clock. `_apply_kernel_timing` writes the same reference onto the
    row, from the same call, so the verdict and the number it was scored
    against cannot come from two places.

    The flusher is BUILT HERE rather than left to `time_kernel`'s own default,
    so `flush_mb` and `flush_mode` stay the run's knobs and stay recorded: the
    read-versus-write choice is load-bearing (a write flush leaves an L2 of
    dirty lines whose writebacks land inside the NEXT timed interval, 10-30% on
    a sub-100-microsecond span) and a column that says "read" while the default
    sized itself elsewhere would be a lie in the cheapest possible place.

    WHAT IT DELIBERATELY DOES NOT PASS is `cfg.warmup` and `cfg.iters`, which
    `time_kernel` has no parameters for. They are not dropped quietly:
    `refuse_dropped_retired_knobs` stops the cell before it is measured if
    either was set while this path is in force, so no sweep reaches here
    believing a call count was honoured.
    """
    kw = dict(warmup_ms=cfg.warmup_ms, target_ms=cfg.target_ms,
              trials=cfg.trials, l2_flush=l2_flush,
              reference_clock_mhz=cfg.reference_for(dtype).mhz)
    if l2_flush:
        kw["flusher"] = T.L2Flusher(cfg.flush_mb, device=cfg.device,
                                    mode=cfg.flush_mode)
    return kw


def _apply_kernel_timing(row: SC.Row, kt, cfg: RunConfig) -> None:
    """One `KernelTiming` onto one row, including what the instrument was.

    THE FIVE RETIRED QUANTITIES ARE LEFT AT THEIR DEFAULTS: `sm_clock_start_mhz`,
    `sm_clock_end_mhz`, `clock_drift_pct` and the two temperatures. `time_kernel`
    does read a first and a last sample, but UNDER LOAD, and those columns hold
    IDLE-instant readings; writing under-load numbers into them would give one
    column two meanings either side of the version boundary. The under-load
    numbers have columns of their own and a consumer asks for them by name.

    `throttled` IS WRITTEN, and that is the correction this docstring used to
    argue against. It is not a quantity, it is the VERDICT "this row's clock
    misbehaved, do not pool it", and four consumers read it as one:
    `scripts/pod_session.sh` gate S6d, `scripts/run_all.sh`,
    `scripts/publish_results.sh` and `scripts/efficiency_report.py`. Leaving it
    False on every v5 row did not make those
    checks conservative, it made them vacuous: S6d "thermal stability" compared
    0.0% against "< 5%" and could no longer FAIL for any reason, on any card, at
    any temperature. A check that examined nothing reporting zero failures is
    this project's documented failure shape and the one the instrument exists to
    remove, so the verdict column carries the instrument's answer to the
    question it was always asking.

    FROM THE TWO CLOCK VERDICTS AND NOT THE THIRD. `host_bound_ok` is a fact
    about the CALLER (the host could not enqueue fast enough to keep the queue
    deep), not about the card's clock, and a row that is host-bound is an upper
    bound rather than a thermal event. Folding it in here would report a Python
    launcher as a hot box.

    AND FROM THE LOW SIDE OF LEVEL ONLY. LEVEL is two-sided since 2026-09-03
    and a cell can fail it HIGH: a memory-shaped cell that boosted to 1980 MHz
    against a compute roof measured at 1515. That is the mirror image of a
    throttle, not a throttle; on an H200 it is the NORMAL state of decode
    work, and writing it into `throttled` would fail S6d on every honest
    decode session and empty `efficiency_report` of the cells the study is
    about. The HIGH failure is on the row as `clock_level_ok = failed` with
    `clock_level_side = high`, its fixed-roof fraction is the thing that is
    wrong, and `roof_at_cell_clock_tflops` is the correction. THE RULE FOR
    EVERY CONSUMER: LOW or DRIFT excludes (that is exactly `throttled`); HIGH
    is not an exclusion, it means "the fixed-roof fraction is not comparable,
    read pct_of_roof_at_cell_clock". A consumer that excludes on
    `clock_level_ok == failed` alone drops boosted cells; it must read the
    side or branch on `throttled`.

    THE REFERENCE IS WRITTEN ONTO THE ROW, from the config and not from the
    record: `cfg.reference_for(row.dtype)` is what `_instrument_kwargs` handed
    the instrument for this cell, so the two are one number on the real path,
    and the config also carries the SOURCE, which the record does not. A row
    that says `clock_level_ok = ok` and does not say against what cannot be
    re-derived once the yaml it came from has been overwritten by a
    recalibration. Per dtype family: an fp8 row names the fp8 GEMM's clock.

    "undetermined" IS NOT THROTTLED, which is the one place this departs from
    "unknown counts against the gate". These consumers are inclusion filters
    over a whole arm, not release gates over a claim: on a pod whose container
    forbids NVML every row is undetermined, and calling all of them throttled
    would empty `efficiency_report` and fail S6d for the whole session on the
    strength of a missing library. `alpha_refit.clock_gate` keeps undetermined
    rows for the same reason and states it. What the two verdicts DO give back
    is a gate that can fail: one FAILED clock verdict marks the row.
    """
    row.instrument = kt.instrument
    row.warmup = kt.warmup_calls
    row.warmup_ms = kt.warmup_ms
    row.iters, row.trials = kt.iters, kt.trials
    row.l2_flush = kt.l2_flush
    row.flush_mb = kt.flush_mb
    row.ms_p50, row.ms_p90 = kt.ms_p50, kt.ms_p90
    row.ms_min, row.ms_std = kt.ms_min, kt.ms_std
    row.jitter_p90_over_p50 = (kt.ms_p90 / kt.ms_p50) if kt.ms_p50 > 0 else 0.0
    row.flush_mode = cfg.flush_mode if kt.l2_flush else ""
    row.sm_clock_load_mhz = kt.sm_clock_load_mhz or 0.0
    row.clock_level_ok = SC.verdict_word(kt.clock_level_ok)
    row.clock_drift_ok = SC.verdict_word(kt.clock_drift_ok)
    # The side is derived from the record's clocks by the instrument's own
    # rule, so a record built by a fake that set the verdict without the side
    # (every fake in tests/ before v6) still gets the right one; the
    # instrument's own field is preferred when it carries one.
    ref = cfg.reference_for(row.dtype)
    side = kt.clock_level_side or (
        T.level_side(kt.sm_clock_load_mhz, ref.mhz) or "")
    if row.clock_level_ok != SC.VERDICT_FAILED:
        side = ""
    row.clock_level_side = side
    # LOW or DRIFT -> throttled; HIGH -> not throttled, per-row roof instead.
    row.throttled = (row.clock_drift_ok == SC.VERDICT_FAILED
                     or (row.clock_level_ok == SC.VERDICT_FAILED
                         and side != T.LEVEL_HIGH))
    row.host_bound_ok = SC.verdict_word(
        None if kt.host_bound is None else not kt.host_bound)
    row.clock_samples = kt.clock_samples
    row.clock_source = kt.clock_source
    row.clock_note = kt.clock_note
    row.host_enqueue_ms = kt.host_enqueue_ms or 0.0
    row.host_backlog_iters = kt.host_backlog_iters or 0.0
    row.host_note = kt.host_note
    row.reference_clock_mhz = ref.mhz or 0.0
    row.reference_clock_source = ref.source


def _apply_legacy_timing(row: SC.Row, res, start, end) -> None:
    """One `TimingResult` plus its two idle-instant clock samples onto one row.

    Only reached when a legacy timer is injected. The row is stamped
    `schema.LEGACY_INSTRUMENT` so it can never be mistaken for, or pooled with,
    a row off the instrument -- which is the entire reason the seam is allowed to
    survive at all.
    """
    row.instrument = SC.LEGACY_INSTRUMENT
    row.warmup, row.iters, row.trials = res.warmup, res.iters, res.trials
    row.flush_mb, row.flush_mode = res.flush_mb, res.flush_mode
    row.ms_p50, row.ms_p90 = res.ms_p50, res.ms_p90
    row.ms_min, row.ms_std = res.ms_min, res.ms_std
    row.jitter_p90_over_p50 = res.jitter_p90_over_p50
    row.sm_clock_start_mhz = start.sm_clock_mhz
    row.sm_clock_end_mhz = end.sm_clock_mhz
    row.temp_start_c, row.temp_end_c = start.temp_c, end.temp_c
    row.clock_drift_pct, row.throttled = T.clock_drift(start, end)


#: Timing and derived columns, zeroed whenever a row did not earn them.
_TIMED_FIELDS = ("ms_p50", "ms_p90", "ms_min", "ms_std", "jitter_p90_over_p50",
                 "tflops", "compulsory_gbps", "pct_of_achieved_tflops",
                 "implied_traffic_ratio", "roof_at_cell_clock_tflops",
                 "pct_of_roof_at_cell_clock")


def _apply_load(row: SC.Row, load) -> None:
    for name, value in load.as_row().items():
        setattr(row, name, value)


def _apply_meta(row: SC.Row, meta: dict) -> None:
    for name, value in meta.items():
        if name in SC.COLUMNS:
            setattr(row, name, value)


def _emit(writer: SC.CsvWriter, manifest: SC.Manifest, row: SC.Row, key: str,
          status: str = SC.STATUS_OK, detail: str = "") -> int:
    """The single place a row reaches the CSV.

    Enforces the harness's headline invariant structurally rather than by
    convention: a row that did not pass the oracle leaves with its timing and
    derived columns zeroed, so `correctness_passed == False` implies
    `ms_p50 == 0` in the file itself. Consumer-side filters then become
    redundancy instead of the only defence.
    """
    if not row.correctness_passed:
        for name in _TIMED_FIELDS:
            setattr(row, name, 0.0)
    writer.write(row)
    manifest.record(key, status, detail)
    return 1


def _cell_key(row: SC.Row, pin: FT.CellPin) -> str:
    """`schema.cell_key` plus the forced tile, when one is active.

    THE PROJECT'S FAILURE MODE 2 and not a hypothetical one: a run id that omits
    a swept parameter lets a second setting resume the first's manifest, skip
    every completed cell, and print the first's numbers under the second's
    label. A forced tile is such a parameter the moment it exists, and the
    suffix is empty without one, so every existing manifest still resumes
    byte-for-byte.
    """
    return SC.cell_key(row) + pin.key_suffix


def _mode_rows(make_row: Callable[[], SC.Row], cfg: RunConfig,
               pin: FT.CellPin) -> list[tuple[bool, bool, SC.Row, str]]:
    """One (graph, l2, row, key) per timing mode this cell would measure."""
    keyed = []
    for use_graph in cfg.graph_modes:
        for l2 in cfg.l2_modes:
            row = make_row()
            row.l2_flush, row.cuda_graph = l2, use_graph
            keyed.append((use_graph, l2, row, _cell_key(row, pin)))
    return keyed


def _record_unpinnable(keyed, impl: str, cfg: RunConfig, manifest: SC.Manifest,
                       pin: FT.CellPin) -> int:
    """A cell nothing can pin: recorded, printed, counted, and NOT measured.

    No CSV row, on purpose. Nothing ran, so every column of such a row would be
    a dataclass default -- and a row with `tile_block_m = 0` written during a
    pinned sweep is indistinguishable from the unpinned rows this whole
    mechanism exists to keep out of the file. The manifest carries the fact
    instead, under a status that is deliberately NOT terminal so the same cell
    is measured normally by any later run without the pin.

    Printed once per implementation rather than once per cell: a dense grid
    skips thousands of cells on the same two spans, and a line each would bury
    the summary under its own repetition.
    """
    for _, _, _, key in keyed:
        manifest.record(key, SC.STATUS_FORCE_TILE_UNHONOURABLE, pin.reason)
    if cfg.force_tile_ledger.record_skip(impl, pin.reason):
        print(f"[force-tile] NOT MEASURED: {impl} -- {pin.reason}")
    return 0


@torch.no_grad()
def run_cell(spec: BenchSpec, pipeline_names: Sequence[str], impl: str,
             cfg: RunConfig, routing: RoutingSource,
             writer: SC.CsvWriter, manifest: SC.Manifest,
             info: dict, sha: str, dirty: bool) -> int:
    """Benchmark one (cell, tiling, target) across the configured timing modes.

    With `cfg.force_tile` set, the whole cell -- correctness check, tile
    observation and every timed trial -- runs inside the target span's own
    pinning context, so the tile the fp32 oracle validated is the tile that was
    timed. A cell no span can pin is recorded rather than measured; see
    `moe/bench/force_tile.py` for why that is not merely tidiness.
    """
    pipe = build(pipeline_names, spec=spec)
    span = _resolve_target(pipe, impl)
    if pipe.env not in (cfg.env_name, BASE_ENV):
        raise PipelineError(
            f"pipeline needs environment {pipe.env!r} but this process is "
            f"{cfg.env_name!r}; rows would claim a framework that did not run")

    pin = FT.pin_for(cfg.force_tile, pipe.spans, span)
    make_row = partial(_base_row, spec, pipe, impl, span, cfg, info, sha, dirty)
    # Build every mode key up front. If they are all already done, skip the
    # cell entirely: re-running the fp32 python-loop oracle to produce zero
    # rows is the single most expensive way to resume a sweep.
    keyed = _mode_rows(make_row, cfg, pin)
    if all(key in manifest for _, _, _, key in keyed):
        if pin.status == FT.PINNED:
            # Its rows are in the file already, and under THIS pin's key rather
            # than some other run's, so they count toward the run standing on
            # something -- but separately, because a session that measured
            # nothing and a session with nothing left to measure are different
            # states and only one of them is a failure.
            cfg.force_tile_ledger.record_resumed()
        return 0
    if pin.status == FT.UNHONOURABLE:
        return _record_unpinnable(keyed, impl, cfg, manifest, pin)
    # Entered per cell rather than once around the sweep so a raising cell
    # cannot leave an override installed over every cell after it, which is the
    # same reason `recording_tile_config` restores in a finally.
    with pin.applied():
        return _run_modes(spec, pipe, span, impl, keyed, make_row, cfg, routing,
                          writer, manifest, pin)


@torch.no_grad()
def _run_modes(spec: BenchSpec, pipe: Pipeline, span, impl: str, keyed,
               make_row: Callable[[], SC.Row], cfg: RunConfig,
               routing: RoutingSource, writer: SC.CsvWriter,
               manifest: SC.Manifest, pin: FT.CellPin) -> int:
    """The measured half of `run_cell`, inside whatever pin is in force.

    Split out only so the pin can wrap it without re-indenting the body; the
    control flow is unchanged from before pinning existed, and `pin` is inert
    (`CellPin()` with status "off") on every unpinned sweep.
    """
    refuse_dropped_retired_knobs(cfg)
    refuse_unreferenced_clock(cfg, spec.dtype)
    written = 0
    x, weights = make_inputs(spec, device=cfg.device, scale=cfg.input_scale,
                             reuse_weights=cfg.reuse_weights)
    forced = routing(spec)
    if forced is not None:
        forced = forced.to(x.device)

    correctness, st, golden, tol = check_correctness(spec, pipe, x, weights,
                                                     forced, cfg)
    routing_meta = cfg.routing_info(spec) if cfg.routing_info else {}
    counts, counts_source = _expert_counts(st, spec, forced)
    load = expert_load(counts)

    # In the prologue, beside the correctness check and above the line where the
    # timed callable is defined, so an observation cannot land inside a timed
    # region. AFTER the load columns are taken, because an observer that needs a
    # real call re-runs the span, and a span covering `router` or `permute`
    # would rewrite the very state _expert_counts reads.
    tile_meta = observe_tile_config(pipe, span, st)

    # RULE 3 of moe/bench/force_tile.py, and the reason the S6a gate means
    # anything: the pin is honoured only if the row can SHOW it. `tile_meta` is
    # what the observer read back out of the framework during a real call, so
    # this compares the tile that ran against the tile that was asked for, and
    # refuses the cell when they differ or when nothing was observed at all.
    # The cell is not measured, because a row carrying "vllm_override" that ran
    # something else is worse than the unpinned sweep this replaced.
    disagreement = pin.disagrees_with(tile_meta)
    if disagreement:
        for _, _, _, key in keyed:
            manifest.record(key, SC.STATUS_FORCE_TILE_NOT_OBSERVED, disagreement)
        if cfg.force_tile_ledger.record_unobserved(impl, disagreement):
            print(f"[force-tile] NOT HONOURED: {disagreement}")
        return 0

    # Cost is scoped to what the timer wraps, and materialisation is taken from
    # THIS tiling rather than from the span in isolation.
    if span is None:
        costed_spans, materialised = list(pipe.spans), list(pipe.materialised)
    else:
        costed_spans, materialised = [span], [pipe.materialised_for(span)]
    cost = BM.pipeline_cost(costed_spans, spec, load.active_experts, materialised)

    def prepare(row: SC.Row, verdict=None) -> SC.Row:
        """Everything every row carries, regardless of which path emitted it.

        INCLUDING THE INSTRUMENT, and that is not bookkeeping. Four of this
        function's callers emit a row that no timer ever saw -- a cell that
        failed the fp32 oracle, a graph mode skipped by cost policy, a span
        that could not be captured, and a timer that raised -- and a `Row`
        starts life with `instrument = ""`, which `schema.instrument_of`
        refuses and `schema.has_kernel_timing` therefore raises on. That
        predicate is the one an analysis is told to split a pool with before it
        reads any v5 column, so leaving those four paths blank made the
        documented usage throw on rows a normal sweep writes by the thousand.
        Stamped HERE, above the branch, so no later path can be added that
        forgets: `schema.NO_INSTRUMENT` is what an untimed row says, and the
        two `_apply_*_timing` functions overwrite it when a timer did run.
        """
        row.instrument = SC.NO_INSTRUMENT
        _apply_correctness(row, verdict or correctness)
        _apply_load(row, load)
        _apply_meta(row, routing_meta)
        _apply_meta(row, tile_meta)
        _apply_cost(row, cost, None, cfg)
        if counts_source != "forced routing":
            row.notes = f"expert load derived from {counts_source}"
        return row

    if not correctness.passed:
        # One row, not four: none of the timing modes ran. capture_status says
        # so explicitly, since the l2_flush/cuda_graph columns would otherwise
        # read as a measured mode that happened to produce no numbers.
        row = prepare(make_row())
        row.capture_status = "not_timed"
        row.notes = (f"correctness failed (rel={correctness.rel_err:.3e} > "
                     f"{correctness.tol_rel_max:.3e}); not timed")
        written += _emit(writer, manifest, row, _cell_key(row, pin),
                         SC.STATUS_CORRECTNESS_FAILED,
                         f"rel={correctness.rel_err:.3e}")
        # Re-running would reproduce this verdict exactly, so every mode key
        # for this cell is terminal.
        for _, _, _, key in keyed:
            manifest.record(key, SC.STATUS_CORRECTNESS_FAILED,
                            f"rel={correctness.rel_err:.3e}")
        return written

    downstream = _downstream_of(pipe, span)

    if span is None:
        # Built once, outside the timed region: allocating a MoEState per
        # iteration would measure python bookkeeping that is not under study.
        timed_state = MoEState(spec=spec, weights=weights, x=x)
        timed_state.forced_topk_ids = forced

        def call():
            pipe.run(timed_state, validate_shapes=False)
    else:
        # A span-scoped call mutates the prologue state in place.
        timed_state = st

        def call():
            span(st)

    # Re-earn the correctness verdict against the graph's own replayed output.
    # Graph replay reuses fixed, graph-private buffers, so a kernel that leaves
    # a tail tile or an empty-expert group unwritten sees the previous replay's
    # correct values still resident and would otherwise look fine.
    replay_verdict = None

    def verify_replay():
        """Compare the output the REPLAY produced, not the prologue's.

        `timed_state` is the state the timed callable actually writes into, and
        for a whole-layer span that is not the state check_correctness left
        behind. Comparing the wrong one made this a no-op for exactly the
        implementation shape (vLLM/SGLang fused_moe) it most needed to check.
        """
        nonlocal replay_verdict
        if span is not None:
            for later in downstream:
                later(timed_state)
        replay_verdict = (compare(timed_state.y, golden, tol)
                          if timed_state.y is not None else correctness)

    graph_ok, graph_skip_reason = should_time_graph(cost, cfg)

    for use_graph, l2, row, key in keyed:
        if key in manifest:
            continue
        prepare(row)

        if use_graph and not graph_ok:
            row.capture_status = "skipped"
            row.graph_skip_reason = graph_skip_reason
            row.notes = "graph timing skipped by cost policy"
            written += _emit(writer, manifest, row, key, SC.STATUS_OK,
                             "graph skipped by policy")
            continue

        legacy = cfg.timer_graph if use_graph else cfg.timer_eager
        clocks_start = cfg.clock_sampler() if legacy is not None else None
        try:
            if legacy is not None:
                extra = {"on_captured": verify_replay} if use_graph else {}
                res = legacy(call, warmup=cfg.warmup, iters=cfg.iters,
                             trials=cfg.trials, l2_flush=l2,
                             flush_mb=cfg.flush_mb,
                             flush_mode=cfg.flush_mode, **extra)
            elif use_graph:
                res = cfg.graph_timer(call, on_captured=verify_replay,
                                      **_instrument_kwargs(cfg, l2, spec.dtype))
            else:
                res = cfg.timer(call, **_instrument_kwargs(cfg, l2, spec.dtype))
        except T.NotCapturable as e:
            # A finding, not a failure: an implementation that cannot be
            # graph-captured cannot be used in real MoE inference. It belongs in
            # the CSV, not only in a sidecar manifest, or every aggregate over
            # the published data is silently conditioned on capturability.
            row.capture_status = "not_capturable"
            row.notes = f"not CUDA-graph capturable: {str(e)[:160]}"
            written += _emit(writer, manifest, row, key,
                             SC.STATUS_NOT_CAPTURABLE, str(e)[:200])
            continue
        except T.TimingRefused:
            # THE SECOND WAY IN, and it was open. `TimingRefused` subclasses
            # RuntimeError, so the handler below caught every refusal the
            # INSTRUMENT ITSELF raises -- no CUDA and no injected fakes,
            # trials=0, a warmup that makes the measurement meaningless -- and
            # filed it as this one cell's timing error. Each of those is a fact
            # about the run, identical for every cell, so the sweep wrote a
            # zeroed STATUS_ERROR row per cell and exited 0 (DONE), while
            # `cli._main`'s `except TimingRefused` around `run_sweep` sat one
            # frame up and could never see one. The config guards above got
            # their escape hatch when they were written; the instrument's own
            # refusal, the other door into the same room, did not.
            raise
        except RuntimeError as e:
            row.notes = f"timing error: {str(e)[:160]}"
            written += _emit(writer, manifest, row, key, SC.STATUS_ERROR,
                             str(e)[:200])
            continue

        verdict = replay_verdict if (use_graph and replay_verdict) else correctness
        _apply_correctness(row, verdict)
        row.capture_status = "captured" if use_graph else "n/a"
        if legacy is not None:
            _apply_legacy_timing(row, res, clocks_start, cfg.clock_sampler())
        else:
            _apply_kernel_timing(row, res, cfg)
        _apply_cost(row, cost, res.ms_p50, cfg)

        if not verdict.passed:
            # _emit zeroes the timing columns, so the file never carries a
            # measurement that failed the oracle.
            row.notes = (f"REPLAYED output failed the oracle "
                         f"(rel={verdict.rel_err:.3e}); timing discarded")
            written += _emit(writer, manifest, row, key,
                             SC.STATUS_CORRECTNESS_FAILED, "graph replay")
            continue

        written += _emit(writer, manifest, row, key)

    if written and pin.status == FT.PINNED:
        # Counted where the rows were written, not where the pin was planned:
        # the question the CLI refuses on is whether anything is IN THE FILE
        # under this tile, and a plan is not a file.
        cfg.force_tile_ledger.record_pinned()
    return written


def describe_exception(e: BaseException, limit: int = 300) -> str:
    """A one-line, never-empty rendering of an exception.

    `f"{e}"` is the empty string for any exception carrying no args, and vLLM
    validates its fused_moe arguments with bare `assert` statements. An fp8
    sweep printed 147 lines ending in a colon because of it, on a machine that
    bills by the second.

    One line because a sweep prints one per failing cell, and bounded because a
    shape mismatch can carry a very long repr.
    """
    kind = type(e).__name__
    msg = " ".join(str(e).split())
    out = f"{kind}: {msg}" if msg else kind
    return out if len(out) <= limit else out[: limit - 3] + "..."


def run_sweep(cells: Iterable[tuple[BenchSpec, Sequence[str], str]],
              cfg: RunConfig, routing: RoutingSource,
              info: dict | None = None) -> Path:
    """cells: (spec, pipeline span names, name of the impl under study)."""
    info = info if info is not None else T.runtime_info()
    sha, dirty = SC.git_provenance()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with SC.CsvWriter(cfg.csv_path) as writer:
        manifest = SC.Manifest(cfg.manifest_path)
        try:
            for spec, names, impl in cells:
                try:
                    total += run_cell(spec, names, impl, cfg, routing, writer,
                                      manifest, info, sha, dirty)
                except T.TimingRefused:
                    # NOT a per-cell crash. A refusal is a fact about the RUN --
                    # a config that cannot be honoured, a card with no ruler, an
                    # instrument with no CUDA and no fakes -- and it is the same
                    # fact for every cell, so recording it as one cell's failure
                    # would repeat it for all of them and still exit 0.
                    #
                    # THE BASE CLASS AND NOT A LIST OF SUBCLASSES. This named
                    # `RetiredKnobRefused` alone, so the refusal added next
                    # would have had to remember to add itself here; a guard
                    # that names its members is the defect it is checking for.
                    # `cli._main` catches the same base around `run_sweep` and
                    # exits REFUSED(2), so everything that reaches here exits
                    # the way it says it does.
                    raise
                except PipelineError as e:
                    manifest.record(f"invalid|{spec.label}|{'+'.join(names)}|{impl}",
                                    SC.STATUS_INVALID_PIPELINE,
                                    describe_exception(e, 200))
                    print(f"[warn] {spec.label} {impl}: {describe_exception(e)}")
                except Exception as e:  # noqa: BLE001
                    manifest.record(f"crash|{spec.label}|{'+'.join(names)}|{impl}",
                                    SC.STATUS_CRASH, traceback.format_exc()[-400:])
                    print(f"[warn] {spec.label} {impl}: {describe_exception(e)}")
        finally:
            manifest.close()
    print(f"[driver] wrote {total} rows -> {cfg.csv_path}")
    for line in cfg.force_tile_ledger.summary(cfg.force_tile):
        print(line)
    return cfg.csv_path
