#!/usr/bin/env python
"""The counter route for the settle rental S1 (docs/registered/2026-10-21-settleS1-gh200): the tile wrapper
(scripts/dram_counter_route_tile.py) plus two instrument choices, with scripts/dram_counter_route.py byte-identical
(pinned at rental 7's sha 130d5dc...). INSTRUMENTATION ONLY: ncu's replay mode, its cache control and the metric
list. The kernel, its config and R3's call are untouched.

    python scripts/dram_counter_route_s1.py [--s1-replay application] [--s1-stall-metrics] \
        <dram_counter_route_tile.py's own arguments>

--s1-replay application   S1-M1 (DIAGNOSIS-A2 Q7 row 1). ncu runs `--replay-mode application --cache-control none`
                          in place of the route's `--replay-mode kernel --cache-control all` (ncu_common_flags): the
                          profiled GEMMs launch in R3's real stream order, w1, the activation kernel and w2 of each
                          call back to back, with no cache flush and no save/restore of device memory before a
                          profiled launch; ncu re-runs the whole child once per metric pass instead of replaying one
                          kernel. The page records what ran: `ncu.replay_mode` "application", `ncu.cache_control`
                          "none", the instrument string, and `s1`; its run id gains `replay=application`. NOT
                          REPRODUCED: ncu still serialises the profiled launches and stops between them to read its
                          counters, so the GEMMs do not run as back to back as in R3's timed burst. The route's
                          VALIDITY gates are read on no histogram page (rental 5), so none misreads the flushless bytes.
--s1-stall-metrics        S1-M5 (DIAGNOSIS-A2 Q7 row 5). Three metrics of the route's floor list (R3_FLOOR_METRICS)
                          ride on the page as RECORDED metrics (cell["recorded"][gemm], parsed SOFT, asked only where
                          the probe proves them, as every recorded metric): long-scoreboard and barrier stall shares
                          per active warp and the tensor pipe's active share. The run id gains `stall=1`.

With neither flag the run is dram_counter_route_tile.py's, unchanged. Both flags may be given.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dram_counter_route_tile as DT  # noqa: E402
import r3_tile as RT  # noqa: E402

#: S1-M5's metrics, each a member of the pinned route's R3_FLOOR_METRICS (so its unit is in NCU_METRIC_UNITS)
S1_STALL_METRICS: tuple[str, ...] = (
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
)
REPLAYS = ("application",)
#: the route's flags (ncu_common_flags("all")) and what S1-M1 passes instead
KERNEL_FLAGS = (("--replay-mode", "kernel"), ("--cache-control", "all"))
APP_FLAGS = (("--replay-mode", "application"), ("--cache-control", "none"))
INSTRUMENT_KERNEL = "replay-mode-kernel/cache-control-all"
INSTRUMENT_APP = "replay-mode-application/cache-control-none"


class Refused(Exception):
    pass


def take_s1(argv: list[str]) -> tuple[str | None, bool, list[str]]:
    """(--s1-replay value or None, --s1-stall-metrics given, argv without either)."""
    rest, replay, stall, i = [], None, False, 0
    while i < len(argv):
        a = argv[i]
        if a == "--s1-replay":
            if i + 1 >= len(argv):
                raise Refused("--s1-replay needs a value")
            replay, i = argv[i + 1], i + 2
            continue
        if a.startswith("--s1-replay="):
            replay, i = a.split("=", 1)[1], i + 1
            continue
        if a == "--s1-stall-metrics":
            stall, i = True, i + 1
            continue
        rest.append(a)
        i += 1
    if replay is not None and replay not in REPLAYS:
        raise Refused(f"--s1-replay {replay!r}: {', '.join(REPLAYS)} (kernel replay is the route's own: leave the flag out)")
    return replay, stall, rest


def app_argv(argv: list[str]) -> list[str]:
    """An ncu argv with the route's kernel replay and cache flush swapped for application replay, no flush. Each
    swapped pair must occur exactly once, so a route that moved its flags refuses instead of running unswapped."""
    out = list(argv)
    for (flag, old), (_f, new) in zip(KERNEL_FLAGS, APP_FLAGS, strict=True):
        at = [i for i in range(len(out) - 1) if out[i] == flag]
        if len(at) != 1 or out[at[0] + 1] != old:
            raise Refused(f"the route's ncu argv holds {flag} {[out[i + 1] for i in at]}, not {old} once: "
                          "S1-M1 swaps exactly that pair")
        out[at[0] + 1] = new
    return out


class _S1PV:
    """provenance with S1's knobs added to every run id (only while the route names its run)."""

    def __init__(self, pv, knobs: dict):
        self._pv, self._knobs = pv, knobs

    def __getattr__(self, k):
        return getattr(self._pv, k)

    def run_id(self, *, card, **swept):
        return self._pv.run_id(card=card, **swept, **self._knobs)


def install(DCR, replay: str | None, stall: bool) -> dict:
    """Patch the route in memory for S1. Returns the record every page carries as `s1`."""
    rec = {"registration": "docs/registered/2026-10-21-settleS1-gh200",
           "replay_mode": replay or "kernel", "cache_control": "none" if replay else "all",
           "stall_metrics": list(S1_STALL_METRICS) if stall else [],
           "wrapper": "scripts/dram_counter_route_s1.py"}
    if getattr(DCR, "_s1_installed", None) is not None:
        return DCR._s1_installed
    knobs = {}
    if replay == "application":
        orig_argv = DCR.r3_ncu_argv

        def r3_ncu_argv(*a, **kw):
            return app_argv(orig_argv(*a, **kw))
        DCR.r3_ncu_argv = r3_ncu_argv
        if INSTRUMENT_KERNEL not in DCR.R3_RUN_INSTRUMENT_AT:
            raise Refused(f"the route's page instrument {DCR.R3_RUN_INSTRUMENT_AT!r} names no {INSTRUMENT_KERNEL}")
        DCR.R3_RUN_INSTRUMENT_AT = DCR.R3_RUN_INSTRUMENT_AT.replace(INSTRUMENT_KERNEL, INSTRUMENT_APP)
        knobs["replay"] = "application"
    if stall:
        missing = [m for m in S1_STALL_METRICS if m not in DCR.R3_FLOOR_METRICS or m not in DCR.NCU_METRIC_UNITS]
        if missing:
            raise Refused(f"{missing}: not in the route's floor list with a unit")
        DCR.R3_RECORDED_METRICS = tuple(DCR.R3_RECORDED_METRICS) + tuple(
            m for m in S1_STALL_METRICS if m not in DCR.R3_RECORDED_METRICS)
        knobs["stall"] = 1
    orig_build = DCR.build_r3_page

    def build_r3_page(**kw):
        if replay:
            kw["ncu"] = dict(kw["ncu"], replay_mode="application", cache_control="none")
        page = orig_build(**kw)
        page["s1"] = dict(rec)
        return page
    DCR.build_r3_page = build_r3_page
    if knobs:
        orig_id = DCR.run_id_for

        def run_id_for(mode, args, card):
            pv = DCR.PV
            DCR.PV = _S1PV(pv, knobs)
            try:
                return orig_id(mode, args, card)
            finally:
                DCR.PV = pv
        DCR.run_id_for = run_id_for
    DCR._s1_installed = rec
    return rec


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        replay, stall, rest = take_s1(argv)
        nw, rest = RT.take_warps(rest)
    except (Refused, RT.Refused) as exc:
        print(f"REFUSE: {exc}")
        return 2
    if nw is not None:
        RT.set_warps(nw)
    import dram_counter_route as DCR
    DT.install(DCR, nw)
    try:
        install(DCR, replay, stall)
    except Refused as exc:
        print(f"REFUSE: {exc}")
        return 2
    return DCR.main(rest)


if __name__ == "__main__":
    sys.exit(main())
