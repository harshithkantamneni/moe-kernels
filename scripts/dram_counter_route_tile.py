#!/usr/bin/env python
"""The counter route at a tile's num_warps, with scripts/dram_counter_route.py byte-identical (it is
pinned at rental 7's sha 130d5dc...; the tile stage, 2026-10-14; design-tile-review 6).

    python scripts/dram_counter_route_tile.py --num-warps 4 <dram_counter_route's own arguments>

Three things the pinned route cannot do at a tile, done here in memory and nowhere else:

1. WARPS. The route writes `"num_warps": SWEEP_FIXED["num_warps"]` into every page's design and
   its census, and SWEEP_FIXED is its own copy of the sweep module (`scripts.block_m_crossing_sweep`,
   a different module object from R3's). `r3_tile.set_warps` sets both FIXED dicts, so the page
   records the warps that ran (a w4 page no longer silently records w8).
2. THE CHILD. ncu runs R3's `--counter-child` in a fresh process: the route's `R3_CHILD` is R3's
   file, and `r3_ncu_argv` binds it as a default at import. Here every `r3_ncu_argv` call is handed
   scripts/r3_tile.py, which reads the warps from MOE_TILE_NUM_WARPS (set by set_warps) and runs
   R3's own main.
3. THE RUN ID. The route's run ids (`run_id_for`) have no warps knob, so a w8 and a w4 page at one
   (model, BLOCK_M, BLOCK_N, G) collided (the review's f64w8 / f64w4). At a warps count other than
   R3's 8 every id gains `warps<N>`; at 8 the ids are the route's own, so no published id moves.

BLOCK_M, BLOCK_N, BLOCK_K and num_stages are the route's own flags and pass through unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_tile as RT  # noqa: E402

TILE_CHILD = HERE / "r3_tile.py"


def install(DCR, nw: int | None) -> None:
    """Point every ncu child at r3_tile.py and, at warps other than 8, key the run ids by warps."""
    if getattr(DCR.r3_ncu_argv, "_tile_child", False):
        return
    orig = DCR.r3_ncu_argv

    def r3_ncu_argv(*a, child=None, **kw):
        c = TILE_CHILD
        if child is not None and not Path(child).is_absolute():
            c = TILE_CHILD.relative_to(REPO)
        return orig(*a, child=c, **kw)
    r3_ncu_argv._tile_child = True
    DCR.r3_ncu_argv = r3_ncu_argv
    DCR.R3_CHILD = TILE_CHILD
    if nw is None or nw == RT.R3_WARPS:
        return
    orig_id = DCR.run_id_for

    def run_id_for(mode, args, card):
        pv = DCR.PV
        DCR.PV = RT._WarpsPV(pv, nw)
        try:
            return orig_id(mode, args, card)
        finally:
            DCR.PV = pv
    DCR.run_id_for = run_id_for


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        nw, rest = RT.take_warps(argv)
    except RT.Refused as exc:
        print(f"REFUSE: {exc}")
        return 2
    if nw is not None:
        RT.set_warps(nw)
    import dram_counter_route as DCR
    install(DCR, nw)
    return DCR.main(rest)


if __name__ == "__main__":
    sys.exit(main())
