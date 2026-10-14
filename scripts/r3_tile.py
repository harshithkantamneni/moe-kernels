#!/usr/bin/env python
"""R3 at a num_warps of the tile's, with R3 itself byte-identical (the tile stage, 2026-10-14;
design-tile 5.1, design-tile-review 6).

    python scripts/r3_tile.py --num-warps 4 <R3's own arguments>
    python scripts/locked_r3.py ... --r3-script scripts/r3_tile.py -- --num-warps 4 <R3's arguments>

R3 (scripts/private_weight_reference.py) pins every config value but BLOCK_M from
`block_m_crossing_sweep.FIXED` (num_warps 8) and has no warps flag; it reads FIXED when it builds a
config (`pinned_config`), so this wrapper sets FIXED["num_warps"] in memory and hands the rest of
argv to R3's own `main`. It is a config value of vLLM's kernel, nothing else: no tiling or
scheduling logic is written here. `locked_r3.py` already takes `--r3-script` (an existing hidden
flag), so the lock wrapper is unchanged too.

The warps also reach R3's children: the value is exported as MOE_TILE_NUM_WARPS, which a counter
child started under ncu (scripts/dram_counter_route_tile.py points ncu at this file) reads when it
has no --num-warps of its own.

R3's run id has no warps knob (private_weight_reference.default_run_id). At a warps count other
than R3's 8 the id gains `warps<N>`, so a w4 and a w8 page at one (model, tile, G, session) never
share a run directory; at 8 every id is the one R3 writes today. Each page's report records the
warps it ran (`pinned.num_warps`, from the same FIXED).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

ENV = "MOE_TILE_NUM_WARPS"
#: the warps vLLM's default configs name (design-tile 1): 4 and 8
ALLOWED = (4, 8)
R3_WARPS = 8


class Refused(Exception):
    pass


def take_warps(argv: list[str]) -> tuple[int | None, list[str]]:
    """(--num-warps N or $MOE_TILE_NUM_WARPS or None, argv without the flag)."""
    rest, nw, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--num-warps":
            if i + 1 >= len(argv):
                raise Refused("--num-warps needs a value")
            nw, i = argv[i + 1], i + 2
            continue
        if a.startswith("--num-warps="):
            nw, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    if nw is None and os.environ.get(ENV):
        nw = os.environ[ENV]
    if nw is None:
        return None, rest
    try:
        v = int(nw)
    except ValueError as exc:
        raise Refused(f"--num-warps {nw!r}: a whole number") from exc
    if v not in ALLOWED:
        raise Refused(f"--num-warps {v}: vLLM's default configs name {ALLOWED}")
    return v, rest


def set_warps(nw: int) -> list:
    """FIXED["num_warps"] = nw in every loaded copy of the sweep module (R3 imports it as
    `block_m_crossing_sweep`, dram_counter_route as `scripts.block_m_crossing_sweep`: two module
    objects), and the environment for children. Returns the FIXED dicts set."""
    import block_m_crossing_sweep as S1
    out = [S1.FIXED]
    try:
        import scripts.block_m_crossing_sweep as S2
        if S2.FIXED is not S1.FIXED:
            out.append(S2.FIXED)
    except ImportError:
        pass
    r3 = sys.modules.get("private_weight_reference")
    if r3 is not None and all(r3.SWEEP.FIXED is not f for f in out):
        out.append(r3.SWEEP.FIXED)          # R3 already loaded against another copy of the module
    for f in out:
        f["num_warps"] = int(nw)
    os.environ[ENV] = str(int(nw))
    return out


class _WarpsPV:
    """provenance with `warps` added to every run id (only while R3 names its run)."""

    def __init__(self, pv, nw: int):
        self._pv, self._nw = pv, nw

    def __getattr__(self, k):
        return getattr(self._pv, k)

    def run_id(self, *, card, **swept):
        return self._pv.run_id(card=card, warps=self._nw, **swept)


def key_run_id(R3, nw: int) -> None:
    """R3's default_run_id with `warps<nw>` added when nw is not R3's 8."""
    if nw == R3_WARPS or getattr(R3.default_run_id, "_tile_warps", None) == nw:
        return
    orig = R3.default_run_id

    def default_run_id(args, card):
        pv = R3.PV
        R3.PV = _WarpsPV(pv, nw)
        try:
            return orig(args, card)
        finally:
            R3.PV = pv
    default_run_id._tile_warps = nw
    R3.default_run_id = default_run_id


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        nw, rest = take_warps(argv)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if nw is not None:
        set_warps(nw)
    import block_m_crossing_sweep as SWEEP
    import private_weight_reference as R3
    assert R3.SWEEP is SWEEP, "R3 reads another sweep module than the one set"
    if nw is not None:
        key_run_id(R3, nw)
    return R3.main(rest)


if __name__ == "__main__":
    sys.exit(main())
