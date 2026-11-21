#!/usr/bin/env python
"""MODEL v4 AT A LOCK OTHER THAN 1710 (tile rental A2, 2026-10-20; docs/registered/2026-10-20-tileA2-floor-gh200).

    python scripts/r3_timing_model_v4L.py --check     # at L = 1710: v4 itself, and through it all 300 registered
                                                      # v3 predictions, bit for bit

v4 (scripts/r3_timing_model_v4.py, pinned) has no clock conversion: it turns counter cycles into time
through rental 6's c_card, 202.548 ns per CTA k-step fitted at the 1710 MHz lock (346.4 cycles), as
c_time(tile) = c_card x c_scale(tile), c_scale = c_counter(tile) / 344.1. A2 times its pages at a lower
lock L (1410 MHz, the owner's decision of 2026-10-20), so v4 is priced there by P1's registered rule
(r3_timing_model.predict_unmeasured, P1_CLOCKS): c ALONE scales by (fit clock / L)^eta with eta 1,
at every tile R3's included. `at_lock(L)` does that in memory: it replaces the module attribute
`r3_timing_model_v4.c_scale`, which every pricing path looks up by name (wave_split_bytes_v4.price,
regtile.dead_increments, r3_timing_model_v4d), by c_scale x 1710 / L, and the roofline rival's clock
`CLOCK_MHZ` by L; both are restored on exit. The pinned files are never touched.

WHAT CARRIES UNCONVERTED (an untested assumption, registered as such; CONV-L in tileA2-convl tests it):
T0, bw, s_small, s_block, h, CORES rho, the dead term's d (the dead term's c part is converted, since
dead_ms reads the scaled c) and MVA2's eta, z0 and z1 (cycles; the mma.sync path, printed only at the
w4 and ms tiles). The wgmma constants c_128, F_wg and v4cal are A2's own, measured in counter cycles at
L, so they need no conversion. At L = 1710 (the ladder's first rung) the multiplier is exactly 1.0 and
at_lock is the identity: v4 itself, no conversion.

`--check`: under at_lock(1710) the multiplier is exactly 1.0, and r3_timing_model_v4.check() must
re-price all 300 registered v3 predictions of rental6-skew and rental7-skew to the registered number
(6 decimals) and to rental 6's pricer bit for bit; it also requires rental 6's fit clock
(`Context.clock_mhz`) to be the 1710 this module converts from.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model_v4 as V4  # noqa: E402

#: the clock rental 6's c_card was fitted at (reg6.pricer().ctx.clock_mhz; `--check` holds it)
FIT_CLOCK_MHZ = 1710.0
#: P1's exponent (r3_timing_model.predict_unmeasured's default; SEEN-selected on 2026-09-27's pages)
ETA = 1.0
_STATE: list = []
#: v4's own c_scale, captured at import: every at_lock multiplies THIS one, so nested calls do not compound
_C_SCALE_V4 = V4.c_scale


def factor(lock_mhz: float) -> float:
    """The c multiplier at lock L under P1's rule: (1710 / L)^1."""
    return (FIT_CLOCK_MHZ / float(lock_mhz)) ** ETA


def current_lock() -> float:
    """The lock at_lock has set (1710 outside any at_lock)."""
    return _STATE[-1] if _STATE else FIT_CLOCK_MHZ


@contextlib.contextmanager
def at_lock(lock_mhz: float):
    """v4 priced at lock L: every tile's c_scale times (1710 / L), the roofline at L; restored on exit.
    Nested calls do not compound: the inner lock alone is in force inside, the outer one again after."""
    L = float(lock_mhz)
    if not 300.0 <= L <= 2000.0:
        raise ValueError(f"lock {L} MHz: not a GH200 graphics clock")
    keep_scale, keep_clock = V4.c_scale, V4.CLOCK_MHZ
    f = factor(L)

    def c_scale(t, g, prm, _base=_C_SCALE_V4, _f=f):
        return _base(t, g, prm) * _f
    c_scale.__doc__ = f"r3_timing_model_v4.c_scale x {f!r} (at_lock {L:g} MHz, P1's rule)"
    V4.c_scale, V4.CLOCK_MHZ = c_scale, L
    _STATE.append(L)
    try:
        yield f
    finally:
        _STATE.pop()
        V4.c_scale, V4.CLOCK_MHZ = keep_scale, keep_clock


def check(limit: int | None = None) -> dict:
    """at_lock(1710) reproduces v4, and through it every registered v3 prediction (V4.check, equality)."""
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "rental6"))
    import reg6 as R6
    clock = float(R6.pricer().ctx.clock_mhz)
    with at_lock(FIT_CLOCK_MHZ) as f:
        got = V4.check(limit)
    got["fit_clock_mhz"] = clock
    got["factor_at_1710"] = f
    got["fit_clock_is_1710"] = clock == FIT_CLOCK_MHZ
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="at_lock(1710) against every registered v3 prediction")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.check:
        ap.print_help()
        return 2
    got = check(a.limit)
    print(f"v4 at_lock(1710) (multiplier {got['factor_at_1710']!r}) against registered v3: {got['compared']} "
          f"predictions compared; {len(got['unequal_registered'])} differ from the registered number (6 decimals), "
          f"{len(got['unequal_pricer'])} differ from rental 6's pricer bit for bit; rental 6's fit clock "
          f"{got['fit_clock_mhz']:g} MHz ({'the 1710 converted from' if got['fit_clock_is_1710'] else 'NOT 1710'})")
    for u in (got["unequal_registered"] + got["unequal_pricer"])[:10]:
        print("  DIFFERS", u)
    ok = got["compared"] and not got["unequal_registered"] and not got["unequal_pricer"] and got["fit_clock_is_1710"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
