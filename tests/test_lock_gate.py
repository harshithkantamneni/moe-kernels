# ruff: noqa: E501
"""The lock check that replaced FL1 / V10's t0 + cycles / f fit (2026-10-05).

Synthetic pages pin each part of `scripts/lock_gate.py`'s rule: a held lock
passes even where the old fit is degenerate; a capture about 5.6% off the
lock fails on each independent reading by itself; a drift inside a capture
fails; a cell too short to read is left out; a page with no null kernel and
no base twin falls back to nvidia-smi, and with no reading at all is not
passed. Then the re-gate of every published lock page is pinned, the
statistics each tolerance is set from are recomputed from the published
held-lock pages, and the two readers that print stored gates are shown to
print the recomputed lock verdict with the stored one labelled stale.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts import lock_gate as LG  # noqa: E402

LOCK = 1710.0
TREADS = (2, 3, 4, 5, 6, 7, 8, 9)
SHORTFALL = 0.02          # the cycle counter's shortfall against the duration


def _dur(gemm: str, n: int) -> float:
    """A duration in ns that grows with treads plus a fixed part."""
    return (40_000.0 + 90_000.0 * n) if gemm == "w1" else (20_000.0 + 45_000.0 * n)


def _floor(*, clock=LOCK, smi=LOCK, treads=TREADS, null=True, null_bias=-0.014,
           drift=None, extra=None, lock=LOCK) -> dict:
    """A floor capture under `lock` whose SM ran at `clock` (times `drift[n]`
    in cell n). Every cell's counter reads SHORTFALL low, the same in every
    cell, so the old fit reads f = (1 - SHORTFALL) x clock and t0 = 0: the
    degenerate fit of the published small-shard floors. `extra` adds cells
    {n: {gemm: (ns, cycles)}}."""
    drift = drift or {}
    cells = []
    for n in treads:
        f = clock * drift.get(n, 1.0)
        per = {g: {"sm__cycles_elapsed.avg": f * (1 - SHORTFALL) * _dur(g, n) / 1e3,
                   "gpu__time_duration.sum": _dur(g, n)} for g in ("w1", "w2")}
        cell = {"arm": "native", "n": n, "per_gemm": per}
        if null:
            cell["null"] = {"sm_clock_mhz": f * (1 + null_bias)}
        cells.append(cell)
    for n, gs in (extra or {}).items():
        cells.append({"arm": "native", "n": n, "per_gemm": {
            g: {"sm__cycles_elapsed.avg": c, "gpu__time_duration.sum": t}
            for g, (t, c) in gs.items()}})
    smi_rec = None if smi is None else {"rows": [{"clocks.sm": f"{smi:g}"}]}
    return {"family": "r3-arms", "kind": "floor", "cells": cells,
            "clock": {"control": "none", "lock_mhz": lock, "smi_before": smi_rec,
                      "smi_after": smi_rec},
            "gates": [{"number": "FL1", "kind": "VALIDITY", "verdict": "FAIL"}]}


def _base_twin(page: dict, ratio: float = LG.RATIO_REF[LOCK]) -> dict:
    """The same cells at ncu's base clock: cycles over duration = the lock
    capture's divided by `ratio`, cell by cell (the shortfall divides out)."""
    twin = json.loads(json.dumps(page))
    twin["clock"] = {"control": "base", "lock_mhz": None}
    for c in twin["cells"]:
        c.pop("null", None)
        for v in c["per_gemm"].values():
            v["sm__cycles_elapsed.avg"] /= ratio
    return twin


def _write(tmp_path: Path, page: dict, twin: dict | None = None) -> Path:
    path = tmp_path / "r3f-g64-lock1710.json"
    path.write_text(json.dumps(page))
    if twin is not None:
        (tmp_path / "r3f-g64.json").write_text(json.dumps(twin))
    return path


def _names(chk, ok: bool) -> list[str]:
    return [r.name for r in chk.readings if r.ok is ok]


def test_a_held_lock_passes_where_the_old_fit_is_degenerate(tmp_path):
    """Counters SHORTFALL low in every cell: the old fit reads 1676 MHz + 0 us
    and would fail 'w1 fitted 1676 MHz'. nvidia-smi, the null kernel (-1.4%)
    and the base twin's ratio all read the lock: PASS, the fit printed."""
    page = _floor()
    chk = LG.check_page(page, _write(tmp_path, page, _base_twin(page)))
    assert chk.verdict == LG.PASS
    assert _names(chk, True) == ["smi", "null", "null spread", "ratio", "ratio spread"]
    assert "w1 fitted 1676 MHz" in chk.fit_off and chk.fit[0].startswith("w1 1675.8 MHz")
    assert "old fit, diagnostic only" in chk.measured()
    assert "-> would read FAIL (w1 fitted 1676 MHz" in chk.measured()


@pytest.mark.parametrize("which", ["smi", "null", "ratio"])
def test_a_capture_5_6_percent_off_the_lock_fails_on_each_reading_alone(tmp_path, which):
    """The 8x22B floor's size of error, +5.6% (1806 MHz at a 1710 lock),
    caught by each independent reading with the other two absent or reading
    the lock: nvidia-smi alone (1980, the lock not in force), the null kernel
    alone, or the base twin's ratio alone."""
    off = LOCK * 1.056
    if which == "smi":
        page = _floor(clock=off, smi=1980.0, null=False)
        path = _write(tmp_path, page)
    elif which == "null":
        page = _floor(clock=off, smi=None)
        path = _write(tmp_path, page)
    else:
        page = _floor(clock=off, smi=None, null=False)
        twin = _base_twin(_floor(smi=None, null=False))
        path = _write(tmp_path, page, twin)
    chk = LG.check_page(page, path)
    assert chk.verdict == LG.FAIL
    assert _names(chk, False) == [which]
    assert chk.measured().startswith("off the lock: ")


def test_the_8x22b_sized_error_is_far_outside_and_a_held_lock_far_inside():
    """Margins: the null median at +5.6% x (1 - 1.4%) reads +4.1%, outside 3%;
    the held null medians read -1.3 to -1.5%, inside."""
    assert abs(1.056 * (1 - 0.014) - 1) > LG.NULL_LEVEL_TOL
    assert abs(1684.4 / LOCK - 1) < LG.NULL_LEVEL_TOL
    assert abs(1.2866 / LG.RATIO_REF[LOCK] - 1) > LG.RATIO_LEVEL_TOL


def test_a_drift_within_a_capture_fails_on_the_ratio_and_on_the_null(tmp_path):
    """The lock slips by 6% for the last two treads only. The capture's median
    stays on the lock (smi brackets read 1710 both sides), and the cell by
    cell readings catch it: the base twin's ratio spread and, separately, the
    null kernel's spread."""
    drift = {8: 1.06, 9: 1.06}
    page = _floor(drift=drift, null=False)
    chk = LG.check_page(page, _write(tmp_path, page, _base_twin(_floor(null=False))))
    assert chk.verdict == LG.FAIL and _names(chk, False) == ["ratio spread"]
    assert "smi" in _names(chk, True) and "ratio" in _names(chk, True)
    page = _floor(drift=drift)
    chk = LG.check_page(page, None)
    assert chk.verdict == LG.FAIL and _names(chk, False) == ["null spread"]
    small = _floor(drift={8: 1.015, 9: 1.015})
    assert LG.check_page(small, None).verdict == LG.PASS   # within the held scatter


def test_a_cell_too_short_to_read_is_left_out(tmp_path):
    """An n=1 cell whose duration is mostly the part that does not scale with
    treads (offset share above 0.30) and whose ratio strays 8% is left out of
    the per-cell check by the stated rule, and named; with the rule switched
    off the same capture fails its ratio spread."""
    t1 = {"w1": 230_000.0, "w2": 115_000.0}        # share (t - 1 x b) / t = 0.61
    extra = {1: {g: (t, LOCK * (1 - SHORTFALL) * 1.08 * t / 1e3) for g, t in t1.items()}}
    page = _floor(extra=extra, null=False)
    twin = _base_twin(_floor(extra={1: {g: (t, LOCK * (1 - SHORTFALL) * t / 1e3)
                                        for g, t in t1.items()}}, null=False))
    path = _write(tmp_path, page, twin)
    shares = LG.offset_shares(LG.page_points(page))
    assert shares[("native", 1, "w1")] == pytest.approx(1 - 90_000.0 / 230_000.0)
    assert max(s for k, s in shares.items() if k[1] >= 2) < LG.SHORT_SHARE_MAX
    chk = LG.check_page(page, path)
    assert chk.verdict == LG.PASS
    assert chk.excluded == ["native/1 w1 share 0.61", "native/1 w2 share 0.61"]
    assert "too short to read, left out: native/1 w1 share 0.61" in chk.measured()
    old = LG.SHORT_SHARE_MAX
    try:
        LG.SHORT_SHARE_MAX = 1.0
        assert LG.check_page(page, path).verdict == LG.FAIL
    finally:
        LG.SHORT_SHARE_MAX = old


def test_a_gemm_with_one_tread_has_no_slope_and_its_cells_are_left_out():
    pts = {("native", 4, "w1"): (1.0e6, 6.0e5)}
    keep, short = LG.readable(pts)
    assert keep == set() and short == ["native/4 w1 no slope"]


def test_no_null_kernel_and_no_base_twin_falls_back_to_nvidia_smi(tmp_path):
    """Every published byte page: no null kernel, no twin with cycles. The
    stated fallback judges the lock on the nvidia-smi bracket alone and says
    drift was not tested; with no smi reading either there is no level
    reading, and the verdict is REFUSE (UNKNOWN), never PASS."""
    page = _floor(null=False)
    chk = LG.check_page(page, _write(tmp_path, page))
    assert chk.verdict == LG.PASS and _names(chk, True) == ["smi"]
    assert chk.fallback.startswith("fallback: no null kernel and no base twin, judged on "
                                   "the nvidia-smi bracket alone; drift within the capture "
                                   "not tested")
    assert LG.check_page(_floor(null=False, smi=None), None).verdict == LG.REFUSE
    off = LG.check_page(_floor(null=False, smi=LOCK + 15.1), None)
    assert off.verdict == LG.FAIL and _names(off, False) == ["smi"]
    assert LG.check_page(_floor(null=False, smi=LOCK + 15.0), None).verdict == LG.PASS


def test_twin_paths_and_a_twin_that_is_not_at_base_is_not_used(tmp_path):
    assert LG.twin_path(Path("x/r3f-g64-lock1710.json")) == Path("x/r3f-g64.json")
    assert LG.twin_path(Path("x/lock1710/r3c-g2.json")) == Path("x/base/r3c-g2.json")
    assert LG.twin_path(Path("x/r3f-g64.json")) is None
    page = _floor(null=False)
    twin = _base_twin(page)
    twin["clock"]["control"] = "none"
    path = _write(tmp_path, page, twin)
    assert LG.load_twin(path, page) is None
    assert LG.check_page(page, path).fallback


def test_a_page_under_no_lock_has_no_lock_check():
    page = _floor()
    page["clock"]["lock_mhz"] = None
    assert LG.check_page(page, None) is None


def test_every_tolerance_is_set_from_the_published_held_lock_pages():
    """`derive` reads the published held-lock captures (held: nvidia-smi read
    the lock both sides) and each tolerance is 2 x its maximum, rounded up to
    0.5%; the short-cell threshold sits in the gap between n >= 2 and n = 1."""
    d = LG.derive()
    assert d["smi_readings"] == 232 and d["smi_max_off_mhz"] == 0.0
    assert d["null_captures"] == 4 and d["ratio_captures"] == 14
    assert d["null_level_max"] == pytest.approx(0.01497, abs=5e-5)
    assert d["null_spread_max"] == pytest.approx(0.01050, abs=5e-5)
    assert d["ratio_ref_1710"] == pytest.approx(LG.RATIO_REF[LOCK], abs=5e-5)
    assert d["ratio_level_max"] == pytest.approx(0.01395, abs=5e-5)
    assert d["ratio_spread_max"] == pytest.approx(0.02244, abs=5e-5)
    assert LG.tolerance_from(d["null_level_max"]) == LG.NULL_LEVEL_TOL
    assert LG.tolerance_from(d["null_spread_max"]) == LG.NULL_SPREAD_TOL
    assert LG.tolerance_from(d["ratio_level_max"]) == LG.RATIO_LEVEL_TOL
    assert LG.tolerance_from(d["ratio_spread_max"]) == LG.RATIO_SPREAD_TOL
    assert d["share_long_max"] < LG.SHORT_SHARE_MAX < d["share_n1_min"]
    assert (d["share_long_max"], d["share_n1_min"]) == pytest.approx((0.2111, 0.3555), abs=5e-4)


#: Every published lock page re-gated on read: path, gate, stored, recomputed.
PINNED = """
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL FAIL
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 PASS PASS
2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 PASS PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 PASS PASS
2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 PASS PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g3.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g4.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g16.json V10 PASS PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g32.json V10 PASS PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters/lock1710/r3c-g8.json V10 PASS PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-atile-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-atile-r3-counters/lock1710/r3c-g32.json V10 PASS PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-atile-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-atile-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-atile-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-atile-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-atile-r3-counters/lock1710/r3c-g16.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-atile-r3-counters/lock1710/r3c-g32.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-atile-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-atile-r3-counters/lock1710/r3c-g8.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g8-lock1710.json FL1 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-l2-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-l2-r3-counters/lock1710/r3c-g2.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-olmoe-1b-7b-g128-r3-counters/lock1710/r3c-g128.json V10 FAIL PASS
2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-qwen2-57b-a14b-g128-r3-counters/lock1710/r3c-g128.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-l2base-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-l2s8-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-atbk128-r3-counters/lock1710/r3c-g64.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-atbk32-r3-counters/lock1710/r3c-g64.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-atbk64-r3-counters/lock1710/r3c-g64.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-ats8-r3-counters/lock1710/r3c-g64.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-floor-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2base-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2bk128-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2bk32-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2pad7-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2s6-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2s8-r3-counters/lock1710/r3c-g1.json V10 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-floor-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-floor1005-r3-counters/r3f-g64-lock1005.json FL1 PASS PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2base-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2s8-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor2-r3-counters/r3f-g64-lock1710.json FL1 FAIL PASS
2026-10-02-nvidia_gh200_480gb-rental2-session/results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp8-l2s8-r3-counters/lock1710/r3c-g1.json V10 FAIL PASS
"""


def test_the_regate_of_every_published_lock_page_is_pinned():
    """117 lock pages (summary.json files repeat their pages' gates and are
    not pages). The 8x22B floor, the lock not in force, fails; every other
    page passes, 78 stored FAILs among them. The published JSONs are read,
    never written."""
    want = [tuple(line.split()) for line in PINNED.strip().splitlines()]
    before = {p: p.stat().st_mtime_ns for p in LG.published_lock_pages()}
    got = [(str(Path(r["path"]).relative_to(LG.PUBLISHED)), r["gate"], str(r["stored"]),
            r["new"]) for r in LG.regate_table()]
    assert got == want
    assert {p: p.stat().st_mtime_ns for p in LG.published_lock_pages()} == before
    fails = [g for g in got if g[3] != LG.PASS]
    assert fails == [("2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-"
                      "nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json", "FL1", "FAIL",
                      "FAIL")]
    assert sum(1 for g in got if g[2] != g[3]) == 78


def test_the_8x22b_floor_fails_on_two_independent_readings():
    path = (LG.PUBLISHED / "2026-09-28-nvidia_gh200_480gb-8x22b-session/results/"
            "2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json")
    r = LG.regate_path(path)
    assert r["new"] == LG.FAIL and _names(r["check"], False) == ["smi", "ratio"]
    assert "implied 1804 MHz" in r["check"].measured()


def test_the_cli_prints_the_table(capsys):
    assert LG.main([]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("117 lock pages; 78 verdicts change; new verdict "
                                           "not PASS on 1: 2026-09-28-nvidia_gh200_480gb-8x22b")


def test_the_readers_print_the_recomputed_lock_verdict_and_label_the_stored_stale():
    """r3_timing_model.py (`stored_validity`) and wave_split_bytes.py
    (`card_lines`, through `lock_gate.reader_gate_entries`) printed the stored
    V10; they now print the recomputed verdict, the stored one labelled stale,
    and every other stored gate as before."""
    import r3_timing_model as T

    from scripts import wave_split_bytes as W
    jet = (LG.PUBLISHED / "2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/"
           "2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/r3c-g8.json")
    page = json.loads(jet.read_text())
    page["_path"] = str(jet)
    assert [g["verdict"] for g in page["gates"] if g["number"] == "V10"] == ["FAIL"]
    assert T.stored_validity(page) == ["V10 PASS (recomputed lock check; stored FAIL, stale)"]
    card = W.load_card(LG.PUBLISHED / "2026-10-01-nvidia_gh200_480gb-rental1-session/results/"
                       "2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-atile-r3-counters/lock1710")
    got = {G: LG.reader_gate_entries(p, card.paths[G]) for G, p in card.pages.items()}
    assert got[8] == ["V6 FAIL", "V10 PASS (recomputed lock check; stored FAIL, stale)", "C1 FAIL"]
    assert all("V10 PASS (recomputed lock check; stored FAIL, stale)" in v for v in got.values())
    held = dict(page, gates=[{"number": "V10", "kind": "VALIDITY", "verdict": "PASS"}])
    assert T.stored_validity(held) == []
