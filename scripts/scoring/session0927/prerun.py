"""Glue that runs the PRE-RUN code of one registering commit, nothing else.

    python -I prerun.py timing TREE OUT_JSON OUT_TXT SESSION0927 RUN [RUN ...] --counters DIR
    python -I prerun.py wsc    TREE OUT_JSON OUT_TXT LOCK0927 COUNTERS0925

`score.py` (next to this file) extracts a registering commit with `git archive`
into TREE and runs this file with `python -I` and cwd = TREE, so the modules it
imports are that commit's `scripts/r3_timing_model.py` or
`scripts/wave_split_bytes.py` and that commit's `moe` package, never this
checkout's (it checks `moe.__file__`). It computes nothing of its own: it calls
the pre-run tool on the 2026-09-25 pages (its printout, captured, is OUT_TXT),
copies the numbers the registration names into OUT_JSON, and reads the
2026-09-27 pages through the same commit's own loaders. The only arithmetic
here is the eta inversion of row 1, done by bisection on the pre-run model's
own `predict_unmeasured` (the rule it inverts is the commit's: c at a clock f
is c (clock / f)^eta).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
from pathlib import Path


def _load(tree: Path, name: str):
    sys.path[:0] = [str(tree / "scripts"), str(tree)]
    spec = importlib.util.spec_from_file_location(name, tree / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import moe
    if not Path(moe.__file__).resolve().is_relative_to(tree.resolve()):
        raise SystemExit(f"moe imported from {moe.__file__}, not from {tree}")
    return mod


def _capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


def _bisect(f, target: float, lo: float = 0.0, hi: float = 3.0, steps: int = 60):
    """eta with f(eta) = target for f increasing in eta; None outside [lo, hi]."""
    flo, fhi = f(lo), f(hi)
    if not (flo <= target <= fhi):
        return None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def timing(tree: Path, out_json: Path, out_txt: Path, session0927: Path, rest: list[str]) -> int:
    M = _load(tree, "r3_timing_model")
    rc, txt = _capture(M.main, rest)
    out_txt.write_text(txt)
    R = M.build(M.build_parser().parse_args(rest))
    P, x, ctx, k_w = R["predictions"], R["main"].x, R["ctx"], R["k_w"]
    cells = R["cells"]
    shared_decl = {c.declared for c in cells if c.arm == "shared"}
    private_decl = {c.declared for c in cells if c.arm == "private"}
    decl = {"native": 8, "shared": min(shared_decl) if shared_decl else 72,
            "private": min(private_decl) if private_decl else 72}
    tab = M.sigma_table(R["source"], ctx, k_w, decl)

    def model_T(G, ns, mhz=None, eta=1.0):
        return [M.predict_unmeasured(x, ctx, k_w, tab, "shared", G, n, decl["shared"],
                                     mhz=mhz, eta=eta) for n in ns]

    pred = {
        "fit": {"x": list(map(float, x)), "names": list(M.NAMES), "clock_mhz": ctx.clock_mhz,
                "k_w": k_w, "sms": ctx.sms, "occupancy": dict(ctx.occupancy)},
        "P1_clocks": list(M.P1_CLOCKS), "P1_etas": list(M.P1_ETAS),
        "P1": {k: {kk: v[kk] for kk in ("T", "inc", "slope_2_6") if kk in v}
               for k, v in P["P1"].items()},
        "P2": {str(G): {k: P["P2"][G][k] for k in ("T", "inc", "step_1_2", "slope_2_6", "band")}
               for G in M.P2_GS},
        "P2_private_G8": {k: P["P2"][M.P2_PRIVATE_G]["private"][k] for k in ("T", "slope_2_6")},
        "P3": {"G2_T": P["P3"]["G2"]["T"], "G4_T": P["P3"]["G4"]["T"],
               "odd_n_excess": {str(n): v for n, v in P["P3"]["odd_n_excess"].items()},
               "min_excess_ms": M.P3_MIN_EXCESS_MS},
        "P4": {"G4_inc": P["P4"]["G4"]["inc"], "inc_8_9": P["P4"]["inc_8_9"],
               "inc_others_median": P["P4"]["inc_others_median"],
               "q_w2": {str(n): v for n, v in P["P4"]["q_w2"].items()}},
        "P5": {"G3_T": P["P5"]["G3"]["T"], "G3_inc": P["P5"]["G3"]["inc"],
               "rho_star_w1": P["P5"]["rho_star_w1"],
               "slab_fraction_max_G3": P["P5"]["slab_fraction_max_G3"], "flat": P["P5"]["flat"]},
        "P6": P["P6"],
        "P7": {"locked_fit": P["P7"]["locked_fit"], "pages": P["P7"]["pages"]},
        "P8": {str(k): v for k, v in P["P8"].items()},
        "falsifiers": list(M.FALSIFIERS),
    }

    # The 2026-09-27 timed pages, read by this commit's own loader.
    pages = M.discover_timed([session0927])
    meas = []
    for p in pages:
        shared = {n: v[0] for (arm, n), v in p.rows.items() if arm == "shared"}
        meas.append({"run": p.run, "G": p.G, "label": p.label, "failed": list(p.failed),
                     "clocks": sorted(c for c in p.clocks if c is not None),
                     "locked": p.locked, "duty": p.duty,
                     "shared_ms": {str(n): shared[n] for n in sorted(shared)}})
    meas.sort(key=lambda d: (d["G"], d["clocks"], d["run"]))

    # Row 1's eta inversion on the pre-run model, against the VALID pages.
    def ladder(G, mhz):
        rows = [d for d in meas if d["G"] == G and d["clocks"] == [mhz] and d["label"] == "VALID"]
        if not rows:
            return None
        ns = sorted({int(n) for d in rows for n in d["shared_ms"]})
        return {n: statistics.median(d["shared_ms"][str(n)] for d in rows) for n in ns}

    inv = {}
    for mhz in M.P1_CLOCKS:
        lad = ladder(4, mhz)
        if lad is None:
            inv[f"G4/f{mhz:.0f}"] = None
            continue
        target = M._slope([lad[n] for n in range(2, 7)])
        inv[f"G4/f{mhz:.0f}"] = {"measured_slope_2_6": target, "eta": _bisect(
            lambda e, m=mhz: M._slope(model_T(4, range(2, 7), m, e)), target)}
    lad = ladder(2, M.P1_CLOCKS[0])
    if lad is not None:
        inc = [lad[n + 1] - lad[n] for n in range(1, 6)]
        low = statistics.fmean(inc[i] for i in (0, 2, 4))

        def low_model(e):
            T = model_T(2, range(1, 7), M.P1_CLOCKS[0], e)
            return statistics.fmean(T[i + 1] - T[i] for i in (0, 2, 4))
        inv[f"G2/f{M.P1_CLOCKS[0]:.0f}/low_steps"] = {"measured_mean": low,
                                                      "eta": _bisect(low_model, low)}
    pred["eta_inversion"] = inv
    out_json.write_text(json.dumps({"rc": rc, "predictions": pred, "measured_0927": meas},
                                   indent=1, default=str) + "\n")
    return 0


def wsc(tree: Path, out_json: Path, out_txt: Path, lock0927: Path, counters0925: str) -> int:
    W = _load(tree, "wave_split_bytes")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "cells.json"
        rc, txt = _capture(W.main, [counters0925, "--out", str(tmp)])
        doc = json.loads(tmp.read_text())
    out_txt.write_text("".join(ln for ln in txt.splitlines(keepends=True)
                               if not ln.startswith("wrote ")))
    (card,) = doc["cards"].values()
    pred = [{k: r[k] for k in ("arm", "G", "n", "gemm", "q_fill", "q_ws", "status")}
            for r in card["predicted"]]
    measured = W.load_card(lock0927).measured
    meas = [{"arm": a, "G": G, "n": n, "gemm": g, "q": q}
            for (a, G, n, g), q in sorted(measured.items())]
    out_json.write_text(json.dumps({"rc": rc, "registered_view": card["registered_view"],
                                    "card": card["card"], "predicted": pred,
                                    "measured_0927": meas}, indent=1) + "\n")
    return 0


def main(argv: list[str]) -> int:
    mode, tree, out_json, out_txt, pages, *rest = argv
    tree = Path(tree).resolve()
    if mode == "timing":
        return timing(tree, Path(out_json), Path(out_txt), Path(pages), rest)
    if mode == "wsc":
        return wsc(tree, Path(out_json), Path(out_txt), Path(pages), rest[0])
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
