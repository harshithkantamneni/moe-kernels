#!/usr/bin/env python
"""E1-TILE power on rental B's 84 cells, and the rental-A stop rule (registrations
2026-10-14-tileB-e1tile and -tileB-stop; design-tile-review 4, owner fix 6).

    python scripts/scoring/tileA/power_tile.py      # MPS float32 against CPU float64, writes the check file

The truth model is design-tile-review work/power_rb075.py's, re-run on the registered cells with
the registered byte sensitivities (dln T / dln bytes of v4 at the design-nominal constants, per
cell): ln(T_meas / T_v4) = b_class + b_(model, class) + cell + bytes x sensitivity + page + board,
with b_class the common-mode tile bias (every model at a tile class; rental A's E1-CAL sees it on
the CAL models), b_(model, class) the model-specific bias (no rental-A page sees it for a blind
model), cell 0.6%, bytes 1.75%, page 0.18%, board 0.26% (rental 3). E1-TILE holds iff the pooled
cells and each BLIND-TILE model's cells meet the bar (deepseek-v2-lite in the pooled bar only).
Rental A's checks: a class mean over about 18 CAL cells per class (b_class + its own model bias +
noise), the stop rule clause A2 (every class mean within 0.75%) and A3 (A2 and, at 128 G 1, two
CAL models within 1%). Normals are drawn on the CPU in float64 from one seed and moved, so the two
devices see the same draws. Memory is bounded: draws in chunks of 25,000.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

CHECK = REPO / "docs" / "registered" / "2026-10-14-tileB-hist" / "power_tile_check.json"
SETTINGS = ((0.0, 0.0), (0.005, 0.0), (0.01, 0.0), (0.02, 0.0), (0.0, 0.005), (0.0, 0.01), (0.0, 0.015),
            (0.005, 0.005), (0.01, 0.01))
NOISE = {"sig_byte": 0.0175, "sig_cell": 0.006, "sig_page": 0.0018, "sig_board": 0.0026, "cal_cells_per_class": 18}
R_DEFAULT = 200_000
CHUNK = 25_000
SEED = 20261414
CLASSES = ("64g1", "128g1", "128g16")


def power(cells: list[dict], common: float, spec: float, *, blind_models, R: int = R_DEFAULT, device: str = "cpu",
          seed: int = SEED, stop_mean: float = 0.0075, stop_pair: float = 0.01, noise: dict = NOISE) -> dict:
    """cells {model, page, cls, s}: the probabilities of E1-TILE (pooled and each blind model),
    with deepseek-v2-lite in each model's clause too (the review's P_in, printed), of the stop
    rule's clauses, and of E1-TILE given the rule passes."""
    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    models = sorted({c["model"] for c in cells})
    pages = sorted({c["page"] for c in cells})
    N = len(cells)
    mi = torch.tensor([models.index(c["model"]) for c in cells])
    ci = torch.tensor([CLASSES.index(c["cls"]) for c in cells])
    pi = torch.tensor([pages.index(c["page"]) for c in cells])
    sb = torch.tensor([float(c["s"]) for c in cells], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    acc = {k: 0.0 for k in ("e1", "e1_dsv_in", "A2", "A3", "e1_and_A2", "e1_and_A3")}
    t_dev = 0.0

    def bar(r):
        return (r.pow(2).mean(1).sqrt() <= 0.02) & (r.abs().amax(1) <= 0.05) & (r.mean(1).abs() <= 0.01)
    done = 0
    while done < R:
        B = min(CHUNK, R - done)
        rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)  # noqa: E731
        bc = rn(B, 3) * common
        sp = rn(B, len(models), 3) * spec
        e = (bc[:, ci] + sp[:, mi, ci] + rn(B, N) * noise["sig_cell"] + rn(B, N) * noise["sig_byte"] * sb[None]
             + rn(B, len(pages))[:, pi] * noise["sig_page"] + rn(B, 1) * noise["sig_board"])
        k = noise["cal_cells_per_class"] ** 0.5
        mA = bc + rn(B, 3) * spec + rn(B, 3) * (noise["sig_cell"] / k) + rn(B, 3) * noise["sig_page"] + rn(B, 1) * noise["sig_board"]
        mA2 = bc + rn(B, 3) * spec + rn(B, 3) * (noise["sig_cell"] / k) + rn(B, 3) * noise["sig_page"] + rn(B, 1) * noise["sig_board"]
        t0 = time.perf_counter()
        e, mA, mA2, mi_d = e.to(device, dt), mA.to(device, dt), mA2.to(device, dt), mi.to(device)
        r = torch.exp(-e) - 1
        pooled = bar(r)
        ok, ok_in = pooled.clone(), pooled.clone()
        for j, m in enumerate(models):
            kk = bar(r[:, mi_d == j])
            ok_in &= kk
            if m in blind_models:
                ok &= kk
        A2 = (mA.abs() <= stop_mean).all(1)
        A3 = A2 & ((mA - mA2)[:, 1].abs() <= stop_pair)
        f = lambda x: float(x.to(dt).sum())  # noqa: E731
        acc["e1"] += f(ok)
        acc["e1_dsv_in"] += f(ok_in)
        acc["A2"] += f(A2)
        acc["A3"] += f(A3)
        acc["e1_and_A2"] += f(ok & A2)
        acc["e1_and_A3"] += f(ok & A3)
        if device == "mps":
            torch.mps.synchronize()
        t_dev += time.perf_counter() - t0
        done += B
    p = {k: v / R for k, v in acc.items()}
    return {"P(E1-TILE)": p["e1"], "P(E1-TILE, deepseek in each-model clause)": p["e1_dsv_in"],
            "P(stop rule A2 passes)": p["A2"], "P(stop rule A2 and A3 pass)": p["A3"],
            "P(E1-TILE | A2 passes)": p["e1_and_A2"] / p["A2"] if p["A2"] else None,
            "P(E1-TILE | A2 and A3 pass)": p["e1_and_A3"] / p["A3"] if p["A3"] else None,
            "seconds_on_device": t_dev}


def grid(cells: list[dict], blind_models, *, R: int = R_DEFAULT, device: str = "cpu") -> dict:
    out = {}
    for common, spec in SETTINGS:
        p = power(cells, common, spec, blind_models=blind_models, R=R, device=device)
        out[f"common {common:g} / model-specific {spec:g}"] = p
    return out


def main(argv=None) -> int:
    import torch
    sys.path.insert(0, str(HERE))
    import tcommon as TC
    reg = TC.registration(REPO, "e1tile")
    cells = reg["power"]["cells"]
    blind = reg["power"]["blind_models"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    got = grid(cells, blind, device=dev)
    t_dev = time.time() - t0
    t0 = time.time()
    ref = grid(cells, blind, device="cpu")
    t_cpu = time.time() - t0
    worst = max(abs((got[k][x] or 0) - (ref[k][x] or 0)) for k in got for x in got[k] if x != "seconds_on_device")
    doc = {"what": "E1-TILE and stop-rule power on rental B's registered cells, laptop GPU against CPU float64",
           "torch": torch.__version__, "device": dev, "dtype": "float32" if dev == "mps" else "float64", "R": R_DEFAULT,
           "settings": len(SETTINGS), "max_abs_diff": worst, "seconds": {"device": round(t_dev, 2), "cpu_float64": round(t_cpu, 2)},
           "device_results": {k: {x: (round(v, 6) if isinstance(v, float) else v) for x, v in d.items()} for k, d in got.items()}}
    CHECK.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {CHECK}: {dev} {t_dev:.2f} s, cpu float64 {t_cpu:.2f} s, max |diff| {worst:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
