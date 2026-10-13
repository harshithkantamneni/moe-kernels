#!/usr/bin/env python
"""The l2open page's row-key rivals, priced on EVERY w1 row of the exact page BEFORE any rental-7
page (registration 2026-10-13-rental7-l2open; design-r7 3.4, design-r7-review (e), owner
decision 3 of 2026-10-13).

    python scripts/scoring/rental7/rowkeys7.py [--device mps|cpu] [--no-cpu-check]

The rival is rental 6's, unchanged: scripts/scoring/rental6/rowkeys.py (pinned by rental 6, imported
and never edited): w1's A keyed by hidden-state row and k-step, vLLM's pid mapping, one exact LRU
of C MiB. It is read at two capacities: 39.5 MiB (rental 6's registered rival) and 60 MiB, the one
capacity rental 7 registers (SEEN-chosen: design-r7 work/olmoe_w1.py swept C over the published
rental-5 and rental-6 OLMoE byte pages, whose skewed rows read best at 55 to 60 MiB).

Every cell of docs/registered/2026-10-13-rental7-skew-hist/olmoe-1b-7b-l2open.json is priced: PT,
PW, uniform and balanced, each arm it runs. A balanced cell's ids are R3's balanced_ids, which are
realize_counts([T k / E] x E) with no shuffle (block_m_crossing_sweep.balanced_ids), so the rival
reads them as counts [T k / E] x E and shuffle seed None. Each cell runs on the laptop GPU (torch
MPS, float32) and again on the CPU in float64; both read-byte totals and wall times are written,
and the registration carries the MPS numbers and the largest relative difference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "rental6")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rowkeys as RK  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

HIST7 = REPO / "docs" / "registered" / "2026-10-13-rental7-skew-hist"
PAGE = HIST7 / "olmoe-1b-7b-l2open.json"
OUT = HIST7 / "l2open_rowkey.json"
MODEL = RK.MODEL
C_MIB = (39.5, 60.0)
REGISTERED_C_MIB = 60.0


def cells() -> tuple[list[dict], int]:
    import r3_timing_model as TM
    cfg = MODEL_CONFIGS[MODEL]
    page = json.loads(PAGE.read_text())
    seed = int(page["shuffle_seed"])
    out = []
    old = TM.set_model(MODEL)
    try:
        for c in page["cells"]:
            n = int(c["n"])
            tokens = RK_tokens(cfg, n)
            if c["counts"] is None:
                per = tokens * cfg.top_k // cfg.num_experts
                counts, cseed = tuple([per] * cfg.num_experts), None
            else:
                counts, cseed = tuple(int(v) for v in c["counts"]), seed
            for arm in c["arms"]:
                D = cfg.num_experts * (1 if arm == "native" else 9)
                out.append(dict(label=c["label"], n=n, arm=arm, declared=D, counts=counts, shuffle_seed=cseed,
                                tokens=tokens, k=cfg.top_k, P=2 * cfg.intermediate_size // RK.BN,
                                S=cfg.hidden_size // RK.BK, R=TM.grid_rows(D, n, counts)))
    finally:
        TM.set_model(old)
    return out, seed


def RK_tokens(cfg, n: int) -> int:
    import block_m_crossing_sweep as SWEEP
    return SWEEP.tokens_for_rows(cfg, n * RK.BM)


def price(c: dict, device: str, dtype) -> dict:
    row = {}
    for C in C_MIB:
        b, sec, N = RK.simulate(c["arm"], c["counts"], c["tokens"], c["k"], c["P"], c["S"], c["R"], C,
                                c["shuffle_seed"], device=device, dtype=dtype)
        row[f"read_bytes_C{C:g}"] = b
        row[f"wall_s_C{C:g}"] = round(sec, 2)
        row["accesses"] = N
    return row


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--no-cpu-check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    dt = torch.float32 if a.device == "mps" else torch.float64
    cs, seed = cells()
    res, t_dev, t_cpu, worst = [], 0.0, 0.0, 0.0
    for c in cs:
        row = {k: v for k, v in c.items() if k not in ("counts",)}
        got = price(c, a.device, dt)
        row.update(got)
        t_dev += sum(got[f"wall_s_C{C:g}"] for C in C_MIB)
        if not a.no_cpu_check:
            ref = price(c, "cpu", torch.float64)
            row["cpu_float64"] = {f"read_bytes_C{C:g}": ref[f"read_bytes_C{C:g}"] for C in C_MIB}
            row["cpu_wall_s"] = round(sum(ref[f"wall_s_C{C:g}"] for C in C_MIB), 2)
            t_cpu += row["cpu_wall_s"]
            for C in C_MIB:
                worst = max(worst, abs(got[f"read_bytes_C{C:g}"] / ref[f"read_bytes_C{C:g}"] - 1))
        print(c["label"], c["n"], c["arm"], {k: v for k, v in got.items() if k.startswith("read")},
              f"{a.device} {sum(got[f'wall_s_C{C:g}'] for C in C_MIB):.1f} s",
              f"cpu {row.get('cpu_wall_s', '-')} s", flush=True)
        res.append(row)
    doc = {"what": "l2open rivals: rental 6's row-key LRU (scripts/scoring/rental6/rowkeys.py, unchanged) on every "
                   "w1 cell of the exact l2open page, priced before any rental-7 page",
           "model": MODEL, "page": str(PAGE.relative_to(REPO)), "shuffle_seed": seed, "C_MiB": list(C_MIB),
           "registered_C_MiB": REGISTERED_C_MIB, "rental6_registered_C_MiB": RK.C_MIB[0],
           "timing": {"sms": RK.SMS, "occ_w1": RK.OCC_W1, "cycles_per_kstep": RK.C_CYC, "sigma": 0.0},
           "device": a.device, "dtype": str(dt).replace("torch.", ""), "wall_s_device": round(t_dev, 1),
           "cpu_float64_check": (None if a.no_cpu_check else
                                 {"cells": len(res), "wall_s_cpu": round(t_cpu, 1), "max_rel_diff_read_bytes": worst}),
           "cells": res}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: {a.device} {t_dev:.1f} s, cpu float64 {t_cpu:.1f} s, max rel diff {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
