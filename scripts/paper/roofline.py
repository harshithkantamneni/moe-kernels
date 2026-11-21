#!/usr/bin/env python
# ruff: noqa: E501  (published page paths and scorer keys are quoted whole)
"""POST-HOC BASELINE, not a registered prediction: a whole-call roofline priced on
the same held-out cells as the registered time tests (OUTLINE gap 6.14).

    python scripts/paper/roofline.py [--out DIR]   # writes DIR/roofline.md, DIR/roofline.csv,
                                                   # DIR/data/roofline_cells.csv

Written 2026-10-05, after every page it prices; nothing in it was registered, and
nothing is fitted. Per cell (arm, G, n):

    T_roof = max(FLOPs / peak, bytes / bandwidth)

FLOPs     both GEMMs of the call, 2 x rows x (H x 2F + F x H), rows = E x n x BLOCK_M
          (R3's balanced routing fills every M-tile; `tokens x top_k` on the pages).
peak      the target session's published bf16 dense ceiling and
bandwidth its triad bandwidth: results/published/<session>/calibration/
          measured_nvidia_gh200_480gb.yaml, `compute_dense_tflops.bf16` and
          `memory.bandwidth_tb_s` (the per-card constants each session measured).
bytes     two variants, each printed:
          counted    the target's lock-1710 counter page at that G,
                     `per_call.dram_bytes_read` (reads, as the timing model prices),
                     and separately reads + `per_call.dram_bytes_write`;
          predicted  the byte model's registered q per GEMM (the registration JSON's
                     cell `q`, MIX view), as bytes q x W_g + n x operand_g
                     (`dram_counter_route.r3_q` inverted, `r3_byte_model`).
cells     exactly the cells each registered time test scored (SHARED and PRIVATE),
          with their measured ms, from the committed scorer outputs in
          scripts/scoring/crossmodel/ (the same committed timed pages).

The roofline has no intercept, no launch or host term and no price for the
call's non-GEMM kernels (alignment, silu_and_mul, moe_sum), so it sits under
the measured call by construction; the table prints it whatever it shows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

XM = ROOT / "scripts/scoring/crossmodel"
REG = ROOT / "docs/registered"
PUB = ROOT / "results/published"
BLOCK_M = 32
SCORED = ("shared", "private")
LIMIT = 0.05
LABEL = "POST-HOC baseline (not registered)"

#: (test, model, committed scorer output, registration, bytes the model priced from)
TESTS = [
    ("Mixtral 8x22B", "mixtral-8x22b", "8x22b.registered.json",
     "2026-09-27-mixtral-8x22b-gh200.json", "predicted"),
    ("Qwen2-57B-A14B", "qwen2-57b-a14b", "qwen2.registered.json",
     "2026-09-28-qwen2-57b-a14b-gh200.json", "predicted"),
    ("OLMoE-1B-7B", "olmoe-1b-7b", "olmoe-1b-7b.score.json",
     "2026-09-29-olmoe-1b-7b-gh200.json", "counted"),
    ("Qwen1.5-MoE-A2.7B", "qwen1.5-moe-a2.7b", "qwen1.5-moe-a2.7b.score.json",
     "2026-09-29-qwen1.5-moe-a2.7b-gh200.json", "counted"),
    ("Phi-3.5-MoE", "phi-3.5-moe", "phi-3.5-moe.score.json",
     "2026-09-29-phi-3.5-moe-gh200.json", "counted"),
    ("JetMoE-8B", "jetmoe-8b", "jetmoe-8b.score.json", "2026-09-29-jetmoe-8b-gh200.json",
     "counted"),
]


def _session(f: str, d: dict) -> Path:
    if "session" in d:
        return ROOT / d["session"]
    man = json.loads((XM / "MANIFEST.json").read_text())
    run = next(r for r in man["runs"] if f"scripts/scoring/crossmodel/{f}" in r["output"])
    cnt = run["argv"][run["argv"].index("--target-counters") + 1]
    return ROOT / cnt.split("/results/")[0]


def _cells(d: dict) -> list[dict]:
    if "time_predicted_bytes" in d:
        return [dict(c, model_ms=c["registered_ms"]) for c in d["time_predicted_bytes"]["cells"]
                if c["arm"] in SCORED]
    return [dict(c, model_ms=c["predicted_ms"]) for c in d["cells"] if c["arm"] in SCORED]


def card(session: Path) -> dict:
    y = yaml.safe_load((session / "calibration/measured_nvidia_gh200_480gb.yaml").read_text())
    return {"peak_tflops": float(y["compute_dense_tflops"]["bf16"]),
            "bw_tb_s": float(y["memory"]["bandwidth_tb_s"])}


def counted(session: Path) -> dict:
    """{(arm, G, n): (reads, writes)} per call from the lock-1710 counter pages."""
    out = {}
    for page in sorted(session.glob("results/*-r3-counters/lock1710/r3c-g*.json")):
        p = json.loads(page.read_text())
        G = int(p["design"]["group_m"])
        for c in p["cells"]:
            pc = c["per_call"]
            out[(c["arm"], G, int(c["n"]))] = (float(pc["dram_bytes_read"]),
                                               float(pc["dram_bytes_write"]))
    return out


def predicted(model: str, reg: str) -> dict:
    """{(arm, G, n): bytes read per call} from the registration's q, MIX view."""
    import dram_counter_route as DCR

    from moe.spec import MODEL_CONFIGS
    bm = DCR.r3_byte_model(MODEL_CONFIGS[model], "bf16", BLOCK_M)
    W = {"w1": bm["W_w1"], "w2": bm["W_w2"]}
    op = {"w1": bm["operand_per_tile_w1"], "w2": bm["operand_per_tile_w2"]}
    out = {}
    for c in json.loads((REG / reg).read_text())["cells"]:
        q = c.get("q") or {}
        if all(q.get(g) is not None for g in W):
            out[(c["arm"], int(c["G"]), int(c["n"]))] = sum(
                q[g] * W[g] + int(c["n"]) * op[g] for g in W)
    return out


def flops(model: str, n: int) -> float:
    from moe.spec import MODEL_CONFIGS
    c = MODEL_CONFIGS[model]
    rows = c.num_experts * n * BLOCK_M
    H, F = c.hidden_size, c.intermediate_size
    return 2.0 * rows * (H * 2 * F + F * H)


def _stats(v: list[float]) -> dict:
    if not v:
        return {"cells": 0, "rms": None, "worst": None, "beyond_5pct": 0, "median": None}
    return {"cells": len(v), "rms": math.sqrt(sum(x * x for x in v) / len(v)),
            "worst": max(v, key=abs), "beyond_5pct": sum(abs(x) > LIMIT for x in v),
            "median": statistics.median(v)}


def build() -> dict:
    tests, cells_out = [], []
    for test, model, f, reg, kind in TESTS:
        d = json.loads((XM / f).read_text())
        ses = _session(f, d)
        k = card(ses)
        cb, pb = counted(ses), predicted(model, reg)
        peak, bw = k["peak_tflops"] * 1e12, k["bw_tb_s"] * 1e12
        res = {"model": [], "roof_counted": [], "roof_counted_rw": [], "roof_predicted": []}
        compute_bound = 0
        closer = {"roof_counted": 0, "roof_predicted": 0}
        for c in _cells(d):
            key = (c["arm"], int(c["G"]), int(c["n"]))
            fl = flops(model, key[2])
            t_c = fl / peak * 1e3
            meas = float(c["measured_ms"])
            m_res = c["model_ms"] / meas - 1
            res["model"].append(m_res)
            row = {"test": test, "arm": key[0], "G": key[1], "n": key[2], "measured_ms": meas,
                   "model_ms": c["model_ms"], "flops_ms": t_c}
            if key in cb:
                r, w = cb[key]
                rc, rcw = max(t_c, r / bw * 1e3), max(t_c, (r + w) / bw * 1e3)
                compute_bound += t_c >= r / bw * 1e3
                res["roof_counted"].append(rc / meas - 1)
                closer["roof_counted"] += abs(rc / meas - 1) < abs(m_res)
                res["roof_counted_rw"].append(rcw / meas - 1)
                row.update(counted_bytes=r, roof_counted_ms=rc, roof_counted_rw_ms=rcw)
            if key in pb:
                rp = max(t_c, pb[key] / bw * 1e3)
                res["roof_predicted"].append(rp / meas - 1)
                closer["roof_predicted"] += abs(rp / meas - 1) < abs(m_res)
                row.update(predicted_bytes=pb[key], roof_predicted_ms=rp)
            cells_out.append(row)
        tests.append({"test": test, "model_bytes": kind, "session": str(ses.relative_to(ROOT)),
                      **k, "compute_bound_cells": compute_bound,
                      "closer_counted": closer["roof_counted"],
                      "closer_predicted": closer["roof_predicted"],
                      **{name: _stats(v) for name, v in res.items()}})
    return {"label": LABEL, "tests": tests, "cells": cells_out}


def _p(x):
    return "" if x is None else f"{100 * x:+.2f}%" if x < 0 or x > 0 else "0.00%"


def _r(x):
    return "" if x is None else f"{100 * x:.2f}%"


HEAD = ["test", "model priced from", "cells", "model rms", "model beyond 5%",
        "roofline (counted reads) rms", "worst", "median", "beyond 5%",
        "roofline (counted reads + writes) rms", "roofline (predicted bytes) rms",
        "predicted: worst", "predicted: beyond 5%",
        "cells where the roofline is closer than the model (counted / predicted)",
        "compute-bound cells", "peak TFLOPS",
        "bandwidth TB/s"]


def table_rows(b: dict) -> list[list]:
    rows = []
    for t in b["tests"]:
        m, c, cw, p = t["model"], t["roof_counted"], t["roof_counted_rw"], t["roof_predicted"]
        rows.append([t["test"], f"{t['model_bytes']} bytes", m["cells"], _r(m["rms"]),
                     m["beyond_5pct"], _r(c["rms"]), _p(c["worst"]), _p(c["median"]),
                     c["beyond_5pct"], _r(cw["rms"]), _r(p["rms"]), _p(p["worst"]),
                     p["beyond_5pct"], f"{t['closer_counted']} / {t['closer_predicted']}",
                     t["compute_bound_cells"], f"{t['peak_tflops']:.1f}",
                     f"{t['bw_tb_s']:.4f}"])
    return rows


def write(b: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = table_rows(b)
    with (out / "roofline.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEAD)
        w.writerows(rows)
    ccols = ["test", "arm", "G", "n", "measured_ms", "model_ms", "flops_ms", "counted_bytes",
             "roof_counted_ms", "roof_counted_rw_ms", "predicted_bytes", "roof_predicted_ms"]
    (out / "data").mkdir(parents=True, exist_ok=True)
    with (out / "data/roofline_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(ccols)
        for r in b["cells"]:
            w.writerow([f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, "")
                        for k in ccols])
    md = [f"# Whole-call roofline: {LABEL}", "",
          "Generated by `python scripts/paper/roofline.py`. POST-HOC: written 2026-10-05, after "
          "every page it prices; a baseline, not a registered prediction, and it scores nothing. "
          "T_roof = max(FLOPs / peak, bytes / bandwidth) per cell, on the cells each registered "
          "time test scored (SHARED and PRIVATE), with their measured ms from the committed "
          "scorer outputs in `scripts/scoring/crossmodel/`. Peak and bandwidth: each target "
          "session's `calibration/measured_nvidia_gh200_480gb.yaml`. Residual = T_roof / "
          "measured - 1. Per-cell values: `docs/paper/data/roofline_cells.csv`.", "",
          "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    md += ["", "The roofline has no intercept and prices neither the non-GEMM kernels nor launch "
           "or host time, so a negative median is the expected sign. The model's columns are "
           "the registered scores, copied from the same committed outputs."]
    (out / "roofline.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper")
    a = ap.parse_args(argv)
    write(build(), a.out)
    print(f"roofline ({LABEL}): written under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
