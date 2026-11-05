"""Shared harness for addendum 5's build records (scripts/scoring/decode/d5_build): CAL2 under a v5d variant (the
addendum-3 stop rule, the CAL constants refitted), and beside it, where asked, the SEEN decode blind re-scored at that
refit (POST-HOC on seen pages, never a choice). Every input is a committed file: DS's readings, the published CAL2 and
decode blind trees, the registered pages. Output lines are deterministic (no timings) so run.py --check can compare."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEC = HERE.parent
for _p in (str(DEC), str(DEC.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402

V5D = D5.V5D
REPO = DC.REPO
CAL2 = REPO / D4.CAL2_PUBLISHED / "results"
BLIND = REPO / D4.BLIND_PUBLISHED / "results"
RD_ = D5.committed_readings()
BASE = D5.terms_from(RD_)
_TP = None
SHORT = {"phi-3.5-moe": "Phi", "qwen1.5-moe-a2.7b": "Qwen1.5", "jetmoe-8b": "JetMoE", "deepseek-v2-lite": "deepseek",
         "granite-3.0-3b-a800m": "Granite-3B"}


def tile():
    global _TP
    if _TP is None:
        _TP = D4.tile_source()
    return _TP


def cal(terms):
    import score_cal_d3 as SC3
    tp, src = tile()
    with D5.v5d(terms):
        return SC3.score(CAL2, DC.registration(REPO, "cal"), tp, src, settle=False)


def rescore(terms, c):
    import gate_d5 as G
    return G.rescore(terms, c, BLIND)


def run(terms, label, blind=True, ctx=contextlib.nullcontext):
    with ctx():
        c = cal(terms)
        st, e1 = c["STOP"], c["E1_CAL_IN_SAMPLE"]["stats"]
        sel = c["FIT"]["selected"]
        fit = c["FIT"]["forms"][sel] if sel else {}
        o = {"label": label, "CAL": st["verdict"], "rms": e1["rms"], "worst": e1["worst"], "mean": e1["mean"],
             "cells": e1["cells"], "T0_us": (fit.get("T0_GR_ms") or 0) * 1e3,
             "sn_us": (fit.get("s_naive_GR_ms") or 0) * 1e3, "sb_us": (fit.get("s_block_GR_ms") or 0) * 1e3}
        if blind and sel:
            e = rescore(terms, c)["E1_DEC"]
            o["pooled"] = e["pooled"]["stats"]
            o["per_model"] = {m: x["stats"] for m, x in e["per_model"].items()}
    return o


def line(o) -> str:
    s = (f"{o['label']:34s} CAL {o['CAL']:4s} rms {o['rms']:.4f} worst {o['worst']:+.4f} mean {o['mean']:+.4f} "
         f"n {o['cells']} T0 {o['T0_us']:.2f} sn {o['sn_us']:.2f} sb {o['sb_us']:.2f}")
    if "pooled" in o:
        p = o["pooled"]
        s += (f" | SEEN blind (POST-HOC) pooled {p['rms']:.4f} n {p['cells']} mean {p['mean']:+.4f} | "
              + " ".join(f"{SHORT[m]} {x['rms']:.4f}/{x['worst']:+.4f}/{x['mean']:+.4f}" for m, x in o["per_model"].items()))
    return s


def emit(lines: list[str], name: str, argv) -> int:
    """Print the lines; with --write save them as <name>.out beside; with --check compare (exit 1 on a difference)."""
    text = "\n".join(lines) + "\n"
    out = HERE / f"{name}.out"
    a = list(argv)
    if "--write" in a:
        out.write_text(text)
    if "--check" in a:
        same = out.exists() and out.read_text() == text
        print(f"{out.name}: {'same' if same else 'DIFFERS'}")
        return 0 if same else 1
    sys.stdout.write(text)
    return 0
