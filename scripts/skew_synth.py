#!/usr/bin/env python
"""Rental 5's SYNTHETIC skew histograms: Zipf or Dirichlet shapes fitted to the routing
traces' summary statistics, drawn deterministically from registered seeds (owner decision 3,
2026-10-07). Nothing here reads or writes a trace-derived count.

    python scripts/skew_synth.py --extract ROUTING_HIST_JSON      # once, off the repo (below)
    python scripts/skew_synth.py --fit                            # targets -> fitted shapes
    python scripts/skew_synth.py --draw                           # shapes -> histogram pages
    python scripts/skew_synth.py --check                          # recompute both, compare

WHAT IS COMMITTED, AND WHAT IS NOT. The routing logs (allenai/analysis_mixtral,
allenai/analysis_olmoe, tkj000/mmlu_Qwen1.5-MoE-A2.7B-Chat_token_patterns shards 0 and 1)
state no licence. Their per-batch expert histograms never enter this repository or the VM.
What enters is a table of SUMMARY STATISTICS of them, TARGETS (docs/registered/
2026-10-07-rental5-skew-hist/trace_stats.json): per (model, tread n, shape) the median over
batches of three statistics of the tokens-per-expert histogram at the batch B = tokens(n):
  cv        population std / mean of c_e
  max_mean  max c_e / mean c_e
  gini      the Gini coefficient of c_e
read off the reanalysis session's routing_hist.json (its derivation: scratchpad
reanalysis/work/routing_hist.py, numpy default_rng(0), prefill = B consecutive tokens of the
packed stream, decode = one token per row or a common decode step; at most 200 batches). The
three shapes per (model, n) are:
  PT  prefill, the median-CV layer at that B (the typical prefill layer)
  PW  prefill, the scheme's most skewed layer (Mixtral 12, OLMoE 15, Qwen1.5 23)
  DW  decode, the scheme's most skewed layer (Mixtral 11, OLMoE 15, Qwen1.5 21)
Qwen1.5's B (480, 960, 3840 tokens at n 1, 2, 8) are not on the reanalysis grid (64..8192 in
powers of 2): each statistic is interpolated linearly in log B between the two grid points
that bracket it, layer by layer, and PT's layer is chosen on the interpolated CV. That
interpolation is this tool's, stated here and in the targets file.

THE FIT (`--fit`). Each target is fitted by both of the harness's parametric kinds
(moe.routing.distributions: zipf, ranks^-s over a seed-permuted expert order; dirichlet,
normalised Gamma(alpha) draws), drawn by the harness's own sample_topk_ids at T = B tokens,
E experts, top-k, FIT_SEEDS seeds (0..15) per grid point; the loss is the sum over the three
statistics of log(median simulated / target)^2; zipf on ZIPF_GRID, dirichlet on
DIRICHLET_GRID (log spaced). The kind with the smaller loss is the shape. Every target's
per-statistic residual is recorded beside it ("which trace statistic each shape matches":
a statistic is MATCHED when its residual is within MATCH_TOL). Writes synth_fit.json.

THE DRAW (`--draw`). Per shape, the registered seed sequence DRAW_SEED_BASE + 1000 i + j
(i the shape's index in synth_fit.json, j = 0..63): the first draw whose three statistics are
each within MATCH_TOL of the TARGET is the histogram; with none, the draw of least loss (a
registered, deterministic rule). The counts are sample_topk_ids' bincount, so they are
feasible (no expert above T rows, sum T k). Pages (histogram-page/1, what
private_weight_reference --histogram reads):
  <model>-A.json   PT and PW (NATIVE and SHARED), the PW label permutations PW-hotfirst
                   (the hottest expert moved to expert 0) and PW-rand1 (a permutation from
                   PERM_SEED; NATIVE only: design-r5-review, the "which expert takes the
                   tokens" control), uniform (counts n BLOCK_M each, shuffled) and balanced
                   (R3's balanced_ids, counts null), at the model's three n
  <model>-B.json   DW, uniform, balanced (NATIVE and SHARED)
  <model>-C.json   the byte leg (Mixtral, OLMoE): PT, PW, uniform, balanced at n <= 9
                   (NATIVE and SHARED), plus OLMoE n 16 NATIVE only
  olmoe-1b-7b-c15.json  SHARED at 15 copies on PT PW DW uniform, NATIVE uniform, n 2 and 4
  olmoe-1b-7b-bk.json  balanced cells only (counts null), so the BK 128 pair (NATIVE, n 1..9)
                   can name its arms: R3 refuses --arms on a ladder page
Every page carries the shuffle seed SHUFFLE_SEED. The registration
(docs/registered/2026-10-07-rental5-skew-gh200) commits each page's sha256 and every count
vector's sha256; the pages themselves are committed openly (they are synthetic).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from moe.spec import MODEL_CONFIGS  # noqa: E402

OUT = REPO / "docs" / "registered" / "2026-10-07-rental5-skew-hist"
TARGETS = OUT / "trace_stats.json"
FIT = OUT / "synth_fit.json"
BLOCK_M = 32
#: the skew stage's treads per model (design-r5 2.2: n at which uniform lands on a tread)
TREADS = {"mixtral-8x7b": (2, 8, 32), "olmoe-1b-7b": (2, 4, 16), "qwen1.5-moe-a2.7b": (1, 2, 8)}
SHAPES = (("PT", "prefill", "typ"), ("PW", "prefill", "worst"), ("DW", "decode", "worst"))
STATS = ("cv", "max_mean", "gini")
GRID_B = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
FIT_SEEDS = 16
ZIPF_GRID = tuple(round(0.025 * i, 3) for i in range(0, 121))          # 0 .. 3
DIRICHLET_GRID = tuple(float(f"{v:.5g}") for v in np.geomspace(0.05, 500.0, 97))
MATCH_TOL = 0.10
DRAW_SEED_BASE = 20261007
DRAW_TRIES = 64
SHUFFLE_SEED = 20261007
PERM_SEED = 20261008
SCHEMA = "moe-kernels/histogram-page/1"


def tokens_for(model: str, n: int) -> int:
    cfg = MODEL_CONFIGS[model]
    t = n * BLOCK_M * cfg.num_experts / cfg.top_k
    if abs(t - round(t)) > 1e-9:
        raise ValueError(f"{model} n={n}: not a whole token count")
    return int(round(t))


def hist_stats(c) -> dict:
    c = np.asarray(c, dtype=float)
    E, tot = c.size, c.sum()
    mean = tot / E
    srt = np.sort(c)
    i = np.arange(1, E + 1)
    return {"cv": float(c.std() / mean), "max_mean": float(c.max() / mean),
            "gini": float(2 * (srt * i).sum() / (E * tot) - (E + 1) / E)}


def sha_counts(counts) -> str:
    return hashlib.sha256(json.dumps(None if counts is None else [int(v) for v in counts],
                                     separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------
# --extract: the one step that reads the reanalysis JSON (outside the repo)
# --------------------------------------------------------------------------

def _layer_stats(sch: dict, B: int, stat: str) -> np.ndarray:
    """Per-layer p50 of `stat` at batch B, interpolated in log B when B is off the grid."""
    if B in GRID_B:
        return np.array([x["p50"] for x in sch[str(B)]["per_layer"][stat]])
    lo = max(b for b in GRID_B if b < B)
    hi = min(b for b in GRID_B if b > B)
    a = np.array([x["p50"] for x in sch[str(lo)]["per_layer"][stat]])
    b = np.array([x["p50"] for x in sch[str(hi)]["per_layer"][stat]])
    w = (math.log(B) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return a + w * (b - a)


def extract(src: Path) -> dict:
    d = json.loads(Path(src).read_text())
    rows = []
    for model, ns in TREADS.items():
        m = d[model]
        for n in ns:
            B = tokens_for(model, n)
            for shape, scheme, which in SHAPES:
                sch = m["schemes"][scheme]
                cvs = _layer_stats(sch, B, "cv")
                layer = (int(np.argsort(cvs)[len(cvs) // 2]) if which == "typ"
                         else int(sch["most_skewed_layer"]))
                row = {"model": model, "n": n, "B": B, "shape": shape, "scheme": scheme,
                       "layer": layer, "layer_rule": ("the median-CV layer at this B" if which == "typ"
                                                      else "the scheme's most skewed layer"),
                       "B_on_grid": B in GRID_B}
                for s in STATS:
                    row[s] = round(float(_layer_stats(sch, B, s)[layer]), 6)
                rows.append(row)
    return {"what": "SUMMARY STATISTICS ONLY (medians over batches of the tokens-per-expert "
                    "histogram's cv, max/mean and Gini) of public routing logs that state no "
                    "licence; no count of any batch is here",
            "sources": {"mixtral-8x7b": "allenai/analysis_mixtral c4_results.jsonl (sha256 55c76a34...)",
                        "olmoe-1b-7b": "allenai/analysis_olmoe c4_results.jsonl (sha256 0c275e8c...)",
                        "qwen1.5-moe-a2.7b": "tkj000/mmlu_Qwen1.5-MoE-A2.7B-Chat_token_patterns shards 0, 1 (sha256 c1a5de24..., 0818222...)"},
            "derivation": "scratchpad reanalysis/work/routing_hist.py (2026-10-06): numpy default_rng(0); "
                          "prefill = B consecutive tokens of the packed stream, decode = one token per "
                          "row (Mixtral, OLMoE) or one decode step across prompts (Qwen1.5); up to 200 "
                          "batches; per layer the p50 over batches. Off-grid B (Qwen1.5's 480, 960, 3840) "
                          "interpolated linearly in log B between the bracketing grid points",
            "label": "SEEN (routing statistics; no GH200 page)", "targets": rows}


# --------------------------------------------------------------------------
# --fit and --draw
# --------------------------------------------------------------------------

def _draw(kind: str, param: float, model: str, B: int, seed: int) -> np.ndarray:
    import torch

    from moe.routing.distributions import sample_topk_ids
    from moe.spec import RoutingSpec
    cfg = MODEL_CONFIGS[model]
    ids = sample_topk_ids(RoutingSpec(kind, float(param)), B, cfg.num_experts, cfg.top_k,
                          seed=int(seed), device="cpu")
    return torch.bincount(ids.reshape(-1).to(torch.int64), minlength=cfg.num_experts).numpy()


def _loss(st: dict, tgt: dict) -> float:
    return sum(math.log(st[s] / tgt[s]) ** 2 for s in STATS)


def _sim(kind, param, model, B) -> dict:
    sts = [hist_stats(_draw(kind, param, model, B, seed)) for seed in range(FIT_SEEDS)]
    return {s: float(np.median([x[s] for x in sts])) for s in STATS}


def fit_one(t: dict) -> dict:
    best = {}
    for kind, grid in (("zipf", ZIPF_GRID), ("dirichlet", DIRICHLET_GRID)):
        cur = None
        for p in grid:
            if kind == "dirichlet" and p <= 0:
                continue
            sm = _sim(kind, p, t["model"], t["B"])
            L = _loss(sm, t)
            if cur is None or L < cur[0] - 1e-12:
                cur = (L, p, sm)
        best[kind] = {"param": cur[1], "loss": round(cur[0], 6),
                      "sim_median": {s: round(v, 6) for s, v in cur[2].items()}}
    kind = min(best, key=lambda k: best[k]["loss"])
    sm = best[kind]["sim_median"]
    resid = {s: round(sm[s] / t[s] - 1, 4) for s in STATS}
    return {"model": t["model"], "n": t["n"], "B": t["B"], "shape": t["shape"], "kind": kind,
            "param": best[kind]["param"], "loss": best[kind]["loss"], "both": best,
            "target": {s: t[s] for s in STATS}, "fit_residual": resid,
            "matches": [s for s in STATS if abs(resid[s]) <= MATCH_TOL],
            "trace_statistic": f"{t['scheme']} layer {t['layer']} ({t['layer_rule']}) at B {t['B']}"
                               + ("" if t["B_on_grid"] else " (interpolated in log B)")}


def fit(targets: dict) -> dict:
    return {"what": "Zipf or Dirichlet shapes fitted to the trace statistics (scripts/skew_synth.py --fit)",
            "kinds": "moe.routing.distributions: zipf (exponent s) and dirichlet (alpha), drawn by sample_topk_ids",
            "fit_seeds": FIT_SEEDS, "zipf_grid": [ZIPF_GRID[0], ZIPF_GRID[-1], len(ZIPF_GRID)],
            "dirichlet_grid": [DIRICHLET_GRID[0], DIRICHLET_GRID[-1], len(DIRICHLET_GRID)],
            "loss": "sum over cv, max_mean, gini of log(median simulated / target)^2",
            "match_tol": MATCH_TOL, "shapes": [fit_one(t) for t in targets["targets"]]}


def draw_one(i: int, f: dict) -> dict:
    best = None
    for j in range(DRAW_TRIES):
        seed = DRAW_SEED_BASE + 1000 * i + j
        c = _draw(f["kind"], f["param"], f["model"], f["B"], seed)
        st = hist_stats(c)
        ok = all(abs(st[s] / f["target"][s] - 1) <= MATCH_TOL for s in STATS)
        L = _loss(st, f["target"])
        if ok:
            best = (seed, c, st, True)
            break
        if best is None or L < _loss(best[2], f["target"]):
            best = (seed, c, st, False)
    seed, c, st, ok = best
    return {"seed": seed, "counts": [int(v) for v in c], "stats": {s: round(v, 6) for s, v in st.items()},
            "within_tol": ok, "draw_residual": {s: round(st[s] / f["target"][s] - 1, 4) for s in STATS}}


def _cell(label, n, counts, arms=None, **prov):
    c = {"label": label, "n": n, "counts": None if counts is None else [int(v) for v in counts]}
    if arms:
        c["arms"] = list(arms)
    if prov:
        c["provenance"] = prov
    return c


def pages(fitd: dict) -> dict:
    """Every histogram page, from the fitted shapes (draws included)."""
    by = {}
    for i, f in enumerate(fitd["shapes"]):
        d = draw_one(i, f)
        by[(f["model"], f["n"], f["shape"])] = (f, d)
    rng = np.random.default_rng(PERM_SEED)
    out = {}
    both = ("native", "shared")
    for model, ns in TREADS.items():
        E = MODEL_CONFIGS[model].num_experts
        A, Bp, C = [], [], []
        for n in ns:
            uni = [n * BLOCK_M] * E

            def prov(shape):
                f, d = by[(model, n, shape)]
                return dict(kind=f["kind"], param=f["param"], seed=d["seed"], stats=d["stats"],
                            trace_statistic=f["trace_statistic"], matches=f["matches"])
            pt = by[(model, n, "PT")][1]["counts"]
            pw = by[(model, n, "PW")][1]["counts"]
            dw = by[(model, n, "DW")][1]["counts"]
            hot = int(np.argmax(pw))
            hotfirst = [pw[hot]] + [pw[e] for e in range(E) if e != hot]
            perm = rng.permutation(E)
            rand1 = [pw[int(e)] for e in perm]
            A += [_cell("PT", n, pt, both, **prov("PT")), _cell("PW", n, pw, both, **prov("PW")),
                  _cell("PW-hotfirst", n, hotfirst, ("native",), of="PW", permutation="the hottest expert moved to expert 0"),
                  _cell("PW-rand1", n, rand1, ("native",), of="PW", permutation=f"numpy default_rng({PERM_SEED}).permutation, the {ns.index(n) + 1}th draw"),
                  _cell("uniform", n, uni, both), _cell("balanced", n, None, both)]
            Bp += [_cell("DW", n, dw, both, **prov("DW")), _cell("uniform", n, uni, both),
                   _cell("balanced", n, None, both)]
            if model != "qwen1.5-moe-a2.7b":
                arms = both if n <= 9 else ("native",)
                if n <= 16:
                    C += [_cell("PT", n, pt, arms, **prov("PT")), _cell("PW", n, pw, arms, **prov("PW")),
                          _cell("uniform", n, uni, arms), _cell("balanced", n, None, arms)]
        for page, cells in (("A", A), ("B", Bp), ("C", C)):
            if cells:
                out[f"{model}-{page}"] = {"schema": SCHEMA, "model": model, "page": page,
                                          "shuffle_seed": SHUFFLE_SEED, "cells": cells,
                                          "synthetic": "fitted Zipf/Dirichlet shapes (scripts/skew_synth.py); no trace count"}
    m = "olmoe-1b-7b"
    cells = []
    for n in (2, 4):
        E = MODEL_CONFIGS[m].num_experts
        for shape in ("PT", "PW", "DW"):
            f, d = by[(m, n, shape)]
            cells.append(_cell(shape, n, d["counts"], ("shared",), kind=f["kind"], param=f["param"], seed=d["seed"]))
        cells.append(_cell("uniform", n, [n * BLOCK_M] * E, ("native", "shared")))
    out[f"{m}-c15"] = {"schema": SCHEMA, "model": m, "page": "c15", "shuffle_seed": SHUFFLE_SEED,
                       "cells": cells, "synthetic": "fitted Zipf/Dirichlet shapes (scripts/skew_synth.py); no trace count"}
    # balanced-only pages: R3's own balanced cells through the histogram path, so a unit can
    # name its arms (R3 refuses --arms on a ladder page): the BK 128 pair (NATIVE, n 1..9)
    for name, model, ns, arms in (("olmoe-1b-7b-bk", "olmoe-1b-7b", range(1, 10), ("native",)),):
        out[name] = {"schema": SCHEMA, "model": model, "page": name.split("-")[-1], "shuffle_seed": SHUFFLE_SEED,
                     "cells": [_cell("balanced", n, None, arms) for n in ns],
                     "synthetic": "balanced cells only (R3's balanced_ids, counts null); no histogram"}
    return out


def dump(d) -> str:
    return json.dumps(d, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extract", type=Path)
    g.add_argument("--fit", action="store_true")
    g.add_argument("--draw", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.extract:
        TARGETS.write_text(dump(extract(a.extract)))
        print(f"wrote {TARGETS}")
        return 0
    targets = json.loads(TARGETS.read_text())
    if a.fit:
        FIT.write_text(dump(fit(targets)))
        print(f"wrote {FIT}")
        return 0
    if a.draw:
        for name, doc in pages(json.loads(FIT.read_text())).items():
            (OUT / f"{name}.json").write_text(dump(doc))
            print(f"wrote {OUT / name}.json")
        return 0
    bad = 0
    got = dump(fit(targets))
    same = got == FIT.read_text()
    print(f"{FIT.name}: {'same' if same else 'DIFFERS'}")
    bad += not same
    for name, doc in pages(json.loads(FIT.read_text())).items():
        p = OUT / f"{name}.json"
        same = p.exists() and p.read_text() == dump(doc)
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
        bad += not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
