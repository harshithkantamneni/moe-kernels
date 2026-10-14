#!/usr/bin/env python
"""The decode stage's SYNTHETIC pages: the (T, coverage) ladder, the controls, G3-D's pairs and the
trace-like draws (design-decode sections 2 and 6; owner decision DD3 of 2026-10-14). Nothing here
reads or writes a trace-derived count.

    python scripts/decode_synth.py --draw          # every page and the coverage table
    python scripts/decode_synth.py --check         # recompute both, compare byte for byte
    python scripts/decode_synth.py --extract LOG.jsonl --model M --out STATS.json   # OFF THE REPO

THE CELL IS (T, counts). T fixes vLLM's tile (default_config); the coverage A = #{c_e > 0} sets the
weight bytes and, with BLOCK_N, the live CTAs. Per model and T, five kinds of cell (decode-page/1):
  L-T<T>-A<A>   the COVERAGE LADDER: LADDER_LEVELS coverage levels A from k to min(E, T k), each
                balanced over its active set (counts floor or ceil of T k / A; feasible because
                A >= k), the active set the first A experts of the page's registered permutation
                (PERM_SEED). The top level is role BALANCED (R3's balanced_ids' histogram).
  MIN-T<T>      MINCOVER: every token to experts 0..k-1 (c_e = T, A = k; multi-tile past BM).
                At G3_LABEL_TS it is G3-D label's twin of L-T<T>-A<k>: the same count multiset on
                another active set.
  U-T<T>        UNIFORM: sample_topk_ids(uniform) at seed UNIFORM_SEED + T.
  US-T<T>       the unshuffled twin of U-T<T> at G3_SHUFFLE_TS (G3-D shuffle): shuffle false.
  DW-T<T>       the trace-like draw (models with a rental-5 DW shape, at TRACE_TS): see below.
G3-D fill's pairs are ladder cells already: one A at two T in one tile band (FILL_PAIRS).

THE PAGES (docs/registered/2026-10-14-decode-pages/<model>-<kind>.json):
  main   every cell above (GRR, the E1-DEC page)
  ctl    BALANCED, the ladder bottom and UNIFORM at CTL_TS (the GR1 instrument page, GD2; and the
         counter route's page)
  e240   BALANCED and UNIFORM at E240_TS (Mixtral only: GD3, cells GPU-bound eager)
  decl   BALANCED at T >= BLOCK_M, arms native and decl, decl_copies DECL_COPIES (CAL only)

THE TRACE-LIKE DRAWS: the FALLBACK of design-decode section 6, by necessity. The primary was a new
off-repo extraction of SUMMARY STATISTICS from rental 5's three routing logs at B = 1..128; on
2026-10-14 those logs are gone (scratchpad reanalysis/data/{mixtral,olmoe,qwen} are empty, no copy
under moe-kernels-work, and traces/mixtral-8x7b-chat-decode.npz holds per-batch sums over 32 decode
steps, not per-step batches). So the draws use rental 5's FITTED DW shapes (synth_fit.json, fitted
at B >= 256), EXTRAPOLATED to decode B: the first draw at seed DRAW_SEED_BASE + 1000 i + j whose
coverage is within MATCH_TOL (relative) of the shape's simulated median coverage at that T, else
the nearest. Labelled EXTRAPOLATED wherever printed. `--extract` exists for when the logs return
(the owner's decision DD3's primary); it is tested on synthetic logs only.

THE COVERAGE TABLE (coverage_stats.json): per model and B in 1..128, uniform and (where fitted) DW
EXTRAPOLATED: median, p10 and p90 of coverage A/E, the empty-expert fraction 1 - A/E and
P(max c_e > BM(B)), over COVERAGE_SEEDS draws. Summary statistics only.
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
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe.spec import MODEL_CONFIGS  # noqa: E402

OUT = REPO / "docs" / "registered" / "2026-10-14-decode-pages"
SYNTH_FIT = REPO / "docs" / "registered" / "2026-10-07-rental5-skew-hist" / "synth_fit.json"
SCHEMA = "moe-kernels/decode-page/1"
CAL_MODELS = ("olmoe-1b-7b", "mixtral-8x7b")
BLIND_MODELS = ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b", "deepseek-v2-lite", "granite-3.0-3b-a800m")
MODELS = CAL_MODELS + BLIND_MODELS
BASE_TS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
LADDER_LEVELS = 4
G3_LABEL_TS = (4, 16, 64)
G3_SHUFFLE_TS = (8, 32, 128)
TRACE_TS = (4, 8, 16, 32)
CTL_TS = BASE_TS
E240_TS = (4, 8, 16, 32, 64, 96, 128)
DECL_COPIES = 9
FILL_PAIRS = ((8, 16), (16, 32))
SHUFFLE_SEED = 20261014
PERM_SEED = 20261015
UNIFORM_SEED = 20261016
DRAW_SEED_BASE = 20261017
DRAW_TRIES = 64
MATCH_TOL = 0.10
COVERAGE_SEEDS = 200
COVERAGE_BS = (1, 2, 4, 8, 16, 32, 64, 128)
CTL_MODELS_BLIND = ("qwen1.5-moe-a2.7b", "granite-3.0-3b-a800m")


def bm_of(T: int, E: int) -> int:
    from moe.bench.tile_resolve import default_config
    return int(default_config(int(T), int(E))["BLOCK_SIZE_M"])


def naive_max_T(model: str) -> int:
    cfg = MODEL_CONFIGS[model]
    return max((T for T in range(1, 129) if T * cfg.top_k * 4 <= cfg.num_experts), default=0)


def ts_of(model: str) -> tuple:
    """BASE_TS, plus the model's deepest naive T when it is not already there (Qwen1.5's T 3)."""
    n = naive_max_T(model)
    return tuple(sorted(set(BASE_TS) | ({n} if n else set())))


def perm_of(model: str) -> list[int]:
    cfg = MODEL_CONFIGS[model]
    i = MODELS.index(model) if model in MODELS else 99
    rng = np.random.default_rng(PERM_SEED + i)
    return [int(v) for v in rng.permutation(cfg.num_experts)]


def levels(model: str, T: int) -> list[int]:
    cfg = MODEL_CONFIGS[model]
    amax = min(cfg.num_experts, T * cfg.top_k)
    return sorted({int(round(v)) for v in np.linspace(cfg.top_k, amax, LADDER_LEVELS)})


def balanced_over(model: str, T: int, active: list[int]) -> list[int]:
    cfg = MODEL_CONFIGS[model]
    numel, A = T * cfg.top_k, len(active)
    c = [0] * cfg.num_experts
    for i, e in enumerate(active):
        c[e] = numel // A + (1 if i < numel % A else 0)
    return c


def mincover(model: str, T: int) -> list[int]:
    cfg = MODEL_CONFIGS[model]
    return [T if e < cfg.top_k else 0 for e in range(cfg.num_experts)]


def _bincount(ids, E: int) -> list[int]:
    import torch
    return torch.bincount(ids.reshape(-1).to(torch.int64), minlength=E).tolist()


def uniform_draw(model: str, T: int) -> list[int]:
    from moe.routing.distributions import sample_topk_ids
    from moe.spec import RoutingSpec
    cfg = MODEL_CONFIGS[model]
    ids = sample_topk_ids(RoutingSpec("uniform", 0.0), T, cfg.num_experts, cfg.top_k, seed=UNIFORM_SEED + T)
    return _bincount(ids, cfg.num_experts)


def dw_shapes() -> dict:
    """Rental 5's fitted DW shape per model, the smallest-B fit (synth_fit.json)."""
    fit = json.loads(SYNTH_FIT.read_text())
    out = {}
    for f in fit["shapes"]:
        if f["shape"] == "DW" and (f["model"] not in out or f["B"] < out[f["model"]]["B"]):
            out[f["model"]] = {"kind": f["kind"], "param": f["param"], "B": f["B"]}
    return out


def _cov_draws(model: str, kind: str, param: float, T: int, seeds) -> list[list[int]]:
    from moe.routing.distributions import sample_topk_ids
    from moe.spec import RoutingSpec
    cfg = MODEL_CONFIGS[model]
    return [_bincount(sample_topk_ids(RoutingSpec(kind, float(param)), T, cfg.num_experts, cfg.top_k, seed=int(s)),
                      cfg.num_experts) for s in seeds]


def coverage_target(model: str, kind: str, param: float, T: int) -> float:
    E = MODEL_CONFIGS[model].num_experts
    cs = _cov_draws(model, kind, param, T, range(COVERAGE_SEEDS))
    return float(np.median([sum(1 for v in c if v) / E for c in cs]))


def trace_draw(model: str, T: int, shape: dict, i: int) -> dict:
    """The first draw whose coverage is within MATCH_TOL (relative) of the shape's median coverage
    at T; with none, the nearest (registered, deterministic)."""
    E = MODEL_CONFIGS[model].num_experts
    tgt = coverage_target(model, shape["kind"], shape["param"], T)
    best = None
    seeds = [DRAW_SEED_BASE + 1000 * i + j for j in range(DRAW_TRIES)]
    for s, c in zip(seeds, _cov_draws(model, shape["kind"], shape["param"], T, seeds)):
        cov = sum(1 for v in c if v) / E
        err = abs(cov / tgt - 1)
        if err <= MATCH_TOL:
            return {"counts": c, "seed": s, "coverage": cov, "target": tgt, "matched": True}
        if best is None or err < best[0]:
            best = (err, {"counts": c, "seed": s, "coverage": cov, "target": tgt, "matched": False})
    return best[1]


def _cell(label, T, counts, role, arms=("native",), shuffle=True, **prov) -> dict:
    d = {"label": label, "T": int(T), "counts": [int(v) for v in counts], "arms": list(arms), "role": role}
    if not shuffle:
        d["shuffle"] = False
    d.update(prov)
    return d


def main_cells(model: str) -> list[dict]:
    cfg = MODEL_CONFIGS[model]
    perm = perm_of(model)
    shapes = dw_shapes()
    out = []
    for T in ts_of(model):
        lv = levels(model, T)
        for A in lv:
            role = "BALANCED" if A == min(cfg.num_experts, T * cfg.top_k) else ("LADDER-BOTTOM" if A == cfg.top_k else "LADDER")
            out.append(_cell(f"L-T{T}-A{A}", T, balanced_over(model, T, perm[:A]), role))
        if T in G3_LABEL_TS:
            out.append(_cell(f"MIN-T{T}", T, mincover(model, T), "MINCOVER", pair=f"L-T{T}-A{cfg.top_k}"))
        u = uniform_draw(model, T)
        out.append(_cell(f"U-T{T}", T, u, "UNIFORM", seed=UNIFORM_SEED + T))
        if T in G3_SHUFFLE_TS:
            out.append(_cell(f"US-T{T}", T, u, "UNIFORM-UNSHUFFLED", shuffle=False, pair=f"U-T{T}"))
        if model in shapes and T in TRACE_TS:
            sh = shapes[model]
            d = trace_draw(model, T, sh, TRACE_TS.index(T))
            out.append(_cell(f"DW-T{T}", T, d["counts"], "TRACE-DW EXTRAPOLATED", seed=d["seed"],
                             coverage=round(d["coverage"], 4), coverage_target=round(d["target"], 4),
                             matched=d["matched"], shape=f"{sh['kind']} {sh['param']} (fitted at B {sh['B']})"))
    return out


def ctl_cells(model: str) -> list[dict]:
    cfg = MODEL_CONFIGS[model]
    perm = perm_of(model)
    out = []
    for T in CTL_TS:
        top = min(cfg.num_experts, T * cfg.top_k)
        out.append(_cell(f"L-T{T}-A{top}", T, balanced_over(model, T, perm[:top]), "BALANCED"))
        if cfg.top_k != top:
            out.append(_cell(f"L-T{T}-A{cfg.top_k}", T, balanced_over(model, T, perm[:cfg.top_k]), "LADDER-BOTTOM"))
        out.append(_cell(f"U-T{T}", T, uniform_draw(model, T), "UNIFORM", seed=UNIFORM_SEED + T))
    return out


def e240_cells(model: str) -> list[dict]:
    cfg = MODEL_CONFIGS[model]
    perm = perm_of(model)
    out = []
    for T in E240_TS:
        top = min(cfg.num_experts, T * cfg.top_k)
        out.append(_cell(f"L-T{T}-A{top}", T, balanced_over(model, T, perm[:top]), "BALANCED"))
        out.append(_cell(f"U-T{T}", T, uniform_draw(model, T), "UNIFORM", seed=UNIFORM_SEED + T))
    return out


def decl_cells(model: str) -> list[dict]:
    cfg = MODEL_CONFIGS[model]
    perm = perm_of(model)
    out = []
    for T in BASE_TS:
        if T < bm_of(T, cfg.num_experts):
            continue
        top = min(cfg.num_experts, T * cfg.top_k)
        out.append(_cell(f"L-T{T}-A{top}", T, balanced_over(model, T, perm[:top]), "BALANCED", arms=("native", "decl")))
    return out


KINDS = {"main": main_cells, "ctl": ctl_cells, "e240": e240_cells, "decl": decl_cells}


def page_names() -> list[tuple[str, str]]:
    out = [(m, "main") for m in MODELS]
    out += [(m, "ctl") for m in CAL_MODELS + CTL_MODELS_BLIND]
    out += [("mixtral-8x7b", "e240")]
    out += [(m, "decl") for m in CAL_MODELS]
    return out


def page(model: str, kind: str) -> dict:
    d = {"schema": SCHEMA, "model": model, "page": f"{model}-{kind}", "shuffle_seed": SHUFFLE_SEED,
         "decl_copies": DECL_COPIES if kind == "decl" else None,
         "label": "SYNTHETIC (scripts/decode_synth.py); DW cells EXTRAPOLATED from rental 5's fitted shapes",
         "cells": KINDS[kind](model)}
    return d


def coverage_stats() -> dict:
    """The coverage table: summary statistics of synthetic draws, uniform and DW EXTRAPOLATED."""
    shapes = dw_shapes()
    rows = []
    for m in MODELS:
        cfg = MODEL_CONFIGS[m]
        E = cfg.num_experts
        kinds = [("uniform", "uniform", 0.0)] + ([("DW", shapes[m]["kind"], shapes[m]["param"])] if m in shapes else [])
        for lab, kind, param in kinds:
            for B in COVERAGE_BS:
                cs = _cov_draws(m, kind, param, B, range(COVERAGE_SEEDS))
                cov = np.array([sum(1 for v in c if v) / E for c in cs])
                bm = bm_of(B, E)
                rows.append({"model": m, "shape": lab, "B": B,
                             "label": "SYNTHETIC uniform" if lab == "uniform" else "EXTRAPOLATED (rental 5 DW, fitted at B >= 256)",
                             "coverage_p50": round(float(np.median(cov)), 4), "coverage_p10": round(float(np.percentile(cov, 10)), 4),
                             "coverage_p90": round(float(np.percentile(cov, 90)), 4),
                             "empty_fraction_p50": round(float(1 - np.median(cov)), 4),
                             "p_max_over_bm": round(float(np.mean([max(c) > bm for c in cs])), 4), "bm": bm})
    return {"what": "SUMMARY STATISTICS ONLY of synthetic draws at decode batch B (one token per row): coverage A/E, "
                    "the empty-expert fraction, P(max c_e > BM(B)). No count of any batch is here.",
            "primary_extract": "NOT RUN: rental 5's three routing logs are gone (scratchpad reanalysis/data empty on "
                               "2026-10-14, no copy under moe-kernels-work); the design's fallback is used and "
                               "registered (owner decision DD3)",
            "seeds": COVERAGE_SEEDS, "rows": rows}


def dump(d) -> str:
    return json.dumps(d, indent=1) + "\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def all_files() -> dict[str, str]:
    out = {f"{m}-{k}.json": dump(page(m, k)) for m, k in page_names()}
    out["coverage_stats.json"] = dump(coverage_stats())
    return out


# --------------------------------------------------------------------------
# --extract: OFF THE REPO, summary statistics of a routing log at decode B
# --------------------------------------------------------------------------

def extract(log: Path, model: str, *, batches: int = 200, seed: int = 0) -> dict:
    """Summary statistics of a decode routing log, never its counts. Input: JSONL, one routed token
    per line, {"prompt": p, "pos": t, "layer": l, "experts": [k ids]}. A decode batch of B is B
    distinct prompts at one position ("one token per row"), drawn with numpy default_rng(seed), up
    to `batches` per (B, layer). Per (B, layer): median and p10/p90 of coverage A/E, the median
    empty fraction and P(max c_e > BM(B)); then per B the median over layers and the most skewed
    layer's (least median coverage)."""
    cfg = MODEL_CONFIGS[model]
    E, k = cfg.num_experts, cfg.top_k
    by: dict = {}
    with Path(log).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ex = [int(x) for x in r["experts"]]
            if len(ex) != k or len(set(ex)) != k or not all(0 <= x < E for x in ex):
                raise ValueError(f"a routed token's experts {ex}: {k} distinct ids in 0..{E - 1}")
            by.setdefault(int(r["layer"]), {}).setdefault(int(r["pos"]), {})[int(r["prompt"])] = ex
    rng = np.random.default_rng(seed)
    out = []
    for B in COVERAGE_BS:
        bm = bm_of(B, E)
        per_layer = []
        for layer, pos in sorted(by.items()):
            covs, maxo = [], []
            cands = [(p, rows) for p, rows in pos.items() if len(rows) >= B]
            for _ in range(batches if cands else 0):
                p, rows = cands[int(rng.integers(len(cands)))]
                pick = rng.choice(sorted(rows), size=B, replace=False)
                c = np.bincount(np.array([rows[int(q)] for q in pick]).ravel(), minlength=E)
                covs.append((c > 0).sum() / E)
                maxo.append(c.max() > bm)
            if covs:
                per_layer.append({"layer": layer, "p50": float(np.median(covs)), "p10": float(np.percentile(covs, 10)),
                                  "p90": float(np.percentile(covs, 90)), "p_max_over_bm": float(np.mean(maxo)),
                                  "batches": len(covs)})
        if not per_layer:
            continue
        worst = min(per_layer, key=lambda x: x["p50"])
        out.append({"B": B, "bm": bm, "layers": len(per_layer),
                    "coverage_p50_median_layer": round(float(np.median([x["p50"] for x in per_layer])), 4),
                    "empty_fraction_median_layer": round(float(1 - np.median([x["p50"] for x in per_layer])), 4),
                    "worst_layer": worst["layer"], "worst": {k2: round(v, 4) if isinstance(v, float) else v for k2, v in worst.items()}})
    return {"what": "SUMMARY STATISTICS ONLY (coverage, empty fraction, P(max c_e > BM)) of a decode routing log; "
                    "no count of any batch", "model": model, "scheme": "decode, one token per row",
            "log_sha256": hashlib.sha256(Path(log).read_bytes()).hexdigest(), "rows": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draw", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--extract", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.extract is not None:
        if not a.model or not a.out:
            print("REFUSED: --extract needs --model and --out (a path outside the repository)")
            return 2
        if REPO in a.out.resolve().parents:
            print("REFUSED: --extract writes outside the repository (the logs state no licence)")
            return 2
        a.out.write_text(dump(extract(a.extract, a.model)))
        print(f"wrote {a.out}")
        return 0
    files = all_files()
    if a.draw:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (OUT / name).write_text(text)
            print(f"wrote {OUT / name} ({sha(text)[:12]})")
        return 0
    bad = 0
    for name, text in files.items():
        p = OUT / name
        same = p.exists() and p.read_text() == text
        print(f"{name}: {'same' if same else 'DIFFERS'}")
        bad += not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
