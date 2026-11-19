"""The job for tests/test_timing_model_v8.py's EXTENSION and SPEED tests, in its own interpreter: v8 on
ERNIE-4.5-21B-A3B calls (no measurement anywhere), `python tests/v8_extension_job.py` prints one JSON document
(last line).

  sweep   T = 1 .. 600 every T, then every 16th T to 16384, plus 4096, 8192, 8255, 8256, 12288, 16384, each on vLLM
          0.27.1's default tile for (T, E 64) at its default GROUP_SIZE_M, counts and token-major ids from fixed-seed
          random routing (each token k 6 distinct experts, uniform over E: uniform-ish counts). Every call either
          prices or is refused with a reason; any exception ends the job (exit 1). Per call: tile, G, n_eff, ms, why,
          wall time
  jumps   per tile (and G), the relative step |ms(T_b) / ms(T_a) - 1| between adjacent sampled T, largest first
          (REPORTED, never a pass condition)
  order   about 30 prefill calls on both prefill tiles: each expert's tokens ascending (the walk's assumption) against
          shuffled within the expert (order_seed 1): the relative change (REPORTED)
  speed   per tile, the median and p90 wall time of one call (the first call, which loads round 5's chain, left out)
  skew    (review 1, MAJOR 11) two non-uniform routings on the same ladder: ZIPF (expert popularity 1 / rank^1.2,
          k distinct experts per token drawn by weight without replacement) at every 64th T and the tile edges, and
          PADDED (CUDA-graph padding: the T_pad - T padded rows all route to the same k experts, real rows uniform) at
          every capture size with T = the size's predecessor + 1. Every call prices or is refused; every refusal of
          every routing is classed by v8.refusal_class (DECLARED_REFUSALS), counted per class
The prefill calls name the constants v8 requires (no defaults): the v7 prefill blind's finalised par, A2's 64x128 par and
round 4's SEEN knee (tests/v8_invariant_job.py SAG128, SAG64, KNEE).
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import r3_timing_model_v8 as V8  # noqa: E402

SEED = 20261119
SPECIAL = (4096, 8192, 8255, 8256, 12288, 16384)
CONST = {"sag128": (64.325, 612.5), "sag64": (30.365, 300.0), "knee": (6.0, 2.75)}
ZIPF_S = 1.2
#: vLLM 0.27.1's capture sizes at max_num_seqs 1024 (scripts/prod/prodlib.capture_sizes)
CAPTURE = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))


def zipf_routing(T: int, seed: int = SEED):
    E, k = V8.ERNIE.num_experts, V8.ERNIE.top_k
    rng = np.random.default_rng(seed + 31 * T)
    w = 1.0 / np.arange(1, E + 1) ** ZIPF_S
    w = w[rng.permutation(E)]
    # k distinct experts per token, by weight without replacement (Gumbel top-k)
    g = np.log(w)[None, :] - np.log(-np.log(rng.random((T, E))))
    ids = np.argsort(-g, axis=1)[:, :k]
    return ids, np.bincount(ids.ravel(), minlength=E).tolist()


def padded_routing(T_pad: int, T: int, seed: int = SEED):
    """Real rows uniform; the T_pad - T padded rows (zero hidden states) all route to the same k experts."""
    E, k = V8.ERNIE.num_experts, V8.ERNIE.top_k
    rng = np.random.default_rng(seed + 7 * T_pad)
    real = np.argsort(rng.random((T, E)), axis=1)[:, :k]
    pad = np.tile(np.arange(k), (T_pad - T, 1))
    ids = np.concatenate([real, pad]) if T_pad > T else real
    return ids, np.bincount(ids.ravel(), minlength=E).tolist()


def sample_T() -> list[int]:
    return sorted(set(range(1, 601)) | set(range(608, 16385, 16)) | set(SPECIAL))


def routing(T: int, seed: int = SEED):
    rng = np.random.default_rng(seed + T)
    ids = np.argsort(rng.random((T, V8.ERNIE.num_experts)), axis=1)[:, :V8.ERNIE.top_k]
    return ids, np.bincount(ids.ravel(), minlength=V8.ERNIE.num_experts).tolist()


def pct(p, xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> dict:
    V8.register_model(V8.ERNIE)
    rows = []
    for i, T in enumerate(sample_T()):
        ids, counts = routing(T)
        t0 = time.perf_counter()
        p = V8.price({"model": V8.ERNIE.name, "T": T, "counts": counts, "ids": ids, **CONST})
        dt = time.perf_counter() - t0
        if p["ms"] is None and not p.get("why"):
            raise AssertionError(f"T {T}: no price and no reason")
        rows.append({"T": T, "tile": p["tile"], "G": p["G"], "n_eff": p.get("n_eff"), "ms": p["ms"],
                     "why": p.get("why"), "wall_s": dt, "first": i == 0})
    refused = [{"T": r["T"], "tile": r["tile"], "G": r["G"], "why": r["why"]} for r in rows if r["ms"] is None]
    seg: dict = {}
    for r in rows:
        if r["ms"] is not None:
            seg.setdefault(f"{r['tile']} G{r['G']}", []).append(r)
    jumps = {}
    for k, rs in seg.items():
        steps = [{"from_T": a["T"], "to_T": b["T"], "rel": b["ms"] / a["ms"] - 1}
                 for a, b in zip(rs, rs[1:], strict=False)]
        steps.sort(key=lambda s: -abs(s["rel"]))
        jumps[k] = {"calls": len(rs), "T_range": [rs[0]["T"], rs[-1]["T"]], "largest": steps[:5],
                    "steps_over_3pct": sum(1 for s in steps if abs(s["rel"]) > 0.03),
                    "decreasing_steps": sum(1 for s in steps if s["rel"] < 0)}
    order = []
    for T in [130, 171, 200, 256, 300, 342, 400, 450, 500, 512, 513, 600, 700, 1024, 1500, 2048, 3000, 4096, 5052, 6000,
              7000, 8191, 8256, 9000, 10000, 11000, 12288, 14000, 15000, 16384]:
        ids, counts = routing(T, SEED + 7)
        call = {"model": V8.ERNIE.name, "T": T, "counts": counts, "ids": ids, **CONST}
        a = V8.price(call)
        b = V8.price(dict(call, order_seed=1))
        order.append({"T": T, "tile": a["tile"], "G": a["G"], "ascending": a["ms"], "shuffled": b["ms"],
                      "rel": None if a["ms"] is None else b["ms"] / a["ms"] - 1, "why": a.get("why")})
    speed = {}
    for k in sorted({r["tile"] for r in rows}):
        w = [r["wall_s"] for r in rows if r["tile"] == k and not r["first"]]
        speed[k] = {"calls": len(w), "median_ms": 1e3 * st.median(w), "p90_ms": 1e3 * pct(0.9, w)}
    skew = []
    zT = sorted(set(range(64, 16385, 64)) | {1, 2, 8, 16, 32, 33, 64, 65, 96, 97, 128, 129, 512, 513, 8191, 8192, 8255,
                                           8256})
    for T in zT:
        ids, counts = zipf_routing(T)
        p = V8.price({"model": V8.ERNIE.name, "T": T, "counts": counts, "ids": ids, **CONST})
        if p["ms"] is None and not p.get("why"):
            raise AssertionError(f"zipf T {T}: no price and no reason")
        skew.append({"routing": "zipf", "T": T, "tile": p["tile"], "G": p["G"], "ms": p["ms"], "why": p.get("why"),
                     "active": sum(1 for c in counts if c), "max_count": max(counts)})
    prev = 0
    for Tp in CAPTURE:
        T = prev + 1
        prev = Tp
        ids, counts = padded_routing(Tp, T)
        p = V8.price({"model": V8.ERNIE.name, "T": Tp, "counts": counts, "ids": ids, **CONST})
        if p["ms"] is None and not p.get("why"):
            raise AssertionError(f"padded T_pad {Tp}: no price and no reason")
        skew.append({"routing": "padded", "T": Tp, "T_real": T, "tile": p["tile"], "G": p["G"], "ms": p["ms"],
                     "why": p.get("why"), "active": sum(1 for c in counts if c), "max_count": max(counts)})
    every = [dict(r, routing="uniform") for r in rows] + skew
    classes: dict = {}
    undeclared = []
    for r in every:
        if r["ms"] is None:
            c = V8.refusal_class(r["why"])
            if c is None:
                undeclared.append({k: r[k] for k in ("routing", "T", "tile", "G", "why")})
            classes.setdefault(r["routing"], {}).setdefault(str(c), 0)
            classes[r["routing"]][str(c)] += 1
    return {"calls": len(rows), "priced": sum(1 for r in rows if r["ms"] is not None), "refused": refused,
            "skew": skew, "refusal_classes": classes, "undeclared_refusals": undeclared,
            "jumps": jumps, "order": order,
            "order_max_abs_rel": max(abs(o["rel"]) for o in order if o["rel"] is not None),
            "speed": speed, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(main(), default=str))
