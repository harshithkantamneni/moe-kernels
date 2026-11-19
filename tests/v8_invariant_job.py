"""The job for tests/test_timing_model_v8.py's INVARIANT test: one side (v7 or v8) of every registered prediction set,
in its OWN interpreter (critique M3: the pricers' caches are module state, so v7 and v8 never share a process).

    python tests/v8_invariant_job.py v7     # v7's own code; r3_timing_model_v8 is never imported
    python tests/v8_invariant_job.py v8     # v8.price on the same cells

prints one JSON document (last line): every number as float.hex, so the test compares bit for bit.

  r3       the 300 registered R3 predictions (r3_timing_model_v4.CHECK_FILES, v7 --check-r3's cells); the v7 side also
           holds each against its registered number (6 decimals)
  b3       B2's 84 NATIVE cells at B3's board pars (b3.predictions.json par, par64), every switch off (round 5, B3's
           frozen predictions: the v7 side holds them to 6 decimals and the reads to the byte) and on (v7)
  pblind   the v7 prefill blind's 54 NATIVE cells at its four reference pars and its finalised par, v7 (every switch
           on) and its round-5 DESCRIPTIVE column (every switch off), v7pcommon.price's call; the v7 side holds the four
           reference pars to the registration's fingerprint and the finalised par to the score's finalised cells
  decode   the v7 decode blind's 143 cells at its frozen constants (finalise_decode.params_of on the frozen block,
           terms, law, v7bcommon.prm_fix_of); the v7 side holds them to the registered ms['v4d']
  ernie    ERNIE-4.5-21B-A3B whole-tread anchors T 4096 (G 1), 12288 and 16384 (G 16) at 128x128, counts from fixed-
           seed random topk ids, no ids passed (the generated ids): v7's price_prefill at the integer tread and v8's
           price; the v8 side also prices them with the real ids (reported, not compared)
  canary   pid, which model modules are loaded, whether the swaps are in place, v8's Tread-path call counts
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "v7pblind", REPO / "scripts" / "scoring" / "v7blind"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

ERNIE_FIELDS = dict(name="ernie-4.5-21b-a3b", hidden_size=2560, intermediate_size=1536, num_experts=64, top_k=6,
                    gate_fn="softmax", norm_topk_prob=True, shared_experts=2, num_layers=28, first_moe_layer=1,
                    hf_repo="baidu/ERNIE-4.5-21B-A3B-PT", verified=True)
ERNIE_T = (4096, 12288, 16384)
ERNIE_SEED = 20261119
SAG128, SAG64 = (64.325, 612.5), (30.365, 300.0)


def h(x):
    return None if x is None else float(x).hex()


def hreads(rd):
    return None if rd is None else {g: h(v) for g, v in sorted(rd.items())}


def ernie_ids(T: int) -> np.ndarray:
    """Fixed-seed topk ids: per token k distinct experts, uniform over E."""
    rng = np.random.default_rng(ERNIE_SEED + T)
    return np.argsort(rng.random((T, ERNIE_FIELDS["num_experts"])), axis=1)[:, :ERNIE_FIELDS["top_k"]]


def decode_constants_v7():
    """The v7 decode blind scorer's constants (the recipe v8.decode_constants follows), built without v8."""
    import r3_timing_model_v7 as V7
    V7.decode_modules()
    import d5common as D5
    import d6common as D6
    import finalise_decode as FN
    import gate_d6 as G
    import v7bcommon as VB
    reg = json.loads((REPO / "docs/registered/2026-11-17-v7-decode-blind-gh200.json").read_text())
    fz = reg["predictions"]["frozen"]
    m = fz["model"]
    return dict(prm=FN.params_of(fz), terms=D5.V5D.Terms.from_dict(m["terms"]), law=D6.V6D.Law(G._tuples(m["law"])),
                prm_fix=VB.prm_fix_of(m["D6FIX"], fz.get("tile_params")))


def prefill_call(r, par, par64):
    """A v8 call for one round-5 page row: its counts (the uniform histogram on a balanced cell), its tile and G, its
    page's n, label and shuffle seed (None on a balanced cell, as regtile.seed_of)."""
    import r3_timing_model_v7 as V7

    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[r["model"]]
    bm, n = int(r["doc"]["block_m"]), int(r["n"])
    counts = r["c"]["counts"] if r["c"]["counts"] is not None else [n * bm] * cfg.num_experts
    return {"model": r["model"], "T": sum(int(v) for v in counts) // cfg.top_k, "counts": counts,
            "tile": V7.PREFILL_TILES[0] if bm == 128 else V7.PREFILL_TILES[1], "G": int(r["doc"]["group_m"]),
            "sag128": tuple(par), "sag64": tuple(par64),
            "page": {"n": n, "label": r["label"], "page": r.get("page"),
                     "shuffle_seed": None if r["c"]["counts"] is None else r["doc"]["shuffle_seed"]}}


def main(side: str) -> dict:
    assert side in ("v7", "v8"), side
    import r3_timing_model_v7 as V7
    import round5_chain as R5

    from moe.spec import MODEL_CONFIGS, MoEConfig
    if side == "v8":
        import r3_timing_model_v8 as V8
        pre = lambda r, par, par64, sw: V8.price(prefill_call(r, par, par64), sw)  # noqa: E731
        SW_ON, SW_OFF = V8.V8, V8.Switches(RR=False, D6FIX=False)
    else:
        def pre(r, par, par64, sw):
            return V7.price_prefill(r, sag128=tuple(par), sag64=tuple(par64), switches=sw)
        SW_ON, SW_OFF = V7.V7, V7.OFF
    out: dict = {"side": side}

    # ---- r3: the 300 registered v3 predictions (V7.check_r3's cells)
    X = V7.r3_pricer()
    rows = []
    reg_bad = 0
    for f in X["V4"].CHECK_FILES:
        reg = json.loads((REPO / f).read_text())
        hist = {k: json.loads((REPO / v["path"]).read_text()) for k, v in reg["histograms"]["files"].items()}
        for r in reg["predictions"]:
            page = hist[f"{r['model']}-{r['page']}.json"]
            c = next(c for c in page["cells"] if c["label"] == r["label"] and int(c["n"]) == int(r["n"]))
            seed = None if c["counts"] is None else page["shuffle_seed"]
            cell = {"tile": V7.R3_TILE, "model": r["model"], "arm": r["arm"], "n": int(r["n"]), "counts": c["counts"],
                    "copies": r.get("copies", 9), "seed": seed, "G": 8, "A_us": r["A_proxy_us"]}
            got = V8.price({"tile": V7.R3_TILE, "r3": cell}) if side == "v8" else V7.price(cell)
            rows.append(h(got["ms"]))
            if side == "v7" and round(got["ms"], 6) != r["ms"]["v3"]:
                reg_bad += 1
    out["r3"] = rows
    out["r3_unequal_registered"] = reg_bad if side == "v7" else None

    # ---- b3: B2's 84 cells at B3's board pars, off (B3's frozen predictions) and on
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    par, par64 = tuple(fin["par"]), tuple(fin["par64"])
    b2 = R5.b2_rows()
    b3 = {"off": [], "on": []}
    b3_bad = 0
    for r, c in zip(b2, fin["cells"], strict=True):
        a = pre(r, par, par64, SW_OFF)
        b = pre(r, par, par64, SW_ON)
        b3["off"].append([h(a["ms"]), hreads(a.get("reads"))])
        b3["on"].append([h(b["ms"]), hreads(b.get("reads"))])
        if side == "v7" and (round(a["ms"], 6) != c["ms"]
                             or {g: int(round(v)) for g, v in a["reads"].items()} != c["reads"]):
            b3_bad += 1
    out["b3"] = b3
    out["b3_unequal_registered"] = b3_bad if side == "v7" else None

    # ---- pblind: the v7 prefill blind's 54 cells at its four reference pars and its finalised par
    import v7pcommon as VP
    reg = json.loads((REPO / VP.REG_JSON).read_text())
    score = json.loads((REPO / "scripts/scoring/v7pblind/prefill_blind.score.json").read_text())
    pars = dict(VP.REF_PARS, finalised=tuple(score["finalised"]["par"]))
    prow = VP.rows()
    pb = {}
    pb_bad = 0
    for board, pr_ in pars.items():
        on, off = [], []
        for r in prow:
            a = pre(r, pr_, VP.A2_PAR_64, SW_ON)
            b = pre(r, pr_, VP.A2_PAR_64, SW_OFF)
            on.append([h(a["ms"]), hreads(a.get("reads"))])
            off.append(h(b["ms"]))
        pb[board] = {"v7": on, "round5 (DESCRIPTIVE)": off}
        if side == "v7":
            want = (reg["predictions"]["fingerprint (the scorer recomputes it before reading anything)"][board]["ms"]
                    if board != "finalised" else [c["ms"] for c in score["finalised"]["cells"]])
            got = [None if x[0] is None else round(float.fromhex(x[0]), 6) for x in on]
            pb_bad += sum(1 for a, b in zip(got, want, strict=True) if a != b)
            if board == "A2":
                cells = reg["predictions"]["cells at A2's par"]
                pb_bad += sum(1 for c, x in zip(cells, off, strict=True)
                              if round(float.fromhex(x), 6) != c["round5_ms (DESCRIPTIVE, every switch off)"])
    out["pblind"] = pb
    out["pblind_cells"] = len(prow)
    out["pblind_unequal_registered"] = pb_bad if side == "v7" else None

    # ---- decode: the v7 decode blind's 143 cells at its frozen constants
    V7.decode_modules()
    import d5common as D5
    import dcommon as DC
    import v7bcommon as VB
    dreg = json.loads((REPO / "docs/registered/2026-11-17-v7-decode-blind-gh200.json").read_text())["predictions"]
    dc = None if side == "v8" else decode_constants_v7()
    dec, d_bad, i = [], 0, 0
    with D5.blind5():
        for m in VB.MODELS:
            for c in DC.page_doc(m, "main")["cells"]:
                if side == "v8":
                    p = V8.price({"model": m, "T": int(c["T"]), "counts": c["counts"],
                                  "page": {"label": c["label"], "shuffle": bool(c.get("shuffle", True))}})
                    ms = p["ms"]
                else:
                    p = V7.price_decode_cell(m, c, dc["prm"], terms=dc["terms"], law=dc["law"], prm_fix=dc["prm_fix"])
                    ms = None if p is None else p["ms"]
                    w = dreg["cells"][i]
                    assert (w["model"], w["label"]) == (m, c["label"]), (w["model"], w["label"], m, c["label"])
                    d_bad += ms != w["ms"]["v4d"]
                dec.append(h(ms))
                i += 1
    out["decode"] = dec
    out["decode_unequal_registered"] = d_bad if side == "v7" else None

    # ---- ernie: whole-tread anchors, the same counts, no ids
    cfg = MoEConfig(**ERNIE_FIELDS)
    if side == "v8":
        assert V8.ERNIE == cfg, (V8.ERNIE, cfg)
        V8.register_model(V8.ERNIE)
    else:
        MODEL_CONFIGS[cfg.name] = cfg
    er, er_ids = [], []
    for T in ERNIE_T:
        ids = ernie_ids(T)
        counts = np.bincount(ids.ravel(), minlength=cfg.num_experts).tolist()
        G = 16 if T // cfg.num_experts > 128 else 1
        if side == "v8":
            p = V8.price({"model": cfg.name, "T": T, "counts": counts})
            q = V8.price({"model": cfg.name, "T": T, "counts": counts, "ids": ids})
            er_ids.append(h(q["ms"]))
            assert (p["tile"], p["G"]) == ("128x128x64w8s3", G), (p["tile"], p["G"])
        else:
            n = T * cfg.top_k // (cfg.num_experts * 128)
            assert n * cfg.num_experts * 128 == T * cfg.top_k
            row = {"model": cfg.name, "page": "ernie-anchor", "n": n, "label": f"T{T}",
                   "c": {"n": n, "counts": counts, "label": f"T{T}"},
                   "doc": {"block_m": 128, "group_m": G, "shuffle_seed": None}}
            p = V7.price_prefill(row, sag128=SAG128, sag64=SAG64, switches=V7.V7)
        er.append([h(p["ms"]), hreads(p.get("reads"))])
    out["ernie"] = er
    out["ernie_with_ids"] = er_ids if side == "v8" else None

    # ---- canary
    import wave_split_bytes as W
    out["canary"] = {
        "pid": os.getpid(), "v8_loaded": "r3_timing_model_v8" in sys.modules,
        "swaps_in_place": W.Geometry.num_pid_m.__module__ != "wave_split_bytes",
        "tread_calls": dict(sys.modules["r3_timing_model_v8"].STATS) if side == "v8" else {},
        "num_pid_m_module": W.Geometry.num_pid_m.__module__,
    }
    if side == "v7":
        assert not out["canary"]["v8_loaded"], "the v7 side imported v8"
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
