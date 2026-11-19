"""The job for tests/test_timing_model_v8.py's POISON and IDS tests, in its own interpreter (round 5's chain loads
generic module names): `python tests/v8_checks_job.py` prints one JSON document (last line).

  sentinel   a Tread raises PoisonedTread on every numeric use (int, index, float, bool, arithmetic, ordering, == with a
             number) and hashes, prints and compares with another Tread
  missed     a frozen site v8 did not replace fails loudly: the frozen originals (wave_split_bytes.walk,
             Geometry.num_pid_m, v7's own price_prefill) handed a Tread raise PoisonedTread
  pages      whole-tread page cells priced by v8 with the sentinel, on both prefill tiles and G 1 and 16 (B2's 64g1,
             128g1, 128g16 cells and a 64x128 G 16 cell made from a 64g1 cell): every one priced, through the Tread
             path,
             and equal to v7 at the integer tread
  frac       ERNIE at n_eff 3.7 (T 5052) prices differently from n_eff 3.0 (T 4096): no silent truncation
  refusals   128x128 G 1 at n_eff 6 (ERNIE T 8192) refused by HOPPER_G1_N; 64x128 G 1 at n_eff 6 (ERNIE T 4096 on the
             64x128 tile): v7 raises KeyError('reads'), v8 refuses with a reason
  ids        page cells with their own generated ids passed explicitly == the same cells with ids None, bit for bit:
             prefill (B2 histogram and balanced cells, every tile and G) and decode (the v7 decode blind's 143 cells,
             V4DP.ids_of passed); ERNIE's real ids against the generated ids on the same counts (reported)
  bindings   every binding of every swapped function (a scan of every loaded module) equals SWAP_BINDINGS, and every
             one holds its frozen original after the pricing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import r3_timing_model_v7 as V7  # noqa: E402
import r3_timing_model_v8 as V8  # noqa: E402
import round5_chain as R5  # noqa: E402
import v8_invariant_job as J  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402


def raises_poison(f) -> bool:
    try:
        f()
    except V8.PoisonedTread:
        return True
    except BaseException as ex:  # noqa: BLE001
        return f"{type(ex).__name__}: {ex}"
    return False


def sentinel() -> dict:
    t = V8.Tread([128] * 8, 8, 128, label=1)
    u = V8.Tread([128] * 8, 8, 128, label=1)
    ops = {"int": lambda: int(t), "index": lambda: [0, 1][t], "float": lambda: float(t), "bool": lambda: bool(t),
           "add": lambda: t + 1, "radd": lambda: 1 + t, "mul": lambda: t * 128, "rmul": lambda: 128 * t,
           "floordiv": lambda: 7 // t, "sub": lambda: t - 1, "lt": lambda: t < 6, "ge": lambda: t >= 6,
           "gt": lambda: 1 < t, "eq_number": lambda: t == 1, "range": lambda: range(1, t), "min": lambda: min(1, t),
           "numpy": lambda: np.asarray([t], dtype=float), "round": lambda: round(t), "neg": lambda: -t}
    got = {k: raises_poison(f) for k, f in ops.items()}
    return {"raises": got, "hash_equal": hash(t) == hash(u), "eq_tread": t == u, "str": str(t), "fmt": f"n{t}",
            "frac_str": str(V8.Tread([100] * 8, 8, 128))}


def _geometry(model):
    """The CAL engine's geometry of the model (wave_split_bytes_v4.base_geometry), a real W.Geometry."""
    import wave_split_bytes_v4 as WV4
    RG = V8._ORIG["price_parts"].__globals__["RG"]
    return WV4.base_geometry(RG.pricer().eng, model, 9)


def missed() -> dict:
    """A frozen site handed a Tread (outside installed(): the originals) raises."""
    V8._originals()
    r = R5.b2_rows()[6]                       # a 128 G 1 histogram cell
    call = J.prefill_call(r, (59.815, 612.5), (30.365, 300.0))
    row = V8.prefill_row(r["model"], call["counts"], 128, 1, page=call["page"])
    geom = V8._ORIG["Geometry.num_pid_m"]
    return {
        "frozen walk": raises_poison(lambda: V8._ORIG["W.walk"](
            "native", 1, row["n"], 4, 264, 10, MODEL_CONFIGS[r["model"]].num_experts, counts=row["c"]["counts"],
            block_m=128)),
        "frozen num_pid_m": raises_poison(lambda: geom(_geometry(r["model"]), "native", row["n"])),
        "v7 price_prefill": raises_poison(lambda: V7.price_prefill(row, sag128=(59.815, 612.5), sag64=(30.365, 300.0))),
    }


def pages() -> dict:
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    par, par64 = tuple(fin["par"]), tuple(fin["par64"])
    rows = R5.b2_rows()
    pick = {}
    for r in rows:
        k = (r["cls"], r["c"]["counts"] is None)
        pick.setdefault(k, r)
    out = []
    cases = [(k, r) for k, r in sorted(pick.items())]
    g16 = next(r for (cls, bal), r in cases if cls == "64g1" and not bal)
    r64g16 = dict(g16, doc=dict(g16["doc"], group_m=16), cls="64g16")
    cases.append((("64g16", False), r64g16))
    for (cls, bal), r in cases:
        before = sum(V8.STATS.values())
        p = V8.price(J.prefill_call(r, par, par64))
        used = sum(V8.STATS.values()) - before
        q = V7.price_prefill(r, sag128=par, sag64=par64)
        out.append({"cls": cls, "balanced": bal, "model": r["model"], "n": r["n"], "label": r["label"],
                    "tile": p["tile"], "G": p["G"], "ms": p["ms"], "v7_ms": q["ms"], "equal_v7": p["ms"] == q["ms"]
                    and p["reads"] == q["reads"], "tread_calls": used})
    return {"cells": out}


def ernie_call(T, *, tile=None, G=None, ids=True, seed=0):
    rng = np.random.default_rng(seed + T)
    ids_ = np.argsort(rng.random((T, 64)), axis=1)[:, :6]
    c = {"model": V8.ERNIE, "T": T, "counts": np.bincount(ids_.ravel(), minlength=64).tolist(),
         "ids": ids_ if ids else None}
    if tile:
        c["tile"] = tile
    if G:
        c["G"] = G
    return c


def frac_and_refusals() -> dict:
    V8.register_model(V8.ERNIE)
    a = V8.price(ernie_call(4096))
    b = V8.price(ernie_call(5052))
    h1 = V8.price(ernie_call(8192))
    h64 = V8.price(ernie_call(4096, tile="64x128x64w8s3", G=1, ids=False))
    # v7 at the same 64x128 G 1 cell at its integer tread 6
    c = ernie_call(4096, ids=False)
    row = {"model": V8.ERNIE.name, "page": "x", "n": 6, "label": "x",
           "c": {"n": 6, "counts": c["counts"], "label": "x"},
           "doc": {"block_m": 64, "group_m": 1, "shuffle_seed": None}}
    try:
        V7.price_prefill(row, sag128=V8.DEFAULT_SAG128, sag64=V8.DEFAULT_SAG64)
        v7 = "priced"
    except KeyError as ex:
        v7 = f"KeyError({ex})"
    return {"n_eff": [a["n_eff"], b["n_eff"]], "ms": [a["ms"], b["ms"]], "differ": a["ms"] != b["ms"],
            "hopper_128": {"n_eff": h1["n_eff"], "ms": h1["ms"], "why": h1.get("why")},
            "defect_64": {"n_eff": h64["n_eff"], "v8_ms": h64["ms"], "v8_why": h64.get("why"), "v7": v7}}


def gen_ids(model, counts, seed):
    """wave_split_bytes_v3.tile_content_ids_counts' ids: realize_counts, then the page's token-row shuffle."""
    import torch

    from moe.routing.distributions import realize_counts
    cfg = MODEL_CONFIGS[model]
    T = sum(counts) // cfg.top_k
    ids = realize_counts(list(counts), T, cfg.top_k, device="cpu")
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        ids = ids[torch.randperm(T, generator=g)]
    return ids.numpy()


def ids_check() -> dict:
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    par, par64 = tuple(fin["par"]), tuple(fin["par64"])
    pre = []
    for r in R5.b2_rows():
        call = J.prefill_call(r, par, par64)
        a = V8.price(call)
        b = V8.price(dict(call, ids=gen_ids(r["model"], call["counts"], call["page"]["shuffle_seed"])))
        pre.append({"cls": r["cls"], "balanced": r["c"]["counts"] is None,
                    "equal": a["ms"] == b["ms"] and a["reads"] == b["reads"]})
    V7.decode_modules()
    import d5common as D5
    import dcommon as DC
    import v7bcommon as VB
    X = V7.decode_modules()
    RD, V4DP = X["RD"], X["V4DP"]
    dec = []
    with D5.blind5():
        for m in VB.MODELS:
            for c in DC.page_doc(m, "main")["cells"]:
                call = {"model": m, "T": int(c["T"]), "counts": c["counts"],
                        "page": {"label": c["label"], "shuffle": bool(c.get("shuffle", True))}}
                a = V8.price(call)
                cell = {"T": int(c["T"]), "counts": list(c["counts"]), "shuffle": bool(c.get("shuffle", True))}
                b = V8.price(dict(call, ids=V4DP.ids_of(m, cell)))
                naive = RD.geometry(MODEL_CONFIGS[m], int(c["T"]), c["counts"], None).path == RD.NAIVE
                dec.append({"equal": a["ms"] == b["ms"], "naive": naive})
    real = V8.price(ernie_call(12288))
    gen = V8.price(ernie_call(12288, ids=False))
    return {"prefill": pre, "decode": dec,
            "ERNIE T 12288 G 16, real ids against the generated ids (same counts)": {
                "real": real["ms"], "generated": gen["ms"], "rel": real["ms"] / gen["ms"] - 1}}


def binding_check() -> dict:
    """bindings() over every loaded module equals SWAP_BINDINGS, and after all the pricing above every binding holds
    its frozen original again."""
    got = V8.bindings()
    plan = V8._plan()
    restored = all((h.get(a) if isinstance(h, dict) else vars(h).get(a)) is f for h, a, f, _c in plan)
    return {"bindings": got, "declared": {k: sorted(v) for k, v in V8.SWAP_BINDINGS.items()},
            "equal": got == {k: sorted(v) for k, v in V8.SWAP_BINDINGS.items()}, "plan_entries": len(plan),
            "restored": restored}


def main() -> dict:
    return {"sentinel": sentinel(), "missed": missed(), "pages": pages(), "frac": frac_and_refusals(),
            "ids": ids_check(), "bindings": binding_check(), "stats": dict(V8.STATS)}


if __name__ == "__main__":
    print(json.dumps(main(), default=str))
