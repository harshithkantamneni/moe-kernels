#!/usr/bin/env python
"""D5-4: the call-kernel form search on DS's pages only. 16 candidate forms (7 for silu_and_mul, 9 for moe_sum; 15 distinct, as [1, T k I] spans the same columns as [1, B]
and fits identically), each
scored by its in-sample rms and its leave-one-model-out rms over DS's six models (69 NATIVE rows per class); the form
with the lowest leave-one-model-out rms per class is v5d's. A forking path on DS's pages, disclosed.
python call_forms.py [--write | --check]"""
import math
import sys

import d5b
import numpy as np

from moe.spec import MODEL_CONFIGS as MC


def feats(cfg, T, cls):
    H, inter, k = cfg.hidden_size, cfg.intermediate_size, cfg.top_k
    numel = T * k
    if cls == "silu_and_mul":
        return {"1": 1.0, "I": float(inter), "B": numel * 3 * inter * 2 / 1e6, "T k": float(numel), "T k I": numel * inter / 1e6}
    return {"1": 1.0, "k H": float(k * H), "B": (numel * H + T * H) * 2 / 1e6, "k": float(k), "H": float(H),
            "T H": T * H / 1e6}


FORMS = {"silu_and_mul": [("1",), ("1", "B"), ("1", "I"), ("1", "I", "B"), ("1", "T k I"), ("1", "T k"), ("1", "I", "T k")],
         "moe_sum": [("1",), ("1", "B"), ("1", "k H"), ("1", "k H", "B"), ("1", "k"), ("1", "k", "B"), ("1", "H", "k"),
                     ("1", "k", "T H"), ("1", "k H", "T H")]}

if __name__ == "__main__":
    rows = d5b.D5.call_rows(d5b.RD_)
    L = []
    for cls, forms in FORMS.items():
        rs = [r for r in rows if r["class"] == cls]
        models = sorted({r["model"] for r in rs})
        y = np.array([r["us"] for r in rs])
        L.append(f"{cls}: {len(rs)} rows, {len(models)} models")
        best = None
        for f in forms:
            X = np.array([[feats(MC[r["model"]], r["T"], cls)[q] for q in f] for r in rs])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            ins = math.sqrt(np.mean((X @ beta - y) ** 2))
            err = []
            for m in models:
                tr = np.array([r["model"] != m for r in rs])
                b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                err += list(X[~tr] @ b - y[~tr])
            lomo = math.sqrt(np.mean(np.square(err)))
            if best is None or lomo < best[0]:
                best = (lomo, f)
            L.append(f"  [{', '.join(f)}]: in-sample rms {ins:.3f} us, leave-one-model-out rms {lomo:.3f} us")
        L.append(f"  chosen: [{', '.join(best[1])}] ({best[0]:.3f} us)")
    dup = []
    for cls, forms in FORMS.items():
        rs = [r for r in rows if r["class"] == cls]
        cols = {f: np.array([[feats(MC[r["model"]], r["T"], cls)[q] for q in f] for r in rs]) for f in forms}
        for i, f in enumerate(forms):
            for g in forms[:i]:
                X, Y = cols[f], cols[g]
                r = np.linalg.matrix_rank
                if X.shape == Y.shape and r(X) == r(Y) == r(np.hstack([X, Y])):
                    dup.append(f"[{', '.join(f)}] spans the same columns as [{', '.join(g)}] ({cls})")
    n = sum(len(v) for v in FORMS.values())
    L.append(f"forms compared: {n} candidates, {n - len(dup)} distinct" + "".join(f"; {d}" for d in dup))
    sys.exit(d5b.emit(L, "call_forms", sys.argv[1:]))
