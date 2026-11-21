#!/usr/bin/env python
"""The naive-path forms against PC2-R4 (SEEN measurement pages; computed before the freeze): per READ naive cell GEMM of
a seen model, v4d''s naive GEMM price (fix W's reads) and the law's forms (twin: the law on every live CTA; all@fetch:
every CTA at the fetchers' load, PC2-R4's 'same per-CTA rate' read literally), against PC2's t; the follower classes
split by whether the GEMM has a follower (PC2-R4's leaders and followers). Geometry prices without the CAL constants.

    python scripts/scoring/decode/d6_build/naive_check.py [--write | --check]"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import d6v as L2  # noqa: E402
import d4common as D4  # noqa: E402
import d6common as D6  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402


def lines() -> list[str]:
    prm = D4.ds_prm()
    terms = D6.registered_terms()
    page = json.loads((L2.REPO / "docs/registered/2026-11-10-pc2-pages/pc2-n.json").read_text())
    R4 = L2.PC2["PC2-R4"]["cells"]
    forms = {"v4d'": L2.variant(naive="v4dp"), "twin": L2.variant(naive="twin"), "all@fetch": L2.variant(naive="all@fetch")}
    E = {f: {"no follower": [], "follower": []} for f in forms}
    L = ["PC2-R4 READ naive GEMMs of seen models: price (us) and price / t - 1"]
    for c in page["cells"]:
        if c["shape"] not in MC or c["path"] != "naive":
            continue
        ids = D6.V4DP.ids_of(c["shape"], c)
        got = {f: L2.price(lw, c["shape"], int(c["T"]), c["counts"], prm, ids=ids, terms=terms) for f, lw in forms.items()}
        for g in ("w1", "w2"):
            x = R4[c["label"]][g]
            if not str(x["status"]).startswith("READ") or x.get("t_us") is None:
                continue
            k = "follower" if x.get("followers") else "no follower"
            s = f"  {c['label']:13s} {g} {k:11s} t {x['t_us']:7.2f} |"
            for f in forms:
                v = got[f]["gemm"][g]["gemm_ms"] * 1e3
                E[f][k].append(v / x["t_us"] - 1)
                s += f" {f} {v:7.2f} ({v / x['t_us'] - 1:+.1%})"
            L.append(s)
    for f in forms:
        for k, v in E[f].items():
            v = np.array(v)
            L.append(f"{f:9s} {k:11s} n {len(v):2d} rms {np.sqrt(np.mean(v ** 2)):.1%} mean {v.mean():+.1%}")
    return L


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "naive_check", sys.argv[1:]))
