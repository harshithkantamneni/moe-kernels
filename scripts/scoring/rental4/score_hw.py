#!/usr/bin/env python
"""Score rental 4's hardware constants (docs/registered/2026-10-06-rental4-hw-gh200.json):
gpu-benches' constants and the torch rulers, with their consistency checks.

    python scripts/scoring/rental4/score_hw.py <repo> <tree> <out>

Reads `*-<card>-gpubench-*/constants.json` and `*-<card>-rulers-*/rulers.json`. Writes
<out>/hw.score.{json,txt}; exits 0 whatever the checks read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

PART = "hw"


def _one(tree: Path, pat: str):
    hits = sorted(Path(tree).rglob(pat))
    return json.loads(hits[0].read_text()) if len(hits) == 1 else None


def check(value, against, band) -> dict:
    if value is None:
        return {"verdict": "NOT SCORED"}
    rel = value / against - 1
    return {"value": value, "against": against, "rel": rel,
            "verdict": "CONSISTENT" if abs(rel) <= band else "DIFFERS"}


def score(repo: Path, tree: Path) -> dict:
    reg = C4.registration(repo, PART)
    ch = reg["checks"]
    res = {"registration": C4.NAMES[PART], "checks": {}, "printed": {}}
    gb = _one(tree, f"*-{C4.CARD}-gpubench-*/constants.json")
    ru = _one(tree, f"*-{C4.CARD}-rulers-*/rulers.json")
    lat = ((gb or {}).get("parsed") or {}).get("gpu-latency") or {}
    stream = ((gb or {}).get("parsed") or {}).get("gpu-stream") or {}
    l2c = ((gb or {}).get("parsed") or {}).get("gpu-l2-cache") or {}
    res["checks"]["far_l2"] = check(lat.get("far_l2_ns"), ch["far_l2"]["against_ns"], ch["far_l2"]["band"])
    res["checks"]["dram"] = check(lat.get("dram_ns"), ch["dram"]["against_ns"], ch["dram"]["band"])
    res["checks"]["triad"] = check((stream.get("peak_gbps") or {}).get("triad"), ch["triad"]["against_gbps"],
                                   ch["triad"]["band"])
    l2 = ((ru or {}).get("device") or {}).get("l2_cache_size")
    res["checks"]["l2_cache_size"] = check(l2, ch["l2_cache_size"]["against_bytes"], ch["l2_cache_size"]["band"])
    res["printed"]["cycles_at_lock"] = (gb or {}).get("cycles_at_lock")
    res["printed"]["near_l2_ns"] = lat.get("near_l2_ns")
    res["printed"]["l2_half_way_over_L"] = l2c.get("half_way_over_l2")
    res["printed"]["triad_knee_occupancy"] = stream.get("triad_knee_occupancy")
    rates = (ru or {}).get("rates") or {}
    eta = {k: (3598.0 / v["gbps"] if v and v.get("gbps") else None) for k, v in rates.items() if k != "matmul"}
    if (stream.get("peak_gbps") or {}).get("triad"):
        eta["gpubench_triad"] = 3598.0 / stream["peak_gbps"]["triad"]
    res["printed"]["eta_mix"] = eta
    mm = rates.get("matmul")
    if mm and mm.get("tflops"):
        res["printed"]["tau_mma_flop_per_clk_per_sm"] = mm["tflops"] * 1e12 / (C4.SMS * C4.LOCK_MHZ * 1e6)
    res["verdict"] = ("NOT SCORED: neither unit wrote its file" if gb is None and ru is None else
                      "; ".join(f"{k} {v['verdict']}" for k, v in res["checks"].items()))
    return res


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 4 HARDWARE CONSTANTS, {res['registration']}"]
    for k, v in res["checks"].items():
        out.append(f"  {k}: {v['verdict']}" + (f" ({v['value']:.4g} against {v['against']:.4g}, {100 * v['rel']:+.1f}%)"
                                                 if "value" in v else ""))
    for k, v in res["printed"].items():
        out.append(f"  {k}: {json.dumps(v, default=str)}")
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C4.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
