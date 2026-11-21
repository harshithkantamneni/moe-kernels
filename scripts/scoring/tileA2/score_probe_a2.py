#!/usr/bin/env python
"""Read tile rental A2's lock-hold probe pages (docs/registered/2026-10-20-tileA2-gates-gh200 G-PROBE): each rung that
ran (1710, then 1410, then 1305), by probe_a2's rule. DESCRIPTIVE: the rental's lock was chosen on the VM by the
same rule (gh200-driver/gate-probe-<L>.json); this re-reads it, with the hw_rulers bandwidths at the selected
rung against rung 1710's (DRAM at the lower L2 clock). Clocks, states and shares only: no time.

    python scripts/scoring/tileA2/score_probe_a2.py <repo> <tree> <out>

<tree> is the run's results directory; the session (locked-r3/*/summary.json) is read beside it
(<tree>/../session). Writes <out>/probe.score.{json,txt}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
import probe_a2 as PR  # noqa: E402

TC = A2.TC
PART = "probe"


def score(repo: Path, tree: Path) -> dict:
    session = Path(tree).parent / "session"
    out = {"registration": A2.NAMES_A2["gates"], "part": PART, "label": "DESCRIPTIVE (the probe's times are never read)",
           "rungs": {}}
    for lock in A2.LOCK_LADDER:
        pages = {}
        for model, _name, _G, short in PR.PROBES:
            lab = PR.label_of(lock, short)
            if TC.C3.find_gaps(tree, model, lab) is None:
                continue
            pages[f"{model}-{lab}"] = PR.page_rule(tree, session, model, lab, lock)
        if pages:
            out["rungs"][f"{lock:g}"] = {"pages": pages, "every page PASSES": all(v["verdict"] == "PASS" for v in pages.values())}
    out["a2_lock_selected"] = A2.lock_of_session(tree)
    out["rulers (DESIGN-A2 6 risk 7)"] = rulers(tree, out["a2_lock_selected"])
    return out


def rulers(tree: Path, lock: float | None) -> dict:
    """hw_rulers at the selected rung and at 1710: each timing's ms and clock, and ms(L) / ms(1710)."""
    def read(label):
        hits = sorted(Path(tree).rglob(f"*-{A2.CARD}-rulers-{label}/rulers.json"))
        if len(hits) != 1:
            return None
        return {k: {"ms_p50": v.get("ms_p50"), "sm_clock_load_mhz": v.get("sm_clock_load_mhz")}
                for k, v in (json.loads(hits[0].read_text()).get("timings") or {}).items()}
    at_l = read(f"lp{int(round(lock))}") if lock else None
    at_t = read("lp1710")
    out = {"at L": at_l, "at 1710": at_t}
    if at_l and at_t:
        out["ms(L) / ms(1710)"] = {k: A2.r6(at_l[k]["ms_p50"] / at_t[k]["ms_p50"]) for k in at_l
                                   if k in at_t and at_l[k]["ms_p50"] and at_t[k]["ms_p50"]}
    return out


def lines(res: dict) -> list[str]:
    out = [f"READ against {res['registration']} ({res['label']}); A2's lock {res['a2_lock_selected']}"]
    for lk, r in res["rungs"].items():
        out.append(f"  {lk} MHz: {'every page PASSES' if r['every page PASSES'] else 'a page FAILS'}")
        for k, v in r["pages"].items():
            out.append(f"    {k}: {v['verdict']} min cell {v.get('min_cell_mhz')} non-clean {(v.get('non_clean') or {}).get('share')}")
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
