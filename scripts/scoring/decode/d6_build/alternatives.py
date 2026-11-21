#!/usr/bin/env python
"""CAL2 (clause 1's pages) under the candidate and under each one-choice alternative D6-18 to D6-24 name, the CAL
constants refitted each time (score_cal_d3's rules). Computed before the freeze; the SEEN decode blind is NOT re-scored
here under any variant.

    python scripts/scoring/decode/d6_build/alternatives.py [--write | --check]"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6v as L2  # noqa: E402

ALT = {"CANDIDATE (frozen v6d)": {}, "design-D6 v6d (frozen 2026-11-10T05:17Z)": "design",
       "D6-18 c0 const 4.70 (v6d)": {"c0_form": "const"}, "D6-18 c0 const 2.40": {"c0_form": "const_lo"},
       "D6-18 c0 interp in ln G": {"c0_form": "interp_logG"}, "D6-18 multi-wave remainder at c0(k)": {"c0_multi": "last"},
       "D6-20 X_sat 16 DS ratio (v6d)": {"x16": "ds_ratio"}, "D6-20 kappa 16 PC2-R1e as read": {"kap": "own+pcfill"},
       "D6-20 L table 16 PC2's own": {"L16": "pc2"}, "D6-21 kappa 4/SM v6d (PC by m, held)": {"kap": "v6d"},
       "D6-21 kappa PC by fill on every tile": {"kap": "pcfill"}, "D6-21 kappa fill pooled": {"kap": "fill_pooled"},
       "D6-21 X_sat 4/SM PC2-R3": {"x32": "pc2_r3"}, "D6-24 naive v4d' (D6-13)": {"naive": "v4dp"},
       "D6-24 naive twin on every cell": {"naive": "twin"}, "D6-24 naive fetchers only": {"naive": "fetchers"},
       "D6-24 naive all@fetch (R4 literal)": {"naive": "all@fetch"}}


def lines() -> list[str]:
    out = []
    for k, d in ALT.items():
        lw = L2.Law(**L2.DESIGN_D6).build() if d == "design" else L2.variant(**d)
        out.append(L2.cal2_line(k, L2.cal2(lw)))
    return out


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "alternatives", sys.argv[1:]))
