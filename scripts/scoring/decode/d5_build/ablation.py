#!/usr/bin/env python
"""Addendum 5's ablation (the registration's 'ablation' block): v4d', v5d, each term alone and every term but one.
CAL2 refitted each time; the SEEN decode blind re-score beside is POST-HOC on seen pages, a profile of each feature,
never a choice.  python ablation.py [--write | --check]"""
import sys

import d5b

B = d5b.BASE
VARIANTS = (("v4d' (no term)", d5b.V5D.NONE), ("v5d (every term)", B), ("regimes alone", B.only("regime")),
            ("W1 alone", B.only("w1_scale")), ("CALL alone", B.only("call")), ("ALIGN with SLOPE alone", B.only("align")),
            ("every term but the regimes", B.only("w1_scale", "call", "align")),
            ("every term but W1", B.only("regime", "call", "align")),
            ("every term but CALL", B.only("regime", "w1_scale", "align")),
            ("every term but ALIGN", B.only("regime", "w1_scale", "call")))

if __name__ == "__main__":
    sys.exit(d5b.emit([d5b.line(d5b.run(t, k)) for k, t in VARIANTS], "ablation", sys.argv[1:]))
