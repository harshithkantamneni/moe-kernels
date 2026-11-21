#!/usr/bin/env python
"""D5-2's alternative: v5d with SUBWAVE, DRAIN and W1 on every path, the naive path included (r3_timing_model_v5d's
GEMM_TERM_PATHS widened in memory; never registered), against v5d as registered. CAL2 and, beside it, the SEEN decode
blind re-score (POST-HOC on seen pages; the every-path re-score was computed before D5-2 was chosen on CAL2).
python every_path.py [--write | --check]"""
import contextlib
import sys

import d5b


@contextlib.contextmanager
def every_path():
    keep = d5b.V5D.GEMM_TERM_PATHS
    d5b.V5D.GEMM_TERM_PATHS = keep + (d5b.D5.RD.NAIVE,)
    try:
        yield
    finally:
        d5b.V5D.GEMM_TERM_PATHS = keep


if __name__ == "__main__":
    L = [d5b.line(d5b.run(d5b.BASE, "v5d (aligned paths, registered)")),
         d5b.line(d5b.run(d5b.BASE, "every path (D5-2's alternative)", ctx=every_path))]
    sys.exit(d5b.emit(L, "every_path", sys.argv[1:]))
