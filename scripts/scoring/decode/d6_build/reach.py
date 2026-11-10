#!/usr/bin/env python
"""Where the new blind's GEMMs sit under the frozen v6d (configs and addendum 5's registered page lists only; no page
of the three models exists): d6common.new_model_reach. Computed after the freeze; it chose nothing.

    python scripts/scoring/decode/d6_build/reach.py [--write | --check]"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import d6common as D6  # noqa: E402
import d6v as L2  # noqa: E402


def lines() -> list[str]:
    return [f"{m}: {json.dumps(v, sort_keys=True)}" for m, v in D6.new_model_reach().items()]


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "reach", sys.argv[1:]))
