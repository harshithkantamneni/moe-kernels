"""scripts/paper/lock_refit.py: the gate audit's FL1 / V10 refit, in the repo."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import lock_refit as LR  # noqa: E402


def test_refit_regenerates_the_committed_table():
    rows = LR.rows()
    assert json.loads(json.dumps(rows)) == json.loads((ROOT / "docs/paper/data/lock_refit.json").read_text())
    fits = [o for r in rows if r["lock"] == 1710 for o in r["gemms"]]
    assert len(rows) == 98 and len(fits) == 196
    # the degenerate corner: every 1710 fit with t0 <= 4.4 us reads f under 1676 MHz
    assert max(o["f"] for o in fits if o["t0"] <= 4.4) < 1676.0
    # t0 >= 12 us: 1691.4 to 1717.3 MHz apart from the 8x22B floor's unlocked capture
    hi = [o["f"] for r in rows if r["lock"] == 1710 and "8x22b" not in r["file"]
          for o in r["gemms"] if o["t0"] >= 12]
    assert (round(min(hi), 1), round(max(hi), 1)) == (1691.4, 1717.3)
