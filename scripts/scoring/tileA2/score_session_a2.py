#!/usr/bin/env python
"""Read tile rental A2's session record into A2's score directory (docs/registered/2026-10-20-tileA2-gates-gh200):
the lock rung the lock-hold probe selected and the GPU power limit the prelude set and read back, the two
values rental B2 runs at (the owner's decision of 2026-10-20; finalise_a2.py reads them from here only).

    python scripts/scoring/tileA2/score_session_a2.py <repo> <tree> <out>

<tree> is the run's results directory; the session is read beside it (<tree>/../session):
gh200-driver/a2-lock.env (A2_LOCK), clocks-at-start.txt (power.limit, read back after -pl) and
power-set.txt (nvidia-smi -q -d POWER after -pl: the first Current Power Limit, the GPU scope). The two
power readings must agree within 1 W, and the selected rung's gate record gh200-driver/gate-probe-<L>.json must
read PASS with no earlier rung PASS (the prelude writes A2_LOCK=1710 before the probe, so a failed ladder leaves
1710 there: re-review MINOR 3), else NOT READ. Writes <out>/session.score.{json,txt}.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402

TC = A2.TC
PART = "session"


def power_limits(session: Path) -> tuple[float | None, float | None]:
    a = b = None
    p = session / "clocks-at-start.txt"
    if p.exists():
        rows = list(csv.reader(p.read_text().splitlines()))
        if len(rows) >= 2:
            head = [h.strip() for h in rows[0]]
            i = next((k for k, h in enumerate(head) if h.startswith("power.limit")), None)
            if i is not None and len(rows[1]) > i:
                m = re.match(r"\s*([\d.]+)", rows[1][i])
                a = float(m.group(1)) if m else None
    q = session / "power-set.txt"
    if q.exists():
        m = re.search(r"Current Power Limit\s*:\s*([\d.]+)\s*W", q.read_text())
        b = float(m.group(1)) if m else None
    return a, b


def score(repo: Path, tree: Path) -> dict:
    session = Path(tree).parent / "session"
    lock = A2.lock_of_session(tree)
    a, b = power_limits(session)
    why = []
    if lock is None:
        why.append("no gh200-driver/a2-lock.env A2_LOCK")
    elif lock not in A2.LOCK_LADDER:
        why.append(f"A2_LOCK {lock:g} is not a rung of {list(A2.LOCK_LADDER)}")
    if a is None or b is None:
        why.append(f"power limit not read (clocks-at-start {a}, power-set {b})")
    elif abs(a - b) > 1.0:
        why.append(f"the two power readings disagree ({a} and {b} W)")
    probes = {}
    for L in A2.LOCK_LADDER:
        g = session / "gh200-driver" / f"gate-probe-{int(L)}.json"
        if g.exists():
            import json
            probes[f"{L:g}"] = json.loads(g.read_text()).get("verdict")
    if lock is not None and not why:
        passed = [k for k, v in probes.items() if v == "PASS"]
        if not passed:
            why.append("no rung held (no gate-probe record reads PASS)")
        elif passed[0] != f"{lock:g}":
            why.append(f"the first rung that held is {passed[0]}, the session names {lock:g}")
    return {"registration": A2.NAMES_A2["gates"], "part": PART,
            "verdict": "READ" if not why else "NOT READ: " + "; ".join(why),
            "lock_mhz": lock, "power_limit_w": a if not why else None,
            "readings": {"clocks-at-start power.limit": a, "power-set Current Power Limit (GPU)": b},
            "probe rungs (gate records)": probes, "session": str(session)}


def lines(res: dict) -> list[str]:
    return [f"READ against {res['registration']}: {res['verdict']}",
            f"  lock {res['lock_mhz']} MHz; power limit {res['power_limit_w']} W; probe rungs {res['probe rungs (gate records)']}"]


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
