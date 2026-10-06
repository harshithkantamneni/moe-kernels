#!/usr/bin/env python
"""Score rental 4's perturbation gate (docs/registered/2026-10-06-rental4-perturb-gh200.json):
recompute every variant's gate off perturb.json, and name the instrumented units it voids.

    python scripts/scoring/rental4/score_perturb.py <repo> <tree> <out>

Reads `*-<card>-perturb-<label>/perturb.json` and gate.env under <tree>. Writes
<out>/perturb.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

PART = "perturb"


def find(tree: Path) -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{C4.CARD}-perturb-*/perturb.json"))
    return hits[0] if len(hits) == 1 else None


def score(repo: Path, tree: Path) -> dict:
    reg = C4.registration(repo, PART)
    res = {"registration": C4.NAMES[PART], "variants": {}}
    path = find(tree)
    if path is None:
        res["verdict"] = "NOT SCORED: no single perturb.json (every instrumented unit is void)"
        return res
    rep = json.loads(path.read_text())
    up = rep.get("upstream") or {}
    up_ok = bool(up.get("file_matches") and up.get("excerpt_matches"))
    gate_env = {}
    env = path.parent / "gate.env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            gate_env[k] = v.split()[0] if v else ""
    for vid, v in (rep.get("variants") or {}).items():
        got = C4.perturb_verdict([c.get("ratio") for c in v.get("cells") or []], v.get("configs") or [],
                                 upstream_ok=up_ok, sass_equal=v.get("sass_equal"), hint_ok=v.get("hint_ok"),
                                 tol=reg["tolerance"])
        key = "GATE_" + "".join(c if c.isalnum() else "_" for c in vid)
        on_vm = gate_env.get(key)
        res["variants"][vid] = {**got, "kind": v.get("kind"), "on_the_vm": on_vm,
                                "agrees_with_vm": on_vm == got["verdict"]}
    void = [k for k, v in res["variants"].items() if v["verdict"] != "PASS"]
    res["void_units"] = void
    res["upstream"] = up
    disagree = [k for k, v in res["variants"].items() if not v["agrees_with_vm"]]
    res["verdict"] = ("NOT SCORED: no variant" if not res["variants"] else
                      f"PASS for {len(res['variants']) - len(void)} of {len(res['variants'])}"
                      + (f"; VOID {void}" if void else "")
                      + (f"; the VM's gate.env DISAGREES on {disagree}" if disagree else ""))
    return res


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 4 PERTURBATION GATE, {res['registration']}"]
    for k, v in res["variants"].items():
        med = "-" if v["median_dev"] is None else f"{100 * v['median_dev']:.2f}%"
        worst = "-" if v["worst_dev"] is None else f"{100 * v['worst_dev']:.2f}%"
        out.append(f"  {k:14s} {v['verdict']} median {med} worst {worst} (VM {v['on_the_vm']}) {'; '.join(v['why'])}")
    out.append(f"  verdict: {res['verdict']}")
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
