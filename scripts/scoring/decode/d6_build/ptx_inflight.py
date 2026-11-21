#!/usr/bin/env python
"""Q_B per decode tile (addendum 6; design-D6's work/ptx_inflight.py, committed): the B tiles a CTA keeps in flight,
read off every published fused_moe PTX with PC's own rule (pccommon.inflight_stages, pinned and unchanged); the tile
from the TTGIR dot operand shapes and the compile JSON's warps and stages. DESCRIPTIVE record of SEEN files.

    python scripts/scoring/decode/d6_build/ptx_inflight.py [--write | --check]"""
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "scripts/scoring/pc"))
import pccommon as P  # noqa: E402

DOT = re.compile(r"tt\.dot\s.*?:\s*tensor<(\d+)x(\d+)x[a-z0-9]+.*?>\s*\*\s*tensor<(\d+)x(\d+)x")
#: the published sessions scanned: those present at addendum 6's build (PC2's publication 8294278f)
UNTIL = "2026-11-10-nvidia_gh200_480gb-pc2-session"


def scan() -> dict:
    out = {}
    for p in sorted((REPO / "results/published").rglob("fused_moe_kernel.ptx")):
        sess = p.relative_to(REPO / "results/published").parts[0]
        if sess > UNTIL:
            continue
        d = p.parent
        try:
            m = json.loads((d / "fused_moe_kernel.json").read_text())
            g = (d / "fused_moe_kernel.ttgir").read_text()
        except OSError:
            continue
        dm = DOT.search(g)
        if not dm:
            continue
        bm, bk, _bk2, bn = map(int, dm.groups())
        key = f"{bm}x{bn}x{bk}w{m['num_warps']}s{m['num_stages']}"
        r = P.inflight_stages(p.read_text())
        out.setdefault(key, {}).setdefault(f"n_if {r['n_if']} (commit groups {r['groups']}, wait {r['wait']})", []).append(
            str(d.relative_to(REPO)))
    return out


def lines() -> list[str]:
    out = scan()
    L = [f"{sum(len(v) for x in out.values() for v in x.values())} published PTX read"]
    for k in sorted(out):
        for v, ds in sorted(out[k].items()):
            L.append(f"{k:16s} {v}: {len(ds)} files, e.g. {ds[0]}")
    return L


def n_if() -> dict:
    """{tile: n_if} where every file of the tile reads one n_if (else the tile is absent)."""
    out = {}
    for k, v in scan().items():
        vals = {x.split()[1] for x in v}
        if len(vals) == 1 and "None" not in vals:
            out[k] = int(vals.pop())
    return out


if __name__ == "__main__":
    t = "\n".join(lines()) + "\n"
    o = HERE / "ptx_inflight.out"
    if "--write" in sys.argv:
        o.write_text(t)
        print(f"wrote {o.name}")
    elif "--check" in sys.argv:
        same = o.exists() and o.read_text() == t
        print(f"{o.name}: {'same' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    else:
        sys.stdout.write(t)
