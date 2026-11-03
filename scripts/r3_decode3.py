#!/usr/bin/env python
"""The decode page with rental 2's address control (registration 2026-10-29-decode-addendum3-gh200, settling
unit M3): scripts/r3_decode.py, byte-identical, imported and run, with one added flag.

    python scripts/r3_decode3.py --pad-rows P <every r3_decode.py argument>

--pad-rows P   every weight slot allocated with P extra rows (private_weight_reference.build_private_weights'
               own `pad_rows`, rental 2's `--slot-pad-rows`): w1 [2F + P, H] and w2 [H + P, F], the weights the
               views [:, :2F] and [:, :H]. The bytes read are unchanged; only stride(0), the distance between
               expert slots, moves, so every expert but slot 0 sits at another address. P = 0 is r3_decode.py.
The page's plan prints the padding, and report.json carries {"d3": {"pad_rows": P, ...}} beside r3_decode's
own fields; nothing else changes (the instruments, the gates, the timing and the readback are r3_decode's).
No kernel, tiling or scheduling code: this builds inputs only.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_decode as RD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

FLAG = "--pad-rows"


def split_argv(argv: list[str]) -> tuple[int, list[str]]:
    """(P, the rest): --pad-rows P (or --pad-rows=P) taken out once; P a whole number."""
    a, pad, seen = list(argv), 0, 0
    out = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == FLAG or x.startswith(FLAG + "="):
            seen += 1
            v = x.split("=", 1)[1] if "=" in x else (a[i + 1] if i + 1 < len(a) else "")
            i += 1 if "=" in x else 2
            if not v.isdigit():
                raise ValueError(f"{FLAG} {v!r}: a whole number of rows")
            pad = int(v)
            continue
        out.append(x)
        i += 1
    if seen > 1:
        raise ValueError(f"{FLAG} given {seen} times")
    return pad, out


def main(argv=None) -> int:
    try:
        pad, rest = split_argv(list(sys.argv[1:] if argv is None else argv))
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    PWR = RD.PWR
    orig_build, orig_plan, orig_report = PWR.build_private_weights, RD.plan_lines, RD.write_report
    PWR.build_private_weights = functools.partial(orig_build, pad_rows=pad)

    def plan_lines(*a, **k):
        return orig_plan(*a, **k) + [f"pad rows    {pad} per expert slot (r3_decode3.py, rental 2's address control: "
                                     "stride(0) moves, the bytes read do not)"]

    def write_report(args, cfg, page, result, *, out_dir, **k):
        rc = orig_report(args, cfg, page, result, out_dir=out_dir, **k)
        p = Path(out_dir) / "report.json"
        if p.exists():
            doc = json.loads(p.read_text())
            doc["d3"] = {"pad_rows": pad, "wrapper": "scripts/r3_decode3.py", "registration": "2026-10-29-decode-addendum3-gh200"}
            p.write_text(json.dumps(doc, indent=2, default=str))
        return rc
    RD.plan_lines, RD.write_report = plan_lines, write_report
    try:
        return RD.main(rest)
    finally:
        PWR.build_private_weights, RD.plan_lines, RD.write_report = orig_build, orig_plan, orig_report


if __name__ == "__main__":
    sys.exit(main())
