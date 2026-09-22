#!/usr/bin/env python
"""The Python half of scripts/alpha_g_chain.sh: every read of a report.json, a
log or a ledger that the chain needs, as one small command each, so the shell
stays a sequencer and the reading is testable off GPU.

    alpha_g_chain_helpers.py run-id LOG EXPERIMENT  -> the run id the plan page printed
    alpha_g_chain_helpers.py estimate LOG           -> seconds the arm's own --dry-run priced
    alpha_g_chain_helpers.py reading REPORT         -> G, seed, ratio, lo, hi, exit word, duty
    alpha_g_chain_helpers.py pairs REPORT...        -> the reports that FORMED a ratio, one per line
    alpha_g_chain_helpers.py eta REPORT             -> eta, lo, hi, band, exit word (R1)
    alpha_g_chain_helpers.py verdict LOG            -> what its RESULT lines imply (2nd opinion)

Every command prints tab-separated fields on ONE line (or one path per line for
`pairs`) and exits 0; a report that cannot be read prints `unreadable` in the
first field rather than crashing the chain.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from moe.bench import exit_codes  # noqa: E402

#: The ratio arm's registered bands for the clock elasticity, read from the
#: arm that defines them: never restated here.
RUN_ID_LINE = re.compile(r"^experiment\s+(\S+)\s*/\s*(\S+)\s*$", re.M)
ESTIMATE_LINES = (
    re.compile(r"takes about (\d+) s"),                  # R3 at a duty below 1: the wall figure
    re.compile(r"estimated GPU time (\d+) s"),           # R3 at full duty
    re.compile(r"estimated wall time (\d+) s"),          # R1
)


def _load(path: str) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _exit_word(payload: dict) -> str:
    gates = payload.get("gates") or []
    if not gates:
        return "unscored"
    try:
        rc = exit_codes.classify((g["kind"], g.get("tag") or g.get("number"), g["verdict"])
                                 for g in gates)
    except Exception:                                             # noqa: BLE001
        return "unscored"
    return exit_codes.CODE_NAMES.get(rc, str(rc))


def run_id(log: str, experiment: str) -> str:
    """The run id the arm's plan page printed for `experiment`, or ''."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    for name, rid in RUN_ID_LINE.findall(text):
        if name == experiment:
            return rid
    return ""


def estimate(log: str) -> str:
    """Seconds the arm's own --dry-run priced, off its plan page; '' if none."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return ""
    for pattern in ESTIMATE_LINES:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return ""


def reading(report: str) -> list[str]:
    """`G seed ratio lo hi exit duty run_id` for a private_weight_reference report."""
    p = _load(report)
    if p is None or p.get("experiment") != "private_weight_reference":
        return ["unreadable"]
    iv = p.get("ratio_interval") or [None, None]
    fmt = lambda v: "none" if v is None else f"{float(v):.4f}"  # noqa: E731
    return [str((p.get("pinned") or {}).get("GROUP_SIZE_M", "?")),
            str(p.get("seed") if p.get("seed") is not None else "unrecorded"),
            fmt(p.get("ratio")), fmt(iv[0]), fmt(iv[1]), _exit_word(p),
            str(p.get("duty", 1.0)), str(p.get("run_id") or Path(report).parent.name)]


def pairs(reports: list[str]) -> list[str]:
    """The reports a later seed may be scored WITH: this arm's, measured, with a
    ratio and an interval formed. The same rule `load_replicates` applies, so
    a seed-1 line never names a report that would refuse it."""
    out = []
    for r in reports:
        p = _load(r)
        if p is None or p.get("experiment") != "private_weight_reference":
            continue
        if p.get("synthetic"):
            continue
        iv = p.get("ratio_interval") or [None, None]
        if p.get("ratio") is None or iv[0] is None or iv[1] is None:
            continue
        out.append(r)
    return out


def eta(report: str) -> list[str]:
    """`eta lo hi band exit` for a clock_elasticity report: the per-M-tile
    elasticity the arm registered and the band its interval landed in, read
    through the arm's own `band_of` so the edges are never restated."""
    p = _load(report)
    if p is None or "elasticity" not in p:
        return ["unreadable"]
    est = p["elasticity"] or {}
    value, lo, hi = est.get("value"), est.get("lo"), est.get("hi")
    band = "unresolved"
    if lo is not None and hi is not None:
        import clock_elasticity as CE
        got = CE.band_of(float(lo), float(hi))
        band = got[0] if got else "STRADDLES"
    fmt = lambda v: "none" if v is None else f"{float(v):.4f}"  # noqa: E731
    return [fmt(value), fmt(lo), fmt(hi), band, _exit_word(p)]


def verdict(log: str) -> str:
    """What the log's RESULT lines imply, as an exit code, or NONE / UNREADABLE."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return "UNREADABLE"
    try:
        return str(exit_codes.classify_text(text))
    except exit_codes.NoGatesScored:
        return "NONE"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "run-id" and len(rest) == 2:
        print(run_id(rest[0], rest[1]))
    elif cmd == "estimate" and len(rest) == 1:
        print(estimate(rest[0]))
    elif cmd == "reading" and len(rest) == 1:
        print("\t".join(reading(rest[0])))
    elif cmd == "pairs":
        print("\n".join(pairs(rest)))
    elif cmd == "eta" and len(rest) == 1:
        print("\t".join(eta(rest[0])))
    elif cmd == "verdict" and len(rest) == 1:
        print(verdict(rest[0]))
    else:
        print(f"unknown command {argv!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
