#!/usr/bin/env python3
"""How far along is a running sweep, and when will it finish?

Reads the manifest (cells finished) and the CSV timestamps (when they finished)
for a run still in flight. No GPU work and no interference with the sweep: both
files are append-only and are read, not locked.

The estimate is deliberately crude in one direction and honest about it. Cells
are not equal: deepseek-v3 at T=8192 costs orders of magnitude more than mixtral
at T=1, and profiles walk models in order. A rate averaged over the run so far
will UNDERSTATE remaining time whenever the expensive models come last, so the
recent-window rate is reported next to the overall one.

    python scripts/sweep_progress.py --results /workspace/results --profile full
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse(ts: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def read_times(csv_path: Path) -> list[dt.datetime]:
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            t = _parse(r.get("timestamp", ""))
            if t:
                out.append(t)
    return sorted(out)


def count_manifest(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue        # a torn last line while the sweep is writing
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    return counts


def rate_per_hour(times: list[dt.datetime], window: int | None = None) -> float | None:
    """Rows per hour, over the whole run or the most recent `window` rows."""
    pts = times[-window:] if window else times
    if len(pts) < 2:
        return None
    span = (pts[-1] - pts[0]).total_seconds()
    return None if span <= 0 else (len(pts) - 1) / span * 3600.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("/workspace/results"))
    ap.add_argument("--run-id", default=None, help="default: the newest manifest")
    ap.add_argument("--planned", type=int, default=None,
                    help="planned cell count, from `--dry-run`")
    ap.add_argument("--window", type=int, default=200,
                    help="rows in the recent-rate window")
    args = ap.parse_args()

    mans = sorted(args.results.glob(f"run_{args.run_id or '*'}_*.manifest.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not mans:
        print(f"no manifests under {args.results}")
        return 1
    man = mans[-1]
    csv_path = Path(str(man).replace(".manifest.jsonl", ".csv"))

    counts = count_manifest(man)
    times = read_times(csv_path)
    done = sum(counts.values())

    print(f"manifest {man.name}")
    print(f"  cells recorded   {done}")
    for k, v in sorted(counts.items()):
        print(f"    {k:<22} {v}")
    print(f"  rows written     {len(times)}")
    if times:
        elapsed = (times[-1] - times[0]).total_seconds() / 3600.0
        stale = (dt.datetime.now() - times[-1]).total_seconds() / 60.0
        print(f"  elapsed          {elapsed:.2f} h")
        print(f"  last row         {stale:.1f} min ago"
              f"{'   <-- LOOKS STALLED' if stale > 20 else ''}")
        overall = rate_per_hour(times)
        recent = rate_per_hour(times, args.window)
        if overall:
            print(f"  rate overall     {overall:.0f} rows/h")
        if recent:
            print(f"  rate recent      {recent:.0f} rows/h  (last {args.window} rows)")
        if args.planned and overall and recent and done < args.planned:
            left = args.planned - done
            per_cell = len(times) / max(done, 1)
            rows_left = left * per_cell
            print(f"\n  cells left       {left} of {args.planned}"
                  f"  ({done / args.planned * 100:.1f}% done)")
            for label, r in (("overall rate", overall), ("recent rate", recent)):
                print(f"  ETA on {label:<13} {rows_left / r:.1f} h")
            print("\n  The two differ when cost per cell is changing. Profiles walk")
            print("  models in order, so the recent rate is the one to trust while")
            print("  a larger model is in flight.")
        elif not args.planned:
            print("\n  pass --planned N for an ETA; get N from:")
            print("    python -m moe.bench.cli --profile <name> --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
