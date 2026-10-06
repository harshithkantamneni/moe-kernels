#!/usr/bin/env python
"""HARDWARE CONSTANTS FROM THIRD-PARTY MICROBENCHMARKS: RRZE-HPC/gpu-benches on the VM
(rental 4, owner decision 3 of 2026-10-06).

    python scripts/gpubench.py --dest DIR --out DIR [--sm 90] [--lock-mhz 1710] [--l2-mib 60]
    python scripts/gpubench.py --dry-run ...         # the commands; nothing fetched or run (exit 2)
    python scripts/gpubench.py --parse-only DIR      # re-read raw outputs into constants.json

THIRD-PARTY CODE, NOT VENDORED. gpu-benches is GPL-3.0 (Dominik Ernst, RRZE-HPC). This
script FETCHES it on the VM at the pinned commit below into --dest, which must lie outside
the measured checkout (the driver's prelude refuses a dirty checkout, and nothing GPL enters
this repository). It builds three benchmarks with their own Makefiles (nvcc, SM given) and
runs each once, unmodified, under a time cap with its stdout written as it streams (each
prints a line per data point, so a capped run keeps every row it reached). Only their
printed numbers are read here; the program output is a measurement, not their source.

WHAT EACH GIVES, measured independently of any fused_moe page:
  gpu-latency   one warp's pointer chase over a growing buffer: latency (cycles at the
                clock it reads, and ns) of the near L2 partition, the far L2 partition and
                DRAM. lambda_far is the w2 epilogue's one exposed far-L2 hit in the
                re-analysis (REANALYSIS.md 1b, published GH200 plateau 247.5 ns).
  gpu-stream    init / read / scale / triad / 3pt / 5pt bandwidth against occupancy (a
                shared-memory spoiler holds 2 CTAs per SM; block size scans occupancy):
                the DRAM ruler (triad) and bandwidth vs occupancy.
  gpu-l2-cache  per-SM blocks re-reading a buffer of growing total volume: the L2
                bandwidth plateau, the DRAM floor and the volume where it falls half-way
                (an effective L2 capacity, REANALYSIS.md 1d).

The rules that turn rows into constants are fixed here (and in the registration
docs/registered/2026-10-06-rental4-gpubench-gh200.json) before any rental-4 page:
windows in units of the card's L2 size, medians over them, half-way crossings.

EXIT CODES: 0 every bench built, ran and parsed; 2 refused (bad arguments, --dest inside a
git checkout, no nvcc, the fetched commit is not the pin; and --dry-run); 3 a bench failed
to build or run or parse (the others' files are written).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from moe.bench import exit_codes  # noqa: E402

#: the pin: master on 2026-10-06 (git ls-remote), "Merge pull request #18 ... make-cuda-sm-autodetect"
PIN = {"repo": "https://github.com/RRZE-HPC/gpu-benches", "commit": "23e586dd8d8a19e72abd9d385a088623cefabfae",
       "committed": "2025-10-24T09:08:33+02:00", "licence": "GPL-3.0",
       "note": "third-party; fetched on the VM at this commit, never vendored into moe-kernels"}
#: (directory, the Makefile's CUDA target, cap seconds). gpu-latency's own sweep runs to 2 GB
#: with host-side shuffles; its cap keeps every row past 4 x L2 (DRAM) and stops the rest
BENCHES = (("gpu-latency", "cuda-latency", 420), ("gpu-stream", "cuda-stream", 240),
           ("gpu-l2-cache", "cuda-l2-cache", 300))
MIB = 1 << 20

# the rules (registered): windows in units of the L2 size L
LAT_NEAR = (4 * MIB, 0.40)          # bytes from 4 MiB to 0.40 L: the near partition
LAT_FAR = (0.60, 0.95)              # 0.60 L to 0.95 L: the far-partition plateau
LAT_DRAM = 2.5                      # >= 2.5 L: DRAM
L2C_PLATEAU = 0.25                  # total volume <= 0.25 L: the L2 bandwidth plateau
L2C_FLOOR = 4.0                     # >= 4 L: the DRAM floor
STREAM_KNEE = 0.95                  # the least occupancy at which triad reaches 95% of its peak
STREAM_KERNELS = ("init", "read", "scale", "triad", "3pt", "5pt")


class Refused(Exception):
    pass


# --------------------------------------------------------------------------
# parsers (pure)
# --------------------------------------------------------------------------

def _floats(line: str) -> list[float] | None:
    try:
        return [float(x) for x in line.split()]
    except ValueError:
        return None


def parse_latency(text: str, l2_bytes: float) -> dict:
    """gpu-latency rows `iters clockMHz KiB ms cyc_mean cyc_med cyc_p5 cyc_p95`; the
    plateaus as medians of ns = cyc_med / clock over the registered windows."""
    rows = []
    for line in text.splitlines():
        v = _floats(line)
        if v and len(v) == 8 and v[1] > 0:
            rows.append({"bytes": v[2] * 1024, "clock_mhz": v[1], "cycles": v[5],
                         "ns": v[5] / v[1] * 1e3})
    if not rows:
        raise ValueError("no gpu-latency row")

    def window(lo, hi):
        xs = [r["ns"] for r in rows if lo <= r["bytes"] <= hi]
        return (statistics.median(xs), len(xs)) if xs else (None, 0)
    near, n_near = window(LAT_NEAR[0], LAT_NEAR[1] * l2_bytes)
    far, n_far = window(LAT_FAR[0] * l2_bytes, LAT_FAR[1] * l2_bytes)
    dram, n_dram = window(LAT_DRAM * l2_bytes, math.inf)
    clocks = sorted({r["clock_mhz"] for r in rows})
    return {"rows": len(rows), "clock_mhz_read": clocks, "max_bytes": max(r["bytes"] for r in rows),
            "near_l2_ns": near, "far_l2_ns": far, "dram_ns": dram,
            "cells": {"near": n_near, "far": n_far, "dram": n_dram}}


def parse_stream(text: str) -> dict:
    """gpu-stream rows `block smBlocks threads occ% | GB/s: init read scale triad 3pt 5pt`."""
    rows = []
    for line in text.splitlines():
        if "|" not in line or "GB/s:" not in line:
            continue
        left, right = line.split("|", 1)
        lv = left.replace("%", "").split()
        rv = right.replace("GB/s:", "").split()
        try:
            rows.append({"block": int(lv[0]), "occupancy": float(lv[3]) / 100,
                         **{k: float(x) for k, x in zip(STREAM_KERNELS, rv[:6], strict=True)}})
        except (IndexError, ValueError):
            continue
    if not rows:
        raise ValueError("no gpu-stream row")
    peak = {k: max(r[k] for r in rows) for k in STREAM_KERNELS}
    full = max(rows, key=lambda r: (r["occupancy"], r["block"]))
    knee = min((r["occupancy"] for r in rows if r["triad"] >= STREAM_KNEE * peak["triad"]), default=None)
    return {"rows": len(rows), "peak_gbps": peak, "at_full_occupancy_gbps": {k: full[k] for k in STREAM_KERNELS},
            "triad_knee_occupancy": knee,
            "bandwidth_vs_occupancy": [{k: r[k] for k in ("occupancy", "triad", "read")} for r in rows]}


def parse_l2cache(text: str, l2_bytes: float) -> dict:
    """gpu-l2-cache rows `<block kB> kB <total kB> kB <ms>ms <spread>% <bw> GB/s ...`: the
    plateau (total <= 0.25 L), the floor (total >= 4 L) and the half-way volume, read on a
    log-volume interpolation between the last row above and the first row below it."""
    rows = []
    for line in text.splitlines():
        t = line.replace("kB", " ").replace("ms", " ").replace("%", " ").replace("GB/s", " ").split()
        v = _floats(" ".join(t))
        if v and len(v) >= 5:
            rows.append({"bytes": v[1] * 1024, "gbps": v[4]})
    if not rows:
        raise ValueError("no gpu-l2-cache row")
    rows.sort(key=lambda r: r["bytes"])
    plat = [r["gbps"] for r in rows if r["bytes"] <= L2C_PLATEAU * l2_bytes]
    flo = [r["gbps"] for r in rows if r["bytes"] >= L2C_FLOOR * l2_bytes]
    out = {"rows": len(rows), "plateau_gbps": statistics.median(plat) if plat else None,
           "floor_gbps": statistics.median(flo) if flo else None, "half_way_bytes": None}
    if plat and flo:
        half = (out["plateau_gbps"] + out["floor_gbps"]) / 2
        for a, b in zip(rows, rows[1:], strict=False):
            if a["gbps"] >= half > b["gbps"]:
                f = (a["gbps"] - half) / (a["gbps"] - b["gbps"])
                out["half_way_bytes"] = math.exp(math.log(a["bytes"]) + f * (math.log(b["bytes"]) - math.log(a["bytes"])))
                break
    if out["half_way_bytes"] is not None:
        out["half_way_over_l2"] = out["half_way_bytes"] / l2_bytes
    return out


def constants(raw: dict, *, l2_bytes: float, lock_mhz: float | None) -> dict:
    """The hardware constants out of the three raw texts (any may be missing)."""
    out = {"l2_bytes": l2_bytes, "lock_mhz": lock_mhz, "parsed": {}, "errors": {}}
    for name, fn in (("gpu-latency", lambda t: parse_latency(t, l2_bytes)),
                     ("gpu-stream", parse_stream),
                     ("gpu-l2-cache", lambda t: parse_l2cache(t, l2_bytes))):
        if raw.get(name) is None:
            out["errors"][name] = "no output"
            continue
        try:
            out["parsed"][name] = fn(raw[name])
        except ValueError as exc:
            out["errors"][name] = str(exc)
    lat = out["parsed"].get("gpu-latency")
    if lat and lock_mhz:
        out["cycles_at_lock"] = {k: (None if lat[k] is None else lat[k] * lock_mhz / 1e3)
                                 for k in ("near_l2_ns", "far_l2_ns", "dram_ns")}
    return out


# --------------------------------------------------------------------------
# the run (the VM)
# --------------------------------------------------------------------------

def in_git_checkout(path: Path) -> bool:
    p = Path(path).resolve()
    for q in (p, *p.parents):
        if (q / ".git").exists():
            return True
    return False


def cuda_home() -> str:
    """The toolkit root: $CUDA_HOME, else nvcc's grandparent, else /usr/local/cuda."""
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    nv = shutil.which("nvcc")
    return str(Path(nv).resolve().parents[1]) if nv else "/usr/local/cuda"


def commands(dest: Path, sm: int, home: str | None = None) -> list[list[str]]:
    """What --run does, in order (the dry run prints them). CUDA_HOME rides on the make
    command line: gpu-l2-cache's Makefile sets `CUDA_HOME :=` from nvcc's path, which only a
    command-line variable overrides (build-r4-review section 4)."""
    home = home or cuda_home()
    out = [] if (dest / ".git").exists() else [["git", "clone", "--quiet", PIN["repo"], str(dest)]]
    out += [["git", "-C", str(dest), "fetch", "--quiet", "origin", PIN["commit"]],
            ["git", "-C", str(dest), "checkout", "--quiet", "--detach", PIN["commit"]],
            ["git", "-C", str(dest), "rev-parse", "HEAD"]]
    for d, target, _cap in BENCHES:
        out.append(["make", "-C", str(dest / d), f"SM={sm}", f"CUDA_HOME={home}", target])
    for d, target, cap in BENCHES:
        out.append(["timeout", str(cap), str(dest / d / target)])
    return out


def _smi_clock() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unread: {exc}"


def run(args) -> int:
    dest, out = Path(args.dest), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"pin": PIN, "sm": args.sm, "lock_mhz": args.lock_mhz, "l2_mib": args.l2_mib,
           "steps": [], "started": time.time()}
    bad = 0

    def sh(cmd, cap=600, stdout=None):
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdout=stdout or subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=cap)
            rc, text = p.returncode, (p.stdout if stdout is None else "")
        except subprocess.TimeoutExpired:
            rc, text = 124, ""
        log["steps"].append({"cmd": cmd, "rc": rc, "secs": round(time.time() - t0, 1),
                             "tail": (text or "")[-2000:]})
        return rc, text

    for cmd in commands(dest, args.sm)[:-2 * len(BENCHES)]:
        rc, text = sh(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"] and text.strip() != PIN["commit"]:
            log["refused"] = f"the fetched HEAD is {text.strip()!r}, not the pin {PIN['commit']}"
            (out / "gpubench-log.json").write_text(json.dumps(log, indent=1))
            print(f"REFUSED: {log['refused']}")
            return exit_codes.REFUSED
        if rc != 0:
            log["refused"] = f"{' '.join(cmd)} exited {rc}"
            (out / "gpubench-log.json").write_text(json.dumps(log, indent=1))
            print(f"REFUSED: {log['refused']}")
            return exit_codes.REFUSED
    built = {}
    for d, target, _cap in BENCHES:
        rc, _ = sh(["make", "-C", str(dest / d), f"SM={args.sm}", f"CUDA_HOME={cuda_home()}", target], cap=900)
        built[d] = rc == 0 and (dest / d / target).exists()
        bad += not built[d]
    raw = {}
    for d, target, cap in BENCHES:
        if not built[d]:
            continue
        log.setdefault("clocks", {})[d] = {"before": _smi_clock()}
        path = out / f"{d}.txt"
        with open(path, "w") as fh:
            rc, _ = sh(["timeout", str(cap), str(dest / d / target)], cap=cap + 60, stdout=fh)
        log["clocks"][d]["after"] = _smi_clock()
        log["steps"][-1]["capped"] = rc == 124
        raw[d] = path.read_text()
        if rc not in (0, 124) or not raw[d].strip():
            bad += 1
    cons = constants(raw, l2_bytes=args.l2_mib * MIB, lock_mhz=args.lock_mhz)
    cons["pin"] = PIN
    bad += len([e for e in cons["errors"] if e in raw])
    (out / "constants.json").write_text(json.dumps(cons, indent=1, default=str))
    log["ended"] = time.time()
    (out / "gpubench-log.json").write_text(json.dumps(log, indent=1))
    print(f"GPUBENCH {PIN['commit'][:12]}: {sum(built.values())}/{len(BENCHES)} built, "
          f"constants {out / 'constants.json'}")
    return exit_codes.INVALID if bad else exit_codes.DONE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", type=Path, help="where gpu-benches is fetched: outside any git checkout")
    p.add_argument("--out", type=Path, help="the unit's results directory")
    p.add_argument("--sm", type=int, default=90, help="nvcc's compute capability (GH200: 90)")
    p.add_argument("--lock-mhz", type=float, default=None,
                   help="the SM lock in force (the driver sets it), recorded and used for cycles")
    p.add_argument("--l2-mib", type=float, default=60.0, help="the card's L2 (GH200: 60 MiB)")
    p.add_argument("--parse-only", type=Path, default=None, metavar="DIR",
                   help="re-read DIR/<bench>.txt into DIR/constants.json; nothing runs")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.parse_only:
        d = Path(args.parse_only)
        raw = {b: (d / f"{b}.txt").read_text() for b, _t, _c in BENCHES if (d / f"{b}.txt").exists()}
        cons = constants(raw, l2_bytes=args.l2_mib * MIB, lock_mhz=args.lock_mhz)
        (d / "constants.json").write_text(json.dumps(cons, indent=1, default=str))
        print(json.dumps(cons.get("cycles_at_lock") or cons["parsed"].keys().__repr__()))
        return exit_codes.DONE
    if not args.dest or not args.out:
        print("REFUSED: --dest and --out are required (or --parse-only DIR)")
        return exit_codes.REFUSED
    if in_git_checkout(args.dest):
        print(f"REFUSED: --dest {args.dest} lies inside a git checkout: the GPL code stays out of it")
        return exit_codes.REFUSED
    print(f"GPU-BENCHES (GPL-3.0, third party) at {PIN['commit']} into {args.dest}; SM {args.sm}; "
          f"lock {args.lock_mhz} MHz; L2 {args.l2_mib} MiB")
    for cmd in commands(Path(args.dest), args.sm):
        print("  " + " ".join(cmd))
    if args.dry_run:
        print("DRY RUN: nothing fetched, built or run (exit 2)")
        return exit_codes.REFUSED
    if shutil.which("nvcc") is None and Path("/usr/local/cuda/bin/nvcc").exists():
        # the toolkit's usual home on a Lambda image, off PATH in a non-login shell
        os.environ["PATH"] = "/usr/local/cuda/bin" + os.pathsep + os.environ.get("PATH", "")
    for tool in ("git", "make", "nvcc", "timeout"):
        if shutil.which(tool) is None:
            print(f"REFUSED: no {tool} on PATH")
            return exit_codes.REFUSED
    os.environ.setdefault("CUDA_HOME", cuda_home())
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
