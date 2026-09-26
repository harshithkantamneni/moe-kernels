#!/usr/bin/env python3
"""R3's ratio pages at G = 1 2 4 16 64 with the SM clock LOCKED, at the highest lock
this card holds during R3's bursts (Lambda H100, 2026-09-25).

Why: on the GH200 the duty-0.25 pages failed V7 (the arms ran at different clocks), and a
1965 MHz lock was pulled to 1785-1830 MHz by the power cap in the first cells, while 1710
held exactly. So the lock is chosen by trying it on the run itself: lock L, start R3 at
G=1, and read every timed cell's under-load clock (cells.csv sm_clock_load_mhz) as it
lands. A cell more than one 15 MHz step under L means the lock does not hold: the run is
stopped, the next lower lock is tried, and G=1 starts again. A slip at a later G restarts
the ladder from G=1 at the next lower lock, so every kept page shares one clock.
Status lines: ~/moe/session/locked-r3/status. Summary: ~/moe/session/locked-r3/summary.json.
"""
import csv, glob, json, os, re, signal, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
REPO = os.environ["REPO"]; PY = os.environ["PY_VLLM"]
RES = Path(os.environ["MOE_RESULTS_DIR"]) / "private_weight_reference"
BASE = os.environ.get("CHAIN_SESSION", "alpha_g-h100")
LADDER = [int(x) for x in os.environ.get("LADDER", "1980 1890 1800 1710 1620 1530").split()]
GS = [int(x) for x in os.environ.get("GS", "1 2 4 16 64").split()]
STEP = 15
STAGE_CAP_S = int(os.environ.get("STAGE_CAP_S", str(4 * 3600)))
D = HOME / "moe/session/locked-r3"; D.mkdir(parents=True, exist_ok=True)
T0 = time.time()
summary = {"ladder_asked": LADDER, "attempts": [], "pages": [], "lock": None}


def status(msg):
    with open(D / "status", "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")


def save():
    (D / "summary.json").write_text(json.dumps(summary, indent=1))


def smi(args, out=None):
    r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True)
    if out:
        (D / out).write_text(r.stdout + r.stderr)
    return r


def supported():
    txt = smi(["-q", "-d", "SUPPORTED_CLOCKS"], "supported-clocks.txt").stdout
    return sorted({int(m) for m in re.findall(r"Graphics\s*:\s*(\d+) MHz", txt)})


def lock(L):
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-lgc", f"{L},{L}"], capture_output=True, text=True)
    (D / f"lgc-{L}.txt").write_text(r.stdout + r.stderr)
    smi(["-q", "-d", "CLOCK,PERFORMANCE,POWER"], f"clocks-locked-{L}.txt")
    return r.returncode == 0


def reset():
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True, text=True)
    (D / "rgc.txt").write_text(r.stdout + r.stderr)
    smi(["-q", "-d", "CLOCK"], "clocks-after-reset.txt")


def gpu_idle(cap_s=300):
    t = time.time()
    while time.time() - t < cap_s:
        if not smi(["--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
            return True
        time.sleep(5)
    return False


def run_dir(G, t_start):
    best = None
    for d in glob.glob(str(RES / f"*-duty0.25-g{G}-*")):
        c = Path(d) / "cells.csv"
        if c.exists() and c.stat().st_mtime >= t_start - 5:
            if best is None or c.stat().st_mtime > (Path(best) / "cells.csv").stat().st_mtime:
                best = d
    return best


def slipped(d, L):
    """(n_cells_read, worst clock, first slip or None) over the run's timed cells."""
    try:
        rows = list(csv.DictReader(open(Path(d) / "cells.csv")))
    except Exception:
        return 0, None, None
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get("sm_clock_load_mhz") or "nan"))
        except ValueError:
            pass
    vals = [v for v in vals if v == v and v > 0]
    if not vals:
        return 0, None, None
    bad = [v for v in vals if v < L - STEP]
    return len(vals), min(vals), (bad[0] if bad else None)


def stop(p):
    subprocess.run(["pkill", "-INT", "-s", str(p.pid)])
    try:
        p.wait(timeout=90)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-KILL", "-s", str(p.pid)])
        p.wait()


def run_g(G, L, tag):
    t_start = time.time()
    log = open(D / f"r3-g{G}-lock{L}.log", "w")
    argv = ["timeout", "--signal=INT", "--kill-after=60", "1800", PY, "scripts/private_weight_reference.py",
            "--model", "mixtral-8x7b", "--block-m", "32", "--treads", "6", "--repeats", "9",
            "--group-m", str(G), "--duty", "0.25", "--seed", "0", "--session-tag", tag]
    p = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    d = None
    while p.poll() is None:
        time.sleep(10)
        d = d or run_dir(G, t_start)
        if d:
            n, worst, bad = slipped(d, L)
            if bad is not None:
                stop(p)
                return {"G": G, "lock": L, "state": "slipped", "cells": n, "worst_mhz": worst,
                        "slip_mhz": bad, "dir": d, "seconds": round(time.time() - t_start)}
        if time.time() - T0 > STAGE_CAP_S:
            stop(p)
            return {"G": G, "lock": L, "state": "stage-cap", "dir": d, "seconds": round(time.time() - t_start)}
    d = d or run_dir(G, t_start)
    n, worst, bad = slipped(d, L) if d else (0, None, None)
    return {"G": G, "lock": L, "state": "slipped" if bad is not None else "done", "rc": p.returncode,
            "cells": n, "worst_mhz": worst, "slip_mhz": bad, "dir": d, "seconds": round(time.time() - t_start)}


def main():
    sup = supported()
    summary["supported_graphics_mhz"] = sup
    ladder = []
    for c in LADDER:
        below = [s for s in sup if s <= c]
        if below and below[-1] not in ladder:
            ladder.append(below[-1])
    if not ladder:
        ladder = LADDER
    summary["ladder"] = ladder
    status(f"lock ladder {ladder} (supported max {sup[-1] if sup else '?'})")
    smi_log = open(D / "smi.csv", "w")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,"
                                "temperature.gpu,utilization.gpu,clocks_event_reasons.active", "--format=csv",
                                "-lms", "500"], stdout=smi_log, stderr=subprocess.STDOUT)
    rc = 1
    try:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-pm", "1"], capture_output=True)
        for L in ladder:
            if not gpu_idle():
                status("GPU not idle after 300 s; stopping"); break
            if not lock(L):
                status(f"lock {L} refused by nvidia-smi; next"); summary["attempts"].append({"lock": L, "state": "refused"}); continue
            status(f"locked at {L} MHz")
            tag = f"{BASE}-lock{L}"
            ok = True; pages = []
            for G in GS:
                status(f"G={G} start at lock {L}")
                r = run_g(G, L, tag); summary["attempts"].append(r); save()
                status(f"G={G} {r['state']} rc={r.get('rc')} cells={r.get('cells')} worst={r.get('worst_mhz')} {r.get('seconds')} s")
                if r["state"] != "done":
                    ok = False; break
                pages.append(r)
            if ok:
                summary["lock"] = L; summary["pages"] = pages; rc = 0
                status(f"all G done at lock {L} MHz"); break
            if summary["attempts"][-1]["state"] == "stage-cap":
                status("stage time cap reached; stopping"); break
            status(f"lock {L} did not hold; stepping down")
        else:
            status("no lock on the ladder held")
    finally:
        reset(); sampler.terminate(); save()
        status(f"clock reset; DONE rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
