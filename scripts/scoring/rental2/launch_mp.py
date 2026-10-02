"""rental 1's max-plus composition model MP, the P6 function of rental 2's launch
registration (docs/registered/2026-10-01-rental2-launch2-gh200.json, P6).

`timeline` and `mp` below are copied VERBATIM from scripts/scoring/rental1/
score_launch.py (the text between `def timeline` and its P6 block);
tests/test_scoring_rental2.py pins that the two texts are identical. mp(te, tg,
H_us) already scales the trace's host offsets by H_us / P_traced: rental 2
passes H_us = H_cell, as rental 1 did.
"""
import gzip
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import launch_floor as LF  # noqa: E402  the design's own trace helpers


def timeline(path):
    """Per traced call: host times (relative to its start-event record) of the
    flush kernel launch return, the start record return, each call-kernel launch
    return, the end record return; the period to the next start record; kernel
    durations; flush duration; lambda (parse_trace's)."""
    tr = json.load(gzip.open(path))
    ev = sorted(LF._events(tr), key=lambda e: float(e["ts"]))
    calls = [e for e in ev if LF._is_annotation(e, "r3call")]
    flushes = [e for e in ev if LF._is_annotation(e, "r3flush")]
    runtime = [e for e in ev if LF._is_runtime(e)]
    kernels = defaultdict(list)
    for e in ev:
        if LF._is_kernel(e) and LF._corr(e) is not None:
            kernels[LF._corr(e)].append(e)
    end_ = lambda e: float(e["ts"]) + float(e["dur"])
    recs = []
    prev_end = float("-inf")
    for c in calls:
        t0, t1 = float(c["ts"]), end_(c)
        fl = [f for f in flushes if prev_end <= float(f["ts"]) < t0]
        fl = fl[-1] if fl else None
        lo = end_(fl) if fl else prev_end
        rec = [r for r in runtime if LF._is_event_record(r) and lo <= float(r["ts"]) < t0]
        s = end_(rec[-1]) if rec else t0
        erec = [r for r in runtime if LF._is_event_record(r) and float(r["ts"]) >= t1]
        launches = [r for r in runtime if LF._is_launch(r) and t0 <= float(r["ts"]) <= t1]
        ks = sorted((k for r in launches for k in kernels.get(LF._corr(r), [])), key=lambda k: float(k["ts"]))
        lmap = {LF._corr(r): r for r in launches}
        rec_ = dict(s=s, erec=end_(erec[0]) - s if erec else None,
                    kl=[end_(lmap[LF._corr(k)]) - s for k in ks],
                    kd=[float(k["dur"]) for k in ks])
        if fl is not None:
            fls = [r for r in runtime if LF._is_launch(r) and float(fl["ts"]) <= float(r["ts"]) <= end_(fl)]
            fk = [k for r in fls for k in kernels.get(LF._corr(r), [])]
            rec_["fl"] = end_(fls[-1]) - s if fls else None
            rec_["F"] = sum(float(k["dur"]) for k in fk) if fk else 0.0
        else:
            rec_["fl"], rec_["F"] = None, 0.0
        recs.append(rec_)
        prev_end = t1
    for a, b in zip(recs, recs[1:]):
        a["P"] = b["s"] - a["s"]
    use = [r for r in recs[1:-1]]  # drop the first (cold) and the last (no period)
    nk = st.mode([len(r["kd"]) for r in use])
    use = [r for r in use if len(r["kd"]) == nk and r["erec"] is not None]
    med = lambda xs: st.median(xs)
    out = dict(P=med([r["P"] for r in use]), erec=med([r["erec"] for r in use]),
               kl=[med([r["kl"][j] for r in use]) for j in range(nk)],
               kd=[med([r["kd"][j] for r in use]) for j in range(nk)],
               F=med([r["F"] for r in use]),
               fl=med([r["fl"] for r in use]) if use[0]["fl"] is not None else None,
               lam=LF.parse_trace(tr)["median"]["lambda_us"], ncalls=len(use))
    return out


def mp(te, tg, H_us, iters=50):
    """Max-plus: host iteration i at i*H, offsets scaled by H / P_traced; GPU
    durations from TR-G (kernels) and TR-E (flush); each op starts at
    max(previous GPU end, host launch + lambda). Median end - start over the
    steady state (second half)."""
    sc = H_us / te["P"]
    lam = te["lam"]
    kd = tg["kd"] if len(tg["kd"]) == len(te["kl"]) else te["kd"]
    g = -1e9
    I = []
    for i in range(iters):
        t = i * H_us
        if te["fl"] is not None:
            g = max(g, t + te["fl"] * sc + lam) + te["F"]
        start = max(g, t + 0.0 + lam)  # the start record is the anchor (offset 0)
        g = start
        for o, d in zip(te["kl"], kd):
            g = max(g, t + o * sc + lam) + d
        end = max(g, t + te["erec"] * sc + lam)
        g = end
        I.append(end - start)
    return st.median(I[iters // 2:]) / 1e3, sc
