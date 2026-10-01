"""Score the 2026-10-01 launch-floor registration (docs/registered/README.md, files
docs/registered/2026-10-01-launch-floor-gh200.{json,txt} at 436e41c) on rental 1's
pages. Read-only on the run tree; prints score.txt and writes score.json.

Nothing here is fitted. Readings of the registered text that needed a choice are
listed under "readings" in the JSON and in the text output.
"""
import csv, gzip, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(sys.argv[1])
PAGES = Path(sys.argv[2])
OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO / "scripts"))
REG = json.load(open(REPO / "docs/registered/2026-10-01-launch-floor-gh200.json"))
PRED = REG["predictions"]
CREG = REG["C_reg_ms"]
MODELS = ("granite-3.0-3b-a800m", "jetmoe-8b", "granite-3.0-1b-a400m")
ARMS = ("native", "shared", "private")
MIB = 2 ** 20
RULER = 3.7261e12  # B/s, the session ruler the registration used (0.2517e9/3.7261e12 = 0.0676 ms)
F_MODE = {"E240": 0.0676, "E480": 480 * MIB / RULER * 1e3}  # ms; E240 as registered, E480 = 2 x bytes
out_lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)


# ---------------------------------------------------------------- cells
def load_cells(model):
    rows = list(csv.DictReader(open(PAGES / model / "cells.csv")))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["mode"], r["arm"], int(r["tiles"]))].append(r)
    cells = {}
    for k, rs in agg.items():
        assert all(r["status"] == "ok" for r in rs), k
        I = st.median(float(r["ms_p50"]) for r in rs)
        H = st.median(float(r["host_enqueue_ms"]) / float(r["calls_per_burst"]) for r in rs)
        hb = sum(r["host_bound"] == "True" for r in rs) / len(rs)
        clk = sorted({r["sm_clock_load_mhz"] for r in rs})
        cells[k] = dict(I=I, H=H, hbfrac=hb, n_rep=len(rs), clk=clk,
                        I_rep=[float(r["ms_p50"]) for r in rs])
    return cells


C = {m: load_cells(m) for m in MODELS}
PARSED = {m: json.load(open(PAGES / m / "traces" / "parsed.json")) for m in MODELS}


def creg(model, arm, n, key=None):
    return CREG[key or model][arm][str(n)]


def edge(c):
    return 0.2 <= c["hbfrac"] <= 0.8


score = {"readings": [], "P": {}}
R = score["readings"]

# ---------------------------------------------------------------- P1
p("== P1  GR follows C_reg, Granite-3B n <= 5 ([0.80, 1.10] x C_reg)")
m = "granite-3.0-3b-a800m"
bands = PRED["P1"]["bands_ms"]
rows, outside, rival = [], 0, []
for arm in ARMS:
    for n in range(1, 6):
        c = C[m][("GR", arm, n)]
        lo, hi = bands[arm][str(n)]
        ok = lo <= c["I"] <= hi
        outside += not ok
        if n <= 3 and c["I"] >= 0.20:
            rival.append((arm, n))
        rows.append(dict(arm=arm, n=n, GR=round(c["I"], 4), Creg=creg(m, arm, n),
                         ratio=round(c["I"] / creg(m, arm, n), 3), band=[lo, hi], inside=ok,
                         GR_hbfrac=c["hbfrac"]))
        p(f"  {arm:7s} n{n}  GR {c['I']:.4f}  C_reg {creg(m, arm, n):.4f}  x{c['I']/creg(m, arm, n):.3f}  band {lo:.4f}-{hi:.4f}  {'in' if ok else 'OUT'}  (GR host-bound frac {c['hbfrac']:.2f})")
inc = {arm: C[m][("GR", arm, 5)]["I"] - C[m][("GR", arm, 1)]["I"] for arm in ARMS}
p("  increments I_GR(5)-I_GR(1):", {a: round(v, 4) for a, v in inc.items()}, "(>= 0.10 required for shared, private)")
p(f"  outside: {outside} of 15 (fail if > 1); rival line (GR n<=3 >= 0.20 ms): {rival or 'none'}")
p1_fail = outside > 1 or inc["shared"] < 0.10 or inc["private"] < 0.10
score["P"]["P1"] = dict(verdict="FALSIFIED" if p1_fail else "HELD", cells=rows, outside=outside,
                        increments=inc, rival_cells=rival)
p("  P1:", score["P"]["P1"]["verdict"])

# ---------------------------------------------------------------- P2
p("\n== P2  GR = E240 on GPU-bound cells, max(2%, 6 us); Granite n >= 7, JetMoE n >= 3")
rows, bad = [], []
for m, ns in (("granite-3.0-3b-a800m", range(7, 10)), ("jetmoe-8b", range(3, 5))):
    for arm in ARMS:
        for n in ns:
            g, e = C[m][("GR", arm, n)], C[m][("E240", arm, n)]
            d = g["I"] - e["I"]
            band = max(0.02 * e["I"], 0.006)
            ok = abs(d) <= band
            rows.append(dict(model=m, arm=arm, n=n, GR=round(g["I"], 4), E240=round(e["I"], 4),
                             diff_us=round(d * 1e3, 1), rel=round(d / e["I"], 4), band_us=round(band * 1e3, 1),
                             inside=ok, E240_hbfrac=e["hbfrac"]))
            if not ok:
                bad.append((m, arm, n))
            p(f"  {m[:12]:12s} {arm:7s} n{n}  GR {g['I']:.4f} E240 {e['I']:.4f}  diff {d*1e3:+6.1f} us ({d/e['I']*100:+.2f}%)  band {band*1e3:.1f}  {'in' if ok else 'OUT'}  E240 hb {e['hbfrac']:.2f}")
alt = [r for r in rows if r["E240_hbfrac"] < 0.2]
alt_bad = [(r["model"], r["arm"], r["n"]) for r in alt if not r["inside"]]
score["P"]["P2"] = dict(verdict="FALSIFIED" if bad else "HELD", cells=rows, outside=bad,
                        alt_reading_measured_gpu_bound=dict(cells=len(alt), outside=alt_bad))
p(f"  alternative reading (only cells whose E240 is measured GPU-bound, hb < 0.2): {len(alt)-len(alt_bad)}/{len(alt)} inside; outside {alt_bad}")
p("  P2:", score["P"]["P2"]["verdict"], f"({len(bad)} of {len(rows)} outside)")


R.append("P2: scored on the registered cell list (Granite n >= 7, JetMoE n >= 3, every arm) as committed; Granite native and shared n = 7 read host-bound at E240 on this run (H about 0.43 ms), and the result on only the cells measured GPU-bound is printed beside it")
# ---------------------------------------------------------------- P3
p("\n== P3  E0 - E240 = F_trace - h_flush_trace - dD_tail, Granite n = 1, 2, +/- 12 us")
m = "granite-3.0-3b-a800m"
PM = PARSED[m]
rows, out3 = [], 0
for arm in ARMS:
    for n in (1, 2):
        keys = [f"TR-E-E240-{arm}-n{n}.json.gz", f"TR-G-E0-{arm}-n{n}.json.gz", f"TR-G-E240-{arm}-n{n}.json.gz"]
        if not all(k in PM for k in keys):
            rows.append(dict(arm=arm, n=n, scored="MP only (trace missing)"))
            continue
        te = PM[keys[0]]["median"]
        F, hf = te["flush_us"], te["h_flush_us"]
        dD = PM[keys[1]]["median"]["d_tail_us"] - PM[keys[2]]["median"]["d_tail_us"]
        pred = (F - hf - dD) / 1e3
        # alternative reading: D_tail from the eager traces (TR-E) of the same cell
        dDe = PM[f"TR-E-E0-{arm}-n{n}.json.gz"]["median"]["d_tail_us"] - te["d_tail_us"]
        pred_e = (F - hf - dDe) / 1e3
        meas = C[m][("E0", arm, n)]["I"] - C[m][("E240", arm, n)]["I"]
        ok = abs(meas - pred) <= 0.012
        out3 += not ok
        rows.append(dict(arm=arm, n=n, measured_ms=round(meas, 4), F_us=round(F, 1), h_flush_us=round(hf, 1),
                         dD_tail_TRG_us=round(dD, 1), predicted_ms=round(pred, 4), miss_us=round((meas - pred) * 1e3, 1),
                         inside=ok, alt_dD_tail_TRE_us=round(dDe, 1), alt_predicted_ms=round(pred_e, 4),
                         alt_miss_us=round((meas - pred_e) * 1e3, 1),
                         first_order_ms=round((F - hf) / 1e3, 4),
                         E0_hb=C[m][("E0", arm, n)]["hbfrac"], E240_hb=C[m][("E240", arm, n)]["hbfrac"]))
        p(f"  {arm:7s} n{n}  E0-E240 {meas*1e3:+6.1f} us | F {F:.1f} - h_flush {hf:.1f} - dD_tail(TR-G) {dD:+.1f} = {pred*1e3:+6.1f}  miss {(meas-pred)*1e3:+6.1f}  {'in' if ok else 'OUT'}"
          f" | alt dD_tail(TR-E) {dDe:+.1f} -> {pred_e*1e3:+6.1f} miss {(meas-pred_e)*1e3:+6.1f} | hb E0 {C[m][('E0',arm,n)]['hbfrac']:.2f} E240 {C[m][('E240',arm,n)]['hbfrac']:.2f}")
score["P"]["P3"] = dict(verdict="FALSIFIED" if out3 > 1 else "HELD", cells=rows, outside=out3)
p("  P3:", score["P"]["P3"]["verdict"], f"({out3} of 6 outside; fail if > 1)")
R.append("P3: dD_tail read from TR-G as the registration's inputs line says; the graph trace's D_tail includes the GPU backlog the replay loop builds, so an alternative from TR-E is printed beside it (not scored)")

# ---------------------------------------------------------------- P4
p("\n== P4  E480: moved verdict and shift -(F480 - F240) +/- 12 us, Granite non-edge cells")
m = "granite-3.0-3b-a800m"
pr480 = PRED["P4"]["predicted_host_bound_at_E480"]
pr240 = PRED["P4"]["predicted_host_bound_at_E240"]
# F480 - F240 from the traces (TR-E at E480 and E240, n = 1, 2, every arm; the flush is arm-independent)
dF_tr = st.median(PM[f"TR-E-E480-{a}-n{n}.json.gz"]["median"]["flush_us"] - PM[f"TR-E-E240-{a}-n{n}.json.gz"]["median"]["flush_us"]
                  for a in ARMS for n in (1, 2)) / 1e3
p(f"  F480 - F240 from traces (median of 6 TR-E pairs): {dF_tr*1e3:.1f} us (bytes/ruler {(F_MODE['E480']-F_MODE['E240'])*1e3:.1f} us)")
rows, wrong, wrong_rule, shift_bad, edges = [], [], [], [], []
for arm in ARMS:
    for n in range(1, 10):
        e4, e2 = C[m][("E480", arm, n)], C[m][("E240", arm, n)]
        r = dict(arm=arm, n=n, E240=round(e2["I"], 4), E480=round(e4["I"], 4), hb240=round(e2["hbfrac"], 2),
                 hb480=round(e4["hbfrac"], 2), H480=round(e4["H"], 4), H240=round(e2["H"], 4))
        if edge(e4):
            edges.append(("E480", arm, n))
            r["edge480"] = True
        else:
            meas = e4["hbfrac"] >= 0.8
            predl = n in pr480[arm]
            predr = creg(m, arm, n) + F_MODE["E480"] < e4["H"]
            r.update(pred_list=predl, pred_rule_ownH=predr, measured_hb=meas)
            if meas != predl:
                wrong.append((arm, n))
            if meas != predr:
                wrong_rule.append((arm, n))
        if edge(e2):
            edges.append(("E240", arm, n))
        if not edge(e4) and not edge(e2) and e4["hbfrac"] >= 0.8 and e2["hbfrac"] >= 0.8:
            sh = e4["I"] - e2["I"]
            r.update(shift_us=round(sh * 1e3, 1), shift_miss_us=round((sh + dF_tr) * 1e3, 1),
                     shift_miss_nominal_us=round((sh + F_MODE["E480"] - F_MODE["E240"]) * 1e3, 1))
            if abs(sh + dF_tr) > 0.012:
                shift_bad.append((arm, n))
        rows.append(r)
        p(f"  {arm:7s} n{n}  E240 {e2['I']:.4f} hb {e2['hbfrac']:.2f} (pred {'hb' if n in pr240[arm] else 'gpu'}) | E480 {e4['I']:.4f} hb {e4['hbfrac']:.2f} H {e4['H']:.4f} pred-list {'hb' if n in pr480[arm] else 'gpu'} rule {'hb' if creg(m,arm,n)+F_MODE['E480']<e4['H'] else 'gpu'}"
          + (f" | shift {r['shift_us']:+.1f} us miss {r['shift_miss_us']:+.1f}" if "shift_us" in r else ""))
p4_fail = len(wrong_rule) > 2 or bool(shift_bad)
score["P"]["P4"] = dict(verdict="FALSIFIED" if p4_fail else "HELD", cells=rows, verdict_wrong_vs_list=wrong,
                        verdict_wrong_vs_rule_ownH=wrong_rule, shift_outside=shift_bad, edges=edges, dF_trace_ms=dF_tr)
p(f"  wrong E480 verdicts vs rule with own H (scored): {wrong_rule} (fail if > 2); vs the printed list (H 0.3545): {wrong}; shifts outside +/-12 us: {shift_bad}; edge cells: {edges}")
p("  P4:", score["P"]["P4"]["verdict"])
R.append("P4: the moved verdict is scored with each cell's own H (C_reg + F480 < H_cell), as the JSON's fitted_on_what.H says (the 0.3545 ms behind the printed lists native n<=4, shared n<=3, private n<=2 is 'for the printed estimates only'); the printed list is reported beside it. Shift scored on cells host-bound and non-edge at both E240 and E480, against -(F480-F240) from the traces (median of the six n = 1, 2 TR-E pairs); the bytes/ruler value is printed too")

# ---------------------------------------------------------------- P5
p("\n== P5  C_reg + F < H (own H) gives the host-bound verdict, non-edge cells, >= 95%")
p5 = {}
tot_ok = tot = 0
for m, modes, keys in (("granite-3.0-3b-a800m", ("E240", "E480"), (None,)),
                       ("jetmoe-8b", ("E240", "E480"), (None, "jetmoe-8b@6d2595d")),
                       ("granite-3.0-1b-a400m", ("E240",), (None,))):
    for key in keys:
        ok = n_ = 0
        miss, edges = [], []
        for (mode, arm, n), c in sorted(C[m].items()):
            if mode not in modes:
                continue
            if edge(c):
                edges.append((mode, arm, n, round(c["hbfrac"], 2)))
                continue
            pred = creg(m, arm, n, key) + F_MODE[mode] < c["H"]
            meas = c["hbfrac"] >= 0.8
            n_ += 1
            ok += pred == meas
            if pred != meas:
                miss.append((mode, arm, n, round(creg(m, arm, n, key), 4), round(c["H"], 4), "meas hb" if meas else "meas gpu"))
        lab = key or m
        p5[lab] = dict(agree=ok, of=n_, miss=miss, edges=edges)
        p(f"  {lab:22s} {ok}/{n_} ({ok/n_*100:.1f}%)  misses {miss}  edges {edges}")
        if key is None:
            tot_ok += ok
            tot += n_
p(f"  pooled (primary C_reg): {tot_ok}/{tot} ({tot_ok/tot*100:.1f}%)")
per_model_ok = all(v["agree"] / v["of"] >= 0.95 for k, v in p5.items() if "@" not in k)
score["P"]["P5"] = dict(verdict="HELD" if (tot_ok / tot >= 0.95 and per_model_ok) else
                        ("HELD (pooled) / FALSIFIED on a model" if tot_ok / tot >= 0.95 else "FALSIFIED"),
                        per_model=p5, pooled=[tot_ok, tot])
p("  P5:", score["P"]["P5"]["verdict"])
R.append("P5: F at E480 is 480 MiB / the registration's ruler 3.7261 TB/s = 0.1351 ms (the registration's C* 0.219 = 0.3545 - 0.1355); JetMoE uses the primary 09-29 C_reg, the 6d2595d CORES row printed beside it; H is each cell's median host enqueue per call (host_enqueue_ms / calls_per_burst)")


# ---------------------------------------------------------------- MP (P6)
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


p("\n== P6  MP predicts I_eager within max(5%, 12 us), every E240/E0/E480 cell at the trace treads")
TT = {"granite-3.0-3b-a800m": (1, 2, 4, 6, 9), "jetmoe-8b": (1, 2, 3), "granite-3.0-1b-a400m": (1, 6, 9)}
rows, bad, sub_rows, sub_bad = [], [], [], []
for m, ns in TT.items():
    tdir = PAGES / m / "traces"
    for arm in ARMS:
        for n in ns:
            for mode in ("E240", "E0", "E480"):
                if (mode, arm, n) not in C[m]:
                    continue
                c = C[m][(mode, arm, n)]
                fe, fg = tdir / f"TR-E-{mode}-{arm}-n{n}.json.gz", tdir / f"TR-G-{mode}-{arm}-n{n}.json.gz"
                own = fe.exists() and fg.exists()
                if not own:  # substitution: E240's traces, the mode's flush
                    fe, fg = tdir / f"TR-E-E240-{arm}-n{n}.json.gz", tdir / f"TR-G-E240-{arm}-n{n}.json.gz"
                te, tg = timeline(fe), timeline(fg)
                if not own:
                    te = dict(te)
                    if mode == "E0":
                        te["fl"], te["F"] = None, 0.0
                    else:
                        te["F"] = te["F"] + dF_tr * 1e3
                pred, sc = mp(te, tg, c["H"] * 1e3)
                band = max(0.05 * c["I"], 0.012)
                ok = abs(pred - c["I"]) <= band
                r = dict(model=m, arm=arm, n=n, mode=mode, measured=round(c["I"], 4), MP=round(pred, 4),
                         miss_us=round((pred - c["I"]) * 1e3, 1), rel=round(pred / c["I"] - 1, 4),
                         band_us=round(band * 1e3, 1), inside=ok, hbfrac=round(c["hbfrac"], 2),
                         scale=round(sc, 3), H=round(c["H"], 4), own_traces=own,
                         lam_us=round(te["lam"], 1), kernel_sum_us=round(sum(tg["kd"]), 1))
                (rows if own else sub_rows).append(r)
                if not ok:
                    (bad if own else sub_bad).append((m, arm, n, mode))
                p(f"  {m[:12]:12s} {mode:4s} {arm:7s} n{n}  meas {c['I']:.4f} MP {pred:.4f} miss {(pred-c['I'])*1e3:+6.1f} us ({(pred/c['I']-1)*100:+5.1f}%) band {band*1e3:.1f} {'in' if ok else 'OUT'}  hb {c['hbfrac']:.2f} scale {sc:.3f}{'' if own else '  [E240 traces, mode flush substituted]'}")
score["P"]["P6"] = dict(verdict="FALSIFIED" if bad else "HELD", cells=rows, outside=bad,
                        substituted_cells=sub_rows, substituted_outside=sub_bad)
p(f"  own-trace cells: {len(rows) - len(bad)}/{len(rows)} inside; outside {bad}")
p(f"  substituted cells (E0/E480 at treads without their own traces): {len(sub_rows) - len(sub_bad)}/{len(sub_rows)} inside; outside {sub_bad}")
p("  P6:", score["P"]["P6"]["verdict"])
R.append("P6: MP anchors each iteration at its start-event record; host offsets (flush launch return, call-kernel launch returns, end-record return) and the iteration period from TR-E of the cell's own mode, scaled by H_cell / P_traced (P_traced = median start-record to start-record period); kernel durations from TR-G of the same mode, the flush duration from TR-E; lambda is parse_trace's median per-call least lambda; median interval over iterations 25..49 of 50. E0 and E480 cells at the trace treads 4, 6, 9 (Granite) and 3 (JetMoE) have no E0/E480 traces: they are scored separately on E240's traces with the mode's flush substituted, and that set is reported, not counted in the verdict")

# ---------------------------------------------------------------- P7
p("\n== P7  Granite n = 1 classification (gates nothing)")
m = "granite-3.0-3b-a800m"
p7 = {}
for arm in ARMS:
    tg = PM[f"TR-G-E240-{arm}-n1.json.gz"]["median"]
    te_all = PM[f"TR-E-E240-{arm}-n1.json.gz"]
    te = te_all["median"]
    # eager host span of fused_experts itself: the vllm::fused_experts cpu_op inside r3call
    tr = json.load(gzip.open(PAGES / m / "traces" / f"TR-E-E240-{arm}-n1.json.gz"))
    fe = [float(e["dur"]) for e in LF._events(tr) if e.get("name") == "vllm::fused_experts"]
    fe_med = st.median(fe[1:]) if len(fe) > 1 else None
    share = te_all["launch_api_share"]
    # launch-API share of the fused_experts span (same launches, its own span as denominator)
    share_fe = st.median(sum(c["launch_api_us"]) / f for c, f in zip(te_all["calls"][1:], fe[1:])) if fe_med else None
    k_ok, ng_ok = tg["kernel_sum_us"] <= 130, tg["non_gemm_us"] <= 30
    h_ok = (te["host_span_us"] >= 250, fe_med is not None and fe_med >= 250)
    p7[arm] = dict(TRG_kernel_sum_us=round(tg["kernel_sum_us"], 1), TRG_gemm_us=round(tg["gemm_us"], 1),
                   TRG_non_gemm_us=round(tg["non_gemm_us"], 1), TRE_r3call_host_span_us=round(te["host_span_us"], 1),
                   TRE_fused_experts_span_us=round(fe_med, 1) if fe_med else None,
                   launch_api_share_of_r3call=round(share, 3), launch_api_share_of_fused_experts=round(share_fe, 3) if share_fe else None,
                   unprofiled_H_E240_ms=round(C[m][("E240", arm, 1)]["H"], 4), GR_ms=round(C[m][("GR", arm, 1)]["I"], 4),
                   kernel_sum_le_130=k_ok, non_gemm_le_30=ng_ok)
    p(f"  {arm:7s} TR-G kernels {tg['kernel_sum_us']:.1f} us (GEMM {tg['gemm_us']:.1f}, non-GEMM {tg['non_gemm_us']:.1f}) | TR-E r3call span {te['host_span_us']:.1f} us, fused_experts span {fe_med:.1f} us, launch-API share {share*100:.1f}% of r3call / {share_fe*100:.1f}% of fused_experts | unprofiled H {C[m][('E240',arm,1)]['H']*1e3:.1f} us")
allk = all(v["kernel_sum_le_130"] and v["non_gemm_le_30"] and v["TRE_r3call_host_span_us"] >= 250 and v["launch_api_share_of_r3call"] < 0.40 for v in p7.values())
score["P"]["P7"] = dict(verdict="CLASSIFIES", all_four_criteria_met=allk,
                        classification="host Python dispatch: GPU work about 0.08 ms, launch-API about 11% of the host span" if allk else "mixed", cells=p7)
p(f"  P7: all four criteria met on every arm: {allk}")
R.append("P7: traced host spans run about 2x the unprofiled H (profiler overhead); the >= 0.25 ms span criterion is read on the traced span as written, and the unprofiled H is printed beside it")

# ---------------------------------------------------------------- P8
p("\n== P8  Granite-1B (unseen): GR in P1's rule, P5's rule, plateau H - Phi in [0.23, 0.27] (seen)")
m = "granite-3.0-1b-a400m"
gb = PRED["P8"]["GR_bands_ms"]
rows, outside = [], 0
for arm in ARMS:
    for n in range(1, 10):
        c = C[m][("GR", arm, n)]
        lo, hi = gb[arm][str(n)]
        ok = lo <= c["I"] <= hi
        outside += not ok
        e = C[m][("E240", arm, n)]
        rows.append(dict(arm=arm, n=n, GR=round(c["I"], 4), Creg=creg(m, arm, n), ratio=round(c["I"] / creg(m, arm, n), 3),
                         band=[lo, hi], inside=ok, E240=round(e["I"], 4), E240_hb=round(e["hbfrac"], 2), H=round(e["H"], 4),
                         pred_hb_session=n in PRED["P8"]["predicted_host_bound_at_E240_with_H_0.3545"][arm]))
        p(f"  {arm:7s} n{n}  GR {c['I']:.4f} C_reg {creg(m,arm,n):.4f} x{c['I']/creg(m,arm,n):.3f} band {lo:.4f}-{hi:.4f} {'in' if ok else 'OUT'} | E240 {e['I']:.4f} hb {e['hbfrac']:.2f} H {e['H']:.4f} pred(H 0.3545) {'hb' if rows[-1]['pred_hb_session'] else 'gpu'}")
plateau = [r["E240"] for r in rows if r["E240_hb"] >= 0.8]
in_plat = [x for x in plateau if 0.23 <= x <= 0.27]
p5_1b = p5["granite-3.0-1b-a400m"]
p8_fail = outside > 1 or p5_1b["agree"] / p5_1b["of"] < 0.95
sess_wrong = [(r["arm"], r["n"]) for r in rows if not (0.2 <= r["E240_hb"] <= 0.8) and (r["E240_hb"] >= 0.8) != r["pred_hb_session"]]
score["P"]["P8"] = dict(verdict="FALSIFIED" if p8_fail else "HELD", cells=rows, GR_outside=outside,
                        P5_rule=[p5_1b["agree"], p5_1b["of"]],
                        plateau_hostbound_E240=dict(n=len(plateau), inside_0p23_0p27=len(in_plat),
                                                    min=round(min(plateau), 4) if plateau else None,
                                                    max=round(max(plateau), 4) if plateau else None,
                                                    median=round(st.median(plateau), 4) if plateau else None),
                        session_H_list_wrong=sess_wrong)
p(f"  GR outside: {outside} of 27 (fail if > 1); P5 rule on this model {p5_1b['agree']}/{p5_1b['of']}; plateau (host-bound E240 cells) {len(in_plat)}/{len(plateau)} in [0.23, 0.27], range {min(plateau):.4f}-{max(plateau):.4f}; session-H list wrong on {sess_wrong}")
p("  P8:", score["P"]["P8"]["verdict"])
R.append("P8: the fails-if names only P1's rule (GR cells) and P5's rule; the plateau band (Granite-fitted, seen) and the session-H host-bound list are reported as records")

# ---------------------------------------------------------------- control: E240 against the published Granite page
p("\n== control: E240 against the published Granite-3B 09-30 G = 4 cells (record)")
import glob, re
pub = defaultdict(list)
for f in glob.glob(str(REPO / "results/published/2026-09-30-nvidia_gh200_480gb-granite-session/**/private_weight_reference/*/cells.csv"), recursive=True):
    mm = re.search(r"-g(\d+)-", f.split("/")[-2])
    if not mm or mm.group(1) != "4":
        continue
    for r in csv.DictReader(open(f)):
        pub[(r["arm"], int(r["tiles"]))].append(float(r["ms_p50"]))
ctrl = {}
if pub:
    for arm in ARMS:
        for n in range(1, 10):
            if (arm, n) in pub:
                a, b = C["granite-3.0-3b-a800m"][("E240", arm, n)]["I"], st.median(pub[(arm, n)])
                ctrl[f"{arm}-{n}"] = dict(E240=round(a, 4), published=round(b, 4), rel=round(a / b - 1, 4), pages=len(pub[(arm, n)]))
    rels = [v["rel"] for v in ctrl.values()]
    p(f"  {len(ctrl)} cells, E240/published - 1: median {st.median(rels)*100:+.1f}%, range {min(rels)*100:+.1f}% to {max(rels)*100:+.1f}%")
    for k, v in ctrl.items():
        p(f"    {k:10s} E240 {v['E240']:.4f} published {v['published']:.4f} {v['rel']*100:+.1f}% (k={v['pages']})")
else:
    p("  no published G = 4 cells found")
score["control_E240_vs_published"] = ctrl

DIAG = {}
p("\n== diagnosis (not registered): the profiler hangover and the host-paced relation")
for m in MODELS:
    rows = list(csv.DictReader(open(PAGES / m / "cells.csv")))
    first = rows[0]
    pre = [r for r in rows[:4] if r["arm"] == first["arm"] and r["tiles"] == first["tiles"] and r["repeat"] == "0"]
    for r in pre:
        Hc = float(r["host_enqueue_ms"]) / float(r["calls_per_burst"])
        p(f"  {m[:12]:12s} BEFORE the first trace burst: {r['mode']:4s} {r['arm']} n{r['tiles']}  I {float(r['ms_p50']):.4f}  H {Hc:.4f}  hb {r['host_bound']}  H-I {Hc-float(r['ms_p50']):.4f}")
    post = defaultdict(list)
    for r in rows[len(pre):]:
        if r["host_bound"] == "True" and r["mode"] != "GR":
            Hc = float(r["host_enqueue_ms"]) / float(r["calls_per_burst"])
            post[r["mode"]].append((Hc, Hc - float(r["ms_p50"])))
    for mode, v in sorted(post.items()):
        p(f"  {m[:12]:12s} after: {mode:4s} host-bound rows {len(v)}  median H {st.median(x[0] for x in v):.4f}  median H-I (Phi) {st.median(x[1] for x in v):.4f}  IQR {st.quantiles([x[1] for x in v], n=4)[0]:.4f}-{st.quantiles([x[1] for x in v], n=4)[2]:.4f}")
        DIAG[f"{m}:{mode}"] = dict(rows=len(v), H=st.median(x[0] for x in v), Phi=st.median(x[1] for x in v))
    DIAG[f"{m}:pre"] = [dict(mode=r["mode"], I=float(r["ms_p50"]), H=float(r["host_enqueue_ms"]) / float(r["calls_per_burst"]), hb=r["host_bound"]) for r in pre]
# P4 shift with each mode's own H: under I = H - Phi_mode the shift is (H480 - H240) - (F480 - F240)
m = "granite-3.0-3b-a800m"
p("  P4 shift corrected for the H difference between modes, (E480 - E240) - (H480 - H240) against -(F480 - F240):")
for r in score["P"]["P4"]["cells"]:
    if "shift_us" in r:
        corr = r["shift_us"] - (r["H480"] - r["H240"]) * 1e3
        p(f"    {r['arm']:7s} n{r['n']}  shift {r['shift_us']:+.1f}  H480-H240 {(r['H480']-r['H240'])*1e3:+.1f}  corrected {corr:+.1f}  miss {corr + dF_tr*1e3:+.1f} us")
p("  P3 the same way, (E0 - E240) - (H0 - H240):")
for r in score["P"]["P3"]["cells"]:
    a, n = r["arm"], r["n"]
    dH = C[m][("E0", a, n)]["H"] - C[m][("E240", a, n)]["H"]
    p(f"    {a:7s} n{n}  E0-E240 {r['measured_ms']*1e3:+.1f}  H0-H240 {dH*1e3:+.1f}  residual {(r['measured_ms']-dH)*1e3:+.1f} us (F_trace {r['F_us']:.1f})")
# P6 sensitivity: lambda = 0 and the n = 1, 2 E480 cells
p("  P6 sensitivity on the two E480 misses (lambda 0 / lambda 2 us):")
for (mm, arm, n, mode) in score["P"]["P6"]["outside"]:
    tdir = PAGES / mm / "traces"
    te, tg = timeline(tdir / f"TR-E-{mode}-{arm}-n{n}.json.gz"), timeline(tdir / f"TR-G-{mode}-{arm}-n{n}.json.gz")
    c = C[mm][(mode, arm, n)]
    for lam in (0.0, 2.0):
        t2 = dict(te, lam=lam)
        pr, _ = mp(t2, tg, c["H"] * 1e3)
        p(f"    {mm[:12]} {mode} {arm} n{n} lambda {lam}: MP {pr:.4f} vs {c['I']:.4f} miss {(pr-c['I'])*1e3:+.1f} us (band {max(0.05*c['I'],0.012)*1e3:.1f})")
score["diagnosis"] = DIAG

p("\nreadings:")
for r in R:
    p("  -", r)
p("\nverdicts:", {k: v["verdict"] for k, v in score["P"].items()})
(OUT / "score.json").write_text(json.dumps(score, indent=1, default=str))
(OUT / "score.txt").write_text("\n".join(out_lines) + "\n")
