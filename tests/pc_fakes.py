"""A synthetic rental-PC tree for the reader's tests (tests/test_read_pc.py): every registered unit's directory in the
probes' on-disk layout, its stamp rows planted from a WORLD (a per-CTA rate law), so every reading has a known answer.

A world gives a CTA's streaming rate r (GB/s) from its tile, warps, the lock and the count of CTAs it shares the GPU
with. CTAs launch in waves of W = 132 x the tile's occupancy (2 ns apart inside a wave, so the globaltimer resolves
them); a wave starts when the one before it ends; CTA i sits on SM i mod 132. The per-CTA clock is 0.99 x the lock."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import pccommon as P

REPO = Path(__file__).resolve().parents[1]
X_SAT = 3600.0
L0_NS = 700.0
SPREAD_NS = 2


def world_supported(tile, lock, conc):
    """H1: r = Q_B / L0 until the GPU saturates, then X_sat shared; no clock, warps or SM-sharing effect."""
    q = (int(tile[4]) - 1) * int(tile[2]) * int(tile[1]) * 2
    return min(q / L0_NS, X_SAT / conc)


def world_clock(tile, lock, conc):
    """Compute-bound at low load: the rate scales with the SM clock (beta_f 1)."""
    return world_supported(tile, lock, conc) * (lock / 1710.0)


def world_warps(tile, lock, conc):
    """8 warps stream 1.2x faster than 4 at the same loads in flight (omega 1.2: outside SUPPORTED, inside FALSIFIED)."""
    return world_supported(tile, lock, conc) * (1.2 if int(tile[3]) == 8 else 1.0)


def launch_rows(live, dead, tile, N, K, world, lock, t0):
    """[live + dead, 10] int64 rows: live CTAs in waves, dead CTAs a 100 ns exit after the last live start."""
    occ = P.DESIGN_OCCUPANCY[P.tile_key(tile)]
    W = P.SMS * occ
    bn = int(tile[1])
    Pn = -(-int(N) // bn)
    f = 0.99 * lock
    rows = np.full((live + dead, 10), -1, np.int64)
    t = float(t0)
    i = 0
    while i < live:
        n = min(W, live - i)
        for j in range(n):
            pid = i + j
            cols = min(bn, int(N) - (pid % Pn) * bn)
            b = int(K) * cols * 2
            s = t + SPREAD_NS * j
            life = b / world(tile, lock, n)
            e = s + life
            rows[pid] = [pid % P.SMS, round(s), round(s * f / 1e3), -1, -1, -1, -1, round(e), round(e * f / 1e3), 1]
        t = float(rows[i:i + n, 7].max()) + 50
        i += n
    for d in range(live, live + dead):
        rows[d] = [d % P.SMS, round(t0 + 10), round((t0 + 10) * f / 1e3), -1, -1, -1, -1, round(t0 + 110),
                   round((t0 + 110) * f / 1e3), 2]
    return rows, float(rows[:live, 7].max())


def _sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def write_unit(results: Path, label: str, world, *, page_label="CLEAN", corrupt=None):
    """One PC stamps unit (page or Phi E2) in instr_probe_pc.py's layout; returns {(stem, gemm): overhead}."""
    import read_pc as R
    suffix, vid, _uid, src, lock, _pt, _rc = R.STAMP_UNITS[label]
    d = results / f"2026-11-06-{P.CARD}-{suffix}"
    (d / "stamps").mkdir(parents=True, exist_ok=True)
    launches, cells, sass, over = [], {}, {}, {}
    if label in R.PAGE_OF:
        doc = P.page_doc(R.PAGE_OF[label])
        clist = P.check_page_doc(doc, P.load_shapes())
        items = [(c["label"], c, c["cfg"], c["counts"], c["tile"], c["flush"]) for c in clist]
    else:
        from moe.spec import MODEL_CONFIGS
        cfg = MODEL_CONFIGS["phi-3.5-moe"]
        h = json.loads((REPO / src).read_text())
        items = [(f"{c['label']}-n{c['n']}-native", c, cfg, c.get("counts") or [c["n"] * h["block_m"]] * 16, P.P64, True)
                 for c in h["cells"]]
    t = 1_000_000.0
    for stem, c, cfg, counts, tile, flush in items:
        tiles_m = [(e, -(-int(x) // tile[0])) for e, x in enumerate(counts) if int(x)]
        experts = np.array([e for e, m in tiles_m for _ in range(m)], np.int64)
        live = {g: P.live_ctas(cfg, counts, tile, g) for g in ("w1", "w2")}
        cells[stem] = {"counts": counts, "tile": P.tile_key(tile), "flush": flush, "shape": cfg.name}
        if label in R.PAGE_OF:   # the probe records the expected live CTAs of a decode cell only
            cells[stem]["live"] = live
        sass[P.tile_key(tile)] = tile
        calls = P.stamp_calls(live["w2"]) if label in R.PAGE_OF else P.STAMP_CALLS
        for call in range(calls):
            for g in ("w1", "w2"):
                N, K = P.gemm_dims(cfg, g)
                rows, t_end = launch_rows(live[g], 32, tile, N, K, world, lock, t)
                if corrupt and corrupt[0] == stem and corrupt[1] == g:
                    corrupt[2](rows)
                name = f"{stem}-call{call}-{g}"
                np.save(d / "stamps" / f"{name}.npy", rows)
                np.save(d / "stamps" / f"{name}-experts.npy", experts)
                span = t_end - t
                launches.append({"label": c["label"], "stem": stem, "n": c.get("n"), "arm": "native", "call": call,
                                 "gemm": g, "file": f"{name}.npy", "experts_file": f"{name}-experts.npy",
                                 "num_tokens_post_padded": len(experts) * tile[0], "N": N, "K": K,
                                 "BLOCK_SIZE_M": tile[0], "BLOCK_SIZE_N": tile[1], "BLOCK_SIZE_K": tile[2],
                                 "num_warps": tile[3], "num_stages": tile[4], "use_td": False, "late_buffers": 0,
                                 "flush": flush, "launch_ms": (span + 3000) / 1e6, "stamp_span_ns": span})
                over[(stem, g)] = 0.004
                t = t_end + 20_000
    man = {"variant": vid, "kind": "page" if label in R.PAGE_OF else "hist", "page_label": page_label,
           "source_sha256": _sha(src), "versions": dict(P.VERSIONS),
           "sass": [{"config": [tl[0], tl[1], tl[2], 1, tl[4], tl[3], m, k], "checks_c3": {"ok": True},
                     "inflight": {"n_if": tl[4] - 1}} for tl in sass.values() for m, k in ((False, 2), (True, 1))],
           "cells": cells, "launches": launches}
    (d / "stamps.json").write_text(json.dumps(man))
    return vid, over


def write_e1(results: Path, world):
    """Phi E1 in instr_probe_c3.py's layout (128x128x64 w8 s3)."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["phi-3.5-moe"]
    d = results / f"2026-11-06-{P.CARD}-instr-phi-3.5-moe-se1"
    (d / "stamps").mkdir(parents=True, exist_ok=True)
    h = json.loads((REPO / P.PHI_128).read_text())
    launches, over, t = [], {}, 1_000_000.0
    for c in h["cells"]:
        counts = c.get("counts") or [c["n"] * 128] * 16
        tiles_m = [(e, -(-int(x) // 128)) for e, x in enumerate(counts) if int(x)]
        experts = np.array([e for e, m in tiles_m for _ in range(m)], np.int64)
        for call in range(3):
            for g in ("w1", "w2"):
                N, K = P.gemm_dims(cfg, g)
                live = P.live_ctas(cfg, counts, P.P128, g)
                rows, t_end = launch_rows(live, 0, P.P128, N, K, world, 1410, t)
                name = f"{c['label']}-n{c['n']}-native-call{call}-{g}"
                np.save(d / "stamps" / f"{name}.npy", rows)
                np.save(d / "stamps" / f"{name}-experts.npy", experts)
                launches.append({"label": c["label"], "n": c["n"], "arm": "native", "call": call, "gemm": g,
                                 "file": f"{name}.npy", "experts_file": f"{name}-experts.npy",
                                 "num_tokens_post_padded": len(experts) * 128, "N": N, "K": K, "BLOCK_SIZE_M": 128,
                                 "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 3,
                                 "use_td": False, "late_buffers": 0, "launch_ms": (t_end - t + 3000) / 1e6})
                over[(c["label"], c["n"], g)] = 0.004
                t = t_end + 20_000
    man = {"variant": P.VARIANTS["se1"], "page_label": "CLEAN", "histogram_sha256": _sha(P.PHI_128),
           "versions": dict(P.VERSIONS), "sass": [{"config": [128, 128, 64, 1, 3, 8, False, 2], "checks_c3": {"ok": True}}],
           "launches": launches}
    (d / "stamps.json").write_text(json.dumps(man))
    return over


def tree(root: Path, world=world_supported, *, overrides=None) -> Path:
    """A published PC session directory: results/ and session/ for every registered unit."""
    import read_pc as R
    ov = overrides or {}
    pc = root / "pc"
    results, session = pc / "results", pc / "session"
    D = session / "gh200-driver"
    D.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (D / "c2-power.env").write_text(ov.get("power", "C2_POWER_W=700\n"))
    (D / "locks.env").write_text(ov.get("locks", "LOCK_TIMED_OK=1\nLOCK_PROBE_1710=held\nLOCK_PROBE_1410=held\n"))
    (D / "gate-band-1410.json").write_text(json.dumps({"verdict": ov.get("gband", "PASS"),
                                                        "band_in_force": {"which": "fallback"}}))
    pcl = []
    perturb = {"ptp": {}, "ptc": {}}
    for label in ("sa", "sc", "sd", "sr", "se2"):
        vid, over = write_unit(results, label, ov.get("world", {}).get(label, world),
                               corrupt=ov.get("corrupt") if ov.get("corrupt", [None])[0] in _stems(label) else None)
        pt = R.STAMP_UNITS[label][5]
        perturb[pt][vid] = {"cells": [{"stem": s, "gemm": g, "overhead_median": o, "overhead_worst": 2 * o,
                                       "plain_ms": 0.1} for (s, g), o in over.items()]}
        uid, lock = R.STAMP_UNITS[label][2], R.STAMP_UNITS[label][4]
        after = ov.get("after", {}).get(label, lock)
        pcl.append(f"PC_LOCK_{uid.replace('-', '_').replace('.', '_')}={lock} before={lock} after={after} exit=0")
    pcl += [f"PC_LOCK_mixtral_8x7b_ptp=1710 before=1710 after={ov.get('after', {}).get('ptp', 1710)} exit=0",
            "PC_LOCK_mixtral_8x7b_ptc=1410 before=1410 after=1410 exit=0"]
    (D / "pc-locks.env").write_text("\n".join(pcl) + "\n")
    e1 = write_e1(results, world)
    for pt, vs in perturb.items():
        d = results / f"2026-11-06-{P.CARD}-perturb-{pt}"
        d.mkdir(parents=True)
        (d / "perturb.json").write_text(json.dumps({"variants": vs}))
        (d / "gate.env").write_text("".join(f"GATE_{k.replace('-', '_')}={ov.get('gate', 'PASS')} median=0.004 "
                                            f"worst=0.008\nHARD_{k.replace('-', '_')}=PASS\n" for k in vs))
    d = results / f"2026-11-06-{P.CARD}-perturb-pt3"
    d.mkdir()
    vid = P.VARIANTS["se1"]
    (d / "perturb.json").write_text(json.dumps({"variants": {vid: {"cells": [
        {"label": lab, "n": n, "arm": "native", "gemm": g, "overhead_median": o, "overhead_worst": 2 * o, "plain_ms": 0.1}
        for (lab, n, g), o in e1.items()]}}}))
    (d / "gate.env").write_text(f"GATE_{vid.replace('-', '_')}=PASS median=0.004 worst=0.008\nHARD_{vid.replace('-', '_')}=PASS\n")
    for rc, labs in (("rcp", ("sa", "sc", "sd", "sr", "se2")), ("rc3", ("se1",))):
        d = results / f"2026-11-06-{P.CARD}-regcheck-{rc}"
        d.mkdir()
        (d / "regcheck.env").write_text("".join(
            f"REGCHECK_{P.VARIANTS[lab].replace('-', '_')}={ov.get('regcheck', {}).get(lab, 'PASS')}\n" for lab in labs))
    d = results / f"2026-11-06-{P.CARD}-tile-census-mixtral-8x7b-pc"
    d.mkdir()
    occ = {**P.DESIGN_OCCUPANCY, **ov.get("census", {})}
    (d / "tile-census.json").write_text(json.dumps({"verdict": "COMPLETE", "path_mismatch": [], "summary": {
        t: {"by": {f"{m}|{g}": o for m in P.CENSUS_MODELS for g in ("w1", "w2")}} for t, o in occ.items()}}))
    write_counter(results)
    return pc


def _stems(label):
    import read_pc as R
    if label in R.PAGE_OF:
        return [c["label"] for c in P.page_doc(R.PAGE_OF[label])["cells"]]
    return []


def write_counter(results: Path):
    """Unit 16's counter page: per cell and GEMM, an ncu time and the construction's bytes."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    d = results / f"2026-11-06-{P.CARD}-mixtral-8x7b-cnt-r3-counters" / "lock1710"
    d.mkdir(parents=True)
    cells = []
    for lab, T, A, ce in P.CNT_CELLS:
        pg = {}
        for g in ("w1", "w2"):
            N, K = P.gemm_dims(cfg, g)
            pg[g] = {"gpu__time_duration.sum": 100_000.0, "dram__bytes_read.sum": float(A * N * K * 2)}
        cells.append({"label": lab, "T": T, "arm": "native", "per_gemm": pg})
    (d / "r3d.json").write_text(json.dumps({"kernels": "wide", "gates": {"V_ALIGN_grid": "PASS"},
                                            "manifest": {"page_sha256": _sha(f"{P.PAGES_DIR}/{P.CNT_PAGE}.json"),
                                                         "versions": dict(P.VERSIONS)}, "cells": cells}))


def isclose(a, b, rel=1e-6):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)
