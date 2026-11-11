"""A synthetic rental-RR tree for the reader's tests (tests/test_read_rr.py): every registered unit's directory in the
probe's and the counter route's on-disk layout, the stamps planted from a WORLD with a known answer.

The world: on each launch the live CTAs are list-scheduled in pid order onto W_c = 132 x the registered occupancy slots
(a CTA starts when the earliest slot frees, 2 ns later), each living L0 + slab / 30 B per ns, varied deterministically
by up to 15% by pid; CTA i sits on SM i mod 132; the per-CTA clock is 0.99 x 1410 MHz. Every call of a cell is the same
schedule. The counted w1 bytes are planted as D = C_u + m x R with m = the primary form's m_hat on the planted stamps
(read_rr.pair_read, whose integral tests/test_read_rr.py checks against brute force), so the primary form's residual is
zero on every cell and the other forms' are not."""
from __future__ import annotations

import hashlib
import math
import heapq
import json
from pathlib import Path

import numpy as np

import rrcommon as P

REPO = Path(__file__).resolve().parents[1]
O = 0.004
BASE_NS = 1_793_988_699_408_532_736
LOCK = P.SESSION_LOCK
DATE = "2026-11-12"


def life_ns(pid, slab: int) -> np.ndarray:
    pid = np.asarray(pid)
    return (2000.0 + slab / 30.0) * (1.0 + 0.15 * ((pid * 0.6180339887) % 1.0))


def schedule(n: int, W: int, slab: int, t0: float):
    """List scheduling of n CTAs (pid order) onto W slots: (start, end) in ns, both float."""
    life = life_ns(np.arange(n), slab)
    free = [(float(t0), j) for j in range(W)]
    heapq.heapify(free)
    s = np.empty(n)
    for i in range(n):
        t, j = heapq.heappop(free)
        s[i] = t + 2.0
        heapq.heappush(free, (s[i] + life[i], j))
    return s, s + life


def rows_of(s, e, dead: int = 16) -> np.ndarray:
    n = len(s)
    f = 0.99 * LOCK
    rows = np.full((n + dead, 10), -1, np.int64)
    rows[:n, 0] = np.arange(n) % P.SMS
    rows[:n, 1] = np.round(s).astype(np.int64) + BASE_NS
    rows[:n, 7] = np.round(e).astype(np.int64) + BASE_NS
    rows[:n, 2] = np.round(s * f / 1e3)
    rows[:n, 8] = np.round(e * f / 1e3)
    rows[:n, 9] = 1
    t0 = float(s.min()) if n else 0.0
    rows[n:, 0] = np.arange(n, n + dead) % P.SMS
    rows[n:, 1], rows[n:, 7] = round(t0 + 10) + BASE_NS, round(t0 + 110) + BASE_NS
    rows[n:, 2], rows[n:, 8] = round((t0 + 10) * f / 1e3), round((t0 + 110) * f / 1e3)
    rows[n:, 9] = 2
    return rows


def experts_of(counts, bm: int) -> np.ndarray:
    return np.array([e for e, c in enumerate(counts) if c for _ in range(-(-int(c) // int(bm)))], np.int64)


def _sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def stamp_launches(cfg, counts, tile, t0: float):
    """[(gemm, rows, experts, N, K, ntpp, end)] of one call: w1 then w2 after it."""
    out, t = [], t0
    ex = experts_of(counts, tile[0])
    for g in ("w1", "w2"):
        N, K = P.gemm_dims(cfg, g)
        live = P.live_ctas(cfg, counts, tile, g)
        s, e = schedule(live, P.wave(tile), int(K) * int(tile[1]) * 2, t)
        out.append((g, rows_of(s, e), ex, N, K, len(ex) * tile[0], float(e.max())))
        t = float(math.ceil(e.max())) + 5000
    return out


def write_stamps(results: Path, label: str, *, cells=None, corrupt=None, page_label="CLEAN", versions=None,
                 page_sha=None, partial=False, occupancy=None):
    """One RR stamps unit in instr_probe_rr.py's layout; returns ({(stem, gemm): o}, vid, {stem: (cfg, counts, tile,
    w1 rows, experts, N, K, ntpp)})."""
    import read_rr as R
    suffix, vid, _uid, src, _pt, _rc, _cu = R.STAMP_UNITS[label]
    d = results / f"{DATE}-{P.CARD}-{suffix}"
    (d / "stamps").mkdir(parents=True, exist_ok=True)
    if label == "sr":
        from moe.spec import MODEL_CONFIGS
        cfg = MODEL_CONFIGS["phi-3.5-moe"]
        hist = json.loads((REPO / src).read_text())
        hc = R.hist_counts(src)
        clist = [{"label": f"{c['label']}-n{c['n']}-native", "cfg": cfg, "counts": hc[(c["label"], int(c["n"]))],
                  "tile": P.T128, "hist": (c["label"], int(c["n"]))} for c in hist["cells"]]
    else:
        clist = P.check_page_doc(json.loads((REPO / src).read_text()), P.load_shapes())
    if cells is not None:
        clist = [c for c in clist if c["label"] in cells]
    launches, cells_rec, tiles, over, w1 = [], {}, {}, {}, {}
    t = 1_000_000.0
    for c in clist:
        stem, cfg, counts, tile = c["label"], c["cfg"], c["counts"], c["tile"]
        live = {g: P.live_ctas(cfg, counts, tile, g) for g in ("w1", "w2")}
        if label == "sr":
            cells_rec[stem] = {"label": c["hist"][0], "n": c["hist"][1], "shape": "phi-3.5-moe", "kind": "hist",
                               "tile": P.tile_key(tile), "flush": True, "read": "w1", "readback": "PASS", "calls": 3}
        else:
            cells_rec[stem] = {"label": stem, "shape": c["shape"], "kind": "decode", "tile": P.tile_key(tile),
                               "flush": True, "read": "w1", "T": c["T"], "counts": counts, "live": live,
                               "readback": "PASS", "calls": 3}
        tiles[P.tile_key(tile)] = tile
        for call in range(3):
            for g, rows, ex, N, K, ntpp, end in stamp_launches(cfg, counts, tile, t):
                if corrupt and corrupt[0] == stem and corrupt[1] == g:
                    corrupt[2](rows)
                name = f"{stem}-call{call}-{g}"
                np.save(d / "stamps" / f"{name}.npy", rows)
                np.save(d / "stamps" / f"{name}-experts.npy", ex)
                if g == "w1" and call == 0:
                    w1[stem] = (cfg, counts, tile, rows, ex, N, K, ntpp)
                launches.append({"label": stem if label != "sr" else c["hist"][0], "stem": stem,
                                 "n": c["hist"][1] if label == "sr" else None, "arm": "native", "call": call,
                                 "gemm": g, "file": f"{name}.npy", "experts_file": f"{name}-experts.npy",
                                 "num_tokens_post_padded": ntpp, "N": N, "K": K, "BLOCK_SIZE_M": tile[0],
                                 "BLOCK_SIZE_N": tile[1], "BLOCK_SIZE_K": tile[2], "num_warps": tile[3],
                                 "num_stages": tile[4], "use_td": False, "late_buffers": 0, "flush": True,
                                 "launch_ms": (end - t + 3000) / 1e6, "tile": P.tile_key(tile)})
                over[(stem, g)] = O
                t = float(math.ceil(end)) + 20_000   # integer call starts: every call the same schedule, shifted
    occ = {**P.OCCUPANCY, **(occupancy or {})}
    sass = [{"config": [tl[0], tl[1], tl[2], 1, tl[4], tl[3], m, k], "compile": "h", "checks_c3": {"ok": True},
             "inflight": {"n_if": tl[4] - 1}, "ctas_per_sm": occ[key]} for key, tl in tiles.items()
            for m, k in ((False, 2), (True, 1))]
    man = {"tool": "scripts/instr_probe_rr.py --mode stamps", "variant": vid, "kind": "hist" if label == "sr" else "page",
           "page_label": page_label, "source": src, "source_sha256": page_sha or _sha(src),
           "versions": versions or dict(P.VERSIONS), "sass": [] if partial else sass, "complete": not partial,
           "cells": cells_rec, "launches": launches}
    (d / "stamps.json").write_text(json.dumps(man))
    return over, vid, w1


def planted_m(w1: dict) -> dict:
    """{stem: the primary form's m_hat on the planted call-0 stamps} (the world's miss fraction)."""
    import read_rr as R
    out = {}
    for stem, (_cfg, _counts, tile, rows, ex, N, K, ntpp) in w1.items():
        pr = R.pair_read(rows, ex, ntpp, N, K, tile)
        out[stem] = pr["m_hat"][P.PRIMARY_FORM] if pr.get("status") == "READ" else None
    return out


def write_counter(results: Path, label: str, m: dict, *, cells=None, ov=None):
    """One RR counter unit (dram_counter_route_rr.py's page) with D = C_u + m x R per cell."""
    import read_rr as R
    ov = ov or {}
    suffix, _uid, src = R.COUNTER_UNITS[label]
    d = results / f"{DATE}-{P.CARD}-{suffix}" / f"lock{LOCK}"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in P.check_page_doc(json.loads((REPO / src).read_text()), P.load_shapes()):
        if cells is not None and c["label"] not in cells:
            continue
        g = P.geometry(c["cfg"], c["counts"], c["tile"])
        mm = m.get(c["label"])
        D = g["C_u_bytes"] + (mm if mm is not None else 0.5) * g["R_bytes"]
        pc = list((ov.get("per_call") or {}).get(c["label"], [D, D, D]))
        N2, K2 = P.gemm_dims(c["cfg"], "w2")
        D2 = (g["A"] * N2 * K2 * 2 + sum(c["counts"]) * K2 * 2) * 1.001
        rows.append({"label": c["label"], "key": f"rr/{c['label']}", "shape": c["shape"], "T": c["T"],
                     "counts": c["counts"], "tile": P.tile_key(c["tile"]), "series": c["series"], "P": c["P"],
                     "layout": c["layout"], "grid": {"w1": 1, "w2": 1},
                     "per_gemm": {"w1": {"dram__bytes_read.sum": D, "lts__t_sector_op_read_hit_rate.pct": 40.0,
                                         "calls": 3},
                                  "w2": {"dram__bytes_read.sum": D2, "calls": 3}},
                     "per_call": {"w1": {"dram__bytes_read.sum": pc}, "w2": {"dram__bytes_read.sum": [D2] * 3}},
                     "clock_mhz": {"w1": [1330.0] * 3, "w2": [1330.0] * 3}})
    page = {"schema": ov.get("schema", P.COUNTER_SCHEMA), "family": "rr-counter", "registration": P.NAME,
            "design": {"page": ov.get("page", src), "page_sha256": ov.get("page_sha", _sha(src)),
                       "lock_mhz": ov.get("lock_mhz", LOCK)},
            "versions": ov.get("versions", dict(P.VERSIONS)), "cells": rows, "seconds": 400.0,
            "gates": {"V_ALIGN_grid": ov.get("grid", "PASS (planted)"), "complete": ov.get("complete", True)}}
    if not ov.get("missing"):
        (d / "r3d.json").write_text(json.dumps(page))


def write_rep_bytes(results: Path, session: Path, m: dict, ov=None):
    """The replicate's counted page in the pinned counter route's layout (r3c-g1.json: design, stack, ncu, per-call w1
    bytes in per_gemm_values) and the driver's status line, D = C_u + m x R per cell."""
    from moe.spec import MODEL_CONFIGS
    import read_rr as R
    ov = ov or {}
    cfg = MODEL_CONFIGS["phi-3.5-moe"]
    d = results / f"{DATE}-{P.CARD}-{R.REP_BYTES}" / f"lock{LOCK}"
    d.mkdir(parents=True, exist_ok=True)
    hist = json.loads((REPO / P.PHI_128).read_text())
    hc = R.hist_counts(P.PHI_128)
    cells = []
    for c in hist["cells"]:
        counts = hc[(c["label"], int(c["n"]))]
        g = P.geometry(cfg, counts, P.T128)
        mm = m.get(f"{c['label']}-n{c['n']}-native")
        D = g["C_u_bytes"] + (mm or 0.0) * g["R_bytes"]
        vals = list((ov.get("per_call") or {}).get(c["label"] + f"-n{c['n']}", [D, D, D]))
        cells.append({"arm": "native", "histogram": None if c["label"] == "balanced" else c["label"], "n": c["n"],
                      **({} if c["label"] == "balanced" else {"bincount": counts}),
                      "per_gemm": {"w1": {"dram_bytes_read": D, "l2_read_hit_pct": 30.0, "gpu_time_ns": 1e6},
                                   "w2": {"dram_bytes_read": 1.0}},
                      "per_gemm_values": {"w1": vals, "w2": [1.0, 1.0, 1.0]},
                      "recorded": {"w1": {"sm__cycles_elapsed.avg": 1.33e6}}})
    page = {"design": {"model": "phi-3.5-moe", "histogram": {"file_sha256": ov.get("sha", _sha(P.PHI_128))},
                       "block_m": 128, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3,
                       "arms": ["native"], "group_m": 1},
            "stack": ov.get("versions", dict(P.VERSIONS)), "ncu": {"lock_mhz": ov.get("lock_mhz", LOCK)}, "cells": cells}
    if not ov.get("missing"):
        (d / "r3c-g1.json").write_text(json.dumps(page))
    (session / "logs").mkdir(parents=True, exist_ok=True)
    (session / "logs" / f"lock{LOCK}-pages-phi-3.5-moe-br.status").write_text(ov.get("status", "G=1 exit 0 12:00:00 300 s\n"))


def tree(root: Path, *, labels=("s128", "s64", "sr"), cells=None, overrides=None) -> Path:
    """A published RR session directory: results/ and session/ for the registered units named. `cells` restricts the
    page cells planted (a fast tree); `overrides` corrupts it."""
    import read_rr as R
    ov = overrides or {}
    rr = root / "rr"
    results, session = rr / "results", rr / "session"
    D = session / "gh200-driver"
    D.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (D / "c2-power.env").write_text(ov.get("power", "C2_POWER_W=700\n"))
    (D / "locks.env").write_text(ov.get("locks", "LOCK_TIMED_OK=1\nLOCK_PROBE_1410=held\n"))
    pcl, perturb = [], {}
    for label in labels:
        unit_kw = (ov.get("unit") or {}).get(label, {})
        over, vid, w1 = write_stamps(results, label, cells=cells if label != "sr" else None,
                                     corrupt=ov.get("corrupt") if ov.get("corrupt_unit") == label else None, **unit_kw)
        m = planted_m(w1)
        perturb[vid] = {"cells": [{"stem": s, "label": s, "gemm": g, "tile": "x", "overhead_median": o,
                                   "overhead_worst": 2 * o, "plain_ms": 1.0} for (s, g), o in over.items()]}
        for c in perturb[vid]["cells"]:
            if (ov.get("overhead") or {}).get((label, c["stem"])) is not None:
                c["overhead_median"] = ov["overhead"][(label, c["stem"])]
        uid = R.STAMP_UNITS[label][2]
        after = (ov.get("after") or {}).get(label, LOCK)
        pcl.append(f"PC_LOCK_{uid.replace('-', '_').replace('.', '_')}={LOCK} before={LOCK} after={after} exit=0")
        cu = R.STAMP_UNITS[label][6]
        if cu in R.COUNTER_UNITS:
            write_counter(results, cu, m, cells=cells, ov=(ov.get("counter") or {}).get(cu))
            cafter = (ov.get("after") or {}).get(cu, LOCK)
            pcl.append(f"PC_LOCK_mixtral_8x7b_{cu}={LOCK} before={LOCK} after={cafter} exit=0")
        else:
            write_rep_bytes(results, session, m, (ov.get("counter") or {}).get("br"))
    pcl.append(f"PC_LOCK_mixtral_8x7b_pt={LOCK} before={LOCK} after={(ov.get('after') or {}).get('pt', LOCK)} exit=0")
    (D / "pc-locks.env").write_text("\n".join(pcl) + "\n")
    d = results / f"{DATE}-{P.CARD}-perturb-pt"
    d.mkdir()
    (d / "perturb.json").write_text(json.dumps({"variants": perturb}))
    (d / "gate.env").write_text("".join(f"GATE_{k.replace('-', '_')}={ov.get('gate', 'PASS')} median=0.004 "
                                        f"worst=0.008\nHARD_{k.replace('-', '_')}={ov.get('hard', 'PASS')}\n"
                                        for k in perturb))
    d = results / f"{DATE}-{P.CARD}-regcheck-rcp"
    d.mkdir()
    (d / "regcheck.env").write_text("".join(
        f"REGCHECK_{P.VARIANTS[lab].replace('-', '_')}={(ov.get('regcheck') or {}).get(lab, 'PASS')}\n" for lab in labels))
    d = results / f"{DATE}-{P.CARD}-tile-census-mixtral-8x7b-rr"
    d.mkdir()
    occ = {**P.OCCUPANCY, **(ov.get("census") or {})}
    (d / "tile-census.json").write_text(json.dumps({"verdict": "COMPLETE", "path_mismatch": [], "summary": {
        t: {"by": {f"{m}|{g}": o for m in P.CENSUS_MODELS for g in ("w1", "w2")}} for t, o in occ.items()}}))
    return rr
