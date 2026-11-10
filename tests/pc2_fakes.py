"""A synthetic rental-PC2 tree for the reader's tests (tests/test_read_pc2.py): every registered unit's directory in the
probe's on-disk layout (scripts/instr_probe_pc2.py is PC's probe, so the layout is PC's), its stamp rows planted from a
WORLD with a known answer.

The world: a CTA streams its slab b at r = min(Q_B / L0, X_SAT / conc) after a fixed cost C0, so its life is C0 + b / r
(Q_B = (num_stages - 1) x BLOCK_K x BLOCK_N x 2, conc the CTAs of its wave). CTAs launch in waves of the tile's registered
wave (132 x occupancy), 2 ns apart inside a wave; a wave starts when the one before it ends; CTA i sits on SM i mod 132.
The per-CTA clock is 0.99 x the lock. With every overhead o = 0.004, the reader's F ladder reads c0 = C0 / 1.004 exactly
and every CTA of a wave has one life, so a single-wave launch's kappa is (life + 2 (n - 1)) / life."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import pc2common as P

REPO = Path(__file__).resolve().parents[1]
X_SAT = 3600.0
L0_NS = 700.0
C0_NS = 3000.0
SPREAD_NS = 2
O = 0.004
#: the card's globaltimer is absolute (about 1.79e18 ns on PC's pages); the fake adds this base in int64, after
#: rounding, so the reader's EXACT TIME rebase is exercised on every tree
BASE_NS = 1_793_988_699_408_532_736


def q_b(tile) -> int:
    return (int(tile[4]) - 1) * int(tile[2]) * int(tile[1]) * 2


def world(tile, conc):
    return min(q_b(tile) / L0_NS, X_SAT / conc)


def launch_rows(live, dead, tile, N, K, t0, lock=1710, c0=C0_NS, slow=1.0):
    """[live + dead, 10] int64 rows (vectorised)."""
    W = P.wave(tile)
    bn = int(tile[1])
    Pn = -(-int(N) // bn)
    f = 0.99 * lock
    rows = np.full((live + dead, 10), -1, np.int64)
    pid = np.arange(live)
    w_idx = pid // W
    j = pid % W
    cols = np.minimum(bn, int(N) - (pid % Pn) * bn)
    b = int(K) * cols * 2.0
    s = np.zeros(live)
    e = np.zeros(live)
    t = float(t0)
    for w in range(int(w_idx.max()) + 1 if live else 0):
        k = w_idx == w
        n = int(k.sum())
        s[k] = t + SPREAD_NS * j[k]
        e[k] = s[k] + c0 + slow * b[k] / world(tile, n)
        t = float(e[k].max()) + 50
    rows[:live, 0] = pid % P.SMS
    rows[:live, 1], rows[:live, 7] = np.round(s).astype(np.int64) + BASE_NS, np.round(e).astype(np.int64) + BASE_NS
    rows[:live, 2], rows[:live, 8] = np.round(s * f / 1e3), np.round(e * f / 1e3)
    rows[:live, 9] = 1
    d = np.arange(live, live + dead)
    rows[live:, 0] = d % P.SMS
    rows[live:, 1], rows[live:, 7] = round(t0 + 10) + BASE_NS, round(t0 + 110) + BASE_NS
    rows[live:, 2], rows[live:, 8] = round((t0 + 10) * f / 1e3), round((t0 + 110) * f / 1e3)
    rows[live:, 9] = 2
    return rows, float(e.max()) if live else float(t0)


def _sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def experts_of(cfg, counts, tile) -> np.ndarray:
    """expert_ids[:L]: aligned, each active expert ceil(c_e / BM) times; naive, one row per (token, expert) pair."""
    if P.is_naive(cfg, counts):
        return np.array([e for e, c in enumerate(counts) for _ in range(int(c))], np.int64)
    return np.array([e for e, c in enumerate(counts) if c for _ in range(-(-int(c) // int(tile[0])))], np.int64)


def write_unit(results: Path, label: str, *, page_label="CLEAN", corrupt=None, occupancy=None, versions=None,
               page_sha=None, partial=False, c0=C0_NS, slow=None):
    """One PC2 stamps unit in instr_probe_pc2.py's layout; returns ({(stem, gemm): overhead}, vid)."""
    import read_pc2 as R
    suffix, vid, _uid, src, lock, _pt, _rc = R.STAMP_UNITS[label]
    d = results / f"2026-11-10-{P.CARD}-{suffix}"
    (d / "stamps").mkdir(parents=True, exist_ok=True)
    clist = P.check_page_doc(json.loads((REPO / src).read_text()), P.load_shapes())
    launches, cells, tiles, over = [], {}, {}, {}
    t = 1_000_000.0
    for c in clist:
        stem, cfg, counts, tile = c["label"], c["cfg"], c["counts"], c["tile"]
        experts = experts_of(cfg, counts, tile)
        live = {g: P.live_ctas(cfg, counts, tile, g) for g in ("w1", "w2")}
        cells[stem] = {"label": stem, "shape": c["shape"], "kind": "decode", "tile": P.tile_key(tile),
                       "flush": c["flush"], "read": c["read"], "T": c["T"], "counts": counts, "live": live,
                       "readback": "PASS", "calls": P.stamp_calls(live["w2"])}
        tiles[P.tile_key(tile)] = tile
        for call in range(P.stamp_calls(live["w2"])):
            for g in ("w1", "w2"):
                N, K = P.gemm_dims(cfg, g)
                rows, t_end = launch_rows(live[g], 16, tile, N, K, t, lock, c0, (slow or {}).get((stem, g), 1.0))
                if corrupt and corrupt[0] == stem and corrupt[1] == g:
                    corrupt[2](rows)
                name = f"{stem}-call{call}-{g}"
                np.save(d / "stamps" / f"{name}.npy", rows)
                np.save(d / "stamps" / f"{name}-experts.npy", experts)
                span = t_end - t
                launches.append({"label": stem, "stem": stem, "n": None, "arm": "native", "call": call, "gemm": g,
                                 "file": f"{name}.npy", "experts_file": f"{name}-experts.npy",
                                 "num_tokens_post_padded": len(experts) * tile[0], "N": N, "K": K,
                                 "BLOCK_SIZE_M": tile[0], "BLOCK_SIZE_N": tile[1], "BLOCK_SIZE_K": tile[2],
                                 "num_warps": tile[3], "num_stages": tile[4], "use_td": False, "late_buffers": 0,
                                 "flush": c["flush"], "launch_ms": (span + 3000) / 1e6, "stamp_span_ns": span,
                                 "tile": P.tile_key(tile)})
                over[(stem, g)] = O
                t = t_end + 20_000
    occ = {**P.OCCUPANCY, **(occupancy or {})}
    sass = [{"config": [tl[0], tl[1], tl[2], 1, tl[4], tl[3], m, k], "compile": "h", "checks_c3": {"ok": True},
             "inflight": {"n_if": tl[4] - 1}, "ctas_per_sm": occ[key]} for key, tl in tiles.items()
            for m, k in ((False, 2), (True, 1))]
    man = {"tool": "scripts/instr_probe_pc2.py --mode stamps", "variant": vid, "kind": "page", "page_label": page_label,
           "source": src, "source_sha256": page_sha or _sha(src), "versions": versions or dict(P.VERSIONS),
           "sass": [] if partial else sass, "complete": not partial, "cells": cells, "launches": launches}
    (d / "stamps.json").write_text(json.dumps(man))
    return over, vid


def tree(root: Path, *, labels=("sd", "ss", "sn", "sw", "sd3", "sr"), overrides=None, c0=C0_NS) -> Path:
    """A published PC2 session directory: results/ and session/ for the registered units named."""
    import read_pc2 as R
    ov = overrides or {}
    pc2 = root / "pc2"
    results, session = pc2 / "results", pc2 / "session"
    D = session / "gh200-driver"
    D.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (D / "c2-power.env").write_text(ov.get("power", "C2_POWER_W=700\n"))
    (D / "locks.env").write_text(ov.get("locks", "LOCK_TIMED_OK=1\nLOCK_PROBE_1710=held\nLOCK_PROBE_1410=held\n"))
    (D / "gate-band-1410.json").write_text(json.dumps({"verdict": "PASS", "band_in_force": {"which": "fallback"}}))
    pcl, perturb = [], {}
    for label in labels:
        unit_kw = (ov.get("unit") or {}).get(label, {})
        over, vid = write_unit(results, label, corrupt=ov.get("corrupt") if ov.get("corrupt_unit") == label else None,
                               c0=c0, **unit_kw)
        perturb[vid] = {"cells": [{"stem": s, "label": s, "gemm": g, "tile": "x", "overhead_median": o,
                                   "overhead_worst": 2 * o, "plain_ms": ov.get("plain_ms", 0.1)}
                                  for (s, g), o in over.items()]}
        if ov.get("nan_overhead") and label == ov["nan_overhead"][0]:
            for c in perturb[vid]["cells"]:
                if c["stem"] == ov["nan_overhead"][1]:
                    c["overhead_median"] = float("nan")
        uid = R.STAMP_UNITS[label][2]
        after = (ov.get("after") or {}).get(label, 1710)
        pcl.append(f"PC_LOCK_{uid.replace('-', '_').replace('.', '_')}=1710 before=1710 after={after} exit=0")
    pcl.append(f"PC_LOCK_mixtral_8x7b_pt=1710 before=1710 after={(ov.get('after') or {}).get('pt', 1710)} exit=0")
    (D / "pc-locks.env").write_text("\n".join(pcl) + "\n")
    d = results / f"2026-11-10-{P.CARD}-perturb-pt"
    d.mkdir()
    (d / "perturb.json").write_text(json.dumps({"variants": perturb}))
    (d / "gate.env").write_text("".join(f"GATE_{k.replace('-', '_')}={ov.get('gate', 'PASS')} median=0.004 "
                                        f"worst=0.008\nHARD_{k.replace('-', '_')}={ov.get('hard', 'PASS')}\n"
                                        for k in perturb))
    d = results / f"2026-11-10-{P.CARD}-regcheck-rcp"
    d.mkdir()
    (d / "regcheck.env").write_text("".join(
        f"REGCHECK_{P.VARIANTS[lab].replace('-', '_')}={(ov.get('regcheck') or {}).get(lab, 'PASS')}\n" for lab in labels))
    d = results / f"2026-11-10-{P.CARD}-tile-census-mixtral-8x7b-pc2"
    d.mkdir()
    occ = {**P.OCCUPANCY, **(ov.get("census") or {})}
    (d / "tile-census.json").write_text(json.dumps({"verdict": "COMPLETE", "path_mismatch": [], "summary": {
        t: {"by": {f"{m}|{g}": o for m in P.CENSUS_MODELS for g in ("w1", "w2")}} for t, o in occ.items()}}))
    return pc2


def isclose(a, b, rel=1e-6):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)
