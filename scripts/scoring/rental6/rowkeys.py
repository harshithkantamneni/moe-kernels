#!/usr/bin/env python
"""OPEN-L2SKEW's rival, priced BEFORE any rental-6 page: row-granular L2 keys on OLMoE w1.

    python scripts/scoring/rental6/rowkeys.py [--device mps|cpu] [--check-cpu N]

Rental 5's skewed OLMoE w1 cells read +8% (n 4) and +25% (n 16) over the registered byte
model, and model v3's content keys (term (a)) do not move a skewed cell: on unequal counts no
two experts share a token set. The rival (diag-r6 Q2, `l2rows.py`, ported here) keys w1's A
by HIDDEN-STATE ROW and k-step, not by tile: an M-tile reads its 32 token rows as 32 lines
of 128 B per k-step, so tiles that share some rows share those lines, as L2 sectors do. The
launch is vLLM's pid mapping (dead CTAs skipped), each live CTA's k-steps last occ x 344
cycles (open loop, sigma 0), CTAs start when the slot `SMs x occ` ranks earlier frees, and
one global LRU of C MiB (C_MIB, the diag's 39.5 MiB; 60 MiB printed as the capacity edge)
decides every access by its exact stack distance. A miss on a read is a DRAM read.

It writes docs/registered/2026-10-09-rental6-skew-hist/rowkey_rival.json: per OLMoE w1 cell of
the byte page (PT and PW at n 3, 6 NATIVE and SHARED, 12 NATIVE; uniform at each n as the
anchor) the predicted read bytes at each C, the device, the dtype and the wall time. The
registration (rental6-skew, OPEN-L2SKEW) embeds its sha256 and its numbers. LABEL: the rule is
SEEN-SELECTED (diag Q2 picked it on rental 5's skewed OLMoE pages); its numbers here are
BLIND (new draws).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from moe.spec import MODEL_CONFIGS  # noqa: E402

OUT = REPO / "docs" / "registered" / "2026-10-09-rental6-skew-hist" / "rowkey_rival.json"
PAGE = REPO / "docs" / "registered" / "2026-10-09-rental6-skew-hist" / "olmoe-1b-7b-bytes.json"
MODEL = "olmoe-1b-7b"
C_MIB = (39.5, 60.0)
BM, BN, BK = 32, 64, 64
U = 128          # one A row line per k-step (BLOCK_K 64 bf16)
SMS = 132
OCC_W1 = 5
C_CYC = 344.0
G = 8


def walk(R: int, P: int, G: int, live: int):
    """vLLM's pid mapping over R x P pids, live rows only: (pid_m, pid_n) in launch order."""
    pid = np.arange(R * P, dtype=np.int64)
    gid = pid // (G * P)
    first = gid * G
    size = np.minimum(R - first, G)
    pm = first + (pid % (G * P)) % size
    pn = (pid % (G * P)) // size
    keep = pm < live
    return pm[keep], pn[keep]


def expert_rows(counts, tokens: int, k: int, shuffle_seed):
    """Per expert, its token rows (after the page's shuffle), ascending: R3's histogram_ids."""
    import torch

    from moe.routing.distributions import realize_counts
    ids = realize_counts(list(counts), tokens, k, device="cpu")
    if shuffle_seed is not None:
        g = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
        ids = ids[torch.randperm(tokens, generator=g)]
    ids = ids.numpy()
    return [np.nonzero((ids == e).any(axis=1))[0].tolist() for e in range(len(counts))]


def stack_hits(key, tm, tb, size, Cu, dev, dtype):
    """Accesses sorted by (time, tiebreak); hit iff the key was touched before and the distinct
    size touched since plus its own fits in Cu (a weighted merge-sort tree, batched)."""
    import torch
    order = torch.argsort(tb, stable=True)
    order = order[torch.argsort(tm[order], stable=True)]
    key, size = key[order], size[order]
    N = key.numel()
    pos = torch.arange(N, device=dev)
    o2 = torch.argsort(key, stable=True)
    ks = key[o2]
    same = torch.zeros(N, dtype=torch.bool, device=dev)
    same[1:] = ks[1:] == ks[:-1]
    prev = torch.full((N,), -1, dtype=torch.long, device=dev)
    prev[o2] = torch.where(same, torch.cat([o2[:1], o2[:-1]]), torch.full_like(o2, -1))
    nsame = torch.zeros(N, dtype=torch.bool, device=dev)
    nsame[:-1] = same[1:]
    nxt = torch.full((N,), N, dtype=torch.long, device=dev)
    nxt[o2] = torch.where(nsame, torch.cat([o2[1:], o2[-1:]]), torch.full_like(o2, N))
    has = prev >= 0
    qi = pos[has]
    lo = prev[has] + 1
    hi = qi.clone()
    D = torch.zeros(qi.numel(), device=dev, dtype=dtype)
    P2 = 1 << math.ceil(math.log2(max(N, 2)))
    nv = torch.cat([nxt, torch.full((P2 - N,), -1, dtype=torch.long, device=dev)]) + 1
    wv = torch.cat([size, torch.zeros(P2 - N, device=dev, dtype=dtype)])
    BIG = N + 3
    lev = 0
    while (1 << lev) <= P2 and bool((lo < hi).any()):
        bs = 1 << lev
        nb = P2 // bs
        vals, idx = torch.sort(nv.view(nb, bs), dim=1)
        ww = torch.gather(wv.view(nb, bs), 1, idx)
        suf = torch.flip(torch.cumsum(torch.flip(ww, [1]), 1), [1])
        flat = (vals + torch.arange(nb, device=dev)[:, None] * BIG).reshape(-1)
        sufx = torch.cat([suf.reshape(-1), torch.zeros(1, device=dev, dtype=dtype)])
        del vals, idx, ww, suf
        for side in (0, 1):
            take = (lo < hi) & ((lo % 2 == 1) if side == 0 else (hi % 2 == 1))
            blk = lo if side == 0 else hi - 1
            if bool(take.any()):
                bid = blk[take]
                thr = qi[take] + 2
                p = torch.searchsorted(flat, bid * BIG + thr)
                endb = (bid + 1) * bs
                p = torch.minimum(p, endb)
                D[take] += torch.where(p < endb, sufx[p], torch.zeros_like(sufx[p]))
            if side == 0:
                lo = torch.where(take, lo + 1, lo)
            else:
                hi = torch.where(take, hi - 1, hi)
        lo = lo // 2
        hi = hi // 2
        lev += 1
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    hit[qi] = (D + size[qi]) <= Cu
    return order, hit


def access_trace(arm, counts, tokens, k, P, S, R, shuffle_seed, *, occ=OCC_W1, sms=SMS,
                 c_cyc=C_CYC, G=G, device="cpu", dtype=None):
    """Every access of one w1 launch: (key, time, tiebreak, size in 128 B units, is-read)."""
    import torch
    dtype = dtype or torch.float64
    dev = torch.device(device)
    t = [-(-c // BM) for c in counts]
    own = [(e, j) for e, te in enumerate(t) for j in range(te)]
    live = len(own)
    pm, pn = walk(R, P, G, live)
    rows = expert_rows(counts, tokens, k, shuffle_seed)
    tok = np.full((live, BM), -1, dtype=np.int64)
    for r, (e, j) in enumerate(own):
        tt = rows[e][j * BM:(j + 1) * BM]
        tok[r, :len(tt)] = tt
    exp_of = np.array([e for e, _ in own], dtype=np.int64)
    owner = pm if arm == "private" else exp_of[pm]
    b = owner * P + pn
    Lc = len(pm)
    slots = sms * occ
    dur = torch.full((Lc, S), occ * c_cyc, device=dev, dtype=dtype)
    clen = dur.sum(1)
    Wv = -(-Lc // slots)
    cl = torch.cat([clen, torch.zeros(Wv * slots - Lc, device=dev, dtype=dtype)]).view(Wv, slots)
    start = (torch.cumsum(cl, 0) - cl).reshape(-1)[:Lc]
    tstep = start[:, None] + torch.cumsum(dur, 1) - dur
    tend = start + clen
    tk = torch.as_tensor(tok[pm], device=dev)
    valid = tk >= 0
    s_idx = torch.arange(S, device=dev)
    ci, si, ri = torch.nonzero(valid[:, None, :].expand(Lc, S, BM), as_tuple=True)
    keyA = tk[ci, ri] * S + si
    tA = tstep[ci, si]
    tbA = (ci * S + si).to(dtype) * 2
    T = int(tok.max()) + 1
    b_t = torch.as_tensor(b, device=dev)
    keyB = T * S + (b_t[:, None] * S + s_idx[None, :]).reshape(-1)
    nB = int(b.max()) + 1
    keyO = T * S + nB * S + torch.arange(Lc, device=dev)
    key = torch.cat([keyA, keyB, keyO])
    tm = torch.cat([tA, tstep.reshape(-1), tend])
    tb = torch.cat([tbA, (torch.arange(Lc * S, device=dev) * 2 + 1).to(dtype),
                    (torch.arange(Lc, device=dev) * 2 * S + 2 * S - 1).to(dtype)])
    size = torch.cat([torch.ones(keyA.numel(), device=dev, dtype=dtype),
                      (BN * BK * 2 // U) * torch.ones(Lc * S, device=dev, dtype=dtype),
                      (BM * BN * 2 // U) * torch.ones(Lc, device=dev, dtype=dtype)])
    isread = torch.cat([torch.ones(keyA.numel() + Lc * S, device=dev, dtype=torch.bool),
                        torch.zeros(Lc, device=dev, dtype=torch.bool)])
    return key, tm, tb, size, isread


def simulate(arm, counts, tokens, k, P, S, R, C_mib, shuffle_seed, *, device="cpu", dtype=None, **kw):
    """DRAM read bytes of one w1 launch under row keys at C MiB, and the wall seconds."""
    import torch
    dtype = dtype or torch.float64
    t0 = time.time()
    dev = torch.device(device)
    key, tm, tb, size, isread = access_trace(arm, counts, tokens, k, P, S, R, shuffle_seed,
                                             device=device, dtype=dtype, **kw)
    order, hit = stack_hits(key, tm, tb, size, C_mib * (1 << 20) / U, dev, dtype)
    miss = (size[order] * (isread[order] & ~hit).to(dtype)).sum()
    if dev.type == "mps":
        torch.mps.synchronize()
    return float(miss) * U, time.time() - t0, int(key.numel())


def cells():
    """The OLMoE byte page's w1 cells the rival is priced on."""
    import r3_timing_model as TM
    cfg = MODEL_CONFIGS[MODEL]
    page = json.loads(PAGE.read_text())
    out = []
    old = TM.set_model(MODEL)
    try:
        for c in page["cells"]:
            if c["label"] not in ("PT", "PW", "uniform"):
                continue
            for arm in c["arms"]:
                D = cfg.num_experts * (1 if arm == "native" else 9)
                counts = tuple(int(v) for v in c["counts"])
                out.append(dict(label=c["label"], n=int(c["n"]), arm=arm, declared=D, counts=counts,
                                tokens=sum(counts) // cfg.top_k, k=cfg.top_k,
                                P=2 * cfg.intermediate_size // BN, S=cfg.hidden_size // BK,
                                R=TM.grid_rows(D, int(c["n"]), counts)))
    finally:
        TM.set_model(old)
    return out, int(page["shuffle_seed"])


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--check-cpu", type=int, default=1,
                    help="re-run the first N cells on the CPU in float64 and record the difference")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    dt = torch.float32 if a.device == "mps" else torch.float64
    cs, seed = cells()
    t0 = time.time()
    res = []
    for c in cs:
        row = {k: v for k, v in c.items() if k != "counts"}
        for C in C_MIB:
            b, sec, N = simulate(c["arm"], c["counts"], c["tokens"], c["k"], c["P"], c["S"], c["R"], C, seed,
                                 device=a.device, dtype=dt)
            row[f"read_bytes_C{C:g}"] = b
            row[f"wall_s_C{C:g}"] = round(sec, 2)
            row["accesses"] = N
        print(c["label"], c["n"], c["arm"], {k: v for k, v in row.items() if k.startswith("read")}, flush=True)
        res.append(row)
    checks = []
    for c, row in list(zip(cs, res, strict=True))[:a.check_cpu]:
        b, sec, _ = simulate(c["arm"], c["counts"], c["tokens"], c["k"], c["P"], c["S"], c["R"], C_MIB[0], seed,
                             device="cpu", dtype=torch.float64)
        checks.append(dict(label=c["label"], n=c["n"], arm=c["arm"], cpu_float64=b,
                           rel_diff=abs(row[f"read_bytes_C{C_MIB[0]:g}"] / b - 1), cpu_wall_s=round(sec, 2)))
    doc = {"what": "OPEN-L2SKEW rival: row-granular w1 A keys, exact LRU, priced before any rental-6 page",
           "model": MODEL, "page": str(PAGE.relative_to(REPO)), "shuffle_seed": seed, "C_MiB": list(C_MIB),
           "registered_C_MiB": C_MIB[0], "timing": {"sms": SMS, "occ_w1": OCC_W1, "cycles_per_kstep": C_CYC, "sigma": 0.0},
           "device": a.device, "dtype": str(dt).replace("torch.", ""), "wall_s": round(time.time() - t0, 1),
           "cpu_float64_check": checks, "cells": res}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out} in {doc['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
