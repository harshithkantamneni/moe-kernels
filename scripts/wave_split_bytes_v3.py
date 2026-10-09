"""MODEL M4 (rental 6, model v3, 2026-10-09): content keys on HISTOGRAM cells.

A separate module over `wave_split_bytes` (which stays byte-identical to the sha rental 5's
registration pins). `ModelV3(geom)` is `wave_split_bytes.Model(geom, content_a=True)` (w1's A
keyed by tile content on R3's balanced cells) that ALSO keys w1's A by content on a histogram
cell: a cell `(arm, G, n, gemm, counts, shuffle_seed)` takes `content_events_counts`. On the
uniform histogram it equals `content_events` on the balanced cell (tests/test_rental6_model.py).
"""
from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

import wave_split_bytes as W
from moe.spec import MODEL_CONFIGS

_CONTENT_IDS_COUNTS: dict = {}


def tile_content_ids_counts(model: str, counts, block_m: int,
                            shuffle_seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(content id, rows) per live M-tile of a histogram cell, expert-major in `tile_owners`
    order. The routing is R3's histogram page's own (`private_weight_reference.histogram_ids`):
    `realize_counts(counts, T, k)`, then the page's token-row shuffle
    `ids[randperm(T, manual_seed(shuffle_seed))]` (none when `shuffle_seed` is None), each
    expert's rows in ascending token order. On unequal counts no two experts share a token set,
    so every id is distinct; on the uniform histogram the E/k classes survive the shuffle."""
    key = (model, tuple(int(v) for v in counts), int(block_m), shuffle_seed)
    if key not in _CONTENT_IDS_COUNTS:
        import torch

        from moe.routing.distributions import realize_counts
        cfg = MODEL_CONFIGS[model]
        tokens = int(sum(key[1])) // cfg.top_k
        ids = realize_counts(list(key[1]), tokens, cfg.top_k, device="cpu")
        if shuffle_seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
            ids = ids[torch.randperm(tokens, generator=g)]
        ids = ids.numpy()
        seen: dict = {}
        cid, rows = [], []
        for e in range(cfg.num_experts):
            toks = np.nonzero((ids == e).any(axis=1))[0]
            if toks.size != key[1][e]:
                raise W.Refused(f"{model}: expert {e} holds {toks.size} rows, not {key[1][e]}")
            for j in range(-(-toks.size // block_m)):
                ch = tuple(toks[j * block_m:(j + 1) * block_m].tolist())
                cid.append(seen.setdefault(ch, len(seen)))
                rows.append(len(ch))
        _CONTENT_IDS_COUNTS[key] = (np.array(cid, dtype=np.int64), np.array(rows, dtype=np.int64))
    return _CONTENT_IDS_COUNTS[key]


def content_events_counts(geom, ev, counts, shuffle_seed: int | None):
    """`wave_split_bytes.content_events` on a histogram cell (a w1 SHARED or NATIVE walk with
    `counts`): the first read of a content id is compulsory; every later read is an A event at
    its live-rank distance, weighted by its tile's rows / BLOCK_M; the group working set counts
    distinct contents x ATILE + distinct owners x SLAB; the compulsory offset is
    (distinct-content rows - all tile rows) / BLOCK_M A tiles, in slab units."""
    if ev.gemm != "w1":
        return ev
    if ev.arm == "private":
        raise W.Refused("MODEL M4: a histogram cell is NATIVE or SHARED (R3 runs no PRIVATE histogram)")
    G, P, Wc, npm = ev.G, ev.P, ev.W_c, ev.num_pid_m
    ctab, crow = tile_content_ids_counts(geom.model, counts, geom.block_m, shuffle_seed)
    owner, trows = W.tile_owners(counts, geom.block_m)
    live_m = owner.size
    pid = np.arange(npm * P, dtype=np.int64)
    gid, pid_m, _pid_n = W.pid_map(pid, npm, P, G)
    keep = pid_m < live_m
    gid, pid_m = gid[keep], pid_m[keep]
    cid = ctab[pid_m]
    cur, prev = W._previous(cid)
    D = cur - prev
    w1 = cur < Wc
    groups = -(-npm // G)
    ws = np.zeros(groups)
    at, sl = geom.atile("w1"), geom.slab("w1")
    for g in range(groups):
        first = g * G
        tiles = range(first, min(first + G, live_m)) if first < live_m else range(0)
        ws[g] = len({int(ctab[m]) for m in tiles}) * at + len({int(owner[m]) for m in tiles}) * sl
    a: Counter = Counter()
    frac = crow[pid_m[cur]] / geom.block_m
    for k, f in zip(zip(D.tolist(), w1.tolist(), ws[gid[cur]].tolist(), strict=True), frac.tolist(), strict=True):
        a[k] += f
    keys = sorted(a)
    uniq = dict(zip(ctab.tolist(), crow.tolist(), strict=True))
    off = (sum(uniq.values()) - int(trows.sum())) / geom.block_m * at / sl
    return dataclasses.replace(
        ev, slabs=ev.slabs + off,
        a_D=np.array([k[0] for k in keys], dtype=float),
        a_win1=np.array([k[1] for k in keys], dtype=bool),
        a_ws=np.array([k[2] for k in keys], dtype=float),
        a_k=np.array([a[k] for k in keys], dtype=float))


class ModelV3(W.Model):
    """The WSC model with content_a on balanced AND histogram cells (MODEL M4)."""

    def __init__(self, geom):
        super().__init__(geom, content_a=True)

    def events(self, arm: str, G: int, n: int, gemm: str, counts=None, shuffle_seed=None):
        if counts is None:
            return super().events(arm, G, n, gemm)
        ct = tuple(int(v) for v in counts)
        key = (arm, G, n, gemm, ct, shuffle_seed)
        if key not in self._events:
            ev = W.cell_events(self.geom, arm, G, n, gemm, ct)
            self._events[key] = content_events_counts(self.geom, ev, ct, shuffle_seed)
        return self._events[key]
