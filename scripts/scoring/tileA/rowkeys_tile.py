#!/usr/bin/env python
"""The row-key A-key candidates at vLLM's tiles, priced on EVERY scored w1 row of rental A's
G-1 byte pages BEFORE any tile-stage page (registration 2026-10-14-tileA-bytes; design-tile 7,
design-tile-review 5 and 9.13).

    python scripts/scoring/tileA/rowkeys_tile.py [--only PAGE] [--max-cells N]   # writes rowkey_tile.json
    python scripts/scoring/tileA/rowkeys_tile.py --check-pinned N                  # the equality check below

THE RIVAL is rental 6's row-key LRU (scripts/scoring/rental6/rowkeys.py, pinned, never edited):
w1's A keyed by hidden-state row and k-step (one 128 B line per row per k-step at BLOCK_K 64),
B by (expert, N-tile, k-step), vLLM's pid mapping, an open-loop launch in which every CTA k-step
lasts occ x c cycles, and one exact LRU of C MiB deciding every access by its stack distance (a
read is a hit iff the bytes of the distinct lines touched since its previous access, plus its
own, fit in C). Its own `simulate` materialises every access of a launch and a merge-sort tree
over them: at 128x128 that is 4 to 75 million accesses and, on the laptop's unified memory, it
ran past 19 GB (2026-10-14) and was killed.

THIS FILE STREAMS THE SAME LAUNCH. With every CTA the same length, a wave's CTAs start together,
so the rival's access order (time, then its tie-break) is: per wave, per k-step, per CTA in
launch order, the CTA's A rows (ascending tile row) then its B line; after each wave, its CTAs'
output tiles. An LRU cache of weighted capacity C that evicts from the cold end while it holds
more than C is exactly the set of lines whose stack distance plus size fits in C, so a hit here
is a hit there, line for line. Memory is the cache (at most C / 128 B entries) plus one CTA's
rows: about 100 MB. `--check-pinned N` runs rowkeys.simulate on N small cells (OLMoE n 1) and
requires EQUAL read bytes, and every run aborts above MAX_RSS_GB resident.

THE TILE. The rival's BM/BN/BK are module state (rowkeys.BM, BN, BK); this file passes the tile
explicitly instead. c moves no byte (every CTA has the same duration, so the order of accesses
is the same at any c); the occupancy does (the slots are SMs x occ): rental A's census occupancy
(tcommon.occupancy_of, `--census-score`), the shared-memory rule's (2 at 128x128x64 s3, 3 at
64x128x64 w8 s3) until the census exists. Re-running this file with the census, then register.py
with it, is the registered mechanical step.

CAPACITIES. 60 MiB is rental 7's NAMED key (SEEN-chosen, OPEN). 30 MiB is one of the card's two
L2 partitions: the GH200's L2 read 60 MiB on rental 4's hw rulers (CONSISTENT with RRZE's GH200
run), and Hopper's L2 is split into two partitions (NVIDIA H100 architecture whitepaper). This
replaces the design's "Fan et al." attribution, which no source in the repo supports
(design-tile-review 9.13); the review's "50 MB, about 25 MB per partition" is the H100 SXM's
published L2, not the figure this card measured.

NATIVE AND SHARED AT G 1. vLLM's pid mapping at GROUP_SIZE_M 1 orders the live CTAs by
(pid_m, pid_n) whatever the grid's dead rows, so NATIVE's and SHARED's w1 walks are the same
accesses in the same order: each cell is priced once and carried to both arms (a test holds it).

Integers only (no float rounding), on the CPU, one cell at a time, the JSON rewritten after each
cell, so a stopped run keeps every cell it finished.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "rental6")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import rowkeys as RK  # noqa: E402
import skew_synth_tile as ST  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

OUT = ST.OUT["A"] / "rowkey_tile.json"
C_MIB = (60.0, 30.0)
U = RK.U                                   # 128 B: one A row line per k-step at BLOCK_K 64
MAX_RSS_GB = 4.0
#: the G-1 byte pages whose w1 rows the A-key rule scores, and their tiles; the occupancy is read
#: from rental A's census (tcommon.occupancy_of; --census-score), the shared-memory rule's before it
PAGES = {"olmoe-1b-7b-128g1": "128x128x64w8s3", "mixtral-8x7b-64g1": "64x128x64w8s3",
         "qwen2-57b-a14b-128g1": "128x128x64w8s3"}


def page_tile(name: str):
    import r3_timing_model_v4 as V4
    return V4.Tile.parse(PAGES[name])


def page_occ(name: str) -> tuple[int, str]:
    sys.path.insert(0, str(HERE))
    import tcommon as TC
    model = name.rsplit("-", 1)[0]
    occ, src = TC.occupancy_of(page_tile(name), model)
    return occ["w1"], src


class TooBig(Exception):
    pass


def rss_gb() -> float:
    """Peak resident set of this process, GB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


def guard() -> None:
    if rss_gb() > MAX_RSS_GB:
        raise TooBig(f"resident set {rss_gb():.2f} GB is over the {MAX_RSS_GB} GB guard: stopped")


def stream_misses(arm, counts, tokens, k, P, S, R, shuffle_seed, *, bm, bn, bk, occ, sms=RK.SMS, G=1,
                  caps_mib=C_MIB) -> tuple[dict, int]:
    """{C: DRAM read bytes} of one w1 launch under row keys, for every capacity at once, and the
    accesses walked. The access order and the hit rule are rowkeys.simulate's (module docstring)."""
    t = [-(-int(c) // bm) for c in counts]
    own = [(e, j) for e, te in enumerate(t) for j in range(te)]
    live = len(own)
    pm, pn = RK.walk(R, P, G, live)
    rows = RK.expert_rows(counts, tokens, k, shuffle_seed)
    tile_rows = [np.asarray(rows[e][j * bm:(j + 1) * bm], dtype=np.int64) for e, j in own]
    exp_of = np.array([e for e, _ in own], dtype=np.int64)
    owner = pm if arm == "private" else exp_of[pm]
    b = (owner * P + pn).astype(np.int64)
    Lc = int(pm.size)
    T = int(max(int(r.max()) for r in tile_rows if r.size)) + 1
    BKEY = T * S
    nB = int(b.max()) + 1
    OKEY = BKEY + nB * S
    bsize, osize = bn * bk * 2 // U, bm * bn * 2 // U
    caps = [int(C * (1 << 20) // U) for C in caps_mib]
    caches = [OrderedDict() for _ in caps]
    total = [0 for _ in caps]
    miss = [0 for _ in caps]
    slots = sms * occ
    accesses = 0

    def touch(key: int, size: int, read: bool) -> None:
        for i, cache in enumerate(caches):
            if key in cache:
                cache.move_to_end(key)
                continue
            if read:
                miss[i] += size
            cache[key] = size
            total[i] += size
            cap = caps[i]
            while total[i] > cap:
                _k, sz = cache.popitem(last=False)
                total[i] -= sz

    for w0 in range(0, Lc, slots):
        cis = range(w0, min(w0 + slots, Lc))
        for s in range(S):
            for ci in cis:
                for r in (tile_rows[pm[ci]] * S + s).tolist():
                    touch(r, 1, True)
                touch(BKEY + int(b[ci]) * S + s, bsize, True)
                accesses += tile_rows[pm[ci]].size + 1
            guard()
        for ci in cis:
            touch(OKEY + ci, osize, False)
            accesses += 1
    return {C: float(m * U) for C, m in zip(caps_mib, miss, strict=True)}, accesses


def cells(only: str | None = None) -> list[dict]:
    out = []
    for name in PAGES:
        if only and name != only:
            continue
        t = page_tile(name)
        bm, bn, bk = t.bm, t.bn, t.bk
        occ, _src = page_occ(name)
        page = json.loads((ST.OUT["A"] / f"{name}.json").read_text())
        model = page["model"]
        cfg = MODEL_CONFIGS[model]
        seed = int(page["shuffle_seed"])
        for c in page["cells"]:
            n = int(c["n"])
            tokens = ST.tokens_at(model, n, bm)
            if c["counts"] is None:
                counts, cseed = tuple([tokens * cfg.top_k // cfg.num_experts] * cfg.num_experts), None
            else:
                counts, cseed = tuple(int(v) for v in c["counts"]), seed
            D = cfg.num_experts
            out.append(dict(page=name, model=model, label=c["label"], n=n, arms=list(c["arms"]), counts=counts,
                            shuffle_seed=cseed, tokens=tokens, k=cfg.top_k, P=2 * cfg.intermediate_size // bn,
                            S=cfg.hidden_size // bk, R=-(-(sum(counts) + D * (bm - 1)) // bm), tile=[bm, bn, bk],
                            occ=occ, G=int(page["group_m"])))
    return out


def price(c: dict) -> dict:
    t0 = time.time()
    bm, bn, bk = c["tile"]
    got, N = stream_misses("native", c["counts"], c["tokens"], c["k"], c["P"], c["S"], c["R"], c["shuffle_seed"],
                           bm=bm, bn=bn, bk=bk, occ=c["occ"], G=c["G"])
    row = {f"read_bytes_C{C:g}": v for C, v in got.items()}
    row.update(accesses=N, wall_s=round(time.time() - t0, 2), peak_rss_gb=round(rss_gb(), 3))
    return row


def pinned_check(n_cells: int) -> list[dict]:
    """rowkeys.simulate (the pinned rival, its tile set in memory) against the stream on the
    first n_cells OLMoE n 1 cells, CPU float64: their read bytes must be EQUAL. n 1 at 128x128 is
    about 4.2 million accesses, about 1.5 GB for the pinned code; the guard still applies."""
    import torch
    out = []
    for c in [x for x in cells("olmoe-1b-7b-128g1") if x["n"] == 1][:n_cells]:
        bm, bn, bk = c["tile"]
        old = (RK.BM, RK.BN, RK.BK)
        RK.BM, RK.BN, RK.BK = bm, bn, bk
        try:
            ref = {}
            for C in C_MIB:
                v, _s, _N = RK.simulate("native", c["counts"], c["tokens"], c["k"], c["P"], c["S"], c["R"], C,
                                        c["shuffle_seed"], device="cpu", dtype=torch.float64, occ=c["occ"], G=c["G"])
                ref[f"read_bytes_C{C:g}"] = v
                guard()
        finally:
            RK.BM, RK.BN, RK.BK = old
        got = price(c)
        out.append({"label": c["label"], "n": c["n"], "pinned_cpu_float64": ref,
                    "stream": {k: got[k] for k in ref}, "equal": all(got[k] == ref[k] for k in ref),
                    "peak_rss_gb": round(rss_gb(), 3)})
        print(json.dumps(out[-1]), flush=True)
    return out


def _write(path: Path, res: list, meta: dict) -> None:
    doc = dict(meta, cells=res)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, choices=sorted(PAGES))
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--check-pinned", type=int, default=0, metavar="N")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--census-score", type=Path, default=None, help="rental A's census.score.json (the occupancy)")
    a = ap.parse_args(argv)
    if a.census_score is not None:
        sys.path.insert(0, str(HERE))
        import tcommon as TC
        TC.use_census(a.census_score)
    try:
        if a.check_pinned:
            chk = pinned_check(a.check_pinned)
            print(f"pinned check: {sum(x['equal'] for x in chk)} of {len(chk)} cells EQUAL; peak RSS {rss_gb():.2f} GB")
            return 0 if all(x["equal"] for x in chk) else 1
        meta = {"what": "tile-stage A-key candidates: rental 6's row-key LRU (rowkeys.py, pinned) streamed at the tile "
                        "(rowkeys_tile.stream_misses: the same access order and hit rule, bounded memory) on every w1 "
                        "row of rental A's G-1 byte pages, priced before any tile page",
                "pages": {k: {"tile": PAGES[k], "occupancy_w1": page_occ(k)[0], "occupancy_source": page_occ(k)[1]}
                          for k in PAGES},
                "C_MiB": list(C_MIB), "arms": "NATIVE priced; SHARED carried (the same walk at G 1)",
                "timing": {"sms": RK.SMS, "sigma": 0.0,
                           "note": "c moves no byte (every CTA lasts occ x c per k-step); the occupancy does"},
                "device": "cpu", "dtype": "int64 (exact)", "max_rss_gb_guard": MAX_RSS_GB}
        cs = cells(a.only)[:a.max_cells] if a.max_cells else cells(a.only)
        res, t0 = [], time.time()
        for c in cs:
            row = {k: v for k, v in c.items() if k != "counts"}
            row.update(price(c))
            res.append(row)
            print(c["page"], c["label"], c["n"], {k: v for k, v in row.items() if k.startswith("read")},
                  f"{row['wall_s']} s, peak RSS {row['peak_rss_gb']} GB", flush=True)
            _write(a.out, res, dict(meta, wall_s=round(time.time() - t0, 1)))
        print(f"wrote {a.out}: {len(res)} cells in {time.time() - t0:.1f} s, peak RSS {rss_gb():.2f} GB")
        return 0
    except TooBig as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
