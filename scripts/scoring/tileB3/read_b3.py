#!/usr/bin/env python
"""Tile rental B3's finaliser and scorer (docs/registered/2026-11-04-tileB3-gh200): the blind test of the round-5
large-tile model on rental B2's 84 frozen NATIVE cells, at the owner's working-model bar.

    python scripts/scoring/tileB3/read_b3.py finalise --b3 results/published/<date>-nvidia_gh200_480gb-tileB3-session
        # writes scripts/scoring/tileB3/b3.predictions.json (sorted keys) from the frozen model and the board's four
        # sag pages ONLY; the owner commits it before `score` is run
    python scripts/scoring/tileB3/read_b3.py score --b3 <the same directory> [--out-dir scripts/scoring/tileB3]
        # re-finalises, refuses unless the predictions file equals the finalised file (not a git check), then writes
        # tileB3.score.{json,txt}

THE MODEL, FROZEN (moe-kernels-work 2026-10-28 round5/ROUND5.md's selected variant, the final declared round): the
STAG-CUT w1 byte law (M4's staggered rank distance, the carried soft survival, the 60 MiB cut on later-wave w1
re-reads; no fitted constant) with W2 on w2, the K1 wall knee at round 4's SEEN constants (p 6.0, S0 2.75, the A2
triad 3726.2 GB/s), the clock form of F1, F1 'par' from the board's OLMoE f128 and Mixtral-8x7B f128 pages; the
64x128 cells at v5'' (null knee, p 5.75 passed, p_64 3.75, M4 / F3 bytes, v5'''s own 64x128 sag (30.37 us, 300 us),
CARRIED there). The code is round5/work r7.py and what it imports (round 4's r6 -> r5 -> refit4 r4 -> knee-v5 k5 -> diag-v5
d5 -> design-v5 and diag-A2), every file pinned by sha256 in the registration and checked before anything is
imported; the CAL pricer is built at p 14 and T0 43.23 us and bw 3597.8 GB/s are asserted (the first-build trap).

THE 64x128 SAG is the board's own (gate b3sag64, the owner's choice), not A2's CARRIED one; the fingerprint keeps
round 5's CARRIED 30.37 us / 300 us. NOT ADOPTED: the align amendment of B3 v2 (block-scan align priced by numel) was
tried offline and worsened the held-out fit (OLMoE +0.45 -> +2.01); the per-call risk is disclosed in the registration.

FINALISATION (mechanical). The registration holds, computed before the rental, every cell's prediction at A2's, C2's
and C4's pars (the frozen model's map from the board's sag to a time, sampled at three boards) and the fingerprint
the reader must reproduce. On the rental the gates b3sag and b3sag64 fit the four sag pages (b3common.page_fit, the
chain's r4.sag_page_fit copied verbatim), enforce the registered ranges as STOPs before any timed unit, and put both
pars and the pages' sha256 on the ledger. `finalise` then recomputes every fit with the frozen chain AND with the
repo's copy, refuses unless they agree with each other and with the gate records, and prices the 84 cells at those
pars (each par to a relative 1e-9 of the gate record's, b3common.PAR_RTOL). It opens no blind page: an audit hook
refuses the run if any path under the B3 tree inside a gaps-* directory, or naming a blind model (except the
compile-only census), is opened while it runs.

THE TEST (E1-TILE, the registered verdict). e = T_model / T_meas - 1 per valid NATIVE cell (tcommon.timed_page:
G5 provenance, the r7-median-run3 rebuild, the view; tcommon.valid_cells: 6 of 9 clean, G2 host on the finalised
prediction and the timer flag, G4 >= 1405 MHz). HOLDS iff the pooled NATIVE cells and each of Phi-3.5-MoE,
Qwen1.5-MoE-A2.7B and JetMoE-8B meet rms <= 3%, every cell within 8%, |mean| <= 1.5% (b3common.BAR), in ALL and in
CLEAN (rental 2's rule 3); deepseek-v2-lite in the pooled bar only. G3 as r5common.g3_rule reads it: a homogeneous
failure voids every histogram cell (E1-TILE NOT SCORED), a heterogeneous one voids the failing strata's non-balanced
cells (the verdict is read with those voids applied; the bar without them is printed beside). CERTIFIED: a blind
model whose bar holds in CLEAN with at least 2 of its pages there, and only when E1-TILE HOLDS.
DESCRIPTIVE beside: B2's own bar (2%, 5%, 1%), the rivals (v5'', round 4's model, v4 at A2's selections), the split
views, B-BYTES and E2-TILE from the byte pages, the end-of-rental OLMoE f128 repeat (drift of the sag).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(HERE), str(HERE.parent / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import b3common as B  # noqa: E402
import read_c2 as RC  # noqa: E402  (its pure part: board, the commit and pin checks; its chain() is never called)

WORK = Path.home() / "code" / "moe-kernels-work"
ROUND5 = WORK / "2026-10-28" / "round5" / "work"
ROUND4 = WORK / "2026-10-27" / "round3" / "work"
REG_NAME = B.NAME
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
B2_REG_PATH = REPO / "docs" / "registered" / f"{B.B2_NAME}.json"
PLAN = B.PLAN
PRED_FILE = HERE / "b3.predictions.json"

#: every work file the frozen chain imports or reads (traced with an audit hook over r7's import, the 84 blind cells
#: priced, the sag fits and round 5's selected table re-priced: tests/tileB3_chain_job.py trace), and the round-4 and
#: round-5 outputs this reader takes numbers from (knee4.json's K1 constants; bytes8.json's selection; combo5.json's
#: table, estimate and stakes; DECLARED4.txt and DECLARED5.txt, the declared forms); the registration pins each
FROZEN_FILES = (
    "2026-10-20/scoreA2/bytes.score.json", "2026-10-20/scoreA2/census.score.json",
    "2026-10-20/scoreA2/floor.score.json", "2026-10-20/scoreA2/rep.score.json",
    "2026-10-21/diag-A2/work/common.py",
    "2026-10-21/design-v5/work/v5price.py", "2026-10-21/design-v5/work/kneefit.py",
    "2026-10-21/design-v5/work/sag.py", "2026-10-21/design-v5/work/b2risk.py",
    "2026-10-21/design-v5/work/f3fit.json", "2026-10-21/design-v5/work/kneefit.json",
    "2026-10-21/design-v5/work/kneefit64.json",
    "2026-10-22/diag-v5/work/d5.py", "2026-10-22/diag-v5/work/bw.py", "2026-10-22/diag-v5/work/knee2.py",
    "2026-10-22/diag-v5/work/v5prime.py", "2026-10-22/diag-v5/work/bytes_lomo.json",
    "2026-10-22/knee-v5/work/k5.py",
    "2026-10-22/refit4/work/r4.py", "2026-10-22/refit4/work/refit4.py", "2026-10-22/refit4/work/bytes4.json",
    "2026-10-27/round3/work/r5.py", "2026-10-27/round3/work/r6.py", "2026-10-27/round3/work/tailmap.json",
    "2026-10-27/round3/work/knee4.json", "2026-10-27/round3/work/bytes7.json", "2026-10-27/round3/work/combo4.json",
    "2026-10-27/round3/work/DECLARED4.txt",
    "2026-10-28/round5/work/r7.py", "2026-10-28/round5/work/bytes8.json", "2026-10-28/round5/work/combo5.json",
    "2026-10-28/round5/work/DECLARED5.txt",
)

# ------------------------------------------------------------------ frozen constants (registered by value)
O, Q, M, X22, TP2 = "olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b", "mixtral-8x7b-tp2"
LAW = "STAG-CUT"
SELECTED = "SELECTED: STAG-CUT (w1) + W2 (w2) + K1 wall knee + clock form"
K1_SEEN = (6.0, 2.75)
TRIAD_A2_GBPS = 3726.2
#: what the frozen code passes for a 64x128 cell (combo5.py PART D: k5.cell_ms(r, None, 5.75, sag128, sag64) in
#: r5.law3(), p_64 3.75 inside it) and v5'''s own 64x128 sag, CARRIED from A2's f64w8 pages (knee-v5 KNEE.md 'par'):
#: the fingerprint's 64x128 sag; B3's predictions take the board's own (gate b3sag64)
P_V5PP = 5.75
SAG64 = B.SAG64_A2
#: A2's 64x128 par by the registered estimator (b3common.page_fit on A2's f64w8 pages; round 5 priced the rounded
#: 30.37): the pre-rental predictions' 64x128 sag
A2_PAR_64 = (30.365, 300.0)
A2_PAR_128 = (42.02, 425.0)
C2_PAR_128 = (41.59, 412.5)
C4_PAR_128 = (54.25, 587.5)
REF_PARS = {"A2": A2_PAR_128, "C2": C2_PAR_128, "C4": C4_PAR_128}
#: round 5's selected table (combo5.json), re-priced by the frozen code at registration and before any page
ROUND5_TABLE = {**B.ROUND5_MEANS, "mixtral-8x7b C2": B.ROUND5_M_C2}
#: round 5's estimate at B2's bar on 84 cells (combo5.json pass, selected, 'registered (84 cells)')
ROUND5_PASS_B2_BAR = {"A (5 models)": 0.124, "B (5 models)": 0.1903, "D (O, Q, 8x22B, tp2; 8x7B G1 aside)": 0.1381}
#: every repo module the frozen chain and the scorer import from the chain's repo (diag-A2 common.REPO), each required
#: there with its registered sha256; B3's own modules (b3common, read_b3) are loaded from this reader's repo
#: and held by the loaded-module rule (repo_modules_bad) and by code_pins
CHAIN_REPO_MODULES = RC.CHAIN_REPO_MODULES + ("scripts/scoring/cal2/read_c2.py", "scripts/scoring/tileA2/census_a2.py",
                                              "scripts/tile_census_a2.py", "scripts/tile_census.py",
                                              "scripts/scoring/rental5/r5common.py", "scripts/scoring/rental3/r3common.py")
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", PLAN)
GATE_RECORDS = {"census": "gate-b3census-1410.json", "sag": "gate-b3sag-1410.json", "sag64": "gate-b3sag64-1410.json"}
REPEAT_UNIT = "f128r"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(d) -> str:
    return B.dumps(d)


# ================================================================== the frozen chain

def frozen_hashes() -> dict:
    return {f: sha256(WORK / f) for f in FROZEN_FILES}


def check_frozen(reg: dict) -> None:
    want = reg["frozen"]["files"]
    got = frozen_hashes()
    bad = [f for f in FROZEN_FILES if want.get(f) != got[f]]
    if bad or set(want) != set(FROZEN_FILES):
        raise SystemExit(f"REFUSED: the frozen chain is not the registered one: {bad or sorted(set(want) ^ set(FROZEN_FILES))}")


_CHAIN: dict = {}
#: the module names the frozen chain imports from the work folders (generic names other repo modules share)
CHAIN_MODULES = ("r7", "refit4", "r6", "r5", "r4", "k5", "d5", "common", "v5price", "kneefit", "sag", "b2risk", "bw",
                 "knee2", "v5prime")


def chain():
    """round 5's r7 (round5/work), imported once, with refit4 (r7._rc), r6, r5, r4, k5, d5 and b2risk; the CAL pricer
    is built at p 14 and T0 43.23 us and bw 3597.8 GB/s are asserted (the first-build trap: d5, k5, r4, r5, r7)."""
    if not _CHAIN:
        keep_path = list(sys.path)
        keep_mods = {m: sys.modules.pop(m) for m in CHAIN_MODULES if m in sys.modules}
        try:
            sys.path.insert(0, str(ROUND5))
            import r7  # noqa: E402
            RF = r7._rc()          # refit4 (imported from its folder; its RC is this reader's read_c2)
            import b2risk  # noqa: E402
        finally:
            for m in CHAIN_MODULES:
                sys.modules.pop(m, None)
            sys.modules.update(keep_mods)
            sys.path[:] = keep_path
        r6, r5, r4, k5, d5 = r7.r6, r7.r5, r7.r4, r7.k5, r7.d5
        assert d5.TM.P_KNEE == 14.0
        x = [float(v) for v in d5.RG.pricer().f2.x]
        assert abs(x[0] * 1e3 - 43.23) < 0.01 and abs(x[2] - 3597.8) < 0.1, x
        assert abs(k5.RULERS["triad"] - TRIAD_A2_GBPS) < 0.1 and r5.SIGMA_GAP == 0.0375 and r5.L2_MIB == 60.0
        kn = json.loads((ROUND4 / "knee4.json").read_text())
        assert kn["selection"]["chosen"] == "K1" and tuple(kn["candidates"]["K1"]["SEEN"]) == K1_SEEN, kn["selection"]
        assert json.loads((ROUND5 / "bytes8.json").read_text())["selection"]["chosen"] == LAW
        c5 = json.loads((ROUND5 / "combo5.json").read_text())
        assert c5["law"] == LAW and c5["jetmoe_n2_PT"] == "PRICED" and c5["unpriced_b2"] == []
        assert c5["tables"][SELECTED]["summary"]["means"] == B.ROUND5_MEANS
        assert c5["tables"][SELECTED]["summary"]["8x7B C2"] == B.ROUND5_M_C2
        assert c5["pass"][SELECTED]["registered (84 cells)"] == ROUND5_PASS_B2_BAR
        assert r7.W1_LAWS[LAW] == {"walk": "STAG", "surv": "SOFT", "cut": True, "nconst": 0}
        _CHAIN.update(r7=r7, RF=RF, r6=r6, r5=r5, r4=r4, k5=k5, d5=d5, B=b2risk)
    return _CHAIN


def knee():
    C = chain()
    return C["k5"].Form("K1", C["k5"].RULERS["triad"], K1_SEEN[1])


def b2_rows() -> list:
    """B2's 84 NATIVE cells in B2's registered power-cell order (b2risk.b2_cells)."""
    rows = chain()["B"].b2_cells()
    cells = RC.b2_cells()
    assert len(rows) == len(cells) == 84
    assert all(a["model"] == b["model"] and a["cls"] == b["cls"] and a["page"] == b["page"] for a, b in zip(rows, cells))
    return rows


def reads(r: dict):
    """{w1, w2} DRAM reads of a 128x128 cell under STAG-CUT on w1 and W2 on w2, at p 6.0; None: the guard refuses."""
    return chain()["r7"].reads5(r, K1_SEEN[0], LAW)


def price(r: dict, sag128: tuple, sag64: tuple = SAG64) -> tuple:
    """(ms, reads) of one blind NATIVE cell under the frozen round-5 model (combo5.py PART D's loop, verbatim) at the
    128x128 par sag128 and the 64x128 par sag64 (round 5's CARRIED one by default)."""
    C = chain()
    r6, r5, k5 = C["r6"], C["r5"], C["k5"]
    if int(r["doc"]["block_m"]) == 128:
        rd = reads(r)
        if rd is None:
            return None, None
        return r6.cell_ms4(r, knee(), K1_SEEN[0], tuple(sag128), rd, clock=True), {g: float(rd[g]) for g in ("w1", "w2")}
    with r5.law3():
        ms = k5.cell_ms(r, None, P_V5PP, tuple(sag128), tuple(sag64))
        key = (r["model"], r.get("page"), r["n"], r["label"], int(r["doc"]["block_m"]), int(r["doc"]["group_m"]))
        rd = k5._READS.get(key)
    return ms, (None if rd is None else {g: float(rd[g]) for g in ("w1", "w2")})


def rivals(r: dict, sag128: tuple, sag64: tuple = SAG64) -> dict:
    """DESCRIPTIVE rivals at the same pars: v5'' (null knee, M4), round 4's frozen model (W2 on both
    GEMMs), v4 at A2's selections (the CAL pricer, no sag)."""
    C = chain()
    r6, r5, k5, d5 = C["r6"], C["r5"], C["k5"], C["d5"]
    with r5.law3():
        v5pp = k5.cell_ms(r, None, P_V5PP, tuple(sag128), tuple(sag64))
    if int(r["doc"]["block_m"]) == 128:
        rd4 = r6.reads_of(r, K1_SEEN[0], "W2")
        r4m = None if rd4 is None else r6.cell_ms4(r, knee(), K1_SEEN[0], tuple(sag128), rd4, clock=True)
    else:
        r4m = v5pp
    prm = d5.params_for(d5.SEL, r["model"])
    with d5.A2.scoring(d5.RULES):
        v4 = d5.RG.price_cell(r["model"], "native", r["c"], r["doc"], prm, key=d5.SEL["key"])["ms"]
    return {"v5pp": v5pp, "round4": r4m, "v4": float(v4)}


def predictions(sag128: tuple, with_rivals: bool = True, raw: bool = False, sag64: tuple = SAG64) -> list:
    """The 84 cells priced at the pars (ms rounded to 6 decimals, reads to bytes); raw=True keeps the unrounded ms and
    rivals under 'raw' (for the stakes, which round 5 computed from unrounded values)."""
    out = []
    for i, r in enumerate(b2_rows()):
        ms, rd = price(r, sag128, sag64)
        row = {"index": i, "model": r["model"], "page": r["page"], "unit": r["unit"], "cls": r["cls"], "n": int(r["n"]),
               "label": r["label"], "arm": "native", "G": int(r["doc"]["group_m"]), "block_m": int(r["doc"]["block_m"]),
               "ms": None if ms is None else round(float(ms), 6),
               "reads": None if rd is None else {g: int(round(v)) for g, v in rd.items()}}
        if with_rivals:
            rv = rivals(r, sag128, sag64)
            row["rivals"] = {k: (None if v is None else round(float(v), 6)) for k, v in rv.items()}
            if raw:
                row["raw"] = {"ms": ms, **rv}
        out.append(row)
    return out


def fingerprint(sag128: tuple = A2_PAR_128) -> dict:
    """The frozen model's ms (A2's CARRIED 64x128 sag, as round 5 priced it) on B2's 84 cells at a par (A2's by default), and its
    128x128 reads, rounded: the chain has not moved."""
    p = predictions(sag128, with_rivals=False)
    return {"ms": [c["ms"] for c in p], "reads": [None if c["reads"] is None else [c["reads"]["w1"], c["reads"]["w2"]]
                                                  for c in p]}


def stakes(preds: list) -> dict:
    """Per (model, class): the frozen model against v5'' and against round 4's model, % (round 5 PART D's numbers,
    from the unrounded values: predictions(..., raw=True))."""
    by = {}
    for c in preds:
        rv = c["raw"]
        for k in (f"{c['model']} {c['cls']}", f"{c['model']} (all)"):
            d = by.setdefault(k, {"vs_v5pp": [], "vs_round4": [], "round4_vs_v5pp": []})
            d["vs_v5pp"].append(round(100 * (rv["ms"] / rv["v5pp"] - 1), 3))
            if rv["round4"] is not None:
                d["vs_round4"].append(round(100 * (rv["ms"] / rv["round4"] - 1), 3))
                d["round4_vs_v5pp"].append(round(100 * (rv["round4"] / rv["v5pp"] - 1), 3))
    return {k: {kk: ({"n": len(v), "mean": round(st.fmean(v), 2), "min": round(min(v), 2), "max": round(max(v), 2)}
                     if v else {"n": 0}) for kk, v in d.items()} for k, d in sorted(by.items())}


def round5_table() -> dict:
    """Round 5's selected five-model held-out table re-priced by the frozen code (combo5.py's table(LAW) loop: the
    w1 law at each fold's p, K1 at the fold's LOMO (p, S0) (SEEN for tp2), the clock form, F1 par of the fold's
    estimator pages on the timed page's board, C4's both pages for tp2)."""
    C = chain()
    r7, r6, k5, r4, RF = C["r7"], C["r6"], C["k5"], C["r4"], C["RF"]
    KW = json.loads((ROUND4 / "knee4.json").read_text())["candidates"]["K1"]
    fits = RF.sag_fits()
    _c4per, c4par = r7.c4_sag_par()
    a2, c2x, c2m = RF.timed_cells()
    sets = {O: [r for r in a2 if r["model"] == O], Q: [r for r in a2 if r["model"] == Q],
            M: [r for r in a2 if r["model"] == M], X22: c2x, TP2: r7.c4_timed_rows(), "mixtral-8x7b C2": c2m}
    out = {}
    for k, cells in sets.items():
        hm = M if k == "mixtral-8x7b C2" else k
        p, s0 = tuple(KW["SEEN"]) if hm == TP2 else tuple(KW["LOMO"][hm])
        f = k5.Form("K1", k5.RULERS["triad"], s0)
        e = []
        for r in cells:
            sag = tuple(c4par) if r["board"] == "C4" else tuple(RF.fold_F1(fits, "C2" if r["board"] == "C2" else "A2", hm))
            rd = r7.reads5(r, p, LAW)
            e.append(r6.cell_ms4(r, f, p, sag, rd, clock=True) / r["meas"] - 1)
        out[k] = r4.summ(e)["mean"]
    return out


def chain_sag_fits(results: Path, unit: str = B.EST_UNIT) -> dict:
    """{model: (A, tau)} by the frozen chain's estimator on the board's pages: r4.sag_page_fit on the f128 pages, and
    the same design-v5 sag.fit over sag.pts_of (r4's k5.S) on the f64w8 pages."""
    C = chain()
    if unit == B.EST_UNIT:
        return {m: tuple(C["r4"].sag_page_fit(Path(results), m)) for m in B.EST_MODELS}
    S = C["k5"].S
    out = {}
    for m in B.EST_MODELS:
        f = B.page_files(results, m, unit)
        assert len(f) == 1, (results, m, unit, f)
        r = S.fit(S.pts_of(f, float(B.LOCK)))
        out[m] = (r["A_us"], r["tau_us"])
    return out


def repo_modules_bad(R: dict) -> list:
    """[(path, why)]: every repo module loaded from the chain's repo and from this reader's repo (read_c2.repo_modules:
    absolute, git-tracked, outside tests/ and any environment directory) is a registered pin with its registered
    bytes, and every CHAIN_REPO_MODULES path in the chain's repo has its registered sha256."""
    bad = []
    root_chain = Path(chain()["d5"].V.REPO).resolve()
    for root in sorted({root_chain, REPO.resolve()}):
        loaded = RC.repo_modules(list(sys.modules.values()), root, RC.tracked_files(root))
        bad += [(rel, f"loaded from {root} but not pinned") for rel in loaded if rel not in R["code_pins"]]
        bad += [(rel, f"under {root}: not the registered sha256") for rel in loaded
                if rel in R["code_pins"] and sha256(root / rel) != R["code_pins"][rel]]
    bad += [(p, f"under the chain's repo {root_chain}: not the registered sha256") for p in CHAIN_REPO_MODULES
            if p not in R["code_pins"] or not (root_chain / p).is_file() or sha256(root_chain / p) != R["code_pins"][p]]
    return sorted(set(bad))


# ================================================================== the guard: finalisation reads no blind page

_GUARD = {"on": False, "hit": [], "roots": ()}


def blind_path(p: str) -> bool:
    """A path finalisation must not open: inside a timed unit's gaps-* directory, or naming a blind model (Phi-3.5,
    Qwen1.5, JetMoE, deepseek-v2-lite: their timed and byte pages), except the compile-only census."""
    return "/gaps-" in p or (any(m in p for m in B.MODELS) and "-tile-census-" not in p)


def _audit(ev, args):
    """While finalise runs: any open of a path under the B3 tree (as given, and resolved) that blind_path names is
    recorded, and finalise refuses. (The frozen pricer reads earlier published gaps pages of CAL models in a temporary
    as-of tree to build its CAL fit, and B2's registered histogram files; those are outside the B3 tree.)"""
    if _GUARD["on"] and ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if blind_path(p) and any(p.startswith(r) or os.path.abspath(p).startswith(r) for r in _GUARD["roots"]):
            _GUARD["hit"].append(p)


sys.addaudithook(_audit)


def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def preflight(R: dict) -> dict:
    """Before any page: the frozen model reproduces the registered fingerprint (84 cells at A2's par) and round 5's
    selected table; the repo modules loaded so far are pinned."""
    fp = fingerprint()
    if fp != R["frozen"]["fingerprint"]:
        raise SystemExit("REFUSED: the frozen round-5 model does not reproduce the registered fingerprint: the chain moved")
    t5 = round5_table()
    if t5 != ROUND5_TABLE or t5 != R["frozen"]["round 5 selected table, re-priced by the frozen code at registration"]:
        raise SystemExit(f"REFUSED: the frozen model does not reproduce round 5's table: {t5} against {ROUND5_TABLE}")
    bad = repo_modules_bad(R)
    if bad:
        raise SystemExit(f"REFUSED: a repo module loaded before any page is not its registered bytes: {bad[:5]}")
    return {"fingerprint": "equals the registration", "round5_table": t5, "chain_repo_before_pages": "pinned"}


def check_tree(b3: Path, R: dict, *, skip_commit: bool = False) -> dict:
    """The freeze tied to the rental's commit (C2's and C4's rule) and the pins here."""
    out = {}
    if skip_commit:
        out["commit"] = {"sha": None, "checked": False}
    else:
        sha, why = RC.run_commit(Path(b3) / "session")
        if sha is None:
            raise SystemExit(f"REFUSED: the rental's commit is not on its ledger: {why}")
        bad = RC.check_commit(REPO, sha, sorted(set(COMMIT_FILES) | set(R["code_pins"])))
        if bad:
            raise SystemExit(f"REFUSED: the working tree is not the rental's commit {sha}: {bad[:5]}")
        out["commit"] = {"sha": sha, "checked": True}
    bad = RC.check_pins(R, REPO)
    if bad:
        raise SystemExit(f"REFUSED: a pinned file is not its registered bytes: {bad[:5]}")
    return out


def gate_record(session: Path, which: str) -> dict | None:
    f = Path(session) / "gh200-driver" / GATE_RECORDS[which]
    return json.loads(f.read_text()) if f.is_file() else None


def finalise(b3: Path, *, registration: dict | None = None, skip_commit: bool = False) -> dict:
    """The predictions of record: the frozen model at the board's 128x128 and 64x128 pars, from the four sag pages and
    the gate records only. Refuses on the freeze, the commit, the pins, the fingerprint, a sag gate (sag, sag64) that
    did not PASS, the 64x128 clock clause failing, any difference between the frozen chain's fits and the repo copy's,
    a par differing from the gate record's by more than b3common.PAR_RTOL (relative) or a page sha256 differing, and
    any blind page (a gaps-* path, or a path naming a blind model other than the census) opened."""
    R = registration or reg()
    check_frozen(R)
    b3 = Path(b3)
    results, session = b3 / "results", b3 / "session"
    out = {"registration": REG_NAME, "what": ("tile rental B3's predictions of record: the frozen round-5 model at this "
                                               "board's 128x128 and 64x128 sag pars, finalised mechanically before any "
                                               "blind page is read")}
    out.update(check_tree(b3, R, skip_commit=skip_commit))
    _GUARD["on"], _GUARD["hit"] = True, []
    _GUARD["roots"] = tuple(sorted({str(b3) + "/", str(b3.absolute()) + "/", str(b3.resolve()) + "/"}))
    try:
        fp = fingerprint()
        if fp != R["frozen"]["fingerprint"]:
            raise SystemExit("REFUSED: the frozen round-5 model does not reproduce the registered fingerprint")
        g = {k: gate_record(session, k) for k in ("sag", "sag64")}
        for k, v in g.items():
            if v is None or v.get("verdict") != "PASS":
                raise SystemExit(f"REFUSED: the {k} gate did not PASS on this board ({None if v is None else v.get('why')})")
        pars, per = {}, {}
        for k, unit in (("sag", B.EST_UNIT), ("sag64", B.SAG64_UNIT)):
            mine = {m: B.page_fit(results, m, unit) for m in B.EST_MODELS}
            frozen = chain_sag_fits(results, unit)
            if any((mine[m]["A_us"], mine[m]["tau_us"]) != frozen[m] for m in B.EST_MODELS):
                raise SystemExit(f"REFUSED: the repo's sag estimator and the frozen chain's disagree on the {unit} pages: "
                                 f"{mine} against {frozen}")
            rec = B.sag_check(mine, B.SAG_RANGE if k == "sag" else B.SAG64_RANGE, unit)
            if rec["verdict"] != "PASS" or not B.par_close(rec["par"], g[k]["par"]) or any(
                    mine[m]["sha256"] != g[k]["per_page"][m]["sha256"] for m in B.EST_MODELS):
                raise SystemExit(f"REFUSED: the {k} gate record's par or page sha256 is not what the {unit} pages give")
            if k == "sag64":
                clk = B.clock64(results)
                if clk["verdict"] != "PASS":
                    raise SystemExit(f"REFUSED: the f64w8 pages' clock clause fails here: {clk['why']}")
            par = tuple(chain()["r4"].par([frozen[m] for m in B.EST_MODELS]))
            if not B.par_close(par, rec["par"]):
                raise SystemExit(f"REFUSED: r4.par {par} is not the {k} gate's {rec['par']}")
            pars[k] = par
            per[unit] = {m: {kk: mine[m][kk] for kk in ("A_us", "tau_us", "rms_us", "cells", "sha256")} for m in B.EST_MODELS}
        preds = predictions(pars["sag"], sag64=pars["sag64"])
    finally:
        _GUARD["on"] = False
    if _GUARD["hit"]:
        raise SystemExit(f"REFUSED: finalisation opened a blind page: {_GUARD['hit'][:3]}")
    unpriced = [c["index"] for c in preds if c["ms"] is None]
    out.update(par=list(pars["sag"]), par64=list(pars["sag64"]),
               per_page=per[B.EST_UNIT], per_page64=per[B.SAG64_UNIT],
               gate={k: g["sag"].get(k) for k in ("verdict", "par", "ratio_1ms", "inside_held_out_par_span (DESCRIPTIVE)")},
               gate64={k: g["sag64"].get(k) for k in ("verdict", "par", "ratio_1ms")},
               cells=preds, unpriced=unpriced, n_cells=len(preds))
    return out


# ================================================================== the score

def timed_view(tree: Path, preds: dict, tv, files: dict) -> dict:
    """One view (ALL or CLEAN): each blind timed page through tcommon.timed_page, its valid NATIVE cells, G3 with the
    registered per-stratum voids, E1-TILE at the working-model bar, B2's bar and the rivals beside."""
    C = chain()
    d5 = C["d5"]
    TC, RG = d5.TC, d5.RG
    meas, use, dropped, in_view = {}, {}, {}, {}
    for label, model, name, _drop, _est in RG.B_TIMED_PAGES:
        f = files[f"{name}.json"]
        rep, why = TC.timed_page(tree, model, label, f["group_m"], tv, f)
        use[f"{model}-{label}"] = why
        if rep is None:
            continue
        sub = {k: v for k, v in preds.items() if k[0] == model and k[1] == name}
        c, d, _A = TC.valid_cells(rep, sub, model, name)
        if c:
            in_view.setdefault(model, []).append(name)
        dropped[f"{model}-{label}"] = d
        meas.update(c)
    g3 = TC.g3_rows(meas, preds, None)
    g3v = TC.C5.g3_rule(g3) if g3 else {"verdict": "NOT SCORED: no G3 row", "void_strata": None}
    vs = g3v.get("void_strata")
    rows, voided = [], []
    for k, m in meas.items():
        pr = preds[k]
        ns = sorted({x[2] for x in preds if x[:2] == k[:2]})
        row = {"cell": "/".join(map(str, k)), "model": k[0], "cls": pr["cls"], "G": pr["G"], "n": k[2], "label": k[3],
               "e": pr["ms"] / m - 1, "meas": m, "pred": pr["ms"],
               "rivals": {R_: (None if v is None else v / m - 1) for R_, v in (pr.get("rivals") or {}).items()}}
        if isinstance(vs, list) and str(ns.index(k[2])) in vs and k[3] != "balanced":
            voided.append(row["cell"])
            row["voided"] = True
        rows.append(row)
    kept = [r for r in rows if not r.get("voided")]
    E1 = B.e1_rule(kept, in_view, tv.name, B.BAR)
    if vs == "ALL":
        E1 = {"verdict": "NOT SCORED: G3 FAILS (homogeneous), so every histogram cell is voided",
              "descriptive": B.e1_rule(rows, in_view, tv.name, B.BAR)}
    desc = {"B2's bar (2%, 5%, 1%) on the same cells": {k: v for k, v in B.e1_rule(kept, in_view, tv.name, B.B2_BAR).items()},
            "the bar without G3's per-stratum voids": B.e1_rule(rows, in_view, tv.name, B.BAR) if voided else "no cell voided",
            "rivals (e of each rival, the same cells)": {R_: B.stats([r["rivals"][R_] for r in kept if r["rivals"].get(R_) is not None])
                                                         for R_ in ("v5pp", "round4", "v4")},
            "split: 128 G 16 cells": B.stats([r["e"] for r in kept if r["G"] == 16]),
            "split: 128 G 1 cells": B.stats([r["e"] for r in kept if r["cls"] == "128g1"]),
            "split: 64 G 1 cells": B.stats([r["e"] for r in kept if r["cls"] == "64g1"]),
            "per model and class": {f"{m} {c}": B.stats([r["e"] for r in kept if r["model"] == m and r["cls"] == c])
                                    for m in B.MODELS for c in ("64g1", "128g1", "128g16")
                                    if any(r["model"] == m and r["cls"] == c for r in kept)}}
    for d in desc.values():
        if isinstance(d, dict) and "verdict" in d:   # DESCRIPTIVE: keep it out of rule 3's verdict combination
            d["reading"] = d.pop("verdict")
            for pm in (d.get("per_model") or {}).values():
                pm["reading"] = pm.pop("verdict", None)
                pm.pop("certification", None)
            if isinstance(d.get("pooled"), dict):
                d["pooled"]["reading"] = d["pooled"].pop("verdict", None)
    return {"pages": use, "dropped_rows": dropped, "G3": {k: v for k, v in g3v.items() if k != "rows"},
            "voided_cells": voided, "E1-TILE": E1, "DESCRIPTIVE": desc, "cells_measured": len(meas),
            "cells": [{k: v for k, v in r.items() if k != "rivals"} for r in rows]}


def bytes_legs(tree: Path, preds: dict, files: dict, par: tuple, timed: dict) -> dict:
    """DESCRIPTIVE: B-BYTES (predicted against counted w1 and w2 reads per GEMM, NATIVE, B2's bar rms 3% and every row
    5%) and E2-TILE (the frozen model's time from the counted bytes, 128x128 cells, against the ALL view's valid
    NATIVE times; B2's E2 bar beside)."""
    C = chain()
    d5, r6 = C["d5"], C["r6"]
    TC, RG = d5.TC, d5.RG
    rows, e2, pages = [], [], {}
    for label, model, name, _drop, _est in RG.B_BYTE_PAGES:
        f = files[f"{name}.json"]
        p, path, why = TC.counter_page(tree, model, label, f["group_m"], f, f["tile"])
        if p is not None:
            clk = TC.C6.clock_rule(p, path)
            if not clk["page_ok"]:
                p, why = None, f"NOT SCORED (clock rule): {clk['why']}"
        pages[f"{model}-{label}"] = why
        if p is None:
            continue
        for cell, cc in zip(p.get("cells") or [], clk["cells"], strict=True):
            k = (model, name, int(cell["n"]), str(cell.get("histogram") or "balanced"), str(cell["arm"]))
            if k not in preds or not cc["ok"] or preds[k]["reads"] is None:
                continue
            counted = {g: float(cell["per_gemm"][g]["dram_bytes_read"]) for g in ("w1", "w2")}
            for g in ("w1", "w2"):
                rows.append({"cell": "/".join(map(str, k)), "gemm": g, "e": preds[k]["reads"][g] / counted[g] - 1})
            if preds[k]["block_m"] == 128 and k in timed:
                r = next(x for x in b2_rows() if (x["model"], x["page"], int(x["n"]), x["label"]) == k[:4])
                T = r6.cell_ms4(r, knee(), K1_SEEN[0], tuple(par), counted, clock=True)
                e2.append(T / timed[k] - 1)
    bb = {g: B.stats([r["e"] for r in rows if r["gemm"] == g]) for g in ("w1", "w2")}
    ok = all(s["cells"] and s["rms"] <= 0.03 and abs(s["worst"]) <= 0.05 for s in bb.values())
    return {"pages": pages, "B-BYTES (DESCRIPTIVE)": {"reading": ("within" if ok else "outside") + " B2's B-BYTES bar"
                                                      if rows else "no byte row", "per_gemm": bb},
            "E2-TILE (DESCRIPTIVE, 128x128 cells)": {"stats": B.stats(e2),
                                                     "within B2's E2 bar (2%, 5%, 1.5%)": bool(e2) and B.bar(
                                                         e2, {"rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.015})["verdict"] == "HOLDS"}}


def drift(results: Path, early: dict) -> dict:
    """DESCRIPTIVE: the end-of-rental OLMoE f128 repeat (label f128r) against the early OLMoE fit."""
    f = B.page_files(results, O, REPEAT_UNIT)
    if len(f) != 1:
        return {"why": f"{len(f)} OLMoE {REPEAT_UNIT} pages (droppable)"}
    r = B.fit(B.pts_of(f, float(B.LOCK)))
    e = early[O]
    return {"repeat": {k: r[k] for k in ("A_us", "tau_us", "rms_us", "cells")}, "early": {k: e[k] for k in ("A_us", "tau_us")},
            "sag_at_1ms_us": {"early": round(B.sat_us(e["A_us"], e["tau_us"], 1000.0), 2),
                              "repeat": round(B.sat_us(r["A_us"], r["tau_us"], 1000.0), 2)}}


def score(b3: Path, *, registration: dict | None = None, skip_commit: bool = False, committed: Path | None = PRED_FILE) -> dict:
    R = registration or reg()
    b3 = Path(b3)
    results, session = b3 / "results", b3 / "session"
    fin = finalise(b3, registration=R, skip_commit=skip_commit)
    if committed is not None:
        if not Path(committed).is_file():
            raise SystemExit(f"REFUSED: no predictions file {committed}: run `finalise` first (the owner commits it before "
                             "`score`; score checks that the file equals the finalised one, not git)")
        if json.loads(Path(committed).read_text()) != json.loads(dumps(fin)):
            raise SystemExit("REFUSED: the predictions file does not equal the finalised file on these pages")
    out = {"registration": REG_NAME, "b3": str(b3), "status": "BLIND (E1-TILE at the working-model bar)",
           "commit": fin["commit"], "par": fin["par"], "per_page_sag": fin["per_page"], "par64": fin["par64"],
           "per_page_sag64": fin["per_page64"]}
    out["preflight"] = preflight(R)
    out["board"] = brd = RC.board(session)
    gates = {k: gate_record(session, k) for k in GATE_RECORDS}
    out["gates"] = {k: (None if v is None else {"verdict": v.get("verdict"), "why": v.get("why")}) for k, v in gates.items()}
    why = [w for w in ((brd.get("why") if not brd["valid"] else ""),
                       *(f"gate {k} reads {(v or {}).get('verdict')!r}" for k, v in gates.items() if (v or {}).get("verdict") != "PASS")) if w]
    out["validity"] = {"valid": not why, "why": "; ".join(why)}
    C = chain()
    d5 = C["d5"]
    TC, A2 = d5.TC, d5.A2
    files = json.loads(B2_REG_PATH.read_text())["pages"]["files"]
    preds = {(c["model"], c["page"], c["n"], c["label"], "native"): c for c in fin["cells"] if c["ms"] is not None}
    if (gates.get("census") or {}).get("deepseek", "admitted") != "admitted":
        preds = {k: v for k, v in preds.items() if k[0] != B.DSV}
        out["deepseek"] = "NOT SCORED (census drift)"
    which = "fallback" if str(brd.get("g_band_which") or "primary").startswith("fallback") else "primary"
    rules = A2.Rules.tileA2(float(B.LOCK), which)
    if not why:
        with A2.scoring(rules):
            res = TC.C3.two_views(lambda fv, tv: timed_view(results, preds, tv, files), repo=REPO)
            timed = {}
            tv = TC.C3.TimedView("ALL")
            for label, model, name, _d, _e in d5.RG.B_TIMED_PAGES:
                rep, _w = TC.timed_page(results, model, label, files[f"{name}.json"]["group_m"], tv, files[f"{name}.json"])
                if rep is not None:
                    c, _dd, _A = TC.valid_cells(rep, {k: v for k, v in preds.items() if k[0] == model and k[1] == name}, model, name)
                    timed.update(c)
            out["bytes"] = bytes_legs(results, preds, files, tuple(fin["par"]), timed)
        clean = ((res.get("addendum") or {}).get("CLEAN") or {}).get("E1-TILE") or {}
        verdict = (res.get("E1-TILE") or {}).get("verdict")
        out["E1-TILE"] = res.get("E1-TILE")
        out["certification"] = B.certification(clean, verdict)
        out["views"] = {k: v for k, v in res.items() if k != "E1-TILE"}
    else:
        out["E1-TILE"] = {"verdict": f"NOT SCORED: the board or a gate is not valid ({'; '.join(why)})"}
        out["certification"] = {m: "NOT CERTIFIED (not scored)" for m in B.BLIND}
    out["sag drift (DESCRIPTIVE)"] = drift(results, fin["per_page"])
    out["scope"] = R["certification"]["scope"]
    bad = repo_modules_bad(R)
    if bad:
        raise SystemExit(f"REFUSED: the reading ran a repo module that is not its registered bytes: {bad[:5]}")
    return out


def txt(d: dict) -> str:
    e = d.get("E1-TILE") or {}
    L = [f"TILE RENTAL B3 ({d['registration']}): {d['status']}", f"  pages: {d['b3']}",
         f"  commit: {json.dumps(d['commit'], sort_keys=True)}", f"  board: {json.dumps(d['board'], sort_keys=True)}",
         f"  gates: {json.dumps(d['gates'], sort_keys=True)}", f"  par (on-board F1): {d['par']}",
         f"  par64 (on-board 64x128 F1): {d['par64']}",
         f"E1-TILE (rms <= 3%, every cell within 8%, |mean| <= 1.5%): {e.get('verdict')}",
         f"  pooled: {json.dumps((e.get('pooled') or {}).get('stats'), sort_keys=True)}"]
    for m, v in (e.get("per_model") or {}).items():
        L.append(f"  {m}: {v.get('verdict')} {json.dumps(v.get('stats'), sort_keys=True)}")
    L.append(f"certification: {json.dumps(d['certification'], sort_keys=True)}")
    L.append(f"scope: {d['scope']}")
    if "bytes" in d:
        L.append(f"bytes (DESCRIPTIVE): {json.dumps({k: v for k, v in d['bytes'].items() if k != 'pages'}, sort_keys=True)}")
    L.append(f"sag drift (DESCRIPTIVE): {json.dumps(d['sag drift (DESCRIPTIVE)'], sort_keys=True)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("finalise", "score"))
    ap.add_argument("--b3", type=Path, required=True, help="the published B3 session directory (results/, session/)")
    ap.add_argument("--out", type=Path, default=PRED_FILE, help="finalise: the predictions file")
    ap.add_argument("--out-dir", type=Path, default=HERE, help="score: where tileB3.score.{json,txt} go")
    a = ap.parse_args(argv)
    if a.what == "finalise":
        d = finalise(a.b3)
        a.out.write_text(dumps(d))
        print(f"wrote {a.out}: {d['n_cells']} cells, {len(d['unpriced'])} unpriced, par {d['par']}")
        return 0
    d = score(a.b3)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "tileB3.score.json").write_text(dumps(d))
    (a.out_dir / "tileB3.score.txt").write_text(txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
