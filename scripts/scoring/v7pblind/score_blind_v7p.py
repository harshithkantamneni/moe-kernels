#!/usr/bin/env python
"""Finalises and scores model v7's prefill blind (registration docs/registered/2026-11-18-v7-prefill-blind-gh200) on the
rental's published session.

    python scripts/scoring/v7pblind/score_blind_v7p.py <session dir: results/ and session/> <out dir>

It reads the registration at its committed path only (no option names another file) and records that file's sha256
and the checkout's commit. It REFUSES (exit 2) a registration that is not this one, a DRAFT, a malformed registration,
one whose freeze does not hold (v7 as the gate froze it, round 5's vendored files, this blind's pages and code, B3's
driver and rules), or whose fingerprint (every cell's v7 prediction at the four reference pars) the registered pricer
does not reproduce now; a checkout that is dirty; a session whose ledger names no single run commit
(driver start: commit <sha>, checked against commit.txt) or whose commit is not the working copy for the registration,
its plan and every frozen file (B3's read_b3.check_tree rule); and a finalisation that does not hold (the sag par below).

THE POWER CHECK FIRST (the decode blinds' order, B3's board validity): c2-power.env C2_POWER_W=700, locks.env
LOCK_PROBE_1410=held, G-BAND PASS, and the census (v7pcensus) and sag (b3sag) gate records PASS; else NOT SCORED
(exit 1), nothing certified.

FINALISATION (mechanical, before any blind page is read): the board's 128x128 par from its two f128 pages by the
repo's estimator and the frozen chain's, equal to each other and to the b3sag record (v7pcommon.finalise_pars); every
cell priced by the registered pricer (v7, every switch on) at that par. An audit hook refuses the run if finalisation
opens a gaps-* path or a path naming a blind model (the census aside).

THE READING: each timed page through tcommon.timed_page (G5 provenance, the r7-median-run3 rebuild, the registered
tile) and tcommon.valid_cells (row_drop) in ALL and CLEAN; G3 (r5common.g3_rule: homogeneous fail, every histogram
cell voided and NOT SCORED; heterogeneous, the failing strata's non-balanced cells voided); per model and pooled at
rms 3%, worst 8%, |mean| 1.5%, the two views combined by rental 2's rule 3. A model CERTIFIED when its combined bar
HOLDS, the combined pooled bar HOLDS and at least 10 of its cells are scored in CLEAN (the minimum); else NOT
CERTIFIED, BELOW THE MINIMUM named. DESCRIPTIVE beside: B3's E1-TILE conjunction, B2's bar, round 5 (every switch
off), the G 1 / G 16 splits, the bar without G3's voids. G3's homogeneous fail voids every histogram cell: the test reads NOT SCORED
(exit 1). Writes prefill_blind.score.json and .txt. Exit: 0 scored;
1 NOT SCORED; 2 REFUSED."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import read_c2 as RC  # noqa: E402  (its pure part: board)
import v7pcommon as V  # noqa: E402

B = V.B


class Refused(Exception):
    pass


# ================================================================== the guard: finalisation reads no blind page

_GUARD = {"on": False, "hit": [], "roots": ()}


def _audit(ev, args):
    if _GUARD["on"] and ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if V.blind_path(p) and any(p.startswith(r) or os.path.abspath(p).startswith(r) for r in _GUARD["roots"]):
            _GUARD["hit"].append(p)


sys.addaudithook(_audit)


# ================================================================== refusals

def refusals(reg) -> list[str]:
    """Why this registration may not be scored (empty: it may). A malformed registration is a refusal, never a
    traceback."""
    if not isinstance(reg, dict):
        return [f"the registration is malformed: a {type(reg).__name__}"]
    if reg.get("name") != V.NAME:
        return [f"the registration is not {V.NAME}"]
    if reg.get("draft") is not False:
        return ["the registration is a DRAFT (the owner's go is not recorded): nothing may be scored against it"]
    bad = V.verify_freeze(reg)
    if bad:
        return ["the freeze does not hold: " + "; ".join(bad[:10])]
    return fingerprint_refusals(reg)


def fingerprint_refusals(reg: dict, fp: dict | None = None) -> list[str]:
    """The registered pricer recomputes every cell's prediction at the four reference pars; it must equal the
    registration's fingerprint (the model has not moved) and the cells must be the registered ones."""
    import register_v7p_blind as RG
    try:
        want = reg["predictions"]["fingerprint (the scorer recomputes it before reading anything)"]
        cells = [(c["model"], c["page"], c["n"], c["label"]) for c in reg["predictions"]["cells at A2's par"]]
    except (KeyError, TypeError) as ex:
        return [f"the registration's predictions are malformed ({type(ex).__name__}: {ex})"]
    fp = V.frozen_predictions() if fp is None else fp
    got = json.loads(json.dumps(V.compact(fp)))
    if RG.close(got, want):
        return ["the registered pricer does not reproduce the registered fingerprint: " + "; ".join(RG.close(got, want)[:3])]
    if [(c["model"], c["page"], c["n"], c["label"]) for c in fp["boards"]["A2"]["cells"]] != cells:
        return ["the pages' cells are not the registered cells"]
    return []


# ================================================================== the board, the gates and the finalisation

def gate_record(session: Path, which: str) -> dict | None:
    f = Path(session) / "gh200-driver" / V.GATE_RECORDS[which]
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except ValueError:
        return {"verdict": "MALFORMED", "why": f"{f.name} is not JSON"}


def validity(session: Path) -> dict:
    brd = RC.board(session)
    gates = {k: gate_record(session, k) for k in V.GATE_RECORDS}
    why = [w for w in ((brd.get("why") if not brd["valid"] else ""),
                       *(f"gate {k} reads {(v or {}).get('verdict')!r}" for k, v in gates.items()
                         if (v or {}).get("verdict") != "PASS")) if w]
    return {"board": brd, "gates": {k: (None if v is None else {"verdict": v.get("verdict"), "why": v.get("why")})
                                    for k, v in gates.items()},
            "valid": not why, "why": "; ".join(why), "sag record": gates["sag"]}


def finalise(sess: Path, sag_record: dict) -> dict:
    """The par and every cell's prediction of record, from the two f128 pages and the sag record only; Refused on a
    par that does not hold and on any blind page opened."""
    results = Path(sess) / "results"
    _GUARD["on"], _GUARD["hit"] = True, []
    _GUARD["roots"] = tuple(sorted({str(sess) + "/", str(Path(sess).absolute()) + "/", str(Path(sess).resolve()) + "/"}))
    try:
        try:
            fin = V.finalise_pars(results, sag_record)
        except (ValueError, LookupError, KeyError, ZeroDivisionError) as ex:
            raise Refused(f"finalisation: {ex}") from ex
        preds = V.predictions(tuple(fin["par"]))
    finally:
        _GUARD["on"] = False
    if _GUARD["hit"]:
        raise Refused(f"finalisation opened a blind page: {_GUARD['hit'][:3]}")
    return dict(fin, cells=preds, unpriced=[c["index"] for c in preds if c["ms"] is None])


# ================================================================== the reading

def per_view_bar(rows: list[dict]) -> dict:
    """The registered bars in one view: pooled and per model ('verdict' leaves, which rule 3 combines), the counts."""
    pooled = V.bar([r["e"] for r in rows])
    per = {m: V.bar([r["e"] for r in rows if r["model"] == m]) for m in V.MODELS}
    return {"pooled": pooled, "per_model": per, "cells scored": {m: sum(1 for r in rows if r["model"] == m)
                                                                  for m in V.MODELS}}


def describe(d: dict) -> dict:
    """A DESCRIPTIVE bar block: its 'verdict' keys renamed 'reading', so rule 3 leaves it alone."""
    d = json.loads(json.dumps(d, default=str))

    def walk(x):
        if isinstance(x, dict):
            if "verdict" in x and isinstance(x["verdict"], str):
                x["reading"] = x.pop("verdict")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(d)
    return d


def timed_view(TC, tree: Path, preds: dict, tv, files: dict) -> dict:
    """One view (ALL or CLEAN): each timed page through tcommon.timed_page and valid_cells, G3 with the registered
    per-stratum voids, the registered bars; B2's bar, round 5 and the splits beside, DESCRIPTIVE."""
    meas, use, dropped, in_view = {}, {}, {}, {}
    for m, G, _ns, name in V.page_specs():
        label = f"t128g{G}"
        f = files[f"{name}.json"]
        rep, why = TC.timed_page(tree, m, label, f["group_m"], tv, f)
        use[f"{m}-{label}"] = why
        if rep is None:
            continue
        sub = {k: v for k, v in preds.items() if k[0] == m and k[1] == name}
        c, d, _A = TC.valid_cells(rep, sub, m, name)
        if c:
            in_view.setdefault(m, []).append(name)
        dropped[f"{m}-{label}"] = d
        meas.update(c)
    g3 = TC.g3_rows(meas, preds, None)
    g3v = TC.C5.g3_rule(g3) if g3 else {"verdict": "NOT SCORED: no G3 row", "void_strata": None}
    vs = g3v.get("void_strata")
    rows, voided = [], []
    for k, t in meas.items():
        pr = preds[k]
        ns = sorted({x[2] for x in preds if x[:2] == k[:2]})
        r5 = pr.get("round5_ms (DESCRIPTIVE, every switch off)")
        row = {"cell": "/".join(map(str, k)), "model": k[0], "cls": pr["cls"], "G": pr["G"], "n": k[2], "label": k[3],
               "e": pr["ms"] / t - 1, "meas": t, "pred": pr["ms"], "e_round5": None if r5 is None else r5 / t - 1}
        if isinstance(vs, list) and str(ns.index(k[2])) in vs and k[3] != "balanced":
            voided.append(row["cell"])
            row["voided"] = True
        rows.append(row)
    kept = [r for r in rows if not r.get("voided")]
    out = {"pages": use, "dropped_rows": dropped, "G3": {k: v for k, v in g3v.items() if k != "rows"},
           "voided_cells": voided, "pages_in_view": in_view, "cells_measured": len(meas),
           "cells": [dict(r) for r in rows]}
    if vs == "ALL":
        out["BARS"] = {"verdict": "NOT SCORED: G3 FAILS (homogeneous), so every histogram cell is voided",
                       "cells scored": {m: 0 for m in V.MODELS}}
        out["DESCRIPTIVE"] = {"the bars on every cell": describe(per_view_bar(rows))}
        return out
    out["BARS"] = per_view_bar(kept)
    out["DESCRIPTIVE"] = {
        "B2's bar (2%, 5%, 1%) on the same cells": describe({"pooled": V.bar([r["e"] for r in kept], B.B2_BAR),
                                                            "per_model": {m: V.bar([r["e"] for r in kept if r["model"] == m],
                                                                                   B.B2_BAR) for m in V.MODELS}}),
        "the bars without G3's per-stratum voids": describe(per_view_bar(rows)) if voided else "no cell voided",
        "round 5 (every switch off), the same cells": V.stats([r["e_round5"] for r in kept if r["e_round5"] is not None]),
        "split: 128 G 1 cells": V.stats([r["e"] for r in kept if r["G"] == 1]),
        "split: 128 G 16 cells": V.stats([r["e"] for r in kept if r["G"] == 16]),
        "per model and class": {f"{m} {c}": V.stats([r["e"] for r in kept if r["model"] == m and r["cls"] == c])
                                for m in V.MODELS for c in ("128g1", "128g16")}}
    return out


def certify(res: dict, clean: dict, valid: bool) -> dict:
    """Per model: CERTIFIED when the combined (rule 3) bar HOLDS, the combined pooled bar HOLDS and the CLEAN view
    scores at least MIN_CELLS of its cells."""
    if not valid:
        return {m: "NOT CERTIFIED (not scored)" for m in V.MODELS}
    bars = res.get("BARS") or {}
    if "per_model" not in bars:
        return {m: f"NOT CERTIFIED ({bars.get('verdict')})" for m in V.MODELS}
    n = ((clean.get("BARS") or {}).get("cells scored") or {})
    pooled = bars["pooled"]["verdict"]
    out = {}
    for m in V.MODELS:
        v = bars["per_model"][m]["verdict"]
        k = int(n.get(m) or 0)
        if k < V.MIN_CELLS:
            out[m] = f"NOT CERTIFIED: BELOW THE MINIMUM ({k} cells scored in CLEAN, under {V.MIN_CELLS}; bar {v})"
        elif v == "HOLDS" and pooled == "HOLDS":
            out[m] = "CERTIFIED"
        else:
            out[m] = f"NOT CERTIFIED (own bar {v}; pooled bar {pooled})"
    return out


def b3_e1(res: dict, clean: dict, cert: dict) -> dict:
    """B3's E1-TILE conjunction on these models, DESCRIPTIVE: the pooled and every model's combined bar HOLD; its
    certification also asks 2 pages in CLEAN."""
    bars = res.get("BARS") or {}
    if "per_model" not in bars:
        return {"reading": bars.get("verdict")}
    ok = bars["pooled"]["verdict"] == "HOLDS" and all(bars["per_model"][m]["verdict"] == "HOLDS" for m in V.MODELS)
    pages = clean.get("pages_in_view") or {}
    return {"reading": "HOLDS" if ok else "FAILS",
            "certification (B3's, 2 pages in CLEAN)": {m: ("CERTIFIED" if ok and len(pages.get(m, [])) >= 2 and
                                                           cert[m] == "CERTIFIED" else "NOT CERTIFIED")
                                                       for m in V.MODELS}}


def commit_refusals(sess: Path, reg: dict, repo: Path | None = None) -> list[str]:
    """B3's freeze-to-commit rule (read_b3.check_tree; read_c2.run_commit and check_commit): the commit the rental ran
    is on its ledger ('driver start: commit <sha>', checked against commit.txt), and the working copy equals that
    commit for the registration, its plan and every file of the freeze (the gate registration's and this blind's)."""
    repo = V.REPO if repo is None else Path(repo)
    sha, why = RC.run_commit(Path(sess) / "session")
    if sha is None:
        return [f"the rental's commit is not on its ledger: {why}"]
    try:
        paths = sorted({V.REG_JSON, V.REG_TXT, V.PLAN, f"docs/registered/{V.GATE_NAME}.json"}
                       | set(reg["freeze"]["own"]) | set(V.gate_registration()["freeze"]["files"]))
    except (KeyError, TypeError) as ex:
        return [f"the registration's freeze block is malformed ({type(ex).__name__}: {ex})"]
    bad = RC.check_commit(repo, sha, paths)
    if bad:
        return [f"the working tree is not the rental's commit {sha[:12]}: " + "; ".join(f"{p} {w}" for p, w in bad[:5])]
    return []


def dirty_refusals(porcelain: str | None) -> list[str]:
    """The checkout scoring the rental must be clean (git status --porcelain empty); unknown is a refusal too."""
    if porcelain is None:
        return ["git status could not be read: the checkout's state is unknown"]
    if porcelain.strip():
        return [f"the checkout is dirty (git_dirty true): {porcelain.strip().splitlines()[:3]}"]
    return []


def score(sess: Path, reg: dict, check_commit: bool = True) -> dict:
    sess = Path(sess)
    results, session = sess / "results", sess / "session"
    if check_commit:
        why = commit_refusals(sess, reg)
        if why:
            raise Refused("; ".join(why))
    out = {"what": "model v7's prefill blind score (the working-model bar per model and pooled)", "registration": V.NAME,
           "session": str(sess), "bars": dict(V.BARS), "minimum": V.MIN_RULE, "verdict rule": V.CERT_RULE}
    val = validity(session)
    out["validity"] = {k: v for k, v in val.items() if k != "sag record"}
    if not val["valid"]:
        out["verdict"] = "NOT SCORED"
        out["why"] = val["why"]
        out["certified"] = {m: "NOT CERTIFIED (not scored)" for m in V.MODELS}
        return out
    fin = finalise(sess, val["sag record"])
    out["finalised"] = {k: v for k, v in fin.items() if k != "cells"}
    out["finalised"]["cells"] = [{k: c[k] for k in ("index", "model", "page", "n", "label", "ms")} for c in fin["cells"]]
    preds = {(c["model"], c["page"], c["n"], c["label"], "native"): c for c in fin["cells"] if c["ms"] is not None}
    C = V.R5.chain()
    d5 = C["d5"]
    TC, A2 = d5.TC, d5.A2
    files = reg["cells"]["pages"]["files"]
    which = "fallback" if str(val["board"].get("g_band_which") or "primary").startswith("fallback") else "primary"
    with A2.scoring(A2.Rules.tileA2(float(V.LOCK), which)):
        res = TC.C3.two_views(lambda fv, tv: timed_view(TC, results, preds, tv, files), repo=V.REPO)
    clean = (res.get("addendum") or {}).get("CLEAN") or {}
    cert = certify(res, clean, True)
    bars = res.get("BARS") or {}
    if "per_model" not in bars and str(bars.get("verdict")).startswith("NOT SCORED"):
        out["verdict"] = "NOT SCORED"           # the registered voids: G3's homogeneous fail, every histogram cell voided
        out["why"] = str(bars.get("verdict"))
    else:
        out["verdict"] = "SCORED"
    out["BARS (registered: ALL and CLEAN, rule 3)"] = res.get("BARS")
    out["cells scored in CLEAN"] = (clean.get("BARS") or {}).get("cells scored")
    out["certified"] = cert
    out["B3's E1-TILE (DESCRIPTIVE)"] = b3_e1(res, clean, cert)
    out["views"] = {k: v for k, v in res.items() if k != "BARS"}
    return out


def lines(s: dict) -> list[str]:
    L = ["MODEL v7 PREFILL BLIND (Granite-3.0-1B-A400M, Granite-3.0-3B-A800M, Qwen2-57B-A14B-tp8): the working-model "
         f"bar {s['bars']}", f"validity: {json.dumps(s['validity'], sort_keys=True, default=str)}"]
    if s["verdict"] == "NOT SCORED":
        return L + [f"NOT SCORED: {s['why']}", f"certified: {json.dumps(s['certified'])}"]
    f = s["finalised"]
    L.append(f"par (on-board F1): {f['par']}; unpriced {f['unpriced']}")
    b = s["BARS (registered: ALL and CLEAN, rule 3)"] or {}
    if "per_model" in b:
        L.append(f"pooled: {b['pooled']['verdict']} {json.dumps(b['pooled']['stats'])}")
        for m in V.MODELS:
            L.append(f"  {m}: {b['per_model'][m]['verdict']} {json.dumps(b['per_model'][m]['stats'])} -> {s['certified'][m]}")
    else:
        L.append(f"bars: {b.get('verdict')}")
    L.append(f"cells scored in CLEAN: {json.dumps(s['cells scored in CLEAN'])}")
    e1 = s["B3's E1-TILE (DESCRIPTIVE)"]
    L.append(f"B3's E1-TILE (DESCRIPTIVE): {json.dumps(e1)}")
    return L


def provenance(path: Path) -> dict:
    head = subprocess.run(["git", "-C", str(V.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(V.REPO), "status", "--porcelain"], capture_output=True, text=True)
    return {"registration_path": V.REG_JSON, "registration_sha256": V.sha256(path),
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    path = V.REPO / V.REG_JSON
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as ex:
        print(f"REFUSED: the registration cannot be read ({type(ex).__name__}: {ex})")
        return 2
    why = refusals(reg)
    st = subprocess.run(["git", "-C", str(V.REPO), "status", "--porcelain"], capture_output=True, text=True)
    why += dirty_refusals(st.stdout if st.returncode == 0 else None)
    if why:
        print("REFUSED: " + "; ".join(why))
        return 2
    try:
        s = dict(score(a.session, reg), provenance=provenance(path))
    except Refused as ex:
        print(f"REFUSED: {ex}")
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "prefill_blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (a.out / "prefill_blind.score.txt").write_text("\n".join(lines(s) + [
        f"registration sha256 {s['provenance']['registration_sha256']}, commit {s['provenance']['git_head']}"]) + "\n")
    print("\n".join(lines(s)))
    return 0 if s["verdict"] != "NOT SCORED" else 1


if __name__ == "__main__":
    sys.exit(main())
