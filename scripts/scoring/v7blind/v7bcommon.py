#!/usr/bin/env python
"""Model v7's ONE decode blind (registration docs/registered/2026-11-17-v7-decode-blind-gh200): the models, the cells,
the bars, the frozen predictions, the freeze, the plan and the rental, in one module for the registration builder and
the blind scorer. It imports the frozen decode modules and v7 and changes none of them.

THE RULES THIS FILE OWNS:
  models     the decode stage's three reserved models (Mixtral-8x22B, Qwen2-57B-A14B, Granite-3.0-1B-A400M): no decode
             page of any of them has been timed (`census`); config shapes only, synthetic weights, as the apparatus does
  cells      addendum 5's registered pages (docs/registered/2026-11-04-blind5-pages, scripts/decode_synth.py's frozen
             generators: the old decode blind's T grid 1 to 128, its routing classes (BALANCED, UNIFORM, UNIFORM-
             UNSHUFFLED, LADDER, LADDER-BOTTOM, MINCOVER), the NATIVE arm, GRR main pages and GR1 control pages) and
             addendum 5's units, unchanged; addendum 2's second-tile cells DESCRIPTIVE as before
  bars       the bars registered for this blind, verbatim: addendum 5 (2026-11-04) "new blind bars (owner, D4-1):
             rms 3% per model, worst cell 8%, |mean| 1.5%", per model and pooled, not tightened, carried unchanged by
             addendum 6 (2026-11-10); the certification rule is decodeB-blind's, which addenda 5 and 6 inherit
  predict    v7 (every switch on; D6FIX is the decode one, RR is not defined at decode) at the CAL2 refit under v7,
             exactly gate_v7.compute_decode's fit (D6FIX's constants from the CAL2 fit with v4d' on every naive cell),
             then finalise_decode.frozen and .predict on the registered pages: every cell priced before any page
  freeze     the v7 gate registration's freeze (every file at its registered sha256, v7's constants hash) plus this
             blind's own files; the scorer refuses on any difference
  draft      the registration is a DRAFT until the v7 gate is public and reads DECODE: GO; the plan carries a DRAFT
             line until then, and the launcher (scripts/vm_run_v7b.sh) refuses a DRAFT plan
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "decode"),
           str(REPO / "scripts" / "scoring" / "v7")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_timing_model_v7 as V7  # noqa: E402

_X = V7.decode_modules()
import d2common as D2  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402
import gate_v7 as GV  # noqa: E402

D3, V4DP, V4D = _X["D3"], _X["V4DP"], _X["V4D"]

DATE = "2026-11-17"
NAME = f"{DATE}-v7-decode-blind-gh200"
REG_JSON = f"docs/registered/{NAME}.json"
REG_TXT = f"docs/registered/{NAME}.txt"
GATE_NAME = GV.NAME
#: the committed output of gate_v7.py on the published pages, once the gate registration is public and the gate has
#: run (the analogue of gate_d6.out.txt); the registration may leave DRAFT only when it holds a "DECODE: GO" line
GATE_OUT = "scripts/scoring/v7/gate_v7.out.txt"
PLAN = "scripts/plans/rentalV7B-decode-blind-2026-11.plan"
DRIVER = "gh200_decode_session.sh"
LAUNCHER = "scripts/vm_run_v7b.sh"
PARENT_LAUNCHER = "scripts/vm_run_rr.sh"
MODELS = D5.NEW_BLIND
CTL = D5.NEW_CTL
PAGES = D5.PAGES5
DRAFT_LINE = ("# DRAFT: NOT LAUNCHABLE. This plan stays DRAFT until the v7 gate registration is public and gate_v7.py "
              "reads DECODE: GO; scripts/vm_run_v7b.sh refuses it.")

#: the bars registered for this blind (owner, D4-1; addendum 5, carried by addendum 6), verbatim
BARS = {k: D5.BARS_OWNER[k] for k in ("rms_max", "max_abs", "mean_max")}
BARS_SOURCE = ("the bars registered for these cells, verbatim: docs/registered/2026-11-04-decode-addendum5-gh200.txt line 9 "
               "'new blind bars (owner, D4-1): rms 3% per model, worst cell 8%, |mean| 1.5%' and line 103 'bars: the "
               "owner's (D4-1): rms 3%, worst 8%, |mean| 1.5%, per model and pooled, not tightened'; carried unchanged by "
               "docs/registered/2026-11-10-decode-addendum6-gh200.txt line 159 'bars: the owner's (D4-1): rms 3%, worst 8%, "
               "|mean| 1.5%, per model and pooled, not tightened' (d5common.BARS_OWNER)")
CERT_RULE = ("addenda 5 and 6 set the bars and inherit the rule: addendum 6 line 158 'rules (frozen otherwise): addendum "
             "5's: the power check first; E1-DEC pooled and per model without the second-tile DESCRIPTIVE cells; GD2 "
             "voids a model's GRR page; G3-D shuffle voids; G3-D label DESCRIPTIVE; SIGMA-D and the knee rows "
             "DESCRIPTIVE; the rivals by rental 7's rule; the scope line d3common.SCOPE'; addendum 5 line 102 'rules "
             "(frozen otherwise): decodeB-blind's and addenda 2 and 3's'; the certification itself is "
             "docs/registered/2026-10-14-decodeB-blind-gh200.txt's rule line: 'a model is CERTIFIED when its own bar "
             "and the pooled bar hold' (score_blind.score)")
#: the earlier blind registration whose pages, units and rules this blind reuses
ADD6 = "2026-11-10-decode-addendum6-gh200"
ADD6_BLOCK = "blind (owner, 2026-11-04, D4-1; addendum 5's pages and units, registered again in addendum 6)"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add6_blind() -> dict:
    return json.loads((REPO / "docs" / "registered" / f"{ADD6}.json").read_text())[ADD6_BLOCK]


def gate_registration() -> dict:
    return json.loads((REPO / "docs" / "registered" / f"{GATE_NAME}.json").read_text())


# ================================================================== the census: who has a decode page

#: every published or registered record that times a GEMM at a decode tile: the decode stage's pages and sessions
#: (d4common.no_decode_page) and the measurement rentals' registered pages, whose cells name a model or a shape
MEASUREMENT_PAGES = ("docs/registered/2026-11-06-pc-pages", "docs/registered/2026-11-10-pc2-pages",
                     "docs/registered/2026-11-11-rr-pages")
SHAPES = ("docs/registered/2026-11-06-pc-shapes.json", "docs/registered/2026-11-10-pc2-shapes.json",
          "docs/registered/2026-11-11-rr-shapes.json")


def tile_key(cfg: dict) -> str:
    return (f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']}w{cfg['num_warps']}"
            f"s{cfg['num_stages']}")


def census() -> dict:
    """Per model in moe.spec (toy left out): the decode pages that time it (COMPUTED from the tree): the decode stage's
    (no_decode_page), the measurement pages' cells at a decode tile naming it, and registered synthetic shapes at its
    (E, H, I, k) timed at a decode tile."""
    from moe.spec import MODEL_CONFIGS as MC
    meas: dict = {}
    shape_of = {}
    for f in SHAPES:
        for s in json.loads((REPO / f).read_text())["shapes"]:
            shape_of[s["name"]] = (s["E"], s["H"], s["F"], s["k"])
    for d in MEASUREMENT_PAGES:
        for f in sorted((REPO / d).glob("*.json")):
            for c in json.loads(f.read_text()).get("cells", []):
                cfg = c.get("config")
                if not cfg or tile_key(cfg) not in V7.DECODE_TILES:
                    continue
                meas.setdefault(c.get("shape"), set()).add(f"{f.relative_to(REPO)}:{c['label']}")
    out = {}
    for m, cfg in MC.items():
        if m == "toy":
            continue
        nd = D4.no_decode_page(m)
        key = (cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k)
        by_name = sorted(meas.get(m, ()))
        by_shape = sorted(x for s, v in meas.items() if s in shape_of and shape_of[s] == key for x in v)
        hits = nd["registered_pages"] + nd["published_dirs"] + by_name + by_shape
        out[m] = {"decode stage (no_decode_page)": nd["verdict"], "decode stage pages": len(nd["registered_pages"]),
                  "decode stage sessions": len(nd["published_dirs"]),
                  "measurement cells at a decode tile, by name": by_name,
                  "measurement cells at a decode tile, by shape": by_shape,
                  "verdict": "NO DECODE PAGE" if not hits else "HAS A DECODE PAGE"}
    return out


#: the reserved models' names as a scan pattern (d6_build/check_unseen.py's, unchanged)
PAT = re.compile(r"8x22b|qwen2-57b|qwen2_57b|granite-3\.0-1b|granite_3_0_1b|1b-a400m", re.I)
SCAN_FROM = "2026-11-04"


def unseen_scan() -> list[str]:
    """Every published session dated 2026-11-04 or later, scanned for the reserved models' names in file names and in
    text files (check_unseen's rule, extended past PC2 to every later session and to .md files)."""
    L = []
    pub = REPO / "results" / "published"
    later = sorted(p for p in pub.iterdir() if p.is_dir() and p.name[:10] >= SCAN_FROM and p.name[:4].isdigit())
    for s in later:
        names, text = [], {}
        for f in sorted(s.rglob("*")):
            rel = str(f.relative_to(s))
            if PAT.search(rel):
                names.append(rel)
            if f.is_file() and f.suffix in (".json", ".txt", ".log", ".env", ".plan", ".yaml", ".csv", ".md") \
                    and f.stat().st_size < 50e6:
                for mm in PAT.finditer(f.read_text(errors="replace")):
                    text.setdefault(rel, set()).add(mm.group(0).lower())
        L.append(f"{s.name}: file names naming a reserved model: {len(names)}; text files with a hit: {len(text)}"
                 + "".join(f"; {k}: {sorted(v)}" for k, v in sorted(text.items())[:8]))
    return L


# ================================================================== the predictions (v7, frozen before any page)

def decode_fit() -> dict:
    """gate_v7.compute_decode's fit, exactly, on the published CAL2 tree: the law and terms from the committed
    readings, D6FIX's constants (the CAL2 fit with v4d' on every naive cell) and the CAL2 score under v7 (its refit)."""
    import gate_d6 as G
    import score_cal_d3 as SC3
    cal2 = REPO / D4.CAL2_PUBLISHED / "results"
    with tempfile.TemporaryDirectory() as td:
        terms, law = G.readings_law(G.readings_dir_committed(Path(td) / "r"))
    tp, src = D4.tile_source()
    reg = DC.registration(DC.REPO, "cal")

    def cal(prm_fix=None, naive_v4dp=False):
        with V7.decode_scorers(terms=terms, law=law, prm_fix=prm_fix, switches=V7.V7, naive_v4dp=naive_v4dp):
            return SC3.score(cal2, reg, tp, src, settle=False)
    cf = cal(naive_v4dp=True)
    sel = cf["FIT"]["selected"]
    if sel is None:
        raise FN.Refused("D6FIX: the CAL2 fit with v4d' on every naive cell selects no rho form")
    f = cf["FIT"]["forms"][sel]
    fix = {"rho_form": sel, **{k: f[k] for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")}}
    c = cal(V4D.params_from(f, rho_form=sel, tile=tp))
    return {"terms": terms, "law": law, "tile": (tp, src), "fix": fix, "cal": c}


def prm_fix_of(fix: dict, tile_params: dict | None):
    t = DC.V4.Params(**tile_params) if tile_params else DC.V4.Params()
    return V4D.params_from(fix, rho_form=fix["rho_form"], tile=t)


@contextlib.contextmanager
def v7_blind(fz: dict, switches: V7.Switches = V7.V7):
    """Inside this block the frozen decode scorers price every cell by v7 at the frozen terms, law and D6FIX constants,
    and their model list and page paths are the blind's (d5common.blind5)."""
    import d6common as D6
    import gate_d6 as G
    m = fz["model"]
    terms = D5.V5D.Terms.from_dict(m["terms"])
    law = D6.V6D.Law(G._tuples(m["law"]))
    pf = prm_fix_of(m["D6FIX"], fz.get("tile_params"))
    with V7.decode_scorers(terms=terms, law=law, prm_fix=pf, switches=switches), D3.v4dp(V4DP.ALL), D5.blind5():
        yield


def predictions(fit: dict | None = None) -> dict:
    """The blind's frozen document: finalise_decode.frozen at the CAL2 refit under v7 (refuses unless the CAL stop rule
    reads GO), the model record, and finalise_decode.predict on every registered main cell under v7; the D6FIX-off
    rung beside (DESCRIPTIVE)."""
    fit = decode_fit() if fit is None else fit
    c = fit["cal"]
    fz = FN.frozen(c)
    fz["bars"] = dict(D5.BARS_OWNER)                    # the bars registered for this blind (D4-1), as gate_d6.frozen_at
    fz["model"] = {"name": "v7", "module": "scripts/r3_timing_model_v7.py", "switches": V7.V7.as_dict(),
                   "regime": "decode (v6d's law, D6FIX)", "fixes": V4DP.ALL.as_dict(), "terms": fit["terms"].as_dict(),
                   "law": fit["law"].as_dict(), "D6FIX": fit["fix"],
                   "constants_sha256": V7.constants_sha256()}
    fz["g3d_label"] = ("DESCRIPTIVE (addendum 3, owner 2026-11-03; kept by addenda 5 and 6): computed as registered and "
                       "reported per model; never voids a model's certification. G3-D shuffle voids as registered")
    with v7_blind(fz):
        rows = FN.predict(fz, models=MODELS)
        desc = sorted("/".join(k) for k in D2.excluded_keys(MODELS))
    with v7_blind(fz, V7.OFF):
        off = FN.predict(fz, models=MODELS)
    moved = [f"{r['model']}/{r['label']}" for r, q in zip(rows, off, strict=True) if r["ms"]["v4d"] != q["ms"]["v4d"]]
    stats = (c.get("E1_CAL_IN_SAMPLE") or {}).get("stats")
    return {"frozen": fz, "cells": rows, "not_priced": sum(1 for r in rows if r["ms"]["v4d"] is None),
            "descriptive_cells": desc, "scope": D3.SCOPE,
            "CAL2 refit under v7": {"verdict": (c.get("STOP") or {}).get("verdict"), "E1_CAL_IN_SAMPLE": stats,
                                     "rho": fz["rho_form"], "constants": fz["constants"]},
            "D6FIX rung (DESCRIPTIVE)": {"cells moved by D6FIX": moved,
                                         "reading": (f"{len(moved)} of {len(rows)} cells move: every naive main cell of "
                                                     "the three models takes the law branch (addendum 6: 2, 7 and 2), "
                                                     "so v7's blind prices equal v6d's at this CAL2 fit"
                                                     if not moved else f"{len(moved)} cells move")},
            "keys": "ms['v4d'] holds v7's time (the frozen scorers read that key); every rival priced by v7 with the "
                    "rival's own change (rho other, NAIVE-NODEDUP, eager), roofline, line and OLS as frozen"}


# ================================================================== the freeze

#: this blind's own files: its pages, the driver and VM scripts the plan runs, the launcher, its code, and the
#: earlier registrations and committed files its bars, units and fit are quoted from
OWN_FILES = (
    *(f"{PAGES}/{m}-main.json" for m in MODELS), *(f"{PAGES}/{m}-ctl.json" for m in CTL),
    "scripts/decode_synth.py", f"scripts/{DRIVER}", "scripts/setup_vm.sh", "scripts/vm_results_push.sh",
    "scripts/locked_r3.py", LAUNCHER, PARENT_LAUNCHER,
    "scripts/scoring/v7blind/v7bcommon.py", "scripts/scoring/v7blind/register_v7_blind.py",
    "scripts/scoring/v7blind/score_blind_v7.py",
    "scripts/scoring/decode/cal2.score.json", "scripts/scoring/decode/decodeB.finalised.json",
    "scripts/scoring/decode/blind.score.txt", "scripts/scoring/decode/regdecode.py",
    "docs/registered/2026-10-14-decodeB-blind-gh200.json", "docs/registered/2026-10-14-decodeB-blind-gh200.txt",
    "docs/registered/2026-10-28-decode-addendum2-gh200.json", "docs/registered/2026-10-29-decode-addendum3-gh200.json",
    "docs/registered/2026-10-29-decode-addendum3-gh200.txt", "docs/registered/2026-11-04-decode-addendum5-gh200.json",
    "docs/registered/2026-11-04-decode-addendum5-gh200.txt",
    f"docs/registered/{ADD6}.json", f"docs/registered/{ADD6}.txt",
    f"docs/registered/{GATE_NAME}.json", f"docs/registered/{GATE_NAME}.txt",
)


def freeze() -> dict:
    g = gate_registration()["freeze"]
    own = {p: sha256(REPO / p) for p in OWN_FILES if p not in g["files"]}
    return {"statement": ("the scorer (score_blind_v7.py) refuses unless every file of the v7 gate registration's "
                          "freeze has its sha256 there and v7's constants hash is the one registered there (v7 as the "
                          "gate froze it), and every file under 'own' has its sha256 here; the plan's sha256 is "
                          "computed by the registration builder from the same document"),
            "gate registration": {"name": GATE_NAME, "json sha256": sha256(REPO / f"docs/registered/{GATE_NAME}.json"),
                                  "files": len(g["files"]), "constants_sha256": g["constants_sha256"]},
            "own": own}


def verify_freeze(reg: dict, root: Path = REPO) -> list[str]:
    """Every difference from the registered freeze (empty: frozen)."""
    fz = reg["freeze"]
    bad = []
    gp = Path(root) / "docs" / "registered" / f"{GATE_NAME}.json"
    if not gp.is_file() or sha256(gp) != fz["gate registration"]["json sha256"]:
        bad.append(f"{GATE_NAME}.json is not the registered bytes")
    else:
        bad += GV.verify_freeze(json.loads(gp.read_text()), root=root)
    for p, want in fz["own"].items():
        f = Path(root) / p
        if not f.is_file():
            bad.append(f"{p}: missing")
        elif sha256(f) != want:
            bad.append(f"{p}: sha256 {sha256(f)[:12]} is not the registered {want[:12]}")
    if V7.constants_sha256() != fz["gate registration"]["constants_sha256"]:
        bad.append("v7's constants hash moved")
    return bad


# ================================================================== the plan and the rental

def plan_header(fz: dict | None, draft: bool) -> list[str]:
    hdr = ["# MODEL v7'S ONE DECODE BLIND RENTAL: v7 (every switch on) frozen at the CAL2 refit, on the decode stage's",
           "# three reserved models no decode page has timed (addendum 5's pages and units).",
           f"# Registration {NAME}. GENERATED by scripts/scoring/v7blind/register_v7_blind.py; do not edit by hand.",
           f"# Run by the pinned decode driver: bash {LAUNCHER} start ... --plan {PLAN} --driver {DRIVER}",
           "# Drop order, last first: gr1 (the two GR1 control pages); every main page is nodrop.",
           f"# LAUNCH PRECONDITION: the v7 gate registration ({GATE_NAME}) is public and scripts/scoring/v7/gate_v7.py",
           "# reads DECODE: GO; the registration is committed for real (not DRAFT). One blind, pass or fail; then the stage",
           "# stops and is reported either way (owner, 2026-11-17)."]
    if fz:
        c = fz["constants"]
        us = lambda v: "NOT FITTED" if v is None else f"{v * 1e3:.2f} us"  # noqa: E731
        hdr.append(f"# Frozen: T0_GR {us(c['T0_GR_ms'])}, s_naive {us(c['s_naive_GR_ms'])}, s_block {us(c['s_block_GR_ms'])}, "
                   f"rho {fz['rho_form']}, bars rms {BARS['rms_max']:.1%}, worst {BARS['max_abs']:.1%}, "
                   f"mean {BARS['mean_max']:.2%} (owner, D4-1).")
    if draft:
        hdr.append(DRAFT_LINE)
    return hdr


def plan_text(fz: dict | None, draft: bool) -> str:
    return "\n".join(plan_header(fz, draft) + D5.blind_plan_units()) + "\n"


def rental() -> dict:
    """addendum 5's rule on its units (d2common.rental): units x 1/0.84 plus 25 min boot and setup; deadline wall + 15
    rounded to 5; hard cap 15 later."""
    return D5.blind_rental()


def gate_reads_decode_go(root: Path = REPO) -> bool:
    p = Path(root) / GATE_OUT
    return p.is_file() and any(ln.startswith("DECODE: GO") for ln in p.read_text().splitlines())
