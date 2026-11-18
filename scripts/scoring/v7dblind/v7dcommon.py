#!/usr/bin/env python
"""Model v7's decode DOMAIN blind (registration docs/registered/2026-11-18-v7-domain-blind-gh200): the models, the cells,
the frozen predictions, the domain D read from the map's committed readout, the in-domain labels, the bars and the
minimum, the freeze, the plan and the rental, in one module for the registration builder (register_v7d_blind.py) and
the scorer (score_blind_v7d.py). It imports the frozen decode modules, v7, the v7 blind's fit and the map's rule and
changes none of them.

THE RULES THIS FILE OWNS:
  models     DeepSeek-V3, DeepSeek-V3-tp4, DeepSeek-V3-tp8 and Qwen2-57B-A14B-tp8 (moe/spec.py, unchanged): no decode
             page of any has been timed (`census`, computed over every registered page list and every published
             session, the v7 blind's and the map's included); config shapes only, synthetic weights
  cells      the v7 blind's recipe (addendum 5's): scripts/decode_synth.py's frozen generators, unchanged (the T grid 1
             to 128, the routing classes, shuffle seed 20261014, the NATIVE arm), one GRR main page per model and GR1
             control pages on two models (drop group gr1); addendum 2's second-tile cells DESCRIPTIVE as before
  predict    v7 (every switch on) at the CAL2 refit, exactly the v7 blind's (v7bcommon.decode_fit, finalise_decode.frozen
             and .predict); every cell priced before any page
  domain     D is a REQUIRED INPUT read mechanically from the map's committed readout (DOMAIN, written by read_map.py on
             the published map session): its registration and readout hashes verified, the map's freeze verified, D
             recomputed from the readout's own rows by read_map.domain and required equal; a cell is IN-DOMAIN when its
             path has an X_p, T_v7 >= X_p, W_e >= S_p when the slab guard set one, and every active expert fits one
             M-tile; every other cell is OUT, reported DESCRIPTIVE
  score      E1-DEC on the IN-DOMAIN cells only, per model and pooled, at the old bars (owner, D4-1: rms 3%, worst 8%,
             |mean| 1.5%), decodeB-blind's rule (a model CERTIFIED when its own bar and the pooled bar hold), plus the
             MINIMUM (MIN_RULE), registered before D is known
  draft      the registration is a DRAFT until the map's readout is committed and verifies; the plan carries a DRAFT
             line until then, and the launcher (scripts/vm_run_v7d.sh) refuses a DRAFT plan
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "decode"),
           str(REPO / "scripts" / "scoring" / "v7"), str(REPO / "scripts" / "scoring" / "v7blind"),
           str(REPO / "scripts" / "scoring" / "v7domain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_shapes as MS  # noqa: E402
import read_map as RM  # noqa: E402
import v7bcommon as VB  # noqa: E402
import v7mcommon as VM  # noqa: E402

V7, D2, D3, D4, D5, DC, FN, GV = VB.V7, VB.D2, VB.D3, VB.D4, VB.D5, VB.DC, VB.FN, VB.GV
RD = VB._X["RD"]
V4DP, V4D = VB.V4DP, VB.V4D

DATE = "2026-11-18"
NAME = f"{DATE}-v7-domain-blind-gh200"
REG_JSON = f"docs/registered/{NAME}.json"
REG_TXT = f"docs/registered/{NAME}.txt"
PAGES = f"docs/registered/{DATE}-v7-domain-blind-pages"
PLAN = "scripts/plans/rentalV7D-domain-blind-2026-11.plan"
DRIVER = "gh200_decode_session.sh"
LAUNCHER = "scripts/vm_run_v7d.sh"
PARENT_LAUNCHER = "scripts/vm_run_v7b.sh"
GATE_NAME = GV.NAME
BLIND_NAME = VB.NAME
MAP_NAME = VM.NAME
#: the map registration's bytes at 556029e1 (public before any map page): the readout must name exactly this file
MAP_REG_SHA256 = "bf33f4cc100b86f39179cc8dec3188e6b751f62f941c1986f1c5f3630913201b"
MAP_REG_COMMIT = "556029e1"
#: the map's readout: read_map.py's domain.json on the published map session, committed at this path
DOMAIN = "scripts/scoring/v7domain/map.domain.json"
MODELS = ("deepseek-v3", "deepseek-v3-tp4", "deepseek-v3-tp8", "qwen2-57b-a14b-tp8")
#: GR1 control pages (GD2) on the two models whose GRR page rotates the most copies (R 5 and 3; R 1 or 2 elsewhere,
#: where GRR and GR1 time nearly the same thing); drop group gr1, dropped first, last listed first
CTL = ("qwen2-57b-a14b-tp8", "deepseek-v3-tp8")
LIST_RATE = VM.LIST_RATE
PATHS = VM.PATHS
MIN_IN_DOMAIN = 10
MIN_RULE = (
    "MINIMUM (registered 2026-11-18 with this DRAFT, before D is known): a model is certifiable only when at least 10 "
    "of its cells are scored IN-DOMAIN (valid NATIVE main cells after the power check, the page checks, row_drop, "
    "GD2's and G3-D shuffle's voids, second-tile cells excluded, and labelled IN-DOMAIN). A model with fewer is "
    "reported with its in-domain statistics, labelled BELOW THE MINIMUM and NOT CERTIFIED whatever they read; its "
    "in-domain cells still count in the pooled bar. The pooled bar is over every scored in-domain cell of every model "
    "not voided. --final refuses when D is empty (the map registration: if D is empty, no domain blind runs) or when no "
    "model has 10 registered in-domain cells (nothing could be certified)")
DRAFT_LINE = ("# DRAFT: NOT LAUNCHABLE. This plan stays DRAFT until the map's readout (scripts/scoring/v7domain/"
              "map.domain.json) is committed and register_v7d_blind.py --write --final accepts it; scripts/vm_run_v7d.sh "
              "refuses it.")
BARS = dict(VB.BARS)
BARS_SOURCE = VB.BARS_SOURCE
CERT_RULE = VB.CERT_RULE


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


# ================================================================== the pages

def page_path(model: str, kind: str) -> str:
    return f"{PAGES}/{model}-{kind}.json"


PAGE_KEYS = tuple((m, "main") for m in MODELS) + tuple((m, "ctl") for m in CTL)


def page_doc(model: str, kind: str) -> dict:
    """decode_synth.py's frozen generator (main, ctl), unchanged, relabelled (d5common.page5's form)."""
    import decode_synth as SY
    d = SY.page(model, kind)
    d["label"] = (f"{d['label']}; the v7 domain blind (registration {NAME}), generated by scripts/decode_synth.py's "
                  "frozen rules for a model no decode page has timed")
    return d


def page_texts() -> dict[str, str]:
    """{page path: text} for the six pages."""
    return {page_path(m, k): json.dumps(page_doc(m, k), indent=1) + "\n" for m, k in PAGE_KEYS}


def write_pages(root: Path = REPO) -> None:
    for p, t in page_texts().items():
        (Path(root) / p).parent.mkdir(parents=True, exist_ok=True)
        (Path(root) / p).write_text(t)


def reg_file(model: str, kind: str, root: Path = REPO) -> dict:
    """dcommon.decode_page's registered file record (d5common.reg_file5's form)."""
    p = Path(root) / page_path(model, kind)
    doc = json.loads(p.read_text())
    return {"path": page_path(model, kind), "sha256": sha256(p), "shuffle_seed": doc["shuffle_seed"],
            "decl_copies": doc["decl_copies"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]), "role": c["role"],
                       "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


@contextlib.contextmanager
def blind7d():
    """dcommon.BLIND_MODELS the four models and dcommon.page_path their pages (d5common.blind5's form)."""
    models, path = DC.BLIND_MODELS, DC.page_path

    def pp(model, kind):
        return page_path(model, kind) if model in MODELS else path(model, kind)
    DC.BLIND_MODELS, DC.page_path = MODELS, pp
    try:
        yield
    finally:
        DC.BLIND_MODELS, DC.page_path = models, path


@contextlib.contextmanager
def v7_domain_blind(fz: dict, switches=None):
    """v7bcommon.v7_blind with this blind's models and pages in place of addendum 5's."""
    import d6common as D6
    import gate_d6 as G
    m = fz["model"]
    terms = D5.V5D.Terms.from_dict(m["terms"])
    law = D6.V6D.Law(G._tuples(m["law"]))
    pf = VB.prm_fix_of(m["D6FIX"], fz.get("tile_params"))
    with V7.decode_scorers(terms=terms, law=law, prm_fix=pf, switches=switches or V7.V7), D3.v4dp(V4DP.ALL), blind7d():
        yield


# ================================================================== the units, the plan and the rental

def _est(model: str, kind: str) -> int:
    from moe.spec import MODEL_CONFIGS
    page = RD.page_from(page_doc(model, kind), MODEL_CONFIGS[model], model)
    return math.ceil(RD.estimated_seconds(page, 9) / 60)


def units() -> list[dict]:
    """decodeB-blind's unit form: dm (GRR, nodrop) per model, then dc (GR1, drop group gr1)."""
    out = [{"label": "dm", "model": m, "page": f"{m}-main.json", "step": "r3decode", "inst": "GRR", "drop": "nodrop",
            "est_min": _est(m, "main")} for m in MODELS]
    out += [{"label": "dc", "model": m, "page": f"{m}-ctl.json", "step": "r3decode", "inst": "GR1", "drop": "gr1",
             "est_min": _est(m, "ctl")} for m in CTL]
    return out


def unit_lines() -> list[str]:
    tup = tuple((u["label"], u["model"], u["page"].rsplit("-", 1)[1][:-5], u["step"], u["inst"], u["drop"], u["est_min"])
                for u in units())
    with blind7d():
        txt = FN.plan_text(tup, [], census=False)
    return [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]


def rental() -> dict:
    """d2common.rental's rule on the units (prelude and calibrate included): units x 1/0.84 plus 25 min boot and setup;
    deadline launch + wall + 15 rounded to 5; hard cap 15 later; cost at the inherited RATE and at the list price.
    Beside it (DESCRIPTIVE), the same rule without the gr1 pages, which the driver drops first."""
    pre = sum(int(x.split("est=")[1].split()[0]) for x in D4.PRELUDE)
    us = units()
    r = D2.rental(pre + sum(u["est_min"] for u in us))
    r.update({"list_rate": f"gpu_1x_gh200 at ${LIST_RATE}/h (Lambda's list price, 2026-11-18)",
              "cost_usd_at_list": round(r["wall_min"] / 60 * LIST_RATE, 2),
              "cost_at_hard_cap_usd_at_list": round(r["hard_cap_after_launch_min"] / 60 * LIST_RATE, 2)})
    lean = D2.rental(pre + sum(u["est_min"] for u in us if u["drop"] == "nodrop"))
    r["without the gr1 pages (DESCRIPTIVE)"] = {
        "units_min": lean["units_min"], "wall_min": lean["wall_min"],
        "cost_usd_at_list": round(lean["wall_min"] / 60 * LIST_RATE, 2)}
    r["drop order"] = ("pre-declared, last first: the GR1 control pages (gr1: deepseek-v3-tp8's, then "
                       "qwen2-57b-a14b-tp8's); every main page is nodrop; no model is dropped, and no cell is dropped "
                       "from a page (second-tile cells stay on their pages: they cost about 1.6 min a model)")
    return r


def plan_header(fz: dict | None, draft: bool) -> list[str]:
    hdr = ["# MODEL v7'S DECODE DOMAIN BLIND RENTAL: v7 (every switch on) frozen at the CAL2 refit, on four models no",
           "# decode page has timed (DeepSeek-V3, -tp4, -tp8, Qwen2-57B-A14B-tp8); scored only inside the map's domain D.",
           f"# Registration {NAME}. GENERATED by scripts/scoring/v7dblind/register_v7d_blind.py; do not edit by hand.",
           f"# Run by the pinned decode driver: bash {LAUNCHER} start ... --plan {PLAN} --driver {DRIVER}",
           "# Drop order, last first: gr1 (the two GR1 control pages); every main page is nodrop.",
           f"# LAUNCH PRECONDITION: the map's readout ({DOMAIN}) is committed and verifies, and this",
           "# registration is committed for real (not DRAFT). One blind, pass or fail; then the stage stops and is",
           "# reported either way (owner, 2026-11-18)."]
    if fz:
        c = fz["constants"]
        us = lambda v: "NOT FITTED" if v is None else f"{v * 1e3:.2f} us"  # noqa: E731
        hdr.append(f"# Frozen: T0_GR {us(c['T0_GR_ms'])}, s_naive {us(c['s_naive_GR_ms'])}, s_block {us(c['s_block_GR_ms'])}, "
                   f"rho {fz['rho_form']}, bars rms {BARS['rms_max']:.1%}, worst {BARS['max_abs']:.1%}, "
                   f"mean {BARS['mean_max']:.2%} (owner, D4-1), in-domain cells only.")
    if draft:
        hdr.append(DRAFT_LINE)
    return hdr


def plan_text(fz: dict | None, draft: bool) -> str:
    return "\n".join(plan_header(fz, draft) + unit_lines()) + "\n"


# ================================================================== the census: who has a decode page

#: these models' names as a scan pattern (file names and text), check_unseen's rule
PAT = re.compile(r"deepseek[-_]v3|qwen2[-_]57b[-_]a14b[-_]tp8", re.I)
SCAN_FROM = VB.SCAN_FROM


def names_model(text: str, model: str) -> bool:
    """Whether the text names the model exactly ('deepseek-v3' not as the prefix of 'deepseek-v3-tp4'; - or _)."""
    t = text.lower().replace("_", "-")
    tail = "" if "-tp" in model else r"(?!-tp\d)"
    return re.search(r"(?<![0-9a-z])" + re.escape(model) + tail + r"(?![0-9a-z])", t) is not None


def shape_of(cfg) -> tuple:
    return (cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k)


def published_sessions() -> list[str]:
    pub = REPO / "results" / "published"
    return sorted(p.name for p in pub.iterdir() if p.is_dir())


def map_vs_models() -> dict:
    """Each model's (E, H, I, k) against the 14 map shapes' (COMPUTED); the nearest map shape by E and k."""
    from moe.spec import MODEL_CONFIGS as MC
    out = {}
    for m in MODELS:
        key = shape_of(MC[m])
        same = [n for n, _f, E, k, H, I in MS.SHAPES if (E, H, I, k) == key]  # noqa: E741
        near = [f"{n} (H {H}, I {I})" for n, _f, E, k, H, I in MS.SHAPES if (E, k) == (key[0], key[3])]  # noqa: E741
        out[m] = {"E, H, I, k": list(key), "map shapes with this (E, H, I, k)": same,
                  "map shapes with this E and k (other H or I)": near}
    return out


def report_tiles(obj, out=None, bm=None) -> set:
    """Every tile (v7bcommon.tile_key) named by a config dict anywhere in a report; a config without BLOCK_SIZE_M
    takes the report's block_m (the R3 gap pages' form)."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        bm = obj.get("block_m", bm)
        if all(k in obj for k in ("BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")):
            m = obj.get("BLOCK_SIZE_M", bm)
            if m is not None:
                out.add(VB.tile_key(dict(obj, BLOCK_SIZE_M=m)))
        for v in obj.values():
            report_tiles(v, out, bm)
    elif isinstance(obj, list):
        for v in obj:
            report_tiles(v, out, bm)
    return out


def gaps_kind(g: Path) -> dict:
    """A gaps directory's reports: decode-page kind or a decode tile makes it a decode page."""
    kinds, tiles = set(), set()
    for f in sorted(g.rglob("report.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        kinds.add(str(r.get("kind")))
        report_tiles(r, tiles)
    dec = "decode-page" in kinds or bool(tiles & set(V7.DECODE_TILES))
    return {"reports": len(list(g.rglob("report.json"))), "kinds": sorted(kinds), "tiles": sorted(tiles), "decode": dec}


def merged_rows(sessions: list[str]) -> dict:
    """Per model, every published merged.csv row naming it: {session: {gpu, impls, num_tokens}} (earlier
    fused_experts and grouped-GEMM timings; not this apparatus' decode pages)."""
    import csv
    out: dict = {m: {} for m in MODELS}
    pub = REPO / "results" / "published"
    for s in sessions:
        f = pub / s / "merged.csv"
        if not f.is_file():
            continue
        with f.open() as fh:
            for row in csv.DictReader(fh):
                m = row.get("model")
                if m in out:
                    x = out[m].setdefault(s, {"gpu": set(), "impls": set(), "num_tokens": set()})
                    x["gpu"].add(row.get("gpu_name"))
                    x["impls"].add(row.get("impl"))
                    x["num_tokens"].add(int(float(row.get("num_tokens") or 0)))
    return {m: {s: {k: sorted(v) for k, v in x.items()} for s, x in d.items()} for m, d in out.items()}


def census(sessions: list[str] | None = None) -> dict:
    """Per model (COMPUTED) over the given published sessions (default: all) and every registered page list:
    (a) registered page lists (docs/registered/*pages, this blind's own left out) whose page names the model or whose
    model has its (E, H, I, k) (the map's synthetic shapes included); (b) results gaps directories for the model or a
    shape equal to its (E, H, I, k), each classified by its reports (a decode page: kind decode-page or a decode tile);
    (c) v7bcommon.census's measurement cells at a decode tile, by name and by shape; (d) every tile named by a JSON
    file whose path names the model. A decode page in (a), (b) or (c), or a decode tile in (d), makes HAS A DECODE
    PAGE. Reported beside, not pages: file names naming the model, per session; merged.csv rows
    naming it (earlier fused_experts timings, other cards); text hits in sessions dated 2026-11-04 or later."""
    from moe.spec import MODEL_CONFIGS as MC
    sessions = published_sessions() if sessions is None else list(sessions)
    keys = {m: shape_of(MC[m]) for m in MODELS}
    shapes = {n: shape_of(c) for n, c in MC.items()}
    shapes.update({n: (E, H, I, k) for n, _f, E, k, H, I in MS.SHAPES})  # noqa: E741
    regs: dict = {m: [] for m in MODELS}
    for d in sorted((REPO / "docs" / "registered").glob("*pages")):
        if d.name == Path(PAGES).name:
            continue
        for f in sorted(d.glob("*.json")):
            try:
                pm = json.loads(f.read_text()).get("model")
            except (ValueError, AttributeError):
                pm = None
            for m in MODELS:
                if pm == m or (pm in shapes and shapes[pm] == keys[m]):
                    regs[m].append(str(f.relative_to(REPO)))
    gaps: dict = {m: {} for m in MODELS}
    names: dict = {m: {} for m in MODELS}
    tiles: dict = {m: set() for m in MODELS}
    text: dict = {}
    pub = REPO / "results" / "published"
    allnames = sorted(shapes, key=len, reverse=True)
    for s in sessions:
        root = pub / s
        if not root.is_dir():
            continue
        for g in sorted(root.rglob("gaps-*")):
            if not g.is_dir():
                continue
            rest = g.name.split("-", 2)[2] if g.name.count("-") >= 2 else ""
            gm = next((n for n in allnames if rest.startswith(n + "-") and "-" not in rest[len(n) + 1:]), None)
            for m in MODELS:
                if gm == m or (gm is not None and shapes.get(gm) == keys[m]):
                    gaps[m][f"{s}/{g.relative_to(root)}"] = gaps_kind(g)
        for f in sorted(root.rglob("*")):
            rel = str(f.relative_to(root))
            for m in MODELS:
                if names_model(rel, m):
                    names[m][s] = names[m].get(s, 0) + 1
                    if f.suffix == ".json" and f.is_file() and f.stat().st_size < 50e6:
                        try:
                            report_tiles(json.loads(f.read_text()), tiles[m])
                        except ValueError:
                            pass
            if s[:10] >= SCAN_FROM and f.is_file() and f.suffix in (
                    ".json", ".txt", ".log", ".env", ".plan", ".yaml", ".csv", ".md") and f.stat().st_size < 50e6:
                for mm in PAT.finditer(f.read_text(errors="replace")):
                    text.setdefault(f"{s}/{rel}", set()).add(mm.group(0).lower())
    vb = VB.census()
    merged = merged_rows(sessions)
    out = {}
    for m in MODELS:
        x = vb[m]
        meas = x["measurement cells at a decode tile, by name"] + x["measurement cells at a decode tile, by shape"]
        dec_gaps = sorted(k for k, v in gaps[m].items() if v["decode"])
        dec_tiles = sorted(tiles[m] & set(V7.DECODE_TILES))
        hits = regs[m] + dec_gaps + meas + dec_tiles
        out[m] = {"registered page lists naming it or its shape": regs[m],
                  "published gaps directories for it or its shape": gaps[m],
                  "measurement cells at a decode tile (v7bcommon.census)": meas,
                  "tiles named by the JSON files whose names name it": sorted(tiles[m]),
                  "decode tiles among them": dec_tiles,
                  "verdict": "NO DECODE PAGE" if not hits else "HAS A DECODE PAGE",
                  "not pages: published file names naming it, per session": names[m],
                  "not pages: merged.csv rows naming it (earlier timings, other cards)": merged[m]}
    return {"sessions scanned": sessions, "per model": out,
            "text hits (sessions dated 2026-11-04 or later; not pages)": {k: sorted(v) for k, v in sorted(text.items())}}


# ================================================================== the predictions (v7, frozen before any page)

def predictions(fit: dict | None = None) -> dict:
    """The blind's frozen document: v7bcommon.predictions' fit and frozen record, exactly (finalise_decode.frozen at the
    CAL2 refit under v7, the D4-1 bars, the model record), and finalise_decode.predict on every main cell of the four
    models under v7. Each cell carries its geometry for the domain test (path, W_e, second tile, T_v7 in us at the map's
    6-decimal rounding) and V7.price_decode_cell's independent price (the map's pricer, v7mcommon.price's call), which must
    agree to 1e-9 relative (each cell with its own shuffle flag)."""
    from moe.spec import MODEL_CONFIGS as MC
    fit = VB.decode_fit() if fit is None else fit
    c = fit["cal"]
    fz = FN.frozen(c)
    fz["bars"] = dict(D5.BARS_OWNER)
    fz["model"] = {"name": "v7", "module": "scripts/r3_timing_model_v7.py", "switches": V7.V7.as_dict(),
                   "regime": "decode (v6d's law, D6FIX)", "fixes": V4DP.ALL.as_dict(), "terms": fit["terms"].as_dict(),
                   "law": fit["law"].as_dict(), "D6FIX": fit["fix"], "constants_sha256": V7.constants_sha256()}
    fz["g3d_label"] = ("DESCRIPTIVE (addendum 3, owner 2026-11-03; kept by addenda 5 and 6): computed as registered and "
                       "reported per model; never voids a model's certification. G3-D shuffle voids as registered")
    P = {"terms": fit["terms"], "law": fit["law"], "prm": FN.params_of(fz),
         "prm_fix": VB.prm_fix_of(fit["fix"], fz.get("tile_params"))}
    with v7_domain_blind(fz):
        rows = FN.predict(fz, models=MODELS)
        desc = sorted("/".join(k) for k in D2.excluded_keys(MODELS))
    with v7_domain_blind(fz, V7.OFF):
        off = FN.predict(fz, models=MODELS)
    moved = [f"{r['model']}/{r['label']}" for r, q in zip(rows, off, strict=True) if r["ms"]["v4d"] != q["ms"]["v4d"]]
    docs = {m: {x["label"]: x for x in page_doc(m, "main")["cells"]} for m in MODELS}
    worst = 0.0
    for r in rows:
        cell = docs[r["model"]][r["label"]]
        cfg = MC[r["model"]]
        g = RD.geometry(cfg, int(cell["T"]), list(cell["counts"]))
        ms = r["ms"]["v4d"]
        q = V7.price_decode_cell(r["model"], {"T": int(cell["T"]), "counts": list(cell["counts"]),
                                               "shuffle": cell.get("shuffle", True), "label": cell["label"]},
                                 P["prm"], terms=P["terms"], law=P["law"], prm_fix=P["prm_fix"])
        if ms is None or q is None or q.get("ms") is None:
            raise FN.Refused(f"v7 does not price {r['model']}/{r['label']}")
        worst = max(worst, abs(q["ms"] / ms - 1))
        st = D2.second_tile(r["model"], cell)
        r["geometry"] = {"path": g.path, "tile": g.tile, "BLOCK_M": g.bm, "A": g.A,
                         "W_e_MB": r6(RD.expert_bytes(cfg) / 1e6), "waves": r6(max(g.live_ctas) / VM.SLOTS[g.tile]),
                         "second_tile": st is not None, "T_v7_us": r6(ms * 1e3)}
    if worst > 1e-9:
        raise FN.Refused(f"finalise_decode.predict and V7.price_decode_cell disagree by {worst:.3g} relative")
    stats = (c.get("E1_CAL_IN_SAMPLE") or {}).get("stats")
    by = {}
    for r in rows:
        g = r["geometry"]
        b = by.setdefault(r["model"], {"cells": 0, "second tile (DESCRIPTIVE)": 0, "paths": {}})
        b["cells"] += 1
        b["second tile (DESCRIPTIVE)"] += g["second_tile"]
        if not g["second_tile"]:
            b["paths"][g["path"]] = b["paths"].get(g["path"], 0) + 1
    for m, b in by.items():
        t = [r["geometry"]["T_v7_us"] for r in rows if r["model"] == m]
        b["T_v7 range us"] = [min(t), max(t)]
    return {"frozen": fz, "cells": rows, "not_priced": sum(1 for r in rows if r["ms"]["v4d"] is None),
            "descriptive_cells": desc, "scope": D3.SCOPE, "per model": by,
            "pricers agree": f"finalise_decode.predict and V7.price_decode_cell within {worst:.2g} relative on every cell",
            "CAL2 refit under v7": {"verdict": (c.get("STOP") or {}).get("verdict"), "E1_CAL_IN_SAMPLE": stats,
                                     "rho": fz["rho_form"], "constants": fz["constants"]},
            "D6FIX rung (DESCRIPTIVE)": {"cells moved by D6FIX": moved,
                                         "reading": f"{len(moved)} of {len(rows)} cells move under D6FIX"},
            "keys": "ms['v4d'] holds v7's time (the frozen scorers read that key); every rival priced by v7 with the "
                    "rival's own change (rho other, NAIVE-NODEDUP, eager), roofline, line and OLS as frozen"}


# ================================================================== the domain D (a required input)

def is_committed(rel: str, root: Path = REPO) -> bool:
    """The file is tracked and equal to HEAD's bytes in the checkout."""
    g = ["git", "-C", str(root)]
    a = subprocess.run([*g, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
    b = subprocess.run([*g, "diff", "--quiet", "HEAD", "--", rel], capture_output=True, text=True)
    return a.returncode == 0 and b.returncode == 0


def map_sessions(sessions: list[str]) -> list[str]:
    """The published sessions holding a map page (a gaps directory of a map shape)."""
    pub = REPO / "results" / "published"
    return [s for s in sessions if any(g.is_dir() for n in MS.NAMES for g in (pub / s).rglob(f"gaps-*-{n}-*"))]


def domain_refusals(dom: dict, dom_sha: str, map_reg: dict, map_reg_sha: str) -> list[str]:
    """Why the readout may not be used (empty: it may). Pure on its inputs except the map freeze, which reads the
    tree (read_map.refusals: the map registration's freeze and its plan)."""
    bad = []
    if map_reg_sha != MAP_REG_SHA256:
        bad.append(f"the map registration {MAP_NAME}.json is not the bytes registered at {MAP_REG_COMMIT}")
    why = RM.refusals(map_reg)
    if why:
        bad.append("the map registration does not verify: " + "; ".join(why))
    if dom.get("registration") != MAP_NAME:
        bad.append(f"the readout names registration {dom.get('registration')}, not {MAP_NAME}")
    pv = dom.get("provenance") or {}
    if pv.get("registration_sha256") != MAP_REG_SHA256:
        bad.append("the readout's provenance names another map registration sha256")
    if dom.get("verdict") != "READ":
        bad.append(f"the readout's verdict is {dom.get('verdict')}, not READ (no D)")
        return bad
    rows = [c for c in dom.get("rows") or [] if c.get("valid")]
    again = RM.domain(rows)
    for k in ("D", "statement"):
        if again[k] != dom["domain"][k]:
            bad.append(f"read_map.domain on the readout's own rows gives another {k}")
    for p in PATHS:
        a, b = again["per_path"][p], dom["domain"]["per_path"][p]
        for k in ("X_p_us", "S_p_MB", "verdict"):
            if a[k] != b[k]:
                bad.append(f"read_map.domain on the readout's rows: {p} {k} {a[k]!r}, the readout {b[k]!r}")
    if bool(again["slab_guard"]["triggered"]) != bool(dom["domain"]["slab_guard"]["triggered"]):
        bad.append("read_map.domain on the readout's rows: the slab guard reads otherwise")
    return bad


def domain_of(dom: dict) -> dict:
    """The registered D, per path: {path: {X_p_us, S_p_MB, in}} and the statement, copied from the verified readout."""
    d = dom["domain"]
    per = {p: {"X_p_us": d["per_path"][p]["X_p_us"], "S_p_MB": d["per_path"][p]["S_p_MB"],
               "verdict": d["per_path"][p]["verdict"], "in": d["per_path"][p]["verdict"] == "IN THE DOMAIN",
               "cells on the map": d["per_path"][p]["cells"]} for p in PATHS}
    return {"per_path": per, "slab guard": d["slab_guard"]["verdict"], "D": list(d["D"]), "statement": d["statement"]}


def label(geom: dict, D: dict) -> tuple[bool, str]:
    """(IN-DOMAIN, why) for one cell's registered geometry under D (domain_of's form): the map rule's D, verbatim:
    the path has an X_p, T_v7 >= X_p, W_e >= S_p when set, every active expert in one M-tile."""
    if geom["second_tile"]:
        return False, "OUT: an expert needs a second M-tile (addendum 2's DESCRIPTIVE cell)"
    p = D["per_path"].get(geom["path"])
    if p is None or not p["in"] or p["X_p_us"] is None:
        return False, f"OUT: path {geom['path']} is outside D ({p['verdict'] if p else 'no such path'})"
    if geom["T_v7_us"] < p["X_p_us"]:
        return False, f"OUT: T_v7 {geom['T_v7_us']:.3f} us under X_p {p['X_p_us']:.3f} us ({geom['path']})"
    if p["S_p_MB"] is not None and geom["W_e_MB"] < p["S_p_MB"]:
        return False, f"OUT: W_e {geom['W_e_MB']:.3f} MB under S_p {p['S_p_MB']:.3f} MB ({geom['path']})"
    return True, f"IN-DOMAIN: {geom['path']}, T_v7 {geom['T_v7_us']:.3f} >= X_p {p['X_p_us']:.3f} us" + (
        f", W_e {geom['W_e_MB']:.3f} >= S_p {p['S_p_MB']:.3f} MB" if p["S_p_MB"] is not None else "")


def labels(cells: list[dict], D: dict) -> dict:
    """{'model/label': {'in': bool, 'why': str}} for every main cell, and per-model counts."""
    lab = {}
    for r in cells:
        ok, why = label(r["geometry"], D)
        lab[f"{r['model']}/{r['label']}"] = {"in": ok, "why": why}
    count = {m: sum(v["in"] for k, v in lab.items() if k.split("/")[0] == m) for m in MODELS}
    return {"cells": lab, "in-domain cells per model": count,
            "certifiable (at least 10 registered in-domain cells)": {m: n >= MIN_IN_DOMAIN for m, n in count.items()},
            "in-domain cells": sum(count.values()), "out cells": len(lab) - sum(count.values())}


# ================================================================== the freeze

OWN_CODE = (
    "scripts/decode_synth.py", f"scripts/{DRIVER}", "scripts/setup_vm.sh", "scripts/vm_results_push.sh",
    "scripts/locked_r3.py", "scripts/r3_decode.py", LAUNCHER, PARENT_LAUNCHER,
    "scripts/scoring/v7dblind/v7dcommon.py", "scripts/scoring/v7dblind/register_v7d_blind.py",
    "scripts/scoring/v7dblind/score_blind_v7d.py", "scripts/scoring/v7blind/v7bcommon.py",
    "scripts/scoring/v7domain/v7mcommon.py", "scripts/scoring/v7domain/read_map.py",
    "scripts/scoring/v7domain/map_shapes.py", "scripts/replicate_noise_floor.py",
    "docs/registered/2026-10-14-decodeB-blind-gh200.json", "docs/registered/2026-10-14-decodeB-blind-gh200.txt",
    "docs/registered/2026-11-04-decode-addendum5-gh200.txt", "docs/registered/2026-11-10-decode-addendum6-gh200.txt",
    f"docs/registered/{GATE_NAME}.json", f"docs/registered/{GATE_NAME}.txt",
    f"docs/registered/{BLIND_NAME}.json", f"docs/registered/{BLIND_NAME}.txt",
    f"docs/registered/{MAP_NAME}.json", f"docs/registered/{MAP_NAME}.txt", VM.PLAN,
)


def own_files(final: bool) -> tuple:
    return tuple(page_path(m, k) for m, k in PAGE_KEYS) + OWN_CODE + ((DOMAIN,) if final else ())


def freeze(final: bool, root: Path = REPO) -> dict:
    g = VB.gate_registration()["freeze"]
    own = {p: sha256(Path(root) / p) for p in own_files(final) if p not in g["files"]}
    return {"statement": ("the scorer (score_blind_v7d.py) refuses unless every file of the v7 gate registration's "
                          "freeze has its sha256 there and v7's constants hash is the one registered there (v7 as the "
                          "gate froze it, unchanged since), and every file under 'own' has its sha256 here (the map's "
                          "registration, plan and readout among them once final); the plan's sha256 is computed by the "
                          "registration builder from the same document"),
            "gate registration": {"name": GATE_NAME, "json sha256": sha256(REPO / f"docs/registered/{GATE_NAME}.json"),
                                  "files": len(g["files"]), "constants_sha256": g["constants_sha256"]},
            "own": own}


def verify_freeze(reg: dict, root: Path = REPO) -> list[str]:
    """Every difference from the registered freeze (empty: frozen); v7bcommon.verify_freeze's rule."""
    return VB.verify_freeze(reg, root=root)


# ================================================================== GD5 offline

def gd5_file(model: str) -> str:
    """The tuned-config file vLLM's get_moe_configs looks for on bf16 on a GH200 (d5common.gd5_file's rule)."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    return f"E={cfg.num_experts},N={cfg.intermediate_size},device_name=NVIDIA_GH200_480GB.json"


def gd5_offline(configs_dirs: dict[str, Path]) -> dict:
    """GD5 offline on vLLM source trees' fused_moe/configs (d5common.gd5_offline's rule for these models)."""
    out = {}
    for lab, d in configs_dirs.items():
        names = {p.name for p in Path(d).glob("*.json")}
        out[lab] = {m: ("TUNED FILE PRESENT: " + gd5_file(m)) if gd5_file(m) in names else
                    "NO TUNED FILE: the default config" for m in MODELS}
    return out


#: GD5 offline (2026-11-18, this build), gd5_offline on the two local vLLM source trees d5common.GD5_REGISTERED read
#: (~/code/vllm-contrib-scratch/vllm54959/vllm-main at c1d8fad and ~/Desktop/research/spec-decode-kernels/vllm at
#: fafca38; neither in the repository, so typed in here as d5common does): no NVIDIA_GH200_480GB file in either
GD5_REGISTERED = {
    "label": "COMPUTED offline (2026-11-18) on two local vLLM source trees; the VM's runtime GD5 readback binds",
    "files looked up": {m: f"E={e},N={n},device_name=NVIDIA_GH200_480GB.json" for m, e, n in (
        ("deepseek-v3", 256, 2048), ("deepseek-v3-tp4", 256, 512), ("deepseek-v3-tp8", 256, 256),
        ("qwen2-57b-a14b-tp8", 64, 320))},
    "vllm main c1d8fad (2026-11-03)": {m: "NO TUNED FILE: the default config" for m in MODELS},
    "vllm fafca38 (2026-03-28)": {m: "NO TUNED FILE: the default config" for m in MODELS},
    "nearest bf16 files present (not GH200)": {
        "deepseek-v3": [],
        "deepseek-v3-tp4": ["E=256,N=512,device_name=NVIDIA_H100_80GB_HBM3.json", "E=256,N=512,device_name=NVIDIA_H20.json",
                            "E=256,N=512,device_name=NVIDIA_A100_80GB_PCIe.json"],
        "deepseek-v3-tp8": ["E=256,N=256,device_name=NVIDIA_H20.json"],
        "qwen2-57b-a14b-tp8": ["E=64,N=320,device_name=NVIDIA_H200.json", "E=64,N=320,device_name=NVIDIA_H100_80GB_HBM3.json"]},
    "lookup": "one exact file name per (E, N, device, dtype); no device fallback (d5common.GD5_REGISTERED)",
    "verdict": "PASS offline: no model has a GH200 tuned file, so vLLM runs default_config(T, E), the tile v7 prices",
    "not the pinned version": ("neither tree is the pinned vLLM 0.27.1 (not installed off the VM); the VM's runtime GD5 "
                               "readback still refuses any cell that received another config"),
}
