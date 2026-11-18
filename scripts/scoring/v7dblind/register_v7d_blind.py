#!/usr/bin/env python
"""Writes (--write) or checks (--check) model v7's decode DOMAIN blind registration,
docs/registered/2026-11-18-v7-domain-blind-gh200.{json,txt}, its pages (docs/registered/2026-11-18-v7-domain-blind-pages)
and its plan scripts/plans/rentalV7D-domain-blind-2026-11.plan.

    python scripts/scoring/v7dblind/register_v7d_blind.py --write [--final] | --check

--write writes a DRAFT: D is unknown (the map rental runs; its readout does not exist yet), every cell is labelled
PENDING and the plan carries a DRAFT line that scripts/vm_run_v7d.sh refuses. --write --final refuses unless the map's
readout (v7dcommon.DOMAIN) is committed and verifies: it names the map registration at its registered bytes
(v7dcommon.MAP_REG_SHA256, 556029e1), the map registration's freeze and plan verify (read_map.refusals), its verdict is
READ, read_map.domain recomputed on its own rows gives the same D, D is not empty, the map's session is among the
published sessions the census reads, no model has a decode page, and at least one model has 10 registered in-domain
cells; it then labels every cell IN-DOMAIN or OUT mechanically and writes the document without the DRAFT marks.
--check rebuilds the document in the mode the committed JSON records (the census over the sessions it records) and
compares: the text, the pages and the plan byte for byte, the JSON with every float within 1e-9 relative.

Building it computes the CAL2 refit under v7 (the v7 blind's fit, unchanged) and v7's price of every cell; it reads no
page of the four models (none exists)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7dcommon as V  # noqa: E402

REPO = V.REPO
REL = 1e-9


# ================================================================== the domain input

def domain_input(sessions: list[str]) -> tuple[dict | None, str | None, list[str]]:
    """(readout, its sha256, why it may not be used) for --final. Reads the committed readout and the map
    registration from the checkout."""
    p = REPO / V.DOMAIN
    if not p.is_file():
        return None, None, [f"{V.DOMAIN} does not exist (the map's readout is not committed yet): the registration stays DRAFT"]
    why = []
    if not V.is_committed(V.DOMAIN):
        why.append(f"{V.DOMAIN} is not committed (untracked, or differs from HEAD)")
    mp = REPO / VMREG
    dom = json.loads(p.read_text())
    why += V.domain_refusals(dom, V.sha256(p), json.loads(mp.read_text()), V.sha256(mp))
    if not V.map_sessions(sessions):
        why.append("no published session holds a map page: the census must read the map's session")
    return dom, V.sha256(p), why


VMREG = f"docs/registered/{V.MAP_NAME}.json"


# ================================================================== the document

def labels_text() -> dict:
    return {
        "time": "BLIND: the decode call, its tiles and its instrument have no page on any of the four models",
        "pages": "SYNTHETIC (decode_synth.py's frozen rules; config shapes only, synthetic weights)",
        "per_model": {
            "deepseek-v3": ("SEEN-MODEL: vLLM and SGLang fused_experts and torch grouped GEMMs timed on H200 at T 1 to "
                            "128 and beyond (2026-08-22 to 2026-09-01, the standard sweep, the three-way, fp8, "
                            "whole-layer, ridge and alpha sessions) and vLLM fused_experts on A100 (2026-08-28 "
                            "a100-cross-card): its decode-size calls on other cards, span-timed, not this apparatus' "
                            "GRR decode page (census, merged.csv rows); no GH200 page, no decode-stage page"),
            "deepseek-v3-tp4": "UNSEEN: no file, row or page names it (census)",
            "deepseek-v3-tp8": "UNSEEN: no file, row or page names it (census)",
            "qwen2-57b-a14b-tp8": ("SEEN-MODEL: R3-tile pages on GH200 (rentals 3, 4 and 5, 2026-10-06 to 2026-10-09: "
                                   "gap pages, floor and counter runs at the R3 tile 32x64x64 w8 s4 and 32x64x32 w8 "
                                   "s4, census); no page at a decode tile, no decode-stage page"),
        },
        "everything published": "SEEN",
    }


def models_block(cen: dict) -> dict:
    per = cen["per model"]
    return {
        "chosen": list(V.MODELS),
        "why": ("the plan's Phase 2 models, named before any map page in the map registration (its 'next step'): "
                "DeepSeek-V3 and its tp4 and tp8 shards and Qwen2-57B-A14B-tp8, real models in moe/spec.py "
                "(unchanged); no decode page of any has been timed (census); each fits one GH200 NATIVE (weights "
                "below); the v7 blind's routing classes and T grid carry over unchanged"),
        "footprint (COMPUTED, d5common.footprint's rule on these pages)": footprints(),
        "census (COMPUTED)": cen,
        "every model with no decode page": [m for m in V.MODELS if per[m]["verdict"] == "NO DECODE PAGE"],
        "GD5 offline (vLLM tuned configs)": V.GD5_REGISTERED,
        "the map's synthetic shapes against these models (COMPUTED)": V.map_vs_models(),
        "map shapes reading": ("no map shape has any model's (E, H, I, k); the nearest are the map's E 256, k 8 shapes "
                               "(family A at H 2048, I 512; family C at H 7168, I 384, DeepSeek-V3's H with another I, "
                               "Phase 0's 'DeepSeek-V3-like H, NOT its shards' (H, I)'; and H 4096, I 1024) and "
                               "Qwen2-57B-tp8's E 64, k 8 at H 2048, I 512: the map fixed D on shapes near these "
                               "models' E and k, by design, never on their shapes"),
        "labels": labels_text(),
    }


def footprints() -> dict:
    from moe.spec import MODEL_CONFIGS
    out = {}
    for m in V.MODELS:
        cfg = MODEL_CONFIGS[m]
        page = V.RD.page_from(V.page_doc(m, "main"), cfg, m)
        R, calls, why = V.RD.rotation(cfg, page)
        we = V.RD.expert_bytes(cfg)
        out[m] = {"E": cfg.num_experts, "k": cfg.top_k, "H": cfg.hidden_size, "I": cfg.intermediate_size,
                  "W_e_MB": V.r6(we / 1e6), "R": R, "calls": calls, "weights_GB": V.r6(R * cfg.num_experts * we / 1e9),
                  "why": why}
    return out


def cells_block(pred: dict) -> dict:
    us = V.units()
    return {
        "structure": ("the v7 blind's recipe (addendum 5's, decodeB-blind's unit form): scripts/decode_synth.py's frozen "
                      "generators, unchanged: one GRR main page per model (nodrop) and GR1 control pages on "
                      f"{' and '.join(V.CTL)} (drop group gr1; the two models whose GRR page rotates the most copies), "
                      "T 1, 2, 4, 8, 16, 32, 64, 96, 128, the routing classes BALANCED, UNIFORM, UNIFORM-UNSHUFFLED, "
                      "LADDER, LADDER-BOTTOM and MINCOVER (scaled to each model's E and k by the generator), the NATIVE "
                      "arm, shuffle seed 20261014, the 1710 MHz lock at G 1 and 700 W; no TRACE-DW cell (no rental-5 "
                      "DW shape for these models)"),
        "per model": pred["per model"],
        "units": us, "drop_order": ["gr1"],
        "pages": {"dir": V.PAGES, "label": "this blind's page lists (decode_synth's frozen generators); never timed",
                  "files": {f"{m}-{k}.json": V.reg_file(m, k) for m, k in V.PAGE_KEYS}},
        "plan_units": V.unit_lines(),
        "descriptive": ("addendum 2's second-tile cells (an expert past BLOCK_M rows) are DESCRIPTIVE, out of E1-DEC, "
                        "read as SIGMA-D; their list is the predictions' descriptive_cells; they are OUT of D by D's "
                        "own scope condition"),
    }


def domain_block(draft: bool, pred: dict, dom: dict | None, dom_sha: str | None) -> dict:
    base = {
        "input": (f"{V.DOMAIN}: read_map.py's domain.json on the published map session (registration {V.MAP_NAME}, "
                  f"sha256 {V.MAP_REG_SHA256}, public at {V.MAP_REG_COMMIT} before any map page), committed; a "
                  "REQUIRED INPUT, read mechanically"),
        "rule (the map registration's, verbatim)": V.VM.RULE_TEXT,
        "labelling": ("a cell is IN-DOMAIN when its frozen geometry (path by r3_decode.geometry, T_v7 the registered "
                      "prediction in us at 6 decimals as the map's, W_e in MB at 6 decimals, the second-tile test by "
                      "d2common.second_tile) satisfies D: its path has an X_p (verdict IN THE DOMAIN), T_v7 >= X_p, "
                      "W_e >= S_p when the slab guard set S_p, and every active expert fits one M-tile; else OUT, "
                      "DESCRIPTIVE (v7dcommon.label)"),
        "verification at --final": ("the readout is committed; it names the map registration at its registered "
                                    "sha256; the map registration's freeze and plan verify (read_map.refusals); its "
                                    "verdict is READ; read_map.domain on the readout's own valid rows gives the same "
                                    "X_p, S_p, verdicts, slab guard and D; D is not empty; the map's session is "
                                    "published and in the census"),
    }
    if draft:
        return dict(base, status=("NOT KNOWN: the map rental is running; its readout does not exist. Every cell is "
                                  "PENDING; register_v7d_blind.py --write --final labels them from the committed "
                                  "readout and refuses without it"))
    D = V.domain_of(dom)
    lab = V.labels(pred["cells"], D)
    return dict(base, status="READ from the committed readout and verified (verification at --final)",
                readout={"path": V.DOMAIN, "sha256": dom_sha, "provenance": dom.get("provenance"),
                         "cells": dom.get("cells")},
                D=D, labels=lab)


def build(draft: bool = True, pred: dict | None = None, sessions: list[str] | None = None,
          dom: dict | None = None, dom_sha: str | None = None) -> dict:
    pred = V.predictions() if pred is None else pred
    cen = V.census(sessions)
    fz = pred["frozen"]
    plan = V.plan_text(fz, draft)
    gate = V.VB.gate_registration()
    r = V.rental()
    status = ("DRAFT, NOT REGISTERED: D is unknown (the map rental is running); may be committed for real only when "
              f"the map's readout ({V.DOMAIN}) is committed and verifies; until then every cell is PENDING, the plan "
              f"carries a DRAFT line and {V.LAUNCHER} refuses it" if draft else
              f"REGISTERED BEFORE ANY PAGE of the four models, after the map's readout was committed and verified ({V.DOMAIN})")
    return {
        "registered": V.DATE,
        "name": V.NAME,
        "draft": draft,
        "status": status,
        "design": ("the owner-approved plan of 2026-11-18 (moe-kernels-work 2026-11-18 build-domain/PLAN.md), Phase 2: "
                   "one blind GH200 rental that tests v7 at vLLM's decode tiles inside the validity domain D the map "
                   f"({V.MAP_NAME}) produces by its registered rule, on four models no decode page has timed; v7 "
                   "frozen and unchanged; the old bars; decodeB-blind's rule; pass or fail, the stage then stops"),
        "tool": ("scripts/scoring/v7dblind/register_v7d_blind.py --write [--final] / --check (this file, the pages and "
                 "the plan), scripts/scoring/v7dblind/score_blind_v7d.py <results tree> <out dir> (the scorer), "
                 f"{V.LAUNCHER} (the launcher), scripts/{V.DRIVER} (the pinned decode driver, unchanged)"),
        "what": ("E1-DEC BLIND under model v7 INSIDE D: v7 with every switch on at its CAL2 refit, frozen before any "
                 "page, on four models no decode page has timed, scored on the in-domain cells only"),
        "models": models_block(cen),
        "cells": cells_block(pred),
        "domain": domain_block(draft, pred, dom, dom_sha),
        "bars": dict(V.BARS, source=V.BARS_SOURCE),
        "statistic": ("e_dec = (T_v7 - T_meas) / T_meas per valid NATIVE main cell labelled IN-DOMAIN, 0 at or under "
                      "1 us absolute (dcommon.e_dec); OUT cells are scored the same way and reported DESCRIPTIVE"),
        "certification": V.CERT_RULE + "; here on the IN-DOMAIN cells only, with the minimum below",
        "minimum": V.MIN_RULE,
        "claim if certified": ("v7 predicts vLLM's fused MoE decode call on GH200 within 3% rms / 8% worst on unseen "
                               "models, inside D (per certified model); a fail closes decode"),
        "scope line": V.D3.SCOPE,
        "model v7 (frozen)": {
            "module": "scripts/r3_timing_model_v7.py",
            "as frozen by": (f"the v7 gate registration {V.GATE_NAME} (its freeze: {len(gate['freeze']['files'])} files "
                             f"and constants_sha256 {gate['freeze']['constants_sha256']}); unchanged since, the same "
                             f"fit as the v7 blind ({V.BLIND_NAME}) and the map ({V.MAP_NAME})"),
            "switches": V.V7.V7.as_dict(),
            "CAL2 refit under v7 (COMPUTED; v7bcommon.decode_fit)": pred["CAL2 refit under v7"],
            "D6FIX constants (COMPUTED)": fz["model"]["D6FIX"],
        },
        "predictions": {
            "label": ("COMPUTED by v7 before any page of the four models (finalise_decode.frozen and .predict at the "
                      "CAL2 refit under v7, this blind's pages); the scorer reads them from this file"),
            "not_priced": pred["not_priced"], "descriptive_cells": pred["descriptive_cells"],
            "pricers agree": pred["pricers agree"],
            "D6FIX rung (DESCRIPTIVE)": pred["D6FIX rung (DESCRIPTIVE)"], "keys": pred["keys"],
            "frozen": fz, "cells": pred["cells"],
        },
        "rental": dict(r, plan=V.PLAN, plan_sha256=hashlib.sha256(plan.encode()).hexdigest(),
                       driver=f"scripts/{V.DRIVER} (pinned, unchanged)",
                       launcher=(f"{V.LAUNCHER}: {V.PARENT_LAUNCHER} with lines marked V7D: the DRAFT refusal names "
                                 "this registration; no map hook (the four models are in moe/spec.py)"),
                       lock="1710 MHz at G 1 (G4 1705), 700 W (prelude), as every decode rental",
                       guardian=("outside the repository (moe-kernels-work 2026-11-18 guardians/tV7D_*): launches only "
                                 "when this registration is FINAL and its plan_sha256 is the committed plan's; deadline "
                                 f"launch + {r['deadline_after_launch_min']} min, hard cap launch + "
                                 f"{r['hard_cap_after_launch_min']} min, terminate on the cap")),
        "disclosures": [
            ("the second reopening of decode (owner, 2026-11-18): decode closed after v6d (2026-11-10), reopened for v7 "
             "(2026-11-17); v7's one decode blind failed (E1-DEC pooled FAILS, 125 cells rms 4.17%, worst +17.20%, no "
             "model certified); the owner reopened it a second time for a validity domain, with every decode page, "
             "the blind's failure and its cause in view (a forking path the paper must disclose)"),
            ("POST-HOC origin of the domain idea: the domain question, Phase 0's diagnosis and the map's rule were "
             "designed after the v7 blind's pages were read (moe-kernels-work 2026-11-18 v7-domain/REPORT.md: every "
             "cell over 8% is a Granite small-batch 16x64x128 cell under one wave, over-priced by +2.5 to +4.6 us; "
             "domain-phase0/REPORT.md); the rule's margins were chosen on seen data"),
            (f"D is computed mechanically by the threshold-selection rule registered with the map ({V.MAP_NAME}, "
             f"public at {V.MAP_REG_COMMIT}) before any map page; this blind registers how D labels each cell and the "
             "minimum before D is known (this DRAFT), and takes D from the committed readout, verified, at --final"),
            ("v7 is frozen and unchanged: every switch on, the CAL2 refit (GO 1.60%; fixed rho, T0 -1.87, s_naive "
             "-2.14, s_block -0.72 us; D6FIX scaled, s_naive +9.55 us), the v7 gate's freeze; no offset fix, no refit; "
             "this blind certifies a range, it does not refit"),
            ("Phase 0 priced these four models offline before this registration (moe-kernels-work 2026-11-18 "
             "v7-domain/s3_unseen.py and REPORT.md: predictions without measurements, in-scope 43 of 48 cells each, "
             "the paths, the T_v7 ranges); the cells and predictions here equal that pricing; no page of any of the "
             "four exists"),
            ("DeepSeek-V3's decode-size calls were timed before, on other cards: vLLM and SGLang fused_experts and torch "
             "grouped GEMMs on H200 at T 1 to 128 (2026-08-22 to 2026-09-01) and vLLM fused_experts on A100 (2026-08-28 "
             "a100-cross-card); none on a GH200, none under this apparatus, no decode constant fitted on them; "
             "Qwen2-57B-A14B-tp8 has GH200 pages at the R3 tile (rentals 3 to 5), none at a decode tile (census)"),
            ("the map's shapes are near these models' E and k by design (family C at E 256, k 8, H 7168; family A at "
             "E 64, k 8) and equal to none of their (E, H, I, k) (models, the map's synthetic shapes)"),
            (f"cost: {r['units_min']} unit-min, about {r['wall_min']} min wall: ${r['cost_usd']} at the inherited "
             f"d2common.RATE $2.29/h, ${r['cost_usd_at_list']} at Lambda's list price of 2026-11-18 ($2.59/h); "
             f"deadline launch + {r['deadline_after_launch_min']} min, hard cap + {r['hard_cap_after_launch_min']} min "
             f"(${r['cost_at_hard_cap_usd']} / ${r['cost_at_hard_cap_usd_at_list']}); over the plan's ~1 h / ~$3: "
             "DeepSeek-V3's three pages are full pages of 48 cells; the gr1 control pages are dropped first when time "
             f"runs short (without them about {r['without the gr1 pages (DESCRIPTIVE)']['wall_min']} min)"),
            ("the bars are the ones registered for decode blinds before any page (owner, D4-1: addendum 5 line 9, "
             "carried by addendum 6 line 159): rms 3%, worst 8%, |mean| 1.5%, per model and pooled, not tightened"),
        ],
        "expected (ESTIMATE, judgment, not computed)": (
            "Phase 0 put every DeepSeek-V3 cell (207 to 6293 us) and nearly every -tp4 and -tp8 cell past a 44 us "
            "call-time cut, on the naive and block-scan paths; block-scan has 15 out-of-sample cells on record (3.15% "
            "rms), so whether D admits block-scan at all is the map's to say; if D excludes block-scan, the three "
            "DeepSeek-V3 models keep at most their 19 naive cells each and Qwen2-57B-tp8 its in-domain small-batch and "
            "naive cells"),
        "unchanged": ("every earlier registration, page, verdict and pinned file; moe/spec.py; the decode driver and "
                      "setup_vm.sh; the 1710 lock and 700 W; v7 and its gate registration; the map registration and "
                      "its rule"),
        "freeze": V.freeze(final=not draft),
    }


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and depth < 2:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {x if isinstance(x, str) else json.dumps(x, default=str)}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {x if isinstance(x, str) else json.dumps(x, default=str)}")
    else:
        out.append(f"{prefix}{v}")


def text(d: dict) -> list[str]:
    head = "DRAFT (not registered)" if d["draft"] else "REGISTERED"
    L = [f"{head} {d['registered']}, before any page of DeepSeek-V3, DeepSeek-V3-tp4, DeepSeek-V3-tp8 or "
         f"Qwen2-57B-A14B-tp8: model v7's decode domain blind, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", f"minimum: {d['minimum']}", ""]
    for k, v in d.items():
        if k in ("registered", "name", "draft", "status", "design", "tool", "minimum", "predictions", "freeze", "domain"):
            continue
        _flat("", {k: v}, L)
    dm = d["domain"]
    L += ["", f"DOMAIN: {dm['status']}", f"  input: {dm['input']}", f"  labelling: {dm['labelling']}",
          f"  verification at --final: {dm['verification at --final']}"]
    if "D" in dm:
        L.append(f"  readout sha256 {dm['readout']['sha256']}")
        L.append(f"  {dm['D']['statement']}")
        for p, x in dm["D"]["per_path"].items():
            L.append(f"  {p}: X_p {x['X_p_us']} us, S_p {x['S_p_MB']} MB, {x['verdict']}")
        lab = dm["labels"]
        L.append(f"  in-domain cells per model: {json.dumps(lab['in-domain cells per model'])}; certifiable: "
                 f"{json.dumps(lab['certifiable (at least 10 registered in-domain cells)'])}")
    p = d["predictions"]
    fz = p["frozen"]
    L += ["", f"PREDICTIONS: {p['label']}",
          f"  constants: {json.dumps(fz['constants'])}, rho {fz['rho_form']}, D6FIX {json.dumps(fz['model']['D6FIX'])}",
          f"  bars: {json.dumps(fz['bars'])}", f"  not priced: {p['not_priced']}; descriptive: {len(p['descriptive_cells'])}",
          f"  {p['pricers agree']}; D6FIX rung: {p['D6FIX rung (DESCRIPTIVE)']['reading']}",
          "  model / label / T / path / tile / W_e MB / v7 us (3 decimals) / domain label; DESC = second-tile cell"]
    labs = (dm.get("labels") or {}).get("cells", {})
    for r in p["cells"]:
        g = r["geometry"]
        key = f"{r['model']}/{r['label']}"
        tag = ("IN" if labs[key]["in"] else "OUT") if key in labs else "PENDING"
        L.append(f"    {r['model']} {r['label']} T{r['T']} {g['path']} {g['tile']} {g['W_e_MB']:.3f} {g['T_v7_us']:.3f} "
                 f"{tag}" + (" DESC" if g["second_tile"] else ""))
    fr = d["freeze"]
    L += ["", f"FREEZE: {fr['statement']}", f"  gate registration: {json.dumps(fr['gate registration'])}",
          f"  own files ({len(fr['own'])}):"]
    L += [f"    {h}  {q}" for q, h in fr["own"].items()]
    return L


def close(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{path}: keys differ"]
        return [x for k in a for x in close(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [x for i, (u, w) in enumerate(zip(a, b, strict=True)) for x in close(u, w, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=REL, abs_tol=1e-15) else [f"{path}: {a!r} != {b!r}"]
    return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} != {b!r}"]


def final_refusals(doc_sessions: list[str], pred: dict) -> tuple[dict | None, str | None, list[str]]:
    """Every reason --final may not write (empty: it may), with the verified readout."""
    dom, sha, why = domain_input(doc_sessions)
    cen = V.census(doc_sessions)["per model"]
    why += [f"{m}: {x['verdict']}" for m, x in cen.items() if x["verdict"] != "NO DECODE PAGE"]
    if dom is not None and not why:
        D = V.domain_of(dom)
        if not D["D"]:
            why.append("D is EMPTY: no path has an X_p (the map registration: no domain blind runs)")
        else:
            lab = V.labels(pred["cells"], D)
            if not any(lab["certifiable (at least 10 registered in-domain cells)"].values()):
                why.append(f"no model has {V.MIN_IN_DOMAIN} registered in-domain cells "
                           f"({json.dumps(lab['in-domain cells per model'])}): nothing could be certified")
    return dom, sha, why


def write(root: Path, doc: dict) -> None:
    jpath, tpath, ppath = root / V.REG_JSON, root / V.REG_TXT, root / V.PLAN
    for p in (jpath, ppath):
        p.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tpath.write_text("\n".join(text(doc)) + "\n")
    ppath.write_text(V.plan_text(doc["predictions"]["frozen"], doc["draft"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--final", action="store_true", help="label the cells from the committed map readout and register")
    ap.add_argument("--root", type=Path, default=REPO, help="where to write or read the files (tests)")
    a = ap.parse_args(argv)
    jpath, tpath, ppath = a.root / V.REG_JSON, a.root / V.REG_TXT, a.root / V.PLAN
    if a.write:
        sessions = V.published_sessions()
        pred = V.predictions()
        dom = sha = None
        if a.final:
            dom, sha, why = final_refusals(sessions, pred)
            if why:
                print("REFUSED: " + "; ".join(why))
                return 2
        if a.root == REPO:
            V.write_pages(a.root)
        doc = build(draft=not a.final, pred=pred, sessions=sessions, dom=dom, dom_sha=sha)
        write(a.root, doc)
        print(f"wrote {jpath.name}, {tpath.name}, {ppath.name} ({'DRAFT' if doc['draft'] else 'final'})")
        return 0
    if not jpath.is_file():
        print(f"{jpath.name}: missing")
        return 1
    want = json.loads(jpath.read_text())
    draft = bool(want.get("draft", True))
    sessions = want["models"]["census (COMPUTED)"]["sessions scanned"]
    dom = sha = None
    if not draft:
        p = REPO / V.DOMAIN
        dom, sha = (json.loads(p.read_text()), V.sha256(p)) if p.is_file() else (None, None)
        if dom is None:
            print(f"{V.DOMAIN}: missing")
            return 1
    doc = json.loads(json.dumps(build(draft=draft, sessions=sessions, dom=dom, dom_sha=sha), default=str))
    ok = True
    bad = close(doc, want)
    print(f"{jpath.name}: {'same' if not bad else 'DIFFERS: ' + '; '.join(bad[:5])}")
    ok &= not bad
    texts = [(tpath, "\n".join(text(doc)) + "\n"), (ppath, V.plan_text(doc["predictions"]["frozen"], doc["draft"]))]
    texts += [(a.root / p, t) for p, t in V.page_texts().items()]
    for p, t in texts:
        same = p.is_file() and p.read_text() == t
        ok &= same
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
