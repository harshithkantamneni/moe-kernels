#!/usr/bin/env python
"""Writes (--write) or checks (--check) model v7's decode domain map registration,
docs/registered/2026-11-18-v7-domain-map-gh200.{json,txt}, its 14 pages (docs/registered/2026-11-18-v7-domain-map-pages)
and its plan scripts/plans/rentalV7M-domain-map-2026-11.plan.

    python scripts/scoring/v7domain/register_v7_domain.py --write | --check

No DRAFT stage: the registration is final when pushed, before any page of the map. --check rebuilds the document and
compares: the pages, the text and the plan byte for byte, the JSON with every float within 1e-9 relative (the CAL2
fit's lstsq differs in the last bits between BLAS builds, the 2026-11-17 numdrift record), everything else exact.

Building it computes the CAL2 refit under v7 (the v7 blind's fit path, unchanged) and v7's price of every map cell; no
map shape has a page."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7mcommon as M  # noqa: E402

REPO = M.REPO
REL = 1e-9


def shapes_block() -> dict:
    return {
        "what": ("14 SYNTHETIC decode shapes, Phase 0's map geometry (synA-/synB-/synC- there, renamed map-<family>-"
                 "e<E>k<k>-h<H>i<I> here): no real model's name, config or checkpoint; synthetic weights, as the "
                 "apparatus always uses; routing never reads the carried fields (each cell names its counts)"),
        "shapes": M.MS.as_dicts(),
        "Phase 0 names": M.MS.PHASE0_NAME,
        "where they live": ("scripts/scoring/v7domain/map_shapes.py, NOT moe/spec.py: moe/spec.py is frozen at sha256 "
                            "58fb9b7f... by twelve registrations (CAL2, CAL4, tileB3, decode addenda 2 to 6, PC, PC2, RR "
                            "and the v7 gate; the v7 decode blind through the gate's freeze), and adding the 14 entries "
                            "there voids those freezes "
                            "(16 tests fail; gate_v7.py and score_blind_v7.py refuse). On the laptop map_shapes."
                            "installed() adds them for one block; on the VM "
                            f"{M.VMSITE}/sitecustomize.py, on the driver's PYTHONPATH (set by {M.LAUNCHER}), adds them "
                            "when a process first imports moe.spec. The pinned driver, r3_decode.py and locked_r3.py "
                            "run unchanged"),
        "no collision": ("no map name is a moe.spec model name, an hf_repo, or matches the reserved-model scan "
                         "(v7bcommon.PAT); tests/test_v7_domain_map.py holds this"),
    }


def cells_block(pred: dict) -> dict:
    cells = pred["cells"]
    cnt = lambda key: dict(sorted(Counter(key(c) for c in cells).items()))  # noqa: E731
    pages = {m: M.reg_file(M.page_path(m), pred["pages"][M.page_path(m)]) for m in M.MS.NAMES}
    return {
        "design": (f"Phase 0's map, regenerated exactly ({M.PHASE0}): A, path twins across E (naive T1 base, MIN-T2/4/8/16 "
                   "aligned twins with the same GEMMs, BALANCED T32/64/96/128); B, slab per expert 2.36 to 37.75 MB at "
                   "E 32, k 4 (naive T1 twin, MIN-T4/T16, a T8 ladder whose A gives v7 about 40/60/100/200 us, chosen "
                   "by v7's price, T64/96/128 BALANCED); C, block-scan at E 256, k 8 (naive T1 and T8, block-scan "
                   "MIN-T12/T16 twins, ladders T16 to T128). Every cell a ladder: the first A experts, T k rows "
                   "balanced, no expert over BLOCK_M; one GRR main page per shape, NATIVE arm, shuffle seed 20261014"),
        "count": len(cells),
        "by family and path": {f"{k[0]} {k[1]}": v for k, v in cnt(lambda c: (c["family"], c["path"])).items()},
        "by path": cnt(lambda c: c["path"]), "by tile": cnt(lambda c: c["tile"]),
        "predicted call time (us)": {"under 60": sum(c["pred_us"] < 60 for c in cells),
                                     "60 to 200": sum(60 <= c["pred_us"] < 200 for c in cells),
                                     "200 and over": sum(c["pred_us"] >= 200 for c in cells)},
        "in scope (every active expert in one M-tile)": sum(c["in_scope"] for c in cells),
        "D6FIX-branch cells": sum(c["d6fix"] for c in cells),
        "pages": pages,
        "page checks (COMPUTED: r3_decode.page_from, rotation, estimated_seconds at 9 repeats)": pred["checks"],
        "units": M.units(pred["checks"]),
        "reproduces Phase 0": ("the 130 cells equal Phase 0's out/map.json cell for cell (shape, label, T, counts, path, "
                               "tile) and v7's prices to 5e-7 us (map.json's rounding)"),
    }


def build(pred: dict | None = None) -> dict:
    pred = M.predictions() if pred is None else pred
    fz = pred["frozen"]
    plan = M.plan_text(pred["checks"], fz)
    gate = M.VB.gate_registration()
    r = M.rental(pred["checks"])
    return {
        "registered": M.DATE,
        "name": M.NAME,
        "status": ("REGISTERED BEFORE ANY PAGE of the map's shapes (none exists): final when pushed; no DRAFT stage. "
                   "DESCRIPTIVE: a measurement that decides v7's decode validity domain D by the rule below, not a "
                   "certification"),
        "design": ("the owner-approved plan of 2026-11-18 (moe-kernels-work 2026-11-18 build-domain/PLAN.md), Phase 1: one "
                   "mapping rental, DESCRIPTIVE, its threshold-selection rule registered before any page so the map "
                   f"decides the range mechanically; Phase 0's diagnosis and map design ({M.PHASE0})"),
        "tool": ("scripts/scoring/v7domain/register_v7_domain.py --write / --check (this file, the pages and the plan), "
                 "scripts/scoring/v7domain/read_map.py <results tree> <out dir> (the readout), "
                 f"{M.LAUNCHER} (the launcher), scripts/{M.DRIVER} (the pinned decode driver, unchanged)"),
        "what": ("v7 with every switch on at its CAL2 refit, frozen and UNCHANGED (no offset fix, no refit), priced on "
                 "130 cells of 14 synthetic shapes built to separate slab size, call length, decode path, tile and wave "
                 "fill; the map's pages decide, by the registered rule, the decode calls v7 is valid on (D)"),
        "label": "DESCRIPTIVE",
        "shapes": shapes_block(),
        "cells": cells_block(pred),
        "rule": {
            "text": M.RULE_TEXT,
            "source": "Phase 0 section 6 ('Rule text, for registration before Phase 1's pages'), verbatim",
            "points (how read_map.py computes it; fixed before any page)": list(M.RULE_POINTS),
            "constants": {"e_ub_max": M.E_UB_MAX, "rms_max": M.RMS_MAX, "mean_max": M.MEAN_MAX,
                          "min_cells": M.MIN_CELLS, "t975_8": M.T975_8, "median_se": M.MEDIAN_SE,
                          "slab_span_MB": M.SLAB_SPAN_MB, "slab_min_us": M.SLAB_MIN_US, "slots": M.SLOTS,
                          "paths": list(M.PATHS)},
            "why these margins (Phase 0)": ("rms 2.5%, not 2%: 2% is unattainable on the seen data at any X; leave-one-"
                                            "shape-out because the step error varies about 1.3 us sd between models "
                                            "while replicate noise is about 0.04%; 6% / 2.5% / 1.0% sit under the "
                                            "blind bars 8% / 3% / 1.5%"),
        },
        "the domain registration this feeds": M.PHASE2,
        "model v7 (frozen)": {
            "module": "scripts/r3_timing_model_v7.py",
            "as frozen by": (f"the v7 gate registration {M.GATE_NAME} (its freeze: {len(gate['freeze']['files'])} files "
                             f"and constants_sha256 {gate['freeze']['constants_sha256']}), unchanged since and used "
                             f"unchanged by the v7 decode blind {M.BLIND_NAME}; this registration holds that freeze and "
                             "adds its own files (freeze, below)"),
            "switches": M.V7.V7.as_dict(),
            "fit": ("the CAL2 refit under v7 by the v7 blind's code path (v7bcommon.decode_fit, gate_v7.compute_decode's "
                    "fit), every switch on; D6FIX's constants from the CAL2 fit with v4d' on every naive cell"),
            "CAL2 refit under v7 (COMPUTED)": pred["CAL2 refit under v7"],
            "D6FIX constants (COMPUTED)": fz["model"]["D6FIX"],
        },
        "predictions": {
            "label": ("COMPUTED by v7 before any page of the map (v7mcommon.price: r3_timing_model_v7.price_decode_cell at "
                      "the CAL2 refit under v7, D6FIX on); read_map.py reads them from this file. Phase 0 priced the same "
                      "cells offline before this registration (v7's prices, no measurement)"),
            "frozen": fz,
            "cells": pred["cells"],
        },
        "rental": dict(r, plan=M.PLAN, plan_sha256=hashlib.sha256(plan.encode()).hexdigest(),
                       driver=f"scripts/{M.DRIVER} (pinned, unchanged)",
                       launcher=(f"{M.LAUNCHER}: {M.PARENT_LAUNCHER} (RR's, with the apt-lock fix and the prepare lock "
                                 "handling, and V7B's decode-driver-only and DRAFT refusal) with lines marked V7M: this "
                                 f"plan named, and start runs the driver with PYTHONPATH=$HOME/moe/repo/{M.VMSITE} so the "
                                 "map shapes reach moe.spec"),
                       lock="1710 MHz at G 1 (G4 1705), 700 W (prelude), as every decode rental",
                       machine="one gpu_1x_gh200 (Lambda), the decode driver's prelude and calibrate, then the 14 pages",
                       guardian=("outside the repository (moe-kernels-work 2026-11-18 guardians/tV7M_*): deadline launch "
                                 f"+ {r['deadline_after_launch_min']} min, hard cap launch + "
                                 f"{r['hard_cap_after_launch_min']} min, terminate on the cap"),
                       estimate_against_phase0=(f"{r['units_min']} unit-min against Phase 0's 69: the decode "
                                                "registrations' rule rounds each page's estimate up to whole minutes "
                                                "(d5common._rd_est), Phase 0 summed the fractions (63.5 page-min)")),
        "disclosures": [
            ("the second reopening of decode (owner, 2026-11-18): decode closed after v6d (2026-11-10), reopened for v7 "
             "(2026-11-17), and v7's one decode blind failed (published: E1-DEC pooled FAILS, 125 cells rms 4.17%, "
             "worst +17.20%; Mixtral-8x22B and Qwen2-57B-A14B hold their own bars, Granite-3.0-1B-A400M fails at 6.74%); "
             "this map and the domain blind it feeds reopen it a second time, with every decode page in view"),
            ("POST-HOC origin: the domain question and the map were designed after the blind's pages were read: "
             "moe-kernels-work 2026-11-18 v7-domain/REPORT.md (every cell over 8% among the 345 out-of-sample decode "
             "cells is a Granite small-batch 16x64x128 cell under one wave, over-priced by +2.5 to +4.6 us; a 44 us "
             "call-time cut separates them) and domain-phase0/REPORT.md (the over-price is the aligned path's fixed "
             "block, s_path plus ALIGN, a per-call offset on the aligned paths; the rule's margins chosen on seen "
             "data)"),
            ("Phase 0 priced these map shapes offline before this registration: the predictions are v7's, computed "
             "without any measurement of a map shape; family B's T8 ladder picks each A by v7's price; no page of any "
             "map shape exists"),
            ("v7 is frozen and unchanged (every switch on, the CAL2 refit, the v7 gate's freeze): no offset fix and no "
             "refit, which would be a refit on seen pages"),
            (f"cost: {r['units_min']} unit-min, about {r['wall_min']} min wall: ${r['cost_usd']} at the inherited "
             f"d2common.RATE $2.29/h, ${r['cost_usd_at_list']} at Lambda's list price of 2026-11-18 ($2.59/h); "
             f"deadline launch + {r['deadline_after_launch_min']} min, hard cap + {r['hard_cap_after_launch_min']} min "
             f"(${r['cost_at_hard_cap_usd']} / ${r['cost_at_hard_cap_usd_at_list']}); above the plan's ~1 h / ~$3"),
            ("the map shapes are not in moe/spec.py (its bytes are frozen by twelve registrations and through the v7 "
             "gate's freeze by the v7 decode blind); they reach the "
             "pinned driver through a sitecustomize hook on the driver's PYTHONPATH, set by the launcher copy"),
        ],
        "seen-data analogue (ESTIMATE, Phase 0 section 4, not computed here)": (
            "per path on the seen decode pages, the smallest call time with every cell's bound under 6%: small-batch "
            "112.5 us, block-scan 51.6, naive 56.3 (under 8%: 44.1, 40.1, 20.8); the map, not these, decides D"),
        "unchanged": ("every earlier registration, page, verdict and pinned file; moe/spec.py; the decode driver and "
                      "setup_vm.sh; the 1710 lock and 700 W; v7 and the v7 gate registration"),
        "freeze": M.freeze(),
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
    L = [f"REGISTERED {d['registered']}, before any page of the map's 14 synthetic shapes: model v7's decode domain map "
         f"(DESCRIPTIVE), {d['name']}", f"status: {d['status']}", f"design: {d['design']}", f"tool: {d['tool']}", "",
         "THRESHOLD-SELECTION RULE (verbatim, Phase 0 section 6):"]
    L += ["  " + ln for ln in d["rule"]["text"].splitlines()]
    L += ["", "HOW read_map.py COMPUTES IT (fixed before any page):"]
    L += [f"  - {x}" for x in d["rule"]["points (how read_map.py computes it; fixed before any page)"]]
    L.append("")
    for k, v in d.items():
        if k in ("registered", "name", "status", "design", "tool", "predictions", "freeze", "rule", "cells"):
            continue
        _flat("", {k: v}, L)
    c = d["cells"]
    L += ["", "cells:"]
    for k, v in c.items():
        if k in ("pages", "page checks (COMPUTED: r3_decode.page_from, rotation, estimated_seconds at 9 repeats)", "units"):
            continue
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L.append("  pages (path, sha256, cells, est min, R, weights GB):")
    chk = c["page checks (COMPUTED: r3_decode.page_from, rotation, estimated_seconds at 9 repeats)"]
    for m, p in c["pages"].items():
        L.append(f"    {p['path']} {p['sha256']} {len(p['cells'])} {chk[m]['est_min']} {chk[m]['R']} {chk[m]['weights_GB']}")
    L.append(f"  rule constants: {json.dumps(d['rule']['constants'])}")
    p = d["predictions"]
    fz = p["frozen"]
    L += ["", f"PREDICTIONS: {p['label']}",
          f"  constants: {json.dumps(fz['constants'])}, rho {fz['rho_form']}, D6FIX {json.dumps(fz['model']['D6FIX'])}",
          "  family / shape / label / T / path / tile / A / waves / W_e MB / in scope / v7 us (3 decimals) / why"]
    for r in p["cells"]:
        L.append(f"    {r['family']} {r['model']} {r['label']} T{r['T']} {r['path']} {r['tile']} A{r['A']} "
                 f"{r['waves']:.2f} {r['W_e_MB']:.2f} {'in' if r['in_scope'] else 'OUT'} {r['pred_us']:.3f} | {r['why']}")
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
        return [x for i, (u, w) in enumerate(zip(a, b)) for x in close(u, w, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=REL, abs_tol=1e-15) else [f"{path}: {a!r} != {b!r}"]
    return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} != {b!r}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    jpath, tpath, ppath = REPO / M.REG_JSON, REPO / M.REG_TXT, REPO / M.PLAN
    pred = M.predictions()
    if a.write:
        for p, t in pred["pages"].items():
            (REPO / p).parent.mkdir(parents=True, exist_ok=True)
            (REPO / p).write_text(t)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        ppath.write_text(M.plan_text(pred["checks"], pred["frozen"]))
        doc = build(pred)
        jpath.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        tpath.write_text("\n".join(text(doc)) + "\n")
        print(f"wrote {jpath.name}, {tpath.name}, {ppath.name} and {len(pred['pages'])} pages")
        return 0
    ok = True
    for p, t in pred["pages"].items():
        same = (REPO / p).is_file() and (REPO / p).read_text() == t
        ok &= same
        if not same:
            print(f"{p}: DIFFERS")
    if not jpath.is_file():
        print(f"{jpath.name}: missing")
        return 1
    want = json.loads(jpath.read_text())
    doc = json.loads(json.dumps(build(pred), default=str))
    bad = close(doc, want)
    print(f"{jpath.name}: {'same' if not bad else 'DIFFERS: ' + '; '.join(bad[:5])}")
    ok &= not bad
    for p, t in ((tpath, "\n".join(text(doc)) + "\n"), (ppath, M.plan_text(pred["checks"], pred["frozen"]))):
        same = p.is_file() and p.read_text() == t
        ok &= same
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
