#!/usr/bin/env python
"""Writes (--write) or checks (--check) model v7's decode blind registration,
docs/registered/2026-11-17-v7-decode-blind-gh200.{json,txt}, and its plan scripts/plans/rentalV7B-decode-blind-2026-11.plan.

    python scripts/scoring/v7blind/register_v7_blind.py --write [--final] | --check

--write writes a DRAFT: the registration may be committed for real only after the v7 gate registration
(2026-11-17-v7-gate-gh200) is public and gate_v7.py reads DECODE: GO. --write --final refuses unless the committed gate
output (v7bcommon.GATE_OUT) holds a 'DECODE: GO' line; it then writes the same document without the DRAFT marks.
--check rebuilds the document in the mode the committed JSON records and compares: the text and the plan byte for
byte, the JSON with every float within 1e-9 relative (the CAL2 fit's lstsq differs in the last bits between BLAS
builds, the 2026-11-17 numdrift record), everything else exact.

Building it computes the CAL2 refit under v7 (the fit gate_v7's decode clause computes, already computed and disclosed
by the v7 build) and v7's price of every blind cell; it reads no page of the three blind models (none exists) and does
not re-score the SEEN decode blind."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7bcommon as V  # noqa: E402

REPO = V.REPO
REL = 1e-9


def labels() -> dict:
    return {
        "time": "BLIND: the decode call, its tiles and its instrument have no page on any of the three models",
        "pages": "SYNTHETIC (decode_synth.py's frozen rules; config shapes only, synthetic weights)",
        "per_model": {
            "mixtral-8x22b": ("SEEN-MODEL: R3-tile pages (2026-09-28, rentals 2 to 6) and vLLM's 128x128 prefill tile as "
                              "C2's held-out CAL model; no decode page"),
            "qwen2-57b-a14b": ("SEEN-MODEL: R3-tile pages (2026-09-28 onward), vLLM's prefill tiles as a CAL model (A, A2, "
                               "S1, C3), and vLLM/SGLang fused_experts timed at T 1 to 128 on H200 (2026-08-22 to "
                               "2026-09-10, vLLM's tuned E=64,N=2560 H200 config) and A100 (2026-08-28 a100-cross-card, "
                               "vLLM 0.27.1; no A100 file for E=64,N=2560 bf16 in either local vLLM tree, so vLLM's "
                               "default config, its decode tiles, on another card, span-timed, not this apparatus' GRR "
                               "decode page); no GH200 decode page and no decode-stage page"),
            "granite-3.0-1b-a400m": ("SEEN-MODEL: R3-tile pages (rentals 1 to 3: launch floor, census, floor counters); "
                                     "no decode page"),
        },
        "everything published": "SEEN",
    }


def models_block() -> dict:
    cen = V.census()
    none = [m for m, x in cen.items() if x["verdict"] == "NO DECODE PAGE"]
    return {
        "chosen": list(V.MODELS),
        "why": ("the decode stage's three reserved models (docs/FINDINGS.md, Decode closes after v6d: 'The three "
                "reserved models remain untimed at decode'; addendum 4's names, addenda 5 and 6's new blind): no decode "
                "page of any has been timed; their page lists were registered on 2026-11-04 (addendum 5), before v7 "
                "existed, so the cells are not chosen with v7 in view; each fits one GH200 NATIVE (weights under 5 GB, "
                "below); the old decode blind's routing classes and T grid carry over unchanged"),
        "footprint (COMPUTED, d5common.footprint)": {m: V.D5.footprint(m) for m in V.MODELS},
        "census (COMPUTED: every model in moe.spec, toy left out)": cen,
        "every model with no decode page": none,
        "not chosen": {
            "deepseek-v3": ("no decode page; 22.6 GB per layer NATIVE fits; not a reserved model; one GH200 row of 2026-09-25 "
                            "(alpha-G pin probe, T 4096, 128x64x64) times its layer at another tile; design-v7 7.3 "
                            "proposed it for a combined prefill and decode blind; adding it would add units and cells "
                            "chosen after v7, so it is left for the owner (open question)"),
            "deepseek-v3-tp4, deepseek-v3-tp8, qwen2-57b-a14b-tp8": ("no decode page; tp shards of the models above, "
                                                                     "not reserved"),
            "mixtral-8x7b-tp2, -tp4, -tp8": ("PC timed their w2 GEMM on vLLM's 64x128x64 w4 s3 decode tile (pc-a.json, "
                                             "A-TP2/4/8-G128): a decode-tile page, so excluded under the strict reading"),
        },
        "unseen scan (COMPUTED: published sessions dated 2026-11-04 or later, check_unseen's pattern)": V.unseen_scan(),
        "unseen scan reading": ("two kinds of text hit, no page: session/setup_vm.log's git fetch listing of the old run "
                                "branch run-gh200-8x22b-2026-09-28 (an R3-tile session), as d6_build/check_unseen.out "
                                "read through PC2; and the DS, PC, PC2 and RR session READMEs' sentence that no file "
                                "names or loads the three models; no file name names a reserved model"),
        "labels": labels(),
    }


def cells_block() -> dict:
    b = V.add6_blind()
    by = {}
    for m in V.MODELS:
        doc = json.loads((REPO / f"{V.PAGES}/{m}-main.json").read_text())
        roles = {}
        for c in doc["cells"]:
            roles[c["role"]] = roles.get(c["role"], 0) + 1
        by[m] = {"main cells": len(doc["cells"]), "T": sorted({c["T"] for c in doc["cells"]}), "roles": roles}
    return {
        "structure": ("the old decode blind's, exactly (decodeB-blind's unit form, addendum 5's pages from decode_synth's "
                      "frozen generators): one GRR main page per model (nodrop) and one GR1 control page on the two "
                      "models with matched control cells (drop group gr1), T 1, 2, 4, 8, 16, 32, 64, 96, 128, the "
                      "routing classes BALANCED, UNIFORM, UNIFORM-UNSHUFFLED, LADDER, LADDER-BOTTOM and MINCOVER (the "
                      "ladder rungs scaled to each model's E and k by the generator, as on the old pages), the NATIVE "
                      "arm, shuffle seed 20261014, the 1710 MHz lock at G 1 and 700 W; the old pages' T 3 rows and "
                      "TRACE-DW cells were Qwen1.5-specific (its trace), as on every other old page"),
        "per model": by,
        "units": b["units"], "drop_order": b["drop_order"], "pages": b["pages"], "plan_units": b["plan_units"],
        "descriptive": ("addendum 2's second-tile cells (an expert past BLOCK_M rows) are DESCRIPTIVE, out of E1-DEC, "
                        "read as SIGMA-D; their list is the predictions' descriptive_cells"),
        "source": f"docs/registered/{V.ADD6}.json, block '{V.ADD6_BLOCK}' (addendum 5's, unchanged)",
    }


def build(draft: bool = True, pred: dict | None = None) -> dict:
    pred = V.predictions() if pred is None else pred
    fz = pred["frozen"]
    plan = V.plan_text(fz, draft)
    gate = V.gate_registration()
    status = ("DRAFT, NOT REGISTERED: may be committed for real only after the v7 gate registration "
              f"({V.GATE_NAME}) is public and gate_v7.py reads DECODE: GO on the published pages; until then the plan "
              "carries a DRAFT line and scripts/vm_run_v7b.sh refuses it" if draft else
              f"REGISTERED BEFORE ANY PAGE of the three models, after the v7 gate read DECODE: GO ({V.GATE_OUT})")
    return {
        "registered": V.DATE,
        "name": V.NAME,
        "draft": draft,
        "status": status,
        "design": ("design-v7 (moe-kernels-work 2026-11-17 DESIGN.md, section 7, the blind proposal) and the owner's "
                   "decisions of 2026-11-17: one blind GH200 rental for a regime that reads GO at the v7 gate; this "
                   "one tests v7 at vLLM's decode tiles on models no decode page has timed, with the old decode "
                   "blind's cells and the bars registered for these cells (owner, D4-1), changing nothing"),
        "tool": ("scripts/scoring/v7blind/register_v7_blind.py --write [--final] / --check (this file and the plan), "
                 "scripts/scoring/v7blind/score_blind_v7.py <results tree> <out dir> (the scorer), "
                 f"{V.LAUNCHER} (the launcher), scripts/{V.DRIVER} (the pinned decode driver, unchanged)"),
        "what": ("E1-DEC BLIND under model v7: v7 with every switch on (D6FIX at decode; RR is not defined at decode) "
                 "at its CAL2 refit, frozen before any page, on three models no decode page has timed"),
        "models": models_block(),
        "cells": cells_block(),
        "bars": dict(V.BARS, source=V.BARS_SOURCE),
        "statistic": "e_dec = (T_v7 - T_meas) / T_meas per valid NATIVE main cell, 0 at or under 1 us absolute (dcommon.e_dec)",
        "certification": V.CERT_RULE,
        "scope line": V.D3.SCOPE,
        "model v7 (frozen)": {
            "module": "scripts/r3_timing_model_v7.py",
            "as frozen by": (f"the v7 gate registration {V.GATE_NAME} (its freeze: {len(gate['freeze']['files'])} files "
                             f"and constants_sha256 {gate['freeze']['constants_sha256']}); this registration holds that "
                             "freeze and adds its own files (freeze, below)"),
            "switches": V.V7.V7.as_dict(),
            "decode branch": ("v6d's law at its registered terms; D6FIX re-prices a naive cell v6d prices by v4d''s "
                              "naive GEMMs at the CAL2 fit v4d' selects with v4d' on every naive cell"),
            "CAL2 refit under v7 (COMPUTED; gate_v7.compute_decode's fit, the same code path)": pred["CAL2 refit under v7"],
            "D6FIX constants (COMPUTED)": fz["model"]["D6FIX"],
        },
        "predictions": {
            "label": ("COMPUTED by v7 before any page of the three models (finalise_decode.frozen and .predict at the "
                      "CAL2 refit under v7, d5common.blind5's pages); the scorer reads them from this file"),
            "not_priced": pred["not_priced"], "descriptive_cells": pred["descriptive_cells"],
            "D6FIX rung (DESCRIPTIVE)": pred["D6FIX rung (DESCRIPTIVE)"], "keys": pred["keys"],
            "frozen": fz, "cells": pred["cells"],
        },
        "rental": dict(V.rental(), plan=V.PLAN, plan_sha256=hashlib.sha256(plan.encode()).hexdigest(),
                       driver=f"scripts/{V.DRIVER} (pinned, unchanged)",
                       launcher=(f"{V.LAUNCHER}: {V.PARENT_LAUNCHER} (RR's, with the apt-lock fix MOE_VM_APT_BUDGET_S "
                                 "and the prepare lock handling) with three changes marked V7B: the decode driver is "
                                 "its only session driver and the default, and start refuses a plan with a DRAFT line"),
                       lock="1710 MHz at G 1 (G4 1705), 700 W (prelude), as every decode rental",
                       guardian=("outside the repository (moe-kernels-work 2026-11-17 guardians/tV7B_*): deadline launch "
                                 "+ 125 min, hard cap launch + 140 min, terminate on the cap")),
        "disclosures": [
            ("the owner's reopening of 2026-11-17: decode closed after v6d under the stopping rule of 2026-11-10 and "
             "reopened for v7 with every decode page, both decode blind failures and both NO-GO gates in view (a "
             "forking path the paper must disclose)"),
            ("the gate outcome is known before this registration: design-v7 priced the decode gate pages (CAL2 GO; "
             "re-score about 3.19% with D6FIX), and the v7 build computed CAL2 under v7 (GO, 1.60% rms) and the "
             "re-score (3.19%) before the gate registration; this draft recomputes the CAL2 refit (the same numbers) "
             "and does not re-score the SEEN blind"),
            ("D6-24 was found on SEEN pages after addendum 6's gate, and D6FIX chosen knowing it moves the 13 seen "
             "v4d'-branch cells from 13.2% to 1.36% rms; on this blind's 143 cells D6FIX moves none (every naive main "
             "cell takes the law branch), so v7's blind prices equal v6d's at the same CAL2 fit"),
            ("Qwen2-57B-A14B's decode-size calls were timed before the decode stage: vLLM and SGLang fused_experts at "
             "T 1 to 128 on H200 (tuned config) and on A100 (2026-08-28 a100-cross-card, vLLM's default config: its "
             "decode tiles on another card); none on a GH200, none under this apparatus, and no decode constant was "
             "fitted on them"),
            ("every v7 decode constant was read or fitted on SEEN pages (PC, PC2, DS, CAL2); v6d's choices were made "
             "after the old decode blind was seen"),
            ("the bars are the ones registered for these cells before any page: addendum 5 line 9 'new blind bars "
             "(owner, D4-1): rms 3% per model, worst cell 8%, |mean| 1.5%', carried unchanged by addendum 6 (2026-11-10, "
             "line 159); the cells were registered on 2026-11-04 (addendum 5) and again under v6d (addendum 6), never run"),
            ("v7's re-score of the SEEN decode blind (about 3.19% pooled; Granite-3B 5.85% rms, Qwen1.5 with D6FIX "
             "about 2%) suggests a hard test at 3% per model (ESTIMATE below)"),
        ],
        "expected (ESTIMATE, judgment, not computed)": (
            "v7 prices these cells exactly as v6d did at the same fit; addendum 6 estimated, at these bars, the pooled "
            "bar holding about 0.05, at least one model certified about 0.05, per-model odds at a model's own bars "
            "Qwen2-57B about 0.45, Mixtral-8x22B about 0.35, Granite-1B about 0.05 (addendum 6 'expected', quoted, not "
            "recomputed)"),
        "unchanged": ("every earlier registration, page, verdict and pinned file; the decode driver and setup_vm.sh; "
                      "the 1710 lock and 700 W; the v7 gate registration and its bars"),
        "freeze": V.freeze(),
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
    L = [f"{head} {d['registered']}, before any page of Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M: model v7's "
         f"decode blind, {d['name']}", f"status: {d['status']}", f"design: {d['design']}", f"tool: {d['tool']}", ""]
    for k, v in d.items():
        if k in ("registered", "name", "draft", "status", "design", "tool", "predictions", "freeze"):
            continue
        _flat("", {k: v}, L)
    p = d["predictions"]
    fz = p["frozen"]
    L += ["", f"PREDICTIONS: {p['label']}",
          f"  constants: {json.dumps(fz['constants'])}, rho {fz['rho_form']}, D6FIX {json.dumps(fz['model']['D6FIX'])}",
          f"  bars: {json.dumps(fz['bars'])}", f"  not priced: {p['not_priced']}; descriptive: {len(p['descriptive_cells'])}",
          f"  D6FIX rung: {p['D6FIX rung (DESCRIPTIVE)']['reading']}",
          "  model / label / T / path / tile / v7 us (3 decimals); DESC = DESCRIPTIVE second-tile cell"]
    desc = set(p["descriptive_cells"])
    for r in p["cells"]:
        ms = r["ms"]["v4d"]
        L.append(f"    {r['model']} {r['label']} T{r['T']} {r.get('path')} {r.get('tile')} "
                 f"{'NOT PRICED' if ms is None else f'{ms * 1e3:.3f}'}"
                 + (" DESC" if f"{r['model']}/{r['label']}" in desc else ""))
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
    ap.add_argument("--final", action="store_true", help="write the registration for real (needs the gate's DECODE: GO)")
    ap.add_argument("--root", type=Path, default=REPO, help="where to write or read the files (tests)")
    a = ap.parse_args(argv)
    jpath, tpath, ppath = a.root / V.REG_JSON, a.root / V.REG_TXT, a.root / V.PLAN
    if a.write:
        if a.final and not V.gate_reads_decode_go():
            print(f"REFUSED: {V.GATE_OUT} holds no 'DECODE: GO' line: the registration stays DRAFT")
            return 2
        doc = build(draft=not a.final)
        for p in (jpath, ppath):
            p.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        tpath.write_text("\n".join(text(doc)) + "\n")
        ppath.write_text(V.plan_text(doc["predictions"]["frozen"], doc["draft"]))
        print(f"wrote {jpath.name}, {tpath.name}, {ppath.name} ({'DRAFT' if doc['draft'] else 'final'})")
        return 0
    if not jpath.is_file():
        print(f"{jpath.name}: missing")
        return 1
    want = json.loads(jpath.read_text())
    doc = json.loads(json.dumps(build(draft=bool(want.get("draft", True))), default=str))
    ok = True
    bad = close(doc, want)
    print(f"{jpath.name}: {'same' if not bad else 'DIFFERS: ' + '; '.join(bad[:5])}")
    ok &= not bad
    for p, t in ((tpath, "\n".join(text(doc)) + "\n"), (ppath, V.plan_text(doc["predictions"]["frozen"], doc["draft"]))):
        same = p.is_file() and p.read_text() == t
        ok &= same
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
