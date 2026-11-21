#!/usr/bin/env python
"""Writes the decode stage's second addendum, docs/registered/2026-10-28-decode-addendum2-gh200.{json,txt},
before any decode page. 2026-10-14-decodeC-cal-gh200, 2026-10-14-decodeB-blind-gh200 and
2026-10-20-decode-tilesource-gh200 stay byte-identical (this file pins them).

    python scripts/scoring/decode/register_decode_d2.py --write [--out-dir docs/registered]
    python scripts/scoring/decode/register_decode_d2.py --check

Every number is computed here from committed files (the registered decode pages, r3_decode.geometry,
v4d at the registrations' DESIGN-NOMINAL constants, regdecode's power simulation, the generated plan) or
is a typed-in input with its provenance. `--check` recomputes and compares the JSON (code_pins excluded,
as every registration) and the text; exit 1 on any difference. No decode page exists when this runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402
import regdecode as RG  # noqa: E402
import register_decode as RGD  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
r6 = DC.r6
TILESOURCE = "2026-10-20-decode-tilesource-gh200"
FROZEN = tuple(f"docs/registered/{n}.{x}" for n in (DC.NAMES["cal"], DC.NAMES["blind"], TILESOURCE) for x in ("json", "txt"))
D2_CODE = ("scripts/scoring/decode/d2common.py", "scripts/scoring/decode/score_cal_d2.py",
           "scripts/scoring/decode/score_blind_d2.py", "scripts/scoring/decode/finalise_decode_d2.py",
           "scripts/scoring/decode/register_decode_d2.py", D2.CAL_PLAN_D2)
#: the frozen files the wrappers run, and A2's score directory (the tile source) as committed
RUN_FROZEN = ("scripts/scoring/decode/score_cal.py", "scripts/scoring/decode/score_blind.py",
              "scripts/scoring/decode/register_decode.py", FN.CAL_PLAN) + tuple(
    f"scripts/scoring/tileA2/{n}.score.json" for n in ("census", "floor", "bytes", "rep", "e1cal"))
#: the rule's inputs no earlier decode registration pins (MODEL_CONFIGS, default_config; review m8)
RULE_INPUTS = ("moe/spec.py", "moe/bench/tile_resolve.py")
#: the operator's power step (review m2), run on the laptop after the prelude's push
POWER_STEP = ("git fetch origin <run-branch> && D=$(mktemp -d) && mkdir -p \"$D/session\" \"$D/results\" && "
              "git show origin/<run-branch>:vm/session/power-set.txt > \"$D/session/power-set.txt\" && "
              "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/scoring/decode/d2common.py --power \"$D/results\"")
SKIP_TXT = {"code_pins", "excluded_cells", "nominal_sigma_d"}

DESIGN = ("decode-ready READINESS.md (2026-10-28, read-only against 0144010d), recommendation (b) followed exactly, and the "
          "owner's decisions of 2026-10-28: D2-1 decision DD6 superseded (decode CAL runs now, tile rental B is not run; "
          "decode uses A2's constants through the first addendum, its gate reading GO); D2-2 the cells where an active "
          "expert holds a second M-tile are DESCRIPTIVE (out of the fit, the rho selection and every E1 bar; read as "
          "SIGMA-D against v4d's sigma-1 price and an L2-served estimate); D2-3 700 W stated and read back from the "
          "session's power-set.txt, else NOT SCORED; D2-4 the knee rows (p 5.75, 3.75) DESCRIPTIVE and the two byte "
          "units nodrop; D2-5 the rental sequence: CAL, then on GO the mechanical finalisation and the blind rental; and "
          "after the build review (build-D2-review REVIEW.md, GO-WITH-FIXES): M1 the certified domain stated verbatim, m9 "
          "SIGMA-D's reading bands and NOT READ rule, m3 the blind bars from the kept cells' replicate noise")


def cells_text(rows: list[dict]) -> str:
    return "; ".join(f"{r['label']} (T {r['T']}, {r['tile']}, {r['experts_over_BLOCK_M']} over BM {r['BLOCK_M']}, "
                     f"largest {r['largest_count']})" for r in rows)


def excluded() -> dict:
    out = {}
    for part, models in (("cal", DC.CAL_MODELS), ("blind", DC.BLIND_MODELS)):
        for kind in ("main", "ctl"):
            rows = D2.excluded_cells(models, kind)
            if rows:
                out[f"{part} {kind}"] = rows
    return out


def nominal_sigma_d(ex: dict) -> list[dict]:
    """DESIGN-NOMINAL (the registrations' nominal constants, rho fixed): v4d and the L2-served estimate per
    DESCRIPTIVE main cell, for scale only (never a prediction)."""
    prm = DC.nominal_params("fixed")
    out = []
    for part in ("cal main", "blind main"):
        for r in ex[part]:
            c = next(x for x in DC.page_doc(r["model"], "main")["cells"] if x["label"] == r["label"])
            a, b = DC.price_cell(r["model"], c, prm), D2.served_price(r["model"], c, prm)
            out.append({"model": r["model"], "label": r["label"], "v4d_nominal_ms": r6(a["ms"]),
                        "served_nominal_ms": r6(b["ms"]), "served_over_v4d": r6(b["ms"] / a["ms"])})
    return out


def power_amended() -> dict:
    """regdecode's power simulation (DESIGN-NOMINAL, sigma ASSUMED) on the amended E1 cell sets."""
    out = {}
    for part, models in (("cal", DC.CAL_MODELS), ("blind", DC.BLIND_MODELS)):
        keys = D2.excluded_keys(models)
        cells = [c for c in RG.nominal_cells(models) if (c["model"], c["label"]) not in keys]
        out[part] = RG.power(cells, DC.BARS_PROVISIONAL, RG.SCENARIOS)["p_holds"]
        out[f"{part}_cells"] = len(cells)
    out["label"] = ("DESIGN-NOMINAL predictions, sigma ASSUMED, the shared-memory occupancy (as decodeC-cal's table, I8); "
                    "the cells of E1 under this addendum. The constants are held fixed (no refit): the blind rows are the "
                    "frozen-constant case, but the CAL rows do not model the CAL refit, which absorbs a uniform bias, so "
                    "'+1%' and '+2% everywhere' overstate the CAL risk (review m7)")
    return out


def build() -> dict:
    ex = excluded()
    n_cal = len(DC.page_doc("olmoe-1b-7b", "main")["cells"]) + len(DC.page_doc("mixtral-8x7b", "main")["cells"])
    cal_min = json.loads((REG / f"{DC.NAMES['cal']}.json").read_text())["rental"]["units_min"]
    blind_min = json.loads((REG / f"{DC.NAMES['blind']}.json").read_text())["rental"]["units_min"]
    plan = (REPO / D2.CAL_PLAN_D2).read_text()
    return {
        "registered": f"{D2.DATE}, before any decode page",
        "name": D2.NAME,
        "status": "REGISTERED BEFORE ANY DECODE PAGE: the decode stage's second addendum; 2026-10-14-decodeC-cal-gh200, "
                  "2026-10-14-decodeB-blind-gh200 and 2026-10-20-decode-tilesource-gh200 stay byte-identical (pinned here)",
        "design": DESIGN,
        "tool": "scripts/scoring/tileA2/decode_gate_a2.py scripts/scoring/tileA2 (GO), then scripts/scoring/decode/"
                "score_cal_d2.py <tree> <out> --tile-scores scripts/scoring/tileA2; on GO finalise_decode_d2.py <out> "
                "<finalised.json> --plan-out scripts/plans/rentalD-blind-2026-10.plan; after the blind rental "
                "score_blind_d2.py <tree> <finalised.json> <out> (all new; the frozen scorers run inside, unchanged)",
        "scope (owner, 2026-10-28, review M1)": D2.SCOPE,
        "DD6 superseded (owner, 2026-10-28, D2-1)": "decodeC-cal's DD6 ('decode runs after the tile stage, rented after tile "
            "rental B') no longer holds: tile rental B2 never runs (A2 STOP) and B3 is not registered (FINDINGS.md 'Tile stage: "
            "scope decision (2026-10-28)'). The decode CAL rental runs now. Its tile source stays the first addendum's: A2's "
            "score directory scripts/scoring/tileA2/, read by score_cal.tile_params unchanged, decode_gate_a2.py reading GO",
        "descriptive cells (owner, 2026-10-28, D2-2)": {
            "rule": "a cell is DESCRIPTIVE when its path is not naive and some active expert holds more than BLOCK_M rows at "
                    "vLLM's default config for its T (r3_decode.geometry multi_tile_experts > 0): v4d's G 1 schedule gives "
                    "that expert a second M-tile whose CTAs re-read its weight slab from DRAM (sigma 1, decodeC-cal scorer "
                    "ambiguity 4). Computed from the registered pages' counts and that rule alone, before any decode page",
            "out of": "the CAL fit (v4d.fit_constants), the rho selection (sub-wave rms), E1-CAL-IN-SAMPLE, the replicate's "
                      "E1 row, the rivals' CAL training rows (line, OLS), blind E1-DEC pooled and per model, the "
                      "certification and the rival exclusions, and the replicate's sigma_page and the tightened blind bars (owner, "
                      "2026-10-28, review m3: from the kept cells' pairs only; the every-cell figure printed beside, "
                      "DESCRIPTIVE)",
            "kept in": "G3-D shuffle and label (each pair holds the same counts or the same multiset on both sides, so the "
                       "re-read cancels in d_us; G3-D label's MIN-T64 against L-T64-A<k> pair stays as registered), G3-D "
                       "fill (BALANCED cells only; no DESCRIPTIVE cell is BALANCED), GD2, GD3 (instrument readings, no model in "
                       "them), V-ALIGN, GD5, COPIES and G1 (a "
                       "DESCRIPTIVE cell's refusal is still a STOP: the decode rules are not vLLM's)",
            "cal_main": len(ex["cal main"]), "cal_main_of": n_cal,
            "blind_main": len(ex["blind main"]),
            "ctl": "the CAL ctl pages' second-tile cells are read by BYTES-D2 (the CAL plan's dn counter pages); the "
                   "blind ctl pages' second-tile cells are listed only: the blind plan has no counter page (review m5). "
                   "GD2 reads every matched cell on both",
        },
        "excluded_cells": ex,
        "SIGMA-D (DESCRIPTIVE)": {
            "rows": "per valid NATIVE DESCRIPTIVE main cell (CAL at the amended fit's constants and rho, blind at the "
                    "finalised ones): the measured time, v4d's sigma-1 price, the L2-served estimate, e of each "
                    "(dcommon.e_dec), served / sigma-1, and position = (v4d - meas) / (v4d - served) (0 at sigma 1, 1 at "
                    "the served estimate); per model the C5.stats of both e and the median position",
            "L2-served estimate": "v4d's price of the same cell with every CTA of an expert's second or later M-tile reading "
                                  "no weight slab (its slab served from L2 at zero DRAM bytes; its activation share "
                                  "kept), everything else v4d's unchanged: the same live CTAs and launch order at G 1, "
                                  "the floor, the partial-wave tail under CORES, rho, the dead term, B_other and the "
                                  "constants (d2common.served_price: r3_timing_model.window's G 1 reads, slab removed where "
                                  "pid_m // npn is not the expert's first M-tile)",
            "never": "fitted, selected, a bar, or a STOP clause",
            "reading (owner, 2026-10-28, review m9)": f"position under {D2.POS_CUTS[0]:.4f} (1/3) FULL-REREAD (the "
                "second-tile re-reads come from DRAM, sigma 1 holds), over {:.4f} (2/3) L2-SERVED (served from L2), between "
                "MIXED; per cell, and per model on the median position over its readable cells. NOT READ when the "
                "sigma-1 and served prices are under {:g} us or {:g}% of the sigma-1 price apart (a 3% price error, "
                "E1's rms bar, then moves position by at most 0.3, about one band); a model with no readable cell is NOT "
                "READ. A position below 0 or above 1 keeps its band and is reported as measured: the fit's own error "
                "enters position (the build review's synthetic kept-cell defect read 1.24)".format(
                    D2.POS_CUTS[1], D2.MIN_GAP_US, 100 * D2.MIN_GAP_REL),
            "BYTES-D2": "the CAL ctl counter pages' second-tile cells: counted dram__bytes_read per GEMM against v4d's "
                        "reads and the served reads, both computed on the laptop with A2's tile parameters (the page's own "
                        "v4d_reads is an error on the wgmma T 128 cells, review m1; kept beside); share_not_seen = "
                        "(v4d - counted) / (v4d - served)",
        },
        "nominal_sigma_d": nominal_sigma_d(ex),
        "power (owner, 2026-10-28, D2-3)": {
            "limit": f"{D2.POWER_LIMIT_W:g} W, GPU scope (-sc 0), the owner's 2026-09-26 decision; the pinned decode driver "
                     "sets it (POWER_LIMIT_W=700, gh200_decode_session.sh) and refuses the prelude unless power.limit reads it",
            "check": "the scorer wrappers read the session's power-set.txt (nvidia-smi -q -d POWER after -pl, written by the "
                     "pinned driver and pushed with the session; <tree>/session or <tree>/../session, exactly one): the "
                     f"GPU Power Readings section's Current Power Limit must read {D2.POWER_LIMIT_W:g} W within "
                     f"{D2.POWER_TOL_W:g} W; else the decode readings are NOT SCORED (the CAL score is a STOP, the blind "
                     "score certifies nothing)",
            "driver": "no driver change: the pinned driver already writes power-set.txt; vm_run.sh and docs/LAMBDA.md "
                      "are unchanged",
        },
        "knee rows (owner, 2026-10-28, D2-4, DESCRIPTIVE)": "v4d with r3_timing_model.P_KNEE 5.75 and 3.75 (the tile stage's "
            "p_128 and round 5's p_64) in place of v3's 14, at the same constants, on every E1 cell: e against the "
            "provisional bars and the mean and largest price shift per tile, beside the registered prediction; never "
            "a bar, a selection or a STOP clause",
        "byte units (owner, 2026-10-28, D2-4)": {
            "plan": D2.CAL_PLAN_D2, "plan_sha256": RGD.sha256(REPO / D2.CAL_PLAN_D2),
            "change": "the two decodebytes (dn) units nodrop=1 in place of drop-group=ncu (d2common.cal_plan_d2 over the "
                      "frozen generator); every other unit line byte-identical to scripts/plans/rentalD-cal-2026-10.plan; "
                      "minutes unchanged",
            "units": [ln for ln in plan.splitlines() if ln and not ln.startswith("#")],
            "drop_order": "last first: rep, decl, e240",
            "scoring": "BYTES-D and BYTES-D2 stay DESCRIPTIVE; the scorer reads decodeC-cal's unit flags, so a lost dn page "
                       "is not a STOP",
        },
        "stop rule (as amended, CAL)": "decodeC-cal's stop rule (dcommon.stop_rule, unchanged) read on the amended score: "
            "the fit, E1-CAL-IN-SAMPLE and NOT PRICED without the DESCRIPTIVE cells; G3-D shuffle and label on every valid "
            "cell at the amended fit; GD2 on every matched cell; the nodrop units, V-ALIGN, GD5 and COPIES as registered; "
            "and a STOP when the power check does not read 700 W. Else GO",
        "blind rule (as amended)": "decodeB-blind's rule on the cells left: E1-DEC pooled and per model at the finalised "
            "bars, a model CERTIFIED when its own bar and the pooled bar hold; GD2 and G3-D voids as registered, on every "
            "cell",
        "rentals (owner, 2026-10-28, D2-5)": {
            "sequence": "the CAL rental (this plan); if its stop rule as amended reads GO, finalise_decode_d2.py "
                        "mechanically, then the blind rental (finalise's generated plan); on a STOP the blind rental does "
                        "not run",
            "cal": D2.rental(cal_min), "blind": D2.rental(blind_min),
            "worst case": "the CAL unit caps sum to 355 min (decodeC-cal); the guardian's hard cap ends the rental first",
            "operator power step (review m2)": "each rental: after the prelude's push, fetch the run branch and run "
                f"`{POWER_STEP}` from the repository root; terminate the instance unless it prints READ (exit 0). The "
                "pinned driver refuses on power.limit only; this reads the -q GPU scope the scorer reads, before any "
                "page is paid for",
        },
        "power_amended (DESIGN-NOMINAL)": power_amended(),
        "unchanged": "the lock 1710 and G4 1705, the instruments, the pages, the bars, the census rule, the tile source, the "
                     "card constants, every pin of the three earlier decode registrations, and every registered verdict",
        "notes": [
            "score_cal.py:37 labels the tile source \"tile rental A's selections\" when given A2's directory "
            "(readiness review I6): a cosmetic mislabel left in place because the first addendum registers score_cal.py "
            "as frozen and byte-identical; score_cal_d2.py prints the correct label",
            "the plan's tilecensus unit runs the pinned tile_census.py and is expected to exit 3 on deepseek-v2-lite "
            "(readiness I7); no decode scorer reads it",
            "moe/spec.py (MODEL_CONFIGS) and moe/bench/tile_resolve.py (default_config), the excluded-cell rule's "
            "inputs, were in no earlier decode registration's pins (pre-existing, review m8); this addendum pins them "
            "(no frozen file changes)",
            "the blind score carries a DESCRIPTIVE row scoring every rival on every cell (RIVALS_ALL_CELLS) beside the "
            "registered rival exclusions on the kept cells",
            "decodeC-cal's power table predates the census and assumes sigma 1 everywhere (I8); power_amended is the same "
            "simulation on the amended cell sets, DESIGN-NOMINAL",
        ],
        "seen": [
            "no decode page exists; every published page is SEEN",
            "the readiness review's COMPUTED and INFERENCE readings (decode-ready/work): which cells hold a second M-tile, "
            "its served-from-L2 ratios 0.36 to 0.65 at DESIGN-NOMINAL T0_GR 25 us, and the inference that OLMoE's "
            "re-reads are served and Mixtral's miss (ROUND5.md, S1 M3, C3-R2): the reason for D2-2; the rule itself "
            "reads counts only",
        ],
    }


def text(d: dict) -> list[str]:
    r = d["rentals (owner, 2026-10-28, D2-5)"]
    L = [f"REGISTERED {d['registered']}: the decode stage (a second addendum), {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", ""]
    L.append(d["scope (owner, 2026-10-28, review M1)"])
    for part in ("cal", "blind"):
        x = r[part]
        L.append(f"{part} rental: {x['units_min']} min of units, about {x['wall_min']} min wall with {x['boot_setup_min']} "
                 f"min of boot and setup: ${x['cost_usd']:.2f} at ${D2.RATE}/h; deadline launch + "
                 f"{x['deadline_after_launch_min']} min; guardian hard cap launch + {x['hard_cap_after_launch_min']} min "
                 f"(${x['cost_at_hard_cap_usd']:.2f} at most)")
    L.append("")
    L.append("DESCRIPTIVE cells (by identity):")
    for part, rows in d["excluded_cells"].items():
        for m in dict.fromkeys(x["model"] for x in rows):
            L.append(f"  {part} {m}: {cells_text([x for x in rows if x['model'] == m])}")
    L.append("DESIGN-NOMINAL served / v4d (for scale only): " + "; ".join(
        f"{x['model']} {x['label']} {x['served_over_v4d']:.3f}" for x in d["nominal_sigma_d"]))
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        RGD._flat("", {k: v}, L)
    return L


def pins() -> dict:
    paths = tuple(dict.fromkeys(RGD.PINNED + tuple(json.loads((REG / f"{TILESOURCE}.json").read_text())["code_pins"])
                                + RUN_FROZEN + RULE_INPUTS + FROZEN + D2_CODE))
    return {p: RGD.sha256(REPO / p) for p in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    doc = build()
    doc["code_pins"] = pins()
    jpath, tpath = a.out_dir / f"{D2.NAME}.json", a.out_dir / f"{D2.NAME}.txt"
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    if a.check:
        strip = lambda x: {k: v for k, v in x.items() if k != "code_pins"}  # noqa: E731
        same_j = jpath.exists() and strip(json.loads(jpath.read_text())) == strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        for p, same in ((jpath, same_j), (tpath, same_t)):
            print(f"{p.name}: {'same' if same else 'DIFFERS'}")
        return 0 if same_j and same_t else 1
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
