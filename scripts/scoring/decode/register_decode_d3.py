#!/usr/bin/env python
"""Writes the decode stage's third addendum, docs/registered/2026-10-29-decode-addendum3-gh200.{json,txt}, and
its settling pages (docs/registered/2026-10-29-decode3-pages/), before any CAL2 page. Every earlier decode
registration (decodeC-cal, decodeB-blind, the tile-source addendum, addendum 2) stays byte-identical (pinned here).

    python scripts/scoring/decode/register_decode_d3.py --write [--out-dir docs/registered]
    python scripts/scoring/decode/register_decode_d3.py --check

Every number is computed here from committed files (the published decode CAL session and its score, v4d and v4d',
the registered pages, the generated plan) or is a typed-in input with its provenance (the owner's decisions and the
diagnosis's estimates, labelled). `--check` recomputes and compares the JSON (code_pins excluded, as every
registration), the text, the settling pages and the CAL2 plan; exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import dcommon as DC  # noqa: E402
import register_decode as RGD  # noqa: E402
import register_decode_d2 as RG2  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
r6 = DC.r6
V4DP = D3.V4DP
D3_CODE = ("scripts/r3_timing_model_v4dp.py", "scripts/r3_decode3.py", "scripts/dram_counter_route_decode3.py",
           "scripts/gh200_decode3_session.sh", "scripts/scoring/decode/d3common.py",
           "scripts/scoring/decode/score_cal_d3.py", "scripts/scoring/decode/finalise_decode_d3.py",
           "scripts/scoring/decode/score_blind_d3.py", "scripts/scoring/decode/register_decode_d3.py", D3.CAL2_PLAN)
FROZEN = tuple(f"docs/registered/{D2.NAME}.{x}" for x in ("json", "txt"))
SEEN_INPUTS = (D3.PUBLISHED_SCORE, f"{D3.PUBLISHED}/session/power-set.txt",
               "scripts/scoring/decode/cal.score.txt")
SKIP_TXT = {"code_pins", "pages", "source_pairs", "settle_units", "seen_reading", "reduction"}
DIAG = "~/code/moe-kernels-work/2026-10-29/diag-decode/DIAGNOSIS.md (offline, read-only against a398c6a0)"

DESIGN = ("the decode CAL STOP's offline diagnosis (" + DIAG + "), its fixes 1 to 3 and settling units M1 to M4, and "
          "the owner's decisions of 2026-10-29: D3-1 model v4d' = v4d plus fixes W and R (0 fitted constants; the diagnosis's A dropped by D3-10), in a new "
          "module that reduces to v4d exactly; D3-2 G3-D label DESCRIPTIVE (reported, not a stop clause), every other stop "
          "clause unchanged; D3-3 the blind bars and per-model certification as registered, the diagnosis's odds stated "
          "as estimates; D3-4 CAL2 = the registered CAL plan plus the settling units, DESCRIPTIVE readings registered "
          "before pages, never used to refit v4d' before the CAL2 reading; a D3 driver copy for the two capabilities the "
          "pinned driver lacks; D3-5 new scorer wrappers over the frozen d2 ones, the decode blind registration frozen as "
          "is apart from v4d' replacing v4d; and the owner's decisions of 2026-11-03 on the build's open list: D3-6 the "
          "blind G3-D label void becomes DESCRIPTIVE too (reported per model, never a void; in the scope line), D3-7 fix R "
          "stays a time ratio on the whole w1 GEMM price, D3-8 M3's pad-0 control kept, D3-9 M1's per-tile rates or a "
          "remeasured A may enter a later model only under its own registration; and after the build review (build-Dcal2-"
          "review REVIEW.md, GO-WITH-FIXES) on 2026-11-03: D3-10 fix A DROPPED (review MAJOR-1), v4d' = W + R; D3-11 M3's "
          "shifted pad at least 52 rows (review m2), 52 chosen; and the review's mechanical fixes m1 (M3 refuses an "
          "unrecorded pad), m3 (M4's w2 knife edge), m4 (the CAL2 score's scope line), m5 (MEASURED on SEEN pages), m7 "
          "(the scope line excludes a dependence on which experts are active)")


def _tree() -> Path:
    return REPO / D3.PUBLISHED / "results"


def _published():
    import contextlib

    import score_cal as SC
    with contextlib.chdir(REPO):        # the published score names the tile source by its repo-relative path
        tp, src = SC.tile_params(Path("scripts/scoring/tileA2"))
    s = json.loads((REPO / D3.PUBLISHED_SCORE).read_text())
    sel = s["FIT"]["selected"]
    return s, tp, src, DC.V4D.params_from(s["FIT"]["forms"][sel], rho_form=sel, tile=tp)


def reduction() -> dict:
    """v4d' with every fix off, run through the addendum-2 CAL scorer on the published tree, against the published
    cal.score.json: byte for byte."""
    import score_cal_d2 as SC2
    _s, tp, src, _prm = _published()
    import contextlib
    with D3.v4dp(V4DP.NONE), contextlib.chdir(REPO):      # the published score names its tree repo-relative
        got = SC2.score(Path(D3.PUBLISHED) / "results", DC.registration(REPO, "cal"), tp, src)
    text = json.dumps(got, indent=1, default=str) + "\n"
    want = (REPO / D3.PUBLISHED_SCORE).read_text()
    pub = json.loads(want)
    same_rows = [r["pred_ms"] for r in got["rows"]] == [r["pred_ms"] for r in pub["rows"]]
    return {"verdict": "IDENTICAL" if text == want else "DIFFERS",
            "what": f"score_cal_d2.score under d3common.v4dp(fixes NONE) on {D3.PUBLISHED}/results against "
                    f"{D3.PUBLISHED_SCORE}, byte for byte",
            "predictions": len(got["rows"]), "predictions_identical": same_rows,
            "score_sha256": hashlib.sha256(text.encode()).hexdigest()}


def constants() -> dict:
    _s, _tp, _src, prm = _published()
    w = D3.w1_marginal(_tree(), prm)
    a = D3.align_slopes(_tree(), DC.registration(REPO, "cal"))
    ok_w = abs(w["scale"] / V4DP.W1_SCALE - 1) < 1e-12 and abs(w["counter_GBps"] / V4DP.W1_COUNTER_GBPS - 1) < 1e-12
    return {
        "R (w1 rate, 16x64x128w4s4)": {
            "value": V4DP.W1_SCALE, "counter_GBps": V4DP.W1_COUNTER_GBPS, "v4d_GBps": V4DP.W1_V4D_GBPS,
            "recomputed": {"scale": w["scale"], "counter_GBps": w["counter_GBps"], "v4d_GBps": w["model_GBps"],
                           "agrees": "READ (to 1e-12 relative)" if ok_w else "DIFFERS"},
            "source": f"MEASURED on SEEN pages: the published decode CAL session's two dn counter pages ({D3.PUBLISHED}/results/"
                      "*-dn-r3-counters/lock1710/r3d.json, ctl-page cells), kept cells of one (T, path) at the lowest and "
                      "highest w1 counted bytes (at least 100 MB apart): the counters' marginal rate d(counted bytes) / "
                      "d(gpu__time_duration) pooled (mean) over the 7 pairs, against v4d's own marginal rate on the same "
                      "pairs (d(counted bytes) / d(v4d's w1 GEMM price)) at the published amended fit's rho (scaled) and "
                      "A2's tile parameters; value = v4d / counters. 0 fitted",
            "source_pairs": w["pairs"]},
        "A (align loop): DROPPED, not in v4d' (owner, 2026-11-03, D3-10)": {
            "decision": "the diagnosis's third fix (0.96 us x ceil(numel / max(E, 32)) on the small-batch path) is not part "
                        "of v4d', its fingerprint (fixes), the blind finalisation or the expected in-sample figures",
            "why (review MAJOR-1, recomputed here, SEEN pages)": {
                m: {k: a[m].get(k) for k in ("cells", "iterations", "slope_us", "intercept_us", "B_other_slope_us",
                                              "slope_net_of_B_other_us", "slope_with_held_out_cells_us", "held_out")
                    if k in a[m]} for m in DC.CAL_MODELS} | {
                "mean_slope_us": a["mean_slope_us"], "mean_slope_with_held_out_us": a.get("mean_slope_with_held_out_us"),
                "reading": "the 0.96 us value rests on 3 Mixtral cells left out after the pages were seen (1.16 us with "
                           "them); v4d's own B_other already grows 0.95 us per iteration on Mixtral (0.20 on OLMoE), so "
                           "A double-counted it there (net slope -0.03 on Mixtral, 0.80 on OLMoE); and in-sample on the "
                           "published CAL, W + R reads rms 2.13% against W + R + A's 2.26%"},
            "where it goes": "the CAL2 settling unit M2 measures the align slope DESCRIPTIVELY (its reading prints the "
                             "diagnosis's 0.96 us beside, never priced); a later model may add an align term only under "
                             "its own registration, before any page it is scored on",
        },
        "W (naive order)": {
            "rule": "naive rows in vLLM's token-major launch order (r3_decode.cell_ids); a follower CTA is served in its "
                    "leader's wave (CTA // (132 x the census occupancy, 3 on 16x64x128)) or in a later wave while under "
                    f"{V4DP.L2_CUT_BYTES} bytes (60 MiB) of slabs have streamed since its leader; else it re-fetches",
            "constants": "0: the census occupancy MEASURED (A2's tile census), the 60 MiB cut SEEN (round 5)",
            "source": "the diagnosis's naive_rules.py rule 'wavecut', reproduced: L-T2-A8 w1 97.6 MB predicted against "
                      "85.9 counted, U-T2 134.3 against 133.3"},
    }


def seen_reading() -> dict:
    """v4d' on the published (SEEN) CAL pages, in-sample: what a CAL2 that reproduced them would read."""
    import score_cal_d3 as SC3
    _s, tp, src, _prm = _published()
    s = SC3.score(_tree(), DC.registration(REPO, "cal"), tp, src, settle=False)
    f = s["FIT"]
    return {"label": "SEEN, IN-SAMPLE, DESCRIPTIVE: v4d' fitted on the published decode CAL pages (every page SEEN); "
                     "not a prediction of CAL2 and not a registered verdict",
            "selected": f["selected"],
            "forms": {k: {x: v.get(x) for x in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms", "in_sample", "sub_wave")}
                      for k, v in f["forms"].items()},
            "E1_CAL_IN_SAMPLE": s["E1_CAL_IN_SAMPLE"], "G3D": {t: g.get("verdict") for t, g in s["G3D"].items()},
            "STOP_D3": {k: s["STOP"][k] for k in ("verdict", "why", "descriptive")},
            "SIGMA_D": {m: x["band"] for m, x in s["SIGMA_D"]["per_model"].items()}}


def pages() -> dict:
    return {"dir": D3.PAGES3, "files": {f"{m}-{k}.json": D3.reg_file(m, k) | {"what": D3.PAGE_WHAT[k]}
                                         for m, k in D3.PAGE_SPECS}}


def settle_text() -> dict:
    return {
        "M1 tiles (units m1: dt on both models)": "counter pages (the pinned decodebytes route) on main-page kept cells at "
            "two A on each of the T 64, T 96 and T 128 tiles (32x64x128w4s3, 32x128x64w4s3, 64x128x64w4s3). READS per "
            "model and tile the kept pair's marginal DRAM rate per GEMM (d counted bytes / d ncu time) against v4d''s "
            "(d counted bytes / d v4d' GEMM price at the CAL2 fit), and the ratio; and fix R's 7 pairs re-read on the CAL2 "
            "dn pages (R_replicate). It would replace the diagnosis's fitted K with measured per-tile rates IN A LATER "
            "REGISTRATION; here it changes nothing",
        "M2 wide (units m2: dw on both models)": "the same counter route with ncu's -k filter widened (dram_counter_route_"
            "decode3.py --kernels wide: moe_align_block_size*, count_and_sort_expert_tokens*, act_and_mul*, silu*, "
            "moe_sum* and the GEMMs; kernel replay, --cache-control all) on ctl-page cells spanning 1 to 12 align "
            "iterations. READS per model the align class's time against ceil(numel / max(E, 32)) (OLS slope and intercept, "
            "the diagnosis's 0.96 us beside; fix A is not in v4d', D3-10) and silu_and_mul + moe_sum against v4d's B_other price. Cold L2 per kernel: whether "
            "the intermediates are L2-resident in the real call is NOT READ",
        f"M3 label (units m3: dp0, dp{D3.M3_PADS[1]} on Mixtral)": "six Mixtral T 4 expert sets (two experts at 4 rows: "
            "S01 {0,1} the reference, S25 {2,5}, S46 {4,6}, S15 {1,5}, S37 {3,7}, S26 {2,6}) in GRR at R 1, through "
            f"r3_decode3.py at pad rows 0 (the control) and {D3.M3_PADS[1]} (rental 2's address control). READS d = T(set) "
            "- T(S01) per pad; the expert-5 sets' mean d0 and d_pad and q = d_pad / d0: FOLLOWS-EXPERT at q >= 2/3 (the "
            "effect does not follow a 2 MiB-page shift of the slab: identity or data), FOLLOWS-ADDRESS at q <= 1/3 "
            "(placement), MIXED between; NOT READ when d0 < 0.5 us, or when a page's report.json does not record the pad "
            "rows its unit registers (review m1)",
        "M3 pad (owner, 2026-11-03, D3-11)": "at least 52 rows; the smallest pad that moves expert 5's w1 slab past a 2 MiB "
            "page boundary: a pad row adds H x 2 = 4096 x 2 = 8192 B to w1's slot stride, slot 5 (copies 1 at R 1, "
            "expert-first) starts 5 x P x 8192 B later, and 5 x P x 8192 >= 2097152 B needs P >= 51.2, so P = "
            f"{D3.M3_PADS[1]} (2129920 B, 2.03 MiB; w2's slot 5 moves 5 x 52 x 14336 x 2 = 7454720 B, 7.11 MiB). Rental "
            "2's pad 7 moved it 280 KiB, under a page. The units' minutes are unchanged (the padding adds 52 rows per "
            "slot, under 0.1% of the weights, and no cell)",
        "M4 gaps (units m4: dg GRR and dgn counters on OLMoE)": "OLMoE T 2 cells with three repeated experts at row gaps "
            "8 to 13 (G08-R3 to G13-R3: followers at rows 8 + s to 10 + s), one at 15 (G15-R1) and the no-repeat control "
            "(BASE-A16). The wave boundary is CTA 396 (row 12.375), so the followers cross from fix W's 'served' side to "
            "its 're-fetched' side inside the page; gaps under 8 rows are not reachable at T 2 (realize_counts' greedy "
            "order and the page seed's identity permutation of two rows), so the L2-cut half of W (a later wave under 60 MiB) "
            "is touched only at its knife edge (review m3): every w2 follower here streams 8 to 15 rows x 32 CTAs x "
            "128 KiB = 32 to 60 MiB, under the cut until G15-R1's w2, which sits at exactly 60 MiB and re-fetches only "
            "because the rule reads >=. W therefore predicts the w2 rows SERVED (G15 RE-FETCHED); U-T2's published w2 "
            "re-read 48% at a 13-row gap, so expect w2 rows to band MIXED and DISAGREE with W: DESCRIPTIVE. READS per cell the GRR time against v4d' and v4d (dedup) at the CAL2 fit, and per GEMM "
            "f = (counted - dedup) / (nodedup - dedup) against W's f_W; bands SERVED < 1/3, RE-FETCHED > 2/3, MIXED; "
            "a row AGREES when its two bands match",
    }


def build() -> dict:
    mins = D3.cal2_minutes()
    plan = (REPO / D3.CAL2_PLAN).read_text()
    pub = json.loads((REPO / D3.PUBLISHED_SCORE).read_text())
    lab = D3.label_effect(pub)
    return {
        "registered": f"{D3.DATE}, before any CAL2 page",
        "name": D3.NAME,
        "status": "REGISTERED BEFORE ANY CAL2 PAGE: the decode stage's third addendum; decodeC-cal, decodeB-blind, the "
                  "tile-source addendum and addendum 2 stay byte-identical (pinned here); every registered verdict stands "
                  "(the decode CAL rental's STOP of 2026-10-29 included)",
        "design": DESIGN,
        "tool": "scripts/scoring/decode/score_cal_d3.py <tree> <out> --tile-scores scripts/scoring/tileA2; on GO "
                "finalise_decode_d3.py <out> <finalised.json> --plan-out scripts/plans/rentalD-blind-2026-10.plan; after the "
                "blind rental score_blind_d3.py <tree> <finalised.json> <out> (new wrappers; the frozen d2 wrappers and the "
                "frozen scorers run inside, unchanged)",
        "model v4d' (owner, 2026-10-29, D3-1)": {
            "module": "scripts/r3_timing_model_v4dp.py (new; r3_timing_model_v4d.py byte-identical, imported and called)",
            "formula": "T = v4d(T) with W (naive rows in launch order, the follower rule), + (W1_SCALE - 1) x the w1 GEMM time "
                       "on 16x64x128w4s4 (R, a time ratio on the whole w1 GEMM price); no align term (A dropped, D3-10)",
            "fitted constants added": 0,
            "CAL fit": "v4d's own registered fit, unchanged: T0_GR, s_naive, s_block by v4d.fit_constants and the rho "
                       "selection (dcommon.fit_and_select) on the kept cells (addendum 2's exclusions), on v4d''s body",
            "fixes": constants(),
            "reduction (fixes off)": reduction(),
        },
        "seen_reading": seen_reading(),
        "G3-D label (owner, 2026-10-29, D3-2)": {
            "rule": "G3-D label is computed exactly as registered (dcommon.g3d at the CAL2 fit) and REPORTED; it is not a "
                    "stop clause. dcommon.stop_rule runs unchanged and its 'G3-D label' line is moved to DESCRIPTIVE "
                    "(d3common.stop_rule_d3). G3-D shuffle and every other clause stay stop clauses",
            "the published effect (Mixtral, quoted)": f"the published decode CAL (v4d, amended fit): G3-D label "
                f"{lab['verdict']}, mean {lab['mean_us']} us against margin {lab['margin_us']} us; Mixtral's MIN set "
                f"{{0, 1}} against L-T-A2's {{2, 5}} reads d_us {lab['per_model']['mixtral-8x7b']['d_us']} (mean "
                f"{lab['per_model']['mixtral-8x7b']['mean_us']} us; sets holding expert 5 and probably 2 run 1 to 2.5 us "
                f"slower, the diagnosis's section 3), OLMoE's pairs {lab['per_model']['olmoe-1b-7b']['d_us']} (null)",
            "blind (owner, 2026-11-03, D3-6)": "G3-D label is DESCRIPTIVE in the blind score too: computed per blind main "
                     "page exactly as decodeB-blind registers it (dcommon.g3d at the finalised constants, the CAL2 margin "
                     "rule) and reported per model (G3D_LABEL), but it never voids a model's certification (d3common."
                     "label_descriptive hands the frozen score_blind a NOT SCORED verdict for the label test only). G3-D "
                     "shuffle and GD2 still void as registered. Registered here before any blind page; the scope line "
                     "(d3common.SCOPE) states it beside every CERTIFIED, and finalise_decode_d3.py freezes it (g3d_label). "
                     "Phi-3.5 runs at R 1 like Mixtral and was the model exposed to the expert-set effect",
            "scope line (blind and finalisation)": D3.SCOPE,
        },
        "stop rule (as amended, CAL2)": "addendum 2's stop rule (dcommon.stop_rule on the amended score: power, the fit, "
            "E1-CAL-IN-SAMPLE at the provisional bars and NOT PRICED without the DESCRIPTIVE cells, G3-D shuffle, GD2, "
            "the nodrop units, V-ALIGN, GD5, COPIES) with v4d' pricing and the G3-D label line DESCRIPTIVE. Else GO. The "
            "settling units are never in it",
        "bars and certification (owner, 2026-10-29, D3-3)": {
            "CAL2": "E1-CAL-IN-SAMPLE at the provisional bars (rms 3%, worst 8%, |mean| 1.5%), as registered",
            "blind": "the blind bars as registered: tightened from the CAL2 replicate's kept-cell sigma_page (dcommon.tighten, "
                     "floors rms 1.5%, worst 4%, |mean| 0.75%; the published CAL's replicate read 0.188% and hit the floors); "
                     "a model CERTIFIED when its own bar and the pooled bar hold (decodeB-blind, addendum 2's exclusions and "
                     "scope line)",
            "expected odds (ESTIMATES, the diagnosis's, not measured)": {
                "CAL2 E1-CAL-IN-SAMPLE holds at the provisional bars": "about 0.75",
                "CAL2 reads GO with G3-D label DESCRIPTIVE": "about 0.7",
                "blind pooled E1-DEC holds at the tightened bars": "about 0.05",
                "at least one blind model CERTIFIED": "about 0.15",
                "label": "the diagnosis's estimates, stated as registered on 2026-10-29 (made with A on); the review put "
                         "CAL2 GO at about 0.8 (INFERENCE); not recomputed for W + R",
                "why": "in-sample on the published CAL v4d' (W + R) reads rms 2.13%, worst -5.7%, mean +0.15% (seen_reading, SEEN); the "
                       "best in-sample fit with 4 more fitted constants is 1.83%, over the 1.5% blind bar; T0_GR's transfer "
                       "moves Granite 4 to 6%; Granite reaches 20 align iterations against CAL's 12 and v4d' carries no align term "
                       "(A dropped); W is untested "
                       "at the blind pages' gaps (1 to 13 rows)",
            },
        },
        "CAL2 plan (owner, 2026-10-29, D3-4)": {
            "plan": D3.CAL2_PLAN, "plan_sha256": RGD.sha256(REPO / D3.CAL2_PLAN),
            "rule": f"addendum 2's CAL plan ({D2.CAL_PLAN_D2}), every unit line byte-identical and in its order, then the "
                    "settling units in the order m1, m4, m2, m3 (the driver drops the last first: m3, m2, m4, m1, then "
                    "rep, decl, e240); generated by d3common.cal2_plan",
            "driver": f"scripts/{D3.DRIVER3} (the pinned gh200_decode_session.sh with hunks marked D3: pad-rows= on r3decode "
                      "units runs scripts/r3_decode3.py --pad-rows; kernels=wide on decodebytes units runs scripts/"
                      "dram_counter_route_decode3.py --kernels wide); vm_run.sh's SESSION_DRIVERS and usage line, and "
                      "docs/LAMBDA.md's stop line, name it",
            "settle_units": D3.settle_units(),
            "settling readings (DESCRIPTIVE, registered before pages)": settle_text(),
            "never": "a settling page is never a STOP, never in the CAL2 fit, the rho selection or any E1 bar, and never "
                     "used to refit v4d' before the CAL2 reading (the CAL2 fit is v4d's registered fit on the kept main-page "
                     "cells); a later registration may use them",
            "estimates": "units from the tools' own estimators: r3decode pages r3_decode.estimated_seconds (9 repeats, 2 "
                         "min setup each); counter pages the route's 0.6 s per profiled launch, warmups profiled, plus 2 "
                         "min setup (the registered dn units' margin); the wide route at 6 launches per call",
        },
        "pages": pages(),
        "scorers (owner, 2026-10-29, D3-5)": {
            "score_cal_d3.py": "score_cal_d2.score (frozen) under d3common.v4dp; STOP by stop_rule_d3; addendum 2's "
                               "exclusions, scope line, power check, SIGMA-D, knee rows and BYTES-D2 kept as they run; "
                               "DESCRIPTIVE beside: G3D_LABEL, V4D_READING (v4d as registered), SETTLE (M1 to M4), "
                               "CONSTANTS_REPLICATE",
            "finalise_decode_d3.py": "refuses unless the score is score_cal_d3's under this addendum with v4d' at the "
                                     "registered fixes, power READ, addendum 2's excluded cells, STOP GO and a rho form "
                                     "selected; then finalise_decode.frozen and .predict (frozen) under v4dp",
            "blind finalisation freezes": "the model (v4d': fixes W and R and their constants as above; no align term), T0_GR, s_naive, "
                                          "s_block and the rho form of the CAL2 fit, the tile stage's v4 Params as the CAL2 "
                                          "scorer used them, the blind bars (REP's tightened bars from the kept-cell "
                                          "replicate), sigma_page, the GD2 band, the G3-D margin rule (max(0.3 us, 0.36% x "
                                          "the median reference time)), the rivals' CAL2 fits (line, OLS), and per blind "
                                          "main cell (220 kept, 27 DESCRIPTIVE) v4d''s time and every rival's (rho other, "
                                          "NAIVE-NODEDUP and eager priced by v4d' with the rival's change; roofline, line, "
                                          "OLS as frozen), the G3-D rule (shuffle voids, label DESCRIPTIVE and never a void, "
                                          "owner 2026-11-03) and the scope line; the blind plan is finalise_decode.blind_plan, "
                                          "decodeB-blind's units",
            "score_blind_d3.py": "score_blind_d2.score (frozen) under v4dp at the finalised fixes and d3common."
                                 "label_descriptive: decodeB-blind's rules and addendum 2's exclusions, power check and "
                                 "SIGMA-D unchanged, the G3-D label void removed (DESCRIPTIVE, G3D_LABEL per model), the "
                                 "scope line d3common.SCOPE",
        },
        "rentals": {
            "sequence": "the CAL2 rental (this plan, the D3 driver); if its stop rule as amended reads GO, finalise_decode_d3.py "
                        "mechanically, then the blind rental (the frozen blind plan, the pinned decode driver); on a STOP "
                        "the blind rental does not run",
            "cal2": dict(mins["cal2"], composition=f"{mins['cal_units_min']} min of the registered CAL units (prelude, "
                         f"calibrate and census included) + {mins['settle_units_min']} min of settling units"),
            "blind": mins["blind"],
            "operator power step": "as addendum 2 (d2common.py --power on the prelude's power-set.txt; terminate unless READ)",
            "versus the diagnosis": "the diagnosis estimated about 13 to 20 unit-minutes for M1 to M4 (about $1.7 as a "
                                    "standalone rental); the tools' estimators here read "
                                    f"{mins['settle_units_min']} min (each r3decode page carries 2 min of setup, and M3 runs "
                                    "its pad-0 control in-session), about "
                                    f"${mins['settle_units_min'] / 0.84 / 60 * D2.RATE:.2f} added to the CAL rental",
        },
        "owner decisions of 2026-11-03 (on the build's open list)": [
            "D3-6 the blind G3-D label void becomes DESCRIPTIVE: reported per model, never voids certification; in the "
            "scope line, score_blind_d3.py and finalise_decode_d3.py (above)",
            "D3-7 fix R stays as built: a time ratio on the whole w1 GEMM price (tail included), as the diagnosis tested it, "
            "not a bandwidth override; measured at rho scaled and applied whatever rho form CAL2 selects",
            "D3-8 M3's pad-0 control is kept (the settling units stay 30 min)",
            "D3-10 fix A DROPPED (review MAJOR-1): v4d' = W + R; M2 measures the align slope DESCRIPTIVELY; a later model "
            "may add it only under its own registration",
            "D3-11 M3's shifted pad: 52 rows (at least 52; the arithmetic under 'M3 pad'), the pad-0 control kept",
            "D3-9 later models: M1's per-tile rates, or an align term remeasured on M2 or CAL2, may enter a later model only under "
            "its own registration, before any page that model is scored on; addendum 3 never uses them, and v4d''s "
            "constants are the ones registered here whatever the settling units read",
        ],
        "unchanged": "the lock 1710 and G4 1705, the instruments, every registered page, the census rule, the tile source, "
                     "the card constants, v4d (byte-identical), every pin of the four earlier decode registrations, and "
                     "every registered verdict",
        "seen": [
            "the published decode CAL session (2026-10-29) and its score: SEEN; fix R's ratio (and the dropped fix A's slope) were "
            "measured on them, and seen_reading fits v4d' on them in-sample",
            "the diagnosis's COMPUTED, SEEN-FITTED and INFERENCE readings (" + DIAG + "): the reason for the fixes and the "
            "settling units; no blind page exists",
        ],
    }


def text(d: dict) -> list[str]:
    r = d["rentals"]
    L = [f"REGISTERED {d['registered']}: the decode stage (a third addendum), {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", ""]
    for part in ("cal2", "blind"):
        x = r[part]
        L.append(f"{part} rental: {x['units_min']} min of units, about {x['wall_min']} min wall with {x['boot_setup_min']} "
                 f"min of boot and setup: ${x['cost_usd']:.2f} at ${D2.RATE}/h; deadline launch + "
                 f"{x['deadline_after_launch_min']} min; guardian hard cap launch + {x['hard_cap_after_launch_min']} min "
                 f"(${x['cost_at_hard_cap_usd']:.2f} at most)")
    m = d["model v4d' (owner, 2026-10-29, D3-1)"]
    L.append(f"reduction: {m['reduction (fixes off)']['verdict']} ({m['reduction (fixes off)']['predictions']} predictions; "
             f"{m['reduction (fixes off)']['what']})")
    s = d["seen_reading"]
    L.append(f"seen reading (SEEN, IN-SAMPLE, DESCRIPTIVE): rho {s['selected']}, E1 {s['E1_CAL_IN_SAMPLE']['verdict']} "
             f"{s['E1_CAL_IN_SAMPLE']['stats']}, stop (addendum 3) {s['STOP_D3']['verdict']}")
    L.append("settling units:")
    L += ["  " + D3.settle_line(u) for u in d["CAL2 plan (owner, 2026-10-29, D3-4)"]["settle_units"]]
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        RGD._flat("", {k: v}, L)
    return L


def pins() -> dict:
    prev = json.loads((REG / f"{D2.NAME}.json").read_text())["code_pins"]
    paths = tuple(dict.fromkeys(tuple(prev) + FROZEN + RG2.D2_CODE + ("scripts/r3_timing_model_v4d.py",) + SEEN_INPUTS
                                + D3_CODE + tuple(D3.page3_path(m, k) for m, k in D3.PAGE_SPECS)))
    return {p: RGD.sha256(REPO / p) for p in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    files = D3.page3_files()
    pdir = REPO / D3.PAGES3
    plan = D3.cal2_plan()
    if a.write:
        pdir.mkdir(parents=True, exist_ok=True)
        for n, t in files.items():
            (pdir / n).write_text(t)
        (REPO / D3.CAL2_PLAN).write_text(plan)
    doc = build()
    doc["code_pins"] = pins()
    jpath, tpath = a.out_dir / f"{D3.NAME}.json", a.out_dir / f"{D3.NAME}.txt"
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    if a.check:
        strip = lambda x: {k: v for k, v in x.items() if k != "code_pins"}  # noqa: E731
        same_j = jpath.exists() and strip(json.loads(jpath.read_text())) == strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        same_p = all((pdir / n).exists() and (pdir / n).read_text() == t for n, t in files.items())
        same_plan = (REPO / D3.CAL2_PLAN).exists() and (REPO / D3.CAL2_PLAN).read_text() == plan
        for name, same in ((jpath.name, same_j), (tpath.name, same_t), ("settling pages", same_p), ("CAL2 plan", same_plan)):
            print(f"{name}: {'same' if same else 'DIFFERS'}")
        return 0 if same_j and same_t and same_p and same_plan else 1
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {len(files)} pages under {D3.PAGES3}\nwrote {D3.CAL2_PLAN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
