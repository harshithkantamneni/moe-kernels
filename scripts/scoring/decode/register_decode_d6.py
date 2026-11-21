#!/usr/bin/env python
"""Writes the decode stage's sixth addendum, docs/registered/2026-11-10-decode-addendum6-gh200.{json,txt}, before any
page of the three new blind models: model v6d (frozen 2026-11-10T22:35:03Z on PC's, PC2's, DS's and CAL2's pages; its
constants the readings, mechanically), the gate (the owner's two clauses as addendum 5 coded them) and its reading on
the published trees (computed after the freeze), addendum 5's new blind (pages, units, bars, rental) registered again
under v6d, and every forking path. The last decode round under the owner's stopping rule (2026-11-10). Every earlier
decode registration stays byte-identical (pinned here), and so do v4d, v4d', v5d and every earlier decode scorer.

    python scripts/scoring/decode/register_decode_d6.py --write [--out-dir docs/registered]
    python scripts/scoring/decode/register_decode_d6.py --check

Every number is computed here from committed files (the readings, the published CAL2 and decode blind trees, the
registered pages, v6d) or is a typed-in record with its provenance (design-D6's post-freeze profile; the estimates).
`--check` recomputes and compares the JSON (code_pins excluded, as every registration), the text and the gate output,
and checks that no blind plan file exists; exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import d6common as D6  # noqa: E402
import dcommon as DC  # noqa: E402
import register_decode as RGD  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
V6D = D6.V6D
GATE_OUT = "scripts/scoring/decode/gate_d6.out.txt"
WORK = "scripts/scoring/decode/d6_build"
D6_CODE = ("scripts/r3_timing_model_v6d.py", "scripts/scoring/decode/d6common.py", "scripts/scoring/decode/gate_d6.py",
           "scripts/scoring/decode/finalise_decode_d6.py", "scripts/scoring/decode/score_blind_d6.py",
           "scripts/scoring/decode/register_decode_d6.py", GATE_OUT, D6.PC_READ, D6.PC2_READ)
FROZEN = tuple(f"docs/registered/{D5.NAME}.{x}" for x in ("json", "txt"))
BUILD = tuple(f"{WORK}/{n}" for n in (
    "FROZEN.txt", "v6d_constants.json", "pc_spans.py", "pc_spans.json", "ptx_inflight.py", "ptx_inflight.out", "d6v.py",
    "run.py", "pc_check.py", "pc_check.out", "pc2_check.py", "pc2_check.out", "ds_check.py", "ds_check.out",
    "naive_check.py", "naive_check.out", "alternatives.py", "alternatives.out", "reach.py", "reach.out",
    "check_unseen.py", "check_unseen.out", "post_gate.py", "post_gate.out"))
SKIP_TXT = {"code_pins", "pages", "units", "rows", "cells", "worst_cells", "law constants"}
FREEZE_TS = "2026-11-10T22:35:03Z"
GATE_TS = "2026-11-10T22:36:41Z"

DESIGN = ("v6d is design-D6's GEMM law (2026-11-09) with PC2's READ values taken and PC2's empty slots filled by explicit "
          "choices (D6-18 to D6-24), each made on SEEN measurement pages only (PC, PC2, DS, CAL2; CAL, the first CAL rental, "
          "not used) and frozen at " + FREEZE_TS + " (scripts/scoring/decode/d6_build/FROZEN.txt: sha256 of the model "
          "file and its constants) before the gate ran. Nothing is fitted on the earlier decode blind pages or on any page "
          "of the three new models. The gate is the owner's two clauses as addendum 5 coded them; it ran at " + GATE_TS +
          " on the published trees and reads NO-GO on clause 2. Under the owner's stopping rule (2026-11-10) decode closes "
          "as calibrated, not certified: the new blind is registered and NOT RUN, its plan is never written, and the NO-GO "
          "is not overridden")
CLOSED = ("the owner's stopping rule (2026-11-10), applied to the gate's reading: decode gets PC2 plus at most one decode "
          "blind, run only on a gate GO, and closes whatever the result. The gate reads NO-GO, so the new blind is not "
          "launched and decode closes as calibrated, not certified; Mixtral-8x22B, Qwen2-57B-A14B and Granite-3.0-1B-A400M "
          "remain unseen")


def _out(name: str) -> list[str]:
    return (REPO / WORK / name).read_text().splitlines()


def decisions() -> dict:
    lab = (f"COMPUTED by the committed build records under {WORK} (pc2_check.py, pc_check.py, ds_check.py, naive_check.py, "
           "alternatives.py; outputs *.out beside; run.py --check recomputes them), on SEEN measurement pages; the SEEN decode "
           "blind was not re-scored under any addendum-6 variant. PROVENANCE: the numbers were first computed before the "
           "freeze by build-D6's work/ scripts (outside the repository); the committed d6_build scripts were rewritten "
           "from them after the gate and reproduce those numbers. The frozen hashes (d6_build/FROZEN.txt) cover the model "
           "file, its constants, PC's span reading and d6common.py, not these scripts")
    return {
        "label": lab,
        "PC2's READ values taken": (
            "16x64x128w4s4: Q_B 48 KB (PC2-R1g n_if 3 on the stamped s4 PTX; v6d's value), X_sat 3781.89 GB/s (PC2-R1d, 12 "
            "members, board 588570; v6d held 3717.9), c0 at G 8 2.72 us (PC2-R1b, READ, 5 of 6 rungs). 64x128x64w4s3: c0 at "
            "G 8 2.40 us (PC2-R2a, READ). The 4-per-SM tiles: kappa at the READ K 14336 rungs (PC2-R3: 32x64x128 G 264, 396, "
            "462, 528 at 1.079, 1.100, 1.094, 1.026; 32x128x64 G 198, 264, 396, 528 at 1.077, 1.038, 1.105, 1.019). Board "
            "check: PC2's 64x128x64 long-slab lives at G 8, 32, 128, 256 sit +1.4%, -0.2%, +0.4%, +0.9% from PC's (board "
            "c9f689), so PC2's tile values are used beside PC's table"),
        "D6-18 c0 against load (PC2, PC; informed by PC2)": (
            "c0 piecewise linear in the live CTAs G between G 8 (PC2-R1b 2.72 us on 16x64x128, PC2-R2a 2.40 us on 64x128x64 "
            "and, never read on their own, the 4-per-SM tiles) and G 128 (PC-R2 4.70 us, read on 64x128x64; 16x64x128's G 128 "
            "is NOT IDENTIFIED, so it keeps PC's), held outside; every wave of a multi-wave GEMM at c0(S) (its CTAs launch "
            "into a full GPU). Why, on PC2's pages that set no constant (pc2_check.out): the G 32 and G 256 short rungs on "
            "64x128x64 (single rungs READ, their ladders NOT IDENTIFIED) read +4.5% and -1.6% mean under the interpolation, "
            "+26.0% and -1.6% at a constant 4.70 us, -1.0% and -12.8% at a constant 2.40 us, +12.8% and -1.6% interpolated "
            "in ln G; all 74 READ PC2 launches read 3.8% rms against 12.5% for design-D6's constant. The remainder at c0(S): "
            "PC's 20 multi-wave w1 launches read 3.10% against 6.16% at c0(k) (pc_check.out). CAL2 cannot choose (T0 absorbs "
            "c0): 1.81% against 1.83% (4.70 constant), 1.82% (ln G), 2.03% (remainder at c0(k)); the 2.40 constant reads "
            "STOP (alternatives.out). Alternatives considered and not taken: a constant (4.70 or 2.40), ln G, the "
            "remainder at c0(k), c0 = kc L (design-D6, D6-3)"),
        "D6-19 the L table": (
            "PC-R7's rungs G 4 to G 384 with PC-R1's median life, as v6d, with c0(G) of D6-18 removed at each rung (L = Q "
            "(life - c0) / b, N_eff = n Q (life - c0) / life): a consequence of D6-18, not a separate choice"),
        "D6-20 16x64x128's c0, L table and kappa (informed by PC2)": (
            "c0 by D6-18. L table: PC's (v6d's; PC2-R1c is NOT IDENTIFIED because c0 at G 128 is unread); the alternative, a "
            "table from PC2-R1a's own lives at D6-18's c0, reads 2.1% against 2.4% on PC2's 9 long single-wave rungs and CAL2 "
            "1.96% against 1.81%: not taken. kappa: PC's by fill (v6d's), NOT PC2-R1e as read. THE ONE PLACE THE BRIEF ('kappa "
            "where read') IS NOT FOLLOWED, and why: inside v6d's form R1e's G 256 reading (1.196, span over a median life "
            "that beats the fair-share ceiling the form bounds a life by) prices PC2's own D-G256 t +14.8% and D-G256-F1024 "
            "+15.7% (PC's kappa: +0.2%, +1.0%); its 9 long single-wave rungs read 5.7% rms against 2.4%; and CAL2 reads STOP "
            "(3.13%, worst -9.80%; Mixtral L-T2-A4's w2 is 256 CTAs of this tile at D-G256's slab). R1e's values are printed "
            "beside in pc2.read.json"),
        "D6-21 term 3, the tail at big single waves (informed by PC2)": (
            "on the two 4-per-SM tiles kappa is PC2-R3's READ K 14336 rungs by fill (n / 528), and below a tile's lowest read "
            "fill PC's by fill: the 7 to 8% fall from G 396 to G 528 that PC2-R3 reads (term 3 DOES NOT HOLD) is in the table. "
            "PC2's 8 long single-wave 4-per-SM rungs read 2.3% rms (in sample) against 2.6% for v6d's held kappa (W128-G396 "
            "+1.0% against -5.2%); CAL2 1.81% against 1.80%. Alternatives: v6d's kappa by m held past G 384 (above); a rule "
            "pooled on PC's and PC2's kappa against fill (median per 0.1 fill bin): PC2 2.5% on those rungs, CAL2 2.47% with "
            "worst -7.49%; PC's kappa by fill on every tile: CAL2 1.78%. X_sat on these tiles stays v6d's (PC x DS's W1RATE "
            "ratio, 3703.8 and 3694.4); PC2-R3's own 3726 and 3650 read CAL2 1.78%: not taken"),
        "D6-22 kappa indexed by fill": (
            "kappa is read at the fill m / S of the GEMM's tile (S 396 on the 3-per-SM tiles, where it equals v6d's kappa(m); "
            "S 528 on the 4-per-SM tiles), held outside the read fills; with D6-21 this replaces v6d's kappa(m) held past G 384"),
        "D6-23 kappa at short slabs (PC2 slot 2)": (
            "kept by load only (v6d's form): PC2-R2c reads short-slab kappa at or above the long anchor (1.01 to 1.22) but "
            "about half the short cells are NOT READ on the per-call spread; a slab term would be a form change on partial "
            "readings. Not taken"),
        "D6-24 the naive path (informed by PC2-R4, with the seen blind in view)": (
            "a naive cell whose naive GEMMs fetch every CTA's slab (no follower, or every follower re-fetches under v4d''s fix "
            "W) is priced by the law as its aligned twin (PC2-R4: twin t ratios 0.93 to 1.02); a cell with any non-fetching "
            "follower keeps v4d''s naive GEMM prices. One CAL2 fit (its constants and its rho form) serves both. Evidence (naive_check.out, "
            "seen models' READ PC2-R4 GEMMs): no-follower GEMMs read 12.1% rms (mean +9.9%) under the law against 26.3% (mean "
            "-3.0%) under v4d'; follower GEMMs read 9.5% under all@fetch (every CTA at the fetchers' load, R4's 'same per-CTA "
            "rate' read literally) and 18.6% under v4d'. CAL2: the candidate 1.81% GO; v4d' everywhere (D6-13) 1.48% GO; "
            "all@fetch STOP (2.42%, worst -11.39%); the law on every naive cell STOP (2.47%, +11.22%); the fetchers alone STOP "
            "(2.53%, -13.57%). DISCLOSED: this form is design-D6's post-freeze fork naive_nofollow6 (under design-D6's law it "
            "read CAL2 STOP 4.58% and re-scored the seen blind at 5.82%), and the seen blind's 29 naive cells were known to "
            "re-score at 8.17% under v6d (design-D6 section 6.2); it was chosen here on PC2-R4 and CAL2, under the new law, "
            "and its re-score was not computed before the freeze"),
        "carried from design-D6 (D6-1 to D6-17)": (
            "D6-1 blend+ multi-wave, D6-2 the duty load, D6-4, D6-5 (kappa; now by fill, D6-22), D6-6 X_sat per tile (16x64x128 "
            "now PC2-R1d, D6-20), D6-7 no beta_f, D6-8 Q_B from the PTX, D6-9 the table to G 384, D6-10 the B slab, D6-11 no "
            "W1 scale, D6-12 CALL, ALIGN and SLOPE kept, D6-14 the CAL constants refitted on CAL2, D6-15 PC's C, R, E1 and E2 "
            "blocks unused, D6-16 CAL unused, D6-17 the forking paths below. Replaced: D6-3 (c0 constant, by D6-18) and D6-13 "
            "(naive v4d', by D6-24)"),
        "carried from addendum 5 (D5-1 to D5-8)": (
            "D5-4 (the CALL forms, a forking path on DS's pages), D5-5 (ALIGN; Granite-1B's block-scan E 32 held at E 40's "
            "8.704 us) and D5-6 (the CAL refit) carry over unchanged; D5-1 and D5-3 (the regimes and W1) are emptied by the "
            "law; D5-2 (naive v4d') is replaced by D6-24; D5-7's weak DS readings no longer price anything; D5-8's list "
            "(choices informed by the old blind's diagnosis) carries over: that the call kernels and align are model terms "
            "at all"),
    }


def forking_paths() -> dict:
    return {
        "the old decode blind was seen": (
            "the decode blind pages of 2026-11-04 (Phi-3.5, Qwen1.5, JetMoE, deepseek-v2-lite, Granite-3B) were read by the "
            "two offline diagnoses of 2026-11-04 that chose what DS measured, re-scored under v5d variants in addendum 5's "
            "build, and re-scored under v6d variants in design-D6 (its section 6, ablation6.out, every_path6.out, "
            "naive_nofollow6.out). build-D6 needed no fact about them beyond design-D6's record and read none before the "
            "freeze (design-D6's rescore_detail.out and preview.out were not opened); its gate re-score and the post-gate "
            "record are the first build-D6 computations on them"),
        "design-D6's post-freeze variants, not taken (re-scores POST-HOC, design-D6 section 6.4)": [
            "no duty 3.99% (CAL2 GO 1.64%): the one variant that read GO on both clauses; not taken (owner, 2026-11-10: v6d "
            "keeps the duty factor)", "c0 0: 4.01%", "the table to G 288: 4.02%", "load X: 4.03%", "c0 2.69 us: 4.08%",
            "X_sat PC on every tile: 4.30%", "CALL off: 4.40%", "blend, no lag: 4.48%", "no span factor: 4.69%",
            "c0 = kc L: 4.82% (CAL2 STOP)", "X_sat DS levels: 5.73%", "ALIGN and SLOPE off: 6.83%",
            "exact waves: 10.47% (CAL2 STOP)", "the law on the naive leaders: 3.69% (CAL2 STOP 2.62%)",
            "the law on follower-free naive cells (D6-24's form under design-D6's law): 5.82% (CAL2 STOP 4.58%)"],
        "build-D6's choices with PC2 in view": "D6-18, D6-20, D6-21 and D6-24 (above), each after PC2's pages were read",
        "the brief not followed in one place": "D6-20: PC2-R1e's kappa on 16x64x128 not taken, with the evidence above",
        "the last round": D6.STOPPING_RULE,
    }


def gate_block() -> tuple[dict, list[str]]:
    import gate_d6 as G
    cal2, blind = REPO / D4.CAL2_PUBLISHED / "results", REPO / D4.BLIND_PUBLISHED / "results"
    with tempfile.TemporaryDirectory() as td:
        g = G.gate(G.readings_dir_committed(Path(td)), cal2, blind)
    return {
        "rule (owner, 2026-11-04, recorded in addendum 4, coded in addendum 5, applied to v6d here)": (
            "gate_d6.py prints one GO / NO-GO line; GO needs every clause: READINGS (the readings give this registration's "
            "law and DS's give addendum 5's terms, nothing BLOCKING), TREES (both published trees reproduce their committed "
            "scores), CAL (clause 1: CAL2 under v6d reads GO by the addendum-3 stop rule at 3% rms, 8% worst cell, 1.5% "
            f"|mean|) and RE-SCORE (clause 2: the published decode blind re-scored under v6d at the CAL2 refit reads a pooled "
            f"rms under {D6.RESCORE_RMS_MAX:.1%} over the committed {D6.RESCORE_CELLS} cells). finalise_decode_d6.py runs the "
            "same evaluation and refuses unless it reads GO"),
        "clause 2 label": D6.WEAK_FLOOR,
        "the gate on the published trees (COMPUTED after the freeze)": {
            "ran": f"{GATE_TS}, after the freeze at {FREEZE_TS}; no choice changed after it",
            "output file": GATE_OUT + " (gate_d6.lines on the published trees; --check recomputes it)",
            "line": G.line(g), "verdict": g["verdict"], "why": g["why"],
            "CAL": {k: g.get("CAL", {}).get(k) for k in ("verdict", "E1_CAL_IN_SAMPLE", "bars", "constants", "rho",
                                                         "worst_cells")},
            "RE-SCORE (POST-HOC on SEEN pages; a weak sanity floor, never a test)": {
                k: (g.get("RESCORE") or {}).get(k) for k in ("verdict", "pooled", "per_model", "read_at")}},
    }, G.lines(g)


def after_gate() -> dict:
    return {
        "label": (f"POST-HOC, after the gate ({WORK}/post_gate.py, post_gate.out): arithmetic on the gate's own rows, "
                  "the naive branch each cell takes, CAL2's two rho forms and two cell counterfactuals; it changes no "
                  "choice and no verdict"),
        "record": _out("post_gate.out"),
        "a defect in D6-24, found after the freeze (reported, NOT fixed)": (
            "D6-24 prices a naive cell by the law or by v4d', and both branches take one CAL2 fit. That fit selects its rho "
            "form (fixed or scaled) by the registered rule, the lower rms on CAL2's sub-wave cells, and the law-priced "
            "cells dominate it: under v6d it selects FIXED (sub-wave 1.48% against 3.32% scaled; in sample 1.81%, s_naive "
            "-2.14 us). Every earlier decode CAL fit selected SCALED (CAL under v4d, CAL2 under v4d', v5d's gate), and with "
            "v4d' on every naive cell CAL2 also selects scaled (sub-wave 1.44% against 7.88% fixed; in sample 1.48%, "
            "s_naive +9.55 us); at the fixed form that fit reads 5.47% in sample with s_naive -1.83 us, close to v6d's "
            "-2.14 us. The rho form changes v4d''s naive GEMM bodies themselves: OLMoE L-T2-A11's body is 69.20 us under "
            "fixed and 53.81 us under scaled. So the v4d' branch inherits a rho form chosen by a fit the law dominates; the "
            "constant is not what moved. CAL2 holds 4 v4d'-branch cells (OLMoE T 2) and reads them within 6%. On the seen "
            "blind the 13 v4d'-branch cells (Qwen1.5 7, deepseek-v2-lite 4, Phi-3.5 2) read 13.2% rms, mean -8.6% (Qwen1.5 "
            "L-T2-A4 -31.5%, A5 -26.6%, A7 -20.2%), and the other 207 cells 3.27%. POST-HOC counterfactuals on those 13 "
            "cells (d6_build/post_gate.out), never a choice: the fixed form with s_naive raised by 11.69 us alone reads "
            "12.7% rms (mean +10.3%); the scaled form with s_naive +9.55 us reads 1.36%. The code does what D6-24 says; the "
            "defect is the rule's shared fit. Any fix would be chosen on the seen re-score, so none is made and the NO-GO "
            "stands. The new blind's naive main cells all take the law branch (2, 7 and 2)"),
    }


def blind_block() -> dict:
    return {"models": list(D6.NEW_BLIND), "bars": D6.BARS_OWNER, "units": D5.blind_units(), "drop_order": ["gr1"],
            "pages": {"dir": D5.PAGES5, "label": "addendum 5's registered page lists (decode_synth's frozen generators), "
                                                 "reused unchanged; never timed",
                      "files": {f"{m}-{k}.json": D5.reg_file5(m, k) for m, k in D5.PAGE_KEYS}},
            "plan_units": D5.blind_plan_units(), "plan": D6.BLIND6_PLAN,
            "status": "NOT RUN: the gate reads NO-GO and decode closes under the owner's stopping rule",
            "plan_written_by": "finalise_decode_d6.py --plan-out, which refuses on NO-GO: the plan file is never written",
            "driver": f"scripts/{D5.BLIND_DRIVER} (pinned)"}


def new_models() -> dict:
    with D5.blind5():
        cells = {m: len(DC.page_doc(m, "main")["cells"]) for m in D6.NEW_BLIND}
        ctl = {m: len(DC.page_doc(m, "ctl")["cells"]) for m in D6.NEW_CTL}
    return {"models": {m: {"footprint (COMPUTED)": D5.footprint(m),
                           "timed decode pages (COMPUTED)": D4.no_decode_page(m)["verdict"], "main cells": cells[m],
                           "ctl cells (GR1)": ctl.get(m, 0)} for m in D6.NEW_BLIND},
            "main cells": sum(cells.values()), "unseen check (COMPUTED)": _out("check_unseen.out"),
            "reach under v6d (COMPUTED after the freeze; configs and page lists only)": D6.new_model_reach(),
            "GD5 offline (tuned config)": D5.gd5_offline()}


def expected() -> dict:
    return {
        "label": ("ESTIMATES (mine, addendum 5's method: judgement on the seen analogues' re-score and the reach table), "
                  "not measured; the blind is NOT RUN, so none is ever tested. No pre-gate estimate was recorded before the "
                  "gate ran; the gate's reading is computed"),
        "the gate reads GO on these trees": "no: it reads NO-GO (COMPUTED), clause 1 GO, clause 2 4.52% against 4.0%",
        "had it run, the pooled bar (3% / 8% / 1.5%) holding": (
            "about 0.07: Granite-1B is the likely breaker (its nearest seen model, Granite-3B, re-scores at 5.85% with a "
            "+2.1% mean and holds nine of the re-score's twelve worst cells, +9.2% to +15.6%: the law prices its short "
            "16x64x128 slabs slow, as it prices PC2's no-follower naive GEMMs, +9.9% mean); Granite-1B's slabs are half as long, it runs 24 align iterations and its "
            "block-scan E 32 is held at E 40"),
        "had it run, at least one new model CERTIFIED": (
            "about 0.05, bounded by the pooled estimate (certification needs the pooled bar, score_blind.py). DESCRIPTIVE "
            "per-model odds at a model's own bars: Qwen2-57B about 0.45 (deepseek 1.39% and the aligned Qwen1.5 cells as "
            "analogues; every GEMM multi-wave; its 7 naive cells on the law branch), Mixtral-8x22B about 0.35 (CAL2's "
            "Mixtral-8x7B in sample; the 1.5% mean bar is tight; 11 GEMMs past PC's 3.67 MB slab), Granite-1B about 0.05"),
        "cost not spent": "about $4.24 (72 unit-min, 111 min wall at $2.29/h), $5.34 at the hard cap (re-checked: "
                          "d5common.blind_rental recomputes addendum 5's figures)"}


def build() -> dict:
    gate, glines = gate_block()
    build.GATE_LINES = glines
    pc, pc2, sp = D6.committed()
    r = D5.blind_rental()
    return {
        "registered": f"{D6.DATE}, before any page of Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M",
        "name": D6.NAME,
        "status": ("REGISTERED BEFORE ANY NEW BLIND PAGE: the decode stage's sixth and last addendum: model v6d (frozen), "
                   "the gate and addendum 5's new blind under v6d; the gate reads NO-GO on the published trees (clause 1 "
                   "GO, clause 2 FAILS); under the owner's stopping rule decode closes as calibrated, not certified: the new "
                   "blind is registered and NOT RUN; every earlier decode registration stays byte-identical (pinned here) "
                   "and every registered verdict stands"),
        "design": DESIGN,
        "tool": (f"scripts/scoring/decode/gate_d6.py <readings dir> {D4.CAL2_PUBLISHED}/results "
                 f"{D4.BLIND_PUBLISHED}/results [out] (the readings dir holds pc.read.json, pc2.read.json, pc_spans.json "
                 f"and ds.readings.json; its output on the published trees: {GATE_OUT}); finalise_decode_d6.py and "
                 "score_blind_d6.py refuse on NO-GO and are not run"),
        "decision (owner's stopping rule, 2026-11-10)": CLOSED,
        "model v6d": {
            "module": "scripts/r3_timing_model_v6d.py (over v5d over v4d', both byte-identical)",
            "form": V6D.describe(),
            "law constants": V6D.LAW.as_dict(),
            "terms (v5d's registered terms, regimes and W1 scale emptied)": D6.registered_terms().as_dict(),
            "constants source": ("d6common.law_from on the committed readings (pc.read.json, pc2.read.json, "
                                 f"{D6.SPANS}, ds.readings.json), mechanically; equal to the model's numbers (COMPUTED): "
                                 + ("yes" if not D6.same_law(D6.law_from(pc, pc2, sp), D6.registered_law()) else "NO")),
            "BLOCKING readings (COMPUTED)": D6.blocking(pc, pc2, sp) or "none",
            "freeze (d6_build/FROZEN.txt)": _out("FROZEN.txt"),
            "reduction (COMPUTED)": D6.reduction(),
            "decisions": decisions(),
            "CAL2 alternatives (COMPUTED before the freeze, alternatives.out)": _out("alternatives.out"),
            "forking paths": forking_paths(),
        },
        "gate": gate,
        "after the gate": after_gate(),
        "blind (owner, 2026-11-04, D4-1; addendum 5's pages and units, registered again in addendum 6)": blind_block(),
        "new blind design": {
            "new models": new_models(),
            "rules (frozen otherwise)": ("addendum 5's: the power check first; E1-DEC pooled and per model without the "
                                         "second-tile DESCRIPTIVE cells; GD2 voids a model's GRR page; G3-D shuffle voids; "
                                         "G3-D label DESCRIPTIVE; SIGMA-D and the knee rows DESCRIPTIVE; the rivals by "
                                         "rental 7's rule; the scope line d3common.SCOPE"),
            "bars": "the owner's (D4-1): rms 3%, worst 8%, |mean| 1.5%, per model and pooled, not tightened",
            "finalisation": ("finalise_decode_d6.py, mechanical: the gate (refuses unless GO), then finalise_decode.frozen "
                             "and .predict (frozen) under d6common.v6d and d5common.blind5 on the registered pages"),
            "scope line": D3.SCOPE,
        },
        "rentals": {
            "new blind": r | {"composition": "prelude 3 + calibrate 2 + three GRR main pages "
                                             + " + ".join(str(u["est_min"]) for u in D5.blind_units()[:3])
                                             + " + two GR1 ctl pages "
                                             + " + ".join(str(u["est_min"]) for u in D5.blind_units()[3:])},
            "status": "NOT RUN: the gate reads NO-GO; nothing is rented",
            "launch precondition (not met)": "gate_d6.py prints GO",
        },
        "expected (ESTIMATES, not measured)": expected(),
        "unchanged": ("the lock 1710 and G4 1705, the instruments, every registered page (addendum 5's blind5 pages "
                      "included), the census rule, the tile source, the card constants, v4d, v4d' and v5d (byte-identical), "
                      "every pin of the earlier decode registrations, the drivers, vm_run.sh, docs/LAMBDA.md, and every "
                      "registered verdict (the decode blind FAIL, the CAL2 GO under addendum 3, addendum 5's NO-GO)"),
    }


def text(d: dict) -> list[str]:
    x = d["rentals"]["new blind"]
    g = d["gate"]["the gate on the published trees (COMPUTED after the freeze)"]
    L = [f"REGISTERED {d['registered']}: the decode stage (a sixth addendum), {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", ""]
    L.append(f"GATE (COMPUTED after the freeze): {g['line']}")
    L.append(f"decision: {d["decision (owner's stopping rule, 2026-11-10)"]}")
    L.append(f"new blind rental (registered, NOT RUN): {x['units_min']} min of units, about {x['wall_min']} min wall with "
             f"{x['boot_setup_min']} min of boot and setup: ${x['cost_usd']:.2f} at ${D2.RATE}/h; deadline launch + "
             f"{x['deadline_after_launch_min']} min; guardian hard cap launch + {x['hard_cap_after_launch_min']} min "
             f"(${x['cost_at_hard_cap_usd']:.2f} at most)")
    red = d["model v6d"]["reduction (COMPUTED)"]
    L.append(f"reduction: {red['verdict']} ({red['identical to v5d with the law off']} of {red['cells']} page cells identical "
             "to v5d with the law off)")
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool",
                                  "decision (owner's stopping rule, 2026-11-10)"):
            continue
        _flat("", {k: v}, L)
    return L


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT:
                out.append(f"{prefix}{k}: (in the JSON)")
                continue
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


def pins() -> dict:
    prev = json.loads((REG / f"{D5.NAME}.json").read_text())["code_pins"]
    paths = tuple(dict.fromkeys(tuple(prev) + FROZEN + D6_CODE + BUILD + tuple(D5.page5_path(m, k) for m, k in D5.PAGE_KEYS)))
    return {p: RGD.sha256(REPO / p) for p in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    doc = build()
    gtext = "\n".join(build.GATE_LINES) + "\n"
    jpath, tpath = a.out_dir / f"{D6.NAME}.json", a.out_dir / f"{D6.NAME}.txt"
    if a.write:
        (REPO / GATE_OUT).write_text(gtext)
    doc["code_pins"] = pins()
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    if a.check:
        strip = lambda x: {k: v for k, v in x.items() if k != "code_pins"}  # noqa: E731
        same_j = jpath.exists() and strip(json.loads(jpath.read_text())) == strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        no_plan = not (REPO / D6.BLIND6_PLAN).exists() and not (REPO / D5.BLIND5_PLAN).exists()
        same_g = (REPO / GATE_OUT).exists() and (REPO / GATE_OUT).read_text() == gtext
        pages = all((REPO / D5.page5_path(m, k)).read_text() == t for (m, k), t in
                    zip(D5.PAGE_KEYS, D5.page5_files().values(), strict=True))
        for name, same in ((jpath.name, same_j), (tpath.name, same_t), ("no blind plan (NOT RUN)", no_plan),
                           (Path(GATE_OUT).name, same_g), ("addendum 5's pages", pages)):
            print(f"{name}: {'same' if same else 'DIFFERS'}")
        return 0 if same_j and same_t and no_plan and same_g and pages else 1
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {GATE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
