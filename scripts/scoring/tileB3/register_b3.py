#!/usr/bin/env python
"""Writes tile rental B3's registration, docs/registered/2026-11-04-tileB3-gh200.{json,txt}, before any B3 page.

    python scripts/scoring/tileB3/register_b3.py --write [--out-dir docs/registered]
    python scripts/scoring/tileB3/register_b3.py --check      # recompute, compare (exit 1 on any difference)

B3 is the BLIND test of the round-5 large-tile model (moe-kernels-work 2026-10-28 round5/ROUND5.md's selected variant)
on rental B2's 84 frozen NATIVE cells, at the owner's working-model bar (2026-11-04: rms <= 3% per model, every cell
within 8%, |mean| <= 1.5%, per blind model and pooled), the model frozen as round 5 selected it (no amendment) with
its 64x128 sag measured on B3's board. It discloses the per-call risk the B3 per-call check found and why the align
amendment tried against it is not adopted. The registration says, before any page: which cells, which
on-board measurements enter (MEASURED-ON-BOARD) and how, the stop rule enacted by the driver before any timed unit,
the bars and the certification rules, the scope of what a pass certifies, every SEEN item, and the model FROZEN by
sha256 with its constants by value, its predictions on the 84 cells at A2's, C2's and C4's pars (computed here, before
the rental) and the fingerprint the reader reproduces before it reads a page.

Building it imports the frozen chain (round5/work r7.py over round 4's r6: the CAL pricer is built at p 14, about 60 s),
prices the 84 cells at three pars with the rivals, re-prices round 5's selected five-model table and reproduces round 5's
estimate at B2's bar. `--check` recomputes everything and compares with the committed JSON (code_pins excluded, as every
registration since rental 3) and text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import b3common as B  # noqa: E402
import read_b3 as R3  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = B.NAME
PLAN = B.PLAN
DESIGN = ("moe-kernels-work 2026-11-04 build-B3 (DESIGN.md v3; offline, read-only over the repo, built at bfc45176 from "
          "the v2 patch and rebased onto 3160bed5, v1 having been built at 82364213), after round5/ROUND5.md (the final declared round, its gate FAILED), the scope decision "
          "(docs/FINDINGS.md 'Tile stage: scope decision (2026-10-28)'), design-v5-review REVIEW.md and the B3 per-call "
          "check (moe-kernels-work 2026-11-04 b3-percall REPORT.md)")
#: every repo file B3's driver, gates, census, finaliser, scorer and register run, and the repo modules the frozen chain
#: and the reader import (read_b3.CHAIN_REPO_MODULES)
PINNED = (
    "scripts/gh200_tileB3_session.sh", "scripts/gh200_cal2_session.sh", "scripts/gh200_settleS1_session.sh",
    "scripts/gh200_tileA2_session.sh",
    "scripts/tileA2_gate.py", "scripts/tile_census_a2.py", "scripts/tile_census.py", "scripts/r3_tile.py",
    "scripts/dram_counter_route_tile.py", "scripts/dram_counter_route.py", "scripts/private_weight_reference.py",
    "scripts/locked_r3.py", "scripts/calibrate_hardware.py",
    "scripts/scoring/tileB3/b3common.py", "scripts/tileB3_gate.py", "scripts/scoring/tileB3/read_b3.py",
    "scripts/scoring/tileB3/register_b3.py", "scripts/scoring/cal2/read_c2.py",
    "scripts/scoring/tileA2/census_a2.py", "scripts/scoring/tileA2/score_floor_a2.py", "scripts/scoring/tileA/score_floor.py",
    "scripts/scoring/tileA2/census-merged/2026-10-20-nvidia_gh200_480gb-tile-census-merged-ta2m/tile-census.json",
    "scripts/skew_synth_tile.py", "scripts/skew_synth.py",
)
PINNED = tuple(dict.fromkeys(PINNED + R3.CHAIN_REPO_MODULES))
BOOT, DEADLINE, CAP, RATE = 25, 186, 201, 2.29

SOTA = ("the working-model bar is set against published accuracy for this problem, none of it pre-registered: TileSight "
        "(arXiv 2607.22432) 10.35% wMAPE on vLLM MoE configurations end to end (12.35% pooled MAPE on GEMM time); "
        "PipeWeave (ISCA 2026, arXiv 2601.14910) 6.0% on seen GPUs and 11.5% on unseen GPUs for Triton fused_moe, "
        "trained on 33,264 samples; NeuSight (ASPLOS 2025) 8.9% on unseen GPUs (~/reports/MoE kernel model prior art.md). "
        "A blind pass at rms <= 3%, every cell within 8% and |mean| <= 1.5% per model is a factor of 2 to 4 tighter than "
        "those mean errors and is registered before any page")


RELATION = ("B3's earlier gate (round 4's, applied to round 5: G1 every held-out 128 G 1 mean within 1.0%, Mixtral-8x7B on "
            "C2's board too; G2 an estimated P(A) >= 0.5 at B2's bar 2% / 5% / 1%) was a MODEL-ROUND gate deciding whether to "
            "register B3 against B2's bar. Round 5 FAILED it (max held-out mean +2.39, tp2; P(A) 0.124) and that FAIL stands: "
            "this registration does not reverse it, and the model registered here is round 5's selected model exactly as it "
            "failed, not a model that passed (the align amendment tried after it is NOT adopted: the 'per-call risk' block). "
            "B3 is registered because the owner chose to go blind (ROUND5.md section 7's second option) under a new bar for "
            "every new blind test, the working-model bar, decided 2026-11-04 after round 5's numbers were known (disclosed in "
            "SEEN). Under the new bar the same estimate method gives A 0.428 (still under the old G2's 0.5; 0.30 to 0.40 with "
            "the per-call risk the B3 per-call check estimated) and a 1.5% G1 would still fail (tp2 +2.39, Mixtral-8x7B "
            "+1.77 / +2.01): the registration is made knowingly below both old thresholds. B2's own bar (2%, 5%, 1%) is "
            "printed beside every reading, DESCRIPTIVE: a B3 pass certifies at 3 / 8 / 1.5 only, never at B2's bar. A FAIL "
            "is final and spends Phi-3.5-MoE, Qwen1.5-MoE-A2.7B and JetMoE-8B (and deepseek-v2-lite) as blind models: after "
            "B3 they are SEEN whatever the verdict")

PERCALL = {
    "statement": ("DISCLOSED RISK, NOT IN THE MODEL. Round 5's model prices every NATIVE block-scan call's per-call work (the "
                  "align kernels, the silu_and_mul and moe_sum intercepts, launch gaps, the event overhead) as ONE constant, "
                  "T0 + s_block = 43.23 - 7.07 = 36.16 us, fitted on the 2026-09-27 R3 Mixtral-8x7B pages at numel 256 to "
                  "1536 and 1710 MHz; it does NOT price the block-scan align by numel. B3's 84 cells span numel 1,024 (Phi "
                  "64g1) to 46,080 (Qwen1.5 128g16), every one on the block-scan path"),
    "the B3 per-call check (moe-kernels-work 2026-11-04 b3-percall REPORT.md, offline, read-only)": {
        "the align kernel": ("the launch-floor traces (rentals 1 to 3, CUPTI, R3's old tile, 1710 MHz) put the two block-scan "
                             "align kernels at 4.663 + 0.532 us per 1000 ids (9 points, numel 1,024 to 11,520, largest "
                             "residual 0.20 us; 5.65 + 0.645 at 1410 if SM-clock bound), E-independent; over B3's numel span "
                             "that is about 6 to 35 us against the model's constant budget, the fit measured only to 11,520"),
        "implied per-model error from the per-call terms alone, mean (CAL anchor, model byte rate)": {
            "phi-3.5-moe": "+0.28%", "qwen1.5-moe-a2.7b": "-0.71%", "jetmoe-8b": "+0.74%", "deepseek-v2-lite (pool)": "-0.85%",
            "reading": ("no B3 bar crossed by this term alone; with the estimate's common bias (+0.68%) JetMoE's expected "
                        "mean goes to about +0.9 to +1.4% and Phi-3.5's to about +0.9%; with Phi's large-P w1 risk Phi's "
                        "is about +1.9%, past the 1.5% |mean| bar. The +1% is ROUND5.md section 5's estimate that Phi "
                        "128g1 lands about 2 to 3% high if Phi behaves like tp2, spread over Phi's 30 cells (12 of the 30 "
                        "are 128g1), as the per-call check's REPORT.md derives it")},
        "the small-kernel byte terms": ("the model prices silu_and_mul and moe_sum by their bytes at 3597.8 GB/s; B3's cells "
                                        "move 80 to 625 MB through them, past the 60 MiB L2 and beyond the range the check's "
                                        "trace fits were measured on (and the model's rate was fitted at): extrapolated, so "
                                        "the check read three rate cases (R0, R1, R2)"),
        "estimate A at the working-model bar with the per-call shifts": (
            "0.428 without them; 0.30 to 0.40 under the CAL anchor (byte-rate cases R0 0.2995, R1 0.3271, R2 0.399; JetMoE "
            "alone 0.52 to 0.67); 0.46 to 0.52 under the Mixtral anchor; 0.14 with decode's unexplained JetMoE +5 us and "
            "Qwen1.5 +4 us carried to prefill (an INFERENCE, a stress case)"),
        "verdict of the check": "MARGINAL: the term alone crosses no bar, but it is unmodelled and lowers the estimate"},
    "the amendment tried, NOT ADOPTED (B3 v2, offline, 2026-11-04)": (
        "price + slope x (numel - 1280) / 1000 us on every NATIVE block-scan cell at the launch-floor slope (0.6451 us per "
        "1000 ids at 1410; T0 and s_block kept, the refit on the R3 fit pages differing by 0.196 us per call), on round 5's "
        "held-out cells: OLMoE +0.45 -> +2.01, Qwen2-57B -1.46 -> -1.09, Mixtral-8x7B +1.77 -> +1.80 (C2 +2.01 -> +2.04), "
        "Mixtral-8x22B +0.24 -> +0.25, tp2 +2.39 -> +2.44; rms of the five means 1.50 -> 1.70, pooled cell rms 1.64 -> "
        "1.91; estimate A 0.428 -> 0.278. It worsened the held-out fit (OLMoE, the one CAL model with large numel, 8,192 "
        "to 32,768, moved +1.56 points: the frozen model's OLMoE agreement holds a compensation the align term removes, "
        "INFERENCE, not identified), so the owner did not adopt it (2026-11-04). No on-board align unit is run. The numbers "
        "are SEEN, from moe-kernels-work 2026-11-04 build-B3 DESIGN.v2.md and work-v2/amend.json, not recomputed here"),
    "consequence": ("the per-call error is a known, unmodelled risk of this registration. It pushes JetMoE's 128 G 1 mean "
                    "positive and Qwen1.5's and deepseek-v2-lite's negative (the check's signs), it is not in the registered "
                    "estimate below, and a FAIL driven by it is still a FAIL"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_units() -> list[dict]:
    """The plan's units as the driver parses them (model, step, label, est, cap, nodrop or drop group, keys)."""
    out = []
    for line in (REPO / PLAN).read_text().splitlines():
        body = line.split("#", 1)[0].split()
        if not body:
            continue
        kv = dict(w.split("=", 1) for w in body[2:])
        drop = "nodrop" if (body[1] == "prelude" or kv.get("nodrop") == "1") else kv.get("drop-group")
        keys = {k: v for k, v in kv.items() if k not in ("label", "est", "cap", "nodrop", "drop-group")}
        out.append({"model": body[0], "step": body[1], "label": kv.get("label", ""), "est": int(kv["est"]),
                    "cap": int(kv["cap"]), "drop": drop, "keys": keys})
    return out


def cell_list(preds: list) -> list:
    """The 84 blind NATIVE cells with B2's per-row labels (tileB2-e1tile nominal_cells, NATIVE rows)."""
    b2 = json.loads(R3.B2_REG_PATH.read_text())
    lab = {(c["model"], c["page"], int(c["n"]), c["label"]): c for c in b2["nominal_cells"] if c["arm"] == "native"}
    out = []
    for c in preds:
        n = lab[(c["model"], c["page"], c["n"], c["label"])]
        out.append({"index": c["index"], "model": c["model"], "page": c["page"], "unit": c["unit"], "cls": c["cls"],
                    "G": c["G"], "n": c["n"], "label": c["label"], "counts_sha256": n["counts_sha256"],
                    "blind_label": n["label_blind"]})
    return out


def build() -> dict:
    units = plan_units()
    est = sum(u["est"] for u in units)
    nodrop = sum(u["est"] for u in units if u["drop"] == "nodrop")
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    wall = est + BOOT
    P64 = R3.A2_PAR_64
    # ---- the frozen model as round 5 priced it (A2's CARRIED 64x128 sag): the fingerprint and the stakes
    fz = R3.predictions(R3.A2_PAR_128, raw=True)
    assert all(c["ms"] is not None for c in fz), "a blind cell is UNPRICED"
    sk = R3.stakes(fz)
    fp = R3.fingerprint()
    assert fp["ms"] == [c["ms"] for c in fz]
    c5 = json.loads((R3.ROUND5 / "combo5.json").read_text())
    assert {k: v["vs_v5pp"] for k, v in sk.items()} == {k: v["vs_v5pp"] for k, v in c5["b2"].items()}, "round 5's stakes"
    assert {k: v["vs_round4"] for k, v in sk.items()} == {k: v["vs_round4"] for k, v in c5["b2"].items()}
    t5 = R3.round5_table()
    assert t5 == R3.ROUND5_TABLE, t5
    pe_b2 = B.pass_estimates(rule=B.B2_BAR)
    assert {k: v["E1-TILE"] for k, v in pe_b2.items()} == R3.ROUND5_PASS_B2_BAR, pe_b2
    pe = B.pass_estimates(rule=B.BAR)
    # ---- the predictions before the rental: three boards' 128x128 pars, A2's 64x128 par by the registered estimator
    preds = {b: R3.predictions(p, raw=False, sag64=P64) for b, p in R3.REF_PARS.items()}
    assert all(c["ms"] is not None for v in preds.values() for c in v), "a blind cell is UNPRICED"
    a2 = preds["A2"]
    jet = a2[72]
    assert (jet["model"], jet["cls"], jet["n"], jet["label"]) == ("jetmoe-8b", "128g1", 2, "PT") and jet["ms"] is not None
    assert all(a["ms"] == f["ms"] for a, f in zip(a2, fz, strict=True) if a["cls"] != "64g1"), "only 64g1 takes the 64x128 par"
    by_model = {}
    for c in a2:
        by_model.setdefault(f"{c['model']} {c['cls']}", []).append(c)
    lever = {}
    for k in sorted(by_model):
        idx = [c["index"] for c in by_model[k]]
        lever[k] = {b: round(100 * (sum(preds[b][i]["ms"] / preds["A2"][i]["ms"] for i in idx) / len(idx) - 1), 2)
                    for b in ("C2", "C4")}
    a2res = REPO / "results" / "published" / "2026-10-21-nvidia_gh200_480gb-tileA2-session" / "results"
    g64 = B.sag_gate(a2res, B.SAG64_UNIT)
    assert g64["verdict"] == "PASS" and g64["par"] == list(P64), g64
    assert R3.chain_sag_fits(a2res, B.SAG64_UNIT) == {m: (g64["per_page"][m]["A_us"], g64["per_page"][m]["tau_us"])
                                                      for m in B.EST_MODELS}
    rows_b2 = R3.b2_rows()
    i64 = [c["index"] for c in a2 if c["cls"] == "64g1"]
    lever64 = {}
    for m in B.EST_MODELS:
        s64 = (g64["per_page"][m]["A_us"], g64["per_page"][m]["tau_us"])
        lever64[f"at {m}'s page fit {list(s64)}"] = round(100 * (sum(
            R3.price(rows_b2[i], R3.A2_PAR_128, s64)[0] / a2[i]["ms"] for i in i64) / len(i64) - 1), 2)
    sag = {b: {} for b in ("A2", "C2", "C4")}
    for b, root in (("A2", "2026-10-21-nvidia_gh200_480gb-tileA2-session"), ("C2", "2026-10-21-nvidia_gh200_480gb-cal2-session"),
                    ("C4", "2026-10-27-nvidia_gh200_480gb-cal4-session")):
        res = REPO / "results" / "published" / root / "results"
        g = B.sag_gate(res)
        frozen = R3.chain_sag_fits(res)
        assert all((g["per_page"][m]["A_us"], g["per_page"][m]["tau_us"]) == frozen[m] for m in B.EST_MODELS), (b, g, frozen)
        assert g["verdict"] == "PASS" and [round(g["par"][0], 2), g["par"][1]] == [round(R3.REF_PARS[b][0], 2), R3.REF_PARS[b][1]]
        sag[b] = {"per_page": {m: {k: g["per_page"][m][k] for k in ("A_us", "tau_us", "rms_us", "cells")} for m in B.EST_MODELS},
                  "par": g["par"], "ratio_1ms": g["ratio_1ms"], "gate": g["verdict"], "page": root}
    for c in fz:
        del c["raw"]
    d = {
        "registered": "2026-11-04, before any B3 page",
        "name": NAME,
        "status": ("BLIND: the registered verdict is E1-TILE at the working-model bar (owner, 2026-11-04) on rental B2's 84 "
                   "frozen NATIVE cells, priced by round 5's frozen model (no amendment); B2's registered bar is printed "
                   "beside, DESCRIPTIVE. Every B3 page is SEEN once read"),
        "design": (f"{DESIGN}; the owner's decisions of 2026-11-04: every NEW blind test is held to the working-model bar "
                   "(3% rms per model, 8% worst cell, 1.5% |mean|), registered before any page, with its reason; the 64x128 "
                   "sag measured on B3's board; B3 v3: the frozen model without the align amendment, the per-call risk "
                   "disclosed"),
        "tool": ("scripts/gh200_tileB3_session.sh (C2's pinned driver plus lines marked B3: the gates b3census, b3sag and "
                 "b3sag64), scripts/tileB3_gate.py (the three B3 STOP gates, on the VM), scripts/tileA2_gate.py band (A2's "
                 "G-BAND), scripts/tile_census_a2.py (the census), scripts/scoring/tileB3/read_b3.py finalise and score"),
        "bar": {
            "values": dict(B.BAR),
            "applies to": ("the pooled NATIVE cells AND each blind model (Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, JetMoE-8B); "
                           "deepseek-v2-lite in the pooled bar only, its own bar printed DESCRIPTIVE"),
            "decided": "owner, 2026-11-04: for every NEW blind test; registered here before any B3 page",
            "reason": SOTA,
            "B2's bar (DESCRIPTIVE beside)": dict(B.B2_BAR),
            "relation to the round-5 gate (stated plainly)": RELATION,
        },
        "per-call risk (disclosed, not modelled)": PERCALL,
        "seen": [
            ("every CAL page is SEEN: tile rentals A (1710 MHz) and A2, settle rental S1, calibration rentals C2, C3 and C4; "
             "OD13 kept: no time is priced against rental A's 1710 E1-CAL pages"),
            ("the launch-floor rentals 1 to 3 (2026-10-01, 10-02, 10-06; published): Granite-3.0-1B-A400M, Granite-3.0-3B-A800M, "
             "Mixtral-8x7B-tp8 and JetMoE-8B traces at R3's old tile (32x64x64), 1710 MHz. They give the launch-floor align "
             "slope (Granite-1B, Granite-3B and Mixtral-tp8 block-scan traces) and the per-call check's silu_and_mul and "
             "moe_sum fits, which JetMoE-8B, a B3 blind model, feeds (its align ran small-batch there); B3 v2's build read "
             "their parsed traces and cells to reproduce the slope"),
            ("the B3 per-call check (moe-kernels-work 2026-11-04 b3-percall: REPORT.md and work/ parts.py, lf_traces.py, "
             "lf_align.py, lf_bytes.py, percall.py, estimate.py and their outputs, read for this registration): the frozen "
             "model's per-call terms on the 84 blind cells and the held-out cells, the implied per-model error, estimate A "
             "with per-call shifts; it cites the DS rental's BLOCK readings (block-scan align at E 40 8.70 us and E 64 8.13 "
             "us, ncu) for the align's E-independence; no B3-tile page of a blind model was opened"),
            ("B3 v2 (moe-kernels-work 2026-11-04 build-B3 DESIGN.v2.md, CHANGES.v2.md, build.patch.v2, work-v2/): the align "
             "amendment computed offline (the T0 / s_block refit on the 2026-09-27 R3 fit pages through the frozen pricer, "
             "the held-out table and the estimate under it, the blind deltas: Qwen1.5 +1.1 to +1.4%, deepseek +1.3%, Phi and "
             "JetMoE 0.2% or less), NOT adopted (the per-call risk block)"),
            ("every constant of the frozen model is labelled in FROZEN constants; the fitted ones are SEEN-FITTED on the "
             "CAL models (the K1 knee on four CAL models' 138 NATIVE 128x128 counter rows, round 4), CARRIED from R3-tile "
             "fits (M4's theta1, C_B, beta_B: SEEN-INFORMED) or MEASURED-PERTURBED (sigma_gap, in-sample for Mixtral-8x7B)"),
            ("rounds 2 to 5 were each declared with the earlier rounds' held-out numbers in view; round 5 also after a "
             "diagnosis of C4's counted w1 bytes (diag5.out); its selected law STAG-CUT won the last tie-break against M4w1 "
             "by 0.03 points of pooled rms (6.85 against 6.88); the held-out means carry a forking-paths risk"),
            "tp2 (C4's CAL model) is a TP shard of Mixtral-8x7B and shares its router (and Phi-3.5's routing shapes, rental 6 owner decision 4)",
            ("the blind stakes were computed before this registration, model against model, no blind time: round 5 PART D "
             "(Phi-3.5 128g1 +3.64% against v5'', -0.40 to +6.68; +6.13% against round 4's model; JetMoE 128g1 +0.06; "
             "Qwen1.5 128g1 +0.52; every 128g16 class within 0.12 points; 64g1 equal to v5''), reproduced here"),
            ("the large-P w1 law is not identified (ROUND5.md section 2): Phi-3.5's w1 (P 100, K 4096) sits in tp2's regime "
             "where the declared laws read tp2's w1 bytes +7.12% (STAG-CUT), -7.25% (M4w1), -20.71% (W2); Phi's 128g1 stake "
             "against v5'' moved -2.32 to +3.64 across rounds 4 and 5"),
            ("B-BYTES at 128 G 1 w1 is SEEN-expected outside B2's byte bar: STAG-CUT's held-out w1 bytes on the CAL models read "
             "OLMoE +3.45% (rms 7.12), Qwen2-57B -6.16% (8.93), tp2 +7.12% (7.25); B-BYTES is DESCRIPTIVE here"),
            ("the knee conflict is unresolved (ROUND4.md section 6, ROUND5.md sections 3 and 7): with counted bytes the model "
             "prices Mixtral-8x7B 128 G 1 +1.62% (+1.86% on C2's board) and Qwen2-57B -1.21%"),
            ("this build's own computations before registering: the 84 blind predictions at A2's, C2's and C4's pars and their "
             "levers (the 128x128 sag, the 64x128 sag); the pass estimate at both bars; the sag gate run on A2's, C2's and "
             "C4's published f128 pages and the 64x128 sag gate on A2's f64w8 pages (every one PASSES); B3's census gate run "
             "on A2's own census (PASS)"),
            ("the owner's working-model bar was decided on 2026-11-04 with round 5's results known; the estimate at the new "
             "bar (A 0.428) and under the per-call risk (0.30 to 0.40) were computed before this registration and are "
             "disclosed beside it"),
            ("B2's 14 SEEN-WORKLOAD rows (a published R3-tile page timed the counts; only the tile is blind), deepseek-v2-lite "
             "SEEN-SHAPE (D7), each blind model's GEMM shapes timed at R3's tile on GH200 (rentals 5 to 7; v3 is CERTIFIED on "
             "Phi-3.5), and the blind models' cubins compiled on rental A's and A2's boards (compile only): B2's leakage block"),
            ("the design-v5 review's held-out numbers, the p_64 lever (Phi 64g1 at p_64 6.0 against 3.75 moves its cells -4.45% "
             "mean) and F4 at 128x128 (JetMoE 128g1 -1.51% mean): both levers are in the frozen v5'' parts unchanged"),
            "B2's noise model's SEEN inputs (rentals 3 and 7)",
            ("the 64x128 sag is SEEN on one board only (A2's f64w8 pages: OLMoE 24.79 us / 275, Mixtral-8x7B 35.94 / 325, par "
             "30.365 / 300); its range brackets that board with margin"),
            ("the decode CAL2 and blind rentals (2026-11-03 and 2026-11-04, published at b2fb3402, before this registration): "
             "every B3 blind model was timed at decode tiles (T 1 to 128 on 16x64x128 w4 s4, 32x64x128 w4 s3, 32x128x64 w4 s3 "
             "and the 64x128x64 w4 s3 wgmma tile at T 97 to 128) on board ae5fc5, under v4d', not this model; the decode blind "
             "test FAILS (pooled 11.4% rms; Phi-3.5 mean +5.22%, JetMoE-8B +15.11%), read there as a call-level constant too "
             "large for these models (INFERENCE, not separated). B3's prefill tiles (w8 s3) and cells are in none of those "
             "pages; the per-call risk is disclosed above and is not in the estimate"),
            ("the decode stage's fourth addendum (docs/registered/2026-11-04-decode-addendum4-gh200, 3a46f182) and its "
             "settling rental DS, published at bfc45176 before this registration (results/published/2026-11-04-"
             "nvidia_gh200_480gb-decodeDS-session, scripts/scoring/decode/ds.readings.txt): wide decode counter pages on "
             "SEEN models including Phi-3.5, Qwen1.5 and JetMoE-8B, all DESCRIPTIVE. The DS summary in docs/registered/"
             "README.md was in view while this build merged that file: its per-call readings on decode cells of B3's blind "
             "models among others (CALLS: align medians 5.60 us JetMoE to 11.78 us Qwen1.5, silu_and_mul 3.87 to 6.46 us, "
             "moe_sum 2.67 to 4.24 us; ALIGN small-batch E 8 to 64 4.75 to 11.63 us; BLOCK E 40 8.70 us, E 64 8.13 us; "
             "SLOPE 0.527 us per loop iteration), and the per-call check cites DS BLOCK (E 40 8.70, E 64 8.13 us). This "
             "build did not open the DS pages or the readings files. Nothing in B3 is fitted on any DS number"),
            ("the decode stage's fifth addendum (3160bed5: v5d fails its own CAL clause, its gate NO-GO, no decode model "
             "certified) and the rental2 register fix (f843d397), committed before this registration; the addendum 5 "
             "summary in docs/registered/README.md was in view while this build merged that file. Nothing in B3 is "
             "fitted on it"),
        ],
        "frozen": {
            "statement": ("The round-5 model (round5 ROUND5.md's selected variant: the STAG-CUT w1 byte law with W2 on w2, the "
                          "K1 knee in the wall domain at round 4's SEEN constants, the clock form of F1, F1 'par'; 64x128 cells "
                          "at v5'') is FROZEN as of this registration. No constant below is refitted, re-chosen or re-read "
                          "from a B3 page; the on-board inputs are the sag par of the board's two f128 pages and the 64x128 "
                          "sag par of its two f64w8 pages (MEASURED-ON-BOARD), each entered by the registered estimator. The "
                          "reader refuses unless every work file below has its sha256, the frozen model reproduces the "
                          "fingerprint on the 84 cells at A2's par and round 5's selected five-model table, and every repo "
                          "module loaded is pinned. THE FREEZE IS TIED TO "
                          "THE RENTAL'S COMMIT, as C2's and C4's: read_b3 takes the commit from the ledger's 'driver start: "
                          "commit <sha>' line (gh200-driver/status, checked against session/commit.txt) and refuses unless "
                          "git show <sha>:<path> equals the working copy for the registration JSON and text, the plan and every "
                          "code_pins path, every code_pins path has its registered sha256, and every repo module loaded from "
                          "the chain's repo (diag-A2 common.REPO) and from the reader's repo is a code_pins path with its "
                          "registered bytes: a repo module is a git-tracked file under that repo, outside tests/ and any "
                          ".venv, venv, site-packages or dist-packages directory, with an absolute __file__ "
                          "(read_c2.repo_modules, C2's rule)"),
            "chain_repo_modules": list(R3.CHAIN_REPO_MODULES),
            "root": "~/code/moe-kernels-work (paths below relative to it); the chain imports the repo at ~/code/moe-kernels",
            "files": R3.frozen_hashes(),
            "constants": {
                "w1 byte law STAG-CUT (no fitted constant)": ("M4's staggered rank distance on w1 slab x events, the carried "
                                                              "soft survival (C_B 72.52 MiB, beta_B 1.324), theta1 0.1967 in the "
                                                              "first window; a later-wave w1 re-read whose fill reaches the 60 MiB "
                                                              "L2 misses (r7.law5('STAG-CUT')); form declared after a SEEN "
                                                              "diagnosis (DECLARED5.txt)"),
                "w2 byte law W2": ("round 4's LS law: later-wave cross-group slab re-reads at D_eff = W_c dt / L from list "
                                   "scheduling (R 16, seed 20261027 x 1000 + i), sigma_gap 0.0375 (MEASURED-PERTURBED, in-sample "
                                   "for Mixtral-8x7B), the carried soft survival, the 60 MiB cut; the first window, in-group "
                                   "and A events M4's (v3 key, CARRIED)"),
                "L2 capacity": "60 MiB (the page card): MEASURED",
                "W_c": "264 = 132 SMs x 2 CTAs per SM at 128x128 (A2's census, C2's and C4's): MEASURED; B3's census gate holds it",
                "K1 knee (p_128, S0)": "(6.0, 2.75) k-steps: SEEN-FITTED (round 4, knee4.json K1 SEEN); no blind model is in any fit",
                "K1 rate": "the A2 triad 3726.2 GB/s: MEASURED (rental A2)",
                "clock form of F1": ("fac_g = 1 + D_net(t0) / t0 on the floor side, D_net(t) = A (1 - exp(-t / tau)) - 6.11 (1 - "
                                     "exp(-t / 50 us)) us, t0 the GEMM's sag-free time (r6.v5_parts_clock): no new constant"),
                "F1 (A, tau) at 128x128": ("MEASURED-ON-BOARD: 'par' of B3's OLMoE f128 and Mixtral-8x7B f128 fits "
                                           "(b3common.page_fit = refit4 r4.sag_page_fit = design-v5 sag.fit over sag.pts_of at "
                                           "1410), A and tau averaged (r4.par)"),
                "R3's 1410 share": "6.11 us, tau 50 us: MEASURED (rental 3), CARRIED",
                "T0, c, bw, F4": ("43.23 us, 202.548 ns, 3597.8 GB/s (asserted at import, the first-build trap at p 14), rho x 2: "
                                  "CARRIED; s_block -7.07 us, s_small 12.99 us: CARRIED (one per-call constant at every numel: "
                                  "the disclosed per-call risk)"),
                "64x128 cells (Phi-3.5 64g1, 6 cells)": ("v5'' as round 5 PART D priced them (k5.cell_ms(r, None, 5.75, par, "
                                                        "par64) under r5.law3(): null knee, p_64 3.75 (SEEN-FITTED, a tail "
                                                        "compensator per the design-v5 review), M4 / F3 bytes, F4 x 2) at the "
                                                        "64x128 sag par64 MEASURED-ON-BOARD: 'par' of B3's OLMoE f64w8 and "
                                                        "Mixtral-8x7B f64w8 fits by the same estimator (gate b3sag64; the "
                                                        "owner's choice; round 5 CARRIED A2's 30.37 us / 300 us, which the "
                                                        "fingerprint keeps)"),
                "selections and scoring rules": "A2's (moe-kernels-work 2026-10-20 scoreA2 via finalise.selections); a2common.Rules.tileA2(1410)",
                "JetMoE-8B 128g1 n2 PT (B2 index 72)": ("PRICED: under W2 on w1 (round 4) its fixed point had two roots and the "
                                                        "cell was UNPRICED; under STAG-CUT the guard refuses no blind cell "
                                                        "(confirmed here: 84 of 84 priced at A2's, C2's and C4's pars). B3 "
                                                        "scores all 84 cells; the unpriced rule stays: a cell the finalised "
                                                        "model cannot price is listed and left out"),
            },
            "round 5 selected table, re-priced by the frozen code at registration": t5,
            "round 5 estimate at B2's bar on 84 cells, reproduced": {k: v["E1-TILE"] for k, v in pe_b2.items()},
            "fingerprint": fp,
            "predictions before the rental (frozen model at three boards' 128x128 pars and A2's 64x128 par; the rental's "
            "own pars finalise)": {b: {"par": list(R3.REF_PARS[b]), "par64": list(P64), "ms": [c["ms"] for c in preds[b]]}
                                   for b in preds},
            "levers: mean prediction at C2's and C4's par over A2's, % (the board's sag on the prediction)": lever,
            "levers: Phi-3.5 64g1 mean prediction at A2's per-page f64w8 fits over their par, %": lever64,
            "blind stakes, reproduced (round 5 PART D: frozen model against v5'' and round 4, A2's par)": sk,
            "rivals at A2's par (v5'', round 4's model, v4)": [c["rivals"] for c in a2],
        },
        "cells": {"source": f"docs/registered/{B.B2_NAME}.json (pages, files, power cells: B2's, unchanged)",
                  "count": 84, "unpriced": [], "list": cell_list(a2),
                  "SHARED": "each timed page also runs SHARED cells (B2's instrument); they carry no B3 prediction and are not scored"},
        "measured_on_board": {
            "sag (F1 at 128x128)": ("B3-F: OLMoE f128 and Mixtral-8x7B f128 (128x128x64 w8 s3, G 64, treads 1 to 9, NATIVE, "
                                    "SHARED and PRIVATE: C2's and C4's units), fitted on the VM by gate b3sag; par enters every "
                                    "128x128 prediction"),
            "sag (F1 at 64x128)": ("B3-F64: OLMoE f64w8 and Mixtral-8x7B f64w8 (64x128x64 w8 s3, G 64: A2's units), fitted on "
                                   "the VM by gate b3sag64; par64 enters the six Phi-3.5 64g1 predictions"),
            "power": "700 W set and read back by C2's prelude (c2-power.env C2_POWER_W=700)",
            "lock": "1410 MHz locked and read back by the prelude (locks.env LOCK_PROBE_1410=held)",
            "G-BAND": ("A2's G-BAND on the two f128 pages (primary [0.975 L, L], else the fallback [0.959 L, L]); the band in "
                       "force sets the counter band of the byte pages"),
            "census": "B3's compile-only census (label tb) of the blind models at both tiles",
            "sag drift (DESCRIPTIVE)": "B3-R: OLMoE f128 again at the end (label f128r, droppable), never priced",
            "board references, SEEN (gate b3sag run on the published pages)": sag,
            "64x128 board reference, SEEN (gate b3sag64 run on A2's f64w8 pages)": {
                "per_page": {m: {k: g64["per_page"][m][k] for k in ("A_us", "tau_us", "rms_us", "cells")} for m in B.EST_MODELS},
                "par": g64["par"], "ratio_1ms": g64["ratio_1ms"], "gate": g64["verdict"], "clock": g64["clock"],
                "page": "2026-10-21-nvidia_gh200_480gb-tileA2-session"},
        },
        "stop_rule": {
            "where": "every STOP is a driver stop before the first timed unit; a STOP ends the session (card reset, results pushed)",
            "power": ("C2's prelude: nvidia-smi -pl 700 -sc 0, then enforced.power.limit and -q -d POWER's GPU-scope Current "
                      "Power Limit both at 700 W within 1 W, polled every 2 s up to 30 s, else STOP; no fallback limit"),
            "lock": "the prelude locks 1410, reads clocks.sm back (two tries) and STOPS when it does not hold",
            "census": ("gate b3census (tileB3_gate.py census) after the census unit: B3's census equals A2's merged census on every "
                       "compiled fact (tile_census_a2.FIELDS) for Phi-3.5-MoE, Qwen1.5-MoE-A2.7B and JetMoE-8B at "
                       "128x128x64w8s3 and 64x128x64w8s3, every merged entry present, no MMA path mismatch, else STOP; "
                       "deepseek-v2-lite drifting is no STOP (its page is NOT SCORED)"),
            "band": "A2's G-BAND (tileA2_gate.py band) after the two f128 pages, else STOP",
            "sag-fit range": {"gate": "b3sag (tileB3_gate.py sag) after G-BAND", "range": {k: (list(v) if isinstance(v, tuple) else v)
                                                                                      for k, v in B.SAG_RANGE.items()},
                              "a lost f128 page": "no fit: FAIL, a STOP",
                              "held-out par span (DESCRIPTIVE, no STOP)": {k: list(v) for k, v in B.HELD_OUT_PAR_SPAN.items()},
                              "why": ("each clause brackets the SEEN boards with margin (A2, C2, C4 per page: A 29.65 to 67.55 "
                                      "us, tau 300 to 650 us, rms 4.02 to 13.01 us, 54 points; ratio 1.52 to 1.59; par A 41.59 "
                                      "to 54.25): a fit outside it is an instrument or board fault, not a sag the frozen "
                                      "clock form was read at. design-v5's rms <= 10 us would have stopped every SEEN board "
                                      "(Mixtral's pages 10.25 to 13.01), so it is 20 here")},
            "64x128 sag-fit range": {"gate": "b3sag64 (tileB3_gate.py sag64) after the two f64w8 pages",
                                     "range": {k: (list(v) if isinstance(v, tuple) else v) for k, v in B.SAG64_RANGE.items()},
                                     "a lost f64w8 page": "no fit: FAIL, a STOP",
                                     "clock clause": {"rule": ("r6common.clock_rule on each f64w8 page under A2's scoring "
                                                               "rules in the band in force (score_floor_a2.choose_band on "
                                                               "the f128 pages, as G-BAND): lock_gate.check_page PASS and "
                                                               "at least cells_in_band_min of 27 cells in band; the rule "
                                                               "B3's byte pages are read under"),
                                                      **B.CLOCK64,
                                                      "why this and not A2's G-64": (
                                                          "G-64 asks that a 64x128 CONV calibration (joint_cF) exists in "
                                                          "the band, a fit B3's pricing never reads; the clock rule checks "
                                                          "what the sag reading needs, that the page ran at the lock and its "
                                                          "cells' clocks sit in the band, per cell. Every cell in band is not "
                                                          "asked: the sag itself takes cells under the band (A2's f64w8 26 "
                                                          "and 22 of 27 in band; the f128 pages on A2, C2, C4 17 to 24)")},
                                     "why": ("each clause brackets A2's f64w8 pages (the only SEEN board: A 24.79 and 35.94 "
                                             "us, tau 275 and 325 us, rms 3.32 and 7.62 us, 54 points, ratio 1.42, par A "
                                             "30.37) with the 128x128 range's margins; every clause fails on a NaN")},
            "order": ["prelude (power, lock)", "calibrate", "tilecensus", "gate b3census", "OLMoE f128", "Mixtral-8x7B f128",
                      "gate band", "gate b3sag", "OLMoE f64w8", "Mixtral-8x7B f64w8", "gate b3sag64", "the timed units"],
        },
        "finalisation": {
            "before the rental": ("this registration: the frozen code and constants, the 84 predictions at A2's, C2's and C4's "
                                  "128x128 pars and A2's 64x128 par, the fingerprint (A2's par, A2's CARRIED 64x128 sag), the "
                                  "stakes and rivals"),
            "on the rental": ("gates b3sag and b3sag64 fit the four sag pages, apply the registered ranges and write par, par64 "
                              "and every page's sha256 to the ledger (gate-b3sag-1410.json, gate-b3sag64-1410.json) before the "
                              "first timed unit"),
            "after the rental, before any blind page is read": ("read_b3.py finalise: the freeze, the commit and the pins; the "
                                                                "fingerprint; both sag gate records PASS; every fit recomputed "
                                                                "by the frozen chain (r4.sag_page_fit; its design-v5 sag.fit on "
                                                                "the f64w8 pages) and by b3common.page_fit, equal to each other; "
                                                                "each par equal to its gate record's to a relative 1e-9 "
                                                                "(b3common.PAR_RTOL: the record is computed on the GH200, the "
                                                                "check on the laptop, the same estimator on the same page bytes, "
                                                                "so the tolerance absorbs float summation order only) and every "
                                                                "page sha256 equal; the f64w8 "
                                                                "clock clause recomputed; the 84 cells priced at par and par64; "
                                                                "an audit hook refuses the run if any path under the B3 tree in "
                                                                "a gaps-* directory, or naming a blind model (except the "
                                                                "compile-only census), is opened. Its file scripts/scoring/tileB3/"
                                                                "b3.predictions.json (sorted keys) is committed by the owner "
                                                                "before `score` runs; `score` re-finalises and refuses unless the "
                                                                "file equals the finalised file (a content check, not a git "
                                                                "check). No argument sets any constant"),
        },
        "tests": {
            "E1-TILE (the registered verdict)": {
                "statistic": ("e = T_model / T_meas - 1 per valid NATIVE cell (all six labels), the finalised prediction; "
                              "valid as B2: tcommon.timed_page (G5 provenance, r7-median-run3 rebuild, the view, the page's "
                              "tile) and tcommon.valid_cells (6 of 9 clean, G2 host on the finalised prediction and the timer "
                              "flag, G4 cell clock >= 1405 MHz)"),
                "rule": ("HOLDS iff the POOLED NATIVE cells AND each blind model's meet rms <= 3%, every cell within 8%, "
                         "|mean| <= 1.5% (b3common.e1_rule at BAR), in ALL and in CLEAN; the registered verdict is rental 2's "
                         "rule 3 over the two views (a disagreement reads INCONCLUSIVE), as B2's scorer combines them"),
                "deepseek-v2-lite": "in the pooled bar; its own bar printed DESCRIPTIVE (one page, never CERTIFIED); dropped (dsv) or census-drifted, it leaves the pool",
                "a blind model with no valid cell": "E1-TILE reads NOT SCORED (incomplete), not FAILS, unless the pool or another blind model fails",
                "G3": ("rental 7's TOST on uniform minus balanced, as r5common.g3_rule reads it: homogeneous and failing, every "
                       "histogram cell is voided and E1-TILE reads NOT SCORED (printed DESCRIPTIVE); heterogeneous, the failing "
                       "strata's non-balanced cells are voided and the verdict is read WITH those voids; the bar without them "
                       "is printed beside, DESCRIPTIVE. This replaces score_tileB.py's all-strata-only void for B3 (the "
                       "design-v5 review's MAJOR 9: the verdict view named before any page)"),
                "validity": ("NOT SCORED unless the board reads valid (c2-power.env C2_POWER_W=700, LOCK_PROBE_1410=held, "
                             "G-BAND PASS) and all three B3 gate records (census, sag, sag64) read PASS"),
            },
            "certification": ("a blind model is CERTIFIED when E1-TILE HOLDS, its own bar holds in CLEAN and at least 2 of its "
                              "pages are in CLEAN (Phi-3.5 3 pages, Qwen1.5 2, JetMoE 2: every one nodrop); a model left with "
                              "one page has its bar scored and is NOT CERTIFIED"),
            "DESCRIPTIVE beside (no verdict)": ["B2's bar (2%, 5%, 1%) on the same cells",
                                                "the bar without G3's per-stratum voids",
                                                "rivals: v5'', round 4's frozen model, v4 at A2's selections (e on the same cells)",
                                                "splits: 128 G 16, 128 G 1, 64 G 1, per model and class",
                                                "B-BYTES (predicted reads against counted, per GEMM; B2's byte bar) and E2-TILE "
                                                "(the frozen model's time from counted bytes, 128x128 cells)",
                                                "the sag drift (f128r against the early OLMoE fit)"],
        },
        "certification": {
            "scope": ("A B3 pass certifies the frozen round-5 model's NATIVE time at vLLM's two reachable wgmma tiles "
                      "(128x128x64 w8 s3 at G 1 and G 16, 64x128x64 w8 s3 at G 1; prefill) on the GH200 at 1410 MHz and "
                      "700 W, on the registered cells only (B2's 84: Phi-3.5-MoE 30, Qwen1.5-MoE-A2.7B 30, JetMoE-8B 18, "
                      "deepseek-v2-lite 6 pooled), at the working-model bar (rms <= 3%, every cell within 8%, |mean| <= "
                      "1.5%), each CERTIFIED model named. It does not certify B2's bar, any other lock, power limit, tile, "
                      "card or model, decode, SHARED cells, the byte law (B-BYTES is DESCRIPTIVE), the per-call constant at "
                      "other numel, or either sag on another board"),
        },
        "pass_estimate": {
            "method": ("ESTIMATE, by the knee round's method (read_c4.pass_estimates on five means): read_c2.power (diag-v5 "
                       "power5's, verbatim, the bar an argument: b3common.power) on B2's 84 registered NATIVE cells under B2's "
                       "noise model (cell 0.6%, bytes 1.75% x dlnT/dlnB, page 0.18%, board 0.26%), R 200,000, seed 20261022; "
                       "the held-out bias and spread from round 5's five held-out means (OLMoE +0.45, Qwen2-57B -1.46, "
                       "Mixtral-8x7B +1.77, Mixtral-8x22B +0.24, tp2 +2.39): A = power(the mean of the five b = -ln(1 + e), "
                       "their SD), B = power(0, their RMS), D = A with Mixtral-8x7B's G 1 set aside"),
            "at the working-model bar (3 / 8 / 1.5)": pe,
            "at B2's bar (2 / 5 / 1), reproduced": pe_b2,
            "with the per-call risk (the B3 per-call check, SEEN, not recomputed)": (
                "A 0.30 to 0.40 at the working-model bar under the check's CAL anchor (0.46 to 0.52 under its Mixtral anchor; "
                "0.14 in its decode-residual stress case)"),
            "reading": ("P(E1-TILE HOLDS) is about 0.43 (A), 0.57 (B), 0.47 (D) at the working-model bar, against 0.12 to 0.19 at "
                        "B2's. Each blind model alone passes about 0.73 to 0.79 of draws under A; the binding clause is the "
                        "per-model |mean| <= 1.5% against a model-specific SD of about 1.5%. The estimate does not carry the "
                        "per-call risk (A 0.30 to 0.40 with it, the 'per-call risk' block), the large-P w1 model-choice "
                        "uncertainty on Phi-3.5 128g1 (ROUND5.md section 5) beyond what the five means' spread holds, or the "
                        "board's 64x128 sag"),
        },
        "after": ("OWNER DECISIONS 2026-11-04 (registered): every NEW blind test at the working-model bar; B3 goes blind with "
                  "round 5's selected model as frozen here, without the align amendment (tried offline, worsened the held-out "
                  "fit, not adopted), with the per-call risk disclosed and the 64x128 sag measured on B3's board. A FAIL is "
                  "final: it spends the four blind models and changes no earlier verdict. Every registered verdict already "
                  "scored stands (A2's E1-CAL FAIL, C2's, C4's readings, round 5's gate FAIL)"),
        "plan": {"file": PLAN, "sha256": sha256(REPO / PLAN), "units_count": len(units), "units": units,
                 "est_total_min": est, "est_nodrop_min": nodrop, "drop_order_last_first": order,
                 "boot_setup_min": BOOT, "wall_min": wall, "rate_usd_per_h": RATE, "cost_usd": round(wall / 60 * RATE, 2),
                 "deadline_after_launch_min": DEADLINE, "hard_cap_after_launch_min": CAP,
                 "cost_at_hard_cap_usd": round(CAP / 60 * RATE, 2)},
        "pages": {"B2 registration": {"path": f"docs/registered/{B.B2_NAME}.json", "sha256": sha256(R3.B2_REG_PATH)},
                  "files": {k: {"path": v["path"], "sha256": v["sha256"]}
                            for k, v in json.loads(R3.B2_REG_PATH.read_text())["pages"]["files"].items()}},
    }
    TCA = B._census_mods()
    cen = B.census_check(json.loads((REPO / B.A2_MERGED_CENSUS).read_text()),
                         json.loads(B.find_census(a2res, "ta2").read_text()), TCA.facts, TCA.FIELDS)
    assert cen["verdict"] == "PASS" and cen["deepseek"] == "admitted", cen
    d["measured_on_board"]["census reference, SEEN (gate b3census's rule on A2's own census, label ta2)"] = {
        "verdict": cen["verdict"], "compared": cen["compared"], "ctas_per_sm": cen["ctas_per_sm"]}
    for v in d["pages"]["files"].values():
        assert sha256(REPO / v["path"]) == v["sha256"], v
    d["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    return d


def txt(d: dict) -> list[str]:
    p = d["plan"]
    fz = d["frozen"]
    L = [f"REGISTERED {d['registered']}: tile rental B3, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"tool: {d['tool']}", "", "BAR"]
    for k, v in d["bar"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L += ["", "PER-CALL RISK (disclosed, not modelled; the align amendment NOT ADOPTED)"]
    for k, v in d["per-call risk (disclosed, not modelled)"].items():
        if isinstance(v, dict):
            L.append(f"  {k}:")
            L += [f"    {kk}: {vv if isinstance(vv, str) else json.dumps(vv, sort_keys=True)}" for kk, vv in v.items()]
        else:
            L.append(f"  {k}: {v}")
    L += ["", "SEEN"] + [f"  - {s}" for s in d["seen"]]
    L += ["", "FROZEN", f"  {fz['statement']}", f"  root: {fz['root']}"]
    for k, v in fz["files"].items():
        L.append(f"  {k} {v}")
    for k, v in fz["constants"].items():
        L.append(f"  {k}: {v}")
    L.append(f"  round 5 selected table, re-priced by the frozen code at registration: {json.dumps(fz['round 5 selected table, re-priced by the frozen code at registration'])}")
    k5 = "round 5 estimate at B2's bar on 84 cells, reproduced"
    L.append(f"  {k5}: {json.dumps(fz[k5])}")
    lev = fz["levers: mean prediction at C2's and C4's par over A2's, % (the board's sag on the prediction)"]
    L.append("  levers (mean prediction at C2's / C4's par over A2's, %): " + "; ".join(
        f"{k} {v['C2']:+.2f} / {v['C4']:+.2f}" for k, v in lev.items()))
    L.append("  levers (Phi-3.5 64g1 at A2's per-page f64w8 fits over their par, %): " + json.dumps(
        fz["levers: Phi-3.5 64g1 mean prediction at A2's per-page f64w8 fits over their par, %"]))
    sk = fz["blind stakes, reproduced (round 5 PART D: frozen model against v5'' and round 4, A2's par)"]
    L.append("  blind stakes (frozen model against v5'' / against round 4, mean %): " + "; ".join(
        f"{k} {v['vs_v5pp']['mean']:+.2f} / " + (f"{v['vs_round4']['mean'] + 0.0:+.2f}" if v['vs_round4'].get('n') else "n/a")
        for k, v in sk.items() if "(all)" not in k))
    L.append("  fingerprint and the predictions at A2's, C2's and C4's pars: (in the JSON, 84 cells each)")
    L += ["", f"CELLS: {d['cells']['count']} NATIVE cells, {d['cells']['source']}; unpriced {d['cells']['unpriced']}; {d['cells']['SHARED']}"]
    L += ["", "MEASURED ON BOARD"]
    for k, v in d["measured_on_board"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v, sort_keys=True)}")
    L += ["", "STOP RULE"]
    for k, v in d["stop_rule"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L += ["", "FINALISATION"] + [f"  {k}: {v}" for k, v in d["finalisation"].items()]
    L += ["", "TESTS"]
    for k, v in d["tests"].items():
        if isinstance(v, dict):
            L.append(f"  {k}:")
            L += [f"    {kk}: {vv}" for kk, vv in v.items()]
        else:
            L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L += ["", f"SCOPE: {d['certification']['scope']}", "", "PASS ESTIMATE (ESTIMATE)"]
    for k, v in d["pass_estimate"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L += ["", f"AFTER: {d['after']}"]
    L += ["", f"PLAN {p['file']} sha256 {p['sha256']}: {p['units_count']} units, {p['est_total_min']} min estimated "
              f"({p['est_nodrop_min']} nodrop), drop order last first {p['drop_order_last_first']}"]
    for u in p["units"]:
        L.append(f"  {u['model']} {u['step']} {u['label']} est {u['est']} cap {u['cap']} {u['drop']}")
    L += [f"  wall about {p['wall_min']} min with {p['boot_setup_min']} min of boot and setup: ${p['cost_usd']:.2f} at "
          f"${p['rate_usd_per_h']}/h; deadline launch + {p['deadline_after_launch_min']} min; guardian hard cap launch + "
          f"{p['hard_cap_after_launch_min']} min (${p['cost_at_hard_cap_usd']:.2f} at most)", "", "PAGES (B2's, by sha256)"]
    L.append(f"  {d['pages']['B2 registration']['path']} {d['pages']['B2 registration']['sha256']}")
    for k, v in d["pages"]["files"].items():
        L.append(f"  {v['path']} {v['sha256']}")
    L += ["", "code_pins: (in the JSON)"]
    return L


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    doc = build()
    jpath, tpath = a.out_dir / f"{NAME}.json", a.out_dir / f"{NAME}.txt"
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(txt(doc)) + "\n"
    if a.check:
        bad = 0
        same_j = jpath.exists() and _strip(json.loads(jpath.read_text())) == _strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        for p, same in ((jpath, same_j), (tpath, same_t)):
            print(f"{p.name}: {'same' if same else 'DIFFERS'}")
            bad += not same
        return 1 if bad else 0
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
