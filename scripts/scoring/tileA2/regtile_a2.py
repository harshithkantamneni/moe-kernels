"""Tile rental A2's registrations (docs/registered/2026-10-20-tileA2-{census,floor,bytes,e1cal,dead,rep,convl,gates}-gh200),
rental B2's two (2026-10-20-tileB2-{e1tile,stop}-gh200) and the decode stage's tile-source addendum
(2026-10-20-decode-tilesource-gh200). Built for register_a2.py, which renders, pins and --checks them.

GENERATED OVER THE PINNED regtile (scripts/scoring/tileA/regtile.py): every tileA / tileB builder runs
unchanged inside a2common.scoring(tileA2 at L), so its numbers are v4 at L by P1's rule
(r3_timing_model_v4L.at_lock: c x 1710 / L at every tile, the roofline at L) and its names are A2's;
this module then replaces a WHITELIST of fields (the lock, G4, the roofline clock, the band, the tool
and plan lines, the status) and adds A2's own blocks (the owner's lock decision, the SEEN disclosure
of REVIEW-A2 5, the score directory of REVIEW-A2 6a, the fallback rung's numbers). The frozen
2026-10-14 files are never touched; tests/test_tileA2_registrations.py requires each tileB2 JSON to
equal its frozen tileB twin on every key outside the whitelist.

The ladder is 1710, 1410, 1305 (the owner's decision of 2026-10-20). At rung 1710 v4L is v4 exactly, so every
L-dependent number there is the frozen 2026-10-14 file's own; the numbers that move with L are registered at
1410 (the body of each file) and at 1305 beside it ("at the fallback rung 1305"); the selected lock is the
lock-hold probe's mechanical outcome (tileA2-gates), never a choice made after a page.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402

TC = A2.TC
REPO = A2.REPO
import census_a2 as CA  # noqa: E402
import hist_a2 as H  # noqa: E402
import probe_a2 as PR  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r3_timing_model_v4L as V4L  # noqa: E402
import regtile as RG  # noqa: E402
import tile_census_a2 as TCA  # noqa: E402
import wave_split_bytes_v4 as WV4  # noqa: E402

r6 = A2.r6
LT, L1, L2 = A2.LOCK_LADDER
REGISTERED = f"{A2.DATE}, before any tile rental A2 page"
DESIGN = ("design-A2 DESIGN.md (2026-10-20) corrected by design-A2-review REVIEW.md (the review wins: every MAJOR and "
          "MINOR fix), the build review (GO-WITH-FIXES) and the owner's decisions of 2026-10-20: A 900 W GPU POWER LIMIT AND "
          "THE LOCK LADDER 1710, 1410, 1305 (the first rung whose probe passes is the lock of every later unit; the 900 W "
          "limit changes the owner's 2026-09-26 700 W decision for tile rental A2, and rental B2 at the limit A2 held, "
          "only); rental B2 at A2's power limit and lock rung; decode uses A2's constants only once A2's floor calibration "
          "is scored; the probe strict (one LOST cell fails a rung); every other decision is the design's recommendation "
          "as amended by the review: OD3 the byte pages re-run, OD4 sigma_page re-measured, OD5 w4, ms and dk droppable, OD6 CONV-L "
          "a registered test, nodrop, not a stop clause (REVIEW 3a: timed at 1710 and at L, a per-cell ratio), OD7 decode "
          "stays at 1710 by addendum, OD8 the band [0.975 L, L] (REVIEW 2c: the scaled band a registered fallback), OD9 "
          "the calibrate gate under the lowest rung (build review fix 3), OD10 the scorer import order fixed in place, OD11 the probe thresholds of DESIGN "
          "6, OD12 the offline conversion check done before registering, OD13 rental A's timed pages against v4 forbidden")
OWNER_LOCK = {
    "decision (owner, 2026-10-20)": "a 900 W GPU power limit (GPU scope, -sc 0) and the lock ladder 1710, 1410, 1305 MHz: "
                                    "the probe tries 1710 first; the first rung whose probe passes is the run's lock",
    "changes": "the owner's 2026-09-26 decision (the GPU power limit 700 W at the start, -sc 0) FOR TILE RENTAL A2 AND RENTAL "
               "B2 ONLY (B2 runs at the limit and the lock rung A2 held, read from A2's score directory); every other "
               "rental's driver keeps 700 W",
    "why (DESIGN-A2 1, REVIEW-A2 1)": "rental A's wgmma pages sagged under the 1710 lock: reason 0x4 (SW power cap) only, 60 C "
                                      "at most; the smi trace went to 1290 MHz on the three heavy 128x128 pages; the "
                                      "inferred cause is the software power cap (GPU or module scope unresolved), so the "
                                      "limit rises to the card's maximum and the lock steps down only when the probe "
                                      "says the cap still bites",
    "ladder (owner, 2026-10-20)": f"{LT:g}, then {L1:g}, then {L2:g} MHz: the lock-hold probe's first rung that holds is A2's "
                                  "lock (tileA2-gates G-PROBE). At 1710 no clock conversion is used (v4L is v4 exactly) and "
                                  "CONV-L does not run; 1410 and 1305 carry the conversion",
    "conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)": "the offline check (tileA2-convl) reads v3's "
        "c-only conversion about 1.36% fast on R1's G 1 cells at 1410 (clock-attributable, the level cancelled), and about "
        "1.4x that at 1305 (ln(1710/1305) / ln(1710/1410)); against tileB2-stop's 0.75% class-mean clause, a 64 G 1 or "
        "128 G 1 class at those rungs can STOP on the carried constants, not the wgmma calibration (SEEN-expected, stated "
        "before any page)",
    "rules at L": {"at 1710": "rental A's own rules: the band [1640, 1710] (derived at that lock; the fallback band at 1710 is "
                              "the same), G4 1705, v4 itself",
                   "counter band": f"below 1710 [L x 0.975, L] = [{r6(A2.band_of(L1)[0])}, {L1:g}] at {L1:g}, "
                                   f"[{r6(A2.band_of(L2)[0])}, {L2:g}] at {L2:g} (2 x the 1.02% largest held-lock "
                                   "shortfall at 1410 on rental 3's tp4 floor, rounded up to 0.5%: a NEW rule in "
                                   "lock_gate's convention; the 1710 band [1640, 1710] is 4.1% wide and does not follow it)",
                   "fallback band (REVIEW-A2 2c)": "[L x 0.959, L], applied only when the primary leaves tcommon.joint_cF "
                                                   "None on the f128 pages; then every counter scorer's band, labelled",
                   "G4": "cell clock >= L - 5 MHz (1705, 1405, 1300)",
                   "dips": "r7-median-run3 at the page's own lock_mhz (a read under L - 15 is dipped: 1395 is not at 1410, 1380 is)",
                   "lock_gate": "lock_gate.RATIO_REF holds 1710 only (lock_gate.py:127), so at L check_page reads the nvidia-smi "
                                "bracket and the null / spread readings, no base-twin level (REVIEW-A2 2d)",
                   "v4 at L": "r3_timing_model_v4L.at_lock(L): c x 1710 / L at every tile, R3's included (P1's rule, eta 1); "
                              "the roofline at L"},
}
SEEN = {
    "label": "SEEN before this registration (REVIEW-A2 5): none of it is a lever, the selection rules are tcommon's pinned "
             "functions unchanged (variant_rule, f_form_rule, joint_cF, floor_test); stated so a reader can weigh it",
    "(i) rental A's out-of-band fits (DESIGN-A2 7d, work/fit.py, DESCRIPTIVE)": "tcommon.joint_cF on rental A's f128 pages with "
        "the band removed: c_128 698.67 +- 10.63 cycles, F_w1 2740 +- 615, F_w2 3113 +- 341, 36 cells, rel rms 0.53%, recorded "
        "occupancy 2; on the f64w8 pages c_64 459.1 +- 19.4; the cells' clocks 1601 to 1663 MHz",
    "(ii) the 64x128 accounting outcome": "predictable by interpolation from (i) and tileA-floor's registered table (W and T "
                                          "c_64 at c_128 600 and 750); nobody computed it",
    "(iii) SEEN-expected": "with c_128 near 700, accounting B is FALSIFIED by construction (its pipe alone, 769 cycles, exceeds "
                           "c_128) and the kappa comparison is NOT SCORED (it needs c_128 >= 900)",
    "(iv) tA-dead": "rental A measured the lever: delta 20.006 us, se 0.612; D 10.02, D2 10.10, SLOT 24.69, d 1.06 11.41 us, "
                    "each EXCLUDED: A2's dk pages are SEEN-REPLICATE (DESCRIPTIVE), not a test",
    "(v) the A key": "rental A's key test on 19 of 56 rows: v3 5.0254% against CLAMP 5.0253% rms, NOT NAMED (CLAMP): the near "
                     "tie is known; the 53 rows its band dropped stay unread",
    "(vi) sigma_page": "rental A's 0.1275% (24 cells, at a sagging 1710 clock); A2 re-measures it, and never copies rental A's in",
    "(vii) what chose the lock, the probe cells and the gates": "rental A's per-repeat clocks (cells.csv), the in-flight NVML "
        "reads, the smi traces (locked-r3/*/smi.csv: clock, power, temperature, reasons) and the calibrate log",
    "(viii) rental A's times": "rental A's 1710 times of every E1-CAL cell and its counted bytes of every byte row exist "
        "(results/published/2026-10-20-nvidia_gh200_480gb-tileA-session) UNREAD against v4: no v4 time is computed against "
        "the E1-CAL times, and the 53 band-dropped byte rows are not re-scored, before or after A2 (OD13, a rule of this "
        "registration); its rep and dead pages are scored and published ((iv), (vi)); A2's E1-CAL, byte, rep and dead "
        "pages read the same histogram files, so their workloads are SEEN while their times on A2's board are not; the "
        "per-row workload labels below are tileA's (as of 2026-10-14) and are not updated",
    "(ix) the offline conversion check (OD12; REVIEW-A2 3b)": "see tileA2-convl 'offline_check (SEEN)': on the 2026-09-27 and "
        "R1 pages v3's c-only rule holds on the c-heavy G 4 / G 2 ladders and on the G 1 per-M-tile elasticity, and "
        "underpredicts the G 1 per-call clock elasticity",
}
SCORE_DIR = {
    "A2's score directory (REVIEW-A2 6a; build review fix 4)": f"{A2.SCORE_DIR_A2}/ (the score files committed beside A2's "
                                           "scorers, as rental A's are in scripts/scoring/tileA/), every file written by A2's own "
                                           "wrappers (score_*_a2.py) from A2's pages at the lock the driver selected; no rental-A "
                                           "score file is copied in",
    "session.score.json": "score_session_a2.py: the lock rung the probe selected (a2-lock.env) and the GPU power limit read "
                          "back after -pl (two readings within 1 W): rental B2's lock and limit (finalise_a2.py)",
    "census.score.json": "the NAMED MERGE STEP (score_census_a2.py, census_a2.merge): rental A's 72 entries (held to A2's by "
                         "the equality gate) plus A2's deepseek-v2-lite entries at tread 3 when the OLMoE n 1 / n 3 rule "
                         "reads SAME, scored by score_census.py; the merged file is kept beside it",
    "floor.score.json": "score_floor_a2.py at L (the band in force recorded)",
    "bytes.score.json": "score_bytes_a2.py at L, in the band in force; NOT SCORED or NOT NAMED gives CLAMP by tcommon's rule",
    "rep.score.json": "score_rep_a2.py at L; dropped, lost or not scored: tcommon.sigma_page_of's fallback, rental 3's 0.18% "
                      "(rental A's 0.1275% is never copied in)",
    "e1cal.score.json": "score_e1cal_a2.py at L (v4 at_lock L from A2's selections)",
    "dead.score.json, convl.score.json": "score_dead_a2.py (SEEN-REPLICATE) and score_convl_a2.py",
    "order": "session, census, floor, bytes, rep, e1cal, dead, convl, probe (each reads the ones before it)",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def at(lock: float):
    """The tileA builders' module state at `lock` with A2's names (a2common.scoring)."""
    return A2.scoring(A2.Rules.tileA2(lock))


def _fresh_e1tile():
    RG._E1TILE.clear()
    try:
        return RG.e1tile()
    finally:
        RG._E1TILE.clear()


def head(part: str, status: str, tool: str) -> dict:
    return {"registered": REGISTERED, "name": A2.NAMES_A2[part], "status": status, "design": DESIGN, "tool": tool}


def files_block_a2(names) -> dict:
    """A2's histogram pages (docs/registered/2026-10-20-tileA2-hist), as regtile.files_block writes rental A's."""
    out = {}
    for nm in names:
        p = H.path_of(nm)
        doc = json.loads(p.read_text())
        bm = int(doc["block_m"])
        t = RG.PAGE_TILE.get(bm, V4.R3_TILE if bm == 32 else None)
        out[f"{nm}.json"] = {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p), "shuffle_seed": doc["shuffle_seed"],
                             "tile": [t.bm, t.bn, t.bk, t.warps, t.stages], "group_m": doc["group_m"],
                             "source": doc["synthetic"],
                             "cells": [{"label": c["label"], "n": c["n"], "arms": c["arms"],
                                        "counts_sha256": RG.R6.sha_counts(c["counts"])} for c in doc["cells"]]}
    return out


def _merge(base: dict, over: dict) -> dict:
    """base with `over`'s keys replacing (in place, keeping base's order) or appended."""
    out = dict(base)
    out.update(over)
    return out


# --------------------------------------------------------------------------
# tileA2-census
# --------------------------------------------------------------------------

def census() -> dict:
    with at(L1):
        base = RG.census()
    ref = REPO / CA.RENTAL_A_CENSUS
    rdoc = json.loads(ref.read_text())
    treads = {m: {t.key: TCA.tread_of(m, t.bm) for t in V4.VLLM_TILES} for m in RG.CAL + RG.BLIND_TILE + (RG.DSV,)}
    return _merge(base, {
        **head("census", "REGISTERED BEFORE ANY A2 PAGE (an addendum to tileA-census: rental A's VALID (INCOMPLETE) stands)",
               "scripts/tile_census_a2.py (compile only) in A2's tilecensus unit (scripts/gh200_tileA2_session.sh); the merge "
               "step and scorer scripts/scoring/tileA2/score_census_a2.py (census_a2.py); the gate scripts/tileA2_gate.py census"),
        "unit": "- tilecensus label=ta2, A2's fourth unit, nodrop, before any lock-hold probe: rental A's six tiles on the seven "
                "models, each model at its smallest whole tread, and OLMoE again at n 3",
        "rental_A_census": {"path": CA.RENTAL_A_CENSUS, "sha256": sha256_file(ref), "entries": len(rdoc["entries"]),
                            "errors": rdoc["errors"], "verdict": rdoc["verdict"],
                            "copied_from": "results/published/2026-10-20-nvidia_gh200_480gb-tileA-session/results/"
                                           "2026-10-20-nvidia_gh200_480gb-tile-census-mixtral-8x7b-ta/tile-census.json "
                                           "(verbatim; the same sha256)"},
        "a2_rules": {
            "tread (DESIGN-A2 5b)": "each model compiles at the smallest n with n x BLOCK_M x E divisible by k "
                                    "(tile_census_a2.tread_of); deepseek-v2-lite n 3 at every tile, every other model n 1",
            "treads": treads,
            "equality gate (DESIGN-A2 3.1; tileA2-gates G-CEN)": "every (model, tile, GEMM) rental A compiled must be in A2's "
                                                                 "census with every compiled fact equal ("
                                                                 + ", ".join(TCA.FIELDS) + "); any difference is STACK "
                                                                 "DRIFT: the gate ends the rental, and a drifted census "
                                                                 "scores REFUSED (a STOP; finalise_a2 refuses)",
            "OLMoE n 1 / n 3 (REVIEW-A2 8a)": "OLMoE compiled again at n 3 on every tile; any field differing from its n 1 entry "
                                              "makes deepseek-v2-lite's entries NOT SCORED (left out of the merge), and "
                                              "rental B2's deepseek pages stay NOT SCORED as today",
            "merge (REVIEW-A2 6a)": SCORE_DIR["census.score.json"],
            "rental_B2_scope": "rental B2's own census unit (label tb, tile_census_a2.py) is held to A2's merged census by "
                               "score_tileB.census_check, unchanged"},
        "labels": {"census": "compile only: no time", "deepseek-v2-lite at n 3": "the tread rental B's deepseek page times; "
                   "Triton specialises integers divisible by 16 and num_valid_tokens and EM are at n 3 as at OLMoE n 1 "
                   "(REVIEW-A2 8, inference), which the n 1 / n 3 rule checks"},
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN, "a2_score_directory": SCORE_DIR,
    })


# --------------------------------------------------------------------------
# tileA2-floor
# --------------------------------------------------------------------------

def floor() -> dict:
    with at(L1):
        base = RG.floor()
    m = copy.deepcopy(base["model_v4"])
    m["eta_mma"]["why_per_clock (build-tile-review 1)"] = ("v4 counts MMA cycles at a held lock (A2: L, 1410 MHz), so eta is a "
                                                           "per-clock efficiency; Table 11 is the only wgmma rate with its "
                                                           "clock recorded (1260 MHz, nearer 1410 than 1710 is)")
    m["v4 at L (A2)"] = OWNER_LOCK["rules at L"]["v4 at L"]
    p = copy.deepcopy(base["parameters"])
    p["hardware (no fit)"] = ("132 SMs, A2's lock L (1410 MHz; the ladder's 1305 when the probe selects it), 4096 flop/clk/SM "
                              "dense bf16, 128 B/clk/SM shared, the sm_90 occupancy limits; c_card converted to L by P1's rule")
    with at(L2):
        lev2 = RG.lever_counts_B()
    return _merge(base, {
        **head("floor", "REGISTERED BEFORE ANY A2 PAGE",
               "scorer scripts/scoring/tileA2/score_floor_a2.py (scripts/scoring/tileA/score_floor.py inside a2common.scoring); "
               "tcommon.joint_cF, floor_test, variant_rule, f_form_rule, unchanged; model v4 at L "
               "(scripts/r3_timing_model_v4L.py over scripts/r3_timing_model_v4.py)"),
        "model_v4": m, "parameters": p,
        "counter band at L (A2)": {
            "primary": OWNER_LOCK["rules at L"]["counter band"],
            "fallback (REVIEW-A2 2c)": OWNER_LOCK["rules at L"]["fallback band (REVIEW-A2 2c)"],
            "derivation and its limit (REVIEW-A2 2a)": "the 1.02% is rental 3's tp4 floor (mma.sync, R3's tile) at 1410: most of "
                                                       "the held-lock shortfall at 1710 is real sub-lock clock (inference: it "
                                                       "shrinks about 0.4x at 1410 and 0.02x at 1005, not like a fixed-ns "
                                                       "counter artifact), so the rule is calibrated on an mma.sync tile's "
                                                       "residual sag; wgmma cells could read 1.5 to 2.2% low at 1410, inside "
                                                       "2.5% but near its edge; G-BAND checks it on the first f128 pages",
            "lock_gate": OWNER_LOCK["rules at L"]["lock_gate"]},
        "expected outcomes (SEEN, DESIGN-A2 7d)": SEEN["(iii) SEEN-expected"],
        "selection_for_rental_B (D3)": _merge(base["selection_for_rental_B (D3)"],
                                              {"lever_counts_on_B at the fallback rung 1305": lev2}),
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    })


# --------------------------------------------------------------------------
# tileA2-bytes
# --------------------------------------------------------------------------

def bytes_() -> dict:
    with at(L1):
        base = RG.bytes_()
    g = dict(base["gates"])
    g["clock"] = ("rental 6's counter clock rule at A2's lock: lock_gate.check_page PASS at the page's lock, each cell in the "
                  "counter band in force (tileA2-floor: [0.975 L, L], or the registered fallback [0.959 L, L])")
    return _merge(base, {
        **head("bytes", "REGISTERED BEFORE ANY A2 PAGE",
               "scorer scripts/scoring/tileA2/score_bytes_a2.py (scripts/scoring/tileA/score_bytes.py inside a2common.scoring, "
               "in the band floor.score.json records); tcommon.named_key; bytes scripts/wave_split_bytes_v4.py"),
        "gates": g,
        "re-run (OD3)": "the four byte pages re-run at L give the registered rule its 56 rows; bytes are clock-invariant (base and "
                        "lock bytes agree to 0.03%), so every byte prediction below is tileA-bytes' own number; rental A's "
                        "printed verdict (NOT NAMED, CLAMP on 19 rows) stands, and A2's key is rental B2's",
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    })


# --------------------------------------------------------------------------
# tileA2-e1cal
# --------------------------------------------------------------------------

def nominal_ms(lock: float, pages) -> list:
    with at(lock):
        rows = RG.nominal_rows(pages, "A")
    return [[r["page"], r["label"], r["n"], r["arm"], r["ms_nominal"]] for r in rows]


def e1cal() -> dict:
    with at(L1):
        base = RG.e1cal()
    g = dict(base["gates"])
    g["G4"] = f"cell clock >= L - 5 MHz ({LT - 5:g} at {LT:g}, {L1 - 5:g} at {L1:g}, {L2 - 5:g} at {L2:g})"
    g["G1"] = "rental7-g1rule (r7-median-run3 rebuilt from cells.csv at the page's own lock; 6 of 9 clean; 10% non-clean cap)"
    return _merge(base, {
        **head("e1cal", "REGISTERED BEFORE ANY A2 PAGE",
               "scorer scripts/scoring/tileA2/score_e1cal_a2.py (scripts/scoring/tileA/score_e1cal.py inside a2common.scoring); "
               "tcommon.bar, class_means, stop_rule; v4 at A2's constants and lock (finalise.predict_pages under "
               "r3_timing_model_v4L.at_lock)"),
        "predictions": "COMPUTED AT A2'S CONSTANTS AND LOCK by finalise.predict_pages under at_lock(L) (mechanical: the merged "
                       "census occupancy, c_128 and F_wg, the selected accounting or v4cal, the F form and the A key, all by "
                       "the registered rules); the DESIGN-NOMINAL numbers below (at 1410) show the scale only",
        "gates": g,
        "re-measured (DESIGN-A2 3.1)": "all five E1-CAL pages re-run at L: the calibration is in cycles at L and v4 converts at "
                                       "L; two of rental A's five failed G1 at 1710 (one cause, the clock: two STOP clauses); "
                                       "mixing locks across the five would break A3",
        "conversion (UNTESTED here)": "at rung 1710 none (v4 itself); at 1410 or 1305, T0, bw, s_small, s_block, h, CORES rho, "
                                      "d and MVA2's eta, z0, z1 carry unconverted from 1710; E1-CAL tests the conversion only "
                                      "jointly with c_128; tileA2-convl tests it on R3's tile",
        "conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)": OWNER_LOCK[
            "conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)"],
        "nominal_ms at the fallback rung 1305 (page, label, n, arm, ms)": nominal_ms(L2, RG.A_TIMED_PAGES),
        "a2_score_directory": SCORE_DIR, "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    })


# --------------------------------------------------------------------------
# tileA2-dead and tileA2-rep
# --------------------------------------------------------------------------

def dead() -> dict:
    with at(L1):
        base = RG.dead()
    with at(L2):
        inc2 = {f"c128 {c:g}": RG.dead_increments(c) for c in (600.0, 900.0, 1100.0)}
    return _merge(base, {
        **head("dead", "REGISTERED BEFORE ANY A2 PAGE: SEEN-REPLICATE (DESCRIPTIVE), NOT A TEST (REVIEW-A2 5 iv)",
               "scorer scripts/scoring/tileA2/score_dead_a2.py (scripts/scoring/tileA/score_dead.py inside a2common.scoring); "
               "tcommon.dead_rule"),
        "laws at L (A2)": "the v2 forms at the tile under r3_timing_model_v4L.at_lock(L): the hidden window kappa x occ x S' x c reads the converted "
                          "c, d stays in ns (P1's form); the other form (d x 1710 / L) moves NATIVE's dead term by at most 0.4 "
                          "us here, so D against SLOT (12 to 17 us apart) does not depend on it",
        "laws_predicted_us at the fallback rung 1305": inc2,
        "label (A2)": SEEN["(iv) tA-dead"],
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    })


def rep() -> dict:
    with at(L1):
        base = RG.rep()
    return _merge(base, {
        **head("rep", "REGISTERED BEFORE ANY A2 PAGE",
               "scorer scripts/scoring/tileA2/score_rep_a2.py (scripts/scoring/tileA/score_rep.py inside a2common.scoring); "
               "tcommon.rep_sigma, sigma_page_of"),
        "re-measured (OD4)": "OLMoE's E1-CAL page and its replicate at L (drop-group rep)",
        "fallback (design-tile-review 9.8)": "dropped, lost, or under 3 shared cells: rental 3's rep8 sigma_page 0.18% (R3's tile), "
                                             "labelled as such; rental A's SEEN 0.1275% (a sagging 1710 clock) is never copied in",
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    })


# --------------------------------------------------------------------------
# tileA2-convl (REVIEW-A2 MAJOR 3a)
# --------------------------------------------------------------------------

#: (page, model, G, class, label at L, label at 1710, est min)
CONVL_UNITS = (("convl-mixtral-8x7b-g1", "mixtral-8x7b", 1, "G1", "cvm1", "cvm1t", 3),
               ("convl-olmoe-1b-7b-g1", "olmoe-1b-7b", 1, "G1", "cvo1", "cvo1t", 2),
               ("convl-mixtral-8x7b-g8", "mixtral-8x7b", 8, "G8", "cvm8", "cvm8t", 5))
OFFLINE = "docs/registered/2026-10-20-tileA2-hist/conv_offline.json"


def convl_predictions() -> list[dict]:
    pr = RG.pricer()
    out = []
    for page, model, G, cls, _a, _b, _e in CONVL_UNITS:
        doc = json.loads(H.path_of(page).read_text())
        for c in doc["cells"]:
            seed = None if c["counts"] is None else doc["shuffle_seed"]
            ms = {}
            for lk in (1710.0, L1, L2):
                with V4L.at_lock(lk):
                    p = WV4.price(pr, model, "native", int(c["n"]), c["counts"], RG.COPIES, seed, V4.R3_TILE, G, V4.Params(), key="v3")
                ms[str(int(lk))] = r6(p["ms"])
            out.append({"model": model, "page": page, "G": G, "class": cls, "n": int(c["n"]), "label": c["label"],
                        "counts_sha256": RG.R6.sha_counts(c["counts"]), "ms": ms,
                        "ratio": {str(int(lk)): r6(ms[str(int(lk))] / ms["1710"]) for lk in (L1, L2)},
                        "host_ok_1710": TC.C6.host_ok(ms["1710"]),
                        "T0_share_1710": r6(float(pr.f2.x[0]) / ms["1710"])})
    return out


def convl() -> dict:
    preds = convl_predictions()
    off = REPO / OFFLINE
    offd = json.loads(off.read_text())
    units = {"at_L": [{"page": p, "model": m, "G": G, "class": c, "label": a, "est": e} for p, m, G, c, a, _b, e in CONVL_UNITS],
             "at_1710": [{"page": p, "model": m, "G": G, "class": c, "label": b, "est": e} for p, m, G, c, _a, b, e in CONVL_UNITS]}
    return {
        **head("convl", "REGISTERED BEFORE ANY A2 PAGE: a registered test, nodrop, NOT a stop clause (OD6)",
               "scorer scripts/scoring/tileA2/score_convl_a2.py; v3 (= v4 at R3's tile) priced by scripts/wave_split_bytes_v4.py "
               "under r3_timing_model_v4L.at_lock"),
        "what": "CONV-L (REVIEW-A2 fix 2): the conversion of v4's carried constants from 1710 to L, alone: the SAME R3-tile anchor "
                "pages timed on A2's board at 1710 and at L, scored as a per-cell ratio against v3's predicted ratio under "
                "P1's rule, so the board's level and v3's own level error cancel",
        "pages": {"files": files_block_a2(H.CONVL_PAGES), "units": units, "unless_lock": 1710,
                  "why these": "R3's tile (BLOCK_M 32, the CAL tile of c_card) held 1710 on rentals 6 and 7; Mixtral 8x7B is "
                               "the CAL model; at G 1 Mixtral's tread-3 cells are DRAM-bound (bw at the lower L2 clock) and "
                               "OLMoE's are short (T0 about 7% of the cell: the T0 carry); at G 8 the cells are c-heavy (the "
                               "P1 share); v3 is ILL-POSED at G 1 from tread 6 on Hopper, so G 1 runs tread 3 only",
                  "label": "SEEN-WORKLOAD (rentals 6 and 7 timed these counts at 1710 on other boards); the ratio is new"},
        "predictions": preds,
        "statistic": "per NATIVE cell valid on both pages: r = (T_L / T_1710) / (P_L / P_1710) - 1, P v3 at rental 6's constants "
                     "with c x 1710 / L (P1's rule) and every other constant carried",
        "at rung 1710 (owner, 2026-10-20)": "no conversion is used: CONV-L is NOT SCORED and its six units carry unless-lock=1710, "
                                            "so the driver skips them by rule (about 20 min saved)",
        "rule": {"classes": ["G1", "G8"],
                 "bar": "the E1 bar on r per class (rms <= 2%, worst <= 5%, |mean| <= 1%); CONV-L HOLDS iff both classes hold, "
                        "in the CLEAN view (ALL printed); a class with no cell valid at both locks: NOT SCORED",
                 "validity": "each page rebuilt under r7-median-run3 at its own lock; rental 6's row rule (6 of 9 clean, the host "
                             "rule on that lock's prediction above and the timer flag, G4 at the page's lock - 5); G5 against "
                             "the files above; the page's recorded tile R3's (32x64x64 w8 s4)",
                 "not a stop clause (OD6)": "CONV-L FAILING stops nothing; it localises a failure of E1-CAL"},
        "reading (DESCRIPTIVE, REVIEW-A2 3a)": {
            "CONV-L HOLDS, E1-CAL FAILS": "indicates the wgmma calibration or v4's wgmma terms, not the conversion; a carried "
                                          "constant can still weigh more on 128x128 G 1 cells (more bytes, bw at the lower L2 "
                                          "clock) than on R3's tile, so the conversion is not ruled out",
            "CONV-L FAILS, E1-CAL FAILS": "indicates the conversion as at least one cause",
            "CONV-L FAILS, E1-CAL HOLDS": "indicates the conversion errs on R3's tile and not at the wgmma cells",
            "both HOLD": "both pass their bars"},
        "offline_check (SEEN, OD12; REVIEW-A2 3b and 3c)": {
            "file": OFFLINE, "sha256": sha256_file(off), "tool": "scripts/scoring/tileA2/conv_offline.py (--check recomputes it)",
            "7e (the 2026-09-27 eta pages, R3's tile, G 4 and G 2 against the same session's 1710 deep pages)":
                offd.get("headline", {}).get("7e"),
            "3b (R1's lock pages, G 1 per-M-tile elasticity)": offd.get("headline", {}).get("3b"),
            "reading": "the c-only rule holds on the c-heavy ladders and on the G 1 per-M-tile elasticity (not far from 0.312), "
                       "and UNDERPREDICTS the G 1 per-call clock elasticity: at 1410 v3 reads R1's G 1 n 2..5 cells 2.0 to 2.2% "
                       "fast in level, 1.36% fast in the clock-attributable ratio (level cancelled), about 1.4x that at 1305; "
                       "CONV-L's G1 class may fail its mean clause on this (SEEN-expected, stated before the page). Taken to "
                       "the owner (build review MAJOR 1): the ladder now starts at 1710, where no conversion is used",
            "labels (REVIEW-A2 3c)": "eta 1 is SEEN-SELECTED (P1 scored eta 1 against 0.35 on the 2026-09-27 pages); the c-only "
                                     "carry is a design choice made after P1 held (SEEN-SELECTED); the 0.6 to 0.8% P1 shortfall "
                                     "is SEEN and unmodelled; cycle invariance is SEEN for mma.sync only (R3's floor 350 to "
                                     "367 cycles at any clock; K3 moves 6 to 9% from 1005 to 1710); nothing here is fitted on "
                                     "rental A's data"},
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
        "seen_data": ["rentals 6 and 7's R3-tile pages of these counts", "the 2026-09-27 eta pages and R1's lock pages (offline check)"],
    }


# --------------------------------------------------------------------------
# tileA2-gates: the session's rules, the plan
# --------------------------------------------------------------------------

#: A2's plan, unit by unit: (step, model, label, est, cap, nodrop or drop group); the plan file must say exactly this
PLAN_UNITS = (
    ("prelude", "-", "", 1, 15, "nodrop"),
    ("calibrate", "mixtral-8x7b", "ta2", 1, 30, "nodrop"),
    ("gate", "-", "gcal", 1, 5, "nodrop"),
    ("tilecensus", "-", "ta2", 2, 25, "nodrop"),
    ("gate", "-", "gcen", 1, 5, "nodrop"),
    ("lockprobe", "-", "lp", 20, 70, "nodrop"),
    ("bytes", "olmoe-1b-7b", "f128", 4, 20, "nodrop"),
    ("bytes", "mixtral-8x7b", "f128", 8, 25, "nodrop"),
    ("gate", "-", "gband", 1, 5, "nodrop"),
    ("bytes", "olmoe-1b-7b", "f64w8", 4, 20, "nodrop"),
    ("bytes", "mixtral-8x7b", "f64w8", 7, 25, "nodrop"),
    ("gate", "-", "g64", 1, 5, "nodrop"),
    *[("timed", m, a, e, 15, "nodrop") for _p, m, _G, _c, a, _b, e in CONVL_UNITS],
    *[("timed", m, b, e, 15, "nodrop") for _p, m, _G, _c, _a, b, e in CONVL_UNITS],
    ("timed", "olmoe-1b-7b", "e128g1", 14, 30, "nodrop"),
    ("timed", "mixtral-8x7b", "e128g16", 14, 40, "nodrop"),
    ("timed", "mixtral-8x7b", "e64g1", 10, 30, "nodrop"),
    ("timed", "qwen2-57b-a14b", "e128g1", 10, 35, "nodrop"),
    ("timed", "mixtral-8x7b", "e128g1", 5, 20, "nodrop"),
    ("bytes", "olmoe-1b-7b", "b128g1", 2, 20, "nodrop"),
    ("bytes", "mixtral-8x7b", "b64g1", 4, 25, "nodrop"),
    ("bytes", "mixtral-8x7b", "b128g16", 7, 25, "nodrop"),
    ("bytes", "qwen2-57b-a14b", "b128g1", 5, 25, "nodrop"),
    ("timed", "olmoe-1b-7b", "e128g1r", 14, 30, "rep"),
    ("bytes", "olmoe-1b-7b", "f64w4", 3, 20, "w4"),
    ("bytes", "olmoe-1b-7b", "f32w4", 3, 20, "ms"),
    ("timed", "olmoe-1b-7b", "d9", 3, 15, "dk"),
    ("timed", "olmoe-1b-7b", "d15", 3, 15, "dk"),
)
RATE_USD_H = 2.29
CONVL_MIN = sum(e for *_x, e in CONVL_UNITS) * 2
#: the deadline arithmetic (build review section 4): the worst nodrop path is the 138 nodrop minutes plus two more probe
#: rungs (32), 170 min of units at 1.0x the estimate; the driver keeps RESERVE_MIN 8 and setup takes about 25
PROBE_RUNG_MIN = 20
SETUP_MIN, RESERVE_MIN = 25, 8
DEADLINE = {"deadline_min_after_launch": 220, "hard_cap_min_after_launch": 235,
            "arithmetic": "the unit estimates are rental A's ACTUAL minutes (no further discount): units have 220 - 25 (boot, "
                          "driver upgrade and setup) - 8 (the driver's reserve) = 187 min. The worst nodrop path is 142 nodrop "
                          "min (the probe's first rung, 20, counter clause included) + two more full rungs (40) = 182 <= 187; "
                          "every path fits (see minutes). The guardian terminates at launch + 235 min whatever the ledger says "
                          "(15 min for the final push and verify)",
            "drop": "when the estimates overrun the driver drops dk, ms, w4, rep from the end (never a nodrop unit): at 1410 it "
                    "drops dk; at 1305 it drops dk, ms, w4 AND rep (sigma_page then falls back to rental 3's 0.18%, OD4 lost), "
                    "leaving 5 min of nodrop slack"}


def _pass_prob(cells: int, p: float, cap: float = 0.10, reps: int = 9) -> float:
    """P(a probe page passes P3's LOST clause and P4) when every repeat is non-clean iid with p: no cell has 4 or more
    non-clean repeats of 9, and the page's non-clean total is at most cap x rows (exact, by convolution)."""
    from math import comb
    per = [comb(reps, k) * p ** k * (1 - p) ** (reps - k) for k in range(4)]
    tot = [1.0]
    for _ in range(cells):
        nxt = [0.0] * (len(tot) + 3)
        for i, a in enumerate(tot):
            for k, b in enumerate(per):
                nxt[i + k] += a * b
        tot = nxt
    lim = int(cap * cells * reps + 1e-9)
    return sum(tot[: lim + 1])


def probe_power() -> dict:
    out = {"model": "the TIMED pages only: every repeat non-clean independently with probability p (iid; real dips cluster, "
                    "so this is a guide); a page passes when no cell is LOST (P3) and at most 10% of its rows are non-clean "
                    "(P4); P3's clock clause and P2 left out (on rental A's real pages they never decided). The counter clause "
                    "P5 is a further hard clause on a separate page, not modelled: a rung passes at most as often as below"}
    for name, cells in (("Mixtral page (16 cells, 144 rows)", 16), ("Qwen2-57B page (8 cells, 72 rows)", 8)):
        out[name] = {f"P(pass) at p {p:g}": r6(_pass_prob(cells, p)) for p in (0.04, 0.08, 0.12, 0.15)}
    out["rung"] = {f"P(both pass) at p {p:g}": r6(_pass_prob(16, p) * _pass_prob(8, p)) for p in (0.04, 0.08, 0.12, 0.15)}
    out["reading"] = ("under this model the two timed pages of a rung whose true non-clean rate is 4% pass together about 98 "
                      "times in 100, at 8% about three in five, at 12% about one in twelve and at 15% about one in two hundred; "
                      "P5 can only lower these: the strict rule trades false step-downs on a moderate sag for never passing a "
                      "gross one")
    return out


def plan_block() -> dict:
    p = REPO / A2.PLAN_A2
    total = sum(u[3] for u in PLAN_UNITS)
    nodrop = sum(u[3] for u in PLAN_UNITS if u[5] == "nodrop")
    order = []
    for u in reversed(PLAN_UNITS):
        if u[5] != "nodrop" and u[5] not in order:
            order.append(u[5])
    avail = DEADLINE["deadline_min_after_launch"] - SETUP_MIN - RESERVE_MIN
    drops = [(6, "dk"), (3, "ms"), (3, "w4"), (14, "rep")]
    wall, dropped = {}, {}
    for lk, extra in ((1710, -CONVL_MIN), (1410, PROBE_RUNG_MIN), (1305, 2 * PROBE_RUNG_MIN)):
        u, gone = total + extra, []
        for m, g in drops:
            if u <= avail:
                break
            u, gone = u - m, gone + [g]
        wall[lk], dropped[lk] = u + SETUP_MIN, gone
    return {"path": A2.PLAN_A2, "sha256": sha256_file(p) if p.exists() else "NOT WRITTEN",
            "units": [{"step": s, "model": m, "label": lab, "est": e, "cap": c, "drop": d} for s, m, lab, e, c, d in PLAN_UNITS],
            "units_count": len(PLAN_UNITS), "est_total_min": total, "est_nodrop_min": nodrop, "drop_order_last_first": order,
            "minutes": f"{total} min of units estimated ({nodrop} nodrop); the probe {PROBE_RUNG_MIN} min per rung (Mixtral about "
                       "10, Qwen2-57B 5, the OLMoE f128 counter page 4, hw_rulers 1), one rung in the estimate; a failing rung "
                       "stops at its first failing page, so it costs at most 20. Expected wall at the estimates (rental A's "
                       f"actual minutes) plus about {SETUP_MIN} min of boot, driver upgrade and setup: "
                       + "; ".join(f"lock {lk}: about {w} min, ${RATE_USD_H * w / 60:.1f}"
                                   + (f" (the driver drops {', '.join(dropped[lk])})" if dropped[lk] else "")
                                   for lk, w in wall.items())
                       + f". At 1710 the six CONV-L units ({CONVL_MIN} min) are skipped by rule and not budgeted",
            "deadline": DEADLINE,
            "aborts": "a gate abort ends the rental at about 3 (G-CAL), 6 (G-CEN), at most 26, 46 or 66 (G-PROBE after one, "
                      "two or three rungs) min of units, and G-BAND or G-64 about 13 and 25 min after the selected rung"}


def gates() -> dict:
    a_power = {"Current Power Limit (GPU, at the start)": 700.0, "Default Power Limit (GPU)": 900.0, "Max Power Limit (GPU)": 900.0,
               "Min Power Limit (GPU)": 100.0, "Max Power Limit (module, -sc 1)": 1000.0,
               "source": "rental A's TREE/vm/session/power-at-start.txt (nvidia-smi -q -d POWER; the board 03e9ec)"}
    return {
        **head("gates", "REGISTERED BEFORE ANY A2 PAGE: the session's mechanical rules and the plan",
               "the driver scripts/gh200_tileA2_session.sh (the pinned tile driver plus lines marked A2), the gate helper "
               "scripts/tileA2_gate.py, the plan " + A2.PLAN_A2),
        "power (rule P-W; the owner's 2026-10-20 decision)": {
            "rule": "the prelude reads nvidia-smi power.max_limit (GPU scope) BEFORE setting anything: under the plan's 900 W "
                    "the prelude REFUSES and the session stops before any unit (no fallback to 700 W); then -pl 900 -sc 0, and "
                    "the read-back power.limit must be 900 W (the pinned prelude's own check), else REFUSED",
            "rental A's board": a_power,
            "module": "the module's 1000 W (-sc 1, shared with Grace) is left at its default and recorded: nvidia-smi -q -d "
                      "POWER (both scopes) before and after every probe rung and sampled each second during it (DESCRIPTIVE)",
            "risk (REVIEW-A2 9)": "idle module minus GPU read about 70 W on rental A, so a 900 W GPU could meet the module cap; the "
                                  "probe is what sees it"},
        "G-CAL (after calibrate; DESIGN-A2 6 risk 1, OD9; build review fix 3)": "calibrate's settled dense bf16 GEMM clock (its "
                                                            "summary's gemm_clock_mhz, the median under load) under the ladder's "
                                                            "LOWEST rung, 1305 MHz: the rental ends (about 5 min in); a board the "
                                                            "lowest rung could hold always reaches the probe. "
                                                            "At 900 W the reading is no longer the 700 W figure of the 14 "
                                                            "recorded boards (1410 to 1470); calibrate is a weak proxy "
                                                            "(unlocked, L2 flushed: REVIEW-A2 1), the probe is the real check",
        "G-CEN (after the census)": "tileA2-census's equality gate and no MMA path mismatch, else the rental ends",
        "G-PROBE (the lock-hold probe; DESIGN-A2 6 risk 2, REVIEW-A2 1a, 1c, 6)": {
            "pages": files_block_a2(H.PROBE_PAGES),
            "rung": PR.__doc__.split("THE PROBE: ", 1)[1].split("A probe page's TIME", 1)[0].strip(),
            "ladder": f"{LT:g}, then {L1:g}, then {L2:g} MHz (owner, 2026-10-20), each rung probed by the prelude (lock, read "
                      "back, reset); the first rung that HOLDS writes a2-lock.env A2_LOCK and every later unit runs there; none "
                      "holds: the rental ends",
            "strict (owner, 2026-10-20)": "one LOST cell (under 6 clean repeats of 9) fails its page and so the rung",
            "counter clause P5 (owner, 2026-10-20: the re-review's MAJOR 1, option a)": "each rung also runs OLMoE's f128 counter "
                "page (label lp<L>f, est 4 min) and passes only if it gives at least 6 floor-bound cells in the rung's counter "
                "band (primary, else the registered fallback) after rental 6's counter clock rule: the band gate's rule on one "
                "page (joint_cF's own minimum; joint_cF needs two models). A rung whose counter clause fails steps down like a "
                "timed-probe failure. On rental A's board at 700 W this clause fails at 1710 (OLMoE f128's in-band cells were "
                "PRIVATE only: 0 floor-bound), where the timed OLMoE pages passed; the plan's own f128 units still run at the "
                "selected rung",
            "early stop (the re-review's MINOR 6, the cheaper option)": "pages run in the order Mixtral, Qwen2-57B, the OLMoE "
                "counter page; the rule is read after each and the rung's remaining pages are not run after the first failure",
            "pass rates (build review fix 6, MINOR 3)": probe_power(),
            "hw_rulers at L (REVIEW-A2 1c; DESIGN-A2 6 risk 7)": "before each rung's pages, DESCRIPTIVE: read2d / copy / add "
                                                                 "bandwidth and the matmul clock; score_probe_a2 prints the "
                                                                 "selected rung's against rung 1710's (DRAM at the lower L2 "
                                                                 "clock against v3's 3597.8 GB/s)",
            "ms unread": "no rule, no scorer and no person reads a probe page's time (they are E1-CAL cells at L)"},
        "G-BAND (after f128 Mixtral; REVIEW-A2 2b)": "tcommon.joint_cF over the f128 pages in the band in force is not None (the "
                                                    "primary band, else the registered fallback), else the rental ends",
        "G-64 (after f64w8 Mixtral; REVIEW-A2 2b)": "tcommon.joint_cF over the f64w8 pages in the band in force is not None, else "
                                                    "the rental ends (finalise refuses without it when no accounting is SELECTED)",
        "abort mechanics": "a gate is a plan unit (`- gate label=... gate=...`, nodrop); its exit is the gate helper's; ANY exit "
                           "but 0, or a gate or lockprobe unit SKIPPED for time, ends the session at once: the card reset, the "
                           "results pushed, the driver exit 3",
        "1710 at 900 W (REMOVED, folded into rung 1)": "the earlier build's DESCRIPTIVE 1710 probe and rulers at the plan's end are "
                                                        "removed: rung 1 of the ladder is the same two pages and hw_rulers at 1710 "
                                                        "at 900 W, read by the same rule, and it now decides the lock",
        "CONV-L's 1710 twin": "the three CONV-L pages timed at 1710 (lock-mhz=1710), nodrop (tileA2-convl); all six CONV-L units "
                              "carry unless-lock=1710 and are skipped by rule when rung 1710 holds",
        "score directory (build review fix 4)": f"{A2.SCORE_DIR_A2}/ (a2_score_directory)",
        "plan": plan_block(),
        "lock": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN, "a2_score_directory": SCORE_DIR,
    }


# --------------------------------------------------------------------------
# tileB2-e1tile and tileB2-stop (generated over the frozen tileB builders)
# --------------------------------------------------------------------------

#: the keys of tileB2-e1tile that may differ from the frozen tileB-e1tile (dotted paths; a new key is listed too)
WHITELIST_B2 = {
    "e1tile": ("registered", "name", "status", "design", "tool", "nominal_cells", "nominal", "tests.rivals.roofline",
               "tests.rivals.lever_cells_nominal", "gates.G4", "gates.census (tileA-census)", "power", "pages.plan.minutes",
               "pages.plan.driver (A2)", "power limit and lock (owner, 2026-10-20)", "at rung 1710", "lock", "seen (REVIEW-A2 5)", "at the fallback rung 1305",
               "a2", "seen_data"),
    "stop": ("registered", "name", "status", "design", "tool", "power (from tileB-e1tile, CPU float64)",
             "rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10).census", "rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10).finalise",
             "lock", "seen (REVIEW-A2 5)", "a2"),
}
B2_STATUS = ("FROZEN BEFORE ANY TILE RENTAL A2 PAGE; FINALISED MECHANICALLY FROM A2'S SELECTIONS (D3) at A2's lock: the cells, "
             "files, rules, rivals and gates are tileB-e1tile's; the numeric v4 predictions are written by "
             "scripts/scoring/tileA2/finalise_a2.py from A2's scored selections under at_lock(L), by the registered rules, before "
             f"rental B2 runs, into {A2.PREDICTIONS_B2} (a test holds finalise_a2.py to synthetic selections)")


def _b2_points(lock: float) -> tuple[dict, dict, list]:
    with at(lock):
        d = _fresh_e1tile()
    lev = d["tests"]["rivals"]["lever_cells_nominal"]
    ms = [[r["page"], r["label"], r["n"], r["arm"], r["ms_nominal"]] for r in d["nominal_cells"]]
    return d["power"]["points"], lev, ms


def e1tile() -> dict:
    with at(L1):
        base = _fresh_e1tile()
    pts2, lev2, ms2 = _b2_points(L2)
    out = dict(base)
    out.update(head("e1tile", B2_STATUS,
                    "scorer scripts/scoring/tileA2/score_tileB2_a2.py (scripts/scoring/tileA/score_tileB.py inside a2common.scoring); "
                    "tcommon.e1tile_rule, rival_rule, bar; predictions finalise_a2.py"))
    t = copy.deepcopy(base["tests"])
    t["rivals"]["roofline"] = "T0 + sum over GEMMs of max(flops / (4096 x eta_mma x 132 x L), bytes / bw), L A2's lock (1.71, 1.41 or 1.305 GHz)"
    out["tests"] = t
    g = dict(base["gates"])
    g["G4"] = "cell clock >= L - 5 MHz (L A2's lock: 1705, 1405 or 1300)"
    g["census (tileA-census)"] = ("a (model, tile) whose rental-B2 census occupancy differs from A2's merged census (tileA2-census): "
                                  "NOT SCORED")
    out["gates"] = g
    pg = copy.deepcopy(base["pages"])
    pg["plan"]["minutes"] = pg["plan"]["minutes"] + "; at A2's lock the timed units run fixed cell budgets, about unchanged"
    pg["plan"]["driver (A2)"] = ("scripts/gh200_tileA2_session.sh with `- prelude lock-mhz=<A2's lock>`; the plan is written by "
                                 f"finalise_a2.py ({A2.PLAN_B2})")
    out["pages"] = pg
    out["power limit and lock (owner, 2026-10-20)"] = ("rental B2 runs at the GPU power limit and the lock rung A2 actually held, "
                                                       "both read by finalise_a2.py from A2's score directory (session.score.json; "
                                                       "floor and e1cal must name the same lock, else it refuses); no argument "
                                                       "sets either; both go on B2's plan's prelude line")
    out["at rung 1710"] = "every L-dependent number is tileB-e1tile's own (v4L at 1710 is v4: r3_timing_model_v4L.py --check)"
    frozen = json.loads((REPO / "docs" / "registered" / "2026-10-14-tileB-e1tile-gh200.json").read_text())
    out["power"] = dict(out["power"], **{"points at 1710 (tileB-e1tile's, printed beside; REVIEW-A2 7b)": frozen["power"]["points"]})
    out["at the fallback rung 1305"] = {"power points": pts2, "lever_cells_nominal": lev2,
                                        "nominal_ms (page, label, n, arm, ms)": ms2}
    out["a2"] = {"stop rule input": "A2's CLEAN E1-CAL score (tileB2-stop)", "census": "A2's merged census (tileA2-census)",
                 "power (REVIEW-A2 7b)": "the power block is regenerated at L (its cells' dlnT/dlnB move with c at L) and printed "
                                         "beside tileB-e1tile's; whitelisted, never held equal",
                 "rental B at L (REVIEW-A2 9: an inference)": "rental B's 128 G 16 wgmma tiles on Phi, Qwen1.5 and JetMoE are as "
                                                              "power-heavy as rental A's, so rental B runs at A2's lock, not 1710"}
    out["lock"], out["seen (REVIEW-A2 5)"] = OWNER_LOCK, SEEN
    out["seen_data"] = list(base["seen_data"]) + ["rental A's pages and scores (tileA2 SEEN block)"]
    return out


def stop() -> dict:
    with at(L1):
        RG._E1TILE.clear()
        try:
            base = RG.stop()
        finally:
            RG._E1TILE.clear()
    out = dict(base)
    out.update(head("stop", "REGISTERED BEFORE ANY A2 PAGE (read on A2's scores)",
                    "scripts/scoring/tileA2/score_e1cal_a2.py (tcommon.stop_rule), on A2's CLEAN view"))
    k = "rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10)"
    r = dict(base[k])
    r["census"] = "A2's merged census refused (an MMA path mismatch or stack drift) or missing is a STOP"
    r["finalise"] = "scripts/scoring/tileA2/finalise_a2.py refuses (exit 2, nothing written) unless this rule reads GO"
    out[k] = r
    out["lock"], out["seen (REVIEW-A2 5)"] = OWNER_LOCK, SEEN
    out["a2"] = {"pages": "the five E1-CAL pages at A2's lock (tileA2-e1cal)", "CONV-L": "not a clause (OD6)",
                 "conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)": OWNER_LOCK[
                     "conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)"],
                 "at rung 1710": "no conversion: the clauses read v4 itself"}
    return out


# --------------------------------------------------------------------------
# the decode stage's tile-source addendum (DESIGN-A2 4.3, OD7; REVIEW-A2 7c, 7d)
# --------------------------------------------------------------------------

def tilesource() -> dict:
    return {
        "registered": REGISTERED, "name": A2.TILESOURCE,
        "status": "REGISTERED BEFORE ANY A2 OR DECODE PAGE: an addendum binding the decode stage's tile input; "
                  "2026-10-14-decodeC-cal-gh200 and 2026-10-14-decodeB-blind-gh200 stay byte-identical",
        "design": DESIGN, "tool": f"scripts/scoring/tileA2/decode_gate_a2.py {A2.SCORE_DIR_A2} (new), then "
                                  f"scripts/scoring/decode/score_cal.py <tree> <out> --tile-scores {A2.SCORE_DIR_A2} (unchanged)",
        "binds": "'the tile stage's selections' (decodeC-cal model.tile_stage, tileA/finalise.selections) means A2's score directory "
                 f"{A2.SCORE_DIR_A2}/ (tileA2's SCORE DIRECTORY), read by score_cal.tile_params unchanged",
        "the decode gate (owner, 2026-10-20)": "decode uses A2's constants ONLY IF A2's floor calibration is scored (floor.score.json "
            "calibration CAL-A: a joint c_128 and F fit) and finalise.selections reads the directory without refusing, WHATEVER "
            "ELSE A2's stop rule reads (its verdict printed beside): scripts/scoring/tileA2/decode_gate_a2.py exits 0 GO and "
            "prints score_cal's command. Otherwise it exits 3 WAIT: decode does not use A2's constants and the decode CAL rental "
            "waits (it is not launched). A new file: the frozen score_cal.py stays byte-identical",
        "lock": "the decode stage stays at 1710 (OD7): its card constants are rental 6's 1710 fit, so none needs a conversion; its "
                "cells (T 1 to 128, GRR graphs on cold weights) are DRAM-shaped and light (inference from rental A: OLMoE 128 G 1 "
                "n 1 held 1710 in 251 to 259 of about 260 samples)",
        "cycle invariance (an assumption, stated)": "the T 97..128 cells (the one wgmma decode tile, 64x128x64 w4) price A2's cycle "
                                                    "constants, measured at L, through v4's 1710 c_card: the cycle invariance of "
                                                    "tileA2-convl run backwards, tested only jointly by E1-DEC",
        "on an A2 STOP (REVIEW-A2 7d; owner, 2026-10-20)": "a STOP of A2's stop rule alone does not stop decode: the decode gate "
            "reads the floor calibration, not the stop rule; with no calibration decode waits",
        "census (REVIEW-A2 7d)": "decode's occupancy is A2's merged census (census.score.json, via selections -> tcommon.use_census): "
                                 "deepseek-v2-lite's decode occupancy moves from the shared-memory rule to A2's tread-3 entries "
                                 "when the OLMoE n 1 / n 3 rule admits them, else stays the shared-memory rule",
        "decode's own census unit": "the decode CAL plan's tilecensus unit (scripts/plans/rentalD-cal-2026-10.plan:10) runs the "
                                    "pinned tile_census.py and will log deepseek-v2-lite's six token-count errors again; no decode "
                                    "scorer reads that unit's file for occupancy, so it changes nothing",
        "cross-clock readout (REVIEW-A2 7c, DESCRIPTIVE)": "A2's c_128 at L against rental A's SEEN out-of-band 698.67 +- 10.63 "
                                                          "cycles at 1601 to 1640 MHz (difference, relative and z on the two se), "
                                                          "printed by score_floor_a2.py in floor.score.{json,txt}: the only "
                                                          "cross-clock wgmma reading the study has",
        "rival (not taken)": "re-register decode at L (every v4d card constant by P1's rule, a CONV-L-like anchor for v4d): more "
                             "cost, no gain the evidence asks for",
        "lock_block": OWNER_LOCK, "seen (REVIEW-A2 5)": SEEN,
    }


BUILDERS = {"census": census, "floor": floor, "bytes": bytes_, "e1cal": e1cal, "dead": dead, "rep": rep, "convl": convl,
            "gates": gates, "e1tile": e1tile, "stop": stop, "tilesource": tilesource}
#: the tileA renderers' text blocks reused where the part is a tileA part
TEXT = dict(RG.TEXT)
TEXT["convl"] = lambda d: [f"{p['page']} {p['label']} n{p['n']}: ratio at 1410 {p['ratio']['1410']}, at 1305 {p['ratio']['1305']}"
                           for p in d["predictions"]]
TEXT["gates"] = lambda d: [f"plan: {d['plan']['minutes']}"] + [f"  {u['step']} {u['model']} {u['label']} est {u['est']} cap "
                                                               f"{u['cap']} {u['drop']}" for u in d["plan"]["units"]]
TEXT["tilesource"] = lambda d: [d["binds"], d["lock"]]
