#!/usr/bin/env python
"""Writes the settle rental S1's registration, docs/registered/2026-10-21-settleS1-gh200.{json,txt}, before any page.

    python scripts/scoring/settleS1/register_s1.py [--out-dir docs/registered] [--check]

S1 is DESCRIPTIVE: it measures what DIAGNOSIS-A2 (section Q7, "What the pages cannot settle") could not, so v5's sag
(F1) and re-read (F3) terms are measured, not fitted; it certifies nothing and no model verdict is scored on its pages.
It is registered anyway, before any page: per measurement the unit, the quantity, the reader (script and function), the
registered reading bands and which candidate fix each reading constrains, and v4's byte predictions for S1-M3's cells.

Every number is computed here from committed files (v4's bytes by rental A's pinned regtile, as tileA-bytes was) or is
a typed-in input with its source. `--check` recomputes and compares with the committed JSON (the `code_pins` block
excluded, as every registration since rental 3) and text, exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REG = REPO / "docs" / "registered"
NAME = "2026-10-21-settleS1-gh200"
PLAN = "scripts/plans/rentalS1-settle-2026-10.plan"
HIST = "docs/registered/2026-10-21-settleS1-hist"
DIAG = "DIAGNOSIS-A2 (moe-kernels-work 2026-10-21 diag-A2/DIAGNOSIS.md, offline, read-only over A2's tree at 36291487)"
#: every file S1's driver, wrapper, reader and generator run, and what they stand on
PINNED = ("scripts/gh200_settleS1_session.sh", "scripts/dram_counter_route_s1.py", "scripts/power_proof_s1.py",
          "scripts/gh200_tileA2_session.sh",
          "scripts/dram_counter_route_tile.py", "scripts/r3_tile.py", "scripts/dram_counter_route.py",
          "scripts/private_weight_reference.py", "scripts/wave_split_bytes_v4.py", "scripts/r3_timing_model_v4.py",
          "scripts/scoring/tileA/regtile.py", "scripts/scoring/tileA/tcommon.py",
          "scripts/scoring/settleS1/hist_s1.py", "scripts/scoring/settleS1/read_s1.py",
          "scripts/scoring/settleS1/register_s1.py")
TILE128 = "128x128x64w8s3"
#: A2's CAL-A floor constants (A2's floor score, 'calibration': c and F per GEMM, counter cycles; SEEN)
CAL_A = {"c_cycles": 691.6475590359868, "F_cycles": {"w1": 2898.052707811491, "w2": 3346.3006086105324}}


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


def m3_predictions() -> list[dict]:
    """v4's q per GEMM (keys v3 and CLAMP) for S1-M3's NATIVE cells, by the calls tileA-bytes' byte_predictions makes
    (scripts/scoring/tileA/regtile.py): WV4.q_v3 at the tile and the occupancy tcommon reads, CLAMP's floor at q_ideal."""
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "tileA"))
    import tcommon as TC  # noqa: F401  (puts the repo and scripts/ on sys.path first)
    import regtile as RG
    import r3_timing_model_v4 as V4
    import wave_split_bytes_v4 as WV4
    from moe.spec import MODEL_CONFIGS
    pr = RG.pricer()
    doc = json.loads((REPO / HIST / "mixtral-8x7b-128g1-n.json").read_text())
    model = doc["model"]
    t = RG.PAGE_TILE[int(doc["block_m"])]
    G = int(doc["group_m"])
    occ, src = TC.occupancy_of(t, model)
    E = MODEL_CONFIGS[model].num_experts
    out = []
    for c in doc["cells"]:
        n = int(c["n"])
        counts = None if c["counts"] is None else tuple(int(v) for v in c["counts"])
        with V4.at_tile(model, t):
            q3 = WV4.q_v3(pr.eng, model, n, counts, RG.COPIES, RG.seed_of(doc, c), t, G, occ)
        qi = WV4.q_ideal(model, n, counts if counts is not None else tuple([n * t.bm] * E), t.bm)
        for arm in c["arms"]:
            out.append({"page": doc["page"], "model": model, "tile": t.key, "G": G, "n": n, "label": c["label"],
                        "arm": arm, "q_pred": {"v3": {g: round(q3[g], 6) for g in q3} if q3 else None,
                                               "CLAMP": {g: round(max(q3[g], qi), 6) for g in q3} if q3 else None},
                        "q_ideal": round(qi, 6), "occupancy": occ, "occupancy_source": src})
    return out, round(float(pr.f2.x[2]), 6)


def build() -> dict:
    sys.path.insert(0, str(HERE))
    import hist_s1 as HS
    import read_s1 as RS
    units = plan_units()
    preds, bw = m3_predictions()
    est = sum(u["est"] for u in units)
    nodrop = sum(u["est"] for u in units if u["drop"] == "nodrop")
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    boot = 25
    wall = est + boot
    deadline, cap = 85, 100
    d = {
        "registered": "2026-10-21, before any S1 page",
        "name": NAME,
        "status": ("DESCRIPTIVE: S1 measures; it certifies nothing and no model verdict is scored on its pages. The reading "
                   "bands below choose the FORM v5 registers for F1, F2 and F3; they are not verdicts"),
        "design": (f"{DIAG}, section Q7's five measurements; the owner's decision of 2026-10-21: run S1 first so the sag (F1) "
                   "and the re-reads (F3) are measured, not fitted, then design v5"),
        "tool": ("scripts/gh200_settleS1_session.sh (A2's pinned driver plus lines marked S1), "
                 "scripts/dram_counter_route_s1.py (instrumentation only: ncu replay mode, cache control, metric list), "
                 "scripts/power_proof_s1.py (S1-M0's load: torch.matmul, no kernel of ours), "
                 "scripts/scoring/settleS1/read_s1.py (the reader)"),
        "seen": ("Every S1 page is SEEN for v5: any constant v5 takes from an S1 reading (A and tau per lock, the knee's form, "
                 "a re-read refit) is SEEN-MEASURED or SEEN-FITTED, and no S1 page or E1-CAL page can test it. v5's blind test "
                 "must use NEW shapes: rental B2's frozen pages (docs/registered/2026-10-20-tileB2-e1tile-gh200: JetMoE-8B, "
                 "Phi-3.5-MoE and Qwen1.5-MoE-A2.7B, the shapes DIAGNOSIS-A2 Q6 names for each fix), never OLMoE-1B-7B, "
                 "Mixtral 8x7B or Qwen2-57B-A14B, whose cells S1 and A2 measured"),
        "power_and_lock": {
            "power_limit_w": 900,
            "rule": ("A2's (tileA2-gates P-W): the pre-setup dry run and the prelude refuse a card whose power.max_limit is "
                     "under 900 W, no fallback to 700 W; the limit is set as A2 set it, nvidia-smi -pl 900 -sc 0 (scope 0 is "
                     "the GPU's, 1 the module's, per nvidia-smi's documentation). S1 adds the read-back: the prelude polls "
                     "(every 2 s, up to 30 s) until --query-gpu=enforced.power.limit (the power management algorithm's "
                     "ceiling, the minimum of every limiter) and nvidia-smi -q -d POWER's GPU-scope Current Power Limit "
                     "both read 900 W within 1 W, then writes power-set.txt; otherwise it REFUSES. A2 read 700 W from -q "
                     "(Current and Requested) and 900 W from power.limit on every read of its session and never read "
                     "enforced.power.limit, so the limit A2 ran under is not known (A2 publication)"),
            "fallback": ("the owner's decision of 2026-10-21, pre-authorised: a read-back refusal at 900 W is recorded "
                         "(s1-power.env S1_POWER_900=refused, power-set-unsettled-900.txt) and the SAME session sets "
                         "nvidia-smi -pl 700 -sc 0 and repeats the read-back, requiring enforced.power.limit and the -q "
                         "GPU-scope Current Power Limit both at 700 W; every unit then runs at 700 W (S1_POWER_W=700). A "
                         "refusal at 700 W too stops the session. No other scope is tried (-sc 2 is not used)"),
            "outcomes": {"900 W read back": ("the owner's 900 W limit is in force as both readings say; M0 confirms it "
                                             "(ABOVE-700 possible) or shows a lower limiter (AT-700, CAP-BELOW-680)"),
                         "900 W refused, 700 W read back": ("the board behaves as A2's most likely did; S1's pages are at "
                                                            "700 W, comparable to A2's; the 900 W decision is recorded as "
                                                            "not attainable on this board; M0 states whether 700 W held "
                                                            "(AT-700) or another limiter is lower (CAP-BELOW-680); ABOVE-700 "
                                                            "would mean the readings do not name the enforced limit"),
                         "both refused": "the session stops before any page (about $1 of setup); no S1 reading exists"},
            "session_lock_mhz": 1410,
            "lock_rule": ("every unit but M0 runs under nvidia-smi -lgc at the session lock 1410 MHz (the rung A2 held) with ncu "
                          "--clock-control none, except S1-M2's pages at their own locks 1305 and 1005 MHz (and its "
                          "same-board 1410 reference), and M0 (the power proof), which runs with no lock; the prelude locks, reads back and resets each unit lock once"),
            "probe": ("no lock-hold probe: S1 certifies nothing and chooses no lock, and every page carries its own clock "
                      "reading (sm__cycles_elapsed.avg over gpu__time_duration.sum per GEMM), so a lock that sags is the "
                      "measurement itself, not a reason to move the session"),
        },
        "measurements": {
            "M0": {
                "question": ("which GPU power limit holds: A2's -q readings (700 W) or its power.limit (900 W)? The limit "
                             "every later S1 page runs under, and the likeliest source of the sag (DIAGNOSIS-A2 Q5: the SW "
                             "power cap, 0x4, fired on A2's heavy pages at 1410)"),
                "units": ["- pp"],
                "cells": "no cells: torch.matmul 8192^3 bf16, back to back, 90 s, no clock lock (scripts/power_proof_s1.py)",
                "instrument": ("nvidia-smi --query-gpu=timestamp,power.draw,enforced.power.limit,power.limit,clocks.sm,"
                               "clocks.max.sm,clocks_event_reasons.active,temperature.gpu -lms 100 for the GEMM's length; "
                               "-q -d POWER before and after"),
                "unit": "watts (GPU scope power.draw); seconds; share of samples",
                "quantity": ("busy samples (power.draw > 200 W); 1 s means (10 consecutive busy samples); the seconds of 1 s "
                             "means over 720 W; the largest 1 s mean; the share of busy samples with the SW power cap bit "
                             "0x4; the busy clock's median against clocks.max.sm; enforced.power.limit as read"),
                "read_by": "scripts/scoring/settleS1/read_s1.py power_proof",
                "reading": {"ABOVE-700": "1 s means over 720 W cover >= 10 s",
                            "AT-700": "largest 1 s mean in [680, 720] W, SW power cap on >= half the busy samples, busy clock under max",
                            "CAP-BELOW-680": "SW power cap on >= half the busy samples and the largest 1 s mean under 680 W",
                            "UNDECIDED": "otherwise (the GEMM reached no limit)"},
                "constrains": {
                    "session": ("ABOVE-700: a ceiling above 700 W held under S1's 900 W readings; S1's pages ran above A2's "
                                "-q value. AT-700: a 700 W ceiling held although both read-backs said 900 W: the read-backs "
                                "do not name the enforced limit on this platform, S1's pages ran at 700 W (as A2's most "
                                "likely did) and the owner's 900 W decision was not in force. CAP-BELOW-680: another limiter "
                                "(module or platform) holds below both; S1-M2's sag is read against it. UNDECIDED: no reading"),
                    "F1": ("AT-700 or CAP-BELOW-680 with S1-M2 PRESENT: the sag is a power-cap transient at the enforced "
                           "limit, so F1's A and tau are per limit as well as per lock and tile; ABOVE-700 with S1-M2 "
                           "PRESENT: the sag persists with headroom, a kernel-start clock effect at the lock, not the cap")},
            },
            "M1": {
                "question": ("DIAGNOSIS-A2 Q7 row 1: is the sag in the timed burst the one ncu's kernel replay shows (each "
                             "kernel replayed in isolation after a cache flush and a save/restore)?"),
                "units": ["olmoe-1b-7b kr", "olmoe-1b-7b ar", "qwen2-57b-a14b kr", "qwen2-57b-a14b ar"],
                "cells": ("E1-CAL cells OLMoE 128x128 G 1 n 1 PT and n 4 PT, Qwen2-57B n 2 PT, NATIVE (copied cells, same "
                          f"shuffle seed: {HIST}/olmoe-1b-7b-128g1-rp.json, qwen2-57b-a14b-128g1-rp.json)"),
                "instrument": {"kr": ("A2's byte-page instrument: ncu --replay-mode kernel --cache-control all --clock-control "
                                      "none under nvidia-smi -lgc 1410 (scripts/dram_counter_route_tile.py)"),
                               "ar": ("ncu --replay-mode application --cache-control none, everything else the same "
                                      "(scripts/dram_counter_route_s1.py --s1-replay application)")},
                "unit": "microseconds per GEMM kernel launch (mean of the page's 3 measured calls); per cell, the call's w1 + w2",
                "quantity": ("off = gpu__time_duration.sum - sm__cycles_elapsed.avg / 1410 MHz per GEMM; d = off_app - off_kernel "
                             "per GEMM and per cell"),
                "read_by": "scripts/scoring/settleS1/read_s1.py replay_contrast (rows: gemm_rows)",
                "reading": {"INSTRUMENT-INDEPENDENT": "|median d_cell| <= 10 us and every |d_cell| <= 0.25 x off_kernel_cell + 5 us",
                            "INSTRUMENT-INFLATED": "median d_cell < -10 us", "INSTRUMENT-DEFLATED": "median d_cell > +10 us",
                            "MIXED": "otherwise"},
                "constrains": {
                    "F1": ("INSTRUMENT-INDEPENDENT: the sag is not made by kernel replay's flush or save/restore; v5's F1 takes "
                           "A and tau from counter pages (A2's, S1-M2's) as measured. INSTRUMENT-INFLATED: the kernel-replay A "
                           "overstates the sag; v5's F1 takes its size from the application-replay offsets (per cell) and "
                           "DIAGNOSIS-A2 Q1's clock/call split is re-read with them. INSTRUMENT-DEFLATED: the sag is larger "
                           "in stream order; F1's A is taken from the application-replay pages, and the per-call timed "
                           "residual (-32.3 us at 128) is compared with their per-cell sum. MIXED: no counter-page A enters "
                           "v5 without a per-cell check against these three cells")},
                "follow_up": ("in-burst per-GEMM CUDA events (a wrapper around pinned R3's timed path) are built only if "
                              "M1 reads MIXED or INSTRUMENT-DEFLATED (owner, 2026-10-21)"),
                "not_reproduced": ("application replay still serialises the profiled launches (ncu stops between them to read "
                                   "its counters), so the GEMMs are not as back to back as in R3's timed burst; the contrast "
                                   "isolates the flush and the save/restore, not the gaps"),
            },
            "M2": {
                "question": "DIAGNOSIS-A2 Q7 row 2: does the sag vanish at a lower lock (an apparatus fix), and where?",
                "units": ["olmoe-1b-7b f1305", "olmoe-1b-7b f1005", "olmoe-1b-7b f1410 (the same-board reference, nodrop)"],
                "cells": "the OLMoE f128 counter page: 128x128x64 w8 s3, G 64, treads 1 to 9, NATIVE, SHARED and PRIVATE (54 GEMM cells)",
                "instrument": "A2's byte-page instrument at the unit's own lock (1305, 1005, 1410 MHz)",
                "unit": "microseconds per GEMM kernel; A in microseconds, tau in microseconds; f in MHz",
                "quantity": ("per lock L: off = t_ncu - cycles / L per GEMM cell; A(L), tau(L) of off = A (1 - exp(-t / tau)) "
                             "(tau on a grid 50 to 4975 us, step 25; A least squares; DIAGNOSIS-A2 work/transient.py), the "
                             "linear t = a + cycles / f, the median off by duration bin"),
                "read_by": "scripts/scoring/settleS1/read_s1.py lock_ladder (sag_fit over gemm_rows)",
                "reading": {"GONE": "A(L) <= 8 us (R3's own 1410 sag is 6.1 us and sits in T0)",
                            "REDUCED": ("A(L) <= 0.5 x A(1410) on this board's f1410 page (nodrop); if f1410 is still missing, "
                                        f"A2's fit of the same page, {RS.A2_REF_A_US} us (tau {RS.A2_REF_TAU_US:g} us; "
                                        f"{RS.A2_REF_PAGE}, another board)"),
                            "PRESENT": "otherwise"},
                "constrains": {
                    "F1": ("GONE at 1305: an apparatus fix exists at the ladder's own rung (time the wgmma tiles at 1305, F1 "
                           "keeps only R3's baseline in T0); GONE at 1005 only: the fix needs 1005 (an owner decision on the "
                           "timing lock); PRESENT at both: no apparatus escape, F1 is priced, and A(L), tau(L) are its per-lock "
                           "constants (SEEN-MEASURED); REDUCED: F1 is priced with A(L) at the lock v5 times at")},
                "decided_after": ("whether v5 and B2 time at 1005 MHz (should M2 read GONE only there) is decided by the owner "
                                  "after S1, with M2's numbers (owner, 2026-10-21); nothing here pre-commits it"),
            },
            "M3": {
                "question": "DIAGNOSIS-A2 Q7 row 3: does v4 over-predict Mixtral 128 G 1's w1 bytes (Q2's inference, about 7%)?",
                "units": ["mixtral-8x7b b128g1"],
                "cells": f"Mixtral 8x7B 128x128 G 1, n 2 PT, PW, uniform, balanced, NATIVE ({HIST}/mixtral-8x7b-128g1-n.json)",
                "instrument": "A2's byte-page instrument at 1410",
                "unit": "weight sets per GEMM (q); e as a fraction",
                "quantity": ("q_counted = (dram__bytes_read.sum - n x the GEMM's A operand per tread) / the GEMM's weight set; "
                             "e = q_pred / q_counted - 1 per key (v3, CLAMP) with q_pred registered below "
                             "(byte_predictions_m3)"),
                "read_by": "scripts/scoring/settleS1/read_s1.py m3_bytes",
                "reading": {"W1-OVER": "mean e(CLAMP, w1) over the 4 cells >= +4%", "W1-EXACT": "|mean e(CLAMP, w1)| < 2%",
                            "W1-PARTIAL": "otherwise"},
                "constrains": {
                    "F3": ("W1-OVER: Q2's inference holds; F3's w1 re-read refit takes these rows (SEEN-FITTED) and is what "
                           "moves Mixtral 128 G 1. W1-EXACT: bytes do not carry Mixtral 128 G 1's residual under F1 + F2 + F4 "
                           "(+2.8 to +3.5%); F3 is not needed there and v5's knee and sag forms are re-examined on Mixtral. "
                           "W1-PARTIAL: both")},
            },
            "M4": {
                "question": "DIAGNOSIS-A2 Q7 row 4: in the w2 (and w1) re-read excess at G 16 and 64 G 1, activation against weight re-reads",
                "units": ["mixtral-8x7b p128g16", "mixtral-8x7b p64g1"],
                "cells": ("ladder (balanced) cells, arms NATIVE, SHARED and PRIVATE: Mixtral 128x128 G 16 treads 3, 4, 8; "
                          "Mixtral 64x128 w8 G 1 tread 2 (the balanced cells of A2's b128g16 and b64g1 pages; R3 runs PRIVATE "
                          "on its balanced ladder only, PRIVATE_SKEW_REFUSAL)"),
                "instrument": "A2's byte-page instrument at 1410, the ladder's treads by the S1 key byte-treads",
                "unit": "weight sets per GEMM",
                "quantity": ("e_P = max(q_P - n, 0), PRIVATE's excess over its n weight sets (each M-tile its own copy: no "
                             "weight slab is read twice, so the excess is re-read activations), at the K-call means and at "
                             "PRIVATE's highest call (dram_counter_route.r3_weight_bracket, pinned); SHARED's re-read q_S - 1; "
                             "a = min(e_P / (q_S - 1), 1), the largest share of SHARED's excess that can be activations"),
                "read_by": "scripts/scoring/settleS1/read_s1.py private_split",
                "reading": {"B-DOMINATED": "at every tread e_P <= 0.2 x (q_S - 1) or e_P <= 0.02 weight sets",
                            "A-DOMINATED": "e_P >= 0.5 x (q_S - 1) at most treads", "MIXED": "otherwise"},
                "constrains": {
                    "F3": ("B-DOMINATED (w2): the w2 excess is weight re-reads; F3 refits M4's slab constants (C_B, beta_B, "
                           "theta1, eps1_w2) per tile family and nothing else. A-DOMINATED: M4's charge-once A assumption "
                           "fails at the wgmma tiles; F3 needs an activation re-read term, and the weight part of w2's "
                           "over-prediction is larger than the byte pages show. MIXED: both terms; the reduced form (eps1, "
                           "theta1 only) is not enough")},
            },
            "M5": {
                "question": "DIAGNOSIS-A2 Q7 row 5: the knee's mechanism (latency exposure at 2 CTAs per SM?)",
                "units": ["qwen2-57b-a14b st", "olmoe-1b-7b st"],
                "cells": (f"128x128 G 1 byte cells, NATIVE: Qwen2-57B n 1, 2 and OLMoE n 1, 2, 4, four labels each "
                          f"({HIST}/qwen2-57b-a14b-128g1-n.json, olmoe-1b-7b-128g1-n.json); x from about 0.3 to 1.15"),
                "instrument": ("A2's byte-page instrument at 1410 plus three recorded metrics (scripts/dram_counter_route_s1.py "
                               "--s1-stall-metrics): " + RS.STALL_LS + ", " + RS.STALL_BAR + ", " + RS.TENSOR),
                "unit": "percent (per active warp for the stalls, of peak sustained active for the tensor pipe); x and y ratios",
                "quantity": ("per GEMM: the three metrics; floor_cyc = q (S c + F_g), q = ceil(live CTAs / 132), S = K / 64 (the "
                             "tcommon.joint_cF form at the CAL-A constants below); bytes_cyc = counted reads / bw x lock; "
                             "x_s1 = bytes_cyc / floor_cyc; y_s1 = cycles / max(floor_cyc, bytes_cyc); Spearman rho of x_s1 "
                             "with each metric over the page's GEMM cells"),
                "read_by": "scripts/scoring/settleS1/read_s1.py stall_table",
                "reading": {"LATENCY": "rho(x, long scoreboard) >= 0.6 and rho(x, tensor active) <= -0.6",
                            "BARRIER": "rho(x, barrier) >= 0.6 and rho(x, long scoreboard) < 0.6", "NONE-SEEN": "otherwise"},
                "constrains": {
                    "F2": ("LATENCY: the soft knee is DRAM latency exposed at 2 CTAs per SM and 3 stages (Q5.2's INFERENCE "
                           "supported); v5's knee exponent is a per-tile property tied to occupancy and stages, and 64x128 "
                           "w8 (3 CTAs per SM) is expected between 128x128's 5.25 and R3's 14. BARRIER: the knee is pipeline "
                           "synchronisation; its exponent is tied to num_stages. NONE-SEEN: F2 stays a SEEN-FITTED shape "
                           "with no mechanism claimed")},
            },
        },
        "constants": {
            "M1": {"abs_us": RS.M1_ABS_US, "rel": RS.M1_REL, "slack_us": RS.M1_SLACK_US},
            "M2": {"gone_us": RS.M2_GONE_US, "reduced_frac": RS.M2_REDUCED, "ref_lock_mhz": 1410.0, "a2_ref_A_us": RS.A2_REF_A_US, "a2_ref_tau_us": RS.A2_REF_TAU_US,
                   "a2_ref_page": RS.A2_REF_PAGE,
                   "tau_grid_us": [RS.TAU_GRID_US[0], RS.TAU_GRID_US[-1], 25]},
            "M3": {"over": RS.M3_OVER, "exact": RS.M3_EXACT},
            "M4": {"a_dom": RS.M4_A_DOM, "b_dom": RS.M4_B_DOM, "tiny": RS.M4_TINY},
            "M5": {"rho": RS.M5_RHO, **CAL_A, "bw_gbps": bw, "sms": RS.SMS,
                   "source": ("c and F: A2's CAL-A floor calibration (A2's floor score, 'calibration'; SEEN); bw: v4's bandwidth, "
                              "the CAL engine's v2 timing fit x[2] (rental 6's Pricer, regtile.pricer), as DIAGNOSIS-A2 knee.py's "
                              "bytes_only term")},
        },
        "byte_predictions_m3": preds,
        "plan": {"file": PLAN, "sha256": sha256(REPO / PLAN), "units_count": len(units), "units": units,
                 "est_total_min": est, "est_nodrop_min": nodrop, "drop_order_last_first": order,
                 "boot_setup_min": boot, "wall_min": wall, "rate_usd_per_h": 2.29, "cost_usd": round(wall / 60 * 2.29, 2),
                 "deadline_after_launch_min": deadline, "hard_cap_after_launch_min": cap,
                 "cost_at_hard_cap_usd": round(cap / 60 * 2.29, 2)},
        "pages": {"files": {f"{n}.json": {"path": f"{HIST}/{n}.json", "sha256": sha256(REPO / HIST / f"{n}.json")}
                            for n in HS.PAGES}},
    }
    d["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    return d


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: the settle rental S1, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", "", f"SEEN: {d['seen']}", "",
         "POWER AND LOCK"]
    for k, v in d["power_and_lock"].items():
        L.append(f"  {k}: {v}")
    for m, x in d["measurements"].items():
        L += ["", f"{m}. {x['question']}"]
        for k in ("units", "cells", "instrument", "unit", "quantity", "read_by", "reading", "constrains", "follow_up",
                  "decided_after", "not_reproduced"):
            if k in x:
                v = x[k]
                L.append(f"  {k}: " + (v if isinstance(v, str) else json.dumps(v)))
    L += ["", "CONSTANTS"] + [f"  {k}: {json.dumps(v)}" for k, v in d["constants"].items()]
    L += ["", "V4'S BYTE PREDICTIONS FOR S1-M3 (q per GEMM, weight sets)"]
    for r in d["byte_predictions_m3"]:
        L.append(f"  {r['page']} n{r['n']}/{r['label']}/{r['arm']}: v3 {r['q_pred']['v3']} CLAMP {r['q_pred']['CLAMP']}")
    L += ["", f"PLAN {p['file']} sha256 {p['sha256']}: {p['units_count']} units, {p['est_total_min']} min estimated "
              f"({p['est_nodrop_min']} nodrop), drop order last first {p['drop_order_last_first']}"]
    for u in p["units"]:
        L.append(f"  {u['model']} {u['step']} {u['label']} est {u['est']} cap {u['cap']} {u['drop']}")
    L += [f"  wall about {p['wall_min']} min with {p['boot_setup_min']} min of boot and setup: ${p['cost_usd']:.2f} at "
          f"${p['rate_usd_per_h']}/h; deadline launch + {p['deadline_after_launch_min']} min; guardian hard cap launch + "
          f"{p['hard_cap_after_launch_min']} min (${p['cost_at_hard_cap_usd']:.2f} at most)", "", "PAGES"]
    for k, v in d["pages"]["files"].items():
        L.append(f"  {v['path']} {v['sha256']}")
    L += ["", "code_pins: (in the JSON)"]
    return L


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REG)
    ap.add_argument("--check", action="store_true")
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
