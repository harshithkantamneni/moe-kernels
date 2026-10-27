#!/usr/bin/env python
"""Writes the calibration rental C3's registration, docs/registered/2026-10-26-cal3-gh200.{json,txt}, before any page.

    python scripts/scoring/cal3/register_c3.py --write [--out-dir docs/registered]
    python scripts/scoring/cal3/register_c3.py --check      # recompute, compare (exit 1 on any difference)

C3 is a CALIBRATION rental and DESCRIPTIVE: it certifies nothing, moves no registered verdict, and every C3 page is SEEN
once read. It is registered anyway, before any page: what each page is for; exactly which quantities are read and how
(scripts/scoring/cal3/read_c3.py, by function); the bands and what each means for the next model round; the gate the
instrumented copy must pass before any stamp page; the priced times the w1 reading compares against, by value
(docs/registered/2026-10-26-cal3-priced.json, its sha256 here); and the code by sha256. `--check` recomputes everything
and compares with the committed JSON (the `code_pins` block excluded, as every registration since rental 3) and text.
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
import read_c3 as R3  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = R3.REG_NAME
WORK = "moe-kernels-work 2026-10-26 build-C3 (CHECK.md, DESIGN.md; offline, read-only over the repo at e324867c)"
#: every repo file C3's driver, probe, gate and reader run on the VM or the laptop
PINNED = (
    "scripts/gh200_cal3_session.sh", "scripts/gh200_cal2_session.sh", "scripts/gh200_settleS1_session.sh",
    "scripts/gh200_tileA2_session.sh",
    "scripts/instr_probe_c3.py", "scripts/instr_probe.py", "scripts/launch_floor.py",
    "moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py",
    "moe/instrumented/UPSTREAM.json", "moe/instrumented/upstream_v0.27.1_fused_moe.excerpt",
    "moe/bench/timing.py", "moe/bench/exit_codes.py",
    "scripts/private_weight_reference.py", "scripts/block_m_crossing_sweep.py", "scripts/locked_r3.py",
    "scripts/dram_counter_route_tile.py", "scripts/dram_counter_route.py", "scripts/r3_tile.py",
    "scripts/tileA2_gate.py", "scripts/scoring/tileA2/a2common.py", "scripts/scoring/tileA2/score_floor_a2.py",
    "scripts/scoring/tileA/tcommon.py",
    "scripts/scoring/cal3/read_c3.py", "scripts/scoring/cal3/register_c3.py", "scripts/scoring/cal2/read_c2.py",
)
RATE = 2.29
BOOT = 25
DEADLINE, HARD_CAP = 70, 85


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_units() -> list[dict]:
    """The plan's units as the driver parses them (model, step, label, est, cap, nodrop or drop group, keys)."""
    out = []
    for line in (REPO / R3.PLAN).read_text().splitlines():
        body = line.split("#", 1)[0].split()
        if not body:
            continue
        kv = dict(w.split("=", 1) for w in body[2:])
        drop = "nodrop" if (body[1] == "prelude" or kv.get("nodrop") == "1") else kv.get("drop-group")
        keys = {k: v for k, v in kv.items() if k not in ("label", "est", "cap", "nodrop", "drop-group")}
        out.append({"model": body[0], "step": body[1], "label": kv.get("label", ""), "est": int(kv["est"]),
                    "cap": int(kv["cap"]), "drop": drop, "keys": keys})
    return out


def page_entry(model: str) -> dict:
    rel = R3.HISTS[model]
    doc = json.loads((REPO / rel).read_text())
    label, vid = R3.UNITS[model]
    return {"path": rel, "sha256": sha256(REPO / rel), "shuffle_seed": doc["shuffle_seed"], "group_m": doc["group_m"],
            "block_m": doc["block_m"], "cells": [[c["label"], c["n"]] for c in doc["cells"]], "arms_stamped": ["native"],
            "unit": f"{model} stamps {label}", "variant": vid,
            "a2_e1cal": "the file rental A2's E1-CAL timed page ran (docs/registered/2026-10-20-tileA2-e1cal-gh200 pages)"}


def build() -> dict:
    units = plan_units()
    est = sum(u["est"] for u in units)
    nodrop = sum(u["est"] for u in units if u["drop"] == "nodrop")
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    wall = est + BOOT
    priced = json.loads((REPO / R3.PRICED).read_text())
    a2 = json.loads((REPO / "docs/registered/2026-10-20-tileA2-e1cal-gh200.json").read_text())["pages"]["files"]
    for m, rel in R3.HISTS.items():
        assert a2[Path(rel).name]["path"] == rel and a2[Path(rel).name]["sha256"] == sha256(REPO / rel), m
    d = {
        "registered": "2026-10-26, before any C3 page",
        "name": NAME,
        "status": ("CALIBRATION, DESCRIPTIVE: C3 certifies nothing, scores no verdict and moves no registered verdict; "
                   "every C3 page becomes SEEN once read. The bands say what each reading means for the next model round; "
                   "the round is the owner's"),
        "design": (f"{WORK}; the owner's option (b) of 2026-10-22 (moe-kernels-work 2026-10-22 refit4/ROUND2.md section 5): "
                   "one short instrumented rental measuring per-CTA launch and finish times for Mixtral-8x7B and "
                   "Qwen2-57B, w1 and w2, at vLLM's 128x128x64 w8 s3 G 1 tile, after refit4's two offline rounds failed "
                   "their gate (REFIT.md, ROUND2.md: Mixtral-8x7B's 128 G 1 w1 runs about 4% faster than priced under "
                   "every knee form, and C5's theta_L is SEEN-FITTED at 0.25 to 0.30)"),
        "tool": ("scripts/gh200_cal3_session.sh (C2's pinned driver plus lines marked C3), scripts/instr_probe_c3.py (the "
                 "probe), moe/instrumented/fused_moe_instr.py (the copy, unchanged, STAMPS 3), scripts/tileA2_gate.py band "
                 "(G-BAND), scripts/scoring/cal3/read_c3.py (the reader)"),
        "seen": ("Every C3 page is SEEN once read: Mixtral-8x7B and Qwen2-57B are CAL models already; no blind model of "
                 "B3 (Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, JetMoE-8B) and no new model is run. A constant read off a C3 page "
                 "and carried into a model is MEASURED (theta_L) or SEEN-FITTED (anything fitted to these pages). OD13 "
                 "kept: nothing is priced against rental A's 1710 E1-CAL pages"),
        "check": {"verdict": "WORKS AS IS (the copy at the ends level; no instrumentation change)",
                  "file": "moe-kernels-work 2026-10-26 build-C3/CHECK.md",
                  "facts": ("A2's tile census compiles 128x128x64 w8 s3 at 116 registers, 98,304 B shared, 2 CTAs per SM "
                            "(shared binds at 2; the copy can only lose occupancy above 128 registers), mma wgmma; no "
                            "stamp line reads a tile parameter; rentals 4 and 5 never produced a stamp page (20 gate "
                            "FAILs, median 1.45 to 9.51%, the cost a compile effect: 1-in-17 sampling cost as much as "
                            "every CTA); the ends level was compiled, never timed"),
                  "harness": ("instr_probe.py pins R3's tile and ladder cells and cubin.sass_checks finds the loop by "
                              "HMMA: C3's probe and SASS check are new host-side files; both pinned files unchanged")},
        "power_and_lock": {
            "power_limit_w": 700,
            "rule": ("C2's prelude, unchanged: nvidia-smi -pl 700 -sc 0, then the read-back (enforced.power.limit and -q -d "
                     "POWER's GPU-scope Current Power Limit both at 700 W within 1 W, polled every 2 s up to 30 s), else "
                     "the session STOPS (c2-power.env C2_POWER_700=refused, power-set-unsettled-700.txt); no fallback"),
            "session_lock_mhz": 1410,
            "lock_rule": ("the prelude locks 1410, reads clocks.sm back (two tries) and STOPS when it does not hold "
                          "(LOCK_PROBE_1410=held in locks.env); every timed instrumented unit runs under -lgc 1410 "
                          "(locked_run); regcheck is compile only"),
            "g_band": ("A2's G-BAND (tileA2_gate.py band) after C3's two f128 pages: joint_cF over them in the counter "
                       "band at 1410 (primary [0.975 L, L], else the fallback [0.959 L, L]) is not None, else the session "
                       "ends there")},
        "instrument": {
            "copy": "moe/instrumented/fused_moe_instr.py (vLLM v0.27.1 fused_moe_kernel plus rental 4's approved stamps)",
            "copy_sha256": sha256(REPO / "moe/instrumented/fused_moe_instr.py"),
            "spec": "stamps=ends,evict_a=none,evict_b=none (STAMPS 3: CTA start before the dead check, a dead CTA's exit, "
                    "the end after the final store; nothing near the wgmma k-loop)",
            "row": list(R3.COLUMNS),
            "tile": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3, "GROUP_SIZE_M": 1},
            "declared_copies": 9, "arms": ["native"], "warmup_calls": 3, "stamped_calls": 3,
            "flush": ("before every stamped call: the stamp buffers allocated, the GPU held ahead by instr_probe's "
                      "25 ms sleep (review MAJOR-2, as the burst timer does, so no launch waits on the host), then an L2 "
                      "flush outside the launch event pair"),
            "span": ("each launch's stamp_span_ns (first live start to last live end, globaltimer) is recorded beside "
                     "its event-pair ms; the reader prints event / span per cell"),
            "kept_per_launch": ("the int64 rows, expert_ids and num_tokens_post_padded vLLM handed the launch (references "
                                "held, copied to the host after the call's synchronize), USE_TD (must be False), the "
                                "launch's event-pair ms")},
        "gate": {
            "rule": ("owner decision 2 (2026-10-06) with rental 5's occupancy leg, on EVERY C3 cell: per (cell, GEMM) r = "
                     "median plain ms / median copy ms over 8 alternating bursts of 10 calls each way; PASS iff median "
                     "|r - 1| <= 1% and worst <= 2%, CTAs per SM identical per config, the all-off copy's SASS equal to "
                     "the plain kernel's (line info stripped), the installed fused_moe.py the upstream file, and C3's SASS "
                     "legs on every stamped config: a clock read before the first LDG and none inside the k-loop (the "
                     "back-branch spanning the most HGMMA; instr_probe_c3.sass_checks_c3)"),
            "regcheck": ("compile only, before any timed unit: the all-off copy and the stamped copy at the plain kernel's "
                         "CTAs per SM, and the plain kernel and the copy on the wgmma path (instr_probe_c3.regcheck_c3)"),
            "on_fail": ("owner OD-1 (2026-10-26; changes decision 2's refuse-on-fail for C3 only): a gate that fails on its "
                        "TIMING legs alone (median |r - 1| over 1% or worst over 2%) keeps the stamps units; their pages "
                        "are labelled PERTURBED (stamps.json page_label) and every reading carries the measured overhead "
                        "(perturb.json per (cell, GEMM): overhead_median = median copy / median plain - 1 and "
                        "overhead_worst = the worst burst pair). The HARD legs still refuse the unit (gate.env HARD_<id>): "
                        "CTAs per SM identical, the all-off copy's SASS equal to the plain kernel's, the upstream file, "
                        "C3's SASS legs (a clock read before the first LDG, none inside the HGMMA loop); and regcheck "
                        "(CTAs per SM, the wgmma path) must PASS"),
            "perturbed_readings": ("C3-R1 (theta_L) and C3-R2 (the re-read launch gap) are ORDERING quantities: read on "
                                   "PERTURBED pages with the overhead disclosed, unless the read GEMM's overhead passes "
                                   "the NOT-READ bound. C3-R3 (the w1 LIFE / PACKING split) is DESCRIPTIVE ONLY on "
                                   "PERTURBED pages: printed, never used to price or select a model form"),
            "not_read_bound": ("C3-R1 and C3-R2 are NOT READ when, over the read GEMM's cells (Mixtral-8x7B w2), the "
                               "median of the cells' |median overhead| exceeds 10% or any cell's |worst burst pair| "
                               "exceeds 20% (read_c3.overhead_ok; two-sided, review MINOR-1: a copy faster than the "
                               "plain kernel is a compile effect too)"),
            "perturbed_theta": ("review MINOR-2 and MINOR-3: on PERTURBED pages theta and theta (1 + o) are both printed "
                                "(o the w2 median |overhead|) and the band is reported only when both fall in one band, "
                                "else '<band> or <band> (PERTURBED)'; a PERTURBED theta_L is never registered as "
                                "MEASURED: at most MEASURED-PERTURBED, and a later round may use it only as such. Any "
                                "theta_L C3 reads on Mixtral-8x7B is IN-SAMPLE for Mixtral-8x7B in any later held-out "
                                "round"),
            "not_read_bound_why": ("theta_L and g are ratios of a launch spread to a lifetime. A slowdown that scales the "
                                   "whole CTA (a slower main loop, the compile effect rentals 4 and 5 saw) scales the "
                                   "spread and the lifetime alike and leaves them unchanged; a fixed per-CTA cost c "
                                   "lengthens the lifetime only, so at overhead o they read low by at most 1 / (1 + o); "
                                   "a whole-CTA slowdown leaves theta unbiased only when the finish spread scales with L, "
                                   "and reads it low by the same 1 / (1 + o) when the spread is fixed in ns, so the bias "
                                   "lies in [1 / (1 + o), 1] either way (it reads high only under a cost that is not "
                                   "uniform across CTAs). "
                                   "C3-R1's IN band is 0.275 +- 9%: at o = 10% a true 0.275 reads 0.250, IN's edge, and "
                                   "no worse than NEAR up to o = 37%; C3-R2's cut at g 0.5 separates C5's same-wave "
                                   "0.25 to 0.30 from M4's 1 at any o up to 80% (1 / 1.8 > 0.5). The theta band binds, "
                                   "so the median cap is 10%. The worst-pair cap, 20%, allows burst noise around a "
                                   "median under 10% but voids a page whose cost is far from uniform across bursts, "
                                   "where order could move. A stamped-against-plain launch-order test is not possible: "
                                   "the plain kernel carries no stamps"),
            "reference": ("the perturb unit's plain-kernel medians are C3's uninstrumented reference: C3-R3's e_plain and the "
                          "stamped-over-plain cost are read against them")},
        "units": {
            "C3-P": {"units": ["- prelude"], "for": "the power read-back and the 1410 lock probe (validity)"},
            "C3-F": {"units": ["olmoe-1b-7b bytes f128", "mixtral-8x7b bytes f128", "- gate gband"],
                     "for": ("C2's sag estimator pages on this board (128x128 G 64, treads 1 to 9), so the board is "
                             "comparable with A2's and C2's, then G-BAND (validity)")},
            "C3-G": {"units": ["- regcheck rc3", "- perturb pt3"],
                     "for": "the copy's two checks before any stamp, and the uninstrumented reference (the gate)"},
            "C3-S7": {"units": ["mixtral-8x7b stamps st7"], "for": "C3-R1, C3-R2, C3-R3 (Mixtral-8x7B, w1 and w2)",
                      "cells": "A2's E1-CAL Mixtral 128 G 1 cells: n 2, PT, PW, uniform, balanced; NATIVE"},
            "C3-SQ": {"units": ["qwen2-57b-a14b stamps stq"],
                      "for": "the Qwen2-57B contrast (C3-R3's Qwen2 w1, theta and same-wave on Qwen2 w2, DESCRIPTIVE)",
                      "cells": "A2's E1-CAL Qwen2 128 G 1 cells: n 1 and 2, PT, PW, uniform, balanced; NATIVE"}},
        "readings": {
            "C3-R1": {"what": "theta_L, C5's launch spread of a later wave (refit4 DECLARED.txt AM2), on Mixtral-8x7B w2",
                      "read_by": "read_c3.launch_reading via wave_theta; model_reading (median over launches)",
                      "rule": ("per w2 launch, each wave k >= 1 of W_c = 264 live ranks with at least 132 CTAs: least "
                               "squares start_ns = a + s u / W_c (u = r - k W_c), theta_k = s / L, L the launch's median "
                               "live lifetime; the launch's theta the CTA-weighted mean; theta_L the median over the "
                               "model's valid w2 launches"),
                      "band": "IN [0.25, 0.30]; NEAR [0.20, 0.35] outside IN; OUT otherwise",
                      "meaning": R3.MEANING["theta"]},
            "C3-R2": {"what": "the share of w2 slab re-reads launched with their partner, Mixtral-8x7B",
                      "read_by": "read_c3.pairs, same_wave, gap_ratio (launch_reading); pooled in model_reading",
                      "rule": ("a re-read is live rank r whose M-tile pid_m - 1 holds the same expert (expert_ids as vLLM "
                               "passed the launch); g = (start(r) - start(r - P)) / (P L / W_c), P = N / 128 (M4's "
                               "staggered gap is 1, C5's same-wave pair theta_L, C3's synchronized wave 0); the banded "
                               "share (review MAJOR-1): g < 0.5 over the re-reads in waves k >= 1 that the schedule puts "
                               "in their partner's wave, floor(r / W_c) = floor((r - P) / W_c) (wave 0 launches together "
                               "under every picture; C5 and C3 read 1.000, M4 0.000), pooled over the model's valid w2 "
                               "launches whose globaltimer resolution is at most a quarter of P L / W_c (review MINOR-4); "
                               "the all-pairs share and the schedule share printed beside (DESCRIPTIVE; REFIT.md's 82 to "
                               "88% is the schedule_note's)"),
                      "band": "LAUNCH-TOGETHER >= 0.75; STAGGERED <= 0.25; MIXED between",
                      "meaning": R3.MEANING["samewave"]},
            "C3-R3": {"what": "why Mixtral-8x7B's 128 G 1 w1 runs faster than priced: per-CTA life against packing",
                      "read_by": "read_c3.r3_cell, r3_model",
                      "rule": ("per cell T_pred = K1 at its LOMO constants (refit4 ROUND2's selected knee; N0 printed "
                               "beside), registered by value; L_pred = T_pred W_c / live; T_stamp the stamped launch's "
                               "event-pair time (median over calls; each call held ahead), L_med its median live "
                               "lifetime (globaltimer); ln(T_pred / T_stamp) = ln(L_pred / L_med) [LIFE] + ln(live L_med "
                               "/ (W_c T_stamp)) [PACKING], exactly; owner 2026-10-26 (review MAJOR-3): LIFE and the "
                               "total are corrected for the cell's measured w1 overhead o (L and T_stamp over 1 + o; "
                               "PACKING unchanged; the raw terms printed beside); per model the means over cells; life "
                               "share = mean LIFE / mean total (read when |mean total| >= 0.005); e_plain = mean T_pred / "
                               "T_plain - 1 over the cells, T_plain the perturb unit's plain median on the same cell"),
                      "band": ("on Mixtral-8x7B w1 under K1: NOT REPRODUCED when e_plain < +2%; DESCRIPTIVE ONLY when the "
                               "w1 overhead (the median over cells of |o|) exceeds a quarter of e_plain; else LIFE when "
                               "the life share >= 2/3, PACKING <= 1/3, BOTH between; on PERTURBED pages DESCRIPTIVE ONLY "
                               "whatever the band (owner OD-1)"),
                      "meaning": R3.MEANING["w1"]},
            "C3-R4": {"what": "validity of every reading", "read_by": "read_c3.read, launch_reading",
                      "rule": ("NOT READ unless: the board valid (read_c2.board: C2_POWER_W=700, LOCK_PROBE_1410=held, "
                               "G-BAND PASS); the variant's regcheck line and the gate's HARD line PASS (a timing FAIL "
                               "alone reads PERTURBED, owner OD-1), the page's label the gate's; every stamped config passes "
                               "C3's SASS legs; the page's histogram is the registered file; and per launch every row "
                               "stamped live or dead, the live CTAs exactly the pids under (num_tokens_post_padded / 128) "
                               "x P, USE_TD False, late_buffers 0 (review MINOR-7), 132 distinct SMs, the globaltimer "
                               "resolution (the least positive "
                               "difference of distinct start stamps) at most L / 10: a 1.024 us timer still admits every C3 "
                               "GEMM, whose CTAs live about 20 us and longer at 1410 MHz")}},
        "descriptive": ["theta on w1 and on Qwen2-57B w2", "the pair-gap theta: median g over schedule-same-wave pairs in "
                        "waves >= 1", "the median g over all pairs", "the per-CTA SM clock (life cycles / life ns)",
                        "first-wave over later-wave median lifetime", "life IQR / median", "the stamped over the plain "
                        "time per cell (the instrument's cost on the stamped call)", "Qwen2-57B's C3-R3 under K1 and N0"],
        "constants": {"W_c": R3.W_C, "SMs": R3.SMS, "CTAs_per_SM": R3.CTAS_PER_SM, "min_wave_ctas": R3.MIN_WAVE,
                      "r2_timer_fraction_of_gap": R3.R2_TIMER_FRAC, "r3_overhead_fraction_of_miss": R3.R3_OVERHEAD_FRAC,
                      "overhead_median_max": R3.OVERHEAD_MEDIAN_MAX, "overhead_worst_max": R3.OVERHEAD_WORST_MAX,
                      "theta_in": list(R3.THETA_IN), "theta_near": list(R3.THETA_NEAR),
                      "together_gap_below": R3.GAP_TOGETHER, "together_band": R3.TOGETHER, "staggered_band": R3.STAGGERED,
                      "life_share_hi": R3.LIFE_HI, "life_share_lo": R3.LIFE_LO, "reproduce_min_pct": R3.REPRO_MIN_PCT,
                      "timer_max_fraction_of_L": R3.TIMER_MAX_FRAC, "share_min_total": R3.SHARE_MIN_TOTAL,
                      "form": R3.FORM, "form_beside": R3.FORM_BESIDE},
        "priced": {"file": R3.PRICED, "sha256": sha256(REPO / R3.PRICED), "what": priced["what"],
                   "generator": priced["generator"], "sources_sha256": priced["sources_sha256"],
                   "rows": len(priced["rows"])},
        "next_round_gate": ("unchanged from refit4 (DECLARED.txt C, DECLARED2.txt R2-D): before B3 is registered, a model "
                            "round must put every held-out 128 G 1 model mean within 1.0% (OLMoE, Qwen2-57B, Mixtral-8x7B on "
                            "A2's cells, Mixtral-8x22B, and Mixtral-8x7B on C2's board as well) AND have an estimated "
                            "P(B3 E1-TILE holds) under A of at least 0.5 (read_c2.pass_estimates' method). C3 changes "
                            "neither clause; a constant C3 measures enters that round as MEASURED"),
        "schedule_note": {
            "statement": ("OD-4 (owner, 2026-10-26): refit4 REFIT.md section 2a's 'a re-read P = 32 or 48 ranks behind its "
                          "partner falls in the same wave for 82 to 88% of pairs' does NOT reproduce for Mixtral-8x7B on "
                          "C3's cells: the schedule share is 0.896 (PT, PW) and 0.969 (uniform, balanced), pooled 0.925. "
                          "Mixtral-8x22B's C2 cells give 0.812 to 0.900 (pooled 0.844), so the quoted range matches "
                          "8x22B's cells, not 8x7B's. It enters no band"),
            "derivation": ("128 G 1 w2, GROUP_SIZE_M 1: pid = pid_m P + pid_n with P = N / 128 (8x7B 4096 / 128 = 32; "
                           "8x22B 6144 / 128 = 48); expert e owns ceil(c_e / 128) consecutive M-tiles (moe_align_block_size, "
                           "experts ascending), c_e the cell's NATIVE rows per expert (the histogram's counts; balanced: "
                           "T top_k / E). A re-read is live rank r whose M-tile pid_m - 1 holds the same expert (its "
                           "partner r - P read the same B slab); it is same-wave when floor(r / 264) = floor((r - P) / "
                           "264), W_c = 132 SMs x 2 CTAs. 8x7B uniform and balanced n 2: 2 tiles per expert, 16 tiles, "
                           "512 live, 256 re-reads (the second tile of each expert); only r in [264, 295] crosses, pid_m 9, "
                           "pid_n 0 to 7: 8 pairs, 248 / 256 = 0.969. 8x7B PT (tiles 2,2,3,2,3,3,2,3) and PW "
                           "(4,3,2,1,3,2,2,3): 20 tiles, 640 live, 384 re-reads, 40 cross: 0.896. Restricted to waves "
                           "k >= 1 the shares are 0.815 to 0.938. read_c3.pairs and same_wave compute these on every "
                           "page from the expert_ids vLLM handed the launch")},
        "owner_decisions": [
            "OD-1 (decided 2026-10-26): a gate failing on its timing legs alone keeps the stamps units, pages labelled "
            "PERTURBED with the overhead beside each reading; C3-R1 and C3-R2 read under the NOT-READ bound, C3-R3 "
            "descriptive only; the hard legs and regcheck still refuse",
            "OD-4 (decided 2026-10-26): REFIT.md's 82 to 88% does not reproduce for Mixtral-8x7B (schedule_note)",
            "MAJOR-3 (decided 2026-10-26): C3-R3's split corrected for the measured overhead, its band read only when "
            "the w1 |overhead| is at most a quarter of the plain-kernel miss, else DESCRIPTIVE ONLY",
            "OD-2: no published rental 4/5 stamp page exists, so the reader is tested on synthetic pages in the probe's "
            "on-disk layout and on rental 5's real gate.env and regcheck.env"],
        "plan": {"file": R3.PLAN, "sha256": sha256(REPO / R3.PLAN), "units_count": len(units), "units": units,
                 "est_total_min": est, "est_nodrop_min": nodrop, "drop_order_last_first": order,
                 "boot_setup_min": BOOT, "wall_min": wall, "rate_usd_per_h": RATE, "cost_usd": round(wall / 60 * RATE, 2),
                 "deadline_after_launch_min": DEADLINE, "hard_cap_after_launch_min": HARD_CAP,
                 "cost_at_hard_cap_usd": round(HARD_CAP / 60 * RATE, 2)},
        "pages": {m: page_entry(m) for m in R3.HISTS},
    }
    d["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    return d


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: the calibration rental C3, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", "", f"SEEN: {d['seen']}", "", "STEP 1 CHECK"]
    L += [f"  {k}: {v}" for k, v in d["check"].items()]
    for sec in ("power_and_lock", "instrument", "gate"):
        L += ["", sec.upper().replace("_", " ")] + [f"  {k}: {v if isinstance(v, str) else json.dumps(v)}"
                                                     for k, v in d[sec].items()]
    L += ["", "UNITS"]
    for k, v in d["units"].items():
        L.append(f"  {k}: {json.dumps(v['units'])}: {v['for']}" + (f" [{v['cells']}]" if "cells" in v else ""))
    L += ["", "READINGS"]
    for k, v in d["readings"].items():
        L.append(f"  {k}:")
        for kk, vv in v.items():
            if isinstance(vv, dict):
                L.append(f"    {kk}:")
                L += [f"      {b}: {t}" for b, t in vv.items()]
            else:
                L.append(f"    {kk}: {vv}")
    L += ["", "DESCRIPTIVE (no band): " + "; ".join(d["descriptive"]), "",
          "CONSTANTS: " + json.dumps(d["constants"], sort_keys=True), "",
          f"PRICED: {d['priced']['file']} sha256 {d['priced']['sha256']} ({d['priced']['rows']} rows): {d['priced']['what']}",
          f"  generator: {d['priced']['generator']}"]
    L += [f"  {k} {v}" for k, v in d["priced"]["sources_sha256"].items()]
    L += ["", "SCHEDULE NOTE (OD-4)", f"  {d['schedule_note']['statement']}", f"  derivation: {d['schedule_note']['derivation']}"]
    L += ["", f"NEXT ROUND GATE: {d['next_round_gate']}", "", "OWNER DECISIONS"] + [f"  {x}" for x in d["owner_decisions"]]
    L += ["", f"PLAN {p['file']} sha256 {p['sha256']}: {p['units_count']} units, {p['est_total_min']} min estimated "
              f"({p['est_nodrop_min']} nodrop), drop order last first {p['drop_order_last_first']}"]
    for u in p["units"]:
        L.append(f"  {u['model']} {u['step']} {u['label']} est {u['est']} cap {u['cap']} {u['drop']}")
    L += [f"  wall about {p['wall_min']} min with {p['boot_setup_min']} min of boot and setup: ${p['cost_usd']:.2f} at "
          f"${p['rate_usd_per_h']}/h; deadline launch + {p['deadline_after_launch_min']} min; guardian hard cap launch + "
          f"{p['hard_cap_after_launch_min']} min (${p['cost_at_hard_cap_usd']:.2f} at most)", "", "PAGES"]
    for m, v in d["pages"].items():
        L.append(f"  {m}: {v['path']} {v['sha256']} cells {v['cells']} ({v['unit']}, variant {v['variant']})")
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
    jtext = json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n"
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
