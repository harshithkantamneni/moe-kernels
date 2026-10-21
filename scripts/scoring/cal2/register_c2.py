#!/usr/bin/env python
"""Writes the calibration rental C2's registration, docs/registered/2026-10-22-cal2-gh200.{json,txt}, before any page.

    python scripts/scoring/cal2/register_c2.py --write [--out-dir docs/registered]
    python scripts/scoring/cal2/register_c2.py --check      # recompute, compare (exit 1 on any difference)

C2 is a CALIBRATION rental: it certifies nothing; every C2 page becomes SEEN for v5'' and rental B3 once read. It is
registered anyway, before any page: what each page is for; exactly how the four-model held-out table is computed
(the knee round's held-out method and pass-probability method, by script path and function); what each outcome means
for the B3 decision (knee-v5 KNEE.md section 8's bounds); and that v5'' and the averaged-fit sag estimator are FROZEN
as of this registration: their code by sha256 (every work file the chain imports or reads, and the repo modules it
runs), their constants by value, and a fingerprint (v5'''s ms on 8x22B's six NATIVE cells at A2's board sag) the
reader must reproduce bit for bit before it reads a C2 page, so C2 cannot re-tune them before the held-out reading.

Building the fingerprint imports the frozen chain (knee-v5/work/k5.py builds the CAL pricer, about 50 s). `--check`
recomputes everything and compares with the committed JSON (the `code_pins` block excluded, as every registration
since rental 3) and text.
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
import read_c2 as RC  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = RC.REG_NAME
PLAN = "scripts/plans/rentalC2-cal-2026-10.plan"
HIST = RC.HIST
MIX_HIST = "docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json"
KNEE = "moe-kernels-work 2026-10-22 knee-v5/KNEE.md (offline, read-only over the repo at 6dfa9e35)"
#: every repo file C2's driver, gate, census, reader, generator and register run, and the repo modules the frozen
#: chain imports (traced: k5, sagtau and a priced cell)
PINNED = (
    "scripts/gh200_cal2_session.sh", "scripts/gh200_settleS1_session.sh", "scripts/gh200_tileA2_session.sh",
    "scripts/tileA2_gate.py", "scripts/tile_census_a2.py", "scripts/tile_census.py", "scripts/r3_tile.py",
    "scripts/dram_counter_route_tile.py", "scripts/dram_counter_route.py", "scripts/private_weight_reference.py",
    "scripts/locked_r3.py", "scripts/calibrate_hardware.py",
    "scripts/scoring/cal2/hist_c2.py", "scripts/scoring/cal2/read_c2.py", "scripts/scoring/cal2/register_c2.py",
    "scripts/skew_synth_tile.py", "scripts/skew_synth.py",
    "scripts/r3_timing_model.py", "scripts/r3_timing_model_v3.py", "scripts/r3_timing_model_v4.py",
    "scripts/r3_timing_model_v4L.py", "scripts/wave_split_bytes.py", "scripts/wave_split_bytes_v3.py",
    "scripts/wave_split_bytes_v4.py",
    "scripts/scoring/tileA/regtile.py", "scripts/scoring/tileA/tcommon.py", "scripts/scoring/tileA/finalise.py",
    "scripts/scoring/tileA/power_tile.py", "scripts/scoring/tileA2/a2common.py", "scripts/scoring/tileA2/score_floor_a2.py",
    "scripts/scoring/rental6/r6model.py", "scripts/scoring/rental6/r6common.py", "scripts/scoring/rental6/reg6.py",
    "scripts/scoring/rental7/r7common.py", "scripts/scoring/rental3/r3common.py",
)
#: the pins: the files above and every repo module the frozen chain imports from its repo (read_c2.CHAIN_REPO_MODULES)
PINNED = tuple(dict.fromkeys(PINNED + RC.CHAIN_REPO_MODULES))
BOUNDS_X = (0.0, 1.0, 1.6, 3.13)


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


def file_entry(rel: str) -> dict:
    """The page's registration entry in rental A's files_block form (tcommon.timed_page reads it for G5)."""
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "rental6"))
    import reg6 as R6
    doc = json.loads((REPO / rel).read_text())
    return {"path": rel, "sha256": sha256(REPO / rel), "shuffle_seed": doc["shuffle_seed"], "tile": [128, 128, 64, 8, 3],
            "group_m": doc["group_m"], "model": doc["model"],
            "cells": [{"label": c["label"], "n": c["n"], "arms": c["arms"], "counts_sha256": R6.sha_counts(c["counts"]),
                       **({"seed": c["provenance"]["seed"], "kind": c["provenance"]["kind"],
                           "param": c["provenance"]["param"], "source_model": c["provenance"]["source_model"]}
                          if c.get("provenance") else {})} for c in doc["cells"]]}


def fingerprint() -> tuple[list, list]:
    """v5'''s ms (k5.cell_ms at p 5.75, F1 = A2's par) and v4's ms (the drop rule's prediction) on 8x22B's six
    NATIVE cells, rounded to 6 decimals."""
    doc = json.loads((REPO / HIST / "mixtral-8x22b-128g1.json").read_text())
    rows = [{"model": RC.MODEL, "page": doc["page"], "n": c["n"], "label": c["label"], "doc": doc, "c": c}
            for c in doc["cells"] if "native" in c["arms"]]
    v5 = [round(x, 6) for x in RC.predict(rows, RC.A2_PAR_128)]
    k5, _ = RC.chain()
    d5 = k5.d5
    prm = d5.params_for(d5.SEL, RC.MODEL)
    with d5.A2.scoring(d5.RULES):
        v4 = [round(d5.RG.price_cell(RC.MODEL, "native", r["c"], doc, prm, key=d5.SEL["key"])["ms"], 6) for r in rows]
    return [f"n{r['n']}/{r['label']}" for r in rows], v5, v4


def build() -> dict:
    import hist_c2 as HC
    units = plan_units()
    est = sum(u["est"] for u in units)
    nodrop = sum(u["est"] for u in units if u["drop"] == "nodrop")
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    boot = 25
    wall = est + boot
    deadline, cap = 90, 105
    keys, v5, v4 = fingerprint()
    x22 = json.loads((RC.WORK / "2026-10-22/knee-v5/work/x22.json").read_text())["power_bounds"]
    bounds = {f"X {X:+.2f}": RC.pass_estimates(X) for X in BOUNDS_X}
    assert bounds == x22, "the ported power does not reproduce knee-v5 x22.json's bounds"
    sagtau = json.loads((RC.WORK / "2026-10-22/knee-v5/work/sagtau.json").read_text())
    cal = sagtau["e1"]["v5'' (null, carried)"]["par"]["LOMO 128g1"]
    assert cal == RC.CAL_HELD_OUT, cal
    per = sagtau["per_page_fits"]["SEEN"]["128"]
    assert {m: tuple(v) for m, v in per.items()} == RC.A2_PAGE_FITS, per
    d = {
        "registered": "2026-10-22, before any C2 page",
        "name": NAME,
        "status": ("CALIBRATION: C2 certifies nothing and scores no verdict; every C2 page becomes SEEN for v5'' and B3 "
                   "once read. The bands below say what the fourth held-out model means for the B3 decision; the "
                   "decision is the owner's"),
        "design": (f"{KNEE}, section 8 and section 5's units; the owner's option (b) of 2026-10-22: one short calibration "
                   "rental adding Mixtral-8x22B as a fourth held-out CAL model in the large-K, x near 1, G 1 regime "
                   "(where blind Phi-3.5 sits) plus the missing A2-type Mixtral-8x7B b128g1 byte page, before B3 is "
                   "registered. v5'' is design-v5's v5 (moe-kernels-work 2026-10-21 design-v5/DESIGN.md) as revised by "
                   "diag-v5 (2026-10-22 diag-v5/DIAGNOSIS.md), with knee-v5's recommended F1 estimator 'par'"),
        "tool": ("scripts/gh200_cal2_session.sh (S1's pinned driver plus lines marked C2), scripts/tileA2_gate.py band "
                 "(A2's G-BAND), scripts/tile_census_a2.py (the census), scripts/scoring/cal2/hist_c2.py (the 8x22B "
                 "page), scripts/scoring/cal2/read_c2.py (the reader)"),
        "seen": ("Every C2 page is SEEN for v5'' and B3 once read: Mixtral-8x22B joins OLMoE-1B-7B, Mixtral-8x7B and "
                 "Qwen2-57B-A14B as a CAL model, and any later fit that uses it (a four-model refit) is SEEN-FITTED on it. "
                 "C2 runs no cell of B3's blind models (Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, JetMoE-8B; rental B2's frozen "
                 "pages, docs/registered/2026-10-20-tileB2-e1tile-gh200) and no 8x22B cell exists at 128x128 or 64x128 "
                 "before C2 (knee-v5 section 8: every 8x22B page is at R3's tile, BLOCK_M 32, 1710 MHz). OD13 kept: no "
                 "time is priced against rental A's 1710 E1-CAL pages"),
        "frozen": {
            "statement": ("v5'' and the averaged-fit sag estimator are FROZEN as of this registration. The reader refuses "
                          "unless every file below has its sha256 and v5'' at A2's board sag reproduces the fingerprint "
                          "to 6 decimals; no constant below is refitted, re-chosen or re-read from a C2 page before "
                          "C2-R1 is read. A change made after C2's pages are seen is SEEN-FITTED on 8x22B and is "
                          "disclosed as such. THE FREEZE IS TIED TO THE RENTAL'S COMMIT: read_c2.read takes the commit from "
                          "the ledger's 'driver start: commit <sha>' line (gh200-driver/status, checked against "
                          "session/commit.txt) and refuses unless git show <sha>:<path> equals the working copy for the "
                          "registration JSON and text, the 8x22B page, the Mixtral page, the plan and every code_pins path "
                          "(read_c2.py, the driver and the repo modules the chain imports among them), every code_pins "
                          "path has its registered sha256, and every repo module the chain loads from the repo it runs "
                          "(diag-A2 common.REPO) is a code_pins path with its registered bytes there"),
            "chain_repo_modules": list(RC.CHAIN_REPO_MODULES),
            "root": "~/code/moe-kernels-work (paths below relative to it); the chain imports the repo at ~/code/moe-kernels",
            "files": RC.frozen_hashes(),
            "constants": {
                "knee form": "null (v5''), the carried bw 3597.8 GB/s (the CAL engine's v2 fit x[2], SEEN-FITTED, CARRIED)",
                "p_128": RC.P128, "p_64": RC.P64,
                "p_128 source": ("knee-v5 forms.json \"v5'' (null, carried bw)\" SEEN constants: the fit on all three CAL "
                                 "models' 118 NATIVE 128x128 counter rows (A2 and S1); 8x22B is in none, so p_128 5.75 IS "
                                 "its held-out fold"),
                "bytes": "M4 registered at 128x128, v3 key (k5.F3_M4: diag-v5 bytes_lomo.json 'none (SEEN)' 'M4')",
                "F4": "rho x 2 at both tiles",
                "T0, c, bw": "43.23 us, 202.548 ns, 3597.8 GB/s, as v4 (asserted at import by d5.py and k5.py)",
                "R3's 1410 sag share in T0": [6.11, 50.0],
                "F1 estimator": ("'par' (knee-v5 sagtau.py estimators): per estimator page one (A, tau) by sagtau.fit "
                                 "(design-v5/work/sag.py fit: tau on 50 to 4975 us step 25, A least squares, A rounded to "
                                 "0.01 us) over sag.pts_of(page, 1410) (off = t_ncu - cycles / 1410 per GEMM), then A "
                                 "and tau averaged; the Sc form A (1 - exp(-t / tau)) on each GEMM's floor-only time"),
                "F1 pages": "C2's OLMoE f128 and Mixtral-8x7B f128 (C2-F), both: 8x22B has no estimator page to drop",
                "A2 board F1 (the fingerprint's)": {"per_page": {m: list(v) for m, v in RC.A2_PAGE_FITS.items()},
                                                    "par": list(RC.A2_PAR_128)},
                "selections": "A2's (moe-kernels-work 2026-10-20 scoreA2 via finalise.selections; occupancy every model's)",
                "scoring rules": "a2common.Rules.tileA2(1410)",
            },
            "fingerprint_cells": keys,
            "fingerprint_ms_at_A2_par": v5,
            "v4_ms (the drop rule's prediction)": v4,
        },
        "power_and_lock": {
            "power_limit_w": RC.POWER_W,
            "rule": ("the plan prelude's power-limit-w 700: the pre-setup dry run and the prelude refuse a card whose "
                     "power.max_limit is under it (A2's P-W); nvidia-smi -pl 700 -sc 0, then S1's read-back: poll every 2 s, "
                     "up to 30 s, until --query-gpu=enforced.power.limit and nvidia-smi -q -d POWER's GPU-scope Current "
                     "Power Limit both read 700 W within 1 W (power-set.txt), else the session STOPS (c2-power.env "
                     "C2_POWER_700=refused, the last reading in power-set-unsettled-700.txt, which the ledger names). No "
                     "fallback limit. 900 W is not attempted: in S1's publication -pl 900 -sc 0 left the enforced GPU "
                     "limit at 700 W, and S1 and (most likely) A2 ran at 700 W, so 700 W keeps C2 comparable with both"),
            "session_lock_mhz": RC.LOCK,
            "lock_rule": ("every unit but calibrate and the census runs under nvidia-smi -lgc 1410 with ncu --clock-control "
                          "none (A2's and S1's rung; v5'' is fitted there)"),
            "probe": ("A2's lock-hold probe in its cheapest form: (1) the prelude locks 1410, reads clocks.sm back "
                      "(lock_and_check, two tries), resets, records LOCK_PROBE_1410=held, and STOPS the session when it "
                      "does not read back (S1's prelude probe, made a stop); (2) A2's G-BAND (tileA2_gate.py band, "
                      "score_floor_a2.choose_band) after the two f128 pages: joint_cF over them in the counter band at "
                      "1410 (primary [0.975 L, L], else the registered fallback [0.959 L, L]) is not None, else the session "
                      "ends there. A2's timed probe pages (27 min on A2) are not run"),
        },
        "units": {
            "C2-P": {"units": ["- prelude", "mixtral-8x7b calibrate tc2", "- tilecensus tc2"],
                     "for": ("the power read-back and the lock probe (the reading's validity); calibrate's ruler, which "
                             "the timed unit hands R3 (A2's); 8x22B's compiled occupancy at 128x128 (C2-S5, DESCRIPTIVE)")},
            "C2-F": {"units": ["olmoe-1b-7b bytes f128", "mixtral-8x7b bytes f128", "- gate gband"],
                     "for": ("the averaged-fit sag estimator's two pages ON C2'S BOARD, so 8x22B's cells are priced as B3 "
                             "will price blind cells (per-page fits averaged, MEASURED-ON-BOARD); then G-BAND"),
                     "cells": "128x128x64 w8 s3, G 64, treads 1 to 9, NATIVE, SHARED and PRIVATE (A2's f128 pages)"},
            "C2-T": {"units": ["mixtral-8x22b timed t128g1"],
                     "for": "C2-R1: Mixtral-8x22B's held-out 128 G 1 time, the fourth held-out model",
                     "cells": (f"{HIST}/mixtral-8x22b-128g1.json: n 2 (the only tread whose vLLM default config is "
                               "128x128x64 w8 s3 G 1 for E 8, top_k 2), PT, PW, PT2, PW2, uniform, balanced, NATIVE and "
                               "SHARED; E1-CAL's instrument (R3 through r3_tile.py, 9 declared copies, slip-policy cell, "
                               "r7-median-run3)")},
            "C2-B": {"units": ["mixtral-8x22b bytes b128g1"],
                     "for": "C2-S2 and C2-S3: 8x22B's counted bytes, cycles and sag on the same cells",
                     "cells": "the same file, A2's byte-page instrument at 1410"},
            "C2-MT": {"units": ["mixtral-8x7b timed t128g1m"],
                      "for": ("C2-S6 (owner, 2026-10-22; DESCRIPTIVE, changes no band): Mixtral-8x7B's held-out 128 G 1 "
                              "mean on C2's board beside A2's +3.13, and with C2-M the GEMM against per-call split on one "
                              "board; drop-group m7 with C2-M"),
                      "cells": f"{MIX_HIST} (A2's E1-CAL Mixtral 128 G 1 cells), NATIVE and SHARED, NATIVE scored; E1-CAL's instrument"},
            "C2-M": {"units": ["mixtral-8x7b bytes b128g1"],
                     "for": ("C2-S4 and C2-S6's split: the A2-type Mixtral-8x7B 128 G 1 byte page (A2 ran none): is S1's b128g1 page (board "
                             "9da3c0, the only one, +5.42% w1 cycles under v5'') the outlier, or the model?"),
                     "cells": f"{MIX_HIST} (the file A2's E1-CAL Mixtral 128 G 1 timed page ran), NATIVE and SHARED"},
        },
        "readings": {
            "C2-R1": {
                "what": "X, Mixtral-8x22B's held-out 128 G 1 mean, percent; e = T_model / T_meas - 1 per cell",
                "read_by": "scripts/scoring/cal2/read_c2.py read (timed_rows, sag_fits, held_out)",
                "method": ("the knee round's held-out method (knee-v5/work/forms.py main, LOMO block, and sagtau.py main, "
                           "'par'): cells by v5price.e1cal_rows' loop (tcommon.timed_page at G 1 with the registered "
                           "file entry below: G5 provenance, the r7-median-run3 rebuild, the ALL view; tcommon.row_drop "
                           "against v4's ms); each kept NATIVE cell priced by knee-v5/work/k5.py cell_ms(r, None, 5.75, "
                           "(A, tau) par of C2-F, None); X = v5prime.summ(...)['model_class_means']['mixtral-8x22b "
                           "128g1'] (k5.summ)"),
                "table": ("read_c2.heldout_table: the three CAL models' registered LOMO held-out 128 G 1 means under v5'' "
                          "+ F1 par (knee-v5/work/sagtau.json e1[\"v5'' (null, carried)\"][\"par\"][\"LOMO 128g1\"]: "
                          f"{json.dumps(RC.CAL_HELD_OUT)}), unchanged by C2 (no C2 page enters their folds), and X"),
                "pass_estimate": ("read_c2.pass_estimates(X): diag-v5/work/power5.py power (ported verbatim as read_c2.power; "
                                  "it reproduces x22.json's twelve bounds bit for bit, asserted here) on B2's 84 registered "
                                  "NATIVE cells, R 200,000, seed 20261022, as knee-v5/work/x22.py: A = power(b_class the "
                                  "mean of the four b = -ln(1 + e), sd their SD), B = power(0, their RMS), D = power(the "
                                  "mean of OLMoE, Qwen2-57B and X, their SD)"),
                "beside X (DESCRIPTIVE)": ("X4: X over A2's four labels (PT, PW, uniform, balanced; uniform and balanced "
                                           "have the same counts, so the six-cell X weights the balanced routing twice); "
                                           "X at A2's par (42.02 us, 425 us): X less this separates C2's board sag from "
                                           "the model's error"),
                "refusals": ("read_c2.read refuses (no reading at all) on a frozen work file, the rental's commit, a "
                             "code_pins sha256, a chain repo module, or the fingerprint (FROZEN above); the timed page "
                             "is NOT SCORED unless tcommon.timed_page admits it and its report records lock_mhz 1410"),
                "validity": (f"the reading is NOT READ unless the board reads valid (c2-power.env C2_POWER_W=700, "
                             f"LOCK_PROBE_1410=held, G-BAND PASS), both C2-F pages give a fit, the C2-T page is admitted, and "
                             f"at least {RC.MIN_CELLS} of its 6 NATIVE cells are kept"),
                "bands": {
                    "CLEAN": f"|X| <= {RC.BAND['clean']:g}",
                    "PARTIAL": f"{RC.BAND['clean']:g} < X < {RC.BAND['regime']:g}",
                    "REGIME": f"X >= {RC.BAND['regime']:g}",
                    "OPPOSITE": f"X < {RC.BAND['opposite']:g}",
                    "NOT READ": "the validity clause fails",
                },
                "means_for_B3": {b: RC.outcome(x, 6)["b3"] for b, x in
                                 (("CLEAN", 0.0), ("PARTIAL", 1.5), ("REGIME", 3.0), ("OPPOSITE", -2.0))}
                | {"NOT READ": RC.outcome(None, 0)["b3"]},
                "bounds (knee-v5 section 8; the estimates at X, v5'' + F1 par held fixed)": bounds,
                "why these cuts": ("CLEAN changes which estimate is credible (D over A), not A: the four-model A stays at "
                                   "or under 0.09 at every X. 1.0 is the gate's own bar on a held-out mean (knee-v5 section 5) and the X at which D "
                                   "is 0.565; 2.0 is two thirds of Mixtral-8x7B's +3.13, past which D falls under 0.3 "
                                   "(0.303 at +1.60): the miss is mostly reproduced. Below -1.0 the miss changes sign"),
            },
            "C2-S1": {"what": "8x22B's x (bytes-only / floor-only) per GEMM on its kept cells (knee-v5 x22.py's xs)",
                      "read_by": "scripts/scoring/cal2/read_c2.py x_of_rows", "expected (x22, INFERENCE)":
                      "w1 1.16 to 1.19, w2 0.97 to 1.07 on Mixtral-8x7B's cells with the model swapped"},
            "C2-S2": {"what": ("8x22B b128g1: M4 (v3 key) reads over counted dram_bytes_read - 1, and the cycles-domain "
                               "error of the v5'' knee (null, carried bw, p 5.75) per GEMM, NATIVE cells"),
                      "read_by": "scripts/scoring/cal2/read_c2.py counter_reading (k5.capture_rows, k5.eval_rows, v5price.f3_reads)"},
            "C2-S3": {"what": "8x22B's own sag on its b128g1 page: C2's par estimator minus the measured off, us; its own fit",
                      "read_by": "scripts/scoring/cal2/read_c2.py sag_domain"},
            "C2-S4": {"what": ("Mixtral-8x7B b128g1 (A2 cells): the cycles-domain error at p 5.75 per GEMM on C2's board "
                               "beside S1's b128g1 page (+5.42% w1, -0.94% w2 in knee-v5 forms.json) and A2's held-out "
                               "time +3.13% on the same cells"),
                      "read_by": "scripts/scoring/cal2/read_c2.py counter_reading",
                      "reading": (f"DESCRIPTIVE (read_c2.s4_reading on the w1 means, points, S1's w1 +5.42): SAME-AS-S1 "
                                  f"when |C2 - S1| <= {RC.S4_NEAR:g} (the G 1 counter behaviour is Mixtral-8x7B's on two "
                                  f"boards, not a board outlier); BOARD-SPECIFIC when |C2| <= {RC.S4_NEAR:g} (S1's page was "
                                  "its board's); BETWEEN only strictly between those two (0 and S1's, each widened by "
                                  f"{RC.S4_NEAR:g}); BEYOND-S1 past S1's; OPPOSITE past 0 on the far side. None changes C2-R1")},
            "C2-S6": {"what": ("Mixtral-8x7B t128g1m on C2's board (A2's E1-CAL cells, NATIVE): its held-out 128 G 1 mean "
                               f"M in the fold A2's +3.13 was read in (p_128 {RC.MIX_FOLD_P}, forms.json's Mixtral LOMO "
                               "fold; F1 from C2's OLMoE f128 page alone, sagtau's Mixtral fold), beside +3.13; and the "
                               "split on one board, cell by cell with C2-M: the time error (ms) of that pricing, the "
                               "GEMMs' error (ms) on C2-M's cycles at the same p (k5.capture_rows, the null form, counted "
                               "bytes, no sag on either side), and their difference, the per-call part (T0, other, the "
                               "sag); M4's bytes against the counted reads beside it"),
                      "read_by": "scripts/scoring/cal2/read_c2.py timed_rows, held_out, split",
                      "reading": (f"DESCRIPTIVE (read_c2.s6_reading on M, percent): REPRODUCES when |M - 3.13| <= "
                                  f"{RC.S6_NEAR:g} (X and the three CAL means compare across boards for this class); BOARD "
                                  f"when |M| <= {RC.S6_NEAR:g} (boards move this class by about 3 points, far past B2's 0.26% "
                                  "board term: B3's estimate falls in every reading, and X minus M is the within-board "
                                  "contrast to read); BETWEEN strictly between the two; ABOVE past +3.13; OPPOSITE past 0 "
                                  "on the far side. It changes no band")},
            "C2-S5": {"what": ("the board: C2's per-page (A, tau) beside A2's, the power read-back, the lock probe, G-BAND; "
                               "8x22B's compiled CTAs per SM at 128x128 beside the 2 per SM v5'' prices (a difference is "
                               "reported, the frozen pricing is not changed)"),
                      "read_by": "scripts/scoring/cal2/read_c2.py board, census, triad",
                      "also": ("which G-BAND band passed (primary or fallback; A2 passed on the fallback); calibrate's "
                               f"triad ruler beside A2's {RC.TRIAD_A2_GBPS} GB/s (15 rentals {RC.TRIAD_RANGE_GBPS[0]} to "
                               f"{RC.TRIAD_RANGE_GBPS[1]})")},
        },
        "plan": {"file": PLAN, "sha256": sha256(REPO / PLAN), "units_count": len(units), "units": units,
                 "est_total_min": est, "est_nodrop_min": nodrop, "drop_order_last_first": order,
                 "boot_setup_min": boot, "wall_min": wall, "rate_usd_per_h": 2.29, "cost_usd": round(wall / 60 * 2.29, 2),
                 "deadline_after_launch_min": deadline, "hard_cap_after_launch_min": cap,
                 "cost_at_hard_cap_usd": round(cap / 60 * 2.29, 2)},
        "pages": {"files": {"mixtral-8x22b-128g1.json": file_entry(f"{HIST}/mixtral-8x22b-128g1.json"),
                            "mixtral-8x7b-128g1.json": file_entry(MIX_HIST)},
                  "generator": f"scripts/scoring/cal2/hist_c2.py ({', '.join(HC.PAGES)})"},
    }
    d["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    return d


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: the calibration rental C2, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", "", f"SEEN: {d['seen']}", "", "FROZEN", f"  {d['frozen']['statement']}",
         f"  root: {d['frozen']['root']}"]
    for k, v in d["frozen"]["files"].items():
        L.append(f"  {k} {v}")
    for k, v in d["frozen"]["constants"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L.append("  fingerprint (v5'' ms at A2's par; v4 ms): " + "; ".join(
        f"{c} {a} {b}" for c, a, b in zip(d["frozen"]["fingerprint_cells"], d["frozen"]["fingerprint_ms_at_A2_par"],
                                          d["frozen"]["v4_ms (the drop rule's prediction)"])))
    L += ["", "POWER AND LOCK"] + [f"  {k}: {v}" for k, v in d["power_and_lock"].items()]
    L += ["", "UNITS"]
    for k, v in d["units"].items():
        L.append(f"  {k}: {json.dumps(v['units'])}: {v['for']}" + (f" [{v['cells']}]" if "cells" in v else ""))
    L += ["", "READINGS"]
    for k, v in d["readings"].items():
        L.append(f"  {k}:")
        for kk, vv in v.items():
            L.append(f"    {kk}: {vv if isinstance(vv, str) else json.dumps(vv)}")
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
