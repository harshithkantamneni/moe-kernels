#!/usr/bin/env python
"""Writes the calibration rental C4's registration, docs/registered/2026-10-27-cal4-gh200.{json,txt}, before any page.

    python scripts/scoring/cal4/register_c4.py --write [--out-dir docs/registered]
    python scripts/scoring/cal4/register_c4.py --check      # recompute, compare (exit 1 on any difference)

C4 is a CALIBRATION rental: it certifies nothing; every C4 page becomes SEEN once read. It is registered anyway, before
any page: what each page is for; exactly how the five-model held-out table, the pass estimate and the w1 bytes reading
are computed (by script path and function); what each outcome means for the B3 decision; and that the round-4 model
is FROZEN as of this registration: its code by sha256 (every work file the chain imports or reads, and the repo
modules it runs), its constants by value, the reproduction of round 4's selected four-model table by the frozen code,
and a fingerprint (the frozen model's ms and W2's and M4's reads on tp2's six NATIVE cells at A2's board sag) the
reader must reproduce before it reads a C4 page.

Building it imports the frozen chain (round3/work r6.py: the CAL pricer is built at p 14, about 50 s) and re-prices
round 4's 32 timed cells from A2's and C2's exfil trees. `--check` recomputes everything and compares with the
committed JSON (the `code_pins` block excluded, as every registration since rental 3) and text.
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
import read_c4 as R4  # noqa: E402
import hist_c4 as HC  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = R4.REG_NAME
PLAN = R4.PLAN
HIST = R4.HIST
DESIGN = ("moe-kernels-work 2026-10-27 build-C4 (CANDIDATE.md, DESIGN.md; offline, read-only over the repo at 3f0566a9), "
          "after refit4 REFIT.md, ROUND2.md and round3 ROUND3.md, ROUND4.md")
#: every repo file C4's driver, gate, census, reader, generator and register run, and the repo modules the frozen
#: chain and the reader import (read_c4.CHAIN_REPO_MODULES)
PINNED = (
    "scripts/gh200_cal4_session.sh", "scripts/gh200_cal2_session.sh", "scripts/gh200_settleS1_session.sh",
    "scripts/gh200_tileA2_session.sh",
    "scripts/tileA2_gate.py", "scripts/tile_census_a2.py", "scripts/tile_census.py", "scripts/r3_tile.py",
    "scripts/dram_counter_route_tile.py", "scripts/dram_counter_route.py", "scripts/private_weight_reference.py",
    "scripts/locked_r3.py", "scripts/calibrate_hardware.py",
    "scripts/scoring/cal4/hist_c4.py", "scripts/scoring/cal4/read_c4.py", "scripts/scoring/cal4/register_c4.py",
    "scripts/scoring/cal2/read_c2.py",
    "scripts/skew_synth_tile.py", "scripts/skew_synth.py",
    "scripts/scoring/tileA2/score_floor_a2.py",
)
PINNED = tuple(dict.fromkeys(PINNED + R4.CHAIN_REPO_MODULES))
BOOT, DEADLINE, CAP, RATE = 25, 65, 80, 2.29


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


def stakes() -> dict:
    """The frozen model against v5'' on tp2's six cells at A2's par (stakes, not a prediction of record), and x."""
    rows = R4.native_rows(R4.page_doc())
    d = [round(100 * (R4.price(r, R4.A2_PAR_128) / R4.v5pp(r, R4.A2_PAR_128) - 1), 2) for r in rows]
    x = R4.x_of_rows(rows)
    lev = []
    for r in rows:
        w2, m4 = R4.reads(r), R4.reads(r, None)
        lev.append(round(100 * (w2["w1"] / m4["w1"] - 1), 2))
    return {"frozen model / v5'' - 1, % (n2/PT, PW, PT2, PW2, uniform, balanced)": d,
            "mean %": round(sum(d) / len(d), 2), "W2 / M4 w1 bytes - 1, %": lev, "x": x}


def build() -> dict:
    units = plan_units()
    est = sum(u["est"] for u in units)
    nodrop = sum(u["est"] for u in units if u["drop"] == "nodrop")
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    wall = est + BOOT
    assert (REPO / HIST / R4.PAGE).read_text() == HC.build(R4.PAGE[:-5]), "the page is not hist_c4's"
    combo = json.loads((R4.ROUND4 / "combo4.json").read_text())
    sel = combo["tables"][R4.SELECTED]["summary"]
    assert sel["means"] == R4.CAL_HELD_OUT and sel["8x7B C2"] == R4.M_C2_BOARD, sel
    assert combo["pass"][R4.SELECTED]["registered (83 cells)"] == R4.ROUND4_PASS
    assert combo["unpriced_b2"] == [f"{R4.UNPRICED_B2['model']} {R4.UNPRICED_B2['cls']} n{R4.UNPRICED_B2['n']} {R4.UNPRICED_B2['label']}"]
    assert R4.b2_unpriced_identity() == {k: R4.UNPRICED_B2[k] for k in ("model", "cls", "n", "label")}
    cells = R4.registered_cells()
    four = R4.round4_pass()
    assert four == R4.RC.pass_estimates(R4.CAL_HELD_OUT[R4.X22], cells=cells, base=[R4.CAL_HELD_OUT[m] for m in (R4.O, R4.Q, R4.M)])
    assert four == R4.ROUND4_PASS, four
    t4 = R4.round4_table()
    want = {**R4.CAL_HELD_OUT, "mixtral-8x7b C2": R4.M_C2_BOARD}
    assert t4 == want, (t4, want)
    fp = R4.fingerprint()
    assert all(m is not None for m in fp["ms"]) and all(w is not None for w in fp["w2_reads"]), fp
    bounds = {f"X {X:+.2f}": R4.pass_estimates(X) for X in R4.BOUNDS_X}
    sk = stakes()
    w1w2 = [w[0] for w in fp["w2_reads"]]
    w1m4 = [m[0] for m in fp["m4_reads"]]
    lever_sum = round(100 * sum(a - b for a, b in zip(w1m4, w1w2)) / sum(w1m4), 2)
    per = sorted(round(100 * (a - b) / a, 2) for a, b in zip(w1m4, w1w2))
    lever_med = round((per[2] + per[3]) / 2, 2)
    d = {
        "registered": "2026-10-27, before any C4 page",
        "name": NAME,
        "status": ("CALIBRATION: C4 certifies nothing and scores no verdict; every C4 page becomes SEEN once read. The "
                   "bands below say what the fifth held-out model means for the B3 decision and for the one declared "
                   "model round that follows (owner, 2026-10-27); both decisions are the owner's"),
        "design": (f"{DESIGN}; the owner's choice of 2026-10-27: one calibration rental with a fifth non-blind CAL model in "
                   "Phi-3.5's w1 regime (large K, x near 1, P near 100, ROUND4.md's definitions), then ONE declared model "
                   "round. The fifth model is Mixtral-8x7B's tensor-parallel-2 shard, moe/spec.py 'mixtral-8x7b-tp2' "
                   "(H 4096, I 7168, E 8, top_k 2: w1 K 4096, N 14336, P 112, x 0.92 to 0.98 at n 2 with M4's bytes; w2 K 7168, P 32, "
                   "expert w2 0.93 L2), the only repo model in that regime that fits (CANDIDATE.md)"),
        "tool": ("scripts/gh200_cal4_session.sh (C2's pinned driver; no line past its header changed), scripts/tileA2_gate.py "
                 "band (A2's G-BAND), scripts/tile_census_a2.py (the census), scripts/scoring/cal4/hist_c4.py (the tp2 "
                 "page), scripts/scoring/cal4/read_c4.py (the reader)"),
        "seen": ("Every C4 page is SEEN once read: mixtral-8x7b-tp2 joins OLMoE-1B-7B, Qwen2-57B-A14B, Mixtral-8x7B and "
                 "Mixtral-8x22B as a CAL model, and any later fit that uses it (the declared five-model round) is "
                 "SEEN-FITTED on it. tp2 is a TP shard of the CAL model Mixtral-8x7B: its router, E 8, top_k 2 and H 4096 "
                 "are Mixtral-8x7B's (so are its page's routing shapes, and Phi-3.5's, rental 6 owner decision 4); its "
                 "w1 N (14336) and w2 K (7168) are new. No tp2 cell exists at 128x128 or 64x128 before C4 (its uniform and balanced cells have the counts of tp2's R3-tile tread-8 cells, 256 rows per expert, from rentals 1 to 3: SEEN by counts at another tile, never timed and in no round-4 fit); its R3-tile "
                 "pages (rentals 1, 2 and 3, BLOCK_M 32, at 1710 and 1005 MHz) are SEEN, and M4's carried constants were fitted at "
                 "R3's tile on rentals that ran them (SEEN-INFORMED). C4 runs no cell of B3's blind models (Phi-3.5-MoE, "
                 "Qwen1.5-MoE-A2.7B, JetMoE-8B, deepseek-v2-lite; docs/registered/2026-10-20-tileB2-*). OD13 kept: no "
                 "time is priced against rental A's 1710 E1-CAL pages"),
        "frozen": {
            "statement": ("The round-4 model (round3 ROUND4.md's selected variant: the W2 byte law on both GEMMs, the K1 knee "
                          "fitted in the wall domain at its SEEN constants, the clock form of F1, F1 'par') is FROZEN as of "
                          "this registration. The reader refuses unless every file below has its sha256, the frozen model "
                          "reproduces the fingerprint on tp2's six cells, and B2's cell at the registered UNPRICED index is "
                          "round 4's UNPRICED cell; no constant below is refitted, re-chosen or re-read from a C4 page "
                          "before C4-R1 and C4-R2 are read. A change made after C4's pages are seen is SEEN-FITTED on tp2 "
                          "and is disclosed as such. THE FREEZE IS TIED TO THE RENTAL'S COMMIT, as C2's: read_c4.read takes "
                          "the commit from the ledger's 'driver start: commit <sha>' line (gh200-driver/status, checked "
                          "against session/commit.txt) and refuses unless git show <sha>:<path> equals the working copy for "
                          "the registration JSON and text, the tp2 page, the plan and every code_pins path, every code_pins "
                          "path has its registered sha256, and every repo module loaded from the chain's repo (diag-A2 "
                          "common.REPO) and from the reader's repo is a code_pins path with its registered bytes: a repo "
                          "module is a git-tracked file under that repo, outside tests/ and any .venv, venv, site-packages "
                          "or dist-packages directory, with an absolute __file__ (read_c2.repo_modules, C2's rule)"),
            "chain_repo_modules": list(R4.CHAIN_REPO_MODULES),
            "root": "~/code/moe-kernels-work (paths below relative to it); the chain imports the repo at ~/code/moe-kernels",
            "files": R4.frozen_hashes(),
            "constants": {
                "byte law": ("W2 at 128x128 on both GEMMs (round3 r5 LS law, r6 law4('W2')): later-wave cross-group slab "
                             "re-reads at D_eff = W_c dt / L from list scheduling (R 16 replicas, seed 20261027 x 1000 + i), "
                             "sigma_gap 0.0375 (MEASURED-PERTURBED, in-sample for Mixtral-8x7B), the carried soft survival, "
                             "the capacity cut at the 60 MiB L2 (MEASURED); the first window, in-group and A events M4's "
                             "(v3 key, CARRIED); W_c 264 = 132 SMs x 2 CTAs per SM (the C2 and A2 censuses)"),
                "knee": ("K1 at the A2 triad (3726.2 GB/s), p_128 6.0, S0 2.75 k-steps: knee4.json candidates K1 SEEN, the "
                         "wall-domain fit on all four CAL models' 138 NATIVE 128x128 counter rows; tp2 is in none, so the "
                         "SEEN constants ARE its held-out fold"),
                "clock form": ("fac_g = 1 + D_net(t0) / t0 on the floor side, D_net(t) = A (1 - exp(-t / tau)) - 6.11 (1 - "
                               "exp(-t / 50 us)) us, t0 the GEMM's sag-free time (r6.v5_parts_clock)"),
                "F1 estimator": ("'par': per estimator page one (A, tau) by refit4 r4.sag_page_fit (design-v5 sag.fit over "
                                 "sag.pts_of(page, 1410)), A and tau averaged; pages: C4's OLMoE f128 and Mixtral-8x7B f128 "
                                 "(C4-F), both (tp2 has no estimator page to drop)"),
                "A2 board F1 (the fingerprint's)": {"per_page": {m: list(v) for m, v in R4.A2_PAGE_FITS.items()},
                                                    "par": list(R4.A2_PAR_128)},
                "T0, c, bw, p_64, F4, R3's 1410 share": ("43.23 us, 202.548 ns, 3597.8 GB/s (asserted at import), 3.75, rho x 2, "
                                                         "(6.11 us, 50 us): CARRIED as round 4"),
                "selections": "A2's (moe-kernels-work 2026-10-20 scoreA2 via finalise.selections)",
                "scoring rules": "a2common.Rules.tileA2(1410)",
                "registered UNPRICED blind cell": {**R4.UNPRICED_B2, "rule": ("round 4's: excluded from E1-TILE's statistics, "
                                                                              "so every pass estimate here is on 83 cells")},
            },
            "round 4 selected table, re-priced by the frozen code at registration": t4,
            "round 4 pass estimate on 83 cells, reproduced": four,
            "fingerprint": fp,
        },
        "power_and_lock": {
            "power_limit_w": R4.POWER_W,
            "rule": ("C2's prelude, unchanged: nvidia-smi -pl 700 -sc 0, then the read-back (enforced.power.limit and -q -d "
                     "POWER's GPU-scope Current Power Limit both at 700 W within 1 W, polled every 2 s up to 30 s), else the "
                     "session STOPS (c2-power.env C2_POWER_700=refused, the last reading in power-set-unsettled-700.txt). "
                     "No fallback limit"),
            "session_lock_mhz": R4.LOCK,
            "lock_rule": ("every unit but calibrate and the census runs under nvidia-smi -lgc 1410 with ncu --clock-control none; "
                          "the prelude locks 1410, reads clocks.sm back (two tries) and STOPS when it does not hold "
                          "(LOCK_PROBE_1410=held in locks.env)"),
            "g_band": ("A2's G-BAND (tileA2_gate.py band) after C4's two f128 pages: joint_cF over them in the counter band at "
                       "1410 (primary [0.975 L, L], else the fallback [0.959 L, L]) is not None, else the session ends there"),
        },
        "units": {
            "C4-P": {"units": ["- prelude", "mixtral-8x7b calibrate tc4", "- tilecensus tc4"],
                     "for": ("the power read-back and the lock probe (validity); calibrate's ruler, which the timed unit hands "
                             "R3, and the triad K1's rate is read beside (C4-S5); tp2's compiled occupancy at 128x128: 2 CTAs "
                             "per SM on both GEMMs is a validity clause of C4-R1 and C4-R2 (W2 prices with W_c 264)")},
            "C4-F": {"units": ["olmoe-1b-7b bytes f128", "mixtral-8x7b bytes f128", "- gate gband"],
                     "for": ("the sag estimator's two pages ON C4'S BOARD, so tp2's cells are priced as B3 will price blind "
                             "cells; then G-BAND"),
                     "cells": "128x128x64 w8 s3, G 64, treads 1 to 9, NATIVE, SHARED and PRIVATE (A2's f128 pages)"},
            "C4-T": {"units": ["mixtral-8x7b-tp2 timed t128g1"],
                     "for": "C4-R1: tp2's held-out 128 G 1 time, the fifth held-out model",
                     "cells": (f"{HIST}/{R4.PAGE}: n 2 (the only tread whose vLLM default config is 128x128x64 w8 s3 G 1 for "
                               "E 8, top_k 2), PT, PW, PT2, PW2, uniform, balanced, NATIVE and SHARED (NATIVE scored); "
                               "E1-CAL's instrument (R3 through r3_tile.py, 9 declared copies, slip-policy cell, r7-median-run3)")},
            "C4-B": {"units": ["mixtral-8x7b-tp2 bytes b128g1"],
                     "for": ("C4-R2 (the w1 bytes reading), C4-S2 and C4-S3: tp2's counted bytes, cycles, ncu time and sag "
                             "on the same cells"),
                     "cells": "the same file, A2's byte-page instrument at 1410, NATIVE and SHARED"},
        },
        "readings": {
            "C4-R1": {
                "what": "X, tp2's held-out 128 G 1 mean, percent; e = T_model / T_meas - 1 per cell",
                "read_by": "scripts/scoring/cal4/read_c4.py read (timed_rows, sag_fits, held_out)",
                "method": ("cells by read_c2.timed_rows' loop (tcommon.timed_page at G 1 with the registered file entry: G5 "
                           "provenance, the r7-median-run3 rebuild, the ALL view; tcommon.row_drop against v4's ms); each kept "
                           "NATIVE cell priced by r6.cell_ms4(r, K1 at the A2 triad with S0 2.75, p 6.0, F1 par of C4-F, "
                           "r6.reads_of(r, 6.0, 'W2'), clock=True); a cell whose W2 fixed point the guard refuses is "
                           "UNPRICED, listed and left out of X; X = refit4 r4.summ's mean over the priced cells"),
                "table": ("read_c4.table: round 4's selected LOMO held-out 128 G 1 means (combo4.json: "
                          f"{json.dumps(R4.CAL_HELD_OUT)}; Mixtral-8x7B on C2's board {R4.M_C2_BOARD} beside), unchanged by "
                          "C4 (no C4 page enters their folds), and X; max |mean|, spread, rms of means; the round-4 gate's G1 "
                          "on the five (DESCRIPTIVE: it already fails on the four, so C4 cannot pass it)"),
                "pass_estimate": ("read_c4.pass_estimates(X): read_c2.power (diag-v5 power5's, verbatim) on B2's registered "
                                  "NATIVE cells less the registered UNPRICED cell (83), R 200,000, seed 20261022, the knee "
                                  "round's method on five means: A = power(b_class the mean of the five b = -ln(1 + e), sd "
                                  "their SD), B = power(0, their RMS), D = power(the mean of OLMoE, Qwen2-57B, Mixtral-8x22B "
                                  "and X, their SD: Mixtral-8x7B's G 1 set aside). An ESTIMATE"),
                "beside X (DESCRIPTIVE)": ("X at A2's par (C4's board sag against the model's error); X with M4's bytes "
                                           "(the W2 lever's share in time); X with C4-B's counted bytes (oracle); X with K1 at "
                                           "C4's own triad; v5'''s held-out mean on the same cells"),
                "refusals": ("read_c4.read refuses (no reading at all) on a frozen work file, the rental's commit, a code_pins "
                             "sha256, a repo module (checked right after the fingerprint, before any page, and again at the end), the fingerprint, the UNPRICED identity, or the PREFLIGHT: right after the fingerprint and before any page, the frozen code re-prices round 4's selected table (r4.sag_page_fit on A2's and C2's f128 pages, the path C4's F1 takes) and its four-model estimate, and refuses on any difference from the registered values; the timed page is NOT SCORED "
                             "unless tcommon.timed_page admits it and its report records lock_mhz 1410"),
                "validity": (f"NOT READ unless the board reads valid (c2-power.env C2_POWER_W=700, LOCK_PROBE_1410=held, "
                             f"G-BAND PASS), both C4-F pages give a fit, the C4-T page is admitted, the tc4 census compiles "
                             f"tp2 at 2 CTAs per SM on both GEMMs with verdict COMPLETE, and at least {R4.MIN_CELLS} of its 6 NATIVE cells are kept "
                             "and priced"),
                "bands": {"CLEAN": f"|X| <= {R4.BAND['clean']:g}", "HIGH": f"{R4.BAND['clean']:g} < X < {R4.BAND['far']:g}",
                          "HIGH-FAR": f"X >= {R4.BAND['far']:g}", "LOW": f"-{R4.BAND['far']:g} < X < -{R4.BAND['clean']:g}",
                          "LOW-FAR": f"X <= -{R4.BAND['far']:g}", "NOT READ": "the validity clause fails"},
                "means_for_B3": dict(R4.MEANS_FOR_B3),
                "joint (C4-R1 band x C4-R2 band: the registered reading of the two together, for the final declared round)": R4.JOINT,
                "bounds (the five-model estimates at X, round 4's four means held fixed; 83 cells)": bounds,
                "why these cuts": ("1.0 is the gate's own bar on a held-out mean. 2.0 is past both CAL residuals the knee "
                                   "trades (Mixtral-8x7B +1.25 on A2, +1.49 on C2's board; Qwen2-57B -1.44) and near "
                                   "Phi-3.5's 128g1 stake (-2.32 against v5''): a reading past it is larger than any "
                                   "residual the four models leave"),
                "stakes (frozen model against v5'' on tp2's cells at A2's par; NOT a prediction of record)": sk,
            },
            "C4-R2": {
                "what": ("Lambda = sum(counted - W2) / sum(M4 - W2) over C4-B's NATIVE cells' w1 dram_bytes_read: 0 when the "
                         "counters are W2's, 1 when they are M4's. The W2 lever (sum(M4 - W2) / sum(M4)) is beside. Phi-3.5's "
                         "bytes lever, held out for the first time"),
                "read_by": "scripts/scoring/cal4/read_c4.py bytes_reading (counter_cells, lambda_of)",
                "method": ("counted reads by kneefit.cells_of (refit4 r4.counter_cells: NATIVE, BLOCK_M 128); W2's and M4's "
                           "reads by r6.reads_of(cell, 6.0, 'W2') and r6.reads_of(cell, 6.0, None) on the same counts and "
                           "shuffle seed; the same quantities on w2 beside (DESCRIPTIVE)"),
                "validity": (f"NOT READ unless the board reads valid, the page is the registered tile and file (read_c2."
                             f"check_counter_page), the census reads 2 CTAs per SM with verdict COMPLETE, at least {R4.MIN_CELLS} NATIVE cells are "
                             f"priced under both laws, and the W2 lever is at least {R4.MIN_LEVER:g} (at registration, on the page's six cells: "
                             f"{lever_sum}% on the sum-based measure the cut tests, {lever_med}% per-cell median)"),
                "bands": {"W2-SIDE": "-1/3 <= Lambda <= 1/3", "BETWEEN": "1/3 < Lambda < 2/3", "M4-SIDE": "2/3 <= Lambda <= 4/3",
                          "BEYOND-W2": "Lambda < -1/3", "BEYOND-M4": "Lambda > 4/3", "NOT READ": "the validity clause fails"},
                "means": dict(R4.BYTES_MEANS),
                "why these cuts": ("thirds of the distance between the two laws: a reading is W2's or M4's when the counters "
                                   f"sit within a third of it from that law, so the lever ({lever_sum}% at registration, "
                                   "against the byte-page noise B2 assumes, 1.75% x s) is resolved three ways"),
            },
            "C4-S1": {"what": "tp2's x (bytes-only / floor-only) per GEMM on its kept cells, with M4's bytes and with W2's",
                      "read_by": "scripts/scoring/cal4/read_c4.py x_of_rows",
                      "expected (registration, INFERENCE)": sk["x"]},
            "C4-S2": {"what": ("tp2 b128g1: K1 at its SEEN constants on counted bytes per GEMM in the wall domain (knee4.py's "
                               "target and kappa_row) and in the cycles domain beside, and the per-cell w1 + w2 sum: does the "
                               "knee carry X?"),
                      "read_by": "scripts/scoring/cal4/read_c4.py knee_reading (k5.capture_rows, k5.gemm_form)"},
            "C4-S3": {"what": "tp2's own sag on its b128g1 page: C4's par estimator minus the measured off, us; its own fit",
                      "read_by": "scripts/scoring/cal4/read_c4.py sag_domain"},
            "C4-S5": {"what": ("the board: C4's per-page (A, tau) beside A2's and C2's, the power read-back, the lock probe, "
                               "G-BAND; tp2's compiled CTAs per SM at 128x128 beside the 2 per SM the frozen pricing uses; "
                               "calibrate's triad beside A2's 3726.2 GB/s (15 rentals 3725.6 to 3727.6)"),
                      "read_by": "scripts/scoring/cal4/read_c4.py census, triad (read_c2.board)"},
        },
        "after": (f"OWNER DECISIONS 2026-10-27 (registered): {R4.SEQUENCE}. The round-4 gate is unchanged for that final "
                  "round; G1 covers the five CAL models when C4 is read, the four otherwise"),
        "plan": {"file": PLAN, "sha256": sha256(REPO / PLAN), "units_count": len(units), "units": units,
                 "est_total_min": est, "est_nodrop_min": nodrop, "drop_order_last_first": order,
                 "boot_setup_min": BOOT, "wall_min": wall, "rate_usd_per_h": RATE, "cost_usd": round(wall / 60 * RATE, 2),
                 "deadline_after_launch_min": DEADLINE, "hard_cap_after_launch_min": CAP,
                 "cost_at_hard_cap_usd": round(CAP / 60 * RATE, 2)},
        "pages": {"files": {R4.PAGE: file_entry(f"{HIST}/{R4.PAGE}")},
                  "generator": f"scripts/scoring/cal4/hist_c4.py ({', '.join(HC.PAGES)})"},
    }
    d["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    return d


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: the calibration rental C4, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", "", f"SEEN: {d['seen']}", "", "FROZEN", f"  {d['frozen']['statement']}",
         f"  root: {d['frozen']['root']}"]
    for k, v in d["frozen"]["files"].items():
        L.append(f"  {k} {v}")
    for k, v in d["frozen"]["constants"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
    L.append(f"  round 4 selected table, re-priced by the frozen code at registration: {json.dumps(d['frozen']['round 4 selected table, re-priced by the frozen code at registration'])}")
    L.append(f"  round 4 pass estimate on 83 cells, reproduced: {json.dumps(d['frozen']['round 4 pass estimate on 83 cells, reproduced'])}")
    fp = d["frozen"]["fingerprint"]
    L.append("  fingerprint (frozen model ms at A2's par; W2 reads w1, w2; M4 reads w1, w2): " + "; ".join(
        f"{c} {a} {b} {m}" for c, a, b, m in zip(fp["cells"], fp["ms"], fp["w2_reads"], fp["m4_reads"])))
    L += ["", "POWER AND LOCK"] + [f"  {k}: {v}" for k, v in d["power_and_lock"].items()]
    L += ["", "UNITS"]
    for k, v in d["units"].items():
        L.append(f"  {k}: {json.dumps(v['units'])}: {v['for']}" + (f" [{v['cells']}]" if "cells" in v else ""))
    L += ["", "READINGS"]
    for k, v in d["readings"].items():
        L.append(f"  {k}:")
        for kk, vv in v.items():
            L.append(f"    {kk}: {vv if isinstance(vv, str) else json.dumps(vv)}")
    L += ["", f"AFTER: {d['after']}"]
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
