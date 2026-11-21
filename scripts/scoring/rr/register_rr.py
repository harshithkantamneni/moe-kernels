#!/usr/bin/env python
"""Writes rental RR's registration, docs/registered/2026-11-11-rr-gh200.{json,txt}, and its plan, before any RR page.

    python scripts/scoring/rr/register_rr.py --write [--out-dir docs/registered]
    python scripts/scoring/rr/register_rr.py --check      # recompute, compare (exit 1 on any difference)

RR is a MEASUREMENT rental (owner, 2026-11-11): prefill gets this one measurement rental and no blind, then closes; RR
measures the w1 re-read L2 hit mechanism and publishes it as a DESCRIPTIVE mechanism. It registers no model form, no
constant, no gate and no blind. What it registers before any page: the units and cells (the pages and shapes, by
sha256), the stop rules, the validity and row-sanity rules, what each reading is and how the reader computes it
(scripts/scoring/rr/read_rr.py, by function), the four forms of the per-pair hit rule read on the pages (each constant
stated as found post-hoc on seen pages), the plan and its figures, the launcher's change (flagged for the owner) and the
code by sha256. `--check` recomputes everything and compares the JSON (code_pins excluded, as every registration since
rental 3), the text, the plan and the generated pages. Every write is atomic (a temporary file, then os.replace).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(HERE.parent / "pc2"), str(HERE.parent / "pc"), str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import gen_rr_pages as GEN  # noqa: E402
import register_pc2 as RPC2  # noqa: E402
import rrcommon as P  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = P.NAME
DESIGN = ("moe-kernels-work 2026-11-11 design-P1/DESIGN.md section 4 (the measurement-rental sketch) and section 2 (the "
          "per-pair rule and the wave-start spread, POST-HOC on SEEN pages), read-only over the repo at 019ace6a; built "
          "in moe-kernels-work 2026-11-11 build-RR")
#: every repo file RR's driver, launcher, probe, counter route and reader run on the VM or the laptop: PC2's pins (all
#: byte-identical, PC2's driver, probe, launcher and reader among them) and RR's own files
PINNED = tuple(RPC2.PINNED) + (
    P.DRIVER, P.PROBE, P.LAUNCHER, P.COUNTER, "scripts/scoring/rr/rrcommon.py", "scripts/scoring/rr/gen_rr_pages.py",
    "scripts/scoring/rr/register_rr.py", "scripts/scoring/rr/read_rr.py",
    # the pinned counter route's own imports, run on the VM by RR's counter units (unchanged files)
    "moe/bench/counter_probe_kernel.py", "scripts/lock_gate.py",
)
#: the registrations whose files RR reads, imports or copies and keeps byte-identical (their --check must read same)
EARLIER = tuple(RPC2.EARLIER) + ("2026-11-10-pc2-gh200",)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_units() -> list[dict]:
    """The plan's units as the driver parses them (PC's parser on RR's plan text)."""
    out = []
    for line in P.plan_text().splitlines():
        body = line.split("#", 1)[0].split()
        if not body:
            continue
        kv = dict(w.split("=", 1) for w in body[2:])
        drop = "nodrop" if (body[1] == "prelude" or kv.get("nodrop") == "1") else kv.get("drop-group")
        keys = {k: v for k, v in kv.items() if k not in ("label", "est", "cap", "nodrop", "drop-group", "depends")}
        out.append({"model": body[0], "step": body[1], "label": kv.get("label", ""), "est": int(kv["est"]),
                    "cap": int(kv["cap"]), "drop": drop, "depends": kv.get("depends"), "keys": keys})
    return out


def pages() -> dict:
    out = {}
    for rel, text in sorted(GEN.files().items()):
        if rel == P.SHAPES_FILE:
            continue
        doc = json.loads(text)
        out[rel] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "schema": doc["schema"],
                    "cells": [c["label"] for c in doc["cells"]], "lock_mhz": doc["lock_mhz"], "tile": doc["tile"]}
    out[P.PHI_128] = {"sha256": sha256(REPO / P.PHI_128), "schema": "moe-kernels/histogram-page/1",
                      "cells": "the replicate's page (tile rental B's Phi-3.5 128g1, 2026-10-14), unchanged",
                      "lock_mhz": P.SESSION_LOCK, "tile": "128x128x64w8s3"}
    return out


def cells_table() -> dict:
    """Per RR page: each cell's series, P, layout, shape, T, tile, the M-tiles per expert, W_c / P, the re-read pairs,
    F_stag, the w1 waves, live CTAs per GEMM, C_u and R, stamped calls (COMPUTED)."""
    sh = P.load_shapes()
    out = {}
    for pg in P.BLOCKS:
        rows = []
        for c in P.check_page_doc(P.page_doc(pg), sh):
            g = P.geometry(c["cfg"], c["counts"], c["tile"])
            live = {x: P.live_ctas(c["cfg"], c["counts"], c["tile"], x) for x in ("w1", "w2")}
            rows.append({"cell": c["label"], "series": c["series"], "P": c["P"], "layout": c["layout"],
                         "role": c["role"], "shape": c["shape"], "T": c["T"], "tile": P.tile_key(c["tile"]),
                         "m_e": g["m_e"], "W_c_over_P": round(g["W_c_over_P"], 4), "pairs": g["pairs"],
                         "F_stag_MiB": g["F_stag_MiB"], "waves_w1": round(g["waves_w1"], 3), "live": live,
                         "C_u_bytes": g["C_u_bytes"], "R_bytes": g["R_bytes"],
                         "stamped_calls": P.stamp_calls(live["w1"])})
        out[pg] = rows
    return out


def readings() -> dict:
    forms = "; ".join(f"{k} ({v[3]})" for k, v in P.FORMS.items())
    return {
        "every reading": (
            "DESCRIPTIVE: computed from RR's pages by the rule below (read_rr.py), registered before any RR page; none is a "
            "model term, a constant priced into anything, a STOP, a bar, a gate or a verdict, and no blind follows (owner, "
            "2026-11-11). A reading that does not clear its rule reads NOT READ or NOT IDENTIFIED with the reason; nothing "
            "falls back and nothing reads 0 by default"),
        "STATED PLAINLY": (
            "the 42.5 MiB constant and the four forms below were FOUND POST-HOC ON SEEN PAGES (design-P1 DESIGN.md "
            "section 2.3, work/fit_s.out: stamps of C3's Mixtral-8x7B and Qwen2-57B and PC's Phi-3.5 against counted bytes "
            "of seven boards; the Phi split and PC-R10). RR's purpose is to MEASURE them on FRESH synthetic cells (64 of "
            "its 80 cells; the 16 at P 100 and 224 carry seen models' w1 GEMMs and are read apart as anchors)"),
        "RR-R0 (validity; read_rr.stamp_unit, read_rr.counter_unit, read_rr.board)": (
            "stamps units (PC2-R0, unchanged in substance, on RR's geometry): per launch read_pc.launch's legs and ROW "
            "SANITY (every row stamped live or dead; the live CTAs exactly the pids under (ntpp / BLOCK_M) x ceil(N / "
            "BLOCK_N) and equal to the cell's sum over experts of ceil(c_e / BLOCK_M) x ceil(N / BLOCK_N); USE_TD False; "
            "no late buffer; the first w2 start after the last w1 end of the call; every live start and end positive, end "
            f"after start, clock64 life positive, the per-CTA clock in (0, {P.F_CEILING_MHZ:g}] MHz, every per-CTA rate at "
            f"most {P.RATE_CEILING_GBPS:g} GB/s; no slab is read once on an RR cell, so neither the X ceiling nor the L2 "
            "guard applies, as on C3's and Phi's re-read cells); per unit the gate's HARD line and the variant's regcheck "
            "line PASS (a timing FAIL alone reads PERTURBED, labelled), the page the registered file (sha256), the "
            "versions (python on major.minor; torch, triton, vllm exact), every stamped compile passing C3's SASS legs, "
            "the page COMPLETE (a PARTIAL page is REFUSED), the unit's lock and its perturb unit's lock read back after "
            "the tool (pc-locks.env), EVERY stamped compile's CTAs per SM equal to its tile's registered occupancy, and the "
            "census reading that occupancy (W_c and every gap need it); EXACT TIME and the unit's timer as PC2's. Counter "
            "units: the page present (lock1410/r3d.json), its schema, the counted page the registered file (sha256), the "
            "versions, every fused_moe_kernel launch attributed (the pinned route's attribute_launches: grid = the child's "
            "readback), the page complete, counted at the session lock, and the unit's lock read back after the tool. The "
            f"board: C2_POWER_W={P.POWER_W} and LOCK_PROBE_{P.SESSION_LOCK}=held. A failing launch, cell or unit is NOT "
            "READ, never zero"),
        "RR-R1 the counted miss fraction per cell (read_rr.counter_cell; rrcommon.geometry, counted_m)": (
            "D = the median over the cell's counted calls of ncu's w1 dram__bytes_read.sum (cold L2 per launch, the "
            f"session lock), READ with all {P.COUNT_CALLS} calls counted and their spread (max - min over the median) at "
            f"most {P.COUNT_SPREAD_MAX:g}; C_u = each active expert's w1 once plus the unique tokens' A rows once (T x H x "
            "2; design-P1 P1-F4); R = the re-read demand, sum over experts of (m_e - 1) x the expert's w1 bytes; m = (D - "
            f"C_u) / R; counter sanity: D within [{P.COUNT_LO} C_u, {P.COUNT_HI} (C_u + R)], else NOT READ. Printed beside: "
            "D / C_u - 1, w2's D against its compulsory bytes (each active expert's w2 once plus the intermediate rows "
            "once), ncu's w1 L2 read hit rate where proven, and the ncu clock (sm__cycles_elapsed.avg over "
            "gpu__time_duration.sum; ncu's clock under the 1410 lock read 1304 to 1350 MHz on B3's board, printed, never "
            "a refusal here)"),
        "RR-R2 the per-pair hit rule (read_rr.pair_read, cell_stamps, join, pooled, refit)": (
            "per valid w1 launch: C3-R2's pairs (rank r = pid_m P + pid_n whose M-tile pid_m - 1 holds the same expert; "
            "the only w1 slab re-read at G 1); per pair d_start, the B bytes streamed GPU-wide between the partners' "
            "starts (each CTA's slab K x its BLOCK_N columns x 2 spread evenly over its life: PC-R10's integral) and d_mid, "
            "the same between their mid-lives, on exact stamps; each form's m_hat = the mean over pairs of its miss "
            "indicator (step) or probability (exp); per cell (READ with at least "
            f"{P.MIN_VALID_CALLS} readable w1 launches, the cell's w1 overhead within C3's OD-1 bound and the unit valid) "
            "the median over calls, per-call values printed. THE FORMS (registered, constants POST-HOC on SEEN pages): "
            f"{forms}. Per cell read on BOTH the stamps and the counter unit of its page (same cells, same board): the "
            "residual m_hat - m per form and the w1 byte residual (m_hat - m) R / D. Pooled per tile, both tiles, the FRESH "
            "cells (P not 100 or 224), the seen-anchor cells, and per series: n, rms, mean, largest residual, rms w1 byte "
            "residual, and which form has the lowest rms (DESCRIPTIVE). The refit (DESCRIPTIVE, printed beside, never a "
            "registered constant): per (distance, law) in (starts, step), (mid-lives, step), (starts, exp) the constant "
            f"on the grid {P.REFIT_GRID_MIB[0]:g} to {P.REFIT_GRID_MIB[-1]:g} MiB by 2.5 minimising rms(m_hat - m) over "
            "RR's READ cells (m_hat the mean over a cell's calls), ties to the smallest, on all cells and on the fresh ones. "
            "Also printed per cell: the together share (gap g < 0.5 of P L / W_c, C3's rule), the gap median, d's 10th, "
            "50th and 90th percentiles, and the stamps unit's page label (CLEAN or PERTURBED, the perturb unit's gate). "
            "THE CLOCK: the stamps and the counted bytes share the 1410 MHz lock, not the SM clock; the counted run's SM "
            "clock under ncu differs from the stamps' 1410 MHz (B3's board read 1304 to 1350 MHz) and is printed per cell "
            "in RR-R1; the gaps (in bytes) are read off the stamps only"),
        "RR-R3 the wave-start spread against wave index (read_rr.wave_spreads, wave_numbers, pooled_curve, wave_reading)": (
            f"NUMBERS ONLY (owner, 2026-11-11). Per valid w1 launch and full schedule wave k (pids k W_c to (k + 1) W_c - 1, "
            f"all live; W_c = 132 x the registered occupancy): (q{int(100 * P.WAVE_Q[1])} - q{int(100 * P.WAVE_Q[0])} of "
            "the wave's starts) / the launch's median life; per cell the median over calls per k; pooled curves (the median "
            f"over READ cells with wave k read, at least {P.POOL_MIN_CELLS} cells) per tile, per series (A, B, C) and per "
            f"W_c / P group (edges {list(P.WCP_EDGES)}); for each cell and each pooled curve three numbers: w_inf = the "
            f"median spread over k >= {P.K_FLAT} (at least {P.FLAT_MIN} such waves, else NOT IDENTIFIED), k_90 = the first "
            f"wave whose spread reaches 0.9 w_inf, and the OLS slope of the spread over k >= {P.K_FLAT}, per wave. No "
            "label and no band. The stand-in's numbers on SEEN pages (C3's Mixtral-8x7B and Qwen2-57B stamps, PC's Phi-3.5) "
            "are printed beside, frozen at registration ('stand-in wave numbers (SEEN pages, frozen)'). DISCLOSED: an "
            "earlier draft of this registration labelled the pooled curve AS FOUND or DIFFERENT against a band (w_inf "
            "0.45 to 0.75, k_90 at most 5, |slope| at most 0.01 per wave); the band was set with PC's published pages in "
            "view, and it was removed on the owner's decision (2026-11-11) before any RR page. Printed beside: the life "
            "spread past the first wave ((q84 - q16) / 2 over the median life) and the re-reading CTAs' median life on the "
            "primary form's misses over its hits (design-P1's two-valued 64x128 lives, INFERENCE there)"),
        "RR-R4 the replicate (read_rr.replicate, rep_counter_unit; units sr and br, droppable; kept by the owner)": (
            "Phi-3.5's 128g1 cells on this board: stamps (sr, RR's probe on the registered histogram page, RR-R0's stamps "
            "legs) and counted bytes (br, the existing counter tool, NATIVE) under RR-R1's rules: the unit's page present "
            "(lock1410/r3c-g1.json), the registered histogram page (sha256), the tile, the NATIVE arm, the versions (the "
            "page's stack), counted at the session lock (the route's --page-lock-mhz; the driver read the lock back before "
            "the page, and the pinned bytes step records no read-back after its tool: stated, the one leg br lacks against "
            "the counter units), the page's status line exit 0 or 1, the board; per cell every call counted, the 3% "
            "spread and counter sanity. An invalid unit reads no number. Then m, each form's m_hat and the residuals; a "
            "cross-board anchor to PC's (c9f689) and B3's Phi pages; DESCRIPTIVE"),
        "STAND-IN (read_rr.standin; on SEEN published pages)": (
            "both boards' stamps, as design-P1 read them: C3's Mixtral-8x7B (st7) and Qwen2-57B (stq), PC's Phi-3.5 (se1 "
            "128x128, se2 64x128); each stamped cell's form m_hat against the counted m of every board holding the cell "
            "(design-P1's 18 counted pages, its 54 (cell, board) rows), pooled, Phi-3.5 only, by family and on one board "
            "(PC's Phi); design-P1's figures beside (pooled rms 0.048 for F1, 0.043 for F2; Phi-3.5 only 0.045 and 0.035); "
            "RR-R3's numbers per stamped family; RR-R3 on PC2's multi-wave w1 launches. These are the pages the forms were "
            "found on: a check of the code, never an RR reading"),
    }


def stop_rules() -> dict:
    return {
        "prelude (C3's, unchanged)": (
            f"the session STOPS before any page unless the checkout is clean, sudo works, the GPU is idle, {P.POWER_W} W "
            "reads back on both enforced.power.limit and -q's GPU-scope Current Power Limit within 1 W (polled up to 30 s; "
            f"no fallback), persistence is Enabled and the session lock {P.SESSION_LOCK} MHz reads back"),
        "unit locks (PC's, unchanged)": (
            f"every unit runs at the session lock {P.SESSION_LOCK} MHz (B3's setting; no unit names another); each probe "
            "and counter tool locks, reads back (two tries) and runs, and the lock is read back again after the tool "
            "(pc-locks.env); the reader refuses a stamps or counter unit whose lock, or whose perturb unit's lock, did not "
            "read back after its tool"),
        "the regcheck drop rule (PC's owner decision 2, kept)": (
            "the probe=pc regcheck (unit 3) exiting anything but 0 or 1 (refused, crashed, stopped at its cap, skipped) or "
            "writing no regcheck line for a probe=pc stamps unit drops every probe=pc unit (4, 5, 7, 9); the counter units "
            "(6, 8) and the replicate's bytes (10) still run, so a failed start still counts every cell's bytes (RR-R1), "
            "with no pair reading. Exit 1 (a variant's occupancy or path FAIL) refuses that variant's stamps unit by its "
            "line. A cell failing its readback is excluded alone. The probe exits 4 on any uncaught error, never 0 or 1"),
        "the gate and the stamps (C3's)": (
            "a stamps unit runs only when its regcheck line and the gate's HARD line (occupancy, the all-off SASS, the "
            "upstream file, C3's SASS legs, every cell's readback) read PASS; a timing FAIL alone runs it PERTURBED"),
        "the counter units": (
            "a counter unit's dry run must exit 2 with no REFUSED: line (the page checked as the probe checks it), else the "
            "unit is REFUSED; a cell whose readback (V-BIN, V-ALIGN) fails stops the child before any measured call and "
            "the page is not written (INVALID); an attribution failure writes no page (INVALID)"),
        "budget": ("the driver drops the LAST droppable unit first: the replicate (rep: br, then sr); a unit whose "
                   "estimate no longer fits is SKIPPED; the guardian terminates at the hard cap"),
        "the deadline and the upgrade step's budget (review MEDIUM 3)": (
            "the deadline stays launch-based (launch + the plan's figure); prepare's driver upgrade has ONE total budget "
            "of 600 s (vm_run_rr.sh, MOE_VM_APT_BUDGET_S) for the lock wait and both install attempts, and prepare refuses "
            "when it is spent (the instance is terminated, never started). With the instance's boot (about 5 to 8 min), "
            "the reboot (about 3 min) and setup_vm.sh (about 5 min) the time before the session's first unit stays within "
            "the plan's 25 min boot allowance; the nodrop core (35 min) and the driver's 8 min reserve and 5 min slack fit "
            "until that time reaches the deadline less 48 min (37 min at the registered deadline), so a slow lock cannot "
            "skip a nodrop unit: it ends prepare first"),
        "never": "no RR reading is a STOP, a gate or a verdict; a reading that does not identify reads NOT IDENTIFIED",
    }


def seen_block() -> dict:
    return {
        "reserved (owner decision, 2026-11-06, kept)": {
            "models": list(P.RESERVED),
            "rule": ("never loaded, never timed, never shaped: the probe and the counter route refuse a cell or weight "
                     "build naming one, a shape equal to one, or any shape with a GEMM whose (N, K) equals either GEMM of a "
                     "reserved model (PC2's gemm_guard, unchanged); tests/test_instr_probe_rr.py asserts nothing in the "
                     "plan loads them")},
        "loaded by name (both SEEN)": list(P.REAL_ALLOWED),
        "synthetic": "the registered rrs- shapes (series A E 8 H 4096, series B E 8 H 8192, series C E 64 k 8)",
        "blind candidates (owner, 2026-11-11: no blind follows, so keeping them unspent no longer matters)": (
            "RR's GEMMs avoid every design-P1 candidate's GEMM anyway (P 28 and 56 at K 4096 are not in the ladder; "
            "series B and C sit off deepseek-v3's (4096, 7168) and (7168, 2048)); a printed fact checked by a test, "
            "never a refusal"),
        "skew layouts": ("count vectors only, from SEEN published pages: Mixtral-8x7B-tp2's PT (C4) and Qwen2-57B's PT "
                         "(A2), rescaled; Qwen2-57B's shape, GEMMs and weights are never used"),
        "statement": ("every RR cell's routing is synthetic; the only seen GEMMs are the anchors' (P 100 is Phi-3.5's w1 "
                      "GEMM at E 8, P 224 is Mixtral-8x7B's shape) and the replicate's Phi-3.5 page"),
    }


def owner_decisions() -> list[str]:
    return [
        "prefill gets this one measurement rental and NO blind; prefill then closes and the project moves to the write-up",
        "RR measures the w1 re-read L2 hit mechanism and publishes it as a DESCRIPTIVE mechanism: no model form is "
        "registered, there is no gate and there is no blind",
        "keeping blind candidates unspent no longer matters; still, no model is loaded that does not need loading",
        "measurement only; the owner writes kernels: no GPU kernel is written or edited, the instrumented copy carries "
        "only C3's existing stamps (ends), every counted byte comes from the existing counter tooling, every new line is "
        "host-side",
        "the apt-lock launcher fix is ADOPTED (vm_run_rr.sh's driver-upgrade step)",
        "the Phi-3.5 128g1 replicate is KEPT (droppable, cut first)",
        "the wave-start spread reading is NUMBERS ONLY: no label and no band",
    ]


def build_decisions() -> list[str]:
    return [
        "the design's sketch (design-P1 section 4) as built: series A (E 8, H 4096, P 20 to 224; balanced m_e 2, 4, 8 and "
        "one skew), series B (K 8192; P 24, 40; m_e 2, 4), series C (E 64; P 36, 64; m_e 2 and one skew), both tiles: 80 "
        "cells, stamps and counted bytes on the same cells and board. Series C takes k 8 (as deepseek-v3): at k 2, E 64 "
        "with two full M-tiles per expert needs T 8192, past the probe's 4096 input rows",
        "the skew layouts are count vectors of SEEN published pages, rescaled by the largest remainder: Mixtral-8x7B-tp2's "
        "PT n 2 (C4's counter page) to 4 x BLOCK_M x 8 rows at E 8, Qwen2-57B's PT n 1 (A2's counter page) to 2 x BLOCK_M "
        "x 64 rows at E 64 (design-P1's own choice of layouts, cand.py)",
        "series A's P 224 cell at E 8 IS Mixtral-8x7B's shape (E 8, H 4096, F 14336, k 2): it runs by that name (a SEEN "
        "model, private random weights at its shape), since a synthetic shape may not equal a model's",
        "the counted bytes come from a new wrapper, scripts/dram_counter_route_rr.py, over the pinned counter route's "
        "functions (probe_ncu, r3_ncu_argv, _r3_capture, _r3_reduce, parse_ncu_csv, attribute_launches), as the decode "
        "route's wrappers do: the pinned route takes a model by name and R3's arms, and RR's cells are synthetic shapes. "
        "Its child builds every cell with RR's probe's own Card, seed and ids, runs the readback (the first warm-up), "
        "then ncu skips the warm-ups and keeps exactly the measured launches (R3's pages' --launch-skip / --launch-count "
        "rule), every shape's weights resident (about 20 GiB on a page)",
        "the driver runs a counter unit as PC's decodebytes step with kernels=rr (lines marked RR); its plan line takes "
        "'-' as its model (the page names its cells' shapes); the unit locks and pc-locks.env are PC's",
        "every unit runs at the 1410 MHz session lock (B3's setting for the prefill tiles; PC's Phi units ran there): the "
        "stamps and the counted bytes share one lock. ncu's own clock under that lock is printed (B3's read 1304 to 1350)",
        "no G-BAND gate (PC ran A2's G-BAND before its Phi pages because they were read against B3's priced chain; RR "
        "prices nothing), no f128 pages",
        "the replicate (Phi-3.5 128g1, stamps through RR's probe on the registered histogram page and NATIVE counted bytes "
        "through the existing tool, read under RR-R1's rules) is the only droppable group, kept by the owner: about 8 min "
        "(the bytes unit's estimate is 6 min, PC's b128g1 page with both arms took 5 min); without it the plan is 35 min "
        "of units, about 60 min wall, $2.29",
        "not built: design-P1 item 6, the align slope by numel (the per-call risk the owner declined to model); the "
        "L-TABLE / L-SCHED / L-SYNC launch-layer fit (P1-F5): no model form is registered, so the gaps are printed as "
        "measured (RR-R2's together share and d percentiles, RR-R3's spreads) and no launch law is fitted",
        "PC2's driver, probe and launcher are pinned by PC2's registration, so RR runs on copies: scripts/gh200_rr_session.sh "
        "(PC2's driver), scripts/instr_probe_rr.py (PC2's probe; its rules module is rrcommon) and scripts/vm_run_rr.sh "
        "(vm_run_pc2.sh), each changed only on lines marked RR and held to its parent by a test",
        "THE LAUNCHER'S DRIVER-UPGRADE STEP (ADOPTED BY THE OWNER, 2026-11-11; lines marked RR in scripts/vm_run_rr.sh): "
        "another session's Lambda H100 hung on the VM's package lock in this step. The pinned step stops and disables apt's "
        "timers only (an apt-daily or apt-daily-upgrade service already running keeps the lock), runs apt-get over ssh "
        "with no wall-clock bound and the laptop's stdin (a dpkg conffile prompt would wait on it), and waits up to 600 s "
        "per install on the lock (two installs). RR's copy gives the WHOLE step ONE budget of 600 s: it waits for the "
        "package locks (no apt, apt-get, dpkg or unattended-upgrade process by name, no holder of dpkg's or apt's lock "
        "files; a VM without fuser reads as held, fail closed) naming the holder, and refuses when the budget is spent "
        "without killing anything; each install attempt (the pinned build, then the newest 580) runs as ONE command under "
        "`timeout --kill-after=30 <the budget left>` with stdin /dev/null, --force-confdef, --force-confold and "
        "NEEDRESTART_MODE=a, and starts with `dpkg --force-confdef --force-confold --configure -a`. THE BOUND KILLS DPKG: "
        "timeout signals the attempt's whole process group (apt-get and the dpkg under it, TERM then KILL 30 s later), "
        "which can leave packages half-configured. RECOVERY: the next attempt's dpkg --configure -a finishes that "
        "configuration before it installs; when the budget is spent, prepare refuses and the instance is terminated, "
        "never started or reused, so no session runs on a box whose install was killed. Tests run the step through stub "
        "apt-get, dpkg, pgrep, fuser, sudo and timeout tools: a process holding the lock, a lock-file holder, a "
        "non-matching process name, a lock never freed, no fuser, an interrupted dpkg left before the step, a hanging "
        "install killed with its process group (its child dead, dpkg left interrupted, prepare refusing), a prompt",
        "the probe and the counter route write stamps.json, the stamp arrays, the PTX, the regcheck and gate files, the "
        "manifest and the counted page atomically; the page generator and the registration write atomically too",
    ]


def build() -> dict:
    units = plan_units()
    r = P.rental()
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    assert tuple(order) == P.CUT_ORDER, order
    shapes = P.shapes_doc()
    return {
        "registered": f"{P.DATE}, before any RR page",
        "name": NAME,
        "status": ("MEASUREMENT, DESCRIPTIVE: RR certifies nothing, registers no model form, no constant, no bar, no gate "
                   "and no blind, and moves no registered verdict. Every reading is DESCRIPTIVE"),
        "design": DESIGN,
        "motivation": (
            "design-P1 (2026-11-11) identified a per-pair hit rule on SEEN stamps (a w1 slab re-read hits L2 when fewer "
            "than about 42.5 MiB of B stream between its partner's start and its own; rms 0.048 in m on four stamped "
            "families) and found the wave-start spread rising to about 0.6 of a CTA life within 3 to 5 waves, then flat; "
            "no existing page has stamps and counted bytes on the same cells and board, and every constant is POST-HOC. "
            "RR measures both on fresh synthetic cells at vLLM's prefill tiles, so the write-up can state the mechanism "
            "as measured"),
        "tool": ("scripts/gh200_rr_session.sh (PC2's pinned driver plus lines marked RR), scripts/vm_run_rr.sh "
                 "(vm_run_pc2.sh plus lines marked RR), scripts/instr_probe_rr.py (PC2's pinned probe plus lines marked RR), "
                 "scripts/dram_counter_route_rr.py (the counter wrapper), moe/instrumented/fused_moe_instr.py (the copy, "
                 "unchanged, STAMPS 3), scripts/tile_census_a2.py (the census), scripts/dram_counter_route_tile.py (the "
                 "replicate's bytes), scripts/scoring/rr/read_rr.py (the reader, over read_pc.py's and read_pc2.py's "
                 "pinned functions)"),
        "owner_decisions (2026-11-11)": owner_decisions(),
        "build decisions (to design-P1's sketch, before any page)": build_decisions(),
        "power_and_lock": {"power_limit_w": P.POWER_W, "session_lock_mhz": P.SESSION_LOCK,
                           "every unit": f"at the session lock {P.SESSION_LOCK} MHz"},
        "instrument": {
            "copy": "moe/instrumented/fused_moe_instr.py, unchanged (no kernel written or edited; no new stamp point; no hint)",
            "copy_sha256": sha256(REPO / "moe/instrumented/fused_moe_instr.py"),
            "spec": "stamps=ends,evict_a=none,evict_b=none",
            "probe": ("scripts/instr_probe_rr.py: PC2's probe with rrcommon as its rules module; imports instr_probe_c3.py "
                      "and instr_probe.py (both byte-identical)"),
            "per cell": ("PC's: the readback (V-BIN, V-ALIGN at the cell's config), 3 warm-up calls, then 3 stamped calls, "
                         "each with its buffers reserved, the GPU held ahead and a 240 MiB L2 flush outside the launch "
                         "event pair"),
            "counter": ("scripts/dram_counter_route_rr.py: per cell the readback and 1 more warm-up (skipped by ncu), then "
                        "3 measured calls (both GEMMs), cold L2 per launch, the plain kernel, the same weights and ids as "
                        "the probe's"),
            "weights": "one NATIVE copy per shape, private random weights at the shape"},
        "units": {"plan": P.PLAN, "rows": units},
        "cells (COMPUTED)": cells_table(),
        "pages": pages(),
        "shapes": {"file": P.SHAPES_FILE, "sha256": hashlib.sha256(P.shapes_text().encode()).hexdigest(),
                   "shapes": shapes["shapes"]},
        "readings": readings(),
        "forms (registered; constants POST-HOC on SEEN pages)": {
            k: {"distance": v[0], "law": v[1], "C_mib": v[2], "origin": v[3]} for k, v in P.FORMS.items()},
        "stop rules": stop_rules(),
        "census and occupancy": {
            "unit": "unit 2: tile_census_a2.py over RR's 2 configs on Mixtral-8x7B and Phi-3.5 (compile only)",
            "registered occupancy": P.OCCUPANCY,
            "regcheck scope": ("one specialisation per tile per variant is regchecked (the probe's rule, PC's); the "
                               "reader's occupancy leg checks every stamped compile and fails closed"),
            "rule": ("every reading needs W_c = 132 x the registered occupancy: a census off it makes the tile's stamps "
                     "unit NOT READ; every stamped compile's CTAs per SM must equal the registered occupancy (RR-R0)")},
        "page identity and versions": {
            "pages": ("each stamps manifest's source_sha256 and each counted page's page_sha256 equals the registered "
                      "page's"),
            "versions": P.VERSIONS, "major_minor": list(P.VERSIONS_MAJOR_MINOR)},
        "constants": {"count_spread_max": P.COUNT_SPREAD_MAX, "count_lo": P.COUNT_LO, "count_hi": P.COUNT_HI,
                      "count_calls": P.COUNT_CALLS, "count_warmups": P.COUNT_WARMUPS,
                      "min_valid_calls": P.MIN_VALID_CALLS, "wave_q": list(P.WAVE_Q), "k_flat": P.K_FLAT,
                      "flat_min": P.FLAT_MIN, "wcp_edges": list(P.WCP_EDGES), "pool_min_cells": P.POOL_MIN_CELLS,
                      "life_q": list(P.LIFE_Q),
                      "refit_grid_mib": [P.REFIT_GRID_MIB[0], P.REFIT_GRID_MIB[-1], 2.5],
                      "overhead_median_max": P.OVERHEAD_MEDIAN_MAX, "overhead_worst_max": P.OVERHEAD_WORST_MAX,
                      "rate_ceiling_gbps": P.RATE_CEILING_GBPS, "f_ceiling_mhz": P.F_CEILING_MHZ},
        "frozen literals": {
            "skew E 8 (Mixtral-8x7B-tp2 PT n 2, NATIVE bincount)": {"counts": list(P.SKEW_E8),
                                                                    "source": list(P.SKEW_E8_SOURCE)},
            "skew E 64 (Qwen2-57B PT n 1, NATIVE bincount)": {"counts": list(P.SKEW_E64),
                                                              "source": list(P.SKEW_E64_SOURCE)},
            "design-P1 printed beside the readings (POST-HOC on SEEN pages)": P.DESIGN_PRINTED},
        "stand-in wave numbers (SEEN pages, frozen)": P.STANDIN_WAVE,
        "seen": seen_block(),
        "earlier registrations kept byte-identical": list(EARLIER),
        "plan": {"file": P.PLAN, "sha256": hashlib.sha256(P.plan_text().encode()).hexdigest(), "units_count": len(units),
                 "drop_order_last_first": order, **r},
    }


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: rental RR, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"motivation: {d['motivation']}", f"tool: {d['tool']}", "", "OWNER DECISIONS (2026-11-11)"]
    L += [f"  {x}" for x in d["owner_decisions (2026-11-11)"]]
    L += ["", "BUILD DECISIONS"] + [f"  {x}" for x in d["build decisions (to design-P1's sketch, before any page)"]]
    L += ["", f"PLAN {p['file']} sha256 {p['sha256']}: {p['units_count']} units, {p['units_min']} min estimated "
              f"({p['nodrop_min']} nodrop), drop order last first {p['drop_order_last_first']}"]
    for u in d["units"]["rows"]:
        L.append(f"  {u['model']} {u['step']} {u['label']} est {u['est']} cap {u['cap']} {u['drop']}"
                 f"{' depends ' + u['depends'] if u['depends'] else ''} {json.dumps(u['keys'], sort_keys=True)}")
    L += [f"  wall about {p['wall_min']} min with {p['boot_setup_min']} min of boot and setup: ${p['cost_usd']:.2f} at "
          f"${p['rate_usd_per_h']}/h; deadline launch + {p['deadline_after_launch_min']} min (${p['cost_at_deadline_usd']:.2f}); "
          f"guardian hard cap launch + {p['hard_cap_after_launch_min']} min (${p['cost_at_hard_cap_usd']:.2f} at most); "
          f"the nodrop core {p['nodrop_min']} min, about {p['nodrop_wall_min']} min wall (${p['nodrop_cost_usd']:.2f})",
          f"  rule: {p['rule']}", ""]
    for sec in ("stop rules", "readings", "forms (registered; constants POST-HOC on SEEN pages)"):
        L.append(sec.upper())
        for k, v in d[sec].items():
            L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v, sort_keys=True)}")
        L.append("")
    L.append("CELLS (COMPUTED)")
    for pg, rows in d["cells (COMPUTED)"].items():
        L.append(f"  {pg}:")
        for c in rows:
            L.append(f"    {c['cell']}: {c['shape']} T {c['T']} {c['tile']} {c['role']} m_e max {max(c['m_e'])} W_c/P "
                     f"{c['W_c_over_P']} pairs {c['pairs']} F_stag {c['F_stag_MiB']:g} MiB waves {c['waves_w1']} live w1 "
                     f"{c['live']['w1']} w2 {c['live']['w2']} C_u {c['C_u_bytes']} R {c['R_bytes']} calls "
                     f"{c['stamped_calls']}")
    L += ["", "PAGES"] + [f"  {k}: {v['sha256']} ({v['tile']}; {len(v['cells']) if isinstance(v['cells'], list) else v['cells']}"
                          f"{' cells' if isinstance(v['cells'], list) else ''})" for k, v in d["pages"].items()]
    L += [f"SHAPES {d['shapes']['file']} sha256 {d['shapes']['sha256']} ({len(d['shapes']['shapes'])} shapes)",
          "", "SEEN", f"  {json.dumps(d['seen'], sort_keys=True)}", "",
          f"CENSUS AND OCCUPANCY: {json.dumps(d['census and occupancy'], sort_keys=True)}",
          f"PAGE IDENTITY AND VERSIONS: {json.dumps(d['page identity and versions'], sort_keys=True)}",
          f"CONSTANTS: {json.dumps(d['constants'], sort_keys=True)}",
          f"FROZEN LITERALS: {json.dumps(d['frozen literals'], sort_keys=True)}",
          f"EARLIER REGISTRATIONS KEPT BYTE-IDENTICAL: {', '.join(d['earlier registrations kept byte-identical'])}",
          "", "code_pins: (in the JSON)"]
    return L


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    doc = build()
    doc["code_pins"] = {p: sha256(REPO / p) for p in PINNED}
    jpath, tpath = a.out_dir / f"{NAME}.json", a.out_dir / f"{NAME}.txt"
    jtext = json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n"
    ttext = "\n".join(txt(doc)) + "\n"
    plan = P.plan_text()
    if a.check:
        same_j = jpath.exists() and _strip(json.loads(jpath.read_text())) == _strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        same_p = (REPO / P.PLAN).is_file() and (REPO / P.PLAN).read_text() == plan
        same_g = all((REPO / rel).is_file() and (REPO / rel).read_text() == t for rel, t in GEN.files().items())
        for name, same in ((jpath.name, same_j), (tpath.name, same_t), ("plan", same_p), ("pages and shapes", same_g)):
            print(f"{name}: {'same' if same else 'DIFFERS'}")
        return 0 if same_j and same_t and same_p and same_g else 1
    atomic_write(REPO / P.PLAN, plan)
    atomic_write(jpath, jtext)
    atomic_write(tpath, ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {P.PLAN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
