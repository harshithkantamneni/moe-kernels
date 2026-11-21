#!/usr/bin/env python
"""Writes rental PC's registration, docs/registered/2026-11-06-pc-gh200.{json,txt}, and its plan, before any PC page.

    python scripts/scoring/pc/register_pc.py --write [--out-dir docs/registered]
    python scripts/scoring/pc/register_pc.py --check      # recompute, compare (exit 1 on any difference)

PC is a MEASUREMENT rental (owner, 2026-11-06): per-CTA DRAM rate against loads in flight. It registers no model form,
no constant, no bar and no gate on any model; every reading is DESCRIPTIVE. What it registers before any page: the
units and cells (the pages and shapes, by sha256), the stop rules, the validity and row-sanity rules, what each reading
is and how the reader computes it (scripts/scoring/pc/read_pc.py, by function), H1's falsification bands and its
SUPPORTED / FALSIFIED / MIXED label (owner decision 4, DESCRIPTIVE), the by-value inputs of the Phi readings, the plan
and its figures, and the code by sha256. `--check` recomputes everything and compares the JSON (code_pins excluded, as
every registration since rental 3), the text, the plan and the generated pages.
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
for p in (str(HERE), str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import gen_pc_pages as GEN  # noqa: E402
import pccommon as P  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = P.NAME
DESIGN = "moe-kernels-work 2026-11-06 design-PC/DESIGN.md (read-only over the repo at 6c800f84)"
#: every repo file PC's driver, probe, gate, counter step and reader run on the VM or the laptop
PINNED = (
    P.DRIVER, P.PARENT_DRIVER, "scripts/gh200_cal2_session.sh", "scripts/vm_run.sh",
    P.PROBE, "scripts/instr_probe_c3.py", "scripts/instr_probe.py", "scripts/launch_floor.py", "scripts/r3_decode.py",
    "moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py",
    "moe/instrumented/UPSTREAM.json", "moe/instrumented/upstream_v0.27.1_fused_moe.excerpt",
    "moe/instrumented/UPSTREAM_DECODE.json",
    "moe/__init__.py", "moe/pipeline.py", "moe/stages.py", "moe/state.py", "moe/spec.py",
    "moe/routing/__init__.py", "moe/routing/distributions.py", "moe/bench/timing.py", "moe/bench/exit_codes.py",
    "moe/bench/__init__.py", "moe/bench/ai_model.py", "moe/bench/provenance.py", "moe/bench/roofline.py",
    "moe/bench/schema.py", "moe/bench/weights.py",
    "moe/bench/tile_resolve.py",
    "scripts/private_weight_reference.py", "scripts/block_m_crossing_sweep.py", "scripts/locked_r3.py",
    "scripts/dram_counter_route_tile.py", "scripts/dram_counter_route.py", "scripts/r3_tile.py",
    "scripts/dram_counter_route_decode.py", "scripts/dram_counter_route_decode3.py", "scripts/scoring/decode/d3common.py",
    "scripts/tileA2_gate.py", "scripts/tile_census_a2.py", "scripts/tile_census.py", "scripts/r3_timing_model_v4.py",
    "scripts/scoring/tileA2/a2common.py", "scripts/scoring/tileA2/score_floor_a2.py", "scripts/scoring/tileA/tcommon.py",
    "scripts/scoring/pc/pccommon.py", "scripts/scoring/pc/gen_pc_pages.py", "scripts/scoring/pc/register_pc.py",
    "scripts/scoring/pc/read_pc.py", "scripts/scoring/cal3/read_c3.py", "scripts/scoring/cal2/read_c2.py",
    "scripts/scoring/decode/ds.readings.json", "scripts/scoring/tileB3/b3.predictions.json",
)
#: the registrations whose files PC reads or imports and keeps byte-identical (their --check must read same)
EARLIER = ("2026-10-26-cal3-gh200", "2026-11-04-decode-addendum4-gh200", "2026-11-04-tileB3-gh200")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_units() -> list[dict]:
    """The plan's units as the driver parses them (model, step, label, est, cap, nodrop or drop group, keys)."""
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
    """Every page a unit reads, by path: its sha256 and cells (the generated pages are written by gen_pc_pages)."""
    out = {}
    for rel, text in sorted(GEN.files().items()):
        if rel == P.SHAPES_FILE:
            continue
        doc = json.loads(text)
        out[rel] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "schema": doc["schema"],
                    "cells": [c["label"] for c in doc["cells"]],
                    "lock_mhz": doc.get("lock_mhz", P.DECODE_LOCK if rel.endswith("cnt.json") else None)}
    for rel in (P.PHI_128, P.PHI_64, P.PHI_128G16):
        doc = json.loads((REPO / rel).read_text())
        out[rel] = {"sha256": sha256(REPO / rel), "schema": doc["schema"],
                    "cells": [[c["label"], c["n"]] for c in doc["cells"]],
                    "what": "B3's registered histogram page (docs/registered/2026-11-04-tileB3-gh200), unchanged"}
    return out


def cells_table() -> dict:
    """Per PC page: each cell's shape, T, A, rows per expert, tile, live CTAs per GEMM (COMPUTED), stamped calls."""
    sh = P.load_shapes()
    out = {}
    for pg in P.BLOCKS:
        rows = []
        for c in P.check_page_doc(P.page_doc(pg), sh):
            live = {g: P.live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")}
            rows.append({"cell": c["label"], "shape": c["shape"], "T": c["T"], "A": sum(1 for x in c["counts"] if x),
                         "c_e": max(c["counts"]), "tile": P.tile_key(c["tile"]), "live": live,
                         "slab_bytes_w2": P.slab_bytes(c["cfg"], c["tile"], "w2"), "flush": c["flush"],
                         "stamped_calls": P.stamp_calls(live[c["read"]]), "role": c["role"]})
        out[pg] = rows
    return out


def readings() -> dict:
    return {
        "every reading": (
            "DESCRIPTIVE: computed from PC's pages by the rule below (read_pc.py), registered before any PC page; none is "
            "a model term, a constant, a STOP, a bar or a verdict. A reading that does not clear its rule reads NOT "
            "IDENTIFIED with the reason; nothing falls back and nothing reads 0 by default. Common rule: a reading is "
            "read only when PC-R0 holds for every launch it uses, the per-call medians' spread is at most 3% (5% at G "
            "<= 8), and its own rule holds"),
        "PC-R0 (validity; read_pc.launch, stamp_unit, board)": (
            "per launch: every row stamped live or dead; the live CTAs exactly the pids under (ntpp / BLOCK_M) x ceil(N "
            "/ BLOCK_N) and equal to the cell's sum over experts of ceil(c_e / BLOCK_M) x ceil(N / BLOCK_N); USE_TD "
            "False; no late buffer; the first w2 start after the last w1 end of the call; ROW SANITY. Per unit: the "
            "board (C2_POWER_W=700, LOCK_PROBE_1410=held, LOCK_PROBE_1710=held; G-BAND PASS for the Phi units only), "
            "the variant's regcheck line and the gate's HARD line PASS (a timing FAIL alone reads PERTURBED), the "
            "page's label the gate's, the page the registered file (sha256), the versions (python on major.minor; "
            "torch, triton, vllm exact), every stamped compile passing C3's SASS legs (every DISTINCT compile of the unit, keyed by "
            "its config and the CompiledKernel's hash, so a config Triton compiles twice is checked twice; the loads in "
            "flight must agree over every compile of a tile), "
            "the page COMPLETE (a PARTIAL page, cut by a cap or a crash, is REFUSED: its SASS legs are written only at the "
            "unit's end), and the unit's lock read back "
            "after its tool (pc-locks.env). A failing launch or unit is NOT READ, never zero. vLLM's EM padding "
            "launches dead CTAs (kind 2) past the live ones; they are counted, printed and excluded"),
        "ROW SANITY (read_pc.launch)": (
            f"every live start and end positive and finite, end after start, clock64 life positive; the per-CTA clock "
            f"in (0, {P.F_CEILING_MHZ:g}] MHz; every per-CTA rate at most {P.RATE_CEILING_GBPS:g} GB/s (the HBM3 spec "
            f"peak) and, where every slab is read by one CTA (PC's decode cells), the launch's X too; on a flushed "
            f"decode cell no CTA faster than Q_max / {P.L2_LATENCY_NS:g} ns, Q_max the tile's in-flight B bytes at "
            f"s4 ({P.L2_GUARD_STAGES} tiles; L2-served: the launch is void); the "
            "overhead finite. Else the launch is NOT READ"),
        "per CTA (read_pc.launch)": (
            "s_c, e_c globaltimer ns; k_c clock64 cycles; lambda_c = (e_c - s_c) / (1 + o), o the cell GEMM's overhead "
            "median from its perturb unit; where the globaltimer resolution (the least positive difference of the "
            "launch's distinct starts) exceeds a tenth of the median life, lambda_c = k_c / f_l / (1 + o), f_l the "
            "median clock of the CTAs living at least 10 resolutions, the call's w1 launch's resolution and clock where "
            "this launch has neither (a few CTAs in one tick; the source is recorded) (clock64 is never compared across "
            "SMs); f_c = "
            "k_c / (e_c - s_c); b_c = K x the CTA's BLOCK_N columns x 2 (the B slab; A rows are L2-hot by vLLM's order "
            "and printed apart); r_c = b_c / lambda_c; n_c and m_c the time averages over [s_c, e_c] of the launch's "
            "live CTAs GPU-wide and on the CTA's SM (itself included); plateau CTAs n_c >= 0.9 x the launch's peak live "
            "count; X_l = the B bytes streamed in the steady window (the live count at least 0.9 x its peak) over its "
            "length, times (1 + o)"),
        "per cell and GEMM (read_pc.cell_gemm)": (
            f"pooled over the stamped calls: r = the median r_c over plateau CTAs, NOT READ under "
            f"{P.MIN_PLATEAU_CTAS}; the spread of the per-call medians at most 3% (5% at G <= 8); NOT READ when the "
            f"GEMM's |overhead median| exceeds {P.OVERHEAD_MEDIAN_MAX:g} or its |worst pair| {P.OVERHEAD_WORST_MAX:g} "
            "(C3's OD-1 bound per cell, owner decision 5); PERTURBED pages are read under it and labelled. w2 is the "
            "read GEMM of every decode cell; w1 is stamped in every call"),
        "PC-R1 rate curve (read): r(rung) against G_live and the measured n and m, per block (tile, config, lock)": (
            "each rung's median per-CTA clock printed with whether it sits in the lock's band (A2's fallback band at 1410, "
            "rental 6's at 1710; printed, not applied); the medians split by SM sharing (every live CTA alone, m_c <= "
            f"{P.ALONE_M}, and sharing, m_c >= {P.SHARED_M}); and trap 13 (DRAM addresses): each CTA's expert read off "
            "expert_ids[pid // ceil(N / BLOCK_N)] (the saved -experts.npy), the median r_c over plateau CTAs per expert, "
            f"flagged SPREAD OVER 5% when (max - min) / median of the expert medians exceeds {P.EXPERT_SPREAD_FLAG:g} "
            "(read_pc.per_expert, by_sharing; DESCRIPTIVE, never a refusal)"),
        "PC-R2 F ladders (read_pc.f_ladder)": (
            "per G (8: pcs-e8-h512 at F 1792, 3584, 7168 and A-G8 at 14336; 128: Mixtral tp8, tp4, tp2 and 8x7B T128 A4): "
            "lambda_med = c0 + b / r_inf by OLS over the 4 rungs, READ with all 4 and the standard error of 1 / r_inf "
            "at most 5% of it; DS's per-ladder offsets printed beside (cross-board)"),
        "PC-R3 stage and BLOCK_K elasticity (read_pc.beta_q)": (
            "beta_Q = d ln r / d ln Q_B by least squares over (s2, s3, s4 at BK 64; 3 READ) and (s2, s3 at BK 128; 2 "
            "READ) at G 8, 128 and 384 (BK 64 only at 384); Q_B = n_if x BLOCK_K x BLOCK_N x 2 with n_if read off the "
            "stamped copy's PTX (pccommon.inflight_stages: in the innermost loop, cp.async.wait_group's immediate over "
            "the commit groups per k-step, plus 1) and required to equal num_stages - 1, else that config is NOT "
            "IDENTIFIED; rates c0-corrected (b / (lambda_med - c0)) by PC-R2 at the same G when READ, uncorrected and "
            "labelled otherwise; rho_seg = r(s2, BK 128) / r(s3, BK 64) at equal Q_B; scored against the PTX in-flight "
            "ratio, never against 1. D16's (s2, s3, s4) at G 16 and 128 printed, DESCRIPTIVE"),
        "PC-R4 clock elasticity (read_pc.beta_f)": (
            "beta_f = ln(r_1410 / r_1710) / ln(f_1410 / f_1710) per twin (block C against block A), f the rungs' measured "
            "median per-CTA clock, not the nominal lock; both nulls printed (compute-bound 1, latency-bound 0)"),
        "PC-R5 warps": "omega = r(w8) / r(w4) at 64x128x64 s3, G 8 and G 128; both READ",
        "PC-R6 SM sharing inside one launch (read_pc.sigma_of)": (
            f"at A-G160, A-G192, A-G288, over EVERY live CTA of the cell's valid launches: sigma = the median r_c of the "
            f"CTAs with m_c >= {P.SHARED_M} over the median r_c of the CTAs with m_c <= {P.ALONE_M}; read only when the "
            f"cell reads READ under the common rule; at least {P.MIN_SHARE_CTAS} CTAs in each group, else NOT IDENTIFIED "
            "(the scheduler did not mix them; at G 288 every SM holds 2 or more); each group's median n_c printed beside. "
            "The design's 10% n window is DROPPED (review MAJOR-4): a CTA sharing its SM at half the rate outlives the "
            "alone ones, so its n_c falls out of any window around theirs and the H3 falsifier (sigma < 0.6) could never "
            "fire; the groups' n_c are printed instead"),
        "PC-R7 transition into saturation (read_pc.transition)": (
            "over the D64 ladder (A-G2 to A-G1024, w2): X_l against G and N_if = n Q_B; L_gpu = Q_B n / X; X_sat = the "
            f"median X over the w2 rungs at G >= 396 and every flushed D64 w1 launch with at least one full wave and K >= "
            f"{P.X_SAT_W1_MIN_K} (each member's K printed; 396; read only "
            "when the census reads 3 CTAs per SM at 64x128x64w4s3), printed beside triad 3,726 GB/s; r0 = the median r "
            "over G 2, 4, 8, 16 (READ with at least 3 of them within 10% of r0); saturation READ with at least 2 rungs "
            "within 3% of X_sat; G_90 = the least rung with X >= 0.9 X_sat; d(G) = X / min(G r0, X_sat) - 1"),
        "PC-R8 transfer (read)": (
            "t = r(anchor) / r(synthetic rung) - 1: Mixtral T128 A4 against A-G128 (same tile, config, G and F); JetMoE "
            "T128 A4 against A-G64 and Mixtral tp8, tp4, tp2 against A-G128, each labelled 'F differs'; the 1410 twin; "
            "D16's Mixtral T16 A2 against D-G128, and the row-fill rungs Mixtral T4 A2 and T2 A2 against D-G128; each anchor's per-expert spread and flag (trap 13) printed beside its t. "
            "The CAL2 cell: r(Mixtral T128 A4) against Q_B / L_gpu(X), L_gpu "
            "read off the ladder at the anchor's X (piecewise linear, NOT IDENTIFIED outside the ladder's X span): "
            "within 10% CARRIES, over 10% SHAPE-SPECIFIC"),
        "PC-R9 L2 control": ("r(A-G8, flush off) / r(A-G8, flush on); both READ. The flush-off cell's overhead o is the "
                             "perturb unit's, measured on flushed bursts (the burst timer always flushes)"),
        "PC-R10 Phi w1 re-read pairs (read_pc.reread, phi_pairs; coordinator 2026-11-06: E1 and E2)": (
            "per Phi cell (the n x BLOCK_M per-expert counts only for 'balanced'; a cell with no counts is NOT READ) from its w1 stamps, E1 at 128x128 (W_c 264, C3's probe) and E2 at 64x128 (W_c 396, PC's "
            "probe), each read only when the census reads the registered occupancy: re-read pairs (C3-R2's pairs: "
            "same expert, same pid_n, pid_m - 1), their launch gap in units of M4's gap P L / W_c, the share launched "
            "together (g < 0.5), the share whose lives overlap, the B bytes streamed GPU-wide between partner starts "
            "dB; compulsory C = each active expert's w1 once plus the gathered A rows once (T k rows x H x 2: phi-split's C, so "
            "the miss fraction compares with REPORT.md's), re-read demand R = pairs x the slab, "
            "the measured miss fraction (D - C) / R with D B3's counted w1 bytes for the cell (board ebedab, by value), "
            "and the zero-constant rule's bytes (a re-read misses iff its life does not overlap its partner's and more "
            f"than {P.REREAD_L2_MIB:g} MiB stream between their starts) against D; printed, never selected or fitted. "
            "This board's own counted w1 bytes (units b128g1, b64g1; read_pc.board_counted) are printed beside, DESCRIPTIVE (they will likely "
            "fail B3's per-cell clock rule again)"),
        "PC-R11 Phi 64g1 life against packing (read_pc.phi_life_packing; a printed extra)": (
            "C3-R3's rule per cell and GEMM at W_c 396: ln(T_pred / T_stamp) = ln(L_pred / L_med) [LIFE] + ln(live L_med / "
            "(W_c T_stamp)) [PACKING], T_pred the frozen round-5 per-GEMM price by value, overhead-corrected, the stamped "
            "live CTAs equal to the priced live; the means over cells and the skew contrast; READ when e_plain (the mean of "
            "T_pred / T_plain - 1 over the skewed cells, T_plain the perturb unit's plain median) is at least 2% in size and "
            "the median |o| at most a quarter of it, else NOT REPRODUCED or DESCRIPTIVE ONLY"),
        "PC-R12 counter against time (read_pc.counter_vs_time)": (
            "per matching cell (Mixtral T128 A4, T128 A8, T16 A2, T4 A2): launch_ms - the stamps' span; with unit 16's "
            "counter page (page sha256, V_ALIGN_grid PASS, kernels wide, versions), gpu__time_duration - span and the "
            "counted DRAM bytes over the construction's (each active expert's weights once; the A operand's bytes printed "
            "beside, so a w1 ratio a little over 1 is no defect). w2's ncu times are "
            "confounded: ncu flushes L2 before each kernel, so w2's A operand (L2-hot in vLLM's order) is a DRAM read "
            "under ncu, and ncu's replays run at a lower clock (coordinator, 2026-11-06)"),
        "PC-R13 clock at kernel start": (
            "per launch: the median f_c of the CTAs starting in the first 50 us over the rest's, at least 8 in each; "
            "flagged under 0.97"),
        "PC-R14 placement": ("for launches of at most 132 live CTAs: distinct SMs, the most CTAs on one SM, Spearman's "
                             "rank of start order against pid; always printed"),
        "replicate": "r(R-G8, R-G128, R-G512) / r(A-G8, A-G128, A-G512) - 1: page noise, printed",
    }


def h1_block() -> dict:
    f, s = P.H1["falsify"], P.H1["support"]
    return {
        "hypothesis": ("H1, per-CTA memory-level parallelism: a CTA's streaming rate is its loads in flight over the "
                       "memory latency, r = Q_B / L, with L rising as the GPU's demand nears saturation"),
        "label": ("owner decision 4 (2026-11-06): the reader prints SUPPORTED, FALSIFIED or MIXED beside every number, "
                  "DESCRIPTIVE; the bands are registered here before any page (pccommon.h1_label)"),
        "inputs (low load: D64 at 1710 MHz, G 8; sigma at G 192)": (
            "beta_Q over (s2, s3, s4) at BK 64, G 8 (PC-R3); beta_f of C-G8 against A-G8 (PC-R4); omega of A-G8-w8 against "
            "A-G8 (PC-R5); the plateau ratio max / min of r over A-G2, A-G4, A-G8, A-G16 with X(G 16) / X_sat (PC-R1, "
            "PC-R7); sigma at A-G192 (PC-R6); X(G 8) / X_sat, the low-load premise"),
        "FALSIFIED when any read falsifier holds": [
            f"beta_Q < {f['beta_Q_below']} (the rate barely moves when loads in flight triple)",
            f"beta_f > {f['beta_f_above']} (the rate tracks the SM clock)",
            f"omega > {f['omega_above']} (more warps at the same loads in flight run faster)",
            f"no plateau: r(G 2) / r(G 16) > {f['plateau_ratio_above']} with X(G 16) <= "
            f"{f['plateau_load_frac_max']} X_sat (the limit is shared, H4)",
            f"sigma < {f['sigma_below']} (CTAs sharing an SM run at about half the rate, H3)"],
        "SUPPORTED when every band holds": [
            f"beta_Q in [{s['beta_Q'][0]}, {s['beta_Q'][1]}]", f"|beta_f| <= {s['beta_f_abs_max']}",
            f"omega in [{s['omega'][0]}, {s['omega'][1]}]", f"the plateau within {s['plateau_within']:.0%} over G 2 to 16",
            f"sigma >= {s['sigma_min']}"],
        "otherwise": ("NOT IDENTIFIED when no falsifier fires and an input is unread (the low-load premise included: "
                      "it is unread only when X(G 8) is, since an unread X_sat falls back to the triad), or the premise "
                      "fails (X(G 8) over 0.25 of its denominator: G 8 is not low load); else "
                      "MIXED. A beta_Q under 1 at G 128 or 384 does not falsify H1 (its own prediction once L rises with "
                      "load)"),
        "the CAL2 sub-reading": "PC-R8: within 10% CARRIES; over 10% SHAPE-SPECIFIC",
        "printed beside the label": ("whether beta_Q's rates were c0-corrected (PC-R2 READ at G 8) or not; and which "
                                     "denominator the low-load premise used"),
        "the low-load premise and its fallback (owner decision, 2026-11-06)": (
            "X(G 8) / X_sat with PC-R7's X_sat when it is read; when it is not, X(G 8) / the published triad, "
            f"{P.TRIAD_FALLBACK_GBPS:g} GB/s ({P.TRIAD_FALLBACK_SOURCE}), and the output labels the clause FALLBACK "
            "every time it is used. The fallback replaces X_sat in this premise only: the plateau falsifier's load "
            "condition (X(G 16) <= 0.25 X_sat) keeps X_sat, so no falsifier is loosened"),
        "bands": P.H1,
    }


def stop_rules() -> dict:
    return {
        "prelude (C3's, unchanged)": (
            f"the session STOPS before any page unless the checkout is clean, sudo works, the GPU is idle, {P.POWER_W} W "
            "reads back on both enforced.power.limit and -q's GPU-scope Current Power Limit within 1 W (polled up to 30 "
            f"s; no fallback), persistence is Enabled and the session lock {P.SESSION_LOCK} MHz reads back; the prelude "
            f"also probes {P.DECODE_LOCK} MHz (lock, read back, reset)"),
        "unit locks (PC)": (
            f"a perturb, stamps or decodebytes unit at lock-mhz={P.DECODE_LOCK} refuses unless the prelude's probe held it; "
            "it locks, reads back (two tries) and runs; after the tool the lock is read back again (pc-locks.env); the "
            "reset on exit restores the session, whose next unit locks its own; the reader refuses a stamps unit whose own "
            "lock, or whose probe=pc perturb unit's lock, did not read back after its tool"),
        "the regcheck drop rule (owner decision 2)": (
            "the probe=pc regcheck (unit 3) exiting anything but 0 or 1 (refused, crashed, stopped at its cap, skipped) "
            "or that writes no regcheck line for a probe=pc stamps unit drops every probe=pc unit (4 to 7, 14, 17, 20); "
            "the existing tools still run (units 8 to 13, 15, 16, 18, 19). Exit 1 (a variant's occupancy or path FAIL) "
            "refuses that variant's stamps unit by its regcheck line; a cell failing its readback is excluded alone. The "
            "probe exits 4 on any uncaught error, never 0 or 1. The rule drops E2 while E1 (C3's probe) still runs: owner "
            "decision 2 is read as winning over 'never E2 before E1' (an owner item)"),
        "G-BAND (owner decision 5: C3's STOP rule kept for the Phi pages)": (
            "A2's G-BAND (tileA2_gate.py band) after the two f128 pages: on a FAIL, or when the gate is skipped, the Phi "
            "group (drop-group phi: the f128 pages, the gate, C3's regcheck and perturb, E1, E2, b128g1, b64g1) and b128g16 "
            "(depends=phi) are dropped and the session goes on; the decode units at 1710 are not gated by it. The reader reads the Phi "
            "units only on a G-BAND PASS"),
        "the gate and the stamps (C3's)": (
            "a stamps unit runs only when its regcheck line and the gate's HARD line (occupancy, the all-off SASS, the "
            "upstream file, C3's SASS legs, and for probe=pc every cell's readback) read PASS; a timing FAIL alone runs "
            "it PERTURBED (C3's OD-1, owner decision 5)"),
        "budget": ("the driver drops the LAST droppable unit first: rep, b16, cnt, d16, then the phi group (one drop-group: "
                   "E1 and E2 together, never E2 before E1); a unit whose estimate no longer fits is SKIPPED; the "
                   "guardian terminates at the hard cap"),
        "never": "no PC reading is a STOP; a reading that does not identify reads NOT IDENTIFIED",
    }


def seen_block() -> dict:
    return {
        "reserved (owner decision 3)": {"models": list(P.RESERVED),
                                        "rule": ("never loaded, never timed, never shaped: the probe refuses a variant, cell "
                                                 "or weight build naming one or a shape equal to one (pccommon.shape_cfg), "
                                                 "and tests/test_instr_probe_pc.py asserts nothing in the plan loads them")},
        "loaded (all SEEN)": list(P.REAL_ALLOWED) + ["the registered synthetic shapes"],
        "becomes SEEN at decode tiles": ["mixtral-8x7b-tp4", "mixtral-8x7b-tp8"],
        "still unseen at vLLM's prefill tiles (COMPUTED at 6c800f84, FROZEN)": list(P.UNSEEN_AT_PREFILL_6C800F84),
        "statement": ("PC times none of the three models held for the decode blind and keeps every unseen model unseen "
                      "at prefill tiles"),
    }


def build() -> dict:
    units = plan_units()
    r = P.rental()
    order = []
    for u in reversed(units):
        if u["drop"] != "nodrop" and u["drop"] not in order:
            order.append(u["drop"])
    assert tuple(order) == P.CUT_ORDER, order
    priced = json.loads((REPO / P.PRICED).read_text())
    shapes = P.shapes_doc()
    return {
        "registered": f"{P.DATE}, before any PC page",
        "name": NAME,
        "status": ("MEASUREMENT, DESCRIPTIVE: PC certifies nothing, registers no model form, no constant, no bar and no "
                   "gate on any model, and moves no registered verdict. Every reading is DESCRIPTIVE; H1's label is "
                   "printed beside the numbers (owner decision 4)"),
        "design": DESIGN,
        "motivation": (
            "The PC core (blocks A, C, D) is motivated by the decode CAL2 few-CTA failure: v5d's CAL2 worst cell, Mixtral "
            "T128 A4 w2 (128 CTAs in one wave, about 1 CTA per SM), ran slower than bandwidth, and no ceiling read off "
            "DS's pages priced it (decode closure, 3160bed5). Phi-3.5's B3 miss is the w1 byte law, the L2 hit fraction "
            "of w1 re-reads, not a per-CTA rate miss (moe-kernels-work 2026-11-06 phi-split/REPORT.md, POST-HOC on SEEN "
            "pages: counted bytes take 128g1 from +3.38% to +0.20% and 64g1 from -3.24% to -0.07%); the Phi units ask "
            "the launch-schedule question (PC-R10), not the rate question (coordinator, 2026-11-06)"),
        "tool": ("scripts/gh200_pc_session.sh (C3's pinned driver plus lines marked PC), scripts/instr_probe_pc.py (the new "
                 "host-side probe), scripts/instr_probe_c3.py (Phi E1, unchanged), moe/instrumented/fused_moe_instr.py (the "
                 "copy, unchanged, STAMPS 3), scripts/tileA2_gate.py band (G-BAND), scripts/dram_counter_route_decode3.py "
                 "(unit 16), scripts/scoring/pc/read_pc.py (the reader)"),
        "owner_decisions (2026-11-06)": [
            "1 the full plan, cut last unit first: the replicate, b128g16, the counter cross-check, the T <= 32 block, then "
            "the Phi group as one drop group (the design's order with the coordinator's changes: E1 and E2 cut together, "
            "so the design's separate Phi 64g1 step is gone, and b128g16 cut right after the replicate)",
            "2 the new probe and driver accepted on a first-GPU run, after full laptop tests, a driver dry run and an "
            "independent review; a failed start-up regcheck drops only the new probe's units",
            "3 synthetic shapes (top_k 2, the least grid 2 CTAs), the Mixtral tp shards and JetMoE at decode tiles; "
            "Mixtral-8x22B, Qwen2-57B and Granite-3.0-1B untouched, a test asserts nothing loads them",
            "4 the reader prints H1's SUPPORTED / FALSIFIED / MIXED label beside every number; the bands registered "
            "before any page; every reading DESCRIPTIVE; no model form and no constant registered",
            "5 the 1410 MHz clock twin on the decode tile; C3's STOP rule (its f128 pages and G-BAND) kept for the Phi "
            "pages; DS's counter step ported for the same-board cross-check; gpubench dropped; PERTURBED pages read "
            "under C3's rule"],
        "coordinator (2026-11-06, phi-split REPORT.md)": [
            "PC-R10's pairs on E2 (64x128) as well as E1, E2 against B3's counted 64g1 bytes; PC-R11 a printed extra",
            "E1 and E2 one drop group (never E2 before E1)",
            "Phi counted-bytes units at 1410 on the existing bytes tool, as B3 ran them: b128g1 and b64g1 in the Phi "
            "group, b128g16 its own lower-priority group; read DESCRIPTIVELY",
            "PC-R12 notes that w2's ncu times are confounded",
            "this registration says the core is motivated by the decode CAL2 few-CTA failure; Phi's miss is the byte law"],
        "build corrections (to the design, before any page)": [
            "loads in flight: the design's 'cp.async.wait_group immediate + 1, expected s - 1' reads 3 on C3's published "
            "s3 PTX (wait 2, A and B committed as two groups per k-step); n_if = wait / groups per step + 1 (2 there), "
            "required to equal num_stages - 1",
            "PC-R1's 8-CTA rule at G 2: cells whose read GEMM has under 4 live CTAs take 6 stamped calls",
            "plateau CTAs: n_c >= 0.9 x the launch's PEAK live count (G_live within one wave; past a wave G_live cannot be "
            "concurrent)",
            "independent review (GO-WITH-FIXES), applied before any page: each variant's regcheck and gate lines are written "
            "as it finishes and a cell failing its readback is excluded, not its block; an uncaught probe error exits 4; a "
            "probe=pc regcheck that writes no line for a probe=pc stamps unit is a failed start; a perturb unit gates no "
            "dropped unit; b128g16 depends on the Phi group (C3's STOP rule); the start-up regcheck's cap is 20 min; sigma "
            "reads every live CTA of a READ cell; the L2 guard takes the tile's s4 loads in flight; a launch of a few CTAs "
            "takes its call's w1 timer for the clock64 path; X_sat's w1 members need K >= 4096 and a flush; the plateau "
            "falsifier is r(G 2) / r(G 16) as the design; PC-R11 follows C3-R3 (e_plain on the plain reference, means, the "
            "live check); b128g1 and b64g1 take B3's registered estimates (6, 5); price_pc.py and phi-split's split.json "
            "live outside the repo (moe-kernels-work 2026-11-06), recorded by sha256 in the priced file",
            "the X ceiling and the L2 guard apply where every slab is read by one CTA (PC's decode cells); on Phi's tread "
            "cells X is a demand rate (C3's design note: Mixtral w2 at 3.8 to 4.1 TB/s against triad 3.73)",
            "the census: the tile census dry run gives 10 distinct configs (64x128x64w8s3 is both D64's warps factorial "
            "and Phi E2's tile); the s4 BK 64 config holds 2 CTAs per SM, so A-G384-s4k64 crosses a wave",
            "the C3 probe's E1 units and PC's units write one gate env and one regcheck env (ids u<unit>-<label>); a PC "
            "regcheck line ends 'excluded N: labels' when cells failed their readback (readers parse the first word)",
            "the probe writes stamps.json, the stamp arrays, the PTX and the regcheck and gate files atomically (a "
            "temporary file, then os.replace)"],
        "power_and_lock": {"power_limit_w": P.POWER_W, "session_lock_mhz": P.SESSION_LOCK, "decode_lock_mhz": P.DECODE_LOCK,
                           "per_cta_clock_band_printed": {str(k): list(v) for k, v in P.F_BAND.items()}},
        "instrument": {
            "copy": "moe/instrumented/fused_moe_instr.py, unchanged (no kernel written or edited; no new stamp point; no hint)",
            "copy_sha256": sha256(REPO / "moe/instrumented/fused_moe_instr.py"),
            "spec": "stamps=ends,evict_a=none,evict_b=none",
            "probe": ("scripts/instr_probe_pc.py: host-side Python driving vLLM v0.27.1's pinned kernel through the "
                      "instrumented copy; imports instr_probe_c3.py and instr_probe.py (both byte-identical)"),
            "per cell": ("the readback (V-BIN, V-ALIGN: r3_decode.check_readback at the cell's config), 3 warm-up calls, "
                         "then 3 stamped calls (6 where the read GEMM has under 4 live CTAs), each with its buffers "
                         "reserved, the GPU held ahead by a 25 ms sleep and a 240 MiB L2 flush outside the launch event "
                         "pair (none on the flush-off control)"),
            "weights": "one NATIVE copy per shape (synthetic or SEEN); Phi E2 C3's 9 declared copies, NATIVE"},
        "units": {"plan": P.PLAN, "rows": units},
        "cells (COMPUTED)": cells_table(),
        "pages": pages(),
        "shapes": {"file": P.SHAPES_FILE, "sha256": hashlib.sha256(P.shapes_text().encode()).hexdigest(),
                   "shapes": shapes["shapes"]},
        "priced": {"file": P.PRICED, "sha256": sha256(REPO / P.PRICED), "what": priced["what"],
                   "generator": priced["generator"], "sources_sha256": priced["sources_sha256"],
                   "rows": len(priced["rows"])},
        "readings": readings(),
        "H1": h1_block(),
        "stop rules": stop_rules(),
        "census and occupancy": {
            "unit": "unit 2: tile_census_a2.py over the 10 PC configs and C3's 128x128 on Mixtral-8x7B and Phi-3.5 (compile only)",
            "registered occupancy": P.OCCUPANCY,
            "design occupancy (dry run, printed)": P.DESIGN_OCCUPANCY,
            "rule": ("the wave-dependent readings (PC-R7's X_sat pool, PC-R10, PC-R11) read only where the census reads the "
                     "registered occupancy; a missing census or a path mismatch makes them NOT IDENTIFIED")},
        "page identity and versions": {
            "pages": "each stamps manifest's source_sha256 equals the registered page's (histogram_sha256 on C3's E1)",
            "versions": P.VERSIONS, "major_minor": list(P.VERSIONS_MAJOR_MINOR)},
        "constants": {"plateau_frac": P.PLATEAU_FRAC, "min_plateau_ctas": P.MIN_PLATEAU_CTAS, "spread_max": P.SPREAD_MAX,
                      "spread_max_small": P.SPREAD_MAX_SMALL, "spread_small_G": P.SPREAD_SMALL_G, "f_rungs": P.F_RUNGS,
                      "se_inv_rinf_max": P.SE_INV_RINF_MAX, "shared_m": P.SHARED_M, "alone_m": P.ALONE_M,
                      "min_share_ctas": P.MIN_SHARE_CTAS, "sigma_Gs": list(P.SIGMA_GS), "plateau_Gs": list(P.PLATEAU_GS),
                      "plateau_within": P.PLATEAU_WITHIN, "sat_within": P.SAT_WITHIN, "knee_frac": P.KNEE_FRAC,
                      "sat_rungs": P.SAT_RUNGS, "carry_within": P.CARRY_WITHIN, "l2_latency_ns": P.L2_LATENCY_NS,
                      "start_window_ns": P.START_WINDOW_NS, "start_flag": P.START_FLAG,
                      "start_min_ctas": P.START_MIN_CTAS, "repro_min": P.REPRO_MIN, "overhead_frac": P.OVERHEAD_FRAC,
                      "reread_l2_mib": P.REREAD_L2_MIB, "overhead_median_max": P.OVERHEAD_MEDIAN_MAX,
                      "overhead_worst_max": P.OVERHEAD_WORST_MAX, "timer_max_frac": P.TIMER_MAX_FRAC,
                      "rate_ceiling_gbps": P.RATE_CEILING_GBPS, "f_ceiling_mhz": P.F_CEILING_MHZ,
                      "triad_gbps": P.TRIAD_GBPS},
        "seen": seen_block(),
        "earlier registrations kept byte-identical": list(EARLIER),
        "plan": {"file": P.PLAN, "sha256": hashlib.sha256(P.plan_text().encode()).hexdigest(), "units_count": len(units),
                 "drop_order_last_first": order, **r},
    }


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: rental PC, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"motivation: {d['motivation']}", f"tool: {d['tool']}", "", "OWNER DECISIONS (2026-11-06)"]
    L += [f"  {x}" for x in d["owner_decisions (2026-11-06)"]]
    L += ["", "COORDINATOR (2026-11-06, phi-split REPORT.md)"] + [f"  {x}" for x in d["coordinator (2026-11-06, phi-split REPORT.md)"]]
    L += ["", "BUILD CORRECTIONS"] + [f"  {x}" for x in d["build corrections (to the design, before any page)"]]
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
    for sec in ("stop rules", "readings", "H1"):
        L.append(sec.upper())
        for k, v in d[sec].items():
            L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v)}")
        L.append("")
    L.append("CELLS (COMPUTED)")
    for pg, rows in d["cells (COMPUTED)"].items():
        L.append(f"  {pg}:")
        for c in rows:
            L.append(f"    {c['cell']}: {c['shape']} T {c['T']} A {c['A']} c_e {c['c_e']} {c['tile']} live w1 "
                     f"{c['live']['w1']} w2 {c['live']['w2']} calls {c['stamped_calls']}{'' if c['flush'] else ' NO FLUSH'}")
    L += ["", "PAGES"] + [f"  {k}: {v['sha256']} ({len(v['cells'])} cells)" for k, v in d["pages"].items()]
    L += [f"SHAPES {d['shapes']['file']} sha256 {d['shapes']['sha256']}",
          f"PRICED {d['priced']['file']} sha256 {d['priced']['sha256']} ({d['priced']['rows']} rows): {d['priced']['what']}",
          "", "SEEN", f"  {json.dumps(d['seen'], sort_keys=True)}", "",
          f"CENSUS AND OCCUPANCY: {json.dumps(d['census and occupancy'], sort_keys=True)}",
          f"PAGE IDENTITY AND VERSIONS: {json.dumps(d['page identity and versions'], sort_keys=True)}",
          f"CONSTANTS: {json.dumps(d['constants'], sort_keys=True)}",
          f"EARLIER REGISTRATIONS KEPT BYTE-IDENTICAL: {', '.join(d['earlier registrations kept byte-identical'])}",
          "", "code_pins: (in the JSON)"]
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
    (REPO / P.PLAN).write_text(plan)
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {P.PLAN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
