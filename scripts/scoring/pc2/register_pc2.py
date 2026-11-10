#!/usr/bin/env python
"""Writes rental PC2's registration, docs/registered/2026-11-10-pc2-gh200.{json,txt}, and its plan, before any PC2 page.

    python scripts/scoring/pc2/register_pc2.py --write [--out-dir docs/registered]
    python scripts/scoring/pc2/register_pc2.py --check      # recompute, compare (exit 1 on any difference)

PC2 is a MEASUREMENT rental (owner, 2026-11-10): what v6d cannot yet price (design-D6 section 7), on SEEN or synthetic
shapes. It registers no model form, no constant, no bar and no gate on any model; every reading is DESCRIPTIVE. What it
registers before any page: the units and cells (the pages and shapes, by sha256), the stop rules, the validity and
row-sanity rules, what each reading is and how the reader computes it (scripts/scoring/pc2/read_pc2.py, by function),
how each of v6d's open terms will be read off PC2's pages (the reading, never a form or a value), the plan and its
figures, and the code by sha256. `--check` recomputes everything and compares the JSON (code_pins excluded, as every
registration since rental 3), the text, the plan and the generated pages. Every write is atomic (a temporary file, then
os.replace).
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
for p in (str(HERE), str(HERE.parent / "pc"), str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import gen_pc2_pages as GEN  # noqa: E402
import pc2common as P  # noqa: E402
import register_pc as RPC  # noqa: E402

REG = REPO / "docs" / "registered"
NAME = P.NAME
DESIGN = ("moe-kernels-work 2026-11-09 design-D6/DESIGN.md section 7 (what v6d cannot price, the PC2 sketch), read-only "
          "over the repo at 3e3cfe53; built in moe-kernels-work 2026-11-10 build-PC2")
#: every repo file PC2's driver, launcher, probe and reader run on the VM or the laptop: PC's pins (all byte-identical,
#: PC's driver, probe, launcher and reader among them) and PC2's own files
PINNED = tuple(RPC.PINNED) + (
    P.DRIVER, P.PROBE, P.LAUNCHER, "scripts/scoring/pc2/pc2common.py", "scripts/scoring/pc2/gen_pc2_pages.py",
    "scripts/scoring/pc2/register_pc2.py", "scripts/scoring/pc2/read_pc2.py",
)
#: the registrations whose files PC2 reads, imports or copies and keeps byte-identical (their --check must read same)
EARLIER = ("2026-10-26-cal3-gh200", "2026-11-04-decode-addendum4-gh200", "2026-11-04-decode-addendum5-gh200",
           "2026-11-04-tileB3-gh200", "2026-11-06-pc-gh200")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_units() -> list[dict]:
    """The plan's units as the driver parses them (PC's parser on PC2's plan text)."""
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
                    "cells": [c["label"] for c in doc["cells"]], "lock_mhz": doc["lock_mhz"]}
    return out


def cells_table() -> dict:
    """Per PC2 page: each cell's shape, T, path, A, rows per expert, tile, live CTAs per GEMM, the w2 slab, whether
    every slab is read once, its partner, stamped calls (COMPUTED)."""
    sh = P.load_shapes()
    out = {}
    for pg in P.BLOCKS:
        rows = []
        for c in P.check_page_doc(P.page_doc(pg), sh):
            live = {g: P.live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")}
            rows.append({"cell": c["label"], "shape": c["shape"], "T": c["T"], "path": c["path"],
                         "A": sum(1 for x in c["counts"] if x), "c_e": max(c["counts"]), "tile": P.tile_key(c["tile"]),
                         "live": live, "slab_bytes": {g: P.slab_bytes(c["cfg"], c["tile"], g) for g in ("w1", "w2")},
                         "slab_once": P.slab_once(c["cfg"], c["counts"]), "twin": c.get("twin"), "pair": c.get("pair"),
                         "stamped_calls": P.stamp_calls(live[c["read"]]), "role": c["role"]})
        out[pg] = rows
    return out


def readings() -> dict:
    return {
        "every reading": (
            "DESCRIPTIVE: computed from PC2's pages by the rule below (read_pc2.py), registered before any PC2 page; none "
            "is a model term, a constant, a STOP, a bar or a verdict. A reading that does not clear its rule reads NOT "
            "IDENTIFIED (or NOT READ) with the reason; nothing falls back and nothing reads 0 by default. Common rule (PC's): "
            "a reading is read only when PC2-R0 holds for every launch it uses, the per-call medians' spread is at most 3% "
            "(5% at G <= 8), and its own rule holds"),
        "PC2-R0 (validity; read_pc.launch unchanged, read_pc2.stamp_unit, read_pc.board)": (
            "PC-R0, unchanged, with PC2's geometry: per launch every row stamped live or dead; the live CTAs exactly the "
            "pids under (ntpp / BLOCK_M) x ceil(N / BLOCK_N) and equal to the cell's (aligned: sum over experts of ceil(c_e "
            "/ BLOCK_M) x ceil(N / BLOCK_N); naive: T k x ceil(N / BLOCK_N), r3_decode.geometry's rule); USE_TD False; no "
            "late buffer; the first w2 start after the last w1 end of the call; ROW SANITY. Per unit: the board "
            "(C2_POWER_W=700, LOCK_PROBE_1410=held, LOCK_PROBE_1710=held), the variant's regcheck line and the gate's HARD "
            "line PASS (a timing FAIL alone reads PERTURBED), the page's label the gate's, the page the registered file "
            "(sha256), the versions (python on major.minor; torch, triton, vllm exact), every stamped compile passing C3's "
            "SASS legs, the page COMPLETE (a PARTIAL page is REFUSED), the unit's lock and its perturb unit's lock read back "
            "after the tool (pc-locks.env), and (PC2's new leg) EVERY stamped compile's CTAs per SM, from the stamps "
            "manifest, equal to its tile's registered occupancy: a stamped kernel whose occupancy is not the plain one's "
            "is not read (design-D6 section 7 item 1; PC's block D was refused at 7 against 8). A failing launch or unit "
            "is NOT READ, never zero"),
        "ROW SANITY (read_pc.launch, unchanged)": (
            f"every live start and end positive and finite, end after start, clock64 life positive; the per-CTA clock in "
            f"(0, {P.F_CEILING_MHZ:g}] MHz; every per-CTA rate at most {P.RATE_CEILING_GBPS:g} GB/s (the HBM3 spec peak) "
            f"and, where every slab is read by one CTA (pc2common.slab_once: every aligned cell, and a naive cell with no "
            f"follower), the launch's X too; on such a flushed cell no CTA faster than Q_max / {P.L2_LATENCY_NS:g} ns, "
            f"Q_max the tile's in-flight B bytes at s4 ({P.L2_GUARD_STAGES} tiles: L2-served, the launch is void); a naive "
            "cell with followers takes neither the X ceiling nor the L2 guard (its re-reads may be L2-served by design); "
            "the overhead finite. Else the launch is NOT READ"),
        "per CTA and per cell (read_pc.launch, read_pc.cell_gemm, unchanged)": (
            "PC's: lambda_c = (e_c - s_c) / (1 + o), o the cell GEMM's overhead median from the perturb unit (the clock64 "
            "path where the globaltimer resolution exceeds a tenth of the median life); b_c = K x the CTA's BLOCK_N columns "
            "x 2; r_c = b_c / lambda_c; n_c, m_c time averages; plateau CTAs n_c >= 0.9 x the launch's peak; X_l over the "
            "steady window x (1 + o); per (cell, GEMM): r = the median r_c over plateau CTAs (NOT READ under 8), the per-call "
            "spread at most 3% (5% at G <= 8), NOT READ when |overhead median| > 0.1 or |worst| > 0.2 (C3's OD-1 bound)"),
        "EXACT TIME (read_pc2.time_offset, rebase; PC2's reader rule)": (
            "read_pc.launch casts globaltimer stamps to float64, and an absolute stamp (about 1.7e18 ns) has a float64 "
            "step of 256 ns there, so PC's lives and starts are quantised to 256 ns although the timer's tick is 32 ns "
            "(found building PC2; PC's pinned reader and its published reading are unchanged). PC2 subtracts ONE offset "
            "per unit from both globaltimer columns of every row stamped live or dead before read_pc.launch: the least of "
            "those rows' stamps past 2^40 ns, less 2^20 ns (0 when none is past 2^40 ns; a garbled small stamp cannot "
            "move it). Every difference is unchanged "
            "and exact in float64; a stamp that was zero or negative stays not positive, so ROW SANITY still refuses it. "
            "The sanity rule: after the rebase every stamp of a live or dead row must lie under 2^40 ns, else PC2-R0 reads "
            "EXACT TIME NOT APPLIED with the reason and the unit is read on its stamps as written (PC's reading); APPLIED "
            "or NOT NEEDED otherwise. The status, the offset, the unit's timer reference and the count of launches per "
            "timer source are printed in PC2-R0"),
        "TIMER (read_pc2.unit_timer; PC2's change to PC's timer rule)": (
            "PC's rule takes a launch's own globaltimer resolution (the least positive difference of its distinct starts), "
            "else its call's w1 launch's; a launch whose live CTAs all start in one tick has none, and on PC's pages two "
            "cells were lost that way (A-F7168-G8 call 0, A-TP8-G128 w1 calls 0 and 1: NOT READ; under EXACT TIME those "
            "launches do have a resolution, 32 ns: PC's 256 ns ticks were the float64 step). The resolution is the "
            "card's (32 ns on 226 of PC's 227 block A launches, exact), so PC2 adds a last source: the UNIT's "
            f"resolution, the most common own resolution over the unit's launches, read only when at least "
            f"{P.TIMER_UNIT_MIN} launches report one and it holds half of them; the clock (f_long) is never taken across "
            "launches. The source is recorded per launch (res_from). Registered before any page"),
        "SPAN (read_pc2.span_of; the GEMM's own time, new in PC2)": (
            f"per valid launch whose live CTAs are at most its tile's registered wave (132 x occupancy; the census must read "
            f"it): kappa_l = span / the median life (first start to last end over the median CTA life; o cancels); on every "
            f"valid launch, single-wave or not: t_l = span / (1 + o) us, the GEMM's own time; per (cell, GEMM): kappa and t "
            f"the medians over calls, read with at least {P.SPAN_MIN_LAUNCHES} valid launches and the cell GEMM READ under "
            "the common rule, else NOT READ (no t, no kappa). A multi-wave launch reads t only (no kappa: not one wave). "
            "Every ratio of t (PC2-R4, R5) reads only where both t are READ. This registers the span reading design-D6 "
            "section 10 found unregistered in PC"),
        "PLAIN (read_pc2.plain_of; the plain-kernel event time, its limit stated)": (
            "per (cell, GEMM) of every gated variant, stamped or refused: the perturb unit's plain-kernel median (8 bursts "
            "of 10 calls, an L2 flush before each call outside the event pair, alternating with the stamped copy), in us, "
            "and the overhead o; printed beside every stamp reading, DESCRIPTIVE. Its limit: an event pair times the "
            "whole launch, the launch's fixed cost included; it has no per-CTA life, so it cannot split c0 from the stream "
            "or read kappa, and it fills no v6d slot alone. Where a stamped variant is refused (its regcheck or its "
            "occupancy leg), PLAIN is all PC2 holds for its cells, read descriptively"),
        "PC2-R1 the decode tile 16x64x128 w4 s4 (unit sd; read_pc2.r1)": (
            "R1a the rate curve (PC-R1's columns) over D-G4 to D-G1024 and the short-slab rungs; R1b the F ladders at G 8 "
            "(D-G8-F512 to F7168 and D-G8) and G 128 (D-G128-F512 to F7168 and D-G128): read_pc.f_ladder (lambda_med = c0 + "
            "b / r_inf by OLS over the READ rungs, READ with at least 4 and the standard error of 1 / r_inf at most 5% of "
            "it); R1c the L table: per READ single-wave rung of the long-slab ladder (D-G4 to D-G384), Q_B = n_if x 128 x "
            "64 x 2 with n_if off the stamped PTX (required = num_stages - 1 = 3), L = Q_B (lambda_med - c0) / b and N_eff "
            "= n Q_B (lambda_med - c0) / lambda_med, c0 the R1b G 128 value when READ, else the table is NOT IDENTIFIED; "
            "R1d X_sat on the tile, PC's definition (PC-R7): the median X over READ long-slab w2 rungs (w2 K >= 14336) strictly "
            "past one wave (G > 396) and READ flushed w1 launches with at least 396 live CTAs and K >= 4096, read with 2 "
            "or more members and the census reading 3 CTAs per SM, the pool's composition printed; R1e SPAN per "
            "rung; R1f the s3 factorial (unit sd3): read_pc.beta_q over (s3, s4) at G 16, 128, 384 (2 READ, Q_B off the "
            "PTX), DESCRIPTIVE; R1g n_if per tile, every compile agreeing; R1h PLAIN for every D cell; R1i r(D-MX-T16-A2) / "
            "r(D-G128) - 1 (same tile, G and F; E 8 against 32), with its per-expert spread (trap 13)"),
        "PC2-R2 short slabs, 66 to 459 KB (unit ss and block D's short rungs; read_pc2.r2)": (
            "R2a c0 and r_inf per G (8, 32, 128, 256) on 64x128x64 w4 s3: read_pc.f_ladder over S-G{G}-F256, F512, F1024, "
            "F1792 and S-G{G} (F 14336, the same-board 3.67 MB anchor); R2b the same over the four short rungs alone "
            "(66 to 459 KB): whether c0 read on short slabs equals c0 read across to 3.67 MB, printed side by side; R2c "
            "SPAN per short-slab cell (kappa at short slabs, on 64x128x64 and on 16x64x128); R2d per cell the life split "
            "lambda_med = (lambda_med - b / r_inf) + b / r_inf at its G's R2a r_inf, printed; R2e every READ w1 launch of "
            "blocks S and D (short K = H slabs from 32 KB at many n): n, b, lambda_med, t, kappa, printed; R2f PLAIN; "
            "PC's c0 (4.704722 us at G 128, PC-R2, board c9f689) printed beside, cross-board, never pooled"),
        "PC2-R3 single waves of 165/198 to 528 CTAs on the 4-per-SM tiles (unit sw; read_pc2.r3)": (
            "per tile (32x64x128 w4 s3 at G 264, 396, 462, 528, 594, 1056 long and 198 to 528 short; 32x128x64 w4 s3 at "
            "G 198, 264, 396, 462, 528, 594, 1056 long and 165 to 528 short; F 14336 and the short F 1024 and 512): per "
            "rung r, X, n, lambda_med, t, kappa (single-wave rungs), L_gpu = Q_B n / X (Q_B off the PTX), and the median "
            "over the rung's valid single-wave launches of the distinct SMs and of the most CTAs on one SM; X_sat on the "
            "tile, PC's definition: the median X over READ long-slab w2 rungs (w2 K >= 14336) strictly past one wave (G > "
            "528: 594, 1056) and READ flushed w1 launches with at least 528 live CTAs and K >= 4096, G 528's cell EXCLUDED "
            "(it is read against this X_sat), read with 2 or more members and the census reading 4 CTAs per SM, the pool's "
            "composition printed; at G 528: SATURATED when |X(528) / X_sat - 1| <= 3% (PC's SAT_WITHIN), else BELOW or "
            "ABOVE with the fraction; kappa(528) / kappa(396): HELD when within 3% of 1, else MOVED; NOT IDENTIFIED when "
            "either input is unread. A refused sw unit (regcheck or occupancy) leaves R3 NOT READ and R3p (PLAIN) the "
            "only reading"),
        "PC2-R4 the naive path (unit sn; read_pc2.r4)": (
            "per naive cell and GEMM (16x64x128 w4 s4, one M-tile per (token, expert) row): t, kappa, the median life, r "
            "over plateau CTAs, live CTAs, the slab, PLAIN; leaders and followers (a CTA whose expert sits on an earlier row "
            "of its launch is a follower): their median per-CTA rates, printed; the TWIN ratios (naive without followers "
            "against its aligned twin: the same shape, tile, active experts and live CTAs): t, PLAIN and the median life, "
            "naive over twin; the PAIR ratios (naive with followers against its naive pair: the same live CTAs, no "
            "follower, twice the distinct experts): t and PLAIN; each ratio read only when both cells' quantities are read. "
            "The SEEN cells' t and PLAIN printed beside the published decode blind's naive cell counts per model"),
        "PC2-R5 replicate": "R-D-G8, R-D-G128, R-S-G128-F512 against D-G8, D-G128, S-G128-F512: r and t ratios - 1, page noise",
        "PC2-R6 clock at kernel start and placement": (
            "PC-R13 (the median f_c of CTAs starting in the first 50 us over the rest's, at least 8 in each, flagged under "
            "0.97) on every valid launch of a valid unit, and PC-R14 (distinct SMs, the most CTAs on one SM, Spearman's "
            "rank of start order against pid) on call 0's w2 launch of every single-wave cell of a valid unit, printed"),
    }


def v6d_terms() -> dict:
    """How each of v6d's open terms (design-D6 section 7) will be read off PC2's pages. Registered before any page.
    Readings only: no form, no value; v6d stays as frozen (owner, 2026-11-10)."""
    return {
        "status": ("REGISTERED BEFORE ANY PC2 PAGE. v6d stays as frozen, with the duty factor (owner, 2026-11-10); its "
                   "law, choices D6-1 to D6-17 and FROZEN.txt hashes are not changed by PC2. PC2 registers which reading "
                   "fills which open slot of v6d's frozen law; it registers no form and no value. Anything addendum 6 does "
                   "with these readings is chosen after PC2's pages are read, and is disclosed as informed by them"),
        "1 the 16x64x128 law constants (v6d's slots for 16x64x128w4s4: Q_B, the L table, X_sat, kappa, c0)": (
            "Q_B: PC2-R1g's n_if on the stamped s4 PTX x 128 x 64 x 2 (v6d holds 48 KB, n_if 3, off 160 published PTX "
            "files). The L table on this tile: PC2-R1c's rows, built as v6d built its 64x128x64 table from PC (design-D6 "
            "section 2, the L row). X_sat on this tile: PC2-R1d (v6d holds PC's 64x128x64 level x DS's tile ratio, 3717.9 "
            "GB/s). kappa on this tile: PC2-R1e per rung (v6d holds kappa read on 64x128x64). c0 on this tile: PC2-R1b at "
            "G 128 (v6d holds PC's 4.70 us, read on 64x128x64). A reading that reads NOT IDENTIFIED leaves its slot as v6d "
            "froze it and says so; nothing is filled by a fallback"),
        "2 c0 and the span at short slabs (v6d's slots: c0, a constant; kappa(m), read on 3.67 MB slabs)": (
            "c0 at low load: PC2-R2a per G (8, 32, 128, 256) on 64x128x64 and PC2-R1b (G 8, 128) on 16x64x128; R2b's "
            "short-slab-only c0 beside. v6d's c0 slot is one constant: PC2 reports c0(G) as read; a load-dependent c0 "
            "would be a form change, not a value, and is the owner's. kappa at short slabs: PC2-R2c per (G, F), against "
            "v6d's kappa(m) read at 3.67 MB. The duty factor is v6d's (owner, 2026-11-10) and is not re-read"),
        "3 the big-single-wave behaviour (v6d: Little's law at X_sat past G 384, kappa held at its G 384 value 1.032)": (
            "PC2-R3 per 4-per-SM tile, a registered label (read_pc2.term3): HOLDS when G 528 reads SATURATED (|X(528) / "
            "X_sat - 1| <= 3%) and kappa(528) / kappa(396) reads HELD (|ratio - 1| <= 3%); DOES NOT HOLD when both are "
            "read and either is not; NOT IDENTIFIED when either is unread. DESCRIPTIVE: the label says whether v6d's "
            "assumption holds on PC2's pages, it changes nothing in v6d"),
        "4 the naive-path term (v6d: none; the naive path keeps v4d''s price)": (
            "PC2-R4's t, PLAIN, twin ratios and pair ratios per naive cell. STATED PLAINLY: v6d has no naive-path slot. "
            "The naive-path term addendum 6 may take will be CHOSEN AFTER PC2's pages are read, WITH THE SEEN DECODE "
            "BLIND IN VIEW (its 29 naive cells re-score at 8.17% rms under v6d, worst JetMoE U-T1 +22.2%, design-D6 "
            "section 6.2), and will be DISCLOSED AS SUCH: a choice informed by seen pages, never a blind test of itself. "
            "PC2 registers only the readings"),
        "v6d's frozen values printed beside (never used by a reading)": P.V6D_PRINTED,
    }


def stop_rules() -> dict:
    return {
        "prelude (C3's, unchanged)": (
            f"the session STOPS before any page unless the checkout is clean, sudo works, the GPU is idle, {P.POWER_W} W "
            "reads back on both enforced.power.limit and -q's GPU-scope Current Power Limit within 1 W (polled up to 30 s; "
            f"no fallback), persistence is Enabled and the session lock {P.SESSION_LOCK} MHz reads back; the prelude also "
            f"probes {P.DECODE_LOCK} MHz (lock, read back, reset)"),
        "unit locks (PC's, unchanged)": (
            f"every perturb and stamps unit carries lock-mhz={P.DECODE_LOCK}; it refuses unless the prelude's probe held "
            "it; it locks, reads back (two tries) and runs; after the tool the lock is read back again (pc-locks.env); the "
            "reader refuses a stamps unit whose own lock, or whose perturb unit's lock, did not read back after its tool"),
        "the regcheck drop rule (PC's owner decision 2, kept)": (
            "the probe=pc regcheck (unit 3) exiting anything but 0 or 1 (refused, crashed, stopped at its cap, skipped) or "
            "writing no regcheck line for a probe=pc stamps unit drops every probe=pc unit (4 to 10). PC2 has no other "
            "measuring unit, so a failed start leaves the census alone (about 30 min of boot, setup, prelude and census, "
            "about $1.15). Exit 1 (a variant's occupancy or path FAIL) refuses that variant's stamps unit by its line; its "
            "cells keep their PLAIN medians from the perturb unit. A cell failing its readback is excluded alone. The probe "
            "exits 4 on any uncaught error, never 0 or 1"),
        "the gate and the stamps (C3's)": (
            "a stamps unit runs only when its regcheck line and the gate's HARD line (occupancy, the all-off SASS, the "
            "upstream file, C3's SASS legs, every cell's readback) read PASS; a timing FAIL alone runs it PERTURBED"),
        "budget": ("the driver drops the LAST droppable unit first: rep, d3, then w; a unit whose estimate no longer fits "
                   "is SKIPPED; the guardian terminates at the hard cap"),
        "never": "no PC2 reading is a STOP; a reading that does not identify reads NOT IDENTIFIED",
    }


def seen_block() -> dict:
    return {
        "reserved (owner decision, 2026-11-06, kept 2026-11-10)": {
            "models": list(P.RESERVED),
            "rule": ("never loaded, never timed, never shaped: the probe refuses a variant, cell or weight build naming one, a "
                     "shape equal to one, or (PC2's stricter guard, pc2common.gemm_guard) any shape with a GEMM whose (N, K) "
                     "equals either GEMM of a reserved model; tests/test_instr_probe_pc2.py asserts nothing in the plan "
                     "loads them")},
        "loaded (all SEEN)": list(P.REAL_ALLOWED) + ["the registered synthetic shapes (PC's pcs- and PC2's pcs2-)"],
        "Granite-1B's regime": ("reached with synthetic shapes only (owner, 2026-11-10): short slabs of 66 to 459 KB at "
                                "n 8 to 528 on the four tiles Granite-3.0-1B's GEMMs use, never its shape and never its GEMMs"),
        "still unseen at vLLM's prefill tiles (PC's frozen literal, unchanged)": list(P.UNSEEN_AT_PREFILL_6C800F84),
        "statement": ("PC2 times none of the three models held for the new decode blind; Granite-3.0-3B runs at the decode "
                      "tile only, so it stays unseen at prefill tiles"),
    }


def build_decisions() -> list[str]:
    return [
        "item 1 (the decode tile): PC's own regcheck page (2026-11-06, regcheck-rcp) reads the stamped copy's occupancy "
        "equal to the plain kernel's at 16x64x128 s4 (3 = 3, both GEMMs) and s3 (5 = 5), and not at s2 (w2: 8 against 7, "
        "which refused PC's whole block D through one variant). PC2 stamps s4 in its own unit (sd, nodrop) and s3 in "
        "another (sd3, droppable), and leaves s2 out; the reader refuses any stamped compile whose CTAs per SM are not its "
        "tile's registered occupancy, on every compile (the naive specialisation included)",
        "the plain-kernel limit: every gated cell's plain-kernel event median comes from the perturb unit (PLAIN), "
        "printed beside every stamp reading and standing alone where a stamped variant is refused; its limit is stated "
        "(no per-CTA life, no c0 split, no kappa). No counter (ncu) unit is planned: no PC2 question needs a counter, "
        "PC's published counter page already holds D16 Mixtral T4 and T16, and the counter route has never run a naive "
        "cell on a GPU",
        "item 3: the 4-per-SM tiles' stamped occupancy is unknown until PC2's regcheck (PC never compiled them stamped); "
        "a FAIL refuses unit sw's stamps and leaves PLAIN alone for its cells, DESCRIPTIVE",
        "item 4: the naive cells run at vLLM's T <= 32 tile; each naive cell without followers has an aligned twin (the "
        "same active experts and live CTAs), each with followers a naive pair (the same live CTAs, no follower)",
        "item 5, the optional E 32 block-scan align page, is NOT BUILT: the block-scan path at E 32 declared needs T k >= "
        "1024 (private_weight_reference.align_path), the pinned decode page caps T at 128 (r3_decode.page_from), and no "
        "SEEN model with k >= 8 declares 32 experts (Granite-3.0-3B E 40, OLMoE E 64); a synthetic shape cannot enter the "
        "pinned counter step. The optional plain-kernel ladders are built in: PLAIN covers every cell at no extra unit",
        "PC's driver, probe and launcher are pinned by PC's registration (and vm_run.sh lists the drivers it may start), "
        "so PC2 runs on copies: scripts/gh200_pc2_session.sh (PC's driver; probe=pc runs scripts/instr_probe_pc2.py), "
        "scripts/instr_probe_pc2.py (PC's probe; its rules module is pc2common) and scripts/vm_run_pc2.sh (vm_run.sh; "
        "gh200_pc2_session.sh one of its drivers), each changed only on lines marked PC2 and held to its parent by a test",
        "the regcheck compiles ONE specialisation per tile per variant (the first cell's: on pc2-n, the naive "
        "specialisation of 16x64x128 s4; the aligned twins' specialisation is regchecked in pc2-d's variant, not in "
        "pc2-n's), as PC's probe does; every other stamped compile is held by the reader's occupancy leg, which FAILS "
        "CLOSED: a stamped compile whose CTAs per SM are not its tile's registered occupancy refuses the whole unit "
        "(NOT READ), after its GPU time is spent",
        "PC2's stricter shape guard (pc2common.gemm_guard): no PC2 GEMM, synthetic or SEEN, w1 or w2, has the (N, K) of "
        "either GEMM of a reserved model, so the Granite-1B regime is reached without Granite-1B's GEMM dimensions (the "
        "guard moved the 64x128x64 G 8 ladder from H 512 to H 256: pcs2-e8-h512-f512's w1 is (1024, 512), Granite-1B's w2)",
        "EXACT TIME and TIMER: PC's reader quantises every globaltimer stamp to 256 ns (float64 at 1.7e18 ns; the tick "
        "is 32 ns), which cost PC A-F7168-G8 and A-TP8-G128 (no resolution) and moves PC's block A median lives by up "
        "to 1.6% (median 0.07%, call 0, on PC's published pages; POST-HOC on SEEN pages, a defect noted, PC's pinned "
        "reader and its reading unchanged). PC2 rebases each unit's stamps before read_pc.launch and adds the unit's "
        "resolution as the timer's last source; PC2's few-CTA short-slab lives (about 3 to 8 us) are where 256 ns "
        "would have mattered most",
        "the probe writes stamps.json, the stamp arrays, the PTX and the regcheck and gate files atomically (PC's code, "
        "unchanged); the page generator and the registration write atomically too",
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
        "registered": f"{P.DATE}, before any PC2 page",
        "name": NAME,
        "status": ("MEASUREMENT, DESCRIPTIVE: PC2 certifies nothing, registers no model form, no constant, no bar and no "
                   "gate on any model, and moves no registered verdict. Every reading is DESCRIPTIVE"),
        "design": DESIGN,
        "motivation": (
            "v6d (design-D6, frozen 2026-11-09) cannot price four things its new blind needs (DESIGN.md section 7): the "
            "16x64x128 decode tile (56, 48 and 58 of the three new models' GEMMs; PC's block D was refused), short slabs "
            "of 66 to 262 KB at low n (74 of Granite-1B's 92 GEMMs, under PC's shortest single-wave slab of 459 KB; c0 is "
            "19% to 63% of those GEMMs), single waves of 384 to 528 CTAs on the 4-per-SM tiles, and the naive path (v4d''s "
            "price; the 29 naive cells of the seen decode blind re-score at 8.17% rms, which fails clause 2)"),
        "tool": ("scripts/gh200_pc2_session.sh (PC's pinned driver plus lines marked PC2), scripts/vm_run_pc2.sh (vm_run.sh "
                 "plus lines marked PC2), scripts/instr_probe_pc2.py (PC's pinned probe plus lines marked PC2), "
                 "moe/instrumented/fused_moe_instr.py (the copy, unchanged, STAMPS 3), scripts/tile_census_a2.py (the "
                 "census), scripts/scoring/pc2/read_pc2.py (the reader, over read_pc.py's pinned functions)"),
        "owner_decisions (2026-11-10)": [
            "run PC2 before building v6d's addendum 6",
            "v6d stays as frozen, with the duty factor",
            "Granite-3.0-1B stays in the blind (PC2 reaches its regime with synthetic shapes, never the model)",
            "measurement only; the owner writes kernels: no GPU kernel is written or edited, the instrumented copy carries "
            "only C3's existing stamps (ends), every new line is host-side"],
        "build decisions (to design-D6's sketch, before any page)": build_decisions(),
        "power_and_lock": {"power_limit_w": P.POWER_W, "session_lock_mhz": P.SESSION_LOCK, "decode_lock_mhz": P.DECODE_LOCK,
                           "per_cta_clock_band_printed": {str(k): list(v) for k, v in P.F_BAND.items()}},
        "instrument": {
            "copy": "moe/instrumented/fused_moe_instr.py, unchanged (no kernel written or edited; no new stamp point; no hint)",
            "copy_sha256": sha256(REPO / "moe/instrumented/fused_moe_instr.py"),
            "spec": "stamps=ends,evict_a=none,evict_b=none",
            "probe": ("scripts/instr_probe_pc2.py: PC's probe with pc2common as its rules module; imports instr_probe_c3.py "
                      "and instr_probe.py (both byte-identical)"),
            "per cell": ("PC's: the readback (V-BIN, V-ALIGN at the cell's config, the naive path's readback for a naive "
                         "cell), 3 warm-up calls, then 3 stamped calls (6 where the read GEMM has under 4 live CTAs), each "
                         "with its buffers reserved, the GPU held ahead and a 240 MiB L2 flush outside the launch event pair"),
            "weights": "one NATIVE copy per shape (synthetic or SEEN), private random weights at the shape"},
        "units": {"plan": P.PLAN, "rows": units},
        "cells (COMPUTED)": cells_table(),
        "pages": pages(),
        "shapes": {"file": P.SHAPES_FILE, "sha256": hashlib.sha256(P.shapes_text().encode()).hexdigest(),
                   "shapes": shapes["shapes"]},
        "readings": readings(),
        "v6d terms (how each open term will be read off PC2's pages)": v6d_terms(),
        "stop rules": stop_rules(),
        "census and occupancy": {
            "unit": "unit 2: tile_census_a2.py over PC2's 5 configs on Mixtral-8x7B and Phi-3.5 (compile only)",
            "registered occupancy": P.OCCUPANCY,
            "PC's regcheck on the decode tile (frozen literal, plain against stamped CTAs per SM)": {
                t: {g: list(v) for g, v in d.items()} for t, d in P.PC_D16_REGCHECK.items()},
            "regcheck scope": ("one specialisation per tile per variant is regchecked (the probe's rule, PC's); the "
                               "reader's occupancy leg checks every stamped compile and fails closed"),
            "rule": ("the wave-dependent readings (every X_sat pool, SPAN's single-wave rule, PC2-R3) read only where the "
                     "census reads the registered occupancy; a missing census or a path mismatch makes them NOT IDENTIFIED; "
                     "every stamped compile's CTAs per SM must equal the registered occupancy (PC2-R0)")},
        "page identity and versions": {
            "pages": "each stamps manifest's source_sha256 equals the registered page's",
            "versions": P.VERSIONS, "major_minor": list(P.VERSIONS_MAJOR_MINOR)},
        "constants": {"plateau_frac": P.PLATEAU_FRAC, "min_plateau_ctas": P.MIN_PLATEAU_CTAS, "spread_max": P.SPREAD_MAX,
                      "spread_max_small": P.SPREAD_MAX_SMALL, "spread_small_G": P.SPREAD_SMALL_G, "f_rungs": P.F_RUNGS,
                      "se_inv_rinf_max": P.SE_INV_RINF_MAX, "sat_within": P.SAT_WITHIN, "x_sat_w1_min_k": P.X_SAT_W1_MIN_K,
                      "l2_latency_ns": P.L2_LATENCY_NS, "l2_guard_stages": P.L2_GUARD_STAGES,
                      "overhead_median_max": P.OVERHEAD_MEDIAN_MAX, "overhead_worst_max": P.OVERHEAD_WORST_MAX,
                      "span_min_launches": P.SPAN_MIN_LAUNCHES, "timer_unit_min": P.TIMER_UNIT_MIN,
                      "x_sat_w2_min_k": P.X_SAT_W2_MIN_K, "kappa_held_within": P.KAPPA_HELD_WITHIN,
                      "start_window_ns": P.START_WINDOW_NS,
                      "start_flag": P.START_FLAG, "start_min_ctas": P.START_MIN_CTAS,
                      "rate_ceiling_gbps": P.RATE_CEILING_GBPS, "f_ceiling_mhz": P.F_CEILING_MHZ},
        "frozen literals": {"Granite-1B reach per tile (n min, n max, slab min KB, slab max KB; design-D6 reach6.out)":
                            {k: list(v) for k, v in P.GRANITE1B_REACH.items()},
                            "the seen decode blind's naive cells per model (design-D6 rescore_detail.out)":
                            P.BLIND_NAIVE_CELLS},
        "seen": seen_block(),
        "earlier registrations kept byte-identical": list(EARLIER),
        "plan": {"file": P.PLAN, "sha256": hashlib.sha256(P.plan_text().encode()).hexdigest(), "units_count": len(units),
                 "drop_order_last_first": order, **r},
    }


def txt(d: dict) -> list[str]:
    p = d["plan"]
    L = [f"REGISTERED {d['registered']}: rental PC2, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"motivation: {d['motivation']}", f"tool: {d['tool']}", "", "OWNER DECISIONS (2026-11-10)"]
    L += [f"  {x}" for x in d["owner_decisions (2026-11-10)"]]
    L += ["", "BUILD DECISIONS"] + [f"  {x}" for x in d["build decisions (to design-D6's sketch, before any page)"]]
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
    for sec in ("stop rules", "readings", "v6d terms (how each open term will be read off PC2's pages)"):
        L.append(sec.upper())
        for k, v in d[sec].items():
            L.append(f"  {k}: {v if isinstance(v, str) else json.dumps(v, sort_keys=True)}")
        L.append("")
    L.append("CELLS (COMPUTED)")
    for pg, rows in d["cells (COMPUTED)"].items():
        L.append(f"  {pg}:")
        for c in rows:
            part = f" twin {c['twin']}" if c["twin"] else (f" pair {c['pair']}" if c["pair"] else "")
            L.append(f"    {c['cell']}: {c['shape']} T {c['T']} {c['path']} A {c['A']} c_e {c['c_e']} {c['tile']} live w1 "
                     f"{c['live']['w1']} w2 {c['live']['w2']} slab w2 {c['slab_bytes']['w2']} B calls {c['stamped_calls']}"
                     f"{'' if c['slab_once'] else ' FOLLOWERS'}{part}")
    L += ["", "PAGES"] + [f"  {k}: {v['sha256']} ({len(v['cells'])} cells)" for k, v in d["pages"].items()]
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
