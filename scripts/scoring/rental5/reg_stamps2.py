"""Rental 5's timestamps-v2 registration (docs/registered/2026-10-07-rental5-stamps2-gh200).

    from reg_stamps2 import BUILDERS, TEXT     # register.py writes and --checks it

Design: scratchpad design-r5/DESIGN.md part 3, corrected by design-r5-review/REVIEW.md section
e, with the owner's decision 2 of 2026-10-07 (live and dead CTA sampling at 1 in 17, pid mod
17; the gate: identical CTAs per SM, timing median <= 1% and worst <= 2%; regcheck right after
calibrate, before any timed unit). Every number is computed here from committed files (the
timing model's geometry, rental 4's registered law parameters and RRZE constants, the
published rental-4 gpubench constants) or typed in with its provenance and label.

LEAKAGE: every rival constant is SEEN (CAL, SEEN-fitted, or SEEN third-party, per entry). No
stamp page of this kernel exists: rental 4's perturbation gate failed every stamped config, so
it published none. Every unit is on a CAL model (mixtral-8x7b, mixtral-8x22b, olmoe-1b-7b):
BLIND-CALMODEL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
C5 = None


def _c5():
    global C5
    if C5 is None:
        import importlib.util
        import sys
        name = "rental5_r5common"
        if name in sys.modules:
            C5 = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, HERE / "r5common.py")
            C5 = importlib.util.module_from_spec(spec)
            sys.modules[name] = C5
            spec.loader.exec_module(C5)
    return C5


PART = "stamps2"
#: the units, their cells and the readout blocks they carry (DESIGN 3.2; the corrected plan)
UNITS = {
    "stf": {"model": "mixtral-8x7b", "G": 8, "treads": [2, 4, 6], "arms": ["native"], "num_stages": 4,
            "blocks": ["F2", "K2"], "drop_group": "st", "variants": ["v1", "v2", "v3"]},
    "stt8x22": {"model": "mixtral-8x22b", "G": 8, "treads": [1], "arms": ["native"], "num_stages": 4,
                "blocks": ["K2"], "drop_group": "stt", "depends": "st", "variants": ["v1", "v2"]},
    "stt": {"model": "mixtral-8x7b", "G": 8, "treads": [1, 2], "arms": ["native"], "num_stages": 4,
            "blocks": ["K2"], "drop_group": "stt", "depends": "st", "variants": ["v1", "v2"]},
    "std4": {"model": "olmoe-1b-7b", "G": 8, "treads": [2, 4], "arms": ["native", "shared"], "num_stages": 4,
             "blocks": ["D2"], "drop_group": "std", "depends": "st", "variants": ["v2", "v3"]},
    "std2": {"model": "olmoe-1b-7b", "G": 8, "treads": [2, 4], "arms": ["native", "shared"], "num_stages": 8,
             "blocks": ["D2"], "drop_group": "std", "depends": "st", "variants": ["v2", "v3"]},
}
#: CTAs per SM (w1, w2) of the 32x64 w8 configs, the SEEN cubins (r4common.OCC)
OCC = {4: {"w1": 5, "w2": 4}, 8: {"w1": 2, "w2": 2}}
SAMPLE_MOD = 17
EVERY = 16
#: R(j) readout's admission (review e7): at least this many tail CTAs in a residency bin
RJ_MIN_CTAS = 30
#: the class rule's band at j = 2 and at occ
RJ_BAND = 0.03
#: d_meas's relative error the kappa reading carries beside its binomial one (review e6: the
#: design's +-0.013 is binomial only; with d's error the total is about +-0.02 at 690 sampled
#: hidden dead CTAs): 5%, the spread of rental 4's per-GEMM d (0.95 to 1.10 ns about 0.995)
#: divided by 3, a stated allowance and not a fit
D_REL_ERR = 0.05
SEP_SIGMA = 2.0
DISPATCH_MIN_CTAS = 100
DISPATCH_MIN_TICKS = 20


def _laws():
    c5 = _c5()
    r4reg = c5._load("rental4_register", HERE.parent / "rental4" / "register.py")
    return r4reg


def rj_rivals() -> dict:
    """R(j) = j c_law(BK 64, s4, j) cycles per k-step for a CTA sharing its SM with j - 1
    others (the closed queue at N = j for MVA2; PS is the model's CORES g(k) = k), j = 1 to 5;
    the same numbers as design-r5/work/rj_pred.txt, recomputed from rental 4's registered law
    parameters (r4common.law_c)."""
    r4 = _laws()
    C4 = _c5().C4
    out = {}
    for j in range(1, 6):
        row = {}
        for law in ("MVA2", "LK", "PS", "OCC"):
            c = C4.law_c(law, r4.LAWS[law], 64, 4, j)
            row[law] = None if c is None else round(j * c, 1)
        out[str(j)] = row
    return out


def geometry() -> dict:
    import r3_timing_model as TM
    out = {}
    for model in sorted({u["model"] for u in UNITS.values()}):
        old = TM.set_model(model)
        try:
            out[model] = {}
            for g in TM.GEMMS:
                from moe.instrumented import sample_ks
                S = TM.GEOMETRY[g].ksteps
                out[model][g] = {"S": S, "S_prime": round(TM.floor_ksteps(g), 6), "npn": TM.GEOMETRY[g].npn,
                                 "mark_k": sample_ks(S, EVERY)}
        finally:
            TM.set_model(old)
    return out


#: the occupancy bins (CTAs per SM of BK 64 s4, w1 / w2) the class rule reads, and the
#: occupancy Z divides by (w1's), registered here and read by the scorer (build-r5-review F7)
OCC_BINS = (4, 5)
OCC_Z = 5


def tail_table() -> dict:
    """Per K2 unit, cell and GEMM: live CTAs, the wave (132 x occ), the final partial wave's
    CTAs (the j < occ residency the tail readout bins) and how many of them a 1-in-17 iteration
    sample stamps (build-r5-review F7: under v2 the tail bins may not reach 30 CTAs)."""
    import r3_timing_model as TM
    out = {}
    for label in ("stt8x22", "stt", "stf"):
        u = UNITS[label]
        old = TM.set_model(u["model"])
        try:
            for n in u["treads"]:
                for g in TM.GEMMS:
                    live = TM.live_rows(n) * TM.GEOMETRY[g].npn
                    wave = 132 * OCC[4][g]
                    tail = live - (live // wave) * wave if live > wave else live
                    out[f"{label} n{n} {g}"] = {"live": int(live), "wave": wave, "tail_ctas": int(tail),
                                                "sampled_1_in_17": int(tail // SAMPLE_MOD)}
        finally:
            TM.set_model(old)
    return out


def stamps2() -> dict:
    c5 = _c5()
    C4 = c5.C4
    r4 = _laws()
    from moe import instrumented as I
    pub = (c5.REPO / "results" / "published" / "2026-10-07-nvidia_gh200_480gb-rental4-session" / "results"
           / "2026-10-06-nvidia_gh200_480gb-gpubench-r4" / "constants.json")
    gb = json.loads(pub.read_text())["parsed"]["gpu-latency"]
    lock = c5.LOCK_MHZ / 1e3
    near, far = gb["near_l2_ns"] * lock, gb["far_l2_ns"] * lock
    dram = r4.RRZE_GH200["dram_ns"] * lock
    F = dict(C4.F_CTA)
    occ4 = OCC[4]
    geo = geometry()
    rj = rj_rivals()
    tail = tail_table()
    c_ns = C4.C_NS
    olmoe = geo["olmoe-1b-7b"]
    # kappa's predicted resolution at the design's OLMoE w1 occupancy-5 count (11,000 hidden
    # dead CTAs, 1 in 17 sampled)
    n_hid = 11000 / SAMPLE_MOD
    sig_rel = math.sqrt(1 / n_hid + D_REL_ERR ** 2)
    return {
        "registered": f"{c5.DATE}, before any rental-5 page",
        "name": c5.NAMES[PART],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r5 DESIGN.md part 3 (3.1 gate, cadences, variants; 3.2 readouts, rivals, rules) with design-r5-review REVIEW.md section e (regcheck moved before any timed unit; pid mod 17; live-CTA sampling needs the owner; a deterministic variant rule; kappa's error with d's; a minimum per R(j) bin) and the owner's decision 2 of 2026-10-07 (live and dead CTA sampling at 1 in 17 approved; the gate below)",
        "tool": "scripts/instr_probe.py --mode regcheck, --mode perturb, --mode stamps; scorer scripts/scoring/rental5/score_stamps2.py",
        "instrument": {
            "copy": "moe/instrumented/fused_moe_instr.py: gate-fix.patch (level 3 'ends', the iteration end mark only at every = 1) plus level 4 'sample', the only kernel change: STAMPS == 4 blocks and constexpr sampling, no tiling or scheduling logic",
            "sample_level": "the CTA start, a dead CTA's exit, the end, and clock64 at the TOP of k-iteration 0 and of every k with k mod 16 = (S - 1) mod 16 (STAMP_PHASE, set per launch from K and BLOCK_K), so the last iteration S - 1 is always marked; no prologue, epilogue or end-of-iteration site",
            "sampling": f"constexpr moduli: a live CTA is stamped when pid mod STAMP_CTA_MOD is 0, its iteration tops when also pid mod STAMP_ITER_MOD is 0, a dead CTA when pid mod STAMP_DEAD_MOD is 0; every modulus 1 by default (levels 0 to 3 compile as before). pid mod {SAMPLE_MOD}, never 16: under the GROUP_M swizzle pid mod 16 = 0 stamps only pid_m offset 0 of every group, the slab-leading CTAs (design-r5-review work/sample_bias.txt)",
            "buffer": "every stamp buffer is allocated and filled with -1 BEFORE the L2 flush (moe.instrumented.BufferPool; gate-fix diagnosis item 4)",
            "mark_k": geo,
        },
        "cadences": {"cta": "CTA start and end (and a dead CTA's exit)", "iteration": f"a 1-in-{EVERY} iteration sample (tops only)",
                     "sample_level": "the phase-aligned sample level above, which is what F2 and K2 need (the last iteration's top)"},
        "variants": {"texts": dict(I.R5_VARIANTS),
                     "V1": "the sample level on every CTA", "V2": f"ends on every live CTA; iteration tops and dead CTAs 1 in {SAMPLE_MOD}",
                     "V3": f"everything on 1 in {SAMPLE_MOD} CTAs (live-CTA sampling approved by the owner, decision 2 of 2026-10-07)",
                     "choice_rule": "per unit, the FIRST of its listed variants (in the listed order) whose regcheck line (regcheck.env REGCHECK_<id>=PASS) AND perturbation gate line (gate.env GATE_<id>=PASS) both read PASS; deterministic. The driver records the chosen id and spec text in the unit's directory (CHOSEN_VARIANT.txt) and the manifest (stamps.json) carries the spec text; the scorer recomputes the choice off the two env files and refuses a unit whose chosen variant differs, or whose manifest spec differs from its variant's",
                     "per_unit": {k: v["variants"] for k, v in UNITS.items()},
                     "readout_needs": "R(j) needs every live CTA's start and end on its SM (residency): V1 or V2 only; under V3 K2's R(j) is NOT SCORED and only R(occ) is read"},
        "gate": {"regcheck": "scripts/instr_probe.py --mode regcheck, compile only, IMMEDIATELY after calibrate and before any timed unit (review e1): per variant, the all-off copy and the variant's own spec at the plain kernel's CTAs per SM on every config (registers and shared printed, not gating)",
                 "perturb": "median |plain / copy - 1| <= 1%, worst <= 2% over the variant's (cell, GEMM) ratios; CTAs per SM identical to the plain kernel's on every paired config (registers printed: ptxas reallocates the whole kernel around any stamp, gate-fix diagnosis item 2); the all-off copy's SASS equal to the plain kernel's; the installed vLLM fused_moe.py upstream's",
                 "tolerance": {"median_max": 0.01, "worst_max": 0.02, "occupancy": "identical CTAs per SM"},
                 "use": "a stamps unit runs only on its chosen variant, with both lines PASS; no variant passing: the unit is not run (NOT SCORED, the gate numbers printed)",
                 "priors": {"V1": "about 1.0 to 1.4%, likely FAIL", "V2": "about 0.8 to 1.2%, borderline", "V3": "about 0.1 to 0.2%, likely PASS",
                            "basis": "gate-fix.patch's cadence table, scaled from rental 4's measured per-site costs; ESTIMATES, not measured"}},
        "sass_precondition": {"rule": "r4common.sass_precondition on the unit's own SASS, per block: F2 and D2 a clock read before the first LDG (num_tokens) on both GEMMs; K2 at least 2 clock reads inside the k-loop; else NOT SCORED (the sample level has no epilogue stamp, so F2 needs no epi bracket)",
                              "blocks": {"F2": "D", "K2": "K", "D2": "D"}},
        "units": UNITS,
        "F2": {"units": ["stf"],
               "delta_epi": {"readout": "epi = end - top(S - 1) - T_iter per live CTA, T_iter = (top(S - 1) - top(previous mark)) / (k gap); median over live CTAs per launch; Delta_epi = median over n of [epi(w2) - epi(w1)], cycles",
                             "rivals_cycles": {"NEAR": round(near, 1), "FAR": round(far, 1), "DRAM": round(dram, 1)},
                             "rivals_source": "NEAR and FAR: rental 4's own gpubench on the board it ran (152.4 and 275.0 ns, SEEN, results/published 2026-10-07 rental4 gpubench-r4 constants.json), DRAM: RRZE's published GH200 346.4 ns (SEEN third-party, gpubench read no DRAM plateau); each x 1.710",
                             "rule": "r4common.nearest (linear), half-gap"},
               "t_fix": {"readout": "T_fix,g = median over STEADY-WAVE live CTAs (start order past the first wave of 132 x occ_g and before the final partial wave) of [lifetime - S_g T_iter,g], lifetime = end - start, cycles",
                         "rivals_cycles": {"w1": {"PS": F["w1"] * occ4["w1"], "LAT": F["w1"]}, "w2": {"PS": F["w2"] * occ4["w2"], "LAT": F["w2"]}},
                         "rivals_basis": "PS: F is shared work, so a CTA among occ pays occ F (occ 5 / 4); LAT: F is exposed latency, paid once (F_w1 520, F_w2 979 clk, CAL counters, CTA_FIXED_KSTEPS)",
                         "rule": "r4common.nearest in log space (a 4 to 5x gap)"},
               "printed": "the prologue is not stamped at this level (none is read)"},
        "K2": {"units": ["stt8x22", "stt", "stf"],
               "readout": "R = (top(k + 16) - top(k)) / 16 cycles per k-step on every 16-step window of every stamped live CTA (the tail wave's CTAs give j < occ, the steady waves j = occ), binned by j = round(the time-averaged number of live CTAs resident on the same smid over that window, the CTA itself included), from every live CTA's start and end stamps (clock64 is per SM); R(j) = the median per bin, both GEMMs pooled (BK 64 on both)",
               "rivals_cycles": rj,
               "rivals_basis": "R(j) = j c_law(BK 64, s4, j) with rental 4's registered law parameters (MVA2 closed queue at N = j; LK; PS = the model's CORES g(k) = k; OCC c0 + lam / j), SEEN-fitted or CAL per law (docs/registered/2026-10-06-rental4-occlaw-gh200); design-r5/work/rj_pred.txt",
               "sync_cost": "Z = R(1) - R(occ) / occ, printed",
               "classes": {"PS~LK": ["PS", "LK"], "MVA2": ["MVA2"], "OCC": ["OCC"]},
               "rule": f"r4common.nearest (linear) at j = 1; a class is SELECTED when it is nearest at j = 1 and within {int(100 * RJ_BAND)}% at j = 2 and at occ; PS and LK are one class (1.8% apart); MVA2 and OCC separate at j = 1 only (18%); else UNDECIDED",
               "bin_minimum": f"a bin with fewer than {RJ_MIN_CTAS} distinct stamped CTAs (not windows: one CTA gives many windows) is NOT SCORED (review e7, build-r5-review F7)",
               "bin_minimum_ctas": RJ_MIN_CTAS, "band": RJ_BAND,
               "occ_bins": list(OCC_BINS), "occ_z": OCC_Z,
               "occ_note": "the occupancy bins j = 4 and 5 (w2 and w1 at BK 64 s4) and Z = R(1) - R(5) / 5 are registered here; the scorer reads them",
               "tail_ctas": tail,
               "tail_printed_only": [k for k, v in tail.items() if v["sampled_1_in_17"] < RJ_MIN_CTAS],
               "tail_rule": "under a variant with iter_mod 17 (v2) the tail bins (j < 4) of every (unit, n, GEMM) whose registered sampled tail count (tail_ctas[].sampled_1_in_17) is under 30 are PRINTED ONLY, never scored: " + ", ".join(k for k, v in tail.items() if v["sampled_1_in_17"] < RJ_MIN_CTAS) + "; at 1 in 17 those tails cannot reach 30 CTAs, so their j < occ reading needs v1; the other launches' tails are counted under the 30-CTA minimum",
               "variants": "R(j < occ) exists under V1 and V2 only (residency needs every live CTA's start and end); under V3 only R(occ) is read and the class rule is NOT SCORED",
               "prior": "the rental-1 counters' g_w1(1) = 0.76 (SEEN) is the PS-side prior",
               "bk128": "MVA2's Z is 0 at BK 128: no lever there, no BK 128 K unit"},
        "D2": {"units": ["std4", "std2"],
               "occupancy": {"std4": OCC[4], "std2": OCC[8]},
               "life": {"readout": "median over sampled dead CTAs of end - start, cycles", "rivals_cycles": {"FAST": 150.0, "MIX": 451.0, "DRAM": round(r4.RRZE_GH200["dram_ns"] * lock + 75, 1)},
                        "rule": "r4common.nearest (linear), as rental 4"},
               "dispatch": {"readout": f"SHARED cells: the sampled dead CTAs that start after the last stamped live CTA ends; (max - min of their start globaltimer) / (count - 1) / {SAMPLE_MOD} (consecutive sampled dead CTAs are {SAMPLE_MOD} pids apart), ns per dead CTA",
                            "admission": f"at least {DISPATCH_MIN_CTAS} sampled dead CTAs over at least {DISPATCH_MIN_TICKS} globaltimer ticks, else NOT SCORED",
                            "rivals_ns": {"2": {"DISP": 0.995, "MIX": round(0.53 + 264.0 / (c5.SMS * 2), 3), "SLOT": round(4.05 / 2, 3)},
                                          "5": {"DISP": 0.995, "MIX": round(0.53 + 264.0 / (c5.SMS * 5), 3), "SLOT": round(4.05 / 5, 3)},
                                          "4": {"DISP": 0.995, "MIX": round(0.53 + 264.0 / (c5.SMS * 4), 3), "SLOT": round(4.05 / 4, 3)}},
                            "rule": "r4common.nearest (linear) at the GEMM's own occupancy; at occupancy 5 and 4 they read BETWEEN by design; per-GEMM d (Dg) printed",
                            "cta_mod_rule": "dispatch and kappa are NOT SCORED when the unit's variant samples live CTAs (cta_mod != 1, v3): the last live end is then read off sampled CTAs only, which biases both (build-r5-review F2); life is still scored"},
               "hiding": {"readout": f"kappa_meas = {SAMPLE_MOD} N_hid,sampled d_meas / (occ S' c): N_hid,sampled the sampled dead CTAs that start BEFORE the last live end, d_meas the dispatch readout of the same GEMM (ns), S' = S + PHI (r3_timing_model.floor_ksteps), c = {c_ns} ns per k-step (CAL, 8x7B fit)",
                          "rivals": {"D": 0.325, "D2": 0.30, "SLOT": 0.245, "K5": 0.5, "K0": 0.0},
                          "error": f"sigma_kappa = kappa sqrt(1 / N_hid,sampled + {D_REL_ERR}^2): binomial AND d_meas's error (review e6); at the design's 11,000 hidden OLMoE w1 dead CTAs ({n_hid:.0f} sampled) that is +-{0.325 * sig_rel:.3f}",
                          "kappa_source": "the only kappa reading of rental 5: the tp2 lever (9 against 32 copies) was CUT by the owner on 2026-10-07",
                          "rule": f"nearest (linear); SELECTED only when the second-nearest rival is at least {SEP_SIGMA:g} sigma_kappa from the reading, else NOT SEPARATED (nearest and second named); D against D2 is printed NOT SEPARATED whatever the reading (0.025 apart, about 1 sigma)",
                          "S_prime": {g: olmoe[g]["S_prime"] for g in ("w1", "w2")},
                          "c_ns": c_ns, "d_rel_err": D_REL_ERR, "sample_mod": SAMPLE_MOD, "separation_sigma": SEP_SIGMA}},
        "secondary": "calibration line and gamma: NOT APPLICABLE on every block here (fewer than 3 predicted values per block: Delta_epi, T_fix, R(j), life, dispatch, kappa); top-1 regret NOT APPLICABLE (one tile)",
        "missing": "a unit with no stamps.json, no PASS variant or a variant mismatch is NOT SCORED; drop-groups st (perturb, stf), stt and std (both depend on st)",
        "labels": {k: "BLIND-CALMODEL (a CAL model; no stamp page of this kernel exists)" for k in UNITS},
        "seen_data": ["every rival constant is SEEN: F_w1 520 / F_w2 979 and c 202.55 ns CAL; the k-step law parameters SEEN-fitted (rental 4 occlaw); NEAR / FAR SEEN (rental 4's gpubench); DRAM SEEN third-party (RRZE); MIX's d0 0.53 ns and L 264 ns SEEN-fitted; D's d 0.995 / kappa 0.325 CAL-counters (4 models); D2 SEEN-fitted on 16 GEMMs; SLOT's t 4.05 ns CAL refit",
                      "no stamp page exists: rental 4's gate failed every stamped config and published none"],
    }


def text(d: dict) -> list[str]:
    out = ["variants (first passing regcheck and perturb, per unit):"]
    out += [f"  {k}: {v}" for k, v in d["variants"]["texts"].items()]
    out.append("R(j) rivals, cycles per k-step at BK 64 s4: " + json.dumps(d["K2"]["rivals_cycles"]))
    out.append("Delta_epi rivals: " + json.dumps(d["F2"]["delta_epi"]["rivals_cycles"]))
    return out


BUILDERS = {PART: stamps2}
TEXT = {PART: text}
