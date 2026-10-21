#!/usr/bin/env python
"""The calibration rental C2's reader (docs/registered/2026-10-22-cal2-gh200): the four-model held-out table.

    python scripts/scoring/cal2/read_c2.py --c2 results/published/<date>-nvidia_gh200_480gb-cal2-session \\
        [--out-dir scripts/scoring/cal2]          # writes cal2.read.json (sorted keys) and cal2.read.txt

C2 is a CALIBRATION rental: it certifies nothing, and every C2 page becomes SEEN for v5'' and B3 once read. This
reader prices C2's Mixtral-8x22B pages with v5'' and the averaged-fit sag estimator EXACTLY AS FROZEN at the
registration (the knee round's code, moe-kernels-work 2026-10-22 knee-v5/work and what it imports; every file is
pinned by sha256 in the registration and checked here before anything is imported), and adds 8x22B as the fourth
held-out model beside the three CAL models' registered held-out means.

THE PRIMARY READING (C2-R1). X = 8x22B's held-out 128 G 1 mean, in percent, e = T_model / T_meas - 1:
  cells    C2-T's NATIVE cells (docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json, n 2, six labels), the
           page admitted by tcommon.timed_page (G5 provenance, the r7-median-run3 rebuild, the ALL view) and each row
           kept unless tcommon.row_drop drops it against v4's price (v5price.e1cal_rows' loop, A2's selections)
  model    v5'' at its frozen constants: the null knee form at the carried bw 3597.8 GB/s, p_128 5.75 (the SEEN fit
           on all three CAL models' counter rows: 8x22B is in none of them, so this IS the held-out fold), M4 bytes
           with the v3 key, F4 x 2, R3's 1410 share (6.11 us, 50 us) in T0: knee-v5/work/k5.py cell_ms(r, None,
           5.75, F1, None)
  F1       the averaged-fit sag estimator ("par", knee-v5/work/sagtau.py estimators): one (A, tau) per C2 estimator
           page (OLMoE f128 and Mixtral-8x7B f128 on C2's board, sagtau.fit over design-v5/work/sag.py pts_of at
           1410), A and tau averaged. With 8x22B held out no estimator page is dropped (it has none).
  X        k5.summ's model_class_means["mixtral-8x22b 128g1"] (v5prime.summ: the mean of e over the kept cells, in
           percent, rounded to 0.01)
THE TABLE: the three CAL models' registered held-out means (v5'' + F1 par, LOMO, knee-v5 sagtau.json:
OLMoE +0.13, Qwen2-57B -0.82, Mixtral-8x7B +3.13) and X. THE PASS ESTIMATE: power (diag-v5/work/power5.py's power,
ported here verbatim and held to x22.json's twelve bounds bit for bit) on B2's registered 84 cells, as knee-v5
x22.py: A (four models: b_class the mean of the four, sd their SD), B (no common bias, sd their RMS), D (OLMoE,
Qwen2-57B and X: Mixtral-8x7B's G 1 read as an isolated outlier). THE BAND (registered): CLEAN, PARTIAL, REGIME,
OPPOSITE or NOT READ (`outcome`).

SECONDARY READINGS (DESCRIPTIVE; they explain X, they change no band):
  C2-S1  8x22B's x per GEMM on its own cells (bytes-only over floor-only time, v5'' terms): is it in the regime?
  C2-S2  8x22B b128g1: counted bytes against M4's (v3 key) reads, and the cycles-domain error of the v5'' knee at
         p 5.75 per GEMM (k5.capture_rows / eval_rows, null form, carried bw), held out
  C2-S3  8x22B's own sag on its b128g1 page against the C2 estimator (sag domain, us)
  C2-S4  Mixtral-8x7B b128g1 on the A2 cell set, C2's board: the cycles-domain error at p 5.75 per GEMM beside the
         same number on S1's b128g1 page (board 9da3c0, the only Mixtral 128 G 1 counter page so far) and the
         time-domain held-out +3.13 of A2's timed page of the SAME cells
  C2-S5  the board: C2's per-page (A, tau) beside A2's (29.65 / 300, 54.39 / 550), the power read-back, the lock
         probe and G-BAND (and which band passed); 8x22B's compiled CTAs per SM at 128x128 (the tc2 census) beside
         the 2 per SM v5'' prices; calibrate's triad ruler beside A2's 3726.2 GB/s
  C2-S6  Mixtral-8x7B timed on A2's E1-CAL cells, C2's board, in the fold A2's +3.13 was read in (p 5.25, F1 from
         C2's OLMoE f128 alone): its held-out mean beside +3.13, and the GEMM against per-call split on one board
         (this page's time error against C2-M's cycles error, cell by cell)
Beside X (DESCRIPTIVE): X over A2's four labels, and X priced with A2's board sag (42.02 us, 425 us).
THE FREEZE IS TIED TO THE RENTAL'S COMMIT: `read` takes the commit from the ledger's driver start line and refuses
unless that commit holds the working copies of the registration, the page, the plan and every code_pins path, the
code_pins hold here, and every repo module the chain loads from its repo is pinned with its registered bytes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import subprocess
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORK = Path.home() / "code" / "moe-kernels-work"
KNEE = WORK / "2026-10-22" / "knee-v5" / "work"
REG_NAME = "2026-10-22-cal2-gh200"
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
B2_REG = REPO / "docs" / "registered" / "2026-10-20-tileB2-e1tile-gh200.json"
HIST = "docs/registered/2026-10-22-cal2-hist"

#: every work file the frozen chain imports or reads (found by tracing k5, sagtau and a priced cell), and the two
#: knee-round outputs this reader takes numbers from; the registration pins each by sha256
FROZEN_FILES = (
    "2026-10-20/scoreA2/bytes.score.json", "2026-10-20/scoreA2/census.score.json",
    "2026-10-20/scoreA2/floor.score.json", "2026-10-20/scoreA2/rep.score.json",
    "2026-10-21/diag-A2/work/common.py",
    "2026-10-21/design-v5/work/v5price.py", "2026-10-21/design-v5/work/kneefit.py",
    "2026-10-21/design-v5/work/sag.py", "2026-10-21/design-v5/work/b2risk.py",
    "2026-10-21/design-v5/work/f3fit.json", "2026-10-21/design-v5/work/kneefit.json",
    "2026-10-21/design-v5/work/kneefit64.json",
    "2026-10-22/diag-v5/work/d5.py", "2026-10-22/diag-v5/work/bw.py", "2026-10-22/diag-v5/work/knee2.py",
    "2026-10-22/diag-v5/work/v5prime.py", "2026-10-22/diag-v5/work/bytes_lomo.json",
    "2026-10-22/diag-v5/work/power5.py",
    "2026-10-22/knee-v5/work/k5.py", "2026-10-22/knee-v5/work/sagtau.py",
    "2026-10-22/knee-v5/work/sagtau.json", "2026-10-22/knee-v5/work/x22.json",
)

# ------------------------------------------------------------------ frozen constants (registered by value)
MODEL = "mixtral-8x22b"
CAL_MODELS = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b")
#: v5'' + F1 par, LOMO held-out 128 G 1 means in percent (knee-v5 sagtau.json e1["v5'' (null, carried)"]["par"])
CAL_HELD_OUT = {"olmoe-1b-7b": 0.13, "qwen2-57b-a14b": -0.82, "mixtral-8x7b": 3.13}
P128, P64 = 5.75, 3.75
BW_CARRIED_GBPS = 3597.8
LOCK = 1410
#: A2's per-page (A us, tau us) at 128x128 (knee-v5 sagtau.json per_page_fits SEEN) and their average ("par")
A2_PAGE_FITS = {"olmoe-1b-7b": (29.65, 300.0), "mixtral-8x7b": (54.39, 550.0)}
A2_PAR_128 = (42.02, 425.0)
#: C2's estimator pages (the plan's labels) and the units it reads
EST_UNITS = {"olmoe-1b-7b": "f128", "mixtral-8x7b": "f128"}
T_LABEL, B_LABEL, M_LABEL, MT_LABEL = "t128g1", "b128g1", "b128g1", "t128g1m"
MIX_HIST = "docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json"
#: Mixtral-8x7B's LOMO fold in knee-v5 (forms.json "v5'' (null, carried bw)" LOMO mixtral-8x7b: p 5.25, F1 from the
#: OLMoE estimator page alone): the fold A2's +3.13 was read in, applied to C2's Mixtral timed page (C2-S6)
MIX_FOLD_P = 5.25
#: C2-S6's descriptive cut (points)
S6_NEAR = 1.0
#: A2's four E1-CAL labels (X4 beside the six-cell X, DESCRIPTIVE)
A2_LABELS = ("PT", "PW", "uniform", "balanced")
#: the triad ruler: A2's calibration page and the 15 calibrated rentals' range (knee-v5 section 1, diag-v5 item 2)
TRIAD_A2_GBPS, TRIAD_RANGE_GBPS = 3726.2, (3725.6, 3727.6)
#: every repo module the frozen chain imports from the repo it runs (diag-A2 common.REPO), traced over k5, sagtau, a
#: priced cell, a timed page, a counter page, a sag fit and the census reading; the registration pins each
CHAIN_REPO_MODULES = (
    "moe/__init__.py", "moe/baselines/__init__.py", "moe/baselines/_framework_config.py", "moe/bench/__init__.py",
    "moe/bench/ai_model.py", "moe/bench/calibrate.py", "moe/bench/counter_probe_kernel.py", "moe/bench/exit_codes.py",
    "moe/bench/provenance.py", "moe/bench/published.py", "moe/bench/roofline.py", "moe/bench/schema.py",
    "moe/bench/timing.py", "moe/bench/weights.py", "moe/pipeline.py", "moe/quant.py", "moe/routing/__init__.py",
    "moe/routing/distributions.py", "moe/spec.py", "moe/stages.py", "moe/state.py",
    "scripts/atile_pooled_fit.py", "scripts/block_m_crossing_sweep.py", "scripts/clock_elasticity.py",
    "scripts/cores_heldout_predict.py", "scripts/cross_model_predict.py", "scripts/dram_counter_route.py",
    "scripts/lock_gate.py", "scripts/per_tile_model_fit.py", "scripts/private_weight_reference.py",
    "scripts/r3_timing_model.py", "scripts/r3_timing_model_v3.py", "scripts/r3_timing_model_v4.py",
    "scripts/r3_timing_model_v4L.py", "scripts/ruler_rebaseline.py", "scripts/scoring/rental2/common.py",
    "scripts/scoring/rental3/r3common.py", "scripts/scoring/rental4/r4common.py", "scripts/scoring/rental5/r5common.py",
    "scripts/scoring/rental5/skewmodel.py", "scripts/scoring/rental6/r6common.py", "scripts/scoring/rental6/r6model.py",
    "scripts/scoring/rental6/reg6.py", "scripts/scoring/rental7/r7common.py", "scripts/scoring/tileA/finalise.py",
    "scripts/scoring/tileA/power_tile.py", "scripts/scoring/tileA/regtile.py", "scripts/scoring/tileA/tcommon.py",
    "scripts/scoring/tileA2/a2common.py", "scripts/skew_synth.py", "scripts/skew_synth_tile.py",
    "scripts/wave_split_bytes.py", "scripts/wave_split_bytes_v3.py", "scripts/wave_split_bytes_v4.py",
)
#: the repo files tied to the rental's commit besides code_pins (the registration, the page, the plan)
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt",
                "docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json", MIX_HIST,
                "scripts/plans/rentalC2-cal-2026-10.plan")
#: the band (registered): X in percent
BAND = {"clean": 1.0, "regime": 2.0, "opposite": -1.0}
MIN_CELLS = 4
#: C2-S4's descriptive cut on the w1 cycles means (points)
S4_NEAR = 2.0
POWER_W = 700
NOISE = {"sig_byte": 0.0175, "sig_cell": 0.006, "sig_page": 0.0018, "sig_board": 0.0026}
CLASSES = ("64g1", "128g1", "128g16")
BLIND = ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=str) + "\n"


# ================================================================== the pure part (no frozen chain)

def power(cells, bclass, sd_model, R=200_000, chunk=25_000, seed=20261022):
    """diag-v5/work/power5.py's power, verbatim: P(E1-TILE HOLDS) under B2's registered truth model."""
    models = sorted({c["model"] for c in cells})
    pages = sorted({c["page"] for c in cells})
    mi = np.array([models.index(c["model"]) for c in cells])
    ci = np.array([CLASSES.index(c["cls"]) for c in cells])
    pi = np.array([pages.index(c["page"]) for c in cells])
    s = np.array([float(c["s"]) for c in cells])
    rng = np.random.default_rng(seed)
    ok_n = {"E1-TILE": 0, **{m: 0 for m in models}, "pooled": 0}
    done = 0
    while done < R:
        B = min(chunk, R - done)
        sp = rng.standard_normal((B, len(models), 3)) * sd_model
        e = (np.asarray(bclass)[ci][None, :] + sp[:, mi, ci] + rng.standard_normal((B, len(cells))) * NOISE["sig_cell"]
             + rng.standard_normal((B, len(cells))) * NOISE["sig_byte"] * s[None, :]
             + rng.standard_normal((B, len(pages)))[:, pi] * NOISE["sig_page"] + rng.standard_normal((B, 1)) * NOISE["sig_board"])
        r = np.exp(-e) - 1          # pred / meas - 1

        def bar(x):
            return (np.sqrt((x ** 2).mean(1)) <= 0.02) & (np.abs(x).max(1) <= 0.05) & (np.abs(x.mean(1)) <= 0.01)
        pooled = bar(r)
        ok = pooled.copy()
        ok_n["pooled"] += int(pooled.sum())
        for j, m in enumerate(models):
            k = bar(r[:, mi == j])
            ok_n[m] += int(k.sum())
            if m in BLIND:
                ok &= k
        ok_n["E1-TILE"] += int(ok.sum())
        done += B
    return {k: round(v / R, 4) for k, v in ok_n.items()}


def b2_cells() -> list:
    return json.loads(B2_REG.read_text())["power"]["cells"]


def to_b(e_pct: float) -> float:
    return -math.log(1 + e_pct / 100)


def pass_estimates(X: float, cells=None, base=None, R=200_000) -> dict:
    """knee-v5 x22.py's three estimates at a fourth held-out mean X (percent)."""
    cells = b2_cells() if cells is None else cells
    base = [CAL_HELD_OUT[m] for m in CAL_MODELS] if base is None else list(base)
    b = [to_b(x) for x in base + [X]]
    A = power(cells, [st.fmean(b)] * 3, st.stdev(b), R=R)["E1-TILE"]
    Bp = power(cells, [0.0] * 3, math.sqrt(st.fmean(x * x for x in b)), R=R)["E1-TILE"]
    three = [to_b(x) for x in (base[0], base[1], X)]
    D = power(cells, [st.fmean(three)] * 3, st.stdev(three), R=R)["E1-TILE"]
    return {"A (4 models)": A, "B (4 models)": Bp, "D (O, Q, X; 8x7B G1 aside)": D}


def outcome(X: float | None, n_cells: int, valid: bool = True, why: str = "") -> dict:
    """The registered band of the primary reading and what it means for the B3 decision."""
    if not valid or X is None or n_cells < MIN_CELLS:
        return {"band": "NOT READ", "why": why or f"{n_cells} kept 8x22B cells (at least {MIN_CELLS} needed)",
                "b3": "no fourth held-out model: the B3 decision stands where knee-v5 left it (option (a) or (b) again)"}
    if abs(X) <= BAND["clean"]:
        return {"band": "CLEAN", "b3": ("Mixtral-8x7B's G 1 miss reads as isolated: B3 may be registered with v5'' and the "
                                        "averaged-fit sag estimator as frozen here, the estimate D disclosed beside A. The "
                                        "four-model A stays at or under 0.09 at every X (0.086 at X 0): CLEAN changes which "
                                        "estimate is credible (D over A), not A")}
    if X >= BAND["regime"]:
        return {"band": "REGIME", "b3": ("the large-K, x near 1, G 1 miss is a regime or E 8 family property (8x22B shares "
                                         "Mixtral-8x7B's E 8, top_k 2; Phi-3.5, E 16, only partly inherits it): B3 is not "
                                         "registered with v5''; the next step is a four-model refit (8x22B SEEN), then a new "
                                         "held-out check")}
    if X < BAND["opposite"]:
        return {"band": "OPPOSITE", "b3": ("8x22B misses the other way: the 128 G 1 error is model-specific in both "
                                           "directions; A and B apply, not D; the owner decides (go blind, refit, or neither)")}
    return {"band": "PARTIAL", "b3": ("between the bounds: neither going blind nor a refit is indicated by this reading; "
                                      "the owner decides with A, B and D at X")}


def s4_reading(c2_w1: float | None, s1_w1: float | None) -> str:
    """C2-S4 (DESCRIPTIVE): Mixtral-8x7B's 128 G 1 w1 cycles error on C2's board against S1's (points). BETWEEN only
    between the two ends (0 and S1's reading, each widened by S4_NEAR), BEYOND-S1 past S1's, OPPOSITE past 0 on
    the far side."""
    if c2_w1 is None or s1_w1 is None:
        return "NOT READ"
    if abs(c2_w1 - s1_w1) <= S4_NEAR:
        return "SAME-AS-S1"
    if abs(c2_w1) <= S4_NEAR:
        return "BOARD-SPECIFIC"
    sign = 1.0 if s1_w1 >= 0 else -1.0
    c, s = sign * c2_w1, sign * s1_w1
    if S4_NEAR < c < s - S4_NEAR:
        return "BETWEEN"
    return "BEYOND-S1" if c > s else "OPPOSITE"


def s6_reading(m: float | None, a2: float | None = None) -> str:
    """C2-S6 (DESCRIPTIVE): Mixtral-8x7B's held-out 128 G 1 mean on C2's board against A2's (+3.13), percent.
    REPRODUCES within S6_NEAR of A2's; BOARD within S6_NEAR of 0; BETWEEN between the two; ABOVE past A2's;
    OPPOSITE past 0 on the far side."""
    a2 = CAL_HELD_OUT["mixtral-8x7b"] if a2 is None else a2
    if m is None:
        return "NOT READ"
    if abs(m - a2) <= S6_NEAR:
        return "REPRODUCES"
    if abs(m) <= S6_NEAR:
        return "BOARD"
    if S6_NEAR < m < a2 - S6_NEAR:
        return "BETWEEN"
    return "ABOVE" if m > a2 else "OPPOSITE"


def heldout_table(X: float | None) -> dict:
    t = {m: {"held_out_128g1_mean_pct": CAL_HELD_OUT[m], "source": "registered (knee-v5 sagtau.json, LOMO, v5'' + F1 par)"}
         for m in CAL_MODELS}
    t[MODEL] = {"held_out_128g1_mean_pct": X, "source": "C2-R1 (this reader)"}
    v = [x["held_out_128g1_mean_pct"] for x in t.values() if x["held_out_128g1_mean_pct"] is not None]
    t["max |mean|"] = round(max(abs(x) for x in v), 2)
    t["spread"] = round(max(v) - min(v), 2)
    return t


def cell_summary(es: list) -> dict:
    e = [100 * x for x in es]
    if not e:
        return {"n": 0}
    return {"n": len(e), "mean_pct": round(st.fmean(e), 2), "rms_pct": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "worst_pct": round(max(e, key=abs), 2)}


def board(session: Path | None) -> dict:
    """C2-S5's preliminaries from the session directory: the power read-back, the session lock's probe, G-BAND."""
    if session is None or not Path(session).is_dir():
        return {"valid": False, "why": "no session directory"}
    D = Path(session) / "gh200-driver"
    out, why = {}, []
    pw = (D / "c2-power.env").read_text() if (D / "c2-power.env").is_file() else ""
    out["c2_power_env"] = pw.strip()
    if f"C2_POWER_W={POWER_W}" not in pw.split():
        why.append(f"c2-power.env does not record C2_POWER_W={POWER_W}")
    locks = (D / "locks.env").read_text() if (D / "locks.env").is_file() else ""
    out["lock_probe"] = next((ln for ln in locks.split() if ln.startswith(f"LOCK_PROBE_{LOCK}=")), None)
    if f"LOCK_PROBE_{LOCK}=held" not in locks.split():
        why.append(f"locks.env has no LOCK_PROBE_{LOCK}=held")
    g = D / f"gate-band-{LOCK}.json"
    gj = json.loads(g.read_text()) if g.is_file() else {}
    gv = gj.get("verdict")
    out["g_band"] = gv
    out["g_band_which"] = (gj.get("band_in_force") or {}).get("which")
    if gv != "PASS":
        why.append(f"G-BAND reads {gv!r}")
    out["valid"] = not why
    out["why"] = "; ".join(why)
    return out


# ================================================================== the frozen chain

def frozen_hashes() -> dict:
    return {f: sha256(WORK / f) for f in FROZEN_FILES}


def check_frozen(reg: dict) -> None:
    want = reg["frozen"]["files"]
    got = frozen_hashes()
    bad = [f for f in FROZEN_FILES if want.get(f) != got[f]]
    if bad or set(want) != set(FROZEN_FILES):
        raise SystemExit(f"REFUSED: the frozen chain is not the registered one: {bad or sorted(set(want) ^ set(FROZEN_FILES))}")


_CHAIN: dict = {}
#: the module names the frozen chain imports from the work folders
CHAIN_MODULES = ("common", "v5price", "kneefit", "sag", "b2risk", "d5", "bw", "knee2", "v5prime", "k5", "sagtau")


def chain():
    """k5 and sagtau (knee-v5/work), imported once; k5 builds the pricer at p 14 and asserts T0 and bw."""
    if not _CHAIN:
        # the chain's modules have generic names (common, sag, bw, ...) that other repo modules share (rental2's
        # common): import them with any such entry set aside and sys.path as it was, then restore both
        keep_path = list(sys.path)
        keep_mods = {m: sys.modules.pop(m) for m in CHAIN_MODULES if m in sys.modules}
        try:
            sys.path.insert(0, str(KNEE))
            import k5  # noqa: E402
            import sagtau  # noqa: E402
        finally:
            for m in CHAIN_MODULES:
                sys.modules.pop(m, None)
            sys.modules.update(keep_mods)
            sys.path[:] = keep_path
        assert abs(k5.BW_CARRIED - BW_CARRIED_GBPS) < 0.1 and k5.d5.P5 == {128: P128, 64: P64}
        _CHAIN.update(k5=k5, sagtau=sagtau)
    return _CHAIN["k5"], _CHAIN["sagtau"]


def page_files(results: Path, model: str, unit: str, lock: int = LOCK) -> list:
    return sorted(Path(results).glob(f"*-{model}-{unit}-r3-counters/lock{lock}/r3c-g*.json"))


def check_counter_page(results: Path, model: str, unit: str, *, G: int, reg_file: dict | None) -> tuple:
    """(path, why): one page, the 128x128x64 w8 s3 tile at G, and when reg_file is given its histogram's sha256,
    shuffle seed and every cell's counts sha256."""
    f = page_files(results, model, unit)
    if len(f) != 1:
        return None, f"{len(f)} {model} {unit} counter pages at lock{LOCK}"
    p = json.loads(f[0].read_text())
    de = p["design"]
    got = [int(de.get(k) or 0) for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m")]
    if got != [128, 128, 64, 8, 3, G]:
        return None, f"{f[0].name}: tile and G {got}, not 128x128x64 w8 s3 G {G}"
    if reg_file is not None:
        h = de.get("histogram") or {}
        if h.get("file_sha256") != reg_file["sha256"] or h.get("shuffle_seed") != reg_file["shuffle_seed"]:
            return None, f"{f[0].name}: histogram sha256 or shuffle seed is not the registered one"
        want = {(c["label"], int(c["n"])): c["counts_sha256"] for c in reg_file["cells"]}
        for c in p["cells"]:
            if c["arm"] == "private":
                continue
            if want.get((str(c.get("histogram") or "balanced"), int(c["n"]))) != c.get("counts_sha256"):
                return None, f"{f[0].name}: cell n {c['n']} {c.get('histogram')} carries another counts sha256"
    return f[0], ""


def sag_fits(results: Path) -> dict:
    """{model: (A us, tau us)} per C2 estimator page (sagtau.fit over sag.pts_of at 1410) and their average."""
    k5, ST = chain()
    per = {}
    for m, unit in EST_UNITS.items():
        files = page_files(results, m, unit)
        if len(files) != 1:
            raise LookupError(f"{len(files)} {m} {unit} pages")
        per[m] = ST.fit(k5.S.pts_of(files, float(LOCK)))
    par = (st.fmean(A for A, _ in per.values()), st.fmean(t for _, t in per.values()))
    return {"per_page": per, "par": par}


def timed_rows(results: Path, model: str, label: str, reg_file: dict, doc: dict) -> tuple:
    """(rows, why): the timed page's NATIVE cells as v5price.e1cal_rows builds them (A2's selections, the ALL view,
    tcommon.row_drop against v4's price)."""
    k5, _ = chain()
    d5 = k5.d5
    TC, RG = d5.TC, d5.RG
    prm = d5.params_for(d5.SEL, model)
    with d5.A2.scoring(d5.RULES):
        rep, why = TC.timed_page(Path(results), model, label, int(reg_file["group_m"]), TC.C3.TimedView("ALL"), reg_file)
        if rep is None:
            return [], why
        if rep.get("lock_mhz") is None or int(rep["lock_mhz"]) != LOCK:
            return [], f"NOT SCORED: the timed page records lock_mhz {rep.get('lock_mhz')!r}, not {LOCK}"
        tt = {(int(r["tiles"]), str(r["histogram"]), str(r["arm"])): r for r in rep["treads_table"]}
        rows = []
        for c in doc["cells"]:
            if "native" not in c["arms"]:
                continue
            k = (int(c["n"]), c["label"], "native")
            tr = tt.get(k)
            base = RG.price_cell(model, "native", c, doc, prm, key=d5.SEL["key"])["ms"]
            if tr is None:
                rows.append({"model": model, "page": doc["page"], "cls": "128g1", "n": k[0], "label": k[1], "meas": None,
                             "drop": "no row on the page", "c": c, "doc": doc, "v4": base})
                continue
            rows.append({"model": model, "page": doc["page"], "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])),
                         "n": k[0], "label": k[1], "meas": float(tr["ms_p50"]), "drop": TC.row_drop(tr, base),
                         "c": c, "doc": doc, "v4": base})
    return rows, ""


def predict(rows: list, sag128: tuple) -> list:
    """v5'' ms of each row at p_128 5.75 and F1 (A, tau) = sag128 (k5.cell_ms, the null form)."""
    k5, _ = chain()
    return [k5.cell_ms(r, None, P128, tuple(sag128), None) for r in rows]


def held_out(rows: list, sag128: tuple, p128: float = P128) -> dict:
    """X: k5.summ over the kept rows' e = pred / meas - 1."""
    k5, _ = chain()
    keep = [r for r in rows if not r["drop"] and r["meas"]]
    res = [(r, k5.cell_ms(r, None, p128, tuple(sag128), None) / r["meas"] - 1) for r in keep]
    out = {"cells": [{"n": r["n"], "label": r["label"], "meas_ms": round(r["meas"], 6), "e_pct": round(100 * x, 3)} for r, x in res],
           "dropped": [{"n": r["n"], "label": r["label"], "why": r["drop"]} for r in rows if r["drop"] or not r["meas"]]}
    if res:
        s = k5.summ(res)
        m = rows[0]["model"]
        out["X_pct"] = s["model_class_means"].get(f"{m} 128g1")
        out["summ"] = {k: s[k] for k in ("n", "rms_pct", "worst_pct", "mean_pct")}
    else:
        out["X_pct"] = None
    return out


def x_of_rows(rows: list) -> dict:
    """C2-S1: x = bytes-only / floor-only per GEMM (v5'' terms, F1 off: knee-v5 x22.py's xs) over the rows."""
    k5, _ = chain()
    d5 = k5.d5
    xs = {"w1": [], "w2": []}
    for r in rows:
        prm = d5.params_for(d5.SEL, r["model"])
        with d5.A2.scoring(d5.RULES):
            p = d5.v5_parts(r["model"], r["c"], r["doc"], prm, F3=k5.F3_M4, P={128: P128, 64: P64}, sag="none")
        for g in xs:
            xs[g].append(p[g]["bytes_only"] / p[g]["floor_only"])
    return {g: {"min": round(min(v), 3), "median": round(st.median(v), 3), "max": round(max(v), 3)} for g, v in xs.items() if v}


@contextlib.contextmanager
def pages_as(root: Path, model: str, unit: str):
    """kneefit.PAGES set to one (root, model, unit) page for k5.capture_rows, restored on exit."""
    k5, _ = chain()
    keep = k5.KF.PAGES
    k5.KF.PAGES = [(Path(root), model, unit, LOCK, "c2")]
    try:
        yield
    finally:
        k5.KF.PAGES = keep


def counter_reading(root: Path, model: str, unit: str, p128: float = P128) -> dict:
    """The cycles-domain error of the v5'' knee (null form, carried bw, p) per GEMM on one byte page's NATIVE
    128x128 cells (k5.capture_rows, eval_rows), with the counted reads beside M4's v3 reads."""
    k5, _ = chain()
    d5 = k5.d5
    with pages_as(root, model, unit):
        rows = k5.capture_rows(("c2",))
    if not rows:
        return {"rows": 0}
    e = k5.eval_rows(rows, k5.Form("null", k5.BW_CARRIED), p128)
    out = {"rows": len(rows)}
    for g in ("w1", "w2"):
        eg = [x for r, x in zip(rows, e) if r["gemm"] == g]
        out[g] = {**cell_summary(eg), "x_median": round(st.median(r["x"] for r in rows if r["gemm"] == g), 3),
                  "K": next(r["K"] for r in rows if r["gemm"] == g)}
    # bytes: M4 (v3 key) reads against the counted reads, per NATIVE cell (the predicted reads cell_ms prices)
    cells = [c for c in k5.KF.cells_of(Path(root), model, unit, LOCK) if c["arm"] == "native" and c["doc"]["block_m"] == 128]
    prm = d5.params_for(d5.SEL, model)
    eb = {"w1": [], "w2": []}
    with d5.A2.scoring(d5.RULES):
        for c in cells:
            with d5.V.swapped(P128, d5.V.RHO_X[128]):
                reads = d5.V.f3_reads(d5.RG.pricer(), model, int(c["c"]["n"]), c["c"]["counts"], d5.RG.seed_of(c["doc"], c["c"]),
                                      d5.RG.PAGE_TILE[128], int(c["doc"]["group_m"]), prm, k5.F3_M4[128], "v3")[0]
            if reads is None:
                continue
            for g in eb:
                eb[g].append(float(reads[g]) / c["reads"][g] - 1)
    out["bytes_m4_over_counted"] = {g: cell_summary(v) for g, v in eb.items()}
    return out


def census(results: Path) -> dict:
    """C2-S5: 8x22B's compiled CTAs per SM at 128x128 (the tc2 tile census) beside the occupancy v5'' prices it at
    (A2's selections, finalise.params: the census of every model, 2 per SM)."""
    k5, _ = chain()
    d5 = k5.d5
    prm = d5.params_for(d5.SEL, MODEL)
    t = d5.RG.PAGE_TILE[128]
    out = {"priced": {g: int(prm.occ(t, g)) for g in ("w1", "w2")}}
    f = sorted(Path(results).glob("*-tile-census-*-tc2/tile-census.json"))
    if len(f) != 1:
        return {**out, "why": f"{len(f)} tc2 census files"}
    p = json.loads(f[0].read_text())
    got = {e["gemm"]: int(e["occupancy"]["ctas_per_sm"]) for e in p.get("entries") or []
           if e.get("model") == MODEL and e.get("tile") == "128x128x64w8s3"}
    out.update(compiled=got, verdict=p.get("verdict"), same=got == out["priced"])
    return out


def sag_domain(root: Path, model: str, unit: str, est: tuple) -> dict:
    """C2-S3: estimator minus measured off (us) on one counter page's GEMMs (sag.pts_of at 1410)."""
    k5, _ = chain()
    p = k5.S.pts_of(page_files(root, model, unit), float(LOCK))
    if not p:
        return {"n": 0}
    A, tau = est
    e = [A * (1 - math.exp(-t / tau)) - off for t, off in p]
    own = k5.S.fit(p)
    return {"n": len(e), "mean_err_us": round(st.fmean(e), 2), "rms_us": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "own_fit_A_us": own["A_us"], "own_fit_tau_us": own["tau_us"]}


# ================================================================== the whole reading

def run_commit(session: Path) -> tuple:
    """(sha, why): the commit the rental ran, from the driver's ledger line `driver start: commit <sha>` (status,
    else gh200-driver.out), checked against session/commit.txt when that file exists."""
    S = Path(session)
    shas = set()
    for f in (S / "gh200-driver" / "status", S / "gh200-driver.out"):
        if f.is_file():
            shas |= set(re.findall(r"driver start: commit ([0-9a-f]{40})\b", f.read_text(errors="replace")))
    if len(shas) != 1:
        return None, f"{len(shas)} commits on the ledger's driver start lines"
    sha = next(iter(shas))
    ct = S / "commit.txt"
    if ct.is_file() and ct.read_text().strip() != sha:
        return None, f"commit.txt reads {ct.read_text().strip()!r}, the ledger {sha}"
    return sha, ""


def check_commit(repo: Path, sha: str, paths) -> list:
    """[(path, why)] for every path whose working copy in `repo` is not `git show <sha>:<path>` byte for byte."""
    bad = []
    for p in paths:
        got = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{p}"], capture_output=True)
        if got.returncode != 0:
            bad.append((p, f"absent at {sha[:12]}"))
        elif not (Path(repo) / p).is_file():
            bad.append((p, "no working copy"))
        elif hashlib.sha256(got.stdout).hexdigest() != sha256(Path(repo) / p):
            bad.append((p, f"differs from {sha[:12]}"))
    return bad


def commit_paths(R: dict) -> list:
    """The files the freeze ties to the rental's commit: the registration, the page, the plan, every code_pins path
    (the reader, the driver and the chain's repo modules among them)."""
    return sorted(set(COMMIT_FILES) | set(R["code_pins"]))


def check_pins(R: dict, repo: Path) -> list:
    """[(path, why)] for every code_pins path whose copy under `repo` is not the registered sha256."""
    return [(p, f"sha256 under {repo} is not the registered one") for p, h in sorted(R["code_pins"].items())
            if not (Path(repo) / p).is_file() or sha256(Path(repo) / p) != h]


def chain_repo() -> Path:
    k5, _ = chain()
    return Path(k5.d5.V.REPO)


def check_chain_repo(R: dict) -> list:
    """[(path, why)]: every module the chain has loaded from its repo is a registered pin with its registered bytes,
    and every CHAIN_REPO_MODULES path there has its registered sha256."""
    root = chain_repo().resolve()
    bad = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or "/.venv/" in f:
            continue
        f = Path(f).resolve()
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            continue
        if rel not in R["code_pins"]:
            bad.append((rel, "loaded by the chain from its repo but not pinned"))
    bad += [(p, f"under the chain's repo {root}: not the registered sha256") for p in CHAIN_REPO_MODULES
            if p not in R["code_pins"] or not (root / p).is_file() or sha256(root / p) != R["code_pins"][p]]
    return sorted(set(bad))


def triad(results: Path) -> dict:
    """C2-S5: calibrate's triad ruler on C2's board beside A2's 3726.2 GB/s (15 rentals 3725.6 to 3727.6)."""
    f = sorted(Path(results).glob("calibration/*ceilingtriad*/measured_nvidia_gh200_480gb.yaml"))
    out = {"A2_gbps": TRIAD_A2_GBPS, "range_15_rentals_gbps": list(TRIAD_RANGE_GBPS)}
    if len(f) != 1:
        return {**out, "why": f"{len(f)} ceilingtriad pages"}
    for line in f[0].read_text().splitlines():
        if line.strip().startswith("achieved_bandwidth_gbps:"):
            v = float(line.split(":")[1])
            return {**out, "C2_gbps": v, "inside_range": TRIAD_RANGE_GBPS[0] <= v <= TRIAD_RANGE_GBPS[1]}
    return {**out, "why": "no achieved_bandwidth_gbps"}


def split(timed: list, sag128: tuple, root: Path, model: str, unit: str, p128: float) -> dict:
    """C2-S6's GEMM against per-call split on one board, per NATIVE cell present on both pages: time error (ms) of
    v5'' on the timed page at (p128, sag128), the GEMMs' error (ms) on the byte page's cycles at p128 (k5.capture_rows,
    the null form, counted bytes, no sag on either side), and their difference, the per-call part (T0, other, the
    sag); with M4's bytes against the counted reads beside it (counter_reading)."""
    k5, _ = chain()
    keep = [r for r in timed if not r["drop"] and r["meas"]]
    t_err = {(r["n"], r["label"]): (k5.cell_ms(r, None, p128, tuple(sag128), None) - r["meas"], r["meas"]) for r in keep}
    with pages_as(root, model, unit):
        rows = k5.capture_rows(("c2",))
    if not rows:
        return {"why": f"no {model} {unit} counter rows"}
    e = k5.eval_rows(rows, k5.Form("null", k5.BW_CARRIED), p128)
    g_err = {}
    for r, x in zip(rows, e):
        g_err.setdefault((r["n"], r["label"]), {})[r["gemm"]] = x * r["meas"]
    cells = []
    for k in sorted(set(t_err) & {k for k, v in g_err.items() if set(v) == {"w1", "w2"}}):
        te, meas = t_err[k]
        ge = g_err[k]["w1"] + g_err[k]["w2"]
        cells.append({"n": k[0], "label": k[1], "meas_ms": round(meas, 6), "time_err_ms": round(te, 6),
                      "gemm_err_ms": round(ge, 6), "per_call_err_ms": round(te - ge, 6),
                      "time_pct": round(100 * te / meas, 3), "gemm_pct_of_time": round(100 * ge / meas, 3),
                      "per_call_pct_of_time": round(100 * (te - ge) / meas, 3)})
    out = {"cells": cells}
    if cells:
        for k in ("time_pct", "gemm_pct_of_time", "per_call_pct_of_time"):
            out[f"mean_{k}"] = round(st.fmean(c[k] for c in cells), 2)
    return out


def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def read(c2: Path, *, registration: dict | None = None, s1_results: Path | None = None,
         _skip_commit_check_for_tests: bool = False) -> dict:
    """The C2 reading from a published C2 session directory (results/ and session/ under it). Refuses (SystemExit)
    unless: the 22 work files are the registered bytes; the rental's commit (the ledger's driver start line) holds the
    working copies of the registration, the page, the plan and every code_pins path byte for byte; every code_pins
    path has its registered sha256 here; every repo module the chain loaded is pinned and has its registered bytes in
    the chain's repo; and v5'' at A2's par reproduces the registered fingerprint."""
    R = registration or reg()
    check_frozen(R)
    c2 = Path(c2)
    results, session = c2 / "results", c2 / "session"
    out = {"registration": REG_NAME, "c2": str(c2), "status": "CALIBRATION: certifies nothing; every C2 page is SEEN once read"}
    if _skip_commit_check_for_tests:
        out["commit"] = {"sha": None, "checked": False}
    else:
        sha, why = run_commit(session)
        if sha is None:
            raise SystemExit(f"REFUSED: the rental's commit is not on its ledger: {why}")
        bad = check_commit(REPO, sha, commit_paths(R))
        if bad:
            raise SystemExit(f"REFUSED: the working tree is not the rental's commit {sha}: {bad[:5]}")
        out["commit"] = {"sha": sha, "checked": True, "files": len(commit_paths(R))}
    bad = check_pins(R, REPO)
    if bad:
        raise SystemExit(f"REFUSED: a pinned file is not its registered bytes: {bad[:5]}")
    out["board"] = brd = board(session)
    k5, _ = chain()
    # the fingerprint: v5'' at A2's par on 8x22B's cells equals the registered predictions
    doc = json.loads((REPO / HIST / "mixtral-8x22b-128g1.json").read_text())
    nat = [{"model": MODEL, "page": doc["page"], "n": c["n"], "label": c["label"], "doc": doc, "c": c}
           for c in doc["cells"] if "native" in c["arms"]]
    fp = [round(x, 6) for x in predict(nat, A2_PAR_128)]
    out["fingerprint"] = {"v5_ms_at_A2_par": fp, "equals_registered": fp == R["frozen"]["fingerprint_ms_at_A2_par"]}
    if not out["fingerprint"]["equals_registered"]:
        raise SystemExit("REFUSED: v5'' at A2's par does not reproduce the registered fingerprint: the chain moved")
    try:
        sf = sag_fits(results)
    except LookupError as e:
        sf = None
        out["F1"] = {"why": str(e)}
    reg_file = R["pages"]["files"]["mixtral-8x22b-128g1.json"]
    why, rows = "", []
    if sf is not None:
        out["F1"] = {"per_page": {m: list(v) for m, v in sf["per_page"].items()}, "par": [round(sf["par"][0], 4), sf["par"][1]],
                     "A2 per_page": {m: list(v) for m, v in A2_PAGE_FITS.items()}, "A2 par": list(A2_PAR_128)}
        rows, why = timed_rows(results, MODEL, T_LABEL, reg_file, doc)
        r1 = held_out(rows, sf["par"]) if rows else {"X_pct": None, "cells": [], "dropped": []}
        if rows:
            out["C2-S1 x"] = x_of_rows([r for r in rows if not r["drop"]])
    else:
        r1 = {"X_pct": None, "cells": [], "dropped": []}
    X = r1.get("X_pct")
    n = len(r1["cells"])
    valid = brd["valid"] and sf is not None and not why
    out["C2-R1"] = {**r1, "page_why": why}
    if rows:   # DESCRIPTIVE beside X: over A2's four labels; at A2's board sag (the board's sag against the model's error)
        out["C2-R1"]["X4_pct (A2's labels PT, PW, uniform, balanced)"] = held_out([r for r in rows if r["label"] in A2_LABELS], sf["par"])["X_pct"]
        out["C2-R1"]["X_at_A2_par_pct"] = held_out(rows, A2_PAR_128)["X_pct"]
    out["table"] = heldout_table(X if valid else None)
    out["outcome"] = outcome(X, n, valid, why or brd.get("why", ""))
    out["pass_estimates"] = pass_estimates(X) if out["outcome"]["band"] != "NOT READ" else None
    # secondary readings
    f, w = check_counter_page(results, MODEL, B_LABEL, G=1, reg_file=reg_file)
    out["C2-S2 8x22B b128g1"] = counter_reading(results, MODEL, B_LABEL) if f else {"why": w}
    out["C2-S3 8x22B sag"] = sag_domain(results, MODEL, B_LABEL, sf["par"]) if (f and sf) else {"why": w or "no F1"}
    mf = R["pages"]["files"]["mixtral-8x7b-128g1.json"]
    mf_ok, mw = check_counter_page(results, "mixtral-8x7b", M_LABEL, G=1, reg_file=mf)
    s4 = {"C2 board": counter_reading(results, "mixtral-8x7b", M_LABEL) if mf_ok else {"why": mw}}
    s1 = Path(s1_results) if s1_results else k5.KF.S1T
    s4["S1 board (9da3c0)"] = counter_reading(s1, "mixtral-8x7b", "b128g1") if page_files(s1, "mixtral-8x7b", "b128g1") else {"why": "no S1 page"}
    s4["A2 timed, same cells, held out (time domain)"] = CAL_HELD_OUT["mixtral-8x7b"]
    s4["reading"] = s4_reading((s4["C2 board"].get("w1") or {}).get("mean_pct"), (s4["S1 board (9da3c0)"].get("w1") or {}).get("mean_pct"))
    out["C2-S4 Mixtral-8x7B b128g1"] = s4
    # C2-S6: Mixtral-8x7B timed on C2's board, in A2's +3.13 fold (p 5.25, F1 from C2's OLMoE page alone)
    s6 = {"A2 held out": CAL_HELD_OUT["mixtral-8x7b"], "p_128": MIX_FOLD_P}
    if sf is not None:
        mdoc = json.loads((REPO / MIX_HIST).read_text())
        olm = sf["per_page"]["olmoe-1b-7b"]
        mrows, mwhy = timed_rows(results, "mixtral-8x7b", MT_LABEL, mf, mdoc)
        s6["F1 (C2 OLMoE f128 alone)"] = list(olm)
        if mrows:
            h = held_out(mrows, olm, p128=MIX_FOLD_P)
            s6.update(held_out=h, M_pct=h["X_pct"])
            s6["split (one board: C2's timed and byte pages)"] = split(mrows, olm, results, "mixtral-8x7b", M_LABEL, MIX_FOLD_P) if mf_ok else {"why": mw}
        else:
            s6.update(M_pct=None, why=mwhy)
    else:
        s6.update(M_pct=None, why="no F1")
    s6["reading"] = s6_reading(s6.get("M_pct"))
    out["C2-S6 Mixtral-8x7B t128g1"] = s6
    out["C2-S5 census"] = census(results)
    out["C2-S5 triad"] = triad(results)
    bad = check_chain_repo(R)
    if bad:
        raise SystemExit(f"REFUSED: the chain ran a repo module that is not its registered bytes: {bad[:5]}")
    out["chain_repo"] = {"path": str(chain_repo()), "modules_checked": len(CHAIN_REPO_MODULES)}
    return out


def txt(d: dict) -> str:
    L = [f"C2 READING ({d['registration']}): {d['status']}", f"  pages: {d['c2']}", f"  commit: {json.dumps(d['commit'], sort_keys=True)}",
         f"  board: {json.dumps(d['board'], sort_keys=True)}",
         f"  fingerprint (v5'' at A2's par equals the registration): {d['fingerprint']['equals_registered']}",
         f"  F1 (C2 board): {json.dumps(d.get('F1'), sort_keys=True)}", "",
         "FOUR-MODEL HELD-OUT TABLE (128 G 1 means, %)"]
    for m, v in d["table"].items():
        L.append(f"  {m}: {json.dumps(v, sort_keys=True)}")
    r = d["C2-R1"]
    L += ["", f"C2-R1: X = {r.get('X_pct')} over {len(r['cells'])} cells; dropped {r['dropped']}; {r.get('page_why', '')}"]
    for k in ("X4_pct (A2's labels PT, PW, uniform, balanced)", "X_at_A2_par_pct"):
        if k in r:
            L.append(f"  {k} (DESCRIPTIVE): {r[k]}")
    for c in r["cells"]:
        L.append(f"  n {c['n']} {c['label']}: meas {c['meas_ms']} ms, e {c['e_pct']:+.3f}%")
    L += [f"OUTCOME: {d['outcome']['band']}: {d['outcome']['b3']}", f"PASS ESTIMATES: {d['pass_estimates']}", ""]
    for k in ("C2-S1 x", "C2-S2 8x22B b128g1", "C2-S3 8x22B sag", "C2-S4 Mixtral-8x7B b128g1", "C2-S6 Mixtral-8x7B t128g1",
              "C2-S5 census", "C2-S5 triad", "chain_repo"):
        if k in d:
            L.append(f"{k}: {json.dumps(d[k], sort_keys=True)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c2", type=Path, required=True, help="the published C2 session directory (results/, session/)")
    ap.add_argument("--s1-results", type=Path, default=None, help="S1's results directory (default: where kneefit reads it)")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    d = read(a.c2, s1_results=a.s1_results)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "cal2.read.json").write_text(dumps(d))
    (a.out_dir / "cal2.read.txt").write_text(txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
