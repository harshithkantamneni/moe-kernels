#!/usr/bin/env python
"""Writes (--write) or checks (--check) model v7's gate registration, docs/registered/2026-11-17-v7-gate-gh200.{json,txt}.

    python scripts/scoring/v7/register_v7.py --write | --check | --trace

The registration states the old bars per regime unchanged, each regime's GO independent, the disclosures and the
freeze (the sha256 of every code, data and registration file the gate reads, and of v7's constants). The code list is
not written by hand: `--trace` (run in a fresh interpreter by freeze()) makes the gate's load without pricing a cell
(gate_v7 imported; r3_pricer; round5_chain.chain, b2_rows, held_out_sets; decode_modules and the gate's decode
imports) and lists every repository file in sys.modules, then closes that set over every import statement in those
files, function-level imports included, resolved on the interpreter's final sys.path (a lazy import that may run on a
price path is pinned whether or not the load ran it). Building it prices nothing: no gate statistic is computed
here, and gate_v7.py runs only after this file is public."""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate_v7 as GV  # noqa: E402
import r3_timing_model_v7 as V7  # noqa: E402
import round5_chain as R5  # noqa: E402

NAME = GV.NAME
REG = REPO / "docs" / "registered"
DATE = "2026-11-17"

#: the gate's decode imports (gate_v7.compute_decode), loaded by the trace
GATE_DECODE_IMPORTS = ("d4common", "dcommon", "finalise_decode", "gate_d6", "score_blind_d3", "score_cal_d3")
#: not on the gate's path: this writer itself
TRACE_EXCLUDE = ("scripts/scoring/v7/register_v7.py",)
#: the readings and committed numbers the branches price with or the gate reads
FREEZE_DATA = (
    "scripts/scoring/pc/pc.read.json", "scripts/scoring/pc2/pc2.read.json",
    "scripts/scoring/decode/d6_build/pc_spans.json", "scripts/scoring/decode/ds.readings.json",
    "scripts/scoring/decode/d6_build/v6d_constants.json", "scripts/scoring/decode/gate_d6.out.txt",
    "scripts/scoring/tileB3/b3.predictions.json", "scripts/scoring/tileB3/tileB3.score.json",
)
#: the registrations the clauses quote and the reductions are held to
FREEZE_REG = (
    "docs/registered/2026-10-09-rental6-skew-gh200.json", "docs/registered/2026-10-13-rental7-skew-gh200.json",
    "docs/registered/2026-10-27-cal4-gh200.json", "docs/registered/2026-10-27-cal4-gh200.txt",
    "docs/registered/2026-11-04-tileB3-gh200.json", "docs/registered/2026-11-04-tileB3-gh200.txt",
    "docs/registered/2026-11-10-decode-addendum6-gh200.json", "docs/registered/2026-11-10-decode-addendum6-gh200.txt",
)
FREEZE_VENDORED = tuple(f"scripts/round5_frozen/{f}" for f in R5.FROZEN_FILES)


def _repo_rel(f) -> str | None:
    """f's path relative to the repository (symlinks resolved), or None outside it."""
    if not f or not os.path.isabs(f):
        return None
    r = os.path.realpath(f)
    root = os.path.realpath(REPO)
    return os.path.relpath(r, root) if r.startswith(root + os.sep) and os.path.isfile(r) else None


def _resolve(name: str, dirs: list[str]) -> list[str]:
    """The files `import name` binds, searched on `dirs` in order (first hit wins, as the import system's path
    finder): the module and its packages' __init__.py; [] when not found as a source file."""
    parts = name.split(".")
    for d in dirs:
        base, files = Path(d), []
        for i, part in enumerate(parts):
            if (base / part / "__init__.py").is_file():
                base = base / part
                files.append(str(base / "__init__.py"))
            elif i == len(parts) - 1 and (base / f"{part}.py").is_file():
                files.append(str(base / f"{part}.py"))
            elif (base / part).is_dir() and i < len(parts) - 1:
                base = base / part                                    # a namespace package
            else:
                files = None
                break
        if files:
            return files
    return []


def _imports(path: str, dirs: list[str]) -> set[str]:
    """Every file an import statement in `path` may bind (module level and inside functions alike)."""
    out: set[str] = set()
    tree = ast.parse(Path(path).read_text(), path)
    here = os.path.dirname(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.update(_resolve(a.name, dirs))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here
                for _ in range(node.level - 1):
                    base = os.path.dirname(base)
                look = [base]
                mod = node.module or ""
            else:
                look, mod = dirs, node.module or ""
            if mod:
                out.update(_resolve(mod, look))
            for a in node.names:
                if a.name != "*":
                    out.update(_resolve(f"{mod}.{a.name}" if mod else a.name, look))
    return out


def trace_job() -> dict:
    """In a fresh interpreter (`--trace`): the gate's load and the static import closure of what it loaded."""
    sys.dont_write_bytecode = True
    V7.r3_pricer()
    R5.chain()
    R5.b2_rows()
    R5.held_out_sets()
    V7.decode_modules()
    for m in GATE_DECODE_IMPORTS:
        importlib.import_module(m)
    loaded = {os.path.realpath(m.__file__) for m in list(sys.modules.values())
              if _repo_rel(getattr(m, "__file__", None))}
    dirs = [d for d in sys.path if d and os.path.isabs(d) and os.path.isdir(d)]
    seen, todo = set(), list(loaded)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        if f.endswith(".py"):
            todo.extend(x for x in (os.path.realpath(y) for y in _imports(f, dirs)) if _repo_rel(x) and x not in seen)
    rel = lambda fs: sorted({_repo_rel(f) for f in fs} - set(TRACE_EXCLUDE))     # noqa: E731
    return {"loaded": rel(loaded), "closure": rel(seen)}


@functools.lru_cache(maxsize=1)
def trace() -> dict:
    """trace_job's result from a fresh interpreter (`register_v7.py --trace`; -I isolated from the caller's paths and
    environment, -B writing no bytecode), so the closure does not depend on what the caller has imported."""
    r = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--trace"], capture_output=True,
                       text=True, cwd=str(REPO), timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"the import trace failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def freeze_code() -> tuple[str, ...]:
    """Every repository source file on the gate's import closure (trace), less the vendored round 5 (pinned at B3's
    hashes in FREEZE_VENDORED)."""
    return tuple(p for p in trace()["closure"] if p not in FREEZE_VENDORED)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze() -> dict:
    code = freeze_code()
    paths = code + FREEZE_VENDORED + FREEZE_DATA + FREEZE_REG
    assert len(set(paths)) == len(paths)
    tr = trace()
    return {"statement": ("the gate (gate_v7.verify_freeze, reading this registration at its committed path only) "
                          "refuses every regime unless each file below has its sha256 and "
                          "r3_timing_model_v7.constants_sha256() is constants_sha256; the code list is computed, not "
                          f"written by hand (register_v7.trace): the {len(code)} repository source files on the gate's "
                          f"import closure, {len([p for p in tr['loaded'] if p not in FREEZE_VENDORED])} of them "
                          "loaded by the gate's load in a fresh interpreter (sys.modules) and the rest reached from "
                          "those files' import statements, function-level imports included; the vendored files are "
                          "round 5's 32 work files at B3's registered hashes"),
            "constants_sha256": V7.constants_sha256(), "constants": V7.constants(),
            "files": {p: sha256(REPO / p) for p in paths}}


GATE_TEXT = {
    "GO rule": ("one GO / NO-GO per regime (R3, PREFILL, DECODE), each regime's GO independent of the others; a regime "
                "may go to ONE blind only on its own GO, the blind (models, cells, bars) registered separately before "
                "any blind page; no bar may be changed after this file is public; the gate (gate_v7.py) runs only "
                "after this file is public, on v7 with every switch on (RR, D6FIX), and its first reading on the "
                "published pages is the reading"),
    "bars": "the old bars, unchanged: R3's reduction (v4's --check, 2026-10-14), C4's next-round gate (2026-10-27), "
            "addendum 6's two clauses (2026-11-10)",
    "R3": {
        "G-R3": ("v7 at R3's tile (32x64x64 w8 s4) re-prices the 300 registered v3 predictions (rentals 6 and 7: "
                 "docs/registered/2026-10-09-rental6-skew-gh200.json, 2026-10-13-rental7-skew-gh200.json) bit for bit: "
                 "every one equal to the registered number at 6 decimals and to rental 6's pricer exactly "
                 "(r3_timing_model_v7.check_r3)"),
    },
    "PREFILL": {
        "G1": ("every held-out 128 G 1 model mean within 1.0% (|mean| <= 1.0%, the mean as r4.summ prints it, 2 "
               "decimals): OLMoE-1B-7B, Qwen2-57B-A14B and Mixtral-8x7B on A2's cells, Mixtral-8x22B and Mixtral-8x7B "
               "on C2's board, and Mixtral-8x7B-tp2 on C4's (C4's restatement: G1 covers the five CAL models and "
               "Mixtral-8x7B on C2's board); held out leave-one-model-out exactly as round 5 computed it (the fold's "
               "LOMO K1 constants, SEEN for tp2; the fold's F1 par on the timed page's board, C4's two pages for tp2); "
               "RR's C and sigma are carried unchanged into every fold (not refitted per fold; fitted on RR's "
               "synthetic pages and chosen with this table in view, see disclosures)"),
        "G2": ("estimated P(A) >= 0.5 by the registered method: r7.pass_estimates5 (read_c2.power at B2's bar 2% / "
               "5% / 1%, R 200,000, seed 20261022) on the five models' held-out means (OLMoE, Qwen2-57B, Mixtral-8x7B "
               "on A2, Mixtral-8x22B, tp2) over B2's power cells that v7 prices, its 'A (5 models)'"),
        "refusal (TREES, as decode's)": ("before any clause, v7 with every switch off must reproduce round 5's "
                                         "held-out table (tileB3.score.json preflight.round5_table) and B3's 84 "
                                         "committed predictions (b3.predictions.json, ms to 6 decimals, w1 and w2 "
                                         "reads to the byte) on the pages the clauses read (gate_v7.prefill_trees); "
                                         "any difference and the regime reads NO-GO (REFUSED), nothing judged"),
        "printed, never a clause": ("B3's 84 NATIVE cells re-priced at B3's board pars (b3.predictions.json par, "
                                    "par64) against B3's measured times: pooled and per model mean / rms / worst. "
                                    "SEEN: B3's FAIL stands and this is not a pass"),
    },
    "DECODE": {
        "CAL (addendum 6's clause 1)": ("CAL2 under v7 reads GO by the addendum-3 stop rule at 3% rms, 8% worst cell, "
                                        "1.5% |mean| (v6d's three CAL constants refitted on v7's "
                                        "body on CAL2's kept cells, and, for the v4d'-branch cells only, D6FIX's "
                                        "three constants from a second CAL2 fit with v4d' on every naive cell; "
                                        "score_cal_d3's rules)"),
        "RE-SCORE (addendum 6's clause 2)": ("the published decode blind (five SEEN models) re-scored under v7 at the "
                                             "CAL2 refit reads a pooled rms under 4.0% over the committed 220 cells; a "
                                             "WEAK SANITY FLOOR on SEEN pages, never a test and never a certification"),
        "refusals (as gate_d6)": ("READINGS: the committed readings give v6d's registered law and terms; TREES: the "
                                  "CAL2 and decode blind trees reproduce their committed scores (d4common.reproduce)"),
        "D6FIX's constants": ("the CAL2 fit with v4d' on every naive cell, its rho form by the registered selection "
                              "rule, read by the gate on CAL2's tree"),
    },
    "ablation (DESCRIPTIVE, never a verdict)": ("each regime's clauses printed beside at every switch off (prefill: "
                                                "round 5; decode: v6d); R3 has no v7 term"),
}


def build() -> dict:
    D = "moe-kernels-work 2026-11-17 design-v7 DESIGN.md"
    return {
        "registered": DATE,
        "name": NAME,
        "status": ("REGISTERED BEFORE gate_v7.py RAN, NOT BEFORE ITS VALUES WERE KNOWN: during implementation the "
                   "build computed every clause value except G2's P(A) (R3 GO by construction; PREFILL NO-GO on G1, "
                   "four of six held-out means outside 1.0% whatever G2 reads; DECODE GO, CAL2 1.60% rms and the "
                   "re-score 3.19%), so this gate registers known results and decides nothing new; the blind, "
                   "registered separately, is the test. Model v7 (one price() over a declared regime switch) and "
                   "its gate, the old bars per regime unchanged and each regime's GO independent. A DRAFT in the "
                   "build clone until the owner pushes it; gate_v7.py runs only after this file is public. Every "
                   "earlier registered verdict stands (B3 FAIL, round 5's gate FAIL, the decode blind FAIL of "
                   "2026-11-04, addendum 6's NO-GO)"),
        "design": (f"{D} (offline, 2026-11-17) and the owner's decisions on it (2026-11-17): the structure (one module, "
                   "one price(), the regime switch: R3's tile v3 exactly, the prefill tiles round 5 plus the RR w1 "
                   "re-read term at G 1, the decode tiles v6d's law); the RR rule F2 at 47.5 MiB between mid-lives "
                   "with life spread sigma 0.08 (128x128) and 0.06 (64x128); the D6-24 fix adopted behind its own "
                   "switch; every new term behind its own switch; the owner's piece miss_fraction implemented in the "
                   "repository at the owner's request; the old bars per regime, unchanged"),
        "tool": ("scripts/r3_timing_model_v7.py (the model; --check-r3), scripts/round5_chain.py and "
                 "scripts/round5_frozen/ (round 5, vendored), scripts/scoring/v7/gate_v7.py [--out dir] (the gate), "
                 "scripts/scoring/v7/register_v7.py --write / --check / --trace (this file; --trace, the freeze's "
                 "import closure)"),
        "model v7": {
            "module": "scripts/r3_timing_model_v7.py",
            "structure": ("a declared REGIME SWITCH, not one physical law: one price(cell) dispatching on the tile key "
                          "(r3: 32x64x64w8s4; prefill: 128x128x64w8s3, 64x128x64w8s3; decode: 16x64x128w4s4, "
                          "32x64x128w4s3, 32x128x64w4s3, 64x128x64w4s3; any other key Refused) and, for the RR term, on "
                          "GROUP_SIZE_M (1 only), each a domain of measured support"),
            "branches": {
                "r3": "v3 = v4 at R3's tile (c scale exactly 1.0, F-step, the v3 byte key); no v7 term is defined there",
                "prefill": ("round 5 as B3 froze it (STAG-CUT w1 and W2 w2, the K1 wall knee, the clock form of F1, "
                            "v5'' at 64x128), its 32 work files vendored byte for byte (scripts/round5_frozen, the "
                            "sha256 of B3's frozen.files); switch RR replaces the w1 DRAM reads at G 1 only"),
                "decode": ("v6d (addendum 6) at its registered law and terms; switch D6FIX re-prices only the naive "
                           "cells v6d prices by v4d''s naive GEMMs (a non-fetching follower, D6-24)"),
            },
            "switches (registered: both on)": {
                "RR": ("prefill tiles, G 1: Bytes_w1 = C_u + mhat R; C_u = (experts with a row) x W_e + (unique tokens) "
                       "x H x 2; R = sum_e (m_e - 1) W_e, m_e = ceil(c_e / BLOCK_M); mhat = the share of re-read "
                       "pairs (r, r - P) whose F2 distance d = ((s_r + l_r / 2) - (s_{r-P} + l_{r-P} / 2)) x W_c x "
                       "slab exceeds 47.5 MiB (strict), with starts s and lives l from L-SCHED (pid order onto W_c "
                       "slots, 264 at 128x128 and 396 at 64x128; life max(0.05, 1 + sigma eps), sigma 0.08 at "
                       "128x128 and 0.06 at 64x128; numpy default_rng seed 7, 6 replicas); slab 128 x K x 2"),
                "D6FIX": ("a decode cell v6d prices by v4d''s naive GEMMs is priced at the CAL2 fit v4d' selects with "
                          "v4d' on every naive cell (its constants and rho form; on the committed CAL2 tree scaled, "
                          "T0 -1.87, s_naive +9.55, s_block -0.72 us, d6_build/post_gate.out); every other cell at "
                          "v6d's CAL2 fit; the CAL fit itself (bodies with no constants) is v6d's"),
            },
            "constants": {"RR": V7.rr_constants()},
            "not built": ("RR at decode: the design counts 0 aligned w1 re-reads on CAL2's 91 kept cells and on the 220 "
                          "re-score cells (design-v7 section 0 item 3), so the term would be inert on every gate page; "
                          "the decode branch is v6d's law. A deviation from design-v7's table 1.1, made because the "
                          "owner's decision named v6d's law for the decode tiles and an untested branch with no effect "
                          "on any page cannot be checked"),
            "miss_fraction": ("the step rule on measured stamps (the owner's piece: each CTA pulls its slab evenly "
                              "over its life; F1 starts, F2 mid-lives; d > C strict): over sorted start and end events "
                              "with running byte sums, on times centered on the earliest stamp before any float "
                              "conversion (integer arithmetic for integer stamps; a non-positive life refused); equals "
                              "read_rr's m_hat on all 810 (launch, step form) readings of RR's published stamps; it is "
                              "the measurement side of the term, the launch layer's byte clock (dt x W_c x slab) the "
                              "pricing side, the two equal while every slot is busy"),
            "reductions (COMPUTED by the build, tests/test_timing_model_v7*.py)": {
                "r3": "300 of 300 registered v3 predictions equal to 6 decimals and to rental 6's pricer bit for bit, "
                      "with RR and D6FIX on and every v7 term patched to raise",
                "prefill, RR off": "B3's 84 committed predictions (ms to 6 decimals, reads to the byte) and round 5's "
                                   "held-out table (tileB3.score.json preflight.round5_table) exactly",
                "decode, D6FIX off": "gate_d6's CAL2 rows, fit and stop verdict and its 220 re-score rows exactly",
                "switch isolation": "RR moves only B3's 48 G 1 cells and only their w1; D6FIX moves no prefill cell and "
                                    "exactly the 17 v4d'-branch decode page cells; RR moves no decode cell",
            },
            "numerical hygiene": ("v7 fits nothing on stamps and differences no absolute stamp (miss_fraction and "
                                  "bytes_streamed center on the earliest stamp before any float conversion, in integer "
                                  "arithmetic when the stamps are integers such as int64 globaltimer ns, and refuse a "
                                  "non-positive life; the launch layer's times start at 0). Constants v7 carries from "
                                  "PC: c0 at G 128 (PC-R2, an OLS of rung lives on slab bytes, no stamp in the fit), "
                                  "the L table and X_sat (PC-R7) and kappa (d6_build/pc_spans.py) were read by PC's "
                                  "pinned reader, which casts absolute globaltimer stamps to float64 (a 256 ns step; "
                                  "the defect of record of 2026-11-12, median lives moved up to 1.6%, median 0.07%); "
                                  "on the 116 spans kappa is read from one step is at most 1.7% of a span, median "
                                  "0.19%. PC2's constants were read after EXACT TIME's rebase; DS's W1RATE ratio "
                                  "carries the environment dependence of record (1.4e-7 relative). Not refitted: a "
                                  "reading would change v6d's registered constants"),
        },
        "gate": GATE_TEXT,
        "disclosures": {
            "every v7 constant and rule is SEEN": [
                ("RR's form F2 and C 47.5 MiB: RR's DESCRIPTIVE refit on its own fresh pages (rr.read.txt), never "
                 "registered there; F1 and F2 were found POST-HOC on four stamp families with B3, PC and PC2 in view "
                 "(design-P1 2.3)"),
                ("RR's sigma 0.08 / 0.06: fitted on RR's 57 fresh cells at C fixed, chosen AFTER pricing launch-layer "
                 "variants on SEEN pages: six scored on the 125 seen counted G 1 w1 cells and four priced in time on "
                 "B3's and round 5's pages (design-v7 sections 3 and 5.2); the forks {F1, F2} x {C from P1, RR, free} "
                 "x {one sigma, per tile} are 12 paths"),
                ("D6FIX: the D6-24 defect was found on SEEN pages after addendum 6's gate (addendum 6 'after the "
                 "gate'); the fix is chosen knowing it moves the 13 seen v4d'-branch cells from 13.2% to 1.36% rms "
                 "(post_gate.out's counterfactual). Addendum 6 itself says of this defect: 'Any fix would be chosen "
                 "on the seen re-score, so none is made and the NO-GO stands.' v7 makes that fix, and clause 2 is "
                 "read on the same 220 seen cells, so under D6FIX clause 2 is not evidence: the fix was chosen with "
                 "its effect on 13 of these 220 cells known, and a DECODE GO is the owner's 2026-11-17 decision to "
                 "reopen applied to a SEEN floor, not a test the model passed"),
                ("round 5's terms were built over five declared rounds, each with the earlier rounds' held-out numbers "
                 "in view (B3's SEEN block); its gate FAILED and B3 FAILED; v6d's choices were made on PC, PC2, DS and "
                 "CAL2 pages after the decode blind was seen"),
            ],
            "the gate is a registered statement of a known estimate, not a blind test": (
                f"{D} section 5 already priced these pages (POST-HOC, SEEN). Prefill, round 5's held-out 128 G 1 "
                "table under the RR term: means OLMoE +0.37, Qwen2-57B -1.50, Mixtral-8x7B +1.42 (C2's board +1.66), "
                "Mixtral-8x22B -0.01, tp2 -1.99 (section 5.2): four of six outside 1.0%, so G1 reads NO-GO as "
                "estimated (G2 not estimated there). Decode: CAL2 1.81% GO under v6d (section 5.3); the re-score 4.52% "
                "as frozen and about 3.19% with D6FIX (207 cells at 3.27% and 13 at 1.36%): GO only with the D6-24 "
                "fix, as estimated. R3: 300 of 300 by construction"),
            "computed by the build before this registration (POST-HOC, SEEN; disclosed, not a gate reading)": [
                ("while checking the RR implementation against the design's prototype: the six held-out means under "
                 "RR (equal to the design's to 1e-9: +0.37, -1.50, +1.42, -0.01, -1.99, C2 +1.66) and B3's 84 cells "
                 "(equal to the design's per cell)"),
                ("while checking D6FIX: CAL2 under v7 reads GO (rms 1.60%, worst -5.40%, mean -0.17%; the 4 OLMoE "
                 "v4d'-branch cells move) and the re-score pools 3.19% over 220 cells (Granite-3B 5.85% rms "
                 "unchanged). All 13 seen-blind v4d'-branch cells take a new price at the CAL2 refit (17 page cells "
                 "in all with CAL2's 4, as at v6d's committed constants); 11 of the 13 change their scored error and "
                 "2 do not, Qwen1.5 L-T3-A4 and deepseek-v2-lite L-T2-A6, because E1-DEC counts an absolute error "
                 "at or under 1 us as 0 (dcommon.FLOOR_US) and both are inside that floor before and after the fix "
                 "(0.94 us and 1.00 us under v6d, 0.17 us and 0.12 us under v7); P(A) and the ablation print were "
                 "not computed"),
            ],
            "the owner's reopening (2026-11-17)": ("decode closed after v6d under the owner's stopping rule (2026-11-10) "
                                                   "and prefill after RR (2026-11-11); the owner reopened both on "
                                                   "2026-11-17 for one unified model (docs/FINDINGS.md: Decode closes "
                                                   "after v6d, Rental RR; README): a forking path the paper must "
                                                   "disclose whatever the gate reads"),
            "no verdict moves": ("this registration changes no earlier verdict, bar, label or file; every registration "
                                 "it quotes is in the freeze at its bytes"),
        },
        "freeze": freeze(),
    }


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and depth < 3:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {x if isinstance(x, str) else json.dumps(x, default=str)}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {x if isinstance(x, str) else json.dumps(x, default=str)}")
    else:
        out.append(f"{prefix}{v}")


def text(d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}, before gate_v7.py ran but NOT before its values were known (every clause "
         f"value except G2's P(A) computed during implementation; see status): model v7 and its gate, {d['name']}",
         f"status: {d['status']}", f"design: {d['design']}", f"tool: {d['tool']}", ""]
    for k, v in d.items():
        if k in ("registered", "name", "status", "design", "tool", "freeze"):
            continue
        _flat("", {k: v}, L)
    fz = d["freeze"]
    L += ["", f"FREEZE: {fz['statement']}", f"  constants_sha256: {fz['constants_sha256']}",
          f"  constants: {json.dumps(fz['constants'], sort_keys=True)}", f"  files ({len(fz['files'])}):"]
    L += [f"    {h}  {p}" for p, h in fz["files"].items()]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--trace", action="store_true", help="print the gate's import closure (JSON); freeze() runs it")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    if a.trace:
        print(json.dumps(trace_job()))
        return 0
    doc = build()
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    jpath, tpath = a.out_dir / f"{NAME}.json", a.out_dir / f"{NAME}.txt"
    if a.write:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        jpath.write_text(jtext)
        tpath.write_text(ttext)
        print(f"wrote {jpath.name}, {tpath.name}")
        return 0
    ok = True
    for p, t in ((jpath, jtext), (tpath, ttext)):
        same = p.exists() and p.read_text() == t
        ok &= same
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
