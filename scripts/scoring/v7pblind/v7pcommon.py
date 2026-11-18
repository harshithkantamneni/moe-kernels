#!/usr/bin/env python
"""Model v7's PREFILL blind (registration docs/registered/2026-11-18-v7-prefill-blind-gh200): the census, the models,
the pages and cells, the frozen predictions, the fix the prefill path needs, the bars and rules, the census gate, the
finalisation helpers, the freeze, the plan and the rental, in one module for the registration builder
(register_v7p_blind.py), the VM gate (scripts/v7p_gate.py) and the scorer (score_blind_v7p.py). It imports v7
(scripts/r3_timing_model_v7.py), round 5's loader (scripts/round5_chain.py), the v7 gate (scripts/scoring/v7/gate_v7.py)
and B3's pure rules (scripts/scoring/tileB3/b3common.py) and changes none of them.

THE RULES THIS FILE OWNS:
  census     every model of moe/spec.py (toy aside) against every registered page list and every JSON record of every
             published session: a page at one of v7's prefill tiles (128x128x64 w8 s3, 64x128x64 w8 s3) naming the model
             or a synthetic shape with its (E, H, I, k) makes HAS A PREFILL-TILE PAGE; every other timing of the model
             (other tiles, decode pages, other cards) is listed as a disclosure
  models     Granite-3.0-1B-A400M, Granite-3.0-3B-A800M and Qwen2-57B-A14B-tp8 (moe/spec.py, unchanged), every one with
             no prefill-tile page (census); MODEL_CHOICE says why these three
  cells      B3's recipe (scripts/skew_synth_tile.py's rule, its pinned generators skew_synth._draw and shape_param6
             called unchanged, a new seed base): per model one 128x128x64 w8 s3 G 1 page at the two smallest treads vLLM
             0.27.1 runs there at G 1 (n < 6, M4's Hopper G-1 rule) and one G 16 page at the smallest G-16 tread (n <= 9,
             the declared copies); per tread PT, PW, PT2, PW2 (OLMoE's fitted shapes, top_k 8), uniform and balanced;
             arms NATIVE and SHARED, 9 declared copies, B3's instrument
  predict    v7 with every switch on (RR at G 1), the prefill branch exactly (V7.price_prefill), at the K1 knee (round 4's
             SEEN constants, knee4.json) and the board's 128x128 sag par; frozen before any page at four reference pars (A2, C2, C4,
             B3's board); the rental's own par finalises mechanically (finalise_pars) before any blind page is read
  fix        round 5's r7._rc lazily re-imports refit4, which round5_chain.chain() has popped (the v7 gate's PREFILL
             REFUSED); apply_fix makes r7._rc return the refit4 module the chain imported, in memory, outside every
             frozen file, as scripts/scoring/v7/prefill_rerun_v7.py does; FIX_SOURCE is registered verbatim
  score      the working-model bar (rms 3%, worst 8%, |mean| 1.5%), per model and pooled, each read in ALL and CLEAN and
             combined by rental 2's rule 3; a model CERTIFIED when its own bar and the pooled bar hold and at least
             MIN_CELLS of its cells are scored in CLEAN (CERT_RULE, MIN_RULE); B3's E1-TILE conjunction printed beside
  draft      the registration is a DRAFT until the owner's go is recorded by register_v7p_blind.py --write --final; the
             plan carries a DRAFT line until then, and scripts/vm_run_v7p.sh refuses it
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "v7"),
           str(REPO / "scripts" / "scoring" / "tileB3"), str(REPO / "scripts" / "scoring" / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b3common as B  # noqa: E402
import gate_v7 as GV  # noqa: E402
import r3_timing_model_v7 as V7  # noqa: E402
import round5_chain as R5  # noqa: E402

DATE = "2026-11-18"
NAME = f"{DATE}-v7-prefill-blind-gh200"
REG_JSON = f"docs/registered/{NAME}.json"
REG_TXT = f"docs/registered/{NAME}.txt"
#: a -hist directory, as the tile stage's prefill pages (2026-10-14-tileB-hist): the decode censuses read *pages
PAGES = f"docs/registered/{DATE}-v7-prefill-blind-hist"
PLAN = "scripts/plans/rentalV7P-prefill-blind-2026-11.plan"
DRIVER = "gh200_v7p_session.sh"
PARENT_DRIVER = "gh200_tileB3_session.sh"
LAUNCHER = "scripts/vm_run_v7p.sh"
PARENT_LAUNCHER = "scripts/vm_run_v7b.sh"
GATE_SCRIPT = "scripts/v7p_gate.py"
GATE_NAME = GV.NAME
B3_NAME = B.NAME
RERUN = "scripts/scoring/v7/prefill_rerun_v7"
DECODE_BLIND_NAME = "2026-11-17-v7-decode-blind-gh200"
DOMAIN_BLIND_NAME = "2026-11-18-v7-domain-blind-gh200"

MODELS = ("granite-3.0-1b-a400m", "granite-3.0-3b-a800m", "qwen2-57b-a14b-tp8")
#: the fitted shapes every model draws: OLMoE's (synth_fit.json fits Mixtral, OLMoE and Qwen1.5 only); the same top_k 8,
#: as B2 drew Qwen2-57B-A14B ("the same E 64 and top_k 8") and deepseek-v2-lite (E 64) from OLMoE's
SHAPE_SOURCE = "olmoe-1b-7b"
SHAPE_WHY = "OLMoE's fitted shapes (the same top_k 8, as B2 drew Qwen2-57B-A14B and deepseek-v2-lite; SEEN-INFORMED)"
BLOCK_M = 128
TILE = "128x128x64w8s3"
TILE_LIST = [128, 128, 64, 8, 3]
GPU_NAME = "NVIDIA GH200 480GB"
DTYPE = "bf16"
#: the seed base (skew_synth_tile.py's rule with a new base: rental A 20261410, B 20261420; no seed repeats theirs)
BASE = 20281118
TILE_INDEX = 1
LABELS = ("PT", "PW", "PT2", "PW2")
LABEL_INDEX = {"PT": 0, "PW": 1, "PT2": 2, "PW2": 3}
ARMS = ("native", "shared")
COPIES = 9
G1_TREADS = 2
HOPPER_G1_N = 6
LOCK = 1410
POWER_W = 700
LIST_RATE = 2.59
#: the K1 knee at round 4's SEEN constants (knee4.json K1 SEEN, B3's), the 64x128 par the price call is handed (no 64x128 cell here)
KNEE = R5.K1_SEEN
A2_PAR_64 = (30.365, 300.0)
#: the four boards' 128x128 sag pars the predictions are frozen at before any page (A2's, C2's, C4's: B3's reference
#: pars; B3's own board: b3.predictions.json par, the par the gate's B3 information was read at)
REF_PARS = {"A2": (42.02, 425.0), "C2": (41.59, 412.5), "C4": (54.25, 587.5), "B3": (59.815, 612.5)}

BARS = dict(B.BAR)
BARS_SOURCE = (
    "the working-model bar, verbatim: docs/registered/2026-11-04-tileB3-gh200.txt lines 7 to 9 'values: "
    "{\"rms_max\": 0.03, \"max_abs\": 0.08, \"mean_max\": 0.015}', 'applies to: the pooled NATIVE cells AND each blind "
    "model', 'decided: owner, 2026-11-04: for every NEW blind test; registered here before any B3 page' (b3common.BAR); "
    "the same numbers the decode blinds carry (owner, D4-1: docs/registered/2026-11-04-decode-addendum5-gh200.txt line "
    "9, docs/registered/2026-11-10-decode-addendum6-gh200.txt line 159, the v7 decode blind 2026-11-17 'bars'); "
    "unchanged, not tightened, not loosened")
CERT_RULE = (
    "the verdict is the working-model bar read per model, as the v7 decode blind (docs/registered/"
    "2026-11-17-v7-decode-blind-gh200.txt 'certification') and the decode domain blind read it: "
    "docs/registered/2026-10-14-decodeB-blind-gh200.txt's rule line 'a model is CERTIFIED when its own bar and the pooled "
    "bar hold', carried by addenda 5 and 6 and by both v7 decode blinds. Here each bar (pooled and per model: rms <= 3%, "
    "every cell within 8%, |mean| <= 1.5%) is read in B3's two views (ALL and CLEAN, tcommon.timed_page) and combined "
    "by rental 2's rule 3 (agreeing views give the verdict; a disagreement reads INCONCLUSIVE, which certifies nothing). "
    "A model is CERTIFIED when its combined bar HOLDS, the combined pooled bar HOLDS and the minimum is met")
MIN_CELLS = 10
MIN_RULE = (
    "MINIMUM (the decode domain blind's rule, docs/registered/2026-11-18-v7-domain-blind-gh200 'minimum', copied): a "
    "model is certifiable only when at least 10 of its cells are scored in CLEAN (valid NATIVE cells after the power "
    "check, the gates, the page checks (G5 provenance, the r7-median-run3 rebuild, the registered tile), row_drop and "
    "G3's voids). A model with fewer is reported with its statistics, labelled BELOW THE MINIMUM and NOT CERTIFIED "
    "whatever they read; its scored cells still count in the pooled bar")
VOID_RULES = (
    "the power check first (the decode blinds' order; B3's board validity): the board must read c2-power.env "
    "C2_POWER_W=700, locks.env LOCK_PROBE_1410=held and G-BAND PASS, and the census gate (v7pcensus) and the sag gate "
    "(b3sag) records PASS, else the test is NOT SCORED (exit 1) and nothing is certified. Then B3's page and cell rules, "
    "unchanged (docs/registered/2026-11-04-tileB3-gh200.txt TESTS): a page whose provenance (G5: histogram sha256, "
    "shuffle seed, counts sha256), rebuild or recorded tile fails is NOT SCORED (its cells are missing); a page whose "
    "rebuilt G1 fails is counted in ALL only (voided from CLEAN); a cell failing row_drop (6 of 9 clean, the host rule "
    "on the finalised prediction and the timer flag, G4 >= 1405 MHz) is dropped; G3 (rental 7's TOST on uniform minus "
    "balanced, r5common.g3_rule): homogeneous and failing, every histogram cell is voided and the test reads NOT SCORED; "
    "heterogeneous, the failing strata's non-balanced cells are voided and the verdict is read with those voids (the "
    "bar without them printed beside). The decode blinds' GD2 and G3-D voids are decode instruments with no prefill page "
    "here; B3's G3 is their prefill counterpart")
B3_E1_TILE = (
    "B3's E1-TILE, DESCRIPTIVE here (docs/registered/2026-11-04-tileB3-gh200.txt TESTS): HOLDS iff the POOLED NATIVE "
    "cells AND each blind model meet rms <= 3%, every cell within 8%, |mean| <= 1.5% in ALL and in CLEAN (rule 3); a "
    "model CERTIFIED only when E1-TILE HOLDS, its bar holds in CLEAN and at least 2 of its pages are in CLEAN. It is "
    "printed beside, never the verdict: it is B3's conjunction over its own four models; the verdict here is the per-"
    "model rule above, the rule v7's other blind was registered under (CERT_RULE)")
DRAFT_LINE = ("# DRAFT: NOT LAUNCHABLE. This plan stays DRAFT until the owner's go is recorded by "
              "register_v7p_blind.py --write --final; scripts/vm_run_v7p.sh refuses it.")

MODEL_CHOICE = (
    "the census's models with no prefill-tile page that B3's instrument can run (NATIVE and SHARED at 9 declared "
    "copies: R3 refuses a SHARED declaration of E x 9 >= 1024 experts, so DeepSeek-V3 and its tp4 and tp8 shards (E 256) "
    "cannot run it; DeepSeek-V3 also needs 9 x 22.5 GB of weights on a 96 GB card and has vLLM fused_experts timings at "
    "prefill sizes on H200 and A100, 2026-08-22 to 2026-09-01) are Granite-3.0-1B-A400M, Granite-3.0-3B-A800M, "
    "Qwen2-57B-A14B-tp8, Mixtral-8x7B-tp4 and Mixtral-8x7B-tp8. Three fit the 2 h budget with B3's 18 cells a model. "
    "Chosen: the two Granite models and Qwen2-57B-A14B-tp8 (E 32, 40 and 64, H 1024, 1536 and 3584). Left unspent: "
    "Mixtral-8x7B-tp4 and -tp8, TP shards of Mixtral-8x7B, a CAL model of round 5 whose tp2 shard is a CAL model too "
    "(rental C4): they share its E 8, k 2, H 4096 and router, so they are the nearest to SEEN. Qwen2-57B-A14B-tp8 is "
    "a TP shard of Qwen2-57B-A14B, a CAL model too (held-out mean -1.50% under v7): disclosed, kept as the only "
    "eligible E 64 model. A judgment, made after the gate and the census were read, before any page")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


# ================================================================== the fix (registered verbatim)

def apply_fix() -> None:
    """r7._rc returns the refit4 module round5_chain.chain() already imported (prefill_rerun_v7.apply_fix's fix)."""
    C = R5.chain()
    RF = C["RF"]
    assert RF.__name__ == "refit4", RF
    C["r7"]._rc = lambda: RF


FIX_SOURCE = inspect.getsource(apply_fix)
FIX_STATEMENT = (
    "the v7 gate read PREFILL: NO-GO: REFUSED: ModuleNotFoundError: No module named 'refit4' "
    "(scripts/scoring/v7/gate_v7.out.txt): round 5's r7._rc (scripts/round5_frozen/2026-10-28/round5/work/r7.py) "
    "imports refit4 lazily, and round5_chain.chain() pops refit4 from sys.modules after its import. The fix, applied in "
    "memory and outside every frozen file exactly as scripts/scoring/v7/prefill_rerun_v7.py applies it: r7._rc returns "
    "round5_chain.chain()['RF'], the refit4 module the chain imported. It changes no price (V7.price_prefill never calls "
    "r7._rc); it lets round 5's pass estimate (r7.pass_estimates5) run. It is part of the frozen model registered here")


# ================================================================== the treads and the pages (B3's recipe)

def resolve(model: str, M: int):
    """vLLM 0.27.1's tile for M tokens of the model in bf16 on a GH200, offline (moe.bench.tile_resolve over the
    vendored v0.27.1 config listing): (tile key, G, provenance, the tuned file looked up)."""
    from moe.bench import tile_resolve as TR
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    d = TR.resolve_tile(cfg.num_experts, cfg.intermediate_size, DTYPE, GPU_NAME, int(M))
    key = f"{d.block_m_derived}x{d.block_n_derived}x{d.block_k_derived}w{d.num_warps_derived}s{d.num_stages_derived}"
    return key, d.group_m_derived, d.provenance, d.config_file


def tokens_at(model: str, n: int, bm: int = BLOCK_M) -> int:
    """n x BLOCK_M x E / k (skew_synth_tile.tokens_at's rule), refused when not whole."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    rows = n * bm * cfg.num_experts
    if rows % cfg.top_k:
        raise ValueError(f"{model} n={n} at BLOCK_M {bm}: {rows} / {cfg.top_k} is not a whole token count")
    return rows // cfg.top_k


def treads(model: str) -> dict:
    """{1: the G-1 treads, 16: the G-16 tread}: the G1_TREADS smallest n < HOPPER_G1_N at which vLLM runs TILE at G 1,
    and the smallest n <= COPIES at which it runs TILE at G 16 (B3's rule: the smallest vLLM-consistent treads)."""
    g1, g16 = [], []
    for n in range(1, COPIES + 1):
        try:
            M = tokens_at(model, n)
        except ValueError:
            continue
        key, G, _p, _f = resolve(model, M)
        if key != TILE:
            continue
        if G == 1 and n < HOPPER_G1_N and len(g1) < G1_TREADS:
            g1.append(n)
        elif G == 16 and not g16:
            g16.append(n)
    return {1: tuple(g1), 16: tuple(g16)}


def page_specs() -> list[tuple]:
    """[(model, G, treads, page name)]: each model's G-1 page, then its G-16 page."""
    out = []
    for m in MODELS:
        t = treads(m)
        out += [(m, 1, t[1], f"{m}-128g1"), (m, 16, t[16], f"{m}-128g16")]
    return out


def seed_of(model: str, n: int, label: str) -> int:
    return BASE + 10000 * TILE_INDEX + 1000 * MODELS.index(model) + 10 * n + LABEL_INDEX[label]


def page_doc(model: str, G: int, ns, name: str) -> dict:
    """skew_synth_tile.page's rule (its pinned generators, unchanged) for one page of this blind."""
    import skew_synth as SS

    from moe.spec import MODEL_CONFIGS
    fitd = json.loads(SS.FIT.read_text())
    E = MODEL_CONFIGS[model].num_experts
    cells = []
    for n in ns:
        Bt = tokens_at(model, n)
        for lab in LABELS:
            shape = lab.rstrip("2")
            kind, prm, rule = SS.shape_param6(fitd, SHAPE_SOURCE, shape, Bt)
            seed = seed_of(model, n, lab)
            c = [int(v) for v in SS._draw(kind, prm, model, Bt, seed)]
            st = SS.hist_stats(c)
            cells.append(SS._cell(lab, n, c, ARMS, kind=kind, param=prm, seed=seed, B=Bt, source_model=SHAPE_SOURCE,
                                  source_shape=shape, rule=rule, stats={s: round(v, 6) for s, v in st.items()}))
        cells += [SS._cell("uniform", n, [n * BLOCK_M] * E, ARMS), SS._cell("balanced", n, None, ARMS)]
    return {"schema": SS.SCHEMA, "model": model, "page": name, "shuffle_seed": BASE, "block_m": BLOCK_M,
            "group_m": G, "cells": cells,
            "synthetic": (f"fitted Zipf/Dirichlet shapes of {SHAPE_SOURCE} ({SHAPE_WHY}); scripts/skew_synth_tile.py's "
                          f"rule (seed base {BASE}), model v7's prefill blind ({NAME}); no trace count")}


def page_path(name: str) -> str:
    return f"{PAGES}/{name}.json"


def page_texts() -> dict[str, str]:
    import skew_synth as SS
    return {page_path(name): SS.dump(page_doc(m, G, ns, name)) for m, G, ns, name in page_specs()}


def write_pages(root: Path = REPO) -> None:
    for p, t in page_texts().items():
        (Path(root) / p).parent.mkdir(parents=True, exist_ok=True)
        (Path(root) / p).write_text(t)


def load_page(name: str, root: Path = REPO) -> dict:
    return json.loads((Path(root) / page_path(name)).read_text())


def sha_counts(counts) -> str:
    """R3's counts_sha256 (private_weight_reference.counts_sha256): sha256 of the compact JSON, 'null' for balanced."""
    return hashlib.sha256(json.dumps(None if counts is None else [int(v) for v in counts],
                                     separators=(",", ":")).encode()).hexdigest()


def reg_file(name: str, root: Path = REPO) -> dict:
    """B2's page record (regtile.files_block's form, which tcommon.timed_page reads)."""
    p = Path(root) / page_path(name)
    doc = json.loads(p.read_text())
    return {"path": page_path(name), "sha256": sha256(p), "shuffle_seed": doc["shuffle_seed"], "tile": list(TILE_LIST),
            "group_m": doc["group_m"],
            "cells": [{"label": c["label"], "n": c["n"], "arms": c["arms"], "counts_sha256": sha_counts(c["counts"]),
                       **({"seed": c["provenance"]["seed"], "kind": c["provenance"]["kind"],
                           "param": c["provenance"]["param"], "source_model": c["provenance"]["source_model"]}
                          if c.get("provenance") else {})} for c in doc["cells"]]}


def files_block(root: Path = REPO) -> dict:
    return {f"{name}.json": reg_file(name, root) for _m, _G, _ns, name in page_specs()}


def gd5() -> dict:
    """GD5 offline: every cell's token count through vLLM 0.27.1's lookup (moe.bench.tile_resolve, the vendored
    v0.27.1 listing): the tuned file name it looks for, whether v0.27.1 ships it, the tile and G it runs."""
    from moe.bench import tile_resolve as TR
    out, bad = {}, []
    for m, G, ns, name in page_specs():
        for n in ns:
            M = tokens_at(m, n)
            key, g, prov, f = resolve(m, M)
            out[f"{name} n {n}"] = {"M": M, "tile": key, "G": g, "provenance": prov, "file looked up": f,
                                    "shipped in v0.27.1": TR.ships(f)}
            if (key, g) != (TILE, G) or TR.ships(f):
                bad.append(f"{name} n {n}")
    return {"label": ("COMPUTED offline by moe.bench.tile_resolve (vLLM v0.27.1's get_moe_configs, nearest-key lookup "
                      "and get_default_config, over the vendored v0.27.1 config listing "
                      "moe/bench/hardware/vllm_configs/SHIPPED_FILE_NAMES.txt, which names no NVIDIA_GH200_480GB file); "
                      "the VM's runtime readback (each page's recorded tile against the registered one) binds"),
            "cells": out, "verdict": "PASS" if not bad else "FAIL: " + ", ".join(bad)}


# ================================================================== the rows and the predictions

def rows(root: Path = REPO) -> list[dict]:
    """Every NATIVE cell of the pages as a round-5 row (b2risk.b2_cells' form), in page order."""
    out = []
    for m, G, _ns, name in page_specs():
        doc = load_page(name, root)
        for c in doc["cells"]:
            if "native" in c["arms"]:
                out.append({"unit": f"t128g{G}", "model": m, "page": name, "doc": doc, "c": c, "cls": f"128g{G}",
                            "n": int(c["n"]), "label": c["label"]})
    return out


def price(r: dict, sag128, switches=None) -> dict:
    """v7's price of one cell (the registered pricer): V7.price_prefill at the board's 128x128 par, the K1 knee at round
    4's SEEN constants (knee4.json), with every switch on unless told otherwise."""
    return V7.price_prefill(r, sag128=tuple(sag128), sag64=A2_PAR_64, knee=None,
                            switches=V7.V7 if switches is None else switches)


def predictions(sag128, rs: list[dict] | None = None) -> list[dict]:
    """Every cell priced at a par: v7 (every switch on) and, DESCRIPTIVE beside, round 5 (every switch off); ms to 6
    decimals, reads to the byte, RR's parts at G 1."""
    out = []
    for i, r in enumerate(rows() if rs is None else rs):
        a = price(r, sag128)
        b = price(r, sag128, V7.OFF)
        rr = (a.get("terms") or {}).get("RR")
        out.append({"index": i, "model": r["model"], "page": r["page"], "cls": r["cls"], "G": int(r["doc"]["group_m"]),
                    "n": r["n"], "label": r["label"], "arm": "native",
                    "ms": None if a["ms"] is None else round(float(a["ms"]), 6),
                    "reads": None if a["ms"] is None else {g: int(round(v)) for g, v in a["reads"].items()},
                    "w1_round5": None if a["ms"] is None else int(round(a["w1_frozen"])),
                    "RR": None if rr is None else {"m_hat": r6(rr["m_hat"]), "R": int(rr["R"]), "C_u": int(rr["C_u"]),
                                                   "pairs": int(rr["pairs"])},
                    "round5_ms (DESCRIPTIVE, every switch off)": None if b["ms"] is None else round(float(b["ms"]), 6)})
    return out


def frozen_predictions() -> dict:
    """The predictions at the four reference pars (frozen before any page) and the per-model stakes of RR."""
    rs = rows()
    by = {}
    for board, par in REF_PARS.items():
        p = predictions(par, rs)
        by[board] = {"par": list(par), "cells": p}
    a2 = by["A2"]["cells"]
    stakes = {}
    for m in MODELS:
        for cls in ("128g1", "128g16"):
            d = [100 * (c["ms"] / c["round5_ms (DESCRIPTIVE, every switch off)"] - 1) for c in a2
                 if c["model"] == m and c["cls"] == cls and c["ms"] is not None]
            stakes[f"{m} {cls}"] = {"n": len(d), "mean": round(sum(d) / len(d), 3) if d else None,
                                    "min": round(min(d), 3) if d else None, "max": round(max(d), 3) if d else None}
    return {"boards": by, "unpriced": [c["index"] for c in a2 if c["ms"] is None], "n_cells": len(a2),
            "RR stakes (v7 against round 5, %, at A2's par)": stakes}


def compact(fp: dict) -> dict:
    """{board: {par, ms list}}: the fingerprint the scorer recomputes before it reads anything."""
    return {b: {"par": v["par"], "ms": [c["ms"] for c in v["cells"]]} for b, v in fp["boards"].items()}


# ================================================================== the gate's prefill numbers, reproduced

def gate_reproduction(estimate: bool = True) -> dict:
    """prefill_rerun_v7's every-switch-on numbers recomputed with this module's pricer (price, with the fix): round 5's
    six held-out 128 G 1 means (round5_chain.held_out_sets, each fold's knee and sag, r4.summ's 2-decimal means), the
    estimate (round5_chain.pass_estimate at B2's bar on B2's 84 power cells v7 prices) and B3's 84 cells at B3's board
    pars (SEEN information)."""
    apply_fix()
    C = R5.chain()
    means = {}
    for k, (cells, hm) in R5.held_out_sets().items():
        knee = R5.held_out_knee(hm)
        e = [V7.price_prefill(r, sag128=R5.held_out_sag(r, hm), knee=knee, switches=V7.V7)["ms"] / r["meas"] - 1
             for r in cells]
        means[k] = C["r4"].summ(e)["mean"]
    out = {"means_pct": means}
    rws = R5.b2_rows()
    if estimate:
        cells84 = C["RF"].RC.b2_cells()
        priced = [c for r, c in zip(rws, cells84, strict=True)
                  if int(r["doc"]["block_m"]) != 128 or R5.reads(r, R5.K1_SEEN[0]) is not None]
        out["estimate"] = R5.pass_estimate({k: means[k] for k in GV.FIVE}, priced)
        out["estimate_cells"] = len(priced)
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    score = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())
    meas = {}
    for c in score["views"]["cells"]:
        m, page, n, lab, _arm = c["cell"].split("/")
        meas[(m, page, int(n), lab)] = c["meas"]
    by: dict = {}
    for r in rws:
        k = (r["model"], r["page"], int(r["n"]), r["label"])
        if k not in meas:
            continue
        p = V7.price_prefill(r, sag128=tuple(fin["par"]), sag64=tuple(fin["par64"]), switches=V7.V7)
        e = p["ms"] / meas[k] - 1
        by.setdefault(r["model"], []).append(e)
        by.setdefault("pooled", []).append(e)
    out["b3_information"] = {m: GV.summ(v) for m, v in by.items()}
    return out


def committed_rerun() -> dict:
    """prefill_rerun_v7.json's every-switch-on block, as committed."""
    d = json.loads((REPO / f"{RERUN}.json").read_text())["prefill"]
    inf = d["information"]
    return {"means_pct": d["G1"]["means_pct"], "estimate": inf["estimate"], "estimate_cells": inf["estimate_cells"],
            "b3_information": inf["B3 84 cells (SEEN, never a clause)"]}


def working_bar_estimate(means: dict, R: int = 200_000) -> dict:
    """ESTIMATE, DESCRIPTIVE: B3's method (b3common.pass_estimates, read_c2.power with the bar an argument) at the
    working-model bar on B2's 84 power cells with v7's five held-out means; B3 printed the same with round 5's."""
    return B.pass_estimates({k: means[k] for k in GV.FIVE}, rule=BARS, R=R)


# ================================================================== the census: who has a prefill-tile page

PREFILL_TILES = set(V7.PREFILL_TILES)
SHAPE_FILES = ("docs/registered/2026-11-06-pc-shapes.json", "docs/registered/2026-11-10-pc2-shapes.json",
               "docs/registered/2026-11-11-rr-shapes.json")
#: registered page lists of the tile stage and its CAL rentals: every page at BLOCK_M 64 or 128 is a prefill-tile page
#: (their own registrations name the tiles); decode-page lists name decode tiles only
TEXT_SUFFIXES = (".json", ".txt", ".log", ".env", ".plan", ".yaml", ".csv", ".md")


def spec_models() -> list[str]:
    from moe.spec import MODEL_CONFIGS
    return [m for m in MODEL_CONFIGS if m != "toy"]


def shape_of_cfg(cfg) -> tuple:
    return (cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k)


def synthetic_shapes() -> dict:
    """{name: (E, H, I, k)} of every registered synthetic shape (PC, PC2, RR, the decode map)."""
    out = {}
    for f in SHAPE_FILES:
        for s in json.loads((REPO / f).read_text())["shapes"]:
            out[s["name"]] = (s["E"], s["H"], s["F"], s["k"])
    try:
        sys.path.insert(0, str(REPO / "scripts" / "scoring" / "v7domain"))
        import map_shapes as MS  # noqa: E402
        for n, _f, E, k, H, I in MS.SHAPES:  # noqa: E741
            out[n] = (E, H, I, k)
    except ImportError:
        pass
    return out


def names_model(text: str, model: str) -> bool:
    """The text names the model exactly ('deepseek-v3' not as the prefix of 'deepseek-v3-tp4'; - or _)."""
    t = text.lower().replace("_", "-")
    tail = "" if "-tp" in model else r"(?!-tp\d)"
    return re.search(r"(?<![0-9a-z])" + re.escape(model) + tail + r"(?![0-9a-z])", t) is not None


def _tk(m, n, k, w, s) -> str:
    return f"{m}x{n}x{k}w{w}s{s}"


def json_tiles(o, model, bm, out: dict) -> None:
    """(model or shape, tile) pairs in a JSON document: a dict naming 'model' or 'shape' sets the context for what it
    holds; a config dict (BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages; BLOCK_SIZE_M or the context's block_m)
    or a design dict (block_m, block_n, block_k, num_warps, num_stages) names a tile."""
    if isinstance(o, dict):
        for key in ("model", "shape"):
            if isinstance(o.get(key), str):
                model = o[key]
                break
        if isinstance(o.get("block_m"), int):
            bm = o["block_m"]
        if all(k in o for k in ("BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")):
            m = o.get("BLOCK_SIZE_M", bm)
            if m is not None:
                out.setdefault(model, set()).add(_tk(m, o["BLOCK_SIZE_N"], o["BLOCK_SIZE_K"], o["num_warps"],
                                                     o["num_stages"]))
        if all(k in o for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages")):
            out.setdefault(model, set()).add(_tk(o["block_m"], o["block_n"], o["block_k"], o["num_warps"],
                                                 o["num_stages"]))
        for v in o.values():
            json_tiles(v, model, bm, out)
    elif isinstance(o, list):
        for v in o:
            json_tiles(v, model, bm, out)


def path_model(rel: str, names: list[str]) -> str | None:
    for m in names:
        if names_model(rel, m):
            return m
    return None


def published_sessions() -> list[str]:
    pub = REPO / "results" / "published"
    return sorted(p.name for p in pub.iterdir() if p.is_dir())


def merged_rows(sessions: list[str], models) -> dict:
    """Per model, every published merged.csv row naming it: {session: {gpu, impls, num_tokens range, rows}}."""
    import csv
    out: dict = {m: {} for m in models}
    pub = REPO / "results" / "published"
    for s in sessions:
        for f in sorted((pub / s).rglob("merged.csv")):
            with f.open() as fh:
                for row in csv.DictReader(fh):
                    m = row.get("model")
                    if m in out:
                        x = out[m].setdefault(s, {"gpu": set(), "impls": set(), "T": set(), "rows": 0})
                        x["gpu"].add(row.get("gpu_name"))
                        x["impls"].add(row.get("impl"))
                        x["T"].add(int(float(row.get("num_tokens") or 0)))
                        x["rows"] += 1
    return {m: {s: {"gpu": sorted(x["gpu"]), "impls": sorted(x["impls"]),
                    "num_tokens": [min(x["T"]), max(x["T"])], "rows": x["rows"]} for s, x in d.items()}
            for m, d in out.items()}


def registered_pages() -> dict:
    """{model or shape: [(page path, tile)]} over docs/registered/*-hist and *-pages (this blind's own left out)."""
    out: dict = {}
    for d in sorted((REPO / "docs" / "registered").iterdir()):
        if not d.is_dir() or d.name == Path(PAGES).name or not (d.name.endswith("-hist") or d.name.endswith("-pages")):
            continue
        for f in sorted(d.glob("*.json")):
            try:
                doc = json.loads(f.read_text())
            except ValueError:
                continue
            if not isinstance(doc, dict):
                continue
            rel = str(f.relative_to(REPO))
            bm = doc.get("block_m")
            if doc.get("model") and bm in (64, 128):
                out.setdefault(doc["model"], []).append((rel, f"BLOCK_M {bm} ({d.name})"))
            found: dict = {}
            json_tiles(doc, doc.get("model"), bm if isinstance(bm, int) else None, found)
            for k, ts in found.items():
                for t in sorted(ts & PREFILL_TILES):
                    out.setdefault(k, []).append((rel, t))
    return out


def census(sessions: list[str] | None = None) -> dict:
    """Per model of moe/spec.py (toy aside), COMPUTED: (a) registered page lists naming it, or a synthetic shape with
    its (E, H, I, k), at a prefill tile; (b) every JSON record of every published session (file path naming the model
    as context; 'model' and 'shape' keys inside) at a prefill tile, by name and by shape; HAS A PREFILL-TILE PAGE on
    any hit. Beside, as disclosures (not prefill pages): the other tiles its records name per session, compile-only
    censuses, merged.csv rows (fused_experts and grouped-GEMM timings, other cards), and prefill-tile records with no
    model or shape context."""
    from moe.spec import MODEL_CONFIGS as MC
    sessions = published_sessions() if sessions is None else list(sessions)
    models = spec_models()
    names = sorted(models, key=len, reverse=True)
    syn = synthetic_shapes()
    key = {m: shape_of_cfg(MC[m]) for m in models}
    reg = registered_pages()
    pub = REPO / "results" / "published"
    recs: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    census_only: dict = collections.defaultdict(set)
    for s in sessions:
        root = pub / s
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*.json")):
            if not f.is_file() or f.stat().st_size > 50e6:
                continue
            try:
                doc = json.loads(f.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            rel = str(f.relative_to(root))
            found: dict = {}
            json_tiles(doc, path_model(rel, names), None, found)
            compile_only = "tile-census" in rel or rel.split("/")[-1] == "tile-census.json"
            for k, ts in found.items():
                if compile_only:
                    census_only[k].add(s)
                else:
                    recs[k][s] |= ts
    out = {}
    for m in models:
        same = [n for n, sh in syn.items() if sh == key[m]] + [x for x in models if x != m and key[x] == key[m]]
        a = [f"{p} ({t})" for k in [m, *same] for p, t in reg.get(k, [])]
        b = {f"{s} ({k})": sorted(ts & PREFILL_TILES) for k in [m, *same] for s, ts in recs.get(k, {}).items()
             if ts & PREFILL_TILES}
        other = {s: sorted(ts - PREFILL_TILES) for s, ts in recs.get(m, {}).items() if ts - PREFILL_TILES}
        out[m] = {"E, H, I, k": list(key[m]),
                  "synthetic shapes or models with its (E, H, I, k)": same,
                  "registered page lists at a prefill tile": a,
                  "published records at a prefill tile": b,
                  "verdict": "NO PREFILL-TILE PAGE" if not a and not b else "HAS A PREFILL-TILE PAGE",
                  "disclosure: its published records at other tiles, per session": other,
                  "disclosure: compile-only censuses naming it": sorted(census_only.get(m, set()))}
    mr = merged_rows(sessions, models)
    for m in models:
        out[m]["disclosure: merged.csv rows (fused_experts and grouped-GEMM timings)"] = mr[m]
        big = sorted(s for s, x in mr[m].items() if x["num_tokens"][1] > 512)
        out[m]["prefill-size timings on other cards (merged.csv rows past 512 tokens, where vLLM's default is a "
               "prefill tile)"] = big
        out[m]["unseen at prefill"] = out[m]["verdict"] == "NO PREFILL-TILE PAGE" and not big
    unattributed = {s: sorted(ts & PREFILL_TILES) for s, ts in recs.get(None, {}).items() if ts & PREFILL_TILES}
    return {"sessions scanned": sessions, "per model": out,
            "prefill-tile records with no model or shape context (by session)": unattributed,
            "registered synthetic shapes equal to a model of moe/spec.py": sorted(
                n for n, sh in syn.items() if sh in set(key.values()))}


def census_table(cen: dict) -> list[str]:
    """One line per model: verdict, prefill-tile hits, other timings (sessions), merged.csv rows."""
    L = []
    for m, x in cen["per model"].items():
        hits = len(x["registered page lists at a prefill tile"]) + len(x["published records at a prefill tile"])
        L.append(f"{m}: {x['verdict']} (prefill-tile hits {hits}; other-tile sessions "
                 f"{len(x['disclosure: its published records at other tiles, per session'])}; merged.csv sessions "
                 f"{len(x['disclosure: merged.csv rows (fused_experts and grouped-GEMM timings)'])})")
    return L


def instrument_check() -> dict:
    """B3's instrument (NATIVE and SHARED at COPIES declared copies) against R3's scan-kernel bound, per model: SHARED
    declares E x COPIES experts, and R3 refuses 1024 or more (its dry run's refusal, the rule restated)."""
    from moe.spec import MODEL_CONFIGS as MC
    return {m: {"E": MC[m].num_experts, "shared declares": MC[m].num_experts * COPIES,
                "runs": MC[m].num_experts * COPIES < 1024} for m in spec_models()}


# ================================================================== the census gate (on the VM)

CENSUS_LABEL = "tp"
CENSUS_TILES = (TILE,)
CENSUS_REF_MODEL = "olmoe-1b-7b"


def census_check(ref: dict, got: dict, facts, fields) -> dict:
    """This blind's compile-only census (label tp) against A2's merged census: each model's entry at (TILE, w1) and
    (TILE, w2) equals the reference model's (OLMoE-1B-7B: the same top_k 8, which the config tuple carries) on every
    compiled fact (tile_census_a2.FIELDS), and no MMA path mismatch. Any drift is a FAIL (a STOP)."""
    refd = {(e["tile"], e["gemm"]): e for e in ref.get("entries") or [] if e["model"] == CENSUS_REF_MODEL}
    by = {(e["model"], e["tile"], e["gemm"]): e for e in got.get("entries") or []}
    drift = {m: [] for m in MODELS}
    compared = 0
    for m in MODELS:
        for t in CENSUS_TILES:
            for g in ("w1", "w2"):
                r = refd.get((t, g))
                if r is None:
                    drift[m].append(f"{t}|{g}: no {CENSUS_REF_MODEL} entry in A2's merged census")
                    continue
                e = by.get((m, t, g))
                if e is None:
                    drift[m].append(f"{m}|{t}|{g}: not compiled on this board")
                    continue
                compared += 1
                d = [f for f in fields if facts(e)[f] != facts(r)[f]]
                if d:
                    drift[m].append(f"{m}|{t}|{g}: {', '.join(d)} differ from {CENSUS_REF_MODEL}'s")
    bad = list(got.get("path_mismatch") or [])
    why = (["MMA path mismatch: " + "; ".join(map(str, bad[:4]))] if bad else []) + [x for m in MODELS for x in drift[m]][:8]
    occ = {f"{e['model']}|{e['tile']}|{e['gemm']}": (e.get("occupancy") or {}).get("ctas_per_sm")
           for e in got.get("entries") or [] if e["model"] in MODELS and e["tile"] in CENSUS_TILES}
    return {"verdict": "FAIL" if why else "PASS", "why": "; ".join(why), "compared": compared, "drift": drift,
            "path_mismatch": bad, "ctas_per_sm": dict(sorted(occ.items())), "reference": CENSUS_REF_MODEL}


def census_mods():
    for p in (str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "tileA2")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tile_census_a2 as TCA  # noqa: E402
    return TCA


def find_census(results: Path, label: str = CENSUS_LABEL) -> Path | None:
    hits = sorted(Path(results).rglob(f"*-{B.CARD}-tile-census-*-{label}/tile-census.json"))
    return hits[0] if len(hits) == 1 else None


def census_gate(results: Path, repo: Path = REPO) -> dict:
    TCA = census_mods()
    p = find_census(results)
    if p is None:
        return {"gate": "census", "verdict": "FAIL", "why": f"no single tile-census.json (label {CENSUS_LABEL})"}
    try:
        got = json.loads(p.read_text())
        ref = json.loads((Path(repo) / B.A2_MERGED_CENSUS).read_text())
        rec = census_check(ref, got, TCA.facts, TCA.FIELDS)
    except (ValueError, KeyError, TypeError, AttributeError) as ex:
        return {"gate": "census", "verdict": "FAIL", "why": f"the census is malformed ({type(ex).__name__}: {ex})"}
    return {"gate": "census", "file": str(p), "sha256": sha256(p), **rec}


GATE_RECORDS = {"census": "gate-v7pcensus-1410.json", "sag": "gate-b3sag-1410.json"}


# ================================================================== the units, the plan and the rental

def units() -> list[dict]:
    """B3's unit form: the prelude, calibrate, the census and its gate, the two f128 sag pages, G-BAND and the sag gate,
    then the timed pages (every model's G-1 page, then every G-16 page); every unit nodrop."""
    pre = [
        {"line": f"- prelude lock-mhz={LOCK} power-limit-w={POWER_W} est=2 cap=15", "est": 2},
        {"line": f"mixtral-8x7b calibrate label={CENSUS_LABEL} nodrop=1 est=2 cap=30", "est": 2},
        {"line": (f"- tilecensus label={CENSUS_LABEL} tiles={','.join(CENSUS_TILES)} models={','.join(sorted(MODELS))} "
                  "nodrop=1 est=4 cap=15"), "est": 4},
        {"line": "- gate label=gcen gate=v7pcensus nodrop=1 est=1 cap=5", "est": 1},
        {"line": ("olmoe-1b-7b bytes label=f128 block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3 "
                  "byte-groups=64 nodrop=1 est=3 cap=20"), "est": 3},
        {"line": ("mixtral-8x7b bytes label=f128 block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3 "
                  "byte-groups=64 nodrop=1 est=7 cap=25"), "est": 7},
        {"line": "- gate label=gband gate=band nodrop=1 est=1 cap=5", "est": 1},
        {"line": "- gate label=gsag gate=b3sag nodrop=1 est=1 cap=5", "est": 1},
    ]
    timed = []
    specs = page_specs()
    for G in (1, 16):
        for m, g, ns, name in specs:
            if g != G:
                continue
            est = 7 * len(ns)
            timed.append({"line": (f"{m} timed label=t128g{G} block-m=128 block-n=128 block-k=64 num-warps=8 "
                                   f"num-stages=3 histogram={page_path(name)} arms=native,shared "
                                   f"declared-copies={COPIES} timed-groups={G} slip-policy=cell g1-rule=r7-median-run3 "
                                   f"nodrop=1 est={est} cap={3 * est}"), "est": est})
    return pre + timed


def unit_lines() -> list[str]:
    return [u["line"] for u in units()]


def rental() -> dict:
    """d2common.rental's rule (units x 1/0.84 plus 25 min boot and setup; deadline wall + 15 rounded to 5; hard cap 15
    later) on the plan's minutes; cost at the inherited d2common.RATE and at Lambda's list price."""
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "decode"))
    import d2common as D2  # noqa: E402
    r = D2.rental(sum(u["est"] for u in units()))
    r.update({"list_rate": f"gpu_1x_gh200 at ${LIST_RATE}/h (Lambda's list price, 2026-11-18)",
              "cost_usd_at_list": round(r["wall_min"] / 60 * LIST_RATE, 2),
              "cost_at_hard_cap_usd_at_list": round(r["hard_cap_after_launch_min"] / 60 * LIST_RATE, 2),
              "minutes": ("B3's per-unit minutes (its ledger, 2026-11-05: 7 min a tread of 6 cells on every timed page "
                          "at 0.5 to 3 ms a cell; the f128 pages 3 and 7, the census 4, the gates 1 each); this "
                          "blind's cells are 0.12 to 1.7 ms (A2's par)"),
              "drop order": "none: every unit is nodrop (no DESCRIPTIVE unit is planned)"})
    return r


def plan_header(draft: bool) -> list[str]:
    hdr = ["# MODEL v7'S ONE PREFILL BLIND RENTAL: v7 (every switch on, RR at G 1) frozen, round 5's prefill branch with",
           "# the refit4 fix, on three models no prefill-tile page has timed (Granite-3.0-1B-A400M, Granite-3.0-3B-A800M,",
           "# Qwen2-57B-A14B-tp8), B3's recipe and instrument at 128x128x64 w8 s3, G 1 and G 16.",
           f"# Registration {NAME}. GENERATED by scripts/scoring/v7pblind/register_v7p_blind.py; do not edit by hand.",
           f"# Run by B3's driver plus lines marked V7P: bash {LAUNCHER} start ... --plan {PLAN} --driver {DRIVER}",
           "# STOPS before any timed unit: power 700 W and the 1410 lock read back (prelude), the census gate (v7pcensus:",
           "# each model's compiled facts equal OLMoE's in A2's merged census), G-BAND, the sag gate (b3sag).",
           "# Every unit is nodrop. The THIRD reopening of prefill, against the v7 gate's PREFILL NO-GO (REFUSED); the",
           "# owner's choice of 2026-11-18. One blind, pass or fail; then the stage stops and is reported either way.",
           f"# Bars rms {BARS['rms_max']:.1%}, worst {BARS['max_abs']:.1%}, mean {BARS['mean_max']:.2%} per model and "
           "pooled (the working-model bar)."]
    if draft:
        hdr.append(DRAFT_LINE)
    return hdr


def plan_text(draft: bool) -> str:
    return "\n".join(plan_header(draft) + unit_lines()) + "\n"


# ================================================================== the freeze

OWN_CODE = (
    "scripts/scoring/v7pblind/v7pcommon.py", "scripts/scoring/v7pblind/register_v7p_blind.py",
    "scripts/scoring/v7pblind/score_blind_v7p.py", GATE_SCRIPT, f"scripts/{DRIVER}", f"scripts/{PARENT_DRIVER}",
    LAUNCHER, PARENT_LAUNCHER, "scripts/setup_vm.sh", "scripts/vm_results_push.sh", "scripts/locked_r3.py",
    "scripts/r3_tile.py", "scripts/private_weight_reference.py", "scripts/tile_census_a2.py", "scripts/tile_census.py",
    "scripts/tileB3_gate.py", "scripts/tileA2_gate.py", "scripts/skew_synth.py", "scripts/skew_synth_tile.py",
    "docs/registered/2026-10-07-rental5-skew-hist/synth_fit.json", "moe/bench/tile_resolve.py",
    "moe/bench/hardware/vllm_configs/SHIPPED_FILE_NAMES.txt", "scripts/scoring/tileB3/b3common.py",
    "scripts/scoring/tileB3/read_b3.py", "scripts/scoring/tileB3/b3.predictions.json",
    "scripts/scoring/tileB3/tileB3.score.json", "scripts/scoring/cal2/read_c2.py", "scripts/scoring/tileA/tcommon.py",
    B.A2_MERGED_CENSUS, f"{RERUN}.py", f"{RERUN}.json", f"{RERUN}.txt", "scripts/scoring/v7/gate_v7.out.txt",
    "scripts/scoring/v7/gate_v7.json", f"docs/registered/{B3_NAME}.json", f"docs/registered/{B3_NAME}.txt",
    f"docs/registered/{GATE_NAME}.json", f"docs/registered/{GATE_NAME}.txt",
    f"docs/registered/{DECODE_BLIND_NAME}.txt", f"docs/registered/{DOMAIN_BLIND_NAME}.txt",
    "docs/registered/2026-10-14-decodeB-blind-gh200.txt", "docs/registered/2026-10-20-tileB2-e1tile-gh200.json",
)


def own_files() -> tuple:
    return tuple(page_path(name) for _m, _G, _ns, name in page_specs()) + OWN_CODE


def gate_registration() -> dict:
    return json.loads((REPO / "docs" / "registered" / f"{GATE_NAME}.json").read_text())


def freeze(root: Path = REPO) -> dict:
    g = gate_registration()["freeze"]
    own = {p: sha256(Path(root) / p) for p in own_files() if p not in g["files"]}
    return {"statement": ("the scorer (score_blind_v7p.py) refuses unless every file of the v7 gate registration's "
                          "freeze has its sha256 there (v7, round 5's loader and the vendored round 5 among them) and "
                          "v7's constants hash is the one registered there, and every file under 'own' has its sha256 "
                          "here (the pages, this blind's code, B3's driver, gate and rules, the launcher and its parent, "
                          "the pinned R3 and census tools, the re-run whose fix is registered, the quoted registrations); "
                          "the plan's sha256 is computed by the registration builder from the same document"),
            "gate registration": {"name": GATE_NAME, "json sha256": sha256(REPO / f"docs/registered/{GATE_NAME}.json"),
                                  "files": len(g["files"]), "constants_sha256": g["constants_sha256"]},
            "own": own}


def verify_freeze(reg: dict, root: Path = REPO) -> list[str]:
    """Every difference from the registered freeze (empty: frozen); v7bcommon.verify_freeze's rule."""
    fz = reg.get("freeze") or {}
    bad = []
    try:
        gp = Path(root) / "docs" / "registered" / f"{GATE_NAME}.json"
        if not gp.is_file() or sha256(gp) != fz["gate registration"]["json sha256"]:
            bad.append(f"{GATE_NAME}.json is not the registered bytes")
        else:
            bad += GV.verify_freeze(json.loads(gp.read_text()), root=root)
        for p, want in fz["own"].items():
            f = Path(root) / p
            if not f.is_file():
                bad.append(f"{p}: missing")
            elif sha256(f) != want:
                bad.append(f"{p}: sha256 {sha256(f)[:12]} is not the registered {want[:12]}")
        if V7.constants_sha256() != fz["gate registration"]["constants_sha256"]:
            bad.append("v7's constants hash moved")
    except (KeyError, TypeError) as ex:
        bad.append(f"the registration's freeze block is malformed ({type(ex).__name__}: {ex})")
    return bad


def committed_draft() -> dict | None:
    """This registration's JSON as committed at HEAD (the DRAFT a --final must agree with), or None."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{REG_JSON}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


# ================================================================== the finalisation (after the rental, before any blind page)

def blind_path(p: str) -> bool:
    """A path finalisation must not open: inside a timed unit's gaps-* directory, or naming a blind model (its timed
    pages), except the compile-only census (B3's rule)."""
    return "/gaps-" in p or (any(m in p for m in MODELS) and "-tile-census-" not in p)


def chain_sag_fits(results: Path) -> dict:
    """{model: (A, tau)} by the frozen chain's estimator on the board's f128 pages (r4.sag_page_fit)."""
    C = R5.chain()
    return {m: tuple(C["r4"].sag_page_fit(Path(results), m)) for m in B.EST_MODELS}


def finalise_pars(results: Path, sag_record: dict | None) -> dict:
    """The board's 128x128 par from its two f128 pages, by the repo's estimator AND the frozen chain's, equal to each
    other and to the sag gate's record (b3common.PAR_RTOL, each page's sha256); raises ValueError with the reason."""
    if sag_record is None or sag_record.get("verdict") != "PASS":
        raise ValueError(f"the sag gate did not PASS on this board ({None if sag_record is None else sag_record.get('why')})")
    mine = {m: B.page_fit(results, m, B.EST_UNIT) for m in B.EST_MODELS}
    frozen = chain_sag_fits(results)
    if any((mine[m]["A_us"], mine[m]["tau_us"]) != frozen[m] for m in B.EST_MODELS):
        raise ValueError(f"the repo's sag estimator and the frozen chain's disagree on the f128 pages: {mine} against {frozen}")
    rec = B.sag_check(mine, B.SAG_RANGE, B.EST_UNIT)
    if rec["verdict"] != "PASS" or not B.par_close(rec["par"], sag_record.get("par")) or any(
            mine[m]["sha256"] != ((sag_record.get("per_page") or {}).get(m) or {}).get("sha256") for m in B.EST_MODELS):
        raise ValueError("the sag gate record's par or page sha256 is not what the f128 pages give")
    par = tuple(R5.chain()["r4"].par([frozen[m] for m in B.EST_MODELS]))
    if not B.par_close(par, rec["par"]):
        raise ValueError(f"r4.par {par} is not the sag gate's {rec['par']}")
    return {"par": list(par), "per_page": {m: {k: mine[m][k] for k in ("A_us", "tau_us", "rms_us", "cells", "sha256")}
                                           for m in B.EST_MODELS},
            "inside_held_out_par_span (DESCRIPTIVE)": rec.get("inside_held_out_par_span (DESCRIPTIVE)")}


# ================================================================== the bars (pure)

def stats(v) -> dict:
    return B.stats(v)


def bar(v, rule: dict = BARS) -> dict:
    return B.bar(v, rule)
