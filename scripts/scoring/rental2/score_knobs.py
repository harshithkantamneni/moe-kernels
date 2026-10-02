#!/usr/bin/env python
"""Score rental 2's part 1 (docs/registered/2026-10-01-rental2-knobs-gh200.json):
B's knobs (the lag law L2r against NL and ST) and C's BLOCK_K separator T5.

    python scripts/scoring/rental2/score_knobs.py <repo> <tree> <out>

Reads only the registered JSON under <repo> and the r3c pages under <tree>
(found by name, `*-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1710/
r3c-g<G>.json`). Writes <out>/knobs.score.{json,txt}. Exits 0 whatever the
verdicts; 2 only when the registration cannot be read. Written and tested on
synthetic pages before any rental-2 page existed; no judgment is made here
that the registration does not state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

PART = "knobs"


def find_page(tree: Path, model: str, label: str, G: int, sub: str = "lock1710") -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{CM.CARD}-{model}-{label}-r3-counters/{sub}/r3c-g{G}.json"))
    return hits[0] if len(hits) == 1 else None


def load(path: Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def _n(d: dict, n: int):
    return d.get(n, d.get(str(n)))


def cell_bands(reg: dict, base: dict, knob: dict, gemm: str, n: int) -> dict:
    """The three bands at one n, from the same-board base page and the knob page's
    own W_c (survival dicts from scripts/l2_survival.py)."""
    law = reg["law_L2r"]
    lam = law["lambda_by_n_SEEN"][str(n)]
    knots = law["s_cap_knots_by_n"][str(n)]
    b, k = base[gemm], knob[gemm]
    x = b["x"]
    return CM.bands(_n(b["s"], n), CM.s_cap(x, knots), k["d"] - b["d"], lam=lam["lam"],
                    lam_sigma=lam["sigma"], lam_steep=reg["rivals"]["lambda_steep_by_n"][str(n)],
                    past_edge=x >= 0.92, s_noise=reg["bands"]["s_noise"],
                    nl_half=reg["bands"]["nl_half"])


def page_controls(reg: dict, surv: dict, model: str) -> list[str]:
    """Why a page's SELECTED verdicts are demoted: V6 not PASS, F / Mn out of band."""
    why = []
    if surv.get("v6") not in (None, "PASS"):
        why.append(f"V6 {surv['v6']} (FLAGGED)")
    lo, hi = reg["controls"]["F_over_Mn"]["band"]
    bad = [(g, n, round(v, 3)) for g in ("w1", "w2")
           for n, v in surv[g]["fabric_over_mn"].items() if v is not None and not lo <= v <= hi]
    if bad:
        why.append(f"PRIVATE F/Mn outside [{lo}, {hi}] on {bad}")
    return why


def tp2_w1_control(reg: dict, base: dict, knob: dict) -> str:
    """'' when tp2 w1 sits at or under its own L2r upper edge at every n = 4..9."""
    over = []
    for n in range(4, 10):
        hi = cell_bands(reg, base, knob, "w1", n)["L2r"][1]
        v = _n(knob["w1"]["s"], n)
        if v > hi:
            over.append((n, round(v, 3), round(hi, 3)))
    return f"tp2 w1 above its L2r upper edge at {over}" if over else ""


def score_test(reg: dict, name: str, spec: dict, base: dict, knob: dict) -> dict:
    g, ns, need = spec["gemm"], spec["n"], spec["need"]
    hyps = spec["hypotheses"]
    by_n, meas = {}, {}
    for n in ns:
        b = cell_bands(reg, base, knob, g, n)
        if "LAG" in hyps:
            b = {"LAG": CM.lag_union(b), "NL": b["NL"]}
        by_n[n] = b
        meas[n] = _n(knob[g]["s"], n)
    out = CM.select(meas, by_n, hyps, ns, need=need)
    out.update(measured={str(n): meas[n] for n in ns},
               bands={str(n): by_n[n] for n in ns},
               W_c={"base": base[g]["W_c"], "knob": knob[g]["W_c"]},
               dd=knob[g]["d"] - base[g]["d"], x=base[g]["x"])
    return out


def score_t5(reg: dict, pages: dict, *, price=None) -> dict:
    """`pages` maps BK64 / BK32 / BK128 to page dicts (None when missing).
    `price(page, params, content_a)` -> f at n = 9 on that page's geometry;
    the default re-prices with wave_split_bytes."""
    import dram_counter_route as DCR
    t5 = reg["T5"]
    n = int(t5["n"])

    def f_meas(page, nn):
        return CM.f_of_q(DCR.r3_q(page)["private"]["w2"][nn], nn)
    if pages.get("BK32") is None or pages.get("BK128") is None:
        return {"verdict": "NOT SCORED: a BK32 or BK128 page is missing"}
    if price is None:
        import wave_split_bytes as W

        def price(page, prm, ca):
            return CM.price_f(W.page_geometry(page), prm, ca, n=n, G=int(t5["G"]))
    f = {k: {str(nn): f_meas(p, nn) for nn in (8, 9)} for k, p in pages.items() if p is not None}
    rho = f["BK128"][str(n)] / f["BK32"][str(n)]
    rho8 = f["BK128"]["8"] / f["BK32"]["8"] if f["BK32"]["8"] else None
    cands = CM.t5_candidates(t5)
    board = None
    if pages.get("BK64") is not None:
        bc = t5["board_check"]
        rel = f["BK64"][str(n)] / bc["f_rental1"] - 1
        board = {"f": f["BK64"][str(n)], "rel": rel, "ok": abs(rel) <= bc["tolerance_rel"]}
    per = {}
    for name, (prm, ca) in cands.items():
        centre = price(pages["BK128"], prm, ca) / price(pages["BK32"], prm, ca)
        hw = t5["registered_at_default_Wc"][name]["half_width"]
        band = [centre - hw, centre + hw]
        held = band[0] <= rho <= band[1]
        v = "HELD" if held else "FALSIFIED"
        if name == "R2h" and (board is None or not board["ok"]):
            v = f"INCONCLUSIVE (board check {'missing' if board is None else 'failed'}; measured {v})"
        per[name] = {"centre": centre, "band": band, "verdict": v}
    return {"f": f, "rho_BK": rho, "rho_BK_n8_printed": rho8, "board_check": board,
            "candidates": per}


def score_view(repo: Path, tree: Path, view) -> dict:
    """Every verdict on the pages `view` counts (the addendum's ALL or CLEAN)."""
    import l2_survival as L
    reg = CM.registration(repo, PART)
    G = int(reg["pages"]["G"])
    cache: dict = {}

    def surv(model, label):
        key = (model, label)
        if key not in cache:
            p = find_page(tree, model, label, G)
            page = view.load(p)
            cache[key] = (L.survival(page), page, str(p) if p else None) if page else (None, None, None)
        return cache[key][0]
    res = {"registration": CM.NAMES[PART], "tests": {}, "controls": {}, "pages": {}}
    # page-level controls on every page the tests read
    flags: dict = {}
    for name, spec in reg["tests"].items():
        for pg in ([spec["page"]] if "page" in spec else spec.get("pages", [])):
            for lab in (pg[1], spec.get("base")):
                if lab and surv(pg[0], lab) is not None:
                    flags.setdefault((pg[0], lab), page_controls(reg, surv(pg[0], lab), pg[0]))
    neg = reg["controls"]["tp8 w2 s8 negative control"]
    tp8 = surv(*neg["page"])
    neg_ok = None
    if tp8 is not None:
        flags.setdefault(tuple(neg["page"]), page_controls(reg, tp8, neg["page"][0]))
        low = [(n, round(v, 3)) for n, v in tp8["w2"]["s"].items() if 4 <= int(n) <= 9 and v < 0.95]
        neg_ok = not low
        res["controls"]["tp8 w2 s8"] = {"ok": neg_ok, "below_0.95": low}
    else:
        res["controls"]["tp8 w2 s8"] = {"ok": None, "why": "page missing"}
    for knob in ("l2s8", "l2s6"):
        b, k = surv("mixtral-8x7b-tp2", "l2base"), surv("mixtral-8x7b-tp2", knob)
        if b is not None and k is not None:
            why = tp2_w1_control(reg, b, k)
            res["controls"][f"tp2 w1 on {knob}"] = {"ok": not why, "detail": why}
            if why:
                # L2r's own prediction: it demotes an L2r (or LAG) selection only
                flags.setdefault(("mixtral-8x7b-tp2", knob), []).append("L2R-ONLY " + why)
    res["controls"]["page_flags"] = {f"{m} {lab}": v for (m, lab), v in flags.items()}
    for name, spec in reg["tests"].items():
        if "hypotheses" in spec:
            m, lab = spec["page"]
            b, k = surv(m, spec["base"]), surv(m, lab)
            if b is None or k is None:
                res["tests"][name] = {"verdict": "NOT SCORED: a page is missing"}
                continue
            out = score_test(reg, name, spec, b, k)
            demote = flags.get((m, lab), []) + flags.get((m, spec["base"]), [])
            if neg_ok is False:
                demote = demote + ["the tp8 w2 s8 negative control failed"]
            demote = [d for d in demote
                      if not d.startswith("L2R-ONLY ") or out["verdict"] in ("L2r", "LAG")]
            if any("V6" in d for d in demote):
                out["verdict"] = "INCONCLUSIVE (FLAGGED V6)"
            elif demote and out["verdict"] not in ("INCONCLUSIVE",):
                out["verdict_before_controls"] = out["verdict"]
                out["verdict"] = "INCONCLUSIVE"
            out["demoted_by"] = demote
            res["tests"][name] = out
        elif name.startswith("8x7B w2 s8"):
            a, t = surv(*spec["page"]), surv(*spec["against"])
            if a is None or t is None:
                res["tests"][name] = {"verdict": "NOT SCORED: a page is missing"}
                continue
            d = {n: _n(a["w2"]["s"], n) - _n(t["w2"]["s"], n) for n in spec["n"]}
            ok = sum(abs(v) <= 0.05 for v in d.values())
            res["tests"][name] = {"diff": {str(n): v for n, v in d.items()}, "within_0.05": ok,
                                  "verdict": "HELD" if ok >= 4 else "FALSIFIED"}
        elif name.startswith("H2c"):
            out = {}
            b = surv("mixtral-8x7b-tp2", spec["base"])
            for m, lab in spec["pages"]:
                k = surv(m, lab)
                if b is None or k is None:
                    out[lab] = {"verdict": "NOT SCORED: a page is missing"}
                    continue
                for g in spec["gemms"]:
                    same = k[g]["W_c"] == b[g]["W_c"]
                    cen = {}
                    for n in range(4, 10):
                        if same:
                            cen[n] = _n(b[g]["s"], n)
                        else:
                            law = reg["law_L2r"]
                            lam = law["lambda_by_n_SEEN"][str(n)]["lam"]
                            cen[n] = CM.law_value(_n(b[g]["s"], n),
                                                  CM.s_cap(b[g]["x"], law["s_cap_knots_by_n"][str(n)]),
                                                  k[g]["d"] - b[g]["d"], lam, b[g]["x"] >= 0.92)
                    dev = {n: _n(k[g]["s"], n) - cen[n] for n in cen}
                    inn = sum(abs(v) <= 0.05 for v in dev.values())
                    v = ("H2c FALSIFIED (no k-step effect)" if inn >= 4 else
                         "H2c SELECTED" if 6 - inn >= 4 else "INCONCLUSIVE")
                    out[f"{lab} {g}"] = {"W_c_same": same, "deviation": {str(n): x for n, x in dev.items()},
                                         "within_0.05": inn, "verdict": v}
            res["tests"][name] = out
        elif name.startswith("H3"):
            b, k = surv("mixtral-8x7b-tp2", spec["base"]), surv(*spec["page"])
            if b is None or k is None:
                res["tests"][name] = {"verdict": "NOT SCORED: a page is missing"}
                continue
            pb, pk = cache[("mixtral-8x7b-tp2", spec["base"])][1], cache[tuple(spec["page"])][1]
            bb = {(c["n"], g): c["per_gemm"][g]["dram_bytes_read"] for c in pb["cells"]
                  if c["arm"] == "private" for g in ("w1", "w2")}
            kb = {(c["n"], g): c["per_gemm"][g]["dram_bytes_read"] for c in pk["cells"]
                  if c["arm"] == "private" for g in ("w1", "w2")}
            rel = {f"{n}/{g}": kb[(n, g)] / bb[(n, g)] - 1 for (n, g) in bb if (n, g) in kb}
            ds = {g: {n: _n(k[g]["s"], n) - _n(b[g]["s"], n) for n in range(4, 10)} for g in ("w1", "w2")}
            moved = any(sum(abs(v) > 0.03 for v in ds[g].values()) >= 4 for g in ds) or \
                any(abs(v) > 0.01 for v in rel.values())
            still = all(abs(v) <= 0.003 for v in rel.values()) and \
                all(sum(abs(v) <= 0.03 for v in ds[g].values()) >= 4 for g in ds)
            res["tests"][name] = {"private_bytes_rel": rel,
                                  "s_deviation": {g: {str(n): x for n, x in v.items()} for g, v in ds.items()},
                                  "verdict": "H3 SELECTED" if moved else
                                  "H3 FALSIFIED (no address effect)" if still else "INCONCLUSIVE"}
        else:
            m, lab = spec["page"]
            b, k = surv(m, spec["base"]), surv(m, lab)
            res["tests"][name] = ({"verdict": "RECORD", "s": k[spec["gemm"]]["s"], "s_base": b[spec["gemm"]]["s"]}
                                  if b is not None and k is not None else {"verdict": "RECORD: a page is missing"})
    # the partition cross-check, an instrument check
    pc = {}
    for m, lab in reg["partition_cross_check"]["pages"]:
        s = surv(m, lab)
        if s is None:
            pc[f"{m} {lab}"] = {"verdict": "NOT SCORED: page missing"}
            continue
        vals = [(g, k2, v) for g in ("w1", "w2") for k2, v in s[g]["partition"].items() if v is not None]
        if not vals:
            pc[f"{m} {lab}"] = {"verdict": "NOT SCORED: the page carries no partition metrics"}
            continue
        bad = [(g, k2, round(v, 4)) for g, k2, v in vals if abs(v - 1) > 0.05]
        frac = len(bad) / len(vals)
        pc[f"{m} {lab}"] = {"cells": len(vals), "outside": bad, "fraction_outside": frac,
                            "verdict": "AGREE" if frac <= 0.10 else "DISAGREE: far share printed from the direct count only",
                            "far": {g: s[g]["far"] for g in ("w1", "w2")},
                            "far_direct": {g: s[g]["far_direct"] for g in ("w1", "w2")}}
    res["partition_cross_check"] = pc
    t5p = {}
    for k2, (m, lab) in reg["T5"]["pages"].items():
        if k2.startswith("BK"):
            t5p[k2] = view.load(find_page(tree, m, lab, int(reg["T5"]["G"])))
    res["T5"] = score_t5(reg, t5p)
    s8 = view.load(find_page(tree, *reg["T5"]["pages"]["s8 RECORD"], int(reg["T5"]["G"])))
    if s8 is not None:
        import dram_counter_route as DCR
        res["T5"]["s8_record_f9"] = CM.f_of_q(DCR.r3_q(s8)["private"]["w2"][9], 9)
    res["pages"] = {f"{m} {lab}": v[2] for (m, lab), v in cache.items()}
    return res


def score(repo: Path, tree: Path) -> dict:
    """The registered verdicts: each computed on ALL and on CLEAN pages and
    combined by the addendum's rule (common.two_views). V6 and V10 keep the
    handling this part registered and gate neither view."""
    return CM.two_views(lambda v: score_view(repo, tree, v), repo, PART)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 2 PART 1 (knobs), {res['registration']}"]
    for name, t in res["tests"].items():
        out.append(f"  {name}: {t.get('verdict') if isinstance(t, dict) and 'verdict' in t else ''}")
        if isinstance(t, dict) and "measured" in t:
            out.append("    s " + " ".join(f"n{n} {v:.3f}" for n, v in t["measured"].items())
                       + f"  dd {t['dd']:.3f} W_c {t['W_c']}")
            for h, p in t["per_hypothesis"].items():
                out.append(f"    {h}: {p['status']} (exclusive n {p['exclusive']}, outside n {p['outside']})")
            if t.get("demoted_by"):
                out.append(f"    demoted by: {t['demoted_by']}")
        elif isinstance(t, dict) and "verdict" not in t:
            for k, v in t.items():
                out.append(f"    {k}: {v.get('verdict')}")
    out.append(f"  controls: {json.dumps(res['controls'], default=str)}")
    for k, v in res["partition_cross_check"].items():
        out.append(f"  partition {k}: {v.get('verdict')}")
    t5 = res["T5"]
    if "rho_BK" in t5:
        out.append(f"  T5 rho_BK {t5['rho_BK']:.3f} (n = 8 printed {t5['rho_BK_n8_printed']}); board "
                   f"{t5['board_check']}")
        for k, v in t5["candidates"].items():
            out.append(f"    {k}: centre {v['centre']:.3f} band [{v['band'][0]:.3f}, {v['band'][1]:.3f}] {v['verdict']}")
    else:
        out.append(f"  T5: {t5['verdict']}")
    return out + CM.addendum_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    try:
        res = score(repo, tree)
    except FileNotFoundError as exc:
        print(f"REFUSED: {exc}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "knobs.score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = "\n".join(lines(res)) + "\n"
    (out / "knobs.score.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
