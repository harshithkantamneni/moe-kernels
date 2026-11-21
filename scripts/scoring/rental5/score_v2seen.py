#!/usr/bin/env python
"""Rental 5 part 1: every earlier held-out time test re-predicted under model v2, SEEN
diagnostic only, no verdict (docs/registered/2026-10-07-rental5-v2-gh200.json).

    python scripts/scoring/rental5/score_v2seen.py <repo> <out>

For each test it runs the timing model twice, as registered (model M, DEAD_CTA_NS 1.333 and
k_w in the dead window) and under v2 (`r3_timing_model.dead_model("v2")`: d 0.995 ns, kappa
0.325), the 8x7B 2026-09-27 source REFIT under each (T0, c, bw, s_small, s_block), and every
target priced from its own counted bytes (rental 3's E1 also from the registration's predicted
bytes), nothing fitted on a target:
  - the cross-model targets of scripts/scoring/crossmodel (8x22B, Qwen2-57B, OLMoE, JetMoE,
    Qwen1.5, Phi-3.5, Granite-3.0-3B), their 1710-lock timed pages and counter pages;
  - rental 3's E2 (counted bytes) and E1 (registered q_pred), the core G = 3, 8, 32 pages;
  - rental 4's A1 and A2 (score_dead on the published rental-4 pages): measured Delta beside
    M's and D's (v2's) registered Delta;
  - the floor-only captures (jetmoe-floor, mixtral-floor) carry no dead term (cycles per CTA
    k-step), so they are listed as unaffected.
The set scored is SHARED + PRIVATE under rental 3's host rule (T_pred + F240 >= 0.40 ms), as
design-r5/OFFSET.md section 2 and 3 computed it. It must reproduce the registration's expected
rms table to 0.01 point, or it exits 1.

LABELS. 8x7B is in-sample (CAL). 8x22B, Qwen2-57B and OLMoE are IN-SAMPLE (d, kappa): their
counter pages are in the CAL-counters set D was fitted on. Every page read is SEEN.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402

PART = "v2"
F240, HOST = 0.067551, 0.40
TARGETS = {
    "mixtral-8x22b": "2026-09-28-nvidia_gh200_480gb-8x22b-session",
    "qwen2-57b-a14b": "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session",
    "olmoe-1b-7b": "2026-09-29-nvidia_gh200_480gb-olmoe-session",
    "jetmoe-8b": "2026-09-29-nvidia_gh200_480gb-jetmoe-session",
    "qwen1.5-moe-a2.7b": "2026-09-29-nvidia_gh200_480gb-qwen1.5-session",
    "phi-3.5-moe": "2026-09-29-nvidia_gh200_480gb-phi3.5-session",
    "granite-3.0-3b-a800m": "2026-09-30-nvidia_gh200_480gb-granite-session",
}
IN_SAMPLE_DK = ("mixtral-8x22b", "qwen2-57b-a14b", "olmoe-1b-7b")
R3S = "2026-10-06-nvidia_gh200_480gb-rental3-session"
R4S = "2026-10-07-nvidia_gh200_480gb-rental4-session"
FLOORS = ("jetmoe-floor", "mixtral-floor")
#: targets the timing model refused (printed, with the reason)
SKIPPED: dict = {}


def label_of(key: str) -> str:
    if key.startswith("mixtral-8x7b"):
        return "CAL (in-sample)"
    if key in IN_SAMPLE_DK:
        return "IN-SAMPLE (d, kappa): its counter pages are in D's CAL-counters fit set; SEEN"
    return "held-out, SEEN"


def _targets(repo: Path):
    import cross_model_score as XS
    import r3_timing_model as TM
    pub = repo / "results" / "published"
    out = {}
    for model, sess in TARGETS.items():
        s = pub / sess / "results"
        gaps = s / "gaps-nvidia_gh200_480gb" / "private_weight_reference"
        if not gaps.is_dir():
            continue
        TM.set_model(model)
        timed = [d for d in sorted(gaps.iterdir())
                 if (pg := TM.discover_timed([d])) and all(p.locked and next(iter(p.clocks)) == 1710.0 for p in pg)]
        cnt = sorted(s.glob("*-r3-counters"))
        if not timed or not cnt:
            continue
        try:
            tgt = XS._build(timed, cnt[0] / "lock1710")
        except TM.Refused as exc:
            SKIPPED[model] = f"REFUSED by the timing model: {exc}"
            continue
        out[model] = (model, tgt["cells"], tgt["ctx"])
    r3 = pub / R3S / "results"
    pw = r3 / "gaps-nvidia_gh200_480gb-qwen2-57b-a14b-tp8-e2e" / "private_weight_reference"
    cnt = r3 / "2026-10-06-nvidia_gh200_480gb-qwen2-57b-a14b-tp8-e2e-r3-counters" / "lock1710"
    if pw.is_dir():
        reg = json.loads((repo / "docs/registered/2026-10-05-rental3-e2e-gh200.json").read_text())
        TM.set_model("qwen2-57b-a14b-tp8")
        admit0 = TM.admit

        def admit_as_viewed(pages):
            # the E2 view counts the V5-only G = 32 page (rental 3's post-page reading fix)
            return admit0([p if p.label == TM.PTF.VALID else
                           dataclasses.replace(p, label=TM.PTF.VALID, failed=()) for p in pages])
        TM.admit = admit_as_viewed
        try:
            core = [d for d in pw.iterdir() if any(f"-g{g}-" in d.name for g in (3, 8, 32))]
            tgt = XS._build(core, cnt)
        finally:
            TM.admit = admit0
        cells, ctx = tgt["cells"], tgt["ctx"]
        out["rental3 E2 (counted)"] = ("qwen2-57b-a14b-tp8", cells, ctx)
        pc = []
        for c in cells:
            q = reg["q_pred"].get(f"{'shared' if c.arm == 'native' else c.arm} G={c.G} n={c.n}")
            if not q or q["w1"] is None:
                continue
            reads = {g: q[g] * TM.BYTE_MODEL[f"W_{g}"] + c.n * TM.BYTE_MODEL[f"operand_per_tile_{g}"] for g in TM.GEMMS}
            sig = {g: reads[g] / TM.window(c.arm, c.declared, c.G, c.n, g, TM.window_width(
                0.5, ctx.sms, ctx.occupancy[g])).reads for g in TM.GEMMS}
            pc.append(dataclasses.replace(c, reads=reads, sigma=sig))
        out["rental3 E1 (predicted)"] = ("qwen2-57b-a14b-tp8", pc, ctx)
    TM.set_model("mixtral-8x7b")
    return out


def _rows_stats(rows: list) -> dict:
    keep = [r for r in rows if r["pred"] + F240 >= HOST]
    sp = [r for r in keep if r["arm"] in ("shared", "private")]
    rel = [r["pred"] / r["meas"] - 1 for r in sp]
    s = C5.stats(rel)
    s["beyond_5pct"] = sum(abs(x) > 0.05 for x in rel)
    s["mean_us_by_arm"] = {a: C5.r6(sum((r["pred"] - r["meas"]) * 1e3 for r in keep if r["arm"] == a)
                                    / max(1, sum(r["arm"] == a for r in keep)))
                           for a in ("native", "shared", "private") if any(r["arm"] == a for r in keep)}
    s["secondary"] = C5.secondary([{"pred": r["pred"], "meas": r["meas"], "cluster": f"G{r['G']}",
                                    "stratum": f"{r['arm']} n{r['n']}"} for r in sp])
    return s


def timing_tests(repo: Path) -> dict:
    import cores_heldout_predict as CP
    import r3_timing_model as TM
    tg = _targets(repo)
    res = {}
    for variant in ("m", "v2"):
        TM.set_model("mixtral-8x7b")
        with TM.dead_model(variant):
            src = TM.build(TM.build_parser().parse_args([*map(str, CP.source_pages()), "--counters", str(CP.C27)]))
            fit = src["main"]
            tests = {"mixtral-8x7b (in-sample)": [
                {"arm": c.arm, "G": c.G, "n": c.n, "meas": c.ms, "pred": TM.call_ms(fit.x, c, src["ctx"], fit.k_w)}
                for c in src["cells"]]}
            for key, (model, cells, ctx) in tg.items():
                TM.set_model(model)
                tests[key] = [{"arm": c.arm, "G": c.G, "n": c.n, "meas": c.ms,
                               "pred": TM.call_ms(fit.x, c, ctx, fit.k_w)} for c in cells]
                TM.set_model("mixtral-8x7b")
        res[variant] = {"version": TM.MODEL_VERSION[variant],
                        "source_fit": {k: C5.r6(v) for k, v in fit.params.items()},
                        "tests": {k: dict(_rows_stats(v), label=label_of(k)) for k, v in tests.items()}}
    return res


def dead_contrasts(repo: Path) -> dict:
    """Rental 4's A1 / A2: the published pages' Delta beside M's and D's registered Delta."""
    tree = repo / "results" / "published" / R4S / "results"
    if not tree.is_dir():
        return {"verdict": "NOT SCORED: no rental-4 tree"}
    sd = C5._load("rental4_score_dead", HERE.parent / "rental4" / "score_dead.py")
    res = sd.score(repo, tree)
    reg = C5.C4.registration(repo, "dead")
    out = {}
    for k, c in res["contrasts"].items():
        if "Delta_us" not in c:
            out[k] = {"verdict": c.get("verdict")}
            continue
        riv = reg["predictions_us"][k]["rivals"]
        out[k] = {"model": c["model"], "Delta_meas_us": C5.r6(c["Delta_us"]),
                  "M_pred_us": riv["M"]["total"], "v2_pred_us": riv["D"]["total"],
                  "M_minus_meas_us": C5.r6(riv["M"]["total"] - c["Delta_us"]),
                  "v2_minus_meas_us": C5.r6(riv["D"]["total"] - c["Delta_us"]),
                  "label": "SEEN (rental 4 published); v2's d and kappa are D's, which A1 / A2 tested blind"}
    out["per_gemm_d_printed"] = res.get("per_gemm_d_printed")
    return out


def check(reg: dict, tt: dict) -> list[str]:
    bad = []
    for key, (m, v) in reg["expected_rms_pct"].items():
        for variant, want in (("m", m), ("v2", v)):
            got = tt[variant]["tests"].get(key, {}).get("rms")
            if got is None or abs(100 * got - want) > 0.01 + 1e-9:
                bad.append(f"{key} {variant}: expected {want:.2f}%, got {None if got is None else round(100 * got, 3)}")
    return bad


def _registration(repo: Path) -> dict:
    """The committed registration; before register.py has written it (the build), the
    builder's own dict, which is what register.py commits."""
    try:
        return C5.registration(repo, PART)
    except FileNotFoundError:
        import reg_levers as RL
        return RL.BUILDERS[PART]()


def score(repo: Path) -> dict:
    reg = _registration(repo)
    tt = timing_tests(repo)
    bad = check(reg, tt)
    return {"registration": C5.NAMES[PART], "status": "SEEN DIAGNOSTIC, NO VERDICT",
            "host_rule": {"F240_ms": F240, "host_ms": HOST, "set": "SHARED + PRIVATE"},
            "timing": tt, "dead_contrasts": dead_contrasts(repo),
            "skipped": dict(SKIPPED),
            "floors": {f: "unaffected: a floor capture reads cycles per CTA k-step, no dead term" for f in FLOORS},
            "reproduces_expected": not bad, "mismatches": bad}


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 5 v2 RE-PREDICTIONS ({res['registration']}): {res['status']}",
           "  SHARED + PRIVATE under rental 3's host rule; own counted bytes unless named; every page SEEN"]
    for v in ("m", "v2"):
        t = res["timing"][v]
        out.append(f"  {t['version']}: 8x7B refit " + ", ".join(f"{k} {x:.6g}" for k, x in t["source_fit"].items()))
    keys = list(res["timing"]["m"]["tests"])
    out.append(f"  {'test':<26} {'M rms':>7} {'v2 rms':>7} {'M worst':>8} {'v2 worst':>8} {'v2 bias':>8}  label")
    for k in keys:
        a, b = res["timing"]["m"]["tests"][k], res["timing"]["v2"]["tests"][k]
        f = (lambda x: f"{100 * x:+.2f}%" if x is not None else "-")
        out.append(f"  {k:<26} {100 * a['rms']:6.2f}% {100 * b['rms']:6.2f}% {f(a['worst']):>8} {f(b['worst']):>8} "
                   f"{f(b['mean']):>8}  {b['label']}")
        out += C5.secondary_lines(b["secondary"], indent="      v2 ")[:2]
    out.append("  rental 4 dead contrasts (us): " + json.dumps(res["dead_contrasts"], default=str))
    for k, v in res.get("skipped", {}).items():
        out.append(f"  {k}: not scored: {v}")
    out.append("  floors: " + "; ".join(f"{k} {v}" for k, v in res["floors"].items()))
    out.append("  expected table: " + ("REPRODUCED to 0.01 point" if res["reproduces_expected"]
                                       else "NOT REPRODUCED: " + "; ".join(res["mismatches"])))
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 2:
        print(__doc__)
        return 2
    repo, out = map(Path, a)
    res = score(repo)
    print(C5.write_score(out, "v2seen", res, lines(res)))
    return 0 if res["reproduces_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
