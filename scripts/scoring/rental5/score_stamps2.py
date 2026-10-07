#!/usr/bin/env python
"""Score rental 5's timestamps v2 (docs/registered/2026-10-07-rental5-stamps2-gh200.json):
F2 (Delta_epi, T_fix), K2 (R(j), the sync cost) and D2 (dead life, dispatch, hiding kappa).

    python scripts/scoring/rental5/score_stamps2.py <repo> <tree> <out>

Reads under <tree>: each unit's `*-<card>-instr-<model>-<label>/stamps.json` (+ stamps/*.npy)
and its CHOSEN_VARIANT.txt (`id=...`, `spec=...`, written by the driver), every perturb unit's
gate.env (GATE_<id>=...), every regcheck unit's regcheck.env (REGCHECK_<id>=...) and the
driver's instr-variants.json. A unit is NOT SCORED with no stamps.json, a chosen variant other
than the registered choice (the first listed variant whose regcheck and gate lines both read
PASS), a manifest spec other than its variant's, or a failed SASS precondition. Writes
<out>/stamps2.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C5 = _load("rental5_r5common", HERE / "r5common.py")
C4 = C5.C4
PART = "stamps2"


def key_of(vid: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in vid)


def env_lines(tree: Path, pattern: str) -> dict:
    out = {}
    for env in sorted(Path(tree).rglob(pattern)):
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k:
                out[k] = v.split()[0] if v.split() else ""
    return out


def listed_variants(tree: Path, model: str, label: str) -> list[str]:
    """The plan's variant ids of one stamps unit, in the driver's listed order."""
    for p in sorted(Path(tree).rglob("instr-variants.json")):
        try:
            vs = json.loads(p.read_text())
        except ValueError:
            continue
        ids = [str(v.get("id")) for v in vs if v.get("kind") == "stamps" and v.get("model") == model
               and "-".join(str(v.get("id", "")).split("-")[1:-1]) == label]
        if ids:
            return ids
    return []


def read_chosen(unit_dir: Path) -> dict:
    f = Path(unit_dir) / "CHOSEN_VARIANT.txt"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def expected_choice(ids: list[str], gates: dict, regs: dict) -> str | None:
    for vid in ids:
        if gates.get(f"GATE_{key_of(vid)}") == "PASS" and regs.get(f"REGCHECK_{key_of(vid)}") == "PASS":
            return vid
    return None


def load_unit(tree: Path, model: str, label: str, block: str, gates: dict, regs: dict):
    """(manifest, launches, spec, use) for one unit, or (None, None, None, why)."""
    from moe import instrumented as I
    hits = sorted(Path(tree).rglob(f"*-{C5.CARD}-instr-{model}-{label}/stamps.json"))
    ids = listed_variants(tree, model, label)
    if not hits:
        want = expected_choice(ids, gates, regs)
        if ids and want is None:
            lines = {v: (gates.get(f"GATE_{key_of(v)}") or "-", regs.get(f"REGCHECK_{key_of(v)}") or "-") for v in ids}
            return None, None, None, f"NOT SCORED: NOT RUN (no listed variant passes regcheck and the gate: {lines})"
        return None, None, None, "NOT SCORED: no stamps.json"
    if len(hits) != 1:
        return None, None, None, f"NOT SCORED: {len(hits)} stamps.json for {model} {label}"
    udir = hits[0].parent
    chosen = read_chosen(udir)
    man = json.loads(hits[0].read_text())
    vid = chosen.get("id") or man.get("variant")
    if not vid:
        return None, None, None, "NOT SCORED: no chosen variant recorded"
    if man.get("variant") != vid:
        return None, None, None, f"NOT SCORED: CHOSEN_VARIANT {vid} but the manifest ran {man.get('variant')}"
    if gates.get(f"GATE_{key_of(vid)}") != "PASS" or regs.get(f"REGCHECK_{key_of(vid)}") != "PASS":
        return None, None, None, (f"NOT SCORED: {vid}'s gate reads {gates.get('GATE_' + key_of(vid)) or 'nothing'}, "
                                  f"its regcheck {regs.get('REGCHECK_' + key_of(vid)) or 'nothing'}")
    if ids:
        want = expected_choice(ids, gates, regs)
        if want != vid:
            return None, None, None, f"NOT SCORED: variant mismatch, the registered rule chooses {want}, the unit ran {vid}"
    spec_text = man.get("spec_text") or (man.get("spec") or {}).get("text")
    if chosen.get("spec") and spec_text and I.parse_spec(chosen["spec"]) != I.parse_spec(spec_text):
        return None, None, None, f"NOT SCORED: the manifest's spec {spec_text} is not the chosen {chosen['spec']}"
    spec = I.parse_spec(spec_text or chosen.get("spec") or "")
    if spec.stamps != I.SAMPLE:
        return None, None, None, f"NOT SCORED: spec {spec.text()} is not the sample level"
    ok, why = C4.sass_precondition(man, block)
    if not ok:
        return None, None, None, f"NOT SCORED: the SASS precondition fails ({why})"
    import numpy as np
    launches = []
    for rec in man.get("launches") or []:
        if "file" not in rec:
            continue
        cols = I.decode(np.load(udir / "stamps" / rec["file"]), int(rec.get("marks") or 0), rec.get("mark_k"))
        launches.append({**rec, "cols": cols})
    return man, launches, spec, f"counted ({vid}: {spec.text()})"


def by(launches, **kw):
    return [x for x in launches if all(x.get(k) == v for k, v in kw.items())]


def med(xs):
    xs = [float(x) for x in xs if x is not None]
    return st.median(xs) if xs else None


# --------------------------------------------------------------------------
# readouts (pure, on decoded columns)
# --------------------------------------------------------------------------

def last_two(cols):
    """Per live CTA (kind 1) with both marks: (top(S - 1), T_iter from the previous mark)."""
    import numpy as np
    top, ks = cols["iter_top"], cols["mark_k"]
    if top.shape[1] < 2:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    live = np.flatnonzero((cols["kind"] == 1) & (top[:, -1] >= 0) & (top[:, -2] >= 0))
    gap = int(ks[-1] - ks[-2])
    t_iter = (top[live, -1] - top[live, -2]) / gap
    return live, top[live, -1].astype(float), t_iter


def epilogue_cycles(cols) -> float | None:
    live, last, t_iter = last_two(cols)
    if not len(live):
        return None
    return float(st.median(cols["end_clk"][live] - last - t_iter))


def t_fix_cycles(cols, S: int, wave: int) -> float | None:
    """median over steady-wave live CTAs (wave <= pid < N - wave, N the live count read as
    the largest live pid + 1) of lifetime - S T_iter."""
    import numpy as np
    live, _last, t_iter = last_two(cols)
    if not len(live):
        return None
    N = int(np.flatnonzero(cols["kind"] == 1).max()) + 1
    keep = (live >= wave) & (live < N - wave)
    if not keep.any():
        return None
    life = cols["end_clk"][live] - cols["start_clk"][live]
    return float(st.median((life - S * t_iter)[keep]))


def windows(cols, every: int = 16):
    """(R cycles per k-step, j, the CTA's row) for every 16-step window of every live CTA; j the
    time-averaged count of live CTAs on the same smid over the window, itself included."""
    import numpy as np
    top, ks = cols["iter_top"], cols["mark_k"]
    live = cols["kind"] == 1
    out = []
    sm, s0, s1 = cols["smid"], cols["start_clk"], cols["end_clk"]
    for a in range(top.shape[1] - 1):
        if int(ks[a + 1] - ks[a]) != every:
            continue
        rows = np.flatnonzero(live & (top[:, a] >= 0) & (top[:, a + 1] >= 0))
        for i in rows:
            t0, t1 = int(top[i, a]), int(top[i, a + 1])
            if t1 <= t0:
                continue
            same = live & (sm == sm[i]) & (s0 >= 0) & (s1 >= 0)
            ov = np.clip(np.minimum(s1[same], t1) - np.maximum(s0[same], t0), 0, None).sum()
            out.append(((t1 - t0) / every, ov / (t1 - t0), int(i)))
    return out


def rj_verdict(reg: dict, R: dict, occ_bins=None) -> dict:
    """The registered class rule on R(j) {j: cycles}; the occupancy bins are the registration's."""
    occ_bins = tuple(reg["K2"]["occ_bins"]) if occ_bins is None else occ_bins
    riv = reg["K2"]["rivals_cycles"]
    band = float(reg["K2"]["band"])
    if 1 not in R:
        return {"verdict": "NOT SCORED: no R(1) bin"}
    n1 = C4.nearest(R[1], riv["1"])
    best = n1.get("nearest", n1["verdict"])
    classes = reg["K2"]["classes"]
    out = {"j1": n1, "classes": {}}
    for cname, laws in classes.items():
        nearest_at_1 = best in laws and not str(n1["verdict"]).startswith("BETWEEN")
        hold2 = 2 in R and any(abs(R[2] / riv["2"][law] - 1) <= band for law in laws if riv["2"].get(law))
        hold_occ = all(any(abs(R[j] / riv[str(j)][law] - 1) <= band for law in laws if riv[str(j)].get(law))
                       for j in occ_bins if j in R)
        out["classes"][cname] = {"nearest_j1": nearest_at_1, "within_3pct_j2": hold2, "within_3pct_occ": hold_occ}
    sel = [c for c, v in out["classes"].items() if all(v.values())]
    out["verdict"] = f"SELECTED {sel[0]}" if len(sel) == 1 else "UNDECIDED"
    return out


def dead_reads(cols, dead_mod: int) -> dict:
    """Life, dispatch (ns per dead CTA, the sampled interval / dead_mod) and the hidden count."""
    import numpy as np
    live, dead = cols["kind"] == 1, cols["kind"] == 2
    out = {"life": med((cols["end_clk"] - cols["start_clk"])[dead].tolist()) if dead.any() else None}
    if not live.any():
        out.update(dispatch=None, why="no live CTA", hidden=None)
        return out
    last = cols["end_ns"][live].max()
    after = np.sort(cols["start_ns"][dead & (cols["start_ns"] > last)])
    out["hidden"] = int((dead & (cols["start_ns"] <= last)).sum())
    res = C4.timer_resolution_ns({"cols": cols})
    out.update(qualifying=int(len(after)), resolution_ns=res, dispatch=None)
    if len(after) < C5.C4.DISPATCH_MIN_CTAS:
        out["why"] = f"{len(after)} sampled dead CTAs start after the last live end, under {C4.DISPATCH_MIN_CTAS}"
        return out
    span = float(after[-1] - after[0])
    if res is None or span < C4.DISPATCH_MIN_TICKS * res:
        out["why"] = f"span {span:.0f} ns under {C4.DISPATCH_MIN_TICKS} ticks of {res} ns"
        return out
    out["dispatch"] = span / (len(after) - 1) / dead_mod
    return out


def separated(value, sigma, rivals: dict, k: float = 2.0) -> dict:
    """nearest, but SELECTED only when the second-nearest rival is >= k sigma from value."""
    ds = sorted((abs(value - v), name) for name, v in rivals.items() if v is not None)
    first, second = ds[0][1], (ds[1][1] if len(ds) > 1 else None)
    if second is None or (sigma and abs(value - rivals[second]) >= k * sigma):
        return {"verdict": first, "nearest": first, "second": second, "sigma": sigma}
    return {"verdict": f"NOT SEPARATED ({first} nearest, {second} within {k:g} sigma)", "nearest": first,
            "second": second, "sigma": sigma}


# --------------------------------------------------------------------------

def score(repo: Path, tree: Path) -> dict:
    reg = C5.registration(repo, PART)
    gates = env_lines(tree, f"*-{C5.CARD}-perturb-*/gate.env")
    regs = env_lines(tree, "regcheck.env")
    res = {"registration": C5.NAMES[PART], "units": {}}
    U = reg["units"]
    geo = reg["instrument"]["mark_k"]
    loaded = {}
    for label, u in U.items():
        block = u["blocks"][0]
        need = {"F2": "D", "K2": "K", "D2": "D"}[block]
        loaded[label] = load_unit(tree, u["model"], label, need, gates, regs)
        res["units"][label] = {"use": loaded[label][3]}

    # F2
    f2 = res["F2"] = {}
    man, L, spec, _use = loaded["stf"]
    if man is None:
        f2["verdict"] = "NOT SCORED"
    else:
        model = U["stf"]["model"]
        per_n, tfix = {}, {"w1": [], "w2": []}
        for n in sorted({x["n"] for x in L}):
            ep = {g: med([epilogue_cycles(x["cols"]) for x in by(L, n=n, gemm=g)]) for g in ("w1", "w2")}
            if ep["w1"] is not None and ep["w2"] is not None:
                per_n[n] = ep["w2"] - ep["w1"]
        occ = {"w1": 5, "w2": 4}
        for g in ("w1", "w2"):
            tfix[g] = med([t_fix_cycles(x["cols"], geo[model][g]["S"], C5.SMS * occ[g]) for x in by(L, gemm=g)])
        if per_n:
            d = st.median(per_n.values())
            f2["delta_epi"] = {"cycles": d, "per_n": per_n, **C4.nearest(d, reg["F2"]["delta_epi"]["rivals_cycles"])}
        else:
            f2["delta_epi"] = {"verdict": "NOT SCORED: no live CTA with its last two marks"}
        f2["t_fix"] = {g: ({"cycles": tfix[g], **C4.nearest(tfix[g], reg["F2"]["t_fix"]["rivals_cycles"][g], log=True)}
                           if tfix[g] and tfix[g] > 0 else {"verdict": "NOT SCORED: no steady-wave CTA"})
                       for g in ("w1", "w2")}

    # K2
    k2 = res["K2"] = {"bins": {}}
    allw, rocc_only = [], []
    occ_bins = tuple(reg["K2"]["occ_bins"])
    for label in reg["K2"]["units"]:
        man, L, spec, _use = loaded[label]
        if man is None:
            continue
        printed_keys = set(reg["K2"]["tail_printed_only"])
        for li, x in enumerate(L):
            tail_printed = spec.iter_mod != 1 and f"{label} n{x.get('n')} {x.get('gemm')}" in printed_keys
            w = windows(x["cols"])
            if spec.cta_mod == 1:
                allw += [(r, j, (label, li, i), tail_printed) for r, j, i in w]
            else:
                rocc_only += [r for r, _j, _i in w]
    bins, ctas, printed = {}, {}, {}
    for r, j, cta, tp in allw:
        jj = int(round(j))
        if tp and jj < min(occ_bins):
            printed.setdefault(jj, []).append(r)   # registered tail_rule: printed only
            continue
        bins.setdefault(jj, []).append(r)
        ctas.setdefault(jj, set()).add(cta)
    R = {}
    for j, v in sorted(bins.items()):
        n = len(ctas[j])
        ok = n >= reg_min(reg)
        k2["bins"][str(j)] = {"ctas": n, "windows": len(v), "R": st.median(v),
                              "use": "counted" if ok else f"NOT SCORED: {n} CTAs, under {reg_min(reg)}"}
        if ok:
            R[j] = st.median(v)
    for j, v in sorted(printed.items()):
        k2["bins"].setdefault(str(j), {})["printed_only"] = {"windows": len(v), "R": st.median(v),
                                                             "why": "registered tail_rule (v2, sampled tail under 30)"}
    if R:
        k2["rule"] = rj_verdict(reg, R)
        oz = int(reg["K2"]["occ_z"])
        if 1 in R and oz in R:
            k2["Z_cycles"] = R[1] - R[oz] / oz
        k2["verdict"] = k2["rule"]["verdict"]
    else:
        k2["verdict"] = "NOT SCORED: no R(j) bin (no V1 / V2 unit, or every bin under the minimum)"
    if rocc_only:
        k2["R_occ_V3_printed"] = st.median(rocc_only)

    # D2
    d2 = res["D2"] = {}
    occ_by = reg["D2"]["occupancy"]
    hid = reg["D2"]["hiding"]
    c_ns = float(hid["c_ns"])
    for label in reg["D2"]["units"]:
        man, L, spec, _use = loaded[label]
        ent = d2[label] = {}
        if man is None:
            ent["verdict"] = "NOT SCORED"
            continue
        for g in ("w1", "w2"):
            occ = occ_by[label][g]
            reads = [dead_reads(x["cols"], spec.dead_mod) for x in by(L, gemm=g)]
            life = med([r["life"] for r in reads])
            sh = [dead_reads(x["cols"], spec.dead_mod) for x in by(L, gemm=g, arm="shared")]
            disp = med([r["dispatch"] for r in sh])
            e = {"occ": occ, "life_cycles": life,
                 "life": C4.nearest(life, reg["D2"]["life"]["rivals_cycles"]) if life else {"verdict": "NOT SCORED"},
                 "dispatch_ns": disp, "dispatch_reads": sh}
            riv = reg["D2"]["dispatch"]["rivals_ns"].get(str(occ))
            if spec.cta_mod != 1:   # registered cta_mod_rule (build-r5-review F2)
                why = f"NOT SCORED: the variant samples live CTAs (cta_mod {spec.cta_mod}): the last live end is biased"
                e["dispatch"] = {"verdict": why}
                e["kappa"] = {"verdict": why}
                ent[g] = e
                continue
            e["dispatch"] = C4.nearest(disp, riv) if disp and riv else {"verdict": "NOT SCORED"}
            nh = med([r["hidden"] for r in sh if r.get("hidden")])
            if disp and nh:
                S1 = hid["S_prime"][g]
                kappa = spec.dead_mod * nh * disp / (occ * S1 * c_ns)
                sigma = kappa * math.sqrt(1 / nh + reg_drel(reg) ** 2)
                e["kappa"] = {"value": kappa, "sigma": sigma, "hidden_sampled": nh,
                              **separated(kappa, sigma, hid["rivals"], float(hid["separation_sigma"])), "D_vs_D2": "NOT SEPARATED (registered)"}
            else:
                e["kappa"] = {"verdict": "NOT SCORED: no dispatch reading or no hidden dead CTA"}
            ent[g] = e
    res["secondary"] = reg["secondary"]
    return res


def reg_min(reg: dict) -> int:
    return int(reg["K2"]["bin_minimum_ctas"])


def reg_drel(reg: dict) -> float:
    return float(reg["D2"]["hiding"]["d_rel_err"])


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 5 STAMPS v2, {res['registration']}"]
    for k, v in res["units"].items():
        out.append(f"  unit {k}: {v['use']}")
    f2 = res.get("F2", {})
    out.append(f"  F2 Delta_epi: {(f2.get('delta_epi') or {}).get('verdict', f2.get('verdict'))}")
    for g, v in (f2.get("t_fix") or {}).items():
        out.append(f"  F2 T_fix {g}: {v.get('verdict')}")
    out.append(f"  K2: {res['K2'].get('verdict')}")
    for label, ent in res.get("D2", {}).items():
        for g in ("w1", "w2"):
            e = ent.get(g)
            if e:
                out.append(f"  D2 {label} {g}: life {e['life'].get('verdict')}, dispatch {e['dispatch'].get('verdict')}, "
                           f"kappa {e['kappa'].get('verdict')}")
        if "verdict" in ent:
            out.append(f"  D2 {label}: {ent['verdict']}")
    out.append(f"  secondary: {res['secondary']}")
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C5.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
