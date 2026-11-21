#!/usr/bin/env python
"""Table T2 of the paper: every fitted parameter of the timing model
(scripts/r3_timing_model.py) and the byte model (scripts/wave_split_bytes.py),
its value, the pages it was fitted on, and which registrations carry it.

    python scripts/paper/t2_parameters.py                    # print the table
    python scripts/paper/t2_parameters.py --write            # write docs/paper/T2_parameters.{md,csv}
    python scripts/paper/t2_parameters.py --check            # exit 1 if the committed files differ

WHAT COUNTS AS A FITTED PARAMETER. A number the model uses that was chosen to
match measured pages: by a least-squares or simplex fit, by a scan of an rms, or
by a regression on counter pages ("MEASURED" in the code's own comments, which
is still a fit to data). Not counted: arithmetic from the kernel and the shapes
(grid rows, B_other, q_g, slab and A-tile bytes), quantities read off each page
(SM count, L2 size, occupancy, grid size), rules with no constant (TAIL,
LATER_MISS, CORES's g(k) = k), falsifier bands and design thresholds of the
fixed-point guard (KAPPA_MAX, ROOT_GAP_MAX and the like).

Every value is read from the code (module constants, the fit's parameter
names) or from the registration JSONs in docs/registered (each one's
`timing_params` and `byte_params[byte_view]`). The floor law's c and F are
read from the comment in r3_timing_model.py that records their fit, because no
committed script fits them; PHI = F / c is checked against CTA_FIXED_KSTEPS.
Which model version each registration carries is stated in its
docs/registered/README.md section (cited per row of REGISTRATIONS below) and
checked against the JSON where the JSON records it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import r3_timing_model as TM  # noqa: E402
import wave_split_bytes as W  # noqa: E402

D = ROOT / "docs/registered"
OUT_MD = ROOT / "docs/paper/T2_parameters.md"
OUT_CSV = ROOT / "docs/paper/T2_parameters.csv"

#: The registrations that carry a fitted timing or byte model, in commit order,
#: with the model features in force (docs/registered/README.md, the section
#: named in `src`). `dead`: DEAD_CTA_NS; `cta`: the per-CTA fixed cost (c_floor,
#: F_w1, F_w2); `cores`: CORES_RHO_GBPS; `stage3`: C_A2, beta_A2; `bytes`:
#: whether the registered predictions use the byte model at all; `per_card`: the
#: per-card timing parameters that enter its predictions (default: all five).
REGISTRATIONS = [
    {"stem": "2026-09-27-mixtral-8x22b-gh200", "label": "8x22B",
     "dead": False, "cta": False, "cores": False, "stage3": False, "bytes": True,
     "src": "D/README.md 2026-09-27 (knee p = 14; TAIL and stage 3 came after these pages, :48-50)"},
    {"stem": "2026-09-28-qwen2-57b-a14b-gh200", "label": "Qwen2-57B",
     "dead": False, "cta": False, "cores": False, "stage3": True, "bytes": True,
     "src": "D/README.md 2026-09-28 (p = 14, the partial-last-wave rule; LATER_MISS and stage 3 in force, :57-60)"},
    {"stem": "2026-09-29-olmoe-1b-7b-gh200", "label": "OLMoE",
     "dead": True, "cta": False, "cores": False, "stage3": True, "bytes": True,
     "src": "D/README.md 2026-09-29 OLMoE (dead-CTA term ea2c77c, capacity rule 15a9533, :111-115)"},
    {"stem": "2026-09-29-olmoe-1b-7b-h100", "label": "OLMoE H100 (not run)",
     "dead": True, "cta": False, "cores": False, "stage3": True, "bytes": True,
     "src": "D/README.md 2026-09-29 H100 (the GH200 registration made from the H100 fit, :147-151)"},
    {"stem": "2026-09-29-qwen1.5-moe-a2.7b-gh200", "label": "Qwen1.5",
     "dead": True, "cta": True, "cores": False, "stage3": True, "bytes": True,
     "src": "D/README.md 2026-09-29 four held-out models (0f77622, c 344.1, F 520 / 979, :176-181)"},
    {"stem": "2026-09-29-phi-3.5-moe-gh200", "label": "Phi-3.5",
     "dead": True, "cta": True, "cores": False, "stage3": True, "bytes": True,
     "src": "same section"},
    {"stem": "2026-09-29-jetmoe-8b-gh200", "label": "JetMoE",
     "dead": True, "cta": True, "cores": False, "stage3": True, "bytes": True,
     "src": "same section"},
    {"stem": "2026-09-29-granite-3.0-3b-a800m-gh200", "label": "Granite-3B",
     "dead": True, "cta": True, "cores": False, "stage3": True, "bytes": True,
     "src": "same section"},
    {"stem": "2026-09-29-granite-3.0-3b-a800m-h100", "label": "Granite-3B H100 (not run)",
     "dead": True, "cta": True, "cores": False, "stage3": True, "bytes": True,
     "src": "D/README.md 2026-09-29 Granite H100 (refit with the dead-CTA and per-CTA terms, :262-267)"},
    {"stem": "2026-09-30-mixtral-8x7b-tp8-floor-gh200", "label": "tp8 floor (CORES)",
     "dead": True, "cta": True, "cores": True, "stage3": False, "bytes": False,
     "per_card": ("c", "bw"),
     "src": "D/README.md 2026-09-30 tp8 (CORES, rho 11.48; bytes sigma = 1, the launch order's own "
            "reads, :338-339, :383-384); per-GEMM cycles, so T0, s_small, s_block do not enter "
            "(scripts/cores_heldout_predict.py predict: gemm_ms + dead_ms at c, bw)"},
]

#: Pages each fit was taken on, as each registration's README section names them.
GH200_0927 = ("results/published/2026-09-27-nvidia_gh200_480gb-session: five lock-1710 timed pages "
              "(G = 2, 3, 4, 8, 32)")
GH200_0927_C = ("results/published/2026-09-27-nvidia_gh200_480gb-session: eight lock-1710 counter "
                "pages (r3c-g{1,2,3,4,8,16,32,64}.json)")
H100_0925 = ("results/published/2026-09-25-nvidia_h100_80gb_hbm3-session: five lock-1710 timed "
             "pages (G = 1, 2, 4, 16, 64) and its base-clock counter pages")
CAL4 = ("the four calibration models' lock-1710 counter pages: 8x7B 2026-09-27, 8x22B and "
        "Qwen2-57B 2026-09-28, OLMoE 2026-09-29")


def _read_reg(stem: str) -> dict:
    return json.loads((D / f"{stem}.json").read_text())


def floor_law_fit() -> dict:
    """c, F_w1, F_w2 of the per-CTA floor, from the comment that records their fit."""
    src = (ROOT / "scripts/r3_timing_model.py").read_text()
    m = re.search(r"c (\d+\.\d+)\s*#?:?\s*cycles, F_w1 (\d+) \(se (\d+)\), F_w2 (\d+) \(se (\d+)\)",
                  src)
    if not m:
        raise SystemExit("r3_timing_model.py no longer records the c / F fit")
    c, f1, se1, f2, se2 = (float(x) for x in m.groups())
    for g, f in (("w1", f1), ("w2", f2)):
        if abs(f / c - TM.CTA_FIXED_KSTEPS[g]) > 0.002:
            raise SystemExit(f"PHI_{g} = {TM.CTA_FIXED_KSTEPS[g]} is not F / c = {f / c:.4f}")
    return {"c": c, "F_w1": f1, "se_w1": se1, "F_w2": f2, "se_w2": se2}


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:.2f}"
    if a >= 1:
        return f"{v:.4g}"
    if a >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _distinct(regs, key, sub=None) -> str:
    """The distinct values a parameter takes across registrations, with labels."""
    vals: dict[str, list[str]] = {}
    for r, d in regs:
        if sub == "timing":
            v = (d.get("timing_params") or (d.get("predictions") or {}).get("source_fit") or {}).get(key)
        else:
            if not r["bytes"]:
                continue
            v = d["byte_params"][d["byte_view"]].get(key)
            if key in W.STAGE3_NAMES and not v:
                continue
        if v is None:
            continue
        vals.setdefault(_fmt(v), []).append(r["label"])
    return "; ".join(f"{v} ({', '.join(ls)})" for v, ls in vals.items())


def rows() -> list[dict]:
    regs = [(r, _read_reg(r["stem"])) for r in REGISTRATIONS]
    fl = floor_law_fit()
    tp8 = regs[-1][1]
    if tp8["predictions"]["cores_rho_gbps"] != TM.CORES_RHO_GBPS:
        raise SystemExit("tp8 registration's rho differs from CORES_RHO_GBPS")
    for r, d in regs:
        if d.get("p_knee", TM.P_KNEE) != TM.P_KNEE or d.get("k_w", TM.K_W) != TM.K_W:
            raise SystemExit(f"{r['stem']}: p or k_w differs from the code")
    carriers = {
        "all": ", ".join(r["label"] for r, _ in regs),
        "dead": ", ".join(r["label"] for r, _ in regs if r["dead"]),
        "cta": ", ".join(r["label"] for r, _ in regs if r["cta"]),
        "cores": ", ".join(r["label"] for r, _ in regs if r["cores"]),
        "bytes": ", ".join(r["label"] for r, _ in regs if r["bytes"]),
        "stage3": ", ".join(r["label"] for r, _ in regs if r["bytes"] and r["stage3"]),
    }
    out = []
    units = TM.UNITS
    for name in TM.NAMES:
        out.append({
            "model": "timing", "scope": "per card", "parameter": name, "unit": units[name],
            "value": _distinct([(r, d) for r, d in regs if name in r.get("per_card", TM.NAMES)],
                               name, "timing"),
            "fitted_on": f"GH200: {GH200_0927}, refitted at each model version; H100: {H100_0925}",
            "how": "bounded least squares on call time (r3_timing_model.fit, NAMES)",
            "carried_by": ", ".join(r["label"] for r, _ in regs
                                    if name in r.get("per_card", TM.NAMES))})
    out += [
        {"model": "timing", "scope": "study", "parameter": "k_w", "unit": "", "value": _fmt(TM.K_W),
         "fitted_on": "the 2026-09-25/26 judge's scan on three cards (rms optimum 0.40 to 0.50; "
                      "r3_timing_model.py docstring, WHAT IT FITS)",
         "how": "rms scan, held fixed", "carried_by": carriers["all"]},
        {"model": "timing", "scope": "study", "parameter": "P_KNEE (p)", "unit": "",
         "value": _fmt(TM.P_KNEE),
         "fitted_on": "8x7B timed pages of three Hopper boards (free fits 13.9, 13.9, 12.6; "
                      "8900699); 8x22B's own pages want about 23",
         "how": "free fit, rounded and held", "carried_by": carriers["all"]},
        {"model": "timing", "scope": "study", "parameter": "DEAD_CTA_NS", "unit": "ns",
         "value": _fmt(TM.DEAD_CTA_NS),
         "fitted_on": "2026-09-27 GH200 8x7B counter pages, SHARED minus NATIVE in-kernel "
                      "cycles, 72 cells (ea2c77c)",
         "how": "regression on counters", "carried_by": carriers["dead"]},
        {"model": "timing", "scope": "study", "parameter": "c_floor", "unit": "cycles per CTA k-step",
         "value": _fmt(fl["c"]), "fitted_on": f"{CAL4}; SHARED and NATIVE, G >= 8, n = 2 to 9, "
                                                "512 GEMM cells (0f77622)",
         "how": "joint regression with F_w1, F_w2 on sm__cycles_elapsed.avg; the floor "
                "predictions use it directly, the timing model through PHI = F / c",
         "carried_by": carriers["cta"]},
        {"model": "timing", "scope": "study", "parameter": "F_w1", "unit": "cycles per CTA",
         "value": f"{fl['F_w1']:.0f} (se {fl['se_w1']:.0f}); PHI_w1 = {TM.CTA_FIXED_KSTEPS['w1']}",
         "fitted_on": "same fit", "how": "same fit", "carried_by": carriers["cta"]},
        {"model": "timing", "scope": "study", "parameter": "F_w2", "unit": "cycles per CTA",
         "value": f"{fl['F_w2']:.0f} (se {fl['se_w2']:.0f}); PHI_w2 = {TM.CTA_FIXED_KSTEPS['w2']}",
         "fitted_on": "same fit", "how": "same fit", "carried_by": carriers["cta"]},
        {"model": "timing", "scope": "study", "parameter": "CORES_RHO_GBPS (rho)",
         "unit": "GB/s per resident CTA", "value": _fmt(TM.CORES_RHO_GBPS),
         "fitted_on": "72 calibration GEMM cells in the rule's domain (8x7B w2 n = 1, 2; 8x22B w2 "
                      "n = 1); the pooled form chosen on JetMoE-8B's published n = 3 page (SEEN)",
         "how": "regression on counters", "carried_by": carriers["cores"]},
    ]
    stage_of = {**{k: (1, "PRIVATE, every G and tread, both GEMMs") for k in W.STAGE1_NAMES},
                **{k: (2, "SHARED and NATIVE at G in {1, 2, 4}, stage 1 held") for k in W.STAGE2_NAMES},
                **{k: (3, "SHARED and NATIVE w2 at G >= 32, n >= 2, stages 1 and 2 held; form "
                          "chosen after the 8x22B pages (fb440c5)") for k in W.STAGE3_NAMES}}
    for name in (*W.STAGE1_NAMES, *W.STAGE2_NAMES, *W.STAGE3_NAMES):
        st, cells = stage_of[name]
        out.append({
            "model": "bytes", "scope": "per card", "parameter": name,
            "unit": "MiB" if name.startswith("C_") else "",
            "value": _distinct(regs, name, "bytes"),
            "fitted_on": f"{GH200_0927_C}; stage {st}: {cells}",
            "how": f"simplex on (q_pred / q_meas - 1)^2, stage {st} (wave_split_bytes.fit_view)",
            "carried_by": carriers["stage3"] if st == 3 else carriers["bytes"]})
    out.append({
        "model": "bytes", "scope": "optional, off", "parameter": "k0_A, a_A", "unit": "",
        "value": "0 (off) in every registered byte prediction",
        "fitted_on": "candidate R2 of 2026-10-01-atile-ksteps only (scripts/atile_pooled_fit.py)",
        "how": "not part of the registered model", "carried_by": "none (R2 candidate, FALSIFIED by T4)"})
    return out


def counts(rs: list[dict]) -> dict:
    t_card = sum(r["model"] == "timing" and r["scope"] == "per card" for r in rs)
    t_study = sum(r["model"] == "timing" and r["scope"] == "study" for r in rs)
    b_card = sum(r["model"] == "bytes" and r["scope"] == "per card" for r in rs)
    b_pre3 = b_card - len(W.STAGE3_NAMES)
    per_reg = {}
    for r in REGISTRATIONS:
        n = len(r.get("per_card", TM.NAMES)) + 2  # plus k_w and p, in every registration
        n += r["dead"] + 3 * r["cta"] + r["cores"]
        if r["bytes"]:
            n += b_card if r["stage3"] else b_pre3
        per_reg[r["label"]] = n
    return {"timing_per_card": t_card, "timing_study": t_study, "bytes_per_card": b_card,
            "bytes_per_card_before_stage3": b_pre3, "total": t_card + t_study + b_card,
            "per_registration": per_reg}


HEAD = ["model", "scope", "parameter", "unit", "value", "fitted_on", "how", "carried_by"]


def render_md(rs: list[dict], c: dict) -> str:
    lines = [
        "# Table T2: fitted parameters of the timing and byte models",
        "",
        "Generated by `python scripts/paper/t2_parameters.py --write` from the code",
        "(`scripts/r3_timing_model.py`, `scripts/wave_split_bytes.py`) and the registration JSONs",
        "in `docs/registered/`. Do not edit by hand; `--check` fails when this file is stale.",
        "",
        f"Count: {c['total']} fitted parameters. Timing model: {c['timing_per_card']} per card "
        f"({', '.join(TM.NAMES)}) and {c['timing_study']} study-level (k_w, p, DEAD_CTA_NS, "
        f"c_floor, F_w1, F_w2, rho). Byte model: {c['bytes_per_card']} per card in the registered "
        f"MIX view ({c['bytes_per_card_before_stage3']} before stage 3). Two further byte "
        "parameters (k0_A, a_A) exist in the code and are off in every registered prediction.",
        "",
        "Fitted parameters each registration's predictions carry:",
        "",
        "| registration | fitted parameters |",
        "|---|---:|",
    ]
    for k, v in c["per_registration"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    for r in rs:
        lines.append("| " + " | ".join(str(r[h]).replace("|", "/") for h in HEAD) + " |")
    lines += ["", "Not counted (arithmetic, read off the page, or a rule with no constant): grid "
              "rows, B_other, q_g, slab and A-tile bytes, SM count, L2 size, occupancy, TAIL, "
              "LATER_MISS, CORES's g(k) = k, the fixed-point guard's thresholds, every falsifier "
              "band.", ""]
    return "\n".join(lines)


def render_csv(rs: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HEAD, lineterminator="\n")
    w.writeheader()
    w.writerows(rs)
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    rs = rows()
    c = counts(rs)
    md, cs = render_md(rs, c), render_csv(rs)
    if a.check:
        bad = [p for p, t in ((OUT_MD, md), (OUT_CSV, cs)) if not p.exists() or p.read_text() != t]
        for p in bad:
            print(f"STALE: {p.relative_to(ROOT)}")
        return 1 if bad else 0
    if a.write:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(md)
        OUT_CSV.write_text(cs)
        print(f"wrote {OUT_MD.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
