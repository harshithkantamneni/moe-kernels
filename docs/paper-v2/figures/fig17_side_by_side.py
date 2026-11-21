"""Tables 3 and 4 of paper-v2 (section 5.1), recomputed per cell, and Figure 1 (fig17_side_by_side.svg).

    python fig17_side_by_side.py <moe-kernels repo root>

Read-only on the repo. Every number comes from a committed per-cell file, read by repo-relative path:

  B. prefill, first (round 5, 84 cells)   scripts/scoring/tileB3/tileB3.score.json       views.cells[].e
  C. prefill, second (v7, 24 cells)       scripts/scoring/v7pblind/prefill_blind.score.json  views.cells[].e
  D. decode, first (v4d', 220 cells)      scripts/scoring/decode/blind.score.json        rows[] (meas, pred_v4, pred{rivals})
  E. decode, second (v7, 125 cells)       scripts/scoring/v7blind/blind.score.json       rows[] (meas, pred_v4, pred{rivals})
  A. test tile (v3, 84 cells)             scripts/scoring/rental7/skew.score.json holds only the pooled E1-SKEW
                                          stats (cells, rms, worst, mean), not the 84 per-cell errors. The per-cell
                                          errors are rebuilt by running the registered scorer
                                          scripts/scoring/rental7/score_skew.py in memory on the published pages
                                          results/published/2026-10-14-nvidia_gh200_480gb-rental7-session (nothing
                                          is written), capturing the list it passes to its E1-SKEW verdict, and
                                          checked against the committed skew.score.json stats before use.

Metrics (paper section 3.4): e = predicted / measured - 1 per cell; MAPE = mean |e|; rms of e; mean e; worst |e|;
cells with |e| <= 5% and <= 10%. Decode cells (D, E) whose |predicted - measured| <= 1 us score e = 0, as the
registered decode scorer does; the floor applies to the registered baselines too. p50 / p95 (Table 5) are numpy's
default linear percentiles of |e|.

Every recomputed value is compared, as printed, with the paper's printed value; any mismatch is listed and the
script exits 1 (after still writing the figure, so the mismatch can be seen).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

HERE = Path(__file__).resolve().parent
FLOOR_MS = 0.001  # 1 us, in the decode rows' unit (ms)

SRC = {
    "A": "scripts/scoring/rental7/skew.score.json",
    "A_scorer": "scripts/scoring/rental7/score_skew.py",
    "A_pages": "results/published/2026-10-14-nvidia_gh200_480gb-rental7-session",
    "B": "scripts/scoring/tileB3/tileB3.score.json",
    "C": "scripts/scoring/v7pblind/prefill_blind.score.json",
    "D": "scripts/scoring/decode/blind.score.json",
    "E": "scripts/scoring/v7blind/blind.score.json",
}

# ---- the paper's printed values (paper-v2.md, section 5.1) --------------------------------------------------------
# Table 3: cells, MAPE, rms, mean (bias), worst cell, within 5%, within 10%
PAPER_T3 = {
    "B": ("B. prefill, first (round 5)", "84", "1.71%", "2.41%", "+0.29%", "8.22%", "78 / 84", "84 / 84"),
    "C": ("C. prefill, second (v7)", "24", "2.99%", "3.13%", "-2.99%", "4.72%", "24 / 24", "24 / 24"),
    "D": ("D. decode, first (v4d')", "220", "6.70%", "11.40%", "+5.99%", "39.24%", "144 / 220", "173 / 220"),
    "E": ("E. decode, second (v7)", "125", "2.90%", "4.17%", "+1.61%", "17.20%", "104 / 125", "118 / 125"),
    "A": ("A. test tile (v3, trunk only)", "84", "0.45%", "0.61%", "-0.24%", "1.61%", "84 / 84", "84 / 84"),
}
T3_GROUPS = [("vLLM's prefill tiles", "BC"), ("vLLM's decode tiles", "DE"),
             ("controlled check (not a serving tile)", "A")]
# Table 4: MAPE on D and E. key in the score rows' "pred" dict (None = the model itself, pred_v4)
PAPER_T4 = [
    ("our model (v4d' in D, v7 in E)", None, "6.70%", "2.90%"),
    ("roofline: T0 + weight bytes / bandwidth", "roofline", "8.22%", "24.67%"),
    ("ridge regression on weight bytes, waves, dead CTAs and path", "OLS", "9.77%", "11.80%"),
    ("linear in tokens x k", "line", "293.8%", "301.2%"),
]
# Table 5, this work's rows: MAPE; p50 / p95
PAPER_T5 = {"B": ("1.71%", "1.14%", "5.70%"), "C": ("2.99%", "2.96%", "4.43%"),
            "D": ("6.70%", "2.39%", "25.73%"), "E": ("2.90%", "1.95%", "10.19%")}

# ---- published predictors, as reported (constants; paper Table 5 and its references) -----------------------------
# (label, scope, statistic, value in %, p95 in % or None, reference key in paper-v2.md's References)
PUBLISHED = [
    ("Frontier", "MoE GroupedGEMM, H800", "p50 |e|", 1.4, 5.3, "Feng 2026, arXiv:2605.21312v2"),
    ("PipeWeave, seen GPUs", "6 kernels incl. SGLang fused MoE, 11 GPUs", "MAPE", 6.0, None,
     "Zhang 2026, ISCA 2026, arXiv:2601.14910"),
    ("PipeWeave, unseen GPUs", "same", "MAPE", 11.5, None, "Zhang 2026, ISCA 2026, arXiv:2601.14910"),
    ("NeuSight, unseen GPUs", "DL kernels", "MAPE", 8.9, None, "Lee 2025, ASPLOS 2025, arXiv:2407.13853"),
    ("KernelSight-LM, unseen GPU generation", "LLM kernels, dense models", "MAPE", 12.1, None,
     "Yao 2026, arXiv:2606.28565v2"),
    ("TileSight", "703 GEMM shapes, 4 GPUs", "MAPE", 12.35, None, "Mo 2026, arXiv:2607.22432"),
    ("TileSight", "vLLM MoE configurations end to end", "wMAPE", 10.35, None, "Mo 2026, arXiv:2607.22432"),
    ("roofline, as measured by TileSight", "same 703 GEMM shapes", "MAPE", 33.85, None, "Mo 2026, arXiv:2607.22432"),
]


# ---- statistics ---------------------------------------------------------------------------------------------------
def stats(e) -> dict:
    e = np.asarray(e, dtype=float)
    a = np.abs(e)
    return {"cells": int(e.size), "mape": a.mean() * 100, "rms": np.sqrt((e ** 2).mean()) * 100,
            "mean": e.mean() * 100, "worst": a.max() * 100, "w5": int((a <= 0.05).sum()),
            "w10": int((a <= 0.10).sum()), "p50": np.percentile(a, 50) * 100, "p95": np.percentile(a, 95) * 100}


def pct(x: float, nd: int = 2, sign: bool = False) -> str:
    return f"{x:+.{nd}f}%" if sign else f"{x:.{nd}f}%"


def decode_e(rows, key) -> list:
    """Per-cell e for one predictor on decode rows, with the registered 1 us floor."""
    out = []
    for r in rows:
        p = r["pred_v4"] if key is None else r["pred"][key]
        out.append(0.0 if abs(p - r["meas"]) <= FLOOR_MS else p / r["meas"] - 1)
    return out


def load(repo: Path, key: str) -> dict:
    return json.loads((repo / SRC[key]).read_text())


def test_a_cells(repo: Path) -> list:
    """Rebuild test A's 84 per-cell errors with the registered rental 7 scorer, in memory."""
    p = repo / SRC["A_scorer"]
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("r7_score_skew", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []
    verdict = mod._verdict

    def capture(v, rule):
        seen.append(list(v))
        return verdict(v, rule)

    mod._verdict = capture
    res = mod.score(repo, repo / SRC["A_pages"])
    committed = load(repo, "A")["E1-SKEW"]["stats"]
    # the first verdict call of the first (registered, ALL) view is the pooled E1-SKEW list
    e = seen[0]
    if res["E1-SKEW"]["stats"] != committed or mod.C5.stats(e) != committed:
        raise SystemExit(f"test A: rebuilt cells do not reproduce {SRC['A']} E1-SKEW stats {committed}")
    return e


# ---- main ---------------------------------------------------------------------------------------------------------
def main(argv) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    repo = Path(argv[0]).resolve()
    mism = []

    def check(where, paper, got):
        if paper != got:
            mism.append(f"{where}: paper {paper}, recomputed {got}")

    D, E = load(repo, "D"), load(repo, "E")
    for nm, d in (("D", D), ("E", E)):
        stored = [r["e"] for r in d["rows"]]
        rebuilt = decode_e(d["rows"], None)
        worst = max(abs(a - b) for a, b in zip(stored, rebuilt))
        if worst > 1e-6:  # stored e is rounded to 6 places
            mism.append(f"{nm}: rebuilt per-cell e differs from the stored e by up to {worst:.2e}")

    cells = {
        "B": [c["e"] for c in load(repo, "B")["views"]["cells"]],
        "C": [c["e"] for c in load(repo, "C")["views"]["cells"]],
        "D": decode_e(D["rows"], None),
        "E": decode_e(E["rows"], None),
        "A": test_a_cells(repo),
    }
    S = {k: stats(v) for k, v in cells.items()}

    print("Sources (repo-relative, root " + str(repo) + "):")
    for k in "ABCDE":
        extra = f" (per-cell e rebuilt by {SRC['A_scorer']} on {SRC['A_pages']})" if k == "A" else ""
        print(f"  {k}: {SRC[k]}{extra}")
    print()

    # Table 3
    print("**Table 3. Blind accuracy on GH200 (recomputed).**\n")
    print("| test | cells | MAPE | rms | mean (bias) | worst cell | within 5% | within 10% |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for head, keys in T3_GROUPS:
        print(f"| *{head}* | | | | | | | |")
        for k in keys:
            s, p = S[k], PAPER_T3[k]
            n = s["cells"]
            got = (str(n), pct(s["mape"]), pct(s["rms"]), pct(s["mean"], sign=True), pct(s["worst"]),
                   f"{s['w5']} / {n}", f"{s['w10']} / {n}")
            print(f"| {p[0]} | " + " | ".join(got) + " |")
            for col, a, b in zip(("cells", "MAPE", "rms", "mean", "worst", "within 5%", "within 10%"), p[1:], got):
                check(f"Table 3 {k} {col}", a, b)
    print()

    # Table 4
    print("**Table 4. Registered baselines on the same decode cells, MAPE (recomputed).**\n")
    print("| predictor | score-file key | D (220 cells) | E (125 cells) |")
    print("|---|---|---:|---:|")
    t4 = {}
    for label, key, pd, pe in PAPER_T4:
        md, me = stats(decode_e(D["rows"], key))["mape"], stats(decode_e(E["rows"], key))["mape"]
        t4[key] = (md, me)
        nd = len(pd.rstrip("%").split(".")[1])  # the paper's printed decimals
        gd, ge = pct(md, nd), pct(me, nd)
        print(f"| {label} | {key or 'pred_v4'} | {gd} | {ge} |")
        check(f"Table 4 {key or 'model'} D", pd, gd)
        check(f"Table 4 {key or 'model'} E", pe, ge)
    print()

    # Table 5, this work's rows
    print("**Table 5, this work's rows (recomputed): MAPE; p50 / p95 of |e|.**\n")
    print("| test | MAPE | p50 | p95 |")
    print("|---|---:|---:|---:|")
    for k, (pm, p50, p95) in PAPER_T5.items():
        g = (pct(S[k]["mape"]), pct(S[k]["p50"]), pct(S[k]["p95"]))
        print(f"| {k} | " + " | ".join(g) + " |")
        for col, a, b in zip(("MAPE", "p50", "p95"), (pm, p50, p95), g):
            check(f"Table 5 {k} {col}", a, b)
    print()

    draw(S, t4)

    if mism:
        print(f"MISMATCH: {len(mism)} value(s) differ from the paper:")
        for m in mism:
            print("  " + m)
        return 1
    n = 7 * 5 + 2 * 4 + 3 * 4
    print(f"OK: all {n} printed values match the paper (Table 3: 35, Table 4: 8, Table 5 this-work rows: 12).")
    print(f"Wrote {HERE / 'fig17_side_by_side.svg'}")
    return 0


# ---- Figure 1 -----------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, GREY = "#2a78d6", "#b4b2aa"
XMAX = 40.0  # percent; bars past it are cut and their value printed


def _font() -> str:
    names = {f.name for f in fm.fontManager.ttflist}
    for n in ("Helvetica Neue", "Inter", "Arial"):
        if n in names:
            return n
    return "DejaVu Sans"


def draw(S: dict, t4: dict) -> None:
    plt.rcParams.update({"font.family": _font(), "font.size": 9, "svg.fonttype": "path",
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK, "svg.hashsalt": "fig17"})
    # rows: (label, value %, style, p95 or None)
    rows = []
    rows.append(("This study, blind, GH200 (vLLM fused MoE)", None, "head", None))
    for k, lab in (("B", "B. prefill, first (round 5), 84 cells"), ("C", "C. prefill, second (v7), 24 cells"),
                   ("D", "D. decode, first (v4d'), 220 cells"), ("E", "E. decode, second (v7), 125 cells"),
                   ("A", "A. test tile 32x64x64, trunk only (v3), 84 cells")):
        rows.append((lab, S[k]["mape"], "fill", None))
    rows.append(("Registered baselines, same decode cells", None, "head", None))
    for key, lab in (("roofline", "roofline"), ("OLS", "ridge regression"), ("line", "linear in tokens x k")):
        rows.append((f"{lab}, test D", t4[key][0], "outline", None))
        rows.append((f"{lab}, test E", t4[key][1], "outline", None))
    rows.append(("Published predictors, as reported (not blind)", None, "head", None))
    for name, scope, stat, v, p95, _ref in PUBLISHED:
        rows.append((f"{name} ({scope}), {stat}", v, "grey", p95))

    h = 0.30 * len(rows) + 1.6
    fig, ax = plt.subplots(figsize=(10, h))
    fig.subplots_adjust(left=0.43, right=0.95, top=1 - 0.75 / h, bottom=0.95 / h)
    ys = list(range(len(rows)))[::-1]
    for y, (lab, v, style, p95) in zip(ys, rows):
        if style == "head":
            ax.text(-0.765, y, lab, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9,
                    fontweight="bold", color=INK)
            continue
        w = min(v, XMAX)
        kw = {"fill": dict(color=BLUE, edgecolor=BLUE),
              "outline": dict(color=SURFACE, edgecolor=BLUE, linewidth=1.3),
              "grey": dict(color=GREY, edgecolor=GREY)}[style]
        ax.barh(y, w, height=0.62, **kw)
        if style == "grey":
            txt = f"{v:g}%"  # as the source paper prints it
        else:
            txt = f"{v:.2f}%" if v < 100 else f"{v:.1f}%"
        if v > XMAX:
            for dx in (-2.2, -1.5):  # a break mark: the bar runs on past the axis
                ax.plot([XMAX + dx - 0.25, XMAX + dx + 0.25], [y - 0.42, y + 0.42], color=SURFACE, lw=2.6,
                        solid_capstyle="butt", zorder=3)
            ax.text(XMAX - 3.0, y, f"{txt}, off scale", ha="right", va="center", fontsize=8, color=INK)
        else:
            ax.text(w + 0.4, y, txt, ha="left", va="center", fontsize=8, color=INK)
        if p95 is not None:
            ax.plot([p95, p95], [y - 0.31, y + 0.31], color=INK2, lw=1.4)
            ax.text(p95 + 0.4, y + 0.02, f"p95 {p95:g}%", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([y for y, r in zip(ys, rows) if r[2] != "head"])
    ax.set_yticklabels([r[0] for r in rows if r[2] != "head"], fontsize=8.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("MAPE, mean |predicted / measured - 1| (%); Frontier row: p50 of |e| with its p95 marked")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.text(0.015, 1 - 0.22 / h, "Blind MAPE of this study beside registered baselines and published predictors",
             fontsize=11.5, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 1 - 0.48 / h,
             "Filled: the five blind tests. Outlined: baselines frozen in the same registrations, on the same decode "
             "cells. Grey: other papers' numbers, as reported, on their own kernels and GPUs.",
             fontsize=8, color=INK2, va="top")
    fig.text(0.015, 0.12 / h,
             "Data: committed score files at 73508b92 (test A: its registered scorer on the published pages), listed in "
             "fig17_side_by_side.py.\nPublished values: Feng 2026, Zhang 2026, Lee 2025, Yao 2026, Mo 2026 (paper-v2 "
             "references).",
             fontsize=7.5, color=MUTED, va="bottom")
    fig.savefig(HERE / "fig17_side_by_side.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
