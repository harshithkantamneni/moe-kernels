#!/usr/bin/env python
"""ONE PRIVATE A-TILE LAW POOLED OVER SEVERAL CARDS' PAGES (2026-10-01).

    python scripts/atile_pooled_fit.py --form mix  --cards DIR [DIR ...] --out R1.json
    python scripts/atile_pooled_fit.py --form mixT --cards DIR [DIR ...] --out R2.json

WHAT IT IS FOR. The registered byte model (`wave_split_bytes.py`, MIX view) is
fitted per card, and its 8x7B fit carried to another model misses PRIVATE's
A-tile reads at large G (docs/registered/2026-10-01-atile-ksteps-gh200): w1 is
over-predicted on every top_k = 8 model at every G (the harness's routing gives
k experts identical token sets, so k M-tiles read one A tile), and w2 is
under-predicted at G >= 32 on the long-K models. This fits the two candidate
fixes the registration names, on PRIVATE cells alone (every G, every tread,
both GEMMs; PRIVATE reads each slab once, so its excess over n is A alone):

  mix   R1: content-keyed w1 A (`Model(content_a=True)`), one A survival law
        (C_A, beta_A, theta1_A) for both GEMMs, w1 in the fill view and w2 in
        the ws view (wave_split_bytes' MIX), no duration term;
  mixT  R2: R1 with the duration factor s(ks) = (1 + ks / k0_A)^(-a_A) on
        every A distance (`Params.k0_A`, `Params.a_A`).

POOLED, AND SAID SO. wave_split_bytes never pools cards; this tool exists to
pool them, and its output names every card it was fitted on and whether that
set is the calibration set alone (Mixtral 8x7B, Mixtral 8x22B, Qwen2-57B-A14B,
OLMoE-1B-7B) or includes diagnosis models, whose pages were seen. The knee
form min(1, k0 / ks)^a with the knee fixed at 40 k-steps is NOT offered: the
40 was read off the diagnosis models (JetMoE, Phi), so it is not a
calibration result (review, 2026-10-01).

Loss: the sum over cells of (q_pred / q_meas - 1)^2; C and beta as logs,
theta1_A as a logit, k0_A as a log; wave_split_bytes' simplex, fixed-seed
starts, each polished at a third of the step. The slab law is irrelevant on
PRIVATE (no slab re-reads) and is pinned out of the way (C_B 1e9 MiB).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import wave_split_bytes as W  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

FORMS = ("mix", "mixT")
#: The study's calibration models: a pooled fit on these alone is primary.
CALIBRATION_MODELS = ("mixtral-8x7b", "mixtral-8x22b", "qwen2-57b-a14b", "olmoe-1b-7b")
STARTS = 4
ITERS = 800
SEED = 2


def private_cells(card: W.Card) -> list:
    return [("private", G, n, g) for g in W.GEMMS for G in sorted(card.pages)
            for n in card.treads]


def unpack(v, form: str) -> W.Params:
    p = W.Params(C_A=math.exp(v[0]), beta_A=math.exp(v[1]), theta1_A=W._sig(v[2]),
                 C_B=1e9, beta_B=1.0, theta1=1.0, eps1_w1=0.0, eps1_w2=0.0, view=W.MIX)
    if form == "mixT":
        p = p.replace(k0_A=math.exp(v[3]), a_A=float(v[4]))
    return p


def _batches(cards):
    out = []
    for c in cards:
        cells = private_cells(c)
        m = W.Model(c.geom, content_a=True)
        out.append((c, cells, m.batch(cells), np.array([c.measured[x] for x in cells])))
    return out


def card_key(c: W.Card) -> str:
    """Card slug and model: every GH200 card carries one slug, so the slug alone
    would fold eight models' residuals into one."""
    return f"{c.label}:{c.geom.model}"


def residuals(batches, prm: W.Params) -> dict:
    out = {}
    for c, cells, b, y in batches:
        k = card_key(c)
        if k in out:
            raise W.Refused(f"two inputs are one card and one model ({k}); pool each once")
        out[k] = (cells, b.solve(prm) / y - 1.0)
    return out


def fit_pooled(cards, form: str) -> tuple[W.Params, dict]:
    """The pooled fit of `form` on `cards`' PRIVATE cells, and its residuals
    {card: (cells, q_pred / q_meas - 1)}."""
    if form not in FORMS:
        raise W.Refused(f"no form {form!r}; the forms are {FORMS}")
    B = _batches(cards)
    x0 = [math.log(60), math.log(1.5), W._logit(0.2)]
    st = [0.4, 0.3, 0.8]
    if form == "mixT":
        x0 += [math.log(40), 0.7]
        st += [0.4, 0.3]

    def sse(v):
        return float(sum(np.sum(r ** 2) for _c, r in residuals(B, unpack(v, form)).values()))

    best = None
    rng = np.random.default_rng(SEED)
    for s in range(STARTS):
        xs = np.array(x0) + (0 if s == 0 else rng.normal(0, 0.5, len(x0)))
        v, val = W.nelder_mead(sse, xs, st, iters=ITERS)
        v, val = W.nelder_mead(sse, v, [z / 3 for z in st], iters=ITERS)
        if best is None or val < best[1]:
            best = (v, val)
    prm = unpack(best[0], form)
    return prm, residuals(B, prm)


def label_for(cards) -> str:
    models = {c.geom.model for c in cards}
    if models <= set(CALIBRATION_MODELS):
        return "calibration-only"
    return "includes diagnosis models (their pages were seen)"


def doc_for(cards, form: str, prm: W.Params, res: dict) -> dict:
    allr = np.concatenate([r for _c, r in res.values()])
    return {"tool": "scripts/atile_pooled_fit.py", "form": form, "content_a": True,
            "label": label_for(cards), "view": W.MIX,
            "fitted_on": {card_key(c): {str(G): str(p) for G, p in
                                                         sorted(c.paths.items())}
                          for c in cards},
            "models": sorted({c.geom.model for c in cards}),
            "params": prm.as_json(),
            "rms_pct": round(100 * float(np.sqrt(np.mean(allr ** 2))), 3),
            "worst_pct": round(100 * float(allr[np.argmax(abs(allr))]), 3),
            "per_card_rms_pct": {k: round(100 * float(np.sqrt(np.mean(r ** 2))), 3)
                                 for k, (_c, r) in res.items()}}


def read_pooled(path: Path) -> tuple[W.Params, bool, dict]:
    """(Params, content_a, the whole document) of a pooled params file."""
    d = json.loads(Path(path).read_text())
    if d.get("tool") != "scripts/atile_pooled_fit.py" or "params" not in d:
        raise W.Refused(f"{path} is not an atile_pooled_fit params file")
    return W.Params.from_json(d["params"]), bool(d.get("content_a")), d


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--form", required=True, choices=FORMS)
    p.add_argument("--cards", nargs="+", type=Path, required=True,
                   help="each card's r3c-g*.json directory (lock pages)")
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cards = [W.load_card(p) for p in args.cards]
        prm, res = fit_pooled(cards, args.form)
    except W.Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    d = doc_for(cards, args.form, prm, res)
    print(f"{args.form} ({d['label']}) on {', '.join(d['models'])}: "
          f"C_A {prm.C_A:.2f} MiB beta_A {prm.beta_A:.3f} theta1_A {prm.theta1_A:.4f}"
          + (f" k0_A {prm.k0_A:.2f} a_A {prm.a_A:.3f}" if prm.k0_A else "")
          + f"; rms {d['rms_pct']:.2f}% worst {d['worst_pct']:+.2f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(d, indent=1) + "\n")
        print(f"wrote {args.out}")
    return exit_codes.DONE


if __name__ == "__main__":
    sys.exit(main())
