"""The wave-split capacity byte model (`scripts/wave_split_bytes.py`), pinned to
the published r3-arms counter pages. The model is NOT FINAL; these tests hold
it to what the w2 judge re-ran and reproduced on 2026-09-26, so a change that
moves any of it is visible.

What the tests hold it to, and why each can go wrong silently:

    the EVENTS   -- the exact pid walk and the closed form must agree on every
                    cell (504 of 504), reduce to `DCR.group_reads` and to n in
                    the limit where L2 serves nothing across groups, and give
                    the judge's event counts; a wrong walk is a wrong model
                    that still fits
    the INPUTS   -- W_c, the SM count and L2 come off each page (the research
                    code typed them), and a grid the walk would not reproduce
                    is refused
    the MODEL    -- the judge's pinned predictions with the reference
                    parameters, to 1e-4, and the judge's per-card scores
    the GUARD    -- the Hopper G=1 n >= 6 branch prints no number, for the
                    judge's three reasons (kappa, two roots, monotonicity);
                    nothing on the A100 trips it
    the FIT      -- per card, never pooled: the reference SSEs (pinned at <=
                    1.01 x so another optimizer still passes) and parameters,
                    and the A100's "slab law NOT IDENTIFIED" flag
    the CONSUMER -- `per_tile_model_fit`'s ovl.gemm.bytes on the published
                    GH200 and H100 lock pages: the judge's fit rms, bandwidth
                    and n=5 errors; the H200 gets two columns, never averaged

The review of 2026-09-26 (F1 to F10) added tests for what had none: the JSON
carries no number for an ILL-POSED cell; a params JSON finds its counter
pages from any cwd and refuses when they are gone; a timed page of another
kernel, declaration or unnamed device is refused; the Hopper rule holds where
the guard would pass; the w2 decomposition sums to q (s10's rows); the
judge's total-q scores; a page with two occupancies is refused; Phi sits on
both floors of ovl.gemm.bytes; the consumer prints NOT FINAL and says when a
fit leaned on OUT-OF-DOMAIN q; a flat loss flags a parameter at its bound;
and a missing ws view prints n/a, not the ILL-POSED '--'.

Every pinned number is from the judge's report (section 6, the test table),
computed there from the same published pages; the tolerances are the judge's.
The decomposition rows are s10's (printed to three decimals, so +-5e-4).
"""
from __future__ import annotations

import dataclasses
import glob
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.dram_counter_route as DCR  # noqa: E402
from moe.bench import exit_codes  # noqa: E402

SCRIPT = ROOT / "scripts" / "wave_split_bytes.py"
PTF_SCRIPT = ROOT / "scripts" / "per_tile_model_fit.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


W = _load("wave_split_bytes", SCRIPT)
PTF = _load("per_tile_model_fit", PTF_SCRIPT)

PUB = ROOT / "results" / "published"
PAGES = {
    "A100": PUB / "2026-09-25-nvidia_a100_sxm4_40gb-r3-counters" / "results"
    / "2026-09-25-nvidia_a100_sxm4_40gb-r3-counters",
    "GH200": PUB / "2026-09-25-nvidia_gh200_480gb-session" / "results"
    / "2026-09-25-nvidia_gh200_480gb-r3-counters",
    "H100": PUB / "2026-09-25-nvidia_h100_80gb_hbm3-session" / "results"
    / "2026-09-25-nvidia_h100_80gb_hbm3-r3-counters",
}

#: The timed pages the judge's overlap map used: the 1710 MHz lock runs, one
#: per G, of each card's gaps session.
TIMED = {
    "GH200": ("2026-09-25-nvidia_gh200_480gb-session",
              ("d9f1f37c", "df37ea07", "01c08abd", "1b285de2", "6ff34777")),
    "H100": ("2026-09-25-nvidia_h100_80gb_hbm3-session",
             ("ff84f0aa", "0708f122", "48bc690e", "22da27e7", "ceb347ad")),
}

#: THE REFERENCE PARAMETERS: the research fit (s07, both views), refitted from
#: scratch by the judge with a maximum relative difference of 0.0. Full
#: precision: the judge's rounded table (section 6, item 6b) moves GH200 w2
#: SHARED G=1 n=3 by 3e-4, over item 7's 1e-4, because kappa there is 2.44.
PINNED = {
    "GH200": {
        "fill": dict(C_A=104.77298201019305, beta_A=1.363657505113176,
                     theta1_A=0.13607235446913873, C_B=69.72419380973116,
                     beta_B=2.110207215616868, theta1=0.3129200794356074,
                     eps1_w1=7.057485540293951e-08, eps1_w2=0.13062526387044826),
        "ws": dict(C_A=123.21227383926845, beta_A=1.550096808147655,
                   theta1_A=0.0012466126148755975, C_B=69.74998375083558,
                   beta_B=2.1155659389861055, theta1=0.3135056006642324,
                   eps1_w1=1.014750573033949e-08, eps1_w2=0.12698866780995183)},
    "H100": {
        "fill": dict(C_A=130.64874129609117, beta_A=1.1628233930192484,
                     theta1_A=0.25757594734712674, C_B=70.18625081470832,
                     beta_B=2.088618071300818, theta1=0.31759221549880984,
                     eps1_w1=2.6121401881226874e-05, eps1_w2=0.13953459374065463),
        "ws": dict(C_A=156.27724130637426, beta_A=1.2781028048858412,
                   theta1_A=0.23910386773067657, C_B=70.23787237425202,
                   beta_B=2.0994666731124942, theta1=0.3170363442641671,
                   eps1_w1=1.3726012864514416e-05, eps1_w2=0.13211057653311625)},
    "A100": {
        "fill": dict(C_A=56.35369973842013, beta_A=1.5403113074252093,
                     theta1_A=0.45484634900190096, C_B=6.7458380032830325,
                     beta_B=9.879671931097418, theta1=0.044055759183218704,
                     eps1_w1=0.1224428550195925, eps1_w2=0.6468048574586234),
        "ws": dict(C_A=66.02465174415669, beta_A=1.8214984543101471,
                   theta1_A=0.4570005157703069, C_B=8.682059032604895,
                   beta_B=9.744823340887626, theta1=0.05687695974408642,
                   eps1_w1=0.12233685334236392, eps1_w2=0.6465528503777356)},
}

#: The judge's table (item 6b): the pinned parameters rounded as printed.
TABLE_6B = {
    "GH200": (104.77, 1.364, 0.136, 69.72, 2.110, 0.313, 0.0, 0.1306),
    "H100": (130.65, 1.163, 0.258, 70.19, 2.089, 0.318, 0.0, 0.1395),
    "A100": (56.35, 1.540, 0.455, 6.75, 9.880, 0.044, 0.1224, 0.6468),
}
NAMES_6B = ("C_A", "beta_A", "theta1_A", "C_B", "beta_B", "theta1", "eps1_w1", "eps1_w2")


def pinned(card: str, view: str = "fill"):
    return W.Params(**PINNED[card][view], view=view)


@pytest.fixture(scope="module")
def cards():
    return {c: W.load_card(d) for c, d in PAGES.items()}


#: The judge's pins (2026-09-26) were fitted with the slab law at every re-read;
#: they are held there. The later-miss rule (LATER_MISS, 2026-09-28) has its own
#: tests at the end of this file.
@pytest.fixture(autouse=True)
def _judge_law(request):
    if "later_miss" in request.keywords:
        yield
        return
    with W.later_law(False):
        yield


@pytest.fixture(scope="module")
def pinned_results(cards):
    with W.later_law(False):
        return {c: W.analyse(cards[c], params={v: pinned(c, v) for v in ("fill", "ws")})
                for c in cards}


@pytest.fixture(scope="module")
def fitted(cards):
    """The fill view fitted from scratch on each card alone (about five
    seconds a card)."""
    with W.later_law(False):
        return {c: W.fit_view(cards[c], "fill") for c in cards}


def _npm(arm: str, n: int) -> int:
    """The published pages' grid rows: 72 declared slots for SHARED and
    PRIVATE, 8 for NATIVE, E = 8, BLOCK_M = 32."""
    slots = 8 if arm == "native" else 72
    return -(-(256 * n + slots * 31) // 32)


# --------------------------------------------------------------------------
# 1-3. The events.
# --------------------------------------------------------------------------

def test_1_the_closed_form_equals_the_pid_walk_on_all_504_cells():
    agree = total = 0
    for arm in ("shared", "native", "private"):
        for G in (1, 2, 4, 8, 16, 64):
            for n in (1, 2, 3, 4, 5, 6, 8):
                for P, Wc in ((448, 660), (64, 528), (64, 432), (448, 432)):
                    walked = W.walk(arm, G, n, P, Wc, _npm(arm, n), 8).counts()
                    total += 1
                    agree += walked == W.closed_form(arm, G, n, P, Wc, 8)
    assert (agree, total) == (504, 504)


def test_2_with_no_cross_group_survival_it_is_the_group_model_exactly(cards):
    """sigma_B = 0 (every cross-group re-read misses), eps1 = 0, sigma_A = 1
    (no activation re-read reaches DRAM): SHARED and NATIVE must be
    `DCR.group_reads`, PRIVATE n, with no mismatch, in both views."""
    model = W.Model(cards["GH200"].geom)
    cells = [(a, G, n, g) for a in ("shared", "native", "private")
             for G in (1, 2, 4, 8, 16, 64) for n in (1, 2, 3, 4, 5, 6, 8) for g in ("w1", "w2")]
    want = np.array([float(n) if a == "private" else DCR.group_reads(8, n, G)
                     for a, G, n, _g in cells])
    for view in ("fill", "ws"):
        limit = W.Params(C_A=1e30, beta_A=1.0, theta1_A=0.5, C_B=1e-30, beta_B=1.0,
                         theta1=0.5, eps1_w1=0.0, eps1_w2=0.0, view=view)
        got = model.q(limit, cells)
        assert int(np.sum(np.abs(got - want) > 1e-12)) == 0, view


def _counts(arm, G, n, P, Wc):
    return W.walk(arm, G, n, P, Wc, _npm(arm, n), 8).counts()


def test_3_the_event_counts_are_the_judges():
    c = _counts("shared", 2, 3, 64, 528)
    assert c["slabs"] == 512
    assert c["x"] == {(127, True): 136, (127, False): 376}
    assert (c["in1"], c["inL"]) == (192, 320)
    assert c["A"] == {(2, True): 518, (2, False): 994}
    c = _counts("shared", 2, 3, 64, 432)
    assert c["x"] == {(127, True): 128, (127, False): 384}
    assert (c["in1"], c["inL"]) == (152, 360)
    assert c["A"] == {(2, True): 424, (2, False): 1088}
    c = _counts("shared", 1, 2, 448, 660)
    assert c["x"] == {(448, True): 212, (448, False): 3372}
    assert c["A"] == {(1, True): 658, (1, False): 6494}
    c = _counts("shared", 1, 2, 448, 432)
    assert c["x"] == {(448, False): 3584}
    c = _counts("shared", 4, 6, 64, 528)
    assert c["x"] == {(253, True): 68, (253, False): 444}
    assert (c["in1"], c["inL"]) == (332, 1716)
    assert c["A"] == {(4, True): 516, (4, False): 2508}
    c = _counts("native", 64, 2, 64, 528)
    assert c["x"] == {}
    assert (c["in1"], c["inL"]) == (264, 248)
    assert c["A"] == {(16, True): 512, (16, False): 496}


# --------------------------------------------------------------------------
# 4-5. The inputs, off the pages.
# --------------------------------------------------------------------------

def test_4_the_coresidency_window_is_read_off_each_page(cards):
    for card, want in {"A100": (432, 432), "GH200": (660, 528), "H100": (660, 528)}.items():
        g = cards[card].geom
        assert (g.get("W_c", "w1"), g.get("W_c", "w2")) == want, card
    assert (cards["GH200"].geom.l2_bytes, cards["H100"].geom.l2_bytes,
            cards["A100"].geom.l2_bytes) == (60 * 2 ** 20, 50 * 2 ** 20, 40 * 2 ** 20)
    assert {cards[c].geom.sm_count for c in cards} == {108, 132}


def _copy_pages(src: Path, dst: Path, edit) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("r3c-g*.json")):
        page = json.loads(p.read_text())
        edit(page)
        (dst / p.name).write_text(json.dumps(page))
    return dst


def test_4b_a_page_with_another_occupancy_gets_its_own_window(tmp_path):
    """Nothing about a card is typed: record 3 CTAs per SM for w2 on every
    GH200 cell and W_c,w2 becomes 132 x 3, with w1 untouched."""
    def three(page):
        for c in page["cells"]:
            c["recorded"]["w2"]["launch__occupancy_limit_registers"] = 3.0
    card = W.load_card(_copy_pages(PAGES["GH200"], tmp_path / "gh", three))
    assert (card.geom.get("W_c", "w1"), card.geom.get("W_c", "w2")) == (660, 396)


def test_4c_a_page_whose_cells_record_two_occupancies_is_refused(tmp_path):
    """One kernel binary per GEMM gives one occupancy per page; a page whose
    cells disagree has no single W_c, so it is refused, not averaged."""
    def one_cell(page):
        if page["design"]["group_m"] == 2:
            page["cells"][0]["recorded"]["w2"]["launch__occupancy_limit_registers"] = 3.0
    with pytest.raises(W.Refused, match=r"\[3, 4\] CTAs per SM"):
        W.load_card(_copy_pages(PAGES["GH200"], tmp_path / "gh", one_cell))


def test_5_the_measured_q_is_the_judges(cards):
    m = {c: cards[c].measured for c in cards}
    assert m["GH200"][("shared", 2, 2, "w2")] == pytest.approx(1.039, abs=5e-4)
    assert m["GH200"][("shared", 1, 6, "w2")] == pytest.approx(3.773, abs=5e-4)
    calls = cards["GH200"].calls[("shared", 1, 6, "w2")]
    assert (min(calls), max(calls)) == (pytest.approx(3.536, abs=5e-4),
                                        pytest.approx(3.984, abs=5e-4))
    assert m["H100"][("shared", 1, 2, "w1")] == pytest.approx(1.944, abs=5e-4)
    assert m["A100"][("shared", 2, 2, "w2")] == pytest.approx(1.363, abs=5e-4)
    assert m["GH200"][("shared", 64, 2, "w2")] == pytest.approx(1.297, abs=5e-4)
    assert m["GH200"][("native", 64, 2, "w2")] == pytest.approx(1.179, abs=5e-4)


# --------------------------------------------------------------------------
# 6. The fit, per card.
# --------------------------------------------------------------------------

def test_6_each_cards_fit_reaches_the_reference_sse(fitted):
    ref = {"GH200": (0.001935, 0.04353), "H100": (0.0009502, 0.05624),
           "A100": (0.004941, 0.06562)}
    for card, (s1, s2) in ref.items():
        f = fitted[card]
        assert (f.stage1.cells, f.stage2.cells) == (50, 60), card
        assert f.stage1.sse <= s1 * 1.01, (card, f.stage1.sse)
        assert f.stage2.sse <= s2 * 1.01, (card, f.stage2.sse)


def test_6b_the_reference_parameters_and_their_flags(fitted, cards):
    for card, row in TABLE_6B.items():
        p = pinned(card)
        for name, want in zip(NAMES_6B, row, strict=True):
            assert getattr(p, name) == pytest.approx(want, abs=0.006 * max(1.0, abs(want))), \
                (card, name)
    for card in ("GH200", "H100"):
        f = fitted[card].params
        ref = pinned(card)
        for name in ("C_A", "beta_A", "C_B", "beta_B"):
            assert getattr(f, name) == pytest.approx(getattr(ref, name), rel=0.02), (card, name)
        for name in ("theta1_A", "theta1", "eps1_w2"):
            assert getattr(f, name) == pytest.approx(getattr(ref, name), abs=0.01), (card, name)
        assert any(flag.startswith("eps1_w1 at bound 0") for flag in fitted[card].flags), card
        assert not any("NOT IDENTIFIED" in flag for flag in fitted[card].flags), card
    a = fitted["A100"].params
    for name in ("C_A", "beta_A"):
        assert getattr(a, name) == pytest.approx(getattr(pinned("A100"), name), rel=0.02)
    assert any("slab law NOT IDENTIFIED" in flag for flag in fitted["A100"].flags)
    assert any("slab law NOT IDENTIFIED" in flag
               for flag in W.param_flags(pinned("A100"), cards["A100"].geom))


# --------------------------------------------------------------------------
# 7-8. The model's numbers and its guard.
# --------------------------------------------------------------------------

PINS_7 = {
    "GH200": [("shared", 2, 2, "w2", 1.0695), ("shared", 2, 6, "w2", 2.6186),
              ("shared", 1, 3, "w2", 1.9637), ("shared", 4, 4, "w2", 1.1075),
              ("shared", 1, 2, "w1", 1.9786), ("private", 64, 6, "w2", 7.6242),
              ("shared", 16, 6, "w2", 1.4162)],
    "H100": [("shared", 2, 2, "w2", 1.0761), ("shared", 1, 2, "w1", 1.9798),
             ("private", 64, 6, "w2", 7.3942)],
    "A100": [("shared", 2, 2, "w2", 1.2777), ("shared", 1, 2, "w2", 1.6297),
             ("private", 64, 6, "w2", 8.7156)],
}


def test_7_the_pinned_predictions(cards):
    for card, pins in PINS_7.items():
        model = W.Model(cards[card].geom)
        got = model.evaluate(pinned(card), [p[:4] for p in pins])
        for r, p in zip(got, pins, strict=True):
            assert W.shown(r) == pytest.approx(p[4], abs=1e-4), (card, p)
    ws = W.Model(cards["GH200"].geom).evaluate(pinned("GH200", "ws"),
                                               [("shared", 16, 6, "w2")])[0]
    assert W.shown(ws) == pytest.approx(1.4982, abs=1e-4)


def test_8_the_guard_prints_no_number_on_the_hopper_branch(cards):
    for arm in ("shared", "native"):
        gh = W.Model(cards["GH200"].geom).evaluate(
            pinned("GH200"), [(arm, 1, n, "w2") for n in (6, 7, 8)])
        n6, n7, n8 = gh
        assert n6["kappa"] == pytest.approx(11.2, abs=0.1)
        assert any(w.startswith("kappa") for w in n6["guard"])
        assert (n7["q"], n7["q_high"]) == (pytest.approx(1.612, abs=5e-4),
                                           pytest.approx(4.551, abs=5e-4))
        assert any(w.startswith("roots") for w in n7["guard"])
        assert n8["q"] == pytest.approx(1.433, abs=5e-4)
        assert n8["q_high"] == pytest.approx(n8["q"], abs=1e-6), "n=8 has one root"
        assert any(w.startswith("monotonicity: 1.433 < 3.746") for w in n8["guard"]), n8
        assert [W.shown(r) for r in gh] == [None, None, None]
    h6 = W.Model(cards["H100"].geom).evaluate(pinned("H100"), [("shared", 1, 6, "w2")])[0]
    assert h6["kappa"] == pytest.approx(17.1, abs=0.1)
    assert W.shown(h6) is None
    a100 = W.Model(cards["A100"].geom)
    cells = [(a, G, n, g) for a in ("shared", "native", "private")
             for G in (1, 2, 4, 8, 16, 32, 64) for n in range(1, 9) for g in ("w1", "w2")]
    ill = [r["cell"] for r in a100.evaluate(pinned("A100"), cells) if r["ill"]]
    assert ill == []


#: Every cross-group re-read misses, no partner loss, no activation miss: F
#: is constant in q, so kappa is 1, the roots coincide and G=1 rises with n.
#: The guard passes everywhere; only a rule can mark a cell.
NO_RECOVERY = dict(C_A=1e30, beta_A=1.0, theta1_A=0.5, C_B=1e-30, beta_B=1.0, theta1=0.5,
                   eps1_w1=0.0, eps1_w2=0.0)


def test_8b_the_hopper_rule_holds_where_the_guard_would_pass(cards):
    """The spec's domain rule: Hopper w2 SHARED/NATIVE at G=1, n >= 6 is
    ILL-POSED whatever the guard says, so a refit whose map happens to be
    well posed there still prints nothing. The same cells on the A100 (not
    a Hopper), and w1, PRIVATE and G=2 on GH200, print their number."""
    prm = W.Params(**NO_RECOVERY)
    cells = [(a, 1, n, "w2") for a in ("shared", "native") for n in (6, 7, 8)]
    for card in ("GH200", "H100"):
        for r in W.Model(cards[card].geom).evaluate(prm, cells):
            assert r["guard"] == [] and r["kappa"] == pytest.approx(1.0), (card, r["cell"])
            assert r["rule"] == W.ILL_POSED and W.shown(r) is None, (card, r["cell"])
    rest = W.Model(cards["GH200"].geom).evaluate(
        prm, [("private", 1, 6, "w2"), ("shared", 1, 6, "w1"), ("shared", 2, 6, "w2")])
    assert [W.shown(r) for r in rest] == pytest.approx([6.0, 6.0, 3.0])
    a100 = W.Model(cards["A100"].geom).evaluate(prm, cells)
    assert [W.shown(r) for r in a100] == pytest.approx([6.0, 7.0, 8.0, 6.0, 7.0, 8.0])


# --------------------------------------------------------------------------
# 9-10. Scores per card, and the cross-card L2 check.
# --------------------------------------------------------------------------

def test_9_the_judges_scores_per_card_in_the_fill_view(pinned_results):
    want = {"A100": (3.75, 18.04, 5.17, 1.26), "GH200": (6.20, 17.96, 4.84, 0.84),
            "H100": (6.13, 18.73, 5.10, 0.56)}
    for card, (all120, group, w2sn, w2p) in want.items():
        j = W.scores(pinned_results[card])["judge"]
        e = j["120 cells: S+N+P, w1+w2"]["fill"]
        assert e["cells"] == 120
        assert 100 * e["rms"] == pytest.approx(all120, abs=0.006), card
        assert 100 * e["group_rms"] == pytest.approx(group, abs=0.006), card
        assert 100 * j["w2 S+N, G<=16"]["fill"]["rms"] == pytest.approx(w2sn, abs=0.006), card
        assert 100 * j["w2 PRIVATE"]["fill"]["rms"] == pytest.approx(w2p, abs=0.006), card


def test_9b_the_judges_total_q_scores(pinned_results):
    """Section 1's last row, (q_w1 W_w1 + q_w2 W_w2) / W against the measured
    total, S+N+P at n >= 2 (60 cells a card): what a timed model consumes.
    Each GEMM weighted by its own W_g: equal weights give 5.21 on GH200's fill
    view, not 3.66."""
    want = {"A100": (2.37, 2.34), "GH200": (3.66, 1.53), "H100": (3.76, 1.77)}
    for card, (fill, ws) in want.items():
        t = W.scores(pinned_results[card])["judge"]["total q, S+N+P (what timing uses)"]
        assert (t["fill"]["cells"], t["ws"]["cells"]) == (60, 60), card
        assert 100 * t["fill"]["rms"] == pytest.approx(fill, abs=0.006), card
        assert 100 * t["ws"]["rms"] == pytest.approx(ws, abs=0.006), card


def test_10_the_cross_card_l2_check(pinned_results):
    checks = W.cross_card([pinned_results[c] for c in ("GH200", "H100", "A100")])
    assert len(checks) == 1, "only GH200 and H100 share a geometry"
    c = checks[0]
    assert (c["card"], c["minus"]) == ("nvidia_h100_80gb_hbm3", "nvidia_gh200_480gb")
    assert {(r["G"], r["n"]) for r in c["rows"]} == {(G, n) for G in (2, 4, 16)
                                                      for n in (2, 3, 4, 6)}
    assert c["worst_gap"] <= 0.05
    assert c["worst_gap"] == pytest.approx(0.036, abs=0.001)
    assert c["verdict"] == "PASS"


# --------------------------------------------------------------------------
# 11. The consumer: per_tile_model_fit's ovl.gemm.bytes.
# --------------------------------------------------------------------------

def _timed(card: str):
    sess, rids = TIMED[card]
    dirs = []
    for rid in rids:
        found = glob.glob(str(PUB / sess / "results" / "gaps-*" / "private_weight_reference"
                              / f"*{rid}"))
        assert len(found) == 1, rid
        dirs.append(Path(found[0]))
    pages = [PTF.load_r3(d, PTF.R3_ARMS) for d in dirs]
    ctx = PTF._context(pages, PTF._resolve_ruler(pages, None), None)
    return pages, ctx, [c for p in pages for c in p.cells]


def test_11_the_time_consumer_on_the_lock_pages(cards):
    want = {"GH200": (0.012, 3503.0, 1.69), "H100": (0.008, 2960.0, 0.39)}
    for card, (rms_max, bw, t5) in want.items():
        pages, ctx, cells = _timed(card)
        source = W.QSource(cards[card].geom, pinned(card), cards[card])
        col = PTF.fit_gemm_bytes(cells, ctx, source, {p.run: p for p in pages})
        assert len(col.fitted) == 75 and not col.model_only, card
        assert {c.q_origin for c in col.fitted} == {"measured"}
        assert col.fit.rms <= rms_max, (card, col.fit.rms)
        assert col.fit.bandwidth_gbps == pytest.approx(bw, rel=0.01), card
        assert len(col.scored) == 15 and {c.tread for c in col.scored} == {5}
        rms5 = 100 * float(np.sqrt(np.mean(col.scored_rel ** 2)))
        assert rms5 == pytest.approx(t5, abs=0.006), (card, rms5)
        # The section says the byte model is NOT FINAL and what would falsify
        # it, the test aimed at this consumer included (review F5).
        text = "\n".join(PTF.gemm_bytes_lines([col], ctx))
        assert col.status == "NOT FINAL" and "): NOT FINAL;" in text, card
        assert "WHAT WOULD FALSIFY IT" in text and "above 3% at G=1 or G=2 SHARED" in text
        assert "OUT-OF-DOMAIN q" not in text, "a measured-q fit has no domain to leave"


def test_11b_a_card_with_no_counter_page_gets_one_column_per_source(cards):
    """The H200 has no counter pages: its cells take GH200's and H100's model
    q, two columns, each fitted alone, never averaged; the G=1 n=6 cells are
    ILL-POSED in both and are not timed."""
    found = glob.glob(str(PUB / "2026-09-23-nvidia_h200-session5" / "results" / "gaps-*"
                          / "private_weight_reference" / "*4efa4954"))
    page = PTF.load_r3(Path(found[0]), PTF.R3_ARMS)
    ctx = PTF._context([page], PTF._resolve_ruler([page], None), None)
    cols = [PTF.fit_gemm_bytes(page.cells, ctx,
                               W.QSource(cards[c].geom, pinned(c), cards[c]),
                               {page.run: page}) for c in ("GH200", "H100")]
    assert [c.model_only for c in cols] == [True, True]
    assert cols[0].label != cols[1].label
    for col in cols:
        assert {(c.arm, c.tread) for c in col.untimed} == {("R3 shared", 6), ("R3 native", 6)}
        assert {c.q_origin for c in col.fitted} == {col.label}
    q = {c.arm + str(c.tread): c.q_w2 for c in cols[0].fitted}
    q2 = {c.arm + str(c.tread): c.q_w2 for c in cols[1].fitted}
    assert q["R3 shared2"] != q2["R3 shared2"], "each column keeps its own card's q"


def test_11b2_a_model_only_fit_on_out_of_domain_q_says_so(cards):
    """The H200 G=64 page: 12 of its 18 cells are SHARED/NATIVE, whose w2 q is
    OUT-OF-DOMAIN at G=64. The model-only fit uses them (as the judge's j03c
    did) and must say how many, with its rms without them (review F7)."""
    found = glob.glob(str(PUB / "2026-09-23-nvidia_h200-session5" / "results" / "gaps-*"
                          / "private_weight_reference" / "*46f8f227"))
    page = PTF.load_r3(Path(found[0]), PTF.R3_ARMS)
    ctx = PTF._context([page], PTF._resolve_ruler([page], None), None)
    col = PTF.fit_gemm_bytes(page.cells, ctx,
                             W.QSource(cards["GH200"].geom, pinned("GH200"), cards["GH200"]),
                             {page.run: page})
    assert col.model_only and len(col.fitted) == 18
    ood = col.fitted_out_of_domain()
    assert int(ood.sum()) == 12
    assert {c.arm for c, o in zip(col.fitted, ood, strict=True) if o} == {"R3 shared",
                                                                          "R3 native"}
    text = "\n".join(PTF.gemm_bytes_lines([col], ctx))
    assert "12 of the 18 fitted cells take an OUT-OF-DOMAIN q" in text
    rms6 = 100 * float(np.sqrt(np.mean(np.asarray(col.fit.rel)[~ood] ** 2)))
    assert f"rms without them {rms6:.3f}% over 6 cells" in text


def _planted_ctx():
    return PTF.Context(weight_bytes=2_818_572_288, experts=8, tau_pin_ms=0.1, f_ref=1980.0,
                       f_top=1980.0, weight_bytes_by_gemm=(("w1", 1_879_048_192),
                                                           ("w2", 939_524_096)))


def test_11c_ovl_gemm_bytes_puts_phi_on_both_floors():
    """T = T0 + max(q_w1 (W_w1/W) tau, n c_w1 Phi) + max(q_w2 (W_w2/W) tau,
    n c_w2 Phi), Phi = (f_ref / f)^e, against a hand computation at two
    clocks; the q_w2 = 0.4 n cells sit on the c_w2 floor at both clocks, so
    a floor without Phi misses them at 1400 MHz, and a fit of the planted
    times recovers e only with Phi on both floors."""
    ctx = _planted_ctx()
    t0, tau, cw1, cw2, e = 0.05, 0.75, 0.42, 0.13, 1.2
    cells = [PTF.Cell(page="planted", arm="R3 shared", state="0", group_m=2, tread=n, mhz=f,
                      ms=1.0, q_w1=float(n), q_w2=k * n, q_origin="planted")
             for f in (1980.0, 1400.0) for k in (1.0, 0.4) for n in range(1, 7)]

    def by_hand(c):
        phi = (1980.0 / c.mhz) ** e
        return (t0 + max(c.q_w1 * (2 / 3) * tau, c.tread * cw1 * phi)
                + max(c.q_w2 * (1 / 3) * tau, c.tread * cw2 * phi))

    floor_w2 = [c for c in cells if c.q_w2 * tau / 3 < c.tread * cw2 * (1980.0 / c.mhz) ** e]
    assert {c.mhz for c in floor_w2} == {1980.0, 1400.0}
    data = PTF.Data.of(cells, ctx)
    lay = PTF.layout(PTF.GEMM_BYTES, data, ctx)
    assert list(lay.names) == ["T0", "tau", "c_w1", "e", "c_w2"]
    got = PTF.predict(PTF.GEMM_BYTES, lay, np.array([t0, tau, cw1, e, cw2]), data, ctx)
    want = np.array([by_hand(c) for c in cells])
    assert got == pytest.approx(want, rel=1e-12)
    planted = [dataclasses.replace(c, ms=float(w)) for c, w in zip(cells, want, strict=True)]
    f = PTF.fit(PTF.GEMM_BYTES, planted, ctx)
    assert f.rms < 1e-6
    assert dict(zip(f.names, f.values, strict=True))["e"] == pytest.approx(e, rel=1e-3)


def test_11d_a_timed_page_of_another_kernel_or_declaration_is_refused(cards):
    """q from one kernel attached to another's timings is a wrong number that
    still fits (review F3). Each edit of one lock page, alone, refuses: the
    tile, report.json's copies_declared, an arm's declared expert slots; and a
    source that cannot name its kernel."""
    pages, _ctx, cells = _timed("GH200")
    source = W.QSource(cards["GH200"].geom, pinned("GH200"), cards["GH200"])
    have, _lack = PTF.with_gemm_bytes(cells, source, {p.run: p for p in pages})
    assert len(have) == 90
    tile = list(pages[0].tile)
    tile[2] = 64
    edits = [({"tile": tuple(tile)}, "kernel is"),
             ({"copies_declared": 10}, "declared 10 copies"),
             ({"declared_by_arm": dict(pages[0].declared_by_arm, shared=(80,))},
              r"shared rows declare \[80\]")]
    for edit, why in edits:
        bad = {p.run: (dataclasses.replace(p, **edit) if p is pages[0] else p) for p in pages}
        with pytest.raises(PTF.Refused, match=why):
            PTF.with_gemm_bytes(cells, source, bad)
    with pytest.raises(PTF.Refused, match="which kernel"):
        PTF.with_gemm_bytes(cells, lambda *a: (1.0, 1.0, "x"), {p.run: p for p in pages})


def _copy_timed(dirs, dst: Path, edit=None, device: bool = True) -> list[Path]:
    """Copies of timed page directories (no triton cache), `edit` applied to
    each report.json, DEVICE dropped when `device` is False."""
    out = []
    for d in dirs:
        to = dst / d.name
        to.mkdir(parents=True)
        for name in ("cells.csv", "report.json", "report.txt") + (("DEVICE",) if device else ()):
            shutil.copy(d / name, to / name)
        if edit is not None:
            report = json.loads((to / "report.json").read_text())
            edit(report)
            (to / "report.json").write_text(json.dumps(report))
        out.append(to)
    return out


def _timed_dirs(card: str) -> list[Path]:
    sess, rids = TIMED[card]
    return [Path(glob.glob(str(PUB / sess / "results" / "gaps-*" / "private_weight_reference"
                                / f"*{rid}"))[0]) for rid in rids]


def test_11e_the_cli_refuses_before_fitting(pinned_results, tmp_path, capsys):
    """Through `per_tile_model_fit.main`, before any fit: pages all edited to
    BLOCK_M 64 (the review's reproduction, which had exited 0), and a page
    with no DEVICE among pages that name one (review F10: it had not been
    counted, so its cells took model q beside the others' measured q)."""
    dirs = _timed_dirs("GH200")
    pj = tmp_path / "gh200.json"
    pj.write_text(json.dumps(W.params_doc(pinned_results["GH200"])))
    common = ["--r3", "--r3-arms", "native,shared,private", "--gemm-bytes", str(pj),
              "--ruler", str(PTF.find_ruler(dirs[0])), "--no-scan", "--no-logo",
              "--include-invalid", "--include-unscored"]

    def bm64(report):
        report["block_m"] = 64
    edited = _copy_timed(dirs, tmp_path / "bm64", bm64)
    assert PTF.main([*map(str, edited), *common]) == exit_codes.REFUSED
    assert "kernel is ('mixtral-8x7b', 'bf16', 32" in capsys.readouterr().err
    nodev = _copy_timed(dirs[:1], tmp_path / "nodev", device=False)
    assert PTF.main([str(nodev[0]), *map(str, dirs[1:]), *common]) == exit_codes.REFUSED
    assert "record no DEVICE" in capsys.readouterr().err
    page = PTF.Page(path=tmp_path, kind="R3", run="a", label="VALID", card="u1")
    assert PTF.gemm_bytes_card([page, dataclasses.replace(page, run="b")]) == "u1"
    with pytest.raises(PTF.Refused, match="2 cards"):
        PTF.gemm_bytes_card([page, dataclasses.replace(page, run="b", card="u2")])


# --------------------------------------------------------------------------
# The rest of the spec: refusals, statuses, and NOT FINAL on every output.
# --------------------------------------------------------------------------

def test_a_grid_the_walk_would_not_reproduce_is_refused(tmp_path):
    def wider(page):
        for c in page["cells"]:
            if c["arm"] == "native" and c["n"] == 2:
                c["per_gemm"]["w2"]["grid_size"] += 64
    with pytest.raises(W.Refused, match="grid"):
        W.load_card(_copy_pages(PAGES["GH200"], tmp_path / "gh", wider))


def test_statuses_domain_and_what_is_scored(pinned_results):
    rows = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in pinned_results["GH200"].rows}
    assert rows[("shared", 64, 4, "w2")]["status"] == W.OUT_OF_DOMAIN
    assert rows[("native", 64, 4, "w2")]["status"] == W.OUT_OF_DOMAIN
    assert rows[("shared", 64, 4, "w1")]["status"] == W.HELD_OUT
    assert rows[("shared", 16, 4, "w2")]["status"] == W.HELD_OUT
    assert rows[("private", 64, 4, "w2")]["status"] == W.FIT
    assert rows[("shared", 2, 4, "w2")]["status"] == W.FIT
    assert rows[("shared", 1, 6, "w2")]["status"] == W.ILL_POSED
    own = W.scores(pinned_results["GH200"])["own"]
    everything = own["every in-domain, well-posed cell"]["fill"]
    assert everything["cells"] == 120 - 8 - 2
    assert everything["out_of_domain"] == 0 and everything["ill_posed"] == 0
    a100 = pinned_results["A100"]
    assert all(r["status"] != W.ILL_POSED for r in a100.rows)
    pred = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in pinned_results["GH200"].predicted}
    assert pred[("shared", 8, 2, "w2")]["status"] == W.PREDICTED
    assert pred[("shared", 8, 2, "w2")]["q_fill"] == pytest.approx(1.082, abs=5e-4)
    assert pred[("shared", 32, 2, "w2")]["status"] == W.OUT_OF_DOMAIN
    assert pred[("shared", 1, 8, "w2")]["status"] == W.ILL_POSED
    fw = W.first_wave(pinned_results["GH200"])
    assert fw["s1"] == pytest.approx(212 / 3584) and fw["events_in_window"] == 212
    assert [round(x["phi"], 2) for x in fw["per_n"]] == [0.26, 0.27, 0.26, 0.27]
    assert W.first_wave(pinned_results["A100"])["s1"] == 0.0


def test_the_output_says_not_final_and_what_would_falsify_it(cards, pinned_results, tmp_path,
                                                            capsys):
    doc = W.__doc__
    assert "NOT FINAL" in doc and "WHAT WOULD FALSIFY IT" in doc
    pdir = tmp_path / "params"
    pdir.mkdir()
    both = {"cards": {pinned_results[c].label: W.params_doc(pinned_results[c])
                      for c in ("GH200", "A100")}}
    (pdir / "pinned.json").write_text(json.dumps(both))
    out = tmp_path / "out.json"
    code = W.main([str(PAGES["GH200"]), str(PAGES["A100"]), "--params", str(pdir / "pinned.json"),
                   "--params-out", str(tmp_path / "written"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "STATUS: NOT FINAL" in printed and "WHAT WOULD FALSIFY IT" in printed
    assert "already give -3.14% at G=2 SHARED" in printed, "the open falsifier (review F6)"
    assert "[nvidia_gh200_480gb] CELLS" in printed and "[nvidia_a100_sxm4_40gb] CELLS" in printed
    assert "slab law NOT IDENTIFIED" in printed
    doc = json.loads(out.read_text())
    assert doc["status"] == "NOT FINAL"
    written = json.loads((tmp_path / "written" / "nvidia_gh200_480gb.wsc-params.json")
                         .read_text())
    assert written["status"] == "NOT FINAL" and written["W_c"] == {"w1": 660, "w2": 528}
    assert written["views"]["fill"]["params"]["C_B"] == pytest.approx(PINNED["GH200"]["fill"]
                                                                        ["C_B"])
    src = W.q_source(tmp_path / "written" / "nvidia_gh200_480gb.wsc-params.json")
    q1, q2, origin = src("shared", 2, 5)
    assert origin == "WSC-fill[nvidia_gh200_480gb]"
    assert q2 == pytest.approx(2.690, abs=5e-4)
    assert src("shared", 2, 2, cards["GH200"].geom.uuid)[2] == "measured"
    assert src("shared", 1, 6) is None, "an ILL-POSED cell has no number to time"


def test_the_printed_table_carries_no_number_for_an_ill_posed_cell(pinned_results):
    res = pinned_results["H100"]
    lines = W.card_lines(res, W.scores(res))
    row = next(line for line in lines if line.split()[:4] == ["shared", "1", "6", "w2"])
    fields = row.split()
    assert fields[5:7] == ["--", "--"] and "ILL-POSED" in row



def _pinned_json(pinned_results, tmp_path: Path, *cards_) -> Path:
    path = tmp_path / "pinned.json"
    path.write_text(json.dumps({"cards": {pinned_results[c].label: W.params_doc(pinned_results[c])
                                          for c in cards_}}))
    return path


def test_the_json_carries_no_number_for_an_ill_posed_cell(pinned_results, tmp_path, capsys):
    """--out's JSON: q_fill and q_ws are null on every ILL-POSED cell, page
    cells and predictions alike; the spurious root the guard rejected lives
    only under least_root_fill (review F1: it had sat under q_fill)."""
    out = tmp_path / "cells.json"
    assert W.main([str(PAGES["GH200"]), "--params", str(_pinned_json(pinned_results, tmp_path,
                                                                      "GH200")),
                   "--out", str(out)]) == 0
    capsys.readouterr()
    doc = json.loads(out.read_text())["cards"]["nvidia_gh200_480gb"]
    rows = doc["cells"] + doc["predicted"]
    ill = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in rows if r["status"] == W.ILL_POSED}
    assert {("shared", 1, 6, "w2"), ("native", 1, 6, "w2"), ("shared", 1, 7, "w2"),
            ("shared", 1, 8, "w2")} <= set(ill)
    for key, r in ill.items():
        assert r["q_fill"] is None, key
        assert r["q_ws"] is None or not r["ill_ws"], key
    assert ill[("shared", 1, 8, "w2")]["least_root_fill"] == pytest.approx(1.433, abs=5e-4)
    assert ill[("shared", 1, 6, "w2")]["least_root_fill"] == pytest.approx(3.746, abs=5e-4)
    fine = [r for r in rows if r["status"] != W.ILL_POSED]
    assert fine and all(isinstance(r["q_fill"], float) for r in fine)
    assert all(r["q_fill"] == r["least_root_fill"] for r in fine)


def test_a_params_json_finds_its_counter_pages_from_any_cwd(pinned_results, tmp_path,
                                                           monkeypatch, capsys):
    """Pages named on the command line relative to the repo, the params JSON
    written elsewhere, then read from a third directory: the measured q is
    still found (review F2: it had dropped silently to the model's q). Moved
    away from its pages, the JSON is refused, not read as a card without
    counter pages."""
    pj = _pinned_json(pinned_results, tmp_path, "GH200")
    monkeypatch.chdir(ROOT)
    rel = PAGES["GH200"].relative_to(ROOT)
    assert W.main([str(rel), "--params", str(pj), "--params-out", str(tmp_path / "w")]) == 0
    capsys.readouterr()
    written = tmp_path / "w" / "nvidia_gh200_480gb.wsc-params.json"
    doc = json.loads(written.read_text())
    assert doc["counter_pages_base"] == "the directory of this JSON"
    assert not any(Path(p).is_absolute() for p in doc["counter_pages"].values())
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    src = W.q_source(written)
    assert src.card is not None and sorted(src.card.pages) == [1, 2, 4, 16, 64]
    uuid = pinned_results["GH200"].card.geom.uuid
    assert src("shared", 2, 2, uuid) == (pytest.approx(pinned_results["GH200"].card.measured[
        ("shared", 2, 2, "w1")]), pytest.approx(1.039, abs=5e-4), "measured")
    moved = tmp_path / "a" / "b" / "c"
    moved.mkdir(parents=True)
    (moved / written.name).write_text(written.read_text())
    with pytest.raises(W.Refused, match="not there"):
        W.q_source(moved / written.name)


def test_a_parameter_the_loss_cannot_tell_from_its_bound_is_flagged(cards):
    """GH200 ws theta1_A stopped at 0.00125, inside the 1e-4 edge's reach of
    nothing, with a stage-1 SSE the same to nine decimals at 0 (review F8).
    It is flagged at bound 0 by its logit alone; at 0.004 (logit -5.5) only
    the card's loss can say so, and does; at 0.02, and in the fill view
    (0.136), it is not at a bound."""
    card = cards["GH200"]
    model = W.Model(card.geom)

    def theta_flags(prm, with_card: bool):
        return [f for f in W.param_flags(prm, card.geom, card if with_card else None, model)
                if f.startswith("theta1_A")]

    ws = pinned("GH200", "ws")
    assert theta_flags(ws, False)[0].startswith("theta1_A at bound 0 (0.0012; logit -6.7")
    assert theta_flags(ws, True)[0].startswith("theta1_A at bound 0")
    near = ws.replace(theta1_A=0.004)
    assert theta_flags(near, False) == []
    assert theta_flags(near, True)[0].startswith("theta1_A at bound 0 (0.004; its stage-1 SSE")
    assert theta_flags(ws.replace(theta1_A=0.02), True) == []
    assert theta_flags(pinned("GH200"), True) == []


def test_a_missing_ws_view_prints_na_not_the_ill_posed_mark(cards):
    """Parameters with the fill view only: every q_ws cell prints n/a, and
    '--' is left to mean ILL-POSED (review F9)."""
    res = W.analyse(cards["H100"], params={"fill": pinned("H100")})
    lines = W.card_lines(res, W.scores(res))

    def row(*key):
        return next(line for line in lines if line.split()[:4] == list(key)).split()
    assert row("shared", "1", "6", "w2")[5:7] == ["--", "n/a"]
    assert row("shared", "2", "2", "w2")[6] == "n/a"
    assert row("shared", "2", "2", "w2")[5] == "1.076"


#: s10.log section 3 (the research fit's w2 SHARED decomposition, fill view,
#: n=6), printed to three decimals: group, recovered in the first window (of
#: the cross-group re-reads there), recovered after it (of those), the
#: first-window partner loss, activation, q.
S10_ROWS = {
    ("GH200", 2): (3.000, 0.223, 0.266, 0.233, 1.734, 0.067, 0.008, 2.619),
    ("H100", 2): (3.000, 0.222, 0.266, 0.241, 1.734, 0.072, 0.014, 2.623),
    ("A100", 4): (2.000, 0.002, 0.086, 0.000, 0.914, 0.354, 0.026, 2.377),
}


def test_the_w2_decomposition_sums_to_q_and_is_s10s(pinned_results):
    """Every printed decomposition closes: group - recovered(first window) -
    recovered(later) + partner loss + activation = q, to 1e-9 (q is a fixed
    point, so the terms at its fbar return it); s10's rows to +-5e-4; and an
    ILL-POSED cell carries no terms."""
    keys = ("group", "recovered_w1", "x_w1", "recovered_later", "x_later", "partner_loss_w1",
            "activation", "q")
    for card in ("A100", "GH200", "H100"):
        rows = W.decomposition(pinned_results[card])
        assert len(rows) == 3 * 4, card
        for d in rows:
            if d["status"] == W.ILL_POSED:
                assert "q" not in d, (card, d)
                continue
            total = (d["group"] - d["recovered_w1"] - d["recovered_later"]
                     + d["partner_loss_w1"] + d["activation"])
            assert total == pytest.approx(d["q"], abs=1e-9), (card, d["G"], d["n"])
            assert d["group"] == DCR.group_reads(8, d["n"], d["G"])
        for (c, G), want in S10_ROWS.items():
            if c != card:
                continue
            d = next(x for x in rows if (x["G"], x["n"]) == (G, 6))
            assert [d[k] for k in keys] == pytest.approx(list(want), abs=5e-4), (card, G)
    gh = {(d["G"], d["n"]): d for d in W.decomposition(pinned_results["GH200"])}
    assert gh[(1, 6)]["status"] == W.ILL_POSED and "q" not in gh[(1, 6)]


# --------------------------------------------------------------------------
# The mix view (2026-09-27): w1 reads the fill distance, w2 the working set.
# --------------------------------------------------------------------------

NEW_GH200 = (ROOT / "results" / "published" / "2026-09-27-nvidia_gh200_480gb-session"
             / "results" / "2026-09-27-nvidia_gh200_480gb-r3-counters" / "lock1710")


def test_mix_is_fill_on_w1_and_ws_on_w2_with_one_parameter_set(cards):
    """Same parameters, three views: on w1 cells mix equals fill exactly, on w2
    cells it equals ws exactly, so it adds no parameter."""
    card = cards["GH200"]
    model = W.Model(card.geom)
    prm = pinned("GH200", "fill")
    cells = [c for c in W.page_cells(card) if c[2] >= 2]
    got = {v: dict(zip(cells, model.evaluate(prm.replace(view=v), cells), strict=True))
           for v in ("fill", "ws", "mix")}
    for c in cells:
        want = got["fill"][c] if c[3] == "w1" else got["ws"][c]
        assert got["mix"][c]["q"] == pytest.approx(want["q"], rel=1e-12), c
    assert W.REGISTERED_VIEW == "mix" and "mix" in W.VIEWS


def test_mix_on_the_0927_board_beats_fill_where_its_data_say_it_should():
    """On the 2026-09-27 GH200 (treads to 9), each view fitted on the board:
    mix is best on w1 and w2 at G >= 32, and moves the deep w2 cells toward
    what was measured (G=16 n=8 1.384: fill 1.245, mix 1.407 with stage 3's
    SHARED/NATIVE w2 law)."""
    res = W.analyse(W.load_card(NEW_GH200))
    rows = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in res.rows}

    def rms(sel, v):
        rel = [r[f"least_root_{v}"] / r["measured"] - 1 for r in res.rows
               if r["n"] >= 2 and sel(r) and r[f"least_root_{v}"] is not None]
        return float(np.sqrt(np.mean(np.square(rel))))

    w1 = lambda r: r["gemm"] == "w1"  # noqa: E731
    wide = lambda r: r["gemm"] == "w2" and r["arm"] != "private" and r["G"] >= 32  # noqa: E731
    assert rms(w1, "mix") < rms(w1, "fill") < rms(w1, "ws")
    assert rms(wide, "mix") < rms(wide, "ws") < rms(wide, "fill")
    assert rms(wide, "mix") < 0.5 * rms(wide, "fill")
    g16 = rows[("shared", 16, 8, "w2")]
    assert g16["least_root_fill"] == pytest.approx(1.245, abs=0.002)
    assert g16["least_root_mix"] == pytest.approx(1.407, abs=0.002)
    assert res.params("mix").view == "mix"


@pytest.mark.later_miss
def test_stage3_gives_shared_and_native_w2_their_own_a_law():
    """8x7B board A, MIX view: stage 3 fits (C_A2, beta_A2) on SHARED/NATIVE w2
    at G >= 32, n >= 2 with stages 1 and 2 held; those cells go 12.9 -> 3.7%
    rms and PRIVATE and w1 do not move (their events keep (C_A, beta_A))."""
    card = W.load_card(NEW_GH200)
    f = W.fit_view(card, W.MIX)
    p = f.params
    assert f.stage3.names == ("C_A2", "beta_A2") and f.stage3.cells == 32
    assert p.C_A2 == pytest.approx(86.18, abs=0.05)
    assert p.beta_A2 == pytest.approx(1.476, abs=0.005)
    m = W.Model(card.geom)

    def rms(prm, sel):
        cs = [c for c in sorted(card.measured) if sel(c)]
        r = m.q(prm, cs) / np.array([card.measured[c] for c in cs]) - 1
        return float(np.sqrt(np.mean(r ** 2)))

    wide = lambda c: c[3] == "w2" and c[0] != "private" and c[1] >= 32 and c[2] >= 2  # noqa: E731
    off = p.replace(C_A2=0.0)
    assert rms(off, wide) == pytest.approx(0.1293, abs=0.0005)
    assert rms(p, wide) == pytest.approx(0.0368, abs=0.0005)
    for sel in (lambda c: c[0] == "private", lambda c: c[3] == "w1"):
        assert rms(p, sel) == rms(off, sel)
    assert W.fit_view(card, W.FILL).stage3 is None, "stage 3 is the MIX view's only"




@pytest.mark.later_miss
def test_the_later_cross_group_reread_misses_at_g2_and_up():
    """8x7B board A, fitted on itself: with the later re-reads past the first
    window counted as misses, w2 SHARED at G=2 comes to within about 4% (the
    survival law's tail put n=8 at -14.8%); w1 is untouched; G=1 keeps the law."""
    card = W.load_card(NEW_GH200)
    assert W.LATER_MISS is True
    ev = W.cell_events(card.geom, "shared", 2, 8, "w2")
    assert ev.x_win1.all(), "no later cross-group event is left to the survival law"
    ev1 = W.cell_events(card.geom, "shared", 1, 8, "w2")
    assert (~ev1.x_win1).any(), "G=1 keeps its later events under the law"
    res = W.analyse(card)
    rows = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in res.rows}
    g2 = [rows[("shared", 2, n, "w2")] for n in range(2, 10)]
    rel = [r["least_root_mix"] / r["measured"] - 1 for r in g2]
    assert max(abs(x) for x in rel) < 0.05, rel
    with W.later_law(False):
        old = W.analyse(card)
    orow = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in old.rows}
    assert orow[("shared", 2, 8, "w2")]["least_root_mix"] / orow[("shared", 2, 8, "w2")][
        "measured"] - 1 < -0.10
