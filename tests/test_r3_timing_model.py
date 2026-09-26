"""The R3 timing model LOWM-1/2 (`scripts/r3_timing_model.py`), held to the
published pages the timing judge pinned it on (2026-09-26).

The model is NOT final (the script says so and prints what would falsify it),
so these tests do not assert that it is right. They assert that it is the
model the judge chose, computed from the pages the judge read, so that a later
edit which moves a pinned number is seen as moving it. Four things can make
its page wrong, and each has tests here:

    the SCHEDULE -- the launch order must be vLLM's pid mapping: its lead count
                    is `group_reads(8, n, G) x 8 x npn` (n x 8 x npn for
                    PRIVATE) and its grid is every counter cell's `grid_size`
                    on the GH200, H100 and A100 pages
    the BYTES    -- sigma x the schedule's reads is the page's counted bytes,
                    and the time rule has the limits its formula says (a floor
                    with infinite bandwidth, pure streaming with no floor), with
                    the fast path equal to the per-CTA sum
    the NUMBERS  -- the GH200, H100 and H200 fits, gates and registered
                    predictions at the judge's values and tolerances, and the
                    gate rules themselves on made-up fits at their edges
    the FOOTING  -- mixed cards (by name or device), locked pages at different
                    clocks or beside unlocked ones, two duties, counted bytes
                    from another card, counter pages of two cards at any Gs,
                    a G with no counter page, a card with no counter page
                    whose occupancy nothing gives (a borrow from another
                    architecture lends bytes only), and a tile the constants
                    do not describe are refused, never fitted

The PTX count (8 mma.sync, 8 ldmatrix.x4, 3 cp.async.cg per thread in the main
loop of all four published Hopper census kernels, 45056 B per CTA k-step, 352
cycles at 128 B per clock) is arithmetic on committed files; that shared memory
sets the floor is an INTERPRETATION and nothing here tests it.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "r3_timing_model.py"


def _load():
    spec = importlib.util.spec_from_file_location("r3_timing_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves annotations through it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load()
DCR = M.DCR

from moe.bench import exit_codes  # noqa: E402

PUB = ROOT / "results" / "published"
GH200 = PUB / "2026-09-25-nvidia_gh200_480gb-session"
GH200_COUNTERS = GH200 / "results" / "2026-09-25-nvidia_gh200_480gb-r3-counters"
H100 = PUB / "2026-09-25-nvidia_h100_80gb_hbm3-session"
H100_COUNTERS = H100 / "results" / "2026-09-25-nvidia_h100_80gb_hbm3-r3-counters"
A100_COUNTERS = (PUB / "2026-09-25-nvidia_a100_sxm4_40gb-r3-counters" / "results"
                 / "2026-09-25-nvidia_a100_sxm4_40gb-r3-counters")
H200_SESSIONS = (PUB / "2026-09-23-nvidia_h200-session5", PUB / "2026-09-24-nvidia_h200-session6")

#: The pages the judge fitted: each card's VALID R3 pages at the 1710 MHz lock.
GH200_RUNS = ("d9f1f37c", "df37ea07", "01c08abd", "1b285de2", "6ff34777")
H100_RUNS = ("ff84f0aa", "0708f122", "48bc690e", "22da27e7", "ceb347ad")
GS = (1, 2, 4, 16, 64)

pytestmark = pytest.mark.skipif(
    not (GH200_COUNTERS.exists() and H100_COUNTERS.exists()),
    reason="the 2026-09-25 GH200 and H100 sessions are not in this tree")


def run_dir(session: Path, run: str) -> Path:
    hits = sorted(session.glob(f"results/gaps-*/private_weight_reference/*{run}"))
    assert len(hits) == 1, (run, hits)
    return hits[0]


def build(*argv):
    return M.build(M.build_parser().parse_args([str(a) for a in argv]))


def pct(v: float) -> float:
    return 100.0 * v


@pytest.fixture(scope="module")
def gh200():
    return build(*(run_dir(GH200, r) for r in GH200_RUNS), "--counters", GH200_COUNTERS)


@pytest.fixture(scope="module")
def h100():
    return build(*(run_dir(H100, r) for r in H100_RUNS), "--counters", H100_COUNTERS)


@pytest.fixture(scope="module")
def h200_groupmodel():
    return build(*H200_SESSIONS, "--bytes", "groupmodel")


@pytest.fixture(scope="module")
def h200_borrowed():
    return build(*H200_SESSIONS, "--bytes", f"borrowed:{GH200_COUNTERS}")


# --------------------------------------------------------------------------
# 1. The schedule: vLLM's pid mapping, against the group model's closed form
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", M.ARMS)
@pytest.mark.parametrize("G", (1, 2, 3, 4, 8, 16, 32, 64))
def test_the_launch_order_leads_are_the_group_models_reads_and_every_live_cta_runs(arm, G):
    """A lead is the first live CTA of a (group, slab) pair: the one that pulls
    the slab from DRAM. SHARED and NATIVE fetch each expert's slab once per
    GROUP_SIZE_M group its tiles fall in, which is `group_reads` x E per
    N-tile; PRIVATE's every M-tile has its own copy. At every G (the four the
    study measured and four it did not) and n = 1 to 8, on both GEMMs."""
    declared = 8 if arm == "native" else 72
    for gemm, geo in M.GEOMETRY.items():
        for n in range(1, 9):
            r, leads = M.schedule(arm, declared, G, n, gemm)
            want = (n if arm == "private" else DCR.group_reads(8, n, G)) * 8 * geo.npn
            assert leads == pytest.approx(want, abs=1e-9), (arm, G, n, gemm)
            assert r.size == 8 * n * geo.npn, (arm, G, n, gemm)


# --------------------------------------------------------------------------
# 2. The grid: R x npn is what ncu read off every launch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("counters", (GH200_COUNTERS, H100_COUNTERS, A100_COUNTERS),
                         ids=("gh200", "h100", "a100"))
def test_grid_rows_times_n_tiles_is_every_counter_cells_grid_size(counters):
    pages = sorted(counters.glob("r3c-g*.json"))
    assert len(pages) == 5
    checked = 0
    for f in pages:
        for cell in json.loads(f.read_text())["cells"]:
            for gemm, geo in M.GEOMETRY.items():
                got = M.grid_rows(int(cell["declared"]), int(cell["n"])) * geo.npn
                assert got == int(cell["per_gemm"][gemm]["grid_size"]), (f.name, cell["arm"],
                                                                         cell["n"], gemm)
                checked += 1
    assert checked == 5 * 15 * 2


# --------------------------------------------------------------------------
# 3. The bytes: sigma scales the schedule onto the page's counted reads
# --------------------------------------------------------------------------

def test_sigma_times_the_schedules_reads_is_the_pages_counted_bytes(gh200):
    """Read off the page here, not through the script's loader."""
    raw = {}
    for G in GS:
        page = json.loads((GH200_COUNTERS / f"r3c-g{G}.json").read_text())
        for c in page["cells"]:
            raw[(c["arm"], G, int(c["n"]))] = {g: float(c["per_gemm"][g]["dram_bytes_read"])
                                               for g in M.GEMMS}
    ctx, counted = gh200["ctx"], [c for c in gh200["cells"] if c.source == "COUNTED"]
    assert len(counted) == 3 * 5 * 5
    for c in counted:
        for g in M.GEMMS:
            r, _ = M.schedule(c.arm, c.declared, c.G, c.n, g)
            want = raw[(c.arm, c.G, c.n)][g]
            assert c.sigma[g] * r.sum() == pytest.approx(want, rel=1e-9), (c.key, g)
            w = M.window_width(0.5, ctx.sms, ctx.occupancy[g])
            assert M.window(c.arm, c.declared, c.G, c.n, g, w).reads == pytest.approx(
                r.sum(), rel=1e-12)


def test_n5_bytes_are_interpolated_by_the_specs_rule_and_labelled(gh200):
    """G=1 and PRIVATE: the mean of n=4 and n=6. Otherwise the group model
    times the median counted/model ratio at n in {2, 3, 4, 6}."""
    cells = {c.key: c for c in gh200["cells"]}
    for key in ("shared/G1/n5", "private/G16/n5", "shared/G4/n5", "native/G64/n5"):
        c = cells[key]
        assert c.source == "INTERPOLATED" and not c.fit
        n4, n6 = cells[key.replace("n5", "n4")], cells[key.replace("n5", "n6")]
        for g in M.GEMMS:
            if c.arm == "private" or c.G == 1:
                want = 0.5 * (n4.reads[g] + n6.reads[g])
            else:
                ratio = np.median([cells[key.replace("n5", f"n{m}")].reads[g]
                                   / M.group_model_bytes(c.arm, c.G, m, g) for m in (2, 3, 4, 6)])
                want = ratio * M.group_model_bytes(c.arm, c.G, 5, g)
            assert c.reads[g] == pytest.approx(want, rel=1e-12), (key, g)


# --------------------------------------------------------------------------
# 4. The limits of the time rule
# --------------------------------------------------------------------------

def test_with_infinite_bandwidth_every_cta_costs_the_quantised_floor():
    sms, c_ns = 132, 206.9
    for arm, G, n, gemm in (("shared", 1, 3, "w1"), ("native", 4, 6, "w2"),
                            ("private", 64, 2, "w1"), ("shared", 2, 5, "w2")):
        D = 8 if arm == "native" else 72
        win = M.window(arm, D, G, n, gemm, 330)
        N, geo = win.live, M.GEOMETRY[gemm]
        q = np.ceil(N / sms) / (N / sms)
        want = q * geo.ksteps * c_ns * 1e-6 * N / sms
        assert M.gemm_ms(win, gemm, 1.0, c_ns, 1e15, sms) == pytest.approx(want, rel=1e-12)


def test_with_no_floor_private_at_g1_streams_its_bytes_exactly():
    """Every PRIVATE CTA leads, so the window mean is each CTA's own bytes and
    the call is sum(sigma r_i + o) / bw with nothing averaged away."""
    sigma, bw = 1.0037, 3600.0
    for n in (1, 2, 6):
        for gemm in M.GEMMS:
            r, leads = M.schedule("private", 72, 1, n, gemm)
            assert leads == r.size
            win = M.window("private", 72, 1, n, gemm, 264)
            want = float(np.sum(sigma * r + M.OUT_TILE)) / (bw * 1e6)
            assert M.gemm_ms(win, gemm, sigma, 0.0, bw, 132) == pytest.approx(want, rel=1e-12)


def test_the_fast_sum_is_the_per_cta_maximum_summed():
    """The fit sums max(floor, b_bar / bw) by sorting the window means once;
    that is a reordering of the spec's per-CTA sum, not an approximation."""
    for arm, G, n, gemm, sigma, c_ns, bw in (("shared", 2, 3, "w1", 0.95, 206.9, 3606.5),
                                             ("shared", 1, 4, "w2", 0.62, 206.9, 3606.5),
                                             ("native", 64, 6, "w2", 1.99, 180.0, 2500.0),
                                             ("private", 16, 5, "w1", 1.01, 250.0, 4000.0)):
        D = 8 if arm == "native" else 72
        for w in (1, 264, 330, 660):
            fast = M.gemm_ms(M.window(arm, D, G, n, gemm, w), gemm, sigma, c_ns, bw, 132)
            slow = float(M.cta_ms(arm, D, G, n, gemm, sigma, c_ns, bw, 132, w).sum())
            assert fast == pytest.approx(slow, rel=1e-12), (arm, G, n, gemm, w)


def test_the_constants_are_the_specs():
    w1, w2 = M.GEOMETRY["w1"], M.GEOMETRY["w2"]
    assert (w1.K, w1.N, w1.npn, w1.ksteps) == (4096, 28672, 448, 64)
    assert (w2.K, w2.N, w2.npn, w2.ksteps) == (14336, 4096, 64, 224)
    assert (w1.slab, w1.arow, w2.slab, w2.arow) == (64 * 4096 * 2, 32 * 4096 * 2,
                                                   64 * 14336 * 2, 32 * 14336 * 2)
    assert M.OUT_TILE == 4096
    assert M.B_OTHER == 25_165_824


# --------------------------------------------------------------------------
# 5. The PTX
# --------------------------------------------------------------------------

def test_the_hopper_census_ptx_main_loop_moves_45056_bytes_per_cta_kstep():
    ptx = sorted(GH200.glob("session/census.profiles/**/fused_moe_kernel.ptx")) + sorted(
        H100.glob("session/census.profiles/**/fused_moe_kernel.ptx"))
    assert len(ptx) == 4
    for p in ptx:
        counts = M.ptx_main_loop(p.read_text())
        assert (counts["mma_sync"], counts["ldmatrix"], counts["cp_async_cg"]) == (8, 8, 3), p
        assert counts["threads"] == 256
        assert M.smem_bytes_per_kstep(counts) == 45056
        assert M.smem_bytes_per_kstep(counts) / M.SMEM_BYTES_PER_CLK == 352


# --------------------------------------------------------------------------
# 6. The GH200, the judge's pins
# --------------------------------------------------------------------------

def test_gh200_parameters(gh200):
    p, ctx = gh200["main"].params, gh200["ctx"]
    assert ctx.locked and ctx.clock_mhz == 1710.0 and ctx.sms == 132
    assert ctx.occupancy == {"w1": 5, "w2": 4}
    assert p["T0"] == pytest.approx(0.05717, abs=0.002)
    assert p["c"] == pytest.approx(206.91, abs=1.0)
    assert p["c"] * ctx.clock_mhz * 1e-3 == pytest.approx(353.8, abs=1.0)
    assert p["bw"] == pytest.approx(3606.5, abs=30)
    assert p["s_small"] == pytest.approx(0.00398, abs=0.002)
    assert p["s_block"] == pytest.approx(-0.02048, abs=0.003)
    assert all(gh200["main"].identified) and not any(gh200["main"].at_bound)


def test_gh200_scores(gh200):
    sc = gh200["score"]
    assert pct(sc["rms"]) == pytest.approx(0.43, abs=0.006) and pct(sc["rms"]) <= 0.50
    assert sc["worst_cell"] == "shared/G64/n2"
    assert pct(sc["worst"]) == pytest.approx(-0.88, abs=0.006) and abs(pct(sc["worst"])) <= 1.0
    for G, want in zip(GS, (0.44, 0.30, 0.38, 0.45, 0.55), strict=True):
        assert pct(sc["per_G"][G]) == pytest.approx(want, abs=0.006), G
    assert pct(sc["scored_only_rms"]) == pytest.approx(0.55, abs=0.006)
    lm = M.logo_mean(gh200["logo"])
    assert pct(lm) == pytest.approx(0.48, abs=0.006) and pct(lm) <= 0.55
    assert all(gh200["gates"][t]["pass"] for t in ("T1", "T2", "T3", "T4"))


def test_gh200_shape_and_rates(gh200):
    F = gh200["F"]
    model = F["F2_G2_shared_increments"]["model"]
    for got, want in zip(model, (0.316, 0.693, 0.407, 0.695, 0.405), strict=True):
        assert got == pytest.approx(want, abs=0.01)
    for G in (4, 16, 64):
        assert F["F3_shared_slope_2_6"][G]["model"] == pytest.approx(0.5499, abs=0.0005), G
    t2 = gh200["gates"]["T2"]["bw_over_R_P_minus_1"]
    assert pct(t2) == pytest.approx(1.38, abs=0.05)
    assert gh200["rho_star"]["w1"] == pytest.approx(0.690, abs=0.001)


def test_gh200_k_w_1_is_worse_than_the_half_window(gh200):
    ref = gh200["reference_score"]
    assert gh200["reference"].k_w == 1.0
    assert pct(ref["rms"]) == pytest.approx(0.522, abs=0.006)
    assert ref["rms"] > gh200["score"]["rms"]


def test_gh200_registered_predictions(gh200):
    """The judge's P1 to P7 for the next GH200 run, from this fit."""
    P = gh200["predictions"]
    for eta, want in ((1.0, (0.6654, 0.6259, 0.5872)), (0.35, (0.5878, 0.5754, 0.5627))):
        for f, w in zip(M.P1_CLOCKS, want, strict=True):
            assert P["P1"][f"G4/f{f:.0f}/eta{eta}"]["slope_2_6"] == pytest.approx(w, abs=5e-4)
    for eta, want in ((1.0, (0.550, 0.731, 0.600, 0.733, 0.598)),
                      (0.35, (0.393, 0.703, 0.473, 0.704, 0.471))):
        got = P["P1"][f"G2/f1410/eta{eta}"]["inc"]
        assert got == pytest.approx(want, abs=1.5e-3), eta
    for G in M.P2_GS:
        assert P["P2"][G]["step_1_2"] == pytest.approx(0.315, abs=1.5e-3)
        assert P["P2"][G]["slope_2_6"] == pytest.approx(0.5499, abs=5e-4)
        assert P["P2"][G]["band"] == (0.538, 0.556)
        # The band holds the increments from n=2 to 6; the model's own sit in it.
        lo, hi = P["P2"][G]["band"]
        assert all(lo <= v <= hi for v in P["P2"][G]["inc"][1:5]), G
    # PRIVATE at G=8, sigma from the nearest measured G (a tie between 4 and
    # 16, which takes 4): the judge's reference printed this ladder.
    pv = P["P2"][8]["private"]
    assert pv["n"] == [1, 2, 3, 4, 5, 6]
    assert pv["T"] == pytest.approx((0.8547, 1.6517, 2.4534, 3.2544, 4.0562, 4.8590), abs=5e-4)
    assert pv["slope_2_6"] == pytest.approx(0.8018, abs=5e-4)
    assert "private" not in P["P2"][32]
    assert P["P3"]["G2"]["T"][6:10] == pytest.approx((4.0768, 4.4830, 5.1374, 5.5365), abs=5e-4)
    assert min(P["P3"]["odd_n_excess"].values()) >= M.P3_MIN_EXCESS_MS
    assert P["P4"]["inc_8_9"] == pytest.approx(0.504, abs=1.5e-3)
    assert P["P4"]["q_w2"][8] == pytest.approx(1.031, abs=5e-4)
    assert P["P4"]["q_w2"][9] == pytest.approx(1.003, abs=5e-4)
    assert P["P5"]["G3"]["inc"] == pytest.approx(
        (0.342, 0.523, 0.550, 0.567, 0.533, 0.563, 0.566), abs=1.5e-3)
    assert P["P5"]["slab_fraction_max_G3"] == pytest.approx(2 / 3)
    assert P["P5"]["flat"]
    assert P["P6"]["ptx_smem_cycles"] == [352.0]
    assert P["P7"]["locked_fit"] and P["P7"]["pages"] == []   # no unlocked page among the inputs
    assert P["P8"] == {}


def test_a_prediction_no_measured_g_brackets_prints_n_a_and_the_fit_still_runs(tmp_path):
    """sigma at an unmeasured G is imputed from the counter pages either side.
    With the GH200's G=16 and G=64 pages alone, G=2, 3, 4 and 8 have none, so
    P1 to P5 print n/a with the reason; the fit, its gates and P6 still print."""
    for G in (16, 64):
        shutil.copy(GH200_COUNTERS / f"r3c-g{G}.json", tmp_path / f"r3c-g{G}.json")
    R = build(run_dir(GH200, "1b285de2"), run_dir(GH200, "6ff34777"), "--counters", tmp_path)
    P = R["predictions"]
    assert sorted(P["unavailable"]) == ["P1", "P2", "P3", "P4", "P5"]
    assert all(P[t] is None for t in ("P1", "P2", "P3", "P4", "P5"))
    assert R["main"].fitted_gs == (16, 64) and P["P6"]["cycles_per_cta_kstep"] > 0
    assert "P1 n/a: no measured G brackets G=4" in "\n".join(M.lines_of(R))


def test_the_imputation_rules_on_a_made_up_table():
    """Every branch of the sigma imputation, on a table whose values name
    their own cell (100 G + n), so a wrong pick reads as the wrong number.
    PRIVATE is the judge's `predict_cell`: the nearest measured G by log2 (a
    tie takes the smaller), its own n, the mean of n=4 and n=6 at n=5, n=6's
    value beyond it. SHARED is the median over the Gs either side, n=1 apart.
    Review, 2026-09-26: the PRIVATE branch past n=6 is on no printed ladder,
    so only this test holds it."""
    tab = {(arm, "w1", G): {n: 100.0 * G + n for n in M.LADDER}
           for arm in ("shared", "private") for G in (1, 4, 16)}
    imp = M.imputed_sigma
    # PRIVATE: G=8 ties 4 and 16 (log2 3 against 2 and 4) and takes 4; G=2
    # ties 1 and 4 and takes 1; G=12 is nearer 16.
    assert imp(tab, "private", "w1", 8, 3) == 403.0
    assert imp(tab, "private", "w1", 2, 2) == 102.0
    assert imp(tab, "private", "w1", 12, 1) == 1601.0
    assert imp(tab, "private", "w1", 8, 5) == 405.0          # (404 + 406) / 2
    assert imp(tab, "private", "w1", 8, 7) == 406.0          # capped at n=6
    assert imp(tab, "private", "w1", 8, 10) == 406.0
    # SHARED at G=8: n=1 the median of 401 and 1601; n >= 2 the median of
    # 402 403 404 406 1602 1603 1604 1606, whatever n.
    assert imp(tab, "shared", "w1", 8, 1) == 1001.0
    assert imp(tab, "shared", "w1", 8, 4) == imp(tab, "shared", "w1", 8, 9) == 1004.0
    # At a measured G the SHARED median is that G's own.
    assert imp(tab, "shared", "w1", 4, 2) == 403.5
    with pytest.raises(M.Refused, match="no measured G brackets G=64"):
        imp(tab, "shared", "w1", 64, 2)
    assert imp(None, "private", "w1", 8, 7) == 1.0           # GROUPMODEL: sigma = 1


def test_the_whole_gh200_session_dir_fits_the_same_five_valid_pages(gh200):
    """Its INVALID (unlocked) and UNSCORED pages are listed and not fitted. The
    two unlocked INVALID pages at G >= 2 are read for P7 alone: c refitted with
    everything else held, and a fixed 353.8 cycles per CTA k-step reads them
    as an in-kernel clock of 1782 and 1787 MHz against NVML's 1935."""
    R = build(GH200)
    assert sorted(p.run for p in R["use"]) == sorted(GH200_RUNS)
    excluded = {p.run: p.label for p in R["all_pages"] if p not in R["use"]}
    assert excluded == {"78c2414e": "INVALID", "308f0841": "INVALID", "da469da0": "INVALID",
                        "257313aa": "UNSCORED"}
    assert R["main"].x == pytest.approx(gh200["main"].x, rel=1e-9)
    p7 = {d["run"]: d for d in R["predictions"]["P7"]["pages"]}
    assert sorted(p7) == ["308f0841", "da469da0"]
    assert {r: d["in_kernel_mhz"] for r, d in p7.items()} == pytest.approx(
        {"308f0841": 1782.0, "da469da0": 1787.0}, abs=2.0)
    assert {r: d["nvml_mhz"] for r, d in p7.items()} == {"308f0841": 1935.0, "da469da0": 1935.0}
    # The in-kernel clock is the judge's INTERPRETATION, and every P7 line says so.
    lines = [ln for ln in M.lines_of(R) if ln.lstrip().startswith("P7 unlocked page")]
    assert len(lines) == 2
    assert all("INTERPRETATION: holds only if the floor is a fixed cycle count" in ln
               for ln in lines)


# --------------------------------------------------------------------------
# 7. The H100: the known failure at G=1, asserted
# --------------------------------------------------------------------------

def test_h100_all_g_fit_fails_at_g1_and_says_so(h100):
    p, sc, gt = h100["main"].params, h100["score"], h100["gates"]
    assert p["c"] == pytest.approx(206.98, abs=1.0)
    assert p["bw"] == pytest.approx(2971.9, abs=30)
    assert pct(sc["rms"]) == pytest.approx(1.51, abs=0.006)
    assert pct(sc["per_G"][1]) == pytest.approx(2.96, abs=0.006)
    assert not gt["T3"]["pass"] and gt["T3"]["fails_at_G"] == [1]
    # Its worst cell is G=1's too, far outside T4; T1 and T2 pass (bw is
    # PRIVATE's own rate here: the failure is at G=1, not in the bytes' rate).
    assert not gt["T4"]["pass"] and gt["T4"]["worst_cell"] == "shared/G1/n6"
    assert pct(gt["T4"]["worst"]) == pytest.approx(-6.27, abs=0.006)
    assert gt["T1"]["pass"]
    assert gt["T2"]["pass"] and pct(gt["T2"]["bw_over_R_P_minus_1"]) == pytest.approx(0.22,
                                                                                    abs=0.05)
    f1 = h100["F"]["F1_G1_shared_slope_2_6"]
    assert f1["measured"] == pytest.approx(0.9071, abs=5e-4)
    assert f1["model"] == pytest.approx(0.8207, abs=5e-4)
    assert -0.12 <= f1["model"] / f1["measured"] - 1 <= -0.07


def test_h100_fit_without_g1_is_added_and_predicts_g1_badly(h100):
    d = h100["t3"][1]
    f, sc = d["fit"], d["score"]
    assert f.fitted_gs == (2, 4, 16, 64)
    p = f.params
    assert p["T0"] == pytest.approx(0.05119, abs=0.002)
    assert p["c"] == pytest.approx(205.95, abs=1.0)
    assert p["bw"] == pytest.approx(3018.2, abs=30)
    assert p["s_block"] == pytest.approx(-0.01738, abs=0.003)
    for G, want in zip(GS, (3.65, 0.48, 0.32, 0.43, 0.57), strict=True):
        assert pct(sc["per_G"][G]) == pytest.approx(want, abs=0.006), G
    p8 = {r["n"]: r["w2_needed_Ww2"] for r in h100["predictions"]["P8"][1]}
    assert p8 == pytest.approx({2: 1.32, 3: 2.35, 4: 3.03, 5: 3.97, 6: 4.84}, abs=0.006)


def test_p8_solves_the_w2_bytes_or_says_why_it_cannot(h100):
    """The bisection on a real cell (the H100's SHARED G=1 n=3, under the fit
    without G=1): the multiple it returns makes the model meet the measured
    time. A cell 10x slower than measured is beyond 3x the bytes, and one 10x
    faster is floor-bound; each comes back None with its reason, never the
    range's endpoint (review, 2026-09-26)."""
    x, ctx = h100["t3"][1]["fit"].x, h100["ctx"]
    cell = next(c for c in h100["cells"] if c.key == "shared/G1/n3")
    m, why = M.w2_needed(x, cell, ctx, 0.5)
    assert why == "" and M.P8_RANGE[0] < m < M.P8_RANGE[1]
    assert M.call_ms(x, cell, ctx, 0.5, w2_scale=m) == pytest.approx(cell.ms, rel=1e-9)
    assert M.w2_needed(x, dataclasses.replace(cell, ms=10 * cell.ms), ctx, 0.5) == (
        None, "beyond 3.0x the w2 bytes")
    m, why = M.w2_needed(x, dataclasses.replace(cell, ms=0.1 * cell.ms), ctx, 0.5)
    assert m is None and why.startswith("floor-bound")


# --------------------------------------------------------------------------
# 8. The H200: no bytes of its own
# --------------------------------------------------------------------------

def test_h200_with_group_model_bytes_fails_t2_by_the_missing_bytes(h200_groupmodel):
    R = h200_groupmodel
    assert R["ctx"].occupancy_source.startswith("ASSUMED")
    assert R["ctx"].occupancy == {"w1": 5, "w2": 4}
    assert "capability 9.0" in R["ctx"].occupancy_source
    assert all(c.source == "GROUPMODEL" for c in R["cells"])
    assert pct(R["score"]["rms"]) == pytest.approx(2.90, abs=0.006)
    t2 = R["gates"]["T2"]
    assert not t2["pass"] and pct(t2["bw_over_R_P_minus_1"]) == pytest.approx(7.47, abs=0.05)


def test_h200_with_group_model_bytes_trips_t3_at_g1_and_t4_at_private_g64(h200_groupmodel):
    """The judge's H200 own-card numbers: G=1 is out of form (4.63% against a
    median of 1.44% over the other Gs, beyond 3x; the mean, 1.90%, would not
    trip it) and PRIVATE G=64 is the worst cell at -12.4%."""
    gt = h200_groupmodel["gates"]
    assert gt["T1"]["pass"]
    assert not gt["T3"]["pass"] and gt["T3"]["fails_at_G"] == [1]
    assert pct(h200_groupmodel["score"]["per_G"][1]) == pytest.approx(4.63, abs=0.006)
    assert not gt["T4"]["pass"] and gt["T4"]["worst_cell"] == "private/G64/n6"
    assert pct(gt["T4"]["worst"]) == pytest.approx(-12.40, abs=0.006)


def test_p8_with_no_bracketing_w2_count_prints_n_a_not_the_range_end(h200_groupmodel):
    """T3 trips at G=1 on the H200 with group-model bytes, and the fit without
    G=1 overshoots every SHARED G=1 cell even at 0.2x its w2 bytes. The needed
    bytes are then no number at all (None, printed n/a), with the reason."""
    rows = h200_groupmodel["predictions"]["P8"][1]
    assert [r["n"] for r in rows] == [2, 3, 4, 5, 6]
    for r in rows:
        assert r["w2_needed_Ww2"] is None and r["note"].startswith("floor-bound"), r
        assert r["w2_have_Ww2"] > 0
    p8 = [ln for ln in M.lines_of(h200_groupmodel) if ln.lstrip().startswith("P8 SHARED")]
    assert len(p8) == 1 and p8[0].count("-> n/a (floor-bound") == 5


def test_h200_with_borrowed_gh200_bytes_passes_t2_labelled_as_a_borrow(h200_borrowed):
    R = h200_borrowed
    assert {c.source for c in R["cells"]} == {"BORROWED-nvidia_gh200_480gb/COUNTED",
                                              "BORROWED-nvidia_gh200_480gb/INTERPOLATED"}
    assert not R["ctx"].locked and R["ctx"].clock_mhz == 1965.0
    # The GH200 is the H200's architecture (9.0), so it lends its occupancy too.
    assert R["ctx"].occupancy == {"w1": 5, "w2": 4}
    assert "same architecture (capability 9.0" in R["ctx"].occupancy_source
    assert R["main"].params["c"] == pytest.approx(196.43, abs=1.0)
    assert pct(R["score"]["rms"]) == pytest.approx(0.45, abs=0.006)
    t2 = R["gates"]["T2"]
    assert t2["pass"] and pct(t2["bw_over_R_P_minus_1"]) == pytest.approx(1.63, abs=0.05)


def test_a_borrow_from_another_architecture_lends_bytes_not_occupancy():
    """The A100 (capability 8.0) records 4 / 4 CTAs per SM; the H200's timed
    kernel targets sm_90a. Borrowing the A100's bytes must leave the H200's
    occupancy, and so its windows (330 w1, 264 w2 CTAs at k_w 1/2), at the
    Hopper 5 / 4, and say why."""
    R = build(*H200_SESSIONS, "--bytes", f"borrowed:{A100_COUNTERS}")
    ctx = R["ctx"]
    assert ctx.occupancy == {"w1": 5, "w2": 4}
    assert "nvidia_a100_sxm4_40gb is capability 8.0, another architecture" in ctx.occupancy_source
    assert [M.window_width(0.5, ctx.sms, ctx.occupancy[g]) for g in M.GEMMS] == [330, 264]
    assert {c.source.split("/")[0] for c in R["cells"]} == {"BORROWED-nvidia_a100_sxm4_40gb"}


def _h200_page_without_counters(tmp_path, target: str | None) -> Path:
    """One VALID H200 page, copied under tmp_path beside the H200's ruler (its
    SM count), with its kernel's PTX `.target` set to `target` (None: no PTX)."""
    src = next(p.path for p in M.discover_timed([H200_SESSIONS[0]]) if p.label == "VALID")
    (tmp_path / "calibration").mkdir()
    shutil.copy(M.PTF.find_ruler(src), tmp_path / "calibration")
    dst = tmp_path / src.name
    dst.mkdir()
    for name in ("report.json", "cells.csv", "DEVICE"):
        shutil.copy(src / name, dst / name)
    if target is not None:
        ptx = sorted(src.rglob("fused_moe_kernel.ptx"))[0]
        text = ptx.read_text()
        assert ".target sm_90a" in text
        (dst / "triton-cache" / "k").mkdir(parents=True)
        (dst / "triton-cache" / "k" / "fused_moe_kernel.ptx").write_text(
            text.replace(".target sm_90a", f".target {target}", 1))
    return dst


def test_the_timed_pages_ptx_gives_a_counterless_cards_architecture(tmp_path):
    assert M.timed_capability(M.discover_timed([H200_SESSIONS[0]])) == "9.0"
    assert M.timed_capability(M.discover_timed([run_dir(GH200, "d9f1f37c")])) == "9.0"
    page = _h200_page_without_counters(tmp_path, "sm_80")
    assert M.timed_capability(M.discover_timed([page])) == "8.0"


@pytest.mark.parametrize("target, why", (("sm_80", "capability 8.0"),
                                         (None, "no fused_moe_kernel.ptx")))
def test_a_counterless_card_whose_occupancy_nothing_gives_is_refused(tmp_path, target, why):
    """Not Hopper and no same-architecture borrow, or no architecture at all:
    the window cannot be sized, so no fit."""
    page = _h200_page_without_counters(tmp_path, target)
    with pytest.raises(M.Refused, match=why):
        build(page, "--bytes", "groupmodel")


# --------------------------------------------------------------------------
# 9. The byte scales sigma
# --------------------------------------------------------------------------

def test_gh200_sigma_pins(gh200):
    """SHARED w2 at G=1 reads about two thirds of the launch-order model's
    bytes (a shared slab outlives its group there); SHARED w1 at G >= 2 reads
    what the model says."""
    cells = {c.key: c for c in gh200["cells"]}
    got = [cells[f"shared/G1/n{n}"].sigma["w2"] for n in (2, 3, 4, 6)]
    assert got == pytest.approx((0.670, 0.655, 0.625, 0.632), abs=1.5e-3)
    for G in (2, 4, 16, 64):
        for n in M.LADDER:
            assert cells[f"shared/G{G}/n{n}"].sigma["w1"] == pytest.approx(1.0, abs=0.03), (G, n)


# --------------------------------------------------------------------------
# The footing: what the tool refuses
# --------------------------------------------------------------------------

def test_locked_pages_at_different_clocks_are_refused():
    """The H100 session holds VALID G=1 pages locked at 1710, 1800 and 1980."""
    with pytest.raises(M.Refused, match="cell clocks differ"):
        build(H100)


def test_pages_of_two_cards_are_refused():
    with pytest.raises(M.Refused, match="more than one card"):
        build(run_dir(GH200, "d9f1f37c"), run_dir(H100, "0708f122"), "--counters",
              GH200_COUNTERS)


def test_counted_bytes_from_another_card_are_refused():
    with pytest.raises(M.Refused, match="counted bytes come from the card's own pages"):
        build(*(run_dir(GH200, r) for r in GH200_RUNS), "--counters", H100_COUNTERS)


def test_a_tile_the_constants_do_not_describe_is_refused(tmp_path):
    src = run_dir(GH200, "df37ea07")
    dst = tmp_path / src.name
    dst.mkdir()
    for name in ("report.json", "cells.csv", "DEVICE"):
        shutil.copy(src / name, dst / name)
    rep = json.loads((dst / "report.json").read_text())
    rep["pinned"]["BLOCK_SIZE_N"] = 128
    (dst / "report.json").write_text(json.dumps(rep))
    with pytest.raises(M.Refused, match="BLOCK_N"):
        build(dst, "--counters", GH200_COUNTERS)


def _copy_page(run: str, tmp_path: Path) -> Path:
    """A GH200 run dir's report, cells and DEVICE under tmp_path: still VALID
    (the label is its stored gates), free to edit."""
    src = run_dir(GH200, run)
    dst = tmp_path / src.name
    dst.mkdir()
    for name in ("report.json", "cells.csv", "DEVICE"):
        shutil.copy(src / name, dst / name)
    return dst


def _edit_report(page: Path, edit) -> None:
    rep = json.loads((page / "report.json").read_text())
    edit(rep)
    (page / "report.json").write_text(json.dumps(rep))


def test_pages_at_two_duties_are_refused(tmp_path):
    a, b = _copy_page("d9f1f37c", tmp_path), _copy_page("df37ea07", tmp_path)
    _edit_report(b, lambda rep: rep.update(duty=0.5))
    with pytest.raises(M.Refused, match="2 duties"):
        build(a, b, "--counters", GH200_COUNTERS)


def test_locked_and_unlocked_pages_together_are_refused(tmp_path):
    """One cell off the lock makes a page unlocked; a locked page beside it
    would put two clocks in one fit that has no clock term."""
    a, b = _copy_page("d9f1f37c", tmp_path), _copy_page("df37ea07", tmp_path)
    _edit_report(b, lambda rep: rep["treads_table"][0].update(sm_clock_load_mhz=1935.0))
    with pytest.raises(M.Refused, match="cell clocks differ"):
        build(a, b, "--counters", GH200_COUNTERS)


def test_a_g_with_no_counter_page_is_refused(tmp_path):
    for G in (1, 2, 4, 16):
        shutil.copy(GH200_COUNTERS / f"r3c-g{G}.json", tmp_path)
    with pytest.raises(M.Refused, match=r"no counter page at G=\[64\]"):
        build(*(run_dir(GH200, r) for r in GH200_RUNS), "--counters", tmp_path)


def test_two_devices_under_one_card_name_are_refused(tmp_path):
    b = _copy_page("df37ea07", tmp_path)
    (b / "DEVICE").write_text("00000000-0000-0000-0000-000000000000\n")
    with pytest.raises(M.Refused, match="more than one card"):
        build(run_dir(GH200, "d9f1f37c"), b, "--counters", GH200_COUNTERS)


def _two_card_counters(tmp_path: Path) -> tuple[Path, Path]:
    """GH200 counter pages at G=1, 2, 4 and H100 ones at G=16, 64: no G twice."""
    gh, h1 = tmp_path / "gh200", tmp_path / "h100"
    gh.mkdir()
    h1.mkdir()
    for G in (1, 2, 4):
        shutil.copy(GH200_COUNTERS / f"r3c-g{G}.json", gh)
    for G in (16, 64):
        shutil.copy(H100_COUNTERS / f"r3c-g{G}.json", h1)
    return gh, h1


def test_counter_pages_of_two_cards_in_one_dir_are_refused(tmp_path):
    """Counted, or borrowed: the pages under one dir are one card's."""
    gh, h1 = _two_card_counters(tmp_path)
    for f in h1.iterdir():
        shutil.copy(f, gh)
    with pytest.raises(M.Refused, match="counter pages .* come from more than one card"):
        build(*(run_dir(GH200, r) for r in GH200_RUNS), "--counters", gh)
    with pytest.raises(M.Refused, match="counter pages .* come from more than one card"):
        build(*H200_SESSIONS, "--bytes", f"borrowed:{gh}")


def test_counter_pages_of_two_cards_across_the_inputs_are_refused(tmp_path):
    """Found under the inputs with no --counters: each dir is one card, the
    two together are two."""
    gh, h1 = _two_card_counters(tmp_path)
    with pytest.raises(M.Refused, match="counter pages under the inputs come from more than one"):
        build(*(run_dir(GH200, r) for r in GH200_RUNS), gh, h1)


def _gh200_counters_with_uuid(tmp_path: Path, uuid: str, gs) -> Path:
    """The GH200's five counter pages under tmp_path, with card.uuid set to
    `uuid` on the pages at gs: one card name, another device."""
    out = tmp_path / "counters"
    out.mkdir(parents=True)
    for G in GS:
        page = json.loads((GH200_COUNTERS / f"r3c-g{G}.json").read_text())
        if G in gs:
            page["card"]["uuid"] = uuid
        (out / f"r3c-g{G}.json").write_text(json.dumps(page))
    return out


def test_the_counter_pages_device_is_checked_not_only_the_card_name(tmp_path):
    """A card is its name AND its device (uuid). Counter pages named for the
    GH200 but captured on another device are not its counted bytes, and one
    such page among five makes the set two cards (review, 2026-09-26)."""
    runs = [run_dir(GH200, r) for r in GH200_RUNS]
    other = "00000000-0000-0000-0000-000000000000"
    every = _gh200_counters_with_uuid(tmp_path / "a", other, GS)
    with pytest.raises(M.Refused, match="counted bytes come from the card's own pages"):
        build(*runs, "--counters", every)
    one = _gh200_counters_with_uuid(tmp_path / "b", other, (64,))
    with pytest.raises(M.Refused, match="counter pages .* come from more than one card"):
        build(*runs, "--counters", one)


def _unit_fit(bw=3000.0, identified=(True,) * 5, at_bound=(False,) * 5, gs=GS):
    return M.Fit(label="unit", k_w=0.5, x=np.array([0.05, 206.0, bw, 0.0, 0.0]),
                 fitted_gs=tuple(gs), identified=tuple(identified), at_bound=tuple(at_bound),
                 scale=np.ones(5), invisible=np.zeros((0, 5)), cost=0.0)


def _unit_score(per_G, worst=0.01):
    return {"per_G": per_G, "worst": worst, "worst_cell": "unit"}


def test_the_gates_rules():
    """T1 to T4 on made-up fits, at the edges the spec sets: T2 and T4 at 2%,
    T3 at 3x the MEDIAN of the other fitted Gs' rms (the mean is not it)."""
    flat = {G: 0.004 for G in GS}
    ok = {"R_P_gbps": 3000.0 / 1.019}
    g = M.gates(_unit_fit(), _unit_score(flat, 0.0199), ok)
    assert all(g[t]["pass"] for t in ("T1", "T2", "T3", "T4"))
    # T1: identified and off the bounds, both.
    assert not M.gates(_unit_fit(at_bound=(False, True, False, False, False)),
                       _unit_score(flat), ok)["T1"]["pass"]
    assert not M.gates(_unit_fit(identified=(True, True, False, True, True)),
                       _unit_score(flat), ok)["T1"]["pass"]
    # T2: |bw / R_P - 1| <= 2%, either sign; no R_P fails it.
    for rel, want in ((0.03, False), (-0.03, False), (0.019, True), (-0.019, True)):
        t2 = M.gates(_unit_fit(), _unit_score(flat), {"R_P_gbps": 3000.0 / (1 + rel)})["T2"]
        assert t2["pass"] is want and t2["bw_over_R_P_minus_1"] == pytest.approx(rel), rel
    assert not M.gates(_unit_fit(), _unit_score(flat), {"R_P_gbps": None})["T2"]["pass"]
    # T3: G=1 at 0.05 against others 0.01, 0.01, 0.02, 0.03: median 0.015 (x3
    # = 0.045, a trip), mean 0.0175 (x3 = 0.0525, none). No other G trips.
    t3 = M.gates(_unit_fit(), _unit_score({1: 0.05, 2: 0.01, 4: 0.01, 16: 0.02, 64: 0.03}),
                 ok)["T3"]
    assert not t3["pass"] and t3["fails_at_G"] == [1]
    # A G outside the fit is scored, never a trip.
    t3 = M.gates(_unit_fit(gs=(2, 4, 16, 64)), _unit_score({**flat, 1: 0.5}), ok)["T3"]
    assert t3["pass"] and t3["fails_at_G"] == []
    # T4: the worst fitted cell within 2%, either sign.
    for worst, want in ((0.0201, False), (-0.0201, False), (0.0199, True), (-0.0199, True)):
        assert M.gates(_unit_fit(), _unit_score(flat, worst), ok)["T4"]["pass"] is want, worst


def test_the_cli_refuses_with_refused_and_writes_json_when_it_runs(tmp_path, capsys):
    assert M.main([str(H100)]) == exit_codes.REFUSED
    out = tmp_path / "fit.json"
    assert M.main([str(GH200), "--out", str(out)]) == exit_codes.DONE
    text = capsys.readouterr().out
    assert "NOT FINAL" in text and "INTERPRETATION" in text and "k_w = 1.0" in text
    # P2's band holds n >= 2; the n1->2 step is said not to be held to it.
    assert "(a point prediction, not held to the band)" in text
    assert "this tool's reading of the judge's" in text
    doc = json.loads(out.read_text())
    assert doc["status"] == "NOT FINAL" and doc["card"] == "nvidia_gh200_480gb"
    assert len(doc["falsified_by"]) == 6
    assert {c["bytes_source"] for c in doc["cells"]} == {"COUNTED", "INTERPOLATED"}
    assert doc["reference_k_w_1"]["k_w"] == 1.0
