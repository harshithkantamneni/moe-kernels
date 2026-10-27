"""The calibration rental C3's probe off the GPU (scripts/instr_probe_c3.py): its variants and cells, vLLM's config at
the 128x128 tile, the wgmma-aware SASS check against a planted HGMMA kernel and against rental 4's published SASS (where
it must read what moe.instrumented.cubin.sass_checks reads), the launch wrapper that keeps expert_ids, the regcheck and
gate rules, and the pinned files it imports left byte-identical."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import instr_probe_c3 as P  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.instrumented import cubin as CB  # noqa: E402

HA = "docs/registered/2026-10-14-tileA-hist"
R4 = REPO / "results/published/2026-10-07-nvidia_gh200_480gb-rental4-session/results/2026-10-06-nvidia_gh200_480gb-perturb-pt"


def variant(**kw) -> dict:
    v = {"id": "u7-st7", "kind": "stamps", "model": "mixtral-8x7b", "groups": [1], "arms": ["native"], "block_m": 128,
         "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3, "histogram": f"{HA}/mixtral-8x7b-128g1.json",
         "spec": "stamps=ends"}
    v.update(kw)
    return v


def test_the_variants_cells_are_a2s_e1cal_cells_on_native():
    v = P.check_variant(variant())
    assert v["cells"] == [("PT", 2, "native"), ("PW", 2, "native"), ("uniform", 2, "native"), ("balanced", 2, "native")]
    q = P.check_variant(variant(id="u8-stq", model="qwen2-57b-a14b", histogram=f"{HA}/qwen2-57b-a14b-128g1.json"))
    assert [(c[0], c[1]) for c in q["cells"]] == [(lab, n) for n in (1, 2) for lab in ("PT", "PW", "uniform", "balanced")]
    both = P.check_variant(variant(arms=["native", "shared"]))
    assert len(both["cells"]) == 8 and {c[2] for c in both["cells"]} == {"native", "shared"}
    assert v["spec_obj"].stamps == I.STAMP_LEVELS["ends"] and v["gate_spec"] == v["spec_obj"]
    assert {P.stem(*c) for c in v["cells"]} == {"PT-n2-native", "PW-n2-native", "uniform-n2-native", "balanced-n2-native"}


def test_the_config_is_vllms_128x128x64_w8_s3_g1():
    c = P.conf(P.check_variant(variant()))
    assert {k: c[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")} == {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}
    import block_m_crossing_sweep as S
    assert S.FIXED == {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4}


@pytest.mark.parametrize("kw, why", [
    ({"spec": "stamps=cta"}, "C3 stamps the CTA ends only"),
    ({"spec": "stamps=sample,every=16"}, "C3 stamps the CTA ends only"),
    ({"spec": "stamps=ends,evict_a=first"}, "C3 stamps the CTA ends only"),
    ({"block_m": 32, "block_n": 64, "num_stages": 4}, "is not one of C3's"),
    ({"num_warps": 4}, "is not one of C3's"),
    ({"groups": [8]}, "C3's cells are G 1"),
    ({"arms": ["private"]}, "are not distinct arms"),
    ({"kind": "bytes"}, "C3 runs stamps units only"),
    ({"model": "no-such"}, "unknown model"),
    ({"histogram": f"{HA}/qwen2-57b-a14b-128g1.json"}, "the histogram file is 'qwen2-57b-a14b'"),
])
def test_a_variant_off_c3s_design_is_refused(kw, why):
    with pytest.raises(P.Refused, match=why):
        P.check_variant(variant(**kw))
    with pytest.raises(P.Refused, match="has no"):
        P.check_variant({k: v for k, v in variant().items() if k != "histogram"})


# --------------------------------------------------------------------------
# SASS
# --------------------------------------------------------------------------

def _sass(lines: list[str]) -> str:
    return "\n".join(f"        /*{16 * i:04x}*/                   {ln} ;" for i, ln in enumerate(lines)) + "\n"


HGMMA_OK = [
    "S2UR UR4, SR_CTAID.X", "CS2R R12, SR_GLOBALTIMERLO", "CS2R R14, SR_CLOCKLO", "STG.E.64 desc[UR12][R16.64+0x8], R12",
    "LDG.E R0, desc[UR12][R10.64]", "LDGSTS.E.BYPASS.128 [R38], desc[UR12][R8.64]", "BAR.SYNC.DEFER_BLOCKING 0x0",
    "WARPGROUP.ARRIVE", "HGMMA.64x128x16.F32.BF16 R24, gdesc[UR4], R24", "HGMMA.64x128x16.F32.BF16 R24, gdesc[UR8], R24",
    "WARPGROUP.DEPBAR.LE gsb0, 0x1", "BRA 0x70", "BAR.SYNC.DEFER_BLOCKING 0x0", "STG.E.128 desc[UR12][R4.64], R24",
    "CS2R R2, SR_GLOBALTIMERLO", "CS2R R4, SR_CLOCKLO", "STG.E.64 desc[UR12][R6.64+0x38], R2", "EXIT"]


def test_the_c3_sass_check_finds_the_hgmma_loop_and_no_clock_in_it():
    c = P.sass_checks_c3(_sass(HGMMA_OK))
    assert c["mma"] == "HGMMA" and c["loop"] == [0x70, 0xb0] and c["loop_mma"] == 2 and c["loop_clock_reads"] == 0
    assert c["start_before_first_ldg"] and c["clock_reads"] == 4 and c["bars_after_loop"] == 1 and c["ok"]
    # the pinned check is blind here: no HMMA, so its "loop" spans none
    old = CB.sass_checks(_sass(HGMMA_OK))
    assert old["loop"] is not None and old["loop_clock_reads"] == 0   # it found a back-branch, by chance not by HGMMA
    bad = HGMMA_OK[:9] + ["CS2R R9, SR_CLOCKLO"] + HGMMA_OK[9:]
    b = P.sass_checks_c3(_sass(bad))
    assert b["loop_clock_reads"] == 1 and not b["ok"]
    late = [ln for ln in HGMMA_OK if "SR_GLOBALTIMER" not in ln and "SR_CLOCK" not in ln]
    late = late[:4] + ["CS2R R12, SR_GLOBALTIMERLO"] + late[4:]
    assert P.sass_checks_c3(_sass(late))["start_before_first_ldg"] is False
    assert P.sass_checks_c3(_sass([ln for ln in HGMMA_OK if not ln.startswith("BRA")]))["ok"] is False   # no loop: unread
    assert P.sass_checks_c3(None)["ok"] is False and P.sass_checks_c3("")["instructions"] == 0


def test_on_rental_4s_published_mma_sync_sass_the_c3_check_reads_what_the_pinned_one_reads():
    files = sorted(R4.glob("sass/*/copy-*.sass"))
    assert len(files) == 18
    for f in files:
        text = f.read_text()
        new, old = P.sass_checks_c3(text), CB.sass_checks(text)
        assert new["mma"] == "HMMA" and new["loop"] == old["loop"], f.name
        assert (new["start_before_first_ldg"], new["loop_clock_reads"]) == (old["start_before_first_ldg"],
                                                                            old["loop_clock_reads"]), f.name
    # the per-CTA (cta) copies keep the loop clean, the per-iteration (iter) copies do not
    rep = json.loads((R4 / "perturb.json").read_text())["variants"]
    for vid, v in rep.items():
        for key, chk in v["copy_sass_checks_printed"].items():
            got = P.sass_checks_c3((R4 / "sass" / IP.gate_key(vid) / f"copy-{key}.sass").read_text())
            assert got["ok"] == (v["unit_spec"]["stamps"] in (1, 3)), (vid, key)


def test_the_ptx_path_reads_the_census_counts():
    assert P.ptx_path("wgmma.mma_async.x\nwgmma.mma_async.y\nwgmma.wait_group.sync 1")["mma"] == "wgmma"
    assert P.ptx_path("mma.sync.aligned.m16n8k16")["mma"] == "mma.sync"
    assert P.ptx_path("")["mma"] == "?"


# --------------------------------------------------------------------------
# the launch wrapper, regcheck and the gate
# --------------------------------------------------------------------------

def test_keep_launches_the_kernel_and_keeps_expert_ids_and_ntpp_by_reference():
    calls = []

    class K:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                calls.append((grid, a, kw))
                return "compiled"
            return launch
    sink = []
    args = list(range(14))
    out = P.Keep(K(), sink)[(7,)](*args, BLOCK_SIZE_M=128, USE_TD=False)
    assert out == "compiled" and calls == [((7,), tuple(args), {"BLOCK_SIZE_M": 128, "USE_TD": False})]
    assert sink == [{"expert_ids": 8, "num_tokens_post_padded": 9, "use_td": False}]
    P.Keep(K(), sink)[(7,)](*args)
    assert sink[-1]["use_td"] is False
    P.Keep(K(), sink)[(7,)](*args, USE_TD=True)
    assert sink[-1]["use_td"] is True


def _facts(regs, ctas):
    return {"regs": regs, "shared": 98304, "occupancy": {"ctas_per_sm": ctas}}


KEY = (128, 128, 64, 1, 3, 8, False, 2)


def test_regcheck_needs_equal_ctas_per_sm_and_the_wgmma_path():
    own = "stamps=ends,evict_a=none,evict_b=none"
    rows = IP.regcheck_rows({KEY: _facts(116, 2)}, {KEY: _facts(116, 2)}, {own: {KEY: _facts(118, 2)}})
    wg = {"plain": {KEY: P.ptx_path("wgmma.mma_async")}, "copy": {KEY: P.ptx_path("wgmma.mma_async")}}
    assert P.regcheck_c3(rows, own, wg) == (True, "")
    ok, why = P.regcheck_c3(IP.regcheck_rows({KEY: _facts(116, 2)}, {KEY: _facts(116, 2)}, {own: {KEY: _facts(130, 1)}}),
                            own, wg)
    assert not ok and "at 1 CTAs/SM against the plain 2" in why
    ms = {"plain": {KEY: P.ptx_path("mma.sync")}, "copy": {KEY: P.ptx_path("wgmma.mma_async")}}
    ok, why = P.regcheck_c3(rows, own, ms)
    assert not ok and "plain 128-128-64-1-3-8-False-2 on the mma.sync path" in why
    assert P.regcheck_c3(rows, own, {"plain": {}, "copy": {}})[0] is False


def test_the_gate_is_decision_2_at_rental_5s_legs_plus_the_c3_sass_legs():
    cfg = IP.compare_configs({KEY: _facts(116, 2)}, {KEY: _facts(112, 2)}, IP.OCC_LEGS_R5)
    good = {"k": {"ok": True}}
    g = P.gate_c3([1.004, 0.997, 1.009, 0.995], cfg, upstream_ok=True, sass_equal=True, brackets=good)
    assert g["verdict"] == "PASS" and g["why"] == []
    assert P.gate_c3([1.004, 1.025], cfg, upstream_ok=True, sass_equal=True, brackets=good)["verdict"] == "FAIL"
    assert P.gate_c3([1.012, 1.013], cfg, upstream_ok=True, sass_equal=True, brackets=good)["verdict"] == "FAIL"
    assert P.gate_c3([1.0], cfg, upstream_ok=True, sass_equal=True, brackets={})["verdict"] == "FAIL"
    bad = P.gate_c3([1.0], cfg, upstream_ok=True, sass_equal=True, brackets={"k": {"ok": False, "loop": None}})
    assert bad["verdict"] == "FAIL" and "k: start_before_first_ldg" in bad["why"][0]
    occ = IP.compare_configs({KEY: _facts(116, 2)}, {KEY: _facts(130, 1)}, IP.OCC_LEGS_R5)
    assert P.gate_c3([1.0], occ, upstream_ok=True, sass_equal=True, brackets=good)["verdict"] == "FAIL"
    assert P.TOL == IP.PERTURB_TOL_R5 == {"median_max": 0.01, "worst_max": 0.02, "occupancy": "identical CTAs per SM"}


def test_the_pinned_files_c3_imports_are_untouched():
    import test_pinned_files as TP
    for rel in ("moe/instrumented/__init__.py", "moe/instrumented/cubin.py", "moe/instrumented/fused_moe_instr.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == TP.BASELINE[rel], rel
    pin = "a5b421bfd6825da67a8241a9535319c83824620928309633ac3b096ca479a6fe"
    assert hashlib.sha256((REPO / "scripts/instr_probe.py").read_bytes()).hexdigest() == pin
    reg = json.loads((REPO / "docs/registered/2026-10-26-cal3-gh200.json").read_text())
    for rel in ("scripts/instr_probe.py", "scripts/instr_probe_c3.py", "moe/instrumented/fused_moe_instr.py"):
        assert reg["code_pins"][rel] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest(), rel
    assert reg["instrument"]["copy_sha256"] == TP.BASELINE["moe/instrumented/fused_moe_instr.py"]


def test_od1_splits_the_gate_into_timing_and_hard_legs(tmp_path):
    """Owner OD-1 (2026-10-26): a timing FAIL alone keeps the stamps unit (PERTURBED); the hard legs refuse it."""
    cfg = IP.compare_configs({KEY: _facts(116, 2)}, {KEY: _facts(112, 2)}, IP.OCC_LEGS_R5)
    good = {"k": {"ok": True}}
    g = P.gate_c3([1.05, 1.08], cfg, upstream_ok=True, sass_equal=True, brackets=good)
    assert g["verdict"] == "FAIL" and g["hard_verdict"] == "PASS" and g["hard_why"] == [] and len(g["timing_why"]) == 2
    occ = IP.compare_configs({KEY: _facts(116, 2)}, {KEY: _facts(130, 1)}, IP.OCC_LEGS_R5)
    for kw in ({"configs": occ}, {"sass_equal": False}, {"brackets": {}}, {"brackets": {"k": {"ok": False}}},
               {"upstream_ok": False}):
        a = {"configs": cfg, "upstream_ok": True, "sass_equal": True, "brackets": good, **kw}
        h = P.gate_c3([1.0], a.pop("configs"), **a)
        assert h["hard_verdict"] == "FAIL" and h["verdict"] == "FAIL" and h["timing_why"] == [], kw
    o = P.overhead(1.0, 1.05, [1.0, 1.0, 2.0], [1.04, 1.12, 2.0])
    assert abs(o["overhead_median"] - 0.05) < 1e-12 and abs(o["overhead_worst"] - 0.12) < 1e-12
    env = tmp_path / "gate.env"
    env.write_text("GATE_u7_st7=FAIL median=0.05 worst=0.12\nHARD_u7_st7=PASS\n")
    assert (P.read_env(env, "GATE", "u7-st7"), P.read_env(env, "HARD", "u7-st7")) == ("FAIL", "PASS")
    assert P.read_env(env, "HARD", "u8-stq") == "" and P.read_env(None, "HARD", "x") == ""
    got = __import__("subprocess").run(
        [sys.executable, "-B", str(REPO / "scripts" / "instr_probe_c3.py"), "--mode", "stamps", "--variant", "u7-st7",
         "--variants", str(_variants(tmp_path)), "--gate", str(env), "--regcheck", str(_rc(tmp_path)), "--out",
         str(tmp_path / "o"), "--dry-run"], capture_output=True, text=True, timeout=120)
    assert got.returncode == 2 and "PERTURBED: the overhead gate for u7-st7 reads FAIL" in got.stdout
    assert "REFUSED" not in got.stdout and "DRY RUN" in got.stdout
    env.write_text("GATE_u7_st7=FAIL median=0.05 worst=0.12\nHARD_u7_st7=FAIL\n")
    got = __import__("subprocess").run(
        [sys.executable, "-B", str(REPO / "scripts" / "instr_probe_c3.py"), "--mode", "stamps", "--variant", "u7-st7",
         "--variants", str(_variants(tmp_path)), "--gate", str(env), "--regcheck", str(_rc(tmp_path)), "--out",
         str(tmp_path / "o"), "--dry-run"], capture_output=True, text=True, timeout=120)
    assert "REFUSED: the perturbation gate's hard legs for u7-st7 read FAIL" in got.stdout


def _variants(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps([variant()]))
    return f


def _rc(tmp_path):
    f = tmp_path / "regcheck.env"
    f.write_text("REGCHECK_u7_st7=PASS\n")
    return f



def test_review_fixes_in_the_probe():
    """MINOR-1 the worst pair two-sided; MAJOR-2 the stamps' own span; MINOR-5 the loop SASS against the plain one."""
    import numpy as np
    assert P.overhead(1.0, 0.9, [1.0, 1.0], [0.85, 1.05])["overhead_worst"] == pytest.approx(-0.15)
    rows = np.full((4, 10), -1, np.int64)
    rows[:3, 1], rows[:3, 7], rows[:3, 9] = [100, 90, 120], [500, 700, 650], 1
    rows[3, 1], rows[3, 7], rows[3, 9] = 0, 10, 2
    assert P.stamp_span(rows) == 700 - 90
    plain = ["NOP" if ("SR_GLOBALTIMER" in ln or "SR_CLOCK" in ln) else ln for ln in HGMMA_OK]   # same addresses
    c = P.loop_compare(_sass(plain), _sass(HGMMA_OK))
    assert c["same_ops"] is True and c["plain_len"] == c["copy_len"] == 5
    moved = HGMMA_OK[:9] + ["NOP"] + HGMMA_OK[9:]
    assert P.loop_compare(_sass(plain), _sass(moved))["same_ops"] is False
    assert P.loop_compare(None, _sass(HGMMA_OK))["identical"] is None
    src = (REPO / "scripts" / "instr_probe_c3.py").read_text()
    body = src[src.index("def stamps("):src.index("def build_parser(")]
    assert body.index("card.sleep_ahead()") < body.index("fl.flush()")
