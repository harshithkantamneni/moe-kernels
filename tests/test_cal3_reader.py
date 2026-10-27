"""The calibration rental C3's reader and registration off the GPU (scripts/scoring/cal3/read_c3.py, register_c3.py,
docs/registered/2026-10-26-cal3-gh200): theta_L, the re-read launch gap and the LIFE / PACKING split on planted
schedules with known answers; every per-launch validity leg; the bands at their registered cuts; the env-file reading
against rental 5's real published gate.env and regcheck.env (no rental 4 or 5 stamp page exists: CHECK.md); the priced
table against the histogram geometry; a whole synthetic C3 session in the probe's on-disk layout read end to end in a
child process (its own repo-module check included); the commit tie; repo modules in the canonical layout (C2's
defect); and the registration's --check, constants, pins and text."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
C3 = REPO / "scripts" / "scoring" / "cal3"
for p in (REPO, REPO / "scripts", C3, REPO / "scripts" / "scoring" / "cal2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import read_c3 as R  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / "2026-10-26-cal3-gh200.json").read_text())
R5 = REPO / "results/published/2026-10-09-nvidia_gh200_480gb-rental5-session/results"
W = R.W_C
HA = "docs/registered/2026-10-14-tileA-hist"


# --------------------------------------------------------------------------
# planted launches
# --------------------------------------------------------------------------

def schedule(live: int, L: float, *, theta: float = 0.27, mode: str = "c5", ramp: float = 0.02, t0: float = 1e9):
    """Live ranks' start ns: C5's picture (wave k >= 1 spread theta L over its W positions), M4's picture (the first
    co-residency window launched together, every later CTA one staggered gap L / W after the one before; review
    MAJOR-1), or synchronized waves."""
    r = np.arange(live)
    k, u = r // W, r % W
    if mode == "stagger":
        return np.where(k >= 1, t0 + L + (r - W) * L / W, t0 + ramp * L * u / W)
    th = 0.0 if mode == "sync" else theta
    return t0 + k * L + np.where(k >= 1, th, ramp) * L * u / W


def rows_of(start, life, *, dead: int = 64, sms: int = 132, mhz: float = 1410.0):
    live = len(start)
    a = np.full((live + dead, R.HDR), -1, np.int64)
    s, e = np.round(start), np.round(start + life)
    a[:live, 0] = np.arange(live) % sms
    a[:live, 1], a[:live, 2] = s, np.round(s * mhz / 1e3)
    a[:live, 7], a[:live, 8] = e, np.round(e * mhz / 1e3)
    a[:live, 9] = 1
    a[live:, 0], a[live:, 1], a[live:, 2], a[live:, 7], a[live:, 8], a[live:, 9] = 0, 1e9, 0, 1e9 + 40, 56, 2
    return a


def lives(live, L, spread=0.01, seed=0):
    return L * (1 + spread * np.random.default_rng(seed).standard_normal(live))


EXPERTS_8x2 = np.repeat(np.arange(8), 2)   # Mixtral balanced n 2: 16 M-tiles, two per expert


def test_theta_reads_c5s_spread_m4s_stream_and_a_synchronized_wave():
    P, live = 32, 16 * 32
    for mode, want in (("c5", 0.27), ("stagger", 1.0), ("sync", 0.0)):
        s = schedule(live, 30000.0, mode=mode)
        lr = R.launch_reading(rows_of(s, lives(live, 30000.0)), EXPERTS_8x2, 16 * 128, 4096, launch_ms=0.5)
        assert lr["valid"], lr["why"]
        assert abs(lr["theta"] - want) < 0.01, (mode, lr["theta"])
        assert lr["pairs"] == 8 * 32 and lr["same_wave_sched"] == 248
        assert lr["r2_admitted"] and lr["pairs_sw_later"] == 120   # m 9 .. 15 odd, pid_n 8 .. 31 and 0 .. 31
        if mode == "stagger":
            assert lr["together_sw_later"] == 0 and abs(lr["theta_pair"] - 1) < 0.01
        else:
            assert lr["together_sw_later"] == 120 and abs(lr["theta_pair"] - want) < 0.01
        assert lr["event_over_span"] is not None
        assert abs(lr["f_mhz_median"] - 1410) < 1 and lr["sms"] == 132 and lr["dead"] == 64
    # a wave under MIN_WAVE CTAs is not fitted: 1.5 waves leave wave 1 with 132 (fitted) or fewer (not)
    th = R.wave_theta(schedule(W + R.MIN_WAVE - 1, 1e4), 1e4)
    assert th["waves"] == [] and th["theta"] is None
    th = R.wave_theta(schedule(W + R.MIN_WAVE, 1e4), 1e4)
    assert [w["k"] for w in th["waves"]] == [1] and abs(th["theta"] - 0.27) < 1e-6


def test_pairs_are_same_expert_consecutive_m_tiles_over_every_column():
    r = R.pairs(np.array([0, 0, 0, 1, 2, 2]), 3)
    assert r.tolist() == [3, 4, 5, 6, 7, 8, 15, 16, 17]   # m 1, 2 (expert 0), m 5 (expert 2)
    assert R.pairs(np.array([0, 1, 2]), 4).size == 0
    assert R.same_wave(np.array([300, 264, 263]), 32).tolist() == [True, False, True]
    assert R.gap_ratio(np.array([0.0, 10.0, 30.0]), np.array([2]), 2, 66.0, 264).tolist() == [30.0 / (2 * 66 / 264)]


@pytest.mark.parametrize("break_, why", [
    ("unstamped", "rows not stamped live or dead"), ("live", "the live CTAs are not the pids under"),
    ("td", "USE_TD True"), ("sms", "distinct SMs, not 132"), ("timer", "globaltimer resolution"),
])
def test_every_per_launch_validity_leg(break_, why):
    live, L = 512, 30000.0
    s = schedule(live, L)
    a = rows_of(s, lives(live, L), sms=100 if break_ == "sms" else 132)
    ntpp = 16 * 128
    if break_ == "unstamped":
        a[5, 9] = -1
    if break_ == "live":
        ntpp = 15 * 128
    if break_ == "timer":
        a[:live, 1] = (a[:live, 1] // 5000) * 5000   # a 5 us timer against L / 10 = 3 us
    lr = R.launch_reading(a, EXPERTS_8x2, ntpp, 4096, use_td=(break_ == "td"))
    assert not lr["valid"] and why in lr["why"], lr["why"]
    with pytest.raises(ValueError, match=r"\[ctas, 10\]"):
        R.decode(np.zeros((4, 12), np.int64))


def test_the_timer_resolution_is_rental_4s_rule():
    assert R.timer_resolution([5, 5, 37, 69, 69, 133]) == 32 and R.timer_resolution([7, 7]) is None


@pytest.mark.parametrize("x, band", [(0.25, "IN"), (0.30, "IN"), (0.2499, "NEAR"), (0.20, "NEAR"), (0.35, "NEAR"),
                                     (0.1999, "OUT"), (0.3501, "OUT"), (1.0, "OUT"), (None, "NOT READ")])
def test_theta_band_cuts(x, band):
    assert R.theta_band(x) == band


@pytest.mark.parametrize("f, band", [(0.75, "LAUNCH-TOGETHER"), (0.7499, "MIXED"), (0.2501, "MIXED"), (0.25, "STAGGERED"),
                                     (None, "NOT READ")])
def test_samewave_band_cuts(f, band):
    assert R.samewave_band(f) == band


@pytest.mark.parametrize("e, share, band", [(1.99, 0.9, "NOT REPRODUCED"), (2.0, 2 / 3, "LIFE"), (3.0, 0.66, "BOTH"),
                                            (3.0, 1 / 3, "PACKING"), (3.0, None, "NOT READ"), (None, 0.9, "NOT READ")])
def test_w1_band_cuts(e, share, band):
    assert R.w1_band(e, share) == band


def test_every_band_has_its_registered_meaning():
    for k, bands in (("theta", ("IN", "NEAR", "OUT", "NOT READ")), ("w1", ("LIFE", "PACKING", "BOTH", "NOT REPRODUCED", "DESCRIPTIVE ONLY", "NOT READ")),
                     ("samewave", ("LAUNCH-TOGETHER", "STAGGERED", "MIXED", "NOT READ"))):
        assert set(R.MEANING[k]) == set(bands)
    assert REG["readings"]["C3-R1"]["meaning"] == R.MEANING["theta"]
    assert REG["readings"]["C3-R2"]["meaning"] == R.MEANING["samewave"]
    assert REG["readings"]["C3-R3"]["meaning"] == R.MEANING["w1"]


def test_the_life_and_packing_terms_sum_exactly_to_the_whole_miss():
    pr = {"pred_ms_K1": 1.4133, "pred_ms_N0": 1.4238, "live": 4480}
    Lp = 1.4133e6 * W / 4480
    cell = {"live": 4480, "L_ns": Lp * 0.97, "launch_ms": 1.4133 / 1.034}
    x = R.r3_cell(cell, pr, 1.4133 / 1.035)
    assert abs(x["identity_residual"]) < 1e-12 and abs(x["life_term"] - math.log(1 / 0.97)) < 1e-12
    assert abs(x["e_plain_pct"] - 3.5) < 1e-9 and abs(x["L_pred_ns"] - Lp) < 1e-6
    m = R.r3_model([x, x])
    assert abs(m["life_share"] - math.log(1 / 0.97) / math.log(1.034)) < 1e-12
    assert R.r3_cell({**cell, "live": 3584}, pr, 1.0)["why"].startswith("no valid launch")
    assert R.r3_model([])["life_share"] is None


# --------------------------------------------------------------------------
# rental 5's real env files; the priced table
# --------------------------------------------------------------------------

def test_env_lines_read_rental_5s_published_gate_and_regcheck_files_as_the_probe_does():
    import instr_probe as IP
    gate = R5 / "2026-10-09-nvidia_gh200_480gb-perturb-pt" / "gate.env"
    rc = R5 / "2026-10-08-nvidia_gh200_480gb-regcheck-rc" / "regcheck.env"
    ids = ["u14-stf-v1", "u14-stf-v2", "u14-stf-v3", "u15-stt8x22-v1", "u18-std2-v2", "u18-std2-v3", "nope"]
    for vid in ids:
        assert R.env_line(gate, "GATE", vid) == IP.read_gate(gate, vid)
        assert R.env_line(rc, "REGCHECK", vid) == IP.read_regcheck(rc, vid)
    assert R.env_line(gate, "GATE", "u18-std2-v2") == "FAIL" and R.env_line(rc, "REGCHECK", "u18-std2-v2") == "PASS"
    assert R.env_line(None, "GATE", "x") == "" and R.env_line(gate, "GATE", "nope") == ""


def test_the_priced_lives_are_the_histogram_geometry_and_the_rows_are_the_registered_ones():
    import block_m_crossing_sweep as S

    from moe.spec import MODEL_CONFIGS
    rows = R.priced_rows()
    assert len(rows) == 24 and hashlib.sha256((REPO / R.PRICED).read_bytes()).hexdigest() == REG["priced"]["sha256"]
    for m, pg in REG["pages"].items():
        cfg, doc = MODEL_CONFIGS[m], json.loads((REPO / pg["path"]).read_text())
        for c in doc["cells"]:
            T = S.tokens_for_rows(cfg, c["n"] * 128)
            counts = c["counts"] or [T * cfg.top_k // cfg.num_experts] * cfg.num_experts
            tiles = sum(math.ceil(x / 128) for x in counts)
            for g, N in (("w1", 2 * cfg.intermediate_size), ("w2", cfg.hidden_size)):
                row = rows[(m, c["label"], c["n"], g)]
                assert row["live"] == tiles * -(-N // 128), (m, c["label"], g)
                assert row["board"] == {"mixtral-8x7b": "C2", "qwen2-57b-a14b": "A2"}[m]
    # the K1 miss the reading starts from: Mixtral-8x7B w1 priced 3.3 to 3.6% slow on C2's board
    e = [rows[("mixtral-8x7b", lab, 2, "w1")]["pred_ms_K1"] / rows[("mixtral-8x7b", lab, 2, "w1")]["meas_ms"] - 1
         for lab in ("PT", "PW", "uniform", "balanced")]
    assert all(0.033 < x < 0.036 for x in e), e


# --------------------------------------------------------------------------
# a whole synthetic C3 session, read end to end in a child process
# --------------------------------------------------------------------------

def build_session(root: Path, *, life_factor=0.97, pack_factor=1.0, mode="c5", gate_mixtral="PASS",
                  hard_mixtral="PASS", sass_ok=True, overhead=(0.003, 0.006), theta=0.27, late=0) -> Path:
    """A published C3 session in the probe's on-disk layout. Mixtral-8x7B's lives are life_factor of priced (K1),
    each stamped launch takes pack_factor x its waves of median lives, and the plain kernel is 0.3% faster than the
    stamped one."""
    from moe.spec import MODEL_CONFIGS
    import block_m_crossing_sweep as S
    res, ses = root / "results", root / "session"
    D = ses / "gh200-driver"
    D.mkdir(parents=True)
    (D / "c2-power.env").write_text("C2_POWER_W=700\n")
    (D / "locks.env").write_text("LOCK_PROBE_1410=held\n")
    (D / "gate-band-1410.json").write_text(json.dumps({"verdict": "PASS", "band_in_force": {"which": "primary"}}))
    priced = R.priced_rows()
    gate, rcl, pvars = [], [], {}
    for m, pg in REG["pages"].items():
        label, vid = R.UNITS[m]
        cfg, doc = MODEL_CONFIGS[m], json.loads((REPO / pg["path"]).read_text())
        d = res / f"2026-10-26-nvidia_gh200_480gb-instr-{m}-{label}"
        (d / "stamps").mkdir(parents=True)
        idx, cells = [], []
        for c in doc["cells"]:
            T = S.tokens_for_rows(cfg, c["n"] * 128)
            counts = c["counts"] or [T * cfg.top_k // cfg.num_experts] * cfg.num_experts
            ex = np.repeat(np.arange(cfg.num_experts), [math.ceil(x / 128) for x in counts]).astype(np.int32)
            for g, N in (("w1", 2 * cfg.intermediate_size), ("w2", cfg.hidden_size)):
                pr = priced[(m, c["label"], c["n"], g)]
                live = pr["live"]
                L = pr["pred_ms_K1"] * 1e6 * W / live * (life_factor if m == "mixtral-8x7b" else 1.0)
                ms = live * L / W / 1e6 * pack_factor
                cells.append({"label": c["label"], "n": c["n"], "arm": "native", "gemm": g, "plain_ms": ms / 1.003,
                              "overhead_median": overhead[0] if m == "mixtral-8x7b" else 0.003,
                              "overhead_worst": overhead[1] if m == "mixtral-8x7b" else 0.006})
                for call in range(3):
                    name = f"{c['label']}-n{c['n']}-native-call{call}-{g}"
                    np.save(d / "stamps" / f"{name}.npy", rows_of(schedule(live, L, mode=mode, theta=theta),
                                                                  lives(live, L, seed=call)))
                    np.save(d / "stamps" / f"{name}-experts.npy", np.concatenate([ex, np.full(8, -1, np.int32)]))
                    idx.append({"label": c["label"], "n": c["n"], "arm": "native", "call": call, "gemm": g,
                                "file": f"{name}.npy", "experts_file": f"{name}-experts.npy", "N": N,
                                "num_tokens_post_padded": len(ex) * 128, "use_td": False, "launch_ms": ms,
                                "late_buffers": late if m == "mixtral-8x7b" else 0})
        verdict = gate_mixtral if m == "mixtral-8x7b" else "PASS"
        (d / "stamps.json").write_text(json.dumps({"histogram_sha256": pg["sha256"], "tile": [128, 128, 64, 8, 3],
                                                   "page_label": "CLEAN" if verdict == "PASS" else "PERTURBED",
                                                   "spec_text": "stamps=ends,evict_a=none,evict_b=none",
                                                   "sass": [{"checks_c3": {"ok": sass_ok}}], "launches": idx}))
        key = "".join(ch if ch.isalnum() else "_" for ch in vid)
        gate.append(f"GATE_{key}={verdict} median=0.003 worst=0.006")
        gate.append(f"HARD_{key}={hard_mixtral if m == 'mixtral-8x7b' else 'PASS'}")
        rcl.append(f"REGCHECK_{key}=PASS")
        pvars[vid] = {"cells": cells, "median_dev": 0.003, "worst_dev": 0.006, "why": [], "sass_checks_c3": {}}
    g = res / "2026-10-26-nvidia_gh200_480gb-perturb-pt3"
    g.mkdir()
    (g / "gate.env").write_text("\n".join(gate) + "\n")
    (g / "perturb.json").write_text(json.dumps({"variants": pvars}))
    rc = res / "2026-10-26-nvidia_gh200_480gb-regcheck-rc3"
    rc.mkdir()
    (rc / "regcheck.env").write_text("\n".join(rcl) + "\n")
    return root


def child_read(c3: Path, *, skip_commit: bool = True) -> dict:
    code = ("import json, sys; sys.path.insert(0, %r); import read_c3 as R; "
            "d = R.read(sys.argv[1], _skip_commit_check_for_tests=%s); "
            "print(json.dumps({'R1': d['C3-R1'], 'R2': d['C3-R2'], 'R3': d['C3-R3'], 'R4': d['C3-R4'], "
            "'q': d['models']['qwen2-57b-a14b']['gemms'].get('w2', {}).get('theta_L'), 'txt': R.txt(d)}, default=str))"
            % (str(C3), skip_commit))
    got = subprocess.run([sys.executable, "-B", "-c", code, str(c3)], capture_output=True, text=True, timeout=600,
                         cwd=REPO)
    if got.returncode != 0:
        return {"error": got.stdout + got.stderr}
    return json.loads(got.stdout)


@pytest.fixture(scope="module")
def life(tmp_path_factory):
    return child_read(build_session(tmp_path_factory.mktemp("c3life")))


def test_a_c5_launch_with_short_lives_reads_in_launch_together_and_life(life):
    assert "error" not in life, life.get("error")
    assert life["R4"]["valid"] == {"mixtral-8x7b": True, "qwen2-57b-a14b": True}, life["R4"]
    assert life["R1"]["band"] == "IN" and abs(life["R1"]["theta_L_mixtral_w2"] - 0.27) < 0.005
    assert life["R2"]["band"] == "LAUNCH-TOGETHER" and life["R2"]["together_sw_later_share_mixtral_w2"] == 1.0
    assert life["R1"]["constant_label"] == "MEASURED" and life["R1"]["in_sample_for"] == "mixtral-8x7b"
    r3 = life["R3"]
    assert r3["band"] == "LIFE" and abs(r3["mixtral_w1"]["life_share"] - 1.0) < 0.05
    assert abs(r3["mixtral_w1"]["e_plain_mean_pct"] - 100 * (1.003 / 0.97 - 1)) < 0.05
    assert abs(r3["mixtral_w1"]["overhead_abs_median"] - 0.003) < 1e-12
    assert abs(life["q"] - 0.27) < 0.005 and "C3-R1: IN [CLEAN pages]" in life["txt"]
    assert life["R1"]["page_label"] == "CLEAN" and life["R3"]["use"].startswith("registered reading")
    assert life["R1"]["overhead_w2"]["median"] == 0.003 and life["R1"]["overhead_w2"]["worst"] == 0.006


def test_a_timing_only_gate_fail_reads_perturbed_with_the_overhead_disclosed(tmp_path):
    """Owner OD-1: theta_L and the gap are read on PERTURBED pages; the w1 split is descriptive only."""
    d = child_read(build_session(tmp_path, gate_mixtral="FAIL", overhead=(0.06, 0.15)))
    assert "error" not in d, d.get("error")
    assert d["R4"]["valid"]["mixtral-8x7b"] is True
    assert d["R1"]["band"] == "IN" and d["R1"]["page_label"] == "PERTURBED" and d["R1"]["overhead_w2"]["median"] == 0.06
    assert d["R2"]["band"] == "LAUNCH-TOGETHER" and d["R2"]["overhead_bound"] == [0.10, 0.20]
    assert d["R3"]["band"] == "DESCRIPTIVE ONLY" and d["R3"]["use"].startswith("DESCRIPTIVE ONLY")   # 6% > 3.4% / 4
    assert d["R1"]["constant_label"] == "MEASURED-PERTURBED" and abs(d["R1"]["theta_times_1_plus_o"] - 0.27 * 1.06) < 0.01
    assert "[PERTURBED pages]" in d["txt"] and "never used to price or select a model form" in d["txt"]


@pytest.mark.parametrize("over", [(0.11, 0.15), (0.05, 0.21)])
def test_over_the_overhead_bound_the_ordering_readings_are_not_read(tmp_path, over):
    d = child_read(build_session(tmp_path, gate_mixtral="FAIL", overhead=over))
    assert "error" not in d, d.get("error")
    assert d["R1"]["band"] == d["R2"]["band"] == "NOT READ" and "over the bound" in d["R1"]["not_read_why"]
    assert d["R3"]["band"] == "DESCRIPTIVE ONLY" and d["R3"]["use"].startswith("DESCRIPTIVE ONLY")


def test_a_perturbed_theta_straddling_two_bands_reports_both(tmp_path):
    d = child_read(build_session(tmp_path, gate_mixtral="FAIL", overhead=(0.06, 0.10), theta=0.24))
    assert "error" not in d, d.get("error")
    assert d["R1"]["band"] == "NEAR or IN (PERTURBED)" and "neither band's meaning applies" in d["R1"]["meaning"]


def test_a_late_buffer_voids_the_launch(tmp_path):
    d = child_read(build_session(tmp_path, late=1))
    assert "error" not in d, d.get("error")
    assert d["R1"]["band"] == "NOT READ" and "late_buffers 1" in d["R4"]["why"]["mixtral-8x7b"]


def test_the_overhead_bound_is_two_sided_and_r3s_band_needs_a_small_overhead():
    assert not R.overhead_ok(R.overhead_summary([{"median": -0.15, "worst": -0.16}]))[0]   # a copy 15% faster
    assert R.overhead_summary([{"median": -0.02, "worst": -0.21}])["worst"] == 0.21
    assert R.w1_band(4.0, 0.9, 0.0099) == "LIFE" and R.w1_band(4.0, 0.9, 0.0101) == "DESCRIPTIVE ONLY"
    assert R.w1_band(4.0, 0.9, None) == "NOT READ" and R.w1_band(1.5, 0.9, 0.0) == "NOT REPRODUCED"
    assert REG["constants"]["r3_overhead_fraction_of_miss"] == 0.25 and REG["constants"]["r2_timer_fraction_of_gap"] == 0.25


def test_a_coarse_timer_drops_the_launch_from_r2_only():
    live, L = 512, 30000.0
    a = rows_of(schedule(live, L), lives(live, L))
    a[:live, 1] = (a[:live, 1] // 1024) * 1024   # 1.024 us: under L / 10 = 3 us, over P L / (4 W) = 0.91 us
    lr = R.launch_reading(a, EXPERTS_8x2, 16 * 128, 4096)
    assert lr["valid"] and not lr["r2_admitted"] and lr["pairs_sw_later"] == 0 and lr["theta"] is not None


def test_the_overhead_bound_and_summary():
    s = R.overhead_summary([{"median": 0.02, "worst": 0.05}, {"median": 0.08, "worst": 0.2}, None, {"median": 0.04, "worst": 0.1}])
    assert s == {"median": 0.04, "worst": 0.2, "cells": 3}
    assert R.overhead_ok(s) == (True, "")
    assert R.overhead_ok({"median": 0.1, "worst": 0.2})[0] and not R.overhead_ok({"median": 0.1001, "worst": 0.1})[0]
    assert not R.overhead_ok({"median": 0.01, "worst": 0.2001})[0] and not R.overhead_ok({"median": None, "worst": 0.1})[0]
    assert REG["gate"]["not_read_bound"].startswith("C3-R1 and C3-R2 are NOT READ")
    assert (REG["constants"]["overhead_median_max"], REG["constants"]["overhead_worst_max"]) == (0.10, 0.20)


def test_the_od4_schedule_note_is_the_pages_geometry():
    import block_m_crossing_sweep as S

    from moe.spec import MODEL_CONFIGS
    note = REG["schedule_note"]
    assert "does NOT reproduce for Mixtral-8x7B" in note["statement"]
    shares = {}
    for m, f in (("mixtral-8x7b", f"{HA}/mixtral-8x7b-128g1.json"),
                 ("mixtral-8x22b", "docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json")):
        cfg, doc = MODEL_CONFIGS[m], json.loads((REPO / f).read_text())
        n_all = n_same = 0
        for c in doc["cells"]:
            T = S.tokens_for_rows(cfg, c["n"] * 128)
            counts = c["counts"] or [T * cfg.top_k // cfg.num_experts] * cfg.num_experts
            ex = np.repeat(np.arange(cfg.num_experts), [math.ceil(x / 128) for x in counts])
            P = cfg.hidden_size // 128
            r = R.pairs(ex, P)
            sw = R.same_wave(r, P)
            shares[(m, c["label"])] = round(float(sw.mean()), 3)
            n_all, n_same = n_all + r.size, n_same + int(sw.sum())
        shares[m] = round(n_same / n_all, 3)
    assert [shares[("mixtral-8x7b", x)] for x in ("PT", "PW", "uniform", "balanced")] == [0.896, 0.896, 0.969, 0.969]
    assert shares["mixtral-8x7b"] == 0.925 and shares["mixtral-8x22b"] == 0.844
    x22 = [v for k, v in shares.items() if isinstance(k, tuple) and k[0] == "mixtral-8x22b"]
    assert (min(x22), max(x22)) == (0.812, 0.9)
    for s in ("0.896", "0.969", "0.925", "0.812 to 0.900", "0.844"):
        assert s in note["statement"], s


def test_a_packing_miss_and_a_staggered_launch_read_as_such(tmp_path):
    d = child_read(build_session(tmp_path, life_factor=1.0, pack_factor=0.97, mode="stagger"))
    assert "error" not in d, d.get("error")
    assert d["R1"]["band"] == "OUT" and abs(d["R1"]["theta_L_mixtral_w2"] - 1.0) < 0.01
    assert d["R2"]["band"] == "STAGGERED" and d["R2"]["together_sw_later_share_mixtral_w2"] == 0
    assert d["R2"]["together_meas_share_all_pairs_mixtral_w2"] > 0.3   # wave 0 together: why the band excludes it
    # the overhead correction: LIFE carries ln(1.003) of the 0.034 miss
    assert d["R3"]["band"] == "PACKING" and abs(d["R3"]["mixtral_w1"]["life_share"] - math.log(1.003) / (
        math.log(1.003) + math.log(1 / 0.97))) < 0.01


@pytest.mark.parametrize("kw, why", [({"gate_mixtral": "FAIL", "hard_mixtral": "FAIL"}, "gate hard legs FAIL"),
                                     ({"sass_ok": False}, "C3's SASS legs")])
def test_a_failed_hard_leg_or_sass_leg_reads_not_read(tmp_path, kw, why):
    d = child_read(build_session(tmp_path, **kw))
    assert "error" not in d, d.get("error")
    assert d["R1"]["band"] == d["R2"]["band"] == d["R3"]["band"] == "NOT READ"
    assert not d["R4"]["valid"]["mixtral-8x7b"] and why in d["R4"]["why"]["mixtral-8x7b"]


def test_read_refuses_without_the_rentals_commit_or_on_a_commit_that_lacks_c3(tmp_path):
    root = build_session(tmp_path)
    d = child_read(root, skip_commit=False)
    assert "REFUSED: the rental's commit is not on its ledger" in d["error"]
    base = subprocess.run(["git", "-C", str(REPO), "rev-parse", "e324867c"], capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip("no e324867c in this checkout")
    (root / "session" / "gh200-driver" / "status").write_text(f"t driver start: commit {base.stdout.strip()}, steps\n")
    d = child_read(root, skip_commit=False)
    assert "REFUSED: the working tree is not the rental's commit" in d["error"] and "absent at e324867c" in d["error"]


def test_repo_modules_in_the_canonical_layout(tmp_path, monkeypatch):
    """C2's defect, held for C3: only tracked, non-environment, non-test modules under the repo count."""
    import subprocess as sp
    root = tmp_path / "repo"
    files = {"scripts/x.py": "x", "tests/job.py": "h", "_ops.py": "trap", "_classes.py": "trap",
             ".venv/lib/python3.12/site-packages/torch/_ops.py": "t", "venv/site-packages/y.py": "t",
             "lib/dist-packages/z.py": "t"}
    for f, body in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(body)
    for args in (("init", "-q"), ("add", "-A", "-f"), ("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c")):
        sp.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "scripts" / "untracked.py").write_text("u")
    monkeypatch.chdir(root)

    def mod(f):
        m = types.ModuleType("m")
        m.__file__ = f
        return m
    mods = [mod("_ops.py"), mod("_classes.py"), mod(str(root / "scripts" / "x.py")), mod(str(root / "tests" / "job.py")),
            mod(str(root / ".venv/lib/python3.12/site-packages/torch/_ops.py")), mod(str(root / "venv/site-packages/y.py")),
            mod(str(root / "lib/dist-packages/z.py")), mod(str(root / "scripts" / "untracked.py")), mod(None)]
    x_sha = hashlib.sha256(b"x").hexdigest()
    assert R.check_repo_modules({"code_pins": {"scripts/x.py": x_sha}}, root, mods) == []
    assert R.check_repo_modules({"code_pins": {}}, root, mods) == [("scripts/x.py", "loaded by the reader but not pinned")]
    assert R.check_repo_modules({"code_pins": {"scripts/x.py": "0" * 64}}, root, mods) == [
        ("scripts/x.py", "not its registered sha256")]


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------

def test_the_registration_is_its_generators_output():
    got = subprocess.run([sys.executable, "-B", str(C3 / "register_c3.py"), "--check"], capture_output=True, text=True,
                         timeout=300)
    assert got.returncode == 0 and got.stdout.count("same") == 2, got.stdout + got.stderr


def test_the_registrations_pins_hold_and_its_constants_are_the_readers():
    import register_c3 as RG
    for p, s in REG["code_pins"].items():
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == s, p
    assert set(REG["code_pins"]) == set(RG.PINNED)
    assert {"scripts/scoring/cal3/read_c3.py", "scripts/scoring/cal2/read_c2.py"} <= set(REG["code_pins"])
    c = REG["constants"]
    assert (c["W_c"], c["SMs"], c["CTAs_per_SM"], c["min_wave_ctas"]) == (264, 132, 2, 132)
    assert c["theta_in"] == [0.25, 0.30] and c["theta_near"] == [0.20, 0.35] and c["together_gap_below"] == 0.5
    assert (c["together_band"], c["staggered_band"], c["reproduce_min_pct"], c["form"]) == (0.75, 0.25, 2.0, "K1")
    from moe import instrumented as I
    assert R.COLUMNS == I.COLUMNS and R.HDR == I.HDR and REG["instrument"]["row"] == list(I.COLUMNS)
    assert {m: v["variant"] for m, v in REG["pages"].items()} == {m: u[1] for m, u in R.UNITS.items()}
    assert REG["readings"].keys() == {"C3-R1", "C3-R2", "C3-R3", "C3-R4"}
    assert "within 1.0%" in REG["next_round_gate"] and "at least 0.5" in REG["next_round_gate"]
    text = (REPO / "docs" / "registered" / "2026-10-26-cal3-gh200.txt").read_text()
    assert text.startswith("REGISTERED 2026-10-26, before any C3 page") and REG["priced"]["sha256"] in text
    for k in ("C3-R1", "C3-R2", "C3-R3", "C3-R4", "OWNER DECISIONS", "NEXT ROUND GATE"):
        assert k in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_dumps_sorts_keys():
    assert R.dumps({"b": 1, "a": {"d": 2, "c": 3}}) == '{\n "a": {\n  "c": 3,\n  "d": 2\n },\n "b": 1\n}\n'
