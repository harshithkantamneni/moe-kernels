"""scripts/paper/noise_floors.py: the committed table regenerates byte for byte
from the published pages, and its key numbers are pinned."""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import noise_floors as NF  # noqa: E402


def test_committed_table_regenerates(tmp_path):
    NF.write(NF.build(), tmp_path)
    for name in ("noise_floors.csv", "noise_floors.md"):
        assert (tmp_path / name).read_text() == (ROOT / "docs/paper" / name).read_text(), name


def test_every_row_has_a_floor_or_says_none():
    rows = list(csv.DictReader((ROOT / "docs/paper/noise_floors.csv").open()))
    assert rows
    for r in rows:
        assert r["noise_floor"].strip(), r["metric"]
        if r["method"] == "none":
            assert r["noise_floor"] == NF.NONE


def test_key_numbers():
    tb = NF.time_board()
    assert tb["cells"] == 24
    assert abs(tb["rms_diff"] - 0.0026) < 0.0005
    t2 = {r[0]: r for r in NF.t2_metric()["rows"]}
    assert abs(t2["tp2 G=64"][4] - 0.5802) < 1e-3  # D/README.md rental 1 T2 f(tp2 G=64)
    assert abs(t2["8x7B G=16"][5] - 0.2472) < 1e-3 and abs(t2["8x7B G=32"][5] - 0.4900) < 1e-3
    d = NF.tp8_D()
    assert round(d["r3f-g64.json"]) == 156941 and round(d["r3f-g8.json"]) == 155820
    p4 = NF.part4_cells()
    assert 2000 < p4["sigma_single"] < 3500  # the registered sigma_cell is 2678
    t8 = NF.time_rep8()  # rental 3 part R: scripts/scoring/rental3/replicate.score.txt
    assert t8["reg_cells"] == 9 and abs(t8["reg_rms"] - 0.001802) < 2e-6
    rk = NF.rk_floor()["per_gemm"]  # rental 3 RK, as score_replicate computes it
    assert round(rk["w1"]["sigma_L"]) == 3888 and round(rk["w2"]["sigma_D_imb"]) == 471
    tn = NF.time_native_r4()["models"]  # rental 4: NATIVE at 15 over 9 copies, n = 3..9
    assert tn["qwen2-57b-a14b-tp8"]["cells"] == 7 and abs(tn["qwen2-57b-a14b-tp8"]["rms"] - 0.001446) < 2e-6
    assert abs(tn["olmoe-1b-7b"]["rms"] - 0.000426) < 2e-6
    ob = NF.olmoe_g64_board()  # OLMoE G = 64 k64s4 on boards d663f7 and bb7a34
    assert len(ob["bytes"]["private"]) == 18 and NF.rms(ob["bytes"]["private"]) < 0.001
    assert len(ob["cycles"]["w1"]) == 12 and NF.rms(ob["cycles"]["w1"]) < 0.001
    ts = NF.time_skew_pair_r5()["models"]  # rental 5: the A and B skew pages' shared cells
    assert ts["olmoe-1b-7b"]["cells"] == 12 and abs(ts["olmoe-1b-7b"]["rms"] - 0.001124) < 2e-6
    assert ts["qwen1.5-moe-a2.7b"]["cells"] == 12 and abs(ts["qwen1.5-moe-a2.7b"]["rms"] - 0.002583) < 2e-6
    t6 = NF.time_skew_triple_r6()["models"]  # rental 6: the A, B and C skew pages' shared cells
    assert {m: v["cells"] for m, v in t6.items()} == {"mixtral-8x7b": 12, "olmoe-1b-7b": 12, "phi-3.5-moe": 11}
    assert abs(t6["olmoe-1b-7b"]["rms"] - 0.001256) < 2e-6 and abs(t6["mixtral-8x7b"]["rms"] - 0.000585) < 2e-6
    t7 = NF.time_skew_triple_r7()["models"]  # rental 7: the A, B and C pages, medians rebuilt under r7-median-run3
    assert {m: v["cells"] for m, v in t7.items()} == {"mixtral-8x7b": 12, "phi-3.5-moe": 12}
    assert abs(t7["mixtral-8x7b"]["rms"] - 0.001332) < 2e-6 and abs(t7["phi-3.5-moe"]["rms"] - 0.000740) < 2e-6
    n7 = NF.time_native_r7()["models"]  # rental 7: NATIVE at 15 copies over 9 on the Q1 pages
    assert {m: v["treads"] for m, v in n7.items()} == {"qwen1.5-moe-a2.7b": [1, 2], "phi-3.5-moe": [1, 2],
                                                         "jetmoe-8b": [1, 2, 4]}
    assert abs(n7["phi-3.5-moe"]["rms"] - 0.000270) < 2e-6


def test_cli_runs(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT / "scripts/paper/noise_floors.py"),
                        "--out-dir", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "noise_floors.csv").exists()


def test_native_and_launch_offsets_have_floors():
    rows = {r["metric"]: r for r in csv.DictReader((ROOT / "docs/paper/noise_floors.csv").open())}
    for m in ("byte error % (NATIVE per GEMM)", "bytes, NATIVE, pooled board to board (8x7B), G >= 2",
              "launch P2 E240 - GR (us)", "launch P3 E0 - E240 (us)", "launch P4 E480 - E240 (us)",
              "launch P8 H - I_E240 (us)", "launch P6 eager I (relative)"):
        assert m in rows and rows[m]["noise_floor"] != NF.NONE, m
