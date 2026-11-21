"""scripts/r3_decode.py's page run off the GPU: the whole page loop (R3's histogram_schedule) through a
fake backend into the real cells.csv and report.json, read back by rental 7's r7-median-run3 rebuild;
the dry run's plan lines read by locked_r3's own parser; the spy against a fake vLLM module; GD1;
refusals; the counter plan; and, on a CUDA box only, the capture tests for the naive and small-batch
paths."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "rental7"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_fakes as F  # noqa: E402
import r3_decode as RD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS as MC  # noqa: E402

PAGES = REPO / "docs" / "registered" / "2026-10-14-decode-pages"


def _flat(model, kind):
    return lambda c, a: 0.05 + 0.001 * c.T


@pytest.fixture(scope="module")
def page_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tree")
    out = F.write_page(root, "olmoe-1b-7b", "dm", PAGES / "olmoe-1b-7b-ctl.json", "GRR", _flat("o", "c"),
                       bad_readback=("U-T4",))
    return out


def test_the_page_lands_as_r3s_readers_read_it(page_dir):
    rep = json.loads((page_dir / "report.json").read_text())
    assert rep["kind"] == "decode-page" and rep["slip_policy"] == "cell" and rep["lock_mhz"] == 1710
    assert rep["session_tag"] == "T-lock1710" and rep["inst"] == "GRR"
    rows = rep["treads_table"]
    page = json.loads((PAGES / "olmoe-1b-7b-ctl.json").read_text())
    assert len(rows) == len(page["cells"])
    bad = [r for r in rows if r["refused"]]
    assert [r["histogram"] for r in bad] == ["U-T4"] and bad[0]["lost"] and bad[0]["ms_p50"] is None
    good = [r for r in rows if not r["refused"]]
    assert all(r["clean_repeats"] == 9 and r["gd1"]["verdict"] == "PASS" and r["gd4"] for r in good)
    assert rep["histogram_gates"]["V_ALIGN"]["verdict"] == "FAIL" and rep["histogram_gates"]["GD4_graph_fidelity"]["verdict"] == "PASS"
    # rental 7's r7-median-run3 rebuild reads the page as it reads a histogram page
    import r7common as C7
    r, raw = C7.read_page(page_dir)
    rb, why = C7.rebuild(r, raw)
    assert rb is not None, why
    assert rb["g1_r7"]["repeat_rows"] == 9 * len(rows)
    for t in rb["treads_table"]:
        if not t["refused"]:
            assert t["ms_p50"] == pytest.approx(0.05 + 0.001 * t["tiles"])
    # locked_r3 reads every row's clock
    import locked_r3 as LR
    assert len(LR.cell_clocks(page_dir / "cells.csv")) == 9 * (len(rows) - 1)


def test_the_repeat_loop_is_r3s_reversed_on_odd_repeats():
    cfg = MC["mixtral-8x7b"]
    page = RD.load_page(PAGES / "mixtral-8x7b-decl.json", cfg, "mixtral-8x7b")
    be = F.FakeBackend(cfg, page, truth=lambda c, a: 0.5, copies=9)
    store = RD.Store(Path("/dev/null")) if False else None

    class S:
        def append(self, s):
            pass
    RD.run_page(cfg, page, be, inst="GRR", repeats=2, store=S(), lock_mhz=1710, min_clean=0, say=lambda m: None)
    order = [c.label for c in page.ordered()]
    labels = [lab for lab, _a in be.timed]
    assert labels[:2 * len(order)] == [x for x in order for _ in (0, 1)]
    assert labels[2 * len(order):] == [x for x in reversed(order) for _ in (0, 1)]
    arms0 = [a for _l, a in be.timed[:2]]
    arms1 = [a for _l, a in be.timed[2 * len(order):2 * len(order) + 2]]
    assert arms0 == ["native", "decl"] and arms1 == ["decl", "native"]


def test_a_slipped_repeat_is_flagged_and_kept_out_of_the_median(tmp_path):
    cfg = MC["olmoe-1b-7b"]
    page = RD.load_page(PAGES / "olmoe-1b-7b-decl.json", cfg, "olmoe-1b-7b")
    be = F.FakeBackend(cfg, page, truth=lambda c, a: 0.2 if a == "native" else 0.3, copies=9,
                       slip={("L-T32-A64", "native", 2)})
    store = RD.Store(tmp_path / "cells.csv")
    res = RD.run_page(cfg, page, be, inst="GRR", repeats=9, store=store, lock_mhz=1710, min_clean=0, say=lambda m: None)
    store.close()
    rows = RD.rows_of(cfg, page, res, 1710, 0)
    r = next(x for x in rows if x["histogram"] == "L-T32-A64" and x["arm"] == "native")
    assert r["lock_slip_repeats"] == 1 and r["clean_repeats"] == 8 and r["ms_p50"] == pytest.approx(0.2)


def test_gd1_excludes_a_host_bound_cell():
    ok = RD.gd1({"h_pre_ms": 0.01, "h_post_ms": 0.012}, 0.05, 8)
    bad = RD.gd1({"h_pre_ms": 0.2, "h_post_ms": None}, 0.05, 1)
    assert ok["verdict"] == "PASS" and ok["host_ms"] == 0.012
    assert bad["verdict"] == "FAIL" and "under 3 x host" in bad["why"]
    assert RD.gd1({"h_pre_ms": None, "h_post_ms": None}, 0.05, 1)["verdict"] == "FAIL"


def test_the_spy_records_vllms_kernel_inputs_and_restores_the_module():
    import torch

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                return ("launched", grid(kw))
            return launch

    def dispatch(*a, **kw):
        return FM.fused_moe_kernel[lambda m: (7 * m["BLOCK_SIZE_M"],)](BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, USE_TD=False,
                                                                        naive_block_assignment=False)
    FM = SimpleNamespace(dispatch_fused_moe_kernel=dispatch, fused_moe_kernel=FakeKernel())
    srt = torch.tensor([0, 1, 4, 4], dtype=torch.int32)
    with RD.Spy(FM) as spy:
        got = FM.dispatch_fused_moe_kernel(torch.zeros(2, 3), torch.zeros(4, 5, 3), None, None, None, None, None,
                                           srt, torch.tensor([3, -1]), torch.tensor([4]), False, 2,
                                           {"BLOCK_SIZE_M": 16})
    assert got == ("launched", (112,))
    assert FM.dispatch_fused_moe_kernel is dispatch and isinstance(FM.fused_moe_kernel, FakeKernel)
    r = spy.records[0]
    assert r["sorted"] == [0, 1, 4, 4] and r["expert_ids"] == [3, -1] and r["post_padded"] == 4
    assert r["grid"] == 112 and r["use_td"] is False and r["a_rows"] == 2 and r["top_k"] == 2


def run_cli(*args):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "r3_decode.py"), *args], capture_output=True,
                          text=True, timeout=120, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
                                                       "MOE_RESULTS_DIR": "/tmp/decode-dry-results"})


def test_the_dry_run_prints_r3s_plan_lines_that_locked_r3_reads():
    got = run_cli("--model", "olmoe-1b-7b", "--histogram", str(PAGES / "olmoe-1b-7b-main.json"), "--inst", "GRR",
                  "--session-tag", "B-dec-lock1710", "--slip-policy", "cell", "--lock-mhz", "1710",
                  "--min-clean-repeats", "0", "--dry-run")
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert got.stdout.startswith("DECODE PAGE") and "reason: --dry-run was given" in got.stdout
    assert "REFUSED:" not in got.stdout
    import locked_r3 as LR
    tag, where = LR.plan_names(got.stdout)
    assert tag == "B-dec-lock1710" and where.startswith("/tmp/decode-dry-results/private_weight_reference/")
    assert "one graph of 9 calls, call j on copy j mod 3" in got.stdout and "naive" in got.stdout


@pytest.mark.parametrize("extra, why", [
    (["--group-m", "4"], "GROUP_SIZE_M 1"),
    (["--slip-policy", "cell"], "needs --lock-mhz"),
    (["--rot", "2"], "under the 3 copies"),
    (["--inst", "GR1", "--rot", "3"], "--rot is GRR's"),
])
def test_the_cli_refuses(extra, why):
    args = ["--model", "olmoe-1b-7b", "--histogram", str(PAGES / "olmoe-1b-7b-main.json"), "--inst", "GRR", "--dry-run"]
    for i in range(0, len(extra), 2):
        if extra[i] in args:
            args[args.index(extra[i]) + 1] = extra[i + 1]
        else:
            args += extra[i:i + 2]
    got = run_cli(*args)
    assert got.returncode == exit_codes.REFUSED and why in got.stdout, got.stdout


def test_a_decl_page_under_grr_needs_its_copies_to_cover_the_rotation():
    cfg = MC["olmoe-1b-7b"]
    page = RD.load_page(PAGES / "olmoe-1b-7b-decl.json", cfg, "olmoe-1b-7b")
    R, _c, _w = RD.rotation(cfg, page)
    assert page.decl_copies >= R and RD.copies_for(page, "GRR", R) == page.decl_copies


def test_the_counter_plan_is_the_page_and_nothing_else():
    import dram_counter_route_decode as DD
    cfg = MC["mixtral-8x7b"]
    page = RD.load_page(PAGES / "mixtral-8x7b-ctl.json", cfg, "mixtral-8x7b")
    plan = DD.plan_of("mixtral-8x7b", page, "x.json", profile_dir=Path("/tmp"), stem="s")
    assert plan["family"] == "r3-decode" and len(plan["cells"]) == len(page.cells)
    assert DD.launches(plan) == len(page.cells) * 3 * 2
    for k, v in (("family", "r3-arms"), ("page_sha256", "0" * 64), ("gemms_per_call", 3)):
        with pytest.raises(RD.DecodeRefused):
            RD.validate_counter_plan(dict(plan, **{k: v}), page)
    with pytest.raises(RD.DecodeRefused):
        RD.validate_counter_plan(dict(plan, cells=[["nope", "native"]]), page)
    got = subprocess.run([sys.executable, str(REPO / "scripts" / "dram_counter_route_decode.py"), "--dry-run",
                          "--family", "r3-decode", "--model", "mixtral-8x7b", "--histogram",
                          str(PAGES / "mixtral-8x7b-ctl.json")], capture_output=True, text=True, timeout=120)
    assert got.returncode == exit_codes.REFUSED and "COUNTER ROUTE r3-decode" in got.stdout


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                                  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda(), reason="the capture tests run on a CUDA box with vLLM (the VM)")
@pytest.mark.parametrize("T", [1, 2, 16])
def test_on_card_the_graph_reproduces_the_eager_call_and_the_readback_holds(T):
    cfg = MC["olmoe-1b-7b"]
    page = RD.load_page(PAGES / "olmoe-1b-7b-main.json", cfg, "olmoe-1b-7b")
    cell = next(c for c in page.cells if c.T == T and c.role == "BALANCED")
    be = RD.GPUBackend(cfg, page, inst="GRR", R=3, calls=9, copies=3, seed=0)
    ids, calls = be.build(cell, "native")
    exp = RD.expected_readback(cfg, T, cell.counts, ids.reshape(-1).tolist(), declared=cfg.num_experts, copies=1)
    assert RD.check_readback(be.readback(cell, "native", ids, calls), exp) == []
    replay, per, holder = be.replay_fn(calls)
    ok, why = be.fidelity(calls, replay, holder)
    assert ok, why


def test_the_store_refuses_a_cells_csv_with_another_header(tmp_path):
    p = tmp_path / "cells.csv"
    p.write_text("arm,repeat,tiles\n")
    with pytest.raises(RD.DecodeRefused, match="another header"):
        RD.Store(p)
    q = tmp_path / "ok.csv"
    RD.Store(q).close()
    RD.Store(q).close()                     # its own header: resumes


def test_grr_refuses_a_cell_whose_calls_share_one_weight_copy(tmp_path):
    cfg = MC["olmoe-1b-7b"]
    page = RD.load_page(PAGES / "olmoe-1b-7b-ctl.json", cfg, "olmoe-1b-7b")
    be = F.FakeBackend(cfg, page, truth=lambda c, a: 0.1, R=3, same_copy=("U-T8",))
    store = RD.Store(tmp_path / "cells.csv")
    res = RD.run_page(cfg, page, be, inst="GRR", repeats=1, store=store, lock_mhz=1710, min_clean=0, say=lambda m: None)
    store.close()
    bad = {k[0]: v["readback_bad"] for k, v in res["checks"].items() if v["readback_bad"]}
    assert list(bad) == ["U-T8"] and "distinct addresses, not 3" in bad["U-T8"][0]
    assert RD.check_copies({"copy_ptrs": [5]}, 1) == [] and RD.check_copies({"copy_ptrs": [5, 5]}, 2)
