"""Rental 5's skew plumbing in R3 and the counter route (design-r5 2.1 S1 to S5 with
design-r5-review (b) and (c)): the histogram file, its cells' ids, the PRIVATE refusal,
the lifted copies rule, the provenance, and the byte leg's counter plan. Off GPU: torch
on the CPU, vLLM stubbed where an import is reached."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dram_counter_route as DCR  # noqa: E402
import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

BM = 32
SEED = 20261007


def _stub_vllm(monkeypatch):
    activation = types.ModuleType("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = lambda value: ("MoEActivation", value)
    for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
            ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
            ("vllm.model_executor.layers.fused_moe",
             types.ModuleType("vllm.model_executor.layers.fused_moe")),
            ("vllm.model_executor.layers.fused_moe.activation", activation)):
        monkeypatch.setitem(sys.modules, name, module)


def _tokens(cfg, n):
    return PW.SWEEP.tokens_for_rows(cfg, n * BM)


def _uniform(cfg, n):
    return [_tokens(cfg, n) * cfg.top_k // cfg.num_experts] * cfg.num_experts


def _skewed(cfg, n):
    c = _uniform(cfg, n)
    d = c[0] // 2
    c[0] += d
    c[1] -= d
    return c


def _doc(model="mixtral-8x7b", cells=None, **kw):
    cfg = MODEL_CONFIGS[model]
    cells = cells if cells is not None else [
        {"label": "PT", "n": 2, "counts": _skewed(cfg, 2)},
        {"label": "uniform", "n": 2, "counts": _uniform(cfg, 2)},
        {"label": "balanced", "n": 2, "counts": None}]
    return {"schema": PW.HISTOGRAM_SCHEMA, "model": model, "page": "A",
            "shuffle_seed": SEED, "cells": cells, **kw}


def _page(model="mixtral-8x7b", **kw):
    return PW.histogram_page_from(_doc(model, **kw), MODEL_CONFIGS[model], BM, model, "f" * 64)


def _write(tmp_path, doc, name="h.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


# --------------------------------------------------------------------------
# the arm-input tuple (review (b) fix 1, in place of the circular balanced_ids test)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model,n", [("mixtral-8x7b", 1), ("mixtral-8x7b", 3),
                                     ("mixtral-8x7b", 8), ("olmoe-1b-7b", 1),
                                     ("olmoe-1b-7b", 4)])
def test_the_uniform_histogram_path_unshuffled_reproduces_the_whole_arm_inputs_tuple(
        monkeypatch, model, n):
    """tokens, x, ids_by_arm (NATIVE and SHARED), weights and kw, bitwise: the
    histogram route through realize_counts at the uniform histogram builds R3's own
    balanced cell, so the unshuffled uniform path needs no GPU cell."""
    import torch
    _stub_vllm(monkeypatch)
    cfg = MODEL_CONFIGS[model]
    cell = PW.HistCell(label="uniform", n=n, counts=tuple(_uniform(cfg, n)))
    arms = (PW.NATIVE, PW.SHARED)
    torch.manual_seed(0)
    a = PW.arm_inputs(cfg, n, BM, 9, 0, "bf16", torch.bfloat16, device="cpu", arms=arms)
    torch.manual_seed(0)
    b = PW.arm_inputs(cfg, n, BM, 9, 0, "bf16", torch.bfloat16, device="cpu", arms=arms,
                      cell=cell, shuffle_seed=SEED, shuffle=False)
    assert a[0] == b[0]
    assert a[1].dtype == b[1].dtype and torch.equal(a[1], b[1])
    assert list(a[2]) == list(b[2]) == [PW.NATIVE, PW.SHARED]
    for arm in arms:
        assert a[2][arm].dtype == b[2][arm].dtype and torch.equal(a[2][arm], b[2][arm]), arm
    assert a[3].dtype == b[3].dtype and torch.equal(a[3], b[3])
    assert a[4] == b[4]


def test_the_balanced_cell_is_balanced_ids_itself_and_the_shuffle_keeps_the_histogram(
        monkeypatch):
    import torch
    cfg = MODEL_CONFIGS["olmoe-1b-7b"]
    bal = PW.HistCell(label="balanced", n=2, counts=None)
    assert torch.equal(PW.histogram_ids(cfg, bal, BM, SEED),
                       PW.SWEEP.balanced_ids(cfg, _tokens(cfg, 2), "cpu"))
    cell = PW.HistCell(label="PT", n=2, counts=tuple(_skewed(cfg, 2)))
    shuf = PW.histogram_ids(cfg, cell, BM, SEED)
    flat = PW.histogram_ids(cfg, cell, BM, SEED, shuffle=False)
    assert PW.realised_bincount(shuf, cfg.num_experts) == list(cell.counts)
    assert not torch.equal(shuf, flat), "the token-row shuffle moved rows"
    # a row permutation: the same multiset of rows, each token's k experts distinct
    assert sorted(map(tuple, shuf.tolist())) == sorted(map(tuple, flat.tolist()))
    assert all(len(set(r)) == cfg.top_k for r in shuf.tolist())
    assert torch.equal(shuf, PW.histogram_ids(cfg, cell, BM, SEED)), "deterministic"
    assert not torch.equal(shuf, PW.histogram_ids(cfg, cell, BM, SEED + 1))
    with pytest.raises(PW.HistogramRefused):
        PW.histogram_ids(cfg, cell, BM, None)


# --------------------------------------------------------------------------
# the file and its refusals
# --------------------------------------------------------------------------

def test_a_histogram_file_is_read_in_page_order_with_its_sha():
    page = _page(cells=[{"label": "PT", "n": 8, "counts": _skewed(MODEL_CONFIGS["mixtral-8x7b"], 8)},
                        {"label": "uniform", "n": 2, "counts": _uniform(MODEL_CONFIGS["mixtral-8x7b"], 2)},
                        {"label": "balanced", "n": 2, "counts": None, "arms": ["native"]}])
    assert page.treads == [2, 8]
    assert [(c.label, c.n) for c in page.ordered()] == [("uniform", 2), ("balanced", 2), ("PT", 8)]
    cells = PW.histogram_cells(page, (PW.NATIVE, PW.SHARED))
    assert [(c.label, c.n, a) for c, a in cells] == [
        ("uniform", 2, "native"), ("uniform", 2, "shared"), ("balanced", 2, "native"),
        ("PT", 8, "native"), ("PT", 8, "shared")]


@pytest.mark.parametrize("mutate,why", [
    (lambda d: d.update(schema="other/1"), "schema"),
    (lambda d: d.update(model="olmoe-1b-7b"), "this run is"),
    (lambda d: d["cells"].append(dict(d["cells"][0])), "twice"),
    (lambda d: d["cells"][0].update(counts=d["cells"][0]["counts"][:-1]), "whole numbers"),
    (lambda d: d["cells"][0]["counts"].__setitem__(0, d["cells"][0]["counts"][0] + 1), "sum to"),
    (lambda d: d["cells"][0].update(counts=[256 + 200, 256 - 200] + [64] * 6), "tokens"),
    (lambda d: d["cells"][0].update(label="P T"), "label"),
    (lambda d: d["cells"][2].update(counts=[64] * 8), "balanced"),
    (lambda d: d.update(shuffle_seed=-1), "shuffle_seed"),
    (lambda d: d.update(cells=[]), "no cells"),
])
def test_the_histogram_file_refuses_what_the_run_would_meet(mutate, why):
    d = _doc()
    mutate(d)
    with pytest.raises(PW.HistogramRefused, match=why):
        PW.histogram_page_from(d, MODEL_CONFIGS["mixtral-8x7b"], BM, "mixtral-8x7b")


def test_private_refuses_a_histogram_and_the_arms_flag_is_checked(monkeypatch):
    import torch
    _stub_vllm(monkeypatch)
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    cell = PW.HistCell(label="PT", n=2, counts=tuple(_skewed(cfg, 2)))
    with pytest.raises(PW.HistogramRefused, match="PRIVATE cannot run skewed"):
        PW.arm_inputs(cfg, 2, BM, 9, 0, "bf16", torch.bfloat16, device="cpu", cell=cell,
                      shuffle_seed=SEED)
    assert PW.parse_arms(None) == PW.ARMS
    assert PW.parse_arms("shared,native") == (PW.NATIVE, PW.SHARED)
    for bad in ("native,native", "dense", ""):
        if bad:
            with pytest.raises(PW.HistogramRefused):
                PW.parse_arms(bad)
    page = _page(cells=[{"label": "PT", "n": 2, "counts": list(cell.counts), "arms": ["private"]}])
    with pytest.raises(PW.HistogramRefused, match="PRIVATE cannot run skewed"):
        PW.histogram_cells(page, PW.ARMS)
    with pytest.raises(PW.HistogramRefused, match="does not run"):
        PW.histogram_cells(_page(cells=[{"label": "PT", "n": 2, "counts": list(cell.counts),
                                          "arms": ["shared"]}]), (PW.NATIVE,))


def _dry(tmp_path, *extra, doc=None):
    p = _write(tmp_path, doc or _doc())
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rc = PW.main(["--histogram", str(p), "--dry-run", "--device-memory-gb", "140",
                      "--session-tag", "t", *extra])
    return rc, log.getvalue()


def test_the_dry_run_refuses_private_and_lifts_the_copies_rule_without_it(tmp_path):
    rc, out = _dry(tmp_path, "--declared-copies", "9")
    assert rc == exit_codes.REFUSED and "REFUSED: PRIVATE cannot run skewed" in out
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    deep = _doc(cells=[{"label": "PT", "n": 32, "counts": _skewed(cfg, 32)},
                       {"label": "balanced", "n": 32, "counts": None}])
    rc, out = _dry(tmp_path, "--declared-copies", "9", "--arms", "native,shared", doc=deep)
    assert rc == exit_codes.REFUSED
    assert not [ln for ln in out.splitlines() if ln.startswith("REFUSED:")], out
    # the driver's check_dry reads these three lines
    assert "n_decl = 9 against n_max = 32" in out
    assert "retracted tread 32 against the 32 planned" in out
    assert any(ln.startswith("WRITES TO   ") for ln in out.splitlines())
    assert any(ln.startswith("session     t") for ln in out.splitlines())
    rc, out = _dry(tmp_path, "--arms", "native,shared", doc=deep)
    assert "REFUSED: a histogram page names its declaration" in out


def test_arms_and_shuffle_seed_ride_with_the_histogram_only(tmp_path):
    for extra, why in ((["--arms", "native"], "rides with --histogram"),
                       (["--shuffle-seed", "3"], "rides with --histogram")):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            rc = PW.main(["--dry-run", "--device-memory-gb", "140", *extra])
        assert rc == exit_codes.REFUSED and why in log.getvalue()


def test_the_run_id_takes_the_histogram_only_when_given(tmp_path):
    p = _write(tmp_path, _doc())
    plain = PW.build_parser().parse_args([])
    hist = PW.build_parser().parse_args(["--histogram", str(p), "--arms", "native,shared"])
    a, b = PW.default_run_id(plain, "card"), PW.default_run_id(hist, "card")
    assert "hist" not in a and "hist" in b and a != b


# --------------------------------------------------------------------------
# provenance (S5) and the native-only gates
# --------------------------------------------------------------------------

def test_the_provenance_block_names_every_cell_by_its_counts_sha():
    import hashlib
    page = _page()
    prov = PW.histogram_provenance(page, (PW.NATIVE, PW.SHARED), SEED)
    assert prov["file_sha256"] == "f" * 64 and prov["shuffle_seed"] == SEED and prov["page"] == "A"
    c0 = page.cells[0]
    want = hashlib.sha256(json.dumps(list(c0.counts), separators=(",", ":")).encode()).hexdigest()
    assert prov["cells"][0] == {"label": "PT", "n": 2, "counts_sha256": want,
                                "arms": ["native", "shared"]}
    assert prov["cells"][2]["counts_sha256"] == hashlib.sha256(b"null").hexdigest()


def _rows(**over):
    base = {"arm": "native", "tiles": 2, "histogram": "PT", "counts_sha256": "x",
            "bincount_ok": True, "sm_clock_load_mhz": 1710.0, "host_bound": False}
    return [dict(base, **over), dict(base, histogram="uniform")]


def test_the_histogram_gates():
    g = PW.histogram_gates(_rows(), {})
    assert g["G1_lock_thermal"]["verdict"] == PW.PASS and g["G5_provenance"]["verdict"] == PW.PASS
    assert g["G4_worst_cell_clock"]["worst_mhz"] == 1710.0
    assert PW.histogram_gates(_rows(sm_clock_load_mhz=1650.0), {})["G1_lock_thermal"]["verdict"] == PW.FAIL
    assert PW.histogram_gates(_rows(bincount_ok=False), {})["G5_provenance"]["verdict"] == PW.FAIL
    g = PW.histogram_gates(_rows(host_bound=True), {})
    assert g["G2_host_bound"]["excluded"] == ["PT/native/n2"] and g["G2_host_bound"]["verdict"] == PW.PASS


# --------------------------------------------------------------------------
# the byte leg: the counter plan, the child, the attribution
# --------------------------------------------------------------------------

def _byte_page(model="mixtral-8x7b"):
    cfg = MODEL_CONFIGS[model]
    return PW.histogram_page_from(_doc(model, cells=[
        {"label": "PT", "n": 2, "counts": _skewed(cfg, 2)},
        {"label": "uniform", "n": 2, "counts": _uniform(cfg, 2)},
        {"label": "balanced", "n": 2, "counts": None},
        {"label": "PT", "n": 16, "counts": _skewed(cfg, 16), "arms": ["native"]}]),
        cfg, BM, model, "e" * 64)


def _hplan(tmp_path, page=None, arms=(PW.NATIVE, PW.SHARED), **kw):
    return DCR.r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=BM, block_n=64, num_stages=4,
                       group_m=8, treads=[1, 2, 3], kind="measure", arms=arms,
                       calls=2, warmups=1, profile_dir=tmp_path, stem="g8",
                       histogram=page or _byte_page(), **kw)


def test_a_histogram_counter_plan_is_its_pages_cells_and_r3s_declaration(tmp_path):
    plan = _hplan(tmp_path)
    assert plan["treads"] == [2, 16] and plan["arms"] == ["native", "shared"]
    assert plan["cells"][:2] == [["native", 2, "PT"], ["shared", 2, "PT"]]
    assert plan["cells"][-1] == ["native", 16, "PT"]
    assert plan["copies_declared"] == PW.counter_declaration(MODEL_CONFIGS["mixtral-8x7b"], BM)[0]
    PW.validate_counter_plan(plan)
    design = DCR.r3_design(plan)
    assert design["histogram_page"] is True and design["histogram"]["file_sha256"] == "e" * 64
    for bad, why in ((dict(cells=plan["cells"][::-1]), "page order"),
                     (dict(arms=["native", "shared", "private"]), "PRIVATE cannot run skewed"),
                     (dict(copies_declared=12), "declaration")):
        with pytest.raises(PW.CounterPlanRefused, match=why):
            PW.validate_counter_plan({**plan, **bad})
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    shared16 = PW.histogram_page_from(_doc(cells=[{"label": "PT", "n": 16, "counts": _skewed(cfg, 16)}]),
                                      cfg, BM, "mixtral-8x7b")
    with pytest.raises(PW.CounterPlanRefused, match="shared counter ladder"):
        _hplan(tmp_path, page=shared16)
    s = PW.counter_schedule(plan)
    assert s.measured[0] == ("native", 2, "PT", 0) and len(s.warmups) == len(plan["cells"])


def test_a_balanced_counter_plan_is_unchanged(tmp_path):
    plan = DCR.r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=BM, block_n=64, num_stages=4,
                       group_m=8, treads=[1, 2, 3], kind="measure", arms=PW.ARMS,
                       calls=2, warmups=1, profile_dir=tmp_path, stem="g8")
    assert "histogram" not in plan and all(len(c) == 2 for c in plan["cells"])
    assert "histogram" not in DCR.r3_design(plan)
    assert PW.counter_schedule(plan).measured[0] == ("native", 1, 0)


def test_the_counter_child_builds_histogram_cells_through_the_timed_pages_function(
        monkeypatch, tmp_path):
    import torch
    _stub_vllm(monkeypatch)
    plan = _hplan(tmp_path)
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    seen = []
    real = PW.arm_inputs

    def spy(*a, **k):
        seen.append((a[1], k.get("cell").label if k.get("cell") else None, tuple(k["arms"])))
        return real(*a, **k)
    monkeypatch.setattr(PW, "arm_inputs", spy)
    monkeypatch.setattr(PW, "build_private_weights", lambda cfg_, dtype, c, seed, **k: (
        torch.zeros(cfg.num_experts * c, 2, 2), torch.zeros(cfg.num_experts * c, 2, 2), None))
    ranges = []

    @contextlib.contextmanager
    def nvtx(name):
        ranges.append(name)
        yield
    stack = PW.CounterStack(
        fused_experts=lambda **k: None, override_config=lambda conf: contextlib.nullcontext(),
        align=lambda ids, bm, d, emap: (torch.empty(PW.predicted_sorted_ids(ids.numel(), d, bm)),),
        device="cpu", synchronize=lambda: None, nvtx_range=nvtx,
        device_free=lambda: (None, "planted"), versions={"planted": True},
        device_identity={"name": "planted", "uuid": "planted"})
    man = PW.counter_child(plan, stack)
    assert seen == [(2, "PT", ("native", "shared")), (2, "uniform", ("native", "shared")),
                    (2, "balanced", ("native", "shared")), (16, "PT", ("native",))]
    assert ranges[0] == "r3/native/g8/n2/PT" and len(set(ranges)) == len(plan["cells"])
    assert man["order"] == plan["cells"]
    h = man["histogram"]["cells"]
    assert h["native/2/PT"]["bincount"] == _skewed(cfg, 2)
    assert h["shared/2/uniform"]["counts_sha256"] == PW.counts_sha256(_uniform(cfg, 2))
    assert set(man["grids"]) == {DCR.r3_order_key(c) for c in plan["cells"]}
    seq = DCR.r3_launch_sequence(man)
    assert seq[0] == ("native/2/PT", 0, "w1", False)
    assert len(seq) == len(plan["cells"]) * 2 * 2


def test_a_shuffle_seed_override_off_the_files_seed_is_refused(tmp_path):
    """build-r5-review (optional fix): the counter route reads the file's seed only, so R3 refuses
    a --shuffle-seed that would leave a timed page unpaired with its byte-leg page."""
    import subprocess
    f = ROOT / "docs" / "registered" / "2026-10-07-rental5-skew-hist" / "olmoe-1b-7b-B.json"
    seed = json.loads(f.read_text())["shuffle_seed"]
    base = [sys.executable, str(ROOT / "scripts" / "private_weight_reference.py"), "--model", "olmoe-1b-7b",
            "--block-m", "32", "--histogram", str(f), "--arms", "native,shared", "--declared-copies", "9",
            "--group-m", "8", "--session-tag", "t", "--dry-run", "--device-memory-gb", "96"]
    bad = subprocess.run(base + ["--shuffle-seed", str(seed + 1)], capture_output=True, text=True, timeout=300)
    assert "differs from the page's own shuffle_seed" in bad.stdout, bad.stdout[-2000:]
    ok = subprocess.run(base + ["--shuffle-seed", str(seed)], capture_output=True, text=True, timeout=300)
    assert "differs from the page's own" not in ok.stdout and "n_decl = 9 against n_max = 16" in ok.stdout
