"""Rental RR's counter route off the GPU (scripts/dram_counter_route_rr.py): the dry run (exit 2, no REFUSED: line, the
driver's dry_ok contract); every refusal of a page and a plan; the launch window (every warm-up skipped, exactly the
measured launches kept); the child's launch order on a fake card (every cell's readback and warm-ups before any measured
call; a failed readback stops it before any measured call); the parent on a faked ncu (a closed route, a missing clock
metric, a failed capture, a refused readback, a mis-gridded launch: each a refusal and no page; a clean capture: the page
written atomically with each cell's per-call bytes and medians); and the pinned route left byte-identical."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "rr", REPO / "scripts" / "scoring" / "pc2",
          REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dram_counter_route as DCR  # noqa: E402
import dram_counter_route_rr as CR  # noqa: E402
import r3_decode as RD  # noqa: E402
import rrcommon as P  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PAGE = P.PAGE_OF["s128"]
REG = json.loads((REPO / "docs" / "registered" / f"{P.NAME}.json").read_text())


def run(*args):
    return subprocess.run([sys.executable, "-B", str(REPO / P.COUNTER), *args], capture_output=True, text=True,
                          timeout=120)


def test_the_dry_run_lists_the_page_and_exits_2_with_no_refused_colon():
    for page in (PAGE, P.PAGE_OF["s64"]):
        got = run("--dry-run", "--page", page)
        assert got.returncode == exit_codes.REFUSED and "REFUSED:" not in got.stdout, got.stdout
        assert "40 cells" in got.stdout and "240 profiled launches" in got.stdout and "REFUSED. Nothing" in got.stdout


@pytest.mark.parametrize("args, why", [
    (["--dry-run"], "--page <an RR page> is required"),
    (["--dry-run", "--page", "/etc/passwd"], "a repo-relative path"),
    (["--dry-run", "--page", f"{P.PAGES_DIR}/../x.json"], "a repo-relative path"),
    (["--dry-run", "--page", f"{P.PAGES_DIR}/none.json"], "cannot read the page"),
    (["--dry-run", "--page", "docs/registered/2026-11-10-pc2-pages/pc2-d.json"], "schema"),
    (["--run", "--page", PAGE], "--run needs --out"),
    (["--dry-run", "--page", PAGE, "--warmups", "0"], "warmup_calls >= 1"),
    (["--dry-run", "--page", PAGE, "--calls", "1"], "calls_per_cell >= 2"),
])
def test_every_refusal_of_the_page_and_the_plan(args, why):
    got = run(*args)
    assert got.returncode == exit_codes.REFUSED and any(ln.startswith("REFUSED: ") and why in ln for ln in got.stdout.splitlines()), got.stdout


def _page():
    return CR.load_page(PAGE)


def test_the_plan_its_validation_and_the_launch_window(tmp_path):
    doc, cells, sha = _page()
    plan = CR.plan_of(PAGE, doc, cells, sha, profile_dir=tmp_path, stem="x")
    assert sha == REG["pages"][PAGE]["sha256"] and len(plan["cells"]) == 40
    assert plan["cells"] == [c["label"] for c in CR.order(cells)]
    assert CR.window(plan) == (40 * 2 * 2, 40 * 3 * 2)
    for bad, why in ((dict(plan, family="r3-decode"), "family"), (dict(plan, gemms_per_call=3), "gemms_per_call"),
                     (dict(plan, cells=plan["cells"] + ["NOPE"]), "not the page's"),
                     (dict(plan, cells=plan["cells"][:1] * 2), "not the page's"), ({"family": CR.FAMILY}, "lacks")):
        with pytest.raises(P.Refused, match=why):
            CR.validate_plan(bad, cells)


# --------------------------------------------------------------------------
# the child on a fake card
# --------------------------------------------------------------------------

class FakeSpy:
    last = None

    def __init__(self, module=None):
        self.records = []
        FakeSpy.last = self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_child(monkeypatch, tmp_path, bad_label=None, cells=3):
    doc, cl, sha = _page()
    log = []
    torch = argparse.Namespace(cuda=argparse.Namespace(synchronize=lambda: None, nvtx=argparse.Namespace(
        range_push=lambda s: None, range_pop=lambda: None)))

    class Card:
        def __init__(self):
            self.torch, self.FM, self.upstream, self.where, self.cfg = torch, None, {}, "hook", None

        @contextlib.contextmanager
        def override_config(self, cf):
            yield

        def cell_fn(self, v, c):
            self.cfg = c["cfg"]
            ids = RD.cell_ids(c["cfg"], RD.DecodeCell(label=c["label"], T=c["T"], counts=tuple(c["counts"]),
                                                      arms=(RD.NATIVE,)), v["shuffle_seed"])
            conf = CR_conf(c["tile"])
            exp = RD.expected_readback(c["cfg"], c["T"], c["counts"], ids.reshape(-1).tolist(),
                                       declared=c["cfg"].num_experts, copies=1, conf=conf)

            def fn():
                log.append(c["label"])
                if FakeSpy.last is not None and not FakeSpy.last.records:
                    FakeSpy.last.records = [{"sorted": list(range(exp["numel"])) + [exp["numel"]] * (
                        exp["sorted_len"] - exp["numel"]), "expert_ids": exp["expert_order"],
                        "post_padded": exp["post_padded"] + (1 if c["label"] == bad_label else 0), "grid": g,
                        "use_td": False, "config": dict(conf)} for g in exp["grid"]]
            return fn, ids
    import instr_probe_rr as PR
    CR_conf = PR.conf_of
    monkeypatch.setattr(CR, "counter_card", lambda a: (Card(), argparse.Namespace(conf_of=PR.conf_of, PW=argparse.Namespace(
        _package_versions=lambda: dict(P.VERSIONS)))))
    monkeypatch.setattr(RD, "Spy", FakeSpy)
    plan = CR.plan_of(PAGE, doc, cl, sha, profile_dir=tmp_path, stem="x")
    plan["cells"] = plan["cells"][:cells]
    f = tmp_path / "plan.json"
    f.write_text(json.dumps(plan))
    orig_enter = FakeSpy.__enter__

    def enter(self):
        FakeSpy.last = self
        return orig_enter(self)
    monkeypatch.setattr(FakeSpy, "__enter__", enter)
    rc = CR.counter_child(f)
    FakeSpy.last = None
    return rc, log, json.loads(Path(plan["manifest"]).read_text()), plan


def test_the_child_runs_every_warm_up_before_any_measured_call(monkeypatch, tmp_path):
    rc, log, man, plan = fake_child(monkeypatch, tmp_path)
    labs = plan["cells"]
    assert rc == exit_codes.DONE and man["refused"] == []
    assert log == [x for lab in labs for x in [lab] * 2] + [x for lab in labs for x in [lab] * 3]
    assert man["order"] == [["rr", lab] for lab in labs] and man["launch_count"] == 3 * 3 * 2
    assert man["launch_skip"] == 3 * 2 * 2 and set(man["grids"]) == {f"rr/{lab}" for lab in labs}
    assert DCR.r3_launch_sequence(man)[0] == (f"rr/{labs[0]}", 0, "w1", False)
    assert len(DCR.r3_launch_sequence(man)) == man["launch_count"]


def test_a_failed_readback_stops_the_child_before_any_measured_call(monkeypatch, tmp_path):
    doc, cl, sha = _page()
    second = CR.plan_of(PAGE, doc, cl, sha, profile_dir=tmp_path, stem="y")["cells"][1]
    rc, log, man, plan = fake_child(monkeypatch, tmp_path, bad_label=second)
    assert rc == exit_codes.INVALID and man["refused"][0]["cell"] == f"rr/{second}"
    assert log == [plan["cells"][0]] * 2 + [second] and man["order"] == [["rr", plan["cells"][0]]]


# --------------------------------------------------------------------------
# the parent on a faked ncu
# --------------------------------------------------------------------------

def fake_parent(monkeypatch, tmp_path, *, open_=True, names=True, rc=0, refused=False, bad_grid=False):
    doc, cl, sha = _page()
    labs = [c["label"] for c in CR.order(cl)]
    monkeypatch.setattr(DCR, "probe_ncu", lambda fam, clock_control=None: {"binary": "ncu", "version": "x",
                                                                          "metrics_proven": list(DCR.R3_STRICT_METRICS)})
    monkeypatch.setattr(DCR, "counter_route_is_open", lambda n: open_)
    monkeypatch.setattr(DCR, "query_metric_names", lambda b: ({"sm__cycles_elapsed"} if names else None, "q"))

    def capture(argv, log, timeout, stale):
        for s in stale:
            Path(s).unlink(missing_ok=True)
        assert "--launch-skip" in argv and argv[argv.index("--launch-skip") + 1] == str(40 * 2 * 2)
        assert argv[argv.index("--launch-count") + 1] == str(40 * 3 * 2) and "--counter-child" in argv
        man = {"order": [["rr", lab] for lab in labs], "grids": {f"rr/{lab}": {"w1": 7, "w2": 5} for lab in labs},
               "refused": [{"cell": "rr/X", "why": ["planted"]}] if refused else [], "calls_per_cell": 3,
               "warmup_calls": 2, "gemms_per_call": 2, "launch_count": 240, "versions": dict(P.VERSIONS)}
        Path(stale[0]).write_text(json.dumps(man))
        return rc, "tail"
    monkeypatch.setattr(DCR, "_r3_capture", capture)
    monkeypatch.setattr(DCR, "_r3_reduce", lambda b, r, c: "csv")
    launches = []
    for lab in labs:
        for k in range(3):
            for g, grid in (("w1", 7), ("w2", 5)):
                launches.append(DCR.Launch(launch_id=str(len(launches)), kernel="fused_moe_kernel", metrics={
                    "launch__grid_size": 6 if (bad_grid and len(launches) == 9) else grid,
                    "dram__bytes_read.sum": 1000.0 + k, "gpu__time_duration.sum": 1e6,
                    "sm__cycles_elapsed.avg": 1.33e6}))
    monkeypatch.setattr(DCR, "parse_ncu_csv", lambda text, soft=frozenset(): launches)
    out = tmp_path / "lock1410" / "r3d.json"
    out.parent.mkdir(parents=True)
    a = argparse.Namespace(page=PAGE, out=out, calls=3, warmups=2, page_lock_mhz=1410, timeout_s=10.0)
    return CR.run(a), out


@pytest.mark.parametrize("kw, rc, msg", [({"open_": False}, exit_codes.REFUSED, "not open"),
                                         ({"names": False}, exit_codes.REFUSED, "sm__cycles_elapsed"),
                                         ({"rc": 1}, exit_codes.INVALID, "the capture exited 1"),
                                         ({"refused": True}, exit_codes.INVALID, "VALIDITY readback FAIL"),
                                         ({"refused": True, "rc": 3}, exit_codes.INVALID, "(child exit 3, after 40 cells"),
                                         ({"bad_grid": True}, exit_codes.INVALID, "VALIDITY grid FAIL")])
def test_every_parent_refusal_writes_no_page(monkeypatch, tmp_path, capsys, kw, rc, msg):
    got, out = fake_parent(monkeypatch, tmp_path, **kw)
    assert got == rc and msg in capsys.readouterr().out and not out.exists()


def test_a_clean_capture_writes_the_page_atomically(monkeypatch, tmp_path):
    got, out = fake_parent(monkeypatch, tmp_path)
    assert got == exit_codes.DONE and not list(out.parent.glob("*.tmp"))
    page = json.loads(out.read_text())
    assert page["schema"] == P.COUNTER_SCHEMA and page["gates"]["complete"] is True
    assert page["design"]["page_sha256"] == REG["pages"][PAGE]["sha256"] and page["design"]["lock_mhz"] == 1410
    c = page["cells"][0]
    assert c["per_call"]["w1"]["dram__bytes_read.sum"] == [1000.0, 1001.0, 1002.0]
    assert c["per_gemm"]["w1"]["dram__bytes_read.sum"] == 1001.0 and c["clock_mhz"]["w1"] == [pytest.approx(1330.0)] * 3
    assert len(page["cells"]) == 40


def test_the_pinned_route_is_untouched():
    pins = json.loads((REPO / "docs/registered/2026-11-06-pc-gh200.json").read_text())["code_pins"]
    for rel in ("scripts/dram_counter_route.py", "scripts/dram_counter_route_decode3.py", "scripts/r3_decode.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == pins[rel] == REG["code_pins"][rel]
