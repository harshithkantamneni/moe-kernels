"""Rental 4's scaffolding off the GPU (scripts/plans/rental4-2026-10.plan): the driver's new
keys and steps, the counter route's eviction-hint key, gpu-benches, the rulers, and the
probe that runs the instrumented copy. The driver runs on test_gh200_model_session's FAKE
BOX, its fake interpreter extended with the three new tools."""
from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dram_counter_route as DCR  # noqa: E402
import gpubench as GB  # noqa: E402
import hw_rulers as HR  # noqa: E402
import instr_probe as IP  # noqa: E402
import private_weight_reference as PW  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.bench import exit_codes  # noqa: E402

PLAN4 = REPO / "scripts" / "plans" / "rental4-2026-10.plan"
DRIVER = TG.DRIVER
CARD = TG.CARD

FAKE_TOOLS = r'''
if tool == "instr_probe":
    if "--dry-run" in args:
        print("INSTRUMENTED KERNEL (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "cache-dir.txt").write_text(os.environ.get("TRITON_CACHE_DIR", ""))
    if val("--mode") == "perturb":
        vs = json.loads(Path(val("--variants")).read_text())
        bad = SC.get("gate_fail", [])
        (out / "gate.env").write_text("".join(
            f"GATE_{re.sub('[^A-Za-z0-9]', '_', v['id'])}={'FAIL' if v['id'] in bad else 'PASS'} median=0.001 worst=0.002\n"
            for v in vs))
        (out / "perturb.json").write_text("{}")
        sys.exit(1 if bad else 0)
    (out / "stamps.json").write_text("{}")
    sys.exit(0)
if tool == "gpubench":
    if "--dry-run" in args:
        print("GPU-BENCHES (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "constants.json").write_text("{}")
    sys.exit(0)
if tool == "hw_rulers":
    if "--dry-run" in args:
        print("RULERS (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "rulers.json").write_text("{}")
    sys.exit(0)
'''


def fakepy4() -> str:
    body = TG.FAKEPY.replace(
        'print(f"            n_decl = 9 against n_max = {T} read: planted")',
        'print(f"            n_decl = {val(\'--declared-copies\', \'9\')} against n_max = {T} read: planted")')
    assert body != TG.FAKEPY
    head, tail = body.rsplit("sys.exit(0)\n", 1)
    return head + FAKE_TOOLS + "sys.exit(0)\n" + tail


def box4(tmp_path, body, scenario=None, **kw):
    box = TG.Box(tmp_path, scenario or {})
    TG._exe(box.bin / "fakepy", fakepy4())
    f = tmp_path / "test.plan"
    f.write_text(body)
    return box, box.run("--plan", str(f), **kw)


def dry(*args):
    return subprocess.run(["bash", str(DRIVER), "--dry-run", *args], capture_output=True, text=True,
                          timeout=120, env={**__import__("os").environ, "HOME": "/nonexistent"})


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def test_the_rental4_plan_parses_in_its_order_with_paired_drop_groups_and_fits_three_hours():
    got = dry("--plan", str(PLAN4))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "a PLAN of 25 units" in got.stdout
    units = TG._units(got.stdout)
    steps = [s for _, _m, s, *_r in units]
    assert steps[:7] == ["prelude", "calibrate", "rulers", "gpubench", "timed", "timed", "perturb"]
    assert steps[7:16] == ["stamps"] * 9 and steps[16:23] == ["bytes"] * 7 and steps[23:] == ["timed"] * 2
    total = sum(int(e) for _, _m, _s, e, *_r in units)
    assert f"{total} min of units" in got.stdout and total <= 185, total   # about 3 h
    assert total == 144, total   # the eviction trio cut (build-r4-review E1)
    tags = [m.group(1) if (m := re.search(r"\[drop-group (\w+)\]", u[5])) else None for u in units]
    assert tags[4:6] == ["a1", "a1"] and tags[23:] == ["a2", "a2"]
    assert tags.count("st") == 9 and tags.count("b") == 7 and "ev" not in tags
    assert "instr-evict" not in got.stdout
    for u in units:
        if "declared-copies" in u[5]:
            assert u[2] == "timed"
    assert "mixtral-8x7b-tp2" not in got.stdout and "phi-3.5-moe" not in got.stdout   # A3 cut, Z left out
    assert "instrumented units, each run only on its perturb unit's PASS: u8-stf" in got.stdout
    dirs = [TG._unit_dir(u[5]) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)


def test_the_plans_variants_pass_the_probes_own_checks(tmp_path):
    got = subprocess.run(["bash", str(DRIVER), "--print-variants", "--plan", str(PLAN4)], capture_output=True,
                         text=True, timeout=120)
    assert got.returncode == exit_codes.REFUSED, got.stderr
    vs = json.loads(got.stdout)
    assert [v["id"] for v in vs] == ["u8-stf", "u9-stk64s4", "u10-stk32s4", "u11-stk128s4", "u12-stk64s8",
                                     "u13-sttail", "u14-sttail", "u15-stdead4", "u16-stdead2"]
    f = tmp_path / "v.json"
    f.write_text(got.stdout)
    for mode, extra in (("perturb", []), ("stamps", ["--variant", "u10-stk32s4"])):
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "instr_probe.py"), "--mode", mode,
                            "--variants", str(f), "--out", str(tmp_path / "o"), "--dry-run", *extra],
                           capture_output=True, text=True, timeout=120)
        if mode == "perturb":
            assert r.returncode == exit_codes.REFUSED and "REFUSED" not in r.stdout, r.stdout
            assert "9 variant(s)" in r.stdout
        else:   # no gate file given: a stamps unit refuses before anything
            assert "REFUSED: the perturbation gate for u10-stk32s4 reads nothing" in r.stdout
    loaded = IP.load_variants(f)
    assert {v["id"]: (v["block_k"], v["num_stages"]) for v in loaded}["u11-stk128s4"] == (128, 4)
    assert all(v["kind"] == "stamps" and not v["gate_spec"].hints for v in loaded)


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=a declared-copies=15\n", "declared-copies belongs to a timed unit, not bytes: a counter page declares R3's 9"),
    ("- prelude\nolmoe-1b-7b floor label=a floor-groups=64 declared-copies=15\n", "declared-copies belongs to a timed unit, not floor"),
    ("- prelude\nolmoe-1b-7b deep label=a declared-copies=15\n", "declared-copies belongs to a timed unit, not deep"),
    ("- prelude\nolmoe-1b-7b timed label=a declared-copies=0\n", "declared-copies 0"),
    ("- prelude\n- perturb\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-treads=2 stamp-arms=native instr-stamps=cta instr-evict-a=first\n",
     "instr-evict-a belongs to a bytes unit"),
    ("- prelude\nolmoe-1b-7b bytes label=b instr-stamps=cta\n", "instr-stamps belongs to a stamps unit"),
    ("- prelude\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-treads=2 stamp-arms=native instr-stamps=cta\n",
     "an instrumented unit needs a perturb unit before it"),
    ("- prelude\nolmoe-1b-7b bytes label=b byte-groups=2 instr-evict-b=last\n", "an instrumented unit needs a perturb unit before it"),
    ("- prelude\n- perturb\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-arms=native instr-stamps=cta\n",
     "a stamps unit needs stamp-treads"),
    ("- prelude\n- perturb\nolmoe-1b-7b timed label=a\n", "no instrumented unit after it"),
    ("- prelude\n- perturb\nolmoe-1b-7b stamps label=s stamp-groups=8 stamp-treads=2 stamp-arms=all instr-stamps=cta\n",
     "stamp-arms all"),
    ("- prelude\nolmoe-1b-7b bytes label=b instr-evict-a=keep\n", "first, last or none"),
    ("- prelude\nolmoe-1b-7b timed label=a block-k=32\n", "block-k belongs to a bytes unit (or a stamps unit), not timed"),
])
def test_rental4_keys_are_refused_off_their_step(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# the units on the fake box
# --------------------------------------------------------------------------

PLAN_SMALL = """- prelude
- rulers label=r4 est=3 cap=10
- gpubench label=r4 est=3 cap=10
olmoe-1b-7b timed label=c15 timed-groups=8 timed-treads=9 declared-copies=15 est=5 cap=10
olmoe-1b-7b timed label=c9 timed-groups=8 timed-treads=9 est=5 cap=10
- perturb label=pt est=3 cap=10
olmoe-1b-7b stamps label=stk stamp-groups=64 stamp-treads=4,6 stamp-arms=native instr-stamps=iter instr-marks=64 block-k=32 est=2 cap=5
mixtral-8x7b bytes label=evbase byte-groups=2 est=2 cap=5
mixtral-8x7b bytes label=evafirst byte-groups=2 instr-evict-a=first est=2 cap=5
"""


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return box4(tmp_path_factory.mktemp("r4"), PLAN_SMALL)


def test_the_units_run_in_order_and_every_lock_is_reset(small):
    box, got = small
    assert got.returncode == exit_codes.DONE, got.stdout + got.stderr + box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", box.ledger(), re.M)
    assert starts == ["prelude", "rulers", "gpubench", "timed", "timed", "perturb", "stamps", "bytes", "bytes"]
    locked = None
    for a in box.smi():
        if "-lgc" in a:
            assert locked is None
            locked = a[a.index("-lgc") + 1]
        if "-rgc" in a:
            locked = None
    assert locked is None


def test_declared_copies_reaches_r3_and_labels_the_page_directory(small):
    box, _ = small
    runs = box.tool("locked_r3")
    by_dir = {}
    for c in box.calls():
        if c.get("tool") == "locked_r3":
            by_dir[Path(c["results_dir"]).name] = TG.r3_side(c["argv"])
    assert TG.val(by_dir[f"gaps-{CARD}-olmoe-1b-7b-c15"], "--declared-copies") == "15"
    assert TG.val(by_dir[f"gaps-{CARD}-olmoe-1b-7b-c9"], "--declared-copies") == "9"
    lab = (box.results / f"gaps-{CARD}-olmoe-1b-7b-c15" / "DECLARED_COPIES.txt").read_text()
    assert "NOT JOINABLE TO COUNTER BYTES" in lab
    assert not (box.results / f"gaps-{CARD}-olmoe-1b-7b-c9" / "DECLARED_COPIES.txt").exists()
    assert "may not lock" not in box.ledger() and len(runs) == 2
    for a in box.tool("private_weight_reference", dry=True):
        PW.build_parser().parse_args(a)


def test_the_perturb_unit_gates_the_instrumented_units_and_they_run_on_its_pass(small):
    box, _ = small
    ip = box.tool("instr_probe", dry=False)
    assert [TG.val(a, "--mode") for a in ip] == ["perturb", "stamps"]
    v = json.loads(Path(TG.val(ip[0], "--variants")).read_text())
    assert [x["id"] for x in v] == ["u7-stk", "u9-evafirst"]
    assert v[0]["block_k"] == 32 and v[0]["spec"] == "stamps=iter,every=1,marks=64"
    assert v[1]["spec"] == "evict_a=first,evict_b=none" and v[1]["groups"] == [2]
    assert TG.val(ip[1], "--variant") == "u7-stk" and TG.val(ip[1], "--gate").endswith("instr-gate.env")
    for a in ip:
        IP.build_parser().parse_args(a)
    pages = [a for a in box.tool("dram_counter_route") if "--run" in a and "--census-only" not in a]
    instr = [TG.val(a, "--instr") if "--instr" in a else None for a in pages]
    assert instr == [None, "evict_a=first,evict_b=none"]
    for a in pages:
        DCR.build_parser().parse_args(a)
    # the plain installed kernel stays the object of every timed unit
    for a in box.tool("locked_r3"):
        assert "--instr" not in a
    assert "perturb: GATE_u7_stk=PASS" in box.ledger()
    # each unit compiles into its own Triton cache (build-r4-review G1)
    for name in ("perturb-pt", "instr-olmoe-1b-7b-stk"):
        d = next(p for p in box.results.iterdir() if p.name.endswith(name))
        assert (d / "cache-dir.txt").read_text() == str(d / "triton-cache")


def test_gpubench_and_the_rulers_run_under_the_lock_outside_the_checkout(small):
    box, _ = small
    g = box.tool("gpubench", dry=False)[0]
    assert TG.val(g, "--dest") == str(box.moe / "ext" / "gpu-benches") and TG.val(g, "--lock-mhz") == "1710"
    GB.build_parser().parse_args(g)
    r = box.tool("hw_rulers", dry=False)[0]
    HR.build_parser().parse_args(r)
    assert TG.val(r, "--lock-mhz") == "1710"
    assert (box.results / next(p.name for p in box.results.iterdir() if p.name.endswith("gpubench-r4"))).is_dir()


def test_a_failed_gate_refuses_its_unit_in_no_time(tmp_path):
    box, got = box4(tmp_path, PLAN_SMALL, {"gate_fail": ["u7-stk", "u9-evafirst"]})
    led = box.ledger()
    assert "stamps REFUSED: the perturbation gate for u7-stk reads 'FAIL" in led
    assert "bytes REFUSED: the perturbation gate for u9-evafirst" in led
    assert [TG.val(a, "--mode") for a in box.tool("instr_probe", dry=False)] == ["perturb"]
    pages = [a for a in box.tool("dram_counter_route") if "--run" in a and "--census-only" not in a]
    assert all("--instr" not in a for a in pages) and len(pages) == 1
    assert got.returncode == exit_codes.INVALID


# --------------------------------------------------------------------------
# the counter route's eviction-hint key
# --------------------------------------------------------------------------

def _plan(**kw):
    return DCR.r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=32, block_n=64, num_stages=4, group_m=2,
                       treads=[1, 2, 3], kind="measure", arms=PW.ARMS, calls=3, warmups=2,
                       profile_dir=Path("p"), stem="g2", **kw)


def test_r3_plan_carries_instr_only_when_given_and_the_design_records_it():
    assert "instr" not in _plan()
    p = _plan(instr="evict_a=first")
    assert p["instr"] == "evict_a=first"
    d = DCR.r3_design(p)
    assert d["instr"]["evict_a"] == "first" and d["instr"]["stamps"] == 0
    assert "instr" not in DCR.r3_design(_plan())


@pytest.mark.parametrize("spec, why", [("stamps=cta", "carries stamps"), ("evict_a=none", "turns nothing on"),
                                       ("evict_a=keep", "evict_a")])
def test_a_counter_plan_refuses_stamps_and_an_all_off_spec(spec, why):
    with pytest.raises(PW.CounterPlanRefused, match=why):
        _plan(instr=spec)


def test_the_counter_child_installs_the_copy_around_its_calls_and_restores_it():
    class Mod:
        pass

    class K:
        def __init__(self):
            self.kw = []

        def __getitem__(self, grid):
            return lambda *a, **kw: self.kw.append(kw)
    mod, plain, copy = Mod(), K(), K()
    mod.fused_moe_kernel = plain
    with PW.counter_instr({"instr": "evict_b=last"}, module=mod, kernel=copy) as rec:
        assert isinstance(mod.fused_moe_kernel, I._Wrapper)
        TG_args = ["A", "B", "C", None, None, None, None, None, None, None, 64, 64, 64, 64]
        mod.fused_moe_kernel[(1,)](*TG_args, BLOCK_SIZE_M=32, BLOCK_SIZE_N=64)
    assert mod.fused_moe_kernel is plain and copy.kw[0]["EVICT_B"] == "evict_last" and rec.launches
    with PW.counter_instr({}, module=mod) as none:
        assert none is None and mod.fused_moe_kernel is plain


def _main(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = DCR.main(list(argv))
    return rc, buf.getvalue()


def test_instr_belongs_to_a_page_and_moves_the_run_id():
    rc, out = _main("--dry-run", "--family", "r3-arms", "--floor", "--instr", "evict_a=first")
    assert rc == exit_codes.REFUSED and "--instr belongs to a page" in out

    def rid(*extra):
        a = DCR.build_parser().parse_args(["--family", "r3-arms", "--run", *extra])
        DCR.resolve_r3_defaults(a, ["--family", "r3-arms", "--run", *extra])
        return DCR.run_id_for("r3-run", a, "card")
    assert rid() != rid("--instr", "evict_a=first") != rid("--instr", "evict_b=last")


# --------------------------------------------------------------------------
# gpu-benches and the rulers
# --------------------------------------------------------------------------

LAT = "".join(f"  1000000  1710 {kb:8.1f}  1.0  {c:.1f} {c:.1f} {c:.1f} {c:.1f}\n"
              for kb, c in ((16, 32.0), (8_000, 242.0), (20_000, 242.0), (40_000, 430.0), (50_000, 436.0),
                            (160_000, 590.0), (400_000, 594.0)))
STREAM = "block smBlocks   threads    occ%   |  init read scale triad 3pt 5pt\n" + "".join(
    f" {b:4d}  {b * 2}  2  {o:5.1f}%     |  GB/s:  {i} {r} {s} {t} {s} {s}\n"
    for b, o, i, r, s, t in ((64, 6.2, 590, 245, 464, 798), (512, 50.0, 3718, 1715, 2269, 3338),
                             (768, 75.0, 3941, 2314, 2854, 3686), (1024, 100.0, 3942, 2775, 3174, 3783)))
L2C = "".join(f"       512 kB {kb:10.0f} kB        5ms       0.1% {bw:10.1f} GB/s        0 GB/s      0 GB/s\n"
              for kb, bw in ((3_000, 10_200.0), (9_000, 10_100.0), (40_000, 9_100.0), (70_000, 4_500.0),
                             (300_000, 3_890.0), (600_000, 3_888.0)))


def test_the_gpubench_rules_read_their_windows():
    lat = GB.parse_latency(LAT, 60 * GB.MIB)
    assert lat["near_l2_ns"] == pytest.approx(242 / 1.71, rel=1e-6)
    assert lat["far_l2_ns"] == pytest.approx(433 / 1.71, rel=1e-6) and lat["cells"]["far"] == 2
    assert lat["dram_ns"] == pytest.approx(592 / 1.71, rel=1e-6)
    st = GB.parse_stream(STREAM)
    assert st["peak_gbps"]["triad"] == 3783 and st["triad_knee_occupancy"] == 0.75
    l2 = GB.parse_l2cache(L2C, 60 * GB.MIB)
    assert l2["plateau_gbps"] == pytest.approx(10_150) and l2["floor_gbps"] == pytest.approx(3_889)
    assert 40_000 * 1024 < l2["half_way_bytes"] < 70_000 * 1024
    c = GB.constants({"gpu-latency": LAT, "gpu-stream": STREAM}, l2_bytes=60 * GB.MIB, lock_mhz=1710)
    assert c["cycles_at_lock"]["far_l2_ns"] == pytest.approx(433) and c["errors"] == {"gpu-l2-cache": "no output"}
    with pytest.raises(ValueError):
        GB.parse_stream("nothing here")


def test_gpubench_is_pinned_fetched_outside_the_checkout_and_its_dry_run_runs_nothing(tmp_path):
    assert re.fullmatch(r"[0-9a-f]{40}", GB.PIN["commit"]) and GB.PIN["licence"] == "GPL-3.0"
    cmds = GB.commands(tmp_path / "gb", 90)
    assert cmds[0][:2] == ["git", "clone"] and ["git", "-C", str(tmp_path / "gb"), "checkout", "--quiet",
                                                 "--detach", GB.PIN["commit"]] in cmds
    cmds = GB.commands(tmp_path / "gb", 90, home="/opt/cuda-12.8")
    assert [c for c in cmds if c[0] == "make"] == [["make", "-C", str(tmp_path / "gb" / d), "SM=90",
                                                    "CUDA_HOME=/opt/cuda-12.8", t] for d, t, _ in GB.BENCHES]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = GB.main(["--dest", str(REPO / "ext"), "--out", str(tmp_path / "o")])
    assert rc == exit_codes.REFUSED and "inside a git checkout" in buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()):
        assert GB.main(["--dest", str(tmp_path / "gb"), "--out", str(tmp_path / "o"), "--dry-run"]) == exit_codes.REFUSED
    assert not (tmp_path / "gb").exists() and not (tmp_path / "o").exists()
    assert not list(REPO.glob("**/gpu-latency/main.cu"))   # nothing of it vendored


def test_the_rulers_plan_and_rates():
    p = HR.plan(4.0)
    assert p["copy"]["bytes"] == 2 * p["read2d"]["bytes"] and p["matmul"]["flop"] == 2 * 8192 ** 3
    r = HR.rates({"read2d": 1.0, "copy": 2.0, "add": None, "matmul": 1.0}, p)
    assert r["read2d"]["gbps"] == pytest.approx(p["read2d"]["bytes"] / 1e-3 / 1e9) and r["add"] is None
    with contextlib.redirect_stdout(io.StringIO()):
        assert HR.main(["--out", "/nonexistent/x", "--dry-run"]) == exit_codes.REFUSED


# --------------------------------------------------------------------------
# the probe's pure parts
# --------------------------------------------------------------------------

def _v(**kw):
    v = {"id": "u1-x", "kind": "stamps", "model": "olmoe-1b-7b", "groups": [64], "treads": [4, 6],
         "arms": ["native"], "block_k": 64, "num_stages": 4, "spec": "stamps=iter,marks=32"}
    v.update(kw)
    return v


@pytest.mark.parametrize("kw, why", [
    ({"spec": "stamps=cta,evict_a=first"}, "carries stamps and no hint"),
    ({"kind": "bytes", "spec": "stamps=cta"}, "carries hints and no stamps"),
    ({"treads": [12], "arms": ["native", "shared"]}, "not on R3's ladder"),
    ({"block_k": 48}, "cannot run"),
    ({"arms": ["all"]}, "arms"),
    ({"model": "nope"}, "unknown model"),
    ({"kind": "floor"}, "kind"),
])
def test_a_bad_variant_is_refused(kw, why):
    with pytest.raises(IP.Refused, match=why):
        IP.check_variant(_v(**kw))


def test_a_byte_variants_gate_is_the_all_off_copy_on_three_treads_of_every_arm():
    v = IP.check_variant(_v(kind="bytes", model="mixtral-8x7b", groups=[2], treads=list(range(1, 10)),
                            arms=list(PW.ARMS), spec="evict_b=last"))
    assert v["gate_spec"].off and v["gate_cells"] == [(2, a, n) for a in PW.ARMS for n in (1, 5, 9)]
    s = IP.check_variant(_v())
    assert s["gate_spec"].stamps == 2 and not s["gate_spec"].hints and s["gate_cells"] == s["cells"]


def _facts(regs, shared, occ):
    return {"regs": regs, "shared": shared, "occupancy": {"ctas_per_sm": occ}}


def test_the_gate_pairs_by_config_and_needs_every_leg(tmp_path):
    k1, k2 = (32, 64, 64, 8, 4, 8, False, 8), (32, 64, 64, 8, 4, 8, True, 8)
    plain = {k1: _facts(48, 36864, 5), k2: _facts(55, 36864, 4)}
    same = IP.compare_configs(plain, dict(plain))
    assert all(c["equal"] for c in same)
    ok = dict(sass_equal=True)
    assert IP.gate_verdict([1.004, 0.995], same, **ok)["verdict"] == "PASS"
    # a swap of the w1 / w2 occupancies passes a multiset check and fails the pairing
    swapped = IP.compare_configs(plain, {k1: _facts(55, 36864, 4), k2: _facts(48, 36864, 5)})
    assert IP.gate_verdict([1.0], swapped, **ok)["verdict"] == "FAIL"
    regs_only = IP.compare_configs(plain, {k1: _facts(50, 36864, 5), k2: plain[k2]})
    assert "config" in IP.gate_verdict([1.0], regs_only, **ok)["why"][0]
    missing = IP.compare_configs(plain, {k1: plain[k1]})
    assert "not paired" in IP.gate_verdict([1.0], missing, **ok)["why"][0]
    assert IP.gate_verdict([1.004, 1.0, 0.999, 1.021], same, **ok)["why"][0].startswith("worst")
    assert IP.gate_verdict([1.0], same, upstream_ok=False, **ok)["verdict"] == "FAIL"
    assert IP.gate_verdict([None, 1.0], same, **ok)["verdict"] == "FAIL"
    assert "SASS leg unread" in IP.gate_verdict([1.0], same)["why"][0]
    assert "SASS differs" in IP.gate_verdict([1.0], same, sass_equal=False)["why"][0]


def test_a_hinted_unit_is_refused_unless_the_ptx_carries_the_l2_hint():
    k = (32, 64, 64, 2, 4, 8, True, 2)
    cfg = IP.compare_configs({k: _facts(55, 36864, 4)}, {k: _facts(55, 36864, 4)})
    v = IP.gate_verdict([1.0], cfg, sass_equal=True, hint_ok=False)
    assert v["verdict"] == "FAIL" and "dropped the eviction hint" in v["why"][0]
    assert IP.gate_verdict([1.0], cfg, sass_equal=True, hint_ok=True)["verdict"] == "PASS"

    class CK:   # and the counter child refuses such a page: no manifest
        def __init__(self, ptx):
            self.asm, self.metadata = {"ptx": ptx, "cubin": None}, {"shared": 36864, "num_warps": 8}
    rec = I.Recorder(I.parse_spec("evict_b=last"), [{"compiled": CK("cp.async.cg.shared.global [%r1], [%rd2], 16;")}])
    assert "dropped the eviction hint" in PW.counter_hint_refusal(rec)
    rec.launches[0]["compiled"] = CK("cp.async.cg.shared.global.L2::cache_hint [%r1], [%rd2], 16, %rd9;")
    assert PW.counter_hint_refusal(rec) == ""
    assert "no compiled kernel" in PW.counter_hint_refusal(I.Recorder(I.parse_spec("evict_a=first"), [{}]))
    assert PW.counter_hint_refusal(None) == ""


def test_the_gate_file(tmp_path):
    f = tmp_path / "gate.env"
    f.write_text("GATE_u1_x=FAIL m\nGATE_u1_x=PASS m\nGATE_u2_y=PASS\n")
    assert IP.read_gate(f, "u2-y") == "PASS" and IP.read_gate(f, "u3") == ""
    assert IP.split_gemms([1, 2, 3, 4]) == {"w1": [1, 3], "w2": [2, 4]}
    with pytest.raises(ValueError):
        IP.split_gemms([1, 2, 3])
    dup = tmp_path / "v.json"
    dup.write_text(json.dumps([_v(), _v()]))
    with pytest.raises(IP.Refused, match="duplicate"):
        IP.load_variants(dup)
