"""The instrumented copy of vLLM v0.27.1's fused_moe kernel (rental 4, owner decision 5).

The copy (`moe/instrumented/fused_moe_instr.py`) may add ONLY timestamps and the two
eviction hints, both off by default, and change no tiling, scheduling or optimisation.
These tests hold that without Triton or vLLM (neither is in the laptop's interpreter):
the copy is read as text and as an AST.

1. The excerpt of upstream the copy is diffed against is the upstream file's own lines
   (sha256 in UPSTREAM.json), and the published pages' compiled TTIR names those lines.
2. Every copy line that is not an upstream line carries `# INSTR`; the only upstream line
   replaced is the B load (split to carry its eviction_policy); nothing upstream is deleted.
3. With every `if STAMPS ...:` block, the six instrument parameters and the two
   `eviction_policy=` keywords removed, the copy's two functions are AST-equal to upstream.
4. The instruments are off by default: STAMPS 0, both hints "" (tl.load's default).
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import moe.instrumented as I  # noqa: E402
from moe.instrumented import cubin as C  # noqa: E402

PKG = REPO / "moe" / "instrumented"
UP = json.loads((PKG / "UPSTREAM.json").read_text())
EXCERPT = (PKG / "upstream_v0.27.1_fused_moe.excerpt").read_text()
COPY = (PKG / "fused_moe_instr.py").read_text()
FUNCS = ("write_zeros_to_output", "fused_moe_kernel")
INSTR_PARAMS = ("stamps_ptr", "STAMPS", "STAMP_EVERY", "STAMP_MARKS", "EVICT_A", "EVICT_B")
#: the one upstream line the copy replaces (split over INSTR lines to carry the hint)
EDITED = ["            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)\n"]


def _funcs(src: str) -> dict:
    tree = ast.parse(src)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS}


def _copy_body() -> str:
    """The copy's text from the first upstream function on (the instruments' own helpers
    sit above it)."""
    i = COPY.index("@triton.jit\ndef write_zeros_to_output(")
    return COPY[i:]


class _Strip(ast.NodeTransformer):
    def visit_If(self, node):
        if any(isinstance(n, ast.Name) and n.id == "STAMPS" for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords
                         if not (k.arg == "eviction_policy" and isinstance(k.value, ast.Name)
                                 and k.value.id in ("EVICT_A", "EVICT_B"))]
        return self.generic_visit(node)


def _strip(fn: ast.FunctionDef) -> ast.FunctionDef:
    fn = copy.deepcopy(fn)
    a = fn.args
    n_def = len(a.defaults)
    keep, defaults = [], []
    for i, arg in enumerate(a.args):
        d = a.defaults[i - (len(a.args) - n_def)] if i >= len(a.args) - n_def else None
        if arg.arg in INSTR_PARAMS:
            continue
        keep.append(arg)
        if d is not None:
            defaults.append(d)
    a.args, a.defaults = keep, defaults
    return _Strip().visit(fn)


# --------------------------------------------------------------------------
# 1. the excerpt is upstream's
# --------------------------------------------------------------------------

def test_the_excerpt_is_the_recorded_upstream_lines():
    assert hashlib.sha256(EXCERPT.encode()).hexdigest() == UP["excerpt_sha256"]
    assert UP["tag"] == "v0.27.1" and UP["file_lines"] > 1800
    assert UP["file_sha256"] == "d5955a3460746b66740b024f470aeca79bbebd10872c327f765f2d7fcb805f28"
    fs = _funcs(EXCERPT)
    assert set(fs) == set(FUNCS)
    # the excerpt is two whole functions in upstream's order, nothing else
    assert EXCERPT.startswith("@triton.jit\ndef write_zeros_to_output(")
    assert EXCERPT.rstrip().endswith("tl.store(c_ptrs, accumulator, mask=c_mask)")


def test_installed_check_reads_a_whole_upstream_file_and_refuses_another(tmp_path):
    """The VM's check of its installed fused_moe.py: rebuild a file around the excerpt at
    the recorded lines; the excerpt sha matches, the file sha does not (it is not upstream's)."""
    a0, a1 = UP["functions"]["write_zeros_to_output"]
    b0, b1 = UP["functions"]["fused_moe_kernel"]
    wz, k = EXCERPT.split("\n\n\n", 1)
    lines = ["# filler\n"] * (a0 - 1) + (wz + "\n").splitlines(True)
    lines += ["# filler\n"] * (b0 - 1 - len(lines)) + k.splitlines(True)
    lines += ["# filler\n"] * (UP["file_lines"] - len(lines))
    f = tmp_path / "fused_moe.py"
    f.write_text("".join(lines))
    got = I.installed_check(f)
    assert got["excerpt_matches"] is True and got["file_matches"] is False
    f.write_text("".join(lines).replace("other=0.0)", "other=1.0)"))
    assert I.installed_check(f)["excerpt_matches"] is False


def test_the_published_ttir_names_the_excerpts_own_lines():
    """Every page's compiled kernel (its triton-cache `.source`, Triton 3.7.1 TTIR) names
    fused_moe.py lines: def fused_moe_kernel at 299, write_zeros_to_output at 45, and every
    statement line it cites lies in the excerpt's ranges on a non-blank upstream line."""
    hits = sorted((REPO / "results" / "published").glob(
        "2026-10-06-*/session/*/census.triton-cache/*/fused_moe_kernel.source"))
    assert hits, "no published rental-3 census cache"
    src = hits[0].read_text()
    locs = {(int(a), int(b)) for a, b in re.findall(r'fused_moe/fused_moe\.py":(\d+):(\d+)', src)}
    assert (299, 0) in locs and (45, 0) in locs
    a0, a1 = UP["functions"]["write_zeros_to_output"]
    b0, b1 = UP["functions"]["fused_moe_kernel"]
    wz, k = EXCERPT.split("\n\n\n", 1)
    up = {a0 + i: ln for i, ln in enumerate(wz.split("\n"))}
    up.update({b0 + i: ln for i, ln in enumerate(k.split("\n"))})
    for line, col in locs:
        assert a0 <= line <= a1 or b0 <= line <= b1, line
        assert up[line].strip(), (line, "a blank upstream line")
        assert col <= len(up[line]), (line, col, up[line])


# --------------------------------------------------------------------------
# 2. and 3. the copy is upstream plus marked instruments
# --------------------------------------------------------------------------

def test_every_line_the_copy_adds_carries_the_marker_and_only_the_b_load_is_replaced():
    up = EXCERPT.splitlines(True)
    cp = _copy_body().splitlines(True)
    sm = difflib.SequenceMatcher(a=up, b=cp, autojunk=False)
    replaced, added = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        assert op != "delete", f"upstream lines deleted: {up[i1:i2]}"
        new = cp[j1:j2]
        unmarked = [ln for ln in new if "# INSTR" not in ln]
        assert not unmarked, f"copy lines without the marker: {unmarked}"
        added += new
        replaced += up[i1:i2]
    assert replaced == EDITED, replaced
    assert len(added) >= 20


def test_with_the_instruments_removed_the_copy_is_upstreams_ast():
    up, cp = _funcs(EXCERPT), _funcs(_copy_body())
    assert set(cp) == set(FUNCS)
    for name in FUNCS:
        a = ast.dump(up[name], include_attributes=False)
        b = ast.dump(_strip(cp[name]), include_attributes=False)
        assert a == b, name
    # and with them in, it is not (the strip is doing work)
    assert ast.dump(up["fused_moe_kernel"]) != ast.dump(cp["fused_moe_kernel"])


def test_the_instruments_are_off_by_default_and_appended_after_upstreams_parameters():
    fn = _funcs(_copy_body())["fused_moe_kernel"]
    names = [a.arg for a in fn.args.args]
    assert names[-len(INSTR_PARAMS):] == list(INSTR_PARAMS)
    up_names = [a.arg for a in _funcs(EXCERPT)["fused_moe_kernel"].args.args]
    assert names[:len(up_names)] == up_names and up_names[-1] == "USE_TD"
    defaults = dict(zip(names[-len(fn.args.defaults):],
                        [ast.literal_eval(d) for d in fn.args.defaults], strict=True))
    assert defaults == {"USE_TD": False, "stamps_ptr": None, "STAMPS": 0, "STAMP_EVERY": 1,
                        "STAMP_MARKS": 0, "EVICT_A": "", "EVICT_B": ""}
    # the eviction hints ride only on the default path's A and B loads
    loads = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "load"
             and any(k.arg == "eviction_policy" for k in n.keywords)]
    assert [ast.unparse(n.args[0]) for n in loads] == ["a_ptrs", "b_ptrs"]
    assert {ast.unparse(k.value) for n in loads for k in n.keywords if k.arg == "eviction_policy"} \
        == {"EVICT_A", "EVICT_B"}


def test_every_stamp_block_is_a_constexpr_if_on_stamps():
    fn = _funcs(_copy_body())["fused_moe_kernel"]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "").startswith("_stamp")]
    assert len(calls) >= 9
    parents = {}
    for node in ast.walk(fn):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node
    for c in calls:
        p = c
        while p in parents and not (isinstance(p, ast.If) and "STAMPS" in ast.unparse(p.test)):
            p = parents[p]
        assert isinstance(p, ast.If), ast.unparse(c)
        assert re.fullmatch(r"STAMPS >= [12]", ast.unparse(p.test)), ast.unparse(p.test)


def test_the_timers_are_the_ptx_special_registers_and_the_row_layout_matches_the_decoder():
    assert 'mov.u64 $0, %globaltimer;' in COPY and 'mov.u64 $0, %clock64;' in COPY
    assert 'mov.u32 $0, %smid;' in COPY and COPY.count("tl.inline_asm_elementwise(") == 3
    assert re.search(r"^STAMP_HDR = tl\.constexpr\((\d+)\)$", COPY, re.M).group(1) == str(I.HDR)
    cols = {int(c) for c in re.findall(r"_stamp_pair\(stamps_ptr, pid, STAMP_MARKS, (\d+)\)", COPY)}
    assert cols == {3, 5, 7}
    assert I.COLUMNS[3:9] == ("prologue_ns", "prologue_clk", "epi_ns", "epi_clk", "end_ns", "end_clk")
    assert "tl.store(row + 9," in COPY and I.COLUMNS[9] == "kind"
    assert "import vllm" not in COPY and "from vllm" not in COPY


# --------------------------------------------------------------------------
# the spec, the installer and the decoder
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, kw", [
    ("", {"STAMPS": 0, "STAMP_EVERY": 1, "STAMP_MARKS": 0, "EVICT_A": "", "EVICT_B": ""}),
    ("stamps=cta", {"STAMPS": 1, "STAMP_EVERY": 1, "STAMP_MARKS": 0, "EVICT_A": "", "EVICT_B": ""}),
    ("stamps=iter,every=2,marks=8,evict_b=last",
     {"STAMPS": 2, "STAMP_EVERY": 2, "STAMP_MARKS": 8, "EVICT_A": "", "EVICT_B": "evict_last"}),
    ("evict_a=first", {"STAMPS": 0, "STAMP_EVERY": 1, "STAMP_MARKS": 0, "EVICT_A": "evict_first",
                       "EVICT_B": ""}),
])
def test_a_spec_is_parsed_into_the_copys_constexprs(text, kw):
    s = I.parse_spec(text)
    assert s.kernel_kwargs() == kw
    assert I.parse_spec(s.text()) == s
    assert s.off is (text == "")


@pytest.mark.parametrize("text", ["stamps=all", "evict_a=keep", "colour=red", "every=0,stamps=iter",
                                  "stamps", "marks=x"])
def test_a_bad_spec_is_refused(text):
    with pytest.raises(I.SpecError):
        I.parse_spec(text)


class _FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.calls.append((grid, args, kw))
        return launch


def _upstream_call(module):
    """invoke_fused_moe_triton_kernel's launch, as upstream writes it."""
    args = ["A", "B", "C", None, None, None, "tw", "sorted", "eids", "ntpp", 1024, 2048, 4160, 4096]
    args += [0] * 18
    cfg = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1,
           "num_warps": 8, "num_stages": 4}
    grid = lambda META: (-(-4160 // META["BLOCK_SIZE_M"]) * -(-1024 // META["BLOCK_SIZE_N"]),)  # noqa: E731
    module.fused_moe_kernel[grid](*args, MUL_ROUTED_WEIGHT=False, top_k=8, BLOCK_SIZE_K=64,
                                  USE_TD=False, **cfg)


def test_install_routes_each_launch_to_the_copy_and_restores_the_plain_kernel():
    class Mod:
        pass
    mod, plain, instr = Mod(), _FakeKernel(), _FakeKernel()
    mod.fused_moe_kernel = plain
    bufs = []

    def alloc(ctas, width, like):
        bufs.append((ctas, width, like))
        return f"buf{len(bufs)}"
    spec = I.parse_spec("stamps=iter,marks=32")
    with I.install(spec, module=mod, kernel=instr, alloc=alloc) as rec:
        _upstream_call(mod)
        _upstream_call(mod)
    assert mod.fused_moe_kernel is plain and not plain.calls
    assert len(instr.calls) == 2 and len(rec.launches) == 2
    _grid, args, kw = instr.calls[0]
    assert kw["STAMPS"] == 2 and kw["STAMP_MARKS"] == 32 and kw["stamps_ptr"] == "buf1"
    assert kw["BLOCK_SIZE_K"] == 64 and args[10:14] == (1024, 2048, 4160, 4096)
    ctas = 130 * 16
    assert bufs[0] == (ctas, I.HDR + 64, "A")
    assert rec.launches[0]["ctas"] == ctas and rec.launches[1]["launch"] == 1


def test_hints_only_allocate_nothing_and_an_exception_still_restores():
    class Mod:
        pass
    mod, plain, instr = Mod(), _FakeKernel(), _FakeKernel()
    mod.fused_moe_kernel = plain
    with pytest.raises(RuntimeError):
        with I.install(I.parse_spec("evict_a=first"), module=mod, kernel=instr,
                       alloc=lambda *a: pytest.fail("no buffer for hints")) as rec:
            _upstream_call(mod)
            raise RuntimeError("boom")
    assert mod.fused_moe_kernel is plain
    kw = instr.calls[0][2]
    assert kw["EVICT_A"] == "evict_first" and "stamps_ptr" not in kw and rec.launches[0]["buffer"] is None


def test_install_refuses_to_nest():
    class Mod:
        pass
    mod = Mod()
    mod.fused_moe_kernel = _FakeKernel()
    with I.install(I.InstrSpec(), module=mod, kernel=_FakeKernel(), alloc=lambda *a: None):
        with pytest.raises(I.SpecError):
            with I.install(I.InstrSpec(), module=mod, kernel=_FakeKernel(), alloc=lambda *a: None):
                pass


def test_decode_and_phases_read_a_synthetic_row():
    import numpy as np
    marks = 3
    row = [17, 1000, 5000, 1100, 5200, 1900, 6700, 2000, 7100, 1] + [5300, 5700, 5800, 6200, 6300, 6650]
    dead = [3, 1000, 4000, -1, -1, -1, -1, 1400, 4400, 2] + [-1] * 6
    cols = I.decode(np.array([row, dead]), marks)
    assert list(cols["smid"]) == [17, 3] and list(cols["kind"]) == [1, 2]
    ph = I.phases(cols)
    assert list(ph["prologue"]) == [200, -1] and list(ph["loop"]) == [1500, -1]
    assert list(ph["epilogue"]) == [400, -1] and list(ph["life"]) == [2100, 400]
    assert ph["iter_cycles"].tolist() == [[500, 500], [-1, -1]]
    with pytest.raises(ValueError):
        I.decode(np.zeros((2, 11)), 0)


# --------------------------------------------------------------------------
# the occupancy leg of the gate
# --------------------------------------------------------------------------

def test_cubin_registers_and_occupancy_reproduce_the_published_limits():
    """Rental 2/3 GH200 cubins of 32x64 w8: registers and shared give ncu's occupancy
    (k64s4 5 / 4, BK 32 s4 5 / 5, BK 128 s4 3 / 3, s6 3 / 3, s8 2 / 2; REVIEW.md section 0)."""
    want = {(4, 36864): {48: 5, 55: 4}, (4, 18432): {47: 5, 48: 5}, (4, 73728): {57: 3, 64: 3},
            (6, 61440): {48: 3, 55: 3}, (8, 86016): {48: 2, 55: 2}, (2, 24576): {56: 4, 63: 4}}
    seen = set()
    for e in C.cache_entries(REPO / "results" / "published" / "2026-10-02-nvidia_gh200_480gb-rental2-session"):
        key = (e["num_stages"], e["shared"])
        if e["num_warps"] == 8 and key in want and e["regs"] in want[key]:
            assert e["occupancy"]["ctas_per_sm"] == want[key][e["regs"]], e
            assert e["origin"] == "plain" and not e["timer_in_ptx"]
            seen.add((key, e["regs"]))
    assert len(seen) >= 8, seen


def test_occupancy_rule_limits():
    assert C.occupancy(48, 36864, 8)["ctas_per_sm"] == 5
    assert C.occupancy(55, 36864, 8) == {"ctas_per_sm": 4, "binding": ["registers"],
                                         "limits": {"registers": 4, "shared": 6, "warps": 8, "ctas": 32}}
    assert C.occupancy(48, 86016, 8)["binding"] == ["shared"]
    with pytest.raises(ValueError):
        C.regcount(b"not an elf")


# --------------------------------------------------------------------------
# the review's legs: compiled facts, the L2 hint in PTX, SASS (build-r4-review)
# --------------------------------------------------------------------------

def test_each_launch_records_its_compiled_kernel_and_config():
    class Mod:
        pass

    class K(_FakeKernel):
        def __getitem__(self, grid):
            inner = super().__getitem__(grid)
            return lambda *a, **kw: (inner(*a, **kw), "compiled-object")[1]
    mod = Mod()
    mod.fused_moe_kernel = _FakeKernel()
    with I.install(I.InstrSpec(), module=mod, kernel=K(), alloc=lambda *a: None) as rec:
        _upstream_call(mod)
    e = rec.launches[0]
    assert e["compiled"] == "compiled-object"
    assert I.config_key(e) == (32, 64, 64, 8, 4, 8, False, 8)


def test_compiled_facts_read_a_published_cubin():
    cub = next(p for p in (REPO / "results" / "published" / "2026-10-02-nvidia_gh200_480gb-rental2-session").rglob(
        "fused_moe_kernel.cubin") if (p.with_suffix(".json")).exists())
    md = json.loads(cub.with_suffix(".json").read_text())

    class CK:
        asm = {"cubin": cub.read_bytes(), "ptx": cub.with_suffix(".ptx").read_text()}
        metadata = md
    f = C.compiled_facts(CK())
    assert f["regs"] == C.regcount(cub.read_bytes()) and f["shared"] == md["shared"]
    assert f["occupancy"]["ctas_per_sm"] >= 1
    assert C.hint_in_ptx(f["ptx"]) is False      # the plain kernel carries no L2 hint
    assert C.compiled_facts(None) is None


@pytest.mark.parametrize("ptx, want", [
    ("cp.async.cg.shared.global.L2::cache_hint [%r1], [%rd2], 16, %rd9;", True),
    ("ld.global.L1::evict_last.L2::cache_hint.v4.b32 {%r1}, [%rd1], %rd7;", True),
    ("cp.async.cg.shared.global [%r1], [%rd2], 16;", False),
    ("ld.global.L1::evict_first.v4.b32 {%r1}, [%rd1];", False),     # an L1 policy is not an L2 hint
    ("createpolicy.fractional.L2::evict_first.b64 %rd9, 1.0;", False),
])
def test_the_hint_leg_reads_l2_cache_hint_on_the_copies_only(ptx, want):
    assert C.hint_in_ptx(ptx) is want


SASS = """
        /*0000*/                   LDC R1, c[0x0][0x28] ;                       /* 0x00000a00ff017b82 */
        /*0010*/                   CS2R R4, SR_CLOCKLO ;                        /* 0x0000000000047805 */
        /*0020*/                   LDG.E R2, desc[UR4][R2.64] ;
        /*0100*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0110*/                   CS2R R6, SR_CLOCKLO ;
        /*0120*/                   HMMA.16816.F32.BF16 R8, R12, R16, R8 ;
        /*0130*/                   CS2R R6, SR_CLOCKLO ;
        /*0140*/              @P0  BRA 0x100 ;
        /*0150*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0160*/                   CS2R R20, SR_CLOCKLO ;
        /*0170*/                   LDG.E R22, desc[UR4][R24.64] ;
        /*0180*/                   STG.E desc[UR4][R2.64], R8 ;
        /*0190*/                   EXIT ;
"""


def test_sass_checks_find_the_brackets_and_a_misplaced_read():
    ch = C.sass_checks(SASS)
    assert ch["start_before_first_ldg"] and ch["loop_clock_reads"] == 2 and ch["epi_bracket"] is True
    assert ch["loop"] == [0x100, 0x140]
    late = SASS.replace("/*0160*/                   CS2R R20, SR_CLOCKLO ;", "/*0160*/  NOP ;")
    assert C.sass_checks(late)["epi_bracket"] is False
    no_start = SASS.replace("/*0010*/                   CS2R R4, SR_CLOCKLO ;", "/*0010*/  NOP ;")
    assert C.sass_checks(no_start)["start_before_first_ldg"] is False
    w1 = SASS.replace("/*0170*/                   LDG.E R22, desc[UR4][R24.64] ;", "/*0170*/  NOP ;")
    assert C.sass_checks(w1)["epi_bracket"] is None


def test_normalized_sass_drops_addresses_encodings_and_line_info():
    a = C.normalize_sass(SASS)
    shifted = "\n".join(ln.replace("/*0", "/*1") for ln in SASS.splitlines()) + "\n  //## File \"x.py\", line 9\n"
    assert C.normalize_sass(shifted)[:3] == a[:3] and len(a) == 13
    assert C.normalize_sass(SASS.replace("CS2R R20", "CS2R R21")) != a
