"""Round 5's frozen prefill model, loaded from inside the repository (model v7's prefill branch).

Round 5 (moe-kernels-work 2026-10-28 round5, ROUND5.md) is the model tile rental B3 froze and failed
(docs/registered/2026-11-04-tileB3-gh200, FROZEN): the STAG-CUT w1 byte law with W2 on w2, the K1 knee in the wall
domain at round 4's constants, the clock form of F1 and v5'' at 64x128. It existed only as 32 work files outside the
repository. They are copied here BYTE FOR BYTE under scripts/round5_frozen/<their path under moe-kernels-work>, and
`verify()` holds every one to the sha256 B3's registration pins (frozen.files): not one line of round 5 is rewritten.

The work files name their neighbours and their data by absolute paths under the home directory
(~/code/moe-kernels-work/..., ~/code/moe-kernels, ~/moe-kernels-exfil/runs/<run>/tree/vm/results). `chain()` imports
them with HOME pointed at a private directory of symlinks (`frozen_home`), made once per process:

    code/moe-kernels-work                          -> scripts/round5_frozen      (the 32 files)
    code/moe-kernels                               -> this repository            (its pinned repo modules and pages)
    moe-kernels-exfil/runs/<run>/tree/vm/results   -> results/published/<session>/results, for the four runs below
                                                      (mirrored: real directories, every file a symlink)

Every file the chain opens under the four runs has a byte-identical published copy (checked when this module was
built: 24 of 24 by sha256), and tests/v7_chain_job.py traces every open and refuses any file outside the repository.
HOME is restored before `chain()` returns; the work files compute their paths at import, so nothing reads HOME later.
Module names the work files share with nothing else (r7, r6, r5, r4, k5, d5, common, ...) are removed from
sys.modules after the import, as read_b3.chain() does, and the loaded modules are kept here.

The pricing functions below are read_b3's (price, reads, round5_table), the same calls in the same order; the
repository test holds them to B3's committed predictions (scripts/scoring/tileB3/b3.predictions.json) and round 5's
table (tileB3.score.json preflight.round5_table) exactly."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FROZEN_ROOT = HERE / "round5_frozen"
B3_REG = REPO / "docs" / "registered" / "2026-11-04-tileB3-gh200.json"
PUB = REPO / "results" / "published"

#: the four exfil runs the work files name, and the published session holding each run's results tree
RUNS = {
    "gh200-tile2-20261020T2242Z": "2026-10-21-nvidia_gh200_480gb-tileA2-session",
    "gh200-tile1-20261021T0704Z": "2026-10-21-nvidia_gh200_480gb-settleS1-session",
    "gh200-tile2-20261021T1842Z": "2026-10-21-nvidia_gh200_480gb-cal2-session",
    "gh200-tile4-20261027T2238Z": "2026-10-27-nvidia_gh200_480gb-cal4-session",
}
#: B3's FROZEN_FILES (read_b3.FROZEN_FILES), the same order
FROZEN_FILES = (
    "2026-10-20/scoreA2/bytes.score.json", "2026-10-20/scoreA2/census.score.json",
    "2026-10-20/scoreA2/floor.score.json", "2026-10-20/scoreA2/rep.score.json",
    "2026-10-21/diag-A2/work/common.py",
    "2026-10-21/design-v5/work/v5price.py", "2026-10-21/design-v5/work/kneefit.py",
    "2026-10-21/design-v5/work/sag.py", "2026-10-21/design-v5/work/b2risk.py",
    "2026-10-21/design-v5/work/f3fit.json", "2026-10-21/design-v5/work/kneefit.json",
    "2026-10-21/design-v5/work/kneefit64.json",
    "2026-10-22/diag-v5/work/d5.py", "2026-10-22/diag-v5/work/bw.py", "2026-10-22/diag-v5/work/knee2.py",
    "2026-10-22/diag-v5/work/v5prime.py", "2026-10-22/diag-v5/work/bytes_lomo.json",
    "2026-10-22/knee-v5/work/k5.py",
    "2026-10-22/refit4/work/r4.py", "2026-10-22/refit4/work/refit4.py", "2026-10-22/refit4/work/bytes4.json",
    "2026-10-27/round3/work/r5.py", "2026-10-27/round3/work/r6.py", "2026-10-27/round3/work/tailmap.json",
    "2026-10-27/round3/work/knee4.json", "2026-10-27/round3/work/bytes7.json", "2026-10-27/round3/work/combo4.json",
    "2026-10-27/round3/work/DECLARED4.txt",
    "2026-10-28/round5/work/r7.py", "2026-10-28/round5/work/bytes8.json", "2026-10-28/round5/work/combo5.json",
    "2026-10-28/round5/work/DECLARED5.txt",
)
ROUND5 = "2026-10-28/round5/work"
ROUND4 = "2026-10-27/round3/work"
#: module names the chain imports from the work folders (read_b3.CHAIN_MODULES)
CHAIN_MODULES = ("r7", "refit4", "r6", "r5", "r4", "k5", "d5", "common", "v5price", "kneefit", "sag", "b2risk", "bw",
                 "knee2", "v5prime", "bytes4")

# ------------------------------------------------------------------ frozen constants (B3's, registered by value)
O, Q, M, X22, TP2 = "olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b", "mixtral-8x7b-tp2"
M_C2 = "mixtral-8x7b C2"
LAW = "STAG-CUT"
SELECTED = "SELECTED: STAG-CUT (w1) + W2 (w2) + K1 wall knee + clock form"
K1_SEEN = (6.0, 2.75)
TRIAD_A2_GBPS = 3726.2
P_V5PP = 5.75
SAG64_A2 = (30.37, 300.0)
#: round 5's selected table (combo5.json) and its five-model estimate on B2's 84 cells
ROUND5_MEANS = {O: 0.45, Q: -1.46, M: 1.77, X22: 0.24, TP2: 2.39}
ROUND5_M_C2 = 2.01
ROUND5_PASS_A = 0.124


class Refused(RuntimeError):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify() -> None:
    """Every vendored work file is the one B3 froze (sha256 against the registration's frozen.files); Refused
    otherwise."""
    want = json.loads(B3_REG.read_text())["frozen"]["files"]
    if set(want) != set(FROZEN_FILES):
        raise Refused(f"B3's frozen file list is not this module's: {sorted(set(want) ^ set(FROZEN_FILES))}")
    bad = [f for f in FROZEN_FILES if not (FROZEN_ROOT / f).is_file() or sha256(FROZEN_ROOT / f) != want[f]]
    if bad:
        raise Refused(f"vendored round-5 files differ from B3's frozen hashes: {bad}")


_HOME: dict = {}


def mirror(src: Path, dst: Path) -> None:
    """dst as a tree of real directories whose files are symlinks to src's: Path.rglob (Python 3.12) does not descend
    into a symlinked directory, and the chain finds its pages with rglob."""
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            (dst / rel / f).symlink_to(Path(root) / f)


def frozen_home() -> Path:
    """The private HOME the work files resolve their paths against (made once per process, removed at exit)."""
    if "dir" not in _HOME:
        d = Path(tempfile.mkdtemp(prefix="v7-round5-home-"))
        atexit.register(shutil.rmtree, d, True)
        (d / "code").mkdir()
        (d / "code" / "moe-kernels-work").symlink_to(FROZEN_ROOT, target_is_directory=True)
        (d / "code" / "moe-kernels").symlink_to(REPO, target_is_directory=True)
        for run, session in RUNS.items():
            mirror(PUB / session / "results", d / "moe-kernels-exfil" / "runs" / run / "tree" / "vm" / "results")
        _HOME["dir"] = d
    return _HOME["dir"]


_CHAIN: dict = {}


def chain() -> dict:
    """round 5's r7 imported once (with refit4, r6, r5, r4, k5, d5 and b2risk) from the vendored files; the CAL pricer
    is built at p 14 and T0 43.23 us and bw 3597.8 GB/s are asserted (the first-build trap), and every selection B3
    asserts is asserted again."""
    if _CHAIN:
        return _CHAIN
    verify()
    import moe.spec  # noqa: F401  (imported under the real HOME)
    home = frozen_home()
    keep_path = list(sys.path)
    keep_mods = {m: sys.modules.pop(m) for m in CHAIN_MODULES if m in sys.modules}
    keep_home = os.environ.get("HOME")
    keep_cwd = os.getcwd()
    keep_bc = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        os.environ["HOME"] = str(home)
        sys.path.insert(0, str(home / "code" / "moe-kernels-work" / ROUND5))
        import r7  # noqa: E402
        RF = r7._rc()
        import b2risk  # noqa: E402
    finally:
        if keep_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = keep_home
        os.chdir(keep_cwd)
        for m in CHAIN_MODULES:
            sys.modules.pop(m, None)
        sys.modules.update(keep_mods)
        sys.path[:] = keep_path
        sys.dont_write_bytecode = keep_bc
    r6, r5, r4, k5, d5 = r7.r6, r7.r5, r7.r4, r7.k5, r7.d5
    assert d5.TM.P_KNEE == 14.0
    x = [float(v) for v in d5.RG.pricer().f2.x]
    assert abs(x[0] * 1e3 - 43.23) < 0.01 and abs(x[2] - 3597.8) < 0.1, x
    assert abs(k5.RULERS["triad"] - TRIAD_A2_GBPS) < 0.1 and r5.SIGMA_GAP == 0.0375 and r5.L2_MIB == 60.0
    w = FROZEN_ROOT
    kn = json.loads((w / ROUND4 / "knee4.json").read_text())
    assert kn["selection"]["chosen"] == "K1" and tuple(kn["candidates"]["K1"]["SEEN"]) == K1_SEEN, kn["selection"]
    assert json.loads((w / ROUND5 / "bytes8.json").read_text())["selection"]["chosen"] == LAW
    c5 = json.loads((w / ROUND5 / "combo5.json").read_text())
    assert c5["law"] == LAW and c5["jetmoe_n2_PT"] == "PRICED" and c5["unpriced_b2"] == []
    assert c5["tables"][SELECTED]["summary"]["means"] == ROUND5_MEANS
    assert c5["tables"][SELECTED]["summary"]["8x7B C2"] == ROUND5_M_C2
    assert r7.W1_LAWS[LAW] == {"walk": "STAG", "surv": "SOFT", "cut": True, "nconst": 0}
    _CHAIN.update(r7=r7, RF=RF, r6=r6, r5=r5, r4=r4, k5=k5, d5=d5, B=b2risk, KW=kn["candidates"]["K1"])
    return _CHAIN


def knee(s0: float = K1_SEEN[1]):
    C = chain()
    return C["k5"].Form("K1", C["k5"].RULERS["triad"], s0)


def b2_rows() -> list:
    """B2's 84 NATIVE cells in B2's registered power-cell order (b2risk.b2_cells, as read_b3.b2_rows)."""
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "cal2"))
    import read_c2 as RC
    rows = chain()["B"].b2_cells()
    cells = RC.b2_cells()
    assert len(rows) == len(cells) == 84
    assert all(a["model"] == b["model"] and a["cls"] == b["cls"] and a["page"] == b["page"] for a, b in zip(rows, cells))
    return rows


def reads(r: dict, p: float):
    """{w1, w2} DRAM reads of a 128x128 cell under STAG-CUT on w1 and W2 on w2 at knee p; None: the guard refuses."""
    return chain()["r7"].reads5(r, p, LAW)


def price_128(r: dict, sag128: tuple, rd, p: float = K1_SEEN[0], s0: float = K1_SEEN[1]) -> float:
    """r6.cell_ms4 at the K1 knee (p, s0), the clock form of F1 at sag128, with the reads rd (read_b3.price's call)."""
    return chain()["r6"].cell_ms4(r, knee(s0), p, tuple(sag128), rd, clock=True)


def held_out_sets() -> dict:
    """Round 5's held-out 128 G 1 sets (read_b3.round5_table's): {set: (cells, fold model)}."""
    C = chain()
    r7, RF = C["r7"], C["RF"]
    a2, c2x, c2m = RF.timed_cells()
    return {O: ([r for r in a2 if r["model"] == O], O), Q: ([r for r in a2 if r["model"] == Q], Q),
            M: ([r for r in a2 if r["model"] == M], M), X22: (c2x, X22), TP2: (r7.c4_timed_rows(), TP2),
            M_C2: (c2m, M)}


_SAG: dict = {}


def held_out_sag(r: dict, hm: str) -> tuple:
    """The fold's sag on the timed page's board: C4's both pages for tp2's board, else F1 par of the fold's estimator
    pages (RF.fold_F1)."""
    C = chain()
    if "fits" not in _SAG:
        _SAG["fits"] = C["RF"].sag_fits()
        _SAG["c4par"] = C["r7"].c4_sag_par()[1]
    if r["board"] == "C4":
        return tuple(_SAG["c4par"])
    return tuple(C["RF"].fold_F1(_SAG["fits"], "C2" if r["board"] == "C2" else "A2", hm))


def held_out_knee(hm: str) -> tuple:
    """(p, S0) of the fold: K1 at the fold's LOMO constants, SEEN for tp2 (round 5's)."""
    KW = chain()["KW"]
    return tuple(KW["SEEN"]) if hm == TP2 else tuple(KW["LOMO"][hm])


def pass_estimate(means: dict, cells=None) -> dict:
    """Round 5's five-model estimate at B2's bar (r7.pass_estimates5: read_c2.power, R 200,000, seed 20261022) on the
    per-model held-out means (%, O, Q, M on A2, 8x22B, tp2); cells default to B2's 84 power cells."""
    C = chain()
    if cells is None:
        cells = C["RF"].RC.b2_cells()
    return C["r7"].pass_estimates5(means, cells)
