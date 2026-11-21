#!/usr/bin/env python
"""MODEL v4's timing term (the tile stage, 2026-10-14; docs/registered/2026-10-14-tileA-floor-gh200).

    python scripts/r3_timing_model_v4.py --check     # v4 at R3's tile against every registered v3 prediction

v4 is model v3 with every tile-dependent quantity read from the tile. A separate module over
`r3_timing_model` and `r3_timing_model_v3` (both byte-identical to their registered pins): the tile
globals of `r3_timing_model` are swapped in memory for one call (`at_tile`) and restored, the file
on disk is never touched. Nothing here writes tiling or scheduling logic: a tile is the config
values (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) vLLM's fused_moe_kernel is handed.

THE REDUCTION TO v3 (design-tile-review 1a). v3 holds the per-CTA fixed cost in k-steps of the
COUNTER c (PHI = F / 344.1, `r3_timing_model.CTA_FIXED_KSTEPS`), and its time c is the CAL fit's
202.55 ns (346.4 cycles at 1710). v4 keeps the two apart:

    c_time(tile) = c_card x c_counter(tile) / 344.1        (c_scale = c_counter / 344.1)
    PHI(tile, g) = F(tile, g) / c_counter(tile)             (F in cycles, PHI in k-steps)

At R3's tile c_counter is 344.1 by definition, so c_scale is exactly 1.0, and PHI is v3's own
CTA_FIXED_KSTEPS (F-step, below): the cell is priced by the same calls in the same order as
rental 6's pricer (`reg6.priced`), and `--check` reproduces every registered v3 prediction to 0
(equality, not a tolerance).

c_counter PER PATH.
  mma.sync (BLOCK_M % 64 != 0 or num_warps % 4 != 0): MVA2, rental 4's registered law (eta 0.832,
     z0 1503.751, z1 -1251.132; SEEN-fitted, owner decision D5), as a RATIO to R3's tile at R3's
     recorded occupancy: c_counter = 344.1 x MVA2(W_tile, BK, occ) / MVA2(W_R3, 64, occ_R3,g). W is
     the LSU shared wavefronts per CTA k-step (4 per ldmatrix.x4 warp instruction, 1 per 512 B
     LDGSTS warp instruction), read from the compiled PTX by scripts/tile_census.py.
  wgmma, the calibration tile 128x128x64 w8 s3: c_128, measured on rental A's G-64 counter pages
     of two CAL models (CAL-A), one c for both GEMMs, as v3 has.
  wgmma, another tile: a closed MVA with pipe service D(tile) and one delay Z_wg calibrated so the
     calibration tile reproduces c_128: D = max(shared bytes / 128 B per clock / eta_smem, MMA
     cycles), the shared bytes by one of three registered accountings (VARIANTS), MMA cycles =
     2 BM BN BK / 4096 flop per clock / eta_mma(instruction).
  v4cal: the per-tile c and F measured on the CAL pages of that tile (CAL-A), the registered
     fallback when no accounting holds on rental A's 64x128 pages.

F, THE PER-CTA FIXED COST. F-step (v3's own form, registered): PHI in k-steps carries from the
tile it was measured at (R3's PHI on the mma.sync path; PHI_wg = F_wg / c_128 on the wgmma path).
F-cyc (the rival): F in cycles carries, PHI = F / c_counter(tile). They are equal at the tile F
was measured at; rental A's 64x128 pages choose between them by a registered rule.

eta_mma (design-tile-review 1b; build-tile-review 1). Luo et al., "Dissecting the NVIDIA Hopper
Architecture through Microbenchmarking and Multiple Level Analysis", arXiv 2501.12084, measured on an
H800 PCIe (114 SMs, 350 W cap) with FP16 inputs (wgmma.m64nNk16.f32.f16.f16). v4 counts MMA CYCLES at a
locked clock, so eta must be a per-clock efficiency. PRIMARY: Table 11's per-clock rate, the only wgmma
rate with a measured clock: m64n256k16 dense SS random data, 583 TFLOPS at 1260 MHz on 114 SMs = 0.991
per clock, carried to each instruction by Table 9's within-table random-data ratios (N 128 659.8 / 665.4,
N 64 648.3 / 665.4): m64n128k16 (the 128x128 w8 tile) 0.983, m64n64k16 (64x128 w8) 0.965. RIVAL:
Table 9's chip-relative rates against the 756.5 TFLOPS chip peak at the 1620 MHz whitepaper clock:
0.872 (N 128) and 0.857 (N 64), which fold the H800 PCIe's power-capped clock in. REGISTERED ASSUMPTION:
BF16 on the GH200 runs at the same per-clock rate as FP16 on the H800 PCIe.

eta_smem: the wgmma accountings divide shared bytes by MVA2's eta, 0.832, fitted on the mma.sync
path's LSU wavefronts: an UNTESTED reuse on the wgmma path (registered as such).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v3 as TV3  # noqa: E402

SMS = 132
CLOCK_MHZ = 1710.0
#: the counter c of the CAL counters (r3_timing_model.C_CYC_PER_KSTEP, 344.1 cycles)
C_CNT_R3 = TM.C_CYC_PER_KSTEP
#: R3's recorded occupancy on Hopper, w1 then w2
OCC_R3 = dict(TM.ASSUMED_OCCUPANCY["9.0"])
F_FORMS = ("F-step", "F-cyc")
F_FORM_REGISTERED = "F-step"
VARIANTS = ("B", "W", "T")
SMEM_B_PER_CLK = 128.0
FLOP_PER_CLK = 4096.0
#: rental 4's MVA2 (docs/registered/2026-10-06-rental4-occlaw-gh200 "laws"), SEEN-fitted
MVA2 = {"eta": 0.832, "z0": 1503.751, "z1": -1251.132}
#: Luo et al. Table 11, per clock: 583 TFLOPS / (114 SMs x 4096 x 1.260 GHz), m64n256k16 SS random data
ETA_PER_CLOCK_N256 = 583.0 / (114 * 4096 * 1.260e-3)
#: PRIMARY, per wgmma instruction: the per-clock rate times Table 9's within-table random-data ratio to N 256
ETA_MMA = {"m64n128k16": ETA_PER_CLOCK_N256 * 659.8 / 665.4, "m64n64k16": ETA_PER_CLOCK_N256 * 648.3 / 665.4}
#: RIVAL, per instruction: Luo et al. Table 9's chip TFLOPS over the 756.5 chip peak
ETA_MMA_RIVAL = {"m64n128k16": 659.8 / 756.5, "m64n64k16": 648.3 / 756.5}
#: one LDGSTS warp instruction (32 threads x 16 B) is one LSU wavefront (the occlaw reading)
LDGSTS_BYTES_PER_WAVEFRONT = 512.0


@dataclass(frozen=True)
class Tile:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int

    @property
    def path(self) -> str:
        """Triton's MMA path: wgmma when BLOCK_M % 64 == 0 and num_warps % 4 == 0 (confirmed on
        every published cubin, design-tile 1)."""
        return "wgmma" if self.bm % 64 == 0 and self.warps % 4 == 0 else "mma.sync"

    @property
    def key(self) -> str:
        return f"{self.bm}x{self.bn}x{self.bk}w{self.warps}s{self.stages}"

    def shared_bytes(self) -> int:
        """The compiled cache's rule (design-tile 1, census of 1,304 published cubins): wgmma
        allocates num_stages buffers, mma.sync num_stages - 1."""
        bufs = self.stages if self.path == "wgmma" else self.stages - 1
        return bufs * (self.bm * self.bk + self.bk * self.bn) * 2

    def occ_shared(self) -> int:
        """CTAs per SM from shared memory and warps alone (sm_90: 233,472 B, 1 KiB reserved per CTA,
        64 warps, 32 CTAs): the design occupancy before the census reads registers."""
        return min(233472 // (self.shared_bytes() + 1024), 64 // self.warps, 32)

    @staticmethod
    def parse(s: str) -> Tile:
        import re
        m = re.fullmatch(r"(\d+)x(\d+)x(\d+)w(\d+)s(\d+)", s)
        if not m:
            raise ValueError(f"tile {s!r}: BMxBNxBKw<warps>s<stages>")
        return Tile(*map(int, m.groups()))


R3_TILE = Tile(32, 64, 64, 8, 4)
CAL_TILE = Tile(128, 128, 64, 8, 3)
T64W8 = Tile(64, 128, 64, 8, 3)
T64W4 = Tile(64, 128, 64, 4, 3)
T32W4 = Tile(32, 128, 64, 4, 3)
#: vLLM's default-config tiles (design-tile 1); D1 puts the four decode tiles in a later stage
VLLM_TILES = (Tile(16, 64, 128, 4, 4), Tile(32, 64, 128, 4, 3), T32W4, T64W4, T64W8, CAL_TILE)
REACHABLE = (T64W8, CAL_TILE)


# --------------------------------------------------------------------------
# per CTA k-step counts: from the compiled PTX (the census), else the layout rule
# --------------------------------------------------------------------------

def mma_layout_estimate(t: Tile) -> tuple[int, int]:
    """ESTIMATE of Triton's MMAv2 warpsPerCTA (the compiled PTX decides): split M first while a
    warp still has 16 rows and M per warp >= N per warp, else N. R3's 32x64 w8 gives [2, 4]."""
    wm, wn = 1, 1
    while wm * wn < t.warps:
        if t.bm // (wm * 2) >= 16 and t.bm / (wm * 2) >= t.bn / (wn * 4):
            wm *= 2
        else:
            wn *= 2
    return wm, wn


def wavefronts_from_counts(ptx: dict, warps: int) -> float:
    """LSU shared wavefronts per CTA k-step from per-thread PTX loop counts
    (`r3_timing_model.ptx_main_loop`): every warp issues each instruction once; an ldmatrix.xN
    warp instruction is N wavefronts (`ldmatrix_bytes_per_thread` / 4 per thread = sum of N),
    a cp.async warp instruction is (bytes per thread x 32) / 512 wavefronts."""
    ldm = ptx["ldmatrix_bytes_per_thread"] / 4.0
    cp = ptx["cp_async_bytes_per_thread"] * 32.0 / LDGSTS_BYTES_PER_WAVEFRONT
    return warps * (ldm + cp)


def wavefronts_estimate(t: Tile) -> float:
    """The design's estimate before the census: each warp column re-reads its A rows, each warp
    row its B columns, 128 B per ldmatrix wavefront, plus one LDGSTS wavefront per 512 B."""
    wm, wn = mma_layout_estimate(t)
    ldm = (t.bm * t.bk * 2 * wn + t.bk * t.bn * 2 * wm) / 128.0
    return ldm + (t.bm * t.bk + t.bk * t.bn) * 2 / LDGSTS_BYTES_PER_WAVEFRONT


def wgmma_layout(t: Tile) -> tuple[int, int, int]:
    """(warpgroups along M, along N, instruction N): warpgroups take M first in m64 steps, the
    rest split N (Triton MMAv3; the published H200 PTX: 128x128 w8 issues m64n128k16 with the
    two warpgroups on M, 64x128 w8 issues m64n64k16 with the two on N)."""
    wg = t.warps // 4
    gm = min(wg, t.bm // 64)
    gn = wg // gm
    return gm, gn, t.bn // gn


def wgmma_counts_estimate(t: Tile) -> dict:
    gm, gn, ninst = wgmma_layout(t)
    return {"instruction": f"m64n{ninst}k16", "per_warpgroup": t.bk // 16, "warpgroups": gm * gn,
            "source": "layout rule (design estimate; the census replaces it)"}


def wgmma_bytes(t: Tile, counts: dict | None = None) -> dict:
    """Per CTA k-step on the wgmma path (SS: both operands by shared-memory descriptor): the
    operand bytes the wgmma instructions read, the cp.async bytes written, and the instruction."""
    c = counts or wgmma_counts_estimate(t)
    import re
    m = re.fullmatch(r"m64n(\d+)k16", c["instruction"])
    if not m:
        raise ValueError(f"wgmma instruction {c['instruction']!r}: expected m64nNk16")
    ninst = int(m.group(1))
    per = c["per_warpgroup"] * c["warpgroups"]
    reads = per * (64 * 16 * 2 + 16 * ninst * 2)
    writes = (t.bm * t.bk + t.bk * t.bn) * 2
    return {"reads": float(reads), "writes": float(writes), "instruction": c["instruction"]}


def mva(s: float, Z: float, N: int) -> float:
    """r4common.mva (rental 4's): exact MVA of a closed two-station queue, cycles per completion."""
    Q, X = 0.0, None
    for n in range(1, N + 1):
        R = s * (1 + Q)
        X = n / (R + Z)
        Q = X * R
    return 1.0 / X


def mva2_c(W: float, bk: int, occ: int) -> float:
    return mva(W / MVA2["eta"], max(MVA2["z0"] + MVA2["z1"] * bk / 64, 0.0), occ)


def pipe_service(t: Tile, variant: str, eta_mma: float, counts: dict | None = None) -> dict:
    """D(tile) of one wgmma accounting (cycles before the MVA): B operand reads and cp.async writes
    on one 128 B/clk port; W operand reads at 128 B/clk, writes as LDGSTS wavefronts; T tensor
    only. The shared divisor is MVA2's eta (UNTESTED reuse)."""
    b = wgmma_bytes(t, counts)
    mma = 2.0 * t.bm * t.bn * t.bk / FLOP_PER_CLK / eta_mma
    if variant == "B":
        D = max((b["reads"] + b["writes"]) / SMEM_B_PER_CLK / MVA2["eta"], mma)
    elif variant == "W":
        D = max((b["reads"] / SMEM_B_PER_CLK + b["writes"] / LDGSTS_BYTES_PER_WAVEFRONT) / MVA2["eta"], mma)
    elif variant == "T":
        D = mma
    else:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    return {"D": D, "mma_cycles": mma, **b}


def eta_mma_of(t: Tile, rival: bool = False, counts: dict | None = None) -> float:
    return (ETA_MMA_RIVAL if rival else ETA_MMA)[wgmma_bytes(t, counts)["instruction"]]


def solve_z(D: float, occ: int, c_target: float, z_max: float = 20000.0) -> float | None:
    """The delay Z with mva(D, Z, occ) = c_target (mva rises with Z); None when even Z = 0 gives
    more than the target (the accounting cannot reach it: FALSIFIED by construction)."""
    lo, hi = 0.0, z_max
    if mva(D, lo, occ) > c_target:
        return None
    if mva(D, hi, occ) < c_target:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mva(D, mid, occ) < c_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# the constants v4 carries per tile (CAL-A, SEEN or derived)
# --------------------------------------------------------------------------

@dataclass
class Params:
    """Every rental-A constant and selection v4 reads. The defaults are the registered design
    values; `from_selections` fills them from rental A's scored selections (tileA files)."""
    c128: float | None = None                       # counter cycles per CTA k-step at CAL_TILE (CAL-A)
    F_wg: dict = field(default_factory=dict)        # {"w1": cycles, "w2": cycles} at CAL_TILE (CAL-A)
    variant: str = "W"                              # B, W or T
    f_form: str = F_FORM_REGISTERED
    eta_rival: bool = False                         # True: Table 9's chip-relative rates
    v4cal: dict = field(default_factory=dict)       # {tile key: {"c": cycles, "F": {"w1", "w2"}}} (CAL-A)
    occupancy: dict = field(default_factory=dict)   # {tile key: {"w1": occ, "w2": occ}} (census)
    counts: dict = field(default_factory=dict)      # {tile key: census PTX counts}
    #: "v4", or the rival "v3own": v3 at its own constants, only the geometry moving (c 344.1 and
    #: R3's PHI at every tile)
    law: str = "v4"

    def occ(self, t: Tile, g: str) -> int:
        if t == R3_TILE:
            return OCC_R3[g]
        o = self.occupancy.get(t.key)
        return int(o[g]) if o else t.occ_shared()


def c_counter(t: Tile, g: str, prm: Params) -> float:
    """Counter cycles per CTA k-step of GEMM g at tile t (see the module docstring)."""
    if t == R3_TILE or prm.law == "v3own":
        return C_CNT_R3
    if t.key in prm.v4cal:
        return float(prm.v4cal[t.key]["c"])
    occ = prm.occ(t, g)
    if t.path == "mma.sync":
        cnt = prm.counts.get(t.key)
        Wt = wavefronts_from_counts(cnt, t.warps) if cnt else wavefronts_estimate(t)
        cr = prm.counts.get(R3_TILE.key)
        W3 = wavefronts_from_counts(cr, R3_TILE.warps) if cr else wavefronts_estimate(R3_TILE)
        return C_CNT_R3 * mva2_c(Wt, t.bk, occ) / mva2_c(W3, R3_TILE.bk, OCC_R3[g])
    if prm.c128 is None:
        raise TM.Refused(f"{t.key}: v4 has no c_128 yet (rental A's calibration)")
    if t == CAL_TILE:
        return float(prm.c128)
    z = z_wg(prm)
    if z is None:
        raise TM.Refused(f"accounting {prm.variant} cannot reach c_128 = {prm.c128:.1f} at {CAL_TILE.key}")
    eta = eta_mma_of(t, prm.eta_rival, prm.counts.get(t.key))
    return mva(pipe_service(t, prm.variant, eta, prm.counts.get(t.key))["D"], z, occ)


def z_wg(prm: Params) -> float | None:
    """Z_wg: the delay that puts the calibration tile at c_128 under the accounting."""
    occ = prm.occ(CAL_TILE, "w1")
    eta = eta_mma_of(CAL_TILE, prm.eta_rival, prm.counts.get(CAL_TILE.key))
    D = pipe_service(CAL_TILE, prm.variant, eta, prm.counts.get(CAL_TILE.key))["D"]
    return solve_z(D, occ, float(prm.c128))


def phi(t: Tile, g: str, prm: Params) -> float:
    """PHI(tile, g) in k-steps of the tile's own c (see the module docstring)."""
    if t == R3_TILE or prm.law == "v3own":
        return TM.CTA_FIXED_KSTEPS[g]
    if t.key in prm.v4cal:
        cal = prm.v4cal[t.key]
        return float(cal["F"][g]) / float(cal["c"])
    if t.path == "mma.sync":
        if prm.f_form == "F-step":
            return TM.CTA_FIXED_KSTEPS[g]
        return TM.CTA_FIXED_KSTEPS[g] * (C_CNT_R3 / c_counter(t, g, prm))
    if not prm.F_wg or prm.c128 is None:
        raise TM.Refused(f"{t.key}: v4 has no F_wg yet (rental A's calibration)")
    if t == CAL_TILE or prm.f_form == "F-step":
        return float(prm.F_wg[g]) / float(prm.c128)
    return float(prm.F_wg[g]) / c_counter(t, g, prm)


def c_scale(t: Tile, g: str, prm: Params) -> float:
    """c_time / c_card = c_counter / 344.1: exactly 1.0 at R3's tile."""
    if t == R3_TILE or prm.law == "v3own":
        return 1.0
    return c_counter(t, g, prm) / C_CNT_R3


# --------------------------------------------------------------------------
# the tile, swapped in memory
# --------------------------------------------------------------------------

_TILE_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "OUT_TILE", "CTA_FIXED_KSTEPS")


@contextlib.contextmanager
def at_tile(model: str, t: Tile, phis: dict | None = None):
    """r3_timing_model at tile t for `model` (BLOCK_M/N/K, the geometry, the byte model, the ids
    per tread, the other kernels' bytes, the output tile, PHI), restored on exit. At R3's tile
    with no PHI given this is `set_model(model)` alone, what rental 6's pricer does."""
    keep = {k: getattr(TM, k) for k in _TILE_KEYS}
    old = TM.MODEL
    try:
        TM.BLOCK_M, TM.BLOCK_N, TM.BLOCK_K = t.bm, t.bn, t.bk
        TM.OUT_TILE = t.bm * t.bn * TM.BYTES
        if phis is not None:
            TM.CTA_FIXED_KSTEPS = dict(phis)
        TM.set_model(model)
        yield
    finally:
        for k, v in keep.items():
            setattr(TM, k, v)
        TM.set_model(old)


def call_ms(x, cell, ctx, k_w: float, scale: dict, *, native_path: str | None, align_ms: float | None) -> float:
    """v3's call time (`r3_timing_model_v3.call_ms`) with each GEMM's c scaled by its own
    c_counter / 344.1. One scale for both GEMMs (always so at R3's tile, scale 1.0) goes through
    r3_timing_model.call_ms's own c_scale, so v4 at R3's tile is the same call as v3."""
    if scale["w1"] == scale["w2"]:
        return TV3.call_ms(x, cell, ctx, k_w, native_path=native_path, align_ms=align_ms, c_scale=scale["w1"])
    T0, c_ns, bw, s_small, s_block = (float(v) for v in x)
    counts = getattr(cell, "counts", None)
    treads = cell.n if counts is None else TM.numel_of(cell.n, counts) / TM.IDS_PER_TREAD
    t = T0 + treads * TM.B_OTHER / (bw * 1e6)
    for g in TM.GEMMS:
        cg = c_ns * scale[g]
        win = TM.window(cell.arm, cell.declared, cell.G, cell.n, g,
                        TM.window_width(k_w, ctx.sms, ctx.occupancy[g]), counts)
        t += TM.gemm_ms(win, g, cell.sigma[g], cg, bw, ctx.sms, occ=ctx.occupancy[g])
        t += TM.dead_ms(cell.declared, cell.n, g, cg, k_w, ctx.occupancy[g], counts)
        t += TM.gemm_const_ms(g, cg)
    if cell.arm == "native":
        t += s_small if cell.path == TM.SMALL_BATCH else s_block
    return t + TV3.v3_b_ms(cell.arm, native_path, align_ms, key=cell.key)


def make_cell(eng, model: str, arm: str, n: int, counts, copies: int, G: int, t: Tile, reads: dict,
              occ: dict, k_w: float, sms: int):
    """A TM.Cell at tile t (TM must be at_tile): rental 5's Engine.cell with the tile's G, its
    BLOCK_M's path and its occupancy's window (Engine.cell's own code at R3's tile and G 8)."""
    import private_weight_reference as PWR

    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    D = cfg.num_experts if arm == "native" else cfg.num_experts * copies
    ct = None if counts is None else tuple(int(v) for v in counts)
    numel = PWR.ids_for_tread(cfg, n, t.bm)
    path = PWR.align_path(numel, D)
    sigma = {}
    for g in TM.GEMMS:
        w = TM.window_width(k_w, sms, occ[g])
        sigma[g] = reads[g] / TM.window(arm, D, G, n, g, w, ct).reads
    return TM.Cell(arm=arm, G=G, n=n, ms=float("nan"), mhz=None, path=path, declared=D, runs=(),
                   source="PREDICTED", reads=reads, sigma=sigma, fit=False, counts=ct)


def native_path_of(model: str, n: int, t: Tile) -> str:
    import private_weight_reference as PWR

    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    return PWR.align_path(PWR.ids_for_tread(cfg, n, t.bm), cfg.num_experts)


# --------------------------------------------------------------------------
# --check: v4 at R3's tile is v3, to 0
# --------------------------------------------------------------------------

#: the registered v3 predictions --check reproduces (every row with an "ms" carrying "v3")
CHECK_FILES = ("docs/registered/2026-10-09-rental6-skew-gh200.json", "docs/registered/2026-10-13-rental7-skew-gh200.json")


def check(limit: int | None = None, files=CHECK_FILES) -> dict:
    """Every registered v3 prediction (model, arm, n, counts, copies 9, seed, its registered A
    proxy) re-priced by v4 at R3's tile (G 8, occupancy 5 / 4, the v3 byte key, F-step). Two
    comparisons, both equality: v4 rounded as the registrations round (6 decimals, r6) against the
    registered number, and v4 unrounded against rental 6's pricer (`r6model.Pricer.price`, the code
    that wrote those numbers) on the same cell, bit for bit."""
    import wave_split_bytes_v4 as WV4
    sys.path.insert(0, str(REPO / "scripts" / "scoring" / "rental6"))
    import reg6 as R6
    pr = R6.pricer()
    out = {"compared": 0, "unequal_registered": [], "unequal_pricer": [], "files": list(files)}
    prm = Params()
    for f in files:
        reg = json.loads((REPO / f).read_text())
        hist = {k: json.loads((REPO / v["path"]).read_text()) for k, v in reg["histograms"]["files"].items()}
        for r in reg["predictions"]:
            if limit is not None and out["compared"] >= limit:
                return out
            page = hist[f"{r['model']}-{r['page']}.json"]
            cell = next(c for c in page["cells"] if c["label"] == r["label"] and int(c["n"]) == int(r["n"]))
            seed = None if cell["counts"] is None else page["shuffle_seed"]
            copies = r.get("copies", 9)
            got = WV4.price(pr, r["model"], r["arm"], int(r["n"]), cell["counts"], copies, seed,
                            R3_TILE, 8, prm, key="v3", A_us=r["A_proxy_us"])
            ref = pr.price(r["model"], r["arm"], int(r["n"]), cell["counts"], copies, seed, A_us=r["A_proxy_us"])
            out["compared"] += 1
            row = [r["model"], r["page"], r["n"], r["label"], r["arm"]]
            if round(got["ms"], 6) != r["ms"]["v3"]:
                out["unequal_registered"].append({"row": row, "registered": r["ms"]["v3"], "v4": got["ms"]})
            if got["ms"] != ref["v3"]:
                out["unequal_pricer"].append({"row": row, "pricer": ref["v3"], "v4": got["ms"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="v4 at R3's tile against every registered v3 prediction")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.check:
        ap.print_help()
        return 2
    got = check(a.limit)
    print(f"v4 at R3's tile against registered v3: {got['compared']} predictions compared; "
          f"{len(got['unequal_registered'])} differ from the registered number (6 decimals), "
          f"{len(got['unequal_pricer'])} differ from rental 6's pricer bit for bit")
    for u in (got["unequal_registered"] + got["unequal_pricer"])[:10]:
        print("  DIFFERS", u)
    return 0 if got["compared"] and not got["unequal_registered"] and not got["unequal_pricer"] else 1


if __name__ == "__main__":
    sys.exit(main())
