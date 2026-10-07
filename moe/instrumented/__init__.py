"""The instrumented copy of vLLM v0.27.1's fused_moe kernel: its spec, its row layout and
its installer (rental 4, owner decision 5 of 2026-10-06).

The kernel itself is `fused_moe_instr.py` (Triton at import time, so the VM only). This
module imports neither Triton nor vLLM at import time, so the laptop's suite reads it.

    spec = parse_spec("stamps=iter,every=1,evict_a=none,evict_b=none")
    with install(spec) as rec:          # patches vLLM's module attribute, undone on exit
        fused_experts(...)              # each fused_moe_kernel launch goes to the copy
    rows = rec.launches                 # one record per launch: grid, N, K, EM, the buffer

INSTRUMENTED UNITS ONLY. Nothing in the study's timing path imports this module: the
plain installed kernel stays the measured object of every registered timing test, and
an instrumented unit's use is gated by the perturbation unit (decision 2: median
|plain / instrumented - 1| <= 1%, worst <= 2%, identical occupancy limit).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = json.loads((HERE / "UPSTREAM.json").read_text())
EXCERPT = HERE / "upstream_v0.27.1_fused_moe.excerpt"
COPY = HERE / "fused_moe_instr.py"
#: the vLLM module whose `fused_moe_kernel` attribute the installer swaps
VLLM_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe"

#: header columns of a stamp row; fused_moe_instr.STAMP_HDR is the same number (a test)
HDR = 10
COLUMNS = ("smid", "start_ns", "start_clk", "prologue_ns", "prologue_clk", "epi_ns", "epi_clk",
           "end_ns", "end_clk", "kind")
KIND = {1: "live", 2: "dead", -1: "unstamped"}
#: "ends" (3): the CTA start, a dead CTA's exit and the end only, nothing next to the k-loop
#: (the cadence proposed after rental 4's perturbation gate failed; fused_moe_instr.py)
#: "sample" (4, rental 5, owner decision 2 of 2026-10-07): the CTA start, a dead CTA's exit, the
#: end, and the TOP of k-iteration 0 and of every k with k mod every = (S - 1) mod every
#: (phase-aligned, so the last iteration is always marked), with CTA sampling (cta_mod,
#: iter_mod, dead_mod: a CTA is stamped only when pid mod the modulus is 0)
STAMP_LEVELS = {"off": 0, "cta": 1, "iter": 2, "ends": 3, "sample": 4}
SAMPLE = 4
#: RENTAL 5's three registered variants of the sample level, in the driver's preference order
#: (docs/registered/2026-10-07-rental5-stamps2-gh200): pid mod 17, never 16, because pid mod 16
#: stamps only pid_m offset 0 of every GROUP_M group (design-r5-review work/sample_bias.txt)
R5_VARIANTS = {
    "v1": "stamps=sample,every=16,cta_mod=1,iter_mod=1,dead_mod=1",
    "v2": "stamps=sample,every=16,cta_mod=1,iter_mod=17,dead_mod=17",
    "v3": "stamps=sample,every=16,cta_mod=17,iter_mod=17,dead_mod=17",
}
EVICT = {"none": "", "first": "evict_first", "last": "evict_last"}
#: iteration marks a row holds when a spec does not say: K / BLOCK_K of the deepest GEMM
#: rental 4 stamps (OLMoE w1 at BK 32: 2048 / 32 = 64)
DEFAULT_MARKS = 64


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class InstrSpec:
    """Which instruments a unit turns on. The default is every instrument off."""

    stamps: int = 0
    every: int = 1
    marks: int = 0
    evict_a: str = "none"
    evict_b: str = "none"
    #: rental 5's sampling (the sample level only): stamp a live CTA when pid mod cta_mod is
    #: 0, its iteration tops when also pid mod iter_mod is 0, a dead CTA when pid mod
    #: dead_mod is 0
    cta_mod: int = 1
    iter_mod: int = 1
    dead_mod: int = 1

    def __post_init__(self):
        if self.stamps not in (0, 1, 2, 3, 4):
            raise SpecError(f"stamps {self.stamps}: 0 off, 1 cta, 2 iter, 3 ends, 4 sample")
        for name in ("cta_mod", "iter_mod", "dead_mod"):
            m = getattr(self, name)
            if m < 1:
                raise SpecError(f"{name} {m}: at least 1")
            if m != 1 and self.stamps != SAMPLE:
                raise SpecError(f"{name} {m}: sampling belongs to stamps=sample only")
        if self.every < 1:
            raise SpecError(f"every {self.every}: at least 1")
        if self.marks < 0 or (self.stamps == 2 and self.marks < 1):
            raise SpecError(f"marks {self.marks}: stamps=iter needs at least one mark")
        for name, v in (("evict_a", self.evict_a), ("evict_b", self.evict_b)):
            if v not in EVICT:
                raise SpecError(f"{name} {v!r}: one of {sorted(EVICT)}")

    @property
    def off(self) -> bool:
        return self.stamps == 0 and self.evict_a == "none" and self.evict_b == "none"

    @property
    def hints(self) -> bool:
        return self.evict_a != "none" or self.evict_b != "none"

    @property
    def width(self) -> int:
        """int64 columns per stamp row (the sample level's depends on the launch's K: see
        `layout`; this is its width with no mark)."""
        return HDR + 2 * (self.marks if self.stamps == 2 else 0)

    def kernel_kwargs(self) -> dict:
        """The instrument constexprs the copy takes, beside upstream's own arguments. The
        sample level adds STAMP_PHASE and its own STAMP_MARKS per launch (`layout`)."""
        out = {"STAMPS": self.stamps, "STAMP_EVERY": self.every,
               "STAMP_MARKS": self.marks if self.stamps == 2 else 0,
               "EVICT_A": EVICT[self.evict_a], "EVICT_B": EVICT[self.evict_b]}
        if self.stamps == SAMPLE:
            out.update(STAMP_CTA_MOD=self.cta_mod, STAMP_ITER_MOD=self.iter_mod,
                       STAMP_DEAD_MOD=self.dead_mod)
        return out

    def layout(self, K: int, block_k: int) -> dict:
        """One launch's row layout: the constexprs it adds, its width and which k each mark
        holds. The sample level: S = cdiv(K, BLOCK_K), phase (S - 1) mod every, marks at
        k = 0 and every k = phase (mod every) (sample_ks); other levels: as kernel_kwargs."""
        if self.stamps != SAMPLE:
            ks = list(range(0, self.marks * self.every, self.every)) if self.stamps == 2 else []
            return {"extra": {}, "width": self.width, "marks": len(ks), "mark_k": ks}
        S = -(-int(K) // int(block_k))
        ks = sample_ks(S, self.every)
        return {"extra": {"STAMP_PHASE": (S - 1) % self.every, "STAMP_MARKS": len(ks)},
                "width": HDR + 2 * len(ks), "marks": len(ks), "mark_k": ks, "ksteps": S}

    def text(self) -> str:
        level = {v: k for k, v in STAMP_LEVELS.items()}[self.stamps]
        out = [f"stamps={level}"]
        if self.stamps == 2:
            out += [f"every={self.every}", f"marks={self.marks}"]
        if self.stamps == SAMPLE:
            out += [f"every={self.every}", f"cta_mod={self.cta_mod}", f"iter_mod={self.iter_mod}",
                    f"dead_mod={self.dead_mod}"]
        out += [f"evict_a={self.evict_a}", f"evict_b={self.evict_b}"]
        return ",".join(out)

    def as_dict(self) -> dict:
        return {"stamps": self.stamps, "every": self.every, "marks": self.marks,
                "evict_a": self.evict_a, "evict_b": self.evict_b, "cta_mod": self.cta_mod,
                "iter_mod": self.iter_mod, "dead_mod": self.dead_mod, "text": self.text()}


def sample_ks(S: int, every: int) -> list[int]:
    """The k-iterations the sample level marks, in slot order: 0, then every k with
    k mod every = (S - 1) mod every (so S - 1, the last, is always one)."""
    if S < 1 or every < 1:
        raise SpecError(f"sample_ks needs S and every >= 1, got {S}, {every}")
    phase = (S - 1) % every
    ks = list(range(phase, S, every))
    return ks if phase == 0 else [0] + ks


def sample_slot(k: int, S: int, every: int) -> int | None:
    """The kernel's slot rule for mark k (the copy computes the same): phase 0, k // every;
    else 0 for k = 0 and 1 + k // every for k = phase (mod every); None when unmarked."""
    phase = (S - 1) % every
    if phase == 0:
        return k // every if k % every == 0 else None
    if k == 0:
        return 0
    return 1 + k // every if k % every == phase else None


def parse_spec(text: str) -> InstrSpec:
    """`stamps=off|cta|iter,every=N,marks=N,evict_a=none|first|last,evict_b=...`; any
    key left out takes its default (off). Refuses an unknown key or value."""
    kw: dict = {}
    for part in [p for p in (text or "").split(",") if p.strip()]:
        if "=" not in part:
            raise SpecError(f"{part!r} is not key=value")
        k, v = (s.strip() for s in part.split("=", 1))
        if k == "stamps":
            if v not in STAMP_LEVELS:
                raise SpecError(f"stamps {v!r}: one of {sorted(STAMP_LEVELS)}")
            kw["stamps"] = STAMP_LEVELS[v]
        elif k in ("every", "marks", "cta_mod", "iter_mod", "dead_mod"):
            if not v.isdigit():
                raise SpecError(f"{k} {v!r}: a whole number")
            kw[k] = int(v)
        elif k in ("evict_a", "evict_b"):
            kw[k] = v
        else:
            raise SpecError(f"no instrument key {k!r} (stamps every marks cta_mod iter_mod dead_mod "
                            "evict_a evict_b)")
    if kw.get("stamps") == 2 and "marks" not in kw:
        kw["marks"] = DEFAULT_MARKS
    return InstrSpec(**kw)


# --------------------------------------------------------------------------
# the stamp rows
# --------------------------------------------------------------------------

def decode(rows, marks: int = 0, mark_k=None) -> dict:
    """A [num_ctas, HDR + 2 marks] int64 array (numpy or nested lists) as named columns:
    every header column, `iter_top` / `iter_end` [num_ctas, marks] (-1 where no stamp) and
    `mark_k`, which k-iteration each mark column holds (the launch record's, else 0..marks-1)."""
    import numpy as np
    a = np.asarray(rows, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != HDR + 2 * marks:
        raise ValueError(f"a stamp array is [ctas, {HDR} + 2 x {marks}]; got {a.shape}")
    out = {c: a[:, i] for i, c in enumerate(COLUMNS)}
    out["iter_top"] = a[:, HDR::2][:, :marks] if marks else np.zeros((len(a), 0), np.int64)
    out["iter_end"] = a[:, HDR + 1::2][:, :marks] if marks else np.zeros((len(a), 0), np.int64)
    ks = list(range(marks)) if mark_k is None else [int(k) for k in mark_k]
    if len(ks) != marks:
        raise ValueError(f"{len(ks)} mark k values for {marks} marks")
    out["mark_k"] = np.asarray(ks, dtype=np.int64)
    return out


def phases(cols: dict) -> dict:
    """Per CTA, in clock64 cycles (-1 where a stamp is missing): prologue (start to the
    loop), loop (loop to the epilogue), epilogue (epilogue to end) and life (start to
    end, a dead CTA's included); the per-iteration time is the end-to-end difference of
    consecutive top marks (`iter_cycles`)."""
    import numpy as np

    def diff(b, a):
        ok = (cols[a] >= 0) & (cols[b] >= 0)
        return np.where(ok, cols[b] - cols[a], -1)
    top = cols["iter_top"]
    if top.shape[1] >= 2:
        ok = (top[:, 1:] >= 0) & (top[:, :-1] >= 0)
        it = np.where(ok, top[:, 1:] - top[:, :-1], -1)
    else:
        it = np.zeros((len(top), 0), np.int64)
    return {"prologue": diff("prologue_clk", "start_clk"), "loop": diff("epi_clk", "prologue_clk"),
            "epilogue": diff("end_clk", "epi_clk"), "life": diff("end_clk", "start_clk"),
            "iter_cycles": it}


# --------------------------------------------------------------------------
# the installer
# --------------------------------------------------------------------------

@dataclass
class Recorder:
    """What an `install` block launched: one record per fused_moe_kernel launch."""

    spec: InstrSpec
    launches: list = field(default_factory=list)


class _Launcher:
    """One launch: the copy with the spec's constexprs, or (spec None) the kernel as it
    was. With `events`, a (start, end) pair is recorded on the stream right around the
    launch itself (after any stamp buffer is allocated), the same way for both."""

    def __init__(self, kernel, grid, spec: InstrSpec | None, rec: Recorder, alloc, events=None):
        self.kernel, self.grid, self.spec, self.rec, self.alloc = kernel, grid, spec, rec, alloc
        self.events = events

    def __call__(self, *args, **kwargs):
        extra = self.spec.kernel_kwargs() if self.spec is not None else {}
        lay = None
        if self.spec is not None and self.spec.stamps:
            lay = self.spec.layout(int(args[11]), int(kwargs.get("BLOCK_SIZE_K") or 0) or 1)
            extra.update(lay["extra"])
        clash = sorted(set(extra) & set(kwargs))
        if clash:
            raise SpecError(f"the launch already passes {clash}: not upstream's call")
        meta = dict(kwargs)
        g = self.grid(meta) if callable(self.grid) else self.grid
        ctas = int(g[0])
        entry = {"launch": len(self.rec.launches), "ctas": ctas, "N": int(args[10]),
                 "K": int(args[11]), "EM": int(args[12]), "num_valid_tokens": int(args[13]),
                 "BLOCK_SIZE_M": meta.get("BLOCK_SIZE_M"), "BLOCK_SIZE_N": meta.get("BLOCK_SIZE_N"),
                 "BLOCK_SIZE_K": meta.get("BLOCK_SIZE_K"), "GROUP_SIZE_M": meta.get("GROUP_SIZE_M"),
                 "num_stages": meta.get("num_stages"), "num_warps": meta.get("num_warps"),
                 "MUL_ROUTED_WEIGHT": meta.get("MUL_ROUTED_WEIGHT"), "buffer": None}
        if lay is not None:
            entry["buffer"] = self.alloc(ctas, lay["width"], args[0])
            entry.update(width=lay["width"], marks=lay["marks"], mark_k=lay["mark_k"])
            extra["stamps_ptr"] = entry["buffer"]
        self.rec.launches.append(entry)
        entry["top_k"] = meta.get("top_k")
        if self.events is None:
            out = self.kernel[self.grid](*args, **kwargs, **extra)
        else:
            start, end = self.events(), self.events()
            start.record()
            out = self.kernel[self.grid](*args, **kwargs, **extra)
            end.record()
            entry["events"] = (start, end)
        # Triton's launch returns the CompiledKernel: its registers, shared, PTX and cubin
        # are read off it, keyed by this launch's config (config_key), never off a cache scan
        entry["compiled"] = out
        return out


class _Wrapper:
    """Stands in for vLLM's `fused_moe_kernel`: `wrapper[grid](...)` launches the copy
    (or, spec None, the plain kernel it wraps)."""

    def __init__(self, kernel, spec: InstrSpec | None, rec: Recorder, alloc, events=None):
        self.kernel, self.spec, self.rec, self.alloc, self.events = kernel, spec, rec, alloc, events

    def __getitem__(self, grid):
        return _Launcher(self.kernel, grid, self.spec, self.rec, self.alloc, self.events)


def _torch_alloc(ctas: int, width: int, like):
    import torch
    return torch.full((ctas, width), -1, dtype=torch.int64, device=like.device)


class BufferPool:
    """Stamp buffers allocated and filled with -1 BEFORE the L2 flush (gate-fix diagnosis
    item 4: a torch.full inside the call left 11 to 16 MB of dirty lines after the flush).
    `reserve` makes one buffer per (ctas, width) a call's launches will ask for (read off an
    earlier call's records), the caller then flushes and calls; the installer's alloc takes
    them in order. A launch with no reserved buffer still gets one, counted in `late`."""

    def __init__(self, make=_torch_alloc):
        self.make, self.ready, self.late = make, [], 0

    @staticmethod
    def shapes(launches) -> list[tuple[int, int]]:
        return [(int(r["ctas"]), int(r["width"])) for r in launches if r.get("width")]

    def reserve(self, shapes, like) -> None:
        self.ready = [((c, w), self.make(c, w, like)) for c, w in shapes]

    def __call__(self, ctas: int, width: int, like):
        for i, (shape, buf) in enumerate(self.ready):
            if shape == (ctas, width):
                del self.ready[i]
                return buf
        self.late += 1
        return self.make(ctas, width, like)


def _module(module):
    if module is None:
        import importlib
        module = importlib.import_module(VLLM_MODULE)
    return module


@contextlib.contextmanager
def timed_plain(*, module=None, events=None):
    """The PLAIN kernel, untouched, with a (start, end) event pair around each launch: the
    perturbation unit's reference, timed exactly as `install(..., events=)` times the copy."""
    module = _module(module)
    original = module.fused_moe_kernel
    if isinstance(original, _Wrapper):
        raise SpecError("an instrumented kernel is already installed")
    rec = Recorder(InstrSpec())
    module.fused_moe_kernel = _Wrapper(original, None, rec, None, events)
    try:
        yield rec
    finally:
        module.fused_moe_kernel = original


@contextlib.contextmanager
def install(spec: InstrSpec, *, module=None, kernel=None, alloc=None, events=None):
    """Swap `module.fused_moe_kernel` for the instrumented copy for the block's duration.

    `module` defaults to vLLM's fused_moe module, `kernel` to the copy's
    `fused_moe_kernel`, `alloc` to a torch int64 buffer filled with -1 on A's device (the
    suite passes fakes). The original attribute is restored on exit, an exception included.
    """
    module = _module(module)
    if kernel is None:
        from . import fused_moe_instr
        kernel = fused_moe_instr.fused_moe_kernel
    rec = Recorder(spec)
    original = module.fused_moe_kernel
    if isinstance(original, _Wrapper):
        raise SpecError("an instrumented kernel is already installed")
    module.fused_moe_kernel = _Wrapper(kernel, spec, rec, alloc or _torch_alloc, events)
    try:
        yield rec
    finally:
        module.fused_moe_kernel = original


def config_key(entry: dict) -> tuple:
    """The compile identity of a launch record: what makes Triton compile anew."""
    return tuple(entry.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                        "num_stages", "num_warps", "MUL_ROUTED_WEIGHT", "top_k"))


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def excerpt_of(source: str) -> str:
    """The two upstream functions out of a whole fused_moe.py, at UPSTREAM.json's lines."""
    lines = source.splitlines(True)
    (a0, a1), (b0, b1) = (UPSTREAM["functions"][f] for f in ("write_zeros_to_output",
                                                             "fused_moe_kernel"))
    return "".join(lines[a0 - 1:a1]) + "\n\n" + "".join(lines[b0 - 1:b1])


def installed_check(path: Path | str) -> dict:
    """Whether an installed vllm/.../fused_moe.py is the upstream file the copy was taken
    from: the whole file's sha256 and the excerpt's. The perturb unit refuses on a miss."""
    text = Path(path).read_bytes()
    file_sha = hashlib.sha256(text).hexdigest()
    ex_sha = hashlib.sha256(excerpt_of(text.decode()).encode()).hexdigest()
    return {"path": str(path), "file_sha256": file_sha, "excerpt_sha256": ex_sha,
            "file_matches": file_sha == UPSTREAM["file_sha256"],
            "excerpt_matches": ex_sha == UPSTREAM["excerpt_sha256"]}
