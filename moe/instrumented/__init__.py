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
STAMP_LEVELS = {"off": 0, "cta": 1, "iter": 2}
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

    def __post_init__(self):
        if self.stamps not in (0, 1, 2):
            raise SpecError(f"stamps {self.stamps}: 0 off, 1 cta, 2 iter")
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
        """int64 columns per stamp row."""
        return HDR + 2 * (self.marks if self.stamps == 2 else 0)

    def kernel_kwargs(self) -> dict:
        """The instrument constexprs the copy takes, beside upstream's own arguments."""
        return {"STAMPS": self.stamps, "STAMP_EVERY": self.every,
                "STAMP_MARKS": self.marks if self.stamps == 2 else 0,
                "EVICT_A": EVICT[self.evict_a], "EVICT_B": EVICT[self.evict_b]}

    def text(self) -> str:
        level = {v: k for k, v in STAMP_LEVELS.items()}[self.stamps]
        out = [f"stamps={level}"]
        if self.stamps == 2:
            out += [f"every={self.every}", f"marks={self.marks}"]
        out += [f"evict_a={self.evict_a}", f"evict_b={self.evict_b}"]
        return ",".join(out)

    def as_dict(self) -> dict:
        return {"stamps": self.stamps, "every": self.every, "marks": self.marks,
                "evict_a": self.evict_a, "evict_b": self.evict_b, "text": self.text()}


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
        elif k in ("every", "marks"):
            if not v.isdigit():
                raise SpecError(f"{k} {v!r}: a whole number")
            kw[k] = int(v)
        elif k in ("evict_a", "evict_b"):
            kw[k] = v
        else:
            raise SpecError(f"no instrument key {k!r} (stamps every marks evict_a evict_b)")
    if kw.get("stamps") == 2 and "marks" not in kw:
        kw["marks"] = DEFAULT_MARKS
    return InstrSpec(**kw)


# --------------------------------------------------------------------------
# the stamp rows
# --------------------------------------------------------------------------

def decode(rows, marks: int = 0) -> dict:
    """A [num_ctas, HDR + 2 marks] int64 array (numpy or nested lists) as named columns:
    every header column, `iter_top` / `iter_end` [num_ctas, marks] (-1 where no stamp)."""
    import numpy as np
    a = np.asarray(rows, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != HDR + 2 * marks:
        raise ValueError(f"a stamp array is [ctas, {HDR} + 2 x {marks}]; got {a.shape}")
    out = {c: a[:, i] for i, c in enumerate(COLUMNS)}
    out["iter_top"] = a[:, HDR::2][:, :marks] if marks else np.zeros((len(a), 0), np.int64)
    out["iter_end"] = a[:, HDR + 1::2][:, :marks] if marks else np.zeros((len(a), 0), np.int64)
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
        if self.spec is not None and self.spec.stamps:
            entry["buffer"] = self.alloc(ctas, self.spec.width, args[0])
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
