"""A streaming read with no cross-CTA combine, so the reduction shape is out of it.

WHY THIS IS A SEPARATE MODULE AND NOT A STRING. `triton.jit` calls
`inspect.getsourcelines` on the decorated function and refuses one that has no
file. This project has already shipped a check that compiled a probe kernel by
piping source to stdin: it could not pass on any machine, and the error was
swallowed, so it read as a clean PASS for as long as it existed. Any probe
kernel lives in a real file on disk. That is why this is a module.

WHY THE PROBE EXISTS. `calibrate.measure_bandwidth` measures a pattern that
reads every byte once and calls it `read_reduce`, using `torch.sum` over a 2-D
view reduced along the contiguous axis. That is honest about the traffic -- 1N
in, `rows` floats out -- but it is still a reduction, so what it reports is the
rate at which ATen's reduction can consume a stream. It is a LOWER bound on the
machine's read rate.

The shape has already been worth 1.7% once. Reducing a 1-D buffer to a single
scalar measured 4389.4 GB/s; reducing a `[4096, n/4096]` view along the
contiguous axis measured 4469.6 on the same buffer with the same traffic, purely
because the global combine went away. A change of that size means the shape was
still inside the number, and the way to find out how much is left is to remove
the combine entirely.

WHAT THIS KERNEL DOES. Each program owns `TILES x BLOCK` contiguous elements,
accumulates them into a `BLOCK`-wide vector of registers, reduces that vector
once inside the program, and stores ONE float. There is no atomic, no second
pass, no cross-program communication. Write traffic is `grid x 4` bytes: at the
8 GiB default with BLOCK=1024 and TILES=8 that is 1 MB against 8589 MB read,
0.01%, which is below the run-to-run spread of the measurement.

WHY THE ACCUMULATOR IS A VECTOR AND NOT A SCALAR. `acc += tl.sum(x)` would put a
full in-register tree reduction between every pair of loads and serialise them
behind it, which measures the reduction this probe exists to remove. A vector
add is one FMA-free instruction per element per tile and leaves the loads
independent.

WHY IT CANNOT BE FOLDED AWAY. The values are runtime data and every one of them
reaches the store through the accumulator, so there is no dead-code path. The
guard against a compiler that finds one anyway is arithmetic rather than
inspection: a figure above the memory bus pin rate is impossible, and
`calibrate` refuses it.
"""
from __future__ import annotations

#: Elements per program per tile. 1024 fp32 is 4 KB, a comfortable vectorised
#: load width, and keeps the register accumulator to one vector.
DEFAULT_BLOCK = 1024

#: Tiles each program walks. 8 keeps the grid at 262144 programs for an 8 GiB
#: buffer, which is 1986 CTAs per SM on a 132-SM part: enough to hide launch
#: ramp without making the per-program tail matter.
DEFAULT_TILES = 8


class ProbeUnavailable(RuntimeError):
    """Triton is absent, too old, or refused to compile the probe.

    A typed refusal rather than a returned zero. This module's whole purpose is
    to supply a bandwidth number, and a zero would divide into every ridge that
    quoted it; an absent pattern is the correct answer and the caller records
    the reason.
    """


def _build():
    """Import Triton and define the kernel, or raise `ProbeUnavailable`.

    Deferred because `moe.bench.calibrate` is imported on laptops with no
    Triton, by the off-GPU half of the test suite, and a module-level
    `@triton.jit` would make that import fail.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:                       # pragma: no cover - no triton here
        raise ProbeUnavailable(f"triton is not importable: {exc}") from None

    @triton.jit
    def stream_read(x_ptr, out_ptr, n_elements, TILES: tl.constexpr,
                    BLOCK: tl.constexpr):
        # int64 FROM THE FIRST TERM, and this is not defensive typing. The
        # default buffer is 8 GiB of fp32, so n = 2^31 elements. `tl.program_id`
        # and `tl.arange` are int32, and the largest offset this kernel forms is
        # exactly n - 1 = 2147483647 -- one below the int32 maximum at the last
        # element and OVER it for any buffer larger than 8 GiB, or for the same
        # buffer at a smaller BLOCK. The overflow is silent: offsets go
        # negative, the mask `offs < n_elements` passes them, and the kernel
        # reads outside the tensor at full speed, which reports a perfectly
        # plausible bandwidth. Promoting the base makes the whole expression
        # int64.
        pid = tl.program_id(0).to(tl.int64)
        base = pid * BLOCK * TILES
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(TILES):
            offs = base + i * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            acc += tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    return triton, stream_read


def make_stream_read(a, block: int = DEFAULT_BLOCK, tiles: int = DEFAULT_TILES):
    """`(callable, bytes_read, write_bytes)` for a 1-D fp32 CUDA tensor `a`.

    The callable takes no arguments and reads every element of `a` exactly once,
    so it drops straight into the same timing helper the other patterns use.

    Raises `ProbeUnavailable` rather than returning something that looks like a
    measurement, and raises `ValueError` on a tensor it cannot read once: a
    non-contiguous or non-fp32 buffer would change the byte count without
    changing the reported figure, which is the exact shape of a silent 2x error
    this project has already made once on the write pattern.
    """
    if a.dtype.itemsize != 4 or not a.is_contiguous() or a.dim() != 1:
        raise ValueError(
            f"stream read needs a contiguous 1-D 4-byte tensor, got "
            f"dim={a.dim()} dtype={a.dtype} contiguous={a.is_contiguous()}")

    import torch

    _, kernel = _build()
    n = a.numel()
    per_program = block * tiles
    grid = (n + per_program - 1) // per_program
    out = torch.empty(grid, dtype=torch.float32, device=a.device)

    # `n` stays a PYTHON int. Triton specialises a Python int to i32 when it
    # fits and i64 when it does not, and the comparison `offs < n_elements`
    # promotes to the wider of the two, so an i32 `n` against the kernel's i64
    # offsets is still evaluated in 64 bits. A torch scalar here would be worse
    # than useless: the JIT turns every tensor argument into a POINTER, and the
    # mask would silently compare an address.
    def call():
        kernel[(grid,)](a, out, n, TILES=tiles, BLOCK=block, num_warps=8)

    # Compile and run once here rather than inside the timed loop. A Triton
    # compile is seconds; landing it in the first timed iteration would put it
    # in ms_min, which is the optimistic bound this file publishes.
    try:
        call()
        torch.cuda.synchronize()
        # AND CHECK THE ANSWER, which is the only thing that can distinguish a
        # fast kernel from one that read the wrong memory. Both failure modes
        # this probe is exposed to -- loads eliminated as dead, and offsets
        # overflowing int32 into negative addresses -- produce a plausible GB/s
        # and a wrong sum. Accumulated in fp64 because the fp32 total of 2^31
        # ones is not the fp32 total of anything.
        want = torch.sum(a, dtype=torch.float64).item()
        got = torch.sum(out, dtype=torch.float64).item()
        if want == 0.0 or abs(got - want) > 1e-6 * abs(want):
            raise ProbeUnavailable(
                f"the stream-read probe summed {got!r} where the buffer sums "
                f"{want!r}. It did not read this tensor exactly once, so its "
                "bandwidth figure is not about this tensor.")
    except ProbeUnavailable:
        raise
    except Exception as exc:                          # noqa: BLE001
        # Deliberately broad. Triton raises CompilationError, the driver raises
        # RuntimeError, and an old Triton without tl.static_range raises
        # AttributeError from inside the JIT. All three mean the same thing to
        # the caller: there is no stream figure this session.
        raise ProbeUnavailable(
            f"the stream-read probe did not run: {type(exc).__name__}: {exc}") from None

    return call, n * a.dtype.itemsize, grid * 4
