#!/usr/bin/env python3
"""One real CUDA kernel, launched once, so a profiler has something to count.

WHY THIS FILE EXISTS, AND WHAT IT COST NOT TO HAVE IT.
`scripts/dram_counter_route.py --probe` used to decide "is a DRAM counter
readable on this box" by running

    ncu --metrics dram__bytes_read.sum /bin/true

`/bin/true` creates no CUDA context and launches no kernel. ncu attaches, finds
nothing to profile, prints `==WARNING== No kernels were profiled.`, and exits 0.
THE PERMISSION CHECK NEVER HAPPENS, because on this driver it happens at the
first counter collection -- that is, at the first kernel launch -- and not at
attach. So the probe proved ncu could start a process and proved nothing at all
about counters, and it returned OPEN on every box it was ever run on.

On 2026-09-15 a rented H200 booked two 120-minute counter arms on that word.
Both died in 35 seconds with

    ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
              NVIDIA GPU Performance Counters on the target device 0.

confirmed on the same pod by `capsh` (!cap_sys_admin AND !cap_perfmon) and by
running ncu by hand over a torch matmul. The published 2026-09-09 and
2026-09-10 probe payloads carry ncu's own admission in a field the probe
captured and never read: `output_head == "==WARNING== No kernels were
profiled."` beside `cause == "attached with no permission error"`.

WHY IT IS A FILE AND NOT A `-c` STRING. Same rule as `moe/bench/read_probe.py`:
a probe kernel lives in a real file on disk. A probe piped to an interpreter
has no source path, is invisible to `git grep`, and cannot be run by hand by an
operator on the pod who wants to see the failure for themselves. The one-liner
this replaces already existed twice in the tree, spelled differently each time.

WHY THE KERNEL IS TORCH AND NOT TRITON. `read_probe` needs Triton because it is
measuring a read pattern. This measures nothing: it needs a launch that reads
every element of a real buffer and certainly finishes. `torch.Tensor.add_` is
both, in one launch, with no compile step. Triton here would add a JIT to the
critical path of a permission check. This paragraph said "over a buffer larger
than any L2 in this study" until 2026-09-15, which `PROBE_ELEMENTS` below
contradicts in the same file: 4 MiB sits INSIDE the L2 of both cards, on
purpose, because the probe asserts that a value came back and never that the
value is large.

WHY THE BUFFER IS FILLED ON THE HOST. `ncu --launch-count 1` profiles the
FIRST kernel launch of the process. `torch.ones(N, device="cuda")` is `empty`
plus a `fill_` KERNEL, so a first draft of this file profiled the fill -- a
write-only launch -- while every comment in it described the `add_`. The
buffer is therefore built on the CPU and copied, which is a `cudaMemcpyAsync`
and not a kernel launch, so the in-place add is launch zero and the launch the
caller's payload names is the launch it counted.

WHAT THIS IS NOT. It is not a benchmark and nothing timed here is reported.
The caller asks exactly one question of the profile it produces: did ncu return
a NUMBER for the registered metric. The number's VALUE is not asserted by
anyone -- a counter that reads 0 on a 4 MiB buffer is a reading, and a counter
that is refused is not. That is also why the paragraph above is a matter of
honest reporting and not of correctness: had a torch build still slipped a
launch in ahead of the add, the permission question would still be answered.

Exit codes are this file's own, not `moe.bench.exit_codes`: the caller reads the
marker line, and these say which of four worlds the child landed in.
"""
from __future__ import annotations

import sys

#: The one line this file prints, which is the only thing the caller parses.
#: A prefix rather than a bare word so it cannot be confused with torch's or
#: ncu's own chatter in an interleaved stream.
MARKER = "COUNTER_PROBE"

#: The four worlds, and they are four because a log cannot tell them apart.
#: LAUNCHED is the only one where ncu was given a kernel to count, so it is the
#: only one from which "counters are readable" or "counters are refused" can be
#: concluded at all. The other three are facts about the PROBE INTERPRETER, and
#: reporting any of them as a counter verdict is how the defect above happened.
LAUNCHED = "LAUNCHED"
NO_TORCH = "NO_TORCH"
NO_CUDA_DEVICE = "NO_CUDA_DEVICE"
LAUNCH_FAILED = "LAUNCH_FAILED"

WORDS: tuple[str, ...] = (LAUNCHED, NO_TORCH, NO_CUDA_DEVICE, LAUNCH_FAILED)

#: fp32 elements in the probe buffer. 1 Mi elements is 4 MiB, which is chosen
#: against the L2 of the cards this study rents rather than picked round: the
#: H200 carries 60 MB of L2 and the A100 40 MB, so 4 MiB would sit inside
#: either and a warm re-read could be served without touching DRAM. That is
#: FINE and deliberate -- ncu's default `--cache-control all` flushes before
#: each replay pass, so the read is cold, and in any case this probe asserts
#: that a value came back and never that the value is positive. The size is
#: kept small because the whole point is that the probe is cheap: 4 MiB is one
#: allocation, no model, no weights, and it is gone when the process exits.
PROBE_ELEMENTS = 1 << 20


def probe() -> tuple[str, str]:
    """Launch one kernel. Returns `(word, detail)`, never raises."""
    try:
        import torch
    except Exception as exc:                                # noqa: BLE001
        return NO_TORCH, f"cannot import torch: {type(exc).__name__}: {exc}"
    try:
        if not torch.cuda.is_available():
            return NO_CUDA_DEVICE, "torch.cuda.is_available() is False"
        name = torch.cuda.get_device_name(0)
    except Exception as exc:                                # noqa: BLE001
        # A driver that is present but refusing to initialise is NOT the same as
        # no card, but it is the same as "no kernel can be launched here", which
        # is the only distinction this probe is entitled to draw.
        return NO_CUDA_DEVICE, f"CUDA did not initialise: {type(exc).__name__}: {exc}"
    try:
        # THE HOST FILL IS DELIBERATE, see the module docstring: `torch.ones`
        # with `device="cuda"` would launch a `fill_` kernel ahead of the add
        # and `--launch-count 1` would profile that one. `ones` on the CPU is
        # no launch, and `.to("cuda")` is a memcpy, so the add below is the
        # first kernel this process launches. The values are defined either
        # way, so no compiler pass can argue the read away.
        buf = torch.ones(PROBE_ELEMENTS, dtype=torch.float32).to("cuda")
        # One in-place add over a real buffer: it reads every element and
        # writes every element, so `dram__bytes_read.sum` has something to be.
        buf.add_(1.0)
        torch.cuda.synchronize()
        # Read one element back to prove the launch retired. `.item()` is a
        # device-to-host copy, not a second kernel, and it happens AFTER the
        # add in any case.
        got = float(buf[0].item())
    except Exception as exc:                                # noqa: BLE001
        return LAUNCH_FAILED, f"{type(exc).__name__}: {exc}"
    if got != 2.0:
        return LAUNCH_FAILED, f"the probe kernel ran and returned {got}, expected 2.0"
    return LAUNCHED, f"{name}: one add_ over {PROBE_ELEMENTS} fp32"


def main() -> int:
    word, detail = probe()
    print(f"{MARKER} {word} {detail}", flush=True)
    return 0 if word == LAUNCHED else 1


if __name__ == "__main__":
    sys.exit(main())
