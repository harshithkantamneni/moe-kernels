# Adding a kernel: Triton, CUDA, CuTe, or anything else

## The harness is language-agnostic

The seam is a Python callable. A `StageSpan.__call__` receives a `MoEState`,
reads tensors off it, and assigns tensors back. Nothing in the harness cares how
those tensors were produced. There is no Triton dependency in `moe/` at all:
grep for `import triton` and the only hit is a version string in `runtime_info`.

So the answer to "CUDA only, or Triton, or something else" is: whichever you
want, including several at once in the same sweep, compared against each other
on identical inputs by the same oracle. That is the point of the registry.

```python
@register
class MyKernel(StageSpan):
    name = "tri_grouped_gemm_v1"
    covers = ("up_gemm",)
    dtypes = ("bf16",)
    cuda_graph_safe = False          # a claim; the harness measures the truth

    def __call__(self, st):
        x_perm, offsets = st.require("x_perm", "expert_offsets")
        out = torch.empty((st.spec.rows, 2 * st.spec.model.intermediate_size),
                          dtype=x_perm.dtype, device=x_perm.device)
        launch_however_you_like(x_perm, st.weights.w1, offsets, out)
        st.h_up = out
```

## Triton

Nothing to set up. `@triton.jit` the kernel, compute the grid, launch.

```python
import triton
import triton.language as tl

@triton.jit
def _grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, expert_ids_ptr, ...,
                         BLOCK_M: tl.constexpr, ...):
    ...            # yours

def launch(x_perm, w1, offsets, out, block_m=64):
    grid = lambda META: (triton.cdiv(padded_rows, META["BLOCK_M"]) *
                         triton.cdiv(N, META["BLOCK_N"]),)
    _grouped_gemm_kernel[grid](x_perm, w1, out, expert_ids, ...)
```

`TRITON_CACHE_DIR` is on the network volume (see RUNPOD.md), so JIT compilation
and autotuning results survive the pod.

**Autotune and graph capture interact.** `triton.autotune` benchmarks its config
space on the first call with a new key, and that involves synchronising, which
cannot happen inside a graph capture. `time_graph` runs three warmup iterations
on a side stream before capturing, which is enough to populate the autotune
cache. If you build a very large autotune space, warm it once outside the timed
path rather than discovering the cost inside it.

## CUDA C++

Two routes. JIT is the one to start with, because it needs no build system and
`TORCH_EXTENSIONS_DIR` can live on the volume so it compiles once.

```python
# moe/kernels/cuda_grouped_gemm.py
from pathlib import Path
from torch.utils.cpp_extension import load

from moe.kernels import cuda_arch_flags

_SRC = Path(__file__).parent / "csrc"
_ext = load(name="moe_grouped_gemm",
            sources=[str(_SRC / "grouped_gemm.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", *cuda_arch_flags()],
            verbose=False)

@register
class CudaGroupedGemm(StageSpan):
    name = "cu_grouped_gemm_v1"
    covers = ("up_gemm",)
    def __call__(self, st):
        ...
        _ext.grouped_gemm(x_perm, st.weights.w1, offsets, out)
        st.h_up = out
```

`cuda_arch_flags()` reads the attached device rather than hardcoding one. On an
H200 it emits `-gencode=arch=compute_90a,code=sm_90a`, and the `a` matters: the
`a` variants are what expose wgmma and the TMA instructions, so plain `sm_90`
builds for the part but loses what the part exists for. Hardcoding Hopper is
also wrong the moment a sweep runs on whatever GPU was available, since an A100
is `sm_80` and RTX Blackwell is `sm_120a`.

Because `moe/kernels/load_all()` catches import failures with a warning, a
kernel whose extension fails to build does not break the harness or the CPU test
suite. It just does not appear in the registry, and `--dry-run` says so.

For a kernel you want to keep, move to a real `setup.py` with
`CUDAExtension` and build it in `setup_runpod.sh`, so compilation is not on the
metered path at all.

## CUTLASS / CuTe DSL

`nvidia-cutlass-dsl` is already in the vLLM and SGLang environments (they pin
4.6.0 and 4.6.2 respectively — one of the four conflicts that forced the
separate venvs). It is a Python DSL, so it plugs in exactly like Triton: build
the kernel, get a callable, launch it from `__call__`. Add it to
`requirements/base.txt` if you want it in your own environment rather than
borrowing a baseline's.

## cuTile

Real, generally available, and it supports your hardware. It is also the one
option where the sequencing matters, so this section is longer than the others.

**What it is.** NVIDIA's tile-based model. The naming stacks up as CUDA Tile
(the model) / CUDA Tile IR (the MLIR-based virtual ISA) / cuTile Python (the
frontend) / CUDA Tile C++ (new in CUDA 13.3). The Python frontend is an
array-level DSL close in spirit to Triton: `@ct.kernel`, `ct.bid(0)` for the
block id, `ct.load`/`ct.store`, `ct.mma`, launched with
`ct.launch(stream, grid, kernel, args)`. Individual threads are not exposed and
intra-block sync is not permitted; the compiler distributes tile ops onto
threads.

Two differences from Triton matter for a grouped GEMM. `ct.load` indexes in
**tile space** with a `padding_mode` rather than pointer arithmetic plus an
explicit mask. And `Array.slice(axis, start, stop)` accepts **device-resident
0-D tiles as bounds**, which gives a ragged segment a real zero-copy sliced
view. Triton has no direct equivalent.

**Install.** The package is `cuda-tile`; the import namespace is `cuda.tile`.
Two name traps: `pip install cutile` and `pip install cutile-python` both hit
NVIDIA-owned `0.0.0a0` placeholder stubs on PyPI, yet `cutile-python` *is* the
correct name on conda-forge. See `requirements/cutile.txt`, which pins the
compiler explicitly because cuda-tile's own `>=13.2` floor also admits 13.2.x,
which has no Hopper support and fails on sm_90.

It gets its **own venv**, and is not in the default `setup_runpod.sh` set:
torch 2.13.0 pins `cuda-toolkit==13.0.3` while `cuda-tile[tileiras]` wants
`>=13.2,<13.4`. Empty intersection. Prove cuTile works alone before attempting
coexistence.

```bash
bash scripts/setup_runpod.sh cutile
python scripts/preflight_cutile.py
```

**Hopper support is real but recent.** It landed in cuTile 1.4.0 / Tile IR 13.3
on 2026-05-26. Three sources will tell you otherwise and all three are stale:
the `cutile-python` README on main (scoped to 13.2, true of 13.2), the original
December 2025 launch blog, and arXiv 2604.23466 which says flatly that cuTile
does not support sm_90 (it tested 1.1.0). Expect to meet all three; ignore them.

**Torch interop is first-class and zero-copy.** Torch CUDA tensors pass straight
into `ct.launch` through a dedicated `torch._C._to_dlpack` fast path, strides
honoured, non-contiguous views fine. Pass `stream=torch.cuda.current_stream()`.

**CUDA graphs: yes in practice, undocumented in theory.** NVIDIA's own
`test/test_cudagraph.py` captures and replays both a trivial kernel and a
`ct.mma` matmul. But the documentation is entirely silent on graph capture, so
this is behaviour that could regress without a release note. Three constraints
to design around:

- **Python list arguments are explicitly rejected under capture.** This
  disqualifies cuTile's own shipped grouped GEMM, which takes lists of tensors.
- **The grid is a host-side tuple.** No device-side launch, no grid-from-tensor.
  Oversize from static shapes and skip on device, exactly as with Triton.
- **JIT and autotune must happen outside the capture region**, and each distinct
  `ct.Constant` compiles a separate kernel.

Note issue #80, "CuTile doesn't support graph mode", is CLOSED and is about
torch.compile/Dynamo, not CUDA graphs. It is the first hit when you search.

**A ragged GEMM is expressible.** Device scalars drive conditionals, loop trip
counts, tile indices, and `Array.slice` bounds. NVIDIA's TileGym ships
`ragged_bmm.py` taking a device `m_indptr`, plus a fully device-side
`moe_align_block_size` — the piece that makes a device-offsets MoE capturable
end to end. One gotcha worth internalising: `.item()` inside a cuTile *kernel*
is **not** a host sync, it is `Tile.item()` returning a scalar tile, so a naive
grep for sync patterns produces false positives.

### Do the SASS gate before anything else

There is **zero published cuTile performance data on Hopper, for any workload**.
TileGym validates only Blackwell and Ampere, its autotune tables have no sm_90
block, and every published number is tagged with a version predating Hopper
support. You would not be benchmarking a known quantity; you would be producing
the first datapoint, tuning included.

So settle the decisive question in ten minutes rather than twenty pod-hours:

```bash
python scripts/preflight_cutile.py --kernel your_module:your_kernel
# export a cubin at sm_90, then:
cuobjdump -sass k.cubin | grep -oE '(HGMMA|GMMA|QGMMA|HMMA|IMMA)[^ ]*' | sort | uniq -c
```

GMMA or HGMMA present means the Hopper tensor-core fast path. Only `HMMA.16816`
means you would be measuring codegen maturity rather than your kernel design —
and "cuTile Hopper codegen is not yet on the WGMMA path" is itself a publishable
finding that cost one pod-hour.

Also budget for open `tileiras` SIGSEGV bugs (#94-#97) sitting on exactly the
MMA-loop and advanced-index paths a hand-written grouped GEMM uses, and for
NVIDIA's repo contradicting itself on whether TMA works on sliced arrays feeding
`ct.mma`.

## Anything else

If it can be called from Python and it reads and writes torch tensors, it works:
CUDA Python, numba.cuda, a raw `cuLaunchKernel` through ctypes, or a compiled
artifact wrapped in a torch custom op.

## Baselines worth having before you write anything

Three mature Hopper grouped GEMMs exist today and need none of the above.
`moe/baselines/torch_grouped_mm.py` is already wired and tested.

| backend | install | notes |
|---|---|---|
| `torch.nn.functional.grouped_mm` | none, in torch 2.13.0 | CUTLASS sm90, device int32 offsets, no host sync, **no tile-alignment constraint on the ragged M dimension**. bf16, group_count < 1024. |
| CUTLASS CuTe DSL grouped GEMM | `nvidia-cutlass-dsl` | WGMMA + TMA + persistent warp-specialized, Hopper-native. The mature Python path on sm_90 and the right ceiling. |
| DeepGEMM masked grouped GEMM | `deep-gemm` | FP8. Its **masked** layout is documented for "CUDA graph enabled, CPU unaware of the number of tokens each expert receives". Avoid the contiguous variant, which needs per-expert M-block alignment. |

---

# CUDA graphs: what they change and why the harness cares

## Why they are in the harness at all

Production MoE inference runs under CUDA graphs. Per-kernel launch overhead is a
few microseconds, and a decode step through a 58-layer MoE model launches
hundreds of kernels, so eager launch is a first-order cost in exactly the
small-batch regime this project targets. A kernel that cannot be captured cannot
be used, however fast it is eagerly.

That is why `capture_status` is a column and why a non-capturable implementation
still gets a CSV row saying so, rather than being silently dropped.

## What capture actually does

`torch.cuda.graph(g)` records the work your callable enqueues instead of running
it. On `g.replay()`, the recorded kernels re-run with **the same arguments and
the same memory addresses**. Three consequences:

**1. No host synchronisation, anywhere in the call.** `.item()`, `.tolist()`,
`.cpu()`, `torch.cuda.synchronize()`, `if some_device_tensor > 0`, and a Python
loop over per-expert sizes read from the device all force a sync and abort the
capture. The harness catches that and records `not_capturable` with the
exception message.

**2. The launch grid is frozen at capture time.** This is the constraint that
actually shapes a grouped-GEMM design. You cannot compute the grid from the
number of tokens each expert received, because that is a device value and
reading it is a sync. The technique every capturable implementation uses is:
size the grid from **tensor shapes**, which are static, and exit early on the
device.

vLLM's `moe_align_block_size` allocates
`sorted_token_ids` of size `topk_ids.numel() + num_experts * (BLOCK_M - 1)` —
a shape that depends only on other shapes — launches
`cdiv(that, BLOCK_M) * cdiv(N, BLOCK_N)` blocks, and then has each block do

```
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
if pid_m * BLOCK_M >= num_tokens_post_padded:
    return
```

Blocks beyond the real work retire immediately. That single trick is what makes
the path capturable, and it is worth internalising before you design your
scheduling.

**3. Allocations during capture come from a graph-private pool.** Tensors your
kernel allocates inside the captured region get addresses in that pool, and
every replay reuses those exact buffers.

## Why the harness re-validates the replayed output

Point 3 has a nasty consequence, and it is the reason for a specific piece of
machinery in `driver.py`. Because replay writes into the same buffers every
time, a kernel that leaves part of its output unwritten — a tail tile, an
empty-expert group — sees the **previous replay's correct values still
resident**. It looks right. It is fast, because it is doing less work.

So `time_graph` takes an `on_captured` callback, and the driver uses it to
re-run the correctness comparison against the output the *replay* produced, not
the output the eager prologue produced. A graph row earns its own verdict. If it
fails, the row is written with `correctness_passed=False` and its timing zeroed.

(This is also where a real bug lived: for whole-layer spans the callback was
comparing the prologue's state, which the replays never touch, making the check
a no-op for exactly the vLLM/SGLang-shaped implementation it most needed to
cover. `test_graph_replay_is_validated_for_pipeline_scoped_cells` pins it now.)

## What the harness will tell you

Per cell, per L2 state, you get an eager row and a graph row, with
`capture_status` in `{n/a, captured, not_capturable, skipped}`. The delta
between them is launch overhead **plus** Python dispatch plus allocator work,
because the eager arm re-enters Python every iteration and the graph arm does
not. It is not a pure launch-overhead measurement, and the README says so.

Graph mode is skipped by cost policy when the predicted roofline minimum makes
launch overhead a sub-1% term — at DeepSeek-V3 with 8 tokens that minimum is
already ~0.78 ms against a ~5 µs launch. The skipped row records
`graph_skip_reason` with the arithmetic rather than vanishing.

## Practical checklist for your kernel

- Compute the grid from shapes; exit early on the device.
- Build indirection tables (`sorted_token_ids`, `expert_ids`,
  `num_tokens_post_padded`) with device kernels, never a host loop.
- Do not assume every expert is non-empty; under real skew most are.
- Do not assume group sizes are multiples of `BLOCK_M`. That assumption is the
  thing this project exists to attack.
- Write every output element you claim to write, including the tail.
- Warm the autotune cache before capture.
- Set `cuda_graph_safe = True` only after you have seen `capture_status=captured`
  in a row. It is a claim; the CSV records the fact.
