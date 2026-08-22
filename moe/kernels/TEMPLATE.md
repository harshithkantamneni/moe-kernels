# Writing a kernel in this repo

Copy this checklist, not code. The tiling strategy, the scheduling, and the
optimisation ideas are the point of the project and are yours to write.

## 1. Pick your span

Decide which contiguous canonical stages your kernel covers:

    router -> permute -> up_gemm -> act -> down_gemm -> unpermute

A plain grouped GEMM is `covers = ("up_gemm",)`. A fused up-projection plus
SwiGLU is `("up_gemm", "act")`. The fused down-projection plus scatter is
`("down_gemm", "unpermute")`.

## 2. Declare the span

```python
from ..stages import StageSpan, register
from ..state import MoEState


@register
class MyGroupedGemm(StageSpan):
    name = "tri_grouped_gemm_v1"     # unique; becomes the CSV impl column
    covers = ("up_gemm",)
    env = "base"                      # base | vllm | sglang
    requires_cuda = True
    cuda_graph_safe = False           # flip to True only when you have removed
                                      # every .item() and host-side expert loop
    dtypes = ("bf16",)

    def supports(self, spec) -> bool:
        # Reject geometries your tiling cannot handle, rather than producing
        # silently wrong numbers. Example: a BLOCK_N that assumes F % 128 == 0.
        return super().supports(spec) and spec.model.intermediate_size % 128 == 0

    def __call__(self, st: MoEState) -> None:
        x_perm, offsets = st.require("x_perm", "expert_offsets")
        ...
        st.h_up = out    # every field in self.writes must be produced
```

`self.writes` is derived automatically from `covers`. The pipeline validates it.

## 3. What the harness guarantees you

- `st.expert_offsets` is `[E+1]` int32 on device, monotonically non-decreasing,
  `offsets[0] == 0`, `offsets[E] == spec.rows`. Experts with zero tokens are
  normal and common under skew: `offsets[e] == offsets[e+1]`.
- `st.x_perm` rows are already in expert-contiguous order, stable within expert.
- `st.perm_index[r]` is the flat `token * top_k + slot` index of permuted row r.
- Weight layouts: `w1` is `[E, 2F, H]` as `[gate | up]` along dim 1, `w2` is
  `[E, H, F]`. Both match the vLLM and SGLang convention so baselines are
  comparable without a transpose.

## 4. Constraints that are not negotiable

- **No `.item()`, no `.tolist()`, no host-side loop over experts.** Those break
  CUDA-graph capture, which is how MoE inference actually runs. If you need a
  launch grid that depends on group sizes, it has to come from a device-side
  indirection table.
- **Do not assume every expert is non-empty.** At high skew most are empty.
- **Do not assume group sizes are multiples of BLOCK_M.** That assumption is
  precisely the TritonMoE limitation this project exists to attack.

## 5. Before you benchmark it

`pytest tests/ -m gpu -k your_kernel_name` must pass. The driver refuses to
benchmark an implementation that has no passing correctness record for the
cell it is about to run.
