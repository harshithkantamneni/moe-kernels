"""Naive, obviously-correct torch implementations of every canonical stage.

These are the correctness oracle. They are written for readability, not speed:
the grouped GEMMs are literal python `for e in range(E)` loops. They run on CPU,
so the whole dataflow is debuggable on a laptop at `toy` geometry.

Numerics: each stage accumulates in fp32 and casts back to the working dtype,
which is what a tensor-core kernel does. `golden_forward` instead keeps fp32
throughout and is the reference the correctness tests compare against.
"""
from __future__ import annotations

import torch
import torch.nn.functional as tF

from ..spec import BenchSpec, torch_dtype
from ..stages import StageSpan, register
from ..state import MoEState, MoEWeights

# --------------------------------------------------------------------------
# helpers shared by the spans and by golden_forward
# --------------------------------------------------------------------------

def gate_scores(logits, cfg):
    """Router logits -> per-expert scores, using the model's own gating function.

    Mixtral, Qwen2 and DeepSeek-V2-Lite use softmax over experts. DeepSeek-V3
    uses an independent sigmoid per expert, so its scores do not sum to 1.
    """
    if cfg.gate_fn == "sigmoid":
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def gate_weights(scores, ids, cfg):
    """Combine weights for the chosen experts.

    Order matters and is model-specific: gather, then renormalise if the model
    does, then apply the routed scaling factor. DeepSeek-V3 multiplies by 2.5
    AFTER renormalising, so its combine weights deliberately do not sum to 1.
    Qwen2 and DeepSeek-V2-Lite set norm_topk_prob=false and do not renormalise
    at all.
    """
    w = torch.gather(scores, 1, ids.long()) if ids is not None else scores
    if cfg.norm_topk_prob:
        # The 1e-20 matches DeepSeek's reference implementation and keeps an
        # all-zero sigmoid row from producing NaN.
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    if cfg.routed_scaling_factor != 1.0:
        w = w * cfg.routed_scaling_factor
    return w.float()


def route(x, wg, cfg):
    """[T,H] x [E,H] -> (logits [T,E], topk_ids [T,k] int32, topk_weights [T,k] fp32).

    Deliberate simplification: DeepSeek-V3's group-limited (noaux_tc) expert
    SELECTION is not modelled. Every benchmark cell forces expert selection from
    a trace or a parametric distribution, so the selection rule never affects
    the grouped GEMM under measurement; only the gate weights do, and those are
    modelled faithfully above.
    """
    logits = x.float() @ wg.float().t()
    scores = gate_scores(logits, cfg)
    _, topk_ids = torch.topk(scores, cfg.top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    return logits, topk_ids, gate_weights(scores, topk_ids, cfg)


def weights_for_forced_ids(logits, forced_ids, cfg):
    """Replay path: expert choice comes from a trace, gate weights still come
    from the model's own logits so the combine step stays meaningful."""
    return gate_weights(gate_scores(logits, cfg), forced_ids, cfg)


def build_permutation(topk_ids, num_experts: int):
    """topk_ids [T,k] -> (expert_offsets [E+1] int32, perm_index [T*k] int32).

    perm_index[r] is the flat (token*k + slot) index whose row lands at permuted
    position r. A stable sort keeps token order within an expert, which makes
    results reproducible and makes tiling bugs easier to read.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    perm_index = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return offsets, perm_index


def grouped_gemm_loop(a, w, expert_offsets, out_features: int):
    """The oracle grouped GEMM: one dense GEMM per expert, python loop, fp32 accum.

    a: [Ntot, K]  w: [E, out_features, K]  expert_offsets: [E+1]
    """
    ntot = a.shape[0]
    out = torch.empty((ntot, out_features), dtype=a.dtype, device=a.device)
    off = expert_offsets.tolist()
    for e in range(w.shape[0]):
        lo, hi = off[e], off[e + 1]
        if hi <= lo:
            continue  # an expert can receive zero tokens; this is the normal case at high skew
        out[lo:hi] = (a[lo:hi].float() @ w[e].float().t()).to(a.dtype)
    return out


def swiglu(h_up):
    """h_up is [Ntot, 2F] laid out as [gate | up] along dim 1."""
    gate, up = h_up.chunk(2, dim=-1)
    return (tF.silu(gate.float()) * up.float()).to(h_up.dtype)


def combine(y_perm, perm_index, topk_weights, num_tokens: int, top_k: int):
    """Scatter permuted rows back and accumulate the weighted sum per token."""
    flat_w = topk_weights.reshape(-1)[perm_index.long()]
    contrib = y_perm.float() * flat_w.unsqueeze(-1)
    rows = (perm_index.long() // top_k)
    y = torch.zeros((num_tokens, y_perm.shape[-1]), dtype=torch.float32, device=y_perm.device)
    y.index_add_(0, rows, contrib)
    return y.to(y_perm.dtype)


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------

class _Ref(StageSpan):
    env = "base"
    requires_cuda = False
    cuda_graph_safe = False   # python loop over experts, host-side offsets
    dtypes = ("fp32", "fp16", "bf16")


@register
class RefRouter(_Ref):
    name = "ref_router"
    covers = ("router",)

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        logits, ids, w = route(st.x, st.weights.wg, cfg)
        if st.forced_topk_ids is not None:
            ids = st.forced_topk_ids.to(torch.int32)
            w = weights_for_forced_ids(logits, ids, cfg)
        st.router_logits, st.topk_ids, st.topk_weights = logits, ids, w


@register
class RefPermute(_Ref):
    name = "ref_permute"
    covers = ("permute",)

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        offsets, perm = build_permutation(st.topk_ids, cfg.num_experts)
        st.expert_offsets, st.perm_index = offsets, perm
        st.x_perm = st.x[perm.long() // cfg.top_k]


@register
class RefUpGemm(_Ref):
    name = "ref_up_gemm"
    covers = ("up_gemm",)

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        st.h_up = grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets, 2 * cfg.intermediate_size
        )


@register
class RefAct(_Ref):
    name = "ref_act"
    covers = ("act",)

    def __call__(self, st: MoEState) -> None:
        st.h_act = swiglu(st.h_up)


@register
class RefDownGemm(_Ref):
    name = "ref_down_gemm"
    covers = ("down_gemm",)

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        st.y_perm = grouped_gemm_loop(
            st.h_act, st.weights.w2, st.expert_offsets, cfg.hidden_size
        )


@register
class RefUnpermute(_Ref):
    name = "ref_unpermute"
    covers = ("unpermute",)

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        st.y = combine(st.y_perm, st.perm_index, st.topk_weights,
                       st.spec.num_tokens, cfg.top_k)


# --------------------------------------------------------------------------
# golden path and input construction
# --------------------------------------------------------------------------

@torch.no_grad()
def golden_forward(spec: BenchSpec, weights: MoEWeights, x, forced_topk_ids=None):
    """Whole layer in fp32, no dtype round-trips. The value correctness tests
    compare against, so that a kernel's error is measured against the maths and
    not against another low-precision implementation."""
    cfg = spec.model
    xf = x.float()
    logits, ids, w = route(xf, weights.wg, cfg)
    if forced_topk_ids is not None:
        ids = forced_topk_ids.to(torch.int32)
        w = weights_for_forced_ids(logits, ids, cfg)
    offsets, perm = build_permutation(ids, cfg.num_experts)
    x_perm = xf[perm.long() // cfg.top_k]
    h_up = grouped_gemm_loop(x_perm, weights.w1.float(), offsets, 2 * cfg.intermediate_size)
    h_act = swiglu(h_up)
    y_perm = grouped_gemm_loop(h_act, weights.w2.float(), offsets, cfg.hidden_size)
    return combine(y_perm, perm, w, spec.num_tokens, cfg.top_k)


@torch.no_grad()
def make_inputs(spec: BenchSpec, device: str = "cpu", scale: float = 0.02):
    """Random weights and activations for one cell.

    `scale` keeps SwiGLU in a numerically sane range: a standard-normal init at
    H=7168 pushes pre-activation magnitudes far enough that fp16 overflows and
    bf16 loses most of its mantissa, which would make tolerance tuning
    meaningless rather than informative.
    """
    cfg = spec.model
    g = torch.Generator(device=device).manual_seed(spec.seed)
    dt = torch_dtype(spec.dtype)

    def rnd(shape, dtype=dt):
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32)
                * scale).to(dtype)

    weights = MoEWeights(
        w1=rnd(cfg.w1_shape),
        w2=rnd(cfg.w2_shape),
        wg=rnd((cfg.num_experts, cfg.hidden_size), torch.float32),
    )
    weights.validate(spec)
    x = rnd((spec.num_tokens, cfg.hidden_size))
    return x, weights
