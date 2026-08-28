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

from ..quant import FP8_DTYPES, dequantize_per_expert, quantize_per_expert
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


def expert_counts(topk_ids, num_experts: int):
    """[T, k] expert ids -> [E] row counts. The one spelling of this histogram.

    Three copies existed and had already drifted on dtype and device handling
    (cast before vs after, .cpu() or not), which is exactly how two callers end
    up disagreeing about how many experts were active.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    return torch.bincount(flat, minlength=num_experts)


def build_permutation(topk_ids, num_experts: int):
    """topk_ids [T,k] -> (expert_offsets [E+1] int32, perm_index [T*k] int32).

    perm_index[r] is the flat (token*k + slot) index whose row lands at permuted
    position r. A stable sort keeps token order within an expert, which makes
    results reproducible and makes tiling bugs easier to read.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    perm_index = torch.argsort(flat, stable=True).to(torch.int32)
    counts = expert_counts(flat, num_experts)
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
    # fp8 is here because the reference fills every stage a baseline does not
    # cover, so without it an fp8 sweep plans zero cells: vLLM covers five of
    # six and the sixth has nowhere to run. The spans dequantise per expert,
    # which is the same value the kernel reconstructs, so the reference stays
    # the oracle rather than becoming a second low-precision implementation.
    dtypes = ("fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2")


@register
class RefRouter(_Ref):
    name = "ref_router"
    covers = ("router",)
    # Nothing downstream reads the raw logits, so liveness would not charge for
    # them. This implementation stores them anyway, and the bytes model should
    # say so rather than flatter it.
    materialises = ("router_logits",)

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
            st.x_perm, _weight(st.weights, "w1"), st.expert_offsets,
            2 * cfg.intermediate_size
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
            st.h_act, _weight(st.weights, "w2"), st.expert_offsets,
            cfg.hidden_size
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

def _weight(weights: MoEWeights, which: str):
    """`w1`/`w2`, dequantised if the cell is fp8.

    Reconstructing `q * scale` is what the kernel does too, so the reference
    computes the same function and remains an oracle. Handing it raw fp8 would
    make it a second low-precision implementation and the correctness gate would
    compare two approximations to each other.
    """
    w = getattr(weights, which)
    if not weights.quantised:
        return w
    return dequantize_per_expert(w, getattr(weights, f"{which}_scale"))


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
    # No .float() on the weights here: grouped_gemm_loop already converts
    # w[e] per expert, and only for experts that received rows. Pre-casting the
    # whole tensor converted it twice, and at DeepSeek-V3 geometry allocated a
    # 45 GB fp32 transient on a 141 GB card for every cell. Verified
    # bit-identical, and 72-99% faster across the token range.
    # fp8 weights are DEQUANTISED before the oracle sees them. If the reference
    # used the original fp32 draw while the kernel used `q * scale`, the
    # correctness gate would be measuring quantisation error, which is ~2.6% RMS,
    # a property of the format, and identical for every implementation. That is
    # not what the gate is for: it exists to catch a kernel computing the wrong
    # thing. Dequantising here makes both sides compute the same function and
    # differ only in arithmetic precision.
    #
    # grouped_gemm_loop converts per expert, so this is not a whole-tensor
    # pre-cast; it reconstructs the value the kernel will also reconstruct.
    w1, w2 = _weight(weights, "w1"), _weight(weights, "w2")
    h_up = grouped_gemm_loop(x_perm, w1, offsets, 2 * cfg.intermediate_size)
    h_act = swiglu(h_up)
    y_perm = grouped_gemm_loop(h_act, w2, offsets, cfg.hidden_size)
    return combine(y_perm, perm, w, spec.num_tokens, cfg.top_k)


#: One entry, deliberately. Expert weights are the largest allocation in a
#: cell (22.5 GB at DeepSeek-V3), and holding two models' worth would cost more
#: memory than the regeneration costs time. `sweep()` varies model, dtype and
#: seed slowest precisely so a single entry hits almost every cell.
_WEIGHT_CACHE: dict[tuple, tuple] = {}


def clear_weight_cache() -> None:
    _WEIGHT_CACHE.clear()


@torch.no_grad()
def make_inputs(spec: BenchSpec, device: str = "cpu", scale: float = 1.0,
                reuse_weights: bool = True):
    """Random weights and activations for one cell, initialised realistically.

    Activations entering an MoE layer come out of a normalisation, so they are
    close to unit variance. Weights use fan-in scaling, which is what real
    checkpoints look like. Together these put the layer output at order 0.1,
    the same place a real model's is.

    This matters for more than realism. An earlier version scaled everything to
    0.02, which drove outputs down to 1e-6 and made every absolute-tolerance
    comparison meaningless. Correctness is now judged by a scale-free relative
    metric, but keeping the numerics in a realistic range also keeps bf16 from
    losing mantissa to underflow and keeps fp16 away from overflow.

    Weights depend only on (model, dtype, seed, device, scale) and are drawn
    BEFORE `x`, so consecutive cells that differ only in token count or routing
    can reuse them. The generator state is cached alongside and restored, which
    makes `x` bit-identical to the uncached path rather than merely equivalent.
    Regenerating them costs 203 GB of traffic per DeepSeek-V3 cell.

    The reuse contract is read-only: an implementation that writes to
    `weights.w1` would leak into every later cell. Nothing in the harness does,
    and `reuse_weights=False` opts out.
    """
    cfg = spec.model
    dt = torch_dtype(spec.dtype)
    key = (cfg.name, spec.dtype, spec.seed, str(device), float(scale))

    cached = _WEIGHT_CACHE.get(key) if reuse_weights else None
    if cached is not None:
        weights, state = cached
        g = torch.Generator(device=device)
        g.set_state(state)
    else:
        g = torch.Generator(device=device).manual_seed(spec.seed)

    def rnd(shape, std, dtype=dt):
        # normal_ folds the scale into the RNG kernel. The previous
        # randn(...) * std kept three tensors live at once (the draw, the
        # scaled result, the cast), peaking at 75 GB while building
        # DeepSeek-V3's w1 alone. Verified bit-identical to the old form.
        drawn = (torch.empty(shape, device=device, dtype=torch.float32)
                 .normal_(0.0, std * scale, generator=g))
        if dtype is torch.float32 or drawn.ndim < 2:
            return drawn.to(dtype)
        return drawn.to(dtype)

    if cached is None:
        # fp8 is drawn in fp32 and quantised per expert, not cast. A bare cast
        # would leave 27.6% of fan-in scaled elements subnormal, and vLLM
        # requires the scales regardless: fp8_w8a8_moe_quant_config takes
        # w1_scale and w2_scale and reconstructs as `q * scale`.
        #
        # The GATE stays fp32 on purpose. Routing decides which experts run, so
        # quantising it would change the experiment rather than the arithmetic
        # under test, and the two dtypes would no longer be the same cells.
        if spec.dtype in FP8_DTYPES:
            w1f = rnd(cfg.w1_shape, cfg.hidden_size ** -0.5, torch.float32)
            w2f = rnd(cfg.w2_shape, cfg.intermediate_size ** -0.5, torch.float32)
            w1q, w1s = quantize_per_expert(w1f, spec.dtype)
            w2q, w2s = quantize_per_expert(w2f, spec.dtype)
            del w1f, w2f
            weights = MoEWeights(
                w1=w1q, w2=w2q,
                wg=rnd((cfg.num_experts, cfg.hidden_size),
                       cfg.hidden_size ** -0.5, torch.float32),
                w1_scale=w1s, w2_scale=w2s,
            )
        else:
            weights = MoEWeights(
                w1=rnd(cfg.w1_shape, cfg.hidden_size ** -0.5),
                w2=rnd(cfg.w2_shape, cfg.intermediate_size ** -0.5),
                wg=rnd((cfg.num_experts, cfg.hidden_size), cfg.hidden_size ** -0.5,
                       torch.float32),
            )
        weights.validate(spec)
        if reuse_weights:
            # Evict first: never hold two models' weights at once.
            _WEIGHT_CACHE.clear()
            _WEIGHT_CACHE[key] = (weights, g.get_state())

    x = rnd((spec.num_tokens, cfg.hidden_size), 1.0)
    return x, weights
