"""The layer's expert weight set, in bytes, and the per-M-tile slope measured
in units of one complete stream of it.

WHY THIS MODULE EXISTS. Every unphysical number this study has argued about
came out of one estimator: `block_m_crossing_sweep.LadderFit.alpha = B/(A+B)`,
the per-tile slope divided by a LEVEL that is an extrapolation of the ladder
back to n = 0 across a lever arm of up to 44 treads on a ladder that is not
exactly affine. On the 2026-09-10 H200 session that estimator produced ten
values above 1.0 across four arms, three negative fitted intercepts, and
bn_decomposition's alpha_a = -0.8143, whose entire SIGN lies inside the
reference fixed cost's own jackknife error. Its high sibling
`alpha_upper = B/(A + B - D)` exceeds 1 exactly when `D > A`: arithmetic
about an extrapolation, not physics about a miss fraction.

WHAT REPLACES THE DENOMINATOR. Not the level, but a measured quantity that
owes nothing to the fit:

    w = (ms per extra M-tile) / (ms to stream the layer's expert weight set
                                 once at a bandwidth the caller measured)

No fitted level, no intercept, no delta, no D. The numerator is the ladder's
own slope `B`, which is the one number in the fit that no extrapolation
touches. The denominator is a byte count from `moe.spec` divided by a rate the
CALLER supplies, so the number always names the rate it was divided by. On
mixtral bf16 the expert weight set is 2.8186 GB and one stream is 0.6443 ms at
this H200's calibrated triad rate of 4374.3 GB/s. Measured that way over 23
ladders of the session, w runs 0.68 to 1.37 with a per-repeat sd of 0.002 to
0.005 over 17 repeats.

THIS ADDS A STATISTIC, IT DOES NOT REPLACE ONE. The 100,144 published rows
were all scored on B/(A+B) and stay readable exactly as they were. `w` is
printed and persisted BESIDE alpha, never instead of it, the way the clock
rule put both the fixed-roof and the own-clock fraction on every row.

WHAT w IS NOT. It is not alpha_b, the weight miss fraction, and reading it as
one is the mistake this module is written to make hard. Under the three-term
model the slope is `alpha_b + phi` in weight-read units with `phi >= 0`, so at
the rate the weights actually stream at, w is an UPPER BOUND on alpha_b. And
w scales exactly linearly in the assumed rate, and in the direction a
denominator implies: a FASTER assumed rate makes one stream take less time, so
the same slope is MORE streams. Double `bandwidth_gbps` and w doubles. That 1:1
confound is the reason the rate is a required argument with no default,
since a w quoted without its rate is not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from moe.spec import DTYPE_BYTES, MODEL_CONFIGS, MoEConfig


class WeightSetRefused(ValueError):
    """A model, dtype or rate whose weight-stream time this module will not guess."""


def _resolve(model: str | MoEConfig) -> MoEConfig:
    """The named config, or the one handed in. Refuses a name it does not know
    rather than falling back to a default geometry."""
    if isinstance(model, MoEConfig):
        cfg = model
    else:
        try:
            cfg = MODEL_CONFIGS[model]
        except KeyError:
            raise WeightSetRefused(
                f"unknown model {model!r}; known: {sorted(MODEL_CONFIGS)}"
            ) from None
    if not cfg.verified:
        # `verified` flips to True per model once scripts/verify_model_configs.py
        # has checked the geometry against the upstream config.json. An
        # unverified geometry multiplies out to a byte count that looks exactly
        # as authoritative as a checked one, and the whole point of w is that
        # its denominator is not a guess.
        raise WeightSetRefused(
            f"{cfg.name}: geometry is not `verified` against the upstream "
            "config.json, so its weight set would be a guess. Run "
            "scripts/verify_model_configs.py before dividing a slope by it")
    return cfg


def _dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise WeightSetRefused(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
        ) from None


def routed_expert_weight_bytes(model: str | MoEConfig, dtype: str) -> int:
    """Bytes of ROUTED expert weights read in one full pass of one MoE layer.

        E * (2F*H + H*F) * bytes(dtype)  =  E * 3FH * bytes(dtype)

    the fused gate+up slab `w1[E, 2F, H]` plus the down slab `w2[E, H, F]`,
    which is exactly the operand set vLLM's `fused_moe` streams. `F` is the
    per-shard intermediate width, so a TP entry returns the weight set ONE
    DEVICE streams, which is the set the kernel on that device re-reads.

    CROSS-CHECK. mixtral-8x7b bf16 = 8 * 3 * 14336 * 4096 * 2 = 2 818 572 288
    bytes = 2.8186 GB, the figure `analysis/synth/s8_common_currency.py`
    divides every ladder slope by and the one every w in the 2026-09-10
    synthesis is quoted against. It agrees term for term with
    `bn_decomposition.weight_elements(cfg) * cfg.num_experts * bytes`, which is
    that script's own route to the same number, and with
    `moe.spec.MoEConfig.weight_bytes`, which is where the multiplication
    actually lives. This function is the documented name for it, not a second
    copy of the arithmetic: two copies of one byte count is how the two halves
    of a study end up dividing by different denominators.

    ROUTED ONLY, AND SAID SO. A shared expert is a dense FFN outside the
    grouped GEMM, its width is not `intermediate_size` (qwen2-57b-a14b's is
    20480 against a routed 2560) and `MoEConfig` does not carry it at all. Its
    bytes are therefore neither counted here nor silently folded in;
    `layer_weight_bytes` is the function that refuses when the caller asks for
    the whole layer on such a model.
    """
    cfg = _resolve(model)
    b = _dtype_bytes(dtype)
    w1 = cfg.num_experts * 2 * cfg.intermediate_size * cfg.hidden_size
    w2 = cfg.num_experts * cfg.hidden_size * cfg.intermediate_size
    return (w1 + w2) * b


def layer_weight_bytes(model: str | MoEConfig, dtype: str) -> int:
    """Every expert weight byte one MoE layer reads in a full pass, or a refusal.

    Equal to `routed_expert_weight_bytes` on a model with no shared expert, and
    a REFUSAL on one that has any, because `MoEConfig` carries
    `shared_experts` as a COUNT and not a width: qwen2-57b-a14b's shared expert
    is 20480 wide where its routed experts are 2560, and deepseek-v3's is
    `n_shared_experts * moe_intermediate_size`, neither of which is recoverable
    from the fields this repository stores. Returning the routed set under this
    name would understate qwen2's layer by a factor this module cannot compute,
    silently, which is the shape of defect the whole round is about.

    A caller measuring the fused grouped GEMM wants `routed_expert_weight_bytes`
    and should call it by that name: the shared expert runs in a different
    kernel and its bytes are not on the ladder being fitted.
    """
    cfg = _resolve(model)
    if cfg.shared_experts:
        raise WeightSetRefused(
            f"{cfg.name}: {cfg.shared_experts} shared expert(s), whose FFN "
            "width is not on MoEConfig (qwen2-57b-a14b's is 20480 against a "
            "routed 2560), so the layer's whole weight set cannot be resolved "
            "from this config. Call routed_expert_weight_bytes for the set the "
            "fused grouped GEMM streams, and name it as routed-only")
    return routed_expert_weight_bytes(cfg, dtype)


def _check_bandwidth(bandwidth_gbps: float | None) -> float:
    """A rate that came from a measurement, or a refusal. NEVER a default.

    The whole content of `w` is that it names the rate it was divided by, and
    that rate is a 1:1 confound: `w` scales exactly linearly in it. A module
    that supplied its own would be quoting the card's datasheet as if it were
    the memory branch's achieved bandwidth, which is the confound statement (5)
    of the 2026-09-10 synthesis is about, dressed as a convenience.
    """
    if bandwidth_gbps is None:
        raise WeightSetRefused(
            "bandwidth_gbps is None. A weight-stream count is a slope divided "
            "by a MEASURED rate and there is no default rate to fall back on: "
            "w scales 1:1 in it, so a w without a named rate is not a "
            "measurement. Pass the card's calibrated figure "
            "(hardware yaml `memory.bandwidth_tb_s` * 1000, or a "
            "`bandwidth_patterns` entry) and say which one")
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0.0:
        raise WeightSetRefused(
            f"bandwidth_gbps={bandwidth_gbps} is not a rate: it must be finite "
            "and positive. Zero would make one stream take forever and every w "
            "zero; a negative or NaN rate is a broken calibration read, not a "
            "slow card")
    return float(bandwidth_gbps)


def weight_stream_ms(model: str | MoEConfig, dtype: str,
                     bandwidth_gbps: float | None) -> float:
    """Milliseconds to stream the routed expert weight set once at this rate.

    `routed_expert_weight_bytes / (bandwidth_gbps * 1e9) * 1e3`. On mixtral
    bf16 at this H200's calibrated triad rate of 4374.2997 GB/s it is
    0.644348 ms; at the same card's measured `read_stream` pattern
    (4612.2534 GB/s) it is 0.611105 ms, and the ratio of the two is exactly
    the ratio a w quoted at one rate differs from the same w at the other.
    """
    bw = _check_bandwidth(bandwidth_gbps)
    return 1e3 * routed_expert_weight_bytes(model, dtype) / (bw * 1e9)


@dataclass(frozen=True)
class WeightStreamSlope:
    """A ladder slope in weight-stream units, carrying the rate it was divided by.

    Kept as an object rather than a bare float for one reason: `streams` is
    meaningless without `bandwidth_gbps`, and a float loses it at the first
    assignment. `render()` is the one-line form the report and every caller
    print, so the number and its rate cannot drift apart in the prose the way
    alpha and its estimator did.
    """

    #: `w`: milliseconds per extra M-tile divided by milliseconds per stream.
    streams: float
    #: The rate the denominator was computed at. Named, never defaulted.
    bandwidth_gbps: float
    #: Where that rate came from, in the caller's own words ("triad, measured
    #: 2026-09-10"). Empty is legal and prints as NOT STATED, which is the
    #: honest rendering of a rate whose provenance the caller did not give.
    bandwidth_source: str
    #: Milliseconds for one full stream of the weight set at that rate.
    stream_ms: float
    #: The weight set itself, so a reader can check the division.
    weight_bytes: int
    model: str
    dtype: str

    @property
    def descending(self) -> bool:
        """The slope was NEGATIVE: this ladder got faster with another M-tile.

        A state to read, not a value to hide. `streams` is then negative and is
        not a fraction of anything, and the cause is upstream in the branch
        membership, a memory branch fitted through treads that were not all
        on it, and not in this division. `render` says so on the line.
        """
        return self.streams < 0.0

    def render(self) -> str:
        note = ("" if not self.descending else
                "; NEGATIVE, this ladder's fitted slope falls with another "
                "M-tile, so it is not a fraction of a stream and its branch "
                "membership is what to look at")
        return (f"{self.streams:.4f} weight-streams/M-tile at "
                f"{self.bandwidth_gbps:.1f} GB/s "
                f"({self.bandwidth_source or 'rate NOT STATED by the caller'}; "
                f"{self.weight_bytes / 1e9:.4f} GB in {self.stream_ms:.4f} ms, "
                f"{self.model} {self.dtype}){note}")


def weight_streams_per_tile(slope_ms: float, model: str | MoEConfig, dtype: str,
                            bandwidth_gbps: float | None, *,
                            bandwidth_source: str = "") -> WeightStreamSlope:
    """`w`: what one more M-tile costs, in complete streams of the weight set.

        w = slope_ms / weight_stream_ms(model, dtype, bandwidth_gbps)

    `slope_ms` is a ladder's fitted per-tile slope `B` in milliseconds, the
    only number in a `LadderFit` that no extrapolation to n = 0 touches.

    A NEGATIVE SLOPE IS LABELLED, NOT REFUSED, and the reason is the same one
    that keeps a `D > A` ladder in the table: a ladder whose fitted slope falls
    with another M-tile has said something about its branch membership, and
    dropping it deletes the evidence. `WeightStreamSlope.descending` is True
    there, `render` says on the line that the number is not a fraction of a
    stream, and the sweep prints it as the negative it is. What IS refused is a
    non-finite slope, which is not a measurement at all: a zero-variance or
    single-tread branch produces a NaN, and dividing it by a stream time gives
    a w no ladder measured. Zero is legal and means what it says, a per-tile
    cost of nothing.

    THE ACCEPTANCE NUMBERS, reproduced from the committed cells of
    results/published/2026-09-10-nvidia_h200-gaps-session by
    tests/test_ai_model.py: at the triad rate, bn_decomposition's G=16 ladders
    give w = 1.254 / 1.368 at BLOCK_N=32 (BLOCK_M 32 / 64), 0.863 / 0.897 at
    BLOCK_N=64 and 0.683 / 0.731 at BLOCK_N=128, and tile_cap's BLOCK_M=16,
    G=1 ladder gives 1.0514. Those are the numbers in section 1 of the
    2026-09-10 synthesis, and they are what this function is for.
    """
    if not math.isfinite(slope_ms):
        raise WeightSetRefused(
            f"slope_ms={slope_ms} is not finite: a degenerate ladder fit "
            "(a zero-variance or single-tread branch) produces one, and "
            "dividing it by a stream time yields a w that no ladder measured")
    cfg = _resolve(model)
    ms = weight_stream_ms(cfg, dtype, bandwidth_gbps)
    return WeightStreamSlope(
        streams=slope_ms / ms,
        bandwidth_gbps=float(bandwidth_gbps),
        bandwidth_source=bandwidth_source,
        stream_ms=ms,
        weight_bytes=routed_expert_weight_bytes(cfg, dtype),
        model=cfg.name,
        dtype=dtype,
    )
