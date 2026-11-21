"""Arguments the framework baselines must be handed, with no framework import.

The spans themselves import vLLM or SGLang at module scope, deliberately, so
`baselines.load_all()` skips them wherever the framework is absent rather than
letting a cell crash partway through a paid session. That makes the spans
untestable off the GPU box.

The argument construction is the part worth testing, and it does not need either
library. Every value here is one whose DEFAULT would produce a call that runs,
returns a tensor of the right shape, and computes a different layer than the
oracle is checking against. Those are the failures that survive a green test
suite, so they live in a module a laptop can import.
"""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..quant import FP8_DTYPES
from ..spec import BenchSpec


def vllm_call_kwargs(spec: BenchSpec) -> dict:
    """Non-tensor arguments for vLLM's `fused_experts`.

    `activation` is returned as its enum VALUE rather than the enum, so this
    module stays import-free; the span reconstructs `MoEActivation(value)`.
    """
    return {
        # SwiGLU: gated, hence not one of vLLM's explicit *_NO_MUL variants.
        "activation": "silu",
        # The reference combine() scales expert OUTPUTS by the combine weights.
        # Scaling inputs instead is a different computation wherever the expert
        # is nonlinear, which is everywhere.
        "apply_router_weight_on_input": False,
        # Defaults to -1 and is inferred. State it, so a shape disagreement is
        # an error rather than a silently different launch.
        "global_num_experts": spec.model.num_experts,
        # Single GPU: no expert-parallel remapping.
        "expert_map": None,
        # None resolves to FUSED_MOE_UNQUANTIZED_CONFIG inside fused_experts,
        # which is what bf16 wants.
        "quant_config": None,
    }


def sglang_runner_kwargs(spec: BenchSpec) -> dict:
    """MoeRunnerConfig fields, four of which fight their own defaults.

    routed_scaling_factor  defaults to None, which is correct, and is stated
        because it is load-bearing: gate_weights() has already applied
        DeepSeek-V3's 2.5x by the time any span runs, so a runner applying it
        again squares it to 6.25.
    gate_up_interleaved    defaults to TRUE, the opposite of this harness's
        layout. w1 is [E, 2F, H] as [gate | up] blocked, so interleaved pairs
        the wrong halves in every SwiGLU.
    inplace                defaults to TRUE and mutates hidden_states. `x` is
        built once per cell and reused by every timed iteration, so correctness
        would pass on the first call while the timing measured a decaying input.
    num_experts /          both default to None, and fused_experts computes
    num_local_experts      `filter_expert = num_experts is None or
        num_experts != num_local_experts`, so leaving them unset takes an
        expert-parallel masking path with no meaning on one GPU.
    """
    cfg = spec.model
    return {
        "num_experts": cfg.num_experts,
        "num_local_experts": cfg.num_experts,
        "hidden_size": cfg.hidden_size,
        "intermediate_size_per_partition": cfg.intermediate_size,
        "top_k": cfg.top_k,
        "activation": "silu",
        "is_gated": True,
        "apply_router_weight_on_input": False,
        "inplace": False,
        "no_combine": False,
        "routed_scaling_factor": None,
        "gate_up_interleaved": False,
    }


def vllm_quant_spec(spec: BenchSpec) -> dict | None:
    """What kind of quant config this cell needs, with no vLLM import.

    Returns None for float dtypes, which is what `fused_experts` wants: a
    `quant_config` of None resolves internally to FUSED_MOE_UNQUANTIZED_CONFIG.

    For fp8 the span builds `fp8_w8a8_moe_quant_config(w1_scale=..., w2_scale=...,
    **rest)`. The two flags below are stated rather than defaulted because both
    describe the quantisation this harness actually performed:

    per_act_token_quant  False. Activations are NOT quantised here, only weights.
        True would have vLLM expect a per-token activation scale that does not
        exist, and the shapes would be wrong rather than merely different.
    block_shape          None. Scales are per expert, one number for the whole
        tensor, not per [128, 128] block. A block shape sends the kernel down a
        blocked path expecting a 2-D scale grid.

    The scales themselves are tensors and so live on MoEWeights, not here; this
    module stays import-free so it can be tested on a laptop.
    """
    if spec.dtype not in FP8_DTYPES:
        return None
    return {
        "kind": "fp8_w8a8",
        "per_act_token_quant": False,
        "per_out_ch_quant": False,
        "block_shape": None,
    }


#: SGLang's fp8 support, probed on 0.5.18 / H200 / 2026-08-28. e4m3 only: its
#: w8a8 path is built on that dtype, and e5m2 weights under an e4m3 flag would
#: run and compute a different layer.
SGLANG_DTYPES: tuple[str, ...] = ("bf16", "fp8_e4m3")


def sglang_quant_kwargs(spec: BenchSpec) -> dict:
    """Quantisation keyword arguments for SGLang's `fused_experts`.

    Empty for a float dtype, which is what every published SGLang row used.

    Simpler than vLLM's equivalent: `fused_experts` takes flat keywords rather
    than a config object, and `use_fp8_w8a8` is a plain bool. The scales
    themselves are attached by the span, since only it holds the weights.

    Three fields are stated rather than defaulted, each because the default
    describes a layout this harness did not produce:

    per_channel_quant  the harness quantises one scale per EXPERT. True would
        send the kernel looking for a scale per output channel.
    block_shape        block-wise scaling is a third layout again.
    a1_scale/a2_scale  left None so SGLang quantises activations itself, which
        is what vLLM does internally. Supplying them would charge the two
        implementations differently for identical work.
    """
    if spec.dtype not in ("fp8_e4m3", "fp8_e5m2"):
        return {}
    if spec.dtype != "fp8_e4m3":
        raise ValueError(
            f"SGLang's w8a8 path is e4m3; {spec.dtype!r} would run under an "
            "e4m3 flag and compute a different layer")
    return {
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_channel_quant": False,
        "block_shape": None,
        "a1_scale": None,
        "a2_scale": None,
    }


# --------------------------------------------------------------------------
# which tile actually ran
# --------------------------------------------------------------------------
#
# The published CSVs recorded no fact about the tile the kernel ran. The only
# tile columns were load_tile_eff_bm64 / load_tile_eff_bm128, HYPOTHETICAL
# efficiencies computed from the routing histogram at ASSUMED block sizes, and a
# wrong BLOCK_SIZE_M sat unchallenged in an analysis for days because nothing in
# a row could contradict it.
#
# So the tile is observed from vLLM itself rather than predicted. It lives here,
# next to the other argument construction, because the span imports vLLM at
# module scope and is therefore untestable on a laptop, while everything below
# except the two probes is pure and testable off the GPU box.

#: Module namespaces that may hold vLLM's config hooks, most specific first.
#: Probed rather than assumed, exactly as scripts/tile_sweep.find_override
#: probes for override_config: the import path has moved between versions, and a
#: wrong guess records nothing while looking like it worked.
VLLM_CONFIG_MODULES: tuple[str, ...] = (
    "vllm.model_executor.layers.fused_moe.fused_moe",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.config",
)

#: vLLM's config-dict key -> the schema column that records it.
CONFIG_KEY_TO_COLUMN: dict[str, str] = {
    "BLOCK_SIZE_M": "tile_block_m",
    "BLOCK_SIZE_N": "tile_block_n",
    "BLOCK_SIZE_K": "tile_block_k",
    "GROUP_SIZE_M": "tile_group_m",
    "num_warps": "tile_num_warps",
    "num_stages": "tile_num_stages",
}


def nearest_config_key(keys: Iterable[int], m: int) -> int:
    """Which tuned-file entry vLLM's lookup selects for a batch of `m` rows.

    NEAREST, not floor. vLLM resolves the entry with
    `min(configs.keys(), key=lambda x: abs(x - M))`, so M=787 selects the 1024
    key and NOT the 512 one. Reading that lookup as a floor -- the natural
    assumption, and the one a bucketed lookup would justify -- attributes a tile
    to a row that never ran it, in the direction that matters: the 1024 entry is
    tuned for a shape eight times larger than a decode batch.

    Ties break toward the SMALLER key. python's min keeps the first minimum, and
    a tuned JSON is a dict built by json.load in file order, which is ascending
    in every shipped file. Stated because it is an assumption about those files
    rather than a rule vLLM enforces; a tie is a knife-edge case either way.
    """
    ordered = sorted(int(k) for k in keys)
    if not ordered:
        raise ValueError("no tuned config keys to choose from")
    return min(ordered, key=lambda k: (abs(k - m), k))


@dataclass
class TileCall:
    """One observed config lookup.

    `tuned_keys` is the discriminator the whole record turns on:

        None      the tuned-file lookup returned nothing, so this call took the
                  hardcoded fallback ladder (M<=32 -> 16, M<=96 -> 32,
                  M<=512 -> 64, else 128).
        [ints]    a tuned file exists for this (num_experts, N, dtype, device)
                  and these are its M keys, so the nearest-key rule applies.

    `lookup_observed` is separate from `tuned_keys is None` on purpose: "the
    lookup said there is no tuned file" and "nothing watched the lookup" are
    different facts, and only the first justifies writing vllm_default.
    """

    m: int | None = None
    config: dict | None = None
    tuned_keys: list[int] | None = None
    lookup_observed: bool = False


@dataclass
class TileCapture:
    """Everything one recorded fused_experts call revealed about its tile."""

    calls: list = field(default_factory=list)
    #: Set by the lookup wrapper, consumed by the config wrapper that encloses
    #: it, so a lookup is paired with the call it happened inside rather than by
    #: matching list positions -- which would misalign the moment a call skips
    #: the lookup, which is exactly what an active override makes it do.
    pending: list = field(default_factory=list)


def tile_meta_from_capture(capture: TileCapture,
                           override_active: bool = False) -> dict:
    """The schema columns one capture justifies, and no more.

    Takes the FIRST recorded call. vLLM computes the config once per
    fused_experts call and only re-derives it per chunk above
    VLLM_FUSED_MOE_CHUNK_SIZE, so below that there is exactly one; above it, the
    first chunk is the full-size one and the tail chunk that follows is both
    smaller and shorter-running, so the first call is the one that describes the
    time.

    Every branch that cannot name the source writes "unrecorded" rather than the
    likelier of two answers. The tile ints are still written when they were
    genuinely read back, because they were: what is unknown in that branch is
    where the config came from, not what it was.
    """
    if not capture.calls:
        return {"tile_config_source": "unrecorded"}
    call = capture.calls[0]
    meta: dict = {}
    for key, column in CONFIG_KEY_TO_COLUMN.items():
        value = (call.config or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            meta[column] = value

    if override_active:
        # tile_sweep.py forces a tile through override_config, and a row
        # produced under one is not evidence about what vLLM would have chosen.
        meta["tile_config_source"] = "vllm_override"
        return meta
    if not call.lookup_observed:
        meta["tile_config_source"] = "unrecorded"
        return meta
    if call.tuned_keys is None:
        meta["tile_config_source"] = "vllm_default"
        return meta
    meta["tile_config_source"] = "vllm_tuned"
    if call.m is not None:
        meta["tile_config_key"] = nearest_config_key(call.tuned_keys, call.m)
    return meta


def called_with_m(fn, args: tuple, kwargs: dict) -> int | None:
    """The M a config lookup was made for.

    Bound through inspect.signature rather than read as args[4]: the parameter
    has moved position between vLLM versions, and a positional index that is
    silently wrong attributes a config to the wrong batch size, which is the
    exact class of error the tile columns exist to catch. Returns None rather
    than guessing when the signature does not bind.
    """
    import inspect

    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    for name in ("M", "m", "num_tokens"):
        value = bound.arguments.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _config_key_list(configs) -> list[int] | None:
    """Tuned-file M keys, or None when the lookup found no tuned file.

    Keys come back as ints from vLLM's own loader, but a str key would compare
    against M as a string and silently pick the lexicographic nearest, so they
    are coerced here and an uncoercible key set is treated as no tuned file at
    all rather than as a half-read one.
    """
    if not configs:
        return None
    try:
        return [int(k) for k in configs]
    except (TypeError, ValueError):
        return None


def bindings_of(attr: str, module_names: Iterable[str] = VLLM_CONFIG_MODULES
                ) -> list:
    """Every importable module namespace that holds `attr`.

    ALL of them, not the first match. `from ... import try_get_optimal_moe_config`
    copies the function OBJECT into the importing module's globals, so rebinding
    only the module that defines it leaves the name the caller actually looks up
    pointing at the original -- the recorder would observe nothing while
    appearing to be installed.

    Returns [] where vLLM is absent, which is how this stays importable on a
    laptop: the caller degrades to an unrecorded row instead of failing.
    """
    import importlib

    found = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(module, attr, None) is not None:
            found.append(module)
    return found


def vllm_forced_config(module_names: Iterable[str] = VLLM_CONFIG_MODULES
                       ) -> dict | None:
    """The config an override_config context is forcing right now, or None.

    try_get_optimal_moe_config consults get_config() first and a truthy value
    bypasses both the tuned file and the default ladder, so a row measured under
    one says nothing about what vLLM would have chosen.

    Returns the DICT rather than a bool because "an override is active" is not
    the only question asked of it: `forcing_tile_config` has to know whether the
    active override is the one it just installed, and a bool cannot tell a
    forced 128 from someone else's forced 64. First truthy answer wins; every
    binding reads the same module-level global, so they cannot disagree.
    """
    for module in bindings_of("get_config", module_names):
        try:
            config = module.get_config()
        except Exception:  # noqa: BLE001
            # Deliberately broad: a probe that cannot answer must not take a
            # metered cell down with it. An unanswered probe reads as "no
            # override", which is the state every unpinned sweep runs in.
            continue
        if config:
            return dict(config)
    return None


def vllm_override_active(module_names: Iterable[str] = VLLM_CONFIG_MODULES
                         ) -> bool:
    """Is an override_config context in force right now?

    Asked directly rather than inferred from the config's shape, because a
    forced tile can be identical to the one that would have been picked anyway.
    """
    return vllm_forced_config(module_names) is not None


class ForceTileNotHonoured(RuntimeError):
    """A tile was forced onto a path that did not take it.

    Raised rather than measuring the cell anyway. The whole value of pinning is
    that a row can be read as "the kernel ran THIS tile"; a cell that was asked
    to pin and did not, measured anyway, produces a row that claims exactly that
    and is wrong -- which is worse than the unpinned sweep it replaced, because
    the unpinned sweep at least records vLLM's own choice honestly.

    Lives here, beside the probes it is raised by, so `moe.bench.force_tile`
    (which owns the policy) depends on this module and not the other way round.
    """


@contextmanager
def forcing_tile_config(config: dict,
                        module_names: Iterable[str] = VLLM_CONFIG_MODULES):
    """Run everything inside this context under EXACTLY `config`.

    vLLM's own hook, probed rather than imported from a fixed path, exactly as
    `scripts/tile_sweep.find_override` and `block_m_crossing_sweep.find_override`
    probe it: the import path has moved between versions and a wrong guess
    forces nothing while looking installed.

    THEN VERIFIED, which is the part those scripts do differently and the reason
    this one exists. `block_m_crossing_sweep` proves the override took by
    counting Triton artefacts per setting (its gate 0); a sweep cannot, because
    it runs one setting. So the check here is direct: after entering, ask
    get_config() what vLLM would now hand its kernel and refuse if it is not
    what we asked for. Entering a context and assuming it took is how
    MOE_FORCE_TILE came to be set for a whole session with nothing reading it.

    Compared key by key rather than as whole dicts: a future vLLM that copies
    the override and adds a key of its own would still be honouring every value
    that was forced, and refusing that would be a false alarm. A value that came
    back changed or missing is not.

    Yields the dotted name of the hook that took, so a caller can print WHICH
    binding was used rather than asserting one.
    """
    holders = bindings_of("override_config", module_names)
    if not holders:
        raise ForceTileNotHonoured(
            f"vLLM exposes no override_config in any of {list(module_names)}, "
            f"so no tile can be forced in this process. Check the installed "
            f"version: try_get_optimal_moe_config reads it through get_config(), "
            f"so the hook exists under some name.")
    module = holders[0]
    wanted = dict(config)
    with module.override_config(wanted):
        seen = vllm_forced_config(module_names)
        mismatch = (seen is None or
                    any(seen.get(k) != v for k, v in wanted.items()))
        if mismatch:
            raise ForceTileNotHonoured(
                f"{module.__name__}.override_config was entered with {wanted} "
                f"but get_config() reports {seen!r}. The kernel would run a "
                f"tile nobody asked for while every row claimed the forced one.")
        yield f"{module.__name__}.override_config"


@contextmanager
def recording_tile_config(capture: TileCapture,
                          module_names: Iterable[str] = VLLM_CONFIG_MODULES):
    """Watch vLLM choose a tile, for the duration of ONE call.

    PATCH-FREE, and that is what makes it safe to use mid-sweep:
    fused_experts_impl rebuilds its `functools.partial(try_get_optimal_moe_config,
    ...)` from the module global on every call, so rebinding the global is
    enough and nothing inside vLLM is edited. The originals are restored in a
    finally, so a raising call cannot leave the recorder installed and silently
    slow or alter every later cell.

    Wraps TWO names because one of them alone cannot answer the question.
    try_get_optimal_moe_config gives the config and the M it was chosen for;
    only get_moe_configs distinguishes a TUNED file from the hardcoded fallback
    ladder, and that distinction is the finding: nothing ships for
    NVIDIA_A100-SXM4-80GB at any of this study's four shapes, so those cells all
    took the ladder while the H200's mixtral cells did not, and the two cards
    therefore ran different tiles at the measured crossings.

    Yields the capture. Records nothing at all where vLLM is absent.

    KNOWN DEGRADATION, named so it is recognisable in a CSV. If a vLLM release
    ever memoises try_get_optimal_moe_config itself, the observation call below
    hits that cache, get_moe_configs is never re-entered, and the row comes back
    with real tile ints and a source of "unrecorded". That is the intended
    failure: an unlabelled tile, never a tile labelled tuned when it was not.
    """
    import copy
    import functools

    config_modules = bindings_of("try_get_optimal_moe_config", module_names)
    lookup_modules = bindings_of("get_moe_configs", module_names)
    if not config_modules:
        yield capture
        return

    # Per module, so a restore puts back exactly what that namespace held. In
    # practice `from ... import f` copies one function object into every
    # namespace and they are all the same, but restoring the first module's
    # binding into all of them would quietly repair or break that assumption
    # instead of preserving it.
    config_originals = [(m, m.try_get_optimal_moe_config) for m in config_modules]
    lookup_originals = [(m, m.get_moe_configs) for m in lookup_modules]
    original_config = config_originals[0][1]
    original_lookup = lookup_originals[0][1] if lookup_originals else None

    @functools.wraps(original_config)
    def record_config(*args, **kwargs):
        del capture.pending[:]
        config = original_config(*args, **kwargs)
        seen = capture.pending[-1] if capture.pending else None
        capture.calls.append(TileCall(
            m=called_with_m(original_config, args, kwargs),
            # Deepcopied: fused_experts_impl mutates the returned dict
            # downstream (it overwrites BLOCK_SIZE_M for the tail path), so the
            # object handed back is not the one that was chosen.
            config=copy.deepcopy(config) if isinstance(config, dict) else None,
            tuned_keys=seen[0] if seen else None,
            lookup_observed=seen is not None,
        ))
        return config

    def record_lookup(*args, **kwargs):
        configs = original_lookup(*args, **kwargs)
        # A 1-tuple, so "the lookup ran and found no tuned file" (an appended
        # (None,)) stays distinguishable from "the lookup never ran" (nothing
        # appended). A bare None in the list would collapse the two.
        capture.pending.append((_config_key_list(configs),))
        return configs

    for module in config_modules:
        module.try_get_optimal_moe_config = record_config
    if original_lookup is not None:
        record_lookup = functools.wraps(original_lookup)(record_lookup)
        for module in lookup_modules:
            module.get_moe_configs = record_lookup
    try:
        yield capture
    finally:
        # In a finally, so a raising call cannot leave the recorder installed:
        # it would then wrap every later cell of a metered sweep, appending to a
        # capture nobody reads.
        for module, original in config_originals:
            module.try_get_optimal_moe_config = original
        for module, original in lookup_originals:
            module.get_moe_configs = original
