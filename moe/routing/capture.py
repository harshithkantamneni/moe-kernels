"""Locating MoE gates in an unfamiliar model, and recording what they route.

This is library code, not script code. `find_gate_modules` encodes a subtle
model-layout rule with a near-miss that has already bitten once, `routed_ids`
encodes a second one about what a gate module actually RETURNS, and the
recorder must apply the same per-model gating function the reference oracle
does. Keeping all three here means one gating rule, tested in one place.

`scripts/capture_traces.py` is the CLI over it.
"""
from __future__ import annotations

import torch

from ..reference.torch_ref import expert_counts, gate_scores
from ..spec import MoEConfig


def find_gate_modules(model, num_experts: int) -> list[tuple[str, object]]:
    """Every routed-expert gate in a model, without branching per architecture.

    Mixtral puts it at `block_sparse_moe.gate`, Qwen2 and DeepSeek at
    `mlp.gate`. So: any module whose FINAL path component is exactly "gate" and
    whose output width equals the routed-expert count.

    The last-component match is the load-bearing part. A suffix match on "gate"
    also catches `attn_gate` and Qwen2's `shared_expert_gate`, and hooking
    those records counts that are not routed-expert counts at all.

    The `weight.shape[0]` fallback is what catches DeepSeek: its gate is not an
    `nn.Linear` and has no `out_features`, only a bare `nn.Parameter` of shape
    [num_experts, hidden]. That module is correctly found here and returns
    something entirely different from logits, which is `routed_ids`' problem.
    """
    found = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "gate":
            continue
        out = getattr(module, "out_features", None)
        if out is None:
            weight = getattr(module, "weight", None)
            out = None if weight is None else weight.shape[0]
        if out == num_experts:
            found.append((name, module))
    return found


def routed_ids(output, cfg: MoEConfig, where: str = "gate",
               check_values: bool = False) -> torch.Tensor:
    """The [N, top_k] expert ids a gate just chose, whatever shape it returned.

    Two gate shapes exist in the wild and they are NOT interchangeable.

    A plain `nn.Linear` router (mixtral, qwen2) returns float logits
    [N, num_experts]; the scoring function and the top-k are still ours to
    apply, and `gate_scores` owns which scoring function that is. DeepSeek's
    `MoEGate.forward` returns the 3-tuple `(topk_idx, topk_weight, aux_loss)`
    whose element 0 is ALREADY the chosen ids, int64 [N, top_k], and whose
    aux_loss is None outside training.

    The hook used to take element 0 and treat it as logits either way. On
    DeepSeek-V2-Lite that softmaxes expert INDEX VALUES. The old
    `.reshape(-1, num_experts)` raises at decode, where a batch of 4 supplies
    4*6 = 24 ids against 64 experts, and SILENTLY SUCCEEDS at prefill whenever
    batch*seq_len*top_k happens to divide by num_experts -- writing a trace
    that passes every check in `write_trace` and describes nothing. This
    function is why a capture cannot do that: it dispatches on what actually
    arrived and refuses anything that is neither shape.

    Taking the model's own ids is also strictly more faithful than recomputing
    them. DeepSeek's `topk_method="group_limited_greedy"` zeroes whole expert
    groups before the top-k, so a plain top-k over the logits picks different
    experts than the model did.

    `check_values` costs a device sync, so callers do it once per layer rather
    than once per decode step.
    """
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{where}: gate returned {type(tensor).__name__}, not a tensor. "
            "This module is not a router and must not be hooked.")
    tensor = tensor.detach()
    if tensor.ndim != 2:
        raise ValueError(
            f"{where}: expected a 2-D gate output [tokens, ...], got shape "
            f"{tuple(tensor.shape)}. A gate scores a flattened token stream; "
            "anything else means the wrong module is hooked.")

    if tensor.is_floating_point():
        if tensor.shape[1] != cfg.num_experts:
            raise ValueError(
                f"{where}: float gate output is {tensor.shape[1]} wide, not "
                f"num_experts={cfg.num_experts}. These are not router logits.")
        scores = gate_scores(tensor.float(), cfg)
        return torch.topk(scores, cfg.top_k, dim=-1).indices

    if tensor.shape[1] != cfg.top_k:
        raise ValueError(
            f"{where}: integer gate output is {tensor.shape[1]} wide, not "
            f"top_k={cfg.top_k}. Expected already-selected expert ids.")
    if check_values:
        lo, hi = int(tensor.min()), int(tensor.max())
        if lo < 0 or hi >= cfg.num_experts:
            raise ValueError(
                f"{where}: expert ids span [{lo}, {hi}], outside "
                f"[0, {cfg.num_experts - 1}]. The config and the model "
                "disagree about how many experts there are.")
    return tensor


class GateRecorder:
    """Accumulates per-layer expert counts for the current batch.

    Counts stay on the device until `snapshot()`. Copying to host inside the
    hook would force a sync per layer per decode step, serialising the
    generation loop for the sake of a few hundred bytes.
    """

    def __init__(self, cfg: MoEConfig, num_layers: int):
        self.cfg = cfg
        self.num_layers = num_layers
        self.counts: torch.Tensor | None = None
        self.handles: list = []
        self.index: dict[str, int] = {}
        self.names: dict[int, str] = {}
        self.token_mask: torch.Tensor | None = None
        self._validated: set[int] = set()

    def attach(self, gates) -> None:
        for i, (name, module) in enumerate(gates):
            self.index[name] = i
            self.names[i] = name
            self.handles.append(module.register_forward_hook(self._make_hook(i)))

    def set_token_mask(self, mask) -> None:
        """Which rows of the NEXT forward pass are real tokens, not padding.

        A padded batch runs its PAD positions through the router like any other
        token. The attention mask keeps them out of the attention scores; it
        does not stop the gate from scoring them, and every PAD routes on the
        same embedding, so they all pile onto whichever experts that one vector
        happens to like. A prefill capture over a ragged batch can be mostly
        padding, and nothing downstream can tell -- the histogram is
        well-formed, it just describes the pad token.

        `mask` is a flat bool tensor of length batch*seq_len, matching the rows
        the gate sees. `None` counts every row, which is what a decode step
        wants: one real token per sequence and no padding at all.
        """
        self.token_mask = None if mask is None else mask.reshape(-1).bool()

    def _ensure(self, device) -> torch.Tensor:
        """Allocate the accumulator once, on the first device that reports.

        `device_map="auto"` can place later layers on a second GPU or offload
        them to CPU. Reallocating on a device change would silently discard
        everything accumulated so far and write a trace covering only the last
        device's layers. So the buffer is pinned to the first device seen and
        later contributions are moved to it.
        """
        if self.counts is None:
            self.counts = torch.zeros((self.num_layers, self.cfg.num_experts),
                                      dtype=torch.long, device=device)
        return self.counts

    def _make_hook(self, layer: int):
        cfg = self.cfg

        def hook(_module, _inputs, output):
            where = self.names.get(layer, f"gate {layer}")
            # The value-range check syncs, so pay for it once per layer. A
            # config that disagrees with the model disagrees on every step.
            first_time = layer not in self._validated
            ids = routed_ids(output, cfg, where=where, check_values=first_time)
            self._validated.add(layer)

            mask = self.token_mask
            if mask is not None:
                if mask.numel() != ids.shape[0]:
                    raise ValueError(
                        f"{where}: token mask covers {mask.numel()} rows but "
                        f"the gate saw {ids.shape[0]}. The mask is stale; set "
                        "it for every forward pass or clear it with None.")
                ids = ids[mask.to(ids.device)]

            counts = self._ensure(ids.device)
            contribution = expert_counts(ids, cfg.num_experts)
            counts[layer] += contribution.to(counts.device)

        return hook

    def reset(self) -> None:
        if self.counts is not None:
            self.counts.zero_()

    def snapshot(self):
        """Per-layer counts as a host int32 array. One transfer per batch."""
        import numpy as np

        if self.counts is None:
            return np.zeros((self.num_layers, self.cfg.num_experts), dtype=np.int32)
        return self.counts.cpu().numpy().astype(np.int32)

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
