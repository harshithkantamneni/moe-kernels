"""Locating MoE gates in an unfamiliar model, and recording what they route.

This is library code, not script code: `find_gate_modules` encodes a subtle
model-layout rule with a near-miss that has already bitten once, and the
recorder must apply the same per-model gating function the reference oracle
does. Keeping it here means one gating rule, tested in one place.

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

    def attach(self, gates) -> None:
        for i, (name, module) in enumerate(gates):
            self.index[name] = i
            self.handles.append(module.register_forward_hook(self._make_hook(i)))

    def _ensure(self, device) -> torch.Tensor:
        if self.counts is None or self.counts.device != device:
            self.counts = torch.zeros((self.num_layers, self.cfg.num_experts),
                                      dtype=torch.long, device=device)
        return self.counts

    def _make_hook(self, layer: int):
        cfg = self.cfg

        def hook(_module, _inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.detach().float().reshape(-1, cfg.num_experts)
            # The library owns the per-model gating rule. A copy of it here
            # would silently keep recording softmax counts the day a model
            # needs a third scoring function.
            ids = torch.topk(gate_scores(logits, cfg), cfg.top_k, dim=-1).indices
            counts = self._ensure(logits.device)
            counts[layer] += expert_counts(ids, cfg.num_experts)

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
