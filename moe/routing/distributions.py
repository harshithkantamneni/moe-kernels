"""Parametric routing distributions and exact histogram realisation.

Two distinct jobs live here.

`sample_topk_ids` draws a routing decision from a parametric distribution. It
sweeps imbalance continuously, from uniform to pathological, which is what makes
the load-imbalance story legible as a curve rather than two anecdotes.

`realize_counts` does the opposite: given a target per-expert histogram, it
constructs a concrete top-k assignment whose column histogram matches EXACTLY.
That is what turns a captured trace (which is stored as a histogram, since only
group sizes affect the grouped GEMM) back into a replayable routing decision.
"""
from __future__ import annotations

import heapq

import torch

from ..spec import BenchSpec, RoutingSpec


def expert_probs(routing: RoutingSpec, num_experts: int,
                 generator: torch.Generator) -> torch.Tensor:
    """Per-expert selection probability, shape [E], summing to 1.

    Which expert ends up hot is randomised per seed, so results do not depend on
    an artefact of expert 0 always being the favoured one.
    """
    E = num_experts
    device = generator.device
    kind, param = routing.kind, routing.param

    if kind == "uniform":
        return torch.full((E,), 1.0 / E, dtype=torch.float64, device=device)

    if kind == "zipf":
        ranks = torch.arange(1, E + 1, dtype=torch.float64, device=device)
        p = ranks ** (-param)
        p = p / p.sum()
        perm = torch.randperm(E, generator=generator, device=device)
        return p[perm]

    if kind == "hot":
        if E == 1:
            return torch.ones(1, dtype=torch.float64, device=device)
        p = torch.full((E,), (1.0 - param) / (E - 1), dtype=torch.float64,
                       device=device)
        hot = int(torch.randint(0, E, (1,), generator=generator, device=device))
        p[hot] = param
        return p / p.sum()

    if kind == "dirichlet":
        # Dirichlet(alpha) via normalised Gamma(alpha, 1) draws. Small alpha
        # concentrates nearly all mass on a few experts.
        g = torch._standard_gamma(
            torch.full((E,), float(param), dtype=torch.float32, device=device))
        g = g.double().clamp_min(1e-12)
        return g / g.sum()

    raise ValueError(f"{kind!r} has no parametric form; use a trace")


def sample_topk_ids(routing: RoutingSpec, num_tokens: int, num_experts: int,
                    top_k: int, seed: int = 0,
                    device: str = "cpu") -> torch.Tensor:
    """Draw [T, k] distinct expert ids per token from a parametric distribution.

    Sampling without replacement proportional to `expert_probs` is done with the
    Gumbel top-k trick: perturb log-probabilities with Gumbel noise and take the
    k largest. That is exact Plackett-Luce sampling without replacement, and it
    is one vectorised operation rather than a rejection loop.
    """
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    g = torch.Generator(device=device).manual_seed(seed)
    p = expert_probs(routing, num_experts, g)

    logp = torch.log(p.clamp_min(1e-300)).float()
    u = torch.rand((num_tokens, num_experts), generator=g, device=device)
    # Gumbel(0,1) = -log(-log(U)). Note the parentheses: clamping must apply to
    # the negated logarithm, not to log(u) itself, or every key becomes NaN and
    # topk silently degenerates to "the first k experts".
    neg_log_u = (-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20)
    gumbel = -torch.log(neg_log_u)
    keys = logp.unsqueeze(0) + gumbel
    if not torch.isfinite(keys).all():  # pragma: no cover - defensive
        raise AssertionError("non-finite Gumbel keys; routing would be degenerate")
    return torch.topk(keys, top_k, dim=-1).indices.to(torch.int32)


def feasible(counts, num_tokens: int, top_k: int) -> tuple[bool, str]:
    """Can a [T, k] assignment with distinct experts per token realise `counts`?

    Two conditions, both necessary and together sufficient: the totals match,
    and no expert needs more rows than there are tokens (a token can contribute
    at most one row to any single expert).
    """
    c = [int(v) for v in counts]
    if any(v < 0 for v in c):
        return False, "counts contain a negative value"
    total = sum(c)
    if total != num_tokens * top_k:
        return False, (f"counts sum to {total}, need num_tokens*top_k = "
                       f"{num_tokens * top_k}")
    if max(c) > num_tokens:
        return False, (f"expert needs {max(c)} rows but there are only "
                       f"{num_tokens} tokens, and a token cannot pick the same "
                       "expert twice")
    return True, ""


def realize_counts(counts, num_tokens: int, top_k: int,
                   device: str = "cpu") -> torch.Tensor:
    """Build [T, k] int32 whose per-expert histogram equals `counts` exactly.

    Greedy by largest remaining demand. At each token, take the k experts that
    still need the most rows. This realises any feasible target: if some expert
    still needed rows at the end, it would have been among the k largest at
    every remaining token, so it could not have been skipped.
    """
    c = [int(v) for v in counts]
    ok, why = feasible(c, num_tokens, top_k)
    if not ok:
        raise ValueError(f"infeasible target histogram: {why}")

    heap = [(-n, e) for e, n in enumerate(c) if n > 0]
    heapq.heapify(heap)

    out = torch.zeros((num_tokens, top_k), dtype=torch.int32)
    for t in range(num_tokens):
        taken = []
        for slot in range(top_k):
            if not heap:
                raise AssertionError(
                    "ran out of demand before filling every slot; feasibility "
                    "check should have prevented this")
            neg, e = heapq.heappop(heap)
            out[t, slot] = e
            taken.append((neg + 1, e))  # one row consumed
        for neg, e in taken:
            if neg < 0:
                heapq.heappush(heap, (neg, e))

    if heap:
        raise AssertionError("unconsumed demand remains after realisation")
    return out.to(device)


def routing_source(spec: BenchSpec, device: str = "cpu",
                   traces=None) -> torch.Tensor | None:
    """Resolve a BenchSpec's RoutingSpec into forced top-k ids, or None.

    None means "let the model's own router decide", which is only meaningful
    when real weights are loaded. For random weights the router is arbitrary, so
    every benchmark cell should carry an explicit routing decision.
    """
    r = spec.routing
    if r.kind == "trace":
        if traces is None:
            raise ValueError(
                f"routing {r.label} needs a TraceSet; none was provided")
        return traces.forced_ids(r.trace_id, spec, device=device)
    return sample_topk_ids(r, spec.num_tokens, spec.model.num_experts,
                           spec.model.top_k, seed=spec.seed, device=device)
