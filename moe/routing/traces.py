"""Captured routing traces: storage format, selection, rescaling, replay.

What is stored is a per-layer expert-count histogram, not a token log. Only the
multiset of group sizes affects the grouped GEMM, so a histogram is sufficient
and it is kilobytes rather than megabytes, which is what makes these traces
committable and the benchmarks reproducible by anyone who clones the repo.

A trace id may name a specific slice: "mixtral-chat-decode@b3l17" pins batch 3,
layer 17. Without a suffix, the slice is chosen deterministically from the cell
seed, so sweeping seeds samples different real layers rather than averaging them
into a mush (averaging skewed layers with different hot experts would look far
more uniform than any real layer ever is).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..spec import BenchSpec
from .distributions import feasible, realize_counts

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"
_SLICE_RE = re.compile(r"^(?P<base>.+?)@b(?P<batch>\d+)l(?P<layer>\d+)$")


@dataclass(frozen=True)
class Trace:
    trace_id: str
    counts: np.ndarray          # int64 [n_batches, n_layers, num_experts]
    meta: dict
    sha: str = ""               # sha256 of the .npz, first 16 hex chars

    @property
    def num_experts(self) -> int:
        return int(self.counts.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.counts.shape)

    def select(self, batch: int | None = None, layer: int | None = None,
               seed: int = 0) -> tuple[np.ndarray, int, int]:
        B, L, _ = self.shape
        b = seed % B if batch is None else batch
        ln = seed % L if layer is None else layer
        if not (0 <= b < B and 0 <= ln < L):
            raise IndexError(
                f"{self.trace_id}: slice b{b}l{ln} outside shape {self.shape}")
        return self.counts[b, ln].astype(np.int64), b, ln


def _largest_remainder(shares: np.ndarray, total: int) -> np.ndarray:
    """Round `shares` to integers summing exactly to `total`.

    Floor everything, then hand the leftover units to the largest fractional
    parts. Naive rounding distorts the shape of a distribution; this does not,
    which is why both the rescale and the redistribution below need it.
    """
    out = np.floor(shares).astype(np.int64)
    left = total - int(out.sum())
    if left > 0:
        order = np.argsort(-(shares - out))
        out[order[:left]] += 1
    return out


def rescale_counts(counts, num_tokens: int, top_k: int) -> list[int]:
    """Fit a captured histogram to this cell's token budget, exactly.

    The captured batch almost never has the same token count as the cell being
    benchmarked, so counts are scaled to the target total. Two properties are
    preserved by construction: the total is exactly num_tokens*top_k, and no
    expert exceeds num_tokens (a token cannot select one expert twice). The
    largest-remainder method is used for the fractional part so the shape of the
    distribution survives, rather than being distorted by naive rounding.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {c.shape}")
    E = c.size
    target = num_tokens * top_k
    if E < top_k:
        raise ValueError(f"cannot route top_k={top_k} across only {E} experts")
    if c.sum() <= 0:
        base = target // E
        out = [base] * E
        for i in range(target - base * E):
            out[i] += 1
    else:
        out = _largest_remainder(c * (target / c.sum()), target).tolist()

    out = _cap_and_redistribute(out, num_tokens, target)
    ok, why = feasible(out, num_tokens, top_k)
    if not ok:  # pragma: no cover - guarded by the construction above
        raise AssertionError(f"rescale produced an infeasible histogram: {why}")
    return out


def _cap_and_redistribute(counts: list[int], cap: int, target: int) -> list[int]:
    """Clamp every expert to `cap`, pushing the excess onto experts with headroom."""
    out = list(counts)
    for _ in range(len(out) + 1):
        excess = sum(max(0, v - cap) for v in out)
        if excess == 0:
            break
        out = [min(v, cap) for v in out]
        headroom = [cap - v for v in out]
        total_head = sum(headroom)
        if total_head <= 0:
            raise ValueError(
                f"cannot fit {target} rows into {len(out)} experts capped at {cap}")
        # Largest-remainder again, so the redistribution does not itself skew.
        share = np.array(headroom, dtype=np.float64) * (excess / total_head)
        add = _largest_remainder(share, excess)
        out = [v + int(a) for v, a in zip(out, add, strict=True)]
    else:  # pragma: no cover - the loop bound is generous
        raise ValueError("redistribution did not converge")
    return out


class TraceSet:
    """All traces available on disk, keyed by trace id."""

    def __init__(self, traces: dict[str, Trace]):
        self.traces = traces

    @classmethod
    def load(cls, directory: str | Path = TRACE_DIR) -> TraceSet:
        directory = Path(directory)
        found: dict[str, Trace] = {}
        for path in sorted(directory.glob("*.npz")):
            t = load_trace(path)
            found[t.trace_id] = t
        return cls(found)

    def __contains__(self, trace_id: str) -> bool:
        return _parse_slice(trace_id)[0] in self.traces

    def __len__(self) -> int:
        return len(self.traces)

    def get(self, trace_id: str) -> tuple[Trace, int | None, int | None]:
        base, batch, layer = _parse_slice(trace_id)
        if base not in self.traces:
            raise KeyError(
                f"no trace {base!r}; available: {sorted(self.traces)}. "
                "Run scripts/capture_traces.py on the GPU box to create one.")
        return self.traces[base], batch, layer

    def forced_ids(self, trace_id: str, spec: BenchSpec, device: str = "cpu"):
        """Replay a captured distribution as concrete [T, k] expert ids."""
        trace, batch, layer = self.get(trace_id)
        if trace.num_experts != spec.model.num_experts:
            raise ValueError(
                f"trace {trace_id} has {trace.num_experts} experts but "
                f"{spec.model.name} has {spec.model.num_experts}; a routing "
                "trace is only meaningful for the model it was captured from")
        counts, _, _ = trace.select(batch, layer, seed=spec.seed)
        fitted = rescale_counts(counts, spec.num_tokens, spec.model.top_k)
        return realize_counts(fitted, spec.num_tokens, spec.model.top_k,
                              device=device)

    def resolved_slice(self, trace_id: str, seed: int) -> str:
        trace, batch, layer = self.get(trace_id)
        _, b, ln = trace.select(batch, layer, seed=seed)
        return f"{trace.trace_id}@b{b}l{ln}"

    def provenance(self, spec) -> dict:
        """Row fields identifying exactly which slice of which file was replayed.

        `trace_id` alone is not enough: the same id can name different file
        contents, and an unsuffixed id resolves to a seed-dependent slice.
        """
        r = spec.routing
        if r.kind != "trace" or not r.trace_id:
            return {}
        trace, _, _ = self.get(r.trace_id)
        return {"trace_sha": trace.sha,
                "trace_id": self.resolved_slice(r.trace_id, spec.seed)}


def _parse_slice(trace_id: str) -> tuple[str, int | None, int | None]:
    m = _SLICE_RE.match(trace_id)
    if not m:
        return trace_id, None, None
    return m["base"], int(m["batch"]), int(m["layer"])


# --------------------------------------------------------------------------
# file format
# --------------------------------------------------------------------------

REQUIRED_META = ("trace_id", "model", "hf_repo", "corpus", "phase",
                 "num_experts", "top_k", "captured_at")


def write_trace(path: str | Path, counts: np.ndarray, meta: dict) -> Path:
    """Persist a trace. `counts` is [n_batches, n_layers, num_experts]."""
    counts = np.asarray(counts)
    if counts.ndim != 3:
        raise ValueError(f"counts must be [batches, layers, experts], got {counts.shape}")
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"trace metadata is missing {missing}")
    if int(meta["num_experts"]) != counts.shape[2]:
        raise ValueError(
            f"metadata says {meta['num_experts']} experts, counts has {counts.shape[2]}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, counts=counts.astype(np.int32),
                        meta=np.array(json.dumps(meta)))
    return path


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    with np.load(path, allow_pickle=False) as z:
        counts = z["counts"]
        meta = json.loads(str(z["meta"]))
    return Trace(trace_id=meta["trace_id"], counts=counts.astype(np.int64),
                 meta=meta, sha=sha)
