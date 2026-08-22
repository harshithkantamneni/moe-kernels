"""The versioned result schema.

One CSV row per (cell x pipeline x timing mode). The schema is deliberately
wide and flat: every fact needed to interpret or reproduce a number lives in
the row itself, so a CSV committed to results/published/ is self-describing a
year later.

Bump SCHEMA_VERSION on any column change and leave old CSVs alone. The plotting
code checks the version and refuses to mix incompatible files.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Row:
    # --- provenance -------------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    git_sha: str = ""
    git_dirty: bool = False

    # --- machine ----------------------------------------------------------
    gpu_name: str = ""
    gpu_count: int = 0
    device_index: int = 0
    sm_count: int = 0
    l2_bytes: int = 0
    total_memory: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    triton_version: str = ""
    env_name: str = "base"          # base | vllm | sglang
    env_version: str = ""           # version of the framework this env provides
    host_cpu: str = ""
    python_version: str = ""
    # Numerics switches whose defaults have moved between torch releases. They
    # change both the reference's numerics and any torch-backed baseline's speed.
    allow_tf32: bool = False
    allow_bf16_reduced_reduction: bool = False
    allow_fp16_reduced_reduction: bool = False

    # --- cell -------------------------------------------------------------
    model: str = ""
    hidden_size: int = 0
    intermediate_size: int = 0
    num_experts: int = 0
    top_k: int = 0
    num_tokens: int = 0
    rows: int = 0                   # num_tokens * top_k
    dtype: str = ""
    routing_kind: str = ""
    routing_param: float = 0.0
    trace_id: str = ""
    seed: int = 0

    # --- what was measured ------------------------------------------------
    pipeline: str = ""              # full tiling label
    impl: str = ""                  # the span under study, or __pipeline__
    scope: str = "span"             # span | pipeline: what the timer wrapped
    covers: str = ""                # e.g. "down_gemm+unpermute"
    cuda_graph_safe: bool = False

    # --- observed load ----------------------------------------------------
    load_num_experts: int = 0
    load_total_rows: int = 0
    load_active_experts: int = 0
    load_empty_experts: int = 0
    load_max_rows: int = 0
    load_min_rows: int = 0
    load_mean_rows: float = 0.0
    load_max_over_mean: float = 0.0
    load_cv: float = 0.0
    load_entropy_norm: float = 0.0
    load_gini: float = 0.0
    load_top1_share: float = 0.0

    # --- timing method ----------------------------------------------------
    # Recorded, never assumed. Most published MoE numbers omit these two and
    # are therefore not comparable to each other.
    l2_flush: bool = False
    flush_mb: int = 0
    flush_mode: str = ""            # read | write
    cuda_graph: bool = False
    capture_status: str = ""        # captured | not_capturable | n/a
    warmup: int = 0
    iters: int = 0
    trials: int = 0

    # --- timing result ----------------------------------------------------
    ms_p50: float = 0.0
    ms_p90: float = 0.0
    ms_min: float = 0.0
    ms_std: float = 0.0
    jitter_p90_over_p50: float = 0.0

    # --- derived ----------------------------------------------------------
    flops: float = 0.0
    # Compulsory minimum traffic from the tiling's own contracts. A real kernel
    # re-reads tiles, so this is a LOWER bound on traffic and the derived
    # intensity is an UPPER bound. Named accordingly so no reader mistakes
    # compulsory_gbps for achieved HBM bandwidth.
    compulsory_bytes: float = 0.0
    tflops: float = 0.0
    compulsory_gbps: float = 0.0
    arith_intensity_compulsory: float = 0.0

    # --- input construction -----------------------------------------------
    input_init: str = "fan_in"      # how weights/activations were generated
    input_scale: float = 1.0
    trace_sha: str = ""             # fingerprint of the replayed trace, if any

    # --- correctness gate -------------------------------------------------
    # A timing row is only written when correctness_passed is True. For a
    # cuda_graph row the verdict is re-earned against the REPLAYED output.
    correctness_passed: bool = False
    max_abs_err: float = 0.0
    # Scale-free: max|got-ref| / max|ref|. Compared against tol_rel_max.
    rel_err: float = 0.0
    tol_rel_max: float = 0.0
    tol_calibrated: bool = False
    oracle: str = "golden_fp32"

    # --- thermal / clock drift -------------------------------------------
    sm_clock_start_mhz: int = 0
    sm_clock_end_mhz: int = 0
    temp_start_c: int = 0
    temp_end_c: int = 0
    clock_drift_pct: float = 0.0
    throttled: bool = False

    notes: str = ""


COLUMNS: list[str] = [f.name for f in fields(Row)]


def cell_key(row: Row) -> str:
    """Identity of a unit of work, for resume. Excludes timing results."""
    parts = [row.model, str(row.num_tokens), row.dtype, row.routing_kind,
             f"{row.routing_param:g}", row.trace_id, str(row.seed),
             row.pipeline, row.impl, row.scope,
             str(int(row.l2_flush)), str(int(row.cuda_graph))]
    return "|".join(parts)


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

class CsvWriter:
    """Append-only, flushed per row, so a killed pod loses at most one cell."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Row) -> None:
        self._writer.writerow(asdict(row))
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


#: Outcomes that are deterministic, so re-running would reproduce them exactly.
#: Anything else (a CUDA OOM, a crash) is transient and MUST stay retryable, or
#: a single bad moment permanently blanks that cell from every future run.
TERMINAL_STATUSES = frozenset({"ok", "correctness_failed", "not_capturable",
                               "invalid_pipeline", "unsupported"})


class Manifest:
    """JSONL record of completed cell keys, for resuming an interrupted sweep."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a killed pod is not fatal
                if "key" not in rec:
                    continue
                if rec.get("status", "ok") in TERMINAL_STATUSES:
                    self.done.add(rec["key"])
        self._fh = self.path.open("a")

    def __contains__(self, key: str) -> bool:
        return key in self.done

    def record(self, key: str, status: str = "ok", detail: str = "") -> None:
        """Log an outcome. Only a deterministic outcome marks the cell done."""
        if status in TERMINAL_STATUSES:
            self.done.add(key)
        self._fh.write(json.dumps({"key": key, "status": status, "detail": detail}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------
# provenance capture
# --------------------------------------------------------------------------

def git_provenance(repo_root: str | os.PathLike | None = None) -> tuple[str, bool]:
    root = str(repo_root or Path(__file__).resolve().parents[2])
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return "", False
        return sha.stdout.strip(), bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "", False


def read_csv(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a results CSV, refusing to load a schema version this code cannot read."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        v = int(r.get("schema_version", -1))
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {v}, this code reads {SCHEMA_VERSION}. "
                "Re-run the benchmark or read it with the matching commit."
            )
    return rows


def merge_csvs(paths: Iterable[str | os.PathLike], out_path: str | os.PathLike) -> int:
    """Concatenate result CSVs from separate venv subprocesses into one file."""
    written = 0
    with CsvWriter(out_path) as w:
        for p in paths:
            for r in read_csv(p):
                row = Row(**{k: _coerce(k, v) for k, v in r.items() if k in COLUMNS})
                w.write(row)
                written += 1
    return written


def _coerce(name: str, value: str):
    field_type = {f.name: f.type for f in fields(Row)}[name]
    t = field_type if isinstance(field_type, str) else field_type.__name__
    if t == "bool":
        return value in ("True", "true", "1")
    if t == "int":
        return int(float(value)) if value != "" else 0
    if t == "float":
        return float(value) if value != "" else 0.0
    return value
