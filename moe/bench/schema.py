"""The versioned result schema.

One CSV row per (cell x pipeline x timing mode). The schema is deliberately
wide and flat: every fact needed to interpret or reproduce a number lives in
the row itself, so a CSV committed to results/published/ is self-describing a
year later.

Bump SCHEMA_VERSION on any column change and leave old CSVs alone. The plotting
code checks the version and refuses to mix incompatible files.

Old CSVs stay READABLE (see READABLE_VERSIONS) as long as the change is
additive, because a published arm is a measurement that cannot be re-taken
cheaply. What an old row must never do is answer a question it has no data for,
so every column added after a row was written reads back as UNRECORDED.
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

SCHEMA_VERSION = 4
# v4: added the tile_* block and sm_capability, i.e. the tile configuration that
#     ACTUALLY ran. Until v4 the only tile columns were load_tile_eff_bm64 and
#     load_tile_eff_bm128, which are HYPOTHETICAL efficiencies computed from the
#     routing histogram at ASSUMED block sizes; nothing in a row said which tile
#     Triton chose. A wrong BLOCK_SIZE_M therefore sat unchallenged in an
#     analysis for days. Probing vLLM 0.27.1 directly settled it: only 2 of 8
#     (model x card) cells ran a TUNED config, nothing ships for
#     NVIDIA_A100-SXM4-80GB at any of the four shapes, and the tuned lookup uses
#     NEAREST key rather than floor -- so the A100 and the H200 ran DIFFERENT
#     tiles at the measured crossings, which no published row could have shown.
# v3: added bw_ceiling_pattern. Without it, two CSVs run with a different
#     --ceiling are silently incomparable, which defeats the point of naming a
#     pattern instead of taking max() across them.
# v2: dropped pct_of_achieved_bw (exactly reciprocal to implied_traffic_ratio),
#     added pct_of_achieved_tflops, load_tile_eff_bm64, load_tile_eff_bm128,
#     and renamed achieved_bf16_tflops -> achieved_peak_tflops.

#: Versions this code can still READ. Ten published arms were measured under v3
#: and cost real GPU hours; a version gate that refuses them retires the DATA
#: rather than the code, which is why `tile_efficiency_for_row` had to
#: reconstruct a missing tile column from stored ones instead of a new column
#: being added. Reading an old arm is safe here only because every v4 column is
#: NEW: no v3 column changed meaning, so a v3 row is a v4 row with a known hole
#: in it, and `read_csv` marks the hole rather than filling it.
#:
#: WRITING is still single-version. CsvWriter refuses to append under a header
#: from another schema, so a v3 run cannot be resumed by v4 code, and
#: merge_csvs refuses to mix versions in one output file.
READABLE_VERSIONS = frozenset({3, 4})

#: What a v4-only column reads as on a row that predates it.
#:
#: NOT 0, and this is the entire point of the column set. `tile_block_m == 0`
#: is a number: it survives a float(), it plots, it averages, and it reads as
#: "BLOCK_M was zero" -- the same shape of silent-default bug that
#: `pct_of_achieved_bw` printed as a clean column of 0% down a page. A string
#: that no numeric path accepts forces every reader to make a decision.
UNRECORDED = "<unrecorded>"

#: Which columns each schema version ADDED. Used to stamp UNRECORDED into the
#: columns an older row could not have carried, so "this row predates the
#: column" stays distinguishable from "this row measured zero".
COLUMNS_ADDED_IN: dict[int, tuple[str, ...]] = {
    4: ("tile_block_m", "tile_block_n", "tile_block_k", "tile_group_m",
        "tile_num_warps", "tile_num_stages", "tile_config_source",
        "tile_config_key", "sm_capability"),
}

#: Legal values of tile_config_source. Closed, and validated where a row is
#: populated, for the reason TERMINAL_STATUSES is closed: a typo'd source is not
#: a loud failure, it is a value no analysis matches, so those rows quietly
#: disappear from every group-by that keys on the source.
TILE_SOURCES: frozenset[str] = frozenset({
    "vllm_tuned",      # vLLM found a tuned JSON for this (E, N, dtype, device)
    "vllm_default",    # no tuned file: the hardcoded M<=32/96/512 fallback ladder
    "vllm_override",   # an override_config context was active, e.g. tile_sweep.py
    "sglang",          # SGLang's own selection, not read out of the process
    "cutlass_static",  # torch grouped_mm: CUTLASS picks its tile, not Triton
    "unrecorded",      # the hook did not run or could not observe
    "n/a",             # this span has no tile configuration to record
})


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
    #: torch.cuda.get_device_capability as "9.0" / "8.0". The gpu_name column
    #: names the card but nothing derived the ARCHITECTURE from it, and that is
    #: the exact discriminator the wgmma question needed: an H200 is sm90 and
    #: emits wgmma at BLOCK_M % 64 == 0 with num_warps % 4 == 0, while an A100
    #: is sm80 and can never emit it whatever tile it runs.
    sm_capability: str = ""
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
    # Useful rows / rows a fixed-BLOCK_M schedule must compute. HYPOTHETICAL:
    # both are computed from the routing histogram at an ASSUMED block size, and
    # neither is evidence about the tile the kernel ran. The tile_* block below
    # is the observed one.
    load_tile_eff_bm64: float = 1.0
    load_tile_eff_bm128: float = 1.0

    # --- tile configuration that actually ran (v4) ------------------------
    # Observed from the implementation itself, outside every timed region. 0
    # means UNRECORDED, never "the kernel used a block size of zero", and
    # `tile_field` refuses to hand a 0 back as a measurement for that reason.
    # Read them through `tile_field`, not through row_float.
    tile_block_m: int = 0
    tile_block_n: int = 0
    tile_block_k: int = 0
    tile_group_m: int = 0
    tile_num_warps: int = 0
    tile_num_stages: int = 0
    #: One of TILE_SOURCES. Defaults to "unrecorded" so a span with no observer
    #: says so, rather than a plausible-looking source being assumed for it.
    tile_config_source: str = "unrecorded"
    #: The M key the tuned file's NEAREST-key rule selected, 0 when there is no
    #: tuned file to key into. Load-bearing and non-obvious: vLLM resolves the
    #: entry with `min(configs.keys(), key=lambda x: abs(x - M))`, so M=787
    #: selects the key 1024, NOT 512. A floor reading of that lookup is how a
    #: wrong BLOCK_SIZE_M gets attributed to a row that never ran it.
    tile_config_key: int = 0

    # --- timing method ----------------------------------------------------
    # Recorded, never assumed. Most published MoE numbers omit these two and
    # are therefore not comparable to each other.
    l2_flush: bool = False
    flush_mb: int = 0
    flush_mode: str = ""            # read | write
    cuda_graph: bool = False
    capture_status: str = ""        # captured | not_capturable | n/a | skipped
    graph_skip_reason: str = ""
    # Every timed iteration replays ONE routing decision, so branch prediction
    # and cache behaviour are best-case relative to production, where routing
    # changes every step. Recorded rather than left for a reader to discover.
    routing_fixed_across_iters: bool = True
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
    # Measured ceilings from scripts/calibrate_hardware.py, so efficiency can be
    # quoted against what this machine actually delivers rather than a datasheet
    # peak it will never reach.
    achieved_bw_gbps: float = 0.0
    # Which STREAM pattern defined achieved_bw_gbps: read | copy | triad | write.
    bw_ceiling_pattern: str = ""
    # The measured compute ceiling for THIS row's dtype, not necessarily bf16.
    achieved_peak_tflops: float = 0.0
    # Compute-side efficiency against the measured cuBLAS ceiling. Not
    # derivable from the memory-side number, so both are carried.
    pct_of_achieved_tflops: float = 0.0
    # Counter-free stand-in for measured DRAM traffic, which needs Nsight
    # Compute and a host permission a rented pod does not grant. Only emitted
    # for memory-bound cells; an UPPER bound on the re-read factor, since it
    # also absorbs occupancy and latency losses. See calibrate.py.
    #
    # A pct_of_achieved_bw column was removed as exactly redundant with this:
    # the two multiply to 100 for every input. Memory-side efficiency is
    # 100 / implied_traffic_ratio.
    implied_traffic_ratio: float = 0.0

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


#: How a bool is spelled once it has been through the CSV.
TRUTHY = ("True", "true", "1")

_FIELD_TYPES: dict[str, str] = {
    f.name: (f.type if isinstance(f.type, str) else f.type.__name__)
    for f in fields(Row)
}


_COLUMN_SET = frozenset(COLUMNS)


def _schema_key(key: str) -> str:
    """Reject a column name this schema has never had.

    A retired or mistyped column read through row_float() comes back as the
    default, which is indistinguishable from a real measurement of zero.
    `pct_of_achieved_bw` was dropped in v2 as redundant with
    implied_traffic_ratio, and an analysis that still asked for it printed a
    clean column of `0%` for every row without a single warning -- a number
    that looked like a finding.

    A missing VALUE still defaults, because an older CSV legitimately lacks a
    newer column. A key that is not in the schema at all is a caller bug.
    """
    if key not in _COLUMN_SET:
        import difflib
        near = difflib.get_close_matches(key, COLUMNS, n=3)
        hint = f"; did you mean {', '.join(near)}?" if near else ""
        raise KeyError(f"{key!r} is not a column in schema v{SCHEMA_VERSION}{hint}")
    return key


class TileConfigUnrecorded(LookupError):
    """This row does not say which tile ran, and nothing may stand in for it.

    Raised rather than returning 0 because the whole reason the tile_* columns
    exist is that a plausible-looking number with no measurement behind it is
    indistinguishable from a real one, and one such number steered an analysis
    for days.
    """


def _reject_sentinel(key: str, value) -> None:
    """A column stamped UNRECORDED is never a number, never a bool, never a
    default. Checked in every reader, so no path can quietly coerce it."""
    if value == UNRECORDED:
        raise TileConfigUnrecorded(
            f"{key!r} is not recorded on this row: it comes from a CSV written "
            f"under an older schema, which had no such column. Filter these "
            f"rows out with has_tile_config(row) instead of reading them.")


def row_bool(row: dict, key: str, default: bool = False) -> bool:
    """Read a bool from a CSV row. One spelling, everywhere.

    Three separate open-codings of this test had already drifted: one accepted
    "True"/"true"/"1" and another only "True", so two figures filtered the same
    column differently.
    """
    value = row.get(_schema_key(key))
    _reject_sentinel(key, value)
    if value is None or value == "":
        return default
    return str(value) in TRUTHY


def row_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(_schema_key(key))
    # Before the try: float("<unrecorded>") is a ValueError, which the clause
    # below would turn into the default -- exactly the silent substitution the
    # sentinel exists to prevent.
    _reject_sentinel(key, value)
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def has_tile_config(row: dict) -> bool:
    """Did this row record the tile the kernel actually ran?

    The predicate to filter on BEFORE grouping by BLOCK_M, so an analysis never
    has to catch TileConfigUnrecorded row by row.

    False for every v3 row, since the column did not exist; false for a v4 row
    whose span had no observer or whose observer could not read the config back;
    false for torch's CUTLASS grouped GEMM and for SGLang, which record a source
    and no numbers.

    Keyed on tile_block_m rather than on the source, because those are two
    different questions. A row can know the tile it ran and not know where the
    tile came from, and such a row is usable for everything except a
    tuned-versus-default split.
    """
    value = row.get("tile_block_m")
    if value in (None, "", UNRECORDED):
        return False
    return (_int_or_none(value) or 0) != 0


def _int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def tile_field(row: dict, key: str) -> int | str:
    """Read a v4 tile-provenance column, or raise. Never a usable default.

    The three ways this raises are the three ways the previous analysis went
    wrong, and each one has to be a stop rather than a value:

      - the row predates v4, so the column is stamped UNRECORDED;
      - the row is v4 but the observer never ran or could not read the config
        back, so tile_config_source is "unrecorded" and the ints are 0;
      - the field does not apply to this implementation (a CUTLASS grouped GEMM
        has no Triton BLOCK_SIZE_M, and a vllm_default row has no tuned-file
        key), so the int is 0 while the SOURCE is a real answer.

    tile_config_source comes back as a STRING on any row that carries the
    column, including "unrecorded" and "n/a": those are honest answers to "where
    did this configuration come from", so the second and third cases above raise
    for the tile ints and not for the source. On a v3 row it raises with the
    rest, because that row predates the column entirely and there is no answer to
    return -- "the source is unrecorded" and "this file is older than the
    concept of a source" are different facts, and only the sentinel keeps them
    apart.

    sm_capability is read through here too: it is not a tile field, but it is
    the sm80-vs-sm90 discriminator the tile question needs, it arrived in the
    same version, and an empty string from a v3 row would read as "no CUDA
    device" rather than "this file is older than the column".
    """
    _schema_key(key)
    if key not in COLUMNS_ADDED_IN[4]:
        raise ValueError(
            f"{key!r} is not a v4 provenance column; read it with row_float, "
            f"row_bool or row.get. tile_field covers "
            f"{', '.join(COLUMNS_ADDED_IN[4])}.")
    value = row.get(key)
    _reject_sentinel(key, value)
    if value is None or value == "":
        raise TileConfigUnrecorded(
            f"{key!r} is empty on this row; nothing observed the tile "
            f"configuration when it was written.")
    if key in ("tile_config_source", "sm_capability"):
        return str(value)
    parsed = _int_or_none(value)
    if parsed is None:
        raise TileConfigUnrecorded(f"{key!r} is {value!r}, which is not an int")
    if parsed == 0:
        raise TileConfigUnrecorded(
            f"{key!r} is 0, which means UNRECORDED and not a measured zero "
            f"(tile_config_source={row.get('tile_config_source')!r}). "
            f"A block size of zero does not exist.")
    return parsed


def passed(row: dict) -> bool:
    """Did this row earn its timing numbers?

    The driver also enforces this structurally by zeroing timing fields on a
    failed row, so this is a second line of defence rather than the only one.
    """
    return row_bool(row, "correctness_passed")


def series_label(row: dict, label_col: str = "impl") -> str:
    """Plot series key that never folds incomparable methodologies together.

    An L2-flushed measurement and an L2-warm one are different experiments, and
    so are eager and graph replay. This lived in two files with two slightly
    different answers, so the same row was labelled differently depending on
    which figure it landed in.
    """
    bits = [row.get(label_col, "")]
    bits.append("L2-flushed" if row_bool(row, "l2_flush") else "L2-warm")
    if row_bool(row, "cuda_graph"):
        bits.append("graph")
    if row.get("scope") == "pipeline":
        bits.append("full layer")
    return " / ".join(b for b in bits if b)


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
        if not new:
            # Appending under a header from a different schema would write rows
            # in the new field order beneath the old names, misaligning every
            # column from that point on. Resume must refuse instead.
            with self.path.open(newline="") as fh:
                existing = next(csv.reader(fh), [])
            if existing != COLUMNS:
                raise ValueError(
                    f"{self.path} has a header from a different schema "
                    f"({len(existing)} columns, this code writes {len(COLUMNS)}). "
                    "Start a new run id rather than appending; merge_csvs "
                    "refuses mixed versions too.")
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


# Manifest outcome codes. Constants rather than bare literals at the call
# sites: a typo'd status silently falls outside TERMINAL_STATUSES, and the cell
# would then retry forever with nothing to indicate why.
STATUS_OK = "ok"
STATUS_CORRECTNESS_FAILED = "correctness_failed"
STATUS_NOT_CAPTURABLE = "not_capturable"
STATUS_INVALID_PIPELINE = "invalid_pipeline"
STATUS_ERROR = "error"
STATUS_CRASH = "crash"
#: MOE_FORCE_TILE was set and this cell's implementation has no way to honour
#: it, so it was recorded rather than measured. See moe/bench/force_tile.py.
STATUS_FORCE_TILE_UNHONOURABLE = "force_tile_unhonourable"
#: The cell ran pinned but the tile it was OBSERVED running is not the tile that
#: was forced, so no row was written: a row that says it was pinned when it was
#: not is worse than no pinning at all.
STATUS_FORCE_TILE_NOT_OBSERVED = "force_tile_not_observed"

#: Outcomes that are deterministic, so re-running would reproduce them exactly.
#: Anything else (a CUDA OOM, a crash) is transient and MUST stay retryable, or
#: a single bad moment permanently blanks that cell from every future run.
#:
#: NEITHER force-tile status is terminal, and that is deliberate rather than an
#: omission. Both describe the cell UNDER A PIN, not the cell: the same cell is
#: measurable in a run without MOE_FORCE_TILE set, and marking it done here
#: would blank it from every future unpinned resume of the same run id. The pin
#: is in the manifest KEY (see driver._cell_key), so a pinned skip and an
#: unpinned measurement of the same cell are different records either way.
TERMINAL_STATUSES = frozenset({STATUS_OK, STATUS_CORRECTNESS_FAILED,
                               STATUS_NOT_CAPTURABLE, STATUS_INVALID_PIPELINE})


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


def _stamp_unrecorded(row: dict, version: int) -> None:
    """Mark the columns a row of this version could not have carried.

    Called on every read rather than on the write side, so the mark is derived
    from schema_version and cannot drift out of sync with it: a row that
    round-trips through merge_csvs comes back stamped the same way, because the
    version travels with the row.
    """
    for added_in, names in COLUMNS_ADDED_IN.items():
        if version >= added_in:
            continue
        for name in names:
            row[name] = UNRECORDED


def read_csv(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a results CSV, refusing a schema version this code cannot read.

    A version in READABLE_VERSIONS but older than SCHEMA_VERSION loads with the
    columns it predates stamped UNRECORDED. That is the whole reason the gate
    widened: the ten published v3 arms have no tile_* columns, and the choice is
    between loading them with the hole marked or not loading them at all. It is
    NOT a choice between marking the hole and filling it -- filling it with the
    dataclass defaults would say every published row ran BLOCK_M 0.
    """
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        v = int(r.get("schema_version", -1))
        if v not in READABLE_VERSIONS:
            raise ValueError(
                f"{path}: schema_version {v}, this code reads "
                f"{sorted(READABLE_VERSIONS)}. "
                "Re-run the benchmark or read it with the matching commit."
            )
        if v != SCHEMA_VERSION:
            _stamp_unrecorded(r, v)
    return rows


def merge_csvs(paths: Iterable[str | os.PathLike], out_path: str | os.PathLike) -> int:
    """Concatenate result CSVs from separate venv subprocesses into one file.

    REBUILDS out_path. CsvWriter is append-only because a run CSV must survive a
    killed pod and be resumable, but a merge is derived: everything in it comes
    from `paths`, so appending to a previous merge can only add rows that no
    longer have an input to justify them. run_all.sh merges into
    `results/merged.csv` once per sweep and results/ outlives a session, so under
    the old behaviour every sweep silently inherited every sweep before it. The
    2026-08-26 published arm carried 872 such rows, 840 of them from a sweep
    measured against a different calibration, which makes an efficiency column
    read against the wrong ceiling with nothing in the row to say so.
    """
    written = 0
    out = Path(out_path)
    versions: set[int] = set()
    if out.exists():
        out.unlink()
    with CsvWriter(out) as w:
        for p in paths:
            for r in read_csv(p):
                versions.add(int(r.get("schema_version", -1)))
                if len(versions) > 1:
                    # One file, one header, one column set. read_csv accepts v3
                    # and v4 so the published arms stay loadable, but a merge
                    # writes them under a SINGLE header, and a v3 row sitting
                    # under v4 column names would claim tile columns it never
                    # had. Refuse instead: analyse the arms separately.
                    raise ValueError(
                        f"{out_path}: inputs mix schema versions "
                        f"{sorted(versions)}. Merge each version into its own "
                        "file; the tile_* columns exist in one and not the "
                        "other, and one header cannot describe both.")
                row = Row(**{k: _coerce(k, v) for k, v in r.items() if k in COLUMNS})
                w.write(row)
                written += 1
    return written


def _coerce(name: str, value: str):
    t = _FIELD_TYPES[name]
    if value == UNRECORDED:
        # Back to the typed zero rather than into the dataclass as a string.
        # Nothing is lost: the row keeps its schema_version, so read_csv stamps
        # the sentinel back on the way out and the hole stays marked. Keeping
        # the string here would put a str in an int column and every consumer
        # of a merged file would have to know that.
        return {"bool": False, "int": 0, "float": 0.0}.get(t, "")
    if t == "bool":
        return value in TRUTHY
    if t == "int":
        return int(float(value)) if value != "" else 0
    if t == "float":
        return float(value) if value != "" else 0.0
    return value
