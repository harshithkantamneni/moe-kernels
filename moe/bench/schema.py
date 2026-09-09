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

SCHEMA_VERSION = 7
# v7: added the under-load clock TRACE, the board power beside it, and what the
#     warmup did to settle the clock. Until v7 a row carried the median, the
#     two verdicts and nothing a reader could re-examine: `sm_clock_start_mhz`
#     and `sm_clock_end_mhz` hold IDLE instants from the retired seam, so the
#     under-load first and last were computed and dropped, and the sample list
#     was dropped by every writer. The 2026-09-09 H200 session produced 135
#     DRIFT verdicts of which only four could be looked at, and the question
#     they had to answer (a governor settling after a workload change, or a
#     card hunting?) is exactly the one a median and two endpoints cannot.
#     v7 writes `sm_clock_load_first_mhz`,
#     `sm_clock_load_last_mhz`, `clock_samples_mhz` (the whole ordered list),
#     `clock_drift_direction` (settling upward / dropping / oscillating) and
#     `power_w`, read at the same NVML call as the clock so a LOW clock can be
#     told apart as a hungry tile at the 700 W cap from a card in trouble; plus
#     `warmup_settle_ms`, `warmup_settle_calls` and `warmup_clock_settled`,
#     which say whether the extra warmup that now runs until two consecutive
#     clock reads agree (timing v4) actually converged. The retired columns
#     KEEP their meaning: one column may not mean two things either side of a
#     version boundary, which is why the new pair is named for the load.
#
#     THE EXCLUSION RULE CHANGED IN THE SAME SESSION and no column moved with
#     it: `throttled` is now `clock_drift_ok == failed` alone. `clock_level_ok`
#     and `clock_level_side` are unchanged in meaning and are now a RECORD of
#     the tile's power state, never an exclusion. See `driver.
#     _apply_kernel_timing`.
# v6: added the reference the LEVEL verdict was scored against and the roof AT
#     THE CLOCK THE CELL RAN. Until v6 a v5 row said `clock_level_ok = ok`
#     without saying against what: the reference was resolved from a per-device
#     yaml that every recalibration overwrites, so a verdict could not be
#     re-derived from the CSV, and a recalibration that changed the field it was
#     read from (idle scalar to under-load median) silently changed what "ok"
#     meant between arms carrying the same basis string. Worse, LEVEL was
#     one-sided: a memory-shaped cell boosted to 1980 MHz against a compute roof
#     measured at 1515 passed, and `pct_of_achieved_tflops` on it was inflated
#     by 1980/1515 = 1.31x, toward the study's claim. v6 writes
#     `reference_clock_mhz` and `reference_clock_source` (with its grade) onto
#     every row, names the SIDE a LEVEL failure went (`clock_level_side`), and
#     carries `roof_at_cell_clock_tflops` / `pct_of_roof_at_cell_clock`, the
#     compute roof rescaled to the row's own `sm_clock_load_mhz` when the
#     reference is an under-load median, with `roof_note` saying why when it is
#     not. `achieved_peak_tflops` and `pct_of_achieved_tflops` KEEP their v2
#     meaning (the fixed roof) rather than being redefined: one column may not
#     mean two things either side of a version boundary. Also
#     `host_backlog_iters`, the ratio the host-bound verdict thresholds.
# v5: added the instrument block, i.e. WHICH TIMER produced the ms_* columns and
#     what the card was doing while it did. Until v5 the driver timed through
#     `timing.time_eager`/`time_graph` -- a COUNT of warmup calls, an iteration
#     count sized from one isolated call, and two idle-instant clock samples --
#     and no column said so, so every one of the 100,144 published rows is
#     indistinguishable, on its face, from a row measured on the instrument that
#     replaced it. The damage was not hypothetical: `throttled` (v1) compares two
#     samples taken AFTER a synchronise and fires when the START sample caught
#     the idle boost, which is not throttling, and it fired on 91% of vLLM rows
#     above T=4096 while flagged and unflagged replicates of one cell timed at
#     ratio 0.998. An analysis that drops those rows drops 0% at T=1 and 100% at
#     T=16384, i.e. along the very axis alpha is identified on.
#     `moe.bench.timing.time_kernel` warms for a DURATION of sustained load,
#     polls the SM clock from a background thread WHILE the trials run, and
#     reports three verdicts (LEVEL, DRIFT, host-bound). v5 is those columns plus
#     the instrument's own name, so a reader can tell the two apparatus apart
#     without knowing which commit wrote the file.
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
READABLE_VERSIONS = frozenset({3, 4, 5, 6, 7})

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
    5: ("instrument", "warmup_ms", "sm_clock_load_mhz", "clock_level_ok",
        "clock_drift_ok", "host_bound_ok", "clock_samples", "clock_source",
        "clock_note", "host_enqueue_ms", "host_note"),
    6: ("reference_clock_mhz", "reference_clock_source", "clock_level_side",
        "host_backlog_iters", "roof_at_cell_clock_tflops",
        "pct_of_roof_at_cell_clock", "roof_note"),
    7: ("sm_clock_load_first_mhz", "sm_clock_load_last_mhz",
        "clock_samples_mhz", "clock_drift_direction", "power_w",
        "warmup_settle_ms", "warmup_settle_calls", "warmup_clock_settled"),
}

#: What the ms_* columns of a row written before v5 were measured with.
#:
#: A NAME RATHER THAN A BLANK, and the difference is the whole point of the
#: version. `timing.TIMING_BASIS` names the instrument that replaced it; a row
#: that carries neither string carries no answer at all, and "no answer" is what
#: 100,144 published rows would otherwise say about the apparatus that produced
#: the study's headline number. So a pre-v5 row reads back as THIS, which is a
#: claim about it: warmed for a COUNT of calls, iteration count sized from one
#: isolated call on an idle GPU, no clock read during the trials, and the two
#: clock columns it does carry sampled at idle instants either side of the cell.
#: Never write it into a new row: `instrument_of` derives it from the version.
LEGACY_INSTRUMENT = "time_eager+time_graph/idle-instant-clock/pre-v5"

#: What a row the driver WROTE BUT NEVER TIMED says about its apparatus.
#:
#: THE THIRD ANSWER, and it exists because the first cut of this boundary had
#: only two and was wrong about the world. `instrument_of` refused an empty
#: `instrument` as "a row the driver wrote without going through either timer,
#: which cannot happen", and the same commit's driver wrote exactly that row on
#: four paths: a cell that failed the fp32 oracle, a graph mode skipped by cost
#: policy, a span that could not be graph-captured, and a timer that raised.
#: None of those is a bug, all of them belong in the CSV, and so
#: `has_kernel_timing` -- the predicate an analysis is told to split a pool on
#: BEFORE it reads a v5 column -- was the thing that threw, on rows an ordinary
#: sweep emits by the thousand.
#:
#: A NAME AND NOT A BLANK, for the reason `LEGACY_INSTRUMENT` is one. "Nothing
#: timed this row" is a fact about it, not an absence to guess at: its ms_*
#: columns are zero because no measurement was taken, not because a measurement
#: came out zero, and `capture_status` says which of the four paths it was.
NO_INSTRUMENT = "none/not-timed"

#: The instrument names whose rows carry NO readable v5 timing column. Legacy:
#: the columns did not exist when the row was written. Untimed: nothing ran, so
#: the columns hold dataclass defaults. `has_kernel_timing` is False for both
#: and `instrument_of` still tells them apart, which is the split that matters:
#: a legacy row has numbers measured the retired way, an untimed row has none.
NO_KERNEL_TIMING: frozenset[str] = frozenset({LEGACY_INSTRUMENT, NO_INSTRUMENT})

#: The three v5 verdict columns, and the closed vocabulary all three speak.
#:
#: STRINGS AND NOT BOOLS, because each verdict has three states and a bool has
#: two. `time_kernel` returns None for "not determined" -- NVML absent, a
#: container that forbids it, a trial too short for the poller to land a sample
#: -- and a None flattened into False reads as "this check FAILED", which is a
#: measurement of the card that nobody took. Closed for the reason TILE_SOURCES
#: is closed: a typo'd verdict is not a loud failure, it is a value no filter
#: matches, so those rows quietly leave every group-by that keys on it.
VERDICT_OK = "ok"
VERDICT_FAILED = "failed"
VERDICT_UNDETERMINED = "undetermined"
TIMING_VERDICTS: frozenset[str] = frozenset({
    VERDICT_OK, VERDICT_FAILED, VERDICT_UNDETERMINED})

#: The columns that speak it. LEVEL and DRIFT are `timing.clock_flags`; the
#: third is `timing.host_bound_verdict` INVERTED, so all three read the same way
#: round: "ok" is the state a row may be quoted in.
TIMING_VERDICT_COLUMNS: tuple[str, ...] = (
    "clock_level_ok", "clock_drift_ok", "host_bound_ok")


def verdict_word(ok: bool | None) -> str:
    """A `time_kernel` tri-state verdict as the word a CSV cell holds.

    None is `VERDICT_UNDETERMINED` and never `VERDICT_FAILED`: "the check could
    not be run" and "the check failed" are different facts about the run, and
    only one of them is a reason to distrust the number beside it.
    """
    if ok is None:
        return VERDICT_UNDETERMINED
    return VERDICT_OK if ok else VERDICT_FAILED

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
    #: A CALL COUNT, and it stays one. Under the retired instrument it is the
    #: count that was REQUESTED; under `timing.time_kernel` it is the count the
    #: sustained-load warmup actually delivered. Either way it is calls, and
    #: `warmup_ms` below is the duration.
    warmup: int = 0
    iters: int = 0
    trials: int = 0
    #: WHICH TIMER produced the ms_* columns below (v5).
    #:
    #: `timing.TIMING_BASIS` on a row measured by the one instrument,
    #: `NO_INSTRUMENT` on a row the driver wrote without timing it at all, and
    #: `LEGACY_INSTRUMENT` on anything older -- but read it through
    #: `instrument_of` and never off the column, because a pre-v5 row has no
    #: column at all and the reader is what turns that absence into the name
    #: rather than into an empty string that reads like a missing field.
    #:
    #: The default is the empty string and NOTHING MAY SHIP IT. It is the state
    #: of a freshly constructed `Row` before any writer has touched it, and
    #: `instrument_of` refuses it precisely so that state cannot reach a CSV
    #: unnoticed; the driver's `prepare()` overwrites it on every row.
    instrument: str = ""
    #: Milliseconds of DELIVERED GPU time the warmup ran for (v5), 0.0 under the
    #: retired instrument, which warmed for a count and measured nothing.
    #:
    #: A COUNT IS THE WRONG UNIT and this column is the fix. A 1 ms kernel needs
    #: hundreds of calls before the clock governor responds; a 30 ms GEMM needs
    #: one. So a corpus warmed at a fixed `warmup=25` compared cells at
    #: different clock states, and nothing in a row said which.
    warmup_ms: float = 0.0

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
    # THE FIXED ROOF: the calibration's number at the clock the calibration's
    # GEMM ran at, the same on every row of a run. Kept with that meaning.
    achieved_peak_tflops: float = 0.0
    # Compute-side efficiency against the FIXED roof above. Not derivable from
    # the memory-side number, so both are carried. KNOWN BIAS, stated rather
    # than repaired in place: a cell whose under-load clock differs from the
    # reference ran at a different tensor-core issue rate than this roof
    # assumed, so this figure is off by `sm_clock_load_mhz /
    # reference_clock_mhz` (up to 1.31x on an H200 decode cell boosted to 1980
    # against a 1515 roof, toward the study's claim). The honest figure is
    # `pct_of_roof_at_cell_clock` below (v6); this one stays because a column
    # may not change meaning across a version boundary, and it is the number
    # to compare with a pre-v6 row's.
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

    # --- thermal / clock drift, THE RETIRED INSTRUMENT'S ------------------
    # A v5 row leaves the five QUANTITIES at their defaults instead of refilling
    # them with numbers that would mean something else, and that is deliberate
    # rather than an omission. These clocks were sampled at IDLE INSTANTS either
    # side of the cell -- both after a synchronise, with the GPU no longer
    # working -- so `clock_drift_pct` is the gap between two idle readings.
    # Putting the under-load samples into these same columns would give one
    # column two meanings across the version boundary, which is the drift
    # `_stamp_unrecorded` exists to stop. The under-load numbers are their own
    # columns, below.
    #
    # `throttled` IS THE EXCEPTION, because it is a VERDICT and not a reading.
    # On a pre-v5 row it fires on a >5% DROP between the two idle samples, which
    # detects whether the FIRST sample caught the idle boost rather than whether
    # the card throttled under load: on the published alpha-0558 arm it flagged
    # 91% of vLLM rows above T=4096 while flagged and unflagged replicates of
    # the same cell timed at ratio 0.998 with identical end clocks. On a v5 row
    # `driver._apply_kernel_timing` writes the instrument's answer to the same
    # question, because four consumers read this one column as "do not pool
    # this row", and a column that is False by construction turns all of them
    # into checks that cannot fail.
    #
    # SINCE 2026-09-09 THE ANSWER IS DRIFT ALONE. Neither side of LEVEL sets
    # it. The HIGH side never did: a cell that boosted above the roof's clock
    # is the mirror image of a throttle and on an H200 the normal state of a
    # memory-bound cell. The LOW side stopped, because the first H200 session's
    # 750 cells showed the under-load clock is set per tile by the kernel's own
    # power draw under the 700 W cap (BM=128/N=64 at 1395 MHz in every rep
    # against a calibration GEMM at 1485), so LOW named a tile, and excluding
    # on it removed the study's two primary tiles and nothing else.
    # `clock_level_side` is the record of where the cell sat; this is the
    # verdict a consumer excludes on.
    sm_clock_start_mhz: int = 0
    sm_clock_end_mhz: int = 0
    temp_start_c: int = 0
    temp_end_c: int = 0
    clock_drift_pct: float = 0.0
    throttled: bool = False

    # --- the instrument's own verdicts (v5) -------------------------------
    # What `timing.time_kernel` observed WHILE the trials ran. The clock is
    # polled from a background thread during the measurement, not sampled at an
    # idle instant beside it, and the three verdicts are the three ways a cell
    # can be untrustworthy without the timer noticing anything wrong:
    #   LEVEL   the card sat outside `timing.level_band` of the clock the roof
    #           was measured at (two-sided since 2026-09-03, edges snapped to
    #           the 15 MHz NVML grid since 2026-09-09; `clock_level_side` names
    #           the side). A RECORD, not an exclusion: either side means the
    #           fixed-roof fraction is delivered throughput at a different
    #           issue rate, and `pct_of_roof_at_cell_clock` is the other
    #           fraction, on the row beside it.
    #   DRIFT   first and last under-load samples disagree by more than
    #           `timing.DRIFT_FRACTION` in EITHER direction. THE exclusion: the
    #           trials were not at one operating point. A rise is a defect too,
    #           and `clock_drift_direction` names which it was.
    #   HOST    the GPU's queue had drained before the host finished enqueueing,
    #           so the intervals carry host time and ms_* bound the kernel from
    #           ABOVE. Stored INVERTED (`host_bound_ok`) so all three columns
    #           read the same way round and one filter covers them.
    # All three take a word from TIMING_VERDICTS; read them with
    # `timing_verdict`, which refuses a pre-v5 row rather than calling its
    # silence a pass.
    sm_clock_load_mhz: float = 0.0
    clock_level_ok: str = VERDICT_UNDETERMINED
    clock_drift_ok: str = VERDICT_UNDETERMINED
    host_bound_ok: str = VERDICT_UNDETERMINED
    #: Usable under-load clock samples the median above was taken from. Below
    #: `timing.CLOCK_SAMPLE_FLOOR` there is no median and LEVEL is undetermined.
    clock_samples: int = 0
    #: Which reader polled: nvml | injected | none.
    clock_source: str = ""
    #: Why a clock column is missing or a flag undetermined, in words.
    clock_note: str = ""
    #: Median per-trial host wall of the enqueue loop. Divided by `iters` it is
    #: the host's per-call cost, the number to hold beside ms_p50 when
    #: `host_bound_ok` is "failed".
    host_enqueue_ms: float = 0.0
    host_note: str = ""

    # --- the reference LEVEL was scored against, and the roof per row (v6) --
    #: The SM clock `clock_level_ok` compared `sm_clock_load_mhz` against, in
    #: MHz, 0.0 when none was resolved. ON THE ROW so the verdict can be
    #: re-derived from the CSV alone: the yaml it came from is overwritten by
    #: every recalibration, and `CalibrationStamp` documents an arm that
    #: shipped with a ruler it never used.
    reference_clock_mhz: float = 0.0
    #: Where that number came from, `roofline.ReferenceClock.source`: the
    #: file, the field, the GRADE, and which GEMM. Read it before trusting the
    #: verdict: a source that says DISOWNED was the post-hoc idle scalar (30%
    #: spread across one card's calibrations) and LEVEL against it is
    #: provisional. PER DTYPE FAMILY: an fp8 row's reference is the fp8
    #: GEMM's clock (`detail.fp8_gemm_clock*`), every other row's the bf16
    #: GEMM's, because those are two roofs measured at two clocks.
    reference_clock_source: str = ""
    #: Which way a LEVEL failure went: "low" (a tile drawing more power than
    #: the ruler's GEMM under the same cap, so it holds a lower clock), "high"
    #: (a memory-shaped cell that boosted, whose fixed-roof fraction is
    #: inflated by the ratio), "" when level or undetermined. NEITHER SETS
    #: `throttled` and neither excludes the row: the 2026-09-09 session found
    #: the LOW side is a tile's steady state, not a thermal event, and a
    #: consumer that dropped it dropped BM=128/N=64 and BM=64/G=1 entirely.
    clock_level_side: str = ""
    #: Smallest per-trial GPU backlog when the host finished enqueueing, in
    #: iterations of the trial's own per-iteration wall. The host-bound verdict
    #: is `< timing.HOST_BOUND_BACKLOG_ITERS` on this; the ratio shows how far
    #: from the boundary the row sat. 0.0 on an untimed or retired-seam row.
    host_backlog_iters: float = 0.0
    #: The compute roof AT THE CLOCK THIS CELL RAN: `achieved_peak_tflops *
    #: sm_clock_load_mhz / reference_clock_mhz`, `roofline.cell_clock_roof`,
    #: with the reference of the row's dtype family. Written only when that
    #: reference is an UNDER-LOAD median; 0.0 with the reason in `roof_note`
    #: otherwise. 0.0 means "not scored", never a roof of zero: read it
    #: through `has_cell_clock_roof`. Written by two mirrors, `driver.
    #: _apply_cost` and `recompute.ceiling_columns`, through one rule.
    roof_at_cell_clock_tflops: float = 0.0
    #: `tflops` as a percentage of that roof. The compute-side efficiency to
    #: quote from a v6 row; `pct_of_achieved_tflops` is the fixed-roof figure
    #: with the bias its own comment states.
    pct_of_roof_at_cell_clock: float = 0.0
    #: WHICH OF THE TWO FRACTIONS A GATE READS, and why the other one is 0.0
    #: when it is. On a scored row this is `roofline.ROOF_NOTE_SCORED`, which
    #: names the fixed-roof fraction as the compute-bound gate input (every
    #: cell and the calibration GEMM ran under one power cap, so the fixed roof
    #: is the fair delivered-throughput comparison) and the own-clock fraction
    #: as issue efficiency. On an unscored row it is the refusal: no under-load
    #: clock on the row, no reference, or a reference whose grade is not
    #: under-load (the committed calibrations' idle scalar).
    roof_note: str = ""

    # --- the under-load clock trace, power, and the settle (v7) -----------
    #: First and last UNDER-LOAD samples, MHz, the pair `clock_drift_ok` is
    #: computed from. NOT `sm_clock_start_mhz`/`sm_clock_end_mhz` above, which
    #: are the retired seam's idle instants on 100,144 published rows and keep
    #: that meaning; these are new columns because one column may not mean two
    #: things either side of a version boundary.
    sm_clock_load_first_mhz: float = 0.0
    sm_clock_load_last_mhz: float = 0.0
    #: Every usable under-load sample, in order, space separated, MHz. The
    #: 2026-09-09 session flagged 135 cells as DRIFT and kept the samples for
    #: four of them, so the question the flag exists to raise (a governor
    #: settling after a workload change, or a card hunting between two states?)
    #: was unanswerable for 131 of them. Empty on a row nothing polled.
    clock_samples_mhz: str = ""
    #: Which way it moved when DRIFT failed: `timing.DRIFT_UP`,
    #: `timing.DRIFT_DOWN`, `timing.DRIFT_OSCILLATING`, "" otherwise.
    clock_drift_direction: str = ""
    #: Median board power over the under-load samples, W, 0.0 when unread. Read
    #: at the SAME NVML call as the clock, so a cell below the LEVEL band can
    #: be told apart as a hungry tile at the board cap (the session's BM=128/
    #: N=64 tiles, against a calibration GEMM holding 1485 MHz at 691 W) from a
    #: card in trouble. No card draws zero, so 0.0 is unambiguous.
    power_w: float = 0.0
    #: What the settle loop added after `warmup_ms`: delivered GPU time and
    #: calls spent waiting for two consecutive clock reads to agree within one
    #: 15 MHz step (`timing.warm_until`). 0.0/0 with
    #: `warmup_clock_settled = undetermined` means no clock reader was
    #: available to settle against, which is not "it settled at once".
    warmup_settle_ms: float = 0.0
    warmup_settle_calls: int = 0
    #: "ok" when the two reads agreed, "failed" when the
    #: `timing.SETTLE_CAP_MULTIPLE` cap stopped the loop first, "undetermined"
    #: when there was no reader. A word from TIMING_VERDICTS, but NOT one of
    #: `TIMING_VERDICT_COLUMNS`: it is a fact about the warmup, not one of the
    #: three under-load checks a gate filters on.
    warmup_clock_settled: str = VERDICT_UNDETERMINED

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


class ColumnUnrecorded(LookupError):
    """This row predates the column being asked for, and nothing stands in.

    The base of the two named refusals below, so a reader that does not care
    WHICH version's hole it hit can catch one thing. Both subclasses exist
    because the two holes are answered differently: a missing tile is filtered
    with `has_tile_config`, a missing instrument with `has_kernel_timing`, and
    an exception that could not say which would send a caller to the wrong
    predicate.
    """


class TileConfigUnrecorded(ColumnUnrecorded):
    """This row does not say which tile ran, and nothing may stand in for it.

    Raised rather than returning 0 because the whole reason the tile_* columns
    exist is that a plausible-looking number with no measurement behind it is
    indistinguishable from a real one, and one such number steered an analysis
    for days.
    """


class TimingInstrumentUnrecorded(ColumnUnrecorded):
    """This row does not say how it was timed, and nothing may stand in for it.

    Raised on a pre-v5 row asked for one of the three under-load checks, which
    did not exist when it was written. The alternative -- returning "ok", or an
    empty string a filter reads as not-failed -- would let a corpus that never
    looked at its clocks under load pass the gate that exists to catch exactly
    that, which is the shape of every defect v5 was cut for.
    """


class CellClockRoofUnrecorded(ColumnUnrecorded):
    """This row predates the per-row roof and the reference it was scored against.

    Raised on a pre-v6 row asked for `roof_at_cell_clock_tflops`,
    `reference_clock_mhz` or the other v6 columns. A v5 row DID carry a LEVEL
    verdict, so this is not the v5 hole under another name: the row was timed
    on the instrument and its verdict is readable, but what it was level
    AGAINST is not on it, and its fixed-roof fraction may carry the boost bias
    v6 exists to name. Split with `has_cell_clock_roof` before reading these.
    """


class LoadClockTraceUnrecorded(ColumnUnrecorded):
    """This row predates the under-load clock trace, the power and the settle.

    Raised on a pre-v7 row asked for `clock_samples_mhz`,
    `sm_clock_load_first_mhz`, `power_w` or the other v7 columns. Such a row
    DID carry a DRIFT verdict computed from a first and a last sample; what it
    does not carry is the samples themselves, which is why 131 of the
    2026-09-09 session's 135 DRIFT verdicts could not be examined after the
    fact. Split with `has_load_clock_trace` before reading these.
    """


#: Which refusal a stamped column raises, and which predicate the message sends
#: the caller to, by the version that added the column. A v5 column asked of a
#: v4 row is not a tile problem and must not arrive as one: the two holes have
#: different predicates and different fixes.
#:
#: THE DEFAULT IS THE TILE ONE, and that is compatibility rather than taxonomy.
#: `TileConfigUnrecorded` was the only sentinel refusal for two versions, every
#: `except` clause in the tree names it, and a column outside the map is one
#: that predates the map. It is now a subclass of `ColumnUnrecorded`, so a
#: caller that wants "any version hole" can catch the base instead.
_UNRECORDED_ERRORS: dict[int, type[ColumnUnrecorded]] = {
    4: TileConfigUnrecorded,
    5: TimingInstrumentUnrecorded,
    6: CellClockRoofUnrecorded,
    7: LoadClockTraceUnrecorded,
}
_PREDICATE_FOR: dict[int, str] = {4: "has_tile_config", 5: "has_kernel_timing",
                                  6: "has_cell_clock_roof",
                                  7: "has_load_clock_trace"}


def _added_in(key: str) -> int:
    """The schema version that introduced `key`, or 0 for one older than the map."""
    for version, names in COLUMNS_ADDED_IN.items():
        if key in names:
            return version
    return 0


def _reject_sentinel(key: str, value) -> None:
    """A column stamped UNRECORDED is never a number, never a bool, never a
    default. Checked in every reader, so no path can quietly coerce it."""
    if value != UNRECORDED:
        return
    version = _added_in(key)
    arrived = f" (it arrived in v{version})" if version else ""
    predicate = _PREDICATE_FOR.get(version, "has_tile_config")
    raise _UNRECORDED_ERRORS.get(version, TileConfigUnrecorded)(
        f"{key!r} is not recorded on this row: it comes from a CSV written "
        f"under an older schema, which had no such column{arrived}. Filter "
        f"these rows out with {predicate}(row) instead of reading them.")


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


def instrument_of(row: dict) -> str:
    """Which timing apparatus produced this row's ms_* columns. Never a default.

    THE ONE READER FOR THE VERSION BOUNDARY. Three inputs, three answers:

      - no `instrument` key at all (a raw `csv.DictReader` over a pre-v5 file)
        or the UNRECORDED sentinel (`read_csv` stamped it): `LEGACY_INSTRUMENT`.
        The absence IS the answer, and naming it is the point -- a row from
        before the fix is not a row of unknown provenance, it is a row of known
        bad provenance;
      - a non-empty string: that string, whatever it is. A future
        `TIMING_BASIS` bump has to read back as itself here, or the reader would
        quietly relabel an instrument it has not heard of. `NO_INSTRUMENT` is
        one such string and reads back as itself: the driver stamps it on every
        row it emits without timing, so "nothing measured this" is an answer
        this function gives rather than an exception it raises;
      - a v5 row carrying an EMPTY instrument: refused. Not a driver row --
        `prepare()` stamps `NO_INSTRUMENT` before any path can emit -- so an
        empty one means something else wrote the file, and what that something
        did to the ms_* columns is exactly what must not be guessed at.

    Callers that only want the ms_* numbers should filter on `ms_p50 > 0`
    first: a cell that failed the oracle is written with its timing zeroed,
    keeps the name of the instrument that took the discarded measurement, and
    is not a number anyone may quote.
    """
    value = row.get("instrument")
    if value is None or value == UNRECORDED:
        return LEGACY_INSTRUMENT
    text = str(value).strip()
    if not text:
        raise TimingInstrumentUnrecorded(
            "this row carries the v5 `instrument` column and it is empty, so "
            "nothing timed it and nothing may assume what did. A row written "
            "by the driver always names its timer; an empty one is a bug in "
            "whatever produced the file, not a row to guess about.")
    return text


def has_kernel_timing(row: dict) -> bool:
    """Was this row measured on the instrument, rather than on what preceded it?

    The predicate to split a pool on BEFORE any gate that reads a v5 column, so
    an analysis never has to catch `TimingInstrumentUnrecorded` row by row. The
    mirror of `has_tile_config`, and it exists for the same reason: two
    different apparatus in one pool is a fact to branch on, not a hole to fill.

    FALSE FOR TWO DIFFERENT ROWS, on purpose, because the question it asks has
    one answer for both: a pre-v5 row has no verdict columns at all and an
    untimed v5 row has them at their defaults. `timing_verdict` refuses BOTH,
    the first because the column is absent and the second because the row's
    instrument is `NO_INSTRUMENT`, so this predicate and that reader agree on
    exactly one set of rows. They did not until 2026-09-02: the untimed v5 row
    read back "undetermined" and was admitted by every gate that branched on the
    word. The two are told apart by `instrument_of`, which names them, and a
    caller that needs the distinction must ask for it by name rather than read
    it out of a bool that was never carrying it.
    """
    return instrument_of(row) not in NO_KERNEL_TIMING


def has_cell_clock_roof(row: dict) -> bool:
    """Was this row's compute roof rescaled to the clock the cell ran at?

    The predicate to split a pool on BEFORE reading `roof_at_cell_clock_tflops`
    or `pct_of_roof_at_cell_clock`. False for every pre-v6 row (the columns did
    not exist), false for a v6 row the driver could not score: no under-load
    clock on the row (the retired seam, a container that forbids NVML), no
    reference, or a reference the calibration disowns (the idle scalar both
    committed calibrations carry). `roof_note` says which. Keyed on the roof
    being positive, because a roof of zero does not exist and 0.0 is the
    driver's "not scored".
    """
    value = row.get("roof_at_cell_clock_tflops")
    if value in (None, "", UNRECORDED):
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def has_load_clock_trace(row: dict) -> bool:
    """Does this row carry the under-load samples the DRIFT verdict came from?

    The predicate to split a pool on BEFORE reading `clock_samples_mhz`,
    `sm_clock_load_first_mhz`, `clock_drift_direction` or `power_w`. False for
    every pre-v7 row (the columns did not exist) and false for a v7 row nothing
    polled: a container that forbids NVML, or a trial too short for a sample to
    land. Keyed on the sample LIST being non-empty, because that is the column
    the others are re-derivable from, and an empty one is "nothing was read"
    rather than "the clock was zero".
    """
    value = row.get("clock_samples_mhz")
    if value in (None, "", UNRECORDED):
        return False
    return True


def timing_verdict(row: dict, key: str) -> str:
    """One of the three v5 under-load verdicts, or raise. Never a usable default.

    Raises `TimingInstrumentUnrecorded` on a pre-v5 row, where the check did not
    exist, and on a v5 row whose column is empty. Raises `ValueError` on a word
    outside `TIMING_VERDICTS`, because a verdict no filter matches removes the
    row from every group-by that keys on it while looking like a pass.

    "undetermined" is RETURNED, not raised: the check ran and could not decide
    (no NVML, a trial too short for the poller), which is a real state of the
    measurement and one a caller may legitimately choose to keep or drop. What
    it must never be is silently folded into "ok".

    AN UNTIMED ROW IS REFUSED, and until 2026-09-02 it was not. The driver writes
    a row for every cell it declines or fails, stamps `NO_INSTRUMENT` on it and
    leaves these three columns at their `Row` defaults, which are the WORD
    "undetermined". So a row nothing measured read back as "the check ran and
    could not decide", and `alpha_refit.clock_gate` read three of those and
    returned ADMIT. Nothing broke only because both of its callers happen to drop
    `ms_p50 <= 0` a few lines earlier -- an incidental filter standing in for the
    intended one, which is the exact accident this module's own header complains
    about. `has_kernel_timing` says the two apparatus have to be split before any
    v5 column is read; this is that sentence enforced rather than asserted.
    """
    _schema_key(key)
    if key not in TIMING_VERDICT_COLUMNS:
        raise ValueError(
            f"{key!r} is not one of the v5 timing verdicts; read it with "
            f"row_float, row_bool or row.get. timing_verdict covers "
            f"{', '.join(TIMING_VERDICT_COLUMNS)}.")
    if instrument_of(row) == NO_INSTRUMENT:
        raise TimingInstrumentUnrecorded(
            f"{key!r} is {row.get(key)!r} on a row whose instrument is "
            f"{NO_INSTRUMENT!r}: nothing timed this cell, so no under-load "
            f"check was made and the default word is not a verdict. Split the "
            f"pool with has_kernel_timing(row) first, which is False here.")
    value = row.get(key)
    _reject_sentinel(key, value)
    if value is None or value == "":
        raise TimingInstrumentUnrecorded(
            f"{key!r} is absent on this row: it predates schema v5, whose "
            f"instrument is the first one to read the clock while the trials "
            f"run. Split the pool with has_kernel_timing(row) instead of "
            f"reading a check that was never made.")
    _reject_sentinel(key, value)
    word = str(value)
    if word not in TIMING_VERDICTS:
        raise ValueError(
            f"{key!r} is {word!r}, which is not one of "
            f"{sorted(TIMING_VERDICTS)}. A verdict outside the set matches no "
            f"filter and silently leaves every group-by that keys on it.")
    return word


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
