import json

import pytest

from moe.bench import schema as SC


def make_row(**kw):
    base = dict(model="toy", num_tokens=32, dtype="bf16", routing_kind="uniform",
                seed=0, pipeline="a -> b", impl="a", ms_p50=1.25,
                correctness_passed=True)
    base.update(kw)
    return SC.Row(**base)


def test_csv_round_trip(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row())
        w.write(make_row(num_tokens=64))
    rows = SC.read_csv(path)
    assert len(rows) == 2
    assert rows[0]["model"] == "toy"
    assert float(rows[1]["num_tokens"]) == 64


def test_header_written_once_when_appending(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row())
    with SC.CsvWriter(path) as w:
        w.write(make_row(num_tokens=64))
    assert path.read_text().count("schema_version") == 1


def test_read_refuses_a_foreign_schema_version(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(schema_version=99))
    with pytest.raises(ValueError, match="schema_version 99"):
        SC.read_csv(path)


def test_read_refuses_a_version_older_than_the_readable_window(tmp_path):
    """READABLE_VERSIONS widened to keep the published v3 arms loadable, not to
    accept anything. v2 renamed achieved_bf16_tflops, so its columns do not mean
    what these names mean and no stamping can repair that."""
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(schema_version=2))
    with pytest.raises(ValueError, match="schema_version 2"):
        SC.read_csv(path)


def test_cell_key_ignores_timing_results():
    a = make_row(ms_p50=1.0)
    b = make_row(ms_p50=999.0)
    assert SC.cell_key(a) == SC.cell_key(b)


def test_cell_key_separates_timing_modes():
    assert SC.cell_key(make_row(l2_flush=True)) != SC.cell_key(make_row(l2_flush=False))
    assert SC.cell_key(make_row(cuda_graph=True)) != SC.cell_key(make_row(cuda_graph=False))


def test_manifest_resume(tmp_path):
    path = tmp_path / "m.jsonl"
    m = SC.Manifest(path)
    m.record("cell-a")
    m.close()

    m2 = SC.Manifest(path)
    assert "cell-a" in m2
    assert "cell-b" not in m2
    m2.close()


def test_manifest_survives_a_torn_final_line(tmp_path):
    """A pod killed mid-write leaves a partial JSON line. Resume must still work."""
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"key": "cell-a"}) + "\n" + '{"key": "cell-')
    m = SC.Manifest(path)
    assert "cell-a" in m
    m.close()


def test_merge_csvs_from_separate_envs(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    with SC.CsvWriter(a) as w:
        w.write(make_row(env_name="vllm", impl="vllm_fused_moe"))
    with SC.CsvWriter(b) as w:
        w.write(make_row(env_name="sglang", impl="sglang_fused_moe"))
    out = tmp_path / "all.csv"
    assert SC.merge_csvs([a, b], out) == 2
    envs = {r["env_name"] for r in SC.read_csv(out)}
    assert envs == {"vllm", "sglang"}


def test_bools_survive_the_round_trip(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(correctness_passed=True, throttled=False, l2_flush=True))
    out = tmp_path / "m.csv"
    SC.merge_csvs([path], out)
    r = SC.read_csv(out)[0]
    assert r["correctness_passed"] == "True"
    assert r["throttled"] == "False"


def test_every_column_is_declared():
    assert "correctness_passed" in SC.COLUMNS
    assert "l2_flush" in SC.COLUMNS
    assert "arith_intensity_compulsory" in SC.COLUMNS
    assert "compulsory_gbps" in SC.COLUMNS
    assert "capture_status" in SC.COLUMNS
    assert "flush_mb" in SC.COLUMNS
    assert len(SC.COLUMNS) == len(set(SC.COLUMNS))


def test_appending_under_a_foreign_header_is_refused(tmp_path):
    """Resuming with --run-id reopens the CSV in append mode. Writing new-order
    rows beneath an old header would misalign every column from there on."""
    path = tmp_path / "r.csv"
    path.write_text("model,num_tokens,ms_p50\ntoy,32,1.0\n")
    with pytest.raises(ValueError, match="different schema"):
        SC.CsvWriter(path)


def test_appending_under_a_matching_header_is_fine(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row())
    with SC.CsvWriter(path) as w:
        w.write(make_row(num_tokens=64))
    assert len(SC.read_csv(path)) == 2


def test_schema_version_tracks_the_column_set():
    """A reminder in code: the version must move whenever COLUMNS does."""
    assert SC.SCHEMA_VERSION == 5
    assert "pct_of_achieved_bw" not in SC.COLUMNS
    assert "pct_of_achieved_tflops" in SC.COLUMNS
    assert "achieved_peak_tflops" in SC.COLUMNS
    # v3: which STREAM pattern produced achieved_bw_gbps. Two CSVs run with a
    # different --ceiling are otherwise silently incomparable.
    assert "bw_ceiling_pattern" in SC.COLUMNS
    # v4: the tile that actually ran, and the architecture that ran it. Before
    # these the only tile columns were hypothetical efficiencies at ASSUMED
    # block sizes, so no row could contradict a wrong BLOCK_SIZE_M.
    for name in SC.COLUMNS_ADDED_IN[4]:
        assert name in SC.COLUMNS, name
    assert "tile_block_m" in SC.COLUMNS
    assert "tile_config_source" in SC.COLUMNS
    assert "sm_capability" in SC.COLUMNS
    # v5: which timer produced ms_*, and what the card was doing while it did.
    # Before these, a row measured on `time_eager` with a COUNT of warmup calls
    # and two idle-instant clock samples was indistinguishable from a row
    # measured on `time_kernel`, and the study's headline alpha came from the
    # first while a ladder script's alpha came from the second.
    for name in SC.COLUMNS_ADDED_IN[5]:
        assert name in SC.COLUMNS, name
    assert "instrument" in SC.COLUMNS
    assert "warmup_ms" in SC.COLUMNS
    assert set(SC.TIMING_VERDICT_COLUMNS) <= set(SC.COLUMNS)
    # And `warmup` survives beside `warmup_ms` rather than changing meaning:
    # every published row records a call count under that name.
    assert "warmup" in SC.COLUMNS


# --- the instrument boundary ------------------------------------------------

def v4_row_dict(tmp_path, **kw):
    """A row on disk at v4, i.e. one of the 100,144 published ones in shape.

    Written through a header that omits every v5 column, because that is what a
    published arm's file really looks like; `read_csv` then stamps the hole.
    """
    import csv
    path = tmp_path / "v4.csv"
    columns = [c for c in SC.COLUMNS if c not in SC.COLUMNS_ADDED_IN[5]]
    values = dict(schema_version=4, model="mixtral-8x7b", ms_p50=1.25,
                  correctness_passed=True, throttled=False)
    values.update(kw)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerow({c: values.get(c, "") for c in columns})
    return SC.read_csv(path)[0]


def test_a_published_row_still_loads_and_says_which_instrument_made_it(tmp_path):
    """THE CONSTRAINT THE VERSION HAD TO SATISFY. Ten published arms and 100,144
    rows were measured under v3/v4 and cannot be re-taken; a gate that refused
    them would retire the DATA rather than the code. They load, and they answer
    the new question honestly: the absence of the column IS the answer, and the
    answer has a name."""
    row = v4_row_dict(tmp_path)
    assert row["instrument"] == SC.UNRECORDED
    assert SC.instrument_of(row) == SC.LEGACY_INSTRUMENT
    assert SC.has_kernel_timing(row) is False


def test_a_row_from_before_v5_cannot_answer_a_check_it_never_made(tmp_path):
    """The FAIL branch of the verdict reader. Returning "ok", or an empty string
    a filter reads as not-failed, would let a corpus that never looked at its
    clocks under load pass the very gate v5 was cut for."""
    row = v4_row_dict(tmp_path)
    for name in SC.TIMING_VERDICT_COLUMNS:
        with pytest.raises(SC.TimingInstrumentUnrecorded, match="has_kernel_timing"):
            SC.timing_verdict(row, name)
    # The numeric v5 columns refuse through the sentinel on the same row, and
    # they refuse as a TIMING hole rather than as a tile one: the two send a
    # caller to different predicates, so the exception has to say which.
    with pytest.raises(SC.TimingInstrumentUnrecorded, match="has_kernel_timing"):
        SC.row_float(row, "sm_clock_load_mhz")
    # A v3 row, whose hole is the TILE one, still refuses under its own name and
    # sends the caller to its own predicate.
    v3 = dict(row, schema_version="3")
    SC._stamp_unrecorded(v3, 3)
    with pytest.raises(SC.TileConfigUnrecorded, match="has_tile_config"):
        SC.tile_field(v3, "tile_block_m")
    assert issubclass(SC.TimingInstrumentUnrecorded, SC.ColumnUnrecorded)
    assert issubclass(SC.TileConfigUnrecorded, SC.ColumnUnrecorded)


def test_a_v5_row_reads_its_verdicts_back_as_the_words_it_wrote(tmp_path):
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(instrument="queue-deep/l2-flush/clock-under-load/v2",
                         clock_level_ok=SC.VERDICT_OK,
                         clock_drift_ok=SC.VERDICT_FAILED,
                         host_bound_ok=SC.VERDICT_UNDETERMINED))
    row = SC.read_csv(path)[0]
    assert SC.has_kernel_timing(row)
    assert SC.timing_verdict(row, "clock_level_ok") == "ok"
    assert SC.timing_verdict(row, "clock_drift_ok") == "failed"
    assert SC.timing_verdict(row, "host_bound_ok") == "undetermined"


def test_a_verdict_outside_the_closed_set_is_refused_not_matched(tmp_path):
    """A typo'd verdict is not a loud failure; it is a value no filter matches,
    so the row leaves every group-by that keys on it while looking like a pass.
    Same reason TILE_SOURCES is closed."""
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(instrument="x", clock_level_ok="OK"))
    with pytest.raises(ValueError, match="not one of"):
        SC.timing_verdict(SC.read_csv(path)[0], "clock_level_ok")
    with pytest.raises(ValueError, match="not one of the v5 timing verdicts"):
        SC.timing_verdict({"throttled": "False"}, "throttled")


def test_a_v5_row_that_names_no_instrument_is_refused():
    """Not reachable through the driver, whose `prepare()` stamps EVERY row it
    emits, timed or not. An empty column therefore means something other than
    the driver wrote the file, and what that something did to the ms_* columns
    is exactly what must not be guessed at."""
    with pytest.raises(SC.TimingInstrumentUnrecorded, match="empty"):
        SC.instrument_of({"instrument": "  ", "ms_p50": "1.0"})
    with pytest.raises(SC.TimingInstrumentUnrecorded, match="empty"):
        SC.has_kernel_timing({"instrument": "", "ms_p50": "0.0"})


def test_an_untimed_row_gets_a_name_rather_than_a_refusal(tmp_path):
    """THE DEFECT THE FIRST CUT OF THIS BOUNDARY INTRODUCED. `has_kernel_timing`
    is documented as the predicate to split a pool on BEFORE any gate reads a v5
    column, and it raised on four kinds of row the driver writes routinely: a
    cell that failed the oracle, a graph mode skipped by cost policy, a span
    that could not be captured, and a timer that raised. Nothing broke only
    because no v5 data existed yet."""
    path = tmp_path / "r.csv"
    with SC.CsvWriter(path) as w:
        w.write(make_row(instrument=SC.NO_INSTRUMENT,
                         capture_status="not_capturable", ms_p50=0.0))
    row = SC.read_csv(path)[0]
    assert SC.instrument_of(row) == SC.NO_INSTRUMENT
    assert SC.has_kernel_timing(row) is False


def test_untimed_and_legacy_are_both_unreadable_but_are_not_the_same_row():
    """`has_kernel_timing` answers one question -- may a v5 column be read off
    this row -- and the answer is no for both. The distinction that does matter
    is kept by `instrument_of`: a legacy row HAS numbers, measured the retired
    way; an untimed row has none at all."""
    legacy = {"instrument": SC.UNRECORDED}
    untimed = {"instrument": SC.NO_INSTRUMENT}
    assert SC.has_kernel_timing(legacy) is False
    assert SC.has_kernel_timing(untimed) is False
    assert SC.instrument_of(legacy) == SC.LEGACY_INSTRUMENT
    assert SC.instrument_of(untimed) == SC.NO_INSTRUMENT
    assert SC.instrument_of(legacy) != SC.instrument_of(untimed)
    assert SC.NO_KERNEL_TIMING == {SC.LEGACY_INSTRUMENT, SC.NO_INSTRUMENT}


def test_verdict_word_never_turns_an_unknown_into_a_failure():
    """None is "the check could not be run" and False is "the check failed".
    A None flattened to False reads as a measurement of the card that nobody
    took."""
    assert SC.verdict_word(True) == SC.VERDICT_OK
    assert SC.verdict_word(False) == SC.VERDICT_FAILED
    assert SC.verdict_word(None) == SC.VERDICT_UNDETERMINED
    assert SC.TIMING_VERDICTS == {"ok", "failed", "undetermined"}


def test_a_retired_column_name_raises_instead_of_reading_as_zero():
    """The bug this prevents, verbatim: an analysis of the first published
    sweep asked for `pct_of_achieved_bw`, a column dropped in schema v2 as
    redundant with implied_traffic_ratio. row_float returned 0.0 for all 840
    rows and the table printed `0%` down the page, which reads as a finding
    rather than a typo."""
    row = {"tflops": "123.0"}
    with pytest.raises(KeyError, match="pct_of_achieved_bw"):
        SC.row_float(row, "pct_of_achieved_bw")
    with pytest.raises(KeyError, match="not a column"):
        SC.row_bool(row, "l2_flushed")          # real column is l2_flush


def test_the_rejection_suggests_the_column_the_caller_probably_meant():
    with pytest.raises(KeyError, match="did you mean.*l2_flush"):
        SC.row_float({}, "l2_flushh")


def test_a_valid_column_missing_from_an_older_csv_still_defaults():
    """Schema evolution must stay backward compatible: v1 CSVs have no
    load_tile_eff_bm128 column, and reading one is a default, not an error."""
    assert SC.row_float({"impl": "x"}, "load_tile_eff_bm128") == 0.0
    assert SC.row_float({"impl": "x"}, "load_tile_eff_bm128", default=-1.0) == -1.0
    assert SC.row_bool({"impl": "x"}, "throttled") is False


def test_merge_replaces_an_existing_merge_rather_than_appending(tmp_path):
    """merged.csv is DERIVED, so re-merging must rebuild it, not grow it.

    CsvWriter is append-only on purpose: a run CSV must survive a killed pod,
    and resume appends to it. merged.csv is the opposite kind of file. run_all.sh
    merges into `results/merged.csv`, and results/ outlives a session, so an
    appending merge silently accumulates every sweep ever run on the pod. That is
    where the 872 foreign rows in the 2026-08-26 published arm came from,
    including all 840 rows of the August-22 sweep, which were measured against a
    different calibration and so make every efficiency column in that file
    unreadable without knowing which run a row belongs to.
    """
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    with SC.CsvWriter(a) as w:
        w.write(make_row(run_id="first", env_name="base"))
    with SC.CsvWriter(b) as w:
        w.write(make_row(run_id="second", env_name="vllm"))

    out = tmp_path / "merged.csv"
    assert SC.merge_csvs([a], out) == 1
    assert SC.merge_csvs([b], out) == 1, "the return value counts rows merged"

    rows = SC.read_csv(out)
    assert len(rows) == 1, f"the earlier merge was appended to, not replaced: {rows}"
    assert {r["run_id"] for r in rows} == {"second"}


def test_merging_the_same_inputs_twice_is_idempotent(tmp_path):
    """Re-publishing an arm must not double it."""
    a = tmp_path / "a.csv"
    with SC.CsvWriter(a) as w:
        w.write(make_row(run_id="x"))
        w.write(make_row(run_id="x", num_tokens=64))
    out = tmp_path / "merged.csv"
    SC.merge_csvs([a], out)
    SC.merge_csvs([a], out)
    assert len(SC.read_csv(out)) == 2
