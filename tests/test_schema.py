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
    assert SC.SCHEMA_VERSION == 3
    assert "pct_of_achieved_bw" not in SC.COLUMNS
    assert "pct_of_achieved_tflops" in SC.COLUMNS
    assert "achieved_peak_tflops" in SC.COLUMNS
    # v3: which STREAM pattern produced achieved_bw_gbps. Two CSVs run with a
    # different --ceiling are otherwise silently incomparable.
    assert "bw_ceiling_pattern" in SC.COLUMNS


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
