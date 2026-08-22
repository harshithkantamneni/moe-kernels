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
