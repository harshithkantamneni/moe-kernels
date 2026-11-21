"""The tile observer can report a tuned config that does not exist. This catches it.

THE BUG THIS PREVENTS. `_framework_config.tile_meta_from_capture` writes
`tile_config_source = "vllm_tuned"` the moment `get_moe_configs` returns any
dict, and records nothing about WHICH file that dict came from. So the column
is an assertion with no evidence behind it: if the recorder is ever installed on
the wrong namespace, if a lookup from an earlier cell leaks into a later
capture, if vLLM changes what the lookup returns, or if the harness asks for a
shape it did not think it was asking for, the row says "tuned" and no reader can
contradict it. That is the same shape of error as the `BLOCK_SIZE_M = 128` that
sat unchallenged in an analysis for days, and it is worse, because this time the
wrong value is in a column whose whole purpose is to be believed.

AND THE EXISTING TESTS CANNOT CATCH IT. `tests/test_tile_provenance.py` has only
ever run in the base venv, which has no vLLM installed, so every one of its
recorder assertions is against a hand-built stub module: it proves the wrapper
records what a stub returns, never that the stub resembles vLLM. Nothing in the
suite compares a recorded source against the world.

WHAT THIS FILE DOES INSTEAD. vLLM's tuned lookup is a FILE lookup, and the file
set is fixed at a tag. `tests/test_deployment_shapes.py` carries all 327 names
shipped at v0.27.1, so from `(E, N, dtype, gpu_name)` -- every one of which a row
already records -- the exact filename vLLM would open is computable, and whether
it exists is decidable. A row claiming `vllm_tuned` for a shape with no shipped
name is the observer lying, and it fails here loudly.

The check runs over synthetic rows and over a real arm's CSVs, through the same
`schema.read_csv` an analysis uses. As a command:

    .venv/bin/python tests/test_tile_source_crosscheck.py results/published/*/run_*.csv
    .venv/bin/python tests/test_tile_source_crosscheck.py /workspace/results/run_*.csv

Today it reports every published row as skipped, because all ten arms are schema
v3 and carry the UNRECORDED sentinel. That is the correct answer and it is why
the command matters: it is the first thing to run against tomorrow's v4 arm,
before any number is read out of it.

TWO THINGS THIS CANNOT DECIDE, both of which would make a `vllm_tuned` honest on
a shape with no shipped file, and both of which are checked for where possible:

  - `VLLM_TUNED_CONFIG_FOLDER` is consulted BEFORE the shipped directory
    (fused_moe.py v0.27.1, `get_moe_configs`), so a user-supplied JSON can tune
    any shape at all. No sweep in this study sets it.
  - a vLLM that is not v0.27.1 ships a different tree. The version is not in the
    schema, so this is scoped by assertion rather than by data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Under pytest both of these are already on the path; run as the command in the
# docstring, neither is. Same `sys.path.insert` + `noqa: E402` shape the scripts
# in scripts/ use, and it is here so that the command and the test exercise the
# SAME code rather than the command being a second, untested entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `moe`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # for the sibling test

import pytest  # noqa: E402
from test_deployment_shapes import (  # noqa: E402
    SHIPPED,
    config_file_name,
    vllm_device_file_name,
)

from moe.baselines._framework_config import vllm_quant_spec  # noqa: E402
from moe.bench import schema as SC  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec  # noqa: E402

PUBLISHED = Path(__file__).resolve().parents[1] / "results" / "published"

#: The only two sources that make a claim about a FILE. Everything else in
#: schema.TILE_SOURCES describes a path that never opened one, and a check that
#: treated them as disagreements would fail on every honest SGLang row.
FILE_CLAIMING_SOURCES = frozenset({"vllm_tuned", "vllm_default"})

LIED = "LIED"
MISSED = "MISSED"
AGREES = "agrees"


@dataclass(frozen=True)
class Verdict:
    """One row, checked or skipped, with the reason either way.

    A skip carries its reason as text rather than being dropped, because a check
    that silently examines nothing looks exactly like a check that passes. The
    report counts skips by reason for that reason alone.
    """

    where: str
    status: str
    source: str = ""
    predicted_file: str = ""
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status in (LIED, MISSED)


@dataclass
class Report:
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def failures(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.failed]

    @property
    def checked(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.status in (AGREES, LIED, MISSED)]

    def skipped_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.verdicts:
            if v.status not in (AGREES, LIED, MISSED):
                out[v.status] = out.get(v.status, 0) + 1
        return out

    def summary(self) -> str:
        lines = [f"rows {len(self.verdicts)}  checked {len(self.checked)}  "
                 f"failures {len(self.failures)}"]
        for reason, n in sorted(self.skipped_by_reason().items()):
            lines.append(f"  skipped {n:>7}  {reason}")
        for v in self.failures:
            lines.append(f"  {v.status}  {v.where}  source={v.source} "
                         f"predicted={v.predicted_file}  {v.detail}")
        return "\n".join(lines)


def predicted_config_file(model: str, dtype: str, gpu_name: str) -> str:
    """The exact filename vLLM would try to open for this cell.

    Every input is read from the harness's own definitions rather than restated,
    which is the property that makes a disagreement mean something. `(E, N)`
    comes off `w2_shape` because `try_get_optimal_moe_config` does
    `E, _, N = w2_shape`; the dtype selector and block shape come from
    `vllm_quant_spec`, which is what the span actually passes. Restating either
    here would let this test agree with a wrong prediction.
    """
    cfg = MODEL_CONFIGS[model]
    quant = vllm_quant_spec(BenchSpec(cfg, num_tokens=1, dtype=dtype))
    if quant is not None and quant["kind"] == "int4_w4a16":
        # vLLM doubles N for int4 before building the name. No cell in this
        # study is int4, and a silent wrong N is exactly what this file exists
        # to stop, so it refuses rather than predicting.
        raise NotImplementedError("int4_w4a16 doubles N in the lookup; not handled")
    return config_file_name(
        model,
        dtype=None if quant is None else quant["kind"],
        block_shape=None if quant is None else quant["block_shape"],
        device_name=vllm_device_file_name(gpu_name),
    )


def check_row(row: dict, where: str = "") -> Verdict:
    """Does this row's `tile_config_source` agree with vLLM's shipped file tree?

    Reads the source through `schema.tile_field` rather than off the dict, so a
    v3 row and a v4 row that observed nothing stay distinguishable from each
    other and from a real answer. Both are skips, and they are skipped under
    different names because only one of them is fixable by re-running.
    """
    raw = row.get("tile_config_source")
    if raw in (None, ""):
        return Verdict(where, "no tile_config_source column")
    if raw == SC.UNRECORDED:
        return Verdict(where, "predates schema v4")
    source = SC.tile_field(row, "tile_config_source")
    if source not in FILE_CLAIMING_SOURCES:
        return Verdict(where, f"source {source!r} claims nothing about a file",
                       source=source)

    model = str(row.get("model", ""))
    if model not in MODEL_CONFIGS:
        return Verdict(where, f"model {model!r} is not in MODEL_CONFIGS",
                       source=source)
    gpu = str(row.get("gpu_name", ""))
    if not gpu:
        return Verdict(where, "no gpu_name, so no device selector", source=source)

    predicted = predicted_config_file(model, str(row.get("dtype", "")), gpu)
    ships = predicted in SHIPPED
    if source == "vllm_tuned" and not ships:
        return Verdict(
            where, LIED, source, predicted,
            "the row claims a TUNED config for a shape vLLM v0.27.1 ships no "
            "file for. Either the observer attributed another cell's lookup to "
            "this one, or VLLM_TUNED_CONFIG_FOLDER was set, or this is not "
            "v0.27.1. None of those is a tile this row may be read for.")
    if source == "vllm_default" and ships:
        return Verdict(
            where, MISSED, source, predicted,
            "a tuned file DOES ship for this shape, so the row should not have "
            "taken the fallback ladder. Either (E, N, dtype, device) is not "
            "what the harness thinks it asked for, or VLLM_BATCH_INVARIANT was "
            "set, which returns None from get_moe_configs before any lookup.")
    return Verdict(where, AGREES, source, predicted)


def crosscheck(rows, label: str = "") -> Report:
    """Every row, in order, with its index in the label so a failure is findable."""
    report = Report()
    for i, row in enumerate(rows):
        model = row.get("model", "?")
        tokens = row.get("num_tokens", "?")
        impl = row.get("impl", "?")
        where = f"{label}[{i}] {model} T={tokens} {impl}"
        report.verdicts.append(check_row(row, where))
    return report


def crosscheck_files(paths) -> Report:
    report = Report()
    for path in paths:
        p = Path(path)
        report.verdicts.extend(crosscheck(SC.read_csv(p), p.name).verdicts)
    return report


# --------------------------------------------------------------------------
# synthetic rows: the lie, and the four ways a row is not a lie
# --------------------------------------------------------------------------

def v4_row(model="mixtral-8x7b", dtype="bf16", gpu="NVIDIA H200",
           source="vllm_tuned", **extra) -> dict:
    """The minimum a checked row needs, in the string forms a CSV reader yields."""
    row = {"schema_version": "4", "model": model, "dtype": dtype,
           "gpu_name": gpu, "num_tokens": "787", "impl": "vllm_fused_experts",
           "tile_config_source": source}
    row.update(extra)
    return row


def test_a_tuned_claim_on_a_shape_with_no_shipped_file_fails_loudly():
    """DeepSeek-V3 unsharded is `E=256,N=2048`, and NO device has a file at that
    key. Its run log says `Using default MoE config`, so a row claiming tuned is
    contradicting the run it came from."""
    v = check_row(v4_row(model="deepseek-v3", source="vllm_tuned"))
    assert v.status == LIED
    assert v.predicted_file == "E=256,N=2048,device_name=NVIDIA_H200.json"
    assert "TUNED" in v.detail


def test_the_two_cells_that_do_have_a_tuned_file_are_accepted():
    """The other half of the check. If mixtral and qwen2 on H200 failed here the
    test would only be asserting that nothing is ever tuned, which the A100
    rows already establish for free."""
    for model in ("mixtral-8x7b", "qwen2-57b-a14b"):
        v = check_row(v4_row(model=model, source="vllm_tuned"))
        assert v.status == AGREES, v
        assert v.predicted_file in SHIPPED


@pytest.mark.parametrize("model", ["mixtral-8x7b", "qwen2-57b-a14b",
                                   "deepseek-v2-lite", "deepseek-v3"])
def test_no_a100_cell_in_this_study_may_claim_a_tuned_config(model):
    """FINDINGS C5 defect 3, made checkable. Nothing ships for
    NVIDIA_A100-SXM4-80GB at any of the four shapes, so all four A100 cells took
    the hardcoded ladder. A `vllm_tuned` on any of them would mean the cross-card
    comparison was even less like-for-like than C5 already says."""
    gpu = "NVIDIA A100-SXM4-80GB"
    assert check_row(v4_row(model=model, gpu=gpu, source="vllm_tuned")).status == LIED
    assert check_row(v4_row(model=model, gpu=gpu, source="vllm_default")).status == AGREES


def test_the_fp8_shapes_flip_the_answer_for_two_models_and_not_for_deepseek():
    """The dtype selector is part of the filename, so the check has to move with
    it. C3's second rescope rests on exactly this: mixtral and qwen2 ship a
    `dtype=fp8_w8a8` file on H200 where deepseek-v3 ships nothing in either
    dtype."""
    for model in ("mixtral-8x7b", "qwen2-57b-a14b"):
        v = check_row(v4_row(model=model, dtype="fp8_e4m3", source="vllm_tuned"))
        assert v.status == AGREES, v
        assert v.predicted_file.endswith("dtype=fp8_w8a8.json")
    v = check_row(v4_row(model="deepseek-v3", dtype="fp8_e4m3", source="vllm_tuned"))
    assert v.status == LIED


def test_a_default_claim_on_a_shape_that_does_ship_a_file_is_also_a_disagreement():
    """The reverse direction, and it is not symmetric with the first: a lie
    inflates the evidence, this one only says the lookup and the prediction
    disagree. It still fails, because on tomorrow's pod both explanations for it
    (a wrong shape, or VLLM_BATCH_INVARIANT) are things the operator needs told
    before any tile column is read."""
    v = check_row(v4_row(model="mixtral-8x7b", source="vllm_default"))
    assert v.status == MISSED
    assert "VLLM_BATCH_INVARIANT" in v.detail


@pytest.mark.parametrize("source", ["sglang", "cutlass_static", "vllm_override",
                                    "unrecorded", "n/a"])
def test_a_source_that_never_opened_a_config_file_is_skipped_not_failed(source):
    """A torch grouped_mm row on deepseek-v3 has no shipped file and never asked
    for one. Failing it would make the whole check unusable on a real arm, where
    those rows are the majority."""
    v = check_row(v4_row(model="deepseek-v3", source=source))
    assert not v.failed
    assert v.status.startswith(f"source {source!r}")


def test_every_legal_source_is_either_checked_or_named_as_a_skip():
    """Closes the set against `schema.TILE_SOURCES`, so a source added later
    cannot fall through this file unnoticed and be counted as examined."""
    for source in SC.TILE_SOURCES:
        v = check_row(v4_row(model="mixtral-8x7b", source=source))
        assert v.status != "", source
        if source not in FILE_CLAIMING_SOURCES:
            assert not v.failed, source


def test_a_v3_row_is_skipped_as_predating_the_column_not_read_as_unrecorded():
    """Two different facts, and only one of them is fixable by re-running. A v3
    row has no column at all; a v4 row whose observer did not run has the column
    and an honest "unrecorded" in it."""
    assert check_row(v4_row(source=SC.UNRECORDED)).status == "predates schema v4"
    assert check_row(v4_row(source="unrecorded")).status.startswith("source")
    assert check_row({"model": "mixtral-8x7b"}).status == "no tile_config_source column"


def test_the_h200_nvl_family_fold_does_not_turn_an_honest_row_into_a_lie():
    """vLLM rewrites any device whose name contains the token H200 to plain
    NVIDIA_H200 before building the filename. A checker that used the raw
    gpu_name would predict a file that ships for nobody and would then score
    every tuned row on such a pod as a lie."""
    v = check_row(v4_row(model="mixtral-8x7b", gpu="NVIDIA H200 NVL",
                         source="vllm_tuned"))
    assert v.status == AGREES
    assert v.predicted_file == "E=8,N=14336,device_name=NVIDIA_H200.json"


# --------------------------------------------------------------------------
# over an arm, through the real CSV reader
# --------------------------------------------------------------------------

def test_the_check_runs_over_a_written_arm_and_catches_the_lie_through_read_csv(tmp_path):
    """End to end on a file, because every way this could be wrong in practice
    lives between the CSV and the dict: a source column read as bytes, a
    schema_version gate, a stamped sentinel. A test that only ever built dicts
    would pass while the command in the docstring did nothing.
    """
    csv_path = tmp_path / "run_synthetic.csv"
    header = ",".join(SC.COLUMNS)
    def line(**vals):
        return ",".join(str(vals.get(c, "")) for c in SC.COLUMNS)
    rows = [
        # honest: mixtral on H200 ships a bf16 file
        line(schema_version=4, model="mixtral-8x7b", dtype="bf16", num_tokens=787,
             impl="vllm_fused_experts", gpu_name="NVIDIA H200",
             tile_config_source="vllm_tuned", tile_block_m=128),
        # the lie: deepseek-v3 unsharded has no file on any device
        line(schema_version=4, model="deepseek-v3", dtype="bf16", num_tokens=16,
             impl="vllm_fused_experts", gpu_name="NVIDIA H200",
             tile_config_source="vllm_tuned", tile_block_m=16),
        # a torch row, which never consults a config file at all
        line(schema_version=4, model="deepseek-v3", dtype="bf16", num_tokens=16,
             impl="torch_grouped_mm_up", gpu_name="NVIDIA H200",
             tile_config_source="cutlass_static"),
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")

    report = crosscheck_files([csv_path])
    assert len(report.verdicts) == 3
    assert [v.status for v in report.verdicts][:2] == [AGREES, LIED]
    assert len(report.failures) == 1
    assert "deepseek-v3" in report.failures[0].where
    assert "E=256,N=2048" in report.failures[0].predicted_file
    assert "LIED" in report.summary()


def published_arms() -> list[Path]:
    return sorted(p for p in PUBLISHED.glob("*/run_*.csv"))


def test_the_published_corpus_exists_so_the_two_tests_below_are_not_vacuous():
    assert len(published_arms()) >= 10


@pytest.mark.parametrize("csv_path", published_arms(), ids=lambda p: p.parent.name)
def test_no_published_row_claims_a_tuned_config_it_could_not_have_had(csv_path):
    """Durable. It passes today because every published row is v3 and is skipped,
    and it keeps its meaning unchanged the moment a v4 arm is published here."""
    report = crosscheck_files([csv_path])
    assert report.failures == [], report.summary()


def test_every_published_row_is_skipped_for_a_reason_this_file_can_name():
    """The non-vacuity guard. `failures == []` is also what a check that examined
    nothing and knew nothing would report, so this asserts the corpus was
    actually read and that no row fell through into an unnamed state."""
    report = crosscheck_files(published_arms())
    assert len(report.verdicts) > 30_000
    reasons = report.skipped_by_reason()
    # The first v4 arm (2026-09-01) added two reasons that could not occur while
    # the corpus was all-v3: its cutlass and sglang spans DO carry a
    # tile_config_source, and that source names an engine rather than a vLLM
    # config file, so there is nothing for this check to compare against. That
    # is a skip with a name, which is what this test is guarding -- the failure
    # mode being prevented is a row falling through into an UNNAMED state, not a
    # row being skipped.
    assert set(reasons) <= {"predates schema v4", "no tile_config_source column",
                            "source 'cutlass_static' claims nothing about a file",
                            "source 'sglang' claims nothing about a file"}
    assert len(report.checked) + sum(reasons.values()) == len(report.verdicts)
    # And the check is no longer vacuous: 924 vLLM rows really were compared.
    assert len(report.checked) >= 900, (
        f"only {len(report.checked)} rows were actually checked; a corpus that "
        "skips everything reports zero failures too")


def main(argv: list[str]) -> int:
    """`python tests/test_tile_source_crosscheck.py <csv> ...`, for a fresh arm.

    Exits nonzero on a failure so it can gate a publish step, and prints the
    skip census either way: a run that examined zero rows is the answer that
    matters most on the first v4 arm, and it is invisible in a bare "0 failures".
    """
    if not argv:
        print(__doc__)
        return 2
    report = crosscheck_files(argv)
    print(report.summary())
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
