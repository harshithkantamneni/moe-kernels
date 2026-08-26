"""publish_results.sh must publish a whole sweep, not one arm of it.

A sweep runs the same arguments in several venvs, and each venv writes its own
`run_<id>_<env>.csv`. The three arms are one experiment. Publishing the newest
CSV therefore publishes a third of the result and looks complete, which is what
happened to `2026-08-26-nvidia_h200-full-three-way`: the sglang arm landed, the
base and vLLM arms had to be committed by hand afterwards.

The other half of the same bug is merged.csv. `results/` accumulates across
sessions, so the merged file can hold run ids from experiments that have nothing
to do with this one. Copying it wholesale is how 716 foreign rows, measured
against a different calibration, ended up inside a published arm.

These run the real script against a synthetic results directory. `--dry-run`
keeps git out of it; MOE_PUBLISH_ROOT keeps the repo's own results/published
untouched.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from moe.bench.schema import COLUMNS, SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "publish_results.sh"


def write_run(results: Path, run_id: str, env: str, *, gpu: str = "NVIDIA H200",
              n: int = 2) -> Path:
    """A minimal but schema-valid CSV, the shape a real arm writes."""
    path = results / f"run_{run_id}_{env}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for i in range(n):
            row = dict.fromkeys(COLUMNS, "")
            row.update(schema_version=SCHEMA_VERSION, run_id=run_id,
                       env_name=env, gpu_name=gpu, impl=f"{env}_impl",
                       model="toy", num_tokens=32 * (i + 1), ms_p50=1.0 + i,
                       correctness_passed="True", l2_flush="True",
                       cuda_graph="False")
            w.writerow(row)
    path.with_suffix(".manifest.jsonl").write_text("{}\n")
    return path


def run_publish(results: Path, publish_root: Path, *args: str):
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=REPO, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(results.parent),
             "MOE_RESULTS_DIR": str(results), "MOE_PUBLISH_ROOT": str(publish_root)},
    )


@pytest.fixture
def sweep(tmp_path):
    """Three arms of one sweep, plus a stale run from an earlier session."""
    results = tmp_path / "results"
    results.mkdir()
    write_run(results, "aaa111", "base")
    write_run(results, "bbb222", "vllm")
    write_run(results, "ccc333", "sglang")
    return results, tmp_path / "published"


def published_dir(publish_root: Path) -> Path:
    dirs = sorted(p for p in publish_root.iterdir() if p.is_dir())
    assert len(dirs) == 1, f"expected one published arm, got {dirs}"
    return dirs[0]


def test_all_publishes_every_arm_of_the_sweep(sweep):
    results, publish_root = sweep
    r = run_publish(results, publish_root, "--all", "--label", "three-way")
    assert r.returncode == 0, r.stderr
    dest = published_dir(publish_root)
    assert sorted(p.name for p in dest.glob("run_*.csv")) == [
        "run_aaa111_base.csv", "run_bbb222_vllm.csv", "run_ccc333_sglang.csv"]


def test_refuses_to_guess_when_several_runs_are_present(sweep):
    """The old behaviour silently took the newest. Silence is the bug."""
    results, publish_root = sweep
    r = run_publish(results, publish_root)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    for run_id in ("aaa111", "bbb222", "ccc333"):
        assert run_id in out, f"{run_id} not named in the refusal:\n{out}"
    assert "--all" in out, "the refusal must say how to proceed"


def test_named_run_ids_select_exactly_those(sweep):
    results, publish_root = sweep
    r = run_publish(results, publish_root, "--run-id", "aaa111", "--run-id", "ccc333")
    assert r.returncode == 0, r.stderr
    dest = published_dir(publish_root)
    assert sorted(p.name for p in dest.glob("run_*.csv")) == [
        "run_aaa111_base.csv", "run_ccc333_sglang.csv"]


def test_a_lone_run_still_needs_no_flag(tmp_path):
    """Refusing must not make the ordinary single-arm case harder."""
    results = tmp_path / "results"
    results.mkdir()
    write_run(results, "solo42", "base")
    r = run_publish(results, tmp_path / "published")
    assert r.returncode == 0, r.stderr
    dest = published_dir(tmp_path / "published")
    assert [p.name for p in dest.glob("run_*.csv")] == ["run_solo42_base.csv"]


def test_merged_csv_is_rebuilt_and_carries_no_foreign_rows(sweep):
    """The contamination path: results/ persists, so merged.csv outlives a sweep."""
    results, publish_root = sweep
    stale = write_run(results, "old999", "base", n=3)
    merged = results / "merged.csv"
    with merged.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for src in (results / "run_aaa111_base.csv", stale):
            with src.open(newline="") as s:
                w.writerows(csv.DictReader(s))

    r = run_publish(results, publish_root, "--run-id", "aaa111")
    assert r.returncode == 0, r.stderr
    dest = published_dir(publish_root)
    out = dest / "merged.csv"
    assert out.exists(), "a published arm without merged.csv loses the joined view"
    ids = {row["run_id"] for row in csv.DictReader(out.open(newline=""))}
    assert ids == {"aaa111"}, f"foreign run ids leaked into the published merge: {ids}"


def test_refuses_to_mix_devices_in_one_arm(tmp_path):
    """Each device gets its own arm with its own calibration beside it, so a
    directory holding two GPUs' rows has no single calibration to quote."""
    results = tmp_path / "results"
    results.mkdir()
    write_run(results, "h200aa", "base", gpu="NVIDIA H200")
    write_run(results, "a100bb", "base", gpu="NVIDIA A100-SXM4-80GB")
    r = run_publish(results, tmp_path / "published", "--all")
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "H200" in out and "A100" in out, out
