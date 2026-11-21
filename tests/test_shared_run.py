"""The shared-run helper (tests/_shared_run.py): one run of an expensive child
per pytest run, whichever worker asks first, and every other test reads it."""
from __future__ import annotations

import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared_run as S  # noqa: E402


def _counting(log: Path, value="made"):
    def produce(where: Path):
        with log.open("a") as fh:
            fh.write("run\n")
        (where / "out.txt").write_text(value)
        return value
    return produce


def test_a_name_runs_its_producer_once_and_every_call_reads_the_same_result(tmp_path):
    log = tmp_path / "log"
    first = S.shared_at(tmp_path, "one", _counting(log))
    again = S.shared_at(tmp_path, "one", _counting(log, value="a second run"))
    assert first == again == (tmp_path / "one", "made")
    assert log.read_text() == "run\n"
    other = S.shared_at(tmp_path, "two", _counting(log, value="another name"))
    assert other[1] == "another name" and log.read_text() == "run\nrun\n"


_RACER = """
import sys, time
sys.path.insert(0, {tests!r})
import _shared_run as S
from pathlib import Path
root, log = Path({root!r}), Path({log!r})
def produce(where):
    with log.open("a") as fh:
        fh.write("run\\n")
    time.sleep(0.5)
    return "one run"
print(S.shared_at(root, "raced", produce)[1])
"""


def test_four_processes_racing_for_one_name_run_its_producer_once(tmp_path):
    """The xdist case: separate processes, one flock. The producer sleeps so
    every racer arrives while it runs."""
    log = tmp_path / "log"
    code = _RACER.format(tests=str(Path(__file__).resolve().parent),
                         root=str(tmp_path), log=str(log))
    racers = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                               text=True) for _ in range(4)]
    outs = [r.communicate(timeout=60)[0].strip() for r in racers]
    assert [r.returncode for r in racers] == [0] * 4
    assert outs == ["one run"] * 4
    assert log.read_text() == "run\n"


def test_a_producer_that_raises_fails_every_consumer_and_is_not_run_again(tmp_path):
    calls = []

    def broken(where):
        calls.append(where)
        raise RuntimeError("the dry run timed out")
    for _ in range(2):
        with pytest.raises(S.SharedRunFailed, match="RuntimeError: the dry run timed out"):
            S.shared_at(tmp_path, "broken", broken)
    assert len(calls) == 1


def test_a_fixture_on_a_failed_run_errors_rather_than_skips(tmp_path):
    """`shared` turns the recorded failure into pytest.fail, which ERRORs the
    consuming test at setup; a skip would pass the suite with nothing run."""
    factory = types.SimpleNamespace(getbasetemp=lambda: tmp_path)
    request = types.SimpleNamespace(config=types.SimpleNamespace())

    def broken(where):
        raise OSError("no bash")
    # BaseException: a skip is one too, and a skip raised here would skip this
    # test rather than fail it, which is the very thing it guards against.
    with pytest.raises(BaseException, match="did not finish: OSError: no bash") as raised:
        S.shared(request, factory, "broken", broken)
    assert raised.type is pytest.fail.Exception, raised.type


def test_the_published_tree_is_frozen_read_only(tmp_path):
    def produce(where):
        (where / "logs").mkdir()
        (where / "logs" / "a.log").write_text("x")
        (where / "top.txt").write_text("y")
        return None
    where, _ = S.shared_at(tmp_path, "tree", produce)
    for path, mode in ((where, 0o555), (where / "logs", 0o555),
                       (where / "logs" / "a.log", 0o444), (where / "top.txt", 0o444)):
        assert stat.S_IMODE(path.stat().st_mode) == mode, path


def test_a_partial_directory_a_dead_worker_left_is_rebuilt(tmp_path):
    """A worker killed mid-run leaves the directory and no published result;
    the next taker deletes it, even frozen, and runs the producer afresh."""
    stale = tmp_path / "partial"
    (stale / "sub").mkdir(parents=True)
    (stale / "sub" / "half.log").write_text("cut short")
    S.freeze(stale)
    where, value = S.shared_at(tmp_path, "partial", _counting(tmp_path / "log"))
    assert value == "made" and not (where / "sub").exists()
    assert (where / "out.txt").read_text() == "made"


def test_the_root_is_the_runs_own_for_a_worker_and_for_a_serial_run(tmp_path):
    """An xdist worker's basetemp is <run>/popen-gwN, so the run's root is its
    parent; a serial run's basetemp is the run's own."""
    worker = tmp_path / "pytest-7" / "popen-gw3"
    worker.mkdir(parents=True)
    factory = types.SimpleNamespace(getbasetemp=lambda: worker)
    on_worker = S.run_root(factory, types.SimpleNamespace(workerinput={}))
    assert on_worker == tmp_path / "pytest-7" / "shared-runs" and on_worker.is_dir()
    serial = S.run_root(factory, types.SimpleNamespace())
    assert serial == worker / "shared-runs"


def test_a_name_that_is_not_a_plain_file_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a plain file name"):
        S.shared_at(tmp_path, "../escape", _counting(tmp_path / "log"))
