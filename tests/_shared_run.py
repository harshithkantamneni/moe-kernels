"""One run of an expensive child, shared by every test that asks a question of
it, across pytest-xdist workers.

A module or session fixture runs once PER WORKER: under `-n auto` the gaps
session's dry run (about 30 s) ran once per test that read it, 17 times a
suite, and the chain's dry run on every worker. `shared` runs it ONCE per
pytest run: the first worker to take the name's lock runs the producer into a
fresh directory under the run's shared tmp root, freezes that tree read-only
and publishes the producer's return value; every other test, on any worker,
waits on the lock and reads what was published.

What the callers keep to:
- One fixture per distinct invocation, so "the same fixture" means "a
  byte-identical call": argv, environment and planted state. A test that needs
  its own state takes `tmp_path`, not a shared run.
- The producer returns stdlib values only (CompletedProcess, Path, str, list,
  dict): the value crosses processes by pickle.
- Consumers only READ. The tree is frozen when it is published (files 0o444,
  directories 0o555), so a stray write fails loudly instead of changing what a
  sibling reads (root ignores the bits, so on a pod this is a convention).
- A producer that raises is recorded once, and every consumer ERRORs at setup
  with its message: nothing skips, and no other worker pays for the run again.

The lock is `fcntl.flock` on a file beside the result (POSIX: the laptop and
the pods). The OS releases it when a worker dies; the next taker finds nothing
published, deletes the partial directory and runs the producer again. Without
xdist the root is the run's own basetemp, so a serial run shares within itself.
"""
from __future__ import annotations

import fcntl
import os
import pickle
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest


class SharedRunFailed(Exception):
    """The producer of a shared run raised; its message, recorded once."""


def run_root(tmp_path_factory, config) -> Path:
    """The directory every worker of THIS pytest run shares, and no other run
    does: an xdist worker's basetemp is `<run>/popen-gwN`, so its parent;
    without xdist, the run's own basetemp. Detected by `config.workerinput`,
    which exists only on an xdist worker (a pod venv may have no xdist)."""
    base = Path(tmp_path_factory.getbasetemp())
    root = (base.parent if hasattr(config, "workerinput") else base) / "shared-runs"
    root.mkdir(exist_ok=True)
    return root


@contextmanager
def _locked(path: Path):
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def freeze(tree: Path) -> None:
    """Every file 0o444 and every directory 0o555, the tree's root last."""
    for dirpath, _dirs, files in os.walk(tree):
        for name in files:
            path = Path(dirpath, name)
            if not path.is_symlink():
                path.chmod(0o444)
    for dirpath, dirs, _files in os.walk(tree, topdown=False):
        for name in dirs:
            path = Path(dirpath, name)
            if not path.is_symlink():
                path.chmod(0o555)
    tree.chmod(0o555)


def _remove(tree: Path) -> None:
    """A partial (or frozen) tree, made writable again and deleted."""
    for dirpath, _dirs, _files in os.walk(tree):
        os.chmod(dirpath, 0o755)
    shutil.rmtree(tree)


def shared_at(root: Path, name: str, produce) -> tuple[Path, object]:
    """`produce(directory)` run once under `root` for `name`; every call after
    the first, from any process, returns the same (directory, value). Raises
    `SharedRunFailed` with the recorded message when the producer raised."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"shared run name {name!r} is not a plain file name")
    where, done = root / name, root / f"{name}.pickle"
    with _locked(root / f"{name}.lock"):
        if not done.exists():
            if where.exists():
                _remove(where)
            where.mkdir()
            try:
                payload = pickle.dumps(("ok", produce(where)))
            except Exception as exc:                             # noqa: BLE001
                payload = pickle.dumps(("failed", f"{type(exc).__name__}: {exc}"))
            freeze(where)
            staged = done.with_name(done.name + ".tmp")
            staged.write_bytes(payload)
            os.replace(staged, done)
    # Pickle, because a CompletedProcess and Paths must cross processes. Safe:
    # the only file read is one this pytest run wrote, in its own basetemp.
    status, value = pickle.loads(done.read_bytes())
    if status != "ok":
        raise SharedRunFailed(value)
    return where, value


def shared(request, tmp_path_factory, name: str, produce) -> tuple[Path, object]:
    """`shared_at` under this run's shared root, for a fixture: a producer
    that raised ERRORs the consuming test, never skips it."""
    try:
        return shared_at(run_root(tmp_path_factory, request.config), name, produce)
    except SharedRunFailed as exc:
        pytest.fail(f"the shared run {name!r} did not finish: {exc}", pytrace=False)
