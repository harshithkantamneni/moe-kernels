"""Loaded by pytest before tests/conftest.py and before any test module is imported: the pytest process, and the child
processes that inherit os.environ, write no bytecode however pytest is invoked. Several tests assert that no __pycache__
is left in a registered directory (for example tests/test_pc2_registration.py), and importing a scorer module in the
pytest process would write one there unless PYTHONDONTWRITEBYTECODE=1 was set by hand. Setting it here makes `pytest`
and `PYTHONDONTWRITEBYTECODE=1 pytest` the same run for those tests. Two caches it does not prevent: this conftest's own
(pytest compiles it before the first line runs), in the repo root's __pycache__; and a test that builds its child's
environment from scratch (tests/test_h200_gaps_session.py's run() hands _spawn PATH and HOME only) can still write
scripts/__pycache__, with or without the variable, as it did before this file. .gitignore covers both and no test
inspects them."""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
