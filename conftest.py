"""Loaded by pytest before tests/conftest.py and before any test module is imported: the suite writes no bytecode
however it is invoked. Several tests assert that no __pycache__ is left in a registered directory (for example
tests/test_pc2_registration.py), and importing a scorer module in the pytest process would write one there unless
PYTHONDONTWRITEBYTECODE=1 was set by hand. Setting it here makes `pytest` and `PYTHONDONTWRITEBYTECODE=1 pytest` the
same run; the environment variable is also set so that child processes started with os.environ behave the same.
The one cached file this cannot prevent is this conftest's own (pytest compiles it before the first line runs), in
the repo root's __pycache__, which .gitignore covers and no test inspects."""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
