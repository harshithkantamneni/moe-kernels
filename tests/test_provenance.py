"""The provenance block, and what it must say when it does not know.

`moe/bench/provenance.py` is the one block every script writes. What the tests
pin, in the order the audit found them missing:

1. THE COMMIT AND THE DIRTY FLAG. In a temp repo with one commit the block
   carries that sha and dirty=False; touch a file and it is dirty=True with a
   count; outside a repo the git fields are None WITH A REASON, never "".
2. VERSIONS THAT ARE NOT THERE. A package that is not installed is None with
   "package not installed", not an empty string that reads as a version.
3. THE CARD. Off-GPU, gpu_name is None with "no CUDA"; with a fake torch that
   says CUDA is present, the name comes through. Nothing defaults to "NVIDIA
   H200". A torch that is installed and dies on import with an OSError (the
   pod's libcudart ABI failure) is a reason, not an exception out of the block;
   an empty device name is refused, not recorded as "".
4. A RIDGE WITHOUT A SOURCE is recorded and flagged, so the publish gate that
   checks `ridge_source` fails on it as it should.
5. SERIALISATION. `as_dict` is JSON-serialisable in a stable order; every
   `as_columns` key starts with `prov_`; `stamp` puts the audit's five keys at
   the top level and refuses to overwrite a different value.
6. RUN IDS. The card slug is in the id; any swept value changes it; key order
   does not; an empty card is refused; a None knob is refused; an empty-string
   knob is refused; the slug IS the calibration stem rule, not a copy of it.
7. EVERY REFUSAL PATH IS PLANTED. git status failing after rev-parse succeeded,
   nvidia-smi exiting non-zero or printing nothing, gethostname raising, and an
   empty torch.version.cuda each have a test that drives the branch and reads
   the reason back.

Nothing here needs a GPU. Git is needed for the temp-repo tests and they skip
if it is absent.
"""
from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
import types

import pytest

from moe.bench import provenance as PV

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git not on PATH")


def _git(root, *args):
    return subprocess.run([GIT, "-C", str(root), *args], capture_output=True, text=True,
                          check=True, env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                                           "HOME": str(root),
                                           "GIT_CONFIG_GLOBAL": "/dev/null",
                                           "GIT_CONFIG_SYSTEM": "/dev/null"})


@pytest.fixture
def temp_repo(tmp_path):
    """A fresh repository with exactly one commit and a clean tree."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / "a.txt").write_text("one\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "one")
    return root


# --------------------------------------------------------------------------
# 1. git
# --------------------------------------------------------------------------

@needs_git
def test_clean_repo_has_sha_and_is_not_dirty(temp_repo):
    want = _git(temp_repo, "rev-parse", "HEAD").stdout.strip()
    p = PV.provenance_block(repo_root=temp_repo)
    assert p.git_sha == want
    assert len(p.git_sha) == 40
    assert p.git_dirty is False
    assert p.git_dirty_files == 0
    assert "git_sha" not in p.missing
    assert "git_dirty" not in p.missing


@needs_git
def test_touched_file_makes_the_tree_dirty_with_a_count(temp_repo):
    (temp_repo / "a.txt").write_text("two\n")           # modified, tracked
    (temp_repo / "b.txt").write_text("new\n")           # untracked
    p = PV.provenance_block(repo_root=temp_repo)
    assert p.git_dirty is True
    assert p.git_dirty_files == 2
    assert "git_dirty_files" not in p.missing


@needs_git
def test_untracked_directory_counts_its_files_not_itself(temp_repo):
    """`--untracked-files=normal` reports a new directory as ONE entry; the
    docstring promises a count of files, so it must be three here."""
    new = temp_repo / "results"
    new.mkdir()
    for i in range(3):
        (new / f"cell{i}.json").write_text("{}\n")
    p = PV.provenance_block(repo_root=temp_repo)
    assert p.git_dirty is True
    assert p.git_dirty_files == 3


@needs_git
def test_ignored_files_are_not_dirty(temp_repo):
    (temp_repo / ".gitignore").write_text("*.log\n")
    _git(temp_repo, "add", ".gitignore")
    _git(temp_repo, "commit", "-q", "-m", "ignore")
    (temp_repo / "run.log").write_text("noise\n")
    p = PV.provenance_block(repo_root=temp_repo)
    assert p.git_dirty is False
    assert p.git_dirty_files == 0


@needs_git
def test_outside_a_repo_git_fields_are_none_with_a_reason(tmp_path, monkeypatch):
    # Stop git from walking up into whatever repository the temp dir lives under.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    p = PV.provenance_block(repo_root=plain)
    assert p.git_sha is None
    assert p.git_dirty is None
    assert p.git_dirty_files is None
    for name in ("git_sha", "git_dirty", "git_dirty_files"):
        assert name in p.missing
        assert p.missing[name]
    assert "not a git repository" in p.missing["git_sha"] or "git rev-parse" in p.missing["git_sha"]


def test_git_absent_is_a_named_reason(monkeypatch, tmp_path):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(PV.subprocess, "run", no_git)
    sha, dirty, n, reason = PV._git(tmp_path)
    assert (sha, dirty, n) == (None, None, None)
    assert reason == "git not on PATH"


def _fake_run(sha_rc=0, status_rc=0, status_raises=None, sha="a" * 40):
    """A `subprocess.run` whose rev-parse succeeds and whose status can be
    made to fail either way."""
    def run(cmd, **kw):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, sha_rc, stdout=sha + "\n", stderr="")
        if status_raises is not None:
            raise status_raises
        return subprocess.CompletedProcess(cmd, status_rc, stdout="", stderr="boom")
    return run


def test_git_status_that_raises_keeps_the_sha_and_names_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(PV.subprocess, "run",
                        _fake_run(status_raises=subprocess.TimeoutExpired("git", 10)))
    sha, dirty, n, reason = PV._git(tmp_path)
    assert sha == "a" * 40
    assert (dirty, n) == (None, None)
    assert reason == "git status failed: TimeoutExpired"
    p = PV.provenance_block(repo_root=tmp_path)
    assert p.git_sha == "a" * 40
    assert "git_sha" not in p.missing
    assert p.missing["git_dirty"] == reason
    assert p.missing["git_dirty_files"] == reason


def test_git_status_non_zero_keeps_the_sha_and_names_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(PV.subprocess, "run", _fake_run(status_rc=128))
    sha, dirty, n, reason = PV._git(tmp_path)
    assert sha == "a" * 40
    assert (dirty, n) == (None, None)
    assert reason == "git status returned non-zero"


def test_default_repo_root_is_this_repository():
    p = PV.provenance_block()
    # Either this checkout is a repo and the sha is a full hex string, or the
    # block says why it is not. Never an empty string.
    assert p.git_sha != ""
    if p.git_sha is None:
        assert p.missing["git_sha"]
    else:
        int(p.git_sha, 16)
        assert len(p.git_sha) == 40


# --------------------------------------------------------------------------
# 2. versions
# --------------------------------------------------------------------------

def test_absent_package_is_none_with_reason(monkeypatch):
    real = importlib.metadata.version

    def fake(dist):
        if dist == "vllm":
            raise importlib.metadata.PackageNotFoundError(dist)
        return real(dist)
    monkeypatch.setattr(importlib.metadata, "version", fake)
    p = PV.provenance_block()
    assert p.vllm is None
    assert p.missing["vllm"] == "package not installed"


def test_present_package_version_is_recorded(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda dist: {"torch": "2.13.0", "triton": "3.7.1",
                                      "vllm": "0.27.1", "sglang": "0.5.18"}[dist])
    p = PV.provenance_block()
    assert (p.torch, p.triton, p.vllm, p.sglang) == ("2.13.0", "3.7.1", "0.27.1", "0.5.18")
    for name in PV.PACKAGE_FIELDS:
        assert name not in p.missing


def test_metadata_lookup_crash_is_a_reason_not_an_exception(monkeypatch):
    def boom(dist):
        raise RuntimeError("corrupt metadata")
    monkeypatch.setattr(importlib.metadata, "version", boom)
    p = PV.provenance_block()
    assert p.torch is None
    assert "RuntimeError" in p.missing["torch"]


# --------------------------------------------------------------------------
# 3. the card
# --------------------------------------------------------------------------

def _fake_torch(available: bool, name: str = "NVIDIA H200", cuda: str | None = "12.8"):
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: 0,
        get_device_name=lambda i: name,
    )
    return types.SimpleNamespace(cuda=cuda_ns, version=types.SimpleNamespace(cuda=cuda))


def test_no_cuda_names_the_reason_and_never_defaults_a_card(monkeypatch):
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(False), None))
    p = PV.provenance_block()
    assert p.gpu_name is None
    assert p.cuda_version is None
    assert p.driver_version is None
    assert p.missing["gpu_name"] == "no CUDA"
    assert p.missing["cuda_version"] == "no CUDA"
    assert p.missing["driver_version"] == "no CUDA"
    assert "H200" not in json.dumps(p.as_dict())


def test_torch_not_importable_is_a_reason(monkeypatch):
    monkeypatch.setattr(PV, "_import_torch", lambda: (None, "torch not importable"))
    p = PV.provenance_block()
    assert p.gpu_name is None
    assert p.missing["gpu_name"] == "torch not importable"


def test_torch_import_that_dies_with_oserror_is_a_reason_not_an_exception(monkeypatch):
    """The pod's ABI failure: torch is installed and `import torch` raises
    OSError on libcudart. Not an ImportError, so an `except ImportError` let it
    out of `provenance_block`, which is documented as never raising."""
    import builtins
    real_import = builtins.__import__

    def dying(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise OSError("libcudart.so.12: cannot open shared object file")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", dying)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    mod, reason = PV._import_torch()
    assert mod is None
    assert reason == "torch import failed: OSError"
    p = PV.provenance_block()
    assert p.gpu_name is None
    assert p.missing["gpu_name"] == "torch import failed: OSError"
    assert p.missing["cuda_version"] == "torch import failed: OSError"
    assert p.missing["driver_version"] == "torch import failed: OSError"


def test_torch_import_that_dies_with_runtimeerror_is_a_reason(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def dying(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("ABI mismatch")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", dying)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert PV._import_torch() == (None, "torch import failed: RuntimeError")


def test_torch_genuinely_absent_is_not_importable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def absent(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", absent)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert PV._import_torch() == (None, "torch not importable")


@pytest.mark.parametrize("name", ["", None])
def test_empty_or_none_device_name_is_refused_with_a_reason(monkeypatch, name):
    """A falsy card name must never be recorded as gpu_name="" with no entry in
    `missing`, and must never raise a KeyError out of `provenance_block`."""
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(True, name=name), None))
    probed = []
    monkeypatch.setattr(PV, "_nvidia_smi_driver_version",
                        lambda: probed.append(1) or ("570", None))
    p = PV.provenance_block()
    assert p.gpu_name is None
    assert p.missing["gpu_name"] == "torch returned an empty device name"
    assert p.driver_version is None
    assert p.missing["driver_version"] == "torch returned an empty device name"
    assert probed == [], "nvidia-smi must not be probed for a card that has no name"


def test_empty_cuda_version_is_named(monkeypatch):
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(True, cuda=""), None))
    monkeypatch.setattr(PV, "_nvidia_smi_driver_version", lambda: ("570.86.15", None))
    p = PV.provenance_block()
    assert p.gpu_name == "NVIDIA H200"
    assert p.cuda_version is None
    assert p.missing["cuda_version"] == "torch.version.cuda is empty"
    assert "gpu_name" not in p.missing


def test_cuda_present_records_card_cuda_and_driver(monkeypatch):
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(True), None))
    monkeypatch.setattr(PV, "_nvidia_smi_driver_version", lambda: ("570.86.15", None))
    p = PV.provenance_block()
    assert p.gpu_name == "NVIDIA H200"
    assert p.cuda_version == "12.8"
    assert p.driver_version == "570.86.15"
    for name in ("gpu_name", "cuda_version", "driver_version"):
        assert name not in p.missing


def test_cuda_present_but_nvidia_smi_missing_is_a_reason(monkeypatch):
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(True), None))
    monkeypatch.setattr(PV, "_nvidia_smi_driver_version",
                        lambda: (None, "nvidia-smi not on PATH"))
    p = PV.provenance_block()
    assert p.gpu_name == "NVIDIA H200"
    assert p.driver_version is None
    assert p.missing["driver_version"] == "nvidia-smi not on PATH"


def test_cuda_query_that_raises_is_a_reason(monkeypatch):
    def raising(i):
        raise RuntimeError("CUDA driver initialization failed")
    fake = _fake_torch(True)
    fake.cuda.get_device_name = raising
    monkeypatch.setattr(PV, "_import_torch", lambda: (fake, None))
    p = PV.provenance_block()
    assert p.gpu_name is None
    assert "RuntimeError" in p.missing["gpu_name"]


def test_nvidia_smi_non_zero_exit_is_a_reason(monkeypatch):
    monkeypatch.setattr(PV.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 9, stdout="", stderr="NVIDIA-SMI has failed"))
    assert PV._nvidia_smi_driver_version() == (None, "nvidia-smi exited 9")


def test_nvidia_smi_printing_nothing_is_a_reason(monkeypatch):
    monkeypatch.setattr(PV.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="\n  \n", stderr=""))
    assert PV._nvidia_smi_driver_version() == (None, "nvidia-smi printed nothing")


def test_nvidia_smi_that_raises_is_a_reason(monkeypatch):
    def timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)
    monkeypatch.setattr(PV.subprocess, "run", timeout)
    assert PV._nvidia_smi_driver_version() == (None, "nvidia-smi failed: TimeoutExpired")


def test_nvidia_smi_first_line_is_the_driver_version(monkeypatch):
    monkeypatch.setattr(PV.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="570.86.15\n570.86.15\n", stderr=""))
    assert PV._nvidia_smi_driver_version() == ("570.86.15", None)


def test_gethostname_failure_is_a_reason(monkeypatch):
    def failing():
        raise OSError("no hostname")
    monkeypatch.setattr(PV.socket, "gethostname", failing)
    assert PV._hostname() == (None, "gethostname failed: OSError")
    p = PV.provenance_block()
    assert p.hostname is None
    assert p.missing["hostname"] == "gethostname failed: OSError"


def test_empty_hostname_is_a_reason(monkeypatch):
    monkeypatch.setattr(PV.socket, "gethostname", lambda: "")
    assert PV._hostname() == (None, "gethostname returned empty")


def test_nvidia_smi_absent_is_named():
    drv, reason = PV._nvidia_smi_driver_version()
    if shutil.which("nvidia-smi") is None:
        assert drv is None
        assert reason == "nvidia-smi not on PATH"
    else:
        assert (drv is None) == (reason is not None)


# --------------------------------------------------------------------------
# 4. caller-owned fields
# --------------------------------------------------------------------------

def test_caller_fields_are_recorded_and_absent_ones_named():
    p = PV.provenance_block(instrument="time_call/per-iter-sync", ridge=163.7,
                            ridge_source="measured_nvidia_h200.yaml 2026-09-01",
                            bandwidth=4374.5, bandwidth_source="calibrate triad",
                            warmup_ms=20.0, iters=50, target_ms=200.0)
    assert p.instrument == "time_call/per-iter-sync"
    assert p.ridge == 163.7
    assert p.iters == 50
    for name in PV.CALLER_FIELDS:
        assert name not in p.missing
    q = PV.provenance_block(instrument="time_call")
    for name in PV.CALLER_FIELDS:
        if name != "instrument":
            assert getattr(q, name) is None
            assert q.missing[name] == "not supplied by caller"


def test_ridge_without_source_is_recorded_and_flagged():
    p = PV.provenance_block(ridge=160.3)
    assert p.ridge == 160.3
    assert p.ridge_source is None
    assert p.missing["ridge_source"] == "supplied without a source"
    q = PV.provenance_block(bandwidth=4374.5)
    assert q.missing["bandwidth_source"] == "supplied without a source"


def test_explicit_none_is_distinguished_from_not_supplied():
    p = PV.provenance_block(ridge=None)
    assert p.missing["ridge"] == "supplied as None"


def test_misspelt_keyword_is_a_type_error_not_a_silent_drop():
    with pytest.raises(TypeError) as e:
        PV.provenance_block(ridge_src="x")
    assert "ridge_src" in str(e.value)


# --------------------------------------------------------------------------
# 5. serialisation
# --------------------------------------------------------------------------

def test_as_dict_is_json_serialisable_and_stable():
    p = PV.provenance_block(instrument="time_call", ridge=163.7, ridge_source="yaml")
    d = p.as_dict()
    text = json.dumps(d)
    assert json.loads(text) == d
    assert list(d)[:2] == ["provenance_version", "git_sha"]
    assert list(d)[-1] == "missing"
    assert list(d) == list(PV.provenance_block().as_dict())
    assert d["provenance_version"] == PV.PROVENANCE_VERSION
    assert d["utc"].endswith("+00:00")
    assert list(d["missing"]) == sorted(d["missing"])


def test_as_columns_keys_all_start_with_prov_and_are_scalar():
    cols = PV.provenance_block(iters=50).as_columns()
    assert cols
    assert all(k.startswith(PV.COLUMN_PREFIX) for k in cols)
    assert all(v is None or isinstance(v, (str, int, float, bool)) for v in cols.values())
    assert cols["prov_iters"] == 50
    assert "gpu_name=" in cols["prov_missing"]
    assert set(cols) == {f"prov_{k}" for k in PV.provenance_block().as_dict()}


def test_stamp_puts_the_audit_keys_at_top_level():
    p = PV.provenance_block(instrument="time_call", ridge=163.7, ridge_source="yaml",
                            bandwidth=4374.5, bandwidth_source="triad")
    out = p.stamp({"gates": [], "alpha": 0.9})
    assert set(PV.TOP_LEVEL_KEYS) <= set(out)
    assert out["provenance"] == p.as_dict()
    assert out["gates"] == [] and out["alpha"] == 0.9
    assert out["ridge_source"] == "yaml"
    assert json.loads(json.dumps(out)) == out


def test_stamp_writes_none_for_an_unknown_key_and_the_reason_beside_it(monkeypatch):
    """The audit's publish gate must test non-None, not presence: this is the
    shape a presence-only gate would wave through."""
    monkeypatch.setattr(PV, "_import_torch", lambda: (_fake_torch(False), None))
    out = PV.provenance_block().stamp({})
    assert "gpu_name" in out and out["gpu_name"] is None
    assert out["provenance"]["missing"]["gpu_name"] == "no CUDA"


def test_stamp_does_not_mutate_and_refuses_a_conflicting_value():
    p = PV.provenance_block(instrument="a")
    payload = {"instrument": "b"}
    with pytest.raises(PV.ProvenanceCollision):
        p.stamp(payload)
    assert payload == {"instrument": "b"}
    # the same value is not a collision
    assert p.stamp({"instrument": "a"})["instrument"] == "a"


def test_provenance_is_frozen():
    p = PV.provenance_block()
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.git_sha = "x"                                  # type: ignore[misc]


# --------------------------------------------------------------------------
# 6. run ids
# --------------------------------------------------------------------------

def test_run_id_contains_the_card_slug_first():
    rid = PV.run_id(card="NVIDIA H200", model="mixtral-8x7b", g=1)
    assert rid.startswith("nvidia_h200-")
    rid = PV.run_id(card="NVIDIA A100-SXM4-80GB", model="mixtral-8x7b", g=1)
    assert rid.startswith("nvidia_a100_sxm4_80gb-")


def test_run_id_is_identical_for_the_same_inputs_in_any_order():
    a = PV.run_id(card="NVIDIA H200", model="mixtral-8x7b", tiles=[32, 64], g=1, iters=50)
    b = PV.run_id(card="NVIDIA H200", iters=50, g=1, tiles=(32, 64), model="mixtral-8x7b")
    assert a == b


@pytest.mark.parametrize("change", [
    {"g": 16}, {"iters": 200}, {"tiles": [32]}, {"model": "qwen2-57b"},
    {"warmup": 5}, {"dtype": "fp8"},
])
def test_run_id_differs_when_any_swept_value_differs(change):
    base = dict(model="mixtral-8x7b", tiles=[32, 64], g=1, iters=50, warmup=20, dtype="bf16")
    a = PV.run_id(card="NVIDIA H200", **base)
    b = PV.run_id(card="NVIDIA H200", **{**base, **change})
    assert a != b
    assert a.rsplit("-", 1)[1] != b.rsplit("-", 1)[1]


def test_run_id_differs_across_cards_with_identical_knobs():
    """Collision 2: the committed A100/H200 filename that was byte-identical."""
    knobs = dict(model="mixtral-8x7b", dtype="bf16", r=1024, g=1, n=64)
    h200 = PV.run_id(card="NVIDIA H200", **knobs)
    a100 = PV.run_id(card="NVIDIA A100-SXM4-80GB", **knobs)
    assert h200 != a100
    assert h200.rsplit("-", 1)[1] != a100.rsplit("-", 1)[1]


def test_run_id_adding_a_knob_changes_it():
    a = PV.run_id(card="NVIDIA H200", model="m")
    b = PV.run_id(card="NVIDIA H200", model="m", g=1)
    assert a != b


def test_run_id_ends_with_an_eight_hex_hash_and_is_filesystem_safe():
    rid = PV.run_id(card="NVIDIA H200", model="mixtral-8x7b", tiles=[32, 64])
    tail = rid.rsplit("-", 1)[1]
    assert len(tail) == 8
    int(tail, 16)
    assert "/" not in rid and " " not in rid
    assert rid == rid.lower()


def test_run_id_visible_part_is_capped_but_hash_still_separates():
    long_a = PV.run_id(card="NVIDIA H200", **{f"knob{i}": i for i in range(40)})
    long_b = PV.run_id(card="NVIDIA H200", **{f"knob{i}": i for i in range(39)}, knob39=99)
    assert len(long_a) <= PV._VISIBLE_MAX + 9
    assert long_a != long_b


@pytest.mark.parametrize("card", [None, "", "   ", "---"])
def test_run_id_refuses_a_missing_card(card):
    with pytest.raises(PV.NoCard):
        PV.run_id(card=card, model="m")


def test_run_id_refuses_a_none_knob():
    with pytest.raises(PV.UnresolvedKnob):
        PV.run_id(card="NVIDIA H200", model="m", ridge=None)


@pytest.mark.parametrize("value", ["", "   "])
def test_run_id_refuses_an_empty_string_knob(value):
    """An argparse default of '' is as unresolved as None."""
    with pytest.raises(PV.UnresolvedKnob):
        PV.run_id(card="NVIDIA H200", model="m", tag=value)


def test_run_id_refuses_an_empty_string_inside_a_list_knob():
    with pytest.raises(PV.UnresolvedKnob):
        PV.run_id(card="NVIDIA H200", tiles=[32, ""])


def test_run_id_refuses_a_non_json_native_knob():
    with pytest.raises(TypeError):
        PV.run_id(card="NVIDIA H200", thing=object())


@pytest.mark.parametrize("name", [
    "NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 4090",
    "NVIDIA H200 ²", "Tesla V100-SXM2-32GB", "  odd   spacing  ", "H100_80GB-HBM3",
])
def test_card_slug_is_the_calibration_stem_rule_not_a_copy_of_it(name):
    """Including a non-ASCII alphanumeric, where a regex re-implementation and
    `str.isalnum` once disagreed."""
    from moe.bench.roofline import measured_slug
    assert measured_slug(name) == f"measured_{PV.card_slug(name)}"
    assert PV.run_id(card=name, model="m").startswith(PV.card_slug(name) + "-")
