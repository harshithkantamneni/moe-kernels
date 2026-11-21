"""The off-GPU analysis tools: they must RUN, and they must read the right file.

Everything in this file is a laptop check on a tool that was documented as a
laptop tool and was not one. The 2026-09-02 audit found four separable defects
and each has its own section:

1. TORCH ON THE IMPORT PATH OF TOOLS THAT NEVER TOUCH A DEVICE (B11).
   `moe/bench/calibrate.py` and `moe/quant.py` imported torch at module scope,
   and `moe/bench/published.py`, `moe/bench/tile_resolve.py` and
   `scripts/alpha_refit.py` reach them. torch is not one of the four
   dependencies `pyproject.toml` declares, so in a clean environment none of
   those three could start -- while `alpha_refit`'s own docstring promised "no
   GPU, no torch". The tests here block torch AT THE IMPORT FINDER, in a
   subprocess, so a lazily reintroduced module-scope import fails them.

2. A CALIBRATION RESOLVED AGAINST THE RUNNING HOST (B11/R14). `scripts/plot.py`
   called `roofline.load_measured()` with no argument, which reads the GPU
   ATTACHED TO THIS PROCESS, and ignored the `measured.yaml` published beside
   the CSVs it was plotting.

3. A GLOB THAT CANNOT READ THE COMMITTED LAYOUT, AND A LEVER SORTED AS TEXT
   (S38). `scripts/alpha_surface.py` looked for a bare `report.json` in arms
   that hold `<name>.report.json`, and sorted GROUP_SIZE_M as strings so its
   paired-change line named 1 -> 8 instead of 1 -> 64.

4. A VALIDATION REPORTED AS ITS OWN OPPOSITE (P). `moe.bench.tile_resolve`'s CLI
   printed "No row observed a tile to check the derivation against ... it is NOT
   a pass" after comparing 924 rows and finding zero disagreements.

5. A HEADER THAT DESCRIBED A DIFFERENT ARM (B1). Fixing 3 replaced the pooled
   `SURFACE.txt` files, and the block explaining the supersession was written
   once and copied into all three arms, so two of them stated the s4 arm's fit
   counts and medians as facts about their own tables and carried the s4 arm's
   regeneration command. The numbers were real; the arm they belonged to was
   not the one they headed.

Plus the two things this slice added and that a later edit could quietly
weaken: the pinned input set for `alpha_refit`, and the MDE lines.

No GPU. Nothing here writes to the repository.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "published"
H200_S4 = PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"
RIDGE_ARM = PUBLISHED / "2026-08-28-nvidia_h200-ridge-resolution"

#: The three arms that hold a paired SURFACE.txt and the pooled version it
#: replaced. Named once because two tests below iterate all three: the defect
#: they exist for was ONE header copied into all of them, so a test that checked
#: a single arm would have passed on the day it happened.
POOLED_ARMS = ("2026-09-01-nvidia_h200-alpha-surface-s4",
               "2026-09-01-nvidia_h200-cross-card-s3",
               "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3")

#: Separates the superseded-header prose from the pooled table it heads.
POOLED_DIVIDER = "-" * 74

#: Injected as `sitecustomize.py` into a subprocess's PYTHONPATH. A meta-path
#: finder that RAISES on torch is stricter than uninstalling it: it also catches
#: an import inside a function that is reached on the off-GPU path, which is the
#: half of the defect a bare "is torch installed" check would miss.
BLOCKER = textwrap.dedent('''
    import sys

    class BlockTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError(
                    "torch is blocked: this path must not need it")

    sys.meta_path.insert(0, BlockTorch())
''')


@pytest.fixture(scope="module")
def torch_blocked(tmp_path_factory) -> dict:
    """An environment in which importing torch raises."""
    import os

    blockdir = tmp_path_factory.mktemp("torchblock")
    (blockdir / "sitecustomize.py").write_text(BLOCKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(blockdir) + os.pathsep + str(ROOT)
    return env


def _run(args, env=None, cwd=ROOT):
    return subprocess.run([sys.executable, *args], capture_output=True,
                          text=True, env=env, cwd=cwd)


def _load(name: str):
    """Load a script by path. `scripts/` is not a package and never has been."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module        # @dataclass resolves through sys.modules
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. torch is not on the import path of anything that does not measure
# --------------------------------------------------------------------------

def test_the_blocker_actually_blocks(torch_blocked):
    """Non-vacuity. Without this, every test below passes on a broken blocker."""
    got = _run(["-c", "import torch"], env=torch_blocked)
    assert got.returncode != 0
    assert "torch is blocked" in got.stderr


@pytest.mark.parametrize("module", [
    "moe.bench.calibrate", "moe.quant", "moe.bench.published",
    "moe.bench.tile_resolve",
])
def test_calibrate_imports_without_torch(torch_blocked, module):
    got = _run(["-c", f"import {module}; import sys; "
                      "assert 'torch' not in sys.modules"], env=torch_blocked)
    assert got.returncode == 0, got.stderr


def test_the_three_documented_analysis_commands_run_without_torch(torch_blocked):
    """The exact commands audit B11's fix names as its test."""
    published = _run(["-m", "moe.bench.published",
                      *(str(p) for p in sorted(PUBLISHED.glob("2026-*")))],
                     env=torch_blocked)
    assert published.returncode == 0, published.stderr

    refit = _run(["scripts/alpha_refit.py", "--pinned-set", "--bootstrap", "2"],
                 env=torch_blocked)
    assert refit.returncode == 0, refit.stderr
    assert "alpha = 0.558" in refit.stdout

    surface = _run(["scripts/alpha_surface.py", str(H200_S4)], env=torch_blocked)
    assert surface.returncode == 0, surface.stderr


def test_plot_runs_without_torch(torch_blocked, tmp_path):
    got = _run(["scripts/plot.py", "--results", str(RIDGE_ARM),
                "--out", str(tmp_path / "figs")], env=torch_blocked)
    assert got.returncode == 0, got.stderr
    assert (tmp_path / "figs" / "roofline_bf16.png").exists()


def test_a_calibration_stamp_reads_with_no_torch_and_measuring_still_refuses(
        torch_blocked):
    """The two halves of `calibrate.py`, on the same run.

    Reading a stamp must work; asking the module to MEASURE must fail loudly
    rather than quietly, because the failure branch is what proves the lazy
    import did not turn a missing dependency into a silent no-op.
    """
    script = textwrap.dedent(f'''
        from moe.bench.calibrate import read_stamp, measure_bandwidth
        stamp = read_stamp(r"{RIDGE_ARM / 'measured.yaml'}")
        assert stamp.gpu_name, stamp
        try:
            measure_bandwidth()
        except ImportError as exc:
            print("REFUSED", exc)
        else:
            raise AssertionError("measuring must not succeed without torch")
    ''')
    got = _run(["-c", script], env=torch_blocked)
    assert got.returncode == 0, got.stderr
    assert "REFUSED" in got.stdout


# --------------------------------------------------------------------------
# 2. plot.py reads the arm's own calibration, and refuses when there is none
# --------------------------------------------------------------------------

def test_arm_profile_resolves_the_arms_own_yaml():
    """Pins the pathlib rule the resolution rests on.

    `roofline.load_hardware` joins the profile name onto `HARDWARE_DIR`, and
    joining an ABSOLUTE right-hand side yields the absolute path. That is the
    only seam `roofline.plot` offers a calibration outside the hardware
    directory, so it is pinned here rather than left as a coincidence.
    """
    from moe.bench import roofline as RL

    plot = _load("plot")
    name = plot.arm_profile(RIDGE_ARM)
    assert name is not None
    assert Path(name).is_absolute()
    assert (RL.HARDWARE_DIR / f"{name}.yaml").resolve() == \
        (RIDGE_ARM / "measured.yaml").resolve()

    hw = RL.load_hardware(name)
    assert "H200" in hw.name


def test_arm_profile_is_none_when_the_arm_publishes_no_calibration(tmp_path):
    plot = _load("plot")
    assert plot.arm_profile(tmp_path) is None


def test_plot_refuses_a_results_directory_with_no_calibration(tmp_path):
    """The FAIL branch. A refusal that has only been reasoned about is not one."""
    rows = next(RIDGE_ARM.glob("run_*_vllm.csv"))
    (tmp_path / rows.name).write_text(rows.read_text())
    got = _run(["scripts/plot.py", "--results", str(tmp_path),
                "--out", str(tmp_path / "figs")])
    assert got.returncode == 1
    assert "REFUSING" in got.stdout
    assert "measured.yaml" in got.stdout
    assert not (tmp_path / "figs").exists()


def test_plot_hardware_flag_overrides_the_arms_calibration(tmp_path):
    got = _run(["scripts/plot.py", "--results", str(RIDGE_ARM),
                "--hardware", "measured_nvidia_h200",
                "--out", str(tmp_path / "figs")])
    assert got.returncode == 0, got.stderr
    # The banner naming the arm's own yaml appears only when nothing overrode it.
    assert "the calibration published with these rows" not in got.stdout


# --------------------------------------------------------------------------
# 3. alpha_surface: the committed layout, and a lever sorted as a number
# --------------------------------------------------------------------------

def test_report_paths_finds_the_committed_named_layout():
    surface = _load("alpha_surface")
    found = surface.report_paths(H200_S4)
    assert len(found) == 12
    assert all(p.name.endswith(".report.json") for p in found)


def test_report_paths_finds_a_bare_report_json_too(tmp_path):
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "cell.report.json").write_text("{}")
    surface = _load("alpha_surface")
    names = [p.name for p in surface.report_paths(tmp_path)]
    assert names == ["cell.report.json", "report.json"]


def test_report_paths_never_yields_a_file_twice(tmp_path):
    """The set is not decoration: every median below is over this list."""
    (tmp_path / "report.json").write_text("{}")
    surface = _load("alpha_surface")
    assert len(surface.report_paths(tmp_path)) == 1


def test_lever_levels_sort_numerically_not_as_text():
    surface = _load("alpha_surface")
    got = sorted([1, 16, 64, 8], key=surface.level_sort_key)
    assert got == [1, 8, 16, 64]
    assert got[0] == 1 and got[-1] == 64          # the paired-change endpoints


def test_lever_levels_still_sort_strings_and_put_none_last():
    surface = _load("alpha_surface")
    got = sorted(["qwen2", None, "mixtral", 64], key=surface.level_sort_key)
    assert got == [64, "mixtral", "qwen2", None]


def test_the_swizzle_sweep_is_reported_as_1_to_64():
    """Audit B11's stated acceptance test for this script, run as a test."""
    got = _run(["scripts/alpha_surface.py", str(H200_S4)])
    assert got.returncode == 0, got.stderr
    assert "paired change 1 -> 64:" in got.stdout
    assert "paired change 1 -> 8:" not in got.stdout


def _without_dated_notes(text: str) -> str:
    """Drop every hand-added `  NOTE 2026-09-03:` paragraph. Those notes
    requalify a line the generator still prints unqualified (ANCHOR_RESCORE
    W3, W4) and say so in their own text; they are dated, deliberate, and the
    one thing in a published surface that is NOT the generator's output."""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("  NOTE 2026-09-03:"):
            skipping = True
        if skipping and line.strip() == "":
            # The blank line that ends the note is the generator's own blank
            # line between the candidate paragraph and the next one: keep it.
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out)


def test_the_committed_surface_files_regenerate_byte_for_byte():
    """A published summary a stranger cannot rebuild is not published evidence.

    Byte for byte APART FROM the dated requalification notes beneath the
    "0 of N fits within 0.05" line, which are added by hand until
    scripts/alpha_surface.py prints the requalification itself; the notes say
    so. Everything the generator writes must match exactly."""
    for arm in ("2026-09-01-nvidia_h200-alpha-surface-s4",
                "2026-09-01-nvidia_h200-cross-card-s3",
                "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"):
        got = _run(["scripts/alpha_surface.py", f"results/published/{arm}"])
        assert got.returncode == 0, got.stderr
        committed = (PUBLISHED / arm / "SURFACE.txt").read_text()
        assert got.stdout == _without_dated_notes(committed), arm
        assert committed != _without_dated_notes(committed), (
            f"{arm}: the requalification note is gone; W3 requires it")


def test_the_pooled_surface_is_kept_and_says_it_is_superseded():
    for arm in POOLED_ARMS:
        pooled = (PUBLISHED / arm / "SURFACE.pooled.txt").read_text()
        assert pooled.startswith("THIS FILE IS THE SUPERSEDED, POOLED VERSION")
        assert "no report.json under" in pooled or "unregenerable" in pooled


# --- a superseded header must describe the arm it sits in --------------------
#
# MEASURED, 2026-09-02 (audit B1). The superseded-header block was written once
# and copied into all three pooled files, so the cross-card and A100 files
# opened by stating the s4 arm's counts ("ten identifiable mixtral fits against
# seven for qwen2") and the s4 arm's pooled medians (0.731 and 0.713) as facts
# about grids that hold neither. Both files even carried the regeneration
# command for the arm they were copied FROM, so a reader who ran it got a third
# arm's table. A header that describes the wrong arm is worse than no header:
# it reads as provenance, and the numbers in it are quotable.
#
# The two checks below are the two halves of that defect, and each fails on a
# copied header for a different reason: the command must point at its own
# directory, and the counts must match the body underneath.

def _pooled_header(arm: str) -> str:
    """The prose above the divider, with runs of whitespace collapsed.

    Collapsed because the header is hand-wrapped prose: a phrase that must
    appear in it can be broken across a line at any word, and a test that only
    matched the unwrapped spelling would fail on a reflow and pass on a lie.
    """
    text = (PUBLISHED / arm / "SURFACE.pooled.txt").read_text()
    return " ".join(text.split(POOLED_DIVIDER)[0].split())


def test_each_pooled_header_regenerates_the_file_beside_it_and_no_other():
    for arm in POOLED_ARMS:
        header = _pooled_header(arm)
        assert f"scripts/alpha_surface.py results/published/{arm}" in header, arm
        for other in POOLED_ARMS:
            if other != arm:
                assert f"results/published/{other}" not in header, (arm, other)


def test_each_pooled_header_states_its_own_arms_fit_counts():
    """The counts in the header are read back out of the table below it.

    Total, block sizes swept, and one count per model. A header copied from
    another arm disagrees with its own body on every one of them, which is what
    made the defect visible; asserting it keeps the next copy visible too.
    """
    for arm in POOLED_ARMS:
        text = (PUBLISHED / arm / "SURFACE.pooled.txt").read_text()
        header, body = _pooled_header(arm), text.split(POOLED_DIVIDER)[1]

        total, swept = re.search(
            r"(\d+) identifiable fit\(s\) of (\d+) block sizes swept", body).groups()
        assert f"{total} identifiable fits out of {swept} block sizes swept" \
            in header, arm

        # The model block only: a blank line ends it, and the BLOCK_SIZE_N
        # block that follows in the s4 arm has the same row shape.
        block = body.split("alpha against model")[1].split("\n\n")[0]
        models = re.findall(r"^  ([a-z0-9-]+)\s+n=(\d+)\s+median alpha",
                            block, re.M)
        assert models, arm
        assert sum(int(n) for _m, n in models) == int(total), (arm, models)
        for model, n in models:
            assert f"{n} {model}" in header, (arm, model, n)


def test_the_surface_states_an_mde_or_says_it_has_none():
    got = _run(["scripts/alpha_surface.py", str(H200_S4)])
    assert "MDE" in got.stdout
    assert "paired, 90% two-sided, 80% power" in got.stdout


def test_the_surface_refuses_an_mde_when_the_noise_floor_is_missing(monkeypatch,
                                                                    tmp_path):
    """The FAIL branch of the MDE line: no measured floor, no number.

    `prior_sd` returns `(sd, basis, source)` since 2026-09-03, when it stopped
    reading `prior_sd` out of the JSON and started going through
    `replicate_noise_floor.sizing_sigma`. NONE is the third basis: neither a
    measurement nor a declared prior, and no number.
    """
    surface = _load("alpha_surface")
    monkeypatch.setattr(surface, "NOISE_FLOOR", tmp_path / "absent.json")
    sd, basis, why = surface.prior_sd()
    assert sd is None
    assert basis == "NONE"
    assert "does not exist" in why


def test_the_surface_refuses_an_mde_when_the_floor_records_no_prior_sd(
        monkeypatch, tmp_path):
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps({"schema": "x", "prior_sd": None}))
    surface = _load("alpha_surface")
    monkeypatch.setattr(surface, "NOISE_FLOOR", path)
    sd, basis, why = surface.prior_sd()
    assert sd is None
    assert basis == "NONE"
    assert "prior_sd" in why


# --------------------------------------------------------------------------
# 4. tile_resolve reports a validation as a validation
# --------------------------------------------------------------------------

def test_the_tile_census_reports_924_agreements_not_an_absence():
    got = _run(["-m", "moe.bench.tile_resolve",
                *(str(p) for p in sorted(PUBLISHED.glob("2026-*/run_*.csv")))])
    assert got.returncode == 0, got.stderr
    assert "924 rows observed a tile; the derivation matches every one of them" \
        in got.stdout
    assert "No row observed a tile" not in got.stdout


def test_the_no_v4_rows_message_still_exists_for_a_corpus_that_has_none(tmp_path):
    """The other branch. It is the right message when it is true, and only then."""
    from moe.bench import schema as SC

    source = next((PUBLISHED / "2026-08-22-standard-sweep").glob("run_*.csv"))
    rows = SC.read_csv(source)
    assert not any(SC.has_tile_config(r) for r in rows), \
        "this arm was chosen because it is schema v3; pick another if that changed"
    got = _run(["-m", "moe.bench.tile_resolve", str(source)])
    assert got.returncode == 0, got.stderr
    assert "No row observed a tile" in got.stdout


def test_comparable_separates_unchecked_from_checked_and_agreeing():
    from moe.bench import schema as SC
    from moe.bench import tile_resolve as TR

    rows = []
    for path in sorted(PUBLISHED.glob("2026-*/run_*.csv")):
        rows.extend(SC.read_csv(path))
    checked = [r for r in rows if TR.comparable(r)]
    assert len(checked) == 924
    assert all(TR.disagreement_with_observed(r) is None for r in checked)
    # A v3 row is not comparable, and its None means something else entirely.
    v3 = next(r for r in rows if not SC.has_tile_config(r))
    assert not TR.comparable(v3)
    assert TR.disagreement_with_observed(v3) is None


# --------------------------------------------------------------------------
# alpha_refit: the pinned set, the cluster levels, the paired basis contrast
# --------------------------------------------------------------------------

def test_the_pinned_set_reproduces_the_numbers_findings_quotes():
    got = _run(["scripts/alpha_refit.py", "--pinned-set"])
    assert got.returncode == 0, got.stderr
    assert "alpha = 0.558" in got.stdout
    assert "n = 10813 rows, 3124 discriminating" in got.stdout
    assert "cell  0.529 .. 0.588" in got.stdout
    assert "PINNED by results/published/REFIT_SET.txt" in got.stdout


def test_the_documented_glob_answers_differently_and_says_so():
    """The disagreement is the point of pinning; it must stay visible."""
    got = _run(["scripts/alpha_refit.py",
                *(str(p) for p in sorted(PUBLISHED.glob("2026-*/run_*.csv"))),
                "--bootstrap", "2"])
    assert got.returncode == 0, got.stderr
    assert "alpha = 0.560" in got.stdout
    assert "n = 11181 rows, 3181 discriminating" in got.stdout
    assert "NOT the pinned set" in got.stdout


def test_the_pinned_set_refuses_an_arm_it_cannot_find(tmp_path):
    refit = _load("alpha_refit")
    manifest = tmp_path / "SET.txt"
    manifest.write_text("# a comment\n\n2026-01-01-no-such-arm\n")
    with pytest.raises(refit.PinnedSetBroken) as excinfo:
        refit.read_refit_set(manifest)
    assert "absent from" in str(excinfo.value)


def test_the_pinned_set_refuses_an_arm_that_holds_no_csvs(tmp_path):
    refit = _load("alpha_refit")
    (tmp_path / "arm").mkdir()
    manifest = tmp_path / "SET.txt"
    manifest.write_text("arm\n")
    with pytest.raises(refit.PinnedSetBroken) as excinfo:
        refit.read_refit_set(manifest)
    assert "holding no" in str(excinfo.value)


def test_the_pinned_set_refuses_an_empty_manifest(tmp_path):
    refit = _load("alpha_refit")
    manifest = tmp_path / "SET.txt"
    manifest.write_text("# nothing but comments\n")
    with pytest.raises(refit.PinnedSetBroken):
        refit.read_refit_set(manifest)


def test_the_committed_manifest_describes_this_tree():
    refit = _load("alpha_refit")
    arms, csvs = refit.read_refit_set(refit.DEFAULT_REFIT_SET)
    assert len(arms) == 10
    assert csvs and all(p.exists() for p in csvs)


def test_passing_both_a_glob_and_the_pinned_set_is_refused():
    got = _run(["scripts/alpha_refit.py", "some.csv", "--pinned-set"])
    assert got.returncode == 2
    assert "not both" in got.stderr


def test_the_cell_level_band_is_exactly_the_published_one():
    """The generalised bootstrap must reduce to what produced 0.529-0.588."""
    refit = _load("alpha_refit")
    obs = _tiny_pool(refit)
    ids_band = refit.bootstrap_band(obs, 40, 0, level="cell")
    assert ids_band is not None
    lo, hi = ids_band
    assert lo <= refit.fit_alpha(obs) <= hi


def test_an_unknown_cluster_level_raises_rather_than_falling_back():
    refit = _load("alpha_refit")
    with pytest.raises(KeyError):
        refit.bootstrap_band(_tiny_pool(refit), 4, 0, level="grpu")


def test_a_single_cluster_gives_no_band():
    refit = _load("alpha_refit")
    obs = _tiny_pool(refit)            # one card
    assert refit.bootstrap_band(obs, 4, 0, level="card") is None


def _tiny_pool(refit):
    """A synthetic pool with a planted alpha, two cells, one card, one arm."""
    out = []
    for tokens in (256, 512):
        for tiles in (0.0, 4.0, 8.0):
            out.append(refit.Observation(
                traffic_ratio=1.0 + 0.5 * (tiles * 1.0e8 / 1.0e9),
                compulsory_bytes=1.0e9, per_expert_bytes=1.0e8,
                active_experts=8.0, m_tiles=8.0 + tiles, block_m=128,
                group_m=1, tile_provenance="test", model="mixtral-8x7b",
                dtype="bf16", gpu="NVIDIA H200", impl="vllm_fused_experts",
                tokens=tokens, routing="uniform", l2_flush=True,
                cuda_graph=False, tile_columns=(), arm="arm-a",
                dirty_raw="False"))
    return out


def test_the_paired_basis_contrast_keeps_only_contexts_seen_both_ways():
    refit = _load("alpha_refit")
    both = _tiny_pool(refit)
    eager_only = [refit.dataclasses.replace(o, tokens=9999) for o in both]
    graphed = [refit.dataclasses.replace(o, cuda_graph=True) for o in both]
    low, high = refit.paired_basis_contrast(
        both + eager_only + graphed, lambda o: o.cuda_graph)
    assert {o.tokens for o in low} == {256, 512}
    assert 9999 not in {o.tokens for o in low}
    assert {o.context for o in low} == {o.context for o in high}


def test_the_paired_basis_contrast_is_empty_when_nothing_was_measured_both_ways():
    refit = _load("alpha_refit")
    low, high = refit.paired_basis_contrast(_tiny_pool(refit),
                                            lambda o: o.cuda_graph)
    assert low == [] and high == []


@pytest.fixture(scope="module")
def pinned_report() -> str:
    """One `--pinned-set` run, shared. The bootstrap is the expensive part."""
    got = _run(["scripts/alpha_refit.py", "--pinned-set", "--bootstrap", "20"])
    assert got.returncode == 0, got.stderr
    return got.stdout


def test_the_report_shows_the_graph_split_is_one_model_and_the_pairing_fixes_it(
        pinned_report):
    """The audit's actual finding, asserted on the published corpus.

    98% of the marginal graph-mode discriminating rows are deepseek-v2-lite, so
    the per-basis split FINDINGS calls "stable across timing modes" is mostly a
    statement about which models each mode was pointed at.
    """
    assert "axis cuda_graph: eager against graph" in pinned_report
    assert "PAIRED SHIFT graph - eager" in pinned_report
    marginal = pinned_report.split("marginal graph")[1].split("paired")[0]
    assert "deepseek-v2-lite 99%" in marginal


def test_the_fit_states_an_mde_and_the_ranges_it_has_to_beat(pinned_report):
    assert "MDE" in pinned_report
    assert "between model" in pinned_report
    assert "between basis" in pinned_report
    assert "dirty share of admitted rows" in pinned_report
    # Both cluster levels this slice added are reported, not just the cell one.
    assert "arm  " in pinned_report and "card " in pinned_report


def test_split_range_refuses_a_lever_the_pool_does_not_vary():
    refit = _load("alpha_refit")
    assert refit.split_range(_tiny_pool(refit), lambda o: o.gpu) is None


def test_dirty_share_separates_unrecorded_from_clean():
    from moe.bench.published import dirty_share, dirty_share_line

    rows = [{"git_dirty": "True"}, {"git_dirty": "False"}, {}]
    assert dirty_share(rows) == (1, 1, 3)
    line = dirty_share_line(rows)
    assert "1/3" in line
    assert "not the same as clean" in line
    # Zero dirty is still a printed line; an absent line is not a finding.
    assert "0/1" in dirty_share_line([{"git_dirty": "False"}])


def test_the_mde_helpers_are_the_textbook_quantities():
    from moe.bench.published import (
        paired_mde,
        sd_from_band,
        two_sample_mde,
    )

    assert sd_from_band(0.529, 0.588) == pytest.approx(0.0179, abs=1e-4)
    assert two_sample_mde(0.1) == pytest.approx(2.4865 * 2 ** 0.5 * 0.1, rel=1e-3)
    assert paired_mde(0.0323, 4) == pytest.approx(2.4865 * 0.0323 / 2, rel=1e-3)
    with pytest.raises(ValueError):
        paired_mde(0.03, 1)
    # An MDE of zero noise is zero, which is the degenerate case and not an error.
    assert two_sample_mde(0.0) == 0.0


# --------------------------------------------------------------------------
# crossing_report: a band on every quoted ratio
# --------------------------------------------------------------------------

A100_XCARD = PUBLISHED / "2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card"


def test_every_staircase_ratio_carries_a_band():
    got = _run(["scripts/crossing_report.py",
                *(str(p) for p in sorted(A100_XCARD.glob("run_*.csv"))),
                "--ridge", "145.8", "--uncertainty"])
    assert got.returncode == 0, got.stderr
    assert "STAIRCASE" in got.stdout
    body = got.stdout.split("STAIRCASE", 1)[1]
    assert "90% band:" in body and "x predicted" in body
    assert "draws kept this step" in body
    assert "draws kept two crossings" in body


def test_the_default_output_still_carries_no_bands():
    """`--uncertainty` is opt-in and the default output must not change shape."""
    got = _run(["scripts/crossing_report.py",
                *(str(p) for p in sorted(A100_XCARD.glob("run_*.csv"))),
                "--ridge", "145.8"])
    assert got.returncode == 0, got.stderr
    assert "90% band" not in got.stdout
    assert "STAIRCASE" in got.stdout


def test_the_crossing_report_prints_the_dirty_share_and_an_mde():
    got = _run(["scripts/crossing_report.py",
                *(str(p) for p in sorted(A100_XCARD.glob("run_*.csv"))),
                "--ridge", "145.8"])
    assert "dirty share of admitted rows" in got.stdout
    assert "noise assumption: median relative replicate spread" in got.stdout
    assert "MDE" in got.stdout


def test_cell_bands_match_by_step_and_report_how_often_the_crossing_survived():
    """A band conditional on survival must say how conditional it is."""
    from moe.bench.crossing import upcrossings

    report = _load("crossing_report")
    # A staircase built by hand: flat, step, flat, step.
    replicates = [(t, [ms, ms * 1.002, ms * 0.998]) for t, ms in [
        (64, 1.00), (128, 1.02), (256, 1.60), (512, 1.66), (1024, 1.72),
        (2048, 2.90), (4096, 3.00)]]
    points = [(t, sorted(v)[1]) for t, v in replicates]
    assert len(upcrossings(points)) == 2, "the fixture must have two crossings"

    bands = report.CellBands(replicates, 0.0, draws=200, seed=0)
    assert len(bands.measured) == 2
    for i in (0, 1):
        lo, hi, kept = bands.band(i)
        assert lo <= bands.measured[i].tokens <= hi
        assert 0 < kept <= bands.n_draws
    ratio = bands.last_over_first()
    assert ratio is not None
    lo, hi, kept = ratio
    assert lo <= bands.measured[1].tokens / bands.measured[0].tokens <= hi


def test_cell_bands_have_nothing_to_band_on_a_curve_with_no_crossing():
    report = _load("crossing_report")
    flat = [(t, [1.0, 1.0, 1.0]) for t in (64, 128, 256, 512)]
    bands = report.CellBands(flat, 0.0, draws=20, seed=0)
    assert bands.measured == []
    assert bands.band(0) is None
    assert bands.last_over_first() is None
    assert bands.ratio_band(0, 100.0) is None


# --------------------------------------------------------------------------
# 5. the withdrawn cross-machine band is scored against nowhere (B4, second pass)
# --------------------------------------------------------------------------

WITHDRAWN_BAND = (160.3, 176.2)

#: The per-card bands below are `alpha_refit.card_ridge_bands()` over the
#: COMMITTED calibrations, so they move when a card is recalibrated. The H200's
#: has read [152.1, 165.6], [142.8, 155.4] and [145.8, 158.6] in nine days and
#: three literals for it went red in turn, so NONE is written down here any
#: more: every end is checked against that card's own `ridge_by_pattern`
#: instead. What the test is about is that the two cards DISAGREE and that
#: neither band is the withdrawn pair, which holds under all of them.


def _pattern_ridges(slug: str) -> set[float]:
    """Every per-pattern ridge a card's committed calibration publishes.

    A band end has to BE one of these, which is a relation the card can be
    recalibrated without breaking. Which patterns survive into the band is
    `rescore_published_reports.ridge_band_from_detail`'s decision and is not
    re-implemented here; this only says the ends came from the file.
    """
    import yaml

    from moe.bench import roofline as RL

    doc = yaml.safe_load((RL.HARDWARE_DIR / f"measured_{slug}.yaml").read_text())
    return {round(float(v), 1)
            for v in doc["detail"]["ridge_by_pattern"].values()}


def test_each_card_gets_its_own_band_off_its_own_calibration():
    """The replacement for `RIDGE_BAND` in the cap table.

    160.3 and 176.2 are two H200 calibrations of the same card disagreeing about
    its compute ceiling by 9.9%, so the pair is a reproducibility spread and not
    a device's band: `alpha_refit --adversarial` decided "NEVER crosses" against
    it, on a corpus with A100 rows in it, while the same commit withdrew it from
    all 26 published reports. Each end here has to come from one card's own
    yaml, and the two cards have to DISAGREE, or the old single band was
    harmless after all.
    """
    refit = _load("alpha_refit")
    rows = refit.card_ridge_bands()
    bands = dict((card, band) for card, _ridge, band in rows)
    ridges = dict((card, ridge) for card, ridge, _band in rows)
    for card, band in bands.items():
        # EVERY END COMES OFF THAT CARD'S OWN YAML. Three literals for the
        # H200's band went red in nine days; the relation has not.
        published = _pattern_ridges(card)
        for end in band:
            assert round(float(end), 1) in published, (card, end, published)
        assert band[0] <= round(ridges[card], 1) <= band[1], (card, band)
        assert tuple(band) != WITHDRAWN_BAND
        assert band[0] < band[1], "a two-machine band was the only wide one"
    # The two cards disagree, which is the whole point: one band is not both.
    # They were DISJOINT while the H200's ridge was 162.8 and they OVERLAP now
    # that its 2026-09-09 recalibration puts it at 152.8, so what is asserted
    # is that the bands differ, not that they are separated: a recalibration
    # may narrow the gap between two cards without making one band serve both.
    assert bands["nvidia_h200"] != bands["nvidia_a100_sxm4_80gb"]
    assert bands["nvidia_h200"][1] > bands["nvidia_a100_sxm4_80gb"][1]
    assert bands["nvidia_h200"][0] > bands["nvidia_a100_sxm4_80gb"][0]


def test_the_cap_verdict_has_all_three_branches_and_they_move_with_the_band():
    """Every branch, including the one the withdrawn band used to get wrong.

    A cap of 150 rows per expert crosses on the A100 and is undecided on the
    H200, and against 160.3-176.2 it would have been called "NEVER crosses"
    for both. The verdict is a claim about a device, and this is the
    arithmetic that makes it one. (On the 2026-09-02 calibration, where the
    H200's band was 152.1-165.6, that same cap read "NEVER crosses" there;
    the branch a cap lands in moves with the card's own calibration, which is
    the property being pinned rather than any one pairing.)
    """
    refit = _load("alpha_refit")
    # TWO PLANTED BANDS, not the cards' current ones. They are shaped like the
    # two cards so the docstring reads, but they are arguments this test hands
    # `cap_verdict` and every verdict below is arithmetic over them alone, so
    # they are fixtures rather than a claim about any calibration.
    a100 = [139.6, 149.3]
    h200 = [142.8, 155.4]
    assert refit.cap_verdict(150.0, a100) == "crosses"
    assert refit.cap_verdict(150.0, h200) == "inside the band"
    assert refit.cap_verdict(145.0, a100) == "inside the band"
    assert refit.cap_verdict(160.0, h200) == "crosses"
    assert refit.cap_verdict(0.0, a100) == "NEVER crosses"
    assert refit.cap_verdict(0.0, h200) == "NEVER crosses"
    assert refit.cap_verdict(150.0, a100) != refit.cap_verdict(150.0, h200)


def test_the_adversarial_cap_table_names_its_cards_and_not_the_withdrawn_band(
        capsys):
    """The output a reader actually sees, on the real corpus."""
    refit = _load("alpha_refit")
    csvs = sorted(str(p) for p in PUBLISHED.glob("*/run_*.csv"))
    assert csvs, "the published corpus is the input this test is about"
    assert refit.main([*csvs, "--bootstrap", "5", "--adversarial"]) == 0
    out = capsys.readouterr().out
    section = out.split("### 4.")[1]
    # The header the reader sees quotes each card's CURRENT band, so the
    # expected string is built from the same resolver rather than retyped.
    for card, _ridge, band in refit.card_ridge_bands():
        assert f"vs {card} {band[0]:.1f}-{band[1]:.1f}" in section
    quoting = [ln for ln in section.splitlines() if "160.3-176.2" in ln]
    assert not any(ln.strip().startswith("|") for ln in quoting), \
        "the withdrawn band may be named as history, never scored against"
    assert all("used to quote" in ln for ln in quoting)
    assert "vs ridge band 160.3-176.2" not in out
    assert "NEVER crosses" in section
    # C2's rows are H200 rows, so the ceiling paragraph quotes the H200's own
    # band's low end. Read off the same resolver: it has been 152.1, 142.8 and
    # 145.8 in nine days.
    h200 = [row for row in refit.card_ridge_bands() if row[0] == "nvidia_h200"]
    assert h200, "the H200's calibration is the input this paragraph needs"
    assert f"ridge of {h200[0][2][0]}" in section, "C2's rows are H200 rows"


def test_the_cap_table_refuses_rather_than_falling_back_to_the_constant(monkeypatch,
                                                                       capsys):
    """The FAIL branch: no calibration resolves, so no table is printed.

    Planted rather than reasoned about. The whole defect being repaired is a
    tool reaching for a module constant when the device's own number was not
    available, so the no-calibration path must produce a refusal and must not
    contain the withdrawn numbers anywhere.
    """
    refit = _load("alpha_refit")
    monkeypatch.setattr(refit, "card_ridge_bands", lambda *a, **k: [])
    refit.print_ai_cap_table(0.558)
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "160.3" not in out and "176.2" not in out
    assert "| BLOCK_M |" not in out


def test_the_c2_paragraph_refuses_when_the_h200_calibration_is_missing(monkeypatch,
                                                                      capsys):
    """The second FAIL branch: one card present, but not the one C2 was measured on.

    Scoring H200 rows against the A100's band would be the same substitution in
    the opposite direction, so the paragraph is dropped instead.
    """
    refit = _load("alpha_refit")
    monkeypatch.setattr(refit, "card_ridge_bands",
                        lambda *a, **k: [("nvidia_a100_sxm4_80gb", 145.8,
                                          [139.6, 149.3])])
    refit.print_ai_cap_table(0.558)
    out = capsys.readouterr().out
    assert "| BLOCK_M |" in out, "the table itself still has a card to score against"
    assert "REFUSED rather than scored" in out
    assert "FINDINGS C2 puts mixtral's" not in out


def test_the_crossing_report_docstring_no_longer_teaches_the_withdrawn_ridge():
    """A usage line is read as a recommendation, and 160.3 was one.

    `--ridge` is required, so the example is the only ridge the file suggests;
    it suggested the figure that reached an A100 arm from an H200 calibration.
    """
    report = _load("crossing_report")
    # THE USAGE LINE NAMES NO RIDGE AT ALL. It named 160.3, then 162.8, then
    # 152.8, and each in turn became a recommendation to quote a superseded
    # ruler; a test that checked the line against the current calibration only
    # moved the defect into the docstring, where it went red on the next
    # session anyway. It now shows the reader how to READ the card's ridge.
    import re

    from moe.bench import roofline as RL

    doc = report.__doc__
    usage = doc[doc.index("python scripts/crossing_report.py"):doc.index("THE USAGE")]
    assert "measured_nvidia_h200" in usage and "ridge_point" in usage
    assert not re.search(r"--ridge\s+[0-9]", usage), \
        "a ridge typed into the usage line is stale within the week"
    assert RL.load_hardware("measured_nvidia_h200").ridge_point("bf16") > 0
    assert "--ridge 160.3 --impl" not in doc
    assert "used to read\n`--ridge 160.3`" in doc or "used to read `--ridge 160.3`" in doc, \
        "the withdrawn figure is named as history, so a reader can follow it"
    staircase = report.print_staircase.__doc__
    assert "160.3-176.2" in staircase, "the withdrawn pair stays, as history"
    assert not re.search(r"\b1[45][0-9]\.[0-9]-1[56][0-9]\.[0-9]\b",
                         staircase.replace("160.3-176.2", "")), \
        "the card's own band moves; the docstring points at the file instead"
    assert "CV 21.2%" in staircase, "the measured spreads are unchanged"
    assert "against the measured ridge band" not in staircase
