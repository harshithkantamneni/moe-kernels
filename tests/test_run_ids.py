"""Every run id names its card, and changes when any swept knob changes.

WHAT THIS IS ABOUT, in one sentence: two runs that differ in anything that
moves a number must not land in the same directory, because every one of these
scripts RESUMES by default and a resume into the wrong directory is silent.

THE INCIDENT, which is in the repository and not hypothetical. The sibling
sweep's run id omitted the card. `results/published/2026-09-01-nvidia_h200-
cross-card-s3` and `results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-
surface-s3` both contain `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json`:
one id, two cards, `sm_count` 132 against 108, ridge 162.8 against 145.8. The
resume keys of these scripts carry no device either -- `(model, tokens, arm,
dtype)` in dtype_tile_confound, `r["id"]` in group_m_alpha_sweep, `rung.key` in
alias_ablation -- so the second card finds every measurement present, spends no
GPU time, and reports the first card's timings scored against its own ceilings.
Nothing downstream looks wrong.

The results root is what makes it reachable rather than theoretical: it is
`$MOE_RESULTS_DIR`, else `/workspace/results`, a network volume the runbook uses
BECAUSE it outlives the pod. Two pods with two cards share it by design.

THE SECOND OMISSION IS THE TIMING KNOBS. `--warmup`, `--trials`, the trial
budget and the L2 flush state each set the measured milliseconds of every row.
A re-run at a different one landing in the same directory prints the old
numbers under the new label, and the report renders the arguments from argv
rather than from the rows it read, so the label is the new one.

Every case below plants BOTH branches: the id changes when a knob changes, and
it does NOT change when nothing does, because an id that changed on every call
would make resume dead code and hide the same failure the other way round.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import provenance as PV  # noqa: E402

H200 = "NVIDIA H200"
A100 = "NVIDIA A100-SXM4-80GB"


def _load(name: str):
    """Load a script by path. `scripts/` is not a package and never has been.

    Registered in `sys.modules` BEFORE exec because `@dataclass` resolves its
    annotations through `sys.modules[cls.__module__]`, and a module that is not
    there yet fails with an AttributeError about NoneType naming nothing useful.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RULER = _load("ruler_rebaseline")
GROUP_M = _load("group_m_alpha_sweep")
ALIAS = _load("alias_ablation")
TILE = _load("tile_sweep")


# --------------------------------------------------------------------------
# ruler_rebaseline: the one whose collision DESTROYED data rather than hid it
# --------------------------------------------------------------------------

def _ruler_args(**over):
    args = RULER.build_parser().parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_the_ruler_run_id_carries_the_card_slug():
    args = _ruler_args(card=H200)
    assert RULER.default_run_id(args).startswith("nvidia_h200-")


def test_two_cards_no_longer_share_the_rulers_output_directory():
    """This one did not collide quietly: it overwrote.

    `main` writes `report.txt` and `report.json` with `write_text`, which
    truncates. Measuring the H200 and then the A100 with the same flags left one
    report, and it was the second card's.
    """
    h200 = RULER.default_run_id(_ruler_args(card=H200))
    a100 = RULER.default_run_id(_ruler_args(card=A100))
    assert h200 != a100
    assert a100.startswith("nvidia_a100_sxm4_80gb-")


@pytest.mark.parametrize("knob,value", [
    ("card", A100), ("buffer_gb", 4.0), ("gemm_n", 4096),
    ("ceiling", "read_reduce"), ("settle_seconds", 10.0), ("settle", False),
    ("corpus_only", True), ("corpus", Path("/somewhere/else")),
])
def test_every_ruler_knob_changes_the_run_id(knob, value):
    base = _ruler_args(card=H200)
    moved = _ruler_args(**{"card": H200, **{knob: value}})
    assert RULER.default_run_id(base) != RULER.default_run_id(moved), knob


def test_the_ruler_run_id_is_stable_when_nothing_moves():
    """Resume is the default, so an id that changed per call would make the
    resume path dead code the first time anyone used it."""
    assert (RULER.default_run_id(_ruler_args(card=H200))
            == RULER.default_run_id(_ruler_args(card=H200)))


def test_a_run_that_measures_nothing_is_labelled_as_having_no_card():
    """`--corpus-only` prices published rows from several devices at once.

    There is no card, and `provenance.run_id` refuses an id without one, so the
    honest answer is a label no nvidia-smi can produce rather than a real card
    name attached to a run that measured neither.
    """
    assert RULER.resolve_card(_ruler_args()) == RULER.NO_CARD
    assert "no_card" in RULER.default_run_id(_ruler_args())
    assert RULER.NO_CARD not in {H200, A100}


def test_the_ruler_refuses_to_measure_without_a_card(tmp_path, capsys):
    """The FAIL branch: gates 1 to 3 score a fresh calibration against THIS
    card's registered constants, so an unnamed card makes them meaningless."""
    from moe.bench import exit_codes
    code = RULER.main(["--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert "REFUSED" in out and "--card" in out


# --------------------------------------------------------------------------
# group_m_alpha_sweep
# --------------------------------------------------------------------------

def _gm(**over):
    args = GROUP_M.parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args, GROUP_M.build_plan(args)


def test_the_group_m_run_id_carries_the_card_slug():
    args, plan = _gm()
    assert GROUP_M.default_run_id(args, H200, plan).startswith("nvidia_h200-")


@pytest.mark.parametrize("knob,value", [
    ("warmup", 900.0), ("trials", 7), ("l2_flush", False),
    ("model", "qwen2-57b-a14b"), ("block_m", 32), ("passes", 2),
    ("seeds", 2), ("tokens", (16, 32)), ("group_m", (1, 16)),
    ("routings", ("uniform",)),
])
def test_every_group_m_knob_changes_the_run_id(knob, value):
    """The last four reach the id through `plan.fingerprint`, the first three
    only through this function; both routes are exercised so neither can be
    dropped without a failure here."""
    base_args, base_plan = _gm()
    moved_args, moved_plan = _gm(**{knob: value})
    assert (GROUP_M.default_run_id(base_args, H200, base_plan)
            != GROUP_M.default_run_id(moved_args, H200, moved_plan)), knob


def test_two_cards_no_longer_share_the_group_m_directory():
    args, plan = _gm()
    assert (GROUP_M.default_run_id(args, H200, plan)
            != GROUP_M.default_run_id(args, A100, plan))


def test_the_group_m_run_id_is_stable_when_nothing_moves():
    args, plan = _gm()
    assert (GROUP_M.default_run_id(args, H200, plan)
            == GROUP_M.default_run_id(args, H200, plan))


def test_the_group_m_analysis_knobs_stay_out_of_the_run_id():
    """`--bootstrap` and `--seed` re-analyse a set of measurements rather than
    change one, so two analyses of one sweep belong in one directory."""
    base_args, base_plan = _gm()
    for knob, value in (("bootstrap", 999), ("seed", 7)):
        moved_args, moved_plan = _gm(**{knob: value})
        assert (GROUP_M.default_run_id(base_args, H200, base_plan)
                == GROUP_M.default_run_id(moved_args, H200, moved_plan)), knob


# --------------------------------------------------------------------------
# alias_ablation
# --------------------------------------------------------------------------

def _ab(**over):
    args = ALIAS.parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args, ALIAS.build_design(args)


def test_the_alias_run_id_carries_the_card_slug():
    args, design = _ab()
    assert ALIAS.default_run_id(args, H200, design).startswith("nvidia_h200-")


@pytest.mark.parametrize("knob,value", [
    ("seed", 11), ("l2_flush", False), ("compute", "dot"), ("block_m", 32),
    ("replicates", 3), ("tiles", (1, 2)), ("models", ("mixtral-8x7b",)),
])
def test_every_alias_knob_changes_the_run_id(knob, value):
    base_args, base_design = _ab()
    moved_args, moved_design = _ab(**{knob: value})
    assert (ALIAS.default_run_id(base_args, H200, base_design)
            != ALIAS.default_run_id(moved_args, H200, moved_design)), knob


def test_two_cards_no_longer_share_the_alias_directory():
    """The subject of this experiment is L2 behaviour and the two cards' L2
    differ by 20 MiB, so a shared directory is not a small error here."""
    args, design = _ab()
    assert (ALIAS.default_run_id(args, H200, design)
            != ALIAS.default_run_id(args, A100, design))


def test_the_alias_run_id_is_stable_when_nothing_moves():
    args, design = _ab()
    assert (ALIAS.default_run_id(args, H200, design)
            == ALIAS.default_run_id(args, H200, design))


def test_the_alias_bootstrap_stays_out_of_the_run_id():
    base_args, base_design = _ab()
    moved_args, moved_design = _ab(bootstrap=99)
    assert (ALIAS.default_run_id(base_args, H200, base_design)
            == ALIAS.default_run_id(moved_args, H200, moved_design))


# --------------------------------------------------------------------------
# tile_sweep
# --------------------------------------------------------------------------

def _tile(**over):
    args = TILE.build_parser().parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_the_tile_sweep_run_id_carries_the_card_slug():
    assert TILE.default_run_id(_tile(), H200).startswith("nvidia_h200-")


@pytest.mark.parametrize("knob,value", [
    ("model", "mixtral-8x7b"), ("tokens", "16,64"), ("tiles", "16,32"),
    ("seed", 3), ("iters", 100), ("warmup", 900.0),
    ("cell_budget_ms", 400.0), ("trials", 7), ("no_l2_flush", True),
])
def test_every_tile_sweep_knob_changes_the_run_id(knob, value):
    assert (TILE.default_run_id(_tile(), H200)
            != TILE.default_run_id(_tile(**{knob: value}), H200)), knob


def test_two_cards_no_longer_share_the_tile_sweep_directory():
    assert TILE.default_run_id(_tile(), H200) != TILE.default_run_id(_tile(), A100)


def test_the_dump_ptx_flag_stays_out_of_the_tile_sweep_run_id():
    """It adds a column to the report and does not change a time, so two runs
    that differ only there belong together."""
    assert (TILE.default_run_id(_tile(), H200)
            == TILE.default_run_id(_tile(dump_ptx=Path("/tmp/ptx")), H200))


def test_the_tile_sweep_refuses_a_run_with_no_card():
    with pytest.raises(PV.NoCard):
        TILE.default_run_id(_tile(), "")


def test_the_tile_sweep_refuses_off_gpu_rather_than_naming_a_card(capsys):
    """R6. There is no CUDA device in this process, which is the live case."""
    from moe.bench import exit_codes
    with pytest.raises(TILE.NoCardToLabel):
        TILE.resolve_card(_tile())
    assert TILE.main(["--dry-run"]) == exit_codes.REFUSED
    assert "NoCardToLabel" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the rule itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", [RULER, GROUP_M, ALIAS, TILE])
def test_every_run_id_goes_through_the_shared_builder(module):
    """Three of these had each re-implemented a subset of `provenance.run_id`
    and each had left a different knob out.

    Checked in the source rather than by behaviour because the failure mode is a
    NEW private hash appearing beside the shared one, which behaves identically
    until the day it does not.
    """
    source = Path(module.__file__).read_text()
    body = source.split("def default_run_id")[1].split("\ndef ")[0]
    assert "PV.run_id(" in body, "the id is not built by the shared function"
    assert "hashlib" not in body, "a private hash is back in the run id"
