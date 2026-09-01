"""The fallback-config price, and the four ways this measurement could lie.

`scripts/tuned_vs_fallback.py` answers one question -- what does it cost to run
vLLM's hardcoded fallback config instead of a tuned one -- and the answer is a
ratio. A ratio has exactly four failure modes worth a test file, and every test
below belongs to one of them:

1. THE SIGN IS BACKWARDS. `penalty = fallback / tuned` with `> 1` meaning the
   fallback is slower. Inverted, the study's conclusion flips from "the untuned
   kernel costs 15%" to "vLLM's own tuning is harmful", which is a far bigger
   claim and would be made by accident. Every printed penalty therefore carries
   the word SLOWER or FASTER, and the tests read those words rather than the
   number.
2. THE TWO SIDES ARE THE SAME KERNEL. If `override_config` forced nothing, all
   arms would run the native config and the script would report a beautifully
   tight 1.000 that means "the experiment did not happen". So the run watches
   vLLM's own lookup, and the gates that check it must read UNKNOWN rather than
   PASS when nothing was watched.
3. THE GAP IS ATTRIBUTED TO THE WRONG KNOB. The two configs differ in up to five
   places at once. The plan tests below pin the fact that makes the tile-height
   story implausible before any GPU exists: on H200 bf16, BLOCK_SIZE_M AGREES
   between the two sides in 19 of the 28 default cells, so in those cells the
   penalty is 0% tile height by construction.
4. THE RUN IS LOST. A pod is billed by the minute and dies on teardown, so the
   store must round-trip a written arm, and the run id must be a function of the
   plan so that re-running the same command resumes instead of starting over.

Every timing here is SYNTHETIC. Nothing in this file needs a GPU, vLLM, or a
published CSV, which is the point: the reduction, the sign, the gates and the
persistence are all testable the day before the pod goes up, and the only thing
the pod adds is the numbers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    """Load the script by path. `scripts/` is not a package and never has been.

    Registered in sys.modules BEFORE exec for the same reason
    tests/test_alpha_refit.py does it: `@dataclass` resolves its annotations
    through `sys.modules[cls.__module__]`, and a module that is not there yet
    fails with an AttributeError about NoneType that names nothing useful.
    """
    spec = importlib.util.spec_from_file_location(
        "tuned_vs_fallback", ROOT / "scripts" / "tuned_vs_fallback.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TVF = _load_script()

H200 = "NVIDIA H200"


def make_cell(model: str = "mixtral-8x7b", tokens: int = 32,
              gpu: str = H200) -> TVF.Cell:
    """One planned cell, built the way the script builds it and not by hand.

    Hand-building a Cell would let a test assert against a config the planner
    would never produce, which is how a green suite ends up describing a
    different experiment than the one that runs.
    """
    cells, _ = TVF.plan_cells([model], [tokens], "bf16", gpu)
    assert cells, f"{model} T={tokens} did not plan on {gpu}"
    return cells[0]


def arm(cell, name: str, ms: float | None, **kw) -> TVF.ArmResult:
    """A synthetic ArmResult with a planted median time."""
    config = TVF.arm_config(cell, name)
    result = TVF.ArmResult(cell.model, cell.num_tokens, name, config,
                           config_origin="forced" if config else "observed",
                           **kw)
    if ms is not None:
        TVF.summarise_samples(result, [ms])
    return result


def planted(cell, *, tuned: float, fallback: float, replica: float | None = None,
            knobs: dict[str, float] | None = None,
            observed_native: dict | None = "derived") -> dict:
    """A full arm set for one cell with every time planted.

    `observed_native` defaults to the cell's own derived tuned config, i.e. the
    world in which vLLM loaded exactly what `tile_resolve` predicted, because
    that is the world every other test wants to hold fixed while it varies one
    thing.
    """
    seen = cell.tuned if observed_native == "derived" else observed_native
    arms = {
        "native": arm(cell, "native", tuned, observed_config=seen),
        "tuned": arm(cell, "tuned", tuned, observed_config=dict(cell.tuned),
                     override_verified=True),
        "fallback": arm(cell, "fallback", fallback,
                        observed_config=dict(cell.fallback),
                        override_verified=True, rel_err_vs_native=1e-3),
        "replica": arm(cell, "replica", tuned if replica is None else replica,
                       observed_config=seen),
    }
    for name in TVF.KNOB_GROUPS:
        if TVF.arm_is_identical_to_tuned(cell, name):
            arms[name] = TVF.ArmResult(cell.model, cell.num_tokens, name,
                                       TVF.arm_config(cell, name),
                                       config_origin="derived",
                                       identical_to_tuned=True)
        else:
            ms = (knobs or {}).get(name, tuned)
            arms[name] = arm(cell, name, ms,
                             observed_config=TVF.arm_config(cell, name),
                             override_verified=True, rel_err_vs_native=1e-3)
    return arms


# --------------------------------------------------------------------------
# 1. the sign
# --------------------------------------------------------------------------

def test_a_penalty_above_one_says_the_fallback_is_the_slower_side():
    text = TVF.penalty_sentence(1.23)
    assert "SLOWER" in text and "FASTER" not in text
    assert "23.0%" in text


def test_a_penalty_below_one_says_the_fallback_won_and_names_the_loser():
    text = TVF.penalty_sentence(0.8)
    assert "FASTER" in text and "SLOWER" not in text
    assert "tuned file LOST" in text


def test_a_planted_twenty_percent_penalty_comes_back_as_twenty_percent():
    """The reduction must not invert or rescale what was planted."""
    cell = make_cell(tokens=32)
    got = TVF.analyse([cell], {cell.key: planted(cell, tuned=1.0, fallback=1.2)})
    assert got.headline == pytest.approx(1.2)
    assert TVF.verdict_of(got.headline) == "MATERIAL"
    assert "SLOWER" in TVF.render_headline(got)


def test_the_headline_paragraph_spells_out_which_side_pays():
    cell = make_cell(tokens=32)
    got = TVF.analyse([cell], {cell.key: planted(cell, tuned=1.0, fallback=1.2)})
    text = TVF.render_headline(got)
    assert "no tuned vLLM config" in text
    assert "20.0% more time" in text


def test_the_headline_is_a_median_of_per_cell_ratios_not_of_pooled_times():
    """A pooled ratio would be the largest cell wearing a costume.

    Two cells three orders of magnitude apart in absolute time, with OPPOSITE
    penalties: pooling the milliseconds answers ~1.10 because the big cell owns
    the sum, while the median of the two ratios is 1.05. The second is the one
    this script must report.
    """
    small = make_cell(tokens=32)
    big = make_cell(tokens=4096)
    results = {small.key: planted(small, tuned=1.0, fallback=1.5),
               big.key: planted(big, tuned=1000.0, fallback=1100.0)}
    got = TVF.analyse([small, big], results)
    pooled = (1.5 + 1100.0) / (1.0 + 1000.0)
    assert got.headline == pytest.approx(1.3, abs=1e-9)
    assert abs(got.headline - pooled) > 0.15


# --------------------------------------------------------------------------
# 2. the two sides must be different kernels, and provably so
# --------------------------------------------------------------------------

def test_a_gate_nobody_could_evaluate_reads_unknown_and_never_pass():
    """G1 with nothing observed is the shape every retraction here began in."""
    cell = make_cell(tokens=32)
    arms = planted(cell, tuned=1.0, fallback=1.2, observed_native=None)
    for one in arms.values():
        one.observed_config = None
        one.override_verified = None
    got = TVF.analyse([cell], {cell.key: arms})
    gates = {g.name.split()[0]: g for g in TVF.build_gates(got)}
    assert gates["G1"].passed is None
    assert gates["G2"].passed is None
    assert "unchecked" in gates["G1"].observed
    assert "UNKNOWN" in TVF.render_gates(list(gates.values()))


def test_a_config_vllm_did_not_actually_load_fails_the_derivation_gate():
    cell = make_cell(tokens=32)
    wrong = dict(cell.tuned, BLOCK_SIZE_M=cell.tuned["BLOCK_SIZE_M"] * 2)
    got = TVF.analyse([cell], {cell.key: planted(cell, tuned=1.0, fallback=1.2,
                                                 observed_native=wrong)})
    gate = next(g for g in TVF.build_gates(got) if g.name.startswith("G1"))
    assert gate.passed is False
    assert "BLOCK_SIZE_M" in gate.observed


def test_an_override_that_did_not_take_effect_fails_its_own_gate():
    """Without this the whole script can report 1.000 and mean nothing."""
    cell = make_cell(tokens=32)
    arms = planted(cell, tuned=1.0, fallback=1.0)
    arms["fallback"].override_verified = False
    arms["fallback"].observed_config = dict(cell.tuned)
    got = TVF.analyse([cell], {cell.key: arms})
    gate = next(g for g in TVF.build_gates(got) if g.name.startswith("G2"))
    assert gate.passed is False
    assert "not given" in gate.observed


def test_a_noisy_box_fails_the_placebo_gate_whatever_the_penalty_says():
    cell = make_cell(tokens=32)
    got = TVF.analyse([cell], {cell.key: planted(cell, tuned=1.0, fallback=1.5,
                                                 replica=1.2)})
    gates = {g.name.split()[0]: g for g in TVF.build_gates(got)}
    assert gates["G3"].passed is False
    assert got.placebo_band == pytest.approx(0.2)


def test_an_arm_that_computed_a_different_layer_fails_the_same_layer_gate():
    cell = make_cell(tokens=32)
    arms = planted(cell, tuned=1.0, fallback=1.2)
    arms["fallback"].rel_err_vs_native = 0.5
    got = TVF.analyse([cell], {cell.key: arms})
    gate = next(g for g in TVF.build_gates(got) if g.name.startswith("G0"))
    assert gate.passed is False


# --------------------------------------------------------------------------
# 3. the plan, and the fact that makes the tile-height story implausible
# --------------------------------------------------------------------------

def test_block_size_m_agrees_between_the_two_sides_in_most_default_cells():
    """19 of 28. In those cells the penalty is 0% tile height BY CONSTRUCTION.

    This is the reason G6 is registered as a prediction rather than discovered
    afterwards, and it is arithmetic over vLLM's shipped file, so it is knowable
    today. If a future vLLM snapshot moves it, this test says so before the pod
    is rented.
    """
    cells, _ = TVF.plan_cells(list(TVF.TUNED_H200_MODELS),
                              list(TVF.DEFAULT_TOKENS), "bf16", H200)
    assert len(cells) == 28
    agree = [c for c in cells
             if c.tuned["BLOCK_SIZE_M"] == c.fallback["BLOCK_SIZE_M"]]
    assert len(agree) == 19
    for cell in agree:
        assert TVF.arm_is_identical_to_tuned(cell, "bm")


def test_the_ladder_pins_group_size_m_to_one_across_the_whole_decode_range():
    """The mechanism G7 predicts, stated as a property of vLLM's own branch.

    `get_default_config` sets GROUP_SIZE_M to 16 only when M//E > 128, so on
    E=64 it needs more than 8192 tokens. Every decode cell therefore runs the
    swizzle width that today's refit measured the WORST L2 reuse at.
    """
    cells, _ = TVF.plan_cells(["qwen2-57b-a14b"], list(TVF.DEFAULT_TOKENS),
                              "bf16", H200)
    for cell in cells:
        assert cell.fallback["GROUP_SIZE_M"] == 1, cell.num_tokens
    differing = [c for c in cells if "group" in c.differing_groups]
    assert differing, "no cell would test G7 at all"


def test_the_fallback_side_is_vllms_ladder_and_not_restated_here():
    """The script must not carry its own copy of the ladder.

    A second transcription of `M<=32 -> 16, M<=96 -> 32, ...` is a second thing
    to keep in sync with upstream, and the one that drifts is always the copy.
    """
    from moe.bench.tile_resolve import default_config
    for tokens in (1, 32, 33, 96, 97, 512, 513):
        cell = make_cell("mixtral-8x7b", tokens)
        assert cell.fallback == default_config(tokens, 8, None)


def test_a_model_with_no_tuned_file_on_this_card_is_dropped_with_a_reason():
    """deepseek-v3 is the 79.2% case and is unmeasurable for that exact reason."""
    cells, notes = TVF.plan_cells(["deepseek-v3"], [64], "bf16", H200)
    assert cells == []
    assert any("NO tuned file" in n and "79.2%" in n for n in notes)


def test_a_card_with_no_tuned_files_at_all_plans_nothing():
    cells, notes = TVF.plan_cells(list(TVF.TUNED_H200_MODELS), [64], "bf16",
                                  "NVIDIA A10G")
    assert cells == []
    assert len(notes) == 2


def test_a_cell_whose_two_configs_agree_is_excluded_rather_than_scored_as_free():
    """Counting it as a penalty of 1.0 would answer a different question.

    It would drag the headline toward 1 in proportion to how often vLLM's ladder
    happens to agree with its own tuned file, which is a fact about the file and
    not about what the ladder costs where it differs.
    """
    cell = make_cell(tokens=32)
    same = TVF.Cell(model=cell.model, num_tokens=cell.num_tokens,
                    dtype=cell.dtype, gpu_name=cell.gpu_name, tile=cell.tile,
                    tuned=dict(cell.tuned), fallback=dict(cell.tuned))
    got = TVF.analyse([same], {same.key: planted(same, tuned=1.0, fallback=1.9)})
    assert got.headline is None
    assert "nothing to price" in got.cells[0].excluded


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def test_each_knob_arm_moves_exactly_one_group_and_leaves_the_rest_tuned():
    cell = make_cell("mixtral-8x7b", 512)
    for name, keys in TVF.KNOB_GROUPS.items():
        forced = TVF.arm_config(cell, name)
        for key, value in cell.tuned.items():
            expected = cell.fallback[key] if key in keys else value
            assert forced[key] == expected, (name, key)


def test_the_native_arms_force_nothing_because_a_deployment_forces_nothing():
    cell = make_cell(tokens=32)
    assert TVF.arm_config(cell, "native") is None
    assert TVF.arm_config(cell, "replica") is None
    assert TVF.arm_config(cell, "fallback") == cell.fallback
    assert TVF.arm_config(cell, "tuned") == cell.tuned


def test_an_unknown_arm_name_raises_instead_of_silently_timing_the_tuned_config():
    cell = make_cell(tokens=32)
    with pytest.raises(KeyError):
        TVF.arm_config(cell, "whatever")


def test_an_arm_whose_knob_group_already_agrees_contributes_exactly_nothing():
    """Marked identical, not timed, and scored as 1.0 by construction."""
    cell = make_cell("qwen2-57b-a14b", 32)
    assert cell.differing_groups == ("warpstages",)
    arms = planted(cell, tuned=1.0, fallback=1.2)
    assert arms["bm"].identical_to_tuned
    assert arms["bm"].ms_median is None
    got = TVF.analyse([cell], {cell.key: arms})
    assert got.cells[0].knob["bm"] == 1.0


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------

def test_one_knob_carrying_the_whole_gap_is_reported_as_the_whole_gap():
    cell = make_cell("mixtral-8x7b", 16)
    assert "group" in cell.differing_groups
    arms = planted(cell, tuned=1.0, fallback=1.2,
                   knobs={"group": 1.2, "nk": 1.0, "warpstages": 1.0})
    got = TVF.analyse([cell], {cell.key: arms})
    assert got.knob_share_median["group"] == pytest.approx(1.0)
    assert got.knob_share_median["nk"] == pytest.approx(0.0)
    gates = {g.name.split()[0]: g for g in TVF.build_gates(got)}
    assert gates["G7"].passed is True
    assert gates["G6"].passed is True


def test_a_gap_that_is_all_tile_height_fails_the_tile_height_prediction():
    """G6 must be able to fail. It is a prediction, not a description."""
    cell = make_cell("mixtral-8x7b", 512)
    assert "bm" in cell.differing_groups
    arms = planted(cell, tuned=1.0, fallback=1.3,
                   knobs={"bm": 1.3, "nk": 1.0, "group": 1.0,
                          "warpstages": 1.0})
    got = TVF.analyse([cell], {cell.key: arms})
    gates = {g.name.split()[0]: g for g in TVF.build_gates(got)}
    assert gates["G6"].passed is False
    assert gates["G7"].passed is False


def test_the_decomposition_prints_a_residual_so_four_shares_are_not_read_as_all():
    """One-at-a-time effects are not a partition, and the table must say so."""
    cell = make_cell("mixtral-8x7b", 16)
    arms = planted(cell, tuned=1.0, fallback=1.4,
                   knobs={"group": 1.1, "nk": 1.1, "warpstages": 1.0})
    got = TVF.analyse([cell], {cell.key: arms})
    text = TVF.render_decomposition(got)
    assert "residual" in text
    assert "+50%" in text          # 0.40 total, 0.20 explained one at a time


def test_a_gap_that_does_not_exist_is_not_apportioned():
    assert TVF.knob_share(1.2, 1.0) is None
    assert TVF.knob_share(1.2, 0.9) is None
    assert TVF.knob_share(1.1, 1.2) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# the claim gates
# --------------------------------------------------------------------------

def _headline_gates(penalties: list[float]):
    cells, results = [], {}
    for tokens, penalty in zip(TVF.DEFAULT_TOKENS, penalties, strict=True):
        cell = make_cell("mixtral-8x7b", tokens)
        cells.append(cell)
        results[cell.key] = planted(cell, tuned=1.0, fallback=penalty)
    got = TVF.analyse(cells, results)
    return got, {g.name.split()[0]: g for g in TVF.build_gates(got)}


def test_a_three_percent_penalty_fails_the_material_gate_and_is_called_a_footnote():
    got, gates = _headline_gates([1.03] * len(TVF.DEFAULT_TOKENS))
    assert gates["G4"].passed is True          # still not harmful
    assert gates["G5"].passed is False         # but not a headline either
    assert TVF.verdict_of(got.headline) == "FOOTNOTE"
    assert "FOOTNOTE" in gates["G5"].observed


def test_a_twenty_percent_penalty_passes_the_material_gate():
    got, gates = _headline_gates([1.2] * len(TVF.DEFAULT_TOKENS))
    assert gates["G5"].passed is True
    assert TVF.verdict_of(got.headline) == "MATERIAL"


def test_a_wide_interval_fails_the_material_gate_even_when_the_median_clears_it():
    """The gate is on the interval's low end, not on the point estimate alone."""
    spread = [1.0, 1.0, 1.0, 1.0, 1.0, 1.16, 1.2, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9,
              2.0]
    got, gates = _headline_gates(spread)
    assert got.headline >= TVF.MATERIAL_PENALTY
    assert got.interval[0] < TVF.MATERIAL_PENALTY
    assert gates["G5"].passed is False


def test_a_fallback_that_wins_fails_the_sign_gate_rather_than_being_hidden():
    got, gates = _headline_gates([0.9] * len(TVF.DEFAULT_TOKENS))
    assert gates["G4"].passed is False
    assert TVF.verdict_of(got.headline) == "INVERTED"
    assert "FASTER" in gates["G4"].observed


def test_the_bootstrap_interval_is_deterministic_and_brackets_the_median():
    values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    first = TVF.bootstrap_interval(values)
    assert first == TVF.bootstrap_interval(values)
    assert first[0] <= 1.25 <= first[1]
    assert TVF.bootstrap_interval([1.2]) is None


def test_percentile_returns_a_number_that_was_actually_measured():
    values = [0.01, 0.02, 0.30]
    assert TVF.percentile(values, 0.90) in values
    assert TVF.percentile([], 0.5) is None


# --------------------------------------------------------------------------
# 4. surviving the teardown
# --------------------------------------------------------------------------

def test_the_run_id_is_a_hash_of_the_plan_so_the_same_command_resumes():
    args = (["mixtral-8x7b"], [1, 2], "bf16", H200, 3, 15, 0, "uniform")
    assert TVF.plan_run_id(*args) == TVF.plan_run_id(*args)


@pytest.mark.parametrize("index,value", [
    (0, ["qwen2-57b-a14b"]), (1, [1, 2, 4]), (2, "fp8_e4m3"),
    (3, "NVIDIA A100-SXM4-80GB"), (4, 5), (5, 30), (6, 1), (7, "zipf"),
])
def test_changing_any_plan_parameter_changes_the_run_id(index, value):
    """A changed plan must land in a new directory, never mix with old rows."""
    base = [["mixtral-8x7b"], [1, 2], "bf16", H200, 3, 15, 0, "uniform"]
    changed = list(base)
    changed[index] = value
    assert TVF.plan_run_id(*base) != TVF.plan_run_id(*changed)


def test_a_written_arm_comes_back_off_disk_with_its_numbers_intact(tmp_path):
    cell = make_cell(tokens=32)
    meta = {"run_id": "r", "gpu_name": H200, "vllm_version": "0.27.1",
            "torch_version": "2.13.0", "routing": "uniform", "seed": 0}
    store = TVF.Store(tmp_path / "timings.csv")
    original = arm(cell, "fallback", 1.25, override_verified=True,
                   rel_err_vs_native=1.5e-3,
                   observed_config=dict(cell.fallback))
    store.write(original, cell, meta)
    store.close()

    reopened = TVF.Store(tmp_path / "timings.csv")
    back = reopened.restore((cell.model, cell.num_tokens, "fallback"))
    reopened.close()
    assert back.ms_median == pytest.approx(1.25)
    assert back.config == cell.fallback
    assert back.override_verified is True
    assert back.rel_err_vs_native == pytest.approx(1.5e-3, rel=1e-3)
    assert back.observed_config == cell.fallback


def test_an_identical_arm_round_trips_as_identical_and_not_as_a_zero(tmp_path):
    """`identical_to_tuned` is written as an int and read back from text."""
    cell = make_cell("qwen2-57b-a14b", 32)
    meta = {"run_id": "r", "gpu_name": H200, "vllm_version": "", "seed": 0,
            "torch_version": "", "routing": "uniform"}
    store = TVF.Store(tmp_path / "t.csv")
    store.write(TVF.ArmResult(cell.model, cell.num_tokens, "bm",
                              TVF.arm_config(cell, "bm"),
                              config_origin="derived", identical_to_tuned=True),
                cell, meta)
    same_session = store.restore((cell.model, cell.num_tokens, "bm"))
    store.close()
    from_disk = TVF.Store(tmp_path / "t.csv")
    reread = from_disk.restore((cell.model, cell.num_tokens, "bm"))
    from_disk.close()
    assert same_session.identical_to_tuned is True
    assert reread.identical_to_tuned is True


def test_a_second_store_on_the_same_file_does_not_lose_the_first_ones_rows(tmp_path):
    """Resume is append, not truncate. A truncating open would cost a whole pod."""
    cell = make_cell(tokens=32)
    meta = {"run_id": "r", "gpu_name": H200, "vllm_version": "", "seed": 0,
            "torch_version": "", "routing": "uniform"}
    first = TVF.Store(tmp_path / "t.csv")
    first.write(arm(cell, "native", 1.0), cell, meta)
    first.close()
    second = TVF.Store(tmp_path / "t.csv")
    second.write(arm(cell, "fallback", 1.2), cell, meta)
    assert second.has((cell.model, cell.num_tokens, "native"))
    second.close()
    text = (tmp_path / "t.csv").read_text()
    assert text.count("run_id") == 1          # exactly one header
    assert "native" in text and "fallback" in text


def test_fresh_discards_the_old_rows_on_purpose(tmp_path):
    cell = make_cell(tokens=32)
    meta = {"run_id": "r", "gpu_name": H200, "vllm_version": "", "seed": 0,
            "torch_version": "", "routing": "uniform"}
    first = TVF.Store(tmp_path / "t.csv")
    first.write(arm(cell, "native", 1.0), cell, meta)
    first.close()
    second = TVF.Store(tmp_path / "t.csv", fresh=True)
    assert not second.has((cell.model, cell.num_tokens, "native"))
    second.close()


def test_the_results_path_prefers_the_volume_over_the_repo(monkeypatch):
    """A default under the checkout is a default that dies with the pod."""
    monkeypatch.setenv("MOE_RESULTS_DIR", "/workspace/results")
    assert TVF.default_out_dir() == Path("/workspace/results")
    monkeypatch.delenv("MOE_RESULTS_DIR")
    assert TVF.default_out_dir().is_absolute()


# --------------------------------------------------------------------------
# off the box
# --------------------------------------------------------------------------

def test_with_no_gpu_the_script_prints_the_plan_and_refuses_to_call_it_a_result(
        tmp_path, capsys):
    code = TVF.main(["--plan-only", "--out-dir", str(tmp_path),
                     "--tokens", "1,32,512"])
    out = capsys.readouterr().out
    assert code == TVF.EXIT_NOT_MEASURED
    assert code != TVF.EXIT_OK
    assert "NOT A RESULT" in out
    assert "Nothing was measured" in out
    # The plan is still there, and so is the sign convention and the prediction.
    assert "penalty = time(FALLBACK ladder config) / time(TUNED file config)" in out
    assert "Predictions, registered before the run" in out
    assert "EVERYTHING IS SAVED TO" in out


def test_the_plan_survives_to_disk_even_when_nothing_is_measured(tmp_path):
    TVF.main(["--plan-only", "--out-dir", str(tmp_path), "--tokens", "1,32"])
    plans = list(tmp_path.rglob("plan.json"))
    assert len(plans) == 1
    import json
    data = json.loads(plans[0].read_text())
    assert data["vllm_tag"] == TVF.VLLM_TAG
    assert len(data["cells"]) == 4
    assert all("tuned" in c and "fallback" in c for c in data["cells"])


def test_rerunning_the_same_command_lands_in_the_same_directory(tmp_path):
    for _ in range(2):
        TVF.main(["--plan-only", "--out-dir", str(tmp_path), "--tokens", "1,32"])
    assert len(list(tmp_path.iterdir())) == 1


def test_dropping_either_side_of_the_comparison_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        TVF.main(["--arms", "native,tuned", "--out-dir", str(tmp_path)])


def test_an_unexpected_vllm_version_is_called_out_rather_than_assumed_away():
    assert TVF.version_warning({"vllm_version": TVF.VLLM_TAG}) == ""
    assert TVF.version_warning({"vllm_version": "0.27.1"}) == ""
    assert "WARNING" in TVF.version_warning({"vllm_version": "0.28.0"})
    assert TVF.version_warning({"vllm_version": ""}) == ""


# --------------------------------------------------------------------------
# the census that motivates the whole thing
# --------------------------------------------------------------------------

def test_the_census_reads_the_shipped_listing_and_not_the_vendored_snapshot():
    """Only four config files are vendored here; 327 ship upstream.

    A census built on the vendored copies would report almost every shape as
    uncovered, which is the same answer the real census gives for a completely
    different reason, and would therefore look right.
    """
    rows = TVF.coverage_census(["deepseek-v2-lite"], ["NVIDIA_B200"], "bf16")
    (model, gpu, has_tuned, name) = rows[0]
    assert has_tuned is True
    assert name == "E=64,N=1408,device_name=NVIDIA_B200.json"
    assert not (ROOT / "moe/bench/hardware/vllm_configs" / name).exists()


def test_the_census_says_most_shapes_take_the_ladder():
    rows = TVF.coverage_census([m for m in TVF.MODEL_CONFIGS if m != "toy"],
                               list(TVF.CENSUS_GPUS), "bf16")
    covered = sum(1 for _, _, has_tuned, _ in rows if has_tuned)
    assert covered < len(rows) / 2
    assert TVF.render_census(rows).splitlines()[-1].startswith(str(covered))


def test_the_census_counts_the_tuned_column_and_not_every_row():
    """The count and the table must agree, or the motivation reads as 100%."""
    rows = TVF.coverage_census(list(TVF.TUNED_H200_MODELS),
                               ["NVIDIA_H200", "NVIDIA_A10G"], "bf16")
    text = TVF.render_census(rows)
    assert text.count("| tuned ") == 2
    assert "2 of 4 pairs" in text


# --------------------------------------------------------------------------
# the report assembles at all, including on the sad paths
# --------------------------------------------------------------------------

def test_an_excluded_cell_is_named_in_the_table_rather_than_dropped_silently():
    """A cell that produced no number must still appear, saying why.

    A missing row and a row of dashes look the same in a summary and mean very
    different things: one is a cell nobody ran, the other is a cell that ran and
    crashed.
    """
    good = make_cell("mixtral-8x7b", 16)
    broken = make_cell("mixtral-8x7b", 512)
    arms = planted(broken, tuned=1.0, fallback=1.2)
    arms["fallback"].ms_median = None
    arms["fallback"].error = "OutOfResources: out of shared memory"
    got = TVF.analyse([good, broken],
                      {good.key: planted(good, tuned=1.0, fallback=1.2),
                       broken.key: arms})
    text = TVF.render_results(got)
    assert "EXCLUDED" in text
    assert "out of shared memory" in text
    assert len(got.measured) == 1
    assert any("out of shared memory" in line
               for line in got.compile_failures)


def test_the_report_written_to_the_volume_carries_the_sign_the_gates_and_the_gaps():
    """What a reader finds after the pod is gone must stand on its own.

    The report is the artefact; the CSV is the evidence and the terminal is
    gone. So it has to carry the sign convention, the headline in words, every
    gate with its verdict, the partial-run note, and the arms that never
    produced a number.
    """
    good = make_cell("mixtral-8x7b", 16)
    broken = make_cell("mixtral-8x7b", 512)
    arms = planted(broken, tuned=1.0, fallback=1.2)
    arms["fallback"].ms_median = None
    arms["fallback"].error = "OutOfResources: out of shared memory"
    got = TVF.analyse([good, broken],
                      {good.key: planted(good, tuned=1.0, fallback=1.2,
                                         knobs={"group": 1.2}),
                       broken.key: arms})
    text = TVF.render_report(TVF.SIGN_BANNER, got, TVF.build_gates(got),
                             stopped="interrupted")
    for section in ("## Per cell", "## Which knob is the cost?", "## Headline",
                    "## Gates", "## Arms that produced no timing"):
        assert section in text, section
    assert "penalty = time(FALLBACK ladder config)" in text
    assert "SLOWER" in text
    assert "PARTIAL RUN: interrupted." in text
    assert "out of shared memory" in text
    assert "[PASS]" in text and "[FAIL]" not in text


def test_a_report_with_nothing_measured_refuses_to_invent_a_headline():
    cell = make_cell("mixtral-8x7b", 16)
    empty = {name: TVF.ArmResult(cell.model, cell.num_tokens, name,
                                 TVF.arm_config(cell, name),
                                 config_origin="forced", error="never ran")
             for name in TVF.ARM_ORDER}
    got = TVF.analyse([cell], {cell.key: empty})
    text = TVF.render_report("", got, TVF.build_gates(got))
    assert got.headline is None
    assert "That is not a null result." in text
