"""MOE_FORCE_TILE: it takes effect, it refuses bad input, it records a skip.

THE STATE THIS REPLACES. `scripts/pod_session.sh` set MOE_FORCE_TILE and gated
(S6a) on reading the OBSERVED `tile_block_m` back; `docs/POD_RUNBOOK.md`
documented it. Nothing in the sweep path read the variable, so a 54-minute,
3,696-row dense grid ran on vLLM's own ladder and its crossing could not be
quoted as tile-pinned. The ledger entry is in `moe/bench/force_tile.py`.

Three things therefore have to be true, and each has tests here rather than a
comment claiming it:

1. IT TAKES EFFECT. The pin is open across the correctness check, the tile
   observation and every timed trial, and the row that comes out shows the tile
   that was forced -- verified end to end against a stand-in for vLLM's own
   `override_config` / `try_get_optimal_moe_config` pair, so the chain under
   test is the chain the pod runs.
2. BAD INPUT REFUSES, AT STARTUP. Not JSON, not an object, missing a key,
   carrying an unknown one, a float, a bool, a zero, a block size that is not a
   power of two: every one of them raises before a GPU is touched.
3. AN UNHONOURABLE PATH IS RECORDED, NOT SILENTLY RUN. A span with no pinning
   hook writes no row at all under an active pin, and the manifest says why in
   a status that is deliberately not terminal.
"""
from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager

import pytest

from moe.baselines import _framework_config as FC
from moe.bench import cli
from moe.bench import driver as D
from moe.bench import force_tile as FT
from moe.bench import schema as SC
from moe.bench import timing as T
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

TILE = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4}
#: Exactly what scripts/pod_session.sh line 1702 exports.
TILE_JSON = json.dumps(TILE)


def forced_tile(**overrides) -> FT.ForcedTile:
    return FT.parse(json.dumps({**TILE, **overrides}))


# --------------------------------------------------------------------------
# 2. what the variable is allowed to say
# --------------------------------------------------------------------------

def test_the_pod_sessions_own_value_parses_into_vllms_six_keys():
    """The literal string in scripts/pod_session.sh, so a change to either side
    of that contract fails here rather than on a rented pod."""
    tile = FT.parse(TILE_JSON)
    assert tile.config == TILE
    assert tile.block_m == 128
    assert [k for k, _ in tile.items] == list(FT.REQUIRED_KEYS)
    assert "BLOCK_SIZE_M=128" in tile.describe()


def test_an_absent_variable_is_the_ordinary_unpinned_sweep():
    """None, not an empty config: an unpinned sweep must behave exactly as it
    did before this module existed, and every branch downstream keys on None."""
    assert FT.from_env({}) is None


def test_a_variable_that_is_set_but_empty_refuses_rather_than_unpinning():
    """`MOE_FORCE_TILE="${PIN:-}"` with PIN unset is the common way to get here,
    and taking it as "no pin" is the exact silence this module exists to end."""
    with pytest.raises(FT.ForceTileMalformed, match="set but empty"):
        FT.from_env({FT.ENV_VAR: "  "})


def test_text_that_is_not_json_refuses_and_shows_a_working_example():
    with pytest.raises(FT.ForceTileMalformed, match="not JSON") as e:
        FT.parse("BLOCK_SIZE_M=128")
    assert "BLOCK_SIZE_M" in str(e.value) and "MOE_FORCE_TILE=" in str(e.value)


def test_json_that_is_not_an_object_refuses():
    with pytest.raises(FT.ForceTileMalformed, match="not an object"):
        FT.parse("[128, 128]")


def test_a_missing_key_refuses_and_names_it():
    """vLLM's override REPLACES the config rather than merging into it, and the
    result is splatted into the kernel launch, so a missing key is a TypeError
    several minutes into a metered sweep."""
    partial = {k: v for k, v in TILE.items() if k != "num_stages"}
    with pytest.raises(FT.ForceTileMalformed, match=r"missing \['num_stages'\]"):
        FT.parse(json.dumps(partial))


def test_an_unknown_key_refuses_and_names_it():
    """BLOCK_M is the kernel parameter's name, not the config file's. Dropping
    it silently would launch a tile nobody typed."""
    with pytest.raises(FT.ForceTileMalformed, match=r"unknown \['BLOCK_M'\]"):
        FT.parse(json.dumps({**TILE, "BLOCK_M": 64}))


def test_a_float_refuses_because_triton_specialises_on_the_type():
    with pytest.raises(FT.ForceTileMalformed, match="not an int"):
        FT.parse(json.dumps({**TILE, "BLOCK_SIZE_M": 128.0}))


def test_a_bool_refuses_even_though_python_calls_it_an_int():
    """isinstance(True, int) is True, so BLOCK_SIZE_M=true would otherwise sail
    through every check below it as a 1."""
    with pytest.raises(FT.ForceTileMalformed, match="not an int"):
        FT.parse(json.dumps({**TILE, "BLOCK_SIZE_M": True}))


@pytest.mark.parametrize("value", [0, -64])
def test_a_non_positive_value_refuses(value):
    """There is no tile of height zero, and schema.tile_field refuses to read a
    0 back as a measurement for the same reason."""
    with pytest.raises(FT.ForceTileMalformed, match="not positive"):
        FT.parse(json.dumps({**TILE, "BLOCK_SIZE_M": value}))


def test_a_block_size_that_is_not_a_power_of_two_refuses_here_not_mid_sweep():
    with pytest.raises(FT.ForceTileMalformed, match="not a power of two"):
        FT.parse(json.dumps({**TILE, "BLOCK_SIZE_K": 96}))


def test_a_swizzle_width_that_is_not_a_power_of_two_is_allowed():
    """Deliberate asymmetry, not an oversight. The block sizes index tl.arange
    and Triton refuses anything else; GROUP_SIZE_M only divides the program id,
    so refusing 3 would refuse a setting the kernel runs perfectly well."""
    assert FT.parse(json.dumps({**TILE, "GROUP_SIZE_M": 3})).config["GROUP_SIZE_M"] == 3


# --------------------------------------------------------------------------
# the resume key: the project's failure mode 2
# --------------------------------------------------------------------------

def test_two_tiles_that_differ_only_in_the_swizzle_get_different_keys():
    """A run id that omits a swept parameter lets the second setting resume the
    first, skip every completed cell, and print the first's numbers under the
    second's label. GROUP_SIZE_M is exactly such a parameter: alpha measured
    0.84 at G=1 against 0.67 at G=64 on both cards."""
    assert forced_tile().fingerprint() != forced_tile(GROUP_SIZE_M=64).fingerprint()
    assert forced_tile().fingerprint() == forced_tile().fingerprint()


def test_an_unpinned_run_keeps_the_manifest_keys_it_has_always_had():
    """Empty suffix, so every existing manifest still resumes byte for byte."""
    assert FT.CellPin().key_suffix == ""
    row = SC.Row(model="toy", num_tokens=32, impl="x")
    assert D._cell_key(row, FT.CellPin()) == SC.cell_key(row)
    pinned = FT.pin_for(forced_tile(), [_PINNABLE], _PINNABLE)
    assert D._cell_key(row, pinned) != SC.cell_key(row)
    assert pinned.forced is not None
    assert pinned.forced.fingerprint() in D._cell_key(row, pinned)


# --------------------------------------------------------------------------
# 3. which spans can honour it
# --------------------------------------------------------------------------

#: Every config a pinnable span was run under, newest last, plus the calls made
#: while one was open. Module level because the driver builds its own spans out
#: of the registry and a fixture cannot reach into them.
EVENTS: list = []
IN_FORCE: list = []


class _UpGemm(StageSpan):
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        EVENTS.append(("call", dict(IN_FORCE[-1]) if IN_FORCE else None))
        st.h_up = R.grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets,
            2 * st.spec.model.intermediate_size)


@register
class ForcePinnableUpGemm(_UpGemm):
    """Stands in for vLLM's span: it can be pinned, and it observes what ran."""

    name = "t_force_pinnable_up_gemm"

    def force_tile_config(self, config: dict):
        @contextmanager
        def ctx():
            IN_FORCE.append(dict(config))
            EVENTS.append(("enter", dict(config)))
            try:
                yield "fake.override_config"
            finally:
                EVENTS.append(("exit", dict(config)))
                IN_FORCE.pop()
        return ctx()

    def observe_tile_config(self, st: MoEState) -> dict:
        """Reads what is in force, exactly as vLLM's observer reads the config
        back out of try_get_optimal_moe_config during a real call."""
        EVENTS.append(("observe", dict(IN_FORCE[-1]) if IN_FORCE else None))
        if not IN_FORCE:
            return {"tile_config_source": "unrecorded"}
        meta = {col: IN_FORCE[-1][key]
                for key, col in FC.CONFIG_KEY_TO_COLUMN.items()}
        meta["tile_config_source"] = "vllm_override"
        return meta


@register
class ForceUnpinnableUpGemm(_UpGemm):
    """No hook at all: the shape of torch's CUTLASS grouped GEMM, SGLang and
    every reference span."""

    name = "t_force_unpinnable_up_gemm"


@register
class ForceLyingUpGemm(ForcePinnableUpGemm):
    """Pinnable, but what it observes is not what was forced.

    The failure this exists for is an override that is entered and does not
    take. Measured anyway, its rows say `vllm_override` and carry a tile the
    kernel never ran, which is worse than the unpinned sweep they replaced.
    """

    name = "t_force_lying_up_gemm"

    def observe_tile_config(self, st: MoEState) -> dict:
        meta = super().observe_tile_config(st)
        meta["tile_block_m"] = 64
        return meta


@register
class ForceBadHookUpGemm(_UpGemm):
    name = "t_force_bad_hook_up_gemm"

    def force_tile_config(self, config: dict):
        return None


_PINNABLE = ForcePinnableUpGemm()
_UNPINNABLE = ForceUnpinnableUpGemm()


def test_a_span_with_the_hook_is_pinned_and_one_without_is_refused():
    pin = FT.pin_for(forced_tile(), [_PINNABLE], _PINNABLE)
    assert pin.status == FT.PINNED and pin.target == _PINNABLE.name

    refused = FT.pin_for(forced_tile(), [_UNPINNABLE], _UNPINNABLE)
    assert refused.status == FT.UNHONOURABLE
    assert _UNPINNABLE.name in refused.reason
    assert "CUTLASS" in refused.reason and "SGLang" in refused.reason


def test_a_whole_layer_row_finds_the_hook_on_any_span_of_the_tiling():
    """`span is None` means the timer wraps the layer, and the target rule is
    the same one driver.observe_tile_config uses: ask every span, first answer
    wins."""
    pin = FT.pin_for(forced_tile(), [_UNPINNABLE, _PINNABLE], None)
    assert pin.status == FT.PINNED and pin.target == _PINNABLE.name


def test_nothing_is_pinned_when_nothing_was_asked_for():
    pin = FT.pin_for(None, [_PINNABLE], _PINNABLE)
    assert pin.status == FT.OFF
    with pin.applied():
        pass


def test_applying_an_unhonourable_pin_raises_rather_than_doing_nothing():
    """The driver never reaches this, because it records the skip first. It
    raises anyway: a silent no-op here would be a cell measured unpinned inside
    a pinned sweep, which is the whole failure."""
    pin = FT.pin_for(forced_tile(), [_UNPINNABLE], _UNPINNABLE)
    with pytest.raises(FC.ForceTileNotHonoured, match="recorded rather than run"):
        pin.applied()


def test_a_hook_that_returns_a_non_context_manager_is_named_in_the_error():
    """`with None:` fails with an AttributeError naming neither the span nor the
    tile, several minutes into a metered sweep."""
    span = ForceBadHookUpGemm()
    pin = FT.pin_for(forced_tile(), [span], span)
    with pytest.raises(FC.ForceTileNotHonoured, match="not a context manager"):
        pin.applied()


def test_split_by_pinnability_is_what_the_dry_run_prints():
    can, cannot = FT.split_by_pinnability([_UNPINNABLE, _PINNABLE])
    assert can == [_PINNABLE.name] and cannot == [_UNPINNABLE.name]


# --------------------------------------------------------------------------
# 1. it takes effect, through the driver
# --------------------------------------------------------------------------

def fake_timer(fn, warmup=1, iters=2, trials=1, l2_flush=True, flush_mb=8,
               flush_mode="read", target_ms=200.0, on_captured=None):
    EVENTS.append(("timed", dict(IN_FORCE[-1]) if IN_FORCE else None))
    fn()
    return T.TimingResult(ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05,
                          jitter_p90_over_p50=1.2, warmup=warmup,
                          iters=iters or 2, trials=trials, l2_flush=l2_flush,
                          cuda_graph=False, samples=3, flush_mb=flush_mb,
                          flush_mode=flush_mode)


def sweep(tmp_path, impl, forced=None, run_id="fixed", l2_modes=(True,)):
    """One cell through the real driver, with the timer injected."""
    cfg = D.RunConfig(out_dir=tmp_path, run_id=run_id, device="cpu", warmup=1,
                      trials=1, iters=2, l2_modes=l2_modes, graph_modes=(False,),
                      timer_eager=fake_timer, timer_graph=fake_timer,
                      clock_sampler=lambda: T.ClockState(1980, 45),
                      force_tile=forced)
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"))
    names = ["ref_router", "ref_permute", impl, "ref_act", "ref_down_gemm",
             "ref_unpermute"]
    D.run_sweep([(spec, names, impl)], cfg, routing=lambda s: None,
                info={"gpu_name": "FakeH200", "sm_capability": "9.0"})
    rows = SC.read_csv(cfg.csv_path) if cfg.csv_path.exists() else []
    return cfg, rows


def manifest_records(cfg) -> list[dict]:
    return [json.loads(line)
            for line in cfg.manifest_path.read_text().splitlines() if line.strip()]


def test_the_forced_tile_reaches_the_row_the_S6a_gate_reads(tmp_path):
    """The gate is `observed tile_block_m == 128`, read out of the CSV. This is
    that read, against the driver that produces it."""
    EVENTS.clear()
    _, rows = sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    assert len(rows) == 1
    row = rows[0]
    assert SC.tile_field(row, "tile_block_m") == 128
    assert SC.tile_field(row, "tile_block_n") == 128
    assert SC.tile_field(row, "tile_group_m") == 1
    assert SC.tile_field(row, "tile_config_source") == "vllm_override"
    assert SC.has_tile_config(row)


def test_the_pin_is_open_across_correctness_observation_and_timing(tmp_path):
    """Not only the timed region. The fp32 oracle has to validate the SAME tile
    that is timed, or the correctness gate is checking a kernel the row does not
    describe."""
    EVENTS.clear()
    sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile(),
          l2_modes=(True, False))
    kinds = [k for k, _ in EVENTS]
    assert kinds[0] == "enter" and kinds[-1] == "exit"
    assert "observe" in kinds and kinds.count("timed") == 2
    # Nothing ran outside the pin: every call, observation and timing saw it.
    assert all(config == TILE for kind, config in EVENTS
               if kind in ("call", "observe", "timed"))


def test_an_unpinned_sweep_still_runs_and_records_no_tile(tmp_path):
    """The default path, unchanged: no context is entered and the row says the
    observer saw nothing forced."""
    EVENTS.clear()
    _, rows = sweep(tmp_path, "t_force_pinnable_up_gemm")
    assert len(rows) == 1
    assert rows[0]["tile_config_source"] == "unrecorded"
    assert "enter" not in [k for k, _ in EVENTS]


# --------------------------------------------------------------------------
# 3. the recorded skip
# --------------------------------------------------------------------------

def test_a_span_that_cannot_be_pinned_writes_no_row_at_all(tmp_path):
    """A row with tile_block_m = 0 written during a pinned sweep cannot be told
    from the unpinned rows the pin exists to keep out of the file."""
    cfg, rows = sweep(tmp_path, "t_force_unpinnable_up_gemm", forced_tile())
    assert rows == []
    statuses = {r["status"] for r in manifest_records(cfg)}
    assert statuses == {SC.STATUS_FORCE_TILE_UNHONOURABLE}
    detail = manifest_records(cfg)[0]["detail"]
    assert "t_force_unpinnable_up_gemm" in detail


def test_the_skip_status_is_not_terminal_so_an_unpinned_run_measures_the_cell(tmp_path):
    """It describes the cell UNDER A PIN, not the cell. A terminal status would
    blank it from every future unpinned resume of the same run id."""
    assert SC.STATUS_FORCE_TILE_UNHONOURABLE not in SC.TERMINAL_STATUSES
    assert SC.STATUS_FORCE_TILE_NOT_OBSERVED not in SC.TERMINAL_STATUSES
    cfg, rows = sweep(tmp_path, "t_force_unpinnable_up_gemm", forced_tile())
    assert rows == []
    _, rows = sweep(tmp_path, "t_force_unpinnable_up_gemm")
    assert len(rows) == 1, "the unpinned re-run must measure what the pin skipped"


def test_a_pinned_run_cannot_resume_an_unpinned_ones_manifest(tmp_path):
    """FAILURE MODE 2, end to end: same run id, same cell, same directory. The
    unpinned rows are already there; without the fingerprint in the key the
    pinned run would skip the cell, write nothing, and the session would report
    the unpinned numbers under a pinned heading."""
    EVENTS.clear()
    cfg, rows = sweep(tmp_path, "t_force_pinnable_up_gemm")
    assert len(rows) == 1 and rows[0]["tile_config_source"] == "unrecorded"

    cfg, rows = sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    assert len(rows) == 2, "the pinned cell was skipped as already done"
    assert SC.tile_field(rows[1], "tile_block_m") == 128
    assert cfg.force_tile_ledger.pinned_cells == 1


def test_resuming_a_pinned_run_counts_the_cell_without_remeasuring_it(tmp_path):
    """Not the same state as measuring nothing, and the vacuity refusal has to
    tell them apart or a finished sweep re-run would exit non-zero."""
    sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    cfg, rows = sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    assert len(rows) == 1, "the completed cell was measured twice"
    assert cfg.force_tile_ledger.pinned_cells == 0
    assert cfg.force_tile_ledger.resumed_cells == 1
    assert not cfg.force_tile_ledger.vacuous()


def test_a_row_that_does_not_show_the_pin_is_refused_rather_than_written(tmp_path):
    """The override was entered and did not take. The timing may be perfectly
    good; what cannot be produced is a row that says which tile it describes."""
    cfg, rows = sweep(tmp_path, "t_force_lying_up_gemm", forced_tile())
    assert rows == []
    records = manifest_records(cfg)
    assert {r["status"] for r in records} == {SC.STATUS_FORCE_TILE_NOT_OBSERVED}
    assert "forced 128, observed 64" in records[0]["detail"]
    assert cfg.force_tile_ledger.unobserved


def test_an_unobservable_pin_is_a_disagreement_too():
    """A pinned row whose observer saw nothing is indistinguishable from an
    unpinned row in every group-by, which is instrument defect 3 exactly."""
    pin = FT.pin_for(forced_tile(), [_PINNABLE], _PINNABLE)
    assert pin.disagrees_with({"tile_config_source": "unrecorded"})
    assert not pin.disagrees_with(
        {**{col: TILE[key] for key, col in FC.CONFIG_KEY_TO_COLUMN.items()},
         "tile_config_source": "vllm_override"})


def test_a_row_from_vllms_own_ladder_is_a_disagreement_under_a_pin():
    """Even when the tile happens to match: the source column is what tells a
    reader the row was pinned rather than chosen."""
    pin = FT.pin_for(forced_tile(), [_PINNABLE], _PINNABLE)
    meta = {col: TILE[key] for key, col in FC.CONFIG_KEY_TO_COLUMN.items()}
    assert pin.disagrees_with({**meta, "tile_config_source": "vllm_default"})


# --------------------------------------------------------------------------
# non-vacuity: a pin that applied to nothing must not exit 0
# --------------------------------------------------------------------------

def test_a_sweep_that_pinned_nothing_exits_non_zero(tmp_path):
    cfg, _ = sweep(tmp_path, "t_force_unpinnable_up_gemm", forced_tile())
    ledger = cfg.force_tile_ledger
    assert ledger.vacuous() and ledger.skipped_cells == 1
    assert cli.force_tile_verdict(forced_tile(), ledger) == 3


def test_a_sweep_whose_rows_did_not_show_the_pin_exits_non_zero(tmp_path):
    cfg, _ = sweep(tmp_path, "t_force_lying_up_gemm", forced_tile())
    assert cli.force_tile_verdict(forced_tile(), cfg.force_tile_ledger) == 4


def test_a_sweep_that_pinned_something_exits_zero(tmp_path):
    cfg, rows = sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    assert len(rows) == 1
    assert cli.force_tile_verdict(forced_tile(), cfg.force_tile_ledger) == 0


def test_an_unpinned_sweep_is_never_judged_on_the_pin(tmp_path):
    cfg, _ = sweep(tmp_path, "t_force_unpinnable_up_gemm")
    assert cli.force_tile_verdict(None, cfg.force_tile_ledger) == 0


def test_the_gates_are_numbers_against_thresholds(tmp_path, capsys):
    """Both are VALIDITY gates: a FAIL means no number from the run may be
    quoted as tile-pinned, not that the tile behaved unexpectedly."""
    cfg, rows = sweep(tmp_path, "t_force_pinnable_up_gemm", forced_tile())
    assert [(g[0], g[4]) for g in cfg.force_tile_ledger.gates()] == [("F1", True),
                                                                    ("F2", True)]
    cli.force_tile_verdict(forced_tile(), cfg.force_tile_ledger)
    out = capsys.readouterr().out
    assert "GATE F1" in out and "GATE F2" in out and "FAIL" not in out

    lying, _ = sweep(tmp_path / "b", "t_force_lying_up_gemm", forced_tile())
    assert [g[4] for g in lying.force_tile_ledger.gates()] == [False, False]
    cli.force_tile_verdict(forced_tile(), lying.force_tile_ledger)
    assert "FAIL" in capsys.readouterr().out


def test_the_cli_refuses_an_unpinnable_plan_before_it_spends_anything(tmp_path,
                                                                     monkeypatch,
                                                                     capsys):
    """Not after the sweep. The alternative is discovering, past 22 GB of
    weights and an fp32 oracle, that the plan held no span able to pin."""
    monkeypatch.setenv(FT.ENV_VAR, TILE_JSON)
    code = cli.main(["--profile", "smoke", "--out-dir", str(tmp_path),
                     "--groups", "reference,baselines",
                     "--impl", "torch_grouped_mm_up"])
    assert code == 3
    assert "REFUSED" in capsys.readouterr().out
    assert list(tmp_path.glob("*.csv")) == [], "a refused run must write nothing"


def test_the_summary_says_what_the_pin_did(tmp_path):
    cfg, _ = sweep(tmp_path, "t_force_unpinnable_up_gemm", forced_tile())
    text = "\n".join(cfg.force_tile_ledger.summary(forced_tile()))
    assert "BLOCK_SIZE_M=128" in text and "1 skipped as unpinnable" in text
    assert cfg.force_tile_ledger.summary(None) == []


# --------------------------------------------------------------------------
# the CLI: refusal before anything is spent
# --------------------------------------------------------------------------

def test_the_cli_refuses_a_malformed_variable_before_it_imports_anything(monkeypatch):
    monkeypatch.setenv(FT.ENV_VAR, "{oops")
    with pytest.raises(SystemExit) as e:
        cli.main(["--profile", "smoke", "--dry-run"])
    assert "not JSON" in str(e.value)


def test_the_dry_run_refuses_a_plan_where_nothing_can_be_pinned(monkeypatch, capsys):
    """The review step. On a laptop no framework span registers, which is also
    what a pod whose vLLM import failed would look like."""
    monkeypatch.setenv(FT.ENV_VAR, TILE_JSON)
    # torch's grouped GEMM alone: on the pod that is what `--groups baselines`
    # plans when --env is left at its default, because the vLLM span declares
    # env="vllm" and profiles.candidate_impls filters on it.
    assert cli.main(["--profile", "smoke", "--dry-run",
                     "--groups", "reference,baselines",
                     "--impl", "torch_grouped_mm_up"]) == 1
    out = capsys.readouterr().out
    assert "forced tile" in out and "REFUSED" in out
    assert "--env vllm" in out


def test_the_dry_run_says_nothing_about_pinning_when_nothing_is_forced(monkeypatch,
                                                                      capsys):
    monkeypatch.delenv(FT.ENV_VAR, raising=False)
    cli.main(["--profile", "smoke", "--dry-run"])
    assert "forced tile" not in capsys.readouterr().out


def test_the_dry_run_lists_the_pinnable_and_unpinnable_halves():
    lines, ok = cli.force_tile_plan(forced_tile(), [_PINNABLE, _UNPINNABLE])
    text = "\n".join(lines)
    assert ok
    assert "CAN PIN       t_force_pinnable_up_gemm" in text
    assert "CANNOT PIN    t_force_unpinnable_up_gemm" in text
    assert "RECORDED AND SKIPPED" in text


# --------------------------------------------------------------------------
# the vLLM half, against a stand-in for vLLM's own module namespaces
# --------------------------------------------------------------------------

def fake_vllm(monkeypatch, *, working: bool = True, hook: bool = True):
    """vLLM's fused_moe namespace, in the two ways that matter here.

    `try_get_optimal_moe_config` consults `get_config()` FIRST and a truthy
    value bypasses the tuned file and the ladder both -- that is the mechanism
    a pin rides on, so the fake reproduces it rather than asserting it.

    `working=False` is an override_config that is entered and does not take: the
    silent failure the verification inside `forcing_tile_config` exists for.
    """
    module = types.ModuleType("fake_vllm_force")
    state: dict = {}

    def get_config():
        return state.get("config")

    @contextmanager
    def override_config(config):
        previous = state.get("config")
        if working:
            state["config"] = config
        try:
            yield
        finally:
            state["config"] = previous

    def get_moe_configs(num_experts, n, dtype, block_shape=None):
        return None

    def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M,
                                   block_shape=None):
        forced = module.get_config()
        if forced:
            return dict(forced)
        lookup = module.get_moe_configs
        lookup(len(w1_shape), w1_shape[1], dtype)
        return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}

    module.get_config = get_config
    module.get_moe_configs = get_moe_configs
    module.try_get_optimal_moe_config = try_get_optimal_moe_config
    if hook:
        module.override_config = override_config
    monkeypatch.setitem(sys.modules, "fake_vllm_force", module)
    return module, ("fake_vllm_force",)


def test_forcing_and_observing_compose_into_the_row_the_gate_reads(monkeypatch):
    """The whole chain minus vLLM itself: force, run, read the config back out
    of the lookup, and label the source. This is what the pod does per cell."""
    module, names = fake_vllm(monkeypatch)
    capture = FC.TileCapture()
    with FC.forcing_tile_config(TILE, names) as where:
        assert where.endswith("override_config")
        with FC.recording_tile_config(capture, names):
            module.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                              2, "bf16", 787)
        meta = FC.tile_meta_from_capture(capture, FC.vllm_override_active(names))
    assert meta["tile_block_m"] == 128
    assert meta["tile_config_source"] == "vllm_override"
    # And the pin agrees the row shows what it asked for.
    assert not FT.pin_for(forced_tile(), [_PINNABLE], _PINNABLE).disagrees_with(meta)


def test_the_override_is_restored_when_the_context_closes(monkeypatch):
    module, names = fake_vllm(monkeypatch)
    with FC.forcing_tile_config(TILE, names):
        assert FC.vllm_forced_config(names) == TILE
    assert FC.vllm_forced_config(names) is None
    assert not FC.vllm_override_active(names)


def test_an_override_that_does_not_take_is_refused_rather_than_measured(monkeypatch):
    """Entering a context and assuming it took is how MOE_FORCE_TILE came to be
    set for a whole session with nothing reading it."""
    _, names = fake_vllm(monkeypatch, working=False)
    with pytest.raises(FC.ForceTileNotHonoured, match="get_config"):
        with FC.forcing_tile_config(TILE, names):
            pass


def test_a_vllm_with_no_override_hook_refuses(monkeypatch):
    _, names = fake_vllm(monkeypatch, hook=False)
    with pytest.raises(FC.ForceTileNotHonoured, match="no override_config"):
        with FC.forcing_tile_config(TILE, names):
            pass


def test_an_extra_key_vllm_adds_of_its_own_is_not_a_refusal(monkeypatch):
    """Compared key by key: a future vLLM that copies the override and adds a
    key would still be honouring every value that was forced, and refusing it
    would be a false alarm on a rented pod."""
    module, names = fake_vllm(monkeypatch)
    original = module.override_config

    @contextmanager
    def override_config(config):
        with original({**config, "extra": 1}):
            yield

    module.override_config = override_config
    with FC.forcing_tile_config(TILE, names):
        assert FC.vllm_forced_config(names)["BLOCK_SIZE_M"] == 128


def test_the_hook_name_the_driver_looks_up_is_the_one_the_span_defines():
    """One spelling. A rename on either side turns every vLLM cell unpinnable,
    which is loud rather than silent -- but it is still a bug, and this is where
    it fails."""
    assert FT.FORCE_TILE_HOOK == "force_tile_config"
    assert FT.can_pin(_PINNABLE) and not FT.can_pin(_UNPINNABLE)
