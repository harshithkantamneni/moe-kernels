"""Schema v4: the tile configuration that ACTUALLY ran.

Three facts motivated the column set, and each one is defended by tests here.

1. The published CSVs record NO information about which Triton tile the kernel
   ran. `load_tile_eff_bm64` and `load_tile_eff_bm128` are HYPOTHETICAL
   efficiencies computed from the routing histogram at ASSUMED block sizes, so
   nothing in a row could contradict a wrong BLOCK_SIZE_M, and one sat
   unchallenged in an analysis for days.
2. Verified against vLLM 0.27.1: only 2 of 8 (model x card) cells ran a TUNED
   config. Nothing ships for NVIDIA_A100-SXM4-80GB at any of the four shapes, so
   those took the hardcoded fallback ladder, and the tuned lookup picks the
   NEAREST key rather than the floor. The A100 and the H200 therefore ran
   different tiles at the measured crossings.
3. Nothing about a v3 row may be inferred to fill the gap. A 0 that reads as
   "BLOCK_M was zero" is the exact class of silent default this column exists to
   prevent, so a v3 row raises when read rather than answering.
"""
from __future__ import annotations

import sys
import types

import pytest

from moe.baselines import _framework_config as FC
from moe.bench import driver as D
from moe.bench import schema as SC
from moe.bench import timing as T
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register, registry
from moe.state import MoEState

# --------------------------------------------------------------------------
# a v3 CSV, written the way the ten published arms were
# --------------------------------------------------------------------------

#: Every column v3 had: the current set minus everything v4 added.
V3_COLUMNS = [c for c in SC.COLUMNS if c not in SC.COLUMNS_ADDED_IN[4]]


def write_v3_csv(path, **overrides) -> None:
    """A file with a v3 header, i.e. no tile_* columns at all.

    Written by hand rather than by CsvWriter on purpose: CsvWriter emits the
    CURRENT column set, so it can no longer produce the shape the published arms
    are actually in, and a test that used it would be testing nothing.
    """
    values = dict(schema_version=3, model="mixtral-8x7b", num_tokens=787,
                  dtype="bf16", impl="vllm_fused_experts", ms_p50=1.25,
                  correctness_passed=True, gpu_name="NVIDIA H200")
    values.update(overrides)
    header = ",".join(V3_COLUMNS)
    row = ",".join(str(values.get(c, "")) for c in V3_COLUMNS)
    path.write_text(f"{header}\n{row}\n")


def v3_row(tmp_path):
    write_v3_csv(tmp_path / "v3.csv")
    return SC.read_csv(tmp_path / "v3.csv")[0]


def test_a_v3_csv_still_loads_because_the_published_arms_cannot_be_remeasured(tmp_path):
    """The version gate widened rather than the data being retired.

    This is why `tile_efficiency_for_row` had to reconstruct a missing tile
    efficiency out of stored columns instead of a column being added: under a
    strict `!= SCHEMA_VERSION` gate, every new column made all ten published
    arms unreadable by the code meant to analyse them.
    """
    row = v3_row(tmp_path)
    assert row["model"] == "mixtral-8x7b"
    assert row["schema_version"] == "3"


def test_a_v3_row_materialises_the_new_columns_as_a_sentinel_not_as_zero(tmp_path):
    """0 is a number: it survives float(), it plots, it averages, and it reads
    as "BLOCK_M was zero". The sentinel is a string precisely so that no numeric
    path can accept it."""
    row = v3_row(tmp_path)
    for name in SC.COLUMNS_ADDED_IN[4]:
        assert row[name] == SC.UNRECORDED, name
    assert row["tile_block_m"] != 0
    assert row["tile_block_m"] != ""


def test_reading_a_tile_field_off_a_v3_row_raises_instead_of_answering(tmp_path):
    row = v3_row(tmp_path)
    with pytest.raises(SC.TileConfigUnrecorded, match="older schema"):
        SC.tile_field(row, "tile_block_m")
    with pytest.raises(SC.TileConfigUnrecorded):
        SC.tile_field(row, "tile_config_source")
    with pytest.raises(SC.TileConfigUnrecorded):
        SC.tile_field(row, "sm_capability")


def test_row_float_refuses_the_sentinel_rather_than_defaulting_to_zero(tmp_path):
    """The sentinel has to be rejected in EVERY reader, not only in tile_field.

    float("<unrecorded>") raises ValueError, and row_float's except clause turns
    a ValueError into the default -- so without an explicit check the guarded
    column would come back as 0.0 through the one function every analysis
    already uses.
    """
    row = v3_row(tmp_path)
    with pytest.raises(SC.TileConfigUnrecorded):
        SC.row_float(row, "tile_block_m")
    assert SC.row_float(row, "ms_p50") == 1.25


def test_a_v4_row_that_observed_nothing_raises_rather_than_reporting_block_m_zero():
    """The second way the gap appears: a v4 row whose span had no observer.

    Same answer as a v3 row, for the same reason. The source column says which
    of the two happened; neither hands back a usable number.
    """
    row = {"schema_version": "4", "tile_block_m": "0",
           "tile_config_source": "unrecorded"}
    with pytest.raises(SC.TileConfigUnrecorded, match="not a measured zero"):
        SC.tile_field(row, "tile_block_m")


def test_a_recorded_tile_reads_back_as_the_int_it_was():
    row = {"schema_version": "4", "tile_block_m": "64", "tile_num_warps": "4",
           "tile_config_source": "vllm_tuned", "tile_config_key": "1024",
           "sm_capability": "9.0"}
    assert SC.tile_field(row, "tile_block_m") == 64
    assert SC.tile_field(row, "tile_num_warps") == 4
    assert SC.tile_field(row, "tile_config_key") == 1024
    assert SC.tile_field(row, "tile_config_source") == "vllm_tuned"
    assert SC.tile_field(row, "sm_capability") == "9.0"


def test_the_source_reads_back_even_when_it_says_the_tile_is_unknown():
    """"unrecorded" and "n/a" are honest answers to "where did this config come
    from", so reading the source must not itself be an exception -- that is how
    a caller decides whether to ask anything else."""
    assert SC.tile_field({"tile_config_source": "unrecorded"},
                         "tile_config_source") == "unrecorded"
    assert SC.tile_field({"tile_config_source": "cutlass_static"},
                         "tile_config_source") == "cutlass_static"


def test_tile_field_refuses_a_column_that_is_not_tile_provenance():
    with pytest.raises(ValueError, match="not a v4 provenance column"):
        SC.tile_field({"ms_p50": "1.0"}, "ms_p50")
    with pytest.raises(KeyError, match="not a column"):
        SC.tile_field({}, "tile_block_mm")


def test_has_tile_config_is_the_filter_an_analysis_uses(tmp_path):
    """Keyed on tile_block_m, not on the source: a row can know the tile it ran
    and not know where the tile came from."""
    assert not SC.has_tile_config(v3_row(tmp_path))
    assert not SC.has_tile_config({"tile_block_m": "0",
                                   "tile_config_source": "cutlass_static"})
    assert not SC.has_tile_config({"tile_config_source": "sglang"})
    assert SC.has_tile_config({"tile_block_m": "16",
                               "tile_config_source": "vllm_tuned"})
    assert SC.has_tile_config({"tile_block_m": "64",
                               "tile_config_source": "unrecorded"})


def test_merge_refuses_to_put_two_schema_versions_under_one_header(tmp_path):
    """read_csv accepts v3 and v4 so the published arms stay loadable; a merge
    writes one header, and a v3 row under v4 column names would claim tile
    columns it never had."""
    write_v3_csv(tmp_path / "old.csv")
    new = tmp_path / "new.csv"
    with SC.CsvWriter(new) as w:
        w.write(SC.Row(model="toy", tile_block_m=64,
                       tile_config_source="vllm_tuned"))
    with pytest.raises(ValueError, match="mix schema versions"):
        SC.merge_csvs([tmp_path / "old.csv", new], tmp_path / "merged.csv")


def test_a_v3_row_stays_marked_unrecorded_across_a_merge(tmp_path):
    """The mark is derived from schema_version on every read, so it cannot drift
    out of sync with the row: a round trip through merge_csvs restores it rather
    than freezing a zero into the merged file."""
    write_v3_csv(tmp_path / "old.csv")
    out = tmp_path / "merged.csv"
    assert SC.merge_csvs([tmp_path / "old.csv"], out) == 1
    merged = SC.read_csv(out)[0]
    assert merged["schema_version"] == "3"
    assert merged["tile_block_m"] == SC.UNRECORDED
    with pytest.raises(SC.TileConfigUnrecorded):
        SC.tile_field(merged, "tile_block_m")


# --------------------------------------------------------------------------
# the nearest-key rule
# --------------------------------------------------------------------------

TUNED_KEYS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
              2048, 3072, 4096]


def test_the_tuned_lookup_takes_the_NEAREST_key_and_not_the_floor():
    """vLLM resolves the entry with min(configs.keys(), key=lambda x: abs(x-M)).

    M=787 selects 1024, NOT 512. Reading that lookup as a floor -- the natural
    assumption, and the one a bucketed lookup would justify -- attributes a tile
    tuned for a shape eight times larger to a row that never ran it.
    """
    assert FC.nearest_config_key(TUNED_KEYS, 787) == 1024
    assert FC.nearest_config_key(TUNED_KEYS, 700) == 512
    assert FC.nearest_config_key(TUNED_KEYS, 5000) == 4096
    assert FC.nearest_config_key(TUNED_KEYS, 1) == 1


def test_a_tie_breaks_toward_the_smaller_key():
    """M exactly between two entries. python's min keeps the first minimum and a
    tuned JSON loads in file order, which is ascending in every shipped file."""
    assert FC.nearest_config_key([512, 1024], 768) == 512


def test_no_tuned_keys_is_an_error_rather_than_a_key_of_zero():
    with pytest.raises(ValueError, match="no tuned config keys"):
        FC.nearest_config_key([], 128)


# --------------------------------------------------------------------------
# the vLLM recorder, exercised against fake modules
# --------------------------------------------------------------------------

DEFAULT_LADDER = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                  "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
TUNED_ENTRY = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
               "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4}


def fake_vllm(monkeypatch, tuned: dict | None, *, second_binding: bool = True,
              with_lookup: bool = True, override: bool = False):
    """A stand-in for vLLM's fused_moe module namespaces.

    Shaped like the real thing in the two ways that matter: the config function
    looks up `get_moe_configs` through the module global (so a rebind is
    observable), and a second module holds the SAME function object, the way
    `from ... import try_get_optimal_moe_config` leaves it.
    """
    primary = types.ModuleType("fake_vllm_primary")
    calls: list = []

    def get_moe_configs(num_experts, n, dtype, block_shape=None):
        return tuned

    def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M,
                                   block_shape=None):
        calls.append(M)
        # Through the module global, the way vLLM's own does, so a rebind of
        # that global is observable from inside this call.
        lookup = getattr(primary, "get_moe_configs", primary._lookup)
        configs = lookup(len(w1_shape), w1_shape[1], dtype)
        if configs:
            key = FC.nearest_config_key(configs, M)
            return dict(configs[key])
        return dict(DEFAULT_LADDER)

    primary.try_get_optimal_moe_config = try_get_optimal_moe_config
    #: Left OFF the module in the with_lookup=False case, standing in for a vLLM
    #: whose lookup has moved or been renamed: the recorder then cannot tell a
    #: tuned file from the fallback ladder and must say so.
    primary._lookup = get_moe_configs
    if with_lookup:
        primary.get_moe_configs = get_moe_configs
    primary.get_config = (lambda: {"BLOCK_SIZE_M": 128}) if override else (lambda: None)

    names = ["fake_vllm_primary"]
    monkeypatch.setitem(sys.modules, "fake_vllm_primary", primary)
    if second_binding:
        secondary = types.ModuleType("fake_vllm_secondary")
        secondary.try_get_optimal_moe_config = try_get_optimal_moe_config
        monkeypatch.setitem(sys.modules, "fake_vllm_secondary", secondary)
        names.append("fake_vllm_secondary")
    return primary, tuple(names), calls


def test_the_recorder_reads_the_config_back_without_patching_vllm(monkeypatch):
    """Rebinding the module global is enough, because fused_experts_impl builds
    its functools.partial from that global on every call."""
    primary, names, _ = fake_vllm(monkeypatch, {1024: TUNED_ENTRY, 512: DEFAULT_LADDER})
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        primary.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                           2, "bf16", 787)
    meta = FC.tile_meta_from_capture(capture)
    assert meta["tile_block_m"] == 16
    assert meta["tile_num_warps"] == 8
    assert meta["tile_config_source"] == "vllm_tuned"
    assert meta["tile_config_key"] == 1024


def test_every_module_holding_the_name_is_rebound(monkeypatch):
    """`from ... import f` copies the function OBJECT into the importing
    module's globals, so rebinding only the defining module leaves the name the
    caller looks up pointing at the original: the recorder would observe nothing
    while appearing to be installed."""
    primary, names, _ = fake_vllm(monkeypatch, None)
    secondary = sys.modules["fake_vllm_secondary"]
    before = secondary.try_get_optimal_moe_config
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        assert secondary.try_get_optimal_moe_config is not before
        secondary.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                             2, "bf16", 32)
    assert capture.calls, "a call through the second binding was not recorded"
    assert secondary.try_get_optimal_moe_config is before


def test_the_originals_are_restored_even_when_the_call_raises(monkeypatch):
    """Left installed, the recorder would wrap every later cell of a metered
    sweep and append to a capture nobody reads."""
    primary, names, _ = fake_vllm(monkeypatch, None)
    before = primary.try_get_optimal_moe_config
    with pytest.raises(ZeroDivisionError):
        with FC.recording_tile_config(FC.TileCapture(), names):
            raise ZeroDivisionError("kernel launch went sideways")
    assert primary.try_get_optimal_moe_config is before


def test_the_captured_config_survives_downstream_mutation(monkeypatch):
    """fused_experts_impl mutates the dict it is handed, so the object returned
    is not the object that was chosen. Without the deepcopy the row would record
    whatever the kernel overwrote BLOCK_SIZE_M with."""
    primary, names, _ = fake_vllm(monkeypatch, {512: TUNED_ENTRY})
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        config = primary.try_get_optimal_moe_config((8, 14336, 4096),
                                                    (8, 4096, 7168), 2, "bf16", 512)
        config["BLOCK_SIZE_M"] = 999
    assert FC.tile_meta_from_capture(capture)["tile_block_m"] == 16


def test_a_missing_tuned_file_is_recorded_as_the_default_ladder(monkeypatch):
    """The finding this column exists for: nothing ships for
    NVIDIA_A100-SXM4-80GB at any of the four shapes, so those cells took the
    hardcoded M<=32/96/512 ladder while the H200's mixtral cells did not."""
    primary, names, _ = fake_vllm(monkeypatch, None)
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        primary.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                           2, "bf16", 787)
    meta = FC.tile_meta_from_capture(capture)
    assert meta["tile_config_source"] == "vllm_default"
    assert meta["tile_block_m"] == 64
    # No tuned file, so there is no key, and 0 means UNRECORDED here too.
    assert "tile_config_key" not in meta


def test_an_unwatchable_lookup_is_never_reported_as_default(monkeypatch):
    """"The lookup said there is no tuned file" and "nothing watched the lookup"
    are different facts, and only the first justifies writing vllm_default. The
    tile ints are still written: what is unknown in this branch is where the
    config came from, not what it was."""
    primary, names, _ = fake_vllm(monkeypatch, None, with_lookup=False)
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        primary.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                           2, "bf16", 787)
    meta = FC.tile_meta_from_capture(capture)
    assert meta["tile_config_source"] == "unrecorded"
    assert meta["tile_block_m"] == 64


def test_a_forced_tile_is_never_reported_as_vllms_own_choice(monkeypatch):
    """tile_sweep.py forces a tile through override_config, and a row measured
    under one says nothing about what vLLM would have chosen. A forced tile can
    be identical to the chosen one, so this is asked rather than inferred."""
    primary, names, _ = fake_vllm(monkeypatch, {512: TUNED_ENTRY}, override=True)
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, names):
        primary.try_get_optimal_moe_config((8, 14336, 4096), (8, 4096, 7168),
                                           2, "bf16", 512)
    meta = FC.tile_meta_from_capture(capture, FC.vllm_override_active(names))
    assert meta["tile_config_source"] == "vllm_override"
    assert "tile_config_key" not in meta


def test_nothing_captured_is_reported_as_unrecorded():
    assert FC.tile_meta_from_capture(FC.TileCapture()) == {
        "tile_config_source": "unrecorded"}


def test_the_recorder_is_inert_where_vllm_is_absent():
    """Off the GPU box the probe finds nothing, and the context manager must
    still be enterable: the span degrades to an unrecorded row rather than the
    sweep failing."""
    capture = FC.TileCapture()
    with FC.recording_tile_config(capture, ("no.such.module",)):
        pass
    assert capture.calls == []
    assert FC.bindings_of("try_get_optimal_moe_config", ("no.such.module",)) == []


def test_M_is_bound_by_name_rather_than_by_position():
    """The parameter has moved position between vLLM versions, and args[4] that
    is silently wrong attributes a config to the wrong batch size."""
    def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M,
                                   block_shape=None):
        return {}

    assert FC.called_with_m(try_get_optimal_moe_config,
                            ((8,), (8,), 2, "bf16"), {"M": 787}) == 787
    assert FC.called_with_m(try_get_optimal_moe_config,
                            ((8,), (8,), 2, "bf16", 787), {}) == 787
    # Unbindable rather than guessed at.
    assert FC.called_with_m(try_get_optimal_moe_config, (), {}) is None


# --------------------------------------------------------------------------
# the driver-side hook
# --------------------------------------------------------------------------

OBSERVED: list = []


class _ObservingUpGemm(StageSpan):
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        st.h_up = R.grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets,
            2 * st.spec.model.intermediate_size)


@register
class TileReportingUpGemm(_ObservingUpGemm):
    name = "t_tile_reporting_up_gemm"

    def observe_tile_config(self, st: MoEState) -> dict:
        OBSERVED.append("observed")
        return {"tile_block_m": 16, "tile_block_n": 64, "tile_num_warps": 8,
                "tile_config_source": "vllm_tuned", "tile_config_key": 1024}


@register
class SilentUpGemm(_ObservingUpGemm):
    name = "t_tile_silent_up_gemm"


@register
class BrokenObserverUpGemm(_ObservingUpGemm):
    name = "t_tile_broken_observer_up_gemm"

    def observe_tile_config(self, st: MoEState) -> dict:
        raise AttributeError("module has no attribute 'try_get_optimal_moe_config'")


@register
class StrayColumnUpGemm(_ObservingUpGemm):
    name = "t_tile_stray_column_up_gemm"

    def observe_tile_config(self, st: MoEState) -> dict:
        # An observer is not a second route into the timing columns.
        return {"tile_config_source": "vllm_tuned", "ms_p50": 0.001}


@register
class BadSourceUpGemm(_ObservingUpGemm):
    name = "t_tile_bad_source_up_gemm"

    def observe_tile_config(self, st: MoEState) -> dict:
        return {"tile_block_m": 16, "tile_config_source": "vllm-tuned"}


def fake_timer(fn, warmup=1, iters=2, trials=1, l2_flush=True, flush_mb=8,
               flush_mode="read", target_ms=200.0, on_captured=None):
    OBSERVED.append("timed")
    fn()
    return T.TimingResult(ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05,
                          jitter_p90_over_p50=1.2, warmup=warmup, iters=iters or 2,
                          trials=trials, l2_flush=l2_flush, cuda_graph=False,
                          samples=3, flush_mb=flush_mb, flush_mode=flush_mode)


def sweep(tmp_path, impl, l2_modes=(True,)):
    cfg = D.RunConfig(out_dir=tmp_path, device="cpu", warmup=1, trials=1, iters=2,
                      l2_modes=l2_modes, graph_modes=(False,),
                      timer_eager=fake_timer, timer_graph=fake_timer,
                      clock_sampler=lambda: T.ClockState(1980, 45))
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"))
    names = ["ref_router", "ref_permute", impl, "ref_act", "ref_down_gemm",
             "ref_unpermute"]
    D.run_sweep([(spec, names, impl)], cfg, routing=lambda s: None,
                info={"gpu_name": "FakeH200", "sm_capability": "9.0"})
    return SC.read_csv(cfg.csv_path)


def test_the_driver_writes_the_tile_a_span_reports(tmp_path):
    OBSERVED.clear()
    row = sweep(tmp_path, "t_tile_reporting_up_gemm")[0]
    assert SC.tile_field(row, "tile_block_m") == 16
    assert SC.tile_field(row, "tile_config_key") == 1024
    assert SC.tile_field(row, "tile_config_source") == "vllm_tuned"
    assert SC.has_tile_config(row)


def test_the_observation_runs_outside_every_timed_region(tmp_path):
    """Structural, not conventional: the hook is called from the prologue block
    that already runs the correctness check, above the line where the timed
    callable is defined. So it happens ONCE per cell, before the first timer,
    however many timing modes the cell has."""
    OBSERVED.clear()
    rows = sweep(tmp_path, "t_tile_reporting_up_gemm", l2_modes=(True, False))
    assert len(rows) == 2
    assert OBSERVED == ["observed", "timed", "timed"]


def test_a_span_with_no_observer_leaves_the_row_saying_unrecorded(tmp_path):
    """Which is the truth, and is why the default is a value rather than a blank
    column: "unrecorded" is a claim about the row."""
    row = sweep(tmp_path, "t_tile_silent_up_gemm")[0]
    assert row["tile_config_source"] == "unrecorded"
    assert not SC.has_tile_config(row)
    with pytest.raises(SC.TileConfigUnrecorded):
        SC.tile_field(row, "tile_block_m")


def test_a_broken_observer_costs_the_row_its_tile_and_not_the_cell(tmp_path):
    """An observer reaches into a framework's internals by design, so it can
    fail in ways no narrow except clause predicts. Losing a metered cell over it
    would be a worse trade than losing the column."""
    rows = sweep(tmp_path, "t_tile_broken_observer_up_gemm")
    assert len(rows) == 1
    assert rows[0]["correctness_passed"] == "True"
    assert float(rows[0]["ms_p50"]) > 0
    assert rows[0]["tile_config_source"] == "unrecorded"


def test_an_observer_may_not_write_outside_the_tile_columns(tmp_path):
    """_apply_meta silently drops a key that is not a column, so an observer
    that names one would appear to work and record nothing -- and one that names
    a TIMING column would be a hole in the invariant that no number reaches the
    CSV from outside the timer."""
    with pytest.raises(ValueError, match="ms_p50"):
        D._validated_tile_meta({"tile_config_source": "vllm_tuned",
                                "ms_p50": 0.001}, "t_stray")
    row = sweep(tmp_path, "t_tile_stray_column_up_gemm")[0]
    assert row["tile_config_source"] == "unrecorded"
    assert float(row["ms_p50"]) == 1.0


def test_a_typod_source_is_refused_rather_than_disappearing_from_a_group_by(tmp_path):
    """Closed for the reason TERMINAL_STATUSES is closed: a value no consumer
    matches does not fail loudly, those rows just vanish from every group-by."""
    with pytest.raises(ValueError, match="legal values"):
        D._validated_tile_meta({"tile_config_source": "vllm-tuned"}, "t_bad")
    row = sweep(tmp_path, "t_tile_bad_source_up_gemm")[0]
    assert row["tile_config_source"] == "unrecorded"


def test_a_whole_layer_row_finds_the_observer_inside_its_pipeline(tmp_path):
    """A pipeline-scoped row has no single span, and the tile the framework
    kernel chose is exactly as load-bearing there as in a span-scoped row."""
    cfg = D.RunConfig(out_dir=tmp_path, device="cpu", warmup=1, trials=1, iters=2,
                      l2_modes=(True,), graph_modes=(False,),
                      timer_eager=fake_timer, timer_graph=fake_timer,
                      clock_sampler=lambda: T.ClockState(1980, 45))
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"))
    names = ["ref_router", "ref_permute", "t_tile_reporting_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    impl = D.pipeline_scope_for("t_tile_reporting_up_gemm")
    D.run_sweep([(spec, names, impl)], cfg, routing=lambda s: None,
                info={"gpu_name": "FakeH200"})
    row = SC.read_csv(cfg.csv_path)[0]
    assert row["scope"] == "pipeline"
    assert SC.tile_field(row, "tile_block_m") == 16


def test_the_machine_block_carries_the_capability_when_a_device_answers(tmp_path):
    """sm_capability is the sm80-vs-sm90 discriminator the wgmma question needed
    and gpu_name is a marketing string, so it travels with the other machine
    facts rather than being derived downstream."""
    row = sweep(tmp_path, "t_tile_reporting_up_gemm")[0]
    assert SC.tile_field(row, "sm_capability") == "9.0"


def test_runtime_info_reports_no_capability_where_there_is_no_device():
    """Off the GPU box the column stays empty rather than being invented."""
    import torch

    info = T.runtime_info()
    if torch.cuda.is_available():
        assert info["sm_capability"].count(".") == 1
    else:
        assert "sm_capability" not in info


def test_the_registered_baselines_declare_an_honest_source():
    """torch's grouped_mm has no Triton tile to record and SGLang's selection has
    not been probed the way vLLM's was, so both name their selector and leave
    every int at 0 rather than borrowing vLLM's numbers."""
    import moe

    assert "cutlass_static" in SC.TILE_SOURCES
    assert "sglang" in SC.TILE_SOURCES
    moe.bootstrap("baselines")  # vLLM and SGLang are skipped off their venvs
    span = registry().get("torch_grouped_mm_up")
    if span is None:            # torch too old for grouped_mm; load_all skips it
        pytest.skip("torch_grouped_mm is not registered in this environment")
    meta = D._validated_tile_meta(span.observe_tile_config(None),
                                  "torch_grouped_mm_up")
    assert meta == {"tile_config_source": "cutlass_static"}
