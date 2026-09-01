"""The offline tile resolver must answer exactly what vLLM 0.27.1 would answer.

Every one of these tests defends a way the resolver could be quietly wrong and
still look right. That matters more here than usual, because the values it
produces are DERIVED and will be read next to measured ones: a resolver that is
off by one step in the ladder does not raise, it hands back a plausible
BLOCK_SIZE_M, and this study has already spent days on exactly that.

Three properties are load-bearing and each one has a test that fails if it is
guessed rather than reproduced: the NEAREST-key lookup (not a floor), the
tie-break to file order, and the difference between "upstream ships no file for
this shape" and "upstream ships one and nobody vendored it".
"""
from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_deployment_shapes import SHIPPED  # noqa: E402

from moe.bench import schema as SC  # noqa: E402
from moe.bench import tile_resolve as TR  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

PUBLISHED = Path(__file__).resolve().parents[1] / "results" / "published"

H200 = "NVIDIA H200"
A100 = "NVIDIA A100-SXM4-80GB"

#: `(E, N)` for the four models this study measured, off `w2_shape`, which is
#: what vLLM unpacks. Restating them as literals here would let a wrong shape
#: agree with itself.
SHAPES = {name: (cfg.w2_shape[0], cfg.w2_shape[2])
          for name, cfg in MODEL_CONFIGS.items()}


def resolve(model: str, gpu: str, tokens: int, dtype: str = "bf16") -> TR.DerivedTile:
    experts, intermediate = SHAPES[model]
    return TR.resolve_tile(experts, intermediate, dtype, gpu, tokens)


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------

def test_the_snapshot_listing_is_the_same_327_names_the_shape_tests_transcribed():
    """Two independent transcriptions of one upstream tree, pinned equal.

    The failure mode of a hand-copied listing is truncation, and a truncated
    listing does not error: it answers "no file ships" for everything that fell
    off the end, which is the wrong answer in the direction the resolver trusts.
    """
    assert TR.shipped_file_names() == SHIPPED
    assert len(TR.shipped_file_names()) == 327


def test_every_vendored_file_is_one_the_tag_actually_ships():
    """A JSON in the snapshot that upstream does not ship would make the resolver
    report a tuned config for a shape the measured run could only have taken the
    fallback ladder on."""
    vendored = {p.name for p in TR.SNAPSHOT_DIR.glob("*.json")}
    assert vendored, "the snapshot is empty"
    assert vendored <= TR.shipped_file_names()


def test_the_snapshot_holds_exactly_the_shapes_this_study_can_reach():
    """Four files, and the test says which four. If a fifth appears without this
    list changing, either a new cell was added or something was vendored that no
    row can reach, and both deserve to be noticed."""
    assert {p.name for p in TR.SNAPSHOT_DIR.glob("*.json")} == {
        "E=8,N=14336,device_name=NVIDIA_H200.json",
        "E=64,N=2560,device_name=NVIDIA_H200.json",
        "E=8,N=14336,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
        "E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
    }


def test_the_snapshot_documents_the_tag_it_came_from():
    """A vendored file with no provenance is indistinguishable from one somebody
    hand-edited, which is the whole reason this directory exists."""
    text = (TR.SNAPSHOT_DIR / "SNAPSHOT.md").read_text()
    assert TR.VLLM_TAG in text
    assert "sha256" in text.lower()


def test_a_file_that_ships_but_is_not_vendored_raises_instead_of_falling_back(
        tmp_path, monkeypatch):
    """THE ONE CASE THAT MUST NEVER BE SILENT. "no tuned file exists" and "the
    tuned file was not vendored" both look like "not on disk" and give DIFFERENT
    tiles, so the resolver has to consult the shipped listing and stop."""
    monkeypatch.setattr(TR, "SNAPSHOT_DIR", tmp_path)
    TR.tuned_configs.cache_clear()
    try:
        with pytest.raises(TR.SnapshotMissing) as excinfo:
            resolve("mixtral-8x7b", H200, 1)
        assert "E=8,N=14336,device_name=NVIDIA_H200.json" in str(excinfo.value)
        # and a shape upstream really has no file for still resolves, to the ladder
        assert resolve("deepseek-v3", H200, 1).provenance == TR.DERIVED_DEFAULT
    finally:
        TR.tuned_configs.cache_clear()


def test_the_vendored_ladders_keep_upstream_file_order():
    """The tie-break reads the first key of several equal minima, and "first"
    means file order. A snapshot re-serialised with sorted keys would move
    mixtral's M=112 and M=192 answers without changing a single value."""
    for path in TR.SNAPSHOT_DIR.glob("*.json"):
        raw = json.loads(path.read_text())
        from_file = [int(k) for k in raw]
        from_resolver = [key for key, _ in TR.tuned_configs(path.name)]
        assert from_resolver == from_file


# --------------------------------------------------------------------------
# the filename
# --------------------------------------------------------------------------

def test_the_device_selector_collapses_whitespace_runs_and_slashes_not_just_spaces():
    """`re.sub(r"[\\s/]+", "_", ...)`, which a `.replace(" ", "_")` gets wrong on
    both counts."""
    assert TR.device_selector("NVIDIA  L40 / S") == "NVIDIA_L40_S"
    assert TR.device_selector(" NVIDIA A100-SXM4-80GB ") == "NVIDIA_A100-SXM4-80GB"


def test_the_h200_family_folds_onto_one_name_and_gh200_does_not():
    """The fold matches a whole underscore-separated token. Without it an
    `NVIDIA H200 NVL` pod predicts a file that ships for nobody and every tuned
    row on such a pod resolves to the ladder instead."""
    assert TR.device_selector("NVIDIA H200 NVL") == "NVIDIA_H200"
    assert TR.device_selector("NVIDIA H200") == "NVIDIA_H200"
    assert TR.device_selector("NVIDIA GH200 480GB") == "NVIDIA_GH200_480GB"


def test_bf16_carries_no_dtype_suffix_and_fp8_selects_the_w8a8_file():
    """`_get_config_dtype_str` returns None for bf16, so the bf16 files have no
    suffix at all. Getting this backwards names a file that exists for the other
    dtype, so the lookup succeeds and the tile is simply wrong."""
    assert TR.config_dtype_selector("bf16") is None
    assert TR.config_dtype_selector("fp8_e4m3") == "fp8_w8a8"
    assert TR.config_dtype_selector("fp8_e5m2") == "fp8_w8a8"
    assert resolve("mixtral-8x7b", H200, 1).config_file.endswith("NVIDIA_H200.json")
    assert resolve("mixtral-8x7b", H200, 1, "fp8_e4m3").config_file.endswith(
        "dtype=fp8_w8a8.json")


def test_a_dtype_with_no_known_selector_raises_rather_than_defaulting_to_bf16():
    """int4 and int8 rewrite N in the lookup AND take a different default branch.
    Silently returning None for them would resolve a real file for the wrong
    shape."""
    with pytest.raises(TR.TileNotDerivable):
        TR.config_dtype_selector("int4_w4a16")


def test_a_block_shape_is_refused_because_it_takes_a_branch_this_does_not_model():
    """Real fp8 MoE serving usually passes `block_shape=[128, 128]`, which picks
    a different tile entirely. No cell in this study passes one, and answering
    for it anyway would put a wrong tile on a future arm."""
    with pytest.raises(TR.TileNotDerivable):
        TR.resolve_tile(8, 14336, "fp8_e4m3", H200, 1, block_shape=[128, 128])


# --------------------------------------------------------------------------
# the lookup itself
# --------------------------------------------------------------------------

def test_mixtral_on_h200_resolves_block_m_64_from_113_to_192_and_128_from_193():
    """The interval FINDINGS states, checked at both edges.

    It exists because the lookup is NEAREST: 113 is closer to the key 128 than
    to 96, and 193 is closer to 256 than to 128. A floor lookup would give 32
    and 64 across this whole range and would put a different mechanism under
    mixtral's cross-card deviation.
    """
    assert resolve("mixtral-8x7b", H200, 112).block_m_derived == 32
    for tokens in (113, 128, 160, 192):
        assert resolve("mixtral-8x7b", H200, tokens).block_m_derived == 64, tokens
    for tokens in (193, 256, 512, 4096):
        assert resolve("mixtral-8x7b", H200, tokens).block_m_derived == 128, tokens


def test_an_exact_tie_resolves_to_the_key_the_file_lists_first():
    """M=112 is 16 from both 96 and 128; M=192 is 64 from both 128 and 256. `min`
    keeps the first of equal minima and the keys are inserted in file order, so
    both go DOWN. A sorted or reversed snapshot would flip both."""
    assert resolve("mixtral-8x7b", H200, 112).config_key_derived == 96
    assert resolve("mixtral-8x7b", H200, 192).config_key_derived == 128


def test_the_lookup_is_nearest_and_not_a_floor():
    """qwen2's measured crossing sits at 787 tokens, which selects the key 1024
    and not 512. This is the single sentence in schema.py that says why
    `tile_config_key` exists."""
    assert resolve("qwen2-57b-a14b", H200, 787).config_key_derived == 1024


def test_the_lookup_key_is_the_token_count_and_never_the_row_count():
    """`fused_experts_impl` sets `M = hidden_states.size(0)`. Feeding it `T*k`
    would climb the ladder `k` times too early: mixtral at T=100 with k=2 would
    resolve as if it were 200 tokens and jump from 32 to 128."""
    assert resolve("mixtral-8x7b", H200, 100).block_m_derived == 32
    assert resolve("mixtral-8x7b", H200, 200).block_m_derived == 128


def test_mixtral_on_the_a100_has_no_tuned_file_and_walks_the_fallback_ladder():
    """FINDINGS C5 defect 3. Nothing ships for NVIDIA_A100-SXM4-80GB at any of
    this study's four shapes, so all four A100 cells took the hardcoded ladder,
    and the two cards therefore ran different tiles at the measured crossings."""
    tile = resolve("mixtral-8x7b", A100, 256)
    assert tile.provenance == TR.DERIVED_DEFAULT
    assert not TR.ships(tile.config_file)
    for tokens in (97, 256, 512):
        assert resolve("mixtral-8x7b", A100, tokens).block_m_derived == 64, tokens
    assert resolve("mixtral-8x7b", A100, 513).block_m_derived == 128


@pytest.mark.parametrize("model", ["mixtral-8x7b", "qwen2-57b-a14b",
                                   "deepseek-v2-lite", "deepseek-v3"])
def test_no_a100_cell_in_this_study_reaches_a_tuned_config(model):
    for dtype in ("bf16", "fp8_e4m3"):
        assert resolve(model, A100, 256, dtype).provenance == TR.DERIVED_DEFAULT


def test_unsharded_deepseek_v3_is_tuned_for_no_device_at_all():
    """`E=256,N=2048` is the shape that needs 1.3 TB of bf16 weights, and nobody
    serves it, which is why the run log says `Using default MoE config`. C3's
    third rescope turns on this being a default rather than a grid-search
    optimum."""
    assert SHAPES["deepseek-v3"] == (256, 2048)
    for gpu in (H200, A100):
        for dtype in ("bf16", "fp8_e4m3"):
            tile = resolve("deepseek-v3", gpu, 16, dtype)
            assert tile.provenance == TR.DERIVED_DEFAULT
            assert not TR.ships(tile.config_file)
    assert not any(name.startswith("E=256,N=2048,") for name in TR.shipped_file_names())


def test_deepseek_v2_lite_is_tuned_only_for_a_card_this_study_never_ran():
    """`E=64,N=1408` DOES ship, for exactly one device: NVIDIA_B200. Neither card
    in this study is it, so both deepseek-v2-lite cells took the ladder -- but
    the resolver has to reach that answer through the device selector rather than
    through "nothing exists at this shape", and a test that asserted the latter
    would pass today and mislead the first B200 run."""
    assert SHAPES["deepseek-v2-lite"] == (64, 1408)
    assert {n for n in TR.shipped_file_names() if n.startswith("E=64,N=1408,")} == {
        "E=64,N=1408,device_name=NVIDIA_B200.json"}
    for gpu in (H200, A100):
        assert resolve("deepseek-v2-lite", gpu, 256).provenance == TR.DERIVED_DEFAULT
    assert TR.config_file_name(64, 1408, None, "NVIDIA_B200") in TR.shipped_file_names()


def test_the_fallback_ladder_is_the_one_v0_27_1_hardcodes():
    """M<=32 -> 16, M<=96 -> 32, M<=512 -> 64, else 128, with the rest of the
    branch checked too: the M ladder is the part every write-up quotes and the
    other four fields are the part nobody does."""
    assert [TR.default_config(m, 8)["BLOCK_SIZE_M"] for m in (1, 32, 33, 96, 97, 512, 513)] \
        == [16, 16, 32, 32, 64, 64, 128]
    assert TR.default_config(64, 8)["BLOCK_SIZE_N"] == 64
    assert TR.default_config(65, 8)["BLOCK_SIZE_N"] == 128
    assert TR.default_config(65, 8)["BLOCK_SIZE_K"] == 64
    assert TR.default_config(65, 8, "fp8_w8a8")["BLOCK_SIZE_K"] == 128
    assert TR.default_config(128, 8)["num_warps"] == 4
    assert TR.default_config(129, 8)["num_warps"] == 8
    assert TR.default_config(32, 8)["num_stages"] == 4
    assert TR.default_config(33, 8)["num_stages"] == 3


def test_the_fallback_group_size_m_stays_at_1_across_this_studys_whole_grid():
    """`GROUP_SIZE_M` is 16 only when `M // E > 128`. deepseek-v3 at E=256 would
    need more than 32768 tokens and the largest cell here is 8192, so the whole
    fallback half of the pool sits at one swizzle width. That is why the
    per-GROUP_SIZE_M split of the alpha refit is underpowered rather than
    negative."""
    assert TR.default_config(8192, 256)["GROUP_SIZE_M"] == 1
    assert TR.default_config(129 * 256, 256)["GROUP_SIZE_M"] == 16


def test_a_batch_of_zero_is_refused_rather_than_resolved():
    with pytest.raises(TR.TileNotDerivable):
        resolve("mixtral-8x7b", H200, 0)


# --------------------------------------------------------------------------
# derived can never be mistaken for observed
# --------------------------------------------------------------------------

def test_no_derived_field_name_is_a_schema_column():
    """Structural, and checked again here so the reason is written down: a
    derived value must not be writable into a v4 observed column by a rename, a
    typo, or a `**asdict()`."""
    names = {f for f in TR.DerivedTile.__dataclass_fields__}
    assert not names & set(SC.COLUMNS)
    assert all(name.endswith("_derived")
               for name in names if name.startswith(("block_", "group_", "num_",
                                                     "config_key")))


def test_the_derived_provenance_strings_are_not_schema_tile_sources():
    """Equal strings are how a report that groups by source would silently pool a
    derived row with an observed one."""
    assert TR.DERIVED_TUNED not in SC.TILE_SOURCES
    assert TR.DERIVED_DEFAULT not in SC.TILE_SOURCES


def test_every_derived_tile_says_it_is_derived_when_asked_to_describe_itself():
    for tile in (resolve("mixtral-8x7b", H200, 256), resolve("deepseek-v3", A100, 16)):
        assert "DERIVED" in tile.describe()
        assert tile.vllm_tag == TR.VLLM_TAG
        assert tile.observed is False


def test_a_derived_tile_cannot_be_constructed_claiming_it_was_observed():
    """`observed` was documented as permanently False while being merely
    defaulted to False, so `DerivedTile(**{**asdict(tile), "observed": True})`
    produced a derived tile that claimed to have been measured. Every other
    guard in the module is downstream of a consumer trusting that flag."""
    honest = resolve("mixtral-8x7b", H200, 256)
    forged = dict(dataclasses.asdict(honest), observed=True)
    with pytest.raises(ValueError, match="permanently False"):
        TR.DerivedTile(**forged)
    # and the honest object still round-trips through its own column dict
    assert TR.DerivedTile(**dataclasses.asdict(honest)) == honest
    assert honest.as_columns()["observed"] is False


def test_a_span_whose_tile_is_not_vllms_is_refused_rather_than_answered():
    """SGLang ships its own tuned tree and torch is CUTLASS. Handing either of
    them vLLM's answer is the substitution this module exists to prevent."""
    for impl in ("sglang_fused_experts", "torch_grouped_mm_up",
                 "torch_scaled_grouped_mm_down", "__pipeline__"):
        with pytest.raises(TR.TileNotDerivable):
            TR.resolve_tile_for_row({"impl": impl, "model": "mixtral-8x7b",
                                     "gpu_name": H200, "dtype": "bf16",
                                     "num_tokens": "256"})


def test_a_row_with_no_gpu_name_has_no_device_selector_and_is_refused():
    with pytest.raises(TR.TileNotDerivable):
        TR.resolve_tile_for_row({"impl": "vllm_fused_experts", "model": "mixtral-8x7b",
                                 "gpu_name": "", "dtype": "bf16", "num_tokens": "256"})


def test_a_v3_row_has_nothing_observed_to_disagree_with():
    """`disagreement_with_observed` returning None on a v3 row is the honest
    answer and NOT a pass: there is no observation to check the derivation
    against, which is the entire reason this module had to be written."""
    row = {"schema_version": "3", "impl": "vllm_fused_experts",
           "model": "mixtral-8x7b", "gpu_name": H200, "dtype": "bf16",
           "num_tokens": "256", "tile_block_m": SC.UNRECORDED}
    assert TR.disagreement_with_observed(row) is None


def test_a_v4_row_whose_observation_contradicts_the_derivation_says_so_loudly():
    """The check that turns this module from an argument into a claim. It cannot
    run until a v4 arm exists, so the test builds one."""
    row = {"schema_version": "4", "impl": "vllm_fused_experts",
           "model": "mixtral-8x7b", "gpu_name": H200, "dtype": "bf16",
           "num_tokens": "256", "tile_block_m": "64",
           "tile_config_source": "vllm_tuned"}
    message = TR.disagreement_with_observed(row)
    assert message is not None
    assert "OBSERVED BLOCK_M=64" in message
    assert "DERIVES 128" in message
    row["tile_block_m"] = "128"
    assert TR.disagreement_with_observed(row) is None


# --------------------------------------------------------------------------
# over the published corpus
# --------------------------------------------------------------------------

def _published_vllm_rows() -> list[dict]:
    rows = []
    for path in sorted(PUBLISHED.glob("*/run_*.csv")):
        rows.extend(r for r in SC.read_csv(path) if r.get("impl") in TR.VLLM_IMPLS)
    return rows


def test_every_published_vllm_row_gets_a_derived_block_m():
    """The deliverable, stated as a test. Before this module not one of the
    36,000-odd vLLM rows in `results/published/` could say which tile it ran;
    every one of them can now say which tile vLLM 0.27.1 would have picked, and
    the answer is a step function of the token count with four values in it."""
    rows = _published_vllm_rows()
    assert len(rows) > 30_000
    census: collections.Counter = collections.Counter()
    for row in rows:
        tile = TR.resolve_tile_for_row(row)
        census[tile.block_m_derived] += 1
    assert set(census) == {16, 32, 64, 128}
    assert sum(census.values()) == len(rows)


def test_the_published_corpus_splits_into_tuned_and_ladder_the_way_findings_says():
    """2 of 8 (model x card) cells have a tuned file, both on the H200, and the
    A100 half of the corpus is entirely on the ladder."""
    tuned_cells, ladder_cells = set(), set()
    for row in _published_vllm_rows():
        tile = TR.resolve_tile_for_row(row)
        cell = (row["model"], row["gpu_name"], row["dtype"])
        (tuned_cells if tile.from_tuned_file else ladder_cells).add(cell)
    assert all(gpu == H200 for _, gpu, _ in tuned_cells)
    assert {model for model, _, _ in tuned_cells} == {"mixtral-8x7b", "qwen2-57b-a14b"}
    # The A100 half is entirely on the ladder, which is C5 defect 3.
    assert any(gpu == A100 for _, gpu, _ in ladder_cells)
    assert not any(gpu == A100 for _, gpu, _ in tuned_cells)
    # And on the H200 the ladder cells are exactly the two deepseek geometries.
    assert {model for model, gpu, _ in ladder_cells if gpu == H200} == {
        "deepseek-v3", "deepseek-v2-lite"}


def test_no_published_row_is_a_v4_row_so_none_of_this_is_checkable_against_a_run():
    """The standing caveat, kept as an executable statement. The day a v4 arm
    lands here this test fails, and the thing to do then is not to delete it but
    to run `disagreement_with_observed` over the arm."""
    for row in _published_vllm_rows():
        assert not SC.has_tile_config(row)


def test_the_census_command_covers_every_derivable_row_and_says_nothing_is_observed(
        capsys):
    """The deliverable as a command. It exits 0 with an explicit statement that
    the derivation has NOT been validated against a run, which is the honest
    reading of an all-v3 corpus and is not the same as a pass."""
    assert TR._main([str(p) for p in sorted(PUBLISHED.glob("*/run_*.csv"))]) == 0
    out = capsys.readouterr().out
    assert "DERIVED" in out
    assert "vllm_tuned_derived" in out and "vllm_default_derived" in out
    assert "NOT a pass" in out


def test_the_census_shows_the_fp8_rescope_c3_states():
    """C3's second rescope, made checkable: the SAME H200 shapes tuned for fp8
    pick a warpgroup-sized tile from M=1, while their bf16 twins sit at 16 and 32
    through the small-M range. If that ever stopped being true the rescope would
    need rewriting, and this is where it would show."""
    rows = _published_vllm_rows()
    table = TR.census(rows)
    for model in ("mixtral-8x7b", "qwen2-57b-a14b"):
        fp8 = table[(model, H200, "fp8_e4m3", TR.DERIVED_TUNED)]
        bf16 = table[(model, H200, "bf16", TR.DERIVED_TUNED)]
        assert min(fp8) >= 64, (model, fp8)
        assert min(bf16) == 16, (model, bf16)


def test_the_census_skips_a_row_it_cannot_derive_rather_than_bucketing_it():
    """A skipped row must be a MISSING row, never a zero and never a default: a
    census that quietly bucketed torch rows under some BLOCK_M would report tile
    counts for a kernel that has no Triton config at all."""
    rows = [{"impl": "vllm_fused_experts", "model": "mixtral-8x7b",
             "gpu_name": H200, "dtype": "bf16", "num_tokens": "256"},
            {"impl": "sglang_fused_experts", "model": "mixtral-8x7b",
             "gpu_name": H200, "dtype": "bf16", "num_tokens": "256"}]
    table = TR.census(rows)
    assert sum(sum(b.values()) for b in table.values()) == 1
