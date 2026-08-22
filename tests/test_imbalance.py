import pytest

from moe.routing.imbalance import counts_from_offsets, expert_load, padded_rows, tile_efficiency


def test_uniform_load_is_maximally_balanced():
    load = expert_load([64] * 8)
    assert load.max_over_mean == 1.0
    assert load.cv == 0.0
    assert load.entropy_norm == pytest.approx(1.0)
    assert load.gini == pytest.approx(0.0, abs=1e-9)
    assert load.empty_experts == 0
    assert load.top1_share == pytest.approx(1 / 8)


def test_single_hot_expert_is_maximally_imbalanced():
    load = expert_load([512] + [0] * 7)
    assert load.active_experts == 1
    assert load.empty_experts == 7
    assert load.max_over_mean == 8.0
    assert load.entropy_norm == pytest.approx(0.0)
    assert load.top1_share == 1.0


def test_skew_ordering_is_monotonic():
    balanced = expert_load([32] * 8)
    mild = expert_load([64, 48, 32, 32, 24, 24, 16, 16])
    severe = expert_load([200, 40, 8, 4, 2, 1, 0, 0])
    assert balanced.cv < mild.cv < severe.cv
    assert balanced.entropy_norm > mild.entropy_norm > severe.entropy_norm
    assert balanced.gini < mild.gini < severe.gini


def test_empty_routing_does_not_divide_by_zero():
    load = expert_load([0, 0, 0, 0])
    assert load.total_rows == 0
    assert load.empty_experts == 4
    assert load.entropy_norm == 0.0


def test_negative_counts_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        expert_load([4, -1])


def test_counts_from_offsets_round_trip():
    assert counts_from_offsets([0, 3, 3, 10]) == [3, 0, 7]


def test_counts_from_offsets_validates():
    """One validating implementation, used in production. The unvalidated
    duplicate that production used to call is gone."""
    with pytest.raises(ValueError, match="must start at 0"):
        counts_from_offsets([1, 4])
    with pytest.raises(ValueError, match="non-decreasing"):
        counts_from_offsets([0, 5, 2])
    with pytest.raises(ValueError, match="empty"):
        counts_from_offsets([])


def test_tile_efficiency_is_carried_on_every_load():
    """The padding waste this project targets is a column, not an afterthought."""
    balanced = expert_load([128] * 8)
    assert balanced.tile_eff_bm128 == 1.0 and balanced.tile_eff_bm64 == 1.0
    pathological = expert_load([1] * 256)
    assert pathological.tile_eff_bm128 == pytest.approx(1 / 128)
    assert pathological.tile_eff_bm64 == pytest.approx(1 / 64)
    assert "load_tile_eff_bm128" in pathological.as_row()


# --- the quantitative form of the fixed-BLOCK_M limitation ------------------

def test_balanced_load_wastes_nothing_when_aligned():
    counts = [128] * 8
    assert padded_rows(counts, 128) == 1024
    assert tile_efficiency(counts, 128) == 1.0


def test_ragged_groups_pad_up_to_whole_tiles():
    counts = [129, 1, 255]
    assert padded_rows(counts, 128) == 256 + 128 + 256
    assert tile_efficiency(counts, 128) == pytest.approx(385 / 640)


def test_many_tiny_groups_are_the_pathological_case():
    """256 experts each holding one row: a BLOCK_M of 128 computes 128x the
    work that is actually needed. This is the number that motivates a
    load-imbalance-aware tiling."""
    counts = [1] * 256
    assert tile_efficiency(counts, 128) == pytest.approx(1 / 128)
    assert tile_efficiency(counts, 16) == pytest.approx(1 / 16)


def test_smaller_block_m_helps_ragged_loads():
    counts = [17, 33, 5, 200]
    assert tile_efficiency(counts, 32) > tile_efficiency(counts, 128)


def test_empty_experts_cost_nothing():
    assert padded_rows([0, 0, 128], 128) == 128


def test_block_m_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        padded_rows([4], 0)


def test_unknown_load_does_not_claim_perfect_tile_efficiency():
    """The zero-load row is where the load is UNKNOWN. Letting the dataclass
    default stand would have the CSV assert 1.0, i.e. no padding waste at all,
    exactly where nothing is known."""
    load = expert_load([0, 0, 0, 0])
    assert load.tile_eff_bm64 == 0.0
    assert load.tile_eff_bm128 == 0.0
    assert tile_efficiency([0, 0, 0, 0], 128) == 0.0
