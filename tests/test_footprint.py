"""Device memory and disk are different questions.

Conflating them makes DeepSeek-V3 look impossible when its geometry is routine
to benchmark: a sweep generates random weights for ONE layer and downloads
nothing at all.
"""
import pytest

from moe.bench.footprint import cell_footprint, download_bytes, worst_cell
from moe.spec import MODEL_CONFIGS, BenchSpec

H200 = 141e9


def spec(name, tokens=4096, dtype="bf16"):
    return BenchSpec(MODEL_CONFIGS[name], tokens, dtype)


@pytest.mark.parametrize("name", ["mixtral-8x7b", "qwen2-57b-a14b",
                                  "deepseek-v2-lite", "deepseek-v3"])
def test_every_sweep_model_fits_on_one_h200(name):
    assert cell_footprint(spec(name)).fits_in(H200), name


def test_deepseek_v3_is_cheap_to_benchmark_and_impossible_to_download():
    """The whole point. Its geometry costs one layer of weights; its checkpoint
    is two orders of magnitude larger and does not fit on any single GPU."""
    fp = cell_footprint(spec("deepseek-v3"))
    assert fp.peak < 40e9
    assert fp.fits_in(H200)
    assert download_bytes(MODEL_CONFIGS["deepseek-v3"]) > 1000e9
    assert download_bytes(MODEL_CONFIGS["deepseek-v3"]) > 30 * fp.peak


def test_weights_dominate_at_many_experts():
    fp = cell_footprint(spec("deepseek-v3"))
    assert fp.weights > 5 * fp.activations


def test_activations_dominate_at_few_experts_and_many_tokens():
    fp = cell_footprint(spec("mixtral-8x7b", tokens=16384))
    assert fp.activations > fp.weights


def test_footprint_scales_with_tokens_only_through_activations():
    a, b = cell_footprint(spec("mixtral-8x7b", 1024)), cell_footprint(spec("mixtral-8x7b", 4096))
    assert a.weights == b.weights
    assert b.activations == pytest.approx(4 * a.activations, rel=0.02)


def test_fp8_halves_the_weight_footprint():
    a = cell_footprint(spec("deepseek-v3", dtype="bf16")).weights
    b = cell_footprint(spec("deepseek-v3", dtype="fp8_e4m3")).weights
    assert b == pytest.approx(a / 2)


def test_download_size_scales_with_moe_layer_count():
    lite = MODEL_CONFIGS["deepseek-v2-lite"]
    assert lite.num_moe_layers == 26
    assert 25e9 < download_bytes(lite) < 35e9, "should land near the 31 GB checkpoint"


def test_worst_cell_finds_the_one_that_ooms_first():
    specs = [spec("mixtral-8x7b", 128), spec("deepseek-v3", 4096),
             spec("deepseek-v2-lite", 4096)]
    chosen, fp = worst_cell(specs)
    assert chosen.model.name == "deepseek-v3"
    assert fp.peak == cell_footprint(chosen).peak


def test_empty_sweep_has_no_worst_cell():
    assert worst_cell([]) == (None, None)
