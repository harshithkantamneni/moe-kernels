"""The sweep benchmarks a shape nobody serves, and these tests pin why.

Every published cell in this study is TP=1 bf16. Real MoE serving shards, which
changes `N` and therefore vLLM's config lookup, the tile and the block count,
and quantises, usually fp8 with `block_shape=[128,128]`.

DeepSeek-V3 is `hidden=7168, moe_intermediate=2048, E=256`, so its unsharded
kernel shape is `E=256,N=2048`. NO vLLM config ships at that key for ANY device,
because it is the shape that needs 1.3 TB of bf16 weights and nobody runs it.
What ships is `E=256,N=256` and `E=256,N=512` -- the TP=8 and TP=4 widths -- on
H200, H100, B200 and others. So the run log's `Using default MoE config.
Performance might be sub-optimal!` is not a gap in vLLM's coverage; it is this
study benchmarking a configuration that is not deployed.

The listing below is the evidence, taken at the tag rather than remembered, and
committed so a reader without a network can check the claim:

    curl -s "https://api.github.com/repos/vllm-project/vllm/contents/\\
vllm/model_executor/layers/fused_moe/configs?ref=v0.27.1&per_page=100" \\
      | python3 -c "import json,sys; [print(e['name']) for e in json.load(sys.stdin)]"

327 files at v0.27.1. Reproduced here are every `E=256` name (27 of them, the
whole DeepSeek-V3 expert count across all devices) plus the H200 names at the
`E=8` and `E=64` widths this study's other two models shard to. Nothing else is
asserted about, so nothing else is copied.
"""
import pytest

from moe.bench import profiles as PR
from moe.bench.footprint import cell_footprint, download_bytes, worst_cell_by_model
from moe.bench.ridge import crossing_batch
from moe.spec import MODEL_CONFIGS, BenchSpec

#: Bytes on one card, and the two the study has actually rented.
H200_BYTES = 141e9
A100_80_BYTES = 80e9

#: Every `E=256` config vLLM ships at v0.27.1, across all devices. DeepSeek-V3
#: has 256 routed experts, so this is the complete set of files its kernel could
#: ever match, and `N=2048` is not among them.
SHIPPED_E256 = frozenset({
    "E=256,N=1024,device_name=AMD_Instinct_MI325X,block_shape=[128,128].json",
    "E=256,N=1024,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json",
    "E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8.json",
    "E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json",
    "E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json",
    "E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=128,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_H20.json",
    "E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=256,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=384,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=512,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=512,device_name=NVIDIA_H100_80GB_HBM3.json",
    "E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json",
    "E=256,N=64,device_name=NVIDIA_A800-SXM4-80GB.json",
})

#: The H200 names at the widths mixtral and qwen2 shard to. Both models ship a
#: config in BOTH bf16 (no dtype selector) and plain `fp8_w8a8` (no block shape)
#: at TP=1 and at TP=8, which makes them the only pair in this study that can be
#: measured tuned-against-tuned across a shard.
SHIPPED_H200_OTHER = frozenset({
    "E=8,N=14336,device_name=NVIDIA_H200.json",
    "E=8,N=14336,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
    "E=8,N=1792,device_name=NVIDIA_H200.json",
    "E=8,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
    "E=64,N=2560,device_name=NVIDIA_H200.json",
    "E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
    "E=64,N=320,device_name=NVIDIA_H200.json",
    "E=64,N=320,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
})

SHIPPED = SHIPPED_E256 | SHIPPED_H200_OTHER


def config_key(model: str) -> tuple[int, int]:
    """`(E, N)` exactly as vLLM derives it.

    `try_get_optimal_moe_config` does `E, _, N = w2_shape`, so reading the pair
    off `w2_shape` here rather than off `num_experts` and `intermediate_size`
    means this test cannot agree with the harness while disagreeing with vLLM.
    """
    E, _, N = MODEL_CONFIGS[model].w2_shape
    return E, N


def config_file_name(model: str, dtype: str | None = None,
                     block_shape: list[int] | None = None,
                     device_name: str = "NVIDIA_H200") -> str:
    """vLLM's `get_config_file_name`, reimplemented from the v0.27.1 source."""
    E, N = config_key(model)
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_selector = ("" if not block_shape or not all(block_shape)
                      else f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_selector}.json"


# --------------------------------------------------------------------------
# The shape the study has been benchmarking
# --------------------------------------------------------------------------

def test_unsharded_deepseek_v3_is_tuned_for_no_device_at_all():
    """The limitation this whole file exists to record. Not "no H200 config":
    no config anywhere, because `E=256,N=2048` is 1.3 TB of bf16 weights and is
    not a shape anyone serves."""
    assert config_key("deepseek-v3") == (256, 2048)
    assert not [n for n in SHIPPED_E256 if n.startswith("E=256,N=2048,")]
    # And every OTHER N at E=256 does have a file, so the absence is about this
    # width rather than about the expert count.
    assert {int(n.split("N=")[1].split(",")[0]) for n in SHIPPED_E256} == \
        {64, 128, 256, 384, 512, 1024}


def test_the_shard_widths_that_ship_are_exactly_tp8_and_tp4_on_this_card():
    """2048/8 = 256 and 2048/4 = 512, and both are shipped for H200. Those are
    the deployment widths, and they are the ones the new configs describe."""
    assert config_key("deepseek-v3-tp8") == (256, 256)
    assert config_key("deepseek-v3-tp4") == (256, 512)
    for model in ("deepseek-v3-tp8", "deepseek-v3-tp4"):
        assert config_file_name(model, "fp8_w8a8", [128, 128]) in SHIPPED_E256


def test_a_tp16_shard_would_also_be_a_real_width_but_not_on_this_card():
    """2048/16 = 128 ships for H100, H20 and L20Y and NOT for H200, which is why
    TP=16 is not in `MODEL_CONFIGS`: it would add a shape whose interesting
    property cannot be exercised on the card this study rents."""
    from moe.spec import tensor_parallel_shard

    tp16 = tensor_parallel_shard(MODEL_CONFIGS["deepseek-v3"], 16)
    assert tp16.w2_shape[2] == 128
    assert any(n.startswith("E=256,N=128,") for n in SHIPPED_E256)
    assert not [n for n in SHIPPED_E256
                if n.startswith("E=256,N=128,device_name=NVIDIA_H200")]


# --------------------------------------------------------------------------
# What reaching the SHAPE does not buy: reaching the tuned CONFIG
# --------------------------------------------------------------------------

def test_the_deepseek_shards_still_miss_their_tuned_file_in_this_harness():
    """THE HONEST HALF OF THIS WORK, and the reason the profile carries a
    caveat. Every H200 file at `E=256,N=256` and `E=256,N=512` is
    `dtype=fp8_w8a8,block_shape=[128,128]`, and this harness quantises one scale
    per EXPERT, so `_framework_config.vllm_quant_spec` sets `block_shape: None`
    and the name it looks up has no block-shape selector. That file does not
    exist, so the shard lands back on the same hardcoded fallback the unsharded
    shape did.

    The shape is now the served one either way, which is what changes the tile
    ladder, the block count and the memory footprint. Reaching the TUNED config
    additionally needs block-wise scales in `moe/quant.py` and the matching flag
    in `_framework_config.py`, neither of which exists.
    """
    from moe.baselines._framework_config import vllm_quant_spec

    spec = BenchSpec(MODEL_CONFIGS["deepseek-v3-tp8"], 128, "fp8_e4m3")
    quant = vllm_quant_spec(spec)
    assert quant["kind"] == "fp8_w8a8" and quant["block_shape"] is None

    asked = config_file_name("deepseek-v3-tp8", quant["kind"], quant["block_shape"])
    assert asked == "E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8.json"
    assert asked not in SHIPPED, "if this ever ships, delete this test"
    # The one that does ship, and what the harness would have to ask for.
    assert config_file_name("deepseek-v3-tp8", "fp8_w8a8", [128, 128]) in SHIPPED


@pytest.mark.parametrize("model", ["mixtral-8x7b-tp8", "qwen2-57b-a14b-tp8"])
def test_the_mixtral_and_qwen2_shards_do_reach_a_tuned_config_as_built(model):
    """These two are in the profile for exactly this reason: at TP=8 they hit a
    tuned H200 file in bf16 AND in the plain per-expert fp8 the harness already
    produces, so they can answer "does the shard change the kernel" without
    anyone first teaching the harness block-wise scales."""
    assert config_file_name(model) in SHIPPED                     # bf16
    assert config_file_name(model, "fp8_w8a8") in SHIPPED         # fp8, no block
    base = model.rsplit("-tp", 1)[0]
    assert config_file_name(base) in SHIPPED
    assert config_file_name(base, "fp8_w8a8") in SHIPPED
    # And it is a DIFFERENT file, which is the whole point of running both.
    assert config_file_name(model) != config_file_name(base)


# --------------------------------------------------------------------------
# What the prediction says a shard should do, so the sweep can refute it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["deepseek-v3", "mixtral-8x7b",
                                   "qwen2-57b-a14b"])
def test_sharding_must_not_move_the_predicted_crossing(model):
    """The falsifiable prediction the deployment profile exists to test.
    `crossing_batch` is `ridge * b * E / (2k)`: no `F` in it, because every
    weight element is used once per row whatever shape the matrices are. TP
    divides `F` and leaves `E` and `k` alone, so `2R/b` says a shard crosses at
    the SAME batch as the model it is a slice of.

    That is a strong claim, since the shard runs a different tile, a different
    block count and an eighth of the weight traffic. If the sweep moves the
    crossing, `2R/b` is incomplete in a way no dtype or device test has caught.
    """
    shard = f"{model}-tp8"
    for ridge in PR.H200_RIDGE_BAND:
        assert crossing_batch(shard, ridge) == crossing_batch(model, ridge)


def test_sharding_does_move_the_arithmetic_intensity_at_a_fixed_batch():
    """The other side of it, so the test above is not read as "sharding changes
    nothing". A shard reads an eighth of the weight bytes and the same
    activation bytes, so the FULL byte model's intensity does move, and the
    crossing solved from it moves with it. Only the `2R/b` limit is invariant.
    """
    from moe.bench.bytes_model import grouped_gemm_only_cost

    def ai(model):
        cfg = MODEL_CONFIGS[model]
        spec = BenchSpec(cfg, num_tokens=4096, dtype="bf16")
        return grouped_gemm_only_cost(spec, cfg.num_experts).arithmetic_intensity

    assert ai("deepseek-v3-tp8") < ai("deepseek-v3")


# --------------------------------------------------------------------------
# Memory, known before anything is rented
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["deployment", "crossing-uniform"])
def test_every_shape_in_a_new_profile_fits_on_one_card(profile):
    """Per SHAPE, not per profile. `deepseek-v3` at TP=1 dominates every profile
    it appears in, so a global worst cell that fits says nothing about the seven
    other geometries beside it, and the rent is committed before the OOM."""
    worst = worst_cell_by_model(PR.get(profile).specs())
    assert worst, profile
    for name, (spec, fp) in worst.items():
        assert fp.fits_in(H200_BYTES), (name, spec.label, fp.peak)
        assert fp.fits_in(A100_80_BYTES), (name, spec.label, fp.peak)


def test_the_biggest_cell_the_new_grid_adds_is_still_cheap():
    """The crossing grid reaches T=16384, past anything the study has run. That
    is the cell that would OOM first if any would: DeepSeek-V3, bf16, top of the
    grid. It costs 40 GB, under a third of one H200's budget."""
    spec = BenchSpec(MODEL_CONFIGS["deepseek-v3"], max(PR.CROSSING_TOKENS), "bf16")
    fp = cell_footprint(spec)
    assert 35e9 < fp.peak < 45e9
    assert fp.fits_in(H200_BYTES) and fp.fits_in(A100_80_BYTES)


def test_a_shard_is_cheaper_to_benchmark_than_the_model_it_slices():
    """Weights divide by the shard width; activations do not, since `H` and the
    token count are untouched. So the saving is real and bounded."""
    def peak(model):
        return cell_footprint(BenchSpec(MODEL_CONFIGS[model], 8192, "bf16")).peak

    assert peak("deepseek-v3-tp8") < peak("deepseek-v3-tp4") < peak("deepseek-v3")
    weights = cell_footprint(BenchSpec(MODEL_CONFIGS["deepseek-v3-tp8"], 8192)).weights
    whole = cell_footprint(BenchSpec(MODEL_CONFIGS["deepseek-v3"], 8192)).weights
    assert weights == whole // 8


def test_sharding_does_not_shrink_the_checkpoint():
    """The distinction `full_intermediate_size` exists for. A rank holds a
    slice; the checkpoint holds the whole expert, and it is the checkpoint that
    decides what can be captured. Reporting a TP=8 entry at an eighth of its
    size would make DeepSeek-V3 look capturable, which is the exact confusion
    `footprint.py` was written to prevent."""
    whole = download_bytes(MODEL_CONFIGS["deepseek-v3"])
    for shard in ("deepseek-v3-tp4", "deepseek-v3-tp8"):
        assert download_bytes(MODEL_CONFIGS[shard]) == whole
    assert whole > 1000e9
    # Still an order of magnitude beyond any single card, sharded or not.
    assert whole > 30 * cell_footprint(
        BenchSpec(MODEL_CONFIGS["deepseek-v3-tp8"], 4096)).peak
