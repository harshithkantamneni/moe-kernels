"""vLLM v0.27.1's `fused_moe_kernel`, copied with two instruments added (rental 4).

THIS IS NOT A KERNEL OF THE STUDY'S. It is upstream's kernel, byte for byte, plus two
instruments the owner approved on 2026-10-06 (decision 5), each OFF by default:

  (a) TIMESTAMPS. `%globaltimer` (ns) and `%clock64` (SM cycles) read through
      `tl.inline_asm_elementwise`, plus `%smid`, written to a per-CTA int64 row of
      `stamps_ptr`: CTA start (before the dead check), a dead CTA's exit, the
      prologue's end (before the k-loop in the source, which compiles to BEFORE the
      software pipeliner's peeled fill: the fill lands in iteration 0, and no source point
      separates it without changing the loop), the epilogue's start (after the loop, before
      the routed-weight load) and the end (after the final store); with STAMPS=2 also
      clock64 at the top of every STAMP_EVERY-th k-iteration (and at its end too when
      STAMP_EVERY is 1, the rental-4 layout); with STAMPS=3 ("ends") only the CTA start,
      a dead CTA's exit and the end, so no stamp sits next to the k-loop; with STAMPS=4
      ("sample", rental 5) the CTA start, a dead CTA's exit, the end and clock64 at the TOP
      of k-iteration 0 and of every k with k mod STAMP_EVERY = STAMP_PHASE (the host sets
      STAMP_PHASE = (S - 1) mod STAMP_EVERY, so the last iteration is always marked), each
      only on sampled CTAs: a live CTA when pid mod STAMP_CTA_MOD is 0, its iteration tops
      when also pid mod STAMP_ITER_MOD is 0, a dead CTA when pid mod STAMP_DEAD_MOD is 0
      (every modulus 1 by default; the sampling is a runtime test inside a constexpr
      STAMPS == 4 block, so levels 0 to 3 compile exactly as before).
      Rental 4's perturbation gate (results/published/2026-10-07-*-rental4-session) found
      that any stamp changes ptxas's whole-kernel register allocation (with per-CTA stamps
      the PTX of the k-loop is unchanged; w2 at BK 64 s4: 55 -> 45 or 48 registers, 4 -> 5
      CTAs per SM)
      and that one clock64 + st.global per k-iteration costs 7.5 to 11% at equal
      occupancy; levels 3 and STAMP_EVERY > 1 are the cadences proposed against that.
  (b) EVICTION HINTS. An `eviction_policy` on the A and B `tl.load` of the default
      path (not USE_TD, not SWAP_AB): "" (tl.load's own default), "evict_first" or
      "evict_last". Triton 3.7.1 DROPS it on pipelined loads (they lower to cp.async .ca /
      .cg), so a hinted unit is refused unless the compiled PTX carries L2::cache_hint.

No tiling, scheduling or optimisation is changed: the two upstream functions below
are the excerpt `upstream_v0.27.1_fused_moe.excerpt` (sha256 in UPSTREAM.json) with
lines marked `# INSTR` added, and `tests/test_instrumented_kernel.py` holds that: the
copy's AST, with every `if STAMPS ...:` block, the six instrument parameters and the two
`eviction_policy=` keywords removed, equals upstream's, and every line that is not an
upstream line carries the marker. With STAMPS=0 and both hints "" every instrument is a
constexpr-dead branch or tl.load's default argument.

The kernel keeps upstream's NAME (`fused_moe_kernel`), so ncu's kernel filter and the
counter route's launch accounting read it as they read the plain kernel. It is launched
only through `moe.instrumented.install` (a patch of vLLM's module attribute, undone on
exit) and only by an instrumented unit; the plain installed kernel stays the measured
object of every registered timing test.

Row layout (moe.instrumented.COLUMNS): 0 smid; 1, 2 start (globaltimer, clock64); 3, 4
prologue end; 5, 6 epilogue start; 7, 8 end or dead exit; 9 kind (1 live, 2 dead, -1 a
row no instrument reached: the expert -1 path); then STAMP_MARKS pairs of clock64 (top,
end of iteration). Imports Triton at import time: the VM's vLLM interpreter only.

Upstream licence: Apache-2.0, SPDX-FileCopyrightText: Copyright contributors to the vLLM
project. The instruments: MIT, this repository's licence.
"""
import triton
import triton.language as tl

#: the header columns of a stamp row (moe.instrumented.HDR holds the same number)
STAMP_HDR = tl.constexpr(10)


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _clock64():
    return tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _smid():
    return tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _stamp_row(stamps_ptr, pid, STAMP_MARKS: tl.constexpr):
    return stamps_ptr + pid.to(tl.int64) * (STAMP_HDR + 2 * STAMP_MARKS)


@triton.jit
def _stamp_start(stamps_ptr, pid, STAMP_MARKS: tl.constexpr):
    t = _globaltimer()
    c = _clock64()
    row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
    tl.store(row + 1, t)
    tl.store(row + 2, c)
    tl.store(row + 0, _smid().to(tl.int64))


@triton.jit
def _stamp_pair(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, COL: tl.constexpr):
    t = _globaltimer()
    c = _clock64()
    row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
    tl.store(row + COL, t)
    tl.store(row + COL + 1, c)


@triton.jit
def _stamp_kind(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, KIND: tl.constexpr):
    row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
    tl.store(row + 9, tl.full((), KIND, tl.int64))


@triton.jit
def _stamp_iter(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, STAMP_EVERY: tl.constexpr, k,
                SIDE: tl.constexpr):
    if k % STAMP_EVERY == 0:
        j = k // STAMP_EVERY
        if j < STAMP_MARKS:
            c = _clock64()
            row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
            tl.store(row + STAMP_HDR + 2 * j + SIDE, c)


@triton.jit
def _sampled(pid, MOD: tl.constexpr):
    if MOD == 1:
        return pid >= 0
    else:
        return pid % MOD == 0


@triton.jit
def _sample_read():
    # the CTA start's timers, held in registers: nothing is stored before the dead check
    # (build-r5-review F1: a store there made every dead CTA write under DEAD_MOD 17's v2)
    return _globaltimer(), _clock64(), _smid().to(tl.int64)


@triton.jit
def _sample_start(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, MOD: tl.constexpr, t, c, sm):
    # the start stamp of a CTA whose own path (live: CTA_MOD, dead: DEAD_MOD) samples it
    if _sampled(pid, MOD):
        row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
        tl.store(row + 1, t)
        tl.store(row + 2, c)
        tl.store(row + 0, sm)


@triton.jit
def _sample_exit(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, MOD: tl.constexpr,
                 KIND: tl.constexpr):
    if _sampled(pid, MOD):
        _stamp_pair(stamps_ptr, pid, STAMP_MARKS, 7)
        _stamp_kind(stamps_ptr, pid, STAMP_MARKS, KIND)


@triton.jit
def _sample_top(stamps_ptr, pid, STAMP_MARKS: tl.constexpr, STAMP_EVERY: tl.constexpr,
                STAMP_PHASE: tl.constexpr, CTA_MOD: tl.constexpr, ITER_MOD: tl.constexpr, k):
    # the slot rule of moe.instrumented.sample_slot: phase 0, k // every; else 0 for k = 0
    # and 1 + k // every for k = phase (mod every)
    if STAMP_PHASE == 0:
        hit = k % STAMP_EVERY == 0
        j = k // STAMP_EVERY
    else:
        hit = (k == 0) | (k % STAMP_EVERY == STAMP_PHASE)
        j = tl.where(k == 0, 0, 1 + k // STAMP_EVERY)
    if hit & _sampled(pid, CTA_MOD) & _sampled(pid, ITER_MOD):
        if j < STAMP_MARKS:
            c = _clock64()
            row = _stamp_row(stamps_ptr, pid, STAMP_MARKS)
            tl.store(row + STAMP_HDR + 2 * j, c)


# ---- the two upstream functions, instrumented (lines marked # INSTR are the instruments) ----

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    # Tensor-descriptor path for the A gather and B load in the K-loop.
    USE_TD: tl.constexpr = False,
    # INSTRUMENTATION (moe-kernels, rental 4), every parameter off by default:  # INSTR
    stamps_ptr=None,  # INSTR int64 [num_ctas, STAMP_HDR + 2 * STAMP_MARKS], or None
    STAMPS: tl.constexpr = 0,  # INSTR 0 off, 1 per CTA, 2 per CTA and per k-iteration, 3 CTA ends only
    STAMP_EVERY: tl.constexpr = 1,  # INSTR an iteration mark every N k-iterations
    STAMP_MARKS: tl.constexpr = 0,  # INSTR iteration marks a row holds
    EVICT_A: tl.constexpr = "",  # INSTR eviction_policy of the A load ("" is tl.load's default)
    EVICT_B: tl.constexpr = "",  # INSTR eviction_policy of the B load
    STAMP_PHASE: tl.constexpr = 0,  # INSTR sample level: the marked k's residue, (S - 1) mod STAMP_EVERY
    STAMP_CTA_MOD: tl.constexpr = 1,  # INSTR sample level: a live CTA is stamped when pid mod this is 0
    STAMP_ITER_MOD: tl.constexpr = 1,  # INSTR sample level: its iteration tops when also pid mod this is 0
    STAMP_DEAD_MOD: tl.constexpr = 1,  # INSTR sample level: a dead CTA is stamped when pid mod this is 0
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    if STAMPS >= 1 and STAMPS <= 3:  # INSTR CTA start, before the dead check
        _stamp_start(stamps_ptr, pid, STAMP_MARKS)  # INSTR
    if STAMPS == 4:  # INSTR CTA start, read into registers (sample level; stored past the dead check)
        s4_t, s4_c, s4_sm = _sample_read()  # INSTR
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        if STAMPS >= 1 and STAMPS <= 3:  # INSTR a dead CTA's exit
            _stamp_pair(stamps_ptr, pid, STAMP_MARKS, 7)  # INSTR
            _stamp_kind(stamps_ptr, pid, STAMP_MARKS, 2)  # INSTR
        if STAMPS == 4:  # INSTR a dead CTA's start and exit, sampled by STAMP_DEAD_MOD
            _sample_start(stamps_ptr, pid, STAMP_MARKS, STAMP_DEAD_MOD, s4_t, s4_c, s4_sm)  # INSTR
            _sample_exit(stamps_ptr, pid, STAMP_MARKS, STAMP_DEAD_MOD, 2)  # INSTR
        return
    if STAMPS == 4:  # INSTR a live CTA's start, sampled by STAMP_CTA_MOD
        _sample_start(stamps_ptr, pid, STAMP_MARKS, STAMP_CTA_MOD, s4_t, s4_c, s4_sm)  # INSTR
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # TD gather and the SWAP_AB accumulator layout are mutually exclusive.
    tl.static_assert(not (USE_TD and SWAP_AB))
    if USE_TD:
        # ``tt.descriptor_gather`` requires block_shape[0] == 1 and i32 idx.
        m_td = num_valid_tokens // top_k
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=(m_td, K),
            strides=(stride_am, stride_ak),
            block_shape=(1, BLOCK_SIZE_K),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + off_experts * stride_be,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        gather_idx = (offs_token // top_k).to(tl.int32)
    elif SWAP_AB:
        a_ptrs = a_ptr + (
            offs_k[:, None] * stride_ak + offs_token[None, :] // top_k * stride_am
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        )
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if STAMPS == 1 or STAMPS == 2:  # INSTR the prologue's end, before the k-loop
        _stamp_pair(stamps_ptr, pid, STAMP_MARKS, 3)  # INSTR
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if STAMPS == 2:  # INSTR the top of k-iteration k
            _stamp_iter(stamps_ptr, pid, STAMP_MARKS, STAMP_EVERY, k, 0)  # INSTR
        if STAMPS == 4:  # INSTR the top of k-iteration k, phase-aligned and sampled
            _sample_top(stamps_ptr, pid, STAMP_MARKS, STAMP_EVERY, STAMP_PHASE, STAMP_CTA_MOD, STAMP_ITER_MOD, k)  # INSTR
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if USE_TD:
            a = a_desc.gather(gather_idx, k * BLOCK_SIZE_K)
            b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]).T
        elif SWAP_AB:
            a_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & token_mask[None, :]
            b_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
                eviction_policy=EVICT_A,  # INSTR
            )
            b = tl.load(  # INSTR the upstream line above, split to carry eviction_policy
                b_ptrs,  # INSTR
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,  # INSTR
                other=0.0,  # INSTR
                eviction_policy=EVICT_B,  # INSTR
            )  # INSTR
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if SWAP_AB:
                    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    if SWAP_AB:
                        accumulator = tl.dot(b, a, acc=accumulator)
                    else:
                        accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        if not USE_TD:
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        if STAMPS == 2 and STAMP_EVERY == 1:  # INSTR the end of k-iteration k (no readout uses it past every=1)
            _stamp_iter(stamps_ptr, pid, STAMP_MARKS, STAMP_EVERY, k, 1)  # INSTR

    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))
    if STAMPS == 1 or STAMPS == 2:  # INSTR the epilogue's start, before the routed-weight load
        _stamp_pair(stamps_ptr, pid, STAMP_MARKS, 5)  # INSTR

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    if STAMPS >= 1 and STAMPS <= 3:  # INSTR after the final store
        _stamp_pair(stamps_ptr, pid, STAMP_MARKS, 7)  # INSTR
        _stamp_kind(stamps_ptr, pid, STAMP_MARKS, 1)  # INSTR
    if STAMPS == 4:  # INSTR after the final store, sampled
        _sample_exit(stamps_ptr, pid, STAMP_MARKS, STAMP_CTA_MOD, 1)  # INSTR

