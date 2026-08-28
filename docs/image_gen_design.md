# image_gen design discussion (2026-08-28)

Working notes from a design conversation exploring block-local/parallel decode for
`summformer_jax/image_gen/`, comparing against Fractal Generative Models (arxiv.org/abs/2502.17437),
`qcute/qcute_lagcodec/`, and ARMD (arxiv.org/html/2601.16971v1). Captures the reasoning, not just
conclusions -- several early framings turned out to be wrong or incomplete and the corrections
matter as much as the final positions.

## Starting point: why summformer's trunk is structurally different from Fractal's

Fractal recursively reshapes a 256x256x3 image into a 4-level patch hierarchy (256 patches ->
16 subpatches/patch -> 16 pixels/subpatch -> 3 RGB channels/pixel), each level a genuine nested
transformer conditioned on its parent's single output token. `summformer_jax`'s trunk instead
processes one flat raster-byte sequence throughout, with `fuse_stages` periodically pooling and
cross-attending a coarser summary back in -- no reshape, no nested per-unit transformer spawn.
That's a real, honest structural difference, not just an implementation detail (see chat's
step-by-step walkthrough comparing the two directly).

## Finding 1: `main_window` didn't actually save compute (fixed)

`sdpa_with_sink`/`causal_mask` materialize the full dense `(B,H,T,T)` score matrix and only mask
entries in the softmax -- a small `main_window` changed *which* entries counted, not how many got
computed. Confirmed empirically: a 64x64x3 (seq_len=12288) train step OOM-killed (exit 137) purely
from the dense score tensor, e.g. `(2,4,12288,12288)` fp32 ~= 4.8GB per layer.

**Fix**: ported `chunked_windowed_attention` from `qcute/qcute_lagcodec/qcute_lagcodec_common.py:860`
(torch) into `summformer_jax/image_gen/summformer.py` (JAX) -- reshapes into `window`-sized blocks,
each block attends to itself + the immediately preceding block (`2*window` keys, not `T` keys),
giving real `O(T*window)` cost. Verified numerically identical to the old dense-masked path
(`max_abs_diff` ~2.4e-7, fp32 rounding noise) via a standalone comparison, and confirmed the
previously-OOMing 64x64 config now trains cleanly. Exact-match reasoning: for a query at local
position `li` in its chunk, a same-chunk key is always within window by construction; a
previous-chunk key at local position `lj` is within window iff `li < lj` -- exactly what
`causal_mask`'s `i-j < window` condition gives, not an approximation.

## Finding 2: point-sampling pooling is empirically weak, not just theoretically

`_pool_and_fuse`'s `source[:, K-1::K, :]` is stride *selection* (grabs the literal embedding of
one byte every K positions), not span *aggregation*. Confirmed via a receptive-field probe
(`check_block_locality` in `image_gen/summformer.py`) on a tiny config (K=8): byte 79 (a
K-aligned position, `8*10-1`) was visible to a much later query regardless of window, because it's
literally sampled -- byte 0 (not K-aligned) was invisible even at long range. So most raw bytes
(K-1 out of every K) are structurally invisible to the whole reconnection channel.

**Caveat on "just use more layers/bigger window instead"**: stacked local-window self-attention
does grow receptive field with depth (same mechanism as dilated convs, effective field ~=
`insert_after * window`), so point-sampling isn't fundamentally broken if `insert_after * window
>= K` is deliberately satisfied. But that costs the same order of FLOPs (`O(K)`) as an explicit
pooling/aggregation op would, so it's a code-simplicity argument, not a compute-savings one -- and
it forces a real config constraint most current configs don't satisfy (e.g. K=768 at window=12
would need ~64 layers before the first fuse-stage).

## Finding 3: genuine block-parallel decode needs a role split, not just windowing

**The circularity**: whatever `source_index` is used, the code for block B is derived from
content that requires block B's own bytes to already be decided (either its trunk-processed
hidden state, or -- even with `source_index=0`, decoupled from trunk *processing* -- its own raw
byte *values*). This is backwards from what parallel decode needs: block B's bytes should depend
on its code, not the reverse. No amount of windowing or masking on a *single* trunk escapes this;
pooling always happens after the content it summarizes.

**Not "need another full trunk"**: the fix isn't necessarily duplicating the trunk -- `ConfigV2`
already has a smaller, separate model for this (`code_lms`). The missing piece is using it in a
genuinely *predictive* mode at generation time (autoregressive over already-decided prior codes
only, e.g. `qcute_lagcodec`'s `code_source='pred'` path: `encode_up_to(model, all_bytes, level=1)`)
instead of pooling from the current (not-yet-decided) block's own content. What's required in
general is a *role* separation -- something that resolves a block's code before and independently
of that block's bytes, and something else that decodes bytes block-locally given an already-
resolved code -- not necessarily separate weights. Fractal implements this with separate per-level
transformers; `qcute_lagcodec` with separate `encoders[level]` vs. the `stage_lms` decoder stack;
a shared-weight dual-mode version (same code-LM, "pool from ground truth" during training /
"predict autoregressively over prior codes" at generation) would satisfy the same requirement
without literal duplication.

## `qcute_lagcodec`'s block-local mechanism (`qcute_lagcodec_decoder.py:1986`, `StackDecoderLocal`)

`block_local_track0_decode` is the concrete, already-debugged reference implementation of the role
split above:
- **Code is a real encoder pass**, not point-sampling -- a separate small transformer compresses a
  block's ground-truth bytes into one code vector.
- **A trainable constant seed token** (`bb.self_code_const`, one shared D-dim vector, not
  per-block) bootstraps byte 0 of each block via cross-attention to that block's own code --
  provably zero self-attention contribution under same-block causal masking (proven in the
  function's own docstring: a same-block causal key would need a position before the block's own
  start, impossible for any K>=1), so no K/V cache is needed for the seed at all.
- **True parallelism via batching, not masking**: folds `n_blocks` into the batch dimension and
  runs ordinary causal self-attention *within* each K-length block only -- all blocks decode in
  one parallel batched call.
- **Documented train/inference mismatch bug** (`StackDecoderLocal`'s own docstring, 2026-08-23):
  naive generation reused the base class's cross-block-visible conditioning path, which the model
  was never trained to produce/consume under `block_local_track0_decode`'s same-block-only
  training -- collapsed to repetitive garbage at ~99.5% teacher-forced byte_acc. Fix: every new
  block's hidden state must be computed by calling `block_local_track0_decode` fresh with
  `n_blocks=1`, identical to training, never reusing cross-block machinery at inference.

**Inference is less uniform than training, and it's intrinsic, not sloppy**: training builds the
seed-then-real hidden-state sequence via one vectorized `torch.cat` (no branch). Inference's byte
loop has an explicit `h_query = out["h_seed"][...] if t == 0 else out["h_real"][..., t-1, :]` --
a genuine special case, because at t=0 there's no real previous byte yet. This is intrinsic to any
BOS-substitute AR scheme, not fixable in general -- but it CAN be pushed from loop *body* to loop
*setup* (precompute the seed's hidden state once, seed a buffer, then the loop body is uniform).
Matters more in JAX than torch: a Python-level `if t==0` inside an unrolled `for t in range(K)` (K
static) is fine (resolved at trace time), but would need `lax.cond` (real dispatch overhead) or
this same pre-seed restructuring to work cleanly inside `jax.lax.scan` for larger K.

## Two seed-conditioning designs, audited

1. **Trainable constant seed token, cross-attention-only, separate from real bytes' QKV**
   (`qcute_lagcodec`'s approach): safety is *structural* (provable zero self-attn leakage), not
   something to re-verify per case. Conditioning funnels entirely through the code (a real
   bottleneck, same critique as Fractal's single-vector handoff).
2. **Fractal-style: project some real hidden state, splice into native QKV**: richer signal in
   principle, but safety now depends entirely on *which* state is projected -- projecting the
   trunk's own running state (`source_index=-1`-style) reintroduces exactly the circularity in
   Finding 3. Projecting from the *code itself* (already fully resolved before decode starts)
   would be safe by the same precondition constant-seed relies on -- the real axis isn't
   "constant vs. projected," it's "does the seed's info come from something already resolved
   before decode starts, or from something entangled with the thing being parallelized."

Recommendation reached: constant seed + varying per-block code cross-attention (design 1) -- not
because design 2 is inherently unsafe, but because design 1's safety is structural/provable and
the smaller diff to actually implement and test incrementally (matches how the windowing fix above
was verified: numeric equivalence check before trusting it).

**Correction during discussion**: an earlier critique ("single shared prefill hidden state
broadcast to seed multiple blocks goes stale, since later blocks get zero information about
intervening blocks' content") does NOT apply to design 1 -- that critique targets a *different*,
weaker proposal that drops the per-block code channel entirely and asks a single shared hidden
state to carry both bootstrap-safety AND content-differentiation. Design 1's seed token carries
zero content by design; all differentiation comes from the code, which is fresh per block by
construction (assuming the code-LM producing it is itself a proper sequential AR chain over
codes). No staleness, because the thing that varies per block was never the seed to begin with.

## Fractal's teacher-forcing vs. `qcute_lagcodec`'s vs. a continuous encoder (three positions)

Confirmed and extended during discussion:

1. **Fractal**: every intermediate level's AR target is literal raw pixel/RGB data -- a genuine
   4x4x3 subpatch is always real, already-existing ground truth at any recursion depth, since
   Fractal never leaves the original modality, just recursively re-partitions the same real
   tensor. Teacher forcing needs a target and gets one for free at every level. This is *why* the
   recursive-reshape structure is imposed in the first place -- it's what makes "what's correct at
   this intermediate level" trivially well-defined.
2. **`qcute_lagcodec`**: NLL-valid, but intermediate levels predict abstract learned *codes*, not
   raw bytes -- no natural ground truth exists for an abstract code the way it does for a pixel
   patch. This is exactly why the encoder + `QuantScheme` (Simplex/Binary/Grid/GMM) machinery
   exists: something has to *manufacture* the teacher-forcing target. This is the genuine
   complexity cost of the abstract-code choice, separate from the seed-token/`stack_local` masking
   complexity (a different, generation-time-parallelism concern).
3. **Continuous (non-quantized) encoder** -- what `FuseStageV2`/`code_lms` already are, no VQ
   anywhere in `image_gen`'s current code: no discrete target to define at all, since nothing
   compares the code against a label. It's just a continuous feature trained by ordinary backprop
   through the single downstream byte-NLL loss, same as any other hidden state. Upgrading the
   current stride-sampling to a real pooling encoder (Finding 2's fix) stays in this category as
   long as it's kept continuous -- it does not require importing `qcute_lagcodec`'s quantizer
   complexity.

## ARMD (arxiv.org/html/2601.16971v1) -- Strided Block-Parallel decode, gist only (not copied)

Reframes masked-diffusion generation as block-wise causal: partitions the sequence into `S`
interleaved strided streams (positions `0,S,2S,...` = stream 0, etc.). Stream *heads* generate
sequentially first (ordinary causal AR, each head conditions on all previously-decided heads) --
once every head is resolved, the rest of every stream decodes simultaneously, since remaining
positions depend only on their own stream's resolved head plus their own earlier positions.

Mapped onto the block-local discussion: this is structurally the *same shape* as the code-based
role split above (short sequential coarse pass, then parallel fine pass) -- except the "code" is
a real sampled byte (the stream head), not an invented continuous or quantized code. That's a
genuine simplification: no new encoder or seed-constant module, just a different generation
*schedule* over the existing model, reusing the block-diagonal batching trick for the parallel
phase.

**Where the real cost is, and it's not architecture**: phase 1 requires the model to condition on
a sequence with genuine *gaps* (only stride-K positions filled, the rest unresolved) -- a normally
shifted-CE-trained causal LM has never seen that conditioning pattern. This is exactly what ARMD's
own progressive-permutation training schedule exists to fix; without adapting the training
objective, feeding a gapped sequence to a vanilla-trained model is out-of-distribution and likely
low quality. Phase 2 (block-diagonal fill-in given a resolved head) is safe by comparison -- same
already-proven isolation trick as `qcute_lagcodec`'s design, no gaps involved.

**Difficulty verdict reached**: the schedule change alone is cheap to implement (cheaper than the
seed-token design -- no new modules at all). Getting good quality from it needs a training
objective change (gap-tolerant conditioning), which is the actually hard part and orthogonal to
any of the architecture work above.

## Incremental KV-cache (trunk self-attn + fuse-stage cross-attn), implemented and verified

Added `Attn.forward_incremental`/`Block.forward_incremental` (ported from `lm/model_summformer.py`,
already proven there) plus a new `SummTransformerV2._make_incremental_stepper`/`generate_kv_cache`/
`check_kv_cache_consistency`, adapted for v2's arbitrary `insert_after` insertion points (vs v1's
single `n_fuse` loop) and mixed-dim code-LMs (code_in_proj/code_out_proj applied inside the
incremental path too, same as the dense path).

Verified via `check_kv_cache_consistency` (bit-exact match against `generate_no_cache`) across 7
structural variants: `source_index=0/-1/specific-layer`, `insert_after=0`, mixed-dim single-stage,
mixed-dim multi-stage, and unbounded (`main_window=None`) -- all `match_rate=1.0` under
`compute_dtype=jnp.float32`.

**One real bug caught and fixed during this**: `source_index=-1` needs the accumulated hidden-state
history at the depth the fuse-stage *fires at* (`insert_after`) tracked incrementally, not a
history keyed by the literal `-1` sentinel -- `referenced_depths` now maps `-1` to `spec[0]`
(`insert_after`) per stage before allocating the incremental history dict.

**One non-bug, worth knowing about**: at the *default* `compute_dtype=jnp.bfloat16`, one variant
(`source_index=specific layer`, n_layers=3) showed `match_rate=0.75` -- traced to logits differing
in the 4th-5th decimal place (e.g. `0.0043` vs `0.0045`), not a structural divergence. Confirmed by
switching to `float32`: `match_rate` becomes `1.0` immediately. Root cause: floating-point ops
aren't associative, and the chunked-windowed dense computation vs. the incrementally-cached
computation take genuinely different summation paths that are mathematically equivalent but not
bit-identical in bf16's ~7-bit mantissa -- occasionally enough to flip an argmax when top logits
are close. This is a property of greedy bf16 decoding in general, not specific to this port; worth
remembering as a real caveat before trusting a bf16 `check_kv_cache_consistency` result at face
value -- always cross-check at fp32 if a bf16 run shows anything less than 1.0.

## Mixed-dim trunk/code-LM, implemented

`fuse_stages` spec extended to an optional 7-tuple (`..., code_d_model, code_n_heads`), backward-
compatible with the existing 5-tuple form (both verified working). `code_in_proj`/`code_out_proj`
(plain `Linear`s) bridge trunk-dim <-> code-dim at the fuse-stage boundary, so `Attn`/`FuseStageV2`
themselves never need to support mixed dims directly -- cross-attention always operates in trunk
dim. Lets the trunk stay cheap/shallow (runs O(T) times) while code-LMs are wider/deeper (run
O(T/K) times, amortized). `configs/image64_mixdim.py` (3-level: trunk + 2 fuse-stages, trunk
d_model=128/n_layers=3/window=8, fuse-stage K=8 at code_d_model=256, fuse-stage K=64 at
code_d_model=512 matching gpt2-tiny's width) confirmed: trunk is 3.9% of total params, code-LMs
87.4% -- matches the design intent directly, not just by construction.

## Static (fixed-shape, jit-reusable) trunk KV cache

Motivation: `main_window` is fixed and known at trace time, so the trunk's sliding-window cache
doesn't need to grow via `concatenate`-then-truncate (which changes shape every call for the first
`window` steps, defeating `jax.jit` reuse across generation steps) -- it can be a fixed-size
circular buffer from the start.

Added `Attn.prime_static_cache` (prefill: dense/chunked-windowed forward over the whole prompt,
identical math to the existing path, plus seeds the fixed-size circular buffer) and
`Attn.forward_incremental_static`/`Block.forward_incremental_static` (decode: one new token,
`jax.lax.dynamic_update_slice_in_dim` write into a fixed-shape buffer, no shape change ever) --
plus `SummTransformerV2._make_static_incremental_stepper`/`generate_kv_cache_static`/
`check_kv_cache_consistency_static` wiring these into the same insertion-point/mixed-dim structure
as the existing (dynamic) stepper.

**One real bug caught and fixed**: priming placed the kept K/V/positions *sequentially* into cache
slots `[0..n_keep-1]`, but decode-time addresses slots by `absolute_position % cap` (true circular
addressing). These two schemes only coincide when `prompt_len % cap == 0` -- otherwise a
subsequent decode-time write lands on the wrong slot and evicts a still-valid entry instead of the
truly-oldest one, while a stale entry that should have been evicted lingers. Caught via
`check_kv_cache_consistency_static` failing on non-coincidental prompt lengths (e.g. `pl=12,
cap=8`) while passing on coincidental ones (`pl=16`, `pl=8` -- both `%8==0`), which is exactly the
kind of bug that "happens to pass" on an under-varied test battery. Fixed by scattering into
`slot = absolute_position % cap` via `.at[slots].set(...)` for both the T<cap and T>=cap cases
uniformly (one code path, no separate pad/truncate branches). Re-verified across 9 structural
variants (including per-layer-tuple `main_window`, and a deliberately non-multiple `prompt_len=17`
with 6 varied prompt lengths per case) -- all `match_rate=1.0` at fp32.

**Confirmed genuine `jax.jit` reuse** for the no-fuse-stages (pure trunk) case: a single
`jax.jit`-wrapped decode step, called 32 times with a changing `pos` argument (passed as a regular
traced value, not `static_argnums`), ran without any retracing errors -- one compile, reused for
every step.

**What's NOT statically-shaped yet, and why**: the fuse-stage/code-LM side (pooled history
accumulation via `concatenate`, and the code-LM recompute trigger `if n_blocks >
stage_n_blocks_done[stage_i]:`) still uses the original dynamic-shape approach. This is a
deliberate scope boundary, not an oversight: the code sequence's natural bound is
`context_len // K` (often not small), and making it static requires the SAME circular/sequential-
buffer technique applied recursively to the code-LM's own self-attention, PLUS converting the
recompute trigger from a Python-level shape-changing branch into a `jax.lax.cond`-gated one (both
branches must return same-shaped outputs) -- a comparable-scope second engineering task with its
own bit-exact verification surface, not completed this pass. Net effect: `generate_kv_cache_static`
avoids retracing for the trunk's own computation, but a decode call still retraces whenever some
fuse-stage's code buffer grows (every `K` steps) if the whole thing were wrapped in one outer
`jax.jit` -- which is why `generate_kv_cache_static` is NOT itself wrapped in `jax.jit` end-to-end
for configs with fuse-stages, only demonstrated as fully jittable for the fuse-stage-free case.

## Fully static (code-LM + pooling side included), implemented

Extended the trunk-only static stepper to the fuse-stage/code-LM side, closing the gap flagged in
the previous section. Every buffer is now fixed-shape: `hist_buf` (pooling source history,
sequential write via `dynamic_update_slice_in_dim`, sized to `context_len`, never evicted since
pooling needs the full causal past); the code-LM's own self-attention reuses the trunk's
`prime_static_cache`/`forward_incremental_static` methods directly (cap = `context_len // K`,
window=None -- literally the same code, no new attention primitive needed); a fixed-size
`h_code_out_buf` per code-LM layer (what `FuseStageV2`'s cross-attention reads); and the recompute
trigger is now `jax.lax.cond(need_update, do_update, no_update, ...)` instead of a Python-level
`if`, both branches returning the same pytree structure -- so a decode call no longer retraces when
a fuse-stage's code buffer grows. `self.max_n_blocks = context_len // K` per stage (static, known
at construction) sizes everything.

Per the explicit instruction to keep train/inference from silently diverging: training itself
needed NO changes -- it already processes the whole fixed `context_len` sequence in one shot per
step, so `n_blocks = L // K` was already constant across training batches; there was no
growing-shape loop on the training side to begin with. The actual guarantee against a
qcute_lagcodec-style train/inference mismatch is the same one used throughout this doc: bit-exact
verification (`check_kv_cache_consistency_fully_static`) against `generate_no_cache`, which calls
the exact `_cascade`/`_pool_and_fuse` function training itself uses -- not a structural claim that
padding/masking is "obviously" equivalent.

**Two real bugs caught and fixed**, both via failures on the fp32 consistency check (not rounding):
1. `code_lm_caches` initialized to `None` per layer -- `jax.lax.cond`'s two branches need matching
   pytree structure from the very first call, and `forward_incremental_static` (unlike the dynamic
   stepper's `forward_incremental`) has no `None`-handling path. Fixed by seeding proper
   zero-initialized `(k_buf, v_buf, pos_buf, write_pos)` tuples before the stepper starts.
2. The bigger one: dense `_pool_and_fuse` **skips the fuse-stage entirely** or (`return x`
   unchanged) when no codes exist yet (`n_blocks < 1`). The static version instead always called
   `FuseStageV2` under an all-invalid (sink-only) mask, relying on the zero-KV sink to avoid NaN --
   but the sink avoiding NaN is NOT the same as a true no-op: `FuseStageV2`'s residual MLP still
   applied on every call regardless of whether attention output was "just sink," silently
   diverging from dense's actual skip. Caught on the multi-stage config specifically (a stage whose
   `K` hadn't been reached yet, `insert_after` near the end of a short `n_layers`) -- single-stage
   configs happened not to exercise this window because their one stage usually had codes available
   well before generation started. Fixed by wrapping the whole `FuseStageV2` call in
   `jax.lax.cond(n_blocks_done > 0, ..., lambda: x_new)`, matching dense's skip exactly (both
   branches same shape: `(B, Tn, D)`).

Verified across 8 structural variants (single/multi-stage, mixed-dim, bounded/unbounded main and
fuse windows, `code_n_layers>1`, `prompt_len=17` non-coincidental) -- all `match_rate=1.0` at fp32,
`n_checks=6` each. `generate_kv_cache_fully_static`/`check_kv_cache_consistency_fully_static` are
the new entry points; `generate_kv_cache_static` (trunk-only) is left as-is, still correct, just
superseded in scope by this one.

## `image64_mixdim.py` config rationale

**Trunk** (`d_model=128, n_heads=4, n_layers=3, main_window=8`) -- deliberately cheap/shallow
because it runs once per byte (12,288 times per 64x64 image). `main_window=8` caps its
self-attention at `O(T*8)` instead of `O(T^2)` via `chunked_windowed_attention`, matching the
"small like 4 and 8, smaller better" guidance -- even a full image's trunk pass stays cheap.

**Stride-8 fuse-stage** (`(1, 8, None, 2, 0, 256, 4)`, inserted after trunk layer 1): pools every
8th byte (one pixel's RGB), code length 1536. Stride 8 is the finest granularity worth a dedicated
code-LM for -- smaller barely compresses anything. `code_d_model=256` (2x trunk width), only 2
layers, since it fires often (every 8 trunk positions) and needs to stay cheap too.

**Stride-64 fuse-stage** (`(3, 64, None, 4, 0, 512, 8)`, inserted after all 3 trunk layers): pools
every 64th byte (8x8, roughly a small patch's worth of raster bytes), code length 192. Updates 8x
less often than the stride-8 stage, so it can afford to be wider/deeper (`code_d_model=512`,
matching gpt2-tiny's own width exactly -- `gpt2_jax/train_gpt.py`'s `n_embd=512` -- 4 layers)
without dominating total compute -- the "code-LM wide/deep but updates less frequently" principle
directly traded against stride.

The two strides (8, 64) form a coarse-to-fine hierarchy analogous to Fractal's multi-level grid,
via pooling+cross-attention on a flat byte stream rather than literal 2D patchify -- same rationale
as `image256_fractal2level`/`3level`, re-derived at a size actually fast enough to iterate on now.
Confirmed via param count: trunk is 3.9% of total params, code-LMs 87.4% -- matches the design
intent directly, not just by construction (see the mixed-dim implementation section above).

## First real TPU training run (2026-08-28, tpu8)

Added `log.jsonl` logging to `train.py` (matches `summformer_jax/lm/train_summformer_v2.py`'s own
`log_dir/log.jsonl` convention exactly -- one JSON record per train step and per eval, printed and
flushed immediately), plus `--run-name`/`--save-dir` args so runs are keyed the same way as every
other lineage in this repo (`summformer_jax/image_gen/logs/<run_name>/log.jsonl`).

Launched against tpu8's ImageNet64 prep (`scripts/prep_imagenet64.py`'s output at
`/dev/shm/imagenet64` -- still growing in the background as more shards get written; `train.py`'s
`ImageByteLoader` reads whatever `.npy` shards exist at startup, so this trains on a growing prefix
of the full 1.28M-image set, not the complete thing yet):

```bash
# setup (once): sync code + one config to the node
scp summformer_jax/image_gen/{summformer.py,train.py,imagenet_dataloader.py} \
    muaz@<tpu8-ip>:~/qcute/summformer_jax/image_gen/
scp summformer_jax/image_gen/configs/image64_mixdim.py \
    muaz@<tpu8-ip>:~/qcute/summformer_jax/image_gen/configs/

# launch (tmux session `image_gen_mixdim`)
cd ~/qcute && .venv/bin/python3 summformer_jax/image_gen/train.py \
  --config summformer_jax/image_gen/configs/image64_mixdim.py \
  --resolution 64 --shard-dir /dev/shm/imagenet64 \
  --steps 2000 --steps-per-epoch 100 --eval-samples 1 \
  --run-name image64_mixdim_v1
```

Config: `image64_mixdim.py` -- trunk `d_model=128/n_heads=4/n_layers=3/window=8`, fuse-stage 1
`K=8`@`code_d_model=256`, fuse-stage 2 `K=64`@`code_d_model=512` (matches gpt2-tiny's width). Same
config verified end-to-end (train + eval + sample generation) at tiny scale just before this launch
-- see the training-script section above.

Logs land at `summformer_jax/image_gen/logs/image64_mixdim_v1/log.jsonl` on the node; sample PNGs
at `summformer_jax/image_gen/samples/image64_mixdim_v1/`. Both need pulling back to this repo's own
`logs/`/`samples/` dirs periodically (same `scp`-pull convention as every other TPU run in this
repo, see CLAUDE.md's monitoring routine) -- not yet automated for this lineage specifically.

## Where this left off

Decided to defer all of the above (seed-token block-local decode, ARMD-style scheduled decode) in
favor of a simpler, more directly comparable next step: plain pixel-RNN-style sequential NTP (no
fancy parallel decode), but using modern MTP (multi-token-prediction heads) and speculative
decoding for wall-clock speed instead of architectural parallelism. Goal: show summformer can model
long-range dependencies with a small, tunable-size state and reach competitive bits-per-dim
against FractalAR's reported numbers, using gpt2-tiny and ImageNet64 as reference points. Sized
guesses discussed: small downsampling `Ks`/attention windows (4 or 8, smaller preferred), large
MTP head counts (32-64). Not yet implemented as of this doc -- restate-to-check was requested and
interrupted before the restate happened; pick this up as the actual next task.

`fuse_stages` reformatted to grouped nested tuples
(`((src, dst), (stride, window), (code_n_layers, code_d_model=None, code_n_heads=None))`, parsed
by `_parse_fuse_stage`) instead of a flat 7-tuple -- `image64_mixdim.py` (built with `mtp_heads=24`,
per the plan above) and every other config updated to match, re-verified via the full
`check_kv_cache_consistency_fully_static` battery (9/9 pass, bit-exact vs. dense at fp32).

## JIT fix: `generate_kv_cache_fully_static`'s decode step wasn't actually jitted

`_make_fully_static_incremental_stepper`'s decode closure mutated captured Python state
(`nonlocal`/list-index reassignment on `code_lm_caches`, `h_code_out_bufs`, etc.) on every call --
unsafe under `jax.jit`/`nnx.jit` (a jitted closure that mutates captured variables freezes them at
first-trace tracer values; every later call silently reuses stale state), so the step function was
never actually wrapped in `jit` at all and re-traced/executed eagerly per call. Measured on tpu8
(`image64_mixdim.py`, `bench_generation_speed.py`): **6915ms/token**, extrapolating to ~23.6h for
one full 64x64 image -- would have blocked training's first eval-time sample generation for a day.

Fix: rewrote the whole stepper as pure functions -- `_init_decode_state` (zero-init `DecodeState`
dict pytree, fixed key structure per config), `_pool_and_fuse_pure`/`_embed_and_hist_pure`
(read state, return updated fields instead of mutating), `_prime_pure`/`_decode_step_pure` (thread
state explicitly, return `(logits, new_state)`), with `_decode_step_pure` wrapped in a
module-level `nnx.jit()` (`_jitted_decode_step_pure`) so one compiled trace is reused across all
decode steps. `generate_kv_cache_fully_static` now threads `state` through
`_init_decode_state` -> `_prime_pure` -> a loop over `_jitted_decode_step_pure`. Re-verified
correctness (9/9 consistency battery, bit-exact at fp32) before syncing to tpu8.

**Result**: 674ms/token (10x faster), but still ~137.9 min (2.3h) per full 12288-token image --
real but insufficient. `--eval-samples` stays disabled (0) for the next training launch; the
Python-level `for` loop over 12k decode steps (plus per-fuse-stage-boundary retracing inside
`lax.cond` branches) is still the bottleneck, not closure-mutation -- an outer `lax.scan`/
`fori_loop` over the whole generation loop would be the real fix, not attempted yet.
