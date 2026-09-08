# image_lagcodec -- session notes

## Architecture (decoder_type)

- `cross_attn` / `self_attn` -- original hard-fork designs, code prepended/interleaved as a
  token in the self-attention sequence. **Divergent from the reference qcute_lagcodec** --
  kept for reference/comparison, not recommended.
- `stack_track0` -- **1:1 port of qcute_lagcodec_decoder.py's StackDecoder track0 mechanism.**
  Two-pass: `Track0Layer.forward_pass1` (causal self-attn over real bytes, own-block code
  cross-attn spliced per layer, saves K/V) + `seed_pass2` (per-block seed as pure query against
  saved K/V, own code cross-attn). Self/cross-attn **share** Q/K/V weights (not separate
  modules). Track0 only (level0 conditioned on level1's code) -- track1+ (level2, ...) not
  implemented. **This is the one that actually works** -- use this for new runs.
- Codes above level0 are never decoded by a separate stage -- `HierEncoder`'s own per-level NTP
  heads produce them (matches reference: `enc.quant.sample_next(...)`).
- `self_attn_local` -- `StageSelfAttnDecoder(local=True)`: block-diagonal self-attn, zero
  cross-block visibility (StackDecoderLocal-inspired). Own-code-only capacity-starved alone
  (~7% TF acc plateau on shallow2).
- `self_attn_local_track1` -- new single decoder (`StageLocalTrack1Decoder`): level0 bytes,
  block-diagonal track0 (own code) + cross-attn track1 (level2's code). No per-level cascade.
  Fixes the capacity plateau (69% TF acc, still underfit at epoch 3000).
- `self_attn_lag` -- new `StageLagDecoder`, generalizes lag=0 (own-code-only) to a `lag` knob:
  `-1` = pure causal byte NTP (hardest, no code shortcut); `0` = own code only; `k>=1` = `k+1`
  codes grouped/prepended, block-diagonal across groups; `max` = one group, whole sequence.
  Same weights work for any lag (runtime arg, not baked into model). Pads internally when
  `lag+1` doesn't divide `n_blocks` (e.g. lag=4 on SEQ_LEN=3072, which has no factor of 5).

## Generation speed

- `stack_track0`: use `reconstruct_kv_cache`, NOT `reconstruct` (full recompute, O(n_blocks²),
  ~6h/eval at real scale -- diagnostic reference only). Verified byte-identical to the
  full-recompute reference at toy scale.
- `self_attn_local_track1`: `reconstruct(..., cache_track1=True)` caches track1's K/V at batch
  size `B`, not `B*n_blocks` -- the original design replicated identical per-image K/V once per
  block, causing a real production stall (9.66GB captured constants at qual_gen_n=8/n_blocks=1536).
  Fixed; verified `allclose` (not just argmax match) against the pre-fix computation.
- Greedy argmax across long autoregressive chains (~1500+ blocks) is numerically brittle: tiny
  float32 reduction-order differences between different batch shapes can flip a near-tied argmax
  and cascade into a different (but individually valid) trajectory. Not a logic bug -- confirmed
  via toy-scale sweeps (0 mismatches up to 96 blocks) and a direct `allclose` check on pre-argmax
  logits (max diff ~1.5e-8).

## Key findings

- **Bug found and fixed**: `reconstruct_group`'s KV-cache stepping (self_attn/cross_attn) was
  missing the block's own last-byte cache-population step (K steps instead of K+1) -- silently
  shifted every later block's position by one. Fixed in both decoder types, verified byte-
  identical to `forward()` under teacher forcing (~1e-7 logit diff).
- **Root cause of bad reconstruction (divergent decoders)**: NOT the cache bug, NOT undertraining.
  `recon_ncode` grouping decodes blocks in isolated islands (fresh KV cache per group) -- every
  group boundary is a cold start with zero cross-group visibility, unlike training's single
  continuous causal pass. Confirmed via `first_divergence_byte` lining up exactly with group
  boundaries every time.
  - Secondary, compounding cause: level0's first block has zero preceding context in *any* mode
    (grouped or sequential) -- `code_vocab^pq_chunks=4096` capacity causes real collisions
    (only 75/100 unique codes across 100 training images' block-0), so ~25% of images can't be
    disambiguated at that one position, and the error compounds through the rest of the sequence.
  - `recon_ncode` (2/4/8/16) gave ~1.5-3% free-running byte acc regardless of value -- fewer but
    bigger isolated islands, no net improvement. Fully sequential (no grouping) hit 13-16% with
    the divergent architecture -- clearly better, still far from good.
- **The real fix was architecture, not decode strategy.** `stack_track0` (faithful port) hit
  **75.4% free-running byte acc on first 64 blocks, 6/8 images reconstructed 100% exactly**
  (checkpoint: 99.90% teacher-forced acc, `cifar_lagcodec_overfit1000_stackdecoder_track0`,
  epoch 3000). The 2 failing images are exactly the block-0-collision cases -- an
  information-theoretic limit given current `code_vocab`, not a remaining bug.
- Two independent generation implementations (KV-cached `reconstruct_group`/`reconstruct_sequential`
  vs from-scratch `reconstruct_full_recompute`, zero cache) verified byte-identical on both toy
  and real trained weights -- ruled out any lingering cache-correctness doubt for the older
  decoder types.

## KIV / not implemented

- `stack_track0`: incremental KV-cache generation now implemented (`reconstruct_kv_cache`,
  verified byte-identical to the full-recompute reference at toy scale) -- earlier attempt
  (`encode_like_step`/`seed_step`) diverged and was removed; this is a fresh, correct rewrite.
- Track1+ for `stack_track0` specifically (level2's code) not ported -- but `self_attn_lag`/
  `self_attn_local_track1` now cover multi-code conditioning via different mechanisms.
- `own_code_min_lag=1` for `stack_track0`'s track0 -- superseded by `self_attn_lag`'s general
  `lag` knob (different decoder, same underlying idea).
- `recon_ncode` grouping kept as a documented fast-but-lossy option for `self_attn`/`cross_attn`,
  not the primary path anymore.
- `self_attn_lag` generation only wired into `run_reconstruct` via `reconstruct_kv_cache`
  (no full-parallel/group-size sweep option exposed yet, unlike `self_attn_local_track1`).

## Current runs (2026-09-08)

- `stack_track0` (tpu1), `self_attn_lag` lag=4 (tpu2), lag=max=1535 (tpu3) -- all launched with
  the KV-cache generation fix, 3000 epochs, must produce PNG samples at eval checkpoints.
